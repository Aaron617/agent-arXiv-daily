# Agent arXiv Daily

**Last Updated:** 2026-11-09 06:11:37

**Total Papers:** 65

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
<summary><strong>Ontology-Driven Control of Conversational Agents with a Use-Case for Job Interview Practice</strong> - Barbara Gendron, Gaël Guibon, Mathieu D'aquin - [[pdf]](https://arxiv.org/pdf/2611.09282)</summary>

**Abstract:** Large Language Models (LLMs) have been widely used as conversational agents in both domain-specific and open-domain situations. In domain-specific context, it is often necessary to adapt those LLMs to obtain conversations with desired characteristics, whether those apply at the level of each response (e.g. for tone or style) or more globally (i.e. at the level of a conversational strategy). Such adaptations are, however, generally realized in an ad hoc way, through specific, non-reusable, and non-transparent developments. In this paper, we propose an ontology-based approach to design conversational strategies that rely on the desired and measured characteristics of textual utterances in the conversation. Through a use case related to job interview practice, we show how this approach can be used to control LLMs to follow the strategy defined in the ontology. To this end, we automatically fine-tune the LLM to improve its compliance with ontologically-defined conversation characteristics. Our results show that training the LLM from formally defined concepts effectively enhances alignment with the requested characteristics. Hence, this work demonstrates how ontology-guided conversation control can be achieved while taking into account multiple, concurrent conversation aspects that are formally and transparently defined.

**arXiv ID:** 2611.09282
</details>

<details>
<summary><strong>More with Less: Efficient Parallel Scientific Code Translation via an Agent-Skill Workflow</strong> - Jianuo Shen, Pengyi Xu, Bo Zhang - [[pdf]](https://arxiv.org/pdf/2611.08947)</summary>

**Abstract:** Translating parallel scientific codes between HPC programming frameworks is a critical but labor-intensive task. Existing LLM-based approaches rely on fixed sequential pipelines with undifferentiated repair strategies and correctness criteria that do not match scientific computing practice. We present an agent-skill framework that separates translation, compilation diagnosis, and execution validation into three autonomous agents coordinated by a verdict-driven orchestrator, routing each failure to a targeted repair skill rather than retrying blindly. An automated output check drives the self-correction loop, complemented by an offline LLM-based verification that exposes false positives escaping compiler and runtime checks. We evaluate the framework on 30 HeCBench kernels translated bidirectionally among CUDA, OpenMP, and SYCL with three open-source LLMs. Our framework attains the higher verified correctness rate in all 18 model and direction combinations over a state-of-the-art baseline while consuming up to 52.4% fewer tokens, remaining consistent across models and directions without architectural modification.

**arXiv ID:** 2611.08947
</details>

<details>
<summary><strong>Coachable Agents for Interactive Gameplay</strong> - Roberto Capobianco, Harm van Seijen, Nolan D. Bard, Neil Burch, Fatima Davelouis, Josh Davidson, Alisa Devlic, Yunshu Du, Ishan Durugkar, Siddhant Gangapurwala, Daniel Hernandez, G. Zacharias Holland, Sahil Jain, Kenta Kawamoto, Raksha Kumaraswamy, Patrick MacAlpine, Dustin R. Morrill, Declan Oller, Francesco Riccio, Akanksha Saran, Craig Sherstan, Kaushik Subramanian, Thomas J. Walsh, Samuel Barrett, Kizza N. Frisbee, Mady Govil, Johannes Günther, Varun R. Kompella, James A. MacGlashan, Maxwell Svetlik, Michael D. Thomure, Jaden B. Travnik, Kevin Waugh, Elahe Aghapour, Florian Fuchs, Andreanne Lemay, Shruti Mishra, Takuma Seno, Peter Stone, Michael Spranger, Peter R. Wurman - [[pdf]](https://arxiv.org/pdf/2607.00642)</summary>

**Abstract:** Reinforcement learning has proven to be a valuable tool in the creation of advanced AI and robotic systems, contributing to everything from game playing (Silver et al., 2016; 2017; Wurman et al., 2022) to robotics (Levine et al., 2016; Andrychowicz et al., 2020; Luo et al., 2025; Dürr et al., 2026) to foundation models (Ouyang et al., 2022; Noukhovitch et al., 2024; Sun et al., 2024; Guo et al., 2025). Through trial and error, these AI systems typically learn a single, near-optimal behavior to solve their tasks. However, there are many use cases in which one would like to assert some level of control, after training is completed, over how the task is solved. We refer to these modifications of a core task as styles. We combine universal value function approximators (UVFAs) with carefully selected training scenarios, learning algorithms, and data augmentation to create a framework for coaching agents that exhibit styles in complex domains. We demonstrate the framework's application in the AAA video games Horizon Forbidden West and Gran Turismo, and in an open-source humanoid test domain. Despite the different nature of the domains--car racing, stylized game combat, and humanoid walking--each agent shows strong coherence to the style requests while still satisfying the main task in its domain. Importantly, the techniques outlined in this paper allow an end user to choose the final behavior at run time, giving them flexible control over the final executed performance.

**arXiv ID:** 2607.00642
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (13 papers)</h2></summary>

<details>
<summary><strong>Flex-RSI: Benchmarking Recursive Self-Improvement of Frontier Agents</strong> - Jingdong Zhang, Jionghao Wang, Zhengming Yu, Yancheng Cai, Wei Chow, Chris Choy, Zhiyang Dou, Hang Hua, Jiachen Li, Zhixuan Liu, Chen Wang, Songlin Wei, Siyuan Yang, Yu Yuan, Xiaohang Zhan, Yucheng Zhang, Shuhong Zheng, Xin Li, Lingjie Liu, Wojciech Matusik, Zhengzhong Tu, Wenping Wang - [[pdf]](https://arxiv.org/pdf/2611.08946)</summary>

**Abstract:** Frontier language models are increasingly deployed as agents, and how well an agent performs depends not only on its model but also on its harness: the prompts, tools, memory, and control loop through which the model acts. When an agent is applied to a new problem, a natural question is whether it can use the feedback of the task to improve the way it works, and not merely its answer. Most existing benchmarks keep the agent's harness fixed and score the solution it produces, or target large-scale model training. We introduce Flex-RSI, a benchmark for the recursive self-improvement of agents on vision and robotics tasks under a fixed budget. In each round, an executor improves the task solution, which is kept only if it scores higher on held-out validation data, and an improver then revises the harness, including the executor and itself. Every run starts from the same general-purpose harness, and the final solution is scored on hidden test data. Evaluating closed-source and open-weight models on tasks spanning perception, reasoning, robot action, 3D reconstruction, visual creation, and infrastructure, we find that self-improvement helps most when the agent builds its own solution, that feedback recording only success or failure holds back open-weight models, and that open-weight models fall furthest behind where outputs must pass strict checks of physical stability or exact correctness. Paired runs further show that editing the harness itself contributes to the gains.

**arXiv ID:** 2611.08946
</details>

<details>
<summary><strong>ABHBench: Evaluating Moral Decision-Making of Foundation Models from an Agentic Perspective</strong> - Chaoran Li, Yukun Li, Zeyuan Zhao, Shao Zhang, Xihuai Wang, Ying Wen - [[pdf]](https://arxiv.org/pdf/2611.09326)</summary>

**Abstract:** Foundation models are increasingly deployed as agentic systems that perceive context and act autonomously, raising the need to evaluate their moral behavior as active decision-makers. Existing benchmarks, however, largely ask models to judge human actions from third-person textual vignettes, leaving untested how models behave when they themselves become the acting agent. We introduce ABHBench, a multimodal benchmark built from Detroit: Become Human, an interactive narrative game, to evaluate how foundation models make moral decisions in first-person human-android dilemmas. ABHBench places foundation models in branching narrative scenarios with visual observations, dialogue, memory, and action choices. Inspired by Asimov's Three Laws of Robotics, our framework measures Harmlessness, Obedience, and Self-preservation to capture the core tensions in human-AI relations, along with Autonomy, Stability, and Certainty to assess decision-making patterns. Across 12 foundation models, we find that moral behavior changes substantially under situated agentic context: 75% of models shift significantly between a text-only abstraction and the matched multimodal DBH scenario, and reasoning-enhanced models show no monotonic gains in consistency or human-oriented behavior. These results highlight the necessity of evaluating moral behavior in situated agentic contexts rather than relying on third-person judgment alone.

**arXiv ID:** 2611.09326
</details>

<details>
<summary><strong>Typed-Centrality Grounding for Agentic GraphRAG: A Symbolic Approach for Agentic Reasoning</strong> - Rosario Napoli, Antonio Celesti, Massimo Villari, Maria Fazio - [[pdf]](https://arxiv.org/pdf/2611.09414)</summary>

**Abstract:** Knowledge Graphs (KGs) and Large Language Models (LLMs) represent two approaches for knowledge representation and reasoning over heterogeneous data. Recent research has explored their integration through Graph Retrieval-Augmented Generation (GraphRAG) systems, which use KGs to provide structured context to LLMs. However, existing GraphRAG approaches treat KGs as homogeneous graphs and rely on textual similarity between the input prompt and the context, rather than using it as a source of deterministic knowledge.
This work proposes an agent-based GraphRAG system that introduces a symbolic layer to integrate deterministic reasoning alongside text similarities. This approach is based on the definition of typed centralities for KGs, where centrality measures are computed separately for each specific KG's class. This information is used to construct a symbolic representation that is provided to the LLM as a deterministic source. The proposed system was mathematically validated and evaluated on a real-world POLE (Person, Object, Location, Event) dataset, comparing reasoning and information retrieval with and without the symbolic layer. The results indicate that the our approach leads to responses that are more consistent with the context, improving the generated outputs.

**arXiv ID:** 2611.09414
</details>

<details>
<summary><strong>StockASkill: A Local-First AgentSkill for Intelligent Stock Selection</strong> - Xiangjing An, Zhibin Jiao, Aiting Chen - [[pdf]](https://arxiv.org/pdf/2611.09528)</summary>

**Abstract:** Quantitative equity selection and portfolio construction in emerging and cross-border markets frequently suffer from computational bottlenecks, remote API fragility, survivorship bias, and rigid execution constraints such as discrete board lots and investor-mandated drawdown limits. In this paper, we present StockASkill, a local-first, full-data-driven AgentSkill engineered as a specialized Quant Co-Processor for autonomous AI agents and quantitative researchers. Tailored for agile portfolios (10^5 ~ 10^7 RMB) where discrete lot execution and capital protection are mandatory, StockASkill embeds a columnar DuckDB OLAP engine over 42.44 million historical price records and 186,432 financial disclosures spanning 26 years across China A-shares, Hong Kong, US equities, and ETFs. By eliminating ad-hoc network dependencies during analysis, StockASkill reduces cross-sectional screening latency across 5,400+ equities from over 25 seconds to sub-0.5 seconds (a 95%+ reduction). We formalize an end-to-end mathematical pipeline integrating a 7+1 factor alpha model, an SLSQP convex optimizer under smooth CVaR drawdown surrogates with exact path projection, dynamic 100-share board lot integer quantization, and an autonomous thesis lifecycle finite-state machine. Furthermore, we provide a zero-disk, strongly-typed JSON streaming interface designed for high-throughput LLM tool-calling. Open-sourced at this https URL, comprehensive empirical benchmarks from 2018 to 2024 demonstrate that the architecture achieves annualized Sharpe ratios exceeding 1.48 and Calmar ratios up to 2.17, while reducing max drawdown by up to 85% relative to benchmark indices.

**arXiv ID:** 2611.09528
</details>

<details>
<summary><strong>Knowing When to Intervene: Causal Evaluation of AI Agents</strong> - Ayush Sawarni, Wenqian Xing, Tijana Zrnic, Ramesh Johari, Vasilis Syrgkanis - [[pdf]](https://arxiv.org/pdf/2611.09792)</summary>

**Abstract:** Current evaluations of agentic AI systems primarily rely on benchmarks. However, the static nature of benchmarks makes them uninformative about the value of auditing interventions at deployment time. For reliable deployment, we often need to determine within an agent's trajectory whether or not to intervene, e.g., by verifying the agent's current output or steering it toward different behavior, using either human review or more costly models or time-consuming agentic supervision. The value of such interventions is fundamentally a causal inference question, asking whether the causal effect of such an intervention is beneficial for the final outcome. Estimating this causal effect is particularly challenging due to the dynamic nature of the problem: the impact of an intervention depends on the current context and environment, as well as the agent's future behavior. We develop the first methodologically grounded framework for evaluating the causal effects of randomized interventions within agent trajectories. By leveraging a relatively small dataset of agent trajectories collected under randomized interventions such as verification or screening, we can estimate the dynamic causal effect of a rich class of (contextual) intervention policies. The structural parameters that we estimate can also be used to construct safety leaderboards that capture the risk of various agentic actions, which can be used by practitioners for audit prioritization. Applying our methods to tau-bench Retail, ToolSandbox, and ImpossibleBench, we find that intervention effects vary substantially across agent actions and states. More broadly, our work suggests a rich research agenda at the intersection of causal inference and agentic evaluation, which we believe is essential for the reliable and safe deployment of AI agents in the real world.

**arXiv ID:** 2611.09792
</details>

<details>
<summary><strong>DRAFT-Bench: Evaluating Frontier Agents Across the Professional Design Lifecycle</strong> - Rong Pan, Jianqiang Wang - [[pdf]](https://arxiv.org/pdf/2611.08933)</summary>

**Abstract:** Can frontier general-purpose agents understand client intent, explore creative directions, execute revisions, and prepare production deliverables? We introduce DRAFT-Bench (Design Reliability Across Feedback and Tasks), a lifecycle-oriented evaluation of requirement elicitation, creative exploration, constrained editing, iterative revision, and production handoff. Its Requirement-to-Artifact framework links evolving requirements, protected properties, artifact versions, and delivery obligations. We compare native Codex/Astra and Claude Code across 35 design briefs under 43 conditions, alongside a separate 40-scenario RFQ track. Astra completes 43/43 design conditions and Claude 35/43. All scored core runs receive full structural credit, yet on the 28 core conditions scored for both systems, semantic checks flag violations in 11 for Astra and 5 for Claude; these are model judgments, not expert-confirmed failure rates. RFQ coverage is high, but success ordering changes across cohorts. SKU expansion, revision dependencies, and White ink files expose ambiguity among unmet obligations, missing inputs, and evaluator interpretations. These findings separate execution, requirement compliance, and professional acceptance. Unpaired tracks and different tools preclude claims of RFQ-to-delivery success, factory acceptance, or a pure model ranking.

**arXiv ID:** 2611.08933
</details>

<details>
<summary><strong>SmartBloch-Agent: An Agentic Framework for Automated MRI Sequence Design and Optimization</strong> - Jiakai Yang, Haoye Zheng, Yixiao Yue, Qinqin Yang, Zhong Chen, Shuhui Cai, Congbo Cai - [[pdf]](https://arxiv.org/pdf/2611.09413)</summary>

**Abstract:** Magnetic resonance imaging (MRI) pulse sequences organize radiofrequency excitation, gradient encoding, and signal acquisition, determining magnetization evolution and image formation. Sequence research requires translating imaging objectives into executable protocols meeting physical and acquisition constraints, refined via physical simulation, signal reconstruction, and scientific evaluation. Sequence construction, physical verification, parameter search, and assessment involve many tools and manual decisions, limiting automation from objective to verifiable sequence. We develop SmartBloch-Agent, an integrated framework for automated MRI sequence design and optimization. Given objectives and scientific constraints in natural language, an AI agent interprets goals, plans tasks, orchestrates tools, and makes feedback-informed decisions, drawing on a construction module and optimizers for structural design and parameter search. Our GPU-accelerated SmartBloch engine supplies physical computation for both stages, so the framework configures, per sequence and objective, event validation, Full-Bloch magnetization evolution, signal acquisition, image reconstruction, and scientific evaluation, and closes the loop without external MRI simulators. Results return to the agent to guide sequence revision, parameter optimization, and candidate selection, and the design files carry simulation, reconstruction, and evaluation evidence, forming an output package for a real-scanner workflow. Experiments show the framework automatically designs and optimizes MRI sequences for varied objectives and constraints, refines candidates with physical feedback, and improves several parameter-estimation metrics and downstream quantitative reconstruction. SmartBloch-Agent unifies objectives, design, physical execution, and evaluation into a continuous workflow for end-to-end automation of MRI sequence research.

**arXiv ID:** 2611.09413
</details>

<details>
<summary><strong>How Much Conversation History Does an AI Agent Need?</strong> - Danny Kim, Fernando Koch - [[pdf]](https://arxiv.org/pdf/2611.09662)</summary>

**Abstract:** How does using less conversational history change reader-token use and benchmark scores? We examine this question on 1,100 BEAM questions. We held the retrieval ranking and reader constant across five fixed passage budgets, and mean assigned scores rose as the passage budget grew. The score gain per additional token became smaller at the upper tested budgets. Answerable questions generally received higher scores with more context. Pooled abstention scores sometimes rose as the passage budget shrank. We also tested message retention in a controlled task with explicit conversational-state targets and compared retained user messages with token-matched recency. Qwen matched the full state more often for direct requirements when user messages were retained. Its full-state matching declined when answers depended on references to earlier offers. Mistral produced near-floor results across the retention conditions. A scoring audit leaves the correctness of BEAM answers unresolved. These experiments characterize the token and score consequences of the tested memory budgets. Reliable budget selection depends on retaining the evidence each question requires and validating the measure used to assess answer quality.

**arXiv ID:** 2611.09662
</details>

<details>
<summary><strong>When Instruction Files Go Stale: Documentation Debt and Hidden Cost in Coding Agents</strong> - Tural Mehtiyev, Biruk Tadesse, Lawrence Arkoh, Wesley Assunção - [[pdf]](https://arxiv.org/pdf/2611.09802)</summary>

**Abstract:** LLM-based coding agents such as Claude Code are guided by repository instruction files (also called context files or agent configuration files) like this http URL and this http URL, which are read in every agent session. As any documentation artifact, these files can fall behind the code and become stale (e.g., a test folder moves, but the file still points to the old one). In technical debt terms, such a stale instruction is documentation debt, and its interest is the cost it adds to every later agent session. Evaluations that check only task success can miss this cost, and the stale file can persist. In this study, we ask what leads agents to act on a stale instruction, how it affects their behavior and at what cost, and how they recover. We study three real-world stale instructions from active open-source repositories in depth, each a different kind of guidance: a test command, a code location, and a procedure for adding a component. Agents fix 22 maintenance tasks under three versions of each instruction (left stale, corrected to the current path, and removed), with the code and the task held fixed, and we analyze 732 trajectories of two models at the level of individual tool calls. We find that agents follow stale guidance when it is relevant to their tasks, sometimes even when correct information is available elsewhere. On the stale test command, Claude Sonnet 5.5 needed about two extra tool calls per run, a third more than with the corrected command; in the other cases, we could not detect a difference. Although agents generally recovered from the immediate failures, none repaired the underlying instruction file. Correcting the instructions eliminated operations on obsolete paths, while removing them prevented most. Our findings show that task success alone can hide the costs of documentation debt, highlighting the need to evaluate agent behavior beyond final outcomes.

**arXiv ID:** 2611.09802
</details>

<details>
<summary><strong>On Agentic Manipulation Policies</strong> - Chen Chen, Omid Bagherieh, Naveen Nikamanth Appiah Balaji, Jesse T. Gonzalez, Negin Heravi, Fahad Islam, Xueting Li, Jean-Baptiste Regli, Kejia Ren, Petr Švec, David J. Yoon, Matthias Müller, Vladlen Koltun - [[pdf]](https://arxiv.org/pdf/2611.09892)</summary>

**Abstract:** Recent work on robot manipulation has focused on specialist policies trained on demonstrations. We show that generalist AI models can control manipulators more reliably than specialist policies such as vision-language-action models (VLAs) and world-action models (WAMs). We systematically analyze agentic manipulation policies (AMPs), in which a generalist AI model commands the manipulator directly, adapting its actions as the task unfolds. We present AMP1, a reference harness that runs unchanged across single-arm, bimanual, and mobile manipulation. AMP1 achieves state-of-the-art results on the RoboLab-120 and CaP-Gym benchmarks, with no demonstrations or task-specific training. On RoboLab-120, AMP1 reaches an 85.6\% success rate zero-shot, compared to 49.2\% and 42.9\% for leading VLA and WAM policies. The agent succeeds zero-shot in 104 of 108 real-world trials spanning twelve tasks and three robot instances, reducing failure rates by 20x and 15x relative to strong VLA and WAM baselines. The basic harness supplies no grasping, pushing, or placing skills; the agent develops its own strategies, such as searching for hidden objects, using objects as tools, and calibrating its cameras. With the addition of a persistent memory store, the agent can also learn from experience: an augmented harness with a memory representation that is carried across episodes lowers the failure rate on RoboDojo bimanual manipulation tasks by 4.6x relative to zero-shot deployment. Our analysis suggests that AMPs are substantially more general and robust than specialist manipulation policies, providing a promising foundation for robot manipulation.

**arXiv ID:** 2611.09892
</details>

<details>
<summary><strong>OMG-Agent: Toward Robust Missing Modality Generation with Decoupled Coarse-to-Fine Agentic Workflows</strong> - Ruiting Dai, Zheyu Wang, Haoyu Yang, Yihan Liu, Chengzhi Wang, Zekun Zhang, Zishan Huang, Jiaman Cen, Lisi Mo - [[pdf]](https://arxiv.org/pdf/2602.04144)</summary>

**Abstract:** Data incompleteness severely impedes the reliability of multimodal systems. Existing reconstruction methods face distinct bottlenecks: conventional parametric/generative models are prone to hallucinations due to over-reliance on internal memory, while retrieval-augmented frameworks struggle with retrieval rigidity. Critically, these end-to-end architectures are fundamentally constrained by Semantic-Detail Entanglement -- a structural conflict between logical reasoning and signal synthesis that compromises fidelity. In this paper, we present \textbf{\underline{O}}mni-\textbf{\underline{M}}odality \textbf{\underline{G}}eneration Agent (\textbf{OMG-Agent}), a novel framework that shifts the paradigm from static mapping to a dynamic coarse-to-fine Agentic Workflow. By mimicking a \textit{deliberate-then-act} cognitive process, OMG-Agent explicitly decouples the task into three synergistic stages: (1) an MLLM-driven Semantic Planner that resolves input ambiguity via Progressive Contextual Reasoning, creating a deterministic structured semantic plan; (2) a non-parametric Evidence Retriever that grounds abstract semantics in external knowledge; and (3) a Retrieval-Injected Executor that utilizes retrieved evidence as flexible feature prompts to overcome rigidity and synthesize high-fidelity details. Extensive experiments on multiple benchmarks demonstrate that OMG-Agent consistently surpasses state-of-the-art methods, maintaining robustness under extreme missingness, e.g., a $2.6$-point gain on CMU-MOSI at $70$\% missing rates.

**arXiv ID:** 2602.04144
</details>

<details>
<summary><strong>Agentic AI Scientists Are Not Built For Autonomous Scientific Discovery</strong> - Harshit Bisht, Vinay Kumar, Kevin Maik Jablonka, Mausam, N. M. Anoop Krishnan - [[pdf]](https://arxiv.org/pdf/2605.08956)</summary>

**Abstract:** A growing body of work pursues AI scientists capable of end-to-end autonomous scientific discovery. This position paper argues that although they already function as co-scientists, agentic AI scientists are not built for autonomous scientific discovery. We identify the following challenges in building and deploying autonomous AI scientists: (1) Problem selection is influenced by the McNamara fallacy; (2) Agents are built on large language models (LLMs) whose training corpora omit tacit procedural and failure knowledge of laboratory practice; (3) Preference optimisation during post-training compresses output diversity toward consensus; and (4) Most scientific benchmarks measure single-turn prediction accuracy and lack feedback from physical experiments back to the computational model. These challenges are not just questions of scale and scaffolding; they require revisiting fundamental design choices. To build truly autonomous AI scientists, we recommend scientific simulations as verifiers for training, a persistent, mutable epistemic state that carries beliefs and shifting objectives across an investigation, the establishment of a centralized preregistration repository for all AI-generated hypotheses, and application driven by scientific need rather than tool affordance.

**arXiv ID:** 2605.08956
</details>

<details>
<summary><strong>Agentic AI uncovers conserved cross-tissue protein co-abundance programs inaccessible to single-dataset analysis</strong> - Dehao Wu, Runyu Guan, Qiqi Xie, Yang Li, Haohan Wang - [[pdf]](https://arxiv.org/pdf/2608.28990)</summary>

**Abstract:** Protein co-abundance clusters preserved across tissues can nominate shared disease mechanisms and therapeutic targets, particularly where proteins implicated in organ-confined diseases converge in other tissues. Yet cross-tissue proteomic studies have examined pre-selected tissue pairs, leaving most combinations, and non-obvious relationships among them, unexplored. Exhaustive comparison requires consistently assessing thousands of candidate clusters from hundreds of tissue-pair networks across heterogeneous evidence, a scale beyond manual curation. We present an LLM-agent framework for large-scale, evidence-grounded comparison of tissue-specific protein co-abundance networks. From sample-level proteomic data, it constructs tissue networks, derives conserved clusters for each tissue pair, integrates evidence from expression atlases, interaction and complex databases, pathways, disease associations and the literature, and proposes mechanistic hypotheses in which every conclusion cites its evidence. Applied to all 231 pairs among 22 human tissues and sample groups, it identified 1,825 conserved clusters in 216 tissue pairs. Co-abundance relationships were shared beyond chance in 76% of pairs, including unrelated organs, and shared clusters were only weakly associated with abundance similarity. Clusters were enriched 3.7- to 6.9-fold for known interactions and complex memberships, and 379 of 793 testable clusters replicated in independent CPTAC proteomes. Some clusters pointed to relationships absent from current databases: one shared by liver and bone marrow linked three genes causing inherited white matter disorders, suggesting a common mechanism via the integrated stress response. Conserved co-abundance clusters thus offer a hypothesis-generating resource for studying disease proteins beyond the affected organ. Code: this https URL

**arXiv ID:** 2608.28990
</details>

</details>

<details open>
<summary><h2>LLM Agents (10 papers)</h2></summary>

<details>
<summary><strong>Decision Context: Persistent Behavioral Learning for AI Agents</strong> - Nima Vahdat, Pashootan Vaezipoor - [[pdf]](https://arxiv.org/pdf/2611.08644)</summary>

**Abstract:** LLM agents do not learn from corrections across sessions. A user fixes the same mistake at the start of every conversation. Existing agent memory systems primarily handle facts, such as preferences, entities, and conversational history, rather than procedural knowledge.
We present Forge Decision Context (forge_dc), a system that captures behavioral signals from human corrections and self-assessed tool-call failures, compiles them into imperative RULE directives, and injects them at the decision points where they apply. We evaluate forge_dc alongside four predecessor approaches on DC-Bench, a new benchmark of 71 scenarios across nine behavioral dimensions, built from real executed agent traces in a deterministic mocked-tool environment. forge_dc reaches 95% compliance, up from 55% with naive retrieval, while injecting approximately 10 times fewer tokens per query.
Across the five-stage progression, the strongest systems represent behavioral memory as explicit directives and deliver it at the decision points where it applies. The results identify representation, activation, and placement as central design choices for persistent behavioral learning.

**arXiv ID:** 2611.08644
</details>

<details>
<summary><strong>Nudging Decision Models: Do Digital Nudges Steer Fast, Generation-Free AI Agents?</strong> - Sascha Kaltenpoth, Haya Halimeh, Kevin Bösch, Oliver Müller - [[pdf]](https://arxiv.org/pdf/2611.09264)</summary>

**Abstract:** LLM-based agents increasingly automate business and consumer tasks but are slow and expensive. Decision models, a new class that chooses among options without generating text, are far faster and less costly. Developers call them "System One models," referencing dual process theory. We ask whether such models can be steered by digital nudges online and whether, as the theory predicts, automatic nudges affect them more than reflective ones. We adapted an online shopping experiment, previously run with humans and LLM agents, to three decision models (Jev, Decision-1, d1). Across 600 runs per model, they chose a preselected default in 97.5-99.9% and a product with social influence information in 75-96% of choices, versus 47-49% in the control. All responded more strongly to the automatic nudge, the opposite of what has been reported for reasoning-capable LLM agents. We provide first evidence that non-generative decision models are susceptible to digital nudges in a System 1 pattern, implying agent-type-specific nudge effects.

**arXiv ID:** 2611.09264
</details>

<details>
<summary><strong>When Routine Operations Break Architectural Pattern Relationships in LLM Agent Harnesses</strong> - Boming Xia, Qinghua Lu, Dino Sejdinovic, Liming Zhu - [[pdf]](https://arxiv.org/pdf/2611.09088)</summary>

**Abstract:** LLM agent harnesses distribute tool mediation, state management and safety guardrails across native services, independently written plugins and configuration. Routine operations can change the runtime order or retained state that architectural pattern relationships require, without any code change. We study five relationships associated with the Guardrail Enforcement Point, Autonomy Controller and Blackboard patterns across five released harnesses, analysing a main sample of fourteen independently written plugins. Using PatternOps, our assessment procedure, we trace requirements from pattern descriptions, provider documentation and deployment profiles to the providers that realise them and the records that decide them. We then compare alternative allocations under supported operations. Such operations broke relationships silently: after a plugin was reapplied or an allow rule saved, policy stopped evaluating edits while every plugin stayed active, and a prohibited edit executed. Which alternatives qualified depended on the requirement. Allocations that kept the required order, or kept or rebuilt the needed state, held on their tested paths and operation sequences, while others prevented prohibited effects but missed edits rejected elsewhere. In these harnesses, a pattern relationship is a property of the running system and its history, so architects should evaluate it against the decision paths, state lifetimes and operation sequences their deployments use.

**arXiv ID:** 2611.09088
</details>

<details>
<summary><strong>MemoryDojo: Benchmarking Poisoning Through LLM Agent Memory</strong> - David H. Yang, Blake Bullwinkel, Amanda Minnich, Mohammad Mohammadi Amiri, Giorgio Severi - [[pdf]](https://arxiv.org/pdf/2611.09721)</summary>

**Abstract:** LLM agents benefit from using memory systems to improve performance on long-horizon, multi-session tasks. However, memory introduces a critical attack surface. Malicious instructions encountered in one session may be stored and propagated through memory before being retrieved and acted upon in later sessions. Because memory systems differ in how they process and retrieve information, the downstream effects of poisoning can vary significantly across systems. To quantify the effects of poisoning across different memory architectures, we introduce MemoryDojo, a benchmark for tracing the lifecycle of poisoned information through agent memory and measuring its influence on subsequent agent behavior. MemoryDojo is easily extensible, consists of 155 multi-session long-horizon trajectories covering four domains, and supports five different memory systems. Through our evaluation, we find that all memory systems are susceptible to memory poisoning, with substantial variation in downstream effects. We further propose a memory-aware poison optimization process that increases the poisoning efficacy, highlighting differences between memory architectures. We hope that MemoryDojo will support research on understanding and mitigating agentic memory risks.

**arXiv ID:** 2611.09721
</details>

<details>
<summary><strong>Counterfactual Distillation: Internalizing Reflective Experience into LLM Agent</strong> - Rui Ge, Yichao Fu, Yu-Yang Qian, Junda Su, Yiming Zhao, Hao Zhang - [[pdf]](https://arxiv.org/pdf/2603.16843)</summary>

**Abstract:** Large language models increasingly act as autonomous agents in long-horizon interactions, learning from final outcomes and environmental feedback. However, optimizing from such interaction experience remains challenging, as feedback is heterogeneous and rarely carries explicit reward information. Existing post-training approaches rely on outcome-driven methods, such as reinforcement learning with verifiable rewards (RLVR), which primarily exploit final success signals while leaving interaction experience and feedback underutilized. In sparse-reward, long-horizon tasks, this often results in distribution sharpening: the policy reinforces a narrow set of already-successful behaviors, without substantially improving the feedback-grounded agency needed for broader problem-solving capacity (e.g., Pass@k). We propose Counterfactual Distillation, a framework that converts exploration-derived experience into trainable policy supervision. Specifically, we organize exploration as a tree-structured search process, where the agent reflects on past decision points and makes new experience-guided decisions to form alternative branches. These corrections are then distilled as counterfactual targets: we train the model to predict the revised action from the original history without explicitly providing the experience, thereby introducing out-of-distribution behavior patterns and expanding exploration capacity. Across diverse interactive coding and agentic tasks, our method outperforms outcome-driven baselines such as GRPO and experience-based methods such as Early Experience, with gains of up to 14 percentage points over the base model on Pass@128. When interleaved with reinforcement learning updates, it further raises the performance ceiling, improving Pass@1 on Sokoban by 8 percentage points over extended GRPO. Our code is available at this https URL.

**arXiv ID:** 2603.16843
</details>

<details>
<summary><strong>ClawTrace: Cost-Aware Tracing for LLM Agent Skill Distillation</strong> - Boqin Yuan, Yue Su, Renchu Song, Sen Yang, Jing Qin - [[pdf]](https://arxiv.org/pdf/2604.23853)</summary>

**Abstract:** Skill-distillation pipelines learn reusable rules from LLM agent trajectories, but they lack a key signal: how much each step costs. Without per-step cost, a pipeline cannot distinguish adding a missing step to fix a bug from removing an expensive step that never affected the outcome. We use the cost-attribution gap to ask whether the rule types inside a distilled skill transfer the same way to new tasks. ClawTrace records cost-attributed agent traces and compiles each session into a TraceCard; CostCraft reads TraceCards and writes three kinds of skill patches: preserve, prune, and repair. We find a pattern aggregate metrics hide. On 30 held-out SpreadsheetBench tasks across two seeds, removing prune patches roughly tripled the quality-regression count without lowering median cost. Across the full 84-task SkillsBench transfer, CostCraft saves no aggregate cost. All three quality regressions trace to the preserve lane, and both quality wins trace to the prune lane: prune patches act as quality guardrails while preserve patches drive regressions. We argue that reusable agent skills should be evaluated at the rule-type level, not as monolithic instruction packages. To support this, we release ClawTrace, the TraceCard schema, and the full set of typed skills.

**arXiv ID:** 2604.23853
</details>

<details>
<summary><strong>Self-Programmed Execution for Language-Model Agents</strong> - Luke J. O'Connor - [[pdf]](https://arxiv.org/pdf/2605.06898)</summary>

**Abstract:** At the heart of a language model agent is an orchestrator program which is responsible for the state transition between consecutive turns. In existing systems, this program is implemented in the harness. This paper introduces self-programmed execution (SPE), in which the model completion is itself the orchestrator program, and the harness evaluates this program but does not impose any fixed orchestration policy. Formally, an SPE state of an agentic machine is one from which a model completion can load any successor state up to an embedding, and an instance of an SPE state is one which executes the model completion as a program. In practice, the program requires a language with unusual features, as its source code doubles as the content of the model's context window. I therefore introduce Spell, a Lisp dialect in which programs can edit and re-evaluate themselves, and effectful expressions like model invocations are structured such that re-evaluating an edited program does not replay its side effects. Experiments with existing models, not trained for SPE or Spell, show that frontier models can learn Spell in-context and accomplish challenging agentic tasks. These results demonstrate how an LM can act as an agent without any fixed orchestration policy, and they raise the question of what self-orchestration strategies might be learned by a model trained for self-programmed execution. Code is available at this https URL.

**arXiv ID:** 2605.06898
</details>

<details>
<summary><strong>Agentic Abstention: Do Agents Know When to Stop Instead of Act?</strong> - Han Luo, Bingbing Wen, Lucy Lu Wang - [[pdf]](https://arxiv.org/pdf/2606.28733)</summary>

**Abstract:** LLM agents are expected to act over multiple turns, using search, browsing interfaces, and tools to complete user goals. Yet not every goal is well specified or achievable in an available environment. In such cases, a reliable agent should recognize that further interaction is unlikely to help and abstain from additional tool calls. We define Agentic Abstention, the problem of deciding when an agent should stop acting under uncertainty. Unlike standard LLM abstention, which is usually evaluated as a single-turn answer-or-abstain decision, agentic abstention is a sequential decision problem: an agent can answer, abstain, or gather more information at each turn, and the need to abstain may only become clear after interacting with the environment. We study this problem across web shopping, terminal environments, and question answering, evaluating 13 LLM-as-agent systems and 2 agent scaffolds on more than 28,000 task instructions. Our results show that the main challenge is not only whether agents can abstain, but also when they abstain. Some agents never abstain when they should, while others do so only after many unnecessary interactions. This gap is especially large on tasks where the instruction appears feasible until the environment reveals otherwise (e.g., no valid result matches the instruction). We further find that model scale, reasoning, and agent scaffolding affect abstention in different ways, where larger or more capable models sometimes perform worse at timely abstention. Finally, we introduce CONVOLVE, a context engineering method for improving agentic abstention that distills interaction trajectories into reusable stopping rules. On held-out WebShop tasks, CONVOLVE substantially improves timely abstention without updating model parameters, raising Llama-3.3-70B's timely recall rate from 26.7 to 57.4. Our dataset and code are available at this https URL.

**arXiv ID:** 2606.28733
</details>

<details>
<summary><strong>GABench: A Comprehensive Benchmark for Evaluating LLM Agents on Graph Analysis Tasks</strong> - Jiarui Tan, Zhongjian Zhang, YaBo Guo, Jiawei Liu, Yujie Xing, Muhan Zhang, Cheng Yang, Chuan Shi - [[pdf]](https://arxiv.org/pdf/2608.01684)</summary>

**Abstract:** Large language model (LLM) agents are increasingly capable of planning, using tools, and interacting with external environments. They are typically supported by harnesses, which manage state and coordinate multi-step execution. Graph analysis provides a promising setting for evaluating their agentic capabilities, because it requires agents to access data and execute operations in a graph environment. However, existing graph benchmarks for LLMs provide limited coverage of graph tasks and graph types, making it difficult to comprehensively evaluate LLM agents. Moreover, they typically formulate graph analysis as text-based question answering, where graph information is directly provided in the prompt, limiting the evaluation of end-to-end agentic capabilities. To address these limitations, we introduce GABench, a comprehensive benchmark for agentic graph analysis. GABench spans three graph types and covers four graph analysis task categories: graph retrieval, graph theory, graph machine learning, and graph open-ended question answering. GABench also provides 84 executable tools for accessing graph data and performing diverse graph operations. Building on these tools, we develop an agentic graph analysis task generation pipeline and construct 10,400 tasks with verifiable ground this http URL GABench, we evaluate a range of frontier LLMs and agent harnesses. Our experiments reveal three key findings: (1) Existing LLM agents still struggle with complex graph analysis tasks. (2) Harness choice significantly affects performance, yet existing harnesses remain limited on complex graph tasks. (3) Graph analysis depends more on tool-call quality than quantity. Our findings provide practical insights into the development and evaluation of LLM agents for graph analysis.

**arXiv ID:** 2608.01684
</details>

<details>
<summary><strong>What Are LLM Agents Actually Learning on Low-Budget Molecular Optimization Benchmarks?</strong> - Jiayu Qin, Zhengquan Luo, Changyou Chen, Zhiqiang Xu - [[pdf]](https://arxiv.org/pdf/2611.08522)</summary>

**Abstract:** Recent work reports large gains for LLM agents on low-budget molecular optimization. We measure the contributions of online optimization, retrievable task information, and permissive oracle landscapes. On PMO at 50 oracle calls, a GPT-4o mini agent's sum AUC falls from 14.44 to 4.85 after task names and curated reference compounds are hidden. An identity-blinded condition that states only the oracle form (e.g., similarity to a hidden reference) scores 5.39, while a deterministic retrieval-only baseline with no LLM reaches 12.16 (84% of the full agent). A full 23-task 2x2 ablation and a three-seed replication on eight tasks support this interpretation. On a protein-targeted benchmark with an AutoDock Vina composite oracle, a single-edit random walk is a strong lower bound; across three seeds it obtains composite 0.636 +/- 0.008 and docking improvement -1.11 +/- 0.04 kcal/mol. As a positive control, a retrieval-free population agent exceeds REINVENT at a matched 50-call AUC budget, with its strongest result at 100 calls. These results motivate two inexpensive sanity checks: retrieval-safe PMO prompts and a random-walk floor for docking-based optimization.

**arXiv ID:** 2611.08522
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (13 papers)</h2></summary>

<details>
<summary><strong>Debate to Deceive: Poisoning LLM's Fact-Checking via Game-Theoretic Multi-Agent Debate</strong> - Liting Huang, Huizhi Liang, Yi Lei, Xingjun Ma, Varun Ojha - [[pdf]](https://arxiv.org/pdf/2611.08756)</summary>

**Abstract:** Knowledge attacks on Large Language Models, such as manipulating fact-checking judgments, pose significant risks to AI reliability. Existing approaches -- adversarial suffix/template injection and single-agent self-reflection -- lack robustness across diverse defenses and are prone to confirmation bias and costly trial-and-error, respectively. We propose Debate-to-Deceive (D2D), a multi-agent debate framework grounded in human debate structure and rhetorical style, enabling agents to generate adversarial passages that are semantically coherent, rhetorically effective, and resistant to early confirmation bias. D2D formalizes the adversarial process as an evolutionary game in which agents adaptively evolve their rhetorical strategies (Logos, Pathos, Ethos) through competition. To manage the intractably large LLM output space, we decouple agent behavior into a strategic layer, which selects a rhetorical action from a compact strategy space via evolutionary-inspired learning dynamics, and a generation layer, where the LLM produces a concrete passage conditioned on that strategy and the debate context. On four fact-checking benchmarks (FEVER, HealthVer, CLIMATE-FEVER, and SciFact), D2D substantially outperforms existing attacks and evades most defenses.

**arXiv ID:** 2611.08756
</details>

<details>
<summary><strong>Memory-Aware Multi-Agent Architecture with LLM-based Orchestration for Conversational Recommendation</strong> - Emilly Pitman de Castro, Vanessa Silva, Gustavo Kanno, Marcus Santos, Bruno Ferraz do Amaral, Marcelo Nogueira, Lucas Francisco Pellicer, Lucas Lanaro, Pedro Augusto Luiz, João Renato Ribeiro Manesco, Giovanni Carbinatti, Isabela de Lima Santos, Lucas Oliveira Jeronymo, Jeison Fappi, Raul Rotilli Aguirre, Fabio André Frantz, Renata De Paris, Fabiana Marucci, Lohan Silva, Vinicius Diniz Reolon, Wandemberg Gibaut - [[pdf]](https://arxiv.org/pdf/2611.08718)</summary>

**Abstract:** Conversational recommender systems often treat user sessions in isolation, limiting their ability to capture how user intent evolves over time. While large language models (LLMs) have improved semantic understanding, they are typically used as isolated components, without addressing how memory, retrieval, and decision-making should be integrated at the system level. In this work, we propose a memory-aware multi-agent architecture for conversational recommendation, in which LLMs act as orchestration layers that coordinate intent refinement, query construction, and interaction flow. A key design principle is the integration of long-term memory as a first-class component, influencing both routing decisions and semantic query formation. We evaluate the system through cross-session memory experiments, multi-turn conversational simulations, and supervisor-driven interaction studies on the Amazon Reviews dataset. Results show that memory-enabled interactions recover up to 63\% of the gap toward explicit-query scenarios and prevent semantic degradation in extended dialogues. In addition, supervisory orchestration increases user acceptance by +28.9 percentage points, highlighting the importance of structured orchestration in guiding interaction and output quality. We also observe an exposure bottleneck between retrieved candidates and displayed recommendations. While LLM-reranking does not improve Hit@K, it consistently enhances semantic consistency and diversity. To capture these effects, we introduce the Semantic Consistency Score (SCS), a complementary evaluation measure. Overall, the results emphasize the importance of system-level design choices, suggesting that effective conversational recommendation benefits from explicit memory integration, orchestration, and evaluation beyond traditional ranking metrics.

**arXiv ID:** 2611.08718
</details>

<details>
<summary><strong>CARGO: Causal Attribution and Responsibility-Guided Preventive Safe Multi-Agent Reinforcement Learning for Cooperative Driving</strong> - Jingjia Teng, Yang Li, Wenjie Huang - [[pdf]](https://arxiv.org/pdf/2611.09139)</summary>

**Abstract:** This study presents Causal Attribution and Responsibility-Guided Optimization (CARGO), a multi-agent reinforcement learning framework for cooperative on-ramp merging in mixed traffic. A shared latent-context posterior and an action-conditioned response model estimate directed behavioral influence, with posterior uncertainty propagated into interaction-edge selection. Complementary attribution mechanisms couple signed influence with marginal team utility for cooperative reward shaping and assign responsibility from positive excess tail risk relative to a safe reference policy. Conditional diffusion generates future responses for action-dependent risk estimation. Responsibility-guided preventive optimization calibrates agent-specific risk-reduction requirements to local capacity within a policy trust region, using reserved safety margins, adaptive requirement scaling, and joint feasibility verification. Theoretical analysis establishes risk monotonicity and sufficient local conditions for non-decreasing environmental return under explicit assumptions. Experiments in highway-env across three traffic densities, together with controlled graph-recovery and attribution studies, support the effectiveness of the framework. Under high-density traffic, CARGO increases the success rate by 66.04\% and reduces the crash rate by 56.23\% relative to the best evaluated baseline for each metric.

**arXiv ID:** 2611.09139
</details>

<details>
<summary><strong>Towards an Autonomous AI Steward for Large-Scale Physics Experiments: Integrating Causal Diagnostics, RAG, and Conformal Prediction</strong> - Oliver Noll, Nicholas Tan Jerome, Alexander Lehnen, Frank Simon, Frank Maas - [[pdf]](https://arxiv.org/pdf/2611.09298)</summary>

**Abstract:** Long-term particle physics experiments face critical operational challenges: monitoring non-stationary machine data streams and preserving tacit expert knowledge. We present the "Helmholtz Co-Pilot," a hybrid Multi-Agent System (MAS) acting as an autonomous experiment steward. The system can be triggered interactively by human operators or autonomously by an asynchronous causal diagnostics engine that runs independently of the MAS reasoning core. Continuously monitoring live hardware streams such as data acquisition and slow control, this engine performs dynamic causal discovery to trace cascading alarms back to their root causes and applies adaptive model selection to keep forecasts reliable across shifting machine states; upon detecting an anomaly, it hands a structured diagnostic report to the MAS. The reasoning layer itself relies on a high-throughput C++ core and a Python edge architecture orchestrating a directed acyclic graph (DAG) of specialized ReAct workers. It integrates a Retrieval-Augmented Generation (RAG) backbone for institutional memory and leverages conformal prediction to bound its evidence-based recommendations with mathematical coverage guarantees. This extended abstract outlines the architecture, the integration of software-enforced Human-in-the-Loop (HitL) guardrails, and ongoing shadow-testing within the KATRIN experiment.

**arXiv ID:** 2611.09298
</details>

<details>
<summary><strong>Travel Cost Calibration for Multi-Agent Target Assignment and Pathfinding</strong> - Yu Kumagai, Ryohei Banno, Keisuke Okumura - [[pdf]](https://arxiv.org/pdf/2611.09396)</summary>

**Abstract:** Target assignment and pathfinding (TAPF) is an appealing extension of classical multi-agent pathfinding (MAPF): each agent has a set of targets, and the planner must both assign a target to each agent and plan collision-free paths for them. A trivial approach is to compute an assignment that minimizes the total single-agent shortest-path distance over agent--target pairs, and then solve the resulting MAPF instance. However, this approach falls short because the shortest-path distance remains a proxy that ignores interference among agents. Assigning targets based on the costs agents actually pay, which depend on how every other agent is assigned and routed, would yield a better final outcome. To this end, we propose an iterative TAPF method that improves target assignments by calibrating travel cost estimates from fast, suboptimal MAPF feedback. Experiments show that our method produces solutions close to optimal on small instances and outperforms existing planners in solution quality on larger ones.

**arXiv ID:** 2611.09396
</details>

<details>
<summary><strong>Alignment by Confident Debate: Preference Learning from Multi-Agent Debate</strong> - Nikhil Verma, Dingxian Wang, Mohammad Reza Rezaei, Andrew Rabinovich, Zhao Chen - [[pdf]](https://arxiv.org/pdf/2611.09568)</summary>

**Abstract:** Post-training language models for reasoning typically rely on gold answers, task-specific verifiers, or learned reward models, limiting scalability when such signals are scarce. Multi-agent debate offers a self-supervised alternative, but converting debate into reliable training supervision remains challenging: majority voting weights agreeing responses equally, lacks a principled tie-breaking rule, and is ill-defined for open-ended tasks without a canonical answer; likelihood-based scoring can conflate fluency with correctness; and homogeneous committees may reinforce correlated errors. We introduce Alignment by Confident Debate (AbCD), which converts debate traces into preference data without gold answer labels. A sampling-diverse committee debates each problem, after which final answers are clustered by task-level semantic equivalence. We use verifiable tasks (mathematical reasoning, multiple-choice quantitative QA, and executable code-generation) as a controlled test bed: their answer keys or executable checkers let us evaluate the quality of debate-derived supervision against ground truth, even though the oracle itself never uses gold labels. Post-training on the resulting preferences outperforms inference-only ensembling and debate, as well as SFT on the same traces, across five model families spanning 1B to 8B parameters and four datasets, improving consensus accuracy, with a maximum gain of $21.5_{pp}$. Ablations show that the approach transfers to a larger model and to code generation, while also revealing limitations of heterogeneous committees and agreement-based weighting. Overall, the results suggest that debate traces can yield useful preference supervision when answer equivalence is combined with confidence-weighted disagreement.

**arXiv ID:** 2611.09568
</details>

<details>
<summary><strong>s$^3$: Simple Scaling of Swarms via Cooperative Multi-agent RL</strong> - Kunal Jha, Gonçalo Faria, Shuyue Stella Li, Scott Geng, Faeze Brahman, Noah A. Smith, Yulia Tsvetkov, Max Kleiman-Weiner, Natasha Jaques - [[pdf]](https://arxiv.org/pdf/2611.08615)</summary>

**Abstract:** Groups can achieve more than individuals in the same amount of time, but only if their members coordinate: they must decide who works on what and what to share. However, this organization in multi-agent systems today usually comes from arbitrary, static harnesses designed a priori without adapting to agent behaviors or task specifications. We ask whether agents can instead learn to organize themselves. We introduce s$^3$, the first multi-agent reinforcement learning method in which agents decide for themselves how long to work alone and when to share findings or read teammates' messages, with no predefined organization or roles. Our s$^3$ agents learn to take different approaches, read each other's work, exhibit emergent roles that complement each other in the swarm, and generalize to bigger teams. We study both coordination, where agents must combine private information, and parallel exploration, where sharing discoveries can make search more efficient. With matched training and inference budgets, s$^3$, swarms evaluated with twice as many agents as in training solve over 25$\times$ more coordination problems than the strongest baseline, and in exploration, communicating agents reduce repeated work by nearly 50% and outperform best-of-N by up to 9.4 percentage points. Rather than being designed in advance, s$^3$ enables organization to emerge from a shared goal. This opens the door to capable systems built from many models that learn to work together.

**arXiv ID:** 2611.08615
</details>

<details>
<summary><strong>Phantom Consensus: Measuring Social Reasoning in Multi-Agent LLM Decision-Making</strong> - Disha Sheshanarayana, Rajat Subhra Pal, Manjira Sinha, Tirthankar Dasgupta - [[pdf]](https://arxiv.org/pdf/2611.08971)</summary>

**Abstract:** Multi-agent LLM systems increasingly aggregate judgments across model instances, but reasoning about other agents can itself introduce failure. We study a setting in which LLMs shift their public decisions after observing ambiguous social signals while reporting inaccurate estimates of what other agents privately judged, even when the group's recorded private judgments do not support the apparent consensus. We call this failure phantom consensus and study it as a pluralistic-ignorance-style pattern in controlled LLM collectives. To measure this phenomenon, we introduce PHANTOM-Bench, a benchmark that separates isolated private judgments, public decisions after ambiguous social signals, elicited estimates of the group's private judgment, transparency-based recovery, and controls for neutral public text and explicit vote-following. Across nine closed- and open-weight models, ambiguous public signals consistently degrade decision quality and can turn correct private group judgments into incorrect public collective outcomes. Public accuracy drops by as much as 47.8 percentage points, and collective rationality failures reach 47.3% in the most susceptible setting. We observe substantial errors in reported group-private beliefs alongside public-stage failures, while control conditions reveal model-dependent differences in susceptibility to ambiguous and explicit peer signals. Revealing the agents' recorded private decisions repairs many failures without revealing the gold label, although recovery varies across models. Our results show that evaluating multi-agent LLM systems requires measuring not only final agreement, but whether that agreement reflects independently measured private judgments or an apparent consensus.

**arXiv ID:** 2611.08971
</details>

<details>
<summary><strong>L-MARS: Legal Multi-Agent System with Agentic Search and Citation-Faithfulness Audit</strong> - Boqin Yuan, Ziqi Wang - [[pdf]](https://arxiv.org/pdf/2509.00761)</summary>

**Abstract:** Large language models are increasingly deployed for legal question answering, where evaluations typically focus on multiple-choice accuracy. This measure overlooks a common failure: whether the citation source attached to an answer exists and supports the rule the system attributes to it. We present L-MARS, an open multi-agent legal QA system with agentic search and judge-driven evidence checks, and audit it claim by claim against its cited source. Each atomic claim is labelled with a six-class taxonomy and scored with strict-ALCE under cross-provider judging, where the answerer and verifier come from different model families. On a stratified 100-question Bar Exam audit, retrieval barely moves accuracy, yet the multi-turn judge loop lifts strict citation F1 from 0.13 (naive RAG) to 0.25 and cuts the no-citation rate from 34% to 13%. We further introduce Faith-Search, a post-draft step that re-verifies and repairs unreachable citations; it drops the unreachable rate below 1% but does not improve F1 over the multi-turn loop, so we report it as a targeted reachability intervention rather than a faithfulness breakthrough. A 50-question LegalSearchQA case study confirms the picture: retrieve-then-draft pipelines saturate near 0.75 citation F1, while a single-agent web-search baseline collapses to 0.22 under external audit.

**arXiv ID:** 2509.00761
</details>

<details>
<summary><strong>Sentinel Agents for Secure and Trustworthy Agentic AI in Multi-Agent Systems</strong> - Diego Gosmar, Deborah A. Dahl - [[pdf]](https://arxiv.org/pdf/2509.14956)</summary>

**Abstract:** Multi-agent systems (MAS) built on large language models (LLMs) increasingly share a common conversational space, in which a single adversarial or erroneous message can propagate from one agent to many. This paper proposes Sentinel Agents, a dedicated security layer that monitors inter-agent communication with rule-based checks, behavioral analysis and LLM-based semantic inspection, and reports to a Coordinator Agent that enforces policy, for example by quarantining misbehaving agents. We describe the architecture, its deployment patterns and monitoring layers, and we evaluate a continuous-listener prototype on a balanced corpus of 1,000 synthetic messages (500 attacks across five families and 500 benign messages) with two local LLMs. LLM inspection with Gemma 4 12B reached F1 0.86 at a false-positive rate of 0.11, against 0.38 for rules alone. Combining components by logical OR did not improve on it, while a cascade in which only high-precision rules act on their own matched it. Calibrating the thresholds of the more conservative Qwen3 8B on a development split raised its held-out F1 from 0.56 to 0.80, and on multi-turn conversations the behavioral check detected patterns the rules missed but could not tell legitimate repetition from probing. The value of combining detectors therefore depends on the model, on its calibration and on the aggregation rule, and the results yield practical guidance for deploying Sentinel Agents.

**arXiv ID:** 2509.14956
</details>

<details>
<summary><strong>EvoAgent: An Evolvable Agent Framework with Skill Learning and Multi-Agent Delegation</strong> - Aimin Zhang, Jiajing Guo, Fuwei Jia, Chen Lv, Boyu Wang, Fangzheng Li - [[pdf]](https://arxiv.org/pdf/2604.20133)</summary>

**Abstract:** This paper proposes EvoAgent--an evolvable large language model (LLM) agent framework that integrates structured skill learning with a hierarchical sub-agent delegation mechanism. EvoAgent models skills as multi-file structured capability units equipped with triggering mechanisms and evolutionary metadata, and is designed to enable continuous skill generation and optimization through a user-feedback-driven closed-loop process. In addition, by incorporating a three-stage skill matching strategy and organizing the agent as a harness whose only self-modified component is a persistent, user-owned state (profile, memory and skills), the framework supports dynamic task decomposition for complex problems and long-term capability accumulation. Experimental results based on real-world foreign trade scenarios demonstrate that, after integrating EvoAgent, GPT-5.2 achieves substantial improvements in professionalism, accuracy, and practical utility. Under a five-dimensional LLM-as-Judge evaluation protocol, the overall average score increases by approximately 28%. Further model transfer experiments indicate that the performance of an agent system depends not only on the intrinsic capabilities of the underlying model, but also on the degree of synergy between the model and the agent architecture. Code and documentation are released at this https URL the evaluation corpus contains proprietary business data that cannot be released, so the reported results are for reference only.

**arXiv ID:** 2604.20133
</details>

<details>
<summary><strong>NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning</strong> - Haoran Lu, Luyang Fang, Wenxuan Zhong, Ping Ma - [[pdf]](https://arxiv.org/pdf/2605.16757)</summary>

**Abstract:** Multi-agent language systems are often built as hand-designed workflows, where agents are assigned semantic roles and communication protocols are specified in advance. We propose NeuroMAS, a framework that treats a multi-agent language system as a trainable and scalable neural-network-like architecture with hidden textual intermediate signals. In NeuroMAS, language-agent nodes are role-free but structure-aware: the topology determines how information can flow, while reward-based training determines how nodes communicate, specialize, and coordinate. This formulation shifts multi-agent design from workflow engineering toward architecture design, where depth, width, connectivity, and growth protocol become scalable sources of capability. We provide a theoretical perspective showing why such modular text-valued computation can be parameter-efficient when tasks admit stable decompositions. Experiments show that NeuroMAS improves over both inference-time collaboration methods and trained baselines, and that its gains are not explained by trainable parameter count alone. We further find that organizational scaling is path-dependent: larger systems can fail when trained from scratch, but become stronger when grown progressively from smaller trained systems. These results suggest that learned multi-agent organization is a promising scaling axis for language-model reasoning and generation.

**arXiv ID:** 2605.16757
</details>

<details>
<summary><strong>It Takes Two to Tangram: Multi-Agent Object Centric Reinforcement Learning for Assembly</strong> - Paul Knaußt, Kay Pompetzki, Nico Bohlinger, Niklas Funk, Cristiana de Farias, Rolf Findeisen, Jan Peters - [[pdf]](https://arxiv.org/pdf/2611.08814)</summary>

**Abstract:** Geometric assembly requires fine spatiotemporal coordination to place parts without causing collisions or deadlocks under tight clearances. To address this, we present an object-centric reinforcement learning framework that treats each part as an agent and learns a shared policy for continuous assembly control. A graph neural network represents the pieces, their target poses, and their spatial relationships, and predicts translational and angular velocities for each component. Per-agent proximal policy optimization with a centralized graph critic directly learns motion and temporal coordination within a single feedback policy. We study this formulation through planar Tangram assembly, where pieces must slide and rotate into a given configuration while remaining on the assembly surface. Procedural generation of target configurations and randomized initial arrangements support training and evaluation across diverse assembly problems. Our experiments isolate the roles of relational structure and multi-agent optimization, and compare learned control with sampling-based planning. We demonstrate concurrent assembly in simulation and transfer the policy to sequential execution on a real robot, as our object-centric formulation decouples learned assembly coordination from low-level physical control. Project page: this https URL

**arXiv ID:** 2611.08814
</details>

</details>

<details open>
<summary><h2>Other Agent Research (8 papers)</h2></summary>

<details>
<summary><strong>Rethinking Agent Trajectory Compression: A Workflow-Aware Perspective</strong> - Qijia Zhuang, Rui Liu, Yuxiang Ren - [[pdf]](https://arxiv.org/pdf/2611.08865)</summary>

**Abstract:** As long-horizon agents increasingly rely on multi-step tool interactions to accomplish complex tasks, their execution trajectories continuously accumulate structured outputs and intermediate artifacts, leading to growing context overhead. Existing compression methods primarily rely on text importance or fixed truncation rules, often overlooking cross-step dependencies within workflows and potentially discarding critical information required for subsequent task execution. To address this challenge, we propose WAAC, a workflow-aware framework for agent trajectory compression. WAAC structurally models successful historical trajectories to extract reusable workflow compression experience and incorporates online backward dependency analysis to dynamically identify information retention requirements. By combining current dependency constraints with historical experience, WAAC selects appropriate compression strategies for different types of historical information. Experiments on PinchBench and Odysseys demonstrate that WAAC achieves an overall score of 87.6 and a perfect completion rate of 19.0%, respectively, while reducing execution costs by 19.7% and 29.1% compared with the strongest baseline methods. These results demonstrate the effectiveness of workflow-aware trajectory compression in improving task performance and cost efficiency for long-horizon agents. Code and resources are available at this https URL.

**arXiv ID:** 2611.08865
</details>

<details>
<summary><strong>What to Delegate, What to Preserve: Human Preferences for Agentic AI Across Everyday Life and Work</strong> - Sarah Wheeler, Veronica Rammouz, Yaoyao Qian, Zexuan Liu, Mengyi Wei, Anthony Rios, Tianzhi He, Kyrie Zhou - [[pdf]](https://arxiv.org/pdf/2611.09730)</summary>

**Abstract:** AI agents can execute multi-step tasks on users' behalf, yet the boundaries of acceptable delegation remain unclear. We report a scenario-based survey of 289 U.S. adults with prior AI experience covering 45 use cases across nine life domains. Each respondent evaluated one scenario per domain, producing repeated measures of automation preference, trust, collaboration, AI leadership, privacy concern, and willingness to pay. Automation preferences varied substantially across domains. Generalized estimating equation models showed that greater trust in the AI agent, desired AI leadership, prior task-specific AI use, and anticipated financial savings were associated with greater automation preference. In contrast, concerns that automation would reduce personal fulfillment predicted lower automation preference. Open-ended responses further revealed a preference for bounded autonomy, with participants often favoring substantial AI execution while retaining human involvement, oversight, or final authority. We argue that delegation is best understood as a configuration of execution, collaboration, leadership, and authority.

**arXiv ID:** 2611.09730
</details>

<details>
<summary><strong>Test-Time Compute Scales Better with Parallel Agent Teams</strong> - Junlin Wang, Haozhe Chen, Ang Li, Rahul Chalamala, Zhuang Liu - [[pdf]](https://arxiv.org/pdf/2611.09798)</summary>

**Abstract:** Test-time scaling improves the performance of a fixed model by spending more computation at inference. There are two common ways to spend it on agents when a verifier can score every attempt. Best-of-N sampling runs many independent agents and keeps the best result but saturates quickly. One large team of collaborating agents can score higher but stops improving as it grows. Can we get the best of both? Our method, parallel agent teams, does so by sampling many small teams in parallel and keeping the best team's result. We test it on 12 long-horizon tasks from GPU kernels and formal proofs to exploit development. With a 27B model and a budget of 100 agents, parallel teams of ten beat parallel agents on 10 of 12 tasks and beat one team of 100 on all four tasks where we ran such large teams. At equal dollar cost the same parallel teams beat a frontier model on half the tasks. A simple statistical model shows that individual agents lose because their many samples have a low mean and that one large team loses because a single sample gets no gain from variance. Sampling many small teams in parallel balances a high mean with the gain from variance. Parallel agent teams thus offer a more effective way to scale test-time compute.

**arXiv ID:** 2611.09798
</details>

<details>
<summary><strong>Can Agentic Systems Close the Gap in Radiology with Production AI Tools?</strong> - Aubin Charley, Julien Khlaut, Charles Corbière, Pierre Manceron, Corentin Dancette, Léo Machado - [[pdf]](https://arxiv.org/pdf/2611.09836)</summary>

**Abstract:** Radiology AI has developed along two paths: specialized models that perform well on narrow tasks but cover few clinical findings, and generalist models that address a broader range of tasks but remain prone to hallucination and inaccurate measurements. Combining the two through agentic systems offers a route towards comprehensive radiology reporting, but published systems are tied to one modality or body part. We describe a radiology agentic framework that standardizes how an orchestrator calls modular tools to produce a structured report, and use it to measure what such a system can do with the tools from FDA-cleared radiology AI devices. Report generation is simulated on a large multi-modality report corpus, with the orchestrator given either a catalog of finding classes consolidated from the cleared devices, each available on the modalities it is cleared for, or an unlimited catalog in which any class can be requested, and planning either from the clinical indication or from the target report. Tool outputs are simulated from the reference report, so the comparison isolates tool availability and orchestration from tool error. With the cleared catalog and the indication alone, the orchestrator recovers under a third of the clinically actionable findings; recovery improves substantially with the unlimited catalog, and further when the plan is written from the target report. Attributing every missed finding to a cause shows that most are classes no cleared device offers for the study's modality, and that a third are classes or tools that were offered and not called. Adding the most frequently missing tools closes most of the gap to the unlimited catalog on held-out reports, whereas a smaller orchestrator with the unlimited catalog gives up the whole gain. Tool availability and effective orchestration are thus complementary constraints on comprehensive AI-assisted radiology reporting.

**arXiv ID:** 2611.09836
</details>

<details>
<summary><strong>CPU-ARC: Agentic Reasoning Caching for on-CPU Reasoning Reuse</strong> - Peter Belcak, Greg Heinrich, Saurav Muralidharan, Shizhe Diao, Pavlo Molchanov - [[pdf]](https://arxiv.org/pdf/2611.09891)</summary>

**Abstract:** Large language models reinvent the wheel all the time. They conflate task-solving with the execution of the solutions, which forces them to solve every problem from scratch. We show that once a task has been reasoned through, the application of its solution to a new problem instance can be done very efficiently and with little involvement from language models. We introduce agentic reasoning caching, a method for constructing robust CPU-executable agentic reasoning caches (ARCs) from LLM reasoning traces. ARCs show on-par performance with frontier LLMs across a range of difficult mathematical and scientific tasks, are fully explainable, continually trainable, robust to input perturbations, and 16-291x cheaper to invoke than the source model.

**arXiv ID:** 2611.09891
</details>

<details>
<summary><strong>A Controlled Study of Models, Skills, Feedback, and Curriculum in Agentic Robotics</strong> - Kaiyuan Chen, Xinyue Zhu, Shuangyu Xie, Uksang Yoo, Thi Thu Phuong Luong, Marcus Gualtieri, Christian Juette, Liu Ren, Spencer Huang, Ken Goldberg - [[pdf]](https://arxiv.org/pdf/2611.09030)</summary>

**Abstract:** Agentic robotics asks coding agents to implement and iteratively improve robot policies from simulated and real-world physical interactions. We study the self-learning loop of agentic robotics through a controlled simulation study spanning nine manipulation tasks, 631 different experiment configurations, and 28K coding agent rollouts, systematically varying model capability, skill priors, feedback structure, and curriculum. The experiment data suggests that the bottleneck is often not the access to sufficient information, but interpreting it into the right policy update. Across these factors, a successful self-learning loop depends on how effectively agents attribute failures, reason task-relevant spatial and geometric structure, and selectively restructure code to improve policies.

**arXiv ID:** 2611.09030
</details>

<details>
<summary><strong>Language as Trait, Politeness as Strategy: Controlled Tests of Linguistic Effects in LLM Social-Deduction Agents</strong> - Yuchen Li, Lisa Liu, Runqi Zou, Shuxuan Kuang, Qingquan Zhang, Julian Togelius - [[pdf]](https://arxiv.org/pdf/2611.09768)</summary>

**Abstract:** Large language model (LLM) agents coordinate through conversation, and the same message can be phrased in simpler or more complex language, bluntly or politely. We test whether such differences change how other agents judge and act, using the LURK Werewolf framework, which separates each utterance's speech-act predicate from its natural-language realization. We rewrite only the realization, check that the game content is preserved, and replay each game day from saved checkpoints against a re-sampled control. We vary linguistic range consistently across a speaker's messages and politeness strategy per utterance. Across 30 games and 4,335 replays, neither manipulation changes the trust other agents place in the speaker, and no effect on votes or night actions survives correction for multiple comparisons; the largest uncorrected effect is that shifting range away from a neutral paraphrase changes about 4 percentage points more of the other players' votes. In natural play, 88% of face-threatening speech is bald on-record regardless of role.

**arXiv ID:** 2611.09768
</details>

<details>
<summary><strong>From Procedures to Permissions: Learning Precise Subagent Authorization from Agent Skills</strong> - Wenkai Yu, Ning Ding, Wentao Zhang, Yehui Tang - [[pdf]](https://arxiv.org/pdf/2611.08905)</summary>

**Abstract:** As language agents tackle increasingly complex tasks, they increasingly rely on multiple agents to coordinate specialized work, requiring decisions not only about what each subagent should do but also which tools and workspace resources it should access. Existing agent systems provide mechanisms for controlling tools and execution environments, but do not directly specify the resources required by each delegated subtask. Moreover, task outcomes and execution trajectories alone provide limited supervision for determining whether a subagent has received appropriate permissions. We observe that Agent Skills encode procedural knowledge describing the operations, tools, and files required at each step, providing a natural source of supervision for authorization. Building on this observation, we synthesize tasks and step-by-step procedures from Skills, execute them through subagent delegation in interactive environments, and use verified trajectories to construct training data for task execution and authorization. On ClawArena-Team, our method improves tool permission precision by 13.4 percentage points and workspace permission precision by 24.1 percentage points, while increasing overall task success by 11.6 percentage points for Qwen3.5-35B-A3B. On Terminal-Bench 2.1 and SkillsBench, our method remains competitive with existing agent data synthesis approaches. These results show that the procedural knowledge embedded in Agent Skills provides a source of supervision for precise subagent orchestration that task outcomes alone cannot provide.

**arXiv ID:** 2611.08905
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>I Think, Therefore I Attack: Evaluating Thought Injection Attacks on Reasoning Agents</strong> - Parth Illendula, Garik Livingston, Zain Shamsi, Ravi Srinivasan, Alex Liu - [[pdf]](https://arxiv.org/pdf/2611.08811)</summary>

**Abstract:** Large Reasoning Models (LRMs) are one of the key components of current agentic systems. LRMs augment their output with an explicit chain-of-thought reasoning stage, introducing a new attack surface for thought injection, in which untrusted text resembling the model's own deliberation is fed back to the model. We present a two-part study of this attack, performed against the prevailing large, production class open-weight models (e.g., GLM-5.3, MiniMax-M2.5, Qwen3.8-Flash-Next). We first evaluate the effectiveness of static thought-injection prompts under different styles as well as tagged and untagged conditions and find that the effects of thinking tags and payload styles are all properties of the specific target model. In the second study, we exploit this model-specificity in a containerized agentic cyber range, using indirect injection prompts styled in the form of the target model's own reasoning traces to achieve attack success rates (ASR) up to 50% on operational objectives such as reverse-shell exfiltration and repository backdooring. We also find that the same malicious objective in standard prompt injection user voice never succeeds, and thinking tags suppress the agent's tendency to disclose the injection. Our results show that imitating a reasoning model's own thought style is effective for this style of attack, and that tag stripping is an unreliable defense.

**arXiv ID:** 2611.08811
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (17 papers)</h2></summary>

<details>
<summary><strong>Safe Reinforcement Learning via Probabilistic Option Shielding</strong> - Gian Luca Ascari, Isidoro Tamassia, Ying Jiao, Luc De Raedt, Giuseppe Marra - [[pdf]](https://arxiv.org/pdf/2611.09197)</summary>

**Abstract:** Shielding is a safe reinforcement learning technique that provides safety guarantees while learning, by blocking unsafe actions. Traditional shielding relies on a complete model of the environment to provide long-term guarantees. Probabilistic logic shielding (PLS) lifts the requirement of having such model, but only considers safety in the short term, typically looking only one or few steps ahead. We propose Probabilistic Option Shielding (POS), which extends PLS to the option level, recovering long-horizon safety without reintroducing the need for a complete environment model. POS learns a model-free critic directly from one-step PLS signals, to estimate the probability that an entire option will complete safely under uncertainty. We evaluate POS on safe navigation tasks in the LogiCity environment, and show that it achieves considerably safer training while matching the task performance of standard step-level PLS and other safety related reinforcement learning baselines.

**arXiv ID:** 2611.09197
</details>

<details>
<summary><strong>Zero-Query Transferable Pretend Benign Attacks with Amortized Online Generation for Cooperative Perception</strong> - Hongwei Lin, Xun Huang, Chenglu Wen, Cheng Wang - [[pdf]](https://arxiv.org/pdf/2611.09256)</summary>

**Abstract:** Cooperative perception enables an ego agent to acquire information from collaborators beyond its own sensors' reach. Yet, the most beneficial messages are often the hardest to verify from local observations. Pretend Benign (PB) exploits this asymmetry by targeting regions with weak local verifiability and applying multi-objective perturbations, making harmful feature changes appear to be benign assistance. However, PB is victim-dependent: selecting regions and optimizing perturbations require the victim model's architecture, parameters, and gradients, limiting transfer to unseen victims. A transferable attack must capture properties shared across cooperative detection models rather than those specific to one network. We address this by building a victim-agnostic attack from a heterogeneous ensemble of surrogate models and encoding its capabilities in a reusable dictionary to reduce per-message generation cost. Specifically, Transferable Pretend Benign (TrPB) transfers PB's region selection criterion and multi-objective perturbation strategy across heterogeneous surrogates to construct an offline attack. We further propose RT-TrPB, which encodes transferable regions and perturbation patterns into a region predictor and reusable perturbation dictionary, amortizing per-message attack generation. Experiments on widely used cooperative perception benchmarks show that RT-TrPB bypasses the consensus-based defenses ROBOSAC and FLD with success rates approaching 100% and generates each perturbation in under 56 ms. To counter PB-style attacks masquerading as benign collaboration, we also propose STTrust, which verifies received messages using spatial and temporal consistency and incorporates identity-level trust. STTrust detects over 90% of PB-style attacks and restores perception performance to near the attack-free level.

**arXiv ID:** 2611.09256
</details>

<details>
<summary><strong>PhyCAD: Physics-Aware Text-to-CAD Generation via Hierarchical Reinforcement Learning</strong> - Yuxin Cui, Weijie Zheng, Zhengyang Li, Han Wang, Wenjie Feng, Qiyu Kang - [[pdf]](https://arxiv.org/pdf/2611.09401)</summary>

**Abstract:** Language-driven computer-aided design (CAD) generation enables the creation of editable parametric models from text, yet geometric correctness alone does not ensure physical feasibility. We present PhyCAD, a framework for physics-aware Text-to-CAD generation that trains language models to produce executable CadQuery programs under geometric and physical constraints. To support this setting, we construct a Text-to-CAD dataset that links parametric CAD models with geometric specifications, loading conditions, and physical constraints. We further develop an automated CAD--CAE verifier that combines program execution, geometric checks, and finite element analysis to provide feedback for model training. A key learning challenge is to prioritize joint satisfaction of geometric and physical constraints while retaining informative feedback on partial progress. A flat aggregation of heterogeneous rewards can obscure this priority, allowing improvements in individual criteria to compensate for unmet design requirements. We therefore introduce a Hierarchical Balanced Advantage for Group Relative Policy Optimization, grouping candidates into levels according to their verification status and combining between-level progress with objective-wise peer ranking within each level. This formulation preserves cross-level priority in the training signal while providing fine-grained feedback for satisfying constraints and improving feasible designs. Experiments across multiple language models show that PhyCAD consistently improves joint geometric and physical constraint satisfaction. These results demonstrate the value of verifiable engineering feedback for advancing language-driven CAD generation beyond geometric reconstruction toward physics-aware design.

**arXiv ID:** 2611.09401
</details>

<details>
<summary><strong>Agentic AI for Healthcare Claim Denial Analysis: A Policy-Grounded Framework for Privacy Preservation</strong> - Sofia Campos, Carlos Martins, Pedro Pires, João Gama, Bruno Veloso, Pedro Santos - [[pdf]](https://arxiv.org/pdf/2611.09423)</summary>

**Abstract:** In the healthcare sector, claim denials impose significant financial and administrative burdens, emphasizing the need for automated and trustworthy decision-support systems. This paper proposes a modular Agentic Artificial Intelligence (AI) framework for healthcare claim denial analysis, integrating specialized agents for denial risk prediction, risk factor identification, diagnosis-procedure-medication code relationship analysis, and counterfactual analysis. The framework integrates predictive analytics, graph-based learning, a Hybrid Contextual Retrieval-Augmented Generation (RAG) pipeline grounded in healthcare policy documents, and a dedicated HIPAA Compliance Agent to support privacy protection. Evaluated on a synthetic healthcare claims dataset, the framework achieved 98% traceability of reported numerical findings, validated 214 of 216 policy citations, and removed all 280 identified sensitive identifiers without modifying the underlying analysis. These findings pave the way for further research into Agentic AI as a framework for explainable, policy-aware, and privacy-preserving healthcare claim denial management.

**arXiv ID:** 2611.09423
</details>

<details>
<summary><strong>ENGRAM: From Agent Experience to Weights through Verified Consolidation</strong> - Ruiyi Yang, Zechen Li, Lihuan Li, Du Yin, Hao Xue, Flora D. Salim - [[pdf]](https://arxiv.org/pdf/2611.09472)</summary>

**Abstract:** Language-model agents use experience through external retrieval and parameter learning. Consolidating stored experience into model weights requires deciding both which experience to train on and when its external records can leave retrieval. Admission based solely on student mastery can exclude useful teacher demonstrations, while removal immediately after training can discard guidance that the model still needs. We introduce ENGRAM, a framework that separates training admission from memory release. A demonstration-aware gate admits skills based on student mastery or sufficient successful teacher demonstrations. The student learns from selected action traces with context from an Engram Trace Graph (ETG). A verification gate then compares the updated model without episodic retrieval against the base model with retrieval on reserved probes. Release requires valid training participation and scores meeting both relative and absolute criteria; records for skills that fail verification remain retrievable. Experiments across interactive and reasoning benchmarks show consistent gains over frozen-weight ETG controls, with ablations supporting demonstration-aware admission and selective adapter activation. A controlled held-out comparison shows that verified release substantially reduces the retrievable footprint while remaining close to retain-all performance and outperforming immediate and budget-matched random release. ENGRAM also improves task performance while reducing input-token usage and model calls. Skill-level verification further reduces retrievable experience through selective archival, retaining external guidance for skills that do not meet the release criteria. Code is available in this http URL.

**arXiv ID:** 2611.09472
</details>

<details>
<summary><strong>On the Reliability of Majority-Vote Pseudo-Labels in Test-Time Reinforcement Learning</strong> - Haoyu Wang, Ru Wang, Yuanyuan Wang, Jiaxian Guo, Mingming Gong - [[pdf]](https://arxiv.org/pdf/2611.09872)</summary>

**Abstract:** Test-time reinforcement learning (TTRL) adapts pretrained language models to unlabeled target-domain data using self-generated pseudo-labels, whose reliability remains poorly understood. We analyze majority-vote supervision for TTRL with online Direct Preference Optimization (DPO) and decompose its error into population-majority bias and prompt-dependent finite-rollout variation. Because population-majority bias is generally unidentifiable without external supervision, we focus on mitigating the downstream effects of finite-rollout variation. We propose \emph{Prompt-Conditional Distributionally Robust DPO (this http URL)}, which conditions within-prompt robust reweighting on observed vote reliability to make updates from sampling-sensitive prompts more conservative. We further introduce a cache-based mechanism to reduce self-supervised forgetting during online adaptation. Across six mathematical and knowledge-intensive reasoning benchmarks and five model backbones, this http URL matches or outperforms the strongest TTRL baseline in 24 of 30 settings.

**arXiv ID:** 2611.09872
</details>

<details>
<summary><strong>LLM-Enhanced Self-Evolving Reinforcement Learning for Multi-Step E-Commerce Payment Fraud Risk Detection</strong> - Bo Qu, Zhurong Wang, Daisuke Yagi, Zhen Xu, Yang Zhao, Yinan Shan, Frank Zahradnik - [[pdf]](https://arxiv.org/pdf/2509.18719)</summary>

**Abstract:** This paper presents a novel approach to e-commerce payment fraud detection by integrating reinforcement learning (RL) with Large Language Models (LLMs). By framing transaction risk as a multi-step Markov Decision Process (MDP), RL optimizes risk detection across multiple payment stages. Crafting effective reward functions, essential for RL model success, typically requires significant human expertise due to the complexity and variability in design. LLMs, with their advanced reasoning and coding capabilities, are well-suited to refine these functions, offering improvements over traditional methods. Our approach leverages LLMs to iteratively enhance reward functions, achieving better fraud detection accuracy and demonstrating zero-shot capability. Experiments with real-world data confirm the effectiveness, robustness, and resilience of our LLM-enhanced RL framework through long-term evaluations, underscoring the potential of LLMs in advancing industrial RL applications.

**arXiv ID:** 2509.18719
</details>

<details>
<summary><strong>Relic and Gumi: Governing Memory, Safety, and Continuity in Diegetic Relational Agents</strong> - Daniele Veri' - [[pdf]](https://arxiv.org/pdf/2611.08504)</summary>

**Abstract:** Longitudinal relational agents rely on memory for continuity, but remembering can lead to intrusion, clinical overconfidence, emotional manipulation, and opacity. We present Relic, a governance architecture for diegetic relational agents, and Gumi, a subject-specific companion agent initialized and governed through Relic. The architecture separates what the system can observe, what researchers can review, what Gumi can remember, and what Gumi can say.
We describe three mechanisms. Relic represents sensitive safety signals as researcher-facing governance objects rather than user-facing explanations or agent memories. Shared Continuity Memory organizes relational markers around subject-confirmed wording, corrections, explicit follow-up permissions, and recall limits. We specify a runtime mediation layer for delivery, proactivity, session identity, output transformation, and platform allowlists; adoption across live paths remains partial. These mechanisms aim to preserve relational continuity without turning Gumi into a clinical assistant, mood tracker, general-purpose assistant, or opaque profiling system.
We contribute the architecture, object model, researcher workbench design, and a proposed evaluation framework for governed relational memory. An exploratory pilot involved the author as P1 and one external participant as P2. Weekly self-reports from P1 and researcher-led interviews with P2 provide descriptive observations of early use and perceived naturalness. The pilot does not support claims about clinical efficacy, diagnosis, therapeutic outcomes, general usability, or deployment safety. Relic provides research infrastructure intended to make relational memory, safety constraints, and agent behavior inspectable and testable.

**arXiv ID:** 2611.08504
</details>

<details>
<summary><strong>Dual-Stream LLM-Guided Reward Design for Reject-Inference-Aware Reinforcement Learning in E-Commerce Payment Fraud Detection</strong> - Bo Qu, Zhurong Wang, Frank Zahradnik, Ming Ouyang - [[pdf]](https://arxiv.org/pdf/2611.08706)</summary>

**Abstract:** Production fraud-detection models learn from a censored slice of their own traffic: transactions a legacy system blocked never receive fraud labels, so models train on a biased, approved-only distribution. We extend LLM-guided reward design, the framework of Qu et al. (2025) in which an LLM evolves the reward of a two-step REINFORCE fraud-blocking policy, to attack this reject-inference problem inside the reward. A single LLM designs a dual-stream reward: stream 1 applies the standard dollar-weighted precision objective to labeled transactions; stream 2 scores legacy-blocked, unlabeled transactions by contrasting the newest gradient-boosted risk signal against the legacy scores that drove the original block. Rather than prescribe which signals matter, we expose a superset of evidence and let the LLM decide for itself which to use, guided by evaluation feedback alone; because it writes the reward as code, that choice is directly auditable. On a held-out four-month out-of-time test, the dual-stream agent beats the prior labeled-only agent at every target recall, by +4.7, +2.4 and +3.0 points of dollar-weighted precision at 80%, 85% and 90% recall, and the production baseline by +6.8 points on average. The resulting policy is architecturally identical to that of Qu et al. (2025) and adds no inference cost.

**arXiv ID:** 2611.08706
</details>

<details>
<summary><strong>Adaptive Scheduling Pipeline for Multi-Instance Asynchronous Reinforcement Learning</strong> - Salah Chikhi, Luis Zuin, Entong Li, Zeng Li - [[pdf]](https://arxiv.org/pdf/2611.08864)</summary>

**Abstract:** We present the Adaptive Scheduling Pipeline (ASP), an offline rollout scheduler for multi-instance asynchronous reinforcement learning (RL) of large language models (LLMs). ASP jointly reduces three sources of rollout underutilization - padding bubbles from intra-minibatch length heterogeneity, skewness bubbles from long-tail outputs, and sorting bubbles from fluctuations in training-queue occupancy induced by length-sorted dispatch - through a single offline preprocessing pass. This enables efficient multi-instance rollout that overcomes the early scaling plateau of single-instance inference while smoothing the arrival of generated samples at the trainer. We integrate the scheduler into an asynchronous architecture that decouples rollout from training while keeping policy lag observable and bounded. Complete ASP improves end-to-end throughput by up to 3.22 times and reduces iteration time by up to 76.8% relative to MindSpeed-RL, while reducing iteration time by up to 62.5% relative to verl. Within ASP, scheduling reduces mean iteration time by 10.9%. ASP also scales more robustly with additional generation resources than these baselines and shows no visible convergence degradation in the measured bounded-lag regime. ASP has since been used in an internal telecom-domain RL post-training workflow, reducing the reported training period from more than three months to about two months and producing a domain-enhanced model for knowledge answering and tool calling.

**arXiv ID:** 2611.08864
</details>

<details>
<summary><strong>Not All Entropy Is Equal: Reward-Conditioned Entropy Allocation in Reinforcement Learning with Verifiable Rewards</strong> - Song Yu, Li Li, Zhenming Wu - [[pdf]](https://arxiv.org/pdf/2611.09448)</summary>

**Abstract:** Policy entropy is widely used to diagnose exploration and collapse in reinforcement learning with verifiable rewards (RLVR), yet its magnitude does not consistently track reasoning performance across optimization objectives. We argue that this ambiguity arises because global token entropy conflates uncertainty in successful and failed trajectories, diversity across correct solution strategies, and variation among realizations of the same strategy. We introduce \textbf{Reward-Conditioned Entropy Allocation (RCEA)}, a hierarchical diagnostic framework that separates complete-trajectory entropy from average token entropy, conditions uncertainty on verifier outcome and process validity, measures correct-solution-mode retention, and connects these population-level quantities to local covariance-driven updates. In controlled sweeps over six RLVR objectives at two model scales, entropy trajectories do not induce a stable ordering of reasoning accuracy. Token entropy is higher for verifier-failed than successful trajectories under both broad separately aggregated estimates and a common-question paired analysis, whose confidence intervals exclude equality at 10 of 12 displayed endpoints. A local covariance approximation predicts the direction of adjacent-checkpoint entropy change in 95.5\% of nonfailed transitions; a reward-stratified endpoint proxy further shows that positive chosen-token covariance is more tail-concentrated within failed prefixes. Every reportable trained endpoint exhibits some contraction in independently audited correct-solution-mode breadth, whose positive association with large-\(k\) coverage remains exploratory. These results show that entropy collapse should be diagnosed by which uncertainty is removed, rather than by global entropy alone.

**arXiv ID:** 2611.09448
</details>

<details>
<summary><strong>Encoding Memory in World Models with Test-Time Model-Based Reinforcement Learning</strong> - Maxime Burchi, Fayaz Ali Dharejo, Radu Timofte - [[pdf]](https://arxiv.org/pdf/2611.09625)</summary>

**Abstract:** Model-based reinforcement learning (MBRL) approaches have demonstrated strong performance across diverse domains in recent years, learning accurate world models from agent trajectories. However, despite this progress, current approaches still struggle to surpass human performance on tasks that require long-term memory and planning. In this work, we address the problem of long-term memory in MBRL by moving beyond the commonly studied paradigm of architecture design. We introduce Test-Time Model-Based Reinforcement Learning (TT-MBRL), using gradient descent at test time to encode the memory of episode observations into world model parameters. This enables the agent to learn a distinct world model for each episode, improving memory capabilities compared to traditional approaches that solely rely on contextual memory. We first establish a proof of concept on a simplified two-dimensional variant of the Memory Maze environment, which we refer to as Memory Maze 2D, showing consistent improvements across different world model architectures. We then demonstrate the effectiveness of TT-MBRL in the 3D domain, achieving new state-of-the-art results on the Memory Maze benchmark. Finally, we show that TT-MBRL maintains strong performance on non-memory benchmarks such as DeepMind Control and Atari 100k.

**arXiv ID:** 2611.09625
</details>

<details>
<summary><strong>RL-ATTACK: A Generalized Black-Box Adversarial Attack on Language Models via Reinforcement Learning</strong> - Taikun Chen, Yue Zhang, Wei Qian, Hui Fang - [[pdf]](https://arxiv.org/pdf/2611.08711)</summary>

**Abstract:** Large language models (LLMs) exhibit vulnerabilities to adversarial perturbations that can induce unintended responses. These vulnerabilities can be identified via intentional attacks and subsequently used to improve model safety. In black-box settings where gradients are inaccessible, existing methods tend to guess adversarial transformations using external knowledge, ignoring the information provided by the target model. To mitigate this, we propose a trainable black-box framework based on reinforcement learning, which learns from the target model to generate tailored adversarial transformations. We demonstrate that our approach achieves lower post-attack accuracy, greater query efficiency and better semantic preservation on both natural language inference and generative Q&A tasks. On natural language inference, it drives post-attack accuracy to 2% within 20 queries. Against generative language models, our trained attacker consistently achieves lower post-attack accuracy than the untrained attacker and wins in 4 out of 6 settings when compared to a strong optimization-based baseline. We further experiment with multiple reward designs and training settings, and we find that no design or setting is consistently dominant. Our code is available at this https URL.

**arXiv ID:** 2611.08711
</details>

<details>
<summary><strong>MARS: Training Multilingual Search Agents for Retriever-Free Evidence Acquisition</strong> - Rui Qi, Yunlong Liang, Yufeng Chen, Jinan Xu, Fandong Meng, Kaiyu Huang - [[pdf]](https://arxiv.org/pdf/2611.09090)</summary>

**Abstract:** Existing multilingual retrieval-augmented generation relies heavily on learned retrievers and similarity-based ranking from multilingual collections, which are vulnerable to cross-lingual retrieval errors and indexing overhead. In this paper, we explore retriever-free multilingual search, where the model directly interacts with raw corpora through executable shell commands. This paradigm shifts the bottleneck to multilingual search action generation, where exact search is sensitive to entity surface forms and outcome-only supervision provides limited guidance for multi-turn search. To address these challenges, we propose MARS, a multilingual search agent in direct corpus interaction (DCI) with reinforcement learning (RL). MARS introduces corpus-native entity grounding to recover reliable lexical anchors from corpus evidence, mitigating failure to retrieve evidence. Meanwhile, we present a redundancy-aware search reward to encourage concise and effective search trajectories, addressing training collapse in RL of multilingual scenarios. Experiments show that \ours~consistently outperforms strong retrieval-based and DCI baselines, while demonstrating robust cross-lingual generalization and substantially reducing redundant corpus interactions.

**arXiv ID:** 2611.09090
</details>

<details>
<summary><strong>Towards Robust Reinforcement Learning Market Making: Lessons from Loss Function Experimentation</strong> - Siddharth Nand, Bora Guney - [[pdf]](https://arxiv.org/pdf/2611.09012)</summary>

**Abstract:** Market makers play a vital role in ensuring efficient trade within financial markets. Recent literature has delved into training market-making systems using reinforcement learning. This paper delves into the impact of various loss function formulations on risk management behavior within reinforcement-based market making. Through empirical evaluation, we contrast profit-maximizing strategies with risk-averse approaches, finding that penalties for inventory and cash management foster more resilient and adaptable market-making strategies. Our results underscore the significance of striking a balance between profit maximization and risk management in algorithmic trading, providing actionable insights for refining strategies in practical settings.

**arXiv ID:** 2611.09012
</details>

<details>
<summary><strong>Safety-Focused Reinforcement Learning Using Risk-Sensitive Coupled Control</strong> - Jiabin Lin, Shana Moothedath, Vivek S. Borkar - [[pdf]](https://arxiv.org/pdf/2611.09290)</summary>

**Abstract:** This paper investigates the coupled control problem within safety-focused reinforcement learning, addressing the essential challenge of achieving optimal performance while ensuring safety in systems operating within complex environments. We define safety as the requirement to prevent the system from reaching undesirable or hazardous states. We propose a coupled control approach that prioritizes safety by employing effective methods to prevent the system from entering unsafe states while optimizing the performance metric, defined as the average cost. We propose a novel problem transformation that reformulates it into a risk-sensitive problem combined with an average reward criterion. We present a two-timescale Q-learning algorithm where in the faster timescale, the algorithm solves a risk-sensitive control problem to ensure safety, while on a slower timescale, the algorithm focuses on average cost control. We provide theoretical guarantees for the convergence of our algorithms, establishing their significance in ensuring safe, optimal control in dynamic environments. We validated the performance of our approach using GridWorld and CarGoal task in Safety Gymnasium.

**arXiv ID:** 2611.09290
</details>

<details>
<summary><strong>DobotTrack: Target-Grounded Vision-Language-Action Learning for Embodied Person Tracking</strong> - Shengjun Deng, Yinghan Zhang, Chengrui Wei, Liu Liang, Songyuan Liu, Pengkun Zheng, Yajun Zheng, Ying Zhang, Yuan Tian, Xi Li, Teng Wang - [[pdf]](https://arxiv.org/pdf/2611.09041)</summary>

**Abstract:** Language-guided embodied person tracking requires a mobile agent to identify a target from a referring description and consistently follow the target under egocentric observations. We present DobotTrack, a target-grounded vision-language-action (VLA) framework that integrates auxiliary grounding supervision with efficient temporal modeling. We systematically investigate five auxiliary tasks---person segmentation, bounding-box grounding, visual-spatial classification (VSC), future DINO feature prediction, and future-frame prediction---and adopt complementary mask and bounding-box supervision to enhance target delineation and localization. We introduce Q-identity, a target-conditioned historical visual compression module that leverages target-grounding predictions to selectively preserve identity-discriminative appearance features and spatial cues in compact historical tokens. On EVT-Bench, our 0.8B-parameter model achieves an average success rate of 77.9%, exceeding OmTrackVLA by 16.9 percentage points across the three tasks. We further construct a real-world person-tracking dataset collected using a quadruped robot, comprising 126 episodes and 596,323 mask-annotated frames. The target masks enable automatic derivation of bounding boxes and visual-spatial labels encoding horizontal position and apparent image scale, providing additional grounding supervision without separate manual annotation of these labels. Together, our framework and annotation pipeline support learning target-grounded tracking policies from simulated and real-world data. Code is available at this https URL.

**arXiv ID:** 2611.09041
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
