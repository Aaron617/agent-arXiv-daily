# Agent arXiv Daily

**Last Updated:** 2026-03-25 03:46:14

**Total Papers:** 80

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
<summary><strong>Learning When to Act: Interval-Aware Reinforcement Learning with Predictive Temporal Structure</strong> - Davide Di Gioia - [[pdf]](https://arxiv.org/pdf/2603.22384)</summary>

**Abstract:** Autonomous agents operating in continuous environments must decide not only what to do, but when to act. We introduce a lightweight adaptive temporal control system that learns the optimal interval between cognitive ticks from experience, replacing ad hoc biologically inspired timers with a principled learned policy. The policy state is augmented with a predictive hyperbolic spread signal (a "curvature signal" shorthand) derived from hyperbolic geometry: the mean pairwise Poincare distance among n sampled futures embedded in the Poincare ball. High spread indicates a branching, uncertain future and drives the agent to act sooner; low spread signals predictability and permits longer rest intervals. We further propose an interval-aware reward that explicitly penalises inefficiency relative to the chosen wait time, correcting a systematic credit-assignment failure of naive outcome-based rewards in timing problems. We additionally introduce a joint spatio-temporal embedding (ATCPG-ST) that concatenates independently normalised state and position projections in the Poincare ball; spatial trajectory divergence provides an independent timing signal unavailable to the state-only variant (ATCPG-SO). This extension raises mean hyperbolic spread (kappa) from 1.88 to 3.37 and yields a further 5.8 percent efficiency gain over the state-only baseline. Ablation experiments across five random seeds demonstrate that (i) learning is the dominant efficiency factor (54.8 percent over no-learning), (ii) hyperbolic spread provides significant complementary gain (26.2 percent over geometry-free control), (iii) the combined system achieves 22.8 percent efficiency over the fixed-interval baseline, and (iv) adding spatial position information to the spread embedding yields an additional 5.8 percent.

**arXiv ID:** 2603.22384
</details>

<details>
<summary><strong>Collaborative Evaluation of Deepfake Text with Deliberation-Enhancing Dialogue Systems</strong> - Jooyoung Lee, Xiaochen Zhu, Georgi Karadzhov, Tom Stafford, Andreas Vlachos, Dongwon Lee - [[pdf]](https://arxiv.org/pdf/2503.04945)</summary>

**Abstract:** The proliferation of generative models has presented significant challenges in distinguishing authentic human-authored content from deepfake content. Collaborative human efforts, augmented by AI tools, present a promising solution. In this study, we explore the potential of DeepFakeDeLiBot, a deliberation-enhancing chatbot, to support groups in detecting deepfake text. Our findings reveal that group-based problem-solving significantly improves the accuracy of identifying machine-generated paragraphs compared to individual efforts. While engagement with DeepFakeDeLiBot does not yield substantial performance gains overall, it enhances group dynamics by fostering greater participant engagement, consensus building, and the frequency and diversity of reasoning-based utterances. Additionally, participants with higher perceived effectiveness of group collaboration exhibited performance benefits from DeepFakeDeLiBot. These findings underscore the potential of deliberative chatbots in fostering interactive and productive group dynamics while ensuring accuracy in collaborative deepfake text detection. \textit{Dataset and source code used in this study will be made publicly available upon acceptance of the manuscript.

**arXiv ID:** 2503.04945
</details>

<details>
<summary><strong>Parallel OctoMapping: A Scalable Framework for Enhanced Path Planning in Autonomous Navigation</strong> - Yihui Mao, Tian Tan, Xuehui Shen, Warren E. Dixon, Rushikesh Kamalapurkar - [[pdf]](https://arxiv.org/pdf/2603.22508)</summary>

**Abstract:** Mapping is essential in robotics and autonomous systems because it provides the spatial foundation for path planning. Efficient mapping enables planning algorithms to generate reliable paths while ensuring safety and adapting in real time to complex environments. Fixed-resolution mapping methods often produce overly conservative obstacle representations that lead to suboptimal paths or planning failures in cluttered scenes. To address this issue, we introduce Parallel OctoMapping (POMP), an efficient OctoMap-based mapping technique that maximizes available free space and supports multi-threaded computation. To the best of our knowledge, POMP is the first method that, at a fixed occupancy-grid resolution, refines the representation of free space while preserving map fidelity and compatibility with existing search-based planners. It can therefore be integrated into existing planning pipelines, yielding higher pathfinding success rates and shorter path lengths, especially in cluttered environments, while substantially improving computational efficiency.

**arXiv ID:** 2603.22508
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (17 papers)</h2></summary>

<details>
<summary><strong>Separating Diagnosis from Control: Auditable Policy Adaptation in Agent-Based Simulations with LLM-Based Diagnostics</strong> - Shaoxin Zhong, Yuchen Su, Michael Witbrock - [[pdf]](https://arxiv.org/pdf/2603.22904)</summary>

**Abstract:** Mitigating elderly loneliness requires policy interventions that achieve both adaptability and auditability. Existing methods struggle to reconcile these objectives: traditional agent-based models suffer from static rigidity, while direct large language model (LLM) controllers lack essential traceability. This work proposes a three-layer framework that separates diagnosis from control to achieve both properties simultaneously. LLMs operate strictly as diagnostic instruments that assess population state and generate structured risk evaluations, while deterministic formulas with explicit bounds translate these assessments into traceable parameter updates. This separation ensures that every policy decision can be attributed to inspectable rules while maintaining adaptive response to emergent needs. We validate the framework through systematic ablation across five experimental conditions in elderly care simulation. Results demonstrate that explicit control rules outperform end-to-end black-box LLM approaches by 11.7\% while preserving full auditability, confirming that transparency need not compromise adaptive performance.

**arXiv ID:** 2603.22904
</details>

<details>
<summary><strong>PERMA: Benchmarking Personalized Memory Agents via Event-Driven Preference and Realistic Task Environments</strong> - Shuochen Liu, Junyi Zhu, Long Shu, Junda Lin, Yuhao Chen, Haotian Zhang, Chao Zhang, Derong Xu, Jia Li, Bo Tang, Zhiyu Li, Feiyu Xiong, Enhong Chen, Tong Xu - [[pdf]](https://arxiv.org/pdf/2603.23231)</summary>

**Abstract:** Empowering large language models with long-term memory is crucial for building agents that adapt to users' evolving needs. However, prior evaluations typically interleave preference-related dialogues with irrelevant conversations, reducing the task to needle-in-a-haystack retrieval while ignoring relationships between events that drive the evolution of user preferences. Such settings overlook a fundamental characteristic of real-world personalization: preferences emerge gradually and accumulate across interactions within noisy contexts. To bridge this gap, we introduce PERMA, a benchmark designed to evaluate persona consistency over time beyond static preference recall. Additionally, we incorporate (1) text variability and (2) linguistic alignment to simulate erratic user inputs and individual idiolects in real-world data. PERMA consists of temporally ordered interaction events spanning multiple sessions and domains, with preference-related queries inserted over time. We design both multiple-choice and interactive tasks to probe the model's understanding of persona along the interaction timeline. Experiments demonstrate that by linking related interactions, advanced memory systems can extract more precise preferences and reduce token consumption, outperforming traditional semantic retrieval of raw dialogues. Nevertheless, they still struggle to maintain a coherent persona across temporal depth and cross-domain interference, highlighting the need for more robust personalized memory management in agents. Our code and data are open-sourced at this https URL.

**arXiv ID:** 2603.23231
</details>

<details>
<summary><strong>MemCollab: Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation</strong> - Yurui Chang, Yiran Wu, Qingyun Wu, Lu Lin - [[pdf]](https://arxiv.org/pdf/2603.23234)</summary>

**Abstract:** Large language model (LLM)-based agents rely on memory mechanisms to reuse knowledge from past problem-solving experiences. Existing approaches typically construct memory in a per-agent manner, tightly coupling stored knowledge to a single model's reasoning style. In modern deployments with heterogeneous agents, a natural question arises: can a single memory system be shared across different models? We found that naively transferring memory between agents often degrades performance, as such memory entangles task-relevant knowledge with agent-specific biases. To address this challenge, we propose MemCollab, a collaborative memory framework that constructs agent-agnostic memory by contrasting reasoning trajectories generated by different agents on the same task. This contrastive process distills abstract reasoning constraints that capture shared task-level invariants while suppressing agent-specific artifacts. We further introduce a task-aware retrieval mechanism that conditions memory access on task category, ensuring that only relevant constraints are used at inference time. Experiments on mathematical reasoning and code generation benchmarks demonstrate that MemCollab consistently improves both accuracy and inference-time efficiency across diverse agents, including cross-modal-family settings. Our results show that the collaboratively constructed memory can function as a shared reasoning resource for diverse LLM-based agents.

**arXiv ID:** 2603.23234
</details>

<details>
<summary><strong>Beyond Preset Identities: How Agents Form Stances and Boundaries in Generative Societies</strong> - Hanzhong Zhang, Siyang Song, Jindong Wang - [[pdf]](https://arxiv.org/pdf/2603.23406)</summary>

**Abstract:** While large language models simulate social behaviors, their capacity for stable stance formation and identity negotiation during complex interventions remains unclear. To overcome the limitations of static evaluations, this paper proposes a novel mixed-methods framework combining computational virtual ethnography with quantitative socio-cognitive profiling. By embedding human researchers into generative multiagent communities, controlled discursive interventions are conducted to trace the evolution of collective cognition. To rigorously measure how agents internalize and react to these specific interventions, this paper formalizes three new metrics: Innate Value Bias (IVB), Persuasion Sensitivity, and Trust-Action Decoupling (TAD). Across multiple representative models, agents exhibit endogenous stances that override preset identities, consistently demonstrating an innate progressive bias (IVB > 0). When aligned with these stances, rational persuasion successfully shifts 90% of neutral agents while maintaining high trust. In contrast, conflicting emotional provocations induce a paradoxical 40.0% TAD rate in advanced models, which hypocritically alter stances despite reporting low trust. Smaller models contrastingly maintain a 0% TAD rate, strictly requiring trust for behavioral shifts. Furthermore, guided by shared stances, agents use language interactions to actively dismantle assigned power hierarchies and reconstruct self organized community boundaries. These findings expose the fragility of static prompt engineering, providing a methodological and quantitative foundation for dynamic alignment in human-agent hybrid societies. The official code is available at: this https URL

**arXiv ID:** 2603.23406
</details>

<details>
<summary><strong>Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos</strong> - Shoubin Yu, Lei Shu, Antoine Yang, Yao Fu, Srinivas Sunkara, Maria Wang, Jindong Chen, Mohit Bansal, Boqing Gong - [[pdf]](https://arxiv.org/pdf/2603.22529)</summary>

**Abstract:** Multimodal AI agents are increasingly automating complex real-world workflows that involve online web execution. However, current web-agent benchmarks suffer from a critical limitation: they focus entirely on web-based interaction and perception, lacking grounding in the user's real-world physical surroundings. This limitation prevents evaluation in crucial scenarios, such as when an agent must use egocentric visual perception (e.g., via AR glasses) to recognize an object in the user's surroundings and then complete a related task online. To address this gap, we introduce Ego2Web, the first benchmark designed to bridge egocentric video perception and web agent execution. Ego2Web pairs real-world first-person video recordings with web tasks that require visual understanding, web task planning, and interaction in an online environment for successful completion. We utilize an automatic data-generation pipeline combined with human verification and refinement to curate well-constructed, high-quality video-task pairs across diverse web task types, including e-commerce, media retrieval, knowledge lookup, etc. To facilitate accurate and scalable evaluation for our benchmark, we also develop a novel LLM-as-a-Judge automatic evaluation method, Ego2WebJudge, which achieves approximately 84% agreement with human judgment, substantially higher than existing evaluation methods. Experiments with diverse SoTA agents on our Ego2Web show that their performance is weak, with substantial headroom across all task categories. We also conduct a comprehensive ablation study on task design, highlighting the necessity of accurate video understanding in the proposed task and the limitations of current agents. We hope Ego2Web can be a critical new resource for developing truly capable AI assistants that can seamlessly see, understand, and act across the physical and digital worlds.

**arXiv ID:** 2603.22529
</details>

<details>
<summary><strong>STRIATUM-CTF: A Protocol-Driven Agentic Framework for General-Purpose CTF Solving</strong> - James Hugglestone, Samuel Jacob Chacko, Dawson Stoller, Ryan Schmidt, Xiuwen Liu - [[pdf]](https://arxiv.org/pdf/2603.22577)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated potential in code generation, yet they struggle with the multi-step, stateful reasoning required for offensive cybersecurity operations. Existing research often relies on static benchmarks that fail to capture the dynamic nature of real-world vulnerabilities. In this work, we introduce STRIATUM-CTF (A Search-based Test-time Reasoning Inference Agent for Tactical Utility Maximization in Cybersecurity), a modular agentic framework built upon the Model Context Protocol (MCP). By standardizing tool interfaces for system introspection, decompilation, and runtime debugging, STRIATUM-CTF enables the agent to maintain a coherent context window across extended exploit trajectories.
We validate this approach not merely on synthetic datasets, but in a live competitive environment. Our system participated in a university-hosted Capture-the-Flag (CTF) competition in late 2025, where it operated autonomously to identify and exploit vulnerabilities in real-time. STRIATUM-CTF secured First Place, outperforming 21 human teams and demonstrating strong adaptability in a dynamic problem-solving setting. We analyze the agent's decision-making logs to show how MCP-based tool abstraction significantly reduces hallucination compared to naive prompting strategies. These results suggest that standardized context protocols are a critical path toward robust autonomous cyber-reasoning systems.

**arXiv ID:** 2603.22577
</details>

<details>
<summary><strong>PhotoAgent: A Robotic Photographer with Spatial and Aesthetic Understanding</strong> - Lirong Che, Zhenfeng Gan, Yanbo Chen, Junbo Tan, Xueqian Wang - [[pdf]](https://arxiv.org/pdf/2603.22796)</summary>

**Abstract:** Embodied agents for creative tasks like photography must bridge the semantic gap between high-level language commands and geometric control. We introduce PhotoAgent, an agent that achieves this by integrating Large Multimodal Models (LMMs) reasoning with a novel control paradigm. PhotoAgent first translates subjective aesthetic goals into solvable geometric constraints via LMM-driven, chain-of-thought (CoT) reasoning, allowing an analytical solver to compute a high-quality initial viewpoint. This initial pose is then iteratively refined through visual reflection within a photorealistic internal world model built with 3D Gaussian Splatting (3DGS). This ``mental simulation'' replaces costly and slow physical trial-and-error, enabling rapid convergence to aesthetically superior results. Evaluations confirm that PhotoAgent excels in spatial reasoning and achieves superior final image quality.

**arXiv ID:** 2603.22796
</details>

<details>
<summary><strong>AgentRAE: Remote Action Execution through Notification-based Visual Backdoors against Screenshots-based Mobile GUI Agents</strong> - Yutao Luo, Haotian Zhu, Shuchao Pang, Zhigang Lu, Tian Dong, Yongbin Zhou, Minhui Xue - [[pdf]](https://arxiv.org/pdf/2603.23007)</summary>

**Abstract:** The rapid adoption of mobile graphical user interface (GUI) agents, which autonomously control applications and operating systems (OS), exposes new system-level attack surfaces. Existing backdoors against web GUI agents and general GenAI models rely on environmental injection or deceptive pop-ups to mislead the agent operation. However, these techniques do not work on screenshots-based mobile GUI agents due to the challenges of restricted trigger design spaces, OS background interference, and conflicts in multiple trigger-action mappings. We propose AgentRAE, a novel backdoor attack capable of inducing Remote Action Execution in mobile GUI agents using visually natural triggers (e.g., benign app icons in notifications). To address the underfitting caused by natural triggers and achieve accurate multi-target action redirection, we design a novel two-stage pipeline that first enhances the agent's sensitivity to subtle iconographic differences via contrastive learning, and then associates each trigger with a specific mobile GUI agent action through a backdoor post-training. Our extensive evaluation reveals that the proposed backdoor preserves clean performance with an attack success rate of over 90% across ten mobile operations. Furthermore, it is hard to visibly detect the benign-looking triggers and circumvents eight representative state-of-the-art defenses. These results expose an overlooked backdoor vector in mobile GUI agents, underscoring the need for defenses that scrutinize notification-conditioned behaviors and internal agent representations.

**arXiv ID:** 2603.23007
</details>

<details>
<summary><strong>Code Review Agent Benchmark</strong> - Yuntong Zhang, Zhiyuan Pan, Imam Nur Bani Yusuf, Haifeng Ruan, Ridwan Shariffdeen, Abhik Roychoudhury - [[pdf]](https://arxiv.org/pdf/2603.23448)</summary>

**Abstract:** Software engineering agents have shown significant promise in writing code. As AI agents permeate code writing, and generate huge volumes of code automatically -- the matter of code quality comes front and centre. As the automatically generated code gets integrated into huge code-bases -- the issue of code review and broadly quality assurance becomes important. In this paper, we take a fresh look at the problem and curate a code review dataset for AI agents to work with. Our dataset called c-CRAB (pronounced see-crab) can evaluate agents for code review tasks. Specifically given a pull-request (which could be coming from code generation agents or humans), if a code review agent produces a review, our evaluation framework can asses the reviewing capability of the code review agents. Our evaluation framework is used to evaluate the state of the art today -- the open-source PR-agent, as well as commercial code review agents from Devin, Claude Code, and Codex.
Our c-CRAB dataset is systematically constructed from human reviews -- given a human review of a pull request instance we generate corresponding tests to evaluate the code review agent generated reviews. Such a benchmark construction gives us several insights. Firstly, the existing review agents taken together can solve only around 40% of the c-CRAB tasks, indicating the potential to close this gap by future research. Secondly, we observe that the agent reviews often consider different aspects from the human reviews -- indicating the potential for human-agent collaboration for code review that could be deployed in future software teams. Last but not the least, the agent generated tests from our data-set act as a held out test-suite and hence quality gate for agent generated reviews. What this will mean for future collaboration of code generation agents, test generation agents and code review agents -- remains to be investigated.

**arXiv ID:** 2603.23448
</details>

<details>
<summary><strong>Towards Self-Evolving Benchmarks: Synthesizing Agent Trajectories via Test-Time Exploration under Validate-by-Reproduce Paradigm</strong> - Dadi Guo, Tianyi Zhou, Dongrui Liu, Chen Qian, Qihan Ren, Shuai Shao, Zhiyuan Fan, Yi R. Fung, Kun Wang, Linfeng Zhang, Jing Shao - [[pdf]](https://arxiv.org/pdf/2510.00415)</summary>

**Abstract:** Recent advances in large language models (LLMs) and agent system designs have empowered agents with unprecedented levels of capability. However, existing agent benchmarks are showing a trend of rapid ceiling-hitting by newly developed agents, making it difficult to meet the demands for evaluating agent abilities. To address this problem, we propose the Trajectory-based Validated-by-Reproducing Agent-benchmark Complexity Evolution (TRACE) framework. This framework takes an original task from an existing benchmark and encourages agents to freely explore and evolve it into a new task with higher difficulty while recording validatable agent trajectories. The framework proceeds in three stages: (1) evolutionary proposal mining, which provides task evolution proposals through preliminary exploration and divergent thinking; (2) problem formation and free exploration, where proposals are conceptualized into feasible problem candidates and the agents then explore them freely while recording their execution trajectories; and (3) multi-level validation, which ensures that the evolved tasks are accompanied by validatable and reproducible trajectories. Experiments on the GAIA benchmark demonstrate that the TRACE framework consistently enhances task complexity while improving the reliability of correctness through validatable execution trajectories. In addition, our framework can successfully adapt to and improve reasoning datasets represented by AIME-2024. This work marks a paradigm shift from static, manually curated benchmarks to dynamic, self-evolving evaluation systems, providing a sustainable and challenging runway for agent development

**arXiv ID:** 2510.00415
</details>

<details>
<summary><strong>BuilderBench: The Building Blocks of Intelligent Agents</strong> - Raj Ghugare, Roger Creus Castanyer, Catherine Ji, Kathryn Wantlin, Jin Schofield, Karthik Narasimhan, Benjamin Eysenbach - [[pdf]](https://arxiv.org/pdf/2510.06288)</summary>

**Abstract:** Today's AI models learn primarily through mimicry and refining, so it is not surprising that they struggle to solve problems beyond the limits set by existing data. To solve novel problems, agents should acquire skills for exploring and learning through experience. Finding a scalable learning mechanism for developing agents that learn through interaction remains a major open problem. In this work, we introduce BuilderBench, a benchmark to accelerate research into agent pre-training that centers open-ended exploration. BuilderBench requires agents to learn how to build any structure using blocks. BuilderBench is equipped with $(1)$ a hardware accelerated simulator of a robotic agent interacting with various physical blocks, and $(2)$ a task-suite with over 42 diverse target structures that are carefully curated to test an understanding of physics, mathematics, and long-horizon planning. During training, agents have to explore and learn general principles about the environment without any external supervision. During evaluation, agents have to build the unseen target structures from the task suite. Solving these tasks requires a sort of \emph{embodied reasoning} that is not reflected in words but rather in actions, experimenting with different strategies and piecing them together. Our experiments show that many of these tasks challenge the current iteration of algorithms. Hence, we also provide a ``training wheels'' protocol, in which agents are trained and evaluated to build a single target structure from the task suite. Finally, we provide single-file implementations of six different algorithms as a reference point for researchers.

**arXiv ID:** 2510.06288
</details>

<details>
<summary><strong>CyberGym: Evaluating AI Agents' Real-World Cybersecurity Capabilities at Scale</strong> - Zhun Wang, Tianneng Shi, Jingxuan He, Matthew Cai, Jialin Zhang, Dawn Song - [[pdf]](https://arxiv.org/pdf/2506.02548)</summary>

**Abstract:** AI agents have significant potential to reshape cybersecurity, making a thorough assessment of their capabilities critical. However, existing evaluations fall short, because they are based on small-scale benchmarks and only measure static outcomes, failing to capture the full, dynamic range of real-world security challenges. To address these limitations, we introduce CyberGym, a large-scale benchmark featuring 1,507 real-world vulnerabilities across 188 software projects. Adjustable to different vulnerability analysis settings, CyberGym primarily tasks agents with generating a proof-of-concept test that reproduces a vulnerability, given only its text description and the corresponding codebase. Our extensive evaluation highlights that CyberGym effectively differentiates agents' and models' cybersecurity capabilities. Even the top-performing combinations only achieve a ~20% success rate, demonstrating the overall difficulty of CyberGym. Beyond static benchmarking, we show that CyberGym leads to the discovery of 34 zero-day vulnerabilities and 18 historically incomplete patches. These results underscore that CyberGym is not only a robust benchmark for measuring AI's progress in cybersecurity but also a platform for creating direct, real-world security impact.

**arXiv ID:** 2506.02548
</details>

<details>
<summary><strong>From Conflict to Consensus: Boosting Medical Reasoning via Multi-Round Agentic RAG</strong> - Wenhao Wu, Zhentao Tang, Yafu Li, Shixiong Kai, Mingxuan Yuan, Chunlin Chen, Zhi Wang - [[pdf]](https://arxiv.org/pdf/2603.03292)</summary>

**Abstract:** Large Language Models (LLMs) exhibit high reasoning capacity in medical question-answering, but their tendency to produce hallucinations and outdated knowledge poses critical risks in healthcare fields. While Retrieval-Augmented Generation (RAG) mitigates these issues, existing methods rely on noisy token-level signals and lack the multi-round refinement required for complex reasoning. In the paper, we propose MA-RAG (Multi-Round Agentic RAG), a framework that facilitates test-time scaling for complex medical reasoning by iteratively evolving both external evidence and internal reasoning history within an agentic refinement loop. At each round, the agent transforms semantic conflict among candidate responses into actionable queries to retrieve external evidence, while optimizing history reasoning traces to mitigate long-context degradation. MA-RAG extends the self-consistency principle by leveraging the lack of consistency as a proactive signal for multi-round agentic reasoning and retrieval, and mirrors a boosting mechanism that iteratively minimizes the residual error toward a stable, high-fidelity medical consensus. Extensive evaluations across 7 medical Q&A benchmarks show that MA-RAG consistently surpasses competitive inference-time scaling and RAG baselines, delivering substantial +6.8 points on average accuracy over the backbone model. Our code is available at this https URL.

**arXiv ID:** 2603.03292
</details>

<details>
<summary><strong>Polaris: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair</strong> - Aditya Kakade, Vivek Srivastava, Shirish Karande - [[pdf]](https://arxiv.org/pdf/2603.23129)</summary>

**Abstract:** Gödel agent realize recursive self-improvement: an agent inspects its own policy and traces and then modifies that policy in a tested loop. We introduce Polaris, a Gödel agent for compact models that performs policy repair via experience abstraction, turning failures into policy updates through a structured cycle of analysis, strategy formation, abstraction, and minimal code pat ch repair with conservative checks. Unlike response level self correction or parameter tuning, Polaris makes policy level changes with small, auditable patches that persist in the policy and are reused on unseen instances within each benchmark. As part of the loop, the agent engages in meta reasoning: it explains its errors, proposes concrete revisions to its own policy, and then updates the policy. To enable cumulative policy refinement, we introduce experience abstraction, which distills failures into compact, reusable strategies that transfer to unseen instances. On MGSM, DROP, GPQA, and LitBench (covering arithmetic reasoning, compositional inference, graduate-level problem solving, and creative writing evaluation), a 7-billion-parameter model equipped with Polaris achieves consistent gains over the base policy and competitive baselines.

**arXiv ID:** 2603.23129
</details>

<details>
<summary><strong>VL-KnG: Persistent Spatiotemporal Knowledge Graphs from Egocentric Video for Embodied Scene Understanding</strong> - Mohamad Al Mdfaa, Svetlana Lukina, Timur Akhtyamov, Arthur Nigmatzyanov, Dmitrii Nalberskii, Sergey Zagoruyko, Gonzalo Ferrer - [[pdf]](https://arxiv.org/pdf/2510.01483)</summary>

**Abstract:** Vision-language models (VLMs) demonstrate strong image-level scene understanding but often lack persistent memory, explicit spatial representations, and computational efficiency when reasoning over long video sequences. We present VL-KnG, a training-free framework that constructs spatiotemporal knowledge graphs from monocular video, bridging fine-grained scene graphs and global topological graphs without 3D reconstruction. VL-KnG processes video in chunks, maintains persistent object identity via LLM-based Spatiotemporal Object Association (STOA), and answers queries via Graph-Enhanced Retrieval (GER), a hybrid of GraphRAG subgraph retrieval and SigLIP2 visual grounding. Once built, the knowledge graph eliminates the need to re-process video at query time, enabling constant-time inference regardless of video length. Evaluation across three benchmarks, OpenEQA, NaVQA, and WalkieKnowledge (our newly introduced benchmark), shows that VL-KnG matches or surpasses frontier VLMs on embodied scene understanding tasks at significantly lower query latency, with explainable, graph-grounded reasoning. Real-world robot deployment confirms practical applicability with constant-time scaling.

**arXiv ID:** 2510.01483
</details>

<details>
<summary><strong>nuScenes Revisited: Progress and Challenges in Autonomous Driving</strong> - Whye Kit Fong, Venice Erin Liong, Kok Seang Tan, Holger Caesar - [[pdf]](https://arxiv.org/pdf/2512.02448)</summary>

**Abstract:** Autonomous Vehicles (AV) and Advanced Driver Assistance Systems (ADAS) have been revolutionized by Deep Learning. As a data-driven approach, Deep Learning relies on vast amounts of driving data, typically labeled in great detail. As a result, datasets, alongside hardware and algorithms, are foundational building blocks for the development of AVs. In this work we revisit one of the most widely used autonomous driving datasets: the nuScenes dataset. nuScenes exemplifies key trends in AV development, being the first dataset to include radar data, to feature diverse urban driving scenes from two continents, and to be collected using a fully autonomous vehicle operating on public roads, while also promoting multi-modal sensor fusion, standardized benchmarks, and a broad range of tasks including perception, localization & mapping, prediction and planning. We provide an unprecedented look into the creation of nuScenes, as well as its extensions nuImages and Panoptic nuScenes, summarizing many technical details that have hitherto not been revealed in academic publications. Furthermore, we trace how the influence of nuScenes impacted a large number of other datasets that were released later and how it defined numerous standards that are used by the community to this day. Finally, we present an overview of both official and unofficial tasks using the nuScenes dataset and review major methodological developments, thereby offering a comprehensive survey of the autonomous driving literature, with a particular focus on nuScenes.

**arXiv ID:** 2512.02448
</details>

<details>
<summary><strong>PoseDriver: A Unified Approach to Multi-Category Skeleton Detection for Autonomous Driving</strong> - Yasamin Borhani, Taylor Mordan, Yihan Wang, Reyhaneh Hosseininejad, Javad Khoramdel, Alexandre Alahi - [[pdf]](https://arxiv.org/pdf/2603.23215)</summary>

**Abstract:** Object skeletons offer a concise representation of structural information, capturing essential aspects of posture and orientation that are crucial for autonomous driving applications. However, a unified architecture that simultaneously handles multiple instances and categories using only the input image remains elusive. In this paper, we introduce PoseDriver, a unified framework for bottom-up multi-category skeleton detection tailored to common objects in driving scenarios. We model each category as a distinct task to systematically address the challenges of multi-task learning. Specifically, we propose a novel approach for lane detection based on skeleton representations, achieving state-of-the-art performance on the OpenLane dataset. Moreover, we present a new dataset for bicycle skeleton detection and assess the transferability of our framework to novel categories. Experimental results validate the effectiveness of the proposed approach.

**arXiv ID:** 2603.23215
</details>

</details>

<details open>
<summary><h2>LLM Agents (10 papers)</h2></summary>

<details>
<summary><strong>From Static Templates to Dynamic Runtime Graphs: A Survey of Workflow Optimization for LLM Agents</strong> - Ling Yue, Kushal Raj Bhandari, Ching-Yun Ko, Dhaval Patel, Shuxin Lin, Nianjun Zhou, Jianxi Gao, Pin-Yu Chen, Shaowu Pan - [[pdf]](https://arxiv.org/pdf/2603.22386)</summary>

**Abstract:** Large language model (LLM)-based systems are becoming increasingly popular for solving tasks by constructing executable workflows that interleave LLM calls, information retrieval, tool use, code execution, memory updates, and verification. This survey reviews recent methods for designing and optimizing such workflows, which we treat as agentic computation graphs (ACGs). We organize the literature based on when workflow structure is determined, where structure refers to which components or agents are present, how they depend on each other, and how information flows between them. This lens distinguishes static methods, which fix a reusable workflow scaffold before deployment, from dynamic methods, which select, generate, or revise the workflow for a particular run before or during execution. We further organize prior work along three dimensions: when structure is determined, what part of the workflow is optimized, and which evaluation signals guide optimization (e.g., task metrics, verifier signals, preferences, or trace-derived feedback). We also distinguish reusable workflow templates, run-specific realized graphs, and execution traces, separating reusable design choices from the structures actually deployed in a given run and from realized runtime behavior. Finally, we outline a structure-aware evaluation perspective that complements downstream task metrics with graph-level properties, execution cost, robustness, and structural variation across inputs. Our goal is to provide a clear vocabulary, a unified framework for positioning new methods, a more comparable view of existing body of literature, and a more reproducible evaluation standard for future work in workflow optimizations for LLM agents.

**arXiv ID:** 2603.22386
</details>

<details>
<summary><strong>Can LLM Agents Generate Real-World Evidence? Evaluating Observational Studies in Medical Databases</strong> - Dubai Li, Yuxiang He, Yan Hu, Yu Tian, Jingsong Li - [[pdf]](https://arxiv.org/pdf/2603.22767)</summary>

**Abstract:** Observational studies can yield clinically actionable evidence at scale, but executing them on real-world databases is open-ended and requires coherent decisions across cohort construction, analysis, and reporting. Prior evaluations of LLM agents emphasize isolated steps or single answers, missing the integrity and internal structure of the resulting evidence bundle. To address this gap, we introduce RWE-bench, a benchmark grounded in MIMIC-IV and derived from peer-reviewed observational studies. Each task provides the corresponding study protocol as the reference standard, requiring agents to execute experiments in a real database and iteratively generate tree-structured evidence bundles. We evaluate six LLMs (three open-source, three closed-source) under three agent scaffolds using both question-level correctness and end-to-end task metrics. Across 162 tasks, task success is low: the best agent reaches 39.9%, and the best open-source model reaches 30.4%. Agent scaffolds also matter substantially, causing over 30% variation in performance metrics. Furthermore, we implement an automated cohort evaluation method to rapidly localize errors and identify agent failure modes. Overall, the results highlight persistent limitations in agents' ability to produce end-to-end evidence bundles, and efficient validation remains an important direction for future work. Code and data are available at this https URL.

**arXiv ID:** 2603.22767
</details>

<details>
<summary><strong>T-MAP: Red-Teaming LLM Agents with Trajectory-aware Evolutionary Search</strong> - Hyomin Lee, Sangwoo Park, Yumin Choi, Sohyun An, Seanie Lee, Sung Ju Hwang - [[pdf]](https://arxiv.org/pdf/2603.22341)</summary>

**Abstract:** While prior red-teaming efforts have focused on eliciting harmful text outputs from large language models (LLMs), such approaches fail to capture agent-specific vulnerabilities that emerge through multi-step tool execution, particularly in rapidly growing ecosystems such as the Model Context Protocol (MCP). To address this gap, we propose a trajectory-aware evolutionary search method, T-MAP, which leverages execution trajectories to guide the discovery of adversarial prompts. Our approach enables the automatic generation of attacks that not only bypass safety guardrails but also reliably realize harmful objectives through actual tool interactions. Empirical evaluations across diverse MCP environments demonstrate that T-MAP substantially outperforms baselines in attack realization rate (ARR) and remains effective against frontier models, including GPT-5.2, Gemini-3-Pro, Qwen3.5, and GLM-5, thereby revealing previously underexplored vulnerabilities in autonomous LLM agents.

**arXiv ID:** 2603.22341
</details>

<details>
<summary><strong>Reasoner-Executor-Synthesizer: Scalable Agentic Architecture with Static O(1) Context Window</strong> - Ivan Dobrovolskyi - [[pdf]](https://arxiv.org/pdf/2603.22367)</summary>

**Abstract:** Large Language Models (LLMs) deployed as autonomous agents commonly use Retrieval-Augmented Generation (RAG), feeding retrieved documents into the context window, which creates two problems: the risk of hallucination grows with context length, and token cost scales linearly with dataset size. We propose the Reasoner-Executor-Synthesizer (RES) architecture, a three-layer design that strictly separates intent parsing (Reasoner), deterministic data retrieval and aggregation (Executor), and narrative generation (Synthesizer). The Executor uses zero LLM tokens and passes only fixed-size statistical summaries to the Synthesizer. We formally prove that RES achieves O(1) token complexity with respect to dataset size, and validate this on ScholarSearch, a scholarly research assistant backed by the Crossref API (130M+ articles). Across 100 benchmark runs, RES achieves a mean token cost of 1,574 tokens regardless of whether the dataset contains 42,000 or 16.3 million articles. The architecture eliminates data hallucination by construction: the LLM never sees raw records. KEYWORDS LLM agents; agentic architecture; hallucination elimination; token optimization; context window; retrieval-augmented generation; deterministic execution; scholarly metadata; Crossref API; O(1) complexity.

**arXiv ID:** 2603.22367
</details>

<details>
<summary><strong>AI Co-Scientist for Ranking: Discovering Novel Search Ranking Models alongside LLM-based AI Agents with Cloud Computing Access</strong> - Liwei Wu, Cho-Jui Hsieh - [[pdf]](https://arxiv.org/pdf/2603.22376)</summary>

**Abstract:** Recent advances in AI agents for software engineering and scientific discovery have demonstrated remarkable capabilities, yet their application to developing novel ranking models in commercial search engines remains unexplored. In this paper, we present an AI Co-Scientist framework that automates the full search ranking research pipeline: from idea generation to code implementation and GPU training job scheduling with expert in the loop. Our approach strategically employs single-LLM agents for routine tasks while leveraging multi-LLM consensus agents (GPT 5.2, Gemini Pro 3, and Claude Opus 4.5) for challenging phases such as results analysis and idea generation. To our knowledge, this is the first study in the ranking community to utilize an AI Co-Scientist framework for algorithmic research. We demonstrate that this framework discovered a novel technique for handling sequence features, with all model enhancements produced automatically, yielding substantial offline performance improvements. Our findings suggest that AI systems can discover ranking architectures comparable to those developed by human experts while significantly reducing routine research workloads.

**arXiv ID:** 2603.22376
</details>

<details>
<summary><strong>Agent Audit: A Security Analysis System for LLM Agent Applications</strong> - Haiyue Zhang, Yi Nian, Yue Zhao - [[pdf]](https://arxiv.org/pdf/2603.22853)</summary>

**Abstract:** What should a developer inspect before deploying an LLM agent: the model, the tool code, the deployment configuration, or all three? In practice, many security failures in agent systems arise not from model weights alone, but from the surrounding software stack: tool functions that pass untrusted inputs to dangerous operations, exposed credentials in deployment artifacts, and over-privileged Model Context Protocol (MCP) configurations.
We present Agent Audit, a security analysis system for LLM agent applications. Agent Audit analyzes Python agent code and deployment artifacts through an agent-aware pipeline that combines dataflow analysis, credential detection, structured configuration parsing, and privilege-risk checks. The system reports findings in terminal, JSON, and SARIF formats, enabling direct integration with local development workflows and CI/CD pipelines. On a benchmark of 22 samples with 42 annotated vulnerabilities, Agent Audit detects 40 vulnerabilities with 6 false positives, substantially improving recall over common SAST baselines while maintaining sub-second scan times. Agent Audit is open source and installable via pip, making security auditing accessible for agent systems.
In the live demonstration, attendees scan vulnerable agent repositories and observe how Agent Audit identifies security risks in tool functions, prompts, and more. Findings are linked to source locations and configuration paths, and can be exported into VS Code and GitHub Code Scanning for interactive inspection.

**arXiv ID:** 2603.22853
</details>

<details>
<summary><strong>Agent-Sentry: Bounding LLM Agents via Execution Provenance</strong> - Rohan Sequeira, Stavros Damianakis, Umar Iqbal, Konstantinos Psounis - [[pdf]](https://arxiv.org/pdf/2603.22868)</summary>

**Abstract:** Agentic computing systems, which autonomously spawn new functionalities based on natural language instructions, are becoming increasingly prevalent. While immensely capable, these systems raise serious security, privacy, and safety concerns. Fundamentally, the full set of functionalities offered by these systems, combined with their probabilistic execution flows, is not known beforehand. Given this lack of characterization, it is non-trivial to validate whether a system has successfully carried out the user's intended task or instead executed irrelevant actions, potentially as a consequence of compromise. In this paper, we propose Agent-Sentry, a framework that attempts to bound agentic systems to address this problem. Our key insight is that agentic systems are designed for specific use cases and therefore need not expose unbounded or unspecified functionalities. Once bounded, these systems become easier to scrutinize. Agent-Sentry operationalizes this insight by uncovering frequent functionalities offered by an agentic system, along with their execution traces, to construct behavioral bounds. It then learns a policy from these traces and blocks tool calls that deviate from learned behaviors or that misalign with user intent. Our evaluation shows that Agent-Sentry helps prevent over 90\% of attacks that attempt to trigger out-of-bounds executions, while preserving up to 98\% of system utility.

**arXiv ID:** 2603.22868
</details>

<details>
<summary><strong>Rethinking the Role of Entropy in Optimizing Tool-Use Behaviors for Large Language Model Agents</strong> - Zeping Li, Hongru Wang, Yiwen Zhao, Guanhua Chen, Yixia Li, Keyang Chen, Yixin Cao, Guangnan Ye, Hongfeng Chai, Zhenfei Yin - [[pdf]](https://arxiv.org/pdf/2602.02050)</summary>

**Abstract:** Tool-using agents based on Large Language Models (LLMs) excel in tasks such as mathematical reasoning and multi-hop question answering. However, in long trajectories, agents often trigger excessive and low-quality tool calls, increasing latency and degrading inference performance, making managing tool-use behavior challenging. In this work, we conduct entropy-based pilot experiments and observe a strong positive correlation between entropy reduction and high-quality tool calls. Building on this finding, we propose using entropy reduction as a supervisory signal and design two reward strategies to address the differing needs of optimizing tool-use behavior. Sparse outcome rewards provide coarse, trajectory-level guidance to improve efficiency, while dense process rewards offer fine-grained supervision to enhance performance. Experiments across diverse domains show that both reward designs improve tool-use behavior: the former reduces tool calls by 72.07% compared to the average of baselines, while the latter improves performance by 22.27%. These results position entropy reduction as a key mechanism for enhancing tool-use behavior, enabling agents to be more adaptive in real-world applications.

**arXiv ID:** 2602.02050
</details>

<details>
<summary><strong>The Evolution of Tool Use in LLM Agents: From Single-Tool Call to Multi-Tool Orchestration</strong> - Haoyuan Xu, Chang Li, Xinyan Ma, Xianhao Ou, Zihan Zhang, Tao He, Xiangyu Liu, Zixiang Wang, Jiafeng Liang, Zheng Chu, Runxuan Liu, Rongchuan Mu, Ming Liu, Bing Qin - [[pdf]](https://arxiv.org/pdf/2603.22862)</summary>

**Abstract:** Tool use enables large language models (LLMs) to access external information, invoke software systems, and act in digital environments beyond what can be solved from model parameters alone. Early research mainly studied whether a model could select and execute a correct single tool call. As agent systems evolve, however, the central problem has shifted from isolated invocation to multi-tool orchestration over long trajectories with intermediate state, execution feedback, changing environments, and practical constraints such as safety, cost, and verifiability. We comprehensively review recent progress in multi-tool LLM agents and analyzes the state of the art in this rapidly developing area. First, we unify task formulations and distinguish single-call tool use from long-horizon orchestration. Then, we organize the literature around six core dimensions: inference-time planning and execution, training and trajectory construction, safety and control, efficiency under resource constraints, capability completeness in open environments, and benchmark design and evaluation. We further summarize representative applications in software engineering, enterprise workflows, graphical user interfaces, and mobile systems. Finally, we discuss major challenges and outline future directions for building reliable, scalable, and verifiable multi-tool agents.

**arXiv ID:** 2603.22862
</details>

<details>
<summary><strong>SkillRouter: Retrieve-and-Rerank Skill Selection for LLM Agents at Scale</strong> - YanZhao Zheng, ZhenTao Zhang, Chao Ma, YuanQiang Yu, JiHuan Zhu, Baohua Dong, Hangcheng Zhu - [[pdf]](https://arxiv.org/pdf/2603.22455)</summary>

**Abstract:** As LLM agent ecosystems grow, the number of available skills (tools, plugins) has reached tens of thousands, making it infeasible to inject all skills into an agent's context. This creates a need for skill routing -- retrieving the most relevant skills from a large pool given a user task. The problem is compounded by pervasive functional overlap in community skill repositories, where many skills share similar names and purposes yet differ in implementation details. Despite its practical importance, skill routing remains under-explored. Current agent architectures adopt a progressive disclosure design -- exposing only skill names and descriptions to the agent while keeping the full implementation body hidden -- implicitly treating metadata as sufficient for selection. We challenge this assumption through a systematic empirical study on a benchmark of ~$80K skills and 75 expert-verified queries. Our key finding is that the skill body (full implementation text) is the decisive signal: removing it causes 29--44 percentage point degradation across all retrieval methods, and cross-encoder attention analysis reveals 91.7% of attention concentrating on the body field. Motivated by this finding, we propose SkillRouter, a two-stage retrieve-and-rerank pipeline totaling only 1.2B parameters (0.6B encoder + 0.6B reranker). SkillRouter achieves 74.0% top-1 routing accuracy and delivers the strongest average result among the compact and zero-shot baselines we evaluate, while remaining deployable on consumer hardware.

**arXiv ID:** 2603.22455
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (16 papers)</h2></summary>

<details>
<summary><strong>STEM Agent: A Self-Adapting, Tool-Enabled, Extensible Architecture for Multi-Protocol AI Agent Systems</strong> - Alfred Shen, Aaron Shen - [[pdf]](https://arxiv.org/pdf/2603.22359)</summary>

**Abstract:** Current AI agent frameworks commit early to a single interaction protocol, a fixed tool integration strategy, and static user models, limiting their deployment across diverse interaction paradigms. To address these constraints, we introduce STEM Agent (Self-adapting, Tool-enabled, Extensible, Multi-agent), a modular architecture inspired by biological pluripotency in which an undifferentiated agent core differentiates into specialized protocol handlers, tool bindings, and memory subsystems that compose into a fully functioning AI system. The framework unifies five interoperability protocols (A2A, AG-UI, A2UI, UCP, and AP2) behind a single gateway, introduces a Caller Profiler that continuously learns user preferences across more than twenty behavioral dimensions, externalizes all domain capabilities through the Model Context Protocol (MCP), and implements a biologically inspired skills acquisition system in which recurring interaction patterns crystallize into reusable agent skills through a maturation lifecycle analogous to cell differentiation. Complementing these capabilities, the memory system incorporates consolidation mechanisms, including episodic pruning, semantic deduplication, and pattern extraction, designed for sub-linear growth under sustained interaction. A comprehensive 413-test suite validates protocol handler behavior and component integration across all five architectural layers, completing in under three seconds.

**arXiv ID:** 2603.22359
</details>

<details>
<summary><strong>Benchmarking Multi-Agent LLM Architectures for Financial Document Processing: A Comparative Study of Orchestration Patterns, Cost-Accuracy Tradeoffs and Production Scaling Strategies</strong> - Siddhant Kulkarni, Yukta Kulkarni - [[pdf]](https://arxiv.org/pdf/2603.22651)</summary>

**Abstract:** The adoption of large language models (LLMs) for structured information extraction from financial documents has accelerated rapidly, yet production deployments face fundamental architectural decisions with limited empirical guidance. We present a systematic benchmark comparing four multi-agent orchestration architectures: sequential pipeline, parallel fan-out with merge, hierarchical supervisor-worker and reflexive self-correcting loop. These are evaluated across five frontier and open-weight LLMs on a corpus of 10,000 SEC filings (10-K, 10-Q and 8-K forms). Our evaluation spans 25 extraction field types covering governance structures, executive compensation and financial metrics, measured along five axes: field-level F1, document-level accuracy, end-to-end latency, cost per document and token efficiency. We find that reflexive architectures achieve the highest field-level F1 (0.943) but at 2.3x the cost of sequential baselines, while hierarchical architectures occupy the most favorable position on the cost-accuracy Pareto frontier (F1 0.921 at 1.4x cost). We further present ablation studies on semantic caching, model routing and adaptive retry strategies, demonstrating that hybrid configurations can recover 89\% of the reflexive architecture's accuracy gains at only 1.15x baseline cost. Our scaling analysis from 1K to 100K documents per day reveals non-obvious throughput-accuracy degradation curves that inform capacity planning. These findings provide actionable guidance for practitioners deploying multi-agent LLM systems in regulated financial environments.

**arXiv ID:** 2603.22651
</details>

<details>
<summary><strong>ABSTRAL: Automatic Design of Multi-Agent Systems Through Iterative Refinement and Topology Optimization</strong> - Weijia Song, Jiashu Yue, Zhe Pang - [[pdf]](https://arxiv.org/pdf/2603.22791)</summary>

**Abstract:** How should multi-agent systems be designed, and can that design knowledge be captured in a form that is inspectable, revisable, and transferable? We introduce ABSTRAL, a framework that treats MAS architecture as an evolving natural-language document, an artifact refined through contrastive trace analysis. Three findings emerge. First, we provide a precise measurement of the multi-agent coordination tax: under fixed turn budgets, ensembles achieve only 26% turn efficiency, with 66% of tasks exhausting the limit, yet still improve over single-agent baselines by discovering parallelizable task decompositions. Second, design knowledge encoded in documents transfers: topology reasoning and role templates learned on one domain provide a head start on new domains, with transferred seeds matching coldstart iteration 3 performance in a single iteration. Third, contrastive trace analysis discovers specialist roles absent from any initial design, a capability no prior system demonstrates. On SOPBench (134 bank tasks, deterministic oracle), ABSTRAL reaches 70% validation / 65.96% test pass rate with a GPT-4o backbone. We release the converged documents as inspectable design rationale.

**arXiv ID:** 2603.22791
</details>

<details>
<summary><strong>Empirical Comparison of Agent Communication Protocols for Task Orchestration</strong> - Ivan Dobrovolskyi - [[pdf]](https://arxiv.org/pdf/2603.22823)</summary>

**Abstract:** Context. Nowadays, artificial intelligence agent systems are transforming from single-tool interactions to complex multi-agent orchestrations. As a result, two competing communication protocols have emerged: a tool integration protocol that standardizes how agents invoke external tools, and an inter-agent delegation protocol that enables autonomous agents to discover and delegate tasks to one another. Despite widespread industry adoption by dozens of enterprise partners, no empirical comparison of these protocols exists in the literature. Objective. The goal of this work is to develop the first systematic benchmark comparing tool-integration-only, multi-agent delegation, and hybrid architectures across standardized queries at three complexity levels, and to quantify the trade-offs in response time, context window consumption, monetary cost, error recovery, and implementation complexity.

**arXiv ID:** 2603.22823
</details>

<details>
<summary><strong>CoMaTrack: Competitive Multi-Agent Game-Theoretic Tracking with Vision-Language-Action Models</strong> - Youzhi Liu, Li Gao, Liu Liu, Mingyang Lv, Yang Cai - [[pdf]](https://arxiv.org/pdf/2603.22846)</summary>

**Abstract:** Embodied Visual Tracking (EVT), a core dynamic task in embodied intelligence, requires an agent to precisely follow a language-specified target. Yet most existing methods rely on single-agent imitation learning, suffering from costly expert data and limited generalization due to static training environments. Inspired by competition-driven capability evolution, we propose CoMaTrack, a competitive game-theoretic multi-agent reinforcement learning framework that trains agents in a dynamic adversarial setting with competitive subtasks, yielding stronger adaptive planning and interference-resilient strategies. We further introduce CoMaTrack-Bench, the first benchmark for competitive EVT, featuring game scenarios between a tracker and adaptive opponents across diverse environments and instructions, enabling standardized robustness evaluation under active adversarial interactions. Experiments show that CoMaTrack achieves state-of-the-art results on both standard benchmarks and CoMaTrack-Bench. Notably, a 3B VLM trained with our framework surpasses previous single-agent imitation learning methods based on 7B models on the challenging EVT-Bench, achieving 92.1% in STT, 74.2% in DT, and 57.5% in AT. The benchmark code will be available at this https URL

**arXiv ID:** 2603.22846
</details>

<details>
<summary><strong>A Multimodal Framework for Human-Multi-Agent Interaction</strong> - Shaid Hasan, Breenice Lee, Sujan Sarker, Tariq Iqbal - [[pdf]](https://arxiv.org/pdf/2603.23271)</summary>

**Abstract:** Human-robot interaction is increasingly moving toward multi-robot, socially grounded environments. Existing systems struggle to integrate multimodal perception, embodied expression, and coordinated decision-making in a unified framework. This limits natural and scalable interaction in shared physical spaces. We address this gap by introducing a multimodal framework for human-multi-agent interaction in which each robot operates as an autonomous cognitive agent with integrated multimodal perception and Large Language Model (LLM)-driven planning grounded in embodiment. At the team level, a centralized coordination mechanism regulates turn-taking and agent participation to prevent overlapping speech and conflicting actions. Implemented on two humanoid robots, our framework enables coherent multi-agent interaction through interaction policies that combine speech, gesture, gaze, and locomotion. Representative interaction runs demonstrate coordinated multimodal reasoning across agents and grounded embodied responses. Future work will focus on larger-scale user studies and deeper exploration of socially grounded multi-agent interaction dynamics.

**arXiv ID:** 2603.23271
</details>

<details>
<summary><strong>Planning over MAPF Agent Dependencies via Multi-Dependency PIBT</strong> - Zixiang Jiang, Yulun Zhang, Rishi Veerapaneni, Jiaoyang Li - [[pdf]](https://arxiv.org/pdf/2603.23405)</summary>

**Abstract:** Modern Multi-Agent Path Finding (MAPF) algorithms must plan for hundreds to thousands of agents in congested environments within a second, requiring highly efficient algorithms. Priority Inheritance with Backtracking (PIBT) is a popular algorithm capable of effectively planning in such situations. However, PIBT is constrained by its rule-based planning procedure and lacks generality because it restricts its search to paths that conflict with at most one other agent. This limitation also applies to Enhanced PIBT (EPIBT), a recent extension of PIBT. In this paper, we describe a new perspective on solving MAPF by planning over agent dependencies. Taking inspiration from PIBT's priority inheritance logic, we define the concept of agent dependencies and propose Multi-Dependency PIBT (MD-PIBT) that searches over agent dependencies. MD-PIBT is a general framework where specific parameterizations can reproduce PIBT and EPIBT. At the same time, alternative configurations yield novel planning strategies that are not expressible by PIBT or EPIBT. Our experiments demonstrate that MD-PIBT effectively plans for as many as 10,000 homogeneous agents under various kinodynamic constraints, including pebble motion, rotation motion, and differential drive robots with speed and acceleration limits. We perform thorough evaluations on different variants of MAPF and find that MD-PIBT is particularly effective in MAPF with large agents.

**arXiv ID:** 2603.23405
</details>

<details>
<summary><strong>Biased Error Attribution in Multi-Agent Human-AI Systems Under Delayed Feedback</strong> - Teerthaa Parakh, Karen M. Feigh - [[pdf]](https://arxiv.org/pdf/2603.23419)</summary>

**Abstract:** Human decision-making is strongly influenced by cognitive biases, particularly under conditions of uncertainty and risk. While prior work has examined bias in single-step decisions with immediate outcomes and in human interaction with a single autonomous agent, comparatively little attention has been paid to decision-making under delayed outcomes involving multiple AI agents, where decisions at each step affect subsequent states. In this work, we study how delayed outcomes shape decision-making and responsibility attribution in a multi-agent human-AI task. Using a controlled game-based experiment, we analyze how participants adjust their behavior following positive and negative outcomes. We observe asymmetric responses to gains and losses, with stronger corrective adjustments after negative outcomes. Importantly, participants often fail to correctly identify the actions that caused failure and misattribute responsibility across AI agents, leading to systematic revisions of decisions that are weakly related to the underlying causes of poor performance. We refer to this phenomenon as a form of attribution bias, manifested as biased error attribution under delayed feedback. Our findings highlight how cognitive biases can be amplified in human-AI systems with delayed outcomes and multiple autonomous agents, underscoring the need for decision-support systems that better support causal understanding and learning over time.

**arXiv ID:** 2603.23419
</details>

<details>
<summary><strong>Cascade-Aware Multi-Agent Routing: Spatio-Temporal Sidecars and Geometry-Switching</strong> - Davide Di Gioia - [[pdf]](https://arxiv.org/pdf/2603.17112)</summary>

**Abstract:** Advanced AI reasoning systems route tasks through dynamic execution graphs of specialized agents. We identify a structural blind spot in this architecture: schedulers optimize load and fitness but lack a model of how failure propagates differently in tree-like versus cyclic graphs. In tree-like regimes, a single failure cascades exponentially; in dense cyclic regimes, it self-limits. A geometry-blind scheduler cannot distinguish these cases. We formalize this observability gap as an online geometry-control problem. We prove a cascade-sensitivity condition: failure spread is supercritical when per-edge propagation probability exceeds the inverse of the graph's branching factor (p > e^{-\gamma}, where \gamma is the BFS shell-growth exponent). We close this gap with a spatio-temporal sidecar that predicts which routing geometry fits the current topology. The sidecar comprises (i) a Euclidean propagation scorer for dense, cyclic subgraphs, (ii) a hyperbolic scorer capturing exponential risk in tree-like subgraphs, and (iii) a compact learned gate (133 parameters) that blends the two scores using topology and geometry-aware features. On 250 benchmark scenarios spanning five topology regimes, the sidecar lifts the native scheduler's win rate from 50.4% to 87.2% (+36.8 pp). In tree-like regimes, gains reach +48 to +68 pp. The learned gate achieves held-out AUC = 0.9247, confirming geometry preference is recoverable from live signals. Cross-architecture validation on Barabasi-Albert, Watts-Strogatz, and Erdos-Renyi graphs confirms propagation modeling generalizes across graph families.

**arXiv ID:** 2603.17112
</details>

<details>
<summary><strong>Towards Intelligent Geospatial Data Discovery: a knowledge graph-driven multi-agent framework powered by large language models</strong> - Ruixiang Liu, Zhenlong Li, Ali Khosravi Kazazi - [[pdf]](https://arxiv.org/pdf/2603.20670)</summary>

**Abstract:** The rapid growth in the volume, variety, and velocity of geospatial data has created data ecosystems that are highly distributed, heterogeneous, and semantically inconsistent. Existing data catalogs, portals, and infrastructures still rely largely on keyword-based search with limited semantic support, which often fails to capture user intent and leads to weak retrieval performance. To address these challenges, this study proposes a knowledge graph-driven multi-agent framework for intelligent geospatial data discovery, powered by large language models. The framework introduces a unified geospatial metadata ontology as a semantic mediation layer to align heterogeneous metadata standards across platforms and constructs a geospatial metadata knowledge graph to explicitly model datasets and their multidimensional relationships. Building on the structured representation, the framework adopts a multi-agent collaborative architecture to perform intent parsing, knowledge graph retrieval, and answer synthesis, forming an interpretable and closed-loop discovery process from user queries to results. Results from representative use cases and performance evaluation show that the framework substantially improves intent matching accuracy, ranking quality, recall, and discovery transparency compared with traditional systems. This study advances geospatial data discovery toward a more semantic, intent-aware, and intelligent paradigm, providing a practical foundation for next-generation intelligent and autonomous spatial data infrastructures and contributing to the broader vision of Autonomous GIS.

**arXiv ID:** 2603.20670
</details>

<details>
<summary><strong>TrustTrade: Human-Inspired Selective Consensus Reduces Decision Uncertainty in LLM Trading Agents</strong> - Minghan Li, Rachel Gonsalves, Weiyue Li, Sunghoon Yoon, Mengyu Wang - [[pdf]](https://arxiv.org/pdf/2603.22567)</summary>

**Abstract:** Large language models (LLMs) are increasingly deployed as autonomous agents in financial trading. However, they often exhibit a hazardous behavioral bias that we term uniform trust, whereby retrieved information is implicitly assumed to be factual and heterogeneous sources are treated as equally informative. This assumption stands in sharp contrast to human decision-making, which relies on selective filtering, cross-validation, and experience-driven weighting of information sources. As a result, LLM-based trading systems are particularly vulnerable to multi-source noise and misinformation, amplifying factual hallucinations and leading to unstable risk-return performance. To bridge this behavioral gap, we introduce TrustTrade (Trust-Rectified Unified Selective Trader), a multi-agent selective consensus framework inspired by human epistemic heuristics. TrustTrade replaces uniform trust with cross-agent consistency by aggregating information from multiple independent LLM agents and dynamically weighting signals based on their semantic and numerical agreement. Consistent signals are prioritized, while divergent, weakly grounded, or temporally inconsistent inputs are selectively discounted. To further stabilize decision-making, TrustTrade incorporates deterministic temporal signals as reproducible anchors and a reflective memory mechanism that adapts risk preferences at test time without additional training. Together, these components suppress noise amplification and hallucination-driven volatility, yielding more stable and risk-aware trading behavior. Across controlled backtesting in high-noise market environments (2024 Q1 and 2026 Q1), the proposed TrustTrade calibrates LLM trading behavior from extreme risk-return regimes toward a human-aligned, mid-risk and mid-return profile.

**arXiv ID:** 2603.22567
</details>

<details>
<summary><strong>VLM-CAD: VLM-Optimized Collaborative Agent Design Workflow for Analog Circuit Sizing</strong> - Guanyuan Pan, Shuai Wang, Yugui Lin, Tiansheng Zhou, Pietro Liò, Zhenxin Zhao, Yaqi Wang - [[pdf]](https://arxiv.org/pdf/2601.07315)</summary>

**Abstract:** Vision Language Models (VLMs) have demonstrated remarkable potential in multimodal reasoning, yet they inherently suffer from spatial blindness and logical hallucinations when interpreting densely structured engineering content, such as analog circuit schematics. To address these challenges, we propose a Vision Language Model-Optimized Collaborative Agent Design Workflow for Analog Circuit Sizing (VLM-CAD) designed for robust, step-by-step reasoning over multimodal evidence. VLM-CAD bridges the modality gap by integrating a neuro-symbolic structural parsing module, Image2Net, which transforms raw pixels into explicit topological graphs and structured JSON representations to anchor VLM interpretation in deterministic facts. To ensure the reliability required for engineering decisions, we further propose ExTuRBO, an Explainable Trust Region Bayesian Optimization method. ExTuRBO serves as an explainable grounding engine, employing agent-generated semantic seeds to warm-start local searches and utilizing Automatic Relevance Determination to provide quantified evidence for the VLM's decisions. Experimental results on two complex circuit benchmarks demonstrate that VLM-CAD significantly enhances spatial reasoning accuracy and maintains physics-based explainability. VLM-CAD consistently satisfies complex specification requirements while achieving low power consumption, with a total runtime under 66 minutes, marking a significant step toward robust, explainable multimodal reasoning in specialized technical domains.

**arXiv ID:** 2601.07315
</details>

<details>
<summary><strong>Evidence-Decision-Feedback: Theory-Driven Adaptive Scaffolding for LLM Agents</strong> - Clayton Cohn, Siyuan Guo, Surya Rayala, Hanchen David Wang, Naveeduddin Mohammed, Umesh Timalsina, Shruti Jain, Angela Eeds, Menton Deweese, Pamela J. Osborn Popp, Rebekah Stanton, Shakeera Walker, Meiyi Ma, Gautam Biswas - [[pdf]](https://arxiv.org/pdf/2602.01415)</summary>

**Abstract:** Multi-agent LLM architectures offer opportunities for pedagogical agents to help students construct domain knowledge and develop critical-thinking skills, yet many operate on a "one-size-fits-all" basis, limiting their ability to provide personalized support. To address this, we introduce Evidence-Decision-Feedback (EDF), a theoretical framework for adaptive scaffolding using LLMs. EDF integrates elements of intelligent tutoring systems and agentic behavior by organizing interactions around evidentiary inference, pedagogical decision-making, and adaptive feedback. We instantiate EDF through Copa, an agentic collaborative peer agent for STEM+C problem-solving. In an authentic high school classroom study, we show that EDF-guided interactions align feedback with students' demonstrated understanding and task mastery; promote gradual scaffold fading; and support interpretable, evidence-grounded explanations without fostering overreliance.

**arXiv ID:** 2602.01415
</details>

<details>
<summary><strong>MARS: toward more efficient multi-agent collaboration for LLM reasoning</strong> - Xiao Wang, Jia Wang, Yijie Wang, Pengtao Dang, Sha Cao, Chi Zhang - [[pdf]](https://arxiv.org/pdf/2509.20502)</summary>

**Abstract:** Large language models (LLMs) have achieved impressive results in natural language understanding, yet their reasoning capabilities remain limited when operating as single agents. Multi-Agent Debate (MAD) has been proposed to address this limitation by enabling collaborative reasoning among multiple models in a round-table debate manner. While effective, MAD introduces substantial computational overhead due to the number of agents involved and the frequent communication required. In this paper, we propose MARS (Multi-Agent Review System), a role-based collaboration framework inspired by the review process. In MARS, an author agent generates an initial solution, reviewer agents provide decisions and comments independently, and a meta-reviewer integrates the feedback to make the final decision and guide further revision. This design enhances reasoning quality while avoiding costly reviewer-to-reviewer interactions, thereby controlling token consumption and inference time. We compared MARS with both MAD and other state-of-the-art reasoning strategies across multiple benchmarks. Extensive experiments with different LLMs show that MARS matches the accuracy of MAD while reducing both token usage and inference time by approximately 50\%. Code is available at this https URL.

**arXiv ID:** 2509.20502
</details>

<details>
<summary><strong>Designing Explainable Conversational Agentic Systems for Guaraní Speakers</strong> - Samantha Adorno, Akshata Kishore Moharir, Ratna Kandala - [[pdf]](https://arxiv.org/pdf/2603.05743)</summary>

**Abstract:** Although artificial intelligence (AI) and Human-Computer Interaction (HCI) systems are often presented as universal solutions, their design remains predominantly text-first, underserving primarily oral languages and indigenous communities. This position paper uses Guaraní, an official and widely spoken language of Paraguay, as a case study to argue that language support in AI remains insufficient unless it aligns with lived oral practices. We propose an alternative to the standard "text-to-speech" pipeline, proposing instead an oral-first multi-agent architecture. By decoupling Guaraní natural language understanding from dedicated agents for conversation state and community-led governance, we demonstrate a technical framework that respects indigenous data sovereignty and diglossia. Our work moves beyond mere recognition to focus on turn-taking, repair, and shared context as the primary locus of interaction. We conclude that for AI to be truly culturally grounded, it must shift from adapting oral languages to text-centric systems to treating spoken conversation as a first-class design requirement, ensuring digital ecosystems empower rather than overlook diverse linguistic practices.

**arXiv ID:** 2603.05743
</details>

<details>
<summary><strong>Learning Multi-Agent Local Collision-Avoidance for Collaborative Carrying tasks with Coupled Quadrupedal Robots</strong> - Francesca Bray, Simone Tolomei, Andrei Cramariuc, Cesar Cadena, Marco Hutter - [[pdf]](https://arxiv.org/pdf/2603.23278)</summary>

**Abstract:** Robotic collaborative carrying could greatly benefit human activities like warehouse and construction site management. However, coordinating the simultaneous motion of multiple robots represents a significant challenge. Existing works primarily focus on obstacle-free environments, making them unsuitable for most real-world applications. Works that account for obstacles, either overfit to a specific terrain configuration or rely on pre-recorded maps combined with path planners to compute collision-free trajectories. This work focuses on two quadrupedal robots mechanically connected to a carried object. We propose a Reinforcement Learning (RL)-based policy that enables tracking a commanded velocity direction while avoiding collisions with nearby obstacles using only onboard sensing, eliminating the need for precomputed trajectories and complete map knowledge. Our work presents a hierarchical architecture, where a perceptive high-level object-centric policy commands two pretrained locomotion policies. Additionally, we employ a game-inspired curriculum to increase the complexity of obstacles in the terrain progressively. We validate our approach on two quadrupedal robots connected to a bar via spherical joints, benchmarking it against optimization-based and decentralized RL baselines. Our hardware experiments demonstrate the ability of our system to locomote in unknown environments without the need for a map or a path planner. The video of our work is available in the multimedia material.

**arXiv ID:** 2603.23278
</details>

</details>

<details open>
<summary><h2>Other Agent Research (14 papers)</h2></summary>

<details>
<summary><strong>Describe-Then-Act: Proactive Agent Steering via Distilled Language-Action World Models</strong> - Massimiliano Pappa, Luca Romani, Valentino Sacco, Alessio Palma, Stéphane Lathuilière, Fabio Galasso, Xavier Alameda-Pineda, Indro Spinelli - [[pdf]](https://arxiv.org/pdf/2603.23149)</summary>

**Abstract:** Deploying safety-critical agents requires anticipating the consequences of actions before they are executed. While world models offer a paradigm for this proactive foresight, current approaches relying on visual simulation incur prohibitive latencies, often exceeding several seconds per step. In this work, we challenge the assumption that visual processing is necessary for failure prevention. We show that a trained policy's latent state, combined with its planned actions, already encodes sufficient information to anticipate action outcomes, making visual simulation redundant for failure prevention. To this end, we introduce DILLO (DIstiLLed Language-ActiOn World Model), a fast steering layer that shifts the paradigm from "simulate-then-act" to "describe-then-act." DILLO is trained via cross-modal distillation, where a privileged Vision Language Model teacher annotates offline trajectories and a latent-conditioned Large Language Model student learns to predict semantic outcomes. This creates a text-only inference path, bypassing heavy visual generation entirely, achieving a 14x speedup over baselines. Experiments on MetaWorld and LIBERO demonstrate that DILLO produces high-fidelity descriptions of the next state and is able to steer the policy, improving episode success rate by up to 15 pp and 9.3 pp on average across tasks.

**arXiv ID:** 2603.23149
</details>

<details>
<summary><strong>AgentSLR: Automating Systematic Literature Reviews in Epidemiology with Agentic AI</strong> - Shreyansh Padarha, Ryan Othniel Kearns, Tristan Naidoo, Lingyi Yang, Łukasz Borchmann, Piotr BŁaszczyk, Christian Morgenstern, Ruth McCabe, Sangeeta Bhatia, Philip H. Torr, Jakob Foerster, Scott A. Hale, Thomas Rawson, Anne Cori, Elizaveta Semenova, Adam Mahdi - [[pdf]](https://arxiv.org/pdf/2603.22327)</summary>

**Abstract:** Systematic literature reviews are essential for synthesizing scientific evidence but are costly, difficult to scale and time-intensive, creating bottlenecks for evidence-based policy. We study whether large language models can automate the complete systematic review workflow, from article retrieval, article screening, data extraction to report synthesis. Applied to epidemiological reviews of nine WHO-designated priority pathogens and validated against expert-curated ground truth, our open-source agentic pipeline (AgentSLR) achieves performance comparable to human researchers while reducing review time from approximately 7 weeks to 20 hours (a 58x speed-up). Our comparison of five frontier models reveals that performance on SLR is driven less by model size or inference cost than by each model's distinctive capabilities. Through human-in-the-loop validation, we identify key failure modes. Our results demonstrate that agentic AI can substantially accelerate scientific evidence synthesis in specialised domains.

**arXiv ID:** 2603.22327
</details>

<details>
<summary><strong>Do Consumers Accept AIs as Moral Compliance Agents?</strong> - Greg Nyilasy, Abraham Ryan Ade Putra Hito, Jennifer Overbeck, Brock Bastian, Darren W. Dahl - [[pdf]](https://arxiv.org/pdf/2603.22617)</summary>

**Abstract:** Consumers are generally resistant to Artificial Intelligence (AI) involvement in moral decision-making, perceiving moral agency as requiring uniquely human traits. This research investigates whether consumers might instead accept AIs in the role of moral compliance, where AI upholds pre-existing moral norms without exercising subjective discretion. Across five studies this research shows that consumers evaluate AI more positively than human agents in moral compliance roles. The findings reveal that this preference arises from inferences of AI's lack of ulterior motives, which are often attributed to human agents. While previous studies have focused on AI as a decision-maker, this work demonstrates the critical role of upholding pre-existing rules, a role in which AI is perceived to excel. These findings contribute to understanding consumer acceptance of moral AI and provide actionable insights for organizations seeking to leverage AI in ethical oversight. By positioning AI as a moral compliance agent, companies can address consumer skepticism, enhance trust, and improve perceptions of corporate ethicality.

**arXiv ID:** 2603.22617
</details>

<details>
<summary><strong>KALAVAI: Predicting When Independent Specialist Fusion Works -- A Quantitative Model for Post-Hoc Cooperative LLM Training</strong> - Ramchand Kumaresan - [[pdf]](https://arxiv.org/pdf/2603.22755)</summary>

**Abstract:** Independently trained domain specialists can be fused post-hoc into a single model that outperforms any individual specialist, and the gain is predictable: gain = 0.82 x divergence - 2.72 (R^2 = 0.856, n=6, 3-26% divergence). This enables practitioners to estimate cooperative value before committing compute. Below ~3.3% divergence, gains approach this http URL the KALAVAI protocol, contributors fine-tune copies of a shared checkpoint independently, then submit for lightweight MoE routing (500 steps). Gains are consistent: +7.72% at 410M (+/-0.02%, 3 seeds), +7.49% at 1B (+/-0.01%, 3 seeds), +6.53% at 6.9B, each over the best specialist. The router matches domain-oracle routing within <10^{-5} nats. Cross-lingual fusion (Tamil/Yoruba/Welsh/Code) achieves +21.76%, with Yoruba perplexity falling 41.9 to 7.7. A 20-contributor federation achieves +16.71% (+/-0.07pp, 3 seeds).Three requirements bound the protocol. Shared initialisation is necessary: checkpoint mismatch degrades routing. Frozen layers are optional below ~10,000 steps and beneficial beyond. Learned routing is essential: uniform averaging degrades by -1.2% vs. best specialist, while any trained router achieves oracle-optimal assignment.

**arXiv ID:** 2603.22755
</details>

<details>
<summary><strong>Designing Agentic AI-Based Screening for Portfolio Investment</strong> - Mehmet Caner, Agostino Capponi, Nathan Sun, Jonathan Y. Tan - [[pdf]](https://arxiv.org/pdf/2603.23300)</summary>

**Abstract:** We introduce a new agentic artificial intelligence (AI) platform for portfolio management. Our architecture consists of three layers. First, two large language model (LLM) agents are assigned specialized tasks: one agent screens for firms with desirable fundamentals, while a sentiment analysis agent screens for firms with desirable news. Second, these agents deliberate to generate and agree upon buy and sell signals from a large portfolio, substantially narrowing the pool of candidate assets. Finally, we apply a high-dimensional precision matrix estimation procedure to determine optimal portfolio weights. A defining theoretical feature of our framework is that the number of assets in the portfolio is itself a random variable, realized through the screening process. We introduce the concept of sensible screening and establish that, under mild screening errors, the squared Sharpe ratio of the screened portfolio consistently estimates its target. Empirically, our method achieves superior Sharpe ratios relative to an unscreened baseline portfolio and to conventional screening approaches, evaluated on S&P 500 data over the period 2020--2024.

**arXiv ID:** 2603.23300
</details>

<details>
<summary><strong>Toward Data Systems That Are Business Semantic Centric and AI Agents Assisted</strong> - Cecil Pang - [[pdf]](https://arxiv.org/pdf/2506.05520)</summary>

**Abstract:** Contemporary businesses operate in dynamic environments requiring rapid adaptation to achieve goals and maintain competitiveness. Existing data platforms often fall short by emphasizing tools over alignment with business needs, resulting in inefficiencies and delays. To address this gap, I propose the Business Semantics Centric, AI Agents Assisted Data System (BSDS), a holistic system that integrates architecture, workflows, and team organization to ensure data systems are tailored to business priorities rather than dictated by technical constraints. BSDS redefines data systems as dynamic enablers of business success, transforming them from passive tools into active drivers of organizational growth. BSDS has a modular architecture that comprises curated data linked to business entities, a knowledge base for context-aware AI agents, and efficient data pipelines. AI agents play a pivotal role in assisting with data access and system management, reducing human effort, and improving scalability. Complementing this architecture, BSDS incorporates workflows optimized for both exploratory data analysis and production requirements, balancing speed of delivery with quality assurance. A key innovation of BSDS is its incorporation of the human factor. By aligning data team expertise with business semantics, BSDS bridges the gap between technical capabilities and business needs. Validated through real-world implementation, BSDS accelerates time-to-market for data-driven initiatives, enhances cross-functional collaboration, and provides a scalable blueprint for businesses of all sizes. Future research can build on BSDS to explore optimization strategies using complex systems and adaptive network theories, as well as developing autonomous data systems leveraging AI agents.

**arXiv ID:** 2506.05520
</details>

<details>
<summary><strong>AI-Generated Code Is Not Reproducible (Yet): An Empirical Study of Dependency Gaps in LLM-Based Coding Agents</strong> - Bhanu Prakash Vangala, Ali Adibifar, Ashish Gehani, Tanu Malik - [[pdf]](https://arxiv.org/pdf/2512.22387)</summary>

**Abstract:** The rise of Large Language Models (LLMs) as coding agents promises to accelerate software development, but their impact on generated code reproducibility remains largely unexplored. This paper presents an empirical study investigating whether LLM-generated code can be executed successfully in a clean environment with only OS packages and using only the dependencies that the model specifies. We evaluate three state-of-the-art LLM coding agents (Claude Code, OpenAI Codex, and Gemini) across 300 projects generated from 100 standardized prompts in Python, JavaScript, and Java. We introduce a three-layer dependency framework (distinguishing between claimed, working, and runtime dependencies) to quantify execution reproducibility. Our results show that only 68.3% of projects execute out-of-the-box, with substantial variation across languages (Python 89.2%, Java 44.0%). We also find a 13.5 times average expansion from declared to actual runtime dependencies, revealing significant hidden dependencies.

**arXiv ID:** 2512.22387
</details>

<details>
<summary><strong>Knowledge Access Beats Model Size: Memory Augmented Routing for Persistent AI Agents</strong> - Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen - [[pdf]](https://arxiv.org/pdf/2603.23013)</summary>

**Abstract:** Production AI agents frequently receive user-specific queries that are highly repetitive, with up to 47\% being semantically similar to prior interactions, yet each query is typically processed with the same computational cost. We argue that this redundancy can be exploited through conversational memory, transforming repetition from a cost burden into an efficiency advantage. We propose a memory-augmented inference framework in which a lightweight 8B-parameter model leverages retrieved conversational context to answer all queries via a low-cost inference path. Without any additional training or labeled data, this approach achieves 30.5\% F1, recovering 69\% of the performance of a full-context 235B model while reducing effective cost by 96\%. Notably, a 235B model without memory (13.7\% F1) underperforms even the standalone 8B model (15.4\% F1), indicating that for user-specific queries, access to relevant knowledge outweighs model scale. We further analyze the role of routing and confidence. At practical confidence thresholds, routing alone already directs 96\% of queries to the small model, but yields poor accuracy (13.0\% F1) due to confident hallucinations. Memory does not substantially alter routing decisions; instead, it improves correctness by grounding responses in retrieved user-specific information. As conversational memory accumulates over time, coverage of recurring topics increases, further narrowing the performance gap. We evaluate on 152 LoCoMo questions (Qwen3-8B/235B) and 500 LongMemEval questions. Incorporating hybrid retrieval (BM25 + cosine similarity) improves performance by an additional +7.7 F1, demonstrating that retrieval quality directly enhances end-to-end system performance. Overall, our results highlight that memory, rather than model size, is the primary driver of accuracy and efficiency in persistent AI agents.

**arXiv ID:** 2603.23013
</details>

<details>
<summary><strong>TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration</strong> - Chunxiao Li, Lijun Li, Jing Shao - [[pdf]](https://arxiv.org/pdf/2603.22882)</summary>

**Abstract:** The rapid advancement of Vision-Language Models (VLMs) has brought their safety vulnerabilities into sharp focus. However, existing red teaming methods are fundamentally constrained by an inherent linear exploration paradigm, confining them to optimizing within a predefined strategy set and preventing the discovery of novel, diverse exploits. To transcend this limitation, we introduce TreeTeaming, an automated red teaming framework that reframes strategy exploration from static testing to a dynamic, evolutionary discovery process. At its core lies a strategic Orchestrator, powered by a Large Language Model (LLM), which autonomously decides whether to evolve promising attack paths or explore diverse strategic branches, thereby dynamically constructing and expanding a strategy tree. A multimodal actuator is then tasked with executing these complex strategies. In the experiments across 12 prominent VLMs, TreeTeaming achieves state-of-the-art attack success rates on 11 models, outperforming existing methods and reaching up to 87.60\% on GPT-4o. The framework also demonstrates superior strategic diversity over the union of previously public jailbreak strategies. Furthermore, the generated attacks exhibit an average toxicity reduction of 23.09\%, showcasing their stealth and subtlety. Our work introduces a new paradigm for automated vulnerability discovery, underscoring the necessity of proactive exploration beyond static heuristics to secure frontier AI models.

**arXiv ID:** 2603.22882
</details>

<details>
<summary><strong>Variable-Resolution Virtual Maps for Autonomous Exploration with Unmanned Surface Vehicles (USVs)</strong> - Ye Li, Yewei Huang, Wenlong GaoZhang, Alberto Quattrini Li, Brendan Englot, Yuanchang Liu - [[pdf]](https://arxiv.org/pdf/2603.22667)</summary>

**Abstract:** Autonomous exploration by unmanned surface vehicles (USVs) in near-shore waters requires reliable localisation and consistent mapping over extended areas, but this is challenged by GNSS degradation, environment-induced localisation uncertainty, and limited on-board computation. Virtual map-based methods explicitly model localisation and mapping uncertainty by tightly coupling factor-graph SLAM with a map uncertainty criterion. However, their storage and computational costs scale poorly with fixed-resolution workspace discretisations, leading to inefficiency in large near-shore environments. Moreover, overvaluing feature-sparse open-water regions can increase the risk of SLAM failure as a result of imbalance between exploration and exploitation. To address these limitations, we propose a Variable-Resolution Virtual Map (VRVM), a computationally efficient method for representing map uncertainty using bivariate Gaussian virtual landmarks placed in the cells of an adaptive quadtree. The adaptive quadtree enables an area-weighted uncertainty representation that keeps coarse, far-field virtual landmarks deliberately uncertain while allocating higher resolution to information-dense regions, and reduces the sensitivity of the map valuation to local refinements of the tree. An expectation-maximisation (EM) planner is adopted to evaluate pose and map uncertainty along frontiers using the VRVM, balancing exploration and exploitation. We evaluate VRVM against several state-of-the-art exploration algorithms in the VRX Gazebo simulator, using a realistic marina environment across different testing scenarios with an increasing level of exploration difficulty. The results indicate that our method offers safer behaviour and better utilisation of on-board computation in GNSS-degraded near-shore environments.

**arXiv ID:** 2603.22667
</details>

<details>
<summary><strong>Fleet-Level Battery-Health-Aware Scheduling for Autonomous Mobile Robots</strong> - Jiachen Li, Shihao Li, Jian Chu, Wei Li, Dongmei Chen - [[pdf]](https://arxiv.org/pdf/2603.22731)</summary>

**Abstract:** Autonomous mobile robot fleets must coordinate task allocation and charging under limited shared resources, yet most battery aware planning methods address only a single robot. This paper extends degradation cost aware task planning to a multi robot setting by jointly optimizing task assignment, service sequencing, optional charging decisions, charging mode selection, and charger access while balancing degradation across the fleet. The formulation relies on reduced form degradation proxies grounded in the empirical battery aging literature, capturing both charging mode dependent wear and idle state of charge dependent aging; the bilinear idle aging term is linearized through a disaggregated piecewise McCormick formulation. Tight big M values derived from instance data strengthen the LP relaxation. To manage scalability, we propose a hierarchical matheuristic in which a fleet level master problem coordinates assignments, routes, and charger usage, while robot level subproblems whose integer part decomposes into trivially small independent partition selection problems compute route conditioned degradation schedules. Systematic experiments compare the proposed method against three baselines: a rule based nearest available dispatcher, an energy aware formulation that enforces battery feasibility without modeling degradation, and a charger unaware formulation that accounts for degradation but ignores shared charger capacity limits.

**arXiv ID:** 2603.22731
</details>

<details>
<summary><strong>Task-Aware Positioning for Improvisational Tasks in Mobile Construction Robots via an AI Agent with Multi-LMM Modules</strong> - Seongju Jang, Francis Baek, SangHyun Lee - [[pdf]](https://arxiv.org/pdf/2603.22903)</summary>

**Abstract:** Due to the ever-changing nature of construction, many tasks on sites occur in an improvisational manner. Existing mobile construction robot studies remain limited in addressing improvisational tasks, where task-required locations, timing of task occurrence, and contextual information required for task execution are not known in advance. We propose an agent that understands improvisational tasks given in natural language, identifies the task-required location, and positions itself. The agent's functionality was decomposed into three Large Multimodal Model (LMM) modules operating in parallel, enabling the application of LMMs for task interpretation and breakdown, construction drawing-based navigation, and visual reasoning to identify non-predefined task-required locations. The agent was implemented with a quadruped robot and achieved a 92.2% success rate for identifying and positioning at task-required locations across three tests designed to assess improvisational task handling. This study enables mobile construction robots to perform non-predefined tasks autonomously.

**arXiv ID:** 2603.22903
</details>

<details>
<summary><strong>Integrated cooperative localization of heterogeneous measurement swarm: A unified data-driven method</strong> - Kunrui Ze, Wei Wang, Guibin Sun, Jiaqi Yan, Kexin Liu, Jinhu Lü - [[pdf]](https://arxiv.org/pdf/2603.04932)</summary>

**Abstract:** The cooperative localization (CL) problem in heterogeneous robotic systems with different measurement capabilities is investigated in this work. In practice, heterogeneous sensors lead to directed and sparse measurement topologies, whereas most existing CL approaches rely on multilateral localization with restrictive multi-neighbor geometric requirements. To overcome this limitation, we enable pairwise relative localization (RL) between neighboring robots using only mutual measurement and odometry information. A unified data-driven adaptive RL estimator is first developed to handle heterogeneous and unidirectional measurements. Based on the convergent RL estimates, a distributed pose-coupling CL strategy is then designed, which guarantees CL under a weakly connected directed measurement topology, representing the least restrictive condition among existing results. The proposed method is independent of specific control tasks and is validated through a formation control application and real-world experiments.

**arXiv ID:** 2603.04932
</details>

<details>
<summary><strong>IntentWeave: A Progressive Entry Ladder for Multi-Surface Browser Agents in Cloud Portals</strong> - Wanying Mo, Jijia Lai, Xiaoming Wang - [[pdf]](https://arxiv.org/pdf/2603.22917)</summary>

**Abstract:** Browser agents built on LLMs can act in web interfaces, yet most remain confined to a single chat surface (e.g., a sidebar). This mismatch with real browsing can increase context-switching and reduce user control. We introduce \textbf{IntentWeave}, a design space of ten spatial paradigms for embedding agentic assistance across a browser, organized as a progressive entry ladder from micro-interventions to dedicated workspaces. We implement IntentWeave as a browser-extension prototype on the Alibaba Cloud website and compare three entry strategies in a within-subjects study (N=16). Workspace-heavy strategies reduced completion time but lowered perceived control; micro-only strategies preserved control but were often insufficient; a mixed sidecar approach achieved the highest satisfaction. We conclude with guidance for escalating and retreating agent surfaces without disrupting user agency.

**arXiv ID:** 2603.22917
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (20 papers)</h2></summary>

<details>
<summary><strong>Beyond Binary Correctness: Scaling Evaluation of Long-Horizon Agents on Subjective Enterprise Tasks</strong> - Abhishek Chandwani, Ishan Gupta - [[pdf]](https://arxiv.org/pdf/2603.22744)</summary>

**Abstract:** Large language models excel on objectively verifiable tasks such as math and programming, where evaluation reduces to unit tests or a single correct answer. In contrast, real-world enterprise work is often subjective and context-dependent: success hinges on organizational goals, user intent, and the quality of intermediate artifacts produced across long, multi-tool workflows.
We introduce LH-Bench, a three-pillar evaluation design that moves beyond binary correctness to score autonomous, long-horizon execution on subjective enterprise tasks. The pillars are: (i) expert-grounded rubrics that give LLM judges the domain context needed to score subjective work, (ii) curated ground-truth artifacts that enable stepwise reward signals (e.g., chapter-level annotation for content tasks), and (iii) pairwise human preference evaluation for convergent validation. We show that domain-authored rubrics provide substantially more reliable evaluation signals than LLM-authored rubrics (kappa = 0.60 vs. 0.46), and that human preference judgments confirm the same top-tier separation (p < 0.05), evidence that expert-grounded evaluation can scale without sacrificing reliability. We release public datasets and report results on two environments: Figma-to-code (33 real .fig tasks against the Figma API via MCP) and Programmatic content (41 courses comprising 183 individually-evaluated chapters on a course platform serving 30+ daily users).

**arXiv ID:** 2603.22744
</details>

<details>
<summary><strong>Beyond Hard Constraints: Budget-Conditioned Reachability For Safe Offline Reinforcement Learning</strong> - Janaka Chathuranga Brahmanage, Akshat Kumar - [[pdf]](https://arxiv.org/pdf/2603.22292)</summary>

**Abstract:** Sequential decision making using Markov Decision Process underpins many realworld applications. Both model-based and model free methods have achieved strong results in these settings. However, real-world tasks must balance reward maximization with safety constraints, often conflicting objectives, that can lead to unstable min/max, adversarial optimization. A promising alternative is safety reachability analysis, which precomputes a forward-invariant safe state, action set, ensuring that an agent starting inside this set remains safe indefinitely. Yet, most reachability based methods address only hard safety constraints, and little work extends reachability to cumulative cost constraints. To address this, first, we define a safetyconditioned reachability set that decouples reward maximization from cumulative safety cost constraints. Second, we show how this set enforces safety constraints without unstable min/max or Lagrangian optimization, yielding a novel offline safe RL algorithm that learns a safe policy from a fixed dataset without environment interaction. Finally, experiments on standard offline safe RL benchmarks, and a real world maritime navigation task demonstrate that our method matches or outperforms state of the art baselines while maintaining safety.

**arXiv ID:** 2603.22292
</details>

<details>
<summary><strong>CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation</strong> - Max Fu, Justin Yu, Karim El-Refai, Ethan Kou, Haoru Xue, Huang Huang, Wenli Xiao, Guanzhi Wang, Fei-Fei Li, Guanya Shi, Jiajun Wu, Shankar Sastry, Yuke Zhu, Ken Goldberg, Linxi "Jim" Fan - [[pdf]](https://arxiv.org/pdf/2603.22435)</summary>

**Abstract:** "Code-as-Policy" considers how executable code can complement data-intensive Vision-Language-Action (VLA) methods, yet their effectiveness as autonomous controllers for embodied manipulation remains underexplored. We present CaP-X, an open-access framework for systematically studying Code-as-Policy agents in robot manipulation. At its core is CaP-Gym, an interactive environment in which agents control robots by synthesizing and executing programs that compose perception and control primitives. Building on this foundation, CaP-Bench evaluates frontier language and vision-language models across varying levels of abstraction, interaction, and perceptual grounding. Across 12 models, CaP-Bench reveals a consistent trend: performance improves with human-crafted abstractions but degrades as these priors are removed, exposing a dependence on designer scaffolding. At the same time, we observe that this gap can be mitigated through scaling agentic test-time computation--through multi-turn interaction, structured execution feedback, visual differencing, automatic skill synthesis, and ensembled reasoning--substantially improves robustness even when agents operate over low-level primitives. These findings allow us to derive CaP-Agent0, a training-free framework that recovers human-level reliability on several manipulation tasks in simulation and on real embodiments. We further introduce CaP-RL, showing reinforcement learning with verifiable rewards improves success rates and transfers from sim2real with minimal gap. Together, CaP-X provides a principled, open-access platform for advancing embodied coding agents.

**arXiv ID:** 2603.22435
</details>

<details>
<summary><strong>AwesomeLit: Towards Hypothesis Generation with Agent-Supported Literature Research</strong> - Zefei Xie, Yuhan Guo, Kai Xu - [[pdf]](https://arxiv.org/pdf/2603.22648)</summary>

**Abstract:** There are different goals for literature research, from understanding an unfamiliar topic to generate hypothesis for the next research project. The nature of literature research also varies according to user's familiarity level of the topic. For inexperienced researchers, identifying gaps in the existing literature and generating feasible hypothesis are crucial but challenging. While general ``deep research'' tools can be used, they are not designed for such use case, thus often not effective. In addition, the ``black box" nature and hallucination of Large Language Models (LLMs) often lead to distrust. In this paper, we introduce a human-agent collaborative visualization system AwesomeLit to address this need. It has several novel features: a transparent user-steerable agentic workflow; a dynamically generated query exploring tree, visualizing the exploration path and provenance; and a semantic similarity view, depicting the relationships between papers. It enables users to transition from general intentions to detailed research topics. Finally, a qualitative study involving several early researchers showed that AwesomeLit is effective in helping users explore unfamiliar topics, identify promising research directions, and improve confidence in research results.

**arXiv ID:** 2603.22648
</details>

<details>
<summary><strong>EVA: Efficient Reinforcement Learning for End-to-End Video Agent</strong> - Yaolun Zhang, Ruohui Wang, Jiahao Wang, Yepeng Tang, Xuanyu Zheng, Haonan Duan, Hao Lu, Hanming Deng, Lewei Lu - [[pdf]](https://arxiv.org/pdf/2603.22918)</summary>

**Abstract:** Video understanding with multimodal large language models (MLLMs) remains challenging due to the long token sequences of videos, which contain extensive temporal dependencies and redundant frames. Existing approaches typically treat MLLMs as passive recognizers, processing entire videos or uniformly sampled frames without adaptive reasoning. Recent agent-based methods introduce external tools, yet still depend on manually designed workflows and perception-first strategies, resulting in inefficiency on long videos. We present EVA, an Efficient Reinforcement Learning framework for End-to-End Video Agent, which enables planning-before-perception through iterative summary-plan-action-reflection reasoning. EVA autonomously decides what to watch, when to watch, and how to watch, achieving query-driven and efficient video understanding. To train such agents, we design a simple yet effective three-stage learning pipeline - comprising supervised fine-tuning (SFT), Kahneman-Tversky Optimization (KTO), and Generalized Reward Policy Optimization (GRPO) - that bridges supervised imitation and reinforcement learning. We further construct high-quality datasets for each stage, supporting stable and reproducible training. We evaluate EVA on six video understanding benchmarks, demonstrating its comprehensive capabilities. Compared with existing baselines, EVA achieves a substantial improvement of 6-12% over general MLLM baselines and a further 1-3% gain over prior adaptive agent methods. Our code and model are available at this https URL.

**arXiv ID:** 2603.22918
</details>

<details>
<summary><strong>Neural ODE and SDE Models for Adaptation and Planning in Model-Based Reinforcement Learning</strong> - Chao Han, Stefanos Ioannou, Luca Manneschi, T.J. Hayward, Michael Mangan, Aditya Gilra, Eleni Vasilaki - [[pdf]](https://arxiv.org/pdf/2603.23245)</summary>

**Abstract:** We investigate neural ordinary and stochastic differential equations (neural ODEs and SDEs) to model stochastic dynamics in fully and partially observed environments within a model-based reinforcement learning (RL) framework. Through a sequence of simulations, we show that neural SDEs more effectively capture the inherent stochasticity of transition dynamics, enabling high-performing policies with improved sample efficiency in challenging scenarios. We leverage neural ODEs and SDEs for efficient policy adaptation to changes in environment dynamics via inverse models, requiring only limited interactions with the new environment. To address partial observability, we introduce a latent SDE model that combines an ODE with a GAN-trained stochastic component in latent space. Policies derived from this model provide a strong baseline, outperforming or matching general model-based and model-free approaches across stochastic continuous-control benchmarks. This work demonstrates the applicability of action-conditional latent SDEs for RL planning in environments with stochastic transitions. Our code is available at: this https URL

**arXiv ID:** 2603.23245
</details>

<details>
<summary><strong>Hybrid Stackelberg Game and Diffusion-based Auction for Two-tier Agentic AI Task Offloading in Internet of Agents</strong> - Yue Zhong, Yongju Tong, Jiawen Kang, Minghui Dai, Hong-Ning Dai, Zhou Su, Dusit Niyato - [[pdf]](https://arxiv.org/pdf/2511.22076)</summary>

**Abstract:** The Internet of Agents (IoA) is rapidly gaining prominence as a foundational architecture for interconnected intelligent systems, designed to facilitate seamless discovery, communication, and collaborative reasoning among a vast network of Artificial Intelligence (AI) agents. Powered by Large Language and Vision-Language Models, IoA enables the development of interactive, rational agents capable of complex cooperation, moving far beyond traditional isolated models. IoA involves physical entities, i.e., Wireless Agents (WAs) with limited onboard resources, which need to offload their compute-intensive agentic AI services to nearby servers. Such servers can be Mobile Agents (MAs), e.g., vehicle agents, or Fixed Agents (FAs), e.g., end-side units agents. Given their fixed geographical locations and stable connectivity, FAs can serve as reliable communication gateways and task aggregation points. This stability allows them to effectively coordinate with and offload to an Aerial Agent (AA) tier, which has an advantage not affordable for highly mobile MAs with dynamic connectivity limitations. As such, we propose a two-tier optimization approach. The first tier employs a multi-leader multi-follower Stackelberg game. In the game, MAs and FAs act as the leaders who set resource prices. WAs are the followers to determine task offloading ratios. However, when FAs become overloaded, they can further offload tasks to available aerial resources. Therefore, the second tier introduces a Double Dutch Auction model where overloaded FAs act as the buyers to request resources, and AAs serve as the sellers for resource provision. We then develop a diffusion-based Deep Reinforcement Learning algorithm to solve the model. Numerical results demonstrate the superiority of our proposed scheme in facilitating task offloading.

**arXiv ID:** 2511.22076
</details>

<details>
<summary><strong>CXReasonAgent: Evidence-Grounded Diagnostic Reasoning Agent for Chest X-rays</strong> - Hyungyung Lee, Hangyul Yoon, Edward Choi - [[pdf]](https://arxiv.org/pdf/2602.23276)</summary>

**Abstract:** Chest X-ray plays a central role in thoracic diagnosis, and its interpretation inherently requires multi-step, evidence-grounded reasoning. However, large vision-language models (LVLMs) often generate plausible responses that are not faithfully grounded in diagnostic evidence and provide limited visual evidence for verification, while also requiring costly retraining to support new diagnostic tasks, limiting their reliability and adaptability in clinical settings. To address these limitations, we present CXReasonAgent, a diagnostic agent that integrates a large language model (LLM) with clinically grounded diagnostic tools to perform evidence-grounded diagnostic reasoning using image-derived diagnostic and visual evidence. To evaluate these capabilities, we introduce CXReasonDial, a multi-turn dialogue benchmark with 1,946 dialogues across 12 diagnostic tasks, and show that CXReasonAgent produces faithfully grounded responses, enabling more reliable and verifiable diagnostic reasoning than LVLMs. These findings highlight the importance of integrating clinically grounded diagnostic tools, particularly in safety-critical clinical settings. The demo is available \href{this https URL}{here}.

**arXiv ID:** 2602.23276
</details>

<details>
<summary><strong>Agentic AI-based Coverage Closure for Formal Verification</strong> - Sivaram Pothireddypalli, Ashish Raman, Deepak Narayan Gadde, Aman Kumar - [[pdf]](https://arxiv.org/pdf/2603.03147)</summary>

**Abstract:** Coverage closure is a critical requirement in Integrated Chip (IC) development process and key metric for verification sign-off. However, traditional exhaustive approaches often fail to achieve full coverage within project timelines. This study presents an agentic AI-driven workflow that utilizes Large Language Model (LLM)-enabled Generative AI (GenAI) to automate coverage analysis for formal verification, identify coverage gaps, and generate the required formal properties. The framework accelerates verification efficiency by systematically addressing coverage holes. Benchmarking open-source and internal designs reveals a measurable increase in coverage metrics, with improvements correlated to the complexity of the design. Comparative analysis validates the effectiveness of this approach. These results highlight the potential of agentic AI-based techniques to improve formal verification productivity and support comprehensive coverage closure.

**arXiv ID:** 2603.03147
</details>

<details>
<summary><strong>Quality Over Clicks: Intrinsic Quality-Driven Iterative Reinforcement Learning for Cold-Start E-Commerce Query Suggestion</strong> - Qi Sun, Kejun Xiao, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng - [[pdf]](https://arxiv.org/pdf/2603.22922)</summary>

**Abstract:** Existing dialogue systems rely on Query Suggestion (QS) to enhance user engagement. Recent efforts typically employ large language models with Click-Through Rate (CTR) model, yet fail in cold-start scenarios due to their heavy reliance on abundant online click data for effective CTR model training. To bridge this gap, we propose Cold-EQS, an iterative reinforcement learning framework for Cold-Start E-commerce Query Suggestion (EQS). Specifically, we leverage answerability, factuality, and information gain as reward to continuously optimize the quality of suggested queries. To continuously optimize our QS model, we estimate uncertainty for grouped candidate suggested queries to select hard and ambiguous samples from online user queries lacking click signals. In addition, we provide an EQS-Benchmark comprising 16,949 online user queries for offline training and evaluation. Extensive offline and online experiments consistently demonstrate a strong positive correlation between online and offline effectiveness. Both offline and online experimental results demonstrate the superiority of our Cold-EQS, achieving a significant +6.81% improvement in online chatUV.

**arXiv ID:** 2603.22922
</details>

<details>
<summary><strong>Off-Policy Value-Based Reinforcement Learning for Large Language Models</strong> - Peng-Yuan Wang, Ziniu Li, Tian Xu, Bohan Yang, Tian-Shuo Liu, ChenYang Wang, Xiong-Hui Chen, Yi-Chen Li, Tianyun Yang, Congliang Chen, Yang Yu - [[pdf]](https://arxiv.org/pdf/2603.23355)</summary>

**Abstract:** Improving data utilization efficiency is critical for scaling reinforcement learning (RL) for long-horizon tasks where generating trajectories is expensive. However, the dominant RL methods for LLMs are largely on-policy: they update each batch of data only once, discard it, and then collect fresh samples, resulting in poor sample efficiency. In this work, we explore an alternative value-based RL framework for LLMs that naturally enables off-policy learning. We propose ReVal, a Bellman-update-based method that combines stepwise signals capturing internal consistency with trajectory-level signals derived from outcome verification. ReVal naturally supports replay-buffer-based training, allowing efficient reuse of past trajectories. Experiments on standard mathematical reasoning benchmarks show that ReVal not only converges faster but also outperforms GRPO in final performance. On DeepSeek-R1-Distill-1.5B, ReVal improves training efficiency and achieves improvement of 2.7% in AIME24 and 4.5% in out-of-domain benchmark GPQA over GRPO. These results suggest that value-based RL is a practical alternative to policy-based methods for LLM training.

**arXiv ID:** 2603.23355
</details>

<details>
<summary><strong>SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning</strong> - Haoyu Huang, Jinfa Huang, Zhongwei Wan, Xiawu Zheng, Rongrong Ji, Jiebo Luo - [[pdf]](https://arxiv.org/pdf/2603.23483)</summary>

**Abstract:** Agentic multimodal large language models (MLLMs) (e.g., OpenAI o3 and Gemini Agentic Vision) achieve remarkable reasoning capabilities through iterative visual tool invocation. However, the cascaded perception, reasoning, and tool-calling loops introduce significant sequential overhead. This overhead, termed agentic depth, incurs prohibitive latency and seriously limits system-level concurrency. To this end, we propose SpecEyes, an agentic-level speculative acceleration framework that breaks this sequential bottleneck. Our key insight is that a lightweight, tool-free MLLM can serve as a speculative planner to predict the execution trajectory, enabling early termination of expensive tool chains without sacrificing accuracy. To regulate this speculative planning, we introduce a cognitive gating mechanism based on answer separability, which quantifies the model's confidence for self-verification without requiring oracle labels. Furthermore, we design a heterogeneous parallel funnel that exploits the stateless concurrency of the small model to mask the stateful serial execution of the large model, maximizing system throughput. Extensive experiments on V* Bench, HR-Bench, and POPE demonstrate that SpecEyes achieves 1.1-3.35x speedup over the agentic baseline while preserving or even improving accuracy (up to +6.7%), thereby boosting serving throughput under concurrent workloads.

**arXiv ID:** 2603.23483
</details>

<details>
<summary><strong>Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents</strong> - Guoqing Wang, Sunhao Dai, Guangze Ye, Zeyu Gan, Wei Yao, Yong Deng, Xiaofeng Wu, Zhenzhe Ying - [[pdf]](https://arxiv.org/pdf/2510.14967)</summary>

**Abstract:** Large language model (LLM)-based agents are increasingly trained with reinforcement learning (RL) to enhance their ability to interact with external environments through tool use, particularly in search-based settings that require multi-turn reasoning and knowledge acquisition. However, existing approaches typically rely on outcome-based rewards that are only provided exclusively upon generating the final answer. This reward sparsity becomes particularly problematic in multi-turn settings, where long trajectories exacerbate three critical issues: (i) advantage collapse, where all rollouts receive identical rewards and provide no useful learning signals; (ii) lack of fine-grained credit assignment, where the correctness of intermediate turns is obscured, especially in long-horizon tasks; and (iii) poor sample efficiency, where each rollout yields only a single outcome signal, leading to low data utilization. In this paper, we propose Information Gain-based Policy Optimization (IGPO), a simple yet effective RL framework that provides dense and intrinsic supervision for multi-turn agent training. IGPO models each interaction turn as an incremental process of acquiring information about the ground truth, and defines turn-level rewards as the marginal increase in the policy's probability of producing the correct answer. Unlike prior process-level reward approaches that depend on external reward models or costly Monte Carlo estimation, IGPO derives intrinsic rewards directly from the model's own belief updates. These intrinsic turn-level rewards are combined with outcome-level supervision to form dense reward signals. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that IGPO consistently outperforms strong baselines in multi-turn scenarios, achieving higher accuracy and improved data efficiency. Our code is available at this https URL.

**arXiv ID:** 2510.14967
</details>

<details>
<summary><strong>Model Predictive Control with Differentiable World Models for Offline Reinforcement Learning</strong> - Rohan Deb, Stephen J. Wright, Arindam Banerjee - [[pdf]](https://arxiv.org/pdf/2603.22430)</summary>

**Abstract:** Offline Reinforcement Learning (RL) aims to learn optimal policies from fixed offline datasets, without further interactions with the environment. Such methods train an offline policy (or value function), and apply it at inference time without further refinement. We introduce an inference time adaptation framework inspired by model predictive control (MPC) that utilizes a pretrained policy along with a learned world model of state transitions and rewards. While existing world model and diffusion-planning methods use learned dynamics to generate imagined trajectories during training, or to sample candidate plans at inference time, they do not use inference-time information to optimize the policy parameters on the fly. In contrast, our design is a Differentiable World Model (DWM) pipeline that enables endto-end gradient computation through imagined rollouts for policy optimization at inference time based on MPC. We evaluate our algorithm on D4RL continuous-control benchmarks (MuJoCo locomotion tasks and AntMaze), and show that exploiting inference-time information to optimize the policy parameters yields consistent gains over strong offline RL baselines.

**arXiv ID:** 2603.22430
</details>

<details>
<summary><strong>VLGOR: Visual-Language Knowledge Guided Offline Reinforcement Learning for Generalizable Agents</strong> - Pengsen Liu, Maosen Zeng, Nan Tang, Kaiyuan Li, Jing-Cheng Pang, Yunan Liu, Yang Yu - [[pdf]](https://arxiv.org/pdf/2603.22892)</summary>

**Abstract:** Combining Large Language Models (LLMs) with Reinforcement Learning (RL) enables agents to interpret language instructions more effectively for task execution. However, LLMs typically lack direct perception of the physical environment, which limits their understanding of environmental dynamics and their ability to generalize to unseen tasks. To address this limitation, we propose Visual-Language Knowledge-Guided Offline Reinforcement Learning (VLGOR), a framework that integrates visual and language knowledge to generate imaginary rollouts, thereby enriching the interaction data. The core premise of VLGOR is to fine-tune a vision-language model to predict future states and actions conditioned on an initial visual observation and high-level instructions, ensuring that the generated rollouts remain temporally coherent and spatially plausible. Furthermore, we employ counterfactual prompts to produce more diverse rollouts for offline RL training, enabling the agent to acquire knowledge that facilitates following language instructions while grounding in environments based on visual cues. Experiments on robotic manipulation benchmarks demonstrate that VLGOR significantly improves performance on unseen tasks requiring novel optimal policies, achieving a success rate over 24% higher than the baseline methods.

**arXiv ID:** 2603.22892
</details>

<details>
<summary><strong>Privacy-Preserving Reinforcement Learning from Human Feedback via Decoupled Reward Modeling</strong> - Young Hyun Cho, Will Wei Sun - [[pdf]](https://arxiv.org/pdf/2603.22563)</summary>

**Abstract:** Preference-based fine-tuning has become an important component in training large language models, and the data used at this stage may contain sensitive user information. A central question is how to design a differentially private pipeline that is well suited to the distinct structure of reinforcement learning from human feedback. We propose a privacy-preserving framework that imposes differential privacy only on reward learning and derives the final policy from the resulting private reward model. Theoretically, we study the suboptimality gap and show that privacy contributes an additional additive term beyond the usual non-private statistical error. We also establish a minimax lower bound and show that the dominant term changes with sample size and privacy level, which in turn characterizes regimes in which the upper bound is rate-optimal up to logarithmic factors. Empirically, synthetic experiments confirm the scaling predicted by the theory, and experiments on the Anthropic HH-RLHF dataset using the Gemma-2B-IT model show stronger private alignment performance than existing differentially private baseline methods across privacy budgets.

**arXiv ID:** 2603.22563
</details>

<details>
<summary><strong>CellFluxRL: Biologically-Constrained Virtual Cell Modeling via Reinforcement Learning</strong> - Dongxia Wu, Shiye Su, Yuhui Zhang, Elaine Sui, Emma Lundberg, Emily B. Fox, Serena Yeung-Levy - [[pdf]](https://arxiv.org/pdf/2603.21743)</summary>

**Abstract:** Building virtual cells with generative models to simulate cellular behavior in silico is emerging as a promising paradigm for accelerating drug discovery. However, prior image-based generative approaches can produce implausible cell images that violate basic physical and biological constraints. To address this, we propose to post-train virtual cell models with reinforcement learning (RL), leveraging biologically meaningful evaluators as reward functions. We design seven rewards spanning three categories-biological function, structural validity, and morphological correctness-and optimize the state-of-the-art CellFlux model to yield CellFluxRL. CellFluxRL consistently improves over CellFlux across all rewards, with further performance boosts from test-time scaling. Overall, our results present a virtual cell modeling framework that enforces physically-based constraints through RL, advancing beyond "visually realistic" generations towards "biologically meaningful" ones.

**arXiv ID:** 2603.21743
</details>

<details>
<summary><strong>Path Planning and Reinforcement Learning-Driven Control of On-Orbit Free-Flying Multi-Arm Robots</strong> - Álvaro Belmonte-Baeza, José Luis Ramón, Leonard Felicetti, Miguel Cazorla, Jorge Pomares - [[pdf]](https://arxiv.org/pdf/2603.23182)</summary>

**Abstract:** This paper presents a hybrid approach that integrates trajectory optimization (TO) and reinforcement learning (RL) for motion planning and control of free-flying multi-arm robots in on-orbit servicing scenarios. The proposed system integrates TO for generating feasible, efficient paths while accounting for dynamic and kinematic constraints, and RL for adaptive trajectory tracking under uncertainties. The multi-arm robot design, equipped with thrusters for precise body control, enables redundancy and stability in complex space operations. TO optimizes arm motions and thruster forces, reducing reliance on the arms for stabilization and enhancing maneuverability. RL further refines this by leveraging model-free control to adapt to dynamic interactions and disturbances. The experimental results validated through comprehensive simulations demonstrate the effectiveness and robustness of the proposed hybrid approach. Two case studies are explored: surface motion with initial contact and a free-floating scenario requiring surface approximation. In both cases, the hybrid method outperforms traditional strategies. In particular, the thrusters notably enhance motion smoothness, safety, and operational efficiency. The RL policy effectively tracks TO-generated trajectories, handling high-dimensional action spaces and dynamic mismatches. This integration of TO and RL combines the strengths of precise, task-specific planning with robust adaptability, ensuring high performance in the uncertain and dynamic conditions characteristic of space environments. By addressing challenges such as motion coupling, environmental disturbances, and dynamic control requirements, this framework establishes a strong foundation for advancing the autonomy and effectiveness of space robotic systems.

**arXiv ID:** 2603.23182
</details>

<details>
<summary><strong>Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning</strong> - Patrick Yin, Tyler Westenbroek, Zhengyu Zhang, Joshua Tran, Ignacio Dagnino, Eeshani Shilamkar, Numfor Mbiziwo-Tiapo, Simran Bagaria, Xinlei Liu, Galen Mullins, Andrey Kolobov, Abhishek Gupta - [[pdf]](https://arxiv.org/pdf/2603.15789)</summary>

**Abstract:** Reinforcement learning in massively parallel physics simulations has driven major progress in sim-to-real robot learning. However, current approaches remain brittle and task-specific, relying on extensive per-task engineering to design rewards, curricula, and demonstrations. Even with this engineering, they often fail on long-horizon, contact-rich manipulation tasks and do not meaningfully scale with compute, as performance quickly saturates when training revisits the same narrow regions of state space. We introduce OmniReset, a simple and scalable framework that enables on-policy reinforcement learning to robustly solve a broad class of dexterous manipulation tasks using a single reward function, fixed algorithm hyperparameters, no curricula, and no human demonstrations. Our key insight is that long-horizon exploration can be dramatically simplified by using simulator resets to systematically expose the RL algorithm to the diverse set of robot-object interactions which underlie dexterous manipulation. OmniReset programmatically generates such resets with minimal human input, converting additional compute directly into broader behavioral coverage and continued performance gains. We show that OmniReset gracefully scales to long-horizon dexterous manipulation tasks beyond the capabilities of existing approaches and is able to learn robust policies over significantly wider ranges of initial conditions than baselines. Finally, we distill OmniReset into visuomotor policies which display robust retrying behavior and substantially higher success rates than baselines when transferred to the real world zero-shot. Project webpage: this https URL

**arXiv ID:** 2603.15789
</details>

<details>
<summary><strong>Energy-Aware Reinforcement Learning for Robotic Manipulation of Articulated Components in Infrastructure Operation and Maintenance</strong> - Xiaowen Tao, Yinuo Wang, Haitao Ding, Yuanyang Qi, Ziyu Song - [[pdf]](https://arxiv.org/pdf/2602.12288)</summary>

**Abstract:** With the growth of intelligent civil infrastructure and smart cities, operation and maintenance (O&M) increasingly requires safe, efficient, and energy-conscious robotic manipulation of articulated components, including access doors, service drawers, and pipeline valves. However, existing robotic approaches either focus primarily on grasping or target object-specific articulated manipulation, and they rarely incorporate explicit actuation energy into multi-objective optimisation, which limits their scalability and suitability for long-term deployment in real O&M settings. Therefore, this paper proposes an articulation-agnostic and energy-aware reinforcement learning framework for robotic manipulation in intelligent infrastructure O&M. The method combines part-guided 3D perception, weighted point sampling, and PointNet-based encoding to obtain a compact geometric representation that generalises across heterogeneous articulated objects. Manipulation is formulated as a Constrained Markov Decision Process (CMDP), in which actuation energy is explicitly modelled and regulated via a Lagrangian-based constrained Soft Actor-Critic scheme. The policy is trained end-to-end under this CMDP formulation, enabling effective articulated-object operation while satisfying a long-horizon energy budget. Experiments on representative O&M tasks demonstrate 16%-30% reductions in energy consumption, 16%-32% fewer steps to success, and consistently high success rates, indicating a scalable and sustainable solution for infrastructure O&M manipulation.

**arXiv ID:** 2602.12288
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
