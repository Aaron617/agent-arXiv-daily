# Agent arXiv Daily

**Last Updated:** 2026-07-13 04:40:59

**Total Papers:** 32

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (1 papers)</h2></summary>

<details>
<summary><strong>SolarChain-Eval: A Physics-Constrained Benchmark for Trustworthy Economic Agents in Decentralized Energy Markets</strong> - Shilin Ou, Yifan Xu, Luyao Zhang - [[pdf]](https://arxiv.org/pdf/2607.08681)</summary>

**Abstract:** As agentic AI systems are increasingly applied to cyber-physical environments, their evaluation requires assessment of both task performance and trustworthiness. In decentralized energy markets, autonomous agents may improve market utility, but may also exploit invalid physical data, create artificial liquidity, and produce unstable governance decisions. Therefore, we propose SolarChain-Eval, a physics-constrained benchmark for evaluating trustworthy economic agents. It formulates market governance as a Gymnasium-compatible Markov Decision Process, where agents make hourly decisions. SolarChain-Eval evaluates each policy across multiple dimensions, including market utility, physical safety, slippage, action smoothness, spatial fairness, and auditability. To support agentic evaluation, SolarChain-Eval incorporates an LLM-based Planner/Auditor layer. The Planner defines episode-level action bounds and audit rules, while the Auditor reviews and revises high-risk actions. All interventions are recorded through structured logs, including trigger signals, proposed actions, revised actions, and audit rationales. Experiments with static, random, myopic, RL, and RL+LLM policies reveal a clear utility-safety trade-off. RL agents improve market utility but can still produce unsafe behavior. When the physics penalty is removed, reward-maximizing agents exploit invalid generation and increase artificial liquidity. The LLM Planner/Auditor improves auditability and mitigates selected risks, but it cannot fully compensate for a misspecified reward function. These results indicate that trustworthy agentic AI evaluation requires both physical constraints and transparent intervention traces. We release data and code as open access on GitHub for replicability.

**arXiv ID:** 2607.08681
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (5 papers)</h2></summary>

<details>
<summary><strong>OpenProver: Agentic and Interactive Theorem Proving with Lean 4</strong> - Matěj Kripner, Milan Straka - [[pdf]](https://arxiv.org/pdf/2607.09217)</summary>

**Abstract:** In this system paper, we present OpenProver, an open-source system for LLM-driven automated theorem proving (ATP) with integrated Lean 4 formal verification. OpenProver integrates a Planner-Worker-Verifier architecture inspired by recent ATP agentic systems such as Aletheia. A Planner agent maintains a compact Whiteboard scratchpad and an unbounded Repository of intermediate findings, and decomposes mathematical work into parallel Workers.
OpenProver is fully open-source, offers reproducible evaluation through automatic formal verification of generated proofs, and provides an interactive terminal interface for human-guided proof search. In interactive mode, OpenProver allows the human operator to monitor and steer the proof search process, motivated by the established human-AI synergy in interactive code generation.
To showcase the potential for quantitative ablation experiments enabled by automatic formal verification, we evaluate OpenProver on ProofNet and compare it with a simple baseline. OpenProver is publicly available at this https URL.

**arXiv ID:** 2607.09217
</details>

<details>
<summary><strong>LongMedBench: Benchmarking Medical Agents for Long-Horizon Clinical Decision-Making</strong> - Yanzhen Chen, Zihan Xu, Xiaocheng Zhang, Zhiting Fan, Weiqi Zhai, Hongxia Xu, Zuozhu Liu - [[pdf]](https://arxiv.org/pdf/2607.09322)</summary>

**Abstract:** In this work, we introduce LongMedBench, a real-world EHR-based benchmark for long-horizon clinical decision-making. Prior evaluations of LLM-based medical agents have largely emphasized short-context knowledge QA and tool use. However, real-world medical care is inherently longitudinal, and clinicians must aggregate evidence across repeated visits, tests, and evolving treatments. Therefore, long-horizon interaction is essential for realistic assessment. LongMedBench is constructed via a reproducible pipeline that integrates MIMIC-IV admission records and clinical notes into time-series event streams and long-context memory datasets, enabling long-horizon, multi-session interactions between agents and a clinical environment. It comprises 335 patients, with 19.72 inpatient visits per patient on average and 44.91 medical events per visit. Guided by the long-horizon decision process, we propose an evaluation taxonomy with three suites: fact-based QA, temporal reasoning, and long-horizon decision-making. This taxonomy measures how agents understand and leverage historical patient information over extended horizons. Our experiments show that while recent LLMs can make good use of explicit timestamps, they have challenges in implicit time inference; The RAG and agent memory system can improve the performance of information retrieval tasks, but the performance of decision-making tasks is highly dependent on the model's immediate context.

**arXiv ID:** 2607.09322
</details>

<details>
<summary><strong>SCATE: Learning to Supervise Coding Agents for Cost-Effective Test Generation</strong> - Sijia Gu, Noor Nashid, Ali Mesbah - [[pdf]](https://arxiv.org/pdf/2607.08983)</summary>

**Abstract:** While autonomous coding agents have significantly advanced automated test generation, they remain fundamentally limited by lazy generation, a phenomenon where agents prematurely terminate tasks and systematically avoid complex programmatic logic, resulting in inadequate code coverage. Currently, mitigating this premature termination requires continuous human-in-the-loop supervision. This heavy reliance on human intuition creates a bottleneck that negates the efficiency gains of automated generation. We propose SCATE, a framework for adaptive, automated supervision of coding agents that replaces human intervention during test generation. By formulating supervision as a contextual bandit problem, SCATE learns to select the most promising testing actions based on the current coverage and class testability metrics, maximizing coverage gains while minimizing wasted generation effort. Our empirical evaluation demonstrates that SCATE integrates seamlessly with different coding agents. When applied to GEMINI-CLI, it achieves 32.3% higher line coverage and 30.9% higher branch coverage than the agent-only baseline. A comparison with CLAUDE CODE confirms the framework dynamically adapts its policy to optimize each agent's unique strengths. SCATE also consistently outperforms state-of-the-art non-agentic approaches across all metrics.

**arXiv ID:** 2607.08983
</details>

<details>
<summary><strong>AgentKGV: Agentic LLM-RAG Framework with Two-Stage Training for the Fact Verification of Knowledge Graphs</strong> - Yumin Heo, Hyeon-gu Lee, Sumin Seo, Youngjoong Ko - [[pdf]](https://arxiv.org/pdf/2607.09092)</summary>

**Abstract:** Knowledge graphs (KGs) are often automatically constructed from large-scale corpora, but they inevitably contain factual errors due to noisy sources and extraction failures, and verifying them reliably at industrial scale remains a critical challenge. To address this, we propose AgentKGV, the Agentic LLM-RAG framework for KG fact Verification, that integrates dynamic routing and iterative query rewriting, which handles surface-form mismatch in document-level retrieval. To make this framework more accurate and cost-efficient for industrial deployment, we further introduce a two-stage training strategy: turn-level distillation-based SFT that transfers reasoning ability from a large teacher model into a small model for stable query rewriting and reasoning, and trajectory-level GRPO that optimizes the search policy to reduce unnecessary retrieval at scale. On the long-tail-predicate split of the open-domain T-REx benchmark, our framework improves macro-F1 over single-turn RAG by 5.5 \%p, and two-stage training does it further by 9.4 \%p. GRPO also cuts the average number of search calls from 3.24 to 1.63 without lowering accuracy.

**arXiv ID:** 2607.09092
</details>

<details>
<summary><strong>Task-Specific Multimodal Question Answering Agents via Confidence Calibration and Incremental Reasoning for QANTA 2026</strong> - Nirjhar Das, Md. Al-Mamun Provath - [[pdf]](https://arxiv.org/pdf/2607.09623)</summary>

**Abstract:** We present our submission to the QANTA 2026 shared challenge at the ICML 2026 Workshop on Efficient Multimodal Question Answering (EMM-QA). Quanta evaluates multimodal quizbowl systems that answer pyramid-style questions from incrementally revealed text and accompanying images while operating under realistic efficiency constraints. The challenge consists of two distinct tasks: Tossup questions, which require deciding when to answer under uncertainty, and Bonus questions, which emphasize accurate answer selection and human adoption. To address these differing objectives, we develop a task-specific two-agent architecture. Our Tossup agent utilizes a GPT-4o-mini-class model (referred to as GPT-4.1-mini in the competition logs) with confidence-calibrated answering and a domain-specific numeric reasoning policy that reduces overconfident predictions from isolated quantitative clues. Our Bonus agent uses GPT-4o-class model (referred to as GPT-4.1) with leadin-aware reasoning, structured relational reasoning, and multimodal evidence integration to improve exact answer selection. Rather than relying on a retrieval pipeline or model ensembles, our approach emphasizes efficient reasoning policies and confidence calibration within a hosted-only environment. Our system achieved the highest overall leaderboard score of 0.402, including a Tossup score of 0.238 and a Bonus Effect score of 0.164. The results demonstrate that lightweight, task-specific reasoning strategies can provide strong performance on resource-constrained multimodal question answering benchmarks.

**arXiv ID:** 2607.09623
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>Scoped Verification for Reliable Long-Horizon Agentic Context Evolution under Distribution Shift</strong> - Dan C. Hsu, Luke Lu - [[pdf]](https://arxiv.org/pdf/2607.09175)</summary>

**Abstract:** Deployed LLM agents rely on agentic context, the model-external textual control content assembled by an operational harness. In this work, the mutable component of that context is a persistent system-level instruction that is updated from operational experience while the model, tools, and harness remain fixed. Over long evolution horizons, flat-text maintenance makes verification increasingly difficult as accumulated instructions grow and interact. We propose Graph-Regularized Agentic Context Evolution (GRACE), which maintains the persistent instruction component as a typed semantic graph and validates proposed updates within the local typed neighborhoods of modified nodes. Accepted graph updates are reconstructed as incremental edits to the textual instruction checkpoint used at deployment. We evaluate GRACE within a fixed telecom agent harness derived from $\tau^2$-bench under a controlled distribution-shift protocol. Across five independent replications, GRACE improves strict reliability, measured by pass^3, from the Gemini 2.5 Flash zero-shot value of 0.091 to 0.673$\pm$0.136 at the final checkpoint. This exceeds a Gemini 3.1 Pro zero-shot reference of 0.242 on the same held-out set, while the flat-text HCE baseline finishes at 0.191$\pm$0.051. These results identify two requirements for reliable long-horizon context evolution, a structural substrate that makes verification local and a consolidation mechanism that keeps accumulated instruction content usable.

**arXiv ID:** 2607.09175
</details>

<details>
<summary><strong>Toward Auditable AI Scientists: A Hypothesis Evolution Protocol for LLM Agents</strong> - Izumi Takahara, Teruyasu Mizoguchi - [[pdf]](https://arxiv.org/pdf/2607.09195)</summary>

**Abstract:** Large language model (LLM) agents are increasingly expected to play a central role in AI-driven scientific discovery. Equipped with broad knowledge, flexible reasoning, and tool use, they have the potential to autonomously explore and solve scientific problems by repeatedly proposing hypotheses, testing them, and revising their beliefs in the light of the evidence. In current agents, however, these hypotheses, tests, and belief updates are buried in unstructured logs, and no mechanism lets the agent or the human researcher audit that process. Here we propose the Hypothesis Evolution Protocol (HEP), an agent harness that provides hypothesis generation, evaluation, and evolution as explicit, auditable operations. On materials-science research tasks, a HEP-equipped agent operates the hypothesis--test--evidence--belief cycle that planning-style agents lack, generalizes across research questions, and exploits the protocol more fully as the base LLM becomes more capable. These results mark a step toward auditable AI scientists, whose scientific reasoning can be inspected, verified, and built upon.

**arXiv ID:** 2607.09195
</details>

<details>
<summary><strong>ProofCouncil: An LLM Agent for Solving Open Mathematical Problems</strong> - Johannes Schmitt, Tim Gehrunger, Jasper Dekoninck, Gergely Bérczi, Uri Kreitner, Liam Price, David Holmes - [[pdf]](https://arxiv.org/pdf/2607.09474)</summary>

**Abstract:** Large language models (LLMs) have shown increasing promise in solving open problems in mathematics. However, their performance can be further improved through agentic workflows tailored to real-world mathematical practice. To this end, we introduce ProofCouncil, a mathematical agent that is designed to tackle open problems using an author-critic architecture. ProofCouncil served as a submission to the second batch of FirstProof, a challenge consisting of 10 real-world mathematical problems that agents must solve autonomously. Its submissions for 6 of the 10 problems were judged by the referees to be correct up to at most minor revisions, showing the best performance among participating teams. We also evaluate ProofCouncil on 30 open problems collected from mathematical researchers. Among the 21 solutions that received human feedback, 5 were judged completely correct, 2 more were judged promising pending final verification, and a further 8 contained useful partial progress. In this short paper, we describe the development of ProofCouncil and the agent-building library used to create it, which we release as open source to the community.

**arXiv ID:** 2607.09474
</details>

<details>
<summary><strong>Shared Selective Persistent Memory for Agentic LLM Systems</strong> - Sanjana Pedada, Aditya Dhavala, Neelraj Patil - [[pdf]](https://arxiv.org/pdf/2607.09493)</summary>

**Abstract:** Agentic LLM systems that generate code through multi-turn tool use face a fundamental context problem: each session starts from zero, discarding the configuration choices, domain constraints, data schemas, and tool-use patterns that made previous sessions productive. Naively persisting entire conversation histories is token-inefficient and counterproductive: irrelevant context degrades generation quality. We introduce shared selective persistent memory, an architecture that identifies and retains four categories of reusable context (task specifications, data schemas, tool configurations, and output constraints) while discarding session-specific reasoning traces. Crucially, this memory is shared: workspaces encapsulating selective memory can be transferred across users with role-based access control, enabling collaborative reuse without redundant specification. We implement it in a deployed collaborative workspace platform where LLM agents produce, edit, and maintain git-versioned artifacts (dashboards, reports, and data-driven documents) from heterogeneous sources (CSV, SQL, REST APIs, and MCP servers). A complementary zero-token data refresh mechanism decouples generated programs from runtime data, enabling artifact reuse without re-invocation. Across three enterprise scenarios, shared selective persistent memory achieves 96% task completion (vs. 79% without memory and 71% with full history). Zero-token refresh eliminates LLM re-invocation for recurring updates (14x task-time reduction), while summary-driven generation cuts per-invocation token cost by 97x versus raw data injection. A replication on four public datasets confirms generalizability, with zero-token refresh succeeding in 12/12 trials. Notably, naive full-history persistence actively degrades completion by biasing the agent with stale traces, while selective memory outperforms both extremes.

**arXiv ID:** 2607.09493
</details>

<details>
<summary><strong>Agora: Enhancing LLM Agent Reasoning Via Auction-Based Task Allocation</strong> - Kaiji Zhou, Ales Leonardis, Yue Feng - [[pdf]](https://arxiv.org/pdf/2607.09600)</summary>

**Abstract:** Enhancing the reasoning capabilities of large language model (LLM) agents requires effective orchestration of diverse expert models and tools. However, existing frameworks typically call APIs based on coarse-grained matching between tasks and the functions of expert models or tools, while overlooking critical factors such as performance variability and cost efficiency among functionally similar alternatives. To address this, we propose Agora, a framework that introduces an incentive-compatible auction mechanism for dynamically allocating tasks to expert models and tools. By treating reasoning steps as tradeable items, Agora enables agents to bid based on their rectified competence-ensuring that critical logic is routed to the most capable solver rather than the most overconfident one. Evaluations across five benchmarks show that Agora improves over matched single-model, routing, and cascade baselines under comparable candidate pools, while exposing a controllable cost-quality trade-off through a single auction parameter.

**arXiv ID:** 2607.09600
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (13 papers)</h2></summary>

<details>
<summary><strong>ARCANA: A Reflective Multi-Agent Program Synthesis Framework for ARC-AGI-2 Reasoning</strong> - Kunbo Zhang, Lei Fu, Zeyu Wang, Zijing Liu, Kejian Tong - [[pdf]](https://arxiv.org/pdf/2607.09059)</summary>

**Abstract:** We present ARCANA, a collaborative multi agent framework for solving ARC AGI 2 tasks under strict test time and hardware constraints. ARCANA decomposes each task into iterative perception, hypothesis generation, symbolic execution, and reflective refinement. A perceptual grounding agent builds object centric scene graphs from raw grids, a latent program policy proposes diverse DSL programs, a symbolic executor verifies candidates on demonstrations, and a reflective agent synthesizes failure driven feedback for the next turn. These agents communicate through a shared differentiable blackboard and are scheduled by a learned meta controller. The design combines structured program search with adaptive multi turn correction, improving reasoning efficiency and solution quality on challenging abstract transformation tasks.

**arXiv ID:** 2607.09059
</details>

<details>
<summary><strong>L-MAD: A Systematic Evaluation of Multi-Agent Debate Structures in Legal Reasoning</strong> - Tan-Minh Nguyen, Hoang-Trung Nguyen, Huu-Dong Nguyen, Dinh-Truong Do, Thi-Hai-Yen Vuong, Le-Minh Nguyen - [[pdf]](https://arxiv.org/pdf/2607.09099)</summary>

**Abstract:** While multi-agent debate (MAD) frameworks have shown significant potential in general reasoning, their effectiveness in highly structured, knowledge-heavy legal domains remains under-explored. In this work, we introduce the Legal Multi-Agent Debate (L-MAD) framework to systematically evaluate different debate structures and aggregation methods within Legal Textual Entailment. By assigning distinct expert personas to multiple agents, L-MAD improves upon strong single-agent baselines by up to 8\%. Furthermore, analyzing how debate scales reveals a clear trade-off: increasing the agent population reduces inconsistency and improves accuracy, whereas extending discussion rounds induces a detrimental \textit{over-deliberation drift} where agents reinforce each other's mistakes. Ultimately, our findings outline the practical boundaries and safety margins of deploying collaborative multi-agent systems in high-stakes legal reasoning environments.

**arXiv ID:** 2607.09099
</details>

<details>
<summary><strong>KV-PRM: Efficient Process Reward Modeling via KV-Cache Transfer for Multi-Agent Test-Time Scaling</strong> - Peng Kuang, Haibo Jin, Xiaoyu Han, Yanli Wang, Xiaopeng Yuan, Ye Yu, Kaidi Xu, Haohan Wang - [[pdf]](https://arxiv.org/pdf/2607.09153)</summary>

**Abstract:** Process Reward Models (PRMs) have been proven to be highly effective in guiding test-time scaling (TTS) methods, which significantly boost the capabilities of LLM-based multi-agent systems. However, existing PRMs are text-based: they re-encode the entire trajectory text from scratch. In long multi-agent rollouts, the scoring cost, growing quadratically with respect to sequence length L, creates a severe computational bottleneck, severely limiting PRMs' application in long-context scenarios. To resolve this, we introduce KV-PRM, a highly efficient process reward model that eliminates the heavy text re-encoding by directly reading the KV cache produced naturally during the LLM's generation phase. By processing a single "verify token" against the pre-existing KV cache, KV-PRM reduces the scoring cost from O(L^2) to O(L). We formally prove that the KV cache contains strictly greater information capacity than text, and is more efficient for downstream reward modeling. Empirically, across the MATH, GSM8K, and AIME benchmarks, KV-PRM matches or strictly outperforms text-PRMs under various TTS methods such as Beam Search, MCTS, and Weighted Voting, with up to a 5,000x reduction in scoring FLOPs, a 37x reduction in latency, and a 34x reduction in per-sequence memory footprint compared to text-based PRMs.

**arXiv ID:** 2607.09153
</details>

<details>
<summary><strong>Fictional Worldbuilding: Multi-Agent LLM Collaboration with Hierarchical Context Compression and Iterative Review</strong> - Jingbo Chen, He Wang, Wei Yuan, Yuqiao Lai, Zhenyan Lu - [[pdf]](https://arxiv.org/pdf/2607.09403)</summary>

**Abstract:** Worldbuilding, the construction of coherent fictional worlds, is a foundational task in game design and literary creation. Large Language Models (LLMs) offer new possibilities for automated content generation, but their application to worldbuilding faces three challenges: context explosion that grows linearly with the building process, the tension between creative diversity and content consistency, and the absence of automated quality assurance. This paper presents AutoWorldBuilder, a multi-agent collaborative system that addresses these challenges through five integrated components: a structured concept network with conflict detection; a DAG-based hybrid batch scheduler that groups tasks by semantic locality; a four-layer context compression mechanism achieving approximately 90% token reduction; an iterative review system with specialized Auditor agents that improves proposal pass rates from 42% to over 85%; and a skill-driven agent architecture supporting zero-code extension with differentiated temperature configuration. Two experiments across 20 diverse worldbuilding tasks, using GPT-OSS 120B and DeepSeek v3.2 as LLM backends, demonstrate a 95.0% success rate. The system generated 56-103 self-consistent concepts per world in 18-31 minutes with zero-conflict delivery. The architectural patterns validated here, including layer-as-budget compression, semantic-locality scheduling, and separation of generation and review, transfer to the broader class of knowledge-intensive, multi-agent LLM applications.

**arXiv ID:** 2607.09403
</details>

<details>
<summary><strong>Eluna: An Agentic LLM System for Automating Warehouse Operations with Reasoning and Task Execution</strong> - Ning Liu, Kalle Kujanpää, Zhaoxuan Zhu, P Aditya Sreekar, Kaiwen Liu, Chuanneng Sun, Jorge Marchena Menendez, Matthew Bales, Tianyu Yang, Shahnawaz Alam, Rose Yu, Baoyuan Liu, Kristina Klinkner, Shervin Malmasi - [[pdf]](https://arxiv.org/pdf/2607.08960)</summary>

**Abstract:** Warehouse operations are governed by Standard Operating Procedures (SOPs) that encode complex, multi-system decision logic, which must be executed reliably under strict time constraints, yet LLM agents lack mechanisms to enforce procedural compliance and degrade under the context overload full SOP specifications introduce. We present Eluna, a production-deployed agentic system for reliable SOP execution. Eluna is a graph-guided, multi-agent framework that encodes SOPs as directed acyclic graphs with progressive disclosure and delegates independent tasks to parallel sub-agents, each with persistent code execution and live data access. To meet production latency and accuracy needs, we use asymmetric episodic distillation where a strong teacher is improved through episodic error memories, then a smaller student is fine-tuned on the corrected trajectories with memory stripped, internalizing corrections without inference-time overhead. On a 13-task benchmark and two production applications, our fine-tuned models match or exceed their teacher, beat all larger off-the-shelf baselines, and reach 94% expert agreement on the ticket processing application.

**arXiv ID:** 2607.08960
</details>

<details>
<summary><strong>Mosaic: Runtime-Efficient Multi-Agent Embodied Planning</strong> - Kunjal Panchal, Saayan Mitra, Sunav Choudhary, Victor Bursztyn, Somdeb Sarkhel, Hui Guan - [[pdf]](https://arxiv.org/pdf/2607.09603)</summary>

**Abstract:** LLM-based multi-agent embodied planning remains impractical due to prohibitively high execution latency. We identify failed actions as the dominant bottleneck, stemming from two core challenges: inaccurate state tracking under partial observability and inefficient coordination that produces redundant or conflicting actions. We introduce Mosaic, a runtime-efficient multi-agent planning framework that addresses both challenges. Mosaic maintains accurate yet lightweight state tracking through agent-centric semantic memory that stores objects in relative coordinates, enabling geometric transformations and coordination. It ensures efficient coordination through Integer Linear Programming that allocates actions at every planning step, enforcing physical feasibility and inter-agent coordination constraints. Across AI2-THOR and search-and-rescue benchmarks, Mosaic achieves 27-32% faster execution, 30-33% fewer LLM calls, 25-31% fewer steps, and 4-10% points higher success rates. These results demonstrate that efficient memory and constraint-guided coordination are critical for scalable, low-latency multi-agent planning.

**arXiv ID:** 2607.09603
</details>

<details>
<summary><strong>Offline Nash Solvers Meet Online Tree Search in Multi-Agent Games on Graphs</strong> - Mukesh Kumar, Yue Guan, Panagiotis Tsiotras - [[pdf]](https://arxiv.org/pdf/2607.08892)</summary>

**Abstract:** Computing Nash equilibrium policies in multi-agent Pursuit-Evasion games (PEG) is challenging due to the exponential growth of the joint state and action spaces with the number of agents. Existing approaches either rely on offline equilibrium approximations, which may lack adaptability during execution, or online planning methods, which suffer from large branching factors. In this work, we propose Primitive-Guided Tree Search (PGTS), a hybrid framework that integrates offline exact Nash equilibrium computation with online tree search: PGTS first solves a collection of smaller, tractable sub-games offline; at deployment, PGTS performs online tree search at each time step, using the optimal sub-game policies and value functions to guide tree expansion and estimate leaf-node values. Extensive experiments on varied graph topologies, including real-world networks, demonstrate that PGTS significantly outperforms state-of-the-art learning and heuristic baselines, while maintaining robust performance against adversaries.

**arXiv ID:** 2607.08892
</details>

<details>
<summary><strong>Secret Scanner Agent: Extracting Secrets and Access Context from Unstructured Documents</strong> - Zixiao Chen, Mariko Wakabayashi, Charlotte Siska - [[pdf]](https://arxiv.org/pdf/2607.09011)</summary>

**Abstract:** Exposed documents such as emails, chat threads, tickets, and incident notes routinely leak credentials, but during incident response a leaked secret is only half the story. Responders also need to identify the ``door'' the secret opens: the account, tenant, endpoint, database, cloud resource, or other system that the credential could allow an attacker to access. Traditional secret scanners rely on regular expressions or trained classifiers which work well on well-formatted code, yet they struggle when a credential is fragmented, reformatted, or far from the resource it unlocks, and they report the secret string without naming what it opens. We present Secret Scanner Agent (SSA), a multi-agent large-language-model system that extracts both the secret and its associated door, together with supporting evidence, from unstructured exposed documents. SSA pairs a detection agent that favors recall with a review agent that filters false positives and recovers missing context. Because real credential data is sensitive, we evaluate SSA on synthetic benchmarks we generated that span 23 secret types and multiple document formats, scored with a three-step pipeline of programmatic matching, an LLM judge, and human review. Across six models, multi-agent SSA improves extraction precision over a single-agent variant, with the largest gains on door extraction, by up to 16 percentage points. SSA matches a regular-expression scanner's precision while more than tripling its recall, and against thirteen security analysts it is more precise, recovers nearly twice as many secret--door pairs, and runs five to seventeen times faster. By returning the secret, its door, and supporting evidence in one result, SSA turns credential detection into an actionable finding for triage and remediation.

**arXiv ID:** 2607.09011
</details>

<details>
<summary><strong>Embodied Multi-Agent Coordination by Aligning World Models Through Dialogue</strong> - Vardhan Dongre, Dilek Hakkani-Tür - [[pdf]](https://arxiv.org/pdf/2605.12920)</summary>

**Abstract:** Effective collaboration between embodied agents requires more than acting in a shared environment; it demands communication grounded in each agent's evolving understanding of the world. When agents can only partially observe their surroundings, coordination without communication is provably hard, but communication can, in principle, bridge this gap by allowing agents to share observations and align their world models. In this work, we examine whether LLM-based embodied agents actually realize the ability to communicate. We extend PARTNR, a benchmark for collaborative household robotics, with a natural-language dialogue channel that enables two agents with partial observability to communicate during task execution. To evaluate whether dialogue leads to genuine world-model alignment rather than superficial coordination, we propose a framework for measuring world-model alignment defined over per-agent world graphs: observation convergence (do private world models align over time?), information novelty (do messages convey what the partner lacks?), and belief-sensitive messaging (do agents model what their partner knows?). Our experiments across three LLMs reveal that dialogue reduces action conflicts 40 to 83 percentage points but degrades task success relative to silent coordination. Using our metrics, we characterize the gap between superficial coordination and genuine world-model alignment, and identify where current models fall on this spectrum. Project Website: this https URL

**arXiv ID:** 2605.12920
</details>

<details>
<summary><strong>JustAct: A Framework for Auditable Multi-Agent Systems Regulated by Inter-Organisational Policies</strong> - Christopher A. Esterhuyse, Tim Müller, L. Thomas van Binsbergen - [[pdf]](https://arxiv.org/pdf/2502.00138)</summary>

**Abstract:** In open multi-agent agent systems that cross organisational boundaries, agent actions must be regulated by complex policies. Consider medical data processing systems, which must observe generic laws (e.g., EU data protection regulations) and also specific participants' resource conditions (e.g., Bob consents to sharing his X-Rays with EU hospitals). Presently, we address the implementation of these systems as distributed software. Solutions to key sub-problems are available: existing policy languages capture the necessary normative concepts and formalise the computational representation and reasoning about policies, and existing distributed algorithms and protocols coordinate agents' changing actions and policies. But which policies and protocols are useful in application?
With the JustAct framework, we characterise a class of multi-agent systems where actors justify their actions with sufficient policy information collected from dynamic policy statements and agreements. We prove key properties of these systems, e.g., any decision that an action is permitted now cannot be refuted later, regardless of any added statements or updated agreements. We study a particular instance of the framework by specifying (in Rocq) and implementing (in Rust) a particular policy language and runtime system for mediating agent communications. We demonstrate and assess JustAct via a case study of this implementation: we reproduce the usage scenarios of Brane, an existing policy-regulated, inter-domain, medical data processing system.

**arXiv ID:** 2502.00138
</details>

<details>
<summary><strong>Leveraging Multi-Agent System (MAS) and Fine-Tuned Small Language Models (SLMs) for Automated Telecom Network Troubleshooting</strong> - Chenhua Shi, Bhavika Jalli, Gregor Macdonald, John Zou, Wanlu Lei, Mridul Jain, Joji Philip - [[pdf]](https://arxiv.org/pdf/2511.00651)</summary>

**Abstract:** Telecom networks are rapidly growing in scale and complexity, making effective management, operation, and optimization increasingly challenging. Although Artificial Intelligence (AI) has been applied to many telecom tasks, existing models are often narrow in scope, require large amounts of labeled data, and struggle to generalize across heterogeneous deployments. Consequently, network troubleshooting continues to rely heavily on Subject Matter Experts (SMEs) to manually correlate various data sources to identify root causes and corrective actions. To address these limitations, we propose a Multi-Agent System (MAS) that employs an agentic workflow, with Large Language Models (LLMs) coordinating multiple specialized tools for fully automated network troubleshooting. Once faults are detected by AI/ML-based monitors, the framework dynamically activates agents such as an orchestrator, solution planner, executor, data retriever, and root-cause analyzer to diagnose issues and recommend remediation strategies within a short time frame. A key component of this system is the solution planner, which generates appropriate remediation plans based on internal documentation. To enable this, we fine-tuned a Small Language Model (SLM) on proprietary troubleshooting documents to produce domain-grounded solution plans. Experimental results demonstrate that the proposed framework significantly accelerates troubleshooting automation across both Radio Access Network (RAN) and Core network domains.

**arXiv ID:** 2511.00651
</details>

<details>
<summary><strong>Heterogeneous Information-Bottleneck Coordination Graphs for Multi-Agent Reinforcement Learning</strong> - Wei Duan, Junyu Xuan, En Yu, Xiaoyu Yang, Jie Lu - [[pdf]](https://arxiv.org/pdf/2605.17393)</summary>

**Abstract:** Coordination graphs are a central abstraction in cooperative multi-agent reinforcement learning (MARL), yet existing sparse-graph learners lack a theoretically grounded mechanism to decide which edges should exist and how much information each edge should carry. Current methods rely on heuristic criteria that offer no formal guarantee on the learned topology, and no principled way to allocate different communication capacities to structurally different agent relationships. To address this, we propose Heterogeneous Information-Bottleneck Coordination Graphs (HIBCG), which learns a group-aware sparse graph in which both edge existence and message capacity are theoretically justified. With the graph information bottleneck (GIB) serving as the underlying tool, HIBCG first constructs a group-aligned block-diagonal prior that provides a closed-form criterion for edge retention -- determining which edges should exist and at what density per group block -- and then controls per-agent feature bandwidth on the resulting topology, compressing messages to retain only task-relevant content. We prove that the group-aligned prior strictly tightens the variational bound on topology learning, that the objective decomposes per group block, enabling differential edge control, and that capacity allocation follows a water-filling principle.

**arXiv ID:** 2605.17393
</details>

<details>
<summary><strong>Contract-Based Compositional Shielding for Safe Multi-Agent Reinforcement Learning</strong> - Omar Adalat, Edwin Hamel-De le Court, Francesco Belardinelli - [[pdf]](https://arxiv.org/pdf/2606.14130)</summary>

**Abstract:** Safe coordination problems surface in multi-agent reinforcement learning when global safety cannot be enforced by any agent unilaterally: the admissibility of one agent's action may depend on the dynamics of other agents. Decentralised shields can enforce safety at runtime, but purely factorised permissions often exclude optimal team behaviour that is safe only through coordination. We study deterministic safety guarantees for agents trained and deployed under decentralised execution, recovering team-optimal safe behaviour without centralised runtime control. Agents have a shared global specification $\phi$ in the safety fragment of Linear Temporal Logic ($\mathsf{LTL}_{\mathsf{safe}}$ ), and select among tuples of local $\mathsf{LTL}_{\mathsf{safe}}$ obligations whose conjunction implies the global specification $\phi$. Each agent may rely on the other agents' local obligations as assumptions because the whole contract tuple is certified simultaneously and allows projection into local action masks. At learning time, a non-stationary multi-armed bandit chooses among a library of local $\mathsf{LTL}_{\mathsf{safe}}$ obligations to select the tuple that optimises team reward, all without forgoing end-to-end safety. We evaluate the approach across 6 environments and 15 algorithmic variants.

**arXiv ID:** 2606.14130
</details>

</details>

<details open>
<summary><h2>Other Agent Research (2 papers)</h2></summary>

<details>
<summary><strong>SAGEAgent: A Self-Evolving Agent for Cost-Aware Modality Acquisition in Multimodal Survival Prediction</strong> - Chongyu Qu, Can Cui, Zhengyi Lu, Junchao Zhu, Tianyuan Yao, Junlin Guo, Juming Xiong, Yanfan Zhu, Yuechen Yang, Bennett A. Landman, Yuankai Huo - [[pdf]](https://arxiv.org/pdf/2607.09521)</summary>

**Abstract:** Does every cancer patient truly need a complete diagnostic workup for accurate survival prediction? In multimodal clinical oncology, diagnostic modalities follow a clinically mandated order of escalating burden -- from demographics collected at intake to genomic profiling requiring specialized tissue analysis. Current multimodal survival methods either assume all modalities are available or passively handle missing data, but none actively reason about whether acquiring the next modality is justified for a given patient along this ordered workflow. We formulate this as a sequential decision problem and propose SAGEAgent (Sequential Acquisition Guided by Experience), a self-evolving LLM-based clinical agent that decides which diagnostic modalities to acquire for each patient, balancing predictive accuracy against clinical invasiveness. SAGEAgent reasons about each patient's evolving diagnostic state through clinical tools that translate numerical predictions into text, an episodic memory that retrieves similar past cases, and a semantic memory that accumulates reusable decision patterns from experience. Experiments on a glioma cohort combining TCGA-LGG, TCGA-GBM, and BraTS with four diagnostic modalities demonstrate that SAGEAgent achieves competitive survival prediction accuracy while reducing average acquisition burden by 55%.

**arXiv ID:** 2607.09521
</details>

<details>
<summary><strong>TrustX Agent Risk Classification Framework (ARC): Risk-Tiering Internally Created Agentic AI Systems</strong> - Hannah M. Liu, Rhea Saxena, Shiv Asthana - [[pdf]](https://arxiv.org/pdf/2607.09586)</summary>

**Abstract:** The proliferation of agentic AI systems across enterprise and public-sector contexts has outpaced the capacity of general-purpose AI risk frameworks to classify and govern them. In this paper, we introduce the TrustX Agent Risk Classification Framework, a structured, repeatable instrument that can be applied to seven types of agentic AI systems and is grounded in foundational pre-existing AI governance frameworks. At the core of the framework is a twelve-dimension scoring rubric that robustly quantifies the risk. This rubric is combined with other components, such as the GPA + IAT classification model and the five-level autonomy framework derived from existing literature. These inputs produce a three-tier governance output with mapped control recommendations. A specialised Coding Assistant extension is also included to account for nuances specific to this type of agentic AI system. We then use an illustrative example to show our framework in practice. ARC is intended for AI governance practitioners, risk officers, developers, and regulators, and it will regularly undergo iteration as we continue to expand it and make it more robust. The community can access the interactive framework here: this https URL

**arXiv ID:** 2607.09586
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>GATS: Graph-Augmented Tree Search with Layered World Models for Efficient Agent Planning</strong> - Maureese Williams, Dymitr Nowicki - [[pdf]](https://arxiv.org/pdf/2607.08894)</summary>

**Abstract:** Large Language Model (LLM) agents have shown promise in multi-step planning tasks, but existing approaches like LATS (Language Agent Tree Search) and ReAct rely heavily on LLM inference during planning, leading to high computational costs and stochastic behavior. We present \textbf{GATS} (Graph-Augmented Tree Search), a planning framework that combines systematic UCB1-based tree search with a layered world model to eliminate LLM calls during inference while achieving superior planning performance. Our three-layer world model integrates: (L1) exact symbolic action matching, (L2) statistics learned from execution logs, and (L3) LLM-based prediction for unknown actions. On synthetic planning tasks with branching paths and dead-ends, GATS achieves \textbf{100\% success rate} compared to 92 % for LATS and 64\% for ReAct. On a comprehensive stress test spanning 12 challenging scenarios -- including coding workflows, web navigation, and long-horizon tasks -- GATS maintains \textbf{100\% success} while LATS drops to 88.9 % and ReAct to 23.9%. GATS requires \textbf{zero LLM calls per task} during planning (vs. 37 per task for LATS) and produces deterministic plans with zero variance across runs. Our results demonstrate that systematic search with learned world models can substantially outperform LLM-guided exploration for agent planning.

**arXiv ID:** 2607.08894
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (5 papers)</h2></summary>

<details>
<summary><strong>Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading</strong> - Zongxia Li, Zhongzhi Li, Yucheng Shi, Ruhan Wang, Junyao Yang, Zhichao Liu, Xiyang Wu, Anhao Li, Yue Yu, Ninghao Liu, Lichao Sun, Haotao Mi, LeoweiLiang - [[pdf]](https://arxiv.org/pdf/2607.08964)</summary>

**Abstract:** AI agents have become capable of autonomously completing short, well-specified tasks. However, existing terminal benchmarks largely focus on simple problems that finish within minutes and are evaluated only by their final outcome. This setup overlooks intermediate progress and partial solutions, yielding sparse reward signals and an incomplete picture of agent capability. We introduce Long-Horizon-Terminal-Bench, a terminal benchmark of 46 long-horizon tasks spanning nine categories, including experiment reproduction, software engineering, multimodal analysis, interactive games, and scientific computing. Each task follows a Terminal-Bench-style setup with a reference solution or simulation engine, but is further decomposed into fine-grained graded subtasks. This design enables dense intermediate rewards and partial credit, allowing evaluation to capture not only whether an agent reaches the final goal, but also how far it progresses on open-ended workflows. Tasks in Long-Horizon-Terminal-Bench typically require hundreds of episodes and minutes to hours of execution, stressing long-horizon planning, long-context management, and iterative debugging rather than one-shot problem solving. We evaluate 15 frontier models and find that agents consume on average 9.9M tokens per task, with roughly 231 episodes and 85.3 minutes of execution time per run, making Long-Horizon-Terminal-Bench more demanding than prior terminal-based benchmarks. Even the strongest tested model achieves 15.2% pass@1 at a partial-reward threshold of 0.95 and 10.9% at a perfect-reward threshold of 1.0, while the mean pass rate across models is 4.3% and 1.7% under the two thresholds, respectively. These results reveal headroom for improvement. We further analyze failure modes and error patterns, and release Long-Horizon-Terminal-Bench to support future progress on long-horizon terminal agents.

**arXiv ID:** 2607.08964
</details>

<details>
<summary><strong>Neuro-Agentic Control: A Deep Learning-based LLM-Powered Agentic AI Framework for Controlling Security Controls</strong> - Saroj Gopali, Bipin Chhetri, Deepika Giri, Sima Siami-Namini, Akbar Siami Namin - [[pdf]](https://arxiv.org/pdf/2607.09076)</summary>

**Abstract:** Cyberattacks on operational technology are increasingly causing costly downtime and physical damage, exposing the limitations of traditional rule-based monitoring in industrial IoT environments. While Large Language Models (LLMs) have strong semantic reasoning abilities to assist in decision support, their hallucinatory nature presents unacceptable safety liabilities for closed-loop control. This paper introduces a neuro-agentic control framework, a novel architecture that couples an LLM-based planner (i.e., such as Gemini 2.5 Flash-Lite) with a pre-trained Time-Series Foundation Model (TimesFM), to achieve physics-grounded autonomous defense. The paper introduces a ``Counterfactual Physics Injection'' mechanism that simulates the impact of LLM-proposed interventions within the numerical latent space of the foundation model before actuation, while allowing the system to reject hallucinatory or unsafe actions. Evaluated on an industrial dataset (e.g., the Secure Water Treatment (SWaT)) in the context of stochastic attack scenarios, the framework exhibited better performance compared to LSTM and TCN baselines. The Neuro-Agentic Loop prevented five breaches (33.3%) below the threshold versus LSTM (26.7%) and TCN (13.3%), with zero physically invalid (hallucinated) actions executed. These results demonstrate the efficacy of using foundation models as deterministic ``Sentinels'' to safeguard agentic AI in critical infrastructure.

**arXiv ID:** 2607.09076
</details>

<details>
<summary><strong>Communication-Efficient Digital-Twin Coordination for Heterogeneous LLM Embodied Agents over Computing Power Networks</strong> - Nuocheng Yang, Sihua Wang, Zihan Chen, Tony Q. S. Quek, Changchuan Yin - [[pdf]](https://arxiv.org/pdf/2607.09330)</summary>

**Abstract:** Embodied agent teams powered by heterogeneous large language models (LLMs) are being widely deployed in physical artificial intelligence such as smart factories, warehouses, and service robotics. To enable collaboration among such an agent team, efficient coordination mechanisms that operate reliably under limited network resources are required. However, existing heterogeneous LLM-agent coordination frameworks that rely on multi-round natural-language-based conversations introduce three coupled challenges. First, inter-agent dialogue incurs communication overhead that grows rapidly with team size. Second, the quality of coordination is constrained by the heterogeneous capabilities of the agent team's LLMs. Third, agents may suffer from action delays due to iterative negotiation. To address these challenges, we propose LDT-Coord, a networked coordination framework built upon a lightweight digital twin (DT). Specifically, each agent independently selects its intended action and reports both the action decision and a structured temporal constraint over shared resources to the DT server, thereby decoupling coordination performance from natural-language reasoning ability. Then, DT executes a training-free, rule-based orchestrator algorithm to resolve cross-agent conflicts and returns coordination instructions to prevent such conflicts. To further reduce communication overhead, we formulate agent reporting control as a constrained partially observable Markov decision process (C-POMDP) and solve it with the PPO-Lagrangian algorithm. Simulation results show that LDT-Coord achieves a task success rate comparable to conventional coordination methods while reducing communication overhead by more than 70x and maintaining robustness under LLM heterogeneity.

**arXiv ID:** 2607.09330
</details>

<details>
<summary><strong>Multimodal Reward Hacking in Reinforcement Learning</strong> - Jiayu Yao, Yiwei Wang, Anmeng Zhang, Zhe Sun, Songsong Wang, Lingrui Mei, Yuyao Ge, Shenghua Liu - [[pdf]](https://arxiv.org/pdf/2607.09492)</summary>

**Abstract:** Reinforcement learning (RL) is increasingly used to align multimodal large language models (MLLMs), but higher rewards do not always imply better task performance. This risk is amplified when visual evidence is evaluated by text-only or weakly grounded rewards. We study reward hacking in MLLM RL across safety VQA, chart VQA, and stress-test settings, varying reward design, data ambiguity, model scale (2B-32B), and RL algorithm (GRPO, RLOO, DAPO). We introduce Newly Rewarded Failure Rate (NRFR), which measures failures among samples whose proxy reward improves over the SFT baseline. Outcome-only rewards cause severe hacking, reaching 48.1% Reward Hacking Rate (RHR), while NRFR exceeding RHR shows that RL creates new failures rather than merely inheriting them. Scaling reduces but does not eliminate hacking: even the 32B model retains a 54.9% worse rate under outcome-only rewards, whereas answer-aware rewards improve the oracle trend at every scale. Robustness is also algorithm- and scale-dependent: GRPO is consistently most resistant, RLOO remains vulnerable, and DAPO improves substantially from 2B to 8B. Visual-evidence rewards help only with reliable verification: keyword-based checks increase hacking, while VLM-as-judge semantic verification reduces it. Overall, multimodal reward hacking is a systematic result of optimizing imperfect rewards, and robust alignment requires rewards and verifiers that remain reliable under optimization pressure.

**arXiv ID:** 2607.09492
</details>

<details>
<summary><strong>Inside the Skill Market: From Software Engineering Activities to Reusable Agent Skills</strong> - Jialun Cao, Xinru Yan, Songqiang Chen, Yaojie Lu, Zhongxin Liu, Shing-Chi Cheung - [[pdf]](https://arxiv.org/pdf/2607.09065)</summary>

**Abstract:** Software engineering (abbrev. SE) has continuously evolved through increasingly powerful forms of reuse, from source code and libraries to components and services. Recent advances in AI agents have introduced a potentially new reusable artifact: skills. Emerging agent skill repositories and marketplaces enable developers to package, share, and reuse SE expertise as reusable skills. This trend raises a fundamental question: what SE activities are being encapsulated into reusable skills? Existing studies primarily focus on a broad range of skills acquisition, safety, or benchmarking, while lacking a systematic understanding of SE-specific skills and their coverage across the software development lifecycle. To address this gap, we conduct the first large-scale empirical study of SE skills in public repositories and marketplaces. We collect and analyze a large corpus of SE skills, examining the activities they encapsulate, lifecycle coverage, evolution characteristics, and evaluation mechanisms. Our findings reveal that SE activities are increasingly becoming reusable artifacts via skills and suggest promising research opportunities for skill recommendation and engineering-oriented structuring, as well as the need for mechanisms to encapsulate high-context SE activities into reusable skills. Overall, our study provides the first activity-centric characterization of SE skills and reveals how SE activities are increasingly being transformed into reusable skills. These findings offer new insights into skill reuse, ecosystem development, and the future of agent-centric SE.

**arXiv ID:** 2607.09065
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
