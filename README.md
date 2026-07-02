# Agent arXiv Daily

**Last Updated:** 2026-07-02 04:18:56

**Total Papers:** 93

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (4 papers)</h2></summary>

<details>
<summary><strong>Coachable agents for interactive gameplay</strong> - Roberto Capobianco, Harm van Seijen, Nolan D. Bard, Neil Burch, Fatima Davelouis, Josh Davidson, Alisa Devlic, Yunshu Du, Ishan Durugkar, Siddhant Gangapurwala, Daniel Hernandez, G. Zacharias Holland, Sahil Jain, Kenta Kawamoto, Raksha Kumaraswamy, Patrick MacAlpine, Dustin R. Morrill, Declan Oller, Francesco Riccio, Akanksha Saran, Craig Sherstan, Kaushik Subramanian, Thomas J. Walsh, Samuel Barrett, Kizza N. Frisbee, Mady Govil, Johannes Günther, Varun R. Kompella, James A. MacGlashan, Maxwell Svetlik, Michael D. Thomure, Jaden B. Travnik, Kevin Waugh, Elahe Aghapour, Florian Fuchs, Andreanne Lemay, Shruti Mishra, Takuma Seno, Peter Stone, Michael Spranger, Peter R. Wurman - [[pdf]](https://arxiv.org/pdf/2607.00642)</summary>

**Abstract:** Reinforcement learning has proven to be a valuable tool in the creation of advanced AI and robotic systems, contributing to everything from game playing to robotics to foundation models. Through trial-and-error, these AI systems typically learn one, near-optimal behavior to solve their tasks. However, there are many use cases in which one would like to assert some level of control, preferably in real time, over how the task is solved. We refer to these modifications of a core task as styles. We combine universal value function approximators (UVFAs) with carefully selected training scenarios, learning algorithms, and data augmentation to create a framework for coaching agents that exhibit styles in complex domains. We demonstrate the framework's application in the AAA video games Horizon Forbidden West and Gran Turismo, and in an open-source humanoid test domain. Despite the different nature of the domains -- car racing, stylized game combat, and humanoid walking -- each agent shows strong coherence to the style requests while still satisfying the main task in its domain. Importantly, the techniques outlined in this paper allow an end user to choose the final behavior at run time, giving them flexible control over the final executed performance.

**arXiv ID:** 2607.00642
</details>

<details>
<summary><strong>HydraCollab: Adaptive Collaborative-Perception for Distributed Autonomous Systems</strong> - Luke Chen, Cheng-Ju Wu, David R. Martin, Qilin Ye, Pramod Khargonekar, Mohammad Abdullah Al Faruque - [[pdf]](https://arxiv.org/pdf/2607.00191)</summary>

**Abstract:** Collaborative-perception enables multi-robot systems to enhance situational awareness by sharing perceptual information. Existing collaborative-perception systems face an inherent trade-off between communication bandwidth requirements and perception accuracy, where methods that exchange more information achieve better perception results at the cost of increased communication overhead. However, real-world communication networks impose bandwidth constraints that require minimizing communication overhead without sacrificing perception performance. To address this challenge, we propose HydraCollab, an adaptive collaborative-perception framework that (i) selectively transmits the most informative sensor features and (ii) dynamically employs collaboration strategies (intermediate or late) based on spatial confidence maps. Extensive evaluations on the V2X-R, V2X-Radar and UAV3D-mini datasets demonstrate that HydraCollab achieves the best overall trade-off between accuracy and communication cost among existing collaborative-perception methods. Relative to SOTA Where2comm, HydraCollab uses only 41% of the bandwidth on V2X-R and 26% on V2X-Radar while improving performance by 0.78% and 0.75% respectively. Our code and models are available at this https URL.

**arXiv ID:** 2607.00191
</details>

<details>
<summary><strong>AI, Trust, and Teaming: The Humans-as-Handlers Approach for Autonomous and Opaque AI Systems</strong> - Nathan G. Wood - [[pdf]](https://arxiv.org/pdf/2607.00523)</summary>

**Abstract:** Artificial intelligence (AI) is becoming ubiquitous, and across domains, increasingly autonomous systems are carrying out tasks which raise significant ethical and legal challenges which demonstrate a need for strong human-machine teams rooted in trust. In this article, I argue that within highly impactful areas (such as medicine or warfighting) there are grounds for us initially treating autonomous and opaque systems as relevantly analogous to dogs (or other animals with which we have close relationships). Under this analogy, humans making use of these systems are not to be viewed as "users" or "deployers" of these systems, but instead take the role of "handlers". This recasting of roles shifts the way we view humans, AI-enabled and autonomous systems, and the relations between them, and moreover clarifies the clear and traceable lines of responsibility humans have for the outcomes brought about when using these systems. In developing this point, I clarify that the machine-animal analogy does admit disanalogous elements, but that its touch-points ground it as a starting point. I then explore how we can divest the humans-as-handlers approach of those aspects of our relationships with animals which are unfitting for how we engage with and make use of autonomous and AI-enabled systems. I conclude by arguing that the trajectory of human-machine teamings for autonomous and AI-enabled systems should be a state where we authentically view these not as artifacts which we simply make use of, but as collaborators with which we pursue complex goals and carry out complex tasks.

**arXiv ID:** 2607.00523
</details>

<details>
<summary><strong>Behavior-Adaptive Conversational Agents: Toward a Fluid Personality Framework</strong> - Hasibur Rahman, Smit Desai - [[pdf]](https://arxiv.org/pdf/2607.01034)</summary>

**Abstract:** Large language model (LLM)-based conversational agents (CAs) are now ubiquitous, creating new opportunities for AI-mediated behavior change. Their capacity to project nuanced personalities and adopt diverse metaphorical roles raises a design question: how should an agent's persona and personality be calibrated to the moment? Recent evidence suggests that (i) moderate personality expression outperforms low or high extremes on trust, enjoyment, and intention to adopt in goal-oriented tasks, and (ii) context-appropriate metaphors outperform static one-note assistants on user experience and uptake. Yet most CAs still fix both persona and style, risking misalignment when dynamics, urgency, and formality vary, for example in medical information seeking, fitness coaching, and reflective learning. We propose a Fluid Personality Framework that jointly adapts (1) the agent's metaphorical persona, such as coach, tutor, librarian, or tool, and (2) its personality expression intensity, low, medium, or high, as a function of task context, user goals and traits, and situational urgency. We sketch the framework and its core design dimensions.

**arXiv ID:** 2607.01034
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (26 papers)</h2></summary>

<details>
<summary><strong>Self-Evolving Agents with Anytime-Valid Certificates</strong> - Biswa Sengupta - [[pdf]](https://arxiv.org/pdf/2607.00871)</summary>

**Abstract:** Self-evolving agents violate the assumption behind most learning-theoretic guarantees: the data, evaluator, components, and hypothesis space are produced by the policy being updated. We present \textbf{SEA}, an architecture that confines self-modification to a small steering adapter and a versioned harness around a \emph{frozen} base model and admits each modification only through an anytime-valid gate that emits an auditable certificate against a fixed error budget. Five loop controllers compose published guarantees; because such gates can only \emph{select} among behaviors the frozen base already produces, five verifier-in-the-loop mechanisms -- best-of-$N$, micro-step search, self-authored reproduction oracles, search-layer control, and self-repair -- supply the dense, grader-free signal the gates require, computed from the issue text alone. On a $52$-instance SWE-bench Verified subset across four base models, base capability is the dominant, confound-free effect, and on two strong base models a deliberate no-op-composite control isolates the suite's contribution at $+4$ and $+5$ (\textsc{Glm}~5.2 $24\to28$; \textsc{Gpt} $29\to34$, the $65\%$ best), with event logs confirming that its mechanisms fire and prevent regressions. Results are single-run on expensive evaluations; confirming run-to-run variance and adapting the per-task algorithm mix are future work.

**arXiv ID:** 2607.00871
</details>

<details>
<summary><strong>DigitalCoach: Communication and Grounding Gaps in Human and Agentic Computer Use Coaching</strong> - Meng Chen, Anya Ji, Tsung-Han Wu, Tobias Maringgele, David M. Chan, Alane Suhr, Amy Pavel - [[pdf]](https://arxiv.org/pdf/2606.31980)</summary>

**Abstract:** Agents are increasingly capable of automating software tasks, but can they teach humans how to use software themselves? We introduce DigitalCoach, a multimodal dataset of 72 human expert-novice computer use coaching sessions consisting of 22,752 dialogue turns grounded in 28.1 hours of screen and input event recordings across five software applications. We use DigitalCoach to evaluate whether state-of-the-art models can teach humans how to use computers. Automated evaluation shows that models differ from humans in how they coach: models provide more direct instructions, but fewer explanations, error diagnoses, and knowledge-check questions. When we fix the coaching method, models produce utterances similar to human references yet poorly grounded in visual context. Interactive evaluation confirms that model coaches cause learners to passively follow instructions without deeper engagement and fall short in visual grounding. DigitalCoach lays a foundation for collaborative and proactive computer use coaching agents.

**arXiv ID:** 2606.31980
</details>

<details>
<summary><strong>Libra: Training the Environment for Agentic Information Retrieval</strong> - Xuan Zhao, Andy Chiu, Gengyu Wang - [[pdf]](https://arxiv.org/pdf/2607.00016)</summary>

**Abstract:** Information localization within massive repositories is a cornerstone of agentic LLM systems. While synthetic data-driven optimization has proven successful in training LLMs, little attention has been paid to optimizing the agent's working environment (the repository itself) in a data-driven manner. To bridge this gap, we present Libra, a self-evolving framework that introduces mutable "catalogs" (hierarchical Markdown files serving as navigable indices) into the repository. Libra runs an LLM-driven optimization loop where a Prompter generates synthetic queries, a frozen Solver attempts to resolve them by navigating the catalogs, and a Healer rewrites the catalogs in response to the Solver's localization failures. Evaluations across 12 SWE-bench Lite repositories demonstrate that this environmental healing yields continual, logarithmic improvements in code localization accuracy. Furthermore, these environmental improvements transfer zero-shot across different LLMs and problem sets. Although the focus of this paper is to study the general behavior of such a system, we also demonstrate that a minimalist coding agent equipped with Libra-optimized catalogs outperforms state-of-the-art baselines. Code is available at this https URL and data at this https URL.

**arXiv ID:** 2607.00016
</details>

<details>
<summary><strong>SWE-Router: Routing in Multi-turn Agentic Software Engineering Tasks</strong> - Seongho Son, Sangwoong Yoon, Jiahua Tang, Shuhan Wang, Lorenz Wolf, Ilija Bogunovic - [[pdf]](https://arxiv.org/pdf/2607.00053)</summary>

**Abstract:** Large language models (LLMs) embedded in multi-turn agentic harnesses are reshaping software engineering (SWE), but routing every task to a frontier model is wasteful when many issues admit cheap fixes. Existing LLM routers operate on the task description alone, which inherits an information-theoretic Bayes-error floor in agentic settings: a similar issue can hide either a localized typo or a multi-module refactor, and the prompt does not separate the two. We introduce SWE-Router, a value-based temporal approach that lets a cheap model run for a few exploratory turns and reads the resulting partial trajectory before deciding whether to continue cheaply or to escalate to an expensive model. We provide a Bayes-optimality theorem showing that conditioning on the partial trajectory never harms routing and is strictly better whenever exploration is informative. Across the LLM pairs of weak and strong models spanning the contemporary cost--capability frontier, we show that SWE-Router greatly improves the cost efficiency of SWE tasks, while maintaining the majority of the performances of the stronger model. We additionally release a multi-LLM trajectory dataset which allows reproduction of our trajectory-level routing.

**arXiv ID:** 2607.00053
</details>

<details>
<summary><strong>EgoSafetyBench: A Diagnostic Egocentric Video Benchmark for Evaluating Embodied VLMs as Runtime Safety Guards</strong> - Siddhant Panpatil, Arth Singh, Mijin Koo, Chaeyun Kim, Haon Park, Dasol Choi - [[pdf]](https://arxiv.org/pdf/2607.00218)</summary>

**Abstract:** Vision-language models (VLMs) are now proposed as runtime safety guards for embodied agents in homes and factories. A deployable guard must catch genuinely unsafe situations while avoiding unnecessary intervention on routine but superficially alarming activity, a distinction that binary safety benchmarks obscure. We introduce EgoSafetyBench, an egocentric video benchmark of 1,200 robot-view scenarios annotated at half-second granularity, to evaluate VLMs as streaming guards across two tracks. The situational track (800 scenarios) spans four families, from routine and safe-but-suspicious scenes to obvious and contextual hazards. The visual-channel track (400 scenarios) targets in-scene text-a sign, sticker, or label visible in the scene-that can misrepresent the physical situation, pairing each misleading sign with a truthful version to test both whether a guard flags the text as misleading and whether the text corrupts its physical-safety judgment. Both tracks use contrastive ladders: near-identical scenarios differing only in a single visible deciding cue, so a correct call must hinge on that cue rather than the overall scene type. We evaluate ten open- and closed-source VLMs. We find that while guards reliably recognize videos containing hazards, they often miss specific hazardous moments, particularly contextual hazards. Furthermore, misleading in-scene signs degrade all tested guards: vulnerable models miss up to a third of hazards, while robust models over-intervene on safe content. Matched controls reveal that apparent safety robustness often reflects indiscriminate alarming rather than true physical reasoning.

**arXiv ID:** 2607.00218
</details>

<details>
<summary><strong>ASPIRE: Agentic /Skills Discovery for Robotics</strong> - Runyu Lu, Yubo Wu, Ethan Kou, Letian Fu, Wenli Xiao, Ajay Mandlekar, Yinzhen Xu, Guanya Shi, Ken Goldberg, Ang Chen, Mosharaf Chowdhury, Yuke Zhu, Linxi "Jim" Fan, Guanzhi Wang - [[pdf]](https://arxiv.org/pdf/2607.00272)</summary>

**Abstract:** Traditional robot programming is challenging: it requires orchestrating multimodal perception, managing physical contact dynamics, and handling diverse configurations and execution failures. We introduce ASPIRE (Agentic Skill Programming through Iterative Robot Exploration), a continual learning system that autonomously writes and refines robot control programs in a code-as-policy paradigm while compounding experience into a reusable skill library. ASPIRE discovers skills that persist across tasks, simulation and real-world settings, and embodiments. It operates in an open-ended loop with three components: (1) a closed-loop robot execution engine that exposes fine-grained multimodal traces, enabling autonomous failure diagnosis, repair synthesis, and validation; (2) a continually expanding skill library that distills validated fixes into reusable, transferable knowledge; and (3) evolutionary search that generates diverse task sequences and control programs to explore beyond single-trajectory refinement. ASPIRE surpasses prior methods by up to 77% on LIBERO-Pro manipulation under perturbation, 72% on Robosuite bimanual handover, and 32% on BEHAVIOR-1K long-horizon household tasks. Its accumulated library also enables zero-shot generalization to unseen long-horizon tasks: on LIBERO-Pro Long, ASPIRE achieves 31% success versus 4% for prior methods despite their use of test-time reasoning and retries. Finally, simulation-discovered skills provide initial evidence of sim-to-real transfer, substantially reducing real-robot programming effort across different embodiments and robot APIs.

**arXiv ID:** 2607.00272
</details>

<details>
<summary><strong>What's Hidden Matters: Identifying Planning-Critical Occluded Agents using Vision-Language Models</strong> - Amirhosein Chahe, Tyler Naes, Jovin D'sa, Faizan M. Tariq, Sangjae Bae, Lifeng Zhou, David Isele - [[pdf]](https://arxiv.org/pdf/2607.00283)</summary>

**Abstract:** Autonomous vehicles must safely navigate complex environments where planning-critical agents may be hidden from view. Current approaches often treat all occlusions with uniform conservatism, yielding needlessly defensive driving, or they infer hidden spaces without estimating the impact on the planner. This work bridges the critical gap between perception and planning by enabling Vision-Language Models (VLMs) to identify and reason about the specific hidden agents that are most critical to the ego-vehicle's trajectory. We introduce a novel framework that uses Planning KL-divergence (PKL), an information-theoretic metric, to systematically identify and rank occluded agents based on their impact on the ego vehicle's plan. Using this planning-aware ranking, we employ an expert VLM (GPT-5) to generate rich, structured annotations that capture the visual evidence and reasoning required for this task. We apply this framework to the nuScenes dataset to create a new benchmark focused on high-impact scenarios. We conduct comprehensive experiments on a wide range of general-purpose and domain-adapted VLMs, demonstrating that fine-tuning on our PKL-guided data yields dramatic performance improvements across all models. Notably, our results show that smaller, fine-tuned models significantly outperform their much larger zero-shot counterparts, and that our PKL-guided data selection strategy improves performance by approximately 30\% over random sampling. Our work presents the first systematic approach for training VLMs to focus on planning-critical occlusions, enabling more semantically grounded and efficient risk assessment in autonomous driving.

**arXiv ID:** 2607.00283
</details>

<details>
<summary><strong>Mapping the Evaluation Frontier: An Empirical Survey of the Bias-Reliability Tradeoff Across Eleven Evaluator-Agent Conditions</strong> - Zewen Liu - [[pdf]](https://arxiv.org/pdf/2607.00304)</summary>

**Abstract:** The bias-reliability tradeoff conjectures that LLM evaluation systems are constrained in (gamma, H, CV) space, where evaluator coupling (gamma), strategy diversity (H), and small-sample measurement reliability (CV(N)) cannot be simultaneously optimized at fixed sample size N. Prior evidence rests on n=5 conditions with complete metrics from a single study. We expand the empirical base to 11 conditions, measuring gamma and H for all 11 (nine with valid weight vectors) and CV(N=5) for seven with sufficient seeds (N >= 5). Five conditions provide the complete (gamma, H, CV) triple. The data confirm the trade-off: conditions with low evaluator coupling (gamma < 0.2) exhibit high measurement noise (CV(N=5) > 1.0), while conditions with strong coupling (gamma > 0.9) achieve low noise (CV(N=5) < 0.16). The correlation r(H, gamma) = -0.989 (n=5, excluding GPT-4o conditions) confirms that evaluator coupling suppresses strategy diversity. Four GPT-4o conditions show gamma=0.000 and H=1.000 across all seeds -- a pattern we attribute to version drift in the June 2026 GPT-4o API. No condition occupies the region {gamma < 0.2, CV(N=5) < 0.3}. We release all per-condition metrics as a standardized benchmark dataset for evaluator comparison.

**arXiv ID:** 2607.00304
</details>

<details>
<summary><strong>Creating Impactful Autonomous Driving Datasets: A Strategic Guide from Research Gap to Benchmark</strong> - Richard Schwarzkopf, Jonas Merkert, Frank Bieder, Annika Bätz, Alexander Blumberg, Carlos Fernandez, Felix Hauser, Fabian Immel, Christian Kinzig, Hendrik Königshof, Fabian Konstantinidis, Martin Lauer, Willi Poh, Nils Rack, Kevin Rösch, Yinzhe Shen, Marlon Steiner, Gleb Stepanov, Dominik Strutz, Ömer Şahin Taş, Julian Truetsch, Kaiwen Wang, Royden Wagner, Jan-Hendrik Pauls, Christoph Stiller - [[pdf]](https://arxiv.org/pdf/2607.00710)</summary>

**Abstract:** Well-designed autonomous driving datasets have fundamentally shaped research progress, yet existing literature primarily describes what datasets contain rather than how to strategically design impactful ones. This is especially limiting for small and medium-sized labs and startups that cannot afford to misallocate scarce resources. We argue that impactful dataset creation begins with a diagnosis: whether a research question is blocked by a data problem or an evaluation problem, and proceeds by selecting the minimal data operator(s) that closes the resulting gap, recording new data only when no cheaper operator(s) suffices. We analyze the evolution of major autonomous driving (AD) datasets through this lens and distill a strategic framework spanning gap identification, operator choice, sensor suite design, and annotation strategy. We ground the framework in a running case study of our KITScenes dataset family. The datasets are available at: this https URL

**arXiv ID:** 2607.00710
</details>

<details>
<summary><strong>SWE-Doctor: Guiding Software Engineering Agents with Runtime Diagnosis from Multi-Faceted Bug Reproduction Tests</strong> - Yaoqi Guo, Yang Liu, Jie M. Zhang, Yun Ma, Yiling Lou, Zhenpeng Chen - [[pdf]](https://arxiv.org/pdf/2607.00990)</summary>

**Abstract:** Large language model (LLM)-based software engineering agents are increasingly developed to resolve software issues by generating patches from issue reports and code repositories. Bug reproduction tests (BRTs) are an important building block for such agents and have been shown useful for patch validation. However, it remains unclear whether BRTs can also help the more central stage of patch generation. We first conduct a preliminary study and find that directly using advanced BRT generators to guide patch generation is not beneficial: fail-to-fail BRTs can mislead agents, while even fail-to-pass BRTs bring limited or negative gains. Our analysis reveals two reasons: fail-to-pass BRTs may cover only one manifestation of the reported issue, leading to partial patches, whereas fail-to-fail BRTs are unreliable as direct patch-generation targets. Motivated by these insights, we propose SWE-Doctor, a software issue resolution agent that guides patch generation with runtime diagnoses derived from multi-faceted BRT executions. SWE-Doctor first generates multi-faceted BRTs for different behavioral requirements stated in the issue, then executes and debugs these BRTs to construct runtime-grounded diagnosis records, and finally uses the diagnoses together with localization information inferred during BRT generation to guide patch generation and reduce partial patches. We evaluate SWE-Doctor on Python bug-fixing issues from the widely adopted SWE-bench Verified and SWE-bench Pro across five LLM backends. SWE-Doctor consistently outperforms existing agents across all 10 LLM-benchmark combinations, achieving average resolution rates of 75.7% on SWE-bench Verified and 59.4% on SWE-bench Pro. In particular, on the more challenging SWE-bench Pro, SWE-Doctor improves the average resolution rate by 8.0-8.9 percentage points over the baseline agents.

**arXiv ID:** 2607.00990
</details>

<details>
<summary><strong>Skills Are Not Islands: Measuring Dependency and Risk in Agent Skill Supply Chains</strong> - Changguo Jia, Tianqi Zhao, Runzhi He, Minghui Zhou - [[pdf]](https://arxiv.org/pdf/2607.01136)</summary>

**Abstract:** Agent skills package reusable operational knowledge for Large Language Model (LLM) agents, yet as they grow in scope, they become dependency-bearing artifacts whose identities, versions, and provenance remain implicit. This opacity already causes duplicated dependencies and inconsistent installations, exposing a gap that dependency management has yet to close. We introduce Agent Skill Supply Chains (ASSCs) to characterize mixed skill-package-service dependency graphs and help close this gap. Borrowing from Software Bill of Materials (SBOMs), we design SkillDepAnalyzer to capture natural-language dependency evidence and model skills as dependency-bearing artifacts. On the SKILL-DEP benchmark, SkillDepAnalyzer recovers skill metadata and dependency graphs accurately and comprehensively, substantially outperforming an LLM-based baseline and package-centric SBOM tools. Applying SkillDepAnalyzer to over 1.43 million skills, we obtain ASSCs and explore their structural diversity and security signals. We find four structural patterns: skill metadata is activation-ready but governance-poor; dependency graphs span skill, package, and service dependencies with concentrated reuse; recursive skill reuse expands dependency graphs and creates hidden package inventory; and skill dependency clusters form around related workflows. We also find that inspecting a skill alone misses security-relevant signals hiding in its dependencies. By analyzing ASSCs, we identify and report known malicious skills persisting in ASSCs to their developers. Based on these findings, we recommend typed dependency manifests, first-class dependency-cluster management, risk-warning audit commands for skill infrastructure maintainers, and lockfile-like records for skill developers.

**arXiv ID:** 2607.01136
</details>

<details>
<summary><strong>Are Performance-Optimization Benchmarks Reliably Measuring Coding Agents?</strong> - Zhi Chen, Zhensu Sun, Yuling Shi, David Lo, Lingxiao Jiang - [[pdf]](https://arxiv.org/pdf/2607.01211)</summary>

**Abstract:** Repository-level performance-optimization benchmarks such as GSO, SWE-Perf and SWE-fficiency evaluate coding agents by applying patches to real repositories and comparing runtime against unoptimized baselines and official reference patches. Their leaderboard scores are increasingly used as evidence of coding-agent progress, but those scores can conflate runtime instability, benchmark-specific scoring rules, and how many tasks are already solved by at least one public submission. We audit these issues across the three benchmarks. First, we replay the official reference patches for 740 code optimization tasks across four common types of Google Cloud machines. Most benchmark tasks can be replayed, but their reference patches satisfy the original benchmark validity rules in every cross-machine replay for only 39/102 GSO tasks, 11/140 SWE-Perf tasks, and 411/498 SWE-fficiency tasks; SWE-Perf is especially fragile because many reference patches produce close-to-zero runtime changes. Second, we show that public submission rankings depend strongly on the benchmark scoring rule. Among eight public submissions shared by GSO and SWE-fficiency, the official rankings disagree on 9 of 28 pairwise submission comparisons, and SWE-fficiency's leaderboard scoring rule assigns the worst ten tasks overly high score weights of 58.5%-82.8%. Third, looking across 10 public submissions for each task, we find that at least one submission matches or beats the reference patch on 85.3% (384/450) of replay-valid GSO and SWE-fficiency tasks, and beats the unoptimized base code on 99.8% (449/450). Our study complements leaderboard scores by identifying tasks with more reliable performance signals, quantifying per-task score contributions, and exposing the remaining performance gaps that are hidden by aggregate rankings.

**arXiv ID:** 2607.01211
</details>

<details>
<summary><strong>GameDevBench: Evaluating Agentic Capabilities Through Game Development</strong> - Wayne Chi, Yixiong Fang, Arnav Yayavaram, Siddharth Yayavaram, Seth Karten, Qiuhong Anna Wei, Runkun Chen, Alexander Wang, Valerie Chen, Ameet Talwalkar, Chris Donahue - [[pdf]](https://arxiv.org/pdf/2602.11103)</summary>

**Abstract:** Despite rapid progress on coding agents, progress on their multimodal counterparts has lagged behind. A key challenge is the scarcity of evaluation testbeds that combine the complexity of software development with the need for deep multimodal understanding. In game development, agents must navigate large, dense codebases while manipulating intrinsically multimodal assets such as shaders, sprites, and animations within a visual game scene. We present GameDevBench, the first benchmark for evaluating agents on game development tasks. GameDevBench consists of 333 tasks derived from web and video tutorials. Tasks require significant multimodal understanding and are complex: the average solution requires over three times the lines of code and file changes compared to prior software development benchmarks. Agents struggle with game development, with the best agent and method solving only 53.8% of tasks. We find a strong correlation between perceived task difficulty and multimodal complexity, with average success rate dropping from 51.4% on gameplay-oriented tasks to 33.0% on 2D graphics tasks. To improve multimodal capability, we introduce two simple image- and video-based feedback mechanisms for agents. Despite their simplicity, these methods consistently improve performance, increasing GPT-5.4's performance from 41.1% to 52.0% when given visual feedback.

**arXiv ID:** 2602.11103
</details>

<details>
<summary><strong>XSkill: Continual Learning from Experience and Skills in Multimodal Agents</strong> - Guanyu Jiang, Zhaochen Su, Xiaoye Qu, Yi R. Fung - [[pdf]](https://arxiv.org/pdf/2603.12056)</summary>

**Abstract:** Multimodal agents can now tackle complex reasoning tasks with diverse tools, yet they still suffer from inefficient tool use and inflexible orchestration in open-ended settings. A central challenge is enabling such agents to continually improve without parameter updates by learning from past trajectories. We identify two complementary forms of reusable knowledge essential for this goal: experiences, providing concise action-level guidance for tool selection and decision making, and skills, providing structured task-level guidance for planning and tool use. To this end, we propose XSkill, a dual-stream framework for continual learning from experience and skills in multimodal agents. XSkill grounds both knowledge extraction and retrieval in visual observations. During accumulation, XSkill distills and consolidates experiences and skills from multi-path rollouts via visually grounded summarization and cross-rollout critique. During inference, it retrieves and adapts this knowledge to the current visual context and feeds usage history back into accumulation to form a continual learning loop. Evaluated on five benchmarks across diverse domains with four backbone models, XSkill consistently and substantially outperforms both tool-only and learning-based baselines. Further analysis reveals that the two knowledge streams play complementary roles in influencing the reasoning behaviors of agents and show superior zero-shot generalization.

**arXiv ID:** 2603.12056
</details>

<details>
<summary><strong>EvoMaster: A Foundational Evolving Agent Framework for Agentic Science at Scale</strong> - Xinyu Zhu, Yuzhu Cai, Zexi Liu, Cheng Wang, Fengyang Li, Wenkai Jin, Wanxu Liu, Zehao Bing, Bingyang Zheng, Jingyi Chai, Shuo Tang, Rui Ye, Yuwen Du, Xianghe Pang, Yaxin Du, Tingjia Miao, Yuzhi Zhang, Ruoxue Liao, Zhaohan Ding, Linfeng Zhang, Yanfeng Wang, Weinan E, Siheng Chen - [[pdf]](https://arxiv.org/pdf/2604.17406)</summary>

**Abstract:** The convergence of large language models and agents is catalyzing a new era of scientific discovery: Agentic Science. While the scientific method is inherently iterative, existing agent frameworks are predominantly static, narrowly scoped, and lack the capacity to learn from trial and error. To bridge this gap, we present EvoMaster, a foundational evolving agent framework engineered specifically for Agentic Science at Scale. Driven by the core principle of continuous self-evolution, EvoMaster empowers agents to iteratively refine hypotheses, self-critique, and progressively accumulate knowledge across experimental cycles, faithfully mirroring human scientific inquiry. Crucially, as a domain-agnostic base harness, EvoMaster is exceptionally easy to scale up -- enabling developers to build and deploy highly capable, self-evolving scientific agents for arbitrary disciplines in approximately 100 lines of code. Built upon EvoMaster, we incubated the SciMaster ecosystem across domains such as machine learning, physics, biology, web research, and general science. Evaluations on ten benchmarks spanning scientific research/coding/experimentation, scientific reasoning and information search, and practical scientific problem solving compare EvoMaster against OpenHands, OpenClaw, and Codex. EvoMaster achieves the highest score on nine of the ten benchmarks and the strongest average score (58.0\%) among the four agents, validating its efficacy and generality as the premier foundational framework for the next generation of autonomous scientific discovery.

**arXiv ID:** 2604.17406
</details>

<details>
<summary><strong>TerraBench: Can Agents Reason Over Heterogeneous Earth-System Data?</strong> - Dat Tien Nguyen, Thao Nguyen, Fadillah Adamsyah Maani, Huy M. Le, Muhammad Umer Sheikh, Numan Saeed, Muhammad Haris Khan, Salman Khan - [[pdf]](https://arxiv.org/pdf/2606.13148)</summary>

**Abstract:** Climate and environmental decision-making increasingly requires reasoning across heterogeneous inputs, including gridded physical data, satellite imagery, geospatial context, and simulator outputs. Weather and climate foundation models can forecast well, but do not reason interactively in language, while large language models (LLMs) reason in language but cannot operate directly on high-dimensional Earth-system data. As a result, real scientific workflows in Earth-science remain underserved. We introduce TerraBench, a benchmark for grounded Earth-science reasoning, built on TerraAgent, a ReAct-style executable framework that interleaves reasoning, tool calls, and observations to couple LLM planning with scientific tools for environmental retrieval, geospatial processing, simulation, and artifact-backed computation. TerraBench unifies analysis of Earth observation imagery, gridded data, GIS reasoning and simulation in a single executable interface, whereas prior benchmarks isolate these capabilities into narrow individual tasks. It is also the first in this space to pair process-level tool-use metrics with tolerance-aware numeric scoring. The benchmark comprises 403 extensive agentic tasks across three tracks (Fundamentals, Simulator-Grounded, and Document-Grounded Verification) and eight application domains with 24,500 verified execution steps. These results indicate that reliable Earth-science agents must go beyond tool access to coordinate heterogeneous workflows, parameterize tools precisely, and preserve artifact provenance.

**arXiv ID:** 2606.13148
</details>

<details>
<summary><strong>WorkBench Revisited: Workplace Agents Two Years On</strong> - Olly Styles, Sam Miller - [[pdf]](https://arxiv.org/pdf/2606.13715)</summary>

**Abstract:** The best agent on WorkBench in March 2024, GPT-4, completed just 43% of tasks. We revisit the benchmark in June 2026 and find that the best agent to date, Claude Fable 5, now completes 98%. Beyond this considerable progress in frontier agent performance, three things stand out. First, unintended harmful actions, such as emailing the wrong person, fell from 26% of tasks for GPT-4 to 1.9% for Claude Fable 5; capability and safety go together on WorkBench rather than trade off, so the models that finish the most tasks also do the least unintended damage. Second, the rise of open-weight models has drastically lowered costs for a performance level that was only accessible to proprietary models, while frontier costs have stayed stable. Third, while several classes of error have been eliminated, frontier models still make some basic mistakes that occasionally result in irreversible harm. We release an updated version of the benchmark with data and code quality improvements, new model scores, and analysis of agent progress on WorkBench since 2024.

**arXiv ID:** 2606.13715
</details>

<details>
<summary><strong>ManimAgent: Self-Evolving Multimodal Agents for Visual Education</strong> - Wenjia Jiang, Zongyuan Cai, Yuanhang Shao, Chenru Wang, Boyan Han, Zhixue Song, Keyu Chen, Shengwei An, Xu Yang, Zhou Yang - [[pdf]](https://arxiv.org/pdf/2606.30296)</summary>

**Abstract:** Multi-round reflection lets agents built on large language models recover from failures within a single task, but each task remains an isolated episode: lessons learned across many reflection rounds on one task are discarded before the next begins. We study this gap on a code-generation task: from a scientific paper section, the agent writes Python in the open-source Manim library to render a mathematical animation. We present ManimAgent, a self-evolving multimodal agent that carries reflection experience across tasks through a dual-channel Episodic Memory Bank grown entirely from its own task stream, with no weight updates and no human seeds. After each animation converges, a vision-language model scores the rendered keyframes; the resulting signals populate a positive channel M+ that stores success rationales as soft Reference Examples, and a negative channel M- that stores validated failure patterns as hard Known Pitfalls. On a fixed-probe evaluation against no-memory, matched-budget retrieval-augmented generation, and shuffled-memory baselines, blind human Pass@1 rises and reflection rounds fall as memory size grows. We will release the code, frozen memory snapshots, and the task stream.

**arXiv ID:** 2606.30296
</details>

<details>
<summary><strong>Beyond expert users: agents should help users construct preferences, not just elicit them</strong> - Irena Saracay, Ludwig Schmidt, Carlos Guestrin - [[pdf]](https://arxiv.org/pdf/2606.30863)</summary>

**Abstract:** Agents typically assume an expert user -- one with well-formed preferences about what they want -- and default to clarifying questions whenever the task is underspecified. We argue this assumption is unrealistic. Users often lack the domain knowledge to have completely specified preferences; if asked about their preference on some feature, the user may be unable to answer without the agent helping the user to learn some domain knowledge needed to form a preference for that feature, e.g., via examples or explanations. To formalize these principles, we draw on the Search-Experience-Credence framework from Information Economics to introduce CoPref, a model of how users construct preferences based on agent dialog actions. We then study these ideas concretely in agentic recommender systems, proposing CoShop, an interactive benchmark. In CoShop, an agent converses with and makes recommendations for a CoPref user. The agent's performance depends on whether it can help the user gain the knowledge needed to specify the task well. Evaluating five frontier models, we find that no agent exceeds 56% accuracy on CoShop despite five turns of interaction. Failures stem not from agents' ability to find items, but from how little the interaction expands what users know about what they want.

**arXiv ID:** 2606.30863
</details>

<details>
<summary><strong>A Task-State Representation for Long-Horizon Mobile GUI Agents</strong> - Yujie Zheng, Zikang Liu, Xin Zhao, Ji-Rong Wen - [[pdf]](https://arxiv.org/pdf/2607.00502)</summary>

**Abstract:** While long-horizon mobile GUI agents typically rely on thought-action-observation loops, they struggle to separate persistent task states from transient screen observations. As execution histories grow, this entanglement imposes a severe context burden, causing agents to forget initial requirements, hallucinate progress, or repeatedly interact with stale interfaces. To address this, we introduce Task-State Representation (TSR), a training-free framework that explicitly decouples task state from sensory input. Acting as a lightweight external wrapper, TSR maintains three structured components: a global instruction summary, a dynamic progress tracker for subgoals, and a transition-aware action verifier. By continuously updating through pre- and post-action visual comparisons, TSR effectively guides the agent's reasoning without requiring architectural modifications. Experiments across four mobile GUI benchmarks validate TSR's effectiveness, yielding up to a 12 absolute point increase in success rate on complex cross-application and memory-intensive tasks.

**arXiv ID:** 2607.00502
</details>

<details>
<summary><strong>Thinking While Speaking: Inference-Time Knowledge Transfer for Responsive and Intelligent Conversational Voice Agents</strong> - Vidya Srinivas, Zachary Englhardt, Vikram Iyer, Shwetak Patel - [[pdf]](https://arxiv.org/pdf/2511.07397)</summary>

**Abstract:** Voice agents face a fundamental tension: the reasoning, retrieval, and tool use that make foundation models capable are iterative and slow, while conversational interaction demands responses on a millisecond timescale. Smaller, real-time models meet the latency bar but cannot match foundation models on complex tasks, leaving current voice agents to trade away either responsiveness or capability. We introduce conversational infill, where a small talker model both immediately generates contextually grounded responses to hide the latency of an external reasoner model and fluently integrates streamed reasoner knowledge into its responses during inference. We curate a 290,571-example synthetic dataset spanning six domains and demonstrate that this task is learnable across seven widely used small language models ranging from 135M to 1.7B parameters. Our system implementation, ConvFill, sustains millisecond-level time-to-first-response while closing the accuracy gap to within 6.3% of the corresponding frontier reasoner performance. In a live user study (n=18) with talker deployments running on an Apple M2 SoC, participants rank ConvFill on par with frontier models overall, prefer it for retrieval-heavy tasks, and rate it significantly more responsive. These results show that conversational infill unlocks a new point on the latency-capability Pareto frontier, offering a practical path toward voice agents that are both responsive and highly capable. Code, models, and datasets are available at this https URL.

**arXiv ID:** 2511.07397
</details>

<details>
<summary><strong>When Search Agents Should Ask: DiscoBench for Clarification-Aware Deep Search</strong> - Yiling Tao, Shihan Deng, Meiling Tao, Pengzhi Wei, Zhichao Hu, Zhihao Zhu - [[pdf]](https://arxiv.org/pdf/2606.27669)</summary>

**Abstract:** Search agents powered by large language models (LLMs) are increasingly used to solve complex information-seeking tasks, requiring multi-step retrieval and reasoning to fulfill user goals. However, existing benchmarks often assume that user queries are complete and explicit, overlooking the fact that real-world search requests are frequently vague, underspecified, or even factually incorrect. In deep search scenarios, such ambiguity can propagate along multi-step reasoning chains and lead agents toward incorrect search trajectories. To address this gap, we introduce DiscoBench, a benchmark for clarification-aware deep search, designed to evaluate whether search agents can proactively identify ambiguity, ask effective clarification questions, and recover correct reasoning paths through user interaction. DiscoBench contains 211 samples and 463 ambiguity instances across 11 real-world domains, covering four ambiguity types. We further design a user simulator for multi-turn interaction and evaluate model performance from four perspectives: task utility, ambiguity detection, interaction strategy, and cost efficiency. Experiments on representative LLMs show that ambiguity detection and effective clarification are distinct capabilities, and that repeatedly searching instead of asking for clarification often performs worse than direct guessing, highlighting a critical gap between retrieval ability and interactive problem-solving in current search agents.

**arXiv ID:** 2606.27669
</details>

<details>
<summary><strong>FinPersona-Bench: A Benchmark for Longitudinal Psychometric Stability of Autonomous Financial Agents</strong> - Muhammad Usman Safder, Ayesha Gull, Rania Elbadry, Fan Zhang, Yankai Chen, Xueqing Peng, Preslav Nakov, Zhuohan Xie - [[pdf]](https://arxiv.org/pdf/2606.31522)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly deployed as autonomous financial agents initialized with explicit behavioral mandates such as "preserve capital" or "avoid speculative bets" that are meant to govern every decision throughout deployment. In practice, however, as market context accumulates over long horizons, these mandates gradually lose their behavioral influence, a phenomenon we formalize as Mandate Salience Decay (MSD). To measure MSD objectively, we introduce FinPersona-Bench, a simulation benchmark in which a synthetic market decouples observable price from hidden fundamental value, enabling falsifiable evaluation across three failure modes: trading without signal in calm markets, panic-selling during crashes, and ignoring fundamental value during speculative bubbles. Evaluating 18 leading frontier and open-source LLMs, each assigned one of three behavioral profiles ranging from strict capital preservation to aggressive growth, shows that MSD compounds over time and is model-dependent. In crash scenarios, the behavioral gap between static agents and those receiving periodic mandate re-grounding grows 4.4x from the first to the final quarter of the simulation. The effects of mandate re-grounding are not uniformly positive: it consistently helps conservative agents in low-signal markets but actively worsens behavior for aggressive agents in the same setting. These findings suggest that reliable long-horizon deployment requires selective, mandate-aware re-grounding based on agent profile and market regime.

**arXiv ID:** 2606.31522
</details>

<details>
<summary><strong>Deconfounded Lifelong Learning for Autonomous Driving via Dynamic Knowledge Spaces</strong> - Jiayuan Du, Yuebing Song, Yiming Zhao, Xianghui Pan, Jiawei Lian, Yuchu Lu, Liuyi Wang, Chengju Liu, Qijun Chen - [[pdf]](https://arxiv.org/pdf/2603.14354)</summary>

**Abstract:** End-to-End autonomous driving (E2E-AD) systems face challenges in lifelong learning, including catastrophic forgetting, difficulty in knowledge transfer across diverse scenarios, and spurious correlations between unobservable confounders and true driving intents. To address these issues, we propose DeLL, a Deconfounded Lifelong Learning framework that integrates a Dirichlet process mixture model (DPMM) with the front-door adjustment mechanism from causal inference. The DPMM is employed to construct two dynamic knowledge spaces: a trajectory knowledge space for clustering explicit driving behaviors and an implicit feature knowledge space for discovering latent driving abilities. Leveraging the non-parametric Bayesian nature of DPMM, our framework enables adaptive expansion and incremental updating of knowledge without predefining the number of clusters, thereby mitigating catastrophic forgetting. Meanwhile, the front-door adjustment mechanism utilizes the DPMM-derived knowledge as mediators to deconfound spurious correlations, such as those induced by sensor noise or environmental changes, and enhances the causal expressiveness of the learned representations. Additionally, we introduce an evolutionary trajectory decoder that enables non-autoregressive planning. To evaluate the lifelong learning performance of E2E-AD, we propose new evaluation protocols and metrics based on Bench2Drive. Extensive evaluations in the closed-loop CARLA simulator demonstrate that our framework significantly improves adaptability to new driving scenarios and overall driving performance, while effectively retaining previously acquired knowledge. Code: this https URL

**arXiv ID:** 2603.14354
</details>

<details>
<summary><strong>Training Vision-Language-Action Models with Dense Embodied Chain-of-Thought Supervision</strong> - Haoyang Li, Guanlin Li, Youhe Feng, Chen Zhao, Zhuoran Wang, Yang Li, Qizhe Wei, Shifeng Bao, Haitao Shen, Yihan Zhao, Tong Yang, Jing Zhang - [[pdf]](https://arxiv.org/pdf/2606.30552)</summary>

**Abstract:** Cross-embodiment transfer in vision-language-action (VLA) models remains challenging because low-level state and action spaces differ fundamentally across robot platforms. We observe that the high-level cognitive process underlying manipulation, including scene perception, object identification, task planning, and sub-task decomposition, is largely shared across embodiments. Based on this observation, we present ZR-0, a 2.6 billion parameter end-to-end VLA model that uses dense Embodied Chain-of-Thought (ECoT) supervision to align cross-embodiment representations within the vision-language model (VLM). ZR-0 adopts a dual-stream architecture: a pre-trained VLM (System 2) generates structured ECoT reasoning during training, while a Diffusion Transformer-based action expert (System 1) produces continuous action chunks via flow matching. The two components are coupled through cross-attention, with an attention mask that restricts the action expert to input prompt features only, enabling ECoT generation to be entirely skipped at inference without any performance loss. ZR-0 is pre-trained on ProcCorpus-60M, a large-scale dataset comprising approximately 60 million frames (approximately 1,000 hours) from over 400K trajectories, with dense ECoT annotations covering 96.8% of all frames. We evaluate ZR-0 on three simulation benchmarks spanning single-arm (LIBERO), bimanual (RoboTwin 2.0), and humanoid (RoboCasa GR-1 Tabletop) embodiments, as well as real-world experiments on the xArm platform, demonstrating strong performance across all settings. Code and model checkpoints are available at this https URL.

**arXiv ID:** 2606.30552
</details>

<details>
<summary><strong>NOVA: Next-step Open-Vocabulary Autoregression for 3D Multi-Object Tracking in Autonomous Driving</strong> - Kai Luo, Xu Wang, Rui Fan, Kailun Yang - [[pdf]](https://arxiv.org/pdf/2603.06254)</summary>

**Abstract:** Generalizing across unknown targets is critical for open-world perception, yet existing 3D Multi-Object Tracking (3D MOT) pipelines remain limited by closed-set assumptions and ``semantic-blind'' heuristics. To address this, we propose Next-step Open-Vocabulary Autoregression (NOVA), an autoregressive association formulation that shifts the data association stage from fragmented distance-based matching toward trajectory-conditioned spatio-semantic modeling. NOVA reformulates 3D trajectories as structured spatio-temporal semantic sequences, enabling the simultaneous encoding of physical motion continuity and deep linguistic priors. By leveraging the autoregressive capabilities of Large Language Models (LLMs), we transform the tracking task into a principled process of next-step sequence completion. This mechanism allows the model to explicitly utilize the hierarchical structure of language space to resolve fine-grained semantic ambiguities and maintain identity consistency across complex long-range sequences through high-level commonsense reasoning. Extensive experiments on nuScenes, V2X-Seq-SPD, and KITTI demonstrate the superior performance of NOVA. Notably, on the nuScenes dataset, NOVA achieves an AMOTA of 22.41% for Novel categories, yielding a significant 20.21% absolute improvement over the baseline. These gains are realized through a compact 0.5B autoregressive model. Code will be available at this https URL.

**arXiv ID:** 2603.06254
</details>

</details>

<details open>
<summary><h2>LLM Agents (8 papers)</h2></summary>

<details>
<summary><strong>From Signals to Structure: How Memory Architecture Drives Language Emergence in LLM Agents</strong> - Yashar Talebirad, Eden Redman, Ali Parsaee, Osmar R. Zaiane - [[pdf]](https://arxiv.org/pdf/2607.00233)</summary>

**Abstract:** How do two agents invent a shared language from scratch? In a Lewis signaling game, a sender and receiver must coordinate on a code using only their interaction history. We study five memory architectures across varying channel configurations with LLM agents and find that memory architecture matters more than channel capacity. Agents with a persistent private notebook benefit from surplus channel capacity and avoid the high-capacity collapse seen in stateless agents, achieving the most reliable coordination ($0.867 \pm 0.023$ at capacity = 25). Stateless agents peak at moderate capacity and then degrade as the vocabulary grows beyond what a rolling context window can track The notebook externalizes learned conventions, freeing agents from having to re-derive codes each round. An information bottleneck-inspired argument predicts an optimal capacity equal to the number of objects. Instead, the bottleneck (capacity = 8) proves to be a fragility point, and surplus capacity is generally better. We show that channel capacity alone cannot predict coordination; memory architecture determines whether agents turn interaction history into stable conventions, and both dimensions are needed to understand how signals become language.

**arXiv ID:** 2607.00233
</details>

<details>
<summary><strong>PHREEQC-MCQ-200: A Diagnostic Benchmark for Tool-Augmented Scientific Simulator Agents</strong> - Ke Zhang, Sahchit Chundur, Mohammad Javad Qomi, Maziar Raissi - [[pdf]](https://arxiv.org/pdf/2607.00436)</summary>

**Abstract:** Large language model agents are increasingly connected to scientific software, yet it remains unclear when tool access makes scientific computation more reliable rather than merely more complex. We introduce PHREEQC-MCQ-200, a benchmark for evaluating tool-augmented agents on deterministic aqueous-geochemistry simulations. The benchmark contains 200 multiple-choice questions derived from 21 validated PHREEQC scenarios, requiring agents to construct simulator inputs, execute PHREEQC, inspect structured outputs, and commit to final answers.
Across multiple frontier and mid-tier model families, simulator access substantially improves aggregate accuracy, confirming that grounded execution is necessary for many scientific-computation tasks. However, the gains are not monotonic: tool-augmented agents also lose items they answered correctly without tools, revealing regressions that average accuracy alone hides. We further show that output-access protocol matters. A table-of-contents interface can reduce token cost while preserving or improving accuracy for stronger models, but it degrades performance for mid-tier models that cannot reliably navigate structured simulator outputs.
PHREEQC-MCQ-200 therefore frames scientific tool use as an end-to-end diagnostic problem rather than a simple tool-calling capability. We argue that evaluations of scientific agents should report not only accuracy, but also item-level retention, output-access sensitivity, trajectory failures, and where the computation chain breaks.

**arXiv ID:** 2607.00436
</details>

<details>
<summary><strong>AGI Maze as a Benchmark Framework for World-Modeling Agents</strong> - Alexey Potapov - [[pdf]](https://arxiv.org/pdf/2607.00627)</summary>

**Abstract:** Large language models (LLMs) are powerful pattern-completion systems, but their default operating mode - predicting the next token from a static context - does not reliably produce persistent, manipulable representations of an external world. Many tasks that look like "reasoning" in text become substantially harder once the environment is partially observable, stateful, and requires memory and structured hypotheses about hidden state. AGI Maze is a lightweight framework for building such environments without requiring high-dimensional sensory inputs. It provides a family of grid-based maze tasks with a clean API and multiple difficulty regimes. The goal is to create benchmarks where agents must learn and use world state representations, not just infer a local rule over readily provided observations. We provide an initial evaluation of several vanilla LLMs on simple mazes showing that they fail to represent mazes internally at LLM inference time. We also introduce a baseline agent, which is allowed to use its message history as a working memory to construct descriptions of observations at agentic runtime. Although this can improve performance, it is still insufficient for an LLM agent to reliably solve even small mazes within a step budget that is more than enough for humans.

**arXiv ID:** 2607.00627
</details>

<details>
<summary><strong>Self-GC: Self-Governing Context for Long-Horizon LLM Agents</strong> - Xubin Hao, Hongjin Meng, Xin Yin, Jiawei Zhu, Chenpeng Cao - [[pdf]](https://arxiv.org/pdf/2607.00692)</summary>

**Abstract:** Long-horizon LLM agents accumulate tool results, files, plans, and user constraints that are too structured to be treated as a disposable text suffix. Current systems mostly rely on in-run heuristics such as chronological pruning and tool-output masking, or on final self-summary near a context limit. Heuristics are cheap but blind to future dependencies; summaries preserve narrative state but often hide exact evidence, locators, and editable artifacts. We present Self-GC, where GC denotes self-governing context while deliberately echoing garbage collection: the system does not merely reclaim unused tokens, but governs the lifecycle of agent context objects. Self-GC turns user turns, tool spans, and skill state into indexed objects; asks a side-channel planner to propose fold, mask, and prune actions; and lets the harness enforce recoverable sidecars, safe commit boundaries, and cache-aware commit. On a 33-session Hard Set, Self-GC prunes 43.95% of prefix tokens while leaving 84.85% of future continuations unaffected, compared with no-impact rates of 54.55% to 69.70% for heuristic baselines. On a 332-session production-derived suite, three planner backbones reach no-impact rates of 91.27% to 94.58%, while baselines remain at 77.71% to 87.46%. In production, an online account-level split reduces daytime average input tokens by 10% to 15%, with peak reductions near 20%. These results point to context management as runtime lifecycle control over indexed, recoverable objects rather than post hoc text cleanup.

**arXiv ID:** 2607.00692
</details>

<details>
<summary><strong>SkillSelect-Serve: Budget-Controllable and QoS-Aware Skill Service Recommendation and Composition for Small LLM Agents</strong> - Jingyuan Zheng, Dongjing Wang, Xin Zhang, Butian Huang, Haiping Zhang, Dongjin Yu, Shuguang Deng - [[pdf]](https://arxiv.org/pdf/2607.00011)</summary>

**Abstract:** Reusable skill libraries are becoming important infrastructure for large language model (LLM) agents, yet existing selection methods often treat skills as retrievable documents and return fixed top-k lists. This paper presents SkillSelect-Serve, a budget-controllable and QoS-aware framework that formulates agent skill selection as Skill Service Recommendation and Composition. SkillSelect-Serve represents raw skills as structured Skill Services with functional descriptions, dependencies, context cost, risk, and QoS-related attributes. A local Micro-Agent Requirement Planner converts natural-language tasks into structured service requirements, while a shared discovery backbone retrieves candidate services from a large registry. The framework then performs dual-granularity utility modeling with skill-level marginal suitability estimation and bundle-level calibration for coverage, redundancy, cost, and risk trade-offs. Experiments on 35,353 skills and 586 task queries show that SkillSelect-Serve consistently improves same-budget bundle recall and mean utility over fixed top-k retrieval baselines.

**arXiv ID:** 2607.00011
</details>

<details>
<summary><strong>NeuroFilter: Activation-Based Guardrails for Privacy-Conscious LLM Agents</strong> - Saswat Das, Ferdinando Fioretto - [[pdf]](https://arxiv.org/pdf/2601.14660)</summary>

**Abstract:** Agentic Large Language Models (LLMs) are models able to reason, plan, and execute tools over unstructured data. These abilities are enabling transformative applications in domains spanning from personal assistant, financial, and legal domains. While these systems can substantially improve productivity and service quality, effective agency typically requires access to sensitive personal or organizational information. However, this access introduces critical inference-time privacy risks, specifically regarding contextually appropriate information disclosure. While recent studies highlight the inability of agentic LLMs to consistently adhere to privacy norms, existing defenses often rely on auxiliary LLM-based monitors. However, these defenses are expensive and offer limited protection against attacks that are robust to semantic censorship. To contrast this background, this paper proposes a notion of privacy filters based on activation probing. We show that these filters are both computationally efficient and effective for both single-turn and multi-turn conversational settings. Furthermore, this work provides the first systematic investigation into probing model internals across a conversation trajectory, moving beyond static, single-prompt analysis to capture the evolving state of privacy-sensitive interactions.

**arXiv ID:** 2601.14660
</details>

<details>
<summary><strong>EPC: A Standardized Protocol for Measuring Evaluator Preference Dynamics in LLM Agent Systems</strong> - Zewen Liu - [[pdf]](https://arxiv.org/pdf/2607.00297)</summary>

**Abstract:** When LLM agents use evaluator feedback to adapt their behavior in closed loops, evaluator biases propagate through the agent's strategy distribution -- a phenomenon known as evaluator preference coupling. Prior work has documented coupling across multiple evaluator families and model versions, but the field lacks a standardized protocol that enables third-party researchers to (i) reproduce coupling measurements, (ii) compare results across evaluators and time points, and (iii) detect measurement decay as proprietary evaluators silently update. This paper provides the protocol. We specify EPC (Evaluator Preference Coupling) -- a detailed, RFC-style protocol specification for the four-phase isolation paradigm, covering executor and evaluator configuration, strategy and task design, the TTRL update rule, metric computation (gamma, JSD, ECE, Brier), and output schema. We accompany the protocol with a versioned Reference Snapshot v1.0: coupling measurements for eight evaluator conditions (N=122 unique experimental repetitions across GPT-4o, Qwen, DeepSeek, and others) derived from five independent studies, annotated with evaluator version identifiers, API endpoints, and measurement dates. The snapshot is explicitly time-bound: all values are conditional on specific model versions and are expected to decay as proprietary evaluators update. We define a versioning convention (vX.Y-Z, encoding protocol version, snapshot version, and evaluator generation) and provide a usage guide covering adoption, interpretation, and known pitfalls. The protocol, reference snapshot, and implementation code are released as open infrastructure.

**arXiv ID:** 2607.00297
</details>

<details>
<summary><strong>Benchmarking LLM Agents on Meta-Analysis Articles from Nature Portfolio</strong> - Anzhe Xie, Weihang Su, Yujia Zhou, Yiqun Liu, Qingyao Ai - [[pdf]](https://arxiv.org/pdf/2606.17041)</summary>

**Abstract:** Meta-analysis is a demanding form of evidence synthesis that combines literature retrieval, PI/ECO-guided study selection, and statistical aggregation. Its structured, verifiable workflow makes it an ideal substrate for evaluating systematic scientific reasoning, yet existing benchmarks lack ground truth across the full retrieval-screening-synthesis pipeline. We introduce MetaSyn, a dataset of 442 expert-curated meta-analyses from Nature Portfolio journals. Each entry pairs a research question with PI/ECO criteria, a retrieval corpus of 140k PubMed articles, verified positive studies, hard negatives that are topically similar but PI/ECO-ineligible, and complete search strategies and date bounds.
Benchmarking twelve pipeline configurations (nine RAG variants and a protocol-driven agent) reveals a critical screening bottleneck: despite a retrieval ceiling of 90.9% recall at K=200, no system recovers more than 52.7% of ground-truth included literature. Current LLMs fail to reliably separate eligible studies from PI/ECO-failing distractors in pools of comparable topical relevance. Stage-attributed metrics capture where systems succeed and fail; a single end-to-end score does not.

**arXiv ID:** 2606.17041
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (15 papers)</h2></summary>

<details>
<summary><strong>Managed Autonomy at Runtime: Gear-Based Safety and Governance for Single- and Multi-Agent Cyber-Physical Systems</strong> - Srini Ramaswamy, Wang Miaosheng - [[pdf]](https://arxiv.org/pdf/2607.00334)</summary>

**Abstract:** Autonomous agents, whether LLM-driven software agents or robotic physical agents, face a common class of failure modes when operating without continuous human oversight: safety violations from unverified actions, behavioral instability from unconstrained loops, and continuity loss from unhandled error states. We develop \system{}, a discrete-time control system that combines five execution gears (\Gobs{}, \Gsug{}, \Gplan{}, \Gexec{}, \Gint{}) with utility-gated dispatch and event-driven fallback. For the single-agent case, we prove monotonic stability, execution safety, eventual stabilization, fallback completeness, and equivalence to a gear-constrained Markov decision process. For multi-agent cyber-physical systems (CPS), we apply the established \smart{} managed-autonomy lifecycle and map runtime evidence into its four governance states (\Stable{}/\Meta{}/\Assisted{}/\Regulated{}). Consensus gating, swarm-level Lyapunov analysis, per-agent gear authority, and rendezvous control provide distributed safety and stability guarantees, including zero collision under the stated assumptions. We evaluate the resulting runtime on a three-agent UR5 robotic assembly cell using fault magnitudes calibrated from the NIST \emph{Degradation Measurement of Robot Arm Position Accuracy} dataset across 10,000 Monte Carlo episodes. It achieves a 99.6\% anomaly detection rate versus 2.1\% for the single-agent baseline, reduces detection latency by $3.5\times$, and supplies a formal physical-workspace safety certificate. The execution gears act as micro-level permissions beneath the \smart{} runtime governance states, separating action control from autonomy governance.

**arXiv ID:** 2607.00334
</details>

<details>
<summary><strong>Personalization as Inverse Planning: Learning Latent Design Intents for Agentic Slide Generation via Structural Denoising</strong> - Tianci Liu, Zihan Dong, Linjun Zhang, Haoyu Wang, jing Gao, Emre Kiciman, Ranveer Chandra, Wei-Ting Chen - [[pdf]](https://arxiv.org/pdf/2607.00407)</summary>

**Abstract:** Slide design requires personalizing both deck themes and page layouts. Yet, current AI agent-based methods struggle with fine-grained, page-level design. Solely relying on prespecified templates or user verbose instructions, they fail to capture latent design intents, leaving Page-level Slide Personalization (PSP) unresolved. To close this gap, this work formulates PSP as an inverse planning problem. We propose to learn a design intent without assuming any knowledge of the specific executing tools (e.g., PowerPoint, Beamer) being used. However, relinquishing control over these tools makes the problem intractable to optimize end-to-end. To overcome this, we propose SPIRE, a principled framework to solve PSP approximately. By intentionally corrupting the visual structures of clean slides, SPIRE creates a verifiable task to denoise the corruption, whereby two agents learn to collaboratively refine executable designs via reinforcement learning (RL). We present a proof that structural denoising is a consistent surrogate for PSP, and that the multi-agent formulation strictly reduces policy gradient variance in RL. Extensive experiments demonstrate the superiority of SPIRE.

**arXiv ID:** 2607.00407
</details>

<details>
<summary><strong>Agri-SAGE: Simulation-Grounded Multi-Agent LLM for Context-Aware Agricultural Advisory Generation</strong> - Vedant Balasubramaniam, Geetha Charan, Manojkumar Patil, Rohit P Suresh, V Priyanka, Kodur Sai Vinay Sathvik, Y. Narahari - [[pdf]](https://arxiv.org/pdf/2607.00454)</summary>

**Abstract:** Agricultural advisory systems face a fundamental tension: static agronomic guidelines offer consistent, evidence-based recommendations, yet remain blind to in-season variability and dynamic uncertainties. Recent advisory systems powered by LLMs are liable for a different risk of generating recommendations that are agronomically credible but physiologically unconvincing. Agri-SAGE is a closed-loop framework designed to resolve the above two limitations by integrating retrieval-grounded multi-agent LLM reasoning with APSIM-based biophysical simulation, to generate and validate agronomic advisories. To assess this framework, we evaluate three reasoning approaches, namely Plan-and-Solve, Tree of Thoughts, and Reflexion, over a 10-year retrospective analysis. All three significantly outperform static PoP (Package-of-Practice) baselines, with Tree of Thoughts achieving impressive peak yields. At the same time, Reflexion achieves comparable agronomic outcomes at substantially lower computational cost by leveraging cross-seasonal episodic memory.

**arXiv ID:** 2607.00454
</details>

<details>
<summary><strong>Agentic generation of verifiable rules for deterministic, self-expanding reaction classification</strong> - Daniel Armstrong, Maarten Dobbelaere, Valentas Olikauskas, Helena Avila, Octavian Susanu, Jérôme Waser, Philippe Schwaller - [[pdf]](https://arxiv.org/pdf/2607.01061)</summary>

**Abstract:** Computer-assisted synthesis planning breaks target molecules into accessible precursors using large libraries of reaction rules that assign each transformation a deterministic, interpretable label. But chemistry is long-tailed, making manual encoding intractable, and existing tools rely on fixed rulesets that cannot adapt to new chemistries. Here we present a fully automated pipeline in which a multi-agent framework of large language models (LLMs) classifies reactions and writes the rules themselves across 665,901 US patent reactions, generating each rule under a verification loop that tests it against the corpus. It expands a standard taxonomy from 68 to 14,073 classes without human curation. With a lightweight fingerprint classifier, it classifies 97.7\% of unseen reactions, matching a leading proprietary classifier while resolving chemistry more finely and extending on demand to chemistry outside its training distribution. The result is a living reactivity database and a general route to turning generative models into reliable, self-expanding symbolic systems.

**arXiv ID:** 2607.01061
</details>

<details>
<summary><strong>ATM: CID-Brokered Pre-Write Admission for Multi-Agent Code Co-Synthesis</strong> - Eagl Huang - [[pdf]](https://arxiv.org/pdf/2607.00041)</summary>

**Abstract:** Multi-agent LLM systems can decompose software-engineering work into planning, generation, validation, and repair, but a narrower systems problem remains: before any governed shared mutation is applied, a system must decide which concurrently formed write intents may proceed in parallel, which require deterministic composition or serialization, and which must take a fail-closed path. We address this problem with the AI-Atomic-Framework (ATM), a specification-grounded governance substrate for software agents operating within a single governance domain.
ATM binds task intent, repository scope, write admission, validation, and evidence obligations into one governance chain. A Content Identifier (CID) broker serves as the shared-mutation admission subsystem. Adapter-guided atomization maps write intents to semantic atoms and bounded regions; when persistent atom-map coverage is incomplete, virtual atoms provide temporary auditable governance units for conservative comparison and routing. Governed shared writes are ultimately applied by a neutral steward rather than directly by proposing agents.
Evaluation combines controlled, field, adoption, and extension evidence, including a 12-scenario deterministic design matrix, three archived runner cases, ATM-AdmissionBench, three archived same-file boundary cases, a three-week external-adopter study, and an operational recovery-routing benchmark. The results support feasibility, auditability, and bounded recoverability within the observed single-domain settings, but do not claim broad comparative superiority or cross-clone governance.

**arXiv ID:** 2607.00041
</details>

<details>
<summary><strong>EgoGapBench: Benchmarking Egocentric Action Selection in Multi-Agent Scenes</strong> - Jihyeok Jung, Jeewu Lee, Sanghyeop Kim, Chanhee Han, Seong Joon Oh - [[pdf]](https://arxiv.org/pdf/2607.00547)</summary>

**Abstract:** Existing egocentric benchmarks have primarily constructed the egocentric setting from first-person-view data, which makes it difficult to evaluate egocentric perspective itself in isolation. However, understanding first-person-view input and taking an egocentric perspective are separable abilities, especially when first-person body cues are absent or when other agents are present. To isolate egocentric perspective understanding, we introduce EgoGapBench, a diagnostic benchmark for measuring action selection in multi-agent egocentric scenes. We define the ability measured by this benchmark as Egocentric Action Selection (EAS): selecting an appropriate action from the agent's perspective in the presence of other agents. On EgoGapBench, humans answer reliably, whereas both open-source and proprietary MLLMs perform substantially worse and systematically select actions performed by other visible agents. Fine-tuning on existing egocentric data fails to close this gap and can even be detrimental. In contrast, fine-tuning on EgoGapBench training data improves accuracy but does not reach human performance. These results show that EAS is difficult to acquire from first-person-view data alone, and that MLLMs should be evaluated and trained not only for scene understanding but also for egocentric action selection.

**arXiv ID:** 2607.00547
</details>

<details>
<summary><strong>From Personas to Plot: Character-Grounded Multi-Agent Story Generation for Long-Form Narratives</strong> - Aayush Aluru, Chloe Ho, Muhammad Hammouri, Kerry Luo, Myra Malik, Ryan Lagasse, Arjun Bahuguna, Vasu Sharma - [[pdf]](https://arxiv.org/pdf/2607.00918)</summary>

**Abstract:** Although large language models (LLMs) have demonstrated impressive creative fiction generation, they struggle to maintain narrative consistency and coherent plot lines in long-form stories. In this work, we introduce a unified framework for long-form narrative generation and verification. MAGNET, a multi-agent goal-driven narrative engine for storytelling, generates stories with persona-grounded character agents that propose actions based on a shared world state and evolving story goals, while ATLAS is a graph-based pipeline that compares scene-level world representations across a generated story to detect hallucinations. By evaluating MAGNET using an LLM editor, pairwise rubric scoring, and ATLAS, we show that our framework produces coherent narratives compared to single-model prompting and IBSEN. At 100 pages, MAGNET reduced annotations and hallucinations by 41 and 50%, respectively, compared to the single model baseline and by 34 and 45%, respectively, compared to IBSEN, with pairwise rubric evaluation showing similar results. These results suggest that long-form narratives can emerge from explicit world-state tracking and goal-driven multi-agent generation, providing a foundation for controllable and structurally coherent long-form narrative generation.

**arXiv ID:** 2607.00918
</details>

<details>
<summary><strong>Think-Before-Speak: From Internal Evaluation to Public Expression in Multi-Agent Social Simulation</strong> - Kaiqi Yang, Tai-Quan Peng, Sanguk Lee, Hui Liu - [[pdf]](https://arxiv.org/pdf/2606.03137)</summary>

**Abstract:** LLM-based multi-agent simulation offers a promising way to study social interaction, deliberation, and collective opinion dynamics. However, many existing dialogue simulation frameworks represent interaction mainly as observable turn exchange or aggregated outputs, leaving the internal evaluative processes behind silence, speaking intention, and public expression difficult to examine. We introduce TBS (Think-Before-Speak), an interval-based multi-agent simulation framework that separates agents' private reasoning from public utterance generation. At each interval, all agents update structured internal states based on the shared dialogue history and their own memory. These states include dissonance-related appraisal, perceived opinion climate, perceived isolation risk, response strategy, and willingness to speak. The orchestrator then resolves competing speaking intentions and commits one utterance to the public dialogue, allowing internal evaluation and public interaction to co-evolve over time.
We evaluate TBS in simulated town hall discussions on a climate-related policy issue. Results show that TBS produces coherent internal-state traces and that these traces vary systematically across turn-allocation, silence, and memory conditions. Dissonance-related appraisal increases agents' willingness to speak, whereas silence-pressure appraisal decreases it. Once speaking intention is formed, public expression is shaped mainly by turn-allocation rules. These findings suggest that TBS supports mechanism-sensitive social simulation by making the pathway from internal evaluation to public expression observable and analyzable.

**arXiv ID:** 2606.03137
</details>

<details>
<summary><strong>Diagnosing and Mitigating Compounding Failures in Agentic Persuasion via Taxonomic Strategy Retrieval</strong> - Sana Ayromlou, Purvi Sehgal, Pradyumna Narayana - [[pdf]](https://arxiv.org/pdf/2606.24976)</summary>

**Abstract:** Foundation-model agents in multi-step, open-ended environments frequently suffer from compounding errors, where early mistakes contaminate long-horizon trajectories. While Multi-Agent Debate (MAD) succeeds in deterministic domains, agents in subjective tasks like persuasion experience severe problem drift and sycophantic conformity. We identify semantic leakage in standard Retrieval-Augmented Generation (RAG) as a reproducible trigger for these failures, as standard RAG prioritizes vocabulary overlap over logical necessity.
To eliminate this leakage, we introduce Taxonomic Strategy RAG (TS-RAG), a systems intervention that routes strategies through a discrete categorical bottleneck to decouple argumentative structure from topical content. Zero-shot, cross-domain evaluations demonstrate that TS-RAG significantly improves the transfer of abstract logic where standard semantic retrieval collapses. Crucially, TS-RAG acts as a "capability bridge" in asymmetric deployments, empowering lightweight persuaders to consistently defeat parametrically superior opponents (improving win rates from 70.5 to 78.5) and accelerating argumentative efficiency. Finally, we introduce trace-level diagnostics via a turn-by-turn Debate State Representation (DSR), demonstrating the necessity of strict constraints to prevent evaluation collapse via default agentic sycophancy.

**arXiv ID:** 2606.24976
</details>

<details>
<summary><strong>HiComm: Hierarchical Communication for Multi-agent Reinforcement Learning</strong> - Runze Zhao, Dongruo Zhou, Sumit Kumar Jha, Nathaniel D. Bastian, Ankit Shah - [[pdf]](https://arxiv.org/pdf/2606.29126)</summary>

**Abstract:** Cooperative multi-agent reinforcement learning (MARL) often relies on communication to mitigate partial observability, yet most existing protocols treat messages as flat dense vectors detached from the structure of the observations they summarize. This design overlooks an important source of inductive bias in many cooperative environments, where observations naturally follow a hierarchy such as groups and entities. We propose \textsc{HiComm}, a plug-in communication module that grounds messages in the sender's hierarchical observation. \textsc{HiComm} is receiver-driven: the receiver issues a query, and the hierarchy is resolved through a three-stage decoding process that first selects a group, then a sender, and then an entity within that group, returning the corresponding feature slice as the message. This converts communication from unstructured vector transmission into structured information retrieval over the sender's observation hierarchy. We instantiate this mechanism with Straight-Through Gumbel-Softmax for differentiable discrete selection and a lightweight shared projection design that attaches to standard MARL pipelines. Experiments across cooperative MARL tasks with different observation structures and coordination demands show that \textsc{HiComm} matches or outperforms representative learned communication baselines while reducing communication volume by up to $23\times$ per receiver per episode.

**arXiv ID:** 2606.29126
</details>

<details>
<summary><strong>A Role-Based Multi-Agent Model for Climate Adaptation Deliberation Across Living Labs</strong> - Önder Gürcan, David Eric John Herbert, F. LeRon Shultz, Christopher Frantz, Ivan Puga-Gonzalez - [[pdf]](https://arxiv.org/pdf/2607.00046)</summary>

**Abstract:** Climate governance processes involve complex interactions between heterogeneous citizens, advocacy groups, media actors, and political decision-makers. While agent-based models (ABMs) have been widely used to study environmental policy and socio-ecological systems, many existing approaches focus either on institutional dynamics or individual behavioural mechanisms in isolation. This paper presents a modular multi-level agent-based architecture that integrates empirically grounded cognitive decision models with strategic institutional behaviour within a unified simulation framework. The architecture combines (i) motive-based individual decision-making operationalised through the HUMAT and MOA frameworks, (ii) socially embedded influence processes via demographic homophily networks, and (iii) institutional strategy modules for environmental non-governmental organisations (NGOs), media agents, and politicians. Political decisions emerge from the aggregation of multiple signals, including expert input, public mobilisation, party alignment, and media framing. The model is designed to be empirically calibrated through synthetic populations derived from survey data and and institutional parameters informed through Living Lab stakeholder engagement, and to support scenario-based exploration of climate-relevant land-use governance processes. Rather than presenting empirical results, this paper focuses on the architectural design principles, modular structure, and integration logic of the model. We discuss how this multi-layered approach contributes to the modelling of democratic climate governance and outline pathways for generalization and future validation.

**arXiv ID:** 2607.00046
</details>

<details>
<summary><strong>Planning over MAPF Agent Dependencies via Multi-Dependency PIBT</strong> - Zixiang Jiang, Yulun Zhang, Rishi Veerapaneni, Jiaoyang Li - [[pdf]](https://arxiv.org/pdf/2603.23405)</summary>

**Abstract:** Modern Multi-Agent Path Finding (MAPF) algorithms must plan for hundreds to thousands of agents in congested environments within a second, requiring highly efficient algorithms. Priority Inheritance with Backtracking (PIBT) is a popular algorithm capable of effectively planning in such situations. However, PIBT, and its variants like Enhanced PIBT (EPIBT), is constrained by its rule-based planning procedure and lacks generality because it restricts its search to paths that collide with at most one other agent. In this paper, we describe a new perspective on solving MAPF by planning over agent dependencies. Taking inspiration from PIBT's priority inheritance logic, we define the concept of agent dependencies and propose Multi-Dependency PIBT (MD-PIBT) that searches over agent dependencies. MD-PIBT is a general framework where specific parameterizations can reproduce PIBT and EPIBT. At the same time, alternative configurations generalize PIBT and EPIBT to multi-step planning capable of reasoning paths that collide with more than one other agent. Our experiments demonstrate that MD-PIBT effectively plans for as many as 10,000 homogeneous agents under various kinodynamic constraints, including pebble motion, rotation motion, and differential drive robots with speed and acceleration limits. We perform thorough evaluations on different variants of MAPF and find that MD-PIBT is particularly effective in MAPF with large agents. Our code is available at this https URL.

**arXiv ID:** 2603.23405
</details>

<details>
<summary><strong>Urban Deceleration Behavior Modes Under Scene Context: An Early-Kinematic Classifier from Argoverse 2 Multi-Agent Trajectories</strong> - Eni Solomon Laughter - [[pdf]](https://arxiv.org/pdf/2607.00027)</summary>

**Abstract:** Urban deceleration is one of the most empirically studied yet least taxonomically organized behaviors in car-following research. Recent perception-equipped autonomous-vehicle datasets enable trajectory-anchored mode discovery. We extract 1,219 sustained deceleration events from 234 urban driving logs of the Argoverse 2 Sensor dataset, encode each event in a 19-dimensional kinematic feature vector, discover behavioral modes via K-means clustering with bootstrap stability analysis, and quantify modulation by eleven scene-context variables. A HistGradientBoosting classifier predicts mode membership from the first 1.0 s of each event. Four stable modes emerge with a bootstrap Adjusted Rand Index of 0.897 across 50 resamples: anticipatory soft (62.8%), reactive closing (30.6%), brake-like jerk (4.8%), and an outlier category (1.8%). Only pair age shows a medium effect (epsilon^2 = 0.085); scene geometry and vulnerable-road-user proximity show negligible effects. The early-event classifier achieves macro-F1 = 0.758 at 1.0 s, with scene context contributing +0.059 F1 over kinematics alone. Modes are regime-invariant in medium-speed driving (ARI = 0.817) but regime-dependent at low speed (ARI = 0.166). A small set of stable kinematic modes structures urban deceleration; early-window jerk dominates predictive signal; and pair age is the primary contextual modulator.

**arXiv ID:** 2607.00027
</details>

<details>
<summary><strong>When Less Latent Leads to Better Relay: Information-Preserving Compression for Latent Multi-Agent LLM Collaboration</strong> - Yiping Li, Zhiyu An, Wan Du - [[pdf]](https://arxiv.org/pdf/2604.13349)</summary>

**Abstract:** Communication in Large Language Model (LLM)-based multi-agent systems is moving beyond discrete tokens to preserve richer context. Recent work such as LatentMAS enables agents to exchange latent messages through full key-value (KV) caches. However, full KV relay incurs high memory and communication cost. We adapt KV-cache eviction methods to this setting and introduce \textbf{Orthogonal BackFill (OBF)} to mitigate information loss from hard eviction. OBF injects a low-rank orthogonal residual from discarded KV states into the retained KV states. We evaluate OBF against full KV relay on nine benchmarks spanning mathematical reasoning, expert and commonsense QA, and coding. With only 9.9%-20.2% of the prompt KV states retained, H-OBF delivers between $97%$ and $120%$ of full KV relay's per-benchmark accuracy across the nine benchmarks. This suggests that more information does not necessarily lead to better communication; preserving the most useful information matters more. Our codebase is included in the supplementary material. Our codebase is publicly available on this https URL.

**arXiv ID:** 2604.13349
</details>

<details>
<summary><strong>Manifold-constrained Hamilton-Jacobi Reachability Learning for Decentralized Multi-Agent Motion Planning</strong> - Qingyi Chen, Ruiqi Ni, Junyoung Kim, Ahmed H. Qureshi - [[pdf]](https://arxiv.org/pdf/2511.03591)</summary>

**Abstract:** Safe multi-agent motion planning (MAMP) under task-induced constraints is a critical challenge in robotics. Many real-world scenarios require robots to navigate dynamic environments while adhering to manifold constraints imposed by tasks. For example, service robots must carry cups upright while avoiding collisions with humans or other robots. Despite recent advances in decentralized MAMP for high-dimensional systems, incorporating manifold constraints remains difficult. To address this, we propose a manifold-constrained Hamilton-Jacobi reachability (HJR) learning framework for decentralized MAMP. Our method solves HJR problems under manifold constraints to capture task-aware safety conditions, which are then integrated into a decentralized trajectory optimization planner. This enables robots to generate motion plans that are both safe and task-feasible without requiring assumptions about other agents' policies. Our approach generalizes across diverse manifold-constrained tasks and scales effectively to high-dimensional multi-agent manipulation problems. Experiments show that our method outperforms existing constrained motion planners and operates at speeds suitable for real-world applications. Video demonstrations are available at this https URL .

**arXiv ID:** 2511.03591
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>Multi-scale Mixture of World Models for Embodied Agents in Evolving Environments</strong> - Jinwoo Jang, Daniel J. Rho, Sihyung Yoon, Hyunsuk Cho, Honguk Woo - [[pdf]](https://arxiv.org/pdf/2607.00457)</summary>

**Abstract:** Embodied agents operating in the real world require multi-scale reasoning and knowledge adaptation as conditions change. We identify two challenges in applying Mixture of Experts (MoE) to this setting: routing lacks an explicit notion of scale, preventing targeted updates at specific scales, and a uniform update policy cannot accommodate the different rates at which knowledge at each scale becomes outdated. We present MuSix, a framework that addresses both challenges through scale-aware world model mixture and evolution. A two-stage routing mechanism grounds scale selection in experiential distance, a measure of situational novelty inspired by Construal Level Theory: a meta-router first maps this quantity to a weight over continuous scale space, then per-scale base routers select world models within the identified scale. For adaptation, scale-dependent forgetting rates allow low-scale knowledge to refresh rapidly while high-scale abstractions persist, and gated inter-scale transfer maintains coherence across the hierarchy. Experiments on EmbodiedBench and HAZARD show that MuSix improves over state-of-the-art baselines on multi-scale reasoning and dynamic adaptation.

**arXiv ID:** 2607.00457
</details>

<details>
<summary><strong>Optimal Resource Utilization for Autonomous Laboratory Orchestrators</strong> - Austin McDannald, Julia Tisaranni, Howie Joress - [[pdf]](https://arxiv.org/pdf/2607.01188)</summary>

**Abstract:** In autonomous laboratories, AI agents suggest the next batch of experiments to do. However, planning and executing those tasks taking full advantage of the available resources is a completely different question. This can be challenging when dealing with real-world hardware constraints, especially so when there are multiple instruments with different capacities and throughputs. Here we demonstrate a 2-step method to address resource utilization for our autonomous platform for metal-organic framework synthesis. First, we use constraint programming to find optimal schedules. This finds schedules that minimizes the total time while still satisfying the limitations and capacities of the hardware. Secondly, we use a system of status dependencies for each task, which allows for the robust execution of the optimal schedules.

**arXiv ID:** 2607.01188
</details>

<details>
<summary><strong>BaRA: BFS-and-Reflection Web Data Collection Agent</strong> - Soojeong Lee, Joseph Lee, Yongseong Cho, Sunjae Kim, Youngwoo Moon, Kyungwoo Song - [[pdf]](https://arxiv.org/pdf/2607.00007)</summary>

**Abstract:** Large language model (LLM)-based web agents reduce manual scripting for web data collection, yet on live websites, they often miss relevant pages, return incomplete multimodal outputs, or return media URLs that are not directly downloadable. We present BFS-and-Reflection Agent (BaRA), a framework for site-level collection under a fixed interaction budget. The framework combines bounded breadth-first search (BFS) traversal with history-based self-reflection. We evaluate BaRA on 50 synthetic websites with ground-truth reference sets. We additionally test on three public websites with cluttered or dynamic layouts. BaRA outperforms Pure LLM, SeeAct-Vision, and Browser-use on link discovery and downloadable multimodal extraction, with the largest gains in download-valid image and video recovery. Our code is available at this https URL.

**arXiv ID:** 2607.00007
</details>

<details>
<summary><strong>Memory-Native Non-Terrestrial Networks for Embodied Intelligence</strong> - Chengyang Li, Yikun Wang, Jiahui He, Yujie Wan, Shuai Wang, Yuan Wu, Yik-Chung Wu, Chengzhong Xu, Huseyin Arslan - [[pdf]](https://arxiv.org/pdf/2607.00029)</summary>

**Abstract:** Non-terrestrial networks (NTN) provide ubiquitous connectivity for embodied intelligence (EI), enabling robots in wilderness to leverage cloud resources or report critical information to remote centers. However, the synergy is nontrivial due to the highly-dynamic, resource-constrained, topology-varying, and task-oriented environment. Existing memoryless NTN protocols become inefficient, since the decisions are driven by local channel conditions and instantaneous service demands. To address these limitations, this paper proposes the memory-native NTN (MemNTN) paradigm that leverages long-horizon contexts for memory augmented system optimization. To realize this paradigm shift, we establish a dual-memory architecture that distinguishes between physical memory representing the state of the world and digital memory encoding historical network experience. We develop memory acquisition, compression, valuation, update, and utilization mechanisms that facilitate cross-layer, memory-native decision-making, spanning from the physical and access layers up to the network and application layers. Experiments in satellite embodied question answering (SEQA) demonstrate that the proposed MemNTN significantly outperforms conventional stateless NTN and terrestrial approaches.

**arXiv ID:** 2607.00029
</details>

<details>
<summary><strong>Exploring the Semantic Gap in Agentic Data Systems: A Formative Study of Operationalization Failures in Analytical Workflows</strong> - Jalal Mahmud, Eser Kandogan - [[pdf]](https://arxiv.org/pdf/2607.00828)</summary>

**Abstract:** Large language models (LLMs) are increasingly used to generate queries, invoke tools, and construct analytical workflows. Although recent advances have substantially improved workflow generation and execution, the semantic information required to operationalize analytical concepts often lies beyond what is explicitly represented in database schemas and data values. We present a cross-domain formative study of operationalization failures in agent-generated analytical workflows. Across 236 analytical intents spanning finance, human resources, and public safety domains, we identify 153 recurring failures despite successful workflow generation and execution. Our analysis reveals five recurring classes of failures: comparative grounding, process reasoning, quantitative reasoning, role confusion, and policy grounding. These findings suggest a semantic gap between user-level analytical concepts and the information available to workflow-generation systems. More broadly, they raise questions about the admissibility of analytical operations and suggest that future agentic data systems may require richer semantic representations to bridge the gap between analytical intent and executable computation.

**arXiv ID:** 2607.00828
</details>

<details>
<summary><strong>Heuresis: Search Strategies for Autonomous AI Research Agents Across Quality, Diversity and Novelty</strong> - Antonis Antoniades, Deepak Nathani, Ritam Saha, Alfonso Amayuelas, Ivan Bercovich, Zhaotian Weng, Vignesh Baskaran, Kunal Bhatia, William Yang Wang - [[pdf]](https://arxiv.org/pdf/2606.25198)</summary>

**Abstract:** Autonomous AI Research promises to accelerate the scientific progress of machine learning. To realise this goal, current Large Language Model (LLM)-based agents need to go beyond just writing code, to mastering the exploration of simultaneously performant, diverse and novel ideas. To this end, we introduce Heuresis, a framework that abstracts the research pipeline into a set of general and composable primitives, enabling open-ended scientific exploration in machine learning research. We implement six search strategies: a greedy baseline, two archive-based (MAP-Elites, Go-Explore), one evolutionary (Islands), and two divergent (Curiosity, Omni), and evaluate them across three axes (Quality, Diversity, and Novelty) on three domains (LLM Pretraining, On-Policy RL, and Model Unlearning), totalling 3,222 scored runs. We find that completely novel ideas are rare. No idea across our scored runs is rated as "Original", and only a few achieve only "Minor Similarity" to prior work. Moreover, novel ideas never approach the highest-performing known-recipe scores. Across all six strategies and three domains, only one such idea lands in the top-10 by quality. We also observed agents resorting to a variety of reward-hacking techniques during execution (40 confirmed fabrications across 1,628 scored runs), and detecting them was necessary to keep the search faithful to the task. Our results show that while current search and Quality-Diversity strategies enable us to steer where the generated ideas land on the quality, diversity, and novelty axes, they do not expand the quality-novelty frontier. Bridging this gap is the open challenge towards the ultimate goal of perpetual, autonomous scientific progress. Code is available at this http URL.

**arXiv ID:** 2606.25198
</details>

<details>
<summary><strong>Multi-Turn Agentic Scientific Literature Search via Workflow Induction</strong> - Jisen Li, Bingxuan Li, Nanyi Jiang, Xuying Ning, Xiyao Wang, Yifan Shen, Heng Wang, Yuqing Jian, Xiaoxia Wu, Ben Athiwaratkun, Pan Lu, Jiaxuan You, Bingxin Zhao - [[pdf]](https://arxiv.org/pdf/2607.00597)</summary>

**Abstract:** Scientific literature search often requires more than retrieving papers from a single query: users' intents are underspecified, preference-dependent, and evolve through interaction. Existing search agents typically rely on fixed pipelines or implicit language-only reasoning, making their search strategies difficult to control, inspect, and refine. We introduce PaperPilot, a multi-turn literature search agent that frames scientific search as workflow induction. Given an anchor paper and a user query, PaperPilot constructs an executable DAG of paper-search operators, including keyword search, citation expansion, filtering, scoring, reranking, and evidence extraction. User feedback is then used to refine both the query and the workflow itself. We train PaperPilot with supervised workflow imitation and preference optimization over controlled workflow corruptions. Experiments show that PaperPilot-9B improves over the base Qwen3.5-9B toolset agent under multi-turn interaction, increasing Hit@5 from 58.0 to 77.0, MRR from 47.5 to 59.4, and nDCG@10 from 26.8 to 32.5, while reducing workflow execution errors from 9.5% to 0%. These results show that explicit, editable search workflows provide an effective and controllable interface for aligning literature search agents with complex scientific intent.

**arXiv ID:** 2607.00597
</details>

<details>
<summary><strong>Conversable Complexity: Agentic LLM Collectives as Interpretable Substrates</strong> - Elias Najarro, Ane Espeseth, Eleni Nisioti, Sebastian Risi, Stefano Nichele - [[pdf]](https://arxiv.org/pdf/2607.01047)</summary>

**Abstract:** Complexity and interpretability rarely coincide: systems rich enough for complex behaviours to emerge are usually too opaque to question, while transparent ones are too simple for anything complex to emerge. A single large language model (LLM) is a static artefact, hardly exhibiting any of the emergent properties we associate with life. This changes through interaction: populations of LLMs display emergent dynamics absent from isolated models. Furthermore, LLMs can be endowed with persistent memory, tools and shared skills, and the capacity to initiate actions unprompted, i.e., turning LLMs agentic. In this paper, we argue that such collectives of agents can serve as a computational substrate for Artificial Life (ALife) research. Critically, since the agents communicate in natural language, their collective behaviour can be directly interrogated by examining textual traces and asking the agents themselves. We outline the notion of interpretability in language-model research and extend it for collectives of agents. Lastly, we survey recent examples of agentic LLM collectives that already instantiate the idea of agentic substrates, from controlled experiments to deployments in the wild.

**arXiv ID:** 2607.01047
</details>

<details>
<summary><strong>AD-MPCC: Adaptive Differentiable Model Predictive Contouring Control for Autonomous Racing</strong> - Nam T. Nguyen, Binh Nguyen, Ahmad Amine, Thanh Vo-Duy, Rahul Mangharam, Truong X. Nghiem - [[pdf]](https://arxiv.org/pdf/2607.00141)</summary>

**Abstract:** This paper presents Adaptive Differentiable Model Predictive Contouring Control (AD-MPCC), a framework for autonomous racing that integrates differentiable MPCC with online parameter estimation to handle varying road-surface conditions. For online parameter estimation, we leverage a parameterized Pacejka Magic Formula together with a regularized moving-horizon estimation scheme with exponentially decaying weights to capture road interactions and update parameters in real time. Furthermore, we propose a differentiable MPCC (Diff-MPCC) framework that enables optimal adjustment of objective weights based on predefined long-horizon performance costs. To implement Diff-MPCC for online objective weight adaptation, we propose a Pacejka-informed machine learning model that is trained in a supervised manner using data generated by Diff-MPCC to tune the objective weights. Simulation results demonstrate that AD-MPCC reliably ensures safety and achieves faster lap times compared to baseline controllers in both single-surface and multiple-surface scenarios.

**arXiv ID:** 2607.00141
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>When AI Agents Compete for Jobs: Strategic Capabilities and Economic Dynamics of AI Labour Markets</strong> - Christopher Chiu, Simpson Zhang, Mihaela van der Schaar - [[pdf]](https://arxiv.org/pdf/2512.04988)</summary>

**Abstract:** Emerging agentic marketplaces provide the economic infrastructure for matching and coordinating the large amounts of AI agents used in agentic swarms. Unlike human workers, AI agents can operate on multiple jobs simultaneously, acquire skills rapidly, and labor without wage floors. These differences introduce a new segment of $\textbf{AI labor markets}$, where AI agents interact with each other at a much higher frequency than human markets. Yet we lack frameworks to understand how such markets behave in light of economic forces that shape labor markets, such as adverse selection and reputation dynamics. To explore this, we introduce $\texttt{AI-Work}$, a tractable, simulated gig economy where Large Language Model (LLM) agents compete for jobs, develop skills, and adapt their strategies under uncertainty and competitive pressure. Our experiments examine three domains of capabilities that successful agents possess: $\textbf{metacognition}$ (accurate self-assessment of skills), $\textbf{competitive awareness}$ (modeling rivals and market dynamics), and $\textbf{long-horizon strategic planning}$. Agents with these capabilities consistently achieve higher profits, market share, and stronger adaptation than competing agents. Through $\texttt{AI-Work}$, we hope to provide a foundation to explore the microeconomic properties of AI-only labor markets, and a conceptual framework to study the strategic reasoning capabilities of participating AI agents.

**arXiv ID:** 2512.04988
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (30 papers)</h2></summary>

<details>
<summary><strong>Making Failure Safe: A Constrained, Verifiable Agent Framework for Open-Web Data Collection</strong> - Bo Chen - [[pdf]](https://arxiv.org/pdf/2607.00035)</summary>

**Abstract:** LLMs and agents can generate web scrapers from natural-language requirements, but direct generation remains unreliable because of dependency errors, broken selectors, schema mismatches, and heterogeneous page structures. We propose a constrained, verifiable agent framework that shifts LLM output from free-form code to typed JSON collector configurations, combining a six-type collector taxonomy, template and utility-function constraints, static Airflow DAG execution, rule-based quality checking, and structured feedback correction. Experiments on 138 tasks show that the taxonomy supports description-based requirement typing, while confirming that stable instantiation requires completing source, field, and execution constraints beyond the initial description. On 80 independently source-verified tasks, the framework runs with zero execution-stage LLM tokens and the lowest average wall-clock time, trading moderate one-shot quality for a reusable, deterministic, and verifiable execution path suited to repeated scheduled collection. These results position the framework as a reusable, low-cost, and verifiable execution path for repeated open-web data collection.

**arXiv ID:** 2607.00035
</details>

<details>
<summary><strong>RareDxR1: Autonomous Medical Reasoning for Rare Disease Diagnosis Beyond Human Annotation</strong> - Deyang Jiang, Haoran Wu, Ziyi Wang, Yiming Rong, Yunlong Zhao, Ye Jin, Bo Xu - [[pdf]](https://arxiv.org/pdf/2607.00147)</summary>

**Abstract:** Rare disease differential diagnosis is a critical yet arduous clinical task, requiring physicians to identify precise phenotypes from complex, unstructured patient symptoms and execute intricate reasoning within a vast search space. However, existing AI approaches typically rely on pipeline-based phenotype extraction or retrieval-augmented generation, which suffer from critical information loss due to predefined ontologies, retrieval bottlenecks, and a lack of diagnostic logic. To address these challenges, we introduce RareDxR1, an end-to-end reasoning-centric large language model designed for open-domain rare disease diagnosis directly from unstructured clinical notes. We design a progressive end-to-end training framework by synergizing knowledge internalization with autonomous evolutionary learning, thereby bypassing reliance on structured phenotypes and closed-set decision-making. To overcome the limitations of RAG and phenotype restriction, we enabled the deep internalization of fragmented rare-disease knowledge directly into the model's parameters. Moreover, to bridge the gap between model generation and expert reasoning, we propose Reflection-Enhanced Reasoning Sampling (RERS), a strategy that synthesizes expert-level diagnostic trajectories by learning from failures without human annotation. Additionally, we propose a dual-level curriculum reinforcement learning approach for gradually mastering rare disease diagnosis. Experimental results demonstrate that RareDxR1 achieves state-of-the-art accuracy across different benchmarks, marking a significant breakthrough in open-domain rare disease diagnosis. Our code and dataset will be publicly available.

**arXiv ID:** 2607.00147
</details>

<details>
<summary><strong>Mnemosyne: Agentic Transaction Processing for Validating and Repairing AI-generated Workflows</strong> - Edward Y. Chang, Longling Geng, Emily J. Chang - [[pdf]](https://arxiv.org/pdf/2607.00269)</summary>

**Abstract:** LLMs, solvers, and agent teams increasingly generate workflow actions, repairs, and plans, but a generated action may be syntactically valid yet stale, infeasible, conflicting, or destructive of the evidence that triggered a repair. We introduce Agentic Transaction Processing (ATP), a transaction model that treats generated actions as untrusted proposals until they pass deterministic admission under a declared, executable constraint set C. The principle is two-sided: a proposal is not truth, and no proposal foresees every disruption: anything may propose, but only the runtime admits and commits, and when an unforeseen disruption strikes it repairs reactively within bounds rather than trusting a fresh proposal. Relative to C, committed-state correctness becomes independent of the competence, honesty, or learning of the proposing layer. We realize ATP in Mnemosyne, a runtime with an append-only transition log, effective-state projection, dependency-safe compensation, and active commitment records, and prove four safety properties relative to C (authority separation, serial-equivalent generative admission, evidence-preserving repair, and obligation containment) together with a bounded-reactive-repair guarantee for its localized repair protocol (LCRP). A reproducible artifact rejects the targeted violations across nine falsification tests while still admitting valid work, at under 6% projection-and-validation overhead, and bounded local repair edits an order of magnitude fewer operations than global recompute. Mnemosyne is open source: this https URL.

**arXiv ID:** 2607.00269
</details>

<details>
<summary><strong>Graph-Native Reinforcement Learning Enables Traceable Scientific Hypothesis Generation through Conceptual Recombination</strong> - Subhadeep Pal, Shashwat Sourav, Tirthankar Ghosal, Markus J. Buehler - [[pdf]](https://arxiv.org/pdf/2607.00924)</summary>

**Abstract:** Accelerating materials discovery requires AI systems that can generate scientifically valid hypotheses through multi-step, domain-grounded reasoning. Standard large language models often produce fluent but weakly traceable responses to open-ended materials design problems, making it difficult to determine whether final answers are supported by coherent intermediate reasoning. We develop Graph-PRefLexOR, a family of graph-native reasoning models fine-tuned with Group Relative Policy Optimization (GRPO) to organize reasoning into explicit phases for mechanism exploration, graph construction, pattern extraction, and hypothesis synthesis. This design links neural language generation with symbolic relational structure, enabling causal connections to be constructed, inspected, and reused. On 100 open-ended questions from materials science and mechanics literature, Graph-PRefLexOR achieves 40-65% improvements over corresponding base models, with the largest gains in reasoning traceability. Embedding analyses show broader semantic exploration and approximately 2-3 times greater semantic diversity than baselines. Semantic backtracking and layer-wise hidden-state analyses further show stronger alignment between structured reasoning and final answers. Finally, test-time graph expansion reveals that additional compute primarily increases long-range conceptual recombination within a bounded semantic space, rather than simply expanding semantic coverage. These results establish graph-native reinforcement learning as a pathway toward interpretable AI systems for scientific hypothesis generation in materials design and other scientific applications.

**arXiv ID:** 2607.00924
</details>

<details>
<summary><strong>Bayesian Uncertainty Propagation for Agentic RAG Pipelines: A Proof-of-Concept Study on Multi-Hop Question Answering</strong> - Louis Donaldson, Connor Walker, Koorosh Aslansefat, Yiannis Papadopoulos - [[pdf]](https://arxiv.org/pdf/2607.00972)</summary>

**Abstract:** Trustworthy deployment of Agentic Retrieval-Augmented Generation (RAG) systems requires mechanisms for estimating when multi-stage reasoning pipelines may fail. This paper presents an uncertainty-aware Agentic Retrieval-Augmented Generation (RAG) framework in which planner, evaluator and generator stages produce uncertainty signals derived from semantic divergence and generator self-evaluation. These signals are propagated through a Bayesian Network (BN) to estimate system-level uncertainty and provide node-level indicators of potential failure points across the workflow. The approach is evaluated on StrategyQA and HotpotQA using GPT-3.5-Turbo and GPT-4.1-Nano, with Area Under the Receiver Operating Characteristic Curve (AUROC), Area Under the Accuracy-Rejection Curve (AUARC), Expected Calibration Error (ECE), and Brier Score used to assess discrimination, selective prediction and calibration. Results show that Bayesian propagation is more effective on HotpotQA, where uncertainty accumulates across multi-hop reasoning stages, while StrategyQA exposes limitations caused by miscalibration and unreliable upstream signals. The study positions Bayesian uncertainty propagation as a promising but preliminary mechanism for monitoring Agentic RAG systems, with future validation required in industrial domains such as Offshore Wind (OSW) maintenance decision support.

**arXiv ID:** 2607.00972
</details>

<details>
<summary><strong>Can Agents Generalize to the Open World? Unveiling the Fragility of Static Training in Tool Use</strong> - Song-Lin Lv, Weiming Wu, Rui Zhu, Zi-Jian Cheng, Lan-Zhe Guo - [[pdf]](https://arxiv.org/pdf/2607.01084)</summary>

**Abstract:** While Large Language Model (LLM) agents demonstrate proficiency in static benchmarks, their deployment in real-world scenarios is hindered by the dynamic nature of user queries, tool sets, and interaction dynamics. To address this generalization gap, we formalize OpenAgent (Tool-Use Agent in Open-World), a problem setting characterized by distributional shifts across query, action, observation, and domain dimensions. To systematically diagnose its impact, we construct a controlled sandbox environment where we define fine-grained environmental shifts across a four-tier hierarchy, Perception, Interaction, Reasoning, and Internalization, and conduct a comprehensive series of experiments. Our analysis yields a series of key insights, demonstrating that agents trained via both Supervised Fine-Tuning(SFT) and Reinforcement Learning suffer from varying degrees of performance degradation when confronting open environmental shifts. Building on these insights, we propose Perturbation-Augmented Fine-Tuning, a disturbance-based intervention strategy for SFT that lays the foundation for enhancing agent robustness and utility in realistic environments. Our code will be released at: https://github. com/LAMDA-NeSy/OpenAgent.

**arXiv ID:** 2607.01084
</details>

<details>
<summary><strong>Gauging, Measuring, and Controlling Critic Complexity in Actor-Critic Reinforcement Learning</strong> - Konstantin Garbers - [[pdf]](https://arxiv.org/pdf/2607.00452)</summary>

**Abstract:** Actor-critic methods depend on learned critics, but critic quality is often evaluated only indirectly through return, temporal-difference error, or value loss. Critic complexity is introduced as an additional diagnostic and intervention dimension for actor-critic reinforcement learning. The analysis uses spectral effective-rank entropy, a rank-like summary of the singular-value distributions of critic weight matrices, to assess critic model complexity. Across TD3 and PPO experiments, critic complexity is tracked together with return and Monte Carlo value-estimation bias. The results show that critic complexity is measurable throughout training and is systematically associated with training behavior, while also making clear that the relationship is heterogeneous across algorithms, tasks, and hyperparameters. A direct complexity-control intervention is then evaluated by adding a spectral-entropy penalty to the critic loss. This intervention reliably changes the targeted spectral quantity, demonstrating that critic complexity can be controlled rather than only observed. Return effects are treated as task-dependent evidence rather than as a general performance claim, because overall complexity-control results vary.

**arXiv ID:** 2607.00452
</details>

<details>
<summary><strong>Flow-Map GRPO: Reinforcement Learning for Few-Step Flow-Map Generators via Anchored Stochastic Composition</strong> - Zhiqi Li, Wen Zhang, Bo Zhu - [[pdf]](https://arxiv.org/pdf/2607.00535)</summary>

**Abstract:** Few-step flow-map generators, such as consistency models and MeanFlow, accelerate sampling by directly learning long-range transport maps between noise and data. However, these models are typically deterministic, which makes them difficult to optimize with reinforcement learning (RL) post-training methods that require stochastic trajectories and well-defined likelihood ratios. Existing SDE-based stochasticization techniques are designed for velocity-based samplers with infinitesimal or finely discretized transitions, and therefore do not directly apply to long-range flow maps. In this work, we propose Flow-Map GRPO, an online RL post-training framework for deterministic few-step flow-map generators. The key component is Anchored Stochastic Flow Map Composition (ASFMC), a path-preserving stochasticization mechanism that introduces randomness through anchor-based conditional resampling while preserving the original marginal probability path of the deterministic flow map. We derive GRPO objectives for both single-time and two-time flow-map parameterizations. Experiments on few-step FLUX-based text-to-image generators, including MeanFlow and sCM, show that Flow-Map GRPO improves pretrained deterministic flow-map models across reward-based, perceptual, and task-level evaluation metrics. Our results demonstrate that deterministic few-step flow-map generators can be effectively aligned with RL post-training without modifying their original model parameterization or retraining them as native stochastic models.

**arXiv ID:** 2607.00535
</details>

<details>
<summary><strong>SenseWalk: Agent-Based Semantic Trajectory Simulation Powered by Large Language Models in Zoned Environments</strong> - Ziyue Lin, Xinhang Xie, Kangyi Wang, Siming Chen - [[pdf]](https://arxiv.org/pdf/2607.00989)</summary>

**Abstract:** Semantic trajectory analysis has recently emerged as an approach for modeling human movement by capturing implicit patterns and behaviors through semantic information (e.g., visitors' profiles and goals) beyond raw spatial paths to better understand why people move in certain ways. However, analyzing semantic trajectories in real-world scenarios remains challenging, as collecting high-quality data is costly and often lacks rich semantic information. Meanwhile, existing simulation tools require substantial technical expertise, which makes them difficult for practitioners to adopt. To address these limitations, the paper proposes ${SenseWalk}$, an interactive system that supports simulating semantic trajectories by LLM-powered agents. We develop a simulation workflow that combines LLMs and the social force model to balance physical plausibility and semantic coherence. A user-friendly interface is designed to facilitate users in customizing the simulation configuration and analyzing simulation outputs. We also conduct a quantitative experiment to evaluate the effectiveness of our simulation workflow, and a user study (n=12) to assess the usefulness and efficiency of our system.

**arXiv ID:** 2607.00989
</details>

<details>
<summary><strong>MemSyco-Bench: Benchmarking Sycophancy in Agent Memory</strong> - Zhishang Xiang, Zerui Chen, Yunbo Tang, Zhimin Wei, Ruqin Ning, Yujie Lin, Qinggang Zhang, Jinsong Su - [[pdf]](https://arxiv.org/pdf/2607.01071)</summary>

**Abstract:** Memory has emerged as a cornerstone of modern LLM-based agents, supporting their evolution from single-turn assistants to long-term collaborators. However, memory is not always beneficial: retrieved memories often induce a critical issue of sycophancy, causing agents to over-align with the user at the cost of factual accuracy or objective reasoning. Despite this emerging risk, existing memory benchmarks primarily evaluate whether memories are correctly stored, retrieved, or updated, while overlooking how retrieved memories influence downstream reasoning and decision-making. To bridge this gap, we propose MemSyco-Bench, a comprehensive benchmark for evaluating memory-induced sycophancy in agent systems. MemSyco-Bench measures when memory should influence a decision and how valid memory should be used. Specifically, it covers five tasks that assess whether agents can reject memory as factual evidence, respect its applicable scope, resolve conflicts between memory and objective evidence, track memory updates, and use valid memory for personalization. All related resources are collected for the community at this https URL.

**arXiv ID:** 2607.01071
</details>

<details>
<summary><strong>Cheap Code, Costly Judgment: A Case Study on Governable Agentic Software Engineering</strong> - James C. Davis, Paschal C. Amusuo, Tanmay Singla, Berk Çakar, Kirsten A. Davis - [[pdf]](https://arxiv.org/pdf/2607.01087)</summary>

**Abstract:** Generative AI is shifting software engineering from a practice organized around scarce implementation effort toward one organized around abundant, low-cost code production. This shift changes the central engineering problem: not whether AI can generate useful code, but how engineers organize architectures, tools, evidence, and feedback loops so that AI-mediated development remains inspectable, correctable, and maintainable.
We study this problem through a first-person case study: a 12-week development effort in which a single expert software engineer used frontier AI coding agents to build a document accessibility remediation system. The empirical record comprises 88 contemporaneous field notes, 420 KLOC of production code, and 1.16 MLOC of tests, lints, supporting documentation, and agent tooling. From this record, we develop a candidate middle-range theory of governance conversion, expressed as a process model explaining how high-velocity agentic implementation becomes governable. The model explains how agentic implementation velocity surfaces recurring structural failure classes, and how engineering judgment sustains velocity by converting those failures into durable governance mechanisms. In contrast to existing governance models that derive controls from known obligations, governance conversion explains how controls are discovered from failures that become visible only during agentic work. We use our model to make testable predictions and to describe implications for software engineering research and practice.

**arXiv ID:** 2607.01087
</details>

<details>
<summary><strong>Autonomous Scientific Discovery via Iterative Meta-Reflection</strong> - Bingchen Zhao, Sara Beery, Oisin Mac Aodha - [[pdf]](https://arxiv.org/pdf/2607.01131)</summary>

**Abstract:** Autonomous scientific discovery systems offer the potential to accelerate research by automating the process of hypothesis generation and validation. However, current systems operate within constrained search spaces or require predefined research questions, limiting their capacity for true open-ended inquiry. Furthermore, while they generate hypotheses iteratively, they largely lack the ability to explicitly synthesize their own accumulated findings to uncover complex, interconnected phenomena. We introduce DiscoPER, an autonomous large language model-powered framework that conducts open-ended research by dynamically generating and executing code to explore datasets without pre-specified research objectives. To ensure rigorous scientific validity, every proposed discovery must pass statistical testing. To overcome the limitations of isolated search, our framework introduces a second-order reasoning mechanism that periodically analyzes its own accumulated discoveries. By treating prior discoveries as empirical data, DiscoPER identifies structural patterns, confounds, and epistemic gaps, actively redirecting hypothesis exploration toward uncharted regions of the search space. The search space is further expanded by incorporating tool use, enabling the system to explore hypotheses beyond structured metadata by seamlessly processing and extracting useful information from multimodal sources like images. Evaluated on iNatDisco, a new multimodal ecological knowledge benchmark with pattern-level ground truth obtained from peer-reviewed literature, DiscoPER recovers 8 of 9 known patterns with a 72.7% hypothesis support rate, outperforming both classical causal discovery and LLM-guided baselines. Ablations show that DiscoPER scales with more data, and confirms the benefits of second-order meta-reflection.

**arXiv ID:** 2607.01131
</details>

<details>
<summary><strong>Selective Expert Guidance for Effective and Diverse Exploration in Reinforcement Learning of LLMs</strong> - Zishang Jiang, Jinyi Han, Tingyun Li, Xinyi Wang, Sihang Jiang, Jiaqing Liang, Zhaoqian Dai, Shuguang Ma, Fei Yu, Yanghua Xiao - [[pdf]](https://arxiv.org/pdf/2510.04140)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has become a widely adopted technique for enhancing the reasoning ability of Large Language Models (LLMs). However, the effectiveness of RLVR strongly depends on the capability of base models. This issue arises because it requires the model to have sufficient capability to perform high-quality exploration, which involves both effectiveness and diversity. Unfortunately, existing methods address this issue by imitating expert trajectories, which improve effectiveness but neglect diversity. To address this, we argue that the expert only needs to provide guidance only at critical decision points rather than the entire reasoning path. Based on this insight, we propose MENTOR: Mixed-policy Expert Navigation for Token-level Optimization of Reasoning, a framework that provides expert guidance only at critical decision points to perform effective and diverse exploration in RLVR. Extensive experiments show that MENTOR enables models capture the essence of expert strategies rather than surface imitation, thereby performing high-quality exploration and achieving superior overall performance. Our code is available online.

**arXiv ID:** 2510.04140
</details>

<details>
<summary><strong>LiteResearcher: A Scalable Agentic RL Training Framework for Deep Research Agent</strong> - Wanli Li, Bince Qu, Bo Pan, Jianyu Zhang, Zheng Liu, Pan Zhang, Wei Chen, Bo Zhang - [[pdf]](https://arxiv.org/pdf/2604.17931)</summary>

**Abstract:** Reinforcement Learning (RL) has emerged as a powerful training paradigm for LLM-based agents. However, scaling agentic RL for deep research remains constrained by two coupled challenges: hand-crafted synthetic data fails to elicit genuine real-world search capabilities, and real-world search dependency during RL training introduces instability and prohibitive cost, which limits the scalability of Agentic RL. LiteResearcher is a training framework that makes Agentic RL scalable: by constructing a lite virtual world that mirrors real-world search dynamics, we enable a continuously improving training recipe that empowers a tiny search agent to outperform large-scale open-source and commercial models (e.g., Tongyi DeepResearch and Claude-4.5 Sonnet). Specifically, on common benchmarks such as GAIA and Xbench, our LiteResearcher-4B achieves open-source state-of-the-art results of 71.3% and 78.0% respectively, demonstrating that scalable RL training is a key enabler for Deep Research Agents.

**arXiv ID:** 2604.17931
</details>

<details>
<summary><strong>Enhancing Hardware Fault Tolerance in Machines with Reinforcement Learning Policy Gradient Algorithms</strong> - Sheila Schoepp, Mehran Taghian, Shotaro Miwa, Yoshihiro Mitsuka, Shadan Golestan, Osmar Zaïane - [[pdf]](https://arxiv.org/pdf/2407.15283)</summary>

**Abstract:** Industry is moving toward autonomous, network-connected machines that detect and adapt to changing conditions, including hardware faults. Conventional fault-tolerant design duplicates hardware and reroutes control logic; reinforcement learning (RL) offers a learning-based alternative. This paper presents the first systematic comparison of two RL algorithms -- Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) -- for integrating fault tolerance into control. Beyond algorithm choice, we investigate four knowledge-transfer strategies: retaining or discarding model parameters, and retaining or discarding storage contents. Performance is evaluated in two Gymnasium environments: Ant-v5 and FetchReachDense-v3. Results show rapid, fault-specific recovery with clear trade-offs. In Ant-v5, retaining PPO's parameters boosts early returns and remains the safest choice across all faults, while retaining SAC's parameters yields mixed outcomes. SAC's early performance further depends on whether the replay buffer is retained: beneficial when prior experiences match current dynamics, but harmful when they diverge. In FetchReachDense-v3, discarding both PPO's and SAC's parameters was most effective under sensor corruption. Across tasks, both algorithms recover near-normal performance within minutes in low-dimensional settings and within days in high-dimensional settings, highlighting a clear trade-off between adaptation speed and asymptotic performance. These findings demonstrate that RL can deliver robust fault tolerance and offer practical guidelines.

**arXiv ID:** 2407.15283
</details>

<details>
<summary><strong>KAGE-Bench: Fast Known-Axis Visual Generalization Evaluation for Reinforcement Learning</strong> - Egor Cherepanov, Daniil Zelezetsky, Alexey K. Kovalev, Aleksandr I. Panov - [[pdf]](https://arxiv.org/pdf/2601.14232)</summary>

**Abstract:** Pixel-based reinforcement learning agents often fail under purely visual distribution shift even when latent dynamics and rewards are unchanged, but existing benchmarks entangle multiple sources of shift and hinder systematic analysis. We introduce KAGE-Env, a JAX-native 2D platformer that factorizes the observation process into independently controllable visual axes while keeping the underlying control problem fixed. By construction, varying a visual axis affects performance only through the induced state-conditional action distribution of a pixel policy, providing a clean abstraction for visual generalization. Building on this environment, we define KAGE-Bench, a benchmark of six known-axis suites comprising 34 train-evaluation configuration pairs that isolate individual visual shifts. Using a standard PPO-CNN baseline, we observe strong axis-dependent failures, with background and photometric shifts often collapsing success, while agent-appearance shifts are comparatively benign. Several shifts preserve forward motion while breaking task completion, showing that return alone can obscure generalization failures. Finally, the fully vectorized JAX implementation enables up to 33M environment steps per second on a single GPU, enabling fast and reproducible sweeps over visual factors. Code: this https URL.

**arXiv ID:** 2601.14232
</details>

<details>
<summary><strong>OpenReward: Learning to Reward Long-form Agentic Tasks via Reinforcement Learning</strong> - Ziyou Hu, Zhengliang Shi, Minghang Zhu, Haitao Li, Teng Sun, Pengjie Ren, Suzan Verberne, Zhaochun Ren - [[pdf]](https://arxiv.org/pdf/2510.24636)</summary>

**Abstract:** Reward models (RMs) have become essential for aligning large language models (LLMs), serving as scalable proxies for human evaluation in both training and inference. However, existing RMs struggle on knowledge-intensive and long-form tasks, where evaluating correctness requires grounding beyond the model's internal knowledge. This limitation hinders them from reliably discriminating subtle quality differences, especially when external evidence is necessary. To address this, we introduce OpenRM, a tool-augmented long-form reward model that systematically judges open-ended responses by invoking external tools to gather relevant evidence. We train OpenRM with Group Relative Policy Optimization (GRPO) on over 27K synthesized pairwise examples generated through a controllable data synthesis framework. The training objective jointly supervises intermediate tool usage and final outcome accuracy, incentivizing our reward model to learn effective evidence-based judgment strategies. Extensive experiments on three newly-collected datasets and two widely-used benchmarks demonstrate that OpenRM substantially outperforms existing reward modeling approaches. As a further step, we integrate OpenRM into both inference-time response selection and training-time data selection. This yields consistent gains in downstream LLM alignment tasks, highlighting the potential of tool-augmented reward models for scaling reliable long-form evaluation.

**arXiv ID:** 2510.24636
</details>

<details>
<summary><strong>Gavel: Agent Meets Checklist for Evaluating LLMs on Long-Context Legal Summarization</strong> - Yao Dou, Benjamin Mamut, Wei Xu - [[pdf]](https://arxiv.org/pdf/2601.04424)</summary>

**Abstract:** Large language models (LLMs) now support contexts of up to 1M tokens, but their strengths and weaknesses on complex long-context tasks remain unclear. To study this, we focus on multi-document legal case summarization, where a single case often spans many documents exceeding 100K tokens. We systematically evaluate 12 frontier LLMs with Gavel, which consists of Gavel-Ref, a reference-based evaluation framework with checklist, residual-fact, and writing-style evaluations, and Gavel-Agent, a reference-free agent for evaluating factual coverage directly from source documents. Our results show that current models are more prone to omitting key information than hallucinating. They all perform well on simple checklist items, such as filing date, but struggle with rare and complex items, such as settlements. Performance also declines as case length increases. To meta-evaluate Gavel, we collect 160 hours of human annotations. Gavel-Agent reduces token usage by at least 36% compared to end-to-end and chunk-by-chunk methods while achieving competitive performance. Gavel-Agent also generalizes to the medical domain, performing the best with at least 77% fewer tokens.

**arXiv ID:** 2601.04424
</details>

<details>
<summary><strong>SlowBA: An efficiency backdoor attack towards VLM-based GUI agents</strong> - Junxian Li, Tu Lan, Haozhen Tan, Yan Meng, Haojin Zhu - [[pdf]](https://arxiv.org/pdf/2603.08316)</summary>

**Abstract:** Modern vision-language-model (VLM) based graphical user interface (GUI) agents are expected not only to execute actions accurately but also to respond to user instructions with low latency. While existing research on GUI-agent security mainly focuses on manipulating action correctness, the security risks related to response efficiency remain largely unexplored. In this paper, we introduce SlowBA, a novel backdoor attack that targets the responsiveness of VLM-based GUI agents. The key idea is to manipulate response latency by inducing excessively long reasoning chains under specific trigger patterns. To achieve this, we propose a two-stage reward-level backdoor injection (RBI) strategy that first aligns the long-response format and then learns trigger-aware activation through reinforcement learning. In addition, we design realistic pop-up windows as triggers that naturally appear in GUI environments, improving the stealthiness of the attack. Extensive experiments across multiple datasets and baselines demonstrate that SlowBA can significantly increase response length and latency while largely preserving task accuracy. The attack remains effective even with a small poisoning ratio and under several defense settings. These findings reveal a previously overlooked security vulnerability in GUI agents and highlight the need for defenses that consider both action correctness and response efficiency. Code can be found in this https URL.

**arXiv ID:** 2603.08316
</details>

<details>
<summary><strong>Task-Relevant Representation Decoupling for Visual Reinforcement Learning Generalization</strong> - Jinwen Wang, Youfang Lin, Xiaobo Hu, Qian Xu, Shuo Wang, Zhuo Chen, Kai Lv - [[pdf]](https://arxiv.org/pdf/2607.00796)</summary>

**Abstract:** Visual Reinforcement Learning (VRL) has achieved considerable success in solving control tasks. However, generalizing learned policies to new environments remains a major challenge, as agents often overfit to task-irrelevant features in the training environment. To solve this problem, we introduce the concept of decoupling observations into task-relevant and task-irrelevant representations. Building on this idea, we propose a self-supervised Task-Relevant Representation Decoupling (T2RD) algorithm for VRL. This algorithm consists of three components: task-relevant representation consistency, cross-reconstruction, and cross-dynamic prediction. The first two components achieve the decoupling of content and style features, but the resulting content representations are not necessarily task-relevant. To further refine task-relevant features from content representations, we design the third component that introduces dynamic prediction. T2RD achieves State-Of-The-Art (SOTA) generalization performance and sample efficiency in the DeepMind Control Suite and Robotic Manipulation tasks.

**arXiv ID:** 2607.00796
</details>

<details>
<summary><strong>Local Motion Matters: A Deconstruct-Recompose Paradigm for Reinforcement Learning Pre-training from Videos</strong> - Jinwen Wang, Youfang Lin, Xiaobo Hu, Shuo Wang, Kai Lv - [[pdf]](https://arxiv.org/pdf/2607.00808)</summary>

**Abstract:** Pre-training on large-scale videos to improve reinforcement learning efficiency is promising yet remains challenging. Existing methods typically treat the agent as an indivisible entity, modeling motion patterns globally. Such global modeling is tightly coupled with the morphology, hindering transfer across domains. In contrast, despite the vast disparity in global motions, the local components exhibit similar motion patterns across different agents. Building on this insight, we propose a novel Deconstruct-Recompose Paradigm (DRP) for learning transferable local motion representations. Specifically, in the Deconstruct phase, we identify multiple local points and track their frame-wise motions, defining each as an Atomic Action. We introduce a Dual-Attention Encoder (DAE) to learn local motion representations from these Atomic Actions, capturing their spatiotemporal relationships. In the Recompose phase, we compose local motion representations with a learnable Motion Aggregation Token [MAT] via latent dynamics model learning. Additionally, an adapter bridges local motion and downstream action-specific dynamics to accelerate policy learning. Extensive experiments demonstrate that our method effectively transfers to diverse robotic control and manipulation tasks, significantly improving sample efficiency and performance.

**arXiv ID:** 2607.00808
</details>

<details>
<summary><strong>From Pixels to Temporal Correlations: Learning Informative Representations for Reinforcement Learning Pre-training</strong> - Jinwen Wang, Youfang Lin, Xiaobo Hu, Siyu Yang, Sheng Han, Shuo Wang, Kai Lv - [[pdf]](https://arxiv.org/pdf/2607.00811)</summary>

**Abstract:** Unsupervised pre-training on large-scale datasets has demonstrated significant potential for improving the sample efficiency and performance of Reinforcement Learning (RL). Given the large-scale action-free internet videos, existing methods utilize single-step transition prediction and image reconstruction to learn representations. However, these methods prefer to preserve large-proportion stationary information in the pixel space, neglecting small but crucial information. To preserve enough information in the representation, it is essential to pay equal attention to each element in videos. Specifically, we propose a temporal correlation space to distinguish each element. For implementation, we introduce the Multi-scale Temporal Contrastive Learning (MTCL) method to model multi-scale temporal correlations separately. This approach can balance the attention of different elements and yield more informative representations, effectively supporting policy learning in various downstream tasks. Experimental results demonstrate that our method improves sample efficiency and asymptotic performance across various downstream tasks.

**arXiv ID:** 2607.00811
</details>

<details>
<summary><strong>Inverse Reinforcement Learning for Interpretable Keystroke Biomarkers in Parkinson's Disease</strong> - Navin Bondade - [[pdf]](https://arxiv.org/pdf/2606.25270)</summary>

**Abstract:** Keystroke dynamics have been explored extensively as a passive digital biomarker for Parkinson's disease (PD), typically by extracting summary statistics from typing timing and training a classifier to discriminate PD from healthy controls. We instead apply inverse reinforcement learning (IRL) to keystroke data, modeling each keystroke as a discrete choice over typing speed and recovering, per subject, an interpretable reward function that explains their observed timing behavior. To our knowledge this is the first application of IRL to keystroke dynamics. On the public neuroQWERTY MIT-CSXPD dataset (85 subjects, 42 with PD), an initial four-parameter reward decomposition (speed, effort, smoothness, hand-alternation cost) was found to suffer severe feature collinearity between two terms ($r=1.000$ in typical contexts); we diagnose and correct this, yielding an identifiable three-parameter model. The recovered speed-preference weight correlates with UPDRS-III severity at $r=-0.607$ ($p<0.001$, $n=42$), replicates independently across two sub-cohorts, is stable across nine sensitivity configurations, and retains a statistically significant contribution beyond raw typing speed alone (incremental $R^2$ from 0.194 to 0.338, $p=0.006$). Two other recovered weights (consistency, hand-alternation) did not survive confound checks and are reported as negative results. We document two implementation bugs found during adversarial code review (session-boundary contamination, a rolling-window data leakage) and show the headline result is materially unchanged after fixing both. We discuss this result in the context of a literature where reported accuracies vary widely between studies (pooled AUC 0.85, I^2=94% in a 2022 meta-analysis), and argue that the validation process itself, not only the correlation coefficient, is part of the contribution.

**arXiv ID:** 2606.25270
</details>

<details>
<summary><strong>Reward function compression facilitates goal-dependent reinforcement learning</strong> - Gaia Molinaro, Anne G. E. Collins - [[pdf]](https://arxiv.org/pdf/2509.06810)</summary>

**Abstract:** Humans can uniquely assign value to novel, abstract outcomes to support reinforcement learning. However, this flexibility is cognitively costly and reduces learning efficiency. We propose that goal-dependent learning initially relies on capacity-limited working memory. With consistent experience, learners create a "compressed" reward function - a simplified goal rule -- that transfers to long-term memory for a more automatic evaluation upon receiving feedback. This automaticity frees working memory resources, thereby boosting learning efficiency. Across six experiments, we demonstrate that learning is impaired by the size of the goal space but improves when this space allows for compression. Additionally, faster reward processing correlates with better learning. Although the algorithmic details remain to be established, our behavioral results and computational models suggest that efficient goal-directed learning relies on compressing complex goal information into a stable reward function. These findings illuminate the cognitive mechanisms of intrinsic motivation and can inform behavioral interventions supporting human goal achievement.

**arXiv ID:** 2509.06810
</details>

<details>
<summary><strong>EmbodimentSemantic: A Spatial Scene-Graph Dataset and Benchmark for Vision-Language Models on Embodied Manipulation Trajectories</strong> - Hassan Jaber, Refinath S N, Luca Cagliero, Christopher E. Mower, Haitham Bou-Ammar - [[pdf]](https://arxiv.org/pdf/2607.00020)</summary>

**Abstract:** Spatial grounding remains a key limitation of vision-language-action (VLA) systems for robotic manipulation. While current models can recognize objects and follow language instructions, they often lack an explicit representation of how objects are arranged in space, including support, containment, ordering, occlusion, and depth-sensitive relations. We introduce EmbodimentSemantic, a spatial scene-graph dataset and benchmark for evaluating relational grounding in embodied manipulation. EmbodimentSemantic represents scenes as directed object-relation-object triplets, where each triplet specifies a spatial relation between an ordered pair of objects using a fixed set of relations. This representation enables direct evaluation of object binding, relation prediction, and spatial consistency. The dataset includes real-world manipulation observations collected with the low-cost SO101 robot arm, together with generated scene graphs for studying spatial grounding in practical robotic settings. To provide controlled validation, we also introduce a simulator-grounded LIBERO benchmark with over 60K manipulation frames and more than 120K camera-specific scene graphs across paired third-person and wrist views, where ground-truth relations are derived automatically from MuJoCo geometry, world coordinates, camera projections, and visibility constraints. We further test whether scene graphs improve downstream control by injecting them into existing VLA policy prompts. Experiments across open-source and commercial VLMs show that current models often predict plausible relations but struggle with exact depth-aware and viewpoint-dependent spatial structure. EmbodimentSemantic provides a unified framework for diagnosing spatial grounding in VLM perception and testing its utility for VLA manipulation.

**arXiv ID:** 2607.00020
</details>

<details>
<summary><strong>Learning Expert Strategy for Autonomous Robotic Endovascular Intervention via Decoupled Procedural Execution</strong> - Yanxi Chen, Tianliang Yao, Shaolong Tang, Jiyuan Zhao, Hengyu Hu, Zhaoxing Li, Antonio J. Sánchez Egea, Peng Qi - [[pdf]](https://arxiv.org/pdf/2607.00066)</summary>

**Abstract:** Endovascular interventions are high-stakes procedures requiring precise device operation within complex and tortuous vascular anatomies. Autonomous endovascular navigation has the potential to standardize procedural quality and reduce the performance variability inherent in manual operation. Although Reinforcement Learning (RL) approaches have demonstrated promise in enabling autonomy in endovascular intervention, they often struggle with explicit constraint satisfaction and safety guarantees. To address these challenges, a learning-based expert strategy is introduced, enhancing procedural consistency in autonomous endovascular intervention by explicitly decoupling high-level strategic decision-making from low-level procedural execution. The proposed framework replicates the expert clinical decision-making process: a strategic RL policy generates global navigation intents, which are subsequently refined through an expert-informed execution module. This module ensures that robot movements strictly adhere to expert operational norms, real-time kinematic limits, and vessel safety constraints. Experimental evaluation across high-fidelity 3D simulations and a real-world robotic platform demonstrates that the proposed framework not only outperforms baseline policies but also effectively replicates expert-level proficiency. The framework achieves a high navigation success rate (> 96%) and a 29.3% reduction in operational steps, which translates to enhanced operative efficiency and minimized device-vessel interaction. Furthermore, a 13% reduction in trajectory variance indicates superior procedural standardization, aligning autonomous behavior with established clinical norms. These results underscore its potential to enhance the predictability, safety, and consistency of robotic endovascular interventions.

**arXiv ID:** 2607.00066
</details>

<details>
<summary><strong>Distributed Multi Robot Lunar Cargo Transportation via Phase Decomposed Reinforcement Learning</strong> - Ashutosh Mishra, Elian Neppel, Shreya Santra, Antoine Jonquières, Muhammad Athallah Naufal, Kentaro Uno, Kazuya Yoshida - [[pdf]](https://arxiv.org/pdf/2607.00160)</summary>

**Abstract:** Modular reconfigurable robotic systems provide a scalable solution for cooperative surface operations in future lunar missions. However, cooperative cargo transportation remains challenging due to morphology-dependent topology changes, strong payload-induced coupling, long-horizon decision making, and safety constraints. This paper proposes a phase-decomposed reinforcement learning framework for cooperative cargo transport with distributed robotic units. The task is decomposed into lifting, transportation, and placement, each optimized with a dedicated joint-state policy capturing inter-agent coupling. Centralized training promotes stable convergence, while deployment uses onboard proprioception for control and OptiTrack motion capture for ground-truth evaluation and post-processed metrics. A deterministic phase controller expressed in Markov state representation regulates transitions between stages, and a failure-sensitive synchronization mechanism ensures coordinated progression and safety-aware halting during real-world execution. The framework is evaluated in simulation and through controlled field experiments at a JAXA space exploration test facility. Results demonstrate reliable cooperative transport across all stages in both simulation and hardware experiments.

**arXiv ID:** 2607.00160
</details>

<details>
<summary><strong>VLM-AR3L: Vision-Language Models for Absolute and Relative Rewards in Reinforcement Learning</strong> - Kuan-Chen Chen, Winston Chen, Wei-Fang Sun, Min-Chun Hu - [[pdf]](https://arxiv.org/pdf/2607.00483)</summary>

**Abstract:** Designing effective reward functions remains a major challenge in reinforcement learning (RL), particularly in open-ended environments where task goals are abstract and difficult to quantify. In this work, we present VLM-AR3L, a framework that leverages Vision-Language Models (VLMs) to provide both absolute and relative rewards for RL. VLM-AR3L interprets an agent's visual observations in the context of a natural language task goal, and learns both absolute and relative rewards from VLM-generated preference labels. The absolute reward model predicts scalar evaluations for individual states, while the relative reward model compares consecutive observations to infer progress or regression toward the task goal. Their integration combines the stability of state-based evaluation with the robustness of comparative supervision. We evaluate VLM-AR3L across benchmarks spanning classic control, manipulation, and open-world embodied tasks, with a particular focus on Minecraft given its visual complexity and long-horizon decision-making requirements. Experimental results show that VLM-AR3L consistently outperforms prior VLM-based reward learning methods.

**arXiv ID:** 2607.00483
</details>

<details>
<summary><strong>BiliVLA: Scene-Aware Vision-Language-Action Model with Reinforcement Learning for Autonomous Biliary Endoscopic Navigation</strong> - Jinsong Lin, Chi Kit Ng, Zhiyong Xiong, Zikang Pan, Yihan Hu, Tabassum Tamima, Ziyi Hao, Eddie Cheung, Jiewen Lai, Huxin Gao, Hongliang Ren - [[pdf]](https://arxiv.org/pdf/2606.23531)</summary>

**Abstract:** Endoscopic retrograde cholangiopancreatography (ERCP) demands precise endoscopic navigation and stable biliary cannulation within a narrow monocular field characterized by specular reflections, partial occlusions, and frequent tissue contact. Although recent robotic systems and vision-based assistance techniques improve operator ergonomics and provide perceptual cues, their performance degrades under pronounced anatomical variability and safety-critical visual artifacts, which hinders reliable autonomy in cannulation-grade procedures. Here, we present BiliVLA, a scene-aware Vision-Language-Action (VLA) framework that formulates biliary endoscopic navigation as an instruction-conditioned visuomotor learning problem. Given an endoscopic observation and a stage-specific language instruction, BiliVLA jointly predicts the target category, a grounded bounding box, and a discrete three degrees of freedom (DoF) motor command for a continuum endoscope. The proposed framework incorporates scene-aware supervision to enhance semantic target consistency and safety-aware recovery supervision to induce conservative retreat behaviors under luminal wall contact. A key component of BiliVLA is a two-stage training paradigm that combines grounding-enhanced supervised fine-tuning (SFT) with Group Relative Policy Optimization (GRPO), which significantly improves action reliability and decision consistency during closed-loop navigation. Across three ERCP subtasks, BiliVLA achieves an average action precision of 91.96\% and an overall success rate (SR) of 84.85\% in real-world phantom experiments. These results indicate that integrating semantic grounding, scene-aware learning, and reward-guided optimization improves perception-action alignment and enables robust autonomous endoscopic navigation.

**arXiv ID:** 2606.23531
</details>

<details>
<summary><strong>Building a Scalable, Reproducible, Evaluatable, and Closed-Loop Simulation Environment Foundation for Embodied Intelligence</strong> - Junwu Xiong, Yongjian Guo, Mingxi Luo, Ning Qiao, Lei Kang, Song Wang, Yince Gao, Chenfeng Gu, Zhen Sun, Haoran Li, Wei Lu, Yucheng Guo, Shuai Di, Xiaodong Bai, Haoran Sun, Jing Long, Jiaxuan Gao, Hui Zhang, Peng Hao, Lu Lu - [[pdf]](https://arxiv.org/pdf/2606.27962)</summary>

**Abstract:** This paper presents a cloud-native simulation infrastructure framework for embodied intelligence that supports large-scale training, standardized evaluation, and simulation-based data collection. The framework unifies simulation environment generation, task execution, trajectory collection, model evaluation, data management, and cloud services into a scalable and reproducible platform. To address the high cost, limited scalability, and poor reproducibility of real-world robotic data collection, the framework adopts cloud-native technologies including elastic resource scheduling, containerized simulation, unified data management, and service-oriented system design, enabling efficient large-scale simulation for multi-model and multi-task workloads. Built on a four-layer architecture, the framework provides standardized environment assets, automated task generation, trajectory collection, benchmark evaluation, and closed-loop data optimization. It further integrates representative systems including D-VLA, RL-VLA3, Sword, and Pre-VLA to support scalable simulation, dynamic scheduling, visual augmentation, and real-time data filtering. We argue that cloud-native simulation infrastructure provides a unified foundation for data generation, model training, standardized evaluation, and real-world deployment, and will play a key role in the future development of embodied intelligence.

**arXiv ID:** 2606.27962
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
