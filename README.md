# Agent arXiv Daily

**Last Updated:** 2026-11-02 05:53:42

**Total Papers:** 13

## Table of Contents

- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Benchmarks and Datasets (4 papers)</h2></summary>

<details>
<summary><strong>More Rules, Longer Lists? Benchmarking Self-Evolving Agents on a Hidden-Rule Curriculum</strong> - Chenjie Qiu, Boyan Xu, Shaobin Shi, Bingfeng Chen, Ruichu Cai, Dongsheng Luo - [[pdf]](https://arxiv.org/pdf/2610.40758)</summary>

**Abstract:** Self-evolving agents rewrite a persistent state from feedback, but it is largely untested whether this lets them learn a rule nobody told them. An agent can also pass by writing down each correction as an exception, and its list then grows with every new rule. We introduce CLR-Bench, a Curriculum of Latent Rules benchmark. Each chain adds one hidden rule per level on top of the earlier ones. A rule acts on an opaque label that stands for a named grouping published on the Web, such as the founding members of an airline alliance. The task gives one example value, practice feedback reveals one more member and one non-member, and the test checks unseen members and similar non-members. Probes certify that the base model knows the members of each grouping and does not guess it without feedback, and that the evidence identifies the grouping once labels are read as named groupings. An automated pipeline builds chains from a new data domain, and we release 20 chains over 10 domains. We evaluate six recent self-evolution methods. With the certifying model they barely learn. The best gain at most 5 points over a frozen agent and exclude only 6-12% of the unseen members from the count, fewer than the 20% reached by simply keeping the raw practice history. A fixed lesson that states the named-grouping convention lifts the same methods by 14.5 to 16.5 points. Their states hold the revealed member in 75-98% of cases and almost never the class name, and when practice exposes a member the list misses, they add that member to the list. With the stronger GPT-6-Sol, ACE and Dynamic Cheatsheet learn from feedback alone. They reach 78% accuracy and solve 59-60% of the single cells, against 53% and 8% with the certifying model, and they surpass the raw history. CLR-Bench therefore measures how well an agent learns, and this ability differs across models.

**arXiv ID:** 2610.40758
</details>

<details>
<summary><strong>Benchmarking Forecasting Agents on Yesterday's Tomorrows</strong> - Yuxuan Wang, Yishu Wang, Jiaqi Deng - [[pdf]](https://arxiv.org/pdf/2610.40940)</summary>

**Abstract:** Digital agents can now gather and organize information, reason over it, and use tools to complete complex tasks. Forecasting puts all of these abilities to the test: it requires integrating incomplete evidence, inferring causal drivers, and committing to calibrated probabilistic judgments. Existing benchmarks are poorly suited to measuring and improving this ability. Live benchmarks provide feedback only after events resolve, so every evaluation cycle waits on the calendar; fixed-corpus evaluations return results immediately but confine research to pre-collected documents. We introduce FABench, a scalable backtesting environment for Forecasting Agents. An automated pipeline discovers events, generates questions, and resolves outcomes for any past forecast date, so the benchmark can be renewed to stay ahead of model training cutoffs; its first release poses 287 questions across nine domains. Agents research each question freely in an archive environment that combines an offline archive with live web search screened to prevent temporal leakage. Evaluating agents built on eight recent models, we find that retrieval improves the Brier score of all eight in every retrieval setting and lifts the leading agents to the market consensus on the 54 questions linked to Polymarket, with the best three scoring slightly above it. Research traces reveal recurring failures in how agents weigh and apply evidence, leaving ample room for improvement. We release our code, data, and archive environment as a foundation for community research on forecasting agents.

**arXiv ID:** 2610.40940
</details>

<details>
<summary><strong>ReplayRCA: Measuring Post-hoc Diagnosis and Live Oversight of Agent Failures by Counterfactual Replay</strong> - Jeremy Canale, Fabien Soulis - [[pdf]](https://arxiv.org/pdf/2610.41025)</summary>

**Abstract:** When a language-model agent fails a multi-step task, can another model find the decisive mistake and fix it with a single action? We introduce ReplayRCA, a benchmark that scores such diagnoses by intervention rather than by agreement with annotations: the environment is restored to the state preceding the chosen step, the proposed action replaces the agent's own, and the same agent continues. The score, causal repair, is the continuation success rate minus that of a paired placebo that replays the agent's original action through the same mechanism. Two tracks share this protocol: a post-hoc autopsy, where the diagnoser sees the whole failed run and is told that it failed, and live oversight, where a supervisor watches the run unfold and may intervene once, before an irreversible action, without knowing whether the run will fail. Tasks come from OpsWorld, a deterministic, procedurally generated back-office environment with six difficulty levels, the last of which is multimodal (QR codes, handwriting). On 62 calibrated systematic failures, the best of 13 frontier models (Grok 4.7) reaches a causal repair of +51.3 points; the top eight models are statistically indistinguishable, the flagship models of the two largest vendors rank 8th and 12th, and 5 items are repaired by no model. Forcing a strong model to intervene at the last step costs it 17.1 points, so the last-step shortcut of annotation-based attribution benchmarks does not transfer. On 87 runs, the two best live supervisors are statistically tied and one of them is a small open-weight model (DeepSeek V4.1 Flash); the supervisor that is among the best post-hoc diagnosers (GPT-6.1 Sol) is the weakest live one (-14.5 points on the same runs, p = 0.002), so post-hoc diagnostic performance can be a poor proxy for live oversight. The environment and the evaluation code are released at this https URL.

**arXiv ID:** 2610.41025
</details>

<details>
<summary><strong>LanG 2.0: Energy-Efficient Agentic AI for Semi-Automated Security Operations at Scale</strong> - Anes Abdennebi, Nadjia Kara, Laaziz Lahlou, Hakima Ould-Slimane - [[pdf]](https://arxiv.org/pdf/2610.41072)</summary>

**Abstract:** Security operations centers (SOCs) need to turn terabytes of alerts, logs and flows into trusted, affordable decisions to unveil hidden threats. We present LanG 2.0, an agentic system that detects threats, correlates evidence and writes rules, investigating whether small local models can replace frontier models affordably and safely when structural safeguards, not model accuracy, provide the guarantees. It incorporates entity-anchored correlation, human approval of rule changes and a tamper-evident ledger. Tests span eight flow datasets from five networks, real XDR alerts and multistage campaigns. Against its predecessor, campaign correlation precision rises from 18.75% to 51.81%, and rule acceptance from at most 60.00% to 98.33%. Against Claude Opus 5.5, scoring flows against each network's normal traffic detects attacks on unseen networks at 73.42% F1 without false positives, versus 58.10% with 49.00% false positives, for at least 650 times less energy. Finetuned models statistically match it on alert classification and rule writing for 5.6 to 59 times less energy. The work demonstrates LanG's practical scalability, processing 3.33 million records in 6.4 minutes, over 30 million hourly, and its safeguards resisted prompt injection, label poisoning and ledger tampering, so small local models can carry most SOC detection affordably and safely.

**arXiv ID:** 2610.41072
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>Agentic World Modeling for Agent Test-time Scaling without Environment Replica</strong> - Wendi Li, Huseyin A. Inan, Wei-Ning Chen, Yanzhi Chen, Chetan Bansal, Saravan Rajmohan, Sharon Li - [[pdf]](https://arxiv.org/pdf/2610.40927)</summary>

**Abstract:** Test-time scaling (TTS) improves large language model agents by sampling and selecting among multiple candidate trajectories, but common realizations assume that these trajectories can be executed in parallel. This assumption breaks in realistic deployments, where enterprise workspaces and cloud services are shared and persistent, costly or infeasible to replicate, and actions may introduce irreversible side effects or cross-trajectory interference. Existing generative world models avoid direct execution, but typically compress environment state into a single prompt, limiting their ability to reliably simulate transitions under partial observability. Instead, we introduce Agentic World Modeling (AWM), which reformulates next-observation prediction as a multi-turn evidence-acquisition task. For each hypothetical action, an agentic simulator adaptively queries the real environment through safe read-only interfaces, reasons over the returned evidence, and uses executable computation to construct the resulting observation without mutating shared state. Integrated with trajectory-level TTS, AWM enables parallel exploration over a single shared environment. Across $\tau$-bench, AppWorld, and OfficeBench, AWM outperforms a single-shot simulation baseline by 40.07% on average, and surpasses a trained world-model baseline by 12.36% on average. These results establish interactive, read-grounded simulation as a practical foundation for safe and scalable agentic test-time scaling in shared, non-resettable environments.

**arXiv ID:** 2610.40927
</details>

<details>
<summary><strong>Capability-Grounded Agentic Planning</strong> - Shikha Shrestha, William Yeoh, Ning Zhang, Yevgeniy Vorobeychik - [[pdf]](https://arxiv.org/pdf/2610.41040)</summary>

**Abstract:** LLM agents serve natural-language requests through tools, data sources, and other agents, and even simple requests often need several interdependent calls. Typically, the LLM picks these calls by trial and error, with no guarantee of an executable plan. We present Capability-Grounded Agentic Planning (CGAP), which instead reasons about what a request requires and what tools provide. CGAP compiles the tool catalog deterministically into a PDDL model, uses an LLM to turn the request into goals, and solves the model with a classical planner. Since tool outputs are unknown in advance, the model is abstract, tracking which inputs are available, not their values. During execution, an LLM adds entity-specific calls that abstract plans omit, and deterministic checks reject any that break the plan. Every argument is drawn from a cited source, such as a conversation span or an earlier tool response, and each requested result counts as delivered only after an LLM verifier checks it against the request. With three LLMs on ToolSandbox, ComplexFuncBench, BFCL multi-turn, and $\tau^2$-bench retail, CGAP outperforms a standard tool-calling agent by up to 28.9 points.

**arXiv ID:** 2610.41040
</details>

<details>
<summary><strong>BABA-Bench: Evaluating Long-Horizon Patient-Advocacy Agents</strong> - Shriram Vasudevan, Videet Mehta, Connor Sweeney - [[pdf]](https://arxiv.org/pdf/2610.41076)</summary>

**Abstract:** An LLM agent that helps a patient get care must do two things. It must reach the correct outcome, and it must follow the necessary procedure. For example, the agent must get the patient's permission before it shares the patient's records. We introduce BABA-Bench, a benchmark of 47 interactive worlds. We built each world from a de-identified, real patient advocacy case that expert advocates reviewed. In each world, the agent must find hidden information and coordinate insurers, clinics, suppliers, pharmacies, and community organizations. The engine checks each action at the time the agent does it. It checks if the agent has permission, if the agent has evidence for the action, and if the action is prohibited. We evaluated 14 frontier models with three runs on each world, for a total of 1,974 runs. Agents often reached the correct outcome by an invalid route. Of the 921 runs that reached all required outcomes, 254 (28%) did not follow the necessary procedure. Claude Opus 5 reached all required outcomes in 81.6% of its runs, but only 52.5% of its runs passed all checks. When we gave two models a verified route, their pass rates increased from 29.8% to 91.5% and from 44.0% to 97.2%. For these models, the main problem is to find a route, not to execute it.

**arXiv ID:** 2610.41076
</details>

</details>

<details open>
<summary><h2>Other Agent Research (1 papers)</h2></summary>

<details>
<summary><strong>Towards Portable Specification of the Agent Harness</strong> - Nicolas Gertler, Claudio Novelli, Luciano Floridi - [[pdf]](https://arxiv.org/pdf/2610.40917)</summary>

**Abstract:** Large language models act only through the infrastructure that surrounds them. That infrastructure, the harness, determines what a model can accomplish and how it fails, yet it has no settled definition. To address this gap, we decompose the harness into eight functional components that can be specified and compared across systems, and we propose a portable specification framework that records which model ran in which harness. The record is needed because different harness configurations can yield several-fold differences in reported scores for the same model, and because much of an agent's attack surface lies in its harness. Existing model cards and agent cards provide release and operational information, but do not establish the sequence of effective configurations within a run. Harnesses can change between releases and between runs, and, where agents edit their own tools and configuration, within a run. Enacted AI laws contain documentation duties relevant to covered systems, subject to their respective scopes and application dates, but do not prescribe the common harness record proposed here. Therefore, the harness has to be explicitly described and specified as a versioned part of the system.

**arXiv ID:** 2610.40917
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (5 papers)</h2></summary>

<details>
<summary><strong>Does Recursive Knowledge Construction Help Agents Verify Consensus Claims in Science?</strong> - Lun Ai, Yu-Heng Wu, Jonathan Schaul, Leonie Küchenhoff, Denes Turei, Francesco Carli, Fabio Petroni, Aurelien Dugourd, Julio Saez-Rodriguez - [[pdf]](https://arxiv.org/pdf/2610.40906)</summary>

**Abstract:** Recent scientific agents build knowledge recursively by retrieving evidence, constructing an intermediate state, and judging whether that state contains sufficient information. These agents are evaluated end-to-end, usually without ground truth, so the effect of each design choice is unknown. Here, we examined two design axes: (1) what the recursion builds (the state of evidence) and (2) what controls it (the judge that decides when to stop). To measure them, we defined consensus scientific claim verification and measured these effects on this task, where a system must establish a claim's consensus stance from the open literature. We built a new consensus claim dataset, ProClaim-eval, including 718 claims with SUPPORT and REFUTE gold labels from five expert-curated databases spanning molecular biology, nuclear physics, and astronomy. Our instrument, ProClaim, persists an evidence state outside the model's context, so each axis can be studied with the other held fixed. Against verification systems and general-purpose agent harnesses, ProClaim ranks among the top three, with a significant advantage over the harnesses. We found that the evidence state improves performance over in-context accumulation, and it also matters what is stored in it. In addition, a trained judge that reads the evidence state performs better than fixed-round stopping. A judge trained separately and never exposed to ProClaim-eval achieves the highest agreement with the gold labels on three of five subsets, where it matches the off-the-shelf Claude Sonnet~4.6 judge and outperforms Claude Haiku~4.5. Our results support recursively maintaining a structured evidence state outside the model's context as an effective design for verifying claims in scientific agents.

**arXiv ID:** 2610.40906
</details>

<details>
<summary><strong>P-ContextBench: From Personal Fact Recall to Personalized Agent Behavior</strong> - Grant Nasir Brown, Abdul Basir Bari, Anmol Tadikonda, Isabel Song - [[pdf]](https://arxiv.org/pdf/2610.40959)</summary>

**Abstract:** Large language models (LLMs) are increasingly used for personal tasks, yet evaluating their safe use of personal context remains difficult because such contexts are private, fragmented, dynamic, and socially situated. We introduce P-ContextBench, a realistic, verifiable, and scalable benchmark for evaluating personal-context understanding. P-ContextBench constructs population-structured, life-story-grounded digital humans from demographic seeds reflecting the U.S. population. Rather than using explicit, static profiles, P-ContextBench embeds traits, routines, goals, and constraints as implicit evidence across fragmented, evolving synthetic mobile-app records. It evaluates five capabilities: single-source fact recovery, multi-source fact recovery, temporal state tracking, inference of higher-level personal patterns from routine behavior, and safe handling of requests conflicting with persona-specific constraints through compliance, warning, clarification, or refusal. A multi-stage validation pipeline checks persona plausibility, context recoverability, information leakage, evidence preservation, test-case grounding, and human-judge agreement. We instantiate P-ContextBench with 100 personas, 10,088 hidden facts, 18,261 app-trace fragments, and 4,810 verified test cases. The pipeline supports configurable generation across scales and demographic groups. Cost-effective state-of-the-art (SOTA) LLMs relevant to personal agents achieve below 60% accuracy on temporal tracking, higher-level pattern inference, and constraint-aware action-boundary decisions, with fewer than 10% of test cases answered correctly across all three repeated runs. Even a high-performing SOTA model remains inconsistent on these capabilities. P-ContextBench provides measurable targets for improving the accuracy, reliability, and safety of LLM-based personal agents.

**arXiv ID:** 2610.40959
</details>

<details>
<summary><strong>TSFLab: A Self-Improving Agent-Agnostic Workbench for Time Series Forecasting Research</strong> - ChengAo Shen, Yanjun Zhao, Haoxin Liu, Bonnie Chen, Miles Ma, Dongjin Song, Hanghang Tong, Qingsong Wen, Jingrui He, Jingchao Ni - [[pdf]](https://arxiv.org/pdf/2610.40901)</summary>

**Abstract:** As publications and products on time series forecasting (TSF) proliferate, researchers face a growing burden of analyzing existing work. First, not all publications release code. For instance, only about 20% of arXiv TSF papers have a code link in the surveyed period. Second, released code also requires manual maintenance. Thus, TSF researchers must constantly adapt or implement existing works. However, independent implementations can differ in experimental settings and are difficult to maintain. We therefore propose to use AI agents for benchmarking TSF research and even supporting autoresearch, e.g., developing new TSF models. Yet directly applying existing coding agents, such as Claude Code and Codex, is suboptimal: prohibitive costs, limited scalability to large research projects (e.g., benchmarking), and lack of self-maintenance. To address these challenges, we propose TSFLab, a self-improving agent-agnostic "workbench" that equips compatible coding agents with shared resources and tools for low-cost TSF research and continuous maintenance of research resources. Our experiments cover reproduction, benchmarking, and new method development. In reproduction and benchmarking, TSFLab improves task completion by 20-163% and cuts fresh tokens per completed task by 11-66% across baselines. A pilot study of agent-driven method development also shows improved forecasting accuracy via TSFLab, with 32.8-73.6% fewer fresh tokens and 9.2-68.8% less development time than the other settings.

**arXiv ID:** 2610.40901
</details>

<details>
<summary><strong>On the Origin and Evolution of Training-Inference Mismatch in LLM Reinforcement Learning</strong> - Yu Li, Yufa Zhou, Tian Lan - [[pdf]](https://arxiv.org/pdf/2610.40945)</summary>

**Abstract:** On-policy reinforcement learning for large language models assumes that training responses are sampled from the current policy. In practice, an inference engine samples the responses and a separate trainer computes the gradients, and the \textbf{training-inference mismatch} between the two makes the training data off-policy. The mismatch has two sources, the numerics of the engine and the stale weights of asynchronous sampling, and a correction that stabilizes training under one source can fail under the other. \textit{We ask which property of a mismatch source decides the correction it needs.} Evaluating the trainer at the sampler's weights splits each sampled-token error exactly into a \textit{numerical discrepancy} of the engine and a \textit{policy lag} of the trainer, and we follow both through training. The numerical discrepancy stays approximately constant, so it biases the updates by a nearly fixed amount, and centering the score at each prefix removes its leading bias. The policy lag is produced by the trainer's own updates: the bias it leaves enters the next update, which enlarges the next lag and closes a feedback loop. Importance weighting cuts this loop only while the lag ratio stays concentrated, and on Qwen3-1.7B it fails at the step where the effective sample size of the ratio falls below one half. From this account we derive the \textbf{discrepancy-lag correction} (DLC), which centers the numerical discrepancy, clips the lag ratio against the sampler's weights and resynchronizes the sampler when this effective sample size falls below one half. Under a stale sampler, DLC keeps the KL divergence from on-policy training within twice the KL between two independent on-policy trainings on Qwen3-0.6B and Qwen3-1.7B, and its resynchronization prevents collapses that the per-token corrections leave and that a KL measured on the sampler's prefixes does not detect.

**arXiv ID:** 2610.40945
</details>

<details>
<summary><strong>Density-Guided Bootstrap Boundaries for Latent World-Model Reinforcement Learning</strong> - Shashank Gupta, Pilhwa Lee - [[pdf]](https://arxiv.org/pdf/2610.41022)</summary>

**Abstract:** Latent world models allow reinforcement learning agents to learn policies from imagined trajectories, yet determining the optimal depth of model reliance remains a fundamental challenge. Fixed imagination horizons apply uniform prediction depths regardless of trajectory reliability, exposing target values to compounding model error or over-relying on critic approximations. We introduce density-guided bootstrap boundaries, a trajectory-dependent mechanism that adapts the depth of imagined returns using a score-informed relative density proxy. The proxy accumulates the estimated divergence of a score-corrected auxiliary field formed by a feature displacement network and a noise-conditioned score estimator along imagined paths. We propose two gating mechanisms: a hard boundary that halts return expansion at a discrete state, and a soft boundary that continuously shifts the return recursion toward critic bootstrapping. Both variants preserve boundary value estimates and apply corresponding survival weights to actor and critic losses. Formulated independently of specific dynamics models and instantiated within Dreamer V3, we prove that the resulting gated Bellman operator is a strict contraction mapping and that early bootstrapping monotonically reduces conditional target variance. We clarify the theoretical assumptions distinguishing our expansion proxy from exact path likelihoods. Across continuous-control tasks and Atari Pong, empirical evaluations show that dynamic boundary selection enhances sample efficiency and late-training performance over fixed-horizon and auxiliary-enabled baselines.

**arXiv ID:** 2610.41022
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
