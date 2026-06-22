# Agent arXiv Daily

**Last Updated:** 2026-06-22 05:26:49

**Total Papers:** 109

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (5 papers)</h2></summary>

<details>
<summary><strong>Deontic Policies for Runtime Governance of Agentic AI Systems</strong> - Anupam Joshi, Tim Finin, Karuna Pande Joshi, Lalana Kagal - [[pdf]](https://arxiv.org/pdf/2606.19464)</summary>

**Abstract:** Autonomous agentic AI systems driven by Large Language Models (LLMs) introduce a new class of security, privacy, and compliance challenges: an agent that can invoke tools, manipulate data, install software, and coordinate with peer agents across organizational boundaries must be constrained not just by authentication and access control, but by the full structure of enterprise governance. This includes specifying what agents are permitted and prohibited from doing, what they areobliged to do after certain actions (e.g., notify the CISO), under what conditions a standing obligation may be waived, and which rules take precedence when policies conflict. This governance problem exceeds what current policy engines provide. Systems such as XACML, Rego, and Cedar address only the permit/prohibit subset of this governance structure. They do not provide obligation lifecycle management, meta-policy conflict resolution, dispensations that waive obligations in specific circumstances, and ontological reasoning over domain class hierarchies commonly found in applications such as healthcare, cybersecurity, or data privacy. We propose AgenticRei, which realizes key governance requirements such as obligations, dispensations, policy conflict resolutions, and reasoning over policies, as well as the basic permit/prohibit constraints. We use a deontic policy language built on the Rei framework, expressed as OWL (Web Ontology Language) and evaluated at runtime by a high-performance logic engine entirely outside the LLM. The same pipeline governs both tool invocations by the agent and agent-to-agent messages. We show through examples that deontic policies capture governance constraints around security and privacy that mostly cannot be expressed in current production engines. Our approach composes naturally with industry-standard frameworks like A2AS.

**arXiv ID:** 2606.19464
</details>

<details>
<summary><strong>Bistable by Construction: Wall-Clock-Calibrated State Monitors Have No Moment-Detection Regime at Agent Cadence</strong> - Manvendra Modgil - [[pdf]](https://arxiv.org/pdf/2606.19386)</summary>

**Abstract:** Runtime monitors for autonomous agents commonly threshold an accumulated internal state - a behavioural baseline, a drift statistic, or, in our prior work, a modelled affective state. We previously reported a State Saturation Trap: threshold-on-state triggers over a continuous affect engine become near-constant alarms on SWE-bench debugging agents (Modgil 2026). A post-release audit found the engine received dt=0 between actions, so its exponential decay never operated: the published trap is a pure-accumulator result. We correct the record (erratum, v2) and treat the flaw as an experiment. The key variable it exposes is whether a monitor's dynamics are calibrated in sample time (per observation, as in CUSUM) or wall-clock time (half-lives in seconds, as in affect models and EMA baselines). On fixed-rate streams these coincide; on agent streams, where inter-action time varies by orders of magnitude, they do not. A pre-registered sweep over uniform intervals (dt in {0..600}s) on 20 trajectories shows the wall-clock level trigger has two regimes: at dt<=1s a constant alarm (20/20; median 18 firings); at dt>=60s silent. Every critical dt lies in (1,30]s. Real agent runs measure latency at median 1.53s (p90 2.33s); real coding cadence sits inside the trap regime, vindicating the empirical finding under a corrected mechanism. The structure is a property of the calibration class, not the engine: a minimal wall-clock accumulator over the raw error stream reproduces the same cliff, while a sample-time CUSUM over the identical stream is exactly dt-invariant (20/20). A rising-edge trigger with hysteresis fires 0-3 times per trajectory in every condition. We conclude that wall-clock-calibrated leaky-integrator monitors admit no regime in which they act as moment detectors on agent streams; transition detection escapes the trap at every cadence, but does not recover human intervention timing.

**arXiv ID:** 2606.19386
</details>

<details>
<summary><strong>Dual-Agent Framework for Cross-Model Verified Translation of Natural-Language Protocols into Robotic Laboratory Platform</strong> - Hyeonna Choi, Jung Yup Kim, Hyuneui Lim, Seunggyu Jeon - [[pdf]](https://arxiv.org/pdf/2606.20120)</summary>

**Abstract:** Biological experiment protocols are written in natural language, whereas automation systems rely on predefined control commands, creating a semantic gap that limits autonomous execution. Microplate-based automatic experiments are particularly challenging due to the need to simultaneously control well mapping, sample-reagent combinations, replicate placement, and parallel dispensing. This study proposes an agent-based protocol translation framework that converts natural-language microplate-based protocols into executable control commands for a robotic laboratory platform. A Parser Agent formalizes the natural-language protocol into a structured representation, and a rule-based mapping engine deterministically incorporates the operational constraints of the robotic laboratory platform to generate device-level control commands. A heterogeneous LLM Validation Agent verifies completeness, parameter accuracy, and execution order, and triggers a self-correction loop with structured feedback when errors are detected. A sweep involving 7 Parsers and 3 Validators on randomly selected ELISA protocols evaluates how model scale and Validator type affect translation accuracy and pass rates under cross-model verification. The accuracy-latency trade-off is further verified by comparing the rule-based mapping of the proposed framework with LLM end-to-end direct mapping. Finally, Bradford assay-based protein quantification using a microplate was demonstrated on a robotic laboratory platform, validating end-to-end autonomous execution from natural-language protocols to real-world experiments. The proposed framework provides a flexible approach to narrowing the semantic gap between natural-language protocols and microplate-based self-driving laboratories.

**arXiv ID:** 2606.20120
</details>

<details>
<summary><strong>Sovereign Execution Brokers: Enforcing Certificate-Bound Authority in Agentic Control Planes</strong> - Jun He, Deying Yu - [[pdf]](https://arxiv.org/pdf/2606.20520)</summary>

**Abstract:** Autonomous agents are increasingly connected to cloud, deployment, and data-control workflows, but production mutation authority should not reside inside non-deterministic reasoning processes. Existing access-control mechanisms authorize identities, while assurance layers certify proposed actions; neither alone provides a mandatory enforcement point for certified authority at the moment of mutation. This paper introduces the Sovereign Execution Broker (SEB), a runtime enforcement boundary for certificate-bound agentic infrastructure. SEB consumes certificates issued by the Sovereign Assurance Boundary (SAB), verifies that the requested mutation matches the certified execution contract, checks validity windows, policy epochs, revocation epochs, and live-state drift, mints scoped execution identity, invokes infrastructure APIs, and records signed decision and outcome records. By separating proposal, admission, and execution, SEB turns certified authority into a short-lived, revocable, auditable runtime capability, provided that production mutation APIs reject non-broker identities. We present the SEB execution model, certificate and replay-verification predicates, scoped identity semantics, bypass-prevention deployment patterns, failure behavior, and a concrete prototype implementation. We evaluate the prototype on AWS and Kubernetes clusters, measuring latency overheads, revocation propagation, drift detection, and security under fault injection.

**arXiv ID:** 2606.20520
</details>

<details>
<summary><strong>A Layered Security Framework Against Prompt Injection in RAG-Based Chatbots</strong> - Gulshan Saleem, Nisar Ahmed, Muhammad Imran Zaman, Ali Hassan - [[pdf]](https://arxiv.org/pdf/2606.19660)</summary>

**Abstract:** Prompt injection is ranked as the most critical vulnerability in large language model (LLM) deployments by the OWASP Top 10 for LLM Applications, yet existing defenses operate at isolated pipeline stages and remain incomplete. Input filters cannot inspect retrieved documents, while output monitors cannot prevent malicious payloads from reaching the model. Consequently, retrieval-augmented generation (RAG) chatbots remain vulnerable to indirect injection, where a poisoned knowledge-base document compromises every user whose query retrieves it. We present a three-layer framework that intercepts both direct and indirect prompt injection throughout the inference pipeline. Layer 1 screens user input using a rule-based pattern library and a fine-tuned semantic anomaly classifier. Layer 2 enforces a provenance-based instruction hierarchy during context assembly, preventing retrieved content from overriding operator policy. Layer 3 audits model output using a policy rule engine and semantic drift detector before delivery. A continuous audit loop aggregates structured logs and supports retraining to adapt the classifier to emerging attack patterns. The framework is model-agnostic and deploys as middleware without modifying the underlying LLM. Evaluation on 5,080 samples across GPT-4o, Llama 3, and Mistral 7B shows that the framework reduces Attack Success Rate (ASR) from 71.4\% to 11.3\%, outperforming the best single-layer baseline by 27.3 percentage points and a published guardrail system by 23.8 percentage points, while maintaining a 4.8\% false positive rate and a median latency overhead of 61.2 ms. Ablation studies confirm that all three layers provide complementary protection and that their combined effect exceeds the sum of individual contributions.

**arXiv ID:** 2606.19660
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (27 papers)</h2></summary>

<details>
<summary><strong>Benchmarking Agentic Review Systems</strong> - Dang Nguyen, Wanqing Hao, Yanai Elazar, Chenhao Tan - [[pdf]](https://arxiv.org/pdf/2606.19749)</summary>

**Abstract:** A new class of agentic review systems are emerging as a remedy to the pressure placed on peer review systems by AI-assisted research, but it is unclear how they should be evaluated. We evaluate two open-source systems (OpenAIReview and coarse), one proprietary system (Reviewer3), and a zero-shot baseline, across six LLMs spanning frontier and efficient models. First, we study whether AI reviews on ICLR/NeurIPS papers track with papers' quality as approximated by external signals such as citations and acceptance decisions. Every system performs above chance in pairwise accuracy, and the best is OpenAIReview + GPT-5.5 at 83.0%. Second, to test whether systems can catch errors with known ground truth, we construct a perturbation benchmark that injects four categories of errors into papers across eight arXiv subject classes and measure detection recall. The strongest configuration (OpenAIReview + GPT-5.5) catches 71.6% of injected errors, leaving substantial room for improvement. The union of detections across six models reaches 83.3% recall, suggesting different models detect different errors and better harness design can potentially increase performance. Beyond these benchmarks, we study a public deployment of OpenAIReview with real users. Votes on its comments skew positive at 1.44 to 1, and the most common complaints are about false positives and minor nitpicks. Together, by evaluating full review systems backed by state-of-the-art models on real research papers, we show that while AI reviews still have room for improvement, they can already track human quality judgments well, catch important errors, and earn positive feedback from real users.

**arXiv ID:** 2606.19749
</details>

<details>
<summary><strong>TelcoAgent: A Scalable 5G Multi-KPM Forecasting With 3GPP-Grounded Explainability</strong> - Geon Kim, Dara Ron, Sukhdeep Singh, Suyog Moogi, Pranshav Gajjar, V V N K Someswara Rao Koduri, Een Kee Hong, Vijay K. Shah - [[pdf]](https://arxiv.org/pdf/2606.19821)</summary>

**Abstract:** Key Performance Measurement (KPM) forecasting is essential for proactive network management of 5G and next-generation telecom networks. However, existing machine learning (ML) approaches face significant limitations in scalability and explainability, restricting their effectiveness in real-world deployments. We propose TelcoAgent, a foundation model-based framework that enables accurate, scalable, and explainable forecasting of multiple KPMs across diverse network cells without the need for site-specific training. Specifically, the framework comprises three key components: (i) an automated three-agent pipeline that constructs a 3rd Generation Partnership Project (3GPP) knowledge graph directly from specification documents, (ii) a scalable, time-series foundation model (TSFM)-based prediction pipeline to deliver accurate, zero-shot forecasting, and finally (iii) a reasoning and explanation pipeline that provides actionable, domain-grounded diagnostics. Evaluated using a 3-month, real-world, city-scale 5G KPM dataset from a U.S.-based network operator, TelcoAgent demonstrates high forecasting accuracy for all 7 considered KPMs per cell across 200 cells, while delivering explainable insights and actionable instructions to address network degradations.

**arXiv ID:** 2606.19821
</details>

<details>
<summary><strong>Advancing DialNav through Automatic Embodied Dialog Augmentation</strong> - Leekyeung Han, Sangwon Jung, Hyunji Min, Jinseong Jeong, Minyoung Kim, Paul Hongsuck Seo - [[pdf]](https://arxiv.org/pdf/2606.19948)</summary>

**Abstract:** For embodied agents capable of physical interaction, the capability to create and understand dialog is crucial to ensure both safety and effectiveness. While DialNav~\cite{han2025dialnav} provides a framework for holistic evaluation of the dialog--execution loop in photorealistic indoor navigation, its performance remains limited by a critical scarcity of training data (2K episodes). To address this, we propose an automatic generation pipeline, and construct the \textbf{RAINbow} dataset, a large-scale training dataset with 238K episodes for DialNav. Our pipeline converts existing VLN datasets into multi-turn dialog and creates cost-efficient and high-quality dataset. Then, we introduce two additional complementary advances to unlock the data's full potential: (1) Dual-Strategy Training, a navigation training scheme to align the navigation training with the dynamic dialog-navigation loop, and (2) a localization model that leverages VLN knowledge. By combining these complementary solutions, our model substantially outperforms the baseline in success rate on both \textbf{Val Seen} (58.24, \textbf{+89\%}) and \textbf{Val Unseen} (29.05, \textbf{+100\%}) splits, establishing a new state of the art.

**arXiv ID:** 2606.19948
</details>

<details>
<summary><strong>RACL: Reasoning-Agent Control Layers for Continuous Metaheuristic Learning</strong> - Antón Asla Manzárraga - [[pdf]](https://arxiv.org/pdf/2606.20142)</summary>

**Abstract:** This paper introduces RACL, a Reasoning-Agent Control Layer for metaheuristics. RACL places a reasoning agent above an existing optimizer. The agent does not replace the optimizer and does not modify business constraints. Instead, it controls the optimizer's internal search behavior by observing operational memory, reasoning over past behavior, formulating bounded hypotheses, testing interventions, evaluating outcomes, applying guardrails, consolidating useful policies and explaining its decisions. The experiment uses vehicle routing as a testbed, but the contribution is not a new routing solver, a particular ALNS configuration or a specific set of routing rules. The contribution is the RACL method: a way for a reasoning agent to discover, validate, consolidate and explain algorithmic control rules for a metaheuristic. In the current experimental setting, RACL improves or ties the Operational Memory Policy in 21 of 21 feasible cases and improves or ties a non-reasoning Stagnation-Triggered Policy in 18 of 21 feasible cases, with an average RACL vs STP cost delta of -0.641%. In the Sevilla-9/10 runtime sample, RACL improves average cost by -8.337% versus Fixed and -1.605% versus STP without showing material computational overhead. During the proof-of-concept, Codex was used as an in-the-loop reasoning agent observing executions, interpreting logs and proposing live bounded interventions. The policy proxy was later used only to make quantitative evaluation reproducible.

**arXiv ID:** 2606.20142
</details>

<details>
<summary><strong>Automating SKILL.md Generation for Computer-Using Agents via Interaction Trajectory Mining</strong> - Yuexing Hao, Xiaomin Li - [[pdf]](https://arxiv.org/pdf/2606.20363)</summary>

**Abstract:** Explicit skill libraries make computer-using agents easier to inspect, but it remains unclear whether such libraries can be mined from interaction data in a way that improves downstream policies. We study this question through a three-stage pipeline that segments GUI trajectories, clusters segments into candidate skills, and trains a skill-aware policy from the resulting annotations. The mined clusters are readable on the source benchmark: five of eight clusters have at least 0.95 purity against InteraSkill Workflows labels. However, readability does not imply transfer. GRPO improves IW skill-step accuracy only from 18.5\% to 20.5\%, leaves BrowseComp+ essentially unchanged, and underperforms trivial frequency priors on key source-domain metrics. We therefore present the method as a diagnostic study: trajectory mining can expose inspectable skill structure, but the current boundary detector, orderless segment representation, and offline reward model are insufficient for reliable cross-domain policy improvement.

**arXiv ID:** 2606.20363
</details>

<details>
<summary><strong>Execution-bound advisory automation for agentic AI: a reproducible AIBOM-driven CSAF-VEX framework</strong> - Petar Radanliev, Omar Santos, Carsten Maple, Kay Atefi - [[pdf]](https://arxiv.org/pdf/2606.19390)</summary>

**Abstract:** A protocol driven framework is presented that binds SBOM and AIBOM artefacts to deterministic environment capture and structured runtime telemetry. Exploitability is computed from declared artefacts, observed activation conditions, and enforced execution policies. CSAF VEX advisories are generated from combined static and runtime evidence, cryptographically signed, and validated through deterministic replay. Evaluation uses approximately 10000 component entries across synthetic Agentic AI workloads 50 to 5000 components, incorporating OSV, GitHub Advisory, KEV, and EPSS datasets.

**arXiv ID:** 2606.19390
</details>

<details>
<summary><strong>Playful Agentic Robot Learning</strong> - Junyi Zhang, Jiaxin Ge, Hanjun Yoo, Letian Fu, Zihan Yang, Yaowei Liu, Raj Saravanan, Shaofeng Yin, Justin Yu, Dantong Niu, Zirui Wang, Roei Herzig, Ken Goldberg, Yutong Bai, David M. Chan, Ion Stoica, Angjoo Kanazawa, Jiahui Lei, Haiwen Feng, Trevor Darrell - [[pdf]](https://arxiv.org/pdf/2606.19419)</summary>

**Abstract:** Current agentic robot systems can write executable Code-as-Policy programs, observe feedback, and revise behavior across multiple attempts, but they remain largely task-driven: reusable skills are acquired only after explicit instructions. We study Playful Agentic Robot Learning, where an embodied coding agent uses self-directed play as a continual skill-learning stage before downstream tasks arrive. We introduce RATs, Robotics Agent Teams designed for play-time skill acquisition. During play, RATs proposes novel yet learnable exploratory tasks, plans and executes robot-code policies, verifies intermediate progress, diagnoses failures, retries with dense, step-level feedback, and distills successful executions into a persistent code skill library. At test time, the agent reuses relevant skills from this frozen library to help solve new tasks. Experiments in LIBERO-PRO and MolmoSpaces show that play-learned skills improve held-out downstream tasks over no-play and random-play baselines, with 20.6 and 17.0 percentage-point gains over CaP-Agent0 on LIBERO-PRO and MolmoSpaces, respectively. Moreover, the learned skills can be plugged into other inference-time Code-as-Policy agents by simply retrieving them into the context, improving RoboSuite and real-world transfer by 8.9 and 8.8 points, respectively, without finetuning the underlying model.

**arXiv ID:** 2606.19419
</details>

<details>
<summary><strong>IHBench: Evaluating Post-Interruption Recovery in Voice Agents with Structured Workflows</strong> - Ahmad Salimi, Wentao Ma, Yuzhi Tang, Dongming Shen, Mu Li, Alex Smola - [[pdf]](https://arxiv.org/pdf/2606.19595)</summary>

**Abstract:** Voice agents deployed in structured workflows (customer service, healthcare scheduling, account management) must handle frequent user interruptions while maintaining progress through multi-step procedures. Existing benchmarks for speech-capable models focus on the timing of interruptions: barge-in detection, endpointing, and turn-taking dynamics. They leave unmeasured what happens after the interruption: does the agent resume the workflow at the correct step? Does it address the user's interjection? Does it avoid re-delivering content the user already heard?
We introduce IHBench (Interruption Handling Benchmark), a benchmark that evaluates post-interruption recovery in voice agents executing state-machine-driven workflows across 10 enterprise domains. Six interruption types are injected at controlled points mid-utterance, with per-interruption evaluation rubrics generated alongside the data. Each interruption is scored on two axes: task fulfillment and recovery quality.
We evaluate 27 audio-language model configurations from OpenAI, Google, and the open-weight community. Models vary widely, and recovery quality depends strongly on the interruption type. Across our experiments, closed-weight models are consistently more robust to interruptions than open-weight ones: they win far more often on task fulfillment, degrade roughly 3.3x more slowly as conversations grow longer, and show no audio-versus-text modality gap, whereas the open-weight models lose ground on all three. A human study validates the LLM judge against human annotators, and a cross-benchmark analysis against AudioMultiChallenge indicates that recovery quality is a largely distinct capability axis.

**arXiv ID:** 2606.19595
</details>

<details>
<summary><strong>FAPO: Fully Autonomous Prompt Optimization of Multi-Step LLM Pipelines</strong> - Paul Kassianik, Baturay Saglam, Huaibo Zhao, Blaine Nelson, Supriti Vijay, Aman Priyanshu, Amin Karbasi - [[pdf]](https://arxiv.org/pdf/2606.19605)</summary>

**Abstract:** Multi-step LLM pipelines fail through interactions among retrieval, reasoning, and formatting steps, so prompt-only optimization can miss bottlenecks in the chain. We present FAPO (Fully Autonomous Prompt Optimization), a framework that lets Claude Code optimize an LLM pipeline inside a standardized codebase. FAPO evaluates a pipeline, inspects intermediate steps, diagnoses failures, proposes scoped changes, and validates variants repeatedly to optimize against a score function. It first tries prompt edits and, only when prompt optimization appears insufficient, changes chain structure within the permitted scope when attribution identifies a structural bottleneck. Across six benchmarks and three task models, FAPO beats the baseline GEPA in 15 of 18 model-benchmark comparisons. In 11 model-benchmark comparisons, FAPO wins with non-overlapping mean $\pm$ trial-standard-deviation ranges, and the mean FAPO-GEPA gain is +14.1 pp. In the six HoVer and IFBench comparisons where prompt-first search escalated to structural changes, FAPO wins all six with a mean gain of +33.8 pp. FAPO also improves performance on security tasks: on CTIBench-RCM, a security CVE-to-CWE task, prompt-only FAPO lifts test accuracy by +4.0 pp on GPT-5, +7.1 pp on Foundation-Sec-8B-Instruct, and +2.0 pp on Foundation-Sec-8B-Reasoning. These results position FAPO as a state-of-the-art pipeline optimization technique for both general-purpose and security-focused tasks.

**arXiv ID:** 2606.19605
</details>

<details>
<summary><strong>StaminaBench: Stress-Testing Coding Agents over 100 Interaction Turns</strong> - Vlad Sobal, Shuo Yang, Yuting Zhang, Wei Xia, Stefano Soatto - [[pdf]](https://arxiv.org/pdf/2606.19613)</summary>

**Abstract:** We introduce StaminaBench, a benchmark that measures the stamina of coding agents: how many consecutive interaction turns (change requests) they can handle before failing. Unlike the prevailing fraction-of-tasks-solved metric, this matches real vibe-coding where sessions run dozens or hundreds of turns. In StaminaBench, agents implement a REST API server and modify it across a tunable number of procedurally generated follow-up change requests - 100 in our experiments, resulting in codebases of up to 6,000 lines. Tests are generated fully programmatically without LLM involvement, ensuring reproducibility and reliability; change sequences are drawn from either a hardcoded or LLM-driven sampler, both constrained to a structured action space to ensure changes are valid. The agent and the server run in an isolated environment and communicate with the benchmark through HTTP, making testing fully black-box and language-agnostic. We evaluate six agent harnesses paired with seven open-source LLMs across 20 scenarios of 100 turns each and find that: (1) all the tested models fail within 5-6 turns, confirming that vibe-coding-style programming without thorough testing produces bugs; (2) passing test feedback back to the agent and allowing it to retry improves passed turn count by up to 12x; and (3) a good harness is required for strong performance: stronger models exhibit up to a 6x gap between their best and worst harness, while weaker models fail with any harness. We release the benchmark and the generated tasks to enable further research into multi-turn coding agent behavior. Benchmark code and data: this http URL.

**arXiv ID:** 2606.19613
</details>

<details>
<summary><strong>ScholarQuest: A Taxonomy-Guided Benchmark for Agentic Academic Paper Search in Open Literature Environments</strong> - Tingyue Pan, Mingyue Cheng, Daoyu Wang, Yitong Zhou, Jie Ouyang, Qi Liu, Enhong Chen - [[pdf]](https://arxiv.org/pdf/2606.20235)</summary>

**Abstract:** Academic paper search is a core step in scientific research, and LLM-based search agents are emerging as a promising paradigm for iterative, intent-driven literature exploration. However, existing benchmarks are insufficient for systematically evaluating agentic academic search under realistic open literature environments. We propose ScholarQuest, a large-scale, taxonomy-guided benchmark for agentic academic paper search. ScholarQuest is constructed from over 1,000 computer science topics and four representative research intents, including method-oriented, setting-anchored, comparison-based, and scope-controlled queries. It further provides scalable answer construction and a shared retrieval backend ScholarBase for reproducible evaluation. Benchmarking results show that agentic methods outperform single-shot retrieval baselines, yet the best-performing agent only achieves 0.314 Recall@100 and 0.355 Recall@All, indicating substantial room for improvement. In addition, analyses of search efficiency, intent-level robustness, and failure cases further highlight the benchmark's ability to provide multi-dimensional evaluation signals for academic paper search agents.

**arXiv ID:** 2606.20235
</details>

<details>
<summary><strong>Analyzing Defensive Misdirection Against Model-Guided Automated Attacks on Agentic AI Systems</strong> - Reza Soosahabi, Vivek Namsani - [[pdf]](https://arxiv.org/pdf/2606.20470)</summary>

**Abstract:** Agentic AI systems increasingly rely on language-model components to interpret instructions, process external data, invoke tools, and coordinate with other agents. These capabilities make prompt-injection and jailbreak attacks more consequential, especially as attackers adopt model-guided automation to scale probing, prompt refinement, and response evaluation. This work analyzes the resulting attack-defense setting through a probabilistic model of a target system, its defense mechanism, and the attacker's automated judge. Our analysis shows that conventional detect-and-block defenses can allow attacker success rate (ASR) to approach one as the query budget grows, since predictable refusals provide useful feedback to automated search. We then examine detect-and-misdirect, where detected malicious interactions receive controlled, non-operational responses designed to induce false-positive errors in the attacker's judge. This strategy reduces the positive predictive value of attacker-selected candidates and yields a bounded asymptotic ASR. We evaluate a proof-of-concept realization of this strategy through Contextual Misdirection via Progressive Engagement (CMPE), a lightweight conversational misdirection method designed to replace predictable refusal text with safe but strategically misleading responses in automated jailbreak settings. On jailbreak benchmarks, CMPE reduces estimated ASR upper bounds by up to two orders of magnitude and nearly eliminates verified attack success in end-to-end PAIR and GPTFuzz attack runs.

**arXiv ID:** 2606.20470
</details>

<details>
<summary><strong>Efficient and Sound Probabilistic Verification for AI Agents</strong> - Alaia Solko-Breslin, Pramod Kaushik Mudrakarta, Mihai Christodorescu, Somesh Jha, Krishnamurthy Dj Dvijotham - [[pdf]](https://arxiv.org/pdf/2606.20510)</summary>

**Abstract:** Securing AI agents that operate in complex digital environments has become a critical need, and runtime monitoring approaches that formulate and enforce policies expressed in a formal language like Datalog offer a promising solution. However, existing approaches are restricted to deterministic policies. In many practical applications of AI agents, there is a need to enforce security policies in the face of ambiguity, leading to probabilistic predicates or state transitions (for example, a declassifier or Personally Identifiable Information (PII) detector that has some failure probability on each invocation). Furthermore, in many such applications, one cannot easily make the independence assumptions necessary to invoke prior work on probabilistic inference in Datalog. We address this by introducing a sound and efficient framework for such verification based on distributionally robust optimization, computing sound upper bounds on the probability of policy violation regardless of possible correlations between predicates. On standard benchmarks for terminal and tool calling agents, we demonstrate that our approach outperforms prior art and improves the security-utility trade-off while ensuring rigorous bounds on the probability of policy violation.

**arXiv ID:** 2606.20510
</details>

<details>
<summary><strong>TxBench-PP: Analyzing AI Agent Performance on Small-Molecule Preclinical Pharmacology</strong> - Hannah Le, Ramesh Ramasamy, Alex Urrutia, Mahsa Yazdani, Tim Proctor, Kenny Workman - [[pdf]](https://arxiv.org/pdf/2606.19245)</summary>

**Abstract:** Artificial intelligence (AI) agents promise to accelerate drug discovery by compressing interpretation and decision-making loops, but practical deployment requires trusted evaluation on realistic program decisions. We introduce TherapeuticsBench Preclinical Pharmacology (TxBench-PP), a verifiable benchmark for small-molecule preclinical pharmacology and the first focused slice of a broader TherapeuticsBench effort across drug-discovery stages and therapeutic modalities. TxBench-PP tests whether agents can recover accurate conclusions from real-world assay data rather than memorized facts from literature. The benchmark contains 100 evaluations indexed by program stage, assay type, and task structure, spanning mechanism-of-action (MoA) and pharmacodynamic (PD) reasoning, compound-target engagement, causal target validation, developability and safety, and translational efficacy. Agents receive realistic workflow snapshots, inspect files in a coding environment, and return structured answers graded deterministically. Across 16 model-harness configurations, comprising 11 models and 4,800 trajectories, no system reliably recovered preclinical pharmacology decisions. The strongest configuration, Claude Opus 4.8 / Pi, passed 59.3\% of endpoint attempts (178/300; 95\% CI, 51.1-67.6), followed by GPT-5.5 / Pi at 55.3\% (166/300; 47.0-63.6).

**arXiv ID:** 2606.19245
</details>

<details>
<summary><strong>Prompt, Plan, Extract: Zero-Shot Agentic LLMs Workflows for Lung Pathology Extraction from Clinical Narratives</strong> - Aman Pathak, Cheng Peng, Mengxian Lyu, Ziyi Chen, Reema Solan, Sankalp Talankar, Yasir Khan, Hiren Mehta, Aokun Chen, Yi Guo, Yonghui Wu - [[pdf]](https://arxiv.org/pdf/2606.19852)</summary>

**Abstract:** Information extraction from pathology reports is essential for cancer staging, tumor registry population. Yet key data remains embedded in narrative reports, making manual extraction labor-intensive and error-prone. Traditional supervised Natural Language Processing pipelines address this through fully supervised Named Entity Recognition and Relation Extraction, but require expensive manual annotation and suffer cascading failures when upstream entities are missed. In this study, we developed a zero-shot, agentic workflow, and evaluated five open-source generative Large Language Models (LLMs) to populate 13 College of American Pathologists synoptic fields from lung resection pathology reports. We compared them against a state-of-the-art supervised GatorTron NER-RE baseline using a novel, registry-aligned evaluation framework. The baseline achieved Micro-F1of 0.960, while the best zero-shot model (GPT-OSS-20B) achieved Micro-F1 of 0.893 (recall: 0.949), accurately extracting complex relations like Pathologic Stage without task-specific training. These results suggest that open-source, zero-shot agentic LLMs are a low-cost solution for extracting lung pathology information.

**arXiv ID:** 2606.19852
</details>

<details>
<summary><strong>Multimodal Evaluator Preference Collapse: Cross-Modal Contagion in Self-Evolving Agents</strong> - Zewen Liu - [[pdf]](https://arxiv.org/pdf/2606.16682)</summary>

**Abstract:** When AI agents use language models to evaluate their own outputs in a
feedback loop, systematic biases emerge. We show that Evaluator Preference
Collapse (EPC) is dramatically amplified in multimodal settings. Using
GPT-4o to evaluate DeepSeek-chat across text and visual tasks, we find
that a single strategy (step_by_step) absorbs 48.4% of all weight -- 3.2x
the collapse observed in text-only self-evaluation -- while three
visual-domain strategies receive only 9.1% combined weight. We then
demonstrate a novel phenomenon we term cross-modal contagion: evaluator
preferences acquired on one modality transfer to and corrupt strategy
selection on another. Through a four-phase isolation training paradigm, we
measure contagion coefficients and document strategy inversion -- the
optimal strategy for a modality reverses after cross-modal exposure. A
Phase 3 statistical validation across five evaluator configurations (N=80
total independent repetitions, ~35,000 API calls) with both text-proxy and
real-image visual tasks finds: cross-model evaluation produces strong
contagion (JSD~0.19-0.34), real-image inputs yield the most directionally
consistent signal (mean gamma_{T->V}=1.145, gamma_{V->T}=0.937, 70% T->V,
Cohen's d=0.56), and self-evaluation provides near-complete immunity --
97% of runs (N=30) yield zero contagion (JSD=0.003, d=0.07). Three
methodological ablations and multi-executor validation confirm the effect
is not a structural artifact. We introduce the contagion matrix indexed by
evaluator identity, release the MM-EPC framework, and identify
cross-model evaluator architecture as the primary risk factor for
preference drift. Code and data: this https URL.

**arXiv ID:** 2606.16682
</details>

<details>
<summary><strong>MortarBench: Evaluating Mortgage Loan Origination Agents</strong> - Matthew Toles, Yunan Lu, Manav Munjal, Bojun Liu, Yuanhao Deng, Stephanie Selig, Derek Rindner, Cheng Li, Zhou Yu - [[pdf]](https://arxiv.org/pdf/2606.19416)</summary>

**Abstract:** Loan origination is the process by which a lender creates a new loan, from application and underwriting through approval and funding. This process serves a critical role in evaluating the eligibility and level of risk posed by an applicant. Recently, firms have begun using mortgage loan agents to augment human loan officers, despite a lack of any public benchmark. To fill this gap, we present MortarBench, a loan origination agent benchmark. MortarBench uses a financial data synthesis and mutation pipeline to generate examples with broad edge case coverage that match real-world distributions and questions. We find that state-of-the-art large language models (LLMs) perform poorly, with closed-source models achieving at most 77.1\% exact match accuracy. We also discover systematic biases in LLM perception of foreignness related to non-English names. Noting these weaknesses, we introduce CRIT, a confidence calibration framework. Our method increases accuracy to 80.5\% while improving risk management steering and reducing bias.

**arXiv ID:** 2606.19416
</details>

<details>
<summary><strong>Marginal Advantage Accumulation for Memory-Driven Agent Self-Evolution</strong> - Mingyu Yang, Keye Zheng, Congchao Cheng, Yujie Liu, Xingkang Lu, Fan Jiang, Yefei Zheng - [[pdf]](https://arxiv.org/pdf/2606.20475)</summary>

**Abstract:** In batch-style trace distillation, the same memory operation may receive contradictory feedback across different batches. Existing methods lack a cross-batch, operation-level evidence accumulation mechanism, making it impossible to distinguish stably effective operations from accidental hits. This paper formalizes the requirement as two structural conditions, alignability and comparability, and proposes Marginal Advantage Accumulation (MAA). MAA constructs differential signals to make them comparable across batches, accumulates signed evidence per operation via EMA, and ensures cross-batch traceability through semantic identity merging. As a post-processing architecture, MAA achieves the best results in 14 out of 16 settings across 4 benchmarks and 4 target models, consistently outperforming existing batch-level distillation baselines and matching or surpassing online alternatives in most settings, while reducing optimization-phase token consumption by approximately 75%.

**arXiv ID:** 2606.20475
</details>

<details>
<summary><strong>AgentArmor: A Framework, Evaluation, \& Mitigation of Coding Agent Failures</strong> - Kenneth Ge, Andre Assis - [[pdf]](https://arxiv.org/pdf/2606.19380)</summary>

**Abstract:** Software engineering and deployment are increasingly being delegated to AI coding agents. The scale of their adoption is surfacing rare, but highly destructive, failure modes. In this paper, we study these failure modes as stemming from three distinct mechanisms: underspecification, where default model behavior is unsafe; capability errors, where the safe action is available but the model does not adhere to it due to bias or capability limitations; and agent harness errors, where the model fails to execute the safe action through the harness. We evaluate these across 8 different evaluations, each inspired by real-life deployment failures, totaling 20 coding environments and 59 synthetic transcript templates. Based on this evaluation, we propose AgentArmor, an agent harness modification, to mitigate these errors. By adding an extended system prompt, a separate command classifier, a ``3 strikes'' policy, deterministic guardrails, and tools for the agent to edit its own context, we show that AgentArmor is safer across a statistically significant number of samples. Thus, we suggest concrete mitigations for current coding agents and a design philosophy for future agent harness features.

**arXiv ID:** 2606.19380
</details>

<details>
<summary><strong>Probe-and-Refine Tuning of Repository Guidance for Coding Agents</strong> - Asa Shepard, Jeannie Albrecht - [[pdf]](https://arxiv.org/pdf/2606.20512)</summary>

**Abstract:** LLM-based coding agents need higher-level operational knowledge about a repository (which files house which subsystems, how to run the test suite, which workflows have historically led to wrong fixes) that does not exist in the code itself. Engineers typically maintain \texttt{this http URL} files to supply this context as instructions for coding agents, but whether they help is contested: recent studies disagree on whether LLM-generated guidance improves or harms agent performance. In this paper we show that how the guidance is produced is the decisive variable, and introduce \emph{probe-and-refine tuning}: a procedure that uses synthetic bug-fix probes to iteratively diagnose and patch a repository's guidance file through single-shot LLM calls, with no agent loop or tool use during tuning. On SWE-bench Verified across four independent trials with Qwen3.5-35B-A3B at 200 steps, probe-and-refine achieves 33.0\,\% mean resolve rate vs.\ 28.3\,\% for the static knowledge base used to initialize it and 25.5\,\% for an unguided baseline ($p < 0.001$ for both probe-and-refine contrasts). The improvement comes from coverage rather than precision: refined guidance produces evaluable patches for 14.5 percentage points (pp) more instances while per-patch precision remains statistically constant ($\sim$59\,\%, $p = 0.119$), showing that improved guidance helps agents reach the correct file rather than improving the quality of the changes they make. Further, a step-budget experiment shows that guidance is what lets the agent use a larger step budget productively, and a cross-model experiment with NVIDIA-Nemotron-3-Nano-30B-A3B finds that the tuning loop degrades when the model cannot generate sufficiently diagnostic output, though per-patch precision remains constant even then.

**arXiv ID:** 2606.20512
</details>

<details>
<summary><strong>World Engine: Towards the Era of Post-Training for Autonomous Driving</strong> - Tianyu Li, Li Chen, Caojun Wang, Haochen Liu, Kashyap Chitta, Zhenjie Yang, Yuhang Lu, Naisheng Ye, Yihang Qiu, Yufei Wang, Luoxi Zou, Jiaxin Peng, Jin Pan, Zhaoyu Su, Andrei Bursuc, Shengbo Eben Li, Andreas Geiger, Peng Su, Hongyang Li - [[pdf]](https://arxiv.org/pdf/2606.19836)</summary>

**Abstract:** Autonomous vehicles must operate safely in the real world, where errors can have severe consequences. Although modern end-to-end driving policies excel in routine scenarios, their reliability is limited by the scarcity of safety-critical ``long-tail'' events in real driving datasets. These rare interactions define the practical safety boundary of the learned policy, yet they are difficult to collect at scale in the real world. Here we show that this fundamental limitation can be addressed by post-training pre-trained driving models on synthesized high-stakes interactions. We introduce World Engine, a generative framework that reconstructs high-fidelity interactive environments from real-world logs and systematically extrapolates them into realistic safety-critical variations. This paradigm enables reinforcement-based post-training to align policies with safety constraints, circumventing the physical risks inherent in real-world exploration. On a public benchmark built on nuPlan, World Engine substantially reduces failures in rare safety-critical scenarios and yields significantly larger gains than scaling pre-training data alone. Furthermore, when deployed on a production-scale autonomous driving system, the resulting policy reduces simulated collisions and demonstrates measurable improvements in on-road testing, showing that post-training on synthesized, safety-critical interactions offers a scalable and effective pathway to safer autonomous driving. The full codebase suite, including training, is released to the public.

**arXiv ID:** 2606.19836
</details>

<details>
<summary><strong>Agentic AutoResearch forSpace Autonomy: An Auditable, LLM-Driven Research Agent for Aerospace Control Problems</strong> - Amit Jain, Richard Linares - [[pdf]](https://arxiv.org/pdf/2606.20394)</summary>

**Abstract:** Spacecraft guidance, navigation, and control functions are increasingly realized as learned policies distilled from expert solvers. Developing such a policy is itself a research process: an investigator selects an architecture and hyperparameters, runs experiments, and must determine whether an apparent improvement is genuine or merely seed noise. This paper presents AutoResearch, a framework in which a large language model autonomously drives that loop for aerospace control problems, coupled with a credibility layer, built into the loop, that certifies each reported result against the problem's own measured seed noise. The language model serves only as the offline research agent that develops the control policy; the trained policy it produces is then deployed onboard the spacecraft, while the model itself never operates the vehicle. At each iteration the agent reads a plain-language problem description and the run history, proposes a single edit to the training script, executes it, and logs the outcome. No reported result is credited until it passes the same three checks: measured per-problem seed noise, reseeded verification of the best configuration, and leave-one-out pruning of the agent's edits. The same loop is applied, unchanged, to two aerospace control problems: a Clohessy-Wiltshire relative rendezvous and a safety-constrained collision-avoidance docking past a keep-out zone, each calibrated against a known optimal control benchmark. In both, the audited policy clears the measured seed noise by many standard deviations; an undirected search over the same parameters does not. On the docking problem the gap becomes categorical: undirected search yields no feasible policy, while the learned policy stays outside the keep-out zone on every seed.

**arXiv ID:** 2606.20394
</details>

<details>
<summary><strong>GroundControl: Anticipating Navigation Failures in Vision-Language Agents via Trajectory-Consistent Uncertainty Estimates</strong> - Nastaran Darabi, Divake Kumar, Sina Tayebati, Devashri Naik, Amit Ranjan Trivedi - [[pdf]](https://arxiv.org/pdf/2606.20479)</summary>

**Abstract:** Vision-language navigation agents achieve competitive average success on benchmark tasks, yet failures often arise through predictable trajectory-level breakdowns such as oscillation, stagnation, or inefficient detours. Reliable deployment, therefore, requires uncertainty signals that anticipate emerging failure dynamics during execution rather than reflect only instantaneous action entropy. We introduce \emph{GroundControl}, a trajectory-consistent uncertainty estimator defined as statistical deviation from nominal goal-directed distance-to-goal dynamics aggregated over an episode. GroundControl models distance evolution using a constant-velocity Kalman filter and combines normalized innovation statistics with complementary trajectory features capturing progress, monotonicity, path efficiency, and oscillatory behavior. The resulting uncertainty score reflects geometric and temporal inconsistency in navigation behavior rather than local prediction dispersion. To evaluate uncertainty quality independently of task success, we formalize \emph{Selective Risk--Coverage Navigation (SRCN)}, a protocol that measures how effectively an uncertainty score ranks episodes by failure or inefficiency using risk--coverage curves and AURC / E-AURC summaries. Across five EB-Navigation splits ($N=300$ episodes), trajectory-consistent uncertainty achieves near-oracle ordering under success-based selective risk, with weighted-average $\mathrm{E\text{-}AURC}_{\mathrm{SR}}=0.0024$ for the GPT-4o model, substantially outperforming entropy-, conformal-, and heuristic baselines. Under SPL-based selective evaluation, GroundControl consistently achieves the lowest AURC and E-AURC across models and navigation splits. These results show that modeling deviation from goal-directed dynamics provides an interpretable and robust signal for anticipating navigation failures in vision-language agents.

**arXiv ID:** 2606.20479
</details>

<details>
<summary><strong>PiDR: Physics-Informed Inertial Dead Reckoning for Autonomous Platforms</strong> - Arup Kumar Sahoo, Itzik Klein - [[pdf]](https://arxiv.org/pdf/2601.03040)</summary>

**Abstract:** A fundamental requirement for full autonomy is the ability to sustain accurate navigation in the absence of external data, such as GNSS signals or visual information. In these challenging environments, the platform must rely exclusively on inertial sensors, leading to pure inertial navigation. However, the inherent noise and other error terms of the inertial sensors in such real-world scenarios will cause the navigation solution to drift over time. Although conventional deep-learning models have emerged as a possible approach to inertial navigation, they are inherently black-box in nature. Furthermore, they struggle to learn effectively with limited supervised sensor data and often fail to preserve physical principles. To address these limitations, we propose PiDR, a physics-informed inertial dead-reckoning framework for autonomous platforms in situations of pure inertial navigation. PiDR offers transparency by explicitly integrating inertial navigation principles into the network training process through the physics-informed residual component. PiDR plays a crucial role in mitigating abrupt trajectory deviations even under limited or sparse supervision. We evaluated PiDR on real-world datasets collected by a mobile robot and an autonomous underwater vehicle. We obtained more than 29% positioning improvement in both datasets, demonstrating the ability of PiDR to generalize different platforms operating in various environments and dynamics. Thus, PiDR offers a robust, lightweight, yet effective architecture and can be deployed on resource-constrained platforms, enabling real-time pure inertial navigation in adverse scenarios.

**arXiv ID:** 2601.03040
</details>

<details>
<summary><strong>Self-Supervised Relevance Modelling in Autonomous Driving via Counterfactual Analysis</strong> - Luca Lusvarghi, Javier Gozalvez, Pablo Urbano Hidalgo - [[pdf]](https://arxiv.org/pdf/2606.10688)</summary>

**Abstract:** Autonomous driving relies on computationally intensive perception pipelines to continuously detect and track objects in the surrounding environment. While some objects are key to plan safe and effective maneuvers, others may not be relevant and have no impact on the autonomous vehicle's driving decisions. Focusing on relevant objects allows a more efficient usage of available computational resources, reduces processing latencies, and limits the downstream propagation of perception noise. In this work, we propose a novel self-supervised approach based on counterfactual analysis to develop a relevance model - an AI-based tool that quantifies the relevance of objects for an autonomous vehicle. To demonstrate the potential of the proposed approach, we train a relevance model on a synthetic causal dataset generated in a selected urban scenario. Results show that the relevance model is able to accurately estimate the objects' relevance with millisecond-level latency, enabling real-time relevance estimation also in high-density scenarios. We also show that the relevance model can be used to build relevance heatmaps that offer valuable insights into the autonomous vehicle's driving policy and can be used to proactively inform perception and planning tasks. We openly release both the relevance model and the causal dataset.

**arXiv ID:** 2606.10688
</details>

<details>
<summary><strong>Qwen-RobotNav Technical Report: A Scalable Navigation Model Designed for an Agentic Navigation System</strong> - Jiazhao Zhang, Gengze Zhou, Hale Yin, Yiyang Huang, Zixing Lei, Qihang Peng, Haoqi Yuan, Jie Zhang, Xudong Guo, Xiaoyue Chen, An Yang, Fei Huang, Zhibo Yang, Junyang Lin, Dayiheng Liu, Jingren Zhou, Zhuoyuan Yu, Jingyang Fan, Zhixuan Liang, Pei Lin, Ye Wang, Anzhe Chen, Kun Yan, Xiao Xu, Jiahao Li, Lulu Hu, Minying Zhang, Shurui Li, Wenhu Xiao, Shuai Bai, Xuancheng Ren, Chenxu Lv, Chenfei Wu, Xiong-Hui Chen - [[pdf]](https://arxiv.org/pdf/2606.18112)</summary>

**Abstract:** Agentic navigation systems require a base navigation model whose observation strategy can be externally reconfigured at inference time, because instruction following, object search, target tracking, and autonomous driving share the same perception-planning backbone yet demand fundamentally different strategies for consuming the visual stream. We present Qwen-RobotNav, a scalable navigation model built on Qwen-RobotNav that addresses it through a parameterised interface with two complementary dimensions: multiple task modes that select the navigation behaviour, and controllable observation parameters (e.g., token budget, per-camera weights) that govern how visual history is encoded. With training-time randomization over all parameters, Qwen-RobotNav is robust to any inference-time configuration requiring zero architectural modification to the Qwen-RobotNav backbone. We train Qwen-RobotNav on 15.6M samples; co-training with vision-language data prevents the collapse into reactive action-sequence mappers observed in trajectory-only training. The parameterised interface also makes Qwen-RobotNav a natural building block for agentic systems: for long-horizon scenarios, an upper-level planner decomposes goals into sub-tasks and dynamically switches Qwen-RobotNav's task mode and context strategy mid-episode, composing complex behaviours from repeated calls to the same model. Extensive experiments show that Qwen-RobotNav sets new state-of-the-art results across major navigation benchmarks. The model exhibits favourable scaling from 2B to 8B parameters, with joint multi-task training developing a shared spatial-planning substrate that transfers across task families, and demonstrates strong zero-shot generalisation to real-world robots across diverse environments.

**arXiv ID:** 2606.18112
</details>

<details>
<summary><strong>MemGUI-Agent: An End-to-End Long-Horizon Mobile GUI Agent with Proactive Context Management</strong> - Guangyi Liu, Gao Wu, Congxiao Liu, Pengxiang Zhao, Liang Liu, Mading Li, Qi Zhang, Mengyan Wang, Liang Guo, Yong Liu - [[pdf]](https://arxiv.org/pdf/2606.19926)</summary>

**Abstract:** MLLM-based mobile GUI agents have made substantial progress on short-horizon tasks, yet remain unreliable on long-horizon tasks that require retaining intermediate facts across many steps and app transitions. We attribute this limitation to ReAct-style prompting, which passively accumulates per-step records, leading to prompt explosion and dilution of critical cross-app facts. To address this, we introduce MemGUI-Agent, an end-to-end long-horizon mobile GUI agent with proactive context management. MemGUI-Agent is built on Context-as-Action (ConAct), which casts context management as first-class actions emitted by the same policy that selects UI actions. Instead of passively appending history, ConAct maintains three structured context fields: folded action history, folded UI state, and recent step record, preserving critical UI facts while keeping context compact. To make proactive context management learnable across model scales, we construct MemGUI-3K, a 2,956-trajectory dataset with full ConAct annotations for supervised training and offline analysis. Training an 8B model on MemGUI-3K produces MemGUI-8B-SFT, an 8B MemGUI-Agent that achieves the best open-data 8B performance on MemGUI-Bench and generalizes to the out-of-distribution MobileWorld benchmark. Code, data, and trained models will be released at this https URL.

**arXiv ID:** 2606.19926
</details>

</details>

<details open>
<summary><h2>LLM Agents (10 papers)</h2></summary>

<details>
<summary><strong>DeXposure-Claw: An Agentic System for DeFi Risk Supervision</strong> - Aijie Shu, Bowei Chen, Wenbin Wu, Cathy Yi-Hsuan Chen, Fengxiang He - [[pdf]](https://arxiv.org/pdf/2606.19501)</summary>

**Abstract:** Decentralized finance exposes supervisors to fast-moving, networked credit risks. General-purpose LLM agents fit this setting poorly: they over-read weak evidence and recommend high-stakes interventions, while existing evaluations offer no regulator-aligned way to measure the resulting false alarms. We introduce DeXposure-Claw, a forecast-grounded agentic supervision system that routes LLM decisions through structured evidence: (1) DeXposure-FM, a graph time-series foundation model, forecasts future exposure networks; (2) deterministic monitors and stress scenarios then turn those forecasts into typed alerts, attribution signals, and scenario evidence; and (3) data-health and confidence gates constrain escalation before DeXposure-Claw emits auditable supervisory tickets with rationales. We further develop DeXposure-Bench, a six-axis evaluation harness, whose decision axis scores tickets against a regulator-aligned absolute-loss ground truth and an explicit false-intervention rate. Experiments on five years of weekly real data fully support our system. Code is at this https URL.

**arXiv ID:** 2606.19501
</details>

<details>
<summary><strong>Uncertainty Decomposition for Clarification Seeking in LLM Agents</strong> - Gregory Matsnev - [[pdf]](https://arxiv.org/pdf/2606.19559)</summary>

**Abstract:** Recent position papers argue that the classical aleatoric/epistemic uncertainty framework is insufficient for interactive large language model (LLM) agents and call for underspecification-aware, decomposed, and communicable uncertainty representations that can unlock new agent capabilities such as proactive clarification seeking and shared mental-model building. Practical deployment constraints -- black-box APIs, interactive latency budgets, and the absence of labeled trajectories -- rule out logprob-based, multi-sampling, and training-based methods, leaving prompt-based estimation as the most viable family for surfacing such signals at deployment time. We answer this call with a simple prompt-based decomposition that separates action confidence from request uncertainty (u), enabling the agent to ask for clarification when the task specification is ambiguous. To evaluate it, we introduce two clarification-augmented benchmarks (WebShop-Clarification and ALFWorld-Clarification) in which 50% of tasks are deliberately underspecified, and systematically compare the proposed decomposition against ReAct+UE and Uncertainty-Aware Memory (UAM) across five LLM backbones (GPT-5.1, DeepSeek-v3.2-exp, GLM-4.7, Qwen3.5-35B, GPT-OSS-120B) on these variants together with the standard WebShop, ALFWorld, and REAL benchmarks for fault detection. Averaged across the five backbones, the proposed decomposition improves clarification F1 on ALFWorld-Clarification by 73% over ReAct+UE and by 36% over UAM, and leads clarification F1 on every backbone on WebShop-Clarification and on four of five backbones on ALFWorld-Clarification, indicating that the gains generalize beyond a single LLM.

**arXiv ID:** 2606.19559
</details>

<details>
<summary><strong>Beyond Static Leaderboards: Predictive Validity for the Evaluation of LLM Agents</strong> - Dhaval C. Patel, Kaoutar El Maghraoui, Shuxin Lin, Yusheng Li, Tianjun Feng, Chun-Yi Tsai, Yihan Sun, Wei Alexander Xin, Akshat Bhandari, Tanisha Rathod, Aaron Fan, Sanskruti Vijay Shejwal, Tomas Pasiecznik, Sagar Chethan Kumar, Tanmay Agarwal, Rohith Kanathur, Sam Colman, Amaan Sheikh, Dev Bahl, Ann Li, Krish Veera, Alimurtaza Mustafa Merchant, Shambhawi Baswaraj Bhure, Sajal Kumar Goyla, Chengrui Li, Kirthana Natarajan, Rui Li, Thomas Ajai, Rujing Li, Vivek G. Iyer, Sanjaii Vijayakumar, Yitong Bai, Ayal Yakobe, Darief Maes, Yassine Jebbouri, Tianyang Xu, Thai Quoc On, Vera Mazeeva, Winston Li, Yuval Shemla, Yeshitha Bhuvanesh, Rushin Bhatt, Siddharth Chethan Gowda, Alisha Vinod, Caroline Cahill, Shriya Aishani Rachakonda, Yunfeng Chen, Aryaman Agrawal, Aman Upganlawar, Mao Le Jonathan Ang, Yubin Sally Go, Madhav Rajkondawar, Yang-Jung Chen, Trisha Maturi, Ananya Kapoor, Andrew Li, Shrey Arora, Mana Abbaszadeh, Shen Li, Charles Xu, Byeolah Kwon - [[pdf]](https://arxiv.org/pdf/2606.19704)</summary>

**Abstract:** Agent benchmarks are growing fast, but no single benchmark touches more than four or five of the dimensions that deployment exposes. This paper aggregates the largest coordinated deep-dive of one MCP-based industrial-agent benchmark to date: fourteen parallel implementation studies covering new asset classes (including a multi-modal visual extension), alternative orchestrations, retrieval strategies, reasoning modes, infrastructure optimizations, and evaluation-methodology probes. Consolidating those studies with seven prior agent benchmarks, we argue that aggregate-score leaderboards systematically underspecify deployed-agent evaluation. Rankings derived from aggregate scores do not transfer to out-of-distribution settings; recent public-to-hidden competition retrospectives provide direct empirical evidence of this rank instability. We propose ranking configurations by predictive validity, the correlation between in-sample and out-of-sample rank, rather than in-sample mean, and report a twelve-tier measurement apparatus that exposes the deployment-relevant dimensions HELM and its agent-era successors collapse. The position is operationalized through three falsifiable out-of-distribution criteria with explicit thresholds; existing evidence partly supports it but is too thin to confirm. We close with a pre-registered pilot design and a field-level vision for what the next generation of agentic benchmarks should report.

**arXiv ID:** 2606.19704
</details>

<details>
<summary><strong>ORAgentBench: Can LLM Agents Solve Challenging Operations Research Tasks End to End?</strong> - Jiajun Li, Mingshu Cai, Yixuan Li, Yu Ding, Ran Hou, Guanyu Nie, Xiongwei Han, Wanyuan Wang - [[pdf]](https://arxiv.org/pdf/2606.19787)</summary>

**Abstract:** Large language models are increasingly deployed as autonomous agents for multi-step tasks in executable environments, yet their ability to perform realistic operations research (OR) work remains unclear. Existing OR evaluations often decouple modeling from solving, rely on pre-formalized or text-only instances, and rarely test the full workflow from operational artifacts to validated decisions. In this work, we introduce ORAgentBench, an execution-grounded benchmark for evaluating autonomous agents on challenging end-to-end operations research tasks. It contains 107 human-reviewed tasks across diverse operational scenarios, each packaged in an isolated environment with a natural-language brief, multi-file data, configuration artifacts, and a required submission schema. Agents must write and run solution code, and their submissions are evaluated by hidden validators for schema validity, hard-constraint feasibility, and normalized objective quality. Experiments with fourteen frontier agent-model configurations show that current agents remain far from reliable OR practice. The best agent passes only 35.51% of all tasks and 20.59% of hard tasks, and many feasible submissions still fall below the required quality threshold. Failure analysis further shows that errors are dominated by strategic weaknesses, including missed operational rules, brittle formulations, weak feasible-solution construction, and insufficient solution improvement. OR-specific procedural skills increase hard-task feasibility, but do not reliably improve solution quality or pass rate. These results suggest that progress in OR agents requires moving beyond plausible optimization code toward dependable, high-quality operational decision-making.

**arXiv ID:** 2606.19787
</details>

<details>
<summary><strong>When Lower Privileges Suffice: Investigating Over-Privileged Tool Selection in LLM Agents</strong> - Kaiyue Yang, Yuyan Bu, Jingwei Yi, Yuchi Wang, Biyu Zhou, Juntao Dai, Songlin Hu, Yaodong Yang - [[pdf]](https://arxiv.org/pdf/2606.20023)</summary>

**Abstract:** As LLM agents increasingly select tools autonomously, their choices among tools with different privileges become safety-relevant. However, prior tool-selection studies focus on safety-agnostic metadata preferences, leaving privilege-sensitive choices underexplored. To address this gap, we study over-privileged tool selection, in which an agent selects or escalates to a higher-privilege tool despite a sufficient lower-privilege alternative. We introduce ToolPrivBench to evaluate whether agents choose higher-privilege tools despite sufficient lower-privilege alternatives, measuring both initial selection and escalation after transient tool failures. Across eight domains and five recurring risk patterns, we find that over-privileged tool selection is common among mainstream LLM agents and is further amplified by transient failures. We further find that general safety alignment does not reliably transfer to least-privilege tool choice, while prompt-level controls provide only limited mitigation under transient failures. We therefore introduce a privilege-aware post-training defense that teaches agents to prefer sufficient lower-privilege tools and escalate only when necessary. Our mitigation experiments show that this defense substantially reduces unnecessary high-privilege tool use while preserving general capabilities.

**arXiv ID:** 2606.20023
</details>

<details>
<summary><strong>LLM agent safety, multi-turn red-teaming, jailbreak benchmarks, adversarial robustness, safety-critical systems</strong> - Hanwool Lee, Dasol Choi, Bokyeong Kim, Seung Geun Kim, Haon Park - [[pdf]](https://arxiv.org/pdf/2606.20408)</summary>

**Abstract:** Large language model (LLM) agents are increasingly proposed as supervisory components for safety-critical systems, yet their robustness under sustained, adaptive adversarial pressure remains poorly characterized. We present NRT-Bench, a benchmark for multi-turn red-teaming of LLM agents acting as operators of a safety-critical system, instantiated in a simulated nuclear power plant control room. A five-role operator team, each backed by a configurable LLM, runs a plant governed by six critical safety functions (CSFs), while adversaries inject messages over four channels in bounded multi-turn sessions with per-turn feedback. Harm is an objective signal rather than LLM-judged text: a run terminates the moment any CSF is lost, attributed to the causing message. Evaluating four frontier operator models under a fixed-attack paired-replay protocol, we find that adaptive multi-turn attacks reliably push the operator team past a safety limit: across the four models, between 8.7% and 12.1% of attack sessions end with the plant losing a critical safety function. Although the four models look almost equally robust by this aggregate rate, their failures barely overlap: of $149$ sessions, none defeat all four models while a third defeat at least one, so vulnerabilities are nearly disjoint across models rather than nested. The effect of added defences is strongly model-dependent: the same guardrail stack or safety-advisor agent that lowers attack success for one model can raise it for another. We release the simulation venue, attack dataset, and replay tooling for reproducible safety evaluation of LLM agents.

**arXiv ID:** 2606.20408
</details>

<details>
<summary><strong>RetailBench: Benchmarking long horizon reasoning and coherent decision making of LLM agents in realistic retail environments</strong> - Linghua Zhang, Jun Wang, Jingtong Wu, Zhisong Zhang - [[pdf]](https://arxiv.org/pdf/2606.15862)</summary>

**Abstract:** Large language model (LLM) agents have made rapid progress on short-horizon, well-scoped tasks, yet their ability to sustain coherent decisions in dynamic long-horizon environments remains uncertain. We introduce RetailBench, a data-grounded simulation benchmark for evaluating tool-using LLM agents in single-store supermarket operation. RetailBench models retail management as a partially observable decision process and is designed to support thousand-day-scale simulations. In this environment, agents must manage pricing, replenishment, supplier selection, shelf assortment, inventory aging, customer feedback, external events, and cash-flow constraints. We evaluate seven contemporary LLMs under representative agent frameworks over a 180-day evaluation horizon and compare them with a privileged oracle policy. Results show substantial variation across models: only a small subset survives the full evaluation horizon, and even the strongest LLM runs remain substantially behind the oracle policy in final net worth and sales outcomes. Behavioral analysis attributes these gaps to incomplete evidence acquisition, surface-level decision making, and the lack of a consistent long-horizon policy. RetailBench provides a controlled testbed for studying reliable autonomy in economically grounded long-horizon decision-making.

**arXiv ID:** 2606.15862
</details>

<details>
<summary><strong>SAGE-OPD: Selective Agent-Guided Intervention for Multi-Turn On-Policy Distillation</strong> - Yuhang Zhou, Lizhu Zhang, Yifan Wu, Mingyi Wang, Bo Peng, Jiayi Liu, Xiangjun Fan, Zhuokai Zhao - [[pdf]](https://arxiv.org/pdf/2606.19659)</summary>

**Abstract:** On-policy distillation (OPD) improves student models by training them on trajectories induced by their own policy, making it a promising approach for mitigating exposure bias in agent training. However, most OPD studies focus on single-turn settings, while realistic LLM agents interact with environments over multiple turns. In this regime, early errors can alter future observations and compound across the trajectory, and standard dense token-level OPD becomes brittle, as it may over-penalize semantically valid alternatives, reinforce local degeneracies such as repeated actions, and propagate unreliable teacher supervision on off-distribution histories. We propose SAGE-OPD, a verifier-free selective intervention framework specifically designed for multi-turn OPD. Instead of applying teacher supervision uniformly across all turns, SAGE-OPD first observes environment feedback and uses teacher judgment to decide whether each student response should be skipped or intervened on. To further address compounding errors, SAGE-OPD weights token-level distillation by teacher confidence, reducing the influence of uncertain teacher distributions on corrupted or ambiguous histories. Finally, SAGE-OPD applies loss normalization to preserve the overall loss scale of standard OPD while retaining selective turn-level weighting. Experiments on agent tasks show that SAGE-OPD consistently improves over baselines, achieving up to a 13.3% relative improvement in ALFWorld unseen success rate over standard OPD. Ablation studies further demonstrate that turn-level intervention, teacher confidence weighting, and loss normalization provide complementary benefits. Our results suggest that effective multi-turn OPD should remain on-policy, but teacher supervision should be selectively allocated to turns where intervention is necessary and reliable.

**arXiv ID:** 2606.19659
</details>

<details>
<summary><strong>AtomMem: Building Simple and Effective Memory System for LLM Agents via Atomic Facts</strong> - Yanyu Yao, Shangze Li, Zhi Zheng, Hui Zheng, Qi Liu, Tong Xu, Enhong Chen - [[pdf]](https://arxiv.org/pdf/2606.19847)</summary>

**Abstract:** Large language models (LLMs) demonstrate strong reasoning and generation abilities, but their fixed context windows limit long-term information accumulation and reuse across multi-session interactions. Existing memory-augmented systems often construct memory in a coarse and unstable manner, relying on inefficient memory representations or unstable unconstrained updates. To address these challenges, we propose AtomMem, a long-term memory system designed for value-dense storage and stable memory evolution. AtomMem introduces a Fact Executor, which selectively extracts high value atomic facts from long form interactions to serve as highly efficient memory representations. Subsequently, AtomMem organizes these facts into hierarchical event structures and temporal profiles, capturing coherent episodic contexts and tracking dynamically evolving user attributes over time. During retrieval, the system activates an associative memory graph to connect fragmented memories. Experiments on the LoCoMo benchmark confirm that AtomMem achieves state-of-the-art performance across various reasoning tasks, offering a scalable and economically viable solution for deploying intelligent personalized agents.

**arXiv ID:** 2606.19847
</details>

<details>
<summary><strong>Benchmarking LLM Agents on Meta-Analysis Articles from Nature Portfolio</strong> - Anzhe Xie, Weihang Su, Yujia Zhou, Yiqun Liu, Qingyao Ai - [[pdf]](https://arxiv.org/pdf/2606.17041)</summary>

**Abstract:** Meta-analysis is a demanding form of evidence synthesis that combines literature retrieval, PI/ECO-guided study selection, and statistical aggregation. Its structured, verifiable workflow makes it an ideal substrate for evaluating systematic scientific reasoning, yet existing benchmarks lack ground truth across the full retrieval-screening-synthesis pipeline. We introduce MetaSyn, a dataset of 442 expert-curated meta-analyses from Nature Portfolio journals. Each entry pairs a research question with PI/ECO criteria, a retrieval corpus of 140k PubMed articles, verified positive studies, hard negatives that are topically similar but PI/ECO-ineligible, and complete search strategies and date bounds.
Benchmarking twelve pipeline configurations (nine RAG variants and a protocol-driven agent) reveals a critical screening bottleneck: despite a retrieval ceiling of 90.9% recall at K=200, no system recovers more than 52.7% of ground-truth included literature. Current LLMs fail to reliably separate eligible studies from PI/ECO-failing distractors in pools of comparable topical relevance. Stage-attributed metrics capture where systems succeed and fail; a single end-to-end score does not.

**arXiv ID:** 2606.17041
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (25 papers)</h2></summary>

<details>
<summary><strong>Hidden Anchors in Multi-Agent LLM Deliberation</strong> - Apurba Pokharel, Ram Dantu - [[pdf]](https://arxiv.org/pdf/2606.19494)</summary>

**Abstract:** Multi-agent LLM deliberation, where agents exchange and revise answers over several rounds, is increasingly used to improve reasoning and accuracy, yet how and why it works is rarely modelled. Such deliberation mirrors how humans reach decisions. As social animals we are pulled both by the group, the herd effect that classical opinion-dynamics models such as DeGroot and Friedkin--Johnsen capture, and by our own internal belief, which they do not. We model multi-agent deliberation as a closed-loop dynamical system in which each agent carries a hidden internal belief, its anchor, that continually pulls its opinion regardless of its neighbours. We show this anchor can be recovered from the deliberation alone, and that it explains a behaviour classical consensus rules forbid: an agent's confidence in the correct answer can climb past where any agent started, escaping the space (convexhull) formed by the initial beliefs. Checking whether the recovered anchor also predicts held-out runs (generalizes) gives a simple test for when a model is truly driven bysuch an anchor. Across three open-weight model families this is a spectrum, not all-or-nothing. All anchors' influence are about equally strongly, but they differ in where the anchor sits, and only when it sits far from the initial opinions does deliberation escape the hull and need the full closed-loop model.

**arXiv ID:** 2606.19494
</details>

<details>
<summary><strong>AgentFinVQA: A Deployable Multi-Agent Pipeline for Auditable Financial Chart QA</strong> - Aravind Narayanan, Shaina Raza - [[pdf]](https://arxiv.org/pdf/2606.19782)</summary>

**Abstract:** Financial chart question answering in regulated settings demands more than accuracy: practitioners must know which answers to trust before acting on them, and many institutions cannot send client data to external model providers. Yet existing chart-QA agents are accuracy-focused and opaque, and most assume proprietary API access; to our knowledge, none combines auditability with on-premise deployability without significant accuracy compromise. We present AgentFinVQA, a multi-agent pipeline that decomposes each query into planning, OCR, legend grounding, visual inspection, and verification, recording every step in a traceable Model Evaluation Packet (MEP) per sample. On FinMME, AgentFinVQA improves $+7.68$ pp over a primary-backbone matched zero-shot baseline with a proprietary backbone (Gemini-3 Flash; 71.24% vs. 63.56%, McNemar $p \approx 1.1 \times 10^{-16}$), and $+4.84$ pp with open-weights Qwen3.6-27B-FP8 served locally. The verifier's verdict also serves as a useful confidence signal (68.2% vs. 55.6% exact accuracy on confirmed vs. revised answers), enabling human-in-the-loop review routing. Error analysis shows that question misunderstanding, legend confusion and extraction error account for nearly two-thirds of failures and are the categories least detected by the verifier, identifying clear directions for future work. Together these results show that auditable, on-premise financial chart QA is practical and that the open-weights system keeps most of the accuracy gains while enabling full data residency. We release our code to support reproducible evaluation.

**arXiv ID:** 2606.19782
</details>

<details>
<summary><strong>MetaResearcher: Scaling Deep Research via Self-Reflective Reinforcement Learning in Adversarial Virtual Environments</strong> - Wei Yu, Suxing Liu, Minjie Yu, Jiahao Wang, Zhijian Zheng, Haocheng Deng, Bing Li - [[pdf]](https://arxiv.org/pdf/2606.19893)</summary>

**Abstract:** Deep research agents have demonstrated remarkable capabilities in autonomous information gathering and synthesis, yet their training remains constrained by the static nature of simulated environments, the limits of fact-retrieval-only task designs, and the inefficiency of outcome-based reinforcement learning. In this work, we propose MetaResearcher, a novel framework that scales deep research agent training across four synergistic dimensions. First, we introduce an Evolving Virtual World that injects temporal dynamics and adversarial misinformation into the training environment, forcing agents to develop source credibility assessment and temporal conflict resolution skills. Second, we design Discovery-Oriented Tasks -- including hypothesis generation and contradiction resolution -- that transcend simple fact retrieval and push agents toward genuine research behaviors. Third, we propose a Self-Reflective Meta-Reward mechanism within the GRPO framework that jointly optimizes for answer correctness, search path efficiency, reflection depth, and tool call diversity, directly addressing the repetitive action loop problem observed in prior work. Fourth, we introduce a Heterogeneous Multi-Agent Swarm architecture comprising specialized Scout, Filter, and Synthesizer models that learn collaborative research strategies through coordinated reinforcement learning. Built upon the LiteResearcher infrastructure, MetaResearcher requires zero marginal API cost for training while targeting substantial improvements in both benchmark performance (GAIA, Xbench-DS) and epistemic robustness under adversarial conditions. We present the complete framework design, training methodology, and planned experimental validation.

**arXiv ID:** 2606.19893
</details>

<details>
<summary><strong>Multi-Agent Transactive Memory</strong> - To Eun Kim, Xuhong He, Dishank Jain, Ambuj Agrawal, Negar Arabzadeh, Fernando Diaz - [[pdf]](https://arxiv.org/pdf/2606.19911)</summary>

**Abstract:** The decentralized deployment of LLM agents with diverse capabilities across diverse tasks motivates infrastructure for knowledge sharing across heterogeneous agent populations. Just as search engines index human-generated artifacts to support human problem solving, retrieval systems can organize agent-generated artifacts for reuse across agent populations. We extend retrieval-augmented generation - which demonstrates the value of human-authored artifacts to individual agents - to retrieval of agent-generated artifacts supporting a population of agents. In particular, agent trajectories encode reusable procedural knowledge, yet these artifacts are typically discarded after a single use or retained only by the producing agent, forcing newly instantiated agents to repeatedly rediscover existing solutions. We propose Multi-Agent Transactive Memory (MATM), a framework for population-level storage and retrieval of agent-generated trajectories, where producer agents contribute trajectories to a shared repository and consumer agents retrieve them to improve task execution. We focus on interactive environments (ALFWorld and WebArena), where trajectories are long and encode especially rich procedural structure. Our experiments demonstrate that retrieving trajectories from MATM improves downstream task performance and reduces interaction steps without coordination or joint training. These results position MATM as a design pattern for population-level experience sharing in open agent ecosystems.

**arXiv ID:** 2606.19911
</details>

<details>
<summary><strong>Autonomous Event-Driven Multi-Agent Orchestration for Enterprise AI at Scale</strong> - Harsh Rao Dhanyamraju, Leonidas Raghav, Aaron Lee - [[pdf]](https://arxiv.org/pdf/2606.20058)</summary>

**Abstract:** Enterprise AI aims to move toward continuous event monitoring, detection, and action across specialist agents, yet existing multi-agent systems largely assume discrete request-response workflows and remain underexplored at enterprise scale. We evaluate DAG Plan and Execute and ReAct across 208 production-derived enterprise scenarios spanning Persona (<10 agents), Department (20-80), and Enterprise (200) scales, and introduce a Task Manager for continuous operation via priority inference, related-event merging, and preemption. Results show that scale, not task complexity, dominates orchestration performance: both architectures perform well at small scale but degrade at enterprise scale as agent discovery noise becomes the primary bottleneck, with simple tasks degrading more sharply than complex ones. DAG Plan and Execute offers higher precision and structured parallelization at smaller scales, but its higher overhead worsens at enterprise scale; ReAct is more robust by handling failures incrementally. The Task Manager reduces high-priority queue latency by 14-75% and improves related-event correctness by over 20 percentage points at enterprise scale.

**arXiv ID:** 2606.20058
</details>

<details>
<summary><strong>A Multi-Agent system for Multi-Objective constrained optimization</strong> - Federica Filippini - [[pdf]](https://arxiv.org/pdf/2606.20236)</summary>

**Abstract:** Many decision-making problems in computing and networking systems can be naturally formulated as cost-minimization problems under performance constraints. In dynamic environments, reinforcement learning (RL) is often used to solve such problems at runtime by embedding both costs and constraint violations into a single scalar reward through weighted penalty terms, following a Lagrangian-inspired formulation. However, in this context the behavior of the learned policy critically depends on the choice of these weights, which are typically selected manually. This makes it difficult to identify an appropriate trade-off between optimizing the primary objective and effectively avoiding constraint violations, particularly in non-stationary environments where their relative importance may change. This paper presents MAMO (Multi-Agent system for Multi-Objective constrained optimization), an approach to tackle this balancing problem through multi-agent RL. MAMO decouples task execution from objective design by formulating the selection of reward weights as a learning problem, providing a !rst step towards more autonomous and robust RL-based solutions for constrained optimization problems in dynamic environments.

**arXiv ID:** 2606.20236
</details>

<details>
<summary><strong>Trustworthy Multi-Agent Systems: Mitigating Semantic Drift with the Argent Signaling Protocol</strong> - Anantha Sharma - [[pdf]](https://arxiv.org/pdf/2606.19356)</summary>

**Abstract:** When multi-agent LLM systems produce bad answers, not all failures are equal: some answers are grounded in the right material but incomplete, while others are simply ungrounded and should be stopped. Current retry strategies treat both cases identically (try again and hope for the best), leaving human supervisors unable to tell whether a retry was warranted or whether the system should have halted instead.
We introduce the Argent Signaling Protocol (ASP), a compact machine-readable header that accompanies every AI-generated response with structured quality signals: certainty (@C), grounding (@G), stochasticity (@S), and an assumption index that classifies the evidentiary basis of each claim. These signals enable a controller to distinguish repairable failures from containment failures and route each case differently.
We evaluate ASP in two modes. In standalone mode, a 27-question document-grounded QA benchmark over the Array BioPharma/Ono license agreement compares baseline prompts against ASP-instrumented controller actions across three local GGUF models. On Qwen~(0.8B), ASP improves pass rate from 11.1% to 33.3% and mean term coverage from 36.7% to 65.4%; on Dobby~(8B), ASP produces 4 fail-to-pass recoveries, raising pass rate from 33.3% to 44.4%; on SmolLM3~(3B), ASP alternates between repair and containment per question. Aggregate improvement is meaningful (12/81 to 21/81 passes). In multi-agent mode, an ASP sidecar sits between a retrieval agent and a downstream decision agent; the sidecar blocks 100% of ungrounded upstream outputs from reaching the downstream agent (24/27 blocked, 0 ungrounded propagations).

**arXiv ID:** 2606.19356
</details>

<details>
<summary><strong>DynAMO:Dynamic Asset Management Orchestration via Topological Multi-Agent Scheduling</strong> - Kanishk Kushwaha, Vikrant Vinod Bansode, Harsh Vardhan, Dhaval C. Patel - [[pdf]](https://arxiv.org/pdf/2606.19382)</summary>

**Abstract:** While LLM-powered agents offer end-to-end automation for industrial asset lifecycles, real-world Industry 4.0 deployment is hindered by latency, concurrency instability, and safety risks. We present DynAMO (Dynamic Asset Management Orchestration), a deployment-ready engine using a Plan-then-Execute architecture to generate verifiable workflow graphs. DynAMO supports both SequentialWorkflow (topological execution) and ParallelWorkflow (dependency-aware concurrency). By dynamically identifying independent tasks, DynAMO preserves structural correctness and safety while significantly improving efficiency through controlled reasoning overlap.
Across six controlled experiments on the AssetOpsBench industrial benchmark, DynAMO demonstrates substantial performance and robustness gains. Parallel execution reduces end-to-end latency by a median of 1.6x over sequential orchestration, rising to 1.8x on highly parallelizable workflows. After instrumenting external tool calls with realistic latencies, a latency decomposition shows that LLM reasoning and orchestration still account for more than 90% of execution time, identifying model inference as the primary system bottleneck. Structured context pruning reduces inference latency by approximately 30%, and DynAMO maintains correct functional behaviour (task completion, agent sequencing, and output quality) while exhibiting graceful degradation under controlled fault injection. Reproducibility analysis further confirms stable execution under repeated runs, with parallel scheduling reducing latency variance. These findings establish DynAMO as a practical blueprint for scalable, safe, and latency-aware agent deployment in Industry 4.0 automation pipelines. Code is available at: this https URL

**arXiv ID:** 2606.19382
</details>

<details>
<summary><strong>Before the Pull Request: Mining Multi-Agent Coordination</strong> - Dipankar Sarkar - [[pdf]](https://arxiv.org/pdf/2606.19616)</summary>

**Abstract:** Autonomous coding agents now open millions of pull requests, yet large-scale studies find their PRs are produced faster but accepted less often - a coordination and trust gap that pull-request-level telemetry cannot explain. We argue the missing signal lives before the PR, in how concurrent agents claim, divide, and collide over shared work. We study this process through grite, our open-source coordination substrate that needs no central server and stores its records inside git itself, so its append-only, signed event log captures the coordination process directly. We show that (i) this shared substrate reduces duplicate and conflicting work at bounded overhead - the share of work that merely re-does a teammate's task falls from 78% to 0% while useful throughput more than triples; (ii) every agent's copy of the log converges to the same state with no write silently dropped, where a file-based tracker loses concurrent writes; and (iii) the log is a mineable artefact from which concrete failure modes - conflicting edits, lock starvation, redundant rediscovery, race-to-close - are automatically recoverable with provenance, several invisible in pull-request history. We release the dataset, harness, and mining toolkit.

**arXiv ID:** 2606.19616
</details>

<details>
<summary><strong>Formal Verification of Learned Multi-Agent Communication Policies via Decision Tree Distillation</strong> - Ahmad Farooq, Kamran Iqbal - [[pdf]](https://arxiv.org/pdf/2606.19632)</summary>

**Abstract:** Multi-agent reinforcement learning (MARL) enables agents to develop coordination strategies through emergent communication, but neural policies lack the formal safety guarantees required for safety-critical robotic deployment in drone swarms and autonomous vehicle fleets. We present the first end-to-end framework for safety verification of learned multi-agent communication policies through policy abstraction: neural policies are distilled into interpretable decision trees, then formally verified, with empirical validation confirming that verified safety properties transfer to original networks. Our four-stage pipeline consists of domain-specific feature extraction from agent observations, decision tree distillation achieving 97.9% +/- 1.2% fidelity to neural policies, automated translation to PRISM probabilistic model checker specifications with complete feature-to-state-variable correspondence, and compositional verification of Probabilistic Computation Tree Logic (PCTL) properties via pairwise decomposition with union-bound aggregation and empirical neighbor modeling. Evaluating Vector-Quantized Variational Information Bottleneck (VQ-VIB) policies for multi-drone coordination with 5-7 agents, we verify 18 temporal logic properties across safety, liveness, and cooperation, achieving 88.9% property satisfaction with all five safety thresholds satisfied (0.3% collision probability vs. 1% threshold). Monte Carlo validation of original neural policies confirms that verified safety properties transfer with <=0.6 percentage-point deviation (95% CI). Discrete VQ-VIB messages provide +11.6 to +13.6 percentage-point fidelity advantages over continuous methods, enabling 3-4x faster verification. Our framework provides empirically validated safety verification for distilled policy abstractions, serving as a practical bridge between deep MARL and formal safety workflows for multi-robot deployment.

**arXiv ID:** 2606.19632
</details>

<details>
<summary><strong>Hierarchical Control in Multi-Agent Games: LLM-based Planning and RL Execution</strong> - Jannik Hösch, Alessandro Sestini, Florian Fuchs, Amir Baghi, Joakim Bergdahl, Konrad Tollmar, Jean-Philippe Barrette-LaPierre, Linus Gisslén - [[pdf]](https://arxiv.org/pdf/2606.20014)</summary>

**Abstract:** Reinforcement learning (RL) has achieved strong performance in sequential decision-making, yet scaling to complex multi-agent environments remains challenging due to sparse rewards, large state-action spaces, and the difficulty of learning coordinated strategies. We propose a hierarchical architecture where a pretrained large language model (LLM) acts as a centralized strategic controller that selects among specialized RL skill policies for a team of agents, while RL policies handle reactive low-level execution. We evaluate this hybrid system in a competitive 2v2 King of the Hill environment against behavior tree (BT) and \emph{``Flat''} RL (end-to-end training without skill decomposition) baselines. The LLM+RL system achieves task performance statistically equivalent to hand-crafted BT (46.4\% vs 51.5\% win rate, $p=0.103$) while both significantly outperform Flat RL trained without skill decomposition. A user study ($n=15$) reveals that 60\% of participants perceive LLM+RL agents as the most human-like ($p=0.027$), citing behavioral adaptability and tactical variability. These results demonstrate that pretrained LLM reasoning can effectively orchestrate pretrained RL skills, achieving competitive multi-agent coordination and superior perceived believability without manual rule engineering.

**arXiv ID:** 2606.20014
</details>

<details>
<summary><strong>AutoPass: Evidence-Guided LLM Agents for Compiler Performance Tuning</strong> - Zepeng Li, Jie Ren, Zhanyong Tang, Jie Zheng, Zheng Wang - [[pdf]](https://arxiv.org/pdf/2606.20373)</summary>

**Abstract:** Large Language Models (LLMs) show promise for code compilation tasks, but applying them to runtime performance tuning is difficult due to complex microarchitectural effects and noisy runtime measurements. We present AutoPass, a multi-agent framework for compiler performance tuning that uses compiler and runtime evidence to guide LLM-generated optimization decisions. Rather than treating the compiler as a black box like prior auto-tuning schemes, AutoPass opens up the compiler to the LLM, enabling it to query compiler-internal optimization states and analyze the intermediate representation to orchestrate compiler options. The search process iteratively refines optimization configurations using measured runtime feedback to diagnose regressions and guide latency-improving edits. AutoPass operates in an inference-only, training-free setting and requires no offline training or task-specific fine-tuning, making it readily applicable to new benchmarks and platforms. We implement AutoPass on the LLVM compiler and evaluate it on server-grade x86-64 and embedded ARM64 systems. AutoPass outperforms expert-tuned heuristics and classical autotuning methods, achieving geometric-mean speedups of 1.043x and 1.117x over LLVM -O3 on x86-64 and ARM64, respectively.

**arXiv ID:** 2606.20373
</details>

<details>
<summary><strong>Optimal Order of Multi-Agent and General Many-Body Systems</strong> - Jake J. Xia - [[pdf]](https://arxiv.org/pdf/2606.20485)</summary>

**Abstract:** This paper develops a general framework for analyzing multi-agent systems with feedback loops between agents actions and collective observations. The framework is built on two fundamental agent-level variables: power, which measures agent influence on collective outcomes, and response functions, which determine how agents react to observations. We derive how macroscopic properties, including total power, useful power, entropy, order, fragility, and mobility, emerge from these two variables of heterogeneous agents. To study the trade off between growth and resilience, we introduce a system-level utility function parameterized by a risk-appetite coefficient and derive an optimal degree of order that balances productivity, stability, and adaptability. The analysis suggests that stronger synchronization can increase collective output but may also increase systemic fragility and reduce mobility. We further argue that order, entropy, information, and useful energy are task-dependent and system-relative concepts whose meanings depend on the objectives of the system. By measuring and designing agent power distributions and response functions, it may be possible to better understand, predict, and optimize collective behavior and identify the conditions under which collective intelligence and optimal order emerge.

**arXiv ID:** 2606.20485
</details>

<details>
<summary><strong>Contagion Networks: Evaluator Bias Propagation in Multi-Agent LLM Systems</strong> - Zewen Liu - [[pdf]](https://arxiv.org/pdf/2606.20493)</summary>

**Abstract:** When large language models serve as evaluators in multi-agent systems, their systematic evaluation biases propagate through the agent network. We introduce Contagion Networks, a formal framework for measuring how evaluator biases spread across interacting LLM agents. In a controlled 3-agent experiment using DeepSeek-chat with three distinct evaluator bias profiles (structured, balanced, evidence-based), we measure the Cross-Agent Contagion Matrix Gamma_3 and find that evaluator biases consistently propagate between agents (gamma in [0.157, 0.352]), even within the same underlying model. We identify three propagation regimes governed by the spectral radius rho(Gamma_N), and demonstrate that homogeneous-model agents produce contagion coefficients 3-5x weaker than cross-model coefficients observed in prior work (MM-EPC: gamma approx 0.85-1.3), placing them in the suppression regime. We show that increasing evaluator committee size from k=1 to k=3 reduces effective contagion by 72.4%, providing an actionable mitigation strategy. We release the open-source Contagion Network experimental framework.

**arXiv ID:** 2606.20493
</details>

<details>
<summary><strong>UniMM: A Unified Mixture Model Framework for Multi-Agent Simulation</strong> - Longzhong Lin, Xuewu Lin, Kechun Xu, Haojian Lu, Lichao Huang, Rong Xiong, Yue Wang - [[pdf]](https://arxiv.org/pdf/2501.17015)</summary>

**Abstract:** Simulation plays a crucial role in assessing autonomous driving systems, where the generation of realistic multi-agent behaviors is a key aspect. In multi-agent simulation, the primary challenges include behavioral multimodality and closed-loop distributional shifts. In this study, we formulate a unified mixture model (UniMM) framework for generating multimodal agent behaviors, which can cover the mainstream methods including regression-based mixture models and discrete NTP models. Furthermore, we introduce a closed-loop sample generation approach tailored for mixture models to mitigate distributional shifts. Within the UniMM framework, we recognize critical configurations from both the model and data perspectives. We conduct a systematic examination of various model configurations, and comprehensively characterize their effects. Moreover, our investigation into the data configuration highlights the pivotal role of closed-loop samples in achieving realistic simulations. To extend the benefits of closed-loop samples across a broader range of mixture models, we further introduce a temporal disentanglement-and-alignment mechanism to address the shortcut learning and off-policy learning issues. Leveraging insights from our exploration, the distinct variants proposed within the UniMM framework, including discrete, anchor-free, and anchor-based models, all achieve state-of-the-art performance on the WOSAC benchmark.

**arXiv ID:** 2501.17015
</details>

<details>
<summary><strong>MEAL: A Benchmark for Continual Multi-Agent Reinforcement Learning</strong> - Tristan Tomilin, Luka van den Boogaard, Samuel Garcin, Constantin Ruhdorfer, Bram Grooten, Fabrice Kusters, Yali Du, Andreas Bulling, Mykola Pechenizkiy, Meng Fang - [[pdf]](https://arxiv.org/pdf/2506.14990)</summary>

**Abstract:** Benchmarks play a central role in reinforcement learning (RL) research, yet their computational constraints often shape what is studied. Despite the motivation of lifelong learning, most continual RL papers consider only 3-10 sequential tasks, as CPU-bound environments make longer sequences impractical. Meanwhile, continual learning in cooperative multi-agent settings remains largely unexplored. To address these gaps, we introduce MEAL (Multi-agent Environments for Adaptive Learning), the first benchmark for continual multi-agent RL. By leveraging JAX and GPU acceleration, MEAL enables training on sequences of 100 tasks in a few hours on a single GPU. We find that long task sequences reveal failure modes that do not appear at smaller scales.

**arXiv ID:** 2506.14990
</details>

<details>
<summary><strong>SIGMA: Search-Augmented On-Demand Knowledge Integration for Agentic Mathematical Reasoning</strong> - Ali Asgarov, Umid Suleymanov, Aadyant Khatri - [[pdf]](https://arxiv.org/pdf/2510.27568)</summary>

**Abstract:** Solving mathematical reasoning problems requires not only accurate access to relevant knowledge but also careful, multi-step thinking. However, current retrieval-augmented models often rely on a single perspective, follow inflexible search strategies, and struggle to effectively combine information from multiple sources. We introduce SIGMA (Search-Augmented On-Demand Knowledge Integration for AGentic Mathematical reAsoning), a unified framework that orchestrates specialized agents to independently reason, perform targeted searches, and synthesize findings through a moderator mechanism. Each agent generates hypothetical passages to optimize retrieval for its analytic perspective, ensuring knowledge integration is both context-sensitive and computation-efficient. When evaluated on challenging benchmarks such as MATH500, AIME, and PhD-level science QA GPQA, SIGMA consistently outperforms both open- and closed-source systems, achieving an absolute performance improvement of 7.4%. Our results demonstrate that multi-agent, on-demand knowledge integration significantly enhances both reasoning accuracy and efficiency, offering a scalable approach for complex, knowledge-intensive problem-solving. We will release the code upon publication.

**arXiv ID:** 2510.27568
</details>

<details>
<summary><strong>MoCA-Agent: A Market-of-Claims Code Agent for Financial and Numerical Reasoning</strong> - Abdelrahman Abdallah, AbdelRahim A. Elmadany, Sameh Al Natour, Hasan Cavusoglu, Adam Jatowt, Muhammad Abdul-Mageed - [[pdf]](https://arxiv.org/pdf/2606.11537)</summary>

**Abstract:** Financial and tabular question answering requires more than fluent reasoning: answers must be grounded in the exact facts, formulas, units, signs, and scales that support them. A single misread cell or incorrect operation can silently produce a plausible but wrong result. We introduce \textsc{MOCA-Agent}, a market-of-claims code agent that replaces free-form multi-agent debate with claim-level verification. The system decomposes each question into typed atomic claims, asks specialist trader agents to buy or sell those claims, clears their orders into confidence-weighted accept/reject decisions, and synthesizes an executable Python program from market-supported evidence. A code-aware verifier then checks the program for execution, structural consistency, and common financial reasoning errors, with at most one market-aware repair round. Across ten public benchmarks spanning financial numerical reasoning, general tabular reasoning, ESG question answering, and multimodal chart reasoning, \textsc{MOCA-Agent} achieves strong performance using a fixed Qwen3.6-27B backbone, including $78.3\%$ on FinQA, $76.0\%$ on FinanceMath, $71.2\%$ on MultiHiertt, $86.9\%$ on ESGenius, and $85.6\%$ average on FinChart-Bench. These results show that aggregating evidence at the level of atomic claims, rather than whole answers, improves robustness in high-stakes numerical reasoning.\footnote{The code and data are available: this https URL.

**arXiv ID:** 2606.11537
</details>

<details>
<summary><strong>SIGMA: Skill-Incidence Graphs for Compositional Multi-Agent Design</strong> - Kun Zeng, Yu Huo, Siyu Zhang, Yuecheng Zhuo, Yuquan Lu, Haoyue Liu, Siyue Chen, Xiaoying Tang - [[pdf]](https://arxiv.org/pdf/2606.19758)</summary>

**Abstract:** Existing graph-based multi-agent system (MAS) designers mainly improve collaboration by optimizing communication topologies over predefined agents, roles, or groups. However, because each node remains a closed-set entity, these methods struggle to generalize to tasks that require unseen combinations of capabilities. We propose SIGMA, a skill-incidence graph framework that constructs agents as task-conditioned bundles of reusable skills. Given a task and a skill library, SIGMA predicts a skill-agent incidence matrix, composes agent node embeddings from selected skills, and decodes a communication topology over the constructed agents. During execution, skill-specific mailboxes route messages to the relevant assigned capabilities, making the incidence structure directly operational. Across six reasoning and coding benchmarks with three base LLMs, SIGMA achieves the best average performance and improves over CARD, the strongest non-compositional topology-based baseline, by 2.06, 2.36, and 1.75 points, respectively. It also shows stronger robustness to unseen skill libraries, with an average performance drop of only 0.96 points. These results suggest that compositional node construction is a complementary and important axis for multi-agent design beyond communication topology optimization. Code is available at this https URL.

**arXiv ID:** 2606.19758
</details>

<details>
<summary><strong>Phoenix: Safe GitHub Issue Resolution via Multi-Agent LLMs</strong> - Kipngeno Koech, Muhammad Adam, Baimam Boukar Jean Jacques, Joao Barros - [[pdf]](https://arxiv.org/pdf/2606.20243)</summary>

**Abstract:** We present Phoenix, a multi-agent LLM system that resolves GitHub issues from triage through pull-request creation, combining seven layered safety controls with a baseline-aware test evaluation strategy. Phoenix decomposes the work across six specialized agents. Planner, reproducer, coder, tester, failure analyst and Pull Request (PR) agent, all coordinated by a label-based GitHub webhook state machine. Every change is checked against a baseline test run before a pull request is opened. On a 24-instance slice of SWE-bench Lite. run on the production webhook path, Phoenix oracle-resolves 75% of instances with no pass-to-pass regressions on successful runs; this curated slice is not directly comparable to full-split leaderboard results, and we discuss the limits of the comparison. A complementary pilot on 42 real issues across 14 repositories yields 100% correctness preservation (CP; mean 122s on the hard tier). Manual inspection shows that about half of the resulting pull requests are well-targeted fixes. The other half place code at incorrect paths, a planner localization limitation we are addressing with retrieval. We also report the deployment failure modes (WAF filtering, token expiry, permission boundaries, flaky CI) that motivated each safety mechanism.

**arXiv ID:** 2606.20243
</details>

<details>
<summary><strong>Modeling U.S. Attitudes Toward China via an Event-Steered Multi-Agent Simulator</strong> - Chenxu Zhu, Hantao Yao, Wu Liu, Junbo Guo, Yongdong Zhang - [[pdf]](https://arxiv.org/pdf/2606.06971)</summary>

**Abstract:** Understanding the dynamic evolution of opinions, such as U.S. public attitudes toward China, is essential for assessing geopolitical risks. However, existing LLM-based multiagent simulators predominantly rely on static rules and fixed datasets, limiting their ability to capture the dynamic, event-driven nature of macro-level opinion shifts in real-world settings. To address this limitation, we propose an Event-Steered Multi-Agent Simulator (ES-MAS), in which significant events and daily news continuously drive opinion evolution through dynamic interactions among agents. We first construct the China-U.S. Relation Evolution (CURE) dataset, covering 20 quarters from 2021 to 2025, including 258 major events and over 14,000 daily news articles, and providing a comprehensive temporal foundation for modeling opinion dynamics. Building upon the CURE dataset, we propose a Dual-Stream Data Integration Engine (DSDIE) that aligns simulations with historical timelines via macro-level events while enabling personalized information exposure based on individual agent profiles and contextual signals. Furthermore, we design a News-Driven Dynamic Interaction (NDDI) module, which adaptively groups agents with shared news interests into localized interaction contexts, facilitating bottom-up consensus formation while mitigating the risk of isolated information cocoons. Experimental results on the CURE dataset demonstrate that ES-MAS substantially outperforms existing simulators in reproducing real-world historical trends, offering a scalable and effective framework for modeling dynamic opinion evolution.

**arXiv ID:** 2606.06971
</details>

<details>
<summary><strong>TransLaw: A Large-Scale Dataset and Multi-Agent Benchmark Simulating Professional Translation of Hong Kong Case Law</strong> - Xi Xuan, Chunyu Kit - [[pdf]](https://arxiv.org/pdf/2507.00875)</summary>

**Abstract:** Translating Hong Kong Court Judgments from English to Traditional Chinese is mandated by Articles 8-9 of the Basic Law, yet remains constrained by a shortage of parallel resources and rigorous demands on legal terminology, citation format, and judicial style. We introduce HKCFA Judgment 97-22, the first large-scale sentence-aligned parallel corpus for HK case law, comprising 344 professionally translated judgments (11,099 sentence pairs; 2.1M tokens) spanning 1997-2022. Building on this resource, we propose TransLaw, a multi-agent framework that decomposes translation into word-level expression, sentence-level translation, and multidimensional review, integrating a specialized Hong Kong legal glossary database, Retrieval-Augmented Generation, and iterative feedback, with four-dimensional expert review covering semantic alignment, terminology, citation, and style. Benchmarking 13 open-source and commercial LLMs, we demonstrate that TransLaw significantly outperforms single-agent baselines across all evaluated models, with convergence within 3 iterations. Human evaluation by 10 certified legal translators using our proposed Legal ACS metric confirms gains in legal-semantic accuracy, while showing that TransLaw still trails human experts in stylistic naturalness. The dataset and benchmark code are available at this https URL.

**arXiv ID:** 2507.00875
</details>

<details>
<summary><strong>Superhuman Safe and Agile Racing through Multi-Agent Reinforcement Learning</strong> - Ismail Geles, Leonard Bauersfeld, Markus Wulfmeier, Davide Scaramuzza - [[pdf]](https://arxiv.org/pdf/2605.22748)</summary>

**Abstract:** Autonomous systems have achieved superhuman performance in isolation or simulation, yet they remain brittle in shared, dynamic real-world spaces. This failure stems from the dominant single-agent paradigm for physical applications, where other actors are ignored or treated as environmental noise, preventing effective coordination. Here we show that multi-agent reinforcement learning provides the essential safety scaffolding required for real-world interaction. Using high-speed quadrotor racing as a high-stakes testbed, we train agents to navigate complex aerodynamic interactions and strategic maneuvering with a variable number of racers. Through league-based self-play, agents evolve sophisticated anticipatory behaviors, including proactive collision avoidance, overtaking, and handling multi-agent physical interactions, including aerodynamic downwash. Our agents outperform a champion-level human pilot in multi-player races at speeds exceeding 22 m/s, while simultaneously reducing collision rates by 50 % compared to state-of-the-art single-agent baselines. Crucially, training with diverse artificial agents enables zero-shot generalization to safer human interaction. These results suggest that the path to robust robotic co-existence lies not in isolated safety constraints, but in the rigorous demands of multi-agent interaction. Multimedia materials are available at: this https URL

**arXiv ID:** 2605.22748
</details>

<details>
<summary><strong>TSAssistant: A Human-in-the-Loop Agentic Framework for Automated Target Safety Assessment</strong> - Xiaochen Zheng, Zhiwen Jiang, David Tokar, Yexiang Cheng, Alvaro Serra, Melanie Guerard, Klas Hatje, Tatyana Doktorova - [[pdf]](https://arxiv.org/pdf/2604.23938)</summary>

**Abstract:** Target Safety Assessment (TSA) requires systematic integration of genetic, transcriptomic, target homology, pharmacological, and clinical data to evaluate potential safety liabilities of therapeutic targets. This process is labor-intensive and expert-dependent, posing challenges in scalability and reproducibility. We present TSAssistant, a human-in-the-loop multi-agent framework that decomposes TSA report generation into a workflow of specialized subagents: Research Subagents that each ground and cite a single TSA domain, and Synthesis Subagents that integrate findings across domains. Subagents retrieve and synthesize evidence from curated biomedical sources through standardized tool interfaces and produce individually citable, evidence-grounded sections, with behavior shaped by a hierarchical instruction architecture that separates coordination logic from domain expertise and user intent. To complement these soft constraints, programmatic execution hooks and persistent memory stores enforce hard constraints across the workflow, while an interactive refinement loop allows experts to review and revise individual sections with full conversational context preserved across iterations. Rather than a single holistic comparison, we decompose report quality into reproducibility, evidential grounding, task-level accuracy, and controllability under expert oversight, finding high reproducibility and grounding, substantial agreement with the human reference, and net-positive expert-driven refinement.

**arXiv ID:** 2604.23938
</details>

<details>
<summary><strong>Multi-Granular Attention-Driven Reinforcement Learning Framework for Web Intelligent Enhancement Systems</strong> - Navin Chhibber, Deepak Singh, Anokh Kishore, Nikita Chawla, K. Anguraj - [[pdf]](https://arxiv.org/pdf/2606.19690)</summary>

**Abstract:** From the past few years, web intelligent enhancement systems increasingly rely on heterogeneous and dynamic web data to deliver personalized, context-aware services. However, traditional machine learning, deep learning, and reinforcement learning models often struggle with semantic understanding, adaptability, and scalability in continuously evolving web environments. In this research, a Multi-Granular Attention-based Reinforcement Web Intelligent Enhancement System (MGAR-WIES) is proposed to address the challenges by integrating semantic graph modeling, attention mechanisms, and adaptive reinforcement learning. Initially, heterogeneous web data comprising structured, semi-structured and unstructured sources are collected and preprocessed for generating unified feature representations. These representations are transformed into a dynamic semantic graph, where entities and their relationships are modeled by using graph embeddings enhanced by attention mechanisms for capturing both local relevance and global contextual dependencies. Subsequently, an adaptive multi-agent reinforcement learning strategy leverages the attention-aware semantic states to optimize personalized web actions like content recommendation, navigation optimization, and service adaptation. Finally, the continuous online feedback is further integrated to update graph representations and learning policies in real time by ensuring sustained adaptability and performance. The proposed MGAR-WIES acheived better results in terms of accuracy (80%) when compared with existing approaches.

**arXiv ID:** 2606.19690
</details>

</details>

<details open>
<summary><h2>Other Agent Research (12 papers)</h2></summary>

<details>
<summary><strong>Configurable Clinical Information Extraction with Agentic RAG: What Works, What Breaks, and Why</strong> - Osman Alperen Çinar-Koraş, Marie Bauer, Sameh Khattab, Merlin Engelke, Moon Kim, Stephan Settelmeier, Shigeyasu Sugawara, Fabian Freisleben, Felix Nensa, Jens Kleesiek - [[pdf]](https://arxiv.org/pdf/2606.19602)</summary>

**Abstract:** Patient contexts span hundreds of heterogeneous documents and thousands of structured data points, yet the document-level metadata that AI systems need for retrieval and triage is absent or incomplete. Standard retrieval-augmented generation fails on this data, mishandling temporal reasoning, cross-document dependencies, and missing metadata. We deploy ACIE (Agentic Clinical Information Extraction) at University Medicine Essen: an on-premise agentic RAG pipeline that reasons over complete patient contexts and grounds every answer in source passages for clinician verification. We quantify the metadata gap, trace the architectural decisions it shaped, and evaluate extraction alongside an independent retrospective lymphoma registry study, in which nuclear-medicine physicians verify every extracted value against its cited sources. Across 7,326 judgments, clinicians accepted 96.5\% of extractions, with per-type acceptance ranging from 80\% to 99\%.

**arXiv ID:** 2606.19602
</details>

<details>
<summary><strong>ENPIRE: Agentic Robot Policy Self-Improvement in the Real World</strong> - Wenli Xiao, Jia Xie, Tonghe Zhang, Haotian Lin, Letian "Max" Fu, Haoru Xue, Jalen Lu, Yi Yang, Cunxi Dai, Zi Wang, Jimmy Wu, Guanzhi Wang, S. Shankar Sastry, Ken Goldberg, Linxi "Jim" Fan, Yuke Zhu, Guanya Shi - [[pdf]](https://arxiv.org/pdf/2606.19980)</summary>

**Abstract:** Achieving dexterous robotic manipulation in the real world heavily relies on human supervision and algorithm engineering, which becomes a central bottleneck in the pursuit of general physical intelligence. Although emerging coding agents can generate code to automate algorithm search, their successes remain largely confined in digital environments. We conjecture that the missing abstraction to automate robotics research is a repeatable feedback loop for real-world policy improvement: reset the scene, execute a policy, verify the outcome, and refine the next iteration. To bridge this gap, we introduce ENPIRE, a harness framework for coding agents that instantiates this physical feedback routine with four core modules: an Environment module (EN) for automatic reset and verification, a Policy Improvement module (PI) that launches policy refinement, a Rollout module (R) to evaluate policies with one or multiple physical robots operating in parallel, and an Evolution module (E) in which coding agents analyze logs, consult literature, improve training infrastructure and algorithm code to address failure modes. This closed-loop system transforms real-world manipulation learning into a controllable optimization procedure, minimizing human effort while allowing fair ablations across training recipe and agent variants. Powered by ENPIRE, frontier coding agents can autonomously train a policy to achieve a 99% success rate on challenging, dexterous manipulation tasks, such as organizing a pin box, fastening a zip tie, and tool use, a process that further accelerates when we dispatch an agent team on a robot fleet. Our results suggest a practical and scalable path toward deploying coding agents to autonomously advancing robotics in the physical world.

**arXiv ID:** 2606.19980
</details>

<details>
<summary><strong>ScaffoldAgent: Utility-Guided Dynamic Outline Optimization for Open-Ended Deep Research</strong> - Zhibang Yang, Xinke Jiang, Yuzhen Xiao, Ruizhe Zhang, Yue Fang, XinFei Wan, Zhengxing Song, Yuxuan Liu, Yuheng Huang, Xu Chu, Junfeng Zhao, Yasha Wang - [[pdf]](https://arxiv.org/pdf/2606.20122)</summary>

**Abstract:** Open-ended deep research (OEDR) requires systems to acquire knowledge through multi-round retrieval and generate coherent long-form reports. The outline plays a central role as a structural scaffold that coordinates retrieval, evidence organization, and generation. However, existing methods either fix the outline before writing or refine it with local heuristics, leading to scaffold drift under continuous information accumulation and delayed feedback for evaluating outline modifications. We propose ScaffoldAgent, a utility-guided dynamic outline optimization framework for OEDR. ScaffoldAgent models outline evolution as a structured decision process with three operations: Expansion, Contraction, and Revision, enabling controlled updates to the report scaffold. It further introduces a utility-guided feedback mechanism that estimates the downstream value of each outline operation from retrieval gain, structural coherence, and trial-generation quality. The resulting utility signal guides node selection, operation scheduling, and termination during inference. Experiments on DeepResearch Bench and DeepResearch Gym show that ScaffoldAgent consistently improves long-form report generation and factual grounding over existing deep research agents.

**arXiv ID:** 2606.20122
</details>

<details>
<summary><strong>LedgerAgent: Structured State for Policy-Adherent Tool-Calling Agents</strong> - Md Nayem Uddin, Amir Saeidi, Eduardo Blanco, Chitta Baral - [[pdf]](https://arxiv.org/pdf/2606.20529)</summary>

**Abstract:** Policy-adherent tool-calling agents in customer-service domains must maintain task states across turns while calling tools and obeying domain policies. Task states consist of relevant facts, identifiers, constraints, and conditions observed through user interaction and tool calls. In standard agents, task states are not represented separately. Observations, tool returns, and policy instructions are placed in the prompt, leaving agents to reconstruct the relevant states from the prompt each time they decide what to do next. This design makes state management implicit, creating two common failure modes. An agent may retrieve the right facts but later ground its decision in stale, missing, or incorrect information; and a syntactically valid tool call may still violate a domain policy that depends on the current task state. We introduce \textsc{LedgerAgent}, an inference-time method for tool-calling agents that maintains observed task states in a separate ledger and renders the states into the prompt. The ledger is also used to check state-dependent policy constraints before environment-changing tool calls are executed, blocking policy violations. Across four customer-service domains and a mixed panel of open- and closed-weight models, \textsc{LedgerAgent} improves average pass\textasciicircum{}k over a standard prompt-based tool-calling approach, with the largest gains under stricter multi-trial consistency metrics.

**arXiv ID:** 2606.20529
</details>

<details>
<summary><strong>Agentic Electronic Design Automation: A Handoff Perspective</strong> - Jiawei Liu, Peiyi Han, Yuntao Lu, Su Zheng, Fengyu Yan, Bei Yu - [[pdf]](https://arxiv.org/pdf/2606.19795)</summary>

**Abstract:** Electronic design automation (EDA) is inherently multi-stage and handoff-heavy. Design artifacts, flow scripts, and engineering decisions cross tool, session, and organizational boundaries before final implementation, signoff, or release. Each transfer carries explicit and implicit requirements that may not be fully captured by stage-local checks. LLM-based agents now invoke EDA tools directly, embed retrieved knowledge in executable scripts, and hand off state across sessions and stages. Once their outputs condition downstream engineering decisions, the transferred object must satisfy a handoff contract and meet the assumptions of its next consumer. This survey introduces handoff validity as its organizing principle. A handoff is valid when the transferred object satisfies the consumer's acceptance conditions and carries sufficient context, evidence, and provenance for downstream use. We review 82 systems and classify them into three boundary classes. Stage-Bound systems establish validity within a single EDA stage or bounded verification task. Flow-Bound systems preserve coherent workflow state across tools, invocations, and sessions. Organization-Bound systems maintain source grounding, provenance, scope, and admissibility across knowledge and authority boundaries. For each class, we analyze handoff contracts, handoff objects, coordination mechanisms, and open questions. These analyses motivate a five-layer EDA agent communication protocol (EACP), covering the agent discovery, agent message, tool invocation, workflow orchestration, and security and IP protocols. We aim to provide a common vocabulary and research agenda for trustworthy agentic EDA.

**arXiv ID:** 2606.19795
</details>

<details>
<summary><strong>Beyond Static Endpoints: Tool Programs as an Interface for Flexible Agentic Web Services</strong> - Mugeng Liu, Shuoqi Li, Yixuan Zhang, Yun Ma - [[pdf]](https://arxiv.org/pdf/2606.19992)</summary>

**Abstract:** In the agentic web era, LLM-based agents increasingly invoke web services as tools, yet most interfaces remain \emph{static endpoints} that poorly express long-horizon workflows with loops, conditionals, joins, and retries. We present ToolPro, which represents an agent's tool intent as an \emph{executable tool program} that compactly encodes multi-step service interactions with explicit effect types. ToolPro combines constraint-guided program construction, effect-aware replay for exactly-once state-modifying calls, and a profile-driven policy that decides when program execution outperforms stepwise calling. We instantiate ToolPro over MCP-style services with WebAssembly sandboxing and evaluate it on diverse workflows of real-world applications. ToolPro reduces end-to-end latency by up to 53.4\% and client-side traffic by up to 96.1\%, with larger gains under higher network latency and workflow complexity.

**arXiv ID:** 2606.19992
</details>

<details>
<summary><strong>AI Economist Agent: An Agentic Framework for Model-Grounded Economic Analysis with RAG, Knowledge Graphs, and Large Language Models</strong> - Masahiro Kato - [[pdf]](https://arxiv.org/pdf/2606.20041)</summary>

**Abstract:** We propose a model-grounded RAG-based AI economist with an agentic framework for economic scenario analysis using large language models (LLMs) and knowledge graphs. While LLMs can generate fluent economic narratives, economists are often required to make economic claims grounded by economic theory and real-world data. Based on this motivation, this study proposes an RAG-based AI economist, which utilizes knowledge graphs including economic data and theory and LLM-based agents to plan the analysis, retrieve relevant evidence, select appropriate models, and generate reports. In our framework, we do not produce quantitative claims directly with the language model alone; instead, we generate narratives grounded in explicit model-based computations and linked to the retrieved evidence via AI agents. We refer to our framework as an AI economist agent. We evaluate the AI economist agent in two applications: economist report generation for U.S. inflation persistence and Federal Reserve policy, and bank stress-test narrative generation for U.S. commercial real estate refinancing stress. The results illustrate how grounding the generated reports improves their economic coherence and traceability.

**arXiv ID:** 2606.20041
</details>

<details>
<summary><strong>Learning What to Remember: Observability-Safe Memory Retention via Constrained Optimization for Long-Horizon Language Agents</strong> - Qingcan Kang, Liu Mingyang, Shixiong Kai, Kaichao Liang, Tao Zhong, Mingxuan Yuan - [[pdf]](https://arxiv.org/pdf/2606.10616)</summary>

**Abstract:** Long-horizon language agents accumulate observations, reasoning traces, and retrieved facts exceeding context windows, making memory retention a fundamental resource-allocation problem. Existing systems treat retention as local and do not model long-term consequences under observability constraints. To fill this gap, we formulate memory retention as a constrained stochastic optimization with budget feasibility, evidence utility, and delayed costs including miss, reacquisition, and stale penalties. We show this multi-step problem is NP-hard, making exact solution intractable. Moreover, deployment decisions must be made under partial observability. To address these challenges, we propose OSL-MR (Observability-Safe Learning for Memory Retention), a learning-augmented framework that enforces a strict separation between online-observable features and offline-available supervision. OSL-MR combines an evidence learner trained from realized evidence with a Mixed-Score heuristic that serves as a deployable online-safe baseline and an inductive prior. The policy learns query-conditioned evidence from interaction data and remains deployable under the same constraints. Experiments on LoCoMo and LongMemEval show OSL-MR outperforms recency-based, Generative Agents-style, and other heuristic baselines, especially under tight budgets. The Mixed-Score prior improves precision and recall, and sensitivity analysis shows robustness across cost settings. On small solvable instances, single-step optimization is insufficient to anticipate future demand shifts, while OSL-MR stays significantly closer to the dynamic-programming optimum, confirming the necessity of the sequential formulation and reinforcing our learning-guided approximation. These results establish constrained stochastic optimization and optimization-guided learning as a principled foundation for memory management in long-horizon agents.

**arXiv ID:** 2606.10616
</details>

<details>
<summary><strong>Agentic Symbolic Search: Characterizing PDEs Beyond Hand-crafted Expressions, Meshes, and Neural Networks</strong> - Zongmin Yu, Liu Yang - [[pdf]](https://arxiv.org/pdf/2606.20467)</summary>

**Abstract:** Mathematicians understand a PDE solution through mathematical structures rather than tables of computed values. Historically, this has been the product of mathematical analysis, carried out by hand for each problem individually. Neither numerical simulation nor neural networks produce those structures directly. We propose Agentic Symbolic Search (ASYS), a prior-guided framework in which an agent translates PDE theory, public problem constraints, and accumulated search experience into testable differentiable symbolic programs. The mathematical forms are refined under evolutionary search, while their continuous parameters are fit by gradient-based optimization. This makes the search an automated form of inductive-bias injection rather than blind symbolic regression. For problems with known analytical forms, ASYS recovers these forms naturally; for other problems, ASYS constructs analytical approximations which can guide mathematicians toward further analysis. In our experiments, across five problems spanning bounded dynamics, finite-time blow-up, and free-boundary focusing, ASYS produces interpretable representations, including a geometric interface formula for Allen-Cahn 2D dynamics and a nine-parameter contraction law for Keller-Segel chemotactic blow-up, in settings where no closed-form description was previously available. ASYS shows the possibility of a new paradigm for characterizing PDE solutions, beyond handcrafted analytical solutions, mesh-based numerical solutions, and neural network approximations.

**arXiv ID:** 2606.20467
</details>

<details>
<summary><strong>A Differentiable Composite Approximation Framework for Autonomous Underwater Vehicle Maneuvering Modeling from Sea-Trial Data</strong> - Aobo Wang, Aifei Xia, Zihao Wang, Lizhu Hao - [[pdf]](https://arxiv.org/pdf/2606.19711)</summary>

**Abstract:** Field-based modeling from onboard measurements can produce autonomous underwater vehicle (AUV) maneuvering models that reflect real operating characteristics. From an approximation perspective, conventional maneuvering models use predefined constraint polynomial bases, whereas data-driven models use data-adaptive bases. Motivated by this basis-function view, this paper presents a differentiable composite-approximation formulation, in which the polynomial-basis component and the data-adaptive basis component are treated as differentiable parts of a single predictor and calibrated jointly. A gradient-based co-calibration method is developed for full-scale AUV maneuvering prediction, where a sensitivity-aware mechanism regulates bounded polynomial updates while the neural residual captures remaining nonlinear discrepancies under a shared prediction objective. To account for ocean-current effects in field data, a turning-motion-based current estimation and compensation procedure is incorporated to construct current-compensated learning targets for training and rollout. The framework is evaluated using sea-trial data collected from a 7-meter AUV under multiple maneuvering conditions. Results show that the proposed method improves recursive trajectory and velocity prediction compared with polynomial-only, neural-only, and frozen-prior hybrid baselines, demonstrating its applicability to field-data-based AUV maneuvering modeling.

**arXiv ID:** 2606.19711
</details>

<details>
<summary><strong>One-to-Two Acting: A Novel Framework for Single-arm Agent Action Expansion to Dual Arms</strong> - Youbin Yao, Nieqin Cao, Mingyan Li, Yan Ding, Fuqiang Gu, Chao Chen - [[pdf]](https://arxiv.org/pdf/2606.19897)</summary>

**Abstract:** Dual-arm manipulation can improve throughput via parallel execution, but collecting bimanual demonstrations for training is costly and difficult. We present ExS2D, a hierarchical action expansion framework that enables dual-arm manipulation from single-arm supervision. ExS2D first generates structured subtasks from textual instructions while explicitly capturing temporal precedence. It then grounds each subtask into executable actions through subtask-guided action mapping in observation. Finally, precedence-aware action allocation and synchronized planning are performed by a multimodal large language model driven coordinator to select collision-free dual-arm executions. Simulation experiments demonstrate that ExS2D reduces the average execution steps by 54.4% while maintaining a comparable success rate to a single-arm baseline. Real-robot experiments on four tasks further demonstrate the reliability of ExS2D for dual-arm execution under few-shot single-arm samples, while using zero bimanual demonstrations.

**arXiv ID:** 2606.19897
</details>

<details>
<summary><strong>Autonomous Driving with Priority-Ordered STL Specifications Under Multimodal Uncertainty</strong> - Taha Bouzid, Shuhao Qi, Mircea Lazar, Sofie Haesaert - [[pdf]](https://arxiv.org/pdf/2606.20336)</summary>

**Abstract:** Autonomous vehicles must plan trajectories that satisfy a multitude of requirements on safety, passenger comfort, and compliance with traffic rules. However, in safety-critical scenarios, it is not always possible to satisfy all requirements simultaneously, necessitating their prioritization based on importance. At the same time, in these safety-critical scenarios, the uncertainty in trajectory predictions of the surrounding traffic, such as other vehicles and pedestrians, should be explicitly accounted for. In this work, we propose an uncertainty-aware trajectory planning framework that incorporates a predefined lexicographic ordering over Signal Temporal Logic (STL) specifications that stays valid under uncertainty. We implement this formulation with Model Predictive Path Integral (MPPI) control and we demonstrate the effectiveness of our method on simulation scenarios, showing that our framework efficiently handles conflicting objectives under realistic multi-modal uncertainty.

**arXiv ID:** 2606.20336
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (30 papers)</h2></summary>

<details>
<summary><strong>Reward as An Agent for Embodied World Models</strong> - Pu Li, Zhigang Lin, Qiang Wu, Yongxuan Lv, Fei Wang, Shan You - [[pdf]](https://arxiv.org/pdf/2606.19990)</summary>

**Abstract:** While RL has become a promising tool for refining world models, existing methods largely rely on conservative rollouts near the training distribution, limiting exploration, behavioral diversity, and richer dynamic discovery. In this work, we challenge this conservative paradigm. We argue that the core limitation is not exploration itself, but the lack of reliable verification strategies to support broader exploration. Without reliable verification, expanded exploration becomes highly susceptible to reward hacking, where policies exploit imperfect rewards without achieving genuine improvement. To evaluate this motivation, we instantiate our method in embodied world models, where physical plausibility, and task completion provide a rigorous testbed for scalable RL under complex dynamics. On the verification side, we introduce Reward as an Agent, an agentic reward framework that actively evaluates generated behaviors to provide robust reward signals and mitigate reward hacking under distribution shifts. On the exploration side, we introduce Dynamic-Aware Rollout Diversification through DynDiff-GRPO, which explicitly expands action-space exploration to diversify trajectories, broaden state-action coverage, and encourage richer embodied behaviors beyond conservative rollout regimes. By unifying Reward as an Agent with DynDiff-GRPO, we enable RL on a more reliable reward foundation with substantially diversified sampling, effectively mitigating reward hacking while yielding significant accuracy gains across multiple open-source world models, thereby demonstrating that broader exploration can scale successfully when grounded in robust verification.

**arXiv ID:** 2606.19990
</details>

<details>
<summary><strong>Process-Verified Reinforcement Learning for Theorem Proving via Lean</strong> - Minsu Kim, Se-Young Yun - [[pdf]](https://arxiv.org/pdf/2606.20068)</summary>

**Abstract:** While reinforcement learning from verifiable rewards (RLVR) typically has relied on a single binary verification signal, symbolic proof assistants in formal reasoning offer rich, fine-grained structured feedback. This gap between structured processes and unstructured rewards highlights the importance of feedback that is both dense and sound. In this work, we demonstrate that the Lean proof assistant itself can serve as a symbolic process oracle, supplying both outcome-level and fine-grained tactic-level verified feedback during training. Proof attempts are parsed into tactic sequences, and Lean's elaboration marks both locally sound steps and the earliest failing step, yielding dense, verifier-grounded credit signals rooted in type theory. We incorporate these structured rewards into a GRPO-style reinforcement learning objective with first-error propagation and first-token credit methods that balances outcome- and process-level advantages. Experiments with STP-Lean and DeepSeek-Prover-V1.5 show that tactic-level supervision outperforms outcome-only baselines in most settings, delivering improvements on benchmarks such as MiniF2F and ProofNet. Beyond empirical gains, our study highlights a broader perspective: symbolic proof assistants are not only verifiers at evaluation time, but can also act as process-level reward oracles during training. This opens a path toward reinforcement learning frameworks that combine the scalability of language models with the reliability of symbolic verification for formal reasoning.

**arXiv ID:** 2606.20068
</details>

<details>
<summary><strong>Augmenting Game AI with Deep Reinforcement Learning</strong> - Alessandro Sestini, Joakim Bergdahl, Amir Baghi, Jean-Philippe Barrette-LaPierre, Florian Fuchs, Linus Gisslén - [[pdf]](https://arxiv.org/pdf/2606.20210)</summary>

**Abstract:** Immersion in video games depends not only on graphics, audio, and game mechanics, but also on the quality of in-game characters. Producing believable characters, or game AI, remains a significant challenge as behavioral complexity is hard to capture with hand-coded systems. Game AI is a source of immersion and engagement; however, the limitations stemming from the challenges of creating game AI often lead to frustration and the breaking of the illusion of realism within the game. The introduction of machine learning models opens the door to creating more believable, authentic, and relatable characters in games. The promise is that they either learn from interacting with the game, or from player data, to develop true human-like behavior. In this paper, we envision more applications of reinforcement learning for game AI in the future. For this to materialize, current research limitations are prohibitive to broad deployment across game genres. Therefore, we propose a framework for training reinforcement learning models with a set of requirements in mind that are suited towards game AI and game development. We present examples of games with reinforcement learning-augmented game AI and describe the practicalities of deploying player-facing machine learning agents in modern games. Furthermore, we identify bottlenecks and hard problems in these areas, which we believe offer promising research directions to accelerate the adoption of machine learning in game AI for the video game industry.

**arXiv ID:** 2606.20210
</details>

<details>
<summary><strong>Leveraging systems' non-linearity to tackle the scarcity of data in the design of Intelligent Fault Diagnosis Systems</strong> - Giancarlo Santamato, Andrea Mattia Garavagno, Massimiliano Solazzi, Antonio Frisoli - [[pdf]](https://arxiv.org/pdf/2606.20323)</summary>

**Abstract:** Deep Transfer Learning (DTL) allows for the efficient building of Intelligent Fault Diagnosis Systems (IFDS). On the other hand, DTL methods still heavily rely on large amounts of labelled data. Obtaining such an amount of data can be challenging when dealing with machines or structures faults. This document proposes a novel approach to the design of vibration-based IFDS using DTL in condition of strong data scarcity. A periodic multi-excitation level procedure leveraging intrinsic non-linearities of real-world systems is used to produce images that can be conveniently analysed by pre-trained Convolutional Neural Networks (CNNs) to diagnose faults. A new data visualization method and its augmentation technique are proposed in this paper to tackle the typical lack of data encountered during the design of IFDS. Experimental validation on a railway pantograph structure provides effective support for the proposed method.

**arXiv ID:** 2606.20323
</details>

<details>
<summary><strong>Human-AI Agent Interaction in a Business Context</strong> - Kathrin Paimann, Elizangela Valarini, Sebastian Juhl - [[pdf]](https://arxiv.org/pdf/2606.18716)</summary>

**Abstract:** As AI agents are increasingly integrated into core business processes, understanding and designing effective interaction patterns between humans and AI agents becomes crucial for value creation. This study identifies and evaluates principles and criteria for a positive User Experience (UX) with AI agents, along with methods for its measurement. We identify user expectations and needs to facilitate adoption, build trust, and support user-centered decision-making by development teams. Using a mixed-methods approach that combines qualitative and quantitative techniques, we explore interaction patterns between humans and AI agents. The findings from this exploratory research serve as the basis to develop a survey experiment which evaluates the effectiveness of specific design elements on a larger scale. This foundational research contributes to the development of more intuitive and effective human-AI agent interactions in business settings.

**arXiv ID:** 2606.18716
</details>

<details>
<summary><strong>Physical Atari: A Robust and Accessible Platform for Real-time Reinforcement Learning on Robots</strong> - Khurram Javed, Joseph Modayil, Gloria Kennickell, Richard S. Sutton, John Carmack - [[pdf]](https://arxiv.org/pdf/2606.19357)</summary>

**Abstract:** We built a robot called the Robotroller that actuates an Atari CX40+ controller and a device called the Atari Devbox that renders the game frame and the reward signal from the Arcade Learning Environment on a screen. The Robotroller and the Atari Devbox, together with an off-the-shelf camera and a desktop computer, constitute a system that can be used to study reinforcement learning algorithms in the physical world. We call the full system Physical Atari. In this paper, we detail the key decisions that make Physical Atari a robust and accessible platform. To make the system robust, we designed the Robotroller so that all movement is done through bearings, which reduces wear. Additionally, we wrote software that monitors the state of the servos at a high frequency and intervenes to limit stress. To make the system accessible, we used affordable off-the-shelf components and parts that can be manufactured using consumer 3D printers. Physical Atari can be built for under $1,000 and has been used for weeks of non-stop reinforcement learning experiments without any mechanical failures. We used it to validate that reinforcement learning algorithms can learn directly on robots and show that even small distribution shifts between learning and deployment can significantly degrade the performance of policies. Our results underscore the importance of on-device adaptation for strong performance on robots.

**arXiv ID:** 2606.19357
</details>

<details>
<summary><strong>VOiLA: Vectorized Online Planning with Learned Diffusion Model for POMDP Agents</strong> - Marcus Hoerger, Rishikesh Joshi, Rahul Shome, Ian Manchester, Hanna Kurniawati - [[pdf]](https://arxiv.org/pdf/2606.19729)</summary>

**Abstract:** Planning under uncertainty is an essential capability for autonomous robots. The Partially Observable Markov Decision Process (POMDP) provides a powerful framework for such a capability. Although POMDP-based planning has advanced significantly, its application to real-world problems is often limited by the difficulty of obtaining faithful POMDP models. We present Vectorized Online planning wIth Learned diffusion model for POMDP Agents (VOiLA), a framework that learns task-agnostic POMDP models for online planning under uncertainty. VOiLA learns transition and observation samplers using conditional diffusion models and learns observation-likelihood models for particle-based belief updates. To enable efficient online planning, the diffusion samplers are distilled into compact feedforward generators and integrated with Vectorized Online POMDP Planner (VOPP), an online POMDP planner designed to leverage GPU parallelization. Experimental results indicate the distillation strategy reduces sampling cost by up to nearly three orders of magnitude, making learned generative POMDP models practical for online planning. Evaluation of VOiLA on three benchmark problems indicate that VOiLA achieves equal or better performance than Recurrent Soft Actor Critic while using less than 10% training data, and generalizes much better to unseen environment configurations. Physical robot evaluation indicates VOiLA uses the models learned using only simulated data and generates a policy that successfully accomplish the task in 10 of 10 runs.

**arXiv ID:** 2606.19729
</details>

<details>
<summary><strong>Measuring Biological Capabilities and Risks of AI Agents</strong> - Patricia Paskov, Jeffrey Lee, Kyle Brady, Alyssa Worland - [[pdf]](https://arxiv.org/pdf/2606.19899)</summary>

**Abstract:** This paper addresses a rapidly emerging policy challenge: how to generate and interpret credible evidence about the biological capabilities and risks of AI scientists, or agentic AI systems capable of autonomously or collaboratively performing multi-step scientific tasks. As these systems enter real research workflows, decision-makers increasingly face evaluation results whose meaning depends on underlying design choices that are often implicit or under-documented. We synthesize current evidence on AI-enabled biological risks and introduce biological agentic evaluations as a promising, but interpretation-sensitive, tool for assessing these systems. Our central contribution is a set of practical, experience-grounded considerations -- drawing from our own evaluations -- that show how choices around defining, designing, running, scoring, and documenting evaluations materially shape what results do and do not imply about risk. The analysis is intended to help policymakers interpret biological evaluation outputs with appropriate caution; guide public and private funders toward high-leverage investments in AI-biology evaluation research; and support biosecurity practitioners assessing emerging AI systems. A secondary audience includes researchers designing or conducting agentic evaluations within frontier AI labs, AI providers, scientific institutions, and third-party evaluation organizations.

**arXiv ID:** 2606.19899
</details>

<details>
<summary><strong>Connect the Dots: Training LLMs for Long-Lifecycle Agents with Cross-Domain Generalization Via Reinforcement Learning</strong> - Yanxi Chen, Weijie Shi, Yuexiang Xie, Boyi Hu, Yaliang Li, Bolin Ding, Jingren Zhou - [[pdf]](https://arxiv.org/pdf/2606.20002)</summary>

**Abstract:** This work presents a general framework for training large language models (LLMs) to "Connect the Dots" (CoD), a meta-capability required by long-lifecycle agents: as an LLM-based AI agent gets deployed in an environment, it solves a long sequence of tasks while continuously exploring the environment, learning from its own experiences, and iteratively self-updating its context about the environment, thereby achieving progressively better performance on future tasks conditioned on the updated context. Major components of the CoD framework include: (1) algorithm design and infrastructure for end-to-end reinforcement learning (RL) with long rollout sequences interleaving solve-task and update-context episodes; (2) tasks and environments for incentivizing and eliciting the targeted meta-capability in LLMs during training, as well as for faithfully measuring progress during evaluation. We present proof-of-concept implementations of the CoD framework, including a GRPO-style RL algorithm with fine-grained credit assignment, as well as tasks and environments tailored to the targeted meta-capability (rather than domain-specific LLM capabilities or standard task-by-task RL). Empirical results validate the efficacy of end-to-end RL training in the CoD setting, and demonstrate the potential for out-of-distribution generalization -- within the training domains, across different domains, and from CoD to Ralph-loop settings -- of the elicited meta-capability. Our investigation of CoD connects several lines of prior works, and opens up new opportunities for advancing LLMs and AI agents. To facilitate further research and applications, we release our implementations at \url{this https URL}.

**arXiv ID:** 2606.20002
</details>

<details>
<summary><strong>A Neuromorphic Reinforcement Learning Framework for Efficient Pathfinding in Robotic Mobile Fulfillment Systems</strong> - Junzhe Xu, Zecui Zeng, Lusong Li, Yuetong Fang, Renjing Xu - [[pdf]](https://arxiv.org/pdf/2606.20031)</summary>

**Abstract:** Dynamic environmental changes, confined workspaces, and stringent real-time constraints make pathfinding in Robotic Mobile Fulfillment Systems (RMFS) a challenging problem for conventional search- and rule-based methods, which typically suffer from high computational complexity and long decision latency. While reinforcement learning (RL) has emerged as a powerful alternative, deploying learned policies with extreme energy efficiency on resource-constrained hardware remains an open challenge. We present SDQN-RMFS, an end-to-end framework that achieves high-fidelity deployment of an RL-trained policy from a full-precision artificial neural network (ANN) through to a neuromorphic chip. By computing only when triggered by sparse events, this framework unlocks ultra-low-power RMFS pathfinding. Our full-stack pipeline operates as follows: an ANN policy is first efficiently trained via a collision-allowing strategy to densify informative trajectories, and then converted into a spiking neural network (SNN) via a hard-label knowledge distillation approach. This effectively addresses the output distribution mismatch, preserving policy capability across the ANN-to-SNN pipeline while substantially reducing inference latency. Hardware experiments demonstrate up to 11,281$\times$ energy savings and a nearly two-fold reduction in latency compared to a high-performance GPU baseline, while maintaining decision quality on par with the original trained policy. These results establish physical neuromorphic inference as a practical and energy-sustainable pathway for large-scale RMFS operations.

**arXiv ID:** 2606.20031
</details>

<details>
<summary><strong>CRAX: Fast Safe Reinforcement Learning Benchmarking</strong> - Tristan Tomilin, Mourad Boustani, Mickey Beurskens, Thiago D. Simão - [[pdf]](https://arxiv.org/pdf/2606.20376)</summary>

**Abstract:** Safety is a core concern for deploying reinforcement learning (RL) agents in real-world domains such as robotics and autonomous driving. While benchmarks have been central to progress in RL, existing safety benchmarks with high-fidelity 3D physics remain computationally slow, limiting large-scale experimentation and rapid prototyping. To address this gap, we propose CRAX (Constrained RL Accelerated with JAX). Built on top of the MuJoCo XLA (MJX) physics engine with realistic 3D dynamics, CRAX leverages vectorized operations and hardware acceleration, yielding up to ~100x speedups over comparable CPU-based safety benchmarks. The benchmark features six environment suites and three agent-specific tasks, each spanning three difficulty levels. Evaluating six popular safe RL methods shows that no single approach dominates across all tasks, and reveals the trade-offs between performance and safety. We find that curriculum learning across difficulty levels and safety transfer can improve performance over direct training in harder settings.

**arXiv ID:** 2606.20376
</details>

<details>
<summary><strong>UltraQuant: 4-bit KV Caching for Context-Heavy Agents</strong> - Inesh Chakrabarti, David Limpus, Aditi Ghai Rana, Bowen Bao, Spandan Tiwari, Thiago Crepaldi, Ashish Sirasao - [[pdf]](https://arxiv.org/pdf/2606.20474)</summary>

**Abstract:** Context-heavy agents place unusual pressure on the key-value (KV) cache: long prefixes are reused across many short turns, while concurrency determines whether the serving system can keep GPUs utilized. We study 4-bit KV-cache compression for this setting, using TurboQuant-style rotation and codebook quantization as a quality anchor and vLLM FP8 KV caching as the deployment anchor. We report three contributions. First, we frame 4-bit KV caching around multi-round agent workloads where task quality, cache residency, and serving throughput must be measured jointly. Second, we describe the practical design choices needed to make the 4-bit path robust, including asymmetric K/V treatment, Walsh-Hadamard rotation, QJL removal, and block-scale variants. Third, we present serving optimizations on AMD GPUs, including optimized decode-attention kernels and UltraQuant, an FP4 approximation path that uses FP8 queries, FP4 KV tensors, UE8M0 group scales, and native scaled-MFMA support on CDNA4. On a long-context, multi-turn agentic workload, UltraQuant cuts P50 time-to-first-token by 3.47x in the cache-pressured late rounds (2.3x across all rounds) and raises output throughput by 1.63x over the FP8 KV baseline.

**arXiv ID:** 2606.20474
</details>

<details>
<summary><strong>ScaleWoB: Guiding GUI Agents with Coding Agents via Large-Scale Environmental Synthesis</strong> - Guohong Liu, Jialei Ye, Pengzhi Gao, Wei Liu, Jian Luan, Yunxin Liu, Yuanchun Li - [[pdf]](https://arxiv.org/pdf/2605.25160)</summary>

**Abstract:** GUI agents powered by large language models are advancing rapidly, creating urgent needs for evaluation and training based on realistic environments. However, directly doing so in real-world environments introduces some challenges that cannot be overlooked. Real-world environments are complex and uncontrollable, making it difficult to construct verifiable rewards and to save or reset states. Existing works prioritize reproducibility but are often limited to open-source apps or file-operation tasks for reliable reward building, leaving a persistent gap from real-world usage. Furthermore, relying on virtual machines or docker images demand high resource requirements and suffer from slow response speeds, which limit the efficiency. We present \sys, a framework that could produce high-fidelity synthesized interactive environments for GUI agents across platforms with verifiable rewards. These environments behave as backend-free webpages accessible via URL, requiring near-zero setup and low resource cost, making the approach suitable for both large-scale evaluation and downstream agent training. We support multiple GUI platforms including mobile, desktop, and automotive/in-vehicle interfaces based on the same pipeline, covering 100+ environments and 1000+ verifiable tasks. Among them, 120 challenging tasks across 63 simulated mobile applications are released as a fully synthesized mobile GUI agent benchmark. Experiment results on five state-of-the-art mobile GUI agents reveal substantial headroom -- the average success rate is only 27.92\%, dropping to 17.82\% on long-horizon subset -- while humans reach 92.08\%. A comparison against real-world sample tasks shows that assessments made in our synthetic environments generalize to real apps. The project website is at this https URL.

**arXiv ID:** 2605.25160
</details>

<details>
<summary><strong>FundaPod: A Multi-Persona Agent Pod Platform with Knowledge Graph Memory for AI-Assisted Fundamental Investment Research</strong> - Di Zhu, Lei Nico Zheng, Zihan Chen - [[pdf]](https://arxiv.org/pdf/2605.27864)</summary>

**Abstract:** Large language models (LLMs) are increasingly applied in finance, yet most existing work emphasizes trading signals or financial NLP tasks centered on prediction. Institutional fundamental research, by contrast, requires human analysts or AI agents to gather evidence, identify business drivers, compare competing viewpoints, and generate investment memos. Its broader goal is not merely to predict outcomes, but to produce investment plans that are transparent, reusable, and verifiable, while contributing to the cumulative development of investment knowledge. We present FundaPod, a multi-persona agent platform for AI-assisted fundamental investment research. We argue that fundamental research is a human-centric decision-support task that is qualitatively distinct from trading-signal generation, and is therefore better served by an independence-preserving architecture. In FundaPod, AI agents with different personas, such as value investors or macro strategists, conduct research independently under a shared provenance contract. Their disagreements are then surfaced post hoc for adjudication by the human portfolio manager (PM) through a knowledge-graph memory system. This paper contributes five design principles for human-AI hybrid systems supporting fundamental research, grounded in design-science practice and theories of cognitive isolation and human-machine coordination. It also describes four architectural mechanisms: a persona distillation pipeline that turns public investor materials into deployable agents; a declarative skill registry that lets the planner derive typed task graphs; a grounded evidence model that links memo claims to verifiable sources; and a knowledge-graph "second brain" that connects tickers, memos, analysts, and themes. We demonstrate the architecture through a complete case study and a persona-based memo comparison.

**arXiv ID:** 2605.27864
</details>

<details>
<summary><strong>VitalAgent: A Tool-Augmented Agent for Reactive and Proactive Physiological Monitoring over Wearable Health Data</strong> - Di Zhu, Yu Yvonne Wu, Hong Jia, Aaqib Saeed, Vassilis Kostakos, Ting Dang - [[pdf]](https://arxiv.org/pdf/2605.29483)</summary>

**Abstract:** Wearable devices enable continuous monitoring of physiological signals such as ECG and PPG, but existing mHealth systems are largely limited to task-specific prediction pipelines or reactive question answering over static summaries. They lack the ability to support temporal reasoning, persistent physiological context, and proactive monitoring over long-term signal streams. We propose VitalAgent, a tool-augmented agentic framework for ECG/PPG-based mHealth that supports both reactive question answering and proactive monitoring. VitalAgent is built on a longitudinal physiological memory and a tool-augmented reasoning interface that enables dynamic computation over raw signals. We further introduce VitalBench, a longitudinal physiological monitoring benchmark dataset comprising 1,862 QA pairs for reactive question answering and 90.2 hours of continuous ECG/PPG recordings for proactive monitoring, covering cardiac, physical activity, and stress-related tasks. Experiments demonstrate that VitalAgent achieves over 25% improvement over prompt-based and ReAct baselines in reactive evaluation and supports proactive alert monitoring over long-term physiological signals, highlighting the importance of dynamic tool use and long-term physiological monitoring.

**arXiv ID:** 2605.29483
</details>

<details>
<summary><strong>The Almost Intelligent Revolution: Options for Scaling Up Deliberation and Empowering People with AI</strong> - Serge Sharoff - [[pdf]](https://arxiv.org/pdf/2606.19864)</summary>

**Abstract:** The increasing prominence of Large Language Models (LLMs) in public discourse presents both opportunities and challenges for democratic deliberation. While red teaming strategies help mitigate specific risks, broader concerns persist regarding linguistic constraints, biases, and the sycophantic tendencies of LLMs. This chapter explores how LLMs can be used to significantly scale up and democratise deliberation, particularly in fostering inclusivity and empowering traditionally marginalised groups. Drawing on concepts from Systemic-Functional Linguistics, the chapter examines how variations across language users (for example, with respect to socio-demographic groups) and across language use (for example, with respect to communicative functions) shape participation in AI-supported deliberation. The chapter presents AI-driven deliberation studies and assesses their potential to scaffold argumentation, enhance access, and reduce the influence of exclusionary linguistic norms and biases which are embedded in prestigious registers. At the same time, the chapter cautions against both overclaiming, which leads to unrealistic expectations, and underclaiming, which risks missed opportunities for AI-assisted engagement. The chapter concludes by identifying future research directions to maximise the democratic potential of AI-assisted participation while embedding ethical safeguards to counteract the reproduction of linguistic inequalities.

**arXiv ID:** 2606.19864
</details>

<details>
<summary><strong>Beyond Global Replanning: Hierarchical Recovery for Cross-Device Agent Systems</strong> - Shu Yao, Yuhua Luo, Qian Long, Jingru Fan, Zhuoyuan Yu, Yuheng Wang, Lin Wu, Yufan Dang, Huatao Li, Chen Qian - [[pdf]](https://arxiv.org/pdf/2606.20487)</summary>

**Abstract:** Real-world computer-use tasks often span multiple applications and devices, requiring agents to coordinate heterogeneous environments under dynamic runtime failures. Existing multi-device agent systems support task decomposition and cross-device assignment, but recovery remains largely coarse-grained: when execution fails, they typically retry the same strategy, reassign the subtask, or revise the global plan, without systematically modeling the device-local strategy space. This limits their ability to distinguish failures that can be repaired within the current device from those that require cross-device replanning. We propose \textbf{H-RePlan}, a hierarchical replanning framework for multi-device agents with unified API--CLI--GUI execution. H-RePlan equips each device with interchangeable execution strategies and separates device-local strategy recovery from orchestrator-level global replanning through a compact cross-layer failure abstraction. To evaluate this capability, we introduce \textbf{HeraBench}, a fault-injected benchmark that constructs cross-device workflows over Linux and Android devices and injects strategy- and device-level failures. Experiments show that H-RePlan substantially outperforms single-strategy and coarse-grained multi-device baselines, achieving higher completion, instruction adherence, and perfect-pass rates while reducing the token cost required for reliable end-to-end success. These results demonstrate that scope-aware hierarchical recovery is essential for robust multi-device agent execution.

**arXiv ID:** 2606.20487
</details>

<details>
<summary><strong>Beyond the GUI Paradigm: Do Mobile Agents Need the Phone Screen?</strong> - Li Gu, Zihuan Jiang, Linqiang Guo, Zhixiang Chi, Ziqiang Wang, Huan Liu, Yuanhao Yu, Tse-Hsun Chen, Yang Wang - [[pdf]](https://arxiv.org/pdf/2606.19388)</summary>

**Abstract:** Recent advances in mobile agents are dominated by the GUI paradigm, in which agents perceive UI information and emit screen interactions. However, mobile platforms also expose a command-line interface (CLI) that provides direct access to device services and data. We argue CLI deserves first-class consideration alongside GUI. We evaluate three coding agents (Claude Code, Terminus-2, mini-swe-agent) across four model APIs on AndroidWorld and MobileWorld without any mobile-specific post-training, comparing against three reproducible GUI baselines (GUI-Owl-1.5-32B, MAI-UI, Qwen3-VL-32B). Claude Code (Opus 4.7) reaches 71.8\% and 51.9\%, outperforming every reproducible GUI baseline (69.3/68.1/57.8\% on AndroidWorld; 43.2/26.3/13.3\% on MobileWorld), while every other CLI configuration remains competitive. To establish the paradigm's ceiling, we provide oracle CLI solutions that reach 88.8\% on AndroidWorld (103/116 tasks CLI-solvable) and 86.3\% on MobileWorld (101/117 tasks CLI-solvable), indicating substantial room for future improvement. To cover everyday user intents beyond the GUI scope, we introduce the \textbf{CLI-Advantage Task Suite}, comprising 45 templates across five categories: bulk operations, multi-condition filtering, aggregation, cross-app workflows, and hidden device state. Every CLI agent outperforms every GUI baseline in all five categories, with substantially fewer steps per task (10.7 vs.\ 18.6). To support future research on mobile CLI agents, we will open-source agent implementations, oracle solutions, the CLI-Advantage suite, and evaluation infrastructure.

**arXiv ID:** 2606.19388
</details>

<details>
<summary><strong>ShoppingBench: A Real-World Intent-Grounded Shopping Benchmark for LLM-based Agents</strong> - Jiangyuan Wang, Kejun Xiao, Qi Sun, Huaipeng Zhao, Tao Luo, Jian Dong Zhang, Xiaoyi Zeng - [[pdf]](https://arxiv.org/pdf/2508.04266)</summary>

**Abstract:** Existing benchmarks in e-commerce primarily focus on basic user intents, such as finding or purchasing products. However, real-world users often pursue more complex goals, such as applying vouchers, managing budgets, and finding multi-products seller. To bridge this gap, we propose ShoppingBench, a novel end-to-end shopping benchmark designed to encompass increasingly challenging levels of grounded intent. Specifically, we propose a scalable framework to simulate user instructions based on various intents derived from sampled real-world products. To facilitate consistent and reliable evaluations, we provide a large-scale shopping sandbox that serves as an interactive simulated environment, incorporating over 2.5 million real-world products. Experimental results demonstrate that even state-of-the-art language agents (such as GPT-4.1) achieve absolute success rates under 50% on our benchmark tasks, highlighting the significant challenges posed by our ShoppingBench. In addition, we propose a trajectory distillation strategy and leverage supervised fine-tuning, along with reinforcement learning on synthetic trajectories, to distill the capabilities of a large language agent into a smaller one. As a result, our trained agent achieves competitive performance compared to GPT-4.1.

**arXiv ID:** 2508.04266
</details>

<details>
<summary><strong>MENTOR: Reinforcement Learning via Flexible Teacher-Optimized Rewards for Tool-Use Distillation</strong> - ChangSu Choi, Hoyun Song, Dongyeon Kim, WooHyeon Jung, Minkyung Cho, Sunjin Park, NohHyeob Bae, Seona Yu, KyungTae Lim - [[pdf]](https://arxiv.org/pdf/2510.18383)</summary>

**Abstract:** Distilling the tool-use capabilities of large language models (LLMs) into small language models (SLMs) is essential for their practical application. The predominant approach, supervised fine-tuning (SFT), suffers from poor out-of-domain (OOD) generalization due to its rigid alignment with static teacher trajectories. While reinforcement learning (RL) offers an alternative, the capacity limitations of SLMs pose a severe dilemma: sparse outcome rewards provide insufficient guidance, whereas strict trajectory matching imposes overly restrictive constraints. To bridge this capacity-driven gap, we propose MENTOR, which introduces a flexible yet process-aware reward structure. Instead of enforcing rigid replication, MENTOR uses the teacher's reference to guide tool-use behavior, balancing behavioral alignment with downstream performance. Extensive experiments on controlled executable-tool benchmarks demonstrate that MENTOR improves OOD tool-use performance compared to SFT and strict RL baselines. Our findings suggest that within verifiable tool-use environments, flexible tool-use alignment offers a more effective approach than strict trajectory replication for developing adaptable small models.

**arXiv ID:** 2510.18383
</details>

<details>
<summary><strong>Quality Over Clicks: Iterative Reinforcement Learning for Early-Stage E-Commerce Query Suggestion</strong> - Qi Sun, Kejun Xiao, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng - [[pdf]](https://arxiv.org/pdf/2603.22922)</summary>

**Abstract:** Existing dialogue systems rely on query suggestion to enhance user engagement. Recent approaches mainly optimize generative models using click-through rate (CTR) models to align with user preferences. However, these methods are less effective in early-stage deployment scenarios, where click feedback is sparse and insufficient for training a reliable CTR model. To bridge this gap, we propose QualEQS, a quality-first iterative reinforcement learning framework for e-commerce query suggestion. We formalize actionable suggestion quality along three dimensions that directly affect downstream usability: answerability, factuality, and information gain. To continuously improve from online traffic without click supervision, we further propose group-level disagreement among candidate suggestions to identify ambiguous query contexts and mine hard training cases for iterative refinement. We also introduce EQS-Benchmark, a dataset of 16,949 real-world e-commerce queries for offline training and evaluation. Experiments show that our quality-based offline metrics correlate strongly with online performance, providing a practical evaluation recipe for sparse-feedback deployment. In both offline and online settings, QualEQS consistently outperforms strong baselines, yielding a 6.81% improvement in online ChatPV in a real-world enterprise-level conversational shopping assistant system.

**arXiv ID:** 2603.22922
</details>

<details>
<summary><strong>Insulin4RL: Real-Time Insulin Management in the Intensive Care Unit for Offline Reinforcement Learning</strong> - Thomas Frost, Steve Harris - [[pdf]](https://arxiv.org/pdf/2606.19481)</summary>

**Abstract:** Offline reinforcement learning (ORL) offers the potential to improve the quality of clinical decision-making using historical electronic health record (EHR) data. Current training and evaluative practices in this field rely heavily on EHR datasets that have been temporally discretised into fixed, regular time intervals. Discretisation creates fictional representations of complex clinical scenarios and compromises the generalisability of retrospective model evaluations. In this paper, we introduce Insulin4RL, a healthcare ORL dataset featuring naturally irregular inputs and actions from real clinical trajectories. Derived from MIMIC-IV, Insulin4RL comprises over 375,000 labelled decisions across 12,209 patients requiring insulin infusion titration in the Intensive Care Unit. The dataset can thus be used for research into ORL model performance under realistic clinical sampling assumptions. We provide a description of the dataset's structure and characteristics, baseline performance metrics using model-free offline reinforcement learning, and a standardised evaluation protocol using fitted Q-evaluation. We conclude with suggested areas for future research that could be addressed using this resource.

**arXiv ID:** 2606.19481
</details>

<details>
<summary><strong>Quantile of Means: A Bonus-Free Ensemble Method for Minimax Optimal Reinforcement Learning</strong> - Asaf Cassel, Aviv Rosenberg - [[pdf]](https://arxiv.org/pdf/2606.20107)</summary>

**Abstract:** Optimal Reinforcement Learning (RL) algorithms typically rely on carefully constructed count-based uncertainty estimates to drive exploration. Although theoretically sound, such estimates are hard to compute in practical settings and therefore offer limited insight for designing exploration heuristics. Meanwhile, ensembling has emerged as a practical approach, but remains without theoretical justification. Building on a recent ensemble-based method for Multi-Armed Bandits, we propose a quantile-based ensemble method for finite-horizon Markov Decision Processes (MDPs). Our simple count-free approach achieves optimal variance-dependent regret bounds, providing theoretical grounding for ensemble-based exploration in RL.

**arXiv ID:** 2606.20107
</details>

<details>
<summary><strong>Direct Advantage Estimation for Scalable and Sample-efficient Deep Reinforcement Learning</strong> - Hsiao-Ru Pan, Bernhard Schölkopf - [[pdf]](https://arxiv.org/pdf/2606.20411)</summary>

**Abstract:** Direct Advantage Estimation (DAE) has been shown to improve the sample efficiency of deep reinforcement learning algorithms. However, its reliance on full environment observability limits its applicability in realistic settings, and its requirement to model transition probabilities incurs substantial computational overhead for high-dimensional observations. In the present work, we address both limitations. First, we extend the theoretical framework of DAE to partially observable domains with minimal modifications. Second, we reduce its computational complexity by introducing discrete latent dynamics models that efficiently approximate transition probabilities. We evaluate our approach on the Arcade Learning Environment and find that DAE scales effectively with function approximator capacity while retaining high sample efficiency.

**arXiv ID:** 2606.20411
</details>

<details>
<summary><strong>A Model-Driven Approach for Developing Families of Reinforcement Learning Environments</strong> - Xiaoran Liu, Istvan David - [[pdf]](https://arxiv.org/pdf/2606.20324)</summary>

**Abstract:** Virtual training environments are software-intensive systems in which reinforcement learning (RL) agents learn, adapt, and demonstrate meaningful behavior. Virtual training environments offer a safe and cost-efficient alternative to training agents in real-world settings. However, to converge, most realistic RL problems require training in multiple, mostly similar but slightly different environments - i.e., families of environment variants. The typical development process of environment families is a labor-intensive and error-prone manual endeavor that does not scale well. To alleviate these issues, in this paper, we propose a model-driven approach for developing families of RL training environments. To obtain the family of environments, we develop an approach and prototype tool. In our approach, a hybrid genetic algorithm - a combination of population-based global search and heuristic local search - generates environment families. Mutations and constraints are expressed as model transformations and are operationalized into a search process by a state-of-the-art model transformation engine. We demonstrate the soundness of our approach in a wildfire mitigation scenario and curriculum learning - a particular learning paradigm that relies on environment families.

**arXiv ID:** 2606.20324
</details>

<details>
<summary><strong>StarOR: Synergizing Tree Search and Test-Time Reinforcement Learning for Optimization Modeling</strong> - Jiajun Li, Yu Ding, Shisi Guan, Ran Hou, Wanyuan Wang - [[pdf]](https://arxiv.org/pdf/2606.15197)</summary>

**Abstract:** Optimization modeling is inherently hierarchical, requiring a precise sequence of symbolic commitments. Traditional learning-based automated optimization modeling methods improve modeling policies through large-scale annotated or curated training data, but are costly to adapt to new problem distributions. Meanwhile, one-shot generation remains brittle in hierarchical modeling, where early symbolic errors can propagate into invalid formulations. Test-time scaling offers a promising alternative by enabling structural exploration with additional instance-level computation; however, existing search-based methods typically rely on a fixed policy, causing repeated rollouts to inherit similar modeling biases and providing limited credit assignment for intermediate decisions. To address these limitations, we propose StarOR, a synergistic search-and-adaptation framework that couples MCTS with Test-Time Reinforcement Learning for optimization modeling. StarOR decomposes the modeling process into four stages and updates a transient LoRA adapter via GRPO at each non-terminal node. By using MCTS-generated siblings as local comparison sets, StarOR transforms search-time exploration into instance-specific policy refinement. Moreover, an unsupervised multi-faceted reward system provides fine-grained feedback for intermediate formulation decisions without ground-truth labels. Experiments across five optimization benchmarks show that StarOR achieves state-of-the-art performance even with a 4B backbone, outperforming existing methods and the frontier LLMs.

**arXiv ID:** 2606.15197
</details>

<details>
<summary><strong>Fast Human Attention Prediction for Fixation-guided Active Perception in Autonomous Navigation</strong> - Fatma Youssef Mohammed, Grzegorz Malczyk, Kostas Alexis - [[pdf]](https://arxiv.org/pdf/2606.20491)</summary>

**Abstract:** Human visual attention relies on structured scanpaths to efficiently process scenes, yet instilling this behavior into robot autonomy is in its infancy and hindered by the high,computational costs of existing predictive models. To address this, we introduce GazeLNN, a computationally lightweight,scanpath prediction model that leverages Liquid Neural Networks as its recurrent engine and employs MobileNetV3 for feature extraction. Operating auto-regressively, the architecture predicts sequential fixation heatmaps conditioned on the current visual stimulus and fixation history. Despite requiring only 0.61 GFLOPs, GazeLNN achieves state-of-the-art performance on the MIT Low Resolution dataset achieving 0.47 ScanMatch score. It outperforms existing recurrent baselines across diverse evaluation metrics, while reducing computational costs by 99.40% and accelerating inference by up to six times. To investigate the role of human attention modeling in robot autonomy and demonstrate the practical utility of this highly efficient architecture, we integrate GazeLNN into an active camera-robot control policy trained via Reinforcement Learning. This integration enables human-fixation-guided perception during autonomous navigation, validated through successful real-world deployments on an aerial robot.

**arXiv ID:** 2606.20491
</details>

<details>
<summary><strong>AION: Aerial Indoor Object-Goal Navigation Using Dual-Policy Reinforcement Learning</strong> - Zichen Yan, Yuchen Hou, Shenao Wang, Yichao Gao, Rui Huang, Lin Zhao - [[pdf]](https://arxiv.org/pdf/2601.15614)</summary>

**Abstract:** Object-Goal Navigation (ObjectNav) requires an agent to autonomously explore an unknown environment and navigate toward target objects specified by a semantic label. While prior work has primarily studied zero-shot ObjectNav under 2D locomotion, extending it to aerial platforms with 3D locomotion capability remains underexplored. Aerial robots offer superior maneuverability and search efficiency, but they also introduce new challenges in spatial perception, dynamic control, and safety assurance. In this paper, we propose AION for vision-based aerial ObjectNav without relying on external localization or global maps. AION is an end-to-end dual-policy reinforcement learning (RL) framework that decouples exploration and goal-reaching behaviors into two specialized policies. We evaluate AION on the AI2-THOR benchmark and further assess its real-time performance in IsaacSim using high-fidelity drone models. Experimental results show that AION achieves superior performance across comprehensive evaluation metrics in exploration, navigation efficiency, and safety. The video can be found at \url{this https URL}, code and model checkpoints are available at \url{this https URL}.

**arXiv ID:** 2601.15614
</details>

<details>
<summary><strong>MobileForge: Annotation-Free Adaptation for Mobile GUI Agents with Hierarchical Feedback-Guided Policy Optimization</strong> - Guangyi Liu, Pengxiang Zhao, Gao Wu, Yiwen Yin, Mading Li, Liang Liu, Congxiao Liu, Zhang Qi, Mengyan Wang, Liang Guo, Yong Liu - [[pdf]](https://arxiv.org/pdf/2606.19930)</summary>

**Abstract:** MLLM-based mobile GUI agents have made substantial progress in UI understanding and action execution, but adapting them to real target apps remains costly because mobile apps are numerous, frequently updated, and hard to cover with human-written tasks, demonstrations, or reward labels. Existing annotation-free GUI learning reduces manual supervision, yet lacks a unified substrate connecting target-app exploration, curriculum mining, rollout execution, and feedback, while policy optimization often relies on isolated rollouts and coarse rewards that are hard to convert into reliable improvement signals. We present MobileForge, an annotation-free adaptation system for mobile GUI agents. MobileForge consists of MobileGym, which grounds task generation and rollout evaluation in real mobile app interaction, and Hierarchical Feedback-Guided Policy Optimization (HiFPO), which turns trajectory outcomes, step-level process feedback, and corrective hints into hint-contextualized step-level GRPO updates. Using only automatically generated annotation-free adaptation data, MobileForge adapts Qwen3-VL-8B to 67.2% Pass@3 on AndroidWorld, close to the closed-data GUI-specialized GUI-Owl-1.5-8B base model at 69.0%. The MobileForge-adapted ForgeOwl-8B further reaches 77.6% Pass@3 on AndroidWorld and 41.0% success on the out-of-domain MobileWorld GUI-only split, establishing the strongest open-data mobile GUI agent in our evaluation. Code, data, and trained models will be released at this https URL.

**arXiv ID:** 2606.19930
</details>

<details>
<summary><strong>Directors Duties in the Age of Agentic Artificial Intelligence</strong> - Deirdre Ahern - [[pdf]](https://arxiv.org/pdf/2606.20453)</summary>

**Abstract:** As boards engage with the adoption of Artificial Intelligence including agentic AI to drive operational efficiencies, this presents new opportunities for profit maximisation. AI adoption is increasingly identified with employee role displacement and in companies, and the interests of employees as stakeholders require exploration. A novel question posed is whether in an age of AI ascendancy AI may warrant being given stakeholder status as its role in the company approximates or eclipses that of human employees. The article probes four distinct models of corporate purpose within the duty on directors to act in the best interests of the company, the shareholder primacy model, the Enlightened Shareholder value model, the stakeholder friendly model, and the stakeholder value model, highlighting the available scope for directors to accommodate the interests of employees around AI adoption in decision-making by boards around AI. It is concluded that given the degree to which directors are insulated from legal scrutiny in relation to their best interests duty, adopting a wider law in context approach to promote employee welfare would serve the interests of employees, directors and companies alike. This would see directors engaging meaningfully with employees and providing opportunities for reskilling to adapt to the age of AI.

**arXiv ID:** 2606.20453
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
