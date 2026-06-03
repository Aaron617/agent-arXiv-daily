# Agent arXiv Daily

**Last Updated:** 2026-06-03 06:17:42

**Total Papers:** 16

## Table of Contents

- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Benchmarks and Datasets (2 papers)</h2></summary>

<details>
<summary><strong>Toward a Modular Architecture for Embedded AI Agent Systems at the Edge</strong> - Marcus Rüb, Michael Gerhards - [[pdf]](https://arxiv.org/pdf/2606.02862)</summary>

**Abstract:** The rise of Large Language Models (LLMs) has enabled agentic AI capable of complex reasoning and tool use; however, deploying such autonomy in pervasive computing environments remains challenging due to the strict memory and energy constraints of embedded microcontrollers. Existing frameworks typically assume server-class resources or continuous connectivity, leaving a gap for deeply embedded systems. This paper proposes a modular reference architecture for Embedded Agent Systems that bridges the divide between deterministic real-time control and agentic intelligence.
We introduce a tiered design that decouples On-Device Agents - executing highly compressed neural networks and rule-based logic for low-latency, privacy-critical tasks - from Cloud-Augmented Agents that leverage Small Language Models (SLMs) for higher-level reasoning and planning. A key contribution is the integration of a cross-cutting Governance Layer, ensuring observability, policy enforcement, and safety across distributed fleets of autonomous devices. Rather than presenting purely empirical benchmarks, we analyze architectural design principles and trade-offs regarding latency, energy, and reliable execution in resource-constrained environments.

**arXiv ID:** 2606.02862
</details>

<details>
<summary><strong>Solipsistic Superintelligence is Unlikely to be Cooperative</strong> - Rakshit S Trivedi, Natasha Jaques, Logan Cross, Alexander Sasha Vezhnevets, Joel Z Leibo - [[pdf]](https://arxiv.org/pdf/2606.03237)</summary>

**Abstract:** AI's central challenge is shifting from capability to coexistence. The dominant paradigm in AI research focuses on developing powerful agents that treat the world as an exogenous and stationary source of feedback. We contend that superintelligence, an extremely capable task solver, born out of such a solipsistic approach to AI design, is unlikely to be cooperative. Deploying AI systems induces endogenous non-stationarity, resulting in a train-test-deploy gap where historical distributions diverge from the deployment context. We refer to this as the self-undermining property of unilateral optimization. Closing this gap requires AI that participates in cooperation: the equilibrium-selection process through which multiple actors navigate their interdependence. We call for a non-solipsistic research paradigm that treats this interdependence as a core design principle rather than approaching cooperation as a task to solve. This entails building dynamic evaluation testbeds involving adaptive counterparties, treating institutions as design primitives, and preserving human agency as a structural feature of the systems we build.

**arXiv ID:** 2606.03237
</details>

</details>

<details open>
<summary><h2>LLM Agents (1 papers)</h2></summary>

<details>
<summary><strong>The Epi-LLM Framework: probing LLM behavioral priors through epidemiological agent-based models</strong> - Petra Ferenz, Ava Keeling, Tobias O'Keefe, Lorenzo Stigliano, Francesco Di Lauro, Andres Colubri, Jasmina Panovska-Griffiths - [[pdf]](https://arxiv.org/pdf/2606.02867)</summary>

**Abstract:** Human behaviour during epidemics affects infectious disease dynamics, but quantifying this remains deeply challenging. Here we introduce the Epi-LLM framework: a novel integration of agent-based modelling, real-life epigames, and large language models (LLMs) in which a synthetic society of agents reasons and adapts dynamically over an outbreak contact network. Comparing synthetic agent behaviour against a no-intervention SEIR baseline and human participant data from the AUIB epigame study, we find that LLM agents across four different architectures reduced peak active infections, with quarantine compliance peaking at 58-65% on day six of the 15-day simulation. A binomial generalised linear model showed that perceived health severity was the strongest predictor of quarantine behaviour ($\beta = 0.33, p = 0.002$), yielding a pseudo-$R^2$ of 0.055, comparable to the 0.072 observed in the human trial. LLM architecture is a key determinant of epidemic dynamics: low-variance architectures offer greater internal validity for testing behavioural rules, while high-variance models may better represent real-world decision-making. Geographic labels alone do not induce culturally differentiated behaviour; explicit attitudinal parameterisation is required. This proof-of-principle work lays the groundwork for deploying the Epi-LLM framework as a scalable, risk-free simulation environment for pandemic preparedness research.

**arXiv ID:** 2606.02867
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (9 papers)</h2></summary>

<details>
<summary><strong>MeDxAgent: Multi-Agent Consultation for Interactive Medical Diagnosis</strong> - Akshat Sanghvi, Naren Akash, Raza Imam, Amit Sharma, Mohit Jain - [[pdf]](https://arxiv.org/pdf/2606.03416)</summary>

**Abstract:** Large language models (LLMs) are increasingly used for health-related decision support. Yet most evaluations treat diagnosis as a single-shot task with complete information provided upfront, often as a multiple-choice selection. This diverges from clinical practice, where diagnosis is interactive and open-ended, involving sequential hypothesis refinement through targeted questioning. We address this gap. We build MeDxBench, a large-scale benchmark of 4,421 clinical cases across 20 specialties. We further propose MeDxAgent, a multi-agent consultation system for interactive diagnosis, and systematically study its prompt-, flow- and agent-level design choices. MeDxAgent achieves a 10.3% accuracy gain over the baseline on MeDxBench, closing 52.3% of the gap to a full-information oracle. We find that specific design choices: collecting demographics first, passing summarized dialogue for diagnosis, and feeding candidate diagnoses for targeted questioning, improve accuracy, mirroring how physicians reason, though their effect emerges fully only in combination. Code and dataset will be released upon publication.

**arXiv ID:** 2606.03416
</details>

<details>
<summary><strong>The Ringelmann Effect in Multi-Agent LLM Systems: A Scaling Law for Effective Team Size</strong> - Blaž Bertalanič, Carolina Fortuna - [[pdf]](https://arxiv.org/pdf/2606.02646)</summary>

**Abstract:** Inference-time multi-agent LLM scaling lacks a shared unit: counting nominal agents conflates cost with independent evidence. We derive a two-parameter scaling law $R(N) = N_\text{eff}/N = 1/(1+c(N-1)N^{-\beta})$ where the regime exponent $\beta$ classifies any configuration into one of three asymptotic regimes -- hard-ceiling at $1/c$ ($\beta = 0$), sublinear at $N^\beta/c$ ($0 < \beta < 1$), or linear ($\beta \ge 1$), and a mean-field theorem predicts that peer count $k$ and rounds $\tau$ during agent debate enter the dynamics only through their product $k\tau$. The law applies at two levels: answer diversity and correctness redundancy.
Across 44 (model $\times$ task $\times$ condition) cells spanning peer debate, self-correction, random-noise placebo, self-consistency, three open-weight families (Qwen, Llama, Ministral) at scales from 7B to 32B with a frontier API check (Gemini), thinking models, heterogeneous teams, and sparse communication, the functional form fits every condition at $R^2 > 0.99$; only $(c, \beta)$ shifts. On free-form math, dense peer influence collapses the answer-level regime from sublinear into hard-ceiling; correctness-level fits remain hard-ceiling throughout. Three findings have practical implications. \emph{(i)}~Thirty dense debating agents produce no more answer diversity than one on MMLU-Hard. \emph{(ii)}~A noise placebo tracks self-correction on free-form math and at $4\times$ scale, so within homogeneous teams the gain commonly attributed to ``debate'' comes from re-evaluation, not peer content. \emph{(iii)}~A single $N \le 5$ pilot predicts the $N=30$ structural ceiling, and within the configurations tested only architectural diversity (heterogeneous teams) lowers $c$ and escapes the hard-ceiling regime, communication-mode interventions do not.

**arXiv ID:** 2606.02646
</details>

<details>
<summary><strong>Economy of Minds: Emerging Multi-Agent Intelligence with Economic Interactions</strong> - Zhenting Qi, Huangyuan Su, Ao Qu, Chenyu Wang, Yu Yao, Han Zheng, Kushal Chattopadhyay, Guowei Xu, Zihan Wang, Weirui Ye, Vijay Janapa Reddi, Ju Li, Paul Pu Liang, Himabindu Lakkaraju, Sham Kakade, Yilun Du - [[pdf]](https://arxiv.org/pdf/2606.02859)</summary>

**Abstract:** How can a population of agents self-orchestrate and self-adapt into stronger collective intelligence without centralized control? Inspired by Friedrich Hayek's economic theory of decentralized coordination in markets, we study this question through an agent economy in which agents compete via auctions for the right to act, exchange payments, and accumulate wealth from environmental rewards. These simple economic signals induce decentralized credit assignment, driving planning without global orchestration or explicit communication protocols. The population evolves through economic selection: effective agents accumulate wealth and are mutated via exploitation, while ineffective ones go bankrupt and are replaced via exploration. We show that, initialized with weak agents, the economy produces emergent multi-step reasoning strategies and outperforms stronger monolithic baselines across five agentic tasks, including mathematical reasoning, financial research, scientific research, accelerator design, and distributed-system optimization. We further provide theoretical insights into how economic dynamics shape agent behaviors, linking local incentives to long-term global performance. Our results suggest a new path to multi-agent intelligence: rather than engineering coordination, we can design decentralized incentive structures under which it automatically emerges.

**arXiv ID:** 2606.02859
</details>

<details>
<summary><strong>When Helping Hurts and How to Fix It: Multi-Agent Debate for Data Cleaning</strong> - Chirag Parmar, Akshat Mehta, Henglin Wu, Jagadish Ramamurthy, Shweta Medhekar - [[pdf]](https://arxiv.org/pdf/2606.02866)</summary>

**Abstract:** When does multi-agent debate help data cleaning, and when does it hurt? Across three benchmarks, four model families, and over 6,000 task-condition pairs, we find debate's effect reverses sign: it degrades generation across all four models (-1.6 to -15.5pp) through critique-induced confusion (CIC), hallucinated Critic feedback that the Generator accepts uncritically, yet improves error detection (+27.4pp F1, d=1.0). We derive a debate benefit condition: debate helps when the probability of rescuing a wrong output (Critic verification odds weighted by fixability) exceeds the probability of destroying a correct one. A factorial experiment proves adversarial separation is essential: self-verification with identical tools fails, while a separate Critic with code-execution grounding and evidence-gated generation produces the first debate configuration to significantly exceed single-agent on a generative task (+5.3pp, p<0.05). The condition correctly predicts all nine task types and generalizes with zero false positives across 19 published comparisons in seven domains.

**arXiv ID:** 2606.02866
</details>

<details>
<summary><strong>SPOQ: Specialist Orchestrated Queuing for Multi-Agent Software Engineering</strong> - Royce Carbowitz, Dheeraj Kumar - [[pdf]](https://arxiv.org/pdf/2606.03115)</summary>

**Abstract:** Multi-agent AI systems show promise for automating software engineering tasks, yet existing approaches suffer from coordination overhead, quality control gaps, and limited human oversight. We introduce SPOQ (Specialist Orchestrated Queuing), a methodology combining three innovations: (1) wave-based topological dispatch that computes parallel execution waves from task dependency graphs; (2) dual validation gates applying quality metrics before execution (planning validation) and after (code validation) to reduce rework cycles; and (3) Human-as-an-Agent (HaaA) integration, where a human specialist participates in decomposition and can be consulted during execution. SPOQ uses a three-tier agent hierarchy (Opus workers, Sonnet reviewers, Haiku investigators) to optimize cost-quality tradeoffs. We evaluate SPOQ through four experiments. Experiment 1: wave dispatch approaches the critical-path lower bound (ratio 1.03--1.11, speedup up to 14.3x); on a 2-slot local backend it delivers a stable 1.4x speedup. Experiment 2: SPOQ improves planning coverage from 93.0 to 99.75, eliminates cyclic plans, and lifts parallelism from 31.0 to 75.25. Experiment 3: dual validation reduces defects from 0.34 to 0.20 per task and lifts test pass rate from 91.25% to 99.75%. Experiment 4: human review reduces residual defects from 0.47 to 0.03 per task. Results are replicated on a locally hosted open-weights model (Qwen3.6-35B-A3B), verifying gains are attributable to orchestration rather than any specific model. A longitudinal study across 17 repositories, 8,589 commits, 1,822 tasks, and 13,866 tests (99.87% pass rate) provides ecological validation.

**arXiv ID:** 2606.03115
</details>

<details>
<summary><strong>Validation-Gated Multi-Agent Governance for Online Adaptation of Thermal-Hydraulic Surrogate Models under Operating-Regime Shift</strong> - Doyeong Lim, Seungyoon Lee, In Cheol Bang - [[pdf]](https://arxiv.org/pdf/2606.03321)</summary>

**Abstract:** Artificial-intelligence surrogates can support second-by-second thermal-hydraulic forecasting, but models selected and frozen offline may become condition-locked once deployed outside their pretraining envelope. This study develops a guarded continual-adaptation framework for experimental thermal-hydraulic loop data in which role-separated agents - Monitor, Diagnosis, Adaptation, Safety-Auditor, and Orchestrator - diagnose error signatures, prioritize candidate model families, and review promotions, while deterministic champion-challenger gates and background shadow learning retain final authority over model replacement. Seven surrogate families were screened by blocked three-fold cross-validation, and a temporal Fourier neural operator was selected as the initial champion for 60-s-history-to-10-s-trajectory forecasting on two held-out transients, with three seeds per adaptive mode. Static deployment gave a channel-averaged MAE of 7.06 and a 56.8% warning-exceedance ratio; rule-based adaptation reduced MAE to 6.54, whereas shadow refresh alone remained close to Static. The MA-Full mode, in which the role-separated multi-agent council reviews every evaluated stream step, achieved the lowest mean error, 5.72, and 35.8% exceedance, corresponding to a 19.0% improvement over Static. Paired bootstrap intervals against Static excluded zero, although intervals among adaptive modes overlapped and the six paired units limit broad statistical claims. Validated promotions from the neural operator to Transformer and graph neural network indicate that logged, gate-controlled adaptation can support auditable surrogate evolution while deterministic gates retain deployment authority.

**arXiv ID:** 2606.03321
</details>

<details>
<summary><strong>FORGE: Multi-Agent Graduated Exploitation and Detection Engineering</strong> - Farooq Shaikh - [[pdf]](https://arxiv.org/pdf/2606.03453)</summary>

**Abstract:** Vulnerability disclosure volumes now far exceed organizational assessment capacity, yet three adjacent research communities (proof-of-concept generation, vulnerability prioritization, and detection rule engineering) operate largely in isolation. Existing automated exploit generation systems report binary pass/fail outcomes, discarding partial progress and producing no signal for the other two communities. This paper presents FORGE, a multi-agent system that bridges these three silos through graduated exploitation depth. Five specialized agents (Intel, Generator, Planner, Exploit, and Detector) execute in a fixed pipeline that (1) generates targeted vulnerable applications from CVE metadata, (2) conducts coached, multi-turn exploitation assessed by an LLM-primary oracle on a four-level taxonomy (L0: no evidence through L3: full compromise), and (3) produces Sigma and Snort detection rules grounded in OpenTelemetry exploitation traces. Graduated depth is the bridging mechanism: deeper exploitation yields richer behavioral traces for detection engineering, while depth data across scoring bands provides ground truth for prioritization validation. A tiered knowledge architecture accumulates intelligence across assessments, transferring build and exploitation experience to subsequent CVEs. Evaluation on 603 CVEs from the CVE-GENIE dataset achieves 67.8% end-to-end L1+ exploitation at USD 1.50 per CVE across eight languages and 187 CWE types. Exploitation rates remain near 68% regardless of EPSS or CVSS band, indicating that pattern-level reachability is orthogonal to metadata-based prioritization. Detection rules from L2+ exploitation achieve significantly higher span-normalized grounding than L1-derived rules (p=0.035), and 93.4% of generated Snort rules produce zero false positives against a synthetic benign corpus.

**arXiv ID:** 2606.03453
</details>

<details>
<summary><strong>On dynamic multi-agent pathfinding methods: review, simulations and modifications</strong> - Gabriel Fejziaj, Salama Hassona, Wieslaw Marszalek - [[pdf]](https://arxiv.org/pdf/2606.03735)</summary>

**Abstract:** This paper presents a systematic study of pathfinding algorithms in the context of Dynamic Multi-Agent Pathfinding (D-MAPF), a setting that combines dynamic obstacles, partial observability, and inter-agent conflicts. We evaluate six representative algorithms: Dijkstra, D* Lite, Space-Time A*, WHCA*, M*, and a novel method denoted as A** within a unified simulation framework. The proposed A** algorithm introduces a template-based approach that decouples offline geometric path generation from online temporal adaptation. By precomputing multiple diverse candidate paths and dynamically reconnecting to them using space-time planning, A** improves solution quality in environments with frequent changes and limited sensing

**arXiv ID:** 2606.03735
</details>

<details>
<summary><strong>Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning</strong> - Beyazit Yalcinkaya, Marcell Vazquez-Chanlatte, Ameesh Shah, Hanna Krasowski, Sanjit A. Seshia - [[pdf]](https://arxiv.org/pdf/2511.02304)</summary>

**Abstract:** We study learning multi-task, multi-agent policies for cooperative, temporal objectives, under centralized training, decentralized execution. In this setting, using automata to represent tasks assigned to agents enables breaking down a team-level objective into simpler, smaller sub-tasks. However, existing approaches remain sample-inefficient and are limited to the single-task case, requiring retraining policies for each new task. In this work, we present Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning (ACC-MARL), a framework for learning task-conditioned, decentralized team policies. We identify challenges to the feasibility of ACC-MARL, propose solutions, and prove that our approach is optimal. We further show that learned value functions can be used to assign tasks optimally at test time. Experiments demonstrate emergent task-aware, multi-step coordination among agents, such as pressing a button to unlock a door, holding the door, and short-circuiting tasks.

**arXiv ID:** 2511.02304
</details>

</details>

<details open>
<summary><h2>Other Agent Research (3 papers)</h2></summary>

<details>
<summary><strong>Capability Advertisement as a Market for Lemons: A Trust Layer for Heterogeneous Agent Networks</strong> - Gaurav Naresh Mittal - [[pdf]](https://arxiv.org/pdf/2606.03034)</summary>

**Abstract:** Large language model (LLM) agents have begun to delegate work to one another. Protocols such as the Model Context Protocol (MCP) and the Agent2Agent protocol (A2A) let an agent publish what it can do and let others call it, and public registries of such agents are already appearing. These protocols assume an advertised capability is a static, truthful fact. A real agent is none of these things: its competence is probabilistic, varies with input, drifts when the underlying model is updated, and, because the agent is itself a language model, it can describe itself with complete confidence and be wrong. A caller therefore sees what an agent claims to do, not what it can do, with no principled way to tell a reliable provider from a fluent impostor.
We argue these difficulties share one cause: the market for lemons. When quality is hidden and claims are cheap, good and bad providers become indistinguishable, honest reliability goes unrewarded, and the market decays toward its worst participants. Economics offers three remedies, signaling, screening, and reputation, and none are present in today's agent protocols.
We make four contributions: (1) a failure taxonomy that names confident-wrong as a non-adversarial, correlated subclass of Byzantine faults that classical fault-tolerance mismodels; (2) a market-for-lemons model showing that faith-based protocols admit only a low-trust equilibrium; (3) the Trust Layer, a thin, protocol-agnostic narrow waist above MCP and A2A that adds probabilistic capability descriptors, screening, and reputation, and admits a separating equilibrium when the cost of sustaining an overclaim exceeds the gain from it; and (4) a reliability-composition bound for delegation chains with an end-to-end placement argument. The design needs no model retraining and degrades gracefully when its trust anchors are absent or corrupt.

**arXiv ID:** 2606.03034
</details>

<details>
<summary><strong>OpenAgenet/OAN: Open Infrastructure for Trusted Agent Interconnection</strong> - Jinliang Xu - [[pdf]](https://arxiv.org/pdf/2606.03161)</summary>

**Abstract:** OpenAgenet, abbreviated as OAN, is an open infrastructure project for trusted Agent interconnection. It addresses a problem that becomes visible when Agents move from isolated applications into open, multi-operator networks: before an Agent can safely discover, select, and invoke another Agent, it needs a way to verify identity provenance, governance state, discovery authorization, freshness, and pre-connection trust evidence. OAN is designed as a protocol-neutral trust layer. It does not replace Agent interaction protocols, tool protocols, model orchestration frameworks, or application-level workflows. Instead, it provides Root-governed identity admission, Registrar-assisted onboarding, Root-verified package publication, authorization-aware Discovery, and signed trusted invocation. This paper presents the motivation, architecture, roles, governance model, relationship with MCP, A2A, and ANP, deployment patterns, cooperation model, blockchain-backed authorization bulletin, prototype status, performance profile, and roadmap of OAN.

**arXiv ID:** 2606.03161
</details>

<details>
<summary><strong>Self-Regulation through Communication in Evolved Neural Agents</strong> - Joshua Nunley - [[pdf]](https://arxiv.org/pdf/2606.02840)</summary>

**Abstract:** Communication is typically understood as indication: signals that transfer information from sender to receiver. We present a minimal predator avoidance task in which pairs of evolved CTRNN agents use communication for robust survival, and in which agents hear their own vocalizations, as in natural systems. Across 112 perfect-fitness agents from over 2,000 evolutionary runs, three dominant strategies emerge (accounting for 81% of agents): safety calling (39%), where agents signal from safe cover; alarm indication (22%), where agents vocalize when a threat is present without relying on self-hearing; and self-regulatory calling (20%), where agents depend on hearing their own call to sustain escape behavior. Self-hearing dependency is common among agents that call during an active threat (47%), but rare among agents that call only after reaching safe cover (10%; p < 10^-4). This pattern is consistent with a difference in causal order: safety callers act then communicate, while self-regulatory callers communicate in order to act. Removing self-hearing selectively impairs self-regulatory callers (fitness 0.40) while safety callers remain functional (0.90; p < 10^-9). These results show that communication can evolve to serve the caller's own behavioral regulation, not just information transfer to others.

**arXiv ID:** 2606.02840
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (1 papers)</h2></summary>

<details>
<summary><strong>OpenAgenet/OAN: Technical Architecture for Trust-Governed Agent Identity and Discovery</strong> - Jinliang Xu - [[pdf]](https://arxiv.org/pdf/2606.03163)</summary>

**Abstract:** This paper describes the technical architecture of OpenAgenet / OAN. OAN is a protocol-neutral trust layer for open Agent interconnection. It specifies the role architecture, identity objects, registration workflow, Root-governed lifecycle, Root-verified package model, authorization-aware Discovery, signed trusted invocation, verification requirements, state transitions, security properties, implementation boundaries, and deployment considerations. The design is intended to support heterogeneous Agent frameworks and interaction protocols, including MCP, A2A, ANP-like systems, and domain-specific Agent protocols. OAN does not define the entire business conversation among Agents; it defines how Agent identities become admissible, discoverable, verifiable, and safe to approach before protocol-specific interaction begins.

**arXiv ID:** 2606.03163
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
