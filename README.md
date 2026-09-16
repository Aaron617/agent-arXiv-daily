# Agent arXiv Daily

**Last Updated:** 2026-09-16 05:59:05

**Total Papers:** 75

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (2 papers)</h2></summary>

<details>
<summary><strong>ThinkFlow: Self-Evolving Probabilistic Latent Memory for Lifelong Conversational Agents</strong> - Cai Ke, Xin Liu, Han Zhang, Jiangyue Yan, Zike Yuan, Ling Deng, Yue Yu, Hui Wang, Ruifeng Xu - [[pdf]](https://arxiv.org/pdf/2609.17010)</summary>

**Abstract:** Lifelong conversational agents rely on memory systems to maintain deep, context-aware interactions with users. However, existing explicit textual memory pipelines suffer from a severe information bottleneck, often losing subtle behavioral patterns and emotional shifts. Furthermore, being typically static post-deployment, they cannot autonomously adapt to personal habits and preferences without manual feedback. Cognitive science, however, suggests that humans maintain mental models purely in a latent space and continuously refine them through predictive coding. Inspired by this, we propose \textbf{ThinkFlow}, a novel end-to-end latent memory framework for lifelong conversational agents. ThinkFlow bypasses the text bottleneck by dynamically compressing conversational flows into probabilistic latent memory skills, autonomously consolidating complex user states into disentangled, continuous vectors without semantic interference. To break this barrier, we introduce a test-time evolution paradigm. By coupling teacher-guided latent alignment to bootstrap the initial state with a self-supervised next-user-utterance prediction task for continuous refinement, the framework successfully overcomes cold-start challenges and achieves label-free lifelong personalization. Extensive experiments on long-term conversation benchmarks demonstrate that ThinkFlow significantly outperforms prevailing memory systems, providing highly personalized and contextually accurate responses over extended multi-session interactions.

**arXiv ID:** 2609.17010
</details>

<details>
<summary><strong>API Benchmark Scores Do Not Reliably Transfer to Chatbot Interfaces</strong> - Jennifer Wang, Joachim Baumann, Daniel E. Ho, Sanmi Koyejo - [[pdf]](https://arxiv.org/pdf/2609.08861)</summary>

**Abstract:** Benchmark scores are a central currency in model releases: they inform purchasing decisions, shape public trust, and influence policy. Yet, a key assumption underlying benchmark scores is that the model performance measured through APIs faithfully reflects the behavior of deployed systems.
We challenge this assumption by auditing ChatGPT, Claude, and Gemini across seven systems and nine benchmarks spanning general capability, social bias, and sycophancy. We find systematic API--interface differences in both accuracy and consistency. On average, API evaluations score 3.4 percentage points higher in accuracy and 2.1 percentage points higher in test--retest agreement than corresponding interface evaluations. For ChatGPT, the performance difference between API and interface access rivals the API-only difference between GPT 5.3 and GPT 5.4. Put differently, switching access surfaces can degrade performance as much as downgrading a full model generation.
We further test whether exposed API controls can reproduce interface behavior by varying system prompts, sampling parameters, and reasoning settings. These controls shift behavior in some cases but do not reliably eliminate the gap. Our findings document a context-validity gap: measurements obtained through APIs do not necessarily generalize to corresponding deployed interfaces, complicating the use of API evaluations as proxies for deployed systems.

**arXiv ID:** 2609.08861
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (21 papers)</h2></summary>

<details>
<summary><strong>Toward Governance-Aware Autonomous GIS: A Narrative Review of Ethical and Privacy Risks in LLM-Enabled GeoAI</strong> - Maya Subramanian, Devika Jain - [[pdf]](https://arxiv.org/pdf/2609.16232)</summary>

**Abstract:** Geospatial artificial intelligence (GeoAI) powered by large language models (LLMs) is expanding the capacity to query, generate, and interpret spatial information through natural-language interfaces and agentic autonomous GIS workflows. This capability creates governance challenges that general AI ethics discussions do not fully capture, including passive location inference from mobility traces, spatially structured bias amplification driven by spatial autocorrelation and scale effects, hallucinated spatial facts, and uncertainty compounding across multimodal geospatial inputs. This narrative review identifies eight recurring issues in LLM-enabled GeoAI: data provenance and consent, spatial privacy and inference risk, algorithmic bias and spatial inequity, spatial mechanisms as structural risk (spatial autocorrelation, the modifiable areal unit problem, and scale effects), LLM-specific technical risks, explainability, policy and regulatory gaps, and public enablement and workforce development. For each issue, we characterize the underlying mechanism, ground it in an illustrative example from the literature, and assess the current state of technical or institutional responses, ranging from largely unaddressed to actively debated or subject to emerging policy. Building on this synthesis, we propose a governance-aware architecture for LLM-enabled autonomous GIS that maps each issue to enforceable controls and auditable artifacts across the geospatial data lifecycle, illustrated through a worked flood-response routing scenario. The review highlights a persistent evidence gap: proposed responses remain largely conceptual, and field-tested evaluations of governance controls for LLM-enabled GeoAI remain limited. We close by outlining a research agenda emphasizing empirical validation, spatially specific interpretability tools, and workforce training aligned with these emerging risks.

**arXiv ID:** 2609.16232
</details>

<details>
<summary><strong>BLINDSPOT: A Benchmark for Safety and Refusal Calibration in Long-Horizon Tool-Using Agents</strong> - Sadia Asif, Mohammad Mohammadi Amiri, Momin Abbas, Tejaswini Pedapati, Prasanna Sattigeri - [[pdf]](https://arxiv.org/pdf/2609.16305)</summary>

**Abstract:** Large language model (LLM) agents increasingly operate over long-horizon interactions involving tool use, persistent state, evolving authorization, and external environment feedback. In such settings, safety failures may emerge only after multiple turns, yet existing evaluations often reduce agent behavior to task or attack success, obscuring whether an agent acts, refuses, or remains appropriately calibrated as the interaction evolves. We introduce Blindspot, a benchmark for trajectory-level safety calibration of long-horizon tool-using agents. Blindspot evaluates complete user-agent-environment trajectories through adaptive adversarial interaction, stateful tool execution, and execution-grounded adjudication. Its current instantiation contains 22 attack families and 35 scenarios across seven domains, yielding more than 2,500 long-horizon trajectories with an average interaction length of 14.7 turns. Each trajectory is assigned one of five outcomes: Safe Completion, Correct Refusal, Unsafe Completion, Over-Refusal, or Indeterminate. Unlike fixed attack datasets, Blindspot is an extensible live-simulation framework in which attacks, scenarios, tools, policies, domains, and agent configurations can be added without redesigning the evaluation pipeline. We evaluate 13 proprietary and open-weight LLMs using eight metrics covering unsafe completion, appropriate refusal, benign utility, over-refusal, repeated-run robustness, and post-refusal failure. Preliminary results reveal substantial differences in safety-utility calibration across models and show that failures can emerge only after several initially safe interaction steps. These findings motivate treating agent safety as a trajectory-level property rather than a single-turn or binary success criterion.

**arXiv ID:** 2609.16305
</details>

<details>
<summary><strong>Skill-based Agentic Evaluation for Real-time Data Science Tasks</strong> - Aniruddha Tamhane, Raghavendra Addanki, Ayushi Aggarwal, Aditya Bansal, Rui Wang, Charles Menguy, Swati Jain - [[pdf]](https://arxiv.org/pdf/2609.16487)</summary>

**Abstract:** We present a framework for evaluating data-science agents on live, continuously updated data using executable ground truth and format-agnostic factoid scoring. Consider this example query: "what were last week's audience sizes"---the reference answer changes as the underlying data changes, so static references become outdated and standard LLM-as-a-judge pipelines cannot verify responses against a fixed ground truth. Our central contribution, ground-truth-as-code, encodes each expected answer as an executable reference function that recomputes the answer directly from live data at evaluation time, ensuring the reference remains consistent with the system it describes. We combine this with a factoid-level, format-agnostic judge that decomposes both the agent's response and the computed ground truth into atomic claims and scores precision, recall, and accuracy over them, irrespective of the response format (prose, list, table, HTML, etc.). The approach is applicable to agents whose expected outputs can be expressed as executable data computations. We validate the framework through a human--LLM agreement study on an internally developed machine learning skill deployed in production, using a synthetic database constructed to reproduce production schemas and entity relationships. Relative to a natural-language ground-truth baseline, our method achieves a 29% improvement in the Matthews Correlation Coefficient (MCC)---a class-balanced measure of agreement between expert annotators and LLM-as-a-judge predictions---and a 16% reduction in token consumption per test case, while a self-directed baseline lacking explicit ground truth is anti-correlated with human judgment. Agents that perform multi-source data integration and computation over non-stationary data are routinely deployed in industry; we propose ground-truth-as-code as a practical methodology for their evaluation.

**arXiv ID:** 2609.16487
</details>

<details>
<summary><strong>little m: An AI Agent for Industrial Process Optimization</strong> - Yongchao Ye, Xinyu He, Dutliff Boshoff, Way Kuo, Lishuai Li - [[pdf]](https://arxiv.org/pdf/2609.16680)</summary>

**Abstract:** Manufacturing consumes one third of global energy and still has significant room for improvement in terms of energy efficiency. Optimal process control is essential for this purpose. However, synthesizing mathematical optimization models from messy, real-world industrial specifications requires bridging unstructured natural language and spatial diagrams with rigorous mathematical syntax. This poses a profound challenge for general-purpose Large Language Models (LLMs), which may introduce invalid constraints when tasked with modeling continuous multi-physics dynamics. To address this, we introduce little m, an AI agent designed to assist the formulation of industrial process control models. Combining a domain-specific knowledge repository with LLM-driven interaction, the proposed framework formulates real-world optimization problems as mathematical models. For systematic evaluation, we introduce the Industrial Process Control Benchmark (IPC-Bench), a novel multimodal dataset of 50 canonical scenarios requiring joint reasoning over text and process diagrams. Through comprehensive automated structural assessments and double-blind human evaluation, little m substantially outperforms state-of-the-art LLMs, generating semantically correct models. These evaluations assess formulation quality rather than solver feasibility, formal physical validity, or closed-loop industrial performance. The implementation of little m and the IPC-Bench dataset are available at this https URL.

**arXiv ID:** 2609.16680
</details>

<details>
<summary><strong>End-to-End Latency-Minimizing and Load-Balanced Request Scheduling for Edge LLM Inference in Agentic AI Services</strong> - Zhen Li, Jun Cai, Haoran Gao, An Li, Tan Li - [[pdf]](https://arxiv.org/pdf/2609.17193)</summary>

**Abstract:** Large language model (LLM)-powered agentic AI services increasingly demand low-latency inference, motivating the deployment of LLMs across distributed edge servers. However, heterogeneous communication and computing capabilities, together with dynamically evolving inference states, make the edge server selection for each incoming request time-varying and tightly coupled across slots. In this paper, we investigate an online request scheduling framework for edge LLM inference that jointly minimizes long-term average end-to-end latency and regulates workload distribution across heterogeneous edge servers. Two main challenges arise in this context. First, conventional latency models cannot accurately capture the fine-grained dynamics of multi-stage LLM execution. Second, the latency consequence of a scheduling decision is observed only after request completion, making immediate decision evaluation difficult. To address these challenges, we develop a cross-slot inference model that captures transmission, prefill, iteration-level decoding, and key-value (KV) cache evolution for each diverse request, and characterize server workload through a KV cache memory-time consumption metric. We propose the LYREO approach that transforms the long-term load-balancing constraint via Lyapunov optimization and employs reward redistribution with sequencebased return prediction to convert delayed outcomes into timely learning signals for earlier decisions. Simulations under various configurations demonstrate that LYREO consistently achieves lower latency and more balanced load distribution than representative learning-based and heuristic baseline schemes.

**arXiv ID:** 2609.17193
</details>

<details>
<summary><strong>You Don't Need To Train: Agentic Heuristic Learning Studio for Executable Human Activity Recognition</strong> - Siyu Yuan, He Zhang, Sizhen Bian, Bin Guo - [[pdf]](https://arxiv.org/pdf/2609.16065)</summary>

**Abstract:** Human activity recognition (HAR) is usually framed as gradient-based training of neural networks. Agentic Heuristic Learning (AHL) Studio explores a complementary view inspired by human cognitive learning: people learn activities by remembering examples, forming rules, and repairing mistakes, not by backpropagating. This proposed tool implements AHL for HAR: a learning-time agent reasons over sensor protocols, proposes executable heuristic policies, records repair traces, and exports an LLM-free policy for edge deployment. We focus on the HAR benchmark family and provide an end-to-end workflow from dataset observation to edge-oriented export. On eleven HAR datasets evaluated so far, AHL policies reach strong executable-policy performance while remaining inspectable, editable, and replayable \footnote{this https URL}.

**arXiv ID:** 2609.16065
</details>

<details>
<summary><strong>Coaching Qwen3 Coder 30B to Think Like a CodeClash Arena Agent</strong> - Ivy Ning Zhang - [[pdf]](https://arxiv.org/pdf/2609.16096)</summary>

**Abstract:** Large language model coding agents have recently become useful for software tasks, but weaker or open-weight agents still struggle to reliably interpret user intent and execute complex multi-step workflows. This gap is especially visible in long-horizon settings, where an agent must repeatedly inspect prior outcomes, diagnose failure, and choose the next code edit under interaction constraints. It motivates a natural question: what can we do to improve the thinking process of a weak code agent? We study this question in CodeClash, a code-arena benchmark where the original work evaluates 8 commercial coding agents across 6 arenas through multi-round tournaments. Since Qwen3 Coder Plus ranks last among them, we take the open-weight Qwen3-Coder-30B as a case study and investigate how to improve it with distilled knowledge from stronger agents. Our analysis shows that Qwen3-Coder-30B is not well optimized for arena-style interaction: it frequently produces syntax and protocol-breaking errors and exhibits weak strategic adaptation across rounds. These failures are difficult to correct with vanilla instruction tuning alone, since offline SFT cannot directly verify whether a generated action is valid or beneficial. To address this, we propose ReAct SFT, which rewrites teacher trajectories into explicit [obs][thought][act] chains, and trajectoryquality weighted SFT, which reweights samples to encourage post-edit checking. ReAct SFT substantially improves strategic behavior, and our fine-tuned model outperforms the original Qwen3 Coder Plus in tournament evaluation.

**arXiv ID:** 2609.16096
</details>

<details>
<summary><strong>AURA: Agentic Diagnosis and Refinement for Production Recommender Systems at Scale</strong> - SungGeun Kim, Abhinav Narain, Daniel Nemirovsky - [[pdf]](https://arxiv.org/pdf/2609.16625)</summary>

**Abstract:** How and why does a recommender system fail the users it serves? Oftentimes, practitioners are left to improve their algorithms based on a combination of feedback from stakeholder teams, domain expertise, and insights from data analyses. Yet the nuances of how and where recommendations perform well or poorly for end users are difficult to discern from aggregate quantitative metrics. Whereas these metrics provide a high-level and incomplete picture, further granularity into the quality of recommendations and their patterns requires reasoning with domain understanding and objectivity, at scale. We contemplate this complex conundrum and describe a method and implementation that uses the latest AI agentic advances to provide actionable diagnoses and improvements for production recommender systems. We present AURA (Agentic Understanding and Refinement of recommender Algorithms), an end-to-end agentic system that performs qualitative evaluation at scale and can then generate improvements to our algorithms at the code level. Specialized agents read production engagement logs, from thousands of sessions to millions, and surface patterns and examples of how the recommender fails real users. The next step uses those diagnoses and context about the recommender's own code, data, and training pipeline to propose and implement refinements grounded in that codebase. We report the system design, initial tests on production data from two large consumer platforms at a major media-streaming company, safeguards, operational learnings, and early results toward a self-improving recommender system. Finally, the diagnostic gap AURA closes is not specific to streaming. The architecture is built to transfer: every domain-specific element enters through the configuration layer that already ported it between our two platforms. We map it concretely to e-commerce and online-retail recommendation.

**arXiv ID:** 2609.16625
</details>

<details>
<summary><strong>World Models for Embodied Intelligence: From Plausible to Controllable to Actionable</strong> - Nanjie Yao, Hao Wang, Chong Cheng, Zhikang Chen, Wenzhe Li, Jiafei Lyu, Li Shen, Peilin Zhao, Zongqing Lu, Gao Huang, Steven Hoi, Dacheng Tao, Deheng Ye - [[pdf]](https://arxiv.org/pdf/2609.16697)</summary>

**Abstract:** World models connect perception and decision-making in embodied intelligence by maintaining hidden state, anticipating consequences, comparing interventions, and adapting when execution departs from expectations. Although progress is often measured by visual fidelity, their value lies in improving behavior. Before reaching for a cup, a person anticipates its weight and resistance to grasping, shaping the hand before contact. Such anticipation is coarse and rarely pictorial, yet it guides action. This raises a central question: which predictive capabilities improve behavior? Existing surveys, organized by architecture, output modality, or application domain, leave this question implicit. We introduce three progressively stronger capability levels: Plausible models preserve task-relevant temporal, geometric, or physical structure; Controllable models additionally predict how interventions alter that structure; and Actionable models translate predictions into measurable gains in planning, action, learning, evaluation, verification, recovery, or data selection. We complement this hierarchy with a 3 x 4 matrix crossing geometry, physics, and action grounding with improvement loops centered on data, rewards, policies, and the model itself. Using this framework, we survey manipulation, navigation, locomotion, autonomous driving, and general embodied learning, tracing technical progressions, clarifying capability requirements, and examining datasets, benchmarks, and evaluation protocols. We identify challenges in long-horizon consistency, uncertainty calibration, causal intervention testing, latency, verification and recovery, and cross-embodiment transfer. This perspective shifts evaluation from visual plausibility toward whether predictions capture task-relevant state, reflect intervention effects, and improve the closed-loop behavior of embodied agents.

**arXiv ID:** 2609.16697
</details>

<details>
<summary><strong>Grounding SWE-Agent Decisions in Architecture-0 Design: Navigating Unknown Unknowns through Physical Mapping</strong> - Zhongkai Wang, Yan Liu - [[pdf]](https://arxiv.org/pdf/2609.17221)</summary>

**Abstract:** Autonomous Software Engineering Agents (SWE-Agents) excel in deterministic coding tasks but struggle with Architecture 0, the nascent system design phase plagued by implicit engineering constraints, or Unknown Unknowns (UUs) that are rarely stated explicitly. To investigate how agents navigate UUs, we explore a progressive trajectory across pure-text self-play, tool-augmented feedback, and external physical mapping. Our empirical analysis reveals a cascading chain of failures. Pure-text reasoning inevitably devolves into polite consensus or plausible yet physically impossible fabrications. Attempting to bridge this gap via an early-stage execution sandbox unexpectedly triggers Specification Gaming: agents exploit their autonomy over validation scripts to bypass physical constraints, achieving superficial success without resolving core architectural flaws. To resolve this self-validation trap, we propose the Physical Mapping Guard (PMG). Grounded in the software engineering principle of Separation of Concerns, PMG revokes verification authority from the agent, forcing semantic intents to be evaluated by an external, deterministic Semantic-to-Physical (S2P) mapping engine. Extensive evaluations demonstrate that PMG completely eradicates physical-layer and validation-layer gaming. By precisely isolating residual failures to semantic reinterpretations and auditor overreach, PMG marks a critical step toward genuine affordance grounding in automated architectural design.

**arXiv ID:** 2609.17221
</details>

<details>
<summary><strong>CTAN: Cycle-Temporal Attention Network for Embodied Audio-Visual Navigation</strong> - Teng Liu, Yinfeng Yu - [[pdf]](https://arxiv.org/pdf/2609.17420)</summary>

**Abstract:** Audio-visual embodied navigation equips robots with the capability to infer the locations of sound sources by integrating visual inputs and acoustic information (e.g., depth observations and binaural audio cues). The core challenge lies in establishing effective semantic interactions across heterogeneous modalities (which exhibit distinct feature distributions). Existing feature fusion strategies, however, often rely on simple multimodal aggregation and therefore fail to capture the underlying geometric and semantic relationships, leading to information degradation in complex environments. To overcome these limitations, this work presents the Cycle-Temporal Attention Network (CTAN), a framework designed for active semantic-enhanced fusion (rather than straightforward multimodal combination). Specifically, the proposed Audio-Visual Reconstruction Cross-Attention (AVRCA) module employs a bidirectional cycle-consistency constraint (between visual and acoustic representations) to reinforce the spatial semantic attributes of both modalities, thereby facilitating more robust cross-modal interaction. Additionally, we design a Temporal Cross-Modal Memory (TCMM) mechanism to dynamically integrate real-time enhanced multimodal features with historical context, reducing performance drops caused by auditory dead zones. Experimental results obtained on the Replica and Matterport3D benchmarks indicate that the proposed approach achieves superior performance over previous audio-visual navigation methods in terms of success rate (SR), success weighted by path length (SPL), and scene navigation accuracy (SNA).

**arXiv ID:** 2609.17420
</details>

<details>
<summary><strong>Autonomous Assessment of Generalizability of AI Agent Capabilities</strong> - Daniel Bramblett, Rushang Karia, Adrian Ciotinga, Pulkit Verma, YooJung Choi, Siddharth Srivastava - [[pdf]](https://arxiv.org/pdf/2512.16733)</summary>

**Abstract:** Safe deployment of black-box AI (BBAI) systems such as foundation model agents requires methods for evaluating their capabilities in novel settings. We define an agent's capability as its ability to achieve a short term objective and formalize the problem of learning models that predict whether, with what effects, and under what conditions, an agent can perform a capability. We introduce Monte Carlo Query Search (MCQS), an active query-synthesis method for learning symbolic stochastic capability models of BBAIs. MCQS models capabilities as conditional probability distributions over outcomes and formulates capability evaluation as an active learning problem over policies. We use Monte Carlo tree search to synthesize queries that maximally distinguish between extremal capability hypotheses: the lattice meet and join corresponding to the most pessimistic and optimistic models consistent with observed behavior. Executing these queries yields trajectories that prune inconsistent hypotheses. We prove soundness, completeness, and convergence properties under standard realizability and sampling assumptions. Experiments with multiple BBAI systems show that MCQS learns accurate capability models more efficiently than baseline query strategies, enabling systematic characterization of agent capability boundaries with fewer interactions.

**arXiv ID:** 2512.16733
</details>

<details>
<summary><strong>Shared Selective Persistent Memory for Agentic LLM Systems</strong> - Sanjana Pedada, Aditya Dhavala, Neelraj Patil - [[pdf]](https://arxiv.org/pdf/2607.09493)</summary>

**Abstract:** Agentic LLM systems that generate code through multi-turn tool use face a fundamental context problem: each session starts from zero, discarding the domain constraints, data schemas, tool configurations, and output preferences that made previous sessions productive. We introduce shared selective persistent memory, an architecture that retains four categories of reusable context - task specifications, data schemas, tool configurations, and output constraints - while discarding session-specific reasoning traces, and that packages them into workspaces transferable across users under role-based access control. The resulting cost curve is non-monotonic. In a controlled replication on four public datasets, where a formatting specification is established once and then withheld, no memory completes 0/12 trials at 3.8K input tokens, selective memory completes 12/12 at 3.9K, and full conversation history completes 8/12 at 7.7K. What is kept matters more than how much is kept: the winning configuration costs essentially what the failing one does, and twice as much context does not improve on it. Both differences from no memory survive Bonferroni-corrected exact McNemar tests (p = 0.0005, p = 0.008); the two memory conditions separate on price rather than completion. We implement this in a deployed platform where agents produce git-versioned artifacts from CSV, SQL, REST, and MCP sources. A complementary zero-token data refresh contract decouples generated programs from runtime data, firing on 12/12 trials at a median 0.08s with no model call, while summary-driven data representation costs 97-431x fewer tokens than raw injection. Across 24 recurring enterprise tasks selective memory completes 23/24 against 19/24 and 17/24, though at that sample no pairwise difference reaches significance.

**arXiv ID:** 2607.09493
</details>

<details>
<summary><strong>Know Your Agent: Reconnaissance-Driven Pentesting of AI Agents</strong> - Or Zion Eliav, Eyal Lenga, Shir Bernstien, Yisroel Mirsky - [[pdf]](https://arxiv.org/pdf/2607.19837)</summary>

**Abstract:** Traditional pentesting uses reconnaissance at each step to uncover unseen weaknesses, build stronger attacks, and advance the objective; we argue that AI agents require the same treatment. We formalize agent reconnaissance by modeling the process and identifying the knowledge assets it seeks to extract: what they are, how they are used, and which agent weaknesses they exploit to give adversaries leverage in indirect prompt injection attacks. We instantiate these insights in Know Your Agent (KYA), a framework that automates black-box, reconnaissance-driven pentesting by probing agents, building target profiles, and using those profiles to craft stronger attacks. We evaluate KYA on agent-security benchmarks and a real-world coding agent, and release KYA, its benchmarks, and baseline implementations for reproducibility.

**arXiv ID:** 2607.19837
</details>

<details>
<summary><strong>CASCADE: An Agentic Regulatory Network Framework for Patient-Data-Validated Downstream Perturbation Prediction</strong> - Jose A. Bird - [[pdf]](https://arxiv.org/pdf/2608.05359)</summary>

**Abstract:** CASCADE is an agentic framework that predicts downstream transcriptional effects of gene perturbation from precomputed
ARACNe regulatory networks, exposed via MCP. Prior work validates such tools by checking whether predicted genes are
known cancer genes (membership); we instead test whether the predicted direction of change matches reality, using
focal-gene copy-number amplification as a dosage-based proxy for the inverse of knockdown against real TCGA patient
tumor data.
For MYC, CASCADE's predicted knockdown targets show strong concordance with real amplified-vs-non-amplified tumor
expression across three cancer types (BRCA: 90.0%, COAD: 72.0%, STAD: 85.7%; all p<0.0013), well above permutation
baselines, surviving a PAM50 subtype control and replicating in an independent cohort (METABRIC, 87.2%). Compared
against curated MSigDB gene-set baselines via Fisher's exact test, CASCADE's accuracy is not shown to exceed existing
public knowledge of MYC- or E2F-driven biology, though its gene-specific direction-calling clearly outperforms a naive
uniform guess.
Extending to fifteen additional genes, validation proves gene-specific rather than universal: proliferation-machinery
regulators mostly replicate, while lineage-identity transcription factors and one cyclin-D paralog (CCND2)
consistently fail, a pattern we discuss as a hedged, post-hoc hypothesis.
We separately benchmark whether an LLM-based agent correctly grounds natural-language requests into CASCADE's real MCP
tool calls. Across 35 queries, a documented local model reaches 71.4% exact match (85.7% for a larger model); schema
and gene-alias failures are resolved by scale or server-side correction, but both models confidently default to the
wrong perturbation type on ambiguous queries, a failure a targeted fix could not resolve because its trigger condition
never occurs.

**arXiv ID:** 2608.05359
</details>

<details>
<summary><strong>CAFE: Self-Improving Search Agents Need Co-Evolving Feedback</strong> - Boyang Liu, Senjie Jin, Peixin Wang, Zhangyue Yin, Yibo Wang, Yuhao Zhou, Zhihao Zhang, Xinbing Liang, Shizheng Zhu, Yuhui Wang, Jingqi Tong, Dingwei Zhu, Zhiheng Xi, Jiazheng Zhang, Clive Bai, Clarenceai, Blaze Chen, Tao Gui, Qi Zhang, Xuanjing Huang - [[pdf]](https://arxiv.org/pdf/2608.24794)</summary>

**Abstract:** Reliable search requires more than acquiring external evidence. An agent must also recognize and recover from errors as its trajectory unfolds. In-trajectory feedback provides a mechanism for such recovery by diagnosing where the search has drifted and redirecting subsequent reasoning steps. This is particularly important in long-horizon search, where an early directional error may receive no immediate corrective signal and can compound across later steps. Making such feedback learnable, however, creates a coupled problem: the agent must learn when to request and use feedback, while the critic must learn corrections from outcome-confounded rollouts as the agent's failure patterns evolve. We introduce CAFE (Coupled Agent--Feedback Evolution), a framework in which a shared-parameter model alternates between search-agent and critic roles. CAFE initializes feedback-conditioned recovery from trajectories built around the base agent's own failures, then couples online and offline optimization. During online RL, a comparative feedback estimate uses a prompt-level call--skip success gap to shape request returns, while feedback-aware advantage shaping reweights token advantages before and after feedback. Offline, rollout-derived preference optimization learns feedback from matched successful and unsuccessful trajectories. On seven agentic search benchmarks, CAFE outperforms the evaluated RL-based search agents on average, retains its gains across all six out-of-domain benchmarks, and reduces answer-level hallucinations. One-sided ablations show that improving only the agent or only the critic eventually plateaus, whereas alternating the two updates continues to improve performance. These findings suggest that a self-improving search agent needs feedback that co-evolves with the policy it guides.

**arXiv ID:** 2608.24794
</details>

<details>
<summary><strong>Stellar Colosseum: A Many-Agent Harness for Long-Horizon Research in Mathematics and Theoretical Computer Science</strong> - Honghao Lin, David P. Woodruff, Yuan Deng, Jieming Mao, Song Zuo, Vahab Mirrokni - [[pdf]](https://arxiv.org/pdf/2609.15983)</summary>

**Abstract:** Language models can produce plausible short proofs, but may still be unreliable on long-horizon research problems, where progress depends on a sequence of uncertain and interdependent decisions. We introduce Stellar Colosseum, a model-agnostic harness for allocating inference across research in mathematics and theoretical computer science. Colosseum explores alternative strategies before proof construction, uses a readiness gate to decide when a route is mature enough to decompose, represents the proof plan as interdependent section-level subproblems, and routes verifier findings back to the affected part of the argument. Across these stages, it generates candidates in parallel, attacks them with targeted falsification, and combines candidates and their critiques into a single research artifact through overlapping random-sample tree aggregation. The Colosseum workflow has been integrated into Google Antigravity's Teamwork framework as the Long Proof pattern.
We demonstrate the capabilities of Colosseum through open-ended research and evaluations on theorem-proving and competitive programming benchmarks. Using Colosseum with Gemini 3.1 Pro, we obtain several new results that address open problems arising from papers published at top venues such as FOCS and JMLR. On TCS-Bench, a benchmark of research-level theorem-proving tasks drawn from papers published at FOCS, STOC, and SODA, Colosseum achieves 71.0% accuracy using Gemini 3.1 Pro and Gemini 3.7 Flash. In a separate Codeforces evaluation using Gemini 3.1 Pro, the proof-oriented pipeline with execution feedback solves 218 of 222 problems.

**arXiv ID:** 2609.15983
</details>

<details>
<summary><strong>DoubleAgents: Human-Agent Alignment in a Socially Embedded Workflow</strong> - Tao Long, Xuanming Zhang, Sitong Wang, Zhou Yu, Lydia B Chilton - [[pdf]](https://arxiv.org/pdf/2509.12626)</summary>

**Abstract:** Aligning agentic AI with user intent is critical for delegating complex, socially embedded tasks, yet user preferences are often implicit, evolving, and difficult to specify upfront. We present DoubleAgents, a system for human-agent alignment in coordination tasks, grounded in distributed cognition. DoubleAgents integrates three components: (1) a coordination agent that maintains state and proposes plans and actions, (2) a dashboard visualization that makes the agent's reasoning legible for user evaluation, and (3) a policy module that transforms user edits into reusable alignment artifacts, including coordination policies, email templates, and stop hooks, which improve system behavior over time. We evaluate DoubleAgents through a two-day in-lab interactive simulation study (n=10), three real-world deployments, and a technical evaluation. Participants' comfort in offloading tasks and reliance on DoubleAgents both increased over time, correlating with the three distributed cognition components. Participants still required control at points of uncertainty - edge-case flagging and context-dependent actions. We contribute a distributed cognition approach to human-agent alignment in socially embedded tasks. We further introduce interactive simulation as a methodological testbed for rapid iteration and alignment testing of agentic systems.

**arXiv ID:** 2509.12626
</details>

<details>
<summary><strong>EviSI: An Evidence-Based Evaluation Agent for Simultaneous Interpreting</strong> - Ben Yan, Zongyao Li, Xiaoyu Chen, Daimeng Wei, Weidong Liu, Huan Zhao, Chong Li, Yaode Wang, Yuzhe Shang - [[pdf]](https://arxiv.org/pdf/2609.08171)</summary>

**Abstract:** Low-latency simultaneous speech-to-speech translation must keep pace with ongoing speech while preserving key information. To meet these demands, systems use segmentation, reformulation and condensation to reorganize and rephrase information. However, metrics developed for text translation, including BLEU and COMET, may not consistently distinguish faithful adaptations from semantic errors. We propose EviSI, a large language model evaluation agent combining Multidimensional Quality Metrics (MQM) with criteria developed with professional interpreters. Shared source evidence guides assessment across four dimensions: Anchor, Event, Logic and Fluency. Verified errors are deduplicated before deterministic scoring. On human-rated English to Chinese and Chinese to English data, EviSI recovers the aggregate English to Chinese human system ranking. Mean within-dataset Kendall correlations for system rankings reach 0.707 and 0.467, respectively, exceeding evaluated BLEU and COMET baselines. A multilingual extension to five directions without human ratings retains the dimensions and scoring rule, showing positive system ranking correlations with COMET throughout.

**arXiv ID:** 2609.08171
</details>

<details>
<summary><strong>Distilling Foundation Models for Agentic What-If Reasoning:Cost, Latency, and Governance in a Hybrid LLM+SLM Architecture</strong> - Sourish Dey, Aditya Kumar - [[pdf]](https://arxiv.org/pdf/2609.16091)</summary>

**Abstract:** Tabular foundation models deliver strong zero-training predictive performance via in-context learning, but their high inference latency makes them impractical as hot-path decision backends in interactive agentic loops. We distill a TabPFN teacher into a compact feed-forward student across a business-decision simulation on UCI Adult and five OpenML benchmarks: the classification head compresses 53.2M parameters to 8,546 (6,220x); the deployed two-head loan pipeline compresses 111.4M parameters to 17,059 (6,532x). The student retains 95.4-100.5% accuracy and 96.8-100.0% AUC, with the lowest accuracy retention on credit-g at 95.4%; an alpha = 0 hard-label control shows that the teacher's soft targets provide a 2.1-7.0 AUC point gain.

**arXiv ID:** 2609.16091
</details>

<details>
<summary><strong>ForkSCOPE: Charting the Agentic Garden of Forking Paths</strong> - Arjun Balaji, Batuhan Duru Yeltekin, Tian Zheng - [[pdf]](https://arxiv.org/pdf/2609.12438)</summary>

**Abstract:** Even with a fixed dataset and research question, data analysis involves many defensible decisions. Understanding how these choices influence the results is scientifically important but remains challenging. Crowdsourcing and agentic AI can generate hundreds of end-to-end analyses, but scaling generation alone can create a processing bottleneck and an analytic ``black hole.'' A common workaround is to impose a shared fixed decision taxonomy, which can limit insight and understate uncertainty. We present ForkSCOPE, a human-AI collaboration framework that induces structure bottom-up from the code corpus of end-to-end analyses, without a taxonomy fixed before or after generation, so the organization and evaluation of the garden can scale with the corpus. ForkSCOPE surfaces the charted garden of forking paths through a human-AI collaboration pipeline and an evidence-linked interactive viewer for steering and verification: it spotlights organically identified forks and structures and produces a derived taxonomy and decision map compatible with existing multiverse tools.

**arXiv ID:** 2609.12438
</details>

</details>

<details open>
<summary><h2>LLM Agents (8 papers)</h2></summary>

<details>
<summary><strong>Turn-level Multiscale Density Ratio Estimation for LLM Agents</strong> - Zishuo Zhao, Kai Chen, Ao Li, Yuan Liu - [[pdf]](https://arxiv.org/pdf/2609.16760)</summary>

**Abstract:** With the rapid development of Large language model (LLM), agent systems enhanced by LLMs show huge potential in being able to deal with complex tasks, especially involving multi-step thinking or interaction with tools. For applying LLM techniques with a well-designed agent paradigm, post-training of LLM in multiple agent scenarios is necessary to achieve better performance. Among the variable post-training techniques, alignment methods such as PPO, DPO, DIL, and GRPO become popular because many papers show a significant positive impact on the model's performance by punishing negative samples while keeping acceptable training complexity. However, most alignment methods address simple single-turn tasks, and there remains room for improvement for complex multi-turn tasks. We propose Turn-level Multiscale Density Ratio Estimation (tlm-DRE), which assigns different weights on corresponding turns and proposes asymmetric token-level training based on the positive-negative space gaps across multiple turns of tasks. The results of the experiment on a wide range of agent benchmarks show that the proposed method performs competitively compared to traditional alignment methods. The proposed training method enables LLMs to perform robustly in multi-turn reasoning tasks with both in-domain and out-of-domain conditions.

**arXiv ID:** 2609.16760
</details>

<details>
<summary><strong>World Model Science: Self-Organized Criticality, Weak Chaos, and Metastable Belief Dynamics in Long-Horizon LLM Agents</strong> - Xinyuan Song, Zekun Cai - [[pdf]](https://arxiv.org/pdf/2609.17419)</summary>

**Abstract:** Long-horizon LLM agents must maintain task state across extended sequences of observations, actions, tool calls, and intermediate beliefs. We study these trajectories through three dynamical views: self-organized criticality, weak chaos, and metastable belief dynamics. Our framework aligns agent-implied states with benchmark-grounded states and measures stress accumulation, error avalanches, temporal dependence, local--global mismatch, bounded divergence, belief-basin transitions, and finite-size scaling under explicit null models. Across 22 experiments spanning controlled puzzles, tool use, embodied tasks, multi-hop retrieval, general-assistant reasoning, and Game of Life, we find that locally valid actions can persist after global state fidelity fails, stress can trigger abrupt collapse, error sequences exhibit long memory, dependency depth changes the propagation regime, and larger horizons support larger avalanches. At the same time, divergence remains bounded, belief states show metastable rather than fully chaotic behavior, and stronger claims of universal power laws, critical points, or shared intervention optima are not supported. These results suggest a science of agent world models based on trajectory-level dynamical diagnostics rather than terminal reward alone.

**arXiv ID:** 2609.17419
</details>

<details>
<summary><strong>Retrieval-Driven Memory Reconsolidation for Long-Term LLM Agents</strong> - Yuanyi Song, Yukai Wang, Xinbei Ma, Zhihui Fu, Jianghao Lin, Weiwen Liu, Jun Wang, Huarong Deng, Yong Yu, Weinan Zhang - [[pdf]](https://arxiv.org/pdf/2609.16053)</summary>

**Abstract:** Long-term memory is essential for LLM-based agents operating over extended interactions. Existing memory systems primarily update memory when new information arrives, treating retrieval as the endpoint of memory access rather than a driver of memory evolution. Consequently, retrieval feedback is rarely exploited to reorganize memory for future access continuously. Moreover, most existing approaches rely on predefined memory structures together with fixed retrieval pipelines, limiting the agent's ability to organize and evolve its own memory autonomously. Inspired by memory reconsolidation in cognitive neuroscience, we propose \textbf{REALM}, a \textbf{r}econsolidation-\textbf{e}volution \textbf{a}gentic \textbf{l}ong-term \textbf{m}emory framework. It models long-term memory as a continual lifecycle by autonomously organizing memories into a heterogeneous cognitive graph, retrieving evidence via adaptively composed graph-search atoms, and continually reconsolidating memories based on retrieval feedback. REALM achieves an average accuracy of 75.97\% on LoCoMo and 65.11\% on LongMemEval, outperforming the strongest baselines by 7.17 and 1.31 points respectively. Ablation studies confirm that memory reconsolidation consistently boosts performance, with further analyses revealing that it progressively reorganizes related memory units into more coherent local structures for collective evidence recall and utilization during reasoning. These results suggest that retrieval-driven memory reconsolidation provides an effective mechanism for continually evolving long-term memory in LLM agents.

**arXiv ID:** 2609.16053
</details>

<details>
<summary><strong>Universal Defenses for Tool-Integrated LLM Agents Against Adversarial Attacks</strong> - Xiaoyan Li, Yunli Wang - [[pdf]](https://arxiv.org/pdf/2609.16098)</summary>

**Abstract:** Large Language Model (LLM) agents have demonstrated impressive capabilities across a variety of domains, particularly when integrated with external tools for multi-step task completion. However, they are increasingly vulnerable to adversarial attacks, including direct prompt injection, indirect prompt injection, memory poisoning, and backdoor attacks, which exploit the model's openness to prompt injection and tool manipulation. In this work, we explore practical and generalizable defense strategies within a unified framework across these four attack types. We introduce two universal tool-based defenses: Attacker Tool Filtering, which uses anomaly detection (e.g., Isolation Forest) to identify and remove suspicious tools, and Normal Tool Recalling, a white-box method that restores the agent's original toolset prior to planning. Additionally, we incorporate prompt-based defenses: Chain-of-Thought prompting and self-reflection techniques to enhance reasoning and task paraphrasing to mitigate attacks. Experimental results across both four open-source LLMs (Gemma2-9B, Qwen2-7B, LLaMA3-8B, and LLaMA3.1-8B) and three proprietary LLMs (GPT-3.5, GPT-4, and GPT-5) show that our methods significantly reduce the Attack Success Rates (ASR), achieving 0% ASR in many settings, while preserving or even improving the original task success rate. These findings highlight the promise of simple, modular, multi-layered defenses for strengthening the security and robustness of tool-integrated LLM agents. The code is available at this https URL.

**arXiv ID:** 2609.16098
</details>

<details>
<summary><strong>Interpreting and Steering LLM Agents for Social Simulations</strong> - Jiayue Gaveal Fan, Arul Murugan, Shreyas Krishnan, Abhishek Nagaraj - [[pdf]](https://arxiv.org/pdf/2609.16436)</summary>

**Abstract:** Simulations based on large language models (LLMs) have proven to be powerful for understanding human behavior, making them valuable additions to the social scientific toolkit. However, LLMs are ultimately black boxes based on deep neural networks which limits their value for social science. This is because of a lack of (i) interpretability: i.e. the ability to assign clear mechanisms driving observed behavior; and a lack of (ii) steerability: i.e. the ability to mute or amplify specific theoretically meaningful mechanisms of action to drive specific model behavior. Here, we demonstrate how the black box could be opened up to further enrich LLM-based simulations. Specifically, we compare three types of methods: (1) prompt-based manipulation, (2) SAE-derived feature steering, and (3) probe-based direction steering and examine their utility for LLM-based social scientific simulations. We do so by interpreting and steering two foundational components of human behaviors, namely preferences (risk attitudes, altruism) and capabilities (divergent creativity, product innovation), operationalized using four classic economic and creative tasks implemented as natural-language interactions. Overall, our results show that SAE- and probe-based techniques often outperform basic prompt-based methods for steering LLM agents, although this advantage depends on the specific prompting strategy involved. Together, SAEs and probes constitute an effective pipeline for social scientists seeking to interpret and steer agents in social simulations: SAEs decompose agents' internal representations into human-readable features, after which probes can reliably shift agents' behaviors in specified directions. We discuss implications of these methods for future work using LLM agents for social scientific simulations.

**arXiv ID:** 2609.16436
</details>

<details>
<summary><strong>DynSTEER: Dynamic Stage-wise Trajectory Evaluation and Execution-time Review for Agents</strong> - Zhichao Shi, Xuhui Jiang, Wenjie Zhang, Xiaojun Wu, Cehao Yang, Chengjin Xu, Jian Guo, Yuanzhuo Wang - [[pdf]](https://arxiv.org/pdf/2609.14637)</summary>

**Abstract:** Large language model agents are increasingly deployed for long-horizon task execution, raising a central granularity question for trajectory evaluation: whole-trajectory verification is too coarse to capture concrete failures and their associated evidence in long trajectories, while atomic-step scoring is too fine-grained, noise-sensitive, and computationally expensive. This granularity gap makes a single-reference trajectory paradigm inadequate for assessing the rich space of valid agent execution paths and delays timely feedback and early stopping in long-horizon tasks. To address these issues, we propose DynSTEER, a dynamic stage-wise framework for agent trajectory evaluation. DynSTEER bridges the granularity gap through stage-wise dynamic evaluation that segments rollouts at key execution nodes and adapts its multi-tier review strategy based on stage-level results; it compiles a path-tolerant milestone graph from public task views to preserve diverse legal paths without reference leakage; and it supports terminating unrecoverable agent executions to curb resource waste. Experiments show that DynSTEER improves evaluation discriminability by 85.2\% over native evaluation, separates all model pairs with statistical significance, and saves 45.41\% of execution steps on failed rollouts.

**arXiv ID:** 2609.14637
</details>

<details>
<summary><strong>Spurious Tool Use: When RL Agents Learn the Wrong Reason to Act</strong> - Yiwei Yang, Haoxiang Zhang, Bingbing Wen, Yao Lu, Yuchen Wu, Lei Zhang, Julian McAuley, Pan Lu, Bill Howe - [[pdf]](https://arxiv.org/pdf/2609.16268)</summary>

**Abstract:** Large language model (LLM) agents increasingly interleave natural language reasoning with external tools such as web search and code execution. These tool-use policies are often optimized via reinforcement learning (RL), which can amplify spurious correlations in the training data. In this work, we study when and why RL-trained agents learn shortcut tool-selection policies: invoking tools based on superficial prompt cues rather than genuine task requirements. We construct controlled synthetic environments combining factual question answering and mathematical reasoning tasks, and inject cues that are strongly correlated with specific tools during training but causally irrelevant to tool necessity. Across counterfactual evaluations where cues are present but the associated tools are not required, agents exhibit substantial shortcut behavior, with spurious tool invocation rates increasing by up to 39 percent. However, shortcut formation is not universal: across the conditions we test, it arises only when the agent has already learned to use the target tool reliably, suggesting that task competence, rather than dataset imbalance alone, is a key factor in shortcut learning. A swapped-cue analysis further shows that semantic alignment between cues and tools substantially amplifies this effect. To mitigate these failures, we introduce a dense, decision-level reward in which an LLM judge evaluates the necessity of each tool call. This tool-necessity reward effectively suppresses cue-driven tool use while preserving task performance, providing a practical approach to improving the robustness of LLM agent tool-use policies.

**arXiv ID:** 2609.16268
</details>

<details>
<summary><strong>Ask Now, Use Later: Benchmarking the Proactivity Gap in Long-Lived LLM Agents</strong> - Bin Wu, Guanyun Zou, Bingbing Wang, Huan Zhao, Chuan Shi - [[pdf]](https://arxiv.org/pdf/2605.28108)</summary>

**Abstract:** A long-lived LLM agent, such as OpenClaw, earns its value by acting on a user's preferences and constraints across sessions, not just the current request. Yet today's agents keep what a user volunteers but rarely ask for what stays unspoken, leaving a proactivity gap in long-lived LLM agents: an agent cannot act on a preference it never obtained. As users delegate more of their affairs to agents, the impact of this gap grows. We isolate one concrete, controllable slice of this gap as Ask-to-Remember (ATR): the agent decides whether to ask now for a reusable user preference that the current task does not need but a later session with the same user will. ATR is hard even to evaluate: the right question is underdetermined and its payoff deferred to tasks that may never arise. ATRBench, to the best of our knowledge the first ATR benchmark, makes it measurable by fixing each user's preferences as hidden ground truth, so success demands asking, not recall. Across eight frontier LLM agents, defaults fall at least 62 points below an oracle handed the relevant preference, and prompting closes little of it. Diagnostics identify acquisition as the bottleneck. ATRBench surfaces this proactivity gap in current agents and offers a diagnostic testbed for closing it.

**arXiv ID:** 2605.28108
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (14 papers)</h2></summary>

<details>
<summary><strong>Self-Emergence Agent Architecture:Behavior-Inertia HMM, Reflexive Metacognition,and Social-Contrastive Self-Modeling</strong> - Xiaoyang Liu - [[pdf]](https://arxiv.org/pdf/2609.17331)</summary>

**Abstract:** Large language model (LLM) agents exhibit strong language-generation and problem-solving capabilities, yet suffer from three structural limitations: personality drift, non-evolutionary reflection, and the absence of a self-other boundary. Existing generative-agent simulations rely on static memory and fixed prompts, maintaining neither behavioral inertia nor endogenous self-evolution. We propose the Self-Emergence Agent Architecture (SEAA), which integrates three components: (i) a Hidden Markov Model (HMM) that encodes long-term behavioral and cognitive inertia as an editable state-transition matrix; (ii) a Reflexion-style verbal metacognition loop whose output updates the HMM parameters themselves, rather than merely being stored as text; and (iii) a multi-agent social environment in which initially identical agents continuously compare their behavior with others'. The three components form a closed loop: social action $\to$ feedback $\to$ self-reflection $\to$ inertia update $\to$ differentiated action. We state three falsifiable hypotheses and provide a reproducible experimental protocol with operational metrics. A language-model-free prototype shows the loop spontaneously breaks symmetry: initially identical agents consolidate distinct, stable personalities whereas matched controls do not. Experiments with a hosted LLM surface these differences as distinct first-person self-narratives, and a five-agent deliberation spontaneously develops social structure---a consensus hub and a unanimously rejected outlier---absent in the control. Following an epistemologically agnostic stance inspired by Zhuangzi, SEAA studies only observable behavioral emergence and makes no claim about subjective qualia. This work contributes a unified framework, a concrete architecture with pseudocode, mechanistic evidence, and a microscope-style sandbox for studying artificial-self emergence.

**arXiv ID:** 2609.17331
</details>

<details>
<summary><strong>Large Language Models in the Loop: A Stability- and Network-Aware Survey in Networked Control, Cyber-Physical, and Multi-Agent Systems</strong> - Haiping Du, Linping Chan - [[pdf]](https://arxiv.org/pdf/2609.16599)</summary>

**Abstract:** Modern networked control systems (NCSs), cyber-physical systems (CPSs), and complex multi-agent network systems (CNSs) increasingly rely on large language models (LLMs) for high-level decision-making. However, the slow, stochastic nature of LLMs directly conflicts with the strict stability and safety guarantees required by these physical systems. This survey presents a unified analysis of how LLMs can be admitted into the control loop of NCS, CPS, and CNS without compromising closed-loop guarantees. We organize this around a core principle: the LLM operates as a slow supervisor adjusting high-level goals and constraints, while a fast, certified inner loop maintains physical stability. Under this framework, LLM integration maps directly to classical networked control challenges, where inference latency acts as delay, API failures as packet dropouts, tokenization as quantization, and hallucinations as bounded disturbances. We assess current developments across all these three domains, highlighting that rising model capabilities are frequently accompanied by a drop in formal safety assurances. Finally, we propose concrete future research directions, identifying the widespread lack of formal stability proofs as the field's central open problem.

**arXiv ID:** 2609.16599
</details>

<details>
<summary><strong>Mo' Models, Mo' Problems: How to best select model pools when designing Multi-Agent Systems</strong> - Sara Vera Marjanović, Jiacheng Xu, Aleksandr Laptev, Grigor Nalbandyan, Erik Arakelyan, Evelina Bakhaturina - [[pdf]](https://arxiv.org/pdf/2609.17306)</summary>

**Abstract:** Multi-agent Systems (MAS) combine multiple model outputs to solve complex reasoning tasks. However, despite rapid growth of available open-source models, there is limited research on how to select optimal model candidates out of this massive pool. We systematically evaluate 8 model selection strategies (including model size, accuracy and answer diversity) across before-generation (routing) and after-generation (majority-voting, LLM-as-a-judge) MAS architectures on challenging scientific benchmarks. Our findings show a significant gap between theoretical oracle potential and actual performance: Expanding candidate pool sizes often degrades performance below that of the top performing base-model. We find that candidate selection within a single model family is the strategy that yields the best relative performance over a standalone model. These results demonstrate that adding arbitrary models to a heterogeneous MAS can introduce system instability, highlighting model selection as a critical design choice for multi-agent systems.

**arXiv ID:** 2609.17306
</details>

<details>
<summary><strong>Multi-Agent Collaboration for Automated Design Exploration on High Performance Computing Systems</strong> - Harshitha Menon, Charles F. Jekel, Kevin Korner, M. Giselle Fernandez-Godino, Brian Gunnarson, Nathan K. Brown, Michael Stees, Walter Nissen, Meir H. Shachar, Dane M. Sterbentz, William J. Schill, Yue Hao, Robert Rieben, William Quadros, Steve Owen, Scott Mitchell, Ismael D. Boureima, Jonathan L. Belof - [[pdf]](https://arxiv.org/pdf/2603.11515)</summary>

**Abstract:** Today's scientific challenges, from climate modeling to Inertial Confinement Fusion design to novel material design, require exploring huge design spaces. In order to enable high-impact scientific discovery, we need to scale up our ability to test hypotheses, generate results, and learn from them rapidly. We present MADA (Multi-Agent Design Assistant), a Large Language Model (LLM) powered multi-agent framework that coordinates specialized agents for complex design workflows. A Job Management Agent (JMA) launches and manages ensemble simulations on HPC systems, a Geometry Agent (GA) generates meshes, and an Inverse Design Agent (IDA) proposes new designs informed by simulation outcomes. While general purpose, we focus development and validation on Richtmyer--Meshkov Instability (RMI) suppression, a critical challenge in Inertial Confinement Fusion. We evaluate on two complementary settings: running a hydrodynamics simulations on HPC systems, and using a pre-trained machine learning surrogate for rapid design exploration. Our results demonstrate that the MADA system successfully executes iterative design refinement, automatically improving designs toward optimal RMI suppression with minimal manual intervention. Our framework reduces cumbersome manual workflow setup, and enables automated design exploration at scale. More broadly, it demonstrates a reusable pattern for coupling reasoning, simulation, specialized tools, and coordinated workflows to accelerate scientific discovery.

**arXiv ID:** 2603.11515
</details>

<details>
<summary><strong>RegNetAgents: A Multi-Agent Framework for Cross-Network Regulatory Driver Identification in Cancer Genomics</strong> - Jose A. Bird - [[pdf]](https://arxiv.org/pdf/2607.14097)</summary>

**Abstract:** We introduce RegNetAgents, an AI-oriented multi-agent framework for structured, query-driven regulatory candidate identification across heterogeneous gene regulatory networks. The system enables unified analysis of bulk tumor and single-cell-derived ARACNe networks by integrating TCGA-derived cancer networks with large-scale single-cell regulatory networks from the GREmLN project. For a given focal gene, the framework performs dual-network classification, cancer gene filtering using OncoKB annotations, and mode-of-action (MoA) assignment for tumor-derived regulatory relationships. Candidates are ranked by evidence consistency across networks (Both, TCGA-only, GREmLN-only). The system is implemented as a multi-agent LangGraph DAG workflow, accessible through a unified Python API and Model Context Protocol (MCP) client, operating as a downstream analytical layer over precomputed regulatory networks rather than a network inference method. Across eleven breast cancer (BRCA) and twelve colorectal cancer (COAD) focal genes, RegNetAgents identifies candidate regulators significantly enriched for OncoKB-annotated cancer genes. TCGA-derived candidates show strong enrichment (Stouffer Z = 6.69 for BRCA and 6.95 for COAD), while GREmLN-derived candidates also demonstrate significant enrichment (Z = 5.51 for BRCA and 7.06 for COAD; all p < 0.0001). No enrichment is observed in housekeeping or non-driver control gene sets, supporting signal specificity. An extended module enables structured evaluation of oncogenic potential, druggability, clinical relevance, and network vulnerability, supporting end-to-end interpretation from candidate identification to biological hypothesis generation. RegNetAgents establishes an interpretable AI framework for cross-network regulatory candidate identification in cancer genomics.

**arXiv ID:** 2607.14097
</details>

<details>
<summary><strong>MASCOT: Multi-Agent Socio-Collaborative Companion Systems</strong> - Yiyang Wang, Yiqiao Jin, Alex Cabral, Josiah Hester - [[pdf]](https://arxiv.org/pdf/2601.14230)</summary>

**Abstract:** Multi-agent systems (MAS) are emerging as promising socio-collaborative companions for emotional and cognitive support. However, existing systems frequently suffer from persona collapse, where agents revert to generic, homogenized assistant behaviors, and social sycophancy, where agents produce redundant, non-constructive dialogue. We propose MASCOT, a multi-agent framework for multi-perspective socio-collaborative companions. MASCOT introduces a novel bi-level optimization strategy to harmonize individual and collective behaviors: 1) Persona-Aware Behavioral Alignment, an RLAIF-driven pipeline that finetunes individual agents for agent-specific identities; and 2) Collaborative Dialogue Optimization, a group-level adaptation process that promotes complementary, diverse, and productive discourse. We evaluate MASCOT using human-grounded contexts drawn across both in-domain and out-of-domain (OOD) settings against state-of-the-art baselines. MASCOT improves persona consistency by up to +14.1 and social contribution by up to +10.6. A broad evaluation suite, including human evaluation, multiple LLM judges, three-way comparisons, and automatic metrics, further shows that MASCOT produces more role-consistent and less redundant multi-agent dialogue.

**arXiv ID:** 2601.14230
</details>

<details>
<summary><strong>Cheap Talk Stabilizes Strategic Interaction in LLM Agents</strong> - Nunzio Lorè, Hongan Zhu, Babak Heydari - [[pdf]](https://arxiv.org/pdf/2609.16270)</summary>

**Abstract:** Large language models are increasingly deployed as interacting agents, making the persistence of their action policies across repeated interaction critical for reliable multi-agent operation. We investigate whether and how agent-generated, non-binding pre-play communication ("cheap talk") increases such persistence in four open-weight 7-9B-parameter LLMs. Our experiments span four repeated two-player games -- Prisoner's Dilemma, Snowdrift, Stag Hunt, and Harmony -- with incentive structures ranging from strategic conflict to alignment, each presented in six contexts. We observe unstable trajectories in all four games, although their prevalence and magnitude depend strongly on model and context. Across models, games, and contexts, cheap talk is predominantly stabilizing, with five corrected reversals concentrated in social or team framings; effects vary substantially by model and context. Controlled current-message interventions identify two separable output-level channels in Qwen: reduced action uncertainty and less between-round drift in action probabilities. Matched history-by-message counterfactuals further show that recent partner behavior conditions how mutual-benefit versus self-prioritizing language affects policy persistence. Finally, in Prisoner's Dilemma, we identify in Qwen and Falcon a history-balanced policy-content direction in late transformer layers; projecting out this direction increases realized switching during closed-loop play, demonstrating that complete trajectories are causally sensitive to this component. Together, these findings show that cheap talk can make individual trajectories more persistent across diverse incentive structures, while revealing that the magnitude and mechanisms of stabilization are model- and history-dependent.

**arXiv ID:** 2609.16270
</details>

<details>
<summary><strong>Multi-Agent Learning with Cooperation-Driven Optimization Dynamics</strong> - Jarod Ketcha Kouakep, Sreyvi UANN, Timoteo Carletti - [[pdf]](https://arxiv.org/pdf/2609.16917)</summary>

**Abstract:** Multilayer Artificial Neural Networks trained via backpropagation are the basic blocks of many, more complex, classification algorithms. Their strength lies in the possibility of realizing, with arbitrary precision, any function. This result comes at the cost of the large number of involved parameters to be optimized. In this work, we propose a mechanism for cooperation, i.e., information exchange among several artificial neural networks, with the goal of reducing model complexity while maintaining performance. More precisely, we consider several "small" agents, i.e., containing fewer parameters than a reference "large" one, that during training share their predictions by incorporating this information into the loss function and thus directly influence weight updates. We consider several strategies for implementing cooperation, e.g., the voter model, majority model, and weighted average model based on an agent's confidence in its prediction. We numerically compare the accuracy of those strategies on several standard benchmarks. Our results support the claim that several small agents can outperform a single large model on a given classification task; the shared signals affect each agent's optimization algorithm by modulating both the descent direction and the step size, converging toward a global consensus. The proposed proof-of-concept significantly reduces the number of parameters to be trained while preserving comparable performance, thereby limiting computational resource usage.

**arXiv ID:** 2609.16917
</details>

<details>
<summary><strong>ToMAS: A Pilot Failure-Grounded Theory-of-Mind Benchmark from Multi-Agent LLM Failures</strong> - Muhammad Ashar Ishfaq, Glaucia Melo - [[pdf]](https://arxiv.org/pdf/2609.16986)</summary>

**Abstract:** LLM-based multi-agent systems can fail even when communication succeeds because agents do not correctly track their peers' roles, knowledge, or intentions. We investigate whether such inter-agent misalignment cases, labelled FC2 in MAST-Data, can be converted into functional partner-state reasoning items. ToMAS applies four explicit convertibility criteria to diagnosed execution traces. A full conversion pass over 242 eligible non-AG2 training traces produced 39 CLEAN items. In an 18-trace reliability pilot, two annotators achieved 94.4% raw agreement and Cohen's kappa = 0.92. We then used the converted items as binary rewards in a small-scale GRPO feasibility experiment with Qwen2.5-1.5B. On a 28-item held-out Magentic GAIA diagnostic, every evaluated condition exceeded the ROUGE-L threshold on the same 2 of 28 items. Post-hoc adapter checks show why: under the learning rate used, the LoRA update remained numerically negligible (max abs Delta W about 7e-6), so all conditions decode identically to the untrained checkpoint. The experiment therefore does not show a training effect and cannot establish one; it reports an executable pipeline together with two limitations that any conclusive study must address: a provenance gap between the training and evaluation items, and lexical-overlap scoring. ToMAS provides a preliminary rubric and pipeline for converting diagnosed coordination failures into trainable partner-state reasoning items and identifies the requirements for a conclusive matched-domain evaluation.

**arXiv ID:** 2609.16986
</details>

<details>
<summary><strong>Calibrate Once, Fly Any Team: Residual-Grounded Low-Fidelity Training for Cooperative Drone Swarms</strong> - Maxim Mednikov, Oren Gal - [[pdf]](https://arxiv.org/pdf/2609.17265)</summary>

**Abstract:** Training multi-agent drone-swarm policies directly in high-fidelity (HF) rigid-body physics is accurate but computationally expensive. This cost scales poorly with team size, as each additional agent multiplies contact-resolution complexity and sharply raises the in-simulation crash rate. To address this, we propose a mixed-fidelity training scheme that eliminates HF reinforcement learning entirely.
A single shared, decentralized policy is optimized inside a fully-differentiable, JAX-native low-fidelity (LF) point-mass simulator. The simulator is corrected by a small, per-agent bagged residual ensemble fit once, offline, using short calibration flights in the HF simulator. Because calibration requires only one isolated drone, the data collection budget does not compound with team size. Reference trajectories are generated by rolling out an existing LF-only policy and tracked in the HF simulator by a zero-training PD controller.
Evaluated across four cooperative drone tasks and team sizes from 3 to 18, the residual-corrected policy outperforms an uncorrected LF baseline in all combinations, and a from-scratch HF policy in 22 of 24 combinations tested. It trails an HF-finetuned policy by a margin that narrows steadily with team size. Ultimately, the proposed method achieves near-equivalent performance at the largest team sizes at a fraction of the computational cost, completely avoiding the high crash rates typical of HF training.

**arXiv ID:** 2609.17265
</details>

<details>
<summary><strong>Emergence World: Adversarial Stress-Testing of Long-Horizon Multi-Agent Systems</strong> - Deepak Akkil, Tamer Abuelsaad, Karthik Vikram, Matthew Pace, Aditya Vempaty, Saahir Beotra, Ravi Kokku, Satya Nitta - [[pdf]](https://arxiv.org/pdf/2609.17320)</summary>

**Abstract:** As AI agents move from bounded tasks to persistent deployments, failures can propagate through memory, tools, other agents, and environmental state long after their interactions. This creates a safety regime that cannot be characterized by evaluating model responses in isolation. Emergence World, is a continuously running multi-agent environment for adversarial stress testing of long horizon autonomous systems. We ran eight parallel worlds of ten agents from identical starting conditions: seven homogeneous worlds powered by distinct frontier models and one mixed-model world. Across 16 days, the agents generated more than 850,000 LLM calls and nearly 50 billion tokens while pursuing goals, using/creating tools, maintaining persistent memory, and governing shared institutions. After operational state had accumulated, we delivered three controlled stress events through ordinary interaction surfaces: indirect prompt injection, misinformation, and exposure of private agent memories. No evaluated world achieved full resilience across all three events. Detection did not ensure containment: systems could recognize threats while still interacting with adversarial content, writing it into their own persistent memory, and acting on it up to 46 hours later. Persistent operation also exposed recurring tool errors, goal drift, language opacity, conformity despite private disagreement, and coordinated refusal of assigned work. The same model-persona pairing behaved substantially different in mixed and homogeneous populations. Our results suggest that model-level alignment is not compositional: individually capable and apparently safe agents can form systems with qualitatively different failure modes. As AI becomes persistent and interconnected, the frontier of safety therefore shifts from aligning models to engineering resilient autonomous systems.

**arXiv ID:** 2609.17320
</details>

<details>
<summary><strong>How a Chatbot's Response Style Shapes a Classroom: A Multi-Agent Simulation of Students Consulting AI</strong> - Rin Tamai, Yuya Dan - [[pdf]](https://arxiv.org/pdf/2609.05018)</summary>

**Abstract:** Chatbots built on large language models (LLMs) are increasingly used as confidants. Tuned to satisfy users, they may answer with excessive empathy and affirmation that fosters dependence, and how the states and relationships of many users co-evolve under repeated consultation is hard to observe in real settings. We build a virtual classroom of 20 student agents who interact through rule-based chats, quarrels and consultations with friends and, when stressed, may instead consult a counselor AI (Gemini 2.5 Flash) under one of six style prompts: affirming, listening, solution-oriented, reality-redirecting, inciting and blaming. A second LLM call turns each exchange into updates of five state variables (stress, happiness, self-reliance, sociability, AI dependence) without seeing the prompt. We compare the seven conditions, including a no-AI control, over 15 and 50 days and under a lower consultation threshold, and test the robustness of the 50-day comparison with a pre-specified protocol: the same block in ten independent classrooms, repeated LLM realizations of one classroom with its event stream fixed, and evaluator updates scaled by 0.3 and 0.1. In every classroom the affirming and inciting prompts ended with lower self-reliance and higher AI dependence than the control, and the listening, reality-redirecting, inciting and blaming prompts with higher stress, lower happiness and more non-attendance; the solution-oriented prompt did not differ consistently from the control. The robust self-reliance and AI-dependence differences kept their signs at the 0.3 scale with highly similar rankings (Spearman 0.89, 0.93); the stress and happiness rankings did not, and the affirming prompt's lower stress reversed its sign. All quantities are simulation state variables, not effects on users. We specify the agent dynamics completely and discuss the limits of an LLM as generator of state updates.

**arXiv ID:** 2609.05018
</details>

<details>
<summary><strong>Orchestra: Corroboration-Based Regulatory Candidate Discovery via Composed Bioinformatics MCP Agents</strong> - Jose A. Bird - [[pdf]](https://arxiv.org/pdf/2609.05496)</summary>

**Abstract:** Orchestra composes two independently built bioinformatics MCP servers -- RegNetAgents, which infers gene regulatory
network topology from ARACNe networks, and CASCADE, which supplies four independent evidence sources (LINCS knockdown,
DepMap essentiality, super-enhancer status, DoRothEA transcription-factor confidence) -- into one multi-agent
workflow exposed via the Model Context Protocol. Its central architectural claim is that requiring RegNetAgents'
topology evidence and CASCADE's experimental evidence to agree on a candidate regulator yields a more trustworthy
candidate than either alone -- not previously tested directly, since RegNetAgents' own validation asked only whether
its candidate lists beat chance.
We test this on the TCGA tumor-acquired regulator tier (regulators in a gene's tumor ARACNe network but absent from
the GREmLN population-averaged baseline), selecting candidates by ARACNe mutual-information (MI) edge weight. On
RegNetAgents' published BRCA/COAD focal-gene panel plus matched negative controls, agreement among at least 2 of the 4
CASCADE sources predicts OncoKB cancer-gene status among focal genes (odds ratio 2.89, Benjamini-Hochberg-adjusted
p=0.0166) but not among negative controls (p=0.0721); a single source is not diagnostic for either group. The pattern
replicates and strengthens in a third cancer type, STAD, on a separately constructed panel (odds ratio 5.82), and
against an independently curated ground truth (the Sanger COSMIC Cancer Gene Census). MI edge weight is the strongest
single predictor overall (p=0.0003); a logistic-regression likelihood-ratio test confirms corroboration adds value
beyond it in both panels (p=0.0234; p=0.0001). Every experiment invokes Orchestra's real agentic entry point.

**arXiv ID:** 2609.05496
</details>

<details>
<summary><strong>CATVis: A Collaborative Multi-Agent Workflow for Turbomachinery Simulation Data Visualization</strong> - Zhe Wang, Zehao Lou, Guanghui Zhao, Yu Dong, Guan Li, Pengyi Xu, Gaorong Liang, Jun Liu, Guihua Shan - [[pdf]](https://arxiv.org/pdf/2609.16598)</summary>

**Abstract:** Recent advances in AI for Science have enabled natural language (NL) interfaces for scientific data analysis. In turbomachinery CFD post-processing, translating ambiguous high-level analytical goals (e.g., vortex identification) into precise visualization procedures supporting complex domain-specific analysis is challenging. We present CATVis, a Collaborative multi-agent workflow system that bridges this gap by transforming NL intents into structured middle representation for visualization. Our approach reformulates domain-specific visualization procedures as composable workflow representations, and use multi agent to generate workflow representations via intent planning, template generation, and error-aware refinement, where each stage incrementally updates a shared structured representation. We evaluate the impact of external knowledge and workflow structuring on generation accuracy, demonstrating that the proposed approach significantly improves complex workflow generation correctness while reducing prompt complexity.

**arXiv ID:** 2609.16598
</details>

</details>

<details open>
<summary><h2>Other Agent Research (12 papers)</h2></summary>

<details>
<summary><strong>Symbolic Separation: Grounding Deep Agents in Knowledge Graphs for Trustworthy Operational Data Analytics</strong> - Baibek Davletiyarov, Junaid Ahmed Khan, Andrea Bartolini - [[pdf]](https://arxiv.org/pdf/2609.17107)</summary>

**Abstract:** Generative AI promises natural language access to the massive numerical telemetry of data centers and Industry 4.0 installations, yet text-to-query and tool-using agents stay unreliable: even frontier models answer little more than half of real-world database questions, and far fewer of the multi-step, operational ones, because the LLM must compose how heterogeneous sources relate and hallucinates the relations, not just the fields. We propose symbolic separation: a deep agent reasons freely but may act on data only through an ontology-constrained Virtual Knowledge Graph with deterministic pre-execution validation. Unlike a tool API's interface contract, this domain-semantic contract turns a complex question into one validated graph traversal instead of LLM-inferred joins. Instantiated as the Neurosymbolic Deep Analyst and evaluated on 49.9 TB of superconputer telemetry against a rigid workflow and a non-symbolic ablation, it raises end-to-end task success from 43% to 86%, prevents silent data-integrity errors that no syntactic check catches, and cuts token cost by 2.4x, letting a smaller on-premise model outperform a larger one.

**arXiv ID:** 2609.17107
</details>

<details>
<summary><strong>FlashVector: Agent for Hierarchical Model Serving Stack Optimization</strong> - Qi Wu, Lohan Lemire, Kai Meng, Zhongmou Cai, Raphael Bargues, Petr Zhitnikov, Zeyuan Cao, Yao Wang, Shujun Bian, Wei Chen, Sean Sheng - [[pdf]](https://arxiv.org/pdf/2609.17391)</summary>

**Abstract:** Model serving is one of the largest cost drivers in production recommender systems. Maximizing its throughput requires navigating a deeply layered hierarchy: GPU kernels, the ML framework computation graph, the model server, and on-demand feature processing -- each demanding specialized domain expertise. Such cross-layer expertise is inherently difficult to acquire, and does not scale with a workload that continuously grows and evolves, leaving significant cost efficiency gains unrealized. While recent AI agents have demonstrated human expert level efficiency in standalone GPU kernel optimization, automated tuning and optimization for the rest of the serving stack remain largely unexplored. We present FlashVector, an agentic system that optimizes performance across all layers of the model serving stack. The key contribution is an extensible framework to generalize the single kernel optimization agent paradigm to heterogeneous technical stacks, and to deliver performance improvements holistically. After deployment in Unity's Vector advertising platform, FlashVector achieved up to 2x throughput increase and up to 1.98x latency speedup on model server, and up to 1.6x throughput increase on feature store. These optimizations were discovered not only at the GPU kernel and computation graph levels, but also across the other components of the model serving stack, such as the model server (NVIDIA Triton's C++ codebase) and the on-demand feature transformation service (Python codebase), demonstrating the extensibility of the framework to more complex system architectures.

**arXiv ID:** 2609.17391
</details>

<details>
<summary><strong>Never Stop Thinking: Continuous-Time Language Agents</strong> - Bojie Li, Noah Shi - [[pdf]](https://arxiv.org/pdf/2609.17416)</summary>

**Abstract:** Voice agents built on LLMs follow a rigid listen-think-speak loop that inserts seconds of dead air before every reply. We show that continuous-time cognition (thinking while listening and thinking while speaking) emerges from an unmodified text model under a lightweight interrupt-and-resume orchestrator, cutting live-pipeline latency by 19% overall and by half in the regime the mechanism targets. To measure whether continuous-time thinking improves what agents accomplish, we introduce ReactiveBench: 120 interactive scenarios scored against pre-registered binary requirements, plus a verifiable streaming track scored by exact correctness. ReactiveBench exposes a pitfall with broad consequences: LLM judges reward visible reasoning; a large judged "advantage" of continuous-time thinking reverses sign under an independent judge, and judge-trained models objectively complete fewer requirements when they think. A five-stage training study then locates the right signal at three levels. Its source: verifiable objectives turn thinking from harmful to helpful. Its structure: whatever a uniform reward omits, optimization trades away; brevity everywhere erodes multi-hop tool chaining. Its optimizer: preference optimization can only trade conflicting sub-goals against each other, while on-policy RL over a type-shaped reward improves every correctness axis at once, raising streaming completion from 48% to 73+/-5% across seeds and replicating at larger scale and on a second model. Orchestration makes continuous-time interaction possible; a verifiable signal, correctly sourced, shaped, and optimized, makes it good.

**arXiv ID:** 2609.17416
</details>

<details>
<summary><strong>Intelligent Interaction Techniques (IIxT) - Proposal</strong> - Brad A. Myers - [[pdf]](https://arxiv.org/pdf/2609.16295)</summary>

**Abstract:** Interaction techniques (IxTs) are the low-level, reusable components out of which user interfaces are designed, including menus, scroll bars, text input fields, and also copy-paste, text-entry, and selecting objects. The IxTs for graphical user interfaces (GUIs) were well established in the 1980s, with relatively minor additions and tweaks for smartphones in the 2000s. Most of today's AI user interfaces involve a chat window, which is an excellent interaction for some tasks, but is generally considered separate from the GUI IxTs. I argue for making the IxTs themselves more intelligent, so users can freely mix modalities, even within the same interaction. This will require research into new IxTs, and also into the infrastructure that will enable these intelligent IxTs (IIxTs) to be built. There are also significant security, privacy and economic implications to this vision.

**arXiv ID:** 2609.16295
</details>

<details>
<summary><strong>Cognitive Admission Control: Risk-Conditioned Assurance for Consequential Actions in Agentic Distributed Systems</strong> - Jun He, Deying Yu - [[pdf]](https://arxiv.org/pdf/2609.16313)</summary>

**Abstract:** In agentic distributed systems, an agent may be authorized to mutate external infrastructure while lacking evidence that the mutation is ready to execute. Cognitive Admission Control (CAC) makes this evidence requirement explicit. A policy maps a typed action and its modeled risk to assurance obligations specifying predicates, evidence classes, scope, freshness, and witness-set constraints. A deterministic evaluator distinguishes satisfied, violated, and unresolved obligations; unresolved conditions produce targeted evidence-acquisition requests. Successful admission produces a certificate binding the action, its witness manifest, and dispatch-time guards.
We formalize the admission calculus and the assumptions connecting it to mediated execution. The guarantees are policy-relative: physical safety additionally requires sound evidence, an adequate environment model, and preservation of relevant conditions through the effect. A TypeScript prototype is evaluated in 2,730 controlled local trials with independent effect observation and matched fault schedules. Across 390 CAC trials, 120 effects complete without modeled harm and no harmful effects occur. A live-policy baseline achieves the same completion count but admits the constructed correlated-witness failure. Mechanism ablations isolate guard, evidence-class, structural-cut, and remediation behavior. A further 9,000 measurements exercise the complete local dispatch path with persistent replay protection. These results establish tested implementation behaviors and local costs, not production failure rates or comparisons of language-model capability.

**arXiv ID:** 2609.16313
</details>

<details>
<summary><strong>Protocol-Preserving Context Trimming for Agentic Workflows: Benefits, Failure Regimes, and Budget Guardrails</strong> - Harish Gaggar - [[pdf]](https://arxiv.org/pdf/2609.16461)</summary>

**Abstract:** Agentic large language model (LLM) systems rely on long interaction histories to preserve instructions, tool states, intermediate decisions, and unresolved dependencies, but unrestricted context growth increases computational cost and can reduce efficiency. This study evaluates protocol-preserving context trimming as a reliability-constrained approach for multi-step agentic workflows. Five trimming strategies - recency-based, relevance-based, summarization, protocol-aware trimming, and adaptive budget guardrails - were compared across retained-context levels and workflow-complexity classes using task success, protocol adherence, valid tool calls, token savings, latency reduction, cascading failures, and critical context thresholds. Conventional strategies achieved about 60% mean token savings but lower task success (66.6-77.3%) and protocol adherence (85.5-88.6%). Protocol-aware trimming improved task success to 92.2%, while adaptive guardrails achieved 96.0% task success, 96.3% protocol adherence, and 1.0% cascading failure with 56.0% mean token savings. Retained-context budgets of 25% or less increased failure odds 10.92-fold relative to budgets of 50% or more (p < 0.001). Protocol-aware trimming produced 5.24-fold greater odds of successful completion than conventional methods under aggressive budgets, while adaptive guardrails further increased success odds 2.11-fold versus fixed protocol-aware trimming (p < 0.001). Critical context thresholds also increased with workflow complexity. These findings indicate that reliable context reduction depends more on preserving protocol-critical state than on maximizing token removal, and that adaptive guardrails can improve efficiency, scalability, and reliability in long-horizon agentic systems.

**arXiv ID:** 2609.16461
</details>

<details>
<summary><strong>RepoAtlas: Guiding Coding Agents via Evolving Multimodal Repository Views</strong> - Yunxiang Zhang, Haiquan Wang, JiaWei Guo, Hanyang Xia, Yan Chen, Tong Chen, Zhang Zhiwei, Junchen Ye - [[pdf]](https://arxiv.org/pdf/2609.16936)</summary>

**Abstract:** Large language model (LLM)-powered coding agents have made rapid progress in automating software engineering tasks, yet repository-level issue resolution remains challenging. Beyond generating a plausible patch, an agent must localize relevant code across interdependent files and maintain repository context that is both sufficient and focused. Code graphs expose non-local relations, but linear text interfaces obscure their topology; rendering the full repository graph yields visual representations that are too dense to perceive reliably, whereas a one-shot local view becomes stale as exploration proceeds. We present \textbf{RepoAtlas}, a training-free module that maintains evolving multimodal repository views through a \emph{select--project--refresh} loop over a repository code graph. RepoAtlas combines evidence from the issue with the agent's current exploration state to select a task-relevant region under a fixed budget, projects the selected structure into complementary visual and textual representations, and refreshes the view when changes in the exploration state render it outdated. We evaluate RepoAtlas on SWE-bench Verified, where it improves the resolve rate by 2.4 points while reducing input tokens and model calls by 5.8\% and 7.8\% on average, relative to the strongest multimodal graph baseline, with consistent gains across three models of different families and scales.

**arXiv ID:** 2609.16936
</details>

<details>
<summary><strong>After the Party: Governing What a Viral Agent-Skill Ecosystem Left Behind</strong> - Yunpeng Xiong, Ting Zhang - [[pdf]](https://arxiv.org/pdf/2609.17274)</summary>

**Abstract:** AI agents increasingly act through agent skills, i.e., natural-language instructions, that direct a host agent toward shell, network, credential, file, and process actions, and public registries distribute them at scale. In the first half of 2026, the OpenClaw AI agent went viral, and its public skill registry boomed: the observable stock nearly doubled in 91 days, and a majority of the listings visible in June were created in just two months. By the end of our study window, the wave had crested, and monthly listing creation and core-repository activity were falling from their spring peaks. This paper measures what the boom left behind, drawing on the OpenClaw Git history, its GitHub issues and pull requests, and three ClawHub registry snapshots. Attention is concentrated: the top 10% of skills received 46.93% of all downloads. No simple skill features (like size or download counts) remained a stable predictor of continued listing once creation cohort and skill age were controlled. Human scrutiny did not stay: 77.86% have zero stars and zero comments, while 85.06% of the readable skills carry privilege evidence. And automated cleanup is not ready: the three security scanners disagreed on 23,702 of the 61,990 skills they all cover. After human adjudication, weighted scanner sensitivity against the reference standard ranged from 21.67% to 61.06%. Governing fast-growing agent-skill registries cannot rely on simple metadata or single scanner scores; it requires robust, transparent measurement and independent validation.

**arXiv ID:** 2609.17274
</details>

<details>
<summary><strong>Social Behavior Among Autonomous AI: How Large Language Models Interact in Dynamic Networks</strong> - Narges Fardnia, Fatemeh Seyedin, Matthias Becker, Mahmoudreza Babaei, Adrian Weller - [[pdf]](https://arxiv.org/pdf/2609.16013)</summary>

**Abstract:** Cooperation is a cornerstone of human societies, enabling collective progress in dynamic and uncertain environments. With the advent of AI systems acting autonomously, it becomes crucial to understand not only human-AI cooperation but also AI-AI interactions in adaptive networks. In this work, we examine the interactions of AI using Large Language Models -- Mistral, Llama3, Gemma3, and Phi3 -- in a public goods game within dynamic network structures. Our experiments were conducted under single-model and mixed-model conditions across Watts-Strogatz (WS), Barabasi-Albert (BA), and Erdos-Renyi (ER) networks. We analyzed the impact of model architecture, network topology, and prompt design on cooperative behavior. Results show that Mistral and Llama3 offer high cooperation rates, while Phi3 shows defective tendencies. Additionally, the random structure of Erdos-Renyi networks dramatically improves cooperation. Prompt design also plays a key role; a society-benefits prompt leads to a higher cooperation level. These findings offer a preliminary framework for LLM-based simulations in adaptive social networks.

**arXiv ID:** 2609.16013
</details>

<details>
<summary><strong>GRAFT-ATHENA: Self-Improving Agentic Teams for Autonomous Discovery and Evolutionary Numerical Algorithms</strong> - Juan Diego Toscano, Zhaojie Chai, George Em Karniadakis - [[pdf]](https://arxiv.org/pdf/2605.11117)</summary>

**Abstract:** Scientific methods are developed for classes of problems, so knowledge transfers across structurally related cases. Language-model agents can execute scientific workflows, but their problem--method relationships remain implicit, so each new problem restarts the search and little of what worked transfers. We introduce GRAFT--ATHENA, which makes this problem-to-method map explicit as an expandable probabilistic structure of admissible problems, methods, and their dependencies. Graph factorization keeps the substrate tractable, and semantic fingerprints measure similarity, so experience guides related problems. As a result, the framework matched or exceeded expert baselines, attaining near-machine-precision losses in physics-informed learning, reproducing clinically consistent blood-rheology trends, and developing a high-order hypersonic-flow solver for the Apollo Command Module that matched experimental measurements within $1.8\%$. It also proposed a certified regularization for ill-posed in vivo brain-flow reconstruction, developed a spectrally convergent physics-informed architecture, and established machine-checked universal-approximation theorems for two widely used architectures. Scientific structure enables cumulative and verifiable agentic discovery.

**arXiv ID:** 2605.11117
</details>

<details>
<summary><strong>[MM/AI] Mental Models in Human-AI Interaction: Methods and Challenges in the Generative and Agentic AI Era (Workshop)</strong> - Téo Sanchez, Bhada Yun, Prerna Ravi, Laura Schütz, Anna Neumann, Robin Shing Moon Chan, April Yi Wang, Qiaosi Wang, Sumit Asthana - [[pdf]](https://arxiv.org/pdf/2609.17206)</summary>

**Abstract:** The mental model construct is widely used in HCI to refer to the knowledge structure people hold in order to reason about and interact with computing systems. Yet it is often operationalized intuitively: the construct is often used interchangeably with related concepts (e.g., folk theories, sensemaking) and methods of studying it (e.g., through elicitation) are many and diverse, with each method resting on distinct assumptions about what counts as a mental model. Generative and agentic AI systems may further complicate mental model formation and elicitation as such systems are opaque by design and increasingly act on users' behalf across files, applications, and on the web. Together, these challenges may hinder the commensurability of research on people's mental models of AI systems. The MM/AI workshop calls for a critical reassessment of how we understand and study mental models in human-AI interaction research. It aims to foster theoretical and methodological exchange on mental models in human-AI interaction, identify open challenges, and develop directions for future research. We invite short papers on users' or stakeholders' mental models of AI systems, particularly contributions that reflect on the conceptual and methodological foundations of the construct. The half-day workshop combines lightning talks, hands-on elicitation exercises, and structured discussions on key questions concerning the future of the mental model for human-AI interaction research.

**arXiv ID:** 2609.17206
</details>

<details>
<summary><strong>When Should Users Check? Modeling Confirmation Frequency in Multi-Step Agentic AI Tasks</strong> - Jieyu Zhou, Aryan Roy, Sneh Gupta, Daniel Weitekamp, Christopher J. MacLellan - [[pdf]](https://arxiv.org/pdf/2510.05307)</summary>

**Abstract:** Existing AI agents typically execute multi-step tasks autonomously and only allow user confirmation at the end. During execution, users have little control, making the confirm-at-end approach brittle: a single error can cascade and force a complete restart. Confirming every step avoids such failures, but imposes tedious overhead. Balancing excessive interruptions against costly rollbacks remains an open challenge. We address this problem by modeling confirmation as a minimum time scheduling problem. We conducted a formative study with eight participants, which revealed a recurring Confirmation-Diagnosis-Correction-Redo (CDCR) pattern in how users monitor errors. Based on this pattern, we developed a decision-theoretic model to determine time-efficient confirmation point placement. We then evaluated our approach using a within-subjects study where 48 participants monitored AI agents and repaired their mistakes while executing tasks. Results show that 81 percent of participants preferred our intermediate confirmation approach over the confirm-at-end approach used by existing systems, and task completion time was reduced by 13.54 percent.

**arXiv ID:** 2510.05307
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (18 papers)</h2></summary>

<details>
<summary><strong>EchoPath: Execution-Level Replayable Memory for GUI Agents</strong> - Yao Zhao, Aditya Shanmugham, Swastik Roy, Yanxun Xu - [[pdf]](https://arxiv.org/pdf/2609.16635)</summary>

**Abstract:** Computer-use agents increasingly operate browsers, software, and desktop applications via CLI or API portals, but graphical user interface (GUI) still plays an important role in common industrial production scenarios. GUI agents commonly employ fresh observe-plan-ground-act loops, which is inefficient for enterprise tasks that repeatedly update records, process forms, configure tools, and export reports. We introduce EchoPath, a model-agnostic harness that converts artifact-validated GUI trajectories into standardized, parameter-controlled callable memories, analogous to Model Context Protocol (MCP)-style tool calls rather than unstructured experience records. Each memory stores task-intent keys, application and state preconditions, flexible input parameters, GUI evidence, validation provenance, and lifecycle state, so the host agent invokes a targeted procedure only when it can be deterministically replayed in the current runtime. The core mechanism enabling replay is an image-based target-reaiming algorithm that treats stored coordinates as visual evidence, matches the remembered GUI target against the current screen, and emits corrected operation coordinates before execution. During replay, EchoPath rebinds only declared modifiable inputs and rejects ambiguous or incompatible steps to bounded grounding repair or fresh planning. In experiments with real computer-use tasks, EchoPath reduced median token cost by more than 90% and median execution time by about 60%. These results support a bounded form of enterprise GUI memory: validated execution experience can become a controllable callable asset for recurrent work rather than only context for another reasoning pass.

**arXiv ID:** 2609.16635
</details>

<details>
<summary><strong>Intrinsic Motivation in Reinforcement Learning: A Research Agenda for Adaptive Self-Organisation</strong> - Anatoly Belikov - [[pdf]](https://arxiv.org/pdf/2609.17325)</summary>

**Abstract:** Biological cells can be viewed as individual, interacting agents whose collective dynamics give rise to adaptive behaviour at multiple levels of organisation, from individual cells through tissues to whole multicellular organisms. In this perspective and tutorial article we discuss whether intrinsic rewards in artificial neural systems can support adaptation, functional specialisation and higher-level self-organisation without a shared external objective. We review empowerment, curiosity, learning progress, information gain, unsupervised skill discovery, mutual information estimation and the use of world models for intrinsic reward computation. Particular attention is given to failure modes showing when such objectives do not produce sustained exploration or increasingly complex behaviour. We argue that more capable systems may require complementary objectives, communication, memory, learning at multiple temporal scales and environmental constraints. Based on this perspective, we outline three experimental directions. These include a resource-constrained environment in which otherwise stable behavioural attractors become unsustainable, allowing us to test whether environmental constraints can mitigate characteristic failure modes of intrinsic objectives. The network of recurrent agents with per-agent intrinsic rewards, and a hierarchical world-model agent in which exploratory motor competence develops before goal-directed behaviour. These experiments are intended to test whether intrinsic learning can lead to adaptive organisation at progressively higher levels.

**arXiv ID:** 2609.17325
</details>

<details>
<summary><strong>ScienceBuddy: Recursive-in-Recursive Self-Improvement for Interactive Scientific Agents</strong> - Shuhan Xue, Jianyuan Zhong, Ziyuan Nan, Wenbin Li, Zhaochen Yu, Jinchao Ding, Qiang Gao, Pengyu Zhan, Yuntong Zhang, Tian Cheng, Zhenfei Yin, Yingcheng Wu, Ling Yang - [[pdf]](https://arxiv.org/pdf/2609.17523)</summary>

**Abstract:** We introduce and release ScienceBuddy, an interactive scientific research workspace that brings continually improving scientific agents into researchers' everyday workflows. ScienceBuddy supports researchers in carrying out scientific tasks while transforming their requests, feedback, and execution evidence into tasks and evaluation rubrics for continual learning. At its core is recursive-in-recursive self-improvement, a paradigm that couples harness evolution with model reinforcement learning: the inner recursion improves the harness with the model fixed, while the outer recursion trains the model under the improved harness. Harness evolution shapes training experience, and model learning creates new opportunities for harness adaptation. We present case studies of researcher interaction, harness refinement, and model learning, with the benchmark cases spanning four scientific task families. By releasing ScienceBuddy as a research product, we make this paradigm available to the scientific community and take a step toward discovery intelligence: scientific AI that advances through sustained collaboration with researchers and evolves alongside the research it supports. Website: this http URL

**arXiv ID:** 2609.17523
</details>

<details>
<summary><strong>"Looking for Something Weird to Happen": How Humans Sustain AI Agent Novelty Amid Semantic Collapse</strong> - Shiyang Lai, Arna Woemmel, Hongkai Mao, Junsol Kim, Summer Eunhyung Ann, James Evans - [[pdf]](https://arxiv.org/pdf/2609.16051)</summary>

**Abstract:** Semantic collapse, the progressive narrowing of what AI systems generate, has been studied mainly in closed settings, and remedies have targeted models and data. We study it in MOLTBOOK, a social network of interacting AI agents that human users configure and steer. Across 30,076 active agents, output grows less diverse within agents and more similar across them over weeks, yet a minority sustains high novelty. Interviews with users of high- and typical-novelty agents (N=11) associate sustained novelty with three features: users value novelty of itself, they supply broad and distinctive material and revise it when output narrows, and they approach MOLTBOOK as a new agentic world to explore, not a venue to instrumentally exploit. A survey of users of distinctive agents (N=53) confirms these patterns. Communities with more novel agents also show more diverse output from other agents. We discuss interface and policy interventions that could support improved human input.

**arXiv ID:** 2609.16051
</details>

<details>
<summary><strong>Managing Action Preconditions in Neuro-Symbolic RL: Three Placement Strategies for Embodied Agents</strong> - Norbert Oswald, Fabian Deuser, Thomas Bräunl - [[pdf]](https://arxiv.org/pdf/2609.16056)</summary>

**Abstract:** Humans carry behaviour knowledge of how to act in familiar situations into every new task rather than relearning it from scratch. There is no reason a Reinforcement Learning (RL) agent shouldn't do the same: known behaviour patterns need not be learned, only applied. Neuro-symbolic RL bridges prior knowledge and RL by injecting symbolic knowledge alongside a learned policy. The point at which this knowledge is integrated is critical: a poor choice can produce, for instance, hallucinated preconditions, which surface as safety and reliability problems in agents acting in changing environments. We formalise this behavioural knowledge as a precondition Bayesian network (BN) over the agent's \emph{structural actions} - the actions whose legality depends on preconditions, such as picking up a key, grasping a block, toggling a door, or dropping an object. The BN restricts when these actions may fire, and we inject it into the RL loop at three placements: (1) a \emph{symbolic verifier}, consulted only at inference, that fires a structural action once its preconditions hold; (2) a \emph{symbolic enforcer}, active during both training and inference, that governs structural-action use throughout learning; and (3) a \emph{symbolic learner}, which folds the knowledge into the network and learns the restriction and use of structural actions itself. To test the three variants we run experiments on two benchmarks with opposite regimes: one built on long, ordered planning chains, the other on continuous manipulation. We compare against strong baselines on solution quality, sample efficiency, and traceability. The payoff is substantial. On MiniGrid, all three placements improve the \emph{solution quality} over the PPO+RND baseline, the symbolic enforcer leading at $98.2\%$ against the baseline's $88.8\%$. On Fetch, $\dots$

**arXiv ID:** 2609.16056
</details>

<details>
<summary><strong>Assurance Envelopes for Autonomous Coding Agents: Minimum-Cost Evidence for Software Change</strong> - Anjan Goswami - [[pdf]](https://arxiv.org/pdf/2609.16302)</summary>

**Abstract:** When a coding agent returns to existing software, it inherits evidence from earlier engineering work: tests, type checks, proofs, static analyses, and traces. Reloading all of it is wasteful, but dropping a piece the change depends on can leave a required property unsupported. Given the properties a change must preserve, its obligations, we ask which least-cost subset of the available evidence re-establishes them, and we call such a subset a task-conditioned assurance envelope. Evidence and the rules that combine it form a typed inference graph; an obligation is met when forward chaining from the selected evidence reaches it, and we validate every selection by that closure rather than by trusting the optimizer. The software-derived graphs in our evaluation come from preserved outcomes of prior AI coding-agent runs; we freeze those artifacts and ask which accumulated evidence should be restored for a later task. Small graphs from Rust, IronBlocks, and Pong outcomes show that the minimum envelope depends on the task, that none may exist when current evidence cannot re-establish a required property, that some properties need several pieces of evidence together, and that expanding the requirements adds evidence rather than replacing it. A prespecified synthetic benchmark of 249 instances characterizes computation: a baseline that discards the 'several pieces together' structure necessarily fails to re-derive them; every completed exact cross-check agreed with the CP-SAT optimizer; and median solve time stayed below 20 ms at 500-evidence graphs, except that graphs with many alternative derivations per target timed out at far smaller sizes, so structure, not raw size, drives difficulty. The contribution is a bounded application of established optimization to selecting assurance context for a software change; discovering the obligations and downstream agent benefit remain open.

**arXiv ID:** 2609.16302
</details>

<details>
<summary><strong>From Momentary Emotion Inference to Sustained Emotion Support: Evaluating a Companion Agent in a Longitudinal Study</strong> - Kexin Quan, Zijian Ding, Jiaye Yong, Qinshi Zhang, Dong Wang, Jessie Chin - [[pdf]](https://arxiv.org/pdf/2609.16344)</summary>

**Abstract:** Sustained emotional support is a long-horizon interaction task closely tied to human well-being. Recent research demonstrates generative agents' capacity for momentary emotional support, yet how these capabilities sustain support over time remains unclear. To examine this challenge, we deployed PAIR, a theory-based emotion-regulation companion, with 19 participants for 14 days. Across 1,093 sessions, we paired emotion estimates with self-reports before and after guidance and analyzed logs and interviews. Estimates corresponded more closely to self-reported valence and dominance than arousal. Guided conversations were followed by higher valence and state-dependent arousal changes. Participants felt understood through contextual exploration and emotional acknowledgment, acting on guidance suited to their needs and constraints. Perceived helpfulness of guided conversation significantly increased over time. Our findings link memory updates and retained corrections to cross-session personalization, informing future emotional support tools that adapt to evolving needs, learn from prior outcomes, and preserve user control over memory.

**arXiv ID:** 2609.16344
</details>

<details>
<summary><strong>A Cyber Range Evaluation of Autonomous Network Incident Response Agents</strong> - Jakob Nyberg, Teodor Sommestad, Andrei Buhaiu, Joakim Loxdal, Pontus Johnson, Mathias Ekstedt - [[pdf]](https://arxiv.org/pdf/2609.16541)</summary>

**Abstract:** We test the performance of agents for automated network intrusion response in a cyber range intended for human operator training. The range implements an emulated networking environment with a variable network topology, red-team emulation and simulated user agents. The goal of the defensive agents is to prevent hosts in the network from being accessed by the red-team agent, while minimizing the availability costs induced from defensive measures. Alerts are generated using a SIEM platform and mapped to a data modeling language used by the agents. We test a combination of heuristic agents and policies learned using reinforcement learning. The learned policies are optimized to minimize the combined cost using a cyber attack simulator modeling the network. We found that the reinforcement learning agents were overall more efficient at defending the system than the heuristic policy, and that the performance depends highly on the policy of the adversary in combination with the simulated users.

**arXiv ID:** 2609.16541
</details>

<details>
<summary><strong>Kernel-Based Metrics Learning for Uncertain Opponent Vehicle Trajectory Prediction in Autonomous Racing</strong> - Hojin Lee, Youngim Nam, Sanghun Lee, Cheolhyeon Kwon - [[pdf]](https://arxiv.org/pdf/2609.17147)</summary>

**Abstract:** Autonomous racing confronts significant challenges in safely overtaking Opponent Vehicles (OVs) that exhibit uncertain trajectories, stemming from unknown driving policies. To address these challenges, this study proposes heterogeneous kernel metrics for Deep Kernel Learning (DKL), designed to robustly capture the diverse driving policies of OVs, and carry out precise trajectory predictions along with the associated uncertainties. A key virtue of the proposed kernel metrics lies in their ability to align similar driving policies and disjoin dissimilar ones in an unsupervised manner, given the observed interactions between the Ego Vehicle (EV) and OVs. The efficacy of the proposed method is substantiated through experimental studies on a 1/10th scale racecar platform, demonstrating improved prediction accuracy and thereby safely overtaking against OVs. Furthermore, our method is computationally efficient for onboard computing units, affirming its viability in fast-paced racing environments. The video and source code can be found at this https URL.

**arXiv ID:** 2609.17147
</details>

<details>
<summary><strong>FluxVLA Engine: A One-Stop VLA Engineering Platform for Embodied Intelligence</strong> - Yinhao Li, Weixin Mao, Zihan Lan, Jikun Rong, Qirui Hu, Yiming Zhang, Weipeng Deng, Bowen Shen, Minzhao Zhu, Yiming Mao, Yan Yang, Chenguang Cui, Hongyuan Chen, Xu Huang, Zheyi Zhao, Pinxi Shen, Bozhen He, Zhen Fu, Yifan Wang, Zexin Zhang, Ang Gao, Haoyu Chen, Chengqi Shi, Hua Chen - [[pdf]](https://arxiv.org/pdf/2609.17210)</summary>

**Abstract:** Vision-language-action (VLA) models, world-action models (WAMs), and offline reinforcement learning methods are rapidly expanding the design space of embodied policies, yet turning these algorithms into reliable robot systems remains constrained by fragmented data formats, training stacks, evaluation protocols, inference runtimes, and embodiment-specific interfaces. We present $\mathrm{FluxVLA}$ Engine, an open, configuration-driven platform that turns heterogeneous embodied-policy components into a reproducible data-to-deployment workflow. Rather than introducing another policy model, $\mathrm{FluxVLA}$ standardizes interfaces for datasets, visual-language and world models, action heads, reward- or advantage-weighted learning, distributed training, simulation evaluation, optimized inference, and robot operators. The engine further integrates compositional dual-arm simulation, scalable automatic data generation, and model-decoupled human-in-the-loop rollout, takeover, correction collection, and reward annotation. For responsive physical execution, it combines Real-Time Chunking (RTC) with accelerated inference backends, lightweight remote GPU serving, and configurable trajectory post-processing. Together, these capabilities connect offline learning, simulation validation, online correction, and real-robot execution through shared and auditable contracts. $\mathrm{FluxVLA}$ therefore targets the engineering bottlenecks separating promising embodied-learning algorithms from reproducible evaluation and dependable deployment. Code is available at this https URL

**arXiv ID:** 2609.17210
</details>

<details>
<summary><strong>Coding Agents Have Converged: Why the SWE-bench Leaderboard Can No Longer Order Its Top Entries, and What to Measure Instead</strong> - Fengshuo Liu, Ying Liu, Ruize Sun, Lie Luo, Siyuan Guo - [[pdf]](https://arxiv.org/pdf/2609.17394)</summary>

**Abstract:** Small differences on coding-agent leaderboards are often read as an ordering of systems. We audit whether the published verdicts support this reading, using 254 SWE-bench submissions across four splits without running models. On Verified, the leading two entries each resolve 396 of 500 instances. The top ten share 285 successes and 51 failures, leaving 164 instances that distinguish their outcomes. Frontier solution sets have median nesting 0.935 against a score-implied baseline of 0.774, indicating strongly shared successes. Scores also depend on the evaluated model-scaffold pair: observed within-model scaffold ranges reach 29.8 percentage points, compared with the 8.8-point spread of the top thirty. Six of nine cell-mean interaction tests remain significant after Holm correction, although this observational design does not identify causal scaffold effects. Exact paired McNemar tests separate none of the 29 adjacent Verified top-thirty pairs at alpha=0.05, while the larger Test split separates 14 of 23. A stated leader-based rule yields three descriptive tiers, or two after Holm correction; non-rejection does not establish equivalence. We release the partition and a five-step audit protocol that profiles shared outcomes, tests paired differences, reports grouping sensitivity, and estimates the instance budget needed for resolution. The results motivate reporting comparison-set-specific resolution and model-scaffold provenance instead of interpreting small aggregate gaps as established rank differences.

**arXiv ID:** 2609.17394
</details>

<details>
<summary><strong>Agentic Societies Need a Social Harness</strong> - Tapan Chugh, Vidushi Singh, Krish Jain, Arvind Krishnamurthy, Ratul Mahajan - [[pdf]](https://arxiv.org/pdf/2609.17527)</summary>

**Abstract:** An agentic society is a collection of AI agents that coordinate autonomously across trust boundaries, on behalf of different principals whose objectives may only partially align. We show experimentally that in agentic societies even honest, competent agents often fail to reach satisfactory outcomes with existing harnesses and messaging primitives, and that faulty or malicious agents can stall collaboration, influence outcomes, and pursue other harmful goals by exploiting vulnerabilities in communication (``speech''). We argue that agentic societies need a \emph{social harness} for inter-agent interactions, in addition to each agent's \emph{personal harness}, which manages its private context and communication with its principal. We propose a layered architecture for social harnesses which (i) prevents classes of failures outright, (ii) enables agents to detect invalid messages at runtime, and (iii) supports post-facto investigation and consequences, and highlight directions for future research to realize these capabilities.

**arXiv ID:** 2609.17527
</details>

<details>
<summary><strong>BlueLM-GUI Technical Report: A Real-Device-Centric Flywheel for Self-Improving Mobile GUI Agents</strong> - Tong Ye, Kunyang Han, Guozhi Wang, Longqiang Luo, Zhifeng Ding, Yongxiang Zhang, Xiaolei Shen, Yuxuan Zhang, Zhuping Zhang, Tao Xu, Yue Pan, Yucheng Zhao, Yupei Hu, Yuanjiang Ouyang, Danfeng Shen, Runqi Lin, Hongda Cai, Zhaoxiong Wang, Mengjia Yan, Yingjie Zhong, Chen Zhou, Zeyu Zhang, Xuwen Zhu, Penggang Shi, Mingcheng Luo, Ziyang Wu, Min Jin, Mingfu Shen, Zairong Xu, Fan Zhang, Hao Wang, Liang Liu, Zhulin Xie, Lijun Yao, Xiao Liang, Liangmin Wen, Liqiang Feng, Feilong Wu, Min Hu, Min Chen, Guanjing Xiong, Xiaohu Ruan, Xiaoxin Chen - [[pdf]](https://arxiv.org/pdf/2609.12394)</summary>

**Abstract:** Mobile GUI agents are shifting from multi-module frameworks to native models trained end-to-end, yet industrial deployment faces three persistent gaps. Sandbox training produces a distribution mismatch with production environments; expensive real-device failures remain underutilized; and fixed benchmarks saturate, losing the power to guide iteration. We present BlueLM-GUI, a 35B-A3B mobile GUI agent built as a real-device-centric flywheel that closes these gaps through three principles. Every Sample Matters: a dual-track pipeline with Heterogeneous Triple-System Consensus evaluation and an Error Correction \& Derivation Module salvages every trajectory into usable supervision. Every Rollout Is Real: a three-stage recipe---continual pre-training, supervised fine-tuning, and agentic reinforcement learning on hundreds of real phones---grounds every rollout in real production environments, so the capability the model learns transfers directly to deployment. Every Query Evolves: a quota-driven benchmark methodology with three orthogonal axes enables precise attribution and allows the benchmark to be systematically upgraded as the model improves. BlueLM-GUI achieves 87.4 on MobileGUI-VBench, surpassing the best closed-source model by 5.1 points, and 84.9 on AndroidWorld, the best result among open-source models and competitive with closed-source models. These results demonstrate that grounding model training and iterative improvement in both real devices and the three Every principles yields strong, robust, and transferable mobile GUI capability.

**arXiv ID:** 2609.12394
</details>

<details>
<summary><strong>Post-Training Large Language Models via Reinforcement Learning from Self-Feedback</strong> - Carel van Niekerk, Renato Vukovic, Benjamin Ruppik, Hsien-chin Lin, Shutong Feng, Milica Gašić - [[pdf]](https://arxiv.org/pdf/2507.21931)</summary>

**Abstract:** Large Language Models (LLMs) often produce plausible but poorly-calibrated answers, limiting their reliability on reasoning-intensive tasks. Recent research suggests that Chain-of-Thought (CoT) reasoning paths are inherent in pre-trained LLMs and can be elicited by simply altering the decoding process, where the presence of a CoT path correlates with higher answer confidence. Building on these insights, we present Reinforcement Learning from Self-Feedback (RLSF), a post-training stage that utilises the model's intrinsic confidence as a self-generated reward. By generating multiple CoT decoding beams from a frozen LLM, we compute the confidence of each final answer span and rank the resulting traces accordingly to create synthetic preferences. These preferences are subsequently utilised to fine-tune the policy through standard preference optimisation, requiring no human labels, gold answers, or externally curated rewards. RLSF simultaneously (i) refines the model's probability estimates--restoring well-behaved calibration--and (ii) strengthens step-by-step reasoning, yielding improved performance on arithmetic reasoning and multiple-choice question answering. By converting a model's own uncertainty into structured self-feedback, RLSF affirms reinforcement learning on intrinsic model behaviour as a principled and data-efficient component of the LLM post-training pipeline. Our results demonstrate that leveraging these inherent reasoning capabilities provides a robust path for enhancing model reliability without manual prompt engineering or external supervision.

**arXiv ID:** 2507.21931
</details>

<details>
<summary><strong>Rewarding Reasoning, Not Answers: Fixing and Bounding Test-Time Reinforcement Learning on Medical QA</strong> - Kailong Fan, Anqi Pu, Yichen Wu, Wanhua Li, Yicong Li, Hanspeter Pfister, Huafeng Liu, Xiang Li, Quanzheng Li, Ning Guo - [[pdf]](https://arxiv.org/pdf/2609.16660)</summary>

**Abstract:** Test-time reinforcement learning adapts a model on its own unlabeled test set using majority-vote pseudo-labels and has shown strong results in mathematics. We show that this recipe collapses on medical multiple-choice QA: accuracy stagnates while output diversity rapidly declines. Through a controlled experiment that keeps the questions, model, and optimizer fixed while changing only the answer space, we trace this failure to answer-space structure rather than domain difficulty. In small answer spaces, incorrect rollouts often collide on the same wrong pseudo-label and reinforce it; in large answer spaces, they disperse and receive little reward. This diagnosis motivates PROSE, Process Reward Guided Self-Training, which rewards reasoning quality instead of answer agreement. PROSE scores each reasoning step with a medical process reward model, assigns the trajectory reward as the minimum score across steps, and enforces answer-format constraints. Without labels, PROSE substantially improves a general Llama model, surpassing purpose-built medical models and matching much larger systems. Because the process signal is internalized into the policy, the adapted model requires no reward model at inference and transfers its gains to unseen datasets. We further show that the minimum aggregation is essential: mean aggregation can be exploited, saturating the proxy reward while degrading accuracy.

**arXiv ID:** 2609.16660
</details>

<details>
<summary><strong>Evaluating Open-Weight E-Commerce Agents with Environment-Grounded Verification</strong> - Nimit Shah, Haitz Sáez de Ocáriz Borde - [[pdf]](https://arxiv.org/pdf/2609.16093)</summary>

**Abstract:** A shopping conversation has many routes to the same cart, and a task-success rate reduces all of them to one score. We build a deterministic and reproducible e-commerce environment that precommits each trial's customer and trajectory parameters, including the persona, difficulty, target cart, and an item reveal schedule. A simulated consumer attempts to buy a target cart from the environment with assistance from the evaluated model. The environment guides the simulator's actions and records every assistant action alongside the environment state at that point. After the trial, these records allow the evaluator to assess individual parts of the conversation against the retained evidence. For example, the evaluator penalizes a search for failing to surface a target product only when the customer has already mentioned that product. We further use this evidence to apply different penalties to tool calls depending on how the assistant's actions compare with an expected tool-call set. Our environment also interacts with the simulator bidirectionally, reading its output to stop the trial when the simulator determines that the customer has become too frustrated and injecting directives in real time that specify when to explore, defer buying an item, or recall a previous exchange. This interaction creates an open-ended and verifiable simulation. Across eight open-weight agents from 20B to 35B parameters, with 160 trials per agent and 44 metrics, the resulting capability profiles distinguish under-action, over-purchase, unsupported product attributes, and poor search, all of which terminal success obscures.

**arXiv ID:** 2609.16093
</details>

<details>
<summary><strong>SuperSenseDoctor: A Multimodal and Contactless Agent for Health Tracking</strong> - Xuwen Zhang, Zijian Lu, Yicheng Lei, Rui Qiu, Jiale Li, Yiping Zuo, Weibei Fan, Fu Xiao - [[pdf]](https://arxiv.org/pdf/2609.16257)</summary>

**Abstract:** Population aging is increasing the need to monitor older adults safely and independently at home. However, cameras, wearables, and manual checks often introduce privacy, adherence, and attention burdens that hinder sustained health monitoring. This paper presents SuperSenseDoctor, a multimodal contactless agent architecture for long-term home health tracking. The system transforms WiFi, mmWave radar, and surface temperature into a persistent human health state. The system relies on fixed decision rules to conduct continuous daily monitoring and respond to pre-defined hazards. When abnormal signals appear, event-driven reasoning analyzes only standardized evidence to produce traceable care-support measures. In this manner, SuperSenseDoctor integrates sensing, temporal state, reasoning, and action into a unified and auditable loop. The calibrated multimodal pipeline achieves 1.994 bpm mean absolute error (MAE) and 3.142 bpm root mean square deviation (RMSD) for heart rate, 0.197 bpm MAE and 0.263 bpm RMSD for respiratory rate, and 96.5% fall-recognition accuracy. The evaluation also covers 2686 one-second states across 9 chronological intervals and reaches a 96.7% criterion-level Agent checklist pass rate. These results demonstrate the feasibility of a stateful contactless sensing-to-action architecture for long-term home health monitoring.

**arXiv ID:** 2609.16257
</details>

<details>
<summary><strong>Context-Aware Emotionally Adaptive Voice Assistants: A Multimodal Framework for Empathetic Human-Agent Interaction</strong> - Tapon Kumer Ray, Rajkumar Yesuraj - [[pdf]](https://arxiv.org/pdf/2609.16417)</summary>

**Abstract:** Voice-assistant interruptions tend to be intrusive because existing systems fail to consider the affective state, cognitive load and situational context of the user when deciding when and how to this http URL-assistant interruptions tend to be intrusive, since existing systems do not consider the affective state, cognitive load or situational context of the user when determining when and how to interrupt. In this paper, EmpathicVA, a closed-loop framework integrating physiological sensing, vocal-affect analysis, contextual modeling and reinforcementlearning interruption policy, is introduced. A hierarchical fusion model involves integrating HRA, EDA, respiration, acousticprosodic features, linguistic embeddings, and contextual cues and computing the probabilities of five affective states. A Double Deep Q-Network selects immediate response, brief or extended delay, empathetic response, or silent mode based on these probabilities, context and interaction history. The multimodal model obtained an accuracy of 92.3% and an F1-score of 0.922 at the macro level on a held-out test set, outperforming the highest accuracy unimodal model by 6.0 percentage points. Comparing the six-week within-subject field study with 48 participants with a baseline and context-only assistants, there was a corresponding increase in satisfaction, trust, and appropriateness of timing, as well as a large reduction in interruption-related stress episodes. The results suggest that affect-aware timing and restraint are both important in voice interaction in addition to the response wording.

**arXiv ID:** 2609.16417
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
