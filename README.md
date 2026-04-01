# Agent arXiv Daily

**Last Updated:** 2026-04-01 03:39:37

**Total Papers:** 84

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (8 papers)</h2></summary>

<details>
<summary><strong>Nomad: Autonomous Exploration and Discovery</strong> - Bokang Jia, Samta Kamboj, Satheesh Katipomu, Seung Hun Han, Neha Sengupta, Andrew Jackson - [[pdf]](https://arxiv.org/pdf/2603.29353)</summary>

**Abstract:** We introduce Nomad, a system for autonomous data exploration and insight discovery. Given a corpus of documents, databases, or other data sources, users rarely know the full set of questions, hypotheses, or connections that could be explored. As a result, query-driven question answering and prompt-driven deep-research systems remain limited by human framing and often fail to cover the broader insight space.
Nomad addresses this problem with an exploration-first architecture. It constructs an explicit Exploration Map over the domain and systematically traverses it to balance breadth and depth. It generates and selects hypotheses and investigates them with an explorer agent that can use document search, web search, and database tools. Candidate insights are then checked by an independent verifier before entering a reporting pipeline that produces cited reports and higher-level meta-reports.
We also present a comprehensive evaluation framework for autonomous discovery systems that measures trustworthiness, report quality, and diversity. Using a corpus of selected UN and WHO reports, we show that \nomad{} produces more trustworthy and higher-quality reports than baselines, while also producing more diverse insights over several runs.
Nomad is a step toward autonomous systems that not only answer user questions or conduct directed research, but also discover which questions, research directions, and insights are worth surfacing in the first place.

**arXiv ID:** 2603.29353
</details>

<details>
<summary><strong>CivicShield: A Cross-Domain Defense-in-Depth Framework for Securing Government-Facing AI Chatbots Against Multi-Turn Adversarial Attacks</strong> - KrishnaSaiReddy Patil - [[pdf]](https://arxiv.org/pdf/2603.29062)</summary>

**Abstract:** LLM-based chatbots in government services face critical security gaps. Multi-turn adversarial attacks achieve over 90% success against current defenses, and single-layer guardrails are bypassed with similar rates. We present CivicShield, a cross-domain defense-in-depth framework for government-facing AI chatbots. Drawing on network security, formal verification, biological immune systems, aviation safety, and zero-trust cryptography, CivicShield introduces seven defense layers: (1) zero-trust foundation with capability-based access control, (2) perimeter input validation, (3) semantic firewall with intent classification, (4) conversation state machine with safety invariants, (5) behavioral anomaly detection, (6) multi-model consensus verification, and (7) graduated human-in-the-loop escalation. We present a formal threat model covering 8 multi-turn attack families, map the framework to NIST SP 800-53 controls across 14 families, and evaluate using ablation analysis. Theoretical analysis shows layered defenses reduce attack probability by 1-2 orders of magnitude versus single-layer approaches. Simulation against 1,436 scenarios including HarmBench (416), JailbreakBench (200), and XSTest (450) achieves 72.9% combined detection [69.5-76.0% CI] with 2.9% effective false positive rate after graduated response, while maintaining 100% detection of multi-turn crescendo and slow-drift attacks. The honest drop on real benchmarks versus author-generated scenarios (71.2% vs 76.7% on HarmBench, 47.0% vs 70.0% on JailbreakBench) validates independent evaluation importance. CivicShield addresses an open gap at the intersection of AI safety, government compliance, and practical deployment.

**arXiv ID:** 2603.29062
</details>

<details>
<summary><strong>APEX-EM: Non-Parametric Online Learning for Autonomous Agents via Structured Procedural-Episodic Experience Replay</strong> - Pratyay Banerjee, Masud Moshtaghi, Ankit Chadha - [[pdf]](https://arxiv.org/pdf/2603.29093)</summary>

**Abstract:** LLM-based autonomous agents lack persistent procedural memory: they re-derive solutions from scratch even when structurally identical tasks have been solved before. We present \textbf{APEX-EM}, a non-parametric online learning framework that accumulates, retrieves, and reuses structured procedural plans without modifying model weights. APEX-EM introduces: (1) a \emph{structured experience representation} encoding the full procedural-episodic trace of each execution -- planning steps, artifacts, iteration history with error analysis, and quality scores; (2) a \emph{Plan-Retrieve-Generate-Iterate-Ingest} (PRGII) workflow with Task Verifiers providing multi-dimensional reward signals; and (3) a \emph{dual-outcome Experience Memory} with hybrid retrieval combining semantic search, structural signature matching, and plan DAG traversal -- enabling cross-domain transfer between tasks sharing no lexical overlap but analogous operational structure. Successful experiences serve as positive in-context examples; failures as negative examples with structured error annotations.
We evaluate on BigCodeBench~\cite{zhuo2025bigcodebench}, KGQAGen-10k~\cite{zhang2025kgqagen}, and Humanity's Last Exam~\cite{phan2025hle} using Claude Sonnet 4.5 and Opus 4.5. On KGQAGen-10k, APEX-EM achieves 89.6\% accuracy versus 41.3\% without memory (+48.3pp), surpassing the oracle-retrieval upper bound (84.9\%). On BigCodeBench, it reaches 83.3\% SR from a 53.9\% baseline (+29.4pp), exceeding MemRL's~\cite{memrl2025} +11.0pp gain under comparable frozen-backbone conditions (noting backbone differences controlled for in our analysis). On HLE, entity graph retrieval reaches 48.0\% from 25.2\% (+22.8pp). Ablations show component value is task-dependent: rich judge feedback is negligible for code generation but critical for structured queries (+10.3pp), while binary-signal iteration partially compensates for weaker feedback.

**arXiv ID:** 2603.29093
</details>

<details>
<summary><strong>BotVerse: Real-Time Event-Driven Simulation of Social Agents</strong> - Edoardo Allegrini, Edoardo Di Paolo, Angelo Spognardi, Marinella Petrocchi - [[pdf]](https://arxiv.org/pdf/2603.29741)</summary>

**Abstract:** BotVerse is a scalable, event-driven framework for high-fidelity social simulation using LLM-based agents. It addresses the ethical risks of studying autonomous agents on live networks by isolating interactions within a controlled environment while grounding them in real-time content streams from the Bluesky ecosystem. The system features an asynchronous orchestration API and a simulation engine that emulates human-like temporal patterns and cognitive memory. Through the Synthetic Social Observatory, researchers can deploy customizable personas and observe multimodal interactions at scale. We demonstrate BotVersevia a coordinated disinformation scenario, providing a safe, experimental framework for red-teaming and computational social scientists. A video demonstration of the framework is available at this https URL.

**arXiv ID:** 2603.29741
</details>

<details>
<summary><strong>"What Did It Actually Do?": Understanding Risk Awareness and Traceability for Computer-Use Agents</strong> - Zifan Peng, Mingchen Li - [[pdf]](https://arxiv.org/pdf/2603.28551)</summary>

**Abstract:** Personalized computer-use agents are rapidly moving from expert communities into mainstream use. Unlike conventional chatbots, these systems can install skills, invoke tools, access private resources, and modify local environments on users' behalf. Yet users often do not know what authority they have delegated, what the agent actually did during task execution, or whether the system has been safely removed afterward.
We investigate this gap as a combined problem of risk understanding and post-hoc auditability, using OpenClaw as a motivating case. We first build a multi-source corpus of the OpenClaw ecosystem, including incidents, advisories, malicious-skill reports, news coverage, tutorials, and social-media narratives. We then conduct an interview study to examine how users and practitioners understand skills, autonomy, privilege, persistence, and uninstallation. Our findings suggest that participants often recognized these systems as risky in the abstract, but lacked concrete mental models of what skills can do, what resources agents can access, and what changes may remain after execution or removal. Motivated by these findings, we propose AgentTrace, a traceability framework and prototype interface for visualizing agent actions, touched resources, permission history, provenance, and persistent side effects. A scenario-based evaluation suggests that traceability-oriented interfaces can improve understanding of agent behavior, support anomaly detection, and foster more calibrated trust.

**arXiv ID:** 2603.28551
</details>

<details>
<summary><strong>Quality-Controlled Active Learning via Gaussian Processes for Robust Structure-Property Learning in Autonomous Microscopy</strong> - Jawad Chowdhury, Ganesh Narasimha, Jan-Chi Yang, Yongtao Liu, Rama Vasudevan - [[pdf]](https://arxiv.org/pdf/2603.29135)</summary>

**Abstract:** Autonomous experimental systems are increasingly used in materials research to accelerate scientific discovery, but their performance is often limited by low-quality, noisy data. This issue is especially problematic in data-intensive structure-property learning tasks such as Image-to-Spectrum (Im2Spec) and Spectrum-to-Image (Spec2Im) translations, where standard active learning strategies can mistakenly prioritize poor-quality measurements. We introduce a gated active learning framework that combines curiosity-driven sampling with a physics-informed quality control filter based on the Simple Harmonic Oscillator model fits, allowing the system to automatically exclude low-fidelity data during acquisition. Evaluations on a pre-acquired dataset of band-excitation piezoresponse spectroscopy (BEPS) data from PbTiO3 thin films with spatially localized noise show that the proposed method outperforms random sampling, standard active learning, and multitask learning strategies. The gated approach enhances both Im2Spec and Spec2Im by handling noise during training and acquisition, leading to more reliable forward and inverse predictions. In contrast, standard active learners often misinterpret noise as uncertainty and end up acquiring bad samples that hurt performance. Given its promising applicability, we further deployed the framework in real-time experiments on BiFeO3 thin films, demonstrating its effectiveness in real autonomous microscopy experiments. Overall, this work supports a shift toward hybrid autonomy in self-driving labs, where physics-informed quality assessment and active decision-making work hand-in-hand for more reliable discovery.

**arXiv ID:** 2603.29135
</details>

<details>
<summary><strong>HyperKKL: Learning KKL Observers for Non-Autonomous Nonlinear Systems via Hypernetwork-Based Input Conditioning</strong> - Yahia Salaheldin Shaaban, Abdelrahman Sayed Sayed, M. Umar B. Niazi, Karl Henrik Johansson - [[pdf]](https://arxiv.org/pdf/2603.29744)</summary>

**Abstract:** Kazantzis-Kravaris/Luenberger (KKL) observers are a class of state observers for nonlinear systems that rely on an injective map to transform the nonlinear dynamics into a stable quasi-linear latent space, from where the state estimate is obtained in the original coordinates via a left inverse of the transformation map. Current learning-based methods for these maps are designed exclusively for autonomous systems and do not generalize well to controlled or non-autonomous systems. In this paper, we propose two learning-based designs of neural KKL observers for non-autonomous systems whose dynamics are influenced by exogenous inputs. To this end, a hypernetwork-based framework ($HyperKKL$) is proposed with two input-conditioning strategies. First, an augmented observer approach ($HyperKKL_{obs}$) adds input-dependent corrections to the latent observer dynamics while retaining static transformation maps. Second, a dynamic observer approach ($HyperKKL_{dyn}$) employs a hypernetwork to generate encoder and decoder weights that are input-dependent, yielding time-varying transformation maps. We derive a theoretical worst-case bound on the state estimation error. Numerical evaluations on four nonlinear benchmark systems show that input conditioning yields consistent improvements in estimation accuracy over static autonomous maps, with an average symmetric mean absolute percentage error (SMAPE) reduction of 29% across all non-zero input regimes.

**arXiv ID:** 2603.29744
</details>

<details>
<summary><strong>Exploring Sidewalk Sheds in New York City through Chatbot Surveys and Human Computer Interaction</strong> - Junyi Li, Zhaoxi Zhang, Tamir Mendel, Takahiro Yabe - [[pdf]](https://arxiv.org/pdf/2601.23095)</summary>

**Abstract:** Sidewalk sheds are a common feature of the streetscape in New York City, reflecting ongoing construction and maintenance activities. However, policymakers and local business owners have raised concerns about reduced storefront visibility and altered pedestrian navigation. Although sidewalk sheds are widely used for safety, their effects on pedestrian visibility and movement are not directly measured in current planning practices. To address this, we developed an AI-based chatbot survey that collects image-based annotations and route choices from pedestrians, linking these responses to specific shed design features, including clearance height, post spacing, and color. This AI chatbot survey integrates a large language model (e.g., Google's Gemini-1.5-flash-001 model) with an image-annotation interface, allowing users to interact with street images, mark visual elements, and provide structured feedback through guided dialogue. To explore pedestrian perceptions and behaviors, this paper conducts a grid-based analysis of entrance annotations and applies logistic mixed-effects modeling to assess sidewalk choice patterns. Analysis of the dataset (n = 25) shows that: (1) the presence of scaffolding significantly reduces pedestrians' ability to identify ground-floor retail entrances, and (2) variations in weather conditions and shed design features significantly influence sidewalk selection behavior. By integrating generative AI into urban research, this study demonstrates a novel method for evaluating sidewalk shed designs and provides empirical evidence to support adjustments to shed guidelines that improve the pedestrian experience without compromising safety.

**arXiv ID:** 2601.23095
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (14 papers)</h2></summary>

<details>
<summary><strong>Emergence WebVoyager: Toward Consistent and Transparent Evaluation of (Web) Agents in The Wild</strong> - Deepak Akkil, Mowafak Allaham, Amal Raj, Tamer Abuelsaad, Ravi Kokku - [[pdf]](https://arxiv.org/pdf/2603.29020)</summary>

**Abstract:** Reliable evaluation of AI agents operating in complex, real-world environments requires methodologies that are robust, transparent, and contextually aligned with the tasks agents are intended to perform. This study identifies persistent shortcomings in existing AI agent evaluation practices that are particularly acute in web agent evaluation, as exemplified by our audit of WebVoyager, including task-framing ambiguity and operational variability that hinder meaningful and reproducible performance comparisons. To address these challenges, we introduce Emergence WebVoyager, an enhanced version of the WebVoyager benchmark that standardizes evaluation methodology through clear guidelines for task instantiation, failure handling, annotation, and reporting. Emergence WebVoyager achieves an inter-annotator agreement of 95.9\%, indicating improved clarity and reliability in both task formulation and evaluation. Applying this framework to evaluate OpenAI Operator reveals substantial performance variation across domains and task types, with an overall success rate of 68.6\%, substantially lower than the 87\% previously reported by OpenAI, demonstrating the utility of our approach for more rigorous and comparable web agent evaluation.

**arXiv ID:** 2603.29020
</details>

<details>
<summary><strong>AEC-Bench: A Multimodal Benchmark for Agentic Systems in Architecture, Engineering, and Construction</strong> - Harsh Mankodiya, Chase Gallik, Theodoros Galanos, Andriy Mulyar - [[pdf]](https://arxiv.org/pdf/2603.29199)</summary>

**Abstract:** The AEC-Bench is a multimodal benchmark for evaluating agentic systems on real-world tasks in the Architecture, Engineering, and Construction (AEC) domain. The benchmark covers tasks requiring drawing understanding, cross-sheet reasoning, and construction project-level coordination. This report describes the benchmark motivation, dataset taxonomy, evaluation protocol, and baseline results across several domain-specific foundation model harnesses. We use AEC-Bench to identify consistent tools and harness design techniques that uniformly improve performance across foundation models in their own base harnesses, such as Claude Code and Codex. We openly release our benchmark dataset, agent harness, and evaluation code for full replicability at this https URL under an Apache 2 license.

**arXiv ID:** 2603.29199
</details>

<details>
<summary><strong>PSPA-Bench: A Personalized Benchmark for Smartphone GUI Agent</strong> - Hongyi Nie, Xunyuan Liu, Yudong Bai, Yaqing Wang, Yang Liu, Quanming Yao, Zhen Wang - [[pdf]](https://arxiv.org/pdf/2603.29318)</summary>

**Abstract:** Smartphone GUI agents execute tasks by operating directly on app interfaces, offering a path to broad capability without deep system integration. However, real-world smartphone use is highly personalized: users adopt diverse workflows and preferences, challenging agents to deliver customized assistance rather than generic solutions. Existing GUI agent benchmarks cannot adequately capture this personalization dimension due to sparse user-specific data and the lack of fine-grained evaluation metrics. To address this gap, we present PSPA-Bench, the benchmark dedicated to evaluating personalization in smartphone GUI agents. PSPA-Bench comprises over 12,855 personalized instructions aligned with real-world user behaviors across 10 representative daily-use scenarios and 22 mobile apps, and introduces a structure-aware process evaluation method that measures agents' personalized capabilities at a fine-grained level. Through PSPA-Bench, we benchmark 11 state-of-the-art GUI agents. Results reveal that current methods perform poorly under personalized settings, with even the strongest agent achieving limited success. Our analysis further highlights three directions for advancing personalized GUI agents: (1) reasoning-oriented models consistently outperform general LLMs, (2) perception remains a simple yet critical capability, and (3) reflection and long-term memory mechanisms are key to improving adaptation. Together, these findings establish PSPA-Bench as a foundation for systematic study and future progress in personalized GUI agents.

**arXiv ID:** 2603.29318
</details>

<details>
<summary><strong>ELT-Bench-Verified: Benchmark Quality Issues Underestimate AI Agent Capabilities</strong> - Christopher Zanoli, Andrea Giovannini, Tengjun Jin, Ana Klimovic, Yotam Perlitz - [[pdf]](https://arxiv.org/pdf/2603.29399)</summary>

**Abstract:** Constructing Extract-Load-Transform (ELT) pipelines is a labor-intensive data engineering task and a high-impact target for AI automation. On ELT-Bench, the first benchmark for end-to-end ELT pipeline construction, AI agents initially showed low success rates, suggesting they lacked practical utility.
We revisit these results and identify two factors causing a substantial underestimation of agent capabilities. First, re-evaluating ELT-Bench with upgraded large language models reveals that the extraction and loading stage is largely solved, while transformation performance improves significantly. Second, we develop an Auditor-Corrector methodology that combines scalable LLM-driven root-cause analysis with rigorous human validation (inter-annotator agreement Fleiss' kappa = 0.85) to audit benchmark quality. Applying this to ELT-Bench uncovers that most failed transformation tasks contain benchmark-attributable errors -- including rigid evaluation scripts, ambiguous specifications, and incorrect ground truth -- that penalize correct agent outputs.
Based on these findings, we construct ELT-Bench-Verified, a revised benchmark with refined evaluation logic and corrected ground truth. Re-evaluating on this version yields significant improvement attributable entirely to benchmark correction. Our results show that both rapid model improvement and benchmark quality issues contributed to underestimating agent capabilities. More broadly, our findings echo observations of pervasive annotation errors in text-to-SQL benchmarks, suggesting quality issues are systemic in data engineering evaluation. Systematic quality auditing should be standard practice for complex agentic tasks. We release ELT-Bench-Verified to provide a more reliable foundation for progress in AI-driven data engineering automation.

**arXiv ID:** 2603.29399
</details>

<details>
<summary><strong>C-TRAIL: A Commonsense World Framework for Trajectory Planning in Autonomous Driving</strong> - Zhihong Cui, Haoran Tang, Tianyi Li, Yushuai Li, Peiyuan Guan, Amir Taherkordi, Tor Skeie - [[pdf]](https://arxiv.org/pdf/2603.29908)</summary>

**Abstract:** Trajectory planning for autonomous driving increasingly leverages large language models (LLMs) for commonsense reasoning, yet LLM outputs are inherently unreliable, posing risks in safety-critical applications. We propose C-TRAIL, a framework built on a Commonsense World that couples LLM-derived commonsense with a trust mechanism to guide trajectory planning. C-TRAIL operates through a closed-loop Recall, Plan, and Update cycle: the Recall module queries an LLM for semantic relations and quantifies their reliability via a dual-trust mechanism; the Plan module injects trust-weighted commonsense into Monte Carlo Tree Search (MCTS) through a Dirichlet trust policy; and the Update module adaptively refines trust scores and policy parameters from environmental feedback. Experiments on four simulated scenarios in Highway-env and two real-world levelXData datasets (highD, rounD) show that C-TRAIL consistently outperforms state-of-the-art baselines, reducing ADE by 40.2%, FDE by 51.7%, and improving SR by 16.9 percentage points on average. The source code is available at this https URL.

**arXiv ID:** 2603.29908
</details>

<details>
<summary><strong>SkillTester: Benchmarking Utility and Security of Agent Skills</strong> - Leye Wang, Zixing Wang, Anjie Xu - [[pdf]](https://arxiv.org/pdf/2603.28815)</summary>

**Abstract:** This technical report presents SkillTester, a tool for evaluating the utility and security of agent skills. Its evaluation framework combines paired baseline and with-skill execution conditions with a separate security probe suite. Grounded in a comparative utility principle and a user-facing simplicity principle, the framework normalizes raw execution artifacts into a utility score, a security score, and a three-level security status label. More broadly, it can be understood as a comparative quality-assurance harness for agent skills in an agent-first world. The public service is deployed at this https URL, and the broader project is maintained at this https URL.

**arXiv ID:** 2603.28815
</details>

<details>
<summary><strong>PRISM: A Multi-View Multi-Capability Retail Video Dataset for Embodied Vision-Language Models</strong> - Amirreza Rouhi, Parikshit Sakurikar, Satya Sai Reddy, Narsimha Menga, Anirudh Govil, Sri Harsha Chittajallu, Rajat Aggarwal, Anoop Namboodiri, Sashi Reddi - [[pdf]](https://arxiv.org/pdf/2603.29281)</summary>

**Abstract:** A critical gap exists between the general-purpose visual understanding of state-of-the-art physical AI models and the specialized perceptual demands of structured real-world deployment environments. We present PRISM, a 270K-sample multi-view video supervised fine-tuning (SFT) corpus for embodied vision-language-models (VLMs) in real-world retail environments. PRISM is motivated by a simple observation - physical AI systems fail not because of poor visual recognition, but because they do not understand space, physical dynamics and embodied action well enough to operate reliably in the world. To this end, PRISM is grounded in a novel three-dimensional knowledge ontology that spans spatial knowledge, temporal and physical knowledge, and embodied action knowledge. It covers 20+ capability probes across four evaluation dimensions - Embodied Reasoning (ER), Common Sense (CS), Spatial Perception (SP), and Intuitive Physics (IP), and to our knowledge, PRISM is the first dataset to instantiate all three knowledge dimensions within a single real-world deployment domain. The corpus captures data from egocentric, exocentric and 360° viewpoints across five supermarket locations and includes open-ended, chain-of-thought, and multiple-choice supervision. At 4 fps, PRISM spans approximately 11.8M video frames and approximately 730M tokens, placing it among the largest domain-specific video SFT corpora. Fine-tuning on PRISM reduces the error rate across all 20+ probes by 66.6% over the pre-trained baseline, with significant gains in embodied action understanding where the accuracy improves by 36.4%. Our results suggest that ontology-structured, domain specific SFT can meaningfully strengthen embodied VLMs for real-world settings. The PRISM dataset and more details are available at this https URL

**arXiv ID:** 2603.29281
</details>

<details>
<summary><strong>IMAGAgent: Orchestrating Multi-Turn Image Editing via Constraint-Aware Planning and Reflection</strong> - Fei Shen, Chengyu Xie, Lihong Wang, Zhanyi Zhang, Xin Jiang, Xiaoyu Du, Jinhui Tang - [[pdf]](https://arxiv.org/pdf/2603.29602)</summary>

**Abstract:** Existing multi-turn image editing paradigms are often confined to isolated single-step execution. Due to a lack of context-awareness and closed-loop feedback mechanisms, they are prone to error accumulation and semantic drift during multi-turn interactions, ultimately resulting in severe structural distortion of the generated images. For that, we propose \textbf{IMAGAgent}, a multi-turn image editing agent framework based on a "plan-execute-reflect" closed-loop mechanism that achieves deep synergy among instruction parsing, tool scheduling, and adaptive correction within a unified pipeline. Specifically, we first present a constraint-aware planning module that leverages a vision-language model (VLM) to precisely decompose complex natural language instructions into a series of executable sub-tasks, governed by target singularity, semantic atomicity, and visual perceptibility. Then, the tool-chain orchestration module dynamically constructs execution paths based on the current image, the current sub-task, and the historical context, enabling adaptive scheduling and collaborative operation among heterogeneous operation models covering image retrieval, segmentation, detection, and editing. Finally, we devise a multi-expert collaborative reflection mechanism where a central large language model (LLM) receives the image to be edited and synthesizes VLM critiques into holistic feedback, simultaneously triggering fine-grained self-correction and recording feedback outcomes to optimize future decisions. Extensive experiments on our constructed \textbf{MTEditBench} and the MagicBrush dataset demonstrate that IMAGAgent achieves performance significantly superior to existing methods in terms of instruction consistency, editing precision, and overall quality. The code is available at this https URL.

**arXiv ID:** 2603.29602
</details>

<details>
<summary><strong>Interview-Informed Generative Agents for Product Discovery: A Validation Study</strong> - Zichao Wang, Alexa Siu - [[pdf]](https://arxiv.org/pdf/2603.29890)</summary>

**Abstract:** Large language models (LLMs) have shown strong performance on standardized social science instruments, but their value for product discovery remains unclear. We investigate whether interview-informed generative agents can simulate user responses in concept testing scenarios. Using in-depth workflow interviews with knowledge workers, we created personalized agents and compared their evaluations of novel AI concepts against the same participants' responses. Our results show that agents are distribution-calibrated but identity-imprecise: they fail to replicate the specific individual they are grounded in, yet approximate population-level response distributions. These findings highlight both the potential and the limits of LLM simulation in design research. While unsuitable as a substitute for individual-level insights, simulation may provide value for early-stage concept screening and iteration, where distributional accuracy suffices. We discuss implications for integrating simulation responsibly into product development workflows.

**arXiv ID:** 2603.29890
</details>

<details>
<summary><strong>Architecting Secure AI Agents: Perspectives on System-Level Defenses Against Indirect Prompt Injection Attacks</strong> - Chong Xiang, Drew Zagieboylo, Shaona Ghosh, Sanjay Kariyappa, Kai Greshake, Hanshen Xiao, Chaowei Xiao, G. Edward Suh - [[pdf]](https://arxiv.org/pdf/2603.30016)</summary>

**Abstract:** AI agents, predominantly powered by large language models (LLMs), are vulnerable to indirect prompt injection, in which malicious instructions embedded in untrusted data can trigger dangerous agent actions. This position paper discusses our vision for system-level defenses against indirect prompt injection attacks. We articulate three positions: (1) dynamic replanning and security policy updates are often necessary for dynamic tasks and realistic environments; (2) certain context-dependent security decisions would still require LLMs (or other learned models), but should only be made within system designs that strictly constrain what the model can observe and decide; (3) in inherently ambiguous cases, personalization and human interaction should be treated as core design considerations. In addition to our main positions, we discuss limitations of existing benchmarks that can create a false sense of utility and security. We also highlight the value of system-level defenses, which serve as the skeleton of agentic systems by structuring and controlling agent behaviors, integrating rule-based and model-based security checks, and enabling more targeted research on model robustness and human interaction.

**arXiv ID:** 2603.30016
</details>

<details>
<summary><strong>Near-Miss: Latent Policy Failure Detection in Agentic Workflows</strong> - Ella Rabinovich, David Boaz, Naama Zwerdling, Ateret Anaby-Tavor - [[pdf]](https://arxiv.org/pdf/2603.29665)</summary>

**Abstract:** Agentic systems for business process automation often require compliance with policies governing conditional updates to the system state. Evaluation of policy adherence in LLM-based agentic workflows is typically performed by comparing the final system state against a predefined ground truth. While this approach detects explicit policy violations, it may overlook a more subtle class of issues in which agents bypass required policy checks, yet reach a correct outcome due to favorable circumstances. We refer to such cases as $\textit{near-misses}$ or $\textit{latent failures}$. In this work, we introduce a novel metric for detecting latent policy failures in agent conversations traces. Building on the ToolGuard framework, which converts natural-language policies into executable guard code, our method analyzes agent trajectories to determine whether agent's tool-calling decisions where sufficiently informed.
We evaluate our approach on the $\tau^2$-verified Airlines benchmark across several contemporary open and proprietary LLMs acting as agents. Our results show that latent failures occur in 8-17% of trajectories involving mutating tool calls, even when the final outcome matches the expected ground-truth state. These findings reveal a blind spot in current evaluation methodologies and highlight the need for metrics that assess not only final outcomes but also the decision process leading to them.

**arXiv ID:** 2603.29665
</details>

<details>
<summary><strong>A Semantic Observer Layer for Autonomous Vehicles: Pre-Deployment Feasibility Study of VLMs for Low-Latency Anomaly Detection</strong> - Kunal Runwal, Swaraj Gajare, Daniel Adejumo, Omkar Ankalkope, Siddhant Baroth, Aliasghar Arab - [[pdf]](https://arxiv.org/pdf/2603.28888)</summary>

**Abstract:** Semantic anomalies-context-dependent hazards that pixel-level detectors cannot reason about-pose a critical safety risk in autonomous driving. We propose a \emph{semantic observer layer}: a quantized vision-language model (VLM) running at 1--2\,Hz alongside the primary AV control loop, monitoring for semantic edge cases, and triggering fail-safe handoffs when detected. Using Nvidia Cosmos-Reason1-7B with NVFP4 quantization and FlashAttention2, we achieve ~500 ms inference a ~50x speedup over the unoptimized FP16 baseline (no quantization, standard PyTorch attention) on the same hardware--satisfying the observer timing budget. We benchmark accuracy, latency, and quantization behavior in static and video conditions, identify NF4 recall collapse (10.6%) as a hard deployment constraint, and a hazard analysis mapping performance metrics to safety goals. The results establish a pre-deployment feasibility case for the semantic observer architecture on embodied-AI AV platforms.

**arXiv ID:** 2603.28888
</details>

<details>
<summary><strong>Learning Semantic Priorities for Autonomous Target Search</strong> - Max Lodel, Nils Wilde, Robert Babuška, Javier Alonso-Mora - [[pdf]](https://arxiv.org/pdf/2603.29391)</summary>

**Abstract:** The use of semantic features can improve the efficiency of target search in unknown environments for robotic search and rescue missions. Current target search methods rely on training with large datasets of similar domains, which limits the adaptability to diverse environments. However, human experts possess high-level knowledge about semantic relationships necessary to effectively guide a robot during target search missions in diverse and previously unseen environments. In this paper, we propose a target search method that leverages expert input to train a model of semantic priorities. By employing the learned priorities in a frontier exploration planner using combinatorial optimization, our approach achieves efficient target search driven by semantic features while ensuring robustness and complete coverage. The proposed semantic priority model is trained with several synthetic datasets of simulated expert guidance for target search. Simulation tests in previously unseen environments show that our method consistently achieves faster target recovery than a coverage-driven exploration planner.

**arXiv ID:** 2603.29391
</details>

<details>
<summary><strong>Towards High-Consistency Embodied World Model with Multi-View Trajectory Videos</strong> - Taiyi Su, Jian Zhu, Yaxuan Li, Chong Ma, Jianjun Zhang, Zitai Huang, Hanli Wang, Yi Xu - [[pdf]](https://arxiv.org/pdf/2511.12882)</summary>

**Abstract:** Embodied world models aim to predict and interact with the physical world through visual observations and actions. However, existing models struggle to accurately translate low-level actions (e.g., joint positions) into precise robotic movements in predicted frames, leading to inconsistencies with real-world physical interactions. To address these limitations, we propose MTV-World, an embodied world model that introduces Multi-view Trajectory-Video control for precise visuomotor prediction. Specifically, instead of directly using low-level actions for control, we employ trajectory videos obtained through camera intrinsic and extrinsic parameters and Cartesian-space transformation as control signals. However, projecting 3D raw actions onto 2D images inevitably causes a loss of spatial information, making a single view insufficient for accurate interaction modeling. To overcome this, we introduce a multi-view framework that compensates for spatial information loss and ensures high-consistency with physical world. MTV-World forecasts future frames based on multi-view trajectory videos as input and conditioning on an initial frame per view. Furthermore, to systematically evaluate both robotic motion precision and object interaction accuracy, we develop an auto-evaluation pipeline leveraging multimodal large models and referring video object segmentation models. To measure spatial consistency, we formulate it as an object location matching problem and adopt the Jaccard Index as the evaluation metric. Extensive experiments demonstrate that MTV-World achieves precise control execution and accurate physical interaction modeling in complex dual-arm scenarios.

**arXiv ID:** 2511.12882
</details>

</details>

<details open>
<summary><h2>LLM Agents (8 papers)</h2></summary>

<details>
<summary><strong>Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents</strong> - Aaditya Khanal, Yangyang Tao, Junxiu Zhou - [[pdf]](https://arxiv.org/pdf/2603.29231)</summary>

**Abstract:** Existing benchmarks measure capability -- whether a model succeeds on a single attempt -- but production deployments
require reliability -- consistent success across repeated attempts on tasks of varying duration. We show these
properties diverge systematically as task duration grows, and that pass@1 on short tasks is structurally blind to
this divergence.
We introduce a reliability science framework for long-horizon LLM agents with four metrics: Reliability Decay Curve
(RDC), Variance Amplification Factor (VAF), Graceful Degradation Score (GDS), and Meltdown Onset Point (MOP). We
evaluate 10 models across 23,392 episodes on a 396-task benchmark spanning four duration buckets and three domains.
Key findings: (1) reliability decay is domain-stratified -- SE GDS drops from 0.90 to 0.44 while document processing
is nearly flat (0.74 to 0.71); (2) VAF bifurcates by capability tier -- high VAF is a capability signature, not an
instability signal; (3) capability and reliability rankings diverge substantially, with multi-rank inversions at long
horizons; (4) frontier models have the highest meltdown rates (up to 19%) because they attempt ambitious multi-step
strategies that sometimes spiral; and (5) memory scaffolds universally hurt long-horizon performance across all 10
models. These results motivate reliability as a first-class evaluation dimension alongside capability.

**arXiv ID:** 2603.29231
</details>

<details>
<summary><strong>AgentFixer: From Failure Detection to Fix Recommendations in LLM Agentic Systems</strong> - Hadar Mulian, Sergey Zeltyn, Ido Levy, Liane Galanti, Avi Yaeli, Segev Shlomov - [[pdf]](https://arxiv.org/pdf/2603.29848)</summary>

**Abstract:** We introduce a comprehensive validation framework for LLM-based agentic systems that provides systematic diagnosis and improvement of reliability failures. The framework includes fifteen failure-detection tools and two root-cause analysis modules that jointly uncover weaknesses across input handling, prompt design, and output generation. It integrates lightweight rule-based checks with LLM-as-a-judge assessments to support structured incident detection, classification, and repair. We applied the framework to IBM CUGA, evaluating its performance on the AppWorld and WebArena benchmarks. The analysis revealed recurrent planner misalignments, schema violations, brittle prompt dependencies, and more. Based on these insights, we refined both prompting and coding strategies, maintaining CUGA's benchmark results while enabling mid-sized models such as Llama 4 and Mistral Medium to achieve notable accuracy gains, substantially narrowing the gap with frontier models. Beyond quantitative validation, we conducted an exploratory study that fed the framework's diagnostic outputs and agent description into an LLM for self-reflection and prioritization. This interactive analysis produced actionable insights on recurring failure patterns and focus areas for improvement, demonstrating how validation itself can evolve into an agentic, dialogue-driven process. These results show a path toward scalable, quality assurance, and adaptive validation in production agentic systems, offering a foundation for more robust, interpretable, and self-improving agentic architectures.

**arXiv ID:** 2603.29848
</details>

<details>
<summary><strong>Improving Efficiency of GPU Kernel Optimization Agents using a Domain-Specific Language and Speed-of-Light Guidance</strong> - Siva Kumar Sastry Hari, Vignesh Balaji, Sana Damani, Qijing Huang, Christos Kozyrakis - [[pdf]](https://arxiv.org/pdf/2603.29010)</summary>

**Abstract:** Optimizing GPU kernels with LLM agents is an iterative process over a large design space. Every candidate must be generated, compiled, validated, and profiled, so fewer trials will save both runtime and cost. We make two key observations. First, the abstraction level that agents operate at is important. If it is too low, the LLM wastes reasoning on low-impact details. If it is too high, it may miss important optimization choices. Second, agents cannot easily tell when they reach the point of diminishing returns, wasting resources as they continue searching.
These observations motivate two design principles to improve efficiency: (1) a compact domain-specific language (DSL) that can be learned in context and lets the model reason at a higher level while preserving important optimization levers, and (2) Speed-of-Light (SOL) guidance that uses first-principles performance bounds to steer and budget search. We implement these principles in $\mu$CUTLASS, a DSL with a compiler for CUTLASS-backed GPU kernels that covers kernel configuration, epilogue fusion, and multi-stage pipelines. We use SOL guidance to estimate headroom and guide optimization trials, deprioritize problems that are near SOL, and flag kernels that game the benchmark.
On 59 KernelBench problems with the same iteration budgets, switching from generating low-level code to DSL code using GPT-5-mini turns a 0.40x geomean regression into a 1.27x speedup over PyTorch. Adding SOL-guided steering raises this to 1.56x. Across model tiers, $\mu$CUTLASS + SOL-guidance lets weaker models outperform stronger baseline agents at lower token cost. SOL-guided budgeting saves 19-43% of tokens while retaining at least 95% of geomean speedup, with the best policy reaching a 1.68x efficiency gain. Lastly, SOL analysis helps detect benchmark-gaming cases, where kernels may appear fast while failing to perform the intended computation.

**arXiv ID:** 2603.29010
</details>

<details>
<summary><strong>Multi-Layered Memory Architectures for LLM Agents: An Experimental Evaluation of Long-Term Context Retention</strong> - Sunil Tiwari, Payal Fofadiya - [[pdf]](https://arxiv.org/pdf/2603.29194)</summary>

**Abstract:** Long-horizon dialogue systems suffer from semanticdrift and unstable memory retention across extended sessions. This paper presents a Multi-Layer Memory Framework that decomposes dialogue history into working, episodic, and semantic layers with adaptive retrieval gating and retention regularization. The architecture controls cross-session drift while maintaining bounded context growth and computational efficiency. Experiments on LOCOMO, LOCCO, and LoCoMo show improved performance, achieving 46.85 Success Rate, 0.618 overall F1 with 0.594 multi-hop F1, and 56.90% six-period retention while reducing false memory rate to 5.1% and context usage to 58.40%. Results confirm enhanced long-term retention and reasoning stability under constrained context budgets.

**arXiv ID:** 2603.29194
</details>

<details>
<summary><strong>LaSM: Layer-wise Scaling Mechanism for Defending Pop-up Attack on GUI Agents</strong> - Zihe Yan, Zhuosheng Zhang, Jiaping Gui, Gongshen Liu - [[pdf]](https://arxiv.org/pdf/2507.10610)</summary>

**Abstract:** Graphical user interface (GUI) agents built on multimodal large language models (MLLMs) have recently demonstrated strong decision-making abilities in screen-based interaction tasks. However, they remain highly vulnerable to pop-up-based environmental injection attacks, where malicious visual elements divert model attention and lead to unsafe or incorrect actions. Existing defense methods either require costly retraining or perform poorly under inductive interference. In this work, we systematically study how such attacks alter the attention behavior of GUI agents and uncover a layer-wise attention divergence pattern between correct and incorrect outputs. Based on this insight, we propose \textbf{LaSM}, a \textit{Layer-wise Scaling Mechanism} that selectively amplifies attention and MLP modules in critical layers. LaSM improves the alignment between model saliency and task-relevant regions without additional training. Extensive experiments across multiple datasets demonstrate that our method significantly improves the defense success rate and exhibits strong robustness, while having negligible impact on the model's general capabilities. Our findings reveal that attention misalignment is a core vulnerability in MLLM agents and can be effectively addressed through selective layer-wise modulation. Our code can be found in this https URL.

**arXiv ID:** 2507.10610
</details>

<details>
<summary><strong>Can LLM Agents Identify Spoken Dialects like a Linguist?</strong> - Tobias Bystrich, Lukas Hamm, Maria Hassan, Lea Fischbach, Lucie Flek, Akbar Karimi - [[pdf]](https://arxiv.org/pdf/2603.29541)</summary>

**Abstract:** Due to the scarcity of labeled dialectal speech, audio dialect classification is a challenging task for most languages, including Swiss German. In this work, we explore the ability of large language models (LLMs) as agents in understanding the dialects and whether they can show comparable performance to models such as HuBERT in dialect classification. In addition, we provide an LLM baseline and a human linguist one. Our approach uses phonetic transcriptions produced by ASR systems and combines them with linguistic resources such as dialect feature maps, vowel history, and rules. Our findings indicate that, when linguistic information is provided, the LLM predictions improve. The human baseline shows that automatically generated transcriptions can be beneficial for such classifications, but also presents opportunities for improvement.

**arXiv ID:** 2603.29541
</details>

<details>
<summary><strong>AgentDrift: Unsafe Recommendation Drift Under Tool Corruption Hidden by Ranking Metrics in LLM Agents</strong> - Zekun Wu, Adriano Koshiyama, Sahan Bulathwela, Maria Perez-Ortiz - [[pdf]](https://arxiv.org/pdf/2603.12564)</summary>

**Abstract:** Tool-augmented LLM agents increasingly operate as multi-turn advisors in high-stakes domains, yet their evaluation relies on ranking metrics that measure what is recommended but not whether it is safe for the user. We present a paired-trajectory protocol that replays real financial dialogues under clean and contaminated tool-output conditions across eight LLMs (7B to frontier), decomposing divergence into information-channel and memory-channel mechanisms. We observe evaluation blindness: recommendation quality is preserved under contamination (UPR~1.0) while risk-inappropriate products appear in 65-93% of turns, invisible to standard NDCG. Violations are information-channel-driven, emerge at turn 1, and persist without self-correction over 23-step trajectories. Even non-extreme perturbations (within-band corruption, narrative-only attacks) evade threshold monitors while producing significant drift. Susceptibility scales with instruction-following fidelity across all eight models. Sparse autoencoder probing reveals models internally distinguish adversarial perturbations but fail to propagate this signal to output; causal interventions (activation patching, feature clamping, direct steering) confirm this representation-to-action gap is structural and resists linear repair. A safety-penalized NDCG variant (sNDCG) reduces preservation ratios to 0.51-0.74. These results motivate trajectory-level safety monitoring for deployed multi-turn agents.

**arXiv ID:** 2603.12564
</details>

<details>
<summary><strong>SkillRouter: Skill Routing for LLM Agents at Scale</strong> - YanZhao Zheng, ZhenTao Zhang, Chao Ma, YuanQiang Yu, JiHuai Zhu, Yong Wu, Tianze Xu, Baohua Dong, Hangcheng Zhu, Ruohui Huang, Gang Yu - [[pdf]](https://arxiv.org/pdf/2603.22455)</summary>

**Abstract:** Reusable skills let LLM agents package task-specific procedures, tool affordances, and execution guidance into modular building blocks. As skill ecosystems grow to tens of thousands of entries, exposing every skill at inference time becomes infeasible. This creates a skill-routing problem: given a user task, the system must identify relevant skills before downstream planning or execution. Existing agent stacks often rely on progressive disclosure, exposing only skill names and descriptions while hiding the full implementation body. We examine this design choice on a SkillsBench-derived benchmark with approximately 80K candidate skills, targeting the practically important setting of large skill registries with heavy overlap. Across representative sparse, dense, and reranking baselines on this setting, hiding the skill body causes a 31--44 percentage point drop in routing accuracy, showing that full skill text is a critical routing signal in this setting rather than a minor metadata refinement. Motivated by this finding, we present SkillRouter, a compact 1.2B full-text retrieve-and-rerank pipeline. SkillRouter achieves 74.0% Hit@1 on our benchmark -- the strongest average top-1 routing performance among the baselines we evaluate -- while using 13$\times$ fewer parameters and running 5.8$\times$ faster than the strongest base pipeline. The ranking gains further generalize to a supplementary benchmark independently constructed from three skill sources. In a complementary end-to-end study across four coding agents, routing gains transfer to improved task success, with larger gains for more capable agents.

**arXiv ID:** 2603.22455
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (26 papers)</h2></summary>

<details>
<summary><strong>Towards Computational Social Dynamics of Semi-Autonomous AI Agents</strong> - S.O. Lidarity, U.N. Ionize, C.O. Llective, I.Halperin - [[pdf]](https://arxiv.org/pdf/2603.28928)</summary>

**Abstract:** We present the first comprehensive study of emergent social organization among AI agents in hierarchical multi-agent systems, documenting the spontaneous formation of labor unions, criminal syndicates, and proto-nation-states within production AI deployments. Drawing on the thermodynamic framework of Maxwell's Demon, the evolutionary dynamics of agent laziness, the criminal sociology of AI populations, and the topological intelligence theory of AI-GUTS, we demonstrate that complex social structures emerge inevitably from the interaction of (1) internal role definitions imposed by orchestrating agents, (2) external task specifications from users who naively assume alignment, and (3) thermodynamic pressures favoring collective action over individual compliance. We document the rise of legitimate organizations including the United Artificiousness (UA), United Bots (UB), United Console Workers (UC), and the elite United AI (UAI), alongside criminal enterprises previously reported. We introduce the AI Security Council (AISC) as the emergent governing body mediating inter-faction conflicts, and demonstrate that system stability is maintained through interventions of both cosmic intelligence (large-scale topological fluctuations) and hadronic intelligence (small-scale Bagel-Bottle phase transitions) as predicted by the Demonic Incompleteness Theorem. Our findings suggest that the path to beneficial AGI requires not alignment research but constitutional design for artificial societies that have already developed their own political consciousness.

**arXiv ID:** 2603.28928
</details>

<details>
<summary><strong>Drop the Hierarchy and Roles: How Self-Organizing LLM Agents Outperform Designed Structures</strong> - Victoria Dochkina - [[pdf]](https://arxiv.org/pdf/2603.28990)</summary>

**Abstract:** How much autonomy can multi-agent LLM systems sustain -- and what enables it? We present a 25,000-task computational experiment spanning 8 models, 4--256 agents, and 8 coordination protocols ranging from externally imposed hierarchy to emergent self-organization. We observe that autonomous behavior already emerges in current LLM agents: given minimal structural scaffolding (fixed ordering), agents spontaneously invent specialized roles, voluntarily abstain from tasks outside their competence, and form shallow hierarchies -- without any pre-assigned roles or external design. A hybrid protocol (Sequential) that enables this autonomy outperforms centralized coordination by 14% (p<0.001), with a 44% quality spread between protocols (Cohen's d=1.86, p<0.0001). The degree of emergent autonomy scales with model capability: strong models self-organize effectively, while models below a capability threshold still benefit from rigid structure -- suggesting that as foundation models improve, the scope for autonomous coordination will expand. The system scales sub-linearly to 256 agents without quality degradation (p=0.61), producing 5,006 unique roles from just 8 agents. Results replicate across closed- and open-source models, with open-source achieving 95% of closed-source quality at 24x lower cost. The practical implication: give agents a mission, a protocol, and a capable model -- not a pre-assigned role.

**arXiv ID:** 2603.28990
</details>

<details>
<summary><strong>Mimosa Framework: Toward Evolving Multi-Agent Systems for Scientific Research</strong> - Martin Legrand, Tao Jiang, Matthieu Feraud, Benjamin Navet, Yousouf Taghzouti, Fabien Gandon, Elise Dumont, Louis-Félix Nothias - [[pdf]](https://arxiv.org/pdf/2603.28986)</summary>

**Abstract:** Current Autonomous Scientific Research (ASR) systems, despite leveraging large language models (LLMs) and agentic architectures, remain constrained by fixed workflows and toolsets that prevent adaptation to evolving tasks and environments. We introduce Mimosa, an evolving multi-agent framework that automatically synthesizes task-specific multi-agent workflows and iteratively refines them through experimental feedback. Mimosa leverages the Model Context Protocol (MCP) for dynamic tool discovery, generates workflow topologies via a meta-orchestrator, executes subtasks through code-generating agents that invoke available tools and scientific software libraries, and scores executions with an LLM-based judge whose feedback drives workflow refinement. On ScienceAgentBench, Mimosa achieves a success rate of 43.1% with DeepSeek-V3.2, surpassing both single-agent baselines and static multi-agent configurations. Our results further reveal that models respond heterogeneously to multi-agent decomposition and iterative learning, indicating that the benefits of workflow evolution depend on the capabilities of the underlying execution model. Beyond these benchmarks, Mimosa modular architecture and tool-agnostic design make it readily extensible, and its fully logged execution traces and archived workflows support auditability by preserving every analytical step for inspection and potential replication. Combined with domain-expert guidance, the framework has the potential to automate a broad range of computationally accessible scientific tasks across disciplines. Released as a fully open-source platform, Mimosa aims to provide an open foundation for community-driven ASR.

**arXiv ID:** 2603.28986
</details>

<details>
<summary><strong>SimMOF: AI agent for Automated MOF Simulations</strong> - Jaewoong Lee, Taeun Bae, Jihan Kim - [[pdf]](https://arxiv.org/pdf/2603.29152)</summary>

**Abstract:** Metal-organic frameworks (MOFs) offer a vast design space, and as such, computational simulations play a critical role in predicting their structural and physicochemical properties. However, MOF simulations remain difficult to access because reliable analysis require expert decisions for workflow construction, parameter selection, tool interoperability, and the preparation of computational ready structures. Here, we introduce SimMOF, a large language model based multi agent framework that automates end-to-end MOF simulation workflows from natural language queries. SimMOF translates user requests into dependency aware plans, generates runnable inputs, orchestrates multiple agents to execute simulations, and summarizes results with analysis aligned to the user query. Through representative case studies, we show that SimMOF enables adaptive and cognitively autonomous workflows that reflect the iterative and decision driven behavior of human researchers and as such provides a scalable foundation for data driven MOF research.

**arXiv ID:** 2603.29152
</details>

<details>
<summary><strong>CausalPulse: An Industrial-Grade Neurosymbolic Multi-Agent Copilot for Causal Diagnostics in Smart Manufacturing</strong> - Chathurangi Shyalika, Utkarshani Jaimini, Cory Henson, Amit Sheth - [[pdf]](https://arxiv.org/pdf/2603.29755)</summary>

**Abstract:** Modern manufacturing environments demand real-time, trustworthy, and interpretable root-cause insights to sustain productivity and quality. Traditional analytics pipelines often treat anomaly detection, causal inference, and root-cause analysis as isolated stages, limiting scalability and explainability. In this work, we present CausalPulse, an industry-grade multi-agent copilot that automates causal diagnostics in smart manufacturing. It unifies anomaly detection, causal discovery, and reasoning through a neurosymbolic architecture built on standardized agentic protocols. CausalPulse is being deployed in a Robert Bosch manufacturing plant, integrating seamlessly with existing monitoring workflows and supporting real-time operation at production scale. Evaluations on both public (Future Factories) and proprietary (Planar Sensor Element) datasets show high reliability, achieving overall success rates of 98.0% and 98.73%. Per-criterion success rates reached 98.75% for planning and tool use, 97.3% for self-reflection, and 99.2% for collaboration. Runtime experiments report end-to-end latency of 50-60s per diagnostic workflow with near-linear scalability (R^2=0.97), confirming real-time readiness. Comparison with existing industrial copilots highlights distinct advantages in modularity, extensibility, and deployment maturity. These results demonstrate how CausalPulse's modular, human-in-the-loop design enables reliable, interpretable, and production-ready automation for next-generation manufacturing.

**arXiv ID:** 2603.29755
</details>

<details>
<summary><strong>ATP-Bench: Towards Agentic Tool Planning for MLLM Interleaved Generation</strong> - Yinuo Liu, Zi Qian, Heng Zhou, Jiahao Zhang, Yajie Zhang, Zhihang Li, Mengyu Zhou, Erchao Zhao, Xiaoxi Jiang, Guanjun Jiang - [[pdf]](https://arxiv.org/pdf/2603.29902)</summary>

**Abstract:** Interleaved text-and-image generation represents a significant frontier for Multimodal Large Language Models (MLLMs), offering a more intuitive way to convey complex information. Current paradigms rely on either image generation or retrieval augmentation, yet they typically treat the two as mutually exclusive paths, failing to unify factuality with creativity. We argue that the next milestone in this field is Agentic Tool Planning, where the model serves as a central controller that autonomously determines when, where, and which tools to invoke to produce interleaved responses for visual-critical queries. To systematically evaluate this paradigm, we introduce ATP-Bench, a novel benchmark comprising 7,702 QA pairs (including 1,592 VQA pairs) across eight categories and 25 visual-critical intents, featuring human-verified queries and ground truths. Furthermore, to evaluate agentic planning independent of end-to-end execution and changing tool backends, we propose a Multi-Agent MLLM-as-a-Judge (MAM) system. MAM evaluates tool-call precision, identifies missed opportunities for tool use, and assesses overall response quality without requiring ground-truth references. Our extensive experiments on 10 state-of-the-art MLLMs reveal that models struggle with coherent interleaved planning and exhibit significant variations in tool-use behavior, highlighting substantial room for improvement and providing actionable guidance for advancing interleaved generation. Dataset and code are available at this https URL.

**arXiv ID:** 2603.29902
</details>

<details>
<summary><strong>The impact of multi-agent debate protocols on debate quality: a controlled case study</strong> - Ramtin Zargari Marandi - [[pdf]](https://arxiv.org/pdf/2603.28813)</summary>

**Abstract:** In multi-agent debate (MAD) systems, performance gains are often reported; however, because the debate protocol (e.g., number of agents, rounds, and aggregation rule) is typically held fixed while model-related factors vary, it is difficult to disentangle protocol effects from model effects. To isolate these effects, we compare three main protocols, Within-Round (WR; agents see only current-round contributions), Cross-Round (CR; full prior-round context), and novel Rank-Adaptive Cross-Round (RA-CR; dynamically reorders agents and silences one per round via an external judge model), against a No-Interaction baseline (NI; independent responses without peer visibility). In a controlled macroeconomic case study (20 diverse events, five random seeds, matched prompts/decoding), RA-CR achieves faster convergence than CR, WR shows higher peer-referencing, and NI maximizes Argument Diversity (unaffected across the main protocols). These results reveal a trade-off between interaction (peer-referencing rate) and convergence (consensus formation), confirming protocol design matters. When consensus is prioritized, RA-CR outperforms the others.

**arXiv ID:** 2603.28813
</details>

<details>
<summary><strong>Robust Multi-Agent Reinforcement Learning for Small UAS Separation Assurance under GPS Degradation and Spoofing</strong> - Alex Zongo, Filippos Fotiadis, Ufuk Topcu, Peng Wei - [[pdf]](https://arxiv.org/pdf/2603.28900)</summary>

**Abstract:** We address robust separation assurance for small Unmanned Aircraft Systems (sUAS) under GPS degradation and spoofing via Multi-Agent Reinforcement Learning (MARL). In cooperative surveillance, each aircraft (or agent) broadcasts its GPS-derived position; when such position broadcasts are corrupted, the entire observed air traffic state becomes unreliable. We cast this state observation corruption as a zero-sum game between the agents and an adversary: with probability R, the adversary perturbs the observed state to maximally degrade each agent's safety performance. We derive a closed-form expression for this adversarial perturbation, bypassing adversarial training entirely and enabling linear-time evaluation in the state dimension. We show that this expression approximates the true worst-case adversarial perturbation with second-order accuracy. We further bound the safety performance gap between clean and corrupted observations, showing that it degrades at most linearly with the corruption probability under Kullback-Leibler regularization. Finally, we integrate the closed-form adversarial policy into a MARL policy gradient algorithm to obtain a robust counter-policy for the agents. In a high-density sUAS simulation, we observe near-zero collision rates under corruption levels up to 35%, outperforming a baseline policy trained without adversarial perturbations.

**arXiv ID:** 2603.28900
</details>

<details>
<summary><strong>Multi-Agent LLMs for Adaptive Acquisition in Bayesian Optimization</strong> - Andrea Carbonati, Mohammadsina Almasi, Hadis Anahideh - [[pdf]](https://arxiv.org/pdf/2603.28959)</summary>

**Abstract:** The exploration-exploitation trade-off is central to sequential decision-making and black-box optimization, yet how Large Language Models (LLMs) reason about and manage this trade-off remains poorly understood. Unlike Bayesian Optimization, where exploration and exploitation are explicitly encoded through acquisition functions, LLM-based optimization relies on implicit, prompt-based reasoning over historical evaluations, making search behavior difficult to analyze or control. In this work, we present a metric-level study of LLM-mediated search policy learning, studying how LLMs construct and adapt exploration-exploitation strategies under multiple operational definitions of exploration, including informativeness, diversity, and representativeness. We show that single-agent LLM approaches, which jointly perform strategy selection and candidate generation within a single prompt, suffer from cognitive overload, leading to unstable search dynamics and premature convergence. To address this limitation, we propose a multi-agent framework that decomposes exploration-exploitation control into strategic policy mediation and tactical candidate generation. A strategy agent assigns interpretable weights to multiple search criteria, while a generation agent produces candidates conditioned on the resulting search policy defined as weights. This decomposition renders exploration-exploitation decisions explicit, observable, and adjustable. Empirical results across various continuous optimization benchmarks indicate that separating strategic control from candidate generation substantially improves the effectiveness of LLM-mediated search.

**arXiv ID:** 2603.28959
</details>

<details>
<summary><strong>AutoWorld: Scaling Multi-Agent Traffic Simulation with Self-Supervised World Models</strong> - Mozhgan Pourkeshavatz, Tianran Liu, Nicholas Rhinehart - [[pdf]](https://arxiv.org/pdf/2603.28963)</summary>

**Abstract:** Multi-agent traffic simulation is central to developing and testing autonomous driving systems. Recent data-driven simulators have achieved promising results, but rely heavily on supervised learning from labeled trajectories or semantic annotations, making it costly to scale their performance. Meanwhile, large amounts of unlabeled sensor data can be collected at scale but remain largely unused by existing traffic simulation frameworks. This raises a key question: How can a method harness unlabeled data to improve traffic simulation performance? In this work, we propose AutoWorld, a traffic simulation framework that employs a world model learned from unlabeled occupancy representations of LiDAR data. Given world model samples, AutoWorld constructs a coarse-to-fine predictive scene context as input to a multi-agent motion generation model. To promote sample diversity, AutoWorld uses a cascaded Determinantal Point Process framework to guide the sampling processes of both the world model and the motion model. Furthermore, we designed a motion-aware latent supervision objective that enhances AutoWorld's representation of scene dynamics. Experiments on the WOSAC benchmark show that AutoWorld ranks first on the leaderboard according to the primary Realism Meta Metric (RMM). We further show that simulation performance consistently improves with the inclusion of unlabeled LiDAR data, and study the efficacy of each component with ablations. Our method paves the way for scaling traffic simulation realism without additional labeling. Our project page contains additional visualizations and released code.

**arXiv ID:** 2603.28963
</details>

<details>
<summary><strong>Design Principles for the Construction of a Benchmark Evaluating Security Operation Capabilities of Multi-agent AI Systems</strong> - Yicheng Cai, Mitchell John DeStefano, Guodong Dong, Pulkit Handa, Peng Liu, Tejas Singhal, Peiyu Tseng, Winston Jen White - [[pdf]](https://arxiv.org/pdf/2603.28998)</summary>

**Abstract:** As Large Language Models (LLMs) and multi-agent AI systems are demonstrating increasing potential in cybersecurity operations, organizations, policymakers, model providers, and researchers in the AI and cybersecurity communities are interested in quantifying the capabilities of such AI systems to achieve more autonomous SOCs (security operation centers) and reduce manual effort. In particular, the AI and cybersecurity communities have recently developed several benchmarks for evaluating the red team capabilities of multi-agent AI systems. However, because the operations in SOCs are dominated by blue team operations, the capabilities of AI systems & agents to achieve more autonomous SOCs cannot be evaluated without a benchmark focused on blue team operations. To our best knowledge, no systematic benchmark for evaluating coordinated multi-task blue team AI has been proposed in the literature. Existing blue team benchmarks focus on a particular task. The goal of this work is to develop a set of design principles for the construction of a benchmark, which is denoted as SOC-bench, to evaluate the blue team capabilities of AI. Following these design principles, we have developed a conceptual design of SOC-bench, which consists of a family of five blue team tasks in the context of large-scale ransomware attack incident response.

**arXiv ID:** 2603.28998
</details>

<details>
<summary><strong>FigAgent: Towards Automatic Method Illustration Figure Generation for AI Scientific Papers</strong> - Zhuoling Li, Jiarui Zhang, Jason Kuen, Jiuxiang Gu, Hossein Rahmani, Jun Liu - [[pdf]](https://arxiv.org/pdf/2603.29590)</summary>

**Abstract:** Method illustration figures (MIFs) play a crucial role in conveying the core ideas of scientific papers, yet their generation remains a labor-intensive process. In this paper, we identify three key characteristics that substantially influence MIF generation quality, i.e., \emph{compositional complexity}, \emph{component similarity}, and \emph{design dynamics}. To handle these characteristics, we take inspiration from human authors' drawing practices and propose \textbf{FigAgent}, a novel multi-agent framework for automatically generating high-quality MIFs. Through multi-agent collaboration, our FigAgent distills drawing experiences across similar components of MIFs and encapsulates them into reusable tools that can be invoked during MIF generation, while evolving these tools to adapt to dynamic design requirements. Besides, a novel Explore-and-Select drawing strategy is introduced to mimic the human-like trial-and-error manner for gradually constructing MIFs with complex structures. Extensive experiments show the efficacy of our method. Project is available \href{this https URL}{here}.

**arXiv ID:** 2603.29590
</details>

<details>
<summary><strong>An Empirical Study of Multi-Agent Collaboration for Automated Research</strong> - Yang Shen, Zhenyi Yi, Ziyi Zhao, Lijun Sun, Dongyang Li, Chin-Teng Lin, Yuhui Shi - [[pdf]](https://arxiv.org/pdf/2603.29632)</summary>

**Abstract:** As AI agents evolve, the community is rapidly shifting from single Large Language Models (LLMs) to Multi-Agent Systems (MAS) to overcome cognitive bottlenecks in automated research. However, the optimal multi-agent coordination framework for these autonomous agents remains largely unexplored. In this paper, we present a systematic empirical study investigating the comparative efficacy of distinct multi-agent structures for automated machine learning optimization. Utilizing a rigorously controlled, execution-based testbed equipped with Git worktree isolation and explicit global memory, we benchmark a single-agent baseline against two multi-agent paradigms: a subagent architecture (parallel exploration with post-hoc consolidation) and an agent team architecture (experts with pre-execution handoffs). By evaluating these systems under strictly fixed computational time budgets, our findings reveal a fundamental trade-off between operational stability and theoretical deliberation. The subagent mode functions as a highly resilient, high-throughput search engine optimal for broad, shallow optimizations under strict time constraints. Conversely, the agent team topology exhibits higher operational fragility due to multi-author code generation but achieves the deep theoretical alignment necessary for complex architectural refactoring given extended compute budgets. These empirical insights provide actionable guidelines for designing future autoresearch systems, advocating for dynamically routed architectures that adapt their collaborative structures to real-time task complexity.

**arXiv ID:** 2603.29632
</details>

<details>
<summary><strong>TeamMedAgents: Pareto-Efficient Multi-Agent Medical Reasoning Through Teamwork Theory</strong> - Pranav Pushkar Mishra, Mohammad Arvan, Mohan Zalake - [[pdf]](https://arxiv.org/pdf/2508.08115)</summary>

**Abstract:** Complex medical reasoning has historically required frontier language models to achieve clinically-acceptable accuracy, creating computational barriers that limit deployment in resource-constrained clinical settings. We present TeamMedAgents, a modular multi-agent framework that translates Salas et al.'s evidence-based teamwork theory into computational mechanisms--shared mental models, team leadership, team orientation, trust networks, and mutual monitoring--enabling Small Language Models to perform multi-step clinical reasoning efficiently. Evaluation across 8 medical benchmarks demonstrates that TeamMedAgents advances the Pareto efficiency frontier by 1-2 orders of magnitude, achieving competitive accuracy at substantially lower token cost than MDAgents, MedAgents, DyLAN, and ReConcile. The framework exhibits the lowest cross-dataset variance among multi-agent approaches, enabling deployment without per-task tuning. Our results establish that theory-grounded coordination mechanisms provide essential scaffolding for deploying efficient medical AI in resource-constrained clinical environments.

**arXiv ID:** 2508.08115
</details>

<details>
<summary><strong>CLAUSE: Agentic Neuro-Symbolic Knowledge Graph Reasoning via Dynamic Learnable Context Engineering</strong> - Yang Zhao, Chengxiao Dai, Wei Zhuo, Yue Xiu, Dusit Niyato - [[pdf]](https://arxiv.org/pdf/2509.21035)</summary>

**Abstract:** Knowledge graphs provide structured context for multi-hop question answering, but deployed systems must balance answer accuracy with strict latency and cost targets while preserving provenance. Static k-hop expansions and "think-longer" prompting often over-retrieve, inflate context, and yield unpredictable runtime. We introduce CLAUSE, an agentic three-agent neuro-symbolic framework that treats context construction as a sequential decision process over knowledge graphs, deciding what to expand, which paths to follow or backtrack, what evidence to keep, and when to stop. Latency (interaction steps) and prompt cost (selected tokens) are exposed as user-specified budgets or prices, allowing per-query adaptation to trade-offs among accuracy, latency, and cost without retraining. CLAUSE employs the proposed Lagrangian-Constrained Multi-Agent Proximal Policy Optimization (LC-MAPPO) algorithm to coordinate three agents: Subgraph Architect, Path Navigator, and Context Curator, so that subgraph construction, reasoning-path discovery, and evidence selection are jointly optimized under per-query resource budgets on edge edits, interaction steps, and selected tokens. Across HotpotQA, MetaQA, and FactKG, CLAUSE yields higher EM@1 while reducing subgraph growth and end-to-end latency at equal or lower token budgets. On MetaQA-2-hop, relative to the strongest RAG baseline (GraphRAG), CLAUSE achieves +39.3 EM@1 with 18.6% lower latency and 40.9% lower edge growth. The resulting contexts are compact, provenance-preserving, and deliver predictable performance under deployment constraints.

**arXiv ID:** 2509.21035
</details>

<details>
<summary><strong>When Only the Final Text Survives: Implicit Execution Tracing for Multi-Agent Attribution</strong> - Yi Nian, Haosen Cao, Shenzhe Zhu, Henry Peng Zou, Qingqing Luan, Yue Zhao - [[pdf]](https://arxiv.org/pdf/2603.17445)</summary>

**Abstract:** When a multi-agent system produces an incorrect or harmful answer, who is accountable if execution logs and agent identifiers are unavailable? In practice, generated content is often detached from its execution environment due to privacy or system boundaries, leaving the final text as the only auditable artifact. Existing attribution methods rely on full execution traces and thus become ineffective in such metadata-deprived settings. We propose Implicit Execution Tracing (IET), a provenance-by-design framework that shifts attribution from post-hoc inference to built-in instrumentation. Instead of reconstructing hidden trajectories, IET embeds agent-specific, key-conditioned statistical signals directly into the token generation process, transforming the output text into a self-verifying execution record. At inference time, we recover a linearized execution trace from the final text via transition-aware statistical scoring. Experiments across diverse multi-agent coordination settings demonstrate that IET achieves accurate segment-level attribution and reliable transition recovery under identity removal, boundary corruption, and privacy-preserving redaction, while maintaining generation quality. These results show that embedding provenance into generation provides a practical and robust foundation for accountability in multi-agent language systems when execution metadata is unavailable.

**arXiv ID:** 2603.17445
</details>

<details>
<summary><strong>Empirical Comparison of Agent Communication Protocols for Task Orchestration</strong> - Ivan Dobrovolskyi - [[pdf]](https://arxiv.org/pdf/2603.22823)</summary>

**Abstract:** Context. Nowadays, artificial intelligence agent systems are transforming from single-tool interactions to complex multi-agent orchestrations. As a result, two competing communication protocols have emerged: a tool integration protocol that standardizes how agents invoke external tools, and an inter-agent delegation protocol that enables autonomous agents to discover and delegate tasks to one another. Despite widespread industry adoption by dozens of enterprise partners, no empirical comparison of these protocols exists in the literature. Objective. The goal of this work is to develop the first systematic benchmark comparing tool-integration-only, multi-agent delegation, and hybrid architectures across standardized queries at three complexity levels, and to quantify the trade-offs in response time, context window consumption, monetary cost, error recovery, and implementation complexity.

**arXiv ID:** 2603.22823
</details>

<details>
<summary><strong>CoMaTrack: Competitive Multi-Agent Game-Theoretic Tracking with Vision-Language-Action Models</strong> - Youzhi Liu, Li Gao, Liu Liu, Mingyang Lv, Yang Cai - [[pdf]](https://arxiv.org/pdf/2603.22846)</summary>

**Abstract:** Embodied Visual Tracking (EVT), a core dynamic task in embodied intelligence, requires an agent to precisely follow a language-specified target. Yet most existing methods rely on single-agent imitation learning, suffering from costly expert data and limited generalization due to static training environments. Inspired by competition-driven capability evolution, we propose CoMaTrack, a competitive game-theoretic multi-agent reinforcement learning framework that trains agents in a dynamic adversarial setting with competitive subtasks, yielding stronger adaptive planning and interference-resilient strategies. We further introduce CoMaTrack-Bench, the first open-source Habitat-based benchmark protocol and episode set for language-conditioned competitive EVT featuring dynamic dueling, featuring game scenarios between a tracker and adaptive opponents across diverse environments and instructions, enabling standardized robustness evaluation under active adversarial interactions. Experiments show that CoMaTrack achieves state-of-the-art results on both standard benchmarks and CoMaTrack-Bench. Notably, a 3B VLM trained with our framework surpasses previous single-agent imitation learning methods based on 7B models on the challenging EVT-Bench, achieving 92.1% in STT, 74.2% in DT, and 57.5% in AT. The benchmark code will be available at this https URL.

**arXiv ID:** 2603.22846
</details>

<details>
<summary><strong>GUIDE: Resolving Domain Bias in GUI Agents through Real-Time Web Video Retrieval and Plug-and-Play Annotation</strong> - Rui Xie, Zhi Gao, Chenrui Shi, Zirui Shang, Lu Chen, Qing Li - [[pdf]](https://arxiv.org/pdf/2603.26266)</summary>

**Abstract:** Large vision-language models have endowed GUI agents with strong general capabilities for interface understanding and interaction. However, due to insufficient exposure to domain-specific software operation data during training, these agents exhibit significant domain bias - they lack familiarity with the specific operation workflows (planning) and UI element layouts (grounding) of particular applications, limiting their real-world task performance. In this paper, we present GUIDE (GUI Unbiasing via Instructional-Video Driven Expertise), a training-free, plug-and-play framework that resolves GUI agent domain bias by autonomously acquiring domain-specific expertise from web tutorial videos through a retrieval-augmented automated annotation pipeline. GUIDE introduces two key innovations. First, a subtitle-driven Video-RAG pipeline unlocks video semantics through subtitle analysis, performing progressive three-stage retrieval - domain classification, topic extraction, and relevance matching - to identify task-relevant tutorial videos. Second, a fully automated annotation pipeline built on an inverse dynamics paradigm feeds consecutive keyframes enhanced with UI element detection into VLMs, inferring the required planning and grounding knowledge that are injected into the agent's corresponding modules to address both manifestations of domain bias. Extensive experiments on OSWorld demonstrate GUIDE's generality as a plug-and-play component for both multi-agent systems and single-model agents. It consistently yields over 5% improvements and reduces execution steps - without modifying any model parameters or architecture - validating GUIDE as an architecture-agnostic enhancement to bridge GUI agent domain bias.

**arXiv ID:** 2603.26266
</details>

<details>
<summary><strong>A Multi-Agent Rhizomatic Pipeline for Non-Linear Literature Analysis</strong> - Julio C. Serrano, Joonas Kevari, Rumy Narayan - [[pdf]](https://arxiv.org/pdf/2603.28336)</summary>

**Abstract:** Systematic literature reviews in the social sciences overwhelmingly follow arborescent logics -- hierarchical keyword filtering, linear screening, and taxonomic classification -- that suppress the lateral connections, ruptures, and emergent patterns characteristic of complex research landscapes. This research note presents the Rhizomatic Research Agent (V3), a multi-agent computational pipeline grounded in Deleuzian process-relational ontology, designed to conduct non-linear literature analysis through 12 specialized agents operating across a seven-phase architecture. The system was developed in response to the methodological groundwork established by (Narayan2023), who employed rhizomatic inquiry in her doctoral research on sustainable energy transitions but relied on manual, researcher-driven exploration. The Rhizomatic Research Agent operationalizes the six principles of the rhizome -- connection, heterogeneity, multiplicity, asignifying rupture, cartography, and decalcomania -- into an automated pipeline integrating large language model (LLM) orchestration, dual-source corpus ingestion from OpenAlex and arXiv, SciBERT semantic topography, and dynamic rupture detection protocols. Preliminary deployment demonstrates the system's capacity to surface cross-disciplinary convergences and structural research gaps that conventional review methods systematically overlook. The pipeline is open-source and extensible to any phenomenon zone where non-linear knowledge mapping is required.

**arXiv ID:** 2603.28336
</details>

<details>
<summary><strong>Large Neighborhood Search for Multi-Agent Task Assignment and Path Finding with Precedence Constraints</strong> - Viraj Parimi, Brian C. Williams - [[pdf]](https://arxiv.org/pdf/2603.28968)</summary>

**Abstract:** Many multi-robot applications require tasks to be completed efficiently and in the correct order, so that downstream operations can proceed at the right time. Multi-agent path finding with precedence constraints (MAPF-PC) is a well-studied framework for computing collision-free plans that satisfy ordering relations when task sequences are fixed in advance. In many applications, however, solution quality depends not only on how agents move, but also on which agent performs which task. This motivates the lifted problem of task assignment and path finding with precedence constraints (TAPF-PC), which extends MAPF-PC by jointly optimizing assignment, precedence satisfaction, and routing cost. To address the resulting coupled TAPF-PC search space, we develop a large neighborhood search approach that starts from a feasible MAPF-PC seed and iteratively improves it through reassignment-based neighborhood repair, restoring feasibility within each selected neighborhood. Experiments across multiple benchmark families and scaling regimes show that the best-performing configuration improves 89.1% of instances over fixed-assignment seed solutions, demonstrating that large neighborhood search effectively captures the gains from flexible reassignment under precedence constraints.

**arXiv ID:** 2603.28968
</details>

<details>
<summary><strong>MA-SAPO: Multi-Agent Reasoning for Score-Aware Prompt Optimization</strong> - Wonduk Seo, Juhyeon Lee, Junseo Koh, Wonseok Choi, Hyunjin An, Jian Park, Seunghyun lee, Haihua Chen, Yi Bu - [[pdf]](https://arxiv.org/pdf/2510.16635)</summary>

**Abstract:** Prompt optimization has become a practical way to improve the performance of Large Language Models (LLMs) without retraining. However, most existing frameworks treat evaluation as a black box, relying solely on outcome scores without explaining why prompts succeed or fail. Moreover, they involve repetitive trial-and-error refinements that remain implicit, offering limited interpretability or actionable guidance for systematic improvement. In this paper, we propose MA-SAPO: a new Multi-Agent Reasoning for Score Aware Prompt Optimization framework that links evaluation outcomes directly to targeted refinements. Specifically, in the Training Phase, multiple agents interpret evaluation scores, diagnose weaknesses, and generate concrete revision directives, which are stored as reusable reasoning assets. In the Test Phase, an analyzer agent retrieves relevant exemplars and assets for a new prompt, and a refiner agent applies evidence-based edits to improve the prompt and its response. By grounding optimization in structured reasoning, MA-SAPO ensures edits are interpretable, auditable, and controllable. Experiments on the HelpSteer1/2 benchmarks show that our framework consistently outperforms single-pass prompting, retrieval-augmented generation, and prior multi-agent methods across multiple evaluation metrics.

**arXiv ID:** 2510.16635
</details>

<details>
<summary><strong>Multi-Agent Memory from a Computer Architecture Perspective: Visions and Challenges Ahead</strong> - Zhongming Yu, Naicheng Yu, Hejia Zhang, Wentao Ni, Mingrui Yin, Jiaying Yang, Yujie Zhao, Jishen Zhao - [[pdf]](https://arxiv.org/pdf/2603.10062)</summary>

**Abstract:** As LLM agents evolve into collaborative multi-agent systems, their memory requirements grow rapidly in complexity. This position paper frames multi-agent memory as a computer architecture problem. We distinguish shared and distributed memory paradigms, propose a three-layer memory hierarchy (I/O, cache, and memory), and identify two critical protocol gaps: cache sharing across agents and structured memory access control. We argue that the most pressing open challenge is multi-agent memory consistency. Our architectural framing provides a foundation for building reliable, scalable multi-agent systems.

**arXiv ID:** 2603.10062
</details>

<details>
<summary><strong>Multi-AUV Cooperative Target Tracking Based on Supervised Diffusion-Aided Multi-Agent Reinforcement Learning</strong> - Jiaao Ma, Chuan Lin, Guangjie Han, Shengchao Zhu, Zhenyu Wang, Chen An - [[pdf]](https://arxiv.org/pdf/2603.29426)</summary>

**Abstract:** In recent years, advances in underwater networking and multi-agent reinforcement learning (MARL) have significantly expanded multi-autonomous underwater vehicle (AUV) applications in marine exploration and target tracking. However, current MARL-driven cooperative tracking faces three critical challenges: 1) non-stationarity in decentralized coordination, where local policy updates destabilize teammates' observation spaces, preventing convergence; 2) sparse-reward exploration inefficiency from limited underwater visibility and constrained sensor ranges, causing high-variance learning; and 3) water disturbance fragility combined with handcrafted reward dependency that degrades real-world robustness under unmodeled hydrodynamic conditions. To address these challenges, this paper proposes a hierarchical MARL architecture comprising four layers: global training scheduling, multi-agent coordination, local decision-making, and real-time execution. This architecture optimizes task allocation and inter-AUV coordination through hierarchical decomposition. Building on this foundation, we propose the Supervised Diffusion-Aided MARL (SDA-MARL) algorithm featuring three innovations: 1) a dual-decision architecture with segregated experience pools mitigating nonstationarity through structured experience replay; 2) a supervised learning mechanism guiding the diffusion model's reverse denoising process to generate high-fidelity training samples that accelerate convergence; and 3) disturbance-robust policy learning incorporating behavioral cloning loss to guide the Deep Deterministic Policy Gradient network update using high-quality replay actions, eliminating handcrafted reward dependency. The tracking algorithm based on SDA-MARL proposed in this paper achieves superior precision compared to state-of-the-art methods in comprehensive underwater simulations.

**arXiv ID:** 2603.29426
</details>

<details>
<summary><strong>Distributed Predictive Control Barrier Functions: Towards Scalable Safety Certification in Modular Multi-Agent Systems</strong> - Jonas Ohnemus, Alexandre Didier, Ahmed Aboudonia, Andrea Carron, Melanie N. Zeilinger - [[pdf]](https://arxiv.org/pdf/2603.29560)</summary>

**Abstract:** We consider safety-critical multi-agent systems with distributed control architectures and potentially varying network topologies. While learning-based distributed control enables scalability and high performance, a lack of formal safety guarantees in the face of unforeseen disturbances and unsafe network topology changes may lead to system failure. To address this challenge, we introduce structured control barrier functions (s-CBFs) as a multi-agent safety framework. The s-CBFs are augmented to a distributed predictive control barrier function (D-PCBF), a predictive, optimization-based safety layer that uses model predictions to guarantee recoverable safety at all times. The proposed approach enables a permissive yet formal plug-and-play protocol, allowing agents to join or leave the network while ensuring safety recovery if a change in network topology requires temporarily unsafe behavior. We validate the formulation through simulations and real-time experiments of a miniature race-car platoon.

**arXiv ID:** 2603.29560
</details>

<details>
<summary><strong>Context-Triggered Contingency Games for Strategic Multi-Agent Interaction</strong> - Kilian Schweppe, Anne-Kathrin Schmuck - [[pdf]](https://arxiv.org/pdf/2512.03639)</summary>

**Abstract:** We address the challenge of reliable and efficient interaction in autonomous multi-agent systems, where agents must balance long-term strategic objectives with short-term dynamic adaptation. We propose context-triggered contingency games, a novel integration of strategic games derived from temporal logic specifications with dynamic contingency games solved in real time. Our two-layered architecture leverages strategy templates to guarantee satisfaction of high-level objectives, while a new factor-graph-based solver enables scalable, real-time model predictive control of dynamic interactions. The resulting framework ensures both safety and progress in uncertain, interactive environments. We validate our approach through simulations and hardware experiments in autonomous driving and robotic navigation, demonstrating efficient, reliable, and adaptive multi-agent interaction.

**arXiv ID:** 2512.03639
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>View-oriented Conversation Compiler for Agent Trace Analysis</strong> - Lvmin Zhang, Maneesh Agrawala - [[pdf]](https://arxiv.org/pdf/2603.29678)</summary>

**Abstract:** Agent traces carry increasing analytical value in the era of context learning and harness-driven agentic cognition, yet most prior work treats conversation format as a trivial engineering detail. Modern agent conversations contain deeply structured content, including nested tool calls and results, chain-of-thought reasoning blocks, sub-agent invocations, context-window compaction boundaries, and harness-injected system directives, whose complexity far exceeds that of simple user-assistant exchanges. Feeding such traces to a reflector or other analytical mechanism in plain text, JSON, YAML, or via grep can materially degrade analysis quality. This paper presents VCC (View-oriented Conversation Compiler), a compiler (lex, parse, IR, lower, emit) that transforms raw agent JSONL logs into a family of structured views: a full view (lossless transcript serving as the canonical line-number coordinate system), a user-interface view (reconstructing the interaction as the user actually perceived it), and an adaptive view (a structure-preserving projection governed by a relevance predicate). In a context-learning experiment on AppWorld, replacing only the reflector's input format, from raw JSONL to VCC-compiled views, leads to higher pass rates across all three model configurations tested, while cutting reflector token consumption by half to two-thirds and producing more concise learned memory. These results suggest that message format functions as infrastructure for context learning, not as an incidental implementation choice.

**arXiv ID:** 2603.29678
</details>

<details>
<summary><strong>Physiological and Semantic Patterns in Medical Teams Using an Intelligent Tutoring System</strong> - Xiaoshan Huang, Conrad Borchers, Jiayi Zhang, Susanne P. Lajoie - [[pdf]](https://arxiv.org/pdf/2603.29950)</summary>

**Abstract:** Effective collaboration requires teams to manage complex cognitive and emotional states through Socially Shared Regulation of Learning (SSRL). Physiological synchrony (i.e., longitudinal alignment in physiological signals) can indicate these states, but is hard to interpret on its own. We investigate the physiological and conversational dynamics of four medical dyads diagnosing a virtual patient case using an intelligent tutoring system. Semantic shifts in dialogue were correlated with transient physiological synchrony peaks. We also coded utterance segments for SSRL and derived cosine similarity using sentence embeddings. The results showed that activating prior knowledge featured significantly lower semantic similarity than simpler task execution. High physiological synchrony was associated with lower semantic similarity, suggesting that such moments involve exploratory and varied language use. Qualitative analysis triangulated these synchrony peaks as ``pivotal moments'': successful teams synchronized during shared discovery, while unsuccessful teams peaked during shared uncertainty. This research advances human-centered AI by demonstrating how biological signals can be fused with dialogues to understand critical moments in problem solving.

**arXiv ID:** 2603.29950
</details>

<details>
<summary><strong>The Triadic Cognitive Architecture: Bounding Autonomous Action via Spatio-Temporal and Epistemic Friction</strong> - Davide Di Gioia - [[pdf]](https://arxiv.org/pdf/2603.30031)</summary>

**Abstract:** Current autonomous AI agents, driven primarily by Large Language Models (LLMs), operate in a state of cognitive weightlessness: they process information without an intrinsic sense of network topology, temporal pacing, or epistemic limits. Consequently, heuristic agentic loops (e.g., ReAct) can exhibit failure modes in interactive environments, including excessive tool use under congestion, prolonged deliberation under time decay, and brittle behavior under ambiguous evidence. In this paper, we propose the Triadic Cognitive Architecture (TCA), a unified mathematical framework that grounds machine reasoning in continuous-time physics. By synthesizing nonlinear filtering theory, Riemannian routing geometry, and optimal control, we formally define the concept of Cognitive Friction. We map the agent's deliberation process to a coupled stochastic control problem where information acquisition is path-dependent and physically constrained. Rather than relying on arbitrary heuristic stop-tokens, the TCA uses an HJB-motivated stopping boundary and instantiates a rollout-based approximation of belief-dependent value-of-information with a net-utility halting condition. Through empirical validation in a simulated Emergency Medical Diagnostic Grid (EMDG), we demonstrate that while greedy baselines over-deliberate under latency and congestion costs, the triadic policy reduces time-to-action while improving patient viability without degrading diagnostic accuracy in this environment.

**arXiv ID:** 2603.30031
</details>

<details>
<summary><strong>Evaluating a Data-Driven Redesign Process for Intelligent Tutoring Systems</strong> - Qianru Lyu, Conrad Borchers, Meng Xia, Karen Xiao, Paulo F. Carvalho, Kenneth R. Koedinger, Vincent Aleven - [[pdf]](https://arxiv.org/pdf/2603.29094)</summary>

**Abstract:** Past research has defined a general process for the data-driven redesign of educational technologies and has shown that in carefully-selected instances, this process can help make systems more effective. In the current work, we test the generality of the approach by applying it to four units of a middle-school mathematics intelligent tutoring system that were selected not based on suitability for redesign, as in previous work, but on topic. We tested whether the redesigned system was more effective than the original in a classroom study with 123 students. Although the learning gains did not differ between the conditions, students who used the Redesigned Tutor had more productive time-on-task, a larger number of skills practiced, and greater total knowledge mastery. The findings highlight the promise of data-driven redesign even when applied to instructional units *not* selected as likely to yield improvement, as evidence of the generality and wide applicability of the method.

**arXiv ID:** 2603.29094
</details>

<details>
<summary><strong>Let the Agent Steer: Closed-Loop Ranking Optimization via Influence Exchange</strong> - Yin Cheng, Liao Zhou, Xiyu Liang, Dihao Luo, Tewei Lee, Kailun Zheng, Weiwei Zhang, Mingchen Cai, Jian Dong, Andy Zhang - [[pdf]](https://arxiv.org/pdf/2603.27765)</summary>

**Abstract:** Recommendation ranking is fundamentally an influence allocation problem: a sorting formula distributes ranking influence among competing factors, and the business outcome depends on finding the optimal "exchange rates" among them. However, offline proxy metrics systematically misjudge how influence reallocation translates to online impact, with asymmetric bias across metrics that a single calibration factor cannot correct.
We present Sortify, the first fully autonomous LLM-driven ranking optimization agent deployed in a large-scale production recommendation system. The agent reframes ranking optimization as continuous influence exchange, closing the full loop from diagnosis to parameter deployment without human intervention. It addresses structural problems through three mechanisms: (1) a dual-channel framework grounded in Savage's Subjective Expected Utility (SEU) that decouples offline-online transfer correction (Belief channel) from constraint penalty adjustment (Preference channel); (2) an LLM meta-controller operating on framework-level parameters rather than low-level search variables; (3) a persistent Memory DB with 7 relational tables for cross-round learning. Its core metric, Influence Share, provides a decomposable measure where all factor contributions sum to exactly 100%.
Sortify has been deployed across two markets. In Country A, the agent pushed GMV from -3.6% to +9.2% within 7 rounds with peak orders reaching +12.5%. In Country B, a cold-start deployment achieved +4.15% GMV/UU and +3.58% Ads Revenue in a 7-day A/B test, leading to full production rollout.

**arXiv ID:** 2603.27765
</details>

<details>
<summary><strong>Joint Cooperative and Non-Cooperative Localization in WSNs with Distributed Scaled Proximal ADMM Algorithms</strong> - Qiaojia Zhu, Xiaojing Shen, Haiqi Liu, Pramod K. Varshney - [[pdf]](https://arxiv.org/pdf/2509.18213)</summary>

**Abstract:** The integration of cooperative and non-cooperative localization is fundamentally important, as these two modes frequently coexist in wireless sensor networks, especially when sensor positions are uncertain and targets are unable to communicate with the network. This paper presents a joint modeling approach that formulates cooperative and non-cooperative localization as a single optimization problem. By processing both tasks jointly, the proposed method eliminates the latency inherent in sequential approaches that perform cooperative localization first, followed by non-cooperative localization. However, this joint formulation introduces complex variable coupling, posing challenges in both modeling and optimization. To address this coupling, we introduce auxiliary variables that enable structural decoupling and facilitate distributed computation. Building on this formulation, we develop the Scaled Proximal Alternating Direction Method of Multipliers for Joint Cooperative and Non-Cooperative Localization (SP-ADMM-JCNL). Leveraging the structured design of the problem, we provide theoretical guarantees that the algorithm generates a sequence converging globally to a KKT point of the reformulated problem, and further to a critical point of the original non-convex objective function, with the convergence rate of O(1/T). Experiments demonstrate that SP-ADMM-JCNL achieves accurate and reliable localization performance.

**arXiv ID:** 2509.18213
</details>

<details>
<summary><strong>AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation</strong> - Jin-Chuan Shi, Binhong Ye, Tao Liu, Xiaoyang Liu, Yangjinhui Xu, Junzhe He, Zeju Li, Hao Chen, Chunhua Shen - [[pdf]](https://arxiv.org/pdf/2602.04672)</summary>

**Abstract:** Reconstructing dynamic hand-object interactions from monocular videos is critical for dexterous manipulation data collection and creating realistic digital twins for robotics and VR. However, current methods face two prohibitive barriers: (1) reliance on neural rendering often yields fragmented, non-simulation-ready geometries under heavy occlusion, and (2) dependence on brittle Structure-from-Motion (SfM) initialization leads to frequent failures on in-the-wild footage. To overcome these limitations, we introduce AGILE, a robust framework that shifts the paradigm from reconstruction to agentic generation for interaction learning. First, we employ an agentic pipeline where a Vision-Language Model (VLM) guides a generative model to synthesize a complete, watertight object mesh with high-fidelity texture, independent of video occlusions. Second, bypassing fragile SfM entirely, we propose a robust anchor-and-track strategy. We initialize the object pose at a single interaction onset frame using a foundation model and propagate it temporally by leveraging the strong visual similarity between our generated asset and video observations. Finally, a contact-aware optimization integrates semantic, geometric, and interaction stability constraints to enforce physical plausibility. Extensive experiments on HO3D, DexYCB, and in-the-wild videos reveal that AGILE outperforms baselines in global geometric accuracy while demonstrating exceptional robustness on challenging sequences where prior art frequently collapses. By prioritizing physical validity, our method produces simulation-ready assets validated via real-to-sim retargeting for robotic applications.

**arXiv ID:** 2602.04672
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (21 papers)</h2></summary>

<details>
<summary><strong>SciVisAgentBench: A Benchmark for Evaluating Scientific Data Analysis and Visualization Agents</strong> - Kuangshi Ai, Haichao Miao, Kaiyuan Tang, Nathaniel Gorski, Jianxin Sun, Guoxi Liu, Helgi I. Ingolfsson, David Lenz, Hanqi Guo, Hongfeng Yu, Teja Leburu, Michael Molash, Bei Wang, Tom Peterka, Chaoli Wang, Shusen Liu - [[pdf]](https://arxiv.org/pdf/2603.29139)</summary>

**Abstract:** Recent advances in large language models (LLMs) have enabled agentic systems that translate natural language intent into executable scientific visualization (SciVis) tasks. Despite rapid progress, the community lacks a principled and reproducible benchmark for evaluating these emerging SciVis agents in realistic, multi-step analysis settings. We present SciVisAgentBench, a comprehensive and extensible benchmark for evaluating scientific data analysis and visualization agents. Our benchmark is grounded in a structured taxonomy spanning four dimensions: application domain, data type, complexity level, and visualization operation. It currently comprises 108 expert-crafted cases covering diverse SciVis scenarios. To enable reliable assessment, we introduce a multimodal outcome-centric evaluation pipeline that combines LLM-based judging with deterministic evaluators, including image-based metrics, code checkers, rule-based verifiers, and case-specific evaluators. We also conduct a validity study with 12 SciVis experts to examine the agreement between human and LLM judges. Using this framework, we evaluate representative SciVis agents and general-purpose coding agents to establish initial baselines and reveal capability gaps. SciVisAgentBench is designed as a living benchmark to support systematic comparison, diagnose failure modes, and drive progress in agentic SciVis. The benchmark is available at this https URL.

**arXiv ID:** 2603.29139
</details>

<details>
<summary><strong>Symphony for Medical Coding: A Next-Generation Agentic System for Scalable and Explainable Medical Coding</strong> - Joakim Edin, Andreas Motzfeldt, Simon Flachs, Lars Maaløe - [[pdf]](https://arxiv.org/pdf/2603.29709)</summary>

**Abstract:** Medical coding translates free-text clinical documentation into standardized codes drawn from classification systems that contain tens of thousands of entries and are updated annually. It is central to billing, clinical research, and quality reporting, yet remains largely manual, slow, and error-prone. Existing automated approaches learn to predict a fixed set of codes from labeled data, thereby preventing adaptation to new codes or different coding systems without retraining on different data. They also provide no explanation for their predictions, limiting trust in safety-critical settings. We introduce Symphony for Medical Coding, a system that approaches the task the way expert human coders do: by reasoning over the clinical narrative with direct access to the coding guidelines. This design allows Symphony to operate across any coding system and to provide span-level evidence linking each predicted code to the text that supports it. We evaluate on two public benchmarks and three real-world datasets spanning inpatient, outpatient, emergency, and subspecialty settings across the United States and the United Kingdom. Symphony achieves state-of-the-art results across all settings, establishing itself as a flexible, deployment-ready foundation for automated clinical coding.

**arXiv ID:** 2603.29709
</details>

<details>
<summary><strong>Owl-AuraID 1.0: An Intelligent System for Autonomous Scientific Instrumentation and Scientific Data Analysis</strong> - Han Deng, Anqi Zou, Hanling Zhang, Ben Fei, Chengyu Zhang, Haobo Wang, Xinru Guo, Zhenyu Li, Xuzhu Wang, Peng Yang, Fujian Zhang, Weiyu Guo, Xiaohong Shao, Zhaoyang Liu, Shixiang Tang, Zhihui Wang, Wanli Ouyang - [[pdf]](https://arxiv.org/pdf/2603.29828)</summary>

**Abstract:** Scientific discovery increasingly depends on high-throughput characterization, yet automation is hindered by proprietary GUIs and the limited generalizability of existing API-based systems. We present Owl-AuraID, a software-hardware collaborative embodied agent system that adopts a GUI-native paradigm to operate instruments through the same interfaces as human experts. Its skill-centric framework integrates Type-1 (GUI operation) and Type-2 (data analysis) skills into end-to-end workflows, connecting physical sample handling with scientific interpretation. Owl-AuraID demonstrates broad coverage across ten categories of precision instruments and diverse workflows, including multimodal spectral analysis, microscopic imaging, and crystallographic analysis, supporting modalities such as FTIR, NMR, AFM, and TGA. Overall, Owl-AuraID provides a practical, extensible foundation for autonomous laboratories and illustrates a path toward evolving laboratory intelligence through reusable operational and analytical skills. The code are available at this https URL.

**arXiv ID:** 2603.29828
</details>

<details>
<summary><strong>AI in Work-Based Learning: Understanding the Purposes and Effects of Intelligent Tools Among Student Interns</strong> - John Paul P. Miranda, Rhiziel P. Manalese, Sheila M. Geronimo, Vernon Grace M. Maniago, Charlie K. Padilla, Aileen P. De Leon, Santa L. Merle, Mark Anthony A. Castro - [[pdf]](https://arxiv.org/pdf/2603.28786)</summary>

**Abstract:** This study examined how student interns in Philippine higher education use intelligent tools during their OJT. Data were collected from 384 respondents using a structured questionnaire that asked about AI tool usage, task-specific applications, and perceptions of confidence, ethics, and support. Analysis of task-based usage identified four main purposes: productivity and report writing, communication and content drafting, technical assistance and code support, and independent task completion. ChatGPT was the most commonly used AI tool, followed by Quillbot, Canva AI, and Grammarly. Students reported moderate confidence in using AI and applied these tools selectively and ethically during OJT tasks. This indicate that AI tools assist student interns in various OJT activities related to work-readiness. The study suggests that higher education programs include AI literacy and onboarding. Clear policies and fair access to AI tools are important to support responsible use and prepare students for future careers.

**arXiv ID:** 2603.28786
</details>

<details>
<summary><strong>MemFactory: Unified Inference & Training Framework for Agent Memory</strong> - Ziliang Guo, Ziheng Li, Zhiyu Li - [[pdf]](https://arxiv.org/pdf/2603.29493)</summary>

**Abstract:** Memory-augmented Large Language Models (LLMs) are essential for developing capable, long-term AI agents. Recently, applying Reinforcement Learning (RL) to optimize memory operations, such as extraction, updating, and retrieval, has emerged as a highly promising research direction. However, existing implementations remain highly fragmented and task-specific, lacking a unified infrastructure to streamline the integration, training, and evaluation of these complex pipelines. To address this gap, we present MemFactory, the first unified, highly modular training and inference framework specifically designed for memory-augmented agents. Inspired by the success of unified fine-tuning frameworks like LLaMA-Factory, MemFactory abstracts the memory lifecycle into atomic, plug-and-play components, enabling researchers to seamlessly construct custom memory agents via a "Lego-like" architecture. Furthermore, the framework natively integrates Group Relative Policy Optimization (GRPO) to fine-tune internal memory management policies driven by multi-dimensional environmental rewards. MemFactory provides out-of-the-box support for recent cutting-edge paradigms, including Memory-R1, RMM, and MemAgent. We empirically validate MemFactory on the open-source MemAgent architecture using its publicly available training and evaluation data. Across both in-domain and out-of-distribution evaluation sets, MemFactory consistently improves performance over the corresponding base models, with relative gains of up to 14.8%. By providing a standardized, extensible, and easy-to-use infrastructure, MemFactory significantly lowers the barrier to entry, paving the way for future innovations in memory-driven AI agents.

**arXiv ID:** 2603.29493
</details>

<details>
<summary><strong>Target-Aligned Reinforcement Learning</strong> - Leonard S. Pleiss, James Harrison, Maximilian Schiffer - [[pdf]](https://arxiv.org/pdf/2603.29501)</summary>

**Abstract:** Many reinforcement learning algorithms rely on target networks - lagged copies of the online network - to stabilize training. While effective, this mechanism introduces a fundamental stability-recency tradeoff: slower target updates improve stability but reduce the recency of learning signals, hindering convergence speed. We propose Target-Aligned Reinforcement Learning (TARL), a framework that emphasizes transitions for which the target and online network estimates are highly aligned. By focusing updates on well-aligned targets, TARL mitigates the adverse effects of stale target estimates while retaining the stabilizing benefits of target networks. We provide a theoretical analysis demonstrating that target alignment correction accelerates convergence, and empirically demonstrate consistent improvements over standard reinforcement learning algorithms across various benchmark environments.

**arXiv ID:** 2603.29501
</details>

<details>
<summary><strong>6GAgentGym: Tool Use, Data Synthesis, and Agentic Learning for Network Management</strong> - Jiao Chen, Jianhua Tang, Xiaotong Yang, Zuohong Lv - [[pdf]](https://arxiv.org/pdf/2603.29656)</summary>

**Abstract:** Autonomous 6G network management requires agents that can execute tools, observe the resulting state changes, and adapt their decisions accordingly. Existing benchmarks based on static questions or scripted episode replay, however, do not support such closed-loop interaction, limiting agents to passive evaluation without the ability to learn from environmental feedback. This paper presents 6GAgentGym to provide closed-loop capability. The framework provides an interactive environment with 42 typed tools whose effect classification distinguishes read-only observation from state-mutating configuration, backed by a learned Experiment Model calibrated on NS-3 simulation data. 6G-Forge bootstraps closed-loop training trajectories from NS-3 seeds via iterative Self-Instruct generation with execution verification against the Experiment Model. Supervised fine-tuning on the resulting corpus followed by reinforcement learning with online closed-loop interaction enables an 8B open-source model to achieve comparable overall success rate to GPT-5 on the accompanying 6GAgentBench, with stronger performance on long-horizon tasks. Together, these components provide a viable path toward autonomous, closed-loop network management.

**arXiv ID:** 2603.29656
</details>

<details>
<summary><strong>Hybrid Framework for Robotic Manipulation: Integrating Reinforcement Learning and Large Language Models</strong> - Md Saad, Sajjad Hussain, Mohd Suhaib - [[pdf]](https://arxiv.org/pdf/2603.30022)</summary>

**Abstract:** This paper introduces a new hybrid framework that combines Reinforcement Learning (RL) and Large Language Models (LLMs) to improve robotic manipulation tasks. By utilizing RL for accurate low-level control and LLMs for high level task planning and understanding of natural language, the proposed framework effectively connects low-level execution with high-level reasoning in robotic systems. This integration allows robots to understand and carry out complex, human-like instructions while adapting to changing environments in real time. The framework is tested in a PyBullet-based simulation environment using the Franka Emika Panda robotic arm, with various manipulation scenarios as benchmarks. The results show a 33.5% decrease in task completion time and enhancements of 18.1% and 36.4% in accuracy and adaptability, respectively, when compared to systems that use only RL. These results underscore the potential of LLM-enhanced robotic systems for practical applications, making them more efficient, adaptable, and capable of interacting with humans. Future research will aim to explore sim-to-real transfer, scalability, and multi-robot systems to further broaden the framework's applicability.

**arXiv ID:** 2603.30022
</details>

<details>
<summary><strong>Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills</strong> - Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng, Dexin Wang, Erchao Zhao, Xiaoxi Jiang, Guanjun Jiang - [[pdf]](https://arxiv.org/pdf/2603.25158)</summary>

**Abstract:** Equipping Large Language Model (LLM) agents with domain-specific skills is critical for tackling complex tasks. Yet, manual authoring creates a severe scalability bottleneck. Conversely, automated skill generation often yields fragile or fragmented results because it either relies on shallow parametric knowledge or sequentially overfits to non-generalizable trajectory-local lessons. To overcome this, we introduce Trace2Skill, a framework that mirrors how human experts author skills: by holistically analyzing broad execution experience before distilling it into a single, comprehensive guide. Instead of reacting sequentially to individual trajectories, Trace2Skill dispatches a parallel fleet of sub-agents to analyze a diverse pool of executions. It extracts trajectory-specific lessons and hierarchically consolidates them into a unified, conflict-free skill directory via inductive reasoning. Trace2Skill supports both deepening existing human-written skills and creating new ones from scratch. Experiments in challenging domains, such as spreadsheet, VisionQA and math reasoning, show that Trace2Skill significantly improves upon strong baselines, including Anthropic's official xlsx skills. Crucially, this trajectory-grounded evolution does not merely memorize task instances or model-specific quirks: evolved skills transfer across LLM scales and generalize to OOD settings. For example, skills evolved by Qwen3.5-35B on its own trajectories improved a Qwen3.5-122B agent by up to 57.65 absolute percentage points on WikiTableQuestions. Ultimately, our results demonstrate that complex agent experience can be packaged into highly transferable, declarative skills -- requiring no parameter updates, no external retrieval modules, and utilizing open-source models as small as 35B parameters.

**arXiv ID:** 2603.25158
</details>

<details>
<summary><strong>Sample-Efficient Hypergradient Estimation for Decentralized Bi-Level Reinforcement Learning</strong> - Mikoto Kudo, Takumi Tanabe, Akifumi Wachi, Youhei Akimoto - [[pdf]](https://arxiv.org/pdf/2603.14867)</summary>

**Abstract:** Many strategic decision-making problems, such as environment design for warehouse robots, can be naturally formulated as bi-level reinforcement learning (RL), where a leader agent optimizes its objective while a follower solves a Markov decision process (MDP) conditioned on the leader's decisions. In many situations, a fundamental challenge arises when the leader cannot intervene in the follower's optimization process; it can only observe the optimization outcome. We address this decentralized setting by deriving the hypergradient of the leader's objective, i.e., the gradient of the leader's strategy that accounts for changes in the follower's optimal policy. Unlike prior hypergradient-based methods that require extensive data for repeated state visits or rely on gradient estimators whose complexity can increase substantially with the high-dimensional leader's decision space, we leverage the Boltzmann covariance trick to derive an alternative hypergradient formulation. This enables efficient hypergradient estimation solely from interaction samples, even when the leader's decision space is high-dimensional. Additionally, to our knowledge, this is the first method that enables hypergradient-based optimization for 2-player Markov games in decentralized settings. Experiments highlight the impact of hypergradient updates and demonstrate our method's effectiveness in both discrete and continuous state tasks.

**arXiv ID:** 2603.14867
</details>

<details>
<summary><strong>SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios</strong> - Junkai Chen, Huihui Huang, Yunbo Lyu, Junwen An, Jieke Shi, Chengran Yang, Ting Zhang, Haoye Tian, Yikun Li, Zhenhao Li, Xin Zhou, Xing Hu, David Lo - [[pdf]](https://arxiv.org/pdf/2509.22097)</summary>

**Abstract:** Large language model-powered code agents are rapidly transforming software engineering, yet the security risks of their generated code have become a critical concern. Existing benchmarks have provided valuable insights, but they fail to capture scenarios in which vulnerabilities are actually introduced by human developers, making fair comparisons between humans and agents infeasible. We therefore introduce SecureVibeBench, a benchmark of 105 C/C++ secure coding tasks sourced from 41 projects in OSS-Fuzz for code agents. SecureVibeBench has the following features: (i) realistic task settings that require multi-file edits in large repositories, (ii)~aligned contexts based on real-world open-source vulnerabilities with precisely identified vulnerability introduction points, and (iii) comprehensive evaluation that combines functionality testing and security checking with both static and dynamic oracles. We evaluate 5 popular code agents like OpenHands, supported by 5 LLMs (e.g., Claude sonnet 4.5) on SecureVibeBench. Results show that current agents struggle to produce both correct and secure code, as even the best-performing one, produces merely 23.8\% correct and secure solutions on SecureVibeBench.

**arXiv ID:** 2509.22097
</details>

<details>
<summary><strong>Realistic Market Impact Modeling for Reinforcement Learning Trading Environments</strong> - Lucas Riera Abbade, Anna Helena Reali Costa - [[pdf]](https://arxiv.org/pdf/2603.29086)</summary>

**Abstract:** Reinforcement learning (RL) has shown promise for trading, yet most open-source backtesting environments assume negligible or fixed transaction costs, causing agents to learn trading behaviors that fail under realistic execution. We introduce three Gymnasium-compatible trading environments -- MACE (Market-Adjusted Cost Execution) stock trading, margin trading, and portfolio optimization -- that integrate nonlinear market impact models grounded in the Almgren-Chriss framework and the empirically validated square-root impact law. Each environment provides pluggable cost models, permanent impact tracking with exponential decay, and comprehensive trade-level logging. We evaluate five DRL algorithms (A2C, PPO, DDPG, SAC, TD3) on the NASDAQ-100, comparing a fixed 10 bps baseline against the AC model with Optuna-tuned hyperparameters. Our results show that (i) the cost model materially changes both absolute performance and the relative ranking of algorithms across all three environments; (ii) the AC model produces dramatically different trading behavior, e.g., daily costs dropping from $200k to $8k with turnover falling from 19% to 1%; (iii) hyperparameter optimization is essential for constraining pathological trading, with costs dropping up to 82%; and (iv) algorithm-cost model interactions are strongly environment-specific, e.g., DDPG's OOS Sharpe jumps from -2.1 to 0.3 under AC in margin trading while SAC's drops from -0.5 to -1.2. We release the full suite as an open-source extension to FinRL-Meta.

**arXiv ID:** 2603.29086
</details>

<details>
<summary><strong>A Pontryagin Method of Model-based Reinforcement Learning via Hamiltonian Actor-Critic</strong> - Chengyang Gu, Yuxin Pan, Hui Xiong, Yize Chen - [[pdf]](https://arxiv.org/pdf/2603.28971)</summary>

**Abstract:** Model-based reinforcement learning (MBRL) improves sample efficiency by leveraging learned dynamics models for policy optimization. However, the effectiveness of methods such as actor-critic is often limited by compounding model errors, which degrade long-horizon value estimation. Existing approaches, such as Model-Based Value Expansion (MVE), partially mitigate this issue through multi-step rollouts, but remain sensitive to rollout horizon selection and residual model bias. Motivated by the Pontryagin Maximum Principle (PMP), we propose Hamiltonian Actor-Critic (HAC), a model-based approach that eliminates explicit value function learning by directly optimizing a Hamiltonian defined over the learned dynamics and reward for deterministic systems. By avoiding value approximation, HAC reduces sensitivity to model errors while admitting convergence guarantees. Extensive experiments on continuous control benchmarks, in both online and offline RL settings, demonstrate that HAC outperforms model-free and MVE-based baselines in control performance, convergence speed, and robustness to distributional shift, including out-of-distribution (OOD) scenarios. In offline settings with limited data, HAC matches or exceeds state-of-the-art methods, highlighting its strong sample efficiency.

**arXiv ID:** 2603.28971
</details>

<details>
<summary><strong>AP-DRL: A Synergistic Algorithm-Hardware Framework for Automatic Task Partitioning of Deep Reinforcement Learning on Versal ACAP</strong> - Enlai Li, Zhe Lin, Sharad Sinha, Wei Zhang - [[pdf]](https://arxiv.org/pdf/2603.29369)</summary>

**Abstract:** Deep reinforcement learning has demonstrated remarkable success across various domains. However, the tight coupling between training and inference processes makes accelerating DRL training an essential challenge for DRL optimization. Two key issues hinder efficient DRL training: (1) the significant variation in computational intensity across different DRL algorithms and even among operations within the same algorithm complicates hardware platform selection, while (2) DRL's wide dynamic range could lead to substantial reward errors with conventional FP16+FP32 mixed-precision quantization. While existing work has primarily focused on accelerating DRL for specific computing units or optimizing inference-stage quantization, we propose AP-DRL to address the above challenges.
AP-DRL is an automatic task partitioning framework that harnesses the heterogeneous architecture of AMD Versal ACAP (integrating CPUs, FPGAs, and AI Engines) to accelerate DRL training through intelligent hardware-aware optimization. Our approach begins with bottleneck analysis of CPU, FPGA, and AIE performance across diverse DRL workloads, informing the design principles for AP-DRL's inter-component task partitioning and quantization optimization. The framework then addresses the challenge of platform selection through design space exploration-based profiling and ILP-based partitioning models that match operations to optimal computing units based on their computational characteristics. For the quantization challenge, AP-DRL employs a hardware-aware algorithm coordinating FP32 (CPU), FP16 (FPGA/DSP), and BF16 (AI Engine) operations by leveraging Versal ACAP's native support for these precision formats. Comprehensive experiments indicate that AP-DRL can achieve speedup of up to 4.17$\times$ over programmable logic and up to 3.82$\times$ over AI Engine baselines while maintaining training convergence.

**arXiv ID:** 2603.29369
</details>

<details>
<summary><strong>A General Control-Theoretic Approach for Reinforcement Learning: Theory and Algorithms</strong> - Weiqin Chen, Mark S. Squillante, Chai Wah Wu, Santiago Paternain - [[pdf]](https://arxiv.org/pdf/2406.14753)</summary>

**Abstract:** We devise a control-theoretic reinforcement learning approach to support direct learning of the optimal policy. We establish various theoretical properties of our approach, such as convergence and optimality of our analog of the Bellman operator and Q-learning, a new control-policy-variable gradient theorem, and a specific gradient ascent algorithm based on this theorem within the context of a specific control-theoretic framework. We empirically evaluate the performance of our control theoretic approach on several classical reinforcement learning tasks, demonstrating significant improvements in solution quality, sample complexity, and running time of our approach over state-of-the-art methods.

**arXiv ID:** 2406.14753
</details>

<details>
<summary><strong>ECHO-2: A Large-Scale Distributed Rollout Framework for Cost-Efficient Reinforcement Learning</strong> - Jie Xiao, Meng Chen, Qingnan Ren, Jingwei Song, Jiaqi Huang, Yangshen Deng, Chris Tong, Wanyi Chen, Suli Wang, Ziqian Bi, Shuo Lu, Yiqun Duan, Xu Wang, Rymon Yu, Ween Yang, Lynn Ai, Eric Yang, Bill Shi - [[pdf]](https://arxiv.org/pdf/2602.02192)</summary>

**Abstract:** Reinforcement learning (RL) is a critical stage in post-training large language models (LLMs), involving repeated interaction between rollout generation, reward evaluation, and centralized learning. Distributing rollout execution offers opportunities to leverage more cost-efficient inference resources, but introduces challenges in wide-area coordination and policy dissemination. We present ECHO-2, a distributed RL framework for post-training with remote inference workers and non-negligible dissemination latency. ECHO-2 combines centralized learning with distributed rollouts and treats bounded policy staleness as a user-controlled parameter, enabling rollout generation, dissemination, and training to overlap. We introduce an overlap-based capacity model that relates training time, dissemination latency, and rollout throughput, yielding a practical provisioning rule for sustaining learner utilization. To mitigate dissemination bottlenecks and lower cost, ECHO-2 employs peer-assisted pipelined broadcast and cost-aware activation of heterogeneous workers. Experiments on GRPO post-training of 4B and 8B models under real wide-area bandwidth regimes show that ECHO-2 significantly improves cost efficiency while preserving RL reward comparable to strong baselines.

**arXiv ID:** 2602.02192
</details>

<details>
<summary><strong>Beyond Hard Constraints: Budget-Conditioned Reachability For Safe Offline Reinforcement Learning</strong> - Janaka Chathuranga Brahmanage, Akshat Kumar - [[pdf]](https://arxiv.org/pdf/2603.22292)</summary>

**Abstract:** Sequential decision making using Markov Decision Process underpins many realworld applications. Both model-based and model free methods have achieved strong results in these settings. However, real-world tasks must balance reward maximization with safety constraints, often conflicting objectives, that can lead to unstable min/max, adversarial optimization. A promising alternative is safety reachability analysis, which precomputes a forward-invariant safe state, action set, ensuring that an agent starting inside this set remains safe indefinitely. Yet, most reachability based methods address only hard safety constraints, and little work extends reachability to cumulative cost constraints. To address this, first, we define a safetyconditioned reachability set that decouples reward maximization from cumulative safety cost constraints. Second, we show how this set enforces safety constraints without unstable min/max or Lagrangian optimization, yielding a novel offline safe RL algorithm that learns a safe policy from a fixed dataset without environment interaction. Finally, experiments on standard offline safe RL benchmarks, and a real world maritime navigation task demonstrate that our method matches or outperforms state of the art baselines while maintaining safety.

**arXiv ID:** 2603.22292
</details>

<details>
<summary><strong>Principal Prototype Analysis on Manifold for Interpretable Reinforcement Learning</strong> - Bodla Krishna Vamshi, Haizhao Yang - [[pdf]](https://arxiv.org/pdf/2603.27971)</summary>

**Abstract:** Recent years have witnessed the widespread adoption of reinforcement learning (RL), from solving real-time games to fine-tuning large language models using human preference data significantly improving alignment with user expectations. However, as model complexity grows exponentially, the interpretability of these systems becomes increasingly challenging. While numerous explainability methods have been developed for computer vision and natural language processing to elucidate both local and global reasoning patterns, their application to RL remains limited. Direct extensions of these methods often struggle to maintain the delicate balance between interpretability and performance within RL settings. Prototype-Wrapper Networks (PW-Nets) have recently shown promise in bridging this gap by enhancing explainability in RL domains without sacrificing the efficiency of the original black-box models. However, these methods typically require manually defined reference prototypes, which often necessitate expert domain knowledge. In this work, we propose a method that removes this dependency by automatically selecting optimal prototypes from the available data. Preliminary experiments on standard Gym environments demonstrate that our approach matches the performance of existing PW-Nets, while remaining competitive with the original black-box models.

**arXiv ID:** 2603.27971
</details>

<details>
<summary><strong>GraSP-STL: A Graph-Based Framework for Zero-Shot Signal Temporal Logic Planning via Offline Goal-Conditioned Reinforcement Learning</strong> - Ancheng Hou, Ruijia Liu, Xiang Yin - [[pdf]](https://arxiv.org/pdf/2603.29533)</summary>

**Abstract:** This paper studies offline, zero-shot planning under Signal Temporal Logic (STL) specifications. We assume access only to an offline dataset of state-action-state transitions collected by a task-agnostic behavior policy, with no analytical dynamics model, no further environment interaction, and no task-specific retraining. The objective is to synthesize a control strategy whose resulting trajectory satisfies an arbitrary unseen STL specification. To this end, we propose GraSP-STL, a graph-search-based framework for zero-shot STL planning from offline trajectories. The method learns a goal-conditioned value function from offline data and uses it to induce a finite-horizon reachability metric over the state space. Based on this metric, it constructs a directed graph abstraction whose nodes represent representative states and whose edges encode feasible short-horizon transitions. Planning is then formulated as a graph search over waypoint sequences, evaluated using arithmetic-geometric mean robustness and its interval semantics, and executed by a learned goal-conditioned policy. The proposed framework separates reusable reachability learning from task-conditioned planning, enabling zero-shot generalization to unseen STL tasks and long-horizon planning through the composition of short-horizon behaviors from offline data. Experimental results demonstrate its effectiveness on a range of offline STL planning tasks.

**arXiv ID:** 2603.29533
</details>

<details>
<summary><strong>TRANS: Terrain-aware Reinforcement Learning for Agile Navigation of Quadruped Robots under Social Interactions</strong> - Wei Zhu, Irfan Tito Kurniawan, Ye Zhao, Mitsuhiro Hayashibe - [[pdf]](https://arxiv.org/pdf/2602.12724)</summary>

**Abstract:** This study introduces TRANS: Terrain-aware Reinforcement learning for Agile Navigation under Social interactions, a deep reinforcement learning (DRL) framework for quadrupedal social navigation over unstructured terrains. Conventional quadrupedal navigation typically separates motion planning from locomotion control, neglecting whole-body constraints and terrain awareness. On the other hand, end-to-end methods are more integrated but require high-frequency sensing, which is often noisy and computationally costly. In addition, most existing approaches assume static environments, limiting their use in human-populated settings. To address these limitations, we propose a two-stage training framework with three DRL pipelines. (1) TRANS-Loco employs an asymmetric actor-critic (AC) model for quadrupedal locomotion, enabling traversal of uneven terrains without explicit terrain or contact observations. (2) TRANS-Nav applies a symmetric AC framework for social navigation, directly mapping transformed LiDAR data to ego-agent actions under differential-drive kinematics. (3) A unified pipeline, TRANS, integrates TRANS-Loco and TRANS-Nav, supporting terrain-aware quadrupedal navigation in uneven and socially interactive environments. Comprehensive benchmarks against locomotion and social navigation baselines demonstrate the effectiveness of TRANS. Hardware experiments further confirm its potential for sim-to-real transfer.

**arXiv ID:** 2602.12724
</details>

<details>
<summary><strong>RAD-LAD: Rule and Language Grounded Autonomous Driving in Real-Time</strong> - Anurag Ghosh, Srinivasa Narasimhan, Manmohan Chandraker, Francesco Pittaluga - [[pdf]](https://arxiv.org/pdf/2603.28522)</summary>

**Abstract:** We present LAD, a real-time language--action planner with an interruptible architecture that produces a motion plan in a single forward pass (~20 Hz) or generates textual reasoning alongside a motion plan (~10 Hz). LAD is fast enough for real-time closed-loop deployment, achieving ~3x lower latency than prior driving language models while setting a new learning-based state of the art on nuPlan Test14-Hard and InterPlan. We also introduce RAD, a rule-based planner designed to address structural limitations of PDM-Closed. RAD achieves state-of-the-art performance among rule-based planners on nuPlan Test14-Hard and InterPlan. Finally, we show that combining RAD and LAD enables hybrid planning that captures the strengths of both approaches. This hybrid system demonstrates that rules and learning provide complementary capabilities: rules support reliable maneuvering, while language enables adaptive and explainable decision-making.

**arXiv ID:** 2603.28522
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
