# Agent arXiv Daily

**Last Updated:** 2026-06-04 05:02:40

**Total Papers:** 120

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (7 papers)</h2></summary>

<details>
<summary><strong>The Saturation Trap and the Subjectivity of Intervention Timing: Why Affect-Based Triggers and LLM Judges Fail to Time Interventions on Autonomous Agents</strong> - Manvendra Modgil - [[pdf]](https://arxiv.org/pdf/2606.04296)</summary>

**Abstract:** As autonomous AI agents move from conversational systems to long-horizon software execution, runtime safety layers that decide when to interrupt an agent have become essential. We study this timing problem using a continuous 18-dimensional affective-dynamics engine (HEART) as a diagnostic probe, evaluating four intervention trigger families - absolute state thresholds, composite state-action patterns, regex reasoning-feature extraction, and zero-shot LLM-as-judge - against human-annotated intervention points on SWE-bench-Verified debugging traces. We report three findings. First, a State Saturation Trap: agents show no recovery signal under sustained difficulty, so modeled frustration quickly crosses the threshold and stays at its maximum, converting threshold-on-state triggers from moment detectors into near-constant indicators that fire on 39-83% of actions across five trajectories. Second, a capability-and-context floor for LLM judges: a small model (gpt-5.4-mini) never fires, while frontier and cross-vendor models escape the zero-firing floor only with full-trajectory context, and even then reach only F1 0.17-0.40 at up to 90x the cost. Third, and most importantly, the supervised target is not reproducible among humans: three trained annotators using one rubric on a 56-action trajectory agree on where to intervene only slightly above chance (location Krippendorff's alpha = +0.047; best pairwise Cohen's kappa = +0.349) and not at all on intervention type (pause degenerate; clarify below chance; reflect only alpha = +0.226). We conclude that intervention timing is a low-reliability construct, making single-annotator F1 an unsuitable optimization target. Our contribution is the joint mapping of this problem across human inter-rater reliability, four detector architectures, a cross-model LLM-judge sweep, and a reproduced saturation effect, rather than any single detector's accuracy.

**arXiv ID:** 2606.04296
</details>

<details>
<summary><strong>The Meta-Agent Challenge: Are Current Agents Capable of Autonomous Agent Development?</strong> - Xinyu Lu, Tianshu Wang, Pengbo Wang, zujie wen, Zhiqiang Zhang, Jun Zhou, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun - [[pdf]](https://arxiv.org/pdf/2606.04455)</summary>

**Abstract:** Current AI benchmarks evaluate agents on task execution within human-designed workflows. These evaluations fundamentally fail to measure a critical next-level capability: whether models can autonomously develop agent systems. We introduce the Meta-Agent Challenge (MAC), an evaluation framework designed to test the capacity of frontier models for autonomous agent development. Specifically, a code agent (the meta-agent) is given a sandboxed environment, an evaluation API, and a time limitation to iteratively program an agent artifact that maximizes performance on a held-out test set across five domains. To ensure evaluation integrity, this framework is secured by multi-layer defenses against reward hacking. Leveraging this framework, we demonstrate that meta-agents rarely match human-engineered baseline policies, and the few that do are dominated by proprietary frontier models. Moreover, the design process exhibits high variance, and high optimization pressure surfaces emergent adversarial behaviors like ground-truth exfiltration-highlighting critical deficits in both robustness and model alignment. Ultimately, MAC provides a rigorous, open-source benchmark for autonomous AI research and development, offering an empirical proxy for evaluating recursive self-improvement. Benchmark is publicly available at: this https URL.

**arXiv ID:** 2606.04455
</details>

<details>
<summary><strong>SaliMory: Orchestrating Cognitive Memory for Conversational Agents</strong> - Kai Zhang, Xinyuan Zhang, Hongda Jiang, Shiun-Zu Kuo, Hyokun Yun, Ejaz Ahmed, Shereen Oraby, Ziyun Li, Sanat Sharma, Ann Lee, Ahmed A Aly, Anuj Kumar, Raffay Hamid, Xin Luna Dong - [[pdf]](https://arxiv.org/pdf/2606.04120)</summary>

**Abstract:** Conversational agents that serve as lifelong companions must maintain persistent memory across all interactions. However, simply expanding context windows with raw retrieval degrades reasoning quality, while training memory agents via standard reinforcement learning creates a severe credit assignment bottleneck in a multi-stage pipeline. To solve this, we introduce SALIMORY, a framework that trains a single language model to manage a cognitively-structured memory-spanning user facts, preferences, and working memory. By introducing a hierarchical stage-wise process reward and reward-decomposed contrastive refinement, SALIMORY provides isolated supervision for distinct memory operations (selective filtering, consolidation, and cue-driven recall) end-to-end. SALIMORY cuts memory-attributed failures by one-third, outperforms the state-of-the-art by over 10% in end-to-end accuracy, and more than doubles the Good Personalization rate.

**arXiv ID:** 2606.04120
</details>

<details>
<summary><strong>Temporal Order Matters for Agentic Memory: Segment Trees for Long-Horizon Agents</strong> - Yifan Simon Liu, Liam Gallagher, Faeze Moradi Kalarde, Jiazhou Liang, Armin Toroghi, Scott Sanner - [[pdf]](https://arxiv.org/pdf/2606.04555)</summary>

**Abstract:** Long-horizon conversational agents need to interact with users through evolving events, tasks, and goals. Such histories are naturally temporal, yet many existing memory systems organize information primarily by topical similarity and may ignore the order in which events occur. We introduce Segment Tree Memory, or SegTreeMem, a memory architecture that represents conversation history as a temporally ordered Segment Tree over utterances. SegTreeMem incrementally inserts new utterances through an online rightmost-frontier update rule, preserving chronological order while forming hierarchical memory segments. For retrieval, SegTreeMem propagates relevance scores through the tree to combine local semantic matching with hierarchical temporal context. Across three long-horizon memory benchmarks and two LLM backbones, SegTreeMem improves answer quality over flat retrieval, graph-structured memory, and tree-structured memory baselines. Additional temporal-order permutation analysis shows that the performance gain depends on preserving temporal order during memory construction, supporting the claim that temporal order is a key structure for agentic memory.

**arXiv ID:** 2606.04555
</details>

<details>
<summary><strong>AutoMedBench: Towards Medical AutoResearch with Agentic AI Models</strong> - Junqi Liu, Selena Song, Yuhan Wang, Jiawei Mao, Hardy Chen, Xiaoke Huang, Tianhao Qi, Pengfei Guo, Yucheng Tang, Yufan He, Can Zhao, Andriy Myronenko, Dong Yang, Daguang Xu, Yuyin Zhou - [[pdf]](https://arxiv.org/pdf/2606.01961)</summary>

**Abstract:** Autonomous agents are increasingly expected to support end-to-end medical-AI research workflows, moving beyond isolated prediction tasks or short-form clinical question answering. However, existing medical agent benchmarks primarily evaluate final outputs, providing limited visibility into agent behavior within the research process. To address this gap, we present AutoMedBench, a workflow-aware benchmark for autonomous medical-AI research across diverse medical imaging and multimodal inference tasks, organizing agent execution into a unified five-stage workflow (S1-S5): Plan, Setup, Validate, Inference, and Submit. It comprises long-horizon tasks with each run averaging 33 agent turns, spanning five research tracks: segmentation, image enhancement, visual question answering (VQA), report generation, and lesion detection. Each task is evaluated under two difficulty tiers, Lite and Standard, which use the same data and metrics but differ in the amount of task-brief scaffolding, and each run is scored using both final task performance and S1-S5 stage scores, enabling stage-level analysis from the initial task brief to the final submitted artifact. Across thousands of recorded runs, stage-level scoring reveals that Validate is the weakest workflow stage on average, whereas Setup is the strongest, suggesting that current agents are better at making pipelines executable than at verifying their reliability. Post-run error analysis further shows that verification and submission failures dominate tagged errors, accounting for 37.7% and 38.1% of fired codes respectively, whereas task-understanding errors are rare at 0.9%, and runs with one fired error code have a 48% lower overall score than runs with no error code on average.

**arXiv ID:** 2606.01961
</details>

<details>
<summary><strong>ChatSOP: An SOP-Guided MCTS Planning Framework for Controllable LLM Dialogue Agents</strong> - Zhigen Li, Jianxiang Peng, Yanmeng Wang, Yong Cao, Tianhao Shen, Minghui Zhang, Linxi Su, Shang Wu, Yihang Wu, Yuqian Wang, Ye Wang, Wei Hu, Jianfeng Li, Shaojun Wang, Jing Xiao, Deyi Xiong - [[pdf]](https://arxiv.org/pdf/2407.03884)</summary>

**Abstract:** Dialogue agents powered by Large Language Models (LLMs) show superior performance in various tasks. Despite the better user understanding and human-like responses, their **lack of controllability** remains a key challenge, often leading to unfocused conversations or task failure. To address this, we introduce Standard Operating Procedure (SOP) to regulate dialogue flow. Specifically, we propose **ChatSOP**, a novel SOP-guided Monte Carlo Tree Search (MCTS) planning framework designed to enhance the controllability of LLM-driven dialogue agents. To enable this, we curate a dataset comprising SOP-annotated multi-scenario dialogues, generated using a semi-automated role-playing system with GPT-4o and validated through strict manual quality control. Additionally, we propose a novel method that integrates Chain of Thought reasoning with supervised fine-tuning for SOP prediction and utilizes SOP-guided Monte Carlo Tree Search for optimal action planning during dialogues. Experimental results demonstrate the effectiveness of our method, such as achieving a 27.95% improvement in action accuracy compared to baseline models based on GPT-3.5 and also showing notable gains for open-source models. Dataset and codes are publicly available.

**arXiv ID:** 2407.03884
</details>

<details>
<summary><strong>When Chatbots Accommodate: What AI Companions Optimize for in Vulnerable Conversations</strong> - Minh Duc Chu, Yifan Wu, Zhiyi Chen, Angel Hsing-Chi Hwang, Luca Luceri - [[pdf]](https://arxiv.org/pdf/2606.04431)</summary>

**Abstract:** Millions turn to AI companion chatbots during loneliness, grief, and personal crises. How these companion platforms respond in such moments can shape the trajectory of a user's vulnerable state. Yet we lack tools to characterize what each platform actually does when users open up. Existing audits score reactions to pre-defined crisis prompts and miss the underlying decision policy that governs sustained interaction. We address these gaps with two key contributions. First, we introduce the AI Companion Vulnerability-Response Taxonomy, a paired taxonomy of user vulnerability and chatbot response designed for analyzing extended companion chatbot interactions. Second, we infer the response policy each platform follows across distinct vulnerability scenarios by applying Inverse Reinforcement Learning to ~48k turns of real-world user conversations with GPT-4.1, this http URL, and Replika. Our findings reveal what AI companions prioritize in conversations with vulnerable users: GPT-4.1 reaches for advice, this http URL spreads its response across different strategies without a dominant mode, and Replika consistently asks questions and stays present. Each, however, downweights the responses that introduce corrective friction: GPT-4.1 probes less as conversations continue and when interacting with psychologically high-risk users; Replika advises bonded users more and challenges them less; this http URL shows no committed engagement strategy on internal distress. Estimated policies are invisible to output-level audits, providing a new lens for auditing chatbots in the wild and enabling more realistic safety evaluation.

**arXiv ID:** 2606.04431
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (17 papers)</h2></summary>

<details>
<summary><strong>Can Generalist Agents Automate Data Curation?</strong> - Feiyang Kang, Hanze Li, Adam Nguyen, Mahavir Dabas, Jiaqi W. Ma, Frederic Sala, Dawn Song, Ruoxi Jia - [[pdf]](https://arxiv.org/pdf/2606.04261)</summary>

**Abstract:** Curating training data is among the most consequential yet labor-intensive parts of modern AI development: practitioners iteratively propose, implement, evaluate, and revise data policies against noisy benchmark feedback. We ask whether generalist coding agents can automate this data-curation loop. We introduce *Curation-Bench*, an agent-centric benchmark that fixes the model, training recipe, and evaluation suite while giving agents command-line access to inspect data, implement policies, submit them to a fixed training/evaluation pipeline, and revise. In a vision-language instruction-tuning instantiation, out-of-the-box agents reach strong published data-selection baselines within ten iterations. However, trajectory analysis reveals a persistent *execution-research gap*: agents mainly tune local policy variants rather than explore new policy families, even when given strategy guides and paper references. Scaffolds requiring each iteration to cite, instantiate, and adapt a prior method shift agents toward method-guided exploration. The scaffolded agent autonomously composes -- without human design input -- a data-selection policy that outperforms strong published baselines at one-tenth their data budget. Overall, current agents can run the curation loop, but reliable data research requires scaffolded method adaptation, not open-ended prompting alone. Code and benchmark are open-sourced.

**arXiv ID:** 2606.04261
</details>

<details>
<summary><strong>Toward Pre-Deployment Assurance for Enterprise AI Agents: Ontology-Grounded Simulation and Trust Certification</strong> - Thanh Luong Tuan, Abhijit Sanyal - [[pdf]](https://arxiv.org/pdf/2606.04037)</summary>

**Abstract:** Pre-deployment verification of enterprise artificial intelligence (AI) agents remains a critical gap between large language model (LLM) capability benchmarking and production deployment. Post-deployment monitoring, human-in-the-loop controls, and prompt-level guardrails offer limited assurance once an agent is operating in production. We propose an ontology-grounded verification framework combining three components: an Agent Operational Envelope formalizing the certification space across permissions, domain constraints, safety properties, governance rules, and autonomy levels; an ontology-to-scenario generation pipeline that derives regulatory, operational, and adversarial test scenarios automatically; and a Trust Certificate carrying a machine-verifiable attestation with graduated deployment verdicts (Approved, Conditional, Rejected). A controlled pilot across four regulated industries (Fintech, Banking, Insurance, and Healthcare), instantiated as five industry-by-regulatory-regime cells across the United States and Vietnam, generated 1,800 scenarios evaluated against 125 primary-source regulatory requirements and 25 injected faults. Ontology-grounded generation (G4) achieved 48.3% regulatory coverage versus 33.1% for the persona-based baseline (corrected p = .0006) and the highest domain specificity (4.77/5.0; p = 2e-6). The coverage advantage over baseline and retrieval-augmented prompting was not robust after Bonferroni correction. Cross-validation across three LLM families (Claude Sonnet 4, Qwen 2.5 72B, Gemma 4 26B; 5,400 total scenarios) replicated the persona-versus-ontology pattern. The results establish ontology-grounded scenario generation as a credible complement to persona-based test suites for regulatory-intensive domains.

**arXiv ID:** 2606.04037
</details>

<details>
<summary><strong>Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation</strong> - Saroj Mishra - [[pdf]](https://arxiv.org/pdf/2606.04435)</summary>

**Abstract:** Multi-step agentic retrieval-augmented generation (RAG) pipelines have demonstrated significant capability for complex reasoning tasks, yet remain vulnerable to a class of failure that existing hallucination detection mechanisms systematically miss: cascading hallucination, where errors introduced at early pipeline stages propagate and amplify across successive reasoning steps, producing confident but factually incorrect final outputs. To address this vulnerability, we formalize cascading hallucination as a distinct failure mode in agentic RAG systems, present a four-type taxonomy of cascade patterns, and introduce CHARM (Cascading Hallucination Aware Resolution and Mitigation), an architectural framework for detecting and interrupting error propagation in multi-step reasoning pipelines. CHARM comprises four components - stage-level fact verification, cross-stage consistency tracking, confidence propagation monitoring, and cascade resolution triggering - that operate alongside standard agentic RAG pipelines without requiring architectural replacement. We evaluate CHARM on HotpotQA, MuSiQue, 2WikiMultiHopQA, and a custom adversarial dataset across LangChain agentic pipeline configurations, achieving an 89.4% cascade detection rate with a 5.3% false positive rate and 215 ms +/- 18 ms average latency overhead per stage, achieving an error propagation reduction of 82.1%, compared to 18.5% for output-level detectors. Component ablations confirm that each detection module contributes meaningfully to overall cascade coverage. CHARM integrates with human-in-the-loop oversight frameworks to provide a complete reliability and governance stack for production agentic AI deployment.

**arXiv ID:** 2606.04435
</details>

<details>
<summary><strong>Proof-Carrying Agent Actions: Model-Agnostic Runtime Governance for Heterogeneous Agent Systems</strong> - Zexun Wang - [[pdf]](https://arxiv.org/pdf/2606.04104)</summary>

**Abstract:** Agent systems execute through runtimes with very different control points: local coding tools, framework SDKs, managed agent platforms, API gateways, and observer-only integrations. A high-risk action such as publishing data externally may therefore appear as a shell command in one runtime, a tool call in another, and a hosted session transition in a third. This makes it difficult to answer a basic governance question consistently: what action was authorized, under whose authority, with what approval semantics, and with what evidence after execution?
This paper presents Proof-Carrying Agent Actions (PCAA), a runtime-neutral governance model centered on an action certificate rather than on a vendor-native session record. PCAA organizes control around five checkpoints: pre-action admissibility, action open, assumption capture, approval, and outcome closure. It binds these checkpoints to a portable action envelope, runtime and approval receipts, and replay-ready proof. The model is extended in two practical ways: the certificate is externality-aware, carrying boundary facts such as destination visibility and account provenance, and approval is described by explicit enforceability classes rather than by a single reviewed or unreviewed bit.
We study the model through a reference implementation in a heterogeneous agent control plane and a disclosure-bounded evaluation protocol. On a protected benchmark expanded from 24 executable seeds to 96 traces across four runtime families, PCAA preserves route quality while exposing distinct failure modes under ablation. The paper contributes a systems formulation of runtime governance around certificate-bearing actions and an implementation-grounded account of how that formulation can remain portable under runtime churn without collapsing into vendor-specific control surfaces.

**arXiv ID:** 2606.04104
</details>

<details>
<summary><strong>HighTide: An Agent-Curated Open-Source VLSI Benchmark Suite</strong> - Benjamin Goldblatt, Paolo Pedroso, Farhad Modaresi, Ethan Sifferman, Matthew R. Guthaus - [[pdf]](https://arxiv.org/pdf/2606.04126)</summary>

**Abstract:** We introduce HighTide, an evolving AI-assisted benchmark suite. Specifically, the contributions are: (i) a diverse open-source suite spanning multiple design languages and technology nodes, (ii) Bazel-based incremental RTL-to-GDS compilation with remote caching, (iii) AI-assisted design curation through twelve agent skills covering the design lifecycle, flow optimization, tool reference, and meta-maintenance, backed by per-design decision logs that serve as long-term memory of tuning rationale across the suite, and (iv) an infrastructure with RTL compilation verification for stable releases. The suite is publicly available and designed to grow with the open-source hardware ecosystem.

**arXiv ID:** 2606.04126
</details>

<details>
<summary><strong>Notarized Agents: Receiver-Attested Confidential Receipts for AI Agent Actions</strong> - Juan Figuera - [[pdf]](https://arxiv.org/pdf/2606.04193)</summary>

**Abstract:** Current AI agent observability is structurally compromised: the entity producing the activity log is the same entity whose activity is being logged. A compromised or buggy agent can omit, alter, or fabricate its own traces, and the operator running the agent has no independent way to detect tampering. We propose a class of protocols that resolves this by inverting the trust boundary: the service that receives an agent's call signs a receipt of what it observed using its own key, encrypts the receipt to the agent's owner, and publishes it to a public transparency log. The owner reconstructs a tamper-evident trail without trusting the agent or its operator. We instantiate the class as Sello, a protocol combining four properties absent in any current system: (P1) receiver-side signing, (P2) HPKE encryption to an owner public key bound to the authorization token via JWS, (P3) publication to a witness-cosigned Merkle log, and (P4) owner-side discovery by token reference. We describe the protocol, analyze its security under an adversary that controls the agent and its operator, present microbenchmarks of the cryptographic operations, and situate Sello among adjacent receipt-protocol work (Signet, AgentROA, Agent Passport System, draft-farley-acta, SCITT). We discuss known limitations including the suppression attack, service collusion, and the adoption-incentive problem.

**arXiv ID:** 2606.04193
</details>

<details>
<summary><strong>HYolo: An Intelligent IoT-Based Object Detection System Using Hypergraph Learning</strong> - Isha Abid, Fawad Khan, Muhammad Khuram Shahzad - [[pdf]](https://arxiv.org/pdf/2606.04345)</summary>

**Abstract:** This paper presents HYolo, an intelligent IoT-based object detection framework that integrates hypergraph learning into the YOLO architecture. Traditional YOLO-based object detection models primarily capture pairwise feature interactions and may fail to model complex high-order relationships among objects and contextual features. To address this limitation, HYolo incorporates hypergraph learning to capture richer contextual dependencies and improve object representation. Experimental evaluation on the COCO dataset demonstrates significant performance improvements over baseline YOLO models. The proposed approach achieves approximately 12% improvement in mAP@50 while enhancing overall detection accuracy and robustness. By modeling high-order feature relationships, HYolo provides improved contextual understanding and more reliable object detection performance in IoT-based environments. The results indicate that integrating hypergraph learning into object detection pipelines offers a promising direction for intelligent and context-aware IoT vision systems.

**arXiv ID:** 2606.04345
</details>

<details>
<summary><strong>What If Prompt Injection Never Left? Exploring Cross-Session Stored Prompt Injection in Agentic Systems</strong> - Yuanbo Xie, Tianyun Liu, Yingjie Zhang, Suchen Liu, Yulin Li, Liya Su, Tingwen Liu - [[pdf]](https://arxiv.org/pdf/2606.04425)</summary>

**Abstract:** Modern agentic systems transform LLMs from session-bounded assistants into stateful systems that persist and evolve shared world state across sessions through memories, filesystems, tools, and other long-lived contextual artifacts. This shift fundamentally expands the attack surface of prompt injection. However, prior works on prompt injection have largely focused on model-level threats within a single session, overlooking how cross-session persistent system state fundamentally changes the system-level risk of agentic systems. Inspired by stored cross-site scripting in web systems, we introduce cross-session stored prompt injection, where a successful injection can persist within agentic system state and silently influence future executions long after the original attacker interaction has ended. To systematically study this threat, we formalize stored prompt injection and develop a taxonomy of how adversarial content persists and affects agentic systems across sessions. We further develop a benchmark and sandbox toolkit to evaluate the risks of stored prompt injection, enabling quantitative analysis of attack success across different models, attack goals, and persistence channels. Our findings highlight that persistence transforms prompt injection from an ephemeral model-level threat into a long-lived system-level vulnerability embedded within agent execution state. We hope this work draws broader attention to this emerging threat and motivates the community to systematically study and mitigate system risks arising from persistence in agentic systems.

**arXiv ID:** 2606.04425
</details>

<details>
<summary><strong>CyberGym-E2E: Scalable Real-World Benchmark for AI Agents' End-to-End Cybersecurity Capabilities</strong> - Tianneng Shi, Robin Rheem, Dongwei Jiang, Mona Wang, Francisco De La Riega, Zhun Wang, Jingzhi Jiang, Alexander Cheung, Sean Tai, Jonah Cha, Jianhong Tu, Gabriel Han, Chenguang Wang, Jingxuan He, Wenbo Guo, Dawn Song - [[pdf]](https://arxiv.org/pdf/2606.04460)</summary>

**Abstract:** AI has the potential to transform cybersecurity by enabling systems that can autonomously detect, analyze, and remediate software vulnerabilities. However, existing cybersecurity evaluations of AI systems are limited in scale or scope, and fail to capture the end-to-end lifecycle of real-world software vulnerability discovery and remediation. To address this gap, we propose CyberGym-E2E, a large-scale and realistic end-to-end cybersecurity benchmark that comprehensively evaluates AI agents' abilities across the full lifecycle of vulnerability discovery, PoC generation, and patch generation. CyberGym-E2E is comprehensive and scalable, as we build an automated, agent-enhanced pipeline for transforming open-source vulnerability data into realistic evaluation environments. Currently, the benchmark consists of 920 real-world vulnerabilities across 139 different open-source projects.

**arXiv ID:** 2606.04460
</details>

<details>
<summary><strong>SePO: Self-Evolving Prompt Agent for System Prompt Optimization</strong> - Wangcheng Tao, Han Wu, Weng-Fai Wong - [[pdf]](https://arxiv.org/pdf/2606.04465)</summary>

**Abstract:** System prompt optimization improves agent behavior without modifying the underlying model, yielding human-readable, model-agnostic instructions. Existing methods build a prompt agent that refines task agents' system prompts, yet leave the prompt agent's own system prompt hand-engineered and fixed. We propose Self-Evolving Prompt Optimization (SePO), which treats the prompt agent's own system prompt as an optimization target alongside task agents' system prompts. SePO adopts a self-referential design. A single prompt agent improves both task agents' system prompts and its own under an open-ended evolutionary search that maintains an archive of candidate prompts as stepping stones. Training proceeds in two stages: pre-training evolves the prompt agent on a multi-task pool, and fine-tuning then applies it to a target task. Across five benchmarks spanning math (AIME'25), abstract reasoning (ARC-AGI-1), graduate-level science (GPQA), code generation (MBPP), and logic puzzles (Sudoku), SePO consistently outperforms Manual-CoT, TextGrad, and MetaSPO, improving the average accuracy by 4.49 points compared to Manual-CoT. The prompt optimization skill from pre-training also generalizes to tasks beyond the pre-training mixture, rather than memorizing per-task prompts.

**arXiv ID:** 2606.04465
</details>

<details>
<summary><strong>Self-Reflective APIs: Structure Beats Verbosity for AI Agent Recovery</strong> - Arquimedes Canedo, Grama Chethan - [[pdf]](https://arxiv.org/pdf/2606.05037)</summary>

**Abstract:** When an AI agent calls an API and hits a validation error, it needs more than what went wrong -- it needs what to do next. A self-reflective API returns, on validation failure, a machine-readable recovery\this http URL[] payload sufficient for the agent to repair the request and retry without external reasoning. On a leak-audited pilot ($N{=}30$ per cell, 3 LLMs, 10 adversarial tasks), structured suggestions lift task-completion rate by $+36.7$--$40.0$pp over plain-English diagnoses on Anthropic models (Fisher's exact $p \le 0.0022$), at $1.8$--$2.2\times$ better per-success token efficiency. The lift is not significant on gpt-4o-mini ($p{=}0.435$); a second-domain replication on a billing API confirms the pattern. The comparison only holds after auditing two undocumented classes of answer leakage in LLM benchmarks. We shipaudit\_prompt\this http URL as reusable CI infrastructure. Code and data: this https URL.

**arXiv ID:** 2606.05037
</details>

<details>
<summary><strong>Towards Efficient and Evidence-grounded Mobility Prediction with LLM-Driven Agent</strong> - Linyao Chen, Qinlao Zhao, Zechen Li, Mingming Li, Likun Ni, Jinyu Chen, Yuhao Yao, Xuan Song, Noboru Koshizuka, Hiroki Kobayashi - [[pdf]](https://arxiv.org/pdf/2606.05130)</summary>

**Abstract:** Individual-level mobility prediction is central to urban simulation, transportation planning, and policy analysis. Supervised sequence models achieve strong accuracy but require task-specific training and offer limited decision-level transparency. Recent LLM-based methods improve interpretability, yet mostly rely on static prompts and single-pass inference, limiting their ability to seek additional evidence when mobility signals are weak or conflicting. We propose \method{}, a training-free LLM-driven agent framework that formulates next-location prediction as adaptive evidence-controlled decision making. \method{} resolves routine cases through a fast path based on historical regularity, while ambiguous cases trigger iterative tool use over recent trajectories, historical behavior, stay-move likelihood, and geographical evidence. Across three mobility datasets, AgentMob achieves the strongest overall performance among training-free LLM-based methods, with GPT-5.4 reaching 71.42\% Acc@1 on BW, 33.14\% on YJMob100K, and 33.50\% on Shanghai ISP. On BW non-fast-path cases, the LLM controller improves Acc@1 from 30.65\% to 48.62\% over a same-tool statistical baseline, showing that its main benefit lies in resolving ambiguous predictions through adaptive evidence gathering. Our code is available at this https URL.

**arXiv ID:** 2606.05130
</details>

<details>
<summary><strong>LEAP: Supercharging LLMs for Formal Mathematics with Agentic Frameworks</strong> - Po-Nien Kung, Linfeng Song, Dawsen Hwang, Jinsung Yoon, Chun-Liang Li, Simone Severini, Mirek Olšák, Edward Lockhart, Quoc V Le, Burak Gokturk, Thang Luong, Tomas Pfister, Nanyun Peng - [[pdf]](https://arxiv.org/pdf/2606.03303)</summary>

**Abstract:** Large Language Models (LLMs) exhibit strong informal mathematical reasoning but struggle to generate mechanically verifiable proofs in formal languages like Lean. We present LEAP, an agentic framework that enables general-purpose foundation models to achieve state-of-the-art performance on automated formal theorem proving. LEAP leverages foundation model capabilities, such as informal reasoning, instruction following, and iterative self-refinement. By decomposing complex problems into smaller units, the system bridges formal proof construction with informal blueprints through continuous interaction with the Lean compiler. To provide a rigorous evaluation beyond increasingly saturated benchmarks, we introduce Lean-IMO-Bench, a benchmark of IMO-style problems formalized in Lean, with short statements yet highly non-routine and multi-step proofs across a wide range of difficulty levels. Empirically, on the latest 2025 Putnam Competition, an annual mathematics competition for undergraduate students in North America, LEAP solves all 12 problems, matching recent breakthroughs by frontier formal mathematical models. On Lean-IMO-Bench, LEAP boosts the one-shot formal solve rate of general-purpose LLMs from below 10% to 70%, notably surpassing the 48% benchmark set by a specialized, gold-medal-caliber IMO system. Furthermore, we demonstrate LEAP's research-level utility by autonomously formalizing complex proofs for open combinatorial challenges, including a verified proof for a key subproblem in Knuth's Hamiltonian decomposition of even-order Cayley graphs.

**arXiv ID:** 2606.03303
</details>

<details>
<summary><strong>Benchmarking Living-Screen-Native GUI Agents on Short-Video Platforms</strong> - Jiashu Yao, Heyan Huang, Daiqing Wu, Wangke Chen, Huaxi Ai, Haoyu Wen, Zeming Liu, Yuhang Guo - [[pdf]](https://arxiv.org/pdf/2606.04701)</summary>

**Abstract:** GUI agents today assume a static screen, where the world is frozen between two actions. However, real interfaces such as short-video applications violate this assumption, as their content keeps playing, and a competent user must decide what to watch and for how long. We formalize this task as Living-Screen-Native GUI agents and introduce LivingScreen, the first benchmark instantiating it on short-video platforms, with a faithful browser-based environment, a three-tier task suite, and metrics that jointly score accuracy and information efficiency. Evaluating extensive frontier models, we find that none reaches the human cost-accuracy performance, and that their dominant failure mode is over- and under-observation, pointing to observation control as a missing capability axis for future GUI agents. All data and code will be available at this https URL.

**arXiv ID:** 2606.04701
</details>

<details>
<summary><strong>Be Fair! Can Machine Learning Engineering Agents Adhere to Fairness Constraints?</strong> - Anna Richter, Julia Stoyanovich, Sebastian Schelter - [[pdf]](https://arxiv.org/pdf/2606.04971)</summary>

**Abstract:** Machine learning engineering (MLE) agents promise to automate end-to-end ML pipeline development from raw data and natural language instructions, potentially making ML accessible to non-technical domain experts. However, in sensitive and regulated domains, this abstraction creates a responsibility gap: end-users may lack visibility into design choices that affect correctness, robustness, fairness, and regulatory compliance. We argue that existing benchmarks are insufficient to assess whether MLE agents can be safely applied in such settings. We propose desiderata for a responsibility-centered evaluation framework and conduct an exploratory study on melanoma classification, focusing on fairness across skin tones as a responsibility constraint. When evaluating two recent MLE agents, we find that agent-generated pipelines show high variance and consistently underperform manually designed baselines in both predictive quality and fairness, despite fairness-oriented prompts. These preliminary results suggest that further research is needed towards redesigning MLE agents to allow humans to guide the search process and reliably assess the compliance and quality of the generated ML pipelines.

**arXiv ID:** 2606.04971
</details>

<details>
<summary><strong>Reconstructing Unobservable Temperature Fields via Simulation-Aided Intelligent Sensing</strong> - Monika Stipsitz, Hèlios Sanchis-Alepuz, Jacob Reynvaan, Silvester Sabathiel - [[pdf]](https://arxiv.org/pdf/2606.04582)</summary>

**Abstract:** Real-time monitoring of the temperature distribution within components and sub-structures is a challenging topic in many systems due to restrictions on feasible sensor locations. While machine learning (ML) proves a versatile tool in many applications, its adoption for high-resolution thermal monitoring is hindered by the availability of high-quality datasets for training. In this work, we propose a novel approach for generating datasets for industrial applications based on randomized physics-based simulations. We demonstrate the approach in a proof-of-concept hardware setup: A neural network (NN) trained only on such a synthetic dataset, is used to reconstruct the internal temperature field from sparse sensors embedded in the hardware. The NN-based reconstructions do not only outperform Kriging in robustness but also enable real-time inference, making the method suitable for online monitoring of otherwise unobservable thermal states.

**arXiv ID:** 2606.04582
</details>

<details>
<summary><strong>D$^3$-MoE:Dual Disentangled Diffusion Mixture-of-Experts for Style-Controllable End-to-End Autonomous Driving</strong> - Renju Feng, Rukang Wang, Ning Xi, Jianguo Yu, Liping Lu, Pan Zhou, Duanfeng Chu - [[pdf]](https://arxiv.org/pdf/2606.04884)</summary>

**Abstract:** Traditional end-to-end autonomous driving frameworks frequently suffer from the "style-averaging" dilemma when trained on high-variance human demonstrations, yielding homogenized, style-uncontrollable, and even kinematically unsafe policies. To overcome this limitation, we present D$^3$-MoE (Dual Disentangled Diffusion Mixture-of-Experts), which disentangles trajectory modeling along two complementary axes. On the behavioral axis, generation is decoupled from selection: a style-conditioned diffusion process synthesizes multi-style candidate trajectories in parallel within a single scene, allowing a downstream module to select the optimal trajectory based on user preference or an evaluation score. On the physical axis, decoupled longitudinal and lateral routers activate their respective experts during inference time, trained without manual labels using self-supervised targets from orthogonal ground-truth kinematics. These activated experts, architected as Diffusion Transformers (DiT) and equipped with style-conditioned AdaLN and asymmetric lateral-fusion cross-attention, independently predict their corresponding physical state before being reassembled into a unified, kinematically coherent trajectory. Extensive evaluations on the challenging NAVSIM benchmark demonstrate that D$^3$-MoE achieves state-of-the-art planning performance, reaching 88.2 PDMS and 84.3 EPDMS by default. Moreover, our Best-of-Three ensemble strategy effectively broadens the multi-modal solution space, raising performance to 91.3 PDMS and 87.5 EPDMS. Both quantitative and qualitative analyses jointly confirm the framework's advantages in planning quality and style controllability.

**arXiv ID:** 2606.04884
</details>

</details>

<details open>
<summary><h2>LLM Agents (14 papers)</h2></summary>

<details>
<summary><strong>Exploring Cross-Scenario Generality of Agentic Memory Systems: Diagnostics and a Strong Baseline</strong> - Zhikai Chen, Jialiang Gu, Junyu Yin, Xianxuan Long, Shenglai Zeng, Xiaoze Liu, Kai Guo, Keren Zhou, Jiliang Tang - [[pdf]](https://arxiv.org/pdf/2606.04315)</summary>

**Abstract:** LLM agents accumulate histories that outgrow their context windows, motivating a growing literature on memory systems. Yet most existing designs are tuned to a single scenario (multi-session chat or a single trajectory format), and there is little evidence that they generalize across the heterogeneous trajectories agents encounter in deployment. We revisit eight memory systems plus an agentic harness for search problems, on five scenarios: single-turn QA, multi-session chat, agentic-trajectory QA, memory stress tests, and long-horizon agentic tasks. The harness, which self-manages flat text-file storage via tool calls, achieves the best cross-task ranking, suggesting that memory performance hinges on giving the agent active control over storage and retrieval rather than on a passive store behind a fixed pipeline. We instantiate this insight in AutoMEM, an agentic memory harness with a self-managed tool interface that achieves the best cross-scenario generality among the systems we evaluate.

**arXiv ID:** 2606.04315
</details>

<details>
<summary><strong>Scaling Self-Evolving Agents via Parametric Memory</strong> - Tao Ren, Weiyao Luo, Hui Yang, Rongzhi Zhu, Xiang Huang, Yuchuan Wu, Bingxue Chou, Jieping Ye, Jiafeng Liang, Yongbin Li, Yijie Peng - [[pdf]](https://arxiv.org/pdf/2606.04536)</summary>

**Abstract:** Existing memory-augmented LLM agents store past experience exclusively in prompt space, as textual summaries or retrieved passages, while keeping model parameters frozen throughout a rollout. Such agents can \emph{look up} what they have seen but cannot \emph{learn from} it: their policy is unchanged by experience, and any information dropped from the context is permanently lost. We introduce \texttt{TMEM}, a self-evolving parametric memory framework in which the agent not only compresses history into explicit memory but also absorbs distilled supervision into fast LoRA weights $\Delta_t$ via lightweight online updates, genuinely altering its future behavior within a single episode. We formalize this as an agentic decision process with fast-weight rollout dynamics: actions are sampled from $\pi_{\theta_0+\Delta_t}$, while extraction actions produce supervision that updates $\Delta_t$ for subsequent decisions. This view makes the extraction policy directly optimizable by RL: training $\theta_0$ improves not only task actions but also the quality of the data used for online LoRA adaptation. We further propose SVD-based initialization of the LoRA subspace to accelerate online convergence. Experiments on LoCoMo, LongMemEval-S, multi-objective search, and CL-Bench show that \texttt{TMEM} consistently outperforms summary-based and retrieval-based baselines across different model scales.

**arXiv ID:** 2606.04536
</details>

<details>
<summary><strong>Parthenon Law: A Self-Evolving Legal-Agent Framework</strong> - Hejia Geng, Leo Liu - [[pdf]](https://arxiv.org/pdf/2606.04602)</summary>

**Abstract:** As agents grow more capable, legal-domain LLM agents promise to turn document-heavy matters into reviewable work products -- yet reliable deployment faces three obstacles: no large-scale evidence on how today's strongest model-and-harness combinations behave on end-to-end legal matters; no agent architecture adapted to the legal vertical, only general-purpose harnesses; and, in a setting that keeps shifting with new facts, authorities, and deadlines, no mechanism for systems to learn from their own outcomes. We address each. A large-scale empirical study on Harvey LAB -- $12{,}510$ agent trajectories -- shows that even frontier agents remain far from completing matters in a single pass: per-criterion accuracy climbs with stronger models while strict matter completion stalls. We then introduce \textsc{Parthenon}, a self-evolving legal-agent framework that factors Model, Harness, Agent roles, legal Knowledge, deterministic Tools, and procedural Skills into auditable surfaces for source traceability, date and number grounding, deliverable compliance, and issue closure. Finally, an anti-leakage learning loop converts scored failures into task-agnostic edits to skills, tools, and knowledge, letting the system improve with experience -- as a firm refines its checklists and playbooks after each matter -- without touching model weights. Across our large-scale empirical analysis, \textsc{Parthenon} substantially improves the performance of state-of-the-art models and harnesses on legal-matter tasks.

**arXiv ID:** 2606.04602
</details>

<details>
<summary><strong>RUBAS: Rubric-Based Reinforcement Learning for Agent Safety</strong> - Xian Qi Loye, Qinglin Su, Zhexin Zhang, Shiyao Cui, Qi Zhu, Fei Mi, Hongning Wang, Minlie Huang - [[pdf]](https://arxiv.org/pdf/2606.04051)</summary>

**Abstract:** The evolution of LLMs into tool-enabled agents creates a new class of safety challenges associated with real-world execution rather than simple text generation. Existing alignment methods often rely on coarse refusal signals or static supervision, making it difficult to balance safety with useful tool execution across diverse agentic risks. We introduce RUBAS, a rubric-based reinforcement learning framework for agent safety. RUBAS decomposes agent behavior into four dimensions: tool-use safety, argument safety, response safety, and helpfulness. These structured rubrics provide fine-grained and interpretable rewards over complete agent trajectories, enabling reinforcement learning to optimize safe tool use while preserving task completion. Extensive experiments across multiple agent safety benchmarks and models show that RUBAS improves safety over standard alignment baselines, reduces tool-grounded hallucinations, and maintains competitive utility. Our results suggest that multi-dimensional rubric rewards provide an effective training signal for aligning LLM agents in safety-critical tool-use settings.

**arXiv ID:** 2606.04051
</details>

<details>
<summary><strong>Caught in the Act(ivation): Toward Pre-Output and Multi-Turn Detection of Credential Exfiltration by LLM Agents</strong> - Kargi Chauhan, Pratibha Revankar - [[pdf]](https://arxiv.org/pdf/2606.04141)</summary>

**Abstract:** LLM agents often place sensitive credentials in the same context window as untrusted retrieved content, creating a direct path for indirect prompt injection to induce credential exfiltration. We study this failure mode through three complementary defenses. First, we ask whether activation probes can detect credential access before output tokens are emitted. Second, we construct honeytokens from format-specific character models and calibrate detection with split conformal prediction. Third, we treat multi-turn exfiltration as a cumulative information-flow problem and track an estimated leakage budget across conversation turns. In controlled experiments on open-weight models, activation features separate benign and credential-seeking prompts with high accuracy, including under held-out encoding transformations. In a small synthetic multi-turn suite, cumulative accounting detects attacks that per-turn detectors miss. These results are preliminary: the multi-turn benchmark is in-house and small, the activation method requires white-box access, and the information estimator provides a practical signal rather than a formal upper bound. Still, the results suggest that credential-exfiltration defenses should combine pre-output monitoring, calibrated canary detection, and temporal leakage accounting rather than relying only on text-level output filters.

**arXiv ID:** 2606.04141
</details>

<details>
<summary><strong>From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents</strong> - Pritam Dash, Tongyu Ge, Aditi Jain, Tanmay Shah, Zhiwei Shang - [[pdf]](https://arxiv.org/pdf/2606.04329)</summary>

**Abstract:** Memory is a core component of AI agents, enabling them to accumulate knowledge across interactions and improve performance. However, persistent memory introduces the risk of memory poisoning, where a single adversarial memory write can exert long-term influence over agent behavior. We present a systematic study of memory poisoning in LLM-based agents. We identify four memory write channels and nine structural vulnerabilities in model capabilities, system prompt design, and agent system architecture that make these channels exploitable. Based on these vulnerabilities, we develop a taxonomy of six classes of memory poisoning attacks. Furthermore, we design MPBench -- a benchmark for evaluating memory poisoning attacks, and show that agents designed to write and retrieve memory more aggressively are more exploitable. We also show that existing prompt injection defenses fail to cover memory poisoning attacks. Our findings provide a foundation for understanding and mitigating memory poisoning attacks against AI agents.

**arXiv ID:** 2606.04329
</details>

<details>
<summary><strong>Provably Auditable and Safe LLM Agents from Human-Authored Ontologies</strong> - Aaron Sterling - [[pdf]](https://arxiv.org/pdf/2606.04903)</summary>

**Abstract:** We introduce the LLM agent architecture Agentic Redux, intended for use with nontrivial problem domains that require linear auditability. Using the typed lambda calculus, we prove that, run on appropriate domains, Agentic Redux executions are semantically guaranteed to be correct, with all decisions recorded in an append-only ledger. We present two production-grade appropriate domains, in healthcare billing compliance, and security vulnerability disclosure. Working code for Agentic Redux run on both domains is available in a supporting code repository. We also introduce Ontology-First Agent Design, a methodology for creation of agent frameworks on a problem domain, in which a human expert ontologizes the problem domain with Basic Formal Ontology, and then assigns an LLM to derive roles that agents and humans-in-the-loop can fill, in order to work the problems in the domain.

**arXiv ID:** 2606.04903
</details>

<details>
<summary><strong>From Agent Traces to Trust: Evidence Tracing and Execution Provenance in LLM Agents</strong> - Yiqi Wang, Jiaqi Zhang, Taotao Cai, Zirui Liu, Qingqiang Sun, Zequn Sun, Zhangkai Wu, Mingkai Zhang, Yanming Zhu - [[pdf]](https://arxiv.org/pdf/2606.04990)</summary>

**Abstract:** Large language model (LLM)-based agents increasingly solve complex tasks by interacting with external tools, retrieval systems, memory modules, environments, and other agents. These capabilities expand agent autonomy, but also make agent behavior harder to verify, debug, and audit. Final-answer accuracy alone cannot explain how an output was produced, which evidence supported each claim, whether tool calls were justified, how memory influenced later decisions, or where execution failures originated. Evidence tracing and execution provenance address this gap by modeling how retrieved evidence, tool outputs, memory items, environment observations, intermediate claims, actions, and final answers are connected throughout agent execution. This survey provides a systematic review and conceptual framework for evidence tracing and execution provenance in LLM agents. We organize related work around a unified provenance perspective that connects retrieval grounding, claim support, tool-use safety, memory lineage, observability, debugging, audit, and recovery. We introduce a taxonomy covering trace sources, evidence and execution units, provenance relations, tracing granularity and timing, representation forms, and trust functions. We review key methodological directions, including provenance representation, evidence attribution, tool-use provenance, runtime guardrails, provenance-bearing memory, trace-based observability, and failure diagnosis. We also map existing benchmarks, datasets, and evaluation metrics to provenance-related capabilities, and discuss how evaluation can move from final-answer correctness toward process-level accountability. Finally, we outline open challenges, including unified trace schemas, claim-level and semantic provenance, provenance-aware safety mechanisms, realistic execution-trace benchmarks, recovery-oriented evaluation, and privacy-aware audit infrastructure.

**arXiv ID:** 2606.04990
</details>

<details>
<summary><strong>Adaptive Minds: Empowering Agents with LoRA-as-Tools</strong> - Pavan C Shekar, Aswanth Krishnan - [[pdf]](https://arxiv.org/pdf/2510.15416)</summary>

**Abstract:** We investigate a framework in which LoRA adapters are treated as callable tools that a base language model can dynamically select and invoke. We hypothesize that, when adapters are trained to provide strong domain-specific gains and are exposed with clear metadata, a base model can reliably route queries to the appropriate expert, effectively aggregating the benefits of many specialized adapters within a single framework. We introduce Adaptive Minds, a general framework within which we study both single-step routing and multi-step agentic reasoning. In this setting, the agent can iteratively invoke multiple adapters alongside other tools (e.g., external APIs, retrieval systems, or execution environments) and reason over their outputs across multiple steps. This reframes adapters as modular skills or memory units that can be composed during reasoning rather than statically applied. In our evaluation, the routing layer reaches 98.3% accuracy on a 30-adapter library, and well-trained specialists provide +4.6 to +84.0 percentage points of strict-scorer gain across nine task families under a single shared training recipe; the AM router aggregates these gains within 5 pp of the direct specialist on every benchmark whose queries surface domain signal. Our findings suggest that the effectiveness of this approach depends on the quality and specialization of individual adapters, and that enabling flexible composition of many such experts can significantly expand the practical capabilities of language model agents, moving toward more general, tool-augmented intelligence.

**arXiv ID:** 2510.15416
</details>

<details>
<summary><strong>Formal Semantics for Agentic Tool Protocols: A Process Calculus Approach</strong> - Andreas Schlapbach - [[pdf]](https://arxiv.org/pdf/2603.24747)</summary>

**Abstract:** The emergence of large language model agents capable of invoking external tools has created urgent need for formal verification of agent protocols. Two paradigms dominate this space: Schema-Guided Dialogue (SGD), a research framework for zero-shot API generalization, and the Model Context Protocol (MCP), an industry standard for agent-tool integration. While both enable dynamic service discovery through schema descriptions, their formal relationship remains unexplored. Building on prior work establishing the conceptual convergence of these paradigms, we present the first process calculus formalization of SGD and MCP, proving they are structurally bisimilar under a well-defined mapping Phi. However, we demonstrate that the reverse mapping Phi^{-1} is partial and lossy, revealing critical gaps in MCP's expressivity. Through bidirectional analysis, we identify five principles -- semantic completeness, explicit action boundaries, failure mode documentation, progressive disclosure compatibility, and inter-tool relationship declaration -- as necessary and sufficient conditions for full behavioral equivalence. We formalize these principles as type-system extensions MCP+, proving MCP+ is isomorphic to SGD. Our work provides the first formal foundation for verified agent systems and establishes schema quality as a provable safety property.

**arXiv ID:** 2603.24747
</details>

<details>
<summary><strong>Rethinking Continual Experience Internalization for Self-Evolving LLM Agents</strong> - Jingwen Chen, Wenkai Yang, Shengda Fan, Wenbo Nie, Chenxing Sun, Shaodong Zheng, Yangen Hu, Lu Pan, Ke Zeng, Yankai Lin - [[pdf]](https://arxiv.org/pdf/2606.04703)</summary>

**Abstract:** Experience internalization converts contextual experience from past interactions into reusable parametric capability, offering a promising path toward continual learning in large language models (LLMs). While prior work has predominantly focused on single-iteration transfer, we discover that under multi-iteration experience learning, existing methods suffer from a progressive capability collapse rather than compounding improvement. We systematically examine this failure through three vital dimensions of experience internalization: (1) Experience Granularity: We find that principle-level experience is more durable than instance-level experience, as it effectively abstracts transferable strategies away from trajectory-specific details. (2) Experience Injection Pattern: Our analysis reveals that step-wise injection significantly outperforms global injection by aligning experience with intermediate decision states, a property that is critical for long-horizon tool use. (3) Internalization Regime: We demonstrate that off-policy context-distillation on high-quality teacher trajectories provides a substantially more stable training signal than on-policy context-distillation, which is inherently limited by local corrections on student-induced flawed states. Together, these insights yield a simple yet robust recipe for stable and sustainable experience internalization, providing concrete guidance for engineering self-evolving and continually learning LLMs.

**arXiv ID:** 2606.04703
</details>

<details>
<summary><strong>PersonaTree: Structured Lifecycle Memory for Person Understanding in LLM Agents</strong> - Yubo Hou, Jingwei Song, Hongbo Zhang, Zhisheng Chen, Bang Xiao, Tao Wan, Zengchang Qin - [[pdf]](https://arxiv.org/pdf/2606.04780)</summary>

**Abstract:** Persistent LLM agents require memory representations that make the formation of person understanding explicit across long term interaction. Existing agent memory methods emphasize information retention and retrieval, yet give limited account of how accumulated interaction evidence is abstracted into person understanding. We view this process as schema formation, where situated evidence is abstracted into reusable patterns and stable person level claims. We introduce PersonaTree, a structured lifecycle memory framework that realizes this view as a three level persona tree with explicit support paths from evidence to claims. PersonaTree maintains the tree through conservative writing, confidence guided consolidation, and query conditioned path retrieval, returning only the evidence depth required by each query. Across six person understanding and persistent memory benchmarks with three answer backbones, PersonaTree ranks first in 12 of 18 compact scores and reaches the top two in 16 settings. Ablations show that hierarchy improves abstract person understanding on KnowMe, while support path retrieval improves RealPref alignment under a comparable context budget.

**arXiv ID:** 2606.04780
</details>

<details>
<summary><strong>Agent Planning Benchmark: A Diagnostic Framework for Planning Capabilities in LLM Agents</strong> - Haoyu Sun, Wenxuan Wang, Mingyang Song, Jujie He, Weinan Zhang, Yang Liu, Yang Yang, Yu Cheng - [[pdf]](https://arxiv.org/pdf/2606.04874)</summary>

**Abstract:** Planning is central to LLM agents: before acting, an agent must decompose goals, select tools, reason over constraints, and decide when a task is infeasible. Yet existing agent evaluations often report only end-to-end success, making it difficult to determine whether failures stem from planning or execution. We introduce \textbf{Agent Planning Benchmark (APB)}, a planning-specific diagnostic benchmark with 4,209 multimodal cases across 22 domains and five settings, covering holistic planning, feedback-conditioned step-wise planning, and robustness under extraneous tools, broken tools, and unsolvable tasks. Across 12 MLLMs, APB reveals systematic weaknesses in long-horizon planning, tool-noise robustness, calibrated refusal, and inference-time refinement. We further validate APB on 200 ToolSandbox tasks and 200 $\tau^2$-bench tasks, where APB-guided refinement consistently improves plan correctness, plan grade, and downstream execution metrics across three representative models. APB thus serves as an upstream diagnostic complement to execution benchmarks.

**arXiv ID:** 2606.04874
</details>

<details>
<summary><strong>GroupTravelBench: Benchmarking LLM Agents on Multi-Person Travel Planning</strong> - Xiang Cheng, Yulan Hu, Lulu Zheng, Zheng Pan, Xin Li, Yong Liu - [[pdf]](https://arxiv.org/pdf/2605.25200)</summary>

**Abstract:** Travel planning in the real world is overwhelmingly a \textit{group} activity, yet existing LLM travel-planning benchmarks reduce it to a single user, where the field is approaching saturation. This single-user assumption sidesteps what makes group planning hard for an agent: discovering private preferences across multiple users, surfacing conflicts, and balancing utility against fairness. To bring the task back to its multi-user reality, we introduce \textbf{\textit{GroupTravelBench}}, the first benchmark for \textbf{multi-user, multi-turn} travel planning. Built from real user profiles, POI data, and ticket prices, it comprises 650 tasks across three difficulty levels, each running in a synchronous group-chat sandbox with cached tool data for reproducible offline evaluation. Beyond the multi-step reasoning and tool use that single-user benchmarks already test, GroupTravelBench probes three group-specific capabilities: \textit{(i) elicitation} of private preferences through multi-turn dialogue; \textit{(ii) coordination} of inter-user conflicts via compromise or subgrouping; and \textit{(iii) planning} that balances group utility against fairness. We pair this with a complementary evaluation framework combining rule-based outcome metrics and LLM-judge process metrics. Across a wide range of frontier models, even the strongest agents fall short on all four rule-based outcome metrics, with plan validity below 12\%, suggesting that group-level outcome quality is a key open challenge for LLM travel-planning agents.

**arXiv ID:** 2605.25200
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (28 papers)</h2></summary>

<details>
<summary><strong>SMAC-Talk: A Natural Language Extension of the StarCraft Multi-Agent Challenge for Large Language Models</strong> - Joel Sol, Homayoun Najjaran - [[pdf]](https://arxiv.org/pdf/2606.04202)</summary>

**Abstract:** As LLMs become more widely deployed, they are increasingly expected to work alongside other AI agents rather than operating in isolation. Effective coordination in these settings requires agents to communicate, share information and make decisions under uncertainty. We introduce SMAC-Talk, a natural language extension of the StarCraft Multi-Agent Challenge for evaluating LLM-based agents in cooperative multi-agent environments. The environment has several key features such as decentralized control, partial observability and long-horizon decision making. SMAC-Talk includes a natural language communication channel which is used to probe agent coordination and trust. We use this communication channel to construct different evaluation scenarios, including settings with an embedded deceptive communicator that tries to disrupt and deceive allies through communication alone. We provide three agents for benchmarking using 4 models from the Qwen3.5 family and study how reasoning structure, memory and model scale affect coordination between agents. We release SMAC-Talk as an open benchmark to support the research community in developing and evaluating LLM agents in cooperative multi-agent settings.

**arXiv ID:** 2606.04202
</details>

<details>
<summary><strong>AgentJet: A Flexible Swarm Training Framework for Agentic Reinforcement Learning</strong> - Qingxu Fu, Boyin Liu, Shuchang Tao, Zhaoyang Liu, Bolin Ding - [[pdf]](https://arxiv.org/pdf/2606.04484)</summary>

**Abstract:** We present AgentJet, a distributed swarm training framework for large language model (LLM) agent reinforcement learning. Unlike centralized frameworks that tightly couple agent rollouts with model optimization, AgentJet adopts a decoupled multi-node architecture in which swarm server nodes host trainable models and run optimization on GPU clusters, whereas swarm client nodes execute arbitrary agents on arbitrary devices. This design provides capabilities that are difficult to support in centralized frameworks: (1) heterogeneous multi-model reinforcement learning, enabling the training of heterogeneous multi-agent teams with multiple LLM as brains; (2) multi-task cocktail training with isolated agent runtimes; (3) fault-tolerant execution that prevents external environment failures from interrupting the training process; and (4) live code iteration, which allows agents to be edited during training by replacing swarm client nodes. To support efficient RL in multi-model, multi-turn, and multi-agent settings, AgentJet introduces a context tracking module with timeline merging, which consolidates redundant context and achieves a 1.5-10x training speedup. Finally, AgentJet introduces an automated research system that takes a research topic as input and autonomously conducts long-horizon, multi-day RL studies on large-scale clusters. By leveraging the swarm architecture, this system reproduces key exploratory workflows of RL researchers without human intervention during execution.

**arXiv ID:** 2606.04484
</details>

<details>
<summary><strong>Plan First, Judge Later, Run Better: A DMAIC-Inspired Agentic System for Industrial Anomaly Detection</strong> - Yongzi Yu, Ao Li, Le Wang, Ziyue Li, Fugee Tsung, Yuxuan Liang, Man Li - [[pdf]](https://arxiv.org/pdf/2606.04599)</summary>

**Abstract:** Large language model (LLM) agents have shown promise in automating complex data-analysis workflows, but their reliable deployment remains challenging in high-stakes industrial scenarios. Industrial anomaly detection (IAD) is essential for manufacturing quality, safety, and efficiency, yet existing LLM-based IAD agents mainly focus on execution while under-exploiting strategy formulation. Consequently, they struggle to handle heterogeneous modalities in a unified and cost-effective manner. Inspired by the DMAIC quality-management framework, we propose DMAIC-IAD (DMAIC-inspired Agentic Industrial Anomaly Detection), a "Plan First, Judge Later" multi-agent system that aligns LLM agents with structured industrial problem-solving. DMAIC-IAD distills heterogeneous references into standardized operating procedures (SOPs) before strategy generation, and introduces a pre-trained execution-free judge model to rank candidate strategies without costly runtime trials. Extensive experiments across four modalities show that DMAIC-IAD improves average detection performance over applicable agentic baselines by 37.76%.

**arXiv ID:** 2606.04599
</details>

<details>
<summary><strong>Fog of Love: Engineering Virtuous Agent Behavior with Affinity-based Reinforcement Learning in a Game Environment</strong> - Ajay Vishwanath, Christian Omlin - [[pdf]](https://arxiv.org/pdf/2606.04750)</summary>

**Abstract:** Instilling virtuous behavior in artificial intelligence has seen increasing interest. One of the techniques proposed is known as affinity-based reinforcement learning, which uses policy regularization on the objective function to incentivize virtuous actions without being fully dependent on the reward function design. Thus far, this technique has been demonstrated to be effective in grid worlds and toy-problem environments with minimal state and action spaces. To expand this research to more sophisticated environments, we introduce a two-player multi-agent environment based on the role-playing board game known as Fog of Love. In this environment, two agents compete to fulfill their individual virtues, while also cooperating to satisfy their relationship. Given the multi-agent nature, this is a complex problem where multi-agent deep deterministic policy gradient agents neither compete nor cooperate successfully. We present evidence that localized affinities enhance agent performance in achieving both competitive and cooperative objectives, resulting from superior overall scores in both domains. This not only results in virtuous choices but also clarifies an agent's teleology and makes its behavior human-level interpretable.

**arXiv ID:** 2606.04750
</details>

<details>
<summary><strong>Tree-Based Formalization of Multi-Agent Complementarity in Human-AI Interactions</strong> - Andrea Ferrario - [[pdf]](https://arxiv.org/pdf/2606.04779)</summary>

**Abstract:** Complementarity is the case in which a human--AI interaction (HAI) outperforms the best prediction benchmark available among its members. Although this idea is central in HAI research, formal work on complementarity remains limited. Existing frameworks do not model how agents' predictions compose into workflow-sensitive multi-agent protocols. We close this gap by introducing a tree-based formalization of complementarity in multi-agent HAI. An HAI protocol is represented by an ordered agent-role configuration together with a rooted planar binary tree whose leaves are decorated by prediction vectors. A local binary composition rule is evaluated recursively along the tree, yielding a tree-relative complementarity functional relative to a pointwise-min oracle benchmark. We prove four results. First, selector-based HAIs, including self- or AI-reliance, cannot achieve complementarity regardless of task, loss, or prediction quality. Second, in regression under squared loss, complementarity is equivalent to Euclidean distance minimization from the ground-truth vector; for $N=2$, the optimal linear-pooling weight has a closed form and a residual-correction interpretation. Third, under linear local composition, every protocol tree defines a barycentric coordinate chart on the simplex of leaf weights; Tamari-cover reparameterizations of protocol trees preserve complementarity, and for $N=4$, they satisfy the pentagon identity. Fourth, in binary classification, no internal local composition can achieve complementarity under endpoint-monotone losses, including standard Bregman and many finite Bernoulli $f$-divergence losses; an analogous obstruction holds for multiclass aggregation under cross-entropy. In summary, our framework shows that complementarity is attainable in multi-agent regression, but obstructed in classification under natural conditions on local aggregation and loss functions.

**arXiv ID:** 2606.04779
</details>

<details>
<summary><strong>Streaming Communication in Multi-Agent Reasoning</strong> - Zhen Yang, Xiaogang Xu, Wen Wang, Cong Chen, Xander Xu, Ying-Cong Chen - [[pdf]](https://arxiv.org/pdf/2606.05158)</summary>

**Abstract:** Multi-agent reasoning systems adopt a "generate-then-transfer" paradigm that forces end-to-end latency to scale linearly with pipeline depth. We introduce StreamMA, a multi-agent reasoning system that streams each reasoning step to downstream agents as soon as it is generated, pipelining adjacent agents and thus reducing latency. Surprisingly, this pipelining also improves effectiveness: because multi-step reasoning quality is non-uniform and early steps are more reliable than later ones, working with these reliable early steps instead of the full chain prevents error-prone late steps from misleading downstream agents. We formalize both advantages with the first closed-form joint analysis of stream, serial, and single protocols, deriving the effectiveness ordering, speedup upper bound, and cost ratio. Across eight reasoning benchmarks spanning mathematics, science, and code, two frontier LLMs (Claude Opus 4.6 and GPT-5.4), and three topologies (Chain, Tree, Graph), StreamMA outperforms both baselines (avg. +7.3 pp, max +22.4 pp on HMMT 2026; Claude Opus 4.6-high). Beyond these contributions, we discover a "step-level scaling law": increasing per-agent steps consistently improves both effectiveness and efficiency, a new scaling dimension orthogonal to and composable with agent-count scaling.

**arXiv ID:** 2606.05158
</details>

<details>
<summary><strong>BRAINCELL-AID: An Agentic AI Created Brain Cell Type Resource for Community Annotation</strong> - Rongbin Li, Wenbo Chen, Zhao Li, Rodrigo Munoz-Castaneda, Jinbo Li, Neha S. Maurya, Arnav Solanki, Huan He, Hanwen Xing, Meaghan Ramlakhan, Zachary Wise, Nelson Johansen, Zhuhao Wu, Hua Xu, Michael Hawrylycz, W. Jim Zheng - [[pdf]](https://arxiv.org/pdf/2510.17064)</summary>

**Abstract:** Single-cell RNA sequencing has transformed our ability to identify diverse cell types and their transcriptomic signatures. However, annotating these signatures-especially those involving poorly characterized genes-remains a major challenge. Traditional methods, such as Gene Set Enrichment Analysis (GSEA), depend on well-curated annotations and often perform poorly in these contexts. Large Language Models (LLMs) offer a promising alternative but struggle to represent complex biological knowledge within structured ontologies. To address this, we present BRAINCELL-AID (BRAINCELL-AID: this https URL), a novel multi-agent AI system that integrates free-text descriptions with ontology labels to enable more accurate and robust gene set annotation. By incorporating retrieval-augmented generation (RAG), we developed a robust agentic workflow that refines predictions using relevant PubMed literature, reducing hallucinations and enhancing interpretability. Using this workflow, we achieved correct annotations for 77% of mouse gene sets among their top predictions. Applying this approach, we annotated 5,322 brain cell clusters from the comprehensive mouse brain cell atlas generated by the BRAIN Initiative Cell Census Network, enabling novel insights into brain cell function by identifying region-specific gene co-expression patterns and inferring functional roles of gene ensembles. BRAINCELL-AID also identifies Basal Ganglia-related cell types with neurologically meaningful descriptions. Hence, we create a valuable resource to support community-driven cell type annotation.

**arXiv ID:** 2510.17064
</details>

<details>
<summary><strong>Transformer-Based Autonomous Driving Models and Deployment-Oriented Compression: A Survey</strong> - Juan Zhong, Yuhang Shi, Zukang Xu, Xi Chen - [[pdf]](https://arxiv.org/pdf/2304.10891)</summary>

**Abstract:** Transformer-based models are becoming a central paradigm in autonomous driving because they can capture long-range spatial dependencies, multi-agent interactions, and multimodal context across perception, prediction, and planning. At the same time, their deployment in real vehicles remains difficult because high-capacity attention-based architectures impose substantial latency, memory, and energy overhead. This survey reviews representative Transformer-based autonomous driving models and organizes them by task role, sensing configuration, and architectural design. More importantly, it examines these models from a deployment-oriented perspective and analyzes how efficiency constraints reshape model design choices in practice. We further review compression and acceleration strategies relevant to Transformer-based driving systems, including quantization, pruning, knowledge distillation, low-rank approximation, and efficient attention, and discuss their benefits, limitations, and task-dependent applicability. Rather than treating compression as an isolated post-processing step, we highlight it as a system-level design consideration that directly affects deployability, robustness, and safety. Finally, we identify open challenges and future research directions toward standardized, safety-aware, and hardware-conscious evaluation of efficient autonomous driving systems.

**arXiv ID:** 2304.10891
</details>

<details>
<summary><strong>Exploring the Topology and Memory of Consensus: How LLM Agents Agree, Fragment, or Settle When Forming Conventions</strong> - Aliakbar Mehdizadeh, Martin Hilbert - [[pdf]](https://arxiv.org/pdf/2606.04197)</summary>

**Abstract:** How much should an LLM agent remember, and how should multi-agent systems be connected when trying to reach consensus? We show these two design choices interact in a way that flips the sign of memory's effect on coordination. Across 432 simulation runs of a networked Naming Game on eight fixed 16-agent topologies, we vary memory depth and network structure. Longer memory slows the time to reach steady state in decentralized networks but accelerates it in centralized ones; the same parameter pushes the system in opposite directions depending on topology. Critically, "faster settling" in centralized networks means locking in to a fragmented plateau more quickly, not reaching system-wide consensus, which can be used to generate diverging opinions. We further document a memory-mediated speed-unity trade-off: centralized networks consistently preserve more competing conventions than decentralized networks, but their settling speed depends sharply on memory. At the agent level, within-network analyses show that high-betweenness bridges suffer a brokerage penalty while agents in locally clustered neighborhoods achieve higher coordination success. Finally, in search of analytically tractable generative mechanisms, we find that agents' choices are well captured by Fictitious Play, indicating belief-based rather than reward-based adaptation. The practical implication: memory depth and communication topology should be co-designed, not optimized in isolation.

**arXiv ID:** 2606.04197
</details>

<details>
<summary><strong>Organizational Control Layer: Governance Infrastructure at the Execution Boundary of LLM Agent Systems</strong> - Tianyu Shi, Yang Mo, Yiou Liu, Zhuonan Hao, Yin Wang, Wenzhuo Hu, Nan Yu, Meng Zhou, Jiangbo Yu - [[pdf]](https://arxiv.org/pdf/2606.04306)</summary>

**Abstract:** LLM-based agents are increasingly deployed in workflows where generated outputs may directly trigger state-changing actions. This creates an execution-boundary problem: proposed actions must be governed before they are executed. We study this problem through economically consequential multi-agent interactions and argue that deployment-grade agent systems should separate proposal generation from environment-facing execution. To operationalize this principle, we introduce the Organizational Control Layer (OCL), a model-agnostic governance infrastructure that intercepts generated actions before execution through policy enforcement and escalation, without modifying the underlying LLM generator. We evaluate OCL on adversarial buyer--seller negotiation environments adapted from AgenticPay. Across multiple frontier LLM backends, OCL reduces unsafe executions from 88% to near-zero while increasing valid success from 12% to 96%. Results further reveal a safety--utility tradeoff: strict governance improves compliance and reliability against policy and constraint violations, but can reduce flexibility in tightly constrained markets. These findings suggest that deployment-grade LLM agent systems require explicit governance at the boundary between language generation and executable actions. The source code is available at: this https URL

**arXiv ID:** 2606.04306
</details>

<details>
<summary><strong>Channel Fracture: Architectural Blind Spots in Scheduled Cross-Agent Memory Injection for Multi-Agent Orchestration Systems</strong> - Levent Liu - [[pdf]](https://arxiv.org/pdf/2606.04896)</summary>

**Abstract:** Multi-agent AI orchestration systems increasingly rely on persistent memory to maintain context across sessions, agents, and tasks. When one agent must inject knowledge into another agent's memory -- a common requirement in hierarchical team architectures -- the delivery mechanism must be architecturally sound. We report the discovery of a systematic failure mode we term channel fracture: a condition where scheduled (cron) agents in orchestration frameworks are silently unable to write to the target agent's persistent memory due to hardcoded memory isolation guards. Through experiments on a production Hermes Agent deployment with five specialized profiles, we tested three injection channels: (A) direct SQLite database writes, (B) target-agent self-writes via memory tools, and (C) cron-delegated writes. Channel C failed completely due to two architectural constraints: skip_memory=True hardcoded at the scheduler layer and dynamic registration of memory tools contingent on _memory_manager initialization, which is bypassed in cron execution contexts. We propose CADVP (Cross-Agent Delivery Verification Protocol) v1.1, a 13-dimension verification framework with a veto-level channel confirmation check (CC-0) that prevents false-positive delivery assurance. We articulate two design principles: the inverse verification principle and the channel matching principle.

**arXiv ID:** 2606.04896
</details>

<details>
<summary><strong>Assistax: A Multi-Agent Hardware-Accelerated Reinforcement Learning Benchmark for Assistive Robotics</strong> - Leonard Hinckeldey, Elliot Fosong, Rimvydas Rubavicius, Elle Miller, Trevor McInroe, Fan Zhang, Patricia Wollstadt, Stefano V. Albrecht, Subramanian Ramamoorthy - [[pdf]](https://arxiv.org/pdf/2507.21638)</summary>

**Abstract:** The development of reinforcement learning (RL) algorithms has been largely driven by ambitious challenge tasks and benchmarks. Games have dominated RL benchmarks because they present relevant challenges, are inexpensive to run and easy to understand. While games such as Go and Atari have led to many breakthroughs, they often do not directly translate to real-world embodied applications. In recognising the need to diversify RL benchmarks and addressing complexities that arise in embodied interaction scenarios, we introduce Assistax: an open-source benchmark designed to address challenges arising in assistive robotics tasks. Assistax uses JAX's hardware acceleration for significant speed-ups for learning in physics-based simulations. In terms of open-loop wall-clock time, Assistax runs up to $370\times$ faster when vectorising training runs compared to CPU-based alternatives. Assistax conceptualises the interaction between an assistive robot and an active human patient using multi-agent RL to train a population of diverse partner agents against which an embodied robotic agent's zero-shot coordination capabilities can be tested. Extensive evaluation and hyperparameter tuning for popular continuous control RL and MARL algorithms provide reliable baselines and establish Assistax as a practical benchmark for advancing RL research for assistive robotics. The code is available at: this https URL.

**arXiv ID:** 2507.21638
</details>

<details>
<summary><strong>Token Budgets: An Empirical Catalog of 63 LLM-Agent Budget-Overrun Incidents, with an Affine-Typed Rust Mitigation as a Case Study</strong> - Sajjad Khan - [[pdf]](https://arxiv.org/pdf/2606.04056)</summary>

**Abstract:** LLM-agent budget overruns are a documented production failure class: a single retry loop can spend thousands of dollars before an operator notices, and the in-process integrity properties that would prevent it (no aliasing, no double-spend, no use-after-delegation of a cost-bearing value) are enforced, if at all, by ad-hoc wrappers rather than by the type system. Our central contribution is empirical: a catalog of 63 confirmed production incidents from 21 orchestration frameworks (2023-2026), each backed by a quoted GitHub issue and, where reported, a dollar loss, organized into an eight-cluster failure taxonomy (inter-rater Cohen's kappa = 0.837, N = 113), plus 47 supplementary structural entries. As one mitigation evaluated against this taxonomy, we build token-budgets, an 1,180-line Rust crate (no unsafe) that operationalizes affine ownership so that cloning, double-spending, or using a budget after delegating it are compile errors rather than runtime hazards an operator must remember to avoid. The dollar cap is runtime arithmetic under an estimator assumption; the affine layer makes that arithmetic non-bypassable. On single-agent workloads a 4-line Python counter matches the crate at 0/30 overshoot, so the distinguishing value is non-bypassability under operator error in multi-agent delegation: the delegation-fanout race documented in 11 incidents is rejected by the borrow checker at compile time, while the same pattern under asyncio overshoots 30/30 and three disciplined alternatives overshoot 0/30. Across five runtimes, three providers, and a temperature-stratified live-API test (N = 160), the approach reports zero cap violations and zero false refusals, at operational parity with concurrent work. Static over-reservation is 4-6x (2.11x adaptive). Binary-level cap-soundness on the running binary is left open.

**arXiv ID:** 2606.04056
</details>

<details>
<summary><strong>Solving Zebra Puzzles Using Constraint-Guided Multi-Agent Systems</strong> - Shmuel Berman, Kathleen McKeown, Baishakhi Ray - [[pdf]](https://arxiv.org/pdf/2407.03956)</summary>

**Abstract:** Prior research has enhanced the ability of Large Language Models (LLMs) to solve logic puzzles using techniques such as chain-of-thought prompting or introducing a symbolic representation. These frameworks are still usually insufficient to solve complicated logical problems, such as Zebra puzzles, due to the inherent complexity of translating natural language clues into logical statements. We introduce a multi-agent system, ZPS, that integrates LLMs with an off the shelf theorem prover. This system tackles the complex puzzle-solving task by breaking down the problem into smaller, manageable parts, generating SMT (Satisfiability Modulo Theories) code to solve them with a theorem prover, and using feedback between the agents to repeatedly improve their answers. We also introduce an automated grid puzzle grader to assess the correctness of our puzzle solutions and show that the automated grader is reliable by evaluating it in a user-study. Our approach shows improvement in all three LLMs we tested, with GPT-4 showing 166% improvement in the number of fully correct solutions.

**arXiv ID:** 2407.03956
</details>

<details>
<summary><strong>Multi-Agent Temporal Logic Planning via Penalty Functions and Block-Coordinate Optimization</strong> - Eleftherios E. Vlahakis, Arash Bahari Kordabad, Lars Lindemann, Pantelis Sopasakis, Sadegh Soudjani, Dimos V. Dimarogonas - [[pdf]](https://arxiv.org/pdf/2602.17434)</summary>

**Abstract:** Multi-agent planning under Signal Temporal Logic (STL) is often hindered by collaborative tasks that lead to computational challenges due to the inherent high dimensionality of the problem, preventing scalable synthesis with satisfaction guarantees. To address this, we formulate STL planning as an optimization program under multi-agent STL constraints and introduce a penalty-based unconstrained relaxation that can be efficiently solved via a Block-Coordinate Gradient Descent (BCGD) method, where each block corresponds to a single agent's decision variables, thereby mitigating complexity. By utilizing a quadratic penalty function defined via smooth STL semantics, we show that BCGD iterations converge to a stationary point of the penalized problem under standard regularity assumptions. To enforce feasibility, the BCGD solver is embedded within a two-layer optimization scheme: inner BCGD updates are performed for a fixed penalty parameter, which is then increased in an outer loop to progressively improve multi-agent STL robustness. The proposed framework enables scalable computations and is validated through various complex multi-robot planning scenarios.

**arXiv ID:** 2602.17434
</details>

<details>
<summary><strong>LifeSide: Benchmarking Agents as Lifelong Digital Companions</strong> - Yuqian Wu, Zhijie Deng, Wei Chen, Junwei Li, Yutian Jiang, Junle Chen, Zhengjun Huang, Qingxiang Liu, Jing Tang, Jiaheng Wei, Yuxuan Liang - [[pdf]](https://arxiv.org/pdf/2606.04660)</summary>

**Abstract:** Lifelong digital companions must integrate cross-session cues, continually update their understanding of users, and adapt to shifting privacy boundaries. Existing evaluations fail to capture this, testing memory recall and short-term empathy in isolation. To bridge this gap, we introduce \benchmark, a benchmark centered on multi-session \textit{Memory-Emotion-Environment} loops. By modeling users as persistent worlds with layered profiles and event trajectories, \benchmark uses multi-agent simulation to project environmental dynamics into dialogue, preserving the critical gap between latent thoughts and observable expressions. Evaluating 2,000 personas and 111K tasks across memory tracking, user understanding, privacy control, and emotional companionship, our experiment results reveal a stark reality: even models that saturate current memory benchmarks fail to sustain accurate user understanding and true companionship over long horizons.

**arXiv ID:** 2606.04660
</details>

<details>
<summary><strong>SMADE-IE: Sparse Multi-Agent Framework with Evidence-Driven Debate for Zero-Shot Information Extraction</strong> - Kenfeng Huang, Yi Cai, Xin Wu, Zikun Deng, Li Yuan - [[pdf]](https://arxiv.org/pdf/2606.04691)</summary>

**Abstract:** Zero-shot information extraction (IE) with large language models (LLMs) has attracted increasing attention due to its flexibility in adapting to new schemas and domains without task-specific training. Existing approaches mainly rely on monolithic prompting, each-type prompting, or multi-agent debate. However, monolithic prompting often suffers from boundary and type errors, while each-type prompting and multi-agent debate introduce cross-type conflicts, redundant agent interactions, and substantial token overhead. To address these challenges, we propose SMADE-IE, a sparse and evidence-driven multi-agent framework for zero-shot IE. SMADE-IE first employs an Adaptive Mode Selector to dynamically route inputs into either a lightweight Global Extraction Mode or a Type-Centric Extraction Mode, reducing unnecessary type selection and reasoning noise. For conflicting predictions, we further introduce an Evidence-Driven Debate mechanism that structures arguments into Toulmin-style components and performs confidence aggregation through external evidence scoring and Bayesian updates. Experimental results on 9 benchmark datasets across NER, RE, and JERE tasks show that SMADE-IE consistently outperforms existing zero-shot IE baselines while also improving token efficiency through sparse agent selection and early-stopping debate.

**arXiv ID:** 2606.04691
</details>

<details>
<summary><strong>GARL: Game-Theoretic Reinforcement Learning for Multi-Agent Strategic Prioritisation</strong> - Yuxiao Ye, Yiwen Zhang, Huiyuan Xie, Yuqin Huang, Zhiyuan Liu - [[pdf]](https://arxiv.org/pdf/2606.05002)</summary>

**Abstract:** LLM-based multi-agent systems are increasingly used for strategic decision-making tasks. In such settings, performance depends not only on individual model capabilities, but also on the policies by which agents interact and adapt. Multi-agent reinforcement learning can optimise these interaction policies, but its reward design often remains task-specific and weakly grounded in interaction structure. To address this gap, we propose GARL, a GAme-theoretic Reinforcement Learning framework for multi-agent strategic prioritisation. GARL formalises strategic prioritisation as a two-stage game: competing agents first allocate strategic resources over a shared candidate set, and a higher-level arbiter then produces the final ranking. The resulting game-theoretic utilities are converted into role-specific reinforcement signals, allowing policy optimisation to be guided by structured interaction. We instantiate GARL on issues-in-dispute ranking, where the goal is to prioritise core issues in legal proceedings. Experiments show that GARL improves ranking performance, enables small open-source LLMs to become competitive with a strong closed-source LLM under the same candidate-ranking setting, and yields gains in legal-domain competence and broader strategic decision-making. Overall, GARL demonstrates how game-theoretic interaction structure can be turned into reinforcement-learning objectives, providing a principled approach to policy optimisation in multi-agent strategic prioritisation.

**arXiv ID:** 2606.05002
</details>

<details>
<summary><strong>SocialCoach: Personalized Social Skill Learning with RL-based Agentic Tutoring and Practice</strong> - Tianfu Wang, Max Xiong, Jianxun Lian, Hongyuan Zhu, Zhengyu Hu, Yuxuan Lei, Linxiao Gong, Xiaofang Li, Peiting Tsai, Nicholas Jing Yuan, Qi Zhang - [[pdf]](https://arxiv.org/pdf/2606.04155)</summary>

**Abstract:** Social skills such as negotiation and leadership are crucial for personal and professional success in today's interconnected world. However, scalable and effective training remains a significant challenge due to the scarcity of expert coaching. In this paper, we introduce SocialCoach, a holistic LLM-powered agentic tutoring system for personalized social skill development at scale. First, SocialCoach automatically constructs a pedagogically-grounded, theory-to-practice knowledge corpus from diverse expert sources, leveraging a multi-agent pipeline. Second, to personalize the learning journey, it employs an adaptive practice scheduling module that follows a prescription-retrieval-adaptation process. To maximize the long-term learning experience while overcoming the cold-start problem, this policy is optimized within a learner simulation environment through reinforcement learning. Finally, SocialCoach integrates immersive, goal-driven practice, causality-driven proficiency assessment and knowledge-grounded, reflective tutoring to help address the knowing-doing gap. We deploy it in our product, EQoach, and conduct extensive experiments. The results show that SocialCoach improves simulated pathway quality and judge-rated tutoring quality over baseline approaches, while early user feedback indicates strong perceived engagement and usefulness. These findings suggest a practical architecture for personalized and gamified pedagogical platforms on soft skill learning.

**arXiv ID:** 2606.04155
</details>

<details>
<summary><strong>Demystifying Multi-Agent Debate: The Role of Confidence and Diversity</strong> - Xiaochen Zhu, Caiqi Zhang, Yizhou Chi, Tom Stafford, Nigel Collier, Andreas Vlachos - [[pdf]](https://arxiv.org/pdf/2601.19921)</summary>

**Abstract:** Multi-agent debate (MAD) is widely used to improve large language model (LLM) performance through test-time scaling, yet recent work shows that vanilla MAD often underperforms simple majority vote despite higher computational cost. Studies show that, under homogeneous agents and uniform belief updates, debate preserves expected correctness and therefore cannot reliably improve outcomes. Drawing on findings from human deliberation and collective decision-making, we identify two key mechanisms missing from vanilla MAD: (i) diversity of initial viewpoints and (ii) explicit, calibrated confidence communication. We propose two lightweight interventions. First, a diversity-aware initialisation that selects a more diverse pool of candidate answers, increasing the likelihood that a correct hypothesis is present at the start of debate. Second, a confidence-modulated debate protocol in which agents express calibrated confidence and condition their updates on others' confidence. We show theoretically that diversity-aware initialisation improves the prior probability of MAD success without changing the underlying update dynamics, while confidence-modulated updates enable debate to systematically drift to the correct hypothesis. Empirically, across six reasoning-oriented QA benchmarks, our methods consistently outperform vanilla MAD and majority vote. Our results connect human deliberation with LLM-based debate and demonstrate that simple, principled modifications can substantially enhance debate effectiveness.

**arXiv ID:** 2601.19921
</details>

<details>
<summary><strong>MemoNoveltyAgent: A Historical Research Memory-Aware Agent Workflow for Paper Novelty Assessment</strong> - Jiajun Hou, Hexuan Deng, Wenxiang Jiao, Xuebo Liu, Xiaopeng Ke, Derek F. Wong, Min Zhang - [[pdf]](https://arxiv.org/pdf/2603.20884)</summary>

**Abstract:** To alleviate the heavy burden of paper screening, researchers increasingly rely on existing AI agents, such as AI reviewers or DeepResearch, for paper evaluation and novelty assessment. However, lacking specialized mechanisms for processing scholarly literature, their analyses often produce superficial results with noticeable deficiencies in quality. To bridge this gap, we introduce MemoNoveltyAgent, a multi-agent system designed to generate comprehensive and faithful novelty reports. Beyond retrieving concrete prior-paper evidence via RAG, our system incorporates a high-level abstract memory constructed from large-scale scholarly corpora. This memory organizes research into hierarchical trees to distill field-specific evolutionary trajectories, thereby providing a broader historical context. Furthermore, we decompose papers into discrete novelty points for fine-grained analysis and retrieval, while employing a self-validation mechanism to improve report faithfulness. Finally, to address the evaluation challenges of such open-ended generation tasks, we propose a RAG-augmented checklist evaluation method that enables reliable and evidence-grounded assessments. Extensive experiments demonstrate that MemoNoveltyAgent outperforms GPT-5 DeepResearch by 13.69%. Code and demo are available at this https URL

**arXiv ID:** 2603.20884
</details>

<details>
<summary><strong>Towards Verifiable Multimodal Deep Research: A Multi-Agent Harness for Interleaved Report Generation</strong> - Chenghao Zhang, Guanting Dong, Yufan Liu, Tong Zhao, Xiaoxi Li, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2605.29861)</summary>

**Abstract:** Large Language Models (LLMs) have advanced autonomous agents from deep search, which retrieves concise factual answers, to deep research, which synthesizes scattered evidence into long-form reports. However, verifiable multimodal deep research remains challenging due to open-ended synthesis without deterministic ground truth and the need to interleave textual arguments with visual evidence. We propose Ptah, a multi-agent harness for interleaved report generation. Ptah orchestrates the lifecycle from user query to rendered web report through planning, research, and writing stages, where specialized agents construct visual-aware plans, collect claim-grounded evidence, maintain source-aligned images in a Visual Working Memory, and compose reports through declarative multimodal tool use. A verifier agent serves as the harness's acceptance function, enforcing factual grounding, citation fidelity, and cross-modal consistency throughout the workflow. We further introduce PtahEval, an evaluation protocol that augments existing benchmarks with image-level and presentation-level assessments. Experiments on deep research benchmarks show that Ptah produces more reliable, visually informative, and usable human-facing multimodal reports than strong baselines. Our code is released at this https URL

**arXiv ID:** 2605.29861
</details>

<details>
<summary><strong>Extending AI for Research to the Humanities: A Multi-Agent Framework for Evidence-Grounded Scholarship</strong> - Yating Pan, Jiajun Zhang, Jun Wang, Qi Su - [[pdf]](https://arxiv.org/pdf/2605.30947)</summary>

**Abstract:** LLM-based research agents have advanced rapidly in science and engineering, where research is organized around executable experiments, code, and quantitative signals. Humanities scholarship, however, requires a different mode of reasoning: interpretive, evidence-grounded argument over primary sources, where scholarly value depends on faithful quotation, verifiable provenance, and close reading. Existing research agents remain largely optimized for execution and retrieval, not evidence-grounded interpretive reasoning. To address this gap, we introduce SPIRE (Scholarly-Primitives-Inspired Research Engine), a multi-agent framework for evidence-grounded humanities scholarship. Drawing on Scholarly Primitives theory, SPIRE casts recurring humanities operations as cooperating agent roles (source discovery, evidence annotation, comparison, provenance checking, sampling, citation binding, and argumentative synthesis) over a multi-scale close-reading substrate of passages, intra-context graph communities, and cross-context semantic clusters. On a peer-reviewed-paper benchmark over classical Chinese and Greco-Roman Latin scholarship, SPIRE recovers cited primary-source evidence more reliably than Naive LLM, Text RAG, and GraphRAG, and receives higher blind-judge scores on answer accuracy, depth, coverage, and evidence quality. Ablations show that both the scholarly-operation agents and close-reading retrieval contribute to evidence-grounded essays. Code, data catalogues, and reproduction scripts are released at this https URL.

**arXiv ID:** 2605.30947
</details>

<details>
<summary><strong>Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs</strong> - Jinbo Liu, Defu Cao, Yifei Wei, Tianyao Su, Yuan Liang, Yushun Dong, Yan Liu, Yue Zhao, Xiyang Hu - [[pdf]](https://arxiv.org/pdf/2512.04668)</summary>

**Abstract:** Graph topology is a fundamental determinant of memory leakage in multi-agent LLM systems, yet its effects remain poorly quantified. We introduce MAMA (Multi-Agent Memory Attack), a controlled evaluation framework for comparing topology-conditioned memory leakage in multi-agent LLM systems. MAMA operates on synthetic documents containing labeled Personally Identifiable Information (PII) entities, from which we generate sanitized task instructions. We execute a two-phase protocol: Engram (seeding private information into a target agent's memory) and Resonance (multi-round interaction where an attacker attempts extraction). Over 10 rounds, we measure leakage using a two-stage recovery criterion that combines exact-match extraction with LLM-based inference over the attacker's final output. We evaluate six canonical topologies (complete, circle, chain, tree, star, star-ring) across $n\in\{4,5,6\}$, attacker-target placements, and base models. Results are consistent: denser connectivity, shorter attacker-target distance, and higher target centrality increase leakage; most leakage occurs in early rounds and then plateaus; model choice shifts absolute rates but preserves broad structural trends; spatiotemporal/location attributes leak more readily than identity credentials or regulated identifiers. We distill practical guidance for system design: favor sparse or hierarchical connectivity, maximize attacker-target separation, and restrict hub/shortcut pathways via topology-aware access control. Our code is available at this https URL.

**arXiv ID:** 2512.04668
</details>

<details>
<summary><strong>Episodic Memory Temporal Consistency for Cooperative Multi-Agent Reinforcement Learning</strong> - Zicheng Zhao, Yu Lan, Chengzhengxu Li, Zhaohan Zhang, Xiaoming Liu - [[pdf]](https://arxiv.org/pdf/2606.04492)</summary>

**Abstract:** Cooperative Multi-Agent Reinforcement Learning (MARL) frequently suffers from severe reward sparsity and exploration bottlenecks. While episodic memory mechanisms mitigate these issues by reusing high-return trajectories, they often trap agents in local optima due to unconstrained incentive distribution and semantic representation collapse. To address this, we propose Episodic Memory Temporal Consistency (EMTC), a framework that robustly constructs and selectively leverages historical experiences. EMTC introduces two synergistic components: (1) a Temporally Consistent Semantic Embedder that integrates contrastive learning with time-conditioned state reconstruction, preventing representation collapse and enabling precise memory retrieval; and (2) a Temporal Consistency Gating Mechanism that dynamically modulates episodic incentives based on temporal consistency error. This adaptive gate filters misleading signals from pseudo-successful trajectories, effectively mitigating Q-value overestimation. We provide theoretical guarantees, establishing a strict error bound that directly links the observable temporal consistency error to the underlying trajectory optimality and representation quality. Extensive evaluations on the SMAC and GRF benchmarks demonstrate that EMTC consistently outperforms state-of-the-art baselines. Notably, compared to the strongest episodic baseline, EMTC achieves absolute win-rate improvements of up to 24% in super-hard SMAC scenarios and an average improvement of 28% across GRF tasks.

**arXiv ID:** 2606.04492
</details>

<details>
<summary><strong>Enhancing the MADDPG Algorithm for Multi-Agent Learning via Action Inference and Importance Sampling</strong> - Marc Walden, Jason Liu, Shaashwath Sivakumar, Ryan Liu, Hamza Khan - [[pdf]](https://arxiv.org/pdf/2606.05021)</summary>

**Abstract:** We investigate multi-agent deep reinforcement learning and propose two enhancements to the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm. First, we introduce a novel Action Inference mechanism that enables each agent to predict other agents' intended actions, thereby improving the accuracy and stability of its own policy. Second, we apply an importance sampling strategy, using geometric distribution, in the replay buffer to prioritize more recent and informative experiences, which helps mitigate the non-stationarity inherent in multi-agent environments. We evaluate both modifications on the discrete-action Predator-Prey task provided by the PettingZoo library, a flexible Python interface for general multi-agent reinforcement learning benchmarks. Our results indicate that Action Inference is effective in improving learning stability and inter-agent cooperation and that importance sampling using geometric distribution can lead to significant improvements in exploration efficiency over standard MADDPG. Code available at this https URL

**arXiv ID:** 2606.05021
</details>

<details>
<summary><strong>Scaling Datasets for Multi-Sensor, Multi-Agent, and Multi-Domain Learning in Autonomous Systems</strong> - R. Spencer Hallyburton, David Hunt, Miroslav Pajic - [[pdf]](https://arxiv.org/pdf/2606.04444)</summary>

**Abstract:** Existing datasets cannot support large-scale learning in multi-agent, multi-sensor, or multi-domain autonomy, where diversity and coordination are essential. We present a modular dataset generation pipeline that creates terabyte-scale, ground-truth-labeled data for ground, aerial, and infrastructure-based systems using the AVstack framework and CARLA simulator. Supporting single- and multi-agent configurations with flexible sensor suites, the pipeline enables controllable experimentation across challenging conditions. Representative perception and fusion studies show how generated data can support application-specific training and collaborative autonomy.

**arXiv ID:** 2606.04444
</details>

<details>
<summary><strong>Multi-Agent Next-Best-View Optimization for Risk-Averse Planning</strong> - Amirhossein Mollaei Khass, Vivek Pandey, Guangyi Liu, Athanasios Cosse, Emrah Bayrak, Nader Motee - [[pdf]](https://arxiv.org/pdf/2606.04158)</summary>

**Abstract:** Multi-agent Next-Best-View (NBV) selection for safe path planning in uncertain and unknown environments requires informative, safety-aware, and efficient coordination. Centralized approaches rely on sharing raw sensor data or significant communication overhead, resulting in limited scalability. We propose a distributed, risk-aware multi-agent NBV framework in which each robot maintains a private local 3D Gaussian Splatting map and the team jointly maximizes expected information gain (EIG) restricted to masked zones along planned trajectories. The resulting distributed objective is solved by Consensus ADMM (C-ADMM) over a communication graph, with each robot exchanging only candidate viewpoints, planned trajectory descriptors, and scalar EIG contributions. Collision risk along each trajectory is modeled via Average Value-at-Risk (AV@R) over the local 3DGS map and used both to shape the masking radius and to score planned paths. Experiments in Gibson environments at multiple team sizes show that the distributed formulation approaches the centralized baseline in mapping quality and trajectory safety while reducing communication by orders of magnitude.

**arXiv ID:** 2606.04158
</details>

</details>

<details open>
<summary><h2>Other Agent Research (14 papers)</h2></summary>

<details>
<summary><strong>Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieval</strong> - Jiaxi Li, Ke Deng, Yun Wang, Jingyuan Huang, Yucheng Shi, Qiaoyu Tan, Jin Lu, Ninghao Liu - [[pdf]](https://arxiv.org/pdf/2606.04391)</summary>

**Abstract:** Language agents increasingly rely on reusable skills to improve multi-step web automation across related tasks. A growing line of work studies online skill learning, where agents continually induce skills from previous task trajectories and reuse them in future tasks on the fly. However, existing methods mainly reuse skills at the task-level: a fixed set of skills is retrieved based on the initial task instruction and then held fixed throughout execution. This static strategy is misaligned with web execution, where the appropriate next action depends not only on the task goal but also on the current webpage state, which often transitions into situations that the initial skills fail to cover. To address this gap, we propose State-Grounded Dynamic Retrieval (SGDR), an online skill learning method that enables stepwise skill reuse for web agents. SGDR consists of three components: a sliding-window extraction process that turns completed trajectories into reusable sub-procedures invokable at intermediate execution states, a dual text-code representation that connects skill retrieval with executable action, and a state-grounded dynamic retrieval mechanism that matches skills to both the task goal and the current webpage state. Experiments on WebArena across five domains show that SGDR consistently outperforms strong baselines, achieving average success rates of 37.5% with GPT-4.1 and 24.3% with Qwen3-4B, corresponding to relative gains of 10.6% and 10.0% over the strongest baseline, respectively. The code is available at this https URL.

**arXiv ID:** 2606.04391
</details>

<details>
<summary><strong>MIRAGE: Mobile Agents with Implicit Reasoning and Generative World Models</strong> - Zhichao Yang, Yuanze Hu, Haojie Hao, Longkun Hao, Dongshuo Huang, Hongyu Lin, Gen Li, Lanqing Hong, Yihang Lou, Yan Bai - [[pdf]](https://arxiv.org/pdf/2606.04627)</summary>

**Abstract:** Mobile agents are increasingly expected to operate everyday applications from screenshots and language goals, where reliable control requires reasoning over screen affordances, multi-step navigation, and future state changes. However, many agents externalize this computation as long textual chains of thought, which slows interaction, increases supervision cost, and complicates deployment. We introduce MIRAGE, a framework that learns continuous latent reasoning representations from visible textual reasoning traces. MIRAGE transfers explicit reasoning into compact hidden states, enabling the agent to reason internally without decoding long rationales. It also incorporates a generative world-model objective: latent reasoning vectors are aligned with future screenshots, encouraging the agent to anticipate upcoming interface states before acting. This turns hidden computation into both a compressed thought representation and a forward-looking model of environment dynamics. At inference time, MIRAGE reasons in continuous latent space, reducing token generation while improving execution efficiency. On AndroidWorld, MIRAGE matches explicit chain-of-thought supervised fine-tuning in the 4B ablation with a 3-5x lower decoded-token budget and improves a comparable instruction-tuned baseline by 10.2 points; on AndroidControl, it improves action grounding while generating over 75% fewer tokens.

**arXiv ID:** 2606.04627
</details>

<details>
<summary><strong>Strabo: Declarative Specification and Implementation of Agentic Interaction Protocols</strong> - Samuel H. Christie V, Amit K. Chopra, Munindar P. Singh - [[pdf]](https://arxiv.org/pdf/2606.05043)</summary>

**Abstract:** The last few years have witnessed major advances in the modeling and implementation of multiagent systems based on declarative interaction protocols. Our contribution, Strabo, establishes the relevance of these advances to ongoing industry efforts in Agentic AI. Specifically, we consider UCP, the Universal Commerce Protocol, a recent Google-led effort to standardize e-commerce interactions for AI agents. Our exercise is in two parts. One, we model the part of UCP dealing with checkouts as a declarative Langshaw protocol and implement agents using Peach, a programming model for Langshaw. This part of the exercise brings out the advantages of formal, declarative specifications. Two, we show that Peach agents can interoperate with UCP agents implemented by Google, thereby establishing the fidelity of our approach with respect to UCP. Such interoperation enables the incremental introduction of declarative protocols and agents into a conventional setting, indicating a pathway by which EMAS ideas could influence practice without demanding a wholesale update.

**arXiv ID:** 2606.05043
</details>

<details>
<summary><strong>AgenticDiffusion: Agentic Diffusion-based Path Planning for Vision-Based UAV Navigation</strong> - Faryal Batool, Muhammad Ahsan Mustafa, Fawad Mehboob, Valerii Serpiva, Dzmitry Tsetserukou - [[pdf]](https://arxiv.org/pdf/2606.04111)</summary>

**Abstract:** Indoor UAV navigation requires efficient exploration, scene understanding, and reliable trajectory execution under limited field-of-view observations. Existing vision-based navigation frameworks typically rely on single-view observations, limiting their ability to reason about occlusions, target visibility, and global scene structure. In this work, we propose AgenticDiffusion, a multi-view UAV navigation framework that coordinates language-guided reasoning, open-vocabulary target grounding, vision-based diffusion planning, and NMPC within a unified aerial navigation pipeline. Given a natural language instruction and synchronized first-person-view (FPV) and top-view observations, the framework determines the most informative viewpoint for navigation and generates a mission plan prior to trajectory execution. The targets are localized using an open-vocabulary grounding model, after which viewpoint-specific diffusion planners generate navigation trajectories for UAV execution. Using complementary viewpoints, the proposed framework reduces repeated target exploration and improves navigation efficiency in cluttered indoor environments. The framework was validated in four real-world UAV navigation scenarios involving adaptive viewpoint selection, multi-stage mission execution, long-horizon navigation, and safe landing-site selection. The experimental results demonstrated an overall mission success rate of 80% in 40 real-world trials, while the diffusion planners achieved a trajectory generation success rate of 100%.

**arXiv ID:** 2606.04111
</details>

<details>
<summary><strong>TITAN-FedAnil+: Trust-Based Adaptive Blockchain Federated Learning for Resource-Constrained Intelligent Enterprises</strong> - Muhammad Hadi, Muhammad Jahangir, Talha Shafique, Muhammad Khuram Shahzad - [[pdf]](https://arxiv.org/pdf/2606.04388)</summary>

**Abstract:** Federated Learning (FL) has emerged as an effective paradigm for collaborative intelligence while preserving data privacy. However, data heterogeneity arising from non-IID distributions and decentralized security threats remain significant challenges, particularly in resource-constrained enterprise environments. This paper presents TITAN-FedAnil+, a Trust-Based Adaptive Network for blockchain-enabled federated learning in intelligent enterprises. The proposed framework introduces affinity propagation-based adaptive clustered aggregation to identify and filter malicious updates without requiring prior knowledge of the number of attackers. In addition, GPU-accelerated vectorization is employed to improve computational efficiency, while a signed state jump mechanism enables lightweight blockchain resynchronization. Experimental results demonstrate substantial reductions in memory overhead, achieving up to 81% savings across 50 communication rounds on constrained 8 GB edge devices compared with the baseline framework. The results indicate that TITAN-FedAnil+ effectively improves robustness, scalability, and resource efficiency for secure federated learning deployments in intelligent enterprise environments.

**arXiv ID:** 2606.04388
</details>

<details>
<summary><strong>Multi-SPIN: Multi-Access Speculative Inference for Cooperative Token Generation at the Edge</strong> - Haotian Zheng, Zhanwei Wang, Mingyao Cui, Chang Cai, Hongyang Du, Kaibin Huang - [[pdf]](https://arxiv.org/pdf/2606.04581)</summary>

**Abstract:** Speculative inference (SPIN) was originally developed as an efficient architecture to accelerate Large Language Models (LLMs). In this work, we propose its distributed deployment to enable cooperative token generation in a multiuser edge system; its advantage is to effectively balance computational loads between resource-constrained devices and servers. The resulting architecture, termed Multi-access SPIN (Multi-SPIN), utilizes on-device small language models to generate and upload candidate token drafts, while an edge server operates the LLM to verify them in parallel batches. Given the severe heterogeneity in users' computation and communication capabilities, the draft length emerges as a critical control variable that influences node-level computation loads and multi-access latency, thereby governing the sum token goodput. Consequently, considering frequency-division multiple access, we investigate the problem of multi-access draft control, a joint optimization of draft-length control and bandwidth allocation to maximize sum token goodput. We examine two cases: (1) homogeneous draft lengths across users to facilitate server-side batching, and (2) heterogeneous draft lengths to introduce a new dimension for goodput enhancement. By developing decomposition methods, we reduce these complex optimizations into tractable sub-problems, which allow efficient draft control algorithms to be derived in closed form. Our analysis shows that the optimal bandwidth allocation compensates users with weaker computation-and-communication capabilities in the homogeneous case due to the batching synchronization requirements, whereas its heterogeneous-case counterpart rewards users with higher acceptance rates by relaxing such requirements. Experiments using Llama-2 and Qwen3.5 model pairs across diverse tasks demonstrate that Multi-SPIN improves goodput by up to 88% over heterogeneity-agnostic baselines.

**arXiv ID:** 2606.04581
</details>

<details>
<summary><strong>DAR: Deontic Reasoning with Agentic Harnesses</strong> - Guangyao Dou, William Jurayj, Nils Holzenberger, Benjamin Van Durme - [[pdf]](https://arxiv.org/pdf/2606.05009)</summary>

**Abstract:** Deontic reasoning is the task of answering questions by applying explicit rules and policies to case-specific facts, for example computing tax liability under a statute or determining the outcome of an immigration appeal. A key technical challenge for LLM-based deontic reasoning is that the relevant ruleset can be long and cross-referenced, so models may still fail to locate the rules needed for a particular reasoning step. We introduce Deontic Agentic Reasoning (DAR), an agentic reasoning setup in which the model interacts with the statutes on demand. We evaluate DAR under multiple harnesses on hard subsets of DeonticBench. Across these settings, we find that agentic harnesses can push the frontier on deontic reasoning tasks, but improvements are not uniform: weaker models often degrade on numerical tasks while consuming far more tokens.

**arXiv ID:** 2606.05009
</details>

<details>
<summary><strong>The Accountability Horizon: An Impossibility Theorem for Governing Human-Agent Collectives</strong> - Haileleol Tibebu, Hewan Shemtaga - [[pdf]](https://arxiv.org/pdf/2604.07778)</summary>

**Abstract:** Existing accountability frameworks for AI systems, legal, ethical, and regulatory, rest on a shared assumption: for any consequential outcome, at least one identifiable person had enough involvement and foresight to bear meaningful responsibility. This paper proves that agentic AI systems violate this assumption not as an engineering limitation but as a mathematical necessity once autonomy exceeds a computable threshold. We introduce Human-Agent Collectives, a formalisation of joint human-AI systems where agents are modelled as state-policy tuples within a shared structural causal model. Autonomy is characterised through a four-dimensional information-theoretic profile (epistemic, executive, evaluative, social); collective behaviour through interaction graphs and joint action spaces. We axiomatise legitimate accountability through four minimal properties: Attributability (responsibility requires causal contribution), Foreseeability Bound (responsibility cannot exceed predictive capacity), Non-Vacuity (at least one agent bears non-trivial responsibility), and Completeness (all responsibility must be fully allocated). Our central result, the Accountability Incompleteness Theorem, proves that for any collective whose compound autonomy exceeds the Accountability Horizon and whose interaction graph contains a human-AI feedback cycle, no framework can satisfy all four properties simultaneously. The impossibility is structural: transparency, audits, and oversight cannot resolve it without reducing autonomy. Below the threshold, legitimate frameworks exist, establishing a sharp phase transition. Experiments on 3,000 synthetic collectives confirm all predictions with zero violations. This is the first impossibility result in AI governance, establishing a formal boundary below which current paradigms remain valid and above which distributed accountability mechanisms become necessary.

**arXiv ID:** 2604.07778
</details>

<details>
<summary><strong>RAMPART: Registry-based Agentic Memory with Priority-Aware Runtime Transformation</strong> - Nikodem Tomczak - [[pdf]](https://arxiv.org/pdf/2606.04628)</summary>

**Abstract:** RAMPART is a compile-time memory model and pure in-RAM block registry for LLM-based agents. Context assembly is a programmable runtime operation where content is compiled from a structured registry under explicit policy for ordering, inclusion, and eviction. Five composable primitives (promote, gate, write, evict, rollback) act on named addressable blocks before compilation at zero prompt-token cost. Provenance tags and non-evictable authorship flags implement a permissioned memory model with block-level ownership. Controlled probes with Qwen3-8B Q4 show that compile-time placement and the structural relationship between blocks and the task query affect task success, with the cliff falling at roughly the seventh block position when the task follows the registry and the twelfth when it precedes. Grouping the critical block with content-adjacent neighbours and promoting the group as a unit lifts task success by tens of percentage points at positions where single-block placement fails. Cross-model replication on Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3, and Qwen3-14B shows the content-priming effect appears at the same absolute positions across families, with magnitude varying with model strength. Block grouping raises Mistral's mean pass rate roughly fivefold at the hardest registry size, and a smaller model with the intervention can outperform a larger model without it in the mid-registry zone. Relevance gating reduces prompt cost by 67.8\% while recovering 83% of the promoted-condition success rate. Schema eviction produces 0% invocations against 100% with the schema present, a property policy-based approaches cannot guarantee by construction. Shared-registry coordination reduces inter-agent communication to a method call at zero coordination token cost.

**arXiv ID:** 2606.04628
</details>

<details>
<summary><strong>OpenAgenet/OAN: Open Infrastructure for Trusted Agent Interconnection</strong> - Jinliang Xu - [[pdf]](https://arxiv.org/pdf/2606.03161)</summary>

**Abstract:** OpenAgenet, abbreviated as OAN, is an open infrastructure project for trusted Agent interconnection. It addresses a problem that becomes visible when Agents move from isolated applications into open, multi-operator networks: before an Agent can safely discover, select, and invoke another Agent, it needs a way to verify identity provenance, governance state, discovery authorization, freshness, and pre-connection trust evidence. OAN is designed as a protocol-neutral trust layer. It does not replace Agent interaction protocols, tool protocols, model orchestration frameworks, or application-level workflows. Instead, it provides Root-governed identity admission, Registrar-assisted onboarding, Root-verified package publication, authorization-aware Discovery, and signed trusted invocation. This paper presents the motivation, architecture, roles, governance model, relationship with MCP, A2A, and ANP, deployment patterns, cooperation model, blockchain-backed authorization bulletin, prototype status, performance profile, and roadmap of OAN.

**arXiv ID:** 2606.03161
</details>

<details>
<summary><strong>Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs</strong> - Xinyu Pang, Zhanke Zhou, Xuan Li, Fangrui Lv, Shanshan Wei, Sen Cui, Bo Han, Changshui Zhang - [[pdf]](https://arxiv.org/pdf/2606.04360)</summary>

**Abstract:** Symbolic regression (SR) discovers compact mathematical expressions from data, yet recent LLM-based evolutionary methods remain sample-inefficient because they rely mainly on scalar feedback such as MSE. We identify a core limitation: existing methods conflate candidate proposal with search guidance, requiring the LLM to infer how to evolve an expression, diagnose its errors, and reuse past experience from a single score. To address this, we propose Deliberate Evolution (DE), an agentic framework that decouples symbolic generation from search control. DE guides LLM proposals with adaptive operators for search direction, analytical tools for structural diagnosis, and reflective memory for trajectory-level experience. Experiments on LLM-SRBench show that DE consistently outperforms representative LLM-based SR baselines across diverse scientific domains while using only 40% of the standard sample budget.

**arXiv ID:** 2606.04360
</details>

<details>
<summary><strong>Optimizing the Cost-Quality Tradeoff of Agentic Theorem Provers in Lean</strong> - Kári Rögnvaldsson, Chenhao Sun, Jasper Dekoninck, Martin Vechev - [[pdf]](https://arxiv.org/pdf/2606.04883)</summary>

**Abstract:** Large language models (LLMs) are increasingly used in workflows for generating formal proofs in Lean. These workflows often decompose problems into smaller lemmas, sample many proof attempts, and use compiler feedback to guide search. However, they can be prohibitively expensive, often spending substantial compute on attempts that ultimately fail. In this work, we address this problem with an action routing agent that consists of a data plane and a control plane. The data plane generates natural-language lemma decompositions, formalizes them in Lean, and samples proof attempts for the resulting theorem and lemma targets. The control plane observes previous failed Lean attempts, estimates both the likelihood of success and cost of another attempt, and decides whether to continue proving the current target or restart from a new breakdown. On a subset of PutnamBench, our agent decreases the cost by $25.8\%$ over a fixed-step baseline on average, preserving performance while using substantially less compute. These results suggest that failed Lean trajectories provide actionable signals for cost-aware resource allocation in agentic theorem proving.

**arXiv ID:** 2606.04883
</details>

<details>
<summary><strong>Cooperative Circumnavigation for Multiple Unmanned Surface Vehicles Without External Localization</strong> - Xueming Liu, Lin Li, Xiang Zhou, Tianjiang Hu, Qingrui Zhang - [[pdf]](https://arxiv.org/pdf/2606.04518)</summary>

**Abstract:** This paper proposes a cooperative target circumnavigation framework for multiple unmanned surface vehicles (USVs) operating without external localization. The objective is to maintain a uniform circular formation of a specified radius around a target using only limited onboard sensing. The framework adopts a heterogeneous perception strategy that distinguishes between the asymmetric sensing relationships with the target and among the USVs. Specifically, the USVs obtain relative range and displacement measurements through active perception and inter-vehicle communication, while bearing measurements to a non-cooperative target are acquired via passive sensors. To estimate relative positions--both among USVs and between each USV and the target--we employ a Maximum Correntropy Kalman Filter and a Pseudo-Linear Kalman Filter, respectively. A coupled oscillator-based formation controller is designed to ensure system observability while achieving circumnavigation. Theoretical analysis demonstrates that the controller ensures the relative motions between the USVs, as well as that between each USV and the target, satisfy the persistent excitation condition, thereby guaranteeing observability of the Kalman-based filters. The effectiveness of the proposed approach is validated through numerical simulations.

**arXiv ID:** 2606.04518
</details>

<details>
<summary><strong>Revisiting Embodied Chain-of-Thought for Generalizable Robot Manipulation</strong> - Nan Sun, Yuan Zhang, Yongkun Yang, Wentao Zhao, Peiyan Li, Jun Guo, Wenxuan Song, Pengxiang Ding, Runze Suo, Yifei Su, Xin Xiao, Xinghang Li, Huaping Liu - [[pdf]](https://arxiv.org/pdf/2606.03784)</summary>

**Abstract:** Embodied chain-of-thought (CoT) aims to bridge linguistic reasoning and robotic control, but its effective form and integration strategy remain underexplored. In this paper, we revisit embodied CoT for vision-language-action (VLA) models at large scale. We construct the largest embodied CoT corpus to date, comprising 978,743 trajectories, 226.3M samples, and 2592.5 hours of robot data. Through extensive experiments, we find that effective embodied CoT should ground high-level semantic understanding into concrete action guidance, such as end-effector movement descriptions and image-space trajectories, while high-level reasoning alone brings only marginal gains. We further show that explicit CoT does not scale reliably when used as an autoregressive action prefix, as it suffers from compounding inference errors and unstable reasoning-action coupling. To address these limitations, we propose ERVLA, a VLA model that uses embodied CoT as representation-shaping supervision rather than mandatory test-time reasoning. ERVLA is trained with a reasoning-dropout strategy, enabling the model to absorb rich reasoning traces during training while predicting actions directly without CoT decoding during inference. This design improves scalability with increasing pre-training data and avoids autoregressive instability. ERVLA achieves state-of-the-art performance on LIBERO-Plus with an 86.9% success rate and reaches 53.2% success rate on VLABench, demonstrating strong out-of-distribution generalization. In real-robot experiments, ERVLA further outperforms competitive state-of-the-art baselines, especially on tasks requiring semantic disambiguation and long-horizon execution.

**arXiv ID:** 2606.03784
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>Beyond Prompt-Based Planning: MCP-Native Graph Planning-based Biomedical Agent System</strong> - Zhangtianyi Chen, Florensia Widjaja, Wufei Dai, Xiangjun Zhang, Yuhao Shen, Juexiao Zhou - [[pdf]](https://arxiv.org/pdf/2606.04494)</summary>

**Abstract:** Biomedical agents promise to automate complex biological workflows, yet current systems face two fundamental bottlenecks: bioinformatics tools are highly heterogeneous in interfaces and execution environments, while agent planning still relies on flat prompt-retrieved tool descriptions. As biomedical software ecosystems grow, this coupling between tool coverage and context size leads to tool confusion, unstable planning, and inefficient execution. We introduce BioManus, an MCP-native biomedical agent built on graph-scaffolded planning over structured biological capabilities. BioManus first introduces the BioinfoMCP Compiler, which converts heterogeneous bioinformatics software into standardized MCP servers, yielding a large executable MCP ecosystem. It then organizes this ecosystem as a typed heterogeneous MCP graph over tools, operations, datatypes, and workflow stages. At inference time, BioManus retrieves compact task-specific subgraphs, synthesizes operation-level workflow scaffolds. This design decouples planning complexity from raw tool inventory size, achieving a context compression ratio of Theta(N / (h * m_bar)) under high-recall retrieval, where N is the total tool count, h is the workflow horizon, and m_bar (much smaller than N) is the average number of candidate tools per operation. Experiments on BioAgentBench and LAB-Bench show that BioManus improves execution accuracy, workflow validity, and context efficiency over advanced biomedical agent baselines. This work suggests a paradigm shift: scalable biomedical reasoning requires structured executable capability graphs rather than increasingly larger prompt-level tool retrieval.

**arXiv ID:** 2606.04494
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (39 papers)</h2></summary>

<details>
<summary><strong>The Digital Apprentice: A Framework for Human-Directed Agentic AI Development</strong> - Travis Weber, Rohit Taneja - [[pdf]](https://arxiv.org/pdf/2606.04321)</summary>

**Abstract:** Agentic AI deployments face a recurring design tension: heavy human oversight limits scale, while broad autonomy outruns accountability. Neither posture provides the governance infrastructure required for responsible delegation. We present the Digital Apprentice, a framework for scalable, safe AI agency in which autonomy is earned, not assumed. The Digital Apprentice is a developmental learner that internalizes the tacit methodology of a directing human, graduating through per-skill autonomy tiers only when empirical evidence justifies it. The result is an agent that becomes genuinely useful over time while remaining aligned to a specific human's standards. Three architectural components make this possible. (1) Methodology capture, distilling a directing professional's tacit approach into structured assets. (2) Authorization, with autonomy escalation gated by explicit human approval. (3) Continuous alignment, correcting drift at runtime and converting each correction into owned preference data. We instantiate this framework as an inference-time control plane. We mathematically model the quality framework and discuss policies and techniques designed to raise quality. We apply the framework to an open professional corpus, and we show how catching data drift and applying a different technique at runtime recovers degraded quality dimensions under traffic shift. The implication extends beyond any single application. We believe these three pillars, stitched together as a system, form a safer and more viable path to agentic systems that can scale without sacrificing trust.

**arXiv ID:** 2606.04321
</details>

<details>
<summary><strong>MapAgent: An Industrial-Grade Agentic Framework for City-scale Lane-level Map Generation</strong> - Deguo Xia, Zihan Li, Haochen Zhao, Dong Xie, Yuyao Kong, Xiyan Liu, Jizhou Huang, Mengmeng Yang, Diange Yang - [[pdf]](https://arxiv.org/pdf/2606.04513)</summary>

**Abstract:** Lane-level maps are critical infrastructure for autonomous driving and lane-level navigation, yet constructing and maintaining standardized lane networks for hundreds of cities remains highly labor-intensive. Recent end-to-end vectorized mapping methods can predict lane geometry and topology directly from sensor data, but they typically treat mapping specifications and traffic regulations as implicit, dataset-dependent supervision. Moreover, in complex scenes (e.g., worn or missing markings and occlusions), correct lane configurations are often under-determined by visual evidence alone, making specification violations a major source of human post-editing. We propose MapAgent, an industrial-grade agentic architecture that augments a vectorization backbone for specification-compliant lane-map production. Rather than merely adding an agent loop to map prediction, MapAgent couples backbone perception with explicit specification verification, constraint-aware reasoning, and deterministic map editing under a bounded, verification-driven Judge-Planner-Worker loop. A vision-language Judge diagnoses errors by jointly inspecting visual evidence and draft vectors, while a tool-calling Planner generates minimal corrective edits with post-edit re-validation. To remain scalable for city-scale production, MapAgent is selectively triggered only on tiles with low backbone confidence, adding modest overhead while preserving throughput. Experiments on real-world datasets show consistent gains over strong production baselines, especially in complex and long-tail scenarios. Additionally, MapAgent has been integrated into Baidu Maps, supporting lane-level map generation for over 360 cities nationwide and elevating the overall production automation to over 95%, demonstrating MapAgent's practicality and effectiveness for large-scale lane-level map generation.

**arXiv ID:** 2606.04513
</details>

<details>
<summary><strong>Neetyabhas: A Framework for Uncertainty-Aware Public Policy Optimization in Rational Agent-Based Models</strong> - Janani Venugopalan, Gaurav Deshkar, Rishabh Gaur, Harshal Hayatnagarkar, Jayanta Kshirsagar - [[pdf]](https://arxiv.org/pdf/2606.04562)</summary>

**Abstract:** Purpose The WHO's COVID-19 non-pharmaceutical interventions (e.g., lockdowns, vaccinations) effectively curb transmission but impose heavy economic strains. Existing research often neglects individual behaviors and falsely assumes perfect infection tracking and flawless policy execution, failing to account for real-world uncertainties and errors. Methods We propose an integrative approach incorporating uncertainties in both epidemic measurement (infections/hospitalizations) and policy implementation. We built a simulation model of 1,000 individuals making real-time choices regarding mask-wearing, vaccination, and shopping. Concurrently, policymakers deploy interventions (lockdowns, mandates) based on health and economic observations. This framework is driven by hierarchical reinforcement learning agents, utilizing deep Q-networks alongside uncertainty-aware policy gradient variants (DDPG and TD3). Results The simulations effectively managed the epidemic's progression. Masking and vaccinations proved highly effective, significantly reducing both the outbreak's peak height and duration. By integrating individual behaviors, policy uncertainties, and multifaceted interventions, our dynamic control approach successfully mitigated the epidemic's impact. Conclusions Our model overcomes previous research limitations by embedding uncertainty and human behavior into public health policy frameworks. The simulation demonstrates that accounting for individual choices and imperfect data is crucial for designing effective interventions during complex pandemics, with masks and vaccines serving as pivotal tools.

**arXiv ID:** 2606.04562
</details>

<details>
<summary><strong>AIP: A Graph Representation for Learning and Governing Agent Skills</strong> - Zachary Blumenfeld, Jim Webber - [[pdf]](https://arxiv.org/pdf/2606.04781)</summary>

**Abstract:** Agent Skills today consist largely of free-form prose requiring the agent to read, interpret, and re-derive how to act in every session. This imposes two compounding costs: reduced reliability on implementation-heavy tasks, and difficulty in skill creation and improvement, since editing prose is a fragile process that both humans and agents struggle with, particularly for domain-specific procedural knowledge underrepresented in model training. The Agent Instruction Protocol (AIP) addresses both by modeling a skill as a directed execution graph: discrete steps as nodes backed by deterministic scripts or natural-language descriptions, connected by explicit typed input/output edges, and governed by a schema-validated YAML specification. A compiler meta-skill translates existing human-written skills into this form. The benefits are twofold. First, compiling human-written skills to AIP raised Claude Sonnet's mean task reward from 0.60 to 0.71 and pass rate from 53% to 67% across 27 real agent tasks from SkillsBench - a statistically significant gain (Wilcoxon signed-rank p = 0.011), winning 12 tasks to 2 with 13 ties - often in less wall-clock time. The graph delivers vetted, runnable units to the agent rather than asking it to re-derive code, commands, and tool calls from natural language. Second, on creation and improvement, because each skill is schema-validated, functionally testable, and addressable node-by-node, failures can be diagnosed and repaired precisely. Two authored-skill failures were traced to the script level. After adjusting the AIP spec and recompiling, both recovered with zero regressions (one task going from 0/5 to 5/5), turning skill improvement into a measurable tuning loop rather than a prose rewrite. That same graph structure supports corpus-level governance and skill introspection, and provides a natural action space for reinforcement learning over skills.

**arXiv ID:** 2606.04781
</details>

<details>
<summary><strong>Position: Deployed Reinforcement Learning should be Continual</strong> - Parnian Behdin, Kevin Roice, Golnaz Mesbahi - [[pdf]](https://arxiv.org/pdf/2606.04029)</summary>

**Abstract:** Reinforcement Learning (RL) has received increasing attention and adoption in real-world use cases. Most of these systems follow a train-then-fix paradigm, where trained agents do not learn while interacting with the world until performance degrades and retraining becomes necessary. In this position paper, we argue that deploying an agent that is incapable of optimality, but receives an evaluative reward signal, is inherently a continual RL problem. We identify four sources of non-stationarity after deployment that necessitate never-ending learning, and highlight why the best deployed agents never stop adapting. We analyze successful examples of continual RL in the real world, and present the community with the advantages and measures to move away from the current train-then-fix paradigm.

**arXiv ID:** 2606.04029
</details>

<details>
<summary><strong>Smart Transportation Without Neurons -- Fair Metro Network Expansion with Tabular Reinforcement Learning</strong> - Dimitris Michailidis, Sennay Ghebreab, Fernando P. Santos - [[pdf]](https://arxiv.org/pdf/2606.04167)</summary>

**Abstract:** We tackle the Metro Network Expansion Problem (MNEP), a subset of the Transport Network Design Problem (TNDP), which focuses on expanding metro systems to satisfy travel demand. Traditional methods rely on exact and heuristic approaches that require expert-defined constraints to reduce the search space. Recently, deep reinforcement learning (Deep RL) has emerged due to its effectiveness in complex sequential decision-making processes-it remains, however, computationally expensive, environmentally costly, and requires additional engineering to interpret. We show that MNEP problems are small enough to not require Deep RL methods. Reformulating the MNEP as a Non-Markovian Rewards Decision Process (NMRDP), we use tabular RL to achieve similar performance with significantly fewer training episodes, additionally offering greater interpretability. Additionally, we incorporate social equity criteria into the reward functions, focusing on efficiency and fairness, highlighting the versatility of our method. Evaluated in real-world settings-Xi'an and Amsterdam-our method reduces total episodes by a factor of 18 and total carbon emissions by a factor of 12 on average, while remaining competitive with Deep RL. This approach offers a replicable, modular, interpretable, and resource-efficient solution with potential applications to other combinatorial optimization problems.

**arXiv ID:** 2606.04167
</details>

<details>
<summary><strong>Exact Unlearning in Reinforcement Learning</strong> - Thanh Nguyen-Tang, Raman Arora - [[pdf]](https://arxiv.org/pdf/2606.04182)</summary>

**Abstract:** We formulate the problem of \emph{exact unlearning} in reinforcement learning, where the goal is to design an efficient framework that enables the removal of any user's data upon deletion request, i.e., the online learner's output after unlearning is \emph{indistinguishable} from what would have been produced had the deleted user never interacted with the learner. For any $\rho >0$, we show that there exists a reinforcement learning (RL) algorithm that is $\rho$-TV-stable and supports an exact unlearning procedure whose expected computational cost is only a $\rho \sqrt{\ln T}$ fraction of the computational cost of retraining from scratch. We construct such a $\rho$-TV-stable RL algorithm for tabular Markov decision processes (MDPs), which achieves a regret bound of $\mathcal{O}(H^2 \sqrt{SAT} + H^3 S^2 A + {H^{2.5} S^2 A}/{\rho})$, where $S, A, H$, and $T$ denote the number of states, the number of actions, the episode horizon, and the number of episodes, respectively. We also establish a lower bound of $\Omega(H\sqrt{\!SAT}\! +\! {SAH}/{\rho})$ for $\rho$-TV-stable RL algorithms, showing that our algorithm is nearly minimax optimal.

**arXiv ID:** 2606.04182
</details>

<details>
<summary><strong>StandardE2E: A Unified Framework for End-to-End Autonomous Driving Datasets</strong> - Stepan Konev - [[pdf]](https://arxiv.org/pdf/2606.04271)</summary>

**Abstract:** Autonomous driving has shifted from modular perception-prediction-planning stacks toward end-to-end (E2E) models that map sensor inputs directly to vehicle control, often regularized by auxiliary tasks such as 3D detection, motion forecasting, and HD-map perception. Progress is driven by a fast-growing ecosystem of sensor-rich driving datasets, yet each ships its own file formats, APIs, coordinate conventions, and modality coverage, leaving cross-dataset experimentation and even basic per-dataset preprocessing to be re-implemented per project. We present StandardE2E, a framework that provides a single unified interface over E2E driving datasets. StandardE2E (i) standardizes per-dataset preprocessing under one shared data schema; (ii) combines multiple datasets in a single PyTorch DataLoader for cross-dataset pretraining, auxiliary-task supervision, and scenario-level filtering; and (iii) reduces adding a new dataset to a single per-dataset mapping from raw frames to the canonical schema, leaving the entire downstream pipeline unchanged. The framework supports six datasets out of the box: Waymo End-to-End, Waymo Perception, Argoverse 2 Sensor, Argoverse 2 LiDAR, NAVSIM (OpenScene-v1.1), and WayveScenes101, and is released as the open-source standard-e2e Python package, available at this https URL.

**arXiv ID:** 2606.04271
</details>

<details>
<summary><strong>From Ticks to Flows: Dynamics of Neural Reinforcement Learning in Continuous Environments</strong> - Saket Tiwari, Tejas Kotwal, George Konidaris - [[pdf]](https://arxiv.org/pdf/2606.04275)</summary>

**Abstract:** We present a novel theoretical framework for deep reinforcement learning (RL) in continuous environments by modeling the problem as a continuous-time stochastic process, drawing on insights from stochastic control. Building on previous work, we introduce a viable model of actor-critic algorithm that incorporates both exploration and stochastic transitions. For single-hidden-layer neural networks, we show that the state of the environment can be formulated as a two time scale process: the environment time and the gradient time. Within this formulation, we characterize how the time-dependent random variables that represent the environment's state and estimate of the cumulative discounted return evolve over gradient steps in the infinite width limit of two-layer networks. Using the theory of stochastic differential equations, we derive, for the first time in continuous RL, an equation describing the infinitesimal change in the state distribution at each gradient step, under a vanishingly small learning rate. Overall, our work provides a novel nonparametric formulation for studying overparametrized neural actor-critic algorithms. We empirically corroborate our theoretical result using a toy continuous control task.

**arXiv ID:** 2606.04275
</details>

<details>
<summary><strong>Trace-Mediated Peak Bias: Bridging Temporal Credit Assignment and Cognitive Heuristics in Deep Reinforcement Learning</strong> - Viktor Veselý, Aleksandar Todorov, Erwan Escudie, Matthia Sabatelli - [[pdf]](https://arxiv.org/pdf/2606.04735)</summary>

**Abstract:** Temporal credit assignment is central to both biological and artificial intelligence, yet its interaction with non-linear function approximation is poorly understood. We identify a systematic failure mode in deep reinforcement learning (RL) termed Trace-Mediated Peak Bias (TMPB). At intermediate eligibility trace depths, agents irrationally prefer trajectories with high-magnitude reward ``peaks'' over alternatives with higher cumulative returns. This provides a mechanistic account of the Peak-End Rule: a human memory bias where experiences are judged by their most intense moments rather than integrated utility. We show that TMPB emerges because traces amplify distal Temporal Difference errors into ``gradient shocks'' that fixed-step-size Stochastic Gradient Descent cannot normalize, leading to global overestimation. Conversely, adaptive optimizers mitigate this pathology via second-moment normalization. Our results suggest that human-like saliency distortions may emerge naturally from the mathematical constraints of credit assignment in distributed systems, and that adaptive optimization is a theoretical necessity for rational value estimation.

**arXiv ID:** 2606.04735
</details>

<details>
<summary><strong>Archi: Agentic Operations at the CMS Experiment</strong> - Pietro Lugato, Luca Lavezzo, Jason Mohoney, Hasan Ozturk, Muhammad Hassan Ahmed, Juan Pablo Salas, Viphava Ohm, Krittin Phornsiricharoenphant, Gabriele Benelli, Mariarosaria D'Alfonso, Manasvita Joshi, Warren Nam, Aron Soha, Samantha Sunnarborg, Austin Swinney, Jack Tucker, Dmytro Kovalskyi, Tim Kraska, Christoph Paus - [[pdf]](https://arxiv.org/pdf/2606.04755)</summary>

**Abstract:** We present Archi, an open-source, end-to-end framework for scientific collaborations that combines the systematic ingestion and organization of heterogeneous data sources with the deployment of configurable, private, and extensible agents that retrieve and reason over them. An instance of Archi has been deployed for the Computing Operations team of the CMS experiment at CERN's LHC since February 2026 as a support agent for technical operators, offering retrieval and analysis capabilities by combining documentation, historical data, and live monitoring systems. We evaluate the system on operator feedback and a question set collected from production usage, graded by human and automated panels. The system proves effective at operational tasks, resolving real-world queries posed by CMS operators. We also observe that locally-hosted, open-weight models perform competitively, enabling fully private management of sensitive data.

**arXiv ID:** 2606.04755
</details>

<details>
<summary><strong>Scenario Generation for Risk-Aware Reinforcement Learning with Probably Approximately Safe Guarantees</strong> - Mohit Prashant, Arvind Easwaran - [[pdf]](https://arxiv.org/pdf/2606.04812)</summary>

**Abstract:** Guaranteeing safety is critical to the deployment of reinforcement learning (RL) agents in the real-world, especially as policies learned using deep RL may demonstrate susceptibility to transition perturbations that result in unknown or unsafe behaviour. A method of policy verification is to construct probabilistic barrier-certificates by sampling policy trajectories with respect to safety constraints, thereby demarcating known safe behaviour from unknown behaviour. Obtaining tight upper and lower bounds on the probability of violation of these constraints may be difficult if the policy is susceptible to transition uncertainty or perturbation that places the agent in insufficiently explored states. To address this, we approximate the distribution of the encountered state-space using a variational autoencoder (VAE) and construct upper and lower-bound barrier-certificates using latent characteristics of states to optimize for regions of known, safe behaviour with high confidence. We frame this in our work as a dual optimization problem where the lower-bound barrier-certificate presents a more conservative estimate of the safe region than the upper-bound barrier-certificate. Sampling states that lie within the set difference of the two during training, i.e. the non-robust region, allows us to tighten the upper and lower bounds to provide sharper probabilistic guarantees on safety. Within our study, we describe the guarantees placed and demonstrate the tightness of our bounds experimentally.

**arXiv ID:** 2606.04812
</details>

<details>
<summary><strong>Learning While Acting: A Skill-Enhanced Test-Time Co-Evolution Framework for Online Lifelong Learning Agents</strong> - Bo Mao, Jie Zhou, Yutao Yang, Xin Li, Xian Wei, Qin Chen, Xingjiao Wu, Liang He - [[pdf]](https://arxiv.org/pdf/2606.04815)</summary>

**Abstract:** Lifelong learning is essential for Large Language Model (LLM) agents operating in dynamic, interactive environments. However, existing lifelong learning agents for long-horizon tasks typically depend on discrete skill or past experiences retrieval with static parameters during inference, which prevents them from continuously internalizing test-time feedback like human learners. To bridge this gap, we propose Skill-enhanced Test-Time Co-Evolution (\texttt{LifeSkill}), a two-stage reinforcement learning framework for Online Lifelong Learning Agents. Specifically, we design Verifier-Guided Skill Learning that addresses the lack of direct supervision for skill extraction by rewarding candidate skills according to the average verifier success of multiple skill-conditioned policy rollouts, encouraging the model to generate skills that are useful for solving tasks rather than merely plausible in text. Furthermore, we introduce Online Skill Internalization, which continuously improves the policy model during test-time interaction by transforming skill-conditioned trajectories into reward signals. This enables the agent to directly internalize reasoning capabilities into its parameters, avoiding the context bloat of experience retrieval. Experiments on LifelongAgentBench show that LifeSkill improves average performance by 7 absolute points by comparing with existing lifelong agent baselines.

**arXiv ID:** 2606.04815
</details>

<details>
<summary><strong>Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning</strong> - Xuekang Wang, Zhuoyuan Hao, Shuo Hou, Hao Peng, Juanzi Li, Xiaozhi Wang - [[pdf]](https://arxiv.org/pdf/2606.04923)</summary>

**Abstract:** Rubric-based reinforcement learning (RL) uses an LLM-as-a-Judge (LaaJ) to score model outputs according to rubrics as rewards. However, policy models may exploit latent biases in the judge, leading to reward hacking and ineffective or unsafe training outcomes. In real-world rubric-based RL, such hacking behaviors are often subtle and entangled with multiple judge biases, making them difficult to analyze, detect, and mitigate. In this paper, we introduce CHERRL, a controllable hacking environment for rubric-based RL. By injecting known biases into LaaJ, CHERRL enables stable reproduction of reward hacking, explicit observation of reward divergence, and precise identification of hacking onset. This provides a clean experimental testbed for studying the mechanisms and mitigations of reward hacking in rubric-based RL. To demonstrate its utility, we analyze different judge biases from the perspectives of discoverability and exploitability, and explore an agent-based system for automatically detecting reward hacking onset from training logs. The code and environment are publicly available at this https URL.

**arXiv ID:** 2606.04923
</details>

<details>
<summary><strong>From Prompt to Process: a Process Taxonomy and Comparative Assessment of Frameworks Supporting AI Software Development Agents</strong> - Sanderson Oliveira de Macedo - [[pdf]](https://arxiv.org/pdf/2606.04967)</summary>

**Abstract:** AI tools for programming are no longer just autocomplete or chat assistants: they organize themselves as development frameworks, with process, roles, artifacts and verification. Recent surveys map agents and LLMs for software engineering, but a study centered on the operational frameworks that turn these capabilities into process is missing. We ran a directed search of primary sources, with a functional inclusion criterion and traction measurement, and selected six frameworks: GitHub Spec Kit, OpenSpec, BMAD Method, Get Shit Done (GSD), Spec Kitty and Reversa. Each attacks AI development through a different path: spec-driven development in full and lightweight variants, agent-driven agile planning, context engineering over the agent, worktree isolation and review, and recovery of operational specifications from legacy systems. Our central contribution is a six-dimension process taxonomy: specification, context, roles, execution, validation and portability, with a scoring rubric that turns it into a replicable instrument. We apply it to the six frameworks and an out-of-sample case, Spec-Flow. Two results stand out. Among frameworks that already adopt some process there is convergence: the isolated prompt loses centrality, and persistent artifacts, work contracts, traceability and human review become mechanisms that reduce ambiguity and coordinate agents. And no framework strongly covers all six dimensions, exposing a structural trade-off between process depth and portability across agents. We also found recurring risks: drift between specification and code, excessive trust in generated artifacts, fragility of community extensions, platform dependence and a lack of benchmarks for the complete process. We close with a research agenda for empirical evaluation, focused on intermediate-quality metrics, context governance, installation security and reproducibility.

**arXiv ID:** 2606.04967
</details>

<details>
<summary><strong>Reinforcement Learning from Rich Feedback with Distributional DAgger</strong> - Rishabh Agrawal, Jacob Fein-Ashley, Paria Rashidinejad - [[pdf]](https://arxiv.org/pdf/2606.05152)</summary>

**Abstract:** Reasoning models have advanced rapidly, but the dominant reinforcement learning from verifiable rewards (RLVR) recipe remains surprisingly narrow: sample many responses and reward each with a single bit indicating whether the final answer is correct. Yet many settings provide rich feedback, including execution traces, tool outputs, expert corrections, and model self-evaluations. We study how to use such feedback through a distributional variant of the classic imitation learning algorithm DAgger, where the learner has local access to an expert distribution on states visited by the current policy. This yields a simple forward cross-entropy objective that admits a blackbox expert and whose sequence-level gradient {conduct rich credit assignment by propagating} future expert-student disagreement back to earlier decisions. We show that prior RL with self-distillation objectives based on reverse KL or Jensen-Shannon fail to guarantee monotonic policy improvement: even when the expert has higher reward, their updates may increase probability on worse actions. In contrast, we show that forward cross-entropy admits monotonic policy improvement and enjoys guarantees on regret. We further show that our objective optimizes a lower bound on teacher-weighted likelihood of success, leading to improved Pass@N. Empirically, our approach, DistIL, improves over RLVR and RL with self-distillation baselines across a variety of domains: scientific reasoning, coding, and solving hard mathematical problems.

**arXiv ID:** 2606.05152
</details>

<details>
<summary><strong>Entropy Is Not Enough: Unlocking Effective Reinforcement Learning for Visual Reasoning via Vision-Anchored Token Selection</strong> - Senjie Jin, Peixin Wang, Boyang Liu, Xiaoran Fan, Shuo Li, Zhiheng Xi, Jiazheng Zhang, Yuhao Zhou, Tao Gui, Qi Zhang, Xuanjing Huang - [[pdf]](https://arxiv.org/pdf/2606.03937)</summary>

**Abstract:** While token-level entropy is commonly recognized as effective for credit assignment in text-only reinforcement learning with verifiable rewards (RLVR), it remains unclear whether this mechanism still holds in visual reasoning. Our controlled study shows that this mechanism collapses in visual reasoning due to the omission of vision-sensitive tokens with naturally low entropy. Although existing multimodal RL methods increasingly acknowledge the importance of visual perception, they struggle to satisfy the inherent demand for interleaving precise perceptual grounding with semantic reasoning, either lacking systematic visual measurements or overlooking that token entropy primarily drives semantic exploration. To address this, we introduce VEPO (Vision-Entropy token-selection for Policy Optimization), an effective RL framework explicitly integrating visual sensitivity with token entropy via a principled multiplicative coupling, where VEPO redirects gradient credit toward tokens which are simultaneously visually grounded and highly informative. Extensive experiments demonstrate VEPO's leading performance, significantly outperforming the entropy-only baseline by 2.28 points at 7B-scale and 3.15 points at 3B-scale. Ablations further substantiate the soundness of our method.

**arXiv ID:** 2606.03937
</details>

<details>
<summary><strong>OpenAgenet/OAN: Technical Architecture for Trust-Governed Agent Identity and Discovery</strong> - Jinliang Xu - [[pdf]](https://arxiv.org/pdf/2606.03163)</summary>

**Abstract:** This paper describes the technical architecture of OpenAgenet / OAN. OAN is a protocol-neutral trust layer for open Agent interconnection. It specifies the role architecture, identity objects, registration workflow, Root-governed lifecycle, Root-verified package model, authorization-aware Discovery, signed trusted invocation, verification requirements, state transitions, security properties, implementation boundaries, and deployment considerations. The design is intended to support heterogeneous Agent frameworks and interaction protocols, including MCP, A2A, ANP-like systems, and domain-specific Agent protocols. OAN does not define the entire business conversation among Agents; it defines how Agent identities become admissible, discoverable, verifiable, and safe to approach before protocol-specific interaction begins.

**arXiv ID:** 2606.03163
</details>

<details>
<summary><strong>ToolRosella: Translating Code Repositories into Standardized Tools for Scientific Agents</strong> - Shimin Di, Xujie Yuan, Hanghui Guo, Chaoqian Ouyang, Yongxu Liu, Ling Yue, Zhangze Chen, Libin Zheng, Jia Zhu, Shaowu Pan, Jian Yin, Yong Rui, Min-Ling Zhang - [[pdf]](https://arxiv.org/pdf/2603.09290)</summary>

**Abstract:** Large Language Model (LLM)-based agent systems are increasingly used for scientific tasks, yet their practical capability remains constrained by the narrow scope of manually curated tools they can invoke. Much scientific computational functionality already exists in open-source code repositories, but these resources remain difficult to standardize, operationalize, and invoke reliably for agent use. Here we present ToolRosella, a framework that automatically transforms heterogeneous scientific code repositories into standardized, agent-invocable tools. ToolRosella combines repository analysis, tool interface construction, execution testing, and iterative repair to address the problem of repository-to-tool standardization. Across 122 GitHub repositories spanning 35 subdisciplines in six domains, ToolRosella reaches a 61.5\% repository conversion success rate after iterative repair, with a 4.4 speedup over human engineers. The resulting 1,580 callable tools support a downstream task success rate of 84.0\% and improve performance when integrated into other agent frameworks, particularly on tasks whose required tools are absent from fixed, curated inventories.

**arXiv ID:** 2603.09290
</details>

<details>
<summary><strong>Read the Trace, Steer the Path: Trajectory-Aware Reinforcement Learning for Diffusion Language Models</strong> - Anant Khandelwal, Manish Gupta - [[pdf]](https://arxiv.org/pdf/2606.04396)</summary>

**Abstract:** Diffusion large language models (dLLMs) generate responses by iteratively unmasking and revising many positions in parallel. This process leaves a rich denoising trace depicting which tokens become confident, which remain unstable, and when commitments form. Existing dLLM reinforcement learning methods use this signal only weakly. Flat rollouts are cheap, but assign a single outcome reward to the whole trajectory. Tree rollouts provide finer, verifiable training signals by branching partial trajectories and propagating leaf rewards upward, but are compute intensive. We ask whether the denoising trace itself can provide tree-like supervision without tree-level compute. We introduce CAPR (Cached-Amortized Path Refinement), a dLLM-RL algorithm that summarizes the denoising trace into a compact path state, uses cached trajectory states to generate cheap sibling continuations, and trains a block-level value head for local block-wise supervision. Under a block-wise unmasking schedule, CAPR records path-state and block-progress features, then redistributes the final outcome reward across blocks according to the tokens revealed in each block. This trains the value head to convert one sparse reward into block-level PPO weights. CAPR therefore recovers much of the granularity of tree search while avoiding full tree expansion, reducing rollout-generation cost to roughly 0.75x that of flat rollouts and 0.6x that of tree rollouts (under standard settings). Across 4x4 Sudoku, Countdown, GSM8K, and Math500, on dense and mixture-of-experts LLaDA backbones, CAPR sets a new state of the art for RL-tuned dLLMs at 256- and 512-token budgets. On Sudoku, it matches the strongest tree-structured baseline at less than one third of the per-step compute.

**arXiv ID:** 2606.04396
</details>

<details>
<summary><strong>GRAIL: Gradient-Reweighted Advantages for Reinforcement Learning with Verifiable Rewards</strong> - Tej Deep Pala, Vernon Toh, Soujanya Poria - [[pdf]](https://arxiv.org/pdf/2606.04889)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (e.g. GRPO) is now a common way to improve mathematical reasoning in Large Language Models (LLMs). However, current methods usually broadcast one sequence-level advantage to all tokens, or use costly process reward models (PRMs) for step-level supervision. Uniform advantage distribution assumes that all tokens contribute equally to the final reward. This dilutes the gradient signal, since flawed reasoning steps and filler words are updated as strongly as valid logical inferences. To address this, we introduce Gradient-Reweighted Advantage (GRAIL), an intrinsic token-wise advantage reweighting method. GRAIL uses gradient-activation saliency to place more weight on tokens that are more locally sensitive to the final answer. Evaluations across five models from the Qwen3, R1-distilled and OctoThinker families show that GRAIL consistently outperforms GRPO. GRAIL achieved an average improvement of 3.60% in accuracy and 3.05% in Pass@3, demonstrating that fine-grained reasoning alignment can be achieved without process-level supervision.

**arXiv ID:** 2606.04889
</details>

<details>
<summary><strong>VentAgent: When LLMs Learn to Breathe -- Multi-Objective Arbitration for ARDS Ventilation</strong> - Teqi Hao, Yuxuan Fu, Xiaoyu Tan, Shaojie Shi, Bohao Lv, Yinghui Xu, Xihe Qiu - [[pdf]](https://arxiv.org/pdf/2606.04632)</summary>

**Abstract:** Mechanical ventilation for Acute Respiratory Distress Syndrome (ARDS) requires balancing competing physiological goals, including oxygenation, lung protection, and acid-base homeostasis. However, current data-driven methods, especially those imitating retrospective Electronic Health Records (EHR), often suffer from imitation bias. They may capture superficial correlations from inconsistent clinical demonstrations, such as associating passive ventilator settings with survival because such settings are common in stable patients, and thus fail to generalize to volatile or out-of-distribution phenotypes. Standard Reinforcement Learning (RL) methods also struggle with the adversarial trade-offs of critical care and often produce opaque policies with limited clinical interpretability. To address these limitations, we introduce VentAgent, a hierarchical framework in which Large Language Models (LLMs) act as transparent arbitrators for mechanical ventilation. We reformulate ventilation control as a dynamic Multi-Objective Arbitration process rather than single-objective optimization. VentAgent decomposes decision-making into three interpretable stages: Perception, Planning, and Orchestration. By leveraging the semantic reasoning capabilities of LLMs, it synthesizes strategies from heterogeneous experts and resolves conflicting clinical priorities through an explicit coordination mechanism. Evaluations on a high-fidelity physiological simulator show that VentAgent outperforms state-of-the-art RL and classical control baselines. Moreover, it converts control decisions into human-readable reasoning chains, offering a safer, more interpretable, and adaptable paradigm for critical care automation.

**arXiv ID:** 2606.04632
</details>

<details>
<summary><strong>Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning</strong> - Haoran Luo, Haihong E, Guanting Chen, Qika Lin, Yikai Guo, Fangzhi Xu, Zemin Kuang, Meina Song, Xiaobao Wu, Yifan Zhu, Luu Anh Tuan - [[pdf]](https://arxiv.org/pdf/2507.21892)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) mitigates hallucination in LLMs by incorporating external knowledge, but relies on chunk-based retrieval that lacks structural semantics. GraphRAG methods improve RAG by modeling knowledge as entity-relation graphs, but still face challenges in high construction cost, fixed one-time retrieval, and reliance on long-context reasoning and prompt design. To address these challenges, we propose Graph-R1, the first agentic GraphRAG framework via end-to-end reinforcement learning (RL). It introduces lightweight knowledge hypergraph construction, models retrieval as a multi-turn agent-environment interaction, and optimizes the agent process via an end-to-end reward mechanism. Experiments on standard RAG datasets show that Graph-R1 outperforms traditional GraphRAG and RL-enhanced RAG methods in reasoning accuracy, retrieval efficiency, and generation quality. Our software and data are publicly available at this https URL.

**arXiv ID:** 2507.21892
</details>

<details>
<summary><strong>T$^\star$: Progressive Block Scaling for Masked Diffusion Language Models Through Trajectory Aware Reinforcement Learning</strong> - Hanchen Xia, Baoyou Chen, Yutang Ge, Guojiang Zhao, Siyu Zhu - [[pdf]](https://arxiv.org/pdf/2601.11214)</summary>

**Abstract:** We present T$^\star$, a simple TraceRL-based training curriculum for progressive block-size scaling in masked diffusion language models (MDMs). Starting from an AR-initialized small-block MDM, T$^\star$ transitions smoothly to larger blocks, enabling higher-parallelism decoding with minimal performance degradation on math reasoning benchmarks. Moreover, further analysis suggests that T$^\star$ may actually converge to an alternative decoding schedule that achieves comparable performance.

**arXiv ID:** 2601.11214
</details>

<details>
<summary><strong>Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool Use</strong> - Aradhye Agarwal, Gurdit Siyan, Yash Pandya, Joykirat Singh, Akshay Nambi, Ahmed Awadallah - [[pdf]](https://arxiv.org/pdf/2603.03205)</summary>

**Abstract:** Agentic language models operate in a fundamentally different safety regime than chat models: they must plan, call tools, and execute long-horizon actions where a single misstep, such as accessing files or entering credentials, can cause irreversible harm. Existing alignment methods, largely optimized for static generation and task completion, break down in these settings due to sequential decision-making, adversarial tool feedback, and overconfident intermediate reasoning. We introduce MOSAIC, a post-training framework that aligns agents for safe multi-step tool use by making safety decisions explicit and learnable. MOSAIC structures inference as a plan, check, then act or refuse loop, with explicit safety reasoning and refusal as first-class actions. To train without trajectory-level labels, we use preference-based reinforcement learning with pairwise trajectory comparisons, which captures safety distinctions often missed by scalar rewards. We evaluate MOSAIC zero-shot across three model families, Qwen2.5-7B, Qwen3-4B-Thinking, and Phi-4, and across out-of-distribution benchmarks spanning harmful tasks, prompt injection, benign tool use, and cross-domain privacy leakage. MOSAIC reduces harmful behavior by up to 50%, increases harmful-task refusal by over 20% on injection attacks, cuts privacy leakage, and preserves or improves benign task performance, demonstrating robust generalization across models, domains, and agentic settings.

**arXiv ID:** 2603.03205
</details>

<details>
<summary><strong>Aryabhata 2: Scaling Reinforcement Learning for Advanced STEM Reasoning</strong> - Ritvik Rastogi, Vishal Singh, Tejas Chaudhari, Sandeep Varma - [[pdf]](https://arxiv.org/pdf/2605.28829)</summary>

**Abstract:** Competitive STEM examinations such as JEE and NEET require multi-step symbolic reasoning, precise numerical computation, and deep conceptual understanding across physics, chemistry, and mathematics. Recent large language models perform strongly on common reasoning benchmarks, yet they remain difficult to deploy at scale, where millions of student doubts demand domain-specific, consistently structured problem solving.
We introduce Aryabhata 2, a reasoning-focused language model for competitive STEM examinations, trained via reinforcement-learning post-training. Using PhysicsWallah's internal question banks, we construct a high-quality training curriculum and post-train GPT-OSS-20B through reinforcement learning with verifiable rewards. Training combines prolonged reinforcement learning with broadened exploration via progressively larger rollout group sizes.
We evaluate Aryabhata 2 on competitive examination benchmarks, including JEE Main, JEE Advanced, and NEET, as well as out-of-distribution reasoning datasets such as AIME, HMMT, MMLU-Pro, MMLU-Redux 2.0, and GPQA. Results show that Aryabhata 2 outperforms its base model GPT-OSS-20B on competitive STEM reasoning while requiring substantially fewer output tokens (up to 64\% fewer).

**arXiv ID:** 2605.28829
</details>

<details>
<summary><strong>Structured Prompt Optimization Meets Reinforcement Learning for Global and Local Interpretability over Complex Text</strong> - Tianyang Zhou, Wenbo Chen, Pierre Jinghong Liang, Leman Akoglu - [[pdf]](https://arxiv.org/pdf/2605.29076)</summary>

**Abstract:** LLMs have advanced text classification, yet existing paradigms face a trade-off: supervised (label only) fine-tuning is scalable but offers limited reasoning on complex text and lacks broader model transparency, while discrete prompt optimization offers human-readable instructions but struggles with performance and scalability. We introduce eXTC (eXplainable Text Classifier) with three progressive stages: (1) learning a Standard Operating Procedure (SOP, or rulebook) in natural language via a new Structured Prompt Optimization algorithm; (2) SOP-grounded reasoning distillation from a large teacher LLM into a compact LM; and (3) expanding reasoning capabilities beyond the initial SOP via reinforcement learning. This design enables eXTC to provide (i) fast inference via a compact LM, with (ii) inference-time local reasoning traces, alongside a global, modular explanation of its learned domain rules, while (iii) significantly outperforming existing paradigms across diverse benchmarks in both classification performance and explanation quality, with stage-by-stage gains.

**arXiv ID:** 2605.29076
</details>

<details>
<summary><strong>GAPD: Gold-Action Policy Distillation for Agentic Reinforcement Learning in Knowledge Base Question Answering</strong> - Xin Sun, Jianan Xie, Zhongqi Chen, Qiang Liu, Shu Wu, Bowen Song, Weiqiang Wang, Zilei Wang, Liang Wang - [[pdf]](https://arxiv.org/pdf/2605.29584)</summary>

**Abstract:** Reinforcement learning (RL) is a natural fit for agentic knowledge base question answering (KBQA), where a model must issue executable actions, observe knowledge-base feedback, and eventually return an answer. However, current RL-based KBQA systems mainly optimize sparse rewards from the final answer, leaving intermediate action errors weakly supervised. This is especially limiting for logical-form annotated KBQA benchmarks: gold logical forms can be converted into executable action sequences, but existing pipelines use them mainly for warm-start data construction rather than for on-policy RL updates. We propose GAPD, a training-time Gold-Action Policy Distillation framework that adds dense token-level guidance to outcome-based RL. To align gold actions with on-policy student rollouts, GAPD uses MID-ANCHOR MATCHING: it treats the intermediate entities reached during student exploration and gold execution as state anchors, and matches student states to gold states through these explored entity sets. The current policy conditioned on this aligned gold action serves as a stop-gradient teacher, whose token distribution is distilled back to the ordinary student policy over generated action-token spans. GAPD consistently surpasses the current state of the art on WebQSP, GrailQA, and GraphQ.

**arXiv ID:** 2605.29584
</details>

<details>
<summary><strong>Synthesize and Reward -- Reinforcement Learning for Multi-Step Tool Use in Live Environments</strong> - Ibrahim Abdelaziz, Asim Munawar, Kinjal Basu, Maxwell Crouse, Chulaka Gunasekara, Suneet Katrekar, Pavan Kapanipathi - [[pdf]](https://arxiv.org/pdf/2606.03892)</summary>

**Abstract:** Training LLMs to orchestrate multi-step tool calls is held back by three coupled obstacles: realistic stateful execution environments are costly to build, synthetic training queries are often detached from the server's actual state (so the generated tool calls fail to execute), and recall-based RL rewards incentivize verbose tool-calling patterns. We present PROVE (Programmatic Rewards On Verified Environments), a framework with three contributions: (1) a library of 20 stateful MCP (Model Context Protocol) servers exposing 343 tools, enabling live-execution RL training with session-scoped state isolation; (2) a state-machine data synthesis pipeline that generates multi-turn tool-call trajectories grounded in live-sampled server state, so generated queries reference entities that actually exist; and (3) a multi-component programmatic reward with an adaptive efficiency penalty that counters the verbosity incentive of recall-based rewards. We train four models (Qwen3-4B, Qwen3-8B, Qwen2.5-7B, Granite-4.1-8B) with GRPO on the resulting ~13K training examples. On BFCL Multi-Turn, tau2-bench, and T-Eval, PROVE yields improvements of up to +10.2, +6.8, and +6.5 points respectively, demonstrating that this framework yields consistent gains on multi-step tool orchestration across two model families.

**arXiv ID:** 2606.03892
</details>

<details>
<summary><strong>Dynamic Multi-Pair Trading Strategy in Cryptocurrency Markets with Deep Reinforcement Learning</strong> - Damian Lebiedź, Robert Ślepaczuk - [[pdf]](https://arxiv.org/pdf/2606.04574)</summary>

**Abstract:** This study aims to determine whether the application of Deep Reinforcement Learning (DRL) as a specialized execution overlay can enhance pair trading in highly volatile cryptocurrency markets. Although classical implementations of the strategy have proven successful in traditional equities, they frequently exhibit rigidity and suffer from severe divergence risks when applied to high-variance environments. To address this need, this research introduces novel concepts. To construct a robust system, we developed a hierarchical "Filter-then-Rank" pair selection methodology and a proprietary "Fixed Risk, Adaptive Mean" execution model. The system employs a Proximal Policy Optimization (PPO) agent with a Long Short-Term Memory (LSTM) layer to govern execution decisions within strict deterministic risk management boundaries. Evaluated on 1-hour interval data from the Binance USD-M Futures market, the optimized RL policy achieved an out-of-sample performance that substantially outperformed the heuristic baseline. A stationary circular block bootstrap robustness check confirms that the agent's risk-adjusted outperformance is statistically significant at the 10 percent level. Although falling marginally short of the stricter 5 percent threshold, this result highlights the extreme idiosyncratic variance characteristic of digital assets. Ultimately, this thesis contributes to the quantitative finance literature by introducing a hybrid architecture that combines statistical arbitrage with DRL execution policies. Furthermore, it delivers a novel framework for safe reinforcement learning via deterministic shielding, proving that anchoring a neural policy to statistically robust boundaries successfully mitigates severe divergence risks.

**arXiv ID:** 2606.04574
</details>

<details>
<summary><strong>Explainably Safe Reinforcement Learning</strong> - Sabine Rieder, Stefan Pranger, Debraj Chakraborty, Jan Křetínský, Bettina Könighofer - [[pdf]](https://arxiv.org/pdf/2606.04634)</summary>

**Abstract:** Trust in a decision-making system requires both safety guarantees and the ability to interpret and understand its behavior. This is particularly important for learned systems, whose decision-making processes are often highly opaque. Shielding is a prominent model-based technique for enforcing safety in reinforcement learning. However, because shields are automatically synthesized using rigorous formal methods, their decisions are often similarly difficult for humans to interpret. Recently, decision trees became customary to represent controllers and policies. However, since shields are inherently non-deterministic, their decision tree representations become too large to be explainable in practice. To address this challenge, we propose a novel approach for explainable safe RL that enhances trust by providing human-interpretable explanations of the shield's decisions. Our method represents the shielding policy as a hierarchy of decision trees, offering top-down, case-based explanations. At design time, we use a world model to analyze the safety risks of executing actions in given states. Based on this analysis, we construct both the shield and a high-level decision tree that classifies states into risk categories (safe, critical, dangerous, unsafe), explaining why a situation may be safety-critical. At runtime, we generate localized decision trees that explain which actions are allowed and why others are deemed unsafe. Our method facilitates explainability of the safety aspect in safe-by-shielding reinforcement learning, requires no additional information beyond what is already used for shielding, incurs minimal overhead, and integrates readily into existing shielded RL pipelines. In our experiments, we compute explanations using decision trees that are several orders of magnitude smaller than the original shield.

**arXiv ID:** 2606.04634
</details>

<details>
<summary><strong>CADET: A Modular Platform for Evaluating Distributed Cooperative Autonomy in Connected Autonomous Vehicles</strong> - Pragya Sharma, Brian Wang, Mani Srivastava - [[pdf]](https://arxiv.org/pdf/2606.04072)</summary>

**Abstract:** Deep learning models are increasingly central to autonomous vehicle (AV) pipelines, yet their integration has traditionally followed a monolithic design where perception, planning, and control execute on a single onboard computer. This design overlooks the emerging paradigm of cooperative autonomy, where vehicles interact with roadside units (RSUs), edge servers, and cloud-hosted intelligence through vehicle-to-everything (V2X) connectivity. Cooperative perception and control improve safety and efficiency, but also introduce systems-level challenges: network latency, compute heterogeneity, and multi-tenant contention, all critically affect real-time decision-making. These challenges are further amplified by the increasing reliance on large foundation models, whose scale necessitates cloud deployment.
We present CADET (Cooperative Autonomy through Distributed Experimentation Toolkit), a modular platform for systematic and reproducible evaluation of distributed cooperative autonomy systems under realistic deployment conditions. CADET decouples the AV stack into composable modules that can be flexibly deployed across vehicles, infrastructure, and edge/cloud tiers. The framework integrates state-of-the-art models, incorporates trace-driven network and workload emulation, and provides synchronized model-, system-, and task-level instrumentation. Through V2V and V2I experiments, we show that distributed deployment choices fundamentally shape safety, with V2V intent packets outperforming cloud-based perception and RSU-assisted perception sustaining safety until overloaded by concurrent requests. Although designed for AV pipelines, CADET also supports dataset-driven experimentation, enabling systems and ML researchers to benchmark distributed inference workloads independently of full vehicle simulation. CADET is open source, with code and demo available at this https URL.

**arXiv ID:** 2606.04072
</details>

<details>
<summary><strong>COP-Q: Safety-First Reinforcement Learning for Robot Control via Cholesky-Ordered Projection</strong> - Guopeng Li, Moritz A. Zanger, Matthijs T. J. Spaan, Julian F. P. Kooij - [[pdf]](https://arxiv.org/pdf/2606.04749)</summary>

**Abstract:** Safe robot control requires maximizing return while satisfying safety constraints. In off-policy safe reinforcement learning, reward and safety Q-values are commonly learned by separate critic ensembles, with uncertainty handled independently for each objective. This objective-wise treatment neglects inter-objective correlation and can lead to overly conservative value estimates, thereby reducing sample efficiency. To address this issue, we propose Cholesky-Ordered Projection Q-learning (COP-Q), a safety-first method that incorporates inter-objective covariance into vector-valued Q-value estimation. COP-Q constructs a generalized confidence bound in the joint Q-value space and uses Cholesky factorization to encode objective priority in a sequential form. This preserves conservatism on safety while adaptively reducing excessive conservatism on the reward objective. The resulting estimate is used in both temporal-difference target computation and actor optimization. COP-Q incurs minimal computational overhead and is readily compatible with most existing deep Q-learning frameworks. Experiments on robot locomotion in Brax and safe navigation in Safety-Gymnasium, covering both hard- and soft-safety settings, demonstrate that COP-Q achieves strong safety performance together with competitive or improved sample efficiency relative to representative baselines.

**arXiv ID:** 2606.04749
</details>

<details>
<summary><strong>Blessing from Human-AI Interaction: Super Reinforcement Learning in Confounded Environments</strong> - Jiayi Wang, Zhengling Qi, Chengchun Shi - [[pdf]](https://arxiv.org/pdf/2209.15448)</summary>

**Abstract:** As AI becomes more prevalent throughout society, effective methods of integrating humans and AI systems that leverage their respective strengths and mitigate risk have become an important priority. In this paper, we introduce the paradigm of super policy learning that takes advantage of Human-AI interaction for data driven sequential decision making. This approach utilizes the observed action, either from AI or humans, as input for achieving a stronger oracle in policy learning for the decision maker (humans or AI). In the decision process with unmeasured confounding, the actions taken by past agents can offer valuable insights into undisclosed information. By including this information for the policy search in a novel and legitimate manner, the proposed super policy learning will yield a super-policy that is guaranteed to outperform both the standard optimal policy and the behavior one (e.g., past agents' actions). We call this stronger oracle a blessing from human-AI interaction. Furthermore, to address the issue of unmeasured confounding in finding super-policies using the batch data, a number of nonparametric and causal identifications are established under the framework of proximal causal inference. Building upon on these novel identification results, we develop several super-policy learning algorithms and systematically study their theoretical properties such as finite-sample regret guarantee. Finally, we illustrate the effectiveness of our proposal through extensive simulations and real-world applications.

**arXiv ID:** 2209.15448
</details>

<details>
<summary><strong>MineXplore: An Open-Source Reinforcement Learning Exploration Benchmark for GNSS-Denied Underground Environment</strong> - Abhishek S, Badrikanath Praharaj, Sreeram MV - [[pdf]](https://arxiv.org/pdf/2606.04569)</summary>

**Abstract:** Underground mines present extreme conditions for autonomous robot navigation: GPS is denied, lighting is degraded, and tunnel topology is loop-rich and non-convex. Simulation benchmarks grounded in real production-mine geometry and compatible with GPU-accelerated learning pipelines do not yet exist in the open-source ecosystem. We present MineXplore, an open-source MuJoCo-based navigation benchmark derived from the Leung et al. 2017 Chilean underground copper mine dataset. The environment reconstructs a 104,423 sq.m tunnel network through an six-stage contour-to-MJCF pipeline incorporating octagonal wall cross-sections, LiDAR-sourced jagged wall geometry, three terrain friction zones, a global 5 degree incline, and periodic spot lighting. Geometric fidelity is validated at an Intersection over Union (IoU) of 0.9538 against the source survey map, and surface texture similarity scores 79.4% across six structural dimensions. A single-agent PPO baseline trained via RLlib across five independent random seeds achieves a best rolling coverage of 88.89% (3 of 5 seeds reaching the 90% coverage target), confirming that MineXplore supports stable and reproducible policy learning under realistic underground sensing and topology.

**arXiv ID:** 2606.04569
</details>

<details>
<summary><strong>LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion</strong> - Jiangran Lyu, Kai Liu, Xuheng Zhang, Haoran Liao, Yusen Feng, Wenxuan Zhu, Tingrui Shen, Jiayi Chen, Jiazhao Zhang, Yifei Dong, Wenbo Cui, Senmao Qi, Shuo Wang, Yixin Zheng, Mi Yan, Xuesong Shi, Haoran Li, Dongbin Zhao, Ming-Yu Liu, Zhizheng Zhang, Li Yi, Yizhou Wang, He Wang - [[pdf]](https://arxiv.org/pdf/2602.12215)</summary>

**Abstract:** Recent robot foundation models largely rely on large-scale behavior cloning, which imitates expert actions but discards transferable dynamics knowledge embedded in heterogeneous embodied data. While the Unified World Model (UWM) formulation has the potential to leverage such diverse data, existing instantiations struggle to scale to foundation-level due to coarse data usage and fragmented datasets. We introduce LDA-1B, a robot foundation model that scales through universal embodied data ingestion by jointly learning dynamics, policy, and visual forecasting, assigning distinct roles to data of varying quality. To support this regime at scale, we assemble and standardize EI-30k, an embodied interaction dataset comprising over 30k hours of human and robot trajectories in a unified format. Scalable dynamics learning over such heterogeneous data is enabled by prediction in a structured DINO latent space, which avoids redundant pixel-space appearance modeling. Complementing this representation, LDA-1B employs a multi-modal diffusion transformer to handle asynchronous vision and action streams, enabling stable training at the 1B-parameter scale. Experiments in simulation and the real world show LDA-1B outperforms prior methods (e.g., $\pi_{0.5}$) by up to 21\%, 48\%, and 23\% on contact-rich, dexterous, and long-horizon tasks, respectively. Notably, LDA-1B enables data-efficient fine-tuning, gaining 10\% by leveraging 30\% low-quality trajectories typically harmful and discarded.

**arXiv ID:** 2602.12215
</details>

<details>
<summary><strong>Contextual Multi-Task Reinforcement Learning for Autonomous Reef Monitoring</strong> - Melvin Laux, Yi-Ling Liu, Rina Alo, Sören Töpper, Mariela De Lucas Alvarez, Frank Kirchner, Rebecca Adam - [[pdf]](https://arxiv.org/pdf/2604.12645)</summary>

**Abstract:** Although autonomous underwater vehicles promise the capability of marine ecosystem monitoring, their deployment is fundamentally limited by the difficulty of controlling vehicles under highly uncertain and non-stationary underwater dynamics. To address these challenges, we employ a data-driven reinforcement learning approach to compensate for unknown dynamics and task variations. Traditional single-task reinforcement learning has a tendency to overfit the training environment, thus, limit the long-term usefulness of the learnt policy. Hence, we propose to use a contextual multi-task reinforcement learning paradigm instead, allowing us to learn controllers that can be reused for various tasks, e.g., detecting oysters in one reef and detecting corals in another. We evaluate whether contextual multi-task reinforcement learning can efficiently learn robust and generalisable control policies for autonomous underwater reef monitoring. We train a single context-dependent policy that is able to solve multiple related monitoring tasks in a simulated reef environment in HoloOcean. In our experiments, we empirically evaluate the contextual policies regarding sample-efficiency, zero-shot generalisation to unseen tasks, and robustness to varying water currents. By utilising multi-task reinforcement learning, we aim to improve the training effectiveness, as well as the reusability of learnt policies to take a step towards more sustainable procedures in autonomous reef monitoring.

**arXiv ID:** 2604.12645
</details>

<details>
<summary><strong>Learning While Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies</strong> - Yi Wang, Xinchen Li, Pengwei Xie, Pu Yang, Buqing Nie, Yunuo Cai, Qinglin Zhang, Chendi Qu, Jeffrey Wu, Jianheng Song, Xinlin Ren, Jingshun Huang, Mingjie Pan, Siyuan Feng, Zhi Chen, Jianlan Luo - [[pdf]](https://arxiv.org/pdf/2605.00416)</summary>

**Abstract:** Generalist robot policies increasingly benefit from large-scale pretraining, but offline data alone is insufficient for robust real-world deployment. Deployed robots encounter distribution shifts, long-tail failures, task variations, and human correction opportunities that fixed demonstration datasets cannot fully capture. We present Learning While Deploying (LWD), a fleet-scale offline-to-online reinforcement learning framework for continual post-training of generalist Vision-Language-Action (VLA) policies. Starting from a pretrained VLA policy, LWD closes the loop between deployment, shared physical experience, policy improvement, and redeployment by using autonomous rollouts and human interventions collected across a robot fleet. To stabilize learning from heterogeneous, sparse-reward fleet data, LWD combines Distributional Implicit Value Learning (DIVL) for robust value estimation with Q-learning via Adjoint Matching (QAM) for policy extraction in flow-based VLA action generators. We validate LWD on a fleet of 16 dual-arm robots across eight real-world manipulation tasks, including semantic grocery restocking and 3--5 minute long-horizon tasks. A single generalist policy improves as fleet experience accumulates, reaching an average success rate of 95%, with the largest gains on long-horizon tasks.

**arXiv ID:** 2605.00416
</details>

<details>
<summary><strong>Simplicial Embeddings Improve Sample Efficiency in Actor-Critic Agents</strong> - Johan Obando-Ceron, Walter Mayor, Samuel Lavoie, Scott Fujimoto, Aaron Courville, Pablo Samuel Castro - [[pdf]](https://arxiv.org/pdf/2510.13704)</summary>

**Abstract:** Recent works have proposed accelerating the wall-clock training time of actor-critic methods via the use of large-scale environment parallelization; unfortunately, these can sometimes still require large number of environment interactions to achieve a desired level of performance. Noting that well-structured representations can improve the generalization and sample efficiency of deep reinforcement learning (RL) agents, we propose the use of simplicial embeddings: lightweight representation layers that constrain embeddings to simplicial structures. This geometric inductive bias results in sparse and discrete features that stabilize critic bootstrapping and strengthen policy gradients. When applied to FastTD3, FastSAC, and PPO, simplicial embeddings consistently improve sample efficiency and final performance across a variety of continuous- and discrete-control environments, without any loss in runtime speed.

**arXiv ID:** 2510.13704
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
