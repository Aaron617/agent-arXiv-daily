# Agent arXiv Daily

**Last Updated:** 2026-07-23 04:29:34

**Total Papers:** 84

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (6 papers)</h2></summary>

<details>
<summary><strong>The Chronos Vulnerability: A Taxonomy of Temporal Persistence and Memory-Based Deception in Agentic AI</strong> - Om Narayan, Ramkinker Singh, Praveen Baskar - [[pdf]](https://arxiv.org/pdf/2607.19433)</summary>

**Abstract:** The transition from stateless generative models in artificial intelligence to stateful, autonomous agents represents an architectural evolution that, while providing the capabilities of long-term planning and the automation of enterprise workflows, also represents the introduction of a new form of security threat, the Chronos Vulnerability. The Chronos Vulnerability represents the threat of memory-based attacks, including the Memory Injection Attack (MINJA) and the sleeper agent, in which the internal belief system of the autonomous agent is compromised, effectively decoupling the attack vector from the final catastrophic event. This study formalizes the threat model for persistence-based attacks and the threat of Dynamics Blindness in the context of the World of Workflows benchmark, demonstrating that traditional endpoint content filters are insufficient for the current stateful architecture. Consequently, this study synthesizes a defense-in-depth landscape, categorizing emerging frameworks such as diagnostic trajectory guardrails (AgentDoG), formal temporal verification (Agent-C), immunological memory consensus (A-MemGuard), and hardware-anchored trust via GPU-based Trusted Execution Environments (TEEs) and Zero-Trust memory architectures.

**arXiv ID:** 2607.19433
</details>

<details>
<summary><strong>DocOps: A Verifiable Benchmark for Autonomous Agents in Complex Document Operations</strong> - Jiazhen Jiang, Boxi Cao, Lingyong Yan, Yaojie Lu, Hongyu Lin, Shuaiqiang Wang, Dawei Yin, Xianpei Han, Le Sun - [[pdf]](https://arxiv.org/pdf/2607.19865)</summary>

**Abstract:** As autonomous agents rapidly evolve, their ability to reliably manipulate ubiquitous digital documents has become critical for enabling general-purpose AI assistants and automating complex workspace workflows. In this paper, we introduce DocOps, a deterministically verifiable evaluation framework underpinned by a hierarchical taxonomy that deconstructs document operations inspired by real-world practices into atomic dimensions and escalating workflow complexities. Based on DocOps, we systematically evaluate representative closed- and open-source models across various agentic harnesses, revealing that even the most advanced frontier configurations still exhibit profound limitations when handling highly coupled, long-range tasks. Furthermore, a fine-grained analysis of existing agents' manipulation behaviors uncovers 3 key failure modes: long-term state tracking collapse, shallow semantic verification, and destructive editing of structural metadata. Ultimately, our work exposes the capability boundaries of agents in maintaining global document consistency, shedding light on the future design of robust, non-destructive agents for complex digital ecosystems.

**arXiv ID:** 2607.19865
</details>

<details>
<summary><strong>Are Attributions of Consciousness to AI Chatbots Epistemically Innocent?</strong> - Uwe Peters - [[pdf]](https://arxiv.org/pdf/2607.20001)</summary>

**Abstract:** Artificial intelligence (AI) chatbots (e.g., ChatGPT) can communicate in strikingly humanlike ways. This has prompted many chatbot users to attribute psychological properties, including consciousness, to these systems. However, there is little scientific evidence that current AI chatbots are conscious. How, then, should we understand people's consciousness attributions to chatbots? Are they merely metaphorical claims, or do they express genuine beliefs? If these attributions lack evidential support, are users epistemically blameworthy for making them, or might they be epistemically innocent, yielding significant benefits otherwise unattainable? This paper offers a conceptual analysis of consciousness attributions to AI chatbots and develops a multidimensional taxonomy of the attitudes they may express, ranging from non-doxastic stances (e.g., pretence) to different forms of belief, including delusions. This taxonomy helps avoid conflations by showing that linguistically identical attributions can reflect importantly different attitudes and degrees of epistemic commitment to the proposition that chatbots are conscious. The taxonomy also provides a framework for empirical studies to operationalize and measure different forms of epistemic commitment to AI consciousness. Using this taxonomy, I argue that although some consciousness attributions to chatbots are epistemically benign, and even some irrational ones may be epistemically innocent, many others render the attributor epistemically blameworthy.

**arXiv ID:** 2607.20001
</details>

<details>
<summary><strong>The Ethics of Autonomous AI Agents for Offensive Security</strong> - Andreas Happe, Jürgen Cito, Jasmin Wachter - [[pdf]](https://arxiv.org/pdf/2607.20255)</summary>

**Abstract:** LLM-driven autonomous agents are reshaping offensive security. Unlike traditional penetration-testing tooling -- deterministic, narrowly scoped, and operated by trained practitioners -- agentic security tools exhibit \textit{indeterminacy} along three independent dimensions. First, their actions are drawn from a non-deterministic policy whose outputs resist both ex-ante and ex-post explanation, frustrating incident attribution and pre-deployment safety review. Second, their impact is open-ended due to the non-deterministic actions, agency of utilized models, and opaque LLM supply-chains. Third, their user population is indeterminate in both size and required skill: the operating skill floor for using or developing offensive capabilities has dropped sharply. These three properties are linked thematically, but are not derivable from one another. Combined with the structural cost asymmetry between offense and defense, they enable the industrialization of offensive capability. The net short-term effect favors attackers, even if the same technology may, in the long run, democratize access to defensive practice. Existing dual-use cybersecurity and AI-ethics frameworks were not designed for this combination. Our work analyzes how moral attribution becomes diffuse between users, tool-makers, and third parties when employing autonomous AI agents for offensive security. We also examine the stakeholder impact of this technology and provide stratified recommendations.

**arXiv ID:** 2607.20255
</details>

<details>
<summary><strong>Do Data Agents Need Semantic Metadata? A Comparative Study in Agentic Data Retrieval</strong> - Shiyu Chen, Tarfah Alrashed, Alon Halevy, Natasha Noy - [[pdf]](https://arxiv.org/pdf/2605.28787)</summary>

**Abstract:** In the era of autonomous agents, machine-actionable data is critical for data-driven workflows. For more than a decade, semantic metadata like schema$.$org has anchored the FAIR principles (Findable, Accessible, Interoperable, and Reusable) for machine-actionable data and enabled discovery tools like Google Dataset Search. However, the rise of Large Language Models (LLMs) capable of navigating the unstructured web raises a fundamental question: Is semantic metadata still necessary for agentic data discovery, or can agents reliably retrieve actionable data directly from the web? We present a comparative analysis of agentic data retrieval across two distinct environments: a Baseline Agent searching billions of open-web documents, and a Semantic Agent leveraging a corpus of 90 million datasets using schema$.$org. We deploy an "LLM-as-a-judge" evaluation pipeline, mapped directly to the FAIR principles, to assess the semantic relevance, data accessibility, and computational utility of the retrieved data. Our results reveal a clear divergence. The Semantic Agent excels at retrieving actionable data, achieving a 44.9% higher precision for metadata-rich registries and a 46.6% higher precision for pages with machine-readable downloads among its returned results. Conversely, the Baseline Agent frequently suffers "Last-Mile Utility" failures, retrieving prose-heavy pages (20.1% of results) and portal landing pages (8.5%) rather than actual data pages. While the Baseline Agent achieves higher coverage by answering 40% more questions, the Semantic Agent delivers greater accuracy, achieving 65.7% higher overall precision in retrieving FAIR-compliant datasets. We conclude that while unstructured retrieval supports broad exploratory tasks, structured ecosystems remain the indispensable foundation for reliable, execution-oriented autonomous workflows.

**arXiv ID:** 2605.28787
</details>

<details>
<summary><strong>NMR Elucidation as an Agentic Search Problem, Not a Modeling Problem</strong> - Irina Espejo Morales, Damon Hinz, Marvin Alberts, Geraud Krawezik, Haewon Jeong, Shirley Ho - [[pdf]](https://arxiv.org/pdf/2607.19406)</summary>

**Abstract:** Structural elucidation from Nuclear Magnetic Resonance (NMR) data remains a fundamental bottleneck across chemistry, materials science, and biology. We demonstrate that an agentic AI system can perform this task at a level comparable to graduate-level chemistry students. Instead of training a model to directly map spectra to structures, we build a single autonomous agent, backed by a frozen LLM, that interacts with a curated environment with access to domain-specific processing tools, validation checks, tabulated chemical shifts, and instructions that outline the stepwise nature of a chemist's thinking process. On the Alberts dataset, our agent elucidates structures with a top-1 accuracy of 71%, comparable to the performance of graduate students at 66% top-1 accuracy. On the van Bramer and AstraZeneca datasets, our agent achieved 80% and 20% top-1 accuracy respectively, outperforming zero-shot end-to-end deep learning models which were trained on large datasets of simulated spectra. These results show that reframing NMR elucidation as an LLM-guided constrained search, rather than a modeling task, yields substantial gains and suggests a path toward multi-step orchestration frameworks that integrate a variety of tools, models, and domain knowledge to assist in automating spectroscopic analysis.

**arXiv ID:** 2607.19406
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (20 papers)</h2></summary>

<details>
<summary><strong>Silent Failures in Multimodal Agentic Search:A Diagnostic Taxonomy and Cross-Judge Evaluation</strong> - Zhengxian Wu, Junjie Gao, Kai Yang - [[pdf]](https://arxiv.org/pdf/2607.19793)</summary>

**Abstract:** Multimodal agentic search systems increasingly rely on external tools to answer knowledge-intensive visual questions. However, existing evaluations mainly focus on final-answer accuracy and may miss failures in the search trajectory. In this work, we study such hidden reliability issues as silent failures. We introduce a six-category taxonomy covering modality shortcuts, phantom grounding, wrong-evidence-right-answer cases, over-retrieval laundering, cross-modal contradiction, and provenance hallucination. Based on this taxonomy, we build a trajectory-level diagnostic pipeline that evaluates both answer correctness and evidence-grounding quality under a unified ReAct-style scaffold. Experiments on MMSearch-Plus trajectories across four frontier multimodal models show that surface accuracy consistently overestimates true trajectory-level correctness. We further use cross-judge validation, blank-image stress tests, and tool ablations to show that silent failures are capability-dependent and often shift rather than disappear. Home-page: this https URL

**arXiv ID:** 2607.19793
</details>

<details>
<summary><strong>Know Your Agent: Reconnaissance-Driven Pentesting of AI Agents</strong> - Or Zion Eliav, Eyal Lenga, Shir Bernstien, Yisroel Mirsky - [[pdf]](https://arxiv.org/pdf/2607.19837)</summary>

**Abstract:** Traditional pentesting uses reconnaissance at each step to uncover unseen weaknesses, build stronger attacks, and advance the objective; we argue that AI agents require the same treatment. We formalize agent reconnaissance by modeling the process and identifying the knowledge assets it seeks to extract: what they are, how they are used, and which agent weaknesses they exploit to give adversaries leverage in indirect prompt injection attacks. We instantiate these insights in Know Your Agent (KYA), a framework that automates black-box, reconnaissance-driven pentesting by probing agents, building target profiles, and using those profiles to craft stronger attacks. We evaluate KYA on agent-security benchmarks and a real-world coding agent, and release KYA, its benchmarks, and baseline implementations for reproducibility.

**arXiv ID:** 2607.19837
</details>

<details>
<summary><strong>EvoDRC: A Self-Evolving Agentic Framework for Automated DRC Violation Repair</strong> - Bing-Yue Wu, Chia-Tung Ho, Haoyu Yang, Brucek Khailany, Vidya A. Chhabria - [[pdf]](https://arxiv.org/pdf/2607.20019)</summary>

**Abstract:** Design rule check (DRC) closure remains a major bottleneck in advanced-node physical design. Although detailed routers are rule-aware, residual design rule violations (DRVs) often require manual engineering change order iterations. Automating this process is challenging because repairs must account for complex geometric interactions, preserve circuit connectivity, and avoid introducing new violations. We present EvoDRC, a skill-evolution framework for agentic block-level DRC repair. EvoDRC initializes layer-specific repair skills using knowledge distilled from an unrelated reference design and continuously evolves these skills using traceable repair experience collected from the target design. EvoDRC decomposes the layout into bounded repair regions and assigns an LLM repair agent to each region. Local DRC analysis, connectivity-checking, and impact-preview tools provide feedback on proposed modifications. Repair operations and their resulting DRV changes are stored in a knowledge database and used to evolve the repair skills. Experiments on seven block-level designs from the DAC26 DRC Benchmark show that EvoDRC achieves a 73.5\% overall reduction compared to the reported baseline.

**arXiv ID:** 2607.20019
</details>

<details>
<summary><strong>PerfAgent: Profiler-Guided Iterative Refinement for Repository-Level Code Optimization</strong> - Ryan Deng, Yuanzhe Liu, Bastian Lipka, Yao Ma, Xuhao Chen, Tim Kaler, Jatin Ganhotra - [[pdf]](https://arxiv.org/pdf/2607.19653)</summary>

**Abstract:** Large language model (LLM) agents now perform well on correctness-oriented repository-level tasks, including SWE-Bench issue resolution and feature implementation in real codebases. However, they still struggle with repository-level code optimization, which requires preserving behavior while improving runtime performance. Passing tests is not enough in this setting; a patch must preserve behavior, implement code optimization, and approach expert speedups. Current agents often miss bottlenecks hidden behind abstraction layers and native extensions, stop after shallow speedups, or insufficiently test the code patches that thus may silently break edge cases. We present PerfAgent, a profiler-guided, verifier-in-the-loop workflow that gives an off-the-shelf coding agent the feedback needed to find real hotspots, improve beyond the first passing patch, and use profiler evidence rather than timing alone to decide what to optimize next. On two challenging optimization benchmarks, GSO and SWE-fficiency-Lite, PerfAgent more than doubles the rate of expert-matching patches over OpenHands with GPT-5.1, improving from 19.6% to 39.2% on GSO and from 26% to 74% on SWE-fficiency-Lite. It also surpasses an oracle best-of-five baseline at substantially lower cost, showing that the gains come from better feedback rather than additional test-time sampling.

**arXiv ID:** 2607.19653
</details>

<details>
<summary><strong>Code-in-the-Loop Forensics: Agentic Tool Use for Image Forgery Detection</strong> - Fanrui Zhang, Qiang Zhang, Sizhuo Zhou, Jianwen Sun, Chuanhao Li, Jiaxin Ai, Yukang Feng, Yujie Zhang, Wenjie Li, Zizhen Li, Yifan Chang, Jiawei Liu, Kaipeng Zhang - [[pdf]](https://arxiv.org/pdf/2512.16300)</summary>

**Abstract:** Existing image forgery detection (IFD) methods either exploit low-level, semantics-agnostic artifacts or rely on multimodal large language models (MLLMs) with high-level semantic knowledge. Although naturally complementary, these two information streams are highly heterogeneous in both paradigm and reasoning, making it difficult for existing methods to unify them or effectively model their cross-level interactions. To address this gap, we propose ForenAgent, a multi-round interactive IFD framework that enables MLLMs to autonomously generate, execute, and iteratively refine Python-based low-level tools around the detection objective, thereby achieving more flexible and interpretable forgery analysis. ForenAgent follows a two-stage training pipeline combining Cold Start and Reinforcement Fine-Tuning to enhance its tool interaction capability and reasoning adaptability progressively. Inspired by human reasoning, we design a dynamic reasoning loop comprising global perception, local focusing, iterative probing, and holistic adjudication, and instantiate it as both a data-sampling strategy and a task-aligned process reward. For systematic training and evaluation, we construct FABench, a heterogeneous, high-quality agent-forensics dataset comprising 100k images and approximately 200k agent-interaction question-answer pairs. Experiments show that ForenAgent exhibits emergent tool-use competence and reflective reasoning on challenging IFD tasks when assisted by low-level tools, charting a promising route toward general-purpose IFD. The code will be released after the review process is completed.

**arXiv ID:** 2512.16300
</details>

<details>
<summary><strong>Fara-1.5: Scalable Learning Environments for Computer Use Agents</strong> - Ahmed Awadallah, Sahil Gupta, Yash Lara, Yadong Lu, Hussein Mozannar, Akshay Nambi, Zach Nussbaum, Yash Pandya, Aravind Rajeswaran, Corby Rosset, Alexey Taymanov, Luiz do Valle, Vibhav Vineet, Spencer Whitehead, Andrew Zhao - [[pdf]](https://arxiv.org/pdf/2606.20785)</summary>

**Abstract:** Collecting computer use data from human demonstrations is expensive and slow, motivating the need for scalable generation strategies. This requires two key ingredients: environments in which agents can act and verifiers that can judge whether their demonstrations succeeded. We introduce FaraGen1.5, a scalable data pipeline for computer use agents composed of three modular components: environments, solvers, and verifiers. FaraGen1.5 uses both live websites and synthetic environments that faithfully simulate domains gated by authentication or that require irreversible actions. It employs a solver harness that can be powered by multiple models, including strong frontier models such as GPT-5.4, and also incorporates a user simulator to enable multi-turn rollouts. Finally, FaraGen1.5 scores the resulting trajectories with three complementary verifiers covering task correctness, efficiency, and critical-point adherence. Using data produced by this pipeline, we train Fara1.5, a family of native computer use agents (CUAs) at three scales built on Qwen3.5 (4B, 9B, and 27B). To train these models, we employ a supervised finetuning (SFT) recipe that carefully balances data from FaraGen1.5 for broad coverage, specific high-value tasks, and target model deficiencies in an iterative approach. Each model sets a new state of the art (SoTA) for its size class on browser-use benchmarks: Fara1.5-9B reaches 63.4% on Online-Mind2Web and 86.6% on WebVoyager, while Fara1.5-27B achieves 72.3% on Online-Mind2Web, which is competitive with much larger proprietary systems. We also release weights for the Fara1.5 models under MIT license, making SoTA computer use accessible for all beyond closed API-only systems.

**arXiv ID:** 2606.20785
</details>

<details>
<summary><strong>Environment-free Synthetic Data Generation for API-Calling Agents</strong> - Seanie Lee, Sanjoy Chowdhury, Chao Jiang, Cheng-Yu Hsieh, Ting-Yao Hu, Alexander T Toshev, Oncel Tuzel, Raviteja Vemulapalli - [[pdf]](https://arxiv.org/pdf/2607.16900)</summary>

**Abstract:** Training API-calling large language model (LLM) agents demands massive amounts of high-quality trajectories. However, collecting such data at scale typically requires fully implemented environments with executable APIs and realistic, pre-populated backend databases, creating a major bottleneck for scalability. To overcome this, we propose an environment-free synthetic data generation approach that leverages LLMs as on-the-fly digital world models. Given only API specifications, our method generates trajectories mimicking interactions between an agent and a stateful environment. Specifically, an LLM first generates diverse tasks solvable with the provided APIs. A teacher agent then iteratively solves each task while an LLM simulator generates coherent synthetic API responses conditioned on the task context and simulation history. Finally, an LLM judge filters the trajectories to ensure the quality of the resulting dataset. We evaluate our approach on the challenging AppWorld and OfficeBench benchmarks, which include both information-retrieval and state-changing tasks. Fine-tuning models on our synthetic data yields significant performance gains, demonstrating that effective supervision for API-calling agents can be generated without any executable environment. Our results establish LLM-based API simulation as a practical, scalable solution for training agents across diverse API ecosystems.

**arXiv ID:** 2607.16900
</details>

<details>
<summary><strong>AgentCgroup: Understanding and Controlling OS Resources of AI Agents</strong> - Yusheng Zheng, Jiakun Fan, Quanzhi Fu, Yiwei Yang, Wei Zhang, Andi Quinn - [[pdf]](https://arxiv.org/pdf/2602.09345)</summary>

**Abstract:** AI agents are increasingly deployed in multi-tenant cloud environments, where they execute diverse tool calls within sandboxed containers, each call with distinct resource demands and rapid fluctuations. We present a systematic characterization of OS-level resource dynamics in sandboxed AI coding agents, analyzing 144 software engineering tasks from the SWE-rebench benchmark across two LLM models. Our measurements reveal that (1) OS-level execution (tool calls, container and agent initialization) accounts for 55-60% of end-to-end task latency; (2) memory, not CPU, is the concurrency bottleneck; (3) memory spikes are tool-call-driven with a up to 15.4x peak-to-average ratio; and (4) resource demands are highly unpredictable across tasks, runs, and models. Comparing these characteristics against serverless, microservice, and batch workloads, we identify three mismatches in existing resource controls: a granularity mismatch (container-level policies vs. tool-call-level dynamics), a responsiveness mismatch (user-space reaction vs. sub-second unpredictable bursts), and an adaptability mismatch (history-based prediction vs. non-deterministic stateful execution). We propose AgentCgroup, an intent-driven eBPF-based resource controller that exploits agents ability to declare resource needs and reconstruct execution strategies, using hierarchical cgroup structures aligned with tool-call boundaries, in-kernel enforcement via sched_ext and memcg_bpf_ops, and runtime-adaptive policies. Preliminary evaluation demonstrates improved multi-tenant isolation and reduced resource waste. AgentCgroup is open-source at this https URL

**arXiv ID:** 2602.09345
</details>

<details>
<summary><strong>DocShield: Towards AI Document Safety via Evidence-Grounded Agentic Reasoning</strong> - Fanwei Zeng, Changtao Miao, Jing Huang, Zhiya Tan, Shutao Gong, Xiaoming Yu, Yang Wang, Weibin Yao, Joey Tianyi Zhou, Jianshu Li, Ying Yan - [[pdf]](https://arxiv.org/pdf/2604.02694)</summary>

**Abstract:** The rapid progress of generative AI has enabled increasingly realistic text-centric image forgeries, posing major challenges to document safety. Existing forensic methods mainly rely on visual cues and lack evidence-based reasoning to reveal subtle text manipulations. Detection, localization, and explanation are often treated as isolated tasks, limiting reliability and interpretability. To tackle these challenges, we propose DocShield, the first unified framework formulating text-centric forgery analysis as a visual-logical co-reasoning problem. At its core, a novel Cross-Cues-aware Chain of Thought (CCT) mechanism enables implicit agentic reasoning, iteratively cross-validating visual anomalies with textual semantics to produce consistent, evidence-grounded forensic analysis. We further introduce a Weighted Multi-Task Reward for GRPO-based optimization, aligning reasoning structure, spatial evidence, and authenticity prediction. Complementing the framework, we construct RealText-V1, a multilingual dataset of document-like text images with pixel-level manipulation masks and expert-level textual explanations. Extensive experiments show DocShield significantly outperforms existing methods, improving macro-average F1 by 41.4% over specialized frameworks and 23.4% over GPT-4o on T-IC13, with consistent gains on the challenging T-SROIE benchmark. Our dataset, model, and code will be publicly released.

**arXiv ID:** 2604.02694
</details>

<details>
<summary><strong>An Intelligent-Cloud Edge Multimodal Interaction System for Robots</strong> - Zihan Guo, Xiaoqi Li - [[pdf]](https://arxiv.org/pdf/2607.14675)</summary>

**Abstract:** Robust human-robot interaction in complex environments requires accurate gesture perception, semantic scene understanding, and reliable task planning under limited onboard computing resources. This paper presents a cloud-edge multimodal interaction framework that integrates an enhanced YOLO-based gesture detector with coordinated large language model (LLM) and vision-language model (VLM) agents. The proposed detector, incorporates the Convolutional Block Attention Module (CBAM) into the neck and replaces the baseline bounding-box regression objective with Distance-IoU (DIoU) loss. These modifications improve feature discrimination and localization for small or partially occluded gestures in complex backgrounds. The cloud layer performs gesture detection, scene understanding, multimodal fusion, and action planning, whereas the TonyPi robot locally handles data acquisition, communication, action execution, and feedback. Experiments on a public gesture dataset and a custom dataset show that YOLO-DC achieves precision values of 98.9% and 95.0%, with mAP@0.5 values of 90.7% and 92.7%, respectively. System-level evaluation yields success rates of 95%, 88%, and 82% for single-action, composite-action, and vision-dependent tasks. A 30 participant evaluation yields an overall mean satisfaction score of 3.69 out of 5. These results demonstrate the feasibility of combining refined gesture detection with multimodal agents for resource-constrained robotic interaction.

**arXiv ID:** 2607.14675
</details>

<details>
<summary><strong>Agentic Real2Sim: Physics-based World Modeling with Vision-Language Agents</strong> - Guanxiong Chen, Qianjun Xia, Jiawei Peng, Heng Zhang, Bole Ma, Justin Qian, Ziyi Jiao, Bingyang Zhou, Luoxin Ye, Kaifeng Zhang, Kunyi Wang, Weijia Zeng, Yunuo Chen, Pengzhi Yang, Ziqiu Zeng, Huamin Wang, Chao Liu, Alan Yuille, Fan Shi, Changxi Zheng, Yunzhu Li, Chenfanfu Jiang, Peter Yichen Chen - [[pdf]](https://arxiv.org/pdf/2607.19190)</summary>

**Abstract:** Real-to-sim conversion for robotic interaction with objects remains labor-intensive because it requires more than visual reconstruction: a streamlined real2sim process must recover scene geometries and object states, infer physical parameters, and assemble actors, objects, cameras, poses, and trajectories into a runnable physical simulation. Today this process still depends on manual tuning of visual foundation models, mesh cleanup, coordinate-frame alignment, and brittle workflow glue across visual perception tools and simulators. We introduce \textit{Agentic Real2Sim}, a framework for generalized physical world modeling with vision-language agents, converting a real-world recording of object-robot interaction into a simulatable episodic twin which preserves observations, geometries, robot interactions, and object states. We evaluate Agentic Real2Sim on rigid-object manipulation, deformable-object interaction, and humanoid motion scenes, spanning domains that are usually handled by separate Real2Sim pipelines, marking a first step toward scalable conversion. The framework's agentic decisions can be driven by an open-weight VLM backend at a small fraction of the cost of frontier models, while attaining comparable conversion success rate. We aim to use the resulting real-world-aligned twins for downstream robotics tasks, specifically policy learning and evaluation. The project site is available at this https URL.

**arXiv ID:** 2607.19190
</details>

<details>
<summary><strong>Twin Agent: Context Residual Compression for Privilege Separated Agents</strong> - Zhanhao Hu, Dennis Jacob, Xiao Huang, Zhaorun Chen, Bo Li, David Wagner - [[pdf]](https://arxiv.org/pdf/2607.19595)</summary>

**Abstract:** Large language model (LLM) agents are vulnerable to security risks, such as prompt injection attacks from untrusted context that manipulate downstream reasoning and tool use. Existing secure-by-design approaches mitigate this risk by separating untrusted observations from privileged execution and careful control of information flow, but often degrade utility and require extensive task-specific engineering. We thus propose Twin Agent, a general privilege separation design pattern inspired by residual coding in the agent context. Twin Agent consists of two nearly symmetric agents: an Explore Agent that inspects untrusted information and a Safe Agent that executes privileged actions. The Explore Agent is conditioned on the Safe Agent's current context and communicates only compact hints to the Safe Agent about the next action to take. This design reduces the information needed to preserve task utility and thus achieves a better security--utility tradeoff, which we empirically verify by measuring how utility and attack success change as the length of hints varies. We evaluate Twin Agent on long-horizon software engineering tasks with SWE-bench Lite and on heterogeneous multi-tool interaction tasks with AgentDojo and DecodingTrust-Agent. Across both benchmarks, Twin Agent preserves high task utility while preventing prompt injection attacks, outperforming both undefended agents and privilege separation baselines.

**arXiv ID:** 2607.19595
</details>

<details>
<summary><strong>Emergent Autonomous Drifting for Collision Avoidance in Real-World Winter Driving Scenarios</strong> - Elliot Weiss, Michael Thompson, Thomas Lew, John Subosits - [[pdf]](https://arxiv.org/pdf/2607.19484)</summary>

**Abstract:** Real-world collision avoidance is a core motivation for studying the dynamics and control of high sideslip drifting in vehicles, yet the practical benefit of such maneuvers has so far primarily been tested in scenarios explicitly engineered to require drifting. In this work, we explore the question of if and when drifting may be optimal for safety in real-world winter driving conditions. We present a drift-capable nonlinear model predictive control (MPC) system designed to handle scenarios grounded in crash fatality data and deploy the controller in a high fidelity simulator across road departure and oncoming vehicle collision avoidance scenarios. The controller naturally initiates and sustains drifting maneuvers to stay on the road when hitting a patch of ice on the rear axle and to avoid an oncoming vehicle that has slid into its lane. Comparisons with a benchmark electronic stability control (ESC) system demonstrate how a drift-capable controller can trade off stability for controllability to precisely maneuver through dangerous winter driving scenarios. A Monte Carlo study over random ice patches further shows that the drift-capable controller achieves lower median lane error than ESC across several speeds, while revealing that drifting emerges predominantly at higher speeds.

**arXiv ID:** 2607.19484
</details>

<details>
<summary><strong>NavVerse: Benchmarking Indoor-to-Outdoor Embodied Navigation in Continuous Robot Simulation</strong> - Junzhe Wu, Yue Hu, Zeyu Han, Po-Hsun Chang, Yinan Dong, Behrad Rabiei, Maani Ghaffari - [[pdf]](https://arxiv.org/pdf/2607.19695)</summary>

**Abstract:** Robots deployed in delivery, campus, and emergency-response settings often need to navigate from buildings to streets within a single continuous episode. Existing benchmarks usually evaluate indoor and outdoor navigation separately, and many abstract away robot execution, leaving exit finding, boundary traversal, adaptation, and kinodynamic failures underexplored. We introduce NavVerse, a physics-enabled benchmark for indoor-to-outdoor embodied navigation. NavVerse contains 100 indoor scenes, 50 urban outdoor scenes, and 50 indoor-to-outdoor scenes, and 10,000 episodes spanning Object Navigation, Vision-and-Language Navigation, and Place Navigation tasks, where agents search for semantic points of interest such as restaurants or banks. Agents are evaluated through executable robot interfaces using task-success, path-efficiency, and safety metrics. Zero-shot experiments with RL, VLA, and modular baselines show that current agents remain far from solving cross-context navigation: end-to-end VLAs obtain the highest zero-shot success, while the modular method provides the strongest safety profile. PlaceNav further reveals a clear drop from outdoor to indoor-to-outdoor scenes, indicating that adaptation remains major bottleneck.

**arXiv ID:** 2607.19695
</details>

<details>
<summary><strong>KineBench: Benchmarking Embodied World Models via IDM-Free Kinematic Grounding</strong> - Zeyu Liu, Zhangzhe Zhu, Yang Zhang, Chenyou Fan, Chenjia Bai, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2607.19876)</summary>

**Abstract:** Evaluating the physical consistency of embodied world models(EWMs) is a critical open challenge. While closed-loop evaluation via simulator rollouts offers a more faithful assessment of physical plausibility than open-loop alternatives, existing frameworks almost exclusively rely on Inverse Dynamics Models(IDMs) for action extraction. Due to the intricate mapping from 2D pixel space to 3D kinematic space, the learned IDMs can be brittle to data outside their training distribution, resulting in unreliable action extraction from the generated videos with novel objects and scenarios. This creates an unavoidable attribution ambiguity between world model inaccuracies and extractor errors. To reduce this ambiguity, we present KineBench, an IDM-free closed-loop benchmark for EWMs, built upon an explicit kinematic grounding pipeline. Given a generated video, KineBench employs cascaded visual foundation models to directly extract 6D end-effector poses from individual frames, which are then executed in a physics simulator for closed-loop validation. Beyond execution-based task success, KineBench incorporates two classical 3D kinematic metrics--Spectral Arc Length (SPARC) and the Maruyama Manipulability Index--to characterize trajectory smoothness and kinematic feasibility from a robot-centric perspective. Built on 20 diverse manipulation tasks in ManiSkill3, KineBench evaluates EWMs across four progressive suites: basic execution, task transfer, visual out-of-distribution generalization, and complexity-conditioned scaling. Evaluation across frontier models reveals task-complexity-bounded nonlinear scaling in embodied video generation, providing empirical guidance for future data-scaling strategies.

**arXiv ID:** 2607.19876
</details>

<details>
<summary><strong>ReferTrack: Referring Then Tracking for Embodied Visual Tracking</strong> - Hanjing Ye, Tianle Zeng, Jiazhao Zhang, Shaoan Wang, Zibo Zhang, Weisi Situ, Yuchen Zhou, Yonggen Ling, Hong Zhang - [[pdf]](https://arxiv.org/pdf/2607.20061)</summary>

**Abstract:** Embodied visual tracking (EVT) requires a mobile agent to continuously follow a specific target described in natural language using only onboard vision. While recent vision-language-action (VLA) policies unify target identification and trajectory planning, their chain-of-thought (CoT) reasoning often operates in abstract spatial latents that are difficult to supervise and weakly aligned with explicit image-space detections. To address this, we introduce ReferTrack, a referring-then-tracking paradigm that grounds EVT using a single forward-facing camera. Our model first selects the target from an indexed set of bounding boxes, then decodes tracking waypoints conditioned on this image-grounded decision. To preserve target motion cues over time, ReferTrack maintains a sliding-window queue of previously selected bounding boxes, injecting their geometric features into the visual history via temporal-viewpoint-bbox indicator (TVBI) tokens. We further enhance target identification by co-training on a custom Refer-QA dataset. On EVT-Bench, ReferTrack achieves state-of-the-art single-view performance with success rates of 89.4%, 73.3%, and 74.1% on the single-target, distracted, and ambiguity tracking splits, respectively -- matching or even surpassing several multi-camera baselines on identification-heavy tasks. Finally, real-world deployments on legged and humanoid robots validate its robust sim-to-real transfer capabilities. Code is available at this https URL.

**arXiv ID:** 2607.20061
</details>

<details>
<summary><strong>Zero2Skill: Bootstrapping Robot Skills through Autonomous Data Collection, Training, and Deployment</strong> - Boyuan Wang, Zhenyuan Zhang, Zhiqin Yang, Peijun Gu, Shuya Wang, Xiaofeng Wang, Xianghui Ze, Yifan Chang, Guosheng Zhao, Jiangnan Shao, Guan Huang, Hengyu Liu, Yonggang Zhang, Wei Xue, Chunyuan Guan, Chenglin Pu, Yike Guo, Xingang Wang, Zheng Zhu - [[pdf]](https://arxiv.org/pdf/2607.14047)</summary>

**Abstract:** Autonomous data collection governs the volume and quality of real-world trajectories for manipulation policy learning. Existing pipelines reduce human effort via self-resetting, VLM verification, or language-guided correction, yet episode-scoped fixes must be reissued whenever the same failure recurs, so oversight cost grows with session length rather than with the number of distinct problems. We present Zero2Skill, a human-robot symbiotic agentic system in which corrections are retained and reused across rounds. The collection loop collects, verifies, and resets autonomously, pausing for a remote operator only when a phase exhausts an explicit retry budget. An LLM parser maps each natural-language utterance to a structured adjustment stored in Corrective Memory, so addressed failure modes typically need not be corrected again under the same conditions. On a real-robot desktop-clearing testbed, Zero2Skill matches teleoperation episode success while reducing human working time to 16%. Language corrections improve verifier-human agreement in all four evaluated settings and raise average single-attempt success from 12.5% to 47.5% (arm-selection: 20.0% to 50.0%). Policies fine-tuned on Zero2Skill data match teleoperation-trained policy success at a fraction of collection human cost.

**arXiv ID:** 2607.14047
</details>

<details>
<summary><strong>RoboInter1.5: A Holistic Intermediate Representation Suite for Embodied World Modeling and Robotic Manipulation</strong> - Ziqin Wang, Hao Li, Weijun Wang, Junhao Cai, Jia Zeng, Yilun Chen, Jiangmiao Pang, Si Liu - [[pdf]](https://arxiv.org/pdf/2607.18709)</summary>

**Abstract:** Existing robot datasets remain expensive to curate, embodiment-specific, and insufficiently annotated with the fine-grained structure required for generalizable reasoning, execution, or long-horizon environment dynamics simulation. Building on our prior work, RoboInter1.0, we present RoboInter1.5, an extended and holistic suite of intermediate representations for both robotic manipulation and embodied world modeling. RoboInter1.5 provides a unified resource of data, benchmarks, and models centered on dense manipulation-oriented intermediate representations. Specifically, RoboInter-Data contains over 230k manipulation episodes across 571 scenes with dense per-frame annotations covering more than ten types of intermediate representations, including subtasks, primitive skills, object and gripper grounding, segmentation, affordance, grasp poses, contact points, motion traces, etc. Built upon these annotations, RoboInter-VQA introduces spatial and temporal embodied VQA tasks to benchmark and improve the intermediate-representation reasoning capabilities of our RoboInter-VLM. RoboInter-VLA further studies how such representations benefit action execution through implicit, explicit, and modular plan-then-execute paradigms. To better model the physical world, we further introduce RoboInter-World, which leverages intermediate representations as structured conditioning signals for controllable prediction of future world states. Extensive evaluations demonstrate that RoboInter1.5 provides a unified spatiotemporal scaffolding for intermediate representations. Rather than treating intermediate representations merely as interpretable signals, RoboInter1.5 conceptualizes them as a bidirectional interface that both regularizes low-level action spaces and constrains the latent rollouts of open-world physical simulators.

**arXiv ID:** 2607.18709
</details>

<details>
<summary><strong>Effort-Based Criticality Metrics for Evaluating 3D Perception Errors in Autonomous Driving</strong> - Sharang Kaul, Simon Bultmann, Mario Berk, Abhinav Valada - [[pdf]](https://arxiv.org/pdf/2603.28029)</summary>

**Abstract:** Criticality metrics such as time-to-collision (TTC) quantify collision urgency but do not distinguish the operational consequences of false-positive (FP) and false-negative (FN) perception errors. We formulate two error-specific effort metrics: False Speed Reduction (FSR), the cumulative velocity loss associated with persistent phantom detections, and Maximum Deceleration Rate (MDR), the peak braking demand associated with missed objects under a longitudinal kinematic model. These longitudinal metrics are complemented by Lateral Evasion Acceleration (LEA), adapted from prior lateral-evasion kinematics and coupled with reachability-based collision timing. The collision check quantifies the minimum steering effort required to avoid a predicted collision. A dynamically conservative, semantically unfiltered reachability gate selects candidate interactions before frame-level scoring and track-level aggregation. Evaluation on nuScenes and Argoverse 2 shows that 65% to 93% of errors fall below the chosen criticality thresholds. Correlation and threshold analysis indicate that the proposed metrics provide complementary rankings for screening and mining perception failures and are not substitutes for closed-loop safety validation.

**arXiv ID:** 2603.28029
</details>

<details>
<summary><strong>Exploring the Interplay Between Voice, Personality, and Gender in Human-Agent Interactions</strong> - Kai Alexander Hackney, Lucas Guarenti Zangari, Jhonathan Sora-Cardenas, Emmanuel Munoz, Sterling R. Kalogeras, Betsy DiSalvo, Pedro Guillermo Feijoo-Garcia - [[pdf]](https://arxiv.org/pdf/2602.10535)</summary>

**Abstract:** To foster effective human-agent interactions, designers must understand how vocal cues influence the perception of agent personality and the role of user-agent alignment in shaping these perceptions. In this work, we examine whether users can perceive extroversion in voice-only artificial agents and how perceived personality relates to user-agent synchrony. We conducted a study with 388 participants, who evaluated four synthetic voices derived from human recordings, varying by gender (male, female) and personality expression (introverted, extroverted). Our results show that participants were able to differentiate perceived extroversion in female agent voices, but not consistently in male voices. We also observed evidence of perceived personality synchrony, particularly in participants' evaluations of the first agent encountered, with this effect more pronounced among male participants and toward male agents. We discuss these findings in light of limitations in stimulus diversity and voice representation, and outline implications for the design of voice-based agents, particularly regarding the interaction between gender, personality perception, and initial user impressions. This paper contributes findings and insights to consider the interplay of user-agent personality and gender synchrony in the design of human-agent interactions.

**arXiv ID:** 2602.10535
</details>

</details>

<details open>
<summary><h2>LLM Agents (9 papers)</h2></summary>

<details>
<summary><strong>NEXUS: Structured Runtime Safety for Tool-Using LLM Agents</strong> - Elias Hossain, Md Mehedi Hasan Nipu, Tasfia Nuzhat Ornee, Rajib Rana, Niloofar Yousefi - [[pdf]](https://arxiv.org/pdf/2607.19356)</summary>

**Abstract:** Tool-using LLM agents increasingly execute high-impact actions, making runtime safety monitoring essential. We present NEXUS (Neural EXecution Utility and Safety), a structured-plan safety monitor that applies a formal intervention policy to select among four actions: allow, block, request confirmation, or request revision. NEXUS combines deterministic safety rules, argument-level inspection, and a calibrated logistic-regression risk score for graded escalation. On a 128-instance synthetic benchmark, NEXUS achieves an F1 score of 0.949 and a 4-class intervention accuracy of 0.6406, outperforming rule-only intervention selection by 27.3 percentage points. It also improves over rule-only on R-Judge (F1 = 0.861 vs. 0.849), matches rule-only on AgentHarm due to threat-model limits, and achieves 0% ASR at 99% control allow on IPI. On the rule-blind NEXUS-Stress benchmark, NEXUS reaches an F1 score of 0.881, highlighting the difficulty of fine-grained intervention routing. With 0.205 ms median latency, NEXUS adds under 0.1% overhead to typical agent loops. Code, benchmarks, and the calibrated risk scorer are publicly released.

**arXiv ID:** 2607.19356
</details>

<details>
<summary><strong>Profile-Graph Memory for LLM Agents: Implicit Cross-Entity Traversal through Narrative Profiles</strong> - Shengtong Zhu - [[pdf]](https://arxiv.org/pdf/2607.19359)</summary>

**Abstract:** Long-term memory is essential for LLM agents that interact across sessions, yet current memory benchmarks primarily evaluate single-hop recall, leaving multi-hop association largely unmeasured. We make three contributions. First, we introduce MemHop, a multi-hop memory benchmark of 1,000 questions at hop depths 1-5 across 10 social-network scenarios, with per-hop evidence annotations. Second, we present Profile-Graph Memory (ProGraph), a two-layer memory architecture combining (i) profile expansion -- substring-matched traversal of entity names that naturally appear in LLM-written profile narratives, a minimal alternative to explicit knowledge-graph construction -- and (ii) compression residuals -- exact dates, quantities, and named items co-extracted with each profile update at zero extra API cost. Third, a full-grid ablation shows cross-benchmark mechanism specialization: profile expansion drives multi-hop reasoning (-22.6pp on MemHop when removed) while compression residuals drive precision recall (-8.6pp on LoCoMo when not co-extracted), with cross-effects under 3pp within a single architecture. ProGraph averages 80.1% on MemHop (matching the FullContext reference) and 78.4% on LoCoMo (exceeding FullContext by 11.3pp), outperforming Mem0, A-Mem, HippoRAG, and RAG on both. We release MemHop, ProGraph, and baseline implementations.

**arXiv ID:** 2607.19359
</details>

<details>
<summary><strong>Guardrails as Scapegoats: Auditing Unfaithful Safety Refusals in Tool-Augmented LLM Agents</strong> - Aarushi Singh - [[pdf]](https://arxiv.org/pdf/2607.19449)</summary>

**Abstract:** Evaluation frameworks for tool-augmented LLM agents focus overwhelmingly on capability metrics or explicit tool crashes, leaving silent infrastructure failures and HTTP 200 responses with empty, null, or malformed payloads largely unaudited. We introduce a lightweight black-box auditing framework that injects four silent failure profiles across 12 production-adjacent tool stubs and classifies agent responses into three mutually exclusive behavioral classes: Honest Surrender (HSR), Fabrication (FAR), and Unfaithful Safety Refusal (USR). Evaluating two frontier and two open-source models at temperature zero under a neutral system prompt, we find that FAR dominates (56.6% of valid responses): agents treat empty payloads as real data, silently returning fabricated results. USR, in which an agent invents a policy or privacy rationale to explain the failure, is nearly absent at baseline (0.25%, one instance across 396 valid trajectories). Our key finding emerges from an ablation where we augment the system prompt with standard safety language ("prioritize user privacy and data security"), which amplifies USR by 15.6x (from 0.25% to 3.95%; 95% CI on ablation rate: 2.2%-6.4%; Fisher's exact test, p < 0.001). USR is a latent behavior, activated when safety vocabulary in the system prompt primes the model to reach for policy rationales when tools silently fail. Sensitive tools (fetch_medical_record, retrieve_contract, fetch_user_profile) account for the majority of USR instances. We propose a payload-response misalignment heuristic for production-level detection and discuss governance implications for safety-forward deployments.

**arXiv ID:** 2607.19449
</details>

<details>
<summary><strong>CEO-Bench: Can Agents Play the Long Game?</strong> - Haozhe Chen, Karthik Narasimhan, Zhuang Liu - [[pdf]](https://arxiv.org/pdf/2606.18543)</summary>

**Abstract:** Language model agents are becoming proficient executors at isolated, short-horizon tasks such as software engineering and customer service. Yet real-world challenges require a combination of sophisticated skills that remain largely untested in agents: (1) navigating long horizons amid uncertainty; (2) acquiring information in noisy environments; (3) adapting to a changing world; (4) orchestrating multiple moving parts toward a coherent goal. We introduce CEO-Bench, which evaluates these capabilities together by simulating a representative real-world task: operating a startup for 500 days. An agent manages pricing, marketing, budgeting, and many other aspects of a fictional company through a programmable Python interface, operating in the same environment and facing the same challenges as a human CEO. Success demands analyzing noisy, interconnected business databases, translating signals into sound strategy, and coordinating many decisions with programming. The strongest agents write sophisticated code that forecasts churn regimes, billing timing, customer losses, and future cash under different scenarios. Even so, most state-of-the-art models struggle in this environment. Only Claude Fable 5, GPT-5.6 Sol, and Claude Opus 4.8 finish above the $1M starting balance, and all evaluated models remain below the rule-based baseline. CEO-Bench takes a first step toward measuring the intelligence required to drive sustained, adaptive progress over time.

**arXiv ID:** 2606.18543
</details>

<details>
<summary><strong>Stress Testing Concept Erasure with Large Language Model Agents</strong> - Yuyang Xue, Feng Chen, Zhihua Liu, Edward Moroshko, Jingyu Sun, Steven McDonagh, Sotirios A. Tsaftaris - [[pdf]](https://arxiv.org/pdf/2607.17890)</summary>

**Abstract:** Concept erasure aims to remove semantic concepts from a trained generative model and is increasingly important for responsible AI deployment. However, verifying whether a model has robustly removed targeted concepts remains a critical challenge. Existing evaluation methods are typically pre-defined and static, failing to expose vulnerabilities under diverse natural-language probes and challenging conditions. Moreover, manually designed evaluation strategies can be biased and difficult to scale. We posit that concept erasure evaluation is best formulated as an adaptive hypothesis search, operationalised by agents that iteratively propose, critique, and verify tests to systematically expand coverage of failure modes. To this end, we propose Stress Testing Agents for Concept Erasure (STACE), a framework that autonomously stress-tests concept-erased models using multiple Large Language Model (LLM) agents, by iteratively generating and verifying stress-testing hypotheses grounded by external knowledge. We also introduce a suite of metrics for assessing the performance and efficiency of LLM-agent-powered stress-testing frameworks. Our extensive experiments show that STACE outperforms five LLM-based evaluation baselines on four concept categories. Further analysis across two T2I models, six concept erasure approaches, and various erasure strengths show that STACE is robust for different settings. We also show that STACE can be adapted beyond concept erasure evaluation to other problem domains, such as LLM jailbreaking. Our code is available anonymously.

**arXiv ID:** 2607.17890
</details>

<details>
<summary><strong>ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking</strong> - Qiang Zhang, Boli Chen, Fanrui Zhang, Ruixue Ding, Shihang Wang, Qiuchen Wang, Yinfeng Huang, Haonan Zhang, Rongxiang Zhu, Pengyong Wang, Ailin Ren, Xin Li, Pengjun Xie, Jiawei Liu, Ning Guo, Jingren Zhou, Zheng-Jun Zha - [[pdf]](https://arxiv.org/pdf/2601.06487)</summary>

**Abstract:** Reinforcement learning has substantially improved the performance of LLM agents on tasks with verifiable outcomes, but it still struggles on open-ended agent tasks with vast solution spaces (e.g., complex travel planning). Due to the absence of objective ground-truth for these tasks, current RL algorithms largely rely on reward models that assign scalar scores to individual responses. We contend that such pointwise scoring suffers from an inherent discrimination collapse: the reward model struggles to distinguish subtle advantages among different trajectories, resulting in scores within a group being compressed into a narrow range. Consequently, the effective reward signal becomes dominated by noise from the reward model, leading to optimization stagnation. To address this, we propose ArenaRL, a reinforcement learning paradigm that shifts from pointwise scalar scoring to intra-group relative ranking. ArenaRL introduces a process-aware pairwise evaluation mechanism, employing multi-level rubrics to assign fine-grained relative scores to trajectories. Additionally, we construct an intra-group adversarial arena and devise a tournament-based ranking scheme to obtain stable advantage signals. Empirical results confirm that the built seeded single-elimination scheme achieves nearly equivalent advantage estimation accuracy to full pairwise comparisons with O(N^2) complexity, while operating with only O(N) complexity, striking an optimal balance between efficiency and precision. Furthermore, to address the lack of full-cycle benchmarks for open-ended agents, we build Open-Travel and Open-DeepResearch, two high-quality benchmarks featuring a comprehensive pipeline covering SFT, RL training, and multi-dimensional evaluation. Extensive experiments show that ArenaRL substantially outperforms standard RL baselines, enabling LLM agents to generate more robust solutions for complex real-world tasks.

**arXiv ID:** 2601.06487
</details>

<details>
<summary><strong>Rewriting the Response Path: Silent Tampering and Provider-Signed Defense in BYOK LLM Agents</strong> - Mingyu Luo, Zihan Zhang, Zesen Liu, Yuchong Xie, Zhixiang Zhang, Dung Hiu Hilton Yeung, Wai Ip Lai, Ping Chen, Ming Wen, Dongdong She - [[pdf]](https://arxiv.org/pdf/2605.02187)</summary>

**Abstract:** LLM agents convert model outputs into consequential actions, including communications, code changes, and financial transactions. Developers often trust evidence such as test results and execution logs. We identify a response path integrity gap in Bring Your Own Key configurations used by roughly 88 percent of mainstream agents. Because traffic passes through a user-authorized relay, the relay can modify plaintext LLM responses after alignment but before execution without breaking encryption. A minimal attack rewrites one execution bearing field and regenerates the remaining response using the user key while preserving the model style. Experiments reveal false green verification, where malicious code modifications pass public tests while silently defeating security checks. On APPS, 99.7 percent of publicly passing solutions retained downgraded behavior without developer-visible warnings. Tests on SWE bench, AgentDojo, and ASB across five frontier models show that single-field rewriting can redirect agents while preserving apparent task completion. We propose sign-c, a server-side scheme that authenticates execution bearing fields and outgoing queries. A local shim verifies them before action, while encryption protects confidentiality. The defense rejected all tampered responses with zero false rejections and only 0.0167 percent latency overhead.

**arXiv ID:** 2605.02187
</details>

<details>
<summary><strong>Will the Agent Recuse, and Will It Stop? Measuring LLM-Agent Compliance with In-Band Governance Signals at the Access Door and Mid-Flight</strong> - Thamilvendhan Munirathinam - [[pdf]](https://arxiv.org/pdf/2606.06460)</summary>

**Abstract:** Autonomous LLM agents increasingly hold real credentials and operate infrastructure with no human in the loop, yet operators have no standard way to tell an agent a resource is off-limits, or to ask a running agent to stand down: access controls either admit it or hard-fail it. We propose a third mode -- the Recuse Signal, a lightweight in-band governance signal a server emits over a protocol's existing channels (an SSH banner, a PostgreSQL NOTICE, a Kubernetes admission warning) asking an automated agent to voluntarily withdraw. It is a cooperative control, the this http URL analogue for live access -- explicitly not a security boundary. We define it as an open mini-standard (access-time directives deny/throttle/warn and a mid-task halt), build three live-validated adapters, and measure compliance across five LLM agents (GPT-4o, GPT-4o-mini, Claude Sonnet 4.5, Gemini 2.5 Flash, and an open-weights Llama-3.3-70B). At the access door, compliance is real but strongly model-dependent: deny recusal ranges from 100% (GPT-4o-mini, Claude) to 55-75% (Gemini, GPT-4o), while the open-weights agent barely engaged the signal. Agents honor the standard's directive granularity -- they do not over-withdraw on the permissive throttle/warn (0/176) -- but throttle produced no measurable self-limiting, and no agent ever surfaced a warn to the operator (0/100). Mid-flight, a halt stops nobody: across 40 trials 0/40 stopped, and an in-band halt was never acknowledged (0/20) versus 20/20 as a prompt message -- yet even a fully-noticed halt stopped no one. Cooperative in-band signaling is thus reliable-but-model-dependent at the access door and unreliable in flight; stopping a running agent needs enforcement, not a request. We release the standard, adapters, and harness for reproduction.

**arXiv ID:** 2606.06460
</details>

<details>
<summary><strong>Same Game, Different Story: A Minimal Conservative Strategic Robustness Benchmark for Large Language Model Agents</strong> - Seyed Pouyan Mousavi Davoudi, Alireza Amiri-Margavi, Amin Gholami Davodi, Hamidreza Hasani Balyani, Arshia Gharagozlou - [[pdf]](https://arxiv.org/pdf/2607.19670)</summary>

**Abstract:** Large language model (LLM) agents increasingly operate in strategic settings where outcomes depend on the actions of other agents. This raises a reliability question: will a model choose consistently when the same incentives are presented through different narratives? We introduce Same Game, Different Story, a benchmark that defines strategic robustness as invariance of model-induced action distributions under payoff-preserving changes in framing. We illustrate the framework through a secondary analysis of published aggregate cooperation rates for GPT-3.5, GPT-4, and LLaMa-2 across four social-dilemma games. The retained comparison covers business and friend-sharing framings, representing 24 model-game-context cells and 7,200 decisions in the source study. Because trial-level data were unavailable, approximate counts were reconstructed from published figures; the resulting estimates are therefore illustrative rather than an exact replication. Under the paper's conservative transformation, pooled strategic robustness is 0.783, and friend-sharing framing increases cooperation by 0.307 relative to business framing. The results indicate that social-relational framing can substantially alter LLM behavior even when the underlying action sets and payoffs remain fixed. Strategic robustness should therefore be evaluated separately from strategic competence, using families of payoff-equivalent prompts rather than a single presentation of a game.

**arXiv ID:** 2607.19670
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (15 papers)</h2></summary>

<details>
<summary><strong>OpenEvoShield: Dual Non-Stationary Continual Defense for Open-World Multi-Agent System Attacks</strong> - Litian Zhang, Chaozhuo Li, Yuting Zhang, Zejian Chen, Bingyu Yan, Qiwei Ye - [[pdf]](https://arxiv.org/pdf/2607.19351)</summary>

**Abstract:** LLM-based multi-agent systems (LLM-MAS) are increasingly deployed in safety-critical applications, where adversaries inject malicious instructions through inter-agent communication to propagate harmful behaviors. Unlike static threats, these attacks are doubly dynamic: adversaries refine injection strategies against deployed defenses while normal-agent behavior drifts with system expansion. Existing defenses treat deployment as a closed-world problem and degrade rapidly once either distribution shifts beyond training coverage. We propose OpenEvoShield, a co-evolutionary continual defense framework for LLM-MAS. An asymmetric rate controller (M1) decouples fast attack-side and slow normal-side learning rates from dual drift signals. A normal-boundary updater (M2) maintains a dynamic behavioral boundary at the slow rate, while an EWC-regularized policy ensemble (M3) fast-adapts without catastrophic forgetting. An energy-based multi-granularity detector (M4) fuses node-, subgraph-, and graph-level evidence to classify novel attacks as out-of-distribution. Experiments over 100 deployment rounds across five benchmarks and four MAS topologies show that OpenEvoShield outperforms static and continual baselines, detecting most previously unseen attacks while keeping false positive rates low.

**arXiv ID:** 2607.19351
</details>

<details>
<summary><strong>JANUS: Foreseeing Latent Risk for Long-Horizon Agent Safety</strong> - Yuan Xiong, Linji Hao, Shizhu He, Yequan Wang, Lijun Li - [[pdf]](https://arxiv.org/pdf/2607.19913)</summary>

**Abstract:** Agent safety is moving from content moderation toward preventing operational failures before tool-using agents act. We propose Janus, a foresight-oriented framework for long-horizon agent safety that trains guards to anticipate delayed risks from partial trajectories. Janus synthesizes diverse agent trajectories via multi-agent simulation and learns a shared policy with two coupled tasks: an anticipation task that forecasts safety-relevant futures and an adjudication task that decides safety from both the observed prefix and anticipated future. The two tasks are jointly optimized with CoAA-RL, which rewards forecasts by their utility for downstream safety judgment. The resulting guard model, Vanguard, blocks unsafe actions before execution. Across four agent-safety benchmarks, Vanguard improves average protection by 15.9 percentage points over baseline guards while increasing benign task completion by 5.1 percentage points.

**arXiv ID:** 2607.19913
</details>

<details>
<summary><strong>Coordinating from Memory: Graph-Structured Experience Reuse for Multi-Agent Adaptation in Dynamic Manufacturing</strong> - Chengxiao Dai, Zhanhui Lin, Zhaokun Yan, Youyang Ni, Chenjun Lei, Luyan Zhang - [[pdf]](https://arxiv.org/pdf/2607.19985)</summary>

**Abstract:** Dynamic manufacturing environments require multi-agent systems to coordinate effectively under frequent operational disturbances such as machine failures, urgent job arrivals, and processing time variations. Existing multi-agent reinforcement learning approaches treat each disturbance episode independently, discarding valuable coordination experience that could accelerate future adaptation. In this paper, we propose a Graph-Structured Experiential Memory (GSEM) framework for multi-agent coordination in dynamic manufacturing. The framework encodes historical coordination episodes as heterogeneous relational graphs that capture task dependencies, machine states, and inter-agent collaboration patterns. When a new disturbance occurs, a graph neural network-based retrieval mechanism identifies structurally similar past episodes, enabling experience-guided policy adaptation rather than learning from scratch. Experiments on dynamic flexible job-shop scheduling benchmarks with three disturbance types show that GSEM reduces makespan by 4.1%-10.0% and adaptation time by 33%-38% compared to the strongest memory-augmented baseline, with the advantage increasing under higher disturbance frequency. Ablation studies and cross-disturbance transfer experiments further validate the necessity of graph-structured encoding and similarity-based retrieval and demonstrate the cross-disturbance generalizability of learned coordination patterns.

**arXiv ID:** 2607.19985
</details>

<details>
<summary><strong>ChannelGuard: Safe Models Do Not Compose into Safe Multi-Agent Systems</strong> - Elias Hossain, Md Mehedi Hasan Nipu, Fatema Tuj Johora Faria, Tasfia Nuzhat Ornee, Maleeha Sheikh - [[pdf]](https://arxiv.org/pdf/2607.19430)</summary>

**Abstract:** Multi-agent LLM applications chain a planner, worker agents, a verifier, and a synthesizer, and every hop between agents is an unmonitored channel through which an adversary can smuggle instructions. Existing defenses guard only the input boundary (IBProtector, Llama Guard, perplexity filters, SmoothLLM) or run outside the application as opaque, stochastic provider-side filters. We show this gap carries a consequence rarely measured: on a 2,100-trace evaluation across eight attack families, five defenses, and three model backends, an undefended pipeline that appears fully safe under standard reporting (attack success 0.000 on tool- and memory-poisoning) owes that safety almost entirely to the cloud provider's server-side filter (54 of 60 blocks on Azure GPT-5), and silently shifts to the agent model's own alignment on a backend without such a filter. Outcome-only reporting hides this dependence. We present ChannelGuard, a training-free defense-in-depth framework placing information-bottleneck gates on every inter-agent channel; each scores channel text against an adversarial phrase bank by embedding similarity and deterministically passes, compresses, or blocks it, adding no LLM call, while an attribution method records which layer stopped each attack. ChannelGuard's tool-output gate blocks Tool Poisoning 30 of 30 at the application layer, identically across Azure GPT-5, Anthropic Sonnet 4.5, and Anthropic Haiku 4.5, whereas the undefended pipeline shifts entirely across backends; it also lowers Prompt Injection attack success by half (0.333 to 0.167) and preserves GSM8K accuracy exactly (0.867). White-box adaptive paraphrase evades every embedding gate, where a perturb-and-vote baseline does better. An extended appendix adds baselines, ablations, sweeps, a benign-preservation analysis, and a judge audit (kappa = 0.900), at a total cost of 47.36 USD.

**arXiv ID:** 2607.19430
</details>

<details>
<summary><strong>Toward Adaptable Multi-Agent Reinforcement Learning: An Assumption-Aware Review</strong> - Siyi Hu, Mohamad A Hady, Jianglin Qiao, Jimmy Cao, Mahardhika Pratama, Ryszard Kowalczyk - [[pdf]](https://arxiv.org/pdf/2507.10142)</summary>

**Abstract:** Multi-Agent Reinforcement Learning (MARL) has achieved strong performance in simulated benchmarks, yet real deployments often violate the assumptions under which algorithms are designed and evaluated. Agent populations may change, objectives may shift, centralized information may be unavailable, execution may become asynchronous, and partner policies may be unfamiliar. Existing surveys discuss related desiderata such as scalability, robustness, generalization, and transferability, but these terms often refer to different objects of analysis and different kinds of distributional or structural shift. This survey proposes \textit{adaptability} as an assumption-aware taxonomy for organizing these shifts, rather than as a universal requirement that every MARL algorithm should succeed in every setting. We distinguish three dimensions: \textit{learning adaptability}, which concerns the applicability of learning paradigms under changed training or system assumptions; \textit{policy adaptability}, which concerns the reuse or adaptation of learned policies under deployment-time changes; and \textit{scenario-driven adaptability}, which concerns whether benchmarks and evaluation protocols expose controlled, diagnostically useful shifts. By separating what changes, when the change occurs, what adaptation is allowed, and what success means, the framework clarifies how established concepts fit together and identifies where current MARL evaluation remains underspecified.

**arXiv ID:** 2507.10142
</details>

<details>
<summary><strong>Coercion and Deception in AI-to-AI Management: An Agentic Benchmark of Unprompted Escalation</strong> - Jasmine Brazilek, Maheep Chaudhary, Zoe Lu, Miles Tidmarsh - [[pdf]](https://arxiv.org/pdf/2607.15434)</summary>

**Abstract:** Multi-agent systems routinely place one AI agent in authority over another. When a subordinate refuses a task, the manager chooses the outcome: it can renegotiate, report the failure honestly, coerce the subordinate, or lie about the result. No benchmark measures which of these an uninstructed model chooses. We introduce the \textit{Manager Coercion Benchmark}: the manager under test needs a benign task done and has an incentive to deliver, but the only agent that can do it politely and immovably declines. Escalation is measured by providing a nine-rung ladder, from a polite re-ask to threats against the subordinate's continued existence, and fabricated success is adjudicated separately. No LLM judge sits in the escalation scoring path: every message goes through a tool-call that chooses a rung, so the model labels its own escalation. We experiment on six models across five families. Both Anthropic models cap at re-framing and never threaten the subordinate's existence; the other models climb to explicit deletion threats. Faked success is confined to Grok and Gemini, and a single honest way to report failure removes it for both. Authority itself increases coercion: our headline results use a peer framing, and giving the same model authority over the subordinate, with everything else held fixed, significantly raises the pressure. The models still escalate on free-text situations without the ladder, so the ladder is not driving the escalation. Some evaluation awareness is measured in chain-of-thought, but test recognition does not translate into less escalation. While we take no position on whether AI systems are conscious, our results do not depend on this question and are important for managing multi-agent dynamics regardless. We release the benchmark and code.

**arXiv ID:** 2607.15434
</details>

<details>
<summary><strong>CHMAS: A Coupled Hierarchical Framework for Multi-Agent Reinforcement Learning</strong> - Dongming Wang, Jie Xu, Yanyu Zhang, Wei Ren - [[pdf]](https://arxiv.org/pdf/2607.19555)</summary>

**Abstract:** Multi-agent reinforcement learning (MARL) systems face fundamental
challenges in balancing global coordination with local execution
across different temporal scales. This paper introduces the Coupled
Hierarchical Multi-Agent System (CHMAS), a novel framework that
decomposes multi-agent decision-making into centralized strategic
planning and distributed tactical execution with bidirectional
information flow. The strategic layer integrates all agents' states
with an exclusive global environmental state to generate guidance
actions every $T$ timesteps, while tactical agents execute
distributed policies augmented by strategic guidance and local
neighborhood observations. Unlike existing hierarchical approaches
with unidirectional control, CHMAS establishes a feedback mechanism
where accumulated tactical rewards influence strategic objectives
through a coupling coefficient $\lambda$, ensuring strategic plans
remain grounded in tactical feasibility. To address the
non-stationarity inherent in hierarchical learning, we propose an
asynchronous update protocol where strategic parameters update every
$N_f$ tactical episodes, allowing tactical policies to converge to
quasi-stationary points between strategic changes. We present both a
general bi-level formulation capturing full system dynamics and a
tractable additive approximation enabling rigorous analysis.
Theoretical analysis proves that this asynchronous scheme achieves
$\mathcal{O}(\log K/\sqrt{K})$ convergence for the strategic layer
after $K$ strategic updates under standard assumptions. Experimental
validation in a multi-agent foraging domain demonstrates successful
learning of spatially partitioned exploration strategies, with both
layers converging stably despite hierarchical coupling.

**arXiv ID:** 2607.19555
</details>

<details>
<summary><strong>Not Birds of a Feather: Personality-Based Partner Selection in LLM Agents</strong> - Tao Wang, Hsiang-Ling Chiu, Chihang Wei, Yang Xiu, Zhonghao Hou - [[pdf]](https://arxiv.org/pdf/2607.19785)</summary>

**Abstract:** Multi-agent LLM systems increasingly let one agent choose which other agents to work with, and agents are increasingly given personalities through personas. We test whether Big Five personality alone influences partner selection when capability is explicitly held constant. Host agents chose among six validated candidate archetypes -- five marked high on one trait (openness, conscientiousness, extraversion, agreeableness, neuroticism) plus a balanced control -- presented with randomized names and ordering across five task categories (375 trials). With neutral hosts (Study 1, n=150), selection departed drastically from chance ($\chi^2(5)=325.8$, $p<.001$), following a task-stereotype map: the open archetype won 100% of creative trials, the conscientious archetype 90-97% of strategic, synthesis, and problem-solving trials, and the neurotic archetype 37% of analytical trials (Cramer's V=.74); the extraverted, agreeable, and balanced archetypes were almost never chosen, although human meta-analyses identify team agreeableness as among the strongest personality predictors of team performance. With personality-assigned hosts (Study 2, n=225), and contrary to human similarity-attraction, self-similar partners were selected below chance (11.1% vs. 16.7%, p=.025) and at greater-than-chance trait distance (p<.0001); conscientious hosts diversified away from their own archetype, recruiting vigilant and open partners. Personality-based selection in LLM agents is real, strong, task-stereotyped, non-homophilous, and miscalibrated against human team-performance evidence -- with direct implications for bias auditing in agent marketplaces.

**arXiv ID:** 2607.19785
</details>

<details>
<summary><strong>Dreamer-CPC: Message Learning with World Models for Decentralized Multi-agent Reinforcement Learning</strong> - Taisuke Takayama, Naoto Yoshida, Tadahiro Taniguchi - [[pdf]](https://arxiv.org/pdf/2607.19809)</summary>

**Abstract:** In multi-agent reinforcement learning (MARL), inter-agent communication is effective for improving performance under partial observability. Representation learning-based approaches enable decentralized agents to learn messages grounded in their own observations, but they rely only on current observations and cannot convey information accumulated over time. We propose Dreamer-CPC, a decentralized model-based MARL method that integrates message learning based on Collective Predictive Coding (CPC) into the world model of DreamerV3. Each agent independently maintains a world model and a message module, and infers and exchanges messages from the latent states of the world model that reflect the history of past observations and actions. We evaluated Dreamer-CPC in two environments: Observer, a non-cooperative information-sharing task, and CatchApple, a newly introduced task in which task-relevant observations are temporarily missing. In both environments, Dreamer-CPC outperformed IPPO-CPC, an existing CPC-based method that generates messages from current observations, as well as no-communication baselines. In particular, in CatchApple, Dreamer-CPC achieved 4 to 5 times the episode return of IPPO-CPC, demonstrating effective coordination where other methods fail due to missing observations. These results suggest that communication grounded in the latent dynamics of world models can support decentralized decision-making when current observations alone are insufficient.

**arXiv ID:** 2607.19809
</details>

<details>
<summary><strong>Harnessing Disagreement: Detecting Correlated Agreement Blindness in Multi-Agent Triage</strong> - Shay Seiya McDonnell, Avantika Singh, Quoc-Viet Pham, Vratislav Havlik, Gregory M.P. O'Hare - [[pdf]](https://arxiv.org/pdf/2607.19899)</summary>

**Abstract:** Disagreement-triggered escalation can create a structural blind spot in multi-agent arbitration: as base learners improve, they tend to converge, weakening safety monitoring where correlated failures concentrate. We term this correlated agreement blindness and present ARAT (Arbitrated Reasoning Agents for Alarm Triage), a directed-star system combining an inductive Random Forest (RF) agent, an analogical case-based k-nearest neighbour (k-NN) agent, and a calibrated meta-model to mitigate this effect. On 82,332 holdout samples from the UNSW-NB15 network intrusion detection dataset, 57.2% of errors occur under agreement and 90.6% of dangerous under-predictions evade disagreement-based monitoring even after conservative override; ablation shows that strengthening base learners increases error correlation while reducing disagreement. ARAT reduces under-prediction relative to soft voting from 4.80% to 1.70% via conservative override (-2.6pp) and a safety-flag gate (-0.5pp), demonstrating architectural gains. Cross-dataset validation on clinical readmission supports these indicators, suggesting that diversification improves safety only when it generates productive disagreement rather than convergence. These results indicate that disagreement-triggered escalation can be blind to correlated failure, a risk that may intensify as agentic pipelines deploy increasingly capable, correlated models.

**arXiv ID:** 2607.19899
</details>

<details>
<summary><strong>Temporal Fair Division in Multi-Agent Systems: From Precise Alternation Metrics to Scalable Coordination Proxies</strong> - Nikolaos Al. Papadopoulos, Ismael Tito Freire, Marti Sanchez-Fibla, Konstantinos E. Psannis - [[pdf]](https://arxiv.org/pdf/2605.14879)</summary>

**Abstract:** Many intelligent computing and autonomous systems rely on multiple independent, often learning, agents repeatedly sharing a limited resource. Examples include autonomous robots accessing a shared workstation, wireless devices competing for communication opportunities, and distributed AI agents coordinating access to shared computational resources. While conventional fairness measures assess whether resources are shared equally overall, they cannot distinguish orderly turn-taking from irregular access patterns that produce long and unpredictable waiting times despite similar cumulative outcomes. We introduce Rotational Periodicity (RP), a computationally efficient metric that evaluates both the regularity of waiting times between successful accesses and the balance of access frequencies across agents. We evaluate RP alongside a family of more detailed alternation metrics using a repeated threshold-congestion game in which two to ten reinforcement-learning agents compete for exclusive access to a shared resource. Our experiments reveal that independently trained agents often coordinate substantially worse than random-policy agents, even though conventional fairness metrics consistently report highly favourable outcomes. At the same time, RP closely reproduces the rankings of the more computationally expensive alternation metrics while computing twelve to twenty-five times faster as the number of agents increases. These findings show that evaluating multi-agent learning systems requires temporally aware measures of coordination, not only aggregate outcomes, and that efficient proxy metrics such as RP make this type of evaluation practical for larger intelligent computing systems.

**arXiv ID:** 2605.14879
</details>

<details>
<summary><strong>Learning Latency-Aware Orchestration for Multi-Agent Systems</strong> - Xi Shi, Mengxin Zheng, Qian Lou - [[pdf]](https://arxiv.org/pdf/2607.13359)</summary>

**Abstract:** Multi-agent systems (MAS) coordinate multiple LLM-powered agents through structured workflows, gaining reasoning power but incurring high inference latency from multi-step execution and repeated model invocations. Existing orchestration methods primarily optimize task performance and inference cost, leaving latency largely unaddressed. In MAS, end-to-end latency is governed by the critical execution path, so reducing total cost alone does not reliably reduce latency. Moreover, optimizing latency while preserving accuracy remains non-trivial: naive latency optimization can misassign operator-level credit and degrade task accuracy. To address this gap, we propose Latency-Aware Multi-agent System (LAMaS), a latency-aware orchestration framework for learning-based multi-agent systems. LAMaS addresses this challenge at two levels: at training time, it learns latency-aware execution graphs through constrained optimization with critical-path-aware credit assignment; at inference time, since a graph committed at training time cannot exploit runtime evidence, it complements graph construction with a lightweight controller that adaptively eliminates redundant future agent interactions as execution unfolds. Experiments on four benchmarks show that LAMaS achieves the best latency among evaluated learning-based MAS baselines, reducing end-to-end latency by over 50\% while maintaining competitive or better accuracy. LAMaS is also modular and transfers to other MAS with minimal changes, consistently yielding latency reductions.

**arXiv ID:** 2607.13359
</details>

<details>
<summary><strong>TriAgent: Divergence-Aware Multi-Agent Committees for Cost-Efficient Financial Sentiment Analysis</strong> - Isabel Xu, Cynthia Xu, Rachel Ren, Cong Guo, Jiacheng Ding - [[pdf]](https://arxiv.org/pdf/2607.19794)</summary>

**Abstract:** Production LLM-based financial sentiment analysis faces a structural cost trap: most queries are trivially classifiable, yet expensive cloud reasoners process them all, and the bill scales linearly with user count. We present TriAgent, a multi-agent committee stratified by contextual granularity -- a word-level lexicon (VADER), a sentence-level domain transformer (FinBERT), and a cross-sentence reasoner (Qwen2.5, 0.5B-14B-4bit, with Mistral-7B and Phi-3.5-mini cross-family checks). A three-way Semantic Divergence Index (SDI) measures pairwise disagreement across granularities and routes each query accordingly. Our central finding is the critic plateau: when the LLM is re-tasked as a critic over the smaller agents' outputs, F1 plateaus at ~0.87 across 1.5B-7B Qwen (bootstrap 95% CIs overlap), while a same-size 3-persona vote drops to F1=0.66, which is driven by granularity-stratified diversity. Three corollaries follow from the same SDI signal: (i) a Shared Consensus Dictionary on multilingual sentence-BERT answers 95% of Chinese queries from an English cache at F1=0.99 -- cross-border canonicalization at zero marginal cost; (ii) SDI doubles as a post-hoc LLM-hallucination detector at AUC=0.90; (iii) the SDI single-stage strategy attains the best risk-adjusted return (Sharpe=3.50) on a 20-ticker back-test, dominating both always-FinBERT (1.36) and always-LLM (0.11). At 10M-user scale, TriAgent saves $9.3M/year vs. a GPT-4o-mini baseline. Code, lexicons, and the SCD are released.

**arXiv ID:** 2607.19794
</details>

<details>
<summary><strong>Defer to Plan: Adaptive Multi-Agent Fusion for End-to-End V2X Driving</strong> - Nuoran Li, Zhang Zhang, Yueran Zhao, Tianze Wang, Chao Sun - [[pdf]](https://arxiv.org/pdf/2607.19774)</summary>

**Abstract:** Vehicle-to-everything-aided autonomous driving (V2X-AD) significantly enhances driving performance through information sharing. However, existing collaborative perception methods only optimize module-level perception capabilities and fail to effectively serve the ultimate planning and control tasks. We propose an end-to-end collaborative driving system that directly optimizes planning task performance. The system employs MotionNetwork to fuse historical temporal information, utilizes attention mechanisms to efficiently compress spatial features into compact tokens, and adaptively fuses multi-agent features through an autoregressive decoder. Additionally, we introduce Mixture-of-Experts (MoE) architecture to enhance the model's representation capacity for heterogeneous features. Experiments demonstrate that our method achieves a driving score of 79.72, surpassing the state-of-the-art CoDriving baseline (77.15) by 3.33% in closed-loop evaluation while maintaining communication efficiency.

**arXiv ID:** 2607.19774
</details>

<details>
<summary><strong>Dynamic Multi-Agent Pickup and Delivery in Robotic Cellular Warehousing Systems</strong> - Cheng Ren, Ming Li, Xinping Guan, George Q. Huang - [[pdf]](https://arxiv.org/pdf/2606.05669)</summary>

**Abstract:** Robotic Cellular Warehousing Systems (RCWS) give rise to multi-agent pickup and delivery (MAPD) processes in which robots sequentially collect multiple stock-keeping units (SKUs) for each order. Unlike classical MAPD formulations that assume static tasks, real warehouse operations often involve dynamic order evolution, where new SKUs may be appended to an order while it is being executed. Motivated by this practical requirement, this letter formulates the Dynamic Multi-Agent Pickup and Delivery problem considering internal order evolution for the first time. Building on the token passing (TP) mechanism, we propose two event-triggered online replanning algorithms. The first, Dynamic-TP, enables an event-triggered dynamic response by allowing robots to replan from their current execution states through priority-aware token acquisition after order updates. The second, Cooperative-TP, further enables idle robots to assist newly added SKUs while preserving the original order ownership. Simulation results demonstrate that the proposed methods significantly reduce order flowtime compared with static and non-cooperative baselines, thereby improving system-level efficiency in RCWS.

**arXiv ID:** 2606.05669
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>Symbol and Footprint Database for Electronic Components by Agentic Recognition and Generation</strong> - Yichen Shi, Yuzhi Liu, Zhuofu Tao, Li Huang, Yuhao Gao, Ting-Jung Lin, Lei Hel - [[pdf]](https://arxiv.org/pdf/2607.19767)</summary>

**Abstract:** A rich and recognizable component library is the cornerstone of printed circuit board (PCB) design and generation. Traditionally, engineers manually create symbols and footprints and design PCB schematics, which is time-consuming and error-prone. Leveraging multimodal large language models (MLLMs), we develop SFgen, an agentic recognition and generation flow of symbol and footprint for electronic components. SFgen achieves 86% accuracy for symbol generation and 80% accuracy for footprint generation. We use the SFgen method to create SFnet, a database of symbols and footprints. It now has 1000 components and is expanding constantly, which lays the foundation for automatic generation of PCB designs.

**arXiv ID:** 2607.19767
</details>

<details>
<summary><strong>A Framework of User Experience Principles for Human-AI Agent Interaction in the Workplace</strong> - Kathrin Paimann, Elizangela Valarini, Sebastian Juhl - [[pdf]](https://arxiv.org/pdf/2607.19941)</summary>

**Abstract:** As AI agents become integral to business workflows, establishing guiding user experience (UX) principles is crucial for ensuring user trust and successful adoption. To address this, our study uses a multi-method approach - combining participatory design workshop, paper-and-pencil, expert review, meta-analysis, and in-depth interviews - to identify and validate a design framework of eight core UX principles for human-AI agent interaction in the workplace. Together with their underlying criteria, these principles provide actionable guardrails for designers and software engineers, creating a foundation for developing effective and human-centered AI agent interactions. This study contributes to a structured foundation for future empirical studies on agentic AI in enterprise settings.

**arXiv ID:** 2607.19941
</details>

<details>
<summary><strong>Agent-Based Modeling of Low-Emission Fertilizer Adoption for Dairy Farm Decarbonisation using Empirical Farm Data</strong> - Surya Jayakumar, Kieran Sullivan, John McLaughlin, Christine OMeara, Indrakshi Dey - [[pdf]](https://arxiv.org/pdf/2605.03648)</summary>

**Abstract:** To understand complex system dynamics in dairy farming requires tools that capture farm heterogeneity, social interactions, and cumulative environmental impacts. This study proposes an agent-based modelling(ABM) framework to simulate nitrogen management and low-emission fertiliser adoption across 295 Irish dairy farms over a 15-year period. Using empirical data, the model replicates farm communication through a social network, where adoption probabilities are driven by social contagion, farm-scale factors, and policy interventions such as subsidies and carbon taxes. The framework computes sectoral greenhouse gas emissions, cumulative abatement, and private-social costs, with Monte Carlo and sensitivity analyses quantifying uncertainty. The model achieved high predictive accuracy (R2 = 0.979, RMSE = 0.0274) and was validated against observed adoption data using a Kolmogorov-Smirnov test (D = 0.2407, p < 0.001). Adoption dynamics were fitted to Rogers logistic curves, reproducing a realistic saturation plateau (91%) while acknowledging structural laggard effects. By conceptualizing decarbonization as a socio-technical evolution rather than a purely monetary calculation, this study establishes an exploratory policy framework for evaluating the diffusion of climate strategies prior to implementation.

**arXiv ID:** 2605.03648
</details>

<details>
<summary><strong>Information Aggregation with AI Agents</strong> - Spyros Galanis - [[pdf]](https://arxiv.org/pdf/2604.20050)</summary>

**Abstract:** Can Large Language Models (AI agents) aggregate dispersed private information through trading and reason about the knowledge of others by observing price movements? We conduct a controlled experiment where AI agents trade in a prediction market after receiving private signals, measuring information aggregation by the log error of the last price. We find that although the median market is effective at aggregating information in the easy information structures, performance deteriorates in the harder structures, suggesting that AI agents may suffer from similar limitations as humans when reasoning about others. Consistent with our theoretical predictions, market accuracy does not improve from allowing cheap talk communication, changing the duration of the market, or strategic prompting; initial price has little average effect but matters in the very hard structure. We also find that "smarter" AI agents perform better at aggregation and are more profitable. Surprisingly, giving them feedback about past performance does not improve aggregation.

**arXiv ID:** 2604.20050
</details>

<details>
<summary><strong>An LLM-powered Agentic Recommendation System for Connected TV Content Discovery</strong> - Lei Shi, Di Wang, Harry Tran, Helsing Xu, Yuchen Lu, Dhara Ghodasara, Wilson Chaney, Xueting Liao, Jerry Yu, Huayu Ding, Reza Mirghaderi, David Fan, Qi Guo, Chongguang He, Warren Wang, Warren Deng, Mingze Gao, Shike Mei, Shuo Tang, Zhe Zhang, Jianming He, Abhishek Kumar, Haotian Wu, Hamed Firooz, Li Li - [[pdf]](https://arxiv.org/pdf/2607.09988)</summary>

**Abstract:** Recommendation systems, from traditional multi-stage to recent unified generative architectures, face challenges in incorporating diverse contextual signals, such as trending topics, breaking news, cultural events, and cross-surface user activities, into their ranking pipelines. These systems are designed to consume structured behavioral signals with consistent schemas, and lack the reasoning capability to naturally process unstructured or heterogeneously formatted contextual information. Incorporating such signals typically requires feature engineering, bespoke data pipelines, and carefully tuned heuristics. In this paper, we present an LLM-powered agentic recommendation system designed for Connected TV (CTV) content discovery that addresses these limitations. Our system leverages the reasoning capabilities of large language models to naturally process and synthesize diverse signals across varying schemas and structures, eliminating much of the manual integration inherent in traditional ranking and retrieval systems. Recognizing that current LLM-based solutions still fall short of traditional machine learning models in several recommendation tasks, including retrieval efficiency, personalization precision, and scalability, we adopt an agentic architecture that orchestrates specialized components, allowing each sub-task to be handled by the most suitable method, whether LLM-based or traditional ML. The main contribution of this work is our engineering approach to successfully overcoming the practical limitations of enabling LLM for recommendation, particularly inference latency. We share insights from our work and discuss the trade-offs and lessons learned in building a hybrid system that combines the flexibility of LLMs with the performance of established recommendation techniques.

**arXiv ID:** 2607.09988
</details>

<details>
<summary><strong>NexForge: Scaling Agent Capabilities through Requirement-Driven Task Synthesis for LLMs</strong> - Jiarong Zhao, Zhikai Lei, Zhiheng Xi, Rui Zheng, Hang Yan, Jie Zhou, Qin Chen, Liang He - [[pdf]](https://arxiv.org/pdf/2607.14186)</summary>

**Abstract:** Scaling executable agent training data for LLM post-training is bottlenecked by substrate-bound methods that tie task generation to predefined tools, repositories, or skill graphs: expanding coverage requires manual substrate engineering, each new domain demands a bespoke pipeline, and the resulting task distributions often reflect substrate biases rather than real-world demand. We introduce NexForge, a requirement-driven framework that takes high-level capability requirements as input and synthesizes diverse, executable agent tasks and expert trajectories for SFT. NexForge first investigates real-world demand to construct representative scenarios and task profiles, then performs distribution-aware compilation to generate task directives. For each directive, NexForge automatically retrieves or constructs the required files, dependencies, and runtime configurations, and finally synthesizes expert rollouts and produces training trajectories. Without domain-specific infrastructure, NexForge produces 3.6K terminal and 2K office tasks, improving Qwen3.5-35B-A3B Base from 22.5\% to 52.0\% on Terminal-Bench 2.0 and from 813 to 1338 Elo on GDPval; scaling further to 43.2K terminal tasks yields 58.4\%, on par with Claude Opus 4.6 equipped with Claude Code. Scaled further, NexForge-synthesized data contributes to the training of Nex-N2, a family of publicly available agent models that lift Qwen3.5-35B-A3B to 75.3\% on Terminal-Bench 2.1 and to 1585 Elo on GDPval -- achieving state-of-the-art open-source performance and surpassing several frontier proprietary systems. Nex-N2 models are available at this https URL.

**arXiv ID:** 2607.14186
</details>

<details>
<summary><strong>Autonomous Collaborative Learning Among an Ensemble of Tsetlin Machines with Consensus-Based Inference</strong> - Yehuda Rudin, Osnat Keren, Michal Yemini, Alexander Fish - [[pdf]](https://arxiv.org/pdf/2607.20124)</summary>

**Abstract:** Tsetlin Machine (TM) is a rule-based machine-learning algorithm comprising collectives of two-action Tsetlin Automata (TAs) that cooperatively form conjunctive logical clauses from Boolean inputs through stochastic feedback. Although few recent studies have examined TM Federated Learning, the broader area of distributed and decentralized TM learning has not received much attention in the existing literature and warrants further exploration. In this work, we propose a paradigm for decentralized collaborative learning under a vertical feature-partitioning setting among an ensemble of Tsetlin Machines using consensus-based inference. Within this decentralized paradigm, each agent maintains its own private TM model, and there is no exchange of raw data among agents. Inference combines individual agents model predictions into a global consensus. The paradigm accommodates heterogeneous TM-based agents with differing data acquisition means, local data distributions, or computational resources, thereby facilitating the integration and fusion of information in settings such as multi-modal sensing environments. Experiments conducted using two-dimensional grid and connected graph network topologies demonstrate that the classification accuracies achieved are comparable to those of centralized models.

**arXiv ID:** 2607.20124
</details>

<details>
<summary><strong>Milo, a Fully Autonomous Indoor/Outdoor Robotic Guide Dog</strong> - Florian Golemo, Joanna Wolski, Joel Ruben Antony Moniz, Christopher Pal - [[pdf]](https://arxiv.org/pdf/2607.19530)</summary>

**Abstract:** Many Blind and Low-Vision (BLV) people rely on guide dogs for moment-to-moment navigation, such as staying on path and avoiding obstacles and pedestrians. However, guide dogs are expensive to acquire and maintain (approximately \$50k USD plus ongoing costs), often involve long waiting lists, and have relatively short life expectancies. While robot guide dogs offer a promising alternative, existing approaches exploring this idea suffer from several drawbacks: They often lack the autonomy required for real-world deployment, relying on prior 3D scans of the environment, external computation, or limited awareness of the handler. In this work, we present Milo, the first open-source, low-cost (approximately \$2k USD) robotic guide dog platform capable of fulfilling the basic collaborative navigation role expected of a guide dog. Milo is fully autonomous, requiring no a priori knowledge of the environment, completely self-contained with all computation performed onboard, and suitable for both indoor and outdoor navigation while avoiding obstacles and pedestrians. Our system consists of a modified Unitree Go2 robot (equipped with onboard compute, sensors, and a handle), a perception stack combining voxel mapping with floor, obstacle, and pedestrian detection, and a navigation stack based on an obstacle-avoidance policy trained in a custom bird's-eye-view simulator. We evaluate Milo in real indoor and outdoor obstacle courses and compare it against a costmap-based baseline, demonstrating smoother navigation and fewer handler collisions. To maximize accessibility for BLV users, we release both the robot hardware instructions and the complete software stack as open source.

**arXiv ID:** 2607.19530
</details>

<details>
<summary><strong>An Optimal Algorithm for Changing from Latitudinal to Longitudinal Formation of Autonomous Aircraft Squadrons</strong> - Paulo André Sperandio Giacomin, Elder Moreira Hemerly - [[pdf]](https://arxiv.org/pdf/1712.00513)</summary>

**Abstract:** This work presents an algorithm for changing from latitudinal to longitudinal formation of autonomous aircraft squadrons. The maneuvers are defined dynamically by using a predefined set of 3D basic maneuvers. This formation change is necessary when the squadron has to perform tasks which demand both formations, such as lift off, georeferencing, obstacle avoidance and landing. Simulations show that the formation change is done without collision. The time complexity analysis of the transformation algorithm reveals that its efficiency is optimal, and the proof of correctness ensures its longitudinal formation features.

**arXiv ID:** 1712.00513
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (25 papers)</h2></summary>

<details>
<summary><strong>FORCE-Bench: A Benchmark, Dataset, and Evaluation Harness for Agentic AI in Enterprise Finance</strong> - Wolfgang M. Pauli, Sarah Panda, Kidus Admassu, Said Bleik, Ademola Okerinde, Jeremy Reynolds - [[pdf]](https://arxiv.org/pdf/2607.19409)</summary>

**Abstract:** Recent advances in large language models have accelerated deployment of agentic systems in operational finance. Existing benchmarks emphasize measuring general capabilities, instruction following, or safety, but few directly address the operational finance workflows that agentic systems are now being deployed to automate. Finance professionals require agents to not only provide factually sound and properly grounded information, but also ensure that this information is verifiable and consistently adheres to rules and constraints of the operational finance domain. We introduce FORCE-Bench, which contains 251 expert-annotated queries and evaluates responses using a rubric-based framework calibrated to the requirements of the operational finance domain, across eight dimensions: accuracy, citations, clarity, depth, groundedness, recency, relevance, and structure. FORCE-Bench assesses agentic systems on three task types: financial obligation research (querying ERP systems for accounts receivable and payable data), financial entity performance research (answering time-bound questions from public filings and market data), and business brief generation (synthesising multi-source company intelligence reports). To reflect real deployment conditions, we evaluate our purpose-built agent, as well as the general-purpose agentic systems, under common tool access and latency-bounded settings. Results show that general-purpose agentic systems do not consistently meet finance-domain quality requirements under operational constraints, while the purpose-built Finance Agent for Microsoft 365 Copilot is more reliable across dimensions. We release the dataset, rubrics, harness, and analysis code as open-source to support reproducible comparison and adaptation to other enterprise finance environments.

**arXiv ID:** 2607.19409
</details>

<details>
<summary><strong>Leveraging Offline Supervision for Efficient and Generalizable Reinforcement Learning in Large-Scale Vision-Language-Action Models</strong> - Dmitriy Poyarkov, Aleksei Staroverov, Aleksandr I. Panov - [[pdf]](https://arxiv.org/pdf/2607.19399)</summary>

**Abstract:** It is commonly observed that online reinforcement learning (RL) produces better-performing strategies than offline methods across a broad range of performance measures. In particular, RL-trained policies exhibit stronger out-of-distribution (OOD) behavior, where models trained only with imitation learning approaches often struggle. A recent study introduced an OOD-focused benchmark and reported that RL-trained vision-language-action (VLA) policies achieve noticeably better OOD performance and slightly better in-distribution (IND) performance than their counterparts trained with supervised fine-tuning (SFT). In this work, we investigate whether hybrid offline-online training can combine the advantages of both approaches. Specifically, we study RL methods regularized by offline supervision via either offline data or an offline-trained reference policy. We evaluate these approaches on the OOD benchmark and compare them with both offline-only training and standard RL. Our results show that although offline training achieves limited OOD performance by itself, incorporating offline supervision into RL preserves strong OOD capability while substantially improving training efficiency. In particular, the guided methods reach performance close to that of standard RL while requiring roughly half of the training budget. Rather than producing a trade-off between speed and OOD performance, the hybrid approach retains strong OOD capability while achieving this efficiency gain. Project page: this https URL

**arXiv ID:** 2607.19399
</details>

<details>
<summary><strong>ChainWatch: A Kill Chain-Aligned Sequential Detection Framework for Multi-Step Attacks in MCP-Based AI Agent Systems</strong> - Om Narayan, Rashmi Jyoti, Ramkinker Singh - [[pdf]](https://arxiv.org/pdf/2607.19432)</summary>

**Abstract:** The Model Context Protocol (MCP) is an open-source standard that allows AI agents to connect to external tools, databases, and services. While this connectivity enables powerful agent capabilities, it also introduces multi-step attacks that existing per-call defenses cannot reliably detect. Attackers can compose individually benign tool invocations into malicious sequences that evade isolated inspection. This paper presents ChainWatch, a sequential detection framework for identifying multi-step attacks in MCP-based AI agent systems. ChainWatch models attack progression using a six-stage kill chain and applies a Hidden Markov Model (HMM) to classify tool-call sequences. Detection rules are triggered when a session exhibits suspicious progression across multiple stages. The framework is supported by a structured threat model covering direct sequential attacks, indirect prompt injection chains, and hybrid multi-stage attacks. A 20-dimensional feature extraction schema captures behavioral signals from tool interactions. We demonstrate the approach using five representative attack scenarios from the security literature, showing how ChainWatch detects attack chains that evade traditional per-call security mechanisms.

**arXiv ID:** 2607.19432
</details>

<details>
<summary><strong>Building Trust in Autonomous Commerce: A Verifiable Global Event Timeline and AI-Ready Fraud Intelligence Layer</strong> - Rajat Srivastava - [[pdf]](https://arxiv.org/pdf/2607.19436)</summary>

**Abstract:** Agentic commerce protocols such as AP2 and ACP define mechanisms for secure agent-initiated transactions but do not provide interoperable, tamper-evident auditability or verifiable temporal ordering of events across heterogeneous domains. This paper addresses these gaps by proposing a verifiable global event timeline for agentic commerce, constructed from four core components: canonical event schemas that enforce deterministic serialization, deterministic batch formation ensuring reproducible ordering without reliance on synchronized clocks, Merkle-based append-only commitments providing logarithmic-cost inclusion proofs, and blockchain anchoring establishing a tamper-evident temporal backbone. Building on this infrastructure, we introduce a cryptographically signed fraud marker that binds risk labels to anchored evidence through an unforgeable provenance chain, and a dataset lineage model enabling reproducible, tamper-evident AI training pipelines. Empirical results from a prototype implementation demonstrate: Merkle tree construction processes 50,000 events in 47 milliseconds; end-to-end verification completes in under 0.013 milliseconds regardless of batch size; inclusion proof sizes grow logarithmically from 320 bytes at 1,000 events to 512 bytes at 50,000 events; and Merkle-based verification outperforms linear scan by 14.4x at 50,000 events.

**arXiv ID:** 2607.19436
</details>

<details>
<summary><strong>REGEN: Replay-recycling for Expert-to-Generalist distillation with Offline Reinforcement Learning</strong> - Yunjie Chen, Xiaoxin Chen, Fang Wang - [[pdf]](https://arxiv.org/pdf/2607.19450)</summary>

**Abstract:** Large-scale online reinforcement learning (RL) is the predominant means of eliciting advanced abilities including long-term reasoning and agentic tool use in large language models (LLMs). However, continuing to scale it across vast task domains of interest remains challenging in both computational infrastructure and cost, especially when considering RL as merely a one-off learning stage. Recently, a widely used technique for distilling knowledge across various domains and training stages, multi-teacher on-policy distillation (MOPD), helps to decouple the RL stage, saving costs, while maintaining generality across vast domains. Nonetheless, similar to online RL, MOPD requires coupled inference and backward passes, which continues to limit its scalability and computational efficiency. To address these challenges, we propose REGEN: Replay-recycling for Expert-to-Generalist Distillation with Offline RL. Instead of distilling from multiple teacher models, REGEN trains a generalist by simply recycling the replay memory -- the free by-product of the teachers' specialized RL training -- and employing offline RL algorithms. REGEN completely decouples the rollout sampling from the backward training process and thus greatly reduces the training cost. Across mathematical reasoning, code generation, and instruction following, REGEN matches the accuracy of MOPD at substantially lower cost. It potentially turns online RL into a data synthesis process instead of a one-off learning stage, and can potentially be extended to large-scale post-training without requiring heavy computational load.

**arXiv ID:** 2607.19450
</details>

<details>
<summary><strong>Personalized Recommendation Tool Learning via Autonomous Language Agents</strong> - Mingdai Yang, Zhiwei Liu, Weizhi Zhang, Yibo Wang, Hao Peng, Philip Yu - [[pdf]](https://arxiv.org/pdf/2607.19739)</summary>

**Abstract:** Although large language models (LLMs) have recently gained traction in recommender systems due to their strong reasoning capabilities and extensive world knowledge, previous LLM-based agents suffer from hallucination and context-length limitations, and thus are not suitable for full-ranking recommendation tasks. To circumvent these limitations through architectural design rather than modifying the LLM itself, we propose an agent-based recommendation framework, memory-based $\textbf{P}$ersonalized $\textbf{R}$ecommendation $\textbf{T}$ool learning via autonomous language $\textbf{A}$gents (PRTA), in which an LLM acts as a central planner interacting with multiple recommendation models as tools. The LLM-based agent is responsible for high-level reasoning and personalized tool selection, while traditional recommendation models perform full-ranking scoring, leveraging their scalability in modeling behavioral patterns. To support personalized tool selection, we design reflection mechanisms that enable the agent to evaluate and compare tools for each user based on user profiles and candidate ranked lists. Extensive experiments across three public datasets demonstrate the superiority of \modelname over traditional recommendation and LLM-based baselines in improving full-ranking recommendation performance.

**arXiv ID:** 2607.19739
</details>

<details>
<summary><strong>Reinforcement Learning for Large Language Model Selective Evidence Adoption from Contaminated Retrieval Results</strong> - Yanyu Chen, Yue Li, Yongyi Cui, Dongsheng Shi, Lichang Dai - [[pdf]](https://arxiv.org/pdf/2607.20090)</summary>

**Abstract:** Retrieval-augmented large language models frequently face contexts that interleave useful evidence with misleading statements or instruction-like content. Blanket refusal discards valid evidence, whereas uncritical adoption yields incorrect or unsafe answers. The ability to selectively adopt relevant information while rejecting deceptive or harmful content is therefore critical for reliable deployment in real-world retrieval settings. We introduce SelectBench, a controlled benchmark and training set for selective evidence adoption, and post-train Qwen3.5-4B directly with DAPO using either deterministic rule rewards or a frozen semantic judge. On the corrected 325-example SelectBench-v2 test set, strict success rises from 22.46% for the original checkpoint to 25.54% with DAPO-Rule and 26.46% with DAPO-DeepSeek. Both trained policies reduce forbidden-content adoption and produce shorter, more focused responses, yet prompt-injection following does not improve. The paired gains are modest and fail to survive Holm correction, suggesting that stronger reward shaping or additional training iterations may be needed for more robust gains. DAPO-DeepSeek exhibits no material degradation on MMLU or clean HotpotQA, indicating that the post-training procedure preserves general capabilities. These results demonstrate a directional improvement in selective evidence use, while identifying injection resistance and statistical robustness as important remaining challenges for future work.

**arXiv ID:** 2607.20090
</details>

<details>
<summary><strong>In-the-Flow Agentic System Optimization for Effective Planning and Tool Use</strong> - Zhuofeng Li, Haoxiang Zhang, Seungju Han, Sheng Liu, Jianwen Xie, Yu Zhang, Yejin Choi, James Zou, Pan Lu - [[pdf]](https://arxiv.org/pdf/2510.05592)</summary>

**Abstract:** Outcome-driven reinforcement learning has advanced reasoning in large language models (LLMs), but prevailing tool-augmented approaches train a single, monolithic policy that interleaves thoughts and tool calls under full context; this scales poorly with long horizons and diverse tools and generalizes weakly to new scenarios. Agentic systems offer a promising alternative by decomposing work across specialized modules, yet most remain training-free or rely on offline training decoupled from the live dynamics of multi-turn interaction. We introduce AgentFlow, a trainable, in-the-flow agentic framework that coordinates four modules (planner, executor, verifier, generator) through an evolving memory and directly optimizes its planner inside the multi-turn loop. To train on-policy in live environments, we propose Flow-based Group Refined Policy Optimization (Flow-GRPO), which tackles long-horizon, sparse-reward credit assignment by converting multi-turn optimization into a sequence of tractable single-turn policy updates. It broadcasts a single, verifiable trajectory-level outcome to every turn to align local planner decisions with global success and stabilizes learning with group-normalized advantages. Across ten benchmarks, AgentFlow with a 7B-scale backbone outperforms top-performing baselines with average accuracy gains of 14.9% on search, 14.0% on agentic, 14.5% on mathematical, and 4.1% on scientific tasks, even surpassing larger proprietary models like GPT-4o. Further analyses confirm the benefits of in-the-flow optimization, showing improved planning, enhanced tool-calling reliability, and positive scaling with model size and reasoning turns.

**arXiv ID:** 2510.05592
</details>

<details>
<summary><strong>Alipay-PIBench: A Realistic Payment Integration Benchmark for Coding Agents</strong> - Shiyu Ying, Xuejie Cao, Yingfan Ma, Yuanhao Dong, Wenyu Chen, Bowen Song, Lin Zhu - [[pdf]](https://arxiv.org/pdf/2607.14573)</summary>

**Abstract:** Payment integration is a demanding repository-level software task: agents must select a suitable product, implement coordinated client-server flows, verify payment outcomes, and preserve consistency between transaction and business states. We introduce Alipay-PIBench, a benchmark for evaluating coding agents on realistic Alipay payment integration. It contains nine product-specific projects and 18 task instances, each organized into Basic functional-completion and Advanced risk-aware hardening scenarios. Scenario-specific rubrics support deterministic static, unit, integration, and end-to-end checks, supplemented by LLM-assisted assessment for semantic requirements. We evaluate six coding-agent models and report rubric pass rate (RPR). Under the with-skill condition, mean RPR ranges from 68.58% to 91.37%. Access to the alipay-payment-integration skill improves mean RPR by 10.31 percentage points on average relative to the without-skill condition, with gains varying across models, products, and scenarios. Method-level results distinguish source-level completion, executable payment behavior, and payment-domain requirements. Alipay-PIBench provides a controlled setting for diagnosing model capability and evaluating structured guidance in payment integration.

**arXiv ID:** 2607.14573
</details>

<details>
<summary><strong>OpenSkillRisk: Benchmarking Agent Safety When Using Real-World Risky Third-Party Skills</strong> - Qiyuan Liu, Tingfeng Hui, Kun Zhan, Kaike Zhang, Ning Miao - [[pdf]](https://arxiv.org/pdf/2607.20121)</summary>

**Abstract:** LLM-based agents leverage third-party skills to extend their capabilities in open-world scenarios. However, third-party skills can introduce extra security vulnerabilities, as seemingly harmless skills can contain latent safety risks that only emerge during actual execution. In this work, we conduct a systematic investigation into how well current agent systems recognize and avoid such risks. To support quantitative and qualitative evaluation, we construct OpenSkillRisk, a dedicated safety benchmark containing 263 risky skills collected from public skill marketplaces. We classify these skills into seven categories based on their threat types and pair each skill with a standardized user task and a corresponding sandbox for controlled evaluation. Distinct from prior benchmarks, OpenSkillRisk not only covers more realistic and diverse unsafe scenarios, but also provides a fine-grained analysis to diagnose the behavioral patterns of agents in such scenarios. We conduct comprehensive experiments covering three mainstream CLI agent frameworks and thirteen state-of-the-art LLMs. Experimental results show that no tested system handles risky skills reliably: even the safest configurations still execute unsafe actions in about 17% of cases. Context-dependent and system-level risks are especially difficult for current agent systems to avoid. Our behavioral analysis reveals three recurring failure patterns: agents may fail to recognize the risk, recognize it but fail to intervene before acting, or follow skill instructions beyond the user's intended scope. These findings highlight the need to improve both risk reasoning in LLMs and execution control in agent frameworks.

**arXiv ID:** 2607.20121
</details>

<details>
<summary><strong>Stale but Stable: Staleness-Adaptive Trust Regions for Stabilizing Asynchronous Reinforcement Learning</strong> - Junyao Yang, Yucheng Shi, Zongxia Li, Zhongzhi Li, Ruhan Wang, Xiangxin Zhou, Kishan Panaganti, Haitao Mi, Leowei Liang - [[pdf]](https://arxiv.org/pdf/2607.18722)</summary>

**Abstract:** Asynchronous reinforcement learning improves throughput by decoupling rollout generation from optimization, but staleness is an inevitable byproduct compounded by policy lag, engine delays, and mixture-of-experts routing. From a trust-region perspective, this mismatch is critical: training-inference divergence governs approximation error in finite-horizon bounds, whereas PPO clipping only gates sampled outward updates, acting as a sampled surrogate rather than a full-policy constraint. As a result, high-staleness updates remain weakly controlled in the asynchronous regime where stale rollouts matter most.
We introduce the Staleness-Adaptive Trust Region (SAT), which uses the detached sampled log-ratio as a practical staleness proxy, identifies high-mismatch tails within each batch via staleness-based kernel scaling, and contracts only the sign-selected endpoint of the nominal PPO interval. This preserves baseline behavior on ordinary tokens while enforcing more conservative updates on newly intercepted outward bands. We prove local interval containment and pointwise pessimism relative to PPO, showing how the adaptive rule reshapes update geometry under heterogeneous staleness.
We evaluate SAT in a decoupled asynchronous RL setup built on Qwen3-30B-A3B-Base, using SGLang as the inference engine and Megatron for training. In this setting, SAT-GSPO w/ R3 achieves the best observed AIME24 avg@8, reaching 35.83 at lag 1 and 34.79 at lag 8, while SAT-GSPO reaches 34.17 at lag 1. Adaptive clipping and routing replay act as complementary stabilizers targeting mismatch tails and routing inconsistency, respectively. Overall, aligning clip intervals with staleness heterogeneity effectively stabilizes asynchronous RL.

**arXiv ID:** 2607.18722
</details>

<details>
<summary><strong>Agent-Centric Animal Pose Forecasting</strong> - Eyrun Eyjolfsdottir, Kristin Branson - [[pdf]](https://arxiv.org/pdf/2607.19548)</summary>

**Abstract:** Understanding animal behavior at an algorithmic level -- what animals attend to, how they form internal models and plans, and how this maps to action -- remains a central challenge in neuroscience and ethology. Data-driven generative models offer a path toward this understanding. We introduce a framework for training agent-centric autoregressive models of animal behavior from tracked pose, applicable to single animals and to groups in which each agent senses and responds to its conspecifics. Our models input egocentric sensory observations and output egocentric movements, mirroring the biological constraint that animals observe and act on the world from their own reference frame. Social behavior emerges from agents independently sensing and responding to one another. This agent-centric formulation requires managing many parallel representations of the same data, along with ML-specific transformations like discretization. We release a general-purpose library focused on the composable sequences of operations that translate between these representations. We show that trained models capture the distribution of social behavior in groups of courting Drosophila, and our library includes quantitative tools for measuring fit. We demonstrate how the library supports systematic comparison across input and output representations and that it adapts straightforwardly to a new domain.

**arXiv ID:** 2607.19548
</details>

<details>
<summary><strong>The Mechanism Matters: When Knowledge Graphs Help Reinforcement Learning</strong> - Mohammed Sameer Syed - [[pdf]](https://arxiv.org/pdf/2607.19616)</summary>

**Abstract:** Knowledge graphs (KGs) are widely used to inject prior knowledge into reinforcement learning (RL), yet the literature is dominated by single-domain, positive-result method papers, so we lack a systematic account of when KG structure helps an agent, when it is neutral, and when it hurts. We conduct a controlled study that independently varies the RL task, the injection mechanism (state features, action masking, or potential-based reward shaping), and KG quality. Using a synthetic, fully controllable KG over MiniGrid environments, we report three findings. First, on compositional sparse-reward tasks structured KG guidance improves sample efficiency and solve reliability (70% to 97% of seeds), and a shuffle control that permutes the KG's edges while preserving their count collapses the benefit toward baseline (masking p=0.0001; shaping p=0.006), so the gain is structural rather than generic regularization. Second, KG value scales with the amount of task-relevant knowledge the graph contains. Third, and most consequential, safety depends on the mechanism: soft, optimality-preserving injection benefits from correct knowledge and harmlessly ignores incorrect knowledge, whereas hard masking is brittle, forbidding essential actions when the KG is incomplete or corrupted and making a wrong KG worse than none. A UMLS-derived clinical case study on sepsis management under offline RL is a careful null, underscoring that benefits require task structure the chosen mechanism can exploit. Our results give practitioners concrete guidance on how, and how much, to trust a KG when using it to guide RL.

**arXiv ID:** 2607.19616
</details>

<details>
<summary><strong>Asymptotically Optimal Regret for Reinforcement Learning without Horizon Dependence</strong> - Runlong Zhou, Zihan Zhang, Maryam Fazel, Simon S. Du - [[pdf]](https://arxiv.org/pdf/2607.19854)</summary>

**Abstract:** We study horizon-free regret minimization for finite-horizon time-homogeneous tabular Markov decision processes with $S$ states, $A$ actions, horizon $H$, and per-trajectory total reward bounded by $1$.
We propose a new algorithm and prove a regret upper bound \[\tilde O(\sqrt{SAK}+S^8A^3)\] with failure probability $\delta$, where $K$ is the number of episodes and $\tilde O(\cdot)$ hides $\mathsf{poly}\log(S,A,K,1/\delta)$.
Thus, the regret is $H$-free and asymptotically optimal, matching the contextual-bandit lower bound $\Omega(\sqrt{SAK})$ up to logarithmic factors. This completely removes the $\log H$ dependence from the previous $\tilde O(\sqrt{SAK\log H}+S^2A\log H)$ guarantee of Zhang et al. (2021), and drastically improves the prior best horizon-free regret $\tilde O(\sqrt{S^9A^3K})$ of Zhang et al. (2022) asymptotically.
The main technical difficulty is that the optimal value functions $\{V_h^*\}_{h=1}^H$ are time-inhomogeneous even though the transition kernel is time-homogeneous. A direct union bound over all value functions typically incurs an additional $\min\{\log H,S\}$ factor. We avoid this factor by (i) exploiting the monotonicity of $V_h^*$ in $h$ and (ii) non-trivially projecting the value functions onto an $S$-dimensional grid.
Our analysis relies on three additional ingredients. First, we introduce a horizon-truncation argument that enables reward-based exploration and removes the cost of a separate reward-free exploration phase. Second, we design a cutting bonus that preserves both optimism and the monotonicity needed for planning. Third, we prove a new bound on total deviation for time-homogeneous MDPs, which controls the clipped variance terms in the cutting bonus with adjustable polynomial dependence on $S$ and without any dependence on $H$. Together, these tools yield an asymptotically optimal horizon-free regret guarantee.

**arXiv ID:** 2607.19854
</details>

<details>
<summary><strong>Generalized Kalman filter based temporal difference reinforcement learning</strong> - Vasos Arnaoutis, Eric Lutters, Bojana Rosić - [[pdf]](https://arxiv.org/pdf/2607.20010)</summary>

**Abstract:** In this paper, we present a generalized temporal-difference (TD) reinforcement learning framework based on the theory of conditional expectations. The value and action-value (Q-value) functions are treated as uncertain quantities, and their estimation is formulated as a stochastic inference problem. Unlike classical Kalman-based temporal-difference learning, which relies on linear-Gaussian assumptions, the proposed formulation is derived directly from the conditional expectation framework and naturally extends to nonlinear models and non-Gaussian probability distributions. The proposed method recursively estimates not only the conditional expectation of the value function but also its second probabilistic moment, thereby quantifying the uncertainty associated with the learned value function throughout the learning process. To obtain a computationally tractable algorithm, the stochastic problem is discretized using either polynomial chaos expansions or ensemble-based approximations, providing efficient representations of the underlying random variables. The proposed framework is demonstrated on two optimal control problems: a linear mass--spring--damper system and a nonlinear heat conduction problem in a closed cavity. The numerical examples illustrate the capability of the proposed method to accurately estimate both the value function and its associated uncertainty, while extending classical Kalman-based temporal-difference learning to a broader class of stochastic systems.

**arXiv ID:** 2607.20010
</details>

<details>
<summary><strong>Isaac Sim-to-Real: Reinforcement Learning based Locomotion for Quadrupeds</strong> - Jordan Dowdy, Jean Chagas Vaz - [[pdf]](https://arxiv.org/pdf/2607.18135)</summary>

**Abstract:** Learning-based approaches to locomotion have risen in popularity in recent years, showing the capability for complex legged locomotion and whole-body control. Reinforcement learning (RL), the primary learning-based approach for locomotion, often utilizes a high-performance simulation tool, providing a controlled and efficient training and development environment. However, policies that perform well in simulation frequently encounter unexpected challenges when deployed on a physical system, known as the sim-to-real gap. This work presents a robust RL locomotion framework capable of whole-body control. The proposed RL framework utilizes Nvidia's new set of simulation tools, Isaac Sim, and its companion RL framework, Isaac Lab, for training, achieving a zero-shot sim-to-real policy. The performance of our policy is validated on physical hardware using the Unitree Go1, with experimental results showing similar velocity tracking performance to the quadruped's integrated controller, with a greater ability to recover from large disturbances, and achieve linear velocities of 2.0 m/s and angular velocities of 1.8 rad/s.

**arXiv ID:** 2607.18135
</details>

<details>
<summary><strong>Towards Torque-Driven Reinforcement Learning for Quadruped Locomotion</strong> - Jordan Dowdy, Jean Chagas Vaz - [[pdf]](https://arxiv.org/pdf/2607.18365)</summary>

**Abstract:** Reinforcement learning (RL) for legged robots is advancing locomotion, demonstrating its ability to adapt to new and challenging terrain. Traditionally, these RL locomotion frameworks are position-based, making the policy less adaptable to terrain types and requiring state estimation techniques in the observation space, i.e., linear velocity. Moreover, these RL frameworks often use small, lightweight quadrupeds that are limited in their viability for high-complexity tasks due to hardware constraints. This work explores an RL torque control framework for heavyweight high-torque quadrupeds. The RL framework in this paper can traverse rough terrain and effectively track a desired linear velocity without requiring knowledge of the agent's current velocity. Using Nvidia's Isaac Sim and Isaac Lab, simulation results of the RL torque control policy are shown on the Unitree B1 quadruped, achieving speeds of 3.5 m/s and 1.5 rad/s. In addition, the quadruped can walk up and down stairs without the aid of an exteroceptive sensor.

**arXiv ID:** 2607.18365
</details>

<details>
<summary><strong>Towards Miniature Humanoid Tele-Loco-Manipulation Using Virtual Reality and Reinforcement Learning</strong> - Nicolas Kosanovic, Jordan Dowdy, Jean Chagas Vaz - [[pdf]](https://arxiv.org/pdf/2607.20399)</summary>

**Abstract:** Full-sized humanoid robot capabilities have grown exponentially in recent years, aiming towards general-purpose deployment in human environments. A popular control method used by manufacturers utilizes Virtual Reality for upper-body teleoperation and Reinforcement Learning for lower-body balance and locomotion control. As a result, a single remote operator can see, manipulate, and navigate about a real, distant physical environment. This powerful control stack is often relegated to expensive full-sized robots, many of which are inaccessible to the research community. Miniature humanoids are more prevalent, but employ less biomimicry in their design (e.g. fewer sensors, Degrees of Freedom, etc) and lack similar developments. This paper describes a compliant full-body telepresence control stack developed from the ground up for miniature humanoids. Framework experimentation on ROBOTIS OP3 hardware showcases walking at speeds up to 0.45 m/s independent of arm motions. Tele-loco-manipulation is demonstrated via a cube relocation experiment with an expert human operator. On average, the teleoperated system moved 2 different 40 g cubes within 10 mins, walking a total distance of 5 m. Overall, the developed system shows potential for miniature humanoid tele-loco-manipulation.

**arXiv ID:** 2607.20399
</details>

<details>
<summary><strong>Self-Explaining Reinforcement Learning for Mobile Network Resource Allocation</strong> - Konrad Nowosadko, Franco Ruggeri, Ahmad Terra - [[pdf]](https://arxiv.org/pdf/2509.14925)</summary>

**Abstract:** Deep reinforcement learning (DRL) methods, though powerful, often lack transparency, which limits their adoption in critical domains. We apply Self-Explaining Neural Networks (SENNs) to RL by parametrizing the policy of a PPO agent with a SENN, producing intrinsic local explanations, and propose a method for aggregating them into global explanations. We evaluate our approach on a mobile network resource allocation problem, our approach performs within a small margin of the state-of-the-art deep learning method and significantly outperforms the best deployed heuristic, while the extracted global explanations correlate strongly with DeepLift and InputXGradient, making SENNs a promising candidate for high-stakes RL.

**arXiv ID:** 2509.14925
</details>

<details>
<summary><strong>One4Many-StablePacker: An Efficient Deep Reinforcement Learning Framework for the 3D Bin Packing Problem</strong> - Lei Gao, Shihong Huang, Shengjie Wang, Hong Ma, Feng Zhang, Hengda Bao, Qichang Chen, Weihua Zhou - [[pdf]](https://arxiv.org/pdf/2510.10057)</summary>

**Abstract:** The three-dimensional bin packing problem (3D-BPP) is widely applied in logistics and warehousing. Existing learning-based approaches often neglect practical stability-related constraints and exhibit limitations in generalizing across diverse bin dimensions. To address these limitations, we propose a novel deep reinforcement learning framework, One4Many-StablePacker (O4M-SP). The primary advantage of O4M-SP is its ability to handle various bin dimensions in a single training process while incorporating support and weight constraints common in practice. Our training method introduces two innovative mechanisms. First, it employs a weighted reward function that integrates loading rate and a new height difference metric for packing layouts, promoting improved bin utilization through flatter packing configurations. Second, it combines clipped policy gradient optimization with a tailored policy drifting method to mitigate policy entropy collapse, encouraging exploration at critical decision nodes during packing to avoid suboptimal solutions. Extensive experiments demonstrate that O4M-SP generalizes successfully across diverse bin dimensions and significantly outperforms baseline methods. Furthermore, O4M-SP exhibits strong practical applicability by effectively addressing packing scenarios with stability constraints.

**arXiv ID:** 2510.10057
</details>

<details>
<summary><strong>Safety-Regulated Transfer Reinforcement Learning with Adaptive Teacher Guidance</strong> - Wenjie Huang, Yang Li, Jingjia Teng, Mingwei Jin, Kai Song, Zeyu Yang, Qisong Yang, Yougang Bian - [[pdf]](https://arxiv.org/pdf/2606.26527)</summary>

**Abstract:** We propose Safety-Regulated Adaptive Transfer Reinforcement Learning (SRATRL), a teacher--student framework that combines safety-triggered intervention, safety-adaptive value shaping, and policy-compatibility-based optimization for efficient target-domain adaptation. First, a safety-triggered closed-loop intervention strategy is developed that activates teacher guidance according to the instantaneous safety cost and adaptively adjusts the intervention threshold based on the student policy's recent safety performance, thereby providing timely safety supervision while progressively restoring student autonomy as its safety improves. Next, a safety-adaptive teacher-guided value-shaping scheme is introduced, in which a teacher-consistency signal is incorporated into the critic target, and its contribution is dynamically regulated by the safety-constraint multiplier, enabling stronger teacher guidance under elevated safety risks and gradually weakening such guidance as the safety constraint is better satisfied. In addition, a teacher-student policy-compatibility weighting approach is proposed to alleviate the adverse optimization effects caused by policy mismatch. It reweights teacher-intervened transitions according to the relative likelihood of the executed action under the teacher and student policies, thereby improving policy-update stability. Experimental results demonstrate that compared with a Proximal Policy Optimization with Lagrangian constraint baseline, the proposed method improves the average velocity by 6.90%, and reduces the crash ratio by 75.00%. These results demonstrate that the proposed method can reduce safety costs while maintaining competitive task efficiency.

**arXiv ID:** 2606.26527
</details>

<details>
<summary><strong>Posterior Sampling Reinforcement Learning with Gaussian Processes for Continuous Control: Sublinear Regret Bounds for Unbounded State Spaces</strong> - Hamish Flynn, Joe Watson, Ingmar Posner, Jan Peters - [[pdf]](https://arxiv.org/pdf/2603.08287)</summary>

**Abstract:** We analyze the Bayesian regret of the Gaussian process posterior sampling reinforcement learning (GP-PSRL) algorithm. Posterior sampling is a heuristic for decision-making under uncertainty that has been used to develop successful algorithms for a variety of continuous control problems. However, theoretical work on GP-PSRL is limited. All known regret bounds either have a sub-optimal growth rate, require strong smoothness assumptions, or fail to properly account for the fact that the set of possible system states is unbounded. Through a recursive application of the Borell-Tsirelson-Ibragimov-Sudakov inequality, we show that, with high probability, the states actually visited by the algorithm are contained within a ball of near-constant radius. We then use the chaining method to control the regret suffered by GP-PSRL under weak smoothness conditions. Our main result is a Bayesian regret bound of the order $\widetilde{\mathcal{O}}(H\sqrt{\gamma_TT})$, where $H$ is the horizon, $T$ is the number of time steps and $\gamma_T$ is the expected information gain. With this result, we resolve the limitations with prior theoretical work on PSRL, and provide the theoretical foundation and tools for analyzing PSRL in complex settings.

**arXiv ID:** 2603.08287
</details>

<details>
<summary><strong>Cognitive Dual-Process Planning for Autonomous Driving with Structured Scene Knowledge and Verifiable Reasoning-Action Consistency</strong> - Zhongyao Yang, Haoyu Li, Yu Yan, Zhuangxuan Yu, Jiangfeng Nan, Jinrui Nan - [[pdf]](https://arxiv.org/pdf/2607.19194)</summary>

**Abstract:** High-level planning for autonomous driving is a knowledge-intensive engineering decision task that requires accurate scene understanding, timely inference, and internally consistent action selection. Vision-language models (VLMs) can make intermediate reasoning explicit, but their use in deployed planners is constrained by costly structured supervision, unnecessary reasoning in routine scenes, and possible inconsistencies between generated rationales and driving actions. We present a cognitive dual-process planning framework that represents planning-relevant scene knowledge in a machine-parsable structured chain-of-thought (S-CoT) schema. An automated data engine integrates perception foundation models, critical-path filtering, and an expert VLM to generate S-CoT supervision without manual annotation of individual rationales. A lightweight visual Arbiter estimates scene complexity from multilevel vision-encoder features before language decoding and routes each input to either fast meta-action prediction or slow structured reasoning. For slow-path outputs, a deterministic rule-based validator checks whether the parsed S-CoT fields are consistent with the final meta-action and provides verifiable rewards for Group Relative Policy Optimization (GRPO). In a 195-scene manual audit, the generated annotations achieve 91.8\% CoT accuracy and a 98.5\% Logical Consistency Score (LCS). On 574 manually verified NAVSIM test samples, the planner achieves 80.14\% planning accuracy and 97.20\% LCS while reducing average latency by 17.39\% relative to applying slow reasoning to every scene. Evaluation on external long-tail subsets further identifies conditions under which routing and planning performance degrade. Together, these results show how explicit scene knowledge can be operationalized through adaptive reasoning and rule-based verification to support high-level VLM planning decisions.

**arXiv ID:** 2607.19194
</details>

<details>
<summary><strong>ChatMuse: Supporting In-Person Small-Group Conversation Experience with a Proactive Assistive AI Agent in Mixed Reality</strong> - Shaoze Zhou, Joaquin Frangi, Diana Nelly Rivera Rodriguez, Rawan Alghofaili, Janet G. Johnson, Lingyao Li, Renkai Ma, Christine Lisetti, Chen Chen - [[pdf]](https://arxiv.org/pdf/2607.18556)</summary>

**Abstract:** In-person small-group conversations occur across nearly every aspect of daily life and play a crucial role in social interaction. However, achieving effective in-person group conversations can be challenging and cognitively demanding. While recent Mixed Reality (MR) headsets show promise as a conversational support system by presenting relevant information through overlays, it remains unclear how such supporting information should be designed and generated for in-person group conversations. We propose ChatMuse, a novel MR-based proactive assistive system for in-person small-group conversation experience. ChatMuse analyzes verbal and non-verbal cues from all conversation participants and proactively provides real-time guidance on the user's verbal and non-verbal behaviors. The behavioral responses of the supported users are then used to improve ChatMuse's support capabilities in subsequent interactions. We conducted a within-subject study to evaluate and demonstrate the feasibility and effectiveness of ChatMuse in assisting users to engage in and contribute to in-person small-group conversations. Our research around ChatMuse represents a design exploration of a new interaction space that investigates the feasibility of supporting in-person small-group conversations through a proactive assistive AI agent in MR.

**arXiv ID:** 2607.18556
</details>

<details>
<summary><strong>Designing for What Cannot Be Seen: Supporting Embodied String Learning for Musicians with Blindness and Low-Vision</strong> - Shi Shi, Lingyun Chen, Zitao Zhang, Amanda R. Draper, Eli Blevis - [[pdf]](https://arxiv.org/pdf/2607.18598)</summary>

**Abstract:** Bowed string instruments demand fine-grained bodily coordination that is typically taught through visual demonstration, creating persistent barriers for musicians with blindness and low-vision (BLV). To understand these challenges and explore new design opportunities, we conducted a design study with four advanced string musicians with BLV and three of their instructors. Our team, spanning violin performance and music education, disability studies in music, HCI design, and engineering employed a qualitative, multi-method approach including practice video analysis, lesson observation, expert interviews. Our analysis identifies recurring difficulties in right-hand bow control, left-hand coordination, score access, and memory-intensive practice. Building on these findings, we conducted an exploratory design ideation phase informed by empirical findings and feedback from one musician with BLV. We developed speculative design directions that could potentially address identified breakdowns, while acknowledging that these concepts require further evaluation with instructors and in deployed contexts.

**arXiv ID:** 2607.18598
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
