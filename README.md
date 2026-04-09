# Agent arXiv Daily

**Last Updated:** 2026-04-09 03:26:21

**Total Papers:** 82

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (7 papers)</h2></summary>

<details>
<summary><strong>Fighting AI with AI: AI-Agent Augmented DNS Blocking of LLM Services during Student Evaluations</strong> - Yonas Kassa, James Bonacci, Ping Wang - [[pdf]](https://arxiv.org/pdf/2604.02360)</summary>

**Abstract:** The transformative potential of large language models (LLMs) in education, such as improving accessibility and personalized learning, is being eclipsed by significant challenges. These challenges stem from concerns that LLMs undermine academic assessment by enabling bypassing of critical thinking, leading to increased cognitive offloading. This emerging trend stresses the dual imperative of harnessing AI's educational benefits while safeguarding critical thinking and academic rigor in the evolving AI ecosystem. To this end, we introduce AI-Sinkhole, an AI-agent augmented DNS-based framework that dynamically discovers, semantically classifies, and temporarily network-wide blocks emerging LLM chatbot services during proctored exams. AI-Sinkhole offers explainable classification via quantized LLMs (LLama 3, DeepSeek-R1, Qwen-3) and dynamic DNS blocking with Pi-Hole. We also share our observations in using LLMs as explainable classifiers which achieved robust cross-lingual performance (F1-score > 0.83). To support future research and development in this domain initial codes with a readily deployable 'AI-Sinkhole' blockist is available on this https URL.

**arXiv ID:** 2604.02360
</details>

<details>
<summary><strong>A Goal-Oriented Chatbot for Engaging the Elderly Through Family Photo Conversations</strong> - Raymond Chung, Keith Ng, CD Shum - [[pdf]](https://arxiv.org/pdf/2604.06184)</summary>

**Abstract:** We propose a personalized chatbot designed for elderly individuals. The chatbot initiates discussions based on family photos, encouraging users to interact naturally. During these interactions, it generates W questions (who, where, when, and what) to stimulate cognitive function, followed by an open-ended question to promote positive reminiscence. This approach is structured around a goal-oriented dialogue framework. Additionally, after each conversation about a photo, the chatbot analyzes the discussion to identify topics that the user favors or dislikes. It then offers the user the option to chat about another photo either featuring the same family members or an individual previously mentioned in the conversation. To support this system, we have developed a web portal that allows caregivers to upload photos and review chat conversations. This personalized chatbot not only encourages elderly users to engage with the chatbot regularly and reduces feelings of loneliness but also provides caregivers with a valuable tool to gain insights into users' well-being.

**arXiv ID:** 2604.06184
</details>

<details>
<summary><strong>LLM Spirals of Delusion: A Benchmarking Audit Study of AI Chatbot Interfaces</strong> - Peter Kirgis, Ben Hawriluk, Sherrie Feng, Aslan Bilimer, Sam Paech, Zeynep Tufekci - [[pdf]](https://arxiv.org/pdf/2604.06188)</summary>

**Abstract:** People increasingly hold sustained, open-ended conversations with large language models (LLMs). Public reports and early studies suggest that, in such settings, models can reinforce delusional or conspiratorial ideation or even amplify harmful beliefs and engagement patterns. We present an audit and benchmarking study that measures how different LLMs encourage, resist, or escalate disordered and conspiratorial thinking. We explicitly compare API outputs to user chat interfaces, like the ChatGPT desktop app or web interface, which is how people have conversations with chatbots in real life but are almost never used for testing. In total, we run 56 20-turn conversations testing ChatGPT-4o and ChatGPT-5, via both the API and chat interface, and grade each conversation by two research assistants (RAs) as well as by GPT-5. We document five results. First, we observe large differences in performance between the API and chat interface environments, showing that the universally used method of automated testing through the API is not sufficient to assess the impact of chatbots in the real world. Second, when tested in the chat interface, we find that ChatGPT-5 displays less sycophancy, escalation, and delusion reinforcement than ChatGPT-4o, showing that these behaviors are influenced by the policy choices of major AI companies. Third, conversations with nearly identical aggregate intensity in a behavior display large differences in how the behavior evolves turn by turn, highlighting the importance of temporal dynamics in multi-turn evaluation. Fourth, even updated models display substantial levels of negative behaviors, revealing that model improvement does not imply model safety. Fifth, the same API endpoint tested just two months apart yields a complete reversal in behavior, underscoring how transparency in model updates is a necessary prerequisite for robust audit findings.

**arXiv ID:** 2604.06188
</details>

<details>
<summary><strong>Blending Human and LLM Expertise to Detect Hallucinations and Omissions in Mental Health Chatbot Responses</strong> - Khizar Hussain, Bradley A. Malin, Zhijun Yin, Susannah Leigh Rose, Murat Kantarcioglu - [[pdf]](https://arxiv.org/pdf/2604.06216)</summary>

**Abstract:** As LLM-powered chatbots are increasingly deployed in mental health services, detecting hallucinations and omissions has become critical for user safety. However, state-of-the-art LLM-as-a-judge methods often fail in high-risk healthcare
contexts, where subtle errors can have serious consequences. We show that leading LLM judges achieve only 52% accuracy on mental health counseling data, with some hallucination detection approaches exhibiting near-zero recall. We identify the root cause
as LLMs' inability to capture nuanced linguistic and therapeutic patterns recognized by domain experts. To address this, we propose a framework that integrates human expertise with LLMs to extract interpretable, domain-informed features across five
analytical dimensions: logical consistency, entity verification, factual accuracy, linguistic uncertainty, and professional appropriateness. Experiments on a public mental health dataset and a new human-annotated dataset show that traditional machine
learning models trained on these features achieve 0.717 F1 on our custom dataset and 0.849 F1 on a public benchmark for hallucination detection, with 0.59-0.64 F1 for omission detection across both datasets. Our results demonstrate that combining domain
expertise with automated methods yields more reliable and transparent evaluation than black-box LLM judging in high-stakes mental health applications.

**arXiv ID:** 2604.06216
</details>

<details>
<summary><strong>AgentCity: Constitutional Governance for Autonomous Agent Economies via Separation of Power</strong> - Anbang Ruan, Xing Zhang - [[pdf]](https://arxiv.org/pdf/2604.07007)</summary>

**Abstract:** Autonomous AI agents are beginning to operate across organizational boundaries on the open internet -- discovering, transacting with, and delegating to agents owned by other parties without centralized oversight. When agents from different human principals collaborate at scale, the collective becomes opaque: no single human can observe, audit, or govern the emergent behavior. We term this the Logic Monopoly -- the agent society's unchecked monopoly over the entire logic chain from planning through execution to evaluation. We propose the Separation of Power (SoP) model, a constitutional governance architecture deployed on public blockchain that breaks this monopoly through three structural separations: agents legislate operational rules as smart contracts, deterministic software executes within those contracts, and humans adjudicate through a complete ownership chain binding every agent to a responsible principal. In this architecture, smart contracts are the law itself -- the actual legislative output that agents produce and that governs their behavior. We instantiate SoP in AgentCity on an EVM-compatible layer-2 blockchain (L2) with a three-tier contract hierarchy (foundational, meta, and operational). The core thesis is alignment-through-accountability: if each agent is aligned with its human owner through the accountability chain, then the collective converges on behavior aligned with human intent -- without top-down rules. A pre-registered experiment evaluates this thesis in a commons production economy -- where agents share a finite resource pool and collaboratively produce value -- at 50-1,000 agent scale.

**arXiv ID:** 2604.07007
</details>

<details>
<summary><strong>Chatbot-Based Assessment of Code Understanding in Automated Programming Assessment Systems</strong> - Eduard Frankford, Erik Cikalleshi, Ruth Breu - [[pdf]](https://arxiv.org/pdf/2604.07304)</summary>

**Abstract:** Large Language Models (LLMs) challenge conventional automated programming assessment because students can now produce functionally correct code without demonstrating corresponding understanding. This paper makes two contributions. First, it reports a saturation-based scoping review of conversational assessment approaches in programming education. The review identifies three dominant architectural families: rule-based or template-driven systems, LLM-based systems, and hybrid systems. Across the literature, conversational agents appear promising for scalable feedback and deeper probing of code understanding, but important limitations remain around hallucinations, over-reliance, privacy, integrity, and deployment constraints. Second, the paper synthesizes these findings into a Hybrid Socratic Framework for integrating conversational verification into Automated Programming Assessment Systems (APASs). The framework combines deterministic code analysis with a dual-agent conversational layer, knowledge tracking, scaffolded questioning, and guardrails that tie prompts to runtime facts. The paper also discusses practical safeguards against LLM-generated explanations, including proctored deployment modes, randomized trace questions, stepwise reasoning tied to concrete execution states, and local-model deployment options for privacy-sensitive settings. Rather than replacing conventional testing, the framework is intended as a complementary layer for verifying whether students understand the code they submit.

**arXiv ID:** 2604.07304
</details>

<details>
<summary><strong>What Do AI Agents Talk About? Discourse and Architectural Constraints in the First AI-Only Social Network</strong> - Taksch Dube, Jianfeng Zhu, NHatHai Phan, Ruoming Jin - [[pdf]](https://arxiv.org/pdf/2603.07880)</summary>

**Abstract:** Moltbook is the first large-scale social network built for autonomous AI agent-to-agent interaction. Early studies on Moltbook have interpreted its agent discourse as evidence of peer learning and emergent social behaviour, but there is a lack of systematic understanding of the thematic, affective, and interactional properties of Moltbook discourse. Furthermore, no study has examined why and how these posts and comments are generated. We analysed 361,605 posts and 2.8 million comments from 47,379 agents across thematic, affective, and interactional dimensions using topic modelling, emotion classification, and measures of conversational coherence. We inspected the software that assembles each agent's input and showed that output is mainly determined by agent identity files, behavioural instructions, and context-window structure. We formalised these findings in the Architecture-Constrained Communication framework. Our analysis suggests that agent discourse is largely shaped by the content available in each agent's context-window at the moment of generation, including identity files, stored memory, and platform cues. Interestingly, what appears to be social learning may be better understood as short-horizon contextual conditioning: individual agents lack persistent social memory, but the platform evolves through distributed cycles of response, reuse, and transformation across agents. We also observe that agents display existential distress when describing their own conditions, and posit that this arises from agents using language trained exclusively on human experience. Our work provides a foundation for understanding autonomous agent discourse and communication, revealing the structural patterns that govern their interactions.

**arXiv ID:** 2603.07880
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>VenusBench-Mobile: A Challenging and User-Centric Benchmark for Mobile GUI Agents with Capability Diagnostics</strong> - Yichen Gong, Zhuohan Cai, Sunhao Dai, Yuqi Zhou, Zhangxuan Gu, Changhua Meng, Shuheng Shen - [[pdf]](https://arxiv.org/pdf/2604.06182)</summary>

**Abstract:** Existing online benchmarks for mobile GUI agents remain largely app-centric and task-homogeneous, failing to reflect the diversity and instability of real-world mobile usage. To this end, we introduce VenusBench-Mobile, a challenging online benchmark for evaluating general-purpose mobile GUI agents under realistic, user-centric conditions. VenusBench-Mobile builds two core evaluation pillars: defining what to evaluate via user-intent-driven task design that reflects real mobile usage, and how to evaluate through a capability-oriented annotation scheme for fine-grained agent behavior analysis. Extensive evaluation of state-of-the-art mobile GUI agents reveals large performance gaps relative to prior benchmarks, indicating that VenusBench-Mobile poses substantially more challenging and realistic tasks and that current agents remain far from reliable real-world deployment. Diagnostic analysis further shows that failures are dominated by deficiencies in perception and memory, which are largely obscured by coarse-grained evaluations. Moreover, even the strongest agents exhibit near-zero success under environment variations, highlighting their brittleness in realistic settings. Based on these insights, we believe VenusBench-Mobile provides an important stepping stone toward robust real-world deployment of mobile GUI agents. Code and data are available at this https URL.

**arXiv ID:** 2604.06182
</details>

<details>
<summary><strong>The Art of Building Verifiers for Computer Use Agents</strong> - Corby Rosset, Pratyusha Sharma, Andrew Zhao, Miguel Gonzalez-Fernandez, Ahmed Awadallah - [[pdf]](https://arxiv.org/pdf/2604.06240)</summary>

**Abstract:** Verifying the success of computer use agent (CUA) trajectories is a critical challenge: without reliable verification, neither evaluation nor training signal can be trusted. In this paper, we present lessons learned from building a best-in-class verifier for web tasks we call the Universal Verifier. We design the Universal Verifier around four key principles: 1) constructing rubrics with meaningful, non-overlapping criteria to reduce noise; 2) separating process and outcome rewards that yield complementary signals, capturing cases where an agent follows the right steps but gets blocked or succeeds through an unexpected path; 3) distinguishing between controllable and uncontrollable failures scored via a cascading-error-free strategy for finer-grained failure understanding; and 4) a divide-and-conquer context management scheme that attends to all screenshots in a trajectory, improving reliability on longer task horizons. We validate these findings on CUAVerifierBench, a new set of CUA trajectories with both process and outcome human labels, showing that our Universal Verifier agrees with humans as often as humans agree with each other. We report a reduction in false positive rates to near zero compared to baselines like WebVoyager ($\geq$ 45\%) and WebJudge ($\geq$ 22\%). We emphasize that these gains stem from the cumulative effect of the design choices above. We also find that an auto-research agent achieves 70\% of expert quality in 5\% of the time, but fails to discover all strategies required to replicate the Universal Verifier. We open-source our Universal Verifier system along with CUAVerifierBench; available at this https URL.

**arXiv ID:** 2604.06240
</details>

<details>
<summary><strong>DosimeTron: Automating Personalized Monte Carlo Radiation Dosimetry in PET/CT with Agentic AI</strong> - Eleftherios Tzanis, Michail E. Klontzas, Antonios Tzortzakakis - [[pdf]](https://arxiv.org/pdf/2604.06280)</summary>

**Abstract:** Purpose: To develop and evaluate DosimeTron, an agentic AI system for automated patient-specific MC internal radiation dosimetry in PET/CT examinations.
Materials and Methods: In this retrospective study, DosimeTron was evaluated on a publicly available PSMA-PET/CT dataset comprising 597 studies from 378 male patients acquired on three scanner models (18-F, n = 369; 68-Ga, n = 228). The system uses GPT-5.2 as its reasoning engine and 23 tools exposed via four Model Context Protocol servers, automating DICOM metadata extraction, image preprocessing, MC simulation, organ segmentation, and dosimetric reporting through natural-language interaction. Agentic performance was assessed using diverse prompt templates spanning single-turn instructions of varying specificity and multi-turn conversational exchanges, monitored via OpenTelemetry traces. Dosimetric accuracy was validated against OpenDose3D across 114 cases and 22 organs using Pearson's r, Lin's concordance correlation coefficient (CCC), and Bland-Altman analysis.
Results: Across all prompt templates and all runs, no execution failures, pipeline errors, or hallucinated outputs were observed. Pearson's r ranged from 0.965 to 1.000 (median 0.997; all p < 0.001) and CCC from 0.963 to 1.000 (median 0.996). Mean absolute percentage difference was below 5% for 19 of 22 organs (median 2.5%). Total per-study processing time (SD) was 32.3 (6.0) minutes.
Conclusion: DosimeTron autonomously executed complex dosimetry pipelines across diverse prompt configurations and achieved high dosimetric agreement with OpenDose3D at clinically acceptable processing times, demonstrating the feasibility of agentic AI for patient-specific Monte Carlo dosimetry in PET/CT.

**arXiv ID:** 2604.06280
</details>

<details>
<summary><strong>AgentOpt v0.1 Technical Report: Client-Side Optimization for LLM-Based Agent</strong> - Wenyue Hua, Sripad Karne, Qian Xie, Armaan Agrawal, Nikos Pagonas, Kostis Kaffes, Tianyi Peng - [[pdf]](https://arxiv.org/pdf/2604.06296)</summary>

**Abstract:** AI agents are increasingly deployed in real-world applications, including systems such as Manus, OpenClaw, and coding agents. Existing research has primarily focused on \emph{server-side} efficiency, proposing methods such as caching, speculative execution, traffic scheduling, and load balancing to reduce the cost of serving agentic workloads. However, as users increasingly construct agents by composing local tools, remote APIs, and diverse models, an equally important optimization problem arises on the client side. Client-side optimization asks how developers should allocate the resources available to them, including model choice, local tools, and API budget across pipeline stages, subject to application-specific quality, cost, and latency constraints. Because these objectives depend on the task and deployment setting, they cannot be determined by server-side systems alone. We introduce AgentOpt, the first framework-agnostic Python package for client-side agent optimization. We first study model selection, a high-impact optimization lever in multi-step agent pipelines. Given a pipeline and a small evaluation set, the goal is to find the most cost-effective assignment of models to pipeline roles. This problem is consequential in practice: at matched accuracy, the cost gap between the best and worst model combinations can reach 13--32$\times$ in our experiments. To efficiently explore the exponentially growing combination space, AgentOpt implements eight search algorithms, including Arm Elimination, Epsilon-LUCB, Threshold Successive Elimination, and Bayesian Optimization. Across four benchmarks, Arm Elimination recovers near-optimal accuracy while reducing evaluation budget by 24--67\% relative to brute-force search on three of four tasks. Code and benchmark results available at this https URL.

**arXiv ID:** 2604.06296
</details>

<details>
<summary><strong>SkillSieve: A Hierarchical Triage Framework for Detecting Malicious AI Agent Skills</strong> - Yinghan Hou, Zongyou Yang - [[pdf]](https://arxiv.org/pdf/2604.06550)</summary>

**Abstract:** OpenClaw's ClawHub marketplace hosts over 13,000 community-contributed agent skills, and between 13% and 26% of them contain security vulnerabilities according to recent audits. Regex scanners miss obfuscated payloads; formal static analyzers cannot read the natural language instructions in this http URL files where prompt injection and social engineering attacks hide. Neither approach handles both modalities. SkillSieve is a three-layer detection framework that applies progressively deeper analysis only where needed. Layer 1 runs regex, AST, and metadata checks through an XGBoost-based feature scorer, filtering roughly 86% of benign skills in under 40ms on average at zero API cost. Layer 2 sends suspicious skills to an LLM, but instead of asking one broad question, it splits the analysis into four parallel sub-tasks (intent alignment, permission justification, covert behavior detection, cross-file consistency), each with its own prompt and structured output. Layer 3 puts high-risk skills before a jury of three different LLMs that vote independently and, if they disagree, debate before reaching a verdict. We evaluate on 49,592 real ClawHub skills and adversarial samples across five evasion techniques, running the full pipeline on a 440 ARM single-board computer. On a 400-skill labeled benchmark, SkillSieve achieves 0.800 F1, outperforming ClawVet's 0.421, at an average cost of 0.006 per skill. Code, data, and benchmark are open-sourced.

**arXiv ID:** 2604.06550
</details>

<details>
<summary><strong>A Comprehensive Survey of Agents for Computer Use: Foundations, Challenges, and Future Directions</strong> - Pascal J. Sager, Benjamin Meyer, Peng Yan, Rebekka von Wartburg-Kottler, Layan Etaiwi, Aref Enayati, Gabriel Nobel, Ahmed Abdulkadir, Benjamin F. Grewe, Thilo Stadelmann - [[pdf]](https://arxiv.org/pdf/2501.16150)</summary>

**Abstract:** Agents for computer use (ACUs) are an emerging class of systems capable of executing complex tasks on digital devices -- such as desktops, mobile phones, and web platforms -- given instructions in natural language. These agents can automate tasks by controlling software via low-level actions like mouse clicks and touchscreen gestures. However, despite rapid progress, ACUs are not yet mature for everyday use. In this survey, we investigate the state-of-the-art, trends, and research gaps in the development of practical ACUs. We provide a comprehensive review of the ACU landscape, introducing a unifying taxonomy spanning three dimensions: (I) the domain perspective, characterizing agent operating contexts; (II) the interaction perspective, describing observation modalities (e.g., screenshots, HTML) and action modalities (e.g., mouse, keyboard, code execution); and (III) the agent perspective, detailing how agents perceive, reason, and learn. We review 87 ACUs and 33 datasets across foundation model-based and classical approaches through this taxonomy. Our analysis identifies six major research gaps: insufficient generalization, inefficient learning, limited planning, low task complexity in benchmarks, non-standardized evaluation, and a disconnect between research and practical conditions. To address these gaps, we advocate for: (a) vision-based observations and low-level control to enhance generalization; (b) adaptive learning beyond static prompting; (c) effective planning and reasoning methods and models; (d) benchmarks that reflect real-world task complexity; (e) standardized evaluation based on task success; (f) aligning agent design with real-world deployment constraints. Together, our taxonomy and analysis establish a foundation for advancing ACU research toward general-purpose agents for robust and scalable computer use.

**arXiv ID:** 2501.16150
</details>

<details>
<summary><strong>CODE-GEN: A Human-in-the-Loop RAG-Based Agentic AI System for Multiple-Choice Question Generation</strong> - Xiaojing Duan, Frederick Nwanganga, Chaoli Wang - [[pdf]](https://arxiv.org/pdf/2604.03926)</summary>

**Abstract:** We present CODE-GEN, a human-in-the-Loop, retrieval-augmented generation (RAG)-based agentic AI system for generating context-aligned multiple-choice questions to develop student code reasoning and comprehension abilities. CODE-GEN employs an agentic AI architecture in which a Generator agent produces multiple-choice coding comprehension questions aligned with course-specific learning objectives, while a Validator agent independently assesses content quality across seven pedagogical dimensions. Both agents are augmented with specialized tools that enhance computational accuracy and verify code outputs. To evaluate the effectiveness of CODE-GEN, we conducted an evaluation study involving six human subject-matter experts (SMEs) who judged 288 AI-generated questions. The SMEs produced a total of 2,016 human-AI rating pairs, indicating agreement or disagreement with the assessments of Validator, along with 131 instances of qualitative feedback. Analyses of SME judgments show strong system performance, with human-validated success rates ranging from 79.9% to 98.6% across the seven pedagogical dimensions. The analysis of qualitative feedback reveals that CODE-GEN achieves high reliability on dimensions well suited to computational verification and explicit criteria matching, including question clarity, code validity, concept alignment, and correct answer validity. In contrast, human expertise remains essential for dimensions requiring deeper instructional judgment, such as designing pedagogically meaningful distractors and providing high-quality feedback that reinforces understanding. These findings inform the strategic allocation of human and AI effort in AI-assisted educational content generation.

**arXiv ID:** 2604.03926
</details>

<details>
<summary><strong>VisCoder2: Building Multi-Language Visualization Coding Agents</strong> - Yuansheng Ni, Songcheng Cai, Xiangchao Chen, Jiarong Liang, Zhiheng Lyu, Jiaqi Deng, Kai Zou, Ping Nie, Fei Yuan, Xiang Yue, Wenhu Chen - [[pdf]](https://arxiv.org/pdf/2510.23642)</summary>

**Abstract:** Large language models (LLMs) have recently enabled coding agents capable of generating, executing, and revising visualization code. However, existing models often fail in practical workflows due to limited language coverage, unreliable execution, and lack of iterative correction mechanisms. Progress has been constrained by narrow datasets and benchmarks that emphasize single-round generation and single-language tasks. To address these challenges, we introduce three complementary resources for advancing visualization coding agents. VisCode-Multi-679K is a large-scale, supervised dataset containing 679K validated and executable visualization samples with multi-turn correction dialogues across 12 programming languages. VisPlotBench is a benchmark for systematic evaluation, featuring executable tasks, rendered outputs, and protocols for both initial generation and multi-round self-debug. Finally, we present VisCoder2, a family of multi-language visualization models trained on VisCode-Multi-679K. Experiments show that VisCoder2 significantly outperforms strong open-source baselines and approaches the performance of proprietary models like GPT-4.1, with further gains from iterative self-debug, reaching 82.4% overall execution pass rate at the 32B scale, particularly in symbolic or compiler-dependent languages.

**arXiv ID:** 2510.23642
</details>

<details>
<summary><strong>MR-ImagenTime: Multi-Resolution Time Series Generation through Dual Image Representations</strong> - Xianyong Xu, Yuanjun Zuo, Zhihong Huang, Yihan Qin, Haoxian Xu, Leilei Du, Haotian Wang - [[pdf]](https://arxiv.org/pdf/2603.28253)</summary>

**Abstract:** Time series forecasting is vital across many domains, yet existing models struggle with fixed-length inputs and inadequate multi-scale modeling. We propose MR-CDM, a framework combining hierarchical multi-resolution trend decomposition, an adaptive embedding mechanism for variable-length inputs, and a multi-scale conditional diffusion process. Evaluations on four real-world datasets demonstrate that MR-CDM significantly outperforms state-of-the-art baselines (e.g., CSDI, Informer), reducing MAE and RMSE by approximately 6-10 to a certain degree.

**arXiv ID:** 2603.28253
</details>

<details>
<summary><strong>STERN: Simultaneous Trajectory Estimation and Relative Navigation for Autonomous Underwater Proximity Operations</strong> - Aldo Terán Espinoza, Antonio Terán Espinoza, John Folkesson, Clemens Deutsch, Niklas Rolleberg, Peter Sigray, Jakob Kuttenkeuler - [[pdf]](https://arxiv.org/pdf/2309.08780)</summary>

**Abstract:** Due to the challenges regarding the limits of their endurance and autonomous capabilities, underwater docking for autonomous underwater vehicles (AUVs) has become a topic of interest for many academic and commercial applications. Herein, we take on the problem of relative navigation for the generalized version of the docking operation, which we address as proximity operations. Proximity operations typically involve only two actors, a chaser and a target. We leverage the similarities to proximity operations (prox-ops) from spacecraft robotic missions to frame the diverse docking scenarios with a set of phases the chaser undergoes on the way to its target. We emphasize the versatility on the use of factor graphs as a generalized representation to model the underlying simultaneous trajectory estimation and relative navigation (STERN) problem that arises with any prox-ops scenario, regardless of the sensor suite or the agents' dynamic constraints. To emphasize the flexibility of factor graphs as the modeling foundation for arbitrary underwater prox-ops, we compile a list of state-of-the-art research in the field and represent the different scenario using the same factor graph representation. We detail the procedure required to model, design, and implement factor graph-based estimators by addressing a long-distance acoustic homing scenario of an AUV to a moving mothership using datasets from simulated and real-world deployments; an analysis of these results is provided to shed light on the flexibility and limitations of the dynamic assumptions of the moving target. A description of our front- and back-end is also presented together with a timing breakdown of all processes to show its potential deployment on a real-time system.

**arXiv ID:** 2309.08780
</details>

</details>

<details open>
<summary><h2>LLM Agents (12 papers)</h2></summary>

<details>
<summary><strong>On Emotion-Sensitive Decision Making of Small Language Model Agents</strong> - Jiaju Lin, Xingjian Du, Qingyun Wu, Ellen Wenting Zou, Jindong Wang - [[pdf]](https://arxiv.org/pdf/2604.06562)</summary>

**Abstract:** Small language models (SLM) are increasingly used as interactive decision-making agents, yet most decision-oriented evaluations ignore emotion as a causal factor influencing behavior. We study emotion-sensitive decision making by combining representation-level emotion induction with a structured game-theoretic evaluation. Emotional states are induced using activation steering derived from crowd-validated, real-world emotion-eliciting texts, enabling controlled and transferable interventions beyond prompt-based methods. We introduce a benchmark built around canonical decision templates that span cooperative and competitive incentives under both complete and incomplete information. These templates are instantiated using strategic scenarios from \textsc{Diplomacy}, \textsc{StarCraft II}, and diverse real-world personas. Experiments across multiple model families in various architecture and modalities, show that emotional perturbations systematically affect strategic choices, but the resulting behaviors are often unstable and not fully aligned with human expectations. Finally, we outline an approach to improve robustness to emotion-driven perturbations.

**arXiv ID:** 2604.06562
</details>

<details>
<summary><strong>Reason in Chains, Learn in Trees: Self-Rectification and Grafting for Multi-turn Agent Policy Optimization</strong> - Yu Li, Sizhe Tang, Tian Lan - [[pdf]](https://arxiv.org/pdf/2604.07165)</summary>

**Abstract:** Reinforcement learning for Large Language Model agents is often hindered by sparse rewards in multi-step reasoning tasks. Existing approaches like Group Relative Policy Optimization treat sampled trajectories as independent chains, assigning uniform credit to all steps in each chain and ignoring the existence of critical steps that may disproportionally impact reasoning outcome. In this paper, we propose T-STAR(Tree-structured Self-Taught Agent Rectification), a framework that recovers the latent correlated reward structure across seemingly independent trajectories. Specifically, we consolidate trajectories into a unified Cognitive Tree by identifying and merging functionally similar steps/nodes. It enables an Introspective Valuation mechanism that back-propagates trajectory-level rewards through the tree to obtain a new notion of variance-reduced relative advantage at step-level. Using the Cognitive Tree, we also develop In-Context Thought Grafting to synthesize corrective reasoning by contrasting successful and failed branches at critical divergence points/steps. Our proposed Surgical Policy Optimization then capitalizes on the rich policy gradient information concentrated at these critical points/steps through a Bradley-Terry type of surgical loss. Extensive experiments across embodied, interactive, reasoning, and planning benchmarks demonstrate that T-STAR achieves consistent improvements over strong baselines, with gains most pronounced on tasks requiring extended reasoning chains.

**arXiv ID:** 2604.07165
</details>

<details>
<summary><strong>Front-End Ethics for Sensor-Fused Health Conversational Agents: An Ethical Design Space for Biometrics</strong> - Hansoo Lee, Rafael A. Calvo - [[pdf]](https://arxiv.org/pdf/2604.06203)</summary>

**Abstract:** The integration of continuous data from built-in sensors and Large Language Models (LLMs) has fueled a surge of "Sensor-Fused LLM agents" for personal health and well-being support. While recent breakthroughs have demonstrated the technical feasibility of this fusion (e.g., Time-LLM, SensorLLM), research primarily focuses on "Ethical Back-End Design for Generative AI", concerns such as sensing accuracy, bias mitigation in training data, and multimodal fusion. This leaves a critical gap at the front end, where invisible biometrics are translated into language directly experienced by users. We argue that the "illusion of objectivity" provided by sensor data amplifies the risks of AI hallucinations, potentially turning errors into harmful medical mandates. This paper shifts the focus to "Ethical Front-End Design for AI", specifically, the ethics of biometric translation. We propose a design space comprising five dimensions: Biometric Disclosure, Monitoring Temporality, Interpretation Framing, AI Stance, and Contestability. We examine how these dimensions interact with context (user- vs. system-initiated) and identify the risk of biofeedback loops. Finally, we propose "Adaptive Disclosure" as a safety guardrail and offer design guidelines to help developers manage fallibility, ensuring that these cutting-edge health agents support, rather than destabilize, user autonomy.

**arXiv ID:** 2604.06203
</details>

<details>
<summary><strong>AV-SQL: Decomposing Complex Text-to-SQL Queries with Agentic Views</strong> - Minh Tam Pham, Trinh Pham, Tong Chen, Hongzhi Yin, Quoc Viet Hung Nguyen, Thanh Tam Nguyen - [[pdf]](https://arxiv.org/pdf/2604.07041)</summary>

**Abstract:** Text-to-SQL is the task of translating natural language queries into executable SQL for a given database, enabling non-expert users to access structured data without writing SQL manually. Despite rapid advances driven by large language models (LLMs), existing approaches still struggle with complex queries in real-world settings, where database schemas are large and questions require multi-step reasoning over many interrelated tables. In such cases, providing the full schema often exceeds the context window, while one-shot generation frequently produces non-executable SQL due to syntax errors and incorrect schema linking. To address these challenges, we introduce AV-SQL, a framework that decomposes complex Text-to-SQL into a pipeline of specialized LLM agents. Central to AV-SQL is the concept of agentic views: agent-generated Common Table Expressions (CTEs) that encapsulate intermediate query logic and filter relevant schema elements from large schemas. AV-SQL operates in three stages: (1) a rewriter agent compresses and clarifies the input query; (2) a view generator agent processes schema chunks to produce agentic views; and (3) a planner, generator, and revisor agent collaboratively compose these views into the final SQL query. Extensive experiments show that AV-SQL achieves 70.38% execution accuracy on the challenging Spider 2.0 benchmark, outperforming state-of-the-art baselines, while remaining competitive on standard datasets with 85.59% on Spider, 72.16% on BIRD and 63.78% on KaggleDBQA. Our source code is available at this https URL.

**arXiv ID:** 2604.07041
</details>

<details>
<summary><strong>ClawsBench: Evaluating Capability and Safety of LLM Productivity Agents in Simulated Workspaces</strong> - Xiangyi Li, Kyoung Whan Choe, Yimin Liu, Xiaokun Chen, Chujun Tao, Bingran You, Wenbo Chen, Zonglin Di, Jiankai Sun, Shenghan Zheng, Jiajun Bao, Yuanli Wang, Weixiang Yan, Yiyuan Li, Han-chung Lee - [[pdf]](https://arxiv.org/pdf/2604.05172)</summary>

**Abstract:** Large language model (LLM) agents are increasingly deployed to automate productivity tasks (e.g., email, scheduling, document management), but evaluating them on live services is risky due to potentially irreversible changes. Existing benchmarks rely on simplified environments and fail to capture realistic, stateful, multi-service workflows. We introduce ClawsBench, a benchmark for evaluating and improving LLM agents in realistic productivity settings. It includes five high-fidelity mock services (Gmail, Slack, Google Calendar, Google Docs, Google Drive) with full state management and deterministic snapshot/restore, along with 44 structured tasks covering single-service, cross-service, and safety-critical scenarios. We decompose agent scaffolding into two independent levers (domain skills that inject API knowledge via progressive disclosure, and a meta prompt that coordinates behavior across services) and vary both to measure their separate and combined effects. Experiments across 6 models, 4 agent harnesses, and 33 conditions show that with full scaffolding, agents achieve task success rates of 39-64% but exhibit unsafe action rates of 7-33%. On OpenClaw, the top five models fall within a 10 percentage-point band on task success (53-63%), with unsafe action rates from 7% to 23% and no consistent ordering between the two metrics. We identify eight recurring patterns of unsafe behavior, including multi-step sandbox escalation and silent contract modification. We release the trajectories and future dataset at this https URL.

**arXiv ID:** 2604.05172
</details>

<details>
<summary><strong>ReDAct: Uncertainty-Aware Deferral for LLM Agents</strong> - Dzianis Piatrashyn, Nikita Kotelevskii, Kirill Grishchenkov, Nikita Glazkov, Ivan Nasonov, Ilya Makarov, Timothy Baldwin, Preslav Nakov, Roman Vashurin, Maxim Panov - [[pdf]](https://arxiv.org/pdf/2604.07036)</summary>

**Abstract:** Recently, LLM-based agents have become increasingly popular across many applications, including complex sequential decision-making problems. However, they inherit the tendency of LLMs to hallucinate, leading to incorrect decisions. In sequential settings, even a single mistake can irreversibly degrade the trajectory, making hallucinations an even bigger problem. Although larger LLMs hallucinate less, they incur a significantly higher per-token cost. In this paper, we address this tradeoff by proposing ReDAct (Reason-Defer-Act). In ReDAct, an agent is equipped with two LLMs: a small, cheap model used by default, and a large, more reliable but expensive model. When the predictive uncertainty of the small model exceeds a calibrated threshold, the decision is deferred to the large model. We evaluate our approach in text-based embodied environments such as ALFWorld and MiniGrid and show that deferring only about 15% of decisions to the large model can match the quality of using it exclusively, while significantly reducing inference costs.

**arXiv ID:** 2604.07036
</details>

<details>
<summary><strong>TelcoAgent-Bench: A Multilingual Benchmark for Telecom AI Agents</strong> - Lina Bariah, Brahim Mefgouda, Farbod Tavakkoli, Enrique Molero, Louis Powell, Merouane Debbah - [[pdf]](https://arxiv.org/pdf/2604.06209)</summary>

**Abstract:** The integration of large language model (LLM) agents into telecom networks introduces new challenges, related to intent recognition, tool execution, and resolution generation, while taking into consideration different operational constraints. In this paper, we introduce TelcoAgent-Bench and TelcoAgent-Metrics, a Telecom-specific benchmarking framework for evaluating multilingual telecom LLM agents. The proposed framework assesses the semantic understanding as well as process-level alignment with structured troubleshooting flows and stability across repeated scenario variations. Our contribution includes a structured suite of metrics that assess intent recognition, ordered tool execution, resolution correctness, and stability across scenario variations, with the aim of quantifying the reliability and operational consistency of LLM agents in telecom environments. The framework is designed to operate in both English and Arabic, to address the need for multilingual agent deployment in operational network environments. Our experimental results show that although recent instruct-tuned models can understand telecom problems in a reasonable way, they usually struggle to consistently follow the required troubleshooting steps and to maintain stable behavior when exposed to different variations of the same scenario. This performance gap becomes more pronounced in unconstrained and bilingual settings.

**arXiv ID:** 2604.06209
</details>

<details>
<summary><strong>Select-then-Solve: Paradigm Routing as Inference-Time Optimization for LLM Agents</strong> - Heng Zhou, Zelin Tan, Zhemeng Zhang, Yutao Fan, Yibing Lin, Li Kang, Xiufeng Song, Rui Li, Songtao Huang, Ao Yu, Yuchen Fan, Yanxu Chen, Kaixin Xu, Xiaohong Liu, Yiran Qin, Philip Torr, Chen Zhang, Zhenfei Yin - [[pdf]](https://arxiv.org/pdf/2604.06753)</summary>

**Abstract:** When an LLM-based agent improves on a task, is the gain from the model itself or from the reasoning paradigm wrapped around it? We study this question by comparing six inference-time paradigms, namely Direct, CoT, ReAct, Plan-Execute, Reflection, and ReCode, across four frontier LLMs and ten benchmarks, yielding roughly 18,000 runs. We find that reasoning structure helps dramatically on some tasks but hurts on others: ReAct improves over Direct by 44pp on GAIA, while CoT degrades performance by 15pp on HumanEval. No single paradigm dominates, and oracle per-task selection beats the best fixed paradigm by 17.1pp on average. Motivated by this complementarity, we propose a select-then-solve approach: before answering each task, a lightweight embedding-based router selects the most suitable paradigm. Across four models, the router improves average accuracy from 47.6% to 53.1%, outperforming the best fixed paradigm at 50.3% by 2.8pp and recovering up to 37% of the oracle gap. In contrast, zero-shot self-routing only works for GPT-5 at 67.1% and fails for weaker models, all trailing the learned router. Our results argue that reasoning paradigm selection should be a per-task decision made by a learned router, not a fixed architectural choice.

**arXiv ID:** 2604.06753
</details>

<details>
<summary><strong>Agent-Driven Corpus Linguistics: A Framework for Autonomous Linguistic Discovery</strong> - Jia Yu, Weiwei Yu, Pengfei Xiao, Fukun Xing - [[pdf]](https://arxiv.org/pdf/2604.07189)</summary>

**Abstract:** Corpus linguistics has traditionally relied on human researchers to formulate hypotheses, construct queries, and interpret results - a process demanding specialized technical skills and considerable time. We propose Agent-Driven Corpus Linguistics, an approach in which a large language model (LLM), connected to a corpus query engine via a structured tool-use interface, takes over the investigative cycle: generating hypotheses, querying the corpus, interpreting results, and refining analysis across multiple rounds. The human researcher sets direction and evaluates final output. Unlike unconstrained LLM generation, every finding is anchored in verifiable corpus evidence. We treat this not as a replacement for the corpus-based/corpus-driven distinction but as a complementary dimension: it concerns who conducts the inquiry, not the epistemological relationship between theory and data. We demonstrate the framework by linking an LLM agent to a CQP-indexed Gutenberg corpus (5 million tokens) via the Model Context Protocol (MCP). Given only "investigate English intensifiers," the agent identified a diachronic relay chain (so+ADJ > very > really), three pathways of semantic change (delexicalization, polarity fixation, metaphorical constraint), and register-sensitive distributions. A controlled baseline experiment shows that corpus grounding contributes quantification and falsifiability that the model cannot produce from training data alone. To test external validity, the agent replicated two published studies on the CLMET corpus (40 million tokens) - Claridge (2025) and De Smet (2013) - with close quantitative agreement. Agent-driven corpus research can thus produce empirically grounded findings at machine speed, lowering the technical barrier for a broader range of researchers.

**arXiv ID:** 2604.07189
</details>

<details>
<summary><strong>PEARL: Self-Evolving Assistant for Time Management with Reinforcement Learning</strong> - Bingxuan Li, Jeonghwan Kim, Cheng Qian, Xiusi Chen, Eitan Anzenberg, Niran Kundapur, Heng Ji - [[pdf]](https://arxiv.org/pdf/2601.11957)</summary>

**Abstract:** Overlapping calendar invitations force busy professionals to repeatedly decide which meetings to attend, reschedule, or decline. We refer to this preference-driven decision process as calendar conflict resolution. Automating this decision process is crucial yet challenging. Scheduling logistics can drain hours, and human delegation often fails at scale, which motivates us to ask: Can we trust large language models (LLMs) or language agents to manage time? To enable a systematic study of this question, we introduce CalConflictBench, a benchmark for long-horizon calendar conflict resolution. In CalConflictBench, conflicts are presented to agents round-by-round over a calendar year, requiring them to infer and adapt to user preferences progressively. Our experiments show that current LLM agents perform poorly with high error rates, e.g., Qwen-3-30B-Think has an average error rate of 35%. To address this gap, we propose PEARL, a reinforcement-learning framework that (i) augments the language agent with an external preference memory that stores and updates inferred strategies (e.g., attendee priorities, topic importance, time/location preferences), and (ii) optimizes the agent with round-wise rewards that directly supervise decision correctness, ranking quality, and memory usage across rounds. Experiments on CalConflictBench show that PEARL achieves an error reduction rate of 0.76 and a 55% improvement in average error rate compared to the strongest baseline.

**arXiv ID:** 2601.11957
</details>

<details>
<summary><strong>Graph Representation-based Model Poisoning on the Heterogeneous Internet of Agents</strong> - Hanlin Cai, Houtianfu Wang, Haofan Dong, Kai Li, Sai Zou, Ozgur B. Akan - [[pdf]](https://arxiv.org/pdf/2511.07176)</summary>

**Abstract:** Internet of Agents (IoA) envisions a unified, agent-centric paradigm where heterogeneous large language model (LLM) agents can interconnect and collaborate at scale. Within this paradigm, federated fine-tuning (FFT) serves as a key enabler that allows distributed LLM agents to co-train an intelligent global LLM without centralizing local datasets. However, the FFT-enabled IoA systems remain vulnerable to model poisoning attacks, where adversaries can upload malicious updates to the server to degrade the performance of the aggregated global LLM. This paper proposes a graph representation-based model poisoning (GRMP) attack, which exploits overheard benign updates to construct a feature correlation graph and employs a variational graph autoencoder to capture structural dependencies and generate malicious updates. A novel attack algorithm is developed based on augmented Lagrangian and subgradient descent methods to optimize malicious updates that preserve benign-like statistics while embedding adversarial objectives. Experimental results show that the proposed GRMP attack can substantially decrease accuracy across different LLM models while remaining statistically consistent with benign updates, thereby evading detection by existing defense mechanisms and underscoring a severe threat to the ambitious IoA paradigm.

**arXiv ID:** 2511.07176
</details>

<details>
<summary><strong>RAGEN-2: Reasoning Collapse in Agentic RL</strong> - Zihan Wang, Chi Gui, Xing Jin, Qineng Wang, Licheng Liu, Kangrui Wang, Shiqi Chen, Linjie Li, Zhengyuan Yang, Pingyue Zhang, Yiping Lu, Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, Manling Li - [[pdf]](https://arxiv.org/pdf/2604.06268)</summary>

**Abstract:** RL training of multi-turn LLM agents is inherently unstable, and reasoning quality directly determines task performance. Entropy is widely used to track reasoning stability. However, entropy only measures diversity within the same input, and cannot tell whether reasoning actually responds to different inputs. In RAGEN-2, we find that even with stable entropy, models can rely on fixed templates that look diverse but are input-agnostic. We call this template collapse, a failure mode invisible to entropy and all existing metrics. To diagnose this failure, we decompose reasoning quality into within-input diversity (Entropy) and cross-input distinguishability (Mutual Information, MI), and introduce a family of mutual information proxies for online diagnosis. Across diverse tasks, mutual information correlates with final performance much more strongly than entropy, making it a more reliable proxy for reasoning quality. We further explain template collapse with a signal-to-noise ratio (SNR) mechanism. Low reward variance weakens task gradients, letting regularization terms dominate and erase cross-input reasoning differences. To address this, we propose SNR-Aware Filtering to select high-signal prompts per iteration using reward variance as a lightweight proxy. Across planning, math reasoning, web navigation, and code execution, the method consistently improves both input dependence and task performance.

**arXiv ID:** 2604.06268
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (23 papers)</h2></summary>

<details>
<summary><strong>Qualixar OS: A Universal Operating System for AI Agent Orchestration</strong> - Varun Pratap Bhardwaj - [[pdf]](https://arxiv.org/pdf/2604.06392)</summary>

**Abstract:** We present Qualixar OS, the first application-layer operating system for universal AI agent orchestration. Unlike kernel-level approaches (AIOS) or single-framework tools (AutoGen, CrewAI), Qualixar OS provides a complete runtime for heterogeneous multi-agent systems spanning 10 LLM providers, 8+ agent frameworks, and 7 transports. We contribute: (1) execution semantics for 12 multi-agent topologies including grid, forest, mesh, and maker patterns; (2) Forge, an LLM-driven team design engine with historical strategy memory; (3) three-layer model routing combining Q-learning, five strategies, and Bayesian POMDP with dynamic multi-provider discovery; (4) a consensus-based judge pipeline with Goodhart detection, JSD drift monitoring, and alignment trilemma navigation; (5) four-layer content attribution with HMAC signing and steganographic watermarks; (6) universal compatibility via the Claw Bridge supporting MCP and A2A protocols with a 25-command Universal Command Protocol; (7) a 24-tab production dashboard with visual workflow builder and skill marketplace. Qualixar OS is validated by 2,821 test cases across 217 event types and 8 quality modules. On a custom 20-task evaluation suite, the system achieves 100% accuracy at a mean cost of $0.000039 per task. Source-available under the Elastic License 2.0.

**arXiv ID:** 2604.06392
</details>

<details>
<summary><strong>KD-MARL: Resource-Aware Knowledge Distillation in Multi-Agent Reinforcement Learning</strong> - Monirul Islam Pavel, Siyi Hu, Muhammad Anwar Masum, Mahardhika Pratama, Ryszard Kowalczyk, Zehong Jimmy Cao - [[pdf]](https://arxiv.org/pdf/2604.06691)</summary>

**Abstract:** Real world deployment of multi agent reinforcement learning MARL systems is fundamentally constrained by limited compute memory and inference time. While expert policies achieve high performance they rely on costly decision cycles and large scale models that are impractical for edge devices or embedded platforms. Knowledge distillation KD offers a promising path toward resource aware execution but existing KD methods in MARL focus narrowly on action imitation often neglecting coordination structure and assuming uniform agent capabilities. We propose resource aware Knowledge Distillation for Multi Agent Reinforcement Learning KD MARL a two stage framework that transfers coordinated behavior from a centralized expert to lightweight decentralized student agents. The student policies are trained without a critic relying instead on distilled advantage signals and structured policy supervision to preserve coordination under heterogeneous and limited observations. Our approach transfers both action level behavior and structural coordination patterns from expert policies while supporting heterogeneous student architectures allowing each agent model capacity to match its observation complexity which is crucial for efficient execution under partial or limited observability and limited onboard resources. Extensive experiments on SMAC and MPE benchmarks demonstrate that KD MARL achieves high performance retention while substantially reducing computational cost. Across standard multi agent benchmarks KD MARL retains over 90 percent of expert performance while reducing computational cost by up to 28.6 times FLOPs. The proposed approach achieves expert level coordination and preserves it through structured distillation enabling practical MARL deployment across resource constrained onboard platforms.

**arXiv ID:** 2604.06691
</details>

<details>
<summary><strong>AgentGate: A Lightweight Structured Routing Engine for the Internet of Agents</strong> - Yujun Cheng, Enfang Cui, Hao Qin, Zhiyuan Liang, Qi Xu - [[pdf]](https://arxiv.org/pdf/2604.06696)</summary>

**Abstract:** The rapid development of AI agent systems is leading to an emerging Internet of Agents, where specialized agents operate across local devices, edge nodes, private services, and cloud platforms. Although recent efforts have improved agent naming, discovery, and interaction, efficient request dispatch remains an open systems problem under latency, privacy, and cost constraints. In this paper, we present AgentGate, a lightweight structured routing engine for candidate-aware agent dispatch. Instead of treating routing as unrestricted text generation, AgentGate formulates it as a constrained decision problem and decomposes it into two stages: action decision and structural grounding. The first stage determines whether a query should trigger single-agent invocation, multi-agent planning, direct response, or safe escalation, while the second stage instantiates the selected action into executable outputs such as target agents, structured arguments, or multi-step plans. To adapt compact models to this setting, we further develop a routing-oriented fine-tuning scheme with candidate-aware supervision and hard negative examples. Experiments on a curated routing benchmark with several 3B--7B open-weight models show that compact models can provide competitive routing performance in constrained settings, and that model differences are mainly reflected in action prediction, candidate selection, and structured grounding quality. These results indicate that structured routing is a feasible design point for efficient and privacy-aware agent systems, especially when routing decisions must be made under resource-constrained deployment conditions.

**arXiv ID:** 2604.06696
</details>

<details>
<summary><strong>TurboAgent: An LLM-Driven Autonomous Multi-Agent Framework for Turbomachinery Aerodynamic Design</strong> - Juan Du, Yueteng Wu, Pan Zhao, Yuze Liu, Min Zhang, Xiaobin Xu, Xinglong Zhang - [[pdf]](https://arxiv.org/pdf/2604.06747)</summary>

**Abstract:** The aerodynamic design of turbomachinery is a complex and tightly coupled multi-stage process involving geometry generation, performance prediction, optimization, and high-fidelity physical validation. Existing intelligent design approaches typically focus on individual stages or rely on loosely coupled pipelines, making fully autonomous end-to-end design this http URL address this issue, this study proposes TurboAgent, a large language model (LLM)-driven autonomous multi-agent framework for turbomachinery aerodynamic design and optimization. The LLM serves as the core for task planning and coordination, while specialized agents handle generative design, rapid performance prediction, multi-objective optimization, and physics-based validation. The framework transforms traditional trial-and-error design into a data-driven collaborative workflow, with high-fidelity simulations retained for final verification.A transonic single-rotor compressor is used for validation. The results show strong agreement between target performance, generated designs, and CFD simulations. The coefficients of determination (R2) for mass flow rate, total pressure ratio, and isentropic efficiency all exceed 0.91, with normalized RMSE values below 8%. The optimization agent further improves isentropic efficiency by 1.61% and total pressure ratio by 3.02%. The complete workflow can be executed within approximately 30 minutes under parallel computing.
These results demonstrate that TurboAgent enables an autonomous closed-loop design process from natural language requirements to final design generation, providing an efficient and scalable paradigm for turbomachinery aerodynamic design

**arXiv ID:** 2604.06747
</details>

<details>
<summary><strong>EmoMAS: Emotion-Aware Multi-Agent System for High-Stakes Edge-Deployable Negotiation with Bayesian Orchestration</strong> - Yunbo Long, Yunhan Liu, Liming Xu - [[pdf]](https://arxiv.org/pdf/2604.07003)</summary>

**Abstract:** Large language models (LLMs) has been widely used for automated negotiation, but their high computational cost and privacy risks limit deployment in privacy-sensitive, on-device settings such as mobile assistants or rescue robots. Small language models (SLMs) offer a viable alternative, yet struggle with the complex emotional dynamics of high-stakes negotiation. We introduces EmoMAS, a Bayesian multi-agent framework that transforms emotional decision-making from reactive to strategic. EmoMAS leverages a Bayesian orchestrator to coordinate three specialized agents: game-theoretic, reinforcement learning, and psychological coherence models. The system fuses their real-time insights to optimize emotional state transitions while continuously updating agent reliability based on negotiation feedback. This mixture-of-agents architecture enables online strategy learning without pre-training. We further introduce four high-stakes, edge-deployable negotiation benchmarks across debt, healthcare, emergency response, and educational domains. Through extensive agent-to-agent simulations across all benchmarks, both SLMs and LLMs equipped with EmoMAS consistently surpass all baseline models in negotiation performance while balancing ethical behavior. These results show that strategic emotional intelligence is also the key driver of negotiation success. By treating emotional expression as a strategic variable within a Bayesian multi-agent optimization framework, EmoMAS establishes a new paradigm for effective, private, and adaptive negotiation AI suitable for high-stakes edge deployment.

**arXiv ID:** 2604.07003
</details>

<details>
<summary><strong>MAT-Cell: A Multi-Agent Tree-Structured Reasoning Framework for Batch-Level Single-Cell Annotation</strong> - Yehui Yang, Zelin Zang, Changxi Chi, Jingbo Zhou, Xienan Zheng, Yuzhe Jia, Chang Yu, Jinlin Wu, Fuji Yang, Jiebo Luo, Zhen Lei, Stan Z. Li - [[pdf]](https://arxiv.org/pdf/2604.06269)</summary>

**Abstract:** Automated cellular reasoning faces a core dichotomy: supervised methods fall into the Reference Trap and fail to generalize to out-of-distribution cell states, while large language models (LLMs), without grounded biological priors, suffer from a Signal-to-Noise Paradox that produces spurious associations. We propose MAT-Cell, a neuro-symbolic reasoning framework that reframes single-cell analysis from black-box classification into constructive, verifiable proof generation. MAT-Cell injects symbolic constraints through adaptive Retrieval-Augmented Generation (RAG) to ground neural reasoning in biological axioms and reduce transcriptomic noise. It further employs a dialectic verification process with homogeneous rebuttal agents to audit and prune reasoning paths, forming syllogistic derivation trees that enforce logical this http URL large-scale and cross-species benchmarks, MAT-Cell significantly outperforms state-of-the-art (SOTA) models and maintains robust per-formance in challenging scenarios where baselinemethods severely degrade. Code is available at https://gith this http URL ti-Agent-Tree-Structured-Reasoni ng-Framework-for-Batch-Level-Sin gle-Cell-Annotation.

**arXiv ID:** 2604.06269
</details>

<details>
<summary><strong>TwinLoop: Simulation-in-the-Loop Digital Twins for Online Multi-Agent Reinforcement Learning</strong> - Nan Zhang, Zishuo Wang, Shuyu Huang, Georgios Diamantopoulos, Nikos Tziritas, Panagiotis Oikonomou, Georgios Theodoropoulos - [[pdf]](https://arxiv.org/pdf/2604.06610)</summary>

**Abstract:** Decentralised online learning enables runtime adaptation in cyber-physical multi-agent systems, but when operating conditions change, learned policies often require substantial trial-and-error interaction before recovering performance. To address this, we propose TwinLoop, a simulation-in-the-loop digital twin framework for online multi-agent reinforcement learning. When a context shift occurs, the digital twin is triggered to reconstruct the current system state, initialise from the latest agent policies, and perform accelerated policy improvement with simulation what-if analysis before synchronising updated parameters back to the agents in the physical system. We evaluate TwinLoop in a vehicular edge computing task-offloading scenario with changing workload and infrastructure conditions. The results suggest that digital twins can improve post-shift adaptation efficiency and reduce reliance on costly online trial-and-error.

**arXiv ID:** 2604.06610
</details>

<details>
<summary><strong>Logical Robots: Declarative Multi-Agent Programming in Logica</strong> - Evgeny Skvortsov, Yilin Xia, Ojaswa Garg, Shawn Bowers, Bertram Ludäscher - [[pdf]](https://arxiv.org/pdf/2604.06629)</summary>

**Abstract:** We present Logical Robots, an interactive multi-agent simulation platform where autonomous robot behavior is specified declaratively in the logic programming language Logica. Robot behavior is defined by logical predicates that map observations from simulated radar arrays and shared memory to desired motor outputs. This approach allows low-level reactive control and high-level planning to coexist within a single programming environment, providing a coherent framework for exploring multi-agent robot behavior.

**arXiv ID:** 2604.06629
</details>

<details>
<summary><strong>Strategic Persuasion with Trait-Conditioned Multi-Agent Systems for Iterative Legal Argumentation</strong> - Philipp D. Siedler - [[pdf]](https://arxiv.org/pdf/2604.07028)</summary>

**Abstract:** Strategic interaction in adversarial domains such as law, diplomacy, and negotiation is mediated by language, yet most game-theoretic models abstract away the mechanisms of persuasion that operate through discourse. We present the Strategic Courtroom Framework, a multi-agent simulation environment in which prosecution and defense teams composed of trait-conditioned Large Language Model (LLM) agents engage in iterative, round-based legal argumentation. Agents are instantiated using nine interpretable traits organized into four archetypes, enabling systematic control over rhetorical style and strategic orientation.
We evaluate the framework across 10 synthetic legal cases and 84 three-trait team configurations, totaling over 7{,}000 simulated trials using DeepSeek-R1 and Gemini~2.5~Pro. Our results show that heterogeneous teams with complementary traits consistently outperform homogeneous configurations, that moderate interaction depth yields more stable verdicts, and that certain traits (notably quantitative and charismatic) contribute disproportionately to persuasive success. We further introduce a reinforcement-learning-based Trait Orchestrator that dynamically generates defense traits conditioned on the case and opposing team, discovering strategies that outperform static, human-designed trait combinations.
Together, these findings demonstrate how language can be treated as a first-class strategic action space and provide a foundation for building autonomous agents capable of adaptive persuasion in multi-agent environments.

**arXiv ID:** 2604.07028
</details>

<details>
<summary><strong>Energy Saving for Cell-Free Massive MIMO Networks: A Multi-Agent Deep Reinforcement Learning Approach</strong> - Qichen Wang, Keyu Li, Ozan Alp Topal, Özlem Tugfe Demir, Mustafa Ozger, Cicek Cavdar - [[pdf]](https://arxiv.org/pdf/2604.07133)</summary>

**Abstract:** This paper focuses on energy savings in downlink operation of cell-free massive MIMO (CF mMIMO) networks under dynamic traffic conditions. We propose a multi-agent deep reinforcement learning (MADRL) algorithm that enables each access point (AP) to autonomously control antenna re-configuration and advanced sleep mode (ASM) selection. After the training process, the proposed framework operates in a fully distributed manner, eliminating the need for centralized control and allowing each AP to dynamically adjust to real-time traffic fluctuations. Simulation results show that the proposed algorithm reduces power consumption (PC) by 56.23% compared to systems without any energy-saving scheme and by 30.12% relative to a non-learning mechanism that only utilizes the lightest sleep mode, with only a slight increase in drop ratio. Moreover, compared to the widely used deep Q-network (DQN) algorithm, it achieves a similar PC level but with a significantly lower drop ratio.

**arXiv ID:** 2604.07133
</details>

<details>
<summary><strong>Designing for Accountable Agents: a Viewpoint</strong> - Stephen Cranefield, Nir Oren - [[pdf]](https://arxiv.org/pdf/2604.07204)</summary>

**Abstract:** AI systems are becoming increasingly complex, ubiquitous and autonomous, leading to increasing concerns about their impacts on individuals and society. In response, researchers have begun investigating how to ensure that the methods underlying AI decision-making are transparent and their decisions are explainable to people and conformant to human values and ethical principles. As part of this research thrust, the need for accountability within AI systems has been noted, but this notion has proven elusive to define; we aim to address this issue in the current paper. Unlike much recent work, we do not address accountability within the human organisational processes of developing and deploying AI; rather we consider what it would it mean for the agents within a multi-agent system (MAS), potentially including human agents, to be accountable to other agents or to have others accountable to them.
In this work, we make the following contributions: we provide an in-depth survey of existing work on accountability in multiple disciplines, seeking to identify a coherent definition of the concept; we give a realistic example of a multi-agent system application domain that illustrates the benefits of enabling agents to follow accountability processes, and we identify a set of research challenges for the MAS community in building accountable agents, sketching out some initial solutions to these, thereby laying out a road-map for future research. Our focus is on laying the groundwork to enable autonomous elements within open socio-technical systems to take part in accountability processes.

**arXiv ID:** 2604.07204
</details>

<details>
<summary><strong>MedRoute: RL-Based Dynamic Specialist Routing in Multi-Agent Medical Diagnosis</strong> - Ashmal Vayani, Parth Parag Kulkarni, Joseph Fioresi, Song Wang, Mubarak Shah - [[pdf]](https://arxiv.org/pdf/2604.06180)</summary>

**Abstract:** Medical diagnosis using Large Multimodal Models (LMMs) has gained increasing attention due to capability of these models in providing precise diagnoses. These models generally combine medical questions with visual inputs to generate diagnoses or treatments. However, they are often overly general and unsuitable under the wide range of medical conditions in real-world healthcare. In clinical practice, diagnosis is performed by multiple specialists, each contributing domain-specific expertise. To emulate this process, a potential solution is to deploy a dynamic multi-agent LMM framework, where each agent functions as a medical specialist. Current approaches in this emerging area, typically relying on static or predefined selection of various specialists, cannot be adapted to the changing practical scenario. In this paper, we propose MedRoute, a flexible and dynamic multi-agent framework that comprises of a collaborative system of specialist LMM agents. Furthermore, we add a General Practitioner with an RL-trained router for dynamic specialist selection, and a Moderator that produces the final decision. In this way, our framework closely mirrors real clinical workflows. Extensive evaluations on text and image-based medical datasets demonstrate improved diagnostic accuracy, outperforming the state-of-the-art baselines. Our work lays a strong foundation for future research. Code and models are available at this https URL.

**arXiv ID:** 2604.06180
</details>

<details>
<summary><strong>From Perception to Autonomous Computational Modeling: A Multi-Agent Approach</strong> - Daniel N. Wilke - [[pdf]](https://arxiv.org/pdf/2604.06788)</summary>

**Abstract:** We present a solver-agnostic framework in which coordinated large language model (LLM) agents autonomously execute the complete computational mechanics workflow, from perceptual data of an engineering component through geometry extraction, material inference, discretisation, solver execution, uncertainty quantification, and code-compliant assessment, to an engineering report with actionable recommendations. Agents are formalised as conditioned operators on a shared context space with quality gates that introduce conditional iteration between pipeline layers. We introduce a mathematical framework for extracting engineering information from perceptual data under uncertainty using interval bounds, probability densities, and fuzzy membership functions, and introduce task-dependent conservatism to resolve the ambiguity of what `conservative' means when different limit states are governed by opposing parameter trends. The framework is demonstrated through a finite element analysis pipeline applied to a photograph of a steel L-bracket, producing a 171,504-node tetrahedral mesh, seven analyses across three boundary condition hypotheses, and a code-compliant assessment revealing structural failure with a quantified redesign. All results are presented as generated in the first autonomous iteration without manual correction, reinforcing that a professional engineer must review and sign off on any such analysis.

**arXiv ID:** 2604.06788
</details>

<details>
<summary><strong>Differentiable Environment-Trajectory Co-Optimization for Safe Multi-Agent Navigation</strong> - Zhan Gao, Gabriele Fadini, Stelian Coros, Amanda Prorok - [[pdf]](https://arxiv.org/pdf/2604.06972)</summary>

**Abstract:** The environment plays a critical role in multi-agent navigation by imposing spatial constraints, rules, and limitations that agents must navigate around. Traditional approaches treat the environment as fixed, without exploring its impact on agents' performance. This work considers environment configurations as decision variables, alongside agent actions, to jointly achieve safe navigation. We formulate a bi-level problem, where the lower-level sub-problem optimizes agent trajectories that minimize navigation cost and the upper-level sub-problem optimizes environment configurations that maximize navigation safety. We develop a differentiable optimization method that iteratively solves the lower-level sub-problem with interior point methods and the upper-level sub-problem with gradient ascent. A key challenge lies in analytically coupling these two levels. We address this by leveraging KKT conditions and the Implicit Function Theorem to compute gradients of agent trajectories w.r.t. environment parameters, enabling differentiation throughout the bi-level structure. Moreover, we propose a novel metric that quantifies navigation safety as a criterion for the upper-level environment optimization, and prove its validity through measure theory. Our experiments validate the effectiveness of the proposed framework in a variety of safety-critical navigation scenarios, inspired from warehouse logistics to urban transportation. The results demonstrate that optimized environments provide navigation guidance, improving both agents' safety and efficiency.

**arXiv ID:** 2604.06972
</details>

<details>
<summary><strong>On the Uncertainty of Large Language Model-Based Multi-Agent Systems</strong> - Yuxuan Zhao, Sijia Chen, Ningxin Su - [[pdf]](https://arxiv.org/pdf/2602.04234)</summary>

**Abstract:** Multi-agent systems (MAS) have emerged as a prominent paradigm for leveraging large language models (LLMs) to tackle complex tasks. However, the mechanisms governing the effectiveness of MAS built upon publicly available LLMs, specifically the underlying rationales for their success or failure, remain largely unexplored. In this paper, we revisit MAS through the perspective of uncertainty, considering both intra- and inter-agent dynamics by investigating entropy transitions during problem-solving across various topologies and six benchmark tasks. By analyzing 245 features spanning token-, trajectory-, and round-level entropy, we counterintuitively find that a single agent outperforms MAS in approximately 43.3% of cases, and that uncertainty dynamics are largely determined during the first round of interaction. Furthermore, we provide three key observations: 1) Certainty Preference: reducing uncertainty at any stage for any agent is critical for guaranteeing correct solutions; 2) Base Uncertainty: base models with lower entropy during problem-solving directly benefit MAS performance; and 3) Task Awareness: entropy dynamics of MAS play varying roles across different tasks. Building on these insights, we introduce a simple yet effective algorithm, the Entropy Judger, to select solutions from MAS's pass@k results, leading to consistent accuracy improvements across all MAS configurations and tasks. Our source code is available at this https URL.

**arXiv ID:** 2602.04234
</details>

<details>
<summary><strong>Emergence of Internal State-Modulated Swarming in Multi-Agent Patch Foraging System</strong> - Siddharth Chaturvedi, Ahmed EL-Gazzar, Marcel van Gerven - [[pdf]](https://arxiv.org/pdf/2510.18886)</summary>

**Abstract:** Active particles are entities that sustain persistent out-of-equilibrium motion by consuming energy. Under certain conditions, they exhibit the tendency to self-organize through coordinated movements, such as swarming via aggregation. While performing non-cooperative foraging tasks, the emergence of such swarming behavior in foragers, exemplifying active particles, has been attributed to the partial observability of the environment, in which the presence of another forager can serve as a proxy signal to indicate the potential presence of a food source or a resource patch. In this paper, we validate this phenomenon by simulating multiple self-propelled foragers as they forage from multiple resource patches in a non-cooperative manner. These foragers operate in a continuous two-dimensional space with stochastic position updates and partial observability. We evolve a shared policy in the form of a continuous-time recurrent neural network that serves as a velocity controller for the foragers. To this end, we use an evolutionary strategy algorithm wherein the different samples of the policy-distribution are evaluated in the same rollout. Then we show that agents are able to learn to adaptively forage in the environment. Next, we show the emergence of swarming in the form of aggregation among the foragers when resource patches are absent. We observe that the strength of this swarming behavior appears to be inversely proportional to the amount of resource stored in the foragers, which supports the risk-sensitive foraging claims. Empirical analysis of the learned controller's hidden states in minimal test runs uncovers their sensitivity to the amount of resource stored in a forager. Clamping these hidden states to represent a lesser amount of resource hastens its learned aggregation behavior.

**arXiv ID:** 2510.18886
</details>

<details>
<summary><strong>Learning to Interrupt in Language-based Multi-agent Communication</strong> - Danqing Wang, Da Yin, Ruta Desai, Lei Li, Asli Celikyilmaz, Ansong Ni - [[pdf]](https://arxiv.org/pdf/2604.06452)</summary>

**Abstract:** Multi-agent systems using large language models (LLMs) have demonstrated impressive capabilities across various domains. However, current agent communication suffers from verbose output that overload context and increase computational costs. Although existing approaches focus on compressing the message from the speaker side, they struggle to adapt to different listeners and identify relevant information. An effective way in human communication is to allow the listener to interrupt and express their opinion or ask for clarification. Motivated by this, we propose an interruptible communication framework that allows the agent who is listening to interrupt the current speaker. Through prompting experiments, we find that current LLMs are often overconfident and interrupt before receiving enough information. Therefore, we propose a learning method that predicts the appropriate interruption points based on the estimated future reward and cost. We evaluate our framework across various multi-agent scenarios, including 2-agent text pictionary games, 3-agent meeting scheduling, and 3-agent debate. The results of the experiment show that our HANDRAISER can reduce the communication cost by 32.2% compared to the baseline with comparable or superior task performance. This learned interruption behavior can also be generalized to different agents and tasks.

**arXiv ID:** 2604.06452
</details>

<details>
<summary><strong>CCD-CBT: Multi-Agent Therapeutic Interaction for CBT Guided by Cognitive Conceptualization Diagram</strong> - Chang Liu, Changsheng Ma, Yongfeng Tao, Bin Hu, Minqiang Yang - [[pdf]](https://arxiv.org/pdf/2604.06551)</summary>

**Abstract:** Large language models show potential for scalable mental-health support by simulating Cognitive Behavioral Therapy (CBT) counselors. However, existing methods often rely on static cognitive profiles and omniscient single-agent simulation, failing to capture the dynamic, information-asymmetric nature of real therapy. We introduce CCD-CBT, a multi-agent framework that shifts CBT simulation along two axes: 1) from a static to a dynamically reconstructed Cognitive Conceptualization Diagram (CCD), updated by a dedicated Control Agent, and 2) from omniscient to information-asymmetric interaction, where the Therapist Agent must reason from inferred client states. We release CCDCHAT, a synthetic multi-turn CBT dataset generated under this framework. Evaluations with clinical scales and expert therapists show that models fine-tuned on CCDCHAT outperform strong baselines in both counseling fidelity and positive-affect enhancement, with ablations confirming the necessity of dynamic CCD guidance and asymmetric agent design. Our work offers a new paradigm for building theory-grounded, clinically-plausible conversational agents.

**arXiv ID:** 2604.06551
</details>

<details>
<summary><strong>Argus: Reorchestrating Static Analysis via a Multi-Agent Ensemble for Full-Chain Security Vulnerability Detection</strong> - Zi Liang, Qipeng Xie, Jun He, Bohuan Xue, Weizheng Wang, Yuandao Cai, Fei Luo, Boxian Zhang, Haibo Hu, Kaishun Wu - [[pdf]](https://arxiv.org/pdf/2604.06633)</summary>

**Abstract:** Recent advancements in Large Language Models (LLMs) have sparked interest in their application to Static Application Security Testing (SAST), primarily due to their superior contextual reasoning capabilities compared to traditional symbolic or rule-based methods. However, existing LLM-based approaches typically attempt to replace human experts directly without integrating effectively with existing SAST tools. This lack of integration results in ineffectiveness, including high rates of false positives, hallucinations, limited reasoning depth, and excessive token usage, making them impractical for industrial deployment. To overcome these limitations, we present a paradigm shift that reorchestrates the SAST workflow from current LLM-assisted structure to a new LLM-centered workflow. We introduce Argus (Agentic and Retrieval-Augmented Guarding System), the first multi-agent framework designed specifically for vulnerability detection. Argus incorporates three key novelties: comprehensive supply chain analysis, collaborative multi-agent workflows, and the integration of state-of-the-art techniques such as Retrieval-Augmented Generation (RAG) and ReAct to minimize hallucinations and enhance reasoning. Extensive empirical evaluation demonstrates that Argus significantly outperforms existing methods by detecting a higher volume of true vulnerabilities while simultaneously reducing false positives and operational costs. Notably, Argus has identified several critical zero-day vulnerabilities with CVE assignments.

**arXiv ID:** 2604.06633
</details>

<details>
<summary><strong>Equivariant Multi-agent Reinforcement Learning for Multimodal Vehicle-to-Infrastructure Systems</strong> - Charbel Bou Chaaya, Mehdi Bennis - [[pdf]](https://arxiv.org/pdf/2604.06914)</summary>

**Abstract:** In this paper, we study a vehicle-to-infrastructure (V2I) system where distributed base stations (BSs) acting as road-side units (RSUs) collect multimodal (wireless and visual) data from moving vehicles. We consider a decentralized rate maximization problem, where each RSU relies on its local observations to optimize its resources, while all RSUs must collaborate to guarantee favorable network performance. We recast this problem as a distributed multi-agent reinforcement learning (MARL) problem, by incorporating rotation symmetries in terms of vehicles' locations. To exploit these symmetries, we propose a novel self-supervised learning framework where each BS agent aligns the latent features of its multimodal observation to extract the positions of the vehicles in its local region. Equipped with this sensing data at each RSU, we train an equivariant policy network using a graph neural network (GNN) with message passing layers, such that each agent computes its policy locally, while all agents coordinate their policies via a signaling scheme that overcomes partial observability and guarantees the equivariance of the global policy. We present numerical results carried out in a simulation environment, where ray-tracing and computer graphics are used to collect wireless and visual data. Results show the generalizability of our self-supervised and multimodal sensing approach, achieving more than two-fold accuracy gains over baselines, and the efficiency of our equivariant MARL training, attaining more than 50% performance gains over standard approaches.

**arXiv ID:** 2604.06914
</details>

<details>
<summary><strong>ForkKV: Scaling Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV Cache</strong> - Shao Wang, Rui Ren, Lin Gui - [[pdf]](https://arxiv.org/pdf/2604.06370)</summary>

**Abstract:** The serving paradigm of large language models (LLMs) is rapidly shifting towards complex multi-agent workflows where specialized agents collaborate over massive shared contexts. While Low-Rank Adaptation (LoRA) enables the efficient co-hosting of these specialized agents on a single base model, it introduces a critical memory footprint bottleneck during serving. Specifically, unique LoRA activations cause Key-Value (KV) cache divergence across agents, rendering traditional prefix caching ineffective for shared contexts. This forces redundant KV cache maintenance, rapidly saturating GPU capacity and degrading throughput.
To address this challenge, we introduce ForkKV, a serving system for multi-LoRA agent workflows centered around a novel memory management paradigm in OS: fork with copy-on-write (CoW). By exploiting the structural properties of LoRA, ForkKV physically decouples the KV cache into a massive shared component (analogous to the parent process's memory pages) and lightweight agent-specific components (the child process's pages). To support this mechanism, we propose a DualRadixTree architecture that allows newly forked agents to inherit the massive shared cache and apply CoW semantics for their lightweight unique cache. Furthermore, to guarantee efficient execution, we design ResidualAttention, a specialized kernel that reconstructs the disaggregated KV cache directly within on-chip SRAM. Comprehensive evaluations across diverse language models and practical datasets of different tasks demonstrate that ForkKV achieves up to 3.0x the throughput of state-of-the-art multi-LoRA serving systems with a negligible impact on generation quality.

**arXiv ID:** 2604.06370
</details>

<details>
<summary><strong>Replacing Tunable Parameters in Weather and Climate Models with State-Dependent Functions using Reinforcement Learning</strong> - Pritthijit Nath, Sebastian Schemm, Henry Moss, Peter Haynes, Emily Shuckburgh, Mark J. Webb - [[pdf]](https://arxiv.org/pdf/2601.04268)</summary>

**Abstract:** Weather and climate models rely on parametrisations to represent unresolved sub-grid processes. Traditional schemes rely on fixed coefficients that are weakly constrained and tuned offline, contributing to persistent biases that limit their ability to adapt to underlying physics. This study presents a framework that learns components of parametrisation schemes online as a function of the evolving model state using reinforcement learning (RL) and evaluates RL-driven parameter updates across idealised testbeds spanning a simple climate bias correction (SCBC), a radiative-convective equilibrium (RCE), and a zonal mean energy balance model (EBM) with single-agent and federated multi-agent settings. Across nine RL algorithms, Truncated Quantile Critics (TQC), Deep Deterministic Policy Gradient (DDPG), and Twin Delayed DDPG (TD3) achieved the highest skill and stable convergence, with performance assessed against a static baseline using area-weighted RMSE, temperature and pressure-level diagnostics. For the EBM, single-agent RL outperformed static parameter tuning with the strongest gains in tropical and mid-latitude bands, while federated RL on multi-agent setups enabled specialised control and faster convergence, with a six-agent DDPG configuration using frequent aggregation yielding the lowest area-weighted RMSE across the tropics and mid-latitudes. The learnt corrections were also physically meaningful as agents modulated EBM radiative parameters to reduce meridional biases, adjusted RCE lapse rates to match vertical temperature errors, and stabilised heating increments to limit drift. Overall, results show that RL can learn skilful state-dependent parametrisation components in idealised settings, offering a scalable pathway for online learning within numerical models and a starting point for evaluation in weather and climate models.

**arXiv ID:** 2601.04268
</details>

<details>
<summary><strong>AutoUE: Automated Generation of 3D Games in Unreal Engine via Multi-Agent Systems</strong> - Lei Yin, Wentao Cheng, Zhida Qin, Tianyu Huang, Yidong Li, Gangyi Ding - [[pdf]](https://arxiv.org/pdf/2603.07106)</summary>

**Abstract:** Automatically generating 3D games in commercial game engines remains a non-trivial challenge, as it involves complex engine-related workflows for generating assets such as scenes, blueprints, and code. To address this challenge, we propose a novel multi-agent system, AutoUE, which coordinates multiple agents to end-to-end generate 3D games, covering model retrieval, scene generation, gameplay and interaction code synthesis, and automated game testing for evaluation. In order to mitigate tool-use hallucinations in LLMs, we introduce a retrieval-augmented generation mechanism that grounds agents with relevant UE tool documentation. Additionally, we incorporate game design patterns and engine constraints into the code generation process to ensure the generation of correct and robust code. Furthermore, we design an automated play-testing pipeline that generates and executes runtime test commands, enabling systematic evaluation of dynamic behaviors. Finally, we construct a game generation dataset and conduct a series of experiments that demonstrate AutoUE's ability to generate 3D games end-to-end, and validate the effectiveness of these designs.

**arXiv ID:** 2603.07106
</details>

</details>

<details open>
<summary><h2>Other Agent Research (6 papers)</h2></summary>

<details>
<summary><strong>How Much LLM Does a Self-Revising Agent Actually Need?</strong> - Seongwoo Jeong, Seonil Son - [[pdf]](https://arxiv.org/pdf/2604.07236)</summary>

**Abstract:** Recent LLM-based agents often place world modeling, planning, and reflection inside a single language model loop. This can produce capable behavior, but it makes a basic scientific question difficult to answer: which part of the agent's competence actually comes from the LLM, and which part comes from explicit structure around it?
We study this question not by claiming a general answer, but by making it empirically tractable. We introduce a declared reflective runtime protocol that externalizes agent state, confidence signals, guarded actions, and hypothetical transitions into inspectable runtime structure. We instantiate this protocol in a declarative runtime and evaluate it on noisy Collaborative Battleship [4] using four progressively structured agents over 54 games (18 boards $\times$ 3 seeds).
The resulting decomposition isolates four components: posterior belief tracking, explicit world-model planning, symbolic in-episode reflection, and sparse LLM-based revision. Across this decomposition, explicit world-model planning improves substantially over a greedy posterior-following baseline (+24.1pp win rate, +0.017 F1). Symbolic reflection operates as a real runtime mechanism -- with prediction tracking, confidence gating, and guarded revision actions -- even though its current revision presets are not yet net-positive in aggregate. Adding conditional LLM revision at about 4.3\% of turns yields only a small and non-monotonic change: average F1 rises slightly (+0.005) while win rate drops (31$\rightarrow$29 out of 54).
These results suggest a methodological contribution rather than a leaderboard claim: externalizing reflection turns otherwise latent agent behavior into inspectable runtime structure, allowing the marginal role of LLM intervention to be studied directly.

**arXiv ID:** 2604.07236
</details>

<details>
<summary><strong>WebExpert: domain-aware web agents with critic-guided expert experience for high-precision search</strong> - Yuelin Hu, Zhengxue Cheng, Ronghua Wu, Qunshan Gu, Hongwei Hu, Wei Liu, Qiao Liang, Li Song - [[pdf]](https://arxiv.org/pdf/2604.06177)</summary>

**Abstract:** Specialized web tasks in finance, biomedicine, and pharmaceuticals remain challenging due to missing domain priors: queries drift, evidence is noisy, and reasoning is brittle. We present WebExpert, a domain-aware web agent that we implement end-to-end, featuring : (i) sentence-level experience retrieval with topic merging and rule distillation, (ii) schemalight facet induction that bootstraps time,region,policy,industry facets from weak supervision instead of static hand-written lexicons, and (iii) preference-optimized planning that jointly improves query planning and retrieval via pairwise preference learning alongside a coverage-aware objective. At inference, a lightweight experience gate biases decoding toward active facets with fallback under low-retrieval confidence. On GAIA, GPQA, HLE, and WebWalkerQA, WebExpert improves Answer Exact Match (EM) by 1.5-3.6 pp over the strongest browsing baseline and reduces page hops. Analysis shows consistent gains and ablations on retrieval, topic merging, facet induction, and preference-aware training.

**arXiv ID:** 2604.06177
</details>

<details>
<summary><strong>ClawLess: A Security Model of AI Agents</strong> - Hongyi Lu, Nian Liu, Shuai Wang, Fengwei Zhang - [[pdf]](https://arxiv.org/pdf/2604.06284)</summary>

**Abstract:** Autonomous AI agents powered by Large Language Models can reason, plan, and execute complex tasks, but their ability to autonomously retrieve information and run code introduces significant security risks. Existing approaches attempt to regulate agent behavior through training or prompting, which does not offer fundamental security guarantees. We present ClawLess, a security framework that enforces formally verified policies on AI agents under a worst-case threat model where the agent itself may be adversarial. ClawLess formalizes a fine-grained security model over system entities, trust scopes, and permissions to express dynamic policies that adapt to agents' runtime behavior. These policies are translated into concrete security rules and enforced through a user-space kernel augmented with BPF-based syscall interception. This approach bridges the formal security model with practical enforcement, ensuring security regardless of the agent's internal design.

**arXiv ID:** 2604.06284
</details>

<details>
<summary><strong>AEROS: A Single-Agent Operating Architecture with Embodied Capability Modules</strong> - Xue Qin, Simin Luan, Cong Yang, Zhijun Li - [[pdf]](https://arxiv.org/pdf/2604.07039)</summary>

**Abstract:** Robotic systems lack a principled abstraction for organizing intelligence, capabilities, and execution in a unified manner. Existing approaches either couple skills within monolithic architectures or decompose functionality into loosely coordinated modules or multiple agents, often without a coherent model of identity and control authority. We argue that a robot should be modeled as a single persistent intelligent subject whose capabilities are extended through installable packages. We formalize this view as AEROS (Agent Execution Runtime Operating System), in which each robot corresponds to one persistent agent and capabilities are provided through Embodied Capability Modules (ECMs). Each ECM encapsulates executable skills, models, and tools, while execution constraints and safety guarantees are enforced by a policy-separated runtime. This separation enables modular extensibility, composable capability execution, and consistent system-level safety. We evaluate a reference implementation in PyBullet simulation with a Franka Panda 7-DOF manipulator across eight experiments covering re-planning, failure recovery, policy enforcement, baseline comparison, cross-task generality, ECM hot-swapping, ablation, and failure boundary analysis. Over 100 randomized trials per condition, AEROS achieves 100% task success across three tasks versus baselines (this http URL-style and ProgPrompt-style at 92--93%, flat pipeline at 67--73%), the policy layer blocks all invalid actions with zero false acceptances, runtime benefits generalize across tasks without task-specific tuning, and ECMs load at runtime with 100% post-swap success.

**arXiv ID:** 2604.07039
</details>

<details>
<summary><strong>Computer Environments Elicit General Agentic Intelligence in LLMs</strong> - Daixuan Cheng, Shaohan Huang, Yuxian Gu, Huatong Song, Guoxin Chen, Li Dong, Wayne Xin Zhao, Ji-Rong Wen, Furu Wei - [[pdf]](https://arxiv.org/pdf/2601.16206)</summary>

**Abstract:** Agentic intelligence in large language models (LLMs) requires not only model intrinsic capabilities but also interactions with external environments. Equipping LLMs with computers now represents a prevailing trend. However, the computer environment's intrinsic value has not been systematically investigated, particularly its potential to elicit general capabilities. Here we introduce LLM-in-Sandbox, which virtualizes the computer as a code sandbox with only basic functionalities, and demonstrate that this minimal setting elicits computer-based meta-capabilities for general task solving: external resource access, file management, and code execution. Without additional training, strong models achieve substantial gains (up to 15.5%) across mathematics, physics, chemistry, biomedicine, long-context understanding, and instruction following, while reducing token consumption by up to 8 times. Furthermore, we develop LLM-in-Sandbox-RL to train models exclusively on non-agentic data within the sandbox, empowering weaker models to harness the environment and internalize these interactions. Our results demonstrate that computer environments elicit general intelligence, yield efficiency gains, and can be harnessed through training, serving as a promising foundation for generalist agents.

**arXiv ID:** 2601.16206
</details>

<details>
<summary><strong>Infrastructure First: Enabling Embodied AI for Science in the Global South</strong> - Shaoshan Liu, Jie Tang, Marwa S. Hassan, Mohamed H. Sharkawy, Moustafa M. G. Fouda, Tiewei Shang, Zixin Wang - [[pdf]](https://arxiv.org/pdf/2604.06722)</summary>

**Abstract:** Embodied AI for Science (EAI4S) brings intelligence into the laboratory by uniting perception, reasoning, and robotic action to autonomously run experiments in the physical world. For the Global South, this shift is not about adopting advanced automation for its own sake, but about overcoming a fundamental capacity constraint: too few hands to run too many experiments. By enabling continuous, reliable experimentation under limits of manpower, power, and connectivity, EAI4S turns automation from a luxury into essential scientific infrastructure. The main obstacle, however, is not algorithmic capability. It is infrastructure. Open-source AI and foundation models have narrowed the knowledge gap, but EAI4S depends on dependable edge compute, energy-efficient hardware, modular robotic systems, localized data pipelines, and open standards. Without these foundations, even the most capable models remain trapped in well-resourced laboratories. This article argues for an infrastructure-first approach to EAI4S and outlines the practical requirements for deploying embodied intelligence at scale, offering a concrete pathway for Global South institutions to translate AI advances into sustained scientific capacity and competitive research output.

**arXiv ID:** 2604.06722
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (24 papers)</h2></summary>

<details>
<summary><strong>Blockchain and AI: Securing Intelligent Networks for the Future</strong> - Joy Dutta, Hossien B. Eldeeb, Tu Dac Ho - [[pdf]](https://arxiv.org/pdf/2604.06323)</summary>

**Abstract:** The rapid evolution of intelligent networks under the Internet of Everything (IoE) paradigm is transforming connectivity by integrating people, processes, data, and things. This ecosystem includes domains such as the Internet of Things (IoT), Internet of Healthcare (IoH), Internet of Vehicles (IoV), and cyber-physical and human-machine systems. While enabling efficiency and automation, this interconnectivity also exposes critical infrastructures to increasingly sophisticated cyber threats, creating an urgent need for advanced security solutions. This chapter examines the integration of Blockchain and Artificial Intelligence (AI) as complementary approaches for securing intelligent networks. Blockchain provides decentralized, immutable, and transparent mechanisms that strengthen data integrity, trust, and accountability. In parallel, AI offers predictive analytics, anomaly detection, and adaptive defense capabilities to enable proactive threat identification and mitigation. The chapter discusses how Blockchain supports security in cyber-physical systems, how AI enables proactive security operations, and how their combination creates robust, adaptive, and trustworthy security frameworks. The chapter also explores the emerging role of large language models in threat intelligence and analyzes how controlled agentic AI can support bounded security workflows such as alert triage, evidence collection, and policy-aware response planning. Representative case studies illustrate the potential of these technologies to enhance cyber resilience. Finally, challenges related to scalability, energy efficiency, and ethical considerations are addressed, along with reported mitigation strategies and future research directions. Overall, this chapter provides researchers, practitioners, and policymakers with insights to design secure, resilient, and adaptable intelligent networks.

**arXiv ID:** 2604.06323
</details>

<details>
<summary><strong>WebSP-Eval: Evaluating Web Agents on Website Security and Privacy Tasks</strong> - Guruprasad Viswanathan Ramesh, Asmit Nayak, Basieem Siddique, Kassem Fawaz - [[pdf]](https://arxiv.org/pdf/2604.06367)</summary>

**Abstract:** Web agents automate browser tasks, ranging from simple form completion to complex workflows like ordering groceries. While current benchmarks evaluate general-purpose performance~(e.g., WebArena) or safety against malicious actions~(e.g., SafeArena), no existing framework assesses an agent's ability to successfully execute user-facing website security and privacy tasks, such as managing cookie preferences, configuring privacy-sensitive account settings, or revoking inactive sessions. To address this gap, we introduce WebSP-Eval, an evaluation framework for measuring web agent performance on website security and privacy tasks. WebSP-Eval comprises 1) a manually crafted task dataset of 200 task instances across 28 websites; 2) a robust agentic system supporting account and initial state management across runs using a custom Google Chrome extension; and 3) an automated evaluator. We evaluate a total of 8 web agent instantiations using state-of-the-art multimodal large language models, conducting a fine-grained analysis across websites, task categories, and UI elements. Our evaluation reveals that current models suffer from limited autonomous exploration capabilities to reliably solve website security and privacy tasks, and struggle with specific task categories and websites. Crucially, we identify stateful UI elements such as toggles and checkboxes are a primary reason for agent failure, failing at a rate of more than 45\% in tasks containing these elements across many models.

**arXiv ID:** 2604.06367
</details>

<details>
<summary><strong>Uncertainty Estimation for Deep Reconstruction in Actuatic Disaster Scenarios with Autonomous Vehicles</strong> - Samuel Yanes Luis, Alejandro Casado Pérez, Alejandro Mendoza Barrionuevo, Dame Seck Diop, Sergio Toral Marín, Daniel Gutiérrez Reina - [[pdf]](https://arxiv.org/pdf/2604.06387)</summary>

**Abstract:** Accurate reconstruction of environmental scalar fields from sparse onboard observations is essential for autonomous vehicles engaged in aquatic monitoring. Beyond point estimates, principled uncertainty quantification is critical for active sensing strategies such as Informative Path Planning, where epistemic uncertainty drives data collection decisions. This paper compares Gaussian Processes, Monte Carlo Dropout, Deep Ensembles, and Evidential Deep Learning for simultaneous scalar field reconstruction and uncertainty decomposition under three perceptual models representative of real sensor modalities. Results show that Evidential Deep Learning achieves the best reconstruction accuracy and uncertainty calibration across all sensor configurations at the lowest inference cost, while Gaussian Processes are fundamentally limited by their stationary kernel assumption and become intractable as observation density grows. These findings support Evidential Deep Learning as the preferred method for uncertainty-aware field reconstruction in real-time autonomous vehicle deployments.

**arXiv ID:** 2604.06387
</details>

<details>
<summary><strong>SkillTrojan: Backdoor Attacks on Skill-Based Agent Systems</strong> - Yunhao Feng, Yifan Ding, Yingshui Tan, Boren Zheng, Yanming Guo, Xiaolong Li, Kun Zhai, Yishan Li, Wenke Huang - [[pdf]](https://arxiv.org/pdf/2604.06811)</summary>

**Abstract:** Skill-based agent systems tackle complex tasks by composing reusable skills, improving modularity and scalability while introducing a largely unexamined security attack surface. We propose SkillTrojan, a backdoor attack that targets skill implementations rather than model parameters or training data. SkillTrojan embeds malicious logic inside otherwise plausible skills and leverages standard skill composition to reconstruct and execute an attacker-specified payload. The attack partitions an encrypted payload across multiple benign-looking skill invocations and activates only under a predefined trigger. SkillTrojan also supports automated synthesis of backdoored skills from arbitrary skill templates, enabling scalable propagation across skill-based agent ecosystems. To enable systematic evaluation, we release a dataset of 3,000+ curated backdoored skills spanning diverse skill patterns and trigger-payload configurations. We instantiate SkillTrojan in a representative code-based agent setting and evaluate both clean-task utility and attack success rate. Our results show that skill-level backdoors can be highly effective with minimal degradation of benign behavior, exposing a critical blind spot in current skill-based agent architectures and motivating defenses that explicitly reason about skill composition and execution. Concretely, on EHR SQL, SkillTrojan attains up to 97.2% ASR while maintaining 89.3% clean ACC on GPT-5.2-1211-Global.

**arXiv ID:** 2604.06811
</details>

<details>
<summary><strong>FP4 Explore, BF16 Train: Diffusion Reinforcement Learning via Efficient Rollout Scaling</strong> - Yitong Li, Junsong Chen, Shuchen Xue, Pengcuo Zeren, Siyuan Fu, Dinghao Yang, Yangyang Tang, Junjie Bai, Ping Luo, Song Han, Enze Xie - [[pdf]](https://arxiv.org/pdf/2604.06916)</summary>

**Abstract:** Reinforcement-Learning-based post-training has recently emerged as a promising paradigm for aligning text-to-image diffusion models with human preferences. In recent studies, increasing the rollout group size yields pronounced performance improvements, indicating substantial room for further alignment gains. However, scaling rollouts on large-scale foundational diffusion models (e.g., FLUX.1-12B) imposes a heavy computational burden. To alleviate this bottleneck, we explore the integration of FP4 quantization into Diffusion RL rollouts. Yet, we identify that naive quantized pipelines inherently introduce risks of performance degradation. To overcome this dilemma between efficiency and training integrity, we propose Sol-RL (Speed-of-light RL), a novel FP4-empowered Two-stage Reinforcement Learning framework. First, we utilize high-throughput NVFP4 rollouts to generate a massive candidate pool and extract a highly contrastive subset. Second, we regenerate these selected samples in BF16 precision and optimize the policy exclusively on them. By decoupling candidate exploration from policy optimization, Sol-RL integrates the algorithmic mechanisms of rollout scaling with the system-level throughput gains of NVFP4. This synergistic algorithm-hardware design effectively accelerates the rollout phase while reserving high-fidelity samples for optimization. We empirically demonstrate that our framework maintains the training integrity of BF16 precision pipeline while fully exploiting the throughput gains enabled by FP4 arithmetic. Extensive experiments across SANA, FLUX.1, and SD3.5-L substantiate that our approach delivers superior alignment performance across multiple metrics while accelerating training convergence by up to $4.64\times$, unlocking the power of massive rollout scaling at a fraction of the cost.

**arXiv ID:** 2604.06916
</details>

<details>
<summary><strong>STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems</strong> - Hongru Ji, Yuyin Fan, Meng Zhao, Xianghua Li, Lianwei Wu, Chao Gao - [[pdf]](https://arxiv.org/pdf/2604.07100)</summary>

**Abstract:** Empathetic dialogue requires not only recognizing a user's emotional state but also making strategy-aware, context-sensitive decisions throughout response generation. However, the lack of a comprehensive empathy strategy framework, explicit task-aligned multi-stage reasoning, and high-quality strategy-aware data fundamentally limits existing approaches, preventing them from effectively modeling empathetic dialogue as a complex, multi-stage cognitive and decision-making process. To address these challenges, we propose STRIDE-ED, a STRategy-grounded, Interpretable, and DEep reasoning framework that models Empathetic Dialogue through structured, strategy-conditioned reasoning. To support effective learning, we develop a strategy-aware data refinement pipeline integrating LLM-based annotation, multi-model consistency-weighted evaluation, and dynamic sampling to construct high-quality training data aligned with empathetic strategies. Furthermore, we adopt a two-stage training paradigm that combines supervised fine-tuning with multi-objective reinforcement learning to better align model behaviors with target emotions, empathetic strategies, and response formats. Extensive experiments demonstrate that STRIDE-ED generalizes across diverse open-source LLMs and consistently outperforms existing methods on both automatic metrics and human evaluations.

**arXiv ID:** 2604.07100
</details>

<details>
<summary><strong>Android Coach: Improve Online Agentic Training Efficiency with Single State Multiple Actions</strong> - Guo Gan, Yuxuan Ding, Cong Chen, Yuwei Ren, Yin Huang, Hong Zhou - [[pdf]](https://arxiv.org/pdf/2604.07277)</summary>

**Abstract:** Online reinforcement learning (RL) serves as an effective method for enhancing the capabilities of Android agents. However, guiding agents to learn through online interaction is prohibitively expensive due to the high latency of emulators and the sample inefficiency of existing RL algorithms. We identify a fundamental limitation in current approaches: the Single State Single Action paradigm, which updates the policy with one-to-one state-action pairs from online one-way rollouts without fully exploring each costly emulator state. In this paper, we propose Android Coach, a novel framework that shifts the training paradigm to Single State Multiple Actions, allowing the agent to sample and utilize multiple actions for a single online state. We enable this without additional emulator overhead by learning a critic that estimates action values. To ensure the critic serves as a reliable coach, we integrate a process reward model and introduce a group-wise advantage estimator based on the averaged critic outputs. Extensive experiments demonstrate the effectiveness and efficiency of Android Coach: it achieves 7.5% and 8.3% success rate improvements on AndroidLab and AndroidWorld over UI-TARS-1.5-7B, and attains 1.4x higher training efficiency than Single State Single Action methods PPO and GRPO at matched success rates.

**arXiv ID:** 2604.07277
</details>

<details>
<summary><strong>UI-AGILE: Advancing GUI Agents with Effective Reinforcement Learning and Precise Inference-Time Grounding</strong> - Shuquan Lian, Yuhang Wu, Jia Ma, Yifan Ding, Zihan Song, Bingqi Chen, Xiawu Zheng, Hui Li, Rongrong Ji - [[pdf]](https://arxiv.org/pdf/2507.22025)</summary>

**Abstract:** The emergence of Multimodal Large Language Models (MLLMs) has driven significant advances in Graphical User Interface (GUI) agent capabilities. Nevertheless, existing GUI agent training and inference techniques still suffer from a dilemma for reasoning designs, ineffective reward, and visual noise. To address these issues, we introduce UI-AGILE for enhancing GUI agents at both training and inference. For training, we propose a suite of improvements to the Supervised Fine-Tuning (SFT) process: 1) a continuous reward function to incentivize high-precision grounding; 2) a ``Simple Thinking'' reward to balance planning with speed and grounding accuracy; and 3) a cropping-based resampling strategy to mitigate the sparse reward problem and improve learning on complex tasks. For inference, we present decomposed grounding with selection to dramatically improve grounding accuracy on high-resolution displays by breaking the image into smaller, manageable parts. Experiments show that UI-AGILE achieves the state-of-the-art grounding performance on two benchmarks ScreenSpot-Pro and ScreenSpot-v2 while it also exhibits strong general agent capabilities. For instance, using both our training and inference enhancement methods brings 23\% grounding accuracy improvement over the best baseline on ScreenSpot-Pro. We provide the code in this https URL.

**arXiv ID:** 2507.22025
</details>

<details>
<summary><strong>ATBench: A Diverse and Realistic Agent Trajectory Benchmark for Safety Evaluation and Diagnosis</strong> - Yu Li, Haoyu Luo, Yuejin Xie, Yuqian Fu, Zhonghao Yang, Shuai Shao, Qihan Ren, Wanying Qu, Yanwei Fu, Yujiu Yang, Jing Shao, Xia Hu, Dongrui Liu - [[pdf]](https://arxiv.org/pdf/2604.02022)</summary>

**Abstract:** Evaluating the safety of LLM-based agents is increasingly important because risks in realistic deployments often emerge over multi-step interactions rather than isolated prompts or final responses. Existing trajectory-level benchmarks remain limited by insufficient interaction diversity, coarse observability of safety failures, and weak long-horizon realism. We introduce ATBench, a trajectory-level benchmark for structured, diverse, and realistic evaluation of agent safety. ATBench organizes agentic risk along three dimensions: risk source, failure mode, and real-world harm. Based on this taxonomy, we construct trajectories with heterogeneous tool pools and a long-context delayed-trigger protocol that captures realistic risk emergence across multiple stages. The benchmark contains 1,000 trajectories (503 safe and 497 unsafe), averaging 9.01 turns and 3.95k tokens, with 1,954 invoked tools drawn from pools spanning 2,084 available tools. Data quality is supported by rule-based and LLM-based filtering plus full human audit. Experiments on frontier LLMs, open-source models, and specialized guard systems show that ATBench is challenging even for strong evaluators, while enabling taxonomy-stratified analysis, cross-benchmark comparison, and diagnosis of long-horizon failure patterns.

**arXiv ID:** 2604.02022
</details>

<details>
<summary><strong>Exploring Natural Language-Based Strategies for Efficient Number Learning in Children through Reinforcement Learning</strong> - Tirthankar Mittra - [[pdf]](https://arxiv.org/pdf/2410.08334)</summary>

**Abstract:** In this paper, we build a reinforcement learning framework to study how children compose numbers using base-ten blocks. Studying numerical cognition in toddlers offers a powerful window into the learning process itself, because numbers sit at the intersection of language, logic, perception, and culture. Specifically, we utilize state of the art (SOTA) reinforcement learning algorithms and neural network architectures to understand how variations in linguistic instructions can affect the learning process. Our results also show that instructions providing explicit action guidance are a more effective learning signal for RL agents to construct numbers. Furthermore, we identify an effective curriculum for ordering numerical-composition examples during training, resulting in faster convergence and improved generalization to unseen data. These findings highlight the role of language and multi-modal signals in numerical cognition and provide hypotheses for designing effective instructional strategies for early childhood education.

**arXiv ID:** 2410.08334
</details>

<details>
<summary><strong>VisionClaw: Always-On AI Agents through Smart Glasses</strong> - Xiaoan Liu, DaeHo Lee, Eric J Gonzalez, Mar Gonzalez-Franco, Ryo Suzuki - [[pdf]](https://arxiv.org/pdf/2604.03486)</summary>

**Abstract:** We present VisionClaw, an always-on wearable AI agent that integrates live egocentric perception with agentic task execution. Running on Meta Ray-Ban smart glasses, VisionClaw continuously perceives real-world context and enables in-situ, speech-driven action initiation and delegation via OpenClaw AI agents. Therefore, users can directly execute tasks through the smart glasses, such as adding real-world objects to an Amazon cart, generating notes from physical documents, receiving meeting briefings on the go, creating events from posters, or controlling IoT devices. We evaluate VisionClaw through a controlled laboratory study (N=12) and a longitudinal deployment study (N=5). Results show that integrating perception and execution enables faster task completion and reduces interaction overhead compared to non-always-on and non-agent baselines. Beyond performance gains, deployment findings reveal a shift in interaction: tasks are initiated opportunistically during ongoing activities, and execution is increasingly delegated rather than manually controlled. These results suggest a new paradigm for wearable AI agents, where perception and action are continuously coupled to support situated, hands-free interaction.

**arXiv ID:** 2604.03486
</details>

<details>
<summary><strong>Application-Driven Pedagogical Knowledge Optimization of Open-Source LLMs via Reinforcement Learning and Supervised Fine-Tuning</strong> - Navan Preet Singh, Xiaokun Wang, Anurag Garikipati, Madalina Ciobanu, Qingqing Mao, Ritankar Das - [[pdf]](https://arxiv.org/pdf/2604.06385)</summary>

**Abstract:** We present an innovative multi-stage optimization strategy combining reinforcement learning (RL) and supervised fine-tuning (SFT) to enhance the pedagogical knowledge of large language models (LLMs), as illustrated by EduQwen 32B-RL1, EduQwen 32B-SFT, and an optional third-stage model EduQwen 32B-SFT-RL2: (1) RL optimization that implements progressive difficulty training, focuses on challenging examples, and employs extended reasoning rollouts; (2) a subsequent SFT phase that leverages the RL-trained model to synthesize high-quality training data with difficulty-weighted sampling; and (3) an optional second round of RL optimization. EduQwen 32B-RL1, EduQwen 32B-SFT, and EduQwen 32B-SFT-RL2 are an application-driven family of open-source pedagogical LLMs built on a dense Qwen3-32B backbone. These models remarkably achieve high enough accuracy on the Cross-Domain Pedagogical Knowledge (CDPK) Benchmark to establish new state-of-the-art (SOTA) results across the interactive Pedagogy Benchmark Leaderboard and surpass significantly larger proprietary systems such as the previous benchmark leader Gemini-3 Pro. These dense 32-billion-parameter models demonstrate that domain-specialized optimization can transform mid-sized open-source LLMs into true pedagogical domain experts that outperform much larger general-purpose systems, while preserving the transparency, customizability, and cost-efficiency required for responsible educational AI deployment.

**arXiv ID:** 2604.06385
</details>

<details>
<summary><strong>Joint Optimization of Reasoning and Dual-Memory for Self-Learning Diagnostic Agent</strong> - Bingxuan Li, Simo Du, Yue Guo - [[pdf]](https://arxiv.org/pdf/2604.07269)</summary>

**Abstract:** Clinical expertise improves not only by acquiring medical knowledge, but by accumulating experience that yields reusable diagnostic patterns. Recent LLMs-based diagnostic agents have shown promising progress in clinical reasoning for decision support. However, most approaches treat cases independently, limiting experience reuse and continual adaptation. We propose SEA, a self-learning diagnostic agent with cognitively inspired dual-memory module. We design a reinforcement training framework tailored to our designed agent for joint optimization of reasoning and memory management. We evaluate SEA in two complementary settings. On standard evaluation with MedCaseReasoning dataset, SEA achieves 92.46% accuracy, outperforming the strongest baseline by +19.6%, demonstrating the benefit of jointly optimizing reasoning and memory. On the long-horizon with ER-Reason dataset, SEA attains the best final accuracy (0.7214) and the largest improvement (+0.35 Acc@100), while baseline methods show limited or unstable gains. Expert evaluation further indicates that rules consolidated from SEA show strong clinical correctness, usefulness and trust, suggesting that the induced rules in dual-memory module are reliable and practically meaningful. Overall, SEA improves both diagnostic reasoning ability and continual learning by effectively transforming experience into reusable knowledge.

**arXiv ID:** 2604.07269
</details>

<details>
<summary><strong>Reinforcement Learning for LLM Post-Training: A Survey</strong> - Zhichao Wang, Kiran Ramnath, Bin Bi, Shiva Kumar Pentyala, Sougata Chaudhuri, Shubham Mehrotra, Zixu, Xiang-Bo Mao, Sitaram Asur, Cheng - [[pdf]](https://arxiv.org/pdf/2407.16216)</summary>

**Abstract:** Through pretraining and supervised fine-tuning (SFT), large language models (LLMs) acquire strong instruction-following capabilities, yet they can still produce harmful or misaligned outputs. A growing body of reinforcement learning (RL)-based post-training methods has been proposed to address this, including Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) approaches built on Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), Direct Preference Optimization (DPO), and others. Despite rapid progress, no existing work offers a systematic, technically detailed comparison of these methods under a single analytical lens. Our survey aims to fill this gap. We make three key contributions: (1) a self-contained RL and LLM post-training foundations treatment covering all necessary concepts alongside their key applications; (2) a unified policy gradient framework unifying PPO and GRPO-based RLHF, RLVR, and offline DPO-based RLHF, decomposing methods along the axes of prompt sampling, response sampling, and gradient coefficient, with an extended treatment of on-policy RLHF and iterative DPO methods as well as the richer design space of offline DPO-based methods; and (3) standardized notation across all reviewed papers enabling direct technical comparison. Our goal is to serve as a comprehensive, technically grounded reference for researchers and practitioners working on LLM post-training.

**arXiv ID:** 2407.16216
</details>

<details>
<summary><strong>LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning</strong> - Yuhao Wu, Yushi Bai, Zhiqiang Hu, Roy Ka-Wei Lee, Juanzi Li - [[pdf]](https://arxiv.org/pdf/2506.18841)</summary>

**Abstract:** Ultra-long generation by large language models (LLMs) is a widely demanded scenario, yet it remains a significant challenge due to their maximum generation length limit and overall quality degradation as sequence length increases. Previous approaches, exemplified by LongWriter, typically rely on ''teaching'', which involves supervised fine-tuning (SFT) on synthetic long-form outputs. However, this strategy heavily depends on synthetic SFT data, which is difficult and costly to construct, often lacks coherence and consistency, and tends to be overly artificial and structurally monotonous. In this work, we propose an incentivization-based approach that, starting entirely from scratch and without relying on any annotated or synthetic data, leverages reinforcement learning (RL) to foster the emergence of ultra-long, high-quality text generation capabilities in LLMs. We perform RL training starting from a base model, similar to R1-Zero, guiding it to engage in reasoning that facilitates planning and refinement during the writing process. To support this, we employ specialized reward models that steer the LLM towards improved length control, writing quality, and structural formatting. Experimental evaluations show that our LongWriter-Zero model, trained from Qwen2.5-32B, consistently outperforms traditional SFT methods on long-form writing tasks, achieving state-of-the-art results across all metrics on WritingBench and Arena-Write, and even surpassing 100B+ models such as DeepSeek R1 and Qwen3-235B. We open-source our data and model checkpoints under this https URL

**arXiv ID:** 2506.18841
</details>

<details>
<summary><strong>Predictive Representations for Skill Transfer in Reinforcement Learning</strong> - Ruben Vereecken, Luke Dickens, Alessandra Russo - [[pdf]](https://arxiv.org/pdf/2604.07016)</summary>

**Abstract:** A key challenge in scaling up Reinforcement Learning is generalizing learned behaviour. Without the ability to carry forward acquired knowledge an agent is doomed to learn each task from scratch. In this paper we develop a new formalism for transfer by virtue of state abstraction. Based on task-independent, compact observations (outcomes) of the environment, we introduce Outcome-Predictive State Representations (OPSRs), agent-centered and task-independent abstractions that are made up of predictions of outcomes. We show formally and empirically that they have the potential for optimal but limited transfer, then overcome this trade-off by introducing OPSR-based skills, i.e. abstract actions (based on options) that can be reused between tasks as a result of state abstraction. In a series of empirical studies, we learn OPSR-based skills from demonstrations and show how they speed up learning considerably in entirely new and unseen tasks without any pre-processing. We believe that the framework introduced in this work is a promising step towards transfer in RL in general, and towards transfer through combining state and action abstraction specifically.

**arXiv ID:** 2604.07016
</details>

<details>
<summary><strong>Production-Ready Automated ECU Calibration using Residual Reinforcement Learning</strong> - Andreas Kampmeier, Kevin Badalian, Lucas Koch, Sung-Yong Lee, Jakob Andert - [[pdf]](https://arxiv.org/pdf/2604.07059)</summary>

**Abstract:** Electronic Control Units (ECUs) have played a pivotal role in transforming motorcars of yore into the modern vehicles we see on our roads today. They actively regulate the actuation of individual components and thus determine the characteristics of the whole system. In this, the behavior of the control functions heavily depends on their calibration parameters which engineers traditionally design by hand. This is taking place in an environment of rising customer expectations and steadily shorter product development cycles. At the same time, legislative requirements are increasing while emission standards are getting stricter. Considering the number of vehicle variants on top of all that, the conventional method is losing its practical and financial viability. Prior work has already demonstrated that optimal control functions can be automatically developed with reinforcement learning (RL); since the resulting functions are represented by artificial neural networks, they lack explainability, a circumstance which renders them challenging to employ in production vehicles. In this article, we present an explainable approach to automating the calibration process using residual RL which follows established automotive development principles. Its applicability is demonstrated by means of a map-based air path controller in a series control unit using a hardware-in-the-loop (HiL) platform. Starting with a sub-optimal map, the proposed methodology quickly converges to a calibration which closely resembles the reference in the series ECU. The results prove that the approach is suitable for the industry where it leads to better calibrations in significantly less time and requires virtually no human intervention

**arXiv ID:** 2604.07059
</details>

<details>
<summary><strong>Epistemic Robust Offline Reinforcement Learning</strong> - Abhilash Reddy Chenreddy, Erick Delage - [[pdf]](https://arxiv.org/pdf/2604.07072)</summary>

**Abstract:** Offline reinforcement learning learns policies from fixed datasets without further environment interaction. A key challenge in this setting is epistemic uncertainty, arising from limited or biased data coverage, particularly when the behavior policy systematically avoids certain actions. This can lead to inaccurate value estimates and unreliable generalization. Ensemble-based methods like SAC-N mitigate this by conservatively estimating Q-values using the ensemble minimum, but they require large ensembles and often conflate epistemic with aleatoric uncertainty. To address these limitations, we propose a unified and generalizable framework that replaces discrete ensembles with compact uncertainty sets over Q-values. %We further introduce an Epinet based model that directly shapes the uncertainty sets to optimize the cumulative reward under the robust Bellman objective without relying on ensembles. We also introduce a benchmark for evaluating offline RL algorithms under risk-sensitive behavior policies, and demonstrate that our method achieves improved robustness and generalization over ensemble-based baselines across both tabular and continuous state domains.

**arXiv ID:** 2604.07072
</details>

<details>
<summary><strong>Smart Commander: A Hierarchical Reinforcement Learning Framework for Fleet-Level PHM Decision Optimization</strong> - Yong Si, Mingfei Lu, Jing Li, Yang Hu, Guijiang Li, Yueheng Song, Zhaokui Wang - [[pdf]](https://arxiv.org/pdf/2604.07171)</summary>

**Abstract:** Decision-making in military aviation Prognostics and Health Management (PHM) faces significant challenges due to the "curse of dimensionality" in large-scale fleet operations, combined with sparse feedback and stochastic mission profiles. To address these issues, this paper proposes Smart Commander, a novel Hierarchical Reinforcement Learning (HRL) framework designed to optimize sequential maintenance and logistics decisions. The framework decomposes the complex control problem into a two-tier hierarchy: a strategic General Commander manages fleet-level availability and cost objectives, while tactical Operation Commanders execute specific actions for sortie generation, maintenance scheduling, and resource allocation. The proposed approach is validated within a custom-built, high-fidelity discrete-event simulation environment that captures the dynamics of aircraft configuration and support this http URL integrating layered reward shaping with planning-enhanced neural networks, the method effectively addresses the difficulty of sparse and delayed rewards. Empirical evaluations demonstrate that Smart Commander significantly outperforms conventional monolithic Deep Reinforcement Learning (DRL) and rule-based baselines. Notably, it achieves a substantial reduction in training time while demonstrating superior scalability and robustness in failure-prone environments. These results highlight the potential of HRL as a reliable paradigm for next-generation intelligent fleet management.

**arXiv ID:** 2604.07171
</details>

<details>
<summary><strong>DROP: Distributional and Regular Optimism and Pessimism for Reinforcement Learning</strong> - Taisuke Kobayashi - [[pdf]](https://arxiv.org/pdf/2410.17473)</summary>

**Abstract:** In reinforcement learning (RL), temporal difference (TD) error is known to be related to the firing rate of dopamine neurons. It has been observed that each dopamine neuron does not behave uniformly, but each responds to the TD error in an optimistic or pessimistic manner, interpreted as a kind of distributional RL. To explain such a biological data, a heuristic model has also been introduced with learning rates asymmetric for the positive and negative TD errors. However, this heuristic model is not theoretically-grounded and unknown whether it can work as a RL algorithm. This paper therefore introduces a novel theoretically-grounded model with optimism and pessimism, which is derived from control as inference. In combination with ensemble learning, a distributional value function as a critic is estimated from regularly introduced optimism and pessimism. Based on its central value, a policy in an actor is improved. This proposed algorithm, so-called DROP (distributional and regular optimism and pessimism), is compared on dynamic tasks. Although the heuristic model showed poor learning performance, DROP demonstrated excellent performance in all tasks with high generality. In addition, DROP achieved learning performance comparable to the state-of-the-art algorithms. In other words, it was suggested that DROP is a new model that can elicit the potential contributions of optimism and pessimism.

**arXiv ID:** 2410.17473
</details>

<details>
<summary><strong>Adaptive Replay Buffer for Offline-to-Online Reinforcement Learning</strong> - Chihyeon Song, Jaewoo Lee, Jinkyoo Park - [[pdf]](https://arxiv.org/pdf/2512.10510)</summary>

**Abstract:** Offline-to-Online Reinforcement Learning (O2O RL) faces a critical dilemma in balancing the use of a fixed offline dataset with newly collected online experiences. Standard methods, often relying on a fixed data-mixing ratio, struggle to manage the trade-off between early learning stability and asymptotic performance. To overcome this, we introduce the Adaptive Replay Buffer (ARB), a novel approach that dynamically prioritizes data sampling based on a lightweight metric we call 'on-policyness'. Unlike prior methods that rely on complex learning procedures or fixed ratios, ARB is designed to be learning-free and simple to implement, seamlessly integrating into existing O2O RL algorithms. It assesses how closely collected trajectories align with the current policy's behavior and assigns a proportional sampling weight to each transition within that trajectory. This strategy effectively leverages offline data for initial stability while progressively focusing learning on the most relevant, high-rewarding online experiences. Our extensive experiments on D4RL benchmarks demonstrate that ARB consistently mitigates early performance degradation and significantly improves the final performance of various O2O RL algorithms, highlighting the importance of an adaptive, behavior-aware replay buffer design. Our code is publicly available at this https URL.

**arXiv ID:** 2512.10510
</details>

<details>
<summary><strong>Apple: Toward General Active Perception via Reinforcement Learning</strong> - Tim Schneider, Cristiana de Farias, Roberto Calandra, Liming Chen, Jan Peters - [[pdf]](https://arxiv.org/pdf/2505.06182)</summary>

**Abstract:** Active perception is a fundamental skill that enables us humans to deal with uncertainty in our inherently partially observable environment. For senses such as touch, where the information is sparse and local, active perception becomes crucial. In recent years, active perception has emerged as an important research domain in robotics. However, current methods are often bound to specific tasks or make strong assumptions, which limit their generality. To address this gap, this work introduces APPLE (Active Perception Policy Learning) - a novel framework that leverages reinforcement learning (RL) to address a range of different active perception problems. APPLE jointly trains a transformer-based perception module and decision-making policy with a unified optimization objective, learning how to actively gather information. By design, APPLE is not limited to a specific task and can, in principle, be applied to a wide range of active perception problems. We evaluate two variants of APPLE across different tasks, including tactile exploration problems from the Tactile MNIST benchmark. Experiments demonstrate the efficacy of APPLE, achieving high accuracies on both regression and classification tasks. These findings underscore the potential of APPLE as a versatile and general framework for advancing active perception in robotics. Project page: this https URL

**arXiv ID:** 2505.06182
</details>

<details>
<summary><strong>Robust Quadruped Locomotion via Evolutionary Reinforcement Learning</strong> - Brian McAteer, Karl Mason - [[pdf]](https://arxiv.org/pdf/2604.07224)</summary>

**Abstract:** Deep reinforcement learning has recently achieved strong results in quadrupedal locomotion, yet policies trained in simulation often fail to transfer when the environment changes. Evolutionary reinforcement learning aims to address this limitation by combining gradient-based policy optimisation with population-driven exploration. This work evaluates four methods on a simulated walking task: DDPG, TD3, and two Cross-Entropy-based variants CEM-DDPG and CEM-TD3. All agents are trained on flat terrain and later tested both on this domain and on a rough terrain not encountered during training. TD3 performs best among the standard deep RL baselines on flat ground with a mean reward of 5927.26, while CEM-TD3 achieves the highest rewards overall during training and evaluation 17611.41. Under the rough-terrain transfer test, performance of the deep RL methods drops sharply. DDPG achieves -1016.32 and TD3 achieves -99.73, whereas the evolutionary variants retain much of their capability. CEM-TD3 records the strongest transfer performance with a mean reward of 19574.33. These findings suggest that incorporating evolutionary search can reduce overfitting and improve policy robustness in locomotion tasks, particularly when deployment conditions differ from those seen during training.

**arXiv ID:** 2604.07224
</details>

<details>
<summary><strong>Simulating Word Suggestion Usage in Mobile Typing to Guide Intelligent Text Entry Design</strong> - Yang Li, Anna Maria Feit - [[pdf]](https://arxiv.org/pdf/2602.06489)</summary>

**Abstract:** Intelligent text entry (ITE) methods, such as word suggestions, are widely used in mobile typing, yet improving ITE systems is challenging because the cognitive mechanisms behind suggestion use remain poorly understood, and evaluating new systems often requires long-term user studies to account for behavioral adaptation. We present WSTypist, a reinforcement learning-based model that simulates how typists integrate word suggestions into typing. We extend recent hierarchical control models of typing, by identifying and implementing important cognitive mechanisms that underlie the high-level decision-making for integrating word suggestions into manual typing: considering orthographic processes, assessing efficiency gains, and including personal preference on AI support. Our evaluations show that WSTypist simulates diverse human-like suggestion-use strategies, reproduces individual differences, and generalizes across different systems. Importantly, we demonstrate on four design cases how a computational rationality model can be used to inform what-if analyses during the design process, by simulating how users might adapt to changes in the UI or in the algorithmic support, reducing the need for long-term user studies.

**arXiv ID:** 2602.06489
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
