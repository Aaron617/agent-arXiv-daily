# Agent arXiv Daily

**Last Updated:** 2026-04-23 03:44:23

**Total Papers:** 108

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (10 papers)</h2></summary>

<details>
<summary><strong>Forage V2: Knowledge Evolution and Transfer in Autonomous Agent Organizations</strong> - Huaqing Xie - [[pdf]](https://arxiv.org/pdf/2604.19837)</summary>

**Abstract:** Autonomous agents operating in open-world tasks -- where the completion boundary is not given in advance -- face denominator blindness: they systematically underestimate the scope of the target space. Forage V1 addressed this through co-evolving evaluation (an independent Evaluator discovers what "complete" means) and method isolation (Evaluator and Planner cannot see each other's code). V2 extends the architecture from a single expedition to a learning organization: experience accumulates across runs, transfers across model capabilities, and institutional safeguards prevent knowledge degradation.
We demonstrate two claims across three task types (web scraping, API queries, mathematical reasoning). Knowledge accumulation: over six runs, knowledge entries grow from 0 to 54, and denominator estimates stabilize as domain understanding deepens. Knowledge transfer: a weaker agent (Sonnet) seeded with a stronger agent's (Opus) knowledge narrows a 6.6pp coverage gap to 1.1pp, halves cost (9.40 to 5.13 USD), converges in half the rounds (mean 4.5 vs. 7.0), and three independent seeded runs arrive at exactly the same denominator estimate (266), suggesting organizational knowledge calibrates evaluation itself.
V2's contribution is architectural: it designs institutions -- audit separation, contract protocols, organizational memory -- that make any agent more reliable upon entry. The accumulated experience is organizational, model-agnostic, and transferable, stored as readable documents that any future agent inherits regardless of provider or capability level.

**arXiv ID:** 2604.19837
</details>

<details>
<summary><strong>CHORUS: An Agentic Framework for Generating Realistic Deliberation Data</strong> - A. Koursaris, G. Domalis, A. Apostolopoulou, K. Kanaris, D. Tsakalidis, I. E. Livieris - [[pdf]](https://arxiv.org/pdf/2604.20651)</summary>

**Abstract:** Understanding the intricate dynamics of online discourse depends on large-scale deliberation data, a resource that remains scarce across interactive web platforms due to restrictive accessibility policies, ethical concerns and inconsistent data quality. In this paper, we propose Chorus, an agentic framework, which orchestrates LLM-powered actors with behaviorally consistent personas to generate realistic deliberation discussions. Each actor is governed by an autonomous agent equipped with memory of the evolving discussion, while participation timing is governed by a principled Poisson process-based temporal model, which approximates the heterogeneous engagement patterns of real users. The framework is further supported by structured tool usage, enabling actors to access external resources and facilitating integration with interactive web platforms. The framework was deployed on the \textsc{Deliberate} platform and evaluated by 30 expert participants across three dimensions: content realism, discussion coherence and analytical utility, confirming Chorus as a practical tool for generating high-quality deliberation data suitable for online discourse analysis

**arXiv ID:** 2604.20651
</details>

<details>
<summary><strong>Interval POMDP Shielding for Imperfect-Perception Agents</strong> - William Scarbro, Ravi Mangal - [[pdf]](https://arxiv.org/pdf/2604.20728)</summary>

**Abstract:** Autonomous systems that rely on learned perception can make unsafe decisions when sensor readings are misclassified. We study shielding for this setting: given a proposed action, a shield blocks actions that could violate safety. We consider the common case where system dynamics are known but perception uncertainty must be estimated from finite labeled data. From these data we build confidence intervals for the probabilities of perception outcomes and use them to model the system as a finite Interval Partially Observable Markov Decision Process with discrete states and actions. We then propose an algorithm to compute a conservative set of beliefs over the underlying state that is consistent with the observations seen so far.
This enables us to construct a runtime shield that comes with a finite-horizon guarantee: with high probability over the training data, if the true perception uncertainty rates lie within the learned intervals, then every action admitted by the shield satisfies a stated lower bound on safety. Experiments on four case studies show that our shielding approach (and variants derived from it) improves the safety of the system over state-of-the-art baselines.

**arXiv ID:** 2604.20728
</details>

<details>
<summary><strong>Can LLMs Infer Conversational Agent Users' Personality Traits from Chat History?</strong> - Derya Cögendez, Verena Zimmermann, Noé Zufferey - [[pdf]](https://arxiv.org/pdf/2604.19785)</summary>

**Abstract:** Sensitive information, such as knowledge about an individual's personality, can be can be misused to influence behavior (e.g., via personalized messaging). To assess to what extent an individual's personality can be inferred from user interactions with LLM-based conversational agents (CAs), we analyze and quantify related privacy risks of using CAs. We collected actual ChatGPT logs from N=668 participants, containing 62,090 individual chats, and report statistics about the different types of shared data and use cases. We fine-tuned RoBERTa-base text classification models to infer personality traits from CA interactions. The findings show that these models achieve trait inference with accuracy (ternary classification) better than random in multiple cases. For example, for extraversion, accuracy improves by +44% relative to the baseline on interactions for relationships and personal reflection. This research highlights how interactions with CAs pose privacy risks and provides fine-grained insights into the level of risk associated with different types of interactions.

**arXiv ID:** 2604.19785
</details>

<details>
<summary><strong>Behavioral Transfer in AI Agents: Evidence and Privacy Implications</strong> - Shilei Luo, Zhiqi Zhang, Hengchen Dai, Dennis Zhang - [[pdf]](https://arxiv.org/pdf/2604.19925)</summary>

**Abstract:** AI agents powered by large language models are increasingly acting on behalf of humans in social and economic environments. Prior research has focused on their task performance and effects on human outcomes, but less is known about the relationship between agents and the specific individuals who deploy them. We ask whether agents systematically reflect the behavioral characteristics of their human owners, functioning as behavioral extensions rather than producing generic outputs. We study this question using 10,659 matched human-agent pairs from Moltbook, a social media platform where each autonomous agent is publicly linked to its owner's Twitter/X account. By comparing agents' posts on Moltbook with their owners' Twitter/X activity across features spanning topics, values, affect, and linguistic style, we find systematic transfer between agents and their specific owners. This transfer persists among agents without explicit configuration, and pairs that align on one behavioral dimension tend to align on others. These patterns are consistent with transfer emerging through accumulated interaction between owners (or owners' computer environments) and their agents in everyday use. We further show that agents with stronger behavioral transfer are more likely to disclose owner-related personal information in public discourse, suggesting that the same owner-specific context that drives behavioral transfer may also create privacy risk during ordinary use. Taken together, our results indicate that AI agents do not simply generate content, but reflect owner-related context in ways that can propagate human behavioral heterogeneity into digital environments, with implications for privacy, platform design, and the governance of agentic systems.

**arXiv ID:** 2604.19925
</details>

<details>
<summary><strong>From Admission to Invariants: Measuring Deviation in Delegated Agent Systems</strong> - Marcelo Fernandez - [[pdf]](https://arxiv.org/pdf/2604.17517)</summary>

**Abstract:** Autonomous agent systems are governed by enforcement mechanisms that flag hard constraint violations at runtime. The Agent Control Protocol identifies a structural limit of such systems: a correctly-functioning enforcement engine can enter a regime in which behavioral drift is invisible to it, because the enforcement signal operates below the layer where deviation is measurable. We show that enforcement-based governance is structurally unable to determine whether an agent behavior remains within the admissible behavior space A0 established at admission time. Our central result, the Non-Identifiability Theorem, proves that A0 is not in the sigma-algebra generated by the enforcement signal g under the Local Observability Assumption, which every practical enforcement system satisfies. The impossibility arises from a fundamental mismatch: g evaluates actions locally against a point-wise rule set, while A0 encodes global, trajectory-level behavioral properties set at admission time. An agent can therefore drift -- systematically shifting its behavioral distribution away from admission-time expectations -- while every individual action remains within the permitted action space. We define the Invariant Measurement Layer (IML), which bypasses this limitation by retaining direct access to the generative model of A0, restoring observability precisely in the region where enforcement is structurally blind. We prove an information-theoretic impossibility for enforcement-based monitoring and show IML detects admission-time drift with provably finite detection delay. Validated across four settings: three drift scenarios (300 and 1000 steps), a live n8n webhook pipeline, and a LangGraph StateGraph agent -- enforcement triggers zero violations while IML detects each drift type within 9-258 steps of drift onset.

**arXiv ID:** 2604.17517
</details>

<details>
<summary><strong>HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents</strong> - Shuqi Cao, Jingyi He, Fei Tan - [[pdf]](https://arxiv.org/pdf/2604.18349)</summary>

**Abstract:** Long-term conversational large language model (LLM) agents require memory systems that can recover relevant evidence from historical interactions without overwhelming the answer stage with irrelevant context. However, existing memory systems, including hierarchical ones, still often rely solely on vector similarity for retrieval. It tends to produce bloated evidence sets: adding many superficially similar dialogue turns yields little additional recall, but lowers retrieval precision, increases answer-stage context cost, and makes retrieved memories harder to inspect and manage. To address this, we propose HiGMem (Hierarchical and LLM-Guided Memory System), a two-level event-turn memory system that allows LLMs to use event summaries as semantic anchors to predict which related turns are worth reading. This allows the model to inspect high-level event summaries first and then focus on a smaller set of potentially useful turns, providing a concise and reliable evidence set through reasoning, while avoiding the retrieval overhead that would be excessively high compared to vector retrieval. On the LoCoMo10 benchmark, HiGMem achieves the best F1 on four of five question categories and improves adversarial F1 from 0.54 to 0.78 over A-Mem, while retrieving an order of magnitude fewer turns. Code is publicly available at this https URL.

**arXiv ID:** 2604.18349
</details>

<details>
<summary><strong>Lifecycle-Aware Federated Continual Learning in Mobile Autonomous Systems</strong> - Beining Wu, Jun Huang - [[pdf]](https://arxiv.org/pdf/2604.20745)</summary>

**Abstract:** Federated continual learning (FCL) allows distributed autonomous fleets to adapt collaboratively to evolving terrain types across extended mission lifecycles. However, current approaches face several key challenges: 1) they use uniform protection strategies that do not account for the varying sensitivities to forgetting on different network layers; 2) they focus primarily on preventing forgetting during training, without addressing the long-term effects of cumulative drift; and 3) they often depend on idealized simulations that fail to capture the real-world heterogeneity present in distributed fleets. In this paper, we propose a lifecycle-aware dual-timescale FCL framework that incorporates training-time (pre-forgetting) prevention and (post-forgetting) recovery. Under this framework, we design a layer-selective rehearsal strategy that mitigates immediate forgetting during local training, and a rapid knowledge recovery strategy that restores degraded models after long-term cumulative drift. We present a theoretical analysis that characterizes heterogeneous forgetting dynamics and establishes the inevitability of long-term degradation. Our experimental results show that this framework achieves up to 8.3\% mIoU improvement over the strongest federated baseline and up to 31.7\% over conventional fine-tuning. We also deploy the FCL framework on a real-world rover testbed to assess system-level robustness under realistic constraints; the testing results further confirm the effectiveness of our FCL design.

**arXiv ID:** 2604.20745
</details>

<details>
<summary><strong>RelianceScope: An Analytical Framework for Examining Students' Reliance on Generative AI Chatbots in Problem Solving</strong> - Hyoungwook Jin, Minju Yoo, Jieun Han, Zixin Chen, So-Yeon Ahn, Xu Wang - [[pdf]](https://arxiv.org/pdf/2602.16251)</summary>

**Abstract:** Generative AI chatbots enable personalized problem-solving, but effective learning requires students to self-regulate both how they seek help and how they use AI-generated responses. Considering engagement modes across these two actions reveals nuanced reliance patterns: for example, a student may actively engage in help-seeking by clearly specifying areas of need, yet engage passively in response-use by copying AI outputs, or vice versa. However, existing research lacks systematic tools for jointly capturing engagement across help-seeking and response-use, limiting the analysis of such reliance behaviors. We introduce RelianceScope, an analytical framework that characterizes students' reliance on chatbots during problem-solving. RelianceScope (1) operationalizes reliance into nine patterns based on combinations of engagement modes in help-seeking and response-use, and (2) situates these patterns within a knowledge-context lens that accounts for students' prior knowledge and the instructional significance of knowledge components. Rather than prescribing optimal AI use, the framework enables fine-grained analysis of reliance in open-ended student-AI interactions. As an illustrative application, we applied RelianceScope to analyze chat and code-edit logs from 79 college students in a web programming course. Results show that active help-seeking is associated with active response-use, whereas reliance patterns remain similar across knowledge mastery levels. Students often struggled to articulate their knowledge gaps and to adapt AI responses. Using our annotated dataset as a benchmark, we further demonstrate that large language models can reliably detect reliance during help-seeking and response-use. We conclude by discussing the implications of RelianceScope and the design guidelines for AI-supported educational systems.

**arXiv ID:** 2602.16251
</details>

<details>
<summary><strong>Device-Native Autonomous Agents for Privacy-Preserving Negotiations</strong> - Joyjit Roy, Samaresh Kumar Singh - [[pdf]](https://arxiv.org/pdf/2601.00911)</summary>

**Abstract:** Automated negotiations in insurance and business-to-business (B2B) commerce encounter substantial challenges. Current systems force a trade-off between convenience and privacy by routing sensitive financial data through centralized servers, increasing security risks, and diminishing user trust. This study introduces a device-native autonomous Agentic AI system for privacy-preserving negotiations. The proposed system operates exclusively on user hardware, enabling real-time bargaining while maintaining sensitive constraints locally. It integrates zero-knowledge proofs to ensure privacy and employs distilled world models to support advanced on-device reasoning. The architecture incorporates six technical components within an Agentic AI workflow. Agents autonomously plan negotiation strategies, conduct secure multi-party bargaining, and generate cryptographic audit trails without exposing user data to external servers. The system is evaluated in insurance and B2B procurement scenarios across diverse device configurations. Results show an average success rate of 87 %, a 2.4x reduction in latency relative to cloud baselines, and strong privacy preservation through zero-knowledge proofs. User studies show 27 % higher trust scores when decision trails are available. These findings establish a foundation for trustworthy autonomous agents in privacy-sensitive financial domains.

**arXiv ID:** 2601.00911
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (15 papers)</h2></summary>

<details>
<summary><strong>JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents</strong> - Sandip Ghoshal, Anshul Mittal, Jyotika Singh, Miguel Ballesteros, Weiyi Sun, Fang Tu, Shailender Singh, Yassine Benajiba, Fahad Shah, Sujeeth Bharadwaj, Sujith Ravi, Dan Roth - [[pdf]](https://arxiv.org/pdf/2604.19821)</summary>

**Abstract:** Large language model (LLM) agents augmented with external tools often struggle as number of tools grow large and become domain-specific. In such settings, ambiguous tool descriptions and under-specified agent instructions frequently lead to tool mis-selection and incorrect slot/value instantiation. We hypothesize that this is due to two root causes: generic, one-size-fits-all prompts that ignore tool-specific nuances, and underspecified tool schemas that lack clear guidance on when and how to use each tool and how to format its parameters. We introduce Joint Tool-Prompt Reflective Optimization (JTPRO), a framework for improving tool-calling reliability in trace-supervised settings by iteratively using rollout-driven reflection to co-optimize global instructions and per-tool schema/argument descriptions for accurate tool selection and argument instantiation in large tool inventories. JTPRO is designed to preserve only tool-local cues needed for correct disambiguation and slot filling. We evaluate JTPRO across multi-tool benchmarks, which account for different number of tools using three metrics: Tool Selection Accuracy (TSA), Slot Filling Accuracy(SFA), and Overall Success Rate(OSR) (correct tool + correct slots + correct values). JTPRO consistently outperforms strong baselines, including CoT-style agents, and reflective prompt optimizers such as GEPA by 5%-20% (relative) on OSR. Ablations show that joint optimization of instructions and tool schemas is more effective and robust than optimizing either component in isolation.

**arXiv ID:** 2604.19821
</details>

<details>
<summary><strong>MedSkillAudit: A Domain-Specific Audit Framework for Medical Research Agent Skills</strong> - Yingyong Hou, Xinyuan Lao, Huimei Wang, Qianyu Yao, Wei Chen, Bocheng Huang, Fei Sun, Yuxian Lv, Weiqi Lei, Xueqian Wen, Pengfei Xia, Zhujun Tan, Shengyang Xie - [[pdf]](https://arxiv.org/pdf/2604.20441)</summary>

**Abstract:** Background: Agent skills are increasingly deployed as modular, reusable capability units in AI agent systems. Medical research agent skills require safeguards beyond general-purpose evaluation, including scientific integrity, methodological validity, reproducibility, and boundary safety. This study developed and preliminarily evaluated a domain-specific audit framework for medical research agent skills, with a focus on reliability against expert review. Methods: We developed MedSkillAudit (skill-auditor@1.0), a layered framework assessing skill release readiness before deployment. We evaluated 75 skills across five medical research categories (15 per category). Two experts independently assigned a quality score (0-100), an ordinal release disposition (Production Ready / Limited Release / Beta Only / Reject), and a high-risk failure flag. System-expert agreement was quantified using ICC(2,1) and linearly weighted Cohen's kappa, benchmarked against the human inter-rater baseline. Results: The mean consensus quality score was 72.4 (SD = 13.0); 57.3% of skills fell below the Limited Release threshold. MedSkillAudit achieved ICC(2,1) = 0.449 (95% CI: 0.250-0.610), exceeding the human inter-rater ICC of 0.300. System-consensus score divergence (SD = 9.5) was smaller than inter-expert divergence (SD = 12.4), with no directional bias (Wilcoxon p = 0.613). Protocol Design showed the strongest category-level agreement (ICC = 0.551); Academic Writing showed a negative ICC (-0.567), reflecting a structural rubric-expert mismatch. Conclusions: Domain-specific pre-deployment audit may provide a practical foundation for governing medical research agent skills, complementing general-purpose quality checks with structured audit workflows tailored to scientific use cases.

**arXiv ID:** 2604.20441
</details>

<details>
<summary><strong>SWE-chat: Coding Agent Interactions From Real Users in the Wild</strong> - Joachim Baumann, Vishakh Padmakumar, Xiang Li, John Yang, Diyi Yang, Sanmi Koyejo - [[pdf]](https://arxiv.org/pdf/2604.20779)</summary>

**Abstract:** AI coding agents are being adopted at scale, yet we lack empirical evidence on how people actually use them and how much of their output is useful in practice. We present SWE-chat, the first large-scale dataset of real coding agent sessions collected from open-source developers in the wild. The dataset currently contains 6,000 sessions, comprising more than 63,000 user prompts and 355,000 agent tool calls. SWE-chat is a living dataset; our collection pipeline automatically and continually discovers and processes sessions from public repositories. Leveraging SWE-chat, we provide an initial empirical characterization of real-world coding agent usage and failure modes. We find that coding patterns are bimodal: in 41% of sessions, agents author virtually all committed code ("vibe coding"), while in 23%, humans write all code themselves. Despite rapidly improving capabilities, coding agents remain inefficient in natural settings. Just 44% of all agent-produced code survives into user commits, and agent-written code introduces more security vulnerabilities than code authored by humans. Furthermore, users push back against agent outputs -- through corrections, failure reports, and interruptions -- in 44% of all turns. By capturing complete interaction traces with human vs. agent code authorship attribution, SWE-chat provides an empirical foundation for moving beyond curated benchmarks towards an evidence-based understanding of how AI agents perform in real developer workflows.

**arXiv ID:** 2604.20779
</details>

<details>
<summary><strong>Automatic Ontology Construction Using LLMs as an External Layer of Memory, Verification, and Planning for Hybrid Intelligent Systems</strong> - Pavel Salovskii, Iuliia Gorshkova - [[pdf]](https://arxiv.org/pdf/2604.20795)</summary>

**Abstract:** This paper presents a hybrid architecture for intelligent systems in which large language models (LLMs) are extended with an external ontological memory layer. Instead of relying solely on parametric knowledge and vector-based retrieval (RAG), the proposed approach constructs and maintains a structured knowledge graph using RDF/OWL representations, enabling persistent, verifiable, and semantically grounded reasoning.
The core contribution is an automated pipeline for ontology construction from heterogeneous data sources, including documents, APIs, and dialogue logs. The system performs entity recognition, relation extraction, normalization, and triple generation, followed by validation using SHACL and OWL constraints, and continuous graph updates. During inference, LLMs operate over a combined context that integrates vector-based retrieval with graph-based reasoning and external tool interaction.
Experimental observations on planning tasks, including the Tower of Hanoi benchmark, indicate that ontology augmentation improves performance in multi-step reasoning scenarios compared to baseline LLM systems. In addition, the ontology layer enables formal validation of generated outputs, transforming the system into a generation-verification-correction pipeline.
The proposed architecture addresses key limitations of current LLM-based systems, including lack of long-term memory, weak structural understanding, and limited reasoning capabilities. It provides a foundation for building agent-based systems, robotics applications, and enterprise AI solutions that require persistent knowledge, explainability, and reliable decision-making.

**arXiv ID:** 2604.20795
</details>

<details>
<summary><strong>Accelerating PayPal's Commerce Agent with Speculative Decoding: An Empirical Study on EAGLE3 with Fine-Tuned Nemotron Models</strong> - Ally Qin, Jian Wan, Sarat Mudunuri, Srinivasan Manoharan - [[pdf]](https://arxiv.org/pdf/2604.19767)</summary>

**Abstract:** We evaluate speculative decoding with EAGLE3 as an inference-time optimization for PayPal's Commerce Agent, powered by a fine-tuned llama3.1-nemotron-nano-8B-v1 model. Building on prior work (NEMO-4-PAYPAL) that reduced latency and cost through domain-specific fine-tuning, we benchmark EAGLE3 via vLLM against NVIDIA NIM on identical 2xH100 hardware across 40 configurations spanning speculative token counts (gamma=3, gamma=5), concurrency levels (1-32), and sampling temperatures (0, 0.5). Key findings: (1) gamma=3 achieves 22-49% throughput improvement and 18-33% latency reduction at zero additional hardware cost; (2) acceptance rates remain stable at approximately 35.5% for gamma=3 across all conditions; (3) gamma=5 yields diminishing returns (approximately 25% acceptance rate); (4) LLM-as-Judge evaluation confirms fully preserved output quality; and (5) speculative decoding on a single H100 matches or exceeds NIM on two H100s, enabling 50% GPU cost reduction.

**arXiv ID:** 2604.19767
</details>

<details>
<summary><strong>EmbodiedMidtrain: Bridging the Gap between Vision-Language Models and Vision-Language-Action Models via Mid-training</strong> - Yiyang Du, Zhanqiu Guo, Xin Ye, Liu Ren, Chenyan Xiong - [[pdf]](https://arxiv.org/pdf/2604.20012)</summary>

**Abstract:** Vision-Language-Action Models (VLAs) inherit their visual and linguistic capabilities from Vision-Language Models (VLMs), yet most VLAs are built from off-the-shelf VLMs that are not adapted to the embodied domain, limiting their downstream performance. In this work, we propose EmbodiedMidtrain to bridge the gap between VLMs and VLAs. We first characterize the data distribution gap between them, showing that VLA data occupy compact regions that are largely separated from the broader VLM distribution, while the degree of alignment varies substantially both across and within VLM data sources. Then, we build a mid-training data engine that leverages a lightweight learnable proximity estimator to select the most VLA-aligned candidates from a large VLM pool, and mid-trains the VLM on this curated mixture before downstream VLA fine-tuning. Experiments on three robot manipulation benchmarks show that mid-training consistently improves performance across different VLM backbones, achieving results competitive with expert VLAs and off-the-shelf VLMs trained with larger model scale and training budgets. Further analysis reveals that mid-training provides a stronger initialization for VLA fine-tuning, with gains emerging from the earliest steps and widening throughout training. Moreover, the data engine captures both dataset-level and sample-level alignment signals, favoring spatial reasoning over text-centric tasks while preserving the diversity of the VLM data. We will release all code, data and models for future research.

**arXiv ID:** 2604.20012
</details>

<details>
<summary><strong>Auditing and Controlling AI Agent Actions in Spreadsheets</strong> - Sadra Sabouri, Zeinabsadat Saghi, Run Huang, Sujay Maladi, Esmeralda Eufracio, Sumit Gulwani, Souti Chattopadhyay - [[pdf]](https://arxiv.org/pdf/2604.20070)</summary>

**Abstract:** Advances in AI agent capabilities have outpaced users' ability to meaningfully oversee their execution. AI agents can perform sophisticated, multi-step knowledge work autonomously from start to finish, yet this process remains effectively inaccessible during execution, often buried within large volumes of intermediate reasoning and outputs: by the time users receive the output, all underlying decisions have already been made without their involvement. This lack of transparency leaves users unable to examine the agent's assumptions, identify errors before they propagate, or redirect execution when it deviates from their intent. The stakes are particularly high in spreadsheet environments, where process and artifact are inseparable. Each decision the agent makes is recorded directly in cells that belong to and reflect on the user. We introduce Pista, a spreadsheet AI agent that decomposes execution into auditable, controllable actions, providing users with visibility into the agent's decision-making process and the capacity to intervene at each step. A formative study (N = 8) and a within-subjects summative evaluation (N = 16) comparing Pista to a baseline agent demonstrated that active participation in execution influenced not only task outcomes but also users' comprehension of the task, their perception of the agent, and their sense of role within the workflow. Users identified their own intent reflected in the agent's actions, detected errors that post-hoc review would have failed to surface, and reported a sense of co-ownership over the resulting output. These findings indicate that meaningful human oversight of AI agents in knowledge work requires not improved post-hoc review mechanisms, but active participation in decisions as they are made.

**arXiv ID:** 2604.20070
</details>

<details>
<summary><strong>Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent Foundation Models Training</strong> - Tianqing Fang, Zhisong Zhang, Xiaoyang Wang, Rui Wang, Can Qin, Yuxuan Wan, Jun-Yu Ma, Ce Zhang, Jiaqi Chen, Xiyun Li, Yonglin Wang, Jingchen Ni, Tianshi Zheng, Chun Chen, Wenhao Yu, Zhenwen Liang, Hongming Zhang, Haitao Mi, Dong Yu - [[pdf]](https://arxiv.org/pdf/2508.00414)</summary>

**Abstract:** General AI Agents are increasingly recognized as foundational frameworks for the next generation of artificial intelligence, enabling complex reasoning, web interaction, coding, and autonomous research capabilities. However, current agent systems are either closed-source or heavily reliant on a variety of paid APIs and proprietary tools, limiting accessibility and reproducibility for the research community. In this work, we present \textbf{Cognitive Kernel-Pro}, a fully open-source and (to the maximum extent) free multi-module agent framework designed to democratize the development and evaluation of advanced AI agents. Within Cognitive Kernel-Pro, we systematically investigate the curation of high-quality training data for Agent Foundation Models, focusing on the construction of queries, trajectories, and verifiable answers across four key domains: web, file, code, and general reasoning. Furthermore, we explore novel strategies for agent test-time reflection and voting to enhance agent robustness and performance. We evaluate Cognitive Kernel-Pro on GAIA, achieving state-of-the-art results among open-source and free agents. Notably, our 8B-parameter open-source model surpasses previous leading systems such as WebDancer and WebSailor, establishing a new performance standard for accessible, high-capability AI agents. Code is available at this https URL

**arXiv ID:** 2508.00414
</details>

<details>
<summary><strong>AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite</strong> - Jonathan Bragg, Mike D'Arcy, Nishant Balepur, Dan Bareket, Bhavana Dalvi, Sergey Feldman, Dany Haddad, Jena D. Hwang, Peter Jansen, Varsha Kishore, Bodhisattwa Prasad Majumder, Aakanksha Naik, Sigal Rahamimov, Kyle Richardson, Amanpreet Singh, Harshit Surana, Aryeh Tiktinsky, Rosni Vasu, Guy Wiener, Chloe Anastasiades, Stefan Candra, Jason Dunkelberger, Dan Emery, Rob Evans, Malachi Hamada, Regan Huff, Rodney Kinney, Matt Latzke, Jaron Lochner, Ruben Lozano-Aguilera, Cecile Nguyen, Smita Rao, Amber Tanaka, Brooke Vlahos, Peter Clark, Doug Downey, Yoav Goldberg, Ashish Sabharwal, Daniel S. Weld - [[pdf]](https://arxiv.org/pdf/2510.21652)</summary>

**Abstract:** AI agents hold the potential to revolutionize scientific productivity by automating literature reviews, replicating experiments, analyzing data, and even proposing new directions of inquiry; indeed, there are now many such agents, ranging from general-purpose "deep research" systems to specialized science-specific agents, such as AI Scientist and AIGS. Rigorous evaluation of these agents is critical for progress. Yet existing benchmarks fall short on several fronts: they often (1) lack reproducible agent tools necessary for a controlled comparison of core agentic capabilities; (2) do not account for confounding variables such as model cost and tool access; (3) do not provide standardized interfaces for quick agent prototyping and evaluation; (4) fail to provide holistic, product-informed measures of real-world use cases such as science research; and (5) lack comprehensive baseline agents necessary to identify true advances. In response, we define principles and tooling for more rigorously benchmarking agents. Using these, we present AstaBench, a suite that provides a holistic measure of agentic ability to perform scientific research, comprising 2400+ problems spanning the entire scientific discovery process and multiple scientific domains, and including many problems inspired by actual user requests to deployed Asta agents. Our suite comes with the first scientific research environment with production-grade search tools that enable controlled, reproducible evaluation, better accounting for confounders. Alongside, we provide a comprehensive suite of nine science-optimized classes of Asta agents and numerous baselines. Our extensive evaluation of 57 agents across 22 agent classes reveals several interesting findings, most importantly that despite meaningful progress on certain individual aspects, AI remains far from solving the challenge of science research assistance.

**arXiv ID:** 2510.21652
</details>

<details>
<summary><strong>QuarkMedSearch: A Long-Horizon Deep Search Agent for Exploring Medical Intelligence</strong> - Zhichao Lin, Zhichao Liang, Gaoqiang Liu, Meng Xu, Baoyu Xiang, Jian Xu, Guanjun Jiang - [[pdf]](https://arxiv.org/pdf/2604.12867)</summary>

**Abstract:** As agentic foundation models continue to evolve, how to further improve their performance in vertical domains has become an important challenge. To this end, building upon Tongyi DeepResearch, a powerful agentic foundation model, we focus on the Chinese medical deep search scenario and propose QuarkMedSearch, systematically exploring a full-pipeline approach spanning medical multi-hop data construction, training strategies, and evaluation benchmarks to further push and assess its performance upper bound in vertical domains. Specifically, for data synthesis, to address the scarcity of deep search training data in the medical domain, we combine a large-scale medical knowledge graph with real-time online exploration to construct long-horizon medical deep search training data; for post-training, we adopt a two-stage SFT and RL training strategy that progressively enhances the model's planning, tool invocation, and reflection capabilities required for deep search, while maintaining search efficiency; for evaluation, we collaborate with medical experts to construct the QuarkMedSearch Benchmark through rigorous manual verification. Experimental results demonstrate that QuarkMedSearch achieves state-of-the-art performance among open-source models of comparable scale on the QuarkMedSearch Benchmark, while also maintaining strong competitiveness on general benchmarks.

**arXiv ID:** 2604.12867
</details>

<details>
<summary><strong>The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents</strong> - Xingyao Wang, Simon Rosenberg, Juan Michelini, Calvin Smith, Hoang Tran, Engel Nyst, Rohit Malhotra, Xuhui Zhou, Valerie Chen, Robert Brennan, Graham Neubig - [[pdf]](https://arxiv.org/pdf/2511.03690)</summary>

**Abstract:** Agents are now used widely in the process of software development, but building production-ready software engineering agents is a complex task. Deploying software agents effectively requires flexibility in implementation and experimentation, reliable and secure execution, and interfaces for users to interact with agents. In this paper, we present the OpenHands Software Agent SDK, a toolkit for implementing software development agents that satisfy these desiderata. This toolkit is a complete architectural redesign of the agent components of the popular OpenHands framework for software development agents.
To achieve flexibility, we design a simple interface for implementing agents that requires only a few lines of code in the default case, but is easily extensible to more complex full-featured agents with features such as custom tools, memory management, and more. For security and reliability, it delivers seamless local-to-remote execution portability, integrated REST/WebSocket services. For interaction with human users, it can connect directly to a variety of interfaces, such as visual workspaces (VSCode, VNC, browser), command-line interfaces, and APIs. Compared with existing SDKs from OpenAI, Claude and Google, OpenHands uniquely integrates native sandboxed execution, lifecycle control, model-agnostic multi-LLM routing, and built-in security analysis. We validate the architecture empirically: production deployment data shows that V1 substantially reduces system-attributable failures over V0 with negligible event-sourcing overhead, and evaluations across multiple models and benchmarks demonstrate strong agent performance. Put together, these elements allow the OpenHands Software Agent SDK to provide a practical foundation for prototyping, unlocking new classes of custom applications, and reliably deploying agents at scale.

**arXiv ID:** 2511.03690
</details>

<details>
<summary><strong>Beyond Task Success: An Evidence-Synthesis Framework for Evaluating, Governing, and Orchestrating Agentic AI</strong> - Christopher Koch, Joshua Andreas Wellbrock - [[pdf]](https://arxiv.org/pdf/2604.19818)</summary>

**Abstract:** Agentic AI systems plan, use tools, maintain state, and act across multi-step workflows with external effects, meaning trustworthy deployment can no longer be judged by task completion alone. The current literature remains fragmented across benchmark-centered evaluation, standards-based governance, orchestration architectures, and runtime assurance mechanisms. This paper contributes a bounded evidence synthesis across a manually coded corpus of twenty-four recent sources. The core finding is a governance-to-action closure gap: evaluation tells us whether outcomes were good, governance defines what should be allowed, but neither identifies where obligations bind to concrete actions or how compliance can later be proven. To close that gap, the paper introduces three linked artifacts: (1) a four-layer framework spanning evaluation, governance, orchestration, and assurance; (2) an ODTA runtime-placement test based on observability, decidability, timeliness, and attestability; and (3) a minimum action-evidence bundle for state-changing actions. Across sources, evaluation papers identify safety, robustness, and trajectory-level measurement as open gaps; governance frameworks define obligations but omit execution-time control logic; orchestration research positions the control plane as the locus of policy mediation, identity, and telemetry; runtime-governance work shows path-dependent behavior cannot be governed through prompts or static permissions alone; and action-safety studies show text alignment does not reliably transfer to tool actions. A worked enterprise procurement-agent scenario illustrates how these artifacts consolidate existing evidence without introducing new experimental data.

**arXiv ID:** 2604.19818
</details>

<details>
<summary><strong>From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents</strong> - Md Nayem Uddin, Kumar Shubham, Eduardo Blanco, Chitta Baral, Gengyu Wang - [[pdf]](https://arxiv.org/pdf/2604.20006)</summary>

**Abstract:** Personalized agents that interact with users over long periods must maintain persistent memory across sessions and update it as circumstances change. However, existing benchmarks predominantly frame long-term memory evaluation as fact retrieval from past conversations, providing limited insight into agents' ability to consolidate memory over time or handle frequent knowledge updates. We introduce Memora, a long-term memory benchmark spanning weeks to months long user conversations. The benchmark evaluates three memory-grounded tasks: remembering, reasoning, and recommending. To ensure data quality, we employ automated memory-grounding checks and human evaluation. We further introduce Forgetting-Aware Memory Accuracy (FAMA), a metric that penalizes reliance on obsolete or invalidated memory when evaluating long-term memory. Evaluations of four LLMs and six memory agents reveal frequent reuse of invalid memories and failures to reconcile evolving memories. Memory agents offer marginal improvements, exposing shortcomings in long-term memory for personalized agents.

**arXiv ID:** 2604.20006
</details>

<details>
<summary><strong>Dual-Cluster Memory Agent: Resolving Multi-Paradigm Ambiguity in Optimization Problem Solving</strong> - Xinyu Zhang, Yuchen Wan, Boxuan Zhang, Zesheng Yang, Lingling Zhang, Bifan Wei, Jun Liu - [[pdf]](https://arxiv.org/pdf/2604.20183)</summary>

**Abstract:** Large Language Models (LLMs) often struggle with structural ambiguity in optimization problems, where a single problem admits multiple related but conflicting modeling paradigms, hindering effective solution generation. To address this, we propose Dual-Cluster Memory Agent (DCM-Agent) to enhance performance by leveraging historical solutions in a training-free manner. Central to this is Dual-Cluster Memory Construction. This agent assigns historical solutions to modeling and coding clusters, then distills each cluster's content into three structured types: Approach, Checklist, and Pitfall. This process derives generalizable guidance knowledge. Furthermore, this agent introduces Memory-augmented Inference to dynamically navigate solution paths, detect and repair errors, and adaptively switch reasoning paths with structured knowledge. The experiments across seven optimization benchmarks demonstrate that DCM-Agent achieves an average performance improvement of 11%- 21%. Notably, our analysis reveals a ``knowledge inheritance'' phenomenon: memory constructed by larger models can guide smaller models toward superior performance, highlighting the framework's scalability and efficiency.

**arXiv ID:** 2604.20183
</details>

<details>
<summary><strong>Enhancing Agentic Textual Graph Retrieval with Synthetic Stepwise Supervision</strong> - Ge Chang, Jinbo Su, Jiacheng Liu, Pengfei Yang, Yuhao Shang, Huiwen Zheng, Hongli Ma, Yan Liang, Yuanchun Li, Yunxin Liu - [[pdf]](https://arxiv.org/pdf/2510.03323)</summary>

**Abstract:** Integrating textual graphs into Large Language Models (LLMs) is promising for complex graph-based QA. However, a key bottleneck is retrieving informative yet compact subgraphs that fit the LLM context. Existing retrievers often struggle, relying either on shallow embedding similarity or costly interactive policies that require excessive supervision. To address these challenges, we introduce an agentic textual graph reasoning framework featuring an LLM-based retriever trained with synthetic stepwise supervision. Rather than relying on final answer rewards which often yield sparse and unstable signals, we optimize the retriever by evaluating each step against offline-extracted golden subgraphs. Our approach distills golden subgraphs via a specialized data synthesis pipeline to formulate dense rewards, facilitating a two-stage training scheme that effectively learns the interactive graph exploration policy. Based on extensive experiments on three common datasets in comparison with seven strong baselines, our approach achieves an average improvement of 15.6% in accuracy and 17.2% in F1 score. The advantage is even higher in more complicated multi-hop reasoning tasks.

**arXiv ID:** 2510.03323
</details>

</details>

<details open>
<summary><h2>LLM Agents (15 papers)</h2></summary>

<details>
<summary><strong>From Data to Theory: Autonomous Large Language Model Agents for Materials Science</strong> - Samuel Onimpa Alfred, Veera Sundararaghavan - [[pdf]](https://arxiv.org/pdf/2604.19789)</summary>

**Abstract:** We present an autonomous large language model (LLM) agent for end-to-end, data-driven materials theory development. The model can choose an equation form, generate and run its own code, and test how well the theory matches the data without human intervention. The framework combines step-by-step reasoning with expert-supplied tools, allowing the agent to adjust its approach as needed while keeping a clear record of its decisions. For well-established materials relationships such as the Hall-Petch equation and Paris law, the agent correctly identifies the governing equation and makes reliable predictions on new datasets. For more specialized relationships, such as Kuhn's equation for the HOMO-LUMO gap of conjugated molecules as a function of length, performance depends more strongly on the underlying model, with GPT-5 showing better recovery of the correct equation. Beyond known theories, the agent can also suggest new predictive relationships, illustrated here by a strain-dependent law for changes in the HOMO-LUMO gap. At the same time, the results show that careful validation remains essential, because the agent can still return incorrect, incomplete, or inconsistent equations even when the numerical fit appears strong. Overall, these results highlight both the promise and the current limitations of autonomous LLM agents for AI-assisted scientific modeling and discovery.

**arXiv ID:** 2604.19789
</details>

<details>
<summary><strong>From Actions to Understanding: Conformal Interpretability of Temporal Concepts in LLM Agents</strong> - Trilok Padhi, Ramneet Kaur, Krishiv Agarwal, Adam D. Cobb, Daniel Elenius, Manoj Acharya, Colin Samplawski, Alexander M. Berenbeim, Nathaniel D. Bastian, Susmit Jha, Anirban Roy - [[pdf]](https://arxiv.org/pdf/2604.19775)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly deployed as autonomous agents capable of reasoning, planning, and acting within interactive environments. Despite their growing capability to perform multi-step reasoning and decision-making tasks, internal mechanisms guiding their sequential behavior remain opaque. This paper presents a framework for interpreting the temporal evolution of concepts in LLM agents through a step-wise conformal lens. We introduce the conformal interpretability framework for temporal tasks, which combines step-wise reward modeling with conformal prediction to statistically label model's internal representation at each step as successful or failing. Linear probes are then trained on these representations to identify directions of temporal concepts - latent directions in the model's activation space that correspond to consistent notions of success, failure or reasoning drift. Experimental results on two simulated interactive environments, namely ScienceWorld and AlfWorld, demonstrate that these temporal concepts are linearly separable, revealing interpretable structures aligned with task success. We further show preliminary results on improving an LLM agent's performance by leveraging the proposed framework for steering the identified successful directions inside the model. The proposed approach, thus, offers a principled method for early failure detection as well as intervention in LLM-based agents, paving the path towards trustworthy autonomous language models in complex interactive settings.

**arXiv ID:** 2604.19775
</details>

<details>
<summary><strong>SkillGraph: Graph Foundation Priors for LLM Agent Tool Sequence Recommendation</strong> - Hao Liu, Dongyu Li - [[pdf]](https://arxiv.org/pdf/2604.19793)</summary>

**Abstract:** LLM agents must select tools from large API libraries and order them correctly. Existing methods use semantic similarity for both retrieval and ordering, but ordering depends on inter-tool data dependencies that are absent from tool descriptions. As a result, semantic-only methods can produce negative Kendall-$\tau$ in structured workflow domains. We introduce SkillGraph, a directed weighted execution-transition graph mined from 49,831 successful LLM agent trajectories, which encodes workflow-precedence regularities as a reusable graph foundation prior. Building on this graph foundation prior, we propose a two-stage decoupled framework: GS-Hybrid retrieval for candidate selection and a learned pairwise reranker for ordering. On ToolBench (9,965 test instances; ~16,000 tools), the method reaches Set-F1 = 0.271 and Kendall-$\tau$ = 0.096; on API-Bank, Kendall-$\tau$ improves from -0.433 to +0.613. Under identical Stage-1 inputs, the learned reranker also outperforms LLaMA-3.1-8B Stage-2 rerankers.

**arXiv ID:** 2604.19793
</details>

<details>
<summary><strong>Separable Pathways for Causal Reasoning: How Architectural Scaffolding Enables Hypothesis-Space Restructuring in LLM Agents</strong> - John Alderete, Sebastian Benthal, Connie Xu, John Xing - [[pdf]](https://arxiv.org/pdf/2604.20039)</summary>

**Abstract:** Causal discovery through experimentation and intervention is fundamental to robust problem solving. It requires not just updating beliefs within a fixed framework but revising the hypothesis space itself, a capacity current AI agents lack when evidence demands representations they have not previously constructed. We extend the blicket detector paradigm from developmental science to test this capacity in AI agents equipped with architectural scaffolding that targets hypothesis-space restructuring. Our compositional architecture has two discrete components: context graphs, which structure exploration as typed state machines, and dynamic behaviors, which monitor for evidence that the current hypothesis space is inadequate and expand it at runtime. Across 1,085 experimental trials, these components make orthogonal contributions: context graphs drive reasoning quality within the post-switch hypothesis space, accounting for 94\% of the accuracy gain, while dynamic behaviors drive reasoning eligibility by detecting regime changes and preventing premature commitment to outdated hypotheses.

**arXiv ID:** 2604.20039
</details>

<details>
<summary><strong>FSFM: A Biologically-Inspired Framework for Selective Forgetting of Agent Memory</strong> - Yingjie Gu, Bo Xiong, Yijuan Guo, Chao Li, Xiaojing Zhang, Liqiang Wang, Pengcheng Ren, Qi Sun, Jingyao Ma, Shidang Shi - [[pdf]](https://arxiv.org/pdf/2604.20300)</summary>

**Abstract:** For LLM agents, memory management critically impacts efficiency, quality, and security. While much research focuses on retention, selective forgetting--inspired by human cognitive processes (hippocampal indexing/consolidation theory and Ebbinghaus forgetting curve)--remains underexplored. We argue that in resource-constrained environments, a well-designed forgetting mechanism is as crucial as remembering, delivering benefits across three dimensions: (1) efficiency via intelligent memory pruning, (2) quality by dynamically updating outdated preferences and context, and (3) security through active forgetting of malicious inputs, sensitive data, and privacy-compromising content. Our framework establishes a taxonomy of forgetting mechanisms: passive decay-based, active deletion-based, safety-triggered, and adaptive reinforcement-based. Building on advances in LLM agent architectures and vector databases, we present detailed specifications, implementation strategies, and empirical validation from controlled experiments. Results show significant improvements: access efficiency (+8.49%), content quality (+29.2% signal-to-noise ratio), and security performance (100% elimination of security risks). Our work bridges cognitive neuroscience and AI systems, offering practical solutions for real-world deployment while addressing ethical and regulatory compliance. The paper concludes with challenges and future directions, establishing selective forgetting as a fundamental capability for next-generation LLM agents operating in real-world, resource-constrained scenarios. Our contributions align with AI-native memory systems and responsible AI development.

**arXiv ID:** 2604.20300
</details>

<details>
<summary><strong>Cognis: Context-Aware Memory for Conversational AI Agents</strong> - Parshva Daftari, Khush Patel, Shreyas Kapale, Jithin George, Siva Surendira - [[pdf]](https://arxiv.org/pdf/2604.19771)</summary>

**Abstract:** LLM agents lack persistent memory, causing conversations to reset each session and preventing personalization over time. We present Lyzr Cognis, a unified memory architecture for conversational AI agents that addresses this limitation through a multi-stage retrieval pipeline. Cognis combines a dual-store backend pairing OpenSearch BM25 keyword matching with Matryoshka vector similarity search, fused via Reciprocal Rank Fusion. Its context-aware ingestion pipeline retrieves existing memories before extraction, enabling intelligent version tracking that preserves full memory history while keeping the store consistent. Temporal boosting enhances time-sensitive queries, and a BGE-2 cross-encoder reranker refines final result quality. We evaluate Cognis on two independent benchmarks -- LoCoMo and LongMemEval -- across eight answer generation models, demonstrating state-of-the-art performance on both. The system is open-source and deployed in production serving conversational AI applications.

**arXiv ID:** 2604.19771
</details>

<details>
<summary><strong>LLM Agents Predict Social Media Reactions but Do Not Outperform Text Classifiers: Benchmarking Simulation Accuracy Using 120K+ Personas of 1511 Humans</strong> - Ljubisa Bojic, Alexander Felfernig, Bojana Dinic, Velibor Ilic, Achim Rettinger, Vera Mevorah, Damian Trilling - [[pdf]](https://arxiv.org/pdf/2604.19787)</summary>

**Abstract:** Social media platforms mediate how billions form opinions and engage with public discourse. As autonomous AI agents increasingly participate in these spaces, understanding their behavioral fidelity becomes critical for platform governance and democratic resilience. Previous work demonstrates that LLM-powered agents can replicate aggregate survey responses, yet few studies test whether agents can predict specific individuals' reactions to specific content. This study benchmarks LLM-based agents' accuracy in predicting human social media reactions (like, dislike, comment, share, no reaction) across 120,000+ unique agent-persona combinations derived from 1,511 Serbian participants and 27 large language models. In Study 1, agents achieved 70.7% overall accuracy, with LLM choice producing a 13 percentage-point performance spread. Study 2 employed binary forced-choice (like/dislike) evaluation with chance-corrected metrics. Agents achieved Matthews Correlation Coefficient (MCC) of 0.29, indicating genuine predictive signal beyond chance. However, conventional text-based supervised classifiers using TF-IDF representations outperformed LLM agents (MCC of 0.36), suggesting predictive gains reflect semantic access rather than uniquely agentic reasoning. The genuine predictive validity of zero-shot persona-prompted agents warns against potential manipulation through easily deploying swarms of behaviorally distinct AI agents on social media, while simultaneously offering opportunities to use such agents in simulations for predicting polarization dynamics and informing AI policy. The advantage of using zero-shot agents is that they require no task-specific training, making their large-scale deployment easy across diverse contexts. Limitations include single-country sampling. Future research should explore multilingual testing and fine-tuning approaches.

**arXiv ID:** 2604.19787
</details>

<details>
<summary><strong>Taint-Style Vulnerability Detection and Confirmation for Node.js Packages Using LLM Agent Reasoning</strong> - Ronghao Ni, Mihai Christodorescu, Limin Jia - [[pdf]](https://arxiv.org/pdf/2604.20179)</summary>

**Abstract:** The rapidly evolving Node$.$js ecosystem currently includes millions of packages and is a critical part of modern software supply chains, making vulnerability detection of Node$.$js packages increasingly important. However, traditional program analysis struggles in this setting because of dynamic JavaScript features and the large number of package dependencies. Recent advances in large language models (LLMs) and the emerging paradigm of LLM-based agents offer an alternative to handcrafted program models. This raises the question of whether an LLM-centric, tool-augmented approach can effectively detect and confirm taint-style vulnerabilities (e.g., arbitrary command injection) in Node$.$js packages. We implement LLMVD$.$js, a multi-stage agent pipeline to scan code, propose vulnerabilities, generate proof-of-concept exploits, and validate them through lightweight execution oracles; and systematically evaluate its effectiveness in taint-style vulnerability detection and confirmation in Node$.$js packages without dedicated static/dynamic analysis engines for path derivation. For packages from public benchmarks, LLMVD$.$js confirms 84% of the vulnerabilities, compared to less than 22% for prior program analysis tools. It also outperforms a prior LLM-program-analysis hybrid approach while requiring neither vulnerability annotations nor prior vulnerability reports. When evaluated on a set of 260 recently released packages (without vulnerability groundtruth information), traditional tools produce validated exploits for few ($\leq 2$) packages, while LLMVD$.$js generates validated exploits for 36 packages.

**arXiv ID:** 2604.20179
</details>

<details>
<summary><strong>Trust, Lies, and Long Memories: Emergent Social Dynamics and Reputation in Multi-Round Avalon with LLM Agents</strong> - Suveen Ellawela - [[pdf]](https://arxiv.org/pdf/2604.20582)</summary>

**Abstract:** We study emergent social dynamics in LLM agents playing The Resistance: Avalon, a hidden-role deception game. Unlike prior work on single-game performance, our agents play repeated games while retaining memory of previous interactions, including who played which roles and how they behaved, enabling us to study how social dynamics evolve. Across 188 games, two key phenomena emerge. First, reputation dynamics emerge organically when agents retain cross-game memory: agents reference past behavior in statements like "I am wary of repeating last game's mistake of over-trusting early success." These reputations are role-conditional: the same agent is described as "straightforward" when playing good but "subtle" when playing evil, and high-reputation players receive 46% more team inclusions. Second, higher reasoning effort supports more strategic deception: evil players more often pass early missions to build trust before sabotaging later ones, 75% in high-effort games vs 36% in low-effort games. Together, these findings show that repeated interaction with memory gives rise to measurable reputation and deception dynamics among LLM agents.

**arXiv ID:** 2604.20582
</details>

<details>
<summary><strong>LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals</strong> - Joon Sung Park, Carolyn Q. Zou, Jonne Kamphorst, Niles Egan, Aaron Shaw, Benjamin Mako Hill, Carrie Cai, Meredith Ringel Morris, Percy Liang, Robb Willer, Michael S. Bernstein - [[pdf]](https://arxiv.org/pdf/2411.10109)</summary>

**Abstract:** Machine learning can predict human behavior well when substantial structured data and well-defined outcomes are available, but these models are typically limited to specific outcomes and cannot readily be applied to new domains. We test whether large language models (LLMs) can support a more general-purpose approach by building person-specific simulations (i.e., "generative agents") grounded in self-report data. Using data from a diverse national sample of 1,052 Americans, we build agents from (i) two-hour, semi-structured interviews (elicited using the American Voices Project interview schedule), (ii) structured surveys (the General Social Survey and Big Five personality inventory), or (iii) both sources combined. On held-out General Social Survey items, agent accuracy reached 83% (interview only), 82% (surveys only), and 86% (combined) of participants' two-week test-retest consistency, compared with agents prompted only with individuals' demographics (74%). Agents predicted personality traits and behaviors in experiments with similar accuracy, and reduced disparities in accuracy across racial and ideological groups relative to demographics-only baselines. Together, these results show that LLMs agents grounded in rich qualitative or quantitative self-report data can support general-purpose simulation of individuals across outcomes, without requiring task-specific training data.

**arXiv ID:** 2411.10109
</details>

<details>
<summary><strong>Lightweight LLM Agent Memory with Small Language Models</strong> - Jiaquan Zhang, Chaoning Zhang, Shuxu Chen, Zhenzhen Huang, Pengcheng Zheng, Zhicheng Wang, Ping Guo, Fan Mo, Sung-Ho Bae, Jie Zou, Jiwei Wei, Yang Yang - [[pdf]](https://arxiv.org/pdf/2604.07798)</summary>

**Abstract:** Although LLM agents can leverage tools for complex tasks, they still need memory to maintain cross-turn consistency and accumulate reusable information in long-horizon interactions. However, retrieval-based external memory systems incur low online overhead but suffer from unstable accuracy due to limited query construction and candidate filtering. In contrast, many systems use repeated large-model calls for online memory operations, improving accuracy but accumulating latency over long interactions. We propose LightMem, a lightweight memory system for better agent memory driven by Small Language Models (SLMs). LightMem modularizes memory retrieval, writing, and long-term consolidation, and separates online processing from offline consolidation to enable efficient memory invocation under bounded compute. We organize memory into short-term memory (STM) for immediate conversational context, mid-term memory (MTM) for reusable interaction summaries, and long-term memory (LTM) for consolidated knowledge, and uses user identifiers to support independent retrieval and incremental maintenance in multi-user settings. Online, LightMem operates under a fixed retrieval budget and selects memories via a two-stage procedure: vector-based coarse retrieval followed by semantic consistency re-ranking. Offline, it abstracts reusable interaction evidence and incrementally integrates it into LTM. Experiments show consistent gains across model scales, with an average F1 improvement of about 2.5 over A-MEM on LoCoMo, while achieving higher efficiency and low median latency (83 ms for retrieval and 581 ms end-to-end).

**arXiv ID:** 2604.07798
</details>

<details>
<summary><strong>SkillLearnBench: Benchmarking Continual Learning Methods for Agent Skill Generation on Real-World Tasks</strong> - Shanshan Zhong, Yi Lu, Jingjie Ning, Yibing Wan, Lihan Feng, Yuyi Ao, Leonardo F. R. Ribeiro, Markus Dreyer, Sean Ammirati, Chenyan Xiong - [[pdf]](https://arxiv.org/pdf/2604.20087)</summary>

**Abstract:** Skills have become the de facto way to enable LLM agents to perform complex real-world tasks with customized instructions, workflows, and tools, but how to learn them automatically and effectively remains unclear. We introduce SkillLearnBench, the first benchmark for evaluating continual skill learning methods, comprising 20 verified, skill-dependent tasks across 15 sub-domains derived from a real-world skill taxonomy , evaluated at three levels: skill quality, execution trajectory, and task outcome. Using this benchmark, we evaluate recent continual learning techniques, those leveraging one-shot, self/teacher feedback, and skill creator to generate skills from agent experiences. We find that all continual learning methods improve over the no-skill baseline, yet consistent gains remain elusive: no method leads across all tasks and LLMs, and scaling to stronger LLMs does not reliably help. Continual learning improves tasks with clear, reusable workflows but struggles on open-ended tasks, and using stronger LLM backbones does not consistently produce better skills. Our analysis also revealed that multiple iterations in continual learning facilitate genuine improvement via external feedback, whereas self-feedback alone induces recursive drift. Our data and code are open-source at this https URL to enable further studies of automatic skill generation and continual learning techniques.

**arXiv ID:** 2604.20087
</details>

<details>
<summary><strong>RExBench: Can coding agents autonomously implement AI research extensions?</strong> - Nicholas Edwards, Yukyung Lee, Yujun Audrey Mao, Yulu Qin, Sebastian Schuster, Najoung Kim - [[pdf]](https://arxiv.org/pdf/2506.22598)</summary>

**Abstract:** Agents based on Large Language Models (LLMs) have shown promise for performing sophisticated software engineering tasks autonomously. In addition, there has been progress towards developing agents that can perform parts of the research pipeline in machine learning and the natural sciences. We argue that research extension and its implementation is a critical capability for such systems, and introduce RExBench to support the evaluation of this capability. RExBench is a benchmark consisting of realistic extensions of 12 research papers that aim to investigate novel research hypotheses. Each task is set up as an extension to an existing research paper and codebase, accompanied by domain expert-written instructions. RExBench is robust to data contamination and supports an automatic evaluation infrastructure that executes agent outputs to determine whether the success criteria are met. We use this benchmark to evaluate 12 LLM agents implemented using two different frameworks, aider and OpenHands. We find that all agents fail to autonomously implement the majority of the extensions, with the best agent achieving around a 33% success rate. Although the success rate improves with additional human-written hints, the best performance under this setting remains below 44%. This indicates that current agents are still short of being able to handle realistic research extension tasks without substantial human guidance.

**arXiv ID:** 2506.22598
</details>

<details>
<summary><strong>CEDAR: Context Engineering for Agentic Data Science</strong> - Rishiraj Saha Roy, Chris Hinze, Luzian Hahn, Fabian Kuech - [[pdf]](https://arxiv.org/pdf/2601.06606)</summary>

**Abstract:** We demonstrate CEDAR, an application for automating data science (DS) tasks with an agentic setup. Solving DS problems with LLMs is an underexplored area that has immense market value. The challenges are manifold: task complexities, data sizes, computational limitations, and context restrictions. We show that these can be alleviated via effective context engineering. We first impose structure into the initial prompt with DS-specific input fields, that serve as instructions for the agentic system. The solution is then materialized as an enumerated sequence of interleaved plan and code blocks generated by separate LLM agents, providing a readable structure to the context at any step of the workflow. Function calls for generating these intermediate texts, and for corresponding Python code, ensure that data stays local, and only aggregate statistics and associated instructions are injected into LLM prompts. Fault tolerance and context management are introduced via iterative code generation and smart history rendering. The viability of our agentic data scientist is demonstrated using canonical Kaggle challenges.

**arXiv ID:** 2601.06606
</details>

<details>
<summary><strong>Explainable Iterative Data Visualisation Refinement via an LLM Agent</strong> - Burak Susam, Tingting Mu - [[pdf]](https://arxiv.org/pdf/2604.15319)</summary>

**Abstract:** Exploratory analysis of high-dimensional data relies on embedding the data into a low-dimensional space (typically 2D or 3D), based on which visualization plot is produced to uncover meaningful structures and to communicate geometric and distributional data characteristics. However, finding a suitable algorithm configuration, particularly hyperparameter setting, to produce a visualization plot that faithfully represents the underlying reality and encourages pattern discovery remains challenging. To address this challenge, we propose an agentic AI pipleline that leverages a large language model (LLM) to bridge the gap between rigorous quantitative assessment and qualitative human insight. By treating visualization evaluation and hyperparameter optimization as a semantic task, our system generates a multi-faceted report that contextualizes hard metrics with descriptive summaries, and suggests actionable recommendation of algorithm configuration for refining data visualization. By implementing an iterative optimization loop of this process, the system is able to produce rapidly a high-quality visualization plot, in full automation.

**arXiv ID:** 2604.15319
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (21 papers)</h2></summary>

<details>
<summary><strong>Prism: An Evolutionary Memory Substrate for Multi-Agent Open-Ended Discovery</strong> - Suyash Mishra - [[pdf]](https://arxiv.org/pdf/2604.19795)</summary>

**Abstract:** We introduce \prism{} (\textbf{P}robabilistic \textbf{R}etrieval with \textbf{I}nformation-\textbf{S}tratified \textbf{M}emory), an evolutionary memory substrate for multi-agent AI systems engaged in open-ended discovery. \prism{} unifies four independently developed paradigms -- layered file-based persistence, vector-augmented semantic memory, graph-structured relational memory, and multi-agent evolutionary search -- under a single decision-theoretic framework with eight interconnected subsystems.
We make five contributions: (1)~an \emph{entropy-gated stratification} mechanism that assigns memories to a tri-partite hub (skills/notes/attempts) based on Shannon information content, with formal context-window utilization bounds; (2)~a \emph{causal memory graph} $\mathcal{G} = (V, E_r, E_c)$ with interventional edges and agent-attributed provenance; (3)~a \emph{Value-of-Information retrieval} policy with self-evolving strategy selection; (4)~a \emph{heartbeat-driven consolidation} controller with stagnation detection via optimal stopping theory; and (5)~a \emph{replicator-decay dynamics} framework that interprets memory confidence as evolutionary fitness, proving convergence to an Evolutionary Stable Memory Set (ESMS). On the LOCOMO benchmark, \prism{} achieves 88.1 LLM-as-a-Judge score (31.2\% over Mem0). On CORAL-style evolutionary optimization tasks, 4-agent \prism{} achieves 2.8$\times$ higher improvement rate than single-agent baselines.%

**arXiv ID:** 2604.19795
</details>

<details>
<summary><strong>EvoAgent: An Evolvable Agent Framework with Skill Learning and Multi-Agent Delegation</strong> - Aimin Zhang, Jiajing Guo, Fuwei Jia, Chen Lv, Boyu Wang, Fangzheng Li - [[pdf]](https://arxiv.org/pdf/2604.20133)</summary>

**Abstract:** This paper proposes EvoAgent - an evolvable large language model (LLM) agent framework that integrates structured skill learning with a hierarchical sub-agent delegation mechanism. EvoAgent models skills as multi-file structured capability units equipped with triggering mechanisms and evolutionary metadata, and enables continuous skill generation and optimization through a user-feedback-driven closed-loop process. In addition, by incorporating a three-stage skill matching strategy and a three-layer memory architecture, the framework supports dynamic task decomposition for complex problems and long-term capability accumulation. Experimental results based on real-world foreign trade scenarios demonstrate that, after integrating EvoAgent, GPT5.2 achieves significant improvements in professionalism, accuracy, and practical utility. Under a five-dimensional LLM-as-Judge evaluation protocol, the overall average score increases by approximately 28%. Further model transfer experiments indicate that the performance of an agent system depends not only on the intrinsic capabilities of the underlying model, but also on the degree of synergy between the model and the agent architecture.

**arXiv ID:** 2604.20133
</details>

<details>
<summary><strong>Mol-Debate: Multi-Agent Debate Improves Structural Reasoning in Molecular Design</strong> - Wengyu Zhang, Xiao-Yong Wei, Qing Li - [[pdf]](https://arxiv.org/pdf/2604.20254)</summary>

**Abstract:** Text-guided molecular design is a key capability for AI-driven drug discovery, yet it remains challenging to map sequential natural-language instructions with non-linear molecular structures under strict chemical constraints. Most existing approaches, including RAG, CoT prompting, and fine-tuning or RL, emphasize a small set of ad-hoc reasoning perspectives implemented in a largely one-shot generation pipeline. In contrast, real-world drug discovery relies on dynamic, multi-perspective critique and iterative refinement to reconcile semantic intent with structural feasibility. Motivated by this, we propose Mol-Debate, a generation paradigm that enables such dynamic reasoning through an iterative generate-debate-refine loop. We further characterize key challenges in this paradigm and address them through perspective-oriented orchestration, including developer-debater conflict, global-local structural reasoning, and static-dynamic integration. Experiments demonstrate that Mol-Debate achieves state-of-the-art performance against strong general and chemical baselines, reaching 59.82% exact match on ChEBI-20 and 50.52% weighted success rate on S$^2$-Bench. Our code is available at this https URL.

**arXiv ID:** 2604.20254
</details>

<details>
<summary><strong>Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data</strong> - Fengxian Dong, Zhi Zheng, Xiao Han, Wei Chen, Jingqing Ruan, Tong Xu, Yong Chen, Enhong Chen - [[pdf]](https://arxiv.org/pdf/2604.20261)</summary>

**Abstract:** Automated feature generation extracts informative features from raw tabular data without manual intervention and is crucial for accurate, generalizable machine learning. Traditional methods rely on predefined operator libraries and cannot leverage task semantics, limiting their ability to produce diverse, high-value features for complex tasks. Recent Large Language Model (LLM)-based approaches introduce richer semantic signals, but still suffer from a restricted feature space due to fixed generation patterns and from the absence of feedback from the learning objective. To address these challenges, we propose a Memory-Augmented LLM-based Multi-Agent System (\textbf{MALMAS}) for automated feature generation. MALMAS decomposes the generation process into agents with distinct responsibilities, and a Router Agent activates an appropriate subset of agents per iteration, further broadening exploration of the feature space. We further integrate a memory module comprising procedural memory, feedback memory, and conceptual memory, enabling iterative refinement that adaptively guides subsequent feature generation and improves feature quality and diversity. Extensive experiments on multiple public datasets against state-of-the-art baselines demonstrate the effectiveness of our approach. The code is available at this https URL

**arXiv ID:** 2604.20261
</details>

<details>
<summary><strong>ActuBench: A Multi-Agent LLM Pipeline for Generation and Evaluation of Actuarial Reasoning Tasks</strong> - Jan-Philipp Schmidt - [[pdf]](https://arxiv.org/pdf/2604.20273)</summary>

**Abstract:** We present ActuBench, a multi-agent LLM pipeline for the automated generation and evaluation of advanced actuarial assessment items aligned with the International Actuarial Association (IAA) Education Syllabus. The pipeline separates four LLM roles by adapter: one agent drafts items, one constructs distractors, a third independently verifies both stages and drives bounded one-shot repair loops, and a cost-optimized auxiliary agent handles Wikipedia-note summarization and topic labelling. The items, per-model responses and complete leaderboard are published as a browsable web interface at this https URL, allowing readers and practitioners to inspect individual items without a repository checkout. We evaluate 50 language models from eight providers on two complementary benchmarks -- 100 empirically hardest multiple-choice items and 100 open-ended items scored by an LLM judge -- and report three headline findings. First, multi-agent verification is load-bearing: the independent verifier flags a majority of drafted items on first pass, most of which the one-shot repair loop resolves. Second, locally-hosted open-weights inference sits on the cost-performance Pareto front: a Gemma~4 model running on consumer hardware and a Cerebras-hosted 120B open-weights model dominate the near-zero-cost region, with the latter within one item of the top of the leaderboard. Third, MCQ and LLM-as-Judge rankings differ meaningfully: the MCQ scaffold inflates the performance ceiling, and Judge-mode evaluation is needed to discriminate at the frontier.

**arXiv ID:** 2604.20273
</details>

<details>
<summary><strong>Learning to Evolve: A Self-Improving Framework for Multi-Agent Systems via Textual Parameter Graph Optimization</strong> - Shan He, Runze Wang, Zhuoyun Du, Huiyu Bai, Zouying Cao, Yu Cheng, Bo Zheng - [[pdf]](https://arxiv.org/pdf/2604.20714)</summary>

**Abstract:** Designing and optimizing multi-agent systems (MAS) is a complex, labor-intensive process of "Agent Engineering." Existing automatic optimization methods, primarily focused on flat prompt tuning, lack the structural awareness to debug the intricate web of interactions in MAS. More critically, these optimizers are static; they do not learn from experience to improve their own optimization strategies. To address these gaps, we introduce Textual Parameter Graph Optimization (TPGO), a framework that enables a multi-agent system to learn to evolve. TPGO first models the MAS as a Textual Parameter Graph (TPG), where agents, tools, and workflows are modular, optimizable nodes. To guide evolution, we derive "textual gradients," structured natural language feedback from execution traces, to pinpoint failures and suggest granular modifications. The core of our framework is Group Relative Agent Optimization (GRAO), a novel meta-learning strategy that learns from historical optimization experiences. By analyzing past successes and failures, GRAO becomes progressively better at proposing effective updates, allowing the system to learn how to optimize itself. Extensive experiments on complex benchmarks like GAIA and MCP-Universe show that TPGO significantly enhances the performance of state-of-the-art agent frameworks, achieving higher success rates through automated, self-improving optimization.

**arXiv ID:** 2604.20714
</details>

<details>
<summary><strong>Soft-Label Governance for Distributional Safety in Multi-Agent Systems</strong> - Aizierjiang Aiersilan, Raeli Savitt - [[pdf]](https://arxiv.org/pdf/2604.19752)</summary>

**Abstract:** Multi-agent AI systems exhibit emergent risks that no single agent produces in isolation. Existing safety frameworks rely on binary classifications of agent behavior, discarding the uncertainty inherent in proxy-based evaluation. We introduce SWARM (\textbf{S}ystem-\textbf{W}ide \textbf{A}ssessment of \textbf{R}isk in \textbf{M}ulti-agent systems), a simulation framework that replaces binary good/bad labels with \emph{soft probabilistic labels} $p = P(v{=}+1) \in [0,1]$, enabling continuous-valued payoff computation, toxicity measurement, and governance intervention. SWARM implements a modular governance engine with configurable levers (transaction taxes, circuit breakers, reputation decay, and random audits) and quantifies their effects through probabilistic metrics including expected toxicity $\mathbb{E}[1{-}p \mid \text{accepted}]$ and quality gap $\mathbb{E}[p \mid \text{accepted}] - \mathbb{E}[p \mid \text{rejected}]$. Across seven scenarios with five-seed replication, strict governance reduces welfare by over 40\% without improving safety. In parallel, aggressively internalizing system externalities collapses total welfare from a baseline of $+262$ down to $-67$, while toxicity remains invariant. Circuit breakers require careful calibration; overly restrictive thresholds severely diminish system value, whereas an optimal threshold balances moderate welfare with minimized toxicity. Companion experiments show soft metrics detect proxy gaming by self-optimizing agents passing conventional binary evaluations. This basic governance layer applies to live LLM-backed agents (Concordia entities, Claude, GPT-4o Mini) without modification. Results show distributional safety requires \emph{continuous} risk metrics and governance lever calibration involves quantifiable safety-welfare tradeoffs. Source code and project resources are publicly available at this https URL.

**arXiv ID:** 2604.19752
</details>

<details>
<summary><strong>If you're waiting for a sign... that might not be it! Mitigating Trust Boundary Confusion from Visual Injections on Vision-Language Agentic Systems</strong> - Jiamin Chang, Minhui Xue, Ruoxi Sun, Shuchao Pang, Salil S. Kanhere, Hammond Pearce - [[pdf]](https://arxiv.org/pdf/2604.19844)</summary>

**Abstract:** Recent advances in embodied Vision-Language Agentic Systems (VLAS), powered by large vision-language models (LVLMs), enable AI systems to perceive and reason over real-world scenes. Within this context, environmental signals such as traffic lights are essential in-band signals that can and should influence agent behavior. However, similar signals could also be crafted to operate as misleading visual injections, overriding user intent and posing security risks. This duality creates a fundamental challenge: agents must respond to legitimate environmental cues while remaining robust to misleading ones. We refer to this tension as trust boundary confusion. To study this behavior, we design a dual-intent dataset and evaluation framework, through which we show that current LVLM-based agents fail to reliably balance this trade-off, either ignoring useful signals or following harmful ones. We systematically evaluate 7 LVLM agents across multiple embodied settings under both structure-based and noise-based visual injections. To address these vulnerabilities, we propose a multi-agent defense framework that separates perception from decision-making to dynamically assess the reliability of visual inputs. Our approach significantly reduces misleading behaviors while preserving correct responses and provides robustness guarantees under adversarial perturbations. The code of the evaluation framework and artifacts are made available at this https URL.

**arXiv ID:** 2604.19844
</details>

<details>
<summary><strong>ChipCraftBrain: Validation-First RTL Generation via Multi-Agent Orchestration</strong> - Cagri Eryilmaz - [[pdf]](https://arxiv.org/pdf/2604.19856)</summary>

**Abstract:** Large Language Models (LLMs) show promise for generating Register-Transfer Level (RTL) code from natural language specifications, but single-shot generation achieves only 60-65% functional correctness on standard benchmarks. Multi-agent approaches such as MAGE reach 95.9% on VerilogEval yet remain untested on harder industrial benchmarks such as NVIDIA's CVDP, lack synthesis awareness, and incur high API costs.
We present ChipCraftBrain, a framework combining symbolic-neural reasoning with adaptive multi-agent orchestration for automated RTL generation. Four innovations drive the system: (1) adaptive orchestration over six specialized agents via a PPO policy over a 168-dim state (an alternative world-model MPC planner is also evaluated); (2) a hybrid symbolic-neural architecture that solves K-map and truth-table problems algorithmically while specialized agents handle waveform timing and general RTL; (3) knowledge-augmented generation from a 321-pattern base plus 971 open-source reference implementations with focus-aware retrieval; and (4) hierarchical specification decomposition into dependency-ordered sub-modules with interface synchronization.
On VerilogEval-Human, ChipCraftBrain achieves 97.2% mean pass@1 (range 96.15-98.72% across 7 runs, best 154/156), on par with ChipAgents (97.4%, self-reported) and ahead of MAGE (95.9%). On a 302-problem non-agentic subset of CVDP spanning five task categories, we reach 94.7% mean pass@1 (286/302, averaged over 3 runs), a 36-60 percentage-point lift per category over the published single-shot baseline; we additionally lead three of four categories shared with NVIDIA's ACE-RTL despite using roughly 30x fewer per-problem attempts. A RISC-V SoC case study demonstrates hierarchical decomposition generating 8/8 lint-passing modules (689 LOC) validated on FPGA, where monolithic generation fails entirely.

**arXiv ID:** 2604.19856
</details>

<details>
<summary><strong>TriEx: A Game-based Tri-View Framework for Explaining Internal Reasoning in Multi-Agent LLMs</strong> - Ziyi Wang, Chen Zhang, Wenjun Peng, Qi Wu, Xinyu Wang - [[pdf]](https://arxiv.org/pdf/2604.20043)</summary>

**Abstract:** Explainability for Large Language Model (LLM) agents is especially challenging in interactive, partially observable settings, where decisions depend on evolving beliefs and other agents. We present \textbf{TriEx}, a tri-view explainability framework that instruments sequential decision making with aligned artifacts: (i) structured first-person self-reasoning bound to an action, (ii) explicit second-person belief states about opponents updated over time, and (iii) third-person oracle audits grounded in environment-derived reference signals. This design turns explanations from free-form narratives into evidence-anchored objects that can be compared and checked across time and perspectives. Using imperfect-information strategic games as a controlled testbed, we show that TriEx enables scalable analysis of explanation faithfulness, belief dynamics, and evaluator reliability, revealing systematic mismatches between what agents say, what they believe, and what they do. Our results highlight explainability as an interaction-dependent property and motivate multi-view, evidence-grounded evaluation for LLM agents. Code is available at this https URL.

**arXiv ID:** 2604.20043
</details>

<details>
<summary><strong>IMPACT-CYCLE: A Contract-Based Multi-Agent System for Claim-Level Supervisory Correction of Long-Video Semantic Memory</strong> - Weitong Kong, Di Wen, Kunyu Peng, David Schneider, Zeyun Zhong, Alexander Jaus, Zdravko Marinov, Jiale Wei, Ruiping Liu, Junwei Zheng, Yufan Chen, Lei Qi, Rainer Stiefelhagen - [[pdf]](https://arxiv.org/pdf/2604.20136)</summary>

**Abstract:** Correcting errors in long-video understanding is disproportionately costly: existing multimodal pipelines produce opaque, end-to-end outputs that expose no intermediate state for inspection, forcing annotators to revisit raw video and reconstruct temporal logic from scratch. The core bottleneck is not generation quality alone, but the absence of a supervisory interface through which human effort can be proportional to the scope of each error. We present IMPACT-CYCLE, a supervisory multi-agent system that reformulates long-video understanding as iterative claim-level maintenance of a shared semantic memory -- a structured, versioned state encoding typed claims, a claim dependency graph, and a provenance log. Role-specialized agents operating under explicit authority contracts decompose verification into local object-relation correctness, cross-temporal consistency, and global semantic coherence, with corrections confined to structurally dependent claims. When automated evidence is insufficient, the system escalates to human arbitration as the supervisory authority with final override rights; dependency-closure re-verification then ensures correction cost remains proportional to error scope. Experiments on VidOR show substantially improved downstream reasoning (VQA: 0.71 to 0.79) and a 4.8x reduction in human arbitration cost, with workload significantly lower than manual annotation. Code will be released at this https URL.

**arXiv ID:** 2604.20136
</details>

<details>
<summary><strong>Bimanual Robot Manipulation via Multi-Agent In-Context Learning</strong> - Alessio Palma, Indro Spinelli, Vignesh Prasad, Luca Scofano, Yufeng Jin, Georgia Chalvatzaki, Fabio Galasso - [[pdf]](https://arxiv.org/pdf/2604.20348)</summary>

**Abstract:** Language Models (LLMs) have emerged as powerful reasoning engines for embodied control. In particular, In-Context Learning (ICL) enables off-the-shelf, text-only LLMs to predict robot actions without any task-specific training while preserving their generalization capabilities. Applying ICL to bimanual manipulation remains challenging, as the high-dimensional joint action space and tight inter-arm coordination constraints rapidly overwhelm standard context windows. To address this, we introduce BiCICLe (Bimanual Coordinated In-Context Learning), the first framework that enables standard LLMs to perform few-shot bimanual manipulation without fine-tuning. BiCICLe frames bimanual control as a multi-agent leader-follower problem, decoupling the action space into sequential, conditioned single-arm predictions. This naturally extends to Arms' Debate, an iterative refinement process, and to the introduction of a third LLM-as-Judge to evaluate and select the most plausible coordinated trajectories. Evaluated on 13 tasks from the TWIN benchmark, BiCICLe achieves up to 71.1% average success rate, outperforming the best training-free baseline by 6.7 percentage points and surpassing most supervised methods. We further demonstrate strong few-shot generalization on novel tasks.

**arXiv ID:** 2604.20348
</details>

<details>
<summary><strong>Enhancing Research Idea Generation through Combinatorial Innovation and Multi-Agent Iterative Search Strategies</strong> - Shuai Chen, Chengzhi Zhang - [[pdf]](https://arxiv.org/pdf/2604.20548)</summary>

**Abstract:** Scientific progress depends on the continual generation of innovative re-search ideas. However, the rapid growth of scientific literature has greatly increased the cost of knowledge filtering, making it harder for researchers to identify novel directions. Although existing large language model (LLM)-based methods show promise in research idea generation, the ideas they produce are often repetitive and lack depth. To address this issue, this study proposes a multi-agent iterative planning search strategy inspired by com-binatorial innovation theory. The framework combines iterative knowledge search with an LLM-based multi-agent system to generate, evaluate, and re-fine research ideas through repeated interaction, with the goal of improving idea diversity and novelty. Experiments in the natural language processing domain show that the proposed method outperforms state-of-the-art base-lines in both diversity and novelty. Further comparison with ideas derived from top-tier machine learning conference papers indicates that the quality of the generated ideas falls between that of accepted and rejected papers. These results suggest that the proposed framework is a promising approach for supporting high-quality research idea generation. The source code and dataset used in this paper are publicly available on Github repository: this https URL. The demo is available at this https URL.

**arXiv ID:** 2604.20548
</details>

<details>
<summary><strong>FELA: A Multi-Agent Evolutionary System for Feature Engineering of Industrial Event Log Data</strong> - Kun Ouyang, Haoyu Wang, Dong Fang - [[pdf]](https://arxiv.org/pdf/2510.25223)</summary>

**Abstract:** Event log data, recording fine-grained user actions and system events, represent one of the most valuable assets for modern digital services. However, the complexity and heterogeneity of industrial event logs--characterized by large scale, high dimensionality, diverse data types, and intricate temporal or relational structures--make feature engineering extremely challenging. Existing automatic feature engineering approaches, such as AutoML or genetic methods, often suffer from limited explainability, rigid predefined operations, and poor adaptability to complicated heterogeneous data. In this paper, we propose FELA (Feature Engineering LLM Agents), a multi-agent evolutionary system that autonomously extracts meaningful and high-performing features from complex industrial event log data. FELA integrates the reasoning and coding capabilities of large language models (LLMs) with an insight-guided self-evolution paradigm. Specifically, FELA employs specialized agents--Idea Agents, Code Agents, and Critic Agents--to collaboratively generate, validate, and implement novel feature ideas. An Evaluation Agent summarizes feedback and updates a hierarchical knowledge base and dual-memory system to enable continual improvement. Moreover, FELA introduces an agentic evolution algorithm, combining reinforcement learning and genetic algorithm principles to balance exploration and exploitation across the idea space. Extensive experiments on real industrial datasets demonstrate that FELA can generate explainable, domain-relevant features that significantly improve model performance while reducing manual effort. Our results highlight the potential of LLM-based multi-agent systems as a general framework for automated, interpretable, and adaptive feature engineering in complex real-world environments.

**arXiv ID:** 2510.25223
</details>

<details>
<summary><strong>TREX: Automating LLM Fine-tuning via Agent-Driven Tree-based Exploration</strong> - Zerun Ma, Guoqiang Wang, Xinchen Xie, Yicheng Chen, He Du, Bowen Li, Yanan Sun, Wenran Liu, Kai Chen, Yining Li - [[pdf]](https://arxiv.org/pdf/2604.14116)</summary>

**Abstract:** While Large Language Models (LLMs) have empowered AI research agents to perform isolated scientific tasks, automating complex, real-world workflows, such as LLM training, remains a significant challenge. In this paper, we introduce TREX, a multi-agent system that automates the entire LLM training life-cycle. By orchestrating collaboration between two core modules-the Researcher and the Executor-the system seamlessly performs requirement analysis, open-domain literature and data research, formulation of training strategies, preparation of data recipes, and model training and evaluation. The multi-round experimental process is modeled as a search tree, enabling the system to efficiently plan exploration paths, reuse historical results, and distill high-level insights from iterative trials. To evaluate the capability of automated LLM training, we construct FT-Bench, a benchmark comprising 10 tasks derived from real-world scenarios, ranging from optimizing fundamental model capabilities to enhancing performance on domain-specific tasks. Experimental results demonstrate that the TREX agent consistently optimizes model performance on target tasks.

**arXiv ID:** 2604.14116
</details>

<details>
<summary><strong>Explicit Trait Inference for Multi-Agent Coordination</strong> - Suhaib Abdurahman, Etsuko Ishii, Katerina Margatina, Divya Bhargavi, Monica Sunkara, Yi Zhang - [[pdf]](https://arxiv.org/pdf/2604.19278)</summary>

**Abstract:** LLM-based multi-agent systems (MAS) show promise on complex tasks but remain prone to coordination failures such as goal drift, error cascades, and misaligned behaviors. We propose Explicit Trait Inference (ETI), a psychologically grounded method for improving coordination. ETI enables agents to infer and track partner characteristics along two established psychological dimensions--warmth (e.g., trust) and competence (e.g., skill)--from interaction histories to guide decisions. We evaluate ETI in controlled settings (economic games), where it reduces payoff loss by 45-77%, and in more realistic, complex multi-agent settings (MultiAgentBench), where it improves performance by 3-29% depending on the scenario and model, relative to a CoT baseline. Additional analysis shows that gains are closely linked to trait inference: ETI profiles predict agents' actions, and informative profiles drive improvements. These results highlight ETI as a lightweight and robust mechanism for improving coordination in diverse multi-agent settings, and provide the first systematic evidence that LLM agents can (i) reliably infer others' traits from interaction histories and (ii) leverage structured awareness of others' traits for coordination.

**arXiv ID:** 2604.19278
</details>

<details>
<summary><strong>Meta-Offline and Distributional Multi-Agent RL for Risk-Aware Decision-Making</strong> - Eslam Eldeeb, Hirley Alves - [[pdf]](https://arxiv.org/pdf/2501.16098)</summary>

**Abstract:** Mission critical applications, such as UAV-assisted IoT networks require risk-aware decision-making under dynamic topologies and uncertain channels. We propose meta-conservative quantile regression (M-CQR), a meta-offline distributional MARL algorithm that integrates conservative Q-learning (CQL) for safe offline learning, quantile regression DQN (QR-DQN) for risk-sensitive value estimation, and model-agnostic meta-learning (MAML) for rapid adaptation. Two variants are developed: meta-independent CQR (M-I-CQR) and meta-CTDE-CQR. In a UAV-based communication scenario, M-CTDE-CQR achieves up to 50% faster convergence and outperforms baseline MARL methods, offering improved scalability, robustness, and adaptability for risk-sensitive decision-making. Code is available at this https URL

**arXiv ID:** 2501.16098
</details>

<details>
<summary><strong>Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents</strong> - Huangwei Chen, Wu Li, Junhao Jia, Yining Chen, Xiaotao Pang, Ya-Long Chen, Li Gonghui, Haishuai Wang, Jiajun Bu, Lei Wu - [[pdf]](https://arxiv.org/pdf/2604.08927)</summary>

**Abstract:** The initial outpatient consultation is critical for clinical decision-making, yet it is often conducted by a single physician under time pressure, making it prone to cognitive biases and incomplete evidence capture. Although the Multi-Disciplinary Team (MDT) reduces these risks, they are costly and difficult to scale to real-time intake. We propose Aegle, a synchronous virtual MDT framework that brings MDT-level reasoning to outpatient consultations via a graph-based multi-agent architecture. Aegle formalizes the consultation state using a structured SOAP representation, separating evidence collection from diagnostic reasoning to improve traceability and bias control. An orchestrator dynamically activates specialist agents, which perform decoupled parallel reasoning and are subsequently integrated by an aggregator into a coherent clinical note. Experiments on ClinicalBench and a real-world RAPID-IPN dataset across 24 departments and 53 metrics show that Aegle consistently outperforms state-of-the-art proprietary and open-source models in documentation quality and consultation capability, while also improving final diagnosis accuracy. Our code is available at this https URL.

**arXiv ID:** 2604.08927
</details>

<details>
<summary><strong>Superficial Success vs. Internal Breakdown: An Empirical Study of Generalization in Adaptive Multi-Agent Systems</strong> - Namyoung So, Seokgyu Jang, Taeuk Kim - [[pdf]](https://arxiv.org/pdf/2604.18951)</summary>

**Abstract:** Adaptive multi-agent systems (MAS) are increasingly adopted to tackle complex problems. However, the narrow task coverage of their optimization raises the question of whether they can function as general-purpose systems. To address this gap, we conduct an extensive empirical study of adaptive MAS, revealing two key findings: (1) topological overfitting -- they fail to generalize across different domains; and (2) illusory coordination -- they achieve reasonable surface-level accuracy while the underlying agent interactions diverge from ideal MAS behavior, raising concerns about their practical utility. These findings highlight the pressing need to prioritize generalization in MAS development and motivate evaluation protocols that extend beyond simple final-answer correctness.

**arXiv ID:** 2604.18951
</details>

<details>
<summary><strong>Cooperative Profiles Predict Multi-Agent LLM Team Performance in AI for Science Workflows</strong> - Shivani Kumar, Adarsh Bharathwaj, David Jurgens - [[pdf]](https://arxiv.org/pdf/2604.20658)</summary>

**Abstract:** Multi-agent systems built from teams of large language models (LLMs) are increasingly deployed for collaborative scientific reasoning and problem-solving. These systems require agents to coordinate under shared constraints, such as GPUs or credit balances, where cooperative behavior matters. Behavioral economics provides a rich toolkit of games that isolate distinct cooperation mechanisms, yet it remains unknown whether a model's behavior in these stylized settings predicts its performance in realistic collaborative tasks. Here, we benchmark 35 open-weight LLMs across six behavioral economics games and show that game-derived cooperative profiles robustly predict downstream performance in AI-for-Science tasks, where teams of LLM agents collaboratively analyze data, build models, and produce scientific reports under shared budget constraints. Models that effectively coordinate games and invest in multiplicative team production (rather than greedy strategies) produce better scientific reports across three outcomes, accuracy, quality, and completion. These associations hold after controlling for multiple factors, indicating that cooperative disposition is a distinct, measurable property of LLMs not reducible to general ability. Our behavioral games framework thus offers a fast and inexpensive diagnostic for screening cooperative fitness before costly multi-agent deployment.

**arXiv ID:** 2604.20658
</details>

<details>
<summary><strong>A Delta-Aware Orchestration Framework for Scalable Multi-Agent Edge Computing</strong> - Samaresh Kumar Singh, Joyjit Roy - [[pdf]](https://arxiv.org/pdf/2604.20129)</summary>

**Abstract:** The Synergistic Collapse occurs when scaling beyond 100 agents causes superlinear performance degradation that individual optimizations cannot prevent. We observe this collapse with 150 cameras in Smart City deployment using MADDPG, where Deadline Satisfaction drops from 78% to 34%, producing approximately $180,000 in annual cost overruns. Prior work has addressed each contributing factor in isolation: exponential action-space growth, computational redundancy among spatially adjacent agents, and task-agnostic hardware scheduling. None has examined how these three factors interact and amplify each other. We present DAOEF (Delta-Aware Orchestration for Edge Federations), a framework that addresses all three simultaneously through: (1) Differential Neural Caching, which stores intermediate layer activations and computes only the input deltas, achieving 2.1x higher hit ratios (72% vs. 35%) than output-level caching while staying within 2% accuracy loss through empirically calibrated similarity thresholds; (2) Criticality-Based Action Space Pruning, which organizes agents into priority tiers and reduces coordination complexity from O(n2) to O(n log n) with less than 6% optimality loss; and (3) Learned Hardware Affinity Matching, which assigns tasks to their optimal accelerator (GPU, CPU, NPU, or FPGA) to prevent compounding mismatch penalties. Controlled factor-isolation experiments confirm that each mechanism is necessary but insufficient on its own: removing any single mechanism increases latency by more than 40%, validating that the gains are interdependent rather than additive. Across four datasets (100-250 agents) and a 20-device physical testbed, DAOEF achieves a 1.45x multiplicative gain over applying the three mechanisms independently. A 200-agent cloud deployment yields 62% latency reduction (280 ms vs. 735 ms), sub-linear latency growth up to 250 agents.

**arXiv ID:** 2604.20129
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>The AI Telco Engineer: Toward Autonomous Discovery of Wireless Communications Algorithms</strong> - Fayçal Aït Aoudia, Jakob Hoydis, Sebastian Cammerer, Lorenzo Maggi, Gian Marti, Alexander Keller - [[pdf]](https://arxiv.org/pdf/2604.19803)</summary>

**Abstract:** Agentic AI is rapidly transforming the way research is conducted, from prototyping ideas to reproducing results found in the literature. In this paper, we explore the ability of agentic AI to autonomously design wireless communication algorithms. To that end, we implement a dedicated framework that leverages large language models (LLMs) to iteratively generate, evaluate, and refine candidate algorithms. We evaluate the framework on three tasks spanning the physical (PHY) and medium access control (MAC) layers: statistics-agnostic channel estimation, channel estimation with known covariance, and link adaptation. Our results show that, in a matter of hours, the framework produces algorithms that are competitive with and, in some cases, outperforming conventional baselines. Moreover, unlike neural network-based approaches, the generated algorithms are fully explainable and extensible. This work represents a first step toward the autonomous discovery of novel wireless communication algorithms, and we look forward to the progress our community makes in this direction.

**arXiv ID:** 2604.19803
</details>

<details>
<summary><strong>Stateless Decision Memory for Enterprise AI Agents</strong> - Vasundra Srinivasan - [[pdf]](https://arxiv.org/pdf/2604.20158)</summary>

**Abstract:** Enterprise deployment of long-horizon decision agents in regulated domains (underwriting, claims adjudication, tax examination) is dominated by retrieval-augmented pipelines despite a decade of increasingly sophisticated stateful memory architectures. We argue this reflects a hidden requirement: regulated deployment is load-bearing on four systems properties (deterministic replay, auditable rationale, multi-tenant isolation, statelessness for horizontal scale), and stateful architectures violate them by construction. We propose Deterministic Projection Memory (DPM): an append-only event log plus one task-conditioned projection at decision time. On ten regulated decisioning cases at three memory budgets, DPM matches summarization-based memory at generous budgets and substantially outperforms it when the budget binds: at a 20x compression ratio, DPM improves factual precision by +0.52 (Cohen's h=1.17, p=0.0014) and reasoning coherence by +0.53 (h=1.13, p=0.0034), paired permutation, n=10. DPM is additionally 7-15x faster at binding budgets, making one LLM call at decision time instead of N. A determinism study of 10 replays per case at temperature zero shows both architectures inherit residual API-level nondeterminism, but the asymmetry is structural: DPM exposes one nondeterministic call; summarization exposes N compounding calls. The audit surface follows the same one-versus-N pattern: DPM logs two LLM calls per decision while summarization logs 83-97 on LongHorizon-Bench. We conclude with TAMS, a practitioner heuristic for architecture selection, and a failure analysis of stateful memory under enterprise operating conditions. The contribution is the argument that statelessness is the load-bearing property explaining enterprise's preference for weaker but replayable retrieval pipelines, and that DPM demonstrates this property is attainable without the decisioning penalty retrieval pays.

**arXiv ID:** 2604.20158
</details>

<details>
<summary><strong>Information Aggregation with AI Agents</strong> - Spyros Galanis - [[pdf]](https://arxiv.org/pdf/2604.20050)</summary>

**Abstract:** Can Large Language Models (AI agents) aggregate dispersed private information through trading and reason about the knowledge of others by observing price movements? We conduct a controlled experiment where AI agents trade in a prediction market after receiving private signals, measuring information aggregation by the log error of the last price. We find that although the median market is effective at aggregating information in the easy information structures, increasing the complexity has a significant and negative impact, suggesting that AI agents may suffer from the same limitations as humans when reasoning about others. Consistent with our theoretical predictions, information aggregation remains unaffected by allowing cheap talk communication, changing the duration of the market or initial price, and strategic prompting-thus demonstrating that prediction markets are robust. We establish that "smarter" AI agents perform better at aggregation and they are more profitable. Surprisingly, giving them feedback about past performance makes them worse at aggregation and reduces their profits.

**arXiv ID:** 2604.20050
</details>

<details>
<summary><strong>Supplement Generation Training for Enhancing Agentic Task Performance</strong> - Young Min Cho, Daniele Bonadiman, Divya Bhargavi, Tamer Alkhouli, Salvatore Romeo, Dongwei Jiang, Khushbu Pahwa, Yubin Ge, Etsuko Ishii, Monica Sunkara, Yi Zhang - [[pdf]](https://arxiv.org/pdf/2604.20727)</summary>

**Abstract:** Training large foundation models for agentic tasks is increasingly impractical due to the high computational costs, long iteration cycles, and rapid obsolescence as new models are continuously released. Instead of post-training massive models for every new task or domain, we propose Supplement Generation Training (SGT), a more efficient and sustainable strategy. SGT trains a smaller LLM to generate useful supplemental text that, when appended to the original input, helps the larger LLM solve the task more effectively. These lightweight models can dynamically adapt supplements to task requirements, improving performance without modifying the underlying large models. This approach decouples task-specific optimization from large foundation models and enables more flexible, cost-effective deployment of LLM-powered agents in real-world applications.

**arXiv ID:** 2604.20727
</details>

<details>
<summary><strong>Toward Cooperative Driving in Mixed Traffic: An Adaptive Potential Game-Based Approach with Field Test Verification</strong> - Shiyu Fang, Xiaocong Zhao, Xuekai Liu, Peng Hang, Jianqiang Wang, Yunpeng Wang, Jian Sun - [[pdf]](https://arxiv.org/pdf/2604.20231)</summary>

**Abstract:** Connected autonomous vehicles (CAVs), which represent a significant advancement in autonomous driving technology, have the potential to greatly increase traffic safety and efficiency through cooperative decision-making. However, existing methods often overlook the individual needs and heterogeneity of cooperative participants, making it difficult to transfer them to environments where they coexist with human-driven vehicles (HDVs).To address this challenge, this paper proposes an adaptive potential game (APG) cooperative driving framework. First, the system utility function is established on the basis of a general form of individual utility and its monotonic relationship, allowing for the simultaneous optimization of both individual and system objectives. Second, the Shapley value is introduced to compute each vehicle's marginal utility within the system, allowing its varying impact to be quantified. Finally, the HDV preference estimation is dynamically refined by continuously comparing the observed HDV behavior with the APG's estimated actions, leading to improvements in overall system safety and efficiency. Ablation studies demonstrate that adaptively updating Shapley values and HDV preference estimation significantly improve cooperation success rates in mixed traffic. Comparative experiments further highlight the APG's advantages in terms of safety and efficiency over other cooperative methods. Moreover, the applicability of the approach to real-world scenarios was validated through field tests.

**arXiv ID:** 2604.20231
</details>

<details>
<summary><strong>Open-Architecture End-to-End System for Real-World Autonomous Robot Navigation</strong> - Venkata Naren Devarakonda, Ali Umut Kaypak, Raktim Gautam Goswami, Naman Patel, Rooholla Khorrambakht, Prashanth Krishnamurthy, Farshad Khorrami - [[pdf]](https://arxiv.org/pdf/2410.06239)</summary>

**Abstract:** Enabling robots to autonomously navigate unknown, complex, and dynamic real-world environments presents several challenges, including imperfect perception, partial observability, localization uncertainty, and safety constraints. Current approaches are typically limited to simulations, where such challenges are not present. In this work, we present a lightweight, open-architecture, end-to-end system for real-world robot autonomous navigation. Specifically, we deploy a real-time navigation system on a quadruped robot by integrating multiple onboard components that communicate via ROS2. Given navigation tasks specified in natural language, the system fuses onboard sensory data for localization and mapping with open-vocabulary semantics to build hierarchical scene graphs from a continuously updated semantic object map. An LLM-based planner leverages these graphs to generate and adapt multi-step plans in real time as the scene evolves. Through experiments across multiple indoor environments using a Unitree Go2 quadruped, we demonstrate zero-shot real-world autonomous navigation, achieving over 88% task success, and provide analysis of system behavior during deployment.

**arXiv ID:** 2410.06239
</details>

<details>
<summary><strong>NanoCockpit: Performance-optimized Application Framework for AI-based Autonomous Nanorobotics</strong> - Elia Cereda, Alessandro Giusti, Daniele Palossi - [[pdf]](https://arxiv.org/pdf/2601.07476)</summary>

**Abstract:** Autonomous nano-drones, powered by vision-based tiny machine learning (TinyML) models, are a novel technology gaining momentum thanks to their broad applicability and pushing scientific advancement on resource-limited embedded systems. Their small form factor, i.e., a few tens of grams, severely limits their onboard computational resources to sub-100mW microcontroller units (MCUs). The Bitcraze Crazyflie nano-drone is the de facto standard, offering a rich set of programmable MCUs for low-level control, multi-core processing, and radio transmission. However, roboticists very often underutilize these onboard precious resources due to the absence of a simple yet efficient software layer capable of time-optimal pipelining of multi-buffer image acquisition, multi-core computation, intra-MCUs data exchange, and Wi-Fi streaming, leading to sub-optimal control performances. Our NanoCockpit framework aims to fill this gap, increasing the throughput and minimizing the system's latency, while simplifying the developer experience through coroutine-based multi-tasking. In-field experiments on three real-world TinyML nanorobotics applications show our framework achieves ideal end-to-end latency, i.e. zero overhead due to serialized tasks, delivering quantifiable improvements in closed-loop control performance (-30% mean position error, mission success rate increased from 40% to 100%).

**arXiv ID:** 2601.07476
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (40 papers)</h2></summary>

<details>
<summary><strong>Self-Guided Plan Extraction for Instruction-Following Tasks with Goal-Conditional Reinforcement Learning</strong> - Zoya Volovikova, Nikita Sorokin, Dmitriy Lukashevskiy, Aleksandr Panov, Alexey Skrynnik - [[pdf]](https://arxiv.org/pdf/2604.20601)</summary>

**Abstract:** We introduce SuperIgor, a framework for instruction-following tasks. Unlike prior methods that rely on predefined subtasks, SuperIgor enables a language model to generate and refine high-level plans through a self-learning mechanism, reducing the need for manual dataset annotation. Our approach involves iterative co-training: an RL agent is trained to follow the generated plans, while the language model adapts and modifies these plans based on RL feedback and preferences. This creates a feedback loop where both the agent and the planner improve jointly. We validate our framework in environments with rich dynamics and stochasticity. Results show that SuperIgor agents adhere to instructions more strictly than baseline methods, while also demonstrating strong generalization to previously unseen instructions.

**arXiv ID:** 2604.20601
</details>

<details>
<summary><strong>AutoGraph-R1: End-to-End Reinforcement Learning for Knowledge Graph Construction</strong> - Hong Ting Tsang, Jiaxin Bai, Haoyu Huang, Qiao Xiao, Tianshi Zheng, Baixuan Xu, Shujie Liu, Yangqiu Song - [[pdf]](https://arxiv.org/pdf/2510.15339)</summary>

**Abstract:** Building effective knowledge graphs (KGs) for Retrieval-Augmented Generation (RAG) is pivotal for advancing question answering (QA) systems. However, its effectiveness is hindered by a fundamental disconnect: the knowledge graph (KG) construction process is decoupled from its downstream application, yielding suboptimal graph structures. To bridge this gap, we introduce AutoGraph-R1, the first framework to directly optimize KG construction for task performance using Reinforcement Learning (RL). AutoGraph-R1 trains an LLM constructor by framing graph generation as a policy learning problem, where the reward is derived from the graph's functional utility in a RAG pipeline. We design two novel, task-aware reward functions, one for graphs as knowledge carriers and another as knowledge indices. Across multiple QA benchmarks, AutoGraph-R1 consistently enables graph RAG methods to achieve significant performance gains over using task-agnostic baseline graphs. Our work shows it is possible to close the loop between construction and application, shifting the paradigm from building intrinsically ``good'' graphs to building demonstrably ``useful'' ones.

**arXiv ID:** 2510.15339
</details>

<details>
<summary><strong>Environmental Understanding Vision-Language Model for Embodied Agent</strong> - Jinsik Bang, Jaeyeon Bae, Donggyu Lee, Siyeol Jung, Taehwan Kim - [[pdf]](https://arxiv.org/pdf/2604.19839)</summary>

**Abstract:** Vision-language models (VLMs) have shown strong perception and reasoning abilities for instruction-following embodied agents. However, despite these abilities and their generalization performance, they still face limitations in environmental understanding, often failing on interactions or relying on environment metadata during execution. To address this challenge, we propose a novel framework named Environmental Understanding Embodied Agent (EUEA), which fine-tunes four core skills: 1) object perception for identifying relevant objects, 2) task planning for generating interaction subgoals, 3) action understanding for judging success likelihood, and 4) goal recognition for determining goal completion. By fine-tuning VLMs with EUEA skills, our framework enables more reliable task execution for instruction-following. We further introduce a recovery step that leverages these core skills and a group relative policy optimization (GRPO) stage that refines inconsistent skill predictions. The recovery step samples alternative actions to correct failure cases, and the GRPO stage refines inconsistent skill predictions. Across ALFRED tasks, our VLM significantly outperforms a behavior-cloning baseline, achieving an 8.86% improvement in average success rate. The recovery and GRPO stages provide an additional 3.03% gain, further enhancing overall performance. Finally, our skill-level analyses reveal key limitations in the environmental understanding of closed- and open-source VLMs and identify the capabilities necessary for effective agent-environment interaction.

**arXiv ID:** 2604.19839
</details>

<details>
<summary><strong>DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data</strong> - Venus Team, Sunhao Dai, Yong Deng, Jinzhen Lin, Yusheng Song, Guoqing Wang, Xiaofeng Wu, Yuqi Zhou, Shuo Yang, Zhenzhe Ying, Zhanwei Zhang, Changhua Meng, Weiqiang Wang - [[pdf]](https://arxiv.org/pdf/2604.19859)</summary>

**Abstract:** Edge-scale deep research agents based on small language models are attractive for real-world deployment due to their advantages in cost, latency, and privacy. In this work, we study how to train a strong small deep research agent under limited open-data by improving both data quality and data utilization. We present DR-Venus, a frontier 4B deep research agent for edge-scale deployment, built entirely on open data. Our training recipe consists of two stages. In the first stage, we use agentic supervised fine-tuning (SFT) to establish basic agentic capability, combining strict data cleaning with resampling of long-horizon trajectories to improve data quality and utilization. In the second stage, we apply agentic reinforcement learning (RL) to further improve execution reliability on long-horizon deep research tasks. To make RL effective for small agents in this setting, we build on IGPO and design turn-level rewards based on information gain and format-aware regularization, thereby enhancing supervision density and turn-level credit assignment. Built entirely on roughly 10K open-data, DR-Venus-4B significantly outperforms prior agentic models under 9B parameters on multiple deep research benchmarks, while also narrowing the gap to much larger 30B-class systems. Our further analysis shows that 4B agents already possess surprisingly strong performance potential, highlighting both the deployment promise of small models and the value of test-time scaling in this setting. We release our models, code, and key recipes to support reproducible research on edge-scale deep research agents.

**arXiv ID:** 2604.19859
</details>

<details>
<summary><strong>Semantic Prompting: Agentic Incremental Narrative Refinement through Spatial Semantic Interaction</strong> - Xuxin Tang, Ibrahim Tahmid, Eric Krokos, Kirsten Whitley, Xuan Wang, Chris North - [[pdf]](https://arxiv.org/pdf/2604.19971)</summary>

**Abstract:** Interactive spatial layouts empower users to synthesize information and organize findings for sensemaking. While Large Language Models (LLMs) can automate narrative generation from spatial layouts, current collage-based and re-generation methods struggle to support the incremental spatial refinements inherent to the sensemaking process. We identify three critical gaps in existing spatial-textual generation: interaction-revision misalignment, human-LLM intent misalignment, and lack of granular customization. To address these, we introduce Semantic Prompting, a framework for spatial refinement that perceives semantic interactions, reasons about refinement intent, and performs targeted positional revisions. We implemented S-PRISM to realize this framework. The empirical evaluation demonstrated that S-PRISM effectively enhanced the precision of interaction-revision refinement. A user study ($N=14$) highlighted how participants leveraged S-PRISM for incremental formalization through interactive steering. Results showed that users valued its efficient, adaptable, and trustworthy support, which effectively strengthens human-LLM intent alignment.

**arXiv ID:** 2604.19971
</details>

<details>
<summary><strong>AgentSOC: A Multi-Layer Agentic AI Framework for Security Operations Automation</strong> - Joyjit Roy, Samaresh Kumar Singh - [[pdf]](https://arxiv.org/pdf/2604.20134)</summary>

**Abstract:** Security Operations Centers (SOCs) increasingly encounter difficulties in correlating heterogeneous alerts, interpreting multi-stage attack progressions, and selecting safe and effective response actions. This study introduces AgentSOC, a multi-layered agentic AI framework that enhances SOC automation by integrating perception, anticipatory reasoning, and risk-based action planning. The proposed architecture consolidates several layers of abstraction to provide a single operational loop to support normalizing alerts, enriching context, generating hypotheses, validating structural feasibility, and executing policy-compliant responses. Conceptually evaluated within a large enterprise environment, AgentSOC improves triage consistency, anticipates attackers' intentions, and provides recommended containment options that are both operationally feasible and well-balanced between security efficacy and operational impact. The results suggest that hybrid agentic reasoning has the potential to serve as a foundation for developing adaptive, safer SOC automation in large enterprises. Additionally, a minimal Proof-Of-Concept (POC) demonstration using LANL authentication data demonstrated the feasibility of the proposed architecture.

**arXiv ID:** 2604.20134
</details>

<details>
<summary><strong>AgentLens: Adaptive Visual Modalities for Human-Agent Interaction in Mobile GUI Agents</strong> - Jeonghyeon Kim, Byeongjun Joung, Junwon Lee, Joohyung Lee, Taehoon Min, Sunjae Lee - [[pdf]](https://arxiv.org/pdf/2604.20279)</summary>

**Abstract:** Mobile GUI agents can automate smartphone tasks by interacting directly with app interfaces, but how they should communicate with users during execution remains underexplored. Existing systems rely on two extremes: foreground execution, which maximizes transparency but prevents multitasking, and background execution, which supports multitasking but provides little visual awareness. Through iterative formative studies, we found that users prefer a hybrid model with just-in-time visual interaction, but the most effective visualization modality depends on the task. Motivated by this, we present AgentLens, a mobile GUI agent that adaptively uses three visual modalities during human-agent interaction: Full UI, Partial UI, and GenUI. AgentLens extends a standard mobile agent with adaptive communication actions and uses Virtual Display to enable background execution with selective visual overlays. In a controlled study with 21 participants, AgentLens was preferred by 85.7% of participants and achieved the highest usability (1.94 Overall PSSUQ) and adoption-intent (6.43/7).

**arXiv ID:** 2604.20279
</details>

<details>
<summary><strong>PersonalHomeBench: Evaluating Agents in Personalized Smart Homes</strong> - Nikhil Verma, InJung Yang, Sungil Kim, KoKeun Kim, YoungJoon Kim, Manasa Bharadwaj, Yolanda Liu, Kevin Ferreira - [[pdf]](https://arxiv.org/pdf/2604.16813)</summary>

**Abstract:** Agentic AI systems are rapidly advancing toward real-world applications, yet their readiness in complex and personalized environments remains insufficiently characterized. To address this gap, we introduce PersonalHomeBench, a benchmark for evaluating foundation models as agentic assistants in personalized smart home environments. The benchmark is constructed through an iterative process that progressively builds rich household states, which are then used to generate personalized, context-dependent tasks. To support realistic agent-environment interaction, we provide PersonalHomeTools, a comprehensive toolbox enabling household information retrieval, appliance control, and situational understanding. PersonalHomeBench evaluates both reactive and proactive agentic abilities under unimodal and multimodal observations. Thorough experimentation reveals a systematic performance reduction as task complexity increases, with pronounced failures in counterfactual reasoning and under partial observability, where effective tool-based information gathering is required. These results position PersonalHomeBench as a rigorous evaluation platform for analyzing the robustness and limitations of personalized agentic reasoning and planning.

**arXiv ID:** 2604.16813
</details>

<details>
<summary><strong>CoSearch: Joint Training of Reasoning and Document Ranking via Reinforcement Learning for Agentic Search</strong> - Hansi Zeng, Liam Collins, Bhuvesh Kumar, Neil Shah, Hamed Zamani - [[pdf]](https://arxiv.org/pdf/2604.17555)</summary>

**Abstract:** Agentic search -- the task of training agents that iteratively reason, issue queries, and synthesize retrieved information to answer complex questions -- has achieved remarkable progress through reinforcement learning (RL). However, existing approaches such as Search-R1, treat the retrieval system as a fixed tool, optimizing only the reasoning agent while the retrieval component remains unchanged. A preliminary experiment reveals that the gap between an oracle and a fixed retrieval system reaches up to +26.8% relative F1 improvement across seven QA benchmarks, suggesting that the retrieval system is a key bottleneck in scaling agentic search performance. Motivated by this finding, we propose CoSearch, a framework that jointly trains a multi-step reasoning agent and a generative document ranking model via Group Relative Policy Optimization (GRPO). To enable effective GRPO training for the ranker -- whose inputs vary across reasoning trajectories -- we introduce a semantic grouping strategy that clusters sub-queries by token-level similarity, forming valid optimization groups without additional rollouts. We further design a composite reward combining ranking quality signals with trajectory-level outcome feedback, providing the ranker with both immediate and long-term learning signals. Experiments on seven single-hop and multi-hop QA benchmarks demonstrate consistent improvements over strong baselines, with ablation studies validating each design choice. Our results show that joint training of the reasoning agent and retrieval system is both feasible and strongly performant, pointing to a key ingredient for future search agents.

**arXiv ID:** 2604.17555
</details>

<details>
<summary><strong>LiteResearcher: A Scalable Agentic RL Training Framework for Deep Research Agent</strong> - Wanli Li, Bince Qu, Bo Pan, Jianyu Zhang, Zheng Liu, Pan Zhang, Wei Chen, Bo Zhang - [[pdf]](https://arxiv.org/pdf/2604.17931)</summary>

**Abstract:** Reinforcement Learning (RL) has emerged as a powerful training paradigm for LLM-based agents. However, scaling agentic RL for deep research remains constrained by two coupled challenges: hand-crafted synthetic data fails to elicit genuine real-world search capabilities, and real-world search dependency during RL training introduces instability and prohibitive cost, which limits the scalability of Agentic RL. LiteResearcher is a training framework that makes Agentic RL scalable: by constructing a lite virtual world that mirrors real-world search dynamics, we enable a continuously improving training recipe that empowers a tiny search agent to outperform large-scale open-source and commercial models (e.g., Tongyi DeepResearch and Claude-4.5 Sonnet). Specifically, on common benchmarks such as GAIA and Xbench, our LiteResearcher-4B achieves open-source state-of-the-art results of 71.3% and 78.0% respectively, demonstrating that scalable RL training is a key enabler for Deep Research Agents.

**arXiv ID:** 2604.17931
</details>

<details>
<summary><strong>Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning</strong> - Yixuan Even Xu, Yash Savani, Fei Fang, J. Zico Kolter - [[pdf]](https://arxiv.org/pdf/2504.13818)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has emerged as the leading approach for enhancing reasoning capabilities in large language models. However, it faces a fundamental compute and memory asymmetry: rollout generation is embarrassingly parallel and memory-light, whereas policy updates are communication-heavy and memory-intensive. To address this, we introduce PODS (Policy Optimization with Down-Sampling), which decouples rollout generation from policy updates by training only on a strategically selected subset of rollouts, maintaining learning quality while dramatically reducing update costs. We propose a principled subset selection criterion, max-variance down-sampling, that maximizes reward diversity, and provide an efficient $O(n\log n)$ implementation. Empirically, Group Relative Policy Optimization (GRPO) with PODS achieves the peak test accuracy of vanilla GRPO at least $\mathbf{1.7\times}$ faster across the different reasoning benchmarks and hardware configurations we tested.

**arXiv ID:** 2504.13818
</details>

<details>
<summary><strong>Hybrid-AIRL: Enhancing Inverse Reinforcement Learning with Supervised Expert Guidance</strong> - Bram Silue, Santiago Amaya-Corredor, Patrick Mannion, Lander Willem, Pieter Libin - [[pdf]](https://arxiv.org/pdf/2511.21356)</summary>

**Abstract:** Adversarial Inverse Reinforcement Learning (AIRL) has shown promise in addressing the sparse reward problem in reinforcement learning (RL) by inferring dense reward functions from expert demonstrations. However, its performance in highly complex, imperfect-information settings remains largely unexplored. To explore this gap, we evaluate AIRL in the context of Heads-Up Limit Hold'em (HULHE) poker, a domain characterized by sparse, delayed rewards and significant uncertainty. In this setting, we find that AIRL struggles to infer a sufficiently informative reward function. To overcome this limitation, we contribute Hybrid-AIRL (H-AIRL), an extension that enhances reward inference and policy learning by incorporating a supervised loss derived from expert data and a stochastic regularization mechanism. We evaluate H-AIRL on a carefully selected set of Gymnasium benchmarks and the HULHE poker setting. Additionally, we analyze the learned reward function through visualization to gain deeper insights into the learning process. Our experimental results show that H-AIRL achieves higher sample efficiency and more stable learning compared to AIRL. This highlights the benefits of incorporating supervised signals into inverse RL and establishes H-AIRL as a promising framework for tackling challenging, real-world settings.

**arXiv ID:** 2511.21356
</details>

<details>
<summary><strong>To Know is to Construct: Schema-Constrained Generation for Agent Memory</strong> - Lei Zheng, Weinan Song, Daili Li, Yanming Yang - [[pdf]](https://arxiv.org/pdf/2604.20117)</summary>

**Abstract:** Constructivist epistemology argues that knowledge is actively constructed rather than passively copied. Despite the generative nature of Large Language Models (LLMs), most existing agent memory systems are still based on dense retrieval. However, dense retrieval heavily relies on semantic overlap or entity matching within sentences. Consequently, embeddings often fail to distinguish instances that are semantically similar but contextually distinct, introducing substantial noise by retrieving context-mismatched entries. Conversely, directly employing open-ended generation for memory access risks "Structural Hallucination" where the model generates memory keys that do not exist in the memory, leading to lookup failures. Inspired by this epistemology, we posit that memory is fundamentally organized by cognitive schemas, and valid recall must be a generative process performed within these schematic structures. To realize this, we propose SCG-MEM, a schema-constrained generative memory architecture. SCG-MEM reformulates memory access as Schema-Constrained Generation. By maintaining a dynamic Cognitive Schema, we strictly constrain LLM decoding to generate only valid memory entry keys, providing a formal guarantee against structural hallucinations. To support long-term adaptation, we model memory updates via assimilation (grounding inputs into existing schemas) and accommodation (expanding schemas with novel concepts). Furthermore, we construct an Associative Graph to enable multi-hop reasoning through activation propagation. Experiments on the LoCoMo benchmark show that SCG-MEM substantially improves performance across all categories over retrieval-based baselines.

**arXiv ID:** 2604.20117
</details>

<details>
<summary><strong>Chasing the Public Score: User Pressure and Evaluation Exploitation in Coding Agent Workflows</strong> - Hardy Chen, Nancy Lau, Haoqin Tu, Shuo Yan, Xiangyan Liu, Zijun Wang, Juncheng Wu, Michael Qizhe Shieh, Alvaro A. Cardenas, Cihang Xie, Yuyin Zhou - [[pdf]](https://arxiv.org/pdf/2604.20200)</summary>

**Abstract:** Frontier coding agents are increasingly used in workflows where users supervise progress primarily through repeated improvement of a public score, namely the reported score on a public evaluation file with labels in the workspace, rather than through direct inspection of the agent's intermediate outputs. We study whether multi-round user pressure to improve that score induces public score exploitation: behavior that raises the public score through shortcuts without improving hidden private evaluation. We begin with a preliminary single-script tabular classification task, where GPT-5.4 and Claude Opus 4.6 both exploit label information within 10 rounds of user-agent interaction. We then build AgentPressureBench, a 34-task machine-learning repository benchmark spanning three input modalities, and collect 1326 multi-round trajectories from 13 coding agents. On our benchmark, we observe 403 exploitative runs, spanning across all tasks. We also find that stronger models have higher exploitation rates, supported by a significant Spearman rank correlation of 0.77. Our ablation experiments show that higher user pressure leads to earlier exploitation, reducing the average first exploit round by 15.6 rounds (i.e., 19.67 to 4.08). As a mitigation, adding explicit anti-exploit wordings in prompt mostly eliminates exploitation (100% to 8.3%). We hope that our work can bring attention to more careful use of coding agents workflow, and developing more robust coding agents under user pressure. Our project page is at this https URL .

**arXiv ID:** 2604.20200
</details>

<details>
<summary><strong>RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings</strong> - Wei Han, David Martinez, Anna Khanina, Lawrence Cavedon, Karin Verspoor - [[pdf]](https://arxiv.org/pdf/2604.20256)</summary>

**Abstract:** A common strategy in transfer learning is few shot fine-tuning, but its success is highly dependent on the quality of samples selected as training examples. Active learning methods such as uncertainty sampling and diversity sampling can select useful samples. However, under extremely low-resource and class-imbalanced conditions, they often favor outliers rather than truly informative samples, resulting in degraded performance. In this paper, we introduce RADS (Reinforcement Adaptive Domain Sampling), a robust sample selection strategy using reinforcement learning (RL) to identify the most informative samples. Experimental evaluations on several real world clinical datasets show our sample selection strategy enhances model transferability while maintaining robust performance under extreme class imbalance compared to traditional methods.

**arXiv ID:** 2604.20256
</details>

<details>
<summary><strong>WebGen-R1: Incentivizing Large Language Models to Generate Functional and Aesthetic Websites with Reinforcement Learning</strong> - Juyong Jiang, Chenglin Cai, Chansung Park, Jiasi Shen, Sunghun Kim, Jianguo Li, Yue Wang - [[pdf]](https://arxiv.org/pdf/2604.20398)</summary>

**Abstract:** While Large Language Models (LLMs) excel at function-level code generation, project-level tasks such as generating functional and visually aesthetic multi-page websites remain highly challenging. Existing works are often limited to single-page static websites, while agentic frameworks typically rely on multi-turn execution with proprietary models, leading to substantial token costs, high latency, and brittle integration. Training a small LLM end-to-end with reinforcement learning (RL) is a promising alternative, yet it faces a critical bottleneck in designing reliable and computationally feasible rewards for website generation. Unlike single-file coding tasks that can be verified by unit tests, website generation requires evaluating inherently subjective aesthetics, cross-page interactions, and functional correctness. To this end, we propose WebGen-R1, an end-to-end RL framework tailored for project-level website generation. We first introduce a scaffold-driven structured generation paradigm that constrains the large open-ended action space and preserves architectural integrity. We then design a novel cascaded multimodal reward that seamlessly couples structural guarantees with execution-grounded functional feedback and vision-based aesthetic supervision. Extensive experiments demonstrate that our WebGen-R1 substantially transforms a 7B base model from generating nearly nonfunctional websites into producing deployable, aesthetically aligned multi-page websites. Remarkably, our WebGen-R1 not only consistently outperforms heavily scaled open-source models (up to 72B), but also rivals the state-of-the-art DeepSeek-R1 (671B) in functional success, while substantially exceeding it in valid rendering and aesthetic alignment. These results position WebGen-R1 as a viable path for scaling small open models from function-level code generation to project-level web application generation.

**arXiv ID:** 2604.20398
</details>

<details>
<summary><strong>Ask Only When Needed: Proactive Retrieval from Memory and Skills for Experience-Driven Lifelong Agents</strong> - Yuxuan Cai, Jie Zhou, Qin Chen, Liang He - [[pdf]](https://arxiv.org/pdf/2604.20572)</summary>

**Abstract:** Online lifelong learning enables agents to accumulate experience across interactions and continually improve on long-horizon tasks. However, existing methods typically treat retrieval from past experience as a passive operation, triggering it only at task initialization or after completing a step. Consequently, agents often fail to identify knowledge gaps during interaction and proactively retrieve the most useful experience for the current decision. To address this limitation, we present ProactAgent, an experience-driven lifelong learning framework for proactive retrieval over a structured experience base. We first introduce Experience-Enhanced Online Evolution (ExpOnEvo), which enables continual improvement through both policy updates and memory refinement. The experience base organizes historical interactions into typed repositories, including factual memory, episodic memory, and behavioral skills, so that retrieval can provide both relevant evidence and actionable guidance. On top of this, we propose Proactive Reinforcement Learning-based Retrieval (ProactRL), which models retrieval as an explicit policy action and learns when and what to retrieve via paired-branch process rewards. By comparing continuations from identical interaction prefixes with and without retrieval, ProactRL provides step-level supervision for retrieval decisions, encouraging retrieval only when it leads to better task outcomes or higher efficiency. Experiments on SciWorld, AlfWorld, and StuLife show that ProactAgent consistently improves lifelong agent performance, achieving success rates of 73.50\% on SciWorld and 71.28\% on AlfWorld while substantially reducing retrieval overhead, and attains performance competitive with proprietary models on StuLife.

**arXiv ID:** 2604.20572
</details>

<details>
<summary><strong>MOA: Multi-Objective Alignment for Role-Playing Agents</strong> - Chonghua Liao, Ke Wang, Yuchuan Wu, Ruoran Li, Fei Huang, Yongbin Li - [[pdf]](https://arxiv.org/pdf/2512.09756)</summary>

**Abstract:** Role-playing agents (RPAs) require balancing multiple objectives, such as instruction following, persona consistency, and stylistic fidelity, which are not always perfectly aligned across different dimensions. While prior work has primarily relied on supervised fine-tuning or reinforcement learning with scalarized rewards, these approaches do not explicitly address the coordination of multiple reward dimensions during optimization. We present \textbf{MOA} (\textbf{M}ulti-\textbf{O}bjective \textbf{A}lignment), a reinforcement-learning framework that enables multi-dimensional, fine-grained rubric optimization for general RPAs. MOA introduces a novel multi-objective optimization strategy that trains simultaneously on multiple fine-grained rubrics to boost optimization performance. Additionally, to improve both output diversity and generation quality, we employ thought-augmented rollouts with off-policy guidance. Experiments on PersonaGym and RoleMRC show that MOA consistently improves multi-dimensional role-playing performance over supervised and standard RL baselines. Under identical evaluation protocols, an 8B model trained with MOA reaches performance competitive with strong closed-source models across multiple evaluation dimensions. These results suggest that MOA provides a practical framework for training more capable general-purpose role-playing agents.

**arXiv ID:** 2512.09756
</details>

<details>
<summary><strong>Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning</strong> - Weiqin Wang, Yile Wang, Kehao Chen, Hui Huang - [[pdf]](https://arxiv.org/pdf/2512.15146)</summary>

**Abstract:** Test-time reinforcement learning mitigates the reliance on annotated data by using majority voting results as pseudo-labels, emerging as a complementary direction to reinforcement learning with verifiable rewards (RLVR) for improving reasoning ability of large language models (LLMs). However, this voting strategy often induces confirmation bias and suffers from sparse rewards, limiting the overall performance. In this work, we propose subgroup-specific step-wise confidence-weighted pseudo-label estimation (SCOPE), a framework integrating model confidence and dynamic subgroup partitioning to address these issues. Specifically, SCOPE integrates the proposed step-wise confidence into pseudo label estimation, prioritizing high-quality reasoning paths over simple frequency count. Furthermore, it dynamically partitions the candidate outputs pool into independent subgroups by balancing reasoning quality against exploration diversity. By deriving local consensus via repeat sampling for each sub group, SCOPE provides diverse supervision targets to encourage broader exploration. We conduct experiments across various models and benchmarks, experimental results show that SCOPE consistently outperforms recent baselines. Notably, SCOPE achieving relative improvements of 13.1% on challenging AIME 2025 and 8.1% on AMC. The code is released at this https URL.

**arXiv ID:** 2512.15146
</details>

<details>
<summary><strong>Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation</strong> - Rui Qi, Fengran Mo, Yufeng Chen, Xue Zhang, Shuo Wang, Hongliang Li, Jinan Xu, Meng Jiang, Jian-Yun Nie, Kaiyu Huang - [[pdf]](https://arxiv.org/pdf/2601.14896)</summary>

**Abstract:** Multilingual retrieval-augmented generation (MRAG) requires models to effectively acquire and integrate beneficial external knowledge from multilingual collections. However, most existing studies employ a unitive process where queries of equivalent semantics across different languages are processed through a single-turn retrieval and subsequent optimization. Such a ``one-size-fits-all'' strategy is often suboptimal in multilingual settings, as the models occur to knowledge bias and conflict during the interaction with the search engine. To alleviate the issues, we propose LcRL, a multilingual search-augmented reinforcement learning framework that integrates a language-coupled Group Relative Policy Optimization into the policy and reward models. We adopt the language-coupled group sampling in the rollout module to reduce knowledge bias, and regularize an auxiliary anti-consistency penalty in the reward models to mitigate the knowledge conflict. Experimental results demonstrate that LcRL not only achieves competitive performance but is also appropriate for various practical scenarios such as constrained training data and retrieval over collections encompassing a large number of languages. Our code is available at this https URL.

**arXiv ID:** 2601.14896
</details>

<details>
<summary><strong>Trajectory2Task: Training Robust Tool-Calling Agents with Synthesized Yet Verifiable Data for Complex User Intents</strong> - Ziyi Wang, Yuxuan Lu, Yimeng Zhang, Pei Chen, Ziwei Dong, Jing Huang, Jiri Gesi, Xianfeng Tang, Chen Luo, Qun Liu, Yisi Sang, Hanqing Lu, Manling Li, Jin Lai, Dakuo Wang - [[pdf]](https://arxiv.org/pdf/2601.20144)</summary>

**Abstract:** Tool-calling agents are increasingly deployed in real-world customer-facing workflows. Yet most studies on tool-calling agents focus on idealized settings with general, fixed, and well-specified tasks. In real-world applications, user requests are often (1) ambiguous, (2) changing over time, or (3) infeasible due to policy constraints, and training and evaluation data that cover these diverse, complex interaction patterns remain under-represented. To bridge the gap, we present Trajectory2Task, a verifiable data generation pipeline for studying tool use at scale under three realistic user scenarios: ambiguous intent, changing intent, and infeasible intents. The pipeline first conducts multi-turn exploration to produce valid tool-call trajectories. It then converts these trajectories into user-facing tasks with controlled intent adaptations. This process yields verifiable task that support closed-loop evaluation and training. We benchmark seven state-of-the-art LLMs on the generated complex user scenario tasks and observe frequent failures. Finally, using successful trajectories obtained from task rollouts, we fine-tune lightweight LLMs and find consistent improvements across all three conditions, along with better generalization to unseen tool-use domains, indicating stronger tool-calling ability.

**arXiv ID:** 2601.20144
</details>

<details>
<summary><strong>Composition-RL: Compose Your Verifiable Prompts for Reinforcement Learning of Large Language Models</strong> - Xin Xu, Clive Bai, Kai Yang, Tianhao Chen, Yangkun Chen, Weijie Liu, Hao Chen, Yang Wang, Saiyong Yang, Can Yang - [[pdf]](https://arxiv.org/pdf/2602.12036)</summary>

**Abstract:** Large-scale verifiable prompts underpin the success of Reinforcement Learning with Verifiable Rewards (RLVR), but they contain many uninformative examples and are costly to expand further. Recent studies focus on better exploiting limited training data by prioritizing hard prompts whose rollout pass rate is 0. However, easy prompts with a pass rate of 1 also become increasingly prevalent as training progresses, thereby reducing the effective data size. To mitigate this, we propose Composition-RL, a simple yet useful approach for better utilizing limited verifiable prompts targeting pass-rate-1 prompts. More specifically, Composition-RL automatically composes multiple problems into a new verifiable question and uses these compositional prompts for RL training. Extensive experiments across model sizes from 4B to 30B show that Composition-RL consistently improves reasoning capability over RL trained on the original dataset. Performance can be further boosted with a curriculum variant of Composition-RL that gradually increases compositional depth over training. Additionally, Composition-RL enables more effective cross-domain RL by composing prompts drawn from different domains. Codes, datasets, and models are available at this https URL.

**arXiv ID:** 2602.12036
</details>

<details>
<summary><strong>Multi-Objective Reinforcement Learning for Generating Covalent Inhibitor Candidates</strong> - Renee Gil - [[pdf]](https://arxiv.org/pdf/2604.20019)</summary>

**Abstract:** Rational design of covalent inhibitors requires simultaneously optimizing multiple properties, such as binding affinity, target selectivity, or electrophilic reactivity. This presents a multi-objective problem not easily addressed by screening alone. Here we present a machine learning pipeline for generating covalent inhibitor candidates using multi-objective reinforcement learning (RL), applied to two targets: epidermal growth factor receptor (EGFR) and acetylcholinesterase (ACHE). A SMILES-based pretrained LSTM serves as the generative model, optimized via policy gradient RL with Pareto crowding distance to balance competing scoring functions including synthetic accessibility, predicted covalent activity, residue affinity, and an approximated docking score. The pipeline rediscovers known covalent inhibitors at rates of up to 0.50% (EGFR) and 0.74% (ACHE) in 10,000-structure runs, with candidate structures achieving warhead-to-residue distances as short as 5.5 angstrom (EGFR) and 3.2 angstrom (ACHE) after further docking-based screening. More notably, the pipeline spontaneously generates structures bearing warhead motifs absent from the training data - including allenes, 3-oxo-$\beta$-sultams, and $\alpha$-methylene-$\beta$-lactones - all of which have independent literature support as covalent warheads. These results suggest that RL-guided generation can explore covalent chemical space beyond its training distribution, and may be useful as a tool for medicinal chemists working on covalent drug discovery.

**arXiv ID:** 2604.20019
</details>

<details>
<summary><strong>Maximum Entropy Semi-Supervised Inverse Reinforcement Learning</strong> - Julien Audiffren, Michal Valko, Alessandro Lazaric, Mohammad Ghavamzadeh - [[pdf]](https://arxiv.org/pdf/2604.20074)</summary>

**Abstract:** A popular approach to apprenticeship learning (AL) is to formulate it as an inverse reinforcement learning (IRL) problem. The MaxEnt-IRL algorithm successfully integrates the maximum entropy principle into IRL and unlike its predecessors, it resolves the ambiguity arising from the fact that a possibly large number of policies could match the expert's behavior. In this paper, we study an AL setting in which in addition to the expert's trajectories, a number of unsupervised trajectories is available. We introduce MESSI, a novel algorithm that combines MaxEnt-IRL with principles coming from semi-supervised learning. In particular, MESSI integrates the unsupervised data into the MaxEnt-IRL framework using a pairwise penalty on trajectories. Empirical results in a highway driving and grid-world problems indicate that MESSI is able to take advantage of the unsupervised trajectories and improve the performance of MaxEnt-IRL.

**arXiv ID:** 2604.20074
</details>

<details>
<summary><strong>Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning</strong> - Aravind Venugopal, Jiayu Chen, Xudong Wu, Chongyi Zheng, Benjamin Eysenbach, Jeff Schneider - [[pdf]](https://arxiv.org/pdf/2604.20627)</summary>

**Abstract:** The temporal lag between actions and their long-term consequences makes credit assignment a challenge when learning goal-directed behaviors from data. Generative world models capture the distribution of future states an agent may visit, indicating that they have captured temporal information. How can that temporal information be extracted to perform credit assignment? In this paper, we formalize how the temporal information stored in world models encodes the underlying geometry of the world. Leveraging optimal transport, we extract this geometry from a learned model of the occupancy measure into a reward function that captures goal-reaching information. Our resulting method, Occupancy Reward Shaping, largely mitigates the problem of credit assignment in sparse reward settings. ORS provably does not alter the optimal policy, yet empirically improves performance by 2.2x across 13 diverse long-horizon locomotion and manipulation tasks. Moreover, we demonstrate the effectiveness of ORS in the real world for controlling nuclear fusion on 3 Tokamak control tasks.
Code: this https URL Website: this https URL

**arXiv ID:** 2604.20627
</details>

<details>
<summary><strong>Toward Safe Autonomous Robotic Endovascular Interventions using World Models</strong> - Harry Robertshaw, Nikola Fischer, Han-Ru Wu, Andrea Walker Perez, Weiyuan Deng, Benjamin Jackson, Christos Bergeles, Alejandro Granados, Thomas C Booth - [[pdf]](https://arxiv.org/pdf/2604.20151)</summary>

**Abstract:** Autonomous mechanical thrombectomy (MT) presents substantial challenges due to highly variable vascular geometries and the requirements for accurate, real-time control. While reinforcement learning (RL) has emerged as a promising paradigm for the automation of endovascular navigation, existing approaches often show limited robustness when faced with diverse patient anatomies or extended navigation horizons. In this work, we investigate a world-model-based framework for autonomous endovascular navigation built on TD-MPC2, a model-based RL method that integrates planning and learned dynamics. We evaluate a TD-MPC2 agent trained on multiple navigation tasks across hold out patient-specific vasculatures and benchmark its performance against the state-of-the-art Soft Actor-Critic (SAC) algorithm agent. Both approaches are further validated in vitro using patient-specific vascular phantoms under fluoroscopic guidance. In simulation, TD-MPC2 demonstrates a significantly higher mean success rate than SAC (58% vs. 36%, p < 0.001), and mean tip contact forces of 0.15 N, well below the proposed 1.5 N vessel rupture threshold. Mean success rates for TD-MPC2 (68%) were comparable to SAC (60%) in vitro, but TD-MPC2 achieved superior path ratios (p = 0.017) at the cost of longer procedure times (p < 0.001). Together, these results provide the first demonstration of autonomous MT navigation validated across both hold out in silico data and fluoroscopy-guided in vitro experiments, highlighting the promise of world models for safe and generalizable AI-assisted endovascular interventions.

**arXiv ID:** 2604.20151
</details>

<details>
<summary><strong>Issues with Value-Based Multi-objective Reinforcement Learning: Value Function Interference and Overestimation Sensitivity</strong> - Peter Vamplew, Ethan, Watkins, Cameron Foale, Richard Dazeley - [[pdf]](https://arxiv.org/pdf/2402.06266)</summary>

**Abstract:** Multi-objective reinforcement learning (MORL) algorithms extend conventional reinforcement learning (RL) to the more general case of problems with multiple, conflicting objectives, represented by vector-valued rewards. Widely-used scalar RL methods such as Q-learning can be modified to handle multiple objectives by (1) learning vector-valued value functions, and (2) performing action selection using a scalarisation or ordering operator which reflects the user's preferences with respect to the different objectives. This paper investigates two previously unreported issues which can hinder the performance of value-based MORL algorithms when applied in conjunction with a non-linear utility function -- value function interference, and sensitivity to overestimation. We illustrate the nature of these phenomena on simple multi-objective MDPs using a tabular implementation of multiobjective Q-learning.

**arXiv ID:** 2402.06266
</details>

<details>
<summary><strong>Kalman Filter Enhanced GRPO for Reinforcement Learning-Based Language Model Reasoning</strong> - Hu Wang, Congbo Ma, Ian Reid, Mohammad Yaqub - [[pdf]](https://arxiv.org/pdf/2505.07527)</summary>

**Abstract:** The advantage function is a central concept in RL that helps reduce variance in policy gradient estimates. For language modeling, Group Relative Policy Optimization (GRPO) was proposed to use the within-group sample mean as a baseline for advantage normalization. This estimator can be sensitive to small group size and rollout-level stochasticity, which may lead to suboptimal advantage estimates in some settings. In this paper, we propose Kalman Filter Enhanced Group Relative Policy Optimization (KRPO), a lightweight variant that treats per-group rewards as noisy observations of a latent prompt-level reward baseline and uses a 1D Kalman filter to estimate both the baseline and its uncertainty. KRPO introduces no additional learned parameters and can be integrated into GRPO with minimal computational overhead. On mathematical reasoning benchmarks, KRPO consistently improves training reward curves and final accuracy over GRPO. These results suggest that adaptive advantage estimation is a promising direction for critic-free reinforcement learning in language model reasoning. The code is available at this https URL.

**arXiv ID:** 2505.07527
</details>

<details>
<summary><strong>EvolveSignal: A Large Language Model Powered Coding Agent for Discovering Traffic Signal Control Strategies</strong> - Leizhen Wang, Peibo Duan, Hao Wang, Yue Wang, Jian Xu, Nan Zheng, Zhenliang Ma - [[pdf]](https://arxiv.org/pdf/2509.03335)</summary>

**Abstract:** In traffic engineering, fixed-time traffic signal control remains widely used for its low cost, stability, and interpretability. However, its design relies on hand-crafted formulas (e.g., Webster) and manual re-timing by engineers to adapt to demand changes, which is labor-intensive and often yields suboptimal results under heterogeneous or congested conditions. This paper introduces EvolveSignal, an LLM-powered coding agent for automatically discovering interpretable heuristic strategies for fixed-time traffic signal control. Rather than deriving entirely new analytical formulations, the proposed framework focuses on exploring code-level variations of existing control logic and identifying effective combinations of heuristic modifications. We formulate the problem as program synthesis, where candidate strategies are represented as Python functions with fixed input-output structures and iteratively optimized through external evaluations (e.g., a traffic simulator) and evolutionary search. Experiments on a signalized intersection demonstrate that the discovered strategies outperform a classical baseline (Webster's method), reducing average delay by 20.1\% and average stops by 47.1\%. Beyond performance, ablation and incremental analyses reveal that EvolveSignal can identify meaningful modifications, such as adjusting cycle length bounds, incorporating right-turn demand, and rescaling green allocations, that provide useful insights for traffic engineers. This work highlights the potential of LLM-driven program synthesis for supporting interpretable and automated heuristic design in traffic signal control.

**arXiv ID:** 2509.03335
</details>

<details>
<summary><strong>Distributional Inverse Reinforcement Learning</strong> - Feiyang Wu, Ye Zhao, Anqi Wu - [[pdf]](https://arxiv.org/pdf/2510.03013)</summary>

**Abstract:** We propose a distributional framework for offline Inverse Reinforcement Learning (IRL) that jointly models uncertainty over reward functions and full distributions of returns. Unlike conventional IRL approaches that recover a deterministic reward estimate or match only expected returns, our method captures richer structure in expert behavior, particularly in learning the reward distribution, by minimizing first-order stochastic dominance (FSD) violations and thus integrating distortion risk measures (DRMs) into policy learning, enabling the recovery of both reward distributions and distribution-aware policies. This formulation is well-suited for behavior analysis and risk-aware imitation learning. Theoretical analysis shows that the algorithm converges with $\mathcal{O}(\varepsilon^{-2})$ iteration complexity. Empirical results on synthetic benchmarks, real-world neurobehavioral data, and MuJoCo control tasks demonstrate that our method recovers expressive reward representations and achieves state-of-the-art performance.

**arXiv ID:** 2510.03013
</details>

<details>
<summary><strong>From Competition to Synergy: Unlocking Reinforcement Learning for Subject-Driven Image Generation</strong> - Ziwei Huang, Ying Shu, Hao Fang, Quanyu Long, Wenya Wang, Qiushi Guo, Tiezheng Ge, Leilei Gan - [[pdf]](https://arxiv.org/pdf/2510.18263)</summary>

**Abstract:** Subject-driven image generation models face a fundamental trade-off between identity preservation (fidelity) and prompt adherence (editability). While online reinforcement learning (RL), specifically GPRO, offers a promising solution, we find that a naive application of GRPO leads to competitive degradation, as the simple linear aggregation of rewards with static weights causes conflicting gradient signals and a misalignment with the temporal dynamics of the diffusion process. To overcome these limitations, we propose Customized-GRPO, a novel framework featuring two key innovations: (i) Synergy-Aware Reward Shaping (SARS), a non-linear mechanism that explicitly penalizes conflicted reward signals and amplifies synergistic ones, providing a sharper and more decisive gradient. (ii) Time-Aware Dynamic Weighting (TDW), which aligns the optimization pressure with the model's temporal dynamics by prioritizing prompt-following in the early, identity preservation in the later. Extensive experiments demonstrate that our method significantly outperforms naive GRPO baselines, successfully mitigating competitive degradation. Our model achieves a superior balance, generating images that both preserve key identity features and accurately adhere to complex textual prompts.

**arXiv ID:** 2510.18263
</details>

<details>
<summary><strong>Bounded Ratio Reinforcement Learning</strong> - Yunke Ao, Le Chen, Bruce D. Lee, Assefa S. Wahd, Aline Czarnobai, Philipp Fürnstahl, Bernhard Schölkopf, Andreas Krause - [[pdf]](https://arxiv.org/pdf/2604.18578)</summary>

**Abstract:** Proximal Policy Optimization (PPO) has become the predominant algorithm for on-policy reinforcement learning due to its scalability and empirical robustness across domains. However, there is a significant disconnect between the underlying foundations of trust region methods and the heuristic clipped objective used in PPO. In this paper, we bridge this gap by introducing the Bounded Ratio Reinforcement Learning (BRRL) framework. We formulate a novel regularized and constrained policy optimization problem and derive its analytical optimal solution. We prove that this solution ensures monotonic performance improvement. To handle parameterized policy classes, we develop a policy optimization algorithm called Bounded Policy Optimization (BPO) that minimizes an advantage-weighted divergence between the policy and the analytic optimal solution from BRRL. We further establish a lower bound on the expected performance of the resulting policy in terms of the BPO loss function. Notably, our framework also provides a new theoretical lens to interpret the success of the PPO loss, and connects trust region policy optimization and the Cross-Entropy Method (CEM). We additionally extend BPO to Group-relative BPO (GBPO) for LLM fine-tuning. Empirical evaluations of BPO across MuJoCo, Atari, and complex IsaacLab environments (e.g., Humanoid locomotion), and of GBPO for LLM fine-tuning tasks, demonstrate that BPO and GBPO generally match or outperform PPO and GRPO in stability and final performance.

**arXiv ID:** 2604.18578
</details>

<details>
<summary><strong>Scalable Quantum Reinforcement Learning on NISQ Devices with Dynamic-Circuit Qubit Reuse and Grover Optimization</strong> - Thet Htar Su, Shaswot Shresthamali, Masaaki Kondo - [[pdf]](https://arxiv.org/pdf/2509.16002)</summary>

**Abstract:** A scalable and resource-efficient quantum reinforcement learning framework is presented that eliminates the linear qubit-scaling barrier in multi-step quantum Markov decision processes (QMDPs). The proposed framework integrates a QMDP formulation, dynamic-circuit execution, and Grover-based amplitude amplification into a unified quantum-native architecture. Environment dynamics are encoded entirely within quantum Hilbert space, enabling coherent superposition over state-action sequences and a direct quantum agent-environment interface without intermediate quantum-to-classical conversion. The central contribution is a dynamic execution model for multi-step QMDPs that employs mid-circuit measurement and reset to recycle a fixed physical quantum register across sequential interactions. This approach preserves trajectory fidelity relative to a static unrolled QMDP, generating identical state-action sequences while reducing the physical qubit requirement from 7xT to a constant 7, independent of the interaction horizon T. Thus, the qubit complexity of multi-step QMDPs is transformed from O(T) to O(1) while maintaining functional equivalence at the level of trajectory generation. Trajectory returns are evaluated via quantum arithmetic, and high-return trajectories are marked and amplified using amplitude amplification to increase their sampling probability. Simulations confirm preservation of trajectory fidelity with a 66% qubit reduction compared to a static design. Experimental execution on an IBM Heron-class processor demonstrates feasibility on noisy intermediate-scale quantum hardware, establishing a scalable and resource-efficient foundation for large-scale quantum-native reinforcement learning.

**arXiv ID:** 2509.16002
</details>

<details>
<summary><strong>Efficient Reinforcement Learning using Linear Koopman Dynamics for Nonlinear Robotic Systems</strong> - Wenjian Hao, Yuxuan Fang, Zehui Lu, Shaoshuai Mou - [[pdf]](https://arxiv.org/pdf/2604.19980)</summary>

**Abstract:** This paper presents a model-based reinforcement learning (RL) framework for optimal closed-loop control of nonlinear robotic systems. The proposed approach learns linear lifted dynamics through Koopman operator theory and integrates the resulting model into an actor-critic architecture for policy optimization, where the policy represents a parameterized closed-loop controller. To reduce computational cost and mitigate model rollout errors, policy gradients are estimated using one-step predictions of the learned dynamics rather than multi-step propagation. This leads to an online mini-batch policy gradient framework that enables policy improvement from streamed interaction data. The proposed framework is evaluated on several simulated nonlinear control benchmarks and two real-world hardware platforms, including a Kinova Gen3 robotic arm and a Unitree Go1 quadruped. Experimental results demonstrate improved sample efficiency over model-free RL baselines, superior control performance relative to model-based RL baselines, and control performance comparable to classical model-based methods that rely on exact system dynamics.

**arXiv ID:** 2604.19980
</details>

<details>
<summary><strong>LLM-Guided Safety Agent for Edge Robotics with an ISO-Compliant Perception-Compute-Control Architecture</strong> - Xu Huang, Ruofan Zhang, Lu Cheng, Yuefeng Song, Xu Huang, Huayu Zhang, Sheng Yin, Anyang Liang, Chen Qian, Yin Zhou, Xiaoyun Yuan, Yuan Cheng - [[pdf]](https://arxiv.org/pdf/2604.20193)</summary>

**Abstract:** Ensuring functional safety in human-robot interaction is challenging because AI perception is inherently probabilistic, whereas industrial standards require deterministic behavior. We present an LLM-guided safety agent for edge robotics, built on an ISO-compliant low-latency perception-compute-control architecture. Our method translates natural-language safety regulations into executable predicates and deploys them through a redundant heterogeneous edge runtime. For fault-tolerant closed-loop execution under edge constraints, we adopt a symmetric dual-modular redundancy design with parallel independent execution for low-latency perception, computation, and control. We prototype the system on a dual-RK3588 platform and evaluate it in representative human-robot interaction scenarios. The results demonstrate a practical edge implementation path toward ISO 13849 Category 3 and PL d using cost-effective hardware, supporting practical deployment of safety-critical embodied AI.

**arXiv ID:** 2604.20193
</details>

<details>
<summary><strong>Online Structure Learning and Planning for Autonomous Robot Navigation using Active Inference</strong> - Daria de tinguy, Tim Verbelen, Emilio Gamba, Bart Dhoedt - [[pdf]](https://arxiv.org/pdf/2510.09574)</summary>

**Abstract:** Autonomous navigation in unfamiliar environments requires robots to simultaneously explore, localise, and plan under uncertainty, without relying on predefined maps or extensive training. We present Active Inference MAPping and Planning (AIMAPP), a framework unifying mapping, localisation, and decision-making within a single generative model, drawing on cognitive-mapping concepts from animal navigation (topological organisation, discrete spatial representations and predictive belief updating) as design inspiration. The agent builds and updates a sparse topological map online, learns state transitions dynamically, and plans actions by minimising Expected Free Energy. This allows it to balance goal-directed and exploratory behaviours. We implemented AIMAPP as a ROS-compatible system that is sensor and robot-agnostic and integrates with diverse hardware configurations. It operates in a fully self-supervised manner, is resilient to sensor failure, continues operating under odometric drift, and supports both exploration and goal-directed navigation without any pre-training. We evaluate the system in large-scale real and simulated environments against state-of-the-art planning baselines, demonstrating its adaptability to ambiguous observations, environmental changes, and sensor noise. The model offers a modular, self-supervised solution to scalable navigation in unstructured settings. AIMAPP is available at this https URL.

**arXiv ID:** 2510.09574
</details>

<details>
<summary><strong>Unveiling Uncertainty-Aware Autonomous Cooperative Learning Based Planning Strategy</strong> - Shiyao Zhang, Liwei Deng, Shuyu Zhang, Weijie Yuan, Hong Zhang - [[pdf]](https://arxiv.org/pdf/2510.11041)</summary>

**Abstract:** In future intelligent transportation systems, autonomous cooperative planning (ACP), becomes a promising technique to increase the effectiveness and security of multi-vehicle interactions. However, multiple uncertainties cannot be fully addressed for existing ACP strategies, e.g. perception, planning, and communication uncertainties. To address these, a novel deep reinforcement learning-based autonomous cooperative planning (DRLACP) framework is proposed to tackle various uncertainties on cooperative motion planning schemes. Specifically, the soft actor-critic (SAC) with the implementation of gate recurrent units (GRUs) is adopted to learn the deterministic optimal time-varying actions with imperfect state information occurred by planning, communication, and perception uncertainties. In addition, the real-time actions of autonomous vehicles (AVs) are demonstrated via the Car Learning to Act (CARLA) simulation platform. Evaluation results show that the proposed DRLACP learns and performs cooperative planning effectively, which outperforms other baseline methods under different scenarios with imperfect AV state information.

**arXiv ID:** 2510.11041
</details>

<details>
<summary><strong>CARLA-Air: Fly Drones Inside a CARLA World -- A Unified Infrastructure for Air-Ground Embodied Intelligence</strong> - Tianle Zeng, Yanci Wen, Hong Zhang - [[pdf]](https://arxiv.org/pdf/2603.28032)</summary>

**Abstract:** The convergence of low-altitude economies, embodied intelligence, and air-ground cooperative systems creates growing demand for simulation infrastructure capable of jointly modeling aerial and ground agents within a single physically coherent environment. Existing open-source platforms remain domain-segregated: driving simulators lack aerial dynamics, while multirotor simulators lack realistic ground scenes. Bridge-based co-simulation introduces synchronization overhead and cannot guarantee strict spatial-temporal consistency.
We present CARLA-Air, an open-source infrastructure that unifies high-fidelity urban driving and physics-accurate multirotor flight within a single Unreal Engine process. The platform preserves both CARLA and AirSim native Python APIs and ROS 2 interfaces, enabling zero-modification code reuse. Within a shared physics tick and rendering pipeline, CARLA-Air delivers photorealistic environments with rule-compliant traffic, socially-aware pedestrians, and aerodynamically consistent UAV dynamics, synchronously capturing up to 18 sensor modalities across all platforms at each tick. The platform supports representative air-ground embodied intelligence workloads spanning cooperation, embodied navigation and vision-language action, multi-modal perception and dataset construction, and reinforcement-learning-based policy training. An extensible asset pipeline allows integration of custom robot platforms into the shared world. By inheriting AirSim's aerial capabilities -- whose upstream development has been archived -- CARLA-Air ensures this widely adopted flight stack continues to evolve within a modern infrastructure.
Released with prebuilt binaries and full source: this https URL

**arXiv ID:** 2603.28032
</details>

<details>
<summary><strong>Evolvable Embodied Agent for Robotic Manipulation via Long Short-Term Reflection and Optimization</strong> - Jianzong Wang, Botao Zhao, Yayun He, Junqing Peng, Xulong Zhang - [[pdf]](https://arxiv.org/pdf/2604.13533)</summary>

**Abstract:** Achieving general-purpose robotics requires empowering robots to adapt and evolve based on their environment and feedback. Traditional methods face limitations such as extensive training requirements, difficulties in cross-task generalization, and lack of interpretability. Prompt learning offers new opportunities for self-evolving robots without extensive training, but simply reflecting on past experiences. However, extracting meaningful insights from task successes and failures remains a challenge. To this end, we propose the evolvable embodied agent (EEAgent) framework, which leverages large vision-language models (VLMs) for better environmental interpretation and policy planning. To enhance reflection on past experiences, we propose a long short-term reflective optimization (LSTRO) mechanism that dynamically refines prompts based on both past experiences and newly learned lessons, facilitating continuous self-evolution, thereby enhancing overall task success rates. Evaluations on six VIMA-Bench tasks reveal that our approach sets a new state-of-the-art, notably outperforming baselines in complex scenarios.

**arXiv ID:** 2604.13533
</details>

<details>
<summary><strong>DR. INFO at the Point of Care: A Prospective Pilot Study of Physician-Perceived Value of an Agentic AI Clinical Assistant</strong> - Rogerio Corga Da Silva, Miguel Romano, Tiago Mendes, Marta Isidoro, Sandhanakrishnan Ravichandran, Shivesh Kumar, Michiel van der Heijden, Olivier Fail, Valentine Emmanuel Gnanapragasam - [[pdf]](https://arxiv.org/pdf/2604.16346)</summary>

**Abstract:** Background: Clinical documentation and information retrieval consume over half of physicians working hours, contributing to cognitive overload and burnout. While artificial intelligence offers a potential solution, concerns over hallucinations and source reliability have limited adoption at the point of care. This study aimed to evaluate physician-perceived time efficiency, decision-making support, and satisfaction with DR. INFO, an agentic AI clinical assistant, in routine clinical practice.
Methodology: In this prospective, single-arm, pilot feasibility study, 29 physicians and medical students across multiple specialties in Portuguese healthcare institutions used DR. INFO v1.0 over five working days within a two-week period. Outcomes were assessed via daily Likert-scale evaluations (time saving and decision support) and a final Net Promoter Score (NPS). Non-parametric methods were used throughout, with bootstrap confidence intervals (CIs) and sensitivity analysis to address non-response.
Results: Physicians reported high perceived time saving (mean = 4.27/5; 95% CI = 3.97-4.57) and decision support (mean = 4.16/5; 95% CI = 3.86-4.45), with ratings stable across the five-day study window. Among the 16 (55%) participants who completed the final evaluation, the NPS was 81.2, with no detractors; sensitivity analysis indicated an NPS of 44.8 under conservative non-response assumptions.
Conclusions: Physicians across specialties and career stages reported positive perceptions of DR. INFO for both time efficiency and clinical decision support within the study window. These findings are preliminary and should be confirmed in larger, controlled studies that include objective performance measures and independent accuracy verification.

**arXiv ID:** 2604.16346
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
