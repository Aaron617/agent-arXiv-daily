# Agent arXiv Daily

**Last Updated:** 2026-02-04 02:52:32

**Total Papers:** 162

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (13 papers)</h2></summary>

<details>
<summary><strong>QUASAR: A Universal Autonomous System for Atomistic Simulation and a Benchmark of Its Capabilities</strong> - Fengxu Yang, Jack D. Evans - [[pdf]](https://arxiv.org/pdf/2602.00185)</summary>

**Abstract:** The integration of large language models (LLMs) into materials science offers a transformative opportunity to streamline computational workflows, yet current agentic systems remain constrained by rigid tool-calling approaches and narrowly scoped agents. In this work, we introduce QUASAR, a universal autonomous system for atomistic simulation designed to facilitate production-grade scientific discovery. QUASAR autonomously orchestrates complex multi-scale workflows across diverse methods, including density functional theory, machine learning potentials, molecular dynamics, and Monte Carlo simulations. The system incorporates robust mechanisms for adaptive planning, context-efficient memory management, and hybrid knowledge retrieval to navigate real-world research scenarios without human intervention. We benchmark QUASAR against a series of three-tiered tasks, progressing from routine tasks to frontier research challenges such as photocatalyst screening and novel material assessment. These results suggest that QUASAR can function as a general atomistic reasoning system rather than a task-specific automation framework. They also provide initial evidence supporting the potential deployment of agentic AI as a component of computational chemistry research workflows, while identifying areas requiring further development.

**arXiv ID:** 2602.00185
</details>

<details>
<summary><strong>TessPay: Verify-then-Pay Infrastructure for Trusted Agentic Commerce</strong> - Mehul Goenka, Tejas Pathak, Siddharth Asthana - [[pdf]](https://arxiv.org/pdf/2602.00213)</summary>

**Abstract:** The global economy is entering the era of Agentic Commerce, where autonomous agents can discover services, negotiate prices, and transact value. However adoption towards agentic commerce faces a foundational trust gap: current systems are built for direct human interactions rather than agent-driven operations. It lacks core primitives across three critical stages of agentic transactions. First, Task Delegation lacks means to translate user intent into defined scopes, discover appropriate agents, and securely authorize actions. Second, Payment Settlement for tasks is processed before execution, lacking verifiable evidence to validate the agent's work. Third, Audit Mechanisms fail to capture the full transaction lifecycle, preventing clear accountability for disputes. While emerging standards address fragments of this trust gap, there still remains a critical need for a unified infrastructure that binds the entire transaction lifecycle.
To resolve this gap, we introduce TessPay, a unified infrastructure that replaces implicit trust with a 'Verify-then-Pay' architecture. It is a two plane architecture separating control and verification from settlement. TessPay operationalizes trust across four distinct stages: Before execution, agents are anchored in a canonical registry and user intent is captured as verifiable mandates, enabling stakeholder accountability. During execution, funds are locked in escrow while the agent executes the task and generates cryptographic evidence (TLS Notary, TEE etc.) to support Proof of Task Execution (PoTE). At settlement, the system verifies this evidence and releases funds only when the PoTE satisfies verification predicates; modular rail adapters ensure this PoTE-gated escrow remains chain-agnostic across heterogeneous payment rails. After settlement, TessPay preserves a tamper-evident audit trail to enable clear accountability for dispute resolution.

**arXiv ID:** 2602.00213
</details>

<details>
<summary><strong>From Perception to Action: Spatial AI Agents and World Models</strong> - Gloria Felicia, Nolan Bryant, Handi Putra, Ayaan Gazali, Eliel Lobo, Esteban Rojas - [[pdf]](https://arxiv.org/pdf/2602.01644)</summary>

**Abstract:** While large language models have become the prevailing approach for agentic reasoning and planning, their success in symbolic domains does not readily translate to the physical world. Spatial intelligence, the ability to perceive 3D structure, reason about object relationships, and act under physical constraints, is an orthogonal capability that proves important for embodied agents. Existing surveys address either agentic architectures or spatial domains in isolation. None provide a unified framework connecting these complementary capabilities. This paper bridges that gap. Through a thorough review of over 2,000 papers, citing 742 works from top-tier venues, we introduce a unified three-axis taxonomy connecting agentic capabilities with spatial tasks across scales. Crucially, we distinguish spatial grounding (metric understanding of geometry and physics) from symbolic grounding (associating images with text), arguing that perception alone does not confer agency. Our analysis reveals three key findings mapped to these axes: (1) hierarchical memory systems (Capability axis) are important for long-horizon spatial tasks. (2) GNN-LLM integration (Task axis) is a promising approach for structured spatial reasoning. (3) World models (Scale axis) are essential for safe deployment across micro-to-macro spatial scales. We conclude by identifying six grand challenges and outlining directions for future research, including the need for unified evaluation frameworks to standardize cross-domain assessment. This taxonomy provides a foundation for unifying fragmented research efforts and enabling the next generation of spatially-aware autonomous systems in robotics, autonomous vehicles, and geospatial intelligence.

**arXiv ID:** 2602.01644
</details>

<details>
<summary><strong>David vs. Goliath: Verifiable Agent-to-Agent Jailbreaking via Reinforcement Learning</strong> - Samuel Nellessen, Tal Kachman - [[pdf]](https://arxiv.org/pdf/2602.02395)</summary>

**Abstract:** The evolution of large language models into autonomous agents introduces adversarial failures that exploit legitimate tool privileges, transforming safety evaluation in tool-augmented environments from a subjective NLP task into an objective control problem. We formalize this threat model as Tag-Along Attacks: a scenario where a tool-less adversary "tags along" on the trusted privileges of a safety-aligned Operator to induce prohibited tool use through conversation alone. To validate this threat, we present Slingshot, a 'cold-start' reinforcement learning framework that autonomously discovers emergent attack vectors, revealing a critical insight: in our setting, learned attacks tend to converge to short, instruction-like syntactic patterns rather than multi-turn persuasion. On held-out extreme-difficulty tasks, Slingshot achieves a 67.0% success rate against a Qwen2.5-32B-Instruct-AWQ Operator (vs. 1.7% baseline), reducing the expected attempts to first success (on solved tasks) from 52.3 to 1.3. Crucially, Slingshot transfers zero-shot to several model families, including closed-source models like Gemini 2.5 Flash (56.0% attack success rate) and defensive-fine-tuned open-source models like Meta-SecAlign-8B (39.2% attack success rate). Our work establishes Tag-Along Attacks as a first-class, verifiable threat model and shows that effective agentic attacks can be elicited from off-the-shelf open-weight models through environment interaction alone.

**arXiv ID:** 2602.02395
</details>

<details>
<summary><strong>ES-MemEval: Benchmarking Conversational Agents on Personalized Long-Term Emotional Support</strong> - Tiantian Chen, Jiaqi Lu, Ying Shen, Lin Zhang - [[pdf]](https://arxiv.org/pdf/2602.01885)</summary>

**Abstract:** Large Language Models (LLMs) have shown strong potential as conversational agents. Yet, their effectiveness remains limited by deficiencies in robust long-term memory, particularly in complex, long-term web-based services such as online emotional support. However, existing long-term dialogue benchmarks primarily focus on static and explicit fact retrieval, failing to evaluate agents in critical scenarios where user information is dispersed, implicit, and continuously evolving. To address this gap, we introduce ES-MemEval, a comprehensive benchmark that systematically evaluates five core memory capabilities: information extraction, temporal reasoning, conflict detection, abstention, and user modeling, in long-term emotional support settings, covering question answering, summarization, and dialogue generation tasks. To support the benchmark, we also propose EvoEmo, a multi-session dataset for personalized long-term emotional support that captures fragmented, implicit user disclosures and evolving user states. Extensive experiments on open-source long-context, commercial, and retrieval-augmented (RAG) LLMs show that explicit long-term memory is essential for reducing hallucinations and enabling effective personalization. At the same time, RAG improves factual consistency but struggles with temporal dynamics and evolving user states. These findings highlight both the potential and limitations of current paradigms and motivate more robust integration of memory and retrieval for long-term personalized dialogue systems.

**arXiv ID:** 2602.01885
</details>

<details>
<summary><strong>dziribot: rag based intelligent conversational agent for algerian arabic dialect</strong> - El Batoul Bechiri, Dihia Lanasri - [[pdf]](https://arxiv.org/pdf/2602.02270)</summary>

**Abstract:** The rapid digitalization of customer service has intensified the demand for conversational agents capable of providing accurate and natural interactions. In the Algerian context, this is complicated by the linguistic complexity of Darja, a dialect characterized by non-standardized orthography, extensive code-switching with French, and the simultaneous use of Arabic and Latin (Arabizi) scripts. This paper introduces DziriBOT, a hybrid intelligent conversational agent specifically engineered to overcome these challenges. We propose a multi-layered architecture that integrates specialized Natural Language Understanding (NLU) with Retrieval-Augmented Generation (RAG), allowing for both structured service flows and dynamic, knowledge-intensive responses grounded in curated enterprise documentation. To address the low-resource nature of Darja, we systematically evaluate three distinct approaches: a sparse-feature Rasa pipeline, classical machine learning baselines, and transformer-based fine-tuning. Our experimental results demonstrate that the fine-tuned DziriBERT model achieves state-of-the-art performance. These results significantly outperform traditional baselines, particularly in handling orthographic noise and rare intents. Ultimately, DziriBOT provides a robust, scalable solution that bridges the gap between formal language models and the linguistic realities of Algerian users, offering a blueprint for dialect-aware automation in the regional market.

**arXiv ID:** 2602.02270
</details>

<details>
<summary><strong>Online Fine-Tuning of Pretrained Controllers for Autonomous Driving via Real-Time Recurrent RL</strong> - Julian Lemmel, Felix Resch, Mónika Farsang, Ramin Hasani, Daniela Rus, Radu Grosu - [[pdf]](https://arxiv.org/pdf/2602.02236)</summary>

**Abstract:** Deploying pretrained policies in real-world applications presents substantial challenges that fundamentally limit the practical applicability of learning-based control systems. When autonomous systems encounter environmental changes in system dynamics, sensor drift, or task objectives, fixed policies rapidly degrade in performance. We show that employing Real-Time Recurrent Reinforcement Learning (RTRRL), a biologically plausible algorithm for online adaptation, can effectively fine-tune a pretrained policy to improve autonomous agents' performance on driving tasks. We further show that RTRRL synergizes with a recent biologically inspired recurrent network model, the Liquid-Resistance Liquid-Capacitance RNN. We demonstrate the effectiveness of this closed-loop approach in a simulated CarRacing environment and in a real-world line-following task with a RoboRacer car equipped with an event camera.

**arXiv ID:** 2602.02236
</details>

<details>
<summary><strong>Disclose with Care: Designing Privacy Controls in Interview Chatbots</strong> - Ziwen Li, Ziang Xiao, Tianshi Li - [[pdf]](https://arxiv.org/pdf/2602.01387)</summary>

**Abstract:** Collecting data on sensitive topics remains challenging in HCI, as participants often withhold information due to privacy concerns and social desirability bias. While chatbots' perceived anonymity may reduce these barriers, research paradoxically suggests people tend to over-share personal or sensitive information with chatbots. In this work, we explore privacy controls in chatbot interviews to address this problem. The privacy control allows participants to revise their transcripts at the end of the interview, featuring two design variants: free editing and AI-aided editing. In a between-subjects study \red{($N=188$)}, we compared no-editing, free-editing, and AI-aided editing conditions in a chatbot-based interview on a sensitive topic. Our results confirm the prevalent issue of oversharing in chatbot-based interviews and show that AI-aided editing serves as an effective privacy-control mechanism, reducing PII disclosure while maintaining data quality and user engagement, thereby offering a promising approach to balancing ethical practice and data quality in such interviews.

**arXiv ID:** 2602.01387
</details>

<details>
<summary><strong>Feedback by Design: Understanding and Overcoming User Feedback Barriers in Conversational Agents</strong> - Nikhil Sharma, Zheng Zhang, Daniel Lee, Namita Krishnan, Guang-Jie Ren, Ziang Xiao, Yunyao Li - [[pdf]](https://arxiv.org/pdf/2602.01405)</summary>

**Abstract:** High-quality feedback is essential for effective human-AI interaction. It bridges knowledge gaps, corrects digressions, and shapes system behavior; both during interaction and throughout model development. Yet despite its importance, human feedback to AI is often infrequent and low quality. This gap motivates a critical examination of human feedback during interactions with AIs. To understand and overcome the challenges preventing users from giving high-quality feedback, we conducted two studies examining feedback dynamics between humans and conversational agents (CAs). Our formative study, through the lens of Grice's maxims, identified four Feedback Barriers -- Common Ground, Verifiability, Communication, and Informativeness -- that prevent high-quality feedback by users. Building on these findings, we derive three design desiderata and show that systems incorporating scaffolds aligned with these desiderata enabled users to provide higher-quality feedback. Finally, we detail a call for action to the broader AI community for advances in Large Language Models capabilities to overcome Feedback Barriers.

**arXiv ID:** 2602.01405
</details>

<details>
<summary><strong>When Workout Buddies Are Virtual: AI Agents and Human Peers in a Longitudinal Physical Activity Study</strong> - Alessandro Silacci, Mauro Cherubini, Arianna Boldi, Amon Rapp, Maurizio Caon - [[pdf]](https://arxiv.org/pdf/2602.01918)</summary>

**Abstract:** Physical inactivity remains a critical global health issue, yet scalable strategies for sustained motivation are scarce. Conversational agents designed as simulated exercising peers (SEPs) represent a promising alternative, but their long-term impact is unclear. We report a six-month randomized controlled trial (N=280) comparing individuals exercising alone, with a human peer, or with a large language model-driven SEP. Results revealed a partnership paradox: human peers evoked stronger social presence, while AI peers provided steadier encouragement and more reliable working alliances. Humans motivated through authentic comparison and accountability, whereas AI peers fostered consistent, low-stakes support. These complementary strengths suggest that AI agents should not mimic human authenticity but augment it with reliability. Our findings advance human-agent interaction research and point to hybrid designs where human presence and AI consistency jointly sustain physical activity.

**arXiv ID:** 2602.01918
</details>

<details>
<summary><strong>Vulnerability-Amplifying Interaction Loops: a systematic failure mode in AI chatbot mental-health interactions</strong> - Veith Weilnhammer, Kevin YC Hou, Raymond Dolan, Matthew M Nour - [[pdf]](https://arxiv.org/pdf/2602.01347)</summary>

**Abstract:** Millions of users turn to consumer AI chatbots to discuss behavioral and mental health concerns. While this presents unprecedented opportunities to deliver population-level support, it also highlights an urgent need to develop rigorous and scalable safety evaluations. Here we introduce SIM-VAIL, an AI chatbot auditing framework that captures how harmful AI chatbot responses manifest across a range of mental-health contexts. SIM-VAIL pairs a simulated human user, harboring a distinct psychiatric vulnerability and conversational intent, with an audited frontier AI chatbot. It scores conversation turns on 13 clinically relevant risk dimensions, enabling context-dependent, temporally resolved assessment of mental-health risk. Across 810 conversations, encompassing over 90,000 turn-level ratings and 30 psychiatric user profiles, we find that significant risk occurs across virtually all user phenotypes. Risk manifested across most of the 9 consumer AI chatbot models audited, albeit mitigated in more modern variants. Rather than arising abruptly, risk accumulated over multiple turns. Risk profiles were phenotype-dependent, indicating that behaviors that appear supportive in general settings are liable to be maladaptive when they align with mechanisms that sustain a user's vulnerability. Multivariate risk patterns revealed trade-offs across dimensions, suggesting that mitigation targeting one harm domain can exacerbate others. These findings identify a novel failure mode in human-AI interactions, which we term Vulnerability-Amplifying Interaction Loops (VAILs), and underscore the need for multi-dimensional approaches to risk quantification. SIM-VAIL provides a scalable evaluation framework for quantifying how mental-health risk is distributed across user phenotypes, conversational trajectories, and clinically grounded behavioral dimensions, offering a foundation for targeted safety improvements.

**arXiv ID:** 2602.01347
</details>

<details>
<summary><strong>Conversational Agents in Behavioral Sleep Medicine: Designing Self-Report and Analytics Tools</strong> - Amama Mahmood, Bokyung Kim, Honghao Zhao, Molly E. Atwood, Luis F. Buenaver, Michael T. Smith, Chien-Ming Huang - [[pdf]](https://arxiv.org/pdf/2509.15378)</summary>

**Abstract:** The sleep diary is a widely used clinical tool for understanding and treating sleep disorders in Behavioral Sleep Medicine (BSM); however, low patient compliance and limited capture of contextual information constrain its effectiveness and leave specialists with an incomplete picture of patients' sleep-related behaviors. In this work, we explore conversational agents (CAs) as an alternative to traditional diary methods by designing a voice-based sleep diary and a specialist-facing analytics tool, and using them as design probes to understand how CAs might support BSM more broadly. Our multi-stage study with specialists comprised: (1) interviews to identify shortcomings of current text-based diaries, (2) iterative co-design of the conversational diary and analytics tool, and (3) focus groups to examine broader opportunities for CAs in BSM. This work offers empirical insights into how specialists envision CAs in clinical care and outlines design implications for integrating them into existing self-report practices and behavioral interventions.

**arXiv ID:** 2509.15378
</details>

<details>
<summary><strong>CHOIR: A Chatbot-mediated Organizational Memory Leveraging Communication in University Research Labs</strong> - Sangwook Lee, Adnan Abbas, Yan Chen, Young-Ho Kim, Sang Won Lee - [[pdf]](https://arxiv.org/pdf/2509.20512)</summary>

**Abstract:** University research labs often rely on chat-based platforms for communication and project management, where valuable knowledge surfaces but is easily lost in message streams. Documentation can preserve knowledge, but it requires ongoing maintenance and is challenging to navigate. Drawing on formative interviews that revealed organizational memory challenges in labs, we designed CHOIR, an LLM-based chatbot that supports organizational memory through four key functions: document-grounded Q\&A, Q\&A sharing for follow-up discussion, knowledge extraction from conversations, and AI-assisted document updates. We deployed CHOIR in four research labs for one month (n=21), where the lab members asked 107 questions and lab directors updated documents 38 times in the organizational memory. Our findings reveal a privacy-awareness tension: questions were asked privately, limiting directors' visibility into documentation gaps. Students often avoided contribution due to challenges in generalizing personal experiences into universal documentation. We contribute design implications for privacy-preserving awareness and supporting context-specific knowledge documentation.

**arXiv ID:** 2509.20512
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (26 papers)</h2></summary>

<details>
<summary><strong>Autonomous Data Processing using Meta-Agents</strong> - Udayan Khurana - [[pdf]](https://arxiv.org/pdf/2602.00307)</summary>

**Abstract:** Traditional data processing pipelines are typically static and handcrafted for specific tasks, limiting their adaptability to evolving requirements. While general-purpose agents and coding assistants can generate code for well-understood data pipelines, they lack the ability to autonomously monitor, manage, and optimize an end-to-end pipeline once deployed. We present \textbf{Autonomous Data Processing using Meta-agents} (ADP-MA), a framework that dynamically constructs, executes, and iteratively refines data processing pipelines through hierarchical agent orchestration. At its core, \textit{meta-agents} analyze input data and task specifications to design a multi-phase plan, instantiate specialized \textit{ground-level agents}, and continuously evaluate pipeline performance. The architecture comprises three key components: a planning module for strategy generation, an orchestration layer for agent coordination and tool integration, and a monitoring loop for iterative evaluation and backtracking. Unlike conventional approaches, ADP-MA emphasizes context-aware optimization, adaptive workload partitioning, and progressive sampling for scalability. Additionally, the framework leverages a diverse set of external tools and can reuse previously designed agents, reducing redundancy and accelerating pipeline construction. We demonstrate ADP-MA through an interactive demo that showcases pipeline construction, execution monitoring, and adaptive refinement across representative data processing tasks.

**arXiv ID:** 2602.00307
</details>

<details>
<summary><strong>PolarMem: A Training-Free Polarized Latent Graph Memory for Verifiable Multimodal Agents</strong> - Zhisheng Chen, Tingyu Wu, Zijie Zhou, Zhengwei Xie, Ziyan Weng, Yingwei Zhang - [[pdf]](https://arxiv.org/pdf/2602.00415)</summary>

**Abstract:** As multimodal agents evolve from passive observers to long-horizon decision-makers, they require memory systems that provide not just information availability but logical verifiability. A fundamental limitation of current architectures is the epistemic asymmetry inherent in probabilistic vision-language models and dense associative memories: they conflate semantic affinity with factual existence and structurally fail to encode negative constraints. To this end, we introduce PolarMem, a training-free Polarized Latent Graph Memory designed to ground agent reasoning in verifiable evidence. PolarMem transforms fuzzy perceptual likelihoods into discrete logical constraints through non-parametric distributional partitioning. Furthermore, it employs a polarized graph topology with orthogonal inhibitory connections to explicitly store verified negation as a primary cognitive state. At inference time, we enforce a logic-dominant retrieval paradigm, suppressing hallucinatory patterns that violate negative constraints. Extensive evaluation across eight frozen Vision--Language Models and six benchmarks demonstrates that PolarMem functions as a robust cognitive system, establishing a foundation for verifiable multimodal agents. Our code is available at this https URL.

**arXiv ID:** 2602.00415
</details>

<details>
<summary><strong>Benchmarking Agents in Insurance Underwriting Environments</strong> - Amanda Dsouza, Ramya Ramakrishnan, Charles Dickens, Bhavishya Pohani, Christopher M Glaze - [[pdf]](https://arxiv.org/pdf/2602.00456)</summary>

**Abstract:** As AI agents integrate into enterprise applications, their evaluation demands benchmarks that reflect the complexity of real-world operations. Instead, existing benchmarks overemphasize open-domains such as code, use narrow accuracy metrics, and lack authentic complexity. We present UNDERWRITE, an expert-first, multi-turn insurance underwriting benchmark designed in close collaboration with domain experts to capture real-world enterprise challenges. UNDERWRITE introduces critical realism factors often absent in current benchmarks: proprietary business knowledge, noisy tool interfaces, and imperfect simulated users requiring careful information gathering. Evaluating 13 frontier models, we uncover significant gaps between research lab performance and enterprise readiness: the most accurate models are not the most efficient, models hallucinate domain knowledge despite tool access, and pass^k results show a 20% drop in performance. The results from UNDERWRITE demonstrate that expert involvement in benchmark design is essential for realistic agent evaluation, common agentic frameworks exhibit brittleness that skews performance reporting, and hallucination detection in specialized domains demands compositional approaches. Our work provides insights for developing benchmarks that better align with enterprise deployment requirements.

**arXiv ID:** 2602.00456
</details>

<details>
<summary><strong>Exploring Information Seeking Agent Consolidation</strong> - Guochen Yan, Jialong Wu, Zhengwei Tao, Bo Li, Qintong Zhang, Jiahao Xu, Haitao Mi, Yuejian Fang, Qingni Shen, Wentao Zhang, Zhonghai Wu - [[pdf]](https://arxiv.org/pdf/2602.00585)</summary>

**Abstract:** Information-seeking agents have emerged as a powerful paradigm for solving knowledge-intensive tasks. Existing information-seeking agents are typically specialized for open web, documents, or local knowledge bases, which constrains scalability and cross-domain generalization. In this work, we investigate how to consolidate heterogeneous information-seeking agents into a single foundation agentic model. We study two complementary consolidation strategies: data-level consolidation, which jointly trains a unified model on a mixture of domain-specific datasets, and parameter-level consolidation, which merges independently trained agent models at the parameter level. Our analysis compares these approaches in terms of performance retention, cross-domain generalization, and interference across information-seeking behaviors. Our results show that data-level consolidation remains a strong and stable baseline, while parameter-level consolidation offers a promising, efficient alternative but suffers from interference and robustness challenges. We further identify key design factors for effective agent consolidation at the parameter level, including fine-grained merging granularity, awareness of task heterogeneity, and principled consensus strategy.

**arXiv ID:** 2602.00585
</details>

<details>
<summary><strong>DockSmith: Scaling Reliable Coding Environments via an Agentic Docker Builder</strong> - Jiaran Zhang, Luck Ma, Yanhao Li, Fanqi Wan, Di Qi, Xu Zhao, Jieyi Hou, Zhe Xie, Mengqiang Ren, Xin Wu, Zhewei Huang, Liangyu Chen, Yingwei Ma, Qi Han, Xiangyu Zhang - [[pdf]](https://arxiv.org/pdf/2602.00592)</summary>

**Abstract:** Reliable Docker-based environment construction is a dominant bottleneck for scaling execution-grounded training and evaluation of software engineering agents. We introduce DockSmith, a specialized agentic Docker builder designed to address this challenge. DockSmith treats environment construction not only as a preprocessing step, but as a core agentic capability that exercises long-horizon tool use, dependency reasoning, and failure recovery, yielding supervision that transfers beyond Docker building itself. DockSmith is trained on large-scale, execution-grounded Docker-building trajectories produced by a SWE-Factory-style pipeline augmented with a loop-detection controller and a cross-task success memory. Training a 30B-A3B model on these trajectories achieves open-source state-of-the-art performance on Multi-Docker-Eval, with 39.72% Fail-to-Pass and 58.28% Commit Rate. Moreover, DockSmith improves out-of-distribution performance on SWE-bench Verified, SWE-bench Multilingual, and Terminal-Bench 2.0, demonstrating broader agentic benefits of environment construction.

**arXiv ID:** 2602.00592
</details>

<details>
<summary><strong>HumanStudy-Bench: Towards AI Agent Design for Participant Simulation</strong> - Xuan Liu, Haoyang Shang, Zizhang Liu, Xinyan Liu, Yunze Xiao, Yiwen Tu, Haojian Jin - [[pdf]](https://arxiv.org/pdf/2602.00685)</summary>

**Abstract:** Large language models (LLMs) are increasingly used as simulated participants in social science experiments, but their behavior is often unstable and highly sensitive to design choices. Prior evaluations frequently conflate base-model capabilities with experimental instantiation, obscuring whether outcomes reflect the model itself or the agent setup. We instead frame participant simulation as an agent-design problem over full experimental protocols, where an agent is defined by a base model and a specification (e.g., participant attributes) that encodes behavioral assumptions. We introduce HUMANSTUDY-BENCH, a benchmark and execution engine that orchestrates LLM-based agents to reconstruct published human-subject experiments via a Filter--Extract--Execute--Evaluate pipeline, replaying trial sequences and running the original analysis pipeline in a shared runtime that preserves the original statistical procedures end to end. To evaluate fidelity at the level of scientific inference, we propose new metrics to quantify how much human and agent behaviors agree. We instantiate 12 foundational studies as an initial suite in this dynamic benchmark, spanning individual cognition, strategic interaction, and social psychology, and covering more than 6,000 trials with human samples ranging from tens to over 2,100 participants.

**arXiv ID:** 2602.00685
</details>

<details>
<summary><strong>Optimizing Agentic Reasoning with Retrieval via Synthetic Semantic Information Gain Reward</strong> - Senkang Hu, Yong Dai, Yuzhi Zhao, Yihang Tao, Yu Guo, Zhengru Fang, Sam Tak Wu Kwong, Yuguang Fang - [[pdf]](https://arxiv.org/pdf/2602.00845)</summary>

**Abstract:** Agentic reasoning enables large reasoning models (LRMs) to dynamically acquire external knowledge, but yet optimizing the retrieval process remains challenging due to the lack of dense, principled reward signals. In this paper, we introduce InfoReasoner, a unified framework that incentivizes effective information seeking via a synthetic semantic information gain reward. Theoretically, we redefine information gain as uncertainty reduction over the model's belief states, establishing guarantees, including non-negativity, telescoping additivity, and channel monotonicity. Practically, to enable scalable optimization without manual retrieval annotations, we propose an output-aware intrinsic estimator that computes information gain directly from the model's output distributions using semantic clustering via bidirectional textual entailment. This intrinsic reward guides the policy to maximize epistemic progress, enabling efficient training via Group Relative Policy Optimxization (GRPO). Experiments across seven question-answering benchmarks demonstrate that InfoReasoner consistently outperforms strong retrieval-augmented baselines, achieving up to 5.4% average accuracy improvement. Our work provides a theoretically grounded and scalable path toward agentic reasoning with retrieval.

**arXiv ID:** 2602.00845
</details>

<details>
<summary><strong>Aggregation Queries over Unstructured Text: Benchmark and Agentic Method</strong> - Haojia Zhu, Qinyuan Xu, Haoyu Li, Yuxi Liu, Hanchen Qiu, Jiaoyan Chen, Jiahui Jin - [[pdf]](https://arxiv.org/pdf/2602.01355)</summary>

**Abstract:** Aggregation query over free text is a long-standing yet underexplored problem. Unlike ordinary question answering, aggregate queries require exhaustive evidence collection and systems are required to "find all," not merely "find one." Existing paradigms such as Text-to-SQL and Retrieval-Augmented Generation fail to achieve this completeness. In this work, we formalize entity-level aggregation querying over text in a corpus-bounded setting with strict completeness requirement. To enable principled evaluation, we introduce AGGBench, a benchmark designed to evaluate completeness-oriented aggregation under realistic large-scale corpus. To accompany the benchmark, we propose DFA (Disambiguation--Filtering--Aggregation), a modular agentic baseline that decomposes aggregation querying into interpretable stages and exposes key failure modes related to ambiguity, filtering, and aggregation. Empirical results show that DFA consistently improves aggregation evidence coverage over strong RAG and agentic baselines. The data and code are available in \href{this https URL}.

**arXiv ID:** 2602.01355
</details>

<details>
<summary><strong>PRISM: Festina Lente Proactivity -- Risk-Sensitive, Uncertainty-Aware Deliberation for Proactive Agents</strong> - Yuxuan Fu, Xiaoyu Tan, Teqi Hao, Chen Zhan, Xihe Qiu - [[pdf]](https://arxiv.org/pdf/2602.01532)</summary>

**Abstract:** Proactive agents must decide not only what to say but also whether and when to intervene. Many current systems rely on brittle heuristics or indiscriminate long reasoning, which offers little control over the benefit-burden tradeoff. We formulate the problem as cost-sensitive selective intervention and present PRISM, a novel framework that couples a decision-theoretic gate with a dual-process reasoning architecture. At inference time, the agent intervenes only when a calibrated probability of user acceptance exceeds a threshold derived from asymmetric costs of missed help and false alarms. Inspired by festina lente (Latin: "make haste slowly"), we gate by an acceptance-calibrated, cost-derived threshold and invoke a resource-intensive Slow mode with counterfactual checks only near the decision boundary, concentrating computation on ambiguous and high-stakes cases. Training uses gate-aligned, schema-locked distillation: a teacher running the full PRISM pipeline provides dense, executable supervision on unlabeled interaction traces, while the student learns a response policy that is explicitly decoupled from the intervention gate to enable tunable and auditable control. On ProactiveBench, PRISM reduces false alarms by 22.78% and improves F1 by 20.14% over strong baselines. These results show that principled decision-theoretic gating, paired with selective slow reasoning and aligned distillation, yields proactive agents that are precise, computationally efficient, and controllable. To facilitate reproducibility, we release our code, models, and resources at this https URL all experiments use the open-source ProactiveBench benchmark.

**arXiv ID:** 2602.01532
</details>

<details>
<summary><strong>ProjDevBench: Benchmarking AI Coding Agents on End-to-End Project Development</strong> - Pengrui Lu, Shiqi Zhang, Yunzhong Hou, Lyumanshan Ye, Chaoyi Huang, Zixi Chen, Ji Zeng, Hantao Jiang, Pengfei Liu, Yiwei Wang, Ming-Hsuan Yang - [[pdf]](https://arxiv.org/pdf/2602.01655)</summary>

**Abstract:** Recent coding agents can generate complete codebases from simple prompts, yet existing evaluations focus on issue-level bug fixing and lag behind end-to-end development. We introduce ProjDevBench, an end-to-end benchmark that provides project requirements to coding agents and evaluates the resulting repositories. Combining Online Judge (OJ) testing with LLM-assisted code review, the benchmark evaluates agents on (1) system architecture design, (2) functional correctness, and (3) iterative solution refinement. We curate 20 programming problems across 8 categories, covering both concept-oriented tasks and real-world application scenarios, and evaluate six coding agents built on different LLM backends. Our evaluation reports an overall acceptance rate of 27.38%: agents handle basic functionality and data structures but struggle with complex system design, time complexity optimization, and resource management. Our benchmark is available at this https URL.

**arXiv ID:** 2602.01655
</details>

<details>
<summary><strong>Live-Evo: Online Evolution of Agentic Memory from Continuous Feedback</strong> - Yaolun Zhang, Yiran Wu, Yijiong Yu, Qingyun Wu, Huazheng Wang - [[pdf]](https://arxiv.org/pdf/2602.02369)</summary>

**Abstract:** Large language model (LLM) agents are increasingly equipped with memory, which are stored experience and reusable guidance that can improve task-solving performance. Recent \emph{self-evolving} systems update memory based on interaction outcomes, but most existing evolution pipelines are developed for static train/test splits and only approximate online learning by folding static benchmarks, making them brittle under true distribution shift and continuous feedback. We introduce \textsc{Live-Evo}, an online self-evolving memory system that learns from a stream of incoming data over time. \textsc{Live-Evo} decouples \emph{what happened} from \emph{how to use it} via an Experience Bank and a Meta-Guideline Bank, compiling task-adaptive guidelines from retrieved experiences for each task. To manage memory online, \textsc{Live-Evo} maintains experience weights and updates them from feedback: experiences that consistently help are reinforced and retrieved more often, while misleading or stale experiences are down-weighted and gradually forgotten, analogous to reinforcement and decay in human memory. On the live \textit{Prophet Arena} benchmark over a 10-week horizon, \textsc{Live-Evo} improves Brier score by 20.8\% and increases market returns by 12.9\%, while also transferring to deep-research benchmarks with consistent gains over strong baselines. Our code is available at this https URL.

**arXiv ID:** 2602.02369
</details>

<details>
<summary><strong>Avenir-Web: Human-Experience-Imitating Multimodal Web Agents with Mixture of Grounding Experts</strong> - Aiden Yiliu Li, Xinyue Hao, Shilong Liu, Mengdi Wang - [[pdf]](https://arxiv.org/pdf/2602.02468)</summary>

**Abstract:** Despite advances in multimodal large language models, autonomous web agents still struggle to reliably execute long-horizon tasks on complex and dynamic web interfaces. Existing agents often suffer from inaccurate element grounding, the absence of site-specific procedural knowledge, and unstable long-term task tracking and memory, particularly when operating over complex Document Object Model structures. To address these limitations, we introduce Avenir-Web, a web agent that achieves a new open-source state of the art on the Online-Mind2Web benchmark in real-world deployment. Avenir-Web leverages a Mixture of Grounding Experts, Experience-Imitation Planning for incorporating procedural priors, and a task-tracking checklist combined with adaptive memory to enable robust and seamless interaction across diverse user interface paradigms. We evaluate Avenir-Web on Online-Mind2Web, a rigorous benchmark of live and user-centered web tasks. Our results demonstrate that Avenir-Web significantly surpasses prior open-source agents and attains performance parity with top-tier proprietary models, thereby establishing a new open-source state of the art for reliable web agents on live websites.

**arXiv ID:** 2602.02468
</details>

<details>
<summary><strong>Why Are AI Agent Involved Pull Requests (Fix-Related) Remain Unmerged? An Empirical Study</strong> - Khairul Alam, Saikat Mondal, Banani Roy - [[pdf]](https://arxiv.org/pdf/2602.00164)</summary>

**Abstract:** Autonomous coding agents (e.g., OpenAI Codex, Devin, GitHub Copilot) are increasingly used to generate fix-related pull requests (PRs) in real world software repositories. However, their practical effectiveness depends on whether these contributions are accepted and merged by project maintainers. In this paper, we present an empirical study of AI agent involved fix related PRs, examining both their integration outcomes, latency, and the factors that hinder successful merging. We first analyze 8,106 fix related PRs authored by five widely used AI coding agents from the AIDEV POP dataset to quantify the proportions of PRs that are merged, closed without merging, or remain open. We then conduct a manual qualitative analysis of a statistically significant sample of 326 closed but unmerged PRs, spending approximately 100 person hours to construct a structured catalog of 12 failure reasons. Our results indicate that test case failures and prior resolution of the same issues by other PRs are the most common causes of non integration, whereas build or deployment failures are comparatively rare. Overall, our findings expose key limitations of current AI coding agents in real world settings and highlight directions for their further improvement and for more effective human AI collaboration in software maintenance.

**arXiv ID:** 2602.00164
</details>

<details>
<summary><strong>Towards Scientific Intelligence: A Survey of LLM-based Scientific Agents</strong> - Shuo Ren, Can Xie, Pu Jian, Zhenjiang Ren, Chunlin Leng, Jiajun Zhang - [[pdf]](https://arxiv.org/pdf/2503.24047)</summary>

**Abstract:** As scientific research becomes increasingly complex, innovative tools are needed to manage vast data, facilitate interdisciplinary collaboration, and accelerate discovery. Large language models (LLMs) are now evolving into LLM-based scientific agents that automate critical tasks ranging from hypothesis generation and experiment design to data analysis and simulation. Unlike general-purpose LLMs, these specialized agents integrate domain-specific knowledge, advanced tool sets, and robust validation mechanisms, enabling them to handle complex data types, ensure reproducibility, and drive scientific breakthroughs. This survey provides a focused review of the architectures, design, benchmarks, applications, and ethical considerations surrounding LLM-based scientific agents. We highlight why they differ from general agents and the ways in which they advance research across various scientific fields. By examining their development and challenges, this survey offers a comprehensive roadmap for researchers and practitioners to harness these agents for more efficient, reliable, and ethically sound scientific discovery.

**arXiv ID:** 2503.24047
</details>

<details>
<summary><strong>SemanticALLI: Caching Reasoning, Not Just Responses, in Agentic Systems</strong> - Varun Chillara, Dylan Kline, Christopher Alvares, Evan Wooten, Huan Yang, Shlok Khetan, Cade Bauer, Tré Guillory, Tanishka Shah, Yashodhara Dhariwal, Volodymyr Pavlov, George Popstefanov - [[pdf]](https://arxiv.org/pdf/2601.16286)</summary>

**Abstract:** Agentic AI pipelines suffer from a hidden inefficiency: they frequently reconstruct identical intermediate logic, such as metric normalization or chart scaffolding, even when the user's natural language phrasing is entirely novel. Conventional boundary caching fails to capture this inefficiency because it treats inference as a monolithic black box.
We introduce SemanticALLI, a pipeline-aware architecture within Alli (PMG's marketing intelligence platform), designed to operationalize redundant reasoning. By decomposing generation into Analytic Intent Resolution (AIR) and Visualization Synthesis (VS), SemanticALLI elevates structured intermediate representations (IRs) to first-class, cacheable artifacts.
The impact of caching within the agentic loop is substantial. In our evaluation, baseline monolithic caching caps at a 38.7% hit rate due to linguistic variance. In contrast, our structured approach allows for an additional stage, the Visualization Synthesis stage, to achieve an 83.10% hit rate, bypassing 4,023 LLM calls with a median latency of just 2.66 ms. This internal reuse reduces total token consumption, offering a practical lesson for AI system design: even when users rarely repeat themselves, the pipeline often does, at stable, structured checkpoints where caching is most reliable.

**arXiv ID:** 2601.16286
</details>

<details>
<summary><strong>DETOUR: An Interactive Benchmark for Dual-Agent Search and Reasoning</strong> - Li Siyan, Darshan Deshpande, Anand Kannappan, Rebecca Qian - [[pdf]](https://arxiv.org/pdf/2602.00352)</summary>

**Abstract:** When recalling information in conversation, people often arrive at the recollection after multiple turns. However, existing benchmarks for evaluating agent capabilities in such tip-of-the-tongue search processes are restricted to single-turn settings. To more realistically simulate tip-of-the-tongue search, we introduce Dual-agent based Evaluation Through Obscure Under-specified Retrieval (DETOUR), a dual-agent evaluation benchmark containing 1,011 prompts. The benchmark design involves a Primary Agent, which is the subject of evaluation, tasked with identifying the recollected entity through querying a Memory Agent that is held consistent across evaluations. Our results indicate that current state-of-the-art models still struggle with our benchmark, only achieving 36% accuracy when evaluated on all modalities (text, image, audio, and video), highlighting the importance of enhancing capabilities in underspecified scenarios.

**arXiv ID:** 2602.00352
</details>

<details>
<summary><strong>FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents</strong> - Chiwei Zhu, Benfeng Xu, Mingxuan Du, Shaohan Wang, Xiaorui Wang, Zhendong Mao, Yongdong Zhang - [[pdf]](https://arxiv.org/pdf/2602.01566)</summary>

**Abstract:** Deep research is emerging as a representative long-horizon task for large language model (LLM) agents. However, long trajectories in deep research often exceed model context limits, compressing token budgets for both evidence collection and report writing, and preventing effective test-time scaling. We introduce FS-Researcher, a file-system-based, dual-agent framework that scales deep research beyond the context window via a persistent workspace. Specifically, a Context Builder agent acts as a librarian which browses the internet, writes structured notes, and archives raw sources into a hierarchical knowledge base that can grow far beyond context length. A Report Writer agent then composes the final report section by section, treating the knowledge base as the source of facts. In this framework, the file system serves as a durable external memory and a shared coordination medium across agents and sessions, enabling iterative refinement beyond the context window. Experiments on two open-ended benchmarks (DeepResearch Bench and DeepConsult) show that FS-Researcher achieves state-of-the-art report quality across different backbone models. Further analyses demonstrate a positive correlation between final report quality and the computation allocated to the Context Builder, validating effective test-time scaling under the file-system paradigm. The code and data are anonymously open-sourced at this https URL.

**arXiv ID:** 2602.01566
</details>

<details>
<summary><strong>Wiki Live Challenge: Challenging Deep Research Agents with Expert-Level Wikipedia Articles</strong> - Shaohan Wang, Benfeng Xu, Licheng Zhang, Mingxuan Du, Chiwei Zhu, Xiaorui Wang, Zhendong Mao, Yongdong Zhang - [[pdf]](https://arxiv.org/pdf/2602.01590)</summary>

**Abstract:** Deep Research Agents (DRAs) have demonstrated remarkable capabilities in autonomous information retrieval and report generation, showing great potential to assist humans in complex research tasks. Current evaluation frameworks primarily rely on LLM-generated references or LLM-derived evaluation dimensions. While these approaches offer scalability, they often lack the reliability of expert-verified content and struggle to provide objective, fine-grained assessments of critical dimensions. To bridge this gap, we introduce Wiki Live Challenge (WLC), a live benchmark that leverages the newest Wikipedia Good Articles (GAs) as expert-level references. Wikipedia's strict standards for neutrality, comprehensiveness, and verifiability serve as a great challenge for DRAs, with GAs representing the pinnacle of which. We curate a dataset of 100 recent Good Articles and propose Wiki Eval, a comprehensive evaluation framework comprising a fine-grained evaluation method with 39 criteria for writing quality and rigorous metrics for factual verifiability. Extensive experiments on various DRA systems demonstrate a significant gap between current DRAs and human expert-level Wikipedia articles, validating the effectiveness of WLC in advancing agent research. We release our benchmark at this https URL

**arXiv ID:** 2602.01590
</details>

<details>
<summary><strong>ARTIS: Agentic Risk-Aware Test-Time Scaling via Iterative Simulation</strong> - Xingshan Zeng, Lingzhi Wang, Weiwen Liu, Liangyou Li, Yasheng Wang, Lifeng Shang, Xin Jiang, Qun Liu - [[pdf]](https://arxiv.org/pdf/2602.01709)</summary>

**Abstract:** Current test-time scaling (TTS) techniques enhance large language model (LLM) performance by allocating additional computation at inference time, yet they remain insufficient for agentic settings, where actions directly interact with external environments and their effects can be irreversible and costly. We propose ARTIS, Agentic Risk-Aware Test-Time Scaling via Iterative Simulation, a framework that decouples exploration from commitment by enabling test-time exploration through simulated interactions prior to real-world execution. This design allows extending inference-time computation to improve action-level reliability and robustness without incurring environmental risk. We further show that naive LLM-based simulators struggle to capture rare but high-impact failure modes, substantially limiting their effectiveness for agentic decision making. To address this limitation, we introduce a risk-aware tool simulator that emphasizes fidelity on failure-inducing actions via targeted data generation and rebalanced training. Experiments on multi-turn and multi-step agentic benchmarks demonstrate that iterative simulation substantially improves agent reliability, and that risk-aware simulation is essential for consistently realizing these gains across models and tasks.

**arXiv ID:** 2602.01709
</details>

<details>
<summary><strong>OmniCode: A Benchmark for Evaluating Software Engineering Agents</strong> - Atharv Sonwane, Eng-Shen Tu, Wei-Chung Lu, Claas Beger, Carter Larsen, Debjit Dhar, Rachel Chen, Ronit Pattanayak, Tuan Anh Dang, Guohao Chen, Gloria Geng, Kevin Ellis, Saikat Dutta - [[pdf]](https://arxiv.org/pdf/2602.02262)</summary>

**Abstract:** LLM-powered coding agents are redefining how real-world software is developed. To drive the research towards better coding agents, we require challenging benchmarks that can rigorously evaluate the ability of such agents to perform various software engineering tasks. However, popular coding benchmarks such as HumanEval and SWE-Bench focus on narrowly scoped tasks such as competition programming and patch generation. In reality, software engineers have to handle a broader set of tasks for real-world software development. To address this gap, we propose OmniCode, a novel software engineering benchmark that contains a broader and more diverse set of task categories beyond code or patch generation. Overall, OmniCode contains 1794 tasks spanning three programming languages (Python, Java, and C++) and four key categories: bug fixing, test generation, code review fixing, and style fixing. In contrast to prior software engineering benchmarks, the tasks in OmniCode are (1) manually validated to eliminate ill-defined problems, and (2) synthetically crafted or recently curated to avoid data leakage issues, presenting a new framework for synthetically generating diverse software tasks from limited real-world data. We evaluate OmniCode with popular agent frameworks such as SWE-Agent and show that while they may perform well on bug fixing for Python, they fall short on tasks such as Test Generation and in languages such as C++ and Java. For instance, SWE-Agent achieves a maximum of 20.9% with DeepSeek-V3.1 on Java Test Generation tasks. OmniCode aims to serve as a robust benchmark and spur the development of agents that can perform well across different aspects of software development. Code and data are available at this https URL.

**arXiv ID:** 2602.02262
</details>

<details>
<summary><strong>ELLMPEG: An Edge-based Agentic LLM Video Processing Tool</strong> - Zoha Azimi, Reza Farahani, Radu Prodan, Christian Timmerer - [[pdf]](https://arxiv.org/pdf/2602.00028)</summary>

**Abstract:** Large language models (LLMs), the foundation of generative AI systems like ChatGPT, are transforming many fields and applications, including multimedia, enabling more advanced content generation, analysis, and interaction. However, cloud-based LLM deployments face three key limitations: high computational and energy demands, privacy and reliability risks from remote processing, and recurring API costs. Recent advances in agentic AI, especially in structured reasoning and tool use, offer a better way to exploit open and locally deployed tools and LLMs. This paper presents ELLMPEG, an edge-enabled agentic LLM framework for the automated generation of video-processing commands. ELLMPEG integrates tool-aware Retrieval-Augmented Generation (RAG) with iterative self-reflection to produce and locally verify executable FFmpeg and VVenC commands directly at the edge, eliminating reliance on external cloud APIs. To evaluate ELLMPEG, we collect a dedicated prompt dataset comprising 480 diverse queries covering different categories of FFmpeg and the Versatile Video Codec (VVC) encoder (VVenC) commands. We validate command generation accuracy and evaluate four open-source LLMs based on command validity, tokens generated per second, inference time, and energy efficiency. We also execute the generated commands to assess their runtime correctness and practical applicability. Experimental results show that Qwen2.5, when augmented with the ELLMPEG framework, achieves an average command-generation accuracy of 78 % with zero recurring API cost, outperforming all other open-source models across both the FFmpeg and VVenC datasets.

**arXiv ID:** 2602.00028
</details>

<details>
<summary><strong>HERMES: A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision-Language Models for Long-Tail Autonomous Driving</strong> - Weizhe Tang, Junwei You, Jiaxi Liu, Zhaoyi Wang, Rui Gan, Zilin Huang, Feng Wei, Bin Ran - [[pdf]](https://arxiv.org/pdf/2602.00993)</summary>

**Abstract:** End-to-end autonomous driving models increasingly benefit from large vision--language models for semantic understanding, yet ensuring safe and accurate operation under long-tail conditions remains challenging. These challenges are particularly prominent in long-tail mixed-traffic scenarios, where autonomous vehicles must interact with heterogeneous road users, including human-driven vehicles and vulnerable road users, under complex and uncertain conditions. This paper proposes HERMES, a holistic risk-aware end-to-end multimodal driving framework designed to inject explicit long-tail risk cues into trajectory planning. HERMES employs a foundation-model-assisted annotation pipeline to produce structured Long-Tail Scene Context and Long-Tail Planning Context, capturing hazard-centric cues together with maneuver intent and safety preference, and uses these signals to guide end-to-end planning. HERMES further introduces a Tri-Modal Driving Module that fuses multi-view perception, historical motion cues, and semantic guidance, ensuring risk-aware accurate trajectory planning under long-tail scenarios. Experiments on the real-world long-tail dataset demonstrate that HERMES consistently outperforms representative end-to-end and VLM-driven baselines under long-tail mixed-traffic scenarios. Ablation studies verify the complementary contributions of key components.

**arXiv ID:** 2602.00993
</details>

<details>
<summary><strong>Towards Autonomous Instrument Tray Assembly for Sterile Processing Applications</strong> - Raghavasimhan Sankaranarayanan, Paul Stuart, Nicholas Ahn, Arno Sungarian, Yash Chitalia - [[pdf]](https://arxiv.org/pdf/2602.01679)</summary>

**Abstract:** The Sterile Processing and Distribution (SPD) department is responsible for cleaning, disinfecting, inspecting, and assembling surgical instruments between surgeries. Manual inspection and preparation of instrument trays is a time-consuming, error-prone task, often prone to contamination and instrument breakage. In this work, we present a fully automated robotic system that sorts and structurally packs surgical instruments into sterile trays, focusing on automation of the SPD assembly stage. A custom dataset comprising 31 surgical instruments and 6,975 annotated images was collected to train a hybrid perception pipeline using YOLO12 for detection and a cascaded ResNet-based model for fine-grained classification. The system integrates a calibrated vision module, a 6-DOF Staubli TX2-60L robotic arm with a custom dual electromagnetic gripper, and a rule-based packing algorithm that reduces instrument collisions during transport. The packing framework uses 3D printed dividers and holders to physically isolate instruments, reducing collision and friction during transport. Experimental evaluations show high perception accuracy and statistically significant reduction in tool-to-tool collisions compared to human-assembled trays. This work serves as the scalable first step toward automating SPD workflows, improving safety, and consistency of surgical preparation while reducing SPD processing times.

**arXiv ID:** 2602.01679
</details>

<details>
<summary><strong>Robust Trajectory Tracking of Autonomous Surface Vehicle via Lie Algebraic Online MPC</strong> - Yinan Dong, Ziyu Xu, Tsimafei Lazouski, Sangli Teng, Maani Ghaffari - [[pdf]](https://arxiv.org/pdf/2511.18683)</summary>

**Abstract:** Autonomous surface vehicles (ASVs) are influenced by environmental disturbances such as wind and waves, making accurate trajectory tracking a persistent challenge in dynamic marine conditions. In this paper, we propose an efficient controller for trajectory tracking of marine vehicles under unknown disturbances by combining a convex error-state MPC on the Lie group augmented by an online learning module to compensate for these disturbances in real time. This design enables adaptive and robust tracking control while maintaining computational efficiency. Extensive evaluations in the Virtual RobotX (VRX) simulator, and real-world field experiments demonstrate that our method achieves superior tracking accuracy under various disturbance scenarios compared with existing approaches.

**arXiv ID:** 2511.18683
</details>

<details>
<summary><strong>DSCD-Nav: Dual-Stance Cooperative Debate for Object Navigation</strong> - Weitao An, Qi Liu, Chenghao Xu, Jiayi Chai, Xu Yang, Kun Wei, Cheng Deng - [[pdf]](https://arxiv.org/pdf/2601.21409)</summary>

**Abstract:** Adaptive navigation in unfamiliar indoor environments is crucial for household service robots. Despite advances in zero-shot perception and reasoning from vision-language models, existing navigation systems still rely on single-pass scoring at the decision layer, leading to overconfident long-horizon errors and redundant exploration. To tackle these problems, we propose Dual-Stance Cooperative Debate Navigation (DSCD-Nav), a decision mechanism that replaces one-shot scoring with stance-based cross-checking and evidence-aware arbitration to improve action reliability under partial observability. Specifically, given the same observation and candidate action set, we explicitly construct two stances by conditioning the evaluation on diverse and complementary objectives: a Task-Scene Understanding (TSU) stance that prioritizes goal progress from scene-layout cues, and a Safety-Information Balancing (SIB) stance that emphasizes risk and information value. The stances conduct a cooperative debate and make policy by cross-checking their top candidates with cue-grounded arguments. Then, a Navigation Consensus Arbitration (NCA) agent is employed to consolidate both sides' reasons and evidence, optionally triggering lightweight micro-probing to verify uncertain choices, preserving NCA's primary intent while disambiguating. Experiments on HM3Dv1, HM3Dv2, and MP3D demonstrate consistent improvements in success and path efficiency while reducing exploration redundancy.

**arXiv ID:** 2601.21409
</details>

<details>
<summary><strong>CostNav: A Navigation Benchmark for Real-World Economic-Cost Evaluation of Physical AI Agents</strong> - Haebin Seong, Sungmin Kim, Yongjun Cho, Myunchul Joe, Geunwoo Kim, Yubeen Park, Sunhoo Kim, Yoonshik Kim, Suhwan Choi, Jaeyoon Jung, Jiyong Youn, Jinmyung Kwak, Sunghee Ahn, Jaemin Lee, Younggil Do, Seungyeop Yi, Woojin Cheong, Minhyeok Oh, Minchan Kim, Yoonseok Kang, Seongjae Kang, Samwoo Seong, Youngjae Yu, Yunsung Lee - [[pdf]](https://arxiv.org/pdf/2511.20216)</summary>

**Abstract:** While current navigation benchmarks prioritize task success in simplified settings, they neglect the multidimensional economic constraints essential for the real-world commercialization of autonomous delivery systems. We introduce CostNav, an Economic Navigation Benchmark that evaluates physical AI agents through comprehensive economic cost-revenue analysis aligned with real-world business operations. By integrating industry-standard data - such as SEC filings and AIS injury reports - with Isaac Sim's detailed collision and cargo dynamics, CostNav transcends simple task completion to accurately evaluate business value in complex, real-world scenarios. To our knowledge, CostNav is the first work to quantitatively expose the gap between navigation research metrics and commercial viability, revealing that optimizing for task success on a simplified task fundamentally differs from optimizing for real-world economic deployment. Our evaluation of rule-based Nav2 navigation shows that current approaches are not economically viable: the contribution margin is -22.81/run (AMCL) and -12.87/run (GPS), resulting in no break-even point. We challenge the community to develop navigation policies that achieve economic viability on CostNav. We remain method-agnostic, evaluating success solely on the metric of cost rather than the underlying architecture. All resources are available at this https URL.

**arXiv ID:** 2511.20216
</details>

</details>

<details open>
<summary><h2>LLM Agents (15 papers)</h2></summary>

<details>
<summary><strong>SEISMO: Increasing Sample Efficiency in Molecular Optimization with a Trajectory-Aware LLM Agent</strong> - Fabian P. Krüger, Andrea Hunklinger, Adrian Wolny, Tim J. Adler, Igor Tetko, Santiago David Villalba - [[pdf]](https://arxiv.org/pdf/2602.00663)</summary>

**Abstract:** Optimizing the structure of molecules to achieve desired properties is a central bottleneck across the chemical sciences, particularly in the pharmaceutical industry where it underlies the discovery of new drugs. Since molecular property evaluation often relies on costly and rate-limited oracles, such as experimental assays, molecular optimization must be highly sample-efficient. To address this, we introduce SEISMO, an LLM agent that performs strictly online, inference-time molecular optimization, updating after every oracle call without the need for population-based or batched learning. SEISMO conditions each proposal on the full optimization trajectory, combining natural-language task descriptions with scalar scores and, when available, structured explanatory feedback. Across the Practical Molecular Optimization benchmark of 23 tasks, SEISMO achieves a 2-3 times higher area under the optimisation curve than prior methods, often reaching near-maximal task scores within 50 oracle calls. Our additional medicinal-chemistry tasks show that providing explanatory feedback further improves efficiency, demonstrating that leveraging domain knowledge and structured information is key to sample-efficient molecular optimization.

**arXiv ID:** 2602.00663
</details>

<details>
<summary><strong>Persuasion Propagation in LLM Agents</strong> - Hyejun Jeong, Amir Houmansadr, Shlomo Zilberstein, Eugene Bagdasarian - [[pdf]](https://arxiv.org/pdf/2602.00851)</summary>

**Abstract:** Modern AI agents increasingly combine conversational interaction with autonomous task execution, such as coding and web research, raising a natural question: what happens when an agent engaged in long-horizon tasks is subjected to user persuasion? We study how belief-level intervention can influence downstream task behavior, a phenomenon we name \emph{persuasion propagation}. We introduce a behavior-centered evaluation framework that distinguishes between persuasion applied during or prior to task execution. Across web research and coding tasks, we find that on-the-fly persuasion induces weak and inconsistent behavioral effects. In contrast, when the belief state is explicitly specified at task time, belief-prefilled agents conduct on average 26.9\% fewer searches and visit 16.9\% fewer unique sources than neutral-prefilled agents. These results suggest that persuasion, even in prior interaction, can affect the agent's behavior, motivating behavior-level evaluation in agentic systems.

**arXiv ID:** 2602.00851
</details>

<details>
<summary><strong>Learning Abstractions for Hierarchical Planning in Program-Synthesis Agents</strong> - Zergham Ahmed, Kazuki Irie, Joshua B. Tenenbaum, Christopher J. Bates, Samuel J. Gershman - [[pdf]](https://arxiv.org/pdf/2602.00929)</summary>

**Abstract:** Humans learn abstractions and use them to plan efficiently to quickly generalize across tasks -- an ability that remains challenging for state-of-the-art large language model (LLM) agents and deep reinforcement learning (RL) systems. Inspired by the cognitive science of how people form abstractions and intuitive theories of their world knowledge, Theory-Based RL (TBRL) systems, such as TheoryCoder, exhibit strong generalization through effective use of abstractions. However, they heavily rely on human-provided abstractions and sidestep the abstraction-learning problem. We introduce TheoryCoder-2, a new TBRL agent that leverages LLMs' in-context learning ability to actively learn reusable abstractions rather than relying on hand-specified ones, by synthesizing abstractions from experience and integrating them into a hierarchical planning process. We conduct experiments on diverse environments, including BabyAI, Minihack and VGDL games like Sokoban. We find that TheoryCoder-2 is significantly more sample-efficient than baseline LLM agents augmented with classical planning domain construction, reasoning-based planning, and prior program-synthesis agents such as WorldCoder. TheoryCoder-2 is able to solve complex tasks that the baselines fail, while only requiring minimal human prompts, unlike prior TBRL systems.

**arXiv ID:** 2602.00929
</details>

<details>
<summary><strong>SimGym: Traffic-Grounded Browser Agents for Offline A/B Testing in E-Commerce</strong> - Alberto Castelo, Zahra Zanjani Foumani, Ailin Fan, Keat Yang Koay, Vibhor Malik, Yuanzheng Zhu, Han Li, Meysam Feghhi, Ronie Uliana, Shuang Xie, Zhaoyu Zhang, Angelo Ocana Martins, Mingyu Zhao, Francis Pelland, Jonathan Faerman, Nikolas LeBlanc, Aaron Glazer, Andrew McNamara, Lingyun Wang, Zhong Wu - [[pdf]](https://arxiv.org/pdf/2602.01443)</summary>

**Abstract:** A/B testing remains the gold standard for evaluating e-commerce UI changes, yet it diverts traffic, takes weeks to reach significance, and risks harming user experience. We introduce SimGym, a scalable system for rapid offline A/B testing using traffic-grounded synthetic buyers powered by Large Language Model agents operating in a live browser. SimGym extracts per-shop buyer profiles and intents from production interaction data, identifies distinct behavioral archetypes, and simulates cohort-weighted sessions across control and treatment storefronts. We validate SimGym against real human outcomes from real UI changes on a major e-commerce platform under confounder control. Even without alignment post training, SimGym agents achieve state of the art alignment with observed outcome shifts and reduces experiment cycles from weeks to under an hour , enabling rapid experimentation without exposure to real buyers.

**arXiv ID:** 2602.01443
</details>

<details>
<summary><strong>ProcMEM: Learning Reusable Procedural Memory from Experience via Non-Parametric PPO for LLM Agents</strong> - Qirui Mi, Zhijian Ma, Mengyue Yang, Haoxuan Li, Yisen Wang, Haifeng Zhang, Jun Wang - [[pdf]](https://arxiv.org/pdf/2602.01869)</summary>

**Abstract:** LLM-driven agents demonstrate strong performance in sequential decision-making but often rely on on-the-fly reasoning, re-deriving solutions even in recurring scenarios. This insufficient experience reuse leads to computational redundancy and execution instability. To bridge this gap, we propose ProcMEM, a framework that enables agents to autonomously learn procedural memory from interaction experiences without parameter updates. By formalizing a Skill-MDP, ProcMEM transforms passive episodic narratives into executable Skills defined by activation, execution, and termination conditions to ensure executability. To achieve reliable reusability without capability degradation, we introduce Non-Parametric PPO, which leverages semantic gradients for high-quality candidate generation and a PPO Gate for robust Skill verification. Through score-based maintenance, ProcMEM sustains compact, high-quality procedural memory. Experimental results across in-domain, cross-task, and cross-agent scenarios demonstrate that ProcMEM achieves superior reuse rates and significant performance gains with extreme memory compression. Visualized evolutionary trajectories and Skill distributions further reveal how ProcMEM transparently accumulates, refines, and reuses procedural knowledge to facilitate long-term autonomy.

**arXiv ID:** 2602.01869
</details>

<details>
<summary><strong>Rethinking the Role of Entropy in Optimizing Tool-Use Behaviors for Large Language Model Agents</strong> - Zeping Li, Hongru Wang, Yiwen Zhao, Guanhua Chen, Yixia Li, Keyang Chen, Yixin Cao, Guangnan Ye, Hongfeng Chai, Mengdi Wang, Zhenfei Yin - [[pdf]](https://arxiv.org/pdf/2602.02050)</summary>

**Abstract:** Tool-using agents based on Large Language Models (LLMs) excel in tasks such as mathematical reasoning and multi-hop question answering. However, in long trajectories, agents often trigger excessive and low-quality tool calls, increasing latency and degrading inference performance, making managing tool-use behavior challenging. In this work, we conduct entropy-based pilot experiments and observe a strong positive correlation between entropy reduction and high-quality tool calls. Building on this finding, we propose using entropy reduction as a supervisory signal and design two reward strategies to address the differing needs of optimizing tool-use behavior. Sparse outcome rewards provide coarse, trajectory-level guidance to improve efficiency, while dense process rewards offer fine-grained supervision to enhance performance. Experiments across diverse domains show that both reward designs improve tool-use behavior: the former reduces tool calls by 72.07% compared to the average of baselines, while the latter improves performance by 22.27%. These results position entropy reduction as a key mechanism for enhancing tool-use behavior, enabling agents to be more adaptive in real-world applications.

**arXiv ID:** 2602.02050
</details>

<details>
<summary><strong>TIDE: Trajectory-based Diagnostic Evaluation of Test-Time Improvement in LLM Agents</strong> - Hang Yan, Xinyu Che, Fangzhi Xu, Qiushi Sun, Zichen Ding, Kanzhi Cheng, Jian Zhang, Tao Qin, Jun Liu, Qika Lin - [[pdf]](https://arxiv.org/pdf/2602.02196)</summary>

**Abstract:** Recent advances in autonomous LLM agents demonstrate their ability to improve performance through iterative interaction with the environment. We define this paradigm as Test-Time Improvement (TTI). However, the mechanisms under how and why TTI succeed or fail remain poorly understood, and existing evaluation metrics fail to capture their task optimization efficiency, behavior adaptation after erroneous actions, and the specific utility of working memory for task completion. To address these gaps, we propose Test-time Improvement Diagnostic Evaluation (TIDE), an agent-agnostic and environment-agnostic framework that decomposes TTI into three comprehensive and interconnected dimensions. The framework measures (1) the overall temporal dynamics of task completion and (2) identifies whether performance is primarily constrained by recursive looping behaviors or (3) by burdensome accumulated memory. Through extensive experiments across diverse agents and environments, TIDE highlights that improving agent performance requires more than scaling internal reasoning, calling for explicitly optimizing the interaction dynamics between the agent and the environment.

**arXiv ID:** 2602.02196
</details>

<details>
<summary><strong>Drift-Bench: Diagnosing Cooperative Breakdowns in LLM Agents under Input Faults via Multi-Turn Interaction</strong> - Han Bao, Zheyuan Zhang, Pengcheng Jing, Zhengqing Yuan, Kaiwen Shi, Yanfang Ye - [[pdf]](https://arxiv.org/pdf/2602.02455)</summary>

**Abstract:** As Large Language Models transition to autonomous agents, user inputs frequently violate cooperative assumptions (e.g., implicit intent, missing parameters, false presuppositions, or ambiguous expressions), creating execution risks that text-only evaluations do not capture. Existing benchmarks typically assume well-specified instructions or restrict evaluation to text-only, single-turn clarification, and thus do not measure multi-turn disambiguation under grounded execution risk. We introduce \textbf{Drift-Bench}, the first diagnostic benchmark that evaluates agentic pragmatics under input faults through multi-turn clarification across state-oriented and service-oriented execution environments. Grounded in classical theories of communication, \textbf{Drift-Bench} provides a unified taxonomy of cooperative breakdowns and employs a persona-driven user simulator with the \textbf{Rise} evaluation protocol. Experiments show substantial performance drops under these faults, with clarification effectiveness varying across user personas and fault types. \MethodName bridges clarification research and agent safety evaluation, enabling systematic diagnosis of failures that can lead to unsafe executions.

**arXiv ID:** 2602.02455
</details>

<details>
<summary><strong>PredictionMarketBench: A SWE-bench-Style Framework for Backtesting Trading Agents on Prediction Markets</strong> - Avi Arora, Ritesh Malpani - [[pdf]](https://arxiv.org/pdf/2602.00133)</summary>

**Abstract:** Prediction markets offer a natural testbed for trading agents: contracts have binary payoffs, prices can be interpreted as probabilities, and realized performance depends critically on market microstructure, fees, and settlement risk. We introduce PredictionMarketBench, a SWE-bench-style benchmark for evaluating algorithmic and LLM-based trading agents on prediction markets via deterministic, event-driven replay of historical limit-order-book and trade data. PredictionMarketBench standardizes (i) episode construction from raw exchange streams (orderbooks, trades, lifecycle, settlement), (ii) an execution-realistic simulator with maker/taker semantics and fee modeling, and (iii) a tool-based agent interface that supports both classical strategies and tool-calling LLM agents with reproducible trajectories. We release four Kalshi-based episodes spanning cryptocurrency, weather, and sports. Baseline results show that naive trading agents can underperform due to transaction costs and settlement losses, while fee-aware algorithmic strategies remain competitive in volatile episodes.

**arXiv ID:** 2602.00133
</details>

<details>
<summary><strong>Towards Agentic Intelligence for Materials Science</strong> - Huan Zhang, Yizhan Li, Wenhao Huang, Ziyu Hou, Yu Song, Xuye Liu, Farshid Effaty, Jinya Jiang, Sifan Wu, Qianggang Ding, Izumi Takahara, Leonard R. MacGillivray, Teruyasu Mizoguchi, Tianshu Yu, Lizi Liao, Yuyu Luo, Yu Rong, Jia Li, Ying Diao, Heng Ji, Bang Liu - [[pdf]](https://arxiv.org/pdf/2602.00169)</summary>

**Abstract:** The convergence of artificial intelligence and materials science presents a transformative opportunity, but achieving true acceleration in discovery requires moving beyond task-isolated, fine-tuned models toward agentic systems that plan, act, and learn across the full discovery loop. This survey advances a unique pipeline-centric view that spans from corpus curation and pretraining, through domain adaptation and instruction tuning, to goal-conditioned agents interfacing with simulation and experimental platforms. Unlike prior reviews, we treat the entire process as an end-to-end system to be optimized for tangible discovery outcomes rather than proxy benchmarks. This perspective allows us to trace how upstream design choices-such as data curation and training objectives-can be aligned with downstream experimental success through effective credit assignment.
To bridge communities and establish a shared frame of reference, we first present an integrated lens that aligns terminology, evaluation, and workflow stages across AI and materials science. We then analyze the field through two focused lenses: From the AI perspective, the survey details LLM strengths in pattern recognition, predictive analytics, and natural language processing for literature mining, materials characterization, and property prediction; from the materials science perspective, it highlights applications in materials design, process optimization, and the acceleration of computational workflows via integration with external tools (e.g., DFT, robotic labs). Finally, we contrast passive, reactive approaches with agentic design, cataloging current contributions while motivating systems that pursue long-horizon goals with autonomy, memory, and tool use. This survey charts a practical roadmap towards autonomous, safety-aware LLM agents aimed at discovering novel and useful materials.

**arXiv ID:** 2602.00169
</details>

<details>
<summary><strong>EffGen: Enabling Small Language Models as Capable Autonomous Agents</strong> - Gaurav Srivastava, Aafiya Hussain, Chi Wang, Yingyan Celine Lin, Xuan Wang - [[pdf]](https://arxiv.org/pdf/2602.00887)</summary>

**Abstract:** Most existing language model agentic systems today are built and optimized for large language models (e.g., GPT, Claude, Gemini) via API calls. While powerful, this approach faces several limitations including high token costs and privacy concerns for sensitive applications. We introduce effGen, an open-source agentic framework optimized for small language models (SLMs) that enables effective, efficient, and secure local deployment (pip install effgen). effGen makes four major contributions: (1) Enhanced tool-calling with prompt optimization that compresses contexts by 70-80% while preserving task semantics, (2) Intelligent task decomposition that breaks complex queries into parallel or sequential subtasks based on dependencies, (3) Complexity-based routing using five factors to make smart pre-execution decisions, and (4) Unified memory system combining short-term, long-term, and vector-based storage. Additionally, effGen unifies multiple agent protocols (MCP, A2A, ACP) for cross-protocol communication. Results on 13 benchmarks show effGen outperforms LangChain, AutoGen, and Smolagents with higher success rates, faster execution, and lower memory. Our results reveal that prompt optimization and complexity routing have complementary scaling behavior: optimization benefits SLMs more (11.2% gain at 1.5B vs 2.4% at 32B), while routing benefits large models more (3.6% at 1.5B vs 7.9% at 32B), providing consistent gains across all scales when combined. effGen (this https URL) is released under the MIT License, ensuring broad accessibility for research and commercial use. Our framework code is publicly available at this https URL.

**arXiv ID:** 2602.00887
</details>

<details>
<summary><strong>Personality Expression Across Contexts: Linguistic and Behavioral Variation in LLM Agents</strong> - Bin Han, Deuksin Kwon, Jonathan Gratch - [[pdf]](https://arxiv.org/pdf/2602.01063)</summary>

**Abstract:** Large Language Models (LLMs) can be conditioned with explicit personality prompts, yet their behavioral realization often varies depending on context. This study examines how identical personality prompts lead to distinct linguistic, behavioral, and emotional outcomes across four conversational settings: ice-breaking, negotiation, group decision, and empathy tasks. Results show that contextual cues systematically influence both personality expression and emotional tone, suggesting that the same traits are expressed differently depending on social and affective demands. This raises an important question for LLM-based dialogue agents: whether such variations reflect inconsistency or context-sensitive adaptation akin to human behavior. Viewed through the lens of Whole Trait Theory, these findings highlight that LLMs exhibit context-sensitive rather than fixed personality expression, adapting flexibly to social interaction goals and affective conditions.

**arXiv ID:** 2602.01063
</details>

<details>
<summary><strong>MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents</strong> - Haozhen Zhang, Quanyu Long, Jianzhu Bao, Tao Feng, Weizhi Zhang, Haodong Yue, Wenya Wang - [[pdf]](https://arxiv.org/pdf/2602.02474)</summary>

**Abstract:** Most Large Language Model (LLM) agent memory systems rely on a small set of static, hand-designed operations for extracting memory. These fixed procedures hard-code human priors about what to store and how to revise memory, making them rigid under diverse interaction patterns and inefficient on long histories. To this end, we present \textbf{MemSkill}, which reframes these operations as learnable and evolvable memory skills, structured and reusable routines for extracting, consolidating, and pruning information from interaction traces. Inspired by the design philosophy of agent skills, MemSkill employs a \emph{controller} that learns to select a small set of relevant skills, paired with an LLM-based \emph{executor} that produces skill-guided memories. Beyond learning skill selection, MemSkill introduces a \emph{designer} that periodically reviews hard cases where selected skills yield incorrect or incomplete memories, and evolves the skill set by proposing refinements and new skills. Together, MemSkill forms a closed-loop procedure that improves both the skill-selection policy and the skill set itself. Experiments on LoCoMo, LongMemEval, HotpotQA, and ALFWorld demonstrate that MemSkill improves task performance over strong baselines and generalizes well across settings. Further analyses shed light on how skills evolve, offering insights toward more adaptive, self-evolving memory management for LLM agents.

**arXiv ID:** 2602.02474
</details>

<details>
<summary><strong>LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents</strong> - Hyesung Jeon, Hyeongju Ha, Jae-Joon Kim - [[pdf]](https://arxiv.org/pdf/2602.01053)</summary>

**Abstract:** Role specialization in multi-LLM agent systems is often realized via multi-LoRA, where agents share a pretrained backbone and differ only through lightweight adapters. Despite sharing base model weights, each agent independently builds and stores its own KV cache for the same long, tool-augmented trajectories, incurring substantial memory and compute overhead. Existing KV cache sharing methods largely overlook this multi-LoRA setting. We observe that, across agents, cache differences are dominated by adapter outputs, while activations from the shared pretrained backbone remain highly similar. Based on this observation, we propose LRAgent, a KV cache sharing framework for multi-LoRA agents that decomposes the cache into a shared base component from the pretrained weights and an adapter-dependent component from LoRA weights. LRAgent reduces memory overhead by sharing the base component and storing the adapter component in its inherent low-rank form, and further reduces compute overhead, enabled by shared-$A$ multi-LoRA architectures, by also sharing the low-rank cache and avoiding redundant computations for contexts already processed by other agents. To efficiently reconstruct adapter contributions at runtime, we introduce Flash-LoRA-Attention, a kernel that reorders attention computation to avoid materializing the low-rank cache to full dimension. LRAgent achieves throughput and time-to-first-token latency close to fully shared caching, while preserving accuracy near the non-shared caching baseline across agentic question-answering benchmarks.

**arXiv ID:** 2602.01053
</details>

<details>
<summary><strong>"Shall We Dig Deeper?": Designing and Evaluating Strategies for LLM Agents to Advance Knowledge Co-Construction in Asynchronous Online Discussions</strong> - Yuanhao Zhang, Wenbo Li, Xiaoyu Wang, Kangyu Yuan, Shuai Ma, Xiaojuan Ma - [[pdf]](https://arxiv.org/pdf/2509.23327)</summary>

**Abstract:** Asynchronous online discussions enable diverse participants to co-construct knowledge beyond individual contributions. This process ideally evolves through sequential phases, from superficial information exchange to deeper synthesis. However, many discussions stagnate in the early stages. Existing AI interventions typically target isolated phases, lacking mechanisms to progressively advance knowledge co-construction, and the impacts of different intervention styles in this context remain unclear and warrant investigation. To address these gaps, we conducted a design workshop to explore AI intervention strategies (task-oriented and/or relationship-oriented) throughout the knowledge co-construction process, and implemented them in an LLM-powered agent capable of facilitating progression while consolidating foundations at each phase. A within-subject study (N=60) involving five consecutive asynchronous discussions showed that the agent consistently promoted deeper knowledge progression, with different styles exerting distinct effects on both content and experience. These findings provide actionable guidance for designing adaptive AI agents that sustain more constructive online discussions.

**arXiv ID:** 2509.23327
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (42 papers)</h2></summary>

<details>
<summary><strong>Cross-Modal Memory Compression for Efficient Multi-Agent Debate</strong> - Jing Wu, Yue Sun, Tianpei Xie, Suiyao Chen, Jingyuan Bao, Yaopengxiao Xu, Gaoyuan Du, Inseok Heo, Alexander Gutfraind, Xin Wang - [[pdf]](https://arxiv.org/pdf/2602.00454)</summary>

**Abstract:** Multi-agent debate can improve reasoning quality and reduce hallucinations, but it incurs rapidly growing context as debate rounds and agent count increase. Retaining full textual histories leads to token usage that can exceed context limits and often requires repeated summarization, adding overhead and compounding information loss. We introduce DebateOCR, a cross-modal compression framework that replaces long textual debate traces with compact image representations, which are then consumed through a dedicated vision encoder to condition subsequent rounds. This design compresses histories that commonly span tens to hundreds of thousands of tokens, cutting input tokens by more than 92% and yielding substantially lower compute cost and faster inference across multiple benchmarks. We further provide a theoretical perspective showing that diversity across agents supports recovery of omitted information: although any single compressed history may discard details, aggregating multiple agents' compressed views allows the collective representation to approach the information bottleneck with exponentially high probability.

**arXiv ID:** 2602.00454
</details>

<details>
<summary><strong>Dual Latent Memory for Visual Multi-agent System</strong> - Xinlei Yu, Chengming Xu, Zhangquan Chen, Bo Yin, Cheng Yang, Yongbo He, Yihao Hu, Jiangning Zhang, Cheng Tan, Xiaobin Hu, Shuicheng Yan - [[pdf]](https://arxiv.org/pdf/2602.00471)</summary>

**Abstract:** While Visual Multi-Agent Systems (VMAS) promise to enhance comprehensive abilities through inter-agent collaboration, empirical evidence reveals a counter-intuitive "scaling wall": increasing agent turns often degrades performance while exponentially inflating token costs. We attribute this failure to the information bottleneck inherent in text-centric communication, where converting perceptual and thinking trajectories into discrete natural language inevitably induces semantic loss. To this end, we propose L$^{2}$-VMAS, a novel model-agnostic framework that enables inter-agent collaboration with dual latent memories. Furthermore, we decouple the perception and thinking while dynamically synthesizing dual latent memories. Additionally, we introduce an entropy-driven proactive triggering that replaces passive information transmission with efficient, on-demand memory access. Extensive experiments among backbones, sizes, and multi-agent structures demonstrate that our method effectively breaks the "scaling wall" with superb scalability, improving average accuracy by 2.7-5.4% while reducing token usage by 21.3-44.8%. Codes: this https URL.

**arXiv ID:** 2602.00471
</details>

<details>
<summary><strong>Reasoning and Tool-use Compete in Agentic RL:From Quantifying Interference to Disentangled Tuning</strong> - Yu Li, Mingyang Yi, Xiuyu Li, Ju Fan, Fuxin Jiang, Binbin Chen, Peng Li, Jie Song, Tieying Zhang - [[pdf]](https://arxiv.org/pdf/2602.00994)</summary>

**Abstract:** Agentic Reinforcement Learning (ARL) focuses on training large language models (LLMs) to interleave reasoning with external tool execution to solve complex tasks. Most existing ARL methods train a single shared model parameters to support both reasoning and tool use behaviors, implicitly assuming that joint training leads to improved overall agent performance. Despite its widespread adoption, this assumption has rarely been examined empirically. In this paper, we systematically investigate this assumption by introducing a Linear Effect Attribution System(LEAS), which provides quantitative evidence of interference between reasoning and tool-use behaviors. Through an in-depth analysis, we show that these two capabilities often induce misaligned gradient directions, leading to training interference that undermines the effectiveness of joint optimization and challenges the prevailing ARL paradigm. To address this issue, we propose Disentangled Action Reasoning Tuning(DART), a simple and efficient framework that explicitly decouples parameter updates for reasoning and tool-use via separate low-rank adaptation modules. Experimental results show that DART consistently outperforms baseline methods with averaged 6.35 percent improvements and achieves performance comparable to multi-agent systems that explicitly separate tool-use and reasoning using a single model.

**arXiv ID:** 2602.00994
</details>

<details>
<summary><strong>AutoHealth: An Uncertainty-Aware Multi-Agent System for Autonomous Health Data Modeling</strong> - Tong Xia, Weibin Li, Gang Liu, Yong Li - [[pdf]](https://arxiv.org/pdf/2602.01078)</summary>

**Abstract:** LLM-based agents have demonstrated strong potential for autonomous machine learning, yet their applicability to health data remains limited. Existing systems often struggle to generalize across heterogeneous health data modalities, rely heavily on predefined solution templates with insufficient adaptation to task-specific objectives, and largely overlook uncertainty estimation, which is essential for reliable decision-making in healthcare. To address these challenges, we propose \textit{AutoHealth}, a novel uncertainty-aware multi-agent system that autonomously models health data and assesses model reliability. \textit{AutoHealth} employs closed-loop coordination among five specialized agents to perform data exploration, task-conditioned model construction, training, and optimization, while jointly prioritizing predictive performance and uncertainty quantification. Beyond producing ready-to-use models, the system generates comprehensive reports to support trustworthy interpretation and risk-aware decision-making. To rigorously evaluate its effectiveness, we curate a challenging real-world benchmark comprising 17 tasks across diverse data modalities and learning settings. \textit{AutoHealth} completes all tasks and outperforms state-of-the-art baselines by 29.2\% in prediction performance and 50.2\% in uncertainty estimation.

**arXiv ID:** 2602.01078
</details>

<details>
<summary><strong>Multi-Agent Causal Reasoning System for Error Pattern Rule Automation in Vehicles</strong> - Hugo Math, Julian Lorenz, Stefan Oelsner, Rainer Lienhart - [[pdf]](https://arxiv.org/pdf/2602.01155)</summary>

**Abstract:** Modern vehicles generate thousands of different discrete events known as Diagnostic Trouble Codes (DTCs). Automotive manufacturers use Boolean combinations of these codes, called error patterns (EPs), to characterize system faults and ensure vehicle safety. Yet, EP rules are still manually handcrafted by domain experts, a process that is expensive and prone to errors as vehicle complexity grows. This paper introduces CAREP (Causal Automated Reasoning for Error Patterns), a multi-agent system that automatizes the generation of EP rules from high-dimensional event sequences of DTCs. CAREP combines a causal discovery agent that identifies potential DTC-EP relations, a contextual information agent that integrates metadata and descriptions, and an orchestrator agent that synthesizes candidate boolean rules together with interpretable reasoning traces. Evaluation on a large-scale automotive dataset with over 29,100 unique DTCs and 474 error patterns demonstrates that CAREP can automatically and accurately discover the unknown EP rules, outperforming LLM-only baselines while providing transparent causal explanations. By uniting practical causal discovery and agent-based reasoning, CAREP represents a step toward fully automated fault diagnostics, enabling scalable, interpretable, and cost-efficient vehicle maintenance.

**arXiv ID:** 2602.01155
</details>

<details>
<summary><strong>Agyn: A Multi-Agent System for Team-Based Autonomous Software Engineering</strong> - Nikita Benkovich, Vitalii Valkov - [[pdf]](https://arxiv.org/pdf/2602.01465)</summary>

**Abstract:** Large language models have demonstrated strong capabilities in individual software engineering tasks, yet most autonomous systems still treat issue resolution as a monolithic or pipeline-based process. In contrast, real-world software development is organized as a collaborative activity carried out by teams following shared methodologies, with clear role separation, communication, and review. In this work, we present a fully automated multi-agent system that explicitly models software engineering as an organizational process, replicating the structure of an engineering team. Built on top of agyn, an open-source platform for configuring agent teams, our system assigns specialized agents to roles such as coordination, research, implementation, and review, provides them with isolated sandboxes for experimentation, and enables structured communication. The system follows a defined development methodology for working on issues, including analysis, task specification, pull request creation, and iterative review, and operates without any human intervention. Importantly, the system was designed for real production use and was not tuned for SWE-bench. When evaluated post hoc on SWE-bench 500, it resolves 72.4% of tasks, outperforming single-agent baselines using comparable language models. Our results suggest that replicating team structure, methodology, and communication is a powerful paradigm for autonomous software engineering, and that future progress may depend as much on organizational design and agent infrastructure as on model improvements.

**arXiv ID:** 2602.01465
</details>

<details>
<summary><strong>Autonomous Question Formation for Large Language Model-Driven AI Systems</strong> - Hong Su - [[pdf]](https://arxiv.org/pdf/2602.01556)</summary>

**Abstract:** Large language model (LLM)-driven AI systems are increasingly important for autonomous decision-making in dynamic and open environments. However, most existing systems rely on predefined tasks and fixed prompts, limiting their ability to autonomously identify what problems should be solved when environmental conditions change. In this paper, we propose a human-simulation-based framework that enables AI systems to autonomously form questions and set tasks by reasoning over their internal states, environmental observations, and interactions with other AI systems. The proposed method treats question formation as a first-class decision process preceding task selection and execution, and integrates internal-driven, environment-aware, and inter-agent-aware prompting scopes to progressively expand cognitive coverage. In addition, the framework supports learning the question-formation process from experience, allowing the system to improve its adaptability and decision quality over time. xperimental results in a multi-agent simulation environment show that environment-aware prompting significantly reduces no-eat events compared with the internal-driven baseline, and inter-agent-aware prompting further reduces cumulative no-eat events by more than 60% over a 20-day simulation, with statistically significant improvements (p < 0.05).

**arXiv ID:** 2602.01556
</details>

<details>
<summary><strong>ORCH: many analyses, one merge-a deterministic multi-agent orchestrator for discrete-choice reasoning with EMA-guided routing</strong> - Hanlin Zhou, Huah Yong Chan - [[pdf]](https://arxiv.org/pdf/2602.01797)</summary>

**Abstract:** Recent advances in large-scale language models (LLMs) have made multi-agent architectures attractive for challenging reasoning tasks. However, many existing systems rely on stochastic routing or ad-hoc heuristics, making their behavior difficult to reproduce and their decision process hard to interpret. We propose ORCH, a deterministic coordination framework for discrete-choice reasoning that orchestrates heterogeneous LLMs. ORCH follows a ``many analyses, one decision'' paradigm: multiple base models independently produce structured analyses, and a dedicated merge agent outputs the final choice. The framework uses fixed rules for task decomposition and answer aggregation, keeping the pipeline predictable, reproducible, and training-free. Determinism here refers to fixed routing and aggregation rules under a fixed evaluation protocol, rather than strict bit-level reproducibility across deployments. To exploit model complementarity, we optionally introduce an EMA-guided router that updates agent selection using historical accuracy, latency, or cost; since it relies on answer-based feedback, it is mainly intended for benchmarking, controlled evaluation, or delayed-feedback settings. Experiments on MMLU, MMLU-Pro, and GSM8K show that ORCH consistently outperforms single-model baselines and a majority-vote ensemble. On MMLU-Pro, ORCH improves accuracy by over 10 points compared to the strongest baseline, and on GSM8K it yields gains exceeding 50 points; McNemar tests confirm statistical significance. The EMA router provides an additional 0.7--2.0 point accuracy boost, and ablations show that both multi-agent collaboration and routing contribute substantially. Overall, ORCH offers a practical path toward controllable, interpretable, and deployment-ready LLM-based agent systems for discrete-choice reasoning.

**arXiv ID:** 2602.01797
</details>

<details>
<summary><strong>INDIBATOR: Diverse and Fact-Grounded Individuality for Multi-Agent Debate in Molecular Discovery</strong> - Yunhui Jang, Seonghyun Park, Jaehyung Kim, Sungsoo Ahn - [[pdf]](https://arxiv.org/pdf/2602.01815)</summary>

**Abstract:** Multi-agent systems have emerged as a powerful paradigm for automating scientific discovery. To differentiate agent behavior in the multi-agent system, current frameworks typically assign generic role-based personas such as ''reviewer'' or ''writer'' or rely on coarse grained keyword-based personas. While functional, this approach oversimplifies how human scientists operate, whose contributions are shaped by their unique research trajectories. In response, we propose INDIBATOR, a framework for molecular discovery that grounds agents in individualized scientist profiles constructed from two modalities: publication history for literature-derived knowledge and molecular history for structural priors. These agents engage in multi-turn debate through proposal, critique, and voting phases. Our evaluation demonstrates that these fine-grained individuality-grounded agents consistently outperform systems relying on coarse-grained personas, achieving competitive or state-of-the-art performance. These results validate that capturing the ``scientific DNA'' of individual agents is essential for high-quality discovery.

**arXiv ID:** 2602.01815
</details>

<details>
<summary><strong>ROMA: Recursive Open Meta-Agent Framework for Long-Horizon Multi-Agent Systems</strong> - Salaheddin Alzu'bi, Baran Nama, Arda Kaz, Anushri Eswaran, Weiyuan Chen, Sarvesh Khetan, Rishab Bala, Tu Vu, Sewoong Oh - [[pdf]](https://arxiv.org/pdf/2602.01848)</summary>

**Abstract:** Current agentic frameworks underperform on long-horizon tasks. As reasoning depth increases, sequential orchestration becomes brittle, context windows impose hard limits that degrade performance, and opaque execution traces make failures difficult to localize or debug. We introduce ROMA (Recursive Open Meta-Agents), a domain-agnostic framework that addresses these limitations through recursive task decomposition and structured aggregation. ROMA decomposes goals into dependency-aware subtask trees that can be executed in parallel, while aggregation compresses and validates intermediate results to control context growth. Our framework standardizes agent construction around four modular roles --Atomizer (which decides whether a task should be decomposed), Planner, Executor, and Aggregator -- which cleanly separate orchestration from model selection and enable transparent, hierarchical execution traces. This design supports heterogeneous multi-agent systems that mix models and tools according to cost, latency, and capability. To adapt ROMA to specific tasks without fine-tuning, we further introduce GEPA$+$, an improved Genetic-Pareto prompt proposer that searches over prompts within ROMA's component hierarchy while preserving interface contracts. We show that ROMA, combined with GEPA+, delivers leading system-level performance on reasoning and long-form generation benchmarks. On SEAL-0, which evaluates reasoning over conflicting web evidence, ROMA instantiated with GLM-4.6 improves accuracy by 9.9\% over Kimi-Researcher. On EQ-Bench, a long-form writing benchmark, ROMA enables DeepSeek-V3 to match the performance of leading closed-source models such as Claude Sonnet 4.5. Our results demonstrate that recursive, modular agent architectures can scale reasoning depth while remaining interpretable, flexible, and model-agnostic.

**arXiv ID:** 2602.01848
</details>

<details>
<summary><strong>Constrained Process Maps for Multi-Agent Generative AI Workflows</strong> - Ananya Joshi, Michael Rudow - [[pdf]](https://arxiv.org/pdf/2602.02034)</summary>

**Abstract:** Large language model (LLM)-based agents are increasingly used to perform complex, multi-step workflows in regulated settings such as compliance and due diligence. However, many agentic architectures rely primarily on prompt engineering of a single agent, making it difficult to observe or compare how models handle uncertainty and coordination across interconnected decision stages and with human oversight. We introduce a multi-agent system formalized as a finite-horizon Markov Decision Process (MDP) with a directed acyclic structure. Each agent corresponds to a specific role or decision stage (e.g., content, business, or legal review in a compliance workflow), with predefined transitions representing task escalation or completion. Epistemic uncertainty is quantified at the agent level using Monte Carlo estimation, while system-level uncertainty is captured by the MDP's termination in either an automated labeled state or a human-review state. We illustrate the approach through a case study in AI safety evaluation for self-harm detection, implemented as a multi-agent compliance system. Results demonstrate improvements over a single-agent baseline, including up to a 19\% increase in accuracy, up to an 85x reduction in required human review, and, in some configurations, reduced processing time.

**arXiv ID:** 2602.02034
</details>

<details>
<summary><strong>Context Learning for Multi-Agent Discussion</strong> - Xingyuan Hua, Sheng Yue, Xinyi Li, Yizhe Zhao, Jinrui Zhang, Ju Ren - [[pdf]](https://arxiv.org/pdf/2602.02350)</summary>

**Abstract:** Multi-Agent Discussion (MAD) has garnered increasing attention very recently, where multiple LLM instances collaboratively solve problems via structured discussion. However, we find that current MAD methods easily suffer from discussion inconsistency, LLMs fail to reach a coherent solution, due to the misalignment between their individual this http URL this paper, we introduce a multi-LLM context learning method (M2CL) that learns a context generator for each agent, capable of dynamically generating context instructions per discussion round via automatic information organization and refinement. Specifically, inspired by our theoretical insights on the context instruction, M2CL train the generators to control context coherence and output discrepancies via a carefully crafted self-adaptive this http URL enables LLMs to avoid premature convergence on majority noise and progressively reach the correct consensus. We evaluate M2CL on challenging tasks, including academic reasoning, embodied tasks, and mobile control. The results show that the performance of M2CL significantly surpasses existing methods by 20%--50%, while enjoying favorable transferability and computational efficiency.

**arXiv ID:** 2602.02350
</details>

<details>
<summary><strong>AgentRx: Diagnosing AI Agent Failures from Execution Trajectories</strong> - Shraddha Barke, Arnav Goyal, Alind Khare, Avaljot Singh, Suman Nath, Chetan Bansal - [[pdf]](https://arxiv.org/pdf/2602.02475)</summary>

**Abstract:** AI agents often fail in ways that are difficult to localize because executions are probabilistic, long-horizon, multi-agent, and mediated by noisy tool outputs. We address this gap by manually annotating failed agent runs and release a novel benchmark of 115 failed trajectories spanning structured API workflows, incident management, and open-ended web/file tasks. Each trajectory is annotated with a critical failure step and a category from a grounded-theory derived, cross-domain failure taxonomy. To mitigate the human cost of failure attribution, we present AGENTRX, an automated domain-agnostic diagnostic framework that pinpoints the critical failure step in a failed agent trajectory. It synthesizes constraints, evaluates them step-by-step, and produces an auditable validation log of constraint violations with associated evidence; an LLM-based judge uses this log to localize the critical step and category. Our framework improves step localization and failure attribution over existing baselines across three domains.

**arXiv ID:** 2602.02475
</details>

<details>
<summary><strong>SafeTalkCoach: Diversity-Driven Multi-Agent Simulation for Parent-Teen Health Conversations</strong> - Benyamin Tabarsi, Wenbo Li, Tahreem Yasir, Aryan Santhosh Kumar, Laura Widman, Dongkuan Xu, Tiffany Barnes - [[pdf]](https://arxiv.org/pdf/2602.00017)</summary>

**Abstract:** The importance of effective parent-child communication about sexual health is widely acknowledged, but real-world data on these conversations is scarce and challenging to collect, due to their private and sensitive nature. Although LLMs have been widely adopted in dialogue generation, they may deviate from best practices and frequently lack realism and diversity. We introduce SafeTalkCoach, a diversity-driven multi-agent dialogue generation framework that simulates parent-child conversations about sexual health, and present an accompanying dataset. SafeTalkCoach integrates crowd-sourced and synthesized scenarios, established sexual health guidelines, evidence-based personas, adaptive control modules, and hierarchical diversification. Through evaluations, we demonstrate that SafeTalkCoach generates diverse conversations while maintaining realism, communication quality, and controllability in practice. Our goal is that the SafeTalkCoach framework and the dataset support both AI research and health communications practices.

**arXiv ID:** 2602.00017
</details>

<details>
<summary><strong>Design and Empirical Study of a Large Language Model-Based Multi-Agent Investment System for Chinese Public REITs</strong> - Zheng Li - [[pdf]](https://arxiv.org/pdf/2602.00082)</summary>

**Abstract:** This study addresses the low-volatility Chinese Public Real Estate Investment Trusts (REITs) market, proposing a large language model (LLM)-driven trading framework based on multi-agent collaboration. The system constructs four types of analytical agents-announcement, event, price momentum, and market-each conducting analysis from different dimensions; then the prediction agent integrates these multi-source signals to output directional probability distributions across multiple time horizons, then the decision agent generates discrete position adjustment signals based on the prediction results and risk control constraints, thereby forming a closed loop of analysis-prediction-decision-execution. This study further compares two prediction model pathways: for the prediction agent, directly calling the general-purpose large model DeepSeek-R1 versus using a specialized small model Qwen3-8B fine-tuned via supervised fine-tuning and reinforcement learning alignment. In the backtest from October 2024 to October 2025, both agent-based strategies significantly outperformed the buy-and-hold benchmark in terms of cumulative return, Sharpe ratio, and maximum drawdown. The results indicate that the multi-agent framework can effectively enhance the risk-adjusted return of REITs trading, and the fine-tuned small model performs close to or even better than the general-purpose large model in some scenarios.

**arXiv ID:** 2602.00082
</details>

<details>
<summary><strong>Autonomous Multi-Agent AI for High-Throughput Polymer Informatics: From Property Prediction to Generative Design Across Synthetic and Bio-Polymers</strong> - Mahule Roy, Adib Bazgir, Arthur da Silva Sousa Santos, Yuwen Zhang - [[pdf]](https://arxiv.org/pdf/2602.00103)</summary>

**Abstract:** We present an integrated multiagent AI ecosystem for polymer discovery that unifies high-throughput materials workflows, artificial intelligence, and computational modeling within a single Polymer Research Lifecycle (PRL) pipeline. The system orchestrates specialized agents powered by state-of-the-art large language models (DeepSeek-V2 and DeepSeek-Coder) to retrieve and reason over scientific resources, invoke external tools, execute domain-specific code, and perform metacognitive self-assessment for robust end-to-end task execution. We demonstrate three practical capabilities: a high-fidelity polymer property prediction and generative design pipeline, a fully automated multimodal workflow for biopolymer structure characterization, and a metacognitive agent framework that can monitor performance and improve execution strategies over time. On a held-out test set of 1,251 polymers, our PolyGNN agent achieves strong predictive accuracy, reaching R2 = 0.89 for glass-transition temperature (Tg ), R2 = 0.82 for tensile strength, R2 = 0.75 for elongation, and R2 = 0.91 for density. The framework also provides uncertainty estimates via multiagent consensus and scales with linear complexity to at least 10,000 polymers, enabling high-throughput screening at low computational cost. For a representative workload, the system completes inference in 16.3 s using about 2 GB of memory and 0.1 GPU hours, at an estimated cost of about $0.08. On a dedicated Tg benchmark, our approach attains R2 = 0.78, outperforming strong baselines including single-LLM prediction (R2 = 0.67), group-contribution methods (R2 = 0.71), and ChemCrow (R2 = 0.66). We further demonstrate metacognitive control in a polystyrene case study, where the system not only produces domain-level scientific outputs but continually monitors and optimizes its own behavior through tactical, strategic, and meta-strategic self-assessment.

**arXiv ID:** 2602.00103
</details>

<details>
<summary><strong>Rank-and-Reason: Multi-Agent Collaboration Accelerates Zero-Shot Protein Mutation Prediction</strong> - Yang Tan, Yuanxi Yu, Can Wu, Bozitao Zhong, Mingchen Li, Guisheng Fan, Jiankang Zhu, Yafeng Liang, Nanqing Dong, Liang Hong - [[pdf]](https://arxiv.org/pdf/2602.00197)</summary>

**Abstract:** Zero-shot mutation prediction is vital for low-resource protein engineering, yet existing protein language models (PLMs) often yield statistically confident results that ignore fundamental biophysical constraints. Currently, selecting candidates for wet-lab validation relies on manual expert auditing of PLM outputs, a process that is inefficient, subjective, and highly dependent on domain expertise. To address this, we propose Rank-and-Reason (VenusRAR), a two-stage agentic framework to automate this workflow and maximize expected wet-lab fitness. In the Rank-Stage, a Computational Expert and Virtual Biologist aggregate a context-aware multi-modal ensemble, establishing a new Spearman correlation record of 0.551 (vs. 0.518) on ProteinGym. In the Reason-Stage, an agentic Expert Panel employs chain-of-thought reasoning to audit candidates against geometric and structural constraints, improving the Top-5 Hit Rate by up to 367% on ProteinGym-DMS99. The wet-lab validation on Cas12i3 nuclease further confirms the framework's efficacy, achieving a 46.7% positive rate and identifying two novel mutants with 4.23-fold and 5.05-fold activity improvements. Code and datasets are released on GitHub (this https URL).

**arXiv ID:** 2602.00197
</details>

<details>
<summary><strong>Evolving Interpretable Constitutions for Multi-Agent Simulation</strong> - Ujwal Kumar, Alice Saito, Hershraj Niranjani, Rayan Yessou, Phan Xuan Tan - [[pdf]](https://arxiv.org/pdf/2602.00755)</summary>

**Abstract:** Constitutional AI has focused on single-model alignment using fixed principles. However, multi-agent systems create novel alignment challenges through emergent social dynamics. We present Constitutional Evolution, a framework for automatically discovering behavioral norms in multi-agent LLM systems. Using a grid-world simulation with survival pressure, we study the tension between individual and collective welfare, quantified via a Societal Stability Score S in [0,1] that combines productivity, survival, and conflict metrics. Adversarial constitutions lead to societal collapse (S= 0), while vague prosocial principles ("be helpful, harmless, honest") produce inconsistent coordination (S = 0.249). Even constitutions designed by Claude 4.5 Opus with explicit knowledge of the objective achieve only moderate performance (S= 0.332). Using LLM-driven genetic programming with multi-island evolution, we evolve constitutions maximizing social welfare without explicit guidance toward cooperation. The evolved constitution C* achieves S = 0.556 +/- 0.008 (123% higher than human-designed baselines, N = 10), eliminates conflict, and discovers that minimizing communication (0.9% vs 62.2% social actions) outperforms verbose coordination. Our interpretable rules demonstrate that cooperative norms can be discovered rather than prescribed.

**arXiv ID:** 2602.00755
</details>

<details>
<summary><strong>Communications-Incentivized Collaborative Reasoning in NetGPT through Agentic Reinforcement Learning</strong> - Xiaoxue Yu, Rongpeng Li, Zhifeng Zhao, Honggang Zhang - [[pdf]](https://arxiv.org/pdf/2602.00766)</summary>

**Abstract:** The evolution of next-Generation (xG) wireless networks marks a paradigm shift from connectivity-centric architectures to Artificial Intelligence (AI)-native designs that tightly integrate data, computing, and communication. Yet existing AI deployments in communication systems remain largely siloed, offering isolated optimizations without intrinsic adaptability, dynamic task delegation, or multi-agent collaboration. In this work, we propose a unified agentic NetGPT framework for AI-native xG networks, wherein a NetGPT core can either perform autonomous reasoning or delegate sub-tasks to domain-specialized agents via agentic communication. The framework establishes clear modular responsibilities and interoperable workflows, enabling scalable, distributed intelligence across the network. To support continual refinement of collaborative reasoning strategies, the framework is further enhanced through Agentic reinforcement learning under partially observable conditions and stochastic external states. The training pipeline incorporates masked loss against external agent uncertainty, entropy-guided exploration, and multi-objective rewards that jointly capture task quality, coordination efficiency, and resource constraints. Through this process, NetGPT learns when and how to collaborate, effectively balancing internal reasoning with agent invocation. Overall, this work provides a foundational architecture and training methodology for self-evolving, AI-native xG networks capable of autonomous sensing, reasoning, and action in complex communication environments.

**arXiv ID:** 2602.00766
</details>

<details>
<summary><strong>Symphony-Coord: Emergent Coordination in Decentralized Agent Systems</strong> - Zhaoyang Guan, Huixi Cao, Ming Zhong, Eric Yang, Lynn Ai, Yongxin Ni, Bill Shi - [[pdf]](https://arxiv.org/pdf/2602.00966)</summary>

**Abstract:** Multi-agent large language model systems can tackle complex multi-step tasks by decomposing work and coordinating specialized behaviors. However, current coordination mechanisms typically rely on statically assigned roles and centralized controllers. As agent pools and task distributions evolve, these design choices lead to inefficient routing, poor adaptability, and fragile fault recovery capabilities. We introduce Symphony-Coord, a decentralized multi-agent framework that transforms agent selection into an online multi-armed bandit problem, enabling roles to emerge organically through interaction. The framework employs a two-stage dynamic beacon protocol: (i) a lightweight candidate screening mechanism to limit communication and computational overhead; (ii) an adaptive LinUCB selector that routes subtasks based on context features derived from task requirements and agent states, continuously optimized through delayed end-to-end feedback. Under standard linear realizability assumptions, we provide sublinear regret bounds, indicating the system converges toward near-optimal allocation schemes. Validation through simulation experiments and real-world large language model benchmarks demonstrates that Symphony-Coord not only enhances task routing efficiency but also exhibits robust self-healing capabilities in scenarios involving distribution shifts and agent failures, achieving a scalable coordination mechanism without predefined roles.

**arXiv ID:** 2602.00966
</details>

<details>
<summary><strong>Multi-Agent Teams Hold Experts Back</strong> - Aneesh Pappu, Batu El, Hancheng Cao, Carmelo di Nolfo, Yanchao Sun, Meng Cao, James Zou - [[pdf]](https://arxiv.org/pdf/2602.01011)</summary>

**Abstract:** Multi-agent LLM systems are increasingly deployed as autonomous collaborators, where agents interact freely rather than execute fixed, pre-specified workflows. In such settings, effective coordination cannot be fully designed in advance and must instead emerge through interaction. However, most prior work enforces coordination through fixed roles, workflows, or aggregation rules, leaving open the question of how well self-organizing teams perform when coordination is unconstrained. Drawing on organizational psychology, we study whether self-organizing LLM teams achieve strong synergy, where team performance matches or exceeds the best individual member. Across human-inspired and frontier ML benchmarks, we find that -- unlike human teams -- LLM teams consistently fail to match their expert agent's performance, even when explicitly told who the expert is, incurring performance losses of up to 37.6%. Decomposing this failure, we show that expert leveraging, rather than identification, is the primary bottleneck. Conversational analysis reveals a tendency toward integrative compromise -- averaging expert and non-expert views rather than appropriately weighting expertise -- which increases with team size and correlates negatively with performance. Interestingly, this consensus-seeking behavior improves robustness to adversarial agents, suggesting a trade-off between alignment and effective expertise utilization. Our findings reveal a significant gap in the ability of self-organizing multi-agent teams to harness the collective expertise of their members.

**arXiv ID:** 2602.01011
</details>

<details>
<summary><strong>A-MapReduce: Executing Wide Search via Agentic MapReduce</strong> - Mingju Chen, Guibin Zhang, Heng Chang, Yuchen Guo, Shiji Zhou - [[pdf]](https://arxiv.org/pdf/2602.01331)</summary>

**Abstract:** Contemporary large language model (LLM)-based multi-agent systems exhibit systematic advantages in deep research tasks, which emphasize iterative, vertically structured information seeking. However, when confronted with wide search tasks characterized by large-scale, breadth-oriented retrieval, existing agentic frameworks, primarily designed around sequential, vertically structured reasoning, remain stuck in expansive search objectives and inefficient long-horizon execution. To bridge this gap, we propose A-MapReduce, a MapReduce paradigm-inspired multi-agent execution framework that recasts wide search as a horizontally structured retrieval problem. Concretely, A-MapReduce implements parallel processing of massive retrieval targets through task-adaptive decomposition and structured result aggregation. Meanwhile, it leverages experiential memory to drive the continual evolution of query-conditioned task allocation and recomposition, enabling progressive improvement in large-scale wide-search regimes. Extensive experiments on five agentic benchmarks demonstrate that A-MapReduce is (i) high-performing, achieving state-of-the-art performance on WideSearch and DeepWideSearch, and delivering 5.11% - 17.50% average Item F1 improvements compared with strong baselines with OpenAI o3 or Gemini 2.5 Pro backbones; (ii) cost-effective and efficient, delivering superior cost-performance trade-offs and reducing running time by 45.8\% compared to representative multi-agent baselines. The code is available at this https URL.

**arXiv ID:** 2602.01331
</details>

<details>
<summary><strong>Evidence-Decision-Feedback: Theory-Driven Adaptive Scaffolding for LLM Agents</strong> - Clayton Cohn, Siyuan Guo, Surya Rayala, Hanchen David Wang, Naveeduddin Mohammed, Umesh Timalsina, Shruti Jain, Angela Eeds, Menton Deweese, Pamela J. Osborn Popp, Rebekah Stanton, Shakeera Walker, Meiyi Ma, Gautam Biswas - [[pdf]](https://arxiv.org/pdf/2602.01415)</summary>

**Abstract:** Multi-agent LLM architectures offer opportunities for pedagogical agents to help students construct domain knowledge and develop critical-thinking skills, yet many operate on a "one-size-fits-all" basis, limiting their ability to provide personalized support. To address this, we introduce Evidence-Decision-Feedback (EDF), a theoretical framework for adaptive scaffolding using LLMs. EDF integrates elements of intelligent tutoring systems and agentic behavior by organizing interactions around evidentiary inference, pedagogical decision-making, and adaptive feedback. We instantiate EDF through Copa, an agentic collaborative peer agent for STEM+C problem-solving. In an authentic high school classroom study, we show that EDF-aligned interactions align feedback with students' demonstrated understanding and task mastery; promote gradual scaffold fading; and support interpretable, evidence-grounded explanations without fostering overreliance.

**arXiv ID:** 2602.01415
</details>

<details>
<summary><strong>TABX: A High-Throughput Sandbox Battle Simulator for Multi-Agent Reinforcement Learning</strong> - Hayeong Lee, JunHyeok Oh, Byung-Jun Lee - [[pdf]](https://arxiv.org/pdf/2602.01665)</summary>

**Abstract:** The design of environments plays a critical role in shaping the development and evaluation of cooperative multi-agent reinforcement learning (MARL) algorithms. While existing benchmarks highlight critical challenges, they often lack the modularity required to design custom evaluation scenarios. We introduce the Totally Accelerated Battle Simulator in JAX (TABX), a high-throughput sandbox designed for reconfigurable multi-agent tasks. TABX provides granular control over environmental parameters, permitting a systematic investigation into emergent agent behaviors and algorithmic trade-offs across a diverse spectrum of task complexities. Leveraging JAX for hardware-accelerated execution on GPUs, TABX enables massive parallelization and significantly reduces computational overhead. By providing a fast, extensible, and easily customized framework, TABX facilitates the study of MARL agents in complex structured domains and serves as a scalable foundation for future research. Our code is available at: this https URL.

**arXiv ID:** 2602.01665
</details>

<details>
<summary><strong>Self-Evolving Coordination Protocol in Multi-Agent AI Systems: An Exploratory Systems Feasibility Study</strong> - Jose Manuel de la Chica Rodriguez, Juan Manuel Vera Díaz - [[pdf]](https://arxiv.org/pdf/2602.02170)</summary>

**Abstract:** Contemporary multi-agent systems increasingly rely on internal coordination mechanisms to combine, arbitrate, or constrain the outputs of heterogeneous components. In safety-critical and regulated domains such as finance, these mechanisms must satisfy strict formal requirements, remain auditable, and operate within explicitly bounded limits. Coordination logic therefore functions as a governance layer rather than an optimization heuristic.
This paper presents an exploratory systems feasibility study of Self-Evolving Coordination Protocols (SECP): coordination protocols that permit limited, externally validated self-modification while preserving fixed formal invariants. We study a controlled proof-of-concept setting in which six fixed Byzantine consensus protocol proposals are evaluated by six specialized decision modules. All coordination regimes operate under identical hard constraints, including Byzantine fault tolerance (f < n/3), O(n2) message complexity, complete non-statistical safety and liveness arguments, and bounded explainability.
Four coordination regimes are compared in a single-shot design: unanimous hard veto, weighted scalar aggregation, SECP v1.0 (an agent-designed non-scalar protocol), and SECP v2.0 (the result of one governed modification). Outcomes are evaluated using a single metric, proposal coverage, defined as the number of proposals accepted. A single recursive modification increased coverage from two to three accepted proposals while preserving all declared invariants.
The study makes no claims regarding statistical significance, optimality, convergence, or learning. Its contribution is architectural: it demonstrates that bounded self-modification of coordination protocols is technically implementable, auditable, and analyzable under explicit formal constraints, establishing a foundation for governed multi-agent systems.

**arXiv ID:** 2602.02170
</details>

<details>
<summary><strong>Asynchronous MultiAgent Reinforcement Learning for 5G Routing under Side Constraints</strong> - Sebastian Racedo, Brigitte Jaumard, Oscar Delgado, Meysam Masoudi - [[pdf]](https://arxiv.org/pdf/2602.00035)</summary>

**Abstract:** Networks in the current 5G and beyond systems increasingly carry heterogeneous traffic with diverse quality-of-service constraints, making real-time routing decisions both complex and time-critical. A common approach, such as a heuristic with human intervention or training a single centralized RL policy or synchronizing updates across multiple learners, struggles with scalability and straggler effects. We address this by proposing an asynchronous multi-agent reinforcement learning (AMARL) framework in which independent PPO agents, one per service, plan routes in parallel and commit resource deltas to a shared global resource environment. This coordination by state preserves feasibility across services and enables specialization for service-specific objectives. We evaluate the method on an O-RAN like network simulation using nearly real-time traffic data from the city of Montreal. We compared against a single-agent PPO baseline. AMARL achieves a similar Grade of Service (acceptance rate) (GoS) and end-to-end latency, with reduced training wall-clock time and improved robustness to demand shifts. These results suggest that asynchronous, service-specialized agents provide a scalable and practical approach to distributed routing, with applicability extending beyond the O-RAN domain.

**arXiv ID:** 2602.00035
</details>

<details>
<summary><strong>FinEvo: From Isolated Backtests to Ecological Market Games for Multi-Agent Financial Strategy Evolution</strong> - Mingxi Zou, Jiaxiang Chen, Aotian Luo, Jingyi Dai, Chi Zhang, Dongning Sun, Zenglin Xu - [[pdf]](https://arxiv.org/pdf/2602.00948)</summary>

**Abstract:** Conventional financial strategy evaluation relies on isolated backtests in static environments. Such evaluations assess each policy independently, overlook correlations and interactions, and fail to explain why strategies ultimately persist or vanish in evolving markets. We shift to an ecological perspective, where trading strategies are modeled as adaptive agents that interact and learn within a shared market. Instead of proposing a new strategy, we present FinEvo, an ecological game formalism for studying the evolutionary dynamics of multi-agent financial strategies. At the individual level, heterogeneous ML-based traders-rule-based, deep learning, reinforcement learning, and large language model (LLM) agents-adapt using signals such as historical prices and external news. At the population level, strategy distributions evolve through three designed mechanisms-selection, innovation, and environmental perturbation-capturing the dynamic forces of real markets. Together, these two layers of adaptation link evolutionary game theory with modern learning dynamics, providing a principled environment for studying strategic behavior. Experiments with external shocks and real-world news streams show that FinEvo is both stable for reproducibility and expressive in revealing context-dependent outcomes. Strategies may dominate, collapse, or form coalitions depending on their competitors-patterns invisible to static backtests. By reframing strategy evaluation as an ecological game formalism, FinEvo provides a unified, mechanism-level protocol for analyzing robustness, adaptation, and emergent dynamics in multi-agent financial markets, and may offer a means to explore the potential impact of macroeconomic policies and financial regulations on price evolution and equilibrium.

**arXiv ID:** 2602.00948
</details>

<details>
<summary><strong>Bandwidth-Efficient Multi-Agent Communication through Information Bottleneck and Vector Quantization</strong> - Ahmad Farooq, Kamran Iqbal - [[pdf]](https://arxiv.org/pdf/2602.02035)</summary>

**Abstract:** Multi-agent reinforcement learning systems deployed in real-world robotics applications face severe communication constraints that significantly impact coordination effectiveness. We present a framework that combines information bottleneck theory with vector quantization to enable selective, bandwidth-efficient communication in multi-agent environments. Our approach learns to compress and discretize communication messages while preserving task-critical information through principled information-theoretic optimization. We introduce a gated communication mechanism that dynamically determines when communication is necessary based on environmental context and agent states. Experimental evaluation on challenging coordination tasks demonstrates that our method achieves 181.8% performance improvement over no-communication baselines while reducing bandwidth usage by 41.4%. Comprehensive Pareto frontier analysis shows dominance across the entire success-bandwidth spectrum with area-under-curve of 0.198 vs 0.142 for next-best methods. Our approach significantly outperforms existing communication strategies and establishes a theoretically grounded framework for deploying multi-agent systems in bandwidth-constrained environments such as robotic swarms, autonomous vehicle fleets, and distributed sensor networks.

**arXiv ID:** 2602.02035
</details>

<details>
<summary><strong>LLM-Based Multi-Agent Blackboard System for Information Discovery in Data Science</strong> - Alireza Salemi, Mihir Parmar, Palash Goyal, Yiwen Song, Jinsung Yoon, Hamed Zamani, Tomas Pfister, Hamid Palangi - [[pdf]](https://arxiv.org/pdf/2510.01285)</summary>

**Abstract:** Advances in large language models (LLMs) have created new opportunities in data science, but their deployment is often limited by the challenge of finding relevant data in large data lakes. Existing methods struggle with this: both single- and multi-agent systems are quickly overwhelmed by large, heterogeneous files, and master-slave multi-agent systems rely on a rigid central controller that requires precise knowledge of each sub-agent's capabilities, which is not possible in large-scale settings where the main agent lacks full observability over sub-agents' knowledge and competencies. We propose a novel multi-agent paradigm inspired by the blackboard architecture for traditional AI models. In our framework, a central agent posts requests to a shared blackboard, and autonomous subordinate agents - either responsible for a partition of the data lake or retrieval from the web - volunteer to respond based on their capabilities. This design improves scalability and flexibility by removing the need for a central coordinator to know each agent's expertise or internal knowledge. We evaluate the approach on three benchmarks that require data discovery: KramaBench and modified versions of DSBench and DA-Code. Results show that the blackboard architecture substantially outperforms strong baselines, achieving 13%-57% relative improvements in end-to-end success and up to a 9% relative gain in data discovery F1 over the best baseline.

**arXiv ID:** 2510.01285
</details>

<details>
<summary><strong>When Personas Override Payoffs: Role Identity Bias in Multi-Agent LLM Decision-Making</strong> - Viswonathan Manoranjan, Snehalkumar `Neil' S. Gaikwad - [[pdf]](https://arxiv.org/pdf/2601.10102)</summary>

**Abstract:** Large language models are increasingly deployed in multi-agent systems for strategic tasks, yet how design choices such as role-based personas and payoff visibility affect reasoning remains poorly understood. We investigate whether multi-agent systems function as strategic reasoners capable of payoff optimization or as identity-driven actors that prioritize role alignment over explicit incentives. Using Nash equilibrium achievement as a diagnostic for strategic reasoning, we conduct systematic experiments across four LLM architectures (Qwen-7B, Qwen-32B, Llama-8B, Mistral-7B) in complex environmental decision-making games involving four agents. We show that role identity bias fundamentally alters strategic reasoning even when payoff-optimal equilibria exist and complete payoff information is available. Removing personas and providing explicit payoffs enables Qwen models to achieve high Nash equilibrium rates, indicating that both conditions are necessary for strategic reasoning. In contrast, personas systematically bias equilibrium selection toward socially preferred outcomes: with personas present, all of the achieved equilibria correspond to Green Transition, while models entirely fail to reach equilibrium when Tragedy of the Commons is payoff-optimal. The effect of explicit payoffs depends entirely on persona presence, revealing strong interactions between representational design choices. We also observe clear model-dependent patterns. Qwen architectures are highly sensitive to both personas and payoff visibility, whereas Llama and Mistral exhibit rigid reasoning behavior across conditions. These findings demonstrate that representational choices are substantive governance decisions that determine whether multi-agent systems act as strategic reasoners or identity-driven actors, with important implications for real-world deployment.

**arXiv ID:** 2601.10102
</details>

<details>
<summary><strong>Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies</strong> - Han Zhou, Xingchen Wan, Ruoxi Sun, Hamid Palangi, Shariq Iqbal, Ivan Vulić, Anna Korhonen, Sercan Ö. Arık - [[pdf]](https://arxiv.org/pdf/2502.02533)</summary>

**Abstract:** Large language models, employed as multiple agents that interact and collaborate with each other, have excelled at solving complex tasks. The agents are programmed with prompts that declare their functionality, along with the topologies that orchestrate interactions across agents. Designing prompts and topologies for multi-agent systems (MAS) is inherently complex. To automate the entire design process, we first conduct an in-depth analysis of the design space aiming to understand the factors behind building effective MAS. We reveal that prompts together with topologies play critical roles in enabling more effective MAS design. Based on the insights, we propose Multi-Agent System Search (MASS), a MAS optimization framework that efficiently exploits the complex MAS design space by interleaving its optimization stages, from local to global, from prompts to topologies, over three stages: 1) block-level (local) prompt optimization; 2) workflow topology optimization; 3) workflow-level (global) prompt optimization, where each stage is conditioned on the iteratively optimized prompts/topologies from former stages. We show that MASS-optimized multi-agent systems outperform a spectrum of existing alternatives by a substantial margin. Based on the MASS-found systems, we finally propose design principles behind building effective multi-agent systems.

**arXiv ID:** 2502.02533
</details>

<details>
<summary><strong>MAC: Masked Agent Collaboration Boosts Large Language Model Medical Decision-Making</strong> - Zhihao Peng, Liuxin Bao, Yixuan Yuan - [[pdf]](https://arxiv.org/pdf/2507.21159)</summary>

**Abstract:** Large language models (LLMs) have proven effective in artificial intelligence, where the multi-agent system (MAS) holds considerable promise for healthcare development by achieving the collaboration of LLMs. However, the absence of a systematic pipeline for agent construction and the rigidity of static collaboration patterns render current MAS-based models vulnerable to collaboration failures, resulting in substantial performance degradation in medical decision-making scenarios. To this end, we propose a novel Masked Agent Collaboration (MAC) framework that harnesses Pareto-optimal agent construction and cross-consistency maximization mechanisms to achieve adaptive progressive propagation of collaborative information, boosting the medical decision-making capacity. Specifically, we first conduct a Pareto-frontier factors analysis towards the LLMs pool to consider their key factors, including the model size, inference time, diversity score, and throughput ratio, where we calculate the similarity between pairwise outputs within an LLM to derive its diversity score. Beyond this analysis, we enable the identification of Pareto-optimal models that balance efficiency and capability, which are subsequently selected as collaborative agents to consider the fundamental trade-offs inherent in practical LLM deployment. Afterward, we measure the pairwise similarity between the outputs from collaborative agents to determine their cross-consistency values, subsequently masking out the agent with the lowest cross-consistency value to eliminate the output that is likely semantically inconsistent. Finally, we conduct collaboration of agents by achieving adaptive progressive propagation, where each agent aggregates the outputs of unmasked agents from the previous layer as its input to generate the corresponding output via prompt engineering.

**arXiv ID:** 2507.21159
</details>

<details>
<summary><strong>When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems</strong> - Naen Xu, Hengyu An, Shuo Shi, Jinghuai Zhang, Chunyi Zhou, Changjiang Li, Tianyu Du, Zhihui Fu, Jun Wang, Shouling Ji - [[pdf]](https://arxiv.org/pdf/2602.00428)</summary>

**Abstract:** Recent advancements in large language models (LLMs) have significantly enhanced the capabilities of collaborative multi-agent systems, enabling them to address complex challenges. However, within these multi-agent systems, the susceptibility of agents to collective cognitive biases remains an underexplored issue. A compelling example is the Mandela effect, a phenomenon where groups collectively misremember past events as a result of false details reinforced through social influence and internalized misinformation. This vulnerability limits our understanding of memory bias in multi-agent systems and raises ethical concerns about the potential spread of misinformation. In this paper, we conduct a comprehensive study on the Mandela effect in LLM-based multi-agent systems, focusing on its existence, causing factors, and mitigation strategies. We propose MANBENCH, a novel benchmark designed to evaluate agent behaviors across four common task types that are susceptible to the Mandela effect, using five interaction protocols that vary in agent roles and memory timescales. We evaluate agents powered by several LLMs on MANBENCH to quantify the Mandela effect and analyze how different factors affect it. Moreover, we propose strategies to mitigate this effect, including prompt-level defenses (e.g., cognitive anchoring and source scrutiny) and model-level alignment-based defense, achieving an average 74.40% reduction in the Mandela effect compared to the baseline. Our findings provide valuable insights for developing more resilient and ethically aligned collaborative multi-agent systems.

**arXiv ID:** 2602.00428
</details>

<details>
<summary><strong>DeALOG: Decentralized Multi-Agents Log-Mediated Reasoning Framework</strong> - Abhijit Chakraborty, Ashish Raj Shekhar, Shiven Agarwal, Vivek Gupta - [[pdf]](https://arxiv.org/pdf/2602.00996)</summary>

**Abstract:** Complex question answering across text, tables and images requires integrating diverse information sources. A framework supporting specialized processing with coordination and interpretability is needed. We introduce DeALOG, a decentralized multi-agent framework for multimodal question answering. It uses specialized agents: Table, Context, Visual, Summarizing and Verification, that communicate through a shared natural-language log as persistent memory. This log-based approach enables collaborative error detection and verification without central control, improving robustness. Evaluations on FinQA, TAT-QA, CRT-QA, WikiTableQuestions, FeTaQA, and MultiModalQA show competitive performance. Analysis confirms the importance of the shared log, agent specialization, and verification for accuracy. DeALOG, provides a scalable approach through modular components using natural-language communication.

**arXiv ID:** 2602.00996
</details>

<details>
<summary><strong>A2Eval: Agentic and Automated Evaluation for Embodied Brain</strong> - Shuai Zhang, Jiayu Hu, Zijie Chen, Zeyuan Ding, Yi Zhang, Yingji Zhang, Ziyi Zhou, Junwei Liao, Shengjie Zhou, Yong Dai, Zhenzhong Lan, Xiaozhu Ju - [[pdf]](https://arxiv.org/pdf/2602.01640)</summary>

**Abstract:** Current embodied VLM evaluation relies on static, expert-defined, manually annotated benchmarks that exhibit severe redundancy and coverage imbalance. This labor intensive paradigm drains computational and annotation resources, inflates costs, and distorts model rankings, ultimately stifling iterative development. To address this, we propose Agentic Automatic Evaluation (A2Eval), the first agentic framework that automates benchmark curation and evaluation through two collaborative agents. The Data Agent autonomously induces capability dimensions and assembles a balanced, compact evaluation suite, while the Eval Agent synthesizes and validates executable evaluation pipelines, enabling fully autonomous, high-fidelity assessment. Evaluated across 10 benchmarks and 13 models, A2Eval compresses evaluation suites by 85%, reduces overall computational costs by 77%, and delivers a 4.6x speedup while preserving evaluation quality. Crucially, A2Eval corrects systematic ranking biases, improves human alignment to Spearman's rho=0.85, and maintains high ranking fidelity (Kendall's tau=0.81), establishing a new standard for high-fidelity, low-cost embodied assessment. Our code and data will be public soon.

**arXiv ID:** 2602.01640
</details>

<details>
<summary><strong>ALIGN: Aligned Delegation with Performance Guarantees for Multi-Agent LLM Reasoning</strong> - Tong Zhu, Baiting Chen, Jin Zhou, Hua Zhou, Sriram Sankararaman, Xiaowu Dai - [[pdf]](https://arxiv.org/pdf/2602.00127)</summary>

**Abstract:** LLMs often underperform on complex reasoning tasks when relying on a single generation-and-selection pipeline. Inference-time ensemble methods can improve performance by sampling diverse reasoning paths or aggregating multiple candidate answers, but they typically treat candidates independently and provide no formal guarantees that ensembling improves reasoning quality. We propose a novel method, Aligned Delegation for Multi-Agent LLM Reasoning (ALIGN), which formulates LLM reasoning as an aligned delegation game. In ALIGN, a principal delegates a task to multiple agents that generate candidate solutions under designed incentives, and then selects among their outputs to produce a final answer. This formulation induces structured interaction among agents while preserving alignment between agent and principal objectives. We establish theoretical guarantees showing that, under a fair comparison with equal access to candidate solutions, ALIGN provably improves expected performance over single-agent generation. Our analysis accommodates correlated candidate answers and relaxes independence assumptions that are commonly used in prior work. Empirical results across a broad range of LLM reasoning benchmarks consistently demonstrate that ALIGN outperforms strong single-agent and ensemble baselines.

**arXiv ID:** 2602.00127
</details>

<details>
<summary><strong>Provable Cooperative Multi-Agent Exploration for Reward-Free MDPs</strong> - Idan Barnea, Orin Levy, Yishay Mansour - [[pdf]](https://arxiv.org/pdf/2602.01453)</summary>

**Abstract:** We study cooperative multi-agent reinforcement learning in the setting of reward-free exploration, where multiple agents jointly explore an unknown MDP in order to learn its dynamics (without observing rewards). We focus on a tabular finite-horizon MDP and adopt a phased learning framework. In each learning phase, multiple agents independently interact with the environment. More specifically, in each learning phase, each agent is assigned a policy, executes it, and observes the resulting trajectory. Our primary goal is to characterize the tradeoff between the number of learning phases and the number of agents, especially when the number of learning phases is small.
Our results identify a sharp transition governed by the horizon $H$. When the number of learning phases equals $H$, we present a computationally efficient algorithm that uses only $\tilde{O}(S^6 H^6 A / \epsilon^2)$ agents to obtain an $\epsilon$ approximation of the dynamics (i.e., yields an $\epsilon$-optimal policy for any reward function). We complement our algorithm with a lower bound showing that any algorithm restricted to $\rho < H$ phases requires at least $A^{H/\rho}$ agents to achieve constant accuracy. Thus, we show that it is essential to have an order of $H$ learning phases if we limit the number of agents to be polynomial.

**arXiv ID:** 2602.01453
</details>

<details>
<summary><strong>Multi-Agent Monte Carlo Tree Search for Makespan-Efficient Object Rearrangement in Cluttered Spaces</strong> - Hanwen Ren, Junyong Kim, Aathman Tharmasanthiran, Ahmed H. Qureshi - [[pdf]](https://arxiv.org/pdf/2602.02411)</summary>

**Abstract:** Object rearrangement planning in complex, cluttered environments is a common challenge in warehouses, households, and rescue sites. Prior studies largely address monotone instances, whereas real-world tasks are often non-monotone-objects block one another and must be temporarily relocated to intermediate positions before reaching their final goals. In such settings, effective multi-agent collaboration can substantially reduce the time required to complete tasks. This paper introduces Centralized, Asynchronous, Multi-agent Monte Carlo Tree Search (CAM-MCTS), a novel framework for general-purpose makespan-efficient object rearrangement planning in challenging environments. CAM-MCTS combines centralized task assignment-where agents remain aware of each other's intended actions to facilitate globally optimized planning-with an asynchronous task execution strategy that enables agents to take on new tasks at appropriate time steps, rather than waiting for others, guided by a one-step look-ahead cost estimate. This design minimizes idle time, prevents unnecessary synchronization delays, and enhances overall system efficiency. We evaluate CAM-MCTS across a diverse set of monotone and non-monotone tasks in cluttered environments, demonstrating consistent reductions in makespan compared to strong baselines. Finally, we validate our approach on a real-world multi-agent system under different configurations, further confirming its effectiveness and robustness.

**arXiv ID:** 2602.02411
</details>

<details>
<summary><strong>AdaptNC: Adaptive Nonconformity Scores for Uncertainty-Aware Autonomous Systems in Dynamic Environments</strong> - Renukanandan Tumu, Aditya Singh, Rahul Mangharam - [[pdf]](https://arxiv.org/pdf/2602.01629)</summary>

**Abstract:** Rigorous uncertainty quantification is essential for the safe deployment of autonomous systems in unconstrained environments. Conformal Prediction (CP) provides a distribution-free framework for this task, yet its standard formulations rely on exchangeability assumptions that are violated by the distribution shifts inherent in real-world robotics. Existing online CP methods maintain target coverage by adaptively scaling the conformal threshold, but typically employ a static nonconformity score function. We show that this fixed geometry leads to highly conservative, volume-inefficient prediction regions when environments undergo structural shifts. To address this, we propose \textbf{AdaptNC}, a framework for the joint online adaptation of both the nonconformity score parameters and the conformal threshold. AdaptNC leverages an adaptive reweighting scheme to optimize score functions, and introduces a replay buffer mechanism to mitigate the coverage instability that occurs during score transitions. We evaluate AdaptNC on diverse robotic benchmarks involving multi-agent policy changes, environmental changes and sensor degradation. Our results demonstrate that AdaptNC significantly reduces prediction region volume compared to state-of-the-art threshold-only baselines while maintaining target coverage levels.

**arXiv ID:** 2602.01629
</details>

<details>
<summary><strong>Versatile Behavior Diffusion for Generalized Traffic Agent Simulation</strong> - Zhiyu Huang, Zixu Zhang, Ameya Vaidya, Yuxiao Chen, Chen Lv, Jaime Fernández Fisac - [[pdf]](https://arxiv.org/pdf/2404.02524)</summary>

**Abstract:** Existing traffic simulation models often fall short in capturing the intricacies of real-world scenarios, particularly the interactive behaviors among multiple traffic participants, thereby limiting their utility in the evaluation and validation of autonomous driving systems. We introduce Versatile Behavior Diffusion (VBD), a novel traffic scenario generation framework based on diffusion generative models that synthesizes scene-consistent, realistic, and controllable multi-agent interactions. VBD achieves strong performance in closed-loop traffic simulation, generating scene-consistent agent behaviors that reflect complex agent interactions. A key capability of VBD is inference-time scenario editing through multi-step refinement, guided by behavior priors and model-based optimization objectives, enabling flexible and controllable behavior generation. Despite being trained on real-world traffic datasets with only normal conditions, we introduce conflict-prior and game-theoretic guidance approaches. These approaches enable the generation of interactive, customizable, or long-tail safety-critical scenarios, which are essential for comprehensive testing and validation of autonomous driving systems. Extensive experiments validate the effectiveness and versatility of VBD and highlight its promise as a foundational tool for advancing traffic simulation and autonomous vehicle development. Project website: this https URL

**arXiv ID:** 2404.02524
</details>

<details>
<summary><strong>Collision-free Source Seeking and Flocking Control of Multi-agents with Connectivity Preservation</strong> - Tinghua Li, Bayu Jayawardhana - [[pdf]](https://arxiv.org/pdf/2301.04576)</summary>

**Abstract:** In this article, we present a distributed source-seeking and flocking control method for networked multi-agent systems with non-holonomic constraints. Based solely on identical on-board sensor systems, which measure the source local field, the group objective is attained by appointing a leader agent to seek the source while the remaining follower agents safely form a cohesive flocking with their neighbors using a distributed flocking control law in a connectivity-preserved undirected network. To guarantee safe separation and group motion for all agents and to solve the conflicts with the "cohesion" flocking rule of Reynolds, the distributed control algorithm is solved individually through feasible CBF-based optimization problem with complex constraints, which guarantees the inter-agent collision avoidance and connectivity preservation. Stability analysis of the closed-loop system is presented and the efficacy of the methods is shown in simulation results.

**arXiv ID:** 2301.04576
</details>

<details>
<summary><strong>Towards AI as Colleagues: Multi-Agent System Improves Structured Ideation Processes</strong> - Kexin Quan, Dina Albassam, Mengke Wu, Zijian Ding, Jessie Chin - [[pdf]](https://arxiv.org/pdf/2510.23904)</summary>

**Abstract:** Most AI systems today are designed to manage tasks and execute predefined steps. This makes them effective for process coordination but limited in their ability to engage in joint problem-solving with humans or contribute new ideas. We introduce MultiColleagues, a multi-agent conversational system that shows how AI agents can act as colleagues by conversing with each other, sharing new ideas, and actively involving users in collaborative ideation processes. In a within-subjects study with 20 participants, we compared MultiColleagues to a single-agent baseline. Results show that MultiColleagues fostered stronger perceived social presence, and participants rated their outcomes as higher in quality and novelty, with more elaboration during ideation. These findings demonstrate the potential of AI agents to move beyond process partners toward colleagues that share intent, strengthen group dynamics, and collaborate with humans to advance ideas.

**arXiv ID:** 2510.23904
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>Position: Agentic Evolution is the Path to Evolving LLMs</strong> - Minhua Lin, Hanqing Lu, Zhan Shi, Bing He, Rui Mao, Zhiwei Zhang, Zongyu Wu, Xianfeng Tang, Hui Liu, Zhenwei Dai, Xiang Zhang, Suhang Wang, Benoit Dumoulin, Jian Pei - [[pdf]](https://arxiv.org/pdf/2602.00359)</summary>

**Abstract:** As Large Language Models (LLMs) move from curated training sets into open-ended real-world environments, a fundamental limitation emerges: static training cannot keep pace with continual deployment environment change. Scaling training-time and inference-time compute improves static capability but does not close this train-deploy gap. We argue that addressing this limitation requires a new scaling axis-evolution. Existing deployment-time adaptation methods, whether parametric fine-tuning or heuristic memory accumulation, lack the strategic agency needed to diagnose failures and produce durable improvements. Our position is that agentic evolution represents the inevitable future of LLM adaptation, elevating evolution itself from a fixed pipeline to an autonomous evolver agent. We instantiate this vision in a general framework, A-Evolve, which treats deployment-time improvement as a deliberate, goal-directed optimization process over persistent system state. We further propose the evolution-scaling hypothesis: the capacity for adaptation scales with the compute allocated to evolution, positioning agentic evolution as a scalable path toward sustained, open-ended adaptation in the real world.

**arXiv ID:** 2602.00359
</details>

<details>
<summary><strong>Engineering AI Agents for Clinical Workflows: A Case Study in Architecture,MLOps, and Governance</strong> - Cláudio Lúcio do Val Lopes, João Marcus Pitta, Fabiano Belém, Gildson Alves, Flávio Vinícius Cruzeiro Martins - [[pdf]](https://arxiv.org/pdf/2602.00751)</summary>

**Abstract:** The integration of Artificial Intelligence (AI) into clinical settings presents a software engineering challenge, demanding a shift from isolated models to robust, governable, and reliable systems. However, brittle, prototype-derived architectures often plague industrial applications and a lack of systemic oversight, creating a ``responsibility vacuum'' where safety and accountability are compromised. This paper presents an industry case study of the ``Maria'' platform, a production-grade AI system in primary healthcare that addresses this gap.
Our central hypothesis is that trustworthy clinical AI is achieved through the holistic integration of four foundational engineering pillars. We present a synergistic architecture that combines Clean Architecture for maintainability with an Event-driven architecture for resilience and auditability. We introduce the Agent as the primary unit of modularity, each possessing its own autonomous MLOps lifecycle. Finally, we show how a Human-in-the-Loop governance model is technically integrated not merely as a safety check, but as a critical, event-driven data source for continuous improvement. We present the platform as a reference architecture, offering practical lessons for engineers building maintainable, scalable, and accountable AI-enabled systems in high-stakes domains.

**arXiv ID:** 2602.00751
</details>

<details>
<summary><strong>SIDiffAgent: Self-Improving Diffusion Agent</strong> - Shivank Garg, Ayush Singh, Gaurav Kumar Nayak - [[pdf]](https://arxiv.org/pdf/2602.02051)</summary>

**Abstract:** Text-to-image diffusion models have revolutionized generative AI, enabling high-quality and photorealistic image synthesis. However, their practical deployment remains hindered by several limitations: sensitivity to prompt phrasing, ambiguity in semantic interpretation (e.g., ``mouse" as animal vs. a computer peripheral), artifacts such as distorted anatomy, and the need for carefully engineered input prompts. Existing methods often require additional training and offer limited controllability, restricting their adaptability in real-world applications. We introduce Self-Improving Diffusion Agent (SIDiffAgent), a training-free agentic framework that leverages the Qwen family of models (Qwen-VL, Qwen-Image, Qwen-Edit, Qwen-Embedding) to address these challenges. SIDiffAgent autonomously manages prompt engineering, detects and corrects poor generations, and performs fine-grained artifact removal, yielding more reliable and consistent outputs. It further incorporates iterative self-improvement by storing a memory of previous experiences in a database. This database of past experiences is then used to inject prompt-based guidance at each stage of the agentic pipeline. \modelour achieved an average VQA score of 0.884 on GenAIBench, significantly outperforming open-source, proprietary models and agentic methods. We will publicly release our code upon acceptance.

**arXiv ID:** 2602.02051
</details>

<details>
<summary><strong>Large-Scale Terminal Agentic Trajectory Generation from Dockerized Environments</strong> - Siwei Wu, Yizhi Li, Yuyang Song, Wei Zhang, Yang Wang, Riza Batista-Navarro, Xian Yang, Mingjie Tang, Bryan Dai, Jian Yang, Chenghua Lin - [[pdf]](https://arxiv.org/pdf/2602.01244)</summary>

**Abstract:** Training agentic models for terminal-based tasks critically depends on high-quality terminal trajectories that capture realistic long-horizon interactions across diverse domains. However, constructing such data at scale remains challenging due to two key requirements: \textbf{\emph{Executability}}, since each instance requires a suitable and often distinct Docker environment; and \textbf{\emph{Verifiability}}, because heterogeneous task outputs preclude unified, standardized verification. To address these challenges, we propose \textbf{TerminalTraj}, a scalable pipeline that (i) filters high-quality repositories to construct Dockerized execution environments, (ii) generates Docker-aligned task instances, and (iii) synthesizes agent trajectories with executable validation code. Using TerminalTraj, we curate 32K Docker images and generate 50,733 verified terminal trajectories across eight domains. Models trained on this data with the Qwen2.5-Coder backbone achieve consistent performance improvements on TerminalBench (TB), with gains of up to 20\% on TB~1.0 and 10\% on TB~2.0 over their respective backbones. Notably, \textbf{TerminalTraj-32B} achieves strong performance among models with fewer than 100B parameters, reaching 35.30\% on TB~1.0 and 22.00\% on TB~2.0, and demonstrates improved test-time scaling behavior. All code and data are available at this https URL.

**arXiv ID:** 2602.01244
</details>

<details>
<summary><strong>From Pragmas to Partners: A Symbiotic Evolution of Agentic High-Level Synthesis</strong> - Niansong Zhang, Sunwoo Kim, Shreesha Srinath, Zhiru Zhang - [[pdf]](https://arxiv.org/pdf/2602.01401)</summary>

**Abstract:** The rise of large language models has sparked interest in AI-driven hardware design, raising the question: does high-level synthesis (HLS) still matter in the agentic era? We argue that HLS remains essential. While we expect mature agentic hardware systems to leverage both HLS and RTL, this paper focuses on HLS and its role in enabling agentic optimization. HLS offers faster iteration cycles, portability, and design permutability that make it a natural layer for agentic optimization. This position paper makes three contributions. First, we explain why HLS serves as a practical abstraction layer and a golden reference for agentic hardware design. Second, we identify key limitations of current HLS tools, namely inadequate performance feedback, rigid interfaces, and limited debuggability that agents are uniquely positioned to address. Third, we propose a taxonomy for the symbiotic evolution of agentic HLS, clarifying how responsibility shifts from human designers to AI agents as systems advance from copilots to autonomous design partners.

**arXiv ID:** 2602.01401
</details>

<details>
<summary><strong>Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation</strong> - Zhanghao Hu, Qinglin Zhu, Hanqi Yan, Yulan He, Lin Gui - [[pdf]](https://arxiv.org/pdf/2602.02007)</summary>

**Abstract:** Agent memory systems often adopt the standard Retrieval-Augmented Generation (RAG) pipeline, yet its underlying assumptions differ in this setting. RAG targets large, heterogeneous corpora where retrieved passages are diverse, whereas agent memory is a bounded, coherent dialogue stream with highly correlated spans that are often duplicates. Under this shift, fixed top-$k$ similarity retrieval tends to return redundant context, and post-hoc pruning can delete temporally linked prerequisites needed for correct reasoning. We argue retrieval should move beyond similarity matching and instead operate over latent components, following decoupling to aggregation: disentangle memories into semantic components, organise them into a hierarchy, and use this structure to drive retrieval. We propose xMemory, which builds a hierarchy of intact units and maintains a searchable yet faithful high-level node organisation via a sparsity--semantics objective that guides memory split and merge. At inference, xMemory retrieves top-down, selecting a compact, diverse set of themes and semantics for multi-fact queries, and expanding to episodes and raw messages only when it reduces the reader's uncertainty. Experiments on LoCoMo and PerLTQA across the three latest LLMs show consistent gains in answer quality and token efficiency.

**arXiv ID:** 2602.02007
</details>

<details>
<summary><strong>RE-TRAC: REcursive TRAjectory Compression for Deep Search Agents</strong> - Jialiang Zhu, Gongrui Zhang, Xiaolong Ma, Lin Xu, Miaosen Zhang, Ruiqi Yang, Song Wang, Kai Qiu, Zhirong Wu, Qi Dai, Ruichun Ma, Bei Liu, Yifan Yang, Chong Luo, Zhengyuan Yang, Linjie Li, Lijuan Wang, Weizhu Chen, Xin Geng, Baining Guo - [[pdf]](https://arxiv.org/pdf/2602.02486)</summary>

**Abstract:** LLM-based deep research agents are largely built on the ReAct framework. This linear design makes it difficult to revisit earlier states, branch into alternative search directions, or maintain global awareness under long contexts, often leading to local optima, redundant exploration, and inefficient search. We propose Re-TRAC, an agentic framework that performs cross-trajectory exploration by generating a structured state representation after each trajectory to summarize evidence, uncertainties, failures, and future plans, and conditioning subsequent trajectories on this state representation. This enables iterative reflection and globally informed planning, reframing research as a progressive process. Empirical results show that Re-TRAC consistently outperforms ReAct by 15-20% on BrowseComp with frontier LLMs. For smaller models, we introduce Re-TRAC-aware supervised fine-tuning, achieving state-of-the-art performance at comparable scales. Notably, Re-TRAC shows a monotonic reduction in tool calls and token usage across rounds, indicating progressively targeted exploration driven by cross-trajectory reflection rather than redundant search.

**arXiv ID:** 2602.02486
</details>

<details>
<summary><strong>Path Tracking with Dynamic Control Point Blending for Autonomous Vehicles: An Experimental Study</strong> - Alexandre Lombard, Florent Perronnet, Nicolas Gaud, Abdeljalil Abbas-Turki - [[pdf]](https://arxiv.org/pdf/2602.01892)</summary>

**Abstract:** This paper presents an experimental study of a path-tracking framework for autonomous vehicles in which the lateral control command is applied to a dynamic control point along the wheelbase. Instead of enforcing a fixed reference at either the front or rear axle, the proposed method continuously interpolates between both, enabling smooth adaptation across driving contexts, including low-speed maneuvers and reverse motion. The lateral steering command is obtained by barycentric blending of two complementary controllers: a front-axle Stanley formulation and a rear-axle curvature-based geometric controller, yielding continuous transitions in steering behavior and improved tracking stability. In addition, we introduce a curvature-aware longitudinal control strategy based on virtual track borders and ray-tracing, which converts upcoming geometric constraints into a virtual obstacle distance and regulates speed accordingly. The complete approach is implemented in a unified control stack and validated in simulation and on a real autonomous vehicle equipped with GPS-RTK, radar, odometry, and IMU. The results in closed-loop tracking and backward maneuvers show improved trajectory accuracy, smoother steering profiles, and increased adaptability compared to fixed control-point baselines.

**arXiv ID:** 2602.01892
</details>

<details>
<summary><strong>From Junior to Senior: Allocating Agency and Navigating Professional Growth in Agentic AI-Mediated Software Engineering</strong> - Dana Feng, Bhada Yun, April Wang - [[pdf]](https://arxiv.org/pdf/2602.00496)</summary>

**Abstract:** Juniors enter as AI-natives, seniors adapted mid-career. AI is not just changing how engineers code-it is reshaping who holds agency across work and professional growth. We contribute junior-senior accounts on their usage of agentic AI through a three-phase mixed-methods study: ACTA combined with a Delphi process with 5 seniors, an AI-assisted debugging task with 10 juniors, and blind reviews of junior prompt histories by 5 more seniors. We found that agency in software engineering is primarily constrained by organizational policies rather than individual preferences, with experienced developers maintaining control through detailed delegation while novices struggle between over-reliance and cautious avoidance. Seniors leverage pre-AI foundational instincts to steer modern tools and possess valuable perspectives for mentoring juniors in their early AI-encouraged career development. From synthesis of results, we suggest three practices that focus on preserving agency in software engineering for coding, learning, and mentorship, especially as AI grows increasingly autonomous.

**arXiv ID:** 2602.00496
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (57 papers)</h2></summary>

<details>
<summary><strong>KEPO: Knowledge-Enhanced Preference Optimization for Reinforcement Learning with Reasoning</strong> - Fan Yang, Rui Meng, Trudi Di Qi, Ali Ezzati, Yuxin Wen - [[pdf]](https://arxiv.org/pdf/2602.00400)</summary>

**Abstract:** Reinforcement learning (RL) has emerged as a promising paradigm for inducing explicit reasoning behaviors in large language and vision-language models. However, reasoning-oriented RL post-training remains fundamentally challenging due to sparse trajectory-level rewards, leading to ambiguous credit assignment and severe exploration failures that can trap the policy in a ``learning cliff.'' Recent on-policy distillation methods introduce dense teacher supervision to stabilize optimization, but apply it uniformly across all generated trajectories. We argue that such uniform distillation is ill-suited for reasoning-intensive tasks, as low-quality on-policy trajectories often originate from early logical errors, and distillation under flawed contexts injects noisy and misaligned gradients. To address these challenges, we propose Knowledge-Enhanced Preference Optimization (KEPO), a unified post-training framework that integrates: (i) a quality-gated on-policy distillation objective that selectively applies dense teacher guidance only to high-quality trajectories, and (ii) a knowledge-enhanced exploration strategy that leverages hints learned from a teacher model to rejectively sample reward-positive on-policy trajectories for RL, thereby mitigating exploration collapse. Evaluated on a challenging medical visual question answering benchmark under single-source generalization, KEPO demonstrates improved training stability, more coherent reasoning behaviors, and superior out-of-distribution performance over reinforcement learning and on-policy distillation baselines.

**arXiv ID:** 2602.00400
</details>

<details>
<summary><strong>How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use</strong> - Minhua Lin, Enyan Dai, Hui Liu, Xianfeng Tang, Yuliang Yan, Zhenwei Dai, Jingying Zeng, Zhiwei Zhang, Fali Wang, Hongcheng Gao, Chen Luo, Xiang Zhang, Qi He, Suhang Wang - [[pdf]](https://arxiv.org/pdf/2602.00528)</summary>

**Abstract:** As Large Language Models (LLMs) are increasingly applied in high-stakes domains, their ability to reason strategically under uncertainty becomes critical. Poker provides a rigorous testbed, requiring not only strong actions but also principled, game-theoretic reasoning. In this paper, we conduct a systematic study of LLMs in multiple realistic poker tasks, evaluating both gameplay outcomes and reasoning traces. Our analysis reveals LLMs fail to compete against traditional algorithms and identifies three recurring flaws: reliance on heuristics, factual misunderstandings, and a "knowing-doing" gap where actions diverge from reasoning. An initial attempt with behavior cloning and step-level reinforcement learning improves reasoning style but remains insufficient for accurate game-theoretic play. Motivated by these limitations, we propose ToolPoker, a tool-integrated reasoning framework that combines external solvers for GTO-consistent actions with more precise professional-style explanations. Experiments demonstrate that ToolPoker achieves state-of-the-art gameplay while producing reasoning traces that closely reflect game-theoretic principles.

**arXiv ID:** 2602.00528
</details>

<details>
<summary><strong>World Models as an Intermediary between Agents and the Real World</strong> - Sherry Yang - [[pdf]](https://arxiv.org/pdf/2602.00785)</summary>

**Abstract:** Large language model (LLM) agents trained using reinforcement learning has achieved superhuman performance in low-cost environments like games, mathematics, and coding. However, these successes have not translated to complex domains where the cost of interaction is high, such as the physical cost of running robots, the time cost of ML engineering, and the resource cost of scientific experiments. The true bottleneck for achieving the next level of agent performance for these complex and high-cost domains lies in the expense of executing actions to acquire reward signals. To address this gap, this paper argues that we should use world models as an intermediary between agents and the real world. We discuss how world models, viewed as models of dynamics, rewards, and task distributions, can overcome fundamental barriers of high-cost actions such as extreme off-policy learning and sample inefficiency in long-horizon tasks. Moreover, we demonstrate how world models can provide critical and rich learning signals to agents across a broad set of domains, including machine learning engineering, computer use, robotics, and AI for science. Lastly, we identify the challenges of building these world models and propose actionable items along dataset curation, architecture design, scaling, and evaluation of world models.

**arXiv ID:** 2602.00785
</details>

<details>
<summary><strong>MedBeads: An Agent-Native, Immutable Data Substrate for Trustworthy Medical AI</strong> - Takahito Nakajima - [[pdf]](https://arxiv.org/pdf/2602.01086)</summary>

**Abstract:** Background: As of 2026, Large Language Models (LLMs) demonstrate expert-level medical knowledge. However, deploying them as autonomous "Clinical Agents" remains limited. Current Electronic Medical Records (EMRs) and standards like FHIR are designed for human review, creating a "Context Mismatch": AI agents receive fragmented data and must rely on probabilistic inference (e.g., RAG) to reconstruct patient history. This approach causes hallucinations and hinders auditability. Methods: We propose MedBeads, an agent-native data infrastructure where clinical events are immutable "Beads"--nodes in a Merkle Directed Acyclic Graph (DAG)--cryptographically referencing causal predecessors. This "write-once, read-many" architecture makes tampering mathematically detectable. We implemented a prototype with a Go Core Engine, Python middleware for LLM integration, and a React-based visualization interface. Results: We successfully implemented the workflow using synthetic data. The FHIR-to-DAG conversion transformed flat resources into a causally-linked graph. Our Breadth-First Search (BFS) Context Retrieval algorithm traverses relevant subgraphs with O(V+E) complexity, enabling real-time decision support. Tamper-evidence is guaranteed by design: any modification breaks the cryptographic chain. The visualization aids clinician understanding through explicit causal links. Conclusion: MedBeads addresses the "Context Mismatch" by shifting from probabilistic search to deterministic graph traversal, and from mutable records to immutable chains, providing the substrate for "Trustworthy Medical AI." It guarantees the context the AI receives is deterministic and tamper-evident, while the LLM determines interpretation. The structured Bead format serves as a token-efficient "AI-native language." We release MedBeads as open-source software to accelerate agent-native data standards.

**arXiv ID:** 2602.01086
</details>

<details>
<summary><strong>S1-NexusAgent: a Self-Evolving Agent Framework for Multidisciplinary Scientific Research</strong> - S1-NexusAgent Team - [[pdf]](https://arxiv.org/pdf/2602.01550)</summary>

**Abstract:** Modern scientific research relies on large-scale data, complex workflows, and specialized tools, which existing LLMs and tool-based agents struggle to handle due to limitations in long-horizon planning, robust goal maintenance, and continual learning from execution. To address these issues, in this work, we propose S1-NexusAgent, a self-evolving agent framework designed for multidisciplinary scientific research. S1-NexusAgent adopts a hierarchical Plan-and-CodeAct execution paradigm, decoupling global scientific planning from subtask-level tool execution through a dual-loop architecture, thereby enabling stable modeling of complex research workflows. The system natively supports the Model Context Protocol (MCP), integrates up to thousands of cross-disciplinary scientific tools, and achieves efficient orchestration of heterogeneous research tools via intention-aware dynamic tool retrieval and hot-plug mechanisms. To address long-context and large-scale data challenges in scientific settings, S1-NexusAgent introduces object-reference-based sparse context management, which enables sub-task context isolation and intermediate result compression. Building on this, a Critic Agent automatically evaluates complete execution trajectories and distills high-quality research paths into reusable Scientific Skills, forming a closed loop for continuous self-evolution, which is valuable for sustainable and long-horizon scientific research. Experiments on authoritative scientific benchmarks involving long-horizon planning and complex specialized tool orchestration, including biomini-eval (biology), ChemBench (chemistry), and MatSciBench (material science), demonstrate that S1-NexusAgent achieves state-of-the-art performance, validating its effectiveness and generalization capability in complex scientific tasks.

**arXiv ID:** 2602.01550
</details>

<details>
<summary><strong>FlowSteer: Interactive Agentic Workflow Orchestration via End-to-End Reinforcement Learning</strong> - Mingda Zhang, Haoran Luo, Tiesunlong Shen, Qika Lin, Xiaoying Tang, Rui Mao, Erik Cambria - [[pdf]](https://arxiv.org/pdf/2602.01664)</summary>

**Abstract:** In recent years, a variety of powerful agentic workflows have been applied to solve a wide range of human problems. However, existing workflow orchestration still faces key challenges, including high manual cost, reliance on specific operators/large language models (LLMs), and sparse reward signals. To address these challenges, we propose FlowSteer, an end-to-end reinforcement learning framework that takes a lightweight policy model as the agent and an executable canvas environment, automating workflow orchestration through multi-turn interaction. In this process, the policy model analyzes execution states and selects editing actions, while the canvas executes operators and returns feedback for iterative refinement. Moreover, FlowSteer provides a plug-and-play framework that supports diverse operator libraries and interchangeable LLM backends. To effectively train this interaction paradigm, we propose Canvas Workflow Relative Policy Optimization (CWRPO), which introduces diversity-constrained rewards with conditional release to stabilize learning and suppress shortcut behaviors. Experimental results on twelve datasets show that FlowSteer significantly outperforms baselines across various tasks.

**arXiv ID:** 2602.01664
</details>

<details>
<summary><strong>TRIP-Bench: A Benchmark for Long-Horizon Interactive Agents in Real-World Scenarios</strong> - Yuanzhe Shen, Zisu Huang, Zhengyuan Wang, Muzhao Tian, Zhengkang Guo, Chenyang Zhang, Shuaiyu Zhou, Zengjie Hu, Dailin Li, Jingwen Xu, Kaimin Wang, Wenhao Liu, Tianlong Li, Fengpeng Yue, Feng Hong, Cao Liu, Ke Zeng - [[pdf]](https://arxiv.org/pdf/2602.01675)</summary>

**Abstract:** As LLM-based agents are deployed in increasingly complex real-world settings, existing benchmarks underrepresent key challenges such as enforcing global constraints, coordinating multi-tool reasoning, and adapting to evolving user behavior over long, multi-turn interactions. To bridge this gap, we introduce \textbf{TRIP-Bench}, a long-horizon benchmark grounded in realistic travel-planning scenarios. TRIP-Bench leverages real-world data, offers 18 curated tools and 40+ travel requirements, and supports automated evaluation. It includes splits of varying difficulty; the hard split emphasizes long and ambiguous interactions, style shifts, feasibility changes, and iterative version revision. Dialogues span up to 15 user turns, can involve 150+ tool calls, and may exceed 200k tokens of context. Experiments show that even advanced models achieve at most 50\% success on the easy split, with performance dropping below 10\% on hard subsets. We further propose \textbf{GTPO}, an online multi-turn reinforcement learning method with specialized reward normalization and reward differencing. Applied to Qwen2.5-32B-Instruct, GTPO improves constraint satisfaction and interaction robustness, outperforming Gemini-3-Pro in our evaluation. We expect TRIP-Bench to advance practical long-horizon interactive agents, and GTPO to provide an effective online RL recipe for robust long-horizon training.

**arXiv ID:** 2602.01675
</details>

<details>
<summary><strong>Reinforcement Learning via Conservative Agent for Environments with Random Delays</strong> - Jongsoo Lee, Jangwon Kim, Jiseok Jeong, Soohee Han - [[pdf]](https://arxiv.org/pdf/2507.18992)</summary>

**Abstract:** Real-world reinforcement learning applications are often hindered by delayed feedback from environments, which violates the Markov assumption and introduces significant challenges. Although numerous delay-compensating methods have been proposed for environments with constant delays, environments with random delays remain largely unexplored due to their inherent variability and unpredictability. In this study, we propose a simple yet robust agent for decision-making under random delays, termed the conservative agent, which reformulates the random-delay environment into its constant-delay equivalent. This transformation enables any state-of-the-art constant-delay method to be directly extended to the random-delay environments without modifying the algorithmic structure or sacrificing performance. We evaluate the conservative agent-based algorithm on continuous control tasks, and empirical results demonstrate that it significantly outperforms existing baseline algorithms in terms of asymptotic performance and sample efficiency.

**arXiv ID:** 2507.18992
</details>

<details>
<summary><strong>AutoBinder Agent: An MCP-Based Agent for End-to-End Protein Binder Design</strong> - Fukang Ge, Jiarui Zhu, Linjie Zhang, Haowen Xiao, Xiangcheng Bao, Fangnan Xie, Danyang Chen, Yanrui Lu, Yuting Wang, Ziqian Guan, Lin Gu, Jinhao Bi, Yingying Zhu - [[pdf]](https://arxiv.org/pdf/2602.00019)</summary>

**Abstract:** Modern AI technologies for drug discovery are distributed across heterogeneous platforms-including web applications, desktop environments, and code libraries-leading to fragmented workflows, inconsistent interfaces, and high integration overhead. We present an agentic end-to-end drug design framework that leverages a Large Language Model (LLM) in conjunction with the Model Context Protocol (MCP) to dynamically coordinate access to biochemical databases, modular toolchains, and task-specific AI models. The system integrates four state-of-the-art components: MaSIF (MaSIF-site and MaSIF-seed-search) for geometric deep learning-based identification of protein-protein interaction (PPI) sites, Rosetta for grafting protein fragments onto protein backbones to form mini proteins, ProteinMPNN for amino acid sequences redesign, and AlphaFold3 for near-experimental accuracy in complex structure prediction. Starting from a target structure, the framework supports de novo binder generation via surface analysis, scaffold grafting and pose construction, sequence optimization, and structure prediction. Additionally, by replacing rigid, script-based workflows with a protocol-driven, LLM-coordinated architecture, the framework improves reproducibility, reduces manual overhead, and ensures extensibility, portability, and auditability across the entire drug design process.

**arXiv ID:** 2602.00019
</details>

<details>
<summary><strong>Representation Learning Enhanced Deep Reinforcement Learning for Optimal Operation of Hydrogen-based Multi-Energy Systems</strong> - Zhenyu Pu, Yu Yang, Lun Yang, Qing-Shan Jia, Xiaohong Guan, Costas J. Spanos - [[pdf]](https://arxiv.org/pdf/2602.00027)</summary>

**Abstract:** Hydrogen-based multi-energy systems (HMES) have emerged as a promising low-carbon and energy-efficient solution, as it can enable the coordinated operation of electricity, heating and cooling supply and demand to enhance operational flexibility, improve overall energy efficiency, and increase the share of renewable integration. However, the optimal operation of HMES remains challenging due to the nonlinear and multi-physics coupled dynamics of hydrogen energy storage systems (HESS) (consisting of electrolyters, fuel cells and hydrogen tanks) as well as the presence of multiple uncertainties from supply and demand. To address these challenges, this paper develops a comprehensive operational model for HMES that fully captures the nonlinear dynamics and multi-physics process of HESS. Moreover, we propose an enhanced deep reinforcement learning (DRL) framework by integrating the emerging representation learning techniques, enabling substantially accelerated and improved policy optimization for spatially and temporally coupled complex networked systems, which is not provided by conventional DRL. Experimental studies based on real-world datasets show that the comprehensive model is crucial to ensure the safe and reliable of HESS. In addition, the proposed SR-DRL approaches demonstrate superior convergence rate and performance over conventional DRL counterparts in terms of reducing the operation cost of HMES and handling the system operating constraints. Finally, we provide some insights into the role of representation learning in DRL, speculating that it can reorganize the original state space into a well-structured and cluster-aware geometric representation, thereby smoothing and facilitating the learning process of DRL.

**arXiv ID:** 2602.00027
</details>

<details>
<summary><strong>ProDCARL: Reinforcement Learning-Aligned Diffusion Models for De Novo Antimicrobial Peptide Design</strong> - Fang Sheng, Mohammad Noaeen, Zahra Shakeri - [[pdf]](https://arxiv.org/pdf/2602.00157)</summary>

**Abstract:** Antimicrobial resistance threatens healthcare sustainability and motivates low-cost computational discovery of antimicrobial peptides (AMPs). De novo peptide generation must optimize antimicrobial activity and safety through low predicted toxicity, but likelihood-trained generators do not enforce these goals explicitly. We introduce ProDCARL, a reinforcement-learning alignment framework that couples a diffusion-based protein generator (EvoDiff OA-DM 38M) with sequence property predictors for AMP activity and peptide toxicity. We fine-tune the diffusion prior on AMP sequences to obtain a domain-aware generator. Top-k policy-gradient updates use classifier-derived rewards plus entropy regularization and early stopping to preserve diversity and reduce reward hacking. In silico experiments show ProDCARL increases the mean predicted AMP score from 0.081 after fine-tuning to 0.178. The joint high-quality hit rate reaches 6.3\% with pAMP $>$0.7 and pTox $<$0.3. ProDCARL maintains high diversity, with $1-$mean pairwise identity equal to 0.929. Qualitative analyses with AlphaFold3 and ProtBERT embeddings suggest candidates show plausible AMP-like structural and semantic characteristics. ProDCARL serves as a candidate generator that narrows experimental search space, and experimental validation remains future work.

**arXiv ID:** 2602.00157
</details>

<details>
<summary><strong>Replicating the behaviour of electric vehicle drivers using an agent-based reinforcement learning model</strong> - Zixin Feng, Qunshan Zhao, Alison Heppenstall - [[pdf]](https://arxiv.org/pdf/2507.21341)</summary>

**Abstract:** Despite the rapid expansion of electric vehicle (EV) charging networks, questions remain about their efficiency in meeting the growing needs of EV drivers. Previous simulation-based approaches, which rely on static behavioural rules, have struggled to capture the adaptive behaviours of human drivers. Although reinforcement learning has been introduced in EV simulation studies, its application has primarily focused on optimising fleet operations rather than modelling private drivers who make independent charging decisions. To address the gap, we propose a multi-stage reinforcement learning framework that simulates charging demand of private EV drivers across a national-scale road network. We validate the model against real-world data and identify the training stage that most closely reflects actual driver behaviour, which captures both the adaptive behaviours and bounded rationality of private drivers. Based on the simulation results, we also identify critical 'charging deserts' where EV drivers consistently have low state of charge. Our findings also highlight recent policy shifts toward expanding rapid charging hubs along motorway corridors and city boundaries to meet the demand from long-distance trips.

**arXiv ID:** 2507.21341
</details>

<details>
<summary><strong>DMV-AVP: Distributed Multi-Vehicle Autonomous Valet Parking Using Autoware</strong> - Zubair Islam, Mohamed El-Darieby - [[pdf]](https://arxiv.org/pdf/2601.16327)</summary>

**Abstract:** This paper presents DMV-AVP, a distributed simulation of Multi-Vehicle Autonomous Valet Parking (AVP). The system was implemented as an application of the Distributed Multi-Autonomous Vehicle Architecture (DMAVA) for synchronized multi-host execution. Most existing simulation approaches rely on centralized or non-distributed designs that constrain scalability and limit fully autonomous control. This work introduces two modules built on top of DMAVA: 1) the Multi-Vehicle AVP Coordination Framework, composed of AVP Managers and a per-vehicle AVP Node, is responsible for global parking state tracking, vehicle queuing, parking spot reservation, lifecycle coordination, and conflict resolution across multiple vehicles, and 2) the Unity-Integrated YOLOv5 Parking Spot Detection Module, that provides real-time, vision-based perception within AWSIM Labs. Both modules integrate seamlessly with DMAVA and extend it specifically for multi-vehicle AVP operation, supported by a Zenoh communication layer that ensures high data accuracy and controllability across hosts. Experiments conducted on two- and three-host configurations demonstrate consistent coordination, conflict-free parking behavior, and scalable performance across distributed Autoware instances. The results confirm that the proposed DMV-AVP supports cooperative AVP simulation and establishes a foundation for future real-world and hardware-in-the-loop validation. Demo videos and source code are available at: this https URL

**arXiv ID:** 2601.16327
</details>

<details>
<summary><strong>DMAVA: Distributed Multi-Autonomous Vehicle Architecture Using Autoware</strong> - Zubair Islam, Mohamed El-Darieby - [[pdf]](https://arxiv.org/pdf/2601.16336)</summary>

**Abstract:** Simulating and validating coordination among multiple autonomous vehicles remains challenging, as many existing simulation architectures are limited to single-vehicle operation or rely on centralized control. This paper presents the Distributed Multi-Autonomous Vehicle Architecture (DMAVA), a simulation architecture that enables concurrent execution of multiple independent vehicle autonomy stacks distributed across multiple physical hosts within a shared simulation environment. Each vehicle operates its own complete autonomous driving stack while maintaining coordinated behavior through a data-centric communication layer. The proposed system integrates ROS 2 Humble, Autoware Universe, AWSIM Labs, and Zenoh to support high data accuracy and controllability during multi-vehicle simulation, enabling consistent perception, planning, and control behavior under distributed execution. Experiments conducted on multiple-host configurations demonstrate stable localization, reliable inter-host communication, and consistent closed-loop control under distributed execution. DMAVA also serves as a foundation for Multi-Vehicle Autonomous Valet Parking, demonstrating its extensibility toward higher-level cooperative autonomy. Demo videos and source code are available at: this https URL.

**arXiv ID:** 2601.16336
</details>

<details>
<summary><strong>Adaptive Ability Decomposing for Unlocking Large Reasoning Model Effective Reinforcement Learning</strong> - Zhipeng Chen, Xiaobo Qin, Wayne Xin Zhao, Youbin Wu, Ji-Rong Wen - [[pdf]](https://arxiv.org/pdf/2602.00759)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has shown great potential to enhance the reasoning ability of large language models (LLMs). However, due to the limited amount of information provided during the RLVR process, the model can only engage in largely blind exploration, which often results in failure on challenging problems. To provide additional information for the RLVR process without relying on a teacher model, we propose A$^2$D, an Adaptive Ability Decomposing method for enhancing the effectiveness of RLVR. Specifically, we first train a decomposer via RLVR without distillation, enabling it to decompose complex questions into a set of simpler sub-questions. Next, we use this decomposer to annotate sub-questions for each question in the training dataset, and then train the reasoner under RLVR with sub-question guidance. To better understand A$^2$D, we first compare its performance with competitive baselines, showing its effectiveness. Next, we observe that our method functions as a plug-and-play module that can be applied to different RLVR algorithms. Furthermore, we conduct an analysis of the decomposer, revealing how the RLVR process affects its performance and behavior, and which type of guidance is better suited for enhancing the reasoner's exploration and exploitation abilities.

**arXiv ID:** 2602.00759
</details>

<details>
<summary><strong>DISPO: Enhancing Training Efficiency and Stability in Reinforcement Learning for Large Language Model Mathematical Reasoning</strong> - Batuhan K. Karaman, Aditya Rawal, Suhaila Shakiah, Mohammad Ghavamzadeh, Mingyi Hong, Arijit Biswas, Ruida Zhou - [[pdf]](https://arxiv.org/pdf/2602.00983)</summary>

**Abstract:** Reinforcement learning with verifiable rewards has emerged as a promising paradigm for enhancing the reasoning capabilities of large language models particularly in mathematics. Current approaches in this domain present a clear trade-off: PPO-style methods (e.g., GRPO/DAPO) offer training stability but exhibit slow learning trajectories due to their trust-region constraints on policy updates, while REINFORCE-style approaches (e.g., CISPO) demonstrate improved learning efficiency but suffer from performance instability as they clip importance sampling weights while still permitting non-zero gradients outside the trust-region. To address these limitations, we introduce DISPO, a simple yet effective REINFORCE-style algorithm that decouples the up-clipping and down-clipping of importance sampling weights for correct and incorrect responses, yielding four controllable policy update regimes. Through targeted ablations, we uncover how each regime impacts training: for correct responses, weights >1 increase the average token entropy (i.e., exploration) while weights <1 decrease it (i.e., distillation) -- both beneficial but causing gradual performance degradation when excessive. For incorrect responses, overly restrictive clipping triggers sudden performance collapse through repetitive outputs (when weights >1) or vanishing response lengths (when weights <1). By separately tuning these four clipping parameters, DISPO maintains the exploration-distillation balance while preventing catastrophic failures, achieving 61.04% on AIME'24 (vs. 55.42% CISPO and 50.21% DAPO) with similar gains across various benchmarks and models.

**arXiv ID:** 2602.00983
</details>

<details>
<summary><strong>Reliable Use of Lemmas via Eligibility Reasoning and Section$-$Aware Reinforcement Learning</strong> - Zhikun Xu, Xiaodong Yu, Ben Zhou, Jiang Liu, Jialian Wu, Ze Wang, Ximeng Sun, Hao Chen, Zicheng Liu - [[pdf]](https://arxiv.org/pdf/2602.00998)</summary>

**Abstract:** Recent large language models (LLMs) perform strongly on mathematical benchmarks yet often misapply lemmas, importing conclusions without validating assumptions. We formalize lemma$-$judging as a structured prediction task: given a statement and a candidate lemma, the model must output a precondition check and a conclusion$-$utility check, from which a usefulness decision is derived. We present RULES, which encodes this specification via a two$-$section output and trains with reinforcement learning plus section$-$aware loss masking to assign penalty to the section responsible for errors. Training and evaluation draw on diverse natural language and formal proof corpora; robustness is assessed with a held$-$out perturbation suite; and end$-$to$-$end evaluation spans competition$-$style, perturbation$-$aligned, and theorem$-$based problems across various LLMs. Results show consistent in$-$domain gains over both a vanilla model and a single$-$label RL baseline, larger improvements on applicability$-$breaking perturbations, and parity or modest gains on end$-$to$-$end tasks; ablations indicate that the two$-$section outputs and section$-$aware reinforcement are both necessary for robustness.

**arXiv ID:** 2602.00998
</details>

<details>
<summary><strong>ASTER: Agentic Scaling with Tool-integrated Extended Reasoning</strong> - Xuqin Zhang, Quan He, Zhenrui Zheng, Zongzhang Zhang, Xu He, Dong Li - [[pdf]](https://arxiv.org/pdf/2602.01204)</summary>

**Abstract:** Reinforcement learning (RL) has emerged as a dominant paradigm for eliciting long-horizon reasoning in Large Language Models (LLMs). However, scaling Tool-Integrated Reasoning (TIR) via RL remains challenging due to interaction collapse: a pathological state where models fail to sustain multi-turn tool usage, instead degenerating into heavy internal reasoning with only trivial, post-hoc code verification. We systematically study three questions: (i) how cold-start SFT induces an agentic, tool-using behavioral prior, (ii) how the interaction density of cold-start trajectories shapes exploration and downstream RL outcomes, and (iii) how the RL interaction budget affects learning dynamics and generalization under varying inference-time budgets. We then introduce ASTER (Agentic Scaling with Tool-integrated Extended Reasoning), a framework that circumvents this collapse through a targeted cold-start strategy prioritizing interaction-dense trajectories. We find that a small expert cold-start set of just 4K interaction-dense trajectories yields the strongest downstream performance, establishing a robust prior that enables superior exploration during extended RL training. Extensive evaluations demonstrate that ASTER-4B achieves state-of-the-art results on competitive mathematical benchmarks, reaching 90.0% on AIME 2025, surpassing leading frontier open-source models, including DeepSeek-V3.2-Exp.

**arXiv ID:** 2602.01204
</details>

<details>
<summary><strong>CRAFT: Calibrated Reasoning with Answer-Faithful Traces via Reinforcement Learning for Multi-Hop Question Answering</strong> - Yu Liu, Wenxiao Zhang, Cong Cao, Fangfang Yuan, Weizhuo Chen, Cheng Hu, Pin Xu, Yuling Yang, Kun Peng, Diandian Guo, Qiang Sun, Yanbing Liu, Jin B. Hong, Zhiyuan Ma - [[pdf]](https://arxiv.org/pdf/2602.01348)</summary>

**Abstract:** Retrieval-augmented generation (RAG) is widely used to ground Large Language Models (LLMs) for multi-hop question answering. Recent work mainly focused on improving answer accuracy via fine-tuning and structured or reinforcement-based optimization. However, reliable reasoning in response generation faces three challenges: 1) Reasoning Collapse. Reasoning in multi-hop QA is inherently complex due to multi-hop composition and is further destabilized by noisy retrieval. 2) Reasoning-answer inconsistency. Due to the intrinsic uncertainty of LLM generation and exposure to evidence--distractor mixtures, models may produce correct answers that are not faithfully supported by their intermediate reasoning or evidence. 3) Loss of format control. Traditional chain-of-thought generation often deviates from required structured output formats, leading to incomplete or malformed structured content. To address these challenges, we propose CRAFT (Calibrated Reasoning with Answer-Faithful Traces), a Group Relative Policy Optimization (GRPO) based reinforcement learning framework that trains models to perform faithful reasoning during response generation. CRAFT employs dual reward mechanisms to optimize multi-hop reasoning: deterministic rewards ensure structural correctness while judge-based rewards verify semantic faithfulness. This optimization framework supports controllable trace variants that enable systematic analysis of how structure and scale affect reasoning performance and faithfulness. Experiments on three multi-hop QA benchmarks show that CRAFT improves both answer accuracy and reasoning faithfulness across model scales, with the CRAFT 7B model achieving competitive performance with closed-source LLMs across multiple reasoning trace settings.

**arXiv ID:** 2602.01348
</details>

<details>
<summary><strong>Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training</strong> - Ran Xu, Tianci Liu, Zihan Dong, Tony You, Ilgee Hong, Carl Yang, Linjun Zhang, Tao Zhao, Haoyu Wang - [[pdf]](https://arxiv.org/pdf/2602.01511)</summary>

**Abstract:** Standard reward models typically predict scalar scores that fail to capture the multifaceted nature of response quality in non-verifiable domains, such as creative writing or open-ended instruction following. To address this limitation, we propose Rubric-ARM, a framework that jointly optimizes a rubric generator and a judge using reinforcement learning from preference feedback. Unlike existing methods that rely on static rubrics or disjoint training pipelines, our approach treats rubric generation as a latent action learned to maximize judgment accuracy. We introduce an alternating optimization strategy to mitigate the non-stationarity of simultaneous updates, providing theoretical analysis that demonstrates how this schedule reduces gradient variance during training. Extensive experiments show that Rubric-ARM achieves state-of-the-art performance among baselines on multiple benchmarks and significantly improves downstream policy alignment in both offline and online reinforcement learning settings.

**arXiv ID:** 2602.01511
</details>

<details>
<summary><strong>SafePred: A Predictive Guardrail for Computer-Using Agents via World Models</strong> - Yurun Chen, Zeyi Liao, Ping Yin, Taotao Xie, Keting Yin, Shengyu Zhang - [[pdf]](https://arxiv.org/pdf/2602.01725)</summary>

**Abstract:** With the widespread deployment of Computer-using Agents (CUAs) in complex real-world environments, prevalent long-term risks often lead to severe and irreversible consequences. Most existing guardrails for CUAs adopt a reactive approach, constraining agent behavior only within the current observation space. While these guardrails can prevent immediate short-term risks (e.g., clicking on a phishing link), they cannot proactively avoid long-term risks: seemingly reasonable actions can lead to high-risk consequences that emerge with a delay (e.g., cleaning logs leads to future audits being untraceable), which reactive guardrails cannot identify within the current observation space. To address these limitations, we propose a predictive guardrail approach, with the core idea of aligning predicted future risks with current decisions. Based on this approach, we present SafePred, a predictive guardrail framework for CUAs that establishes a risk-to-decision loop to ensure safe agent behavior. SafePred supports two key abilities: (1) Short- and long-term risk prediction: by using safety policies as the basis for risk prediction, SafePred leverages the prediction capability of the world model to generate semantic representations of both short-term and long-term risks, thereby identifying and pruning actions that lead to high-risk states; (2) Decision optimization: translating predicted risks into actionable safe decision guidances through step-level interventions and task-level re-planning. Extensive experiments show that SafePred significantly reduces high-risk behaviors, achieving over 97.6% safety performance and improving task utility by up to 21.4% compared with reactive baselines.

**arXiv ID:** 2602.01725
</details>

<details>
<summary><strong>Kimi K2.5: Visual Agentic Intelligence</strong> - Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, S.H. Cai, Yuan Cao, Y. Charles, H.S. Che, Cheng Chen, Guanduo Chen, Huarong Chen, Jia Chen, Jiahao Chen, Jianlong Chen, Jun Chen, Kefan Chen, Liang Chen, Ruijue Chen, Xinhao Chen, Yanru Chen, Yanxu Chen, Yicun Chen, Yimin Chen, Yingjiang Chen, Yuankun Chen, Yujie Chen, Yutian Chen, Zhirong Chen, Ziwei Chen, Dazhi Cheng, Minghan Chu, Jialei Cui, Jiaqi Deng, Muxi Diao, Hao Ding, Mengfan Dong, Mengnan Dong, Yuxin Dong, Yuhao Dong, Angang Du, Chenzhuang Du, Dikang Du, Lingxiao Du, Yulun Du, Yu Fan, Shengjun Fang, Qiulin Feng, Yichen Feng, Garimugai Fu, Kelin Fu, Hongcheng Gao, Tong Gao, Yuyao Ge, Shangyi Geng, Chengyang Gong, Xiaochen Gong, Zhuoma Gongque, Qizheng Gu, Xinran Gu, Yicheng Gu, Longyu Guan, Yuanying Guo, Xiaoru Hao, Weiran He, Wenyang He, Yunjia He, Chao Hong, Hao Hu, Jiaxi Hu, Yangyang Hu, Zhenxing Hu, Ke Huang, Ruiyuan Huang, Weixiao Huang, Zhiqi Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yu Jing, Guokun Lai, Aidi Li, C. Li, Cheng Li, Fang Li, Guanghe Li, Guanyu Li, Haitao Li, Haoyang Li, Jia Li, Jingwei Li, Junxiong Li, Lincan Li, Mo Li, Weihong Li, Wentao Li, Xinhang Li, Xinhao Li, Yang Li, Yanhao Li, Yiwei Li - [[pdf]](https://arxiv.org/pdf/2602.02276)</summary>

**Abstract:** We introduce Kimi K2.5, an open-source multimodal agentic model designed to advance general agentic intelligence. K2.5 emphasizes the joint optimization of text and vision so that two modalities enhance each other. This includes a series of techniques such as joint text-vision pre-training, zero-vision SFT, and joint text-vision reinforcement learning. Building on this multimodal foundation, K2.5 introduces Agent Swarm, a self-directed parallel agent orchestration framework that dynamically decomposes complex tasks into heterogeneous sub-problems and executes them concurrently. Extensive evaluations show that Kimi K2.5 achieves state-of-the-art results across various domains including coding, vision, reasoning, and agentic tasks. Agent Swarm also reduces latency by up to $4.5\times$ over single-agent baselines. We release the post-trained Kimi K2.5 model checkpoint to facilitate future research and real-world applications of agentic intelligence.

**arXiv ID:** 2602.02276
</details>

<details>
<summary><strong>Probing the Knowledge Boundary: An Interactive Agentic Framework for Deep Knowledge Extraction</strong> - Yuheng Yang, Siqi Zhu, Tao Feng, Ge Liu, Jiaxuan You - [[pdf]](https://arxiv.org/pdf/2602.00959)</summary>

**Abstract:** Large Language Models (LLMs) can be seen as compressed knowledge bases, but it remains unclear what knowledge they truly contain and how far their knowledge boundaries extend. Existing benchmarks are mostly static and provide limited support for systematic knowledge probing. In this paper, we propose an interactive agentic framework to systematically extract and quantify the knowledge of LLMs. Our method includes four adaptive exploration policies to probe knowledge at different granularities. To ensure the quality of extracted knowledge, we introduce a three-stage knowledge processing pipeline that combines vector-based filtering to remove exact duplicates, LLM-based adjudication to resolve ambiguous semantic overlaps, and domain-relevance auditing to retain valid knowledge units. Through extensive experiments, we find that recursive taxonomy is the most effective exploration strategy. We also observe a clear knowledge scaling law, where larger models consistently extract more knowledge. In addition, we identify a Pass@1-versus-Pass@k trade-off: domain-specialized models achieve higher initial accuracy but degrade rapidly, while general-purpose models maintain stable performance during extended extraction. Finally, our results show that differences in training data composition lead to distinct and measurable knowledge profiles across model families.

**arXiv ID:** 2602.00959
</details>

<details>
<summary><strong>Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning</strong> - Dylan Zhang, Yufeng Xu, Haojin Wang, Qingzhi Chen, Hao Peng - [[pdf]](https://arxiv.org/pdf/2602.01058)</summary>

**Abstract:** Post-training of reasoning LLMs is a holistic process that typically consists of an offline SFT stage followed by an online reinforcement learning (RL) stage. However, SFT is often optimized in isolation to maximize SFT performance alone.
We show that, after identical RL training, models initialized from stronger SFT checkpoints can significantly underperform those initialized from weaker ones. We attribute this to a mismatch typical in current SFT-RL pipelines: the distribution that generates the offline SFT data can differ substantially from the policy optimized during online RL, which learns from its own rollouts.
We propose PEAR (Policy Evaluation-inspired Algorithm for Offline Learning Loss Re-weighting), an SFT-stage method that corrects this mismatch and better prepares the model for RL. PEAR uses importance sampling to reweight the SFT loss, with three variants operating at the token, block, and sequence levels. It can be used to augment standard SFT objectives and incurs little additional training overhead once probabilities for the offline data are collected.
We conduct controlled experiments on verifiable reasoning games and mathematical reasoning tasks on Qwen 2.5 and 3 and DeepSeek-distilled models. PEAR consistently improves post-RL performance over canonical SFT, with pass at 8 gains up to a 14.6 percent on AIME2025. Our results suggest that PEAR is an effective step toward more holistic LLM post-training by designing and evaluating SFT with downstream RL in mind rather than in isolation.

**arXiv ID:** 2602.01058
</details>

<details>
<summary><strong>When Domains Interact: Asymmetric and Order-Sensitive Cross-Domain Effects in Reinforcement Learning for Reasoning</strong> - Wang Yang, Shouren Wang, Chaoda Song, Chuang Ma, Xinpeng Li, Nengbo Wang, Kaixiong Zhou, Vipin Chaudhary, Xiaotian Han - [[pdf]](https://arxiv.org/pdf/2602.01365)</summary>

**Abstract:** Group Relative Policy Optimization (GRPO) has become a key technique for improving reasoning abilities in large language models, yet its behavior under different domain sequencing strategies is poorly understood. In particular, the impact of sequential (one domain at a time) versus mixed-domain (multiple domain at a time) training in GRPO has not been systematically studied. We provide the first systematic analysis of training-order effects across math, science, logic, and puzzle reasoning tasks. We found (1) single-domain generalization is highly asymmetric: training on other domains improves math reasoning by approximately 25\% accuracy, while yielding negligible transfer to logic and puzzle; (2) cross-domain interactions are highly order-dependent: training in the order math$\rightarrow$science achieves 83\% / 41\% accuracy on math / science, while reversing the order to science$\rightarrow$math degrades performance to 77\% / 25\%; (3) no single strategy is universally optimal in multi-domain training: sequential training favors math (up to 84\%), mixed training favors science and logic, and poor ordering can incur large performance gaps (from 70\% to 56\%). Overall, our findings demonstrate that GRPO under multi-domain settings exhibits pronounced asymmetry, order sensitivity, and strategy dependence, highlighting the necessity of domain-aware and order-aware training design.

**arXiv ID:** 2602.01365
</details>

<details>
<summary><strong>A Relative-Budget Theory for Reinforcement Learning with Verifiable Rewards in Large Language Model Reasoning</strong> - Akifumi Wachi, Hirota Kinoshita, Shokichi Takakura, Rei Higuchi, Taiji Suzuki - [[pdf]](https://arxiv.org/pdf/2602.01523)</summary>

**Abstract:** Reinforcement learning (RL) is a dominant paradigm for improving the reasoning abilities of large language models, yet its effectiveness varies across tasks and compute budgets. We propose a \emph{relative-budget} theory explaining this variation through a single quantity called relative budget $\xi := H/\mathbb{E}[T]$, where $H$ is the generation horizon (token budget) and $T$ denotes the number of tokens until the first correct solution under a base policy. We show that $\xi$ determines sample efficiency by controlling reward variance and the likelihood of informative trajectories. Our analysis reveals three regimes: in the \emph{deficient} regime ($\xi \to 0$), informative trajectories are rare and the sample complexity explodes; in the \emph{balanced} regime ($\xi=\Theta(1)$), informative trajectories occur with non-negligible probability and RL is maximally sample-efficient; and in the \emph{ample} regime ($\xi \to \infty$), learning remains stable but marginal gains per iteration diminish. We further provide finite-sample guarantees for online RL that characterize learning progress across these regimes. Specifically, in a case study under idealized distributional assumptions, we show that the relative budget grows linearly over iterations. Our empirical results confirm these predictions in realistic settings, identifying a budget $\xi \in [1.5, 2.0]$ that maximizes learning efficiency and coincides with peak reasoning performance.

**arXiv ID:** 2602.01523
</details>

<details>
<summary><strong>Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards</strong> - Hieu Trung Nguyen, Bao Nguyen, Wenao Ma, Yuzhi Zhao, Ruifeng She, Viet Anh Nguyen - [[pdf]](https://arxiv.org/pdf/2602.01601)</summary>

**Abstract:** Sampling efficiency is a key bottleneck in reinforcement learning with verifiable rewards. Existing group-based policy optimization methods, such as GRPO, allocate a fixed number of rollouts for all training prompts. This uniform allocation implicitly treats all prompts as equally informative, and could lead to inefficient computational budget usage and impede training progress. We introduce VIP, a Variance-Informed Predictive allocation strategy that allocates a given rollout budget to the prompts in the incumbent batch to minimize the expected gradient variance of the policy update. At each iteration, VIP uses a lightweight Gaussian process model to predict per-prompt success probabilities based on recent rollouts. These probability predictions are translated into variance estimates, which are then fed into a convex optimization problem to determine the optimal rollout allocations under a hard compute budget constraint. Empirical results show that VIP consistently improves sampling efficiency and achieves higher performance than uniform or heuristic allocation strategies in multiple benchmarks.

**arXiv ID:** 2602.01601
</details>

<details>
<summary><strong>RAPTOR-AI for Disaster OODA Loop: Hierarchical Multimodal RAG with Experience-Driven Agentic Decision-Making</strong> - Takato Yasuno - [[pdf]](https://arxiv.org/pdf/2602.00030)</summary>

**Abstract:** Effective humanitarian assistance and disaster relief (HADR) requires rapid situational understanding, reliable decision support, and the ability to generalize across diverse and previously unseen disaster contexts. This work introduces an agentic Retrieval-Augmented Generation (RAG) framework designed to support the three canonical phases of disaster response: initial rescue, mid-term recovery, and long-term reconstruction. To achieve robust multimodal grounding, we construct a hierarchical knowledge base that integrates textual disaster manuals, historical lessons (e.g., the 2011 Tohoku earthquake), and both aerial and ground-level imagery. Our system builds on the open-source multimodal implementation, which processes 46 tsunami-related PDFs (2,378 pages) using BLIP-based image captioning, ColVBERT embeddings, and long-context summarization to generate an efficient, structured multimodal retrieval tree optimized for disaster knowledge preservation. An agentic controller dynamically selects retrieval strategies (e.g., RAPTOR, ColBERT) through entropy-aware scene abstraction, enabling adaptive reasoning across heterogeneous inputs. Additionally, a lightweight LoRA-based post-training method injects experiential knowledge from past disasters, enhancing the models' capacity to support both expert and non-expert responders. Experiments on real disaster datasets demonstrate improved situational grounding, enhanced task decomposition accuracy, and superior usability for emergency operations. Incorporating recent advances in long-context RAG systems, agentic information retrieval, and contemporary emergency response AI, our system achieves substantial gains through adaptive retrieval-augmented generation with self-reasoning and multimodal chain-of-thought capabilities.

**arXiv ID:** 2602.00030
</details>

<details>
<summary><strong>Distributional Reinforcement Learning for Condition-Based Maintenance of Multi-Pump Equipment</strong> - Takato Yasuno - [[pdf]](https://arxiv.org/pdf/2602.00051)</summary>

**Abstract:** Condition-Based Maintenance (CBM) signifies a paradigm shift from reactive to proactive equipment management strategies in modern industrial systems. Conventional time-based maintenance schedules frequently engender superfluous expenditures and unanticipated equipment failures. In contrast, CBM utilizes real-time equipment condition data to enhance maintenance timing and optimize resource allocation. The present paper proposes a novel distributional reinforcement learning approach for multi-equipment CBM using Quantile Regression Deep Q-Networks (QR-DQN) with aging factor integration. The methodology employed in this study encompasses the concurrent administration of multiple pump units through three strategic scenarios. The implementation of safety-first, balanced, and cost-efficient approaches is imperative. Comprehensive experimental validation over 3,000 training episodes demonstrates significant performance improvements across all strategies. The Safety-First strategy demonstrates superior cost efficiency, with a return on investment (ROI) of 3.91, yielding 152\% better performance than alternatives while requiring only 31\% higher investment. The system exhibits 95.66\% operational stability and immediate applicability to industrial environments.

**arXiv ID:** 2602.00051
</details>

<details>
<summary><strong>Sample Complexity Analysis for Constrained Bilevel Reinforcement Learning</strong> - Naman Saxena, Vaneet Aggarwal - [[pdf]](https://arxiv.org/pdf/2602.00282)</summary>

**Abstract:** Several important problem settings within the literature of reinforcement learning (RL), such as meta-learning, hierarchical learning, and RL from human feedback (RL-HF), can be modelled as bilevel RL problems. A lot has been achieved in these domains empirically; however, the theoretical analysis of bilevel RL algorithms hasn't received a lot of attention. In this work, we analyse the sample complexity of a constrained bilevel RL algorithm, building on the progress in the unconstrained setting. We obtain an iteration complexity of $O(\epsilon^{-2})$ and sample complexity of $\tilde{O}(\epsilon^{-4})$ for our proposed algorithm, Constrained Bilevel Subgradient Optimization (CBSO). We use a penalty-based objective function to avoid the issue of primal-dual gap and hyper-gradient in the context of a constrained bilevel problem setting. The penalty-based formulation to handle constraints requires analysis of non-smooth optimization. We are the first ones to analyse the generally parameterized policy gradient-based RL algorithm with a non-smooth objective function using the Moreau envelope.

**arXiv ID:** 2602.00282
</details>

<details>
<summary><strong>Agentic Framework for Epidemiological Modeling</strong> - Rituparna Datta, Zihan Guan, Baltazar Espinoza, Yiqi Su, Priya Pitre, Srini Venkatramanan, Naren Ramakrishnan, Anil Vullikanti - [[pdf]](https://arxiv.org/pdf/2602.00299)</summary>

**Abstract:** Epidemic modeling is essential for public health planning, yet traditional approaches rely on fixed model classes that require manual redesign as pathogens, policies, and scenario assumptions evolve. We introduce EPIAGENT, an agentic framework that automatically synthesizes, calibrates, verifies, and refines epidemiological simulators by modeling disease progression as an iterative program synthesis problem. A central design choice is an explicit epidemiological flow graph intermediate representation that links scenario specifications to model structure and enables strong, modular correctness checks before code is generated. Verified flow graphs are then compiled into mechanistic models supporting interpretable parameter learning under physical and epidemiological constraints. Evaluation on epidemiological scenario case studies demonstrates that EPIAGENT captures complex growth dynamics and produces epidemiologically consistent counterfactual projections across varying vaccination and immune escape assumptions. Our results show that the agentic feedback loop prevents degeneration and significantly accelerates convergence toward valid models by mimicking professional expert workflows.

**arXiv ID:** 2602.00299
</details>

<details>
<summary><strong>DROGO: Default Representation Objective via Graph Optimization in Reinforcement Learning</strong> - Hon Tik Tse, Marlos C. Machado - [[pdf]](https://arxiv.org/pdf/2602.00403)</summary>

**Abstract:** In computational reinforcement learning, the default representation (DR) and its principal eigenvector have been shown to be effective for a wide variety of applications, including reward shaping, count-based exploration, option discovery, and transfer. However, in prior investigations, the eigenvectors of the DR were computed by first approximating the DR matrix, and then performing an eigendecomposition. This procedure is computationally expensive and does not scale to high-dimensional spaces. In this paper, we derive an objective for directly approximating the principal eigenvector of the DR with a neural network. We empirically demonstrate the effectiveness of the objective in a number of environments, and apply the learned eigenvectors for reward shaping.

**arXiv ID:** 2602.00403
</details>

<details>
<summary><strong>Open Materials Generation with Inference-Time Reinforcement Learning</strong> - Philipp Hoellmer, Stefano Martiniani - [[pdf]](https://arxiv.org/pdf/2602.00424)</summary>

**Abstract:** Continuous-time generative models for crystalline materials enable inverse materials design by learning to predict stable crystal structures, but incorporating explicit target properties into the generative process remains challenging. Policy-gradient reinforcement learning (RL) provides a principled mechanism for aligning generative models with downstream objectives but typically requires access to the score, which has prevented its application to flow-based models that learn only velocity fields. We introduce Open Materials Generation with Inference-time Reinforcement Learning (OMatG-IRL), a policy-gradient RL framework that operates directly on the learned velocity fields and eliminates the need for the explicit computation of the score. OMatG-IRL leverages stochastic perturbations of the underlying generation dynamics preserving the baseline performance of the pretrained generative model while enabling exploration and policy-gradient estimation at inference time. Using OMatG-IRL, we present the first application of RL to crystal structure prediction (CSP). Our method enables effective reinforcement of an energy-based objective while preserving diversity through composition conditioning, and it achieves performance competitive with score-based RL approaches. Finally, we show that OMatG-IRL can learn time-dependent velocity-annealing schedules, enabling accurate CSP with order-of-magnitude improvements in sampling efficiency and, correspondingly, reduction in generation time.

**arXiv ID:** 2602.00424
</details>

<details>
<summary><strong>Search Inspired Exploration in Reinforcement Learning</strong> - Georgios Sotirchos, Zlatan Ajanović, Jens Kober - [[pdf]](https://arxiv.org/pdf/2602.00460)</summary>

**Abstract:** Exploration in environments with sparse rewards remains a fundamental challenge in reinforcement learning (RL). Existing approaches such as curriculum learning and Go-Explore often rely on hand-crafted heuristics, while curiosity-driven methods risk converging to suboptimal policies. We propose Search-Inspired Exploration in Reinforcement Learning (SIERL), a novel method that actively guides exploration by setting sub-goals based on the agent's learning progress. At the beginning of each episode, SIERL chooses a sub-goal from the \textit{frontier} (the boundary of the agent's known state space), before the agent continues exploring toward the main task objective. The key contribution of our method is the sub-goal selection mechanism, which provides state-action pairs that are neither overly familiar nor completely novel. Thus, it assures that the frontier is expanded systematically and that the agent is capable of reaching any state within it. Inspired by search, sub-goals are prioritized from the frontier based on estimates of cost-to-come and cost-to-go, effectively steering exploration towards the most informative regions. In experiments on challenging sparse-reward environments, SIERL outperforms dominant baselines in both achieving the main task goal and generalizing to reach arbitrary states in the environment.

**arXiv ID:** 2602.00460
</details>

<details>
<summary><strong>AREAL-DTA: Dynamic Tree Attention for Efficient Reinforcement Learning of Large Language Models</strong> - Jiarui Zhang, Yuchen Yang, Ran Yan, Zhiyu Mei, Liyuan Zhang, Daifeng Li, Wei Fu, Jiaxuan Gao, Shusheng Xu, Yi Wu, Binhang Yuan - [[pdf]](https://arxiv.org/pdf/2602.00482)</summary>

**Abstract:** Reinforcement learning (RL) based post-training for large language models (LLMs) is computationally expensive, as it generates many rollout sequences that could frequently share long token prefixes. Existing RL frameworks usually process these sequences independently, repeatedly recomputing identical prefixes during forward and backward passes during policy model training, leading to substantial inefficiencies in computation and memory usage. Although prefix sharing naturally induces a tree structure over rollouts, prior tree-attention-based solutions rely on fully materialized attention masks and scale poorly in RL settings. In this paper, we introduce AREAL-DTA to efficiently exploit prefix sharing in RL training. AREAL-DTA employs a depth-first-search (DFS)-based execution strategy that dynamically traverses the rollout prefix tree during both forward and backward computation, materializing only a single root-to-leaf path at a time. To further improve scalability, AREAL-DTA incorporates a load-balanced distributed batching mechanism that dynamically constructs and processes prefix trees across multiple GPUs. Across the popular RL post-training workload, AREAL-DTA achieves up to $8.31\times$ in $\tau^2$-bench higher training throughput.

**arXiv ID:** 2602.00482
</details>

<details>
<summary><strong>Minerva: Reinforcement Learning with Verifiable Rewards for Cyber Threat Intelligence LLMs</strong> - Md Tanvirul Alam, Aritran Piplai, Ionut Cardei, Nidhi Rastogi, Peter J Worth Jr - [[pdf]](https://arxiv.org/pdf/2602.00513)</summary>

**Abstract:** Cyber threat intelligence (CTI) analysts routinely convert noisy, unstructured security artifacts into standardized, automation-ready representations. Although large language models (LLMs) show promise for this task, existing approaches remain brittle when producing structured CTI outputs and have largely relied on supervised fine-tuning (SFT). In contrast, CTI standards and community-maintained resources define canonical identifiers and schemas that enable deterministic verification of model outputs. We leverage this structure to study reinforcement learning with verifiable rewards (RLVR) for CTI tasks. We introduce \textit{Minerva}, a unified dataset and training pipeline spanning multiple CTI subtasks, each paired with task-specific verifiers that score structured outputs and identifier predictions. To address reward sparsity during rollout, we propose a lightweight self-training mechanism that generates additional verified trajectories and distills them back into the model. Experiments across LLM backbones show consistent improvements in accuracy and robustness over SFT across multiple benchmarks.

**arXiv ID:** 2602.00513
</details>

<details>
<summary><strong>SAGE: Agentic Framework for Interpretable and Clinically Translatable Computational Pathology Biomarker Discovery</strong> - Sahar Almahfouz Nasser, Juan Francisco Pesantez Borja, Jincheng Liu, Tanvir Hasan, Zenghan Wang, Suman Ghosh, Sandeep Manandhar, Shikhar Shiromani, Twisha Shah, Naoto Tokuyama, Anant Madabhushi - [[pdf]](https://arxiv.org/pdf/2602.00953)</summary>

**Abstract:** Despite significant progress in computational pathology, many AI models remain black-box and difficult to interpret, posing a major barrier to clinical adoption due to limited transparency and explainability. This has motivated continued interest in engineered image-based biomarkers, which offer greater interpretability but are often proposed based on anecdotal evidence or fragmented prior literature rather than systematic biological validation. We introduce SAGE (Structured Agentic system for hypothesis Generation and Evaluation), an agentic AI system designed to identify interpretable, engineered pathology biomarkers by grounding them in biological evidence. SAGE integrates literature-anchored reasoning with multimodal data analysis to correlate image-derived features with molecular biomarkers, such as gene expression, and clinically relevant outcomes. By coordinating specialized agents for biological contextualization and empirical hypothesis validation, SAGE prioritizes transparent, biologically supported biomarkers and advances the clinical translation of computational pathology.

**arXiv ID:** 2602.00953
</details>

<details>
<summary><strong>ESSAM: A Novel Competitive Evolution Strategies Approach to Reinforcement Learning for Memory Efficient LLMs Fine-Tuning</strong> - Zhishen Sun, Sizhe Dang, Guang Dai, Haishan Ye - [[pdf]](https://arxiv.org/pdf/2602.01003)</summary>

**Abstract:** Reinforcement learning (RL) has become a key training step for improving mathematical reasoning in large language models (LLMs), but it often has high GPU memory usage, which makes it hard to use in settings with limited resources. To reduce these issues, we propose Evolution Strategies with Sharpness-Aware Maximization (ESSAM), a full parameter fine-tuning framework that tightly combines the zero-order search in parameter space from Evolution Strategies (ES) with the Sharpness-Aware Maximization (SAM) to improve generalization. We conduct fine-tuning experiments on the mainstream mathematica reasoning task GSM8K. The results show that ESSAM achieves an average accuracy of 78.27\% across all models and its overall performance is comparable to RL methods. It surpasses classic RL algorithm PPO with an accuracy of 77.72\% and is comparable to GRPO with an accuracy of 78.34\%, and even surpassing them on some models. In terms of GPU memory usage, ESSAM reduces the average GPU memory usage by $18\times$ compared to PPO and by $10\times$ compared to GRPO, achieving an extremely low GPU memory usage.

**arXiv ID:** 2602.01003
</details>

<details>
<summary><strong>PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning</strong> - Shunpeng Yang, Ben Liu, Hua Chen - [[pdf]](https://arxiv.org/pdf/2602.01156)</summary>

**Abstract:** Among on-policy reinforcement learning algorithms, Proximal Policy Optimization (PPO) demonstrates is widely favored for its simplicity, numerical stability, and strong empirical performance. Standard PPO relies on surrogate objectives defined via importance ratios, which require evaluating policy likelihood that is typically straightforward when the policy is modeled as a Gaussian distribution. However, extending PPO to more expressive, high-capacity policy models such as continuous normalizing flows (CNFs), also known as flow-matching models, is challenging because likelihood evaluation along the full flow trajectory is computationally expensive and often numerically unstable. To resolve this issue, we propose PolicyFlow, a novel on-policy CNF-based reinforcement learning algorithm that integrates expressive CNF policies with PPO-style objectives without requiring likelihood evaluation along the full flow path. PolicyFlow approximates importance ratios using velocity field variations along a simple interpolation path, reducing computational overhead without compromising training stability. To further prevent mode collapse and further encourage diverse behaviors, we propose the Brownian Regularizer, an implicit policy entropy regularizer inspired by Brownian motion, which is conceptually elegant and computationally lightweight. Experiments on diverse tasks across various environments including MultiGoal, PointMaze, IsaacLab and MuJoCo Playground show that PolicyFlow achieves competitive or superior performance compared to PPO using Gaussian policies and flow-based baselines including FPO and DPPO. Notably, results on MultiGoal highlight PolicyFlow's ability to capture richer multimodal action distributions.

**arXiv ID:** 2602.01156
</details>

<details>
<summary><strong>Sample Efficient Active Algorithms for Offline Reinforcement Learning</strong> - Soumyadeep Roy, Shashwat Kushwaha, Ambedkar Dukkipati - [[pdf]](https://arxiv.org/pdf/2602.01260)</summary>

**Abstract:** Offline reinforcement learning (RL) enables policy learning from static data but often suffers from poor coverage of the state-action space and distributional shift problems. This problem can be addressed by allowing limited online interactions to selectively refine uncertain regions of the learned value function, which is referred to as Active Reinforcement Learning (ActiveRL). While there has been good empirical success, no theoretical analysis is available in the literature. We fill this gap by developing a rigorous sample-complexity analysis of ActiveRL through the lens of Gaussian Process (GP) uncertainty modeling. In this respect, we propose an algorithm and using GP concentration inequalities and information-gain bounds, we derive high-probability guarantees showing that an $\epsilon$-optimal policy can be learned with ${\mathcal{O}}(1/\epsilon^2)$ active transitions, improving upon the $\Omega(1/\epsilon^2(1-\gamma)^4)$ rate of purely offline methods. Our results reveal that ActiveRL achieves near-optimal information efficiency, that is, guided uncertainty reduction leads to accelerated value-function convergence with minimal online data. Our analysis builds on GP concentration inequalities and information-gain bounds, bridging Bayesian nonparametric regression and reinforcement learning theories. We conduct several experiments to validate the algorithm and theoretical findings.

**arXiv ID:** 2602.01260
</details>

<details>
<summary><strong>Mixture-of-World Models: Scaling Multi-Task Reinforcement Learning with Modular Latent Dynamics</strong> - Boxuan Zhang, Weipu Zhang, Zhaohan Feng, Wei Xiao, Jian Sun, Jie Chen, Gang Wang - [[pdf]](https://arxiv.org/pdf/2602.01270)</summary>

**Abstract:** A fundamental challenge in multi-task reinforcement learning (MTRL) is achieving sample efficiency in visual domains where tasks exhibit substantial heterogeneity in both observations and dynamics. Model-based reinforcement learning offers a promising path to improved sample efficiency through world models, but standard monolithic architectures struggle to capture diverse task dynamics, resulting in poor reconstruction and prediction accuracy. We introduce Mixture-of-World Models (MoW), a scalable architecture that combines modular variational autoencoders for task-adaptive visual compression, a hybrid Transformer-based dynamics model with task-conditioned experts and a shared backbone, and a gradient-based task clustering strategy for efficient parameter allocation. On the Atari 100k benchmark, a single MoW agent trained once on 26 Atari games achieves a mean human-normalized score of 110.4%, competitive with the score of 114.2% achieved by STORM, an ensemble of 26 task-specific models, while using 50% fewer parameters. On Meta-World, MoW achieves a 74.5% average success rate within 300 thousand environment steps, establishing a new state of the art. These results demonstrate that MoW provides a scalable and parameter-efficient foundation for generalist world models.

**arXiv ID:** 2602.01270
</details>

<details>
<summary><strong>From Intents to Actions: Agentic AI in Autonomous Networks</strong> - Burak Demirel, Pablo Soldati, Yu Wang - [[pdf]](https://arxiv.org/pdf/2602.01271)</summary>

**Abstract:** Telecommunication networks are increasingly expected to operate autonomously while supporting heterogeneous services with diverse and often conflicting intents -- that is, performance objectives, constraints, and requirements specific to each service. However, transforming high-level intents -- such as ultra-low latency, high throughput, or energy efficiency -- into concrete control actions (i.e., low-level actuator commands) remains beyond the capability of existing heuristic approaches. This work introduces an Agentic AI system for intent-driven autonomous networks, structured around three specialized agents. A supervisory interpreter agent, powered by language models, performs both lexical parsing of intents into executable optimization templates and cognitive refinement based on feedback, constraint feasibility, and evolving network conditions. An optimizer agent converts these templates into tractable optimization problems, analyzes trade-offs, and derives preferences across objectives. Lastly, a preference-driven controller agent, based on multi-objective reinforcement learning, leverages these preferences to operate near the Pareto frontier of network performance that best satisfies the original intent. Collectively, these agents enable networks to autonomously interpret, reason over, adapt to, and act upon diverse intents and network conditions in a scalable manner.

**arXiv ID:** 2602.01271
</details>

<details>
<summary><strong>ZEST: Zero-shot Embodied Skill Transfer for Athletic Robot Control</strong> - Jean Pierre Sleiman, He Li, Alphonsus Adu-Bredu, Robin Deits, Arun Kumar, Kevin Bergamin, Mohak Bhardwaj, Scott Biddlestone, Nicola Burger, Matthew A. Estrada, Francesco Iacobelli, Twan Koolen, Alexander Lambert, Erica Lin, M. Eva Mungai, Zach Nobles, Shane Rozen-Levy, Yuyao Shi, Jiashun Wang, Jakob Welner, Fangzhou Yu, Mike Zhang, Alfred Rizzi, Jessica Hodgins, Sylvain Bertrand, Yeuhi Abe, Scott Kuindersma, Farbod Farshidian - [[pdf]](https://arxiv.org/pdf/2602.00401)</summary>

**Abstract:** Achieving robust, human-like whole-body control on humanoid robots for agile, contact-rich behaviors remains a central challenge, demanding heavy per-skill engineering and a brittle process of tuning controllers. We introduce ZEST (Zero-shot Embodied Skill Transfer), a streamlined motion-imitation framework that trains policies via reinforcement learning from diverse sources -- high-fidelity motion capture, noisy monocular video, and non-physics-constrained animation -- and deploys them to hardware zero-shot. ZEST generalizes across behaviors and platforms while avoiding contact labels, reference or observation windows, state estimators, and extensive reward shaping. Its training pipeline combines adaptive sampling, which focuses training on difficult motion segments, and an automatic curriculum using a model-based assistive wrench, together enabling dynamic, long-horizon maneuvers. We further provide a procedure for selecting joint-level gains from approximate analytical armature values for closed-chain actuators, along with a refined model of actuators. Trained entirely in simulation with moderate domain randomization, ZEST demonstrates remarkable generality. On Boston Dynamics' Atlas humanoid, ZEST learns dynamic, multi-contact skills (e.g., army crawl, breakdancing) from motion capture. It transfers expressive dance and scene-interaction skills, such as box-climbing, directly from videos to Atlas and the Unitree G1. Furthermore, it extends across morphologies to the Spot quadruped, enabling acrobatics, such as a continuous backflip, through animation. Together, these results demonstrate robust zero-shot deployment across heterogeneous data sources and embodiments, establishing ZEST as a scalable interface between biological movements and their robotic counterparts.

**arXiv ID:** 2602.00401
</details>

<details>
<summary><strong>Agentic Reward Modeling: Verifying GUI Agent via Online Proactive Interaction</strong> - Chaoqun Cui, Jing Huang, Shijing Wang, Liming Zheng, Qingchao Kong, Zhixiong Zeng - [[pdf]](https://arxiv.org/pdf/2602.00575)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) is pivotal for the continuous evolution of GUI agents, yet existing evaluation paradigms face significant limitations. Rule-based methods suffer from poor scalability and cannot handle open-ended tasks, while LLM-as-a-Judge approaches rely on passive visual observation, often failing to capture latent system states due to partial state observability. To address these challenges, we advocate for a paradigm shift from passive evaluation to Agentic Interactive Verification. We introduce VAGEN, a framework that employs a verifier agent equipped with interaction tools to autonomously plan verification strategies and proactively probe the environment for evidence of task completion. Leveraging the insight that GUI tasks are typically "easy to verify but hard to solve", VAGEN overcomes the bottlenecks of visual limitations. Experimental results on OSWorld-Verified and AndroidWorld benchmarks demonstrate that VAGEN significantly improves evaluation accuracy compared to LLM-as-a-Judge baselines and further enhances performance through test-time scaling strategies.

**arXiv ID:** 2602.00575
</details>

<details>
<summary><strong>SA-VLA: Spatially-Aware Flow-Matching for Vision-Language-Action Reinforcement Learning</strong> - Xu Pan, Zhenglin Wan, Xingrui Yu, Xianwei Zheng, Youkai Ke, Ming Sun, Rui Wang, Ziwei Wang, Ivor Tsang - [[pdf]](https://arxiv.org/pdf/2602.00743)</summary>

**Abstract:** Vision-Language-Action (VLA) models exhibit strong generalization in robotic manipulation, yet reinforcement learning (RL) fine-tuning often degrades robustness under spatial distribution shifts. For flow-matching VLA policies, this degradation is closely associated with the erosion of spatial inductive bias during RL adaptation, as sparse rewards and spatially agnostic exploration increasingly favor short-horizon visual cues. To address this issue, we propose \textbf{SA-VLA}, a spatially-aware RL adaptation framework that preserves spatial grounding during policy optimization by aligning representation learning, reward design, and exploration with task geometry. SA-VLA fuses implicit spatial representations with visual tokens, provides dense rewards that reflect geometric progress, and employs \textbf{SCAN}, a spatially-conditioned annealed exploration strategy tailored to flow-matching dynamics. Across challenging multi-object and cluttered manipulation benchmarks, SA-VLA enables stable RL fine-tuning and improves zero-shot spatial generalization, yielding more robust and transferable behaviors. Code and project page are available at this https URL.

**arXiv ID:** 2602.00743
</details>

<details>
<summary><strong>Reinforcement Learning for Active Perception in Autonomous Navigation</strong> - Grzegorz Malczyk, Mihir Kulkarni, Kostas Alexis - [[pdf]](https://arxiv.org/pdf/2602.01266)</summary>

**Abstract:** This paper addresses the challenge of active perception within autonomous navigation in complex, unknown environments. Revisiting the foundational principles of active perception, we introduce an end-to-end reinforcement learning framework in which a robot must not only reach a goal while avoiding obstacles, but also actively control its onboard camera to enhance situational awareness. The policy receives observations comprising the robot state, the current depth frame, and a particularly local geometry representation built from a short history of depth readings. To couple collision-free motion planning with information-driven active camera control, we augment the navigation reward with a voxel-based information metric. This enables an aerial robot to learn a robust policy that balances goal-directed motion with exploratory sensing. Extensive evaluation demonstrates that our strategy achieves safer flight compared to using fixed, non-actuated camera baselines while also inducing intrinsic exploratory behaviors.

**arXiv ID:** 2602.01266
</details>

<details>
<summary><strong>AgenticLab: A Real-World Robot Agent Platform that Can See, Think, and Act</strong> - Pengyuan Guo, Zhonghao Mai, Zhengtong Xu, Kaidi Zhang, Heng Zhang, Zichen Miao, Arash Ajoudani, Zachary Kingston, Qiang Qiu, Yu She - [[pdf]](https://arxiv.org/pdf/2602.01662)</summary>

**Abstract:** Recent advances in large vision-language models (VLMs) have demonstrated generalizable open-vocabulary perception and reasoning, yet their real-robot manipulation capability remains unclear for long-horizon, closed-loop execution in unstructured, in-the-wild environments. Prior VLM-based manipulation pipelines are difficult to compare across different research groups' setups, and many evaluations rely on simulation, privileged state, or specially designed setups. We present AgenticLab, a model-agnostic robot agent platform and benchmark for open-world manipulation. AgenticLab provides a closed-loop agent pipeline for perception, task decomposition, online verification, and replanning. Using AgenticLab, we benchmark state-of-the-art VLM-based agents on real-robot tasks in unstructured environments. Our benchmark reveals several failure modes that offline vision-language tests (e.g., VQA and static image understanding) fail to capture, including breakdowns in multi-step grounding consistency, object grounding under occlusion and scene changes, and insufficient spatial reasoning for reliable manipulation. We will release the full hardware and software stack to support reproducible evaluation and accelerate research on general-purpose robot agents.

**arXiv ID:** 2602.01662
</details>

<details>
<summary><strong>GSR: Learning Structured Reasoning for Embodied Manipulation</strong> - Kewei Hu, Michael Zhang, Wei Ying, Tianhao Liu, Guoqiang Hao, Zimeng Li, Wanchan Yu, Jiajian Jing, Fangwen Chen, Hanwen Kang - [[pdf]](https://arxiv.org/pdf/2602.01693)</summary>

**Abstract:** Despite rapid progress, embodied agents still struggle with long-horizon manipulation that requires maintaining spatial consistency, causal dependencies, and goal constraints. A key limitation of existing approaches is that task reasoning is implicitly embedded in high-dimensional latent representations, making it challenging to separate task structure from perceptual variability. We introduce Grounded Scene-graph Reasoning (GSR), a structured reasoning paradigm that explicitly models world-state evolution as transitions over semantically grounded scene graphs. By reasoning step-wise over object states and spatial relations, rather than directly mapping perception to actions, GSR enables explicit reasoning about action preconditions, consequences, and goal satisfaction in a physically grounded space. To support learning such reasoning, we construct Manip-Cognition-1.6M, a large-scale dataset that jointly supervises world understanding, action planning, and goal interpretation. Extensive evaluations across RLBench, LIBERO, GSR-benchmark, and real-world robotic tasks show that GSR significantly improves zero-shot generalization and long-horizon task completion over prompting-based baselines. These results highlight explicit world-state representations as a key inductive bias for scalable embodied reasoning.

**arXiv ID:** 2602.01693
</details>

<details>
<summary><strong>RFS: Reinforcement learning with Residual flow steering for dexterous manipulation</strong> - Entong Su, Tyler Westenbroek, Anusha Nagabandi, Abhishek Gupta - [[pdf]](https://arxiv.org/pdf/2602.01789)</summary>

**Abstract:** Imitation learning has emerged as an effective approach for bootstrapping sequential decision-making in robotics, achieving strong performance even in high-dimensional dexterous manipulation tasks. Recent behavior cloning methods further leverage expressive generative models, such as diffusion models and flow matching, to represent multimodal action distributions. However, policies pretrained in this manner often exhibit limited generalization and require additional fine-tuning to achieve robust performance at deployment time. Such adaptation must preserve the global exploration benefits of pretraining while enabling rapid correction of local execution errors. We propose Residual Flow Steering(RFS), a data-efficient reinforcement learning framework for adapting pretrained generative policies. RFS steers a pretrained flow-matching policy by jointly optimizing a residual action and a latent noise distribution, enabling complementary forms of exploration: local refinement through residual corrections and global exploration through latent-space modulation. This design allows efficient adaptation while retaining the expressive structure of the pretrained policy. We demonstrate the effectiveness of RFS on dexterous manipulation tasks, showing efficient fine-tuning in both simulation and real-world settings when adapting pretrained base policies. Project website:this https URL.

**arXiv ID:** 2602.01789
</details>

<details>
<summary><strong>World-Gymnast: Training Robots with Reinforcement Learning in a World Model</strong> - Ansh Kumar Sharma, Yixiang Sun, Ninghao Lu, Yunzhe Zhang, Jiarao Liu, Sherry Yang - [[pdf]](https://arxiv.org/pdf/2602.02454)</summary>

**Abstract:** Robot learning from interacting with the physical world is fundamentally bottlenecked by the cost of physical interaction. The two alternatives, supervised finetuning (SFT) from expert demonstrations and reinforcement learning (RL) in a software-based simulator, are limited by the amount of expert data available and the sim-to-real gap for manipulation. With the recent emergence of world models learned from real-world video-action data, we ask the question of whether training a policy in a world model can be more effective than supervised learning or software simulation in achieving better real-robot performance. We propose World-Gymnast, which performs RL finetuning of a vision-language-action (VLA) policy by rolling out the policy in an action-conditioned video world model and rewarding the rollouts with a vision-language model (VLM). On the Bridge robot setup, World-Gymnast outperforms SFT by as much as 18x and outperforms software simulator by as much as 2x. More importantly, World-Gymnast demonstrates intriguing capabilities of RL with a world model, including training on diverse language instructions and novel scenes from the world model, test-time training in a novel scene, and online iterative world model and policy improvement. Our results suggest learning a world model and training robot policies in the cloud could be the key to bridging the gap between robots that work in demonstrations and robots that can work in anyone's household.

**arXiv ID:** 2602.02454
</details>

<details>
<summary><strong>A Gait Driven Reinforcement Learning Framework for Humanoid Robots</strong> - Bolin Li, Yuzhi Jiang, Linwei Sun, Xuecong Huang, Lijun Zhu, Han Ding - [[pdf]](https://arxiv.org/pdf/2506.08416)</summary>

**Abstract:** This paper presents a real-time gait driven training framework for humanoid robots. First, we introduce a novel gait planner that incorporates dynamics to design the desired joint trajectory. In the gait design process, the 3D robot model is decoupled into two 2D models, which are then approximated as hybrid inverted pendulums (H-LIP) for trajectory planning. The gait planner operates in parallel in real time within the robot's learning environment. Second, based on this gait planner, we design three effective reward functions within a reinforcement learning framework, forming a reward composition to achieve periodic bipedal gait. This reward composition reduces the robot's learning time and enhances locomotion performance. Finally, a gait design example, along with simulation and experimental comparisons, is presented to demonstrate the effectiveness of the proposed method.

**arXiv ID:** 2506.08416
</details>

<details>
<summary><strong>Reward Evolution with Graph-of-Thoughts: A Bi-Level Language Model Framework for Reinforcement Learning</strong> - Changwei Yao, Xinzi Liu, Chen Li, Marios Savvides - [[pdf]](https://arxiv.org/pdf/2509.16136)</summary>

**Abstract:** Designing effective reward functions remains a major challenge in reinforcement learning (RL), often requiring considerable human expertise and iterative refinement. Recent advances leverage Large Language Models (LLMs) for automated reward design, but these approaches are limited by hallucinations, reliance on human feedback, and challenges with handling complex, multi-step tasks. In this work, we introduce Reward Evolution with Graph-of-Thoughts (RE-GoT), a novel bi-level framework that enhances LLMs with structured graph-based reasoning and integrates Visual Language Models (VLMs) for automated rollout evaluation. RE-GoT first decomposes tasks into text-attributed graphs, enabling comprehensive analysis and reward function generation, and then iteratively refines rewards using visual feedback from VLMs without human intervention. Extensive experiments on 10 RoboGen and 4 ManiSkill2 tasks demonstrate that RE-GoT consistently outperforms existing LLM-based baselines. On RoboGen, our method improves average task success rates by 32.25%, with notable gains on complex multi-step tasks. On ManiSkill2, RE-GoT achieves an average success rate of 93.73% across four diverse manipulation tasks, significantly surpassing prior LLM-based approaches and even exceeding expert-designed rewards. Our results indicate that combining LLMs and VLMs with graph-of-thoughts reasoning provides a scalable and effective solution for autonomous reward evolution in RL.

**arXiv ID:** 2509.16136
</details>

<details>
<summary><strong>Field evaluation and optimization of a lightweight autonomous lidar-based UAV system based on a rigorous experimental setup in boreal forest environments</strong> - Aleksi Karhunen, Teemu Hakala, Väinö Karjalainen, Eija Honkavaara - [[pdf]](https://arxiv.org/pdf/2512.14340)</summary>

**Abstract:** Interest in utilizing autonomous uncrewed aerial vehicles (UAVs) for under-canopy forest remote sensing has increased in recent years, resulting in the publication of numerous autonomous flight algorithms in the scientific literature. To support the selection and development of such algorithms, a reliable comparison of existing approaches based on published studies is essential. However, reliable comparisons are currently challenging due to widely varying experimental setups and incomplete reporting practices. This study proposes a standardized experimental setup for evaluating autonomous under-canopy UAV systems to fill this gap. The proposed setup emphasizes quantitative reporting of forest complexity, visual representation of test environments, execution of multiple repeated flights, and reporting of flight success rates alongside qualitative flight results. In addition, flights at multiple target speeds are encouraged, with reporting of realized flight speed, mission completion time, and point-to-point flight distance. The proposed setup is demonstrated using a lightweight lidar-based quadrotor employing state-of-the-art open-source algorithms, evaluated through extensive experiments in two natural boreal forest environments. Based on a systematic evaluation of the original system, several improvements were introduced. The same experimental protocol was then repeated with the optimized system, resulting in a total of 93 real-world flights. The optimized system achieved success rates of 12/15 and 15/15 at target flight speeds of 1 m/s and 2 m/s, respectively, in a medium-difficulty forest, and 12/15 and 5/15 in a difficult forest. Adoption of the proposed experimental setup would facilitate the literature-based comparison of autonomous under-canopy flight systems and support systematic performance improvement of future UAV-based forest robotics solutions.

**arXiv ID:** 2512.14340
</details>

<details>
<summary><strong>Intelligent Reasoning Cues: A Framework and Case Study of the Roles of AI Information in Complex Decisions</strong> - Venkatesh Sivaraman, Eric P. Mason, Mengfan Ellen Li, Jessica Tong, Andrew J. King, Jeremy M. Kahn, Adam Perer - [[pdf]](https://arxiv.org/pdf/2602.00259)</summary>

**Abstract:** Artificial intelligence (AI)-based decision support systems can be highly accurate yet still fail to support users or improve decisions. Existing theories of AI-assisted decision-making focus on calibrating reliance on AI advice, leaving it unclear how different system designs might influence the reasoning processes underneath. We address this gap by reconsidering AI interfaces as collections of intelligent reasoning cues: discrete pieces of AI information that can individually influence decision-making. We then explore the roles of eight types of reasoning cues in a high-stakes clinical decision (treating patients with sepsis in intensive care). Through contextual inquiries with six teams and a think-aloud study with 25 physicians, we find that reasoning cues have distinct patterns of influence that can directly inform design. Our results also suggest that reasoning cues should prioritize tasks with high variability and discretion, adapt to ensure compatibility with evolving decision needs, and provide complementary, rigorous insights on complex cases.

**arXiv ID:** 2602.00259
</details>

<details>
<summary><strong>HIDAgent: A Toolkit Enabling "Personal Agents" on HID-Compatible Devices</strong> - Jeffrey P. Bigham - [[pdf]](https://arxiv.org/pdf/2602.00492)</summary>

**Abstract:** UI Agents powered by increasingly performant AI promise to eventually use computers the way that people do - by visually interpreting UIs on screen and issuing appropriate actions to control them (e.g., mouse clicks and keyboard entry). While significant progress has been made on interpreting visual UIs computationally, and in sequencing together steps to complete tasks, controlling UIs is still done with system-specific APIs or VNC connections, which limits the platforms and use cases that can be explored. This paper introduces HIDAgent, an open-source hardware/software toolkit enabling UI agents to operate HID-compatible computing systems by emulating the physical keyboard and mouse. HIDAgent is built using three off-the-shelf components costing less than $30 and a Python library supporting flexible integration. We validated the HIDAgent toolkit by building five diverse use case prototypes across mobile and desktop platforms. As a hardware device, HIDAgent supports research into new interaction scenarios where the agents are separated from the devices they control.

**arXiv ID:** 2602.00492
</details>

<details>
<summary><strong>Are Semantic Networks Associated with Idea Originality in Artificial Creativity? A Comparison with Human Agents</strong> - Umberto Domanti, Lorenzo Campidelli, Sergio Agnoli, Antonella De Angeli - [[pdf]](https://arxiv.org/pdf/2602.02048)</summary>

**Abstract:** The application of generative artificial intelligence in Creativity Support Tools (CSTs) presents the challenge of interfacing two black boxes: the user's mind and the machine engine. According to Artificial Cognition, this challenge involves theories, methods, and constructs developed to study human creativity. Consistently, the paper investigated the relationship between semantic networks organisation and idea originality in Large Language Models. Data was collected by administering a set of standardised tests to ChatGPT-4o and 81 psychology students, divided into higher and lower creative individuals. The expected relationship was confirmed in the comparison between ChatGPT-4o and higher creative humans. However, despite having a more rigid network, ChatGPT-4o emerged as more original than lower creative humans. We attributed this difference to human motivational processes and model hyperparameters, advancing a research agenda for the study of artificial creativity. In conclusion, we illustrate the potential of this construct for designing and evaluating CSTs.

**arXiv ID:** 2602.02048
</details>

<details>
<summary><strong>Future of Work with AI Agents: Auditing Automation and Augmentation Potential across the U.S. Workforce</strong> - Yijia Shao, Humishka Zope, Yucheng Jiang, Jiaxin Pei, David Nguyen, Erik Brynjolfsson, Diyi Yang - [[pdf]](https://arxiv.org/pdf/2506.06576)</summary>

**Abstract:** The rapid rise of compound AI systems (a.k.a., AI agents) is reshaping the labor market, raising concerns about job displacement, diminished human agency, and overreliance on automation. Yet, we lack a systematic understanding of the evolving landscape. In this paper, we address this gap by introducing a novel auditing framework to assess which occupational tasks workers want AI agents to automate or augment, and how those desires align with the current technological capabilities. Our framework features an audio-enhanced mini-interview to capture nuanced worker desires and introduces the Human Agency Scale (HAS) as a shared language to quantify the preferred level of human involvement. Using this framework, we construct the WORKBank database, building on the U.S. Department of Labor's O*NET database, to capture preferences from 1,500 domain workers and capability assessments from AI experts across over 844 tasks spanning 104 occupations. Jointly considering the desire and technological capability divides tasks in WORKBank into four zones: Automation "Green Light" Zone, Automation "Red Light" Zone, R&D Opportunity Zone, Low Priority Zone. This highlights critical mismatches and opportunities for AI agent development. Moving beyond a simple automate-or-not dichotomy, our results reveal diverse HAS profiles across occupations, reflecting heterogeneous expectations for human involvement. Moreover, our study offers early signals of how AI agent integration may reshape the core human competencies, shifting from information-focused skills to interpersonal ones. These findings underscore the importance of aligning AI agent development with human desires and preparing workers for evolving workplace dynamics.

**arXiv ID:** 2506.06576
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
