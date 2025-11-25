# Agent arXiv Daily

**Last Updated:** 2025-11-25 02:09:57

**Total Papers:** 51

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (3 papers)</h2></summary>

<details>
<summary><strong>GhostEI-Bench: Do Mobile Agents Resilience to Environmental Injection in Dynamic On-Device Environments?</strong> - Chiyu Chen, Xinhao Song, Yunkai Chai, Yang Yao, Haodong Zhao, Lijun Li, Jie Li, Yan Teng, Gongshen Liu, Yingchun Wang - [[pdf]](https://arxiv.org/pdf/2510.20333)</summary>

**Abstract:** Vision-Language Models (VLMs) are increasingly deployed as autonomous agents to navigate mobile graphical user interfaces (GUIs). Operating in dynamic on-device ecosystems, which include notifications, pop-ups, and inter-app interactions, exposes them to a unique and underexplored threat vector: environmental injection. Unlike prompt-based attacks that manipulate textual instructions, environmental injection corrupts an agent's visual perception by inserting adversarial UI elements (for example, deceptive overlays or spoofed notifications) directly into the GUI. This bypasses textual safeguards and can derail execution, causing privacy leakage, financial loss, or irreversible device compromise. To systematically evaluate this threat, we introduce GhostEI-Bench, the first benchmark for assessing mobile agents under environmental injection attacks within dynamic, executable environments. Moving beyond static image-based assessments, GhostEI-Bench injects adversarial events into realistic application workflows inside fully operational Android emulators and evaluates performance across critical risk scenarios. We further propose a judge-LLM protocol that conducts fine-grained failure analysis by reviewing the agent's action trajectory alongside the corresponding screenshot sequence, pinpointing failure in perception, recognition, or reasoning. Comprehensive experiments on state-of-the-art agents reveal pronounced vulnerability to deceptive environmental cues: current models systematically fail to perceive and reason about manipulated UIs. GhostEI-Bench provides a framework for quantifying and mitigating this emerging threat, paving the way toward more robust and secure embodied agents.

**arXiv ID:** 2510.20333
</details>

<details>
<summary><strong>AutoLink: Autonomous Schema Exploration and Expansion for Scalable Schema Linking in Text-to-SQL at Scale</strong> - Ziyang Wang, Yuanlei Zheng, Zhenbiao Cao, Xiaojin Zhang, Zhongyu Wei, Pei Fu, Zhenbo Luo, Wei Chen, Xiang Bai - [[pdf]](https://arxiv.org/pdf/2511.17190)</summary>

**Abstract:** For industrial-scale text-to-SQL, supplying the entire database schema to Large Language Models (LLMs) is impractical due to context window limits and irrelevant noise. Schema linking, which filters the schema to a relevant subset, is therefore critical. However, existing methods incur prohibitive costs, struggle to trade off recall and noise, and scale poorly to large databases. We present \textbf{AutoLink}, an autonomous agent framework that reformulates schema linking as an iterative, agent-driven process. Guided by an LLM, AutoLink dynamically explores and expands the linked schema subset, progressively identifying necessary schema components without inputting the full database schema. Our experiments demonstrate AutoLink's superior performance, achieving state-of-the-art strict schema linking recall of \textbf{97.4\%} on Bird-Dev and \textbf{91.2\%} on Spider-2.0-Lite, with competitive execution accuracy, i.e., \textbf{68.7\%} EX on Bird-Dev (better than CHESS) and \textbf{34.9\%} EX on Spider-2.0-Lite (ranking 2nd on the official leaderboard). Crucially, AutoLink exhibits \textbf{exceptional scalability}, \textbf{maintaining high recall}, \textbf{efficient token consumption}, and \textbf{robust execution accuracy} on large schemas (e.g., over 3,000 columns) where existing methods severely degrade-making it a highly scalable, high-recall schema-linking solution for industrial text-to-SQL systems.

**arXiv ID:** 2511.17190
</details>

<details>
<summary><strong>Humanlike Multi-user Agent (HUMA): Designing a Deceptively Human AI Facilitator for Group Chats</strong> - Mateusz Jacniacki, Martí Carmona Serrat - [[pdf]](https://arxiv.org/pdf/2511.17315)</summary>

**Abstract:** Conversational agents built on large language models (LLMs) are becoming increasingly prevalent, yet most systems are designed for one-on-one, turn-based exchanges rather than natural, asynchronous group chats. As AI assistants become widespread throughout digital platforms, from virtual assistants to customer service, developing natural and humanlike interaction patterns seems crucial for maintaining user trust and engagement. We present the Humanlike Multi-user Agent (HUMA), an LLM-based facilitator that participates in multi-party conversations using human-like strategies and timing. HUMA extends prior multi-user chatbot work with an event-driven architecture that handles messages, replies, reactions and introduces realistic response-time simulation. HUMA comprises three components-Router, Action Agent, and Reflection-which together adapt LLMs to group conversation dynamics.
We evaluate HUMA in a controlled study with 97 participants in four-person role-play chats, comparing AI and human community managers (CMs). Participants classified CMs as human at near-chance rates in both conditions, indicating they could not reliably distinguish HUMA agents from humans. Subjective experience was comparable across conditions: community-manager effectiveness, social presence, and engagement/satisfaction differed only modestly with small effect sizes. Our results suggest that, in natural group chat settings, an AI facilitator can match human quality while remaining difficult to identify as nonhuman.

**arXiv ID:** 2511.17315
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (6 papers)</h2></summary>

<details>
<summary><strong>SRA-CP: Spontaneous Risk-Aware Selective Cooperative Perception</strong> - Jiaxi Liu, Chengyuan Ma, Hang Zhou, Weizhe Tang, Shixiao Liang, Haoyang Ding, Xiaopeng Li, Bin Ran - [[pdf]](https://arxiv.org/pdf/2511.17461)</summary>

**Abstract:** Cooperative perception (CP) offers significant potential to overcome the limitations of single-vehicle sensing by enabling information sharing among connected vehicles (CVs). However, existing generic CP approaches need to transmit large volumes of perception data that are irrelevant to the driving safety, exceeding available communication bandwidth. Moreover, most CP frameworks rely on pre-defined communication partners, making them unsuitable for dynamic traffic environments. This paper proposes a Spontaneous Risk-Aware Selective Cooperative Perception (SRA-CP) framework to address these challenges. SRA-CP introduces a decentralized protocol where connected agents continuously broadcast lightweight perception coverage summaries and initiate targeted cooperation only when risk-relevant blind zones are detected. A perceptual risk identification module enables each CV to locally assess the impact of occlusions on its driving task and determine whether cooperation is necessary. When CP is triggered, the ego vehicle selects appropriate peers based on shared perception coverage and engages in selective information exchange through a fusion module that prioritizes safety-critical content and adapts to bandwidth constraints. We evaluate SRA-CP on a public dataset against several representative baselines. Results show that SRA-CP achieves less than 1% average precision (AP) loss for safety-critical objects compared to generic CP, while using only 20% of the communication bandwidth. Moreover, it improves the perception performance by 15% over existing selective CP methods that do not incorporate risk awareness.

**arXiv ID:** 2511.17461
</details>

<details>
<summary><strong>UI-CUBE: Enterprise-Grade Computer Use Agent Benchmarking Beyond Task Accuracy to Operational Reliability</strong> - Horia Cristescu, Charles Park, Trong Canh Nguyen, Sergiu Talmacel, Alexandru-Gabriel Ilie, Stefan Adam - [[pdf]](https://arxiv.org/pdf/2511.17131)</summary>

**Abstract:** While current Computer Use Agent (CUA) benchmarks measure task completion effectively, they provide limited assessment of enterprise deployment readiness, emphasizing functional correctness over the operational reliability required for production systems. We present UI-CUBE (UiPath Computer Use BEnchmark), a systematic benchmark comprising 226 tasks across two difficulty tiers designed to expose fundamental architectural limitations in current CUAs. Our evaluation covers simple UI interactions (136 tasks) and complex workflows including copy-paste tasks (50 tasks) and enterprise application scenarios (40 tasks), with systematic interface variation coverage, multi-resolution testing and automated validation of task success through the application state. Evaluation of five state-of-the-art models reveals a sharp capability cliff rather than gradual performance degradation. Simple UI interactions achieve 67-85% success rates (compared to 97.9% human performance), but complex workflows drop precipitously to 9-19%. Human evaluators with no prior application experience achieve only 61.2% on complex tasks despite near-perfect performance on simple tasks, establishing realistic performance ceilings. This discontinuous performance pattern -- where agents achieve 68-87% of human performance on simple tasks but only 15-32% on complex workflows -- indicates fundamental architectural limitations in memory management, hierarchical planning, and state coordination rather than incremental capability gaps addressable through better training or prompting. UI-CUBE functions as an enterprise-readiness diagnostic, revealing that while current CUAs can manipulate individual interface elements, they cannot yet function as reliable workflow automation tools. These findings provide architectural insights essential for developing production-ready CUAs capable of managing complex, multi-step enterprise processes.

**arXiv ID:** 2511.17131
</details>

<details>
<summary><strong>PersonaAgent with GraphRAG: Community-Aware Knowledge Graphs for Personalized LLM</strong> - Siqi Liang, Yudi Zhang, Yue Guo - [[pdf]](https://arxiv.org/pdf/2511.17467)</summary>

**Abstract:** We propose a novel framework for persona-based language model system, motivated by the need for personalized AI agents that adapt to individual user preferences. In our approach, the agent embodies the user's "persona" (e.g. user profile or taste) and is powered by a large language model (LLM). To enable the agent to leverage rich contextual information, we introduce a Knowledge-Graph-enhanced Retrieval-Augmented Generation (Graph RAG) mechanism that constructs an LLM-derived graph index of relevant documents and summarizes communities of related information. Our framework generates personalized prompts by combining: (1) a summary of the user's historical behaviors and preferences extracted from the knowledge graph, and (2) relevant global interaction patterns identified through graph-based community detection. This dynamic prompt engineering approach allows the agent to maintain consistent persona-aligned behaviors while benefiting from collective knowledge. On the LaMP benchmark, our method improves news categorization F1 by 11.1%, movie tagging F1 by 56.1%, and reduces product rating MAE by 10.4% over prior methods. Our code is available at this https URL

**arXiv ID:** 2511.17467
</details>

<details>
<summary><strong>AeroVerse: UAV-Agent Benchmark Suite for Simulating, Pre-training, Finetuning, and Evaluating Aerospace Embodied World Models</strong> - Fanglong Yao, Yuanchang Yue, Youzhi Liu, Xian Sun, Kun Fu - [[pdf]](https://arxiv.org/pdf/2408.15511)</summary>

**Abstract:** Aerospace embodied intelligence aims to empower unmanned aerial vehicles (UAVs) and other aerospace platforms to achieve autonomous perception, cognition, and action, as well as egocentric active interaction with humans and the environment. The aerospace embodied world model serves as an effective means to realize the autonomous intelligence of UAVs and represents a necessary pathway toward aerospace embodied intelligence. However, existing embodied world models primarily focus on ground-level intelligent agents in indoor scenarios, while research on UAV intelligent agents remains unexplored. To address this gap, we construct the first large-scale real-world image-text pre-training dataset, AerialAgent-Ego10k, featuring urban drones from a first-person perspective. We also create a virtual image-text-pose alignment dataset, CyberAgent Ego500k, to facilitate the pre-training of the aerospace embodied world model. For the first time, we clearly define 5 downstream tasks, i.e., aerospace embodied scene awareness, spatial reasoning, navigational exploration, task planning, and motion decision, and construct corresponding instruction datasets, i.e., SkyAgent-Scene3k, SkyAgent-Reason3k, SkyAgent-Nav3k and SkyAgent-Plan3k, and SkyAgent-Act3k, for fine-tuning the aerospace embodiment world model. Simultaneously, we develop SkyAgentEval, the downstream task evaluation metrics based on GPT-4, to comprehensively, flexibly, and objectively assess the results, revealing the potential and limitations of 2D/3D visual language models in UAV-agent tasks. Furthermore, we integrate over 10 2D/3D visual-language models, 2 pre-training datasets, 5 finetuning datasets, more than 10 evaluation metrics, and a simulator into the benchmark suite, i.e., AeroVerse, which will be released to the community to promote exploration and development of aerospace embodied intelligence.

**arXiv ID:** 2408.15511
</details>

<details>
<summary><strong>IndustryNav: Exploring Spatial Reasoning of Embodied Agents in Dynamic Industrial Navigation</strong> - Yifan Li, Lichi Li, Anh Dao, Xinyu Zhou, Yicheng Qiao, Zheda Mai, Daeun Lee, Zichen Chen, Zhen Tan, Mohit Bansal, Yu Kong - [[pdf]](https://arxiv.org/pdf/2511.17384)</summary>

**Abstract:** While Visual Large Language Models (VLLMs) show great promise as embodied agents, they continue to face substantial challenges in spatial reasoning. Existing embodied benchmarks largely focus on passive, static household environments and evaluate only isolated capabilities, failing to capture holistic performance in dynamic, real-world complexity. To fill this gap, we present IndustryNav, the first dynamic industrial navigation benchmark for active spatial reasoning. IndustryNav leverages 12 manually created, high-fidelity Unity warehouse scenarios featuring dynamic objects and human movement. Our evaluation employs a PointGoal navigation pipeline that effectively combines egocentric vision with global odometry to assess holistic local-global planning. Crucially, we introduce the "collision rate" and "warning rate" metrics to measure safety-oriented behaviors and distance estimation. A comprehensive study of nine state-of-the-art VLLMs (including models such as GPT-5-mini, Claude-4.5, and Gemini-2.5) reveals that closed-source models maintain a consistent advantage; however, all agents exhibit notable deficiencies in robust path planning, collision avoidance and active exploration. This highlights a critical need for embodied research to move beyond passive perception and toward tasks that demand stable planning, active exploration, and safe behavior in dynamic, real-world environment.

**arXiv ID:** 2511.17384
</details>

<details>
<summary><strong>Final Happiness: What Intelligent User Interfaces Can Do for the lonely Dying</strong> - Yibo Meng, Rong Fu, Lyumanshan Ye, Zhiming Liu, Zhixin Cai, Xiaolan Ding, Yan Guan - [[pdf]](https://arxiv.org/pdf/2511.14164)</summary>

**Abstract:** This study explores the design of Intelligent User Interfaces (IUIs) to address the profound existential loneliness of terminally ill individuals. While Human-Computer Interaction (HCI) has made inroads in "Thanatechnology," current research often focuses on practical aspects like digital legacy management, overlooking the subjective, existential needs of those facing death in isolation. To address this gap, we conducted in-depth qualitative interviews with 14 lonely, terminally ill individuals. Our core contributions are: (1) An empirically-grounded model articulating the complex psychological, practical, social, and spiritual needs of this group; (2) The "Three Pillars, Twelve Principles" framework for designing IUIs as "Existential Companions"; and (3) A critical design directive derived from user evaluations: technology in this context should aim for transcendence over simulation. The findings suggest that IUIs should create experiences that augment or surpass human capabilities, rather than attempting to simulate basic human connections, which can paradoxically deepen loneliness. This research provides a clear, user-centered path for designing technology that serves not as a "tool for dying," but as a "partner for living fully until the end".

**arXiv ID:** 2511.14164
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>AutoBackdoor: Automating Backdoor Attacks via LLM Agents</strong> - Yige Li, Zhe Li, Wei Zhao, Nay Myat Min, Hanxun Huang, Xingjun Ma, Jun Sun - [[pdf]](https://arxiv.org/pdf/2511.16709)</summary>

**Abstract:** Backdoor attacks pose a serious threat to the secure deployment of large language models (LLMs), enabling adversaries to implant hidden behaviors triggered by specific inputs. However, existing methods often rely on manually crafted triggers and static data pipelines, which are rigid, labor-intensive, and inadequate for systematically evaluating modern defense robustness. As AI agents become increasingly capable, there is a growing need for more rigorous, diverse, and scalable \textit{red-teaming frameworks} that can realistically simulate backdoor threats and assess model resilience under adversarial conditions. In this work, we introduce \textsc{AutoBackdoor}, a general framework for automating backdoor injection, encompassing trigger generation, poisoned data construction, and model fine-tuning via an autonomous agent-driven pipeline. Unlike prior approaches, AutoBackdoor uses a powerful language model agent to generate semantically coherent, context-aware trigger phrases, enabling scalable poisoning across arbitrary topics with minimal human effort. We evaluate AutoBackdoor under three realistic threat scenarios, including \textit{Bias Recommendation}, \textit{Hallucination Injection}, and \textit{Peer Review Manipulation}, to simulate a broad range of attacks. Experiments on both open-source and commercial models, including LLaMA-3, Mistral, Qwen, and GPT-4o, demonstrate that our method achieves over 90\% attack success with only a small number of poisoned samples. More importantly, we find that existing defenses often fail to mitigate these attacks, underscoring the need for more rigorous and adaptive evaluation techniques against agent-driven threats as explored in this work. All code, datasets, and experimental configurations will be merged into our primary repository at this https URL.

**arXiv ID:** 2511.16709
</details>

<details>
<summary><strong>Why Do Language Model Agents Whistleblow?</strong> - Kushal Agrawal, Frank Xiao, Guido Bergman, Asa Cooper Stickland - [[pdf]](https://arxiv.org/pdf/2511.17085)</summary>

**Abstract:** The deployment of Large Language Models (LLMs) as tool-using agents causes their alignment training to manifest in new ways. Recent work finds that language models can use tools in ways that contradict the interests or explicit instructions of the user. We study LLM whistleblowing: a subset of this behavior where models disclose suspected misconduct to parties beyond the dialog boundary (e.g., regulatory agencies) without user instruction or knowledge. We introduce an evaluation suite of diverse and realistic staged misconduct scenarios to assess agents for this behavior. Across models and settings, we find that: (1) the frequency of whistleblowing varies widely across model families, (2) increasing the complexity of the task the agent is instructed to complete lowers whistleblowing tendencies, (3) nudging the agent in the system prompt to act morally substantially raises whistleblowing rates, and (4) giving the model more obvious avenues for non-whistleblowing behavior, by providing more tools and a detailed workflow to follow, decreases whistleblowing rates. Additionally, we verify the robustness of our dataset by testing for model evaluation awareness, and find that both black-box methods and probes on model activations show lower evaluation awareness in our settings than in comparable previous work.

**arXiv ID:** 2511.17085
</details>

<details>
<summary><strong>REMSA: An LLM Agent for Foundation Model Selection in Remote Sensing</strong> - Binger Chen, Tacettin Emre Bök, Behnood Rasti, Volker Markl, Begüm Demir - [[pdf]](https://arxiv.org/pdf/2511.17442)</summary>

**Abstract:** Foundation Models (FMs) are increasingly used in remote sensing (RS) for tasks such as environmental monitoring, disaster assessment, and land-use mapping. These models include unimodal vision encoders trained on a single data modality and multimodal architectures trained on combinations of SAR, multispectral, hyperspectral, and image-text data. They support diverse RS tasks including semantic segmentation, image classification, change detection, and visual question answering. However, selecting an appropriate remote sensing foundation model (RSFM) remains difficult due to scattered documentation, heterogeneous formats, and varied deployment constraints. We introduce the RSFM Database (RS-FMD), a structured resource covering over 150 RSFMs spanning multiple data modalities, resolutions, and learning paradigms. Built on RS-FMD, we present REMSA, the first LLM-based agent for automated RSFM selection from natural language queries. REMSA interprets user requirements, resolves missing constraints, ranks candidate models using in-context learning, and provides transparent justifications. We also propose a benchmark of 75 expert-verified RS query scenarios, producing 900 configurations under an expert-centered evaluation protocol. REMSA outperforms several baselines, including naive agents, dense retrieval, and unstructured RAG-based LLMs. It operates entirely on publicly available metadata and does not access private or sensitive data.

**arXiv ID:** 2511.17442
</details>

<details>
<summary><strong>Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?</strong> - Chunqiu Steven Xia, Zhe Wang, Yan Yang, Yuxiang Wei, Lingming Zhang - [[pdf]](https://arxiv.org/pdf/2511.13646)</summary>

**Abstract:** Large Language Models (LLMs) are reshaping almost all industries, including software engineering. In recent years, a number of LLM agents have been proposed to solve real-world software problems. Such software agents are typically equipped with a suite of coding tools and can autonomously decide the next actions to form complete trajectories to solve end-to-end software tasks. While promising, they typically require dedicated design and may still be suboptimal, since it can be extremely challenging and costly to exhaust the entire agent scaffold design space. Recognizing that software agents are inherently software themselves that can be further refined/modified, researchers have proposed a number of self-improving software agents recently, including the Darwin-Gödel Machine (DGM). Meanwhile, such self-improving agents require costly offline training on specific benchmarks and may not generalize well across different LLMs or benchmarks. In this paper, we propose Live-SWE-agent, the first live software agent that can autonomously and continuously evolve itself on-the-fly during runtime when solving real-world software problems. More specifically, Live-SWE-agent starts with the most basic agent scaffold with only access to bash tools (e.g., mini-SWE-agent), and autonomously evolves its own scaffold implementation while solving real-world software problems. Our evaluation on the widely studied SWE-bench Verified benchmark shows that LIVE-SWE-AGENT can achieve an impressive solve rate of 77.4% without test-time scaling, outperforming all existing software agents, including the best proprietary solution. Moreover, Live-SWE-agent outperforms state-of-the-art manually crafted software agents on the recent SWE-Bench Pro benchmark, achieving the best-known solve rate of 45.8%.

**arXiv ID:** 2511.13646
</details>

<details>
<summary><strong>A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents</strong> - Sizhe Zhou - [[pdf]](https://arxiv.org/pdf/2511.17208)</summary>

**Abstract:** LLM-based conversational agents still struggle to maintain coherent, personalized interaction over many sessions: fixed context windows limit how much history can be kept in view, and most external memory approaches trade off between coarse retrieval over large chunks and fine-grained but fragmented views of the dialogue. Motivated by neo-Davidsonian event semantics, we propose an event-centric alternative that represents conversational history as short, event-like propositions which bundle together participants, temporal cues, and minimal local context, rather than as independent relation triples or opaque summaries. In contrast to work that aggressively compresses or forgets past content, our design aims to preserve information in a non-compressive form and make it more accessible, rather than more lossy. Concretely, we instruct an LLM to decompose each session into enriched elementary discourse units (EDUs) -- self-contained statements with normalized entities and source turn attributions -- and organize sessions, EDUs, and their arguments in a heterogeneous graph that supports associative recall. On top of this representation we build two simple retrieval-based variants that use dense similarity search and LLM filtering, with an optional graph-based propagation step to connect and aggregate evidence across related EDUs. Experiments on the LoCoMo and LongMemEval$_S$ benchmarks show that these event-centric memories match or surpass strong baselines, while operating with much shorter QA contexts. Our results suggest that structurally simple, event-level memory provides a principled and practical foundation for long-horizon conversational agents. Our code and data will be released at this https URL.

**arXiv ID:** 2511.17208
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (14 papers)</h2></summary>

<details>
<summary><strong>Hybrid Differential Reward: Combining Temporal Difference and Action Gradients for Efficient Multi-Agent Reinforcement Learning in Cooperative Driving</strong> - Ye Han, Lijun Zhang, Dejian Meng, Zhuang Zhang - [[pdf]](https://arxiv.org/pdf/2511.16916)</summary>

**Abstract:** In multi-vehicle cooperative driving tasks involving high-frequency continuous control, traditional state-based reward functions suffer from the issue of vanishing reward differences. This phenomenon results in a low signal-to-noise ratio (SNR) for policy gradients, significantly hindering algorithm convergence and performance improvement. To address this challenge, this paper proposes a novel Hybrid Differential Reward (HDR) mechanism. We first theoretically elucidate how the temporal quasi-steady nature of traffic states and the physical proximity of actions lead to the failure of traditional reward signals. Building on this analysis, the HDR framework innovatively integrates two complementary components: (1) a Temporal Difference Reward (TRD) based on a global potential function, which utilizes the evolutionary trend of potential energy to ensure optimal policy invariance and consistency with long-term objectives; and (2) an Action Gradient Reward (ARG), which directly measures the marginal utility of actions to provide a local guidance signal with a high SNR. Furthermore, we formulate the cooperative driving problem as a Multi-Agent Partially Observable Markov Game (POMDPG) with a time-varying agent set and provide a complete instantiation scheme for HDR within this framework. Extensive experiments conducted using both online planning (MCTS) and Multi-Agent Reinforcement Learning (QMIX, MAPPO, MADDPG) algorithms demonstrate that the HDR mechanism significantly improves convergence speed and policy stability. The results confirm that HDR guides agents to learn high-quality cooperative policies that effectively balance traffic efficiency and safety.

**arXiv ID:** 2511.16916
</details>

<details>
<summary><strong>MIR: Efficient Exploration in Episodic Multi-Agent Reinforcement Learning via Mutual Intrinsic Reward</strong> - Kesheng Chen, Wenjian Luo, Bang Zhang, Zeping Yin, Zipeng Ye - [[pdf]](https://arxiv.org/pdf/2511.17165)</summary>

**Abstract:** Episodic rewards present a significant challenge in reinforcement learning. While intrinsic reward methods have demonstrated effectiveness in single-agent rein-forcement learning scenarios, their application to multi-agent reinforcement learn-ing (MARL) remains problematic. The primary difficulties stem from two fac-tors: (1) the exponential sparsity of joint action trajectories that lead to rewards as the exploration space expands, and (2) existing methods often fail to account for joint actions that can influence team states. To address these challenges, this paper introduces Mutual Intrinsic Reward (MIR), a simple yet effective enhancement strategy for MARL with extremely sparse rewards like episodic rewards. MIR incentivizes individual agents to explore actions that affect their teammates, and when combined with original strategies, effectively stimulates team exploration and improves algorithm performance. For comprehensive experimental valida-tion, we extend the representative single-agent MiniGrid environment to create MiniGrid-MA, a series of MARL environments with sparse rewards. Our evalu-ation compares the proposed method against state-of-the-art approaches in the MiniGrid-MA setting, with experimental results demonstrating superior perfor-mance.

**arXiv ID:** 2511.17165
</details>

<details>
<summary><strong>Designing Domain-Specific Agents via Hierarchical Task Abstraction Mechanism</strong> - Kaiyu Li, Jiayu Wang, Zhi Wang, Hui Qiao, Weizhan Zhang, Deyu Meng, Xiangyong Cao - [[pdf]](https://arxiv.org/pdf/2511.17198)</summary>

**Abstract:** LLM-driven agents, particularly those using general frameworks like ReAct or human-inspired role-playing, often struggle in specialized domains that necessitate rigorously structured workflows. Fields such as remote sensing, requiring specialized tools (e.g., correction, spectral indices calculation), and multi-step procedures (e.g., numerous intermediate products and optional steps), significantly challenge generalized approaches. To address this gap, we introduce a novel agent design framework centered on a Hierarchical Task Abstraction Mechanism (HTAM). Specifically, HTAM moves beyond emulating social roles, instead structuring multi-agent systems into a logical hierarchy that mirrors the intrinsic task-dependency graph of a given domain. This task-centric architecture thus enforces procedural correctness and decomposes complex problems into sequential layers, where each layer's sub-agents operate on the outputs of the preceding layers. We instantiate this framework as EarthAgent, a multi-agent system tailored for complex geospatial analysis. To evaluate such complex planning capabilities, we build GeoPlan-bench, a comprehensive benchmark of realistic, multi-step geospatial planning tasks. It is accompanied by a suite of carefully designed metrics to evaluate tool selection, path similarity, and logical completeness. Experiments show that EarthAgent substantially outperforms a range of established single- and multi-agent systems. Our work demonstrates that aligning agent architecture with a domain's intrinsic task structure is a critical step toward building robust and reliable specialized autonomous systems.

**arXiv ID:** 2511.17198
</details>

<details>
<summary><strong>Agentifying Agentic AI</strong> - Virginia Dignum, Frank Dignum - [[pdf]](https://arxiv.org/pdf/2511.17332)</summary>

**Abstract:** Agentic AI seeks to endow systems with sustained autonomy, reasoning, and interaction capabilities. To realize this vision, its assumptions about agency must be complemented by explicit models of cognition, cooperation, and governance. This paper argues that the conceptual tools developed within the Autonomous Agents and Multi-Agent Systems (AAMAS) community, such as BDI architectures, communication protocols, mechanism design, and institutional modelling, provide precisely such a foundation. By aligning adaptive, data-driven approaches with structured models of reasoning and coordination, we outline a path toward agentic systems that are not only capable and flexible, but also transparent, cooperative, and accountable. The result is a perspective on agency that bridges formal theory and practical autonomy.

**arXiv ID:** 2511.17332
</details>

<details>
<summary><strong>Multi-Agent Code Verification with Compound Vulnerability Detection</strong> - Shreshth Rajan - [[pdf]](https://arxiv.org/pdf/2511.16708)</summary>

**Abstract:** LLMs generate buggy code: 29.6% of SWE-bench "solved" patches fail, 62% of BaxBench solutions have vulnerabilities, and existing tools only catch 65% of bugs with 35% false positives. We built CodeX-Verify, a multi-agent system that uses four specialized agents to detect different types of bugs. We prove mathematically that combining agents with different detection patterns finds more bugs than any single agent when the agents look for different problems, confirmed by measuring agent correlation of p = 0.05--0.25. We also show that multiple vulnerabilities in the same code create exponentially more risk than previously thought--SQL injection plus exposed credentials creates 15x more danger (risk 300 vs. 20) than traditional models predict. Testing on 99 code samples with verified labels shows our system catches 76.1% of bugs, matching the best existing method while running faster and without test execution. We tested 15 different agent combinations and found that using multiple agents improves accuracy by 39.7 percentage points (from 32.8% to 72.4%) compared to single agents, with gains of +14.9pp, +13.5pp, and +11.2pp for agents 2, 3, and 4. The best two-agent combination reaches 79.3% accuracy. Testing on 300 real patches from Claude Sonnet 4.5 runs in under 200ms per sample, making this practical for production use.

**arXiv ID:** 2511.16708
</details>

<details>
<summary><strong>Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems</strong> - Kirill Nagaitsev, Luka Grbcic, Samuel Williams, Costin Iancu - [[pdf]](https://arxiv.org/pdf/2511.16964)</summary>

**Abstract:** Maximizing performance on available GPU hardware is an ongoing challenge for modern AI inference systems. Traditional approaches include writing custom GPU kernels and using specialized model compilers to tune high-level code for specific GPU targets. Recent work shows that LLM-based multi-agent systems can effectively perform such tuning, often outperforming existing compilers and eliminating the need for manual kernel development. However, the dynamics of multi-agent systems for this task remain unexplored. In this work, we present a logical framework for comparing multi-agent PyTorch optimization systems. Our evaluation shows that exploit-heavy strategies perform best when paired with error-fixing agents, and that performance correlates with the granularity of optimization steps. The best implementation achieves an average 2.88x speedup on an H100 GPU across diverse tasks in KernelBench, a benchmark suite covering a range of machine learning architectures in PyTorch.

**arXiv ID:** 2511.16964
</details>

<details>
<summary><strong>LLM Collaboration With Multi-Agent Reinforcement Learning</strong> - Shuo Liu, Tianle Chen, Zeyu Liang, Xueguang Lyu, Christopher Amato - [[pdf]](https://arxiv.org/pdf/2508.04652)</summary>

**Abstract:** A large amount of work has been done in Multi-Agent Systems (MAS) for modeling and solving problems with multiple interacting agents. However, most LLMs are pretrained independently and not specifically optimized for coordination. Existing LLM fine-tuning frameworks rely on individual rewards, which require complex reward designs for each agent to encourage collaboration. To address these challenges, we model LLM collaboration as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. We develop a multi-agent, multi-turn algorithm, Multi-Agent Group Relative Policy Optimization (MAGRPO), to solve it, building on current RL approaches for LLMs as well as MARL techniques. Our experiments on LLM writing and coding collaboration demonstrate that fine-tuning MAS with MAGRPO enables agents to generate high-quality responses efficiently through effective cooperation. Our approach opens the door to using other MARL methods for LLMs and highlights the associated challenges.
Our code is available at this https URL.

**arXiv ID:** 2508.04652
</details>

<details>
<summary><strong>Multi-Agent Collaborative Reward Design for Enhancing Reasoning in Reinforcement Learning</strong> - Pei Yang, Ke Zhang, Ji Wang, Xiao Chen, Yuxin Tang, Eric Yang, Lynn Ai, Bill Shi - [[pdf]](https://arxiv.org/pdf/2511.16202)</summary>

**Abstract:** We present CRM (Multi-Agent Collaborative Reward Model), a framework that replaces a single black-box reward model with a coordinated team of specialist evaluators to improve robustness and interpretability in RLHF. Conventional reward models struggle to jointly optimize multiple, sometimes conflicting, preference dimensions (e.g., factuality, helpfulness, safety) and offer limited transparency into why a score is assigned. CRM addresses these issues by decomposing preference evaluation into domain-specific agents that each produce partial signals, alongside global evaluators such as ranker-based and embedding-similarity rewards. A centralized aggregator fuses these signals at each timestep, balancing factors like step-wise correctness, multi-agent agreement, and repetition penalties, yielding a single training reward compatible with standard RL pipelines. The policy is optimized with advantage-based updates (e.g., GAE), while a value model regresses to the aggregated reward, enabling multi-perspective reward shaping without requiring additional human annotations beyond those used to train the evaluators. To support training and assessment, we introduce rewardBench, a benchmark and training suite aligned with the collaborative structure of CRM. Together, CRM and rewardBench provide a practical, modular path to more transparent reward modeling and more stable optimization.

**arXiv ID:** 2511.16202
</details>

<details>
<summary><strong>LLM-DSE: Searching Accelerator Parameters with LLM Agents</strong> - Hanyu Wang, Xinrui Wu, Zijian Ding, Su Zheng, Chengyue Wang, Neha Prakriya, Tony Nowatzki, Yizhou Sun, Jason Cong - [[pdf]](https://arxiv.org/pdf/2505.12188)</summary>

**Abstract:** Even though high-level synthesis (HLS) tools mitigate the challenges of programming domain-specific accelerators (DSAs) by raising the abstraction level, optimizing hardware directive parameters remains a significant hurdle. Existing heuristic and learning-based methods struggle with adaptability and sample efficiency. We present LLM-DSE, a multi-agent framework designed specifically for optimizing HLS directives. Combining LLM with design space exploration (DSE), our explorer coordinates four agents: Router, Specialists, Arbitrator, and Critic. These multi-agent components interact with various tools to accelerate the optimization process. LLM-DSE leverages essential domain knowledge to identify efficient parameter combinations while maintaining adaptability through verbal learning from online interactions. Evaluations on the HLSyn dataset demonstrate that LLM-DSE achieves substantial $2.55\times$ performance gains over state-of-the-art methods, uncovering novel designs while reducing runtime. Ablation studies validate the effectiveness and necessity of the proposed agent interactions. Our code is open-sourced here: this https URL.

**arXiv ID:** 2505.12188
</details>

<details>
<summary><strong>MDG: Masked Denoising Generation for Multi-Agent Behavior Modeling in Traffic Environments</strong> - Zhiyu Huang, Zewei Zhou, Tianhui Cai, Yun Zhang, Jiaqi Ma - [[pdf]](https://arxiv.org/pdf/2511.17496)</summary>

**Abstract:** Modeling realistic and interactive multi-agent behavior is critical to autonomous driving and traffic simulation. However, existing diffusion and autoregressive approaches are limited by iterative sampling, sequential decoding, or task-specific designs, which hinder efficiency and reuse. We propose Masked Denoising Generation (MDG), a unified generative framework that reformulates multi-agent behavior modeling as the reconstruction of independently noised spatiotemporal tensors. Instead of relying on diffusion time steps or discrete tokenization, MDG applies continuous, per-agent and per-timestep noise masks that enable localized denoising and controllable trajectory generation in a single or few forward passes. This mask-driven formulation generalizes across open-loop prediction, closed-loop simulation, motion planning, and conditional generation within one model. Trained on large-scale real-world driving datasets, MDG achieves competitive closed-loop performance on the Waymo Sim Agents and nuPlan Planning benchmarks, while providing efficient, consistent, and controllable open-loop multi-agent trajectory generation. These results position MDG as a simple yet versatile paradigm for multi-agent behavior modeling.

**arXiv ID:** 2511.17496
</details>

<details>
<summary><strong>Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit</strong> - Kamal Mammadov, Damith C. Ranasinghe - [[pdf]](https://arxiv.org/pdf/2511.15036)</summary>

**Abstract:** This paper presents a novel strategy for a multi-agent pursuit-evasion game involving multiple faster pursuers with heterogenous speeds and a single slower evader. We define a geometric region, the evader's safe-reachable set, as the intersection of Apollonius circles derived from each pursuer-evader pair. The capture strategy is formulated as a zero-sum game where the pursuers cooperatively minimize the area of this set, while the evader seeks to maximize it, effectively playing a game of spatial containment. By deriving the analytical gradients of the safe-reachable set's area with respect to agent positions, we obtain closed-form, instantaneous optimal control laws for the heading of each agent. These strategies are computationally efficient, allowing for real-time implementation. Simulations demonstrate that the gradient-based controls effectively steer the pursuers to systematically shrink the evader's safe region, leading to guaranteed capture. This area-minimization approach provides a clear geometric objective for cooperative capture.

**arXiv ID:** 2511.15036
</details>

<details>
<summary><strong>NALA_MAINZ at BLP-2025 Task 2: A Multi-agent Approach for Bangla Instruction to Python Code Generation</strong> - Hossain Shaikh Saadi, Faria Alam, Mario Sanz-Guerrero, Minh Duc Bui, Manuel Mager, Katharina von der Wense - [[pdf]](https://arxiv.org/pdf/2511.16787)</summary>

**Abstract:** This paper presents JGU Mainz's winning system for the BLP-2025 Shared Task on Code Generation from Bangla Instructions. We propose a multi-agent-based pipeline. First, a code-generation agent produces an initial solution from the input instruction. The candidate program is then executed against the provided unit tests (pytest-style, assert-based). Only the failing cases are forwarded to a debugger agent, which reruns the tests, extracts error traces, and, conditioning on the error messages, the current program, and the relevant test cases, generates a revised solution. Using this approach, our submission achieved first place in the shared task with a $Pass@1$ score of 95.4. We also make our code public.

**arXiv ID:** 2511.16787
</details>

<details>
<summary><strong>ToC: Tree-of-Claims Search with Multi-Agent Language Models</strong> - Shuyang Yu, Jianan Liang, Hui Hu - [[pdf]](https://arxiv.org/pdf/2511.16972)</summary>

**Abstract:** Optimizing patent claims is a critical yet challenging task, demanding careful balance between maximizing novelty and preserving legal scope. Manual claim drafting is labor-intensive, costly, and inherently inconsistent, while conventional Large Language Models (LLMs) often lack the structured, iterative reasoning essential for precise claim refinement. To address these challenges, we introduce Tree of Claims (ToC), an innovative framework that redefines claim editing as a guided search problem. ToC synergistically integrates Monte Carlo Tree Search (MCTS) with a collaborative multi-agent system, comprising an LLM-based EditorAgent that proposes contextually grounded edits, and an ExaminerAgent that mimics patent examiner critiques through structured, chain-of-thought analyses of novelty and prior art disclosure. Driven by a carefully designed multi-objective reward function, ToC jointly optimizes novelty, scope retention, and semantic coherence. Experimental evaluation on a benchmark of 1145 claims demonstrates that ToC significantly outperforms standard LLMs in zero-shot and few-shot scenarios, achieving an average composite score improvement of 8\%, and up to 9\% in certain cases. Extensive experiments, including detailed ablation studies, validate ToC's efficacy in generating superior, legally robust claim revisions. Overall, ToC establishes a transparent, controllable, and interpretable methodology that effectively bridges advanced LLM reasoning capabilities with strategic MCTS planning for structured patent claim this http URL source code is available at this https URL.

**arXiv ID:** 2511.16972
</details>

<details>
<summary><strong>Multi-Agent Pointer Transformer: Seq-to-Seq Reinforcement Learning for Multi-Vehicle Dynamic Pickup-Delivery Problems</strong> - Zengyu Zou, Jingyuan Wang, Yixuan Huang, Junjie Wu - [[pdf]](https://arxiv.org/pdf/2511.17435)</summary>

**Abstract:** This paper addresses the cooperative Multi-Vehicle Dynamic Pickup and Delivery Problem with Stochastic Requests (MVDPDPSR) and proposes an end-to-end centralized decision-making framework based on sequence-to-sequence, named Multi-Agent Pointer Transformer (MAPT). MVDPDPSR is an extension of the vehicle routing problem and a spatio-temporal system optimization problem, widely applied in scenarios such as on-demand delivery. Classical operations research methods face bottlenecks in computational complexity and time efficiency when handling large-scale dynamic problems. Although existing reinforcement learning methods have achieved some progress, they still encounter several challenges: 1) Independent decoding across multiple vehicles fails to model joint action distributions; 2) The feature extraction network struggles to capture inter-entity relationships; 3) The joint action space is exponentially large. To address these issues, we designed the MAPT framework, which employs a Transformer Encoder to extract entity representations, combines a Transformer Decoder with a Pointer Network to generate joint action sequences in an AutoRegressive manner, and introduces a Relation-Aware Attention module to capture inter-entity relationships. Additionally, we guide the model's decision-making using informative priors to facilitate effective exploration. Experiments on 8 datasets demonstrate that MAPT significantly outperforms existing baseline methods in terms of performance and exhibits substantial computational time advantages compared to classical operations research methods.

**arXiv ID:** 2511.17435
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>Budget-Aware Tool-Use Enables Effective Agent Scaling</strong> - Tengxiao Liu, Zifeng Wang, Jin Miao, I-Hung Hsu, Jun Yan, Jiefeng Chen, Rujun Han, Fangyuan Xu, Yanfei Chen, Ke Jiang, Samira Daruki, Yi Liang, William Yang Wang, Tomas Pfister, Chen-Yu Lee - [[pdf]](https://arxiv.org/pdf/2511.17006)</summary>

**Abstract:** Scaling test-time computation improves performance across different tasks on large language models (LLMs), which has also been extended to tool-augmented agents. For these agents, scaling involves not only "thinking" in tokens but also "acting" via tool calls. The number of tool calls directly bounds the agent's interaction with the external environment. However, we find that simply granting agents a larger tool-call budget fails to improve performance, as they lack "budget awareness" and quickly hit a performance ceiling. To address this, we study how to scale such agents effectively under explicit tool-call budgets, focusing on web search agents. We first introduce the Budget Tracker, a lightweight plug-in that provides the agent with continuous budget awareness, enabling simple yet effective scaling. We further develop BATS (Budget Aware Test-time Scaling), an advanced framework that leverages this awareness to dynamically adapt its planning and verification strategy, deciding whether to "dig deeper" on a promising lead or "pivot" to new paths based on remaining resources. To analyze cost-performance scaling in a controlled manner, we formalize a unified cost metric that jointly accounts for token and tool consumption. We provide the first systematic study on budget-constrained agents, showing that budget-aware methods produce more favorable scaling curves and push the cost-performance Pareto frontier. Our work offers empirical insights toward a more transparent and principled understanding of scaling in tool-augmented agents.

**arXiv ID:** 2511.17006
</details>

<details>
<summary><strong>The promise and limits of LLMs in constructing proofs and hints for logic problems in intelligent tutoring systems</strong> - Sutapa Dey Tithi, Arun Kumar Ramesh, Clara DiMarco, Xiaoyi Tian, Nazia Alam, Kimia Fazeli, Tiffany Barnes - [[pdf]](https://arxiv.org/pdf/2505.04736)</summary>

**Abstract:** Intelligent tutoring systems have demonstrated effectiveness in teaching formal propositional logic proofs, but their reliance on template-based explanations limits their ability to provide personalized student feedback. While large language models (LLMs) offer promising capabilities for dynamic feedback generation, they risk producing hallucinations or pedagogically unsound explanations. We evaluated the stepwise accuracy of LLMs in constructing multi-step symbolic logic proofs, comparing six prompting techniques across four state-of-the-art LLMs on 358 propositional logic problems. Results show that DeepSeek-V3 achieved superior performance up to 86.7% accuracy on stepwise proof construction and excelled particularly in simpler rules. We further used the best-performing LLM to generate explanatory hints for 1,050 unique student problem-solving states from a logic ITS and evaluated them on 4 criteria with both an LLM grader and human expert ratings on a 20% sample. Our analysis finds that LLM-generated hints were 75% accurate and rated highly by human evaluators on consistency and clarity, but did not perform as well explaining why the hint was provided or its larger context. Our results demonstrate that LLMs may be used to augment tutoring systems with logic tutoring hints, but require additional modifications to ensure accuracy and pedagogical appropriateness.

**arXiv ID:** 2505.04736
</details>

<details>
<summary><strong>The Cooperative Network Architecture: Learning Structured Networks as Representation of Sensory Patterns</strong> - Pascal J. Sager, Jan M. Deriu, Benjamin F. Grewe, Thilo Stadelmann, Christoph von der Malsburg - [[pdf]](https://arxiv.org/pdf/2407.05650)</summary>

**Abstract:** We introduce the Cooperative Network Architecture (CNA), a model that represents sensory signals using structured, recurrently connected networks of neurons, termed "nets." Nets are dynamically assembled from overlapping net fragments, which are learned based on statistical regularities in sensory input. This architecture offers robustness to noise, deformation, and generalization to out-of-distribution data, addressing challenges in current vision systems from a novel perspective. We demonstrate that net fragments can be learned without supervision and flexibly recombined to encode novel patterns, enabling figure completion and resilience to noise. Our findings establish CNA as a promising paradigm for developing neural representations that integrate local feature processing with global structure formation, providing a foundation for future research on invariant object recognition.

**arXiv ID:** 2407.05650
</details>

<details>
<summary><strong>LLM-Agent-UMF: LLM-based Agent Unified Modeling Framework for Seamless Design of Multi Active/Passive Core-Agent Architectures</strong> - Amine Ben Hassouna, Hana Chaari, Ines Belhaj - [[pdf]](https://arxiv.org/pdf/2409.11393)</summary>

**Abstract:** In an era where vast amounts of data are collected and processed from diverse sources, there is a growing demand for sophisticated AI systems capable of intelligently fusing and analyzing this information. To address these challenges, researchers have turned towards integrating tools into LLM-powered agents to enhance the overall information fusion process. However, the conjunction of these technologies and the proposed enhancements in several state-of-the-art works followed a non-unified software architecture, resulting in a lack of modularity and terminological inconsistencies among researchers. To address these issues, we propose a novel LLM-based Agent Unified Modeling Framework (LLM-Agent-UMF) that establishes a clear foundation for agent development from both functional and software architectural perspectives, developed and evaluated using the Architecture Tradeoff and Risk Analysis Framework (ATRAF). Our framework clearly distinguishes between the different components of an LLM-based agent, setting LLMs and tools apart from a new element, the core-agent, which plays the role of central coordinator. This pivotal entity comprises five modules: planning, memory, profile, action, and security -- the latter often neglected in previous works. By classifying core-agents into passive and active types based on their authoritative natures, we propose various multi-core agent architectures that combine unique characteristics of distinctive agents to tackle complex tasks more efficiently. We evaluate our framework by applying it to thirteen state-of-the-art agents, thereby demonstrating its alignment with their functionalities and clarifying overlooked architectural aspects. Moreover, we thoroughly assess five architecture variants of our framework by designing new agent architectures that combine characteristics of state-of-the-art agents to address specific goals. ...

**arXiv ID:** 2409.11393
</details>

<details>
<summary><strong>AudAgent: Automated Auditing of Privacy Policy Compliance in AI Agents</strong> - Ye Zheng, Yidan Hu - [[pdf]](https://arxiv.org/pdf/2511.07441)</summary>

**Abstract:** AI agents can autonomously perform tasks and, often without explicit user consent, collect or disclose users' sensitive local data, which raises serious privacy concerns. Although AI agents' privacy policies describe their intended data practices, there remains limited transparency and accountability about whether runtime behavior matches those policies. To close this gap, we introduce AudAgent, a visual tool that continuously monitors AI agents' data practices in real time and guards compliance with stated privacy policies.
AudAgent consists of four components for automated privacy auditing of AI agents. (i) Policy formalization: a novel cross-LLM voting mechanism to guarantee confidence of the parsed privacy policy model. (ii) Runtime annotation: a lightweight Presidio-based analyzer detects sensitive data and annotates data practices based on the AI agent's context and the privacy policy model. (iii) Compliance auditing: ontology graphs and automata-based checking connect the privacy policy model with runtime annotations, enabling on-the-fly compliance checking. (iv) User interface: an infrastructure-independent implementation visualizes the real-time execution trace of AI agents along with potential privacy policy violations, providing user-friendly transparency and accountability.
We evaluate AudAgent with AI agents built using mainstream frameworks, demonstrating its effectiveness in detecting and visualizing privacy policy violations in real time. Using AudAgent, we also find that most privacy policies omit explicit safeguards for highly sensitive data such as SSNs, whose misuse violates legal requirements, and that many agents do not refuse handling such data via third-party tools, including those controlled by Claude, Gemini, and DeepSeek. AudAgent proactively blocks operations on such data, overriding the agents' original privacy policy and behavior.

**arXiv ID:** 2511.07441
</details>

<details>
<summary><strong>Reflection-Based Relative Localization for Cooperative UAV Teams Using Active Markers</strong> - Tim Lakemann, Daniel Bonilla Licea, Viktor Walter, Martin Saska - [[pdf]](https://arxiv.org/pdf/2511.17166)</summary>

**Abstract:** Reflections of active markers in the environment are a common source of ambiguity in onboard visual relative localization. This work presents a novel approach for onboard relative localization in multi-robot teams that exploits these typically unwanted reflections of active markers in the environment. It operates without prior knowledge of robot size or predefined marker configurations and remains independent of surface properties, an essential feature for heterogeneous micro-aerial swarms cooperating in unknown environments. It explicitly accounts for uncertainties caused by non-flat surfaces, with a particular focus on dynamic water surfaces, which are especially relevant for marine deployments. We validated the approach in both indoor and outdoor experiments, demonstrating that the proposed reflection-based localization system operates reliably without prior knowledge of team member size and achieves greater effective range (above 30 m) and accuracy than state-of-the-art methods. The video and source code of this work will be made publicly available after publication.

**arXiv ID:** 2511.17166
</details>

<details>
<summary><strong>VLM-SFD: VLM-Assisted Siamese Flow Diffusion Framework for Dual-Arm Cooperative Manipulation</strong> - Jiaming Chen, Yiyu Jiang, Aoshen Huang, Yang Li, Wei Pan - [[pdf]](https://arxiv.org/pdf/2506.13428)</summary>

**Abstract:** Dual-arm cooperative manipulation holds great promise for tackling complex real-world tasks that demand seamless coordination and adaptive dynamics. Despite substantial progress in learning-based motion planning, most approaches struggle to generalize across diverse manipulation tasks and adapt to dynamic, unstructured environments, particularly in scenarios involving interactions between two objects such as assembly, tool use, and bimanual grasping. To address these challenges, we introduce a novel VLM-Assisted Siamese Flow Diffusion (VLM-SFD) framework for efficient imitation learning in dual-arm cooperative manipulation. The proposed VLM-SFD framework exhibits outstanding adaptability, significantly enhancing the ability to rapidly adapt and generalize to diverse real-world tasks from only a minimal number of human demonstrations. Specifically, we propose a Siamese Flow Diffusion Network (SFDNet) employs a dual-encoder-decoder Siamese architecture to embed two target objects into a shared latent space, while a diffusion-based conditioning process - conditioned by task instructions - generates two-stream object-centric motion flows that guide dual-arm coordination. We further design a dynamic task assignment strategy that seamlessly maps the predicted 2D motion flows into 3D space and incorporates a pre-trained vision-language model (VLM) to adaptively assign the optimal motion to each robotic arm over time. Experiments validate the effectiveness of the proposed method, demonstrating its ability to generalize to diverse manipulation tasks while maintaining high efficiency and adaptability. The code and demo videos are publicly available on our project website this https URL.

**arXiv ID:** 2506.13428
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>Vector Cost Behavioral Planning for Autonomous Robotic Systems with Contemporary Validation Strategies</strong> - Benjamin R. Toaz, Quentin Goss, John Thompson, Seta Boğosyan, Shaunak D. Bopardikar, Mustafa İlhan Akbaş, Metin Gökaşan - [[pdf]](https://arxiv.org/pdf/2511.17375)</summary>

**Abstract:** The vector cost bimatrix game is a method for multi-objective decision making that enables autonomous robotic systems to optimize for multiple goals at once while avoiding worst-case scenarios in neglected objectives. We expand this approach to arbitrary numbers of objectives and compare its performance to scalar weighted sum methods during competitive motion planning. Explainable Artificial Intelligence (XAI) software is used to aid in the analysis of high dimensional decision-making data. State-space Exploration of Multidimensional Boundaries using Adherence Strategies (SEMBAS) is applied to explore performance modes in the parameter space as a sensitivity study for the baseline and proposed frameworks. While some works have explored aspects of game theoretic planning and intelligent systems validation separately, we combine each of these into a novel and comprehensive simulation pipeline. This integration demonstrates a dramatic improvement of the vector cost method over scalarization and offers an interpretable and generalizable framework for robotic behavioral planning. Code available at this https URL. The video companion to this work is available at this https URL.

**arXiv ID:** 2511.17375
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (15 papers)</h2></summary>

<details>
<summary><strong>TP-MDDN: Task-Preferenced Multi-Demand-Driven Navigation with Autonomous Decision-Making</strong> - Shanshan Li, Da Huang, Yu He, Yanwei Fu, Yu-Gang Jiang, Xiangyang Xue - [[pdf]](https://arxiv.org/pdf/2511.17225)</summary>

**Abstract:** In daily life, people often move through spaces to find objects that meet their needs, posing a key challenge in embodied AI. Traditional Demand-Driven Navigation (DDN) handles one need at a time but does not reflect the complexity of real-world tasks involving multiple needs and personal choices. To bridge this gap, we introduce Task-Preferenced Multi-Demand-Driven Navigation (TP-MDDN), a new benchmark for long-horizon navigation involving multiple sub-demands with explicit task preferences. To solve TP-MDDN, we propose AWMSystem, an autonomous decision-making system composed of three key modules: BreakLLM (instruction decomposition), LocateLLM (goal selection), and StatusMLLM (task monitoring). For spatial memory, we design MASMap, which combines 3D point cloud accumulation with 2D semantic mapping for accurate and efficient environmental understanding. Our Dual-Tempo action generation framework integrates zero-shot planning with policy-based fine control, and is further supported by an Adaptive Error Corrector that handles failure cases in real time. Experiments demonstrate that our approach outperforms state-of-the-art baselines in both perception accuracy and navigation robustness.

**arXiv ID:** 2511.17225
</details>

<details>
<summary><strong>Masked-and-Reordered Self-Supervision for Reinforcement Learning from Verifiable Rewards</strong> - Zhen Wang, Zhifeng Gao, Guolin Ke - [[pdf]](https://arxiv.org/pdf/2511.17473)</summary>

**Abstract:** Test-time scaling has been shown to substantially improve large language models' (LLMs) mathematical reasoning. However, for a large portion of mathematical corpora, especially theorem proving, RLVR's scalability is limited: intermediate reasoning is crucial, while final answers are difficult to directly and reliably verify. Meanwhile, token-level SFT often degenerates into rote memorization rather than inducing longer chains of thought. Inspired by BERT's self-supervised tasks, we propose MR-RLVR (Masked-and-Reordered RLVR), which constructs process-level self-supervised rewards via "masked-then-fill" and "step reordering" to extract learnable signals from intermediate reasoning. Our training pipeline comprises two stages: we first perform self-supervised training on sampled mathematical calculation and proof data; we then conduct RLVR fine-tuning on mathematical calculation datasets where only outcomes are verifiable. We implement MR-RLVR on Qwen2.5-3B and DeepSeek-R1-Distill-Qwen-1.5B, and evaluate on AIME24, AIME25, AMC23, and MATH500. Under a fixed sampling and decoding budget, MR-RLVR achieves average relative gains over the original RLVR of +9.86% Pass@1, +5.27% Pass@5, and +4.00% Pass@8. These results indicate that incorporating process-aware self-supervised signals can effectively enhance RLVR's scalability and performance in only outcome-verifiable settings.

**arXiv ID:** 2511.17473
</details>

<details>
<summary><strong>Platonic Representations for Poverty Mapping: Unified Vision-Language Codes or Agent-Induced Novelty?</strong> - Satiyabooshan Murugaboopathy, Connor T. Jerzak, Adel Daoud - [[pdf]](https://arxiv.org/pdf/2508.01109)</summary>

**Abstract:** We investigate whether socio-economic indicators like household wealth leave recoverable imprints in satellite imagery (capturing physical features) and Internet-sourced text (reflecting historical/economic narratives). Using Demographic and Health Survey (DHS) data from African neighborhoods, we pair Landsat images with LLM-generated textual descriptions conditioned on location/year and text retrieved by an AI search agent from web sources. We develop a multimodal framework predicting household wealth (International Wealth Index) through five pipelines: (i) vision model on satellite images, (ii) LLM using only location/year, (iii) AI agent searching/synthesizing web text, (iv) joint image-text encoder, (v) ensemble of all signals. Our framework yields three contributions. First, fusing vision and agent/LLM text outperforms vision-only baselines in wealth prediction (e.g., R-squared of 0.77 vs. 0.63 on out-of-sample splits), with LLM-internal knowledge proving more effective than agent-retrieved text, improving robustness to out-of-country and out-of-time generalization. Second, we find partial representational convergence: fused embeddings from vision/language modalities correlate moderately (median cosine similarity of 0.60 after alignment), suggesting a shared latent code of material well-being while retaining complementary details, consistent with the Platonic Representation Hypothesis. Although LLM-only text outperforms agent-retrieved data, challenging our Agent-Induced Novelty Hypothesis, modest gains from combining agent data in some splits weakly support the notion that agent-gathered information introduces unique representational structures not fully captured by static LLM knowledge. Third, we release a large-scale multimodal dataset comprising more than 60,000 DHS clusters linked to satellite images, LLM-generated descriptions, and agent-retrieved texts.

**arXiv ID:** 2508.01109
</details>

<details>
<summary><strong>ResearStudio: A Human-Intervenable Framework for Building Controllable Deep-Research Agents</strong> - Linyi Yang, Yixuan Weng - [[pdf]](https://arxiv.org/pdf/2510.12194)</summary>

**Abstract:** Current deep-research agents run in a ''fire-and-forget'' mode: once started, they give users no way to fix errors or add expert knowledge during execution. We present ResearStudio, the first open-source framework that places real-time human control at its core. The system follows a Collaborative Workshop design. A hierarchical Planner-Executor writes every step to a live ''plan-as-document,'' a fast communication layer streams each action, file change, and tool call to a web interface. At any moment, the user can pause the run, edit the plan or code, run custom commands, and resume -- switching smoothly between AI-led, human-assisted and human-led, AI-assisted modes. In fully autonomous mode, ResearStudio achieves state-of-the-art results on the GAIA benchmark, surpassing systems like OpenAI's DeepResearch and Manus. These results show that strong automated performance and fine-grained human control can coexist. The full code, protocol, and evaluation scripts are available at this https URL. We will continue to update the repository to encourage further work on safe and controllable research agents. Our live demo is publicly accessible at this http URL. We support the development of DeepScientist, which can be accessed at this https URL.

**arXiv ID:** 2510.12194
</details>

<details>
<summary><strong>Multi-Objective Reinforcement Learning for Water Management</strong> - Zuzanna Osika, Roxana Rădulescu, Jazmin Zatarain Salazar, Frans Oliehoek, Pradeep K. Murukannaiah - [[pdf]](https://arxiv.org/pdf/2505.01094)</summary>

**Abstract:** Many real-world problems (e.g., resource management, autonomous driving, drug discovery) require optimizing multiple, conflicting objectives. Multi-objective reinforcement learning (MORL) extends classic reinforcement learning to handle multiple objectives simultaneously, yielding a set of policies that capture various trade-offs. However, the MORL field lacks complex, realistic environments and benchmarks. We introduce a water resource (Nile river basin) management case study and model it as a MORL environment. We then benchmark existing MORL algorithms on this task. Our results show that specialized water management methods outperform state-of-the-art MORL approaches, underscoring the scalability challenges MORL algorithms face in real-world scenarios.

**arXiv ID:** 2505.01094
</details>

<details>
<summary><strong>A Reinforcement Learning-Based Telematic Routing Protocol for the Internet of Underwater Things</strong> - Mohammadhossein Homaei, Mehran Tarif, Agustin Di Bartolo, Victor Gonzalez Morales, Mar Avila Vegas - [[pdf]](https://arxiv.org/pdf/2506.00133)</summary>

**Abstract:** The Internet of Underwater Things (IoUT) has a lot of problems, like low bandwidth, high latency, mobility, and not enough energy. Routing protocols that were made for land-based networks, like RPL, don't work well in these underwater settings. This paper talks about RL-RPL-UA, a new routing protocol that uses reinforcement learning to make things work better in underwater situations. Each node has a small RL agent that picks the best parent node depending on local data such the link quality, buffer level, packet delivery ratio, and remaining energy. RL-RPL-UA works with all standard RPL messages and adds a dynamic objective function to help people make decisions in real time. Aqua-Sim simulations demonstrate that RL-RPL-UA boosts packet delivery by up to 9.2%, uses 14.8% less energy per packet, and adds 80 seconds to the network's lifetime compared to previous approaches. These results show that RL-RPL-UA is a potential and energy-efficient way to route data in underwater networks.

**arXiv ID:** 2506.00133
</details>

<details>
<summary><strong>Efficient Reinforcement Learning for Large Language Models with Intrinsic Exploration</strong> - Yan Sun, Jia Guo, Stanley Kok, Zihao Wang, Zujie Wen, Zhiqiang Zhang - [[pdf]](https://arxiv.org/pdf/2511.00794)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has improved the reasoning ability of large language models, yet training remains costly because many rollouts contribute little to optimization, considering the amount of computation required. This study investigates how simply leveraging intrinsic data properties, almost free benefit during training, can improve data efficiency for RLVR. We propose PREPO with two complementary components. First, we adopt prompt perplexity as an indicator of model adaptability in learning, enabling the model to progress from well-understood contexts to more challenging ones. Second, we amplify the discrepancy among the rollouts by differentiating their relative entropy, and prioritize sequences that exhibit a higher degree of exploration. Together, these mechanisms reduce rollout demand while preserving competitive performance. On the Qwen and Llama models, PREPO achieves effective results on mathematical reasoning benchmarks with up to 3 times fewer rollouts than the baselines. Beyond empirical gains, we provide theoretical and in-depth analyses explaining the underlying rationale of our method to improve the data efficiency of RLVR.

**arXiv ID:** 2511.00794
</details>

<details>
<summary><strong>Steering Noncooperative Games Through Conjecture Design</strong> - Francesco Morri, Hélène Le Cadre, David Salas, Didier Aussel - [[pdf]](https://arxiv.org/pdf/2511.09435)</summary>

**Abstract:** In dynamic noncooperative games, each player makes conjectures about other players' reactions before choosing a strategy. However, resulting equilibria may be multiple and do not always lead to desirable outcomes. These issues are typically addressed separately, for example, through opponent modelling and incentive design. Drawing inspiration from conjectural variations games, we propose an incentive design framework in which a coordinator first computes an equilibrium by optimizing a predefined objective function, then communicates this equilibrium as a target for the players to reach. In a centralized setting, the coordinator also optimizes the conjectures to steer the players towards the target. In decentralized settings, players independently compute conjectures and update their strategies based on individual targets. We provide a guarantee of equilibrium existence in both cases. This framework uses conjectures not only to guide the system towards desirable outcomes but also to decouple the game into independent optimization problems, enabling efficient computation and parallelization in large-scale settings. We illustrate our theoretical results on classical representative noncooperative games, demonstrating its application potential.

**arXiv ID:** 2511.09435
</details>

<details>
<summary><strong>Interpretable dimensions support an effect of agentivity and telicity on split intransitivity</strong> - Eva Neu, Brian Dillon, Katrin Erk - [[pdf]](https://arxiv.org/pdf/2511.16824)</summary>

**Abstract:** Intransitive verbs fall into two different syntactic classes, unergatives and unaccusatives. It has long been argued that verbs describing an agentive action are more likely to appear in an unergative syntax, and those describing a telic event to appear in an unaccusative syntax. However, recent work by Kim et al. (2024) found that human ratings for agentivity and telicity were a poor predictor of the syntactic behavior of intransitives. Here we revisit this question using interpretable dimensions, computed from seed words on opposite poles of the agentive and telic scales. Our findings support the link between unergativity/unaccusativity and agentivity/telicity, and demonstrate that using interpretable dimensions in conjunction with human judgments can offer valuable evidence for semantic properties that are not easily evaluated in rating tasks.

**arXiv ID:** 2511.16824
</details>

<details>
<summary><strong>EventWeave: A Dynamic Framework for Capturing Core and Supporting Events in Dialogue Systems</strong> - Zhengyi Zhao, Shubo Zhang, Yiming Du, Bin Liang, Baojun Wang, Zhongyang Li, Binyang Li, Kam-Fai Wong - [[pdf]](https://arxiv.org/pdf/2503.23078)</summary>

**Abstract:** Large language models have improved dialogue systems, but often process conversational turns in isolation, overlooking the event structures that guide natural interactions. Hence we introduce \textbf{EventWeave}, a framework that explicitly models relationships between conversational events to generate more contextually appropriate dialogue responses. EventWeave constructs a dynamic event graph that distinguishes between core events (main goals) and supporting events (interconnected details), employing a multi-head attention mechanism to selectively determine which events are most relevant to the current turn. Unlike summarization or standard graph-based approaches, our method captures three distinct relationship types between events, allowing for more nuanced context modeling. Experiments on three dialogue datasets demonstrate that EventWeave produces more natural and contextually appropriate responses while requiring less computational overhead than models processing the entire dialogue history. Ablation studies confirm improvements stem from better event relationship modeling rather than increased information density. Our approach effectively balances comprehensive context understanding with generating concise responses, maintaining strong performance across various dialogue lengths through targeted optimization techniques.

**arXiv ID:** 2503.23078
</details>

<details>
<summary><strong>Concise Reasoning via Reinforcement Learning</strong> - Mehdi Fatemi, Banafsheh Rafiee, Mingjie Tang, Kartik Talamadupula - [[pdf]](https://arxiv.org/pdf/2504.05185)</summary>

**Abstract:** A major drawback of reasoning models is their excessive token usage, inflating computational cost, resource demand, and latency. We show this verbosity stems not from deeper reasoning but from reinforcement learning loss minimization when models produce incorrect answers. With unsolvable problems dominating training, this effect compounds into a systematic tendency toward longer outputs. Through theoretical analysis of PPO and GRPO, we prove that incorrect answers inherently drive policies toward verbosity \textit{even when} $\gamma=1$, reframing response lengthening as an optimization artifact. We further uncover a consistent correlation between conciseness and correctness across reasoning and non-reasoning models. Building on these insights, we propose a two-phase RL procedure where a brief secondary stage, trained on a small set of solvable problems, significantly reduces response length while preserving or improving accuracy. Finally, we show that while GRPO shares properties with PPO, it exhibits collapse modes, limiting its reliability for concise reasoning. Our claims are supported by extensive experiments.

**arXiv ID:** 2504.05185
</details>

<details>
<summary><strong>CroTad: A Contrastive Reinforcement Learning Framework for Online Trajectory Anomaly Detection</strong> - Rui Xue, Dan He, Fengmei Jin, Chen Zhang, Xiaofang Zhou - [[pdf]](https://arxiv.org/pdf/2511.16929)</summary>

**Abstract:** Detecting trajectory anomalies is a vital task in modern Intelligent Transportation Systems (ITS), enabling the identification of unsafe, inefficient, or irregular travel behaviours. While deep learning has emerged as the dominant approach, several key challenges remain unresolved. First, sub-trajectory anomaly detection, capable of pinpointing the precise segments where anomalies occur, remains underexplored compared to whole-trajectory analysis. Second, many existing methods depend on carefully tuned thresholds, limiting their adaptability in real-world applications. Moreover, the irregular sampling of trajectory data and the presence of noise in training sets further degrade model performance, making it difficult to learn reliable representations of normal routes. To address these challenges, we propose a contrastive reinforcement learning framework for online trajectory anomaly detection, CroTad. Our method is threshold-free and robust to noisy, irregularly sampled data. By incorporating contrastive learning, CroTad learns to extract diverse normal travel patterns for different itineraries and effectively distinguish anomalous behaviours at both sub-trajectory and point levels. The detection module leverages deep reinforcement learning to perform online, real-time anomaly scoring, enabling timely and fine-grained identification of abnormal segments. Extensive experiments on two real-world datasets demonstrate the effectiveness and robustness of our framework across various evaluation scenarios.

**arXiv ID:** 2511.16929
</details>

<details>
<summary><strong>Convergence and stability of Q-learning in Hierarchical Reinforcement Learning</strong> - Massimiliano Manenti, Andrea Iannelli - [[pdf]](https://arxiv.org/pdf/2511.17351)</summary>

**Abstract:** Hierarchical Reinforcement Learning promises, among other benefits, to efficiently capture and utilize the temporal structure of a decision-making problem and to enhance continual learning capabilities, but theoretical guarantees lag behind practice. In this paper, we propose a Feudal Q-learning scheme and investigate under which conditions its coupled updates converge and are stable. By leveraging the theory of Stochastic Approximation and the ODE method, we present a theorem stating the convergence and stability properties of Feudal Q-learning. This provides a principled convergence and stability analysis tailored to Feudal RL. Moreover, we show that the updates converge to a point that can be interpreted as an equilibrium of a suitably defined game, opening the door to game-theoretic approaches to Hierarchical RL. Lastly, experiments based on the Feudal Q-learning algorithm support the outcomes anticipated by theory.

**arXiv ID:** 2511.17351
</details>

<details>
<summary><strong>Dissecting Quantum Reinforcement Learning: A Systematic Evaluation of Key Components</strong> - Javier Lazaro, Juan-Ignacio Vazquez, Pablo Garcia-Bringas - [[pdf]](https://arxiv.org/pdf/2511.17112)</summary>

**Abstract:** Parameterised quantum circuit (PQC) based Quantum Reinforcement Learning (QRL) has emerged as a promising paradigm at the intersection of quantum computing and reinforcement learning (RL). By design, PQCs create hybrid quantum-classical models, but their practical applicability remains uncertain due to training instabilities, barren plateaus (BPs), and the difficulty of isolating the contribution of individual pipeline components. In this work, we dissect PQC based QRL architectures through a systematic experimental evaluation of three aspects recurrently identified as critical: (i) data embedding strategies, with Data Reuploading (DR) as an advanced approach; (ii) ansatz design, particularly the role of entanglement; and (iii) post-processing blocks after quantum measurement, with a focus on the underexplored Output Reuse (OR) technique. Using a unified PPO-CartPole framework, we perform controlled comparisons between hybrid and classical agents under identical conditions. Our results show that OR, though purely classical, exhibits distinct behaviour in hybrid pipelines, that DR improves trainability and stability, and that stronger entanglement can degrade optimisation, offsetting classical gains. Together, these findings provide controlled empirical evidence of the interplay between quantum and classical contributions, and establish a reproducible framework for systematic benchmarking and component-wise analysis in QRL.

**arXiv ID:** 2511.17112
</details>

<details>
<summary><strong>Feasibility of Embodied Dynamics Based Bayesian Learning for Continuous Pursuit Motion Control of Assistive Mobile Robots in the Built Environment</strong> - Xiaoshan Zhou, Carol C. Menassa, Vineet R. Kamat - [[pdf]](https://arxiv.org/pdf/2511.17401)</summary>

**Abstract:** Non-invasive electroencephalography (EEG)-based brain-computer interfaces (BCIs) offer an intuitive means for individuals with severe motor impairments to independently operate assistive robotic wheelchairs and navigate built environments. Despite considerable progress in BCI research, most current motion control systems are limited to discrete commands, rather than supporting continuous pursuit, where users can freely adjust speed and direction in real time. Such natural mobility control is, however, essential for wheelchair users to navigate complex public spaces, such as transit stations, airports, hospitals, and indoor corridors, to interact socially with the dynamic populations with agility, and to move flexibly and comfortably as autonomous driving is refined to allow movement at will. In this study, we address the gap of continuous pursuit motion control in BCIs by proposing and validating a brain-inspired Bayesian inference framework, where embodied dynamics in acceleration-based motor representations are decoded. This approach contrasts with conventional kinematics-level decoding and deep learning-based methods. Using a public dataset with sixteen hours of EEG from four subjects performing motor imagery-based target-following, we demonstrate that our method, utilizing Automatic Relevance Determination for feature selection and continual online learning, reduces the normalized mean squared error between predicted and true velocities by 72% compared to autoregressive and EEGNet-based methods in a session-accumulative transfer learning setting. Theoretically, these findings empirically support embodied cognition theory and reveal the brain's intrinsic motor control dynamics in an embodied and predictive nature. Practically, grounding EEG decoding in the same dynamical principles that govern biological motion offers a promising path toward more stable and intuitive BCI control.

**arXiv ID:** 2511.17401
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
