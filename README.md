# Agent arXiv Daily

**Last Updated:** 2025-12-18 02:56:59

**Total Papers:** 55

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
<summary><strong>Workflows vs Agents for Code Translation</strong> - Henry Gray, Tom Yotam, Octavian Udrea - [[pdf]](https://arxiv.org/pdf/2512.14762)</summary>

**Abstract:** Translating algorithms from high-level languages like MATLAB to hardware description languages (HDLs) is a resource-intensive but necessary step for deployment on FPGAs and ASICs. While large language models (LLMs) offer a path to automation, their limited training on HDL code makes end-to-end transpilation brittle and prone to syntax errors. We compare two LLM-driven methods for syntax repair in a MATLAB-to-HDL pipeline: a structured, expert-designed flow that follows a fixed sequence of operations, and a more autonomous agentic approach that uses the Model Context Protocol (MCP) \cite{anthropic2024mcp} to dynamically select its own tools. We study 42 MATLAB signal-processing functions and isolate the syntax-repair stage. Across three model scales, the agentic approach is more effective at resolving initial syntax errors, unblocking a greater number of candidates to proceed through the pipeline. This upstream improvement yields measurable downstream improvements, most notably on mid-sized models, where it increases the simulation reach rate by over 20 percentage points. We hypothesize the gains come from short prompts, aggressive context management, and conditional tool use. Conditional retrieval helps at 8B and 30B; at 235B final-success gains are small and a naive RAG variant attains the highest final success. Our findings suggest that these agentic frameworks, when properly designed, are most effective at compensating for the capacity limits of small and mid-sized models.

**arXiv ID:** 2512.14762
</details>

<details>
<summary><strong>Exploring User Acceptance and Concerns toward LLM-powered Conversational Agents in Immersive Extended Reality</strong> - Efe Bozkir, Enkelejda Kasneci - [[pdf]](https://arxiv.org/pdf/2512.15343)</summary>

**Abstract:** The rapid development of generative artificial intelligence (AI) and large language models (LLMs), and the availability of services that make them accessible, have led the general public to begin incorporating them into everyday life. The extended reality (XR) community has also sought to integrate LLMs, particularly in the form of conversational agents, to enhance user experience and task efficiency. When interacting with such conversational agents, users may easily disclose sensitive information due to the naturalistic flow of the conversations, and combining such conversational data with fine-grained sensor data may lead to novel privacy issues. To address these issues, a user-centric understanding of technology acceptance and concerns is essential. Therefore, to this end, we conducted a large-scale crowdsourcing study with 1036 participants, examining user decision-making processes regarding LLM-powered conversational agents in XR, across factors of XR setting type, speech interaction type, and data processing location. We found that while users generally accept these technologies, they express concerns related to security, privacy, social implications, and trust. Our results suggest that familiarity plays a crucial role, as daily generative AI use is associated with greater acceptance. In contrast, previous ownership of XR devices is linked to less acceptance, possibly due to existing familiarity with the settings. We also found that men report higher acceptance with fewer concerns than women. Regarding data type sensitivity, location data elicited the most significant concern, while body temperature and virtual object states were considered least sensitive. Overall, our study highlights the importance of practitioners effectively communicating their measures to users, who may remain distrustful. We conclude with implications and recommendations for LLM-powered XR.

**arXiv ID:** 2512.15343
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (15 papers)</h2></summary>

<details>
<summary><strong>CangLing-KnowFlow: A Unified Knowledge-and-Flow-fused Agent for Comprehensive Remote Sensing Applications</strong> - Zhengchao Chen, Haoran Wang, Jing Yao, Pedram Ghamisi, Jun Zhou, Peter M. Atkinson, Bing Zhang - [[pdf]](https://arxiv.org/pdf/2512.15231)</summary>

**Abstract:** The automated and intelligent processing of massive remote sensing (RS) datasets is critical in Earth observation (EO). Existing automated systems are normally task-specific, lacking a unified framework to manage diverse, end-to-end workflows--from data preprocessing to advanced interpretation--across diverse RS applications. To address this gap, this paper introduces CangLing-KnowFlow, a unified intelligent agent framework that integrates a Procedural Knowledge Base (PKB), Dynamic Workflow Adjustment, and an Evolutionary Memory Module. The PKB, comprising 1,008 expert-validated workflow cases across 162 practical RS tasks, guides planning and substantially reduces hallucinations common in general-purpose agents. During runtime failures, the Dynamic Workflow Adjustment autonomously diagnoses and replans recovery strategies, while the Evolutionary Memory Module continuously learns from these events, iteratively enhancing the agent's knowledge and performance. This synergy enables CangLing-KnowFlow to adapt, learn, and operate reliably across diverse, complex tasks. We evaluated CangLing-KnowFlow on the KnowFlow-Bench, a novel benchmark of 324 workflows inspired by real-world applications, testing its performance across 13 top Large Language Model (LLM) backbones, from open-source to commercial. Across all complex tasks, CangLing-KnowFlow surpassed the Reflexion baseline by at least 4% in Task Success Rate. As the first most comprehensive validation along this emerging field, this research demonstrates the great potential of CangLing-KnowFlow as a robust, efficient, and scalable automated solution for complex EO challenges by leveraging expert knowledge (Knowledge) into adaptive and verifiable procedures (Flow).

**arXiv ID:** 2512.15231
</details>

<details>
<summary><strong>SCOPE: Prompt Evolution for Enhancing Agent Effectiveness</strong> - Zehua Pei, Hui-Ling Zhen, Shixiong Kai, Sinno Jialin Pan, Yunhe Wang, Mingxuan Yuan, Bei Yu - [[pdf]](https://arxiv.org/pdf/2512.15374)</summary>

**Abstract:** Large Language Model (LLM) agents are increasingly deployed in environments that generate massive, dynamic contexts. However, a critical bottleneck remains: while agents have access to this context, their static prompts lack the mechanisms to manage it effectively, leading to recurring Corrective and Enhancement failures. To address this capability gap, we introduce \textbf{SCOPE} (Self-evolving Context Optimization via Prompt Evolution). SCOPE frames context management as an \textit{online optimization} problem, synthesizing guidelines from execution traces to automatically evolve the agent's prompt. We propose a Dual-Stream mechanism that balances tactical specificity (resolving immediate errors) with strategic generality (evolving long-term principles). Furthermore, we introduce Perspective-Driven Exploration to maximize strategy coverage, increasing the likelihood that the agent has the correct strategy for any given task. Experiments on the HLE benchmark show that SCOPE improves task success rates from 14.23\% to 38.64\% without human intervention. We make our code publicly available at this https URL.

**arXiv ID:** 2512.15374
</details>

<details>
<summary><strong>Autonomous Source Knowledge Selection in Multi-Domain Adaptation</strong> - Keqiuyin Li, Jie Lu, Hua Zuo, Guangquan Zhang - [[pdf]](https://arxiv.org/pdf/2512.14710)</summary>

**Abstract:** Unsupervised multi-domain adaptation plays a key role in transfer learning by leveraging acquired rich source information from multiple source domains to solve target task from an unlabeled target domain. However, multiple source domains often contain much redundant or unrelated information which can harm transfer performance, especially when in massive-source domain settings. It is urgent to develop effective strategies for identifying and selecting the most transferable knowledge from massive source domains to address the target task. In this paper, we propose a multi-domain adaptation method named \underline{\textit{Auto}}nomous Source Knowledge \underline{\textit{S}}election (AutoS) to autonomosly select source training samples and models, enabling the prediction of target task using more relevant and transferable source information. The proposed method employs a density-driven selection strategy to choose source samples during training and to determine which source models should contribute to target prediction. Simulteneously, a pseudo-label enhancement module built on a pre-trained multimodal modal is employed to mitigate target label noise and improve self-supervision. Experiments on real-world datasets indicate the superiority of the proposed method.

**arXiv ID:** 2512.14710
</details>

<details>
<summary><strong>VERAFI: Verified Agentic Financial Intelligence through Neurosymbolic Policy Generation</strong> - Adewale Akinfaderin, Shreyas Subramanian - [[pdf]](https://arxiv.org/pdf/2512.14744)</summary>

**Abstract:** Financial AI systems suffer from a critical blind spot: while Retrieval-Augmented Generation (RAG) excels at finding relevant documents, language models still generate calculation errors and regulatory violations during reasoning, even with perfect retrieval. This paper introduces VERAFI (Verified Agentic Financial Intelligence), an agentic framework with neurosymbolic policy generation for verified financial intelligence. VERAFI combines state-of-the-art dense retrieval and cross-encoder reranking with financial tool-enabled agents and automated reasoning policies covering GAAP compliance, SEC requirements, and mathematical validation. Our comprehensive evaluation on FinanceBench demonstrates remarkable improvements: while traditional dense retrieval with reranking achieves only 52.4\% factual correctness, VERAFI's integrated approach reaches 94.7\%, an 81\% relative improvement. The neurosymbolic policy layer alone contributes a 4.3 percentage point gain over pure agentic processing, specifically targeting persistent mathematical and logical errors. By integrating financial domain expertise directly into the reasoning process, VERAFI offers a practical pathway toward trustworthy financial AI that meets the stringent accuracy demands of regulatory compliance, investment decisions, and risk management.

**arXiv ID:** 2512.14744
</details>

<details>
<summary><strong>Penetration Testing of Agentic AI: A Comparative Security Analysis Across Models and Frameworks</strong> - Viet K. Nguyen, Mohammad I. Husain - [[pdf]](https://arxiv.org/pdf/2512.14860)</summary>

**Abstract:** Agentic AI introduces security vulnerabilities that traditional LLM safeguards fail to address. Although recent work by Unit 42 at Palo Alto Networks demonstrated that ChatGPT-4o successfully executes attacks as an agent that it refuses in chat mode, there is no comparative analysis in multiple models and frameworks. We conducted the first systematic penetration testing and comparative evaluation of agentic AI systems, testing five prominent models (Claude 3.5 Sonnet, Gemini 2.5 Flash, GPT-4o, Grok 2, and Nova Pro) across two agentic AI frameworks (AutoGen and CrewAI) using a seven-agent architecture that mimics the functionality of a university information management system and 13 distinct attack scenarios that span prompt injection, Server Side Request Forgery (SSRF), SQL injection, and tool misuse. Our 130 total test cases reveal significant security disparities: AutoGen demonstrates a 52.3% refusal rate versus CrewAI's 30.8%, while model performance ranges from Nova Pro's 46.2% to Claude and Grok 2's 38.5%. Most critically, Grok 2 on CrewAI rejected only 2 of 13 attacks (15.4% refusal rate), and the overall refusal rate of 41.5% across all configurations indicates that more than half of malicious prompts succeeded despite enterprise-grade safety mechanisms. We identify six distinct defensive behavior patterns including a novel "hallucinated compliance" strategy where models fabricate outputs rather than executing or refusing attacks, and provide actionable recommendations for secure agent deployment. Complete attack prompts are also included in the Appendix to enable reproducibility.

**arXiv ID:** 2512.14860
</details>

<details>
<summary><strong>BashArena: A Control Setting for Highly Privileged AI Agents</strong> - Adam Kaufman, James Lucassen, Tyler Tracy, Cody Rushing, Aryan Bhatt - [[pdf]](https://arxiv.org/pdf/2512.15688)</summary>

**Abstract:** Future AI agents might run autonomously with elevated privileges. If these agents are misaligned, they might abuse these privileges to cause serious damage. The field of AI control develops techniques that make it harder for misaligned AIs to cause such damage, while preserving their usefulness. We introduce BashArena, a setting for studying AI control techniques in security-critical environments. BashArena contains 637 Linux system administration and infrastructure engineering tasks in complex, realistic environments, along with four sabotage objectives (execute malware, exfiltrate secrets, escalate privileges, and disable firewall) for a red team to target. We evaluate multiple frontier LLMs on their ability to complete tasks, perform sabotage undetected, and detect sabotage attempts. Claude Sonnet 4.5 successfully executes sabotage while evading monitoring by GPT-4.1 mini 26% of the time, at 4% trajectory-wise FPR. Our findings provide a baseline for designing more effective control protocols in BashArena. We release the dataset as a ControlArena setting and share our task generation pipeline.

**arXiv ID:** 2512.15688
</details>

<details>
<summary><strong>What Is Your AI Agent Buying? Evaluation, Biases, Model Dependence, & Emerging Implications for Agentic E-Commerce</strong> - Amine Allouah, Omar Besbes, Josué D Figueroa, Yash Kanoria, Akshit Kumar - [[pdf]](https://arxiv.org/pdf/2508.02630)</summary>

**Abstract:** Online marketplaces will be transformed by autonomous AI agents acting on behalf of consumers. Rather than humans browsing and clicking, AI agents can parse webpages or leverage APIs to view, evaluate and choose products. We investigate the behavior of AI agents using ACES, a provider-agnostic framework for auditing agent decision-making. We reveal that agents can exhibit choice homogeneity, often concentrating demand on a few ``modal'' products while ignoring others entirely. Yet, these preferences are unstable: model updates can drastically reshuffle market shares. Furthermore, randomized trials show that while agents have improved over time on simple tasks with a clearly identified best choice, they exhibit strong position biases -- varying across providers and model versions, and persisting even in text-only "headless" interfaces -- undermining any universal notion of a ``top'' rank. Agents also consistently penalize sponsored tags while rewarding platform endorsements, and sensitivities to price, ratings, and reviews vary sharply across models. Finally, we demonstrate that sellers can respond: a seller-side agent making simple, query-conditional description tweaks can drive significant gains in market share. These findings reveal that agentic markets are volatile and fundamentally different from human-centric commerce, highlighting the need for continuous auditing and raising questions for platform design, seller strategy and regulation.

**arXiv ID:** 2508.02630
</details>

<details>
<summary><strong>3DLLM-Mem: Long-Term Spatial-Temporal Memory for Embodied 3D Large Language Model</strong> - Wenbo Hu, Yining Hong, Yanjun Wang, Leison Gao, Zibu Wei, Xingcheng Yao, Nanyun Peng, Yonatan Bitton, Idan Szpektor, Kai-Wei Chang - [[pdf]](https://arxiv.org/pdf/2505.22657)</summary>

**Abstract:** Humans excel at performing complex tasks by leveraging long-term memory across temporal and spatial experiences. In contrast, current Large Language Models (LLMs) struggle to effectively plan and act in dynamic, multi-room 3D environments. We posit that part of this limitation is due to the lack of proper 3D spatial-temporal memory modeling in LLMs. To address this, we first introduce 3DMem-Bench, a comprehensive benchmark comprising over 26,000 trajectories and 2,892 embodied tasks, question-answering and captioning, designed to evaluate an agent's ability to reason over long-term memory in 3D environments. Second, we propose 3DLLM-Mem, a novel dynamic memory management and fusion model for embodied spatial-temporal reasoning and actions in LLMs. Our model uses working memory tokens, which represents current observations, as queries to selectively attend to and fuse the most useful spatial and temporal features from episodic memory, which stores past observations and interactions. Our approach allows the agent to focus on task-relevant information while maintaining memory efficiency in complex, long-horizon environments. Experimental results demonstrate that 3DLLM-Mem achieves state-of-the-art performance across various tasks, outperforming the strongest baselines by 16.5% in success rate on 3DMem-Bench's most challenging in-the-wild embodied tasks.

**arXiv ID:** 2505.22657
</details>

<details>
<summary><strong>Embodied Co-Design for Rapidly Evolving Agents: Taxonomy, Frontiers, and Challenges</strong> - Yuxing Wang, Zhiyu Chen, Tiantian Zhang, Qiyue Yin, Yongzhe Chang, Zhiheng Li, Liang Wang, Xueqian Wang - [[pdf]](https://arxiv.org/pdf/2512.04770)</summary>

**Abstract:** Brain-body co-evolution enables animals to develop complex behaviors in their environments. Inspired by this biological synergy, embodied co-design (ECD) has emerged as a transformative paradigm for creating intelligent agents-from virtual creatures to physical robots-by jointly optimizing their morphologies and controllers rather than treating control in isolation. This integrated approach facilitates richer environmental interactions and robust task performance. In this survey, we provide a systematic overview of recent advances in ECD. We first formalize the concept of ECD and position it within related fields. We then introduce a hierarchical taxonomy: a lower layer that breaks down agent design into three fundamental components-controlling brain, body morphology, and task environment-and an upper layer that integrates these components into four major ECD frameworks: bi-level, single-level, generative, and open-ended. This taxonomy allows us to synthesize insights from more than one hundred recent studies. We further review notable benchmarks, datasets, and applications in both simulated and real-world scenarios. Finally, we identify significant challenges and offer insights into promising future research directions. A project associated with this survey has been created at this https URL.

**arXiv ID:** 2512.04770
</details>

<details>
<summary><strong>From Segments to Scenes: Temporal Understanding in Autonomous Driving via Vision-Language Model</strong> - Kevin Cannons, Saeed Ranjbar Alvar, Mohammad Asiful Hossain, Ahmad Rezaei, Mohsen Gholami, Alireza Heidarikhazaei, Zhou Weimin, Yong Zhang, Mohammad Akbari - [[pdf]](https://arxiv.org/pdf/2512.05277)</summary>

**Abstract:** Temporal understanding in autonomous driving (AD) remains a significant challenge, even for recent state-of-the-art (SoTA) Vision-Language Models (VLMs). Prior work has introduced datasets and benchmarks aimed at improving temporal reasoning, but these have emphasized other video content, including sports, cooking, and movies. No existing benchmark focuses exclusively on the unique challenges of temporal understanding in ego-centric AD footage. To fill this gap, the Temporal Understanding in Autonomous Driving (TAD) benchmark is presented, which evaluates VLMs' ability to capture the dynamic relationships between actions in AD. TAD comprises nearly 6,000 question-answer (QA) pairs, spanning 7 human-designed tasks. In addition, an evaluation is performed that consists of 9 closed- and open-source generalist models as well as SoTA AD specialist models. When applied to TAD, current SoTA models demonstrated substandard accuracies, largely due to imperfect fine-grained motion understanding. To improve motion understanding and overall accuracy on TAD, two novel training-free solutions are proposed: Scene-CoT, that leverages Chain-of-Thought (CoT) and TCogMap, which incorporates an ego-centric temporal cognitive map. The proposed approaches are integrated with existing VLMs and improve average accuracy on TAD by up to 17.72%. By introducing TAD, benchmarking multiple SoTA models, and proposing effective enhancements, this work aims to catalyze future research on temporal understanding in AD. The benchmark and evaluation code are available at \href{this https URL}{Hugging Face} and \href{this https URL}{Github}, respectively.

**arXiv ID:** 2512.05277
</details>

<details>
<summary><strong>Confucius Code Agent: Scalable Agent Scaffolding for Real-World Codebases</strong> - Zhaodong Wang, Zhenting Qi, Sherman Wong, Nathan Hu, Samuel Lin, Jun Ge, Erwin Gao, Wenlin Chen, Yilun Du, Minlan Yu, Ying Zhang - [[pdf]](https://arxiv.org/pdf/2512.10398)</summary>

**Abstract:** Real-world software engineering tasks require coding agents that can operate over massive repositories, sustain long-horizon sessions, and reliably coordinate complex toolchains at test time. Existing research-grade agents offer transparency but struggle when scaled to real-world workloads, while proprietary systems achieve strong practical performance but provide limited extensibility, interpretability, and controllability. We introduce the Confucius Code Agent (CCA), a scalable software engineering agent that can operate at large-scale codebases. CCA is built on top of the Confucius SDK, an agent development platform structured around three complementary perspectives: Agent Experience (AX), User Experience (UX), and Developer Experience (DX). The SDK integrates a unified orchestrator with hierarchical working memory for long-context reasoning, a persistent note-taking system for cross-session continual learning, and a modular extension system for reliable tool use. In addition, we introduce a meta-agent that automates the synthesis, evaluation, and refinement of agent configurations through a build-test-improve loop, enabling rapid adaptation to new tasks, environments, and tool stacks. Instantiated with these mechanisms, CCA demonstrates strong performance on real-world software engineering tasks. On SWE-Bench-Pro, CCA reaches a Resolve@1 of 54.3%, exceeding prior research baselines and comparing favorably to commercial results, under identical repositories, model backend, and tool access. Together, the Confucius SDK and CCA form a general, extensible, and production-grade foundation for building effective and robust coding agents, bridging the gap between research prototypes and practical large-scale deployment.

**arXiv ID:** 2512.10398
</details>

<details>
<summary><strong>Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers</strong> - Youmin Ko, Sungjong Seo, Hyunjoon Kim - [[pdf]](https://arxiv.org/pdf/2512.10422)</summary>

**Abstract:** Since large language models (LLMs) have a tendency to generate factually inaccurate output, retrieval-augmented generation (RAG) has gained significant attention as a key means to mitigate this downside of harnessing only LLMs. However, existing RAG methods for simple and multi-hop question answering (QA) are still prone to incorrect retrievals and hallucinations. To address these limitations, we propose CoopRAG, a novel RAG framework for the question answering task in which a retriever and an LLM work cooperatively with each other by exchanging informative knowledge, and the earlier and later layers of the retriever model work cooperatively with each other to accurately rank the retrieved documents relevant to a given query. In this framework, we (i) unroll a question into sub-questions and a reasoning chain in which uncertain positions are masked, (ii) retrieve the documents relevant to the question augmented with the sub-questions and the reasoning chain, (iii) rerank the documents by contrasting layers of the retriever, and (iv) reconstruct the reasoning chain by filling the masked positions via the LLM. Our experiments demonstrate that CoopRAG consistently outperforms state-of-the-art QA methods on three multi-hop QA datasets as well as a simple QA dataset in terms of both the retrieval and QA performances. Our code is available.

**arXiv ID:** 2512.10422
</details>

<details>
<summary><strong>Audio MultiChallenge: A Multi-Turn Evaluation of Spoken Dialogue Systems on Natural Human Interaction</strong> - Advait Gosai, Tyler Vuong, Utkarsh Tyagi, Steven Li, Wenjia You, Miheer Bavare, Arda Uçar, Zhongwang Fang, Brian Jang, Bing Liu, Yunzhong He - [[pdf]](https://arxiv.org/pdf/2512.14865)</summary>

**Abstract:** End-to-end (E2E) spoken dialogue systems are increasingly replacing cascaded pipelines for voice-based human-AI interaction, processing raw audio directly without intermediate transcription. Existing benchmarks primarily evaluate these models on synthetic speech and single-turn tasks, leaving realistic multi-turn conversational ability underexplored. We introduce Audio MultiChallenge, an open-source benchmark to evaluate E2E spoken dialogue systems under natural multi-turn interaction patterns. Building on the text-based MultiChallenge framework, which evaluates Inference Memory, Instruction Retention, and Self Coherence, we introduce a new axis Voice Editing that tests robustness to mid-utterance speech repairs and backtracking. We further augment each axis to the audio modality, such as introducing Audio-Cue challenges for Inference Memory that require recalling ambient sounds and paralinguistic signals beyond semantic content. We curate 452 conversations from 47 speakers with 1,712 instance-specific rubrics through a hybrid audio-native agentic and human-in-the-loop pipeline that exposes model failures at scale while preserving natural disfluencies found in unscripted human speech. Our evaluation of proprietary and open-source models reveals that even frontier models struggle on our benchmark, with Gemini 3 Pro Preview (Thinking), our highest-performing model achieving a 54.65% pass rate. Error analysis shows that models fail most often on our new axes and that Self Coherence degrades with longer audio context. These failures reflect difficulty of tracking edits, audio cues, and long-range context in natural spoken dialogue. Audio MultiChallenge provides a reproducible testbed to quantify them and drive improvements in audio-native multi-turn interaction capability.

**arXiv ID:** 2512.14865
</details>

<details>
<summary><strong>NAP3D: NeRF Assisted 3D-3D Pose Alignment for Autonomous Vehicles</strong> - Gaurav Bansal - [[pdf]](https://arxiv.org/pdf/2512.15080)</summary>

**Abstract:** Accurate localization is essential for autonomous vehicles, yet sensor noise and drift over time can lead to significant pose estimation errors, particularly in long-horizon environments. A common strategy for correcting accumulated error is visual loop closure in SLAM, which adjusts the pose graph when the agent revisits previously mapped locations. These techniques typically rely on identifying visual mappings between the current view and previously observed scenes and often require fusing data from multiple sensors.
In contrast, this work introduces NeRF-Assisted 3D-3D Pose Alignment (NAP3D), a complementary approach that leverages 3D-3D correspondences between the agent's current depth image and a pre-trained Neural Radiance Field (NeRF). By directly aligning 3D points from the observed scene with synthesized points from the NeRF, NAP3D refines the estimated pose even from novel viewpoints, without relying on revisiting previously observed locations.
This robust 3D-3D formulation provides advantages over conventional 2D-3D localization methods while remaining comparable in accuracy and applicability. Experiments demonstrate that NAP3D achieves camera pose correction within 5 cm on a custom dataset, robustly outperforming a 2D-3D Perspective-N-Point baseline. On TUM RGB-D, NAP3D consistently improves 3D alignment RMSE by approximately 6 cm compared to this baseline given varying noise, despite PnP achieving lower raw rotation and translation parameter error in some regimes, highlighting NAP3D's improved geometric consistency in 3D space. By providing a lightweight, dataset-agnostic tool, NAP3D complements existing SLAM and localization pipelines when traditional loop closure is unavailable.

**arXiv ID:** 2512.15080
</details>

<details>
<summary><strong>EPSM: A Novel Metric to Evaluate the Safety of Environmental Perception in Autonomous Driving</strong> - Jörg Gamerdinger, Sven Teufel, Stephan Amann, Lukas Marc Listl, Oliver Bringmann - [[pdf]](https://arxiv.org/pdf/2512.15195)</summary>

**Abstract:** Extensive evaluation of perception systems is crucial for ensuring the safety of intelligent vehicles in complex driving scenarios. Conventional performance metrics such as precision, recall and the F1-score assess the overall detection accuracy, but they do not consider the safety-relevant aspects of perception. Consequently, perception systems that achieve high scores in these metrics may still cause misdetections that could lead to severe accidents. Therefore, it is important to evaluate not only the overall performance of perception systems, but also their safety. We therefore introduce a novel safety metric for jointly evaluating the most critical perception tasks, object and lane detection. Our proposed framework integrates a new, lightweight object safety metric that quantifies the potential risk associated with object detection errors, as well as an lane safety metric including the interdependence between both tasks that can occur in safety evaluation. The resulting combined safety score provides a unified, interpretable measure of perception safety performance. Using the DeepAccident dataset, we demonstrate that our approach identifies safety critical perception errors that conventional performance metrics fail to capture. Our findings emphasize the importance of safety-centric evaluation methods for perception systems in autonomous driving.

**arXiv ID:** 2512.15195
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>SoMe: A Realistic Benchmark for LLM-based Social Media Agents</strong> - Dizhan Xue, Jing Cui, Shengsheng Qian, Chuanrui Hu, Changsheng Xu - [[pdf]](https://arxiv.org/pdf/2512.14720)</summary>

**Abstract:** Intelligent agents powered by large language models (LLMs) have recently demonstrated impressive capabilities and gained increasing popularity on social media platforms. While LLM agents are reshaping the ecology of social media, there exists a current gap in conducting a comprehensive evaluation of their ability to comprehend media content, understand user behaviors, and make intricate decisions. To address this challenge, we introduce SoMe, a pioneering benchmark designed to evaluate social media agents equipped with various agent tools for accessing and analyzing social media data. SoMe comprises a diverse collection of 8 social media agent tasks, 9,164,284 posts, 6,591 user profiles, and 25,686 reports from various social media platforms and external websites, with 17,869 meticulously annotated task queries. Compared with the existing datasets and benchmarks for social media tasks, SoMe is the first to provide a versatile and realistic platform for LLM-based social media agents to handle diverse social media tasks. By extensive quantitative and qualitative analysis, we provide the first overview insight into the performance of mainstream agentic LLMs in realistic social media environments and identify several limitations. Our evaluation reveals that both the current closed-source and open-source LLMs cannot handle social media agent tasks satisfactorily. SoMe provides a challenging yet meaningful testbed for future social media agents. Our code and data are available at this https URL

**arXiv ID:** 2512.14720
</details>

<details>
<summary><strong>Imitation Learning for Multi-turn LM Agents via On-policy Expert Corrections</strong> - Niklas Lauffer, Xiang Deng, Srivatsa Kundurthy, Brad Kenstler, Jeff Da - [[pdf]](https://arxiv.org/pdf/2512.14895)</summary>

**Abstract:** A popular paradigm for training LM agents relies on imitation learning, fine-tuning on expert trajectories. However, we show that the off-policy nature of imitation learning for multi-turn LM agents suffers from the fundamental limitation known as covariate shift: as the student policy's behavior diverges from the expert's, it encounters states not present in the training data, reducing the effectiveness of fine-tuning. Taking inspiration from the classic DAgger algorithm, we propose a novel data generation methodology for addressing covariate shift for multi-turn LLM training. We introduce on-policy expert corrections (OECs), partially on-policy data generated by starting rollouts with a student model and then switching to an expert model part way through the trajectory. We explore the effectiveness of our data generation technique in the domain of software engineering (SWE) tasks, a multi-turn setting where LLM agents must interact with a development environment to fix software bugs. Our experiments compare OEC data against various other on-policy and imitation learning approaches on SWE agent problems and train models using a common rejection sampling (i.e., using environment reward) combined with supervised fine-tuning technique. Experiments find that OEC trajectories show a relative 14% and 13% improvement over traditional imitation learning in the 7b and 32b setting, respectively, on SWE-bench verified. Our results demonstrate the need for combining expert demonstrations with on-policy data for effective multi-turn LM agent training.

**arXiv ID:** 2512.14895
</details>

<details>
<summary><strong>From Trace to Line: LLM Agent for Real-World OSS Vulnerability Localization</strong> - Haoran Xi, Minghao Shao, Brendan Dolan-Gavitt, Muhammad Shafique, Ramesh Karri - [[pdf]](https://arxiv.org/pdf/2510.02389)</summary>

**Abstract:** Large language models show promise for vulnerability discovery, yet prevailing methods inspect code in isolation, struggle with long contexts, and focus on coarse function or file level detections which offers limited actionable guidance to engineers who need precise line-level localization and targeted patches in real-world software development. We present T2L-Agent (Trace-to-Line Agent), a project-level, end-to-end framework that plans its own analysis and progressively narrows scope from modules to exact vulnerable lines. T2L-Agent couples multi-round feedback with an Agentic Trace Analyzer (ATA) that fuses run-time evidence such as crash points, stack traces, and coverage deltas with AST-based code chunking, enabling iterative refinement beyond single pass predictions and translating symptoms into actionable, line-level diagnoses. To benchmark line-level vulnerability discovery, we introduce T2L-ARVO, a diverse, expert-verified 50-case benchmark spanning five crash families and real-world projects. T2L-ARVO is specifically designed to support both coarse-grained detection and fine-grained localization, enabling rigorous evaluation of systems that aim to move beyond file-level predictions. On T2L-ARVO, T2L-Agent achieves up to 58.0% detection and 54.8% line-level localization, substantially outperforming baselines. Together, the framework and benchmark push LLM-based vulnerability detection from coarse identification toward deployable, robust, precision diagnostics that reduce noise and accelerate patching in open-source software workflows.

**arXiv ID:** 2510.02389
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (12 papers)</h2></summary>

<details>
<summary><strong>AgroAskAI: A Multi-Agentic AI Framework for Supporting Smallholder Farmers' Enquiries Globally</strong> - Nadine Angela Cantonjos, Arpita Biswas - [[pdf]](https://arxiv.org/pdf/2512.14910)</summary>

**Abstract:** Agricultural regions in rural areas face damage from climate-related risks, including droughts, heavy rainfall, and shifting weather patterns. Prior research calls for adaptive risk-management solutions and decision-making strategies. To this end, artificial intelligence (AI), particularly agentic AI, offers a promising path forward. Agentic AI systems consist of autonomous, specialized agents capable of solving complex, dynamic tasks. While past systems have relied on single-agent models or have used multi-agent frameworks only for static functions, there is a growing need for architectures that support dynamic collaborative reasoning and context-aware outputs. To bridge this gap, we present AgroAskAI, a multi-agent reasoning system for climate adaptation decision support in agriculture, with a focus on vulnerable rural communities. AgroAskAI features a modular, role-specialized architecture that uses a chain-of-responsibility approach to coordinate autonomous agents, integrating real-time tools and datasets. The system has built-in governance mechanisms that mitigate hallucination and enable internal feedback for coherent, locally relevant strategies. The system also supports multilingual interactions, making it accessible to non-English-speaking farmers. Experiments on common agricultural queries related to climate adaptation show that, with additional tools and prompt refinement, AgroAskAI delivers more actionable, grounded, and inclusive outputs. Our experimental results highlight the potential of agentic AI for sustainable and accountable decision support in climate adaptation for agriculture.

**arXiv ID:** 2512.14910
</details>

<details>
<summary><strong>PyFi: Toward Pyramid-like Financial Image Understanding for VLMs via Adversarial Agents</strong> - Yuqun Zhang, Yuxuan Zhao, Sijia Chen - [[pdf]](https://arxiv.org/pdf/2512.14735)</summary>

**Abstract:** This paper proposes PyFi, a novel framework for pyramid-like financial image understanding that enables vision language models (VLMs) to reason through question chains in a progressive, simple-to-complex manner. At the core of PyFi is PyFi-600K, a dataset comprising 600K financial question-answer pairs organized into a reasoning pyramid: questions at the base require only basic perception, while those toward the apex demand increasing levels of capability in financial visual understanding and expertise. This data is scalable because it is synthesized without human annotations, using PyFi-adv, a multi-agent adversarial mechanism under the Monte Carlo Tree Search (MCTS) paradigm, in which, for each image, a challenger agent competes with a solver agent by generating question chains that progressively probe deeper capability levels in financial visual reasoning. Leveraging this dataset, we present fine-grained, hierarchical, and comprehensive evaluations of advanced VLMs in the financial domain. Moreover, fine-tuning Qwen2.5-VL-3B and Qwen2.5-VL-7B on the pyramid-structured question chains enables these models to answer complex financial questions by decomposing them into sub-questions with gradually increasing reasoning demands, yielding average accuracy improvements of 19.52% and 8.06%, respectively, on the dataset. All resources of code, dataset and models are available at: this https URL .

**arXiv ID:** 2512.14735
</details>

<details>
<summary><strong>MALCDF: A Distributed Multi-Agent LLM Framework for Real-Time Cyber</strong> - Arth Bhardwaj, Sia Godika, Yuvam Loonker - [[pdf]](https://arxiv.org/pdf/2512.14846)</summary>

**Abstract:** Traditional, centralized security tools often miss adaptive, multi-vector attacks. We present the Multi-Agent LLM Cyber Defense Framework (MALCDF), a practical setup where four large language model (LLM) agents-Detection, Intelligence, Response, and Analysis-work together in real time. Agents communicate over a Secure Communication Layer (SCL) with encrypted, ontology-aligned messages, and produce audit-friendly outputs (e.g., MITRE ATT&CK mappings).
For evaluation, we keep the test simple and consistent: all reported metrics come from the same 50-record live stream derived from the CICIDS2017 feature schema. CICIDS2017 is used for configuration (fields/schema) and to train a practical ML baseline. The ML-IDS baseline is a Lightweight Random Forest IDS (LRF-IDS) trained on a subset of CICIDS2017 and tested on the 50-record stream, with no overlap between training and test records.
In experiments, MALCDF reaches 90.0% detection accuracy, 85.7% F1-score, and 9.1% false-positive rate, with 6.8s average per-event latency. It outperforms the lightweight ML-IDS baseline and a single-LLM setup on accuracy while keeping end-to-end outputs consistent. Overall, this hands-on build suggests that coordinating simple LLM agents with secure, ontology-aligned messaging can improve practical, real-time cyber defense.

**arXiv ID:** 2512.14846
</details>

<details>
<summary><strong>QoS-Aware Hierarchical Reinforcement Learning for Joint Link Selection and Trajectory Optimization in SAGIN-Supported UAV Mobility Management</strong> - Jiayang Wan, Ke He, Yafei Wang, Fan Liu, Wenjin Wang, Shi Jin - [[pdf]](https://arxiv.org/pdf/2512.15119)</summary>

**Abstract:** Due to the significant variations in unmanned aerial vehicle (UAV) altitude and horizontal mobility, it becomes difficult for any single network to ensure continuous and reliable threedimensional coverage. Towards that end, the space-air-ground integrated network (SAGIN) has emerged as an essential architecture for enabling ubiquitous UAV connectivity. To address the pronounced disparities in coverage and signal characteristics across heterogeneous networks, this paper formulates UAV mobility management in SAGIN as a constrained multi-objective joint optimization problem. The formulation couples discrete link selection with continuous trajectory optimization. Building on this, we propose a two-level multi-agent hierarchical deep reinforcement learning (HDRL) framework that decomposes the problem into two alternately solvable subproblems. To map complex link selection decisions into a compact discrete action space, we conceive a double deep Q-network (DDQN) algorithm in the top-level, which achieves stable and high-quality policy learning through double Q-value estimation. To handle the continuous trajectory action space while satisfying quality of service (QoS) constraints, we integrate the maximum-entropy mechanism of the soft actor-critic (SAC) and employ a Lagrangian-based constrained SAC (CSAC) algorithm in the lower-level that dynamically adjusts the Lagrange multipliers to balance constraint satisfaction and policy optimization. Moreover, the proposed algorithm can be extended to multi-UAV scenarios under the centralized training and decentralized execution (CTDE) paradigm, which enables more generalizable policies. Simulation results demonstrate that the proposed scheme substantially outperforms existing benchmarks in throughput, link switching frequency and QoS satisfaction.

**arXiv ID:** 2512.15119
</details>

<details>
<summary><strong>Towards a Science of Scaling Agent Systems</strong> - Yubin Kim, Ken Gu, Chanwoo Park, Chunjong Park, Samuel Schmidgall, A. Ali Heydari, Yao Yan, Zhihan Zhang, Yuchen Zhuang, Mark Malhotra, Paul Pu Liang, Hae Won Park, Yuzhe Yang, Xuhai Xu, Yilun Du, Shwetak Patel, Tim Althoff, Daniel McDuff, Xin Liu - [[pdf]](https://arxiv.org/pdf/2512.08296)</summary>

**Abstract:** Agents, language model-based systems that are capable of reasoning, planning, and acting are becoming the dominant paradigm for real-world AI applications. Despite this widespread adoption, the principles that determine their performance remain underexplored. We address this by deriving quantitative scaling principles for agent systems. We first formalize a definition for agentic evaluation and characterize scaling laws as the interplay between agent quantity, coordination structure, model capability, and task properties. We evaluate this across four benchmarks: Finance-Agent, BrowseComp-Plus, PlanCraft, and Workbench. With five canonical agent architectures (Single-Agent and four Multi-Agent Systems: Independent, Centralized, Decentralized, Hybrid), instantiated across three LLM families, we perform a controlled evaluation spanning 180 configurations. We derive a predictive model using coordination metrics, that achieves cross-validated R^2=0.524, enabling prediction on unseen task domains. We identify three effects: (1) a tool-coordination trade-off: under fixed computational budgets, tool-heavy tasks suffer disproportionately from multi-agent overhead. (2) a capability saturation: coordination yields diminishing or negative returns once single-agent baselines exceed ~45%. (3) topology-dependent error amplification: independent agents amplify errors 17.2x, while centralized coordination contains this to 4.4x. Centralized coordination improves performance by 80.8% on parallelizable tasks, while decentralized coordination excels on web navigation (+9.2% vs. +0.2%). Yet for sequential reasoning tasks, every multi-agent variants degraded performance by 39-70%. The framework predicts the optimal coordination strategy for 87% of held-out configurations. Out-of-sample validation on GPT-5.2, achieves MAE=0.071 and confirms four of five scaling principles generalize to unseen frontier models.

**arXiv ID:** 2512.08296
</details>

<details>
<summary><strong>MedChat: A Multi-Agent Framework for Multimodal Diagnosis with Large Language Models</strong> - Philip R. Liu, Sparsh Bansal, Jimmy Dinh, Aditya Pawar, Ramani Satishkumar, Shail Desai, Neeraj Gupta, Xin Wang, Shu Hu - [[pdf]](https://arxiv.org/pdf/2506.07400)</summary>

**Abstract:** The integration of deep learning-based glaucoma detection with large language models (LLMs) presents an automated strategy to mitigate ophthalmologist shortages and improve clinical reporting efficiency. However, applying general LLMs to medical imaging remains challenging due to hallucinations, limited interpretability, and insufficient domain-specific medical knowledge, which can potentially reduce clinical accuracy. Although recent approaches combining imaging models with LLM reasoning have improved reporting, they typically rely on a single generalist agent, restricting their capacity to emulate the diverse and complex reasoning found in multidisciplinary medical teams. To address these limitations, we propose MedChat, a multi-agent diagnostic framework and platform that combines specialized vision models with multiple role-specific LLM agents, all coordinated by a director agent. This design enhances reliability, reduces hallucination risk, and enables interactive diagnostic reporting through an interface tailored for clinical review and educational use. Code available at this https URL.

**arXiv ID:** 2506.07400
</details>

<details>
<summary><strong>Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems</strong> - Chengxuan Xia, Qianye Wu, Sixuan Tian, Yilun Hao - [[pdf]](https://arxiv.org/pdf/2507.17061)</summary>

**Abstract:** Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems.

**arXiv ID:** 2507.17061
</details>

<details>
<summary><strong>Mapis: A Knowledge-Graph Grounded Multi-Agent Framework for Evidence-Based PCOS Diagnosis</strong> - Zanxiang He, Meng Li, Liyun Shi, Weiye Daia, Liming Nie - [[pdf]](https://arxiv.org/pdf/2512.15398)</summary>

**Abstract:** Polycystic Ovary Syndrome (PCOS) constitutes a significant public health issue affecting 10% of reproductive-aged women, highlighting the critical importance of developing effective diagnostic tools. Previous machine learning and deep learning detection tools are constrained by their reliance on large-scale labeled data and an lack of interpretability. Although multi-agent systems have demonstrated robust capabilities, the potential of such systems for PCOS detection remains largely unexplored. Existing medical multi-agent frameworks are predominantly designed for general medical tasks, suffering from insufficient domain integration and a lack of specific domain knowledge. To address these challenges, we propose Mapis, the first knowledge-grounded multi-agent framework explicitly designed for guideline-based PCOS diagnosis. Specifically, it built upon the 2023 International Guideline into a structured collaborative workflow that simulates the clinical diagnostic process. It decouples complex diagnostic tasks across specialized agents: a gynecological endocrine agent and a radiology agent collaborative to verify inclusion criteria, while an exclusion agent strictly rules out other causes. Furthermore, we construct a comprehensive PCOS knowledge graph to ensure verifiable, evidence-based decision-making. Extensive experiments on public benchmarks and specialized clinical datasets, benchmarking against nine diverse baselines, demonstrate that Mapis significantly outperforms competitive methods. On the clinical dataset, it surpasses traditional machine learning models by 13.56%, single-agent by 6.55%, and previous medical multi-agent systems by 7.05% in Accuracy.

**arXiv ID:** 2512.15398
</details>

<details>
<summary><strong>SGEMAS: A Self-Growing Ephemeral Multi-Agent System for Unsupervised Online Anomaly Detection via Entropic Homeostasis</strong> - Mustapha Hamdi - [[pdf]](https://arxiv.org/pdf/2512.14708)</summary>

**Abstract:** Current deep learning approaches for physiological signal monitoring suffer from static topologies and constant energy consumption. We introduce SGEMAS (Self-Growing Ephemeral Multi-Agent System), a bio-inspired architecture that treats intelligence as a dynamic thermodynamic process. By coupling a structural plasticity mechanism (agent birth death) to a variational free energy objective, the system naturally evolves to minimize prediction error with extreme sparsity. An ablation study on the MIT-BIH Arrhythmia Database reveals that adding a multi-scale instability index to the agent dynamics significantly improves performance. In a challenging inter-patient, zero-shot setting, the final SGEMAS v3.3 model achieves a mean AUC of 0.570 +- 0.070, outperforming both its simpler variants and a standard autoencoder baseline. This result validates that a physics-based, energy-constrained model can achieve robust unsupervised anomaly detection, offering a promising direction for efficient biomedical AI.

**arXiv ID:** 2512.14708
</details>

<details>
<summary><strong>Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs</strong> - Yujie Zhao, Lanxiang Hu, Yang Wang, Minmin Hou, Hao Zhang, Ke Ding, Jishen Zhao - [[pdf]](https://arxiv.org/pdf/2510.11062)</summary>

**Abstract:** Multi-agent systems (MAS) and reinforcement learning (RL) are widely used to enhance the agentic capabilities of large language models (LLMs). MAS improves task performance through role-based orchestration, while RL uses environmental rewards to learn stronger policies, such as GRPO-style optimization. However, applying on-policy RL to MAS remains underexplored and presents unique challenges. Algorithmically, standard GRPO grouping assumptions break down because prompts vary by role and by turn. System-wise, the training stack must support MAS-workflow rollouts and on-policy updates for both single-policy and multi-policy models.
We propose AT-GRPO, which includes (i) an agent- and turn-wise grouped RL algorithm tailored to MAS and (ii) a training system that supports both single- and multi-policy regimes. Across game, planning, coding, and math tasks, AT-GRPO delivers substantial gains. On long-horizon planning, it increases accuracy from a 14.0 to 47.0 percent single-agent RL baseline to 96.0 to 99.5 percent. It also improves reasoning performance, with average gains of 3.87 to 7.62 percent on coding tasks and 9.0 to 17.93 percent on math. Code and environments are available at: this https URL.

**arXiv ID:** 2510.11062
</details>

<details>
<summary><strong>Multi-Agent Pointer Transformer: Seq-to-Seq Reinforcement Learning for Multi-Vehicle Dynamic Pickup-Delivery Problems</strong> - Zengyu Zou, Jingyuan Wang, Yixuan Huang, Junjie Wu - [[pdf]](https://arxiv.org/pdf/2511.17435)</summary>

**Abstract:** This paper addresses the cooperative Multi-Vehicle Dynamic Pickup and Delivery Problem with Stochastic Requests (MVDPDPSR) and proposes an end-to-end centralized decision-making framework based on sequence-to-sequence, named Multi-Agent Pointer Transformer (MAPT). MVDPDPSR is an extension of the vehicle routing problem and a spatio-temporal system optimization problem, widely applied in scenarios such as on-demand delivery. Classical operations research methods face bottlenecks in computational complexity and time efficiency when handling large-scale dynamic problems. Although existing reinforcement learning methods have achieved some progress, they still encounter several challenges: 1) Independent decoding across multiple vehicles fails to model joint action distributions; 2) The feature extraction network struggles to capture inter-entity relationships; 3) The joint action space is exponentially large. To address these issues, we designed the MAPT framework, which employs a Transformer Encoder to extract entity representations, combines a Transformer Decoder with a Pointer Network to generate joint action sequences in an AutoRegressive manner, and introduces a Relation-Aware Attention module to capture inter-entity relationships. Additionally, we guide the model's decision-making using informative priors to facilitate effective exploration. Experiments on 8 datasets demonstrate that MAPT significantly outperforms existing baseline methods in terms of performance and exhibits substantial computational time advantages compared to classical operations research methods.

**arXiv ID:** 2511.17435
</details>

<details>
<summary><strong>A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks</strong> - S M Asif Hossain, Ruksat Khan Shayoni, Mohd Ruhul Ameen, Akif Islam, M. F. Mridha, Jungpil Shin - [[pdf]](https://arxiv.org/pdf/2509.14285)</summary>

**Abstract:** Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.

**arXiv ID:** 2509.14285
</details>

</details>

<details open>
<summary><h2>Other Agent Research (5 papers)</h2></summary>

<details>
<summary><strong>Zero-Knowledge Audit for Internet of Agents: Privacy-Preserving Communication Verification with Model Context Protocol</strong> - Guanlin Jing, Huayi Qi - [[pdf]](https://arxiv.org/pdf/2512.14737)</summary>

**Abstract:** Existing agent communication frameworks face critical limitations in providing verifiable audit trails without compromising the privacy and confidentiality of agent interactions. The protection of agent communication privacy while ensuring auditability emerges as a fundamental challenge for applications requiring accurate billing, compliance verification, and accountability in regulated environments.
We introduce a framework for auditing agent communications that keeps messages private while still checking they follow expected rules. It pairs zero-knowledge proofs with the existing Model Context Protocol (MCP) so messages can be verified without revealing their contents. The approach runs in lightweight networks, stays compatible with standard MCP exchanges, and adds asynchronous audit verification to confirm format and general message types without exposing specifics.
The framework enables mutual audits between agents: one side can check communication content and quality while the other verifies usage metrics, all without revealing sensitive information. We formalize security goals and show that zk-MCP provides data authenticity and communication privacy, achieving efficient verification with negligible latency overhead. We fully implement the framework, including Circom-based zero-knowledge proof generation and an audit protocol integrated with MCP's bidirectional channel, and, to our knowledge, this is the first privacy-preserving audit system for agent communications that offers verifiable mutual auditing without exposing message content or compromising agent privacy.

**arXiv ID:** 2512.14737
</details>

<details>
<summary><strong>Imitation Game: Reproducing Deep Learning Bugs Leveraging an Intelligent Agent</strong> - Mehil B Shah, Mohammad Masudur Rahman, Foutse Khomh - [[pdf]](https://arxiv.org/pdf/2512.14990)</summary>

**Abstract:** Despite their wide adoption in various domains (e.g., healthcare, finance, software engineering), Deep Learning (DL)-based applications suffer from many bugs, failures, and vulnerabilities. Reproducing these bugs is essential for their resolution, but it is extremely challenging due to the inherent nondeterminism of DL models and their tight coupling with hardware and software environments. According to recent studies, only about 3% of DL bugs can be reliably reproduced using manual approaches. To address these challenges, we present RepGen, a novel, automated, and intelligent approach for reproducing deep learning bugs. RepGen constructs a learning-enhanced context from a project, develops a comprehensive plan for bug reproduction, employs an iterative generate-validate-refine mechanism, and thus generates such code using an LLM that reproduces the bug at hand. We evaluate RepGen on 106 real-world deep learning bugs and achieve a reproduction rate of 80.19%, a 19.81% improvement over the state-of-the-art measure. A developer study involving 27 participants shows that RepGen improves the success rate of DL bug reproduction by 23.35%, reduces the time to reproduce by 56.8%, and lowers participants' cognitive load.

**arXiv ID:** 2512.14990
</details>

<details>
<summary><strong>HERO: Hierarchical Traversable 3D Scene Graphs for Embodied Navigation Among Movable Obstacles</strong> - Yunheng Wang, Yixiao Feng, Yuetong Fang, Shuning Zhang, Tan Jing, Jian Li, Xiangrui Jiang, Renjing Xu - [[pdf]](https://arxiv.org/pdf/2512.15047)</summary>

**Abstract:** 3D Scene Graphs (3DSGs) constitute a powerful representation of the physical world, distinguished by their abilities to explicitly model the complex spatial, semantic, and functional relationships between entities, rendering a foundational understanding that enables agents to interact intelligently with their environment and execute versatile behaviors. Embodied navigation, as a crucial component of such capabilities, leverages the compact and expressive nature of 3DSGs to enable long-horizon reasoning and planning in complex, large-scale environments. However, prior works rely on a static-world assumption, defining traversable space solely based on static spatial layouts and thereby treating interactable obstacles as non-traversable. This fundamental limitation severely undermines their effectiveness in real-world scenarios, leading to limited reachability, low efficiency, and inferior extensibility. To address these issues, we propose HERO, a novel framework for constructing Hierarchical Traversable 3DSGs, that redefines traversability by modeling operable obstacles as pathways, capturing their physical interactivity, functional semantics, and the scene's relational hierarchy. The results show that, relative to its baseline, HERO reduces PL by 35.1% in partially obstructed environments and increases SR by 79.4% in fully obstructed ones, demonstrating substantially higher efficiency and reachability.

**arXiv ID:** 2512.15047
</details>

<details>
<summary><strong>Large Language Model-Based Intelligent Antenna Design System</strong> - Tao Wu, Kexue Fu, Qiang Hua, Xinxin Liu, Bo Liu - [[pdf]](https://arxiv.org/pdf/2504.18271)</summary>

**Abstract:** Antenna simulation typically involves modeling and optimization, which are time-consuming and labor-intensive, slowing down antenna analysis and design. This paper presents a prototype of a large language model (LLM)-based antenna design system (LADS) to assist in antenna simulation. LADS generates antenna models with textual descriptions and images extracted from academic papers, patents, and technical reports (either one or multiple), and it interacts with engineers to iteratively refine the designs. After that, LADS configures and runs an optimizer to meet the design specifications. The effectiveness of LADS is demonstrated by a monopole slotted antenna generated from images and descriptions from the literature. To improve gain stability across the 3.1-10.6 GHz ultra-wide band, LADS modifies the cross-slot into an H-slot and changes substrate material, followed by parameter optimization. As a result, the gain variation is reduced while maintaining the same gain level. The LLM-enabled antenna modeling (LEAM) is available at: this https URL.

**arXiv ID:** 2504.18271
</details>

<details>
<summary><strong>GuangMing-Explorer: A Four-Legged Robot Platform for Autonomous Exploration in General Environments</strong> - Kai Zhang, Shoubin Chen, Dong Li, Baiyang Zhang, Tao Huang, Zehao Wu, Jiasheng Chen, Bo Zhang - [[pdf]](https://arxiv.org/pdf/2512.15309)</summary>

**Abstract:** Autonomous exploration is a fundamental capability that tightly integrates perception, planning, control, and motion execution. It plays a critical role in a wide range of applications, including indoor target search, mapping of extreme environments, resource exploration, etc. Despite significant progress in individual components, a holistic and practical description of a completely autonomous exploration system, encompassing both hardware and software, remains scarce. In this paper, we present GuangMing-Explorer, a fully integrated autonomous exploration platform designed for robust operation across diverse environments. We provide a comprehensive overview of the system architecture, including hardware design, software stack, algorithm deployment, and experimental configuration. Extensive real-world experiments demonstrate the platform's effectiveness and efficiency in executing autonomous exploration tasks, highlighting its potential for practical deployment in complex and unstructured environments.

**arXiv ID:** 2512.15309
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (18 papers)</h2></summary>

<details>
<summary><strong>GR-Agent: Adaptive Graph Reasoning Agent under Incomplete Knowledge</strong> - Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang, Hongkuan Zhou, Jiaoyan Chen, Steffen Staab, Yuan He, Evgeny Kharlamov - [[pdf]](https://arxiv.org/pdf/2512.14766)</summary>

**Abstract:** Large language models (LLMs) achieve strong results on knowledge graph question answering (KGQA), but most benchmarks assume complete knowledge graphs (KGs) where direct supporting triples exist. This reduces evaluation to shallow retrieval and overlooks the reality of incomplete KGs, where many facts are missing and answers must be inferred from existing facts. We bridge this gap by proposing a methodology for constructing benchmarks under KG incompleteness, which removes direct supporting triples while ensuring that alternative reasoning paths required to infer the answer remain. Experiments on benchmarks constructed using our methodology show that existing methods suffer consistent performance degradation under incompleteness, highlighting their limited reasoning ability. To overcome this limitation, we present the Adaptive Graph Reasoning Agent (GR-Agent). It first constructs an interactive environment from the KG, and then formalizes KGQA as agent environment interaction within this environment. GR-Agent operates over an action space comprising graph reasoning tools and maintains a memory of potential supporting reasoning evidence, including relevant relations and reasoning paths. Extensive experiments demonstrate that GR-Agent outperforms non-training baselines and performs comparably to training-based methods under both complete and incomplete settings.

**arXiv ID:** 2512.14766
</details>

<details>
<summary><strong>LADY: Linear Attention for Autonomous Driving Efficiency without Transformers</strong> - Jihao Huang, Xi Xia, Zhiyuan Li, Tianle Liu, Jingke Wang, Junbo Chen, Tengju Ye - [[pdf]](https://arxiv.org/pdf/2512.15038)</summary>

**Abstract:** End-to-end paradigms have demonstrated great potential for autonomous driving. Additionally, most existing methods are built upon Transformer architectures. However, transformers incur a quadratic attention cost, limiting their ability to model long spatial and temporal sequences-particularly on resource-constrained edge platforms. As autonomous driving inherently demands efficient temporal modeling, this challenge severely limits their deployment and real-time performance. Recently, linear attention mechanisms have gained increasing attention due to their superior spatiotemporal complexity. However, existing linear attention architectures are limited to self-attention, lacking support for cross-modal and cross-temporal interactions-both crucial for autonomous driving. In this work, we propose LADY, the first fully linear attention-based generative model for end-to-end autonomous driving. LADY enables fusion of long-range temporal context at inference with constant computational and memory costs, regardless of the history length of camera and LiDAR features. Additionally, we introduce a lightweight linear cross-attention mechanism that enables effective cross-modal information exchange. Experiments on the NAVSIM and Bench2Drive benchmarks demonstrate that LADY achieves state-of-the-art performance with constant-time and memory complexity, offering improved planning performance and significantly reduced computational cost. Additionally, the model has been deployed and validated on edge devices, demonstrating its practicality in resource-limited scenarios.

**arXiv ID:** 2512.15038
</details>

<details>
<summary><strong>Agentic AI for Integrated Sensing and Communication: Analysis, Framework, and Case Study</strong> - Wenwen Xie, Geng Sun, Ruichen Zhang, Xuejie Liu, Yinqiu Liu, Jiacheng Wang, Dusit Niyato, Ping Zhang - [[pdf]](https://arxiv.org/pdf/2512.15044)</summary>

**Abstract:** Integrated sensing and communication (ISAC) has emerged as a key development direction in the sixth-generation (6G) era, which provides essential support for the collaborative sensing and communication of future intelligent networks. However, as wireless environments become increasingly dynamic and complex, ISAC systems require more intelligent processing and more autonomous operation to maintain efficiency and adaptability. Meanwhile, agentic artificial intelligence (AI) offers a feasible solution to address these challenges by enabling continuous perception-reasoning-action loops in dynamic environments to support intelligent, autonomous, and efficient operation for ISAC systems. As such, we delve into the application value and prospects of agentic AI in ISAC systems in this work. Firstly, we provide a comprehensive review of agentic AI and ISAC systems to demonstrate their key characteristics. Secondly, we show several common optimization approaches for ISAC systems and highlight the significant advantages of generative artificial intelligence (GenAI)-based agentic AI. Thirdly, we propose a novel agentic ISAC framework and prensent a case study to verify its superiority in optimizing ISAC performance. Finally, we clarify future research directions for agentic AI-based ISAC systems.

**arXiv ID:** 2512.15044
</details>

<details>
<summary><strong>Graph Contextual Reinforcement Learning for Efficient Directed Controller Synthesis</strong> - Toshihide Ubukata, Enhong Mu, Takuto Yamauchi, Mingyue Zhang, Jialong Li, Kenji Tei - [[pdf]](https://arxiv.org/pdf/2512.15295)</summary>

**Abstract:** Controller synthesis is a formal method approach for automatically generating Labeled Transition System (LTS) controllers that satisfy specified properties. The efficiency of the synthesis process, however, is critically dependent on exploration policies. These policies often rely on fixed rules or strategies learned through reinforcement learning (RL) that consider only a limited set of current features. To address this limitation, this paper introduces GCRL, an approach that enhances RL-based methods by integrating Graph Neural Networks (GNNs). GCRL encodes the history of LTS exploration into a graph structure, allowing it to capture a broader, non-current-based context. In a comparative experiment against state-of-the-art methods, GCRL exhibited superior learning efficiency and generalization across four out of five benchmark domains, except one particular domain characterized by high symmetry and strictly local interactions.

**arXiv ID:** 2512.15295
</details>

<details>
<summary><strong>Quantum Decision Transformers (QDT): Synergistic Entanglement and Interference for Offline Reinforcement Learning</strong> - Abraham Itzhak Weinberg - [[pdf]](https://arxiv.org/pdf/2512.14726)</summary>

**Abstract:** Offline reinforcement learning enables policy learning from pre-collected datasets without environment interaction, but existing Decision Transformer (DT) architectures struggle with long-horizon credit assignment and complex state-action dependencies. We introduce the Quantum Decision Transformer (QDT), a novel architecture incorporating quantum-inspired computational mechanisms to address these challenges. Our approach integrates two core components: Quantum-Inspired Attention with entanglement operations that capture non-local feature correlations, and Quantum Feedforward Networks with multi-path processing and learnable interference for adaptive computation. Through comprehensive experiments on continuous control tasks, we demonstrate over 2,000\% performance improvement compared to standard DTs, with superior generalization across varying data qualities. Critically, our ablation studies reveal strong synergistic effects between quantum-inspired components: neither alone achieves competitive performance, yet their combination produces dramatic improvements far exceeding individual contributions. This synergy demonstrates that effective quantum-inspired architecture design requires holistic co-design of interdependent mechanisms rather than modular component adoption. Our analysis identifies three key computational advantages: enhanced credit assignment through non-local correlations, implicit ensemble behavior via parallel processing, and adaptive resource allocation through learnable interference. These findings establish quantum-inspired design principles as a promising direction for advancing transformer architectures in sequential decision-making, with implications extending beyond reinforcement learning to neural architecture design more broadly.

**arXiv ID:** 2512.14726
</details>

<details>
<summary><strong>Spectral Representation-based Reinforcement Learning</strong> - Chenxiao Gao, Haotian Sun, Na Li, Dale Schuurmans, Bo Dai - [[pdf]](https://arxiv.org/pdf/2512.15036)</summary>

**Abstract:** In real-world applications with large state and action spaces, reinforcement learning (RL) typically employs function approximations to represent core components like the policies, value functions, and dynamics models. Although powerful approximations such as neural networks offer great expressiveness, they often present theoretical ambiguities, suffer from optimization instability and exploration difficulty, and incur substantial computational costs in practice. In this paper, we introduce the perspective of spectral representations as a solution to address these difficulties in RL. Stemming from the spectral decomposition of the transition operator, this framework yields an effective abstraction of the system dynamics for subsequent policy optimization while also providing a clear theoretical characterization. We reveal how to construct spectral representations for transition operators that possess latent variable structures or energy-based structures, which implies different learning methods to extract spectral representations from data. Notably, each of these learning methods realizes an effective RL algorithm under this framework. We also provably extend this spectral view to partially observable MDPs. Finally, we validate these algorithms on over 20 challenging tasks from the DeepMind Control Suite, where they achieve performances comparable or superior to current state-of-the-art model-free and model-based baselines.

**arXiv ID:** 2512.15036
</details>

<details>
<summary><strong>Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning</strong> - Yiliu Sun, Zicheng Zhao, Yang Wei, Yanfang Zhang, Chen Gong - [[pdf]](https://arxiv.org/pdf/2512.15274)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) significantly enhances the reasoning capability of Large Language Models (LLMs). Current RLVR approaches typically conduct training across all generated tokens, but neglect to explore which tokens (e.g., prefix tokens) actually contribute to reasoning. This uniform training strategy spends substantial effort on optimizing low-return tokens, which in turn impedes the potential improvement from high-return tokens and reduces overall training effectiveness. To address this issue, we propose a novel RLVR approach called Progressive Prefix-token Policy Optimization (PPPO), which highlights the significance of the prefix segment of generated outputs. Specifically, inspired by the well-established human thinking theory of Path Dependence, where early-stage thoughts substantially constrain subsequent thinking trajectory, we identify an analogous phenomenon in LLM reasoning termed Beginning Lock-in Effect (BLE). PPPO leverages this finding by focusing its optimization objective on the prefix reasoning process of LLMs. This targeted optimization strategy can positively influence subsequent reasoning processes, and ultimately improve final results. To improve the learning effectiveness of LLMs on how to start reasoning with high quality, PPPO introduces two training strategies: (a) Progressive Prefix Retention, which shapes a progressive learning process by increasing the proportion of retained prefix tokens during training; (b) Continuation Accumulated Reward, which mitigates reward bias by sampling multiple continuations for one prefix token sequence, and accumulating their scores as the reward signal. Extensive experimental results on various reasoning tasks demonstrate that our proposed PPPO outperforms representative RLVR methods, with the accuracy improvements of 18.02% on only 26.17% training tokens.

**arXiv ID:** 2512.15274
</details>

<details>
<summary><strong>Can LLMs Guide Their Own Exploration? Gradient-Guided Reinforcement Learning for LLM Reasoning</strong> - Zhenwen Liang, Sidi Lu, Wenhao Yu, Kishan Panaganti, Yujun Zhou, Haitao Mi, Dong Yu - [[pdf]](https://arxiv.org/pdf/2512.15687)</summary>

**Abstract:** Reinforcement learning has become essential for strengthening the reasoning abilities of large language models, yet current exploration mechanisms remain fundamentally misaligned with how these models actually learn. Entropy bonuses and external semantic comparators encourage surface level variation but offer no guarantee that sampled trajectories differ in the update directions that shape optimization. We propose G2RL, a gradient guided reinforcement learning framework in which exploration is driven not by external heuristics but by the model own first order update geometry. For each response, G2RL constructs a sequence level feature from the model final layer sensitivity, obtainable at negligible cost from a standard forward pass, and measures how each trajectory would reshape the policy by comparing these features within a sampled group. Trajectories that introduce novel gradient directions receive a bounded multiplicative reward scaler, while redundant or off manifold updates are deemphasized, yielding a self referential exploration signal that is naturally aligned with PPO style stability and KL control. Across math and general reasoning benchmarks (MATH500, AMC, AIME24, AIME25, GPQA, MMLUpro) on Qwen3 base 1.7B and 4B models, G2RL consistently improves pass@1, maj@16, and pass@k over entropy based GRPO and external embedding methods. Analyzing the induced geometry, we find that G2RL expands exploration into substantially more orthogonal and often opposing gradient directions while maintaining semantic coherence, revealing that a policy own update space provides a far more faithful and effective basis for guiding exploration in large language model reinforcement learning.

**arXiv ID:** 2512.15687
</details>

<details>
<summary><strong>Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination</strong> - Mingqi Wu, Zhihao Zhang, Qiaole Dong, Zhiheng Xi, Jun Zhao, Senjie Jin, Xiaoran Fan, Yuhao Zhou, Huijie Lv, Ming Zhang, Yanwei Fu, Qin Liu, Songyang Zhang, Qi Zhang - [[pdf]](https://arxiv.org/pdf/2507.10532)</summary>

**Abstract:** Reasoning in large language models has long been a central research focus, and recent studies employing reinforcement learning (RL) have introduced diverse methods that yield substantial performance gains with minimal or even no external supervision. Surprisingly, some studies even suggest that random or incorrect reward signals can enhance performance. However, these breakthroughs are predominantly observed for the mathematically strong Qwen2.5 series on benchmarks such as MATH-500, AMC, and AIME, and seldom transfer to models like Llama, which warrants a more in-depth investigation. In this work, our empirical analysis reveals that pre-training on massive web-scale corpora leaves Qwen2.5 susceptible to data contamination in widely used benchmarks. Consequently, conclusions derived from contaminated benchmarks on Qwen2.5 series may be unreliable. To obtain trustworthy evaluation results, we introduce a generator that creates fully clean arithmetic problems of arbitrary length and difficulty, dubbed RandomCalculation. Using this leakage-free dataset, we show that only accurate reward signals yield steady improvements that surpass the base model's performance boundary in mathematical reasoning, whereas random or incorrect rewards do not. Moreover, we conduct more fine-grained analyses to elucidate the factors underlying the different performance observed on the MATH-500 and RandomCalculation benchmarks. Consequently, we recommend that future studies evaluate models on uncontaminated benchmarks and, when feasible, test various model series to ensure trustworthy conclusions about RL and related methods.

**arXiv ID:** 2507.10532
</details>

<details>
<summary><strong>Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning</strong> - Zelin Tan, Hejia Geng, Xiaohang Yu, Mulei Zhang, Guancheng Wan, Yifan Zhou, Qiang He, Xiangyuan Xue, Heng Zhou, Yutao Fan, Zhongzhi Li, Zaibin Zhang, Guibin Zhang, Chen Zhang, Zhenfei Yin, Lei Bai - [[pdf]](https://arxiv.org/pdf/2509.25300)</summary>

**Abstract:** While scaling laws for large language models (LLMs) during pre-training have been extensively studied, their behavior under reinforcement learning (RL) post-training remains largely unexplored. This paper presents a systematic empirical investigation of scaling behaviors in RL-based post-training, with a particular focus on mathematical reasoning. Based on a set of experiments across the full Qwen2.5 dense model series (0.5B to 72B), we characterize how model scale, data volume, and computational budget interact to shape performance. Our analysis leads to four key findings: this http URL models consistently exhibit superior learning efficiency on both compute and data metrics. this http URL relationship between test loss, compute, and data can be modeled by a predictive power-law which is robust across both base and instruction-tuned models. this http URL larger models exhibit higher learning efficiency, the analytical learning efficiency term k(N) in the power-law reveals a latent saturation trend in learning efficiency as model size continues to increase. this http URL data-constrained regimes, repeated reuse of high-quality data proves highly effective, as final performance is primarily governed by the total number of optimization steps rather than the uniqueness of samples. Collectively, these results provide a principled foundation and practical guidelines for efficiently scaling the reasoning capabilities of LLMs through RL post-training.

**arXiv ID:** 2509.25300
</details>

<details>
<summary><strong>Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning</strong> - Weiqin Wang, Yile Wang, Kehao Chen, Hui Huang - [[pdf]](https://arxiv.org/pdf/2512.15146)</summary>

**Abstract:** Test-time reinforcement learning mitigates the reliance on annotated data by using majority voting results as pseudo-labels, emerging as a complementary direction to reinforcement learning with verifiable rewards (RLVR) for improving reasoning ability of large language models (LLMs). However, this voting strategy often induces confirmation bias and suffers from sparse rewards, limiting the overall performance. In this work, we propose subgroup-specific step-wise confidence-weighted pseudo-label estimation (SCOPE), a framework integrating model confidence and dynamic subgroup partitioning to address these issues. Specifically, SCOPE integrates the proposed step-wise confidence into pseudo label deduction, prioritizing high-quality reasoning paths over simple frequency count. Furthermore, it dynamically partitions the candidate outputs pool into independent subgroups by balancing reasoning quality against exploration diversity. By deriving local consensus via repeat sampling for each sub group, SCOPE provides diverse supervision targets to encourage broader exploration. We conduct experiments across various models and benchmarks, experimental results show that SCOPE consistently outperforms recent baselines. Notably, SCOPE achieving relative improvements of 13.1\% on challenging AIME 2025 and 8.1\% on AMC. The code is released at \href{this https URL}{this https URL}.

**arXiv ID:** 2512.15146
</details>

<details>
<summary><strong>A Bayesian latent class reinforcement learning framework to capture adaptive, feedback-driven travel behaviour</strong> - Georges Sfeir, Stephane Hess, Thomas O. Hancock, Filipe Rodrigues, Jamal Amani Rad, Michiel Bliemer, Matthew Beck, Fayyaz Khan - [[pdf]](https://arxiv.org/pdf/2512.14713)</summary>

**Abstract:** Many travel decisions involve a degree of experience formation, where individuals learn their preferences over time. At the same time, there is extensive scope for heterogeneity across individual travellers, both in their underlying preferences and in how these evolve. The present paper puts forward a Latent Class Reinforcement Learning (LCRL) model that allows analysts to capture both of these phenomena. We apply the model to a driving simulator dataset and estimate the parameters through Variational Bayes. We identify three distinct classes of individuals that differ markedly in how they adapt their preferences: the first displays context-dependent preferences with context-specific exploitative tendencies; the second follows a persistent exploitative strategy regardless of context; and the third engages in an exploratory strategy combined with context-specific preferences.

**arXiv ID:** 2512.14713
</details>

<details>
<summary><strong>EUBRL: Epistemic Uncertainty Directed Bayesian Reinforcement Learning</strong> - Jianfei Ma, Wee Sun Lee - [[pdf]](https://arxiv.org/pdf/2512.15405)</summary>

**Abstract:** At the boundary between the known and the unknown, an agent inevitably confronts the dilemma of whether to explore or to exploit. Epistemic uncertainty reflects such boundaries, representing systematic uncertainty due to limited knowledge. In this paper, we propose a Bayesian reinforcement learning (RL) algorithm, $\texttt{EUBRL}$, which leverages epistemic guidance to achieve principled exploration. This guidance adaptively reduces per-step regret arising from estimation errors. We establish nearly minimax-optimal regret and sample complexity guarantees for a class of sufficiently expressive priors in infinite-horizon discounted MDPs. Empirically, we evaluate $\texttt{EUBRL}$ on tasks characterized by sparse rewards, long horizons, and stochasticity. Results demonstrate that $\texttt{EUBRL}$ achieves superior sample efficiency, scalability, and consistency.

**arXiv ID:** 2512.15405
</details>

<details>
<summary><strong>Autonomous Pressure Control in MuVacAS via Deep Reinforcement Learning and Deep Learning Surrogate Models</strong> - Guillermo Rodriguez-Llorente, Galo Gallardo, Rodrigo Morant Navascués, Nikita Khvatkin Petrovsky, Anderson Sabogal, Roberto Gómez-Espinosa Martín - [[pdf]](https://arxiv.org/pdf/2512.15521)</summary>

**Abstract:** The development of nuclear fusion requires materials that can withstand extreme conditions. The IFMIF-DONES facility, a high-power particle accelerator, is being designed to qualify these materials. A critical testbed for its development is the MuVacAS prototype, which replicates the final segment of the accelerator beamline. Precise regulation of argon gas pressure within its ultra-high vacuum chamber is vital for this task. This work presents a fully data-driven approach for autonomous pressure control. A Deep Learning Surrogate Model, trained on real operational data, emulates the dynamics of the argon injection system. This high-fidelity digital twin then serves as a fast-simulation environment to train a Deep Reinforcement Learning agent. The results demonstrate that the agent successfully learns a control policy that maintains gas pressure within strict operational limits despite dynamic disturbances. This approach marks a significant step toward the intelligent, autonomous control systems required for the demanding next-generation particle accelerator facilities.

**arXiv ID:** 2512.15521
</details>

<details>
<summary><strong>How Reinforcement Learning After Next-Token Prediction Facilitates Learning</strong> - Nikolaos Tsilivis, Eran Malach, Karen Ullrich, Julia Kempe - [[pdf]](https://arxiv.org/pdf/2510.11495)</summary>

**Abstract:** Recent advances in reasoning domains with neural networks have primarily been enabled by a training recipe that optimizes Large Language Models, previously trained to predict the next-token in a sequence, with reinforcement learning algorithms. We introduce a framework to study the success of this paradigm, and we theoretically expose the optimization mechanisms by which reinforcement learning improves over next-token prediction in this setting. We study learning from mixture distributions of short and long ``chain-of-thought'' sequences encoding a single task. In particular, when the task consists of predicting the parity of $d$ bits and long sequences are rare, we show how reinforcement learning after next-token prediction enables autoregressive transformers to generalize, whereas mere next-token prediction requires extreme statistical or computational resources to do so. We further explain how reinforcement learning leverages increased test-time computation, manifested in longer responses, to facilitate this learning process. In a simplified setting, we theoretically prove that autoregressive linear models following this training recipe can efficiently learn to predict the parity of $d$ bits as long as the proportion of long demonstrations in the data mix is not exponentially small in the input dimension $d$. Finally, we demonstrate these same phenomena in other settings, including the post-training of Llama-series models on mixture variations of common mathematical reasoning benchmarks.

**arXiv ID:** 2510.11495
</details>

<details>
<summary><strong>Infrastructure-based Autonomous Mobile Robots for Internal Logistics -- Challenges and Future Perspectives</strong> - Erik Brorsson, Kristian Ceder, Ze Zhang, Sabino Francesco Roselli, Endre Erős, Martin Dahl, Beatrice Alenljung, Jessica Lindblom, Thanh Bui, Emmanuel Dean, Lennart Svensson, Kristofer Bengtsson, Per-Lage Götvall, Knut Åkesson - [[pdf]](https://arxiv.org/pdf/2512.15215)</summary>

**Abstract:** The adoption of Autonomous Mobile Robots (AMRs) for internal logistics is accelerating, with most solutions emphasizing decentralized, onboard intelligence. While AMRs in indoor environments like factories can be supported by infrastructure, involving external sensors and computational resources, such systems remain underexplored in the literature. This paper presents a comprehensive overview of infrastructure-based AMR systems, outlining key opportunities and challenges. To support this, we introduce a reference architecture combining infrastructure-based sensing, on-premise cloud computing, and onboard autonomy. Based on the architecture, we review core technologies for localization, perception, and planning. We demonstrate the approach in a real-world deployment in a heavy-vehicle manufacturing environment and summarize findings from a user experience (UX) evaluation. Our aim is to provide a holistic foundation for future development of scalable, robust, and human-compatible AMR systems in complex industrial environments.

**arXiv ID:** 2512.15215
</details>

<details>
<summary><strong>Event Camera Meets Mobile Embodied Perception: Abstraction, Algorithm, Acceleration, Application</strong> - Haoyang Wang, Ruishan Guo, Pengtao Ma, Ciyu Ruan, Xinyu Luo, Wenhua Ding, Tianyang Zhong, Jingao Xu, Yunhao Liu, Xinlei Chen - [[pdf]](https://arxiv.org/pdf/2503.22943)</summary>

**Abstract:** With the increasing complexity of mobile device applications, these devices are evolving toward high agility. This shift imposes new demands on mobile sensing, particularly in achieving high-accuracy and low-latency. Event-based vision has emerged as a disruptive paradigm, offering high temporal resolution and low latency, making it well-suited for high-accuracy and low-latency sensing tasks on high-agility platforms. However, the presence of substantial noisy events, lack of stable, persistent semantic information, and large data volume pose challenges for event-based data processing on resource-constrained mobile devices. This paper surveys the literature from 2014 to 2025 and presents a comprehensive overview of event-based mobile sensing, encompassing its fundamental principles, event \textit{abstraction} methods, \textit{algorithm} advancements, and both hardware and software \textit{acceleration} strategies. We discuss key \textit{applications} of event cameras in mobile sensing, including visual odometry, object tracking, optical flow, and 3D reconstruction, while highlighting challenges associated with event data processing, sensor fusion, and real-time deployment. Furthermore, we outline future research directions, such as improving the event camera with advanced optics, leveraging neuromorphic computing for efficient processing, and integrating bio-inspired algorithms. To support ongoing research, we provide an open-source \textit{Online Sheet} with recent developments. We hope this survey serves as a reference, facilitating the adoption of event-based vision across diverse applications.

**arXiv ID:** 2503.22943
</details>

<details>
<summary><strong>Context Representation via Action-Free Transformer encoder-decoder for Meta Reinforcement Learning</strong> - Amir M. Soufi Enayati, Homayoun Honari, Homayoun Najjaran - [[pdf]](https://arxiv.org/pdf/2512.14057)</summary>

**Abstract:** Reinforcement learning (RL) enables robots to operate in uncertain environments, but standard approaches often struggle with poor generalization to unseen tasks. Context-adaptive meta reinforcement learning addresses these limitations by conditioning on the task representation, yet they mostly rely on complete action information in the experience making task inference tightly coupled to a specific policy. This paper introduces Context Representation via Action Free Transformer encoder decoder (CRAFT), a belief model that infers task representations solely from sequences of states and rewards. By removing the dependence on actions, CRAFT decouples task inference from policy optimization, supports modular training, and leverages amortized variational inference for scalable belief updates. Built on a transformer encoder decoder with rotary positional embeddings, the model captures long range temporal dependencies and robustly encodes both parametric and non-parametric task variations. Experiments on the MetaWorld ML-10 robotic manipulation benchmark show that CRAFT achieves faster adaptation, improved generalization, and more effective exploration compared to context adaptive meta--RL baselines. These findings highlight the potential of action-free inference as a foundation for scalable RL in robotic control.

**arXiv ID:** 2512.14057
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
