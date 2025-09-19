# Agent arXiv Daily

**Last Updated:** 2025-09-19 02:39:32

**Total Papers:** 60

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (4 papers)</h2></summary>

<details>
<summary><strong>Can I Trust This Chatbot? Assessing User Privacy in AI-Healthcare Chatbot Applications</strong> - Ramazan Yener, Guan-Hung Chen, Ece Gumusel, Masooda Bashir - [[pdf]](https://arxiv.org/pdf/2509.14581)</summary>

**Abstract:** As Conversational Artificial Intelligence (AI) becomes more integrated into everyday life, AI-powered chatbot mobile applications are increasingly adopted across industries, particularly in the healthcare domain. These chatbots offer accessible and 24/7 support, yet their collection and processing of sensitive health data present critical privacy concerns. While prior research has examined chatbot security, privacy issues specific to AI healthcare chatbots have received limited attention. Our study evaluates the privacy practices of 12 widely downloaded AI healthcare chatbot apps available on the App Store and Google Play in the United States. We conducted a three-step assessment analyzing: (1) privacy settings during sign-up, (2) in-app privacy controls, and (3) the content of privacy policies. The analysis identified significant gaps in user data protection. Our findings reveal that half of the examined apps did not present a privacy policy during sign up, and only two provided an option to disable data sharing at that stage. The majority of apps' privacy policies failed to address data protection measures. Moreover, users had minimal control over their personal data. The study provides key insights for information science researchers, developers, and policymakers to improve privacy protections in AI healthcare chatbot apps.

**arXiv ID:** 2509.14581
</details>

<details>
<summary><strong>Towards Human-like Multimodal Conversational Agent by Generating Engaging Speech</strong> - Taesoo Kim, Yongsik Jo, Hyunmin Song, Taehwan Kim - [[pdf]](https://arxiv.org/pdf/2509.14627)</summary>

**Abstract:** Human conversation involves language, speech, and visual cues, with each medium providing complementary information. For instance, speech conveys a vibe or tone not fully captured by text alone. While multimodal LLMs focus on generating text responses from diverse inputs, less attention has been paid to generating natural and engaging speech. We propose a human-like agent that generates speech responses based on conversation mood and responsive style information. To achieve this, we build a novel MultiSensory Conversation dataset focused on speech to enable agents to generate natural speech. We then propose a multimodal LLM-based model for generating text responses and voice descriptions, which are used to generate speech covering paralinguistic information. Experimental results demonstrate the effectiveness of utilizing both visual and audio modalities in conversation to generate engaging speech. The source code is available in this https URL

**arXiv ID:** 2509.14627
</details>

<details>
<summary><strong>VLM-E2E: Enhancing End-to-End Autonomous Driving with Multimodal Driver Attention Fusion</strong> - Pei Liu, Haipeng Liu, Haichao Liu, Xin Liu, Jinxin Ni, Jun Ma - [[pdf]](https://arxiv.org/pdf/2502.18042)</summary>

**Abstract:** Human drivers adeptly navigate complex scenarios by utilizing rich attentional semantics, but the current autonomous systems struggle to replicate this ability, as they often lose critical semantic information when converting 2D observations into 3D space. In this sense, it hinders their effective deployment in dynamic and complex environments. Leveraging the superior scene understanding and reasoning abilities of Vision-Language Models (VLMs), we propose VLM-E2E, a novel framework that uses the VLMs to enhance training by providing attentional cues. Our method integrates textual representations into Bird's-Eye-View (BEV) features for semantic supervision, which enables the model to learn richer feature representations that explicitly capture the driver's attentional semantics. By focusing on attentional semantics, VLM-E2E better aligns with human-like driving behavior, which is critical for navigating dynamic and complex environments. Furthermore, we introduce a BEV-Text learnable weighted fusion strategy to address the issue of modality importance imbalance in fusing multimodal information. This approach dynamically balances the contributions of BEV and text features, ensuring that the complementary information from visual and textual modalities is effectively utilized. By explicitly addressing the imbalance in multimodal fusion, our method facilitates a more holistic and robust representation of driving environments. We evaluate VLM-E2E on the nuScenes dataset and achieve significant improvements in perception, prediction, and planning over the baseline end-to-end model, showcasing the effectiveness of our attention-enhanced BEV representation in enabling more accurate and reliable autonomous driving tasks.

**arXiv ID:** 2502.18042
</details>

<details>
<summary><strong>Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System</strong> - Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Yu Liu, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She - [[pdf]](https://arxiv.org/pdf/2509.05755)</summary>

**Abstract:** LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.

**arXiv ID:** 2509.05755
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (9 papers)</h2></summary>

<details>
<summary><strong>Detecting Pipeline Failures through Fine-Grained Analysis of Web Agents</strong> - Daniel Röder, Akhil Juneja, Roland Roller, Sven Schmeier - [[pdf]](https://arxiv.org/pdf/2509.14382)</summary>

**Abstract:** Web agents powered by large language models (LLMs) can autonomously perform complex, multistep tasks in dynamic web environments. However, current evaluations mostly focus on the overall success while overlooking intermediate errors. This limits insight into failure modes and hinders systematic improvement. This work analyzes existing benchmarks and highlights the lack of fine-grained diagnostic tools. To address this gap, we propose a modular evaluation framework that decomposes agent pipelines into interpretable stages for detailed error analysis. Using the SeeAct framework and the Mind2Web dataset as a case study, we show how this approach reveals actionable weaknesses missed by standard metrics - paving the way for more robust and generalizable web agents.

**arXiv ID:** 2509.14382
</details>

<details>
<summary><strong>OpenLens AI: Fully Autonomous Research Agent for Health Infomatics</strong> - Yuxiao Cheng, Jinli Suo - [[pdf]](https://arxiv.org/pdf/2509.14778)</summary>

**Abstract:** Health informatics research is characterized by diverse data modalities, rapid knowledge expansion, and the need to integrate insights across biomedical science, data analytics, and clinical practice. These characteristics make it particularly well-suited for agent-based approaches that can automate knowledge exploration, manage complex workflows, and generate clinically meaningful outputs. Recent progress in large language model (LLM)-based agents has demonstrated promising capabilities in literature synthesis, data analysis, and even end-to-end research execution. However, existing systems remain limited for health informatics because they lack mechanisms to interpret medical visualizations and often overlook domain-specific quality requirements. To address these gaps, we introduce OpenLens AI, a fully automated framework tailored to health informatics. OpenLens AI integrates specialized agents for literature review, data analysis, code generation, and manuscript preparation, enhanced by vision-language feedback for medical visualization and quality control for reproducibility. The framework automates the entire research pipeline, producing publication-ready LaTeX manuscripts with transparent and traceable workflows, thereby offering a domain-adapted solution for advancing health informatics research.

**arXiv ID:** 2509.14778
</details>

<details>
<summary><strong>Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization</strong> - Robert Tjarko Lange, Qi Sun, Aaditya Prasad, Maxence Faldor, Yujin Tang, David Ha - [[pdf]](https://arxiv.org/pdf/2509.14279)</summary>

**Abstract:** Recent advances in large language models (LLMs) demonstrate their effectiveness in scaling test-time compute for software engineering tasks. However, these approaches often focus on high-level solutions, with limited attention to optimizing low-level CUDA kernel implementations. Additionally, existing kernel generation benchmarks suffer from exploitable loopholes and insufficient diversity in testing conditions, hindering true generalization assessment. To address these limitations, we introduce robust-kbench, a new benchmark for rigorous evaluation of kernel performance and correctness across varied scenarios. Furthermore, we present a comprehensive agentic framework that automates CUDA kernel discovery, verification, and optimization. This pipeline enables frontier LLMs to translate torch code to CUDA kernels and iteratively improve their runtime within our robust evaluation setting. Our sequential workflow first translates PyTorch code into equivalent CUDA kernels. It then optimizes their runtime using a novel evolutionary meta-generation procedure tailored to the CUDA ecosystem, guided by LLM-based verifiers for correctness and efficient filtering. Evaluated on robust-kbench, our approach produces CUDA kernels outperforming torch implementations for practical applications, including forward and backward passes. It can fuse operations and deploy various runtime optimization strategies. The verifier workflow accurately classifies incorrect kernels, enhancing hardware verification efficiency.

**arXiv ID:** 2509.14279
</details>

<details>
<summary><strong>FlowDrive: Energy Flow Field for End-to-End Autonomous Driving</strong> - Hao Jiang, Zhipeng Zhang, Yu Gao, Zhigang Sun, Yiru Wang, Yuwen Heng, Shuo Wang, Jinhao Chai, Zhuo Chen, Hao Zhao, Hao Sun, Xi Zhang, Anqing Jiang, Chuan Hu - [[pdf]](https://arxiv.org/pdf/2509.14303)</summary>

**Abstract:** Recent advances in end-to-end autonomous driving leverage multi-view images to construct BEV representations for motion planning. In motion planning, autonomous vehicles need considering both hard constraints imposed by geometrically occupied obstacles (e.g., vehicles, pedestrians) and soft, rule-based semantics with no explicit geometry (e.g., lane boundaries, traffic priors). However, existing end-to-end frameworks typically rely on BEV features learned in an implicit manner, lacking explicit modeling of risk and guidance priors for safe and interpretable planning. To address this, we propose FlowDrive, a novel framework that introduces physically interpretable energy-based flow fields-including risk potential and lane attraction fields-to encode semantic priors and safety cues into the BEV space. These flow-aware features enable adaptive refinement of anchor trajectories and serve as interpretable guidance for trajectory generation. Moreover, FlowDrive decouples motion intent prediction from trajectory denoising via a conditional diffusion planner with feature-level gating, alleviating task interference and enhancing multimodal diversity. Experiments on the NAVSIM v2 benchmark demonstrate that FlowDrive achieves state-of-the-art performance with an EPDMS of 86.3, surpassing prior baselines in both safety and planning quality. The project is available at this https URL.

**arXiv ID:** 2509.14303
</details>

<details>
<summary><strong>An Evaluation-Centric Paradigm for Scientific Visualization Agents</strong> - Kuangshi Ai, Haichao Miao, Zhimin Li, Chaoli Wang, Shusen Liu - [[pdf]](https://arxiv.org/pdf/2509.15160)</summary>

**Abstract:** Recent advances in multi-modal large language models (MLLMs) have enabled increasingly sophisticated autonomous visualization agents capable of translating user intentions into data visualizations. However, measuring progress and comparing different agents remains challenging, particularly in scientific visualization (SciVis), due to the absence of comprehensive, large-scale benchmarks for evaluating real-world capabilities. This position paper examines the various types of evaluation required for SciVis agents, outlines the associated challenges, provides a simple proof-of-concept evaluation example, and discusses how evaluation benchmarks can facilitate agent self-improvement. We advocate for a broader collaboration to develop a SciVis agentic evaluation benchmark that would not only assess existing capabilities but also drive innovation and stimulate future development in the field.

**arXiv ID:** 2509.15160
</details>

<details>
<summary><strong>WebCoT: Enhancing Web Agent Reasoning by Reconstructing Chain-of-Thought in Reflection, Branching, and Rollback</strong> - Minda Hu, Tianqing Fang, Jianshu Zhang, Junyu Ma, Zhisong Zhang, Jingyan Zhou, Hongming Zhang, Haitao Mi, Dong Yu, Irwin King - [[pdf]](https://arxiv.org/pdf/2505.20013)</summary>

**Abstract:** Web agents powered by Large Language Models (LLMs) show promise for next-generation AI, but their limited reasoning in uncertain, dynamic web environments hinders robust deployment. In this paper, we identify key reasoning skills essential for effective web agents, i.e., reflection & lookahead, branching, and rollback, and curate trajectory data that exemplifies these abilities by reconstructing the agent's (inference-time) reasoning algorithms into chain-of-thought rationales. We conduct experiments in the agent self-improving benchmark, OpenWebVoyager, and demonstrate that distilling salient reasoning patterns into the backbone LLM via simple fine-tuning can substantially enhance its performance. Our approach yields significant improvements across multiple benchmarks, including WebVoyager, Mind2web-live, and SimpleQA (web search), highlighting the potential of targeted reasoning skill enhancement for web agents.

**arXiv ID:** 2505.20013
</details>

<details>
<summary><strong>SimCoachCorpus: A naturalistic dataset with language and trajectories for embodied teaching</strong> - Emily Sumner, Deepak E. Gopinath, Laporsha Dees, Patricio Reyes Gomez, Xiongyi Cui, Andrew Silva, Jean Costa, Allison Morgan, Mariah Schrum, Tiffany L. Chen, Avinash Balachandran, Guy Rosman - [[pdf]](https://arxiv.org/pdf/2509.14548)</summary>

**Abstract:** Curated datasets are essential for training and evaluating AI approaches, but are often lacking in domains where language and physical action are deeply intertwined. In particular, few datasets capture how people acquire embodied skills through verbal instruction over time. To address this gap, we introduce SimCoachCorpus: a unique dataset of race car simulator driving that allows for the investigation of rich interactive phenomena during guided and unguided motor skill acquisition. In this dataset, 29 humans were asked to drive in a simulator around a race track for approximately ninety minutes. Fifteen participants were given personalized one-on-one instruction from a professional performance driving coach, and 14 participants drove without coaching. \name\ includes embodied features such as vehicle state and inputs, map (track boundaries and raceline), and cone landmarks. These are synchronized with concurrent verbal coaching from a professional coach and additional feedback at the end of each lap. We further provide annotations of coaching categories for each concurrent feedback utterance, ratings on students' compliance with coaching advice, and self-reported cognitive load and emotional state of participants (gathered from surveys during the study). The dataset includes over 20,000 concurrent feedback utterances, over 400 terminal feedback utterances, and over 40 hours of vehicle driving data. Our naturalistic dataset can be used for investigating motor learning dynamics, exploring linguistic phenomena, and training computational models of teaching. We demonstrate applications of this dataset for in-context learning, imitation learning, and topic modeling. The dataset introduced in this work will be released publicly upon publication of the peer-reviewed version of this paper. Researchers interested in early access may register at this https URL.

**arXiv ID:** 2509.14548
</details>

<details>
<summary><strong>RealMirror: A Comprehensive, Open-Source Vision-Language-Action Platform for Embodied AI</strong> - Cong Tai, Zhaoyu Zheng, Haixu Long, Hansheng Wu, Haodong Xiang, Zhengbin Long, Jun Xiong, Rong Shi, Shizhuang Zhang, Gang Qiu, He Wang, Ruifeng Li, Jun Huang, Bin Chang, Shuai Feng, Tao Shen - [[pdf]](https://arxiv.org/pdf/2509.14687)</summary>

**Abstract:** The emerging field of Vision-Language-Action (VLA) for humanoid robots faces several fundamental challenges, including the high cost of data acquisition, the lack of a standardized benchmark, and the significant gap between simulation and the real world. To overcome these obstacles, we propose RealMirror, a comprehensive, open-source embodied AI VLA platform. RealMirror builds an efficient, low-cost data collection, model training, and inference system that enables end-to-end VLA research without requiring a real robot. To facilitate model evolution and fair comparison, we also introduce a dedicated VLA benchmark for humanoid robots, featuring multiple scenarios, extensive trajectories, and various VLA models. Furthermore, by integrating generative models and 3D Gaussian Splatting to reconstruct realistic environments and robot models, we successfully demonstrate zero-shot Sim2Real transfer, where models trained exclusively on simulation data can perform tasks on a real robot seamlessly, without any fine-tuning. In conclusion, with the unification of these critical components, RealMirror provides a robust framework that significantly accelerates the development of VLA models for humanoid robots. Project page: this https URL

**arXiv ID:** 2509.14687
</details>

<details>
<summary><strong>T-araVLN: Translator for Agricultural Robotic Agents on Vision-and-Language Navigation</strong> - Xiaobei Zhao, Xingqi Lyu, Xiang Li - [[pdf]](https://arxiv.org/pdf/2509.06644)</summary>

**Abstract:** Agricultural robotic agents have been becoming powerful helpers in a wide range of agricultural tasks, however, still heavily rely on manual operation or fixed railways for movement. To address this limitation, the AgriVLN method and the A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the agricultural domain, enabling agents to navigate to the target positions following the natural language instructions. AgriVLN effectively understands the simple instructions, but often misunderstands the complex ones. To bridge this gap, we propose the method of Translator for Agricultural Robotic Agents on Vision-and-Language Navigation (T-araVLN), in which the Instruction Translator module translates the original instruction to be more refined and precise. When evaluated on the A2A benchmark, our T-araVLN effectively improves Success Rate from 0.47 to 0.63 and reduces Navigation Error from 2.91m to 2.28m, demonstrating the state-of-the-art performance in the agricultural domain. Code: this https URL.

**arXiv ID:** 2509.06644
</details>

</details>

<details open>
<summary><h2>LLM Agents (2 papers)</h2></summary>

<details>
<summary><strong>From Correction to Mastery: Reinforced Distillation of Large Language Model Agents</strong> - Yuanjie Lyu, Chengyu Wang, Jun Huang, Tong Xu - [[pdf]](https://arxiv.org/pdf/2509.14257)</summary>

**Abstract:** Large Language Model agents excel at solving complex tasks through iterative reasoning and tool use, but typically depend on ultra-large, costly backbones. Existing distillation approaches train smaller students to imitate full teacher trajectories, yet reasoning and knowledge gaps between the teacher and student often lead to compounding errors. We propose SCoRe, a student-centered framework in which the student generates trajectories and the teacher intervenes only at the first critical error, producing training data matched to the student's ability and exposing specific weaknesses. The student is first fine-tuned on corrected trajectories. Subsequently, short-horizon reinforcement learning starts from the verified prefix before the first critical error, with target rewards assigned at that step. This design encourages autonomous problem-solving beyond imitation and improves training stability. Particularly, on 12 challenging benchmarks, a 7B-parameter student distilled with SCoRe matches the agentic performance of a 72B-parameter teacher.

**arXiv ID:** 2509.14257
</details>

<details>
<summary><strong>Ticket-Bench: A Kickoff for Multilingual and Regionalized Agent Evaluation</strong> - Thales Sales Almeida, João Guilherme Alves Santos, Thiago Laitz, Giovana Kerche Bonás - [[pdf]](https://arxiv.org/pdf/2509.14477)</summary>

**Abstract:** Large language models (LLMs) are increasingly deployed as task-oriented agents, where success depends on their ability to generate accurate function calls under realistic, multilingual conditions. However, existing agent evaluations largely overlook cultural and linguistic diversity, often relying on monolingual or naively translated benchmarks. We introduce Ticket-Bench, a benchmark for multilingual agent evaluation in task-oriented scenarios. Ticket-Bench simulates the domain of soccer ticket purchases across six major languages: Portuguese, English, Spanish, German, Italian, and French. Using localized teams, cities, and user profiles to provide a higher level of realism. We evaluate a wide range of commercial and open-source LLMs, measuring function-calling accuracy and consistency across languages. Results show that reasoning-oriented models (e.g., GPT-5, Qwen3-235B) dominate performance but still exhibit notable cross-lingual disparities. These findings underscore the need for culturally aware, multilingual benchmarks to guide the development of robust LLM agents.

**arXiv ID:** 2509.14477
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (25 papers)</h2></summary>

<details>
<summary><strong>(P)rior(D)yna(F)low: A Priori Dynamic Workflow Construction via Multi-Agent Collaboration</strong> - Yi Lin, Lujin Zhao, Yijie Shi - [[pdf]](https://arxiv.org/pdf/2509.14547)</summary>

**Abstract:** Recent studies have shown that carefully designed workflows coordinating large language models(LLMs) significantly enhance task-solving capabilities compared to using a single model. While an increasing number of works focus on autonomous workflow construction, most existing approaches rely solely on historical experience, leading to limitations in efficiency and adaptability. We argue that while historical experience is valuable, workflow construction should also flexibly respond to the unique characteristics of each task. To this end, we propose an a priori dynamic framework for automated workflow construction. Our framework first leverages Q-table learning to optimize the decision space, guiding agent decisions and enabling effective use of historical experience. At the same time, agents evaluate the current task progress and make a priori decisions regarding the next executing agent, allowing the system to proactively select the more suitable workflow structure for each given task. Additionally, we incorporate mechanisms such as cold-start initialization, early stopping, and pruning to further improve system efficiency. Experimental evaluations on four benchmark datasets demonstrate the feasibility and effectiveness of our approach. Compared to state-of-the-art baselines, our method achieves an average improvement of 4.05%, while reducing workflow construction and inference costs to only 30.68%-48.31% of those required by existing methods.

**arXiv ID:** 2509.14547
</details>

<details>
<summary><strong>Beyond the high score: Prosocial ability profiles of multi-agent populations</strong> - Marko Tesic, Yue Zhao, Joel Z. Leibo, Rakshit S. Trivedi, Jose Hernandez-Orallo - [[pdf]](https://arxiv.org/pdf/2509.14485)</summary>

**Abstract:** The development and evaluation of social capabilities in AI agents require complex environments where competitive and cooperative behaviours naturally emerge. While game-theoretic properties can explain why certain teams or agent populations outperform others, more abstract behaviours, such as convention following, are harder to control in training and evaluation settings. The Melting Pot contest is a social AI evaluation suite designed to assess the cooperation capabilities of AI systems. In this paper, we apply a Bayesian approach known as Measurement Layouts to infer the capability profiles of multi-agent systems in the Melting Pot contest. We show that these capability profiles not only predict future performance within the Melting Pot suite but also reveal the underlying prosocial abilities of agents. Our analysis indicates that while higher prosocial capabilities sometimes correlate with better performance, this is not a universal trend-some lower-scoring agents exhibit stronger cooperation abilities. Furthermore, we find that top-performing contest submissions are more likely to achieve high scores in scenarios where prosocial capabilities are not required. These findings, together with reports that the contest winner used a hard-coded solution tailored to specific environments, suggest that at least one top-performing team may have optimised for conditions where cooperation was not necessary, potentially exploiting limitations in the evaluation framework. We provide recommendations for improving the annotation of cooperation demands and propose future research directions to account for biases introduced by different testing environments. Our results demonstrate that Measurement Layouts offer both strong predictive accuracy and actionable insights, contributing to a more transparent and generalisable approach to evaluating AI systems in complex social settings.

**arXiv ID:** 2509.14485
</details>

<details>
<summary><strong>AgentCompass: Towards Reliable Evaluation of Agentic Workflows in Production</strong> - NVJK Kartik, Garvit Sapra, Rishav Hada, Nikhil Pareek - [[pdf]](https://arxiv.org/pdf/2509.14647)</summary>

**Abstract:** With the growing adoption of Large Language Models (LLMs) in automating complex, multi-agent workflows, organizations face mounting risks from errors, emergent behaviors, and systemic failures that current evaluation methods fail to capture. We present AgentCompass, the first evaluation framework designed specifically for post-deployment monitoring and debugging of agentic workflows. AgentCompass models the reasoning process of expert debuggers through a structured, multi-stage analytical pipeline: error identification and categorization, thematic clustering, quantitative scoring, and strategic summarization. The framework is further enhanced with a dual memory system-episodic and semantic-that enables continual learning across executions. Through collaborations with design partners, we demonstrate the framework's practical utility on real-world deployments, before establishing its efficacy against the publicly available TRAIL benchmark. AgentCompass achieves state-of-the-art results on key metrics, while uncovering critical issues missed in human annotations, underscoring its role as a robust, developer-centric tool for reliable monitoring and improvement of agentic systems in production.

**arXiv ID:** 2509.14647
</details>

<details>
<summary><strong>Sentinel Agents for Secure and Trustworthy Agentic AI in Multi-Agent Systems</strong> - Diego Gosmar, Deborah A. Dahl - [[pdf]](https://arxiv.org/pdf/2509.14956)</summary>

**Abstract:** This paper proposes a novel architectural framework aimed at enhancing security and reliability in multi-agent systems (MAS). A central component of this framework is a network of Sentinel Agents, functioning as a distributed security layer that integrates techniques such as semantic analysis via large language models (LLMs), behavioral analytics, retrieval-augmented verification, and cross-agent anomaly detection. Such agents can potentially oversee inter-agent communications, identify potential threats, enforce privacy and access controls, and maintain comprehensive audit records. Complementary to the idea of Sentinel Agents is the use of a Coordinator Agent. The Coordinator Agent supervises policy implementation, and manages agent participation. In addition, the Coordinator also ingests alerts from Sentinel Agents. Based on these alerts, it can adapt policies, isolate or quarantine misbehaving agents, and contain threats to maintain the integrity of the MAS ecosystem. This dual-layered security approach, combining the continuous monitoring of Sentinel Agents with the governance functions of Coordinator Agents, supports dynamic and adaptive defense mechanisms against a range of threats, including prompt injection, collusive agent behavior, hallucinations generated by LLMs, privacy breaches, and coordinated multi-agent attacks. In addition to the architectural design, we present a simulation study where 162 synthetic attacks of different families (prompt injection, hallucination, and data exfiltration) were injected into a multi-agent conversational environment. The Sentinel Agents successfully detected the attack attempts, confirming the practical feasibility of the proposed monitoring approach. The framework also offers enhanced system observability, supports regulatory compliance, and enables policy evolution over time.

**arXiv ID:** 2509.14956
</details>

<details>
<summary><strong>Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment</strong> - Ankur Samanta, Akshayaa Magesh, Youliang Yu, Runzhe Wu, Ayush Jain, Daniel Jiang, Boris Vidolov, Paul Sajda, Yonathan Efroni, Kaveh Hassani - [[pdf]](https://arxiv.org/pdf/2509.15172)</summary>

**Abstract:** Language Models (LMs) are inconsistent reasoners, often generating contradictory responses to identical prompts. While inference-time methods can mitigate these inconsistencies, they fail to address the core problem: LMs struggle to reliably select reasoning pathways leading to consistent outcomes under exploratory sampling. To address this, we formalize self-consistency as an intrinsic property of well-aligned reasoning models and introduce Multi-Agent Consensus Alignment (MACA), a reinforcement learning framework that post-trains models to favor reasoning trajectories aligned with their internal consensus using majority/minority outcomes from multi-agent debate. These trajectories emerge from deliberative exchanges where agents ground reasoning in peer arguments, not just aggregation of independent attempts, creating richer consensus signals than single-round majority voting. MACA enables agents to teach themselves to be more decisive and concise, and better leverage peer insights in multi-agent settings without external supervision, driving substantial improvements across self-consistency (+27.6% on GSM8K), single-agent reasoning (+23.7% on MATH), sampling-based inference (+22.4% Pass@20 on MATH), and multi-agent ensemble decision-making (+42.7% on MathQA). These findings, coupled with strong generalization to unseen benchmarks (+16.3% on GPQA, +11.6% on CommonsenseQA), demonstrate robust self-alignment that more reliably unlocks latent reasoning potential of language models.

**arXiv ID:** 2509.15172
</details>

<details>
<summary><strong>Constructive Conflict-Driven Multi-Agent Reinforcement Learning for Strategic Diversity</strong> - Yuxiang Mai, Qiyue Yin, Wancheng Ni, Pei Xu, Kaiqi Huang - [[pdf]](https://arxiv.org/pdf/2509.14276)</summary>

**Abstract:** In recent years, diversity has emerged as a useful mechanism to enhance the efficiency of multi-agent reinforcement learning (MARL). However, existing methods predominantly focus on designing policies based on individual agent characteristics, often neglecting the interplay and mutual influence among agents during policy formation. To address this gap, we propose Competitive Diversity through Constructive Conflict (CoDiCon), a novel approach that incorporates competitive incentives into cooperative scenarios to encourage policy exchange and foster strategic diversity among agents. Drawing inspiration from sociological research, which highlights the benefits of moderate competition and constructive conflict in group decision-making, we design an intrinsic reward mechanism using ranking features to introduce competitive motivations. A centralized intrinsic reward module generates and distributes varying reward values to agents, ensuring an effective balance between competition and cooperation. By optimizing the parameterized centralized reward module to maximize environmental rewards, we reformulate the constrained bilevel optimization problem to align with the original task objectives. We evaluate our algorithm against state-of-the-art methods in the SMAC and GRF environments. Experimental results demonstrate that CoDiCon achieves superior performance, with competitive intrinsic rewards effectively promoting diverse and adaptive strategies among cooperative agents.

**arXiv ID:** 2509.14276
</details>

<details>
<summary><strong>The Sum Leaks More Than Its Parts: Compositional Privacy Risks and Mitigations in Multi-Agent Collaboration</strong> - Vaidehi Patil, Elias Stengel-Eskin, Mohit Bansal - [[pdf]](https://arxiv.org/pdf/2509.14284)</summary>

**Abstract:** As large language models (LLMs) become integral to multi-agent systems, new privacy risks emerge that extend beyond memorization, direct inference, or single-turn evaluations. In particular, seemingly innocuous responses, when composed across interactions, can cumulatively enable adversaries to recover sensitive information, a phenomenon we term compositional privacy leakage. We present the first systematic study of such compositional privacy leaks and possible mitigation methods in multi-agent LLM systems. First, we develop a framework that models how auxiliary knowledge and agent interactions jointly amplify privacy risks, even when each response is benign in isolation. Next, to mitigate this, we propose and evaluate two defense strategies: (1) Theory-of-Mind defense (ToM), where defender agents infer a questioner's intent by anticipating how their outputs may be exploited by adversaries, and (2) Collaborative Consensus Defense (CoDef), where responder agents collaborate with peers who vote based on a shared aggregated state to restrict sensitive information spread. Crucially, we balance our evaluation across compositions that expose sensitive information and compositions that yield benign inferences. Our experiments quantify how these defense strategies differ in balancing the privacy-utility trade-off. We find that while chain-of-thought alone offers limited protection to leakage (~39% sensitive blocking rate), our ToM defense substantially improves sensitive query blocking (up to 97%) but can reduce benign task success. CoDef achieves the best balance, yielding the highest Balanced Outcome (79.8%), highlighting the benefit of combining explicit reasoning with defender collaboration. Together, our results expose a new class of risks in collaborative LLM deployments and provide actionable insights for designing safeguards against compositional, context-driven privacy leakage.

**arXiv ID:** 2509.14284
</details>

<details>
<summary><strong>OnlineMate: An LLM-Based Multi-Agent Companion System for Cognitive Support in Online Learning</strong> - Xian Gao, Zongyun Zhang, Ting Liu, Yuzhuo Fu - [[pdf]](https://arxiv.org/pdf/2509.14803)</summary>

**Abstract:** In online learning environments, students often lack personalized peer interactions, which play a crucial role in supporting cognitive development and learning engagement. Although previous studies have utilized large language models (LLMs) to simulate interactive dynamic learning environments for students, these interactions remain limited to conversational exchanges, lacking insights and adaptations to the learners' individualized learning and cognitive states. As a result, students' interest in discussions with AI learning companions is low, and they struggle to gain inspiration from such interactions. To address this challenge, we propose OnlineMate, a multi-agent learning companion system driven by LLMs that integrates the Theory of Mind (ToM). OnlineMate is capable of simulating peer-like agent roles, adapting to learners' cognitive states during collaborative discussions, and inferring their psychological states, such as misunderstandings, confusion, or motivation. By incorporating Theory of Mind capabilities, the system can dynamically adjust its interaction strategies to support the development of higher-order thinking and cognition. Experimental results in simulated learning scenarios demonstrate that OnlineMate effectively fosters deep learning and discussions while enhancing cognitive engagement in online educational settings.

**arXiv ID:** 2509.14803
</details>

<details>
<summary><strong>MARIC: Multi-Agent Reasoning for Image Classification</strong> - Wonduk Seo, Minhyeong Yu, Hyunjin An, Seunghyun Lee - [[pdf]](https://arxiv.org/pdf/2509.14860)</summary>

**Abstract:** Image classification has traditionally relied on parameter-intensive model training, requiring large-scale annotated datasets and extensive fine tuning to achieve competitive performance. While recent vision language models (VLMs) alleviate some of these constraints, they remain limited by their reliance on single pass representations, often failing to capture complementary aspects of visual content. In this paper, we introduce Multi Agent based Reasoning for Image Classification (MARIC), a multi agent framework that reformulates image classification as a collaborative reasoning process. MARIC first utilizes an Outliner Agent to analyze the global theme of the image and generate targeted prompts. Based on these prompts, three Aspect Agents extract fine grained descriptions along distinct visual dimensions. Finally, a Reasoning Agent synthesizes these complementary outputs through integrated reflection step, producing a unified representation for classification. By explicitly decomposing the task into multiple perspectives and encouraging reflective synthesis, MARIC mitigates the shortcomings of both parameter-heavy training and monolithic VLM reasoning. Experiments on 4 diverse image classification benchmark datasets demonstrate that MARIC significantly outperforms baselines, highlighting the effectiveness of multi-agent visual reasoning for robust and interpretable image classification.

**arXiv ID:** 2509.14860
</details>

<details>
<summary><strong>AI-Driven Multi-Agent Vehicular Planning for Battery Efficiency and QoS in 6G Smart Cities</strong> - Rohin Gillgallon, Giacomo Bergami, Reham Almutairi, Graham Morgan - [[pdf]](https://arxiv.org/pdf/2509.14877)</summary>

**Abstract:** While simulators exist for vehicular IoT nodes communicating with the Cloud through Edge nodes in a fully-simulated osmotic architecture, they often lack support for dynamic agent planning and optimisation to minimise vehicular battery consumption while ensuring fair communication times. Addressing these challenges requires extending current simulator architectures with AI algorithms for both traffic prediction and dynamic agent planning. This paper presents an extension of SimulatorOrchestrator (SO) to meet these requirements. Preliminary results over a realistic urban dataset show that utilising vehicular planning algorithms can lead to improved battery and QoS performance compared with traditional shortest path algorithms. The additional inclusion of desirability areas enabled more ambulances to be routed to their target destinations while utilising less energy to do so, compared to traditional and weighted algorithms without desirability considerations.

**arXiv ID:** 2509.14877
</details>

<details>
<summary><strong>Reinforcement Learning Agent for a 2D Shooter Game</strong> - Thomas Ackermann, Moritz Spang, Hamza A. A. Gardi - [[pdf]](https://arxiv.org/pdf/2509.15042)</summary>

**Abstract:** Reinforcement learning agents in complex game environments often suffer from sparse rewards, training instability, and poor sample efficiency. This paper presents a hybrid training approach that combines offline imitation learning with online reinforcement learning for a 2D shooter game agent. We implement a multi-head neural network with separate outputs for behavioral cloning and Q-learning, unified by shared feature extraction layers with attention mechanisms. Initial experiments using pure deep Q-Networks exhibited significant instability, with agents frequently reverting to poor policies despite occasional good performance. To address this, we developed a hybrid methodology that begins with behavioral cloning on demonstration data from rule-based agents, then transitions to reinforcement learning. Our hybrid approach achieves consistently above 70% win rate against rule-based opponents, substantially outperforming pure reinforcement learning methods which showed high variance and frequent performance degradation. The multi-head architecture enables effective knowledge transfer between learning modes while maintaining training stability. Results demonstrate that combining demonstration-based initialization with reinforcement learning optimization provides a robust solution for developing game AI agents in complex multi-agent environments where pure exploration proves insufficient.

**arXiv ID:** 2509.15042
</details>

<details>
<summary><strong>Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning</strong> - Simin Li, Zheng Yuwei, Zihao Mao, Linhao Wang, Ruixiao Xu, Chengdong Ma, Xin Yu, Yuqing Ma, Qi Dou, Xin Wang, Jie Luo, Bo An, Yaodong Yang, Weifeng Lv, Xianglong Liu - [[pdf]](https://arxiv.org/pdf/2509.15103)</summary>

**Abstract:** Partial agent failure becomes inevitable when systems scale up, making it crucial to identify the subset of agents whose compromise would most severely degrade overall performance. In this paper, we study this Vulnerable Agent Identification (VAI) problem in large-scale multi-agent reinforcement learning (MARL). We frame VAI as a Hierarchical Adversarial Decentralized Mean Field Control (HAD-MFC), where the upper level involves an NP-hard combinatorial task of selecting the most vulnerable agents, and the lower level learns worst-case adversarial policies for these agents using mean-field MARL. The two problems are coupled together, making HAD-MFC difficult to solve. To solve this, we first decouple the hierarchical process by Fenchel-Rockafellar transform, resulting a regularized mean-field Bellman operator for upper level that enables independent learning at each level, thus reducing computational complexity. We then reformulate the upper-level combinatorial problem as a MDP with dense rewards from our regularized mean-field Bellman operator, enabling us to sequentially identify the most vulnerable agents by greedy and RL algorithms. This decomposition provably preserves the optimal solution of the original HAD-MFC. Experiments show our method effectively identifies more vulnerable agents in large-scale MARL and the rule-based system, fooling system into worse failures, and learns a value function that reveals the vulnerability of each agent.

**arXiv ID:** 2509.15103
</details>

<details>
<summary><strong>Mastering Multi-Drone Volleyball through Hierarchical Co-Self-Play Reinforcement Learning</strong> - Ruize Zhang, Sirui Xiang, Zelai Xu, Feng Gao, Shilong Ji, Wenhao Tang, Wenbo Ding, Chao Yu, Yu Wang - [[pdf]](https://arxiv.org/pdf/2505.04317)</summary>

**Abstract:** In this paper, we tackle the problem of learning to play 3v3 multi-drone volleyball, a new embodied competitive task that requires both high-level strategic coordination and low-level agile control. The task is turn-based, multi-agent, and physically grounded, posing significant challenges due to its long-horizon dependencies, tight inter-agent coupling, and the underactuated dynamics of quadrotors. To address this, we propose Hierarchical Co-Self-Play (HCSP), a hierarchical reinforcement learning framework that separates centralized high-level strategic decision-making from decentralized low-level motion control. We design a three-stage population-based training pipeline to enable both strategy and skill to emerge from scratch without expert demonstrations: (I) training diverse low-level skills, (II) learning high-level strategy via self-play with fixed low-level skills, and (III) joint fine-tuning through co-self-play. Experiments show that HCSP achieves superior performance, outperforming non-hierarchical self-play and rule-based hierarchical baselines with an average 82.9% win rate and a 71.5% win rate against the two-stage variant. Moreover, co-self-play leads to emergent team behaviors such as role switching and coordinated formations, demonstrating the effectiveness of our hierarchical design and training scheme. The project page is at this https URL.

**arXiv ID:** 2505.04317
</details>

<details>
<summary><strong>Judging with Many Minds: Do More Perspectives Mean Less Prejudice? On Bias Amplifications and Resistance in Multi-Agent Based LLM-as-Judge</strong> - Chiyu Ma, Enpei Zhang, Yilun Zhao, Wenjun Liu, Yaning Jia, Peijun Qing, Lin Shi, Arman Cohan, Yujun Yan, Soroush Vosoughi - [[pdf]](https://arxiv.org/pdf/2505.19477)</summary>

**Abstract:** LLM-as-Judge has emerged as a scalable alternative to human evaluation, enabling large language models (LLMs) to provide reward signals in trainings. While recent work has explored multi-agent extensions such as multi-agent debate and meta-judging to enhance evaluation quality, the question of how intrinsic biases manifest in these settings remains underexplored. In this study, we conduct a systematic analysis of four diverse bias types: position bias, verbosity bias, chain-of-thought bias, and bandwagon bias. We evaluate these biases across two widely adopted multi-agent LLM-as-Judge frameworks: Multi-Agent-Debate and LLM-as-Meta-Judge. Our results show that debate framework amplifies biases sharply after the initial debate, and this increased bias is sustained in subsequent rounds, while meta-judge approaches exhibit greater resistance. We further investigate the incorporation of PINE, a leading single-agent debiasing method, as a bias-free agent within these systems. The results reveal that this bias-free agent effectively reduces biases in debate settings but provides less benefit in meta-judge scenarios. Our work provides a comprehensive study of bias behavior in multi-agent LLM-as-Judge systems and highlights the need for targeted bias mitigation strategies in collaborative evaluation settings.

**arXiv ID:** 2505.19477
</details>

<details>
<summary><strong>METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling</strong> - Bingxuan Li, Yiwei Wang, Jiuxiang Gu, Kai-Wei Chang, Nanyun Peng - [[pdf]](https://arxiv.org/pdf/2502.17651)</summary>

**Abstract:** Chart generation aims to generate code to produce charts satisfying the desired visual properties, e.g., texts, layout, color, and type. It has great potential to empower the automatic professional report generation in financial analysis, research presentation, education, and healthcare. In this work, we build a vision-language model (VLM) based multi-agent framework for effective automatic chart generation. Generating high-quality charts requires both strong visual design skills and precise coding capabilities that embed the desired visual properties into code. Such a complex multi-modal reasoning process is difficult for direct prompting of VLMs. To resolve these challenges, we propose METAL, a multi-agent framework that decomposes the task of chart generation into the iterative collaboration among specialized agents. METAL achieves 5.2% improvement over the current best result in the chart generation task. The METAL framework exhibits the phenomenon of test-time scaling: its performance increases monotonically as the logarithmic computational budget grows from 512 to 8192 tokens. In addition, we find that separating different modalities during the critique process of METAL boosts the self-correction capability of VLMs in the multimodal context.

**arXiv ID:** 2502.17651
</details>

<details>
<summary><strong>Predicting Multi-Agent Specialization via Task Parallelizability</strong> - Elizabeth Mieczkowski, Ruaridh Mon-Williams, Neil Bramley, Christopher G. Lucas, Natalia Velez, Thomas L. Griffiths - [[pdf]](https://arxiv.org/pdf/2503.15703)</summary>

**Abstract:** When should we encourage specialization in multi-agent systems versus train generalists that perform the entire task independently? We propose that specialization largely depends on task parallelizability: the potential for multiple agents to execute task components concurrently. Drawing inspiration from Amdahl's Law in distributed systems, we present a closed-form bound that predicts when specialization improves performance, depending only on task concurrency and team size. We validate our model on two standard MARL benchmarks that represent opposite regimes -- StarCraft Multi-Agent Challenge (SMAC, unlimited concurrency) and Multi-Particle Environment (MPE, unit-capacity bottlenecks) -- and observe close alignment between the bound at each extreme and an empirical measure of specialization. Three follow-up experiments in Overcooked-AI demonstrate that the model works in environments with more complex spatial and resource bottlenecks that allow for a range of strategies. Beyond prediction, the bound also serves as a diagnostic tool, highlighting biases in MARL training algorithms that cause sub-optimal convergence to specialist strategies with larger state spaces.

**arXiv ID:** 2503.15703
</details>

<details>
<summary><strong>LEED: A Highly Efficient and Scalable LLM-Empowered Expert Demonstrations Framework for Multi-Agent Reinforcement Learning</strong> - Tianyang Duan, Zongyuan Zhang, Songxiao Guo, Dong Huang, Yuanye Zhao, Zheng Lin, Zihan Fang, Dianxin Luan, Heming Cui, Yong Cui - [[pdf]](https://arxiv.org/pdf/2509.14680)</summary>

**Abstract:** Multi-agent reinforcement learning (MARL) holds substantial promise for intelligent decision-making in complex environments. However, it suffers from a coordination and scalability bottleneck as the number of agents increases. To address these issues, we propose the LLM-empowered expert demonstrations framework for multi-agent reinforcement learning (LEED). LEED consists of two components: a demonstration generation (DG) module and a policy optimization (PO) module. Specifically, the DG module leverages large language models to generate instructions for interacting with the environment, thereby producing high-quality demonstrations. The PO module adopts a decentralized training paradigm, where each agent utilizes the generated demonstrations to construct an expert policy loss, which is then integrated with its own policy loss. This enables each agent to effectively personalize and optimize its local policy based on both expert knowledge and individual experience. Experimental results show that LEED achieves superior sample efficiency, time efficiency, and robust scalability compared to state-of-the-art baselines.

**arXiv ID:** 2509.14680
</details>

<details>
<summary><strong>Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning</strong> - Xuefeng Wang, Lei Zhang, Henglin Pu, Ahmed H. Qureshi, Husheng Li - [[pdf]](https://arxiv.org/pdf/2509.09135)</summary>

**Abstract:** Existing reinforcement learning (RL) methods struggle with complex dynamical systems that demand interactions at high frequencies or irregular time intervals. Continuous-time RL (CTRL) has emerged as a promising alternative by replacing discrete-time Bellman recursion with differential value functions defined as viscosity solutions of the Hamilton--Jacobi--Bellman (HJB) equation. While CTRL has shown promise, its applications have been largely limited to the single-agent domain. This limitation stems from two key challenges: (i) conventional solution methods for HJB equations suffer from the curse of dimensionality (CoD), making them intractable in high-dimensional systems; and (ii) even with HJB-based learning approaches, accurately approximating centralized value functions in multi-agent settings remains difficult, which in turn destabilizes policy training. In this paper, we propose a CT-MARL framework that uses physics-informed neural networks (PINNs) to approximate HJB-based value functions at scale. To ensure the value is consistent with its differential structure, we align value learning with value-gradient learning by introducing a Value Gradient Iteration (VGI) module that iteratively refines value gradients along trajectories. This improves gradient fidelity, in turn yielding more accurate values and stronger policy learning. We evaluate our method using continuous-time variants of standard benchmarks, including multi-agent particle environment (MPE) and multi-agent MuJoCo. Our results demonstrate that our approach consistently outperforms existing continuous-time RL baselines and scales to complex multi-agent dynamics.

**arXiv ID:** 2509.09135
</details>

<details>
<summary><strong>LLM Agents at the Roundtable: A Multi-Perspective and Dialectical Reasoning Framework for Essay Scoring</strong> - Jinhee Jang, Ayoung Moon, Minkyoung Jung, YoungBin Kim. Seung Jin Lee - [[pdf]](https://arxiv.org/pdf/2509.14834)</summary>

**Abstract:** The emergence of large language models (LLMs) has brought a new paradigm to automated essay scoring (AES), a long-standing and practical application of natural language processing in education. However, achieving human-level multi-perspective understanding and judgment remains a challenge. In this work, we propose Roundtable Essay Scoring (RES), a multi-agent evaluation framework designed to perform precise and human-aligned scoring under a zero-shot setting. RES constructs evaluator agents based on LLMs, each tailored to a specific prompt and topic context. Each agent independently generates a trait-based rubric and conducts a multi-perspective evaluation. Then, by simulating a roundtable-style discussion, RES consolidates individual evaluations through a dialectical reasoning process to produce a final holistic score that more closely aligns with human evaluation. By enabling collaboration and consensus among agents with diverse evaluation perspectives, RES outperforms prior zero-shot AES approaches. Experiments on the ASAP dataset using ChatGPT and Claude show that RES achieves up to a 34.86% improvement in average QWK over straightforward prompting (Vanilla) methods.

**arXiv ID:** 2509.14834
</details>

<details>
<summary><strong>The Art of Storytelling: Multi-Agent Generative AI for Dynamic Multimodal Narratives</strong> - Samee Arif, Taimoor Arif, Muhammad Saad Haroon, Aamina Jamal Khan, Agha Ali Raza, Awais Athar - [[pdf]](https://arxiv.org/pdf/2409.11261)</summary>

**Abstract:** This paper introduces the concept of an education tool that utilizes Generative Artificial Intelligence (GenAI) to enhance storytelling. We evaluate GenAI-driven narrative co-creation, text-to-speech conversion, text-to-music and text-to-video generation to produce an engaging experience for learners. We describe the co-creation process, the adaptation of narratives into spoken words using text-to-speech models, and the transformation of these narratives into contextually relevant visuals through text-to-video technology. Our evaluation covers the linguistics of the generated stories, the text-to-speech conversion quality, and the accuracy of the generated visuals.

**arXiv ID:** 2409.11261
</details>

<details>
<summary><strong>Resolve Highway Conflict in Multi-Autonomous Vehicle Controls with Local State Attention</strong> - Xuan Duy Ta, Bang Giang Le, Thanh Ha Le, Viet Cuong Ta - [[pdf]](https://arxiv.org/pdf/2506.11445)</summary>

**Abstract:** In mixed-traffic environments, autonomous vehicles must adapt to human-controlled vehicles and other unusual driving situations. This setting can be framed as a multi-agent reinforcement learning (MARL) environment with full cooperative reward among the autonomous vehicles. While methods such as Multi-agent Proximal Policy Optimization can be effective in training MARL tasks, they often fail to resolve local conflict between agents and are unable to generalize to stochastic events. In this paper, we propose a Local State Attention module to assist the input state representation. By relying on the self-attention operator, the module is expected to compress the essential information of nearby agents to resolve the conflict in traffic situations. Utilizing a simulated highway merging scenario with the priority vehicle as the unexpected event, our approach is able to prioritize other vehicles' information to manage the merging process. The results demonstrate significant improvements in merging efficiency compared to popular baselines, especially in high-density traffic settings.

**arXiv ID:** 2506.11445
</details>

<details>
<summary><strong>A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks</strong> - S M Asif Hossain, Ruksat Khan Shayoni, Mohd Ruhul Ameen, Akif Islam, M. F. Mridha, Jungpil Shin - [[pdf]](https://arxiv.org/pdf/2509.14285)</summary>

**Abstract:** Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.

**arXiv ID:** 2509.14285
</details>

<details>
<summary><strong>Traffic Co-Simulation Framework Empowered by Infrastructure Camera Sensing and Reinforcement Learning</strong> - Talha Azfar, Kaicong Huang, Andrew Tracy, Sandra Misiewicz, Chenxi Liu, Ruimin Ke - [[pdf]](https://arxiv.org/pdf/2412.03925)</summary>

**Abstract:** Traffic simulations are commonly used to optimize urban traffic flow, with reinforcement learning (RL) showing promising potential for automated traffic signal control, particularly in intelligent transportation systems involving connected automated vehicles. Multi-agent reinforcement learning (MARL) is particularly effective for learning control strategies for traffic lights in a network using iterative simulations. However, existing methods often assume perfect vehicle detection, which overlooks real-world limitations related to infrastructure availability and sensor reliability. This study proposes a co-simulation framework integrating CARLA and SUMO, which combines high-fidelity 3D modeling with large-scale traffic flow simulation. Cameras mounted on traffic light poles within the CARLA environment use a YOLO-based computer vision system to detect and count vehicles, providing real-time traffic data as input for adaptive signal control in SUMO. MARL agents trained with four different reward structures leverage this visual feedback to optimize signal timings and improve network-wide traffic flow. Experiments in a multi-intersection test-bed demonstrate the effectiveness of the proposed MARL approach in enhancing traffic conditions using real-time camera based detection. The framework also evaluates the robustness of MARL under faulty or sparse sensing and compares the performance of YOLOv5 and YOLOv8 for vehicle detection. Results show that while better accuracy improves performance, MARL agents can still achieve significant improvements with imperfect detection, demonstrating scalability and adaptability for real-world scenarios.

**arXiv ID:** 2412.03925
</details>

<details>
<summary><strong>AEGIS: Automated Error Generation and Identification for Multi-Agent Systems</strong> - Fanqi Kong, Ruijie Zhang, Huaxiao Yin, Guibin Zhang, Xiaofei Zhang, Ziang Chen, Zhaowei Zhang, Xiaoyuan Zhang, Song-Chun Zhu, Xue Feng - [[pdf]](https://arxiv.org/pdf/2509.14295)</summary>

**Abstract:** As Multi-Agent Systems (MAS) become increasingly autonomous and complex, understanding their error modes is critical for ensuring their reliability and safety. However, research in this area has been severely hampered by the lack of large-scale, diverse datasets with precise, ground-truth error labels. To address this bottleneck, we introduce \textbf{AEGIS}, a novel framework for \textbf{A}utomated \textbf{E}rror \textbf{G}eneration and \textbf{I}dentification for Multi-Agent \textbf{S}ystems. By systematically injecting controllable and traceable errors into initially successful trajectories, we create a rich dataset of realistic failures. This is achieved using a context-aware, LLM-based adaptive manipulator that performs sophisticated attacks like prompt injection and response corruption to induce specific, predefined error modes. We demonstrate the value of our dataset by exploring three distinct learning paradigms for the error identification task: Supervised Fine-Tuning, Reinforcement Learning, and Contrastive Learning. Our comprehensive experiments show that models trained on AEGIS data achieve substantial improvements across all three learning paradigms. Notably, several of our fine-tuned models demonstrate performance competitive with or superior to proprietary systems an order of magnitude larger, validating our automated data generation framework as a crucial resource for developing more robust and interpretable multi-agent systems. Our project website is available at this https URL.

**arXiv ID:** 2509.14295
</details>

<details>
<summary><strong>CRAFT: Coaching Reinforcement Learning Autonomously using Foundation Models for Multi-Robot Coordination Tasks</strong> - Seoyeon Choi, Kanghyun Ryu, Jonghoon Ock, Negar Mehr - [[pdf]](https://arxiv.org/pdf/2509.14380)</summary>

**Abstract:** Multi-Agent Reinforcement Learning (MARL) provides a powerful framework for learning coordination in multi-agent systems. However, applying MARL to robotics still remains challenging due to high-dimensional continuous joint action spaces, complex reward design, and non-stationary transitions inherent to decentralized settings. On the other hand, humans learn complex coordination through staged curricula, where long-horizon behaviors are progressively built upon simpler skills. Motivated by this, we propose CRAFT: Coaching Reinforcement learning Autonomously using Foundation models for multi-robot coordination Tasks, a framework that leverages the reasoning capabilities of foundation models to act as a "coach" for multi-robot coordination. CRAFT automatically decomposes long-horizon coordination tasks into sequences of subtasks using the planning capability of Large Language Models (LLMs). In what follows, CRAFT trains each subtask using reward functions generated by LLM, and refines them through a Vision Language Model (VLM)-guided reward-refinement loop. We evaluate CRAFT on multi-quadruped navigation and bimanual manipulation tasks, demonstrating its capability to learn complex coordination behaviors. In addition, we validate the multi-quadruped navigation policy in real hardware experiments.

**arXiv ID:** 2509.14380
</details>

</details>

<details open>
<summary><h2>Other Agent Research (5 papers)</h2></summary>

<details>
<summary><strong>Trustless Autonomy: Understanding Motivations, Benefits, and Governance Dilemmas in Self-Sovereign Decentralized AI Agents</strong> - Botao Amber Hu, Yuhan Liu, Helena Rong - [[pdf]](https://arxiv.org/pdf/2505.09757)</summary>

**Abstract:** The recent trend of self-sovereign Decentralized AI Agents (DeAgents) combines Large Language Model (LLM)-based AI agents with decentralization technologies such as blockchain smart contracts and trusted execution environments (TEEs). These tamper-resistant trustless substrates allow agents to achieve self-sovereignty through ownership of cryptowallet private keys and control of digital assets and social media accounts. DeAgents eliminate centralized control and reduce human intervention, addressing key trust concerns inherent in centralized AI systems. This contributes to social computing by enabling new human cooperative paradigm "intelligence as commons." However, given ongoing challenges in LLM reliability such as hallucinations, this creates paradoxical tension between trustlessness and unreliable autonomy. This study addresses this empirical research gap through interviews with DeAgents stakeholders-experts, founders, and developers-to examine their motivations, benefits, and governance dilemmas. The findings will guide future DeAgents system and protocol design and inform discussions about governance in sociotechnical AI systems in the future agentic web.

**arXiv ID:** 2505.09757
</details>

<details>
<summary><strong>Nonlinear Cooperative Salvo Guidance with Seeker-Limited Interceptors</strong> - Lohitvel Gopikannan, Shashi Ranjan Kumar, Abhinav Sinha - [[pdf]](https://arxiv.org/pdf/2509.15136)</summary>

**Abstract:** This paper presents a cooperative guidance strategy for the simultaneous interception of a constant-velocity, non-maneuvering target, addressing the realistic scenario where only a subset of interceptors are equipped with onboard seekers. To overcome the resulting heterogeneity in target observability, a fixed-time distributed observer is employed, enabling seeker-less interceptors to estimate the target state using information from seeker-equipped agents and local neighbors over a directed communication topology. Departing from conventional strategies that approximate time-to-go via linearization or small-angle assumptions, the proposed approach leverages deviated pursuit guidance where the time-to-go expression is exact for such a target. Moreover, a higher-order sliding mode consensus protocol is utilized to establish time-to-go consensus within a finite time. The effectiveness of the proposed guidance and estimation architecture is demonstrated through simulations.

**arXiv ID:** 2509.15136
</details>

<details>
<summary><strong>Deep Learning Agents Trained For Avoidance Behave Like Hawks And Doves</strong> - Aryaman Reddi - [[pdf]](https://arxiv.org/pdf/2503.11452)</summary>

**Abstract:** We present heuristically optimal strategies expressed by deep learning agents playing a simple avoidance game. We analyse the learning and behaviour of two agents within a symmetrical grid world that must cross paths to reach a target destination without crashing into each other or straying off of the grid world in the wrong direction. The agent policy is determined by one neural network that is employed in both agents. Our findings indicate that the fully trained network exhibits behaviour similar to that of the game Hawks and Doves, in that one agent employs an aggressive strategy to reach the target while the other learns how to avoid the aggressive agent.

**arXiv ID:** 2503.11452
</details>

<details>
<summary><strong>Multi-Quadruped Cooperative Object Transport: Learning Decentralized Pinch-Lift-Move</strong> - Bikram Pandit, Aayam Kumar Shrestha, Alan Fern - [[pdf]](https://arxiv.org/pdf/2509.14342)</summary>

**Abstract:** We study decentralized cooperative transport using teams of N-quadruped robots with arm that must pinch, lift, and move ungraspable objects through physical contact alone. Unlike prior work that relies on rigid mechanical coupling between robots and objects, we address the more challenging setting where mechanically independent robots must coordinate through contact forces alone without any communication or centralized control. To this end, we employ a hierarchical policy architecture that separates base locomotion from arm control, and propose a constellation reward formulation that unifies position and orientation tracking to enforce rigid contact behavior. The key insight is encouraging robots to behave as if rigidly connected to the object through careful reward design and training curriculum rather than explicit mechanical constraints. Our approach enables coordination through shared policy parameters and implicit synchronization cues - scaling to arbitrary team sizes without retraining. We show extensive simulation experiments to demonstrate robust transport across 2-10 robots on diverse object geometries and masses, along with sim2real transfer results on lightweight objects.

**arXiv ID:** 2509.14342
</details>

<details>
<summary><strong>Why Johnny Can't Use Agents: Industry Aspirations vs. User Realities with AI Agent Software</strong> - Pradyumna Shome, Sashreek Krishnan, Sauvik Das - [[pdf]](https://arxiv.org/pdf/2509.14528)</summary>

**Abstract:** There is growing imprecision about what "AI agents" are, what they can do, and how effectively they can be used by their intended users. We pose two key research questions: (i) How does the tech industry conceive of and market "AI agents"? (ii) What challenges do end-users face when attempting to use commercial AI agents for their advertised uses? We first performed a systematic review of marketed use cases for 102 commercial AI agents, finding that they fall into three umbrella categories: orchestration, creation, and insight. Next, we conducted a usability assessment where N = 31 participants attempted representative tasks for each of these categories on two popular commercial AI agent tools: Operator and Manus. We found that users were generally impressed with these agents but faced several critical usability challenges ranging from agent capabilities that were misaligned with user mental models to agents lacking the meta-cognitive abilities necessary for effective collaboration.

**arXiv ID:** 2509.14528
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (15 papers)</h2></summary>

<details>
<summary><strong>RationAnomaly: Log Anomaly Detection with Rationality via Chain-of-Thought and Reinforcement Learning</strong> - Song Xu, Yilun Liu, Minggui He, Mingchen Dai, Ziang Chen, Chunguang Zhao, Jingzhou Du, Shimin Tao, Weibin Meng, Shenglin Zhang, Yongqian Sun, Boxing Chen, Daimeng Wei - [[pdf]](https://arxiv.org/pdf/2509.14693)</summary>

**Abstract:** Logs constitute a form of evidence signaling the operational status of software systems. Automated log anomaly detection is crucial for ensuring the reliability of modern software systems. However, existing approaches face significant limitations: traditional deep learning models lack interpretability and generalization, while methods leveraging Large Language Models are often hindered by unreliability and factual inaccuracies. To address these issues, we propose RationAnomaly, a novel framework that enhances log anomaly detection by synergizing Chain-of-Thought (CoT) fine-tuning with reinforcement learning. Our approach first instills expert-like reasoning patterns using CoT-guided supervised fine-tuning, grounded in a high-quality dataset corrected through a rigorous expert-driven process. Subsequently, a reinforcement learning phase with a multi-faceted reward function optimizes for accuracy and logical consistency, effectively mitigating hallucinations. Experimentally, RationAnomaly outperforms state-of-the-art baselines, achieving superior F1-scores on key benchmarks while providing transparent, step-by-step analytical outputs. We have released the corresponding resources, including code and datasets.

**arXiv ID:** 2509.14693
</details>

<details>
<summary><strong>Near-Real-Time Resource Slicing for QoS Optimization in 5G O-RAN using Deep Reinforcement Learning</strong> - Peihao Yan, Jie Lu, Huacheng Zeng, Y. Thomas Hou - [[pdf]](https://arxiv.org/pdf/2509.14343)</summary>

**Abstract:** Open-Radio Access Network (O-RAN) has become an important paradigm for 5G and beyond radio access networks. This paper presents an xApp called xSlice for the Near-Real-Time (Near-RT) RAN Intelligent Controller (RIC) of 5G O-RANs. xSlice is an online learning algorithm that adaptively adjusts MAC-layer resource allocation in response to dynamic network states, including time-varying wireless channel conditions, user mobility, traffic fluctuations, and changes in user demand. To address these network dynamics, we first formulate the Quality-of-Service (QoS) optimization problem as a regret minimization problem by quantifying the QoS demands of all traffic sessions through weighting their throughput, latency, and reliability. We then develop a deep reinforcement learning (DRL) framework that utilizes an actor-critic model to combine the advantages of both value-based and policy-based updating methods. A graph convolutional network (GCN) is incorporated as a component of the DRL framework for graph embedding of RAN data, enabling xSlice to handle a dynamic number of traffic sessions. We have implemented xSlice on an O-RAN testbed with 10 smartphones and conducted extensive experiments to evaluate its performance in realistic scenarios. Experimental results show that xSlice can reduce performance regret by 67% compared to the state-of-the-art solutions. Source code is available on GitHub [1].

**arXiv ID:** 2509.14343
</details>

<details>
<summary><strong>Embodied sensorimotor control: computational modeling of the neural control of movement</strong> - Muhammad Noman Almani, John Lazzari, Jeff Walker, Shreya Saxena - [[pdf]](https://arxiv.org/pdf/2509.14360)</summary>

**Abstract:** We review how sensorimotor control is dictated by interacting neural populations, optimal feedback mechanisms, and the biomechanics of bodies. First, we outline the distributed anatomical loops that shuttle sensorimotor signals between cortex, subcortical regions, and spinal cord. We then summarize evidence that neural population activity occupies low-dimensional, dynamically evolving manifolds during planning and execution of movements. Next, we summarize literature explaining motor behavior through the lens of optimal control theory, which clarifies the role of internal models and feedback during motor control. Finally, recent studies on embodied sensorimotor control address gaps within each framework by aiming to elucidate neural population activity through the explicit control of musculoskeletal dynamics. We close by discussing open problems and opportunities: multi-tasking and cognitively rich behavior, multi-regional circuit models, and the level of anatomical detail needed in body and network models. Together, this review and recent advances point towards reaching an integrative account of the neural control of movement.

**arXiv ID:** 2509.14360
</details>

<details>
<summary><strong>Process-Supervised Reinforcement Learning for Interactive Multimodal Tool-Use Agents</strong> - Weiting Tan, Xinghua Qu, Ming Tu, Meng Ge, Andy T. Liu, Philipp Koehn, Lu Lu - [[pdf]](https://arxiv.org/pdf/2509.14480)</summary>

**Abstract:** Effective interactive tool use requires agents to master Tool Integrated Reasoning (TIR): a complex process involving multi-turn planning and long-context dialogue management. To train agents for this dynamic process, particularly in multi-modal contexts, we introduce a sandbox environment for reinforcement learning (RL) that supports interleaved speech-text rollouts. Our core strategy, Turn-level Adjudicated Reinforcement Learning (TARL), addresses the challenge of credit assignment in long-horizon tasks by employing a Large Language Model (LLM) as a judge to provide turn-level evaluation. To enhance exploration, we integrate a mixed-task training curriculum with mathematical reasoning problems. This unified approach boosts the task pass rate on the text-based $\tau$-bench by over 6% compared to strong RL baselines. Crucially, we demonstrate our framework's suitability for fine-tuning a multi-modal foundation model for agentic tasks. By training a base multi-modal LLM on interleaved speech-text rollouts, we equip it with tool-use abilities, paving the way for more natural, voice-driven interactive agents.

**arXiv ID:** 2509.14480
</details>

<details>
<summary><strong>Empathy-R1: A Chain-of-Empathy and Reinforcement Learning Framework for Long-Form Mental Health Support</strong> - Xianrong Yao, Dong She, Chenxu Zhang, Yimeng Zhang, Yueru Sun, Noman Ahmed, Yang Gao, Zhanpeng Jin - [[pdf]](https://arxiv.org/pdf/2509.14851)</summary>

**Abstract:** Empathy is critical for effective mental health support, especially when addressing Long Counseling Texts (LCTs). However, existing Large Language Models (LLMs) often generate replies that are semantically fluent but lack the structured reasoning necessary for genuine psychological support, particularly in a Chinese context. To bridge this gap, we introduce Empathy-R1, a novel framework that integrates a Chain-of-Empathy (CoE) reasoning process with Reinforcement Learning (RL) to enhance response quality for LCTs. Inspired by cognitive-behavioral therapy, our CoE paradigm guides the model to sequentially reason about a help-seeker's emotions, causes, and intentions, making its thinking process both transparent and interpretable. Our framework is empowered by a new large-scale Chinese dataset, Empathy-QA, and a two-stage training process. First, Supervised Fine-Tuning instills the CoE's reasoning structure. Subsequently, RL, guided by a dedicated reward model, refines the therapeutic relevance and contextual appropriateness of the final responses. Experiments show that Empathy-R1 achieves strong performance on key automatic metrics. More importantly, human evaluations confirm its superiority, showing a clear preference over strong baselines and achieving a Win@1 rate of 44.30% on our new benchmark. By enabling interpretable and contextually nuanced responses, Empathy-R1 represents a significant advancement in developing responsible and genuinely beneficial AI for mental health support.

**arXiv ID:** 2509.14851
</details>

<details>
<summary><strong>Top K Enhanced Reinforcement Learning Attacks on Heterogeneous Graph Node Classification</strong> - Honglin Gao, Xiang Li, Yajuan Sun, Gaoxi Xiao - [[pdf]](https://arxiv.org/pdf/2408.01964)</summary>

**Abstract:** Graph Neural Networks (GNNs) have attracted substantial interest due to their exceptional performance on graph-based data. However, their robustness, especially on heterogeneous graphs, remains underexplored, particularly against adversarial attacks. This paper proposes HeteroKRLAttack, a targeted evasion black-box attack method for heterogeneous graphs. By integrating reinforcement learning with a Top-K algorithm to reduce the action space, our method efficiently identifies effective attack strategies to disrupt node classification tasks. We validate the effectiveness of HeteroKRLAttack through experiments on multiple heterogeneous graph datasets, showing significant reductions in classification accuracy compared to baseline methods. An ablation study underscores the critical role of the Top-K algorithm in enhancing attack performance. Our findings highlight potential vulnerabilities in current models and provide guidance for future defense strategies against adversarial attacks on heterogeneous graphs.

**arXiv ID:** 2408.01964
</details>

<details>
<summary><strong>A Survey of Reinforcement Learning for Large Reasoning Models</strong> - Kaiyan Zhang, Yuxin Zuo, Bingxiang He, Youbang Sun, Runze Liu, Che Jiang, Yuchen Fan, Kai Tian, Guoli Jia, Pengfei Li, Yu Fu, Xingtai Lv, Yuchen Zhang, Sihang Zeng, Shang Qu, Haozhan Li, Shijie Wang, Yuru Wang, Xinwei Long, Fangfu Liu, Xiang Xu, Jiaze Ma, Xuekai Zhu, Ermo Hua, Yihao Liu, Zonglin Li, Huayu Chen, Xiaoye Qu, Yafu Li, Weize Chen, Zhenzhao Yuan, Junqi Gao, Dong Li, Zhiyuan Ma, Ganqu Cui, Zhiyuan Liu, Biqing Qi, Ning Ding, Bowen Zhou - [[pdf]](https://arxiv.org/pdf/2509.08827)</summary>

**Abstract:** In this paper, we survey recent advances in Reinforcement Learning (RL) for reasoning with Large Language Models (LLMs). RL has achieved remarkable success in advancing the frontier of LLM capabilities, particularly in addressing complex logical tasks such as mathematics and coding. As a result, RL has emerged as a foundational methodology for transforming LLMs into LRMs. With the rapid progress of the field, further scaling of RL for LRMs now faces foundational challenges not only in computational resources but also in algorithm design, training data, and infrastructure. To this end, it is timely to revisit the development of this domain, reassess its trajectory, and explore strategies to enhance the scalability of RL toward Artificial SuperIntelligence (ASI). In particular, we examine research applying RL to LLMs and LRMs for reasoning abilities, especially since the release of DeepSeek-R1, including foundational components, core problems, training resources, and downstream applications, to identify future opportunities and directions for this rapidly evolving area. We hope this review will promote future research on RL for broader reasoning models. Github: this https URL

**arXiv ID:** 2509.08827
</details>

<details>
<summary><strong>Online reinforcement learning via sparse Gaussian mixture model Q-functions</strong> - Minh Vu, Konstantinos Slavakis - [[pdf]](https://arxiv.org/pdf/2509.14585)</summary>

**Abstract:** This paper introduces a structured and interpretable online policy-iteration framework for reinforcement learning (RL), built around the novel class of sparse Gaussian mixture model Q-functions (S-GMM-QFs). Extending earlier work that trained GMM-QFs offline, the proposed framework develops an online scheme that leverages streaming data to encourage exploration. Model complexity is regulated through sparsification by Hadamard overparametrization, which mitigates overfitting while preserving expressiveness. The parameter space of S-GMM-QFs is naturally endowed with a Riemannian manifold structure, allowing for principled parameter updates via online gradient descent on a smooth objective. Numerical tests show that S-GMM-QFs match the performance of dense deep RL (DeepRL) methods on standard benchmarks while using significantly fewer parameters, and maintain strong performance even in low-parameter-count regimes where sparsified DeepRL methods fail to generalize.

**arXiv ID:** 2509.14585
</details>

<details>
<summary><strong>Multi-Fidelity Hybrid Reinforcement Learning via Information Gain Maximization</strong> - Houssem Sifaou, Osvaldo Simeone - [[pdf]](https://arxiv.org/pdf/2509.14848)</summary>

**Abstract:** Optimizing a reinforcement learning (RL) policy typically requires extensive interactions with a high-fidelity simulator of the environment, which are often costly or impractical. Offline RL addresses this problem by allowing training from pre-collected data, but its effectiveness is strongly constrained by the size and quality of the dataset. Hybrid offline-online RL leverages both offline data and interactions with a single simulator of the environment. In many real-world scenarios, however, multiple simulators with varying levels of fidelity and computational cost are available. In this work, we study multi-fidelity hybrid RL for policy optimization under a fixed cost budget. We introduce multi-fidelity hybrid RL via information gain maximization (MF-HRL-IGM), a hybrid offline-online RL algorithm that implements fidelity selection based on information gain maximization through a bootstrapping approach. Theoretical analysis establishes the no-regret property of MF-HRL-IGM, while empirical evaluations demonstrate its superior performance compared to existing benchmarks.

**arXiv ID:** 2509.14848
</details>

<details>
<summary><strong>Leveraging Reinforcement Learning, Genetic Algorithms and Transformers for background determination in particle physics</strong> - Guillermo Hijano Mendizabal, Davide Lancierini, Alex Marshall, Andrea Mauri, Patrick Haworth Owen, Mitesh Patel, Konstantinos Petridis, Shah Rukh Qasim, Nicola Serra, William Sutcliffe, Hanae Tilquin - [[pdf]](https://arxiv.org/pdf/2509.14894)</summary>

**Abstract:** Experimental studies of beauty hadron decays face significant challenges due to a wide range of backgrounds arising from the numerous possible decay channels with similar final states. For a particular signal decay, the process for ascertaining the most relevant background processes necessitates a detailed analysis of final state particles, potential misidentifications, and kinematic overlaps, which, due to computational limitations, is restricted to the simulation of only the most relevant backgrounds. Moreover, this process typically relies on the physicist's intuition and expertise, as no systematic method exists.
This paper has two primary goals. First, from a particle physics perspective, we present a novel approach that utilises Reinforcement Learning (RL) to overcome the aforementioned challenges by systematically determining the critical backgrounds affecting beauty hadron decay measurements. While beauty hadron physics serves as the case study in this work, the proposed strategy is broadly adaptable to other types of particle physics measurements. Second, from a Machine Learning perspective, we introduce a novel algorithm which exploits the synergy between RL and Genetic Algorithms (GAs) for environments with highly sparse rewards and a large trajectory space. This strategy leverages GAs to efficiently explore the trajectory space and identify successful trajectories, which are used to guide the RL agent's training. Our method also incorporates a transformer architecture for the RL agent to handle token sequences representing decays.

**arXiv ID:** 2509.14894
</details>

<details>
<summary><strong>Self-Explaining Reinforcement Learning for Mobile Network Resource Allocation</strong> - Konrad Nowosadko, Franco Ruggeri, Ahmad Terra - [[pdf]](https://arxiv.org/pdf/2509.14925)</summary>

**Abstract:** Reinforcement Learning (RL) methods that incorporate deep neural networks (DNN), though powerful, often lack transparency. Their black-box characteristic hinders interpretability and reduces trustworthiness, particularly in critical domains. To address this challenge in RL tasks, we propose a solution based on Self-Explaining Neural Networks (SENNs) along with explanation extraction methods to enhance interpretability while maintaining predictive accuracy. Our approach targets low-dimensionality problems to generate robust local and global explanations of the model's behaviour. We evaluate the proposed method on the resource allocation problem in mobile networks, demonstrating that SENNs can constitute interpretable solutions with competitive performance. This work highlights the potential of SENNs to improve transparency and trust in AI-driven decision-making for low-dimensional tasks. Our approach strong performance on par with the existing state-of-the-art methods, while providing robust explanations.

**arXiv ID:** 2509.14925
</details>

<details>
<summary><strong>Self-Improving Embodied Foundation Models</strong> - Seyed Kamyar Seyed Ghasemipour, Ayzaan Wahid, Jonathan Tompson, Pannag Sanketi, Igor Mordatch - [[pdf]](https://arxiv.org/pdf/2509.15155)</summary>

**Abstract:** Foundation models trained on web-scale data have revolutionized robotics, but their application to low-level control remains largely limited to behavioral cloning. Drawing inspiration from the success of the reinforcement learning stage in fine-tuning large language models, we propose a two-stage post-training approach for robotics. The first stage, Supervised Fine-Tuning (SFT), fine-tunes pretrained foundation models using both: a) behavioral cloning, and b) steps-to-go prediction objectives. In the second stage, Self-Improvement, steps-to-go prediction enables the extraction of a well-shaped reward function and a robust success detector, enabling a fleet of robots to autonomously practice downstream tasks with minimal human supervision. Through extensive experiments on real-world and simulated robot embodiments, our novel post-training recipe unveils significant results on Embodied Foundation Models. First, we demonstrate that the combination of SFT and Self-Improvement is significantly more sample-efficient than scaling imitation data collection for supervised learning, and that it leads to policies with significantly higher success rates. Further ablations highlight that the combination of web-scale pretraining and Self-Improvement is the key to this sample-efficiency. Next, we demonstrate that our proposed combination uniquely unlocks a capability that current methods cannot achieve: autonomously practicing and acquiring novel skills that generalize far beyond the behaviors observed in the imitation learning datasets used during training. These findings highlight the transformative potential of combining pretrained foundation models with online Self-Improvement to enable autonomous skill acquisition in robotics. Our project website can be found at this https URL .

**arXiv ID:** 2509.15155
</details>

<details>
<summary><strong>Scalable Multi-Objective Robot Reinforcement Learning through Gradient Conflict Resolution</strong> - Humphrey Munn, Brendan Tidd, Peter Böhm, Marcus Gallagher, David Howard - [[pdf]](https://arxiv.org/pdf/2509.14816)</summary>

**Abstract:** Reinforcement Learning (RL) robot controllers usually aggregate many task objectives into one scalar reward. While large-scale proximal policy optimisation (PPO) has enabled impressive results such as robust robot locomotion in the real world, many tasks still require careful reward tuning and are brittle to local optima. Tuning cost and sub-optimality grow with the number of objectives, limiting scalability. Modelling reward vectors and their trade-offs can address these issues; however, multi-objective methods remain underused in RL for robotics because of computational cost and optimisation difficulty. In this work, we investigate the conflict between gradient contributions for each objective that emerge from scalarising the task objectives. In particular, we explicitly address the conflict between task-based rewards and terms that regularise the policy towards realistic behaviour. We propose GCR-PPO, a modification to actor-critic optimisation that decomposes the actor update into objective-wise gradients using a multi-headed critic and resolves conflicts based on the objective priority. Our methodology, GCR-PPO, is evaluated on the well-known IsaacLab manipulation and locomotion benchmarks and additional multi-objective modifications on two related tasks. We show superior scalability compared to parallel PPO (p = 0.04), without significant computational overhead. We also show higher performance with more conflicting tasks. GCR-PPO improves on large-scale PPO with an average improvement of 9.5%, with high-conflict tasks observing a greater improvement. The code is available at this https URL.

**arXiv ID:** 2509.14816
</details>

<details>
<summary><strong>Robust Reinforcement Learning under Diffusion Models for Data with Jumps</strong> - Chenyang Jiang, Donggyu Kim, Alejandra Quintos, Yazhen Wang - [[pdf]](https://arxiv.org/pdf/2411.11697)</summary>

**Abstract:** Reinforcement Learning (RL) has proven effective in solving complex decision-making tasks across various domains, but challenges remain in continuous-time settings, particularly when state dynamics are governed by stochastic differential equations (SDEs) with jump components. In this paper, we address this challenge by introducing the Mean-Square Bipower Variation Error (MSBVE) algorithm, which enhances robustness and convergence in scenarios involving significant stochastic noise and jumps. We first revisit the Mean-Square TD Error (MSTDE) algorithm, commonly used in continuous-time RL, and highlight its limitations in handling jumps in state dynamics. The proposed MSBVE algorithm minimizes the mean-square quadratic variation error, offering improved performance over MSTDE in environments characterized by SDEs with jumps. Simulations and formal proofs demonstrate that the MSBVE algorithm reliably estimates the value function in complex settings, surpassing MSTDE's performance when faced with jump processes. These findings underscore the importance of alternative error metrics to improve the resilience and effectiveness of RL algorithms in continuous-time frameworks.

**arXiv ID:** 2411.11697
</details>

<details>
<summary><strong>ExT: Towards Scalable Autonomous Excavation via Large-Scale Multi-Task Pretraining and Fine-Tuning</strong> - Yifan Zhai, Lorenzo Terenzi, Patrick Frey, Diego Garcia Soto, Pascal Egli, Marco Hutter - [[pdf]](https://arxiv.org/pdf/2509.14992)</summary>

**Abstract:** Scaling up the deployment of autonomous excavators is of great economic and societal importance. Yet it remains a challenging problem, as effective systems must robustly handle unseen worksite conditions and new hardware configurations. Current state-of-the-art approaches rely on highly engineered, task-specific controllers, which require extensive manual tuning for each new scenario. In contrast, recent advances in large-scale pretrained models have shown remarkable adaptability across tasks and embodiments in domains such as manipulation and navigation, but their applicability to heavy construction machinery remains largely unexplored. In this work, we introduce ExT, a unified open-source framework for large-scale demonstration collection, pretraining, and fine-tuning of multitask excavation policies. ExT policies are first trained on large-scale demonstrations collected from a mix of experts, then fine-tuned either with supervised fine-tuning (SFT) or reinforcement learning fine-tuning (RLFT) to specialize to new tasks or operating conditions. Through both simulation and real-world experiments, we show that pretrained ExT policies can execute complete excavation cycles with centimeter-level accuracy, successfully transferring from simulation to real machine with performance comparable to specialized single-task controllers. Furthermore, in simulation, we demonstrate that ExT's fine-tuning pipelines allow rapid adaptation to new tasks, out-of-distribution conditions, and machine configurations, while maintaining strong performance on previously learned tasks. These results highlight the potential of ExT to serve as a foundation for scalable and generalizable autonomous excavation.

**arXiv ID:** 2509.14992
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
