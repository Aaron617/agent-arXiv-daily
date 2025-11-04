# Agent arXiv Daily

**Last Updated:** 2025-11-04 02:47:43

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
<summary><strong>AURA: A Reinforcement Learning Framework for AI-Driven Adaptive Conversational Surveys</strong> - Jinwen Tang, Yi Shang - [[pdf]](https://arxiv.org/pdf/2510.27126)</summary>

**Abstract:** Conventional online surveys provide limited personalization, often resulting in low engagement and superficial responses. Although AI survey chatbots improve convenience, most are still reactive: they rely on fixed dialogue trees or static prompt templates and therefore cannot adapt within a session to fit individual users, which leads to generic follow-ups and weak response quality. We address these limitations with AURA (Adaptive Understanding through Reinforcement Learning for Assessment), a reinforcement learning framework for AI-driven adaptive conversational surveys. AURA quantifies response quality using a four-dimensional LSDE metric (Length, Self-disclosure, Emotion, and Specificity) and selects follow-up question types via an epsilon-greedy policy that updates the expected quality gain within each session. Initialized with priors extracted from 96 prior campus-climate conversations (467 total chatbot-user exchanges), the system balances exploration and exploitation across 10-15 dialogue exchanges, dynamically adapting to individual participants in real time. In controlled evaluations, AURA achieved a +0.12 mean gain in response quality and a statistically significant improvement over non-adaptive baselines (p=0.044, d=0.66), driven by a 63% reduction in specification prompts and a 10x increase in validation behavior. These results demonstrate that reinforcement learning can give survey chatbots improved adaptivity, transforming static questionnaires into interactive, self-improving assessment systems.

**arXiv ID:** 2510.27126
</details>

<details>
<summary><strong>"Koyi Sawaal Nahi Hai": Reimagining Maternal Health Chatbots for Collective, Culturally Grounded Care</strong> - Imaan Hameed, Huma Umar, Fozia Umber, Maryam Mustafa - [[pdf]](https://arxiv.org/pdf/2510.27401)</summary>

**Abstract:** In recent years, LLM-based maternal health chatbots have been widely deployed in low-resource settings, but they often ignore real-world contexts where women may not own phones, have limited literacy, and share decision-making within families. Through the deployment of a WhatsApp-based maternal health chatbot with 48 pregnant women in Lahore, Pakistan, we examine barriers to use in populations where phones are shared, decision-making is collective, and literacy varies. We complement this with focus group discussions with obstetric clinicians. Our findings reveal how adoption is shaped by proxy consent and family mediation, intermittent phone access, silence around asking questions, infrastructural breakdowns, and contested authority. We frame barriers to non-use as culturally conditioned rather than individual choices, and introduce the Relational Chatbot Design Grammar (RCDG): four commitments that enable mediated decision-making, recognize silence as engagement, support episodic use, and treat fragility as baseline to reorient maternal health chatbots toward culturally grounded, collective care.

**arXiv ID:** 2510.27401
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (8 papers)</h2></summary>

<details>
<summary><strong>ToolScope: An Agentic Framework for Vision-Guided and Long-Horizon Tool Use</strong> - Mengjie Deng, Guanting Dong, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2510.27363)</summary>

**Abstract:** Recently, large language models (LLMs) have demonstrated remarkable problem-solving capabilities by autonomously integrating with external tools for collaborative reasoning. However, due to the inherently complex and diverse nature of multimodal information, enabling multimodal large language models (MLLMs) to flexibly and efficiently utilize external tools during reasoning remains an underexplored challenge. In this work, we introduce ToolScope, an agentic framework designed to unify global planning with local multimodal perception, adopting a specialized Perceive tool to mitigates visual context degradation in long-horizon VQA task. ToolScope comprises three primary components: the Global Navigator, the Agentic Executor, and the Response Synthesizer. The Global Navigator functions as a "telescope", offering high-level strategic guidance. The Agentic Executor operates iteratively to augment MLLM with local perception through the integration of external tools-Search, Code, and Perceive. Finally, the Response Synthesizer consolidates and organizes the reasoning process into a coherent, user-friendly output. We evaluate ToolScope on four VQA benchmarks across diverse domains, including VQA 2.0, ScienceQA, MAT-Search and MathVista. It demonstrates strong generalization capabilities, achieving an average performance improvement of up to +6.69% across all datasets.

**arXiv ID:** 2510.27363
</details>

<details>
<summary><strong>Visual Backdoor Attacks on MLLM Embodied Decision Making via Contrastive Trigger Learning</strong> - Qiusi Zhan, Hyeonjeong Ha, Rui Yang, Sirui Xu, Hanyang Chen, Liang-Yan Gui, Yu-Xiong Wang, Huan Zhang, Heng Ji, Daniel Kang - [[pdf]](https://arxiv.org/pdf/2510.27623)</summary>

**Abstract:** Multimodal large language models (MLLMs) have advanced embodied agents by enabling direct perception, reasoning, and planning task-oriented actions from visual inputs. However, such vision driven embodied agents open a new attack surface: visual backdoor attacks, where the agent behaves normally until a visual trigger appears in the scene, then persistently executes an attacker-specified multi-step policy. We introduce BEAT, the first framework to inject such visual backdoors into MLLM-based embodied agents using objects in the environments as triggers. Unlike textual triggers, object triggers exhibit wide variation across viewpoints and lighting, making them difficult to implant reliably. BEAT addresses this challenge by (1) constructing a training set that spans diverse scenes, tasks, and trigger placements to expose agents to trigger variability, and (2) introducing a two-stage training scheme that first applies supervised fine-tuning (SFT) and then our novel Contrastive Trigger Learning (CTL). CTL formulates trigger discrimination as preference learning between trigger-present and trigger-free inputs, explicitly sharpening the decision boundaries to ensure precise backdoor activation. Across various embodied agent benchmarks and MLLMs, BEAT achieves attack success rates up to 80%, while maintaining strong benign task performance, and generalizes reliably to out-of-distribution trigger placements. Notably, compared to naive SFT, CTL boosts backdoor activation accuracy up to 39% under limited backdoor data. These findings expose a critical yet unexplored security risk in MLLM-based embodied agents, underscoring the need for robust defenses before real-world deployment.

**arXiv ID:** 2510.27623
</details>

<details>
<summary><strong>Beyond Pixels: Exploring DOM Downsampling for LLM-Based Web Agents</strong> - Thassilo M. Schiepanski, Nicholas Piël - [[pdf]](https://arxiv.org/pdf/2508.04412)</summary>

**Abstract:** Frontier LLMs only recently enabled serviceable, autonomous web agents. At that, a model poses as an instantaneous domain model backend. Ought to suggest interaction, it is consulted with a web-based task and respective application state. The key problem lies in application state serialisation - referred to as snapshot. State-of-the-art web agents are premised on grounded GUI snapshots, i.e., screenshots enhanced with visual cues. Not least to resemble human perception, but for images representing relatively cheap means of model input. LLM vision still lag behind code interpretation capabilities. DOM snapshots, which structurally resemble HTML, impose a desired alternative. Vast model input token size, however, disables reliable implementation with web agents to date. We propose D2Snap, a first-of-its-kind DOM downsampling algorithm. Based on a GPT-4o backend, we evaluate D2Snap on tasks sampled from the Online-Mind2Web dataset. The success rate of D2Snap-downsampled DOM snapshots (67%) matches a grounded GUI snapshot baseline (65%) - within the same input token order of magnitude (1e3). Our best evaluated configurations - one token order above, but within the model's context window - outperform this baseline by 8%. Our evaluation, moreover, yields that DOM-inherent hierarchy embodies a strong UI feature for LLMs.

**arXiv ID:** 2508.04412
</details>

<details>
<summary><strong>AVA: Towards Agentic Video Analytics with Vision Language Models</strong> - Yuxuan Yan, Shiqi Jiang, Ting Cao, Yifan Yang, Qianqian Yang, Yuanchao Shu, Yuqing Yang, Lili Qiu - [[pdf]](https://arxiv.org/pdf/2505.00254)</summary>

**Abstract:** AI-driven video analytics has become increasingly important across diverse domains. However, existing systems are often constrained to specific, predefined tasks, limiting their adaptability in open-ended analytical scenarios. The recent emergence of Vision Language Models (VLMs) as transformative technologies offers significant potential for enabling open-ended video understanding, reasoning, and analytics. Nevertheless, their limited context windows present challenges when processing ultra-long video content, which is prevalent in real-world applications. To address this, we introduce AVA, a VLM-powered system designed for open-ended, advanced video analytics. AVA incorporates two key innovations: (1) the near real-time construction of Event Knowledge Graphs (EKGs) for efficient indexing of long or continuous video streams, and (2) an agentic retrieval-generation mechanism that leverages EKGs to handle complex and diverse queries. Comprehensive evaluations on public benchmarks, LVBench and VideoMME-Long, demonstrate that AVA achieves state-of-the-art performance, attaining 62.3% and 64.1% accuracy, respectively-significantly surpassing existing VLM and video Retrieval-Augmented Generation (RAG) systems. Furthermore, to evaluate video analytics in ultra-long and open-world video scenarios, we introduce a new benchmark, AVA-100. This benchmark comprises 8 videos, each exceeding 10 hours in duration, along with 120 manually annotated, diverse, and complex question-answer pairs. On AVA-100, AVA achieves top-tier performance with an accuracy of 75.8%. The source code of AVA is available at this https URL. The AVA-100 benchmark can be accessed at this https URL.

**arXiv ID:** 2505.00254
</details>

<details>
<summary><strong>NaviTrace: Evaluating Embodied Navigation of Vision-Language Models</strong> - Tim Windecker, Manthan Patel, Moritz Reuss, Richard Schwarzkopf, Cesar Cadena, Rudolf Lioutikov, Marco Hutter, Jonas Frey - [[pdf]](https://arxiv.org/pdf/2510.26909)</summary>

**Abstract:** Vision-language models demonstrate unprecedented performance and generalization across a wide range of tasks and scenarios. Integrating these foundation models into robotic navigation systems opens pathways toward building general-purpose robots. Yet, evaluating these models' navigation capabilities remains constrained by costly real-world trials, overly simplified simulations, and limited benchmarks. We introduce NaviTrace, a high-quality Visual Question Answering benchmark where a model receives an instruction and embodiment type (human, legged robot, wheeled robot, bicycle) and must output a 2D navigation trace in image space. Across 1000 scenarios and more than 3000 expert traces, we systematically evaluate eight state-of-the-art VLMs using a newly introduced semantic-aware trace score. This metric combines Dynamic Time Warping distance, goal endpoint error, and embodiment-conditioned penalties derived from per-pixel semantics and correlates with human preferences. Our evaluation reveals consistent gap to human performance caused by poor spatial grounding and goal localization. NaviTrace establishes a scalable and reproducible benchmark for real-world robotic navigation. The benchmark and leaderboard can be found at this https URL.

**arXiv ID:** 2510.26909
</details>

<details>
<summary><strong>Modified-Emergency Index (MEI): A Criticality Metric for Autonomous Driving in Lateral Conflict</strong> - Hao Cheng, Yanbo Jiang, Qingyuan Shi, Qingwen Meng, Keyu Chen, Wenhao Yu, Jianqiang Wang, Sifa Zheng - [[pdf]](https://arxiv.org/pdf/2510.27333)</summary>

**Abstract:** Effective, reliable, and efficient evaluation of autonomous driving safety is essential to demonstrate its trustworthiness. Criticality metrics provide an objective means of assessing safety. However, as existing metrics primarily target longitudinal conflicts, accurately quantifying the risks of lateral conflicts - prevalent in urban settings - remains challenging. This paper proposes the Modified-Emergency Index (MEI), a metric designed to quantify evasive effort in lateral conflicts. Compared to the original Emergency Index (EI), MEI refines the estimation of the time available for evasive maneuvers, enabling more precise risk quantification. We validate MEI on a public lateral conflict dataset based on Argoverse-2, from which we extract over 1,500 high-quality AV conflict cases, including more than 500 critical events. MEI is then compared with the well-established ACT and the widely used PET metrics. Results show that MEI consistently outperforms them in accurately quantifying criticality and capturing risk evolution. Overall, these findings highlight MEI as a promising metric for evaluating urban conflicts and enhancing the safety assessment framework for autonomous driving. The open-source implementation is available at this https URL.

**arXiv ID:** 2510.27333
</details>

<details>
<summary><strong>Towards a Multi-Embodied Grasping Agent</strong> - Roman Freiberg, Alexander Qualmann, Ngo Anh Vien, Gerhard Neumann - [[pdf]](https://arxiv.org/pdf/2510.27420)</summary>

**Abstract:** Multi-embodiment grasping focuses on developing approaches that exhibit generalist behavior across diverse gripper designs. Existing methods often learn the kinematic structure of the robot implicitly and face challenges due to the difficulty of sourcing the required large-scale data. In this work, we present a data-efficient, flow-based, equivariant grasp synthesis architecture that can handle different gripper types with variable degrees of freedom and successfully exploit the underlying kinematic model, deducing all necessary information solely from the gripper and scene geometry. Unlike previous equivariant grasping methods, we translated all modules from the ground up to JAX and provide a model with batching capabilities over scenes, grippers, and grasps, resulting in smoother learning, improved performance and faster inference time. Our dataset encompasses grippers ranging from humanoid hands to parallel yaw grippers and includes 25,000 scenes and 20 million grasps.

**arXiv ID:** 2510.27420
</details>

<details>
<summary><strong>Panoramic Out-of-Distribution Segmentation for Autonomous Driving</strong> - Mengfei Duan, Yuheng Zhang, Yihong Cao, Fei Teng, Kai Luo, Jiaming Zhang, Kailun Yang, Zhiyong Li - [[pdf]](https://arxiv.org/pdf/2505.03539)</summary>

**Abstract:** Panoramic imaging enables capturing 360° images with an ultra-wide Field-of-View (FoV) for dense omnidirectional perception, which is critical to applications, such as autonomous driving and augmented reality, etc. However, current panoramic semantic segmentation methods fail to identify outliers, and pinhole Out-of-distribution Segmentation (OoS) models perform unsatisfactorily in the panoramic domain due to background clutter and pixel distortions. To address these issues, we introduce a new task, Panoramic Out-of-distribution Segmentation (PanOoS), with the aim of achieving comprehensive and safe scene understanding. Furthermore, we propose the first solution, POS, which adapts to the characteristics of panoramic images through text-guided prompt distribution learning. Specifically, POS integrates a disentanglement strategy designed to materialize the cross-domain generalization capability of CLIP. The proposed Prompt-based Restoration Attention (PRA) optimizes semantic decoding by prompt guidance and self-adaptive correction, while Bilevel Prompt Distribution Learning (BPDL) refines the manifold of per-pixel mask embeddings via semantic prototype supervision. Besides, to compensate for the scarcity of PanOoS datasets, we establish two benchmarks: DenseOoS, which features diverse outliers in complex environments, and QuadOoS, captured by a quadruped robot with a panoramic annular lens system. Extensive experiments demonstrate superior performance of POS, with AuPRC improving by 34.25% and FPR95 decreasing by 21.42% on DenseOoS, outperforming state-of-the-art pinhole-OoS methods. Moreover, POS achieves leading closed-set segmentation capabilities and advances the development of panoramic understanding. Code and datasets will be available at this https URL.

**arXiv ID:** 2505.03539
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>CATArena: Evaluation of LLM Agents through Iterative Tournament Competitions</strong> - Lingyue Fu, Xin Ding, Yaoming Zhu, Shao Zhang, Lin Qiu, Weiwen Liu, Weinan Zhang, Xuezhi Cao, Xunliang Cai, Jiaxin Ding, Yong Yu - [[pdf]](https://arxiv.org/pdf/2510.26852)</summary>

**Abstract:** Large Language Model (LLM) agents have evolved from basic text generation to autonomously completing complex tasks through interaction with external tools. However, current benchmarks mainly assess end-to-end performance in fixed scenarios, restricting evaluation to specific skills and suffering from score saturation and growing dependence on expert annotation as agent capabilities improve. In this work, we emphasize the importance of learning ability, including both self-improvement and peer-learning, as a core driver for agent evolution toward human-level intelligence. We propose an iterative, competitive peer-learning framework, which allows agents to refine and optimize their strategies through repeated interactions and feedback, thereby systematically evaluating their learning capabilities. To address the score saturation issue in current benchmarks, we introduce CATArena, a tournament-style evaluation platform featuring four diverse board and card games with open-ended scoring. By providing tasks without explicit upper score limits, CATArena enables continuous and dynamic evaluation of rapidly advancing agent capabilities. Experimental results and analyses involving both minimal and commercial code agents demonstrate that CATArena provides reliable, stable, and scalable benchmarking for core agent abilities, particularly learning ability and strategy coding.

**arXiv ID:** 2510.26852
</details>

<details>
<summary><strong>Can LLMs Help You at Work? A Sandbox for Evaluating LLM Agents in Enterprise Environments</strong> - Harsh Vishwakarma, Ankush Agarwal, Ojas Patil, Chaitanya Devaguptapu, Mahesh Chandran - [[pdf]](https://arxiv.org/pdf/2510.27287)</summary>

**Abstract:** Enterprise systems are crucial for enhancing productivity and decision-making among employees and customers. Integrating LLM based systems into enterprise systems enables intelligent automation, personalized experiences, and efficient information retrieval, driving operational efficiency and strategic growth. However, developing and evaluating such systems is challenging due to the inherent complexity of enterprise environments, where data is fragmented across multiple sources and governed by sophisticated access controls. We present EnterpriseBench, a comprehensive benchmark that simulates enterprise settings, featuring 500 diverse tasks across software engineering, HR, finance, and administrative domains. Our benchmark uniquely captures key enterprise characteristics including data source fragmentation, access control hierarchies, and cross-functional workflows. Additionally, we provide a novel data generation pipeline that creates internally consistent enterprise tasks from organizational metadata. Experiments with state-of-the-art LLM agents demonstrate that even the most capable models achieve only 41.8% task completion, highlighting significant opportunities for improvement in enterprise-focused AI systems.

**arXiv ID:** 2510.27287
</details>

<details>
<summary><strong>Semantically-Aware LLM Agent to Enhance Privacy in Conversational AI Services</strong> - Jayden Serenari, Stephen Lee - [[pdf]](https://arxiv.org/pdf/2510.27016)</summary>

**Abstract:** With the increasing use of conversational AI systems, there is growing concern over privacy leaks, especially when users share sensitive personal data in interactions with Large Language Models (LLMs). Conversations shared with these models may contain Personally Identifiable Information (PII), which, if exposed, could lead to security breaches or identity theft. To address this challenge, we present the Local Optimizations for Pseudonymization with Semantic Integrity Directed Entity Detection (LOPSIDED) framework, a semantically-aware privacy agent designed to safeguard sensitive PII data when using remote LLMs. Unlike prior work that often degrade response quality, our approach dynamically replaces sensitive PII entities in user prompts with semantically consistent pseudonyms, preserving the contextual integrity of conversations. Once the model generates its response, the pseudonyms are automatically depseudonymized, ensuring the user receives an accurate, privacy-preserving output. We evaluate our approach using real-world conversations sourced from ShareGPT, which we further augment and annotate to assess whether named entities are contextually relevant to the model's response. Our results show that LOPSIDED reduces semantic utility errors by a factor of 5 compared to baseline techniques, all while enhancing privacy.

**arXiv ID:** 2510.27016
</details>

<details>
<summary><strong>Dynamic Affective Memory Management for Personalized LLM Agents</strong> - Junfeng Lu, Yueyan Li - [[pdf]](https://arxiv.org/pdf/2510.27418)</summary>

**Abstract:** Advances in large language models are making personalized AI agents a new research focus. While current agent systems primarily rely on personalized external memory databases to deliver customized experiences, they face challenges such as memory redundancy, memory staleness, and poor memory-context integration, largely due to the lack of effective memory updates during interaction. To tackle these issues, we propose a new memory management system designed for affective scenarios. Our approach employs a Bayesian-inspired memory update algorithm with the concept of memory entropy, enabling the agent to autonomously maintain a dynamically updated memory vector database by minimizing global entropy to provide more personalized services. To better evaluate the system's effectiveness in this context, we propose DABench, a benchmark focusing on emotional expression and emotional change toward objects. Experimental results demonstrate that, our system achieves superior performance in personalization, logical coherence, and accuracy. Ablation studies further validate the effectiveness of the Bayesian-inspired update mechanism in alleviating memory bloat. Our work offers new insights into the design of long-term memory systems.

**arXiv ID:** 2510.27418
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (10 papers)</h2></summary>

<details>
<summary><strong>The Denario project: Deep knowledge AI agents for scientific discovery</strong> - Francisco Villaescusa-Navarro, Boris Bolliet, Pablo Villanueva-Domingo, Adrian E. Bayer, Aidan Acquah, Chetana Amancharla, Almog Barzilay-Siegal, Pablo Bermejo, Camille Bilodeau, Pablo Cárdenas Ramírez, Miles Cranmer, Urbano L. França, ChangHoon Hahn, Yan-Fei Jiang, Raul Jimenez, Jun-Young Lee, Antonio Lerario, Osman Mamun, Thomas Meier, Anupam A. Ojha, Pavlos Protopapas, Shimanto Roy, David N. Spergel, Pedro Tarancón-Álvarez, Ujjwal Tiwari, Matteo Viel, Digvijay Wadekar, Chi Wang, Bonny Y. Wang, Licong Xu, Yossi Yovel, Shuwen Yue, Wen-Han Zhou, Qiyao Zhu, Jiajun Zou, Íñigo Zubeldia - [[pdf]](https://arxiv.org/pdf/2510.26887)</summary>

**Abstract:** We present Denario, an AI multi-agent system designed to serve as a scientific research assistant. Denario can perform many different tasks, such as generating ideas, checking the literature, developing research plans, writing and executing code, making plots, and drafting and reviewing a scientific paper. The system has a modular architecture, allowing it to handle specific tasks, such as generating an idea, or carrying out end-to-end scientific analysis using Cmbagent as a deep-research backend. In this work, we describe in detail Denario and its modules, and illustrate its capabilities by presenting multiple AI-generated papers generated by it in many different scientific disciplines such as astrophysics, biology, biophysics, biomedical informatics, chemistry, material science, mathematical physics, medicine, neuroscience and planetary science. Denario also excels at combining ideas from different disciplines, and we illustrate this by showing a paper that applies methods from quantum physics and machine learning to astrophysical data. We report the evaluations performed on these papers by domain experts, who provided both numerical scores and review-like feedback. We then highlight the strengths, weaknesses, and limitations of the current system. Finally, we discuss the ethical implications of AI-driven research and reflect on how such technology relates to the philosophy of science. We publicly release the code at this https URL. A Denario demo can also be run directly on the web at this https URL, and the full app will be deployed on the cloud.

**arXiv ID:** 2510.26887
</details>

<details>
<summary><strong>Realistic pedestrian-driver interaction modelling using multi-agent RL with human perceptual-motor constraints</strong> - Yueyang Wang, Mehmet Dogar, Gustav Markkula - [[pdf]](https://arxiv.org/pdf/2510.27383)</summary>

**Abstract:** Modelling pedestrian-driver interactions is critical for understanding human road user behaviour and developing safe autonomous vehicle systems. Existing approaches often rely on rule-based logic, game-theoretic models, or 'black-box' machine learning methods. However, these models typically lack flexibility or overlook the underlying mechanisms, such as sensory and motor constraints, which shape how pedestrians and drivers perceive and act in interactive scenarios. In this study, we propose a multi-agent reinforcement learning (RL) framework that integrates both visual and motor constraints of pedestrian and driver agents. Using a real-world dataset from an unsignalised pedestrian crossing, we evaluate four model variants, one without constraints, two with either motor or visual constraints, and one with both, across behavioural metrics of interaction realism. Results show that the combined model with both visual and motor constraints performs best. Motor constraints lead to smoother movements that resemble human speed adjustments during crossing interactions. The addition of visual constraints introduces perceptual uncertainty and field-of-view limitations, leading the agents to exhibit more cautious and variable behaviour, such as less abrupt deceleration. In this data-limited setting, our model outperforms a supervised behavioural cloning model, demonstrating that our approach can be effective without large training datasets. Finally, our framework accounts for individual differences by modelling parameters controlling the human constraints as population-level distributions, a perspective that has not been explored in previous work on pedestrian-vehicle interaction modelling. Overall, our work demonstrates that multi-agent RL with human constraints is a promising modelling approach for simulating realistic road user interactions.

**arXiv ID:** 2510.27383
</details>

<details>
<summary><strong>SIGMA: Search-Augmented On-Demand Knowledge Integration for Agentic Mathematical Reasoning</strong> - Ali Asgarov, Umid Suleymanov, Aadyant Khatri - [[pdf]](https://arxiv.org/pdf/2510.27568)</summary>

**Abstract:** Solving mathematical reasoning problems requires not only accurate access to relevant knowledge but also careful, multi-step thinking. However, current retrieval-augmented models often rely on a single perspective, follow inflexible search strategies, and struggle to effectively combine information from multiple sources. We introduce SIGMA (Search-Augmented On-Demand Knowledge Integration for AGentic Mathematical reAsoning), a unified framework that orchestrates specialized agents to independently reason, perform targeted searches, and synthesize findings through a moderator mechanism. Each agent generates hypothetical passages to optimize retrieval for its analytic perspective, ensuring knowledge integration is both context-sensitive and computation-efficient. When evaluated on challenging benchmarks such as MATH500, AIME, and PhD-level science QA GPQA, SIGMA consistently outperforms both open- and closed-source systems, achieving an absolute performance improvement of 7.4%. Our results demonstrate that multi-agent, on-demand knowledge integration significantly enhances both reasoning accuracy and efficiency, offering a scalable approach for complex, knowledge-intensive problem-solving. We will release the code upon publication.

**arXiv ID:** 2510.27568
</details>

<details>
<summary><strong>VeriMoA: A Mixture-of-Agents Framework for Spec-to-HDL Generation</strong> - Heng Ping, Arijit Bhattacharjee, Peiyu Zhang, Shixuan Li, Wei Yang, Anzhe Cheng, Xiaole Zhang, Jesse Thomason, Ali Jannesari, Nesreen Ahmed, Paul Bogdan - [[pdf]](https://arxiv.org/pdf/2510.27617)</summary>

**Abstract:** Automation of Register Transfer Level (RTL) design can help developers meet increasing computational demands. Large Language Models (LLMs) show promise for Hardware Description Language (HDL) generation, but face challenges due to limited parametric knowledge and domain-specific constraints. While prompt engineering and fine-tuning have limitations in knowledge coverage and training costs, multi-agent architectures offer a training-free paradigm to enhance reasoning through collaborative generation. However, current multi-agent approaches suffer from two critical deficiencies: susceptibility to noise propagation and constrained reasoning space exploration. We propose VeriMoA, a training-free mixture-of-agents (MoA) framework with two synergistic innovations. First, a quality-guided caching mechanism to maintain all intermediate HDL outputs and enables quality-based ranking and selection across the entire generation process, encouraging knowledge accumulation over layers of reasoning. Second, a multi-path generation strategy that leverages C++ and Python as intermediate representations, decomposing specification-to-HDL translation into two-stage processes that exploit LLM fluency in high-resource languages while promoting solution diversity. Comprehensive experiments on VerilogEval 2.0 and RTLLM 2.0 benchmarks demonstrate that VeriMoA achieves 15--30% improvements in Pass@1 across diverse LLM backbones, especially enabling smaller models to match larger models and fine-tuned alternatives without requiring costly training.

**arXiv ID:** 2510.27617
</details>

<details>
<summary><strong>Challenges in Credit Assignment for Multi-Agent Reinforcement Learning in Open Agent Systems</strong> - Alireza Saleh Abadi, Leen-Kiat Soh - [[pdf]](https://arxiv.org/pdf/2510.27659)</summary>

**Abstract:** In the rapidly evolving field of multi-agent reinforcement learning (MARL), understanding the dynamics of open systems is crucial. Openness in MARL refers to the dynam-ic nature of agent populations, tasks, and agent types with-in a system. Specifically, there are three types of openness as reported in (Eck et al. 2023) [2]: agent openness, where agents can enter or leave the system at any time; task openness, where new tasks emerge, and existing ones evolve or disappear; and type openness, where the capabil-ities and behaviors of agents change over time. This report provides a conceptual and empirical review, focusing on the interplay between openness and the credit assignment problem (CAP). CAP involves determining the contribution of individual agents to the overall system performance, a task that becomes increasingly complex in open environ-ments. Traditional credit assignment (CA) methods often assume static agent populations, fixed and pre-defined tasks, and stationary types, making them inadequate for open systems. We first conduct a conceptual analysis, in-troducing new sub-categories of openness to detail how events like agent turnover or task cancellation break the assumptions of environmental stationarity and fixed team composition that underpin existing CAP methods. We then present an empirical study using representative temporal and structural algorithms in an open environment. The results demonstrate that openness directly causes credit misattribution, evidenced by unstable loss functions and significant performance degradation.

**arXiv ID:** 2510.27659
</details>

<details>
<summary><strong>SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents</strong> - Sheng Yin, Xianghe Pang, Yuanzhuo Ding, Menglan Chen, Yutong Bi, Yichen Xiong, Wenhao Huang, Zhen Xiang, Jing Shao, Siheng Chen - [[pdf]](https://arxiv.org/pdf/2412.13178)</summary>

**Abstract:** With the integration of large language models (LLMs), embodied agents have strong capabilities to understand and plan complicated natural language instructions. However, a foreseeable issue is that those embodied agents can also flawlessly execute some hazardous tasks, potentially causing damages in the real world. Existing benchmarks predominantly overlook critical safety risks, focusing solely on planning performance, while a few evaluate LLMs' safety awareness only on non-interactive image-text data. To address this gap, we present SafeAgentBench -- the first comprehensive benchmark for safety-aware task planning of embodied LLM agents in interactive simulation environments, covering both explicit and implicit hazards. SafeAgentBench includes: (1) an executable, diverse, and high-quality dataset of 750 tasks, rigorously curated to cover 10 potential hazards and 3 task types; (2) SafeAgentEnv, a universal embodied environment with a low-level controller, supporting multi-agent execution with 17 high-level actions for 9 state-of-the-art baselines; and (3) reliable evaluation methods from both execution and semantic perspectives. Experimental results show that, although agents based on different design frameworks exhibit substantial differences in task success rates, their overall safety awareness remains weak. The most safety-conscious baseline achieves only a 10% rejection rate for detailed hazardous tasks. Moreover, simply replacing the LLM driving the agent does not lead to notable improvements in safety awareness. Dataset and codes are available in this https URL and this https URL.

**arXiv ID:** 2412.13178
</details>

<details>
<summary><strong>PartnerMAS: An LLM Hierarchical Multi-Agent Framework for Business Partner Selection on High-Dimensional Features</strong> - Lingyao Li, Haolun Wu, Zhenkun Li, Jiabei Hu, Yu Wang, Xiaoshan Huang, Wenyue Hua, Wenqian Wang - [[pdf]](https://arxiv.org/pdf/2509.24046)</summary>

**Abstract:** High-dimensional decision-making tasks, such as business partner selection, involve evaluating large candidate pools with heterogeneous numerical, categorical, and textual features. While large language models (LLMs) offer strong in-context reasoning capabilities, single-agent or debate-style systems often struggle with scalability and consistency in such settings. We propose PartnerMAS, a hierarchical multi-agent framework that decomposes evaluation into three layers: a Planner Agent that designs strategies, Specialized Agents that perform role-specific assessments, and a Supervisor Agent that integrates their outputs. To support systematic evaluation, we also introduce a curated benchmark dataset of venture capital co-investments, featuring diverse firm attributes and ground-truth syndicates. Across 140 cases, PartnerMAS consistently outperforms single-agent and debate-based multi-agent baselines, achieving up to 10--15\% higher match rates. Analysis of agent reasoning shows that planners are most responsive to domain-informed prompts, specialists produce complementary feature coverage, and supervisors play an important role in aggregation. Our findings demonstrate that structured collaboration among LLM agents can generate more robust outcomes than scaling individual models, highlighting PartnerMAS as a promising framework for high-dimensional decision-making in data-rich domains.

**arXiv ID:** 2509.24046
</details>

<details>
<summary><strong>LAFA: Agentic LLM-Driven Federated Analytics over Decentralized Data Sources</strong> - Haichao Ji, Zibo Wang, Cheng Pan, Meng Han, Yifei Zhu, Dan Wang, Zhu Han - [[pdf]](https://arxiv.org/pdf/2510.18477)</summary>

**Abstract:** Large Language Models (LLMs) have shown great promise in automating data analytics tasks by interpreting natural language queries and generating multi-operation execution plans. However, existing LLM-agent-based analytics frameworks operate under the assumption of centralized data access, offering little to no privacy protection. In contrast, federated analytics (FA) enables privacy-preserving computation across distributed data sources, but lacks support for natural language input and requires structured, machine-readable queries. In this work, we present LAFA, the first system that integrates LLM-agent-based data analytics with FA. LAFA introduces a hierarchical multi-agent architecture that accepts natural language queries and transforms them into optimized, executable FA workflows. A coarse-grained planner first decomposes complex queries into sub-queries, while a fine-grained planner maps each subquery into a Directed Acyclic Graph of FA operations using prior structural knowledge. To improve execution efficiency, an optimizer agent rewrites and merges multiple DAGs, eliminating redundant operations and minimizing computational and communicational overhead. Our experiments demonstrate that LAFA consistently outperforms baseline prompting strategies by achieving higher execution plan success rates and reducing resource-intensive FA operations by a substantial margin. This work establishes a practical foundation for privacy-preserving, LLM-driven analytics that supports natural language input in the FA setting.

**arXiv ID:** 2510.18477
</details>

<details>
<summary><strong>Design for One, Deploy for Many: Navigating Tree Mazes with Multiple Agents</strong> - Jahir Argote-Gerald, Genki Miyauchi, Julian Rau, Paul Trodden, Roderich Gross - [[pdf]](https://arxiv.org/pdf/2510.26900)</summary>

**Abstract:** Maze-like environments, such as cave and pipe networks, pose unique challenges for multiple robots to coordinate, including communication constraints and congestion. To address these challenges, we propose a distributed multi-agent maze traversal algorithm for environments that can be represented by acyclic graphs. It uses a leader-switching mechanism where one agent, assuming a head role, employs any single-agent maze solver while the other agents each choose an agent to follow. The head role gets transferred to neighboring agents where necessary, ensuring it follows the same path as a single agent would. The multi-agent maze traversal algorithm is evaluated in simulations with groups of up to 300 agents, various maze sizes, and multiple single-agent maze solvers. It is compared against strategies that are naïve, or assume either global communication or full knowledge of the environment. The algorithm outperforms the naïve strategy in terms of makespan and sum-of-fuel. It is superior to the global-communication strategy in terms of makespan but is inferior to it in terms of sum-of-fuel. The findings suggest it is asymptotically equivalent to the full-knowledge strategy with respect to either metric. Moreover, real-world experiments with up to 20 Pi-puck robots confirm the feasibility of the approach.

**arXiv ID:** 2510.26900
</details>

<details>
<summary><strong>Asynchronous Risk-Aware Multi-Agent Packet Routing for Ultra-Dense LEO Satellite Networks</strong> - Ke He, Thang X. Vu, Le He, Lisheng Fan, Symeon Chatzinotas, Bjorn Ottersten - [[pdf]](https://arxiv.org/pdf/2510.27506)</summary>

**Abstract:** The rise of ultra-dense LEO constellations creates a complex and asynchronous network environment, driven by their massive scale, dynamic topologies, and significant delays. This unique complexity demands an adaptive packet routing algorithm that is asynchronous, risk-aware, and capable of balancing diverse and often conflicting QoS objectives in a decentralized manner. However, existing methods fail to address this need, as they typically rely on impractical synchronous decision-making and/or risk-oblivious approaches. To tackle this gap, we introduce PRIMAL, an event-driven multi-agent routing framework designed specifically to allow each satellite to act independently on its own event-driven timeline, while managing the risk of worst-case performance degradation via a principled primal-dual approach. This is achieved by enabling agents to learn the full cost distribution of the targeted QoS objectives and constrain tail-end risks. Extensive simulations on a LEO constellation with 1584 satellites validate its superiority in effectively optimizing latency and balancing load. Compared to a recent risk-oblivious baseline, it reduces queuing delay by over 70%, and achieves a nearly 12 ms end-to-end delay reduction in loaded scenarios. This is accomplished by resolving the core conflict between naive shortest-path finding and congestion avoidance, highlighting such autonomous risk-awareness as a key to robust routing.

**arXiv ID:** 2510.27506
</details>

</details>

<details open>
<summary><h2>Other Agent Research (10 papers)</h2></summary>

<details>
<summary><strong>Cognition Envelopes for Bounded AI Reasoning in Autonomous UAS Operations</strong> - Pedro Antonio Alarcón Granadeno, Arturo Miguel Bernal Russell, Sofia Nelson, Demetrius Hernandez, Maureen Petterson, Michael Murphy, Walter J. Scheirer, Jane Cleland-Huang - [[pdf]](https://arxiv.org/pdf/2510.26905)</summary>

**Abstract:** Cyber-physical systems increasingly rely on Foundational Models such as Large Language Models (LLMs) and Vision-Language Models (VLMs) to increase autonomy through enhanced perception, inference, and planning. However, these models also introduce new types of errors, such as hallucinations, overgeneralizations, and context misalignments, resulting in incorrect and flawed decisions. To address this, we introduce the concept of Cognition Envelopes, designed to establish reasoning boundaries that constrain AI-generated decisions while complementing the use of meta-cognition and traditional safety envelopes. As with safety envelopes, Cognition Envelopes require practical guidelines and systematic processes for their definition, validation, and assurance.

**arXiv ID:** 2510.26905
</details>

<details>
<summary><strong>Adaptive Data Flywheel: Applying MAPE Control Loops to AI Agent Improvement</strong> - Aaditya Shukla, Sidney Knowles, Meenakshi Madugula, Dave Farris, Ryan Angilly, Santiago Pombo, Anbang Xu, Lu An, Abhinav Balasubramanian, Tan Yu, Jiaxiang Ren, Rama Akkiraju - [[pdf]](https://arxiv.org/pdf/2510.27051)</summary>

**Abstract:** Enterprise AI agents must continuously adapt to maintain accuracy, reduce latency, and remain aligned with user needs. We present a practical implementation of a data flywheel in NVInfo AI, NVIDIA's Mixture-of-Experts (MoE) Knowledge Assistant serving over 30,000 employees. By operationalizing a MAPE-driven data flywheel, we built a closed-loop system that systematically addresses failures in retrieval-augmented generation (RAG) pipelines and enables continuous learning. Over a 3-month post-deployment period, we monitored feedback and collected 495 negative samples. Analysis revealed two major failure modes: routing errors (5.25\%) and query rephrasal errors (3.2\%). Using NVIDIA NeMo microservices, we implemented targeted improvements through fine-tuning. For routing, we replaced a Llama 3.1 70B model with a fine-tuned 8B variant, achieving 96\% accuracy, a 10x reduction in model size, and 70\% latency improvement. For query rephrasal, fine-tuning yielded a 3.7\% gain in accuracy and a 40\% latency reduction. Our approach demonstrates how human-in-the-loop (HITL) feedback, when structured within a data flywheel, transforms enterprise AI agents into self-improving systems. Key learnings include approaches to ensure agent robustness despite limited user feedback, navigating privacy constraints, and executing staged rollouts in production. This work offers a repeatable blueprint for building robust, adaptive enterprise AI agents capable of learning from real-world usage at scale.

**arXiv ID:** 2510.27051
</details>

<details>
<summary><strong>Interaction as Intelligence Part II: Asynchronous Human-Agent Rollout for Long-Horizon Task Training</strong> - Dayuan Fu, Yunze Wu, Xiaojie Cai, Lyumanshan Ye, Shijie Xia, Zhen Huang, Weiye Si, Tianze Xu, Jie Sun, Keyu Li, Mohan Jiang, Junfei Wang, Qishuo Hua, Pengrui Lu, Yang Xiao, Pengfei Liu - [[pdf]](https://arxiv.org/pdf/2510.27630)</summary>

**Abstract:** Large Language Model (LLM) agents have recently shown strong potential in domains such as automated coding, deep research, and graphical user interface manipulation. However, training them to succeed on long-horizon, domain-specialized tasks remains challenging. Current methods primarily fall into two categories. The first relies on dense human annotations through behavior cloning, which is prohibitively expensive for long-horizon tasks that can take days or months. The second depends on outcome-driven sampling, which often collapses due to the rarity of valid positive trajectories on domain-specialized tasks. We introduce Apollo, a sampling framework that integrates asynchronous human guidance with action-level data filtering. Instead of requiring annotators to shadow every step, Apollo allows them to intervene only when the agent drifts from a promising trajectory, by providing prior knowledge, strategic advice, etc. This lightweight design makes it possible to sustain interactions for over 30 hours and produces valuable trajectories at a lower cost. Apollo then applies supervision control to filter out sub-optimal actions and prevent error propagation. Together, these components enable reliable and effective data collection in long-horizon environments. To demonstrate the effectiveness of Apollo, we evaluate it using InnovatorBench. Our experiments show that when applied to train the GLM-4.5 model on InnovatorBench, Apollo achieves more than a 50% improvement over the untrained baseline and a 28% improvement over a variant trained without human interaction. These results highlight the critical role of human-in-the-loop sampling and the robustness of Apollo's design in handling long-horizon, domain-specialized tasks.

**arXiv ID:** 2510.27630
</details>

<details>
<summary><strong>Artificially intelligent agents in the social and behavioral sciences: A history and outlook</strong> - Petter Holme, Milena Tsvetkova - [[pdf]](https://arxiv.org/pdf/2510.05743)</summary>

**Abstract:** We review the historical development and current trends of artificially intelligent agents (agentic AI) in the social and behavioral sciences: from the first programmable computers, and social simulations soon thereafter, to today's experiments with large language models. This overview emphasizes the role of AI in the scientific process and the changes brought about, both through technological advancements and the broader evolution of science from around 1950 to the present. Some of the specific points we cover include: the challenges of presenting the first social simulation studies to a world unaware of computers, the rise of social systems science, intelligent game theoretic agents, the age of big data and the epistemic upheaval in its wake, and the current enthusiasm around applications of generative AI, and many other topics. A pervasive theme is how deeply entwined we are with the technologies we use to understand ourselves.

**arXiv ID:** 2510.05743
</details>

<details>
<summary><strong>Training a Generally Curious Agent</strong> - Fahim Tajwar, Yiding Jiang, Abitha Thankaraj, Sumaita Sadia Rahman, J Zico Kolter, Jeff Schneider, Ruslan Salakhutdinov - [[pdf]](https://arxiv.org/pdf/2502.17543)</summary>

**Abstract:** Efficient exploration is essential for intelligent systems interacting with their environment, but existing language models often fall short in scenarios that require strategic information gathering. In this paper, we present Paprika, a fine-tuning approach that enables language models to develop general decision-making capabilities that are not confined to particular environments. By training on synthetic interaction data from different tasks that require diverse strategies, Paprika teaches models to explore and adapt their behavior on a new task based on environment feedback in-context without more gradient updates. Experimental results show that models fine-tuned with Paprika can effectively transfer their learned decision-making capabilities to entirely unseen tasks without additional training. Unlike traditional training, our approach's primary bottleneck lies in sampling useful interaction data instead of model updates. To improve sample efficiency, we propose a curriculum learning strategy that prioritizes sampling trajectories from tasks with high learning potential. These results suggest a promising path towards AI systems that can autonomously solve novel sequential decision-making problems that require interactions with the external world.

**arXiv ID:** 2502.17543
</details>

<details>
<summary><strong>Dynamic Risk Assessments for Offensive Cybersecurity Agents</strong> - Boyi Wei, Benedikt Stroebl, Jiacen Xu, Joie Zhang, Zhou Li, Peter Henderson - [[pdf]](https://arxiv.org/pdf/2505.18384)</summary>

**Abstract:** Foundation models are increasingly becoming better autonomous programmers, raising the prospect that they could also automate dangerous offensive cyber-operations. Current frontier model audits probe the cybersecurity risks of such agents, but most fail to account for the degrees of freedom available to adversaries in the real world. In particular, with strong verifiers and financial incentives, agents for offensive cybersecurity are amenable to iterative improvement by would-be adversaries. We argue that assessments should take into account an expanded threat model in the context of cybersecurity, emphasizing the varying degrees of freedom that an adversary may possess in stateful and non-stateful environments within a fixed compute budget. We show that even with a relatively small compute budget (8 H100 GPU Hours in our study), adversaries can improve an agent's cybersecurity capability on InterCode CTF by more than 40\% relative to the baseline -- without any external assistance. These results highlight the need to evaluate agents' cybersecurity risk in a dynamic manner, painting a more representative picture of risk.

**arXiv ID:** 2505.18384
</details>

<details>
<summary><strong>FinPos: A Position-Aware Trading Agent System for Real Financial Markets</strong> - Bijia Liu, Ronghao Dang - [[pdf]](https://arxiv.org/pdf/2510.27251)</summary>

**Abstract:** The exceptional potential of large language models (LLMs) in handling text information has garnered significant attention in the field of financial trading. However, current trading agents primarily focus on single-step trading tasks and lack awareness of continuous position management. Therefore, we propose a position-aware trading task designed to simulate a more realistic market. To address this task, we develop a trading agent system, FinPos, optimized for position management. FinPos is able to interpret various types of market information from a professional perspective, providing a reliable basis for positioning decisions. To mitigate the substantial market risks arising from position fluctuations, FinPos employs dual decision agents. Furthermore, the continuous nature of position management necessitates our adoption of multi-timescale rewards, which in turn empowers FinPos to effectively balance short-term fluctuations against long-term trends. Extensive experiments demonstrate that FinPos surpasses state-of-the-art trading agents in the position-aware trading task, which closely mirrors real market conditions. More importantly, our findings reveal that LLM-centered agent systems exhibit a vast, largely unexplored potential in long-term market decision-making.

**arXiv ID:** 2510.27251
</details>

<details>
<summary><strong>Cooperative Integrated Estimation-Guidance for Simultaneous Interception of Moving Targets</strong> - Lohitvel Gopikannan, Shashi Ranjan Kumar, Abhinav Sinha - [[pdf]](https://arxiv.org/pdf/2510.26948)</summary>

**Abstract:** This paper proposes a cooperative integrated estimation-guidance framework for simultaneous interception of a non-maneuvering target using a team of unmanned autonomous vehicles, assuming only a subset of vehicles are equipped with dedicated sensors to measure the target's states. Unlike earlier approaches that focus solely on either estimation or guidance design, the proposed framework unifies both within a cooperative architecture. To circumvent the limitation posed by heterogeneity in target observability, sensorless vehicles estimate the target's state by leveraging information exchanged with neighboring agents over a directed communication topology through a prescribed-time observer. The proposed approach employs true proportional navigation guidance (TPNG), which uses an exact time-to-go formulation and is applicable across a wide spectrum of target motions. Furthermore, prescribed-time observer and controller are employed to achieve convergence to true target's state and consensus in time-to-go within set predefined times, respectively. Simulations demonstrate the effectiveness of the proposed framework under various engagement scenarios.

**arXiv ID:** 2510.26948
</details>

<details>
<summary><strong>Exploiting Agent Symmetries for Performance Analysis of Distributed Optimization Methods</strong> - Sebastien Colla, Julien M. Hendrickx - [[pdf]](https://arxiv.org/pdf/2403.11724)</summary>

**Abstract:** We show that, in many settings, the worst-case performance of a distributed optimization algorithm is independent of the number of agents in the system, and can thus be computed in the fundamental case with just two agents. This result relies on a novel approach that systematically exploits symmetries in worst-case performance computation, framed as Semidefinite Programming (SDP) via the Performance Estimation Problem (PEP) framework. Harnessing agent symmetries in the PEP yields compact problems whose size is independent of the number of agents in the system. When all agents are equivalent in the problem, we establish the explicit conditions under which the resulting worst-case performance is independent of the number of agents and is therefore equivalent to the basic case with two agents. Our compact PEP formulation also allows the consideration of multiple equivalence classes of agents, and its size only depends on the number of equivalence classes. This enables practical and automated performance analysis of distributed algorithms in numerous complex and realistic settings, such as the analysis of the worst agent performance. We leverage this new tool to analyze the performance of the EXTRA algorithm in advanced settings and its scalability with the number of agents, providing a tighter analysis and deeper understanding of the algorithm performance.

**arXiv ID:** 2403.11724
</details>

<details>
<summary><strong>Sensible Agent: A Framework for Unobtrusive Interaction with Proactive AR Agents</strong> - Geonsun Lee, Min Xia, Nels Numan, Xun Qian, David Li, Yanhe Chen, Achin Kulshrestha, Ishan Chatterjee, Yinda Zhang, Dinesh Manocha, David Kim, Ruofei Du - [[pdf]](https://arxiv.org/pdf/2509.09255)</summary>

**Abstract:** Proactive AR agents promise context-aware assistance, but their interactions often rely on explicit voice prompts or responses, which can be disruptive or socially awkward. We introduce Sensible Agent, a framework designed for unobtrusive interaction with these proactive agents. Sensible Agent dynamically adapts both "what" assistance to offer and, crucially, "how" to deliver it, based on real-time multimodal context sensing. Informed by an expert workshop (n=12) and a data annotation study (n=40), the framework leverages egocentric cameras, multimodal sensing, and Large Multimodal Models (LMMs) to infer context and suggest appropriate actions delivered via minimally intrusive interaction modes. We demonstrate our prototype on an XR headset through a user study (n=10) in both AR and VR scenarios. Results indicate that Sensible Agent significantly reduces perceived interaction effort compared to voice-prompted baseline, while maintaining high usability and achieving higher preference.

**arXiv ID:** 2509.09255
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (21 papers)</h2></summary>

<details>
<summary><strong>Reinforcement Learning for Long-Horizon Unordered Tasks: From Boolean to Coupled Reward Machines</strong> - Kristina Levina, Nikolaos Pappas, Athanasios Karapantelakis, Aneta Vulgarakis Feljan, Jendrik Seipp - [[pdf]](https://arxiv.org/pdf/2510.27329)</summary>

**Abstract:** Reward machines (RMs) inform reinforcement learning agents about the reward structure of the environment. This is particularly advantageous for complex non-Markovian tasks because agents with access to RMs can learn more efficiently from fewer samples. However, learning with RMs is ill-suited for long-horizon problems in which a set of subtasks can be executed in any order. In such cases, the amount of information to learn increases exponentially with the number of unordered subtasks. In this work, we address this limitation by introducing three generalisations of RMs: (1) Numeric RMs allow users to express complex tasks in a compact form. (2) In Agenda RMs, states are associated with an agenda that tracks the remaining subtasks to complete. (3) Coupled RMs have coupled states associated with each subtask in the agenda. Furthermore, we introduce a new compositional learning algorithm that leverages coupled RMs: Q-learning with coupled RMs (CoRM). Our experiments show that CoRM scales better than state-of-the-art RM algorithms for long-horizon problems with unordered subtasks.

**arXiv ID:** 2510.27329
</details>

<details>
<summary><strong>InnovatorBench: Evaluating Agents' Ability to Conduct Innovative LLM Research</strong> - Yunze Wu, Dayuan Fu, Weiye Si, Zhen Huang, Mohan Jiang, Keyu Li, Shijie Xia, Jie Sun, Tianze Xu, Xiangkun Hu, Pengrui Lu, Xiaojie Cai, Lyumanshan Ye, Wenhong Zhu, Yang Xiao, Pengfei Liu - [[pdf]](https://arxiv.org/pdf/2510.27598)</summary>

**Abstract:** AI agents could accelerate scientific discovery by automating hypothesis formation, experiment design, coding, execution, and analysis, yet existing benchmarks probe narrow skills in simplified settings. To address this gap, we introduce InnovatorBench, a benchmark-platform pair for realistic, end-to-end assessment of agents performing Large Language Model (LLM) research. It comprises 20 tasks spanning Data Construction, Filtering, Augmentation, Loss Design, Reward Design, and Scaffold Construction, which require runnable artifacts and assessment of correctness, performance, output quality, and uncertainty. To support agent operation, we develop ResearchGym, a research environment offering rich action spaces, distributed and long-horizon execution, asynchronous monitoring, and snapshot saving. We also implement a lightweight ReAct agent that couples explicit reasoning with executable planning using frontier models such as Claude-4, GPT-5, GLM-4.5, and Kimi-K2. Our experiments demonstrate that while frontier models show promise in code-driven research tasks, they struggle with fragile algorithm-related tasks and long-horizon decision making, such as impatience, poor resource management, and overreliance on template-based reasoning. Furthermore, agents require over 11 hours to achieve their best performance on InnovatorBench, underscoring the benchmark's difficulty and showing the potential of InnovatorBench to be the next generation of code-based research benchmark.

**arXiv ID:** 2510.27598
</details>

<details>
<summary><strong>Reinforcement Learning for Accelerator Beamline Control: a simulation-based approach</strong> - Anwar Ibrahim, Alexey Petrenko, Maxim Kaledin, Ehab Suleiman, Fedor Ratnikov, Denis Derkach - [[pdf]](https://arxiv.org/pdf/2510.26805)</summary>

**Abstract:** Particle accelerators play a pivotal role in advancing scientific research, yet optimizing beamline configurations to maximize particle transmission remains a labor-intensive task requiring expert intervention. In this work, we introduce RLABC (Reinforcement Learning for Accelerator Beamline Control), a Python-based library that reframes beamline optimization as a reinforcement learning (RL) problem. Leveraging the Elegant simulation framework, RLABC automates the creation of an RL environment from standard lattice and element input files, enabling sequential tuning of magnets to minimize particle losses. We define a comprehensive state representation capturing beam statistics, actions for adjusting magnet parameters, and a reward function focused on transmission efficiency. Employing the Deep Deterministic Policy Gradient (DDPG) algorithm, we demonstrate RLABC's efficacy on two beamlines, achieving transmission rates of 94% and 91%, comparable to expert manual optimizations. This approach bridges accelerator physics and machine learning, offering a versatile tool for physicists and RL researchers alike to streamline beamline tuning.

**arXiv ID:** 2510.26805
</details>

<details>
<summary><strong>Sybil-Resistant Service Discovery for Agent Economies</strong> - David Shi, Kevin Joo - [[pdf]](https://arxiv.org/pdf/2510.27554)</summary>

**Abstract:** x402 enables Hypertext Transfer Protocol (HTTP) services like application programming interfaces (APIs), data feeds, and inference providers to accept cryptocurrency payments for access. As agents increasingly consume these services, discovery becomes critical: which swap interface should an agent trust? Which data provider is the most reliable? We introduce TraceRank, a reputation-weighted ranking algorithm where payment transactions serve as endorsements. TraceRank seeds addresses with precomputed reputation metrics and propagates reputation through payment flows weighted by transaction value and temporal recency. Applied to x402's payment graph, this surfaces services preferred by high-reputation users rather than those with high transaction volume. Our system combines TraceRank with semantic search to respond to natural language queries with high quality results. We argue that reputation propagation resists Sybil attacks by making spam services with many low-reputation payers rank below legitimate services with few high-reputation payers. Ultimately, we aim to construct a search method for x402 enabled services that avoids infrastructure bias and has better performance than purely volume based or semantic methods.

**arXiv ID:** 2510.27554
</details>

<details>
<summary><strong>Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning</strong> - Yuhong Liu, Beichen Zhang, Yuhang Zang, Yuhang Cao, Long Xing, Xiaoyi Dong, Haodong Duan, Dahua Lin, Jiaqi Wang - [[pdf]](https://arxiv.org/pdf/2510.27606)</summary>

**Abstract:** Spatial understanding remains a weakness of Large Vision-Language Models (LVLMs). Existing supervised fine-tuning (SFT) and recent reinforcement learning with verifiable rewards (RLVR) pipelines depend on costly supervision, specialized tools, or constrained environments that limit scale. We introduce Spatial-SSRL, a self-supervised RL paradigm that derives verifiable signals directly from ordinary RGB or RGB-D images. Spatial-SSRL automatically formulates five pretext tasks that capture 2D and 3D spatial structure: shuffled patch reordering, flipped patch recognition, cropped patch inpainting, regional depth ordering, and relative 3D position prediction. These tasks provide ground-truth answers that are easy to verify and require no human or LVLM annotation. Training on our tasks substantially improves spatial reasoning while preserving general visual capabilities. On seven spatial understanding benchmarks in both image and video settings, Spatial-SSRL delivers average accuracy gains of 4.63% (3B) and 3.89% (7B) over the Qwen2.5-VL baselines. Our results show that simple, intrinsic supervision enables RLVR at scale and provides a practical route to stronger spatial intelligence in LVLMs.

**arXiv ID:** 2510.27606
</details>

<details>
<summary><strong>Towards Automated Semantic Interpretability in Reinforcement Learning via Vision-Language Models</strong> - Zhaoxin Li, Zhang Xi-Jia, Batuhan Altundas, Letian Chen, Rohan Paleja, Matthew Gombolay - [[pdf]](https://arxiv.org/pdf/2503.16724)</summary>

**Abstract:** Semantic interpretability in Reinforcement Learning (RL) enables transparency and verifiability of decision-making. Achieving semantic interpretability in reinforcement learning requires (1) a feature space composed of human-understandable concepts and (2) a policy that is interpretable and verifiable. However, constructing such a feature space has traditionally relied on manual human specification, which often fails to generalize to unseen environments. Moreover, even when interpretable features are available, most reinforcement learning algorithms employ black-box models as policies, thereby hindering transparency. We introduce interpretable Tree-based Reinforcement learning via Automated Concept Extraction (iTRACE), an automated framework that leverages pre-trained vision-language models (VLM) for semantic feature extraction and train a interpretable tree-based model via RL. To address the impracticality of running VLMs in RL loops, we distill their outputs into a lightweight model. By leveraging Vision-Language Models (VLMs) to automate tree-based reinforcement learning, iTRACE loosens the reliance the need for human annotation that is traditionally required by interpretable models. In addition, it addresses key limitations of VLMs alone, such as their lack of grounding in action spaces and their inability to directly optimize policies. We evaluate iTRACE across three domains: Atari games, grid-world navigation, and driving. The results show that iTRACE outperforms other interpretable policy baselines and matches the performance of black-box policies on the same interpretable feature space.

**arXiv ID:** 2503.16724
</details>

<details>
<summary><strong>Reinforcement Learning vs. Distillation: Understanding Accuracy and Capability in LLM Reasoning</strong> - Minwu Kim, Anubhav Shrestha, Safal Shrestha, Aadim Nepal, Keith Ross - [[pdf]](https://arxiv.org/pdf/2505.14216)</summary>

**Abstract:** Recent studies have shown that reinforcement learning with verifiable rewards (RLVR) enhances overall accuracy (pass@1) but often fails to improve capability (pass@k) of LLMs in reasoning tasks, while distillation can improve both. In this paper, we investigate the mechanisms behind these phenomena. First, we demonstrate that RLVR struggles to improve capability as it focuses on improving the accuracy of the easier questions to the detriment of the accuracy of the most difficult questions. Second, we show that RLVR does not merely increase the success probability for the easier questions, but in our small model settings, produces quality responses that were absent in its original output distribution. In addition, we show these responses are neither noticeably longer nor feature more reflection-related keywords, underscoring the need for more reliable indicators of response quality. Third, from the experiment distilling teacher responses to in-distribution problems, we find that capability does not always improve with distillation. We conjecture that capability improves only when new knowledge is introduced, whereas distilling reasoning patterns only improves accuracy but not capability, sacrificing performance on the most difficult questions, similar to RLVR. Together, these findings offer a clearer understanding of how RLVR and distillation shape reasoning behavior in LLMs

**arXiv ID:** 2505.14216
</details>

<details>
<summary><strong>NaviAgent: Bilevel Planning on Tool Navigation Graph for Large-Scale Orchestration</strong> - Yan Jiang, Hao Zhou, LiZhong GU, Ai Han, TianLong Li - [[pdf]](https://arxiv.org/pdf/2506.19500)</summary>

**Abstract:** Large language models (LLMs) have recently demonstrated the ability to act as function call agents by invoking external tools, enabling them to solve tasks beyond their static knowledge. However, existing agents typically call tools step by step at a time without a global view of task structure. As tools depend on each other, this leads to error accumulation and limited scalability, particularly when scaling to thousands of tools. To address these limitations, we propose NaviAgent, a novel bilevel architecture that decouples task planning from tool execution through graph-based modeling of the tool ecosystem. At the task-planning level, the LLM-based agent decides whether to respond directly, clarify user intent, invoke a toolchain, or execute tool outputs, ensuring broad coverage of interaction scenarios independent of inter-tool complexity. At the execution level, a continuously evolving Tool World Navigation Model (TWNM) encodes structural and behavioral relations among tools, guiding the agent to generate scalable and robust invocation sequences. By incorporating feedback from real tool interactions, NaviAgent supports closed-loop optimization of planning and execution, moving beyond tool calling toward adaptive navigation of large-scale tool ecosystems. Experiments show that NaviAgent achieves the best task success rates across models and tasks, and integrating TWMN further boosts performance by up to 17 points on complex tasks, underscoring its key role in toolchain orchestration.

**arXiv ID:** 2506.19500
</details>

<details>
<summary><strong>Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization</strong> - Cheng Tang, Zhishuai Liu, Pan Xu - [[pdf]](https://arxiv.org/pdf/2411.18612)</summary>

**Abstract:** The Robust Regularized Markov Decision Process (RRMDP) is proposed to learn policies robust to dynamics shifts by adding regularization to the transition dynamics in the value function. Existing methods mostly use unstructured regularization, potentially leading to conservative policies under unrealistic transitions. To address this limitation, we propose a novel framework, the $d$-rectangular linear RRMDP ($d$-RRMDP), which introduces latent structures into both transition kernels and regularization. We focus on offline reinforcement learning, where an agent learns policies from a precollected dataset in the nominal environment. We develop the Robust Regularized Pessimistic Value Iteration (R2PVI) algorithm that employs linear function approximation for robust policy learning in $d$-RRMDPs with $f$-divergence based regularization terms on transition kernels. We provide instance-dependent upper bounds on the suboptimality gap of R2PVI policies, demonstrating that these bounds are influenced by how well the dataset covers state-action spaces visited by the optimal robust policy under robustly admissible transitions. We establish information-theoretic lower bounds to verify that our algorithm is near-optimal. Finally, numerical experiments validate that R2PVI learns robust policies and exhibits superior computational efficiency compared to baseline methods.

**arXiv ID:** 2411.18612
</details>

<details>
<summary><strong>On the Mistaken Assumption of Interchangeable Deep Reinforcement Learning Implementations</strong> - Rajdeep Singh Hundal, Yan Xiao, Xiaochun Cao, Jin Song Dong, Manuel Rigger - [[pdf]](https://arxiv.org/pdf/2503.22575)</summary>

**Abstract:** Deep Reinforcement Learning (DRL) is a paradigm of artificial intelligence where an agent uses a neural network to learn which actions to take in a given environment. DRL has recently gained traction from being able to solve complex environments like driving simulators, 3D robotic control, and multiplayer-online-battle-arena video games. Numerous implementations of the state-of-the-art algorithms responsible for training these agents, like the Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms, currently exist. However, studies make the mistake of assuming implementations of the same algorithm to be consistent and thus, interchangeable. In this paper, through a differential testing lens, we present the results of studying the extent of implementation inconsistencies, their effect on the implementations' performance, as well as their impact on the conclusions of prior studies under the assumption of interchangeable implementations. The outcomes of our differential tests showed significant discrepancies between the tested algorithm implementations, indicating that they are not interchangeable. In particular, out of the five PPO implementations tested on 56 games, three implementations achieved superhuman performance for 50% of their total trials while the other two implementations only achieved superhuman performance for less than 15% of their total trials. As part of a meticulous manual analysis of the implementations' source code, we analyzed implementation discrepancies and determined that code-level inconsistencies primarily caused these discrepancies. Lastly, we replicated a study and showed that this assumption of implementation interchangeability was sufficient to flip experiment outcomes. Therefore, this calls for a shift in how implementations are being used.

**arXiv ID:** 2503.22575
</details>

<details>
<summary><strong>Uncertainty-Based Smooth Policy Regularisation for Reinforcement Learning with Few Demonstrations</strong> - Yujie Zhu, Charles A. Hepburn, Matthew Thorpe, Giovanni Montana - [[pdf]](https://arxiv.org/pdf/2509.15981)</summary>

**Abstract:** In reinforcement learning with sparse rewards, demonstrations can accelerate learning, but determining when to imitate them remains challenging. We propose Smooth Policy Regularisation from Demonstrations (SPReD), a framework that addresses the fundamental question: when should an agent imitate a demonstration versus follow its own policy? SPReD uses ensemble methods to explicitly model Q-value distributions for both demonstration and policy actions, quantifying uncertainty for comparisons. We develop two complementary uncertainty-aware methods: a probabilistic approach estimating the likelihood of demonstration superiority, and an advantage-based approach scaling imitation by statistical significance. Unlike prevailing methods (e.g. Q-filter) that make binary imitation decisions, SPReD applies continuous, uncertainty-proportional regularisation weights, reducing gradient variance during training. Despite its computational simplicity, SPReD achieves remarkable gains in experiments across eight robotics tasks, outperforming existing approaches by up to a factor of 14 in complex tasks while maintaining robustness to demonstration quality and quantity. Our code is available at this https URL.

**arXiv ID:** 2509.15981
</details>

<details>
<summary><strong>Unlocking Reasoning Capabilities in LLMs via Reinforcement Learning Exploration</strong> - Wenhao Deng, Long Wei, Chenglei Yu, Tailin Wu - [[pdf]](https://arxiv.org/pdf/2510.03865)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has recently enhanced the reasoning capabilities of large language models (LLMs), particularly for mathematical problem solving. However, a fundamental limitation remains: as the sampling budget increases, the advantage of RLVR-trained models over their pretrained bases often diminishes or even vanishes, revealing a strong dependence on the base model's restricted search space. We attribute this phenomenon to the widespread use of the reverse Kullback-Leibler (KL) divergence regularizer, whose mode-seeking behavior keeps the policy trapped inside the base model's support region and hampers wider exploration. To address this issue, we propose RAPO (Rewards-Aware Policy Optimization), an algorithm to promote broader yet focused exploration. Our method (i) utilizes the forward KL penalty to replace the reverse KL penalty for out-of-distribution exploration, and (ii) reweights the reference policy to facilitate adaptive in-distribution exploration. We train Qwen2.5-3B and 7B models with RAPO on the 8K SimpleRL-Zero dataset, without supervised fine-tuning, and evaluate them on AIME2024 and AIME2025. Results show that RAPO consistently improves problem-solving performance. Notably, RAPO enables models to surpass the base model's performance ceiling and solves previously intractable problems, advancing the frontier of RLVR for challenging reasoning tasks.

**arXiv ID:** 2510.03865
</details>

<details>
<summary><strong>CogPlanner: Unveiling the Potential of Agentic Multimodal Retrieval Augmented Generation with Planning</strong> - Xiaohan Yu, Zhihan Yang, Chong Chen - [[pdf]](https://arxiv.org/pdf/2501.15470)</summary>

**Abstract:** Multimodal Retrieval Augmented Generation (MRAG) systems have shown promise in enhancing the generation capabilities of multimodal large language models (MLLMs). However, existing MRAG frameworks primarily adhere to rigid, single-step retrieval strategies that fail to address real-world challenges of information acquisition and query reformulation. In this work, we introduce the task of Multimodal Retrieval Augmented Generation Planning (MRAG Planning) that aims at effective information seeking and integration while minimizing computational overhead. Specifically, we propose CogPlanner, an agentic plug-and-play framework inspired by human cognitive processes, which iteratively determines query reformulation and retrieval strategies to generate accurate and contextually relevant responses. CogPlanner supports parallel and sequential modeling paradigms. Furthermore, we introduce CogBench, a new benchmark designed to rigorously evaluate the MRAG Planning task and facilitate lightweight CogPlanner integration with resource-efficient MLLMs, such as Qwen2-VL-7B-Cog. Experimental results demonstrate that CogPlanner significantly outperforms existing MRAG baselines, offering improvements in both accuracy and efficiency with minimal additional computational costs.

**arXiv ID:** 2501.15470
</details>

<details>
<summary><strong>MARAG-R1: Beyond Single Retriever via Reinforcement-Learned Multi-Tool Agentic Retrieval</strong> - Qi Luo, Xiaonan Li, Yuxin Wang, Tingshuo Fan, Yuan Li, Xinchi Chen, Xipeng Qiu - [[pdf]](https://arxiv.org/pdf/2510.27569)</summary>

**Abstract:** Large Language Models (LLMs) excel at reasoning and generation but are inherently limited by static pretraining data, resulting in factual inaccuracies and weak adaptability to new information. Retrieval-Augmented Generation (RAG) addresses this issue by grounding LLMs in external knowledge; However, the effectiveness of RAG critically depends on whether the model can adequately access relevant information. Existing RAG systems rely on a single retriever with fixed top-k selection, restricting access to a narrow and static subset of the corpus. As a result, this single-retriever paradigm has become the primary bottleneck for comprehensive external information acquisition, especially in tasks requiring corpus-level reasoning. To overcome this limitation, we propose MARAG-R1, a reinforcement-learned multi-tool RAG framework that enables LLMs to dynamically coordinate multiple retrieval mechanisms for broader and more precise information access. MARAG-R1 equips the model with four retrieval tools -- semantic search, keyword search, filtering, and aggregation -- and learns both how and when to use them through a two-stage training process: supervised fine-tuning followed by reinforcement learning. This design allows the model to interleave reasoning and retrieval, progressively gathering sufficient evidence for corpus-level synthesis. Experiments on GlobalQA, HotpotQA, and 2WikiMultiHopQA demonstrate that MARAG-R1 substantially outperforms strong baselines and achieves new state-of-the-art results in corpus-level reasoning tasks.

**arXiv ID:** 2510.27569
</details>

<details>
<summary><strong>AI Agents in Drug Discovery</strong> - Srijit Seal, Dinh Long Huynh, Moudather Chelbi, Sara Khosravi, Ankur Kumar, Mattson Thieme, Isaac Wilks, Mark Davies, Jessica Mustali, Yannick Sun, Nick Edwards, Daniil Boiko, Andrei Tyrin, Douglas W. Selinger, Ayaan Parikh, Rahul Vijayan, Shoman Kasbekar, Dylan Reid, Andreas Bender, Ola Spjuth - [[pdf]](https://arxiv.org/pdf/2510.27130)</summary>

**Abstract:** Artificial intelligence (AI) agents are emerging as transformative tools in drug discovery, with the ability to autonomously reason, act, and learn through complicated research workflows. Building on large language models (LLMs) coupled with perception, computation, action, and memory tools, these agentic AI systems could integrate diverse biomedical data, execute tasks, carry out experiments via robotic platforms, and iteratively refine hypotheses in closed loops. We provide a conceptual and technical overview of agentic AI architectures, ranging from ReAct and Reflection to Supervisor and Swarm systems, and illustrate their applications across key stages of drug discovery, including literature synthesis, toxicity prediction, automated protocol generation, small-molecule synthesis, drug repurposing, and end-to-end decision-making. To our knowledge, this represents the first comprehensive work to present real-world implementations and quantifiable impacts of agentic AI systems deployed in operational drug discovery settings. Early implementations demonstrate substantial gains in speed, reproducibility, and scalability, compressing workflows that once took months into hours while maintaining scientific traceability. We discuss the current challenges related to data heterogeneity, system reliability, privacy, and benchmarking, and outline future directions towards technology in support of science and translation.

**arXiv ID:** 2510.27130
</details>

<details>
<summary><strong>Diabetes Lifestyle Medicine Treatment Assistance Using Reinforcement Learning</strong> - Yuhan Tang - [[pdf]](https://arxiv.org/pdf/2510.26807)</summary>

**Abstract:** Type 2 diabetes prevention and treatment can benefit from personalized lifestyle prescriptions. However, the delivery of personalized lifestyle medicine prescriptions is limited by the shortage of trained professionals and the variability in physicians' expertise. We propose an offline contextual bandit approach that learns individualized lifestyle prescriptions from the aggregated NHANES profiles of 119,555 participants by minimizing the Magni glucose risk-reward function. The model encodes patient status and generates lifestyle medicine prescriptions, which are trained using a mixed-action Soft Actor-Critic algorithm. The task is treated as a single-step contextual bandit. The model is validated against lifestyle medicine prescriptions issued by three certified physicians from Xiangya Hospital. These results demonstrate that offline mixed-action SAC can generate risk-aware lifestyle medicine prescriptions from cross-sectional NHANES data, warranting prospective clinical validation.

**arXiv ID:** 2510.26807
</details>

<details>
<summary><strong>When AI Trading Agents Compete: Adverse Selection of Meta-Orders by Reinforcement Learning-Based Market Making</strong> - Ali Raza Jafree, Konark Jain, Nick Firoozye - [[pdf]](https://arxiv.org/pdf/2510.27334)</summary>

**Abstract:** We investigate the mechanisms by which medium-frequency trading agents are adversely selected by opportunistic high-frequency traders. We use reinforcement learning (RL) within a Hawkes Limit Order Book (LOB) model in order to replicate the behaviours of high-frequency market makers. In contrast to the classical models with exogenous price impact assumptions, the Hawkes model accounts for endogenous price impact and other key properties of the market (Jain et al. 2024a). Given the real-world impracticalities of the market maker updating strategies for every event in the LOB, we formulate the high-frequency market making agent via an impulse control reinforcement learning framework (Jain et al. 2025). The RL used in the simulation utilises Proximal Policy Optimisation (PPO) and self-imitation learning. To replicate the adverse selection phenomenon, we test the RL agent trading against a medium frequency trader (MFT) executing a meta-order and demonstrate that, with training against the MFT meta-order execution agent, the RL market making agent learns to capitalise on the price drift induced by the meta-order. Recent empirical studies have shown that medium-frequency traders are increasingly subject to adverse selection by high-frequency trading agents. As high-frequency trading continues to proliferate across financial markets, the slippage costs incurred by medium-frequency traders are likely to increase over time. However, we do not observe that increased profits for the market making RL agent necessarily cause significantly increased slippages for the MFT agent.

**arXiv ID:** 2510.27334
</details>

<details>
<summary><strong>Generating Auxiliary Tasks with Reinforcement Learning</strong> - Judah Goldfeder, Matthew So, Hod Lipson - [[pdf]](https://arxiv.org/pdf/2510.22940)</summary>

**Abstract:** Auxiliary Learning (AL) is a form of multi-task learning in which a model trains on auxiliary tasks to boost performance on a primary objective. While AL has improved generalization across domains such as navigation, image classification, and NLP, it often depends on human-labeled auxiliary tasks that are costly to design and require domain expertise. Meta-learning approaches mitigate this by learning to generate auxiliary tasks, but typically rely on gradient based bi-level optimization, adding substantial computational and implementation overhead. We propose RL-AUX, a reinforcement-learning (RL) framework that dynamically creates auxiliary tasks by assigning auxiliary labels to each training example, rewarding the agent whenever its selections improve the performance on the primary task. We also explore learning per-example weights for the auxiliary loss. On CIFAR-100 grouped into 20 superclasses, our RL method outperforms human-labeled auxiliary tasks and matches the performance of a prominent bi-level optimization baseline. We present similarly strong results on other classification datasets. These results suggest RL is a viable path to generating effective auxiliary tasks.

**arXiv ID:** 2510.22940
</details>

<details>
<summary><strong>A Practical-Driven Framework for Transitioning Drive-by-Wire to Autonomous Driving Systems: A Case Study with a Chrysler Pacifica Hybrid Vehicle</strong> - Dada Zhang, Md Ruman Islam, Pei-Chi Huang, Chun-Hsing Ho - [[pdf]](https://arxiv.org/pdf/2410.06492)</summary>

**Abstract:** Transitioning from a Drive-by-Wire (DBW) system to a fully autonomous driving system (ADS) involves multiple stages of development and demands robust positioning and sensing capabilities. This paper presents a practice-driven framework for facilitating the DBW-to-ADS transition using a 2022 Chrysler Pacifica Hybrid Minivan equipped with cameras, LiDAR, GNSS, and onboard computing hardware configured with the Robot Operating System (ROS) and this http URL. The implementation showcases offline autonomous operations utilizing pre-recorded LiDAR and camera data, point clouds, and vector maps, enabling effective localization and path planning within a structured test environment. The study addresses key challenges encountered during the transition, particularly those related to wireless-network-assisted sensing and positioning. It offers practical solutions for overcoming software incompatibility constraints, sensor synchronization issues, and limitations in real-time perception. Furthermore, the integration of sensing, data fusion, and automation is emphasized as a critical factor in supporting autonomous driving systems in map generation, simulation, and training. Overall, the transition process outlined in this work aims to provide actionable strategies for researchers pursuing DBW-to-ADS conversion. It offers direction for incorporating real-time perception, GNSS-LiDAR-camera integration, and fully ADS-equipped autonomous vehicle operations, thus contributing to the advancement of robust autonomous vehicle technologies.

**arXiv ID:** 2410.06492
</details>

<details>
<summary><strong>Mechanical Intelligence-Aware Curriculum Reinforcement Learning for Humanoids with Parallel Actuation</strong> - Yusuke Tanaka, Alvin Zhu, Quanyou Wang, Yeting Liu, Dennis Hong - [[pdf]](https://arxiv.org/pdf/2507.00273)</summary>

**Abstract:** Reinforcement learning (RL) has enabled advances in humanoid robot locomotion, yet most learning frameworks do not account for mechanical intelligence embedded in parallel actuation mechanisms due to limitations in simulator support for closed kinematic chains. This omission can lead to inaccurate motion modeling and suboptimal policies, particularly for robots with high actuation complexity. This paper presents general formulations and simulation methods for three types of parallel mechanisms: a differential pulley, a five-bar linkage, and a four-bar linkage, and trains a parallel-mechanism aware policy through an end-to-end curriculum RL framework for BRUCE, a kid-sized humanoid robot. Unlike prior approaches that rely on simplified serial approximations, we simulate all closed-chain constraints natively using GPU-accelerated MuJoCo (MJX), preserving the hardware's mechanical nonlinear properties during training. We benchmark our RL approach against a model predictive controller (MPC), demonstrating better surface generalization and performance in real-world zero-shot deployment. This work highlights the computational approaches and performance benefits of fully simulating parallel mechanisms in end-to-end learning pipelines for legged humanoids. Project codes with parallel mechanisms: this https URL

**arXiv ID:** 2507.00273
</details>

<details>
<summary><strong>Adaptive Human-Computer Interaction Strategies Through Reinforcement Learning in Complex</strong> - Rui Liu, Yifan Zhuang, Runsheng Zhang - [[pdf]](https://arxiv.org/pdf/2510.27058)</summary>

**Abstract:** This study addresses the challenges of dynamics and complexity in intelligent human-computer interaction and proposes a reinforcement learning-based optimization framework to improve long-term returns and overall experience. Human-computer interaction is modeled as a Markov decision process, with state space, action space, reward function, and discount factor defined to capture the dynamics of user input, system feedback, and interaction environment. The method combines policy function, value function, and advantage function, updates parameters through policy gradient, and continuously adjusts during interaction to balance immediate feedback and long-term benefits. To validate the framework, multimodal dialog and scene-aware datasets are used as the experimental platform, with multiple sensitivity experiments conducted on key factors such as discount factor, exploration rate decay, environmental noise, and data imbalance. Evaluation is carried out using cumulative reward, average episode reward, convergence speed, and task success rate. Results show that the proposed method outperforms existing approaches across several metrics, achieving higher task completion while maintaining strategy stability. Comparative experiments further confirm its advantages in interaction efficiency and long-term return, demonstrating the significant value of reinforcement learning in optimizing human-computer interaction.

**arXiv ID:** 2510.27058
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
