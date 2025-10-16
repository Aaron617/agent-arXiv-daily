# Agent arXiv Daily

**Last Updated:** 2025-10-16 02:41:22

**Total Papers:** 67

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
<summary><strong>On Your Own: Pro-level Autonomous Drone Racing in Uninstrumented Arenas</strong> - Michael Bosello, Flavio Pinzarrone, Sara Kiade, Davide Aguiari, Yvo Keuter, Aaesha AlShehhi, Gyordan Caminati, Kei Long Wong, Ka Seng Chou, Junaid Halepota, Fares Alneyadi, Jacopo Panerati, Giovanni Pau - [[pdf]](https://arxiv.org/pdf/2510.13644)</summary>

**Abstract:** Drone technology is proliferating in many industries, including agriculture, logistics, defense, infrastructure, and environmental monitoring. Vision-based autonomy is one of its key enablers, particularly for real-world applications. This is essential for operating in novel, unstructured environments where traditional navigation methods may be unavailable. Autonomous drone racing has become the de facto benchmark for such systems. State-of-the-art research has shown that autonomous systems can surpass human-level performance in racing arenas. However, direct applicability to commercial and field operations is still limited as current systems are often trained and evaluated in highly controlled environments. In our contribution, the system's capabilities are analyzed within a controlled environment -- where external tracking is available for ground-truth comparison -- but also demonstrated in a challenging, uninstrumented environment -- where ground-truth measurements were never available. We show that our approach can match the performance of professional human pilots in both scenarios. We also publicly release the data from the flights carried out by our approach and a world-class human pilot.

**arXiv ID:** 2510.13644
</details>

<details>
<summary><strong>A Verification Methodology for Safety Assurance of Robotic Autonomous Systems</strong> - Mustafa Adam, David A. Anisi, Pedro Ribeiro - [[pdf]](https://arxiv.org/pdf/2506.19622)</summary>

**Abstract:** Autonomous robots deployed in shared human environments, such as agricultural settings, require rigorous safety assurance to meet both functional reliability and regulatory compliance. These systems must operate in dynamic, unstructured environments, interact safely with humans, and respond effectively to a wide range of potential hazards. This paper presents a verification workflow for the safety assurance of an autonomous agricultural robot, covering the entire development life-cycle, from concept study and design to runtime verification. The outlined methodology begins with a systematic hazard analysis and risk assessment to identify potential risks and derive corresponding safety requirements. A formal model of the safety controller is then developed to capture its behaviour and verify that the controller satisfies the specified safety properties with respect to these requirements. The proposed approach is demonstrated on a field robot operating in an agricultural setting. The results show that the methodology can be effectively used to verify safety-critical properties and facilitate the early identification of design issues, contributing to the development of safer robots and autonomous systems.

**arXiv ID:** 2506.19622
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>An Analytical Framework to Enhance Autonomous Vehicle Perception for Smart Cities</strong> - Jalal Khan, Manzoor Khan, Sherzod Turaev, Sumbal Malik, Hesham El-Sayed, Farman Ullah - [[pdf]](https://arxiv.org/pdf/2510.13230)</summary>

**Abstract:** The driving environment perception has a vital role for autonomous driving and nowadays has been actively explored for its realization. The research community and relevant stakeholders necessitate the development of Deep Learning (DL) models and AI-enabled solutions to enhance autonomous vehicles (AVs) for smart mobility. There is a need to develop a model that accurately perceives multiple objects on the road and predicts the driver's perception to control the car's movements. This article proposes a novel utility-based analytical model that enables perception systems of AVs to understand the driving environment. The article consists of modules: acquiring a custom dataset having distinctive objects, i.e., motorcyclists, rickshaws, etc; a DL-based model (YOLOv8s) for object detection; and a module to measure the utility of perception service from the performance values of trained model instances. The perception model is validated based on the object detection task, and its process is benchmarked by state-of-the-art deep learning models' performance metrics from the nuScense dataset. The experimental results show three best-performing YOLOv8s instances based on mAP@0.5 values, i.e., SGD-based (0.832), Adam-based (0.810), and AdamW-based (0.822). However, the AdamW-based model (i.e., car: 0.921, motorcyclist: 0.899, truck: 0.793, etc.) still outperforms the SGD-based model (i.e., car: 0.915, motorcyclist: 0.892, truck: 0.781, etc.) because it has better class-level performance values, confirmed by the proposed perception model. We validate that the proposed function is capable of finding the right perception for AVs. The results above encourage using the proposed perception model to evaluate the utility of learning models and determine the appropriate perception for AVs.

**arXiv ID:** 2510.13230
</details>

<details>
<summary><strong>Modeling Cultural Bias in Facial Expression Recognition with Adaptive Agents</strong> - David Freire-Obregón, José Salas-Cáceres, Javier Lorenzo-Navarro, Oliverio J. Santana, Daniel Hernández-Sosa, Modesto Castrillón-Santana - [[pdf]](https://arxiv.org/pdf/2510.13557)</summary>

**Abstract:** Facial expression recognition (FER) must remain robust under both cultural variation and perceptually degraded visual conditions, yet most existing evaluations assume homogeneous data and high-quality imagery. We introduce an agent-based, streaming benchmark that reveals how cross-cultural composition and progressive blurring interact to shape face recognition robustness. Each agent operates in a frozen CLIP feature space with a lightweight residual adapter trained online at sigma=0 and fixed during testing. Agents move and interact on a 5x5 lattice, while the environment provides inputs with sigma-scheduled Gaussian blur. We examine monocultural populations (Western-only, Asian-only) and mixed environments with balanced (5/5) and imbalanced (8/2, 2/8) compositions, as well as different spatial contact structures. Results show clear asymmetric degradation curves between cultural groups: JAFFE (Asian) populations maintain higher performance at low blur but exhibit sharper drops at intermediate stages, whereas KDEF (Western) populations degrade more uniformly. Mixed populations exhibit intermediate patterns, with balanced mixtures mitigating early degradation, but imbalanced settings amplify majority-group weaknesses under high blur. These findings quantify how cultural composition and interaction structure influence the robustness of FER as perceptual conditions deteriorate.

**arXiv ID:** 2510.13557
</details>

<details>
<summary><strong>TASER: Table Agents for Schema-guided Extraction and Recommendation</strong> - Nicole Cho, Kirsty Fielding, William Watson, Sumitra Ganesh, Manuela Veloso - [[pdf]](https://arxiv.org/pdf/2508.13404)</summary>

**Abstract:** Real-world financial documents report essential information about an entity's financial holdings that can span millions of different financial instrument types. Yet, these details are often buried in messy, multi-page, fragmented tables - for example, 99.4% of the tables in our dataset have no bounding boxes with the maximum number of rows amounting to 426 per table across 44 pages. To tackle these unique challenges from real-world tables, we present a continuously learning, agentic table extraction system, TASER (Table Agents for Schema-guided Extraction and Recommendation) that extracts highly unstructured, multi-page, heterogeneous tables into normalized, schema-conforming outputs. Our table agents execute on table detection, classification, extraction, and recommendations by leveraging an initial schema. Then, our Recommender Agent reviews the outputs, recommends schema revisions, and decides on the final recommendations, enabling TASER to outperform existing table detection models such as Table Transformer by 10.1%. Within this continuous learning process, we highlight that larger batch sizes result in a 104.3% increase in schema recommendations that are actionable and utilized, resulting in a 9.8% increase in extracted holdings - highlighting the importance of a continuous learning process. To train TASER, we have manually labeled 22,584 pages (28,150,449 tokens), 3,213 tables for $731,685,511,687 of holdings culminating in one of the first real financial table datasets. We release our dataset TASERTab to enable the research community to access real-world financial tables and outputs. Our results highlight the promise of agentic, schema-guided extraction systems for robust understanding of real-world financial tables.

**arXiv ID:** 2508.13404
</details>

<details>
<summary><strong>FlashAdventure: A Benchmark for GUI Agents Solving Full Story Arcs in Diverse Adventure Games</strong> - Jaewoo Ahn, Junseo Kim, Heeseung Yun, Jaehyeon Son, Dongmin Park, Jaewoong Cho, Gunhee Kim - [[pdf]](https://arxiv.org/pdf/2509.01052)</summary>

**Abstract:** GUI agents powered by LLMs show promise in interacting with diverse digital environments. Among these, video games offer a valuable testbed due to their varied interfaces, with adventure games posing additional challenges through complex, narrative-driven interactions. Existing game benchmarks, however, lack diversity and rarely evaluate agents on completing entire storylines. To address this, we introduce FlashAdventure, a benchmark of 34 Flash-based adventure games designed to test full story arc completion and tackle the observation-behavior gap: the challenge of remembering and acting on earlier gameplay information. We also propose CUA-as-a-Judge, an automated gameplay evaluator, and COAST, an agentic framework leveraging long-term clue memory to better plan and solve sequential tasks. Experiments show current GUI agents struggle with full story arcs, while COAST improves milestone completion by bridging the observation-behavior gap. Nonetheless, a marked discrepancy between humans and best-performing agents warrants continued research efforts to narrow this divide.

**arXiv ID:** 2509.01052
</details>

<details>
<summary><strong>SafeSearch: Automated Red-Teaming for the Safety of LLM-Based Search Agents</strong> - Jianshuo Dong, Sheng Guo, Hao Wang, Xun Chen, Zhuotao Liu, Tianwei Zhang, Ke Xu, Minlie Huang, Han Qiu - [[pdf]](https://arxiv.org/pdf/2509.23694)</summary>

**Abstract:** Search agents connect LLMs to the Internet, enabling access to broader and more up-to-date information. However, unreliable search results may also pose safety threats to end users, establishing a new threat surface. In this work, we conduct two in-the-wild experiments to demonstrate both the prevalence of low-quality search results and their potential to misguide agent behaviors. To counter this threat, we introduce an automated red-teaming framework that is systematic, scalable, and cost-efficient, enabling lightweight and harmless safety assessments of search agents. Building on this framework, we construct the SafeSearch benchmark, which includes 300 test cases covering five categories of risks (e.g., misinformation and indirect prompt injection). Using this benchmark, we evaluate three representative search agent scaffolds, covering search workflow, tool-calling, and deep research, across 7 proprietary and 8 open-source backend LLMs. Our results reveal substantial vulnerabilities of LLM-based search agents: when exposed to unreliable websites, the highest ASR reached 90.5% for GPT-4.1-mini under a search workflow setting. Moreover, our analysis highlights the limited effectiveness of common defense practices, such as reminder prompting. This emphasizes the value of our framework in promoting transparency for safer agent development. Our codebase and test cases are publicly available: this https URL.

**arXiv ID:** 2509.23694
</details>

<details>
<summary><strong>A Tale of LLMs and Induced Small Proxies: Scalable Agents for Knowledge Mining</strong> - Sipeng Zhang, Longfei Yun, Zilong Wang, Jingbo Shang, Letian Peng - [[pdf]](https://arxiv.org/pdf/2510.01427)</summary>

**Abstract:** At the core of Deep Research is knowledge mining, the task of extracting structured information from massive unstructured text in response to user instructions. Large language models (LLMs) excel at interpreting such instructions but are prohibitively expensive to deploy at scale, while traditional pipelines of classifiers and extractors remain efficient yet brittle and unable to generalize to new tasks. We introduce Falconer, a collaborative framework that combines the agentic reasoning of LLMs with lightweight proxy models for scalable knowledge mining. In Falconer, LLMs act as planners, decomposing user instructions into executable pipelines, and as annotators, generating supervision to train small proxies. The framework unifies classification and extraction into two atomic operations, get label and get span, enabling a single instruction-following model to replace multiple task-specific components. To evaluate the consistency between proxy models incubated by Falconer and annotations provided by humans and large models, we construct new benchmarks covering both planning and end-to-end execution. Experiments show that Falconer closely matches state-of-the-art LLMs in instruction-following accuracy while reducing inference cost by up to 90% and accelerating large-scale knowledge mining by more than 20x, offering an efficient and scalable foundation for Deep Research.

**arXiv ID:** 2510.01427
</details>

<details>
<summary><strong>Evolution of AI Agent Registry Solutions: Centralized, Enterprise, and Distributed Approaches</strong> - Aditi Singh, Abul Ehtesham, Mahesh Lambe, Jared James Grogan, Abhishek Singh, Saket Kumar, Luca Muscariello, Vijoy Pandey, Guillaume Sauvage De Saint Marc, Pradyumna Chari, Ramesh Raskar - [[pdf]](https://arxiv.org/pdf/2508.03095)</summary>

**Abstract:** Autonomous AI agents now operate across cloud, enterprise, and decentralized domains, creating demand for registry infrastructures that enable trustworthy discovery, capability negotiation, and identity assurance. We analyze five prominent approaches: (1) MCP Registry (centralized publication of this http URL descriptors), (2) A2A Agent Cards (decentralized self-describing JSON capability manifests), (3) AGNTCY Agent Directory Service (IPFS Kademlia DHT content routing extended for semantic taxonomy-based content discovery, OCI artifact storage, and Sigstore-backed integrity), (4) Microsoft Entra Agent ID (enterprise SaaS directory with policy and zero-trust integration), and (5) NANDA Index AgentFacts (cryptographically verifiable, privacy-preserving fact model with credentialed assertions). Using four evaluation dimensions: security, authentication, scalability, and maintainability, we surface architectural trade-offs between centralized control, enterprise governance, and distributed resilience. We conclude with design recommendations for an emerging Internet of AI Agents requiring verifiable identity, adaptive discovery flows, and interoperable capability semantics.

**arXiv ID:** 2508.03095
</details>

<details>
<summary><strong>MATRIX: Multimodal Agent Tuning for Robust Tool-Use Reasoning</strong> - Tajamul Ashraf, Umair Nawaz, Abdelrahman M. Shaker, Rao Anwer, Philip Torr, Fahad Shahbaz Khan, Salman Khan - [[pdf]](https://arxiv.org/pdf/2510.08567)</summary>

**Abstract:** Vision language models (VLMs) are increasingly deployed as controllers with access to external tools for complex reasoning and decision-making, yet their effectiveness remains limited by the scarcity of high-quality multimodal trajectories and the cost of manual annotation. We address this challenge with a vision-centric agent tuning framework that automatically synthesizes multimodal trajectories, generates step-wise preference pairs, and trains a VLM controller for robust tool-use reasoning. Our pipeline first constructs M-TRACE, a large-scale dataset of 28.5K multimodal tasks with 177K verified trajectories, enabling imitation-based trajectory tuning. Building on this, we develop MATRIX Agent, a controller finetuned on M-TRACE for step-wise tool reasoning. To achieve finer alignment, we further introduce Pref-X, a set of 11K automatically generated preference pairs, and optimize MATRIX on it via step-wise preference learning. Across three benchmarks, Agent-X, GTA, and GAIA, MATRIX consistently surpasses both open- and closed-source VLMs, demonstrating scalable and effective multimodal tool use. Our data and code is avaliable at this https URL.

**arXiv ID:** 2510.08567
</details>

<details>
<summary><strong>Saving SWE-Bench: A Benchmark Mutation Approach for Realistic Agent Evaluation</strong> - Spandan Garg, Benjamin Steenhoek, Yufan Huang - [[pdf]](https://arxiv.org/pdf/2510.08996)</summary>

**Abstract:** Current benchmarks for evaluating software engineering agents, such as SWE-Bench Verified, are predominantly derived from GitHub issues and fail to accurately reflect how developers interact with chat-based coding assistants in integrated development environments (IDEs). We posit that this mismatch leads to a systematic overestimation of agent's capabilities in real-world scenarios, especially bug fixing. We introduce a novel benchmarking framework that transforms existing formal benchmarks into realistic user queries through systematic analysis of developer interaction patterns with chat-based agents. Our methodology is flexible and can be easily extended to existing benchmarks. In this paper, we apply our testing framework to SWE-Bench Verified, the TypeScript subset of Multi-SWE-Bench and a private benchmark, SWE-Bench C# and transform formal GitHub issue descriptions into realistic user-style queries based on telemetry analysis of a popular chat-based agent interactions. Our findings reveal that existing benchmarks significantly overestimate agent capabilities for some models by >50% over baseline performance for public benchmarks and ~10-16% for our internal benchmark. This work establishes a new paradigm for evaluating interactive chat-based software engineering agents through benchmark mutation techniques.

**arXiv ID:** 2510.08996
</details>

<details>
<summary><strong>EmbodiedCoder: Parameterized Embodied Mobile Manipulation via Modern Coding Model</strong> - Zefu Lin, Rongxu Cui, Chen Hanning, Xiangyu Wang, Junjia Xu, Xiaojuan Jin, Chen Wenbo, Hui Zhou, Lue Fan, Wenling Li, Zhaoxiang Zhang - [[pdf]](https://arxiv.org/pdf/2510.06207)</summary>

**Abstract:** Recent advances in control robot methods, from end-to-end vision-language-action frameworks to modular systems with predefined primitives, have advanced robots' ability to follow natural language instructions. Nonetheless, many approaches still struggle to scale to diverse environments, as they often rely on large annotated datasets and offer limited this http URL this work, we introduce EmbodiedCoder, a training-free framework for open-world mobile robot manipulation that leverages coding models to directly generate executable robot trajectories. By grounding high-level instructions in code, EmbodiedCoder enables flexible object geometry parameterization and manipulation trajectory synthesis without additional data collection or this http URL coding-based paradigm provides a transparent and generalizable way to connect perception with manipulation. Experiments on real mobile robots show that EmbodiedCoder achieves robust performance across diverse long-term tasks and generalizes effectively to novel objects and this http URL results demonstrate an interpretable approach for bridging high-level reasoning and low-level control, moving beyond fixed primitives toward versatile robot intelligence. See the project page at: this https URL

**arXiv ID:** 2510.06207
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>SENTINEL: A Multi-Level Formal Framework for Safety Evaluation of LLM-based Embodied Agents</strong> - Simon Sinong Zhan, Yao Liu, Philip Wang, Zinan Wang, Qineng Wang, Zhian Ruan, Xiangyu Shi, Xinyu Cao, Frank Yang, Kangrui Wang, Huajie Shao, Manling Li, Qi Zhu - [[pdf]](https://arxiv.org/pdf/2510.12985)</summary>

**Abstract:** We present Sentinel, the first framework for formally evaluating the physical safety of Large Language Model(LLM-based) embodied agents across the semantic, plan, and trajectory levels. Unlike prior methods that rely on heuristic rules or subjective LLM judgments, Sentinel grounds practical safety requirements in formal temporal logic (TL) semantics that can precisely specify state invariants, temporal dependencies, and timing constraints. It then employs a multi-level verification pipeline where (i) at the semantic level, intuitive natural language safety requirements are formalized into TL formulas and the LLM agent's understanding of these requirements is probed for alignment with the TL formulas; (ii) at the plan level, high-level action plans and subgoals generated by the LLM agent are verified against the TL formulas to detect unsafe plans before execution; and (iii) at the trajectory level, multiple execution trajectories are merged into a computation tree and efficiently verified against physically-detailed TL specifications for a final safety check. We apply Sentinel in VirtualHome and ALFRED, and formally evaluate multiple LLM-based embodied agents against diverse safety requirements. Our experiments show that by grounding physical safety in temporal logic and applying verification methods across multiple levels, Sentinel provides a rigorous foundation for systematically evaluating LLM-based embodied agents in physical environments, exposing safety violations overlooked by previous methods and offering insights into their failure modes.

**arXiv ID:** 2510.12985
</details>

<details>
<summary><strong>Training LLM Agents to Empower Humans</strong> - Evan Ellis, Vivek Myers, Jens Tuyls, Sergey Levine, Anca Dragan, Benjamin Eysenbach - [[pdf]](https://arxiv.org/pdf/2510.13709)</summary>

**Abstract:** Assistive agents should not only take actions on behalf of a human, but also step out of the way and cede control when there are important decisions to be made. However, current methods for building assistive agents, whether via mimicking expert humans or via RL finetuning on an inferred reward, often encourage agents to complete tasks on their own rather than truly assisting the human attain their objectives. Additionally, these methods often require costly explicit human feedback to provide a training signal. We propose a new approach to tuning assistive language models based on maximizing the human's empowerment, their ability to effect desired changes in the environment. Our empowerment-maximizing method, Empower, only requires offline text data, providing a self-supervised method for fine-tuning language models to better assist humans. To study the efficacy of our approach, we conducted an 18-person user study comparing our empowerment assistant with a strong baseline. Participants preferred our assistant 78% of the time (p=0.015), with a 31% higher acceptance rate and 38% fewer suggestions. Additionally, we introduce a new environment for evaluating multi-turn code assistance using simulated humans. Using this environment, we show that agents trained with Empower increase the success rate of a simulated human programmer on challenging coding questions by an average of 192% over an SFT baseline. With this empowerment objective, we provide a framework for useful aligned AI agents at scale using only offline data without the need for any additional human feedback or verifiable rewards.

**arXiv ID:** 2510.13709
</details>

<details>
<summary><strong>MADREC: A Multi-Aspect Driven LLM Agent for Explainable and Adaptive Recommendation</strong> - Jiin Park, Misuk Kim - [[pdf]](https://arxiv.org/pdf/2510.13371)</summary>

**Abstract:** Recent attempts to integrate large language models (LLMs) into recommender systems have gained momentum, but most remain limited to simple text generation or static prompt-based inference, failing to capture the complexity of user preferences and real-world interactions. This study proposes the Multi-Aspect Driven LLM Agent MADRec, an autonomous LLM-based recommender that constructs user and item profiles by unsupervised extraction of multi-aspect information from reviews and performs direct recommendation, sequential recommendation, and explanation generation. MADRec generates structured profiles via aspect-category-based summarization and applies Re-Ranking to construct high-density inputs. When the ground-truth item is missing from the output, the Self-Feedback mechanism dynamically adjusts the inference criteria. Experiments across multiple domains show that MADRec outperforms traditional and LLM-based baselines in both precision and explainability, with human evaluation further confirming the persuasiveness of the generated explanations.

**arXiv ID:** 2510.13371
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (18 papers)</h2></summary>

<details>
<summary><strong>Emotional Cognitive Modeling Framework with Desire-Driven Objective Optimization for LLM-empowered Agent in Social Simulation</strong> - Qun Ma, Xiao Xue, Xuwen Zhang, Zihan Zhao, Yuwei Guo, Ming Zhang - [[pdf]](https://arxiv.org/pdf/2510.13195)</summary>

**Abstract:** The advent of large language models (LLMs) has enabled agents to represent virtual humans in societal simulations, facilitating diverse interactions within complex social systems. However, existing LLM-based agents exhibit severe limitations in affective cognition: They fail to simulate the bounded rationality essential for bridging virtual and real-world services; They lack empirically validated integration mechanisms embedding emotions within agent decision architectures. This paper constructs an emotional cognition framework incorporating desire generation and objective management, designed to achieve emotion alignment between LLM-based agents and humans, modeling the complete decision-making process of LLM-based agents, encompassing state evolution, desire generation, objective optimization, decision generation, and action execution. This study implements the proposed framework within our proprietary multi-agent interaction environment. Experimental results demonstrate that agents governed by our framework not only exhibit behaviors congruent with their emotional states but also, in comparative assessments against other agent types, demonstrate superior ecological validity and generate decision outcomes that significantly more closely approximate human behavioral patterns.

**arXiv ID:** 2510.13195
</details>

<details>
<summary><strong>Adaptive Reasoning Executor: A Collaborative Agent System for Efficient Reasoning</strong> - Zehui Ling, Deshu Chen, Yichi Zhang, Yuchen Liu, Xigui Li, Xin Guo, Yuan Cheng - [[pdf]](https://arxiv.org/pdf/2510.13214)</summary>

**Abstract:** Recent advances in Large Language Models (LLMs) demonstrate that chain-of-thought prompting and deep reasoning substantially enhance performance on complex tasks, and multi-agent systems can further improve accuracy by enabling model debates. However, applying deep reasoning to all problems is computationally expensive. To mitigate these costs, we propose a complementary agent system integrating small and large LLMs. The small LLM first generates an initial answer, which is then verified by the large LLM. If correct, the answer is adopted directly; otherwise, the large LLM performs in-depth reasoning. Experimental results show that, for simple problems, our approach reduces the computational cost of the large LLM by more than 50% with negligible accuracy loss, while consistently maintaining robust performance on complex tasks.

**arXiv ID:** 2510.13214
</details>

<details>
<summary><strong>SAJA: A State-Action Joint Attack Framework on Multi-Agent Deep Reinforcement Learning</strong> - Weiqi Guo, Guanjun Liu, Ziyuan Zhou - [[pdf]](https://arxiv.org/pdf/2510.13262)</summary>

**Abstract:** Multi-Agent Deep Reinforcement Learning (MADRL) has shown potential for cooperative and competitive tasks such as autonomous driving and strategic gaming. However, models trained by MADRL are vulnerable to adversarial perturbations on states and actions. Therefore, it is essential to investigate the robustness of MADRL models from an attack perspective. Existing studies focus on either state-only attacks or action-only attacks, but do not consider how to effectively joint them. Simply combining state and action perturbations such as randomly perturbing states and actions does not exploit their potential synergistic effects. In this paper, we propose the State-Action Joint Attack (SAJA) framework that has a good synergistic effects. SAJA consists of two important phases: (1) In the state attack phase, a multi-step gradient ascent method utilizes both the actor network and the critic network to compute an adversarial state, and (2) in the action attack phase, based on the perturbed state, a second gradient ascent uses the critic network to craft the final adversarial action. Additionally, a heuristic regularizer measuring the distance between the perturbed actions and the original clean ones is added into the loss function to enhance the effectiveness of the critic's guidance. We evaluate SAJA in the Multi-Agent Particle Environment (MPE), demonstrating that (1) it outperforms and is more stealthy than state-only or action-only attacks, and (2) existing state or action defense methods cannot defend its attacks.

**arXiv ID:** 2510.13262
</details>

<details>
<summary><strong>KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems</strong> - Hancheng Ye, Zhengqi Gao, Mingyuan Ma, Qinsi Wang, Yuzhe Fu, Ming-Yu Chung, Yueqian Lin, Zhijian Liu, Jianyi Zhang, Danyang Zhuo, Yiran Chen - [[pdf]](https://arxiv.org/pdf/2510.12872)</summary>

**Abstract:** Multi-agent large language model (LLM) systems are increasingly adopted for complex language processing tasks that require communication and coordination among agents. However, these systems often suffer substantial overhead from repeated reprocessing of overlapping contexts across agents. In typical pipelines, once an agent receives a message from its predecessor, the full context-including prior turns-must be reprocessed from scratch, leading to inefficient processing. While key-value (KV) caching is an effective solution for avoiding redundant computation in single-agent settings where prefixes remain unchanged, it cannot be directly reused in multi-agent scenarios due to diverging prefixes introduced by agent-specific context extensions. We identify that the core challenge lies in the offset variance of KV-caches across agents. To address this, we propose KVCOMM, a training-free framework that enables efficient prefilling in multi-agent inference by reusing KV-caches and aligning cache offsets of overlapping contexts under diverse prefix contexts. KVCOMM estimates and adjusts KV-caches for shared content by referencing a pool of cached examples-termed anchors-that store observed cache deviations under varying prefixes. The anchor pool is maintained and updated online, allowing dynamic adaptation to distinct user requests and context structures. KVCOMM achieves over 70% reuse rate across diverse multi-agent workloads, including retrieval-augmented generation, math reasoning, and collaborative coding tasks, all without quality degradation. Particularly, when each fully-connected agent receives 1K input tokens with 512 prefix tokens and 512 output tokens under a five-agent setting, KVCOMM achieves up to 7.8x speedup compared to the standard prefill pipeline, reducing TTFT from ~430 ms to ~55 ms.

**arXiv ID:** 2510.12872
</details>

<details>
<summary><strong>Agentic Discovery: Closing the Loop with Cooperative Agents</strong> - J. Gregory Pauloski, Kyle Chard, Ian T. Foster - [[pdf]](https://arxiv.org/pdf/2510.13081)</summary>

**Abstract:** As data-driven methods, artificial intelligence (AI), and automated workflows accelerate scientific tasks, we see the rate of discovery increasingly limited by human decision-making tasks such as setting objectives, generating hypotheses, and designing experiments. We postulate that cooperative agents are needed to augment the role of humans and enable autonomous discovery. Realizing such agents will require progress in both AI and infrastructure.

**arXiv ID:** 2510.13081
</details>

<details>
<summary><strong>Higher Satisfaction, Lower Cost: A Technical Report on How LLMs Revolutionize Meituan's Intelligent Interaction Systems</strong> - Xuxin Cheng, Ke Zeng, Zhiquan Cao, Linyi Dai, Wenxuan Gao, Fei Han, Ai Jian, Feng Hong, Wenxing Hu, Zihe Huang, Dejian Kong, Jia Leng, Zhuoyuan Liao, Pei Liu, Jiaye Lin, Xing Ma, Jingqing Ruan, Jiaxing Song, Xiaoyu Tan, Ruixuan Xiao, Wenhui Yu, Wenyu Zhan, Haoxing Zhang, Chao Zhou, Hao Zhou, Shaodong Zheng, Ruinian Chen, Siyuan Chen, Ziyang Chen, Yiwen Dong, Yaoyou Fan, Yangyi Fang, Yang Gan, Shiguang Guo, Qi He, Chaowen Hu, Binghui Li, Dailin Li, Xiangyu Li, Yan Li, Chengjian Liu, Xiangfeng Liu, Jiahui Lv, Qiao Ma, Jiang Pan, Cong Qin, Chenxing Sun, Wen Sun, Zhonghui Wang, Abudukelimu Wuerkaixi, Xin Yang, Fangyi Yuan, Yawen Zhu, Tianyi Zhai, Jie Zhang, Runlai Zhang, Yao Xu, Yiran Zhao, Yifan Wang, Xunliang Cai, Yangen Hu, Cao Liu, Lu Pan, Xiaoli Wang, Bo Xiao, Wenyuan Yao, Qianlin Zhou, Benchang Zhu - [[pdf]](https://arxiv.org/pdf/2510.13291)</summary>

**Abstract:** Enhancing customer experience is essential for business success, particularly as service demands grow in scale and complexity. Generative artificial intelligence and Large Language Models (LLMs) have empowered intelligent interaction systems to deliver efficient, personalized, and 24/7 support. In practice, intelligent interaction systems encounter several challenges: (1) Constructing high-quality data for cold-start training is difficult, hindering self-evolution and raising labor costs. (2) Multi-turn dialogue performance remains suboptimal due to inadequate intent understanding, rule compliance, and solution extraction. (3) Frequent evolution of business rules affects system operability and transferability, constraining low-cost expansion and adaptability. (4) Reliance on a single LLM is insufficient in complex scenarios, where the absence of multi-agent frameworks and effective collaboration undermines process completeness and service quality. (5) The open-domain nature of multi-turn dialogues, lacking unified golden answers, hampers quantitative evaluation and continuous optimization. To address these challenges, we introduce WOWService, an intelligent interaction system tailored for industrial applications. With the integration of LLMs and multi-agent architectures, WOWService enables autonomous task management and collaborative problem-solving. Specifically, WOWService focuses on core modules including data construction, general capability enhancement, business scenario adaptation, multi-agent coordination, and automated evaluation. Currently, WOWService is deployed on the Meituan App, achieving significant gains in key metrics, e.g., User Satisfaction Metric 1 (USM 1) -27.53% and User Satisfaction Metric 2 (USM 2) +25.51%, demonstrating its effectiveness in capturing user needs and advancing personalized service.

**arXiv ID:** 2510.13291
</details>

<details>
<summary><strong>AOAD-MAT: Transformer-based multi-agent deep reinforcement learning model considering agents' order of action decisions</strong> - Shota Takayama, Katsuhide Fujita - [[pdf]](https://arxiv.org/pdf/2510.13343)</summary>

**Abstract:** Multi-agent reinforcement learning focuses on training the behaviors of multiple learning agents that coexist in a shared environment. Recently, MARL models, such as the Multi-Agent Transformer (MAT) and ACtion dEpendent deep Q-learning (ACE), have significantly improved performance by leveraging sequential decision-making processes. Although these models can enhance performance, they do not explicitly consider the importance of the order in which agents make decisions. In this paper, we propose an Agent Order of Action Decisions-MAT (AOAD-MAT), a novel MAT model that considers the order in which agents make decisions. The proposed model explicitly incorporates the sequence of action decisions into the learning process, allowing the model to learn and predict the optimal order of agent actions. The AOAD-MAT model leverages a Transformer-based actor-critic architecture that dynamically adjusts the sequence of agent actions. To achieve this, we introduce a novel MARL architecture that cooperates with a subtask focused on predicting the next agent to act, integrated into a Proximal Policy Optimization based loss function to synergistically maximize the advantage of the sequential decision-making. The proposed method was validated through extensive experiments on the StarCraft Multi-Agent Challenge and Multi-Agent MuJoCo benchmarks. The experimental results show that the proposed AOAD-MAT model outperforms existing MAT and other baseline models, demonstrating the effectiveness of adjusting the AOAD order in MARL.

**arXiv ID:** 2510.13343
</details>

<details>
<summary><strong>Improving Planning with Large Language Models: A Modular Agentic Architecture</strong> - Taylor Webb, Shanka Subhra Mondal, Ida Momennejad - [[pdf]](https://arxiv.org/pdf/2310.00194)</summary>

**Abstract:** Large language models (LLMs) demonstrate impressive performance on a wide variety of tasks, but they often struggle with tasks that require multi-step reasoning or goal-directed planning. Both cognitive neuroscience and reinforcement learning (RL) have proposed a number of interacting functional components that together implement search and evaluation in multi-step decision making. These components include conflict monitoring, state prediction, state evaluation, task decomposition, and orchestration. To improve planning with LLMs, we propose an agentic architecture, the Modular Agentic Planner (MAP), in which planning is accomplished via the recurrent interaction of the specialized modules mentioned above, each implemented using an LLM. MAP improves planning through the interaction of specialized modules that break down a larger problem into multiple brief automated calls to the LLM. We evaluate MAP on three challenging planning tasks -- graph traversal, Tower of Hanoi, and the PlanBench benchmark -- as well as an NLP task requiring multi-step reasoning (strategyQA). We find that MAP yields significant improvements over both standard LLM methods (zero-shot prompting, in-context learning) and competitive baselines (chain-of-thought, multi-agent debate, and tree-of-thought), can be effectively combined with smaller and more cost-efficient LLMs (Llama3-70B), and displays superior transfer across tasks. These results suggest the benefit of a modular and multi-agent approach to planning with LLMs.

**arXiv ID:** 2310.00194
</details>

<details>
<summary><strong>Reinforcing Competitive Multi-Agents for Playing 'So Long Sucker'</strong> - Medant Sharan, Chandranath Adak - [[pdf]](https://arxiv.org/pdf/2411.11057)</summary>

**Abstract:** This paper investigates the strategy game So Long Sucker (SLS) as a novel benchmark for multi-agent reinforcement learning (MARL). Unlike traditional board or video game testbeds, SLS is distinguished by its coalition formation, strategic deception, and dynamic elimination rules, making it a uniquely challenging environment for autonomous agents. We introduce the first publicly available computational framework for SLS, complete with a graphical user interface and benchmarking support for reinforcement learning algorithms. Using classical deep reinforcement learning methods (e.g., DQN, DDQN, and Dueling DQN), we train self-playing agents to learn the rules and basic strategies of SLS. Experimental results demonstrate that, although these agents achieve roughly half of the maximum attainable reward and consistently outperform random baselines, they require long training horizons (~2000 games) and still commit occasional illegal moves, highlighting both the promise and limitations of classical reinforcement learning. Our findings establish SLS as a negotiation-aware benchmark for MARL, opening avenues for future research that integrates game-theoretic reasoning, coalition-aware strategies, and advanced reinforcement learning architectures to better capture the social and adversarial dynamics of complex multi-agent games.

**arXiv ID:** 2411.11057
</details>

<details>
<summary><strong>GUARDIAN: Safeguarding LLM Multi-Agent Collaborations with Temporal Graph Modeling</strong> - Jialong Zhou, Lichao Wang, Xiao Yang - [[pdf]](https://arxiv.org/pdf/2505.19234)</summary>

**Abstract:** The emergence of large language models (LLMs) enables the development of intelligent agents capable of engaging in complex and multi-turn dialogues. However, multi-agent collaboration faces critical safety challenges, such as hallucination amplification and error injection and propagation. This paper presents GUARDIAN, a unified method for detecting and mitigating multiple safety concerns in GUARDing Intelligent Agent collaboratioNs. By modeling the multi-agent collaboration process as a discrete-time temporal attributed graph, GUARDIAN explicitly captures the propagation dynamics of hallucinations and errors. The unsupervised encoder-decoder architecture incorporating an incremental training paradigm learns to reconstruct node attributes and graph structures from latent embeddings, enabling the identification of anomalous nodes and edges with unparalleled precision. Moreover, we introduce a graph abstraction mechanism based on the Information Bottleneck Theory, which compresses temporal interaction graphs while preserving essential patterns. Extensive experiments demonstrate GUARDIAN's effectiveness in safeguarding LLM multi-agent collaborations against diverse safety vulnerabilities, achieving state-of-the-art accuracy with efficient resource utilization. The code is available at this https URL

**arXiv ID:** 2505.19234
</details>

<details>
<summary><strong>Do LLM Agents Have Regret? A Case Study in Online Learning and Games</strong> - Chanwoo Park, Xiangyu Liu, Asuman Ozdaglar, Kaiqing Zhang - [[pdf]](https://arxiv.org/pdf/2403.16843)</summary>

**Abstract:** Large language models (LLMs) have been increasingly employed for (interactive) decision-making, via the development of LLM-based autonomous agents. Despite their emerging successes, the performance of LLM agents in decision-making has not been fully investigated through quantitative metrics, especially in the multi-agent setting when they interact with each other, a typical scenario in real-world LLM-agent applications. To better understand the limits of LLM agents in these interactive environments, we propose to study their interactions in benchmark decision-making settings in online learning and game theory, through the performance metric of \emph{regret}. We first empirically study the {no-regret} behaviors of LLMs in canonical (non-stationary) online learning problems, as well as the emergence of equilibria when LLM agents interact through playing repeated games. We then provide some theoretical insights into the no-regret behaviors of LLM agents, under certain assumptions on the supervised pre-training and the rationality model of human decision-makers who generate the data. Notably, we also identify (simple) cases where advanced LLMs such as GPT-4 fail to be no-regret. To promote the no-regret behaviors, we propose a novel \emph{unsupervised} training loss of \emph{regret-loss}, which, in contrast to the supervised pre-training loss, does not require the labels of (optimal) actions. We then establish the statistical guarantee of generalization bound for regret-loss minimization, followed by the optimization guarantee that minimizing such a loss may automatically lead to known no-regret learning algorithms. Our further experiments demonstrate the effectiveness of our regret-loss, especially in addressing the above ``regrettable'' cases.

**arXiv ID:** 2403.16843
</details>

<details>
<summary><strong>Decentralizing Multi-Agent Reinforcement Learning with Temporal Causal Information</strong> - Jan Corazza, Hadi Partovi Aria, Hyohun Kim, Daniel Neider, Zhe Xu - [[pdf]](https://arxiv.org/pdf/2506.07829)</summary>

**Abstract:** Reinforcement learning (RL) algorithms can find an optimal policy for a single agent to accomplish a particular task. However, many real-world problems require multiple agents to collaborate in order to achieve a common goal. For example, a robot executing a task in a warehouse may require the assistance of a drone to retrieve items from high shelves. In Decentralized Multi-Agent RL (DMARL), agents learn independently and then combine their policies at execution time, but often must satisfy constraints on compatibility of local policies to ensure that they can achieve the global task when combined. In this paper, we study how providing high-level symbolic knowledge to agents can help address unique challenges of this setting, such as privacy constraints, communication limitations, and performance concerns. In particular, we extend the formal tools used to check the compatibility of local policies with the team task, making decentralized training with theoretical guarantees usable in more scenarios. Furthermore, we empirically demonstrate that symbolic knowledge about the temporal evolution of events in the environment can significantly expedite the learning process in DMARL.

**arXiv ID:** 2506.07829
</details>

<details>
<summary><strong>Can an Individual Manipulate the Collective Decisions of Multi-Agents?</strong> - Fengyuan Liu, Rui Zhao, Shuo Chen, Guohao Li, Philip Torr, Lei Han, Jindong Gu - [[pdf]](https://arxiv.org/pdf/2509.16494)</summary>

**Abstract:** Individual Large Language Models (LLMs) have demonstrated significant capabilities across various domains, such as healthcare and law. Recent studies also show that coordinated multi-agent systems exhibit enhanced decision-making and reasoning abilities through collaboration. However, due to the vulnerabilities of individual LLMs and the difficulty of accessing all agents in a multi-agent system, a key question arises: If attackers only know one agent, could they still generate adversarial samples capable of misleading the collective decision? To explore this question, we formulate it as a game with incomplete information, where attackers know only one target agent and lack knowledge of the other agents in the system. With this formulation, we propose M-Spoiler, a framework that simulates agent interactions within a multi-agent system to generate adversarial samples. These samples are then used to manipulate the target agent in the target system, misleading the system's collaborative decision-making process. More specifically, M-Spoiler introduces a stubborn agent that actively aids in optimizing adversarial samples by simulating potential stubborn responses from agents in the target system. This enhances the effectiveness of the generated adversarial samples in misleading the system. Through extensive experiments across various tasks, our findings confirm the risks posed by the knowledge of an individual agent in multi-agent systems and demonstrate the effectiveness of our framework. We also explore several defense mechanisms, showing that our proposed attack framework remains more potent than baselines, underscoring the need for further research into defensive strategies.

**arXiv ID:** 2509.16494
</details>

<details>
<summary><strong>Foragax: An Agent-Based Modelling Framework Based on JAX</strong> - Siddharth Chaturvedi, Ahmed El-Gazzar, Marcel van Gerven - [[pdf]](https://arxiv.org/pdf/2409.06345)</summary>

**Abstract:** Foraging for resources is a ubiquitous activity conducted by living organisms in a shared environment to maintain their homeostasis. Modelling multi-agent foraging in-silico allows us to study both individual and collective emergent behaviour in a tractable manner. Agent-based modelling has proven to be effective in simulating such tasks, though scaling the simulations to accommodate large numbers of agents with complex dynamics remains challenging. In this work, we present Foragax, a general-purpose, scalable, hardware-accelerated, multi-agent foraging toolkit. Leveraging the JAX library, our toolkit can simulate thousands of agents foraging in a common environment, in an end-to-end vectorized and differentiable manner. The toolkit provides agent-based modelling tools to model various foraging tasks, including options to design custom spatial and temporal agent dynamics, control policies, sensor models, and boundary conditions. Further, the number of agents during such simulations can be increased or decreased based on custom rules. While applied to foraging, the toolkit can also be used to model and simulate a wide range of other multi-agent scenarios.

**arXiv ID:** 2409.06345
</details>

<details>
<summary><strong>OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning</strong> - Alexandre Oliveira, Katarina Dyreby, Francisco Caldas, Cláudia Soares - [[pdf]](https://arxiv.org/pdf/2504.04160)</summary>

**Abstract:** The increasing number of satellites and orbital debris has made space congestion a critical issue, threatening satellite safety and sustainability. Challenges such as collision avoidance, station-keeping, and orbital maneuvering require advanced techniques to handle dynamic uncertainties and multi-agent interactions. Reinforcement learning (RL) has shown promise in this domain, enabling adaptive, autonomous policies for space operations; however, many existing RL frameworks rely on custom-built environments developed from scratch, which often use simplified models and require significant time to implement and validate the orbital dynamics, limiting their ability to fully capture real-world complexities. To address this, we introduce OrbitZoo, a versatile multi-agent RL environment built on a high-fidelity industry standard library, that enables realistic data generation, supports scenarios like collision avoidance and cooperative maneuvers, and ensures robust and accurate orbital dynamics. The environment is validated against a real satellite constellation, Starlink, achieving a Mean Absolute Percentage Error (MAPE) of 0.16% compared to real-world data. This validation ensures reliability for generating high-fidelity simulations and enabling autonomous and independent satellite operations.

**arXiv ID:** 2504.04160
</details>

<details>
<summary><strong>MACTAS: Self-Attention-Based Module for Inter-Agent Communication in Multi-Agent Reinforcement Learning</strong> - Maciej Wojtala, Bogusz Stefańczyk, Dominik Bogucki, Łukasz Lepak, Jakub Strykowski, Paweł Wawrzyński - [[pdf]](https://arxiv.org/pdf/2508.13661)</summary>

**Abstract:** Communication is essential for the collective execution of complex tasks by human agents, motivating interest in communication mechanisms for multi-agent reinforcement learning (MARL). However, existing communication protocols in MARL are often complex and non-differentiable. In this work, we introduce a self-attention-based communication module that exchanges information between the agents in MARL. Our proposed approach is fully differentiable, allowing agents to learn to generate messages in a reward-driven manner. The module can be seamlessly integrated with any action-value function decomposition method and can be viewed as an extension of such decompositions. Notably, it includes a fixed number of trainable parameters, independent of the number of agents. Experimental results on the SMAC and SMACv2 benchmarks demonstrate the effectiveness of our approach, which achieves state-of-the-art performance on a number of maps.

**arXiv ID:** 2508.13661
</details>

<details>
<summary><strong>Automated Network Protocol Testing with LLM Agents</strong> - Yunze Wei, Kaiwen Wei, Shibo Du, Jianyu Wang, Zhangzhong Liu, Yawen Wang, Zhanyou Li, Congcong Miao, Xiaohui Xie, Yong Cui - [[pdf]](https://arxiv.org/pdf/2510.13248)</summary>

**Abstract:** Network protocol testing is fundamental for modern network infrastructure. However, traditional network protocol testing methods are labor-intensive and error-prone, requiring manual interpretation of specifications, test case design, and translation into executable artifacts, typically demanding one person-day of effort per test case. Existing model-based approaches provide partial automation but still involve substantial manual modeling and expert intervention, leading to high costs and limited adaptability to diverse and evolving protocols. In this paper, we propose a first-of-its-kind system called NeTestLLM that takes advantage of multi-agent Large Language Models (LLMs) for end-to-end automated network protocol testing. NeTestLLM employs hierarchical protocol understanding to capture complex specifications, iterative test case generation to improve coverage, a task-specific workflow for executable artifact generation, and runtime feedback analysis for debugging and refinement. NeTestLLM has been deployed in a production environment for several months, receiving positive feedback from domain experts. In experiments, NeTestLLM generated 4,632 test cases for OSPF, RIP, and BGP, covering 41 historical FRRouting bugs compared to 11 by current national standards. The process of generating executable artifacts also improves testing efficiency by a factor of 8.65x compared to manual methods. NeTestLLM provides the first practical LLM-powered solution for automated end-to-end testing of heterogeneous network protocols.

**arXiv ID:** 2510.13248
</details>

<details>
<summary><strong>RealEngine: Simulating Autonomous Driving in Realistic Context</strong> - Junzhe Jiang, Nan Song, Jingyu Li, Xiatian Zhu, Li Zhang - [[pdf]](https://arxiv.org/pdf/2505.16902)</summary>

**Abstract:** Driving simulation plays a crucial role in developing reliable driving agents by providing controlled, evaluative environments. To enable meaningful assessments, a high-quality driving simulator must satisfy several key requirements: multi-modal sensing capabilities (e.g., camera and LiDAR) with realistic scene rendering to minimize observational discrepancies; closed-loop evaluation to support free-form trajectory behaviors; highly diverse traffic scenarios for thorough evaluation; multi-agent cooperation to capture interaction dynamics; and high computational efficiency to ensure affordability and scalability. However, existing simulators and benchmarks fail to comprehensively meet these fundamental criteria. To bridge this gap, this paper introduces RealEngine, a novel driving simulation framework that holistically integrates 3D scene reconstruction and novel view synthesis techniques to achieve realistic and flexible closed-loop simulation in the driving context. By leveraging real-world multi-modal sensor data, RealEngine reconstructs background scenes and foreground traffic participants separately, allowing for highly diverse and realistic traffic scenarios through flexible scene composition. This synergistic fusion of scene reconstruction and view synthesis enables photorealistic rendering across multiple sensor modalities, ensuring both perceptual fidelity and geometric accuracy. Building upon this environment, RealEngine supports three essential driving simulation categories: non-reactive simulation, safety testing, and multi-agent interaction, collectively forming a reliable and comprehensive benchmark for evaluating the real-world performance of driving agents.

**arXiv ID:** 2505.16902
</details>

</details>

<details open>
<summary><h2>Other Agent Research (6 papers)</h2></summary>

<details>
<summary><strong>Towards Human-Centric Intelligent Treatment Planning for Radiation Therapy</strong> - Adnan Jafar, Xun Jia - [[pdf]](https://arxiv.org/pdf/2510.13062)</summary>

**Abstract:** Current radiation therapy treatment planning is limited by suboptimal plan quality, inefficiency, and high costs. This perspective paper explores the complexity of treatment planning and introduces Human-Centric Intelligent Treatment Planning (HCITP), an AI-driven framework under human oversight, which integrates clinical guidelines, automates plan generation, and enables direct interactions with operators. We expect that HCITP will enhance efficiency, potentially reducing planning time to minutes, and will deliver personalized, high-quality plans. Challenges and potential solutions are discussed.

**arXiv ID:** 2510.13062
</details>

<details>
<summary><strong>MotionBeat: Motion-Aligned Music Representation via Embodied Contrastive Learning and Bar-Equivariant Contact-Aware Encoding</strong> - Xuanchen Wang, Heng Wang, Weidong Cai - [[pdf]](https://arxiv.org/pdf/2510.13244)</summary>

**Abstract:** Music is both an auditory and an embodied phenomenon, closely linked to human motion and naturally expressed through dance. However, most existing audio representations neglect this embodied dimension, limiting their ability to capture rhythmic and structural cues that drive movement. We propose MotionBeat, a framework for motion-aligned music representation learning. MotionBeat is trained with two newly proposed objectives: the Embodied Contrastive Loss (ECL), an enhanced InfoNCE formulation with tempo-aware and beat-jitter negatives to achieve fine-grained rhythmic discrimination, and the Structural Rhythm Alignment Loss (SRAL), which ensures rhythm consistency by aligning music accents with corresponding motion events. Architecturally, MotionBeat introduces bar-equivariant phase rotations to capture cyclic rhythmic patterns and contact-guided attention to emphasize motion events synchronized with musical accents. Experiments show that MotionBeat outperforms state-of-the-art audio encoders in music-to-dance generation and transfers effectively to beat tracking, music tagging, genre and instrument classification, emotion recognition, and audio-visual retrieval. Our project demo page: this https URL.

**arXiv ID:** 2510.13244
</details>

<details>
<summary><strong>In-Browser LLM-Guided Fuzzing for Real-Time Prompt Injection Testing in Agentic AI Browsers</strong> - Avihay Cohen - [[pdf]](https://arxiv.org/pdf/2510.13543)</summary>

**Abstract:** Large Language Model (LLM) based agents integrated into web browsers (often called agentic AI browsers) offer powerful automation of web tasks. However, they are vulnerable to indirect prompt injection attacks, where malicious instructions hidden in a webpage deceive the agent into unwanted actions. These attacks can bypass traditional web security boundaries, as the AI agent operates with the user privileges across sites. In this paper, we present a novel fuzzing framework that runs entirely in the browser and is guided by an LLM to automatically discover such prompt injection vulnerabilities in real time.

**arXiv ID:** 2510.13543
</details>

<details>
<summary><strong>Equilibria in routing games with connected autonomous vehicles will not be strong, as exclusive clubs may form</strong> - Rafał Kucharski, Anastasia Psarou, Natello Descormier - [[pdf]](https://arxiv.org/pdf/2510.12862)</summary>

**Abstract:** User Equilibrium is the standard representation of the so-called routing game in which drivers adjust their route choices to arrive at their destinations as fast as possible. Asking whether this Equilibrium is strong or not was meaningless for human drivers who did not form coalitions due to technical and behavioral constraints. This is no longer the case for connected autonomous vehicles (CAVs), which will be able to communicate and collaborate to jointly form routing coalitions.
We demonstrate this for the first time on a carefully designed toy-network example, where a `club` of three autonomous vehicles jointly decides to deviate from the user equilibrium and benefit (arrive faster). The formation of such a club has negative consequences for other users, who are not invited to join it and now travel longer, and for the system, making it suboptimal and disequilibrated, which triggers adaptation dynamics.
This discovery has profound implications for the future of our cities. We demonstrate that, if not prevented, CAV operators may intentionally disequilibrate traffic systems from their classic Nash equilibria, benefiting their own users and imposing costs on others. These findings suggest the possible emergence of an exclusive CAV elite, from which human-driven vehicles and non-coalition members may be excluded, potentially leading to systematically longer travel times for those outside the coalition, which would be harmful for the equity of public road networks.

**arXiv ID:** 2510.12862
</details>

<details>
<summary><strong>UNCAP: Uncertainty-Guided Planning Using Natural Language Communication for Cooperative Autonomous Vehicles</strong> - Neel P. Bhatt, Po-han Li, Kushagra Gupta, Rohan Siva, Daniel Milan, Alexander T. Hogue, Sandeep P. Chinchali, David Fridovich-Keil, Zhangyang Wang, Ufuk Topcu - [[pdf]](https://arxiv.org/pdf/2510.12992)</summary>

**Abstract:** Safe large-scale coordination of multiple cooperative connected autonomous vehicles (CAVs) hinges on communication that is both efficient and interpretable. Existing approaches either rely on transmitting high-bandwidth raw sensor data streams or neglect perception and planning uncertainties inherent in shared data, resulting in systems that are neither scalable nor safe. To address these limitations, we propose Uncertainty-Guided Natural Language Cooperative Autonomous Planning (UNCAP), a vision-language model-based planning approach that enables CAVs to communicate via lightweight natural language messages while explicitly accounting for perception uncertainty in decision-making. UNCAP features a two-stage communication protocol: (i) an ego CAV first identifies the subset of vehicles most relevant for information exchange, and (ii) the selected CAVs then transmit messages that quantitatively express their perception uncertainty. By selectively fusing messages that maximize mutual information, this strategy allows the ego vehicle to integrate only the most relevant signals into its decision-making, improving both the scalability and reliability of cooperative planning. Experiments across diverse driving scenarios show a 63% reduction in communication bandwidth with a 31% increase in driving safety score, a 61% reduction in decision uncertainty, and a four-fold increase in collision distance margin during near-miss events. Project website: this https URL

**arXiv ID:** 2510.12992
</details>

<details>
<summary><strong>Intelligent4DSE: Optimizing High-Level Synthesis Design Space Exploration with Graph Neural Networks and Large Language Models</strong> - Lei Xu, Shanshan Wang, Emmanuel Casseau, Chenglong Xiao - [[pdf]](https://arxiv.org/pdf/2504.19649)</summary>

**Abstract:** High-Level Synthesis (HLS) Design Space Exploration (DSE) is essential for generating hardware designs that balance performance, power, and area (PPA). To optimize this process, existing works often employs message-passing neural networks (MPNNs) to predict quality of results (QoR). These predictors serve as evaluators in the DSE process, effectively bypassing the time-consuming estimations traditionally required by HLS tools. However, existing models based on MPNNs struggle with over-smoothing and limited expressiveness. Additionally, while meta-heuristic algorithms are widely used in DSE, they typically require extensive domain-specific knowledge to design operators and time-consuming tuning. To address these limitations, we propose ECoGNNs-LLMMHs, a framework that integrates graph neural networks with task-adaptive message passing and large language model-enhanced meta-heuristic algorithms. Compared with state-of-the-art works, ECoGNN exhibits lower prediction error in the post-HLS prediction task, with the error reduced by 57.27\%. For post-implementation prediction tasks, ECoGNN demonstrates the lowest prediction errors, with average reductions of 17.6\% for flip-flop (FF) usage, 33.7\% for critical path (CP) delay, 26.3\% for power consumption, 38.3\% for digital signal processor (DSP) utilization, and 40.8\% for BRAM usage. LLMMH variants can generate superior Pareto fronts compared to meta-heuristic algorithms in terms of average distance from the reference set (ADRS) with average improvements of 87.47\%, respectively. Compared with the SOTA DSE approaches GNN-DSE and IRONMAN-PRO, LLMMH can reduce the ADRS by 68.17\% and 63.07\% respectively.

**arXiv ID:** 2504.19649
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (28 papers)</h2></summary>

<details>
<summary><strong>DeepPlanner: Scaling Planning Capability for Deep Research Agents via Advantage Shaping</strong> - Wei Fan, Wenlin Yao, Zheng Li, Feng Yao, Xin Liu, Liang Qiu, Qingyu Yin, Yangqiu Song, Bing Yin - [[pdf]](https://arxiv.org/pdf/2510.12979)</summary>

**Abstract:** Large language models (LLMs) augmented with multi-step reasoning and action generation abilities have shown promise in leveraging external tools to tackle complex tasks that require long-horizon planning. However, existing approaches either rely on implicit planning in the reasoning stage or introduce explicit planners without systematically addressing how to optimize the planning stage. As evidence, we observe that under vanilla reinforcement learning (RL), planning tokens exhibit significantly higher entropy than other action tokens, revealing uncertain decision points that remain under-optimized. To address this, we propose DeepPlanner, an end-to-end RL framework that effectively enhances the planning capabilities of deep research agents. Our approach shapes token-level advantage with an entropy-based term to allocate larger updates to high entropy tokens, and selectively upweights sample-level advantages for planning-intensive rollouts. Extensive experiments across seven deep research benchmarks demonstrate that DeepPlanner improves planning quality and achieves state-of-the-art results under a substantially lower training budget.

**arXiv ID:** 2510.12979
</details>

<details>
<summary><strong>EvoTest: Evolutionary Test-Time Learning for Self-Improving Agentic Systems</strong> - Yufei He, Juncheng Liu, Yue Liu, Yibo Li, Tri Cao, Zhiyuan Hu, Xinxing Xu, Bryan Hooi - [[pdf]](https://arxiv.org/pdf/2510.13220)</summary>

**Abstract:** A fundamental limitation of current AI agents is their inability to learn complex skills on the fly at test time, often behaving like "clever but clueless interns" in novel environments. This severely limits their practical utility. To systematically measure and drive progress on this challenge, we first introduce the Jericho Test-Time Learning (J-TTL) benchmark. J-TTL is a new evaluation setup where an agent must play the same game for several consecutive episodes, attempting to improve its performance from one episode to the next. On J-TTL, we find that existing adaptation methods like reflection, memory, or reinforcement learning struggle. To address the challenges posed by our benchmark, we present EvoTest, an evolutionary test-time learning framework that improves an agent without any fine-tuning or gradients-by evolving the entire agentic system after every episode. EvoTest has two roles: the Actor Agent, which plays the game, and the Evolver Agent, which analyzes the episode transcript to propose a revised configuration for the next run. This configuration rewrites the prompt, updates memory by logging effective state-action choices, tunes hyperparameters, and learns the tool-use routines. On our J-TTL benchmark, EvoTest consistently increases performance, outperforming not only reflection and memory-only baselines but also more complex online fine-tuning methods. Notably, our method is the only one capable of winning two games (Detective and Library), while all baselines fail to win any.

**arXiv ID:** 2510.13220
</details>

<details>
<summary><strong>MTSQL-R1: Towards Long-Horizon Multi-Turn Text-to-SQL via Agentic Training</strong> - Taicheng Guo, Hai Wang, ChaoChun Liu, Mohsen Golalikhani, Xin Chen, Xiangliang Zhang, Chandan K. Reddy - [[pdf]](https://arxiv.org/pdf/2510.12831)</summary>

**Abstract:** Multi-turn Text-to-SQL aims to translate a user's conversational utterances into executable SQL while preserving dialogue coherence and grounding to the target schema. However, most existing systems only regard this task as a simple text translation task and follow a short-horizon paradigm, generating a query per turn without execution, explicit verification, and refinement, which leads to non-executable or incoherent outputs. We present MTSQL-R1, an agentic training framework for long-horizon multi-turn Text-to-SQL. We cast the task as a Markov Decision Process (MDP) in which an agent interacts with (i) a database for execution feedback and (ii) a persistent dialogue memory for coherence verification, performing an iterative propose to execute -> verify -> refine cycle until all checks pass. Experiments on COSQL and SPARC demonstrate that MTSQL-R1 consistently outperforms strong baselines, highlighting the importance of environment-driven verification and memory-guided refinement for conversational semantic parsing. Full recipes (including code, trained models, logs, reasoning trajectories, etc.) will be released after the internal review to contribute to community research.

**arXiv ID:** 2510.12831
</details>

<details>
<summary><strong>A\textsuperscript{2}FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning</strong> - Qianben Chen, Jingyi Cao, Jiayu Zhang, Tianrui Qin, Xiaowan Li, King Zhu, Dingfeng Shi, He Zhu, Minghao Liu, Xiaobo Liang, Ge Zhang, Jian Yang, Yuchen Eleanor Jiang, Wangchunshu Zhou - [[pdf]](https://arxiv.org/pdf/2510.12838)</summary>

**Abstract:** Large language models split into two families: reasoning-centric LLMs, which strengthen internal chain-of-thought reasoning but cannot invoke external tools, and agentic LLMs, which learn to interact with environments and leverage tools but often lag in deep reasoning. This divide arises from fundamentally different training objectives, leading to mismatched strengths and inefficiency on simple queries, where both families tend to overthink or over-call tools. In this work, we present Adaptive Agent Foundation Model (A\textsuperscript{2}FM), a unified framework that follows a route-then-align principle: the model first learns task-aware routing and then aligns mode-specific trajectories under a shared backbone. To address the inefficiency gap, we introduce a third mode-instant-that handles simple queries directly, preventing unnecessary reasoning or tool calls while complementing the agentic and reasoning modes. To jointly enhance accuracy and efficiency, we propose Adaptive Policy Optimization (APO), which enforces adaptive sampling across modes and applies a cost-regularized reward. On the 32B scale, A\textsuperscript{2}FM achieves 13.4\% on BrowseComp, 70.4\% on AIME25, and 16.7\% on HLE, setting new SOTA among comparable models and performing competitively with frontier LLMs across agentic, reasoning, and general benchmarks. Notably, the adaptive execution achieves a cost of pass of only \$0.00487 per correct answer-cutting cost by 45.2\% relative to reasoning and 33.5\% relative to agentic, thus delivering substantially higher cost efficiency while maintaining comparable accuracy.

**arXiv ID:** 2510.12838
</details>

<details>
<summary><strong>DriveCritic: Towards Context-Aware, Human-Aligned Evaluation for Autonomous Driving with Vision-Language Models</strong> - Jingyu Song, Zhenxin Li, Shiyi Lan, Xinglong Sun, Nadine Chang, Maying Shen, Joshua Chen, Katherine A. Skinner, Jose M. Alvarez - [[pdf]](https://arxiv.org/pdf/2510.13108)</summary>

**Abstract:** Benchmarking autonomous driving planners to align with human judgment remains a critical challenge, as state-of-the-art metrics like the Extended Predictive Driver Model Score (EPDMS) lack context awareness in nuanced scenarios. To address this, we introduce DriveCritic, a novel framework featuring two key contributions: the DriveCritic dataset, a curated collection of challenging scenarios where context is critical for correct judgment and annotated with pairwise human preferences, and the DriveCritic model, a Vision-Language Model (VLM) based evaluator. Fine-tuned using a two-stage supervised and reinforcement learning pipeline, the DriveCritic model learns to adjudicate between trajectory pairs by integrating visual and symbolic context. Experiments show DriveCritic significantly outperforms existing metrics and baselines in matching human preferences and demonstrates strong context awareness. Overall, our work provides a more reliable, human-aligned foundation to evaluating autonomous driving systems.

**arXiv ID:** 2510.13108
</details>

<details>
<summary><strong>Adversarial Fine-tuning in Offline-to-Online Reinforcement Learning for Robust Robot Control</strong> - Shingo Ayabe, Hiroshi Kera, Kazuhiko Kawamoto - [[pdf]](https://arxiv.org/pdf/2510.13358)</summary>

**Abstract:** Offline reinforcement learning enables sample-efficient policy acquisition without risky online interaction, yet policies trained on static datasets remain brittle under action-space perturbations such as actuator faults. This study introduces an offline-to-online framework that trains policies on clean data and then performs adversarial fine-tuning, where perturbations are injected into executed actions to induce compensatory behavior and improve resilience. A performance-aware curriculum further adjusts the perturbation probability during training via an exponential-moving-average signal, balancing robustness and stability throughout the learning process. Experiments on continuous-control locomotion tasks demonstrate that the proposed method consistently improves robustness over offline-only baselines and converges faster than training from scratch. Matching the fine-tuning and evaluation conditions yields the strongest robustness to action-space perturbations, while the adaptive curriculum strategy mitigates the degradation of nominal performance observed with the linear curriculum strategy. Overall, the results show that adversarial fine-tuning enables adaptive and robust control under uncertain environments, bridging the gap between offline efficiency and online adaptability.

**arXiv ID:** 2510.13358
</details>

<details>
<summary><strong>A New Perspective on Transformers in Online Reinforcement Learning for Continuous Control</strong> - Nikita Kachaev, Daniil Zelezetsky, Egor Cherepanov, Alexey K. Kovelev, Aleksandr I. Panov - [[pdf]](https://arxiv.org/pdf/2510.13367)</summary>

**Abstract:** Despite their effectiveness and popularity in offline or model-based reinforcement learning (RL), transformers remain underexplored in online model-free RL due to their sensitivity to training setups and model design decisions such as how to structure the policy and value networks, share components, or handle temporal information. In this paper, we show that transformers can be strong baselines for continuous control in online model-free RL. We investigate key design questions: how to condition inputs, share components between actor and critic, and slice sequential data for training. Our experiments reveal stable architectural and training strategies enabling competitive performance across fully and partially observable tasks, and in both vector- and image-based settings. These findings offer practical guidance for applying transformers in online RL.

**arXiv ID:** 2510.13367
</details>

<details>
<summary><strong>Simplicial Embeddings Improve Sample Efficiency in Actor-Critic Agents</strong> - Johan Obando-Ceron, Walter Mayor, Samuel Lavoie, Scott Fujimoto, Aaron Courville, Pablo Samuel Castro - [[pdf]](https://arxiv.org/pdf/2510.13704)</summary>

**Abstract:** Recent works have proposed accelerating the wall-clock training time of actor-critic methods via the use of large-scale environment parallelization; unfortunately, these can sometimes still require large number of environment interactions to achieve a desired level of performance. Noting that well-structured representations can improve the generalization and sample efficiency of deep reinforcement learning (RL) agents, we propose the use of simplicial embeddings: lightweight representation layers that constrain embeddings to simplicial structures. This geometric inductive bias results in sparse and discrete features that stabilize critic bootstrapping and strengthen policy gradients. When applied to FastTD3, FastSAC, and PPO, simplicial embeddings consistently improve sample efficiency and final performance across a variety of continuous- and discrete-control environments, without any loss in runtime speed.

**arXiv ID:** 2510.13704
</details>

<details>
<summary><strong>The Art of Scaling Reinforcement Learning Compute for LLMs</strong> - Devvrit Khatri, Lovish Madaan, Rishabh Tiwari, Rachit Bansal, Sai Surya Duvvuri, Manzil Zaheer, Inderjit S. Dhillon, David Brandfonbrener, Rishabh Agarwal - [[pdf]](https://arxiv.org/pdf/2510.13786)</summary>

**Abstract:** Reinforcement learning (RL) has become central to training large language models (LLMs), yet the field lacks predictive scaling methodologies comparable to those established for pre-training. Despite rapidly rising compute budgets, there is no principled understanding of how to evaluate algorithmic improvements for scaling RL compute. We present the first large-scale systematic study, amounting to more than 400,000 GPU-hours, that defines a principled framework for analyzing and predicting RL scaling in LLMs. We fit sigmoidal compute-performance curves for RL training and ablate a wide range of common design choices to analyze their effects on asymptotic performance and compute efficiency. We observe: (1) Not all recipes yield similar asymptotic performance, (2) Details such as loss aggregation, normalization, curriculum, and off-policy algorithm primarily modulate compute efficiency without materially shifting the asymptote, and (3) Stable, scalable recipes follow predictable scaling trajectories, enabling extrapolation from smaller-scale runs. Combining these insights, we propose a best-practice recipe, ScaleRL, and demonstrate its effectiveness by successfully scaling and predicting validation performance on a single RL run scaled up to 100,000 GPU-hours. Our work provides both a scientific framework for analyzing scaling in RL and a practical recipe that brings RL training closer to the predictability long achieved in pre-training.

**arXiv ID:** 2510.13786
</details>

<details>
<summary><strong>Provably Invincible Adversarial Attacks on Reinforcement Learning Systems: A Rate-Distortion Information-Theoretic Approach</strong> - Ziqing Lu, Lifeng Lai, Weiyu Xu - [[pdf]](https://arxiv.org/pdf/2510.13792)</summary>

**Abstract:** Reinforcement learning (RL) for the Markov Decision Process (MDP) has emerged in many security-related applications, such as autonomous driving, financial decisions, and drone/robot algorithms. In order to improve the robustness/defense of RL systems against adversaries, studying various adversarial attacks on RL systems is very important. Most previous work considered deterministic adversarial attack strategies in MDP, which the recipient (victim) agent can defeat by reversing the deterministic attacks. In this paper, we propose a provably ``invincible'' or ``uncounterable'' type of adversarial attack on RL. The attackers apply a rate-distortion information-theoretic approach to randomly change agents' observations of the transition kernel (or other properties) so that the agent gains zero or very limited information about the ground-truth kernel (or other properties) during the training. We derive an information-theoretic lower bound on the recipient agent's reward regret and show the impact of rate-distortion attacks on state-of-the-art model-based and model-free algorithms. We also extend this notion of an information-theoretic approach to other types of adversarial attack, such as state observation attacks.

**arXiv ID:** 2510.13792
</details>

<details>
<summary><strong>LLM/Agent-as-Data-Analyst: A Survey</strong> - Zirui Tang, Weizheng Wang, Zihang Zhou, Yang Jiao, Bangrui Xu, Boyu Niu, Xuanhe Zhou, Guoliang Li, Yeye He, Wei Zhou, Yitong Song, Cheng Tan, Xue Yang, Bin Wang, Conghui He, Xiaoyang Wang, Fan Wu - [[pdf]](https://arxiv.org/pdf/2509.23988)</summary>

**Abstract:** Large language model (LLM) and agent techniques for data analysis (a.k.a LLM/Agent-as-Data-Analyst) have demonstrated substantial impact in both academica and industry. In comparison with traditional rule or small-model based approaches, (agentic) LLMs enable complex data understanding, natural language interfaces, semantic analysis functions, and autonomous pipeline orchestration. The technical evolution further distills five key design goals for intelligent data analysis agents, namely semantic-aware design, modality-hybrid integration, autonomous pipelines, tool-augmented workflows, and support for open-world tasks. From a modality perspective, we review LLM-based techniques for (i) structured data (e.g., table question answering for relational data and NL2GQL for graph data), (ii) semi-structured data (e.g., markup languages understanding and semi-structured table modeling), (iii) unstructured data (e.g., chart understanding, document understanding, programming languages vulnerable detection), and (iv) heterogeneous data (e.g., data retrieval and modality alignment for data lakes). Finally, we outline the remaining challenges and propose several insights and practical directions for advancing LLM/Agent-powered data analysis.

**arXiv ID:** 2509.23988
</details>

<details>
<summary><strong>When to Trust Your Simulator: Dynamics-Aware Hybrid Offline-and-Online Reinforcement Learning</strong> - Haoyi Niu, Shubham Sharma, Yiwen Qiu, Ming Li, Guyue Zhou, Jianming Hu, Xianyuan Zhan - [[pdf]](https://arxiv.org/pdf/2206.13464)</summary>

**Abstract:** Learning effective reinforcement learning (RL) policies to solve real-world complex tasks can be quite challenging without a high-fidelity simulation environment. In most cases, we are only given imperfect simulators with simplified dynamics, which inevitably lead to severe sim-to-real gaps in RL policy learning. The recently emerged field of offline RL provides another possibility to learn policies directly from pre-collected historical data. However, to achieve reasonable performance, existing offline RL algorithms need impractically large offline data with sufficient state-action space coverage for training. This brings up a new question: is it possible to combine learning from limited real data in offline RL and unrestricted exploration through imperfect simulators in online RL to address the drawbacks of both approaches? In this study, we propose the Dynamics-Aware Hybrid Offline-and-Online Reinforcement Learning (H2O) framework to provide an affirmative answer to this question. H2O introduces a dynamics-aware policy evaluation scheme, which adaptively penalizes the Q function learning on simulated state-action pairs with large dynamics gaps, while also simultaneously allowing learning from a fixed real-world dataset. Through extensive simulation and real-world tasks, as well as theoretical analysis, we demonstrate the superior performance of H2O against other cross-domain online and offline RL algorithms. H2O provides a brand new hybrid offline-and-online RL paradigm, which can potentially shed light on future RL algorithm design for solving practical real-world tasks.

**arXiv ID:** 2206.13464
</details>

<details>
<summary><strong>Hi-Drive: Hierarchical POMDP Planning for Safe Autonomous Driving in Diverse Urban Environments</strong> - Xuanjin Jin, Chendong Zeng, Shengfa Zhu, Chunxiao Liu, Panpan Cai - [[pdf]](https://arxiv.org/pdf/2409.18411)</summary>

**Abstract:** Uncertainties in dynamic road environments pose significant challenges for behavior and trajectory planning in autonomous driving. This paper introduces Hi-Drive, a hierarchical planning algorithm addressing uncertainties at both behavior and trajectory levels using a hierarchical Partially Observable Markov Decision Process (POMDP) formulation. Hi-Drive employs driver models to represent uncertain behavioral intentions of other vehicles and uses their parameters to infer hidden driving styles. By treating driver models as high-level decision-making actions, our approach effectively manages the exponential complexity inherent in POMDPs. To further enhance safety and robustness, Hi-Drive integrates a trajectory optimization based on importance sampling, refining trajectories using a comprehensive analysis of critical agents. Evaluations on real-world urban driving datasets demonstrate that Hi-Drive significantly outperforms state-of-the-art planning-based and learning-based methods across diverse urban driving situations in real-world benchmarks.

**arXiv ID:** 2409.18411
</details>

<details>
<summary><strong>Reinforcement Learning for Out-of-Distribution Reasoning in LLMs: An Empirical Study on Diagnosis-Related Group Coding</strong> - Hanyin Wang, Zhenbang Wu, Gururaj Kolar, Hariprasad Korsapati, Brian Bartlett, Bryan Hull, Jimeng Sun - [[pdf]](https://arxiv.org/pdf/2505.21908)</summary>

**Abstract:** Diagnosis-Related Group (DRG) codes are essential for hospital reimbursement and operations but require labor-intensive assignment. Large Language Models (LLMs) struggle with DRG coding due to the out-of-distribution (OOD) nature of the task: pretraining corpora rarely contain private clinical or billing data. We introduce DRG-Sapphire, which uses large-scale reinforcement learning (RL) for automated DRG coding from clinical notes. Built on Qwen2.5-7B and trained with Group Relative Policy Optimization (GRPO) using rule-based rewards, DRG-Sapphire introduces a series of RL enhancements to address domain-specific challenges not seen in previous mathematical tasks. Our model achieves state-of-the-art accuracy on the MIMIC-IV benchmark and generates physician-validated reasoning for DRG assignments, significantly enhancing explainability. Our study further sheds light on broader challenges of applying RL to knowledge-intensive, OOD tasks. We observe that RL performance scales approximately linearly with the logarithm of the number of supervised fine-tuning (SFT) examples, suggesting that RL effectiveness is fundamentally constrained by the domain knowledge encoded in the base model. For OOD tasks like DRG coding, strong RL performance requires sufficient knowledge infusion prior to RL. Consequently, scaling SFT may be more effective and computationally efficient than scaling RL alone for such tasks.

**arXiv ID:** 2505.21908
</details>

<details>
<summary><strong>DynaSearcher: Dynamic Knowledge Graph Augmented Search Agent via Multi-Reward Reinforcement Learning</strong> - Chuzhan Hao, Wenfeng Feng, Yuewei Zhang, Hao Wang - [[pdf]](https://arxiv.org/pdf/2507.17365)</summary>

**Abstract:** Multi-step agentic retrieval systems based on large language models (LLMs) have demonstrated remarkable performance in complex information search tasks. However, these systems still face significant challenges in practical applications, particularly in generating factually inconsistent intermediate queries and inefficient search trajectories, which can lead to reasoning deviations or redundant computations. To address these issues, we propose DynaSearcher, an innovative search agent enhanced by dynamic knowledge graphs and multi-reward reinforcement learning (RL). Specifically, our system leverages knowledge graphs as external structured knowledge to guide the search process by explicitly modeling entity relationships, thereby ensuring factual consistency in intermediate queries and mitigating biases from irrelevant information. Furthermore, we employ a multi-reward RL framework for fine-grained control over training objectives such as retrieval accuracy, efficiency, and response quality. This framework promotes the generation of high-quality intermediate queries and comprehensive final answers, while discouraging unnecessary exploration and minimizing information omissions or redundancy. Experimental results demonstrate that our approach achieves state-of-the-art answer accuracy on six multi-hop question answering datasets, matching frontier LLMs while using only small-scale models and limited computational resources. Furthermore, our approach demonstrates strong generalization and robustness across diverse retrieval environments and larger-scale models, highlighting its broad applicability.

**arXiv ID:** 2507.17365
</details>

<details>
<summary><strong>h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning</strong> - Sumeet Ramesh Motwani, Alesia Ivanova, Ziyang Cai, Philip Torr, Riashat Islam, Shital Shah, Christian Schroeder de Witt, Charles London - [[pdf]](https://arxiv.org/pdf/2510.07312)</summary>

**Abstract:** Large language models excel at short-horizon reasoning tasks, but performance drops as reasoning horizon lengths increase. Existing approaches to combat this rely on inference-time scaffolding or costly step-level supervision, neither of which scales easily. In this work, we introduce a scalable method to bootstrap long-horizon reasoning capabilities using only existing, abundant short-horizon data. Our approach synthetically composes simple problems into complex, multi-step dependency chains of arbitrary length. We train models on this data using outcome-only rewards under a curriculum that automatically increases in complexity, allowing RL training to be scaled much further without saturating. Empirically, our method generalizes remarkably well: curriculum training on composed 6th-grade level math problems (GSM8K) boosts accuracy on longer, competition-level benchmarks (GSM-Symbolic, MATH-500, AIME) by up to 2.06x. It also transfers significantly to diverse out-of-distribution ReasoningGym domains and long-context benchmarks, indicating broader generalization. Importantly, our long-horizon improvements are significantly higher than baselines even at high pass@k, showing that models can learn new reasoning paths under RL. Theoretically, we show that curriculum RL with outcome rewards achieves an exponential improvement in sample complexity over full-horizon training, providing training signal comparable to dense supervision. h1 therefore introduces an efficient path towards scaling RL for long-horizon problems using only existing data.

**arXiv ID:** 2510.07312
</details>

<details>
<summary><strong>ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering</strong> - Simon Lupart, Mohammad Aliannejadi, Evangelos Kanoulas - [[pdf]](https://arxiv.org/pdf/2510.13312)</summary>

**Abstract:** We present ChatR1, a reasoning framework based on reinforcement learning (RL) for conversational question answering (CQA). Reasoning plays an important role in CQA, where user intent evolves across dialogue turns, and utterances are often underspecified, requiring contextual interpretation, query reformulation, and dynamic coordination between retrieval and generation. Unlike static `rewrite, retrieve, and generate' pipelines, ChatR1 interleaves search and reasoning across turns, enabling exploratory and adaptive behaviors learned through RL. To address the challenge of sparse and delayed rewards in RL, we propose an intent-aware reward that provides turn-level feedback by aligning retrieval and reasoning with evolving user goals. Our proposed ChatR1 demonstrates strong performance on both 3B and 7B model backbones, outperforming competitive models on five CQA datasets, measured by different metrics (F1, BERTScore, and LLM-as-judge). We include a diverse set of CQA datasets to cover topic shifts, evolving intents, mixed-initiative dialogues, and multi-document grounding, testing ChatR1's performance from various aspects. Ablation studies confirm the effectiveness of the intent-aware reward. Our analyses further reveal diverse reasoning trajectories and effective use of the search tool. ChatR1 also generalizes robustly across domains, demonstrating that RL-based reasoning enables more flexible and context-sensitive behavior than static CQA pipelines.

**arXiv ID:** 2510.13312
</details>

<details>
<summary><strong>RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments</strong> - Zeyi Liao, Jaylen Jones, Linxi Jiang, Yuting Ning, Eric Fosler-Lussier, Yu Su, Zhiqiang Lin, Huan Sun - [[pdf]](https://arxiv.org/pdf/2505.21936)</summary>

**Abstract:** Computer-use agents (CUAs) promise to automate complex tasks across operating systems (OS) and the web, but remain vulnerable to indirect prompt injection. Current evaluations of this threat either lack support realistic but controlled environments or ignore hybrid web-OS attack scenarios involving both interfaces. To address this, we propose RedTeamCUA, an adversarial testing framework featuring a novel hybrid sandbox that integrates a VM-based OS environment with Docker-based web platforms. Our sandbox supports key features tailored for red teaming, such as flexible adversarial scenario configuration, and a setting that decouples adversarial evaluation from navigational limitations of CUAs by initializing tests directly at the point of an adversarial injection. Using RedTeamCUA, we develop RTC-Bench, a comprehensive benchmark with 864 examples that investigate realistic, hybrid web-OS attack scenarios and fundamental security vulnerabilities. Benchmarking current frontier CUAs identifies significant vulnerabilities: Claude 3.7 Sonnet | CUA demonstrates an ASR of 42.9%, while Operator, the most secure CUA evaluated, still exhibits an ASR of 7.6%. Notably, CUAs often attempt to execute adversarial tasks with an Attempt Rate as high as 92.5%, although failing to complete them due to capability limitations. Nevertheless, we observe concerning high ASRs in realistic end-to-end settings, with the strongest-to-date Claude 4.5 Sonnet | CUA exhibiting the highest ASR of 60%, indicating that CUA threats can already result in tangible risks to users and computer systems. Overall, RedTeamCUA provides an essential framework for advancing realistic, controlled, and systematic analysis of CUA vulnerabilities, highlighting the urgent need for robust defenses to indirect prompt injection prior to real-world deployment.

**arXiv ID:** 2505.21936
</details>

<details>
<summary><strong>Rec-R1: Bridging Generative Large Language Models and User-Centric Recommendation Systems via Reinforcement Learning</strong> - Jiacheng Lin, Tian Wang, Kun Qian - [[pdf]](https://arxiv.org/pdf/2503.24289)</summary>

**Abstract:** We propose Rec-R1, a general reinforcement learning framework that bridges large language models (LLMs) with recommendation systems through closed-loop optimization. Unlike prompting and supervised fine-tuning (SFT), Rec-R1 directly optimizes LLM generation using feedback from a fixed black-box recommendation model, without relying on synthetic SFT data from proprietary models such as GPT-4o. This avoids the substantial cost and effort required for data distillation. To verify the effectiveness of Rec-R1, we evaluate it on two representative tasks: product search and sequential recommendation. Experimental results demonstrate that Rec-R1 not only consistently outperforms prompting- and SFT-based methods, but also achieves significant gains over strong discriminative baselines, even when used with simple retrievers such as BM25. Moreover, Rec-R1 preserves the general-purpose capabilities of the LLM, unlike SFT, which often impairs instruction-following and reasoning. These findings suggest Rec-R1 as a promising foundation for continual task-specific adaptation without catastrophic forgetting.

**arXiv ID:** 2503.24289
</details>

<details>
<summary><strong>CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning</strong> - Zhenpeng Su, Leiyu Pan, Minxuan Lv, Yuntao Li, Wenping Hu, Fuzheng Zhang, Kun Gai, Guorui Zhou - [[pdf]](https://arxiv.org/pdf/2509.20712)</summary>

**Abstract:** Reinforcement learning (RL) has become a powerful paradigm for optimizing large language models (LLMs) to handle complex reasoning tasks. A core challenge in this process lies in managing policy entropy, which reflects the balance between exploration and exploitation during training. Existing methods, such as proximal policy optimization (PPO) and its variants, discard valuable gradient signals from low-probability tokens due to the clipping mechanism. We systematically analyze the entropy dynamics and reveal that these clipped tokens play a critical yet overlooked role in regulating entropy evolution. We propose \textbf{C}oordinating \textbf{E}ntropy via \textbf{G}radient-\textbf{P}reserving \textbf{P}olicy \textbf{O}ptimization (CE-GPPO), a novel algorithm that reintroduces gradients from clipped tokens in native PPO in a gentle and bounded manner. By controlling the magnitude of gradients from tokens outside the clipping interval, CE-GPPO is able to achieve an exploration-exploitation trade-off. We provide theoretical justification and empirical evidence showing that CE-GPPO effectively mitigates entropy instability. Extensive experiments on mathematical reasoning benchmarks show that CE-GPPO consistently outperforms strong baselines across different model scales.

**arXiv ID:** 2509.20712
</details>

<details>
<summary><strong>Pruning Cannot Hurt Robustness: Certified Trade-offs in Reinforcement Learning</strong> - James Pedley, Benjamin Etheridge, Stephen J. Roberts, Francesco Quinzan - [[pdf]](https://arxiv.org/pdf/2510.12939)</summary>

**Abstract:** Reinforcement learning (RL) policies deployed in real-world environments must remain reliable under adversarial perturbations. At the same time, modern deep RL agents are heavily over-parameterized, raising costs and fragility concerns. While pruning has been shown to improve robustness in supervised learning, its role in adversarial RL remains poorly understood. We develop the first theoretical framework for certified robustness under pruning in state-adversarial Markov decision processes (SA-MDPs). For Gaussian and categorical policies with Lipschitz networks, we prove that element-wise pruning can only tighten certified robustness bounds; pruning never makes the policy less robust. Building on this, we derive a novel three-term regret decomposition that disentangles clean-task performance, pruning-induced performance loss, and robustness gains, exposing a fundamental performance--robustness frontier. Empirically, we evaluate magnitude and micro-pruning schedules on continuous-control benchmarks with strong policy-aware adversaries. Across tasks, pruning consistently uncovers reproducible ``sweet spots'' at moderate sparsity levels, where robustness improves substantially without harming - and sometimes even enhancing - clean performance. These results position pruning not merely as a compression tool but as a structural intervention for robust RL.

**arXiv ID:** 2510.12939
</details>

<details>
<summary><strong>What is the objective of reasoning with reinforcement learning?</strong> - Damek Davis, Benjamin Recht - [[pdf]](https://arxiv.org/pdf/2510.13651)</summary>

**Abstract:** We show that several popular algorithms for reinforcement learning in large language models with binary rewards can be viewed as stochastic gradient ascent on a monotone transform of the probability of a correct answer given a prompt. In particular, the transformation associated with rejection sampling algorithms is the logarithm and that associated with the GRPO algorithm is the arcsine of the square root.

**arXiv ID:** 2510.13651
</details>

<details>
<summary><strong>Asymptotically optimal reinforcement learning in Block Markov Decision Processes</strong> - Thomas van Vuren, Fiona Sloothaak, Maarten G. Wolf, Jaron Sanders - [[pdf]](https://arxiv.org/pdf/2510.13748)</summary>

**Abstract:** The curse of dimensionality renders Reinforcement Learning (RL) impractical in many real-world settings with exponentially large state and action spaces. Yet, many environments exhibit exploitable structure that can accelerate learning. To formalize this idea, we study RL in Block Markov Decision Processes (BMDPs). BMDPs model problems with large observation spaces, but where transition dynamics are fully determined by latent states. Recent advances in clustering methods have enabled the efficient recovery of this latent structure. However, a regret analysis that exploits these techniques to determine their impact on learning performance remained open. We are now addressing this gap by providing a regret analysis that explicitly leverages clustering, demonstrating that accurate latent state estimation can indeed effectively speed up learning.
Concretely, this paper analyzes a two-phase RL algorithm for BMDPs that first learns the latent structure through random exploration and then switches to an optimism-guided strategy adapted to the uncovered structure. This algorithm achieves a regret that is $O(\sqrt{T}+n)$ on a large class of BMDPs susceptible to clustering. Here, $T$ denotes the number of time steps, $n$ is the cardinality of the observation space, and the Landau notation $O(\cdot)$ holds up to constants and polylogarithmic factors. This improves the best prior bound, $O(\sqrt{T}+n^2)$, especially when $n$ is large. Moreover, we prove that no algorithm can achieve lower regret uniformly on this same class of BMDPs. This establishes that, on this class, the algorithm achieves asymptotic optimality.

**arXiv ID:** 2510.13748
</details>

<details>
<summary><strong>MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control</strong> - Xue Bin Peng - [[pdf]](https://arxiv.org/pdf/2510.13794)</summary>

**Abstract:** MimicKit is an open-source framework for training motion controllers using motion imitation and reinforcement learning. The codebase provides implementations of commonly-used motion-imitation techniques and RL algorithms. This framework is intended to support research and applications in computer graphics and robotics by providing a unified training framework, along with standardized environment, agent, and data structures. The codebase is designed to be modular and easily configurable, enabling convenient modification and extension to new characters and tasks. The open-source codebase is available at: this https URL.

**arXiv ID:** 2510.13794
</details>

<details>
<summary><strong>Of Mice and Machines: A Comparison of Learning Between Real World Mice and RL Agents</strong> - Shuo Han, German Espinosa, Junda Huang, Daniel A. Dombeck, Malcolm A. MacIver, Bradly C. Stadie - [[pdf]](https://arxiv.org/pdf/2505.12204)</summary>

**Abstract:** Recent advances in reinforcement learning (RL) have demonstrated impressive capabilities in complex decision-making tasks. This progress raises a natural question: how do these artificial systems compare to biological agents, which have been shaped by millions of years of evolution? To help answer this question, we undertake a comparative study of biological mice and RL agents in a predator-avoidance maze environment. Through this analysis, we identify a striking disparity: RL agents consistently demonstrate a lack of self-preservation instinct, readily risking ``death'' for marginal efficiency gains. These risk-taking strategies are in contrast to biological agents, which exhibit sophisticated risk-assessment and avoidance behaviors. Towards bridging this gap between the biological and artificial, we propose two novel mechanisms that encourage more naturalistic risk-avoidance behaviors in RL agents. Our approach leads to the emergence of naturalistic behaviors, including strategic environment assessment, cautious path planning, and predator avoidance patterns that closely mirror those observed in biological systems.

**arXiv ID:** 2505.12204
</details>

<details>
<summary><strong>A Hierarchical Bin Packing Framework with Dual Manipulators via Heuristic Search and Deep Reinforcement Learning</strong> - Beomjoon Lee, Changjoo Nam - [[pdf]](https://arxiv.org/pdf/2506.01628)</summary>

**Abstract:** We address the bin packing problem (BPP), which aims to maximize bin utilization when packing a variety of items. The offline problem, where the complete information about the item set and their sizes is known in advance, is proven to be NP-hard. The semi-online and online variants are even more challenging, as full information about incoming items is unavailable. While existing methods have tackled both 2D and 3D BPPs, the 2D BPP remains underexplored in terms of fully maximizing utilization. We propose a hierarchical approach for solving the 2D online and semi-online BPP by combining deep reinforcement learning (RL) with heuristic search. The heuristic search selects which item to pack or unpack, determines the packing order, and chooses the orientation of each item, while the RL agent decides the precise position within the bin. Our method is capable of handling diverse scenarios, including repacking, varying levels of item information, differing numbers of accessible items, and coordination of dual manipulators. Experimental results demonstrate that our approach achieves near-optimal utilization across various practical scenarios, largely due to its repacking capability. In addition, the algorithm is evaluated in a physics-based simulation environment, where execution time is measured to assess its real-world performance.

**arXiv ID:** 2506.01628
</details>

<details>
<summary><strong>Towards Proprioception-Aware Embodied Planning for Dual-Arm Humanoid Robots</strong> - Boyu Li, Siyuan He, Hang Xu, Haoqi Yuan, Xinrun Xu, Yu Zang, Liwei Hu, Junpeng Yue, Zhenxiong Jiang, Pengbo Hu, Börje F. Karlsson, Yehui Tang, Zongqing Lu - [[pdf]](https://arxiv.org/pdf/2510.07882)</summary>

**Abstract:** In recent years, Multimodal Large Language Models (MLLMs) have demonstrated the ability to serve as high-level planners, enabling robots to follow complex human instructions. However, their effectiveness, especially in long-horizon tasks involving dual-arm humanoid robots, remains limited. This limitation arises from two main challenges: (i) the absence of simulation platforms that systematically support task evaluation and data collection for humanoid robots, and (ii) the insufficient embodiment awareness of current MLLMs, which hinders reasoning about dual-arm selection logic and body positions during planning. To address these issues, we present DualTHOR, a new dual-arm humanoid simulator, with continuous transition and a contingency mechanism. Building on this platform, we propose Proprio-MLLM, a model that enhances embodiment awareness by incorporating proprioceptive information with motion-based position embedding and a cross-spatial encoder. Experiments show that, while existing MLLMs struggle in this environment, Proprio-MLLM achieves an average improvement of 19.75% in planning performance. Our work provides both an essential simulation platform and an effective model to advance embodied intelligence in humanoid robotics. The code is available at this https URL.

**arXiv ID:** 2510.07882
</details>

<details>
<summary><strong>TaskAudit: Detecting Functiona11ity Errors in Mobile Apps via Agentic Task Execution</strong> - Mingyuan Zhong, Xia Chen, Davin Win Kyi, Chen Li, James Fogarty, Jacob O. Wobbrock - [[pdf]](https://arxiv.org/pdf/2510.12972)</summary>

**Abstract:** Accessibility checkers are tools in support of accessible app development and their use is encouraged by accessibility best practices. However, most current checkers evaluate static or mechanically-generated contexts, failing to capture common accessibility errors impacting mobile app functionality. We present TaskAudit, an accessibility evaluation system that focuses on detecting functiona11ity errors through simulated interactions. TaskAudit comprises three components: a Task Generator that constructs interactive tasks from app screens, a Task Executor that uses agents with a screen reader proxy to perform these tasks, and an Accessibility Analyzer that detects and reports accessibility errors by examining interaction traces. Evaluation on real-world apps shows that our strategy detects 48 functiona11ity errors from 54 app screens, compared to between 4 and 20 with existing checkers. Our analysis demonstrates common error patterns that TaskAudit can detect in addition to prior work, including label-functionality mismatch, cluttered navigation, and inappropriate feedback.

**arXiv ID:** 2510.12972
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
