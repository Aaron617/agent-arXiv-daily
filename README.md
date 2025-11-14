# Agent arXiv Daily

**Last Updated:** 2025-11-14 02:09:46

**Total Papers:** 67

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (3 papers)</h2></summary>

<details>
<summary><strong>Conversational Agents for Building Energy Efficiency -- Advising Housing Cooperatives in Stockholm on Reducing Energy Consumption</strong> - Shadaab Ghani, Anne Håkansson, Oleksii Pasichnyi, Hossein Shahrokni - [[pdf]](https://arxiv.org/pdf/2511.08587)</summary>

**Abstract:** Housing cooperative is a common type of multifamily building ownership in Sweden. Although this ownership structure grants decision-making autonomy, it places a burden of responsibility on cooperative's board members. Most board members lack the resources or expertise to manage properties and their energy consumption. This ignorance presents a unique challenge, especially given the EU directives that prohibit buildings rated as energy classes F and G by 2033. Conversational agents (CAs) enable human-like interactions with computer systems, facilitating human-computer interaction across various domains. In our case, CAs can be implemented to support cooperative members in making informed energy retrofitting and usage decisions. This paper introduces a Conversational agent system, called SPARA, designed to advise cooperatives on energy efficiency. SPARA functions as an energy efficiency advisor by leveraging the Retrieval-Augmented Generation (RAG) framework with a Language Model(LM). The LM generates targeted recommendations based on a knowledge base composed of email communications between professional energy advisors and cooperatives' representatives in Stockholm. The preliminary results indicate that SPARA can provide energy efficiency advice with precision 80\%, comparable to that of municipal energy efficiency (EE) experts. A pilot implementation is currently underway, where municipal EE experts are evaluating SPARA performance based on questions posed to EE experts by BRF members. Our findings suggest that LMs can significantly improve outreach by supporting stakeholders in their energy transition. For future work, more research is needed to evaluate this technology, particularly limitations to the stability and trustworthiness of its energy efficiency advice.

**arXiv ID:** 2511.08587
</details>

<details>
<summary><strong>Beyond Task-Oriented and Chitchat Dialogues: Proactive and Transition-Aware Conversational Agents</strong> - Yejin Yoon, Yuri Son, Namyoung So, Minseo Kim, Minsoo Cho, Chanhee Park, Seungshin Lee, Taeuk Kim - [[pdf]](https://arxiv.org/pdf/2511.08835)</summary>

**Abstract:** Conversational agents have traditionally been developed for either task-oriented dialogue (TOD) or open-ended chitchat, with limited progress in unifying the two. Yet, real-world conversations naturally involve fluid transitions between these modes. To address this gap, we introduce TACT (TOD-And-Chitchat Transition), a dataset designed for transition-aware dialogue modeling that incorporates structurally diverse and integrated mode flows. TACT supports both user- and agent-driven mode switches, enabling robust modeling of complex conversational dynamics. To evaluate an agent's ability to initiate and recover from mode transitions, we propose two new metrics -- Switch and Recovery. Models trained on TACT outperform baselines in both intent detection and mode transition handling. Moreover, applying Direct Preference Optimization (DPO) to TACT-trained models yields additional gains, achieving 75.74\% joint mode-intent accuracy and a 70.1\% win rate against GPT-4o in human evaluation. These results demonstrate that pairing structurally diverse data with DPO enhances response quality and transition control, paving the way for more proactive and transition-aware conversational agents.

**arXiv ID:** 2511.08835
</details>

<details>
<summary><strong>Steve: LLM Powered ChatBot for Career Progression</strong> - Naveen Mathews Renji, Balaji Rao, Carlo Lipizzi - [[pdf]](https://arxiv.org/pdf/2504.03789)</summary>

**Abstract:** The advancements in systems deploying large language models (LLMs), as well as improvements in their ability to act as agents with predefined templates, provide an opportunity to conduct qualitative, individualized assessments, creating a bridge between qualitative and quantitative methods for candidates seeking career progression. In this paper, we develop a platform that allows candidates to run AI-led interviews to assess their current career stage and curate coursework to enable progression to the next level. Our approach incorporates predefined career trajectories, associated skills, and a method to recommend the best resources for gaining the necessary skills for advancement. We employ OpenAI API calls along with expertly compiled chat templates to assess candidate competence. Our platform is highly configurable due to the modularity of the development, is easy to deploy and use, and available as a web interface where the only requirement is candidate resumes in PDF format. We demonstrate a use-case centered on software engineering and intend to extend this platform to be domain-agnostic, requiring only regular updates to chat templates as industries evolve.

**arXiv ID:** 2504.03789
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>ProBench: Benchmarking GUI Agents with Accurate Process Information</strong> - Leyang Yang, Ziwei Wang, Xiaoxuan Tang, Sheng Zhou, Dajun Chen, Wei Jiang, Yong Li - [[pdf]](https://arxiv.org/pdf/2511.09157)</summary>

**Abstract:** With the deep integration of artificial intelligence and interactive technology, Graphical User Interface (GUI) Agent, as the carrier connecting goal-oriented natural language and real-world devices, has received widespread attention from the community. Contemporary benchmarks aim to evaluate the comprehensive capabilities of GUI agents in GUI operation tasks, generally determining task completion solely by inspecting the final screen state. However, GUI operation tasks consist of multiple chained steps while not all critical information is presented in the final few pages. Although a few research has begun to incorporate intermediate steps into evaluation, accurately and automatically capturing this process information still remains an open challenge. To address this weakness, we introduce ProBench, a comprehensive mobile benchmark with over 200 challenging GUI tasks covering widely-used scenarios. Remaining the traditional State-related Task evaluation, we extend our dataset to include Process-related Task and design a specialized evaluation method. A newly introduced Process Provider automatically supplies accurate process information, enabling presice assessment of agent's performance. Our evaluation of advanced GUI agents reveals significant limitations for real-world GUI scenarios. These shortcomings are prevalent across diverse models, including both large-scale generalist models and smaller, GUI-specific models. A detailed error analysis further exposes several universal problems, outlining concrete directions for future improvements.

**arXiv ID:** 2511.09157
</details>

<details>
<summary><strong>Learning API Functionality from In-Context Demonstrations for Tool-based Agents</strong> - Bhrij Patel, Ashish Jagmohan, Aditya Vempaty - [[pdf]](https://arxiv.org/pdf/2505.24197)</summary>

**Abstract:** Digital tool-based agents, powered by Large Language Models (LLMs), that invoke external Application Programming Interfaces (APIs) often rely on documentation to understand API functionality. However, such documentation is frequently missing, outdated, privatized, or inconsistent-hindering the development of reliable, general-purpose agents. In this work, we propose a new research direction: learning of API functionality directly from in-context demonstrations. This task is a new paradigm applicable in scenarios without documentation. Using API benchmarks, we collect demonstrations from both expert agents and from self-exploration. To understand what information demonstrations must convey for successful task completion, we extensively study how the number of demonstrations and the use of LLM-generated summaries and evaluations affect the task success rate of the API-based agent. Our experiments across 3 datasets and 6 models show that learning functionality from in-context demonstrations remains a non-trivial challenge, even for state-of-the-art LLMs. We find that providing explicit function calls and natural language critiques significantly improves the agent's task success rate due to more accurate parameter filling. We analyze failure modes, identify sources of error, and highlight key open challenges for future work in documentation-free, self-improving, API-based agents.

**arXiv ID:** 2505.24197
</details>

<details>
<summary><strong>Conversational Intent-Driven GraphRAG: Enhancing Multi-Turn Dialogue Systems through Adaptive Dual-Retrieval of Flow Patterns and Context Semantics</strong> - Ziqi Zhu, Tao Hu, Honglong Zhang, Dan Yang, HanGeng Chen, Mengran Zhang, Xilun Chen - [[pdf]](https://arxiv.org/pdf/2506.19385)</summary>

**Abstract:** We present CID-GraphRAG (Conversational Intent-Driven Graph Retrieval Augmented Generation), a novel framework that addresses the limitations of existing dialogue systems in maintaining both contextual coherence and goal-oriented progression in multi-turn customer service conversations. Unlike traditional RAG systems that rely solely on semantic similarity (Conversation RAG) or standard knowledge graphs (GraphRAG), CID-GraphRAG constructs dynamic intent transition graphs from goal achieved historical dialogues and implements a dual-retrieval mechanism that adaptively balances intent-based graph traversal with semantic search. This approach enables the system to simultaneously leverage both conversional intent flow patterns and contextual semantics, significantly improving retrieval quality and response quality. In extensive experiments on real-world customer service dialogues, we employ both automatic metrics and LLM-as-judge assessments, demonstrating that CID-GraphRAG significantly outperforms both semantic-based Conversation RAG and intent-based GraphRAG baselines across all evaluation criteria. Quantitatively, CID-GraphRAG demonstrates substantial improvements over Conversation RAG across automatic metrics, with relative gains of 11% in BLEU, 5% in ROUGE-L, 6% in METEOR, and most notably, a 58% improvement in response quality according to LLM-as-judge evaluations. These results demonstrate that the integration of intent transition structures with semantic retrieval creates a synergistic effect that neither approach achieves independently, establishing CID-GraphRAG as an effective framework for addressing the challenges of maintaining contextual coherence and goal-oriented progression in knowledge-intensive multi-turn dialogues.

**arXiv ID:** 2506.19385
</details>

<details>
<summary><strong>AgentFlux: Decoupled Fine-Tuning & Inference for On-Device Agentic Systems</strong> - Rohan Kadekodi, Zhan Jin, Keisuke Kamahori, Yile Gu, Sean Khatiri, Noah H. Bayindirli, Sergey Gorbunov, Baris Kasikci - [[pdf]](https://arxiv.org/pdf/2510.00229)</summary>

**Abstract:** The deployment of Large Language Models (LLMs) as agentic orchestrators has revolutionized task automation, but the need for privacy-preserving, cost-effective solutions demands on-device inference capabilities. However, local LLMs consistently underperform compared to frontier models in tool calling scenarios, struggling with both tool selection from large tool sets and accurate argument generation for complex parameter structures. We introduce a methodology that disaggregates a tool-calling task into two distinct subtasks: tool selection and argument generation. We propose "decoupled fine-tuning", a novel post-training approach that employs LoRA fine-tuning to create dedicated LoRA adapters for tool selection and tool-specific argument generation using separate loss masking for each of the subtasks. Furthermore, we present AgentFlux, an inference framework that leverages the LoRA adapters created using decoupled fine-tuning to perform efficient agent orchestration with the help of local models on end-user devices. AgentFlux decomposes the tool-call generation step into tool selection and argument generation, and dynamically loads the corresponding LoRA adapters to generate tool calls. Additionally, AgentFlux implements hierarchical orchestration to restrict the number of tools required for tool selection. Our experiments on the MCP-Bench benchmark demonstrate that the Qwen-2.5-7B model trained using decoupled fine-tuning improves the tool calling accuracy of the base model by 46%, and outperforms other local reasoning, non-reasoning and fine-tuned models of similar size in all cases, and models that are 2x larger, in most cases.

**arXiv ID:** 2510.00229
</details>

<details>
<summary><strong>DigiData: Training and Evaluating General-Purpose Mobile Control Agents</strong> - Yuxuan Sun, Manchen Wang, Shengyi Qian, William R. Wong, Eric Gan, Pierluca D'Oro, Alejandro Castillejo Munoz, Sneha Silwal, Pedro Matias, Nitin Kamra, Satwik Kottur, Nick Raines, Xuanyi Zhao, Joy Chen, Joseph Greer, Andrea Madotto, Allen Bolourchi, James Valori, Kevin Carlberg, Karl Ridgeway, Joseph Tighe - [[pdf]](https://arxiv.org/pdf/2511.07413)</summary>

**Abstract:** AI agents capable of controlling user interfaces have the potential to transform human interaction with digital devices. To accelerate this transformation, two fundamental building blocks are essential: high-quality datasets that enable agents to achieve complex and human-relevant goals, and robust evaluation methods that allow researchers and practitioners to rapidly enhance agent performance. In this paper, we introduce DigiData, a large-scale, high-quality, diverse, multi-modal dataset designed for training mobile control agents. Unlike existing datasets, which derive goals from unstructured interactions, DigiData is meticulously constructed through comprehensive exploration of app features, resulting in greater diversity and higher goal complexity. Additionally, we present DigiData-Bench, a benchmark for evaluating mobile control agents on real-world complex tasks. We demonstrate that the commonly used step-accuracy metric falls short in reliably assessing mobile control agents and, to address this, we propose dynamic evaluation protocols and AI-powered evaluations as rigorous alternatives for agent assessment. Our contributions aim to significantly advance the development of mobile control agents, paving the way for more intuitive and effective human-device interactions.

**arXiv ID:** 2511.07413
</details>

<details>
<summary><strong>LLM4AD: Large Language Models for Autonomous Driving -- Concept, Review, Benchmark, Experiments, and Future Trends</strong> - Can Cui, Yunsheng Ma, Sung-Yeon Park, Zichong Yang, Yupeng Zhou, Juanwu Lu, Juntong Peng, Jiaru Zhang, Ruqi Zhang, Lingxi Li, Yaobin Chen, Jitesh H. Panchal, Amr Abdelraouf, Rohit Gupta, Kyungtae Han, Ziran Wang - [[pdf]](https://arxiv.org/pdf/2410.15281)</summary>

**Abstract:** With the broader adoption and highly successful development of Large Language Models (LLMs), there has been growing interest and demand for applying LLMs to autonomous driving technology. Driven by their natural language understanding and reasoning capabilities, LLMs have the potential to enhance various aspects of autonomous driving systems, from perception and scene understanding to interactive decision-making. In this paper, we first introduce the novel concept of designing Large Language Models for Autonomous Driving (LLM4AD), followed by a review of existing LLM4AD studies. Then, we propose a comprehensive benchmark for evaluating the instruction-following and reasoning abilities of LLM4AD systems, which includes LaMPilot-Bench, CARLA Leaderboard 1.0 Benchmark in simulation and NuPlanQA for multi-view visual question answering. Furthermore, we conduct extensive real-world experiments on autonomous vehicle platforms, examining both on-cloud and on-edge LLM deployment for personalized decision-making and motion control. Next, we explore the future trends of integrating language diffusion models into autonomous driving, exemplified by the proposed ViLaD (Vision-Language Diffusion) framework. Finally, we discuss the main challenges of LLM4AD, including latency, deployment, security and privacy, safety, trust and transparency, and personalization.

**arXiv ID:** 2410.15281
</details>

<details>
<summary><strong>TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving</strong> - Yiming Yang, Yueru Luo, Bingkun He, Hongbin Lin, Suzhong Fu, Chao Zheng, Zhipeng Cao, Erlong Li, Chao Yan, Shuguang Cui, Zhen Li - [[pdf]](https://arxiv.org/pdf/2507.00709)</summary>

**Abstract:** Lane segment topology reasoning constructs a comprehensive road network by capturing the topological relationships between lane segments and their semantic types. This enables end-to-end autonomous driving systems to perform road-dependent maneuvers such as turning and lane changing. However, the limitations in consistent positional embedding and temporal multiple attribute learning in existing methods hinder accurate roadnet reconstruction. To address these issues, we propose TopoStreamer, an end-to-end temporal perception model for lane segment topology reasoning. Specifically, TopoStreamer introduces three key improvements: streaming attribute constraints, dynamic lane boundary positional encoding, and lane segment denoising. The streaming attribute constraints enforce temporal consistency in both centerline and boundary coordinates, along with their classifications. Meanwhile, dynamic lane boundary positional encoding enhances the learning of up-to-date positional information within queries, while lane segment denoising helps capture diverse lane segment patterns, ultimately improving model performance. Additionally, we assess the accuracy of existing models using a lane boundary classification metric, which serves as a crucial measure for lane-changing scenarios in autonomous driving. On the OpenLane-V2 dataset, TopoStreamer demonstrates significant improvements over state-of-the-art methods, achieving substantial performance gains of +3.0% mAP in lane segment perception and +1.7% OLS in centerline perception tasks.

**arXiv ID:** 2507.00709
</details>

<details>
<summary><strong>Survey of Vision-Language-Action Models for Embodied Manipulation</strong> - Haoran Li, Yuhui Chen, Wenbo Cui, Weiheng Liu, Kai Liu, Mingcai Zhou, Zhengtao Zhang, Dongbin Zhao - [[pdf]](https://arxiv.org/pdf/2508.15201)</summary>

**Abstract:** Embodied intelligence systems, which enhance agent capabilities through continuous environment interactions, have garnered significant attention from both academia and industry. Vision-Language-Action models, inspired by advancements in large foundation models, serve as universal robotic control frameworks that substantially improve agent-environment interaction capabilities in embodied intelligence systems. This expansion has broadened application scenarios for embodied AI robots. This survey comprehensively reviews VLA models for embodied manipulation. Firstly, it chronicles the developmental trajectory of VLA architectures. Subsequently, we conduct a detailed analysis of current research across 5 critical dimensions: VLA model structures, training datasets, pre-training methods, post-training methods, and model evaluation. Finally, we synthesize key challenges in VLA development and real-world deployment, while outlining promising future research directions.

**arXiv ID:** 2508.15201
</details>

<details>
<summary><strong>FLAD: Federated Learning for LLM-based Autonomous Driving in Vehicle-Edge-Cloud Networks</strong> - Tianao Xiang, Mingjian Zhi, Yuanguo Bi, Lin Cai, Yuhao Chen - [[pdf]](https://arxiv.org/pdf/2511.09025)</summary>

**Abstract:** Large Language Models (LLMs) have impressive data fusion and reasoning capabilities for autonomous driving (AD). However, training LLMs for AD faces significant challenges including high computation transmission costs, and privacy concerns associated with sensitive driving data. Federated Learning (FL) is promising for enabling autonomous vehicles (AVs) to collaboratively train models without sharing raw data. We present Federated LLM-based Autonomous Driving (FLAD), an FL framework that leverages distributed multimodal sensory data across AVs in heterogeneous environment. FLAD has three key innovations: (1) a cloud-edge-vehicle collaborative architecture that reduces communication delay and preserving data privacy; (2) an intelligent parallelized collaborative training with a communication scheduling mechanism that optimizes training efficiency, leveraging end-devices otherwise having insufficient resources for model training; and (3) a knowledge distillation method that personalizes LLM according to heterogeneous edge data. In addition, we prototype FLAD in a testbed with NVIDIA Jetsons, overcoming practical implementation challenges including CPU/GPU memory sharing in resource-constrained devices, dynamic model partitions, and fault-tolerant this http URL experimental evaluation demonstrates that FLAD achieves superior end-to-end AD performance while efficiently utilizing distributed vehicular resources, opening up new possibilities for future collaborative AD model training and knowledge sharing.

**arXiv ID:** 2511.09025
</details>

<details>
<summary><strong>Data Assessment for Embodied Intelligence</strong> - Jiahao Xiao, Bowen Yan, Jianbo Zhang, Jia Wang, Chunyi Li, Zhengxue Cheng, Guangtao Zhai - [[pdf]](https://arxiv.org/pdf/2511.09119)</summary>

**Abstract:** In embodied intelligence, datasets play a pivotal role, serving as both a knowledge repository and a conduit for information transfer. The two most critical attributes of a dataset are the amount of information it provides and how easily this information can be learned by models. However, the multimodal nature of embodied data makes evaluating these properties particularly challenging. Prior work has largely focused on diversity, typically counting tasks and scenes or evaluating isolated modalities, which fails to provide a comprehensive picture of dataset diversity. On the other hand, the learnability of datasets has received little attention and is usually assessed post-hoc through model training, an expensive, time-consuming process that also lacks interpretability, offering little guidance on how to improve a dataset. In this work, we address both challenges by introducing two principled, data-driven tools. First, we construct a unified multimodal representation for each data sample and, based on it, propose diversity entropy, a continuous measure that characterizes the amount of information contained in a dataset. Second, we introduce the first interpretable, data-driven algorithm to efficiently quantify dataset learnability without training, enabling researchers to assess a dataset's learnability immediately upon its release. We validate our algorithm on both simulated and real-world embodied datasets, demonstrating that it yields faithful, actionable insights that enable researchers to jointly improve diversity and learnability. We hope this work provides a foundation for designing higher-quality datasets that advance the development of embodied intelligence.

**arXiv ID:** 2511.09119
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>Benevolent Dictators? On LLM Agent Behavior in Dictator Games</strong> - Andreas Einwiller, Kanishka Ghosh Dastidar, Artur Romazanov, Annette Hautli-Janisz, Michael Granitzer, Florian Lemmerich - [[pdf]](https://arxiv.org/pdf/2511.08721)</summary>

**Abstract:** In behavioral sciences, experiments such as the ultimatum game are conducted to assess preferences for fairness or self-interest of study participants. In the dictator game, a simplified version of the ultimatum game where only one of two players makes a single decision, the dictator unilaterally decides how to split a fixed sum of money between themselves and the other player. Although recent studies have explored behavioral patterns of AI agents based on Large Language Models (LLMs) instructed to adopt different personas, we question the robustness of these results. In particular, many of these studies overlook the role of the system prompt - the underlying instructions that shape the model's behavior - and do not account for how sensitive results can be to slight changes in prompts. However, a robust baseline is essential when studying highly complex behavioral aspects of LLMs. To overcome previous limitations, we propose the LLM agent behavior study (LLM-ABS) framework to (i) explore how different system prompts influence model behavior, (ii) get more reliable insights into agent preferences by using neutral prompt variations, and (iii) analyze linguistic features in responses to open-ended instructions by LLM agents to better understand the reasoning behind their behavior. We found that agents often exhibit a strong preference for fairness, as well as a significant impact of the system prompt on their behavior. From a linguistic perspective, we identify that models express their responses differently. Although prompt sensitivity remains a persistent challenge, our proposed framework demonstrates a robust foundation for LLM agent behavior studies. Our code artifacts are available at this https URL.

**arXiv ID:** 2511.08721
</details>

<details>
<summary><strong>Structured Uncertainty guided Clarification for LLM Agents</strong> - Manan Suri, Puneet Mathur, Nedim Lipka, Franck Dernoncourt, Ryan A. Rossi, Dinesh Manocha - [[pdf]](https://arxiv.org/pdf/2511.08798)</summary>

**Abstract:** LLM agents extend large language models with tool-calling capabilities, but ambiguous user instructions often lead to incorrect invocations and task failures. We introduce a principled formulation of structured uncertainty over tool-call parameters, modeling joint tool-argument clarification as a POMDP with Expected Value of Perfect Information (EVPI) objective for optimal question selection and aspect-based cost modeling to prevent redundancy. Our SAGE-Agent leverages this structured uncertainty to achieve superior efficiency: increasing coverage on ambiguous tasks by 7-39\% while reducing clarification questions by 1.5-2.7$\times$ compared to strong prompting and uncertainty-based baselines. We present ClarifyBench, the first multi-turn tool-augmented disambiguation benchmark with realistic LLM-based user simulation across diverse domains including document editing, vehicle control, and travel booking. Additionally, we demonstrate that structured uncertainty provides effective training signals for reinforcement learning, boosting When2Call accuracy from 36.5\% to 65.2\% (3B model) and 36.7\% to 62.9\% (7B model) through uncertainty-weighted GRPO training. These results establish structured uncertainty as a principled, efficient approach for tool-augmented agents, improving both task success and interaction efficiency in real-world scenarios.

**arXiv ID:** 2511.08798
</details>

<details>
<summary><strong>Structured Cognitive Loop for Behavioral Intelligence in Large Language Model Agents</strong> - Myung Ho Kim - [[pdf]](https://arxiv.org/pdf/2510.05107)</summary>

**Abstract:** Large language models have advanced natural language understanding and generation, but their use as autonomous agents introduces architectural challenges for multi-step tasks. Existing frameworks often mix cognition, memory, and control in a single prompt, reducing coherence and predictability. The Structured Cognitive Loop (SCL) is proposed as an alternative architecture that separates these functions. In SCL, the language model handles cognition, memory is stored externally, and execution is guided by a lightweight controller within a goal-directed loop. This design allows intermediate results to be recorded and verified before actions are taken, improving traceability and evaluation. SCL is evaluated against prompt-based baselines such as ReAct and LangChain agents across three tasks: travel planning, conditional email drafting, and constraint-guided image generation. Under matched settings, SCL achieves an average task success rate of 86.3 percent, compared with 70.5 to 76.8 percent for baselines. It also shows higher goal fidelity, fewer redundant calls, and reduced unsupported assertions. These results indicate that separating cognition, memory, and control can enhance reliability and interpretability without relying on larger models or heavier prompts. The findings should be regarded as preliminary evidence, with broader tests across model families and task domains planned for future work.

**arXiv ID:** 2510.05107
</details>

<details>
<summary><strong>UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning</strong> - Jiawei Zhang, Shuang Yang, Bo Li - [[pdf]](https://arxiv.org/pdf/2503.01908)</summary>

**Abstract:** Large Language Model (LLM) agents equipped with external tools have become increasingly powerful for complex tasks such as web shopping, automated email replies, and financial trading. However, these advancements amplify the risks of adversarial attacks, especially when agents can access sensitive external functionalities. Nevertheless, manipulating LLM agents into performing targeted malicious actions or invoking specific tools remains challenging, as these agents extensively reason or plan before executing final actions. In this work, we present UDora, a unified red teaming framework designed for LLM agents that dynamically hijacks the agent's reasoning processes to compel malicious behavior. Specifically, UDora first generates the model's reasoning trace for the given task, then automatically identifies optimal points within this trace to insert targeted perturbations. The resulting perturbed reasoning is then used as a surrogate response for optimization. By iteratively applying this process, the LLM agent will then be induced to undertake designated malicious actions or to invoke specific malicious tools. Our approach demonstrates superior effectiveness compared to existing methods across three LLM agent datasets. The code is available at this https URL.

**arXiv ID:** 2503.01908
</details>

<details>
<summary><strong>BioVerge: A Comprehensive Benchmark and Study of Self-Evaluating Agents for Biomedical Hypothesis Generation</strong> - Fuyi Yang, Chenchen Ye, Mingyu Derek Ma, Yijia Xiao, Matthew Yang, Wei Wang - [[pdf]](https://arxiv.org/pdf/2511.08866)</summary>

**Abstract:** Hypothesis generation in biomedical research has traditionally centered on uncovering hidden relationships within vast scientific literature, often using methods like Literature-Based Discovery (LBD). Despite progress, current approaches typically depend on single data types or predefined extraction patterns, which restricts the discovery of novel and complex connections. Recent advances in Large Language Model (LLM) agents show significant potential, with capabilities in information retrieval, reasoning, and generation. However, their application to biomedical hypothesis generation has been limited by the absence of standardized datasets and execution environments. To address this, we introduce BioVerge, a comprehensive benchmark, and BioVerge Agent, an LLM-based agent framework, to create a standardized environment for exploring biomedical hypothesis generation at the frontier of existing scientific knowledge. Our dataset includes structured and textual data derived from historical biomedical hypotheses and PubMed literature, organized to support exploration by LLM agents. BioVerge Agent utilizes a ReAct-based approach with distinct Generation and Evaluation modules that iteratively produce and self-assess hypothesis proposals. Through extensive experimentation, we uncover key insights: 1) different architectures of BioVerge Agent influence exploration diversity and reasoning strategies; 2) structured and textual information sources each provide unique, critical contexts that enhance hypothesis generation; and 3) self-evaluation significantly improves the novelty and relevance of proposed hypotheses.

**arXiv ID:** 2511.08866
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (17 papers)</h2></summary>

<details>
<summary><strong>AI Founding Fathers: A Case Study of GIS Search in Multi-Agent Pipelines</strong> - Alvin Chauhan - [[pdf]](https://arxiv.org/pdf/2511.09005)</summary>

**Abstract:** Although Large Language Models (LLMs) show exceptional fluency, efforts persist to extract stronger reasoning capabilities from them. Drawing on search-based interpretations of LLM computation, this paper advances a systematic framework for understanding LLM reasoning and optimization. Namely, that enhancing reasoning is best achieved by structuring a multi-agent pipeline to ensure a traversal of the search space in a gradual, incremental, and sequential (GIS) manner. Stated succinctly, high-quality reasoning is a controlled, incremental search. To test this framework, we investigate the efficacy of recursive refinement (RR)--an iterative process of self-criticism, adversarial stress-testing, and integrating critical feedback--as a practical method for implementing GIS search. We designed an experiment comparing a simple, linear pipeline against a complex, explicitly structured pipeline leveraging a recursive refinement layer. The multi-agent models were constructed to reflect the historical personas of three US Founding Fathers (Hamilton, Jefferson, and Madison) using RAG-powered corpora and were prompted to generate responses to three contemporary political issues. Model performance was evaluated using a two-tiered approach: a quantitative score from an LLM arbiter agent and qualitative human judgment. Our results revealed that the complex model consistently outperformed the simple model across all nine test cases with an average arbiter-outputted score of 88.3 versus 71.7. The complex model's arguments were superior in analytical depth, structural nuance, and strategic framing. We conclude that recursive refinement is a robust architectural feature for enhancing LLM reasoning via GIS search.

**arXiv ID:** 2511.09005
</details>

<details>
<summary><strong>Robust and Diverse Multi-Agent Learning via Rational Policy Gradient</strong> - Niklas Lauffer, Ameesh Shah, Micah Carroll, Sanjit A. Seshia, Stuart Russell, Michael Dennis - [[pdf]](https://arxiv.org/pdf/2511.09535)</summary>

**Abstract:** Adversarial optimization algorithms that explicitly search for flaws in agents' policies have been successfully applied to finding robust and diverse policies in multi-agent settings. However, the success of adversarial optimization has been largely limited to zero-sum settings because its naive application in cooperative settings leads to a critical failure mode: agents are irrationally incentivized to self-sabotage, blocking the completion of tasks and halting further learning. To address this, we introduce Rationality-preserving Policy Optimization (RPO), a formalism for adversarial optimization that avoids self-sabotage by ensuring agents remain rational--that is, their policies are optimal with respect to some possible partner policy. To solve RPO, we develop Rational Policy Gradient (RPG), which trains agents to maximize their own reward in a modified version of the original game in which we use opponent shaping techniques to optimize the adversarial objective. RPG enables us to extend a variety of existing adversarial optimization algorithms that, no longer subject to the limitations of self-sabotage, can find adversarial examples, improve robustness and adaptability, and learn diverse policies. We empirically validate that our approach achieves strong performance in several popular cooperative and general-sum environments. Our project page can be found at this https URL.

**arXiv ID:** 2511.09535
</details>

<details>
<summary><strong>Bio AI Agent: A Multi-Agent Artificial Intelligence System for Autonomous CAR-T Cell Therapy Development with Integrated Target Discovery, Toxicity Prediction, and Rational Molecular Design</strong> - Yi Ni, Liwei Zhu, Shuai Li - [[pdf]](https://arxiv.org/pdf/2511.08649)</summary>

**Abstract:** Chimeric antigen receptor T-cell (CAR-T) therapy represents a paradigm shift in cancer treatment, yet development timelines of 8-12 years and clinical attrition rates exceeding 40-60% highlight critical inefficiencies in target selection, safety assessment, and molecular optimization. We present Bio AI Agent, a multi-agent artificial intelligence system powered by large language models that enables autonomous CAR-T development through collaborative specialized agents. The system comprises six autonomous agents: Target Selection Agent for multi-parametric antigen prioritization across >10,000 cancer-associated targets, Toxicity Prediction Agent for comprehensive safety profiling integrating tissue expression atlases and pharmacovigilance databases, Molecular Design Agent for rational CAR engineering, Patent Intelligence Agent for freedom-to-operate analysis, Clinical Translation Agent for regulatory compliance, and Decision Orchestration Agent for multi-agent coordination. Retrospective validation demonstrated autonomous identification of high-risk targets including FcRH5 (hepatotoxicity) and CD229 (off-tumor toxicity), patent infringement risks for CD38+SLAMF7 combinations, and generation of comprehensive development roadmaps. By enabling parallel processing, specialized reasoning, and autonomous decision-making superior to monolithic AI systems, Bio AI Agent addresses critical gaps in precision oncology development and has potential to accelerate translation of next-generation immunotherapies from discovery to clinic.

**arXiv ID:** 2511.08649
</details>

<details>
<summary><strong>Convergence dynamics of Agent-to-Agent Interactions with Misaligned objectives</strong> - Romain Cosentino, Sarath Shekkizhar, Adam Earle - [[pdf]](https://arxiv.org/pdf/2511.08710)</summary>

**Abstract:** We develop a theoretical framework for agent-to-agent interactions in multi-agent scenarios. We consider the setup in which two language model based agents perform iterative gradient updates toward their respective objectives in-context, using the output of the other agent as input. We characterize the generation dynamics associated with the interaction when the agents have misaligned objectives, and show that this results in a biased equilibrium where neither agent reaches its target - with the residual errors predictable from the objective gap and the geometry induced by the prompt of each agent. We establish the conditions for asymmetric convergence and provide an algorithm that provably achieves an adversarial result, producing one-sided success. Experiments with trained transformer models as well as GPT$5$ for the task of in-context linear regression validate the theory. Our framework presents a setup to study, predict, and defend multi-agent systems; explicitly linking prompt design and interaction setup to stability, bias, and robustness.

**arXiv ID:** 2511.08710
</details>

<details>
<summary><strong>Information-Driven Fault Detection and Identification for Multi-Agent Spacecraft Systems: Collaborative On-Orbit Inspection Mission</strong> - Akshita Gupta, Arna Bhardwaj, Yashwanth Kumar Nakka, Changrak Choi, Amir Rahmani - [[pdf]](https://arxiv.org/pdf/2511.08752)</summary>

**Abstract:** This work presents a global-to-local, task-aware fault detection and identification (FDI) framework for multi-spacecraft systems conducting collaborative inspection missions in low Earth orbit. The inspection task is represented by a global information-driven cost functional that integrates the sensor model, spacecraft poses, and mission-level information-gain objectives. This formulation links guidance, control, and FDI by using the same cost function to drive both global task allocation and local sensing or motion decisions. Fault detection is achieved through comparisons between expected and observed task metrics, while higher-order cost-gradient measures enable the identification of faults among sensors, actuators, and state estimators. An adaptive thresholding mechanism captures the time-varying inspection geometry and dynamic mission conditions. Simulation results for representative multi-spacecraft inspection scenarios demonstrate the reliability of fault localization and classification under uncertainty, providing a unified, information-driven foundation for resilient autonomous inspection architectures.

**arXiv ID:** 2511.08752
</details>

<details>
<summary><strong>TIGER-MARL: Enhancing Multi-Agent Reinforcement Learning with Temporal Information through Graph-based Embeddings and Representations</strong> - Nikunj Gupta, Ludwika Twardecka, James Zachary Hare, Jesse Milzman, Rajgopal Kannan, Viktor Prasanna - [[pdf]](https://arxiv.org/pdf/2511.08832)</summary>

**Abstract:** In this paper, we propose capturing and utilizing \textit{Temporal Information through Graph-based Embeddings and Representations} or \textbf{TIGER} to enhance multi-agent reinforcement learning (MARL). We explicitly model how inter-agent coordination structures evolve over time. While most MARL approaches rely on static or per-step relational graphs, they overlook the temporal evolution of interactions that naturally arise as agents adapt, move, or reorganize cooperation strategies. Capturing such evolving dependencies is key to achieving robust and adaptive coordination. To this end, TIGER constructs dynamic temporal graphs of MARL agents, connecting their current and historical interactions. It then employs a temporal attention-based encoder to aggregate information across these structural and temporal neighborhoods, yielding time-aware agent embeddings that guide cooperative policy learning. Through extensive experiments on two coordination-intensive benchmarks, we show that TIGER consistently outperforms diverse value-decomposition and graph-based MARL baselines in task performance and sample efficiency. Furthermore, we conduct comprehensive ablation studies to isolate the impact of key design parameters in TIGER, revealing how structural and temporal factors can jointly shape effective policy learning in MARL. All codes can be found here: this https URL.

**arXiv ID:** 2511.08832
</details>

<details>
<summary><strong>Achieving Equilibrium under Utility Heterogeneity: An Agent-Attention Framework for Multi-Agent Multi-Objective Reinforcement Learning</strong> - Zhuhui Li, Chunbo Luo, Liming Huang, Luyu Qi, Geyong Min - [[pdf]](https://arxiv.org/pdf/2511.08926)</summary>

**Abstract:** Multi-agent multi-objective systems (MAMOS) have emerged as powerful frameworks for modelling complex decision-making problems across various real-world domains, such as robotic exploration, autonomous traffic management, and sensor network optimisation. MAMOS offers enhanced scalability and robustness through decentralised control and more accurately reflects inherent trade-offs between conflicting objectives. In MAMOS, each agent uses utility functions that map return vectors to scalar values. Existing MAMOS optimisation methods face challenges in handling heterogeneous objective and utility function settings, where training non-stationarity is intensified due to private utility functions and the associated policies. In this paper, we first theoretically prove that direct access to, or structured modeling of, global utility functions is necessary for the Bayesian Nash Equilibrium under decentralised execution constraints. To access the global utility functions while preserving the decentralised execution, we propose an Agent-Attention Multi-Agent Multi-Objective Reinforcement Learning (AA-MAMORL) framework. Our approach implicitly learns a joint belief over other agents' utility functions and their associated policies during centralised training, effectively mapping global states and utilities to each agent's policy. In execution, each agent independently selects actions based on local observations and its private utility function to approximate a BNE, without relying on inter-agent communication. We conduct comprehensive experiments in both a custom-designed MAMO Particle environment and the standard MOMALand benchmark. The results demonstrate that access to global preferences and our proposed AA-MAMORL significantly improve performance and consistently outperform state-of-the-art methods.

**arXiv ID:** 2511.08926
</details>

<details>
<summary><strong>Tele-LLM-Hub: Building Context-Aware Multi-Agent LLM Systems for Telecom Networks</strong> - Vijay K Shah, Cong Shen - [[pdf]](https://arxiv.org/pdf/2511.09087)</summary>

**Abstract:** This paper introduces Tele-LLM-Hub, a user friendly low-code solution for rapid prototyping and deployment of context aware multi-agent (MA) Large Language Model (LLM) systems tailored for 5G and beyond. As telecom wireless networks become increasingly complex, intelligent LLM applications must share a domainspecific understanding of network state. We propose TeleMCP, the Telecom Model Context Protocol, to enable structured and context-rich communication between agents in telecom environments. Tele-LLM-Hub actualizes TeleMCP through a low-code interface that supports agent creation, workflow composition, and interaction with software stacks such as srsRAN. Key components include a direct chat interface, a repository of pre-built systems, an Agent Maker leveraging finetuning with our RANSTRUCT framework, and an MA-Maker for composing MA workflows. The goal of Tele-LLM-Hub is to democratize the design of contextaware MA systems and accelerate innovation in next-generation wireless networks.

**arXiv ID:** 2511.09087
</details>

<details>
<summary><strong>MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning</strong> - Jian Zhang, Zhiyuan Wang, Zhangqi Wang, Fangzhi Xu, Qika Lin, Lingling Zhang, Rui Mao, Erik Cambria, Jun Liu - [[pdf]](https://arxiv.org/pdf/2503.16905)</summary>

**Abstract:** Collaborative reasoning with multiple agents offers the potential for more robust and diverse problem-solving. However, existing approaches often suffer from homogeneous agent behaviors and lack of reflective and rethinking capabilities. We propose Multi-Agent Personality Shaping (MAPS), a novel framework that enhances reasoning through agent diversity and internal critique. Inspired by the Big Five personality theory, MAPS assigns distinct personality traits to individual agents, shaping their reasoning styles and promoting heterogeneous collaboration. To enable deeper and more adaptive reasoning, MAPS introduces a Critic agent that reflects on intermediate outputs, revisits flawed steps, and guides iterative refinement. This integration of personality-driven agent design and structured collaboration improves both reasoning depth and flexibility. Empirical evaluations across three benchmarks demonstrate the strong performance of MAPS, with further analysis confirming its generalizability across different large language models and validating the benefits of multi-agent collaboration.

**arXiv ID:** 2503.16905
</details>

<details>
<summary><strong>From Questions to Queries: An AI-powered Multi-Agent Framework for Spatial Text-to-SQL</strong> - Ali Khosravi Kazazi, Zhenlong Li, M. Naser Lessani, Guido Cervone - [[pdf]](https://arxiv.org/pdf/2510.21045)</summary>

**Abstract:** The complexity of Structured Query Language (SQL) and the specialized nature of geospatial functions in tools like PostGIS present significant barriers to non-experts seeking to analyze spatial data. While Large Language Models (LLMs) offer promise for translating natural language into SQL (Text-to-SQL), single-agent approaches often struggle with the semantic and syntactic complexities of spatial queries. To address this, we propose a multi-agent framework designed to accurately translate natural language questions into spatial SQL queries. The framework integrates several innovative components, including a knowledge base with programmatic schema profiling and semantic enrichment, embeddings for context retrieval, and a collaborative multi-agent pipeline as its core. This pipeline comprises specialized agents for entity extraction, metadata retrieval, query logic formulation, SQL generation, and a review agent that performs programmatic and semantic validation of the generated SQL to ensure correctness (self-verification). We evaluate our system using both the non-spatial KaggleDBQA benchmark and a new, comprehensive SpatialQueryQA benchmark that includes diverse geometry types, predicates, and three levels of query complexity. On KaggleDBQA, the system achieved an overall accuracy of 81.2% (221 out of 272 questions) after the review agent's review and corrections. For spatial queries, the system achieved an overall accuracy of 87.7% (79 out of 90 questions), compared with 76.7% without the review agent. Beyond accuracy, results also show that in some instances the system generates queries that are more semantically aligned with user intent than those in the benchmarks. This work makes spatial analysis more accessible, and provides a robust, generalizable foundation for spatial Text-to-SQL systems, advancing the development of autonomous GIS.

**arXiv ID:** 2510.21045
</details>

<details>
<summary><strong>MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization</strong> - Jian Zhang, Zhangqi Wang, Haiping Zhu, Kangda Cheng, Kai He, Bo Li, Qika Lin, Jun Liu, Erik Cambria - [[pdf]](https://arxiv.org/pdf/2503.16874)</summary>

**Abstract:** Large language models (LLMs) typically operate in a question-answering paradigm, where the quality of the input prompt critically affects the response. Automated Prompt Optimization (APO) aims to overcome the cognitive biases of manually crafted prompts and explore a broader prompt design space. However, existing APO methods often suffer from rigid template structures and inefficient exploration in the prompt space. To this end, we propose a Multi-Agent Adaptive Reasoning with Socratic guidance framework (MARS) for APO. MARS consists of five complementary agents and formulates the optimization process as a Partially Observable Markov Decision Process (POMDP), enabling adaptive prompt refinement through explicit state modeling and interactive feedback. Specifically, a Planner agent generates flexible optimization trajectories, a Teacher-Critic-Student triad engages in Socratic-style dialogue to iteratively optimize the prompt based on pseudo-gradient signals in the text space, and a Target agent executes the prompt in downstream tasks to provide performance feedback. MARS integrates reasoning, feedback, and state transition into a unified hidden-state evolution process, improving both the effectiveness and interpretability of optimization. Extensive experiments on multiple datasets demonstrate that MARS outperforms existing APO methods in terms of optimization performance, search efficiency, and interpretability.

**arXiv ID:** 2503.16874
</details>

<details>
<summary><strong>Rainbow Delay Compensation: A Multi-Agent Reinforcement Learning Framework for Mitigating Delayed Observation</strong> - Songchen Fu, Siang Chen, Shaojing Zhao, Letian Bai, Ta Li, Yonghong Yan - [[pdf]](https://arxiv.org/pdf/2505.03586)</summary>

**Abstract:** In real-world multi-agent systems (MASs), observation delays are ubiquitous, preventing agents from making decisions based on the environment's true state. An individual agent's local observation typically comprises multiple components from other agents or dynamic entities within the environment. These discrete observation components with varying delay characteristics pose significant challenges for multi-agent reinforcement learning (MARL). In this paper, we first formulate the decentralized stochastic individual delay partially observable Markov decision process (DSID-POMDP) by extending the standard Dec-POMDP. We then propose the Rainbow Delay Compensation (RDC), a MARL training framework for addressing stochastic individual delays, along with recommended implementations for its constituent modules. We implement the DSID-POMDP's observation generation pattern using standard MARL benchmarks, including MPE and SMAC. Experiments demonstrate that baseline MARL methods suffer severe performance degradation under fixed and unfixed delays. The RDC-enhanced approach mitigates this issue, remarkably achieving ideal delay-free performance in certain delay scenarios while maintaining generalizability. Our work provides a novel perspective on multi-agent delayed observation problems and offers an effective solution framework. The source code is available at this https URL.

**arXiv ID:** 2505.03586
</details>

<details>
<summary><strong>Learning Efficient Communication Protocols for Multi-Agent Reinforcement Learning</strong> - Xinren Zhang, Jiadong Yu, Zixin Zhong - [[pdf]](https://arxiv.org/pdf/2511.09171)</summary>

**Abstract:** Multi-Agent Systems (MAS) have emerged as a powerful paradigm for modeling complex interactions among autonomous entities in distributed environments. In Multi-Agent Reinforcement Learning (MARL), communication enables coordination but can lead to inefficient information exchange, since agents may generate redundant or non-essential messages. While prior work has focused on boosting task performance with information exchange, the existing research lacks a thorough investigation of both the appropriate definition and the optimization of communication protocols (communication topology and message). To fill this gap, we introduce a generalized framework for learning multi-round communication protocols that are both effective and efficient. Within this framework, we propose three novel Communication Efficiency Metrics (CEMs) to guide and evaluate the learning process: the Information Entropy Efficiency Index (IEI) and Specialization Efficiency Index (SEI) for efficiency-augmented optimization, and the Topology Efficiency Index (TEI) for explicit evaluation. We integrate IEI and SEI as the adjusted loss functions to promote informative messaging and role specialization, while using TEI to quantify the trade-off between communication volume and task performance. Through comprehensive experiments, we demonstrate that our learned communication protocol can significantly enhance communication efficiency and achieves better cooperation performance with improved success rates.

**arXiv ID:** 2511.09171
</details>

<details>
<summary><strong>Modeling multi-agent motion dynamics in immersive rooms</strong> - Mincong, Huang, Stefan T. Radev - [[pdf]](https://arxiv.org/pdf/2511.08763)</summary>

**Abstract:** Immersive rooms are increasingly popular augmented reality systems that support multi-agent interactions within a virtual world. However, despite extensive content creation and technological developments, insights about perceptually-driven social dynamics, such as the complex movement patterns during virtual world navigation, remain largely underexplored. Computational models of motion dynamics can help us understand the underlying mechanism of human interaction in immersive rooms and develop applications that better support spatially distributed interaction. In this work, we propose a new agent-based model of emergent human motion dynamics. The model represents human agents as simple spatial geometries in the room that relocate and reorient themselves based on the salient virtual spatial objects they approach. Agent motion is modeled as an interactive process combining external diffusion-driven influences from the environment with internal self-propelling interactions among agents. Further, we leverage simulation-based inference (SBI) to show that the governing parameters of motion patterns can be estimated from simple observables. Our results indicate that the model successfully captures action-related agent properties but exposes local non-identifiability linked to environmental awareness. We argue that our simulation-based approach paves the way for creating adaptive, responsive immersive rooms -- spaces that adjust their interfaces and interactions based on human collective movement patterns and spatial attention.

**arXiv ID:** 2511.08763
</details>

<details>
<summary><strong>Low-cost Multi-agent Fleet for Acoustic Cooperative Localization Research</strong> - Nelson Durrant, Braden Meyers, Matthew McMurray, Clayton Smith, Brighton Anderson, Tristan Hodgins, Kalliyan Velasco, Joshua G. Mangelson - [[pdf]](https://arxiv.org/pdf/2511.08822)</summary>

**Abstract:** Real-world underwater testing for multi-agent autonomy presents substantial financial and engineering challenges. In this work, we introduce the Configurable Underwater Group of Autonomous Robots (CoUGARs) as a low-cost, configurable autonomous-underwater-vehicle (AUV) platform for multi-agent autonomy research. The base design costs less than $3,000 USD (as of May 2025) and is based on commercially-available and 3D-printed parts, enabling quick customization for various sensor payloads and configurations. Our current expanded model is equipped with a doppler velocity log (DVL) and ultra-short-baseline (USBL) acoustic array/transducer to support research on acoustic-based cooperative localization. State estimation, navigation, and acoustic communications software has been developed and deployed using a containerized software stack and is tightly integrated with the HoloOcean simulator. The system was tested both in simulation and via in-situ field trials in Utah lakes and reservoirs.

**arXiv ID:** 2511.08822
</details>

<details>
<summary><strong>Game Theory and Multi-Agent Reinforcement Learning for Zonal Ancillary Markets</strong> - Francesco Morri, Hélène Le Cadre, Pierre Gruet, Luce Brotcorne - [[pdf]](https://arxiv.org/pdf/2505.03288)</summary>

**Abstract:** We characterize zonal ancillary market coupling relying on noncooperative game theory. To that purpose, we formulate the ancillary market as a multi-leader single follower bilevel problem, that we subsequently cast as a generalized Nash game with side constraints and nonconvex feasibility sets. We determine conditions for equilibrium existence and show that the game has a generalized potential game structure. To compute market equilibrium, we rely on two exact approaches: an integrated optimization approach and Gauss-Seidel best-response, that we compare against multi-agent deep reinforcement learning. On real data from Germany and Austria, simulations indicate that multi-agent deep reinforcement learning achieves the smallest convergence rate but requires pretraining, while best-response is the slowest. On the economics side, multi-agent deep reinforcement learning results in smaller market costs compared to the exact methods, but at the cost of higher variability in the profit allocation among stakeholders. Further, stronger coupling between zones tends to reduce costs for larger zones.

**arXiv ID:** 2505.03288
</details>

<details>
<summary><strong>UniMM-V2X: MoE-Enhanced Multi-Level Fusion for End-to-End Cooperative Autonomous Driving</strong> - Ziyi Song, Chen Xia, Chenbing Wang, Haibao Yu, Sheng Zhou, Zhisheng Niu - [[pdf]](https://arxiv.org/pdf/2511.09013)</summary>

**Abstract:** Autonomous driving holds transformative potential but remains fundamentally constrained by the limited perception and isolated decision-making with standalone intelligence. While recent multi-agent approaches introduce cooperation, they often focus merely on perception-level tasks, overlooking the alignment with downstream planning and control, or fall short in leveraging the full capacity of the recent emerging end-to-end autonomous driving. In this paper, we present UniMM-V2X, a novel end-to-end multi-agent framework that enables hierarchical cooperation across perception, prediction, and planning. At the core of our framework is a multi-level fusion strategy that unifies perception and prediction cooperation, allowing agents to share queries and reason cooperatively for consistent and safe decision-making. To adapt to diverse downstream tasks and further enhance the quality of multi-level fusion, we incorporate a Mixture-of-Experts (MoE) architecture to dynamically enhance the BEV representations. We further extend MoE into the decoder to better capture diverse motion patterns. Extensive experiments on the DAIR-V2X dataset demonstrate our approach achieves state-of-the-art (SOTA) performance with a 39.7% improvement in perception accuracy, a 7.2% reduction in prediction error, and a 33.2% improvement in planning performance compared with UniV2X, showcasing the strength of our MoE-enhanced multi-level cooperative paradigm.

**arXiv ID:** 2511.09013
</details>

</details>

<details open>
<summary><h2>Other Agent Research (5 papers)</h2></summary>

<details>
<summary><strong>Lumine: An Open Recipe for Building Generalist Agents in 3D Open Worlds</strong> - Weihao Tan, Xiangyang Li, Yunhao Fang, Heyuan Yao, Shi Yan, Hao Luo, Tenglong Ao, Huihui Li, Hongbin Ren, Bairen Yi, Yujia Qin, Bo An, Libin Liu, Guang Shi - [[pdf]](https://arxiv.org/pdf/2511.08892)</summary>

**Abstract:** We introduce Lumine, the first open recipe for developing generalist agents capable of completing hours-long complex missions in real time within challenging 3D open-world environments. Lumine adopts a human-like interaction paradigm that unifies perception, reasoning, and action in an end-to-end manner, powered by a vision-language model. It processes raw pixels at 5 Hz to produce precise 30 Hz keyboard-mouse actions and adaptively invokes reasoning only when necessary. Trained in Genshin Impact, Lumine successfully completes the entire five-hour Mondstadt main storyline on par with human-level efficiency and follows natural language instructions to perform a broad spectrum of tasks in both 3D open-world exploration and 2D GUI manipulation across collection, combat, puzzle-solving, and NPC interaction. In addition to its in-domain performance, Lumine demonstrates strong zero-shot cross-game generalization. Without any fine-tuning, it accomplishes 100-minute missions in Wuthering Waves and the full five-hour first chapter of Honkai: Star Rail. These promising results highlight Lumine's effectiveness across distinct worlds and interaction dynamics, marking a concrete step toward generalist agents in open-ended environments.

**arXiv ID:** 2511.08892
</details>

<details>
<summary><strong>3D Guard-Layer: An Integrated Agentic AI Safety System for Edge Artificial Intelligence</strong> - Eren Kurshan, Yuan Xie, Paul Franzon - [[pdf]](https://arxiv.org/pdf/2511.08842)</summary>

**Abstract:** AI systems have found a wide range of real-world applications in recent years. The adoption of edge artificial intelligence, embedding AI directly into edge devices, is rapidly growing. Despite the implementation of guardrails and safety mechanisms, security vulnerabilities and challenges have become increasingly prevalent in this domain, posing a significant barrier to the practical deployment and safety of AI systems. This paper proposes an agentic AI safety architecture that leverages 3D to integrate a dedicated safety layer. It introduces an adaptive AI safety infrastructure capable of dynamically learning and mitigating attacks against the AI system. The system leverages the inherent advantages of co-location with the edge computing hardware to continuously monitor, detect and proactively mitigate threats to the AI system. The integration of local processing and learning capabilities enhances resilience against emerging network-based attacks while simultaneously improving system reliability, modularity, and performance, all with minimal cost and 3D integration overhead.

**arXiv ID:** 2511.08842
</details>

<details>
<summary><strong>Enabling Agents to Communicate Entirely in Latent Space</strong> - Zhuoyun Du, Runze Wang, Huiyu Bai, Zouying Cao, Xiaoyong Zhu, Bo Zheng, Wei Chen, Haochao Ying - [[pdf]](https://arxiv.org/pdf/2511.09149)</summary>

**Abstract:** While natural language is the de facto communication medium for LLM-based agents, it presents a fundamental constraint. The process of downsampling rich, internal latent states into discrete tokens inherently limits the depth and nuance of information that can be transmitted, thereby hindering collaborative problem-solving. Inspired by human mind-reading, we propose Interlat (Inter-agent Latent Space Communication), a paradigm that leverages the last hidden states of an LLM as a representation of its mind for direct transmission (termed latent communication). An additional compression process further compresses latent communication via entirely latent space reasoning. Experiments demonstrate that Interlat outperforms both fine-tuned chain-of-thought (CoT) prompting and single-agent baselines, promoting more exploratory behavior and enabling genuine utilization of latent information. Further compression not only substantially accelerates inference but also maintains competitive performance through an efficient information-preserving mechanism. We position this work as a feasibility study of entirely latent space inter-agent communication, and our results highlight its potential, offering valuable insights for future research.

**arXiv ID:** 2511.09149
</details>

<details>
<summary><strong>Target Tracking via LiDAR-RADAR Sensor Fusion for Autonomous Racing</strong> - Marcello Cellina, Matteo Corno, Sergio Matteo Savaresi - [[pdf]](https://arxiv.org/pdf/2505.20043)</summary>

**Abstract:** High Speed multi-vehicle Autonomous Racing will increase the safety and performance of road-going Autonomous Vehicles. Precise vehicle detection and dynamics estimation from a moving platform is a key requirement for planning and executing complex autonomous overtaking maneuvers. To address this requirement, we have developed a Latency-Aware EKF-based Multi Target Tracking algorithm fusing LiDAR and RADAR measurements. The algorithm explots the different sensor characteristics by explicitly integrating the Range Rate in the EKF Measurement Function, as well as a-priori knowledge of the racetrack during state prediction. It can handle Out-Of-Sequence Measurements via Reprocessing using a double State and Measurement Buffer, ensuring sensor delay compensation with no information loss. This algorithm has been implemented on Team PoliMOVE's autonomous racecar, and was proved experimentally by completing a number of fully autonomous overtaking maneuvers at speeds up to 275 km/h.

**arXiv ID:** 2505.20043
</details>

<details>
<summary><strong>Gaussian-Process-based Adaptive Tracking Control with Dynamic Active Learning for Autonomous Ground Vehicles</strong> - Kristóf Floch, Tamás Péni, Roland Tóth - [[pdf]](https://arxiv.org/pdf/2501.14672)</summary>

**Abstract:** This article proposes an active-learning-based adaptive trajectory tracking control method for autonomous ground vehicles to compensate for modeling errors and unmodeled dynamics. The nominal vehicle model is decoupled into lateral and longitudinal subsystems, which are augmented with online Gaussian Processes (GPs), using measurement data. The estimated mean functions of the GPs are used to construct a feedback compensator, which, together with an LPV state feedback controller designed for the nominal system, gives the adaptive control structure. To assist exploration of the dynamics, the paper proposes a new, dynamic active learning method to collect the most informative samples to accelerate the training process. To analyze the performance of the overall learning tool-chain provided controller, a novel iterative, counterexample-based algorithm is proposed for calculating the induced L2 gain between the reference trajectory and the tracking error. The analysis can be executed for a set of possible realizations of the to-be-controlled system, giving robust performance certificate of the learning method under variation of the vehicle dynamics. The efficiency of the proposed control approach is shown on a high-fidelity physics simulator and in real experiments using a 1/10 scale F1TENTH electric car.

**arXiv ID:** 2501.14672
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (27 papers)</h2></summary>

<details>
<summary><strong>Interpretable by Design: Query-Specific Neural Modules for Explainable Reinforcement Learning</strong> - Mehrdad Zakershahrak - [[pdf]](https://arxiv.org/pdf/2511.08749)</summary>

**Abstract:** Reinforcement learning has traditionally focused on a singular objective: learning policies that select actions to maximize reward. We challenge this paradigm by asking: what if we explicitly architected RL systems as inference engines that can answer diverse queries about their environment? In deterministic settings, trained agents implicitly encode rich knowledge about reachability, distances, values, and dynamics - yet current architectures are not designed to expose this information efficiently. We introduce Query Conditioned Deterministic Inference Networks (QDIN), a unified architecture that treats different types of queries (policy, reachability, paths, comparisons) as first-class citizens, with specialized neural modules optimized for each inference pattern. Our key empirical finding reveals a fundamental decoupling: inference accuracy can reach near-perfect levels (99% reachability IoU) even when control performance remains suboptimal (31% return), suggesting that the representations needed for accurate world knowledge differ from those required for optimal control. Experiments demonstrate that query specialized architectures outperform both unified models and post-hoc extraction methods, while maintaining competitive control performance. This work establishes a research agenda for RL systems designed from inception as queryable knowledge bases, with implications for interpretability, verification, and human-AI collaboration.

**arXiv ID:** 2511.08749
</details>

<details>
<summary><strong>UCO: A Multi-Turn Interactive Reinforcement Learning Method for Adaptive Teaching with Large Language Models</strong> - Shouang Wei, Min Zhang, Xin Lin, Bo Jiang, Kun Kuang, Zhongxiang Dai - [[pdf]](https://arxiv.org/pdf/2511.08873)</summary>

**Abstract:** Large language models (LLMs) are shifting from answer providers to intelligent tutors in educational settings, yet current supervised fine-tuning methods only learn surface teaching patterns without dynamic adaptation capabilities. Recent reinforcement learning approaches address this limitation but face two critical challenges. First, they evaluate teaching effectiveness solely based on whether students produce correct outputs, unable to distinguish whether students genuinely understand or echo teacher-provided answers during interaction. Second, they cannot perceive students' evolving cognitive states in real time through interactive dialogue, thus failing to adapt teaching strategies to match students' cognitive levels dynamically. We propose the Unidirectional Cognitive Optimization (UCO) method to address these challenges. UCO uses a multi-turn interactive reinforcement learning paradigm where the innovation lies in two synergistic reward functions: the Progress Reward captures students' cognitive advancement, evaluating whether students truly transition from confusion to comprehension, while the Scaffold Reward dynamically identifies each student's Zone of Proximal Development (ZPD), encouraging teachers to maintain productive teaching within this zone. We evaluate UCO by comparing it against 11 baseline models on BigMath and MathTutorBench benchmarks. Experimental results demonstrate that our UCO model outperforms all models of equivalent scale and achieves performance comparable to advanced closed-source models. The code and data are available at this https URL.

**arXiv ID:** 2511.08873
</details>

<details>
<summary><strong>Advancing Autonomous Emergency Response Systems: A Generative AI Perspective</strong> - Yousef Emami, Radha Reddy, Azadeh Pourkabirian, Miguel Gutierrez Gaitan - [[pdf]](https://arxiv.org/pdf/2511.09044)</summary>

**Abstract:** Autonomous Vehicles (AVs) are poised to revolutionize emergency services by enabling faster, safer, and more efficient responses. This transformation is driven by advances in Artificial Intelligence (AI), particularly Reinforcement Learning (RL), which allows AVs to navigate complex environments and make critical decisions in real time. However, conventional RL paradigms often suffer from poor sample efficiency and lack adaptability in dynamic emergency scenarios. This paper reviews next-generation AV optimization strategies to address these limitations. We analyze the shift from conventional RL to Diffusion Model (DM)-augmented RL, which enhances policy robustness through synthetic data generation, albeit with increased computational cost. Additionally, we explore the emerging paradigm of Large Language Model (LLM)-assisted In-Context Learning (ICL), which offers a lightweight and interpretable alternative by enabling rapid, on-the-fly adaptation without retraining. By reviewing the state of the art in AV intelligence, DM-augmented RL, and LLM-assisted ICL, this paper provides a critical framework for understanding the next generation of autonomous emergency response systems from a Generative AI perspective.

**arXiv ID:** 2511.09044
</details>

<details>
<summary><strong>OR-R1: Automating Modeling and Solving of Operations Research Optimization Problem via Test-Time Reinforcement Learning</strong> - Zezhen Ding, Zhen Tan, Jiheng Zhang, Tianlong Chen - [[pdf]](https://arxiv.org/pdf/2511.09092)</summary>

**Abstract:** Optimization modeling and solving are fundamental to the application of Operations Research (OR) in real-world decision making, yet the process of translating natural language problem descriptions into formal models and solver code remains highly expertise intensive. While recent advances in large language models (LLMs) have opened new opportunities for automation, the generalization ability and data efficiency of existing LLM-based methods are still limited, asmost require vast amounts of annotated or synthetic data, resulting in high costs and scalability barriers. In this work, we present OR-R1, a data-efficient training framework for automated optimization modeling and solving. OR-R1 first employs supervised fine-tuning (SFT) to help the model acquire the essential reasoning patterns for problem formulation and code generation from limited labeled data. In addition, it improves the capability and consistency through Test-Time Group Relative Policy Optimization (TGRPO). This two-stage design enables OR-R1 to leverage both scarce labeled and abundant unlabeled data for effective learning. Experiments show that OR-R1 achieves state-of-the-art performance with an average solving accuracy of $67.7\%$, using only $1/10$ the synthetic data required by prior methods such as ORLM, exceeding ORLM's solving accuracy by up to $4.2\%$. Remarkably, OR-R1 outperforms ORLM by over $2.4\%$ with just $100$ synthetic samples. Furthermore, TGRPO contributes an additional $3.1\%-6.4\%$ improvement in accuracy, significantly narrowing the gap between single-attempt (Pass@1) and multi-attempt (Pass@8) performance from $13\%$ to $7\%$. Extensive evaluations across diverse real-world benchmarks demonstrate that OR-R1 provides a robust, scalable, and cost-effective solution for automated OR optimization problem modeling and solving, lowering the expertise and data barriers for industrial OR applications.

**arXiv ID:** 2511.09092
</details>

<details>
<summary><strong>History-Aware Reasoning for GUI Agents</strong> - Ziwei Wang, Leyang Yang, Xiaoxuan Tang, Sheng Zhou, Dajun Chen, Wei Jiang, Yong Li - [[pdf]](https://arxiv.org/pdf/2511.09127)</summary>

**Abstract:** Advances in Multimodal Large Language Models have significantly enhanced Graphical User Interface (GUI) automation. Equipping GUI agents with reliable episodic reasoning capabilities is essential for bridging the gap between users' concise task descriptions and the complexities of real-world execution. Current methods integrate Reinforcement Learning (RL) with System-2 Chain-of-Thought, yielding notable gains in reasoning enhancement. For long-horizon GUI tasks, historical interactions connect each screen to the goal-oriented episode chain, and effectively leveraging these clues is crucial for the current decision. However, existing native GUI agents exhibit weak short-term memory in their explicit reasoning, interpreting the chained interactions as discrete screen understanding, i.e., unawareness of the historical interactions within the episode. This history-agnostic reasoning challenges their performance in GUI automation. To alleviate this weakness, we propose a History-Aware Reasoning (HAR) framework, which encourages an agent to reflect on its own errors and acquire episodic reasoning knowledge from them via tailored strategies that enhance short-term memory in long-horizon interaction. The framework mainly comprises constructing a reflective learning scenario, synthesizing tailored correction guidelines, and designing a hybrid RL reward function. Using the HAR framework, we develop a native end-to-end model, HAR-GUI-3B, which alters the inherent reasoning mode from history-agnostic to history-aware, equipping the GUI agent with stable short-term memory and reliable perception of screen details. Comprehensive evaluations across a range of GUI-related benchmarks demonstrate the effectiveness and generalization of our method.

**arXiv ID:** 2511.09127
</details>

<details>
<summary><strong>Perspectives on a Reliability Monitoring Framework for Agentic AI Systems</strong> - Niclas Flehmig, Mary Ann Lundteigen, Shen Yin - [[pdf]](https://arxiv.org/pdf/2511.09178)</summary>

**Abstract:** The implementation of agentic AI systems has the potential of providing more helpful AI systems in a variety of applications. These systems work autonomously towards a defined goal with reduced external control. Despite their potential, one of their flaws is the insufficient reliability which makes them especially unsuitable for high-risk domains such as healthcare or process industry. Unreliable systems pose a risk in terms of unexpected behavior during operation and mitigation techniques are needed. In this work, we derive the main reliability challenges of agentic AI systems during operation based on their characteristics. We draw the connection to traditional AI systems and formulate a fundamental reliability challenge during operation which is inherent to traditional and agentic AI systems. As our main contribution, we propose a two-layered reliability monitoring framework for agentic AI systems which consists of a out-of-distribution detection layer for novel inputs and AI transparency layer to reveal internal operations. This two-layered monitoring approach gives a human operator the decision support which is needed to decide whether an output is potential unreliable or not and intervene. This framework provides a foundation for developing mitigation techniques to reduce risk stemming from uncertain reliability during operation.

**arXiv ID:** 2511.09178
</details>

<details>
<summary><strong>Diffusion Policies with Value-Conditional Optimization for Offline Reinforcement Learning</strong> - Yunchang Ma, Tenglong Liu, Yixing Lan, Xin Yin, Changxin Zhang, Xinglong Zhang, Xin Xu - [[pdf]](https://arxiv.org/pdf/2511.08922)</summary>

**Abstract:** In offline reinforcement learning, value overestimation caused by out-of-distribution (OOD) actions significantly limits policy performance. Recently, diffusion models have been leveraged for their strong distribution-matching capabilities, enforcing conservatism through behavior policy constraints. However, existing methods often apply indiscriminate regularization to redundant actions in low-quality datasets, resulting in excessive conservatism and an imbalance between the expressiveness and efficiency of diffusion modeling. To address these issues, we propose DIffusion policies with Value-conditional Optimization (DIVO), a novel approach that leverages diffusion models to generate high-quality, broadly covered in-distribution state-action samples while facilitating efficient policy improvement. Specifically, DIVO introduces a binary-weighted mechanism that utilizes the advantage values of actions in the offline dataset to guide diffusion model training. This enables a more precise alignment with the dataset's distribution while selectively expanding the boundaries of high-advantage actions. During policy improvement, DIVO dynamically filters high-return-potential actions from the diffusion model, effectively guiding the learned policy toward better performance. This approach achieves a critical balance between conservatism and explorability in offline RL. We evaluate DIVO on the D4RL benchmark and compare it against state-of-the-art baselines. Empirical results demonstrate that DIVO achieves superior performance, delivering significant improvements in average returns across locomotion tasks and outperforming existing methods in the challenging AntMaze domain, where sparse rewards pose a major difficulty.

**arXiv ID:** 2511.08922
</details>

<details>
<summary><strong>Vendor-Aware Industrial Agents: RAG-Enhanced LLMs for Secure On-Premise PLC Code Generation</strong> - Joschka Kersting, Michael Rummel, Gesa Benndorf - [[pdf]](https://arxiv.org/pdf/2511.09122)</summary>

**Abstract:** Programmable Logic Controllers are operated by proprietary code dialects; this makes it challenging to train coding assistants. Current LLMs are trained on large code datasets and are capable of writing IEC 61131-3 compatible code out of the box, but they neither know specific function blocks, nor related project code. Moreover, companies like Mitsubishi Electric and their customers do not trust cloud providers. Hence, an own coding agent is the desired solution to cope with this. In this study, we present our work on a low-data domain coding assistant solution for industrial use. We show how we achieved high quality code generation without fine-tuning large models and by fine-tuning small local models for edge device usage. Our tool lets several AI models compete with each other, uses reasoning, corrects bugs automatically and checks code validity by compiling it directly in the chat interface. We support our approach with an extensive evaluation that comes with code compilation statistics and user ratings. We found that a Retrieval-Augmented Generation (RAG) supported coding assistant can work in low-data domains by using extensive prompt engineering and directed retrieval.

**arXiv ID:** 2511.09122
</details>

<details>
<summary><strong>Thinking Forward and Backward: Multi-Objective Reinforcement Learning for Retrieval-Augmented Reasoning</strong> - Wenda Wei, Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Lixin Su, Shuaiqiang Wang, Dawei Yin, Maarten de Rijke, Xueqi Cheng - [[pdf]](https://arxiv.org/pdf/2511.09109)</summary>

**Abstract:** Retrieval-augmented generation (RAG) has proven to be effective in mitigating hallucinations in large language models, yet its effectiveness remains limited in complex, multi-step reasoning scenarios. Recent efforts have incorporated search-based interactions into RAG, enabling iterative reasoning with real-time retrieval. Most approaches rely on outcome-based supervision, offering no explicit guidance for intermediate steps. This often leads to reward hacking and degraded response quality. We propose Bi-RAR, a novel retrieval-augmented reasoning framework that evaluates each intermediate step jointly in both forward and backward directions. To assess the information completeness of each step, we introduce a bidirectional information distance grounded in Kolmogorov complexity, approximated via language model generation probabilities. This quantification measures both how far the current reasoning is from the answer and how well it addresses the question. To optimize reasoning under these bidirectional signals, we adopt a multi-objective reinforcement learning framework with a cascading reward structure that emphasizes early trajectory alignment. Empirical results on seven question answering benchmarks demonstrate that Bi-RAR surpasses previous methods and enables efficient interaction and reasoning with the search engine during training and inference.

**arXiv ID:** 2511.09109
</details>

<details>
<summary><strong>AdaCuRL: Adaptive Curriculum Reinforcement Learning with Invalid Sample Mitigation and Historical Revisiting</strong> - Renda Li, Hailang Huang, Fei Wei, Feng Xiong, Yong Wang, Xiangxiang Chu - [[pdf]](https://arxiv.org/pdf/2511.09478)</summary>

**Abstract:** Reinforcement learning (RL) has demonstrated considerable potential for enhancing reasoning in large language models (LLMs). However, existing methods suffer from Gradient Starvation and Policy Degradation when training directly on samples with mixed difficulty. To mitigate this, prior approaches leverage Chain-of-Thought (CoT) data, but the construction of high-quality CoT annotations remains labor-intensive. Alternatively, curriculum learning strategies have been explored but frequently encounter challenges, such as difficulty mismatch, reliance on manual curriculum design, and catastrophic forgetting. To address these issues, we propose AdaCuRL, a Adaptive Curriculum Reinforcement Learning framework that integrates coarse-to-fine difficulty estimation with adaptive curriculum scheduling. This approach dynamically aligns data difficulty with model capability and incorporates a data revisitation mechanism to mitigate catastrophic forgetting. Furthermore, AdaCuRL employs adaptive reference and sparse KL strategies to prevent Policy Degradation. Extensive experiments across diverse reasoning benchmarks demonstrate that AdaCuRL consistently achieves significant performance improvements on both LLMs and MLLMs.

**arXiv ID:** 2511.09478
</details>

<details>
<summary><strong>Digital Co-Founders: Transforming Imagination into Viable Solo Business via Agentic AI</strong> - Farhad Rezazadeh, Pegah Bonehgazy - [[pdf]](https://arxiv.org/pdf/2511.09533)</summary>

**Abstract:** This paper investigates how individual entrepreneurs can turn creative ideas into successful solo businesses in an era increasingly shaped by Artificial Intelligence (AI) agents. It highlights the key steps that connect personal vision, structured experimentation, and lasting value creation, and shows how AI agents can act as digital co-founders throughout this journey. Building on research in entrepreneurship, creativity, and innovation, we present a framework with three key stages: (1) Imagination shaping, where vague goals become clear value propositions, supported by AI agents that help with market scanning, idea refinement, and rapid concept generation; (2) Reality testing, where these ideas are tested through low-cost experiments, structured feedback loops, and efficient execution, with AI agents automating tasks such as prototyping, content creation, customer interaction, and data analysis; and (3) Reality scaling, where successful ideas are transformed into repeatable processes, scalable market strategies, and long-term business models, increasingly operated and optimized by autonomous or semi-autonomous AI workflows. We focus on the specific context of solopreneurship, characterized by limited human resources, complete accountability for decision-making, and a strong association between the founder's identity and the business. The framework clearly identifies key enabling factors such as mental adaptability, effective planning, and successful human-AI collaboration within digital ecosystems. It also thoughtfully addresses ongoing challenges, like uncertainty and cognitive overload, which are heightened by our constant connectivity.

**arXiv ID:** 2511.09533
</details>

<details>
<summary><strong>Pushdown Reward Machines for Reinforcement Learning</strong> - Giovanni Varricchione, Toryn Q. Klassen, Natasha Alechina, Mehdi Dastani, Brian Logan, Sheila A. McIlraith - [[pdf]](https://arxiv.org/pdf/2508.06894)</summary>

**Abstract:** Reward machines (RMs) are automata structures that encode (non-Markovian) reward functions for reinforcement learning (RL). RMs can reward any behaviour representable in regular languages and, when paired with RL algorithms that exploit RM structure, have been shown to significantly improve sample efficiency in many domains. In this work, we present pushdown reward machines (pdRMs), an extension of reward machines based on deterministic pushdown automata. pdRMs can recognise and reward temporally extended behaviours representable in deterministic context-free languages, making them more expressive than reward machines. We introduce two variants of pdRM-based policies, one which has access to the entire stack of the pdRM, and one which can only access the top $k$ symbols (for a given constant $k$) of the stack. We propose a procedure to check when the two kinds of policies (for a given environment, pdRM, and constant $k$) achieve the same optimal state values. We then provide theoretical results establishing the expressive power of pdRMs, and space complexity results for the proposed learning problems. Lastly, we propose an approach for off-policy RL algorithms that exploits counterfactual experiences with pdRMs. We conclude by providing experimental results showing how agents can be trained to perform tasks representable in deterministic context-free languages using pdRMs.

**arXiv ID:** 2508.06894
</details>

<details>
<summary><strong>Simpliflow: A Lightweight Open-Source Framework for Rapid Creation and Deployment of Generative Agentic AI Workflows</strong> - Deven Panchal - [[pdf]](https://arxiv.org/pdf/2510.10675)</summary>

**Abstract:** Generative Agentic AI systems are emerging as a powerful paradigm for automating complex, multi-step tasks. However, many existing frameworks for building these systems introduce significant complexity, a steep learning curve, and substantial boilerplate code, hindering rapid prototyping and deployment. This paper introduces simpliflow, a lightweight, open-source Python framework designed to address these challenges. simpliflow enables the rapid development and orchestration of linear, deterministic agentic workflows through a declarative, JSON-based configuration. Its modular architecture decouples agent management, workflow execution, and post-processing, promoting ease of use and extensibility. By integrating with LiteLLM, it supports over 100 Large Language Models (LLMs) out-of-the-box. We present the architecture, operational flow, and core features of simpliflow, demonstrating its utility through diverse use cases ranging from software development simulation to real-time system interaction. A comparative analysis with prominent frameworks like LangChain and AutoGen highlights simpliflow's unique position as a tool optimized for simplicity, control, and speed in deterministic workflow environments.

**arXiv ID:** 2510.10675
</details>

<details>
<summary><strong>A Simple and Effective Reinforcement Learning Method for Text-to-Image Diffusion Fine-tuning</strong> - Shashank Gupta, Chaitanya Ahuja, Tsung-Yu Lin, Sreya Dutta Roy, Harrie Oosterhuis, Maarten de Rijke, Satya Narayan Shukla - [[pdf]](https://arxiv.org/pdf/2503.00897)</summary>

**Abstract:** Reinforcement learning (RL)-based fine-tuning has emerged as a powerful approach for aligning diffusion models with black-box objectives. Proximal policy optimization (PPO) is the most popular choice of method for policy optimization. While effective in terms of performance, PPO is highly sensitive to hyper-parameters and involves substantial computational overhead. REINFORCE, on the other hand, mitigates some computational complexities such as high memory overhead and sensitive hyper-parameter tuning, but has suboptimal performance due to high-variance and sample inefficiency. While the variance of the REINFORCE can be reduced by sampling multiple actions per input prompt and using a baseline correction term, it still suffers from sample inefficiency. To address these challenges, we systematically analyze the efficiency-effectiveness trade-off between REINFORCE and PPO, and propose leave-one-out PPO (LOOP), a novel RL for diffusion fine-tuning method. LOOP combines variance reduction techniques from REINFORCE, such as sampling multiple actions per input prompt and a baseline correction term, with the robustness and sample efficiency of PPO via clipping and importance sampling. Our results demonstrate that LOOP effectively improves diffusion models on various black-box objectives, and achieves a better balance between computational efficiency and performance.

**arXiv ID:** 2503.00897
</details>

<details>
<summary><strong>Teaching Large Language Models to Maintain Contextual Faithfulness via Synthetic Tasks and Reinforcement Learning</strong> - Shuzheng Si, Haozhe Zhao, Cheng Gao, Yuzhuo Bai, Zhitong Wang, Bofei Gao, Kangyang Luo, Wenhao Li, Yufei Huang, Gang Chen, Fanchao Qi, Minjia Zhang, Baobao Chang, Maosong Sun - [[pdf]](https://arxiv.org/pdf/2505.16483)</summary>

**Abstract:** Teaching large language models (LLMs) to be faithful in the provided context is crucial for building reliable information-seeking systems. Therefore, we propose a systematic framework, CANOE, to reduce faithfulness hallucinations of LLMs across different downstream tasks without human annotations. Specifically, we first synthesize short-form question-answering (QA) data with four diverse tasks to construct high-quality and easily verifiable training data without human annotation. Also, we propose Dual-GRPO, a rule-based reinforcement learning method that includes three tailored rule-based rewards derived from synthesized short-form QA data, while simultaneously optimizing both short-form and long-form response generation. Notably, Dual-GRPO eliminates the need to manually label preference data to train reward models and avoids over-optimizing short-form generation when relying only on the synthesized short-form QA data. Experimental results show that CANOE greatly improves the faithfulness of LLMs across 11 different tasks, even outperforming the most advanced LLMs, e.g., GPT-4o and OpenAI o1.

**arXiv ID:** 2505.16483
</details>

<details>
<summary><strong>TaoSR-AGRL: Adaptive Guided Reinforcement Learning Framework for E-commerce Search Relevance</strong> - Jianhui Yang, Yiming Jin, Pengkun Jiao, Chenhe Dong, Zerui Huang, Shaowei Yao, Xiaojiang Zhou, Dan Ou, Haihong Tang - [[pdf]](https://arxiv.org/pdf/2510.08048)</summary>

**Abstract:** Query-product relevance prediction is fundamental to e-commerce search and has become even more critical in the era of AI-powered shopping, where semantic understanding and complex reasoning directly shape the user experience and business conversion. Large Language Models (LLMs) enable generative, reasoning-based approaches, typically aligned via supervised fine-tuning (SFT) or preference optimization methods like Direct Preference Optimization (DPO). However, the increasing complexity of business rules and user queries exposes the inability of existing methods to endow models with robust reasoning capacity for long-tail and challenging cases. Efforts to address this via reinforcement learning strategies like Group Relative Policy Optimization (GRPO) often suffer from sparse terminal rewards, offering insufficient guidance for multi-step reasoning and slowing convergence. To address these challenges, we propose TaoSR-AGRL, an Adaptive Guided Reinforcement Learning framework for LLM-based relevance prediction in Taobao Search Relevance. TaoSR-AGRL introduces two key innovations: (1) Rule-aware Reward Shaping, which decomposes the final relevance judgment into dense, structured rewards aligned with domain-specific relevance criteria; and (2) Adaptive Guided Replay, which identifies low-accuracy rollouts during training and injects targeted ground-truth guidance to steer the policy away from stagnant, rule-violating reasoning patterns toward compliant trajectories. TaoSR-AGRL was evaluated on large-scale real-world datasets and through online side-by-side human evaluations on Taobao Search. It consistently outperforms DPO and standard GRPO baselines in offline experiments, improving relevance accuracy, rule adherence, and training stability. The model trained with TaoSR-AGRL has been successfully deployed in the main search scenario on Taobao, serving hundreds of millions of users.

**arXiv ID:** 2510.08048
</details>

<details>
<summary><strong>Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh for Empowering Access to Justice</strong> - Azmine Toushik Wasi, Wahid Faisal, Mst Rafia Islam - [[pdf]](https://arxiv.org/pdf/2511.08605)</summary>

**Abstract:** Bangladesh's low-income population faces major barriers to affordable legal advice due to complex legal language, procedural opacity, and high costs. Existing AI legal assistants lack Bengali-language support and jurisdiction-specific adaptation, limiting their effectiveness. To address this, we developed Mina, a multilingual LLM-based legal assistant tailored for the Bangladeshi context. It employs multilingual embeddings and a RAG-based chain-of-tools framework for retrieval, reasoning, translation, and document generation, delivering context-aware legal drafts, citations, and plain-language explanations via an interactive chat interface. Evaluated by law faculty from leading Bangladeshi universities across all stages of the 2022 and 2023 Bangladesh Bar Council Exams, Mina scored 75-80% in Preliminary MCQs, Written, and simulated Viva Voce exams, matching or surpassing average human performance and demonstrating clarity, contextual understanding, and sound legal reasoning. These results confirm its potential as a low-cost, multilingual AI assistant that automates key legal tasks and scales access to justice, offering a real-world case study on building domain-specific, low-resource systems and addressing challenges of multilingual adaptation, efficiency, and sustainable public-service AI deployment.

**arXiv ID:** 2511.08605
</details>

<details>
<summary><strong>QOC DAO - Stepwise Development Towards an AI Driven Decentralized Autonomous Organization</strong> - Marc Jansen, Christophe Verdot - [[pdf]](https://arxiv.org/pdf/2511.08641)</summary>

**Abstract:** This paper introduces a structured approach to improving decision making in Decentralized Autonomous Organizations (DAO) through the integration of the Question-Option-Criteria (QOC) model and AI agents. We outline a stepwise governance framework that evolves from human led evaluations to fully autonomous, AI-driven processes. By decomposing decisions into weighted, criterion based evaluations, the QOC model enhances transparency, fairness, and explainability in DAO voting. We demonstrate how large language models (LLMs) and stakeholder aligned AI agents can support or automate evaluations, while statistical safeguards help detect manipulation. The proposed framework lays the foundation for scalable and trustworthy governance in the Web3 ecosystem.

**arXiv ID:** 2511.08641
</details>

<details>
<summary><strong>Steering Noncooperative Games Through Conjecture Design</strong> - Francesco Morri, Hélène Le Cadre, David Salas, Didier Aussel - [[pdf]](https://arxiv.org/pdf/2511.09435)</summary>

**Abstract:** In dynamic noncooperative games, each player makes conjectures about other players' reactions before choosing a strategy. However, resulting equilibria may be multiple and do not always lead to desirable outcomes. These issues are typically addressed separately, for example, through opponent modelling and incentive design. Drawing inspiration from conjectural variations games, we propose an incentive design framework in which a coordinator first computes an equilibrium by optimizing a predefined objective function, then communicates this equilibrium as a target for the players to reach. In a centralized setting, the coordinator also optimizes the conjectures to steer the players towards the target. In decentralized settings, players independently compute conjectures and update their strategies based on individual targets. We provide a guarantee of equilibrium existence in both cases. This framework uses conjectures not only to guide the system towards desirable outcomes but also to decouple the game into independent optimization problems, enabling efficient computation and parallelization in large-scale settings. We illustrate our theoretical results on classical representative noncooperative games, demonstrating its application potential.

**arXiv ID:** 2511.09435
</details>

<details>
<summary><strong>Stabilizing Reinforcement Learning for Honesty Alignment in Language Models on Deductive Reasoning</strong> - Jiarui Liu, Kaustubh Dhole, Yingheng Wang, Haoyang Wen, Sarah Zhang, Haitao Mao, Gaotang Li, Neeraj Varshney, Jingguo Liu, Xiaoman Pan - [[pdf]](https://arxiv.org/pdf/2511.09222)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has recently emerged as a promising framework for aligning language models with complex reasoning objectives. However, most existing methods optimize only for final task outcomes, leaving models vulnerable to collapse when negative rewards dominate early training. This challenge is especially pronounced in honesty alignment, where models must not only solve answerable queries but also identify when conclusions cannot be drawn from the given premises. Deductive reasoning provides an ideal testbed because it isolates reasoning capability from reliance on external factual knowledge. To investigate honesty alignment, we curate two multi-step deductive reasoning datasets from graph structures, one for linear algebra and one for logical inference, and introduce unanswerable cases by randomly perturbing an edge in half of the instances. We find that GRPO, with or without supervised fine tuning initialization, struggles on these tasks. Through extensive experiments across three models, we evaluate stabilization strategies and show that curriculum learning provides some benefit but requires carefully designed in distribution datasets with controllable difficulty. To address these limitations, we propose Anchor, a reinforcement learning method that injects ground truth trajectories into rollouts, preventing early training collapse. Our results demonstrate that this method stabilizes learning and significantly improves the overall reasoning performance, underscoring the importance of training dynamics for enabling reliable deductive reasoning in aligned language models.

**arXiv ID:** 2511.09222
</details>

<details>
<summary><strong>Token Hidden Reward: Steering Exploration-Exploitation in Group Relative Deep Reinforcement Learning</strong> - Wenlong Deng, Yi Ren, Yushu Li, Boying Gong, Danica J. Sutherland, Xiaoxiao Li, Christos Thrampoulidis - [[pdf]](https://arxiv.org/pdf/2510.03669)</summary>

**Abstract:** Reinforcement learning with verifiable rewards has significantly advanced the reasoning capabilities of large language models, yet how to explicitly steer training toward exploration or exploitation remains an open problem. We introduce Token Hidden Reward (THR), a token-level metric that quantifies each token's influence on the likelihood of correct responses under Group Relative Policy Optimization (GRPO). We find that training dynamics are dominated by a small subset of tokens with high absolute THR values. Most interestingly, tokens with positive THR strengthen confidence in correct outputs, thus favoring exploitation, while tokens with negative THR preserve probability mass for alternative outputs, enabling exploration. This insight suggests a natural intervention: a THR-guided reweighting algorithm that modulates GRPO's learning signals to explicitly bias training toward exploitation or exploration. We validate the efficacy of this algorithm on diverse math reasoning benchmarks. By amplifying tokens with positive THR value and weakening negative ones, our algorithm improves greedy-decoding accuracy, favoring exploitation. The reverse strategy yields consistent gains in Pass@K accuracy, favoring exploration. We further demonstrate that our algorithm integrates seamlessly with other RL objectives such as GSPO and generalizes across architectures including Llama. These findings establish THR as a principled and fine-grained mechanism for dynamically controlling exploration and exploitation in RL-tuned LLMs, providing new tools for targeted fine-tuning in reasoning-intensive applications.

**arXiv ID:** 2510.03669
</details>

<details>
<summary><strong>Towards a Generalisable Cyber Defence Agent for Real-World Computer Networks</strong> - Tim Dudman, Martyn Bull - [[pdf]](https://arxiv.org/pdf/2511.09114)</summary>

**Abstract:** Recent advances in deep reinforcement learning for autonomous cyber defence have resulted in agents that can successfully defend simulated computer networks against cyber-attacks. However, many of these agents would need retraining to defend networks with differing topology or size, making them poorly suited to real-world networks where topology and size can vary over time. In this research we introduce a novel set of Topological Extensions for Reinforcement Learning Agents (TERLA) that provide generalisability for the defence of networks with differing topology and size, without the need for retraining. Our approach involves the use of heterogeneous graph neural network layers to produce a fixed-size latent embedding representing the observed network state. This representation learning stage is coupled with a reduced, fixed-size, semantically meaningful and interpretable action space. We apply TERLA to a standard deep reinforcement learning Proximal Policy Optimisation (PPO) agent model, and to reduce the sim-to-real gap, conduct our research using Cyber Autonomy Gym for Experimentation (CAGE) Challenge 4. This Cyber Operations Research Gym environment has many of the features of a real-world network, such as realistic Intrusion Detection System (IDS) events and multiple agents defending network segments of differing topology and size. TERLA agents retain the defensive performance of vanilla PPO agents whilst showing improved action efficiency. Generalisability has been demonstrated by showing that all TERLA agents have the same network-agnostic neural network architecture, and by deploying a single TERLA agent multiple times to defend network segments with differing topology and size, showing improved defensive performance and efficiency.

**arXiv ID:** 2511.09114
</details>

<details>
<summary><strong>Planning in Branch-and-Bound: Model-Based Reinforcement Learning for Exact Combinatorial Optimization</strong> - Paul Strang, Zacharie Alès, Côme Bissuel, Safia Kedad-Sidhoum, Emmanuel Rachelson - [[pdf]](https://arxiv.org/pdf/2511.09219)</summary>

**Abstract:** Mixed-Integer Linear Programming (MILP) lies at the core of many real-world combinatorial optimization (CO) problems, traditionally solved by branch-and-bound (B&B). A key driver influencing B&B solvers efficiency is the variable selection heuristic that guides branching decisions. Looking to move beyond static, hand-crafted heuristics, recent work has explored adapting traditional reinforcement learning (RL) algorithms to the B&B setting, aiming to learn branching strategies tailored to specific MILP distributions. In parallel, RL agents have achieved remarkable success in board games, a very specific type of combinatorial problems, by leveraging environment simulators to plan via Monte Carlo Tree Search (MCTS). Building on these developments, we introduce Plan-and-Branch-and-Bound (PlanB&B), a model-based reinforcement learning (MBRL) agent that leverages a learned internal model of the B&B dynamics to discover improved branching strategies. Computational experiments empirically validate our approach, with our MBRL branching agent outperforming previous state-of-the-art RL methods across four standard MILP benchmarks.

**arXiv ID:** 2511.09219
</details>

<details>
<summary><strong>Distributionally Robust Self Paced Curriculum Reinforcement Learning</strong> - Anirudh Satheesh, Keenan Powell, Vaneet Aggarwal - [[pdf]](https://arxiv.org/pdf/2511.05694)</summary>

**Abstract:** A central challenge in reinforcement learning is that policies trained in controlled environments often fail under distribution shifts at deployment into real-world environments. Distributionally Robust Reinforcement Learning (DRRL) addresses this by optimizing for worst-case performance within an uncertainty set defined by a robustness budget $\epsilon$. However, fixing $\epsilon$ results in a tradeoff between performance and robustness: small values yield high nominal performance but weak robustness, while large values can result in instability and overly conservative policies. We propose Distributionally Robust Self-Paced Curriculum Reinforcement Learning (DR-SPCRL), a method that overcomes this limitation by treating $\epsilon$ as a continuous curriculum. DR-SPCRL adaptively schedules the robustness budget according to the agent's progress, enabling a balance between nominal and robust performance. Empirical results across multiple environments demonstrate that DR-SPCRL not only stabilizes training but also achieves a superior robustness-performance trade-off, yielding an average 11.8\% increase in episodic return under varying perturbations compared to fixed or heuristic scheduling strategies, and achieving approximately 1.9$\times$ the performance of the corresponding nominal RL algorithms.

**arXiv ID:** 2511.05694
</details>

<details>
<summary><strong>Expand Your SCOPE: Semantic Cognition over Potential-Based Exploration for Embodied Visual Navigation</strong> - Ningnan Wang, Weihuang Chen, Liming Chen, Haoxuan Ji, Zhongyu Guo, Xuchong Zhang, Hongbin Sun - [[pdf]](https://arxiv.org/pdf/2511.08935)</summary>

**Abstract:** Embodied visual navigation remains a challenging task, as agents must explore unknown environments with limited knowledge. Existing zero-shot studies have shown that incorporating memory mechanisms to support goal-directed behavior can improve long-horizon planning performance. However, they overlook visual frontier boundaries, which fundamentally dictate future trajectories and observations, and fall short of inferring the relationship between partial visual observations and navigation goals. In this paper, we propose Semantic Cognition Over Potential-based Exploration (SCOPE), a zero-shot framework that explicitly leverages frontier information to drive potential-based exploration, enabling more informed and goal-relevant decisions. SCOPE estimates exploration potential with a Vision-Language Model and organizes it into a spatio-temporal potential graph, capturing boundary dynamics to support long-horizon planning. In addition, SCOPE incorporates a self-reconsideration mechanism that revisits and refines prior decisions, enhancing reliability and reducing overconfident errors. Experimental results on two diverse embodied navigation tasks show that SCOPE outperforms state-of-the-art baselines by 4.6\% in accuracy. Further analysis demonstrates that its core components lead to improved calibration, stronger generalization, and higher decision quality.

**arXiv ID:** 2511.08935
</details>

<details>
<summary><strong>D-AWSIM: Distributed Autonomous Driving Simulator for Dynamic Map Generation Framework</strong> - Shunsuke Ito, Chaoran Zhao, Ryo Okamura, Takuya Azumi - [[pdf]](https://arxiv.org/pdf/2511.09080)</summary>

**Abstract:** Autonomous driving systems have achieved significant advances, and full autonomy within defined operational design domains near practical deployment. Expanding these domains requires addressing safety assurance under diverse conditions. Information sharing through vehicle-to-vehicle and vehicle-to-infrastructure communication, enabled by a Dynamic Map platform built from vehicle and roadside sensor data, offers a promising solution. Real-world experiments with numerous infrastructure sensors incur high costs and regulatory challenges. Conventional single-host simulators lack the capacity for large-scale urban traffic scenarios. This paper proposes D-AWSIM, a distributed simulator that partitions its workload across multiple machines to support the simulation of extensive sensor deployment and dense traffic environments. A Dynamic Map generation framework on D-AWSIM enables researchers to explore information-sharing strategies without relying on physical testbeds. The evaluation shows that D-AWSIM increases throughput for vehicle count and LiDAR sensor processing substantially compared to a single-machine setup. Integration with Autoware demonstrates applicability for autonomous driving research.

**arXiv ID:** 2511.09080
</details>

<details>
<summary><strong>A multimodal AI agent for clinical decision support in ophthalmology</strong> - Danli Shi, Xiaolan Chen, Bingjie Yan, Weiyi Zhang, Pusheng Xu, Jiancheng Yang, Ruoyu Chen, Siyu Huang, Bowen Liu, Xinyuan Wu, Meng Xie, Ziyu Gao, Yue Wu, Senlin Lin, Kai Jin, Xia Gong, Yih Chung Tham, Xiujuan Zhang, Li Dong, Yuzhou Zhang, Jason Yam, Guangming Jin, Xiaohu Ding, Haidong Zou, Yalin Zheng, Zongyuan Ge, Mingguang He - [[pdf]](https://arxiv.org/pdf/2511.09394)</summary>

**Abstract:** Artificial intelligence has shown promise in medical imaging, yet most existing systems lack flexibility, interpretability, and adaptability - challenges especially pronounced in ophthalmology, where diverse imaging modalities are essential. We present EyeAgent, the first agentic AI framework for comprehensive and interpretable clinical decision support in ophthalmology. Using a large language model (DeepSeek-V3) as its central reasoning engine, EyeAgent interprets user queries and dynamically orchestrates 53 validated ophthalmic tools across 23 imaging modalities for diverse tasks including classification, segmentation, detection, image/report generation, and quantitative analysis. Stepwise ablation analysis demonstrated a progressive improvement in diagnostic accuracy, rising from a baseline of 69.71% (using only 5 general tools) to 80.79% when the full suite of 53 specialized tools was integrated. In an expert rating study on 200 real-world clinical cases, EyeAgent achieved 93.7% tool selection accuracy and received expert ratings of more than 88% across accuracy, completeness, safety, reasoning, and interpretability. In human-AI collaboration, EyeAgent matched or exceeded the performance of senior ophthalmologists and, when used as an assistant, improved overall diagnostic accuracy by 18.51% and report quality scores by 19%, with the greatest benefit observed among junior ophthalmologists. These findings establish EyeAgent as a scalable and trustworthy AI framework for ophthalmology and provide a blueprint for modular, multimodal, and clinically aligned next-generation AI systems.

**arXiv ID:** 2511.09394
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
