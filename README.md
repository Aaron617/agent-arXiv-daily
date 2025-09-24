# Agent arXiv Daily

**Last Updated:** 2025-09-24 02:38:14

**Total Papers:** 63

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
<summary><strong>Online Process Reward Leanring for Agentic Reinforcement Learning</strong> - Xiaoqian Liu, Ke Wang, Yuchuan Wu, Fei Huang, Yongbin Li, Junge Zhang, Jianbin Jiao - [[pdf]](https://arxiv.org/pdf/2509.19199)</summary>

**Abstract:** Large language models (LLMs) are increasingly trained with reinforcement learning (RL) as autonomous agents that reason and act over long horizons in interactive environments.
However, sparse and sometimes unverifiable rewards make temporal credit assignment extremely challenging.
Recent work attempts to integrate process supervision into agent learning but suffers from biased annotation, reward hacking, high-variance from overly fine-grained signals or failtures when state overlap is rare.
We therefore introduce Online Process Reward Learning (OPRL), a general credit-assignment strategy for agentic RL that integrates seamlessly with standard on-policy algorithms without relying on additional rollouts or explicit step labels.
In OPRL, we optimize an implicit process reward model (PRM) alternately with the agent's policy to transform trajectory preferences into implicit step rewards through a trajectory-based DPO objective.
These step rewards are then used to compute step-level advantages, which are combined with episode-level advantages from outcome rewards for policy update, creating a self-reinforcing loop.
Theoretical findings guarantee that the learned step rewards are consistent with trajectory preferences and act as potential-based shaping rewards, providing bounded gradients to stabilize training.
Empirically, we evaluate OPRL on three distinct agent benmarks, including WebShop and VisualSokoban, as well as open-ended social interactions with unverfiable rewards in SOTOPIA.
Crucially, OPRL shows superior performance over frontier LLMs and strong RL baselines across domains, achieving state-of-the-art results with higher sample-efficiency and lower variance during training.
Further analysis also demonstrates the efficient exploration by OPRL using fewer actions, underscoring its potential for agentic learning in real-world scenarios.

**arXiv ID:** 2509.19199
</details>

<details>
<summary><strong>Campus AI vs. Commercial AI: How Customizations Shape Trust and Usage of LLM as-a-Service Chatbots</strong> - Leon Hannig, Annika Bush, Meltem Aksoy, Tim Trappen, Steffen Becker, Greta Ontrup - [[pdf]](https://arxiv.org/pdf/2509.15826)</summary>

**Abstract:** As the use of LLM chatbots by students and researchers becomes more prevalent, universities are pressed to develop AI strategies. One strategy that many universities pursue is to customize pre-trained LLM as-a-service (LLMaaS). While most studies on LLMaaS chatbots prioritize technical adaptations, we focus on psychological effects of user-salient customizations, such as interface changes. We assume that such customizations influence users' perception of the system and are therefore important in guiding safe and appropriate use. In a field study, we examine how students and employees (N = 526) at a German university perceive and use their institution's customized LLMaaS chatbot compared to ChatGPT. Participants using both systems (n = 116) reported greater trust, higher perceived privacy and less experienced hallucinations with their university's customized LLMaaS chatbot in contrast to ChatGPT. We discuss theoretical implications for research on calibrated trust, and offer guidance on the design and deployment of LLMaaS chatbots.

**arXiv ID:** 2509.15826
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>OpenLens AI: Fully Autonomous Research Agent for Health Infomatics</strong> - Yuxiao Cheng, Jinli Suo - [[pdf]](https://arxiv.org/pdf/2509.14778)</summary>

**Abstract:** Health informatics research is characterized by diverse data modalities, rapid knowledge expansion, and the need to integrate insights across biomedical science, data analytics, and clinical practice. These characteristics make it particularly well-suited for agent-based approaches that can automate knowledge exploration, manage complex workflows, and generate clinically meaningful outputs. Recent progress in large language model (LLM)-based agents has demonstrated promising capabilities in literature synthesis, data analysis, and even end-to-end research execution. However, existing systems remain limited for health informatics because they lack mechanisms to interpret medical visualizations and often overlook domain-specific quality requirements. To address these gaps, we introduce OpenLens AI, a fully automated framework tailored to health informatics. OpenLens AI integrates specialized agents for literature review, data analysis, code generation, and manuscript preparation, enhanced by vision-language feedback for medical visualization and quality control for reproducibility. The framework automates the entire research pipeline, producing publication-ready LaTeX manuscripts with transparent and traceable workflows, thereby offering a domain-adapted solution for advancing health informatics research.

**arXiv ID:** 2509.14778
</details>

<details>
<summary><strong>EMMA: End-to-End Multimodal Model for Autonomous Driving</strong> - Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, Yin Zhou, James Guo, Dragomir Anguelov, Mingxing Tan - [[pdf]](https://arxiv.org/pdf/2410.23262)</summary>

**Abstract:** We introduce EMMA, an End-to-end Multimodal Model for Autonomous driving. Built upon a multi-modal large language model foundation like Gemini, EMMA directly maps raw camera sensor data into various driving-specific outputs, including planner trajectories, perception objects, and road graph elements. EMMA maximizes the utility of world knowledge from the pre-trained large language models, by representing all non-sensor inputs (e.g. navigation instructions and ego vehicle status) and outputs (e.g. trajectories and 3D locations) as natural language text. This approach allows EMMA to jointly process various driving tasks in a unified language space, and generate the outputs for each task using task-specific prompts. Empirically, we demonstrate EMMA's effectiveness by achieving state-of-the-art performance in motion planning on nuScenes as well as competitive results on the Waymo Open Motion Dataset (WOMD). EMMA also yields competitive results for camera-primary 3D object detection on the Waymo Open Dataset (WOD). We show that co-training EMMA with planner trajectories, object detection, and road graph tasks yields improvements across all three domains, highlighting EMMA's potential as a generalist model for autonomous driving applications. We hope that our results will inspire research to further evolve the state of the art in autonomous driving model architectures.

**arXiv ID:** 2410.23262
</details>

<details>
<summary><strong>ZERA: Zero-init Instruction Evolving Refinement Agent - From Zero Instructions to Structured Prompts via Principle-based Optimization</strong> - Seungyoun Yi, Minsoo Khang, Sungrae Park - [[pdf]](https://arxiv.org/pdf/2509.18158)</summary>

**Abstract:** Automatic Prompt Optimization (APO) improves large language model (LLM) performance by refining prompts for specific tasks. However, prior APO methods typically focus only on user prompts, rely on unstructured feedback, and require large sample sizes and long iteration cycles-making them costly and brittle. We propose ZERA (Zero-init Instruction Evolving Refinement Agent), a novel framework that jointly optimizes both system and user prompts through principled, low-overhead refinement. ZERA scores prompts using eight generalizable criteria with automatically inferred weights, and revises prompts based on these structured critiques. This enables fast convergence to high-quality prompts using minimal examples and short iteration cycles. We evaluate ZERA across five LLMs and nine diverse datasets spanning reasoning, summarization, and code generation tasks. Experimental results demonstrate consistent improvements over strong baselines. Further ablation studies highlight the contribution of each component to more effective prompt construction. Our implementation including all prompts is publicly available at this https URL.

**arXiv ID:** 2509.18158
</details>

<details>
<summary><strong>Disambiguation in Conversational Question Answering in the Era of LLMs and Agents: A Survey</strong> - Md Mehrab Tanjim, Yeonjun In, Xiang Chen, Victor S. Bursztyn, Ryan A. Rossi, Sungchul Kim, Guang-Jie Ren, Vaishnavi Muppala, Shun Jiang, Yongsung Kim, Chanyoung Park - [[pdf]](https://arxiv.org/pdf/2505.12543)</summary>

**Abstract:** Ambiguity remains a fundamental challenge in Natural Language Processing (NLP) due to the inherent complexity and flexibility of human language. With the advent of Large Language Models (LLMs), addressing ambiguity has become even more critical due to their expanded capabilities and applications. In the context of Conversational Question Answering (CQA), this paper explores the definition, forms, and implications of ambiguity for language driven systems, particularly in the context of LLMs. We define key terms and concepts, categorize various disambiguation approaches enabled by LLMs, and provide a comparative analysis of their advantages and disadvantages. We also explore publicly available datasets for benchmarking ambiguity detection and resolution techniques and highlight their relevance for ongoing research. Finally, we identify open problems and future research directions, especially in agentic settings, proposing areas for further investigation. By offering a comprehensive review of current research on ambiguities and disambiguation with LLMs, we aim to contribute to the development of more robust and reliable LLM-based systems.

**arXiv ID:** 2505.12543
</details>

<details>
<summary><strong>Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs</strong> - Ilham Wicaksono, Zekun Wu, Rahul Patel, Theo King, Adriano Koshiyama, Philip Treleaven - [[pdf]](https://arxiv.org/pdf/2509.04802)</summary>

**Abstract:** As large language models transition to agentic systems, current safety evaluation frameworks face critical gaps in assessing deployment-specific risks. We introduce AgentSeer, an observability-based evaluation framework that decomposes agentic executions into granular action and component graphs, enabling systematic agentic-situational assessment. Through cross-model validation on GPT-OSS-20B and Gemini-2.0-flash using HarmBench single turn and iterative refinement attacks, we demonstrate fundamental differences between model-level and agentic-level vulnerability profiles. Model-level evaluation reveals baseline differences: GPT-OSS-20B (39.47% ASR) versus Gemini-2.0-flash (50.00% ASR), with both models showing susceptibility to social engineering while maintaining logic-based attack resistance. However, agentic-level assessment exposes agent-specific risks invisible to traditional evaluation. We discover "agentic-only" vulnerabilities that emerge exclusively in agentic contexts, with tool-calling showing 24-60% higher ASR across both models. Cross-model analysis reveals universal agentic patterns, agent transfer operations as highest-risk tools, semantic rather than syntactic vulnerability mechanisms, and context-dependent attack effectiveness, alongside model-specific security profiles in absolute ASR levels and optimal injection strategies. Direct attack transfer from model-level to agentic contexts shows degraded performance (GPT-OSS-20B: 57% human injection ASR; Gemini-2.0-flash: 28%), while context-aware iterative attacks successfully compromise objectives that failed at model-level, confirming systematic evaluation gaps. These findings establish the urgent need for agentic-situation evaluation paradigms, with AgentSeer providing the standardized methodology and empirical validation.

**arXiv ID:** 2509.04802
</details>

<details>
<summary><strong>Joint Cooperative and Non-Cooperative Localization in WSNs with Distributed Scaled Proximal ADMM Algorithms</strong> - Qiaojia Zhu, Xiaojing Shen, Haiqi Liu, Pramod K. Varshney - [[pdf]](https://arxiv.org/pdf/2509.18213)</summary>

**Abstract:** Cooperative and non-cooperative localization frequently arise together in wireless sensor networks, particularly when sensor positions are uncertain and targets are unable to communicate with the network. While joint processing can eliminate the delay in target estimation found in sequential approaches, it introduces complex variable coupling, posing challenges in both modeling and optimization. This paper presents a joint modeling approach that formulates cooperative and non-cooperative localization as a single optimization problem. To address the resulting coupling, we introduce auxiliary variables that enable structural decoupling and distributed computation. Building on this formulation, we develop the Scaled Proximal Alternating Direction Method of Multipliers for Joint Cooperative and Non-Cooperative Localization (SP-ADMM-JCNL). Leveraging the problem's structured design, we provide theoretical guarantees that the algorithm generates a sequence converging globally to the Karush-Kuhn-Tucker (KKT) point of the reformulated problem and further to a critical point of the original non-convex objective function, with a sublinear rate of O(1/T). Experiments on both synthetic and benchmark datasets demonstrate that SP-ADMM-JCNL achieves accurate and reliable localization performance.

**arXiv ID:** 2509.18213
</details>

<details>
<summary><strong>Integrating Stacked Intelligent Metasurfaces and Power Control for Dynamic Edge Inference via Over-The-Air Neural Networks</strong> - Kyriakos Stylianopoulos, George C. Alexandropoulos - [[pdf]](https://arxiv.org/pdf/2509.18906)</summary>

**Abstract:** This paper introduces a novel framework for Edge Inference (EI) that bypasses the conventional practice of treating the wireless channel as noise. We utilize Stacked Intelligent Metasurfaces (SIMs) to control wireless propagation, enabling the channel itself to perform over-the-air computation. This eliminates the need for symbol estimation at the receiver, significantly reducing computational and communication overhead. Our approach models the transmitter-channel-receiver system as an end-to-end Deep Neural Network (DNN) where the response of the SIM elements are trainable parameters. To address channel variability, we incorporate a dedicated DNN module responsible for dynamically adjusting transmission power leveraging user location information. Our performance evaluations showcase that the proposed metasurfaces-integrated DNN framework with deep SIM architectures are capable of balancing classification accuracy and power consumption under diverse scenarios, offering significant energy efficiency improvements.

**arXiv ID:** 2509.18906
</details>

<details>
<summary><strong>PIE: Perception and Interaction Enhanced End-to-End Motion Planning for Autonomous Driving</strong> - Chengran Yuan, Zijian Lu, Zhanqi Zhang, Yimin Zhao, Zefan Huang, Shuo Sun, Jiawei Sun, Jiahui Li, Christina Dao Wen Lee, Dongen Li, Marcelo H. Ang Jr - [[pdf]](https://arxiv.org/pdf/2509.18609)</summary>

**Abstract:** End-to-end motion planning is promising for simplifying complex autonomous driving pipelines. However, challenges such as scene understanding and effective prediction for decision-making continue to present substantial obstacles to its large-scale deployment. In this paper, we present PIE, a pioneering framework that integrates advanced perception, reasoning, and intention modeling to dynamically capture interactions between the ego vehicle and surrounding agents. It incorporates a bidirectional Mamba fusion that addresses data compression losses in multimodal fusion of camera and LiDAR inputs, alongside a novel reasoning-enhanced decoder integrating Mamba and Mixture-of-Experts to facilitate scene-compliant anchor selection and optimize adaptive trajectory inference. PIE adopts an action-motion interaction module to effectively utilize state predictions of surrounding agents to refine ego planning. The proposed framework is thoroughly validated on the NAVSIM benchmark. PIE, without using any ensemble and data augmentation techniques, achieves an 88.9 PDM score and 85.6 EPDM score, surpassing the performance of prior state-of-the-art methods. Comprehensive quantitative and qualitative analyses demonstrate that PIE is capable of reliably generating feasible and high-quality ego trajectories.

**arXiv ID:** 2509.18609
</details>

<details>
<summary><strong>V2V-GoT: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multimodal Large Language Models and Graph-of-Thoughts</strong> - Hsu-kuang Chiu, Ryo Hachiuma, Chien-Yi Wang, Yu-Chiang Frank Wang, Min-Hung Chen, Stephen F. Smith - [[pdf]](https://arxiv.org/pdf/2509.18053)</summary>

**Abstract:** Current state-of-the-art autonomous vehicles could face safety-critical situations when their local sensors are occluded by large nearby objects on the road. Vehicle-to-vehicle (V2V) cooperative autonomous driving has been proposed as a means of addressing this problem, and one recently introduced framework for cooperative autonomous driving has further adopted an approach that incorporates a Multimodal Large Language Model (MLLM) to integrate cooperative perception and planning processes. However, despite the potential benefit of applying graph-of-thoughts reasoning to the MLLM, this idea has not been considered by previous cooperative autonomous driving research. In this paper, we propose a novel graph-of-thoughts framework specifically designed for MLLM-based cooperative autonomous driving. Our graph-of-thoughts includes our proposed novel ideas of occlusion-aware perception and planning-aware prediction. We curate the V2V-GoT-QA dataset and develop the V2V-GoT model for training and testing the cooperative driving graph-of-thoughts. Our experimental results show that our method outperforms other baselines in cooperative perception, prediction, and planning tasks. Our project website: this https URL .

**arXiv ID:** 2509.18053
</details>

<details>
<summary><strong>EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device</strong> - Gunjan Chhablani, Xiaomeng Ye, Muhammad Zubair Irshad, Zsolt Kira - [[pdf]](https://arxiv.org/pdf/2509.17430)</summary>

**Abstract:** The field of Embodied AI predominantly relies on simulation for training and evaluation, often using either fully synthetic environments that lack photorealism or high-fidelity real-world reconstructions captured with expensive hardware. As a result, sim-to-real transfer remains a major challenge. In this paper, we introduce EmbodiedSplat, a novel approach that personalizes policy training by efficiently capturing the deployment environment and fine-tuning policies within the reconstructed scenes. Our method leverages 3D Gaussian Splatting (GS) and the Habitat-Sim simulator to bridge the gap between realistic scene capture and effective training environments. Using iPhone-captured deployment scenes, we reconstruct meshes via GS, enabling training in settings that closely approximate real-world conditions. We conduct a comprehensive analysis of training strategies, pre-training datasets, and mesh reconstruction techniques, evaluating their impact on sim-to-real predictivity in real-world scenarios. Experimental results demonstrate that agents fine-tuned with EmbodiedSplat outperform both zero-shot baselines pre-trained on large-scale real-world datasets (HM3D) and synthetically generated datasets (HSSD), achieving absolute success rate improvements of 20% and 40% on real-world Image Navigation task. Moreover, our approach yields a high sim-vs-real correlation (0.87-0.97) for the reconstructed meshes, underscoring its effectiveness in adapting policies to diverse environments with minimal effort. Project page: this https URL.

**arXiv ID:** 2509.17430
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>LLMZ+: Contextual Prompt Whitelist Principles for Agentic LLMs</strong> - Tom Pawelek, Raj Patel, Charlotte Crowell, Noorbakhsh Amiri, Sudip Mittal, Shahram Rahimi, Andy Perkins - [[pdf]](https://arxiv.org/pdf/2509.18557)</summary>

**Abstract:** Compared to traditional models, agentic AI represents a highly valuable target for potential attackers as they possess privileged access to data sources and API tools, which are traditionally not incorporated into classical agents. Unlike a typical software application residing in a Demilitarized Zone (DMZ), agentic LLMs consciously rely on nondeterministic behavior of the AI (only defining a final goal, leaving the path selection to LLM). This characteristic introduces substantial security risk to both operational security and information security. Most common existing defense mechanism rely on detection of malicious intent and preventing it from reaching the LLM agent, thus protecting against jailbreak attacks such as prompt injection. In this paper, we present an alternative approach, LLMZ+, which moves beyond traditional detection-based approaches by implementing prompt whitelisting. Through this method, only contextually appropriate and safe messages are permitted to interact with the agentic LLM. By leveraging the specificity of context, LLMZ+ guarantees that all exchanges between external users and the LLM conform to predefined use cases and operational boundaries. Our approach streamlines the security framework, enhances its long-term resilience, and reduces the resources required for sustaining LLM information security. Our empirical evaluation demonstrates that LLMZ+ provides strong resilience against the most common jailbreak prompts. At the same time, legitimate business communications are not disrupted, and authorized traffic flows seamlessly between users and the agentic LLM. We measure the effectiveness of approach using false positive and false negative rates, both of which can be reduced to 0 in our experimental setting.

**arXiv ID:** 2509.18557
</details>

<details>
<summary><strong>LCMF: Lightweight Cross-Modality Mambaformer for Embodied Robotics VQA</strong> - Zeyi Kang, Liang He, Yanxin Zhang, Zuheng Ming, Kaixing Zhao - [[pdf]](https://arxiv.org/pdf/2509.18576)</summary>

**Abstract:** Multimodal semantic learning plays a critical role in embodied intelligence, especially when robots perceive their surroundings, understand human instructions, and make intelligent decisions. However, the field faces technical challenges such as effective fusion of heterogeneous data and computational efficiency in resource-constrained environments. To address these challenges, this study proposes the lightweight LCMF cascaded attention framework, introducing a multi-level cross-modal parameter sharing mechanism into the Mamba module. By integrating the advantages of Cross-Attention and Selective parameter-sharing State Space Models (SSMs), the framework achieves efficient fusion of heterogeneous modalities and semantic complementary alignment. Experimental results show that LCMF surpasses existing multimodal baselines with an accuracy of 74.29% in VQA tasks and achieves competitive mid-tier performance within the distribution cluster of Large Language Model Agents (LLM Agents) in EQA video tasks. Its lightweight design achieves a 4.35-fold reduction in FLOPs relative to the average of comparable baselines while using only 166.51M parameters (image-text) and 219M parameters (video-text), providing an efficient solution for Human-Robot Interaction (HRI) applications in resource-constrained scenarios with strong multimodal decision generalization capabilities.

**arXiv ID:** 2509.18576
</details>

<details>
<summary><strong>On the Soundness and Consistency of LLM Agents for Executing Test Cases Written in Natural Language</strong> - Sébastien Salva, Redha Taguelmimt - [[pdf]](https://arxiv.org/pdf/2509.19136)</summary>

**Abstract:** The use of natural language (NL) test cases for validating graphical user interface (GUI) applications is emerging as a promising direction to manually written executable test scripts, which are costly to develop and difficult to maintain. Recent advances in large language models (LLMs) have opened the possibility of the direct execution of NL test cases by LLM agents. This paper investigates this direction, focusing on the impact on NL test case unsoundness and on test case execution consistency. NL test cases are inherently unsound, as they may yield false failures due to ambiguous instructions or unpredictable agent behaviour. Furthermore, repeated executions of the same NL test case may lead to inconsistent outcomes, undermining test reliability. To address these challenges, we propose an algorithm for executing NL test cases with guardrail mechanisms and specialised agents that dynamically verify the correct execution of each test step. We introduce measures to evaluate the capabilities of LLMs in test execution and one measure to quantify execution consistency. We propose a definition of weak unsoundness to characterise contexts in which NL test case execution remains acceptable, with respect to the industrial quality levels Six Sigma. Our experimental evaluation with eight publicly available LLMs, ranging from 3B to 70B parameters, demonstrates both the potential and current limitations of current LLM agents for GUI testing. Our experiments show that Meta Llama 3.1 70B demonstrates acceptable capabilities in NL test case execution with high execution consistency (above the level 3-sigma). We provide prototype tools, test suites, and results.

**arXiv ID:** 2509.19136
</details>

<details>
<summary><strong>LogicGuard: Improving Embodied LLM agents through Temporal Logic based Critics</strong> - Anand Gokhale, Vaibhav Srivastava, Francesco Bullo - [[pdf]](https://arxiv.org/pdf/2507.03293)</summary>

**Abstract:** Large language models (LLMs) have shown promise in zero-shot and single step reasoning and decision making problems, but in long horizon sequential planning tasks, their errors compound, often leading to unreliable or inefficient behavior. We introduce LogicGuard, a modular actor-critic architecture in which an LLM actor is guided by a trajectory level LLM critic that communicates through Linear Temporal Logic (LTL). Our setup combines the reasoning strengths of language models with the guarantees of formal logic. The actor selects high-level actions from natural language observations, while the critic analyzes full trajectories and proposes new LTL constraints that shield the actor from future unsafe or inefficient behavior. LogicGuard supports both fixed safety rules and adaptive, learned constraints, and is model-agnostic: any LLM-based planner can serve as the actor, with LogicGuard acting as a logic-generating wrapper. We formalize planning as graph traversal under symbolic constraints, allowing LogicGuard to analyze failed or suboptimal trajectories and generate new temporal logic rules that improve future behavior. To demonstrate generality, we evaluate LogicGuard across two distinct settings: short-horizon general tasks and long-horizon specialist tasks. On the Behavior benchmark of 100 household tasks, LogicGuard increases task completion rates by 25% over a baseline InnerMonologue planner. On the Minecraft diamond-mining task, which is long-horizon and requires multiple interdependent subgoals, LogicGuard improves both efficiency and safety compared to SayCan and InnerMonologue. These results show that enabling LLMs to supervise each other through temporal logic yields more reliable, efficient and safe decision-making for both embodied agents.

**arXiv ID:** 2507.03293
</details>

<details>
<summary><strong>Pandora: A Code-Driven Large Language Model Agent for Unified Reasoning Across Diverse Structured Knowledge</strong> - Yongrui Chen, Junhao He, Linbo Fu, Shenyu Zhang, Rihui Jin, Xinbang Dai, Jiaqi Li, Dehai Min, Nan Hu, Yuxin Zhang, Guilin Qi, Yi Huang, Tongtong Wu - [[pdf]](https://arxiv.org/pdf/2504.12734)</summary>

**Abstract:** Unified Structured Knowledge Reasoning (USKR) aims to answer natural language questions (NLQs) by using structured sources such as tables, databases, and knowledge graphs in a unified way. Existing USKR methods either rely on employing task-specific strategies or custom-defined representations, which struggle to leverage the knowledge transfer between different SKR tasks or align with the prior of LLMs, thereby limiting their performance. This paper proposes a novel USKR framework named \textsc{Pandora}, which takes advantage of \textsc{Python}'s \textsc{Pandas} API to construct a unified knowledge representation for alignment with LLM pre-training. It employs an LLM to generate textual reasoning steps and executable Python code for each question. Demonstrations are drawn from a memory of training examples that cover various SKR tasks, facilitating knowledge transfer. Extensive experiments on four benchmarks involving three SKR tasks demonstrate that \textsc{Pandora} outperforms existing unified frameworks and competes effectively with task-specific methods.

**arXiv ID:** 2504.12734
</details>

<details>
<summary><strong>Training Language Model Agents to Find Vulnerabilities with CTF-Dojo</strong> - Terry Yue Zhuo, Dingmin Wang, Hantian Ding, Varun Kumar, Zijian Wang - [[pdf]](https://arxiv.org/pdf/2508.18370)</summary>

**Abstract:** Large language models (LLMs) have demonstrated exceptional capabilities when trained within executable runtime environments, notably excelling at software engineering tasks through verified feedback loops. Yet, scalable and generalizable execution-grounded environments remain scarce, limiting progress in training more capable ML agents. We introduce CTF-Dojo, the first large-scale executable runtime tailored for training LLMs with verifiable feedback, featuring 658 fully functional Capture-The-Flag (CTF)-style challenges containerized in Docker with guaranteed reproducibility. To enable rapid scaling without manual intervention, we develop CTF-Forge, an automated pipeline that transforms publicly available artifacts into ready-to-use execution environments in minutes, eliminating weeks of expert configuration traditionally required. We trained LLM-based agents on just 486 high-quality, execution-verified trajectories from CTF-Dojo, achieving up to 11.6% absolute gains over strong baselines across three competitive benchmarks: InterCode-CTF, NYU CTF Bench, and Cybench. Our best-performing 32B model reaches 31.9% Pass@1, establishing a new open-weight state-of-the-art that rivals frontier models like DeepSeek-V3-0324 and Gemini-2.5-Flash. By framing CTF-style tasks as a benchmark for executable-agent learning, CTF-Dojo demonstrates that execution-grounded training signals are not only effective but pivotal in advancing high-performance ML agents without dependence on costly proprietary systems.

**arXiv ID:** 2508.18370
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (13 papers)</h2></summary>

<details>
<summary><strong>Foam-Agent: An End-to-End Composable Multi-Agent Framework for Automating CFD Simulation in OpenFOAM</strong> - Ling Yue, Nithin Somasekharan, Tingwen Zhang, Yadi Cao, Shaowu Pan - [[pdf]](https://arxiv.org/pdf/2509.18178)</summary>

**Abstract:** Computational Fluid Dynamics (CFD) is an essential simulation tool in engineering, yet its steep learning curve and complex manual setup create significant barriers. To address these challenges, we introduce Foam-Agent, a multi-agent framework that automates the entire end-to-end OpenFOAM workflow from a single natural language prompt. Our key innovations address critical gaps in existing systems: 1. An Comprehensive End-to-End Simulation Automation: Foam-Agent is the first system to manage the full simulation pipeline, including advanced pre-processing with a versatile Meshing Agent capable of handling external mesh files and generating new geometries via Gmsh, automatic generation of HPC submission scripts, and post-simulation visualization via ParaView. 2. Composable Service Architecture: Going beyond a monolithic agent, the framework uses Model Context Protocol (MCP) to expose its core functions as discrete, callable tools. This allows for flexible integration and use by other agentic systems, such as Claude-code, for more exploratory workflows. 3. High-Fidelity Configuration Generation: We achieve superior accuracy through a Hierarchical Multi-Index RAG for precise context retrieval and a dependency-aware generation process that ensures configuration consistency. Evaluated on a benchmark of 110 simulation tasks, Foam-Agent achieves an 88.2% success rate with Claude 3.5 Sonnet, significantly outperforming existing frameworks (55.5% for MetaOpenFOAM). Foam-Agent dramatically lowers the expertise barrier for CFD, demonstrating how specialized multi-agent systems can democratize complex scientific computing. The code is public at this https URL.

**arXiv ID:** 2509.18178
</details>

<details>
<summary><strong>The AGNTCY Agent Directory Service: Architecture and Implementation</strong> - Luca Muscariello, Vijoy Pandey, Ramiz Polic - [[pdf]](https://arxiv.org/pdf/2509.18787)</summary>

**Abstract:** The Agent Directory Service (ADS) is a distributed directory for the discovery of AI agent capabilities, metadata, and provenance. It leverages content-addressed storage, hierarchical taxonomies, and cryptographic signing to enable efficient, verifiable, and multi-dimensional discovery across heterogeneous Multi-Agent Systems (MAS). Built on the Open Agentic Schema Framework (OASF), ADS decouples capability indexing from content location through a two-level mapping realized over a Kademlia-based Distributed Hash Table (DHT). It reuses mature OCI / ORAS infrastructure for artifact distribution, integrates Sigstore for provenance, and supports schema-driven extensibility for emerging agent modalities (LLM prompt agents, MCP servers, A2A-enabled components). This paper formalizes the architectural model, describes storage and discovery layers, explains security and performance properties, and positions ADS within the broader landscape of emerging agent registry and interoperability initiatives.

**arXiv ID:** 2509.18787
</details>

<details>
<summary><strong>AgentInit: Initializing LLM-based Multi-Agent Systems via Diversity and Expertise Orchestration for Effective and Efficient Collaboration</strong> - Chunhao Tian, Yutong Wang, Xuebo Liu, Zhexuan Wang, Liang Ding, Miao Zhang, Min Zhang - [[pdf]](https://arxiv.org/pdf/2509.19236)</summary>

**Abstract:** Proper initialization is crucial for any system, particularly in multi-agent systems (MAS), where it plays a pivotal role in determining both the system's efficiency and effectiveness. However, existing MAS initialization methods do not fully account for the collaborative needs of the generated agents in subsequent stages. Inspired by the principles of effective team composition, we propose AgentInit, which aims to optimize the structure of agent teams. Specifically, in addition to multi-round interactions and reflections between agents during agent generation, AgentInit incorporates a Natural Language to Format mechanism to ensure consistency and standardization. Balanced team selection strategies using Pareto principles are subsequently applied to jointly consider agent team diversity and task relevance to promote effective and efficient collaboration and enhance overall system performance. Experiments show that AgentInit consistently outperforms state-of-the-art initialization methods and pre-defined strategies across various frameworks and tasks, achieving an overall performance improvement of up to 1.2 and 1.6, respectively, while also significantly reducing token consumption. Further analysis confirms its strong transferability to similar tasks and verifies the effectiveness of its key components, demonstrating its capability and adaptability as a reliable MAS initialization method. Source code and models are available at this https URL.

**arXiv ID:** 2509.19236
</details>

<details>
<summary><strong>Context Lineage Assurance for Non-Human Identities in Critical Multi-Agent Systems</strong> - Sumana Malkapuram, Sameera Gangavarapu, Kailashnath Reddy Kavalakuntla, Ananya Gangavarapu - [[pdf]](https://arxiv.org/pdf/2509.18415)</summary>

**Abstract:** The proliferation of autonomous software agents necessitates rigorous frameworks for establishing secure and verifiable agent-to-agent (A2A) interactions, particularly when such agents are instantiated as non-human identities(NHIs). We extend the A2A paradigm [1 , 2] by introducing a cryptographically grounded mechanism for lineage verification, wherein the provenance and evolution of NHIs are anchored in append-only Merkle tree structures modeled after Certificate Transparency (CT) logs. Unlike traditional A2A models that primarily secure point-to-point interactions, our approach enables both agents and external verifiers to cryptographically validate multi-hop provenance, thereby ensuring the integrity of the entire call chain.
A federated proof server acts as an auditor across one or more Merkle logs, aggregating inclusion proofs and consistency checks into compact, signed attestations that external parties can verify without access to the full execution trace. In parallel, we augment the A2A agent card to incorporate explicit identity verification primitives, enabling both peer agents and human approvers to authenticate the legitimacy of NHI representations in a standardized manner. Together, these contributions establish a cohesive model that integrates identity attestation, lineage verification, and independent proof auditing, thereby advancing the security posture of inter-agent ecosystems and providing a foundation for robust governance of NHIs in regulated environments such as FedRAMP.

**arXiv ID:** 2509.18415
</details>

<details>
<summary><strong>A Multi-Agent Framework with Automated Decision Rule Optimization for Cross-Domain Misinformation Detection</strong> - Hui Li, Ante Wang, kunquan li, Zhihao Wang, Liang Zhang, Delai Qiu, Qingsong Liu, Jinsong Su - [[pdf]](https://arxiv.org/pdf/2503.23329)</summary>

**Abstract:** Misinformation spans various domains, but detection methods trained on specific domains often perform poorly when applied to others. With the rapid development of Large Language Models (LLMs), researchers have begun to utilize LLMs for cross-domain misinformation detection. However, existing LLM-based methods often fail to adequately analyze news in the target domain, limiting their detection capabilities. More importantly, these methods typically rely on manually designed decision rules, which are limited by domain knowledge and expert experience, thus limiting the generalizability of decision rules to different domains. To address these issues, we propose a MultiAgent Framework for cross-domain misinformation detection with Automated Decision Rule Optimization (MARO). Under this framework, we first employs multiple expert agents to analyze target-domain news. Subsequently, we introduce a question-reflection mechanism that guides expert agents to facilitate higherquality analysis. Furthermore, we propose a decision rule optimization approach based on carefully-designed cross-domain validation tasks to iteratively enhance the effectiveness of decision rules in different domains. Experimental results and in-depth analysis on commonlyused datasets demonstrate that MARO achieves significant improvements over existing methods.

**arXiv ID:** 2503.23329
</details>

<details>
<summary><strong>EvoAgentX: An Automated Framework for Evolving Agentic Workflows</strong> - Yingxu Wang, Siwei Liu, Jinyuan Fang, Zaiqiao Meng - [[pdf]](https://arxiv.org/pdf/2507.03616)</summary>

**Abstract:** Multi-agent systems (MAS) have emerged as a powerful paradigm for orchestrating large language models (LLMs) and specialized tools to collaboratively address complex tasks. However, existing MAS frameworks often require manual workflow configuration and lack native support for dynamic evolution and performance optimization. In addition, many MAS optimization algorithms are not integrated into a unified framework. In this paper, we present EvoAgentX, an open-source platform that automates the generation, execution, and evolutionary optimization of multi-agent workflows. EvoAgentX employs a modular architecture consisting of five core layers: the basic components, agent, workflow, evolving, and evaluation layers. Specifically, within the evolving layer, EvoAgentX integrates three MAS optimization algorithms, TextGrad, AFlow, and MIPRO, to iteratively refine agent prompts, tool configurations, and workflow topologies. We evaluate EvoAgentX on HotPotQA, MBPP, and MATH for multi-hop reasoning, code generation, and mathematical problem solving, respectively, and further assess it on real-world tasks using GAIA. Experimental results show that EvoAgentX consistently achieves significant performance improvements, including a 7.44% increase in HotPotQA F1, a 10.00% improvement in MBPP pass@1, a 10.00% gain in MATH solve accuracy, and an overall accuracy improvement of up to 20.00% on GAIA. The source code is available at: this https URL

**arXiv ID:** 2507.03616
</details>

<details>
<summary><strong>Abduct, Act, Predict: Scaffolding Causal Inference for Automated Failure Attribution in Multi-Agent Systems</strong> - Alva West, Yixuan Weng, Minjun Zhu, Zhen Lin, Zhiyuan Ning, Yue Zhang - [[pdf]](https://arxiv.org/pdf/2509.10401)</summary>

**Abstract:** Failure attribution in multi-agent systems -- pinpointing the exact step where a decisive error occurs -- is a critical yet unsolved challenge. Current methods treat this as a pattern recognition task over long conversation logs, leading to critically low step-level accuracy (below 17\%), which renders them impractical for debugging complex systems. Their core weakness is a fundamental inability to perform robust counterfactual reasoning: to determine if correcting a single action would have actually averted the task failure. To bridge this \emph{counterfactual inference gap}, we introduce Abduct-Act-Predict (A2P) Scaffolding, a novel agent framework that transforms failure attribution from pattern recognition into a structured causal inference task. A2P explicitly guides a large language model through a formal three-step reasoning process within a single inference pass: (1) Abduction, to infer the hidden root causes behind an agent's actions; (2) Action, to define a minimal corrective intervention; and (3) Prediction, to simulate the subsequent trajectory and verify if the intervention resolves the failure. This structured approach leverages the holistic context of the entire conversation while imposing a rigorous causal logic on the model's analysis. Our extensive experiments on the Who\&When benchmark demonstrate its efficacy. On the Algorithm-Generated dataset, A2P achieves 47.46\% step-level accuracy, a 2.85$\times$ improvement over the 16.67\% of the baseline. On the more complex Hand-Crafted dataset, it achieves 29.31\% step accuracy, a 2.43$\times$ improvement over the baseline's 12.07\%. By reframing the problem through a causal lens, A2P Scaffolding provides a robust, verifiable, and significantly more accurate solution for automated failure attribution. Ours code are released at this https URL.

**arXiv ID:** 2509.10401
</details>

<details>
<summary><strong>Difficulty-Aware Agent Orchestration in LLM-Powered Workflows</strong> - Jinwei Su, Yinghui Xia, Qizhen Lan, Xinyuan Song, Chen Chen, Yang Jingsong, Lewei He, Tianyu Shi - [[pdf]](https://arxiv.org/pdf/2509.11079)</summary>

**Abstract:** Large Language Model (LLM)-based agentic systems have shown strong capabilities across various tasks. However, existing multi-agent frameworks often rely on static or task-level workflows, which either over-process simple queries or underperform on complex ones, while also neglecting the efficiency-performance trade-offs across heterogeneous LLMs. To address these limitations, we propose Difficulty-Aware Agentic Orchestration (DAAO), a dynamic framework that adapts workflow depth, operator selection, and LLM assignment based on the difficulty of each input query. DAAO comprises three interdependent modules: a variational autoencoder (VAE) for difficulty estimation, a modular operator allocator, and a cost- and performance-aware LLM router. By leveraging heterogeneous LLMs and dynamically tailoring workflows, DAAO enables fine-grained, query-specific reasoning strategies. DAAO outperforms prior multi-agent systems in both accuracy and inference efficiency across six benchmarks. We will release our code and implementation details upon publication.

**arXiv ID:** 2509.11079
</details>

<details>
<summary><strong>Policy Gradient with Self-Attention for Model-Free Distributed Nonlinear Multi-Agent Games</strong> - Eduardo Sebastián, Maitrayee Keskar, Eeman Iqbal, Eduardo Montijano, Carlos Sagüés, Nikolay Atanasov - [[pdf]](https://arxiv.org/pdf/2509.18371)</summary>

**Abstract:** Multi-agent games in dynamic nonlinear settings are challenging due to the time-varying interactions among the agents and the non-stationarity of the (potential) Nash equilibria. In this paper we consider model-free games, where agent transitions and costs are observed without knowledge of the transition and cost functions that generate them. We propose a policy gradient approach to learn distributed policies that follow the communication structure in multi-team games, with multiple agents per team. Our formulation is inspired by the structure of distributed policies in linear quadratic games, which take the form of time-varying linear feedback gains. In the nonlinear case, we model the policies as nonlinear feedback gains, parameterized by self-attention layers to account for the time-varying multi-agent communication topology. We demonstrate that our distributed policy gradient approach achieves strong performance in several settings, including distributed linear and nonlinear regulation, and simulated and real multi-robot pursuit-and-evasion games.

**arXiv ID:** 2509.18371
</details>

<details>
<summary><strong>SIRAG: Towards Stable and Interpretable RAG with A Process-Supervised Multi-Agent Framework</strong> - Junlin Wang, Zehao Wu, Shaowei Lu, Yanlan Li, Xinghao Huang - [[pdf]](https://arxiv.org/pdf/2509.18167)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to access external knowledge sources, but the effectiveness of RAG relies on the coordination between the retriever and the generator. Since these components are developed independently, their interaction is often suboptimal: the retriever may return irrelevant or redundant documents, while the generator may fail to fully leverage retrieved evidence. In this work, we propose a process-supervised multi-agent framework to bridge the gap between retriever and generator. The framework introduces two lightweight agents: a Decision Maker, which determines when to continue retrieval or stop for answer generation, and a Knowledge Selector, which filters retrieved documents to retain only the most useful evidence. To provide fine-grained supervision, we employ an LLM-as-a-Judge that evaluates each intermediate action with process-level rewards, ensuring more accurate credit assignment than relying solely on final answer correctness. We further adopt a tree-structured rollout strategy to explore diverse reasoning paths, and train both agents with Proximal Policy Optimization (PPO) in an end-to-end manner. Experiments on single-hop and multi-hop question answering benchmarks show that our approach achieves higher accuracy, more stable convergence, and produces more interpretable reasoning trajectories compared with standard RAG baselines. Importantly, the proposed framework is modular and plug-and-play, requiring no modification to the retriever or generator, making it practical for real-world RAG applications.

**arXiv ID:** 2509.18167
</details>

<details>
<summary><strong>MAPEX: A Multi-Agent Pipeline for Keyphrase Extraction</strong> - Liting Zhang, Shiwan Zhao, Aobo Kong, Qicheng Li - [[pdf]](https://arxiv.org/pdf/2509.18813)</summary>

**Abstract:** Keyphrase extraction is a fundamental task in natural language processing. However, existing unsupervised prompt-based methods for Large Language Models (LLMs) often rely on single-stage inference pipelines with uniform prompting, regardless of document length or LLM backbone. Such one-size-fits-all designs hinder the full exploitation of LLMs' reasoning and generation capabilities, especially given the complexity of keyphrase extraction across diverse scenarios. To address these challenges, we propose MAPEX, the first framework that introduces multi-agent collaboration into keyphrase extraction. MAPEX coordinates LLM-based agents through modules for expert recruitment, candidate extraction, topic guidance, knowledge augmentation, and post-processing. A dual-path strategy dynamically adapts to document length: knowledge-driven extraction for short texts and topic-guided extraction for long texts. Extensive experiments on six benchmark datasets across three different LLMs demonstrate its strong generalization and universality, outperforming the state-of-the-art unsupervised method by 2.44\% and standard LLM baselines by 4.01\% in F1@5 on average. Code is available at this https URL.

**arXiv ID:** 2509.18813
</details>

<details>
<summary><strong>Agentic AutoSurvey: Let LLMs Survey LLMs</strong> - Yixin Liu, Yonghui Wu, Denghui Zhang, Lichao Sun - [[pdf]](https://arxiv.org/pdf/2509.18661)</summary>

**Abstract:** The exponential growth of scientific literature poses unprecedented challenges for researchers attempting to synthesize knowledge across rapidly evolving fields. We present \textbf{Agentic AutoSurvey}, a multi-agent framework for automated survey generation that addresses fundamental limitations in existing approaches. Our system employs four specialized agents (Paper Search Specialist, Topic Mining \& Clustering, Academic Survey Writer, and Quality Evaluator) working in concert to generate comprehensive literature surveys with superior synthesis quality. Through experiments on six representative LLM research topics from COLM 2024 categories, we demonstrate that our multi-agent approach achieves significant improvements over existing baselines, scoring 8.18/10 compared to AutoSurvey's 4.77/10. The multi-agent architecture processes 75--443 papers per topic (847 total across six topics) while targeting high citation coverage (often $\geq$80\% on 75--100-paper sets; lower on very large sets such as RLHF) through specialized agent orchestration. Our 12-dimension evaluation captures organization, synthesis integration, and critical analysis beyond basic metrics. These findings demonstrate that multi-agent architectures represent a meaningful advancement for automated literature survey generation in rapidly evolving scientific domains.

**arXiv ID:** 2509.18661
</details>

<details>
<summary><strong>Global Convergence of Multi-Agent Policy Gradient in Markov Potential Games</strong> - Stefanos Leonardos, Will Overman, Ioannis Panageas, Georgios Piliouras - [[pdf]](https://arxiv.org/pdf/2106.01969)</summary>

**Abstract:** Potential games are arguably one of the most important and widely studied classes of normal form games. They define the archetypal setting of multi-agent coordination as all agent utilities are perfectly aligned with each other via a common potential function. Can this intuitive framework be transplanted in the setting of Markov Games? What are the similarities and differences between multi-agent coordination with and without state dependence? We present a novel definition of Markov Potential Games (MPG) that generalizes prior attempts at capturing complex stateful multi-agent coordination. Counter-intuitively, insights from normal-form potential games do not carry over as MPGs can consist of settings where state-games can be zero-sum games. In the opposite direction, Markov games where every state-game is a potential game are not necessarily MPGs. Nevertheless, MPGs showcase standard desirable properties such as the existence of deterministic Nash policies. In our main technical result, we prove fast convergence of independent policy gradient to Nash policies by adapting recent gradient dominance property arguments developed for single agent MDPs to multi-agent learning settings.

**arXiv ID:** 2106.01969
</details>

</details>

<details open>
<summary><h2>Other Agent Research (3 papers)</h2></summary>

<details>
<summary><strong>Adaptive Learning in Spatial Agent-Based Models for Climate Risk Assessment: A Geospatial Framework with Evolutionary Economic Agents</strong> - Yara Mohajerani - [[pdf]](https://arxiv.org/pdf/2509.18633)</summary>

**Abstract:** Climate risk assessment requires modelling complex interactions between spatially heterogeneous hazards and adaptive economic systems. We present a novel geospatial agent-based model that integrates climate hazard data with evolutionary learning for economic agents. Our framework combines Mesa-based spatial modelling with CLIMADA climate impact assessment, introducing adaptive learning behaviours that allow firms to evolve strategies for budget allocation, pricing, wages, and risk adaptation through fitness-based selection and mutation. We demonstrate the framework using riverine flood projections under RCP8.5 until 2100, showing that evolutionary adaptation enables firms to converge with baseline (no hazard) production levels after decades of disruption due to climate stress. Our results reveal systemic risks where even agents that are not directly exposed to floods face impacts through supply chain disruptions, with the end-of-century average price of goods 5.6% higher under RCP8.5 compared to the baseline. This open-source framework provides financial institutions and companies with tools to quantify both direct and cascading climate risks while evaluating cost-effective adaptation strategies.

**arXiv ID:** 2509.18633
</details>

<details>
<summary><strong>LLM-based Agents Suffer from Hallucinations: A Survey of Taxonomy, Methods, and Directions</strong> - Xixun Lin, Yucheng Ning, Jingwen Zhang, Yan Dong, Yilong Liu, Yongxuan Wu, Xiaohua Qi, Nan Sun, Yanmin Shang, Pengfei Cao, Lixin Zou, Xu Chen, Chuan Zhou, Jia Wu, Shirui Pan, Bin Wang, Yanan Cao, Kai Chen, Songlin Hu, Li Guo - [[pdf]](https://arxiv.org/pdf/2509.18970)</summary>

**Abstract:** Driven by the rapid advancements of Large Language Models (LLMs), LLM-based agents have emerged as powerful intelligent systems capable of human-like cognition, reasoning, and interaction. These agents are increasingly being deployed across diverse real-world applications, including student education, scientific research, and financial analysis. However, despite their remarkable potential, LLM-based agents remain vulnerable to hallucination issues, which can result in erroneous task execution and undermine the reliability of the overall system design. Addressing this critical challenge requires a deep understanding and a systematic consolidation of recent advances on LLM-based agents. To this end, we present the first comprehensive survey of hallucinations in LLM-based agents. By carefully analyzing the complete workflow of agents, we propose a new taxonomy that identifies different types of agent hallucinations occurring at different stages. Furthermore, we conduct an in-depth examination of eighteen triggering causes underlying the emergence of agent hallucinations. Through a detailed review of a large number of existing studies, we summarize approaches for hallucination mitigation and detection, and highlight promising directions for future research. We hope this survey will inspire further efforts toward addressing hallucinations in LLM-based agents, ultimately contributing to the development of more robust and reliable agent systems.

**arXiv ID:** 2509.18970
</details>

<details>
<summary><strong>Position: Human-Robot Interaction in Embodied Intelligence Demands a Shift From Static Privacy Controls to Dynamic Learning</strong> - Shuning Zhang, Hong Jia, Simin Li, Ting Dang, Yongquan `Owen' Hu, Xin Yi, Hewu Li - [[pdf]](https://arxiv.org/pdf/2509.19041)</summary>

**Abstract:** The reasoning capabilities of embodied agents introduce a critical, under-explored inferential privacy challenge, where the risk of an agent generate sensitive conclusions from ambient data. This capability creates a fundamental tension between an agent's utility and user privacy, rendering traditional static controls ineffective. To address this, this position paper proposes a framework that reframes privacy as a dynamic learning problem grounded in theory of Contextual Integrity (CI). Our approach enables agents to proactively learn and adapt to individual privacy norms through interaction, outlining a research agenda to develop embodied agents that are both capable and function as trustworthy safeguards of user privacy.

**arXiv ID:** 2509.19041
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (29 papers)</h2></summary>

<details>
<summary><strong>Towards General Computer Control with Hierarchical Agents and Multi-Level Action Spaces</strong> - Zihan Dong, Xinyu Fan, Zixiang Tang, Yunqing Li - [[pdf]](https://arxiv.org/pdf/2509.18230)</summary>

**Abstract:** Controlling desktop applications via software remains a fundamental yet under-served problem. Existing multi-modal large language models (MLLMs) ingest screenshots and task instructions to generate keystrokes and mouse events, but they suffer from prohibitive inference latency, poor sample efficiency on long-horizon sparse-reward tasks, and infeasible on-device deployment. We introduce a lightweight hierarchical reinforcement learning framework, ComputerAgent, that formulates OS control as a two-level option process (manager and subpolicy), employs a triple-modal state encoder (screenshot, task ID, numeric state) to handle visual and contextual diversity, integrates meta-actions with an early-stop mechanism to reduce wasted interactions, and uses a compact vision backbone plus small policy networks for on-device inference (15M parameters). On a suite of 135 real-world desktop tasks, ComputerAgent attains 92.1% success on simple tasks (<8 steps) and 58.8% on hard tasks (>=8 steps), matching or exceeding 200B-parameter MLLM baselines on simple scenarios while reducing model size by over four orders of magnitude and halving inference time. These results demonstrate that hierarchical RL offers a practical, scalable alternative to monolithic MLLM-based automation for computer control.

**arXiv ID:** 2509.18230
</details>

<details>
<summary><strong>Autonomous Data Agents: A New Opportunity for Smart Data</strong> - Yanjie Fu, Dongjie Wang, Wangyang Ying, Xiangliang Zhang, Huan Liu, Jian Pei - [[pdf]](https://arxiv.org/pdf/2509.18710)</summary>

**Abstract:** As data continues to grow in scale and complexity, preparing, transforming, and analyzing it remains labor-intensive, repetitive, and difficult to scale. Since data contains knowledge and AI learns knowledge from it, the alignment between AI and data is essential. However, data is often not structured in ways that are optimal for AI utilization. Moreover, an important question arises: how much knowledge can we pack into data through intensive data operations? Autonomous data agents (DataAgents), which integrate LLM reasoning with task decomposition, action reasoning and grounding, and tool calling, can autonomously interpret data task descriptions, decompose tasks into subtasks, reason over actions, ground actions into python code or tool calling, and execute operations. Unlike traditional data management and engineering tools, DataAgents dynamically plan workflows, call powerful tools, and adapt to diverse data tasks at scale. This report argues that DataAgents represent a paradigm shift toward autonomous data-to-knowledge systems. DataAgents are capable of handling collection, integration, preprocessing, selection, transformation, reweighing, augmentation, reprogramming, repairs, and retrieval. Through these capabilities, DataAgents transform complex and unstructured data into coherent and actionable knowledge. We first examine why the convergence of agentic AI and data-to-knowledge systems has emerged as a critical trend. We then define the concept of DataAgents and discuss their architectural design, training strategies, as well as the new skills and capabilities they enable. Finally, we call for concerted efforts to advance action workflow optimization, establish open datasets and benchmark ecosystems, safeguard privacy, balance efficiency with scalability, and develop trustworthy DataAgent guardrails to prevent malicious actions.

**arXiv ID:** 2509.18710
</details>

<details>
<summary><strong>MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents</strong> - Yifan Xu, Xiao Liu, Xinghan Liu, Jiaqi Fu, Hanchen Zhang, Bohao Jing, Shudan Zhang, Yuting Wang, Wenyi Zhao, Yuxiao Dong - [[pdf]](https://arxiv.org/pdf/2509.18119)</summary>

**Abstract:** Building general-purpose graphical user interface (GUI) agents has become increasingly promising with the progress in vision language models. However, developing effective mobile GUI agents with reinforcement learning (RL) remains challenging due to the heavy-tailed distribution of task difficulty and the inefficiency of large-scale environment sampling. We present an online agentic reinforcement learning framework MOBILERL to enhance GUI agents in mobile environments. Its core component is the Difficulty-Adaptive GRPO (ADAGRPO) algorithm. In ADAGRPO, we design difficulty-adaptive positive replay and failure curriculum filtering to adapt the model to different task difficulties. We introduce the shortest path reward adjustment strategy to reshape rewards concerning the task length in multi-turn agentic tasks. Those strategies jointly stabilize RL training, improve sample efficiency, and generate strong performance across diverse mobile apps and tasks. We apply MOBILERL to two open models (Qwen2.5-VL-7B-Instruct and GLM-4.1V-9B-Base). The resultant MOBILERL-9B model achieves state-of-the-art results in terms of success rates on both AndroidWorld (75.8%) and AndroidLab (46.8%). The MOBILERL framework is adopted in the AutoGLM products, and also open-sourced at this https URL.

**arXiv ID:** 2509.18119
</details>

<details>
<summary><strong>NurseSchedRL: Attention-Guided Reinforcement Learning for Nurse-Patient Assignment</strong> - Harsha Koduri - [[pdf]](https://arxiv.org/pdf/2509.18125)</summary>

**Abstract:** Healthcare systems face increasing pressure to allocate limited nursing resources efficiently while accounting for skill heterogeneity, patient acuity, staff fatigue, and continuity of care. Traditional optimization and heuristic scheduling methods struggle to capture these dynamic, multi-constraint environments. I propose NurseSchedRL, a reinforcement learning framework for nurse-patient assignment that integrates structured state encoding, constrained action masking, and attention-based representations of skills, fatigue, and geographical context. NurseSchedRL uses Proximal Policy Optimization (PPO) with feasibility masks to ensure assignments respect real-world constraints, while dynamically adapting to patient arrivals and varying nurse availability. In simulation with realistic nurse and patient data, NurseSchedRL achieves improved scheduling efficiency, better alignment of skills to patient needs, and reduced fatigue compared to baseline heuristic and unconstrained RL approaches. These results highlight the potential of reinforcement learning for decision support in complex, high-stakes healthcare workforce management.

**arXiv ID:** 2509.18125
</details>

<details>
<summary><strong>Check Field Detection Agent (CFD-Agent) using Multimodal Large Language and Vision Language Models</strong> - Sourav Halder, Jinjun Tong, Xinyu Wu - [[pdf]](https://arxiv.org/pdf/2509.18405)</summary>

**Abstract:** Checks remain a foundational instrument in the financial ecosystem, facilitating substantial transaction volumes across institutions. However, their continued use also renders them a persistent target for fraud, underscoring the importance of robust check fraud detection mechanisms. At the core of such systems lies the accurate identification and localization of critical fields, such as the signature, magnetic ink character recognition (MICR) line, courtesy amount, legal amount, payee, and payer, which are essential for subsequent verification against reference checks belonging to the same customer. This field-level detection is traditionally dependent on object detection models trained on large, diverse, and meticulously labeled datasets, a resource that is scarce due to proprietary and privacy concerns. In this paper, we introduce a novel, training-free framework for automated check field detection, leveraging the power of a vision language model (VLM) in conjunction with a multimodal large language model (MLLM). Our approach enables zero-shot detection of check components, significantly lowering the barrier to deployment in real-world financial settings. Quantitative evaluation of our model on a hand-curated dataset of 110 checks spanning multiple formats and layouts demonstrates strong performance and generalization capability. Furthermore, this framework can serve as a bootstrap mechanism for generating high-quality labeled datasets, enabling the development of specialized real-time object detection models tailored to institutional needs.

**arXiv ID:** 2509.18405
</details>

<details>
<summary><strong>APRIL: Active Partial Rollouts in Reinforcement Learning to tame long-tail generation</strong> - Yuzhen Zhou, Jiajun Li, Yusheng Su, Gowtham Ramesh, Zilin Zhu, Xiang Long, Chenyang Zhao, Jin Pan, Xiaodong Yu, Ze Wang, Kangrui Du, Jialian Wu, Ximeng Sun, Jiang Liu, Qiaolin Yu, Hao Chen, Zicheng Liu, Emad Barsoum - [[pdf]](https://arxiv.org/pdf/2509.18521)</summary>

**Abstract:** Reinforcement learning (RL) has become a cornerstone in advancing large-scale pre-trained language models (LLMs). Successive generations, including GPT-o series, DeepSeek-R1, Kimi-K1.5, Grok 4, and GLM-4.5, have relied on large-scale RL training to enhance reasoning and coding capabilities. To meet the community's growing RL needs, numerous RL frameworks have been proposed. Most of these frameworks primarily rely on inference engines for rollout generation and training engines for policy updates. However, RL training remains computationally expensive, with rollout generation accounting for more than 90% of total runtime. In addition, its efficiency is often constrained by the long-tail distribution of rollout response lengths, where a few lengthy responses stall entire batches, leaving GPUs idle and underutilized. As model and rollout sizes continue to grow, this bottleneck increasingly limits scalability. To address this challenge, we propose Active Partial Rollouts in Reinforcement Learning (APRIL), which mitigates long-tail inefficiency. In the rollout phase, APRIL over-provisions rollout requests, terminates once the target number of responses is reached, and recycles incomplete responses for continuation in future steps. This strategy ensures that no rollouts are discarded while substantially reducing GPU idle time. Experiments show that APRIL improves rollout throughput by at most 44% across commonly used RL algorithms (GRPO, DAPO, GSPO), accelerates convergence, and achieves at most 8% higher final accuracy across tasks. Moreover, APRIL is both framework and hardware agnostic, already integrated into the slime RL framework, and deployable on NVIDIA and AMD GPUs alike. Taken together, this work unifies system-level and algorithmic considerations in proposing APRIL, with the aim of advancing RL training efficiency and inspiring further optimizations in RL systems.

**arXiv ID:** 2509.18521
</details>

<details>
<summary><strong>Explore the Reinforcement Learning for the LLM based ASR and TTS system</strong> - Changfeng Gao, Yabin Li, Keyu An, Zhifu Gao, Zhihao Du, Han Zhao, Xiangang Li - [[pdf]](https://arxiv.org/pdf/2509.18569)</summary>

**Abstract:** In recent years, large language models (LLMs) have played an important role in automatic speech recognition (ASR) and text-to-speech (TTS) systems. While reinforcement learning (RL) has significantly enhanced LLM performance in text-based tasks, its application to ASR and TTS remains underexplored due to the complexity of training audio-based models. In this study, we propose a lightweight RL framework tailored for audio-based LLMs that can process audio inputs and generate audio outputs. Based on this framework, we evaluate the effectiveness of reinforcement learning on both ASR and TTS tasks. For the ASR task, we experiment with different rule-based reward functions within the Group Relative Policy Optimization (GRPO) framework and investigate the impact of RL data construction. For the TTS task, we compare GRPO with Differentiable Reward Optimization (DiffRO) and further combine the two approaches to achieve improved performance. Our experiments demonstrate that RL can significantly enhance the performance of both ASR and TTS systems, even with limited training data and a small number of optimization steps.

**arXiv ID:** 2509.18569
</details>

<details>
<summary><strong>OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation</strong> - Zhuoxiao Chen, Hongyang Yu, Ying Xu, Yadan Luo, Long Duong, Yuan-Fang Li - [[pdf]](https://arxiv.org/pdf/2509.18600)</summary>

**Abstract:** Radiology report generation (RRG) aims to automatically produce clinically faithful reports from chest X-ray images. Prevailing work typically follows a scale-driven paradigm, by multi-stage training over large paired corpora and oversized backbones, making pipelines highly data- and compute-intensive. In this paper, we propose Oracle-educated GRPO {OraPO) with a FactScore-based reward (FactS) to tackle the RRG task under constrained budgets. OraPO enables single-stage, RL-only training by converting failed GRPO explorations on rare or difficult studies into direct preference supervision via a lightweight oracle step. FactS grounds learning in diagnostic evidence by extracting atomic clinical facts and checking entailment against ground-truth labels, yielding dense, interpretable sentence-level rewards. Together, OraPO and FactS create a compact and powerful framework that significantly improves learning efficiency on clinically challenging cases, setting the new SOTA performance on the CheXpert Plus dataset (0.341 in F1) with 2--3 orders of magnitude less training data using a small base VLM on modest hardware.

**arXiv ID:** 2509.18600
</details>

<details>
<summary><strong>End-to-End Crop Row Navigation via LiDAR-Based Deep Reinforcement Learning</strong> - Ana Luiza Mineiro, Francisco Affonso, Marcelo Becker - [[pdf]](https://arxiv.org/pdf/2509.18608)</summary>

**Abstract:** Reliable navigation in under-canopy agricultural environments remains a challenge due to GNSS unreliability, cluttered rows, and variable lighting. To address these limitations, we present an end-to-end learning-based navigation system that maps raw 3D LiDAR data directly to control commands using a deep reinforcement learning policy trained entirely in simulation. Our method includes a voxel-based downsampling strategy that reduces LiDAR input size by 95.83%, enabling efficient policy learning without relying on labeled datasets or manually designed control interfaces. The policy was validated in simulation, achieving a 100% success rate in straight-row plantations and showing a gradual decline in performance as row curvature increased, tested across varying sinusoidal frequencies and amplitudes.

**arXiv ID:** 2509.18608
</details>

<details>
<summary><strong>Failure Makes the Agent Stronger: Enhancing Accuracy through Structured Reflection for Reliable Tool Interactions</strong> - Junhao Su, Yuanliang Wan, Junwei Yang, Hengyu Shi, Tianyang Han, Junfeng Luo, Yurui Qiu - [[pdf]](https://arxiv.org/pdf/2509.18847)</summary>

**Abstract:** Tool-augmented large language models (LLMs) are usually trained with supervised imitation or coarse-grained reinforcement learning that optimizes single tool calls. Current self-reflection practices rely on heuristic prompts or one-way reasoning: the model is urged to 'think more' instead of learning error diagnosis and repair. This is fragile in multi-turn interactions; after a failure the model often repeats the same mistake. We propose structured reflection, which turns the path from error to repair into an explicit, controllable, and trainable action. The agent produces a short yet precise reflection: it diagnoses the failure using evidence from the previous step and then proposes a correct, executable follow-up call. For training we combine DAPO and GSPO objectives with a reward scheme tailored to tool use, optimizing the stepwise strategy Reflect, then Call, then Final. To evaluate, we introduce Tool-Reflection-Bench, a lightweight benchmark that programmatically checks structural validity, executability, parameter correctness, and result consistency. Tasks are built as mini trajectories of erroneous call, reflection, and corrected call, with disjoint train and test splits. Experiments on BFCL v3 and Tool-Reflection-Bench show large gains in multi-turn tool-call success and error recovery, and a reduction of redundant calls. These results indicate that making reflection explicit and optimizing it directly improves the reliability of tool interaction and offers a reproducible path for agents to learn from failure.

**arXiv ID:** 2509.18847
</details>

<details>
<summary><strong>Tackling GNARLy Problems: Graph Neural Algorithmic Reasoning Reimagined through Reinforcement Learning</strong> - Alex Schutz, Victor-Alexandru Darvariu, Efimia Panagiotaki, Bruno Lacerda, Nick Hawes - [[pdf]](https://arxiv.org/pdf/2509.18930)</summary>

**Abstract:** Neural Algorithmic Reasoning (NAR) is a paradigm that trains neural networks to execute classic algorithms by supervised learning. Despite its successes, important limitations remain: inability to construct valid solutions without post-processing and to reason about multiple correct ones, poor performance on combinatorial NP-hard problems, and inapplicability to problems for which strong algorithms are not yet known. To address these limitations, we reframe the problem of learning algorithm trajectories as a Markov Decision Process, which imposes structure on the solution construction procedure and unlocks the powerful tools of imitation and reinforcement learning (RL). We propose the GNARL framework, encompassing the methodology to translate problem formulations from NAR to RL and a learning architecture suitable for a wide range of graph-based problems. We achieve very high graph accuracy results on several CLRS-30 problems, performance matching or exceeding much narrower NAR approaches for NP-hard problems and, remarkably, applicability even when lacking an expert algorithm.

**arXiv ID:** 2509.18930
</details>

<details>
<summary><strong>Reduced-Order Model-Guided Reinforcement Learning for Demonstration-Free Humanoid Locomotion</strong> - Shuai Liu, Meng Cheng Lau - [[pdf]](https://arxiv.org/pdf/2509.19023)</summary>

**Abstract:** We introduce Reduced-Order Model-Guided Reinforcement Learning (ROM-GRL), a two-stage reinforcement learning framework for humanoid walking that requires no motion capture data or elaborate reward shaping. In the first stage, a compact 4-DOF (four-degree-of-freedom) reduced-order model (ROM) is trained via Proximal Policy Optimization. This generates energy-efficient gait templates. In the second stage, those dynamically consistent trajectories guide a full-body policy trained with Soft Actor--Critic augmented by an adversarial discriminator, ensuring the student's five-dimensional gait feature distribution matches the ROM's demonstrations. Experiments at 1 meter-per-second and 4 meter-per-second show that ROM-GRL produces stable, symmetric gaits with substantially lower tracking error than a pure-reward baseline. By distilling lightweight ROM guidance into high-dimensional policies, ROM-GRL bridges the gap between reward-only and imitation-based locomotion methods, enabling versatile, naturalistic humanoid behaviors without any human demonstrations.

**arXiv ID:** 2509.19023
</details>

<details>
<summary><strong>World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation</strong> - Zhennan Jiang, Kai Liu, Yuxin Qin, Shuai Tian, Yupeng Zheng, Mingcai Zhou, Chao Yu, Haoran Li, Dongbin Zhao - [[pdf]](https://arxiv.org/pdf/2509.19080)</summary>

**Abstract:** Robotic manipulation policies are commonly initialized through imitation learning, but their performance is limited by the scarcity and narrow coverage of expert data. Reinforcement learning can refine polices to alleviate this limitation, yet real-robot training is costly and unsafe, while training in simulators suffers from the sim-to-real gap. Recent advances in generative models have demonstrated remarkable capabilities in real-world simulation, with diffusion models in particular excelling at generation. This raises the question of how diffusion model-based world models can be combined to enhance pre-trained policies in robotic manipulation. In this work, we propose World4RL, a framework that employs diffusion-based world models as high-fidelity simulators to refine pre-trained policies entirely in imagined environments for robotic manipulation. Unlike prior works that primarily employ world models for planning, our framework enables direct end-to-end policy optimization. World4RL is designed around two principles: pre-training a diffusion world model that captures diverse dynamics on multi-task datasets and refining policies entirely within a frozen world model to avoid online real-world interactions. We further design a two-hot action encoding scheme tailored for robotic manipulation and adopt diffusion backbones to improve modeling fidelity. Extensive simulation and real-world experiments demonstrate that World4RL provides high-fidelity environment modeling and enables consistent policy refinement, yielding significantly higher success rates compared to imitation learning and other baselines. More visualization results are available at this https URL.

**arXiv ID:** 2509.19080
</details>

<details>
<summary><strong>Reinforcement Learning on Pre-Training Data</strong> - Siheng Li, Kejiao Li, Zenan Xu, Guanhua Huang, Evander Yang, Kun Li, Haoyuan Wu, Jiajia Wu, Zihao Zheng, Chenchen Zhang, Kun Shi, Kyrierl Deng, Qi Yi, Ruibin Xiong, Tingqiang Xu, Yuhao Jiang, Jianfeng Yan, Yuyuan Zeng, Guanghui Xu, Jinbao Xue, Zhijiang Xu, Zheng Fang, Shuai Li, Qibin Liu, Xiaoxue Li, Zhuoyu Li, Yangyu Tao, Fei Gao, Cheng Jiang, Bo Chao Wang, Kai Liu, Jianchen Zhu, Wai Lam, Wayyt Wang, Bo Zhou, Di Wang - [[pdf]](https://arxiv.org/pdf/2509.19249)</summary>

**Abstract:** The growing disparity between the exponential scaling of computational resources and the finite growth of high-quality text data now constrains conventional scaling approaches for large language models (LLMs). To address this challenge, we introduce Reinforcement Learning on Pre-Training data (RLPT), a new training-time scaling paradigm for optimizing LLMs. In contrast to prior approaches that scale training primarily through supervised learning, RLPT enables the policy to autonomously explore meaningful trajectories to learn from pre-training data and improve its capability through reinforcement learning (RL). While existing RL strategies such as reinforcement learning from human feedback (RLHF) and reinforcement learning with verifiable rewards (RLVR) rely on human annotation for reward construction, RLPT eliminates this dependency by deriving reward signals directly from pre-training data. Specifically, it adopts a next-segment reasoning objective, rewarding the policy for accurately predicting subsequent text segments conditioned on the preceding context. This formulation allows RL to be scaled on pre-training data, encouraging the exploration of richer trajectories across broader contexts and thereby fostering more generalizable reasoning skills. Extensive experiments on both general-domain and mathematical reasoning benchmarks across multiple models validate the effectiveness of RLPT. For example, when applied to Qwen3-4B-Base, RLPT yields absolute improvements of $3.0$, $5.1$, $8.1$, $6.0$, $6.6$, and $5.3$ on MMLU, MMLU-Pro, GPQA-Diamond, KOR-Bench, AIME24, and AIME25, respectively. The results further demonstrate favorable scaling behavior, suggesting strong potential for continued gains with more compute. In addition, RLPT provides a solid foundation, extending the reasoning boundaries of LLMs and enhancing RLVR performance.

**arXiv ID:** 2509.19249
</details>

<details>
<summary><strong>ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning</strong> - Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z. Pan, Wen Zhang, Huajun Chen, Fan Yang, Zenan Zhou, Weipeng Chen - [[pdf]](https://arxiv.org/pdf/2503.19470)</summary>

**Abstract:** Large Language Models (LLMs) have shown remarkable capabilities in reasoning, exemplified by the success of OpenAI-o1 and DeepSeek-R1. However, integrating reasoning with external search processes remains challenging, especially for complex multi-hop questions requiring multiple retrieval steps. We propose ReSearch, a novel framework that trains LLMs to Reason with Search via reinforcement learning without using any supervised data on reasoning steps. Our approach treats search operations as integral components of the reasoning chain, where when and how to perform searches is guided by text-based thinking, and search results subsequently influence further reasoning. We train ReSearch on Qwen2.5-7B(-Instruct) and Qwen2.5-32B(-Instruct) models and conduct extensive experiments. Despite being trained on only one dataset, our models demonstrate strong generalizability across various benchmarks. Analysis reveals that ReSearch naturally elicits advanced reasoning capabilities such as reflection and self-correction during the reinforcement learning process.

**arXiv ID:** 2503.19470
</details>

<details>
<summary><strong>One Subgoal at a Time: Zero-Shot Generalization to Arbitrary Linear Temporal Logic Requirements in Multi-Task Reinforcement Learning</strong> - Zijian Guo, İlker Işık, H. M. Sabbir Ahmad, Wenchao Li - [[pdf]](https://arxiv.org/pdf/2508.01561)</summary>

**Abstract:** Generalizing to complex and temporally extended task objectives and safety constraints remains a critical challenge in reinforcement learning (RL). Linear temporal logic (LTL) offers a unified formalism to specify such requirements, yet existing methods are limited in their abilities to handle nested long-horizon tasks and safety constraints, and cannot identify situations when a subgoal is not satisfiable and an alternative should be sought. In this paper, we introduce GenZ-LTL, a method that enables zero-shot generalization to arbitrary LTL specifications. GenZ-LTL leverages the structure of Büchi automata to decompose an LTL task specification into sequences of reach-avoid subgoals. Contrary to the current state-of-the-art method that conditions on subgoal sequences, we show that it is more effective to achieve zero-shot generalization by solving these reach-avoid problems \textit{one subgoal at a time} through proper safe RL formulations. In addition, we introduce a novel subgoal-induced observation reduction technique that can mitigate the exponential complexity of subgoal-state combinations under realistic assumptions. Empirical results show that GenZ-LTL substantially outperforms existing methods in zero-shot generalization to unseen LTL specifications.

**arXiv ID:** 2508.01561
</details>

<details>
<summary><strong>TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning</strong> - Chuang Jiang, Mingyue Cheng, Xiaoyu Tao, Qingyang Mao, Jie Ouyang, Qi Liu - [[pdf]](https://arxiv.org/pdf/2509.06278)</summary>

**Abstract:** Table reasoning is crucial for leveraging structured data in domains such as finance, healthcare, and scientific research. While large language models (LLMs) show promise in multi-step reasoning, purely text-based methods often struggle with the complex numerical computations and fine-grained operations inherently required in this task. Tool-integrated reasoning improves computational accuracy via explicit code execution, yet existing systems frequently rely on rigid patterns, supervised imitation, and lack true autonomous adaptability. In this paper, we present TableMind, an LLM-driven table reasoning agent that (i) autonomously performs multi-turn tool invocation, (ii) writes and executes data-analyzing code in a secure sandbox environment for data analysis and precise numerical reasoning, and (iii) exhibits high-level capabilities such as planning and self-reflection to adapt strategies. To realize these capabilities, we adopt a two-stage fine-tuning paradigm built on top of a powerful pre-trained language model: supervised fine-tuning on high-quality reasoning trajectories to establish effective tool usage patterns, followed by reinforcement fine-tuning to optimize multi-objective strategies. In particular, we propose Rank-Aware Policy Optimization (RAPO), which increases the update weight of high-quality trajectories when their output probabilities are lower than those of low-quality ones, thereby guiding the model more consistently toward better and more accurate answers. Extensive experiments on several mainstream benchmarks demonstrate that TableMind achieves superior performance compared to competitive baselines, yielding substantial gains in both reasoning accuracy and computational precision.

**arXiv ID:** 2509.06278
</details>

<details>
<summary><strong>A deep reinforcement learning platform for antibiotic discovery</strong> - Hanqun Cao, Marcelo D. T. Torres, Jingjie Zhang, Zijun Gao, Fang Wu, Chunbin Gu, Jure Leskovec, Yejin Choi, Cesar de la Fuente-Nunez, Guangyong Chen, Pheng-Ann Heng - [[pdf]](https://arxiv.org/pdf/2509.18153)</summary>

**Abstract:** Antimicrobial resistance (AMR) is projected to cause up to 10 million deaths annually by 2050, underscoring the urgent need for new antibiotics. Here we present ApexAmphion, a deep-learning framework for de novo design of antibiotics that couples a 6.4-billion-parameter protein language model with reinforcement learning. The model is first fine-tuned on curated peptide data to capture antimicrobial sequence regularities, then optimised with proximal policy optimization against a composite reward that combines predictions from a learned minimum inhibitory concentration (MIC) classifier with differentiable physicochemical objectives. In vitro evaluation of 100 designed peptides showed low MIC values (nanomolar range in some cases) for all candidates (100% hit rate). Moreover, 99 our of 100 compounds exhibited broad-spectrum antimicrobial activity against at least two clinically relevant bacteria. The lead molecules killed bacteria primarily by potently targeting the cytoplasmic membrane. By unifying generation, scoring and multi-objective optimization with deep reinforcement learning in a single pipeline, our approach rapidly produces diverse, potent candidates, offering a scalable route to peptide antibiotics and a platform for iterative steering toward potency and developability within hours.

**arXiv ID:** 2509.18153
</details>

<details>
<summary><strong>Towards Provable Emergence of In-Context Reinforcement Learning</strong> - Jiuqi Wang, Rohan Chandra, Shangtong Zhang - [[pdf]](https://arxiv.org/pdf/2509.18389)</summary>

**Abstract:** Typically, a modern reinforcement learning (RL) agent solves a task by updating its neural network parameters to adapt its policy to the task. Recently, it has been observed that some RL agents can solve a wide range of new out-of-distribution tasks without parameter updates after pretraining on some task distribution. When evaluated in a new task, instead of making parameter updates, the pretrained agent conditions its policy on additional input called the context, e.g., the agent's interaction history in the new task. The agent's performance increases as the information in the context increases, with the agent's parameters fixed. This phenomenon is typically called in-context RL (ICRL). The pretrained parameters of the agent network enable the remarkable ICRL phenomenon. However, many ICRL works perform the pretraining with standard RL algorithms. This raises the central question this paper aims to address: Why can the RL pretraining algorithm generate network parameters that enable ICRL? We hypothesize that the parameters capable of ICRL are minimizers of the pretraining loss. This work provides initial support for this hypothesis through a case study. In particular, we prove that when a Transformer is pretrained for policy evaluation, one of the global minimizers of the pretraining loss can enable in-context temporal difference learning.

**arXiv ID:** 2509.18389
</details>

<details>
<summary><strong>Diffusion Policies with Offline and Inverse Reinforcement Learning for Promoting Physical Activity in Older Adults Using Wearable Sensors</strong> - Chang Liu, Ladda Thiamwong, Yanjie Fu, Rui Xie - [[pdf]](https://arxiv.org/pdf/2509.18433)</summary>

**Abstract:** Utilizing offline reinforcement learning (RL) with real-world clinical data is getting increasing attention in AI for healthcare. However, implementation poses significant challenges. Defining direct rewards is difficult, and inverse RL (IRL) struggles to infer accurate reward functions from expert behavior in complex environments. Offline RL also encounters challenges in aligning learned policies with observed human behavior in healthcare applications. To address challenges in applying offline RL to physical activity promotion for older adults at high risk of falls, based on wearable sensor activity monitoring, we introduce Kolmogorov-Arnold Networks and Diffusion Policies for Offline Inverse Reinforcement Learning (KANDI). By leveraging the flexible function approximation in Kolmogorov-Arnold Networks, we estimate reward functions by learning free-living environment behavior from low-fall-risk older adults (experts), while diffusion-based policies within an Actor-Critic framework provide a generative approach for action refinement and efficiency in offline RL. We evaluate KANDI using wearable activity monitoring data in a two-arm clinical trial from our Physio-feedback Exercise Program (PEER) study, emphasizing its practical application in a fall-risk intervention program to promote physical activity among older adults. Additionally, KANDI outperforms state-of-the-art methods on the D4RL benchmark. These results underscore KANDI's potential to address key challenges in offline RL for healthcare applications, offering an effective solution for activity promotion intervention strategies in healthcare.

**arXiv ID:** 2509.18433
</details>

<details>
<summary><strong>LLM-Enhanced Self-Evolving Reinforcement Learning for Multi-Step E-Commerce Payment Fraud Risk Detection</strong> - Bo Qu, Zhurong Wang, Daisuke Yagi, Zhen Xu, Yang Zhao, Yinan Shan, Frank Zahradnik - [[pdf]](https://arxiv.org/pdf/2509.18719)</summary>

**Abstract:** This paper presents a novel approach to e-commerce payment fraud detection by integrating reinforcement learning (RL) with Large Language Models (LLMs). By framing transaction risk as a multi-step Markov Decision Process (MDP), RL optimizes risk detection across multiple payment stages. Crafting effective reward functions, essential for RL model success, typically requires significant human expertise due to the complexity and variability in design. LLMs, with their advanced reasoning and coding capabilities, are well-suited to refine these functions, offering improvements over traditional methods. Our approach leverages LLMs to iteratively enhance reward functions, achieving better fraud detection accuracy and demonstrating zero-shot capability. Experiments with real-world data confirm the effectiveness, robustness, and resilience of our LLM-enhanced RL framework through long-term evaluations, underscoring the potential of LLMs in advancing industrial RL applications.

**arXiv ID:** 2509.18719
</details>

<details>
<summary><strong>PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generatio</strong> - Alexandre Piché, Ehsan Kamaloo, Rafael Pardinas, Dzmitry Bahdanau - [[pdf]](https://arxiv.org/pdf/2509.19128)</summary>

**Abstract:** Reinforcement Learning (RL) is increasingly utilized to enhance the reasoning capabilities of Large Language Models (LLMs). However, effectively scaling these RL methods presents significant challenges, primarily due to the difficulty in maintaining high AI accelerator utilization without generating stale, off-policy data that harms common RL algorithms. This paper introduces PipelineRL, an approach designed to achieve a superior trade-off between hardware efficiency and data on-policyness for LLM training. PipelineRL employs concurrent asynchronous data generation and model training, distinguished by the novel in-flight weight updates. This mechanism allows the LLM generation engine to receive updated model weights with minimal interruption during the generation of token sequences, thereby maximizing both the accelerator utilization and the freshness of training data. Experiments conducted on long-form reasoning tasks using 128 H100 GPUs demonstrate that PipelineRL achieves approximately $\sim 2x$ faster learning compared to conventional RL baselines while maintaining highly on-policy training data. A scalable and modular open-source implementation of PipelineRL is also released as a key contribution.

**arXiv ID:** 2509.19128
</details>

<details>
<summary><strong>Efficient Reinforcement Learning by Reducing Forgetting with Elephant Activation Functions</strong> - Qingfeng Lan, Gautham Vasan, A. Rupam Mahmood - [[pdf]](https://arxiv.org/pdf/2509.19159)</summary>

**Abstract:** Catastrophic forgetting has remained a significant challenge for efficient reinforcement learning for decades (Ring 1994, Rivest and Precup 2003). While recent works have proposed effective methods to mitigate this issue, they mainly focus on the algorithmic side. Meanwhile, we do not fully understand what architectural properties of neural networks lead to catastrophic forgetting. This study aims to fill this gap by studying the role of activation functions in the training dynamics of neural networks and their impact on catastrophic forgetting in reinforcement learning setup. Our study reveals that, besides sparse representations, the gradient sparsity of activation functions also plays an important role in reducing forgetting. Based on this insight, we propose a new class of activation functions, elephant activation functions, that can generate both sparse outputs and sparse gradients. We show that by simply replacing classical activation functions with elephant activation functions in the neural networks of value-based algorithms, we can significantly improve the resilience of neural networks to catastrophic forgetting, thus making reinforcement learning more sample-efficient and memory-efficient.

**arXiv ID:** 2509.19159
</details>

<details>
<summary><strong>Robotic Skill Diversification via Active Mutation of Reward Functions in Reinforcement Learning During a Liquid Pouring Task</strong> - Jannick van Buuren, Roberto Giglio, Loris Roveda, Luka Peternel - [[pdf]](https://arxiv.org/pdf/2509.18463)</summary>

**Abstract:** This paper explores how deliberate mutations of reward function in reinforcement learning can produce diversified skill variations in robotic manipulation tasks, examined with a liquid pouring use case. To this end, we developed a new reward function mutation framework that is based on applying Gaussian noise to the weights of the different terms in the reward function. Inspired by the cost-benefit tradeoff model from human motor control, we designed the reward function with the following key terms: accuracy, time, and effort. The study was performed in a simulation environment created in NVIDIA Isaac Sim, and the setup included Franka Emika Panda robotic arm holding a glass with a liquid that needed to be poured into a container. The reinforcement learning algorithm was based on Proximal Policy Optimization. We systematically explored how different configurations of mutated weights in the rewards function would affect the learned policy. The resulting policies exhibit a wide range of behaviours: from variations in execution of the originally intended pouring task to novel skills useful for unexpected tasks, such as container rim cleaning, liquid mixing, and watering. This approach offers promising directions for robotic systems to perform diversified learning of specific tasks, while also potentially deriving meaningful skills for future tasks.

**arXiv ID:** 2509.18463
</details>

<details>
<summary><strong>PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable Reinforcement Learning</strong> - Dongchi Huang, Jiaqi Wang, Yang Li, Chunhe Xia, Tianle Zhang, Kaige Zhang - [[pdf]](https://arxiv.org/pdf/2508.02159)</summary>

**Abstract:** Partial observability presents a significant challenge for Safe Reinforcement Learning (Safe RL), as it impedes the identification of potential risks and rewards. Leveraging specific types of privileged information during training to mitigate the effects of partial observability has yielded notable empirical successes. In this paper, we propose Asymmetric Constrained Partially Observable Markov Decision Processes (ACPOMDPs) to theoretically examine the advantages of incorporating privileged information in Safe RL. Building upon ACPOMDPs, we propose the Privileged Information Guided Dreamer (PIGDreamer), a model-based RL approach that leverages privileged information to enhance the agent's safety and performance through privileged representation alignment and an asymmetric actor-critic structure. Our empirical results demonstrate that PIGDreamer significantly outperforms existing Safe RL methods. Furthermore, compared to alternative privileged RL methods, our approach exhibits enhanced performance, robustness, and efficiency. Codes are available at: this https URL.

**arXiv ID:** 2508.02159
</details>

<details>
<summary><strong>EvoCoT: Overcoming the Exploration Bottleneck in Reinforcement Learning</strong> - Huanyu Liu, Jia Li, Chang Yu, Taozhi Chen, Yihong Dong, Lecheng Wang, XiaoLong Hu, Ge Li - [[pdf]](https://arxiv.org/pdf/2508.07809)</summary>

**Abstract:** Reinforcement learning with verifiable reward (RLVR) has become a promising paradigm for post-training large language models (LLMs) to improve their reasoning capability. However, when the rollout accuracy is low on hard problems, the reward becomes sparse, limiting learning efficiency and causing exploration bottlenecks. Existing approaches either rely on teacher models for distillation or filter out difficult problems, which limits scalability or restricts reasoning improvement through exploration.
We propose EvoCoT, a self-evolving curriculum learning framework based on two-stage chain-of-thought (CoT) reasoning optimization. EvoCoT constrains the exploration space by self-generating and verifying CoT trajectories, then gradually shortens CoT steps to expand the space in a controlled way. The framework enables LLMs to stably learn from initially unsolved hard problems under sparse rewards. We apply EvoCoT to multiple LLM families, including Qwen, DeepSeek, and Llama. Experiments show that EvoCoT enables LLMs to solve previously unsolved problems, improves reasoning capability without external CoT supervision, and is compatible with various RL fine-tuning methods. We release the source code to support future research.

**arXiv ID:** 2508.07809
</details>

<details>
<summary><strong>Growing with Your Embodied Agent: A Human-in-the-Loop Lifelong Code Generation Framework for Long-Horizon Manipulation Skills</strong> - Yuan Meng, Zhenguo Sun, Max Fest, Xukun Li, Zhenshan Bing, Alois Knoll - [[pdf]](https://arxiv.org/pdf/2509.18597)</summary>

**Abstract:** Large language models (LLMs)-based code generation for robotic manipulation has recently shown promise by directly translating human instructions into executable code, but existing methods remain noisy, constrained by fixed primitives and limited context windows, and struggle with long-horizon tasks. While closed-loop feedback has been explored, corrected knowledge is often stored in improper formats, restricting generalization and causing catastrophic forgetting, which highlights the need for learning reusable skills. Moreover, approaches that rely solely on LLM guidance frequently fail in extremely long-horizon scenarios due to LLMs' limited reasoning capability in the robotic domain, where such issues are often straightforward for humans to identify. To address these challenges, we propose a human-in-the-loop framework that encodes corrections into reusable skills, supported by external memory and Retrieval-Augmented Generation with a hint mechanism for dynamic reuse. Experiments on Ravens, Franka Kitchen, and MetaWorld, as well as real-world settings, show that our framework achieves a 0.93 success rate (up to 27% higher than baselines) and a 42% efficiency improvement in correction rounds. It can robustly solve extremely long-horizon tasks such as "build a house", which requires planning over 20 primitives.

**arXiv ID:** 2509.18597
</details>

<details>
<summary><strong>Embodied Arena: A Comprehensive, Unified, and Evolving Evaluation Platform for Embodied AI</strong> - Fei Ni, Min Zhang, Pengyi Li, Yifu Yuan, Lingfeng Zhang, Yuecheng Liu, Peilong Han, Longxin Kou, Shaojin Ma, Jinbin Qiao, David Gamaliel Arcos Bravo, Yuening Wang, Xiao Hu, Zhanguang Zhang, Xianze Yao, Yutong Li, Zhao Zhang, Ying Wen, Ying-Cong Chen, Xiaodan Liang, Liang Lin, Bin He, Haitham Bou-Ammar, He Wang, Huazhe Xu, Jiankang Deng, Shan Luo, Shuqiang Jiang, Wei Pan, Yang Gao, Stefanos Zafeiriou, Jan Peters, Yuzheng Zhuang, Yingxue Zhang, Yan Zheng, Hongyao Tang, Jianye Hao - [[pdf]](https://arxiv.org/pdf/2509.15273)</summary>

**Abstract:** Embodied AI development significantly lags behind large foundation models due to three critical challenges: (1) lack of systematic understanding of core capabilities needed for Embodied AI, making research lack clear objectives; (2) absence of unified and standardized evaluation systems, rendering cross-benchmark evaluation infeasible; and (3) underdeveloped automated and scalable acquisition methods for embodied data, creating critical bottlenecks for model scaling. To address these obstacles, we present Embodied Arena, a comprehensive, unified, and evolving evaluation platform for Embodied AI. Our platform establishes a systematic embodied capability taxonomy spanning three levels (perception, reasoning, task execution), seven core capabilities, and 25 fine-grained dimensions, enabling unified evaluation with systematic research objectives. We introduce a standardized evaluation system built upon unified infrastructure supporting flexible integration of 22 diverse benchmarks across three domains (2D/3D Embodied Q&A, Navigation, Task Planning) and 30+ advanced models from 20+ worldwide institutes. Additionally, we develop a novel LLM-driven automated generation pipeline ensuring scalable embodied evaluation data with continuous evolution for diversity and comprehensiveness. Embodied Arena publishes three real-time leaderboards (Embodied Q&A, Navigation, Task Planning) with dual perspectives (benchmark view and capability view), providing comprehensive overviews of advanced model capabilities. Especially, we present nine findings summarized from the evaluation results on the leaderboards of Embodied Arena. This helps to establish clear research veins and pinpoint critical research problems, thereby driving forward progress in the field of Embodied AI.

**arXiv ID:** 2509.15273
</details>

<details>
<summary><strong>XARP Tools: An Extended Reality Platform for Humans and AI Agents</strong> - Arthur Caetano, Radha Kumaran, Kelvin Jou, Tobias Höllerer, Misha Sra - [[pdf]](https://arxiv.org/pdf/2508.04108)</summary>

**Abstract:** Artificial intelligence (AI) and extended reality (XR) are increasingly combined in applications such as motor skill training, personalized feedback, and embodied task guidance. Yet developing AI-XR systems remains challenging due to fragmented toolchains that push developers into ad hoc integrations, diverting their attention away from essential design concerns such as interactivity and context awareness. To address this issue, we present XARP (XR Agent-ready Remote Procedures), a toolkit for AI-XR development designed for both human developers and AI agents. XARP implements JSON-based remote procedure calls that allow server-side Python to control XR clients, providing a high-level abstraction over low-level integration details. Humans can use XARP as a Python library to write XR applications with reduced implementation overhead. AI agents operate with the same abstraction to dynamically call tools to generate XR applications at runtime in response to context changes and user requests. XARP offers Model Context Protocol (MCP) connectivity that allows third-party agents and tools to leverage XR capabilities, previously unavailable. We conducted three case studies that demonstrate XARP supports a variety of AI-XR applications, including AI-guided fencing, drone assistance, and room layout design. We evaluated XARP in a walkthrough study with 24 AI and XR developers. UTAUT scores indicate high potential for adoption, and participants reported that XARP can reduce authoring time, lower entry barriers for developers unfamiliar with AI or XR, and enable the implementation of novel AI-XR systems.

**arXiv ID:** 2508.04108
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
