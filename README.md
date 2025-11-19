# Agent arXiv Daily

**Last Updated:** 2025-11-19 02:49:51

**Total Papers:** 74

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
<summary><strong>From Legacy Fortran to Portable Kokkos: An Autonomous Agentic AI Workflow</strong> - Sparsh Gupta, Kamalavasan Kamalakkannan, Maxim Moraru, Galen Shipman, Patrick Diehl - [[pdf]](https://arxiv.org/pdf/2509.12443)</summary>

**Abstract:** Scientific applications continue to rely on legacy Fortran codebases originally developed for homogeneous, CPU-based systems. As High-Performance Computing (HPC) shifts toward heterogeneous GPU-accelerated architectures, many accelerators lack native Fortran bindings, creating an urgent need to modernize legacy codes for portability. Frameworks like Kokkos provide performance portability and a single-source C++ abstraction, but manual Fortran-to-Kokkos porting demands significant expertise and time. Large language models (LLMs) have shown promise in source-to-source code generation, yet their use in fully autonomous workflows for translating and optimizing parallel code remains largely unexplored, especially for performance portability across diverse hardware. This paper presents an agentic AI workflow where specialized LLM "agents" collaborate to translate, validate, compile, run, test, debug, and optimize Fortran kernels into portable Kokkos C++ programs. Results show the pipeline modernizes a range of benchmark kernels, producing performance-portable Kokkos codes across hardware partitions. Paid OpenAI models such as GPT-5 and o4-mini-high executed the workflow for only a few U.S. dollars, generating optimized codes that surpassed Fortran baselines, whereas open-source models like Llama4-Maverick often failed to yield functional codes. This work demonstrates the feasibility of agentic AI for Fortran-to-Kokkos transformation and offers a pathway for autonomously modernizing legacy scientific applications to run portably and efficiently on diverse supercomputers. It further highlights the potential of LLM-driven agentic systems to perform structured, domain-specific reasoning tasks in scientific and systems-oriented applications.

**arXiv ID:** 2509.12443
</details>

<details>
<summary><strong>Watchdogs and Oracles: Runtime Verification Meets Large Language Models for Autonomous Systems</strong> - Angelo Ferrando - [[pdf]](https://arxiv.org/pdf/2511.14435)</summary>

**Abstract:** 

**arXiv ID:** 2511.14435
</details>

<details>
<summary><strong>Xiangqi-R1: Enhancing Spatial Strategic Reasoning in LLMs for Chinese Chess via Reinforcement Learning</strong> - Yuhao Chen, Shuochen Liu, Yuanjie Lyu, Chao Zhang, Jiayao Shi, Tong Xu - [[pdf]](https://arxiv.org/pdf/2507.12215)</summary>

**Abstract:** Game playing has long served as a fundamental benchmark for evaluating Artificial General Intelligence. While Large Language Models (LLMs) have demonstrated impressive capabilities in general reasoning, their effectiveness in spatial strategic reasoning, which is critical for complex and fully observable board games, remains insufficiently explored. In this work, we adopt Chinese Chess (Xiangqi) as a challenging and rich testbed due to its intricate rules and spatial complexity. To advance LLMs' strategic competence in such environments, we propose a training framework tailored to Xiangqi, built upon a large-scale dataset of five million board-move pairs enhanced with expert annotations and engine evaluations. Building on this foundation, we introduce Xiangqi-R1, a 7B-parameter model trained in multi-stage manner. Our Experimental results indicate that, despite their size and power, general-purpose LLMs struggle to achieve satisfactory performance in these tasks. Compared to general-purpose LLMs, Xiangqi-R1 greatly advances with an 18% rise in move legality and a 22% boost in analysis accuracy. Our results point to a promising path for creating general strategic intelligence in complex areas.

**arXiv ID:** 2507.12215
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (13 papers)</h2></summary>

<details>
<summary><strong>Jailbreaking Large Vision Language Models in Intelligent Transportation Systems</strong> - Badhan Chandra Das, Md Tasnim Jawad, Md Jueal Mia, M. Hadi Amini, Yanzhao Wu - [[pdf]](https://arxiv.org/pdf/2511.13892)</summary>

**Abstract:** Large Vision Language Models (LVLMs) demonstrate strong capabilities in multimodal reasoning and many real-world applications, such as visual question answering. However, LVLMs are highly vulnerable to jailbreaking attacks. This paper systematically analyzes the vulnerabilities of LVLMs integrated in Intelligent Transportation Systems (ITS) under carefully crafted jailbreaking attacks. First, we carefully construct a dataset with harmful queries relevant to transportation, following OpenAI's prohibited categories to which the LVLMs should not respond. Second, we introduce a novel jailbreaking attack that exploits the vulnerabilities of LVLMs through image typography manipulation and multi-turn prompting. Third, we propose a multi-layered response filtering defense technique to prevent the model from generating inappropriate responses. We perform extensive experiments with the proposed attack and defense on the state-of-the-art LVLMs (both open-source and closed-source). To evaluate the attack method and defense technique, we use GPT-4's judgment to determine the toxicity score of the generated responses, as well as manual verification. Further, we compare our proposed jailbreaking method with existing jailbreaking techniques and highlight severe security risks involved with jailbreaking attacks with image typography manipulation and multi-turn prompting in the LVLMs integrated in ITS.

**arXiv ID:** 2511.13892
</details>

<details>
<summary><strong>Beyond Accuracy: A Multi-Dimensional Framework for Evaluating Enterprise Agentic AI Systems</strong> - Sushant Mehta - [[pdf]](https://arxiv.org/pdf/2511.14136)</summary>

**Abstract:** Current agentic AI benchmarks predominantly evaluate task completion accuracy, while overlooking critical enterprise requirements such as cost-efficiency, reliability, and operational stability. Through systematic analysis of 12 main benchmarks and empirical evaluation of state-of-the-art agents, we identify three fundamental limitations: (1) absence of cost-controlled evaluation leading to 50x cost variations for similar precision, (2) inadequate reliability assessment where agent performance drops from 60\% (single run) to 25\% (8-run consistency), and (3) missing multidimensional metrics for security, latency, and policy compliance. We propose \textbf{CLEAR} (Cost, Latency, Efficacy, Assurance, Reliability), a holistic evaluation framework specifically designed for enterprise deployment. Evaluation of six leading agents on 300 enterprise tasks demonstrates that optimizing for accuracy alone yields agents 4.4-10.8x more expensive than cost-aware alternatives with comparable performance. Expert evaluation (N=15) confirms that CLEAR better predicts production success (correlation $\rho=0.83$) compared to accuracy-only evaluation ($\rho=0.41$).

**arXiv ID:** 2511.14136
</details>

<details>
<summary><strong>Orion: A Unified Visual Agent for Multimodal Perception, Advanced Visual Reasoning and Execution</strong> - N Dinesh Reddy, Sudeep Pillai - [[pdf]](https://arxiv.org/pdf/2511.14210)</summary>

**Abstract:** We introduce Orion, a visual agent framework that can take in any modality and generate any modality. Using an agentic framework with multiple tool-calling capabilities, Orion is designed for visual AI tasks and achieves state-of-the-art results. Unlike traditional vision-language models that produce descriptive outputs, Orion orchestrates a suite of specialized computer vision tools, including object detection, keypoint localization, panoptic segmentation, Optical Character Recognition, and geometric analysis, to execute complex multi-step visual workflows. The system achieves competitive performance on MMMU, MMBench, DocVQA, and MMLongBench while extending monolithic vision-language models to production-grade visual intelligence. By combining neural perception with symbolic execution, Orion enables autonomous visual reasoning, marking a transition from passive visual understanding to active, tool-driven visual intelligence.

**arXiv ID:** 2511.14210
</details>

<details>
<summary><strong>Cheating Stereo Matching in Full-scale: Physical Adversarial Attack against Binocular Depth Estimation in Autonomous Driving</strong> - Kangqiao Zhao, Shuo Huai, Xurui Song, Jun Luo - [[pdf]](https://arxiv.org/pdf/2511.14386)</summary>

**Abstract:** Though deep neural models adopted to realize the perception of autonomous driving have proven vulnerable to adversarial examples, known attacks often leverage 2D patches and target mostly monocular perception. Therefore, the effectiveness of Physical Adversarial Examples (PAEs) on stereo-based binocular depth estimation remains largely unexplored. To this end, we propose the first texture-enabled physical adversarial attack against stereo matching models in the context of autonomous driving. Our method employs a 3D PAE with global camouflage texture rather than a local 2D patch-based one, ensuring both visual consistency and attack effectiveness across different viewpoints of stereo cameras. To cope with the disparity effect of these cameras, we also propose a new 3D stereo matching rendering module that allows the PAE to be aligned with real-world positions and headings in binocular vision. We further propose a novel merging attack that seamlessly blends the target into the environment through fine-grained PAE optimization. It has significantly enhanced stealth and lethality upon existing hiding attacks that fail to get seamlessly merged into the background. Extensive evaluations show that our PAEs can successfully fool the stereo models into producing erroneous depth information.

**arXiv ID:** 2511.14386
</details>

<details>
<summary><strong>Agentic AI Systems in Electrical Power Systems Engineering: Current State-of-the-Art and Challenges</strong> - Soham Ghosh, Gaurav Mittal - [[pdf]](https://arxiv.org/pdf/2511.14478)</summary>

**Abstract:** Agentic AI systems have recently emerged as a critical and transformative approach in artificial intelligence, offering capabilities that extend far beyond traditional AI agents and contemporary generative AI models. This rapid evolution necessitates a clear conceptual and taxonomical understanding to differentiate this new paradigm. Our paper addresses this gap by providing a comprehensive review that establishes a precise definition and taxonomy for "agentic AI," with the aim of distinguishing it from previous AI paradigms. The concepts are gradually introduced, starting with a highlight of its diverse applications across the broader field of engineering. The paper then presents four detailed, state-of-the-art use case applications specifically within electrical engineering. These case studies demonstrate practical impact, ranging from an advanced agentic framework for streamlining complex power system studies and benchmarking to a novel system developed for survival analysis of dynamic pricing strategies in battery swapping stations. Finally, to ensure robust deployment, the paper provides detailed failure mode investigations. From these findings, we derive actionable recommendations for the design and implementation of safe, reliable, and accountable agentic AI systems, offering a critical resource for researchers and practitioners.

**arXiv ID:** 2511.14478
</details>

<details>
<summary><strong>Is Your VLM for Autonomous Driving Safety-Ready? A Comprehensive Benchmark for Evaluating External and In-Cabin Risks</strong> - Xianhui Meng, Yuchen Zhang, Zhijian Huang, Zheng Lu, Ziling Ji, Yaoyao Yin, Hongyuan Zhang, Guangfeng Jiang, Yandan Lin, Long Chen, Hangjun Ye, Li Zhang, Jun Liu, Xiaoshuai Hao - [[pdf]](https://arxiv.org/pdf/2511.14592)</summary>

**Abstract:** Vision-Language Models (VLMs) show great promise for autonomous driving, but their suitability for safety-critical scenarios is largely unexplored, raising safety concerns. This issue arises from the lack of comprehensive benchmarks that assess both external environmental risks and in-cabin driving behavior safety simultaneously. To bridge this critical gap, we introduce DSBench, the first comprehensive Driving Safety Benchmark designed to assess a VLM's awareness of various safety risks in a unified manner. DSBench encompasses two major categories: external environmental risks and in-cabin driving behavior safety, divided into 10 key categories and a total of 28 sub-categories. This comprehensive evaluation covers a wide range of scenarios, ensuring a thorough assessment of VLMs' performance in safety-critical contexts. Extensive evaluations across various mainstream open-source and closed-source VLMs reveal significant performance degradation under complex safety-critical situations, highlighting urgent safety concerns. To address this, we constructed a large dataset of 98K instances focused on in-cabin and external safety scenarios, showing that fine-tuning on this dataset significantly enhances the safety performance of existing VLMs and paves the way for advancing autonomous driving technology. The benchmark toolkit, code, and model checkpoints will be publicly accessible.

**arXiv ID:** 2511.14592
</details>

<details>
<summary><strong>AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection</strong> - Peiran Wang, Yang Liu, Yunfei Lu, Yifeng Cai, Hongbo Chen, Qingyou Yang, Jie Zhang, Jue Hong, Ye Wu - [[pdf]](https://arxiv.org/pdf/2508.01249)</summary>

**Abstract:** Large Language Model (LLM) agents offer a powerful new paradigm for solving various problems by combining natural language reasoning with the execution of external tools. However, their dynamic and non-transparent behavior introduces critical security risks, particularly in the presence of prompt injection attacks. In this work, we propose a novel insight that treats the agent runtime traces as structured programs with analyzable semantics. Thus, we present AgentArmor, a program analysis framework that converts agent traces into graph intermediate representation-based structured program dependency representations (e.g., CFG, DFG, and PDG) and enforces security policies via a type system. AgentArmor consists of three key components: (1) a graph constructor that reconstructs the agent's runtime traces as graph-based intermediate representations with control and data flow described within; (2) a property registry that attaches security-relevant metadata of interacted tools \& data, and (3) a type system that performs static inference and checking over the intermediate representation. By representing agent behavior as structured programs, AgentArmor enables program analysis for sensitive data flow, trust boundaries, and policy violations. We evaluate AgentArmor on the AgentDojo benchmark, the results show that AgentArmor can reduce the ASR to 3\%, with the utility drop only 1\%.

**arXiv ID:** 2508.01249
</details>

<details>
<summary><strong>RynnEC: Bringing MLLMs into Embodied World</strong> - Ronghao Dang, Yuqian Yuan, Yunxuan Mao, Kehan Li, Jiangpin Liu, Zhikai Wang, Xin Li, Fan Wang, Deli Zhao - [[pdf]](https://arxiv.org/pdf/2508.14160)</summary>

**Abstract:** We introduce RynnEC, a video multimodal large language model designed for embodied cognition. Built upon a general-purpose vision-language foundation model, RynnEC incorporates a region encoder and a mask decoder, enabling flexible region-level video interaction. Despite its compact architecture, RynnEC achieves state-of-the-art performance in object property understanding, object segmentation, and spatial reasoning. Conceptually, it offers a region-centric video paradigm for the brain of embodied agents, providing fine-grained perception of the physical world and enabling more precise interactions. To mitigate the scarcity of annotated 3D datasets, we propose an egocentric video based pipeline for generating embodied cognition data. Furthermore, we introduce RynnEC-Bench, a region-centered benchmark for evaluating embodied cognitive capabilities. We anticipate that RynnEC will advance the development of general-purpose cognitive cores for embodied agents and facilitate generalization across diverse embodied tasks. The code, model checkpoints, and benchmark are available at: this https URL

**arXiv ID:** 2508.14160
</details>

<details>
<summary><strong>MedBench v4: A Robust and Scalable Benchmark for Evaluating Chinese Medical Language Models, Multimodal Models, and Intelligent Agents</strong> - Jinru Ding, Lu Lu, Chao Ding, Mouxiao Bian, Jiayuan Chen, Renjie Lu, Wenrao Pang, Xiaoqin Wu, Zhiqiang Liu, Luyi Jiang, Bing Han, Yunqiu Wang, Jie Xu - [[pdf]](https://arxiv.org/pdf/2511.14439)</summary>

**Abstract:** 

**arXiv ID:** 2511.14439
</details>

<details>
<summary><strong>Enhancing End-to-End Autonomous Driving with Risk Semantic Distillaion from VLM</strong> - Jack Qin, Zhitao Wang, Yinan Zheng, Keyu Chen, Yang Zhou, Yuanxin Zhong, Siyuan Cheng - [[pdf]](https://arxiv.org/pdf/2511.14499)</summary>

**Abstract:** The autonomous driving (AD) system has exhibited remarkable performance in complex driving scenarios. However, generalization is still a key limitation for the current system, which refers to the ability to handle unseen scenarios or unfamiliar sensor this http URL works have explored the use of Vision-Language Models (VLMs) to address few-shot or zero-shot tasks. While promising, these methods introduce a new challenge: the emergence of a hybrid AD system, where two distinct systems are used to plan a trajectory, leading to potential inconsistencies. Alternative research directions have explored Vision-Language-Action (VLA) frameworks that generate control actions from VLM directly. However, these end-to-end solutions demonstrate prohibitive computational demands. To overcome these challenges, we introduce Risk Semantic Distillation (RSD), a novel framework that leverages VLMs to enhance the training of End-to-End (E2E) AD backbones. By providing risk attention for key objects, RSD addresses the issue of generalization. Specifically, we introduce RiskHead, a plug-in module that distills causal risk estimates from Vision-Language Models into Bird's-Eye-View (BEV) features, yielding interpretable risk-attention this http URL approach allows BEV features to learn richer and more nuanced risk attention representations, which directly enhance the model's ability to handle spatial boundaries and risky this http URL focusing on risk attention, RSD aligns better with human-like driving behavior, which is essential to navigate in complex and dynamic environments. Our experiments on the Bench2Drive benchmark demonstrate the effectiveness of RSD in managing complex and unpredictable driving conditions. Due to the enhanced BEV representations enabled by RSD, we observed a significant improvement in both perception and planning capabilities.

**arXiv ID:** 2511.14499
</details>

<details>
<summary><strong>DepthVision: Enabling Robust Vision-Language Models with GAN-Based LiDAR-to-RGB Synthesis for Autonomous Driving</strong> - Sven Kirchner, Nils Purschke, Ross Greer, Alois C. Knoll - [[pdf]](https://arxiv.org/pdf/2509.07463)</summary>

**Abstract:** Ensuring reliable autonomous operation when visual input is degraded remains a key challenge in intelligent vehicles and robotics. We present DepthVision, a multimodal framework that enables Vision--Language Models (VLMs) to exploit LiDAR data without any architectural changes or retraining. DepthVision synthesizes dense, RGB-like images from sparse LiDAR point clouds using a conditional GAN with an integrated refiner, and feeds these into off-the-shelf VLMs through their standard visual interface. A Luminance-Aware Modality Adaptation (LAMA) module fuses synthesized and real camera images by dynamically weighting each modality based on ambient lighting, compensating for degradation such as darkness or motion blur. This design turns LiDAR into a drop-in visual surrogate when RGB becomes unreliable, effectively extending the operational envelope of existing VLMs. We evaluate DepthVision on real and simulated datasets across multiple VLMs and safety-critical tasks, including vehicle-in-the-loop experiments. The results show substantial improvements in low-light scene understanding over RGB-only baselines while preserving full compatibility with frozen VLM architectures. These findings demonstrate that LiDAR-guided RGB synthesis is a practical pathway for integrating range sensing into modern vision-language systems for autonomous driving.

**arXiv ID:** 2509.07463
</details>

<details>
<summary><strong>StyleDrive: Towards Driving-Style Aware Benchmarking of End-To-End Autonomous Driving</strong> - Ruiyang Hao, Bowen Jing, Haibao Yu, Zaiqing Nie - [[pdf]](https://arxiv.org/pdf/2506.23982)</summary>

**Abstract:** Personalization, while extensively studied in conventional autonomous driving pipelines, has been largely overlooked in the context of end-to-end autonomous driving (E2EAD), despite its critical role in fostering user trust, safety perception, and real-world adoption. A primary bottleneck is the absence of large-scale real-world datasets that systematically capture driving preferences, severely limiting the development and evaluation of personalized E2EAD models. In this work, we introduce the first large-scale real-world dataset explicitly curated for personalized E2EAD, integrating comprehensive scene topology with rich dynamic context derived from agent dynamics and semantics inferred via a fine-tuned vision-language model (VLM). We propose a hybrid annotation pipeline that combines behavioral analysis, rule-and-distribution-based heuristics, and subjective semantic modeling guided by VLM reasoning, with final refinement through human-in-the-loop verification. Building upon this dataset, we introduce the first standardized benchmark for systematically evaluating personalized E2EAD models. Empirical evaluations on state-of-the-art architectures demonstrate that incorporating personalized driving preferences significantly improves behavioral alignment with human demonstrations.

**arXiv ID:** 2506.23982
</details>

<details>
<summary><strong>Final Happiness: What Intelligent User Interfaces Can Do for the lonely Dying</strong> - Yibo Meng, Xiaolan Ding, Lyumanshan Ye, Zhiming Liu, Yan Guan - [[pdf]](https://arxiv.org/pdf/2511.14164)</summary>

**Abstract:** This study explores the design of Intelligent User Interfaces (IUIs) to address the profound existential loneliness of terminally ill individuals. While Human-Computer Interaction (HCI) has made inroads in "Thanatechnology," current research often focuses on practical aspects like digital legacy management, overlooking the subjective, existential needs of those facing death in isolation. To address this gap, we conducted in-depth qualitative interviews with 14 lonely, terminally ill individuals. Our core contributions are: (1) An empirically-grounded model articulating the complex psychological, practical, social, and spiritual needs of this group; (2) The "Three Pillars, Twelve Principles" framework for designing IUIs as "Existential Companions"; and (3) A critical design directive derived from user evaluations: technology in this context should aim for transcendence over simulation. The findings suggest that IUIs should create experiences that augment or surpass human capabilities, rather than attempting to simulate basic human connections, which can paradoxically deepen loneliness. This research provides a clear, user-centered path for designing technology that serves not as a "tool for dying," but as a "partner for living fully until the end".

**arXiv ID:** 2511.14164
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>AutoTool: Efficient Tool Selection for Large Language Model Agents</strong> - Jingyi Jia, Qinbin Li - [[pdf]](https://arxiv.org/pdf/2511.14650)</summary>

**Abstract:** Large Language Model (LLM) agents have emerged as powerful tools for automating complex tasks by leveraging the reasoning and decision-making abilities of LLMs. However, a major bottleneck in current agent frameworks lies in the high inference cost of tool selection, especially in approaches like ReAct that repeatedly invoke the LLM to determine which tool to use at each step. In this work, we propose AutoTool, a novel graph-based framework that bypasses repeated LLM inference by exploiting a key empirical observation: tool usage inertia - the tendency of tool invocations to follow predictable sequential patterns. AutoTool constructs a directed graph from historical agent trajectories, where nodes represent tools and edges capture transition probabilities, effectively modeling the inertia in tool selection. It further integrates parameter-level information to refine tool input generation. By traversing this structured representation, AutoTool efficiently selects tools and their parameters with minimal reliance on LLM inference. Extensive experiments across diverse agent tasks demonstrate that AutoTool reduces inference costs by up to 30% while maintaining competitive task completion rates, offering a practical and scalable enhancement for inference-heavy frameworks. Our work highlights the promise of integrating statistical structure into LLM agent design for greater efficiency without sacrificing performance.

**arXiv ID:** 2511.14650
</details>

<details>
<summary><strong>AI Kill Switch for malicious web-based LLM agent</strong> - Sechan Lee, Sangdon Park - [[pdf]](https://arxiv.org/pdf/2511.13725)</summary>

**Abstract:** Recently, web-based Large Language Model (LLM) agents autonomously perform increasingly complex tasks, thereby bringing significant convenience. However, they also amplify the risks of malicious misuse cases such as unauthorized collection of personally identifiable information (PII), generation of socially divisive content, and even automated web hacking. To address these threats, we propose an AI Kill Switch technique that can immediately halt the operation of malicious web-based LLM agents. To achieve this, we introduce AutoGuard - the key idea is generating defensive prompts that trigger the safety mechanisms of malicious LLM agents. In particular, generated defense prompts are transparently embedded into the website's DOM so that they remain invisible to human users but can be detected by the crawling process of malicious agents, triggering its internal safety mechanisms to abort malicious actions once read. To evaluate our approach, we constructed a dedicated benchmark consisting of three representative malicious scenarios (PII collection, social rift content generation, and web hacking attempts). Experimental results show that the AutoGuard method achieves over 80% Defense Success Rate (DSR) on malicious agents, including GPT-4o, Claude-3, and Llama3.3-70B-Instruct. It also maintains strong performance, achieving around 90% DSR on GPT-5, GPT-4.1, and Gemini-2.5-Flash when used as the malicious agent, demonstrating robust generalization across models and scenarios. Through this research, we have demonstrated the controllability of web-based LLM agents across various scenarios and models, thereby contributing to the broader effort of AI control and safety.

**arXiv ID:** 2511.13725
</details>

<details>
<summary><strong>LoCoBench-Agent: An Interactive Benchmark for LLM Agents in Long-Context Software Engineering</strong> - Jielin Qiu, Zuxin Liu, Zhiwei Liu, Rithesh Murthy, Jianguo Zhang, Haolin Chen, Shiyu Wang, Ming Zhu, Liangwei Yang, Juntao Tan, Roshan Ram, Akshara Prabhakar, Tulika Awalgaonkar, Zixiang Chen, Zhepeng Cen, Cheng Qian, Shelby Heinecke, Weiran Yao, Silvio Savarese, Caiming Xiong, Huan Wang - [[pdf]](https://arxiv.org/pdf/2511.13998)</summary>

**Abstract:** As large language models (LLMs) evolve into sophisticated autonomous agents capable of complex software development tasks, evaluating their real-world capabilities becomes critical. While existing benchmarks like LoCoBench~\cite{qiu2025locobench} assess long-context code understanding, they focus on single-turn evaluation and cannot capture the multi-turn interactive nature, tool usage patterns, and adaptive reasoning required by real-world coding agents. We introduce \textbf{LoCoBench-Agent}, a comprehensive evaluation framework specifically designed to assess LLM agents in realistic, long-context software engineering workflows. Our framework extends LoCoBench's 8,000 scenarios into interactive agent environments, enabling systematic evaluation of multi-turn conversations, tool usage efficiency, error recovery, and architectural consistency across extended development sessions. We also introduce an evaluation methodology with 9 metrics across comprehension and efficiency dimensions. Our framework provides agents with 8 specialized tools (file operations, search, code analysis) and evaluates them across context lengths ranging from 10K to 1M tokens, enabling precise assessment of long-context performance. Through systematic evaluation of state-of-the-art models, we reveal several key findings: (1) agents exhibit remarkable long-context robustness; (2) comprehension-efficiency trade-off exists with negative correlation, where thorough exploration increases comprehension but reduces efficiency; and (3) conversation efficiency varies dramatically across models, with strategic tool usage patterns differentiating high-performing agents. As the first long-context LLM agent benchmark for software engineering, LoCoBench-Agent establishes a rigorous foundation for measuring agent capabilities, identifying performance gaps, and advancing autonomous software development at scale.

**arXiv ID:** 2511.13998
</details>

<details>
<summary><strong>ReflexGrad: Three-Way Synergistic Architecture for Zero-Shot Generalization in LLM Agents</strong> - Ankush Kadu, Ashwanth Krishnan - [[pdf]](https://arxiv.org/pdf/2511.14584)</summary>

**Abstract:** Enabling agents to learn from experience and generalize across diverse tasks without task-specific training remains a fundamental challenge in reinforcement learning and decision-making. While recent approaches have explored episodic memory (Reflexion), gradient-based prompt optimization (TextGrad),and hierarchical task decomposition independently, their potential for synergistic integration remains unexplored. We introduce ReflexGrad, a novel architecture that tightly couples three complementary mechanisms: (1) LLM-based hierarchical TODO decomposition for strategic planning, (2) history-aware causal reflection that analyzes recent action patterns to identify failure root causes and enable within-trial learning, and (3) gradient-based optimization for systematic improvement. Unlike prior work relying on few-shot demonstrations, our system achieves true zero-shot generalization through pure LLM semantic reasoning,requiring no task-specific examples, fine-tuning, or hardcoded similarity metrics. Evaluated on ALFWorld benchmark tasks, ReflexGrad demonstrates 67% zero-shot success rate on Trial 0 without any prior task experience or demonstrations, establishing effective performance on first exposure. Through empirical analysis, we identify the architectural mechanisms underlying stable convergence (zero action loops) and effective cross-task transfer (67% to 78% improvement).Our work demonstrates that synergistic integration of complementary learning mechanisms enables robust zero-shot generalization that approaches few-shot baselines from prior work.

**arXiv ID:** 2511.14584
</details>

<details>
<summary><strong>Can Machines Think Like Humans? A Behavioral Evaluation of LLM Agents in Dictator Games</strong> - Ji Ma - [[pdf]](https://arxiv.org/pdf/2410.21359)</summary>

**Abstract:** As Large Language Model (LLM)-based agents increasingly engage with human society, how well do we understand their prosocial behaviors? We (1) investigate how LLM agents' prosocial behaviors can be induced by different personas and benchmarked against human behaviors; and (2) introduce a social science approach to evaluate LLM agents' decision-making. We explored how different personas and experimental framings affect these AI agents' altruistic behavior in dictator games and compared their behaviors within the same LLM family, across various families, and with human behaviors. The findings reveal that merely assigning a human-like identity to LLMs does not produce human-like behaviors. These findings suggest that LLM agents' reasoning does not consistently exhibit textual markers of human decision-making in dictator games and that their alignment with human behavior varies substantially across model architectures and prompt formulations; even worse, such dependence does not follow a clear pattern. As society increasingly integrates machine intelligence, "Prosocial AI" emerges as a promising and urgent research direction in philanthropic studies.

**arXiv ID:** 2410.21359
</details>

<details>
<summary><strong>Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning</strong> - Mingyue Cheng, Jie Ouyang, Shuo Yu, Ruiran Yan, Yucong Luo, Zirui Liu, Daoyu Wang, Qi Liu, Enhong Chen - [[pdf]](https://arxiv.org/pdf/2511.14460)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly being explored for building Agents capable of active environmental interaction (e.g., via tool use) to solve complex problems. Reinforcement Learning (RL) is considered a key technology with significant potential for training such Agents; however, the effective application of RL to LLM Agents is still in its nascent stages and faces considerable challenges. Currently, this emerging field lacks in-depth exploration into RL approaches specifically tailored for the LLM Agent context, alongside a scarcity of flexible and easily extensible training frameworks designed for this purpose. To help advance this area, this paper first revisits and clarifies Reinforcement Learning methodologies for LLM Agents by systematically extending the Markov Decision Process (MDP) framework to comprehensively define the key components of an LLM Agent. Secondly, we introduce Agent-R1, a modular, flexible, and user-friendly training framework for RL-based LLM Agents, designed for straightforward adaptation across diverse task scenarios and interactive environments. We conducted experiments on Multihop QA benchmark tasks, providing initial validation for the effectiveness of our proposed methods and framework.

**arXiv ID:** 2511.14460
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (19 papers)</h2></summary>

<details>
<summary><strong>Artificial Intelligence Agents in Music Analysis: An Integrative Perspective Based on Two Use Cases</strong> - Antonio Manuel Martínez-Heredia, Dolores Godrid Rodríguez, Andrés Ortiz García - [[pdf]](https://arxiv.org/pdf/2511.13987)</summary>

**Abstract:** This paper presents an integrative review and experimental validation of artificial intelligence (AI) agents applied to music analysis and education. We synthesize the historical evolution from rule-based models to contemporary approaches involving deep learning, multi-agent architectures, and retrieval-augmented generation (RAG) frameworks. The pedagogical implications are evaluated through a dual-case methodology: (1) the use of generative AI platforms in secondary education to foster analytical and creative skills; (2) the design of a multiagent system for symbolic music analysis, enabling modular, scalable, and explainable workflows.
Experimental results demonstrate that AI agents effectively enhance musical pattern recognition, compositional parameterization, and educational feedback, outperforming traditional automated methods in terms of interpretability and adaptability. The findings highlight key challenges concerning transparency, cultural bias, and the definition of hybrid evaluation metrics, emphasizing the need for responsible deployment of AI in educational environments.
This research contributes to a unified framework that bridges technical, pedagogical, and ethical considerations, offering evidence-based guidance for the design and application of intelligent agents in computational musicology and music education.

**arXiv ID:** 2511.13987
</details>

<details>
<summary><strong>AISAC: An Integrated multi-agent System for Transparent, Retrieval-Grounded Scientific Assistance</strong> - Chandrachur Bhattacharya, Sibendu Som - [[pdf]](https://arxiv.org/pdf/2511.14043)</summary>

**Abstract:** AI Scientific Assistant Core (AISAC) is an integrated multi-agent system developed at Argonne National Laboratory for scientific and engineering workflows. AISAC builds on established technologies - LangGraph for orchestration, FAISS for vector search, and SQLite for persistence - and integrates them into a unified system prototype focused on transparency, provenance tracking, and scientific adaptability.
The system implements a Router-Planner-Coordinator workflow and an optional Evaluator role, using prompt-engineered agents coordinated via LangGraph's StateGraph and supported by helper agents such as a Researcher. Each role is defined through custom system prompts that enforce structured JSON outputs. A hybrid memory approach (FAISS + SQLite) enables both semantic retrieval and structured conversation history. An incremental indexing strategy based on file hashing minimizes redundant re-embedding when scientific corpora evolve. A configuration-driven project bootstrap layer allows research teams to customize tools, prompts, and data sources without modifying core code.
All agent decisions, tool invocations, and retrievals are logged and visualized through a custom Gradio interface, providing step-by-step transparency for each reasoning episode. The authors have applied AISAC to multiple research areas at Argonne, including specialized deployments for waste-to-products research and energy process safety, as well as general-purpose scientific assistance, demonstrating its cross-domain applicability.

**arXiv ID:** 2511.14043
</details>

<details>
<summary><strong>APD-Agents: A Large Language Model-Driven Multi-Agents Collaborative Framework for Automated Page Design</strong> - Xinpeng Chen, Xiaofeng Han, Kaihao Zhang, Guochao Ren, Yujie Wang, Wenhao Cao, Yang Zhou, Jianfeng Lu, Zhenbo Song - [[pdf]](https://arxiv.org/pdf/2511.14101)</summary>

**Abstract:** Layout design is a crucial step in developing mobile app pages. However, crafting satisfactory designs is time-intensive for designers: they need to consider which controls and content to present on the page, and then repeatedly adjust their size, position, and style for better aesthetics and structure. Although many design software can now help to perform these repetitive tasks, extensive training is needed to use them effectively. Moreover, collaborative design across app pages demands extra time to align standards and ensure consistent styling. In this work, we propose APD-agents, a large language model (LLM) driven multi-agent framework for automated page design in mobile applications. Our framework contains OrchestratorAgent, SemanticParserAgent, PrimaryLayoutAgent, TemplateRetrievalAgent, and RecursiveComponentAgent. Upon receiving the user's description of the page, the OrchestratorAgent can dynamically can direct other agents to accomplish users' design task. To be specific, the SemanticParserAgent is responsible for converting users' descriptions of page content into structured data. The PrimaryLayoutAgent can generate an initial coarse-grained layout of this page. The TemplateRetrievalAgent can fetch semantically relevant few-shot examples and enhance the quality of layout generation. Besides, a RecursiveComponentAgent can be used to decide how to recursively generate all the fine-grained sub-elements it contains for each element in the layout. Our work fully leverages the automatic collaboration capabilities of large-model-driven multi-agent systems. Experimental results on the RICO dataset show that our APD-agents achieve state-of-the-art performance.

**arXiv ID:** 2511.14101
</details>

<details>
<summary><strong>DataSage: Multi-agent Collaboration for Insight Discovery with External Knowledge Retrieval, Multi-role Debating, and Multi-path Reasoning</strong> - Xiaochuan Liu, Yuanfeng Song, Xiaoming Yin, Xing Chen - [[pdf]](https://arxiv.org/pdf/2511.14299)</summary>

**Abstract:** In today's data-driven era, fully automated end-to-end data analytics, particularly insight discovery, is critical for discovering actionable insights that assist organizations in making effective decisions. With the rapid advancement of large language models (LLMs), LLM-driven agents have emerged as a promising paradigm for automating data analysis and insight discovery. However, existing data insight agents remain limited in several key aspects, often failing to deliver satisfactory results due to: (1) insufficient utilization of domain knowledge, (2) shallow analytical depth, and (3) error-prone code generation during insight generation. To address these issues, we propose DataSage, a novel multi-agent framework that incorporates three innovative features including external knowledge retrieval to enrich the analytical context, a multi-role debating mechanism to simulate diverse analytical perspectives and deepen analytical depth, and multi-path reasoning to improve the accuracy of the generated code and insights. Extensive experiments on InsightBench demonstrate that DataSage consistently outperforms existing data insight agents across all difficulty levels, offering an effective solution for automated data insight discovery.

**arXiv ID:** 2511.14299
</details>

<details>
<summary><strong>Heterogeneous Multi-Agent Proximal Policy Optimization for Power Distribution System Restoration</strong> - Parya Dolatyabi, Mahdi Khodayar - [[pdf]](https://arxiv.org/pdf/2511.14730)</summary>

**Abstract:** Restoring power distribution systems (PDS) after large-scale outages requires sequential switching operations that reconfigure feeder topology and coordinate distributed energy resources (DERs) under nonlinear constraints such as power balance, voltage limits, and thermal ratings. These challenges make conventional optimization and value-based RL approaches computationally inefficient and difficult to scale. This paper applies a Heterogeneous-Agent Reinforcement Learning (HARL) framework, instantiated through Heterogeneous-Agent Proximal Policy Optimization (HAPPO), to enable coordinated restoration across interconnected microgrids. Each agent controls a distinct microgrid with different loads, DER capacities, and switch counts, introducing practical structural heterogeneity. Decentralized actor policies are trained with a centralized critic to compute advantage values for stable on-policy updates. A physics-informed OpenDSS environment provides full power flow feedback and enforces operational limits via differentiable penalty signals rather than invalid action masking. The total DER generation is capped at 2400 kW, and each microgrid must satisfy local supply-demand feasibility. Experiments on the IEEE 123-bus and IEEE 8500-node systems show that HAPPO achieves faster convergence, higher restored power, and smoother multi-seed training than DQN, PPO, MAES, MAGDPG, MADQN, Mean-Field RL, and QMIX. Results demonstrate that incorporating microgrid-level heterogeneity within the HARL framework yields a scalable, stable, and constraint-aware solution for complex PDS restoration.

**arXiv ID:** 2511.14730
</details>

<details>
<summary><strong>Multi-Agent VLMs Guided Self-Training with PNU Loss for Low-Resource Offensive Content Detection</strong> - Han Wang, Deyi Ji, Junyu Lu, Lanyun Zhu, Hailong Zhang, Haiyang Wu, Liqun Liu, Peng Shu, Roy Ka-Wei Lee - [[pdf]](https://arxiv.org/pdf/2511.13759)</summary>

**Abstract:** Accurate detection of offensive content on social media demands high-quality labeled data; however, such data is often scarce due to the low prevalence of offensive instances and the high cost of manual annotation. To address this low-resource challenge, we propose a self-training framework that leverages abundant unlabeled data through collaborative pseudo-labeling. Starting with a lightweight classifier trained on limited labeled data, our method iteratively assigns pseudo-labels to unlabeled instances with the support of Multi-Agent Vision-Language Models (MA-VLMs). Un-labeled data on which the classifier and MA-VLMs agree are designated as the Agreed-Unknown set, while conflicting samples form the Disagreed-Unknown set. To enhance label reliability, MA-VLMs simulate dual perspectives, moderator and user, capturing both regulatory and subjective viewpoints. The classifier is optimized using a novel Positive-Negative-Unlabeled (PNU) loss, which jointly exploits labeled, Agreed-Unknown, and Disagreed-Unknown data while mitigating pseudo-label noise. Experiments on benchmark datasets demonstrate that our framework substantially outperforms baselines under limited supervision and approaches the performance of large-scale models

**arXiv ID:** 2511.13759
</details>

<details>
<summary><strong>Fair-GNE : Generalized Nash Equilibrium-Seeking Fairness in Multiagent Healthcare Automation</strong> - Promise Ekpo, Saesha Agarwal, Felix Grimm, Lekan Molu, Angelique Taylor - [[pdf]](https://arxiv.org/pdf/2511.14135)</summary>

**Abstract:** Enforcing a fair workload allocation among multiple agents tasked to achieve an objective in learning enabled demand side healthcare worker settings is crucial for consistent and reliable performance at runtime. Existing multi-agent reinforcement learning (MARL) approaches steer fairness by shaping reward through post hoc orchestrations, leaving no certifiable self-enforceable fairness that is immutable by individual agents at runtime. Contextualized within a setting where each agent shares resources with others, we address this shortcoming with a learning enabled optimization scheme among self-interested decision makers whose individual actions affect those of other agents. This extends the problem to a generalized Nash equilibrium (GNE) game-theoretic framework where we steer group policy to a safe and locally efficient equilibrium, so that no agent can improve its utility function by unilaterally changing its decisions. Fair-GNE models MARL as a constrained generalized Nash equilibrium-seeking (GNE) game, prescribing an ideal equitable collective equilibrium within the problem's natural fabric. Our hypothesis is rigorously evaluated in our custom-designed high-fidelity resuscitation simulator. Across all our numerical experiments, Fair-GNE achieves significant improvement in workload balance over fixed-penalty baselines (0.89 vs.\ 0.33 JFI, $p < 0.01$) while maintaining 86\% task success, demonstrating statistically significant fairness gains through adaptive constraint enforcement. Our results communicate our formulations, evaluation metrics, and equilibrium-seeking innovations in large multi-agent learning-based healthcare systems with clarity and principled fairness enforcement.

**arXiv ID:** 2511.14135
</details>

<details>
<summary><strong>Enhancing Agentic Autonomous Scientific Discovery with Vision-Language Model Capabilities</strong> - Kahaan Gandhi, Boris Bolliet, Inigo Zubeldia - [[pdf]](https://arxiv.org/pdf/2511.14631)</summary>

**Abstract:** We show that multi-agent systems guided by vision-language models (VLMs) improve end-to-end autonomous scientific discovery. By treating plots as verifiable checkpoints, a VLM-as-a-judge evaluates figures against dynamically generated domain-specific rubrics, enabling agents to correct their own errors and steer exploratory data analysis in real-time. Case studies in cosmology and astrochemistry demonstrate recovery from faulty reasoning paths and adaptation to new datasets without human intervention. On a 10-task benchmark for data-driven discovery, VLM-augmented systems achieve pass at 1 scores of 0.7-0.8, compared to 0.2-0.3 for code-only and 0.4-0.5 for code-and-text baselines, while also providing auditable reasoning traces that improve interpretability. Code available here: this https URL

**arXiv ID:** 2511.14631
</details>

<details>
<summary><strong>Harnessing Diverse Perspectives: A Multi-Agent Framework for Enhanced Error Detection in Knowledge Graphs</strong> - Yu Li, Yi Huang, Guilin Qi, Junlan Feng, Nan Hu, Songlin Zhai, Haohan Xue, Yongrui Chen, Ruoyan Shen, Tongtong Wu - [[pdf]](https://arxiv.org/pdf/2501.15791)</summary>

**Abstract:** Knowledge graphs are widely used in industrial applications, making error detection crucial for ensuring the reliability of downstream applications. Existing error detection methods often fail to effectively utilize fine-grained subgraph information and rely solely on fixed graph structures, while also lacking transparency in their decision-making processes, which results in suboptimal detection performance. In this paper, we propose a novel Multi-Agent framework for Knowledge Graph Error Detection (MAKGED) that utilizes multiple large language models (LLMs) in a collaborative setting. By concatenating fine-grained, bidirectional subgraph embeddings with LLM-based query embeddings during training, our framework integrates these representations to produce four specialized agents. These agents utilize subgraph information from different dimensions to engage in multi-round discussions, thereby improving error detection accuracy and ensuring a transparent decision-making process. Extensive experiments on FB15K and WN18RR demonstrate that MAKGED outperforms state-of-the-art methods, enhancing the accuracy and robustness of KG evaluation. For specific industrial scenarios, our framework can facilitate the training of specialized agents using domain-specific knowledge graphs for error detection, which highlights the potential industrial application value of our framework. Our code and datasets are available at this https URL.

**arXiv ID:** 2501.15791
</details>

<details>
<summary><strong>Towards Automatic Evaluation and Selection of PHI De-identification Models via Multi-Agent Collaboration</strong> - Guanchen Wu, Zuhui Chen, Yuzhang Xie, Carl Yang - [[pdf]](https://arxiv.org/pdf/2510.16194)</summary>

**Abstract:** Protected health information (PHI) de-identification is critical for enabling the safe reuse of clinical notes, yet evaluating and comparing PHI de-identification models typically depends on costly, small-scale expert annotations. We present TEAM-PHI, a multi-agent evaluation and selection framework that uses large language models (LLMs) to automatically measure de-identification quality and select the best-performing model without heavy reliance on gold labels. TEAM-PHI deploys multiple Evaluation Agents, each independently judging the correctness of PHI extractions and outputting structured metrics. Their results are then consolidated through an LLM-based majority voting mechanism that integrates diverse evaluator perspectives into a single, stable, and reproducible ranking. Experiments on a real-world clinical note corpus demonstrate that TEAM-PHI produces consistent and accurate rankings: despite variation across individual evaluators, LLM-based voting reliably converges on the same top-performing systems. Further comparison with ground-truth annotations and human evaluation confirms that the framework's automated rankings closely match supervised evaluation. By combining independent evaluation agents with LLM majority voting, TEAM-PHI offers a practical, secure, and cost-effective solution for automatic evaluation and best-model selection in PHI de-identification, even when ground-truth labels are limited.

**arXiv ID:** 2510.16194
</details>

<details>
<summary><strong>Multi-Agent Deep Research: Training Multi-Agent Systems with M-GRPO</strong> - Haoyang Hong, Jiajun Yin, Yuan Wang, Jingnan Liu, Zhe Chen, Ailing Yu, Ji Li, Zhiling Ye, Hansong Xiao, Yefei Chen, Hualei Zhou, Yun Yue, Minghui Yang, Chunxiao Guo, Junwei Liu, Peng Wei, Jinjie Gu - [[pdf]](https://arxiv.org/pdf/2511.13288)</summary>

**Abstract:** Multi-agent systems perform well on general reasoning tasks. However, the lack of training in specialized areas hinders their accuracy. Current training methods train a unified large language model (LLM) for all agents in the system. This may limit the performances due to different distributions underlying for different agents. Therefore, training multi-agent systems with distinct LLMs should be the next step to solve. However, this approach introduces optimization challenges. For example, agents operate at different frequencies, rollouts involve varying sub-agent invocations, and agents are often deployed across separate servers, disrupting end-to-end gradient flow. To address these issues, we propose M-GRPO, a hierarchical extension of Group Relative Policy Optimization designed for vertical Multi-agent systems with a main agent (planner) and multiple sub-agents (multi-turn tool executors). M-GRPO computes group-relative advantages for both main and sub-agents, maintaining hierarchical credit assignment. It also introduces a trajectory-alignment scheme that generates fixed-size batches despite variable sub-agent invocations. We deploy a decoupled training pipeline in which agents run on separate servers and exchange minimal statistics via a shared store. This enables scalable training without cross-server backpropagation. In experiments on real-world benchmarks (e.g., GAIA, XBench-DeepSearch, and WebWalkerQA), M-GRPO consistently outperforms both single-agent GRPO and multi-agent GRPO with frozen sub-agents, demonstrating improved stability and sample efficiency. These results show that aligning heterogeneous trajectories and decoupling optimization across specialized agents enhances tool-augmented reasoning tasks.

**arXiv ID:** 2511.13288
</details>

<details>
<summary><strong>GMAT: Grounded Multi-Agent Clinical Description Generation for Text Encoder in Vision-Language MIL for Whole Slide Image Classification</strong> - Ngoc Bui Lam Quang, Nam Le Nguyen Binh, Thanh-Huy Nguyen, Le Thien Phuc Nguyen, Quan Nguyen, Ulas Bagci - [[pdf]](https://arxiv.org/pdf/2508.01293)</summary>

**Abstract:** Multiple Instance Learning (MIL) is the leading approach for whole slide image (WSI) classification, enabling efficient analysis of gigapixel pathology slides. Recent work has introduced vision-language models (VLMs) into MIL pipelines to incorporate medical knowledge through text-based class descriptions rather than simple class names. However, when these methods rely on large language models (LLMs) to generate clinical descriptions or use fixed-length prompts to represent complex pathology concepts, the limited token capacity of VLMs often constrains the expressiveness and richness of the encoded class information. Additionally, descriptions generated solely by LLMs may lack domain grounding and fine-grained medical specificity, leading to suboptimal alignment with visual features. To address these challenges, we propose a vision-language MIL framework with two key contributions: (1) A grounded multi-agent description generation system that leverages curated pathology textbooks and agent specialization (e.g., morphology, spatial context) to produce accurate and diverse clinical descriptions; (2) A text encoding strategy using a list of descriptions rather than a single prompt, capturing fine-grained and complementary clinical signals for better alignment with visual features. Integrated into a VLM-MIL pipeline, our approach shows improved performance over single-prompt class baselines and achieves results comparable to state-of-the-art models, as demonstrated on renal and lung cancer datasets.

**arXiv ID:** 2508.01293
</details>

<details>
<summary><strong>Skill-Aligned Fairness in Multi-Agent Learning for Collaboration in Healthcare</strong> - Promise Osaine Ekpo, Brian La, Thomas Wiener, Saesha Agarwal, Arshia Agrawal, Gonzalo Gonzalez-Pumariega, Lekan P. Molu, Angelique Taylor - [[pdf]](https://arxiv.org/pdf/2508.18708)</summary>

**Abstract:** Fairness in multi-agent reinforcement learning (MARL) is often framed as a workload balance problem, overlooking agent expertise and the structured coordination required in real-world domains. In healthcare, equitable task allocation requires workload balance or expertise alignment to prevent burnout and overuse of highly skilled agents. Workload balance refers to distributing an approximately equal number of subtasks or equalised effort across healthcare workers, regardless of their expertise. We make two contributions to address this problem. First, we propose FairSkillMARL, a framework that defines fairness as the dual objective of workload balance and skill-task alignment. Second, we introduce MARLHospital, a customizable healthcare-inspired environment for modeling team compositions and energy-constrained scheduling impacts on fairness, as no existing simulators are well-suited for this problem. We conducted experiments to compare FairSkillMARL in conjunction with four standard MARL methods, and against two state-of-the-art fairness metrics. Our results suggest that fairness based solely on equal workload might lead to task-skill mismatches and highlight the need for more robust metrics that capture skill-task misalignment. Our work provides tools and a foundation for studying fairness in heterogeneous multi-agent systems where aligning effort with expertise is critical.

**arXiv ID:** 2508.18708
</details>

<details>
<summary><strong>Who Gets the Reward, Who Gets the Blame? Evaluation-Aligned Training Signals for Multi-LLM Agents</strong> - Chih-Hsuan Yang, Tanwi Mallick, Le Chen, Krishnan Raghavan, Azton Wells, Amal Gueroudji, Ian T. Foster, Rajeev Thakur - [[pdf]](https://arxiv.org/pdf/2511.10687)</summary>

**Abstract:** Large Language Models (LLMs) in multi-agent systems (MAS) have shown promise for complex tasks, yet current training methods lack principled ways to connect system-level evaluation with agent-level and message-level learning. We propose a theoretical framework that unifies cooperative game-theoretic attribution with process reward modeling to transform system evaluation into agent credit and then into response-level signals. Unlike prior approaches that rely only on attribution (e.g., Shapley) or step-level labels (e.g., PRM), our method produces local, signed, and credit-conserving signals. In success cases, Shapley-based credit assignment fairly allocates outcomes across agents and is refined into per-message rewards that promote cooperation while discouraging redundancy or sabotage. In failure cases, first-error localization yields repair-aware preferences that penalize harmful steps while rewarding corrective attempts. The resulting signals are bounded, cooperative, and directly compatible with reinforcement-based or preference-based post-training, providing a unified and auditable pathway from global evaluation to local supervision in LLM multi-agent training. Our contribution is conceptual: we present a theoretical foundation and training signals, leaving empirical validation for future work.

**arXiv ID:** 2511.10687
</details>

<details>
<summary><strong>FinVet: A Collaborative Framework of RAG and External Fact-Checking Agents for Financial Misinformation Detection</strong> - Daniel Berhane Araya, Duoduo Liao - [[pdf]](https://arxiv.org/pdf/2510.11654)</summary>

**Abstract:** Financial markets face growing threats from misinformation that can trigger billions in losses in minutes. Most existing approaches lack transparency in their decision-making and provide limited attribution to credible sources. We introduce FinVet, a novel multi-agent framework that integrates two Retrieval-Augmented Generation (RAG) pipelines with external fact-checking through a confidence-weighted voting mechanism. FinVet employs adaptive three-tier processing that dynamically adjusts verification strategies based on retrieval confidence, from direct metadata extraction to hybrid reasoning to full model-based analysis. Unlike existing methods, FinVet provides evidence-backed verdicts, source attribution, confidence scores, and explicit uncertainty flags when evidence is insufficient. Experimental evaluation on the FinFact dataset shows that FinVet achieves an F1 score of 0.85, which is a 10.4% improvement over the best individual pipeline (fact-check pipeline) and 37% improvement over standalone RAG approaches.

**arXiv ID:** 2510.11654
</details>

<details>
<summary><strong>Efficient Reinforcement Learning for Zero-Shot Coordination in Evolving Games</strong> - Bingyu Hui, Lebin Yu, Quanming Yao, Yunpeng Qu, Xudong Zhang, Jian Wang - [[pdf]](https://arxiv.org/pdf/2511.11083)</summary>

**Abstract:** Zero-shot coordination(ZSC), a key challenge in multi-agent game theory, has become a hot topic in reinforcement learning (RL) research recently, especially in complex evolving games. It focuses on the generalization ability of agents, requiring them to coordinate well with collaborators from a diverse, potentially evolving, pool of partners that are not seen before without any fine-tuning. Population-based training, which approximates such an evolving partner pool, has been proven to provide good zero-shot coordination performance; nevertheless, existing methods are limited by computational resources, mainly focusing on optimizing diversity in small populations while neglecting the potential performance gains from scaling population size. To address this issue, this paper proposes the Scalable Population Training (ScaPT), an efficient RL training framework comprising two key components: a meta-agent that efficiently realizes a population by selectively sharing parameters across agents, and a mutual information regularizer that guarantees population diversity. To empirically validate the effectiveness of ScaPT, this paper evaluates it along with representational frameworks in Hanabi cooperative game and confirms its superiority.

**arXiv ID:** 2511.11083
</details>

<details>
<summary><strong>FICO: Finite-Horizon Closed-Loop Factorization for Unified Multi-Agent Path Finding</strong> - Jiarui Li, Alessandro Zanardi, Runyu Zhang, Gioele Zardini - [[pdf]](https://arxiv.org/pdf/2511.13961)</summary>

**Abstract:** Multi-Agent Path Finding is a fundamental problem in robotics and AI, yet most existing formulations treat planning and execution separately and address variants of the problem in an ad hoc manner. This paper presents a system-level framework for MAPF that integrates planning and execution, generalizes across variants, and explicitly models uncertainties. At its core is the MAPF system, a formal model that casts MAPF as a control design problem encompassing classical and uncertainty-aware formulations. To solve it, we introduce Finite-Horizon Closed-Loop Factorization (FICO), a factorization-based algorithm inspired by receding-horizon control that exploits compositional structure for efficient closed-loop operation. FICO enables real-time responses -- commencing execution within milliseconds -- while scaling to thousands of agents and adapting seamlessly to execution-time uncertainties. Extensive case studies demonstrate that it reduces computation time by up to two orders of magnitude compared with open-loop baselines, while delivering significantly higher throughput under stochastic delays and agent arrivals. These results establish a principled foundation for analyzing and advancing MAPF through system-level modeling, factorization, and closed-loop design.

**arXiv ID:** 2511.13961
</details>

<details>
<summary><strong>Who Moved My Distribution? Conformal Prediction for Interactive Multi-Agent Systems</strong> - Allen Emmanuel Binny, Anushri Dixit - [[pdf]](https://arxiv.org/pdf/2511.11567)</summary>

**Abstract:** Uncertainty-aware prediction is essential for safe motion planning, especially when using learned models to forecast the behavior of surrounding agents. Conformal prediction is a statistical tool often used to produce uncertainty-aware prediction regions for machine learning models. Most existing frameworks utilizing conformal prediction-based uncertainty predictions assume that the surrounding agents are non-interactive. This is because in closed-loop, as uncertainty-aware agents change their behavior to account for prediction uncertainty, the surrounding agents respond to this change, leading to a distribution shift which we call endogenous distribution shift. To address this challenge, we introduce an iterative conformal prediction framework that systematically adapts the uncertainty-aware ego-agent controller to the endogenous distribution shift. The proposed method provides probabilistic safety guarantees while adapting to the evolving behavior of reactive, non-ego agents. We establish a model for the endogenous distribution shift and provide the conditions for the iterative conformal prediction pipeline to converge under such a distribution shift. We validate our framework in simulation for 2- and 3- agent interaction scenarios, demonstrating collision avoidance without resulting in overly conservative behavior and an overall improvement in success rates of up to 9.6% compared to other conformal prediction-based baselines.

**arXiv ID:** 2511.11567
</details>

<details>
<summary><strong>BeautyGuard: Designing a Multi-Agent Roundtable System for Proactive Beauty Tech Compliance through Stakeholder Collaboration</strong> - Junwei Li, Wenqing Wang, Huiliu Mao, Jiazhe Ni, Zeyu Xiong - [[pdf]](https://arxiv.org/pdf/2511.12645)</summary>

**Abstract:** As generative AI enters enterprise workflows, ensuring compliance with legal, ethical, and reputational standards becomes a pressing challenge. In beauty tech, where biometric and personal data are central, traditional reviews are often manual, fragmented, and reactive. To examine these challenges, we conducted a formative study with six experts (four IT managers, two legal managers) at a multinational beauty company. The study revealed pain points in rule checking, precedent use, and the lack of proactive guidance.
Motivated by these findings, we designed a multi-agent "roundtable" system powered by a large language model. The system assigns role-specialized agents for legal interpretation, checklist review, precedent search, and risk mitigation, synthesizing their perspectives into structured compliance advice.
We evaluated the prototype with the same experts using System Usability Scale(SUS), The Official NASA Task Load Index(NASA-TLX), and interviews. Results show exceptional usability (SUS: 77.5/100) and minimal cognitive workload, with three key findings: (1) multi-agent systems can preserve tacit knowledge into standardized workflows, (2) information augmentation achieves higher acceptance than decision automation, and (3) successful enterprise AI should mirror organizational structures. This work contributes design principles for human-AI collaboration in compliance review, with broader implications for regulated industries beyond beauty tech.

**arXiv ID:** 2511.12645
</details>

</details>

<details open>
<summary><h2>Other Agent Research (11 papers)</h2></summary>

<details>
<summary><strong>Randomized Controlled Trials for Conditional Access Optimization Agent</strong> - James Bono, Beibei Cheng, Joaquin Lozano - [[pdf]](https://arxiv.org/pdf/2511.13865)</summary>

**Abstract:** AI agents are increasingly deployed to automate complex enterprise workflows, yet evidence of their effectiveness in identity governance is limited. We report results from the first randomized controlled trial (RCT) evaluating an AI agent for Conditional Access (CA) policy management in Microsoft Entra. The agent assists with four high-value tasks: policy merging, Zero-Trust baseline gap detection, phased rollout planning, and user-policy alignment. In a production-grade environment, 162 identity administrators were randomly assigned to a control group (no agent) or treatment group (agent-assisted) and asked to perform these tasks. Agent access produced substantial gains: accuracy improved by 48% and task completion time decreased by 43% while holding accuracy constant. The largest benefits emerged on cognitively demanding tasks such as baseline gap detection. These findings demonstrate that purpose-built AI agents can significantly enhance both speed and accuracy in identity administration.

**arXiv ID:** 2511.13865
</details>

<details>
<summary><strong>Randomized Controlled Trials for Phishing Triage Agent</strong> - James Bono - [[pdf]](https://arxiv.org/pdf/2511.13860)</summary>

**Abstract:** Security operations centers (SOCs) face a persistent challenge: efficiently triaging a high volume of user-reported phishing emails while maintaining robust protection against threats. This paper presents the first randomized controlled trial (RCT) evaluating the impact of a domain-specific AI agent - the Microsoft Security Copilot Phishing Triage Agent - on analyst productivity and accuracy. Our results demonstrate that agent-augmented analysts achieved up to 6.5 times as many true positives per analyst minute and a 77% improvement in verdict accuracy compared to a control group. The agent's queue prioritization and verdict explanations were both significant drivers of efficiency. Behavioral analysis revealed that agent-augmented analysts reallocated their attention, spending 53% more time on malicious emails, and were not prone to rubber-stamping the agent's malicious verdicts. These findings offer actionable insights for SOC leaders considering AI adoption, including the potential for agents to fundamentally change the optimal allocation of SOC resources.

**arXiv ID:** 2511.13860
</details>

<details>
<summary><strong>Knowledge-Grounded Agentic Large Language Models for Multi-Hazard Understanding from Reconnaissance Reports</strong> - Chenchen Kuai, Zihao Li, Braden Rosen, Stephanie Paan, Navid Jafari, Jean-Louis Briaud, Yunlong Zhang, Youssef M. A. Hashash, Yang Zhou - [[pdf]](https://arxiv.org/pdf/2511.14010)</summary>

**Abstract:** Post-disaster reconnaissance reports contain critical evidence for understanding multi-hazard interactions, yet their unstructured narratives make systematic knowledge transfer difficult. Large language models (LLMs) offer new potential for analyzing these reports, but often generate unreliable or hallucinated outputs when domain grounding is absent. This study introduces the Mixture-of-Retrieval Agentic RAG (MoRA-RAG), a knowledge-grounded LLM framework that transforms reconnaissance reports into a structured foundation for multi-hazard reasoning. The framework integrates a Mixture-of-Retrieval mechanism that dynamically routes queries across hazard-specific databases while using agentic chunking to preserve contextual coherence during retrieval. It also includes a verification loop that assesses evidence sufficiency, refines queries, and initiates targeted searches when information remains incomplete. We construct HazardRecQA by deriving question-answer pairs from GEER reconnaissance reports, which document 90 global events across seven major hazard types. MoRA-RAG achieves up to 94.5 percent accuracy, outperforming zero-shot LLMs by 30 percent and state-of-the-art RAG systems by 10 percent, while reducing hallucinations across diverse LLM architectures. MoRA-RAG also enables open-weight LLMs to achieve performance comparable to proprietary models. It establishes a new paradigm for transforming post-disaster documentation into actionable, trustworthy intelligence for hazard resilience.

**arXiv ID:** 2511.14010
</details>

<details>
<summary><strong>Towards Deploying VLA without Fine-Tuning: Plug-and-Play Inference-Time VLA Policy Steering via Embodied Evolutionary Diffusion</strong> - Zhuo Li, Junjia Liu, Zhipeng Dong, Tao Teng, Quentin Rouxel, Darwin Caldwell, Fei Chen - [[pdf]](https://arxiv.org/pdf/2511.14178)</summary>

**Abstract:** Vision-Language-Action (VLA) models have demonstrated significant potential in real-world robotic manipulation. However, pre-trained VLA policies still suffer from substantial performance degradation during downstream deployment. Although fine-tuning can mitigate this issue, its reliance on costly demonstration collection and intensive computation makes it impractical in real-world settings. In this work, we introduce VLA-Pilot, a plug-and-play inference-time policy steering method for zero-shot deployment of pre-trained VLA without any additional fine-tuning or data collection. We evaluate VLA-Pilot on six real-world downstream manipulation tasks across two distinct robotic embodiments, encompassing both in-distribution and out-of-distribution scenarios. Experimental results demonstrate that VLA-Pilot substantially boosts the success rates of off-the-shelf pre-trained VLA policies, enabling robust zero-shot generalization to diverse tasks and embodiments. Experimental videos and code are available at: this https URL.

**arXiv ID:** 2511.14178
</details>

<details>
<summary><strong>MI9: An Integrated Runtime Governance Framework for Agentic AI</strong> - Charles L. Wang, Trisha Singhal, Ameya Kelkar, Jason Tuo - [[pdf]](https://arxiv.org/pdf/2508.03858)</summary>

**Abstract:** Agentic AI systems capable of reasoning, planning, and executing actions present fundamentally distinct governance challenges compared to traditional AI models. Unlike conventional AI, these systems exhibit emergent and unexpected behaviors during runtime, introducing novel agent-related risks that cannot be fully anticipated through pre-deployment governance alone. To address this critical gap, we introduce MI9, the first fully integrated runtime governance framework designed specifically for safety and alignment of agentic AI systems. MI9 introduces real-time controls through six integrated components: agency-risk index, agent-semantic telemetry capture, continuous authorization monitoring, Finite-State-Machine (FSM)-based conformance engines, goal-conditioned drift detection, and graduated containment strategies. Operating transparently across heterogeneous agent architectures, MI9 enables the systematic, safe, and responsible deployment of agentic systems in production environments where conventional governance approaches fall short, providing the foundational infrastructure for safe agentic AI deployment at scale. Detailed analysis through a diverse set of scenarios demonstrates MI9's systematic coverage of governance challenges that existing approaches fail to address, establishing the technical foundation for comprehensive agentic AI oversight.

**arXiv ID:** 2508.03858
</details>

<details>
<summary><strong>LLM-based Agents Suffer from Hallucinations: A Survey of Taxonomy, Methods, and Directions</strong> - Xixun Lin, Yucheng Ning, Jingwen Zhang, Yan Dong, Yilong Liu, Yongxuan Wu, Xiaohua Qi, Nan Sun, Yanmin Shang, Kun Wang, Pengfei Cao, Qingyue Wang, Lixin Zou, Xu Chen, Chuan Zhou, Jia Wu, Peng Zhang, Qingsong Wen, Shirui Pan, Bin Wang, Yanan Cao, Kai Chen, Songlin Hu, Li Guo - [[pdf]](https://arxiv.org/pdf/2509.18970)</summary>

**Abstract:** Driven by the rapid advancements of Large Language Models (LLMs), LLM-based agents have emerged as powerful intelligent systems capable of human-like cognition, reasoning, and interaction. These agents are increasingly being deployed across diverse real-world applications, including student education, scientific research, and financial analysis. However, despite their remarkable potential, LLM-based agents remain vulnerable to hallucination issues, which can result in erroneous task execution and undermine the reliability of the overall system design. Addressing this critical challenge requires a deep understanding and a systematic consolidation of recent advances on LLM-based agents. To this end, we present the first comprehensive survey of hallucinations in LLM-based agents. By carefully analyzing the complete workflow of agents, we propose a new taxonomy that identifies different types of agent hallucinations occurring at different stages. Furthermore, we conduct an in-depth examination of eighteen triggering causes underlying the emergence of agent hallucinations. Through a detailed review of a large number of existing studies, we summarize approaches for hallucination mitigation and detection, and highlight promising directions for future research. We hope this survey will inspire further efforts toward addressing hallucinations in LLM-based agents, ultimately contributing to the development of more robust and reliable agent systems.

**arXiv ID:** 2509.18970
</details>

<details>
<summary><strong>Automatic Differentiation of Agent-Based Models</strong> - Arnau Quera-Bofarull, Nicholas Bishop, Joel Dyer, Daniel Jarne Ornia, Anisoara Calinescu, Doyne Farmer, Michael Wooldridge - [[pdf]](https://arxiv.org/pdf/2509.03303)</summary>

**Abstract:** Agent-based models (ABMs) simulate complex systems by capturing the bottom-up interactions of individual agents comprising the system. Many complex systems of interest, such as epidemics or financial markets, involve thousands or even millions of agents. Consequently, ABMs often become computationally demanding and rely on the calibration of numerous free parameters, which has significantly hindered their widespread adoption. In this paper, we demonstrate that automatic differentiation (AD) techniques can effectively alleviate these computational burdens. By applying AD to ABMs, the gradients of the simulator become readily available, greatly facilitating essential tasks such as calibration and sensitivity analysis. Specifically, we show how AD enables variational inference (VI) techniques for efficient parameter calibration. Our experiments demonstrate substantial performance improvements and computational savings using VI on three prominent ABMs: Axtell's model of firms; Sugarscape; and the SIR epidemiological model. Our approach thus significantly enhances the practicality and scalability of ABMs for studying complex systems.

**arXiv ID:** 2509.03303
</details>

<details>
<summary><strong>Characterizing Agent-Based Model Dynamics via $ε$-Machines and Kolmogorov-Style Complexity</strong> - Roberto Garrone - [[pdf]](https://arxiv.org/pdf/2510.12729)</summary>

**Abstract:** We propose a two-level information-theoretic framework for characterizing the informational organization of Agent-Based Model (ABM) dynamics within the broader paradigm of Complex Adaptive Systems (CAS). At the macro level, a pooled $\varepsilon$-machine is reconstructed as a reference model summarizing the system-wide informational regime. At the micro level, $\varepsilon$-machines are reconstructed for each caregiver--elder dyad and variable, complemented by algorithm-agnostic Kolmogorov-style measures, including normalized LZ78 complexity and bits per symbol from lossless compression. The resulting feature set, $\{h_{\mu}, C_{\mu}, E, \mathrm{LZ78}, \mathrm{bps}\}$, enables distributional analysis, stratified comparisons, and unsupervised clustering across agents and scenarios. Empirical results show that coupling $\varepsilon$-machines with compression diagnostics yields a coherent picture of where predictive information resides in the caregiving ABM. Global reconstructions provide a memoryless baseline ($L{=}0$ under coarse symbolizations), whereas per-dyad models reveal localized structure, particularly for walkability under ordinal encodings ($m{=}3$). Compression metrics corroborate these patterns: dictionary compressors agree on algorithmic redundancy, while normalized LZ78 captures statistical novelty. Socioeconomic variables display cross-sectional heterogeneity and near-memoryless dynamics, whereas spatial interaction induces bounded temporal memory and recurrent regimes. The framework thus distinguishes semantic organization (predictive causation and memory) from syntactic simplicity (description length) and clarifies how emergence manifests at different system layers. It is demonstrated on a caregiver--elder case study with dyad-level $\varepsilon$-machine reconstructions and compression-based diagnostics.

**arXiv ID:** 2510.12729
</details>

<details>
<summary><strong>Leveraging LLM-based agents for social science research: insights from citation network simulations</strong> - Jiarui Ji, Runlin Lei, Xuchen Pan, Zhewei Wei, Hao Sun, Yankai Lin, Xu Chen, Yongzheng Yang, Yaliang Li, Bolin Ding, Ji-Rong Wen - [[pdf]](https://arxiv.org/pdf/2511.03758)</summary>

**Abstract:** The emergence of Large Language Models (LLMs) demonstrates their potential to encapsulate the logic and patterns inherent in human behavior simulation by leveraging extensive web data pre-training. However, the boundaries of LLM capabilities in social simulation remain unclear. To further explore the social attributes of LLMs, we introduce the CiteAgent framework, designed to generate citation networks based on human-behavior simulation with LLM-based agents. CiteAgent successfully captures predominant phenomena in real-world citation networks, including power-law distribution, citational distortion, and shrinking diameter. Building on this realistic simulation, we establish two LLM-based research paradigms in social science: LLM-SE (LLM-based Survey Experiment) and LLM-LE (LLM-based Laboratory Experiment). These paradigms facilitate rigorous analyses of citation network phenomena, allowing us to validate and challenge existing theories. Additionally, we extend the research scope of traditional science of science studies through idealized social experiments, with the simulation experiment results providing valuable insights for real-world academic environments. Our work demonstrates the potential of LLMs for advancing science of science research in social science.

**arXiv ID:** 2511.03758
</details>

<details>
<summary><strong>MedBuild AI: An Agent-Based Hybrid Intelligence Framework for Reshaping Agency in Healthcare Infrastructure Planning through Generative Design for Medical Architecture</strong> - Yiming Zhang, Yuejia Xu, Ziyao Wang, Xin Yan, Xiaosai Hao - [[pdf]](https://arxiv.org/pdf/2511.11587)</summary>

**Abstract:** Globally, disparities in healthcare infrastructure remain stark, leaving countless communities without access to even basic services. Traditional infrastructure planning is often slow and inaccessible, and although many architects are actively delivering humanitarian and aid-driven hospital projects worldwide, these vital efforts still fall far short of the sheer scale and urgency of demand. This paper introduces MedBuild AI, a hybrid-intelligence framework that integrates large language models (LLMs) with deterministic expert systems to rebalance the early design and conceptual planning stages. As a web-based platform, it enables any region with satellite internet access to obtain guidance on modular, low-tech, low-cost medical building designs. The system operates through three agents: the first gathers local health intelligence via conversational interaction; the second translates this input into an architectural functional program through rule-based computation; and the third generates layouts and 3D models. By embedding computational negotiation into the design process, MedBuild AI fosters a reciprocal, inclusive, and equitable approach to healthcare planning, empowering communities and redefining agency in global healthcare architecture.

**arXiv ID:** 2511.11587
</details>

<details>
<summary><strong>Safe-ROS: An Architecture for Autonomous Robots in Safety-Critical Domains</strong> - Diana C. Benjumea, Marie Farrell, Louise A. Dennis - [[pdf]](https://arxiv.org/pdf/2511.14433)</summary>

**Abstract:** 

**arXiv ID:** 2511.14433
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (22 papers)</h2></summary>

<details>
<summary><strong>When AI Does Science: Evaluating the Autonomous AI Scientist KOSMOS in Radiation Biology</strong> - Humza Nusrat, Omar Nusrat - [[pdf]](https://arxiv.org/pdf/2511.13825)</summary>

**Abstract:** Agentic AI "scientists" now use language models to search the literature, run analyses, and generate hypotheses. We evaluate KOSMOS, an autonomous AI scientist, on three problems in radiation biology using simple random-gene null benchmarks. Hypothesis 1: baseline DNA damage response (DDR) capacity across cell lines predicts the p53 transcriptional response after irradiation (GSE30240). Hypothesis 2: baseline expression of OGT and CDO1 predicts the strength of repressed and induced radiation-response modules in breast cancer cells (GSE59732). Hypothesis 3: a 12-gene expression signature predicts biochemical recurrence-free survival after prostate radiotherapy plus androgen deprivation therapy (GSE116918). The DDR-p53 hypothesis was not supported: DDR score and p53 response were weakly negatively correlated (Spearman rho = -0.40, p = 0.76), indistinguishable from random five-gene scores. OGT showed only a weak association (r = 0.23, p = 0.34), whereas CDO1 was a clear outlier (r = 0.70, empirical p = 0.0039). The 12-gene signature achieved a concordance index of 0.61 (p = 0.017) but a non-unique effect size. Overall, KOSMOS produced one well-supported discovery, one plausible but uncertain result, and one false hypothesis, illustrating that AI scientists can generate useful ideas but require rigorous auditing against appropriate null models.

**arXiv ID:** 2511.13825
</details>

<details>
<summary><strong>Deep reinforcement learning-based spacecraft attitude control with pointing keep-out constraint</strong> - Juntang Yang, Mohamed Khalil Ben-Larbi - [[pdf]](https://arxiv.org/pdf/2511.13746)</summary>

**Abstract:** This paper implements deep reinforcement learning (DRL) for spacecraft reorientation control with a single pointing keep-out zone. The Soft Actor-Critic (SAC) algorithm is adopted to handle continuous state and action space. A new state representation is designed to explicitly include a compact representation of the attitude constraint zone. The reward function is formulated to achieve the control objective while enforcing the attitude constraint. A curriculum learning approach is used for the agent training. Simulation results demonstrate the effectiveness of the proposed DRL-based method for spacecraft pointing-constrained attitude control.

**arXiv ID:** 2511.13746
</details>

<details>
<summary><strong>GRPO Privacy Is at Risk: A Membership Inference Attack Against Reinforcement Learning With Verifiable Rewards</strong> - Yule Liu, Heyi Zhang, Jinyi Zheng, Zhen Sun, Zifan Peng, Tianshuo Cong, Yilong Yang, Xinlei He, Zhuo Ma - [[pdf]](https://arxiv.org/pdf/2511.14045)</summary>

**Abstract:** Membership inference attacks (MIAs) on large language models (LLMs) pose significant privacy risks across various stages of model training. Recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) have brought a profound paradigm shift in LLM training, particularly for complex reasoning tasks. However, the on-policy nature of RLVR introduces a unique privacy leakage pattern: since training relies on self-generated responses without fixed ground-truth outputs, membership inference must now determine whether a given prompt (independent of any specific response) is used during fine-tuning. This creates a threat where leakage arises not from answer memorization.
To audit this novel privacy risk, we propose Divergence-in-Behavior Attack (DIBA), the first membership inference framework specifically designed for RLVR. DIBA shifts the focus from memorization to behavioral change, leveraging measurable shifts in model behavior across two axes: advantage-side improvement (e.g., correctness gain) and logit-side divergence (e.g., policy drift). Through comprehensive evaluations, we demonstrate that DIBA significantly outperforms existing baselines, achieving around 0.8 AUC and an order-of-magnitude higher TPR@0.1%FPR. We validate DIBA's superiority across multiple settings--including in-distribution, cross-dataset, cross-algorithm, black-box scenarios, and extensions to vision-language models. Furthermore, our attack remains robust under moderate defensive measures.
To the best of our knowledge, this is the first work to systematically analyze privacy vulnerabilities in RLVR, revealing that even in the absence of explicit supervision, training data exposure can be reliably inferred through behavioral traces.

**arXiv ID:** 2511.14045
</details>

<details>
<summary><strong>Object-Centric World Models for Causality-Aware Reinforcement Learning</strong> - Yosuke Nishimoto, Takashi Matsubara - [[pdf]](https://arxiv.org/pdf/2511.14262)</summary>

**Abstract:** World models have been developed to support sample-efficient deep reinforcement learning agents. However, it remains challenging for world models to accurately replicate environments that are high-dimensional, non-stationary, and composed of multiple objects with rich interactions since most world models learn holistic representations of all environmental components. By contrast, humans perceive the environment by decomposing it into discrete objects, facilitating efficient decision-making. Motivated by this insight, we propose \emph{Slot Transformer Imagination with CAusality-aware reinforcement learning} (STICA), a unified framework in which object-centric Transformers serve as the world model and causality-aware policy and value networks. STICA represents each observation as a set of object-centric tokens, together with tokens for the agent action and the resulting reward, enabling the world model to predict token-level dynamics and interactions. The policy and value networks then estimate token-level cause--effect relations and use them in the attention layers, yielding causality-guided decision-making. Experiments on object-rich benchmarks demonstrate that STICA consistently outperforms state-of-the-art agents in both sample efficiency and final performance.

**arXiv ID:** 2511.14262
</details>

<details>
<summary><strong>Tell Me: An LLM-powered Mental Well-being Assistant with RAG, Synthetic Dialogue Generation, and Agentic Planning</strong> - Trishala Jayesh Ahalpara - [[pdf]](https://arxiv.org/pdf/2511.14445)</summary>

**Abstract:** We present Tell Me, a mental well-being system that leverages advances in large language models to provide accessible, context-aware support for users and researchers. The system integrates three components: (i) a retrieval-augmented generation (RAG) assistant for personalized, knowledge-grounded dialogue; (ii) a synthetic client-therapist dialogue generator conditioned on client profiles to facilitate research on therapeutic language and data augmentation; and (iii) a Well-being AI crew, implemented with CrewAI, that produces weekly self-care plans and guided meditation audio. The system is designed as a reflective space for emotional processing rather than a substitute for professional therapy. It illustrates how conversational assistants can lower barriers to support, complement existing care, and broaden access to mental health resources. To address the shortage of confidential therapeutic data, we introduce synthetic client-therapist dialogue generation conditioned on client profiles. Finally, the planner demonstrates an innovative agentic workflow for dynamically adaptive, personalized self-care, bridging the limitations of static well-being tools. We describe the architecture, demonstrate its functionalities, and report evaluation of the RAG assistant in curated well-being scenarios using both automatic LLM-based judgments and a human-user study. This work highlights opportunities for interdisciplinary collaboration between NLP researchers and mental health professionals to advance responsible innovation in human-AI interaction for well-being.

**arXiv ID:** 2511.14445
</details>

<details>
<summary><strong>Agentic Video Intelligence: A Flexible Framework for Advanced Video Exploration and Understanding</strong> - Hong Gao, Yiming Bao, Xuezhen Tu, Yutong Xu, Yue Jin, Yiyang Mu, Bin Zhong, Linan Yue, Min-Ling Zhang - [[pdf]](https://arxiv.org/pdf/2511.14446)</summary>

**Abstract:** Video understanding requires not only visual recognition but also complex reasoning. While Vision-Language Models (VLMs) demonstrate impressive capabilities, they typically process videos largely in a single-pass manner with limited support for evidence revisit and iterative refinement. While recently emerging agent-based methods enable long-horizon reasoning, they either depend heavily on expensive proprietary models or require extensive agentic RL training. To overcome these limitations, we propose Agentic Video Intelligence (AVI), a flexible and training-free framework that can mirror human video comprehension through system-level design and optimization. AVI introduces three key innovations: (1) a human-inspired three-phase reasoning process (Retrieve-Perceive-Review) that ensures both sufficient global exploration and focused local analysis, (2) a structured video knowledge base organized through entity graphs, along with multi-granularity integrated tools, constituting the agent's interaction environment, and (3) an open-source model ensemble combining reasoning LLMs with lightweight base CV models and VLM, eliminating dependence on proprietary APIs or RL training. Experiments on LVBench, VideoMME-Long, LongVideoBench, and Charades-STA demonstrate that AVI achieves competitive performance while offering superior interpretability.

**arXiv ID:** 2511.14446
</details>

<details>
<summary><strong>KnowCoder-A1: Incentivizing Agentic Reasoning Capability with Outcome Supervision for KBQA</strong> - Zhuo Chen, Fei Wang, Zixuan Li, Zhao Zhang, Weiwei Ding, Chuanguang Yang, Yongjun Xu, Xiaolong Jin, Jiafeng Guo - [[pdf]](https://arxiv.org/pdf/2510.25101)</summary>

**Abstract:** Knowledge Base Question Answering (KBQA) aims to answer natural-language questions over a structured Knowledge Base (KB). Recent work improves KBQA by adopting an agentic reasoning paradigm, in which Large Language Models (LLMs) iteratively decompose a question, generate its corresponding logical queries, and interact with the KB to derive the answer. However, these methods typically fine-tune LLMs on reasoning trajectories synthesized via process supervision, which offers weak incentives for exploration and thus fails to strengthen the agentic reasoning ability. In this paper, we propose KnowCoder-A1, an LLM that can autonomously perform agentic reasoning on KBs to obtain answers. To incentivize autonomous exploration, KnowCoder-A1 trains the LLM under outcome-only supervision via a multi-stage curriculum reinforcement learning with an easy-to-hard curriculum. To establish foundational agentic capabilities, KnowCoder-A1 first fine-tunes the LLM on a small set of high-quality trajectories obtained through outcome-based rejection sampling. Then, to alleviate the reward sparsity inherent in outcome-only supervision, it applies multi-stage curriculum RL with reward schedules that progress from easy to hard. Trained with outcome-only supervision, KnowCoder-A1 exhibits powerful reasoning behaviors and consistently outperforms prior approaches across three mainstream datasets. Notably, on the zero-shot subset of GrailQA, KnowCoder-A1 achieves up to an 11.1% relative improvement while using only one-twelfth of the training data, demonstrating strong agentic reasoning capabilities.

**arXiv ID:** 2510.25101
</details>

<details>
<summary><strong>FairDICE: Fairness-Driven Offline Multi-Objective Reinforcement Learning</strong> - Woosung Kim, Jinho Lee, Jongmin Lee, Byung-Jun Lee - [[pdf]](https://arxiv.org/pdf/2506.08062)</summary>

**Abstract:** Multi-objective reinforcement learning (MORL) aims to optimize policies in the presence of conflicting objectives, where linear scalarization is commonly used to reduce vector-valued returns into scalar signals. While effective for certain preferences, this approach cannot capture fairness-oriented goals such as Nash social welfare or max-min fairness, which require nonlinear and non-additive trade-offs. Although several online algorithms have been proposed for specific fairness objectives, a unified approach for optimizing nonlinear welfare criteria in the offline setting-where learning must proceed from a fixed dataset-remains unexplored. In this work, we present FairDICE, the first offline MORL framework that directly optimizes nonlinear welfare objective. FairDICE leverages distribution correction estimation to jointly account for welfare maximization and distributional regularization, enabling stable and sample-efficient learning without requiring explicit preference weights or exhaustive weight search. Across multiple offline benchmarks, FairDICE demonstrates strong fairness-aware performance compared to existing baselines.

**arXiv ID:** 2506.08062
</details>

<details>
<summary><strong>EchoAgent: Guideline-Centric Reasoning Agent for Echocardiography Measurement and Interpretation</strong> - Matin Daghyani, Lyuyang Wang, Nima Hashemi, Bassant Medhat, Baraa Abdelsamad, Eros Rojas Velez, XiaoXiao Li, Michael Y. C. Tsang, Christina Luong, Teresa S.M. Tsang, Purang Abolmaesumi - [[pdf]](https://arxiv.org/pdf/2511.13948)</summary>

**Abstract:** Purpose: Echocardiographic interpretation requires video-level reasoning and guideline-based measurement analysis, which current deep learning models for cardiac ultrasound do not support. We present EchoAgent, a framework that enables structured, interpretable automation for this domain. Methods: EchoAgent orchestrates specialized vision tools under Large Language Model (LLM) control to perform temporal localization, spatial measurement, and clinical interpretation. A key contribution is a measurement-feasibility prediction model that determines whether anatomical structures are reliably measurable in each frame, enabling autonomous tool selection. We curated a benchmark of diverse, clinically validated video-query pairs for evaluation. Results: EchoAgent achieves accurate, interpretable results despite added complexity of spatiotemporal video analysis. Outputs are grounded in visual evidence and clinical guidelines, supporting transparency and traceability. Conclusion: This work demonstrates the feasibility of agentic, guideline-aligned reasoning for echocardiographic video analysis, enabled by task-specific tools and full video-level automation. EchoAgent sets a new direction for trustworthy AI in cardiac ultrasound.

**arXiv ID:** 2511.13948
</details>

<details>
<summary><strong>Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior Toward Beneficence or Harm</strong> - Baixiang Huang, Zhen Tan, Haoran Wang, Zijie Liu, Dawei Li, Ali Payani, Huan Liu, Tianlong Chen, Kai Shu - [[pdf]](https://arxiv.org/pdf/2506.20606)</summary>

**Abstract:** Agents based on Large Language Models (LLMs) have demonstrated strong capabilities across a wide range of tasks. However, deploying LLM-based agents in high-stakes domains comes with significant safety and ethical risks. Unethical behavior by these agents can directly result in serious real-world consequences, including physical harm and financial loss. To efficiently steer the ethical behavior of agents, we frame agent behavior steering as a model editing task, which we term Behavior Editing. Model editing is an emerging area of research that enables precise and efficient modifications to LLMs while preserving their overall capabilities. To systematically study and evaluate this approach, we introduce BehaviorBench, a multi-tier benchmark grounded in psychological moral theories. This benchmark supports both the evaluation and editing of agent behaviors across a variety of scenarios, with each tier introducing more complex and ambiguous scenarios. We first demonstrate that Behavior Editing can dynamically steer agents toward the target behavior within specific scenarios. Moreover, Behavior Editing enables not only scenario-specific local adjustments but also more extensive shifts in an agent's global moral alignment. We demonstrate that Behavior Editing can be used to promote ethical and benevolent behavior or, conversely, to induce harmful or malicious behavior. Through extensive evaluations of agents built on frontier LLMs, BehaviorBench validates the effectiveness of behavior editing across a wide range of models and scenarios. Our findings offer key insights into a new paradigm for steering agent behavior, highlighting both the promise and perils of Behavior Editing.

**arXiv ID:** 2506.20606
</details>

<details>
<summary><strong>MiroThinker: Pushing the Performance Boundaries of Open-Source Research Agents via Model, Context, and Interactive Scaling</strong> - MiroMind Team, Song Bai, Lidong Bing, Carson Chen, Guanzheng Chen, Yuntao Chen, Zhe Chen, Ziyi Chen, Jifeng Dai, Xuan Dong, Wenhan Dou, Yue Deng, Yunjie Fu, Junqi Ge, Chenxia Han, Tammy Huang, Zhenhang Huang, Jerry Jiao, Shilei Jiang, Tianyu Jiao, Xiaoqi Jian, Lei Lei, Ruilin Li, Ryan Luo, Tiantong Li, Xiang Lin, Ziyuan Liu, Zhiqi Li, Jie Ni, Qiang Ren, Pax Sun, Shiqian Su, Chenxin Tao, Bin Wang, Hellen Wang, Haonan Wang, James Wang, Jin Wang, Jojo Wang, Letian Wang, Shizun Wang, Weizhi Wang, Zixuan Wang, Jinfan Xu, Sen Xing, Chenyu Yang, Hai Ye, Jiaheng Yu, Yue Yu, Muyan Zhong, Tianchen Zhao, Xizhou Zhu, Yanpeng Zhou, Yifan Zhang, Zhi Zhu - [[pdf]](https://arxiv.org/pdf/2511.11793)</summary>

**Abstract:** We present MiroThinker v1.0, an open-source research agent designed to advance tool-augmented reasoning and information-seeking capabilities. Unlike previous agents that only scale up model size or context length, MiroThinker explores interaction scaling at the model level, systematically training the model to handle deeper and more frequent agent-environment interactions as a third dimension of performance improvement. Unlike LLM test-time scaling, which operates in isolation and risks degradation with longer reasoning chains, interactive scaling leverages environment feedback and external information acquisition to correct errors and refine trajectories. Through reinforcement learning, the model achieves efficient interaction scaling: with a 256K context window, it can perform up to 600 tool calls per task, enabling sustained multi-turn reasoning and complex real-world research workflows. Across four representative benchmarks-GAIA, HLE, BrowseComp, and BrowseComp-ZH-the 72B variant achieves up to 81.9%, 37.7%, 47.1%, and 55.6% accuracy respectively, surpassing previous open-source agents and approaching commercial counterparts such as GPT-5-high. Our analysis reveals that MiroThinker benefits from interactive scaling consistently: research performance improves predictably as the model engages in deeper and more frequent agent-environment interactions, demonstrating that interaction depth exhibits scaling behaviors analogous to model size and context length. These findings establish interaction scaling as a third critical dimension for building next-generation open research agents, complementing model capacity and context windows.

**arXiv ID:** 2511.11793
</details>

<details>
<summary><strong>O-Mem: Omni Memory System for Personalized, Long Horizon, Self-Evolving Agents</strong> - Piaohong Wang, Motong Tian, Jiaxian Li, Yuan Liang, Yuqing Wang, Qianben Chen, Tiannan Wang, Zhicong Lu, Jiawei Ma, Yuchen Eleanor Jiang, Wangchunshu Zhou - [[pdf]](https://arxiv.org/pdf/2511.13593)</summary>

**Abstract:** Recent advancements in LLM-powered agents have demonstrated significant potential in generating human-like responses; however, they continue to face challenges in maintaining long-term interactions within complex environments, primarily due to limitations in contextual consistency and dynamic personalization. Existing memory systems often depend on semantic grouping prior to retrieval, which can overlook semantically irrelevant yet critical user information and introduce retrieval noise. In this report, we propose the initial design of O-Mem, a novel memory framework based on active user profiling that dynamically extracts and updates user characteristics and event records from their proactive interactions with agents. O-Mem supports hierarchical retrieval of persona attributes and topic-related context, enabling more adaptive and coherent personalized responses. O-Mem achieves 51.67% on the public LoCoMo benchmark, a nearly 3% improvement upon LangMem,the previous state-of-the-art, and it achieves 62.99% on PERSONAMEM, a 3.5% improvement upon A-Mem,the previous state-of-the-art. O-Mem also boosts token and interaction response time efficiency compared to previous memory frameworks. Our work opens up promising directions for developing efficient and human-like personalized AI assistants in the future.

**arXiv ID:** 2511.13593
</details>

<details>
<summary><strong>Self-Supervised Multisensory Pretraining for Contact-Rich Robot Reinforcement Learning</strong> - Rickmer Krohn, Vignesh Prasad, Gabriele Tiboni, Georgia Chalvatzaki - [[pdf]](https://arxiv.org/pdf/2511.14427)</summary>

**Abstract:** Effective contact-rich manipulation requires robots to synergistically leverage vision, force, and proprioception. However, Reinforcement Learning agents struggle to learn in such multisensory settings, especially amidst sensory noise and dynamic changes. We propose MultiSensory Dynamic Pretraining (MSDP), a novel framework for learning expressive multisensory representations tailored for task-oriented policy learning. MSDP is based on masked autoencoding and trains a transformer-based encoder by reconstructing multisensory observations from only a subset of sensor embeddings, leading to cross-modal prediction and sensor fusion. For downstream policy learning, we introduce a novel asymmetric architecture, where a cross-attention mechanism allows the critic to extract dynamic, task-specific features from the frozen embeddings, while the actor receives a stable pooled representation to guide its actions. Our method demonstrates accelerated learning and robust performance under diverse perturbations, including sensor noise, and changes in object dynamics. Evaluations in multiple challenging, contact-rich robot manipulation tasks in simulation and the real world showcase the effectiveness of MSDP. Our approach exhibits strong robustness to perturbations and achieves high success rates on the real robot with as few as 6,000 online interactions, offering a simple yet powerful solution for complex multisensory robotic control.

**arXiv ID:** 2511.14427
</details>

<details>
<summary><strong>Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning</strong> - Ruoyu Qin, Weiran He, Weixiao Huang, Yangkun Zhang, Yikai Zhao, Bo Pang, Xinran Xu, Yingdi Shan, Yongwei Wu, Mingxing Zhang - [[pdf]](https://arxiv.org/pdf/2511.14617)</summary>

**Abstract:** Reinforcement Learning (RL) has become critical for advancing modern Large Language Models (LLMs), yet existing synchronous RL systems face severe performance bottlenecks. The rollout phase, which dominates end-to-end iteration time, suffers from substantial long-tail latency and poor resource utilization due to inherent workload imbalance. We present Seer, a novel online context learning system that addresses these challenges by exploiting previously overlooked similarities in output lengths and generation patterns among requests sharing the same prompt. Seer introduces three key techniques: divided rollout for dynamic load balancing, context-aware scheduling, and adaptive grouped speculative decoding. Together, these mechanisms substantially reduce long-tail latency and improve resource efficiency during rollout. Evaluations on production-grade RL workloads demonstrate that Seer improves end-to-end rollout throughput by 74% to 97% and reduces long-tail latency by 75% to 93% compared to state-of-the-art synchronous RL systems, significantly accelerating RL training iterations.

**arXiv ID:** 2511.14617
</details>

<details>
<summary><strong>SERL: Self-Examining Reinforcement Learning on Open-Domain</strong> - Weixuan Ou, Yanzhao Zheng, Shuoshuo Sun, Wei Zhang, Baohua Dong, Hangcheng Zhu, Ruohui Huang, Gang Yu, Pengwei Yan, Yifan Qiao - [[pdf]](https://arxiv.org/pdf/2511.07922)</summary>

**Abstract:** Reinforcement Learning (RL) has been shown to improve the capabilities of large language models (LLMs). However, applying RL to open-domain tasks faces two key challenges: (1) the inherent subjectivity of these tasks prevents the verifiable rewards as required by Reinforcement Learning with Verifiable Rewards (RLVR); (2) Reinforcement Learning from Human Feedback (RLHF) relies on external reward mechanisms. To overcome these limitations, we propose Self-Examining Reinforcement Learning (SERL), a novel self-improving framework where the LLM serves as both Actor and Judge. SERL introduces two synergistic reward mechanisms without any external signals. On the one hand, to improve the Actor's capability, we derive rewards from Copeland-style pairwise comparison judgments across a group of generated responses. On the other hand, a self-consistency reward that encourages coherent judgments is proposed to improve the Judge's reliability. This process refines the Judge's capability, which in turn provides a more robust reward for Actor. Experiments show that our method outperforms existing self-improvement training methods. SERL improves the LC win rate of Qwen3-8B on AlpacaEval 2 from 52.37% to 59.90%. To the best of our knowledge, our method achieves state-of-the-art performance among self-improving approaches. Furthermore, it achieves a performance comparable to significantly larger models like Qwen3-32B, demonstrating superior effectiveness and robustness on open-domain tasks.

**arXiv ID:** 2511.07922
</details>

<details>
<summary><strong>RoboTidy : A 3D Gaussian Splatting Household Tidying Benchmark for Embodied Navigation and Action</strong> - Xiaoquan Sun, Ruijian Zhang, Kang Pang, Bingchen Miao, Yuxiang Tan, Zhen Yang, Ming Li, Jiayu Chen - [[pdf]](https://arxiv.org/pdf/2511.14161)</summary>

**Abstract:** Household tidying is an important application area, yet current benchmarks neither model user preferences nor support mobility, and they generalize poorly, making it hard to comprehensively assess integrated language-to-action capabilities. To address this, we propose RoboTidy, a unified benchmark for language-guided household tidying that supports Vision-Language-Action (VLA) and Vision-Language-Navigation (VLN) training and evaluation. RoboTidy provides 500 photorealistic 3D Gaussian Splatting (3DGS) household scenes (covering 500 objects and containers) with collisions, formulates tidying as an "Action (Object, Container)" list, and supplies 6.4k high-quality manipulation demonstration trajectories and 1.5k naviagtion trajectories to support both few-shot and large-scale training. We also deploy RoboTidy in the real world for object tidying, establishing an end-to-end benchmark for household tidying. RoboTidy offers a scalable platform and bridges a key gap in embodied AI by enabling holistic and realistic evaluation of language-guided robots.

**arXiv ID:** 2511.14161
</details>

<details>
<summary><strong>MA-SLAM: Active SLAM in Large-Scale Unknown Environment using Map Aware Deep Reinforcement Learning</strong> - Yizhen Yin, Yuhua Qi, Dapeng Feng, Hongbo Chen, Hongjun Ma, Jin Wu, Yi Jiang - [[pdf]](https://arxiv.org/pdf/2511.14330)</summary>

**Abstract:** Active Simultaneous Localization and Mapping (Active SLAM) involves the strategic planning and precise control of a robotic system's movement in order to construct a highly accurate and comprehensive representation of its surrounding environment, which has garnered significant attention within the research community. While the current methods demonstrate efficacy in small and controlled settings, they face challenges when applied to large-scale and diverse environments, marked by extended periods of exploration and suboptimal paths of discovery. In this paper, we propose MA-SLAM, a Map-Aware Active SLAM system based on Deep Reinforcement Learning (DRL), designed to address the challenge of efficient exploration in large-scale environments. In pursuit of this objective, we put forward a novel structured map representation. By discretizing the spatial data and integrating the boundary points and the historical trajectory, the structured map succinctly and effectively encapsulates the visited regions, thereby serving as input for the deep reinforcement learning based decision module. Instead of sequentially predicting the next action step within the decision module, we have implemented an advanced global planner to optimize the exploration path by leveraging long-range target points. We conducted experiments in three simulation environments and deployed in a real unmanned ground vehicle (UGV), the results demonstrate that our approach significantly reduces both the duration and distance of exploration compared with state-of-the-art methods.

**arXiv ID:** 2511.14330
</details>

<details>
<summary><strong>Benchmarking Population-Based Reinforcement Learning across Robotic Tasks with GPU-Accelerated Simulation</strong> - Asad Ali Shahid, Yashraj Narang, Vincenzo Petrone, Enrico Ferrentino, Ankur Handa, Dieter Fox, Marco Pavone, Loris Roveda - [[pdf]](https://arxiv.org/pdf/2404.03336)</summary>

**Abstract:** In recent years, deep reinforcement learning (RL) has shown its effectiveness in solving complex continuous control tasks. However, this comes at the cost of an enormous amount of experience required for training, exacerbated by the sensitivity of learning efficiency and the policy performance to hyperparameter selection, which often requires numerous trials of time-consuming experiments. This work leverages a Population-Based Reinforcement Learning (PBRL) approach and a GPU-accelerated physics simulator to enhance the exploration capabilities of RL by concurrently training multiple policies in parallel. The PBRL framework is benchmarked against three state-of-the-art RL algorithms -- PPO, SAC, and DDPG -- dynamically adjusting hyperparameters based on the performance of learning agents. The experiments are performed on four challenging tasks in Isaac Gym -- Anymal Terrain, Shadow Hand, Humanoid, Franka Nut Pick -- by analyzing the effect of population size and mutation mechanisms for hyperparameters. The results show that PBRL agents achieve superior performance, in terms of cumulative reward, compared to non-evolutionary baseline agents. Moreover, the trained agents are finally deployed in the real world for a Franka Nut Pick task. To our knowledge, this is the first sim-to-real attempt for deploying PBRL agents on real hardware. Code and videos of the learned policies are available on our project website (this https URL).

**arXiv ID:** 2404.03336
</details>

<details>
<summary><strong>The Developments and Challenges towards Dexterous and Embodied Robotic Manipulation: A Survey</strong> - Gaofeng Li, Ruize Wang, Peisen Xu, Qi Ye, Jiming Chen - [[pdf]](https://arxiv.org/pdf/2507.11840)</summary>

**Abstract:** Achieving human-like dexterous robotic manipulation remains a central goal and a pivotal challenge in robotics. The development of Artificial Intelligence (AI) has allowed rapid progress in robotic manipulation. This survey summarizes the evolution of robotic manipulation from mechanical programming to embodied intelligence, alongside the transition from simple grippers to multi-fingered dexterous hands, outlining key characteristics and main challenges. Focusing on the current stage of embodied dexterous manipulation, we highlight recent advances in two critical areas: dexterous manipulation data collection (via simulation, human demonstrations, and teleoperation) and skill-learning frameworks (imitation and reinforcement learning). Then, based on the overview of the existing data collection paradigm and learning framework, three key challenges restricting the development of dexterous robotic manipulation are summarized and discussed.

**arXiv ID:** 2507.11840
</details>

<details>
<summary><strong>Tac2Motion: Contact-Aware Reinforcement Learning with Tactile Feedback for Robotic Hand Manipulation</strong> - Yitaek Kim, Casper Hewson Rask, Christoffer Sloth - [[pdf]](https://arxiv.org/pdf/2509.17812)</summary>

**Abstract:** This paper proposes Tac2Motion, a contact-aware reinforcement learning framework to facilitate the learning of contact-rich in-hand manipulation tasks, such as removing a lid. To this end, we propose tactile sensing-based reward shaping and incorporate the sensing into the observation space through embedding. The designed rewards encourage an agent to ensure firm grasping and smooth finger gaiting at the same time, leading to higher data efficiency and robust performance compared to the baseline. We verify the proposed framework on the opening a lid scenario, showing generalization of the trained policy into a couple of object types and various dynamics such as torsional friction. Lastly, the learned policy is demonstrated on the multi-fingered robot, Shadow Robot, showing that the control policy can be transferred to the real world. The video is available: this https URL.

**arXiv ID:** 2509.17812
</details>

<details>
<summary><strong>Large Language Models and 3D Vision for Intelligent Robotic Perception and Autonomy</strong> - Vinit Mehta, Charu Sharma, Karthick Thiyagarajan - [[pdf]](https://arxiv.org/pdf/2511.11777)</summary>

**Abstract:** With the rapid advancement of artificial intelligence and robotics, the integration of Large Language Models (LLMs) with 3D vision is emerging as a transformative approach to enhancing robotic sensing technologies. This convergence enables machines to perceive, reason and interact with complex environments through natural language and spatial understanding, bridging the gap between linguistic intelligence and spatial perception. This review provides a comprehensive analysis of state-of-the-art methodologies, applications and challenges at the intersection of LLMs and 3D vision, with a focus on next-generation robotic sensing technologies. We first introduce the foundational principles of LLMs and 3D data representations, followed by an in-depth examination of 3D sensing technologies critical for robotics. The review then explores key advancements in scene understanding, text-to-3D generation, object grounding and embodied agents, highlighting cutting-edge techniques such as zero-shot 3D segmentation, dynamic scene synthesis and language-guided manipulation. Furthermore, we discuss multimodal LLMs that integrate 3D data with touch, auditory and thermal inputs, enhancing environmental comprehension and robotic decision-making. To support future research, we catalog benchmark datasets and evaluation metrics tailored for 3D-language and vision tasks. Finally, we identify key challenges and future research directions, including adaptive model architectures, enhanced cross-modal alignment and real-time processing capabilities, which pave the way for more intelligent, context-aware and autonomous robotic sensing systems.

**arXiv ID:** 2511.11777
</details>

<details>
<summary><strong>CARScenes: Semantic VLM Dataset for Safe Autonomous Driving</strong> - Yuankai He, Weisong Shi - [[pdf]](https://arxiv.org/pdf/2511.10701)</summary>

**Abstract:** CAR-Scenes is a frame-level dataset for autonomous driving that enables training and evaluation of vision-language models (VLMs) for interpretable, scene-level understanding. We annotate 5,192 images drawn from Argoverse 1, Cityscapes, KITTI, and nuScenes using a 28-key category/sub-category knowledge base covering environment, road geometry, background-vehicle behavior, ego-vehicle behavior, vulnerable road users, sensor states, and a discrete severity scale (1-10), totaling 350+ leaf attributes. Labels are produced by a GPT-4o-assisted vision-language pipeline with human-in-the-loop verification; we release the exact prompts, post-processing rules, and per-field baseline model performance. CAR-Scenes also provides attribute co-occurrence graphs and JSONL records that support semantic retrieval, dataset triage, and risk-aware scenario mining across sources. To calibrate task difficulty, we include reproducible, non-benchmark baselines, notably a LoRA-tuned Qwen2-VL-2B with deterministic decoding, evaluated via scalar accuracy, micro-averaged F1 for list attributes, and severity MAE/RMSE on a fixed validation split. We publicly release the annotation and analysis scripts, including graph construction and evaluation scripts, to enable explainable, data-centric workflows for future intelligent vehicles. Dataset: this https URL

**arXiv ID:** 2511.10701
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
