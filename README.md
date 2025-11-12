# Agent arXiv Daily

**Last Updated:** 2025-11-12 02:49:33

**Total Papers:** 66

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
<summary><strong>A Supervised Autonomous Resection and Retraction Framework for Transurethral Enucleation of the Prostatic Median Lobe</strong> - Mariana Smith, Tanner Watts, Susheela Sharma Stern, Brendan Burkhart, Hao Li, Alejandro O. Chara, Nithesh Kumar, James Ferguson, Ayberk Acar, Jesse F. d'Almeida, Lauren Branscombe, Lauren Shepard, Ahmed Ghazi, Ipek Oguz, Jie Ying Wu, Robert J. Webster III, Axel Krieger, Alan Kuntz - [[pdf]](https://arxiv.org/pdf/2511.08490)</summary>

**Abstract:** Concentric tube robots (CTRs) offer dexterous motion at millimeter scales, enabling minimally invasive procedures through natural orifices. This work presents a coordinated model-based resection planner and learning-based retraction network that work together to enable semi-autonomous tissue resection using a dual-arm transurethral concentric tube robot (the Virtuoso). The resection planner operates directly on segmented CT volumes of prostate phantoms, automatically generating tool trajectories for a three-phase median lobe resection workflow: left/median trough resection, right/median trough resection, and median blunt dissection. The retraction network, PushCVAE, trained on surgeon demonstrations, generates retractions according to the procedural phase. The procedure is executed under Level-3 (supervised) autonomy on a prostate phantom composed of hydrogel materials that replicate the mechanical and cutting properties of tissue. As a feasibility study, we demonstrate that our combined autonomous system achieves a 97.1% resection of the targeted volume of the median lobe. Our study establishes a foundation for image-guided autonomy in transurethral robotic surgery and represents a first step toward fully automated minimally-invasive prostate enucleation.

**arXiv ID:** 2511.08490
</details>

<details>
<summary><strong>Designing Mental-Health Chatbots for Indian Adolescents: Mixed-Methods Evidence, a Boundary-Object Lens, and a Design-Tensions Framework</strong> - Neil K. R. Sehgal, Hita Kambhamettu, Sai Preethi Matam, Lyle Ungar, Sharath Chandra Guntuku - [[pdf]](https://arxiv.org/pdf/2511.07729)</summary>

**Abstract:** Mental health challenges among Indian adolescents are shaped by unique cultural and systemic barriers, including high social stigma and limited professional support. We report a mixed-methods study of Indian adolescents (survey n=362; interviews n=14) examining how they navigate mental-health challenges and engage with digital tools. Quantitative results highlight low self-stigma but significant social stigma, a preference for text over voice interactions, and low utilization of mental health apps but high smartphone access. Our qualitative findings reveal that while adolescents value privacy, emotional support, and localized content in mental health tools, existing chatbots lack personalization and cultural relevance. We contribute (1) a Design-Tensions framework; (2) an artifact-level probe; and (3) a boundary-objects account that specifies how chatbots mediate adolescents, peers, families, and services. This work advances culturally sensitive chatbot design by centering on underrepresented populations, addressing critical gaps in accessibility and support for adolescents in India.

**arXiv ID:** 2511.07729
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (9 papers)</h2></summary>

<details>
<summary><strong>ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents</strong> - Manasi Sharma, Chen Bo Calvin Zhang, Chaithanya Bandi, Clinton Wang, Ankit Aich, Huy Nghiem, Tahseen Rabbani, Ye Htet, Brian Jang, Sumana Basu, Aishwarya Balwani, Denis Peskoff, Marcos Ayestaran, Sean M. Hendryx, Brad Kenstler, Bing Liu - [[pdf]](https://arxiv.org/pdf/2511.07685)</summary>

**Abstract:** Deep Research (DR) is an emerging agent application that leverages large language models (LLMs) to address open-ended queries. It requires the integration of several capabilities, including multi-step reasoning, cross-document synthesis, and the generation of evidence-backed, long-form answers. Evaluating DR remains challenging because responses are lengthy and diverse, admit many valid solutions, and often depend on dynamic information sources. We introduce ResearchRubrics, a standardized benchmark for DR built with over 2,800+ hours of human labor that pairs realistic, domain-diverse prompts with 2,500+ expert-written, fine-grained rubrics to assess factual grounding, reasoning soundness, and clarity. We also propose a new complexity framework for categorizing DR tasks along three axes: conceptual breadth, logical nesting, and exploration. In addition, we develop human and model-based evaluation protocols that measure rubric adherence for DR agents. We evaluate several state-of-the-art DR systems and find that even leading agents like Gemini's DR and OpenAI's DR achieve under 68% average compliance with our rubrics, primarily due to missed implicit context and inadequate reasoning about retrieved information. Our results highlight the need for robust, scalable assessment of deep research capabilities, to which end we release ResearchRubrics(including all prompts, rubrics, and evaluation code) to facilitate progress toward well-justified research assistants.

**arXiv ID:** 2511.07685
</details>

<details>
<summary><strong>Towards a Standard, Enterprise-Relevant Agentic AI Benchmark: Lessons from 5.5 billion tokens' worth of agentic AI evaluations</strong> - JV Roig - [[pdf]](https://arxiv.org/pdf/2511.08042)</summary>

**Abstract:** Enterprise adoption of agentic AI systems requires reliable evaluation methods that reflect real-world deployment scenarios. Traditional LLM benchmarks suffer from training data contamination and fail to measure agentic capabilities such as multi-step tool use and decision-making under uncertainty. We present the Kamiwaza Agentic Merit Index (KAMI) v0.1, an enterprise-focused benchmark that addresses both contamination resistance and agentic evaluation. Through 170,000 LLM test items processing over 5.5 billion tokens across 35 model configurations, we demonstrate that traditional benchmark rankings poorly predict practical agentic performance. Notably, newer generation models like Llama 4 or Qwen 3 do not always outperform their older generation variants on enterprise-relevant tasks, contradicting traditional benchmark trends. We also present insights on cost-performance tradeoffs, model-specific behavioral patterns, and the impact of reasoning capabilities on token efficiency -- findings critical for enterprises making deployment decisions.

**arXiv ID:** 2511.08042
</details>

<details>
<summary><strong>Towards Outcome-Oriented, Task-Agnostic Evaluation of AI Agents</strong> - Waseem AlShikh, Muayad Sayed Ali, Brian Kennedy, Dmytro Mozolevskyi - [[pdf]](https://arxiv.org/pdf/2511.08242)</summary>

**Abstract:** As AI agents proliferate across industries and applications, evaluating their performance based solely on infrastructural metrics such as latency, time-to-first-token, or token throughput is proving insufficient. These metrics fail to capture the quality of an agent's decisions, its operational autonomy, or its ultimate business value. This white paper proposes a novel, comprehensive framework of eleven outcome-based, task-agnostic performance metrics for AI agents that transcend domain boundaries. These metrics are designed to enable organizations to evaluate agents based on the quality of their decisions, their degree of autonomy, their adaptability to new challenges, and the tangible business value they deliver, regardless of the underlying model architecture or specific use case. We introduce metrics such as Goal Completion Rate (GCR), Autonomy Index (AIx), Multi-Step Task Resilience (MTR), and Business Impact Efficiency (BIE). Through a large-scale simulated experiment involving four distinct agent architectures (ReAct, Chain-of-Thought, Tool-Augmented, Hybrid) across five diverse domains (Healthcare, Finance, Marketing, Legal, and Customer Service), we demonstrate the framework's efficacy. Our results reveal significant performance trade-offs between different agent designs, highlighting the Hybrid Agent as the most consistently high-performing model across the majority of our proposed metrics, achieving an average Goal Completion Rate of 88.8\% and the highest Return on Investment (ROI). This work provides a robust, standardized methodology for the holistic evaluation of AI agents, paving the way for more effective development, deployment, and governance.

**arXiv ID:** 2511.08242
</details>

<details>
<summary><strong>AI-Powered Data Visualization Platform: An Intelligent Web Application for Automated Dataset Analysis</strong> - Srihari R, Pallavi M, Tejaswini S, Vaishnavi R C - [[pdf]](https://arxiv.org/pdf/2511.08363)</summary>

**Abstract:** An AI-powered data visualization platform that automates the entire data analysis process, from uploading a dataset to generating an interactive visualization. Advanced machine learning algorithms are employed to clean and preprocess the data, analyse its features, and automatically select appropriate visualizations. The system establishes the process of automating AI-based analysis and visualization from the context of data-driven environments, and eliminates the challenge of time-consuming manual data analysis. The combination of a Python Flask backend to access the dataset, paired with a React frontend, provides a robust platform that automatically interacts with Firebase Cloud Storage for numerous data processing and data analysis solutions and real-time sources. Key contributions include automatic and intelligent data cleaning, with imputation for missing values, and detection of outliers, via analysis of the data set. AI solutions to intelligently select features, using four different algorithms, and intelligent title generation and visualization are determined by the attributes of the dataset. These contributions were evaluated using two separate datasets to assess the platform's performance. In the process evaluation, the initial analysis was performed in real-time on datasets as large as 100000 rows, while the cloud-based demand platform scales to meet requests from multiple users and processes them simultaneously. In conclusion, the cloud-based data visualization application allowed for a significant reduction of manual inputs to the data analysis process while maintaining a high quality, impactful visual outputs, and user experiences

**arXiv ID:** 2511.08363
</details>

<details>
<summary><strong>Dataset Safety in Autonomous Driving: Requirements, Risks, and Assurance</strong> - Alireza Abbaspour, Tejaskumar Balgonda Patil, B Ravi Kiran, Russel Mohr, Senthil Yogamani - [[pdf]](https://arxiv.org/pdf/2511.08439)</summary>

**Abstract:** Dataset integrity is fundamental to the safety and reliability of AI systems, especially in autonomous driving. This paper presents a structured framework for developing safe datasets aligned with ISO/PAS 8800 guidelines. Using AI-based perception systems as the primary use case, it introduces the AI Data Flywheel and the dataset lifecycle, covering data collection, annotation, curation, and maintenance. The framework incorporates rigorous safety analyses to identify hazards and mitigate risks caused by dataset insufficiencies. It also defines processes for establishing dataset safety requirements and proposes verification and validation strategies to ensure compliance with safety standards. In addition to outlining best practices, the paper reviews recent research and emerging trends in dataset safety and autonomous vehicle development, providing insights into current challenges and future directions. By integrating these perspectives, the paper aims to advance robust, safety-assured AI systems for autonomous driving applications.

**arXiv ID:** 2511.08439
</details>

<details>
<summary><strong>MARC: Multimodal and Multi-Task Agentic Retrieval-Augmented Generation for Cold-Start Recommender System</strong> - Seung Hwan Cho, Yujin Yang, Danik Baeck, Minjoo Kim, Young-Min Kim, Heejung Lee, Sangjin Park - [[pdf]](https://arxiv.org/pdf/2511.08181)</summary>

**Abstract:** Recommender systems (RS) are currently being studied to mitigate limitations during cold-start conditions by leveraging modality information or introducing Agent concepts based on the exceptional reasoning capabilities of Large Language Models (LLMs). Meanwhile, food and beverage recommender systems have traditionally used knowledge graph and ontology concepts due to the domain's unique data attributes and relationship characteristics. On this background, we propose MARC, a multimodal and multi-task cocktail recommender system based on Agentic Retrieval-Augmented Generation (RAG) utilizing graph database under cold-start conditions. The proposed system generates high-quality, contextually appropriate answers through two core processes: a task recognition router and a reflection process. The graph database was constructed by processing cocktail data from Kaggle, and its effectiveness was evaluated using 200 manually crafted questions. The evaluation used both LLM-as-a-judge and human evaluation to demonstrate that answers generated via the graph database outperformed those from a simple vector database in terms of quality. The code is available at this https URL

**arXiv ID:** 2511.08181
</details>

<details>
<summary><strong>AgentFlux: Decoupled Fine-Tuning & Inference for On-Device Agentic Systems</strong> - Rohan Kadekodi, Zhan Jin, Keisuke Kamahori, Yile Gu, Sean Khatiri, Noah H. Bayindirli, Sergey Gorbunov, Baris Kasikci - [[pdf]](https://arxiv.org/pdf/2510.00229)</summary>

**Abstract:** The deployment of Large Language Models (LLMs) as agentic orchestrators has revolutionized task automation, but the need for privacy-preserving, cost-effective solutions demands on-device inference capabilities. However, local LLMs consistently underperform compared to frontier models in tool calling scenarios, struggling with both tool selection from large tool sets and accurate argument generation for complex parameter structures. We introduce a methodology that disaggregates a tool-calling task into two distinct subtasks: tool selection and argument generation. We propose "decoupled fine-tuning", a novel post-training approach that employs LoRA fine-tuning to create dedicated LoRA adapters for tool selection and tool-specific argument generation using separate loss masking for each of the subtasks. Furthermore, we present DualTune, an inference framework that leverages the LoRA adapters created using decoupled fine-tuning to perform efficient agent orchestration with the help of local models on end-user devices. DualTune decomposes the tool-call generation step into tool selection and argument generation, and dynamically loads the corresponding LoRA adapters to generate tool calls. Additionally, DualTune implements hierarchical orchestration to restrict the number of tools required for tool selection. Our experiments on the MCP-Bench benchmark demonstrate that the Qwen-2.5-7B model trained using decoupled fine-tuning improves the tool calling accuracy of the base model by 46%, and outperforms other local reasoning, non-reasoning and fine-tuned models of similar size in all cases, and models that are 2x larger, in most cases.

**arXiv ID:** 2510.00229
</details>

<details>
<summary><strong>How Brittle is Agent Safety? Rethinking Agent Risk under Intent Concealment and Task Complexity</strong> - Zihan Ma, Dongsheng Zhu, Shudong Liu, Taolin Zhang, Junnan Liu, Qingqiu Li, Minnan Luo, Songyang Zhang, Kai Chen - [[pdf]](https://arxiv.org/pdf/2511.08487)</summary>

**Abstract:** Current safety evaluations for LLM-driven agents primarily focus on atomic harms, failing to address sophisticated threats where malicious intent is concealed or diluted within complex tasks. We address this gap with a two-dimensional analysis of agent safety brittleness under the orthogonal pressures of intent concealment and task complexity. To enable this, we introduce OASIS (Orthogonal Agent Safety Inquiry Suite), a hierarchical benchmark with fine-grained annotations and a high-fidelity simulation sandbox. Our findings reveal two critical phenomena: safety alignment degrades sharply and predictably as intent becomes obscured, and a "Complexity Paradox" emerges, where agents seem safer on harder tasks only due to capability limitations. By releasing OASIS and its simulation environment, we provide a principled foundation for probing and strengthening agent safety in these overlooked dimensions.

**arXiv ID:** 2511.08487
</details>

<details>
<summary><strong>Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving</strong> - Yi Huang, Zhan Qu, Lihui Jiang, Bingbing Liu, Hongbo Zhang - [[pdf]](https://arxiv.org/pdf/2511.08214)</summary>

**Abstract:** End-to-end autonomous driving systems, predominantly trained through imitation learning, have demonstrated considerable effectiveness in leveraging large-scale expert driving data. Despite their success in open-loop evaluations, these systems often exhibit significant performance degradation in closed-loop scenarios due to causal confusion. This confusion is fundamentally exacerbated by the overreliance of the imitation learning paradigm on expert trajectories, which often contain unattributable noise and interfere with the modeling of causal relationships between environmental contexts and appropriate driving actions.
To address this fundamental limitation, we propose Perception-Guided Self-Supervision (PGS) - a simple yet effective training paradigm that leverages perception outputs as the primary supervisory signals, explicitly modeling causal relationships in decision-making. The proposed framework aligns both the inputs and outputs of the decision-making module with perception results, such as lane centerlines and the predicted motions of surrounding agents, by introducing positive and negative self-supervision for the ego trajectory. This alignment is specifically designed to mitigate causal confusion arising from the inherent noise in expert trajectories.
Equipped with perception-driven supervision, our method, built on a standard end-to-end architecture, achieves a Driving Score of 78.08 and a mean success rate of 48.64% on the challenging closed-loop Bench2Drive benchmark, significantly outperforming existing state-of-the-art methods, including those employing more complex network architectures and inference pipelines. These results underscore the effectiveness and robustness of the proposed PGS framework and point to a promising direction for addressing causal confusion and enhancing real-world generalization in autonomous driving.

**arXiv ID:** 2511.08214
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>Network and Systems Performance Characterization of MCP-Enabled LLM Agents</strong> - Zihao Ding, Mufeng Zhu, Yao Liu - [[pdf]](https://arxiv.org/pdf/2511.07426)</summary>

**Abstract:** Model Context Protocol (MCP) has recently gained increased attention within the AI community for providing a standardized way for large language models (LLMs) to interact with external tools and services, significantly enhancing their capabilities. However, the inclusion of extensive contextual information, including system prompts, MCP tool definitions, and context histories, in MCP-enabled LLM interactions, dramatically inflates token usage. Given that LLM providers charge based on tokens, these expanded contexts can quickly escalate monetary costs and increase the computational load on LLM services. This paper presents a comprehensive measurement-based analysis of MCP-enabled interactions with LLMs, revealing trade-offs between capability, performance, and cost. We explore how different LLM models and MCP configurations impact key performance metrics such as token efficiency, monetary cost, task completion times, and task success rates, and suggest potential optimizations, including enabling parallel tool calls and implementing robust task abort mechanisms. These findings provide useful insights for developing more efficient, robust, and cost-effective MCP-enabled workflows.

**arXiv ID:** 2511.07426
</details>

<details>
<summary><strong>From Experience to Strategy: Empowering LLM Agents with Trainable Graph Memory</strong> - Siyu Xia, Zekun Xu, Jiajun Chai, Wentian Fan, Yan Song, Xiaohan Wang, Guojun Yin, Wei Lin, Haifeng Zhang, Jun Wang - [[pdf]](https://arxiv.org/pdf/2511.07800)</summary>

**Abstract:** Large Language Models (LLMs) based agents have demonstrated remarkable potential in autonomous task-solving across complex, open-ended environments. A promising approach for improving the reasoning capabilities of LLM agents is to better utilize prior experiences in guiding current decisions. However, LLMs acquire experience either through implicit memory via training, which suffers from catastrophic forgetting and limited interpretability, or explicit memory via prompting, which lacks adaptability. In this paper, we introduce a novel agent-centric, trainable, multi-layered graph memory framework and evaluate how context memory enhances the ability of LLMs to utilize parametric information. The graph abstracts raw agent trajectories into structured decision paths in a state machine and further distills them into high-level, human-interpretable strategic meta-cognition. In order to make memory adaptable, we propose a reinforcement-based weight optimization procedure that estimates the empirical utility of each meta-cognition based on reward feedback from downstream tasks. These optimized strategies are then dynamically integrated into the LLM agent's training loop through meta-cognitive prompting. Empirically, the learnable graph memory delivers robust generalization, improves LLM agents' strategic reasoning performance, and provides consistent benefits during Reinforcement Learning (RL) training.

**arXiv ID:** 2511.07800
</details>

<details>
<summary><strong>AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress</strong> - Zhiheng Xi, Chenyang Liao, Guanyu Li, Yajie Yang, Wenxiang Chen, Zhihao Zhang, Binghai Wang, Senjie Jin, Yuhao Zhou, Jian Guan, Wei Wu, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang - [[pdf]](https://arxiv.org/pdf/2511.08325)</summary>

**Abstract:** Despite rapid development, large language models (LLMs) still encounter challenges in multi-turn decision-making tasks (i.e., agent tasks) like web shopping and browser navigation, which require making a sequence of intelligent decisions based on environmental feedback. Previous work for LLM agents typically relies on elaborate prompt engineering or fine-tuning with expert trajectories to improve performance. In this work, we take a different perspective: we explore constructing process reward models (PRMs) to evaluate each decision and guide the agent's decision-making process. Unlike LLM reasoning, where each step is scored based on correctness, actions in agent tasks do not have a clear-cut correctness. Instead, they should be evaluated based on their proximity to the goal and the progress they have made. Building on this insight, we propose a re-defined PRM for agent tasks, named AgentPRM, to capture both the interdependence between sequential decisions and their contribution to the final goal. This enables better progress tracking and exploration-exploitation balance. To scalably obtain labeled data for training AgentPRM, we employ a Temporal Difference-based (TD-based) estimation method combined with Generalized Advantage Estimation (GAE), which proves more sample-efficient than prior methods. Extensive experiments across different agentic tasks show that AgentPRM is over $8\times$ more compute-efficient than baselines, and it demonstrates robust improvement when scaling up test-time compute. Moreover, we perform detailed analyses to show how our method works and offer more insights, e.g., applying AgentPRM to the reinforcement learning of LLM agents.

**arXiv ID:** 2511.08325
</details>

<details>
<summary><strong>AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments</strong> - Zikang Leng, Megha Thukral, Yaqi Liu, Hrudhai Rajasekhar, Shruthi K. Hiremath, Jiaman He, Thomas Plötz - [[pdf]](https://arxiv.org/pdf/2506.11773)</summary>

**Abstract:** A major challenge in developing robust and generalizable Human Activity Recognition (HAR) systems for smart homes is the lack of large and diverse labeled datasets. Variations in home layouts, sensor configurations, and individual behaviors further exacerbate this issue. To address this, we leverage the idea of embodied AI agents -- virtual agents that perceive and act within simulated environments guided by internal world models. We introduce AgentSense, a virtual data generation pipeline in which agents live out daily routines in simulated smart homes, with behavior guided by Large Language Models (LLMs). The LLM generates diverse synthetic personas and realistic routines grounded in the environment, which are then decomposed into fine-grained actions. These actions are executed in an extended version of the VirtualHome simulator, which we augment with virtual ambient sensors that record the agents' activities. Our approach produces rich, privacy-preserving sensor data that reflects real-world diversity. We evaluate AgentSense on five real HAR datasets. Models pretrained on the generated data consistently outperform baselines, especially in low-resource settings. Furthermore, combining the generated virtual sensor data with a small amount of real data achieves performance comparable to training on full real-world datasets. These results highlight the potential of using LLM-guided embodied agents for scalable and cost-effective sensor data generation in HAR. Our code is publicly available at this https URL.

**arXiv ID:** 2506.11773
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (24 papers)</h2></summary>

<details>
<summary><strong>Agentic Educational Content Generation for African Languages on Edge Devices</strong> - Ravi Gupta, Guneet Bhatia - [[pdf]](https://arxiv.org/pdf/2511.07437)</summary>

**Abstract:** Addressing educational inequity in Sub-Saharan Africa, this research presents an autonomous agent-orchestrated framework for decentralized, culturally adaptive educational content generation on edge devices. The system leverages four specialized agents that work together to generate contextually appropriate educational content. Experimental validation on platforms including Raspberry Pi 4B and NVIDIA Jetson Nano demonstrates significant performance achievements. InkubaLM on Jetson Nano achieved a Time-To-First-Token (TTFT) of 129 ms, an average inter-token latency of 33 ms, and a throughput of 45.2 tokens per second while consuming 8.4 W. On Raspberry Pi 4B, InkubaLM also led with 326 ms TTFT and 15.9 tokens per second at 5.8 W power consumption. The framework consistently delivered high multilingual quality, averaging a BLEU score of 0.688, cultural relevance of 4.4/5, and fluency of 4.2/5 across tested African languages. Through potential partnerships with active community organizations including African Youth & Community Organization (AYCO) and Florida Africa Foundation, this research aims to establish a practical foundation for accessible, localized, and sustainable AI-driven education in resource-constrained environments. Keeping focus on long-term viability and cultural appropriateness, it contributes to United Nations SDGs 4, 9, and 10. Index Terms - Multi-Agent Systems, Edge AI Computing, Educational Technology, African Languages, Rural Education, Sustainable Development, UN SDG.

**arXiv ID:** 2511.07437
</details>

<details>
<summary><strong>SciAgent: A Unified Multi-Agent System for Generalistic Scientific Reasoning</strong> - Xuchen Li, Ruitao Wu, Xuanbo Liu, Xukai Wang, Jinbo Hu, Zhixin Bai, Bohan Zeng, Hao Liang, Leheng Chen, Mingrui Chen, Haitian Zhong, Xuanlin Yang, Xu-Yao Zhang, Liu Liu, Jia Li, Kaiqi Huang, Jiahao Xu, Haitao Mi, Wentao Zhang, Bin Dong - [[pdf]](https://arxiv.org/pdf/2511.08151)</summary>

**Abstract:** Recent advances in large language models have enabled AI systems to achieve expert-level performance on domain-specific scientific tasks, yet these systems remain narrow and handcrafted. We introduce SciAgent, a unified multi-agent system designed for generalistic scientific reasoning-the ability to adapt reasoning strategies across disciplines and difficulty levels. SciAgent organizes problem solving as a hierarchical process: a Coordinator Agent interprets each problem's domain and complexity, dynamically orchestrating specialized Worker Systems, each composed of interacting reasoning Sub-agents for symbolic deduction, conceptual modeling, numerical computation, and verification. These agents collaboratively assemble and refine reasoning pipelines tailored to each task. Across mathematics and physics Olympiads (IMO, IMC, IPhO, CPhO), SciAgent consistently attains or surpasses human gold-medalist performance, demonstrating both domain generality and reasoning adaptability. Additionally, SciAgent has been tested on the International Chemistry Olympiad (IChO) and selected problems from the Humanity's Last Exam (HLE) benchmark, further confirming the system's ability to generalize across diverse scientific domains. This work establishes SciAgent as a concrete step toward generalistic scientific intelligence-AI systems capable of coherent, cross-disciplinary reasoning at expert levels.

**arXiv ID:** 2511.08151
</details>

<details>
<summary><strong>MADD: Multi-Agent Drug Discovery Orchestra</strong> - Gleb V. Solovev, Alina B. Zhidkovskaya, Anastasia Orlova, Nina Gubina, Anastasia Vepreva, Rodion Golovinskii, Ilya Tonkii, Ivan Dubrovsky, Ivan Gurev, Dmitry Gilemkhanov, Denis Chistiakov, Timur A. Aliev, Ivan Poddiakov, Galina Zubkova, Ekaterina V. Skorb, Vladimir Vinogradov, Alexander Boukhanovsky, Nikolay Nikitin, Andrei Dmitrenko, Anna Kalyuzhnaya, Andrey Savchenko - [[pdf]](https://arxiv.org/pdf/2511.08217)</summary>

**Abstract:** Hit identification is a central challenge in early drug discovery, traditionally requiring substantial experimental resources. Recent advances in artificial intelligence, particularly large language models (LLMs), have enabled virtual screening methods that reduce costs and improve efficiency. However, the growing complexity of these tools has limited their accessibility to wet-lab researchers. Multi-agent systems offer a promising solution by combining the interpretability of LLMs with the precision of specialized models and tools. In this work, we present MADD, a multi-agent system that builds and executes customized hit identification pipelines from natural language queries. MADD employs four coordinated agents to handle key subtasks in de novo compound generation and screening. We evaluate MADD across seven drug discovery cases and demonstrate its superior performance compared to existing LLM-based solutions. Using MADD, we pioneer the application of AI-first drug design to five biological targets and release the identified hit molecules. Finally, we introduce a new benchmark of query-molecule pairs and docking scores for over three million compounds to contribute to the agentic future of drug design.

**arXiv ID:** 2511.08217
</details>

<details>
<summary><strong>Multi-Agent GraphRAG: A Text-to-Cypher Framework for Labeled Property Graphs</strong> - Anton Gusarov, Anastasia Volkova, Valentin Khrulkov, Andrey Kuznetsov, Evgenii Maslov, Ivan Oseledets - [[pdf]](https://arxiv.org/pdf/2511.08274)</summary>

**Abstract:** While Retrieval-Augmented Generation (RAG) methods commonly draw information from unstructured documents, the emerging paradigm of GraphRAG aims to leverage structured data such as knowledge graphs. Most existing GraphRAG efforts focus on Resource Description Framework (RDF) knowledge graphs, relying on triple representations and SPARQL queries. However, the potential of Cypher and Labeled Property Graph (LPG) databases to serve as scalable and effective reasoning engines within GraphRAG pipelines remains underexplored in current research literature. To fill this gap, we propose Multi-Agent GraphRAG, a modular LLM agentic system for text-to-Cypher query generation serving as a natural language interface to LPG-based graph data. Our proof-of-concept system features an LLM-based workflow for automated Cypher queries generation and execution, using Memgraph as the graph database backend. Iterative content-aware correction and normalization, reinforced by an aggregated feedback loop, ensures both semantic and syntactic refinement of generated queries. We evaluate our system on the CypherBench graph dataset covering several general domains with diverse types of queries. In addition, we demonstrate performance of the proposed workflow on a property graph derived from the IFC (Industry Foundation Classes) data, representing a digital twin of a building. This highlights how such an approach can bridge AI with real-world applications at scale, enabling industrial digital automation use cases.

**arXiv ID:** 2511.08274
</details>

<details>
<summary><strong>A Negotiation-Based Multi-Agent Reinforcement Learning Approach for Dynamic Scheduling of Reconfigurable Manufacturing Systems</strong> - Manonmani Sekar, Nasim Nezamoddini - [[pdf]](https://arxiv.org/pdf/2511.07707)</summary>

**Abstract:** Reconfigurable manufacturing systems (RMS) are critical for future market adjustment given their rapid adaptation to fluctuations in consumer demands, the introduction of new technological advances, and disruptions in linked supply chain sections. The adjustable hard settings of such systems require a flexible soft planning mechanism that enables realtime production planning and scheduling amid the existing complexity and variability in their configuration settings. This study explores the application of multi agent reinforcement learning (MARL) for dynamic scheduling in soft planning of the RMS settings. In the proposed framework, deep Qnetwork (DQN) agents trained in centralized training learn optimal job machine assignments in real time while adapting to stochastic events such as machine breakdowns and reconfiguration delays. The model also incorporates a negotiation with an attention mechanism to enhance state representation and improve decision focus on critical system features. Key DQN enhancements including prioritized experience replay, nstep returns, double DQN and soft target update are used to stabilize and accelerate learning. Experiments conducted in a simulated RMS environment demonstrate that the proposed approach outperforms baseline heuristics in reducing makespan and tardiness while improving machine utilization. The reconfigurable manufacturing environment was extended to simulate realistic challenges, including machine failures and reconfiguration times. Experimental results show that while the enhanced DQN agent is effective in adapting to dynamic conditions, machine breakdowns increase variability in key performance metrics such as makespan, throughput, and total tardiness. The results confirm the advantages of applying the MARL mechanism for intelligent and adaptive scheduling in dynamic reconfigurable manufacturing environments.

**arXiv ID:** 2511.07707
</details>

<details>
<summary><strong>Adaptive Multi-Agent Response Refinement in Conversational Systems</strong> - Soyeong Jeong, Aparna Elangovan, Emine Yilmaz, Oleg Rokhlenko - [[pdf]](https://arxiv.org/pdf/2511.08319)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated remarkable success in conversational systems by generating human-like responses. However, they can fall short, especially when required to account for personalization or specific knowledge. In real-life settings, it is impractical to rely on users to detect these errors and request a new response. One way to address this problem is to refine the response before returning it to the user. While existing approaches focus on refining responses within a single LLM, this method struggles to consider diverse aspects needed for effective conversations. In this work, we propose refining responses through a multi-agent framework, where each agent is assigned a specific role for each aspect. We focus on three key aspects crucial to conversational quality: factuality, personalization, and coherence. Each agent is responsible for reviewing and refining one of these aspects, and their feedback is then merged to improve the overall response. To enhance collaboration among them, we introduce a dynamic communication strategy. Instead of following a fixed sequence of agents, our approach adaptively selects and coordinates the most relevant agents based on the specific requirements of each query. We validate our framework on challenging conversational datasets, demonstrating that ours significantly outperforms relevant baselines, particularly in tasks involving knowledge or user's persona, or both.

**arXiv ID:** 2511.08319
</details>

<details>
<summary><strong>Designing LLM-based Multi-Agent Systems for Software Engineering Tasks: Quality Attributes, Design Patterns and Rationale</strong> - Yangxiao Cai, Ruiyin Li, Peng Liang, Mojtaba Shahin, Zengyang Li - [[pdf]](https://arxiv.org/pdf/2511.08475)</summary>

**Abstract:** As the complexity of Software Engineering (SE) tasks continues to escalate, Multi-Agent Systems (MASs) have emerged as a focal point of research and practice due to their autonomy and scalability. Furthermore, through leveraging the reasoning and planning capabilities of Large Language Models (LLMs), the application of LLM-based MASs in the field of SE is garnering increasing attention. However, there is no dedicated study that systematically explores the design of LLM-based MASs, including the Quality Attributes (QAs) on which the designers mainly focus, the design patterns used by the designers, and the rationale guiding the design of LLM-based MASs for SE tasks. To this end, we conducted a study to identify the QAs that LLM-based MASs for SE tasks focus on, the design patterns used in the MASs, and the design rationale for the MASs. We collected 94 papers on LLM-based MASs for SE tasks as the source. Our study shows that: (1) Code Generation is the most common SE task solved by LLM-based MASs among ten identified SE tasks, (2) Functional Suitability is the QA on which designers of LLM-based MASs pay the most attention, (3) Role-Based Cooperation is the design pattern most frequently employed among 16 patterns used to construct LLM-based MASs, and (4) Improving the Quality of Generated Code is the most common rationale behind the design of LLM-based MASs. Based on the study results, we presented the implications for the design of LLM-based MASs to support SE tasks.

**arXiv ID:** 2511.08475
</details>

<details>
<summary><strong>Understanding Electro-communication and Electro-sensing in Weakly Electric Fish using Multi-Agent Deep Reinforcement Learning</strong> - Satpreet H. Singh, Sonja Johnson-Yu, Zhouyang Lu, Aaron Walsman, Federico Pedraja, Denis Turcu, Pratyusha Sharma, Naomi Saphra, Nathaniel B. Sawtell, Kanaka Rajan - [[pdf]](https://arxiv.org/pdf/2511.08436)</summary>

**Abstract:** Weakly electric fish, like Gnathonemus petersii, use a remarkable electrical modality for active sensing and communication, but studying their rich electrosensing and electrocommunication behavior and associated neural activity in naturalistic settings remains experimentally challenging. Here, we present a novel biologically-inspired computational framework to study these behaviors, where recurrent neural network (RNN) based artificial agents trained via multi-agent reinforcement learning (MARL) learn to modulate their electric organ discharges (EODs) and movement patterns to collectively forage in virtual environments. Trained agents demonstrate several emergent features consistent with real fish collectives, including heavy tailed EOD interval distributions, environmental context dependent shifts in EOD interval distributions, and social interaction patterns like freeloading, where agents reduce their EOD rates while benefiting from neighboring agents' active sensing. A minimal two-fish assay further isolates the role of electro-communication, showing that access to conspecific EODs and relative dominance jointly shape foraging success. Notably, these behaviors emerge through evolution-inspired rewards for individual fitness and emergent inter-agent interactions, rather than through rewarding agents explicitly for social interactions. Our work has broad implications for the neuroethology of weakly electric fish, as well as other social, communicating animals in which extensive recordings from multiple individuals, and thus traditional data-driven modeling, are infeasible.

**arXiv ID:** 2511.08436
</details>

<details>
<summary><strong>SOCIA-$\nabla$: Textual Gradient Meets Multi-Agent Orchestration for Automated Simulator Generation</strong> - Yuncheng Hua, Sion Weatherhead, Mehdi Jafari, Hao Xue, Flora D. Salim - [[pdf]](https://arxiv.org/pdf/2505.12006)</summary>

**Abstract:** In this paper, we present SOCIA-$\nabla$, an end-to-end, agentic framework that treats simulator construction asinstance optimization over code within a textual computation graph. Specialized LLM-driven agents are embedded as graph nodes, and a workflow manager executes a loss-driven loop: code synthesis -> execution -> evaluation -> code repair. The optimizer performs Textual-Gradient Descent (TGD), while human-in-the-loop interaction is reserved for task-spec confirmation, minimizing expert effort and keeping the code itself as the trainable object. Across three CPS tasks, i.e., User Modeling, Mask Adoption, and Personal Mobility, SOCIA-$\nabla$ attains state-of-the-art overall accuracy. By unifying multi-agent orchestration with a loss-aligned optimization view, SOCIA-$\nabla$ converts brittle prompt pipelines into reproducible, constraint-aware simulator code generation that scales across domains and simulation granularities. We will release the code soon.

**arXiv ID:** 2505.12006
</details>

<details>
<summary><strong>Question-to-Knowledge (Q2K): Multi-Agent Generation of Inspectable Facts for Product Mapping</strong> - Wonduk Seo, Taesub Shin, Hyunjin An, Dokyun Kim, Seunghyun Lee - [[pdf]](https://arxiv.org/pdf/2509.01182)</summary>

**Abstract:** Identifying whether two product listings refer to the same Stock Keeping Unit (SKU) is a persistent challenge in ecommerce, especially when explicit identifiers are missing and product names vary widely across platforms. Rule based heuristics and keyword similarity often misclassify products by overlooking subtle distinctions in brand, specification, or bundle configuration. To overcome these limitations, we propose Question to Knowledge (Q2K), a multi agent framework that leverages Large Language Models (LLMs) for reliable SKU mapping. Q2K integrates: (1) a Reasoning Agent that generates targeted disambiguation questions, (2) a Knowledge Agent that resolves them via focused web searches, and (3) a Deduplication Agent that reuses validated reasoning traces to reduce redundancy and ensure consistency. A human in the loop mechanism further refines uncertain cases. Experiments on real world consumer goods datasets show that Q2K surpasses strong baselines, achieving higher accuracy and robustness in difficult scenarios such as bundle identification and brand origin disambiguation. By reusing retrieved reasoning instead of issuing repeated searches, Q2K balances accuracy with efficiency, offering a scalable and interpretable solution for product integration.

**arXiv ID:** 2509.01182
</details>

<details>
<summary><strong>A Historical Interaction-Enhanced Shapley Policy Gradient Algorithm for Multi-Agent Credit Assignment</strong> - Ao Ding, Licheng Sun, Yongjie Hou, Huaqing Zhang, Hongbin Ma - [[pdf]](https://arxiv.org/pdf/2511.07778)</summary>

**Abstract:** Multi-agent reinforcement learning (MARL) has demonstrated remarkable performance in multi-agent collaboration problems and has become a prominent topic in artificial intelligence research in recent years. However, traditional credit assignment schemes in MARL cannot reliably capture individual contributions in strongly coupled tasks while maintaining training stability, which leads to limited generalization capabilities and hinders algorithm performance. To address these challenges, we propose a Historical Interaction-Enhanced Shapley Policy Gradient Algorithm (HIS) for Multi-Agent Credit Assignment, which employs a hybrid credit assignment mechanism to balance base rewards with individual contribution incentives. By utilizing historical interaction data to calculate the Shapley value in a sample-efficient manner, HIS enhances the agent's ability to perceive its own contribution, while retaining the global reward to maintain training stability. Additionally, we provide theoretical guarantees for the hybrid credit assignment mechanism, ensuring that the assignment results it generates are both efficient and stable. We evaluate the proposed algorithm in three widely used continuous-action benchmark environments: Multi-Agent Particle Environment, Multi-Agent MuJoCo, and Bi-DexHands. Experimental results demonstrate that HIS outperforms state-of-the-art methods, particularly excelling in strongly coupled, complex collaborative tasks.

**arXiv ID:** 2511.07778
</details>

<details>
<summary><strong>Can LLM Agents Really Debate? A Controlled Study of Multi-Agent Debate in Logical Reasoning</strong> - Haolun Wu, Zhenkun Li, Lingyao Li - [[pdf]](https://arxiv.org/pdf/2511.07784)</summary>

**Abstract:** Multi-agent debate (MAD) has recently emerged as a promising framework for improving the reasoning performance of large language models (LLMs). Yet, whether LLM agents can genuinely engage in deliberative reasoning, beyond simple ensembling or majority voting, remains unclear. We address this question through a controlled study using the Knight--Knave--Spy logic puzzle, which enables precise, step-wise evaluation of debate outcomes and processes under verifiable ground truth. We systematically set up six structural and cognitive factors, including agent team size, composition, confidence visibility, debate order, debate depth, and task difficulty, to disentangle their respective effects on collective reasoning. Our results show that intrinsic reasoning strength and group diversity are the dominant drivers of debate success, while structural parameters such as order or confidence visibility offer limited gains. Beyond outcomes, process-level analyses identify key behavioral patterns: majority pressure suppresses independent correction, effective teams overturn incorrect consensus, and rational, validity-aligned reasoning most strongly predicts improvement. These findings provide valuable insights into how and why LLM debates succeed or fail, offering guidance for designing interpretable and truth-seeking multi-agent reasoning systems.

**arXiv ID:** 2511.07784
</details>

<details>
<summary><strong>MA-GTS: A Multi-Agent Framework for Solving Complex Graph Problems in Real-World Applications</strong> - Zike Yuan, Ming Liu, Hui Wang, Bing Qin - [[pdf]](https://arxiv.org/pdf/2502.18540)</summary>

**Abstract:** Graph-theoretic problems arise in real-world applications like logistics, communication networks, and traffic optimization. These problems are often complex, noisy, and irregular, posing challenges for traditional algorithms. Large language models (LLMs) offer potential solutions but face challenges, including limited accuracy and input length constraints. To address these challenges, we propose MA-GTS (Multi-Agent Graph Theory Solver), a multi-agent framework that decomposes these complex problems through agent collaboration. MA-GTS maps the implicitly expressed text-based graph data into clear, structured graph representations and dynamically selects the most suitable algorithm based on problem constraints and graph structure scale. This approach ensures that the solution process remains efficient and the resulting reasoning path is interpretable. We validate MA-GTS using the G-REAL dataset, a real-world-inspired graph theory dataset we created. Experimental results show that MA-GTS outperforms state-of-the-art approaches in terms of efficiency, accuracy, and scalability, with strong results across multiple benchmarks (G-REAL 94.2%, GraCoRe 96.9%, NLGraph 98.4%).MA-GTS is open-sourced at this https URL.

**arXiv ID:** 2502.18540
</details>

<details>
<summary><strong>Socialized Learning and Emergent Behaviors in Multi-Agent Systems based on Multimodal Large Language Models</strong> - Sureyya Akin, Shruti T. Tiwari, Ram Bhattacharya, Sagar A. Raman, Kiran Mohanty, Sita Krishnan - [[pdf]](https://arxiv.org/pdf/2510.18515)</summary>

**Abstract:** This search introduces the Multimodal Socialized Learning Framework (M-S2L), designed to foster emergent social intelligence in AI agents by integrating Multimodal Large Language Models (M-LLMs) with social learning mechanisms. The framework equips agents with multimodal perception (vision and text) and structured action capabilities, enabling physical manipulation and grounded multimodal communication (e.g., text with visual pointers). M-S2L combines direct reinforcement learning with two novel social learning pathways: multimodal observational learning and communication-driven learning from feedback, augmented by an episodic memory system for long-term social context.
We evaluate M-S2L in a Collaborative Assembly Environment (CAE), where agent teams must construct complex devices from ambiguous blueprints under informational asymmetry. Across tasks of increasing complexity, M-S2L agents consistently outperform Text-Only and No-Social-Learning baselines in Task Completion Rate and Time to Completion, particularly in dynamic problem-solving scenarios. Ablation studies confirm the necessity of both multimodality and socialized learning. Our analysis reveals the emergence of efficient communication protocols integrating visual pointers with concise text, alongside rapid role specialization leading to stable labor division. Qualitative case studies demonstrate agents' abilities for shared awareness, dynamic re-planning, and adaptive problem-solving, suggesting a nascent form of machine social cognition. These findings indicate that integrating multimodal perception with explicit social learning is critical for developing human-like collaborative intelligence in multi-agent systems.

**arXiv ID:** 2510.18515
</details>

<details>
<summary><strong>TrustResearcher: Automating Knowledge-Grounded and Transparent Research Ideation with Multi-Agent Collaboration</strong> - Jiawei Zhou, Ruicheng Zhu, Mengshi Chen, Jianwei Wang, Kai Wang - [[pdf]](https://arxiv.org/pdf/2510.20844)</summary>

**Abstract:** Effective research relies on organizing extensive information and stimulating novel solutions. Agentic systems have recently emerged as a promising tool to automate literature-based ideation. However, current systems often remain black-box. Their outputs may appear plausible but weakly grounded, with limited transparency or control for researchers. Our work introduces TrustResearcher, a multi-agent demo system for knowledge-grounded and transparent ideation. Specifically, TrustResearcher integrates meticulously designed four stages into a unified framework: (A) Structured Knowledge Curation, (B) Diversified Idea Generation, (C) Multi-stage Idea Selection, and (D) Expert Panel Review & Synthesis. Different from prior pipelines, our system not only exposes intermediate reasoning states, execution logs, and tunable agents for inspections, but also enables the generation of hypotheses that are both diverse and evidence-aligned. Our design is also domain-agnostic: as long as literature sources exist, the same pipeline can be instantiated in any scientific field. As an illustrative case, we demonstrate TrustResearcher on a graph-mining case study (k-truss breaking problem), where it generates distinct, plausible hypotheses with evidence and critiques. A live demo and source code are available at this https URL.

**arXiv ID:** 2510.20844
</details>

<details>
<summary><strong>Human Machine Social Hybrid Intelligence:A Collaborative Decision Making Framework for Large Model Agent Groups and Human Experts</strong> - Ahmet Akkaya Melih, Yamuna Singh, Kunal L. Agarwal, Priya Mukherjee, Kiran Pattnaik, Hanuman Bhatia - [[pdf]](https://arxiv.org/pdf/2510.24030)</summary>

**Abstract:** The rapid advancements in large foundation models and multi-agent systems offer unprecedented capabilities, yet current Human-in-the-Loop (HiTL) paradigms inadequately integrate human expertise, often leading to cognitive overload and decision-making bottlenecks in complex, high-stakes environments. We propose the "Human-Machine Social Hybrid Intelligence" (HMS-HI) framework, a novel architecture designed for deep, collaborative decision-making between groups of human experts and LLM-powered AI agents. HMS-HI is built upon three core pillars: (1) a \textbf{Shared Cognitive Space (SCS)} for unified, multi-modal situational awareness and structured world modeling; (2) a \textbf{Dynamic Role and Task Allocation (DRTA)} module that adaptively assigns tasks to the most suitable agent (human or AI) based on capabilities and workload; and (3) a \textbf{Cross-Species Trust Calibration (CSTC)} protocol that fosters transparency, accountability, and mutual adaptation through explainable declarations and structured feedback. Validated in a high-fidelity urban emergency response simulation, HMS-HI significantly reduced civilian casualties by 72\% and cognitive load by 70\% compared to traditional HiTL approaches, demonstrating superior decision quality, efficiency, and human-AI trust. An ablation study confirms the critical contribution of each module, highlighting that engineered trust and shared context are foundational for scalable, synergistic human-AI collaboration.

**arXiv ID:** 2510.24030
</details>

<details>
<summary><strong>From Pixels to Cooperation Multi Agent Reinforcement Learning based on Multimodal World Models</strong> - Sureyya Akin, Kavita Srivastava, Prateek B. Kapoor, Pradeep G. Sethi, Sunita Q. Patel, Rahu Srivastava - [[pdf]](https://arxiv.org/pdf/2511.01310)</summary>

**Abstract:** Learning cooperative multi-agent policies directly from high-dimensional, multimodal sensory inputs like pixels and audio (from pixels) is notoriously sample-inefficient. Model-free Multi-Agent Reinforcement Learning (MARL) algorithms struggle with the joint challenge of representation learning, partial observability, and credit assignment. To address this, we propose a novel framework based on a shared, generative Multimodal World Model (MWM). Our MWM is trained to learn a compressed latent representation of the environment's dynamics by fusing distributed, multimodal observations from all agents using a scalable attention-based mechanism. Subsequently, we leverage this learned MWM as a fast, "imagined" simulator to train cooperative MARL policies (e.g., MAPPO) entirely within its latent space, decoupling representation learning from policy learning. We introduce a new set of challenging multimodal, multi-agent benchmarks built on a 3D physics simulator. Our experiments demonstrate that our MWM-MARL framework achieves orders-of-magnitude greater sample efficiency compared to state-of-the-art model-free MARL baselines. We further show that our proposed multimodal fusion is essential for task success in environments with sensory asymmetry and that our architecture provides superior robustness to sensor-dropout, a critical feature for real-world deployment.

**arXiv ID:** 2511.01310
</details>

<details>
<summary><strong>VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use</strong> - Zhehao Zhang, Ryan Rossi, Tong Yu, Franck Dernoncourt, Ruiyi Zhang, Jiuxiang Gu, Sungchul Kim, Xiang Chen, Zichao Wang, Nedim Lipka - [[pdf]](https://arxiv.org/pdf/2410.16400)</summary>

**Abstract:** While vision-language models (VLMs) have demonstrated remarkable performance across various tasks combining textual and visual information, they continue to struggle with fine-grained visual perception tasks that require detailed pixel-level analysis. Effectively eliciting comprehensive reasoning from VLMs on such intricate visual elements remains an open challenge. In this paper, we present VipAct, an agent framework that enhances VLMs by integrating multi-agent collaboration and vision expert models, enabling more precise visual understanding and comprehensive reasoning. VipAct consists of an orchestrator agent, which manages task requirement analysis, planning, and coordination, along with specialized agents that handle specific tasks such as image captioning and vision expert models that provide high-precision perceptual information. This multi-agent approach allows VLMs to better perform fine-grained visual perception tasks by synergizing planning, reasoning, and tool use. We evaluate VipAct on benchmarks featuring a diverse set of visual perception tasks, with experimental results demonstrating significant performance improvements over state-of-the-art baselines across all tasks. Furthermore, comprehensive ablation studies reveal the critical role of multi-agent collaboration in eliciting more detailed System-2 reasoning and highlight the importance of image input for task planning. Additionally, our error analysis identifies patterns of VLMs' inherent limitations in visual perception, providing insights into potential future improvements. VipAct offers a flexible and extensible framework, paving the way for more advanced visual perception systems across various real-world applications.

**arXiv ID:** 2410.16400
</details>

<details>
<summary><strong>ProST: Progressive Sub-task Training for Pareto-Optimal Multi-agent Systems Using Small Language Models</strong> - Biddut Sarker Bijoy, Mohammad Saqib Hasan, Pegah Alipoormolabashi, Avirup Sil, Aruna Balasubramanian, Niranjan Balasubramanian - [[pdf]](https://arxiv.org/pdf/2509.04508)</summary>

**Abstract:** Multi-agent systems with smaller language models (SLMs) present a viable alternative to single agent systems powered by large language models (LLMs) for addressing complex problems. In this work, we study how these alternatives compare in terms of both effectiveness and efficiency. To study this trade-off, we instantiate single and multi-agent systems for the complex problems in the AppWorld environment using different sized language models.
We find that difficulties with long-trajectory learning in smaller language models (SLMs) limit their performance. Even when trained for specialized roles, SLMs fail to learn all subtasks effectively. To address this issue, we introduce a simple progressive sub-task training strategy, which introduces new sub-tasks progressively in each training epoch. We find that this novel strategy, analogous to instance level curriculum learning, consistently improves the effectiveness of multi-agents at all configurations. Our Pareto analysis shows that fine-tuned multi-agent systems yield better effectiveness-efficiency trade-offs. Additional ablations and analyses shows the importance of our progressive training strategy and its ability to reduce subtask error rates.

**arXiv ID:** 2509.04508
</details>

<details>
<summary><strong>FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation</strong> - Song Jin, Shuqi Li, Shukun Zhang, Rui Yan - [[pdf]](https://arxiv.org/pdf/2511.07322)</summary>

**Abstract:** While LLMs have shown great success in financial tasks like stock prediction and question answering, their application in fully automating Equity Research Report generation remains uncharted territory. In this paper, we formulate the Equity Research Report (ERR) Generation task for the first time. To address the data scarcity and the evaluation metrics absence, we present an open-source evaluation benchmark for ERR generation - FinRpt. We frame a Dataset Construction Pipeline that integrates 7 financial data types and produces a high-quality ERR dataset automatically, which could be used for model training and evaluation. We also introduce a comprehensive evaluation system including 11 metrics to assess the generated ERRs. Moreover, we propose a multi-agent framework specifically tailored to address this task, named FinRpt-Gen, and train several LLM-based agents on the proposed datasets using Supervised Fine-Tuning and Reinforcement Learning. Experimental results indicate the data quality and metrics effectiveness of the benchmark FinRpt and the strong performance of FinRpt-Gen, showcasing their potential to drive innovation in the ERR generation field. All code and datasets are publicly available.

**arXiv ID:** 2511.07322
</details>

<details>
<summary><strong>Surgical Agent Orchestration Platform for Voice-directed Patient Data Interaction</strong> - Hyeryun Park, Byung Mo Gu, Jun Hee Lee, Byeong Hyeon Choi, Sekeun Kim, Hyun Koo Kim, Kyungsang Kim - [[pdf]](https://arxiv.org/pdf/2511.07392)</summary>

**Abstract:** In da Vinci robotic surgery, surgeons' hands and eyes are fully engaged in the procedure, making it difficult to access and manipulate multimodal patient data without interruption. We propose a voice-directed Surgical Agent Orchestrator Platform (SAOP) built on a hierarchical multi-agent framework, consisting of an orchestration agent and three task-specific agents driven by Large Language Models (LLMs). These LLM-based agents autonomously plan, refine, validate, and reason to map voice commands into specific tasks such as retrieving clinical information, manipulating CT scans, or navigating 3D anatomical models on the surgical video. We also introduce a Multi-level Orchestration Evaluation Metric (MOEM) to comprehensively assess the performance and robustness from command-level and category-level perspectives. The SAOP achieves high accuracy and success rates across 240 voice commands, while LLM-based agents improve robustness against speech recognition errors and diverse or ambiguous free-form commands, demonstrating strong potential to support minimally invasive da Vinci robotic surgery.

**arXiv ID:** 2511.07392
</details>

<details>
<summary><strong>ARAC: Adaptive Regularized Multi-Agent Soft Actor-Critic in Graph-Structured Adversarial Games</strong> - Ruochuan Shi, Runyu Lu, Yuanheng Zhu, Dongbin Zhao - [[pdf]](https://arxiv.org/pdf/2511.08412)</summary>

**Abstract:** In graph-structured multi-agent reinforcement learning (MARL) adversarial tasks such as pursuit and confrontation, agents must coordinate under highly dynamic interactions, where sparse rewards hinder efficient policy learning. We propose Adaptive Regularized Multi-Agent Soft Actor-Critic (ARAC), which integrates an attention-based graph neural network (GNN) for modeling agent dependencies with an adaptive divergence regularization mechanism. The GNN enables expressive representation of spatial relations and state features in graph environments. Divergence regularization can serve as policy guidance to alleviate the sparse reward problem, but it may lead to suboptimal convergence when the reference policy itself is imperfect. The adaptive divergence regularization mechanism enables the framework to exploit reference policies for efficient exploration in the early stages, while gradually reducing reliance on them as training progresses to avoid inheriting their limitations. Experiments in pursuit and confrontation scenarios demonstrate that ARAC achieves faster convergence, higher final success rates, and stronger scalability across varying numbers of agents compared with MARL baselines, highlighting its effectiveness in complex graph-structured environments.

**arXiv ID:** 2511.08412
</details>

<details>
<summary><strong>Optimism as Risk-Seeking in Multi-Agent Reinforcement Learning</strong> - Runyu Zhang, Na Li, Asuman Ozdaglar, Jeff Shamma, Gioele Zardini - [[pdf]](https://arxiv.org/pdf/2509.24047)</summary>

**Abstract:** Risk sensitivity has become a central theme in reinforcement learning (RL), where convex risk measures and robust formulations provide principled ways to model preferences beyond expected return. Recent extensions to multi-agent RL (MARL) have largely emphasized the risk-averse setting, prioritizing robustness to uncertainty. In cooperative MARL, however, such conservatism often leads to suboptimal equilibria, and a parallel line of work has shown that optimism can promote cooperation. Existing optimistic methods, though effective in practice, are typically heuristic and lack theoretical grounding. Building on the dual representation for convex risk measures, we propose a principled framework that interprets risk-seeking objectives as optimism. We introduce optimistic value functions, which formalize optimism as divergence-penalized risk-seeking evaluations. Building on this foundation, we derive a policy-gradient theorem for optimistic value functions, including explicit formulas for the entropic risk/KL-penalty setting, and develop decentralized optimistic actor-critic algorithms that implement these updates. Empirical results on cooperative benchmarks demonstrate that risk-seeking optimism consistently improves coordination over both risk-neutral baselines and heuristic optimistic methods. Our framework thus unifies risk-sensitive learning and optimism, offering a theoretically grounded and practically effective approach to cooperation in MARL.

**arXiv ID:** 2509.24047
</details>

<details>
<summary><strong>A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses</strong> - Xiangxiang Dai, Yuejin Xie, Maoli Liu, Xuchuang Wang, Zhuohua Li, Huanyu Wang, John C.S. Lui - [[pdf]](https://arxiv.org/pdf/2501.01849)</summary>

**Abstract:** Prompt-based offline methods are commonly used to optimize large language model (LLM) responses, but evaluating these responses is computationally intensive and often fails to accommodate diverse response styles. This study introduces a novel online evaluation framework that employs a multi-agent conversational bandit model to select optimal responses while aligning with user preferences dynamically. To tackle challenges such as high-dimensional features, large response sets, adaptive conversational needs, and multi-device access, we propose MACO, Multi-Agent Conversational Online Learning, which comprises two key components: (1) \texttt{MACO-A}: Executed by local agents, it employs an online elimination mechanism to filter out low-quality responses. (2) \texttt{MACO-S}: Executed by the cloud server, it adaptively adjusts selection strategies based on aggregated preference data. An adaptive preference mechanism triggers asynchronous conversations to enhance alignment efficiency. Theoretical analysis demonstrates that MACO achieves near-optimal regret bounds, matching state-of-the-art performance in various degenerate cases. Extensive experiments utilizing Google and OpenAI text embedding models on the real-world datasets with different response styles, combined with Llama and GPT-4o, show that MACO consistently outperforms baseline methods by at least 8.29\% across varying response set sizes and numbers of agents.

**arXiv ID:** 2501.01849
</details>

</details>

<details open>
<summary><h2>Other Agent Research (5 papers)</h2></summary>

<details>
<summary><strong>Smarter Together: Creating Agentic Communities of Practice through Shared Experiential Learning</strong> - Valentin Tablan, Scott Taylor, Gabriel Hurtado, Kristoffer Bernhem, Anders Uhrenholt, Gabriele Farei, Karo Moilanen - [[pdf]](https://arxiv.org/pdf/2511.08301)</summary>

**Abstract:** The transition from human-centric to agent-centric software development practices is disrupting existing knowledge sharing environments for software developers. Traditional peer-to-peer repositories and developer communities for shared technical knowledge and best practice have witnessed dramatic drops in participation in a short period of time. At the same time, agentic functional equivalents are yet to emerge leaving AI agents, which already generate a significant proportion of all new software code produced, without access to repositories of valuable shared learning.
In this paper, we introduce Spark, a novel shared agentic memory architecture which is designed to emulate the collective intelligence and know-how of human developer communities. Spark enables AI coding agents to both contribute to and draw from a persistent and continuously evolving experiential memory. Agents operating in the same general problem space use the Spark shared memory as a repository of new knowledge to achieve collective continual learning. We evaluate Spark as a coach for AI coding agents performing software development tasks. We demonstrate that recommendations made by Spark improve the quality of code generated by generic code generation models at varying sizes and capability tiers. Boosted by Spark, a small open-weights model with 30 billion parameters was able to match the code quality afforded by a much larger state-of-the-art model. Separately, we measure the intrinsic quality of recommendations generated by Spark against a wide range of criteria inspired by software development best practice, and achieve helpfulness levels of up to 98.2% in the top two (out of five) qualitative helpfulness bands.

**arXiv ID:** 2511.08301
</details>

<details>
<summary><strong>Invisible Triggers, Visible Threats! Road-Style Adversarial Creation Attack for Visual 3D Detection in Autonomous Driving</strong> - Jian Wang, Lijun He, Yixing Yong, Haixia Bi, Fan Li - [[pdf]](https://arxiv.org/pdf/2511.08015)</summary>

**Abstract:** Modern autonomous driving (AD) systems leverage 3D object detection to perceive foreground objects in 3D environments for subsequent prediction and planning. Visual 3D detection based on RGB cameras provides a cost-effective solution compared to the LiDAR paradigm. While achieving promising detection accuracy, current deep neural network-based models remain highly susceptible to adversarial examples. The underlying safety concerns motivate us to investigate realistic adversarial attacks in AD scenarios. Previous work has demonstrated the feasibility of placing adversarial posters on the road surface to induce hallucinations in the detector. However, the unnatural appearance of the posters makes them easily noticeable by humans, and their fixed content can be readily targeted and defended. To address these limitations, we propose the AdvRoad to generate diverse road-style adversarial posters. The adversaries have naturalistic appearances resembling the road surface while compromising the detector to perceive non-existent objects at the attack locations. We employ a two-stage approach, termed Road-Style Adversary Generation and Scenario-Associated Adaptation, to maximize the attack effectiveness on the input scene while ensuring the natural appearance of the poster, allowing the attack to be carried out stealthily without drawing human attention. Extensive experiments show that AdvRoad generalizes well to different detectors, scenes, and spoofing locations. Moreover, physical attacks further demonstrate the practical threats in real-world environments.

**arXiv ID:** 2511.08015
</details>

<details>
<summary><strong>A CODECO Case Study and Initial Validation for Edge Orchestration of Autonomous Mobile Robots</strong> - H. Zhu, T. Samizadeh, R. C. Sofia - [[pdf]](https://arxiv.org/pdf/2511.08354)</summary>

**Abstract:** Autonomous Mobile Robots (AMRs) increasingly adopt containerized micro-services across the Edge-Cloud continuum. While Kubernetes is the de-facto orchestrator for such systems, its assumptions of stable networks, homogeneous resources, and ample compute capacity do not fully hold in mobile, resource-constrained robotic environments.
This paper describes a case study on smart-manufacturing AMRs and performs an initial comparison between CODECO orchestration and standard Kubernetes using a controlled KinD environment. Metrics include pod deployment and deletion times, CPU and memory usage, and inter-pod data rates. The observed results indicate that CODECO offers reduced CPU consumption and more stable communication patterns, at the cost of modest memory overhead (10-15%) and slightly increased pod lifecycle latency due to secure overlay initialization.

**arXiv ID:** 2511.08354
</details>

<details>
<summary><strong>QP Chaser: Polynomial Trajectory Generation for Autonomous Aerial Tracking</strong> - Yunwoo Lee, Jungwon Park, Seungwoo Jung, Boseong Jeon, Dahyun Oh, H. Jin Kim - [[pdf]](https://arxiv.org/pdf/2302.14273)</summary>

**Abstract:** Maintaining the visibility of the target is one of the major objectives of aerial tracking missions. This paper proposes a target-visible trajectory planning pipeline using quadratic programming. Our approach can handle various tracking settings, including single and dual target following and both static and dynamic environments, unlike other works that focus on a single specific setup. In contrast to other studies that fully trust the predicted trajectory of the target and consider only the visibility of the center of the target, our pipeline considers error in target path prediction and the entire body of the target to maintain the target visibility robustly. First, a prediction module uses a sample-check strategy to quickly calculate the reachable areas of moving objects, which represent the areas their bodies can reach, considering obstacles. Subsequently, the planning module formulates a single QP problem, considering path homotopy, to generate a tracking trajectory that maximizes the visibility of the target's reachable area among obstacles. The performance of the planner is validated in multiple scenarios, through high-fidelity simulations and real-world experiments.

**arXiv ID:** 2302.14273
</details>

<details>
<summary><strong>Towards Embodied Agentic AI: Review and Classification of LLM- and VLM-Driven Robot Autonomy and Interaction</strong> - Sahar Salimpour, Lei Fu, Farhad Keramat, Leonardo Militano, Giovanni Toffetti, Harry Edelman, Jorge Peña Queralta - [[pdf]](https://arxiv.org/pdf/2508.05294)</summary>

**Abstract:** Foundation models, including large language models (LLMs) and vision-language models (VLMs), have recently enabled novel approaches to robot autonomy and human-robot interfaces. In parallel, vision-language-action models (VLAs) or large behavior models (LBMs) are increasing the dexterity and capabilities of robotic systems. This survey paper reviews works that advance agentic applications and architectures, including initial efforts with GPT-style interfaces and more complex systems where AI agents function as coordinators, planners, perception actors, or generalist interfaces. Such agentic architectures allow robots to reason over natural language instructions, invoke APIs, plan task sequences, or assist in operations and diagnostics. In addition to peer-reviewed research, due to the fast-evolving nature of the field, we highlight and include community-driven projects, ROS packages, and industrial frameworks that show emerging trends. We propose a taxonomy for classifying model integration approaches and present a comparative analysis of the role that agents play in different solutions in today's literature.

**arXiv ID:** 2508.05294
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (22 papers)</h2></summary>

<details>
<summary><strong>Procedural Knowledge Improves Agentic LLM Workflows</strong> - Vincent Hsiao, Mark Roberts, Leslie Smith - [[pdf]](https://arxiv.org/pdf/2511.07568)</summary>

**Abstract:** Large language models (LLMs) often struggle when performing agentic tasks without substantial tool support, prom-pt engineering, or fine tuning. Despite research showing that domain-dependent, procedural knowledge can dramatically increase planning efficiency, little work evaluates its potential for improving LLM performance on agentic tasks that may require implicit planning. We formalize, implement, and evaluate an agentic LLM workflow that leverages procedural knowledge in the form of a hierarchical task network (HTN). Empirical results of our implementation show that hand-coded HTNs can dramatically improve LLM performance on agentic tasks, and using HTNs can boost a 20b or 70b parameter LLM to outperform a much larger 120b parameter LLM baseline. Furthermore, LLM-created HTNs improve overall performance, though less so. The results suggest that leveraging expertise--from humans, documents, or LLMs--to curate procedural knowledge will become another important tool for improving LLM workflows.

**arXiv ID:** 2511.07568
</details>

<details>
<summary><strong>An Efficient Training Pipeline for Reasoning Graphical User Interface Agents</strong> - Georgios Pantazopoulos, Eda B. Özyiğit - [[pdf]](https://arxiv.org/pdf/2511.08172)</summary>

**Abstract:** Visual grounding is the task of localising image regions from natural language queries and is critical for reasoning capable Graphical User Interface agents. Many existing methods rely on massive, noisy synthetic this http URL work introduces an efficient training pipeline that combines model-based data filtering with parameter-efficient fine-tuning. From 4.8M synthetic examples, 12K clean and diverse instances are curated by first identifying challenging cases, removing misaligned and then selecting a diverse set of multimodal instances. On this data, a 3B-parameter Vision-Language Model is trained under three regimes: supervised fine-tuning, chain-of-thought- augmented fine-tuning, and reinforcement learning via Group Relative Policy Optimization. Models trained with the filtered data and lightweight training strategies match or surpass larger baselines on benchmarks such as ScreenSpot, Multimodal-Mind2Web, and AndroidControl. These results demonstrate that principled data curation and robust adaptation can rival large-scale training, enabling compact yet capable multimodal reasoning agents.

**arXiv ID:** 2511.08172
</details>

<details>
<summary><strong>Beyond Distributions: Geometric Action Control for Continuous Reinforcement Learning</strong> - Zhihao Lin - [[pdf]](https://arxiv.org/pdf/2511.08234)</summary>

**Abstract:** Gaussian policies have dominated continuous control in deep reinforcement learning (RL), yet they suffer from a fundamental mismatch: their unbounded support requires ad-hoc squashing functions that distort the geometry of bounded action spaces. While von Mises-Fisher (vMF) distributions offer a theoretically grounded alternative on the sphere, their reliance on Bessel functions and rejection sampling hinders practical adoption. We propose \textbf{Geometric Action Control (GAC)}, a novel action generation paradigm that preserves the geometric benefits of spherical distributions while \textit{simplifying computation}. GAC decomposes action generation into a direction vector and a learnable concentration parameter, enabling efficient interpolation between deterministic actions and uniform spherical noise. This design reduces parameter count from \(2d\) to \(d+1\), and avoids the \(O(dk)\) complexity of vMF rejection sampling, achieving simple \(O(d)\) operations. Empirically, GAC consistently matches or exceeds state-of-the-art methods across six MuJoCo benchmarks, achieving 37.6\% improvement over SAC on Ant-v4 and the best results on 4 out of 6 tasks. Our ablation studies reveal that both \textbf{spherical normalization} and \textbf{adaptive concentration control} are essential to GAC's success. These findings suggest that robust and efficient continuous control does not require complex distributions, but a principled respect for the geometry of action spaces. Code and pretrained models are available in supplementary materials.

**arXiv ID:** 2511.08234
</details>

<details>
<summary><strong>AudAgent: Automated Auditing of Privacy Policy Compliance in AI Agents</strong> - Ye Zheng, Yidan Hu - [[pdf]](https://arxiv.org/pdf/2511.07441)</summary>

**Abstract:** AI agents can autonomously perform tasks and, often without explicit user consent, collect or disclose users' sensitive local data, which raises serious privacy concerns. Although AI agents' privacy policies may describe their intended data practices, there remains limited transparency and accountability about whether runtime behavior matches those policies. To close this gap, we introduce AudAgent, a visual framework that continuously monitors AI agents' data practices in real time and guards compliance with stated privacy policies.
AudAgent consists of four components for automated privacy auditing of AI agents. (i) Policy parsing: an ensemble of LLMs translates natural-language privacy policies into a structured privacy-policy model, where cross-LLM voting guarantees confidence of the parsing results. (ii) Runtime annotation: a lightweight Presidio-based analyzer detects sensitive data and annotates how the data is used based on the context of the AI agent's operations and the privacy-policy model. (iii) Compliance auditing: ontology alignment and automata-based evaluation connect the policy model with runtime annotations, enabling on-the-fly compliance checks between the natural-language policy and observed unordered data practices of AI agents. (iv) User interface: a platform-independent implementation visualizes the real-time execution trace of AI agents along with potential privacy risks detected during auditing, providing user-friendly transparency and accountability.
In addition to common formatted privacy policies, AudAgent also supports user-defined policies for fine-grained control and customization. We evaluate AudAgent on AI agents built upon mainstream programming frameworks such as AutoGen, experiments show that AudAgent effectively identifies potential privacy policy violations in real time.

**arXiv ID:** 2511.07441
</details>

<details>
<summary><strong>Auto-US: An Ultrasound Video Diagnosis Agent Using Video Classification Framework and LLMs</strong> - Yuezhe Yang, Yiyue Guo, Wenjie Cai, Qingqing Ruan, Siying Wang, Xingbo Dong, Zhe Jin, Yong Dai - [[pdf]](https://arxiv.org/pdf/2511.07748)</summary>

**Abstract:** AI-assisted ultrasound video diagnosis presents new opportunities to enhance the efficiency and accuracy of medical imaging analysis. However, existing research remains limited in terms of dataset diversity, diagnostic performance, and clinical applicability. In this study, we propose \textbf{Auto-US}, an intelligent diagnosis agent that integrates ultrasound video data with clinical diagnostic text. To support this, we constructed \textbf{CUV Dataset} of 495 ultrasound videos spanning five categories and three organs, aggregated from multiple open-access sources. We developed \textbf{CTU-Net}, which achieves state-of-the-art performance in ultrasound video classification, reaching an accuracy of 86.73\% Furthermore, by incorporating large language models, Auto-US is capable of generating clinically meaningful diagnostic suggestions. The final diagnostic scores for each case exceeded 3 out of 5 and were validated by professional clinicians. These results demonstrate the effectiveness and clinical potential of Auto-US in real-world ultrasound applications. Code and data are available at: this https URL.

**arXiv ID:** 2511.07748
</details>

<details>
<summary><strong>Diffusion Guided Adversarial State Perturbations in Reinforcement Learning</strong> - Xiaolin Sun, Feidi Liu, Zhengming Ding, ZiZhan Zheng - [[pdf]](https://arxiv.org/pdf/2511.07701)</summary>

**Abstract:** Reinforcement learning (RL) systems, while achieving remarkable success across various domains, are vulnerable to adversarial attacks. This is especially a concern in vision-based environments where minor manipulations of high-dimensional image inputs can easily mislead the agent's behavior. To this end, various defenses have been proposed recently, with state-of-the-art approaches achieving robust performance even under large state perturbations. However, after closer investigation, we found that the effectiveness of the current defenses is due to a fundamental weakness of the existing $l_p$ norm-constrained attacks, which can barely alter the semantics of image input even under a relatively large perturbation budget. In this work, we propose SHIFT, a novel policy-agnostic diffusion-based state perturbation attack to go beyond this limitation. Our attack is able to generate perturbed states that are semantically different from the true states while remaining realistic and history-aligned to avoid detection. Evaluations show that our attack effectively breaks existing defenses, including the most sophisticated ones, significantly outperforming existing attacks while being more perceptually stealthy. The results highlight the vulnerability of RL agents to semantics-aware adversarial perturbations, indicating the importance of developing more robust policies.

**arXiv ID:** 2511.07701
</details>

<details>
<summary><strong>Dynamic Sparsity: Challenging Common Sparsity Assumptions for Learning World Models in Robotic Reinforcement Learning Benchmarks</strong> - Muthukumar Pandaram, Jakob Hollenstein, David Drexel, Samuele Tosatto, Antonio Rodríguez-Sánchez, Justus Piater - [[pdf]](https://arxiv.org/pdf/2511.08086)</summary>

**Abstract:** The use of learned dynamics models, also known as world models, can improve the sample efficiency of reinforcement learning. Recent work suggests that the underlying causal graphs of such dynamics models are sparsely connected, with each of the future state variables depending only on a small subset of the current state variables, and that learning may therefore benefit from sparsity priors. Similarly, temporal sparsity, i.e. sparsely and abruptly changing local dynamics, has also been proposed as a useful inductive bias.
In this work, we critically examine these assumptions by analyzing ground-truth dynamics from a set of robotic reinforcement learning environments in the MuJoCo Playground benchmark suite, aiming to determine whether the proposed notions of state and temporal sparsity actually tend to hold in typical reinforcement learning tasks.
We study (i) whether the causal graphs of environment dynamics are sparse, (ii) whether such sparsity is state-dependent, and (iii) whether local system dynamics change sparsely.
Our results indicate that global sparsity is rare, but instead the tasks show local, state-dependent sparsity in their dynamics and this sparsity exhibits distinct structures, appearing in temporally localized clusters (e.g., during contact events) and affecting specific subsets of state dimensions. These findings challenge common sparsity prior assumptions in dynamics learning, emphasizing the need for grounded inductive biases that reflect the state-dependent sparsity structure of real-world dynamics.

**arXiv ID:** 2511.08086
</details>

<details>
<summary><strong>Test-driven Reinforcement Learning</strong> - Zhao Yu, Xiuping Wu, Liangjun Ke - [[pdf]](https://arxiv.org/pdf/2511.07904)</summary>

**Abstract:** Reinforcement learning (RL) has been recognized as a powerful tool for robot control tasks. RL typically employs reward functions to define task objectives and guide agent learning. However, since the reward function serves the dual purpose of defining the optimal goal and guiding learning, it is challenging to design the reward function manually, which often results in a suboptimal task representation. To tackle the reward design challenge in RL, inspired by the satisficing theory, we propose a Test-driven Reinforcement Learning (TdRL) framework. In the TdRL framework, multiple test functions are used to represent the task objective rather than a single reward function. Test functions can be categorized as pass-fail tests and indicative tests, each dedicated to defining the optimal objective and guiding the learning process, respectively, thereby making defining tasks easier. Building upon such a task definition, we first prove that if a trajectory return function assigns higher returns to trajectories closer to the optimal trajectory set, maximum entropy policy optimization based on this return function will yield a policy that is closer to the optimal policy set. Then, we introduce a lexicographic heuristic approach to compare the relative distance relationship between trajectories and the optimal trajectory set for learning the trajectory return function. Furthermore, we develop an algorithm implementation of TdRL. Experimental results on the DeepMind Control Suite benchmark demonstrate that TdRL matches or outperforms handcrafted reward methods in policy training, with greater design simplicity and inherent support for multi-objective optimization. We argue that TdRL offers a novel perspective for representing task objectives, which could be helpful in addressing the reward design challenges in RL applications.

**arXiv ID:** 2511.07904
</details>

<details>
<summary><strong>LPPG-RL: Lexicographically Projected Policy Gradient Reinforcement Learning with Subproblem Exploration</strong> - Ruiyu Qiu, Rui Wang, Guanghui Yang, Xiang Li, Zhijiang Shao - [[pdf]](https://arxiv.org/pdf/2511.08339)</summary>

**Abstract:** Lexicographic multi-objective problems, which consist of multiple conflicting subtasks with explicit priorities, are common in real-world applications. Despite the advantages of Reinforcement Learning (RL) in single tasks, extending conventional RL methods to prioritized multiple objectives remains challenging. In particular, traditional Safe RL and Multi-Objective RL (MORL) methods have difficulty enforcing priority orderings efficiently. Therefore, Lexicographic Multi-Objective RL (LMORL) methods have been developed to address these challenges. However, existing LMORL methods either rely on heuristic threshold tuning with prior knowledge or are restricted to discrete domains. To overcome these limitations, we propose Lexicographically Projected Policy Gradient RL (LPPG-RL), a novel LMORL framework which leverages sequential gradient projections to identify feasible policy update directions, thereby enabling LPPG-RL broadly compatible with all policy gradient algorithms in continuous spaces. LPPG-RL reformulates the projection step as an optimization problem, and utilizes Dykstra's projection rather than generic solvers to deliver great speedups, especially for small- to medium-scale instances. In addition, LPPG-RL introduces Subproblem Exploration (SE) to prevent gradient vanishing, accelerate convergence and enhance stability. We provide theoretical guarantees for convergence and establish a lower bound on policy improvement. Finally, through extensive experiments in a 2D navigation environment, we demonstrate the effectiveness of LPPG-RL, showing that it outperforms existing state-of-the-art continuous LMORL methods.

**arXiv ID:** 2511.08339
</details>

<details>
<summary><strong>ReCode: Updating Code API Knowledge with Reinforcement Learning</strong> - Haoze Wu, Yunzhi Yao, Wenhao Yu, Ningyu Zhang - [[pdf]](https://arxiv.org/pdf/2506.20495)</summary>

**Abstract:** Large Language Models (LLMs) exhibit remarkable code generation capabilities but falter when adapting to frequent updates in external library APIs. This critical limitation, stemming from reliance on outdated API knowledge from their training data, even with access to current documentation, impedes reliable code generation in dynamic environments. To tackle this issue, we propose ReCode (rule-based Reinforcement learning for Code Update), a novel framework that mimics human programmer adaptation to API changes. Specifically, we construct a dataset of approximately 2,000 data entries to train the LLMs to perform version migration based on updated information. Then, we introduce a modified string similarity metric for code evaluation as the reward for reinforcement learning. Our experiments demonstrate that ReCode substantially boosts LLMs' code generation performance in dynamic API scenarios, especially on the unseen CodeUpdateArena task. Crucially, compared to supervised fine-tuning, ReCode has less impact on LLMs' general code generation abilities. We apply ReCode on various LLMs and reinforcement learning algorithms (GRPO and DAPO), all achieving consistent improvements. Notably, after training, Qwen2.5-Coder-7B outperforms that of the 32B parameter code instruction-tuned model and the reasoning model with the same architecture. Code is available at this https URL.

**arXiv ID:** 2506.20495
</details>

<details>
<summary><strong>Intelligent Optimization of Multi-Parameter Micromixers Using a Scientific Machine Learning Framework</strong> - Meraj Hassanzadeh, Ehsan Ghaderi, Mohamad Ali Bijarchi, Siamak Kazemzadeh Hannani - [[pdf]](https://arxiv.org/pdf/2511.07702)</summary>

**Abstract:** Multidimensional optimization has consistently been a critical challenge in engineering. However, traditional simulation-based optimization methods have long been plagued by significant limitations: they are typically capable of optimizing only a single problem at a time and require substantial computational time for meshing and numerical simulation. This paper introduces a novel framework leveraging cutting-edge Scientific Machine Learning (Sci-ML) methodologies to overcome these inherent drawbacks of conventional approaches. The proposed method provides instantaneous solutions to a spectrum of complex, multidimensional optimization problems. A micromixer case study is employed to demonstrate this methodology. An agent, operating on a Deep Reinforcement Learning (DRL) architecture, serves as the optimizer to explore the relationships between key problem parameters. This optimizer interacts with an environment constituted by a parametric Physics-Informed Neural Network (PINN), which responds to the agent's actions at a significantly higher speed than traditional numerical methods. The agent's objective, conditioned on the Schmidt number is to discover the optimal geometric and physical parameters that maximize the micromixer's efficiency. After training the agent across a wide range of Schmidt numbers, we analyzed the resulting optimal designs. Across this entire spectrum, the achieved efficiency was consistently greater than the baseline, normalized value. The maximum efficiency occurred at a Schmidt number of 13.3, demonstrating an improvement of approximately 32%. Finally, a comparative analysis with a Genetic Algorithm was conducted under equivalent conditions to underscore the advantages of the proposed method.

**arXiv ID:** 2511.07702
</details>

<details>
<summary><strong>SERL: Self-Examining Reinforcement Learning on Open-Domain</strong> - Weixuan Ou, Yanzhao Zheng, Shuoshuo Sun, Wei Zhang, Baohua Dong, Hangcheng Zhu, Ruohui Huang, Gang Yu, Pengwei Yan, Yifan Qiao - [[pdf]](https://arxiv.org/pdf/2511.07922)</summary>

**Abstract:** Reinforcement Learning (RL) has been shown to improve the capabilities of large language models (LLMs). However, applying RL to open-domain tasks faces two key challenges: (1) the inherent subjectivity of these tasks prevents the verifiable rewards as required by Reinforcement Learning with Verifiable Rewards (RLVR); (2) Reinforcement Learning from Human Feedback (RLHF) relies on external reward mechanisms. To overcome these limitations, we propose Self-Examining Reinforcement Learning (SERL), a novel self-improving framework where the LLM serves as both Actor and Judge. SERL introduces two synergistic reward mechanisms without any external signals. On the one hand, to improve the Actor's capability, we derive rewards from Copeland-style pairwise comparison judgments across a group of generated responses. On the other hand, a self-consistency reward that encourages coherent judgments is proposed to improve the Judge's reliability. This process refines the Judge's capability, which in turn provides a more robust reward for Actor. Experiments show that our method outperforms existing self-improvement training methods. SERL improves the LC win rate of Qwen3-8B on AlpacaEval 2 from 52.37% to 59.90%. To the best of our knowledge, our method achieves state-of-the-art performance among self-improving approaches. Furthermore, it achieves a performance comparable to significantly larger models like Qwen3-32B, demonstrating superior effectiveness and robustness on open-domain tasks.

**arXiv ID:** 2511.07922
</details>

<details>
<summary><strong>Multistep Quasimetric Learning for Scalable Goal-conditioned Reinforcement Learning</strong> - Bill Chunyuan Zheng, Vivek Myers, Benjamin Eysenbach, Sergey Levine - [[pdf]](https://arxiv.org/pdf/2511.07730)</summary>

**Abstract:** Learning how to reach goals in an environment is a longstanding challenge in AI, yet reasoning over long horizons remains a challenge for modern methods. The key question is how to estimate the temporal distance between pairs of observations. While temporal difference methods leverage local updates to provide optimality guarantees, they often perform worse than Monte Carlo methods that perform global updates (e.g., with multi-step returns), which lack such guarantees. We show how these approaches can be integrated into a practical GCRL method that fits a quasimetric distance using a multistep Monte-Carlo return. We show our method outperforms existing GCRL methods on long-horizon simulated tasks with up to 4000 steps, even with visual observations. We also demonstrate that our method can enable stitching in the real-world robotic manipulation domain (Bridge setup). Our approach is the first end-to-end GCRL method that enables multistep stitching in this real-world manipulation domain from an unlabeled offline dataset of visual observations.

**arXiv ID:** 2511.07730
</details>

<details>
<summary><strong>Shocks Under Control: Taming Transonic Compressible Flow over an RAE2822 Airfoil with Deep Reinforcement Learning</strong> - Trishit Mondal, Ricardo Vinuesa, Ameya D. Jagtap - [[pdf]](https://arxiv.org/pdf/2511.07564)</summary>

**Abstract:** Active flow control of compressible transonic shock-boundary layer interactions over a two-dimensional RAE2822 airfoil at Re = 50,000 is investigated using deep reinforcement learning (DRL). The flow field exhibits highly unsteady dynamics, including complex shock-boundary layer interactions, shock oscillations, and the generation of Kutta waves from the trailing edge. A high-fidelity CFD solver, employing a fifth-order spectral discontinuous Galerkin scheme in space and a strong-stability-preserving Runge-Kutta (5,4) method in time, together with adaptive mesh refinement capability, is used to obtain the accurate flow field. Synthetic jet actuation is employed to manipulate these unsteady flow features, while the DRL agent autonomously discovers effective control strategies through direct interaction with high-fidelity compressible flow simulations. The trained controllers effectively mitigate shock-induced separation, suppress unsteady oscillations, and manipulate aerodynamic forces under transonic conditions. In the first set of experiments, aimed at both drag reduction and lift enhancement, the DRL-based control reduces the average drag coefficient by 13.78% and increases lift by 131.18%, thereby improving the lift-to-drag ratio by 121.52%, which underscores its potential for managing complex flow dynamics. In the second set, targeting drag reduction while maintaining lift, the DRL-based control achieves a 25.62% reduction in drag and a substantial 196.30% increase in lift, accompanied by markedly diminished oscillations. In this case, the lift-to-drag ratio improves by 220.26%.

**arXiv ID:** 2511.07564
</details>

<details>
<summary><strong>RL-Exec: Impact-Aware Reinforcement Learning for Opportunistic Optimal Liquidation, Outperforms TWAP and a Book-Liquidity VWAP on BTC-USD Replays</strong> - Enzo Duflot, Stanislas Robineau - [[pdf]](https://arxiv.org/pdf/2511.07434)</summary>

**Abstract:** We study opportunistic optimal liquidation over fixed deadlines on BTC-USD limit-order books (LOB). We present RL-Exec, a PPO agent trained on historical replays augmented with endogenous transient impact (resilience), partial fills, maker/taker fees, and latency. The policy observes depth-20 LOB features plus microstructure indicators and acts under a sell-only inventory constraint to reach a residual target. Evaluation follows a strict time split (train: Jan-2020; test: Feb-2020) and a per-day protocol: for each test day we run ten independent start times and aggregate to a single daily score, avoiding pseudo-replication. We compare the agent to (i) TWAP and (ii) a VWAP-like baseline allocating using opposite-side order-book liquidity (top-20 levels), both executed on identical timestamps and costs. Statistical inference uses one-sided Wilcoxon signed-rank tests on daily RL-baseline differences with Benjamini-Hochberg FDR correction and bootstrap confidence intervals. On the Feb-2020 test set, RL-Exec significantly outperforms both baselines and the gap increases with the execution horizon (+2-3 bps at 30 min, +7-8 bps at 60 min, +23 bps at 120 min).
Code: this http URL

**arXiv ID:** 2511.07434
</details>

<details>
<summary><strong>Toward Autonomous and Efficient Cybersecurity: A Multi-Objective AutoML-based Intrusion Detection System</strong> - Li Yang, Abdallah Shami - [[pdf]](https://arxiv.org/pdf/2511.08491)</summary>

**Abstract:** With increasingly sophisticated cybersecurity threats and rising demand for network automation, autonomous cybersecurity mechanisms are becoming critical for securing modern networks. The rapid expansion of Internet of Things (IoT) systems amplifies these challenges, as resource-constrained IoT devices demand scalable and efficient security solutions. In this work, an innovative Intrusion Detection System (IDS) utilizing Automated Machine Learning (AutoML) and Multi-Objective Optimization (MOO) is proposed for autonomous and optimized cyber-attack detection in modern networking environments. The proposed IDS framework integrates two primary innovative techniques: Optimized Importance and Percentage-based Automated Feature Selection (OIP-AutoFS) and Optimized Performance, Confidence, and Efficiency-based Combined Algorithm Selection and Hyperparameter Optimization (OPCE-CASH). These components optimize feature selection and model learning processes to strike a balance between intrusion detection effectiveness and computational efficiency. This work presents the first IDS framework that integrates all four AutoML stages and employs multi-objective optimization to jointly optimize detection effectiveness, efficiency, and confidence for deployment in resource-constrained systems. Experimental evaluations over two benchmark cybersecurity datasets demonstrate that the proposed MOO-AutoML IDS outperforms state-of-the-art IDSs, establishing a new benchmark for autonomous, efficient, and optimized security for networks. Designed to support IoT and edge environments with resource constraints, the proposed framework is applicable to a variety of autonomous cybersecurity applications across diverse networked environments.

**arXiv ID:** 2511.08491
</details>

<details>
<summary><strong>Bayes Adaptive Monte Carlo Tree Search for Offline Model-based Reinforcement Learning</strong> - Jiayu Chen, Le Xu, Wentse Chen, Jeff Schneider - [[pdf]](https://arxiv.org/pdf/2410.11234)</summary>

**Abstract:** Offline RL is a powerful approach for data-driven decision-making and control. Compared to model-free methods, offline model-based RL (MBRL) explicitly learns world models from a static dataset and uses them as surrogate simulators, improving the data efficiency and enabling the learned policy to potentially generalize beyond the dataset support. However, there could be various MDPs that behave identically on the offline dataset and dealing with the uncertainty about the true MDP can be challenging. In this paper, we propose modeling offline MBRL as a Bayes Adaptive Markov Decision Process (BAMDP), which is a principled framework for addressing model uncertainty. We further propose a novel Bayes Adaptive Monte-Carlo planning algorithm capable of solving BAMDPs in continuous state and action spaces with stochastic transitions. This planning process is based on Monte Carlo Tree Search and can be integrated into offline MBRL as a policy improvement operator in policy iteration. Our ``RL + Search" framework follows in the footsteps of superhuman AIs like AlphaZero, improving on current offline MBRL methods by incorporating more computation input. The proposed algorithm significantly outperforms state-of-the-art offline RL methods on twelve D4RL MuJoCo tasks and three target tracking tasks in a challenging, stochastic tokamak control simulator. The codebase is available at: this https URL.

**arXiv ID:** 2410.11234
</details>

<details>
<summary><strong>Policy-Driven World Model Adaptation for Robust Offline Model-based Reinforcement Learning</strong> - Jiayu Chen, Le Xu, Aravind Venugopal, Jeff Schneider - [[pdf]](https://arxiv.org/pdf/2505.13709)</summary>

**Abstract:** Offline reinforcement learning (RL) offers a powerful paradigm for data-driven control. Compared to model-free approaches, offline model-based RL (MBRL) explicitly learns a world model from a static dataset and uses it as a surrogate simulator, improving data efficiency and enabling potential generalization beyond the dataset support. However, most existing offline MBRL methods follow a two-stage training procedure: first learning a world model by maximizing the likelihood of the observed transitions, then optimizing a policy to maximize its expected return under the learned model. This objective mismatch results in a world model that is not necessarily optimized for effective policy learning. Moreover, we observe that policies learned via offline MBRL often lack robustness during deployment, and small adversarial noise in the environment can lead to significant performance degradation. To address these, we propose a framework that dynamically adapts the world model alongside the policy under a unified learning objective aimed at improving robustness. At the core of our method is a maximin optimization problem, which we solve by innovatively utilizing Stackelberg learning dynamics. We provide theoretical analysis to support our design and introduce computationally efficient implementations. We benchmark our algorithm on twelve noisy D4RL MuJoCo tasks and three stochastic Tokamak Control tasks, demonstrating its state-of-the-art performance.

**arXiv ID:** 2505.13709
</details>

<details>
<summary><strong>ORVIT: Near-Optimal Online Distributionally Robust Reinforcement Learning</strong> - Debamita Ghosh, George K. Atia, Yue Wang - [[pdf]](https://arxiv.org/pdf/2508.03768)</summary>

**Abstract:** We investigate reinforcement learning (RL) in the presence of distributional mismatch between training and deployment, where policies trained in simulators often underperform in practice due to mismatches between training and deployment conditions, and thereby reliable guarantees on real-world performance are essential. Distributionally robust RL addresses this issue by optimizing worst-case performance over an uncertainty set of environments and providing an optimized lower bound on deployment performance. However, existing studies typically assume access to either a generative model or offline datasets with broad coverage of the deployment environment-assumptions that limit their practicality in unknown environments without prior knowledge. In this work, we study a more practical and challenging setting: online distributionally robust RL, where the agent interacts only with a single unknown training environment while seeking policies that are robust with respect to an uncertainty set around this nominal model. We consider general $f$-divergence-based ambiguity sets, including $\chi^2$ and KL divergence balls, and design a computationally efficient algorithm that achieves sublinear regret for the robust control objective under minimal assumptions, without requiring generative or offline data access. Moreover, we establish a corresponding minimax lower bound on the regret of any online algorithm, demonstrating the near-optimality of our method. Experiments across diverse environments with model misspecification show that our approach consistently improves worst-case performance and aligns with the theoretical guarantees.

**arXiv ID:** 2508.03768
</details>

<details>
<summary><strong>Policy Transfer for Continuous-Time Reinforcement Learning: A (Rough) Differential Equation Approach</strong> - Xin Guo, Zijiu Lyu - [[pdf]](https://arxiv.org/pdf/2510.15165)</summary>

**Abstract:** This paper studies policy transfer, one of the well-known transfer learning techniques adopted in large language models, for two classes of continuous-time reinforcement learning problems. In the first class of continuous-time linear-quadratic systems with Shannon's entropy regularization (a.k.a. LQRs), we fully exploit the Gaussian structure of their optimal policy and the stability of their associated Riccati equations. In the second class where the system has possibly non-linear and bounded dynamics, the key technical component is the stability of diffusion SDEs which is established by invoking the rough path theory. Our work provides the first theoretical proof of policy transfer for continuous-time RL: an optimal policy learned for one RL problem can be used to initialize the search for a near-optimal policy in a closely related RL problem, while maintaining the convergence rate of the original algorithm.
To illustrate the benefit of policy transfer for RL, we propose a novel policy learning algorithm for continuous-time LQRs, which achieves global linear convergence and local super-linear convergence. As a byproduct of our analysis, we derive the stability of a concrete class of continuous-time score-based diffusion models via their connection with LQRs.

**arXiv ID:** 2510.15165
</details>

<details>
<summary><strong>Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options</strong> - Joongkyu Lee, Seouh-won Yi, Min-hwan Oh - [[pdf]](https://arxiv.org/pdf/2510.18713)</summary>

**Abstract:** We study online preference-based reinforcement learning (PbRL) with the goal of improving sample efficiency. While a growing body of theoretical work has emerged-motivated by PbRL's recent empirical success, particularly in aligning large language models (LLMs)-most existing studies focus only on pairwise comparisons. A few recent works (Zhu et al., 2023, Mukherjee et al., 2024, Thekumparampil et al., 2024) have explored using multiple comparisons and ranking feedback, but their performance guarantees fail to improve-and can even deteriorate-as the feedback length increases, despite the richer information available. To address this gap, we adopt the Plackett-Luce (PL) model for ranking feedback over action subsets and propose M-AUPO, an algorithm that selects multiple actions by maximizing the average uncertainty within the offered subset. We prove that M-AUPO achieves a suboptimality gap of $\tilde{O}\left( \frac{d}{T} \sqrt{ \sum_{t=1}^T \frac{1}{|S_t|}} \right)$, where $T$ is the total number of rounds, $d$ is the feature dimension, and $|S_t|$ is the size of the subset at round $t$. This result shows that larger subsets directly lead to improved performance and, notably, the bound avoids the exponential dependence on the unknown parameter's norm, which was a fundamental limitation in most previous works. Moreover, we establish a near-matching lower bound of $\Omega \left( \frac{d}{K \sqrt{T}} \right)$, where $K$ is the maximum subset size. To the best of our knowledge, this is the first theoretical result in PbRL with ranking feedback that explicitly shows improved sample efficiency as a function of the subset size.

**arXiv ID:** 2510.18713
</details>

<details>
<summary><strong>On the Convergence and Stability of Upside-Down Reinforcement Learning, Goal-Conditioned Supervised Learning, and Online Decision Transformers</strong> - Miroslav Štrupl, Oleg Szehr, Francesco Faccio, Dylan R. Ashley, Rupesh Kumar Srivastava, Jürgen Schmidhuber - [[pdf]](https://arxiv.org/pdf/2502.05672)</summary>

**Abstract:** This article provides a rigorous analysis of convergence and stability of Episodic Upside-Down Reinforcement Learning, Goal-Conditioned Supervised Learning and Online Decision Transformers. These algorithms performed competitively across various benchmarks, from games to robotic tasks, but their theoretical understanding is limited to specific environmental conditions. This work initiates a theoretical foundation for algorithms that build on the broad paradigm of approaching reinforcement learning through supervised learning or sequence modeling. At the core of this investigation lies the analysis of conditions on the underlying environment, under which the algorithms can identify optimal solutions. We also assess whether emerging solutions remain stable in situations where the environment is subject to tiny levels of noise. Specifically, we study the continuity and asymptotic convergence of command-conditioned policies, values and the goal-reaching objective depending on the transition kernel of the underlying Markov Decision Process. We demonstrate that near-optimal behavior is achieved if the transition kernel is located in a sufficiently small neighborhood of a deterministic kernel. The mentioned quantities are continuous (with respect to a specific topology) at deterministic kernels, both asymptotically and after a finite number of learning cycles. The developed methods allow us to present the first explicit estimates on the convergence and stability of policies and values in terms of the underlying transition kernels. On the theoretical side we introduce a number of new concepts to reinforcement learning, like working in segment spaces, studying continuity in quotient topologies and the application of the fixed-point theory of dynamical systems. The theoretical study is accompanied by a detailed investigation of example environments and numerical experiments.

**arXiv ID:** 2502.05672
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
