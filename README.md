# Agent arXiv Daily

**Last Updated:** 2025-09-12 02:34:01

**Total Papers:** 46

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
<summary><strong>Curriculum-Based Multi-Tier Semantic Exploration via Deep Reinforcement Learning</strong> - Abdel Hakim Drid, Vincenzo Suriani, Daniele Nardi, Abderrezzak Debilou - [[pdf]](https://arxiv.org/pdf/2509.09356)</summary>

**Abstract:** Navigating and understanding complex and unknown environments autonomously demands more than just basic perception and movement from embodied agents. Truly effective exploration requires agents to possess higher-level cognitive abilities, the ability to reason about their surroundings, and make more informed decisions regarding exploration strategies. However, traditional RL approaches struggle to balance efficient exploration and semantic understanding due to limited cognitive capabilities embedded in the small policies for the agents, leading often to human drivers when dealing with semantic exploration. In this paper, we address this challenge by presenting a novel Deep Reinforcement Learning (DRL) architecture that is specifically designed for resource efficient semantic exploration. A key methodological contribution is the integration of a Vision-Language Model (VLM) common-sense through a layered reward function. The VLM query is modeled as a dedicated action, allowing the agent to strategically query the VLM only when deemed necessary for gaining external guidance, thereby conserving resources. This mechanism is combined with a curriculum learning strategy designed to guide learning at different levels of complexity to ensure robust and stable learning. Our experimental evaluation results convincingly demonstrate that our agent achieves significantly enhanced object discovery rates and develops a learned capability to effectively navigate towards semantically rich regions. Furthermore, it also shows a strategic mastery of when to prompt for external environmental information. By demonstrating a practical and scalable method for embedding common-sense semantic reasoning with autonomous agents, this research provides a novel approach to pursuing a fully intelligent and self-guided exploration in robotics.

**arXiv ID:** 2509.09356
</details>

<details>
<summary><strong>FLM-Audio: Natural Monologues Improves Native Full-Duplex Chatbots via Dual Training</strong> - Yiqun Yao, Xiang Li, Xin Jiang, Xuezhi Fang, Naitong Yu, Wenjia Ma, Aixin Sun, Yequan Wang - [[pdf]](https://arxiv.org/pdf/2509.02521)</summary>

**Abstract:** Full-duplex dialog models aim to listen and speak simultaneously, delivering rapid responses to dynamic user input. Among different solutions to full duplexity, a native solution merges multiple channels in each time step, achieving the lowest latency. However, prevailing designs break down the textual monologue sentences for word-level alignment with audio streams, which degrades language modeling abilities. To help address this issue, we introduce natural monologues, which are composed by continuous sentences and waiting intervals, mimicking humanoid cognitive behavior in dialogs. We find a proper training paradigm to be critical for semantically aligning natural monologues with audio. To this end, we develop a dual training paradigm that alternates the position of the monologues, either leading or trailing the audio, across different training stages. A combination of our natural monologue and dual training strategy is applied in developing FLM-Audio, our 7B spoken dialog chatbot with native full-duplexity. As confirmed by experimental results, FLM-Audio achieves superior response qualities and chatting experiences while requiring significantly less training data.

**arXiv ID:** 2509.02521
</details>

<details>
<summary><strong>Flip Co-op: Cooperative Takeovers in Shared Autonomy</strong> - Sandeep Banik, Naira Hovakimyan - [[pdf]](https://arxiv.org/pdf/2509.09281)</summary>

**Abstract:** Shared autonomy requires principled mechanisms for allocating and transferring control between a human and an autonomous agent. Existing approaches often rely on blending control inputs between human and autonomous agent or switching rules, which lack theoretical guarantees. This paper develops a game-theoretic framework for modeling cooperative takeover in shared autonomy. We formulate the switching interaction as a dynamic game in which authority is embedded directly into the system dynamics, resulting in Nash equilibrium(NE)-based strategies rather than ad hoc switching rules. We establish the existence and characterization of NE in the space of pure takeover strategies under stochastic human intent. For the class of linear-quadratic systems, we derive closed-form recursions for takeover strategies and saddle-point value functions, providing analytical insight and efficient computation of cooperative takeover policies. We further introduce a bimatrix potential game reformulation to address scenarios where human and autonomy utilities are not perfectly aligned, yielding a unifying potential function that preserves tractability while capturing intent deviations. The framework is applied to a vehicle trajectory tracking problem, demonstrating how equilibrium takeover strategies adapt across straight and curved path segments. The results highlight the trade-off between human adaptability and autonomous efficiency and illustrate the practical benefits of grounding shared autonomy in cooperative game theory.

**arXiv ID:** 2509.09281
</details>

<details>
<summary><strong>A Systematic Mapping Study on Chatbots in Programming Education</strong> - Marcelino Garcia, Renato Garcia, Arthur Parizotto, Andre Mendes, Pedro Valle, Ricardo Vilela, Renato Balancieri, Williamson Silva - [[pdf]](https://arxiv.org/pdf/2509.08857)</summary>

**Abstract:** Educational chatbots have gained prominence as support tools for teaching programming, particularly in introductory learning contexts. This paper presents a Systematic Mapping Study (SMS) that investigated how such agents have been developed and applied in programming education. From an initial set of 3,216 publications, 54 studies were selected and analyzed based on five research subquestions, addressing chatbot types, programming languages used, educational content covered, interaction models, and application contexts. The results reveal a predominance of chatbots designed for Python instruction, focusing on fundamental programming concepts, and employing a wide variety of pedagogical approaches and technological architectures. In addition to identifying trends and gaps in the literature, this study provides insights to inform the development of new educational tools for programming instruction.

**arXiv ID:** 2509.08857
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (7 papers)</h2></summary>

<details>
<summary><strong>Mind Meets Space: Rethinking Agentic Spatial Intelligence from a Neuroscience-inspired Perspective</strong> - Bui Duc Manh, Soumyaratna Debnath, Zetong Zhang, Shriram Damodaran, Arvind Kumar, Yueyi Zhang, Lu Mi, Erik Cambria, Lin Wang - [[pdf]](https://arxiv.org/pdf/2509.09154)</summary>

**Abstract:** Recent advances in agentic AI have led to systems capable of autonomous task execution and language-based reasoning, yet their spatial reasoning abilities remain limited and underexplored, largely constrained to symbolic and sequential processing. In contrast, human spatial intelligence, rooted in integrated multisensory perception, spatial memory, and cognitive maps, enables flexible, context-aware decision-making in unstructured environments. Therefore, bridging this gap is critical for advancing Agentic Spatial Intelligence toward better interaction with the physical 3D world. To this end, we first start from scrutinizing the spatial neural models as studied in computational neuroscience, and accordingly introduce a novel computational framework grounded in neuroscience principles. This framework maps core biological functions to six essential computation modules: bio-inspired multimodal sensing, multi-sensory integration, egocentric-allocentric conversion, an artificial cognitive map, spatial memory, and spatial reasoning. Together, these modules form a perspective landscape for agentic spatial reasoning capability across both virtual and physical environments. On top, we conduct a framework-guided analysis of recent methods, evaluating their relevance to each module and identifying critical gaps that hinder the development of more neuroscience-grounded spatial reasoning modules. We further examine emerging benchmarks and datasets and explore potential application domains ranging from virtual to embodied systems, such as robotics. Finally, we outline potential research directions, emphasizing the promising roadmap that can generalize spatial reasoning across dynamic or unstructured environments. We hope this work will benefit the research community with a neuroscience-grounded perspective and a structured pathway. Our project page can be found at Github.

**arXiv ID:** 2509.09154
</details>

<details>
<summary><strong>Towards Adaptive ML Benchmarks: Web-Agent-Driven Construction, Domain Expansion, and Metric Optimization</strong> - Hangyi Jia, Yuxi Qian, Hanwen Tong, Xinhui Wu, Lin Chen, Feng Wei - [[pdf]](https://arxiv.org/pdf/2509.09321)</summary>

**Abstract:** Recent advances in large language models (LLMs) have enabled the emergence of general-purpose agents for automating end-to-end machine learning (ML) workflows, including data analysis, feature engineering, model training, and competition solving. However, existing benchmarks remain limited in task coverage, domain diversity, difficulty modeling, and evaluation rigor, failing to capture the full capabilities of such agents in realistic settings. We present TAM Bench, a diverse, realistic, and structured benchmark for evaluating LLM-based agents on end-to-end ML tasks. TAM Bench features three key innovations: (1) A browser automation and LLM-based task acquisition system that automatically collects and structures ML challenges from platforms such as Kaggle, AIcrowd, and Biendata, spanning multiple task types and data modalities (e.g., tabular, text, image, graph, audio); (2) A leaderboard-driven difficulty modeling mechanism that estimates task complexity using participant counts and score dispersion, enabling scalable and objective task calibration; (3) A multi-dimensional evaluation framework incorporating performance, format compliance, constraint adherence, and task generalization. Based on 150 curated AutoML tasks, we construct three benchmark subsets of different sizes -- Lite, Medium, and Full -- designed for varying evaluation scenarios. The Lite version, with 18 tasks and balanced coverage across modalities and difficulty levels, serves as a practical testbed for daily benchmarking and comparative studies.

**arXiv ID:** 2509.09321
</details>

<details>
<summary><strong>Classification of Driver Behaviour Using External Observation Techniques for Autonomous Vehicles</strong> - Ian Nell, Shane Gilroy - [[pdf]](https://arxiv.org/pdf/2509.09349)</summary>

**Abstract:** Road traffic accidents remain a significant global concern, with human error, particularly distracted and impaired driving, among the leading causes. This study introduces a novel driver behavior classification system that uses external observation techniques to detect indicators of distraction and impairment. The proposed framework employs advanced computer vision methodologies, including real-time object tracking, lateral displacement analysis, and lane position monitoring. The system identifies unsafe driving behaviors such as excessive lateral movement and erratic trajectory patterns by implementing the YOLO object detection model and custom lane estimation algorithms. Unlike systems reliant on inter-vehicular communication, this vision-based approach enables behavioral analysis of non-connected vehicles. Experimental evaluations on diverse video datasets demonstrate the framework's reliability and adaptability across varying road and environmental conditions.

**arXiv ID:** 2509.09349
</details>

<details>
<summary><strong>EgoAgent: A Joint Predictive Agent Model in Egocentric Worlds</strong> - Lu Chen, Yizhou Wang, Shixiang Tang, Qianhong Ma, Tong He, Wanli Ouyang, Xiaowei Zhou, Hujun Bao, Sida Peng - [[pdf]](https://arxiv.org/pdf/2502.05857)</summary>

**Abstract:** Learning an agent model that behaves like humans-capable of jointly perceiving the environment, predicting the future, and taking actions from a first-person perspective-is a fundamental challenge in computer vision. Existing methods typically train separate models for these abilities, which fail to capture their intrinsic relationships and prevent them from learning from each other. Inspired by how humans learn through the perception-action loop, we propose EgoAgent, a unified agent model that simultaneously learns to represent, predict, and act within a single transformer. EgoAgent explicitly models the causal and temporal dependencies among these abilities by formulating the task as an interleaved sequence of states and actions. It further introduces a joint embedding-action-prediction architecture with temporally asymmetric predictor and observer branches, enabling synergistic optimization across all three capabilities. Comprehensive evaluations of EgoAgent on representative tasks such as image classification, egocentric future state prediction, and 3D human motion prediction demonstrate the superiority of our method. The code and trained models will be publicly available at this https URL.

**arXiv ID:** 2502.05857
</details>

<details>
<summary><strong>Towards Generalized Routing: Model and Agent Orchestration for Adaptive and Efficient Inference</strong> - Xiyu Guo, Shan Wang, Chunfang Ji, Xuefeng Zhao, Wenhao Xi, Yaoyao Liu, Qinglan Li, Chao Deng, Junlan Feng - [[pdf]](https://arxiv.org/pdf/2509.07571)</summary>

**Abstract:** The rapid advancement of large language models (LLMs) and domain-specific AI agents has greatly expanded the ecosystem of AI-powered services. User queries, however, are highly diverse and often span multiple domains and task types, resulting in a complex and heterogeneous landscape. This diversity presents a fundamental routing challenge: how to accurately direct each query to an appropriate execution unit while optimizing both performance and efficiency. To address this, we propose MoMA (Mixture of Models and Agents), a generalized routing framework that integrates both LLM and agent-based routing. Built upon a deep understanding of model and agent capabilities, MoMA effectively handles diverse queries through precise intent recognition and adaptive routing strategies, achieving an optimal balance between efficiency and cost. Specifically, we construct a detailed training dataset to profile the capabilities of various LLMs under different routing model structures, identifying the most suitable tasks for each LLM. During inference, queries are dynamically routed to the LLM with the best cost-performance efficiency. We also introduce an efficient agent selection strategy based on a context-aware state machine and dynamic masking. Experimental results demonstrate that the MoMA router offers superior cost-efficiency and scalability compared to existing approaches.

**arXiv ID:** 2509.07571
</details>

<details>
<summary><strong>Agentic LLMs for Question Answering over Tabular Data</strong> - Rishit Tyagi, Mohit Gupta, Rahul Bouri - [[pdf]](https://arxiv.org/pdf/2509.09234)</summary>

**Abstract:** Question Answering over Tabular Data (Table QA) presents unique challenges due to the diverse structure, size, and data types of real-world tables. The SemEval 2025 Task 8 (DataBench) introduced a benchmark composed of large-scale, domain-diverse datasets to evaluate the ability of models to accurately answer structured queries. We propose a Natural Language to SQL (NL-to-SQL) approach leveraging large language models (LLMs) such as GPT-4o, GPT-4o-mini, and DeepSeek v2:16b to generate SQL queries dynamically. Our system follows a multi-stage pipeline involving example selection, SQL query generation, answer extraction, verification, and iterative refinement. Experiments demonstrate the effectiveness of our approach, achieving 70.5\% accuracy on DataBench QA and 71.6\% on DataBench Lite QA, significantly surpassing baseline scores of 26\% and 27\% respectively. This paper details our methodology, experimental results, and alternative approaches, providing insights into the strengths and limitations of LLM-driven Table QA.

**arXiv ID:** 2509.09234
</details>

<details>
<summary><strong>GEMINUS: Dual-aware Global and Scene-Adaptive Mixture-of-Experts for End-to-End Autonomous Driving</strong> - Chi Wan, Yixin Cui, Jiatong Du, Shuo Yang, Yulong Bai, Peng Yi, Nan Li, Yanjun Huang - [[pdf]](https://arxiv.org/pdf/2507.14456)</summary>

**Abstract:** End-to-end autonomous driving requires adaptive and robust handling of complex and diverse traffic environments. However, prevalent single-mode planning methods attempt to learn an overall policy while struggling to acquire diversified driving skills to handle diverse scenarios. Therefore, this paper proposes GEMINUS, a Mixture-of-Experts end-to-end autonomous driving framework featuring a Global Expert and a Scene-Adaptive Experts Group, equipped with a Dual-aware Router. Specifically, the Global Expert is trained on the overall dataset, possessing robust performance. The Scene-Adaptive Experts are trained on corresponding scene subsets, achieving adaptive performance. The Dual-aware Router simultaneously considers scenario-level features and routing uncertainty to dynamically activate expert modules. Through the effective coupling of the Global Expert and the Scene-Adaptive Experts Group via the Dual-aware Router, GEMINUS achieves both adaptability and robustness across diverse scenarios. GEMINUS outperforms existing methods in the Bench2Drive closed-loop benchmark and achieves state-of-the-art performance in Driving Score and Success Rate, even with only monocular vision input. The code is available at this https URL.

**arXiv ID:** 2507.14456
</details>

</details>

<details open>
<summary><h2>LLM Agents (2 papers)</h2></summary>

<details>
<summary><strong>Global Constraint LLM Agents for Text-to-Model Translation</strong> - Junyang Cai, Serdar Kadioglu, Bistra Dilkina - [[pdf]](https://arxiv.org/pdf/2509.08970)</summary>

**Abstract:** Natural language descriptions of optimization or satisfaction problems are challenging to translate into correct MiniZinc models, as this process demands both logical reasoning and constraint programming expertise. We introduce a framework that addresses this challenge with an agentic approach: multiple specialized large language model (LLM) agents decompose the modeling task by global constraint type. Each agent is dedicated to detecting and generating code for a specific class of global constraint, while a final assembler agent integrates these constraint snippets into a complete MiniZinc model. By dividing the problem into smaller, well-defined sub-tasks, each LLM handles a simpler reasoning challenge, potentially reducing overall complexity. We conduct initial experiments with several LLMs and show better performance against baselines such as one-shot prompting and chain-of-thought prompting. Finally, we outline a comprehensive roadmap for future work, highlighting potential enhancements and directions for improvement.

**arXiv ID:** 2509.08970
</details>

<details>
<summary><strong>Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents</strong> - Jiawei Wang, Jiacai Liu, Yuqian Fu, Yingru Li, Xintao Wang, Yuan Lin, Yu Yue, Lin Zhang, Yang Wang, Ke Wang - [[pdf]](https://arxiv.org/pdf/2509.09265)</summary>

**Abstract:** In long-horizon tasks, recent agents based on Large Language Models (LLMs) face a significant challenge that sparse, outcome-based rewards make it difficult to assign credit to intermediate steps. Previous methods mainly focus on creating dense reward signals to guide learning, either through traditional reinforcement learning techniques like inverse reinforcement learning or by using Process Reward Models for step-by-step feedback. In this paper, we identify a fundamental problem in the learning dynamics of LLMs: the magnitude of policy gradients is inherently coupled with the entropy, which leads to inefficient small updates for confident correct actions and potentially destabilizes large updates for uncertain ones. To resolve this, we propose Entropy-Modulated Policy Gradients (EMPG), a framework that re-calibrates the learning signal based on step-wise uncertainty and the final task outcome. EMPG amplifies updates for confident correct actions, penalizes confident errors, and attenuates updates from uncertain steps to stabilize exploration. We further introduce a bonus term for future clarity that encourages agents to find more predictable solution paths. Through comprehensive experiments on three challenging agent tasks, WebShop, ALFWorld, and Deep Search, we demonstrate that EMPG achieves substantial performance gains and significantly outperforms strong policy gradient baselines. Project page is at this https URL

**arXiv ID:** 2509.09265
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (11 papers)</h2></summary>

<details>
<summary><strong>Understanding Economic Tradeoffs Between Human and AI Agents in Bargaining Games</strong> - Crystal Qian, Kehang Zhu, John Horton, Benjamin S. Manning, Vivian Tsai, James Wexler, Nithum Thain - [[pdf]](https://arxiv.org/pdf/2509.09071)</summary>

**Abstract:** Coordination tasks traditionally performed by humans are increasingly being delegated to autonomous agents. As this pattern progresses, it becomes critical to evaluate not only these agents' performance but also the processes through which they negotiate in dynamic, multi-agent environments. Furthermore, different agents exhibit distinct advantages: traditional statistical agents, such as Bayesian models, may excel under well-specified conditions, whereas large language models (LLMs) can generalize across contexts. In this work, we compare humans (N = 216), LLMs (GPT-4o, Gemini 1.5 Pro), and Bayesian agents in a dynamic negotiation setting that enables direct, identical-condition comparisons across populations, capturing both outcomes and behavioral dynamics. Bayesian agents extract the highest surplus through aggressive optimization, at the cost of frequent trade rejections. Humans and LLMs can achieve similar overall surplus, but through distinct behaviors: LLMs favor conservative, concessionary trades with few rejections, while humans employ more strategic, risk-taking, and fairness-oriented behaviors. Thus, we find that performance parity -- a common benchmark in agent evaluation -- can conceal fundamental differences in process and alignment, which are critical for practical deployment in real-world coordination tasks.

**arXiv ID:** 2509.09071
</details>

<details>
<summary><strong>ProgD: Progressive Multi-scale Decoding with Dynamic Graphs for Joint Multi-agent Motion Forecasting</strong> - Xing Gao, Zherui Huang, Weiyao Lin, Xiao Sun - [[pdf]](https://arxiv.org/pdf/2509.09210)</summary>

**Abstract:** Accurate motion prediction of surrounding agents is crucial for the safe planning of autonomous vehicles. Recent advancements have extended prediction techniques from individual agents to joint predictions of multiple interacting agents, with various strategies to address complex interactions within future motions of agents. However, these methods overlook the evolving nature of these interactions. To address this limitation, we propose a novel progressive multi-scale decoding strategy, termed ProgD, with the help of dynamic heterogeneous graph-based scenario modeling. In particular, to explicitly and comprehensively capture the evolving social interactions in future scenarios, given their inherent uncertainty, we design a progressive modeling of scenarios with dynamic heterogeneous graphs. With the unfolding of such dynamic heterogeneous graphs, a factorized architecture is designed to process the spatio-temporal dependencies within future scenarios and progressively eliminate uncertainty in future motions of multiple agents. Furthermore, a multi-scale decoding procedure is incorporated to improve on the future scenario modeling and consistent prediction of agents' future motion. The proposed ProgD achieves state-of-the-art performance on the INTERACTION multi-agent prediction benchmark, ranking $1^{st}$, and the Argoverse 2 multi-world forecasting benchmark.

**arXiv ID:** 2509.09210
</details>

<details>
<summary><strong>Enabling Regulatory Multi-Agent Collaboration: Architecture, Challenges, and Solutions</strong> - Qinnan Hu, Yuntao Wang, Yuan Gao, Zhou Su, Linkang Du - [[pdf]](https://arxiv.org/pdf/2509.09215)</summary>

**Abstract:** Large language models (LLMs)-empowered autonomous agents are transforming both digital and physical environments by enabling adaptive, multi-agent collaboration. While these agents offer significant opportunities across domains such as finance, healthcare, and smart manufacturing, their unpredictable behaviors and heterogeneous capabilities pose substantial governance and accountability challenges. In this paper, we propose a blockchain-enabled layered architecture for regulatory agent collaboration, comprising an agent layer, a blockchain data layer, and a regulatory application layer. Within this framework, we design three key modules: (i) an agent behavior tracing and arbitration module for automated accountability, (ii) a dynamic reputation evaluation module for trust assessment in collaborative scenarios, and (iii) a malicious behavior forecasting module for early detection of adversarial activities. Our approach establishes a systematic foundation for trustworthy, resilient, and scalable regulatory mechanisms in large-scale agent ecosystems. Finally, we discuss the future research directions for blockchain-enabled regulatory frameworks in multi-agent systems.

**arXiv ID:** 2509.09215
</details>

<details>
<summary><strong>LightAgent: Production-level Open-source Agentic AI Framework</strong> - Weige Cai, Tong Zhu, Jinyi Niu, Ruiqi Hu, Lingyao Li, Tenglong Wang, Xiaowu Dai, Weining Shen, Liwen Zhang - [[pdf]](https://arxiv.org/pdf/2509.09292)</summary>

**Abstract:** With the rapid advancement of large language models (LLMs), Multi-agent Systems (MAS) have achieved significant progress in various application scenarios. However, substantial challenges remain in designing versatile, robust, and efficient platforms for agent deployment. To address these limitations, we propose \textbf{LightAgent}, a lightweight yet powerful agentic framework, effectively resolving the trade-off between flexibility and simplicity found in existing frameworks. LightAgent integrates core functionalities such as Memory (mem0), Tools, and Tree of Thought (ToT), while maintaining an extremely lightweight structure. As a fully open-source solution, it seamlessly integrates with mainstream chat platforms, enabling developers to easily build self-learning agents. We have released LightAgent at \href{this https URL}{this https URL}

**arXiv ID:** 2509.09292
</details>

<details>
<summary><strong>SEDM: Scalable Self-Evolving Distributed Memory for Agents</strong> - Haoran Xu, Jiacong Hu, Ke Zhang, Lei Yu, Yuxin Tang, Xinyuan Song, Yiqun Duan, Lynn Ai, Bill Shi - [[pdf]](https://arxiv.org/pdf/2509.09498)</summary>

**Abstract:** Long-term multi-agent systems inevitably generate vast amounts of trajectories and historical interactions, which makes efficient memory management essential for both performance and scalability. Existing methods typically depend on vector retrieval and hierarchical storage, yet they are prone to noise accumulation, uncontrolled memory expansion, and limited generalization across domains. To address these challenges, we present SEDM, Self-Evolving Distributed Memory, a verifiable and adaptive framework that transforms memory from a passive repository into an active, self-optimizing component. SEDM integrates verifiable write admission based on reproducible replay, a self-scheduling memory controller that dynamically ranks and consolidates entries according to empirical utility, and cross-domain knowledge diffusion that abstracts reusable insights to support transfer across heterogeneous tasks. Evaluations on benchmark datasets demonstrate that SEDM improves reasoning accuracy while reducing token overhead compared with strong memory baselines, and further enables knowledge distilled from fact verification to enhance multi-hop reasoning. The results highlight SEDM as a scalable and sustainable memory mechanism for open-ended multi-agent collaboration. The code will be released in the later stage of this project.

**arXiv ID:** 2509.09498
</details>

<details>
<summary><strong>MIND: Towards Immersive Psychological Healing with Multi-agent Inner Dialogue</strong> - Yujia Chen, Changsong Li, Yiming Wang, Tianjie Ju, Qingqing Xiao, Nan Zhang, Zifan Kong, Peng Wang, Binyu Yan - [[pdf]](https://arxiv.org/pdf/2502.19860)</summary>

**Abstract:** Mental health issues are worsening in today's competitive society, such as depression and anxiety. Traditional healings like counseling and chatbots fail to engage effectively, they often provide generic responses lacking emotional depth. Although large language models (LLMs) have the potential to create more human-like interactions, they still struggle to capture subtle emotions. This requires LLMs to be equipped with human-like adaptability and warmth. To fill this gap, we propose the MIND (Multi-agent INner Dialogue), a novel paradigm that provides more immersive psychological healing environments. Considering the strong generative and role-playing ability of LLM agents, we predefine an interactive healing framework and assign LLM agents different roles within the framework to engage in interactive inner dialogues with users, thereby providing an immersive healing experience. We conduct extensive human experiments in various real-world healing dimensions, and find that MIND provides a more user-friendly experience than traditional paradigms. This demonstrates that MIND effectively leverages the significant potential of LLMs in psychological healing.

**arXiv ID:** 2502.19860
</details>

<details>
<summary><strong>Demo: Healthcare Agent Orchestrator (HAO) for Patient Summarization in Molecular Tumor Boards</strong> - Matthias Blondeel, Noel Codella, Sam Preston, Hao Qiu, Leonardo Schettini, Frank Tuan, Wen-wai Yim, Smitha Saligrama, Mert Öz, Shrey Jain, Matthew P. Lungren, Thomas Osborne - [[pdf]](https://arxiv.org/pdf/2509.06602)</summary>

**Abstract:** Molecular Tumor Boards (MTBs) are multidisciplinary forums where oncology specialists collaboratively assess complex patient cases to determine optimal treatment strategies. A central element of this process is the patient summary, typically compiled by a medical oncologist, radiation oncologist, or surgeon, or their trained medical assistant, who distills heterogeneous medical records into a concise narrative to facilitate discussion. This manual approach is often labor-intensive, subjective, and prone to omissions of critical information. To address these limitations, we introduce the Healthcare Agent Orchestrator (HAO), a Large Language Model (LLM)-driven AI agent that coordinates a multi-agent clinical workflow to generate accurate and comprehensive patient summaries for MTBs. Evaluating predicted patient summaries against ground truth presents additional challenges due to stylistic variation, ordering, synonym usage, and phrasing differences, which complicate the measurement of both succinctness and completeness. To overcome these evaluation hurdles, we propose TBFact, a ``model-as-a-judge'' framework designed to assess the comprehensiveness and succinctness of generated summaries. Using a benchmark dataset derived from de-identified tumor board discussions, we applied TBFact to evaluate our Patient History agent. Results show that the agent captured 94% of high-importance information (including partial entailments) and achieved a TBFact recall of 0.84 under strict entailment criteria. We further demonstrate that TBFact enables a data-free evaluation framework that institutions can deploy locally without sharing sensitive clinical data. Together, HAO and TBFact establish a robust foundation for delivering reliable and scalable support to MTBs.

**arXiv ID:** 2509.06602
</details>

<details>
<summary><strong>Symmetry-Guided Multi-Agent Inverse Reinforcement Learning</strong> - Yongkai Tian, Yirong Qi, Xin Yu, Wenjun Wu, Jie Luo - [[pdf]](https://arxiv.org/pdf/2509.08257)</summary>

**Abstract:** In robotic systems, the performance of reinforcement learning depends on the rationality of predefined reward functions. However, manually designed reward functions often lead to policy failures due to inaccuracies. Inverse Reinforcement Learning (IRL) addresses this problem by inferring implicit reward functions from expert demonstrations. Nevertheless, existing methods rely heavily on large amounts of expert demonstrations to accurately recover the reward function. The high cost of collecting expert demonstrations in robotic applications, particularly in multi-robot systems, severely hinders the practical deployment of IRL. Consequently, improving sample efficiency has emerged as a critical challenge in multi-agent inverse reinforcement learning (MIRL). Inspired by the symmetry inherent in multi-agent systems, this work theoretically demonstrates that leveraging symmetry enables the recovery of more accurate reward functions. Building upon this insight, we propose a universal framework that integrates symmetry into existing multi-agent adversarial IRL algorithms, thereby significantly enhancing sample efficiency. Experimental results from multiple challenging tasks have demonstrated the effectiveness of this framework. Further validation in physical multi-robot systems has shown the practicality of our method.

**arXiv ID:** 2509.08257
</details>

<details>
<summary><strong>Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning</strong> - Xuefeng Wang, Lei Zhang, Henglin Pu, Ahmed H. Qureshi, Husheng Li - [[pdf]](https://arxiv.org/pdf/2509.09135)</summary>

**Abstract:** Existing reinforcement learning (RL) methods struggle with complex dynamical systems that demand interactions at high frequencies or irregular time intervals. Continuous-time RL (CTRL) has emerged as a promising alternative by replacing discrete-time Bellman recursion with differential value functions defined as viscosity solutions of the Hamilton--Jacobi--Bellman (HJB) equation. While CTRL has shown promise, its applications have been largely limited to the single-agent domain. This limitation stems from two key challenges: (i) conventional solution methods for HJB equations suffer from the curse of dimensionality (CoD), making them intractable in high-dimensional systems; and (ii) even with HJB-based learning approaches, accurately approximating centralized value functions in multi-agent settings remains difficult, which in turn destabilizes policy training. In this paper, we propose a CT-MARL framework that uses physics-informed neural networks (PINNs) to approximate HJB-based value functions at scale. To ensure the value is consistent with its differential structure, we align value learning with value-gradient learning by introducing a Value Gradient Iteration (VGI) module that iteratively refines value gradients along trajectories. This improves gradient fidelity, in turn yielding more accurate values and stronger policy learning. We evaluate our method using continuous-time variants of standard benchmarks, including multi-agent particle environment (MPE) and multi-agent MuJoCo. Our results demonstrate that our approach consistently outperforms existing continuous-time RL baselines and scales to complex multi-agent dynamics.

**arXiv ID:** 2509.09135
</details>

<details>
<summary><strong>Bridging the Capability Gap: Joint Alignment Tuning for Harmonizing LLM-based Multi-Agent Systems</strong> - Minghang Zhu, Zhengliang Shi, Zhiwei Xu, Shiguang Wu, Lingjie Wang, Pengjie Ren, Zhaochun Ren, Zhumin Chen - [[pdf]](https://arxiv.org/pdf/2509.09629)</summary>

**Abstract:** The advancement of large language models (LLMs) has enabled the construction of multi-agent systems to solve complex tasks by dividing responsibilities among specialized agents, such as a planning agent for subgoal generation and a grounding agent for executing tool-use actions. Most existing methods typically fine-tune these agents independently, leading to capability gaps among them with poor coordination. To address this, we propose MOAT, a Multi-Agent Joint Alignment Tuning framework that improves agents collaboration through iterative alignment. MOAT alternates between two key stages: (1) Planning Agent Alignment, which optimizes the planning agent to generate subgoal sequences that better guide the grounding agent; and (2) Grounding Agent Improving, which fine-tunes the grounding agent using diverse subgoal-action pairs generated by the agent itself to enhance its generalization capablity. Theoretical analysis proves that MOAT ensures a non-decreasing and progressively convergent training process. Experiments across six benchmarks demonstrate that MOAT outperforms state-of-the-art baselines, achieving average improvements of 3.1% on held-in tasks and 4.4% on held-out tasks.

**arXiv ID:** 2509.09629
</details>

<details>
<summary><strong>Harmonia: A Multi-Agent Reinforcement Learning Approach to Data Placement and Migration in Hybrid Storage Systems</strong> - Rakesh Nadig, Vamanan Arulchelvan, Rahul Bera, Taha Shahroodi, Gagandeep Singh, Andreas Kakolyris, Mohammad Sadrosadati, Jisung Park, Onur Mutlu - [[pdf]](https://arxiv.org/pdf/2503.20507)</summary>

**Abstract:** Hybrid storage systems (HSS) integrate multiple storage devices with diverse characteristics to deliver high performance and capacity at low cost. The performance of an HSS highly depends on the effectiveness of two key policies: (1) the data-placement policy, which determines the best-fit storage device for incoming data, and (2) the data-migration policy, which dynamically rearranges stored data (i.e., prefetches hot data and evicts cold data) across the devices to sustain high HSS performance. Prior works optimize either data placement or data migration in isolation, which leads to suboptimal HSS performance. Unfortunately, no prior work tries to optimize both policies together.
Our goal is to design a holistic data-management technique that optimizes both data-placement and data-migration policies to fully exploit the potential of an HSS, and thus significantly improve system performance. We propose Harmonia, a multi-agent reinforcement learning (RL)-based data-management technique that employs two lightweight autonomous RL agents, a data-placement agent and a data-migration agent, that adapt their policies for the current workload and HSS configuration while coordinating with each other to improve overall HSS performance.
We evaluate Harmonia on real HSS configurations with up to four heterogeneous storage devices and seventeen data-intensive workloads. On performance-optimized (cost-optimized) HSS with two storage devices, Harmonia outperforms the best-performing prior approach by 49.5% (31.7%) on average. On an HSS with three (four) devices, Harmonia outperforms the best-performing prior work by 37.0% (42.0%) on average. Harmonia's performance benefits come with low latency (240ns for inference) and storage overheads (206 KiB in DRAM for both RL agents combined). We will open-source Harmonia's implementation to aid future research on HSS.

**arXiv ID:** 2503.20507
</details>

</details>

<details open>
<summary><h2>Other Agent Research (5 papers)</h2></summary>

<details>
<summary><strong>Boosting Embodied AI Agents through Perception-Generation Disaggregation and Asynchronous Pipeline Execution</strong> - Shulai Zhang, Ao Xu, Quan Chen, Han Zhao, Weihao Cui, Ningxin Zheng, Haibin Lin, Xin Liu, Minyi Guo - [[pdf]](https://arxiv.org/pdf/2509.09560)</summary>

**Abstract:** Embodied AI systems operate in dynamic environments, requiring seamless integration of perception and generation modules to process high-frequency input and output demands. Traditional sequential computation patterns, while effective in ensuring accuracy, face significant limitations in achieving the necessary "thinking" frequency for real-world applications. In this work, we present Auras, an algorithm-system co-designed inference framework to optimize the inference frequency of embodied AI agents. Auras disaggregates the perception and generation and provides controlled pipeline parallelism for them to achieve high and stable throughput. Faced with the data staleness problem that appears when the parallelism is increased, Auras establishes a public context for perception and generation to share, thereby promising the accuracy of embodied agents. Experimental results show that Auras improves throughput by 2.54x on average while achieving 102.7% of the original accuracy, demonstrating its efficacy in overcoming the constraints of sequential computation and providing high throughput.

**arXiv ID:** 2509.09560
</details>

<details>
<summary><strong>VeriSafe Agent: Safeguarding Mobile GUI Agent via Logic-based Action Verification</strong> - Jungjae Lee, Dongjae Lee, Chihun Choi, Youngmin Im, Jaeyoung Wi, Kihong Heo, Sangeun Oh, Sunjae Lee, Insik Shin - [[pdf]](https://arxiv.org/pdf/2503.18492)</summary>

**Abstract:** Large Foundation Models (LFMs) have unlocked new possibilities in human-computer interaction, particularly with the rise of mobile Graphical User Interface (GUI) Agents capable of interacting with mobile GUIs. These agents allow users to automate complex mobile tasks through simple natural language instructions. However, the inherent probabilistic nature of LFMs, coupled with the ambiguity and context-dependence of mobile tasks, makes LFM-based automation unreliable and prone to errors. To address this critical challenge, we introduce VeriSafe Agent (VSA): a formal verification system that serves as a logically grounded safeguard for Mobile GUI Agents. VSA deterministically ensures that an agent's actions strictly align with user intent before executing the action. At its core, VSA introduces a novel autoformalization technique that translates natural language user instructions into a formally verifiable specification. This enables runtime, rule-based verification of agent's actions, detecting erroneous actions even before they take effect. To the best of our knowledge, VSA is the first attempt to bring the rigor of formal verification to GUI agents, bridging the gap between LFM-driven actions and formal software verification. We implement VSA using off-the-shelf LFM services (GPT-4o) and evaluate its performance on 300 user instructions across 18 widely used mobile apps. The results demonstrate that VSA achieves 94.33%-98.33% accuracy in verifying agent actions, outperforming existing LFM-based verification methods by 30.00%-16.33%, and increases the GUI agent's task completion rate by 90%-130%.

**arXiv ID:** 2503.18492
</details>

<details>
<summary><strong>AEGIS: An Agent for Extraction and Geographic Identification in Scholarly Proceedings</strong> - Om Vishesh, Harshad Khadilkar, Deepak Akkil - [[pdf]](https://arxiv.org/pdf/2509.09470)</summary>

**Abstract:** Keeping pace with the rapid growth of academia literature presents a significant challenge for researchers, funding bodies, and academic societies. To address the time-consuming manual effort required for scholarly discovery, we present a novel, fully automated system that transitions from data discovery to direct action. Our pipeline demonstrates how a specialized AI agent, 'Agent-E', can be tasked with identifying papers from specific geographic regions within conference proceedings and then executing a Robotic Process Automation (RPA) to complete a predefined action, such as submitting a nomination form. We validated our system on 586 papers from five different conferences, where it successfully identified every target paper with a recall of 100% and a near perfect accuracy of 99.4%. This demonstration highlights the potential of task-oriented AI agents to not only filter information but also to actively participate in and accelerate the workflows of the academic community.

**arXiv ID:** 2509.09470
</details>

<details>
<summary><strong>Occupancy-aware Trajectory Planning for Autonomous Valet Parking in Uncertain Dynamic Environments</strong> - Farhad Nawaz, Faizan M. Tariq, Sangjae Bae, David Isele, Avinash Singh, Nadia Figueroa, Nikolai Matni, Jovin D'sa - [[pdf]](https://arxiv.org/pdf/2509.09206)</summary>

**Abstract:** Accurately reasoning about future parking spot availability and integrated planning is critical for enabling safe and efficient autonomous valet parking in dynamic, uncertain environments. Unlike existing methods that rely solely on instantaneous observations or static assumptions, we present an approach that predicts future parking spot occupancy by explicitly distinguishing between initially vacant and occupied spots, and by leveraging the predicted motion of dynamic agents. We introduce a probabilistic spot occupancy estimator that incorporates partial and noisy observations within a limited Field-of-View (FoV) model and accounts for the evolving uncertainty of unobserved regions. Coupled with this, we design a strategy planner that adaptively balances goal-directed parking maneuvers with exploratory navigation based on information gain, and intelligently incorporates wait-and-go behaviors at promising spots. Through randomized simulations emulating large parking lots, we demonstrate that our framework significantly improves parking efficiency, safety margins, and trajectory smoothness compared to existing approaches.

**arXiv ID:** 2509.09206
</details>

<details>
<summary><strong>Sensible Agent: A Framework for Unobtrusive Interaction with Proactive AR Agents</strong> - Geonsun Lee, Min Xia, Nels Numan, Xun Qian, David Li, Yanhe Chen, Achin Kulshrestha, Ishan Chatterjee, Yinda Zhang, Dinesh Manocha, David Kim, Ruofei Du - [[pdf]](https://arxiv.org/pdf/2509.09255)</summary>

**Abstract:** Proactive AR agents promise context-aware assistance, but their interactions often rely on explicit voice prompts or responses, which can be disruptive or socially awkward. We introduce Sensible Agent, a framework designed for unobtrusive interaction with these proactive agents. Sensible Agent dynamically adapts both "what" assistance to offer and, crucially, "how" to deliver it, based on real-time multimodal context sensing. Informed by an expert workshop (n=12) and a data annotation study (n=40), the framework leverages egocentric cameras, multimodal sensing, and Large Multimodal Models (LMMs) to infer context and suggest appropriate actions delivered via minimally intrusive interaction modes. We demonstrate our prototype on an XR headset through a user study (n=10) in both AR and VR scenarios. Results indicate that Sensible Agent significantly reduces perceived interaction effort compared to voice-prompted baseline, while maintaining high usability and achieving higher preference.

**arXiv ID:** 2509.09255
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (17 papers)</h2></summary>

<details>
<summary><strong>Vejde: A Framework for Inductive Deep Reinforcement Learning Based on Factor Graph Color Refinement</strong> - Jakob Nyberg, Pontus Johnson - [[pdf]](https://arxiv.org/pdf/2509.09219)</summary>

**Abstract:** We present and evaluate Vejde; a framework which combines data abstraction, graph neural networks and reinforcement learning to produce inductive policy functions for decision problems with richly structured states, such as object classes and relations. MDP states are represented as data bases of facts about entities, and Vejde converts each state to a bipartite graph, which is mapped to latent states through neural message passing. The factored representation of both states and actions allows Vejde agents to handle problems of varying size and structure. We tested Vejde agents on eight problem domains defined in RDDL, with ten problem instances each, where policies were trained using both supervised and reinforcement learning. To test policy generalization, we separate problem instances in two sets, one for training and the other solely for testing. Test results on unseen instances for the Vejde agents were compared to MLP agents trained on each problem instance, as well as the online planning algorithm Prost. Our results show that Vejde policies in average generalize to the test instances without a significant loss in score. Additionally, the inductive agents received scores on unseen test instances that on average were close to the instance-specific MLP agents.

**arXiv ID:** 2509.09219
</details>

<details>
<summary><strong>Incentivizing Safer Actions in Policy Optimization for Constrained Reinforcement Learning</strong> - Somnath Hazra, Pallab Dasgupta, Soumyajit Dey - [[pdf]](https://arxiv.org/pdf/2509.09208)</summary>

**Abstract:** Constrained Reinforcement Learning (RL) aims to maximize the return while adhering to predefined constraint limits, which represent domain-specific safety requirements. In continuous control settings, where learning agents govern system actions, balancing the trade-off between reward maximization and constraint satisfaction remains a significant challenge. Policy optimization methods often exhibit instability near constraint boundaries, resulting in suboptimal training performance. To address this issue, we introduce a novel approach that integrates an adaptive incentive mechanism in addition to the reward structure to stay within the constraint bound before approaching the constraint boundary. Building on this insight, we propose Incrementally Penalized Proximal Policy Optimization (IP3O), a practical algorithm that enforces a progressively increasing penalty to stabilize training dynamics. Through empirical evaluation on benchmark environments, we demonstrate the efficacy of IP3O compared to the performance of state-of-the-art Safe RL algorithms. Furthermore, we provide theoretical guarantees by deriving a bound on the worst-case error of the optimality achieved by our algorithm.

**arXiv ID:** 2509.09208
</details>

<details>
<summary><strong>OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning</strong> - Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yuzheng Zhuang, Bowen Yang, He Zhu, Lingfeng Zhang, Pengwei Xie, David Gamaliel Arcos Bravo, Yingxue Zhang, Jianye Hao, Xingyue Quan - [[pdf]](https://arxiv.org/pdf/2509.09332)</summary>

**Abstract:** Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically this http URL address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: this https URL

**arXiv ID:** 2509.09332
</details>

<details>
<summary><strong>Feasibility-Guided Fair Adaptive Offline Reinforcement Learning for Medicaid Care Management</strong> - Sanjay Basu, Sadiq Y. Patel, Parth Sheth, Bhairavi Muralidharan, Namrata Elamaran, Aakriti Kinra, Rajaie Batniji - [[pdf]](https://arxiv.org/pdf/2509.09655)</summary>

**Abstract:** We introduce Feasibility-Guided Fair Adaptive Reinforcement Learning (FG-FARL), an offline RL procedure that calibrates per-group safety thresholds to reduce harm while equalizing a chosen fairness target (coverage or harm) across protected subgroups. Using de-identified longitudinal trajectories from a Medicaid population health management program, we evaluate FG-FARL against behavior cloning (BC) and HACO (Hybrid Adaptive Conformal Offline RL; a global conformal safety baseline). We report off-policy value estimates with bootstrap 95% confidence intervals and subgroup disparity analyses with p-values. FG-FARL achieves comparable value to baselines while improving fairness metrics, demonstrating a practical path to safer and more equitable decision support.

**arXiv ID:** 2509.09655
</details>

<details>
<summary><strong>SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning</strong> - Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, Ning Ding - [[pdf]](https://arxiv.org/pdf/2509.09674)</summary>

**Abstract:** Vision-Language-Action (VLA) models have recently emerged as a powerful paradigm for robotic manipulation. Despite substantial progress enabled by large-scale pretraining and supervised fine-tuning (SFT), these models face two fundamental challenges: (i) the scarcity and high cost of large-scale human-operated robotic trajectories required for SFT scaling, and (ii) limited generalization to tasks involving distribution shift. Recent breakthroughs in Large Reasoning Models (LRMs) demonstrate that reinforcement learning (RL) can dramatically enhance step-by-step reasoning capabilities, raising a natural question: Can RL similarly improve the long-horizon step-by-step action planning of VLA? In this work, we introduce SimpleVLA-RL, an efficient RL framework tailored for VLA models. Building upon veRL, we introduce VLA-specific trajectory sampling, scalable parallelization, multi-environment rendering, and optimized loss computation. When applied to OpenVLA-OFT, SimpleVLA-RL achieves SoTA performance on LIBERO and even outperforms $\pi_0$ on RoboTwin 1.0\&2.0 with the exploration-enhancing strategies we introduce. SimpleVLA-RL not only reduces dependence on large-scale data and enables robust generalization, but also remarkably surpasses SFT in real-world tasks. Moreover, we identify a novel phenomenon ``pushcut'' during RL training, wherein the policy discovers previously unseen patterns beyond those seen in the previous training process. Github: this https URL

**arXiv ID:** 2509.09674
</details>

<details>
<summary><strong>CDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models</strong> - Runpeng Dai, Linfeng Song, Haolin Liu, Zhenwen Liang, Dian Yu, Haitao Mi, Zhaopeng Tu, Rui Liu, Tong Zheng, Hongtu Zhu, Dong Yu - [[pdf]](https://arxiv.org/pdf/2509.09675)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful paradigm for enhancing the reasoning ability of Large Language Models (LLMs). Yet current RLVR methods often explore poorly, leading to premature convergence and entropy collapse. To address this challenge, we introduce Curiosity-Driven Exploration (CDE), a framework that leverages the model's own intrinsic sense of curiosity to guide exploration. We formalize curiosity with signals from both the actor and the critic: for the actor, we use perplexity over its generated response, and for the critic, we use the variance of value estimates from a multi-head architecture. Both signals serve as an exploration bonus within the RLVR framework to guide the model. Our theoretical analysis shows that the actor-wise bonus inherently penalizes overconfident errors and promotes diversity among correct responses; moreover, we connect the critic-wise bonus to the well-established count-based exploration bonus in RL. Empirically, our method achieves an approximate +3 point improvement over standard RLVR using GRPO/PPO on AIME benchmarks. Further analysis identifies a calibration collapse mechanism within RLVR, shedding light on common LLM failure modes.

**arXiv ID:** 2509.09675
</details>

<details>
<summary><strong>Deep Reinforcement Learning for Inventory Networks: Toward Reliable Policy Optimization</strong> - Matias Alvo, Daniel Russo, Yash Kanoria, Minuk Lee - [[pdf]](https://arxiv.org/pdf/2306.11246)</summary>

**Abstract:** We argue that inventory management presents unique opportunities for the reliable application of deep reinforcement learning (DRL). To enable this, we emphasize and test two complementary techniques. The first is Hindsight Differentiable Policy Optimization (HDPO), which uses pathwise gradients from offline counterfactual simulations to directly and efficiently optimize policy performance. Unlike standard policy gradient methods that rely on high-variance score-function estimators, HDPO computes gradients by differentiating through the known system dynamics. Via extensive benchmarking, we show that HDPO recovers near-optimal policies in settings with known or bounded optima, is more robust than variants of the REINFORCE algorithm, and significantly outperforms generalized newsvendor heuristics on problems using real time series data. Our second technique aligns neural policy architectures with the topology of the inventory network. We exploit Graph Neural Networks (GNNs) as a natural inductive bias for encoding supply chain structure, demonstrate that they can represent optimal and near-optimal policies in two theoretical settings, and empirically show that they reduce data requirements across six diverse inventory problems. A key obstacle to progress in this area is the lack of standardized benchmark problems. To address this gap, we open-source a suite of benchmark environments, along with our full codebase, to promote transparency and reproducibility. All resources are available at this http URL.

**arXiv ID:** 2306.11246
</details>

<details>
<summary><strong>Knowledge-Guided Biomarker Identification for Label-Free Single-Cell RNA-Seq Data: A Reinforcement Learning Perspective</strong> - Meng Xiao, Weiliang Zhang, Xiaohan Huang, Hengshu Zhu, Min Wu, Xiaoli Li, Yuanchun Zhou - [[pdf]](https://arxiv.org/pdf/2501.04718)</summary>

**Abstract:** Gene panel selection aims to identify the most informative genomic biomarkers in label-free genomic datasets. Traditional approaches, which rely on domain expertise, embedded machine learning models, or heuristic-based iterative optimization, often introduce biases and inefficiencies, potentially obscuring critical biological signals. To address these challenges, we present an iterative gene panel selection strategy that harnesses ensemble knowledge from existing gene selection algorithms to establish preliminary boundaries or prior knowledge, which guide the initial search space. Subsequently, we incorporate reinforcement learning through a reward function shaped by expert behavior, enabling dynamic refinement and targeted selection of gene panels. This integration mitigates biases stemming from initial boundaries while capitalizing on RL's stochastic adaptability. Comprehensive comparative experiments, case studies, and downstream analyses demonstrate the effectiveness of our method, highlighting its improved precision and efficiency for label-free biomarker discovery. Our results underscore the potential of this approach to advance single-cell genomics data analysis.

**arXiv ID:** 2501.04718
</details>

<details>
<summary><strong>Uncertainty-aware Diffusion and Reinforcement Learning for Joint Plane Localization and Anomaly Diagnosis in 3D Ultrasound</strong> - Yuhao Huang, Yueyue Xu, Haoran Dou, Jiaxiao Deng, Xin Yang, Hongyu Zheng, Dong Ni - [[pdf]](https://arxiv.org/pdf/2506.23538)</summary>

**Abstract:** Congenital uterine anomalies (CUAs) can lead to infertility, miscarriage, preterm birth, and an increased risk of pregnancy complications. Compared to traditional 2D ultrasound (US), 3D US can reconstruct the coronal plane, providing a clear visualization of the uterine morphology for assessing CUAs accurately. In this paper, we propose an intelligent system for simultaneous automated plane localization and CUA diagnosis. Our highlights are: 1) we develop a denoising diffusion model with local (plane) and global (volume/text) guidance, using an adaptive weighting strategy to optimize attention allocation to different conditions; 2) we introduce a reinforcement learning-based framework with unsupervised rewards to extract the key slice summary from redundant sequences, fully integrating information across multiple planes to reduce learning difficulty; 3) we provide text-driven uncertainty modeling for coarse prediction, and leverage it to adjust the classification probability for overall performance improvement. Extensive experiments on a large 3D uterine US dataset show the efficacy of our method, in terms of plane localization and CUA diagnosis. Code is available at this https URL.

**arXiv ID:** 2506.23538
</details>

<details>
<summary><strong>MagicGUI: A Foundational Mobile GUI Agent with Scalable Data Pipeline and Reinforcement Fine-tuning</strong> - Liujian Tang, Shaokang Dong, Yijia Huang, Minqi Xiang, Hongtao Ruan, Bin Wang, Shuo Li, Zhiheng Xi, Zhihui Cao, Hailiang Pang, Heng Kong, He Yang, Mingxu Chai, Zhilin Gao, Xingyu Liu, Yingnan Fu, Jiaming Liu, Xuanjing Huang, Yu-Gang Jiang, Tao Gui, Qi Zhang, Kang Wang, Yunke Zhang, Yuran Wang - [[pdf]](https://arxiv.org/pdf/2508.03700)</summary>

**Abstract:** This paper presents MagicGUI, a foundational mobile GUI agent designed to address critical challenges in perception, grounding, and reasoning within real-world mobile GUI environments. The framework is underpinned by following six key components: (1) a comprehensive and accurate dataset, constructed via the scalable GUI Data Pipeline, which aggregates the largest and most diverse GUI-centric multimodal data to date from open-source repositories, automated crawling, and targeted manual annotation; (2) enhanced perception and grounding capabilities, facilitating fine-grained multimodal alignment for UI element referencing, grounding, and screen comprehension; (3) a comprehensive and unified action space, encompassing both fundamental UI operations and complex interactive intents to support human-agent interactions; (4) planning-oriented reasoning mechanisms that enable the model to decompose complex user instructions into sequential actions with explicit intermediate meta-paln reasoning; (5) an iterative two-stage training procedure, combining large-scale continue pre-training on 7.8M samples with reinforcement fine-tuning utilizing a spatially enhanced composite reward and dual filtering strategy; and (6) competitive performance on both the proprietary Magic-RICH benchmark and over a dozen public benchmarks, achieving superior performance across GUI perception and agent tasks, while demonstrating robust generalization and real-world deployment potential in practical mobile GUI scenarios, as detailed in Figure 1.

**arXiv ID:** 2508.03700
</details>

<details>
<summary><strong>Klear-CodeTest: Scalable Test Case Generation for Code Reinforcement Learning</strong> - Jia Fu, Xinyu Yang, Hongzhi Zhang, Yahui Liu, Jingyuan Zhang, Qi Wang, Fuzheng Zhang, Guorui Zhou - [[pdf]](https://arxiv.org/pdf/2508.05710)</summary>

**Abstract:** Precise, correct feedback is crucial for effectively training large language models (LLMs) in code reinforcement learning. However, synthesizing high-quality test cases remains a profoundly challenging and unsolved problem. In this work, we present Klear-CodeTest, a comprehensive test case synthesis framework featuring rigorous verification to ensure quality and reliability of test cases. Our approach achieves broad coverage of programming problems via a novel Generator-Validation (G-V) framework, ensuring correctness through a consistency validation mechanism that verifies outputs against gold solutions. The proposed G-V framework generates comprehensive test cases including both regular and corner cases, enhancing test coverage and discriminative power for solution correctness assessment in code reinforcement learning. In addition, we design a multi-layered security sandbox system optimized for online verification platforms, guaranteeing safe and reliable code execution. Through comprehensive experiments, we demonstrate the effectiveness of our curated dataset, showing significant improvements in model performance and training stability. The source codes, curated dataset and sandbox system are available at: this https URL.

**arXiv ID:** 2508.05710
</details>

<details>
<summary><strong>Group Expectation Policy Optimization for Heterogeneous Reinforcement Learning</strong> - Han Zhang, Ruibin Zheng, Zexuan Yi, Zhuo Zhang, Hanyang Peng, Hui Wang, Zike Yuan, Cai Ke, Shiwei Chen, Jiacheng Yang, Yangning Li, Xiang Li, Jiangyue Yan, Yaoqi Liu, Liwen Jing, Jiayin Qi, Ruifeng Xu, Binxing Fang, Yue Yu - [[pdf]](https://arxiv.org/pdf/2508.17850)</summary>

**Abstract:** As single-center computing approaches power constraints, decentralized training is becoming essential. Reinforcement Learning (RL) post-training enhances Large Language Models (LLMs) but faces challenges in heterogeneous distributed environments due to its tightly-coupled sampling-learning alternation. We propose HeteroRL, an asynchronous RL architecture that decouples rollout sampling from parameter learning, enabling robust deployment across geographically distributed nodes under network delays. We identify that latency-induced KL divergence causes importance sampling failure due to high variance. To address this, we propose Group Expectation Policy Optimization (GEPO), which reduces importance weight variance through a refined sampling mechanism. Theoretically, GEPO achieves exponential variance reduction. Experiments show it maintains superior stability over methods like GRPO, with less than 3% performance degradation under 1800-second delays, demonstrating strong potential for decentralized RL in heterogeneous networks.

**arXiv ID:** 2508.17850
</details>

<details>
<summary><strong>MR-UIE: Multi-Perspective Reasoning with Reinforcement Learning for Universal Information Extraction</strong> - Zhongqiu Li, Shiquan Wang, Ruiyu Fang, Mengjiao Bao, Zhenhe Wu, Shuangyong Song, Yongxiang Li, Zhongjiang He - [[pdf]](https://arxiv.org/pdf/2509.09082)</summary>

**Abstract:** Large language models (LLMs) demonstrate robust capabilities across diverse research domains. However, their performance in universal information extraction (UIE) remains insufficient, especially when tackling structured output scenarios that involve complex schema descriptions and require multi-step reasoning. While existing approaches enhance the performance of LLMs through in-context learning and instruction tuning, significant limitations nonetheless persist. To enhance the model's generalization ability, we propose integrating reinforcement learning (RL) with multi-perspective reasoning for information extraction (IE) tasks. Our work transitions LLMs from passive extractors to active reasoners, enabling them to understand not only what to extract but also how to reason. Experiments conducted on multiple IE benchmarks demonstrate that MR-UIE consistently elevates extraction accuracy across domains and surpasses state-of-the-art methods on several datasets. Furthermore, incorporating multi-perspective reasoning into RL notably enhances generalization in complex IE tasks, underscoring the critical role of reasoning in challenging scenarios.

**arXiv ID:** 2509.09082
</details>

<details>
<summary><strong>Quantum Machine Learning, Quantitative Trading, Reinforcement Learning, Deep Learning</strong> - Jun-Hao Chen, Yu-Chien Huang, Yun-Cheng Tsai, Samuel Yen-Chi Chen - [[pdf]](https://arxiv.org/pdf/2509.09176)</summary>

**Abstract:** The convergence of quantum-inspired neural networks and deep reinforcement learning offers a promising avenue for financial trading. We implemented a trading agent for USD/TWD by integrating Quantum Long Short-Term Memory (QLSTM) for short-term trend prediction with Quantum Asynchronous Advantage Actor-Critic (QA3C), a quantum-enhanced variant of the classical A3C. Trained on data from 2000-01-01 to 2025-04-30 (80\% training, 20\% testing), the long-only agent achieves 11.87\% return over around 5 years with 0.92\% max drawdown, outperforming several currency ETFs. We detail state design (QLSTM features and indicators), reward function for trend-following/risk control, and multi-core training. Results show hybrid models yield competitive FX trading performance. Implications include QLSTM's effectiveness for small-profit trades with tight risk and future enhancements. Key hyperparameters: QLSTM sequence length$=$4, QA3C workers$=$8. Limitations: classical quantum simulation and simplified strategy. \footnote{The views expressed in this article are those of the authors and do not represent the views of Wells Fargo. This article is for informational purposes only. Nothing contained in this article should be construed as investment advice. Wells Fargo makes no express or implied warranties and expressly disclaims all legal, tax, and accounting implications related to this article.

**arXiv ID:** 2509.09176
</details>

<details>
<summary><strong>Near-Optimal Sample Complexity in Reward-Free Kernel-Based Reinforcement Learning</strong> - Aya Kayal, Sattar Vakili, Laura Toni, Alberto Bernacchia - [[pdf]](https://arxiv.org/pdf/2502.07715)</summary>

**Abstract:** Reinforcement Learning (RL) problems are being considered under increasingly more complex structures. While tabular and linear models have been thoroughly explored, the analytical study of RL under nonlinear function approximation, especially kernel-based models, has recently gained traction for their strong representational capacity and theoretical tractability. In this context, we examine the question of statistical efficiency in kernel-based RL within the reward-free RL framework, specifically asking: how many samples are required to design a near-optimal policy? Existing work addresses this question under restrictive assumptions about the class of kernel functions. We first explore this question by assuming a generative model, then relax this assumption at the cost of increasing the sample complexity by a factor of H, the length of the episode. We tackle this fundamental problem using a broad class of kernels and a simpler algorithm compared to prior work. Our approach derives new confidence intervals for kernel ridge regression, specific to our RL setting, which may be of broader applicability. We further validate our theoretical findings through simulations.

**arXiv ID:** 2502.07715
</details>

<details>
<summary><strong>Imagine, Verify, Execute: Memory-guided Agentic Exploration with Vision-Language Models</strong> - Seungjae Lee, Daniel Ekpo, Haowen Liu, Furong Huang, Abhinav Shrivastava, Jia-Bin Huang - [[pdf]](https://arxiv.org/pdf/2505.07815)</summary>

**Abstract:** Exploration is essential for general-purpose robotic learning, especially in open-ended environments where dense rewards, explicit goals, or task-specific supervision are scarce. Vision-language models (VLMs), with their semantic reasoning over objects, spatial relations, and potential outcomes, present a compelling foundation for generating high-level exploratory behaviors. However, their outputs are often ungrounded, making it difficult to determine whether imagined transitions are physically feasible or informative. To bridge the gap between imagination and execution, we present IVE (Imagine, Verify, Execute), an agentic exploration framework inspired by human curiosity. Human exploration is often driven by the desire to discover novel scene configurations and to deepen understanding of the environment. Similarly, IVE leverages VLMs to abstract RGB-D observations into semantic scene graphs, imagine novel scenes, predict their physical plausibility, and generate executable skill sequences through action tools. We evaluate IVE in both simulated and real-world tabletop environments. The results show that IVE enables more diverse and meaningful exploration than RL baselines, as evidenced by a 4.1 to 7.8x increase in the entropy of visited states. Moreover, the collected experience supports downstream learning, producing policies that closely match or exceed the performance of those trained on human-collected demonstrations.

**arXiv ID:** 2505.07815
</details>

<details>
<summary><strong>LIPM-Guided Reinforcement Learning for Stable and Perceptive Locomotion in Bipedal Robots</strong> - Haokai Su, Haoxiang Luo, Shunpeng Yang, Kaiwen Jiang, Wei Zhang, Hua Chen - [[pdf]](https://arxiv.org/pdf/2509.09106)</summary>

**Abstract:** Achieving stable and robust perceptive locomotion for bipedal robots in unstructured outdoor environments remains a critical challenge due to complex terrain geometry and susceptibility to external disturbances. In this work, we propose a novel reward design inspired by the Linear Inverted Pendulum Model (LIPM) to enable perceptive and stable locomotion in the wild. The LIPM provides theoretical guidance for dynamic balance by regulating the center of mass (CoM) height and the torso orientation. These are key factors for terrain-aware locomotion, as they help ensure a stable viewpoint for the robot's camera. Building on this insight, we design a reward function that promotes balance and dynamic stability while encouraging accurate CoM trajectory tracking. To adaptively trade off between velocity tracking and stability, we leverage the Reward Fusion Module (RFM) approach that prioritizes stability when needed. A double-critic architecture is adopted to separately evaluate stability and locomotion objectives, improving training efficiency and robustness. We validate our approach through extensive experiments on a bipedal robot in both simulation and real-world outdoor environments. The results demonstrate superior terrain adaptability, disturbance rejection, and consistent performance across a wide range of speeds and perceptual conditions.

**arXiv ID:** 2509.09106
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
