# Agent arXiv Daily

**Last Updated:** 2025-12-31 03:02:51

**Total Papers:** 76

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (7 papers)</h2></summary>

<details>
<summary><strong>SmartSnap: Proactive Evidence Seeking for Self-Verifying Agents</strong> - Shaofei Cai, Yulei Qin, Haojia Lin, Zihan Xu, Gang Li, Yuchen Shi, Zongyi Li, Yong Mao, Siqi Cai, Xiaoyu Tan, Yitao Liang, Ke Li, Xing Sun - [[pdf]](https://arxiv.org/pdf/2512.22322)</summary>

**Abstract:** Agentic reinforcement learning (RL) holds great promise for the development of autonomous agents under complex GUI tasks, but its scalability remains severely hampered by the verification of task completion. Existing task verification is treated as a passive, post-hoc process: a verifier (i.e., rule-based scoring script, reward or critic model, and LLM-as-a-Judge) analyzes the agent's entire interaction trajectory to determine if the agent succeeds. Such processing of verbose context that contains irrelevant, noisy history poses challenges to the verification protocols and therefore leads to prohibitive cost and low reliability. To overcome this bottleneck, we propose SmartSnap, a paradigm shift from this passive, post-hoc verification to proactive, in-situ self-verification by the agent itself. We introduce the Self-Verifying Agent, a new type of agent designed with dual missions: to not only complete a task but also to prove its accomplishment with curated snapshot evidences. Guided by our proposed 3C Principles (Completeness, Conciseness, and Creativity), the agent leverages its accessibility to the online environment to perform self-verification on a minimal, decisive set of snapshots. Such evidences are provided as the sole materials for a general LLM-as-a-Judge verifier to determine their validity and relevance. Experiments on mobile tasks across model families and scales demonstrate that our SmartSnap paradigm allows training LLM-driven agents in a scalable manner, bringing performance gains up to 26.08% and 16.66% respectively to 8B and 30B models. The synergizing between solution finding and evidence seeking facilitates the cultivation of efficient, self-verifying agents with competitive performance against DeepSeek V3.1 and Qwen3-235B-A22B.

**arXiv ID:** 2512.22322
</details>

<details>
<summary><strong>SuperiorGAT: Graph Attention Networks for Sparse LiDAR Point Cloud Reconstruction in Autonomous Systems</strong> - Khalfalla Awedat, Mohamed Abidalrekab, Gurcan Comert, Mustafa Ayad - [[pdf]](https://arxiv.org/pdf/2512.22439)</summary>

**Abstract:** LiDAR-based perception in autonomous systems is constrained by fixed vertical beam resolution and further compromised by beam dropout resulting from environmental occlusions. This paper introduces SuperiorGAT, a graph attention-based framework designed to reconstruct missing elevation information in sparse LiDAR point clouds. By modeling LiDAR scans as beam-aware graphs and incorporating gated residual fusion with feed-forward refinement, SuperiorGAT enables accurate reconstruction without increasing network depth. To evaluate performance, structured beam dropout is simulated by removing every fourth vertical scanning beam. Extensive experiments across diverse KITTI environments, including Person, Road, Campus, and City sequences, demonstrate that SuperiorGAT consistently achieves lower reconstruction error and improved geometric consistency compared to PointNet-based models and deeper GAT baselines. Qualitative X-Z projections further confirm the model's ability to preserve structural integrity with minimal vertical distortion. These results suggest that architectural refinement offers a computationally efficient method for improving LiDAR resolution without requiring additional sensor hardware.

**arXiv ID:** 2512.22439
</details>

<details>
<summary><strong>Agentic AI for Cyber Resilience: A New Security Paradigm and Its System-Theoretic Foundations</strong> - Tao Li, Quanyan Zhu - [[pdf]](https://arxiv.org/pdf/2512.22883)</summary>

**Abstract:** Cybersecurity is being fundamentally reshaped by foundation-model-based artificial intelligence. Large language models now enable autonomous planning, tool orchestration, and strategic adaptation at scale, challenging security architectures built on static rules, perimeter defenses, and human-centered workflows. This chapter argues for a shift from prevention-centric security toward agentic cyber resilience. Rather than seeking perfect protection, resilient systems must anticipate disruption, maintain critical functions under attack, recover efficiently, and learn continuously. We situate this shift within the historical evolution of cybersecurity paradigms, culminating in an AI-augmented paradigm where autonomous agents participate directly in sensing, reasoning, action, and adaptation across cyber and cyber-physical systems. We then develop a system-level framework for designing agentic AI workflows. A general agentic architecture is introduced, and attacker and defender workflows are analyzed as coupled adaptive processes, and game-theoretic formulations are shown to provide a unifying design language for autonomy allocation, information flow, and temporal composition. Case studies in automated penetration testing, remediation, and cyber deception illustrate how equilibrium-based design enables system-level resiliency design.

**arXiv ID:** 2512.22883
</details>

<details>
<summary><strong>AI Meets Brain: Memory Systems from Cognitive Neuroscience to Autonomous Agents</strong> - Jiafeng Liang, Hao Li, Chang Li, Jiaqi Zhou, Shixin Jiang, Zekun Wang, Changkai Ji, Zhihao Zhu, Runxuan Liu, Tao Ren, Jinlan Fu, See-Kiong Ng, Xia Liang, Ming Liu, Bing Qin - [[pdf]](https://arxiv.org/pdf/2512.23343)</summary>

**Abstract:** Memory serves as the pivotal nexus bridging past and future, providing both humans and AI systems with invaluable concepts and experience to navigate complex tasks. Recent research on autonomous agents has increasingly focused on designing efficient memory workflows by drawing on cognitive neuroscience. However, constrained by interdisciplinary barriers, existing works struggle to assimilate the essence of human memory mechanisms. To bridge this gap, we systematically synthesizes interdisciplinary knowledge of memory, connecting insights from cognitive neuroscience with LLM-driven agents. Specifically, we first elucidate the definition and function of memory along a progressive trajectory from cognitive neuroscience through LLMs to agents. We then provide a comparative analysis of memory taxonomy, storage mechanisms, and the complete management lifecycle from both biological and artificial perspectives. Subsequently, we review the mainstream benchmarks for evaluating agent memory. Additionally, we explore memory security from dual perspectives of attack and defense. Finally, we envision future research directions, with a focus on multimodal memory systems and skill acquisition.

**arXiv ID:** 2512.23343
</details>

<details>
<summary><strong>Do You Feel Comfortable? Detecting Hidden Conversational Escalation in AI Chatbots</strong> - Jihyung Park, Saleh Afroogh, Junfeng Jiao - [[pdf]](https://arxiv.org/pdf/2512.06193)</summary>

**Abstract:** Large Language Models (LLM) are increasingly integrated into everyday interactions, serving not only as information assistants but also as emotional companions. Even in the absence of explicit toxicity, repeated emotional reinforcement or affective drift can gradually escalate distress in a form of \textit{implicit harm} that traditional toxicity filters fail to detect. Existing guardrail mechanisms often rely on external classifiers or clinical rubrics that may lag behind the nuanced, real-time dynamics of a developing conversation. To address this gap, we propose GAUGE (Guarding Affective Utterance Generation Escalation), logit-based framework for the real-time detection of hidden conversational escalation. GAUGE measures how an LLM's output probabilistically shifts the affective state of a dialogue.

**arXiv ID:** 2512.06193
</details>

<details>
<summary><strong>Emotion-Inspired Learning Signals (EILS): A Homeostatic Framework for Adaptive Autonomous Agents</strong> - Dhruv Tiwari - [[pdf]](https://arxiv.org/pdf/2512.22200)</summary>

**Abstract:** The ruling method in modern Artificial Intelligence spanning from Deep Reinforcement Learning (DRL) to Large Language Models (LLMs) relies on a surge of static, externally defined reward functions. While this "extrinsic maximization" approach has rendered superhuman performance in closed, stationary fields, it produces agents that are fragile in open-ended, real-world environments. Standard agents lack internal autonomy: they struggle to explore without dense feedback, fail to adapt to distribution shifts (non-stationarity), and require extensive manual tuning of static hyperparameters. This paper proposes that the unaddressed factor in robust autonomy is a functional analog to biological emotion, serving as a high-level homeostatic control mechanism. We introduce Emotion-Inspired Learning Signals (EILS), a unified framework that replaces scattered optimization heuristics with a coherent, bio-inspired internal feedback engine. Unlike traditional methods that treat emotions as semantic labels, EILS models them as continuous, homeostatic appraisal signals such as Curiosity, Stress, and Confidence. We formalize these signals as vector-valued internal states derived from interaction history. These states dynamically modulate the agent's optimization landscape in real time: curiosity regulates entropy to prevent mode collapse, stress modulates plasticity to overcome inactivity, and confidence adapts trust regions to stabilize convergence. We hypothesize that this closed-loop homeostatic regulation can enable EILS agents to outperform standard baselines in terms of sample efficiency and non-stationary adaptation.

**arXiv ID:** 2512.22200
</details>

<details>
<summary><strong>Relational Mediators: LLM Chatbots as Boundary Objects in Psychotherapy</strong> - Jiatao Quan, Ziyue Li, Tian Qi Zhu, Yuxuan Li, Baoying Wang, Wanda Pratt, Nan Gao - [[pdf]](https://arxiv.org/pdf/2512.22462)</summary>

**Abstract:** As large language models (LLMs) are embedded into mental health technologies, they are often framed either as tools assisting therapists or autonomous therapeutic systems. Such perspectives overlook their potential to mediate relational complexities in therapy, particularly for systemically marginalized clients. Drawing on in-depth interviews with 12 therapists and 12 marginalized clients in China, including LGBTQ+ individuals or those from other marginalized backgrounds, we identify enduring relational challenges: difficulties building trust amid institutional barriers, the burden clients carry in educating therapists about marginalized identities, and challenges sustaining authentic self-disclosure across therapy and daily life. We argue that addressing these challenges requires AI systems capable of actively mediating underlying knowledge gaps, power asymmetries, and contextual disconnects. To this end, we propose the Dynamic Boundary Mediation Framework, which reconceptualizes LLM-enhanced systems as adaptive boundary objects that shift mediating roles across therapeutic stages. The framework delineates three forms of mediation: Epistemic (reducing knowledge asymmetries), Relational (rebalancing power dynamics), and Contextual (bridging therapy-life discontinuities). This framework offers a pathway toward designing relationally accountable AI systems that center the lived realities of marginalized users and more effectively support therapeutic relationships.

**arXiv ID:** 2512.22462
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (6 papers)</h2></summary>

<details>
<summary><strong>Tyee: A Unified, Modular, and Fully-Integrated Configurable Toolkit for Intelligent Physiological Health Care</strong> - Tao Zhou, Lingyu Shu, Zixing Zhang, Jing Han - [[pdf]](https://arxiv.org/pdf/2512.22601)</summary>

**Abstract:** Deep learning has shown great promise in physiological signal analysis, yet its progress is hindered by heterogeneous data formats, inconsistent preprocessing strategies, fragmented model pipelines, and non-reproducible experimental setups. To address these limitations, we present Tyee, a unified, modular, and fully-integrated configurable toolkit designed for intelligent physiological healthcare. Tyee introduces three key innovations: (1) a unified data interface and configurable preprocessing pipeline for 12 kinds of signal modalities; (2) a modular and extensible architecture enabling flexible integration and rapid prototyping across tasks; and (3) end-to-end workflow configuration, promoting reproducible and scalable experimentation. Tyee demonstrates consistent practical effectiveness and generalizability, outperforming or matching baselines across all evaluated tasks (with state-of-the-art results on 12 of 13 datasets). The Tyee toolkit is released at this https URL and actively maintained.

**arXiv ID:** 2512.22601
</details>

<details>
<summary><strong>Multimodal Fact-Checking: An Agent-based Approach</strong> - Danni Xu, Shaojing Fan, Xuanang Cheng, Mohan Kankanhalli - [[pdf]](https://arxiv.org/pdf/2512.22933)</summary>

**Abstract:** The rapid spread of multimodal misinformation poses a growing challenge for automated fact-checking systems. Existing approaches, including large vision language models (LVLMs) and deep multimodal fusion methods, often fall short due to limited reasoning and shallow evidence utilization. A key bottleneck is the lack of dedicated datasets that provide complete real-world multimodal misinformation instances accompanied by annotated reasoning processes and verifiable evidence. To address this limitation, we introduce RW-Post, a high-quality and explainable dataset for real-world multimodal fact-checking. RW-Post aligns real-world multimodal claims with their original social media posts, preserving the rich contextual information in which the claims are made. In addition, the dataset includes detailed reasoning and explicitly linked evidence, which are derived from human written fact-checking articles via a large language model assisted extraction pipeline, enabling comprehensive verification and explanation. Building upon RW-Post, we propose AgentFact, an agent-based multimodal fact-checking framework designed to emulate the human verification workflow. AgentFact consists of five specialized agents that collaboratively handle key fact-checking subtasks, including strategy planning, high-quality evidence retrieval, visual analysis, reasoning, and explanation generation. These agents are orchestrated through an iterative workflow that alternates between evidence searching and task-aware evidence filtering and reasoning, facilitating strategic decision-making and systematic evidence analysis. Extensive experimental results demonstrate that the synergy between RW-Post and AgentFact substantially improves both the accuracy and interpretability of multimodal fact-checking.

**arXiv ID:** 2512.22933
</details>

<details>
<summary><strong>It's a TRAP! Task-Redirecting Agent Persuasion Benchmark for Web Agents</strong> - Karolina Korgul, Yushi Yang, Arkadiusz Drohomirecki, Piotr Błaszczyk, Will Howard, Lukas Aichberger, Chris Russell, Philip H.S. Torr, Adam Mahdi, Adel Bibi - [[pdf]](https://arxiv.org/pdf/2512.23128)</summary>

**Abstract:** Web-based agents powered by large language models are increasingly used for tasks such as email management or professional networking. Their reliance on dynamic web content, however, makes them vulnerable to prompt injection attacks: adversarial instructions hidden in interface elements that persuade the agent to divert from its original task. We introduce the Task-Redirecting Agent Persuasion Benchmark (TRAP), an evaluation for studying how persuasion techniques misguide autonomous web agents on realistic tasks. Across six frontier models, agents are susceptible to prompt injection in 25\% of tasks on average (13\% for GPT-5 to 43\% for DeepSeek-R1), with small interface or contextual changes often doubling success rates and revealing systemic, psychologically driven vulnerabilities in web-based agents. We also provide a modular social-engineering injection framework with controlled experiments on high-fidelity website clones, allowing for further benchmark expansion.

**arXiv ID:** 2512.23128
</details>

<details>
<summary><strong>KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta</strong> - Gang Liao, Hongsen Qin, Ying Wang, Alicia Golden, Michael Kuchnik, Yavuz Yetim, Jia Jiunn Ang, Chunli Fu, Yihan He, Samuel Hsia, Zewei Jiang, Dianshi Li, Uladzimir Pashkevich, Varna Puvvada, Feng Shi, Matt Steiner, Ruichao Xiao, Nathan Yan, Xiayu Yu, Zhou Fang, Abdul Zainul-Abedin, Ketan Singh, Hongtao Yu, Wenyuan Chi, Barney Huang, Sean Zhang, Noah Weller, Zach Marine, Wyatt Cook, Carole-Jean Wu, Gaoxiang Liu - [[pdf]](https://arxiv.org/pdf/2512.23236)</summary>

**Abstract:** Making deep learning recommendation model (DLRM) training and inference fast and efficient is important. However, this presents three key system challenges - model architecture diversity, kernel primitive diversity, and hardware generation and architecture heterogeneity. This paper presents KernelEvolve-an agentic kernel coding framework-to tackle heterogeneity at-scale for DLRM. KernelEvolve is designed to take kernel specifications as input and automate the process of kernel generation and optimization for recommendation model across heterogeneous hardware architectures. KernelEvolve does so by operating at multiple programming abstractions, from Triton and CuTe DSL to low-level hardware agnostic languages, spanning the full hardware-software optimization stack. The kernel optimization process is described as graph-based search with selection policy, universal operator, fitness function, and termination rule, dynamically adapts to runtime execution context through retrieval-augmented prompt synthesis. We designed, implemented, and deployed KernelEvolve to optimize a wide variety of production recommendation models across generations of NVIDIA and AMD GPUs, as well as Meta's AI accelerators. We validate KernelEvolve on the publicly-available KernelBench suite, achieving 100% pass rate on all 250 problems across three difficulty levels, and 160 PyTorch ATen operators across three heterogeneous hardware platforms, demonstrating 100% correctness. KernelEvolve reduces development time from weeks to hours and achieves substantial performance improvements over PyTorch baselines across diverse production use cases and for heterogeneous AI systems at-scale. Beyond performance efficiency improvements, KernelEvolve significantly mitigates the programmability barrier for new AI hardware by enabling automated kernel generation for in-house developed AI hardware.

**arXiv ID:** 2512.23236
</details>

<details>
<summary><strong>Nested Browser-Use Learning for Agentic Information Seeking</strong> - Baixuan Li, Jialong Wu, Wenbiao Yin, Kuan Li, Zhongwang Zhang, Huifeng Yin, Zhengwei Tao, Liwen Zhang, Pengjun Xie, Jingren Zhou, Yong Jiang - [[pdf]](https://arxiv.org/pdf/2512.23647)</summary>

**Abstract:** Information-seeking (IS) agents have achieved strong performance across a range of wide and deep search tasks, yet their tool use remains largely restricted to API-level snippet retrieval and URL-based page fetching, limiting access to the richer information available through real browsing. While full browser interaction could unlock deeper capabilities, its fine-grained control and verbose page content returns introduce substantial complexity for ReAct-style function-calling agents. To bridge this gap, we propose Nested Browser-Use Learning (NestBrowse), which introduces a minimal and complete browser-action framework that decouples interaction control from page exploration through a nested structure. This design simplifies agentic reasoning while enabling effective deep-web information acquisition. Empirical results on challenging deep IS benchmarks demonstrate that NestBrowse offers clear benefits in practice. Further in-depth analyses underscore its efficiency and flexibility.

**arXiv ID:** 2512.23647
</details>

<details>
<summary><strong>Never-Ending Behavior-Cloning Agent for Robotic Manipulation</strong> - Wenqi Liang, Gan Sun, Yao He, Yu Ren, Jiahua Dong, Yang Cong - [[pdf]](https://arxiv.org/pdf/2403.00336)</summary>

**Abstract:** Relying on multi-modal observations, embodied robots (e.g., humanoid robots) could perform multiple robotic manipulation tasks in unstructured real-world environments. However, most language-conditioned behavior-cloning agents in robots still face existing long-standing challenges, i.e., 3D scene representation and human-level task learning, when adapting into a series of new tasks in practical scenarios. We here investigate these above challenges with NBAgent in embodied robots, a pioneering language-conditioned Never-ending Behavior-cloning Agent, which can continually learn observation knowledge of novel 3D scene semantics and robot manipulation skills from skill-shared and skill-specific attributes, respectively. Specifically, we propose a skill-shared semantic rendering module and a skill-shared representation distillation module to effectively learn 3D scene semantics from skill-shared attribute, further tackling 3D scene representation overlooking. Meanwhile, we establish a skill-specific evolving planner to perform manipulation knowledge decoupling, which can continually embed novel skill-specific knowledge like human from latent and low-rank space. Finally, we design a never-ending embodied robot manipulation benchmark, and expensive experiments demonstrate the significant performance of our method.

**arXiv ID:** 2403.00336
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (26 papers)</h2></summary>

<details>
<summary><strong>Agent2World: Learning to Generate Symbolic World Models via Adaptive Multi-Agent Feedback</strong> - Mengkang Hu, Bowei Xia, Yuran Wu, Ailing Yu, Yude Zou, Qiguang Chen, Shijian Wang, Jiarui Jin, Kexin Li, Wenxiang Jiao, Yuan Lu, Ping Luo - [[pdf]](https://arxiv.org/pdf/2512.22336)</summary>

**Abstract:** Symbolic world models (e.g., PDDL domains or executable simulators) are central to model-based planning, but training LLMs to generate such world models is limited by the lack of large-scale verifiable supervision. Current approaches rely primarily on static validation methods that fail to catch behavior-level errors arising from interactive execution. In this paper, we propose Agent2World, a tool-augmented multi-agent framework that achieves strong inference-time world-model generation and also serves as a data engine for supervised fine-tuning, by grounding generation in multi-agent feedback. Agent2World follows a three-stage pipeline: (i) A Deep Researcher agent performs knowledge synthesis by web searching to address specification gaps; (ii) A Model Developer agent implements executable world models; And (iii) a specialized Testing Team conducts adaptive unit testing and simulation-based validation. Agent2World demonstrates superior inference-time performance across three benchmarks spanning both Planning Domain Definition Language (PDDL) and executable code representations, achieving consistent state-of-the-art results. Beyond inference, Testing Team serves as an interactive environment for the Model Developer, providing behavior-aware adaptive feedback that yields multi-turn training trajectories. The model fine-tuned on these trajectories substantially improves world-model generation, yielding an average relative gain of 30.95% over the same model before training. Project page: this https URL.

**arXiv ID:** 2512.22336
</details>

<details>
<summary><strong>SANet: A Semantic-aware Agentic AI Networking Framework for Cross-layer Optimization in 6G</strong> - Yong Xiao, Xubo Li, Haoran Zhou, Yingyu Li, Yayu Gao, Guangming Shi, Ping Zhang, Marwan Krunz - [[pdf]](https://arxiv.org/pdf/2512.22579)</summary>

**Abstract:** Agentic AI networking (AgentNet) is a novel AI-native networking paradigm in which a large number of specialized AI agents collaborate to perform autonomous decision-making, dynamic environmental adaptation, and complex missions. It has the potential to facilitate real-time network management and optimization functions, including self-configuration, self-optimization, and self-adaptation across diverse and complex environments. This paper proposes SANet, a novel semantic-aware AgentNet architecture for wireless networks that can infer the semantic goal of the user and automatically assign agents associated with different layers of the network to fulfill the inferred goal. Motivated by the fact that AgentNet is a decentralized framework in which collaborating agents may generally have different and even conflicting objectives, we formulate the decentralized optimization of SANet as a multi-agent multi-objective problem, and focus on finding the Pareto-optimal solution for agents with distinct and potentially conflicting objectives. We propose three novel metrics for evaluating SANet. Furthermore, we develop a model partition and sharing (MoPS) framework in which large models, e.g., deep learning models, of different agents can be partitioned into shared and agent-specific parts that are jointly constructed and deployed according to agents' local computational resources. Two decentralized optimization algorithms are proposed. We derive theoretical bounds and prove that there exists a three-way tradeoff among optimization, generalization, and conflicting errors. We develop an open-source RAN and core network-based hardware prototype that implements agents to interact with three different layers of the network. Experimental results show that the proposed framework achieved performance gains of up to 14.61% while requiring only 44.37% of FLOPs required by state-of-the-art algorithms.

**arXiv ID:** 2512.22579
</details>

<details>
<summary><strong>LLM Agents as VC investors: Predicting Startup Success via RolePlay-Based Collective Simulation</strong> - Zhongyang Liu, Haoyu Pei, Xiangyi Xiao, Xiaocong Du, Yihui Li, Suting Hong, Kunpeng Zhang, Haipeng Zhang - [[pdf]](https://arxiv.org/pdf/2512.22608)</summary>

**Abstract:** Due to the high value and high failure rate of startups, predicting their success has become a critical challenge across interdisciplinary research. Existing approaches typically model success prediction from the perspective of a single decision-maker, overlooking the collective dynamics of investor groups that dominate real-world venture capital (VC) decisions. In this paper, we propose SimVC-CAS, a novel collective agent system that simulates VC decision-making as a multi-agent interaction process. By designing role-playing agents and a GNN-based supervised interaction module, we reformulate startup financing prediction as a group decision-making task, capturing both enterprise fundamentals and the behavioral dynamics of potential investor networks. Each agent embodies an investor with unique traits and preferences, enabling heterogeneous evaluation and realistic information exchange through a graph-structured co-investment network. Using real-world data from PitchBook and under strict data leakage controls, we show that SimVC-CAS significantly improves predictive accuracy while providing interpretable, multiperspective reasoning, for example, approximately 25% relative improvement with respect to average precision@10. SimVC-CAS also sheds light on other complex group decision scenarios.

**arXiv ID:** 2512.22608
</details>

<details>
<summary><strong>SAMP-HDRL: Segmented Allocation with Momentum-Adjusted Utility for Multi-agent Portfolio Management via Hierarchical Deep Reinforcement Learning</strong> - Xiaotian Ren, Nuerxiati Abudurexiti, Zhengyong Jiang, Angelos Stefanidis, Hongbin Liu, Jionglong Su - [[pdf]](https://arxiv.org/pdf/2512.22895)</summary>

**Abstract:** Portfolio optimization in non-stationary markets is challenging due to regime shifts, dynamic correlations, and the limited interpretability of deep reinforcement learning (DRL) policies. We propose a Segmented Allocation with Momentum-Adjusted Utility for Multi-agent Portfolio Management via Hierarchical Deep Reinforcement Learning (SAMP-HDRL). The framework first applies dynamic asset grouping to partition the market into high-quality and ordinary subsets. An upper-level agent extracts global market signals, while lower-level agents perform intra-group allocation under mask constraints. A utility-based capital allocation mechanism integrates risky and risk-free assets, ensuring coherent coordination between global and local decisions. backtests across three market regimes (2019--2021) demonstrate that SAMP-HDRL consistently outperforms nine traditional baselines and nine DRL benchmarks under volatile and oscillating conditions. Compared with the strongest baseline, our method achieves at least 5\% higher Return, 5\% higher Sharpe ratio, 5\% higher Sortino ratio, and 2\% higher Omega ratio, with substantially larger gains observed in turbulent markets. Ablation studies confirm that upper--lower coordination, dynamic clustering, and capital allocation are indispensable to robustness. SHAP-based interpretability further reveals a complementary ``diversified + concentrated'' mechanism across agents, providing transparent insights into decision-making. Overall, SAMP-HDRL embeds structural market constraints directly into the DRL pipeline, offering improved adaptability, robustness, and interpretability in complex financial environments.

**arXiv ID:** 2512.22895
</details>

<details>
<summary><strong>AKG kernel Agent: A Multi-Agent Framework for Cross-Platform Kernel Synthesis</strong> - Jinye Du, Quan Yuan, Zuyao Zhang, Yanzhi Yi, Jiahui Hu, Wangyi Chen, Yiyang Zhu, Qishui Zheng, Wenxiang Zou, Xiangyu Chang, Zuohe Zheng, Zichun Ye, Chao Liu, Shanni Li, Renwei Zhang, Yiping Deng, Xinwei Hu, Xuefeng Jin, Jie Zhao - [[pdf]](https://arxiv.org/pdf/2512.23424)</summary>

**Abstract:** Modern AI models demand high-performance computation kernels. The growing complexity of LLMs, multimodal architectures, and recommendation systems, combined with techniques like sparsity and quantization, creates significant computational challenges. Moreover, frequent hardware updates and diverse chip architectures further complicate this landscape, requiring tailored kernel implementations for each platform. However, manual optimization cannot keep pace with these demands, creating a critical bottleneck in AI system development. Recent advances in LLM code generation capabilities have opened new possibilities for automating kernel development. In this work, we propose AKG kernel agent (AI-driven Kernel Generator), a multi-agent system that automates kernel generation, migration, and performance tuning. AKG kernel agent is designed to support multiple domain-specific languages (DSLs), including Triton, TileLang, CPP, and CUDA-C, enabling it to target different hardware backends while maintaining correctness and portability. The system's modular design allows rapid integration of new DSLs and hardware targets. When evaluated on KernelBench using Triton DSL across GPU and NPU backends, AKG kernel agent achieves an average speedup of 1.46$\times$ over PyTorch Eager baselines implementations, demonstrating its effectiveness in accelerating kernel development for modern AI workloads.

**arXiv ID:** 2512.23424
</details>

<details>
<summary><strong>Adaptive GPU Resource Allocation for Multi-Agent Collaborative Reasoning in Serverless Environments</strong> - Guilin Zhang, Wulan Guo, Ziqi Tan - [[pdf]](https://arxiv.org/pdf/2512.22149)</summary>

**Abstract:** Multi-agent systems powered by large language models have emerged as a promising paradigm for solving complex reasoning tasks through collaborative intelligence. However, efficiently deploying these systems on serverless GPU platforms presents significant resource allocation challenges due to heterogeneous agent workloads, varying computational demands, and the need for cost-effective scaling. This paper presents an adaptive GPU resource allocation framework that achieves 85\% latency reduction compared to round-robin scheduling while maintaining comparable throughput to static allocation, using an $O(N)$ complexity algorithm for real-time adaptation. Our approach dynamically allocates GPU resources based on workload characteristics, agent priorities, and minimum resource requirements, enabling efficient utilization while maintaining quality of service. The framework addresses three key challenges: (1) heterogeneous computational demands across lightweight coordinators and heavyweight specialists, (2) dynamic workload fluctuations requiring millisecond-scale reallocation, and (3) capacity constraints in serverless environments. Through comprehensive simulations modeling realistic multi-agent workflows with four heterogeneous agents, we demonstrate that adaptive allocation outperforms static equal and round-robin strategies across latency, cost, and GPU utilization metrics. The framework provides a practical solution for deploying cost-efficient multi-agent AI systems on serverless GPU infrastructure.

**arXiv ID:** 2512.22149
</details>

<details>
<summary><strong>Solving Multi-Agent Multi-Goal Path Finding Problems in Polynomial Time</strong> - Stefan Edelkamp - [[pdf]](https://arxiv.org/pdf/2512.22171)</summary>

**Abstract:** In this paper, we plan missions for a fleet of agents in undirected graphs, such as grids, with multiple goals. In contrast to regular multi-agent path-finding, the solver finds and updates the assignment of goals to the agents on its own. In the continuous case for a point agent with motions in the Euclidean plane, the problem can be solved arbitrarily close to optimal. For discrete variants that incur node and edge conflicts, we show that it can be solved in polynomial time, which is unexpected, since traditional vehicle routing on general graphs is NP-hard. We implement a corresponding planner that finds conflict-free optimized routes for the agents. Global assignment strategies greatly reduce the number of conflicts, with the remaining ones resolved by elaborating on the concept of ants-on-the-stick, by solving local assignment problems, by interleaving agent paths, and by kicking agents that have already arrived out of their destinations

**arXiv ID:** 2512.22171
</details>

<details>
<summary><strong>VULCAN: Tool-Augmented Multi Agents for Iterative 3D Object Arrangement</strong> - Zhengfei Kuang, Rui Lin, Long Zhao, Gordon Wetzstein, Saining Xie, Sanghyun Woo - [[pdf]](https://arxiv.org/pdf/2512.22351)</summary>

**Abstract:** Despite the remarkable progress of Multimodal Large Language Models (MLLMs) in 2D vision-language tasks, their application to complex 3D scene manipulation remains underexplored. In this paper, we bridge this critical gap by tackling three key challenges in 3D object arrangement task using MLLMs. First, to address the weak visual grounding of MLLMs, which struggle to link programmatic edits with precise 3D outcomes, we introduce an MCP-based API. This shifts the interaction from brittle raw code manipulation to more robust, function-level updates. Second, we augment the MLLM's 3D scene understanding with a suite of specialized visual tools to analyze scene state, gather spatial information, and validate action outcomes. This perceptual feedback loop is critical for closing the gap between language-based updates and precise 3D-aware manipulation. Third, to manage the iterative, error-prone updates, we propose a collaborative multi-agent framework with designated roles for planning, execution, and verification. This decomposition allows the system to robustly handle multi-step instructions and recover from intermediate errors. We demonstrate the effectiveness of our approach on a diverse set of 25 complex object arrangement tasks, where it significantly outperforms existing baselines. Website: this http URL

**arXiv ID:** 2512.22351
</details>

<details>
<summary><strong>Hierarchical Pedagogical Oversight: A Multi-Agent Adversarial Framework for Reliable AI Tutoring</strong> - Saisab Sadhu, Ashim Dhor - [[pdf]](https://arxiv.org/pdf/2512.22496)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly deployed as automated tutors to address educator shortages; however, they often fail at pedagogical reasoning, frequently validating incorrect student solutions (sycophancy) or providing overly direct answers that hinder learning. We introduce Hierarchical Pedagogical Oversight (HPO), a framework that adapts structured adversarial synthesis to educational assessment. Unlike cooperative multi-agent systems that often drift toward superficial consensus, HPO enforces a dialectical separation of concerns: specialist agents first distill dialogue context, which then grounds a moderated, five-act debate between opposing pedagogical critics. We evaluate this framework on the MRBench dataset of 1,214 middle-school mathematics dialogues. Our 8B-parameter model achieves a Macro F1 of 0.845, outperforming GPT-4o (0.812) by 3.3% while using 20 times fewer parameters. These results establish adversarial reasoning as a critical mechanism for deploying reliable, low-compute pedagogical oversight in resource-constrained environments.

**arXiv ID:** 2512.22496
</details>

<details>
<summary><strong>Reinforcement Networks: novel framework for collaborative Multi-Agent Reinforcement Learning tasks</strong> - Maksim Kryzhanovskiy, Svetlana Glazyrina, Roman Ischenko, Konstantin Vorontsov - [[pdf]](https://arxiv.org/pdf/2512.22876)</summary>

**Abstract:** Modern AI systems often comprise multiple learnable components that can be naturally organized as graphs. A central challenge is the end-to-end training of such systems without restrictive architectural or training assumptions. Such tasks fit the theory and approaches of the collaborative Multi-Agent Reinforcement Learning (MARL) field. We introduce Reinforcement Networks, a general framework for MARL that organizes agents as vertices in a directed acyclic graph (DAG). This structure extends hierarchical RL to arbitrary DAGs, enabling flexible credit assignment and scalable coordination while avoiding strict topologies, fully centralized training, and other limitations of current approaches. We formalize training and inference methods for the Reinforcement Networks framework and connect it to the LevelEnv concept to support reproducible construction, training, and evaluation. We demonstrate the effectiveness of our approach on several collaborative MARL setups by developing several Reinforcement Networks models that achieve improved performance over standard MARL baselines. Beyond empirical gains, Reinforcement Networks unify hierarchical, modular, and graph-structured views of MARL, opening a principled path toward designing and training complex multi-agent systems. We conclude with theoretical and practical directions - richer graph morphologies, compositional curricula, and graph-aware exploration. That positions Reinforcement Networks as a foundation for a new line of research in scalable, structured MARL.

**arXiv ID:** 2512.22876
</details>

<details>
<summary><strong>Heterogeneity in Multi-Agent Reinforcement Learning</strong> - Tianyi Hu, Zhiqiang Pu, Yuan Wang, Tenghai Qiu, Min Chen, Xin Yu - [[pdf]](https://arxiv.org/pdf/2512.22941)</summary>

**Abstract:** Heterogeneity is a fundamental property in multi-agent reinforcement learning (MARL), which is closely related not only to the functional differences of agents, but also to policy diversity and environmental interactions. However, the MARL field currently lacks a rigorous definition and deeper understanding of heterogeneity. This paper systematically discusses heterogeneity in MARL from the perspectives of definition, quantification, and utilization. First, based on an agent-level modeling of MARL, we categorize heterogeneity into five types and provide mathematical definitions. Second, we define the concept of heterogeneity distance and propose a practical quantification method. Third, we design a heterogeneity-based multi-agent dynamic parameter sharing algorithm as an example of the application of our methodology. Case studies demonstrate that our method can effectively identify and quantify various types of agent heterogeneity. Experimental results show that the proposed algorithm, compared to other parameter sharing baselines, has better interpretability and stronger adaptability. The proposed methodology will help the MARL community gain a more comprehensive and profound understanding of heterogeneity, and further promote the development of practical algorithms.

**arXiv ID:** 2512.22941
</details>

<details>
<summary><strong>Agentic AI for Autonomous Defense in Software Supply Chain Security: Beyond Provenance to Vulnerability Mitigation</strong> - Toqeer Ali Syed, Mohammad Riyaz Belgaum, Salman Jan, Asadullah Abdullah Khan, Saad Said Alqahtani - [[pdf]](https://arxiv.org/pdf/2512.23480)</summary>

**Abstract:** The software supply chain attacks are becoming more and more focused on trusted development and delivery procedures, so the conventional post-build integrity mechanisms cannot be used anymore. The available frameworks like SLSA, SBOM and in toto are majorly used to offer provenance and traceability but do not have the capabilities of actively identifying and removing vulnerabilities in software production. The current paper includes an example of agentic artificial intelligence (AI) based on autonomous software supply chain security that combines large language model (LLM)-based reasoning, reinforcement learning (RL), and multi-agent coordination. The suggested system utilizes specialized security agents coordinated with the help of LangChain and LangGraph, communicates with actual CI/CD environments with the Model Context Protocol (MCP), and documents all the observations and actions in a blockchain security ledger to ensure integrity and auditing. Reinforcement learning can be used to achieve adaptive mitigation strategies that consider the balance between security effectiveness and the operational overhead, and LLMs can be used to achieve semantic vulnerability analysis, as well as explainable decisions. This framework is tested based on simulated pipelines, as well as, actual world CI/CD integrations on GitHub Actions and Jenkins, including injection attacks, insecure deserialization, access control violations, and configuration errors. Experimental outcomes indicate better detection accuracy, shorter mitigation latency and reasonable build-time overhead than rule-based, provenance only and RL only baselines. These results show that agentic AI can facilitate the transition to self defending, proactive software supply chains rather than reactive verification ones.

**arXiv ID:** 2512.23480
</details>

<details>
<summary><strong>Toward Trustworthy Agentic AI: A Multimodal Framework for Preventing Prompt Injection Attacks</strong> - Toqeer Ali Syed, Mishal Ateeq Almutairi, Mahmoud Abdel Moaty - [[pdf]](https://arxiv.org/pdf/2512.23557)</summary>

**Abstract:** Powerful autonomous systems, which reason, plan, and converse using and between numerous tools and agents, are made possible by Large Language Models (LLMs), Vision-Language Models (VLMs), and new agentic AI systems, like LangChain and GraphChain. Nevertheless, this agentic environment increases the probability of the occurrence of multimodal prompt injection (PI) attacks, in which concealed or malicious instructions carried in text, pictures, metadata, or agent-to-agent messages may spread throughout the graph and lead to unintended behavior, a breach of policy, or corruption of state. In order to mitigate these risks, this paper suggests a Cross-Agent Multimodal Provenanc- Aware Defense Framework whereby all the prompts, either user-generated or produced by upstream agents, are sanitized and all the outputs generated by an LLM are verified independently before being sent to downstream nodes. This framework contains a Text sanitizer agent, visual sanitizer agent, and output validator agent all coordinated by a provenance ledger, which keeps metadata of modality, source, and trust level throughout the entire agent network. This architecture makes sure that agent-to-agent communication abides by clear trust frames such such that injected instructions are not propagated down LangChain or GraphChain-style-workflows. The experimental assessments show that multimodal injection detection accuracy is significantly enhanced, and the cross-agent trust leakage is minimized, as well as, agentic execution pathways become stable. The framework, which expands the concept of provenance tracking and validation to the multi-agent orchestration, enhances the establishment of secure, understandable and reliable agentic AI systems.

**arXiv ID:** 2512.23557
</details>

<details>
<summary><strong>BOAD: Discovering Hierarchical Software Engineering Agents via Bandit Optimization</strong> - Iris Xu, Guangtao Zeng, Zexue He, Charles Jin, Aldo Pareja, Dan Gutfreund, Chuang Gan, Zhang-Wei Hong - [[pdf]](https://arxiv.org/pdf/2512.23631)</summary>

**Abstract:** Large language models (LLMs) have shown strong reasoning and coding capabilities, yet they struggle to generalize to real-world software engineering (SWE) problems that are long-horizon and out of distribution. Existing systems often rely on a single agent to handle the entire workflow-interpreting issues, navigating large codebases, and implementing fixes-within one reasoning chain. Such monolithic designs force the model to retain irrelevant context, leading to spurious correlations and poor generalization. Motivated by how human engineers decompose complex problems, we propose structuring SWE agents as orchestrators coordinating specialized sub-agents for sub-tasks such as localization, editing, and validation. The challenge lies in discovering effective hierarchies automatically: as the number of sub-agents grows, the search space becomes combinatorial, and it is difficult to attribute credit to individual sub-agents within a team. We address these challenges by formulating hierarchy discovery as a multi-armed bandit (MAB) problem, where each arm represents a candidate sub-agent and the reward measures its helpfulness when collaborating with others. This framework, termed Bandit Optimization for Agent Design (BOAD), enables efficient exploration of sub-agent designs under limited evaluation budgets. On SWE-bench-Verified, BOAD outperforms single-agent and manually designed multi-agent systems. On SWE-bench-Live, featuring more recent and out-of-distribution issues, our 36B system ranks second on the leaderboard at the time of evaluation, surpassing larger models such as GPT-4 and Claude. These results demonstrate that automatically discovered hierarchical multi-agent systems significantly improve generalization on challenging long-horizon SWE tasks. Code is available at this https URL.

**arXiv ID:** 2512.23631
</details>

<details>
<summary><strong>MARPO: A Reflective Policy Optimization for Multi Agent Reinforcement Learning</strong> - Cuiling Wu, Yaozhong Gan, Junliang Xing, Ying Fu - [[pdf]](https://arxiv.org/pdf/2512.22832)</summary>

**Abstract:** We propose Multi Agent Reflective Policy Optimization (MARPO) to alleviate the issue of sample inefficiency in multi agent reinforcement learning. MARPO consists of two key components: a reflection mechanism that leverages subsequent trajectories to enhance sample efficiency, and an asymmetric clipping mechanism that is derived from the KL divergence and dynamically adjusts the clipping range to improve training stability. We evaluate MARPO in classic multi agent environments, where it consistently outperforms other methods.

**arXiv ID:** 2512.22832
</details>

<details>
<summary><strong>Assessing behaviour coverage in a multi-agent system simulation for autonomous vehicle testing</strong> - Manuel Franco-Vivo - [[pdf]](https://arxiv.org/pdf/2512.23445)</summary>

**Abstract:** As autonomous vehicle technology advances, ensuring the safety and reliability of these systems becomes paramount. Consequently, comprehensive testing methodologies are essential to evaluate the performance of autonomous vehicles in diverse and complex real-world scenarios. This study focuses on the behaviour coverage analysis of a multi-agent system simulation designed for autonomous vehicle testing, and provides a systematic approach to measure and assess behaviour coverage within the simulation environment. By defining a set of driving scenarios, and agent interactions, we evaluate the extent to which the simulation encompasses a broad range of behaviours relevant to autonomous driving.
Our findings highlight the importance of behaviour coverage in validating the effectiveness and robustness of autonomous vehicle systems. Through the analysis of behaviour coverage metrics and coverage-based testing, we identify key areas for improvement and optimization in the simulation framework. Thus, a Model Predictive Control (MPC) pedestrian agent is proposed, where its objective function is formulated to encourage \textit{interesting} tests while promoting a more realistic behaviour than other previously studied pedestrian agents. This research contributes to advancing the field of autonomous vehicle testing by providing insights into the comprehensive evaluation of system behaviour in simulated environments. The results offer valuable implications for enhancing the safety, reliability, and performance of autonomous vehicles through rigorous testing methodologies.

**arXiv ID:** 2512.23445
</details>

<details>
<summary><strong>Multi-Agent Framework for Threat Mitigation and Resilience in AI-Based Systems</strong> - Armstrong Foundjem, Lionel Nganyewou Tidjon, Leuson Da Silva, Foutse Khomh - [[pdf]](https://arxiv.org/pdf/2512.23132)</summary>

**Abstract:** Machine learning (ML) underpins foundation models in finance, healthcare, and critical infrastructure, making them targets for data poisoning, model extraction, prompt injection, automated jailbreaking, and preference-guided black-box attacks that exploit model comparisons. Larger models can be more vulnerable to introspection-driven jailbreaks and cross-modal manipulation. Traditional cybersecurity lacks ML-specific threat modeling for foundation, multimodal, and RAG systems. Objective: Characterize ML security risks by identifying dominant TTPs, vulnerabilities, and targeted lifecycle stages. Methods: We extract 93 threats from MITRE ATLAS (26), AI Incident Database (12), and literature (55), and analyze 854 GitHub/Python repositories. A multi-agent RAG system (ChatGPT-4o, temp 0.4) mines 300+ articles to build an ontology-driven threat graph linking TTPs, vulnerabilities, and stages. Results: We identify unreported threats including commercial LLM API model stealing, parameter memorization leakage, and preference-guided text-only jailbreaks. Dominant TTPs include MASTERKEY-style jailbreaking, federated poisoning, diffusion backdoors, and preference optimization leakage, mainly impacting pre-training and inference. Graph analysis reveals dense vulnerability clusters in libraries with poor patch propagation. Conclusion: Adaptive, ML-specific security frameworks, combining dependency hygiene, threat intelligence, and monitoring, are essential to mitigate supply-chain and inference risks across the ML lifecycle.

**arXiv ID:** 2512.23132
</details>

<details>
<summary><strong>Towards Global Optimality in Cooperative MARL with the Transformation And Distillation Framework</strong> - Jianing Ye, Chenghao Li, Yongqiang Dou, Jianhao Wang, Guangwen Yang, Chongjie Zhang - [[pdf]](https://arxiv.org/pdf/2207.11143)</summary>

**Abstract:** Decentralized execution is one core demand in multi-agent reinforcement learning (MARL). Recently, most popular MARL algorithms have adopted decentralized policies to enable decentralized execution, and use gradient descent as the optimizer. However, there is hardly any theoretical analysis of these algorithms taking the optimization method into consideration, and we find that various popular MARL algorithms with decentralized policies are suboptimal in toy tasks when gradient descent is chosen as their optimization method. In this paper, we theoretically analyze two common classes of algorithms with decentralized policies -- multi-agent policy gradient methods and value-decomposition methods, and prove their suboptimality when gradient descent is used. To address the suboptimality issue, we propose the Transformation And Distillation (TAD) framework, which reformulates a multi-agent MDP as a special single-agent MDP with a sequential structure and enables decentralized execution by distilling the learned policy on the derived "single-agent" MDP. The approach is a two-stage learning paradigm that addresses the optimization problem in cooperative MARL, providing optimality guarantee with decent execution performance. Empirically, we implement TAD-PPO based on PPO, which can theoretically perform optimal policy learning in the finite multi-agent MDPs and shows significant outperformance on a large set of cooperative multi-agent tasks, from matrix game, hallway task, to StarCraft II, and football game.

**arXiv ID:** 2207.11143
</details>

<details>
<summary><strong>QLLM: Do We Really Need a Mixing Network for Credit Assignment in Multi-Agent Reinforcement Learning?</strong> - Zhouyang Jiang, Bin Zhang, Yuanjun Li, Zhiwei Xu - [[pdf]](https://arxiv.org/pdf/2504.12961)</summary>

**Abstract:** Credit assignment has remained a fundamental challenge in multi-agent reinforcement learning (MARL). Previous studies have primarily addressed this issue through value decomposition methods under the centralized training with decentralized execution paradigm, where neural networks are utilized to approximate the nonlinear relationship between individual Q-values and the global Q-value. Although these approaches have achieved considerable success in various benchmark tasks, they still suffer from several limitations, including imprecise attribution of contributions, limited interpretability, and poor scalability in high-dimensional state spaces. To address these challenges, we propose a novel algorithm, QLLM, which facilitates the automatic construction of credit assignment functions using large language models (LLMs). Specifically, the concept of TFCAF is introduced, wherein the credit allocation process is represented as a direct and expressive nonlinear functional formulation. A custom-designed coder-evaluator framework is further employed to guide the generation and verification of executable code by LLMs, significantly mitigating issues such as hallucination and shallow reasoning during inference. Furthermore, an IGM-Gating Mechanism enables QLLM to flexibly enforce or relax the monotonicity constraint depending on task demands, covering both IGM-compliant and non-monotonic scenarios. Extensive experiments conducted on several standard MARL benchmarks demonstrate that the proposed method consistently outperforms existing state-of-the-art baselines. Moreover, QLLM exhibits strong generalization capability and maintains compatibility with a wide range of MARL algorithms that utilize mixing networks, positioning it as a promising and versatile solution for complex multi-agent scenarios. The code is available at this https URL.

**arXiv ID:** 2504.12961
</details>

<details>
<summary><strong>MTTR-A: Measuring Cognitive Recovery Latency in Multi-Agent Systems</strong> - Barak Or - [[pdf]](https://arxiv.org/pdf/2511.20663)</summary>

**Abstract:** Reliability in multi-agent systems (MAS) built on large language models is increasingly limited by cognitive failures rather than infrastructure faults. Existing observability tools describe failures but do not quantify how quickly distributed reasoning recovers once coherence is lost. We introduce MTTR-A (Mean Time-to-Recovery for Agentic Systems), a runtime reliability metric that measures cognitive recovery latency in MAS. MTTR-A adapts classical dependability theory to agentic orchestration, capturing the time required to detect reasoning drift and restore coherent operation. We further define complementary metrics, including MTBF and a normalized recovery ratio (NRR), and establish theoretical bounds linking recovery latency to long-run cognitive uptime. Using a LangGraph-based benchmark with simulated drift and reflex recovery, we empirically demonstrate measurable recovery behavior across multiple reflex strategies. This work establishes a quantitative foundation for runtime cognitive dependability in distributed agentic systems.

**arXiv ID:** 2511.20663
</details>

<details>
<summary><strong>Scaling Clinician-Grade Feature Generation from Clinical Notes with Multi-Agent Language Models</strong> - Jiayi Wang, Jacqueline Jil Vallon, Nikhil V. Kotha, Neil Panjwani, Xi Ling, Margaret Redfield, Sushmita Vij, Sandy Srinivas, John Leppert, Mark K. Buyyounouski, Mohsen Bayati - [[pdf]](https://arxiv.org/pdf/2508.01956)</summary>

**Abstract:** Developing accurate clinical prediction models is often bottlenecked by the difficulty of deriving meaningful structured features from unstructured EHR notes, a process that traditionally requires manual, unscalable clinical abstraction. In this study, we first established a rigorous patient-level Clinician Feature Generation (CFG) protocol, in which domain experts manually reviewed notes to define and extract nuanced features for a cohort of 147 patients with prostate cancer. As a high-fidelity ground truth, this labor-intensive process provided the blueprint for SNOW (Scalable Note-to-Outcome Workflow), a transparent multi-agent large language model (LLM) system designed to autonomously mimic the iterative reasoning and validation workflow of clinical experts. On 5-year cancer recurrence prediction, SNOW (AUC-ROC 0.767) achieved performance comparable to manual CFG (0.762) and outperformed structured baselines, clinician-guided LLM extraction, and six representational feature generation (RFG) approaches. Once configured, SNOW produced the full patient-level feature table in 12 hours with 5 hours of clinician oversight, reducing human expert effort by approximately 48-fold versus manual CFG. To test scalability where manual CFG is infeasible, we deployed SNOW on an external heart failure with preserved ejection fraction (HFpEF) cohort from MIMIC-IV (n=2,084); without task-specific tuning, SNOW generated prognostic features that outperformed baseline and RFG methods for 30-day (SNOW: 0.851) and 1-year (SNOW: 0.763) mortality prediction. These results demonstrate that a modular LLM agent-based system can scale expert-level feature generation from clinical notes, while enabling interpretable use of unstructured EHR text in outcome prediction and preserving generalizability across a variety of settings and conditions.

**arXiv ID:** 2508.01956
</details>

<details>
<summary><strong>Multi-agent Self-triage System with Medical Flowcharts</strong> - Yujia Liu, Sophia Yu, Hongyue Jin, Jessica Wen, Alexander Qian, Terrence Lee, Mattheus Ramsis, Gi Won Choi, Lianhui Qin, Xin Liu, Edward J. Wang - [[pdf]](https://arxiv.org/pdf/2511.12439)</summary>

**Abstract:** Online health resources and large language models (LLMs) are increasingly used as a first point of contact for medical decision-making, yet their reliability in healthcare remains limited by low accuracy, lack of transparency, and susceptibility to unverified information. We introduce a proof-of-concept conversational self-triage system that guides LLMs with 100 clinically validated flowcharts from the American Medical Association, providing a structured and auditable framework for patient decision support. The system leverages a multi-agent framework consisting of a retrieval agent, a decision agent, and a chat agent to identify the most relevant flowchart, interpret patient responses, and deliver personalized, patient-friendly recommendations, respectively. Performance was evaluated at scale using synthetic datasets of simulated conversations. The system achieved 95.29% top-3 accuracy in flowchart retrieval (N=2,000) and 99.10% accuracy in flowchart navigation across varied conversational styles and conditions (N=37,200). By combining the flexibility of free-text interaction with the rigor of standardized clinical protocols, this approach demonstrates the feasibility of transparent, accurate, and generalizable AI-assisted self-triage, with potential to support informed patient decision-making while improving healthcare resource utilization.

**arXiv ID:** 2511.12439
</details>

<details>
<summary><strong>AI4Reading: Chinese Audiobook Interpretation System Based on Multi-Agent Collaboration</strong> - Minjiang Huang, Jipeng Qiang, Yi Zhu, Chaowei Zhang, Xiangyu Zhao, Kui Yu - [[pdf]](https://arxiv.org/pdf/2512.23300)</summary>

**Abstract:** Audiobook interpretations are attracting increasing attention, as they provide accessible and in-depth analyses of books that offer readers practical insights and intellectual inspiration. However, their manual creation process remains time-consuming and resource-intensive. To address this challenge, we propose AI4Reading, a multi-agent collaboration system leveraging large language models (LLMs) and speech synthesis technology to generate podcast, like audiobook interpretations. The system is designed to meet three key objectives: accurate content preservation, enhanced comprehensibility, and a logical narrative structure. To achieve these goals, we develop a framework composed of 11 specialized agents,including topic analysts, case analysts, editors, a narrator, and proofreaders that work in concert to explore themes, extract real world cases, refine content organization, and synthesize natural spoken language. By comparing expert interpretations with our system's output, the results show that although AI4Reading still has a gap in speech generation quality, the generated interpretative scripts are simpler and more accurate.

**arXiv ID:** 2512.23300
</details>

<details>
<summary><strong>Close the Loop: Synthesizing Infinite Tool-Use Data via Multi-Agent Role-Playing</strong> - Yuwen Li, Wei Zhang, Zelong Huang, Mason Yang, Jiajun Wu, Shawn Guo, Huahao Hu, Lingyi Sun, Jian Yang, Mingjie Tang, Byran Dai - [[pdf]](https://arxiv.org/pdf/2512.23611)</summary>

**Abstract:** Enabling Large Language Models (LLMs) to reliably invoke external tools remains a critical bottleneck for autonomous agents. Existing approaches suffer from three fundamental challenges: expensive human annotation for high-quality trajectories, poor generalization to unseen tools, and quality ceilings inherent in single-model synthesis that perpetuate biases and coverage gaps. We introduce InfTool, a fully autonomous framework that breaks these barriers through self-evolving multi-agent synthesis. Given only raw API specifications, InfTool orchestrates three collaborative agents (User Simulator, Tool-Calling Assistant, and MCP Server) to generate diverse, verified trajectories spanning single-turn calls to complex multi-step workflows. The framework establishes a closed loop: synthesized data trains the model via Group Relative Policy Optimization (GRPO) with gated rewards, the improved model generates higher-quality data targeting capability gaps, and this cycle iterates without human intervention. Experiments on the Berkeley Function-Calling Leaderboard (BFCL) demonstrate that InfTool transforms a base 32B model from 19.8% to 70.9% accuracy (+258%), surpassing models 10x larger and rivaling Claude-Opus, and entirely from synthetic data without human annotation.

**arXiv ID:** 2512.23611
</details>

<details>
<summary><strong>AnalogSAGE: Self-evolving Analog Design Multi-Agents with Stratified Memory and Grounded Experience</strong> - Zining Wang, Jian Gao, Weimin Fu, Xiaolong Guo, Xuan Zhang - [[pdf]](https://arxiv.org/pdf/2512.22435)</summary>

**Abstract:** Analog circuit design remains a knowledge- and experience-intensive process that relies heavily on human intuition for topology generation and device parameter tuning. Existing LLM-based approaches typically depend on prompt-driven netlist generation or predefined topology templates, limiting their ability to satisfy complex specification requirements. We propose AnalogSAGE, an open-source self-evolving multi-agent framework that coordinates three-stage agent explorations through four stratified memory layers, enabling iterative refinement with simulation-grounded feedback. To support reproducibility and generality, we release the source code. Our benchmark spans ten specification-driven operational amplifier design problems of varying difficulty, enabling quantitative and cross-task comparison under identical conditions. Evaluated under the open-source SKY130 PDK with ngspice, AnalogSAGE achieves a 10$\times$ overall pass rate, a 48$\times$ Pass@1, and a 4$\times$ reduction in parameter search space compared with existing frameworks, demonstrating that stratified memory and grounded reasoning substantially enhance the reliability and autonomy of analog design automation in practice.

**arXiv ID:** 2512.22435
</details>

<details>
<summary><strong>Breaking Symmetry-Induced Degeneracy in Multi-Agent Ergodic Coverage via Stochastic Spectral Control</strong> - Kooktae Lee, Julian Martinez - [[pdf]](https://arxiv.org/pdf/2512.23158)</summary>

**Abstract:** Multi-agent ergodic coverage via Spectral Multiscale Coverage (SMC) provides a principled framework for driving a team of agents so that their collective time-averaged trajectories match a prescribed spatial distribution. While classical SMC has demonstrated empirical success, it can suffer from gradient cancellation, particularly when agents are initialized near symmetry points of the target distribution, leading to undesirable behaviors such as stalling or motion constrained along symmetry axes. In this work, we rigorously characterize the initial conditions and symmetry-induced invariant manifolds that give rise to such directional degeneracy in first-order agent dynamics. To address this, we introduce a stochastic perturbation combined with a contraction term and prove that the resulting dynamics ensure almost-sure escape from zero-gradient manifolds while maintaining mean-square boundedness of agent trajectories. Simulations on symmetric multi-modal reference distributions demonstrate that the proposed stochastic SMC effectively mitigates transient stalling and axis-constrained motion, while ensuring that all agent trajectories remain bounded within the domain.

**arXiv ID:** 2512.23158
</details>

</details>

<details open>
<summary><h2>Other Agent Research (8 papers)</h2></summary>

<details>
<summary><strong>Multi-AI Agent Framework Reveals the "Oxide Gatekeeper" in Aluminum Nanoparticle Oxidation</strong> - Yiming Lu, Tingyu Lu, Di Zhang, Lili Ye, Hao Li - [[pdf]](https://arxiv.org/pdf/2512.22529)</summary>

**Abstract:** Aluminum nanoparticles (ANPs) are among the most energy-dense solid fuels, yet the atomic mechanisms governing their transition from passivated particles to explosive reactants remain elusive. This stems from a fundamental computational bottleneck: ab initio methods offer quantum accuracy but are restricted to small spatiotemporal scales (< 500 atoms, picoseconds), while empirical force fields lack the reactive fidelity required for complex combustion environments. Herein, we bridge this gap by employing a "human-in-the-loop" closed-loop framework where self-auditing AI Agents validate the evolution of a machine learning potential (MLP). By acting as scientific sentinels that visualize hidden model artifacts for human decision-making, this collaborative cycle ensures quantum mechanical accuracy while exhibiting near-linear scalability to million-atom systems and accessing nanosecond timescales (energy RMSE: 1.2 meV/atom, force RMSE: 0.126 eV/Angstrom). Strikingly, our simulations reveal a temperature-regulated dual-mode oxidation mechanism: at moderate temperatures, the oxide shell acts as a dynamic "gatekeeper," regulating oxidation through a "breathing mode" of transient nanochannels; above a critical threshold, a "rupture mode" unleashes catastrophic shell failure and explosive combustion. Importantly, we resolve a decades-old controversy by demonstrating that aluminum cation outward diffusion, rather than oxygen transport, dominates mass transfer across all temperature regimes, with diffusion coefficients consistently exceeding those of oxygen by 2-3 orders of magnitude. These discoveries establish a unified atomic-scale framework for energetic nanomaterial design, enabling the precision engineering of ignition sensitivity and energy release rates through intelligent computational design.

**arXiv ID:** 2512.22529
</details>

<details>
<summary><strong>ReCollab: Retrieval-Augmented LLMs for Cooperative Ad-hoc Teammate Modeling</strong> - Conor Wallace, Umer Siddique, Yongcan Cao - [[pdf]](https://arxiv.org/pdf/2512.22129)</summary>

**Abstract:** Ad-hoc teamwork (AHT) requires agents to infer the behavior of previously unseen teammates and adapt their policy accordingly. Conventional approaches often rely on fixed probabilistic models or classifiers, which can be brittle under partial observability and limited interaction. Large language models (LLMs) offer a flexible alternative: by mapping short behavioral traces into high-level hypotheses, they can serve as world models over teammate behavior. We introduce \Collab, a language-based framework that classifies partner types using a behavior rubric derived from trajectory features, and extend it to \ReCollab, which incorporates retrieval-augmented generation (RAG) to stabilize inference with exemplar trajectories. In the cooperative Overcooked environment, \Collab effectively distinguishes teammate types, while \ReCollab consistently improves adaptation across layouts, achieving Pareto-optimal trade-offs between classification accuracy and episodic return. These findings demonstrate the potential of LLMs as behavioral world models for AHT and highlight the importance of retrieval grounding in challenging coordination settings.

**arXiv ID:** 2512.22129
</details>

<details>
<summary><strong>AI-Generated Code Is Not Reproducible (Yet): An Empirical Study of Dependency Gaps in LLM-Based Coding Agents</strong> - Bhanu Prakash Vangala, Ali Adibifar, Tanu Malik, Ashish Gehani - [[pdf]](https://arxiv.org/pdf/2512.22387)</summary>

**Abstract:** The rise of Large Language Models (LLMs) as coding agents promises to accelerate software development, but their impact on generated code reproducibility remains largely unexplored. This paper presents an empirical study investigating whether LLM-generated code can be executed successfully in a clean environment with only OS packages and using only the dependencies that the model specifies. We evaluate three state-of-the-art LLM coding agents (Claude Code, OpenAI Codex, and Gemini) across 300 projects generated from 100 standardized prompts in Python, JavaScript, and Java. We introduce a three-layer dependency framework (distinguishing between claimed, working, and runtime dependencies) to quantify execution reproducibility. Our results show that only 68.3% of projects execute out-of-the-box, with substantial variation across languages (Python 89.2%, Java 44.0%). We also find a 13.5 times average expansion from declared to actual runtime dependencies, revealing significant hidden dependencies.

**arXiv ID:** 2512.22387
</details>

<details>
<summary><strong>A Unified AI, Embedded, Simulation, and Mechanical Design Approach to an Autonomous Delivery Robot</strong> - Amro Gamar, Ahmed Abduljalil, Alargam Mohammed, Ali Elhenidy, Abeer Tawakol - [[pdf]](https://arxiv.org/pdf/2512.22408)</summary>

**Abstract:** This paper presents the development of a fully autonomous delivery robot integrating mechanical engineering, embedded systems, and artificial intelligence. The platform employs a heterogeneous computing architecture, with RPi 5 and ROS 2 handling AI-based perception and path planning, while ESP32 running FreeRTOS ensures real-time motor control. The mechanical design was optimized for payload capacity and mobility through precise motor selection and material engineering. Key technical challenges addressed include optimizing computationally intensive AI algorithms on a resource-constrained platform and implementing a low-latency, reliable communication link between the ROS 2 host and embedded controller. Results demonstrate deterministic, PID-based motor control through rigorous memory and task management, and enhanced system reliability via AWS IoT monitoring and a firmware-level motor shutdown failsafe. This work highlights a unified, multi-disciplinary methodology, resulting in a robust and operational autonomous delivery system capable of real-world deployment.

**arXiv ID:** 2512.22408
</details>

<details>
<summary><strong>CoAgent: Collaborative Planning and Consistency Agent for Coherent Video Generation</strong> - Qinglin Zeng, Kaitong Cai, Ruiqi Chen, Qinhan Lv, Keze Wang - [[pdf]](https://arxiv.org/pdf/2512.22536)</summary>

**Abstract:** Maintaining narrative coherence and visual consistency remains a central challenge in open-domain video generation. Existing text-to-video models often treat each shot independently, resulting in identity drift, scene inconsistency, and unstable temporal structure. We propose CoAgent, a collaborative and closed-loop framework for coherent video generation that formulates the process as a plan-synthesize-verify pipeline. Given a user prompt, style reference, and pacing constraints, a Storyboard Planner decomposes the input into structured shot-level plans with explicit entities, spatial relations, and temporal cues. A Global Context Manager maintains entity-level memory to preserve appearance and identity consistency across shots. Each shot is then generated by a Synthesis Module under the guidance of a Visual Consistency Controller, while a Verifier Agent evaluates intermediate results using vision-language reasoning and triggers selective regeneration when inconsistencies are detected. Finally, a pacing-aware editor refines temporal rhythm and transitions to match the desired narrative flow. Extensive experiments demonstrate that CoAgent significantly improves coherence, visual consistency, and narrative quality in long-form video generation.

**arXiv ID:** 2512.22536
</details>

<details>
<summary><strong>DECEPTICON: How Dark Patterns Manipulate Web Agents</strong> - Phil Cuvin, Hao Zhu, Diyi Yang - [[pdf]](https://arxiv.org/pdf/2512.22894)</summary>

**Abstract:** Deceptive UI designs, widely instantiated across the web and commonly known as dark patterns, manipulate users into performing actions misaligned with their goals. In this paper, we show that dark patterns are highly effective in steering agent trajectories, posing a significant risk to agent robustness. To quantify this risk, we introduce DECEPTICON, an environment for testing individual dark patterns in isolation. DECEPTICON includes 700 web navigation tasks with dark patterns -- 600 generated tasks and 100 real-world tasks, designed to measure instruction-following success and dark pattern effectiveness. Across state-of-the-art agents, we find dark patterns successfully steer agent trajectories towards malicious outcomes in over 70% of tested generated and real-world tasks -- compared to a human average of 31%. Moreover, we find that dark pattern effectiveness correlates positively with model size and test-time reasoning, making larger, more capable models more susceptible. Leading countermeasures against adversarial attacks, including in-context prompting and guardrail models, fail to consistently reduce the success rate of dark pattern interventions. Our findings reveal dark patterns as a latent and unmitigated risk to web agents, highlighting the urgent need for robust defenses against manipulative designs.

**arXiv ID:** 2512.22894
</details>

<details>
<summary><strong>Embodied Learning of Reward for Musculoskeletal Control with Vision Language Models</strong> - Saraswati Soedarmadji, Yunyue Wei, Chen Zhang, Yisong Yue, Yanan Sui - [[pdf]](https://arxiv.org/pdf/2512.23077)</summary>

**Abstract:** Discovering effective reward functions remains a fundamental challenge in motor control of high-dimensional musculoskeletal systems. While humans can describe movement goals explicitly such as "walking forward with an upright posture," the underlying control strategies that realize these goals are largely implicit, making it difficult to directly design rewards from high-level goals and natural language descriptions. We introduce Motion from Vision-Language Representation (MoVLR), a framework that leverages vision-language models (VLMs) to bridge the gap between goal specification and movement control. Rather than relying on handcrafted rewards, MoVLR iteratively explores the reward space through iterative interaction between control optimization and VLM feedback, aligning control policies with physically coordinated behaviors. Our approach transforms language and vision-based assessments into structured guidance for embodied learning, enabling the discovery and refinement of reward functions for high-dimensional musculoskeletal locomotion and manipulation. These results suggest that VLMs can effectively ground abstract motion descriptions in the implicit principles governing physiological motor control.

**arXiv ID:** 2512.23077
</details>

<details>
<summary><strong>Contingency Model-based Control (CMC) for Communicationless Cooperative Collision Avoidance in Robot Swarms</strong> - Georg Schildbach - [[pdf]](https://arxiv.org/pdf/2512.20391)</summary>

**Abstract:** Cooperative collision avoidance between robots in swarm operations remains an open challenge. Assuming a decentralized architecture, each robot is responsible for making its own control decisions, including motion planning. To this end, most existing approaches mostly rely some form of (wireless) communication between the agents of the swarm. In reality, however, communication is brittle. It may be affected by latency, further delays and packet losses, transmission faults, and is subject to adversarial attacks, such as jamming or spoofing. This paper proposes Contingency Model-based Control (CMC) as a communicationless alternative. It follows the implicit cooperation paradigm, under which the design of the robots is based on consensual (offline) rules, similar to traffic rules. They include the definition of a contingency trajectory for each robot, and a method for construction of mutual collision avoidance constraints. The setup is shown to guarantee the recursive feasibility and collision avoidance between all swarm members in closed-loop operation. Moreover, CMC naturally satisfies the Plug \& Play paradigm, i.e., for new robots entering the swarm. Two numerical examples demonstrate that the collision avoidance guarantee is intact and that the robot swarm operates smoothly under the CMC regime.

**arXiv ID:** 2512.20391
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>Embodied Robot Manipulation in the Era of Foundation Models: Planning and Learning Perspectives</strong> - Shuanghao Bai, Wenxuan Song, Jiayi Chen, Yuheng Ji, Zhide Zhong, Jin Yang, Han Zhao, Wanqi Zhou, Zhe Li, Pengxiang Ding, Cheng Chi, Chang Xu, Xiaolong Zheng, Donglin Wang, Haoang Li, Shanghang Zhang, Badong Chen - [[pdf]](https://arxiv.org/pdf/2512.22983)</summary>

**Abstract:** Recent advances in vision, language, and multimodal learning have substantially accelerated progress in robotic foundation models, with robot manipulation remaining a central and challenging problem. This survey examines robot manipulation from an algorithmic perspective and organizes recent learning-based approaches within a unified abstraction of high-level planning and low-level control. At the high level, we extend the classical notion of task planning to include reasoning over language, code, motion, affordances, and 3D representations, emphasizing their role in structured and long-horizon decision making. At the low level, we propose a training-paradigm-oriented taxonomy for learning-based control, organizing existing methods along input modeling, latent representation learning, and policy learning. Finally, we identify open challenges and prospective research directions related to scalability, data efficiency, multimodal physical interaction, and safety. Together, these analyses aim to clarify the design space of modern foundation models for robotic manipulation.

**arXiv ID:** 2512.22983
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (28 papers)</h2></summary>

<details>
<summary><strong>With Great Capabilities Come Great Responsibilities: Introducing the Agentic Risk & Capability Framework for Governing Agentic AI Systems</strong> - Shaun Khoo, Jessica Foo, Roy Ka-Wei Lee - [[pdf]](https://arxiv.org/pdf/2512.22211)</summary>

**Abstract:** Agentic AI systems present both significant opportunities and novel risks due to their capacity for autonomous action, encompassing tasks such as code execution, internet interaction, and file modification. This poses considerable challenges for effective organizational governance, particularly in comprehensively identifying, assessing, and mitigating diverse and evolving risks. To tackle this, we introduce the Agentic Risk \& Capability (ARC) Framework, a technical governance framework designed to help organizations identify, assess, and mitigate risks arising from agentic AI systems. The framework's core contributions are: (1) it develops a novel capability-centric perspective to analyze a wide range of agentic AI systems; (2) it distills three primary sources of risk intrinsic to agentic AI systems - components, design, and capabilities; (3) it establishes a clear nexus between each risk source, specific materialized risks, and corresponding technical controls; and (4) it provides a structured and practical approach to help organizations implement the framework. This framework provides a robust and adaptable methodology for organizations to navigate the complexities of agentic AI, enabling rapid and effective innovation while ensuring the safe, secure, and responsible deployment of agentic AI systems. Our framework is open-sourced \href{this https URL}{here}.

**arXiv ID:** 2512.22211
</details>

<details>
<summary><strong>Benchmark Success, Clinical Failure: When Reinforcement Learning Optimizes for Benchmarks, Not Patients</strong> - Armin Berger, Manuela Bergau, Helen Schneider, Saad Ahmad, Tom Anglim Lagones, Gianluca Brugnara, Martha Foltyn-Dumitru, Kai Schlamp, Philipp Vollmuth, Rafet Sifa - [[pdf]](https://arxiv.org/pdf/2512.23090)</summary>

**Abstract:** Recent Reinforcement Learning (RL) advances for Large Language Models (LLMs) have improved reasoning tasks, yet their resource-constrained application to medical imaging remains underexplored. We introduce ChexReason, a vision-language model trained via R1-style methodology (SFT followed by GRPO) using only 2,000 SFT samples, 1,000 RL samples, and a single A100 GPU. Evaluations on CheXpert and NIH benchmarks reveal a fundamental tension: GRPO recovers in-distribution performance (23% improvement on CheXpert, macro-F1 = 0.346) but degrades cross-dataset transferability (19% drop on NIH). This mirrors high-resource models like NV-Reason-CXR-3B, suggesting the issue stems from the RL paradigm rather than scale. We identify a generalization paradox where the SFT checkpoint uniquely improves on NIH before optimization, indicating teacher-guided reasoning captures more institution-agnostic features. Furthermore, cross-model comparisons show structured reasoning scaffolds benefit general-purpose VLMs but offer minimal gain for medically pre-trained models. Consequently, curated supervised fine-tuning may outperform aggressive RL for clinical deployment requiring robustness across diverse populations.

**arXiv ID:** 2512.23090
</details>

<details>
<summary><strong>Agentic Physical AI toward a Domain-Specific Foundation Model for Nuclear Reactor Control</strong> - Yoonpyo Lee, Kazuma Kobayashi, Sai Puppala, Sajedul Talukder, Seid Koric, Souvik Chakraborty, Syed Bahauddin Alam - [[pdf]](https://arxiv.org/pdf/2512.23292)</summary>

**Abstract:** The prevailing paradigm in AI for physical systems, scaling general-purpose foundation models toward universal multimodal reasoning, confronts a fundamental barrier at the control interface. Recent benchmarks show that even frontier vision-language models achieve only 50-53% accuracy on basic quantitative physics tasks, behaving as approximate guessers that preserve semantic plausibility while violating physical constraints. This input unfaithfulness is not a scaling deficiency but a structural limitation. Perception-centric architectures optimize parameter-space imitation, whereas safety-critical control demands outcome-space guarantees over executed actions. Here, we present a fundamentally different pathway toward domain-specific foundation models by introducing compact language models operating as Agentic Physical AI, in which policy optimization is driven by physics-based validation rather than perceptual inference. We train a 360-million-parameter model on synthetic reactor control scenarios, scaling the dataset from 10^3 to 10^5 examples. This induces a sharp phase transition absent in general-purpose models. Small-scale systems exhibit high-variance imitation with catastrophic tail risk, while large-scale models undergo variance collapse exceeding 500x reduction, stabilizing execution-level behavior. Despite balanced exposure to four actuation families, the model autonomously rejects approximately 70% of the training distribution and concentrates 95% of runtime execution on a single-bank strategy. Learned representations transfer across distinct physics and continuous input modalities without architectural modification.

**arXiv ID:** 2512.23292
</details>

<details>
<summary><strong>Replay Failures as Successes: Sample-Efficient Reinforcement Learning for Instruction Following</strong> - Kongcheng Zhang, Qi Yao, Shunyu Liu, Wenjian Zhang, Min Cen, Yang Zhou, Wenkai Fang, Yiru Zhao, Baisheng Lai, Mingli Song - [[pdf]](https://arxiv.org/pdf/2512.23457)</summary>

**Abstract:** Reinforcement Learning (RL) has shown promise for aligning Large Language Models (LLMs) to follow instructions with various constraints. Despite the encouraging results, RL improvement inevitably relies on sampling successful, high-quality responses; however, the initial model often struggles to generate responses that satisfy all constraints due to its limited capabilities, yielding sparse or indistinguishable rewards that impede learning. In this work, we propose Hindsight instruction Replay (HiR), a novel sample-efficient RL framework for complex instruction following tasks, which employs a select-then-rewrite strategy to replay failed attempts as successes based on the constraints that have been satisfied in hindsight. We perform RL on these replayed samples as well as the original ones, theoretically framing the objective as dual-preference learning at both the instruction- and response-level to enable efficient optimization using only a binary reward signal. Extensive experiments demonstrate that the proposed HiR yields promising results across different instruction following tasks, while requiring less computational budget. Our code and dataset is available at this https URL.

**arXiv ID:** 2512.23457
</details>

<details>
<summary><strong>SoDA: An Efficient Interaction Paradigm for the Agentic Web</strong> - Zicai Cui, Zhouyuan Jian, Weiwen Liu, Weinan Zhang - [[pdf]](https://arxiv.org/pdf/2512.22135)</summary>

**Abstract:** As the internet evolves from the mobile App-dominated Attention Economy to the Intent-Interconnection of the Agentic Web era, existing interaction modes fail to address the escalating challenges of data lock-in and cognitive overload. Addressing this, we defines a future-oriented user sovereignty interaction paradigm, aiming to realize a fundamental shift from killing time to saving time. Specifically, we argue that decoupling memory from application logic eliminates the structural basis of data lock-in, while shifting from explicit manual instruction to implicit intent alignment resolves cognitive overload by offloading execution complexity. This paradigm is implemented via the Sovereign Digital Avatar (SoDA), which employs an orthogonal decoupling design of storage, computation, and interaction. This establishes the architectural principle of data as a persistent asset, model as a transient tool, fundamentally breaking the platform monopoly on user memory. To support the operation of this new paradigm in zero-trust environments, we design an Intent-Permission Handshake Mechanism based on A2A protocols, utilizing dual-factor (Sensitivity Coefficient and Strictness Parameter) adaptive routing to achieve active risk governance. Empirical evaluation with a high-fidelity simulation environment indicates that this paradigm reduces token consumption by approximately 27-35\% during cross-platform service migration and complex task execution. Furthermore, in the orchestration of multi-modal complex tasks, it reduces user cognitive load by 72\% compared to standard Retrieval-Augmented Generation (RAG) architectures, by 88\% relative to manual workflows, while significantly boosting the Information Signal-to-Noise Ratio (SNR). These results demonstrate that the SoDA is the essential interaction infrastructure for building an efficient, low-friction, and decentralized Agentic Web.

**arXiv ID:** 2512.22135
</details>

<details>
<summary><strong>Scalable Cloud-Native Architectures for Intelligent PMU Data Processing</strong> - Nachiappan Chockalingam, Akshay Deshpande, Lokesh Butra, Ram Sekhar Bodala, Nitin Saksena, Adithya Parthasarathy, Balakrishna Pothineni, Akash Kumar Agarwal - [[pdf]](https://arxiv.org/pdf/2512.22231)</summary>

**Abstract:** Phasor Measurement Units (PMUs) generate high-frequency, time-synchronized data essential for real-time power grid monitoring, yet the growing scale of PMU deployments creates significant challenges in latency, scalability, and reliability. Conventional centralized processing architectures are increasingly unable to handle the volume and velocity of PMU data, particularly in modern grids with dynamic operating conditions. This paper presents a scalable cloud-native architecture for intelligent PMU data processing that integrates artificial intelligence with edge and cloud computing. The proposed framework employs distributed stream processing, containerized microservices, and elastic resource orchestration to enable low-latency ingestion, real-time anomaly detection, and advanced analytics. Machine learning models for time-series analysis are incorporated to enhance grid observability and predictive capabilities. Analytical models are developed to evaluate system latency, throughput, and reliability, showing that the architecture can achieve sub-second response times while scaling to large PMU deployments. Security and privacy mechanisms are embedded to support deployment in critical infrastructure environments. The proposed approach provides a robust and flexible foundation for next-generation smart grid analytics.

**arXiv ID:** 2512.22231
</details>

<details>
<summary><strong>Agentic Software Issue Resolution with Large Language Models: A Survey</strong> - Zhonghao Jiang, David Lo, Zhongxin Liu - [[pdf]](https://arxiv.org/pdf/2512.22256)</summary>

**Abstract:** Software issue resolution aims to address real-world issues in software repositories (e.g., bug fixing and efficiency optimization) based on natural language descriptions provided by users, representing a key aspect of software maintenance. With the rapid development of large language models (LLMs) in reasoning and generative capabilities, LLM-based approaches have made significant progress in automated software issue resolution. However, real-world software issue resolution is inherently complex and requires long-horizon reasoning, iterative exploration, and feedback-driven decision making, which demand agentic capabilities beyond conventional single-step approaches. Recently, LLM-based agentic systems have become mainstream for software issue resolution. Advancements in agentic software issue resolution not only greatly enhance software maintenance efficiency and quality but also provide a realistic environment for validating agentic systems' reasoning, planning, and execution capabilities, bridging artificial intelligence and software engineering.
This work presents a systematic survey of 126 recent studies at the forefront of LLM-based agentic software issue resolution research. It outlines the general workflow of the task and establishes a taxonomy across three dimensions: benchmarks, techniques, and empirical studies. Furthermore, it highlights how the emergence of agentic reinforcement learning has brought a paradigm shift in the design and training of agentic systems for software engineering. Finally, it summarizes key challenges and outlines promising directions for future research.

**arXiv ID:** 2512.22256
</details>

<details>
<summary><strong>RollArt: Scaling Agentic RL Training via Disaggregated Infrastructure</strong> - Wei Gao, Yuheng Zhao, Tianyuan Wu, Shaopan Xiong, Weixun Wang, Dakai An, Lunxi Cao, Dilxat Muhtar, Zichen Liu, Haizhou Zhao, Ju Huang, Siran Yang, Yongbin Li, Wenbo Su, Jiamang Wang, Lin Qu, Bo Zheng, Wei Wang - [[pdf]](https://arxiv.org/pdf/2512.22560)</summary>

**Abstract:** Agentic Reinforcement Learning (RL) enables Large Language Models (LLMs) to perform autonomous decision-making and long-term planning. Unlike standard LLM post-training, agentic RL workloads are highly heterogeneous, combining compute-intensive prefill phases, bandwidth-bound decoding, and stateful, CPU-heavy environment simulations. We argue that efficient agentic RL training requires disaggregated infrastructure to leverage specialized, best-fit hardware. However, naive disaggregation introduces substantial synchronization overhead and resource underutilization due to the complex dependencies between stages.
We present RollArc, a distributed system designed to maximize throughput for multi-task agentic RL on disaggregated infrastructure. RollArc is built on three core principles: (1) hardware-affinity workload mapping, which routes compute-bound and bandwidth-bound tasks to bestfit GPU devices, (2) fine-grained asynchrony, which manages execution at the trajectory level to mitigate resource bubbles, and (3) statefulness-aware computation, which offloads stateless components (e.g., reward models) to serverless infrastructure for elastic scaling. Our results demonstrate that RollArc effectively improves training throughput and achieves 1.35-2.05\(\times\) end-to-end training time reduction compared to monolithic and synchronous baselines. We also evaluate RollArc by training a hundreds-of-billions-parameter MoE model for Qoder product on an Alibaba cluster with more than 3,000 GPUs, further demonstrating RollArc scalability and robustness. The code is available at this https URL.

**arXiv ID:** 2512.22560
</details>

<details>
<summary><strong>FoldAct: Efficient and Stable Context Folding for Long-Horizon Search Agents</strong> - Jiaqi Shao, Yufeng Miao, Wei Zhang, Bing Luo - [[pdf]](https://arxiv.org/pdf/2512.22733)</summary>

**Abstract:** Long-horizon reinforcement learning (RL) for large language models faces critical scalability challenges from unbounded context growth, leading to context folding methods that compress interaction history during task execution. However, existing approaches treat summary actions as standard actions, overlooking that summaries fundamentally modify the agent's future observation space, creating a policy-dependent, non-stationary observation distribution that violates core RL assumptions. This introduces three fundamental challenges: (1) gradient dilution where summary tokens receive insufficient training signal, (2) self-conditioning where policy updates change summary distributions, creating a vicious cycle of training collapse, and (3) computational cost from processing unique contexts at each turn. We introduce \textbf{FoldAct}\footnote{this https URL}, a framework that explicitly addresses these challenges through three key innovations: separated loss computation for independent gradient signals on summary and action tokens, full context consistency loss to reduce distribution shift, and selective segment training to reduce computational cost. Our method enables stable training of long-horizon search agents with context folding, addressing the non-stationary observation problem while improving training efficiency with 5.19$\times$ speedup.

**arXiv ID:** 2512.22733
</details>

<details>
<summary><strong>AutoForge: Automated Environment Synthesis for Agentic Reinforcement Learning</strong> - Shihao Cai, Runnan Fang, Jialong Wu, Baixuan Li, Xinyu Wang, Yong Jiang, Liangcai Su, Liwen Zhang, Wenbiao Yin, Zhen Zhang, Fuli Feng, Pengjun Xie, Xiaobin Wang - [[pdf]](https://arxiv.org/pdf/2512.22857)</summary>

**Abstract:** Conducting reinforcement learning (RL) in simulated environments offers a cost-effective and highly scalable way to enhance language-based agents. However, previous work has been limited to semi-automated environment synthesis or tasks lacking sufficient difficulty, offering little breadth or depth. In addition, the instability of simulated users integrated into these environments, along with the heterogeneity across simulated environments, poses further challenges for agentic RL. In this work, we propose: (1) a unified pipeline for automated and scalable synthesis of simulated environments associated with high-difficulty but easily verifiable tasks; and (2) an environment level RL algorithm that not only effectively mitigates user instability but also performs advantage estimation at the environment level, thereby improving training efficiency and stability. Comprehensive evaluations on agentic benchmarks, including tau-bench, tau2-Bench, and VitaBench, validate the effectiveness of our proposed method. Further in-depth analyses underscore its out-of-domain generalization.

**arXiv ID:** 2512.22857
</details>

<details>
<summary><strong>Sat-EnQ: Satisficing Ensembles of Weak Q-Learners for Reliable and Compute-Efficient Reinforcement Learning</strong> - Ünver Çiftçi - [[pdf]](https://arxiv.org/pdf/2512.22910)</summary>

**Abstract:** Deep Q-learning algorithms remain notoriously unstable, especially during early training when the maximization operator amplifies estimation errors. Inspired by bounded rationality theory and developmental learning, we introduce Sat-EnQ, a two-phase framework that first learns to be ``good enough'' before optimizing aggressively. In Phase 1, we train an ensemble of lightweight Q-networks under a satisficing objective that limits early value growth using a dynamic baseline, producing diverse, low-variance estimates while avoiding catastrophic overestimation. In Phase 2, the ensemble is distilled into a larger network and fine-tuned with standard Double DQN. We prove theoretically that satisficing induces bounded updates and cannot increase target variance, with a corollary quantifying conditions for substantial reduction. Empirically, Sat-EnQ achieves 3.8x variance reduction, eliminates catastrophic failures (0% vs 50% for DQN), maintains 79% performance under environmental noise}, and requires 2.5x less compute than bootstrapped ensembles. Our results highlight a principled path toward robust reinforcement learning by embracing satisficing before optimization.

**arXiv ID:** 2512.22910
</details>

<details>
<summary><strong>Trust Region Masking for Long-Horizon LLM Reinforcement Learning</strong> - Yingru Li, Jiacai Liu, Jiawei Xu, Yuxuan Tong, Ziniu Li, Baoxiang Wang - [[pdf]](https://arxiv.org/pdf/2512.23075)</summary>

**Abstract:** Policy gradient methods for large language models optimize a surrogate objective computed from samples of a rollout policy $\pi_{\text{roll}}$. When $\pi_{\text{roll}} \ne \pi_{\theta}$, there is approximation error between the surrogate and the true objective. Prior work has shown that this off-policy mismatch is unavoidable in modern LLM-RL due to implementation divergence, mixture-of-experts routing discontinuities, and distributed training staleness. Classical trust region bounds on the resulting error scale as $O(T^2)$ with sequence length $T$, rendering them vacuous for long-horizon tasks. We derive two tighter bounds: a Pinsker-Marginal bound scaling as $O(T^{3/2})$ and a Mixed bound scaling as $O(T)$. Crucially, both bounds depend on $D_{kl}^{tok,max}$ -- the maximum token-level KL divergence across all positions in a sequence. This is inherently a sequence-level quantity: it requires examining the entire trajectory to compute, and therefore cannot be controlled by token-independent methods like PPO clipping. We propose Trust Region Masking (TRM), which excludes entire sequences from gradient computation if any token violates the trust region, providing the first non-vacuous monotonic improvement guarantees for long-horizon LLM-RL.

**arXiv ID:** 2512.23075
</details>

<details>
<summary><strong>Taming the Tail: Stable LLM Reinforcement Learning via Dynamic Vocabulary Pruning</strong> - Yingru Li, Jiawei Xu, Jiacai Liu, Yuxuan Tong, Ziniu Li, Tianle Cai, Ge Zhang, Qian Liu, Baoxiang Wang - [[pdf]](https://arxiv.org/pdf/2512.23087)</summary>

**Abstract:** Reinforcement learning for large language models (LLMs) faces a fundamental tension: high-throughput inference engines and numerically-precise training systems produce different probability distributions from the same parameters, creating a training-inference mismatch. We prove this mismatch has an asymmetric effect: the bound on log-probability mismatch scales as $(1-p)$ where $p$ is the token probability. For high-probability tokens, this bound vanishes, contributing negligibly to sequence-level mismatch. For low-probability tokens in the tail, the bound remains large, and moreover, when sampled, these tokens exhibit systematically biased mismatches that accumulate over sequences, destabilizing gradient estimation. Rather than applying post-hoc corrections, we propose constraining the RL objective to a dynamically-pruned ``safe'' vocabulary that excludes the extreme tail. By pruning such tokens, we trade large, systematically biased mismatches for a small, bounded optimization bias. Empirically, our method achieves stable training; theoretically, we bound the optimization bias introduced by vocabulary pruning.

**arXiv ID:** 2512.23087
</details>

<details>
<summary><strong>AGRO-SQL: Agentic Group-Relative Optimization with High-Fidelity Data Synthesis</strong> - Cehua Yang, Dongyu Xiao, Junming Lin, Yuyang Song, Hanxu Yan, Shawn Guo, Wei Zhang, Jian Yang, Mingjie Tang, Bryan Dai - [[pdf]](https://arxiv.org/pdf/2512.23366)</summary>

**Abstract:** The advancement of Text-to-SQL systems is currently hindered by the scarcity of high-quality training data and the limited reasoning capabilities of models in complex scenarios. In this paper, we propose a holistic framework that addresses these issues through a dual-centric approach. From a Data-Centric perspective, we construct an iterative data factory that synthesizes RL-ready data characterized by high correctness and precise semantic-logic alignment, ensured by strict verification. From a Model-Centric perspective, we introduce a novel Agentic Reinforcement Learning framework. This framework employs a Diversity-Aware Cold Start stage to initialize a robust policy, followed by Group Relative Policy Optimization (GRPO) to refine the agent's reasoning via environmental feedback. Extensive experiments on BIRD and Spider benchmarks demonstrate that our synergistic approach achieves state-of-the-art performance among single-model methods.

**arXiv ID:** 2512.23366
</details>

<details>
<summary><strong>Alpha-R1: Alpha Screening with LLM Reasoning via Reinforcement Learning</strong> - Zuoyou Jiang, Li Zhao, Rui Sun, Ruohan Sun, Zhongjian Li, Jing Li, Daxin Jiang, Zuo Bai, Cheng Hua - [[pdf]](https://arxiv.org/pdf/2512.23515)</summary>

**Abstract:** Signal decay and regime shifts pose recurring challenges for data-driven investment strategies in non-stationary markets. Conventional time-series and machine learning approaches, which rely primarily on historical correlations, often struggle to generalize when the economic environment changes. While large language models (LLMs) offer strong capabilities for processing unstructured information, their potential to support quantitative factor screening through explicit economic reasoning remains underexplored. Existing factor-based methods typically reduce alphas to numerical time series, overlooking the semantic rationale that determines when a factor is economically relevant. We propose Alpha-R1, an 8B-parameter reasoning model trained via reinforcement learning for context-aware alpha screening. Alpha-R1 reasons over factor logic and real-time news to evaluate alpha relevance under changing market conditions, selectively activating or deactivating factors based on contextual consistency. Empirical results across multiple asset pools show that Alpha-R1 consistently outperforms benchmark strategies and exhibits improved robustness to alpha decay. The full implementation and resources are available at this https URL.

**arXiv ID:** 2512.23515
</details>

<details>
<summary><strong>PathFound: An Agentic Multimodal Model Activating Evidence-seeking Pathological Diagnosis</strong> - Shengyi Hua, Jianfeng Wu, Tianle Shen, Kangzhe Hu, Zhongzhen Huang, Shujuan Ni, Zhihong Zhang, Yuan Li, Zhe Wang, Xiaofan Zhang - [[pdf]](https://arxiv.org/pdf/2512.23545)</summary>

**Abstract:** Recent pathological foundation models have substantially advanced visual representation learning and multimodal interaction. However, most models still rely on a static inference paradigm in which whole-slide images are processed once to produce predictions, without reassessment or targeted evidence acquisition under ambiguous diagnoses. This contrasts with clinical diagnostic workflows that refine hypotheses through repeated slide observations and further examination requests. We propose PathFound, an agentic multimodal model designed to support evidence-seeking inference in pathological diagnosis. PathFound integrates the power of pathological visual foundation models, vision-language models, and reasoning models trained with reinforcement learning to perform proactive information acquisition and diagnosis refinement by progressing through the initial diagnosis, evidence-seeking, and final decision stages. Across several large multimodal models, adopting this strategy consistently improves diagnostic accuracy, indicating the effectiveness of evidence-seeking workflows in computational pathology. Among these models, PathFound achieves state-of-the-art diagnostic performance across diverse clinical scenarios and demonstrates strong potential to discover subtle details, such as nuclear features and local invasions.

**arXiv ID:** 2512.23545
</details>

<details>
<summary><strong>No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping</strong> - Thanh-Long V. Le, Myeongho Jeon, Kim Vu, Viet Lai, Eunho Yang - [[pdf]](https://arxiv.org/pdf/2509.21880)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful framework for improving the reasoning abilities of Large Language Models (LLMs). However, current methods such as GRPO rely only on problems where the model responses to the same input differ in correctness, while ignoring those where all responses receive the same reward -- so-called zero-variance prompts. In this work, we argue that such prompts are not useless but can, in fact, provide meaningful feedback for policy optimization. To this end, we introduce RL with Zero-Variance Prompts (RL-ZVP), a novel algorithm that extract learning signals from zero-variance prompts. RL-ZVP directly rewards correctness and penalizes errors even without contrasting responses, modulating feedback with token-level characteristics to preserve informative, nuanced signals. Across six math reasoning benchmarks, RL-ZVP achieves significant improvements of up to 8.61 points in accuracy and 7.77 points in pass rate over GRPO, while consistently outperforming other baselines that filter out zero-variance prompts. These results highlight the untapped potential of learning from zero-variance prompts in RLVR.

**arXiv ID:** 2509.21880
</details>

<details>
<summary><strong>Prompt-R1: Collaborative Automatic Prompting Framework via End-to-end Reinforcement Learning</strong> - Wenjin Liu, Haoran Luo, Xueyuan Lin, Haoming Liu, Tiesunlong Shen, Jiapu Wang, Rui Mao, Erik Cambria - [[pdf]](https://arxiv.org/pdf/2511.01016)</summary>

**Abstract:** Recently, advanced large language models (LLMs) have emerged at an increasingly rapid pace. However, when faced with complex problems, most users are often unable to provide accurate and effective prompts to interact with LLMs, thus limiting the performance of LLMs. To address this challenge, we propose Prompt-R1, an end-to-end reinforcement learning framework that uses a small-scale LLM to collaborate with large-scale LLMs, replacing user interaction to solve problems better. This collaboration is cast as a multi-turn prompt interaction, where the small-scale LLM thinks and generates prompts, and the large-scale LLM performs complex reasoning. A dual-constrained reward is designed to optimize for correctness, generation quality, and reasoning accuracy. Prompt-R1 provides a plug-and-play framework that supports both inference and training with various large-scale LLMs. Experiments on multiple public datasets show that Prompt-R1 significantly outperforms baseline models across tasks. Our code is publicly available at this https URL.

**arXiv ID:** 2511.01016
</details>

<details>
<summary><strong>AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent</strong> - Haipeng Luo, Huawen Feng, Qingfeng Sun, Can Xu, Kai Zheng, Yufei Wang, Tao Yang, Han Hu, Yansong Tang, Di Wang - [[pdf]](https://arxiv.org/pdf/2512.20745)</summary>

**Abstract:** Large Reasoning Models (LRMs) like o3 and DeepSeek-R1 have achieved remarkable progress in natural language reasoning with long chain-of-thought. However, they remain computationally inefficient and struggle with accuracy when solving problems requiring complex mathematical operations. In this work, we present AgentMath, an agent framework that seamlessly integrates language models' reasoning capabilities with code interpreters' computational precision to efficiently tackle complex mathematical problems. Our approach introduces three key innovations: (1) An automated method that converts natural language chain-of-thought into structured tool-augmented trajectories, generating high-quality supervised fine-tuning (SFT) data to alleviate data scarcity; (2) A novel agentic reinforcement learning (RL) paradigm that dynamically interleaves natural language generation with real-time code execution. This enables models to autonomously learn optimal tool-use strategies through multi-round interactive feedback, while fostering emergent capabilities in code refinement and error correction; (3) An efficient training system incorporating innovative techniques, including request-level asynchronous rollout scheduling, agentic partial rollout, and prefix-aware weighted load balancing, achieving 4-5x speedup and making efficient RL training feasible on ultra-long sequences with scenarios with massive tool invocation. The evaluations show that AgentMath achieves state-of-the-art performance on challenging mathematical competition benchmarks including AIME24, AIME25, and HMMT25. Specifically, AgentMath-30B-A3B attains 90.6%, 86.4%, and 73.8% accuracy respectively, achieving advanced performance. The results validate the effectiveness of our approach and pave the way for building more sophisticated and scalable mathematical reasoning agents.

**arXiv ID:** 2512.20745
</details>

<details>
<summary><strong>TEACH: Temporal Variance-Driven Curriculum for Reinforcement Learning</strong> - Gaurav Chaudhary, Laxmidhar Behera - [[pdf]](https://arxiv.org/pdf/2512.22824)</summary>

**Abstract:** Reinforcement Learning (RL) has achieved significant success in solving single-goal tasks. However, uniform goal selection often results in sample inefficiency in multi-goal settings where agents must learn a universal goal-conditioned policy. Inspired by the adaptive and structured learning processes observed in biological systems, we propose a novel Student-Teacher learning paradigm with a Temporal Variance-Driven Curriculum to accelerate Goal-Conditioned RL. In this framework, the teacher module dynamically prioritizes goals with the highest temporal variance in the policy's confidence score, parameterized by the state-action value (Q) function. The teacher provides an adaptive and focused learning signal by targeting these high-uncertainty goals, fostering continual and efficient progress. We establish a theoretical connection between the temporal variance of Q-values and the evolution of the policy, providing insights into the method's underlying principles. Our approach is algorithm-agnostic and integrates seamlessly with existing RL frameworks. We demonstrate this through evaluation across 11 diverse robotic manipulation and maze navigation tasks. The results show consistent and notable improvements over state-of-the-art curriculum learning and goal-selection methods.

**arXiv ID:** 2512.22824
</details>

<details>
<summary><strong>Evaluating an Adaptive Multispectral Turret System for Autonomous Tracking Across Variable Illumination Conditions</strong> - Aahan Sachdeva, Dhanvinkumar Ganeshkumar, James E. Gallagher, Tyler Treat, Edward J. Oughton - [[pdf]](https://arxiv.org/pdf/2512.22263)</summary>

**Abstract:** Autonomous robotic platforms are playing a growing role across the emergency services sector, supporting missions such as search and rescue operations in disaster zones and reconnaissance. However, traditional red-green-blue (RGB) detection pipelines struggle in low-light environments, and thermal-based systems lack color and texture information. To overcome these limitations, we present an adaptive framework that fuses RGB and long-wave infrared (LWIR) video streams at multiple fusion ratios and dynamically selects the optimal detection model for each illumination condition. We trained 33 You Only Look Once (YOLO) models on over 22,000 annotated images spanning three light levels: no-light (<10 lux), dim-light (10-1000 lux), and full-light (>1000 lux). To integrate both modalities, fusion was performed by blending aligned RGB and LWIR frames at eleven ratios, from full RGB (100/0) to full LWIR (0/100) in 10% increments. Evaluation showed that the best full-light model (80/20 RGB-LWIR) and dim-light model (90/10 fusion) achieved 92.8% and 92.0% mean confidence; both significantly outperformed the YOLOv5 nano (YOLOv5n) and YOLOv11 nano (YOLOv11n) baselines. Under no-light conditions, the top 40/60 fusion reached 71.0%, exceeding baselines though not statistically significant. Adaptive RGB-LWIR fusion improved detection confidence and reliability across all illumination conditions, enhancing autonomous robotic vision performance.

**arXiv ID:** 2512.22263
</details>

<details>
<summary><strong>Beyond-Diagonal Reconfigurable Intelligent Surfaces for 6G Networks: Principles, Challenges, and Quantum Horizons</strong> - Abd Ullah Khan, Uman Khalid, Muhammad Tanveer, Trung Q. Duong, Hyundong Shin - [[pdf]](https://arxiv.org/pdf/2512.23400)</summary>

**Abstract:** A beyond-diagonal reconfigurable intelligent surface (BD-RIS) is an innovative type of reconfigurable intelligent surface (RIS) that has recently been proposed and is considered a revolutionary advancement in wave manipulation. Unlike the mutually disconnected arrangement of elements in traditional RISs, BD-RIS creates cost-effective and simple inter-element connections, allowing for greater freedom in configuring the amplitude and phase of impinging waves. However, there are numerous underlying challenges in realizing the advantages associated with BD-RIS, prompting the research community to actively investigate cutting-edge schemes and algorithms in this direction. Particularly, the passive beamforming design for BD-RIS under specific environmental conditions has become a major focus in this research area. In this article, we provide a systematic introduction to BD-RIS, elaborating on its functional principles concerning architectural design, promising advantages, and classification. Subsequently, we present recent advances and identify a series of challenges and opportunities. Additionally, we consider a specific case study where beamforming is designed using four different algorithms, and we analyze their performance with respect to sum rate and computation cost. To augment the beamforming capabilities in 6G BD-RIS with quantum enhancement, we analyze various hybrid quantum-classical machine learning (ML) models to improve beam prediction performance, employing real-world communication Scenario 8 from the DeepSense 6G dataset. Consequently, we derive useful insights about the practical implications of BD-RIS.

**arXiv ID:** 2512.23400
</details>

<details>
<summary><strong>A Human-Oriented Cooperative Driving Approach: Integrating Driving Intention, State, and Conflict</strong> - Qin Wang, Shanmin Pang, Jianwu Fang, Shengye Dong, Fuhao Liu, Jianru Xue, Chen Lv - [[pdf]](https://arxiv.org/pdf/2512.23220)</summary>

**Abstract:** Human-vehicle cooperative driving serves as a vital bridge to fully autonomous driving by improving driving flexibility and gradually building driver trust and acceptance of autonomous technology. To establish more natural and effective human-vehicle interaction, we propose a Human-Oriented Cooperative Driving (HOCD) approach that primarily minimizes human-machine conflict by prioritizing driver intention and state. In implementation, we take both tactical and operational levels into account to ensure seamless human-vehicle cooperation. At the tactical level, we design an intention-aware trajectory planning method, using intention consistency cost as the core metric to evaluate the trajectory and align it with driver intention. At the operational level, we develop a control authority allocation strategy based on reinforcement learning, optimizing the policy through a designed reward function to achieve consistency between driver state and authority allocation. The results of simulation and human-in-the-loop experiments demonstrate that our proposed approach not only aligns with driver intention in trajectory planning but also ensures a reasonable authority allocation. Compared to other cooperative driving approaches, the proposed HOCD approach significantly enhances driving performance and mitigates human-machine this http URL code is available at this https URL.

**arXiv ID:** 2512.23220
</details>

<details>
<summary><strong>Think, Act, Learn: A Framework for Autonomous Robotic Agents using Closed-Loop Large Language Models</strong> - Anjali R. Menon, Rohit K. Sharma, Priya Singh, Chengyu Wang, Aurora M. Ferreira, Mateja Novak - [[pdf]](https://arxiv.org/pdf/2507.19854)</summary>

**Abstract:** The integration of Large Language Models (LLMs) into robotics has unlocked unprecedented capabilities in high-level task planning. However, most current systems operate in an open-loop fashion, where LLMs act as one-shot planners, rendering them brittle and unable to adapt to unforeseen circumstances in dynamic physical environments. To overcome this limitation, this paper introduces the "Think, Act, Learn" (T-A-L) framework, a novel architecture that enables an embodied agent to autonomously learn and refine its policies through continuous interaction. Our framework establishes a closed-loop cycle where an LLM first "thinks" by decomposing high-level commands into actionable plans. The robot then "acts" by executing these plans while gathering rich, multimodal sensory feedback. Critically, the "learn" module processes this feedback to facilitate LLM-driven self-reflection, allowing the agent to perform causal analysis on its failures and generate corrective strategies. These insights are stored in an experiential memory to guide future planning cycles. We demonstrate through extensive experiments in both simulation and the real world that our T-A-L agent significantly outperforms baseline methods, including open-loop LLMs, Behavioral Cloning, and traditional Reinforcement Learning. Our framework achieves over a 97% success rate on complex, long-horizon tasks, converges to a stable policy in an average of just 9 trials, and exhibits remarkable generalization to unseen tasks. This work presents a significant step towards developing more robust, adaptive, and truly autonomous robotic agents.

**arXiv ID:** 2507.19854
</details>

<details>
<summary><strong>InDRiVE: Reward-Free World-Model Pretraining for Autonomous Driving via Latent Disagreement</strong> - Feeza Khan Khanzada, Jaerock Kwon - [[pdf]](https://arxiv.org/pdf/2512.18850)</summary>

**Abstract:** Model-based reinforcement learning (MBRL) can reduce interaction cost for autonomous driving by learning a predictive world model, but it typically still depends on task-specific rewards that are difficult to design and often brittle under distribution shift. This paper presents InDRiVE, a DreamerV3-style MBRL agent that performs reward-free pretraining in CARLA using only intrinsic motivation derived from latent ensemble disagreement. Disagreement acts as a proxy for epistemic uncertainty and drives the agent toward under-explored driving situations, while an imagination-based actor-critic learns a planner-free exploration policy directly from the learned world model. After intrinsic pretraining, we evaluate zero-shot transfer by freezing all parameters and deploying the pretrained exploration policy in unseen towns and routes. We then study few-shot adaptation by training a task policy with limited extrinsic feedback for downstream objectives (lane following and collision avoidance). Experiments in CARLA across towns, routes, and traffic densities show that disagreement-based pretraining yields stronger zero-shot robustness and robust few-shot collision avoidance under town shift and matched interaction budgets, supporting the use of intrinsic disagreement as a practical reward-free pretraining signal for reusable driving world models.

**arXiv ID:** 2512.18850
</details>

<details>
<summary><strong>Forecasting in Offline Reinforcement Learning for Non-stationary Environments</strong> - Suzan Ece Ada, Georg Martius, Emre Ugur, Erhan Oztop - [[pdf]](https://arxiv.org/pdf/2512.01987)</summary>

**Abstract:** Offline Reinforcement Learning (RL) provides a promising avenue for training policies from pre-collected datasets when gathering additional interaction data is infeasible. However, existing offline RL methods often assume stationarity or only consider synthetic perturbations at test time, assumptions that often fail in real-world scenarios characterized by abrupt, time-varying offsets. These offsets can lead to partial observability, causing agents to misperceive their true state and degrade performance. To overcome this challenge, we introduce Forecasting in Non-stationary Offline RL (FORL), a framework that unifies (i) conditional diffusion-based candidate state generation, trained without presupposing any specific pattern of future non-stationarity, and (ii) zero-shot time-series foundation models. FORL targets environments prone to unexpected, potentially non-Markovian offsets, requiring robust agent performance from the onset of each episode. Empirical evaluations on offline RL benchmarks, augmented with real-world time-series data to simulate realistic non-stationarity, demonstrate that FORL consistently improves performance compared to competitive baselines. By integrating zero-shot forecasting with the agent's experience, we aim to bridge the gap between offline RL and the complexities of real-world, non-stationary environments.

**arXiv ID:** 2512.01987
</details>

<details>
<summary><strong>ChatGraPhT: A Visual Conversation Interface for Multi-Path Reflection with Agentic LLM Support</strong> - Geoff Kimm, Linus Tan - [[pdf]](https://arxiv.org/pdf/2512.22790)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly used in complex knowledge work, yet linear transcript interfaces limit support for reflection. Schon's Reflective Practice distinguishes between reflection-in-action (during a task) and reflection-on-action (after a task), both benefiting from non-linear, revisitable representations of dialogue. ChatGraPhT is an interactive tool that shows dialogue as a visual map, allowing users to branch and merge ideas, edit past messages, and receive guidance that prompts deeper reflection. It supports non-linear, multi-path dialogue, while two agentic LLM assistants provide moment-to-moment and higher-level guidance. Our inquiry suggests that keeping the conversation structure visible, allowing branching and merging, and suggesting patterns or ways to combine ideas deepened user reflective engagement. Contributions are: (1) the design of a node-link, agentic LLM interface for reflective dialogue, and (2) transferable design knowledge on balancing structure and AI support to sustain reflection in complex, open-ended tasks.

**arXiv ID:** 2512.22790
</details>

<details>
<summary><strong>A Design Space for Intelligent Agents in Mixed-Initiative Visual Analytics</strong> - Tobias Stähle, Matthijs Jansen op de Haar, Sophia Boyer, Rita Sevastjanova, Arpit Narechania, Mennatallah El-Assady - [[pdf]](https://arxiv.org/pdf/2512.23372)</summary>

**Abstract:** Mixed-initiative visual analytics (VA) systems, where human and artificial intelligence (AI) agents collaborate as equal partners during analysis, represented a paradigm shift in human-computer interaction. With recent advances in AI, these systems have seen an increase in sophisticated software agents that have improved task planning, reasoning, and completion capabilities. However, while existing work characterizes agent interplay and communication strategies, there is a limited understanding of the overarching design principles for intelligent agents. Through a systematic review of 90 systems (and 207 unique agents), we propose a design space of intelligent agents comprising six dimensions that collectively characterize an agent's perception, environmental understanding, action capability, and communication strategies. We contribute a novel framework for researchers and designers to explore various design choices for new systems and to situate a system in the current landscape. We conclude with future research opportunities for intelligent agents in mixed-initiative VA systems.

**arXiv ID:** 2512.23372
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
