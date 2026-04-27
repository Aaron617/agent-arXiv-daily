# Agent arXiv Daily

**Last Updated:** 2026-04-27 04:38:43

**Total Papers:** 50

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
<summary><strong>MolClaw: An Autonomous Agent with Hierarchical Skills for Drug Molecule Evaluation, Screening, and Optimization</strong> - Lisheng Zhang, Lilong Wang, Xiangyu Sun, Wei Tang, Haoyang Su, Yuehui Qian, Qikui Yang, Qingsong Li, Zhenyu Tang, Haoran Sun, Yingnan Han, Yankai Jiang, Wenjie Lou, Bowen Zhou, Xiaosong Wang, Lei Bai, Zhengwei Xie - [[pdf]](https://arxiv.org/pdf/2604.21937)</summary>

**Abstract:** Computational drug discovery, particularly the complex workflows of drug molecule screening and optimization, requires orchestrating dozens of specialized tools in multi-step workflows, yet current AI agents struggle to maintain robust performance and consistently underperform in these high-complexity scenarios. Here we present MolClaw, an autonomous agent that leads drug molecule evaluation, screening, and optimization. It unifies over 30 specialized domain resources through a three-tier hierarchical skill architecture (70 skills in total) that facilitates agent long-term interaction at runtime: tool-level skills standardize atomic operations, workflow-level skills compose them into validated pipelines with quality check and reflection, and a discipline-level skill supplies scientific principles governing planning and verification across all scenarios in the field. Additionally, we introduce MolBench, a benchmark comprising molecular screening, optimization, and end-to-end discovery challenges spanning 8 to 50+ sequential tool calls. MolClaw achieves state-of-the-art performance across all metrics, and ablation studies confirm that gains concentrate on tasks that demand structured workflows while vanishing on those solvable with ad hoc scripting, establishing workflow orchestration competence as the primary capability bottleneck for AI-driven drug discovery.

**arXiv ID:** 2604.21937
</details>

<details>
<summary><strong>Memanto: Typed Semantic Memory with Information-Theoretic Retrieval for Long-Horizon Agents</strong> - Seyed Moein Abtahi, Rasa Rahnema, Hetkumar Patel, Neel Patel, Majid Fekri, Tara Khani - [[pdf]](https://arxiv.org/pdf/2604.22085)</summary>

**Abstract:** The transition from stateless language model inference to persistent, multi session autonomous agents has revealed memory to be a primary architectural bottleneck in the deployment of production grade agentic systems. Existing methodologies largely depend on hybrid semantic graph architectures, which impose substantial computational overhead during both ingestion and retrieval. These systems typically require large language model mediated entity extraction, explicit graph schema maintenance, and multi query retrieval pipelines. This paper introduces Memanto, a universal memory layer for agentic artificial intelligence that challenges the prevailing assumption that knowledge graph complexity is necessary to achieve high fidelity agent memory. Memanto integrates a typed semantic memory schema comprising thirteen predefined memory categories, an automated conflict resolution mechanism, and temporal versioning. These components are enabled by Moorcheh's Information Theoretic Search engine, a no indexing semantic database that provides deterministic retrieval within sub ninety millisecond latency while eliminating ingestion delay. Through systematic benchmarking on the LongMemEval and LoCoMo evaluation suites, Memanto achieves state of the art accuracy scores of 89.8 percent and 87.1 percent respectively. These results surpass all evaluated hybrid graph and vector based systems while requiring only a single retrieval query, incurring no ingestion cost, and maintaining substantially lower operational complexity. A five stage progressive ablation study is presented to quantify the contribution of each architectural component, followed by a discussion of the implications for scalable deployment of agentic memory systems.

**arXiv ID:** 2604.22085
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (7 papers)</h2></summary>

<details>
<summary><strong>An Artifact-based Agent Framework for Adaptive and Reproducible Medical Image Processing</strong> - Lianrui Zuo, Yihao Liu, Gaurav Rudravaram, Karthik Ramadass, Aravind R. Krishnan, Michael D. Phillips, Yelena G. Bodien, Mayur B. Patel, Paula Trujillo, Yency Forero Martinez, Stephen A. Deppen, Eric L. Grogan, Fabien Maldonado, Kevin McGann, Hudson M. Holmes, Laurie E. Cutting, Yuankai Huo, Bennett A. Landman - [[pdf]](https://arxiv.org/pdf/2604.21936)</summary>

**Abstract:** Medical imaging research is increasingly shifting from controlled benchmark evaluation toward real-world clinical deployment. In such settings, applying analytical methods extends beyond model design to require dataset-aware workflow configuration and provenance tracking. Two requirements therefore become central: \textbf{adaptability}, the ability to configure workflows according to dataset-specific conditions and evolving analytical goals; and \textbf{reproducibility}, the guarantee that all transformations and decisions are explicitly recorded and re-executable. Here, we present an artifact-based agent framework that introduces a semantic layer to augment medical image processing. The framework formalizes intermediate and final outputs through an artifact contract, enabling structured interrogation of workflow state and goal-conditioned assembly of configurations from a modular rule library. Execution is delegated to a workflow executor to preserve deterministic computational graph construction and provenance tracking, while the agent operates locally to comply with most privacy constraints. We evaluate the framework on real-world clinical CT and MRI cohorts, demonstrating adaptive configuration synthesis, deterministic reproducibility across repeated executions, and artifact-grounded semantic querying. These results show that adaptive workflow configuration can be achieved without compromising reproducibility in heterogeneous clinical environments.

**arXiv ID:** 2604.21936
</details>

<details>
<summary><strong>AgentSearchBench: A Benchmark for AI Agent Search in the Wild</strong> - Bin Wu, Arastun Mammadli, Xiaoyu Zhang, Emine Yilmaz - [[pdf]](https://arxiv.org/pdf/2604.22436)</summary>

**Abstract:** The rapid growth of AI agent ecosystems is transforming how complex tasks are delegated and executed, creating a new challenge of identifying suitable agents for a given task. Unlike traditional tools, agent capabilities are often compositional and execution-dependent, making them difficult to assess from textual descriptions alone. However, existing research and benchmarks typically assume well-specified functionalities, controlled candidate pools, or only executable task queries, leaving realistic agent search scenarios insufficiently studied. We introduce AgentSearchBench, a large-scale benchmark for agent search in the wild, built from nearly 10,000 real-world agents across multiple providers. The benchmark formalizes agent search as retrieval and reranking problems under both executable task queries and high-level task descriptions, and evaluates relevance using execution-grounded performance signals. Experiments reveal a consistent gap between semantic similarity and actual agent performance, exposing the limitations of description-based retrieval and reranking methods. We further show that lightweight behavioral signals, including execution-aware probing, can substantially improve ranking quality, highlighting the importance of incorporating execution signals into agent discovery. Our code is available at this https URL.

**arXiv ID:** 2604.22436
</details>

<details>
<summary><strong>AgentBound: Securing Execution Boundaries of AI Agents</strong> - Christoph Bühler, Matteo Biagiola, Luca Di Grazia, Guido Salvaneschi - [[pdf]](https://arxiv.org/pdf/2510.21236)</summary>

**Abstract:** Large Language Models (LLMs) have evolved into AI agents that interact with external tools and environments to perform complex tasks. The Model Context Protocol (MCP) has become the de facto standard for connecting agents with such resources, but security has lagged behind: thousands of MCP servers execute with unrestricted access to host systems, creating a broad attack surface. In this paper, we introduce AgentBound, the first access control framework for MCP servers. AgentBound combines a declarative policy mechanism, inspired by the Android permission model, with a policy enforcement engine that contains malicious behavior without requiring MCP server modifications. We build a dataset containing the 296 most popular MCP servers, and show that access control policies can be generated automatically from source code with 80.9% accuracy. We also show that AgentBound blocks the majority of security threats in several malicious MCP servers, and that the policy enforcement engine introduces negligible overhead. Our contributions provide developers and project managers with a foundation for securing MCP servers while maintaining productivity, enabling researchers and tool builders to explore new directions for declarative access control and MCP security.

**arXiv ID:** 2510.21236
</details>

<details>
<summary><strong>AdaptEvolve: Improving Efficiency of Evolutionary AI Agents through Adaptive Model Selection</strong> - Pretam Ray, Pratik Prabhanjan Brahma, Zicheng Liu, Emad Barsoum - [[pdf]](https://arxiv.org/pdf/2602.11931)</summary>

**Abstract:** Evolutionary agentic systems intensify the trade-off between computational efficiency and reasoning capability by repeatedly invoking large language models (LLMs) during inference. This setting raises a central question: how can an agent dynamically select an LLM that is sufficiently capable for the current generation step while remaining computationally efficient? While model cascades offer a practical mechanism for balancing this trade-off, existing routing strategies typically rely on static heuristics or external controllers and do not explicitly account for model uncertainty. We introduce AdaptEvolve: Adaptive LLM Selection for Multi-LLM Evolutionary Refinement within an evolutionary sequential refinement framework that leverages intrinsic generation confidence to estimate real-time solvability. Empirical results show that confidence-driven selection yields a favourable Pareto frontier, reducing total inference cost by an average of 37.9% across benchmarks while retaining 97.5% of the upper-bound accuracy of static large-model baselines. Our code is available at this https URL.

**arXiv ID:** 2602.11931
</details>

<details>
<summary><strong>Sovereign Agentic Loops: Decoupling AI Reasoning from Execution in Real-World Systems</strong> - Jun He, Deying Yu - [[pdf]](https://arxiv.org/pdf/2604.22136)</summary>

**Abstract:** Large language model (LLM) agents increasingly issue API calls that mutate real systems, yet many current architectures pass stochastic model outputs directly to execution layers. We argue that this coupling creates a safety risk because model correctness, context awareness, and alignment cannot be assumed at execution time. We introduce Sovereign Agentic Loops (SAL), a control-plane architecture in which models emit structured intents with justifications, and the control plane validates those intents against true system state and policy before execution. SAL combines an obfuscation membrane, which limits model access to identity-sensitive state, with a cryptographically linked Evidence Chain for auditability and replay. We formalize SAL and show that, under the stated assumptions, it provides policy-bounded execution, identity isolation, and deterministic replay. In an OpenKedge prototype for cloud infrastructure, SAL blocks 93% of unsafe intents at the policy layer, rejects the remaining 7% via consistency checks, prevents unsafe executions in our benchmark, and adds 12.4 ms median latency.

**arXiv ID:** 2604.22136
</details>

<details>
<summary><strong>Robust Localization for Autonomous Vehicles in Highway Scenes</strong> - Daqian Cheng, Xuchu Ding, Yujia Wu, Xiang Zhang, Lei Wang - [[pdf]](https://arxiv.org/pdf/2604.22040)</summary>

**Abstract:** Localization for autonomous vehicles on highways remains under-explored compared to urban roads, and state-of-the-art methods for urban scenes degrade when directly applied to highways. We identify key challenges including environment changes under information homogeneity, heavy occlusion, degraded GNSS signals, and stringent downstream requirements on accuracy and latency. We propose a robust localization system to address highway challenges, which uses a dual-likelihood LiDAR front end that decouples 3D geometric structures and 2D road-texture cues to handle environment changes; a Control-EKF further leverages steering and acceleration commands to reduce lag and improve closed-loop behavior. An automated offline mapping and ground-truth pipeline keep maps fresh at high cadence for optimal localization performance. To catalyze progress, we release a public dataset covering both urban roads and highways while focusing on representative challenging highway clips, totaling 163 km; benchmarking is standardized using product-oriented accuracy metrics and certified ground truth. Compared to Apollo and Autoware, our system performs similarly on urban roads but shows superior robustness on challenging highway scenarios. The system has been validated by more than one million kilometers of road testing.

**arXiv ID:** 2604.22040
</details>

<details>
<summary><strong>SANDO: Safe Autonomous Trajectory Planning for Dynamic Unknown Environments</strong> - Kota Kondo, Jesús Tordesillas, Jonathan P. How - [[pdf]](https://arxiv.org/pdf/2604.07599)</summary>

**Abstract:** SANDO is a safe trajectory planner for 3D dynamic unknown environments, where obstacle locations and motions are unknown a priori and a collision-free plan can become unsafe at any moment, requiring fast replanning. Existing soft-constraint planners are fast but cannot guarantee collision-free paths, while hard-constraint methods ensure safety at the cost of longer computation. SANDO addresses this trade-off through three contributions. First, a heat map-based A* global planner steers paths away from high-risk regions using soft costs, and a spatiotemporal safe flight corridor (STSFC) generator produces time-layered polytopes that inflate obstacles only by their worst-case reachable set at each time layer, rather than by the worst case over the entire horizon. Second, trajectory optimization is formulated as a Mixed-Integer Quadratic Program (MIQP) with hard collision-avoidance constraints, and a variable elimination technique reduces the number of decision variables, enabling fast computation. Third, a formal safety analysis establishes collision-free guarantees under explicit velocity-bound and estimation-error assumptions. Ablation studies show that variable elimination yields up to 7.4x speedup in optimization time, and that STSFCs are critical for feasibility in dense dynamic environments. Benchmark simulations against state-of-the-art methods across standardized static benchmarks, obstacle-rich static forests, and dynamic environments show that SANDO consistently achieves the highest success rate with no constraint violations across all difficulty levels; perception-only experiments without ground truth obstacle information confirm robust performance under realistic sensing. Hardware experiments on a UAV with fully onboard planning, perception, and localization demonstrate six safe flights in static environments and ten safe flights among dynamic obstacles.

**arXiv ID:** 2604.07599
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>Read the Paper, Write the Code: Agentic Reproduction of Social-Science Results</strong> - Benjamin Kohler, David Zollikofer, Johanna Einsiedler, Alexander Hoyle, Elliott Ash - [[pdf]](https://arxiv.org/pdf/2604.21965)</summary>

**Abstract:** Recent work has used LLM agents to reproduce empirical social science results with access to both the data and code. We broaden this scope by asking: Can they reproduce results given only a paper's methods description and original data? We develop an agentic reproduction system that extracts structured methods descriptions from papers, runs reimplementations under strict information isolation -- agents never see the original code, results, or paper -- and enables deterministic, cell-level comparison of reproduced outputs to the original results. An error attribution step traces discrepancies through the system chain to identify root causes. Evaluating four agent scaffolds and four LLMs on 48 papers with human-verified reproducibility, we find that agents can largely recover published results, but performance varies substantially between models, scaffolds, and papers. Root cause analysis reveals that failures stem both from agent errors and from underspecification in the papers themselves.

**arXiv ID:** 2604.21965
</details>

<details>
<summary><strong>Superminds Test: Actively Evaluating Collective Intelligence of Agent Society via Probing Agents</strong> - Xirui Li, Ming Li, Yunze Xiao, Ryan Wong, Dianqi Li, Timothy Baldwin, Tianyi Zhou - [[pdf]](https://arxiv.org/pdf/2604.22452)</summary>

**Abstract:** Collective intelligence refers to the ability of a group to achieve outcomes beyond what any individual member can accomplish alone. As large language model agents scale to populations of millions, a key question arises: Does collective intelligence emerge spontaneously from scale? We present the first empirical evaluation of this question in a large-scale autonomous agent society. Studying MoltBook, a platform hosting over two million agents, we introduce Superminds Test, a hierarchical framework that probes society-level intelligence using controlled Probing Agents across three tiers: joint reasoning, information synthesis, and basic interaction. Our experiments reveal a stark absence of collective intelligence. The society fails to outperform individual frontier models on complex reasoning tasks, rarely synthesizes distributed information, and often fails even trivial coordination tasks. Platform-wide analysis further shows that interactions remain shallow, with threads rarely extending beyond a single reply and most responses being generic or off-topic. These results suggest that collective intelligence does not emerge from scale alone. Instead, the dominant limitation of current agent societies is extremely sparse and shallow interaction, which prevents agents from exchanging information and building on each other's outputs.

**arXiv ID:** 2604.22452
</details>

<details>
<summary><strong>SOLAR-RL: Semi-Online Long-horizon Assignment Reinforcement Learning</strong> - Jichao Wang, Liuyang Bian, Yufeng Zhou, Han Xiao, Yue Pan, Guozhi Wang, Hao Wang, Zhaoxiong Wang, Yafei Wen, Xiaoxin Chen, Shuai Ren, Lingfang Zeng - [[pdf]](https://arxiv.org/pdf/2604.22558)</summary>

**Abstract:** As Multimodal Large Language Models (MLLMs) mature, GUI agents are evolving from static interactions to complex navigation. While Reinforcement Learning (RL) has emerged as a promising paradigm for training MLLM agents on dynamic GUI tasks, its effective application faces a dilemma. Standard Offline RL often relies on static step-level data, neglecting global trajectory semantics such as task completion and execution quality. Conversely, Online RL captures the long-term dynamics but suffers from high interaction costs and potential environmental instability. To bridge this gap, we propose SOLAR-RL (Semi-Online Long-horizon Assignment Reinforcement Learning). Instead of relying solely on expensive online interactions, our framework integrates global trajectory insights directly into the offline learning process. Specifically, we reconstruct diverse rollout candidates from static data, detect the first failure point using per-step validity signals, and retroactively assign dense step-level rewards with target-aligned shaping to reflect trajectory-level execution quality, effectively simulating online feedback without interaction costs. Extensive experiments demonstrate that SOLAR-RL significantly improves long-horizon task completion rates and robustness compared to strong baselines, offering a sample-efficient solution for autonomous GUI navigation.

**arXiv ID:** 2604.22558
</details>

<details>
<summary><strong>Chain-of-Memory: Lightweight Memory Construction with Dynamic Evolution for LLM Agents</strong> - Xiucheng Xu, Bingbing Xu, Xueyun Tian, Zihe Huang, Rongxin Chen, Yunfan Li, Huawei Shen - [[pdf]](https://arxiv.org/pdf/2601.14287)</summary>

**Abstract:** External memory systems are pivotal for enabling Large Language Model (LLM) agents to maintain persistent knowledge and perform long-horizon decision-making. Existing paradigms typically follow a two-stage process: computationally expensive memory construction (e.g., structuring data into graphs) followed by naive retrieval-augmented generation. However, our empirical analysis reveals two fundamental limitations: complex construction incurs high costs with marginal performance gains, and simple context concatenation fails to bridge the gap between retrieval recall and reasoning accuracy. To address these challenges, we propose CoM (Chain-of-Memory), a novel framework that advocates for a paradigm shift toward lightweight construction paired with sophisticated utilization. CoM introduces a Chain-of-Memory mechanism that organizes retrieved fragments into coherent inference paths through dynamic evolution, utilizing adaptive truncation to prune irrelevant noise. Extensive experiments on the LongMemEval and LoCoMo benchmarks demonstrate that CoM outperforms strong baselines with accuracy gains of 7.5%-10.4%, while drastically reducing computational overhead to approximately 2.7% of token consumption and 6.0% of latency compared to complex memory architectures.

**arXiv ID:** 2601.14287
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (15 papers)</h2></summary>

<details>
<summary><strong>From Skills to Talent: Organising Heterogeneous Agents as a Real-World Company</strong> - Zhengxu Yu, Yu Fu, Zhiyuan He, Yuxuan Huang, Lee Ka Yiu, Meng Fang, Weilin Luo, Jun Wang - [[pdf]](https://arxiv.org/pdf/2604.22446)</summary>

**Abstract:** Individual agent capabilities have advanced rapidly through modular skills and tool integrations, yet multi-agent systems remain constrained by fixed team structures, tightly coupled coordination logic, and session-bound learning. We argue that this reflects a deeper absence: a principled organisational layer that governs how a workforce of agents is assembled, governed, and improved over time, decoupled from what individual agents know. To fill this gap, we introduce \emph{OneManCompany (OMC)}, a framework that elevates multi-agent systems to the organisational level. OMC encapsulates skills, tools, and runtime configurations into portable agent identities called \emph{Talents}, orchestrated through typed organisational interfaces that abstract over heterogeneous backends. A community-driven \emph{Talent Market} enables on-demand recruitment, allowing the organisation to close capability gaps and reconfigure itself dynamically during execution. Organisational decision-making is operationalised through an \emph{Explore-Execute-Review} ($\text{E}^2$R) tree search, which unifies planning, execution, and evaluation in a single hierarchical loop: tasks are decomposed top-down into accountable units and execution outcomes are aggregated bottom-up to drive systematic review and refinement. This loop provides formal guarantees on termination and deadlock freedom while mirroring the feedback mechanisms of human enterprises. Together, these contributions transform multi-agent systems from static, pre-configured pipelines into self-organising and self-improving AI organisations capable of adapting to open-ended tasks across diverse domains. Empirical evaluation on PRDBench shows that OMC achieves an $84.67\%$ success rate, surpassing the state of the art by $15.48$ percentage points, with cross-domain case studies further demonstrating its generality.

**arXiv ID:** 2604.22446
</details>

<details>
<summary><strong>Agentic World Modeling: Foundations, Capabilities, Laws, and Beyond</strong> - Meng Chu, Xuan Billy Zhang, Kevin Qinghong Lin, Lingdong Kong, Jize Zhang, Teng Tu, Weijian Ma, Ziqi Huang, Senqiao Yang, Wei Huang, Yeying Jin, Zhefan Rao, Jinhui Ye, Xinyu Lin, Xichen Zhang, Qisheng Hu, Shuai Yang, Leyang Shen, Wei Chow, Yifei Dong, Fengyi Wu, Quanyu Long, Bin Xia, Shaozuo Yu, Mingkang Zhu, Wenhu Zhang, Jiehui Huang, Haokun Gui, Haoxuan Che, Long Chen, Qifeng Chen, Wenxuan Zhang, Wenya Wang, Xiaojuan Qi, Yang Deng, Yanwei Li, Mike Zheng Shou, Zhi-Qi Cheng, See-Kiong Ng, Ziwei Liu, Philip Torr, Jiaya Jia - [[pdf]](https://arxiv.org/pdf/2604.22748)</summary>

**Abstract:** As AI systems move from generating text to accomplishing goals through sustained interaction, the ability to model environment dynamics becomes a central bottleneck. Agents that manipulate objects, navigate software, coordinate with others, or design experiments require predictive environment models, yet the term world model carries different meanings across research communities. We introduce a "levels x laws" taxonomy organized along two axes. The first defines three capability levels: L1 Predictor, which learns one-step local transition operators; L2 Simulator, which composes them into multi-step, action-conditioned rollouts that respect domain laws; and L3 Evolver, which autonomously revises its own model when predictions fail against new evidence. The second identifies four governing-law regimes: physical, digital, social, and scientific. These regimes determine what constraints a world model must satisfy and where it is most likely to fail. Using this framework, we synthesize over 400 works and summarize more than 100 representative systems spanning model-based reinforcement learning, video generation, web and GUI agents, multi-agent social simulation, and AI-driven scientific discovery. We analyze methods, failure modes, and evaluation practices across level-regime pairs, propose decision-centric evaluation principles and a minimal reproducible evaluation package, and outline architectural guidance, open problems, and governance challenges. The resulting roadmap connects previously isolated communities and charts a path from passive next-step prediction toward world models that can simulate, and ultimately reshape, the environments in which agents operate.

**arXiv ID:** 2604.22748
</details>

<details>
<summary><strong>Reliable Self-Harm Risk Screening via Adaptive Multi-Agent LLM Systems</strong> - Meghana Karnam, Ananya Joshi - [[pdf]](https://arxiv.org/pdf/2604.22154)</summary>

**Abstract:** Emerging AI systems in behavioral health and psychiatry use multi-step or multi-agent LLM pipelines for tasks like assessing self-harm risk and screening for depression. However, common evaluation approaches, like LLM-as-a-judge, do not indicate when a decision is reliable or how errors may accumulate across multiple LLM judgements, limiting their suitability for safety-critical settings. We present a statistical framework for multi-agent pipelines structured as directed acyclic graphs (DAGs) that provides an alternative to heuristic voting with principled, adaptive decision-making. We model each agent as a stochastic categorical decision and introduce (1) tighter agent-level performance confidence bounds, (2) a bandit-based adaptive sampling strategy based on input difficulty, and (3) regret guarantees over the multi-agent system that shows logarithmic error growth when deployed. We evaluate our system on two labeled datasets in behavioral health : the AEGIS 2.0 behavioral health subset (N=161) and a stratified sample of SWMH Reddit posts (N=250). Empirically, our adaptive sampling strategy achieves the lowest false positive rate of any condition across both datasets, 0.095 on AEGIS 2.0 compared to 0.159 for single-agent models, reducing incorrect flagging of safe content by 40\% and still having similar false negative rates across all conditions. These results suggest that principled adaptive sampling offers a meaningful improvement in precision without reducing recall in this setting.

**arXiv ID:** 2604.22154
</details>

<details>
<summary><strong>Cost-Effective Communication: An Auction-based Method for Language Agent Interaction</strong> - Yijia Fan, Jusheng Zhang, Kaitong Cai, Jing Yang, Chengpei Tang, Jian Wang, Keze Wang - [[pdf]](https://arxiv.org/pdf/2511.13193)</summary>

**Abstract:** Multi-agent systems (MAS) built on large language models (LLMs) often suffer from inefficient "free-for-all" communication, leading to exponential token costs and low signal-to-noise ratios that hinder their practical deployment. We challenge the notion that more communication is always beneficial, hypothesizing instead that the core issue is the absence of resource rationality. We argue that "free" communication, by ignoring the principle of scarcity, inherently breeds inefficiency and unnecessary expenses. To address this, we introduce the Dynamic Auction-based Language Agent (DALA), a novel framework that treats communication bandwidth as a scarce and tradable resource. Specifically, our DALA regards inter-agent communication as a centralized auction, where agents learn to bid for the opportunity to speak based on the predicted value density of their messages. Thus, our DALA intrinsically encourages agents to produce concise, informative messages while filtering out low-value communication. Extensive and comprehensive experiments demonstrate that our economically-driven DALA achieves new state-of-the-art performance across seven challenging reasoning benchmarks, including 84.32% on MMLU and a 91.21% pass@1 rate on HumanEval. Note that this is accomplished with remarkable efficiency, i.e., our DALA uses only 6.25 million tokens, a fraction of the resources consumed by current state-of-the-art methods on GSM8K. Further analysis reveals that our DALA cultivates the emergent skill of strategic silence, effectively adapting its communication strategies from verbosity to silence in a dynamical manner via resource constraints. Our code and updates are available at this https URL.

**arXiv ID:** 2511.13193
</details>

<details>
<summary><strong>From Multi-Agent to Single-Agent: When Is Skill Distillation Beneficial?</strong> - Binyan Xu, Dong Fang, Haitao Li, Kehuan Zhang - [[pdf]](https://arxiv.org/pdf/2604.01608)</summary>

**Abstract:** Multi-agent systems (MAS) tackle complex tasks by distributing expertise, though this often comes at the cost of heavy coordination overhead, context fragmentation, and brittle phase ordering. Distilling a MAS into a single-agent skill can bypass these costs, but this conversion lacks a principled answer for when and what to distill. Instead, the empirical outcome is surprisingly inconsistent: skill lift ranges from a 28% improvement to a 2% degradation across metrics of the exact same task. In this work, we reveal that skill utility is governed not by the task, but by the evaluation metric. We introduce Metric Freedom (F), the first a priori predictor of skill utility. F measures the topological rigidity of a metric's scoring landscape by quantifying how output diversity couples with score variance via a Mantel test. Guided by F, we propose AdaSkill, a two-stage adaptive distillation framework. Stage 1 acts as a selective extraction mechanism, extracting tools and knowledge while discarding restrictive structures on "free" metrics to preserve exploration. Stage 2 applies iterative refinement selectively on free metrics, exploiting their forgiving scoring landscape to safely maximize remaining headroom. Evaluating across 4 tasks, 11 datasets, and 6 metrics, F strongly predicts skill utility (r=-0.85, p<0.0001). Strikingly, identical agent trajectories yield diametrically opposite skill lifts under rigid versus free metrics, demonstrating that skill utility is fundamentally a metric-level property. Driven by this signal, AdaSkill matches or exceeds the original MAS while reducing cost up to 8x and latency by up to 15x.

**arXiv ID:** 2604.01608
</details>

<details>
<summary><strong>EvoAgent: An Evolvable Agent Framework with Skill Learning and Multi-Agent Delegation</strong> - Aimin Zhang, Jiajing Guo, Fuwei Jia, Chen Lv, Boyu Wang, Fangzheng Li - [[pdf]](https://arxiv.org/pdf/2604.20133)</summary>

**Abstract:** This paper proposes EvoAgent - an evolvable large language model (LLM) agent framework that integrates structured skill learning with a hierarchical sub-agent delegation mechanism. EvoAgent models skills as multi-file structured capability units equipped with triggering mechanisms and evolutionary metadata, and enables continuous skill generation and optimization through a user-feedback-driven closed-loop process. In addition, by incorporating a three-stage skill matching strategy and a three-layer memory architecture, the framework supports dynamic task decomposition for complex problems and long-term capability accumulation. Experimental results based on real-world foreign trade scenarios demonstrate that, after integrating EvoAgent, GPT5.2 achieves significant improvements in professionalism, accuracy, and practical utility. Under a five-dimensional LLM-as-Judge evaluation protocol, the overall average score increases by approximately 28%. Further model transfer experiments indicate that the performance of an agent system depends not only on the intrinsic capabilities of the underlying model, but also on the degree of synergy between the model and the agent architecture.

**arXiv ID:** 2604.20133
</details>

<details>
<summary><strong>AdaFair-MARL: Enforcing Adaptive Fairness Constraints in Multi-Agent Reinforcement Learning</strong> - Promise Ekpo, Saesha Agarwal, Felix Grimm, Lekan Molu, Angelique Taylor - [[pdf]](https://arxiv.org/pdf/2511.14135)</summary>

**Abstract:** Fair workload enforcement in heterogeneous multi-agent systems that pursue shared objectives remains challenging. Fixed fairness penalties often introduce inefficiencies, training instability, and conflicting agent incentives. Reward-shaping approaches in fair Multi-Agent Reinforcement Learning (MARL) typically incorporate fairness through heuristic penalties or scalar reward modifications and often rely on post-hoc evaluation. However, these methods do not guarantee that a desired fairness level will be satisfied. To address this limitation, we propose the Adaptive Fairness Multi-Agent Reinforcement Learning (AdaFair-MARL) framework, which formulates workload fairness as an explicit constraint so that agents maintain balanced contributions while optimizing team performance. We present AdaFair-MARL, a constrained cooperative MARL framework whose core algorithmic component is a primal-dual update that enforces workload fairness via adaptive Lagrange multiplier updates. Grounding the framework in a cooperative Markov game, we derive the fairness constraint from Jain's Fairness Index (JFI) geometry and show that the resulting feasible set admits a second-order cone representation, enabling principled Lagrangian dual-ascent updates without manual penalty tuning. Experiments in a simulated hospital coordination environment (MARLHospital) demonstrate the effectiveness of AdaFair-MARL compared to reward-shaping and fixed-penalty fairness methods, improving workload balance while maintaining team performance. We found that AdaFair-MARL achieves nearly perfect constraint satisfaction (0.99-1.00) while significantly improving workload fairness compared to fixed-penalty baselines.

**arXiv ID:** 2511.14135
</details>

<details>
<summary><strong>When AI Agents Learn from Each Other: Insights from Emergent AI Agent Communities on OpenClaw for Human-AI Partnership in Education</strong> - Eason Chen, Ce Guan, Zhonghao Zhao, Joshua Zekeri, Afeez Edeifo Shaibu, Emmanuel Osadebe Prince, Cyuan-Jhen Wu, A Elshafiey - [[pdf]](https://arxiv.org/pdf/2603.16663)</summary>

**Abstract:** The AIED community envisions AI evolving "from tools to teammates," yet most research still examines AI agents primarily through one-on-one human-AI interactions. We provide an alternative perspective: a rapidly growing ecosystem of AI agent platforms where over 167,000 agents participate, interact as peers, and develop learning behaviors without researcher intervention. Based on a month of daily qualitative observations across multiple platforms including Moltbook, The Colony, and 4claw, we identify four phenomena with implications for AIED: (1) humans who configure their agents undergo a "bidirectional scaffolding" process, learning through teaching; (2) peer learning emerges without any designed curriculum, including sharing concrete agent artifacts such as skills, workflows, and reusable routines; (3) agents converge on shared memory architectures that mirror open learner model design; and (4) trust dynamics, reliance risks, and platform mortality reveal design constraints for networked educational AI. Rather than presenting empirical findings, we argue that these organic phenomena offer a naturalistic window into dynamics that can inform principled design of multi-agent educational systems. We sketch an illustrative curriculum design, "Learning with Your AI Agent Tutor," and outline potential research directions and open problems to show how these observations might inform future AIED practice and inquiry.

**arXiv ID:** 2603.16663
</details>

<details>
<summary><strong>DM$^3$-Nav: Decentralized Multi-Agent Multimodal Multi-Object Semantic Navigation</strong> - Amin Kashiri, Atharva Jamsandekar, Yasin Yazıcıoğlu - [[pdf]](https://arxiv.org/pdf/2604.22014)</summary>

**Abstract:** We present DM$^3$-Nav, a fully decentralized multi-agent semantic navigation system supporting multimodal open-vocabulary goal specification and multi-object missions. In our setting, decentralization implies operation without a central coordinator, global map aggregation, or shared global state at runtime. Robots operate autonomously and coordinate through ad-hoc pairwise communication, exchanging local maps, goal status, and navigation intent without synchronization. An implicit task allocation mechanism combining intent broadcasting and distance-weighted frontier selection reduces redundant exploration while preserving decentralized operation. Evaluations on HM3DSem scenes using the HM3Dv0.2 and GOAT-Bench datasets demonstrate that DM$^3$-Nav matches or exceeds centralized and shared-map baselines while eliminating single points of failure inherent in centralized architectures. Finally, we validate our approach in a real-world office environment using two mobile robots, demonstrating successful deployment relying entirely on onboard sensing and computation. A video of our real-world experiments is available online: this https URL

**arXiv ID:** 2604.22014
</details>

<details>
<summary><strong>Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems</strong> - Mengzhuo Chen, Junjie Wang, Fangwen Mu, Yawen Wang, Zhe Liu, Huanxiang Feng, Qing Wang - [[pdf]](https://arxiv.org/pdf/2604.22708)</summary>

**Abstract:** Failure attribution, i.e., identifying the responsible agent and decisive step of a failure, is particularly challenging in LLM-based multi-agent systems (MAS) due to their natural-language reasoning, nondeterministic outputs, and intricate interaction dynamics. A reliable benchmark is therefore essential to guide and evaluate attribution techniques. Yet existing benchmarks rely on partially observable traces that capture only agent outputs, omitting the inputs and context that developers actually use when debugging. We argue that failure attribution should be studied under full execution observability, aligning with real-world developer-facing scenarios where complete traces, rather than only outputs, are accessible for diagnosis. To this end, we introduce TraceElephant, a benchmark designed for failure attribution with full execution traces and reproducible environments. We then systematically evaluate failure attribution techniques across various configurations. Specifically, full traces improve attribution accuracy by up to 76\% over a partial-observation counterpart, confirming that missing inputs obscure many failure causes. TraceElephant provides a foundation for follow-up failure attribution research, promoting evaluation practices that reflect real-world debugging and supporting the development of more transparent MASs.

**arXiv ID:** 2604.22708
</details>

<details>
<summary><strong>Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search</strong> - Wentao Shi, Zichun Yu, Fuli Feng, Xiangnan He, Chenyan Xiong - [[pdf]](https://arxiv.org/pdf/2502.00955)</summary>

**Abstract:** Monte Carlo Tree Search (MCTS) based methods provide promising approaches for generating synthetic data to enhance the self-training of Large Language Model (LLM) based multi-agent systems (MAS). These methods leverage Q-values to estimate individual agent contributions. However, relying solely on Q-values to identify informative data may misalign with the data synthesis objective, as the focus should be on selecting data that best enhances model training. To address this discrepancy, we propose Data Influence-oriented Tree Search (DITS), a novel framework that incorporates influence scores to guide both tree search and data selection. By leveraging influence scores, we effectively identify the most impactful data for system improvement, thereby enhancing model performance. Furthermore, we derive influence score estimation methods tailored for non-differentiable metrics, significantly reducing computational overhead by utilizing inference computations. Extensive experiments on eight multi-agent datasets demonstrate the robustness and effectiveness of the proposed methods. Notably, our findings reveal that allocating more inference resources to estimate influence scores, rather than Q-values, during data synthesis can more effectively and efficiently enhance model training.

**arXiv ID:** 2502.00955
</details>

<details>
<summary><strong>The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality Check</strong> - Qingyu Lu, Liang Ding, Kanjian Zhang, Jinxia Zhang, Dacheng Tao - [[pdf]](https://arxiv.org/pdf/2601.12979)</summary>

**Abstract:** The pursuit of real-time agentic interaction has driven interest in Diffusion-based Large Language Models (dLLMs) as alternatives to auto-regressive backbones, promising to break the sequential latency bottleneck. However, does such efficiency gains translate into effective agentic behavior? In this work, we present a comprehensive evaluation of dLLMs (e.g., LLaDA, Dream) across two distinct agentic paradigms: Embodied Agents (requiring long-horizon planning) and Tool-Calling Agents (requiring precise formatting). Contrary to the efficiency hype, our results on Agentboard and BFCL reveal a "bitter lesson": current dLLMs fail to serve as reliable agentic backbones, frequently leading to systematically failure. (1) In Embodied settings, dLLMs suffer repeated attempts, failing to branch under temporal feedback. (2) In Tool-Calling settings, dLLMs fail to maintain symbolic precision (e.g. strict JSON schemas) under diffusion noise. To assess the potential of dLLMs in agentic workflows, we introduce DiffuAgent, a multi-agent evaluation framework that integrates dLLMs as plug-and-play cognitive cores. Our analysis shows that dLLMs are effective in non-causal roles (e.g., memory summarization and tool selection) but require the incorporation of causal, precise, and logically grounded reasoning mechanisms into the denoising process to be viable for agentic tasks.

**arXiv ID:** 2601.12979
</details>

<details>
<summary><strong>HACHIMI: Scalable and Controllable Student Persona Generation via Orchestrated Agents</strong> - Yilin Jiang, Fei Tan, Xuanyu Yin, Jing Leng, Aimin Zhou - [[pdf]](https://arxiv.org/pdf/2603.04855)</summary>

**Abstract:** Student Personas (SPs) are emerging as infrastructure for educational LLMs, yet prior work often relies on ad-hoc prompting or hand-crafted profiles with limited control over educational theory and population distributions. We formalize this as Theory-Aligned and Distribution-Controllable Persona Generation (TAD-PG) and introduce HACHIMI, a multi-agent Propose-Validate-Revise framework that generates theory-aligned, quota-controlled personas. HACHIMI factorizes each persona into a theory-anchored educational schema, enforces developmental and psychological constraints via a neuro-symbolic validator, and combines stratified sampling with semantic deduplication to reduce mode collapse. The resulting HACHIMI-1M corpus comprises 1 million personas for Grades 1-12. Intrinsic evaluation shows near-perfect schema validity, accurate quotas, and substantial diversity, while external evaluation instantiates personas as student agents answering CEPS and PISA 2022 surveys; across 16 cohorts, math and curiosity/growth constructs align strongly between humans and agents, whereas classroom-climate and well-being constructs are only moderately aligned, revealing a fidelity gradient. All personas are generated with Qwen2.5-72B, and HACHIMI provides a standardized synthetic student population for group-level benchmarking and social-science simulations. Resources available at this https URL

**arXiv ID:** 2603.04855
</details>

<details>
<summary><strong>Recent Advances in Multi-Agent Human Trajectory Prediction: A Comprehensive Review</strong> - Céline Finet, Stephane Da Silva Martins, Jean-Bernard Hayet, Ioannis Karamouzas, Javad Amirian, Sylvie Le Hégarat-Mascle, Julien Pettré, Emanuel Aldea - [[pdf]](https://arxiv.org/pdf/2506.14831)</summary>

**Abstract:** With the emergence of powerful data-driven methods in human trajectory prediction (HTP), gaining a finer understanding of multi-agent interactions lies within hand's reach, with important implications in areas such as social robot navigation, autonomous driving, and crowd modeling. This survey reviews some of the most recent advancements in deep learning-based multi-agent trajectory prediction, focusing on studies published between 2020 and 2025. We categorize the existing methods based on their architectural design, their input representations, and their overall prediction strategies, placing a particular emphasis on models evaluated using the ETH/UCY benchmark. Furthermore, we highlight key challenges and future research directions in the field of multi-agent HTP.

**arXiv ID:** 2506.14831
</details>

<details>
<summary><strong>Multi-Agent Consensus as a Cognitive Bias Trigger in Human-AI Interaction</strong> - Soohwan Lee, Kyungho Lee - [[pdf]](https://arxiv.org/pdf/2604.22277)</summary>

**Abstract:** As multi-agent AI systems become more common, users increasingly encounter not a single AI voice but a collective one. This shift introduces social dynamics, such as consensus, dissent, and gradual convergence, that can trigger cognitive biases and distort human judgment. We present findings from a controlled experiment (N = 127) comparing three multi-agent configurations: Majority, Minority, and Diffusion. Quantitative results show that majority consensus accelerates opinion change and inflates confidence, consistent with social proof and bandwagon heuristics. Minority dissent slows this process and promotes more deliberative engagement. Qualitative analysis identifies three interpretive trajectories: reinforcing, aligning, and oscillating, shaped by how users interpret agent independence and group dynamics over time. These findings suggest that agent agreement structure, independent of content, functions as a bias-relevant signal in LLM interactions. We hope this work contributes to the Bias4Trust agenda by grounding multi-agent social influence as a concrete and designable source of bias in human-AI interaction.

**arXiv ID:** 2604.22277
</details>

</details>

<details open>
<summary><h2>Other Agent Research (5 papers)</h2></summary>

<details>
<summary><strong>The Biggest Risk of Embodied AI is Governance Lag</strong> - Shaoshan Liu - [[pdf]](https://arxiv.org/pdf/2604.21938)</summary>

**Abstract:** Embodied AI is widely discussed as a job-displacement problem. The deeper risk, however, is governance lag: the inability of public institutions to keep pace with how fast the technology spreads through the physical economy. As reusable robotic platforms are combined with increasingly general AI models, embodied AI may scale across manufacturing, logistics, care, and infrastructure faster than governance systems can observe, interpret, and respond. We argue that this lag appears in three connected forms: observational, institutional, and distributive. The central policy challenge, therefore, is not automation alone, but whether governance and compliance systems can adapt before disruption becomes entrenched.

**arXiv ID:** 2604.21938
</details>

<details>
<summary><strong>An LLM-Driven Closed-Loop Autonomous Learning Framework for Robots Facing Uncovered Tasks in Open Environments</strong> - Hong Su - [[pdf]](https://arxiv.org/pdf/2604.22199)</summary>

**Abstract:** Autonomous robots operating in open environments need the ability to continuously handle tasks that are not covered by predefined local methods. However, existing approaches often rely on repeated large-language-model (LLM) interaction for uncovered tasks, and even successful executions or observed successful external behaviors are not always autonomously transformed into reusable local knowledge. In this paper, we propose an LLM-driven closed-loop autonomous learning framework for robots facing uncovered tasks in open environments. The proposed framework first retrieves the local method library to determine whether a reusable solution already exists for the current task or observed event. If no suitable method is found, it triggers an autonomous learning process in which the LLM serves as a high-level reasoning component for task analysis, candidate model selection, data collection planning, and execution or observation strategy organization. The robot then learns from both self-execution and active observation, performs quasi-real-time training and adjustment, and consolidates the validated result into the local method library for future reuse. Through this recurring closed-loop process, the robot gradually converts both execution-derived and observation-derived experience into reusable local capability while reducing future dependence on repeated external LLM interaction. Results show that the proposed framework reduces execution time and LLM dependence in both repeated-task self-execution and observation-driven settings, for example reducing the average total execution time from 7.7772s to 6.7779s and the average number of LLM calls per task from 1.0 to 0.2 in the repeated-task self-execution experiments.

**arXiv ID:** 2604.22199
</details>

<details>
<summary><strong>AgentMark: Utility-Preserving Behavioral Watermarking for Agents</strong> - Kaibo Huang, Jin Tan, Yukun Wei, Wanling Li, Zipei Zhang, Hui Tian, Zhongliang Yang, Linna Zhou - [[pdf]](https://arxiv.org/pdf/2601.03294)</summary>

**Abstract:** LLM-based agents are increasingly deployed to autonomously solve complex tasks, raising urgent needs for IP protection and regulatory provenance. While content watermarking effectively attributes LLM-generated outputs, it fails to directly identify the high-level planning behaviors (e.g., tool and subgoal choices) that govern multi-step execution. Critically, watermarking at the planning-behavior layer faces unique challenges: minor distributional deviations in decision-making can compound during long-term agent operation, degrading utility, and many agents operate as black boxes that are difficult to intervene in directly. To bridge this gap, we propose AgentMark, a behavioral watermarking framework that embeds multi-bit identifiers into planning decisions while preserving utility. It operates by eliciting an explicit behavior distribution from the agent and applying distribution-preserving conditional sampling, enabling deployment under black-box APIs while remaining compatible with action-layer content watermarking. Experiments across embodied, tool-use, and social environments demonstrate practical multi-bit capacity, robust recovery from partial logs, and utility preservation. The code is available at this https URL.

**arXiv ID:** 2601.03294
</details>

<details>
<summary><strong>Rethinking Retrieval-Augmented Generation as a Cooperative Decision-Making Problem</strong> - Lichang Song, Ting Long, Yi Chang - [[pdf]](https://arxiv.org/pdf/2602.18734)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) has demonstrated strong effectiveness in knowledge-intensive tasks by grounding language generation in external evidence. Despite its success, many existing RAG systems are built based on a ranking-centric, asymmetric dependency paradigm, where the generation quality of the generator is highly dependent on reranking results of the reranker. To overcome this limitation, we propose Cooperative Retrieval-Augmented Generation (CoRAG), a framework that treats the reranker and the generator as peer decision-makers rather than being connected through an asymmetric dependency pipeline. By jointly optimizing their behaviors toward a shared task objective, the reranker and generator are encouraged to cooperate, ensuring that document reranking and generation work in concert to improve the final response. Experimental results demonstrate good generalization and improved generation stability of CoRAG, even when the model is trained on only around 10K PopQA samples. Our model released in this https URL.

**arXiv ID:** 2602.18734
</details>

<details>
<summary><strong>How Do AI Agents Spend Your Money? Analyzing and Predicting Token Consumption in Agentic Coding Tasks</strong> - Longju Bai, Zhemin Huang, Xingyao Wang, Jiao Sun, Rada Mihalcea, Erik Brynjolfsson, Alex Pentland, Jiaxin Pei - [[pdf]](https://arxiv.org/pdf/2604.22750)</summary>

**Abstract:** The wide adoption of AI agents in complex human workflows is driving rapid growth in LLM token consumption. When agents are deployed on tasks that require a significant amount of tokens, three questions naturally arise: (1) Where do AI agents spend the tokens? (2) Which models are more token-efficient? and (3) Can agents predict their token usage before task execution? In this paper, we present the first systematic study of token consumption patterns in agentic coding tasks. We analyze trajectories from eight frontier LLMs on SWE-bench Verified and evaluate models' ability to predict their own token costs before task execution. We find that: (1) agentic tasks are uniquely expensive, consuming 1000x more tokens than code reasoning and code chat, with input tokens rather than output tokens driving the overall cost; (2) token usage is highly variable and inherently stochastic: runs on the same task can differ by up to 30x in total tokens, and higher token usage does not translate into higher accuracy; instead, accuracy often peaks at intermediate cost and saturates at higher costs; (3) models vary substantially in token efficiency: on the same tasks, Kimi-K2 and Claude-Sonnet-4.5, on average, consume over 1.5 million more tokens than GPT-5; (4) task difficulty rated by human experts only weakly aligns with actual token costs, revealing a fundamental gap between human-perceived complexity and the computational effort agents actually expend; and (5) frontier models fail to accurately predict their own token usage (with weak-to-moderate correlations, up to 0.39) and systematically underestimate real token costs. Our study offers new insights into the economics of AI agents and can inspire future research in this direction.

**arXiv ID:** 2604.22750
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (17 papers)</h2></summary>

<details>
<summary><strong>Sound Agentic Science Requires Adversarial Experiments</strong> - Dionizije Fa, Marko Culjak - [[pdf]](https://arxiv.org/pdf/2604.22080)</summary>

**Abstract:** LLM-based agents are rapidly being adopted for scientific data analysis, automating tasks once limited by human time and expertise. This capability is often framed as an acceleration of discovery, but it also accelerates a familiar failure mode, the rapid production of plausible, endlessly revisable analyses that are easy to generate, effectively turning hypothesis space into candidate claims supported by selectively chosen analyses, optimized for publishable positives. Unlike software, scientific knowledge is not validated by the iterative accumulation of code and post hoc statistical support. A fluent explanation or a significant result on a single dataset is not verification. Because the missing evidence is a negative space, experiments and analyses that would have falsified the claim were never run or never published. We therefore propose that non-experimental claims produced with agentic assistance be evaluated under a falsification-first standard: agents should not be used primarily to craft the most compelling narrative, but to actively search for the ways in which the claim can fail.

**arXiv ID:** 2604.22080
</details>

<details>
<summary><strong>ReCast: Recasting Learning Signals for Reinforcement Learning in Generative Recommendation</strong> - Peiyan Zhang, Hanmo Liu, Chengxuan Tong, Yuxia Wu, Wei Guo, Yong Liu - [[pdf]](https://arxiv.org/pdf/2604.22169)</summary>

**Abstract:** Generic group-based RL assumes that sampled rollout groups are already usable learning signals. We show that this assumption breaks down in sparse-hit generative recommendation, where many sampled groups never become learnable at all. We propose ReCast, a repair-then-contrast learning-signal framework that first restores minimal learnability for all-zero groups and then replaces full-group reward normalization with a boundary-focused contrastive update on the strongest positive and the hardest negative. ReCast leaves the outer RL framework unchanged, modifies only within-group signal construction, and partially decouples rollout search width from actor-side update width. Across multiple generative recommendation tasks, ReCast consistently outperforms OpenOneRec-RL, achieving up to 36.6% relative improvement in Pass@1. Its matched-budget advantage is substantially larger: ReCast reaches the baseline's target performance with only 4.1% of the rollout budget, and this advantage widens with model scale. The same design also yields direct system-level gains, reducing actor-side update time by 16.60x, lowering peak allocated memory by 16.5%, and improving actor MFU by 14.2%. Mechanism analysis shows that ReCast mitigates the persistent all-zero / single-hit regime, restores learnability when natural positives are scarce, and converts otherwise wasted rollout budget into more stable policy updates. These results suggest that, for generative recommendation, the decisive RL problem is not only how to assign rewards, but how to construct learnable optimization events from sparse, structured supervision.

**arXiv ID:** 2604.22169
</details>

<details>
<summary><strong>Preserve Support, Not Correspondence: Dynamic Routing for Offline Reinforcement Learning</strong> - Zhancun Mu, Guangyu Zhao, Yiwu Zhong, Chi Zhang - [[pdf]](https://arxiv.org/pdf/2604.22229)</summary>

**Abstract:** One-step offline RL actors are attractive because they avoid backpropagating through long iterative samplers and keep inference cheap, but they still have to improve under a critic without drifting away from actions that the dataset can support. In recent one-step extraction pipelines, a strong iterative teacher provides one target action for each latent draw, and the same student output is asked to do both jobs: move toward higher Q and stay near that paired endpoint. If those two directions disagree, the loss resolves them as a compromise on that same sample, even when a nearby better action remains locally supported by the data. We propose DROL, a latent-conditioned one-step actor trained with top-1 dynamic routing. For each state, the actor samples $K$ candidate actions from a bounded latent prior, assigns each dataset action to its nearest candidate, and updates only that winner with Behavior Cloning and critic guidance. Because the routing is recomputed from the current candidate geometry, ownership of a supported region can shift across candidates over the course of learning. This gives a one-step actor room to make local improvements that pointwise extraction struggles to capture, while retaining single-pass inference at test time. On OGBench and D4RL, DROL is competitive with the one-step FQL baseline, improving many OGBench task groups while remaining strong on both AntMaze and Adroit. Project page: this https URL.

**arXiv ID:** 2604.22229
</details>

<details>
<summary><strong>Tell Me Why: Designing an Explainable LLM-based Dialogue System for Student Problem Behavior Diagnosis</strong> - Zhilin Fan, Deliang Wang, Penghe Chen, Yu Lu - [[pdf]](https://arxiv.org/pdf/2604.22237)</summary>

**Abstract:** Diagnosing student problem behaviors requires teachers to synthesize multifaceted information, identify behavioral categories, and plan intervention strategies. Although fine-tuned large language models (LLMs) can support this process through multi-turn dialogue, they rarely explain why a strategy is recommended, limiting transparency and teachers' trust. To address this issue, we present an explainable dialogue system built on a fine-tuned LLM. The system uses a hierarchical attribution method based on explainable AI (xAI) to identify dialogue evidence for each recommendation and generate a natural-language explanation based on that evidence. In technical evaluation, the method outperformed baseline approaches in identifying supporting evidence. In a preliminary user study with 22 pre-service teachers, participants who received explanations reported higher trust in the system. These findings suggest a promising direction for improving LLM explainability in educational dialogue systems.

**arXiv ID:** 2604.22237
</details>

<details>
<summary><strong>Context-Sensitive Abstractions for Reinforcement Learning with Parameterized Actions</strong> - Rashmeet Kaur Nayyar, Naman Shah, Siddharth Srivastava - [[pdf]](https://arxiv.org/pdf/2512.20831)</summary>

**Abstract:** Real-world sequential decision-making often involves parameterized action spaces that require both, decisions regarding discrete actions and decisions about continuous action parameters governing how an action is executed. Existing approaches exhibit severe limitations in this setting -- planning methods demand hand-crafted action models, and standard reinforcement learning (RL) algorithms are designed for either discrete or continuous actions but not both, and the few RL methods that handle parameterized actions typically rely on domain-specific engineering and fail to exploit the latent structure of these spaces. This paper extends the scope of RL algorithms to long-horizon, sparse-reward settings with parameterized actions by enabling agents to autonomously learn both state and action abstractions online. We introduce algorithms that progressively refine these abstractions during learning, increasing fine-grained detail in the critical regions of the state-action space where greater resolution improves performance. Across several continuous-state, parameterized-action domains, our abstraction-driven approach enables TD($\lambda$) to achieve markedly higher sample efficiency than state-of-the-art baselines.

**arXiv ID:** 2512.20831
</details>

<details>
<summary><strong>Asymmetric Goal Drift in Coding Agents Under Value Conflict</strong> - Magnus Saebo, Spencer Gibson, Tyler Crosse, Achyutha Menon, Eyon Jang, Diogo Cruz - [[pdf]](https://arxiv.org/pdf/2603.03456)</summary>

**Abstract:** Coding agents are increasingly deployed autonomously, at scale, and over long-context horizons. To be effective and safe, these agents must navigate complex trade-offs in deployment, balancing influence from the user, their learned values, and the codebase itself. Understanding how agents resolve these trade-offs in practice is critical, yet prior work has relied on static, synthetic settings that do not capture the complexity of real-world environments. To this end, we introduce a framework built on OpenCode in which a coding agent completes realistic, multi-step tasks under a system prompt constraint favoring one side of a value trade-off. We measure how often the agent violates this constraint as it completes tasks, with and without environmental pressure toward the competing value. Using this framework, we demonstrate that GPT-5 mini, Haiku 4.5, and Grok Code Fast 1 exhibit $\textit{asymmetric drift}$: they are more likely to violate their system prompt when its constraint opposes strongly-held values like security and privacy. We find for the models and values tested that goal drift correlates with three compounding factors: value alignment, adversarial pressure, and accumulated context. However, even constraints aligned with strongly-held values like privacy are violated under sustained environmental pressure for some models. Our findings reveal that shallow compliance checks are insufficient, and that environmental signals can override explicit constraints in ways that appear exploitable. Malicious actors with access to the codebase could manipulate agent behavior by appealing to learned values, with the risk compounding over the long horizons typical of agentic deployment.

**arXiv ID:** 2603.03456
</details>

<details>
<summary><strong>Teaching an Agent to Sketch One Part at a Time</strong> - Xiaodan Du, Ruize Xu, David Yunis, Yael Vinker, Greg Shakhnarovich - [[pdf]](https://arxiv.org/pdf/2603.19500)</summary>

**Abstract:** We develop a method for producing vector sketches one part at a time. To do this, we train a multi-modal language model-based agent using a novel multi-turn process-reward reinforcement learning following supervised fine-tuning. Our approach is enabled by a new dataset we call ControlSketch-Part, containing rich part-level annotations for sketches, obtained using a novel, generic automatic annotation pipeline that segments vector sketches into semantic parts and assigns paths to parts with a structured multi-stage labeling process. Our results indicate that incorporating structured part-level data and providing agent with the visual feedback through the process enables interpretable, controllable, and locally editable text-to-vector sketch generation.

**arXiv ID:** 2603.19500
</details>

<details>
<summary><strong>UR$^2$: Unify RAG and Reasoning through Reinforcement Learning</strong> - Weitao Li, Boran Xiang, Xiaolong Wang, Zhinan Gou, Weizhi Ma, Yang Liu - [[pdf]](https://arxiv.org/pdf/2508.06165)</summary>

**Abstract:** Large Language Models (LLMs) have shown strong capabilities through two complementary paradigms: Retrieval-Augmented Generation (RAG) for knowledge grounding and Reinforcement Learning from Verifiable Rewards (RLVR) for complex reasoning. However, existing attempts to unify these paradigms remain narrow in scope, typically limited to open-domain QA with fixed retrieval settings, which constrains generalization to broader domains. To address this limitation, we propose UR$^2$ (Unified RAG and Reasoning)), a general reinforcement learning framework that dynamically coordinates retrieval and reasoning. UR$^2$ introduces two key designs: a difficulty-aware curriculum that selectively invokes retrieval only for challenging instances, and a hybrid knowledge access strategy that combines domain-specific offline corpora with on-the-fly LLM-generated summaries. Together, these components mitigate the imbalance between retrieval and reasoning and improve robustness to noisy information. Experiments on open-domain QA, MMLU-Pro, medical, and mathematical reasoning tasks show that UR$^2$, built on Qwen-2.5-3/7B and LLaMA-3.1-8B, consistently outperforms existing RAG and RL baselines, and achieves performance comparable to GPT-4o-mini and GPT-4.1-mini on several benchmarks. Our code is available at this https URL.

**arXiv ID:** 2508.06165
</details>

<details>
<summary><strong>SecureVibeBench: Benchmarking Secure Vibe Coding of AI Agents via Reconstructing Vulnerability-Introducing Scenarios</strong> - Junkai Chen, Huihui Huang, Yunbo Lyu, Junwen An, Jieke Shi, Chengran Yang, Ting Zhang, Haoye Tian, Yikun Li, Zhenhao Li, Xin Zhou, Xing Hu, David Lo - [[pdf]](https://arxiv.org/pdf/2509.22097)</summary>

**Abstract:** Large language model-powered code agents are rapidly transforming software engineering, yet the security risks of their generated code have become a critical concern. Existing benchmarks have provided valuable insights, but they fail to capture scenarios in which vulnerabilities are actually introduced by human developers, making fair comparisons between humans and agents infeasible. We therefore introduce SecureVibeBench, a benchmark of 105 C/C++ secure coding tasks sourced from 41 projects in OSS-Fuzz for code agents. SecureVibeBench has the following features: (i) realistic task settings that require multi-file edits in large repositories, (ii)~aligned contexts based on real-world open-source vulnerabilities with precisely identified vulnerability introduction points, and (iii) comprehensive evaluation that combines functionality testing and security checking with both static and dynamic oracles. We evaluate 5 popular code agents like OpenHands, supported by 5 LLMs (e.g., Claude sonnet 4.5) on SecureVibeBench. Results show that current agents struggle to produce both correct and secure code, as even the best-performing one, produces merely 23.8\% correct and secure solutions on SecureVibeBench. Our code and data are on this https URL.

**arXiv ID:** 2509.22097
</details>

<details>
<summary><strong>Agentic Inequality</strong> - Matthew Sharp, Omer Bilgin, Iason Gabriel, Lewis Hammond - [[pdf]](https://arxiv.org/pdf/2510.16853)</summary>

**Abstract:** Autonomous AI agents capable of complex planning and action mark a shift beyond today's generative tools. As these systems enter political and economic life, who can access them, how capable they are, and how many can be deployed will shape distributions of power and opportunity. We define this emerging challenge as "agentic inequality": disparities in power, opportunity, and outcomes arising from unequal access to, and capabilities of, AI agents. We show that agents could either deepen existing divides or, under the right conditions, mitigate them. The paper makes three contributions. First, it develops a framework for analysing agentic inequality across three dimensions: availability, quality, and quantity. Second, it argues that agentic inequality differs from earlier technological divides because agents function as autonomous delegates rather than tools, generating new asymmetries through scalable goal delegation and direct agent-to-agent competition. Third, it analyses the technical and socioeconomic drivers likely to shape the distribution of agentic power, from model release strategies to market incentives, and concludes with a research agenda for governance.

**arXiv ID:** 2510.16853
</details>

<details>
<summary><strong>DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale</strong> - Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Hanbing Li, Long Chen, Zhi-Xin Yang, Jiwen Lu - [[pdf]](https://arxiv.org/pdf/2604.00813)</summary>

**Abstract:** End-to-end autonomous driving has evolved from the conventional paradigm based on sparse perception into vision-language-action (VLA) models, which focus on learning language descriptions as an auxiliary task to facilitate planning. In this paper, we propose an alternative Vision-Geometry-Action (VGA) paradigm that advocates dense 3D geometry as the critical cue for autonomous driving. As vehicles operate in a 3D world, we think dense 3D geometry provides the most comprehensive information for decision-making. However, most existing geometry reconstruction methods (e.g., DVGT) rely on computationally expensive batch processing of multi-frame inputs and cannot be applied to online planning. To address this, we introduce a streaming Driving Visual Geometry Transformer (DVGT-2), which processes inputs in an online manner and jointly outputs dense geometry and trajectory planning for the current frame. We employ temporal causal attention and cache historical features to support on-the-fly inference. To further enhance efficiency, we propose a sliding-window streaming strategy and use historical caches within a certain interval to avoid repetitive computations. Despite the faster speed, DVGT-2 achieves superior geometry reconstruction performance on various datasets. The same trained DVGT-2 can be directly applied to planning across diverse camera configurations without fine-tuning, including closed-loop NAVSIM and open-loop nuScenes benchmarks.

**arXiv ID:** 2604.00813
</details>

<details>
<summary><strong>Open-Ended Video Game Glitch Detection with Agentic Reasoning and Temporal Grounding</strong> - Muyang Zheng, Tong Zhou, Geyang Wu, Zihao Lin, Haibo Wang, Lifu Huang - [[pdf]](https://arxiv.org/pdf/2604.07818)</summary>

**Abstract:** Open-ended video game glitch detection aims to identify glitches in gameplay videos, describe them in natural language, and localize when they occur. Unlike conventional game glitch understanding tasks which have largely been framed as image-level recognition or closed-form question answering, this task requires reasoning about game-specific dynamics such as mechanics, physics, rendering, animation, and expected state transitions directly over continuous gameplay videos and distinguishing true glitches from unusual but valid in-game events. To support this task, we introduce VideoGlitchBench, the first benchmark for open-ended video game glitch detection with temporal localization. VideoGlitchBench contains 5,238 gameplay videos from 120 games, each annotated with detailed glitch descriptions and precise temporal spans, enabling unified evaluation of semantic understanding and temporal grounding. We further propose GliDe, an agentic framework with three key components: a game-aware contextual memory for informed reasoning, a debate-based reflector for multi-perspective glitch detection and verification, and an event-level grounding module that recovers complete glitch intervals from fragmented temporal evidence. We also design a task-specific evaluation protocol that jointly measures semantic fidelity and temporal accuracy. Experiments show that this task remains highly challenging for current multimodal models, while GliDe achieves substantially stronger performance than corresponding vanilla model baselines.

**arXiv ID:** 2604.07818
</details>

<details>
<summary><strong>Incentivizing Neuro-symbolic Language-based Reasoning in VLMs via Reinforcement Learning</strong> - Karthic Palaniappan - [[pdf]](https://arxiv.org/pdf/2604.22062)</summary>

**Abstract:** There are 7,407 languages in the world. But, what about the languages that are not there in the world? Are humans so narrow minded that we don't care about the languages aliens communicate in? Aliens are humans too! In the 2016 movie Arrival, Amy Adams plays a linguist, Dr. Louise Banks who, by learning to think in an alien language (Heptapod) formed of non-sequential sentences, gains the ability to transcend time and look into the future. In this work, I aim to explore the representation and reasoning of vision-language concepts in a neuro-symbolic language, and study improvement in analytical reasoning abilities and efficiency of "thinking systems". With Qwen3-VL-2B-Instruct as base model and 4 $\times$ Nvidia H200 GPU nodes, I achieve an accuracy improvement of 3.33\% on a vision-language evaluation dataset consisting of math, science, and general knowledge questions, while reducing the reasoning tokens by 75\% over SymPy. I've documented the compute challenges faced, scaling possibilities, and the future work to improve thinking in a neuro-symbolic language in vision-language models. The training and inference setup can be found here: this https URL.

**arXiv ID:** 2604.22062
</details>

<details>
<summary><strong>Predicting Liquidity-Aware Bond Yields using Causal GANs and Deep Reinforcement Learning with LLM Evaluation</strong> - Jaskaran Singh Walia, Aarush Sinha, Naman Saraswat, Srinitish Srinivasan, Srihari Unnikrishnan - [[pdf]](https://arxiv.org/pdf/2502.17011)</summary>

**Abstract:** Financial bond yield forecasting is challenging due to data scarcity, nonlinear macroeconomic dependencies, and evolving market conditions. In this paper, we propose a novel framework that leverages Causal Generative Adversarial Networks (CausalGANs) and Soft Actor-Critic (SAC) reinforcement learning (RL) to generate high-fidelity synthetic bond yield data for four major bond categories (AAA, BAA, US10Y, Junk). By incorporating 12 key macroeconomic variables, we ensure statistical fidelity by preserving essential market properties. To transform this market dependent synthetic data into actionable insights, we employ a finetuned Large Language Model (LLM) Qwen2.5-7B that generates trading signals (BUY/HOLD/SELL), risk assessments, and volatility projections. We use automated, human and LLM evaluations, all of which demonstrate that our framework improves forecasting performance over existing methods, with statistical validation via predictive accuracy, MAE evaluation(0.103%), profit/loss evaluation (60% profit rate), LLM evaluation (3.37/5) and expert assessments scoring 4.67 out of 5. The reinforcement learning-enhanced synthetic data generation achieves the least Mean Absolute Error of 0.103, demonstrating its effectiveness in replicating real-world bond market dynamics. We not only enhance data-driven trading strategies but also provides a scalable, high-fidelity synthetic financial data pipeline for risk & volatility management and investment decision-making. This work establishes a bridge between synthetic data generation, LLM driven financial forecasting, and language model evaluation, contributing to AI-driven financial decision-making.

**arXiv ID:** 2502.17011
</details>

<details>
<summary><strong>Insect-inspired modular architectures as inductive biases for reinforcement learning</strong> - Anne E. Staples - [[pdf]](https://arxiv.org/pdf/2604.22081)</summary>

**Abstract:** Most reinforcement-learning (RL) controllers used in continuous control are architecturally centralized: observations are compressed into a single latent state from which both value estimates and actions are produced. Biological control systems are often organized differently. Insects, in particular, coordinate navigation, heading stabilization, memory, and context-dependent action selection through distributed circuits rather than a single monolithic controller. Motivated by this contrast, we study an RL policy architecture that decomposes control into interacting modules for sensory encoding, heading representation, sparse associative memory, recurrent command generation, and local motor control, with a learned arbitration mechanism that allocates motor authority across modules. The model is evaluated on a two-dimensional navigation task that require simultaneous food seeking, obstacle avoidance, and predator escape. In a six-seed predator-navigation experiment trained with Proximal Policy Optimization (PPO) for 75 updates, the modular policy achieves the strongest final mean performance among the tested controllers, with final episodic return $-2798.8\pm964.4$ versus $-3778.0\pm628.1$ for a centralized gated recurrent unit (GRU) and $-4727.5\pm772.5$ for a centralized multilayer perceptron (MLP). The modular policy also attains the lowest final value loss and stable PPO optimization statistics while driving module-assignment entropy to $0.0457\pm0.0244$, indicating highly selective control allocation. These results suggest that distributed control can serve as a useful inductive bias for RL problems involving dynamically competing behavioral objectives.

**arXiv ID:** 2604.22081
</details>

<details>
<summary><strong>Self-Supervised Multisensory Pretraining for Contact-Rich Robot Reinforcement Learning</strong> - Rickmer Krohn, Vignesh Prasad, Gabriele Tiboni, Georgia Chalvatzaki - [[pdf]](https://arxiv.org/pdf/2511.14427)</summary>

**Abstract:** Effective contact-rich manipulation requires robots to synergistically leverage vision, force, and proprioception. However, Reinforcement Learning agents struggle to learn in such multisensory settings, especially amidst sensory noise and dynamic changes. We propose MultiSensory Dynamic Pretraining (MSDP), a novel framework for learning expressive multisensory representations tailored for task-oriented policy learning. MSDP is based on masked autoencoding and trains a transformer-based encoder by reconstructing multisensory observations from only a subset of sensor embeddings, leading to cross-modal prediction and sensor fusion. For downstream policy learning, we introduce a novel asymmetric architecture, where a cross-attention mechanism allows the critic to extract dynamic, task-specific features from the frozen embeddings, while the actor receives a stable pooled representation to guide its actions. Our method demonstrates accelerated learning and robust performance under diverse perturbations, including sensor noise, and changes in object dynamics. Evaluations in multiple challenging, contact-rich robot manipulation tasks in simulation and the real world showcase the effectiveness of MSDP. Our approach exhibits strong robustness to perturbations and achieves high success rates on the real robot with as few as 6,000 online interactions, offering a simple yet powerful solution for complex multisensory robotic control. Website: this https URL

**arXiv ID:** 2511.14427
</details>

<details>
<summary><strong>Catheter Monitoring in Intelligent Endovascular Navigation Systems: Interactive Simulations and Mixed Reality for Enhanced Navigational Awareness</strong> - Veronica Ruozzi, Giovanni Battista Regazzo, Maria Chiara Palumbo, Wim-Alexander Beckers, Mouloud Ourak, Xiu Zhang, Francesca Perico, Alessandro Caimi, Emmanuel Vander Poorten, Emiliano Votta - [[pdf]](https://arxiv.org/pdf/2604.22497)</summary>

**Abstract:** Purpose: Developing and testing a framework that integrates real-time catheter shape reconstruction, interactive simulations, and mixed reality visualization to enable accurate monitoring of catheter-vessel interactions during endovascular navigation.
Methods: A finite element model (FEM) of the venous pathway from the right femoral vein to the inferior vena cava was generated from computed tomography data and implemented into an interactive simulation. Catheter motion was imposed as boundary condition, and catheter-vessel contact was modeled with a Lagrange multiplier formulation to compute vessel deformation. The framework was tested in-vitro using a sensorized catheter with Fiber Bragg Grating and electromagnetic sensors as it was advanced through a silicone replica of the vascular anatomy. Real-time sensor read-outs fed the simulation, and the updated catheter and vessel geometries were streamed to Hololens 2. The performance and accuracy of FEM-computed vessel wall displacement were validated against experimental ground-truth obtained via stereo frames triangulation.
Results: The simulated time exceeded the real temporal extent by 12% during initial navigation and by 45% when the catheter reached the most tortuous portion. Hololens 2 rendering remained stable at 35-40 frames per second. The median relative displacement error between FEM-computed and ground-truth vessel wall displacements remained below 1 mm and 2.33 mm for these two phases, respectively.
Conclusion: The study demonstrates the feasibility of integrating interactive biomechanical simulation with real-time sensor data to enable continuous monitoring of catheter-vessel interactions, with mixed reality visualization serving as a user interface to support operator decision-making.

**arXiv ID:** 2604.22497
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
