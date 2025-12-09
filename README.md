# Agent arXiv Daily

**Last Updated:** 2025-12-09 02:18:08

**Total Papers:** 34

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (1 papers)</h2></summary>

<details>
<summary><strong>From Simulation to Strategy: Automating Personalized Interaction Planning for Conversational Agents</strong> - Wen-Yu Chang, Tzu-Hung Huang, Chih-Ho Chen, Yun-Nung Chen - [[pdf]](https://arxiv.org/pdf/2510.08621)</summary>

**Abstract:** Amid the rapid rise of agentic dialogue models, realistic user-simulator studies are essential for tuning effective conversation strategies. This work investigates a sales-oriented agent that adapts its dialogue based on user profiles spanning age, gender, and occupation. While age and gender influence overall performance, occupation produces the most pronounced differences in conversational intent. Leveraging this insight, we introduce a lightweight, occupation-conditioned strategy that guides the agent to prioritize intents aligned with user preferences, resulting in shorter and more successful dialogues. Our findings highlight the importance of rich simulator profiles and demonstrate how simple persona-informed strategies can enhance the effectiveness of sales-oriented dialogue systems.

**arXiv ID:** 2510.08621
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (6 papers)</h2></summary>

<details>
<summary><strong>PRiSM: An Agentic Multimodal Benchmark for Scientific Reasoning via Python-Grounded Evaluation</strong> - Shima Imani, Seungwhan Moon, Adel Ahmadyan, Lu Zhang, Kirmani Ahmed, Babak Damavandi - [[pdf]](https://arxiv.org/pdf/2512.05930)</summary>

**Abstract:** Evaluating vision-language models (VLMs) in scientific domains like mathematics and physics poses unique challenges that go far beyond predicting final answers. These domains demand conceptual understanding, symbolic reasoning, and adherence to formal laws, requirements that most existing benchmarks fail to address. In particular, current datasets tend to be static, lacking intermediate reasoning steps, robustness to variations, or mechanisms for verifying scientific correctness. To address these limitations, we introduce PRiSM, a synthetic, fully dynamic, and multimodal benchmark for evaluating scientific reasoning via grounded Python code. PRiSM includes over 24,750 university-level physics and math problems, and it leverages our scalable agent-based pipeline, PrismAgent, to generate well-structured problem instances. Each problem contains dynamic textual and visual input, a generated figure, alongside rich structured outputs: executable Python code for ground truth generation and verification, and detailed step-by-step reasoning. The dynamic nature and Python-powered automated ground truth generation of our benchmark allow for fine-grained experimental auditing of multimodal VLMs, revealing failure modes, uncertainty behaviors, and limitations in scientific reasoning. To this end, we propose five targeted evaluation tasks covering generalization, symbolic program synthesis, perturbation robustness, reasoning correction, and ambiguity resolution. Through comprehensive evaluation of existing VLMs, we highlight their limitations and showcase how PRiSM enables deeper insights into their scientific reasoning capabilities.

**arXiv ID:** 2512.05930
</details>

<details>
<summary><strong>AREA3D: Active Reconstruction Agent with Unified Feed-Forward 3D Perception and Vision-Language Guidance</strong> - Tianling Xu, Shengzhe Gan, Leslie Gu, Yuelei Li, Fangneng Zhan, Hanspeter Pfister - [[pdf]](https://arxiv.org/pdf/2512.05131)</summary>

**Abstract:** Active 3D reconstruction enables an agent to autonomously select viewpoints to efficiently obtain accurate and complete scene geometry, rather than passively reconstructing scenes from pre-collected images. However, existing active reconstruction methods often rely on hand-crafted geometric heuristics, which can lead to redundant observations without substantially improving reconstruction quality. To address this limitation, we propose AREA3D, an active reconstruction agent that leverages feed-forward 3D reconstruction models and vision-language guidance. Our framework decouples view-uncertainty modeling from the underlying feed-forward reconstructor, enabling precise uncertainty estimation without expensive online optimization. In addition, an integrated vision-language model provides high-level semantic guidance, encouraging informative and diverse viewpoints beyond purely geometric cues. Extensive experiments on both scene-level and object-level benchmarks demonstrate that AREA3D achieves state-of-the-art reconstruction accuracy, particularly in the sparse-view regime. Code will be made available at: this https URL .

**arXiv ID:** 2512.05131
</details>

<details>
<summary><strong>From Segments to Scenes: Temporal Understanding in Autonomous Driving via Vision-Language Model</strong> - Kevin Cannons, Saeed Ranjbar Alvar, Mohammad Asiful Hossain, Ahmad Rezaei, Mohsen Gholami, Alireza Heidarikhazaei, Zhou Weimin, Yong Zhang, Mohammad Akbari - [[pdf]](https://arxiv.org/pdf/2512.05277)</summary>

**Abstract:** Temporal understanding in autonomous driving (AD) remains a significant challenge, even for recent state-of-the-art (SoTA) Vision-Language Models (VLMs). Prior work has introduced datasets and benchmarks aimed at improving temporal reasoning, but these have emphasized other video content, including sports, cooking, and movies. No existing benchmark focuses exclusively on the unique challenges of temporal understanding in ego-centric AD footage. To fill this gap, the Temporal Understanding in Autonomous Driving (TAD) benchmark is presented, which evaluates VLMs' ability to capture the dynamic relationships between actions in AD. TAD comprises nearly 6,000 question-answer (QA) pairs, spanning 7 human-designed tasks. In addition, an evaluation is performed that consists of 9 closed- and open-source generalist models as well as SoTA AD specialist models. When applied to TAD, current SoTA models demonstrated substandard accuracies, largely due to imperfect fine-grained motion understanding. To improve motion understanding and overall accuracy on TAD, two novel training-free solutions are proposed: Scene-CoT, that leverages Chain-of-Thought (CoT) and TCogMap, which incorporates an ego-centric temporal cognitive map. The proposed approaches are integrated with existing VLMs and improve average accuracy on TAD by up to 17.72%. By introducing TAD, benchmarking multiple SoTA models, and proposing effective enhancements, this work aims to catalyze future research on temporal understanding in AD. The benchmark and evaluation code are available at \href{this https URL}{Hugging Face} and \href{this https URL}{Github}, respectively.

**arXiv ID:** 2512.05277
</details>

<details>
<summary><strong>IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks</strong> - Xiaoya Lu, Zeren Chen, Xuhao Hu, Yijin Zhou, Weichen Zhang, Dongrui Liu, Lu Sheng, Jing Shao - [[pdf]](https://arxiv.org/pdf/2506.16402)</summary>

**Abstract:** Flawed planning from VLM-driven embodied agents poses significant safety hazards, hindering their deployment in real-world household tasks. However, existing static, non-interactive evaluation paradigms fail to adequately assess risks within these interactive environments, since they cannot simulate dynamic risks that emerge from an agent's actions and rely on unreliable post-hoc evaluations that ignore unsafe intermediate steps. To bridge this critical gap, we propose evaluating an agent's interactive safety: its ability to perceive emergent risks and execute mitigation steps in the correct procedural order. We thus present IS-Bench, the first multi-modal benchmark designed for interactive safety, featuring 161 challenging scenarios with 388 unique safety risks instantiated in a high-fidelity simulator. Crucially, it facilitates a novel process-oriented evaluation that verifies whether risk mitigation actions are performed before/after specific risk-prone steps. Extensive experiments on leading VLMs, including the GPT-4o and Gemini-2.5 series, reveal that current agents lack interactive safety awareness, and that while safety-aware Chain-of-Thought can improve performance, it often compromises task completion. By highlighting these critical limitations, IS-Bench provides a foundation for developing safer and more reliable embodied AI systems. Code and data are released under this https URL.

**arXiv ID:** 2506.16402
</details>

<details>
<summary><strong>AURA: A Diagnostic Framework for Tracking User Satisfaction of Interactive Planning Agents</strong> - Takyoung Kim, Janvijay Singh, Shuhaib Mehri, Emre Can Acikgoz, Sagnik Mukherjee, Nimet Beyza Bozdag, Sumuk Shashidhar, Gokhan Tur, Dilek Hakkani-Tür - [[pdf]](https://arxiv.org/pdf/2505.01592)</summary>

**Abstract:** The growing capabilities of large language models (LLMs) in instruction-following and context-understanding lead to the era of agents with numerous applications. Among these, task planning agents have become especially prominent in realistic scenarios involving complex internal pipelines, such as context understanding, tool management, and response generation. However, existing benchmarks predominantly evaluate agent performance based on task completion as a proxy for overall effectiveness. We hypothesize that merely improving task completion is misaligned with maximizing user satisfaction, as users interact with the entire agentic process and not only the end result. To address this gap, we propose AURA, an Agent-User inteRaction Assessment framework that conceptualizes the behavioral stages of interactive task planning agents. AURA offers a comprehensive assessment of agent through a set of atomic LLM evaluation criteria, allowing researchers and practitioners to diagnose specific strengths and weaknesses within the agent's decision-making pipeline. Our analyses show that agents excel in different behavioral stages, with user satisfaction shaped by both outcomes and intermediate behaviors. We also highlight future directions, including systems that leverage multiple agents and the limitations of user simulators in task planning.

**arXiv ID:** 2505.01592
</details>

<details>
<summary><strong>Multi-Modal Data-Efficient 3D Scene Understanding for Autonomous Driving</strong> - Lingdong Kong, Xiang Xu, Jiawei Ren, Wenwei Zhang, Liang Pan, Kai Chen, Wei Tsang Ooi, Ziwei Liu - [[pdf]](https://arxiv.org/pdf/2405.05258)</summary>

**Abstract:** Efficient data utilization is crucial for advancing 3D scene understanding in autonomous driving, where reliance on heavily human-annotated LiDAR point clouds challenges fully supervised methods. Addressing this, our study extends into semi-supervised learning for LiDAR semantic segmentation, leveraging the intrinsic spatial priors of driving scenes and multi-sensor complements to augment the efficacy of unlabeled datasets. We introduce LaserMix++, an evolved framework that integrates laser beam manipulations from disparate LiDAR scans and incorporates LiDAR-camera correspondences to further assist data-efficient learning. Our framework is tailored to enhance 3D scene consistency regularization by incorporating multi-modality, including 1) multi-modal LaserMix operation for fine-grained cross-sensor interactions; 2) camera-to-LiDAR feature distillation that enhances LiDAR feature learning; and 3) language-driven knowledge guidance generating auxiliary supervisions using open-vocabulary models. The versatility of LaserMix++ enables applications across LiDAR representations, establishing it as a universally applicable solution. Our framework is rigorously validated through theoretical analysis and extensive experiments on popular driving perception datasets. Results demonstrate that LaserMix++ markedly outperforms fully supervised alternatives, achieving comparable accuracy with five times fewer annotations and significantly improving the supervised-only baselines. This substantial advancement underscores the potential of semi-supervised approaches in reducing the reliance on extensive labeled data in LiDAR-based 3D scene understanding systems.

**arXiv ID:** 2405.05258
</details>

</details>

<details open>
<summary><h2>LLM Agents (1 papers)</h2></summary>

<details>
<summary><strong>Active Video Perception: Iterative Evidence Seeking for Agentic Long Video Understanding</strong> - Ziyang Wang, Honglu Zhou, Shijie Wang, Junnan Li, Caiming Xiong, Silvio Savarese, Mohit Bansal, Michael S. Ryoo, Juan Carlos Niebles - [[pdf]](https://arxiv.org/pdf/2512.05774)</summary>

**Abstract:** Long video understanding (LVU) is challenging because answering real-world queries often depends on sparse, temporally dispersed cues buried in hours of mostly redundant and irrelevant content. While agentic pipelines improve video reasoning capabilities, prevailing frameworks rely on a query-agnostic captioner to perceive video information, which wastes computation on irrelevant content and blurs fine-grained temporal and spatial information. Motivated by active perception theory, we argue that LVU agents should actively decide what, when, and where to observe, and continuously assess whether the current observation is sufficient to answer the query. We present Active Video Perception (AVP), an evidence-seeking framework that treats the video as an interactive environment and acquires compact, queryrelevant evidence directly from pixels. Concretely, AVP runs an iterative plan-observe-reflect process with MLLM agents. In each round, a planner proposes targeted video interactions, an observer executes them to extract time-stamped evidence, and a reflector evaluates the sufficiency of the evidence for the query, either halting with an answer or triggering further observation. Across five LVU benchmarks, AVP achieves highest performance with significant improvements. Notably, AVP outperforms the best agentic method by 5.7% in average accuracy while only requires 18.4% inference time and 12.4% input tokens.

**arXiv ID:** 2512.05774
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (11 papers)</h2></summary>

<details>
<summary><strong>The Seeds of Scheming: Weakness of Will in the Building Blocks of Agentic Systems</strong> - Robert Yang - [[pdf]](https://arxiv.org/pdf/2512.05449)</summary>

**Abstract:** Large language models display a peculiar form of inconsistency: they "know" the correct answer but fail to act on it. In human philosophy, this tension between global judgment and local impulse is called akrasia, or weakness of will. We propose akrasia as a foundational concept for analyzing inconsistency and goal drift in agentic AI systems. To operationalize it, we introduce a preliminary version of the Akrasia Benchmark, currently a structured set of prompting conditions (Baseline [B], Synonym [S], Temporal [T], and Temptation [X]) that measures when a model's local response contradicts its own prior commitments. The benchmark enables quantitative comparison of "self-control" across model families, decoding strategies, and temptation types. Beyond single-model evaluation, we outline how micro-level akrasia may compound into macro-level instability in multi-agent systems that may be interpreted as "scheming" or deliberate misalignment. By reframing inconsistency as weakness of will, this work connects agentic behavior to classical theories of agency and provides an empirical bridge between philosophy, psychology, and the emerging science of agentic AI.

**arXiv ID:** 2512.05449
</details>

<details>
<summary><strong>XR-DT: Extended Reality-Enhanced Digital Twin for Agentic Mobile Robots</strong> - Tianyi Wang, Jiseop Byeon, Ahmad Yehia, Huihai Wang, Yiming Xu, Tianyi Zeng, Ziran Wang, Junfeng Jiao, Christian Claudel - [[pdf]](https://arxiv.org/pdf/2512.05270)</summary>

**Abstract:** As mobile robots increasingly operate alongside humans in shared workspaces, ensuring safe, efficient, and interpretable Human-Robot Interaction (HRI) has become a pressing challenge. While substantial progress has been devoted to human behavior prediction, limited attention has been paid to how humans perceive, interpret, and trust robots' inferences, impeding deployment in safety-critical and socially embedded environments. This paper presents XR-DT, an eXtended Reality-enhanced Digital Twin framework for agentic mobile robots, that bridges physical and virtual spaces to enable bi-directional understanding between humans and robots. Our hierarchical XR-DT architecture integrates virtual-, augmented-, and mixed-reality layers, fusing real-time sensor data, simulated environments in the Unity game engine, and human feedback captured through wearable AR devices. Within this framework, we design an agentic mobile robot system with a unified diffusion policy for context-aware task adaptation. We further propose a chain-of-thought prompting mechanism that allows multimodal large language models to reason over human instructions and environmental context, while leveraging an AutoGen-based multi-agent coordination layer to enhance robustness and collaboration in dynamic tasks. Initial experimental results demonstrate accurate human and robot trajectory prediction, validating the XR-DT framework's effectiveness in HRI tasks. By embedding human intention, environmental dynamics, and robot cognition into the XR-DT framework, our system enables interpretable, trustworthy, and adaptive HRI.

**arXiv ID:** 2512.05270
</details>

<details>
<summary><strong>Trusted AI Agents in the Cloud</strong> - Teofil Bodea, Masanori Misono, Julian Pritzi, Patrick Sabanic, Thore Sommer, Harshavardhan Unnibhavi, David Schall, Nuno Santos, Dimitrios Stavrakakis, Pramod Bhatotia - [[pdf]](https://arxiv.org/pdf/2512.05951)</summary>

**Abstract:** AI agents powered by large language models are increasingly deployed as cloud services that autonomously access sensitive data, invoke external tools, and interact with other agents. However, these agents run within a complex multi-party ecosystem, where untrusted components can lead to data leakage, tampering, or unintended behavior. Existing Confidential Virtual Machines (CVMs) provide only per binary protection and offer no guarantees for cross-principal trust, accelerator-level isolation, or supervised agent behavior. We present Omega, a system that enables trusted AI agents by enforcing end-to-end isolation, establishing verifiable trust across all contributing principals, and supervising every external interaction with accountable provenance. Omega builds on Confidential VMs and Confidential GPUs to create a Trusted Agent Platform that hosts many agents within a single CVM using nested isolation. It also provides efficient multi-agent orchestration with cross-principal trust establishment via differential attestation, and a policy specification and enforcement framework that governs data access, tool usage, and inter-agent communication for data protection and regulatory compliance. Implemented on AMD SEV-SNP and NVIDIA H100, Omega fully secures agent state across CVM-GPU, and achieves high performance while enabling high-density, policy-compliant multi-agent deployments at cloud scale.

**arXiv ID:** 2512.05951
</details>

<details>
<summary><strong>Debate over Mixed-knowledge: A Robust Multi-Agent Reasoning Framework for Incomplete Knowledge Graph Question Answering</strong> - Jilong Liu, Pengyang Shao, Wei Qin, Fei Liu, Yonghui Yang, Richang Hong - [[pdf]](https://arxiv.org/pdf/2511.12208)</summary>

**Abstract:** Knowledge Graph Question Answering (KGQA) aims to improve factual accuracy by leveraging structured knowledge. However, real-world Knowledge Graphs (KGs) are often incomplete, leading to the problem of Incomplete KGQA (IKGQA). A common solution is to incorporate external data to fill knowledge gaps, but existing methods lack the capacity to adaptively and contextually fuse multiple sources, failing to fully exploit their complementary strengths. To this end, we propose Debate over Mixed-knowledge (DoM), a novel framework that enables dynamic integration of structured and unstructured knowledge for IKGQA. Built upon the Multi-Agent Debate paradigm, DoM assigns specialized agents to perform inference over knowledge graphs and external texts separately, and coordinates their outputs through iterative interaction. It decomposes the input question into sub-questions, retrieves evidence via dual agents (KG and Retrieval-Augmented Generation, RAG), and employs a judge agent to evaluate and aggregate intermediate answers. This collaboration exploits knowledge complementarity and enhances robustness to KG incompleteness. In addition, existing IKGQA datasets simulate incompleteness by randomly removing triples, failing to capture the irregular and unpredictable nature of real-world knowledge incompleteness. To address this, we introduce a new dataset, Incomplete Knowledge Graph WebQuestions, constructed by leveraging real-world knowledge updates. These updates reflect knowledge beyond the static scope of KGs, yielding a more realistic and challenging benchmark. Through extensive experiments, we show that DoM consistently outperforms state-of-the-art baselines.

**arXiv ID:** 2511.12208
</details>

<details>
<summary><strong>Designing LLM-based Multi-Agent Systems for Software Engineering Tasks: Quality Attributes, Design Patterns and Rationale</strong> - Yangxiao Cai, Ruiyin Li, Peng Liang, Mojtaba Shahin, Zengyang Li - [[pdf]](https://arxiv.org/pdf/2511.08475)</summary>

**Abstract:** As the complexity of Software Engineering (SE) tasks continues to escalate, Multi-Agent Systems (MASs) have emerged as a focal point of research and practice due to their autonomy and scalability. Furthermore, through leveraging the reasoning and planning capabilities of Large Language Models (LLMs), the application of LLM-based MASs in the field of SE is garnering increasing attention. However, there is no dedicated study that systematically explores the design of LLM-based MASs, including the Quality Attributes (QAs) on which designers mainly focus, the design patterns used by designers, and the rationale guiding the design of LLM-based MASs for SE tasks. To this end, we conducted a study to identify the QAs that LLM-based MASs for SE tasks focus on, the design patterns used in the MASs, and the design rationale for the MASs. We collected 94 papers on LLM-based MASs for SE tasks as the source. Our study shows that: (1) Code Generation is the most common SE task solved by LLM-based MASs among ten identified SE tasks, (2) Functional Suitability is the QA on which designers of LLM-based MASs pay the most attention, (3) Role-Based Cooperation is the design pattern most frequently employed among 16 patterns used to construct LLM-based MASs, and (4) Improving the Quality of Generated Code is the most common rationale behind the design of LLM-based MASs. Based on the study results, we presented the implications for the design of LLM-based MASs to support SE tasks.

**arXiv ID:** 2511.08475
</details>

<details>
<summary><strong>Distributed scalable coupled policy algorithm for networked multi-agent reinforcement learning</strong> - Pengcheng Dai, Dongming Wang, Wenwu Yu, Wei Ren - [[pdf]](https://arxiv.org/pdf/2512.05447)</summary>

**Abstract:** This paper studies networked multi-agent reinforcement learning (NMARL) with interdependent rewards and coupled policies. In this setting, each agent's reward depends on its own state-action pair as well as those of its direct neighbors, and each agent's policy is parameterized by its local parameters together with those of its $\kappa_{p}$-hop neighbors, with $\kappa_{p}\geq 1$ denoting the coupled radius. The objective of the agents is to collaboratively optimize their policies to maximize the discounted average cumulative reward. To address the challenge of interdependent policies in collaborative optimization, we introduce a novel concept termed the neighbors' averaged $Q$-function and derive a new expression for the coupled policy gradient. Based on these theoretical foundations, we develop a distributed scalable coupled policy (DSCP) algorithm, where each agent relies only on the state-action pairs of its $\kappa_{p}$-hop neighbors and the rewards its their $(\kappa_{p}+1)$-hop neighbors. Specially, in the DSCP algorithm, we employ a geometric 2-horizon sampling method that does not require storing a full $Q$-table to obtain an unbiased estimate of the coupled policy gradient. Moreover, each agent interacts exclusively with its direct neighbors to obtain accurate policy parameters, while maintaining local estimates of other agents' parameters to execute its local policy and collect samples for optimization. These estimates and policy parameters are updated via a push-sum protocol, enabling distributed coordination of policy updates across the network. We prove that the joint policy produced by the proposed algorithm converges to a first-order stationary point of the objective function. Finally, the effectiveness of DSCP algorithm is demonstrated through simulations in a robot path planning environment, showing clear improvement over state-of-the-art methods.

**arXiv ID:** 2512.05447
</details>

<details>
<summary><strong>Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs</strong> - Yujie Zhao, Lanxiang Hu, Yang Wang, Minmin Hou, Hao Zhang, Ke Ding, Jishen Zhao - [[pdf]](https://arxiv.org/pdf/2510.11062)</summary>

**Abstract:** Multi-agent systems (MAS) and reinforcement learning (RL) are widely used to enhance the agentic capabilities of large language models (LLMs). MAS improves task performance through role-based orchestration, while RL uses environmental rewards to learn stronger policies, such as GRPO-style optimization. However, applying on-policy RL to MAS remains underexplored and presents unique challenges. Algorithmically, standard GRPO grouping assumptions break down because prompts vary by role and by turn. System-wise, the training stack must support MAS-workflow rollouts and on-policy updates for both single-policy and multi-policy models.
We propose AT-GRPO, which includes (i) an agent- and turn-wise grouped RL algorithm tailored to MAS and (ii) a training system that supports both single- and multi-policy regimes. Across game, planning, coding, and math tasks, AT-GRPO delivers substantial gains. On long-horizon planning, it increases accuracy from a 14.0 to 47.0 percent single-agent RL baseline to 96.0 to 99.5 percent. It also improves reasoning performance, with average gains of 3.87 to 7.62 percent on coding tasks and 9.0 to 17.93 percent on math. Code and environments are available at: this https URL.

**arXiv ID:** 2510.11062
</details>

<details>
<summary><strong>MedTutor-R1: Socratic Personalized Medical Teaching with Multi-Agent Simulation</strong> - Zhitao He, Haolin Yang, Zeyu Qin, Yi R Fung - [[pdf]](https://arxiv.org/pdf/2512.05671)</summary>

**Abstract:** The significant gap between rising demands for clinical training and the scarcity of expert instruction poses a major challenge to medical education. With powerful capabilities in personalized guidance, Large Language Models (LLMs) offer a promising solution to bridge this gap. However, current research focuses mainly on one-on-one knowledge instruction, overlooking collaborative reasoning, a key skill for students developed in teamwork like ward rounds. To this end, we develop ClinEdu, a multi-agent pedagogical simulator with personality-driven patients and diverse student cohorts, enabling controlled testing of complex pedagogical processes and scalable generation of teaching data. Based on ClinEdu, we construct ClinTeach, a large Socratic teaching dialogue dataset that captures the complexities of group instruction. We then train MedTutor-R1, the first multimodal Socratic tutor designed for one-to-many instruction in clinical medical education. MedTutor-R1 is first instruction-tuned on our ClinTeach dataset and then optimized with reinforcement learning, using rewards derived from a three-axis rubric, covering structural fidelity, analytical quality, and clinical safety, to refine its adaptive Socratic strategies. For authentic in-situ assessment, we use simulation-based interactive evaluation that redeploys the tutor back into ClinEdu. Experimental results demonstrate that our MedTutor-R1 outperforms the base model by over 20% in average pedagogical score and is comparable to o3, while also exhibiting high adaptability in handling a varying number of students. This promising performance underscores the effectiveness of our pedagogical simulator, ClinEdu.

**arXiv ID:** 2512.05671
</details>

<details>
<summary><strong>GRASP: Graph Reasoning Agents for Systems Pharmacology with Human-in-the-Loop</strong> - Omid Bazgir, Vineeth Manthapuri, Ilia Rattsev, Mohammad Jafarnejad - [[pdf]](https://arxiv.org/pdf/2512.05502)</summary>

**Abstract:** Quantitative Systems Pharmacology (QSP) modeling is essential for drug development but it requires significant time investment that limits the throughput of domain experts. We present \textbf{GRASP} -- a multi-agent, graph-reasoning framework with a human-in-the-loop conversational interface -- that encodes QSP models as typed biological knowledge graphs and compiles them to executable MATLAB/SimBiology code while preserving units, mass balance, and physiological constraints. A two-phase workflow -- \textsc{Understanding} (graph reconstruction of legacy code) and \textsc{Action} (constraint-checked, language-driven modification) -- is orchestrated by a state machine with iterative validation. GRASP performs breadth-first parameter-alignment around new entities to surface dependent quantities and propose biologically plausible defaults, and it runs automatic execution/diagnostics until convergence. In head-to-head evaluations using LLM-as-judge, GRASP outperforms SME-guided CoT and ToT baselines across biological plausibility, mathematical correctness, structural fidelity, and code quality (\(\approx\)9--10/10 vs.\ 5--7/10). BFS alignment achieves F1 = 0.95 for dependency discovery, units, and range. These results demonstrate that graph-structured, agentic workflows can make QSP model development both accessible and rigorous, enabling domain experts to specify mechanisms in natural language without sacrificing biomedical fidelity.

**arXiv ID:** 2512.05502
</details>

<details>
<summary><strong>Toward Efficient and Robust Behavior Models for Multi-Agent Driving Simulation</strong> - Fabian Konstantinidis, Moritz Sackmann, Ulrich Hofmann, Christoph Stiller - [[pdf]](https://arxiv.org/pdf/2512.05812)</summary>

**Abstract:** Scalable multi-agent driving simulation requires behavior models that are both realistic and computationally efficient. We address this by optimizing the behavior model that controls individual traffic participants. To improve efficiency, we adopt an instance-centric scene representation, where each traffic participant and map element is modeled in its own local coordinate frame. This design enables efficient, viewpoint-invariant scene encoding and allows static map tokens to be reused across simulation steps. To model interactions, we employ a query-centric symmetric context encoder with relative positional encodings between local frames. We use Adversarial Inverse Reinforcement Learning to learn the behavior model and propose an adaptive reward transformation that automatically balances robustness and realism during training. Experiments demonstrate that our approach scales efficiently with the number of tokens, significantly reducing training and inference times, while outperforming several agent-centric baselines in terms of positional accuracy and robustness.

**arXiv ID:** 2512.05812
</details>

<details>
<summary><strong>Optimal Safety-Aware Scheduling for Multi-Agent Aerial 3D Printing with Utility Maximization under Dependency Constraints</strong> - Marios-Nektarios Stamatopoulos, Shridhar Velhal, Avijit Banerjee, George Nikolakopoulos - [[pdf]](https://arxiv.org/pdf/2512.05815)</summary>

**Abstract:** This article presents a novel coordination and task-planning framework to enable the simultaneous conflict-free collaboration of multiple unmanned aerial vehicles (UAVs) for aerial 3D printing. The proposed framework formulates an optimization problem that takes a construction mission divided into sub-tasks and a team of autonomous UAVs, along with limited volume and battery. It generates an optimal mission plan comprising task assignments and scheduling while accounting for task dependencies arising from the geometric and structural requirements of the 3D design, inter-UAV safety constraints, material usage, and total flight time of each UAV. The potential conflicts occurring during the simultaneous operation of the UAVs are addressed at a segment level by dynamically selecting the starting time and location of each task to guarantee collision-free parallel execution. An importance prioritization is proposed to accelerate the computation by guiding the solution toward more important tasks. Additionally, a utility maximization formulation is proposed to dynamically determine the optimal number of UAVs required for a given mission, balancing the trade-off between minimizing makespan and the deployment of excess agents. The proposed framework's effectiveness is evaluated through a Gazebo-based simulation setup, where agents are coordinated by a mission control module allocating the printing tasks based on the generated optimal scheduling plan while remaining within the material and battery constraints of each UAV.

**arXiv ID:** 2512.05815
</details>

</details>

<details open>
<summary><h2>Other Agent Research (4 papers)</h2></summary>

<details>
<summary><strong>CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning</strong> - Ting-Ting Xie, Yixin Zhang - [[pdf]](https://arxiv.org/pdf/2512.05576)</summary>

**Abstract:** Current clinical agent built on small LLMs, such as TxAgent suffer from a \textit{Context Utilization Failure}, where models successfully retrieve biomedical evidence due to supervised finetuning but fail to ground their diagnosis in that information. In this work, we propose the Executor-Analyst Framework, a modular architecture that decouples the syntactic precision of tool execution from the semantic robustness of clinical reasoning. By orchestrating specialized TxAgents (Executors) with long-context foundation models (Analysts), we mitigate the reasoning deficits observed in monolithic models. Beyond simple modularity, we demonstrate that a Stratified Ensemble strategy significantly outperforms global pooling by preserving evidentiary diversity, effectively addressing the information bottleneck. Furthermore, our stress tests reveal critical scaling insights: (1) a \textit{Context-Performance Paradox}, where extending reasoning contexts beyond 12k tokens introduces noise that degrades accuracy; and (2) the \textit{Curse of Dimensionality} in action spaces, where expanding toolsets necessitates hierarchical retrieval strategies. Crucially, our approach underscores the potential of training-free architectural engineering, achieving state-of-the-art performance on CURE-Bench without the need for expensive end-to-end finetuning. This provides a scalable, agile foundation for the next generation of trustworthy AI-driven therapeutics. Code has been released on this https URL.

**arXiv ID:** 2512.05576
</details>

<details>
<summary><strong>Multimodal Oncology Agent for IDH1 Mutation Prediction in Low-Grade Glioma</strong> - Hafsa Akebli, Adam Shephard, Vincenzo Della Mea, Nasir Rajpoot - [[pdf]](https://arxiv.org/pdf/2512.05824)</summary>

**Abstract:** Low-grade gliomas frequently present IDH1 mutations that define clinically distinct subgroups with specific prognostic and therapeutic implications. This work introduces a Multimodal Oncology Agent (MOA) integrating a histology tool based on the TITAN foundation model for IDH1 mutation prediction in low-grade glioma, combined with reasoning over structured clinical and genomic inputs through PubMed, Google Search, and OncoKB. MOA reports were quantitatively evaluated on 488 patients from the TCGA-LGG cohort against clinical and histology baselines. MOA without the histology tool outperformed the clinical baseline, achieving an F1-score of 0.826 compared to 0.798. When fused with histology features, MOA reached the highest performance with an F1-score of 0.912, exceeding both the histology baseline at 0.894 and the fused histology-clinical baseline at 0.897. These results demonstrate that the proposed agent captures complementary mutation-relevant information enriched through external biomedical sources, enabling accurate IDH1 mutation prediction.

**arXiv ID:** 2512.05824
</details>

<details>
<summary><strong>Towards agent-based-model informed neural networks</strong> - Nino Antulov-Fantulin - [[pdf]](https://arxiv.org/pdf/2512.05764)</summary>

**Abstract:** In this article, we present a framework for designing neural networks that remain consistent with the underlying principles of agent-based models. We begin by highlighting the limitations of standard neural differential equations in modeling complex systems, where physical invariants (like energy) are often absent but other constraints (like mass conservation, network locality, bounded rationality) must be enforced. To address this, we introduce Agent-Based-Model informed Neural Networks(ABM-NNs), which leverage restricted graph neural networks and hierarchical decomposition to learn interpretable, structure-preserving dynamics. We validate the framework across three case studies of increasing complexity: (i) a generalized Generalized Lotka--Volterra system, where we recover ground-truth parameters from short trajectories in presence of interventions; (ii) a graph-based SIR contagion model, where our method outperforms state-of-the-art graph learning baselines (GCN, GraphSAGE, Graph Transformer) in out-of-sample forecasting and noise robustness; and (iii) a real-world macroeconomic model of the ten largest economies, where we learn coupled GDP dynamics from empirical data and demonstrate gradient-based counterfactual analysis for policy interventions.

**arXiv ID:** 2512.05764
</details>

<details>
<summary><strong>Over-the-Air Semantic Alignment with Stacked Intelligent Metasurfaces</strong> - Mario Edoardo Pandolfo, Kyriakos Stylianopoulos, George C. Alexandropoulos, Paolo Di Lorenzo - [[pdf]](https://arxiv.org/pdf/2512.05657)</summary>

**Abstract:** Semantic communication systems aim to transmit task-relevant information between devices capable of artificial intelligence, but their performance can degrade when heterogeneous transmitter-receiver models produce misaligned latent representations. Existing semantic alignment methods typically rely on additional digital processing at the transmitter or receiver, increasing overall device complexity. In this work, we introduce the first over-the-air semantic alignment framework based on stacked intelligent metasurfaces (SIM), which enables latent-space alignment directly in the wave domain, reducing substantially the computational burden at the device level. We model SIMs as trainable linear operators capable of emulating both supervised linear aligners and zero-shot Parseval-frame-based equalizers. To realize these operators physically, we develop a gradient-based optimization procedure that tailors the metasurface transfer function to a desired semantic mapping. Experiments with heterogeneous vision transformer (ViT) encoders show that SIMs can accurately reproduce both supervised and zero-shot semantic equalizers, achieving up to 90% task accuracy in regimes with high signal-to-noise ratio (SNR), while maintaining strong robustness even at low SNR values.

**arXiv ID:** 2512.05657
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (11 papers)</h2></summary>

<details>
<summary><strong>MCP-AI: Protocol-Driven Intelligence Framework for Autonomous Reasoning in Healthcare</strong> - Zag ElSayed, Craig Erickson, Ernest Pedapati - [[pdf]](https://arxiv.org/pdf/2512.05365)</summary>

**Abstract:** Healthcare AI systems have historically faced challenges in merging contextual reasoning, long-term state management, and human-verifiable workflows into a cohesive framework. This paper introduces a completely innovative architecture and concept: combining the Model Context Protocol (MCP) with a specific clinical application, known as MCP-AI. This integration allows intelligent agents to reason over extended periods, collaborate securely, and adhere to authentic clinical logic, representing a significant shift away from traditional Clinical Decision Support Systems (CDSS) and prompt-based Large Language Models (LLMs). As healthcare systems become more complex, the need for autonomous, context-aware clinical reasoning frameworks has become urgent. We present MCP-AI, a novel architecture for explainable medical decision-making built upon the Model Context Protocol (MCP) a modular, executable specification for orchestrating generative and descriptive AI agents in real-time workflows. Each MCP file captures clinical objectives, patient context, reasoning state, and task logic, forming a reusable and auditable memory object. Unlike conventional CDSS or stateless prompt-based AI systems, MCP-AI supports adaptive, longitudinal, and collaborative reasoning across care settings. MCP-AI is validated through two use cases: (1) diagnostic modeling of Fragile X Syndrome with comorbid depression, and (2) remote coordination for Type 2 Diabetes and hypertension. In either scenario, the protocol facilitates physician-in-the-loop validation, streamlines clinical processes, and guarantees secure transitions of AI responsibilities between healthcare providers. The system connects with HL7/FHIR interfaces and adheres to regulatory standards, such as HIPAA and FDA SaMD guidelines. MCP-AI provides a scalable basis for interpretable, composable, and safety-oriented AI within upcoming clinical environments.

**arXiv ID:** 2512.05365
</details>

<details>
<summary><strong>A Fast Anti-Jamming Cognitive Radar Deployment Algorithm Based on Reinforcement Learning</strong> - Wencheng Cai, Xuchao Gao, Congying Han, Mingqiang Li, Tiande Guo - [[pdf]](https://arxiv.org/pdf/2512.05753)</summary>

**Abstract:** The fast deployment of cognitive radar to counter jamming remains a critical challenge in modern warfare, where more efficient deployment leads to quicker detection of targets. Existing methods are primarily based on evolutionary algorithms, which are time-consuming and prone to falling into local optima. We tackle these drawbacks via the efficient inference of neural networks and propose a brand new framework: Fast Anti-Jamming Radar Deployment Algorithm (FARDA). We first model the radar deployment problem as an end-to-end task and design deep reinforcement learning algorithms to solve it, where we develop integrated neural modules to perceive heatmap information and a brand new reward format. Empirical results demonstrate that our method achieves coverage comparable to evolutionary algorithms while deploying radars approximately 7,000 times faster. Further ablation experiments confirm the necessity of each component of FARDA.

**arXiv ID:** 2512.05753
</details>

<details>
<summary><strong>Semore: VLM-guided Enhanced Semantic Motion Representations for Visual Reinforcement Learning</strong> - Wentao Wang, Chunyang Liu, Kehua Sheng, Bo Zhang, Yan Wang - [[pdf]](https://arxiv.org/pdf/2512.05172)</summary>

**Abstract:** The growing exploration of Large Language Models (LLM) and Vision-Language Models (VLM) has opened avenues for enhancing the effectiveness of reinforcement learning (RL). However, existing LLM-based RL methods often focus on the guidance of control policy and encounter the challenge of limited representations of the backbone networks. To tackle this problem, we introduce Enhanced Semantic Motion Representations (Semore), a new VLM-based framework for visual RL, which can simultaneously extract semantic and motion representations through a dual-path backbone from the RGB flows. Semore utilizes VLM with common-sense knowledge to retrieve key information from observations, while using the pre-trained clip to achieve the text-image alignment, thereby embedding the ground-truth representations into the backbone. To efficiently fuse semantic and motion representations for decision-making, our method adopts a separately supervised approach to simultaneously guide the extraction of semantics and motion, while allowing them to interact spontaneously. Extensive experiments demonstrate that, under the guidance of VLM at the feature level, our method exhibits efficient and adaptive ability compared to state-of-art methods. All codes are released.

**arXiv ID:** 2512.05172
</details>

<details>
<summary><strong>Please Don't Kill My Vibe: Empowering Agents with Data Flow Control</strong> - Charlie Summers, Haneen Mohammed, Eugene Wu - [[pdf]](https://arxiv.org/pdf/2512.05374)</summary>

**Abstract:** The promise of Large Language Model (LLM) agents is to perform complex, stateful tasks. This promise is stunted by significant risks - policy violations, process corruption, and security flaws - that stem from the lack of visibility and mechanisms to manage undesirable data flows produced by agent actions. Today, agent workflows are responsible for enforcing these policies in ad hoc ways. Just as data validation and access controls shifted from the application to the DBMS, freeing application developers from these concerns, we argue that systems should support Data Flow Controls (DFCs) and enforce DFC policies natively. This paper describes early work developing a portable instance of DFC for DBMSes and outlines a broader research agenda toward DFC for agent ecosystems.

**arXiv ID:** 2512.05374
</details>

<details>
<summary><strong>Bayesian Active Inference for Intelligent UAV Anti-Jamming and Adaptive Trajectory Planning</strong> - Ali Krayani, Seyedeh Fatemeh Sadati, Lucio Marcenaro, Carlo Regazzoni - [[pdf]](https://arxiv.org/pdf/2512.05711)</summary>

**Abstract:** This paper proposes a hierarchical trajectory planning framework for UAVs operating under adversarial jamming conditions. Leveraging Bayesian Active Inference, the approach combines expert-generated demonstrations with probabilistic generative modeling to encode high-level symbolic planning, low-level motion policies, and wireless signal feedback. During deployment, the UAV performs online inference to anticipate interference, localize jammers, and adapt its trajectory accordingly, without prior knowledge of jammer locations. Simulation results demonstrate that the proposed method achieves near-expert performance, significantly reducing communication interference and mission cost compared to model-free reinforcement learning baselines, while maintaining robust generalization in dynamic environments.

**arXiv ID:** 2512.05711
</details>

<details>
<summary><strong>GTM: Simulating the World of Tools for AI Agents</strong> - Zhenzhen Ren, Xinpeng Zhang, Zhenxing Qian, Yan Gao, Yu Shi, Shuxin Zheng, Jiyan He - [[pdf]](https://arxiv.org/pdf/2512.04535)</summary>

**Abstract:** The integration of external tools is pivotal for empowering Large Language Model (LLM) agents with real-world capabilities. However, training these agents through direct, continuous interaction with diverse tools is often prohibitively expensive, slow, and introduces additional development and maintenance overhead. To address this challenge, we introduce the Generalist Tool Model (GTM), a 1.5-billion-parameter model that learns to act as a universal tool simulator. With only prompt-level configuration, GTM accesses tool functionalities along with input arguments and generates outputs that faithfully mimic real tool execution, providing a fast and cost-effective solution that eliminates development overhead. To build GTM, we propose the Context-Aware Response Generation (CARG) pipeline, which synthesizes comprehensive training data covering over 20,000 tools across 300 domains including physics, medicine, robotics, and finance. Through this pipeline, GTM learns to produce not only syntactically correct outputs but also logically coherent and contextually appropriate responses. Experiments demonstrate that GTM produces high-quality outputs with strong consistency and reliability. Besides when used in real reinforcement learning scenarios for agent training, GTM exhibits significantly faster simulation speed compared to real tools while maintaining comparable output quality, along with remarkable generalization and domain adaptability. Our results establish GTM as a foundational component for developing future AI agents, enabling efficient and scalable training of tool-augmented systems.

**arXiv ID:** 2512.04535
</details>

<details>
<summary><strong>Hierarchical Reinforcement Learning for the Dynamic VNE with Alternatives Problem</strong> - Ali Al Housseini, Cristina Rottondi, Omran Ayoub - [[pdf]](https://arxiv.org/pdf/2512.05207)</summary>

**Abstract:** Virtual Network Embedding (VNE) is a key enabler of network slicing, yet most formulations assume that each Virtual Network Request (VNR) has a fixed topology. Recently, VNE with Alternative topologies (VNEAP) was introduced to capture malleable VNRs, where each request can be instantiated using one of several functionally equivalent topologies that trade resources differently. While this flexibility enlarges the feasible space, it also introduces an additional decision layer, making dynamic embedding more challenging. This paper proposes HRL-VNEAP, a hierarchical reinforcement learning approach for VNEAP under dynamic arrivals. A high-level policy selects the most suitable alternative topology (or rejects the request), and a low-level policy embeds the chosen topology onto the substrate network. Experiments on realistic substrate topologies under multiple traffic loads show that naive exploitation strategies provide only modest gains, whereas HRL-VNEAP consistently achieves the best performance across all metrics. Compared to the strongest tested baselines, HRL-VNEAP improves acceptance ratio by up to \textbf{20.7\%}, total revenue by up to \textbf{36.2\%}, and revenue-over-cost by up to \textbf{22.1\%}. Finally, we benchmark against an MILP formulation on tractable instances to quantify the remaining gap to optimality and motivate future work on learning- and optimization-based VNEAP solutions.

**arXiv ID:** 2512.05207
</details>

<details>
<summary><strong>Entropy Ratio Clipping as a Soft Global Constraint for Stable Reinforcement Learning</strong> - Zhenpeng Su, Leiyu Pan, Minxuan Lv, Tiehua Mei, Zijia Lin, Yuntao Li, Wenping Hu, Ruiming Tang, Kun Gai, Guorui Zhou - [[pdf]](https://arxiv.org/pdf/2512.05591)</summary>

**Abstract:** Large language model post-training relies on reinforcement learning to improve model capability and alignment quality. However, the off-policy training paradigm introduces distribution shift, which often pushes the policy beyond the trust region, leading to training instabilities manifested as fluctuations in policy entropy and unstable gradients. Although PPO-Clip mitigates this issue through importance clipping, it still overlooks the global distributional shift of actions. To address these challenges, we propose using the entropy ratio between the current and previous policies as a new global metric that effectively quantifies the relative change in policy exploration throughout updates. Building on this metric, we introduce an \textbf{Entropy Ratio Clipping} (ERC) mechanism that imposes bidirectional constraints on the entropy ratio. This stabilizes policy updates at the global distribution level and compensates for the inability of PPO-clip to regulate probability shifts of un-sampled actions. We integrate ERC into both DAPO and GPPO reinforcement learning algorithms. Experiments across multiple benchmarks show that ERC consistently improves performance.

**arXiv ID:** 2512.05591
</details>

<details>
<summary><strong>Towards a Generalisable Cyber Defence Agent for Real-World Computer Networks</strong> - Tim Dudman, Martyn Bull - [[pdf]](https://arxiv.org/pdf/2511.09114)</summary>

**Abstract:** Recent advances in deep reinforcement learning for autonomous cyber defence have resulted in agents that can successfully defend simulated computer networks against cyber-attacks. However, many of these agents would need retraining to defend networks with differing topology or size, making them poorly suited to real-world networks where topology and size can vary over time. In this research we introduce a novel set of Topological Extensions for Reinforcement Learning Agents (TERLA) that provide generalisability for the defence of networks with differing topology and size, without the need for retraining. Our approach involves the use of heterogeneous graph neural network layers to produce a fixed-size latent embedding representing the observed network state. This representation learning stage is coupled with a reduced, fixed-size, semantically meaningful and interpretable action space. We apply TERLA to a standard deep reinforcement learning Proximal Policy Optimisation (PPO) agent model, and to reduce the sim-to-real gap, conduct our research using Cyber Autonomy Gym for Experimentation (CAGE) Challenge 4. This Cyber Operations Research Gym environment has many of the features of a real-world network, such as realistic Intrusion Detection System (IDS) events and multiple agents defending network segments of differing topology and size. TERLA agents retain the defensive performance of vanilla PPO agents whilst showing improved action efficiency. Generalisability has been demonstrated by showing that all TERLA agents have the same network-agnostic neural network architecture, and by deploying a single TERLA agent multiple times to defend network segments with differing topology and size, showing improved defensive performance and efficiency.

**arXiv ID:** 2511.09114
</details>

<details>
<summary><strong>EnhancedRL: An Enhanced-State Reinforcement Learning Algorithm for Multi-Task Fusion in Recommender Systems</strong> - Peng Liu, Cong Xu, Jiawei Zhu, Ming Zhao, Bin Wang - [[pdf]](https://arxiv.org/pdf/2409.11678)</summary>

**Abstract:** As a key stage of Recommender Systems (RSs), Multi-Task Fusion (MTF) is responsible for merging multiple scores output by Multi-Task Learning (MTL) into a single score, finally determining the recommendation results. Recently, Reinforcement Learning (RL) has been applied to MTF to maximize long-term user satisfaction within a recommendation session. However, due to limitations in modeling paradigm, all existing RL algorithms for MTF can only utilize user features and statistical features as the state to generate actions at the user level, but unable to leverage item features and other valuable features, which leads to suboptimal performance. Overcoming this problem requires a breakthrough in the existing modeling paradigm, yet, to date, no prior work has addressed it. To tackle this challenge, we propose EnhancedRL, an innovative RL algorithm. Unlike existing RL-MTF methods, EnhancedRL takes the enhanced state as input, incorporating not only user features but also item features and other valuable information. Furthermore, it introduces a tailored actor-critic framework - including redesigned actor and critics and a novel learning procedure - to optimize long-term rewards at the user-item pair level within a recommendation session. Extensive offline and online experiments are conducted in an industrial RS and the results demonstrate that EnhancedRL outperforms other methods remarkably, achieving a +3.84% increase in user valid consumption and a +0.58% increase in user duration time. To the best of our knowledge, EnhancedRL is the first work to address this challenge, and it has been fully deployed in a large-scale RS since September 14, 2023, yielding significant improvements.

**arXiv ID:** 2409.11678
</details>

<details>
<summary><strong>Real-time Remote Tracking and Autonomous Planning for Whale Rendezvous using Robots</strong> - Sushmita Bhattacharya, Ninad Jadhav, Hammad Izhar, Karen Li, Kevin George, Robert Wood, Stephanie Gil - [[pdf]](https://arxiv.org/pdf/2512.05808)</summary>

**Abstract:** We introduce a system for real-time sperm whale rendezvous at sea using an autonomous uncrewed aerial vehicle. Our system employs model-based reinforcement learning that combines in situ sensor data with an empirical whale dive model to guide navigation decisions. Key challenges include (i) real-time acoustic tracking in the presence of multiple whales, (ii) distributed communication and decision-making for robot deployments, and (iii) on-board signal processing and long-range detection from fish-trackers. We evaluate our system by conducting rendezvous with sperm whales at sea in Dominica, performing hardware experiments on land, and running simulations using whale trajectories interpolated from marine biologists' surface observations.

**arXiv ID:** 2512.05808
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
