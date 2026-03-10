# Agent arXiv Daily

**Last Updated:** 2026-03-10 02:49:11

**Total Papers:** 76

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
<summary><strong>Agent Tools Orchestration Leaks More: Dataset, Benchmark, and Mitigation</strong> - Yuxuan Qiao, Dongqin Liu, Hongchang Yang, Wei Zhou, Songlin Hu - [[pdf]](https://arxiv.org/pdf/2512.16310)</summary>

**Abstract:** Driven by Large Language Models, the single-agent, multi-tool architecture has become a popular paradigm for autonomous agents. However, this architecture introduces a severe privacy risk, which we term Tools Orchestration Privacy Risk (TOP-R): an agent, to achieve a benign user goal, autonomously aggregates non-sensitive fragments from multiple tools and synthesizes unexpected sensitive information. We provide the first systematic study of this risk. We establish a formal framework characterizing TOP-R through three necessary conditions -- conclusion sensitivity, single-source non-inferability, and compositional inferability. We construct TOP-Bench via a Reverse Inference Seed Expansion (RISE) pipeline, incorporating paired social-context scenarios for diagnostic analysis. We further introduce the H-Score, a harmonic mean of task completion and safety, to quantify the utility-safety trade-off. Evaluation of six state-of-the-art LLMs reveals pervasive risk: the average Overall Leakage Rate reaches 62.11% with an H-Score of only 52.90%. Our experiments identify three root causes: deficient spontaneous privacy awareness, reasoning overshoot, and inference inertia. Guided by these findings, we propose three complementary mitigation strategies targeting the output, reasoning, and review stages of the agent pipeline; the strongest configuration, Dual-Constraint Privacy Enhancement, achieves an H-Score of 79.20%. Our work reveals a new risk class in tool-using agents, analyzes leakage causes, and provides practical mitigation strategies.

**arXiv ID:** 2512.16310
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (17 papers)</h2></summary>

<details>
<summary><strong>Evolving Medical Imaging Agents via Experience-driven Self-skill Discovery</strong> - Lin Fan, Pengyu Dai, Zhipeng Deng, Haolin Wang, Xun Gong, Yefeng Zheng, Yafei Ou - [[pdf]](https://arxiv.org/pdf/2603.05860)</summary>

**Abstract:** Clinical image interpretation is inherently multi-step and tool-centric: clinicians iteratively combine visual evidence with patient context, quantify findings, and refine their decisions through a sequence of specialized procedures. While LLM-based agents promise to orchestrate such heterogeneous medical tools, existing systems treat tool sets and invocation strategies as static after deployment. This design is brittle under real-world domain shifts, across tasks, and evolving diagnostic requirements, where predefined tool chains frequently degrade and demand costly manual re-design. We propose MACRO, a self-evolving, experience-augmented medical agent that shifts from static tool composition to experience-driven tool discovery. From verified execution trajectories, the agent autonomously identifies recurring effective multi-step tool sequences, synthesizes them into reusable composite tools, and registers these as new high-level primitives that continuously expand its behavioral repertoire. A lightweight image-feature memory grounds tool selection in a visual-clinical context, while a GRPO-like training loop reinforces reliable invocation of discovered composites, enabling closed-loop self-improvement with minimal supervision. Extensive experiments across diverse medical imaging datasets and tasks demonstrate that autonomous composite tool discovery consistently improves multi-step orchestration accuracy and cross-domain generalization over strong baselines and recent state-of-the-art agentic methods, bridging the gap between brittle static tool use and adaptive, context-aware clinical AI assistance. Code will be available upon acceptance.

**arXiv ID:** 2603.05860
</details>

<details>
<summary><strong>The World Won't Stay Still: Programmable Evolution for Agent Benchmarks</strong> - Guangrui Li, Yaochen Xie, Yi Liu, Ziwei Dong, Xingyuan Pan, Tianqi Zheng, Jason Choi, Michael J. Morais, Binit Jha, Shaunak Mishra, Bingrou Zhou, Chen Luo, Monica Xiao Cheng, Dawn Song - [[pdf]](https://arxiv.org/pdf/2603.05910)</summary>

**Abstract:** LLM-powered agents fulfill user requests by interacting with environments, querying data, and invoking tools in a multi-turn process. Yet, most existing benchmarks assume static environments with fixed schemas and toolsets, neglecting the evolutionary nature of the real world and agents' robustness to environmental changes. In this paper, we study a crucial problem: how to evolve the agent environment in a scalable and controllable way, thereby better evaluating agents' adaptability to real-world dynamics. We propose ProEvolve, a graph-based framework that makes environment evolution programmable. At its core, a typed relational graph provides a unified, explicit representation of the environment: data, tools, and schema. Under this formalism, adding, removing, or modifying capabilities are expressed as graph transformations that coherently propagate updates across tools, schemas, and data access. Building on this, ProEvolve can (1) program the evolutionary dynamics as graph transformations to generate environments automatically, and (2) instantiate task sandboxes via subgraph sampling and programming. We validate ProEvolve by evolving a single environment into 200 environments and 3,000 task sandboxes, and benchmark representative agents accordingly.

**arXiv ID:** 2603.05910
</details>

<details>
<summary><strong>Tool-Genesis: A Task-Driven Tool Creation Benchmark for Self-Evolving Language Agent</strong> - Bowei Xia, Mengkang Hu, Shijian Wang, Jiarui Jin, Wenxiang Jiao, Yuan Lu, Kexin Li, Ping Luo - [[pdf]](https://arxiv.org/pdf/2603.05578)</summary>

**Abstract:** Research on self-evolving language agents has accelerated, drawing increasing attention to their ability to create, adapt, and maintain tools from task requirements. However, existing benchmarks predominantly rely on predefined specifications, which limits scalability and hinders truly autonomous evolution. While recent studies attempt to dynamically generate tools, they primarily emphasize downstream performance, resulting in a "black-box" evaluation that makes it difficult to attribute failures to specific causes. To address this, we propose Tool-Genesis, a diagnostic benchmark designed to quantify agent capabilities across multiple dimensions, including interface compliance, functional correctness, and downstream utility. Tool-Genesis evaluates whether agents can construct task-relevant tools solely from abstract requirements (without preset specifications) and use them to solve realistic problems. Crucially, we find that even state-of-the-art models struggle to produce precise tool interfaces or executable logic in a one-shot setting. These minor initial flaws are amplified through the pipeline, leading to a sharp degradation in downstream metrics. We hope Tool-Genesis will guide future research toward training and steering models to synthesize persistent, general-purpose tools that better address real-world challenges.

**arXiv ID:** 2603.05578
</details>

<details>
<summary><strong>Autonomous Algorithm Discovery for Ptychography via Evolutionary LLM Reasoning</strong> - Xiangyu Yin, Ming Du, Junjing Deng, Zhi Yang, Yimo Han, Yi Jiang - [[pdf]](https://arxiv.org/pdf/2603.05696)</summary>

**Abstract:** Ptychography is a computational imaging technique widely used for high-resolution materials characterization, but high-quality reconstructions often require the use of regularization functions that largely remain manually designed. We introduce Ptychi-Evolve, an autonomous framework that uses large language models (LLMs) to discover and evolve novel regularization algorithms. The framework combines LLM-driven code generation with evolutionary mechanisms, including semantically-guided crossover and mutation. Experiments on three challenging datasets (X-ray integrated circuits, low-dose electron microscopy of apoferritin, and multislice imaging with crosstalk artifacts) demonstrate that discovered regularizers outperform conventional reconstructions, achieving up to +0.26 SSIM and +8.3~dB PSNR improvements. Besides, Ptychi-Evolve records algorithm lineage and evolution metadata, enabling interpretable and reproducible analysis of discovered regularizers.

**arXiv ID:** 2603.05696
</details>

<details>
<summary><strong>TML-Bench: Benchmark for Data Science Agents on Tabular ML Tasks</strong> - Mykola Pinchuk - [[pdf]](https://arxiv.org/pdf/2603.05764)</summary>

**Abstract:** Autonomous coding agents can produce strong tabular baselines quickly on Kaggle-style tasks. Practical value depends on end-to-end correctness and reliability under time limits. This paper introduces TML-Bench, a tabular benchmark for data science agents on Kaggle-style tasks. This paper evaluates 10 OSS LLMs on four Kaggle competitions and three time budgets (240s, 600s, and 1200s). Each model is run five times per task and budget. A run is successful if it produces a valid submission and a private-holdout score on hidden labels that are not accessible to the agent. This paper reports median performance, success rates, and run-to-run variability. MiniMax-M2.1 model achieves the best aggregate performance score on all four competitions under the paper's primary aggregation. Average performance improves with larger time budgets. Scaling is noisy for some individual models at the current run count. Code and materials are available at this https URL.

**arXiv ID:** 2603.05764
</details>

<details>
<summary><strong>Computational Pathology in the Era of Emerging Foundation and Agentic AI -- International Expert Perspectives on Clinical Integration and Translational Readiness</strong> - Qian Da, Yijiang Chen, Min Ju, Zheyi Ji, Albert Zhou, Wenwen Wang, Matthew A Abikenari, Philip Chikontwe, Guillaume Larghero, Bowen Chen, Peter Neiglinger, Dingrong Zhong, Shuhao Wang, Wei Xu, Drew Williamson, German Corredor, Sen Yang, Le Lu, Xiao Han, Kun-Hsing Yu, Jun-zhou Huang, Laura Barisoni, Geert Litjens, Anant Madabhushi, Lifeng Zhu, Chaofu Wang, Junhan Zhao, Weiguo Hu - [[pdf]](https://arxiv.org/pdf/2603.05884)</summary>

**Abstract:** Recent breakthroughs in artificial intelligence through foundation models and agents have accelerated the evolution of computational pathology. Demonstrated performance gains reported across academia in benchmarking datasets in predictive tasks such as diagnosis, prognosis, and treatment response have ignited substantial enthusiasm for clinical application. Despite this development momentum, real world adoption has lagged, as implementation faces economic, technical, and administrative challenges. Beyond existing discussions of technical architectures and comparative performance, this review considers how these emerging AI systems can be responsibly integrated into medical practice by connecting deployable clinical relevance with downstream analytical capabilities and their technical maturity, operational readiness, and economic and regulatory context. Drawing on perspectives from an international group, we provide a practical assessment of current capabilities and barriers to adoption in patient care settings.

**arXiv ID:** 2603.05884
</details>

<details>
<summary><strong>From Features to Actions: Explainability in Traditional and Agentic AI Systems</strong> - Sindhuja Chaduvula, Jessee Ho, Kina Kim, Aravind Narayanan, Mahshid Alinoori, Muskan Garg, Dhanesh Ramachandram, Shaina Raza - [[pdf]](https://arxiv.org/pdf/2602.06841)</summary>

**Abstract:** Over the last decade, explainable AI has primarily focused on interpreting individual model predictions, producing post-hoc explanations that relate inputs to outputs under a fixed decision structure. Recent advances in large language models (LLMs) have enabled agentic AI systems whose behaviour unfolds over multi-step trajectories. In these settings, success and failure are determined by sequences of decisions rather than a single output. While useful, it remains unclear how explanation approaches designed for static predictions translate to agentic settings where behaviour emerges over time. In this work, we bridge the gap between static and agentic explainability by comparing attribution-based explanations with trace-based diagnostics across both settings. To make this distinction explicit, we empirically compare attribution-based explanations used in static classification tasks with trace-based diagnostics used in agentic benchmarks (TAU-bench Airline and AssistantBench). Our results show that while attribution methods achieve stable feature rankings in static settings (Spearman $\rho = 0.86$), they cannot be applied reliably to diagnose execution-level failures in agentic trajectories. In contrast, trace-grounded rubric evaluation for agentic settings consistently localizes behaviour breakdowns and reveals that state tracking inconsistency is 2.7$\times$ more prevalent in failed runs and reduces success probability by 49\%. These findings motivate a shift towards trajectory-level explainability for agentic systems when evaluating and diagnosing autonomous AI behaviour.
Resources:
this https URL this https URL

**arXiv ID:** 2602.06841
</details>

<details>
<summary><strong>How Well Does Agent Development Reflect Real-World Work?</strong> - Zora Zhiruo Wang, Sanidhya Vijayvargiya, Aspen Chen, Hanmo Zhang, Venu Arvind Arangarajan, Jett Chen, Valerie Chen, Diyi Yang, Daniel Fried, Graham Neubig - [[pdf]](https://arxiv.org/pdf/2603.01203)</summary>

**Abstract:** AI agents are increasingly developed and evaluated on benchmarks relevant to human work, yet it remains unclear how representative these benchmarking efforts are of the labor market as a whole. In this work, we systematically study the relationship between agent development efforts and the distribution of real-world human work by mapping benchmark instances to work domains and skills. We first analyze 43 benchmarks and 72,342 tasks, measuring their alignment with human employment and capital allocation across all 1,016 real-world occupations in the U.S. labor market. We reveal substantial mismatches between agent development that tends to be programming-centric, and the categories in which human labor and economic value are concentrated. Within work areas that agents currently target, we further characterize current agent utility by measuring their autonomy levels, providing practical guidance for agent interaction strategies across work scenarios. Building on these findings, we propose three measurable principles for designing benchmarks that better capture socially important and technically challenging forms of work: coverage, realism, and granular evaluation.

**arXiv ID:** 2603.01203
</details>

<details>
<summary><strong>MOOSEnger -- a Domain-Specific AI Agent for the MOOSE Ecosystem</strong> - Mengnan Li, Jason Miller, Zachary Prince, Alexander Lindsay, Cody Permann - [[pdf]](https://arxiv.org/pdf/2603.04756)</summary>

**Abstract:** MOOSEnger is a tool-enabled AI agent tailored to the Multiphysics Object-Oriented Simulation Environment (MOOSE). MOOSE cases are specified in HIT ".i" input files; the large object catalog and strict syntax make initial setup and debugging slow. MOOSEnger offers a conversational workflow that turns natural-language intent into runnable inputs by combining retrieval-augmented generation over curated docs/examples with deterministic, MOOSE-aware parsing, validation, and execution tools. A core-plus-domain architecture separates reusable agent infrastructure (configuration, registries, tool dispatch, retrieval services, persistence, and evaluation) from a MOOSE plugin that adds HIT-based parsing, syntax-preserving ingestion of input files, and domain-specific utilities for input repair and checking. An input precheck pipeline removes hidden formatting artifacts, fixes malformed HIT structure with a bounded grammar-constrained loop, and resolves invalid object types via similarity search over an application syntax registry. Inputs are then validated and optionally smoke-tested with the MOOSE runtime in the loop via an MCP-backed execution backend (with local fallback), translating solver diagnostics into iterative verify-and-correct updates. Built-in evaluation reports RAG metrics (faithfulness, relevancy, context precision/recall) and end-to-end success by actual execution. On a 125-prompt benchmark spanning diffusion, transient heat conduction, solid mechanics, porous flow, incompressible Navier--Stokes, phase field and plasticity, MOOSEnger achieves a 0.90 execution pass rate versus 0.06 for an LLM-only baseline.

**arXiv ID:** 2603.04756
</details>

<details>
<summary><strong>SEA-TS: Self-Evolving Agent for Autonomous Code Generation of Time Series Forecasting Algorithms</strong> - Longkun Xu, Xiaochun Zhang, Qiantu Tuo, Rui Li - [[pdf]](https://arxiv.org/pdf/2603.04873)</summary>

**Abstract:** Accurate time series forecasting underpins decision-making across domains, yet conventional ML development suffers from data scarcity in new deployments, poor adaptability under distribution shift, and diminishing returns from manual iteration. We propose Self-Evolving Agent for Time Series Algorithms (SEA-TS), a framework that autonomously generates, validates, and optimizes forecasting code via an iterative self-evolution loop. Our framework introduces three key innovations: (1) Metric-Advantage Monte Carlo Tree Search (MA-MCTS), which replaces fixed rewards with a normalized advantage score for discriminative search guidance; (2) Code Review with running prompt refinement, where each executed solution undergoes automated review followed by prompt updates that encode corrective patterns, preventing recurrence of similar errors; and (3) Global Steerable Reasoning, which compares each node against global best and worst solutions, enabling cross-trajectory knowledge transfer. We adopt a MAP-Elites archive for architectural diversity. On the public Solar-Energy benchmark, SEA-TS generated code achieves a 40% MAE reduction relative to TimeMixer, surpassing state-of-the-art methods. On proprietary datasets, SEA-TS generated code reduces WAPE by 8.6% on solar PV forecasting and 7.7% on residential load forecasting compared to human-engineered baselines, and achieves 26.17% MAPE on load forecasting versus 29.34% by TimeMixer. Notably, the evolved models discover novel architectural patterns--including physics-informed monotonic decay heads encoding solar irradiance constraints, per-station learned diurnal cycle profiles, and learnable hourly bias correction--demonstrating that autonomous ML engineering can generate genuinely novel algorithmic ideas beyond manual design.

**arXiv ID:** 2603.04873
</details>

<details>
<summary><strong>Software Development Life Cycle Perspective: A Survey of Benchmarks for Code Large Language Models and Agents</strong> - Kaixin Wang, Tianlin Li, Xiaoyu Zhang, Chong Wang, Weisong Sun, Yang Liu, Aishan Liu, Xianglong Liu, Chao Shen, Bin Shi - [[pdf]](https://arxiv.org/pdf/2505.05283)</summary>

**Abstract:** Code large language models (CodeLLMs) and agents are increasingly being integrated into complex software engineering tasks spanning the entire Software Development Life Cycle (SDLC). Benchmarking is critical for rigorously evaluating these capabilities. However, despite their growing significance, there remains a lack of comprehensive reviews that examine these benchmarks from an SDLC perspective. To bridge this gap, we propose a tiered analysis framework to systematically review 178 benchmarks from 461 papers, comprehensively characterizing them from the perspective of the SDLC. Our findings reveal a notable imbalance in the coverage of current benchmarks, with approximately 61\% focused on the software implementation phase in SDLC, while requirements engineering and software design phases receive minimal attention at only 5\% and 3\%, respectively. % Additionally, anti-contamination strategies are largely absent from current benchmarks, leading to an increased risk of data leakage. Furthermore, current benchmarks lack effective anti-contamination strategies, posing significant risks of data leakage and potentially inflated performance assessments. Finally, we identify key open challenges in current research and outline future directions to narrow the gap between the theoretical capabilities of CodeLLMs and agents and their practical effectiveness in real-world scenarios.

**arXiv ID:** 2505.05283
</details>

<details>
<summary><strong>Towards Autonomous Mathematics Research</strong> - Tony Feng, Trieu H. Trinh, Garrett Bingham, Dawsen Hwang, Yuri Chervonyi, Junehyuk Jung, Joonkyung Lee, Carlo Pagano, Sang-hyun Kim, Federico Pasqualotto, Sergei Gukov, Jonathan N. Lee, Junsu Kim, Kaiying Hou, Golnaz Ghiasi, Yi Tay, YaGuang Li, Chenkai Kuang, Yuan Liu, Hanzhao Lin, Evan Zheran Liu, Nigamaa Nayakanti, Xiaomeng Yang, Heng-Tze Cheng, Demis Hassabis, Koray Kavukcuoglu, Quoc V. Le, Thang Luong - [[pdf]](https://arxiv.org/pdf/2602.10177)</summary>

**Abstract:** Recent advances in foundational models have yielded reasoning systems capable of achieving a gold-medal standard at the International Mathematical Olympiad. The transition from competition-level problem-solving to professional research, however, requires navigating vast literature and constructing long-horizon proofs. In this work, we introduce Aletheia, a math research agent that iteratively generates, verifies, and revises solutions end-to-end in natural language. Specifically, Aletheia is powered by an advanced version of Gemini Deep Think for challenging reasoning problems, a novel inference-time scaling law that extends beyond Olympiad-level problems, and intensive tool use to navigate the complexities of mathematical research. We demonstrate the capability of Aletheia from Olympiad problems to PhD-level exercises and most notably, through several distinct milestones in AI-assisted mathematics research: (a) a research paper (Feng26) generated by AI without any human intervention in calculating certain structure constants in arithmetic geometry called eigenweights; (b) a research paper (LeeSeo26) demonstrating human-AI collaboration in proving bounds on systems of interacting particles called independent sets; and (c) an extensive semi-autonomous evaluation (Feng et al., 2026a) of 700 open problems on Bloom's Erdos Conjectures database, including autonomous solutions to four open questions. In order to help the public better understand the developments pertaining to AI and mathematics, we suggest quantifying standard levels of autonomy and novelty of AI-assisted results, as well as propose a novel concept of human-AI interaction cards for transparency. We conclude with reflections on human-AI collaboration in mathematics and share all prompts as well as model outputs at this https URL.

**arXiv ID:** 2602.10177
</details>

<details>
<summary><strong>"When to Hand Off, When to Work Together": Expanding Human-Agent Co-Creative Collaboration through Concurrent Interaction</strong> - Kihoon Son, Hyewon Lee, DaEun Choi, Yoonsu Kim, Tae Soo Kim, Yoonjoo Lee, John Joon Young Chung, HyunJoon Jung, Juho Kim - [[pdf]](https://arxiv.org/pdf/2603.02050)</summary>

**Abstract:** Human collaborators coordinate dynamically through process visibility and workspace awareness, yet AI agents typically either provide only final outputs or expose read-only execution processes (e.g., planning, reasoning) without interpreting concurrent user actions on shared artifacts. Building on mixed-initiative interaction principles, we explore whether agents can achieve collaborative context awareness -- interpreting concurrent user actions on shared artifacts and adapting in real-time. Study 1 (N=10 professional designers) revealed that process visibility enabled reasoning about agent actions but exposed conflicts when agents could not distinguish feedback from independent work. We developed CLEO, which interprets collaborative intent and adapts in real-time. Study 2 (N=10, two-day with stimulated recall interviews) analyzed 214 turns, identifying five action patterns, six triggers, and four enabling factors explaining when designers choose delegation (70.1%), direction (28.5%), or concurrent work (31.8%). We present a decision model with six interaction loops, design implications, and an annotated dataset.

**arXiv ID:** 2603.02050
</details>

<details>
<summary><strong>CAPS: Context-Aware Priority Sampling for Enhanced Imitation Learning in Autonomous Driving</strong> - Hamidreza Mirkhani, Behzad Khamidehi, Ehsan Ahmadi, Mohammed Elmahgiubi, Weize Zhang, Fazel Arasteh, Umar Rajguru, Kasra Rezaee, Dongfeng Bai - [[pdf]](https://arxiv.org/pdf/2503.01650)</summary>

**Abstract:** In this paper, we introduce Context-Aware Priority Sampling (CAPS), a novel method designed to enhance data efficiency in learning-based autonomous driving systems. CAPS addresses the challenge of imbalanced datasets in imitation learning by leveraging Vector Quantized Variational Autoencoders (VQ-VAEs). In this way, we can get structured and interpretable data representations, which help to reveal meaningful patterns in the data. These patterns are used to group the data into clusters, with each sample being assigned a cluster ID. The cluster IDs are then used to re-balance the dataset, ensuring that rare yet valuable samples receive higher priority during training. We evaluate our method through closed-loop experiments in the CARLA simulator. The results on Bench2Drive scenarios demonstrate the effectiveness of CAPS in enhancing model generalization, with substantial improvements in both driving score and success rate.

**arXiv ID:** 2503.01650
</details>

<details>
<summary><strong>Temporal Misalignment Attacks against Multimodal Perception in Autonomous Driving</strong> - Md Hasan Shahriar, Md Mohaimin Al Barat, Harshavardhan Sundar, Ning Zhang, Naren Ramakrishnan, Y. Thomas Hou, Wenjing Lou - [[pdf]](https://arxiv.org/pdf/2507.09095)</summary>

**Abstract:** Multimodal fusion (MMF) plays a critical role in the perception of autonomous driving, which primarily fuses camera and LiDAR streams for a comprehensive and efficient scene understanding. However, its strict reliance on precise temporal synchronization exposes it to new vulnerabilities. In this paper, we introduce DejaVu, an attack that exploits the in-vehicular network to manipulate the integrity of time and create subtle temporal misalignments, severely degrading downstream MMF-based perception tasks. Our comprehensive attack analysis across different models and datasets reveals the sensors' task-specific imbalanced sensitivities: object detection is overly dependent on LiDAR inputs, while object tracking is highly reliant on the camera inputs. Consequently, with a single-frame LiDAR delay, an attacker can reduce the car detection mAP by up to 88.5%, while with a three-frame camera delay, multiple object tracking accuracy (MOTA) for car drops by 73%. We further demonstrated two attack scenarios using an automotive Ethernet testbed for hardware-in-the-loop validation and the Autoware stack for end-to-end AD simulation, demonstrating the feasibility of the DejaVu attack and its severe impact, such as collisions and phantom braking. Our code and artifacts are publicly available at: this https URL.

**arXiv ID:** 2507.09095
</details>

<details>
<summary><strong>Introducing the transitional autonomous vehicle lane-changing dataset: Empirical Experiments</strong> - Abhinav Sharma, Zijun He, Danjue Chen - [[pdf]](https://arxiv.org/pdf/2603.05716)</summary>

**Abstract:** Transitional autonomous vehicles (tAVs), which operate beyond SAE Level 1-2 automation but short of full autonomy, are increasingly sharing the road with human-driven vehicles (HDVs). As these systems interact during complex maneuvers such as lane changes, new patterns may emerge with implications for traffic stability and safety. Assessing these dynamics, particularly during mandatory lane changes, requires high-resolution trajectory data, yet datasets capturing tAV lane-changing behavior are scarce.
This study introduces the North Carolina Transitional Autonomous Vehicle Lane-Changing (NC-tALC) Dataset, a high-fidelity trajectory dataset designed to characterize tAV interactions during lane-changing maneuvers. The dataset includes two controlled experimental series. In the first, tAV lane-changing experiments, a tAV executes lane changes in the presence of adaptive cruise control (ACC) equipped target vehicles, enabling analysis of lane-changing execution. In the second, tAV responding experiments, two tAVs act as followers and respond to cut-in maneuvers initiated by another tAV, enabling analysis of follower response dynamics. The dataset contains 152 trials (72 lane-changing and 80 responding trials) sampled at 20 Hz with centimeter-level RTK-GPS accuracy. The NC-tALC dataset provides a rigorous empirical foundation for evaluating tAV decision-making and interaction dynamics in controlled mandatory lane-changing scenarios.

**arXiv ID:** 2603.05716
</details>

<details>
<summary><strong>NOVA: Next-step Open-Vocabulary Autoregression for 3D Multi-Object Tracking in Autonomous Driving</strong> - Kai Luo, Xu Wang, Rui Fan, Kailun Yang - [[pdf]](https://arxiv.org/pdf/2603.06254)</summary>

**Abstract:** Generalizing across unknown targets is critical for open-world perception, yet existing 3D Multi-Object Tracking (3D MOT) pipelines remain limited by closed-set assumptions and ``semantic-blind'' heuristics. To address this, we propose Next-step Open-Vocabulary Autoregression (NOVA), an innovative paradigm that shifts 3D tracking from traditional fragmented distance-based matching toward generative spatio-temporal semantic modeling. NOVA reformulates 3D trajectories as structured spatio-temporal semantic sequences, enabling the simultaneous encoding of physical motion continuity and deep linguistic priors. By leveraging the autoregressive capabilities of Large Language Models (LLMs), we transform the tracking task into a principled process of next-step sequence completion. This mechanism allows the model to explicitly utilize the hierarchical structure of language space to resolve fine-grained semantic ambiguities and maintain identity consistency across complex long-range sequences through high-level commonsense reasoning. Extensive experiments on nuScenes, V2X-Seq-SPD, and KITTI demonstrate the superior performance of NOVA. Notably, on the nuScenes dataset, NOVA achieves an AMOTA of 22.41% for Novel categories, yielding a significant 20.21% absolute improvement over the baseline. These gains are realized through a compact 0.5B autoregressive model. Code will be available at this https URL.

**arXiv ID:** 2603.06254
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>DeepFact: Co-Evolving Benchmarks and Agents for Deep Research Factuality</strong> - Yukun Huang, Leonardo F. R. Ribeiro, Momchil Hardalov, Bhuwan Dhingra, Markus Dreyer, Venkatesh Saligrama - [[pdf]](https://arxiv.org/pdf/2603.05912)</summary>

**Abstract:** Search-augmented LLM agents can produce deep research reports (DRRs), but verifying claim-level factuality remains challenging. Existing fact-checkers are primarily designed for general-domain, factoid-style atomic claims, and there is no benchmark to test whether such verifiers transfer to DRRs. Yet building such a benchmark is itself difficult. We first show that static expert-labeled benchmarks are brittle in this setting: in a controlled study with PhD-level specialists, unassisted experts achieve only 60.8% accuracy on a hidden micro-gold set of verifiable claims. We propose Evolving Benchmarking via Audit-then-Score (AtS), where benchmark labels and rationales are explicitly revisable: when a verifier disagrees with the current benchmark, it must submit evidence; an auditor adjudicates the dispute; and accepted revisions update the benchmark before models are scored. Across four AtS rounds, expert micro-gold accuracy rises to 90.9%, indicating experts are substantially more reliable as auditors than as one-shot labelers. We instantiate AtS as DeepFact-Bench, a versioned DRR factuality benchmark with auditable rationales, and DeepFact-Eval, a document-level verification agent (with a grouped lite variant) that outperforms existing verifiers on DeepFact-Bench and transfers well to external factuality datasets.

**arXiv ID:** 2603.05912
</details>

<details>
<summary><strong>Traversal-as-Policy: Log-Distilled Gated Behavior Trees as Externalized, Verifiable Policies for Safe, Robust, and Efficient Agents</strong> - Peiran Li, Jiashuo Sun, Fangzhou Lin, Shuo Xing, Tianfu Fu, Suofei Feng, Chaoqun Ni, Zhengzhong Tu - [[pdf]](https://arxiv.org/pdf/2603.05517)</summary>

**Abstract:** Autonomous LLM agents fail because long-horizon policy remains implicit in model weights and transcripts, while safety is retrofitted post hoc. We propose Traversal-as-Policy: distill sandboxed OpenHands execution logs into a single executable Gated Behavior Tree (GBT) and treat tree traversal -- rather than unconstrained generation -- as the control policy whenever a task is in coverage. Each node encodes a state-conditioned action macro mined and merge-checked from successful trajectories; macros implicated by unsafe traces attach deterministic pre-execution gates over structured tool context and bounded history, updated under experience-grounded monotonicity so previously rejected unsafe contexts cannot be re-admitted. At runtime, a lightweight traverser matches the base model's intent to child macros, executes one macro at a time under global and node-local gating, and when stalled performs risk-aware shortest-path recovery to a feasible success leaf; the visited path forms a compact spine memory that replaces transcript replay. Evaluated in a unified OpenHands sandbox on 15+ software, web, reasoning, and safety/security benchmarks, GBT improves success while driving violations toward zero and reducing cost. On SWE-bench Verified (Protocol A, 500 issues), GBT-SE raises success from 34.6% to 73.6%, reduces violations from 2.8% to 0.2%, and cuts token/character usage from 208k/820k to 126k/490k; with the same distilled tree, 8B executors more than double success on SWE-bench Verified (14.0%58.8%) and WebArena (9.1%37.3%).

**arXiv ID:** 2603.05517
</details>

<details>
<summary><strong>Uncertainty Quantification in LLM Agents: Foundations, Emerging Challenges, and Opportunities</strong> - Changdae Oh, Seongheon Park, To Eun Kim, Jiatong Li, Wendi Li, Samuel Yeh, Xuefeng Du, Hamed Hassani, Paul Bogdan, Dawn Song, Sharon Li - [[pdf]](https://arxiv.org/pdf/2602.05073)</summary>

**Abstract:** Uncertainty quantification (UQ) for large language models (LLMs) is a key building block for safety guardrails of daily LLM applications. Yet, even as LLM agents are increasingly deployed in highly complex tasks, most UQ research still centers on single-turn question-answering. We argue that UQ research must shift to realistic settings with interactive agents, and that a new principled framework for agent UQ is needed. This paper presents three pillars to build a solid ground for future agent UQ research: (1. Foundations) We present the first general formulation of agent UQ that subsumes broad classes of existing UQ setups; (2. Challenges) We identify four technical challenges specifically tied to agentic setups -- selection of uncertainty estimator, uncertainty of heterogeneous entities, modeling uncertainty dynamics in interactive systems, and lack of fine-grained benchmarks -- with numerical analysis on a real-world agent benchmark, $\tau^2$-bench; (3. Future Directions) We conclude with noting on the practical implications of agent UQ and remaining open problems as forward-looking discussion for future explorations.

**arXiv ID:** 2602.05073
</details>

<details>
<summary><strong>Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization</strong> - Zeyuan Liu, Jeonghye Kim, Xufang Luo, Dongsheng Li, Yuqing Yang - [[pdf]](https://arxiv.org/pdf/2602.23008)</summary>

**Abstract:** Exploration remains the key bottleneck for large language model agents trained with reinforcement learning. While prior methods exploit pretrained knowledge, they fail in environments requiring the discovery of novel states. We propose Exploratory Memory-Augmented On- and Off-Policy Optimization (EMPO$^2$), a hybrid RL framework that leverages memory for exploration and combines on- and off-policy updates to make LLMs perform well with memory while also ensuring robustness without it. On ScienceWorld and WebShop, EMPO$^2$ achieves 128.6% and 11.3% improvements over GRPO, respectively. Moreover, in out-of-distribution tests, EMPO$^2$ demonstrates superior adaptability to new tasks, requiring only a few trials with memory and no parameter updates. These results highlight EMPO$^2$ as a promising framework for building more exploratory and generalizable LLM-based agents.

**arXiv ID:** 2602.23008
</details>

<details>
<summary><strong>Both Ends Count! Just How Good are LLM Agents at "Text-to-Big SQL"?</strong> - Germán T. Eizaguirre, Lars Tissen, Marc Sánchez-Artigas - [[pdf]](https://arxiv.org/pdf/2602.21480)</summary>

**Abstract:** Text-to-SQL and Big Data are both extensively benchmarked fields, yet there is limited research that evaluates them jointly. In the real world, Text-to-SQL systems are often embedded with Big Data workflows, such as large-scale data processing or interactive data analytics. We refer to this as "Text-to-Big SQL". However, existing text-to-SQL benchmarks remain narrowly scoped and overlook the cost and performance implications that arise at scale. For instance, translation errors that are minor on small datasets lead to substantial cost and latency overheads as data scales, a relevant issue completely ignored by text-to-SQL metrics. In this paper, we overcome this overlooked challenge by introducing novel and representative metrics for evaluating Text-to-Big SQL. Our study focuses on production-level LLM agents, a database-agnostic system adaptable to diverse user needs. Via an extensive evaluation of frontier models, we show that text-to-SQL metrics are insufficient for Big Data. In contrast, our proposed text-to-Big SQL metrics accurately reflect execution efficiency, cost, and the impact of data scale. Furthermore, we provide LLM-specific insights, including fine-grained, cross-model comparisons of latency and cost.

**arXiv ID:** 2602.21480
</details>

<details>
<summary><strong>Latent Poincaré Shaping for Agentic Reinforcement Learning</strong> - Hanchen Xia, Baoyou Chen, Zelin Zang, Yutang Ge, Guojiang Zhao, Siyu Zhu - [[pdf]](https://arxiv.org/pdf/2602.09375)</summary>

**Abstract:** We propose LaPha, a method for training AlphaZero-like LLM agents in a Poincaré latent space. Under LaPha, the search process can be visualized as a tree rooted at the prompt and growing outward from the origin toward the boundary of the Poincaré ball, where negative curvature provides exponentially increasing capacity with radius. Using hyperbolic geodesic distance to rule-verified correctness, we define a node potential and assign dense process rewards by potential differences. We further attach a lightweight value head on the same shared latent space, enabling self-guided test-time scaling with almost no additional overhead. On MATH-500, LaPha improves Qwen2.5-Math-1.5B from 66.0% to 88.2%. With value-head-guided search, LaPha-1.5B reaches 56.7% accuracy on AIME'24, and LaPha-7B further achieves 60.0% on AIME'24 and 53.3% on AIME'25.

**arXiv ID:** 2602.09375
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (13 papers)</h2></summary>

<details>
<summary><strong>An Interactive Multi-Agent System for Evaluation of New Product Concepts</strong> - Bin Xuan, Ruo Ai, Hakyeon Lee - [[pdf]](https://arxiv.org/pdf/2603.05980)</summary>

**Abstract:** Product concept evaluation is a critical stage that determines strategic resource allocation and project success in enterprises. However, traditional expert-led approaches face limitations such as subjective bias and high time and cost requirements. To support this process, this study proposes an automated approach utilizing a large language model (LLM)-based multi-agent system (MAS). Through a systematic analysis of previous research on product development and team collaboration, this study established two primary evaluation dimensions, namely technical feasibility and market feasibility. The proposed system consists of a team of eight virtual agents representing specialized domains such as R&D and marketing. These agents use retrieval-augmented generation (RAG) and real-time search tools to gather objective evidence and validate concepts through structured deliberations based on the established criteria. The agents were further fine-tuned using professional product review data to enhance their judgment accuracy. A case study involving professional display monitor concepts demonstrated that the system's evaluation rankings were consistent with those of senior industry experts. These results confirm the usability of the proposed multi-agent-based evaluation approach for supporting product development decisions.

**arXiv ID:** 2603.05980
</details>

<details>
<summary><strong>Conversational Demand Response: Bidirectional Aggregator-Prosumer Coordination through Agentic AI</strong> - Reda El Makroum, Sebastian Zwickl-Bernhard, Lukas Kranzl, Hans Auer - [[pdf]](https://arxiv.org/pdf/2603.06217)</summary>

**Abstract:** Residential demand response depends on sustained prosumer participation, yet existing coordination is either fully automated, or limited to one-way dispatch signals and price alerts that offer little possibility for informed decision-making. This paper introduces Conversational Demand Response (CDR), a coordination mechanism where aggregators and prosumers interact through bidirectional natural language, enabled through agentic AI. A two-tier multi-agent architecture is developed in which an aggregator agent dispatches flexibility requests and a prosumer Home Energy Management System (HEMS) assesses deliverability and cost-benefit by calling an optimization-based tool. CDR also enables prosumer-initiated upstream communication, where changes in preferences can reach the aggregator directly. Proof-of-concept evaluation shows that interactions complete in under 12 seconds. The architecture illustrates how agentic AI can bridge the aggregator-prosumer coordination gap, providing the scalability of automated DR while preserving the transparency, explainability, and user agency necessary for sustained prosumer participation. All system components, including agent prompts, orchestration logic, and simulation interfaces, are released as open source to enable reproducibility and further development.

**arXiv ID:** 2603.06217
</details>

<details>
<summary><strong>EigenData: A Self-Evolving Multi-Agent Platform for Function-Calling Data Synthesis, Auditing, and Repair</strong> - Jiaao Chen, Jingyuan Qi, Mingye Gao, Wei-Chen Wang, Hanrui Wang, Di Jin - [[pdf]](https://arxiv.org/pdf/2603.05553)</summary>

**Abstract:** Function-calling agents -- large language models that invoke tools and APIs -- require high-quality, domain-specific training data spanning executable environments, backing databases, and diverse multi-turn trajectories. We introduce EigenData, an integrated, self-evolving platform that automates the full data lifecycle through a multi-agent architecture. A top-level orchestrator, EigenCore, coordinates three specialized sub-systems: DatabaseAgent for realistic domain database construction, CodingAgent for verified executable environment generation with iterative test-debug loops, and DataAgent for multi-turn trajectory synthesis with self-evolving prompt optimization. Cross-component feedback ensures consistency across all artifacts. We apply EigenData to audit and repair the Berkeley Function-Calling Leaderboard (BFCL-V3), identifying systematic errors in function schemas, implementations, and reference trajectories, automatically correcting them through coordinated schema refinement, code-level bug fixes, and trajectory modification, and introducing an outcome-aware evaluation protocol that assesses task success via database-state correctness rather than turn-level trajectory matching. We demonstrate that the repaired benchmark, coupled with outcome-aware metrics, produces model rankings substantially better correlated with human judgments of functional correctness.

**arXiv ID:** 2603.05553
</details>

<details>
<summary><strong>RACAS: Controlling Diverse Robots With a Single Agentic System</strong> - Dylan R. Ashley, Jan Przepióra, Yimeng Chen, Ali Abualsaud, Nurzhan Yesmagambet, Shinkyu Park, Eric Feron, Jürgen Schmidhuber - [[pdf]](https://arxiv.org/pdf/2603.05621)</summary>

**Abstract:** Many robotic platforms expose an API through which external software can command their actuators and read their sensors. However, transitioning from these low-level interfaces to high-level autonomous behaviour requires a complicated pipeline, whose components demand distinct areas of expertise. Existing approaches to bridging this gap either require retraining for every new embodiment or have only been validated across structurally similar platforms. We introduce RACAS (Robot-Agnostic Control via Agentic Systems), a cooperative agentic architecture in which three LLM/VLM-based modules (Monitors, a Controller, and a Memory Curator) communicate exclusively through natural language to provide closed-loop robot control. RACAS requires only a natural language description of the robot, a definition of available actions, and a task specification; no source code, model weights, or reward functions need to be modified to move between platforms. We evaluate RACAS on several tasks using a wheeled ground robot, a recently published novel multi-jointed robotic limb, and an underwater vehicle. RACAS consistently solved all assigned tasks across these radically different platforms, demonstrating the potential of agentic AI to substantially reduce the barrier to prototyping robotic solutions.

**arXiv ID:** 2603.05621
</details>

<details>
<summary><strong>SecureRAG-RTL: A Retrieval-Augmented, Multi-Agent, Zero-Shot LLM-Driven Framework for Hardware Vulnerability Detection</strong> - Touseef Hasan, Blessing Airehenbuwa, Nitin Pundir, Souvika Sarkar, Ujjwal Guin - [[pdf]](https://arxiv.org/pdf/2603.05689)</summary>

**Abstract:** Large language models (LLMs) have shown remarkable capabilities in natural language processing tasks, yet their application in hardware security verification remains limited due to scarcity of publicly available hardware description language (HDL) datasets. This knowledge gap constrains LLM performance in detecting vulnerabilities within HDL designs. To address this challenge, we propose SecureRAG-RTL, a novel Retrieval-Augmented Generation (RAG)-based approach that significantly enhances LLM-based security verification of hardware designs. Our approach integrates domain-specific retrieval with generative reasoning, enabling models to overcome inherent limitations in hardware security expertise. We establish baseline vulnerability detection rates using prompt-only methods and then demonstrate that SecureRAG-RTL achieves substantial improvements across diverse LLM architectures, regardless of size. On average, our method increases detection accuracy by about 30%, highlighting its effectiveness in bridging domain knowledge gaps. For evaluation, we curated and annotated a benchmark dataset of 14 HDL designs containing real-world security vulnerabilities, which we will release publicly to support future research. These findings underscore the potential of RAG-driven augmentation to enable scalable, efficient, and accurate hardware security verification workflows.

**arXiv ID:** 2603.05689
</details>

<details>
<summary><strong>MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing</strong> - Yang Liu, Jinxuan Cai, Yishen Li, Qi Meng, Zedi Liu, Xin Li, Chen Qian, Chuan Shi, Cheng Yang - [[pdf]](https://arxiv.org/pdf/2603.06007)</summary>

**Abstract:** Large language model-based (LLM-based) multi-agent systems (MAS) are increasingly used to extend agentic problem solving via role specialization and collaboration. MAS workflows can be naturally modeled as directed computation graphs, where nodes execute agents/sub-workflows and edges encode dependencies and message passing. However, implementing complex graph workflows in current frameworks still requires substantial manual effort, offers limited reuse, and makes it difficult to integrate heterogeneous external context sources. To overcome these limitations, we present MASFactory, a graph-centric framework for orchestrating LLM-based MAS. It introduces Vibe Graphing, a human-in-the-loop approach that compiles natural-language intent into an editable workflow specification and then into an executable graph. In addition, the framework provides reusable components and pluggable context integration, as well as a visualizer for topology preview, runtime tracing, and human-in-the-loop interaction. We evaluate MASFactory on seven public benchmarks, validating both reproduction consistency for representative MAS methods and the effectiveness of Vibe Graphing. Our code (this https URL) and video (this https URL) are publicly available.

**arXiv ID:** 2603.06007
</details>

<details>
<summary><strong>A Multi-Agent System Enables Versatile Information Extraction from the Chemical Literature</strong> - Yufan Chen, Ching Ting Leung, Bowen Yu, Jianwei Sun, Yong Huang, Linyan Li, Hao Chen, Hanyu Gao - [[pdf]](https://arxiv.org/pdf/2507.20230)</summary>

**Abstract:** To fully expedite AI-powered chemical research, high-quality chemical databases are the foundation. Automatic extraction of chemical information from the literature is essential for constructing reaction databases, but it is currently limited by the multimodality and style variability of chemical information. In this work, we developed a multimodal large language model (MLLM)-based multi-agent system for robust and automated chemical information extraction. It utilizes the MLLM's strong reasoning capability to understand the structure of diverse chemical graphics and decompose the extraction task into sub-tasks. It then coordinates a set of specialized agents, each combining the capabilities of the MLLM with the precise, domain-specific strengths of dedicated tools and web services, to solve the subtasks accurately and integrate the results into a unified output. Our system achieved an F1 score of 76.27% on a benchmark dataset of sophisticated multimodal chemical reaction graphics from the literature, surpassing the previous state-of-the-art model (F1 score of 39.13%) by a significant margin. Additionally, it demonstrated versatile applicability in a range of other information extraction tasks, including molecular image recognition, reaction image parsing, named entity recognition and text-based reaction extraction. This work is a critical step toward automated chemical information extraction into structured datasets, which will be a strong promoter of AI-driven chemical research.

**arXiv ID:** 2507.20230
</details>

<details>
<summary><strong>Information-Theoretic Privacy Control for Sequential Multi-Agent LLM Systems</strong> - Sadia Asif, Mohammad Mohammadi Amiri - [[pdf]](https://arxiv.org/pdf/2603.05520)</summary>

**Abstract:** Sequential multi-agent large language model (LLM) systems are increasingly deployed in sensitive domains such as healthcare, finance, and enterprise decision-making, where multiple specialized agents collaboratively process a single user request. Although individual agents may satisfy local privacy constraints, sensitive information can still be inferred through sequential composition and intermediate representations. In this work, we study \emph{compositional privacy leakage} in sequential LLM agent pipelines. We formalize leakage using mutual information and derive a theoretical bound that characterizes how locally introduced leakage can amplify across agents under sequential execution. Motivated by this analysis, we propose a privacy-regularized training framework that directly constrains information flow between agent outputs and agent-local sensitive variables. We evaluate our approach across sequential agent pipelines of varying depth on three benchmark datasets, demonstrating stable optimization dynamics and consistent, interpretable privacy-utility trade-offs. Our results show that privacy in agentic LLM systems cannot be guaranteed by local constraints alone and must instead be treated as a system-level property during both training and deployment.

**arXiv ID:** 2603.05520
</details>

<details>
<summary><strong>The Coordination Gap: Alternation Metrics for Temporal Dynamics in Multi-Agent Battle of the Exes</strong> - Nikolaos Al. Papadopoulos, Konstantinos Psannis - [[pdf]](https://arxiv.org/pdf/2603.05789)</summary>

**Abstract:** Multi-agent coordination dilemmas expose a fundamental tension between individual optimization and collective welfare, yet characterizing such coordination requires metrics sensitive to temporal structure and collective dynamics. As a diagnostic testbed, we study a BoE-derived multi-agent variant of the Battle of the Exes, formalizing it as a Markov game in which turn-taking emerges as a periodic coordination regime. Conventional outcome-based metrics (e.g., efficiency and min/max fairness) are temporally blind -- they cannot distinguish structured alternation from monopolistic or random access patterns -- and fairness ratios lose discriminative power as n grows, obscuring inequities.
To address this limitation, we introduce Perfect Alternation (PA) as a reference coordination regime and propose six novel Alternation (ALT) metrics designed as temporally sensitive observables of coordination quality. Using Q-learning agents as a minimal adaptive diagnostic baseline, and comparing against random-policy null processes, we uncover a clear measurement failure: despite exhibiting deceptively high traditional metrics (e.g., reward fairness often exceeding 0.9), learned policies perform up to 81% below random baselines under ALT-variant evaluation -- a deficit already present in the two-agent case and intensifying as n grows.
These results demonstrate, in this setting, that high aggregate payoffs can coexist with poor temporal coordination, and that conventional metrics may severely mischaracterize emergent dynamics. Our findings underscore the necessity of temporally aware observables for analyzing coordination in multi-agent games and highlight random-policy baselines as essential null processes for interpreting coordination outcomes relative to chance-level behavior.

**arXiv ID:** 2603.05789
</details>

<details>
<summary><strong>Symmetry-Breaking in Multi-Agent Navigation: Winding Number-Aware MPC with a Learned Topological Strategy</strong> - Tomoki Nakao, Kazumi Kasaura, Tadashi Kozuno - [[pdf]](https://arxiv.org/pdf/2511.15239)</summary>

**Abstract:** In distributed multi-agent navigation without explicit communication, agents can fall into symmetry-induced deadlocks because each agent must autonomously decide how to pass others. To address this problem, we propose WNumMPC, a hierarchical navigation method that quantifies cooperative symmetry-breaking strategies via a topological invariant, the winding number, and learns such strategies through reinforcement learning. The learning-based Planner outputs continuous-valued signed target winding numbers and dynamic importance weights to prioritize critical interactions in dense crossings. Then, the model-based Controller generates collision-free and efficient motions based on the strategy and weights provided by the Planner. Simulation and real-world robot experiments indicate that WNumMPC effectively avoids deadlocks and collisions and achieves better performance than the baselines, particularly in dense and symmetry-prone scenarios. These experiments also suggest that explicitly leveraging winding numbers yields robust sim-to-real transfer with minimal performance degradation. The code for the experiments is available at this https URL.

**arXiv ID:** 2511.15239
</details>

<details>
<summary><strong>Let's Talk, Not Type: An Oral-First Multi-Agent Architecture for Guaraní</strong> - Samantha Adorno, Akshata Kishore Moharir, Ratna Kandala - [[pdf]](https://arxiv.org/pdf/2603.05743)</summary>

**Abstract:** Although artificial intelligence (AI) and Human-Computer Interaction (HCI) systems are often presented as universal solutions, their design remains predominantly text-first, underserving primarily oral languages and indigenous communities. This position paper uses Guaraní, an official and widely spoken language of Paraguay, as a case study to argue that language support in AI remains insufficient unless it aligns with lived oral practices. We propose an alternative to the standard "text-to-speech" pipeline, proposing instead an oral-first multi-agent architecture. By decoupling Guaraní natural language understanding from dedicated agents for conversation state and community-led governance, we demonstrate a technical framework that respects indigenous data sovereignty and diglossia. Our work moves beyond mere recognition to focus on turn-taking, repair, and shared context as the primary locus of interaction. We conclude that for AI to be truly culturally grounded, it must shift from adapting oral languages to text-centric systems to treating spoken conversation as a first-class design requirement, ensuring digital ecosystems empower rather than overlook diverse linguistic practices.

**arXiv ID:** 2603.05743
</details>

<details>
<summary><strong>MARLIN: Multi-Agent Reinforcement Learning with Murmuration Intelligence and LLM Guidance for Reservoir Management</strong> - Heming Fu, Shan Lin, Guojun Xiong - [[pdf]](https://arxiv.org/pdf/2509.25034)</summary>

**Abstract:** As climate change intensifies extreme weather events, water disasters pose growing threats to global communities, making adaptive reservoir management critical for protecting vulnerable populations and ensuring water security. Modern water resource management faces unprecedented challenges from cascading uncertainties propagating through interconnected reservoir networks. These uncertainties, rooted in physical water transfer losses and environmental variability, make precise control difficult. For example, sending 10 tons downstream may yield only 8-12 tons due to evaporation and seepage. Traditional centralized optimization approaches suffer from exponential computational complexity and cannot effectively handle such real-world uncertainties, while existing multi-agent reinforcement learning (MARL) methods fail to achieve effective coordination under uncertainty. To address these challenges, we present MARLIN, a decentralized reservoir management framework inspired by starling murmurations intelligence. Integrating bio-inspired alignment, separation, and cohesion rules with MARL, MARLIN enables individual reservoirs to make local decisions while achieving emergent global coordination. In addition, a LLM provides real-time reward shaping signals, guiding agents to adapt to environmental changes and human-defined preferences. Experiments on USGS data show that MARLIN improves uncertainty handling by 23\%, cuts computation by 35\%, and accelerates flood response by 68\%, exhibiting super-linear coordination, with complexity scaling 5.4x from 400 to 10,000 nodes. These results demonstrate MARLIN's potential for disaster prevention and protecting communities through intelligent, scalable water resource management.

**arXiv ID:** 2509.25034
</details>

<details>
<summary><strong>Diverse and Adaptive Behavior Curriculum for Autonomous Driving: A Student-Teacher Framework with Multi-Agent RL</strong> - Ahmed Abouelazm, Johannes Ratz, Philip Schörner, J. Marius Zöllner - [[pdf]](https://arxiv.org/pdf/2507.19146)</summary>

**Abstract:** Autonomous driving faces challenges in navigating complex real-world traffic, requiring safe handling of both common and critical scenarios. Reinforcement learning (RL), a prominent method in end-to-end driving, enables agents to learn through trial and error in simulation. However, RL training often relies on rule-based traffic scenarios, limiting generalization. Additionally, current scenario generation methods focus heavily on critical scenarios, neglecting a balance with routine driving behaviors. Curriculum learning, which progressively trains agents on increasingly complex tasks, is a promising approach to improving the robustness and coverage of RL driving policies. However, existing research mainly emphasizes manually designed curricula, focusing on scenery and actor placement rather than traffic behavior dynamics. This work introduces a novel student-teacher framework for automatic curriculum learning. The teacher, a graph-based multi-agent RL component, adaptively generates traffic behaviors across diverse difficulty levels. An adaptive mechanism adjusts task difficulty based on student performance, ensuring exposure to behaviors ranging from common to critical. The student, though exchangeable, is realized as a deep RL agent with partial observability, reflecting real-world perception constraints. Results demonstrate the teacher's ability to generate diverse traffic behaviors. The student, trained with automatic curricula, outperformed agents trained on rule-based traffic, achieving higher rewards and exhibiting balanced, assertive driving.

**arXiv ID:** 2507.19146
</details>

</details>

<details open>
<summary><h2>Other Agent Research (11 papers)</h2></summary>

<details>
<summary><strong>RoboLayout: Differentiable 3D Scene Generation for Embodied Agents</strong> - Ali Shamsaddinlou - [[pdf]](https://arxiv.org/pdf/2603.05522)</summary>

**Abstract:** Recent advances in vision language models (VLMs) have shown strong potential for spatial reasoning and 3D scene layout generation from open-ended language instructions. However, generating layouts that are not only semantically coherent but also feasible for interaction by embodied agents remains challenging, particularly in physically constrained indoor environments. In this paper, RoboLayout is introduced as an extension of LayoutVLM that augments the original framework with agent-aware reasoning and improved optimization stability. RoboLayout integrates explicit reachability constraints into a differentiable layout optimization process, enabling the generation of layouts that are navigable and actionable by embodied agents. Importantly, the agent abstraction is not limited to a specific robot platform and can represent diverse entities with distinct physical capabilities, such as service robots, warehouse robots, humans of different age groups, or animals, allowing environment design to be tailored to the intended agent. In addition, a local refinement stage is proposed that selectively reoptimizes problematic object placements while keeping the remainder of the scene fixed, improving convergence efficiency without increasing global optimization iterations. Overall, RoboLayout preserves the strong semantic alignment and physical plausibility of LayoutVLM while enhancing applicability to agent-centric indoor scene generation, as demonstrated by experimental results across diverse scene configurations.

**arXiv ID:** 2603.05522
</details>

<details>
<summary><strong>Agentic LLM Planning via Step-Wise PDDL Simulation: An Empirical Characterisation</strong> - Kai Göbel, Pierrick Lorang, Patrik Zips, Tobias Glück - [[pdf]](https://arxiv.org/pdf/2603.06064)</summary>

**Abstract:** Task planning, the problem of sequencing actions to reach a goal from an initial state, is a core capability requirement for autonomous robotic systems. Whether large language models (LLMs) can serve as viable planners alongside classical symbolic methods remains an open question. We present PyPDDLEngine, an open-source Planning Domain Definition Language (PDDL) simulation engine that exposes planning operations as LLM tool calls through a Model Context Protocol (MCP) interface. Rather than committing to a complete action sequence upfront, the LLM acts as an interactive search policy that selects one action at a time, observes each resulting state, and can reset and retry. We evaluate four approaches on 102 International Planning Competition (IPC) Blocksworld instances under a uniform 180-second budget: Fast Downward lama-first and seq-sat-lama-2011 as classical baselines, direct LLM planning (Claude Haiku 4.5), and agentic LLM planning via PyPDDLEngine. Fast Downward achieves 85.3% success. The direct and agentic LLM approaches achieve 63.7% and 66.7%, respectively, a consistent but modest three-percentage-point advantage for the agentic approach at $5.7\times$ higher token cost per solution. Across most co-solved difficulty blocks, both LLM approaches produce shorter plans than seq-sat-lama-2011 despite its iterative quality improvement, a result consistent with training-data recall rather than generalisable planning. These results suggest that agentic gains depend on the nature of environmental feedback. Coding agents benefit from externally grounded signals such as compiler errors and test failures, whereas PDDL step feedback is self-assessed, leaving the agent to evaluate its own progress without external verification.

**arXiv ID:** 2603.06064
</details>

<details>
<summary><strong>Talk Freely, Execute Strictly: Schema-Gated Agentic AI for Flexible and Reproducible Scientific Workflows</strong> - Joel Strickland, Arjun Vijeta, Chris Moores, Oliwia Bodek, Bogdan Nenchev, Thomas Whitehead, Charles Phillips, Karl Tassenberg, Gareth Conduit, Ben Pellegrini - [[pdf]](https://arxiv.org/pdf/2603.06394)</summary>

**Abstract:** Large language models (LLMs) can now translate a researcher's plain-language goal into executable computation, yet scientific workflows demand determinism, provenance, and governance that are difficult to guarantee when an LLM decides what runs. Semi-structured interviews with 18 experts across 10 industrial R&D stakeholders surface 2 competing requirements--deterministic, constrained execution and conversational flexibility without workflow rigidity--together with boundary properties (human-in-the-loop control and transparency) that any resolution must satisfy. We propose schema-gated orchestration as the resolving principle: the schema becomes a mandatory execution boundary at the composed-workflow level, so that nothing runs unless the complete action--including cross-step dependencies--validates against a machine-checkable specification.
We operationalize the 2 requirements as execution determinism (ED) and conversational flexibility (CF), and use these axes to review 20 systems spanning 5 architectural groups along a validation-scope spectrum. Scores are assigned via a multi-model protocol--15 independent sessions across 3 LLM families--yielding substantial-to-near-perfect inter-model agreement (Krippendorff a=0.80 for ED and a=0.98 for CF), demonstrating that multi-model LLM scoring can serve as a reusable alternative to human expert panels for architectural assessment.
The resulting landscape reveals an empirical Pareto front--no reviewed system achieves both high flexibility and high determinism--but a convergence zone emerges between the generative and workflow-centric extremes. We argue that a schema-gated architecture, separating conversational from execution authority, is positioned to decouple this trade-off, and distill 3 operational principles--clarification-before-execution, constrained plan-act orchestration, and tool-to-workflow-level gating--to guide adoption.

**arXiv ID:** 2603.06394
</details>

<details>
<summary><strong>An Embodied Companion for Visual Storytelling</strong> - Patrick Tresset, Markus Wulfmeier - [[pdf]](https://arxiv.org/pdf/2603.05511)</summary>

**Abstract:** As artificial intelligence shifts from pure tool for delegation toward agentic collaboration, its use in the arts can shift beyond the exploration of machine autonomy toward synergistic co-creation. While our earlier robotic works utilized automation to distance the artist's intent from the final mark, we present Companion: an artistic apparatus that integrates a drawing robot with Large Language Models (LLMs) to re-center human-machine presence. By leveraging in-context learning and real-time tool use, the system engages in bidirectional interaction via speech and sketching. This approach transforms the robot from a passive executor into a playful co-creative partner capable of driving shared visual storytelling into unexpected aesthetic territories. To validate this collaborative shift, we employed the Consensual Assessment Technique (CAT) with a panel of seven art-world experts. Results confirm that the system produces works with a distinct aesthetic identity and professional exhibition merit, demonstrating the potential of AI as a highly capable artistic collaborator.

**arXiv ID:** 2603.05511
</details>

<details>
<summary><strong>Proof-of-Guardrail in AI Agents and What (Not) to Trust from It</strong> - Xisen Jin, Michael Duan, Qin Lin, Aaron Chan, Zhenglun Chen, Junyi Du, Xiang Ren - [[pdf]](https://arxiv.org/pdf/2603.05786)</summary>

**Abstract:** As AI agents become widely deployed as online services, users often rely on an agent developer's claim about how safety is enforced, which introduces a threat where safety measures are falsely advertised. To address the threat, we propose proof-of-guardrail, a system that enables developers to provide cryptographic proof that a response is generated after a specific open-source guardrail. To generate proof, the developer runs the agent and guardrail inside a Trusted Execution Environment (TEE), which produces a TEE-signed attestation of guardrail code execution verifiable by any user offline. We implement proof-of-guardrail for OpenClaw agents and evaluate latency overhead and deployment cost. Proof-of-guardrail ensures integrity of guardrail execution while keeping the developer's agent private, but we also highlight a risk of deception about safety, for example, when malicious developers actively jailbreak the guardrail. Code and demo video: this https URL

**arXiv ID:** 2603.05786
</details>

<details>
<summary><strong>XAI for Coding Agent Failures: Transforming Raw Execution Traces into Actionable Insights</strong> - Arun Joshi - [[pdf]](https://arxiv.org/pdf/2603.05941)</summary>

**Abstract:** Large Language Model (LLM)-based coding agents show promise in automating software development tasks, yet they frequently fail in ways that are difficult for developers to understand and debug. While general-purpose LLMs like GPT can provide ad-hoc explanations of failures, raw execution traces remain challenging to interpret even for experienced developers. We present a systematic explainable AI (XAI) approach that transforms raw agent execution traces into structured, human-interpretable explanations. Our method consists of three key components: (1) a domain-specific failure taxonomy derived from analyzing real agent failures, (2) an automatic annotation system that classifies failures using defined annotation schema, (3) a hybrid explanation generator that produces visual execution flows, natural language explanations, and actionable recommendations. Through a user study with 20 participants (10 technical, 10 non-technical), we demonstrate that our approach enables users to identify failure root causes 2.8 times faster and propose correct fixes with 73% higher accuracy compared to raw execution traces. Importantly, our structured approach outperforms ad-hoc state of the art models explanations by providing consistent, domain-specific insights with integrated visualizations. Our work establishes a framework for systematic agent failure analysis, addressing the critical need for interpretable AI systems in software development workflows

**arXiv ID:** 2603.05941
</details>

<details>
<summary><strong>Lifelong Embodied Navigation Learning</strong> - Xudong Wang, Jiahua Dong, Baichen Liu, Qi Lyu, Lianqing Liu, Zhi Han - [[pdf]](https://arxiv.org/pdf/2603.06073)</summary>

**Abstract:** Embodied navigation agents powered by large language models have shown strong performance on individual tasks but struggle to continually acquire new navigation skills, which suffer from catastrophic forgetting. We formalize this challenge as lifelong embodied navigation learning (LENL), where an agent is required to adapt to a sequence of navigation tasks spanning multiple scenes and diverse user instruction styles, while retaining previously learned knowledge. To tackle this problem, we propose Uni-Walker, a lifelong embodied navigation framework that decouples navigation knowledge into task-shared and task-specific components with Decoder Extension LoRA (DE-LoRA). To learn the shared knowledge, we design a knowledge inheritance strategy and an experts co-activation strategy to facilitate shared knowledge transfer and refinement across multiple navigation tasks. To learn the specific knowledge, we propose an expert subspace orthogonality constraint together and a navigation-specific chain-of-thought reasoning mechanism to capture specific knowledge and enhance instruction-style understanding. Extensive experiments demonstrate the superiority of Uni-Walker for building universal navigation agents with lifelong learning.

**arXiv ID:** 2603.06073
</details>

<details>
<summary><strong>Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People</strong> - Gabriel Grand, Valerio Pepe, Jacob Andreas, Joshua B. Tenenbaum - [[pdf]](https://arxiv.org/pdf/2510.20886)</summary>

**Abstract:** Many emerging applications of AI--from scientific discovery to medical diagnosis--require agents to seek information strategically: forming hypotheses, asking targeted questions, and making decisions under uncertainty. In high-stakes settings with limited resources, do language models (LMs) behave like rational agents? Drawing on insights from human cognition, we develop methods to evaluate and enhance agentic information-seeking. First, we introduce a decision-oriented dialogue task called Collaborative Battleship, in which a Captain must balance exploration (asking questions) and action (taking shots), while a Spotter must supply accurate, contextually-grounded answers. Compared to human players (N=42), we find that many LM agents struggle to ask informative questions, produce accurate answers, and identify high-utility actions. To address these gaps, we develop novel Monte Carlo inference strategies for LMs inspired by Bayesian Experimental Design (BED). For Spotter agents, our approach boosts accuracy by up to 14.7% absolute over LM-only baselines; for Captain agents, it raises expected information gain (EIG) by up to 0.227 bits (94.2% of the achievable noise ceiling). Combined, these components yield sharper targeting (+0.303-0.374 F1), and enable weaker LMs, such as Llama-4-Scout, to outperform both humans (8% -> 82% win rate) and frontier models (0% -> 67% win rate vs. GPT-5) at ~1% of GPT-5's cost. We replicate these findings on Guess Who?, where our methods significantly boost accuracy (+28.3-42.4 p.p.), demonstrating their general applicability for building information-seeking agents.

**arXiv ID:** 2510.20886
</details>

<details>
<summary><strong>Open-Source Based and ETSI Compliant Cooperative, Connected, and Automated Mini-Cars</strong> - Lorenzo Farina, Federico Gavioli, Salvatore Iandolo, Francesco Moretti, Giuseppe Perrone, Matteo Piccoli, Francesco Raviglione, Marco Rapelli, Antonio Solida, Paolo Burgio, Carlo Augusto Grazia, Alessandro Bazzi - [[pdf]](https://arxiv.org/pdf/2603.06343)</summary>

**Abstract:** The automotive sector is following a revolutionary path from vehicles controlled by humans to vehicles that will be fully automated, fully connected, and ultimately fully cooperative. Along this road, new cooperative algorithms and protocols will be designed and field tested, which represents a great challenge in terms of costs. In this context, in particular, moving from simulations to practical experiments requires huge investments that are not always affordable and may become a barrier in some cases. To solve this issue and provide the community with an intermediate step, we here propose the use of 1:10 scaled cooperative, autonomous, and connected mini-cars. The mini-car is equipped with a Jetson Orin board running the open Robot Operating System 2 (ROS2), sensors for autonomous operations, and a Raspberry Pi board for connectivity mounting the open source Open Stack for Car (OScar). A key aspect of the proposal is the use of OScar, which implements a full ETSI cooperative-intelligent transport systems (C-ITS) compliant stack. The feasibility and potential of the proposed platform is here demonstrated through the implementation of a case study where the Day-1 intersection collision warning (ICW) application is implemented and validated.

**arXiv ID:** 2603.06343
</details>

<details>
<summary><strong>Safe Consensus of Cooperative Manipulation with Hierarchical Event-Triggered Control Barrier Functions</strong> - Simiao Zhuang, Bingkun Huang, Zewen Yang - [[pdf]](https://arxiv.org/pdf/2603.06356)</summary>

**Abstract:** Cooperative transport and manipulation of heavy or bulky payloads by multiple manipulators requires coordinated formation tracking, while simultaneously enforcing strict safety constraints in varying environments with limited communication and real-time computation budgets. This paper presents a distributed control framework that achieves consensus coordination with safety guarantees via hierarchical event-triggered control barrier functions (CBFs). We first develop a consensus-based protocol that relies solely on local neighbor information to enforce both translational and rotational consistency in task space. Building on this coordination layer, we propose a three-level hierarchical event-triggered safety architecture with CBFs, which is integrated with a risk-aware leader selection and smooth switching strategy to reduce online computation. The proposed approach is validated through real-world hardware experiments using two Franka manipulators operating with static obstacles, as well as comprehensive simulations demonstrating scalable multi-arm cooperation with dynamic obstacles. Results demonstrate higher precision cooperation under strict safety constraints, achieving substantially reduced computational cost and communication frequency compared to baseline methods.

**arXiv ID:** 2603.06356
</details>

<details>
<summary><strong>Safe Autonomous Lane Changing: Planning with Dynamic Risk Fields and Time-Varying Convex Space Generation</strong> - Yijun Lu, Zhihao Lin, Zhen Tian - [[pdf]](https://arxiv.org/pdf/2511.22829)</summary>

**Abstract:** This paper presents a novel trajectory planning pipeline for complex driving scenarios like autonomous lane changing, by integrating risk-aware planning with guaranteed collision avoidance into a unified optimization framework. We first construct a dynamic risk fields (DRF) that captures both the static and dynamic collision risks from surrounding vehicles. Then, we develop a rigorous strategy for generating time-varying convex feasible spaces that ensure kinematic feasibility and safety requirements. The trajectory planning problem is formulated as a finite-horizon optimal control problem and solved using a constrained iterative Linear Quadratic Regulator (iLQR) algorithm that jointly optimizes trajectory smoothness, control effort, and risk exposure while maintaining strict feasibility. Extensive simulations demonstrate that our method outperforms traditional approaches in terms of safety and efficiency, achieving collision-free trajectories with shorter lane-changing distances (28.59 m) and times (2.84 s) while maintaining smooth and comfortable acceleration patterns. In dense roundabout environments the planner further demonstrates robust adaptability, producing larger safety margins, lower jerk, and superior curvature smoothness compared with APF, MPC, and RRT based baselines. These results confirm that the integrated DRF with convex feasible space and constrained iLQR solver provides a balanced solution for safe, efficient, and comfortable trajectory generation in dynamic and interactive traffic scenarios.

**arXiv ID:** 2511.22829
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (28 papers)</h2></summary>

<details>
<summary><strong>Real-Time AI Service Economy: A Framework for Agentic Computing Across the Continuum</strong> - Lauri Lovén, Alaa Saleh, Reza Farahani, Ilir Murturi, Miguel Bordallo López, Praveen Kumar Donta, Schahram Dustdar - [[pdf]](https://arxiv.org/pdf/2603.05614)</summary>

**Abstract:** Real-time AI services increasingly operate across the device-edge-cloud continuum, where autonomous AI agents generate latency-sensitive workloads, orchestrate multi-stage processing pipelines, and compete for shared resources under policy and governance constraints. This article shows that the structure of service-dependency graphs, modelled as DAGs whose nodes represent compute stages and whose edges encode execution ordering, is a primary determinant of whether decentralised, price-based resource allocation can work reliably at scale. When dependency graphs are hierarchical (tree or series-parallel), prices converge to stable equilibria, optimal allocations can be computed efficiently, and under appropriate mechanism design (with quasilinear utilities and discrete slice items), agents have no incentive to misreport their valuations within each decision epoch. When dependencies are more complex, with cross-cutting ties between pipeline stages, prices oscillate, allocation quality degrades, and the system becomes difficult to manage. To bridge this gap, we propose a hybrid management architecture in which cross-domain integrators encapsulate complex sub-graphs into resource slices that present a simpler, well-structured interface to the rest of the market. A systematic ablation study across six experiments (1,620 runs, 10 seeds each) confirms that (i) dependency-graph topology is a first-order determinant of price stability and scalability,(ii) the hybrid architecture reduces price volatility by up to 70-75% without sacrificing throughput, (iii) governance constraints create quantifiable efficiency-compliance trade-offs that depend jointly on topology and load, and (iv) under truthful bidding the decentralised market matches a centralised value-optimal baseline, confirming that decentralised coordination can replicate centralised allocation quality.

**arXiv ID:** 2603.05614
</details>

<details>
<summary><strong>Artificial Intelligence for Climate Adaptation: Reinforcement Learning for Climate Change-Resilient Transport</strong> - Miguel Costa, Arthur Vandervoort, Carolin Schmidt, João Miranda, Morten W. Petersen, Martin Drews, Karyn Morrisey, Francisco C. Pereira - [[pdf]](https://arxiv.org/pdf/2603.06278)</summary>

**Abstract:** Climate change is expected to intensify rainfall and, consequently, pluvial flooding, leading to increased disruptions in urban transportation systems over the coming decades. Designing effective adaptation strategies is challenging due to the long-term, sequential nature of infrastructure investments, deep climate uncertainty, and the complex interactions between flooding, infrastructure, and mobility impacts. In this work, we propose a novel decision-support framework using reinforcement learning (RL) for long-term flood adaptation planning. Formulated as an integrated assessment model (IAM), the framework combines rainfall projection and flood modeling, transport simulation, and quantification of direct and indirect impacts on infrastructure and mobility. Our RL-based approach learns adaptive strategies that balance investment and maintenance costs against avoided impacts. We evaluate the framework through a case study of Copenhagen's inner city over the 2024-2100 period, testing multiple adaptation options, and different belief and realized climate scenarios. Results show that the framework outperforms traditional optimization approaches by discovering coordinated spatial and temporal adaptation pathways and learning trade-offs between impact reduction and adaptation investment, yielding more resilient strategies. Overall, our results showcase the potential of reinforcement learning as a flexible decision-support tool for adaptive infrastructure planning under climate uncertainty.

**arXiv ID:** 2603.06278
</details>

<details>
<summary><strong>Boosting deep Reinforcement Learning using pretraining with Logical Options</strong> - Zihan Ye, Phil Chau, Raban Emunds, Jannis Blüml, Cedric Derstroff, Quentin Delfosse, Oleg Arenz, Kristian Kersting - [[pdf]](https://arxiv.org/pdf/2603.06565)</summary>

**Abstract:** Deep reinforcement learning agents are often misaligned, as they over-exploit early reward signals. Recently, several symbolic approaches have addressed these challenges by encoding sparse objectives along with aligned plans. However, purely symbolic architectures are complex to scale and difficult to apply to continuous settings. Hence, we propose a hybrid approach, inspired by humans' ability to acquire new skills. We use a two-stage framework that injects symbolic structure into neural-based reinforcement learning agents without sacrificing the expressivity of deep policies. Our method, called Hybrid Hierarchical RL (H^2RL), introduces a logical option-based pretraining strategy to steer the learning policy away from short-term reward loops and toward goal-directed behavior while allowing the final policy to be refined via standard environment interaction. Empirically, we show that this approach consistently improves long-horizon decision-making and yields agents that outperform strong neural, symbolic, and neuro-symbolic baselines.

**arXiv ID:** 2603.06565
</details>

<details>
<summary><strong>CORE-Seg: Reasoning-Driven Segmentation for Complex Lesions via Reinforcement Learning</strong> - Yuxin Xie, Yuming Chen, Yishan Yang, Yi Zhou, Tao Zhou, Zhen Zhao, Jiacheng Liu, Huazhu Fu - [[pdf]](https://arxiv.org/pdf/2603.05911)</summary>

**Abstract:** Medical image segmentation is undergoing a paradigm shift from conventional visual pattern matching to cognitive reasoning analysis. Although Multimodal Large Language Models (MLLMs) have shown promise in integrating linguistic and visual knowledge, significant gaps remain: existing general MLLMs possess broad common sense but lack the specialized visual reasoning required for complex lesions, whereas traditional segmentation models excel at pixel-level segmentation but lack logical interpretability. In this paper, we introduce ComLesion-14K, the first diverse Chain-of-Thought (CoT) benchmark for reasoning-driven complex lesion segmentation. To accomplish this task, we propose CORE-Seg, an end-to-end framework integrating reasoning with segmentation through a Semantic-Guided Prompt Adapter. We design a progressive training strategy from SFT to GRPO, equipped with an adaptive dual-granularity reward mechanism to mitigate reward sparsity. Our Method achieves state-of-the-art results with a mean Dice of 37.06\% (14.89\% higher than the second-best baseline), while reducing the failure rate to 18.42\%. Project Page: this https URL

**arXiv ID:** 2603.05911
</details>

<details>
<summary><strong>TADPO: Reinforcement Learning Goes Off-road</strong> - Zhouchonghao Wu, Raymond Song, Vedant Mundheda, Luis E. Navarro-Serment, Christof Schoenborn, Jeff Schneider - [[pdf]](https://arxiv.org/pdf/2603.05995)</summary>

**Abstract:** Off-road autonomous driving poses significant challenges such as navigating unmapped, variable terrain with uncertain and diverse dynamics. Addressing these challenges requires effective long-horizon planning and adaptable control. Reinforcement Learning (RL) offers a promising solution by learning control policies directly from interaction. However, because off-road driving is a long-horizon task with low-signal rewards, standard RL methods are challenging to apply in this setting. We introduce TADPO, a novel policy gradient formulation that extends Proximal Policy Optimization (PPO), leveraging off-policy trajectories for teacher guidance and on-policy trajectories for student exploration. Building on this, we develop a vision-based, end-to-end RL system for high-speed off-road driving, capable of navigating extreme slopes and obstacle-rich terrain. We demonstrate our performance in simulation and, importantly, zero-shot sim-to-real transfer on a full-scale off-road vehicle. To our knowledge, this work represents the first deployment of RL-based policies on a full-scale off-road platform.

**arXiv ID:** 2603.05995
</details>

<details>
<summary><strong>TaPD: Temporal-adaptive Progressive Distillation for Observation-Adaptive Trajectory Forecasting in Autonomous Driving</strong> - Mingyu Fan, Yi Liu, Hao Zhou, Deheng Qian, Mohammad Haziq Khan, Matthias Raetsch - [[pdf]](https://arxiv.org/pdf/2603.06231)</summary>

**Abstract:** Trajectory prediction is essential for autonomous driving, enabling vehicles to anticipate the motion of surrounding agents to support safe planning. However, most existing predictors assume fixed-length histories and suffer substantial performance degradation when observations are variable or extremely short in real-world settings (e.g., due to occlusion or a limited sensing range). We propose TaPD (Temporal-adaptive Progressive Distillation), a unified plug-and-play framework for observation-adaptive trajectory forecasting under variable history lengths. TaPD comprises two cooperative modules: an Observation-Adaptive Forecaster (OAF) for future prediction and a Temporal Backfilling Module (TBM) for explicit reconstruction of the past. OAF is built on progressive knowledge distillation (PKD), which transfers motion pattern knowledge from long-horizon "teachers" to short-horizon "students" via hierarchical feature regression, enabling short observations to recover richer motion context. We further introduce a cosine-annealed distillation weighting scheme to balance forecasting supervision and feature alignment, improving optimization stability and cross-length consistency. For extremely short histories where implicit alignment is insufficient, TBM backfills missing historical segments conditioned on scene evolution, producing context-rich trajectories that strengthen PKD and thereby improve OAF. We employ a decoupled pretrain-reconstruct-finetune protocol to preserve real-motion priors while adapting to backfilled inputs. Extensive experiments on Argoverse 1 and Argoverse 2 show that TaPD consistently outperforms strong baselines across all observation lengths, delivers especially large gains under very short inputs, and improves other predictors (e.g., HiVT) in a plug-and-play manner. Code will be available at this https URL.

**arXiv ID:** 2603.06231
</details>

<details>
<summary><strong>Agentic retrieval-augmented reasoning reshapes collective reliability under model variability in radiology question answering</strong> - Mina Farajiamiri, Jeta Sopa, Saba Afza, Lisa Adams, Felix Barajas Ordonez, Tri-Thien Nguyen, Mahshad Lotfinia, Sebastian Wind, Keno Bressem, Sven Nebelung, Daniel Truhn, Soroosh Tayebi Arasteh - [[pdf]](https://arxiv.org/pdf/2603.06271)</summary>

**Abstract:** Agentic retrieval-augmented reasoning pipelines are increasingly used to structure how large language models (LLMs) incorporate external evidence in clinical decision support. These systems iteratively retrieve curated domain knowledge and synthesize it into structured reports before answer selection. Although such pipelines can improve performance, their impact on reliability under model variability remains unclear. In real-world deployment, heterogeneous models may align, diverge, or synchronize errors in ways not captured by accuracy. We evaluated 34 LLMs on 169 expert-curated publicly available radiology questions, comparing zero-shot inference with a radiology-specific multi-step agentic retrieval condition in which all models received identical structured evidence reports derived from curated radiology knowledge. Agentic inference reduced inter-model decision dispersion (median entropy 0.48 vs. 0.13) and increased robustness of correctness across models (mean 0.74 vs. 0.81). Majority consensus also increased overall (P<0.001). Consensus strength and robust correctness remained correlated under both strategies (\r{ho}=0.88 for zero-shot; \r{ho}=0.87 for agentic), although high agreement did not guarantee correctness. Response verbosity showed no meaningful association with correctness. Among 572 incorrect outputs, 72% were associated with moderate or high clinically assessed severity, although inter-rater agreement was low (\k{appa}=0.02). Agentic retrieval therefore was associated with more concentrated decision distributions, stronger consensus, and higher cross-model robustness of correctness. These findings suggest that evaluating agentic systems through accuracy or agreement alone may not always be sufficient, and that complementary analyses of stability, cross-model robustness, and potential clinical impact are needed to characterize reliability under model variability.

**arXiv ID:** 2603.06271
</details>

<details>
<summary><strong>ESAA-Security: An Event-Sourced, Verifiable Architecture for Agent-Assisted Security Audits of AI-Generated Code</strong> - Elzo Brito dos Santos Filho - [[pdf]](https://arxiv.org/pdf/2603.06365)</summary>

**Abstract:** AI-assisted software generation has increased development speed, but it has also amplified a persistent engineering problem: systems that are functionally correct may still be structurally insecure. In practice, prompt-based security review with large language models often suffers from uneven coverage, weak reproducibility, unsupported findings, and the absence of an immutable audit trail. The ESAA architecture addresses a related governance problem in agentic software engineering by separating heuristic agent cognition from deterministic state mutation through append-only events, constrained outputs, and replay-based verification. This paper presents ESAA-Security, a domain-specific specialization of ESAA for agent-assisted security auditing of software repositories, with particular emphasis on AI-generated or AI-modified code. ESAA-Security structures auditing as a governed execution pipeline with four phases reconnaissance, domain audit execution, risk classification, and final reporting and operationalizes the workflow into 26 tasks, 16 security domains, and 95 executable checks. The framework produces structured check results, vulnerability inventories, severity classifications, risk matrices, remediation guidance, executive summaries, and a final markdown/JSON audit report. The central idea is that security review should not be modeled as a free-form conversation with an LLM, but as an evidence-oriented audit process governed by contracts and events. In ESAA-Security, agents emit structured intentions under constrained protocols; the orchestrator validates them, persists accepted outputs to an append-only log, reprojects derived views, and verifies consistency through replay and hashing. The result is a traceable, reproducible, and risk-oriented audit architecture whose final report is auditable by construction.

**arXiv ID:** 2603.06365
</details>

<details>
<summary><strong>A Reference Architecture of Reinforcement Learning Frameworks</strong> - Xiaoran Liu, Istvan David - [[pdf]](https://arxiv.org/pdf/2603.06413)</summary>

**Abstract:** The surge in reinforcement learning (RL) applications gave rise to diverse supporting technology, such as RL frameworks. However, the architectural patterns of these frameworks are inconsistent across implementations and there exists no reference architecture (RA) to form a common basis of comparison, evaluation, and integration. To address this gap, we propose an RA of RL frameworks. Through a grounded theory approach, we analyze 18 state-of-the-practice RL frameworks and, by that, we identify recurring architectural components and their relationships, and codify them in an RA. To demonstrate our RA, we reconstruct characteristic RL patterns. Finally, we identify architectural trends, e.g., commonly used components, and outline paths to improving RL frameworks.

**arXiv ID:** 2603.06413
</details>

<details>
<summary><strong>Understanding and Improving Hyperbolic Deep Reinforcement Learning</strong> - Timo Klein, Thomas Lang, Andrii Shkabrii, Alexander Sturm, Kevin Sidak, Lukas Miklautz, Claudia Plant, Yllka Velaj, Sebastian Tschiatschek - [[pdf]](https://arxiv.org/pdf/2512.14202)</summary>

**Abstract:** The exponential volume growth of hyperbolic geometry can embed the hierarchical relationships between states in reinforcement learning (RL) with far less distortion than Euclidean space. However, hyperbolic deep RL faces severe optimization challenges, and formal analysis of why optimization fails is lacking. We identify key factors that determine the success and failure of training hyperbolic deep RL agents. By analyzing the gradients of core operations in the Poincaré Ball and Hyperboloid models of hyperbolic geometry, we show that large-norm embeddings destabilize gradient-based training, leading to trust-region violations in proximal policy optimization (PPO). Based on these insights, we introduce Hyper++, a new hyperbolic deep RL agent that consists of three components: (1) feature regularization guaranteeing bounded norms while avoiding the curse of dimensionality from clipping; (2) a categorical value loss for stable critic training; and (3) a more optimization-friendly formulation of hyperbolic network layers. On ProcGen, we show that Hyper++ guarantees stable learning, outperforms prior hyperbolic agents, and reduces wall-clock time by approximately 30%. On Atari-5 with Double DQN, Hyper++ strongly outperforms Euclidean and hyperbolic baselines. We release our code at this https URL.

**arXiv ID:** 2512.14202
</details>

<details>
<summary><strong>DataChef: Cooking Up Optimal Data Recipes for LLM Adaptation via Reinforcement Learning</strong> - Yicheng Chen, Zerun Ma, Xinchen Xie, Yining Li, Kai Chen - [[pdf]](https://arxiv.org/pdf/2602.11089)</summary>

**Abstract:** In the current landscape of Large Language Models (LLMs), the curation of large-scale, high-quality training data is a primary driver of model performance. A key lever is the \emph{data recipe}, which comprises a data processing pipeline to transform raw sources into training corpora. Despite the growing use of LLMs to automate individual data processing steps, such as data synthesis and filtering, the overall design of data recipes remains largely manual and labor-intensive, requiring substantial human expertise and iteration. To bridge this gap, we formulate \emph{end-to-end data recipe generation} for LLM adaptation. Given a target benchmark and a pool of available data sources, a model is required to output a complete data recipe that adapts a base LLM to the target task. We present DataChef-32B, which performs online reinforcement learning using a proxy reward that predicts downstream performance for candidate recipes. Across six held-out tasks, DataChef-32B produces recipes that yield performance comparable to those curated by human experts. Notably, the recipe from DataChef-32B adapts Qwen3-1.7B-Base to the math domain, achieving 66.7 on AIME'25 and surpassing the official post-training checkpoint (Qwen3-1.7B). This work sheds new light on automating LLM training and developing self-evolving AI systems.

**arXiv ID:** 2602.11089
</details>

<details>
<summary><strong>SWE-MiniSandbox: Container-Free Reinforcement Learning for Building Software Engineering Agents</strong> - Danlong Yuan, Wei Wu, Zhengren Wang, Xueliang Zhao, Huishuai Zhang, Dongyan Zhao - [[pdf]](https://arxiv.org/pdf/2602.11210)</summary>

**Abstract:** Reinforcement learning (RL) has become a key paradigm for training software engineering (SWE) agents, but existing pipelines typically rely on per-task containers for isolation. At scale, pre-built container images incur substantial storage overhead, slow environment setup, and require container-management privileges. We propose SWE-MiniSandbox, a lightweight, container-free method that enables scalable RL training of SWE agents without sacrificing isolation. Instead of relying on per-instance containers, SWE-MiniSandbox executes each task in an isolated workspace backed by kernel-level mechanisms, substantially reducing system overhead. It leverages lightweight environment pre-caching techniques to eliminate the need for bulky container images. As a result, our approach lowers disk usage to approximately 5\% of that required by container-based pipelines and reduces environment preparation time to about 25\% of the container baseline. Empirical results demonstrate that SWE-MiniSandbox achieves evaluation performance comparable to standard container-based pipelines. By removing the dependency on heavy container infrastructure, SWE-MiniSandbox offers a practical and accessible foundation for scaling RL-based SWE agents, particularly in resource-constrained research environments.

**arXiv ID:** 2602.11210
</details>

<details>
<summary><strong>Theory of Code Space: Do Code Agents Understand Software Architecture?</strong> - Grigory Sapunov - [[pdf]](https://arxiv.org/pdf/2603.00601)</summary>

**Abstract:** AI code agents excel at isolated tasks yet struggle with multi-file software engineering requiring architectural understanding. We introduce Theory of Code Space (ToCS), a benchmark that evaluates whether agents can construct, maintain, and update coherent architectural beliefs during codebase exploration. Agents explore procedurally generated codebases under partial observability -- opening files under a budget -- and periodically externalize their belief state as structured JSON, producing a time-series of architectural understanding. Three findings emerge from experiments with four baselines and six frontier LLMs. First, the Active-Passive Gap is model-dependent: one model builds better maps through active exploration than from seeing all files at once, while another shows the opposite -- revealing that active exploration is itself a non-trivial capability absent from some models. Second, retaining structured belief maps in context acts as self-scaffolding for some models but not others, showing that the mechanism is model-dependent. Third, belief state maintenance varies dramatically: a smaller model maintains perfectly stable beliefs across probes while its larger sibling suffers catastrophic belief collapse -- forgetting previously-discovered components between probes. We release ToCS as open-source software. Code: this https URL

**arXiv ID:** 2603.00601
</details>

<details>
<summary><strong>CodeScout: Contextual Problem Statement Enhancement for Software Agents</strong> - Manan Suri, Xiangci Li, Mehdi Shojaie, Songyang Han, Chao-Chun Hsu, Shweta Garg, Aniket Anand Deshmukh, Varun Kumar - [[pdf]](https://arxiv.org/pdf/2603.05744)</summary>

**Abstract:** Current AI-powered code assistance tools often struggle with poorly-defined problem statements that lack sufficient task context and requirements specification. Recent analysis of software engineering agents reveals that failures on such underspecified requests are highly correlated with longer trajectories involving either over-exploration or repeated attempts at applying the same fix without proper evolution or testing, leading to suboptimal outcomes across software development tasks. We introduce CodeScout, a contextual query refinement approach that systematically converts underspecified user requests into comprehensive, actionable problem statements through lightweight pre-exploration of the target codebase. Our key innovation is demonstrating that structured analysis before task execution can supplement existing agentic capabilities without requiring any modifications to their underlying scaffolds. CodeScout performs targeted context scoping, conducts multi-perspective analysis examining potential fixes and exploration opportunities, then synthesizes these insights into enhanced problem statements with reproduction steps, expected behaviors, and targeted exploration hints. This pre-exploration directly addresses the identified failure patterns by reducing non-converging agent trajectories while clarifying user intent in natural language space. We evaluate CodeScout using state-of-the-art agentic scaffolds and language models on SWEBench-Verified, demonstrating a 20\% improvement in resolution rates with up to 27 additional issues resolved compared to the default baseline method. Our results suggest that systematic query refinement through contextual analysis represents a promising direction for enhancing AI code assistance capabilities.

**arXiv ID:** 2603.05744
</details>

<details>
<summary><strong>ReflexiCoder: Teaching Large Language Models to Self-Reflect on Generated Code and Self-Correct It via Reinforcement Learning</strong> - Juyong Jiang, Jiasi Shen, Sunghun Kim, Kang Min Yoo, Jeonghoon Kim, Sungju Kim - [[pdf]](https://arxiv.org/pdf/2603.05863)</summary>

**Abstract:** While Large Language Models (LLMs) have revolutionized code generation, standard "System 1" approaches, generating solutions in a single forward pass, often hit a performance ceiling when faced with complex algorithmic tasks. Existing iterative refinement strategies attempt to bridge this gap at inference time, yet they predominantly rely on external oracles, execution feedback, or computationally expensive prompt-response cycles. In this work, we propose ReflexiCoder, a novel reinforcement learning (RL) framework that internalizes the structured reasoning trajectory, encompassing initial generation, bug and optimization aware reflection, and self-correction, directly into the model's weights. Unlike prior methods, ReflexiCoder shifts the paradigm from external-dependent refinement to an intrinsic, fully autonomous self-reflection and self-correction capabilities at inference time. We utilize an RL-zero training paradigm with granular reward functions to optimize the entire reflection-correction trajectory, teaching the model how to debug without reliance on ground-truth feedback or execution engines at inference time. Extensive experiments across seven benchmarks demonstrate that our ReflexiCoder-8B establishes a new state-of-the-art (SOTA) among leading open-source models in the 1.5B-14B range, achieving 94.51% (87.20%) on HumanEval (Plus), 81.80% (78.57%) on MBPP (Plus), 35.00% on BigCodeBench, 52.21% on LiveCodeBench, and 37.34% on CodeForces in a single-attempt setting, rivaling or surpassing proprietary models like GPT-5.1. Notably, our framework is significantly more token-efficient than base models, reducing inference-time compute overhead by approximately 40% through disciplined, high-speed reasoning and reflection patterns. Source code is available at this https URL.

**arXiv ID:** 2603.05863
</details>

<details>
<summary><strong>Beyond Rows to Reasoning: Agentic Retrieval for Multimodal Spreadsheet Understanding and Editing</strong> - Anmol Gulati, Sahil Sen, Waqar Sarguroh, Kevin Paul - [[pdf]](https://arxiv.org/pdf/2603.06503)</summary>

**Abstract:** Recent advances in multimodal Retrieval-Augmented Generation (RAG) enable Large Language Models (LLMs) to analyze enterprise spreadsheet workbooks containing millions of cells, cross-sheet dependencies, and embedded visual artifacts. However, state-of-the-art approaches exclude critical context through single-pass retrieval, lose data resolution through compression, and exceed LLM context windows through naive full-context injection, preventing reliable multi-step reasoning over complex enterprise workbooks. We introduce Beyond Rows to Reasoning (BRTR), a multimodal agentic framework for spreadsheet understanding that replaces single-pass retrieval with an iterative tool-calling loop, supporting end-to-end Excel workflows from complex analysis to structured editing. Supported by over 200 hours of expert human evaluation, BRTR achieves state-of-the-art performance across three frontier spreadsheet understanding benchmarks, surpassing prior methods by 25 percentage points on FRTR-Bench, 7 points on SpreadsheetLLM, and 32 points on FINCH. We evaluate five multimodal embedding models, identifying NVIDIA NeMo Retriever 1B as the top performer for mixed tabular and visual data, and vary nine LLMs. Ablation experiments confirm that the planner, retrieval, and iterative reasoning each contribute substantially, and cost analysis shows GPT-5.2 achieves the best efficiency-accuracy trade-off. Throughout all evaluations, BRTR maintains full auditability through explicit tool-call traces.

**arXiv ID:** 2603.06503
</details>

<details>
<summary><strong>SPINE: Token-Selective Test-Time Reinforcement Learning with Entropy-Band Regularization</strong> - Jianghao Wu, Yasmeen George, Jin Ye, Yicheng Wu, Daniel F. Schmidt, Jianfei Cai - [[pdf]](https://arxiv.org/pdf/2511.17938)</summary>

**Abstract:** Large language models (LLMs) and multimodal LLMs (MLL-Ms) excel at chain-of-thought reasoning but face distribution shift at test-time and a lack of verifiable supervision. Recent test-time reinforcement learning (TTRL) methods derive label-free pseudo-rewards from self-consistency voting over sampled trajectories, yet they often collapse: the majority-vote reward prevails, responses shorten, and Pass@1 declines. We trace this to uniform sequence updates in which most tokens are low-entropy followers, while a small high-entropy subset determines the reasoning branches. Thus we propose \method, a token-selective test-time reinforcement learning framework that (i) performs distribution-aware forking-token selection to update only decision-critical branch points, and (ii) applies a robust entropy-band regularizer at those tokens to prevent premature collapse and suppress noisy drift. \method plugs into GRPO-style objectives (optionally with a KL anchor) and requires neither labels nor reward models. Across eight benchmarks spanning multimodal VQA, text-only reasoning, \method consistently improves Pass@1 over TTRL while avoiding response-length collapse and yielding more stable training dynamics on both LLM and MLLM backbones. These results indicate that aligning updates with chain-of-thought branch points is a simple and label-free mechanism for stable and effective test-time adaptation in reasoning models. Code will be released.

**arXiv ID:** 2511.17938
</details>

<details>
<summary><strong>A Novel Hybrid Heuristic-Reinforcement Learning Optimization Approach for a Class of Railcar Shunting Problems</strong> - Ruonan Zhao, Joseph Geunes - [[pdf]](https://arxiv.org/pdf/2603.05579)</summary>

**Abstract:** Railcar shunting is a core planning task in freight railyards, where yard planners need to disassemble and reassemble groups of railcars to form outbound trains. Classification tracks with access from one side only can be considered as stack structures, where railcars are added and removed from only one end, leading to a last-in-first-out (LIFO) retrieval order. In contrast, two-sided tracks function like queue structures, allowing railcars to be added from one end and removed from the opposite end, following a first-in-first-out (FIFO) order. We consider a problem requiring assembly of multiple outbound trains using two locomotives in a railyard with two-sided classification track access. To address this combinatorially challenging problem class, we decompose the problem into two subproblems, each with one-sided classification track access and a locomotive on each side. We present a novel Hybrid Heuristic-Reinforcement Learning (HHRL) framework that integrates railway-specific heuristic solution approaches with a reinforcement learning method, specifically Q-learning. The proposed framework leverages methods to decrease the state-action space and guide exploration during reinforcement learning. The results of a series of numerical experiments demonstrate the efficiency and quality of the HHRL algorithm in both one-sided access, single-locomotive problems and two-sided access, two-locomotive problems.

**arXiv ID:** 2603.05579
</details>

<details>
<summary><strong>Reinforcement Learning for Power-Flow Network Analysis</strong> - Alperen Ergur, Julia Lindberg, Vinny Miller - [[pdf]](https://arxiv.org/pdf/2603.05673)</summary>

**Abstract:** The power flow equations are non-linear multivariate equations that describe the relationship between power injections and bus voltages of electric power networks. Given a network topology, we are interested in finding network parameters with many equilibrium points. This corresponds to finding instances of the power flow equations with many real solutions. Current state-of-the art algorithms in computational algebra are not capable of answering this question for networks involving more than a small number of variables. To remedy this, we design a probabilistic reward function that gives a good approximation to this root count, and a state-space that mimics the space of power flow equations. We derive the average root count for a Gaussian model, and use this as a baseline for our RL agents. The agents discover instances of the power flow equations with many more solutions than the average baseline. This demonstrates the potential of RL for power-flow network design and analysis as well as the potential for RL to contribute meaningfully to problems that involve complex non-linear algebra or geometry. \footnote{Author order alphabetic, all authors contributed equally.

**arXiv ID:** 2603.05673
</details>

<details>
<summary><strong>MIRACL: A Diverse Meta-Reinforcement Learning for Multi-Objective Multi-Echelon Combinatorial Supply Chain Optimisation</strong> - Rifny Rachman, Josh Tingey, Richard Allmendinger, Wei Pan, Pradyumn Shukla, Bahrul Ilmi Nasution - [[pdf]](https://arxiv.org/pdf/2603.05760)</summary>

**Abstract:** Multi-objective reinforcement learning (MORL) is effective for multi-echelon combinatorial supply chain optimisation, where tasks involve high dimensionality, uncertainty, and competing objectives. However, its deployment in dynamic environments is hindered by the need for task-specific retraining and substantial computational cost. We introduce MIRACL (Meta multI-objective Reinforcement leArning with Composite Learning), a hierarchical Meta-MORL framework that allows for a few-shot generalisation across diverse tasks. MIRACL decomposes each task into structured subproblems for efficient policy adaptation and meta-learns a global policy across tasks using a Pareto-based adaptation strategy to encourage diversity in meta-training and fine-tuning. To our knowledge, this is the first integration of Meta-MORL with such mechanisms in combinatorial optimisation. Although validated in the supply chain domain, MIRACL is theoretically domain-agnostic and applicable to broader dynamic multi-objective decision-making problems. Empirical evaluations show that MIRACL outperforms conventional MORL baselines in simple to moderate tasks, achieving up to 10% higher hypervolume and 5% better expected utility. These results underscore the potential of MIRACL for robust, efficient adaptation in multi-objective problems.

**arXiv ID:** 2603.05760
</details>

<details>
<summary><strong>Synthetic Monitoring Environments for Reinforcement Learning</strong> - Leonard Pleiss, Carolin Schmidt, Maximilian Schiffer - [[pdf]](https://arxiv.org/pdf/2603.06252)</summary>

**Abstract:** Reinforcement Learning (RL) lacks benchmarks that enable precise, white-box diagnostics of agent behavior. Current environments often entangle complexity factors and lack ground-truth optimality metrics, making it difficult to isolate why algorithms fail. We introduce Synthetic Monitoring Environments (SMEs), an infinite suite of continuous control tasks. SMEs provide fully configurable task characteristics and known optimal policies. As such, SMEs allow for the exact calculation of instantaneous regret. Their rigorous geometric state space bounds allow for systematic within-distribution (WD) and out-of-distribution (OOD) evaluation. We demonstrate the framework's benefit through multidimensional ablations of PPO, TD3, and SAC, revealing how specific environmental properties - such as action or state space size, reward sparsity and complexity of the optimal policy - impact WD and OOD performance. We thereby show that SMEs offer a standardized, transparent testbed for transitioning RL evaluation from empirical benchmarking toward rigorous scientific analysis.

**arXiv ID:** 2603.06252
</details>

<details>
<summary><strong>TIC-GRPO: Provable and Efficient Optimization for Reinforcement Learning from Human Feedback</strong> - Lei Pang, Jun Luo, Ruinan Jin - [[pdf]](https://arxiv.org/pdf/2508.02833)</summary>

**Abstract:** Group Relative Policy Optimization (GRPO), recently introduced by DeepSeek, is a critic-free reinforcement learning algorithm for fine-tuning large language models. GRPO replaces the value function in Proximal Policy Optimization (PPO) with group-normalized rewards while retaining PPO-style token-level importance sampling based on an old policy. Our theoretical analysis reveals that the GRPO update rule estimates the policy gradient at the old policy rather than the current one; however, since the old policy is refreshed every few steps, the resulting discrepancy remains small and the induced bias is negligible in practice. To empirically validate this insight, we conduct an ablation study that entirely removes importance sampling and performs multiple optimization steps using gradients estimated at a fixed old policy. Remarkably, this simplified variant attains performance comparable to standard GRPO.
Motivated by this finding, we propose Trajectory-level Importance-Corrected GRPO (TIC-GRPO), a new algorithm that replaces token-level importance ratios with a single trajectory-level probability ratio, thereby yielding an estimate of the current policy gradient while preserving the critic-free structure. Furthermore, we present the first convergence analysis for GRPO-style methods and show that TIC-GRPO converges faster than GRPO. Finally, empirical results across math reasoning and coding tasks demonstrate the superiority of TIC-GRPO.

**arXiv ID:** 2508.02833
</details>

<details>
<summary><strong>VEGA: Electric Vehicle Navigation Agent via Physics-Informed Neural Operator and Proximal Policy Optimization</strong> - Hansol Lim, Minhyeok Im, Jonathan Boyack, Jee Won Lee, Jongseong Brad Choi - [[pdf]](https://arxiv.org/pdf/2509.13386)</summary>

**Abstract:** We present VEGA, a vehicle-adaptive energy-aware routing system for electric vehicles (EVs) that integrates physics-informed parameter estimation with RL-based charge-aware path planning. VEGA consists of two copupled modules: (1) a physics-informed neural operator (PINO) that estimates vehicle-specific physical parameters-drag, rolling resistance, mass, motor and regenerative-braking efficiencies, and auxiliary load-from short windows of onboard speed and acceleration data; (2) a Proximal Policy Optimization (PPO) agent that navigates a charger-annotated road graph, jointly selecting routes and charging stops under state-of-charge constraints. The agent is initialized via behavior cloning from an A* teacher and fine-tuned with cirriculum-guided PPO on the full U.S. highway network with Tesla Supercharger locations. On a cross-country San Francisco-to-New York route (~4,860km), VEGA produces a feasible 20-stop plan with 56.12h total trip time and minimum SoC 11.41%. Against the controlled Energy-aware A* baseline, the distance and driving-time gaps are small (-8.49km and +0.37h), while inference is >20x faster. The learned policy generalizes without retraining to road networks in France and Japan.

**arXiv ID:** 2509.13386
</details>

<details>
<summary><strong>Expert Knowledge-driven Reinforcement Learning for Autonomous Racing via Trajectory Guidance and Dynamics Constraints</strong> - Bo Leng, Weiqi Zhang, Zhuoren Li, Lu Xiong, Guizhe Jin, Ran Yu, Chen Lv - [[pdf]](https://arxiv.org/pdf/2603.05842)</summary>

**Abstract:** Reinforcement learning has demonstrated significant potential in the field of autonomous driving. However, it suffers from defects such as training instability and unsafe action outputs when faced with autonomous racing environments characterized by high dynamics and strong nonlinearities. To this end, this paper proposes a trajectory guidance and dynamics constraints Reinforcement Learning (TraD-RL) method for autonomous racing. The key features of this method are as follows: 1) leveraging the prior expert racing line to construct an augmented state representation and facilitate reward shaping, thereby integrating domain knowledge to stabilize early-stage policy learning; 2) embedding explicit vehicle dynamic priors into a safe operating envelope formulated via control barrier functions to enable safety-constrained learning; and 3) adopting a multi-stage curriculum learning strategy that shifts from expert-guided learning to autonomous exploration, allowing the learned policy to surpass expert-level performance. The proposed method is evaluated in a high-fidelity simulation environment modeled after the Tempelhof Airport Street Circuit. Experimental results demonstrate that TraD-RL effectively improves both lap speed and driving stability of the autonomous racing vehicle, achieving a synergistic optimization of racing performance and safety.

**arXiv ID:** 2603.05842
</details>

<details>
<summary><strong>Dual-Agent Multiple-Model Reinforcement Learning for Event-Triggered Human-Robot Co-Adaptation in Decoupled Task Spaces</strong> - Yaqi Li, Zhengqi Han, Huifang Liu, Steven W.Su - [[pdf]](https://arxiv.org/pdf/2603.06163)</summary>

**Abstract:** This paper presents a shared-control rehabilitation policy for a custom 6-degree-of-freedom (6-DoF) upper-limb robot that decomposes complex reaching tasks into decoupled spatial axes. The patient governs the primary reaching direction using binary commands, while the robot autonomously manages orthogonal corrective motions. Because traditional fixed-frequency control often induces trajectory oscillations due to variable inverse-kinematics execution times, an event-driven progression strategy is proposed. This architecture triggers subsequent control actions only when the end-effector enters an admission sphere centred on the immediate target waypoint, and was validated in a semi-virtual setup linking a physical pressure sensor to a MuJoCo simulation. To optimise human--robot co-adaptation safely and efficiently, this study introduces Dual Agent Multiple Model Reinforcement Learning (DAMMRL). This framework discretises decision characteristics: the human agent selects the admission sphere radius to reflect their inherent speed--accuracy trade-off, while the robot agent dynamically adjusts its 3D Cartesian step magnitudes to complement the user's cognitive state. Trained in simulation and deployed across mixed environments, this event-triggered DAMMRL approach effectively suppresses waypoint chatter, balances spatial precision with temporal efficiency, and significantly improves success rates in object acquisition tasks.

**arXiv ID:** 2603.06163
</details>

<details>
<summary><strong>Contact-Safe Reinforcement Learning with ProMP Reparameterization and Energy Awareness</strong> - Bingkun Huang, Yuhe Gong, Zewen Yang, Tianyu Ren, Luis Figueredo - [[pdf]](https://arxiv.org/pdf/2511.13459)</summary>

**Abstract:** Reinforcement learning (RL) approaches based on Markov Decision Processes (MDPs) are predominantly applied in the robot joint space, often relying on limited task-specific information and partial awareness of the 3D environment. In contrast, episodic RL has demonstrated advantages over traditional MDP-based methods in terms of trajectory consistency, task awareness, and overall performance in complex robotic tasks. Moreover, traditional step-wise and episodic RL methods often neglect the contact-rich information inherent in task-space manipulation, especially considering the contact-safety and robustness. In this work, contact-rich manipulation tasks are tackled using a task-space, energy-safe framework, where reliable and safe task-space trajectories are generated through the combination of Proximal Policy Optimization (PPO) and movement primitives. Furthermore, an energy-aware Cartesian Impedance Controller objective is incorporated within the proposed framework to ensure safe interactions between the robot and the environment. Our experimental results demonstrate that the proposed framework outperforms existing methods in handling tasks on various types of surfaces in 3D environments, achieving high success rates as well as smooth trajectories and energy-safe interactions.

**arXiv ID:** 2511.13459
</details>

<details>
<summary><strong>Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models</strong> - Liangzhi Shi, Shuaihang Chen, Feng Gao, Yinuo Chen, Kang Chen, Tonghe Zhang, Hongzhi Zang, Weinan Zhang, Chao Yu, Yu Wang - [[pdf]](https://arxiv.org/pdf/2602.12628)</summary>

**Abstract:** Simulation offers a scalable and low-cost way to enrich vision-language-action (VLA) training, reducing reliance on expensive real-robot demonstrations. However, most sim-real co-training methods rely on supervised fine-tuning (SFT), which treats simulation as a static source of demonstrations and does not exploit large-scale closed-loop interaction. Consequently, real-world gains and generalization are often limited. In this paper, we propose an \underline{\textit{RL}}-based sim-real \underline{\textit{Co}}-training \modify{(RL-Co)} framework that leverages interactive simulation while preserving real-world capabilities. Our method follows a generic two-stage design: we first warm-start the policy with SFT on a mixture of real and simulated demonstrations, then fine-tune it with reinforcement learning in simulation while adding an auxiliary supervised loss on real-world data to anchor the policy and mitigate catastrophic forgetting. We evaluate our framework on four real-world tabletop manipulation tasks using two representative VLA architectures, OpenVLA and $\pi_{0.5}$, and observe consistent improvements over real-only fine-tuning and SFT-based co-training, including +24% real-world success on OpenVLA and +20% on $\pi_{0.5}$. Beyond higher success rates, RL co-training yields stronger generalization to unseen task variations and substantially improved real-world data efficiency, providing a practical and scalable pathway for leveraging simulation to enhance real-robot deployment.

**arXiv ID:** 2602.12628
</details>

<details>
<summary><strong>Beyond Scores: Explainable Intelligent Assessment Strengthens Pre-service Teachers' Assessment Literacy</strong> - Yuang Wei, Fei Wang, Yifan Zhang, Brian Y. Lim, Bo Jiang - [[pdf]](https://arxiv.org/pdf/2603.06059)</summary>

**Abstract:** Assessment literacy (AL) is essential for personalized education, yet difficult to cultivate in pre-service teachers. Conventional teacher preparation programs focus on theoretical knowledge, while digital assessment tools commonly provide opaque scores or parameters. These limitations hinder reflection and transfer, leaving AL underdeveloped. We propose XIA, an eXplainable Intelligent Assessment platform that extends statistics-informed support with visualized cognitive diagnostic reasoning, including contrastive and counterfactual explanations. In a pre-post controlled study with 21 pre-service teachers, we combined quantitative tasks and questionnaires with qualitative interviews. The findings offer preliminary evidence that XIA supported reflection, self-regulation, and assessment awareness, and helped reduce assessment errors. Interviews further showed a shift from score-based judgments toward evidence-based reasoning. This work contributes insights into the design of intelligent assessment tools, showing how explanatory scaffolding can bridge assessment theory and classroom practice and support the cultivation of AL in teacher education.

**arXiv ID:** 2603.06059
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
