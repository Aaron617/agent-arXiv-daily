# Agent arXiv Daily

**Last Updated:** 2026-02-26 03:41:54

**Total Papers:** 70

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
<summary><strong>Budget-Aware Agentic Routing via Boundary-Guided Training</strong> - Caiqi Zhang, Menglin Xia, Xuchao Zhang, Daniel Madrigal, Ankur Mallick, Samuel Kessler, Victor Ruehle, Saravan Rajmohan - [[pdf]](https://arxiv.org/pdf/2602.21227)</summary>

**Abstract:** As large language models (LLMs) evolve into autonomous agents that execute long-horizon workflows, invoking a high-capability model at every step becomes economically unsustainable. While model routing is effective for single-turn queries, agentic routing is a sequential, path-dependent problem: early mistakes compound, feedback is often at the end of the episode, and deployments often demand strict per-task spending limits. We propose Budget-Aware Agentic Routing, which selects between a cheap and an expensive model at each step to optimize the cost--success frontier and to operate under strict per-task budgets. We propose Boundary-Guided Training, which leverages two boundary policies (always-small vs.\ always-large) to build a difficulty taxonomy and to anchor learning under sparse rewards. Our approach warms start with boundary-guided SFT data synthesis via stratified sampling of cost-efficient trajectories, then applies Boundary-Guided Policy Optimization (BoPO), combining boundary-relative rewards with a reference-guided advantage to avoid degenerate cheap-failure solutions. Experiment results show that our method improves the efficiency frontier, matching strong routing baselines at substantially lower cost while demonstrating generalization to strict inference-time budget constraints. Overall, our work establishes a foundational framework for agentic routing, shifting the paradigm from static model selection to dynamic, budget-aware sequential decision-making.

**arXiv ID:** 2602.21227
</details>

<details>
<summary><strong>Toward Ultra-Long-Horizon Agentic Science: Cognitive Accumulation for Machine Learning Engineering</strong> - Xinyu Zhu, Yuzhu Cai, Zexi Liu, Bingyang Zheng, Cheng Wang, Rui Ye, Yuzhi Zhang, Linfeng Zhang, Weinan E, Siheng Chen, Yanfeng Wang - [[pdf]](https://arxiv.org/pdf/2601.10402)</summary>

**Abstract:** The advancement of artificial intelligence toward agentic science is currently bottlenecked by the challenge of ultra-long-horizon autonomy, the ability to sustain strategic coherence and iterative correction over experimental cycles spanning days or weeks. While Large Language Models (LLMs) have demonstrated prowess in short-horizon reasoning, they are easily overwhelmed by execution details in the high-dimensional, delayed-feedback environments of real-world research, failing to consolidate sparse feedback into coherent long-term guidance. Here, we present ML-Master 2.0, an autonomous agent that masters ultra-long-horizon machine learning engineering (MLE) which is a representative microcosm of scientific discovery. By reframing context management as a process of cognitive accumulation, our approach introduces Hierarchical Cognitive Caching (HCC), a multi-tiered architecture inspired by computer systems that enables the structural differentiation of experience over time. By dynamically distilling transient execution traces into stable knowledge and cross-task wisdom, HCC allows agents to decouple immediate execution from long-term experimental strategy, effectively overcoming the scaling limits of static context windows. In evaluations on OpenAI's MLE-Bench under 24-hour budgets, ML-Master 2.0 achieves a state-of-the-art medal rate of 56.44%. Our findings demonstrate that ultra-long-horizon autonomy provides a scalable blueprint for AI capable of autonomous exploration beyond human-precedent complexities.

**arXiv ID:** 2601.10402
</details>

<details>
<summary><strong>UNet-Based Keypoint Regression for 3D Cone Localization in Autonomous Racing</strong> - Mariia Baidachna, James Carty, Aidan Ferguson, Joseph Agrane, Varad Kulkarni, Aubrey Agub, Michael Baxendale, Aaron David, Rachel Horton, Elliott Atkinson - [[pdf]](https://arxiv.org/pdf/2602.21904)</summary>

**Abstract:** Accurate cone localization in 3D space is essential in autonomous racing for precise navigation around the track. Approaches that rely on traditional computer vision algorithms are sensitive to environmental variations, and neural networks are often trained on limited data and are infeasible to run in real time. We present a UNet-based neural network for keypoint detection on cones, leveraging the largest custom-labeled dataset we have assembled. Our approach enables accurate cone position estimation and the potential for color prediction. Our model achieves substantial improvements in keypoint accuracy over conventional methods. Furthermore, we leverage our predicted keypoints in the perception pipeline and evaluate the end-to-end autonomous system. Our results show high-quality performance across all metrics, highlighting the effectiveness of this approach and its potential for adoption in competitive autonomous racing systems.

**arXiv ID:** 2602.21904
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (13 papers)</h2></summary>

<details>
<summary><strong>Language Models Exhibit Inconsistent Biases Towards Algorithmic Agents and Human Experts</strong> - Jessica Y. Bo, Lillio Mok, Ashton Anderson - [[pdf]](https://arxiv.org/pdf/2602.22070)</summary>

**Abstract:** Large language models are increasingly used in decision-making tasks that require them to process information from a variety of sources, including both human experts and other algorithmic agents. How do LLMs weigh the information provided by these different sources? We consider the well-studied phenomenon of algorithm aversion, in which human decision-makers exhibit bias against predictions from algorithms. Drawing upon experimental paradigms from behavioural economics, we evaluate how eightdifferent LLMs delegate decision-making tasks when the delegatee is framed as a human expert or an algorithmic agent. To be inclusive of different evaluation formats, we conduct our study with two task presentations: stated preferences, modeled through direct queries about trust towards either agent, and revealed preferences, modeled through providing in-context examples of the performance of both agents. When prompted to rate the trustworthiness of human experts and algorithms across diverse tasks, LLMs give higher ratings to the human expert, which correlates with prior results from human respondents. However, when shown the performance of a human expert and an algorithm and asked to place an incentivized bet between the two, LLMs disproportionately choose the algorithm, even when it performs demonstrably worse. These discrepant results suggest that LLMs may encode inconsistent biases towards humans and algorithms, which need to be carefully considered when they are deployed in high-stakes scenarios. Furthermore, we discuss the sensitivity of LLMs to task presentation formats that should be broadly scrutinized in evaluation robustness for AI safety.

**arXiv ID:** 2602.22070
</details>

<details>
<summary><strong>AgenticTyper: Automated Typing of Legacy Software Projects Using Agentic AI</strong> - Clemens Pohle - [[pdf]](https://arxiv.org/pdf/2602.21251)</summary>

**Abstract:** Legacy JavaScript systems lack type safety, making maintenance risky. While TypeScript can help, manually adding types is expensive. Previous automated typing research focuses on type inference but rarely addresses type checking setup, definition generation, bug identification, or behavioral correctness at repository scale. We present AgenticTyper, a Large Language Model (LLM)-based agentic system that addresses these gaps through iterative error correction and behavior preservation via transpilation comparison. Evaluation on two proprietary repositories (81K LOC) shows that AgenticTyper resolves all 633 initial type errors in 20 minutes, reducing manual effort from one working day.

**arXiv ID:** 2602.21251
</details>

<details>
<summary><strong>Black-Box Reliability Certification for AI Agents via Self-Consistency Sampling and Conformal Calibration</strong> - Charafeddine Mouzouni - [[pdf]](https://arxiv.org/pdf/2602.21368)</summary>

**Abstract:** Given a black-box AI system and a task, at what confidence level can a practitioner trust the system's output? We answer with a reliability level -- a single number per system-task pair, derived from self-consistency sampling and conformal calibration, that serves as a black-box deployment gate with exact, finite-sample, distribution-free guarantees. Self-consistency sampling reduces uncertainty exponentially; conformal calibration guarantees correctness within 1/(n+1) of the target level, regardless of the system's errors -- made transparently visible through larger answer sets for harder questions. Weaker models earn lower reliability levels (not accuracy -- see Definition 2.4): GPT-4.1 earns 94.6% on GSM8K and 96.8% on TruthfulQA, while GPT-4.1-nano earns 89.8% on GSM8K and 66.5% on MMLU. We validate across five benchmarks, five models from three families, and both synthetic and real data. Conditional coverage on solvable items exceeds 0.93 across all configurations; sequential stopping reduces API costs by around 50%.

**arXiv ID:** 2602.21368
</details>

<details>
<summary><strong>GUI-Libra: Training Native GUI Agents to Reason and Act with Action-aware Supervision and Partially Verifiable RL</strong> - Rui Yang, Qianhui Wu, Zhaoyang Wang, Hanyang Chen, Ke Yang, Hao Cheng, Huaxiu Yao, Baoling Peng, Huan Zhang, Jianfeng Gao, Tong Zhang - [[pdf]](https://arxiv.org/pdf/2602.22190)</summary>

**Abstract:** Open-source native GUI agents still lag behind closed-source systems on long-horizon navigation tasks. This gap stems from two limitations: a shortage of high-quality, action-aligned reasoning data, and the direct adoption of generic post-training pipelines that overlook the unique challenges of GUI agents. We identify two fundamental issues in these pipelines: (i) standard SFT with CoT reasoning often hurts grounding, and (ii) step-wise RLVR-tyle training faces partial verifiability, where multiple actions can be correct but only a single demonstrated action is used for verification. This makes offline step-wise metrics weak predictors of online task success. In this work, we present GUI-Libra, a tailored training recipe that addresses these challenges. First, to mitigate the scarcity of action-aligned reasoning data, we introduce a data construction and filtering pipeline and release a curated 81K GUI reasoning dataset. Second, to reconcile reasoning with grounding, we propose action-aware SFT that mixes reasoning-then-action and direct-action data and reweights tokens to emphasize action and grounding. Third, to stabilize RL under partial verifiability, we identify the overlooked importance of KL regularization in RLVR and show that a KL trust region is critical for improving offline-to-online predictability; we further introduce success-adaptive scaling to downweight unreliable negative gradients. Across diverse web and mobile benchmarks, GUI-Libra consistently improves both step-wise accuracy and end-to-end task completion. Our results suggest that carefully designed post-training and data curation can unlock significantly stronger task-solving capabilities without costly online data collection. We release our dataset, code, and models to facilitate further research on data-efficient post-training for reasoning-capable GUI agents.

**arXiv ID:** 2602.22190
</details>

<details>
<summary><strong>InsightX Agent: An LMM-based Agentic Framework with Integrated Tools for Reliable X-ray NDT Analysis</strong> - Jiale Liu, Huan Wang, Yue Zhang, Xiaoyu Luo, Jiaxiang Hu, Zhiliang Liu, Min Xie - [[pdf]](https://arxiv.org/pdf/2507.14899)</summary>

**Abstract:** Non-destructive testing (NDT), particularly X-ray inspection, is vital for industrial quality assurance, yet existing deep-learning-based approaches often lack interactivity, interpretability, and the capacity for critical self-assessment, limiting their reliability and operator trust. To address these shortcomings, this paper proposes InsightX Agent, a novel LMM-based agentic framework designed to deliver reliable, interpretable, and interactive X-ray NDT analysis. Unlike typical sequential pipelines, InsightX Agent positions a Large Multimodal Model (LMM) as a central orchestrator, coordinating between the Sparse Deformable Multi-Scale Detector (SDMSD) and the Evidence-Grounded Reflection (EGR) tool. The SDMSD generates dense defect region proposals from multi-scale feature maps and sparsifies them through Non-Maximum Suppression (NMS), optimizing detection of small, dense targets in X-ray images while maintaining computational efficiency. The EGR tool guides the LMM agent through a chain-of-thought-inspired review process, incorporating context assessment, individual defect analysis, false positive elimination, confidence recalibration and quality assurance to validate and refine the SDMSD's initial proposals. By strategically employing and intelligently using tools, InsightX Agent moves beyond passive data processing to active reasoning, enhancing diagnostic reliability and providing interpretations that integrate diverse information sources. Experimental evaluations on the GDXray+ dataset demonstrate that InsightX Agent not only achieves a high object detection F1-score of 96.54\% but also offers significantly improved interpretability and trustworthiness in its analyses, highlighting the transformative potential of LMM-based agentic frameworks for industrial inspection tasks.

**arXiv ID:** 2507.14899
</details>

<details>
<summary><strong>EO-1: An Open Unified Embodied Foundation Model for General Robot Control</strong> - Delin Qu, Haoming Song, Qizhi Chen, Zhaoqing Chen, Xianqiang Gao, Dong Wang, Xinyi Ye, Qi Lv, Modi Shi, Guanghui Ren, Cheng Ruan, Maoqing Yao, Haoran Yang, Jiacheng Bao, Bin Zhao, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2508.21112)</summary>

**Abstract:** The human ability to seamlessly perform multimodal reasoning and physical interaction in the open world is a core goal for general purpose embodied intelligent systems. Recent vision-language-action (VLA) models, which are co-trained on large-scale robot and visual-text data, have demonstrated notable progress in general robot control. However, they still fail to achieve human-level flexibility in interleaved reasoning and interaction. In this work, we introduce EO-Robotics, consists of EO-1 model and EO-Data1.5M dataset. EO-1 is a unified embodied foundation model that achieves superior performance in multimodal embodied reasoning and robot control through interleaved vision-text-action pre-training. The development of EO-1 is based on two key pillars: (i) a unified architecture that processes multimodal inputs indiscriminately (image, text, video, and action), and (ii) a massive, high-quality multimodal embodied reasoning dataset, EO-Data1.5M, which contains over 1.5 million samples with emphasis on interleaved vision-text-action comprehension. EO-1 is trained through synergies between auto-regressive decoding and flow matching denoising on EO-Data1.5M, enabling seamless robot action generation and multimodal embodied reasoning. Extensive experiments demonstrate the effectiveness of interleaved vision-text-action learning for open-world understanding and generalization, validated through a variety of long-horizon, dexterous manipulation tasks across multiple embodiments. This paper details the architecture of EO-1, the data construction strategy of EO-Data1.5M, and the training methodology, offering valuable insights for developing advanced embodied foundation models. Project Page: this https URL.

**arXiv ID:** 2508.21112
</details>

<details>
<summary><strong>Bypassing AI Control Protocols via Agent-as-a-Proxy Attacks</strong> - Jafar Isbarov, Murat Kantarcioglu - [[pdf]](https://arxiv.org/pdf/2602.05066)</summary>

**Abstract:** As AI agents automate critical workloads, they remain vulnerable to indirect prompt injection (IPI) attacks. Current defenses rely on monitoring protocols that jointly evaluate an agent's Chain-of-Thought (CoT) and tool-use actions to ensure alignment with user intent. We demonstrate that these monitoring-based defenses can be bypassed via a novel Agent-as-a-Proxy attack, where prompt injection attacks treat the agent as a delivery mechanism, bypassing both agent and monitor simultaneously. While prior work on scalable oversight has focused on whether small monitors can supervise large agents, we show that even frontier-scale monitors are vulnerable. Large-scale monitoring models like Qwen2.5-72B can be bypassed by agents with similar capabilities, such as GPT-4o mini and Llama-3.1-70B. On the AgentDojo benchmark, we achieve a high attack success rate against AlignmentCheck and Extract-and-Evaluate monitors under diverse monitoring LLMs. Our findings suggest current monitoring-based agentic defenses are fundamentally fragile regardless of model scale.

**arXiv ID:** 2602.05066
</details>

<details>
<summary><strong>AgentLTV: An Agent-Based Unified Search-and-Evolution Framework for Automated Lifetime Value Prediction</strong> - Chaowei Wu, Huazhu Chen, Congde Yuan, Qirui Yang, Guoqing Song, Yue Gao, Li Luo, Frank Youhua Chen, Mengzhuo Guo - [[pdf]](https://arxiv.org/pdf/2602.21634)</summary>

**Abstract:** Lifetime Value (LTV) prediction is critical in advertising, recommender systems, and e-commerce. In practice, LTV data patterns vary across decision scenarios. As a result, practitioners often build complex, scenario-specific pipelines and iterate over feature processing, objective design, and tuning. This process is expensive and hard to transfer. We propose AgentLTV, an agent-based unified search-and-evolution framework for automated LTV modeling. AgentLTV treats each candidate solution as an {executable pipeline program}. LLM-driven agents generate code, run and repair pipelines, and analyze execution feedback. Two decision agents coordinate a two-stage search. The Monte Carlo Tree Search (MCTS) stage explores a broad space of modeling choices under a fixed budget, guided by the Polynomial Upper Confidence bounds for Trees criterion and a Pareto-aware multi-metric value function. The Evolutionary Algorithm (EA) stage refines the best MCTS program via island-based evolution with crossover, mutation, and migration. Experiments on a large-scale proprietary dataset and a public benchmark show that AgentLTV consistently discovers strong models across ranking and error metrics. Online bucket-level analysis further indicates improved ranking consistency and value calibration, especially for high-value and negative-LTV segments. We summarize practitioner-oriented takeaways: use MCTS for rapid adaptation to new data patterns, use EA for stable refinement, and validate deployment readiness with bucket-level ranking and calibration diagnostics. The proposed AgentLTV has been successfully deployed online.

**arXiv ID:** 2602.21634
</details>

<details>
<summary><strong>CodeCureAgent: Automatic Classification and Repair of Static Analysis Warnings</strong> - Pascal Joos, Islem Bouzenia, Michael Pradel - [[pdf]](https://arxiv.org/pdf/2509.11787)</summary>

**Abstract:** Static analysis tools are widely used to detect bugs, vulnerabilities, and code smells. Traditionally, developers must resolve these warnings manually. Because this process is tedious, developers sometimes ignore warnings, leading to an accumulation of warnings and a degradation of code quality. This paper presents CodeCureAgent, an approach that harnesses LLM-based agents to automatically analyze, classify, and repair static analysis warnings. Unlike previous work, our method does not follow a predetermined algorithm. Instead, we adopt an agentic framework that iteratively invokes tools to gather additional information from the codebase (e.g., via code search) and edit the codebase to resolve the warning. CodeCureAgent detects and suppresses false positives, while fixing true positives when identified. We equip CodeCureAgent with a three-step heuristic to approve patches: (1) build the project, (2) verify that the warning disappears without introducing new warnings, and (3) run the test suite. We evaluate CodeCureAgent on a dataset of 1,000 SonarQube warnings found in 106 Java projects and covering 291 distinct rules. Our approach produces plausible fixes for 96.8% of the warnings, outperforming state-of-the-art baseline approaches by 29.2%-34.0% in plausible-fix rate. Manual inspection of 291 cases reveals a correct-fix rate of 86.3%, showing that CodeCureAgent can reliably repair static analysis warnings. The approach incurs LLM costs of about 2.9 cents (USD) and an end-to-end processing time of about four minutes per warning. We envision CodeCureAgent helping to clean existing codebases and being integrated into CI/CD pipelines to prevent the accumulation of static analysis warnings.

**arXiv ID:** 2509.11787
</details>

<details>
<summary><strong>TRACE: Trajectory-Aware Comprehensive Evaluation for Deep Research Agents</strong> - Yanyu Chen, Jiyue Jiang, Jiahong Liu, Yifei Zhang, Xiao Guo, Irwin King - [[pdf]](https://arxiv.org/pdf/2602.21230)</summary>

**Abstract:** The evaluation of Deep Research Agents is a critical challenge, as conventional outcome-based metrics fail to capture the nuances of their complex reasoning. Current evaluation faces two primary challenges: 1) a reliance on singular metrics like Pass@1, creating a "high-score illusion" that ignores the quality, efficiency, and soundness of the reasoning process; and 2) the failure of static benchmarks to quantify crucial attributes like robustness and latent capability. To address these gaps, we introduce TRACE (Trajectory-Aware Comprehensive Evaluation), a framework that holistically assesses the entire problem-solving trajectory. To counter the "high-score illusion", we propose a Hierarchical Trajectory Utility Function that quantifies process efficiency and cognitive quality, including evidence grounding, alongside accuracy. To measure deeper attributes, TRACE introduces a Scaffolded Capability Assessment protocol, quantifying an agent's latent ability by determining the minimum guidance needed for success. Our contributions include the TRACE framework, its novel metrics, and the accompanying DeepResearch-Bench with controllable complexity. Experiments show TRACE delivers a granular ranking that uncovers critical trade-offs between agent accuracy, efficiency, and robustness entirely missed by singular metrics.

**arXiv ID:** 2602.21230
</details>

<details>
<summary><strong>MERRY: Semantically Decoupled Evaluation of Multimodal Emotional and Role Consistencies of Role-Playing Agents</strong> - Zhenyu Wang, Xiaofen Xing, Yirong Chen, Xiangmin Xu - [[pdf]](https://arxiv.org/pdf/2602.21941)</summary>

**Abstract:** Multimodal Role-Playing Agents (MRPAs) are attracting increasing attention due to their ability to deliver more immersive multimodal emotional interactions. However, existing studies still rely on pure textual benchmarks to evaluate the text responses of MRPAs, while delegating the assessment of their multimodal expressions solely to modality-synthesis metrics. This evaluation paradigm, on the one hand, entangles semantic assessment with modality generation, leading to ambiguous error attribution, and on the other hand remains constrained by the heavy reliance on human judgment. To this end, we propose MERRY, a semantically decoupled evaluation framework for assessing Multimodal Emotional and Role consistencies of Role-playing agents. This framework introduce five refined metrics for EC and three for RC. Notably, we transform the traditional subjective scoring approach into a novel bidirectional-evidence-finding task, significantly improving the human agreement of LLM-as-Judge evaluations. Based on MERRY, we conduct extensive evaluations. Our empirical results primarily reveal that: (1) Training on synthetic datasets tends to reduce emotional consistency, whereas training on real-world datasets improves it; (2) Existing models suffer from emotional templatization and simplification, exhibiting positive-bias and performance bottleneck in fine-grained negative emotions; (3) Simple prompting method strengthens the weak models but constrains the strong ones, while simple fine-tuning method suffers from poor role generalization. Codes and dataset are available.

**arXiv ID:** 2602.21941
</details>

<details>
<summary><strong>Embodied Task Planning via Graph-Informed Action Generation with Large Language Model</strong> - Xiang Li, Ning Yan, Masood Mortazavi - [[pdf]](https://arxiv.org/pdf/2601.21841)</summary>

**Abstract:** While Large Language Models (LLMs) have demonstrated strong zero-shot reasoning capabilities, their deployment as embodied agents still faces fundamental challenges in long-horizon planning. Unlike open-ended text generation, embodied agents must decompose high-level intent into actionable sub-goals while strictly adhering to the logic of a dynamic, observed environment. Standard LLM planners frequently fail to maintain strategy coherence over extended horizons due to context window limitation or hallucinate transitions that violate constraints. We propose GiG, a novel planning framework that structures embodied agents' memory using a Graph-in-Graph architecture. Our approach employs a Graph Neural Network (GNN) to encode environmental states into embeddings, organizing these embeddings into action-connected execution trace graphs within an experience memory bank. By clustering these graph embeddings, the framework enables retrieval of structure-aware priors, allowing agents to ground current decisions in relevant past structural patterns. Furthermore, we introduce a novel bounded lookahead module that leverages symbolic transition logic to enhance the agents' planning capabilities through the grounded action projection. We evaluate our framework on three embodied planning benchmarks-Robotouille Synchronous, Robotouille Asynchronous, and ALFWorld. Our method outperforms state-of-the-art baselines, achieving Pass@1 performance gains of up to 22% on Robotouille Synchronous, 37% on Asynchronous, and 15% on ALFWorld with comparable or lower computational cost.

**arXiv ID:** 2601.21841
</details>

<details>
<summary><strong>A study on the effects of mixed explicit and implicit communications in human-artificial-agent interactions</strong> - Ana Christina Almada Campos, Bruno Vilhena Adorno - [[pdf]](https://arxiv.org/pdf/2409.18745)</summary>

**Abstract:** Communication between humans and artificial agents is essential for their interaction. This is often inspired by human communication, which uses gestures, facial expressions, gaze direction, and other explicit and implicit means. This work presents interaction experiments where humans and artificial agents interact through explicit and implicit communication to evaluate the effect of mixed explicit-implicit communication against purely explicit communication and the impact of the task difficulty in this evaluation. Results obtained using Bayesian parameter estimation show that the task execution time did not significantly change when mixed explicit and implicit communications were used in neither of our experiments, which varied in the type of artificial agent (virtual agent and humanoid robot) used and task difficulty. The number of errors was affected by the communication only when the human was executing a more difficult task, and an impact on the perceived efficiency of the interaction was only observed in the interaction with the robot, for both easy and difficult tasks. In contrast, acceptance, sociability, and transparency of the artificial agent increased when using mixed communication modalities in both our experiments and task difficulty levels. This suggests that task-related measures, such as time, number of errors, and perceived efficiency of the interaction, as well as the impact of the communication on them, are more sensitive to the type of task and the difficulty level, whereas the combination of explicit and implicit communications more consistently improves human perceptions about artificial agents.

**arXiv ID:** 2409.18745
</details>

</details>

<details open>
<summary><h2>LLM Agents (7 papers)</h2></summary>

<details>
<summary><strong>A General Equilibrium Theory of Orchestrated AI Agent Systems</strong> - Jean-Philippe Garnier - [[pdf]](https://arxiv.org/pdf/2602.21255)</summary>

**Abstract:** We establish a general equilibrium theory for systems of large language model (LLM) agents operating under centralized orchestration. The framework is a production economy in the sense of Arrow-Debreu (1954), extended to infinite-dimensional commodity spaces following Bewley (1972). Each LLM agent is modeled as a firm whose production set Y a $\subset$ H = L 2 ([0, T ], R R ) represents the feasible metric trajectories determined by its frozen model weights. The orchestrator is the consumer, choosing a routing policy over the agent DAG to maximize system welfare subject to a budget constraint evaluated at functional prices p $\in$ H A . These prices-elements of the Hilbert dual of the commodity space-assign a shadow value to each metric of each agent at each instant. We prove, via Brouwer's theorem applied to a finitedimensional approximation V K $\subset$ H, that every such economy admits at least one general equilibrium (p * , y * , $\pi$ * ). A functional Walras' law  holds as a theorem: the value of functional excess demand is zero for all prices, as a consequence of the consumer's budget constraint-not by construction. We further establish Pareto optimality (First Welfare Theorem), decentralizability of Pareto optima (Second Welfare Theorem), and uniqueness with geometric convergence under a contraction condition (Banach). The orchestration dynamics constitute a Walrasian t{â}tonnement that converges globally under the contraction condition, unlike classical t{â}tonnement (Scarf, 1960). The framework admits a DSGE interpretation with SLO parameters as policy rates.

**arXiv ID:** 2602.21255
</details>

<details>
<summary><strong>Think like a Scientist: Physics-guided LLM Agent for Equation Discovery</strong> - Jianke Yang, Ohm Venkatachalam, Mohammad Kianezhad, Sharvaree Vadgama, Rose Yu - [[pdf]](https://arxiv.org/pdf/2602.12259)</summary>

**Abstract:** Explaining observed phenomena through symbolic, interpretable formulas is a fundamental goal of science. Recently, large language models (LLMs) have emerged as promising tools for symbolic equation discovery, owing to their broad domain knowledge and strong reasoning capabilities. However, most existing LLM-based systems try to guess equations directly from data, without modeling the multi-step reasoning process that scientists often follow: first inferring physical properties such as symmetries, then using these as priors to restrict the space of candidate equations. We introduce KeplerAgent, an agentic framework that explicitly follows this scientific reasoning process. The agent coordinates physics-based tools to extract intermediate structure and uses these results to configure symbolic regression engines such as PySINDy and PySR, including their function libraries and structural constraints. Across a suite of physical equation benchmarks, KeplerAgent achieves substantially higher symbolic accuracy and greater robustness to noisy data than both LLM and traditional baselines.

**arXiv ID:** 2602.12259
</details>

<details>
<summary><strong>OptiRepair: Closed-Loop Diagnosis and Repair of Supply Chain Optimization Models with LLM Agents</strong> - Ruicheng Ao, David Simchi-Levi, Xinshang Wang - [[pdf]](https://arxiv.org/pdf/2602.19439)</summary>

**Abstract:** Supply chain optimization models frequently become infeasible because of modeling errors. Diagnosis and repair require scarce OR expertise: analysts must interpret solver diagnostics, trace root causes across echelons, and fix formulations without sacrificing operational soundness. Whether AI agents can perform this task remains untested. We decompose this task into two phases: a domain-agnostic feasibility phase that iteratively repairs any LP using IIS-guided diagnosis, and a domain-specific validation phase that enforces five rationality checks grounded in inventory theory. We test 22 API models from seven families on 976 multi-echelon supply chain problems and train two 8B-parameter models with self-taught reasoning and solver-verified rewards. The trained models reach 81.7% Rational Recovery Rate (RRR) -- the fraction of problems resolved to both feasibility and operational rationality -- versus 42.2% for the best API model and 21.3% on average. The gap concentrates in Phase 1 repair, where API models average 27.6% recovery rate versus 97.2% for trained models. Two gaps separate current AI from reliable model repair: solver interaction, as API models restore only 27.6% of infeasible formulations; and operational rationale, as roughly one in four feasible repairs violate supply chain theory. Each gap requires a different intervention -- targeted training closes the solver interaction gap, while explicit specification as solver-verifiable checks closes the rationality gap. For organizations adopting AI in operational planning, formalizing what 'rational' means in their context is the higher-return investment.

**arXiv ID:** 2602.19439
</details>

<details>
<summary><strong>Stabilizing Off-Policy Training for Long-Horizon LLM Agent via Turn-Level Importance Sampling and Clipping-Triggered Normalization</strong> - Chenliang Li, Adel Elmahdy, Alex Boyd, Zhongruo Wang, Siliang Zeng, Alfredo Garcia, Parminder Bhatia, Taha Kass-Hout, Cao Xiao, Mingyi Hong - [[pdf]](https://arxiv.org/pdf/2511.20718)</summary>

**Abstract:** Reinforcement learning (RL) algorithms such as PPO and GRPO are widely used to train large language models (LLMs) for multi-turn agentic tasks. However, in off-policy training pipelines, these methods often exhibit unstable optimization dynamics and are prone to performance collapse. Through empirical analysis, we identify two fundamental sources of instability in this setting: (1)~a granularity mismatch between token-level policy optimization and turn-structured interactions, and (2) high-variance and unreliable gradient updates induced by off-policy importance sampling and inaccurate advantage estimation. To address these challenges, we propose SORL, \underline{S}tabilizing \underline{O}ff-Policy \underline{R}einforcement \underline{L}earning for Long-Horizon Agent Training. SORL introduces principled mechanisms that align policy optimization with the structure of multi-turn interactions and adaptively suppress unreliable off-policy updates, yielding more conservative and robust learning dynamics. Within this framework, we instantiate two stabilized algorithms: SO-PPO and SO-GRPO. Both algorithms are designed to mitigate gradient variance and prevent optimization collapse without requiring careful early stopping or heuristic tuning. We evaluate SO-PPO and SO-GRPO on a range of multi-turn search benchmarks, including general question answering, multi-hop question answering, and medical multiple-choice QA tasks. Experimental results show that both methods consistently prevent training instabilities and performance collapses observed in standard PPO and GRPO, maintain lower clipping ratios and more stable optimization trajectories, and achieve superior or comparable task performance. These results demonstrate that the proposed algorithm provides a practical, scalable, and general framework for stabilizing reinforcement learning in multi-turn LLM agent training.

**arXiv ID:** 2511.20718
</details>

<details>
<summary><strong>Both Ends Count! Just How Good are LLM Agents at "Text-to-Big SQL"?</strong> - Germán T. Eizaguirre, Lars Tissen, Marc Sánchez-Artigas - [[pdf]](https://arxiv.org/pdf/2602.21480)</summary>

**Abstract:** Text-to-SQL and Big Data are both extensively benchmarked fields, yet there is limited research that evaluates them jointly. In the real world, Text-to-SQL systems are often embedded with Big Data workflows, such as large-scale data processing or interactive data analytics. We refer to this as "Text-to-Big SQL". However, existing text-to-SQL benchmarks remain narrowly scoped and overlook the cost and performance implications that arise at scale. For instance, translation errors that are minor on small datasets lead to substantial cost and latency overheads as data scales, a relevant issue completely ignored by text-to-SQL metrics.
In this paper, we overcome this overlooked challenge by introducing novel and representative metrics for evaluating Text-to-Big SQL. Our study focuses on production-level LLM agents, a database-agnostic system adaptable to diverse user needs. Via an extensive evaluation of frontier models, we show that text-to-SQL metrics are insufficient for Big Data. In contrast, our proposed text-to-Big SQL metrics accurately reflect execution efficiency, cost, and the impact of data scale. Furthermore, we provide LLM-specific insights, including fine-grained, cross-model comparisons of latency and cost.

**arXiv ID:** 2602.21480
</details>

<details>
<summary><strong>SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards</strong> - Dengjia Zhang, Xiaoou Liu, Lu Cheng, Yaqing Wang, Kenton Murray, Hua Wei - [[pdf]](https://arxiv.org/pdf/2602.21158)</summary>

**Abstract:** Large language models (LLMs) are increasingly deployed as multi-step decision-making agents, where effective reward design is essential for guiding learning. Although recent work explores various forms of reward shaping and step-level credit assignment, a key signal remains largely overlooked: the intrinsic uncertainty of LLMs. Uncertainty reflects model confidence, reveals where exploration is needed, and offers valuable learning cues even in failed trajectories. We introduce SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards, a reinforcement learning framework that incorporates uncertainty directly into the reward design. SELAUR integrates entropy-, least-confidence-, and margin-based metrics into a combined token-level uncertainty estimate, providing dense confidence-aligned supervision, and employs a failure-aware reward reshaping mechanism that injects these uncertainty signals into step- and trajectory-level rewards to improve exploration efficiency and learning stability. Experiments on two benchmarks, ALFWorld and WebShop, show that our method consistently improves success rates over strong baselines. Ablation studies further demonstrate how uncertainty signals enhance exploration and robustness.

**arXiv ID:** 2602.21158
</details>

<details>
<summary><strong>Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data</strong> - Emre Can Acikgoz, Cheng Qian, Jonas Hübotter, Heng Ji, Dilek Hakkani-Tür, Gokhan Tur - [[pdf]](https://arxiv.org/pdf/2602.21320)</summary>

**Abstract:** Large language models (LLMs) are becoming the foundation for autonomous agents that can use tools to solve complex tasks. Reinforcement learning (RL) has emerged as a common approach for injecting such agentic capabilities, but typically under tightly controlled training setups. It often depends on carefully constructed task-solution pairs and substantial human supervision, which creates a fundamental obstacle to open-ended self-evolution toward superintelligent systems. In this paper, we propose Tool-R0 framework for training general purpose tool-calling agents from scratch with self-play RL, under a zero-data assumption. Initialized from the same base LLM, Tool-R0 co-evolves a Generator and a Solver with complementary rewards: one proposes targeted challenging tasks at the other's competence frontier and the other learns to solve them with real-world tool calls. This creates a self-evolving cycle that requires no pre-existing tasks or datasets. Evaluation on different tool-use benchmarks show that Tool-R0 yields 92.5 relative improvement over the base model and surpasses fully supervised tool-calling baselines under the same setting. Our work further provides empirical insights into self-play LLM agents by analyzing co-evolution, curriculum dynamics, and scaling behavior.

**arXiv ID:** 2602.21320
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (17 papers)</h2></summary>

<details>
<summary><strong>A Hierarchical Multi-Agent System for Autonomous Discovery in Geoscientific Data Archives</strong> - Dmitrii Pantiukhin, Ivan Kuznetsov, Boris Shapkin, Antonia Anna Jost, Thomas Jung, Nikolay Koldunov - [[pdf]](https://arxiv.org/pdf/2602.21351)</summary>

**Abstract:** The rapid accumulation of Earth science data has created a significant scalability challenge; while repositories like PANGAEA host vast collections of datasets, citation metrics indicate that a substantial portion remains underutilized, limiting data reusability. Here we present PANGAEA-GPT, a hierarchical multi-agent framework designed for autonomous data discovery and analysis. Unlike standard Large Language Model (LLM) wrappers, our architecture implements a centralized Supervisor-Worker topology with strict data-type-aware routing, sandboxed deterministic code execution, and self-correction via execution feedback, enabling agents to diagnose and resolve runtime errors. Through use-case scenarios spanning physical oceanography and ecology, we demonstrate the system's capacity to execute complex, multi-step workflows with minimal human intervention. This framework provides a methodology for querying and analyzing heterogeneous repository data through coordinated agent workflows.

**arXiv ID:** 2602.21351
</details>

<details>
<summary><strong>Field-Theoretic Memory for AI Agents: Continuous Dynamics for Context Preservation</strong> - Subhadip Mitra - [[pdf]](https://arxiv.org/pdf/2602.21220)</summary>

**Abstract:** We present a memory system for AI agents that treats stored information as continuous fields governed by partial differential equations rather than discrete entries in a database. The approach draws from classical field theory: memories diffuse through semantic space, decay thermodynamically based on importance, and interact through field coupling in multi-agent scenarios. We evaluate the system on two established long-context benchmarks: LoCoMo (ACL 2024) with 300-turn conversations across 35 sessions, and LongMemEval (ICLR 2025) testing multi-session reasoning over 500+ turns. On LongMemEval, the field-theoretic approach achieves significant improvements: +116% F1 on multi-session reasoning (p<0.01, d= 3.06), +43.8% on temporal reasoning (p<0.001, d= 9.21), and +27.8% retrieval recall on knowledge updates (p<0.001, d= 5.00). Multi-agent experiments show near-perfect collective intelligence (>99.8%) through field coupling. Code is available at this http URL.

**arXiv ID:** 2602.21220
</details>

<details>
<summary><strong>Training Generalizable Collaborative Agents via Strategic Risk Aversion</strong> - Chengrui Qu, Yizhou Zhang, Nicholas Lanzetti, Eric Mazumdar - [[pdf]](https://arxiv.org/pdf/2602.21515)</summary>

**Abstract:** Many emerging agentic paradigms require agents to collaborate with one another (or people) to achieve shared goals. Unfortunately, existing approaches to learning policies for such collaborative problems produce brittle solutions that fail when paired with new partners. We attribute these failures to a combination of free-riding during training and a lack of strategic robustness. To address these problems, we study the concept of strategic risk aversion and interpret it as a principled inductive bias for generalizable cooperation with unseen partners. While strategically risk-averse players are robust to deviations in their partner's behavior by design, we show that, in collaborative games, they also (1) can have better equilibrium outcomes than those at classical game-theoretic concepts like Nash, and (2) exhibit less or no free-riding. Inspired by these insights, we develop a multi-agent reinforcement learning (MARL) algorithm that integrates strategic risk aversion into standard policy optimization methods. Our empirical results across collaborative benchmarks (including an LLM collaboration task) validate our theory and demonstrate that our approach consistently achieves reliable collaboration with heterogeneous and previously unseen partners across collaborative tasks.

**arXiv ID:** 2602.21515
</details>

<details>
<summary><strong>Hierarchical LLM-Based Multi-Agent Framework with Prompt Optimization for Multi-Robot Task Planning</strong> - Tomoya Kawabe, Rin Takano - [[pdf]](https://arxiv.org/pdf/2602.21670)</summary>

**Abstract:** Multi-robot task planning requires decomposing natural-language instructions into executable actions for heterogeneous robot teams. Conventional Planning Domain Definition Language (PDDL) planners provide rigorous guarantees but struggle to handle ambiguous or long-horizon missions, while large language models (LLMs) can interpret instructions and propose plans but may hallucinate or produce infeasible actions. We present a hierarchical multi-agent LLM-based planner with prompt optimization: an upper layer decomposes tasks and assigns them to lower-layer agents, which generate PDDL problems solved by a classical planner. When plans fail, the system applies TextGrad-inspired textual-gradient updates to optimize each agent's prompt and thereby improve planning accuracy. In addition, meta-prompts are learned and shared across agents within the same layer, enabling efficient prompt optimization in multi-agent settings. On the MAT-THOR benchmark, our planner achieves success rates of 0.95 on compound tasks, 0.84 on complex tasks, and 0.60 on vague tasks, improving over the previous state-of-the-art LaMMA-P by 2, 7, and 15 percentage points respectively. An ablation study shows that the hierarchical structure, prompt optimization, and meta-prompt sharing contribute roughly +59, +37, and +4 percentage points to the overall success rate.

**arXiv ID:** 2602.21670
</details>

<details>
<summary><strong>1-2-3 Check: Enhancing Contextual Privacy in LLM via Multi-Agent Reasoning</strong> - Wenkai Li, Liwen Sun, Zhenxiang Guan, Xuhui Zhou, Maarten Sap - [[pdf]](https://arxiv.org/pdf/2508.07667)</summary>

**Abstract:** Addressing contextual privacy concerns remains challenging in interactive settings where large language models (LLMs) process information from multiple sources (e.g., summarizing meetings with private and public information). We introduce a multi-agent framework that decomposes privacy reasoning into specialized subtasks (extraction, classification), reducing the information load on any single agent while enabling iterative validation and more reliable adherence to contextual privacy norms. To understand how privacy errors emerge and propagate, we conduct a systematic ablation over information-flow topologies, revealing when and why upstream detection mistakes cascade into downstream leakage. Experiments on the ConfAIde and PrivacyLens benchmark with several open-source and closed-sourced LLMs demonstrate that our best multi-agent configuration substantially reduces private information leakage (\textbf{18\%} on ConfAIde and \textbf{19\%} on PrivacyLens with GPT-4o) while preserving the fidelity of public content, outperforming single-agent baselines. These results highlight the promise of principled information-flow design in multi-agent systems for contextual privacy with LLMs.

**arXiv ID:** 2508.07667
</details>

<details>
<summary><strong>OMNI-LEAK: Orchestrator Multi-Agent Network Induced Data Leakage</strong> - Akshat Naik, Jay Culligan, Yarin Gal, Philip Torr, Rahaf Aljundi, Alasdair Paren, Adel Bibi - [[pdf]](https://arxiv.org/pdf/2602.13477)</summary>

**Abstract:** As Large Language Model (LLM) agents become more capable, their coordinated use in the form of multi-agent systems is anticipated to emerge as a practical paradigm. Prior work has examined the safety and misuse risks associated with agents. However, much of this has focused on the single-agent case and/or setups missing basic engineering safeguards such as access control, revealing a scarcity of threat modeling in multi-agent systems. We investigate the security vulnerabilities of a popular multi-agent pattern known as the orchestrator setup, in which a central agent decomposes and delegates tasks to specialized agents. Through red-teaming a concrete setup representative of a likely future use case, we demonstrate a novel attack vector, OMNI-LEAK, that compromises several agents to leak sensitive data through a single indirect prompt injection, even in the presence of data access control. We report the susceptibility of frontier models to different categories of attacks, finding that both reasoning and non-reasoning models are vulnerable, even when the attacker lacks insider knowledge of the implementation details. Our work highlights the importance of safety research to generalize from single-agent to multi-agent settings, in order to reduce the serious risks of real-world privacy breaches and financial losses and overall public trust in AI agents.

**arXiv ID:** 2602.13477
</details>

<details>
<summary><strong>OR-Agent: Bridging Evolutionary Search and Structured Research for Automated Algorithm Discovery</strong> - Qi Liu, Ruochen Hao, Can Li, Wanjing Ma - [[pdf]](https://arxiv.org/pdf/2602.13769)</summary>

**Abstract:** Automating scientific discovery in complex, experiment-driven domains requires more than iterative mutation of programs; it demands structured hypothesis management, environment interaction, and principled reflection. We present OR-Agent, a configurable multi-agent research framework designed for automated exploration in rich experimental environments. OR-Agent organizes research as a structured tree-based workflow that explicitly models branching hypothesis generation and systematic backtracking, enabling controlled management of research trajectories beyond simple mutation-crossover loops. At its core, we introduce an evolutionary-systematic ideation mechanism that unifies evolutionary selection of research starting points, comprehensive research plan generation, and coordinated exploration within a research tree. We introduce a hierarchical optimization-inspired reflection system in which short-term reflections act as verbal gradients, long-term reflections as verbal momentum, and memory compression as semantic weight decay, collectively forming a principled mechanism for governing research dynamics. We conduct extensive experiments across classical combinatorial optimization benchmarks as well as simulation-based cooperative driving scenarios. Results demonstrate that OR-Agent outperforms strong evolutionary baselines while providing a general, extensible, and inspectable framework for AI-assisted scientific discovery. All code and experimental data are publicly available at this https URL.

**arXiv ID:** 2602.13769
</details>

<details>
<summary><strong>Multi-agent deep reinforcement learning with centralized training and decentralized execution for transportation infrastructure management</strong> - M. Saifullah, K.G. Papakonstantinou, A. Bhattacharya, S.M. Stoffels, C.P. Andriotis - [[pdf]](https://arxiv.org/pdf/2401.12455)</summary>

**Abstract:** Life-cycle management of large-scale transportation systems requires determining a sequence of inspection and maintenance decisions to minimize long-term risks and costs while dealing with multiple uncertainties and constraints that lie in high-dimensional spaces. Traditional approaches have been widely applied but often suffer from limitations related to optimality, scalability, and the ability to properly handle uncertainty. Moreover, many existing methods rely on unconstrained formulations that overlook critical operational constraints. We address these issues in this work by casting the optimization problem within the framework of constrained Partially Observable Markov Decision Processes (POMDPs), which provide a robust mathematical foundation for stochastic sequential decision-making under observation uncertainties, in the presence of risk and resource limitations. To tackle the high dimensionality of state and action spaces, we propose DDMAC-CTDE, a Deep Decentralized Multi-Agent Actor-Critic (DDMAC) reinforcement learning architecture with Centralized Training and Decentralized Execution (CTDE). To demonstrate the utility of the proposed framework, we also develop a new comprehensive benchmark environment representing an existing transportation network in Virginia, U.S., with heterogeneous pavement and bridge assets undergoing nonstationary degradation. This environment incorporates multiple practical constraints related to budget limits, performance guidelines, traffic delays, and risk considerations. On this benchmark, DDMAC-CTDE consistently outperforms standard transportation management baselines, producing better policies. Together, the proposed framework and benchmark provide (i) a scalable, constraint-aware methodology, and (ii) a realistic, rigorous testbed for comprehensive evaluation of Deep Reinforcement Learning (DRL) for transportation infrastructure management.

**arXiv ID:** 2401.12455
</details>

<details>
<summary><strong>EpidemIQs: Prompt-to-Paper LLM Agents for Epidemic Modeling and Analysis</strong> - Mohammad Hossein Samaei, Faryad Darabi Sahneh, Lee W. Cohnstaedt, Caterina Scoglio - [[pdf]](https://arxiv.org/pdf/2510.00024)</summary>

**Abstract:** Large Language Models (LLMs) offer new opportunities to accelerate complex interdisciplinary research domains. Epidemic modeling, characterized by its complexity and reliance on network science, dynamical systems, epidemiology, and stochastic simulations, represents a prime candidate for leveraging LLM-driven automation. We introduce EpidemIQs, a novel multi-agent LLM framework that integrates user inputs and autonomously conducts literature review, analytical derivation, network modeling, mechanistic modeling, stochastic simulations, data visualization and analysis, and finally documentation of findings in a structured manuscript, through five predefined research phases. We introduce two types of agents: a scientist agent for planning, coordination, reflection, and generation of final results, and a task-expert agent to focus exclusively on one specific duty serving as a tool to the scientist agent. The framework consistently generated complete reports in scientific article format. Specifically, using GPT 4.1 and GPT 4.1 Mini as backbone LLMs for scientist and task-expert agents, respectively, the autonomous process completes with average total token usage 870K at a cost of about $1.57 per study, successfully executing all phases and final report. We evaluate EpidemIQs across several different epidemic scenarios, measuring computational cost, workflow reliability, task success rate, and LLM-as-Judge and human expert reviews to estimate the overall quality and technical correctness of the generated results. Through our experiments, the framework consistently addresses evaluation scenarios with an average task success rate of 79%. We compare EpidemIQs to an iterative single-agent LLM, benefiting from the same system prompts and tools, iteratively planning, invoking tools, and revising outputs until task completion. The comparisons suggest a consistently higher performance of EpidemIQs.

**arXiv ID:** 2510.00024
</details>

<details>
<summary><strong>MALLVI: A Multi-Agent Framework for Integrated Generalized Robotics Manipulation</strong> - Iman Ahmadi, Mehrshad Taji, Arad Mahdinezhad Kashani, AmirHossein Jadidi, Saina Kashani, Babak Khalaj - [[pdf]](https://arxiv.org/pdf/2602.16898)</summary>

**Abstract:** Task planning for robotic manipulation with large language models (LLMs) is an emerging area. Prior approaches rely on specialized models, fine tuning, or prompt tuning, and often operate in an open loop manner without robust environmental feedback, making them fragile in dynamic settings. MALLVI presents a Multi Agent Large Language and Vision framework that enables closed-loop feedback driven robotic manipulation. Given a natural language instruction and an image of the environment, MALLVI generates executable atomic actions for a robot manipulator. After action execution, a Vision Language Model (VLM) evaluates environmental feedback and decides whether to repeat the process or proceed to the next step. Rather than using a single model, MALLVI coordinates specialized agents, Decomposer, Localizer, Thinker, and Reflector, to manage perception, localization, reasoning, and high level planning. An optional Descriptor agent provides visual memory of the initial state. The Reflector supports targeted error detection and recovery by reactivating only relevant agents, avoiding full replanning. Experiments in simulation and real-world settings show that iterative closed loop multi agent coordination improves generalization and increases success rates in zero shot manipulation tasks. Code available at this https URL .

**arXiv ID:** 2602.16898
</details>

<details>
<summary><strong>From Cooperation to Hierarchy: A Study of Dynamics of Hierarchy Emergence in a Multi-Agent System</strong> - Shanshan Mao, Peter Tino - [[pdf]](https://arxiv.org/pdf/2602.21404)</summary>

**Abstract:** A central premise in evolutionary biology is that individual variation can generate information asymmetries that facilitate the emergence of hierarchical organisation. To examine this process, we develop an agent-based model (ABM) to identify the minimal conditions under which hierarchy arises in dynamic multi-agent systems, focusing on the roles of initial heterogeneity and mutation amplitude across generations. Hierarchical organisation is quantified using the Trophic Incoherence (TI) metric, which captures directional asymmetries in interaction networks. Our results show that even small individual differences can be amplified through repeated local interactions involving reproduction, competition, and cooperation, but that hierarchical order is markedly more sensitive to mutation amplitude than to initial heterogeneity. Across repeated trials, stable hierarchies reliably emerge only when mutation amplitude is sufficiently high, while initial heterogeneity primarily affects early formation rather than long-term persistence. Overall, these findings demonstrate how simple interaction rules can give rise to both the emergence and persistence of hierarchical organisation, providing a quantitative account of how structured inequality can develop from initially homogeneous populations.

**arXiv ID:** 2602.21404
</details>

<details>
<summary><strong>Pancake: Hierarchical Memory System for Multi-Agent LLM Serving</strong> - Zhengding Hu, Zaifeng Pan, Prabhleen Kaur, Vibha Murthy, Zhongkai Yu, Yue Guan, Zhen Wang, Steven Swanson, Yufei Ding - [[pdf]](https://arxiv.org/pdf/2602.21477)</summary>

**Abstract:** In this work, we identify and address the core challenges of agentic memory management in LLM serving, where large-scale storage, frequent updates, and multiple coexisting agents jointly introduce complex and high-cost approximate nearest neighbor (ANN) searching problems. We present Pancake, a multi-tier agentic memory system that unifies three key techniques: (i) multi-level index caching for single agents, (ii) coordinated index management across multiple agents, and (iii) collaborative GPU-CPU acceleration. Pancake exposes easy-to-use interface that can be integrated into memory-based agents like Mem-GPT, and is compatible with agentic frameworks such as LangChain and LlamaIndex. Experiments on realistic agent workloads show that Pancake substantially outperforms existing frameworks, achieving more than 4.29x end-to-end throughput improvement.

**arXiv ID:** 2602.21477
</details>

<details>
<summary><strong>Hierarchical Lead Critic based Multi-Agent Reinforcement Learning</strong> - David Eckel, Henri Meeß - [[pdf]](https://arxiv.org/pdf/2602.21680)</summary>

**Abstract:** Cooperative Multi-Agent Reinforcement Learning (MARL) solves complex tasks that require coordination from multiple agents, but is often limited to either local (independent learning) or global (centralized learning) perspectives. In this paper, we introduce a novel sequential training scheme and MARL architecture, which learns from multiple perspectives on different hierarchy levels. We propose the Hierarchical Lead Critic (HLC) - inspired by natural emerging distributions in team structures, where following high-level objectives combines with low-level execution. HLC demonstrates that introducing multiple hierarchies, leveraging local and global perspectives, can lead to improved performance with high sample efficiency and robust policies. Experimental results conducted on cooperative, non-communicative, and partially observable MARL benchmarks demonstrate that HLC outperforms single hierarchy baselines and scales robustly with increasing amounts of agents and difficulty.

**arXiv ID:** 2602.21680
</details>

<details>
<summary><strong>Toward Safe and Human-Aligned Game Conversational Recommendation via Multi-Agent Decomposition</strong> - Zheng Hui, Xiaokai Wei, Yexi Jiang, Kevin Gao, Chen Wang, Frank Ong, Se-eun Yoon, Rachit Pareek, Michelle Gong - [[pdf]](https://arxiv.org/pdf/2504.20094)</summary>

**Abstract:** Conversational recommender systems (CRS) have advanced with large language models, showing strong results in domains like movies. These domains typically involve fixed content and passive consumption, where user preferences can be matched by genre or theme. In contrast, games present distinct challenges: fast-evolving catalogs, interaction-driven preferences (e.g., skill level, mechanics, hardware), and increased risk of unsafe responses in open-ended conversation. We propose MATCHA, a multi-agent framework for CRS that assigns specialized agents for intent parsing, tool-augmented retrieval, multi-LLM ranking with reflection, explanation, and risk control which enabling finer personalization, long-tail coverage, and stronger safety. Evaluated on real user request dataset, MATCHA outperforms six baselines across eight metrics, improving Hit@5 by 20%, reducing popularity bias by 24%, and achieving 97.9% adversarial defense. Human and virtual-judge evaluations confirm improved explanation quality and user alignment.

**arXiv ID:** 2504.20094
</details>

<details>
<summary><strong>Reasoning-Driven Design of Single Atom Catalysts via a Multi-Agent Large Language Model Framework</strong> - Dong Hyeon Mok, Seoin Back, Victor Fung, Guoxiang Hu - [[pdf]](https://arxiv.org/pdf/2602.21533)</summary>

**Abstract:** Large language models (LLMs) are becoming increasingly applied beyond natural language processing, demonstrating strong capabilities in complex scientific tasks that traditionally require human expertise. This progress has extended into materials discovery, where LLMs introduce a new paradigm by leveraging reasoning and in-context learning, capabilities absent from conventional machine learning approaches. Here, we present a Multi-Agent-based Electrocatalyst Search Through Reasoning and Optimization (MAESTRO) framework in which multiple LLMs with specialized roles collaboratively discover high-performance single atom catalysts for the oxygen reduction reaction. Within an autonomous design loop, agents iteratively reason, propose modifications, reflect on results and accumulate design history. Through in-context learning enabled by this iterative process, MAESTRO identified design principles not explicitly encoded in the LLMs' background knowledge and successfully discovered catalysts that break conventional scaling relations between reaction intermediates. These results highlight the potential of multi-agent LLM frameworks as a powerful strategy to generate chemical insight and discover promising catalysts.

**arXiv ID:** 2602.21533
</details>

<details>
<summary><strong>ADM-DP: Adaptive Dynamic Modality Diffusion Policy through Vision-Tactile-Graph Fusion for Multi-Agent Manipulation</strong> - Enyi Wang, Wen Fan, Dandan Zhang - [[pdf]](https://arxiv.org/pdf/2602.21622)</summary>

**Abstract:** Multi-agent robotic manipulation remains challenging due to the combined demands of coordination, grasp stability, and collision avoidance in shared workspaces. To address these challenges, we propose the Adaptive Dynamic Modality Diffusion Policy (ADM-DP), a framework that integrates vision, tactile, and graph-based (multi-agent pose) modalities for coordinated control. ADM-DP introduces four key innovations. First, an enhanced visual encoder merges RGB and point-cloud features via Feature-wise Linear Modulation (FiLM) modulation to enrich perception. Second, a tactile-guided grasping strategy uses Force-Sensitive Resistor (FSR) feedback to detect insufficient contact and trigger corrective grasp refinement, improving grasp stability. Third, a graph-based collision encoder leverages shared tool center point (TCP) positions of multiple agents as structured kinematic context to maintain spatial awareness and reduce inter-agent interference. Fourth, an Adaptive Modality Attention Mechanism (AMAM) dynamically re-weights modalities according to task context, enabling flexible fusion. For scalability and modularity, a decoupled training paradigm is employed in which agents learn independent policies while sharing spatial information. This maintains low interdependence between agents while retaining collective awareness. Across seven multi-agent tasks, ADM-DP achieves 12-25% performance gains over state-of-the-art baselines. Ablation studies show the greatest improvements in tasks requiring multiple sensory modalities, validating our adaptive fusion strategy and demonstrating its robustness for diverse manipulation scenarios.

**arXiv ID:** 2602.21622
</details>

<details>
<summary><strong>HetroD: A High-Fidelity Drone Dataset and Benchmark for Autonomous Driving in Heterogeneous Traffic</strong> - Yu-Hsiang Chen, Wei-Jer Chang, Christian Kotulla, Thomas Keutgens, Steffen Runde, Tobias Moers, Christoph Klas, Wei Zhan, Masayoshi Tomizuka, Yi-Ting Chen - [[pdf]](https://arxiv.org/pdf/2602.03447)</summary>

**Abstract:** We present HetroD, a dataset and benchmark for developing autonomous driving systems in heterogeneous environments. HetroD targets the critical challenge of navi- gating real-world heterogeneous traffic dominated by vulner- able road users (VRUs), including pedestrians, cyclists, and motorcyclists that interact with vehicles. These mixed agent types exhibit complex behaviors such as hook turns, lane splitting, and informal right-of-way negotiation. Such behaviors pose significant challenges for autonomous vehicles but remain underrepresented in existing datasets focused on structured, lane-disciplined traffic. To bridge the gap, we collect a large- scale drone-based dataset to provide a holistic observation of traffic scenes with centimeter-accurate annotations, HD maps, and traffic signal states. We further develop a modular toolkit for extracting per-agent scenarios to support downstream task development. In total, the dataset comprises over 65.4k high- fidelity agent trajectories, 70% of which are from VRUs. HetroD supports modeling of VRU behaviors in dense, het- erogeneous traffic and provides standardized benchmarks for forecasting, planning, and simulation tasks. Evaluation results reveal that state-of-the-art prediction and planning models struggle with the challenges presented by our dataset: they fail to predict lateral VRU movements, cannot handle unstructured maneuvers, and exhibit limited performance in dense and multi-agent scenarios, highlighting the need for more robust approaches to heterogeneous traffic. See our project page for more examples: this https URL

**arXiv ID:** 2602.03447
</details>

</details>

<details open>
<summary><h2>Other Agent Research (4 papers)</h2></summary>

<details>
<summary><strong>Beyond Refusal: Probing the Limits of Agentic Self-Correction for Semantic Sensitive Information</strong> - Umid Suleymanov, Zaur Rajabov, Emil Mirzazada, Murat Kantarcioglu - [[pdf]](https://arxiv.org/pdf/2602.21496)</summary>

**Abstract:** While defenses for structured PII are mature, Large Language Models (LLMs) pose a new threat: Semantic Sensitive Information (SemSI), where models infer sensitive identity attributes, generate reputation-harmful content, or hallucinate potentially wrong information. The capacity of LLMs to self-regulate these complex, context-dependent sensitive information leaks without destroying utility remains an open scientific question. To address this, we introduce SemSIEdit, an inference-time framework where an agentic "Editor" iteratively critiques and rewrites sensitive spans to preserve narrative flow rather than simply refusing to answer. Our analysis reveals a Privacy-Utility Pareto Frontier, where this agentic rewriting reduces leakage by 34.6% across all three SemSI categories while incurring a marginal utility loss of 9.8%. We also uncover a Scale-Dependent Safety Divergence: large reasoning models (e.g., GPT-5) achieve safety through constructive expansion (adding nuance), whereas capacity-constrained models revert to destructive truncation (deleting text). Finally, we identify a Reasoning Paradox: while inference-time reasoning increases baseline risk by enabling the model to make deeper sensitive inferences, it simultaneously empowers the defense to execute safe rewrites.

**arXiv ID:** 2602.21496
</details>

<details>
<summary><strong>Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation</strong> - Zhanghao Hu, Qinglin Zhu, Hanqi Yan, Yulan He, Lin Gui - [[pdf]](https://arxiv.org/pdf/2602.02007)</summary>

**Abstract:** Agent memory systems often adopt the standard Retrieval-Augmented Generation (RAG) pipeline, yet its underlying assumptions differ in this setting. RAG targets large, heterogeneous corpora where retrieved passages are diverse, whereas agent memory is a bounded, coherent dialogue stream with highly correlated spans that are often duplicates. Under this shift, fixed top-$k$ similarity retrieval tends to return redundant context, and post-hoc pruning can delete temporally linked prerequisites needed for correct reasoning. We argue retrieval should move beyond similarity matching and instead operate over latent components, following decoupling to aggregation: disentangle memories into semantic components, organise them into a hierarchy, and use this structure to drive retrieval. We propose xMemory, which builds a hierarchy of intact units and maintains a searchable yet faithful high-level node organisation via a sparsity--semantics objective that guides memory split and merge. At inference, xMemory retrieves top-down, selecting a compact, diverse set of themes and semantics for multi-fact queries, and expanding to episodes and raw messages only when it reduces the reader's uncertainty. Experiments on LoCoMo and PerLTQA across the three latest LLMs show consistent gains in answer quality and token efficiency.

**arXiv ID:** 2602.02007
</details>

<details>
<summary><strong>Environment-Aware Learning of Smooth GNSS Covariance Dynamics for Autonomous Racing</strong> - Y. Deemo Chen, Arion Zimmermann, Thomas A. Berrueta, Soon-Jo Chung - [[pdf]](https://arxiv.org/pdf/2602.21366)</summary>

**Abstract:** Ensuring accurate and stable state estimation is a challenging task crucial to safety-critical domains such as high-speed autonomous racing, where measurement uncertainty must be both adaptive to the environment and temporally smooth for control. In this work, we develop a learning-based framework, LACE, capable of directly modeling the temporal dynamics of GNSS measurement covariance. We model the covariance evolution as an exponentially stable dynamical system where a deep neural network (DNN) learns to predict the system's process noise from environmental features through an attention mechanism. By using contraction-based stability and systematically imposing spectral constraints, we formally provide guarantees of exponential stability and smoothness for the resulting covariance dynamics. We validate our approach on an AV-24 autonomous racecar, demonstrating improved localization performance and smoother covariance estimates in challenging, GNSS-degraded environments. Our results highlight the promise of dynamically modeling the perceived uncertainty in state estimation problems that are tightly coupled with control sensitivity.

**arXiv ID:** 2602.21366
</details>

<details>
<summary><strong>Autonomous Sea Turtle Robot for Marine Fieldwork</strong> - Zach J. Patterson, Emily Sologuren, Levi Cai, Daniel Kim, Alaa Maalouf, Pascal Spino, Daniela Rus - [[pdf]](https://arxiv.org/pdf/2602.21389)</summary>

**Abstract:** Autonomous robots can transform how we observe marine ecosystems, but close-range operation in reefs and other cluttered habitats remains difficult. Vehicles must maneuver safely near animals and fragile structures while coping with currents, variable illumination and limited sensing. Previous approaches simplify these problems by leveraging soft materials and bioinspired swimming designs, but such platforms remain limited in terms of deployable autonomy. Here we present a sea turtle-inspired autonomous underwater robot that closed the gap between bioinspired locomotion and field-ready autonomy through a tightly integrated, vision-driven control stack. The robot combines robust depth-heading stabilization with obstacle avoidance and target-centric control, enabling it to track and interact with moving objects in complex terrain. We validate the robot in controlled pool experiments and in a live coral reef exhibit at the New England Aquarium, demonstrating stable operation and reliable tracking of fast-moving marine animals and human divers. To the best of our knowledge, this is the first integrated biomimetic robotic system, combining novel hardware, control, and field experiments, deployed to track and monitor real marine animals in their natural environment. During off-tether experiments, we demonstrate safe navigation around obstacles (91\% success rate in the aquarium exhibit) and introduce a low-compute onboard tracking mode. Together, these results establish a practical route toward soft-rigid hybrid, bioinspired underwater robots capable of minimally disruptive exploration and close-range monitoring in sensitive ecosystems.

**arXiv ID:** 2602.21389
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (26 papers)</h2></summary>

<details>
<summary><strong>ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning</strong> - Xiaoxuan Wang, Han Zhang, Haixin Wang, Yidan Shi, Ruoyan Li, Kaiqiao Han, Chenyi Tong, Haoran Deng, Renliang Sun, Alexander Taylor, Yanqiao Zhu, Jason Cong, Yizhou Sun, Wei Wang - [[pdf]](https://arxiv.org/pdf/2602.21534)</summary>

**Abstract:** Agentic reinforcement learning (ARL) has rapidly gained attention as a promising paradigm for training agents to solve complex, multi-step interactive tasks. Despite encouraging early results, ARL remains highly unstable, often leading to training collapse. This instability limits scalability to larger environments and longer interaction horizons, and constrains systematic exploration of algorithmic design choices. In this paper, we first propose ARLArena, a stable training recipe and systematic analysis framework that examines training stability in a controlled and reproducible setting. ARLArena first constructs a clean and standardized testbed. Then, we decompose policy gradient into four core design dimensions and assess the performance and stability of each dimension. Through this fine-grained analysis, we distill a unified perspective on ARL and propose SAMPO, a stable agentic policy optimization method designed to mitigate the dominant sources of instability in ARL. Empirically, SAMPO achieves consistently stable training and strong performance across diverse agentic tasks. Overall, this study provides a unifying policy gradient perspective for ARL and offers practical guidance for building stable and reproducible LLM-based agent training pipelines.

**arXiv ID:** 2602.21534
</details>

<details>
<summary><strong>2-Step Agent: A Framework for the Interaction of a Decision Maker with AI Decision Support</strong> - Otto Nyberg, Fausto Carcassi, Giovanni Cinà - [[pdf]](https://arxiv.org/pdf/2602.21889)</summary>

**Abstract:** Across a growing number of fields, human decision making is supported by predictions from AI models. However, we still lack a deep understanding of the effects of adoption of these technologies. In this paper, we introduce a general computational framework, the 2-Step Agent, which models the effects of AI-assisted decision making. Our framework uses Bayesian methods for causal inference to model 1) how a prediction on a new observation affects the beliefs of a rational Bayesian agent, and 2) how this change in beliefs affects the downstream decision and subsequent outcome. Using this framework, we show by simulations how a single misaligned prior belief can be sufficient for decision support to result in worse downstream outcomes compared to no decision support. Our results reveal several potential pitfalls of AI-driven decision support and highlight the need for thorough model documentation and proper user training.

**arXiv ID:** 2602.21889
</details>

<details>
<summary><strong>Overconfident Errors Need Stronger Correction: Asymmetric Confidence Penalties for Reinforcement Learning</strong> - Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang - [[pdf]](https://arxiv.org/pdf/2602.21420)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has become the leading paradigm for enhancing reasoning in Large Language Models (LLMs). However, standard RLVR algorithms suffer from a well-documented pathology: while they improve Pass@1 accuracy through sharpened sampling, they simultaneously narrow the model's reasoning boundary and reduce generation diversity. We identify a root cause that existing methods overlook: the uniform penalization of errors. Current approaches -- whether data-filtering methods that select prompts by difficulty, or advantage normalization schemes -- treat all incorrect rollouts within a group identically. We show that this uniformity allows overconfident errors (incorrect reasoning paths that the RL process has spuriously reinforced) to persist and monopolize probability mass, ultimately suppressing valid exploratory trajectories. To address this, we propose the Asymmetric Confidence-aware Error Penalty (ACE). ACE introduces a per-rollout confidence shift metric, c_i = log(pi_theta(y_i|x) / pi_ref(y_i|x)), to dynamically modulate negative advantages. Theoretically, we demonstrate that ACE's gradient can be decomposed into the gradient of a selective regularizer restricted to overconfident errors, plus a well-characterized residual that partially moderates the regularizer's strength. We conduct extensive experiments fine-tuning Qwen2.5-Math-7B, Qwen3-8B-Base, and Llama-3.1-8B-Instruct on the DAPO-Math-17K dataset using GRPO and DAPO within the VERL framework. Evaluated on MATH-500 and AIME 2025, ACE composes seamlessly with existing methods and consistently improves the full Pass@k spectrum across all three model families and benchmarks.

**arXiv ID:** 2602.21420
</details>

<details>
<summary><strong>Adversarial Intent is a Latent Variable: Stateful Trust Inference for Securing Multimodal Agentic RAG</strong> - Inderjeet Singh, Vikas Pahuja, Aishvariya Priya Rathina Sabapathy, Chiara Picardi, Amit Giloni, Roman Vainshtein, Andrés Murillo, Hisashi Kojima, Motoyoshi Sekiya, Yuki Unno, Junichi Suga - [[pdf]](https://arxiv.org/pdf/2602.21447)</summary>

**Abstract:** Current stateless defences for multimodal agentic RAG fail to detect adversarial strategies that distribute malicious semantics across retrieval, planning, and generation components. We formulate this security challenge as a Partially Observable Markov Decision Process (POMDP), where adversarial intent is a latent variable inferred from noisy multi-stage observations. We introduce MMA-RAG^T, an inference-time control framework governed by a Modular Trust Agent (MTA) that maintains an approximate belief state via structured LLM reasoning. Operating as a model-agnostic overlay, MMA-RAGT mediates a configurable set of internal checkpoints to enforce stateful defence-in-depth. Extensive evaluation on 43,774 instances demonstrates a 6.50x average reduction factor in Attack Success Rate relative to undefended baselines, with negligible utility cost. Crucially, a factorial ablation validates our theoretical bounds: while statefulness and spatial coverage are individually necessary (26.4 pp and 13.6 pp gains respectively), stateless multi-point intervention can yield zero marginal benefit under homogeneous stateless filtering when checkpoint detections are perfectly correlated.

**arXiv ID:** 2602.21447
</details>

<details>
<summary><strong>GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning</strong> - Ningyuan Yang, Weihua Du, Weiwei Sun, Sean Welleck, Yiming Yang - [[pdf]](https://arxiv.org/pdf/2602.21492)</summary>

**Abstract:** Reinforcement learning (RL) has become a central post-training paradigm for large language models (LLMs), but its performance is highly sensitive to the quality of training problems. This sensitivity stems from the non-stationarity of RL: rollouts are generated by an evolving policy, and learning is shaped by exploration and reward feedback, unlike supervised fine-tuning (SFT) with fixed trajectories. As a result, prior work often relies on manual curation or simple heuristic filters (e.g., accuracy), which can admit incorrect or low-utility problems. We propose GradAlign, a gradient-aligned data selection method for LLM reinforcement learning that uses a small, trusted validation set to prioritize training problems whose policy gradients align with validation gradients, yielding an adaptive curriculum. We evaluate GradAlign across three challenging data regimes: unreliable reward signals, distribution imbalance, and low-utility training corpus, showing that GradAlign consistently outperforms existing baselines, underscoring the importance of directional gradient signals in navigating non-stationary policy optimization and yielding more stable training and improved final performance. We release our implementation at this https URL

**arXiv ID:** 2602.21492
</details>

<details>
<summary><strong>Structurally Aligned Subtask-Level Memory for Software Engineering Agents</strong> - Kangning Shen, Jingyuan Zhang, Chenxi Sun, Wencong Zeng, Yang Yue - [[pdf]](https://arxiv.org/pdf/2602.21611)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated significant potential as autonomous software engineering (SWE) agents. Recent work has further explored augmenting these agents with memory mechanisms to support long-horizon reasoning. However, these approaches typically operate at a coarse instance granularity, treating the entire problem-solving episode as the atomic unit of storage and retrieval. We empirically demonstrate that instance-level memory suffers from a fundamental granularity mismatch, resulting in misguided retrieval when tasks with similar surface descriptions require distinct reasoning logic at specific stages. To address this, we propose Structurally Aligned Subtask-Level Memory, a method that aligns memory storage, retrieval, and updating with the agent's functional decomposition. Extensive experiments on SWE-bench Verified demonstrate that our method consistently outperforms both vanilla agents and strong instance-level memory baselines across diverse backbones, improving mean Pass@1 over the vanilla agent by +4.7 pp on average (e.g., +6.8 pp on Gemini 2.5 Pro). Performance gains grow with more interaction steps, showing that leveraging past experience benefits long-horizon reasoning in complex software engineering tasks.

**arXiv ID:** 2602.21611
</details>

<details>
<summary><strong>CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning</strong> - Zhijiang Tang, Linhua Wang, Jiaxin Qi, Weihao Jiang, Peng Hou, Anxiang Zeng, Jianqiang Huang - [[pdf]](https://arxiv.org/pdf/2602.21655)</summary>

**Abstract:** Image captioning remains a fundamental task for vision language understanding, yet ground-truth supervision still relies predominantly on human-annotated references. Because human annotations reflect subjective preferences and expertise, ground-truth captions are often incomplete or even incorrect, which in turn limits caption models. We argue that caption quality should be assessed by two objective aspects: completeness (does the caption cover all salient visual facts?) and correctness (are the descriptions true with respect to the image?). To this end, we introduce CCCaption: a dual-reward reinforcement learning framework with a dedicated fine-tuning corpus that explicitly optimizes these properties to generate \textbf{C}omplete and \textbf{C}orrect \textbf{Captions}. For completeness, we use diverse LVLMs to disentangle the image into a set of visual queries, and reward captions that answer more of these queries, with a dynamic query sampling strategy to improve training efficiency. For correctness, we penalize captions that contain hallucinations by validating the authenticity of sub-caption queries, which are derived from the caption decomposition. Our symmetric dual-reward optimization jointly maximizes completeness and correctness, guiding models toward captions that better satisfy these objective criteria. Extensive experiments across standard captioning benchmarks show consistent improvements, offering a principled path to training caption models beyond human-annotation imitation.

**arXiv ID:** 2602.21655
</details>

<details>
<summary><strong>Following the Diagnostic Trace: Visual Cognition-guided Cooperative Network for Chest X-Ray Diagnosis</strong> - Shaoxuan Wu, Jingkun Chen, Chong Ma, Cong Shen, Xiao Zhang, Jun Feng - [[pdf]](https://arxiv.org/pdf/2602.21657)</summary>

**Abstract:** Computer-aided diagnosis (CAD) has significantly advanced automated chest X-ray diagnosis but remains isolated from clinical workflows and lacks reliable decision support and interpretability. Human-AI collaboration seeks to enhance the reliability of diagnostic models by integrating the behaviors of controllable radiologists. However, the absence of interactive tools seamlessly embedded within diagnostic routines impedes collaboration, while the semantic gap between radiologists' decision-making patterns and model representations further limits clinical adoption. To overcome these limitations, we propose a visual cognition-guided collaborative network (VCC-Net) to achieve the cooperative diagnostic paradigm. VCC-Net centers on visual cognition (VC) and employs clinically compatible interfaces, such as eye-tracking or the mouse, to capture radiologists' visual search traces and attention patterns during diagnosis. VCC-Net employs VC as a spatial cognition guide, learning hierarchical visual search strategies to localize diagnostically key regions. A cognition-graph co-editing module subsequently integrates radiologist VC with model inference to construct a disease-aware graph. The module captures dependencies among anatomical regions and aligns model representations with VC-driven features, mitigating radiologist bias and facilitating complementary, transparent decision-making. Experiments on the public datasets SIIM-ACR, EGD-CXR, and self-constructed TB-Mouse dataset achieved classification accuracies of 88.40%, 85.05%, and 92.41%, respectively. The attention maps produced by VCC-Net exhibit strong concordance with radiologists' gaze distributions, demonstrating a mutual reinforcement of radiologist and model inference. The code is available at this https URL.

**arXiv ID:** 2602.21657
</details>

<details>
<summary><strong>Evaluating the relationship between regularity and learnability in recursive numeral systems using Reinforcement Learning</strong> - Andrea Silvi, Ponrawee Prasertsom, Jennifer Culbertson, Devdatt Dubhashi, Moa Johansson, Kenny Smith - [[pdf]](https://arxiv.org/pdf/2602.21720)</summary>

**Abstract:** Human recursive numeral systems (i.e., counting systems such as English base-10 numerals), like many other grammatical systems, are highly regular. Following prior work that relates cross-linguistic tendencies to biases in learning, we ask whether regular systems are common because regularity facilitates learning. Adopting methods from the Reinforcement Learning literature, we confirm that highly regular human(-like) systems are easier to learn than unattested but possible irregular systems. This asymmetry emerges under the natural assumption that recursive numeral systems are designed for generalisation from limited data to represent all integers exactly. We also find that the influence of regularity on learnability is absent for unnatural, highly irregular systems, whose learnability is influenced instead by signal length, suggesting that different pressures may influence learnability differently in different parts of the space of possible numeral systems. Our results contribute to the body of work linking learnability to cross-linguistic prevalence.

**arXiv ID:** 2602.21720
</details>

<details>
<summary><strong>SWE-Protégé: Learning to Selectively Collaborate With an Expert Unlocks Small Language Models as Software Engineering Agents</strong> - Patrick Tser Jern Kon, Archana Pradeep, Ang Chen, Alexander P. Ellis, Warren Hunt, Zijian Wang, John Yang, Samuel Thompson - [[pdf]](https://arxiv.org/pdf/2602.22124)</summary>

**Abstract:** Small language models (SLMs) offer compelling advantages in cost, latency, and adaptability, but have so far lagged behind larger models on long-horizon software engineering tasks such as SWE-bench, where they suffer from pervasive action looping and low resolution rates. We introduce SWE-Protégé, a post-training framework that reframes software repair as an expert-protégé collaboration problem. In SWE-Protégé, an SLM remains the sole decision-maker while learning to selectively seek guidance from a strong expert model, recognize stalled states, and follow through on expert feedback. Our approach combines supervised fine-tuning on expert-augmented trajectories with agentic reinforcement learning that explicitly discourages degenerative looping and unproductive expert collaboration. We lightly post-train Qwen2.5-Coder-7B-Instruct to achieve 42.4% Pass@1 on SWE-bench Verified, a +25.4% improvement over the prior SLM state of the art, while using expert assistance sparsely (~4 calls per task and 11% of total tokens).

**arXiv ID:** 2602.22124
</details>

<details>
<summary><strong>Evolutionary System Prompt Learning for Reinforcement Learning in LLMs</strong> - Lunjun Zhang, Ryan Chen, Bradly C. Stadie - [[pdf]](https://arxiv.org/pdf/2602.14697)</summary>

**Abstract:** Building agentic systems that can autonomously self-improve from experience is a longstanding goal of AI. Large language models (LLMs) today primarily self-improve via two mechanisms: self-reflection for context updates, and reinforcement learning (RL) for weight updates. In this work, we propose Evolutionary System Prompt Learning (E-SPL), a method for jointly improving model contexts and model weights. In each RL iteration, E-SPL samples trajectories under multiple system prompts in parallel, then jointly applies RL updates to LLM weights and evolutionary updates to system prompts. System prompts evolve via mutation and crossover, two genetic operators driven by LLM self-reflection; selection is based on relative performance ratings updated across RL iterations. E-SPL encourages a natural division between declarative knowledge encoded in prompts and procedural knowledge encoded in weights, resulting in improved performance across reasoning and agentic tasks. For instance, in an easy-to-hard (AIME $\rightarrow$ BeyondAIME) generalization setting, E-SPL improves RL success rate from 38.8% $\rightarrow$ 45.1% while also outperforming reflective prompt evolution (40.0%). Overall, our results demonstrate that RL and system prompt evolution are deeply synergistic, and combining the two yields consistent gains in sample efficiency and generalization. Code: this https URL

**arXiv ID:** 2602.14697
</details>

<details>
<summary><strong>FML-bench: Benchmarking Machine Learning Agents for Scientific Research</strong> - Qiran Zou, Hou Hei Lam, Wenhao Zhao, Yiming Tang, Tingting Chen, Samson Yu, Tianyi Zhang, Chang Liu, Xiangyang Ji, Dianbo Liu - [[pdf]](https://arxiv.org/pdf/2510.10472)</summary>

**Abstract:** Large language models (LLMs) have sparked growing interest in machine learning research agents that can autonomously propose ideas and conduct experiments. However, existing benchmarks predominantly adopt an engineering-oriented perspective: they emphasize application-oriented tasks and evaluate primarily on final performance and computational cost, overlooking agents' research processes and limiting assessment of their capabilities in scientific research settings. To more comprehensively evaluate agents in scientific research settings, we introduce FML-bench, a benchmark comprising 8 diverse and fundamental ML research tasks, and further propose complementary metrics, notably Exploration Diversity, which quantifies the variance of proposals across iterations and reveals how exploration patterns influence research outcomes. We evaluate state-of-the-art research agents on FML-bench, showing that agents employing broad exploration strategies exhibit higher exploration diversity and achieve superior performance, and that exploration diversity positively correlates with performance improvements across multiple tasks. We hope these findings and our benchmark inform future agent design and support the community in further investigating agent behavior. Our benchmark is available at this https URL.

**arXiv ID:** 2510.10472
</details>

<details>
<summary><strong>Aerial Vision-Language Navigation with a Unified Framework for Spatial, Temporal and Embodied Reasoning</strong> - Huilin Xu, Zhuoyang Liu, Yixiang Luomei, Feng Xu - [[pdf]](https://arxiv.org/pdf/2512.08639)</summary>

**Abstract:** Aerial Vision-and-Language Navigation (VLN) aims to enable unmanned aerial vehicles (UAVs) to interpret natural language instructions and navigate complex urban environments using onboard visual observation. This task holds promise for real-world applications such as low-altitude inspection, search-and-rescue, and autonomous aerial delivery. Existing methods often rely on panoramic images, depth inputs, or odometry to support spatial reasoning and action planning. These requirements increase system cost and integration complexity, thus hindering practical deployment for lightweight UAVs. We present a unified aerial VLN framework that operates solely on egocentric monocular RGB observations and natural language instructions. The model formulates navigation as a next-token prediction problem, jointly optimizing spatial perception, trajectory reasoning, and action prediction through prompt-guided multi-task learning. Moreover, we propose a keyframe selection strategy to reduce visual redundancy by retaining semantically informative frames, along with an action merging and label reweighting mechanism that mitigates long-tailed supervision imbalance and facilitates stable multi-task co-training. Extensive experiments on the AerialVLN and OpenFly benchmark validate the effectiveness of our method. Under the challenging monocular RGB-only setting, our model achieves strong results across both seen and unseen environments. It significantly outperforms existing RGB-only baselines and narrows the performance gap with state-of-the-art panoramic RGB-D counterparts. Comprehensive ablation studies further demonstrate the contribution of our task design and architectural choices.

**arXiv ID:** 2512.08639
</details>

<details>
<summary><strong>RebuttalAgent: Strategic Persuasion in Academic Rebuttal via Theory of Mind</strong> - Zhitao He, Zongwei Lyu, Yi R Fung - [[pdf]](https://arxiv.org/pdf/2601.15715)</summary>

**Abstract:** Although artificial intelligence (AI) has become deeply integrated into various stages of the research workflow and achieved remarkable advancements, academic rebuttal remains a significant and underexplored challenge. This is because rebuttal is a complex process of strategic communication under severe information asymmetry rather than a simple technical debate. Consequently, current approaches struggle as they largely imitate surface-level linguistics, missing the essential element of perspective-taking required for effective persuasion. In this paper, we introduce RebuttalAgent, the first framework to ground academic rebuttal in Theory of Mind (ToM), operationalized through a ToM-Strategy-Response (TSR) framework that models reviewer mental state, formulates persuasion strategy, and generates evidence-based response. To train our agent, we construct RebuttalBench, a large-scale dataset synthesized via a novel critique-and-refine approach. Our training process consists of two stages, beginning with a supervised fine-tuning phase to equip the agent with ToM-based analysis and strategic planning capabilities, followed by a reinforcement learning phase leveraging the self-reward mechanism for scalable self-improvement. For reliable and efficient automated evaluation, we further develop Rebuttal-RM, a specialized evaluator trained on over 100K samples of multi-source rebuttal data, which achieves scoring consistency with human preferences surpassing powerful judge GPT-4.1. Extensive experiments show RebuttalAgent significantly outperforms the base model by an average of 18.3% on automated metrics, while also outperforming advanced proprietary models across both automated and human evaluations.

**arXiv ID:** 2601.15715
</details>

<details>
<summary><strong>Quantifying the Expectation-Realisation Gap for Agentic AI Systems</strong> - Sebastian Lobentanzer - [[pdf]](https://arxiv.org/pdf/2602.20292)</summary>

**Abstract:** Agentic AI systems are deployed with expectations of substantial productivity gains, yet rigorous empirical evidence reveals systematic discrepancies between pre-deployment expectations and post-deployment outcomes. We review controlled trials and independent validations across software engineering, clinical documentation, and clinical decision support to quantify this expectation-realisation gap. In software development, experienced developers expected a 24% speedup from AI tools but were slowed by 19% -- a 43 percentage-point calibration error. In clinical documentation, vendor claims of multi-minute time savings contrast with measured reductions of less than one minute per note, and one widely deployed tool showed no statistically significant effect. In clinical decision support, externally validated performance falls substantially below developer-reported metrics. These shortfalls are driven by workflow integration friction, verification burden, measurement construct mismatches, and systematic variation in who benefits and who does not. The evidence motivates structured planning frameworks that require explicit, quantified benefit expectations with human oversight costs factored in.

**arXiv ID:** 2602.20292
</details>

<details>
<summary><strong>Explore-on-Graph: Incentivizing Autonomous Exploration of Large Language Models on Knowledge Graphs with Path-refined Reward Modeling</strong> - Shiqi Yan, Yubo Chen, Ruiqi Zhou, Zhengxi Yao, Shuai Chen, Tianyi Zhang, Shijie Zhang, Wei Qiang Zhang, Yongfeng Huang, Haixin Duan, Yunqi Zhang - [[pdf]](https://arxiv.org/pdf/2602.21728)</summary>

**Abstract:** The reasoning process of Large Language Models (LLMs) is often plagued by hallucinations and missing facts in question-answering tasks. A promising solution is to ground LLMs' answers in verifiable knowledge sources, such as Knowledge Graphs (KGs). Prevailing KG-enhanced methods typically constrained LLM reasoning either by enforcing rules during generation or by imitating paths from a fixed set of demonstrations. However, they naturally confined the reasoning patterns of LLMs within the scope of prior experience or fine-tuning data, limiting their generalizability to out-of-distribution graph reasoning problems. To tackle this problem, in this paper, we propose Explore-on-Graph (EoG), a novel framework that encourages LLMs to autonomously explore a more diverse reasoning space on KGs. To incentivize exploration and discovery of novel reasoning paths, we propose to introduce reinforcement learning during training, whose reward is the correctness of the reasoning paths' final answers. To enhance the efficiency and meaningfulness of the exploration, we propose to incorporate path information as additional reward signals to refine the exploration process and reduce futile efforts. Extensive experiments on five KGQA benchmark datasets demonstrate that, to the best of our knowledge, our method achieves state-of-the-art performance, outperforming not only open-source but also even closed-source LLMs.

**arXiv ID:** 2602.21728
</details>

<details>
<summary><strong>ABM-UDE: Developing Surrogates for Epidemic Agent-Based Models via Scientific Machine Learning</strong> - Sharv Murgai, Utkarsh Utkarsh, Kyle C. Nguyen, Alan Edelman, Erin C. S. Acquesta, Christopher Vincent Rackauckas - [[pdf]](https://arxiv.org/pdf/2602.21588)</summary>

**Abstract:** Agent-based epidemic models (ABMs) encode behavioral and policy heterogeneity but are too slow for nightly hospital planning. We develop county-ready surrogates that learn directly from exascale ABM trajectories using Universal Differential Equations (UDEs): mechanistic SEIR-family ODEs with a neural-parameterized contact rate $\kappa_\phi(u,t)$ (no additive residual). Our contributions are threefold: we adapt multiple shooting and an observer-based prediction-error method (PEM) to stabilize identification of neural-augmented epidemiological dynamics across intervention-driven regime shifts; we enforce positivity and mass conservation and show the learned contact-rate parameterization yields a well-posed vector field; and we quantify accuracy, calibration, and compute against ABM ensembles and UDE baselines. On a representative ExaEpi scenario, PEM-UDE reduces mean MSE by 77% relative to single-shooting UDE (3.00 vs. 13.14) and by 20% relative to MS-UDE (3.75). Reliability improves in parallel: empirical coverage of ABM $10$-$90$% and $25$-$75$% bands rises from 0.68/0.43 (UDE) and 0.79/0.55 (MS-UDE) to 0.86/0.61 with PEM-UDE and 0.94/0.69 with MS+PEM-UDE, indicating calibrated uncertainty rather than overconfident fits. Inference runs in seconds on commodity CPUs (20-35 s per $\sim$90-day forecast), enabling nightly ''what-if'' sweeps on a laptop. Relative to a $\sim$100 CPU-hour ABM reference run, this yields $\sim10^{4}\times$ lower wall-clock per scenario. This closes the realism-cadence gap, supports threshold-aware decision-making (e.g., maintaining ICU occupancy $<75$%), preserves mechanistic interpretability, and enables calibrated, risk-aware scenario planning on standard institutional hardware. Beyond epidemics, the ABM$\to$UDE recipe provides a portable path to distill agent-based simulators into fast, trustworthy surrogates for other scientific domains.

**arXiv ID:** 2602.21588
</details>

<details>
<summary><strong>Object-Centric World Models from Few-Shot Annotations for Sample-Efficient Reinforcement Learning</strong> - Weipu Zhang, Adam Jelley, Trevor McInroe, Amos Storkey, Gang Wang - [[pdf]](https://arxiv.org/pdf/2501.16443)</summary>

**Abstract:** While deep reinforcement learning (RL) from pixels has achieved remarkable success, its sample inefficiency remains a critical limitation for real-world applications. Model-based RL (MBRL) addresses this by learning a world model to generate simulated experience, but standard approaches that rely on pixel-level reconstruction losses often fail to capture small, task-critical objects in complex, dynamic scenes. We posit that an object-centric (OC) representation can direct model capacity toward semantically meaningful entities, improving dynamics prediction and sample efficiency. In this work, we introduce OC-STORM, an object-centric MBRL framework that enhances a learned world model with object representations extracted by a pretrained segmentation network. By conditioning on a minimal number of annotated frames, OC-STORM learns to track decision-relevant object dynamics and inter-object interactions without extensive labeling or access to privileged information. Empirical results demonstrate that OC-STORM significantly outperforms the STORM baseline on the Atari 100k benchmark and achieves state-of-the-art sample efficiency on challenging boss fights in the visually complex game Hollow Knight. Our findings underscore the potential of integrating OC priors into MBRL for complex visual domains. Project page: this https URL

**arXiv ID:** 2501.16443
</details>

<details>
<summary><strong>SERL: Self-Examining Reinforcement Learning on Open-Domain</strong> - Weixuan Ou, Yanzhao Zheng, Shuoshuo Sun, Wei Zhang, Baohua Dong, Hangcheng Zhu, Ruohui Huang, Gang Yu, Pengwei Yan, Yifan Qiao - [[pdf]](https://arxiv.org/pdf/2511.07922)</summary>

**Abstract:** Reinforcement Learning (RL) has been shown to improve the capabilities of large language models (LLMs). However, applying RL to open-domain tasks faces two key challenges: (1) the inherent subjectivity of these tasks prevents the verifiable rewards as required by Reinforcement Learning with Verifiable Rewards (RLVR); (2) Reinforcement Learning from Human Feedback (RLHF) relies on external reward mechanisms. To overcome these limitations, we propose Self-Examining Reinforcement Learning (SERL), a novel self-improving framework where the LLM serves as both Actor and Judge. SERL introduces two synergistic reward mechanisms without any external signals. On the one hand, to improve the Actor's capability, we derive rewards from Copeland-style pairwise comparison judgments across a group of generated responses. On the other hand, a self-consistency reward that encourages coherent judgments is proposed to improve the Judge's reliability. This process refines the Judge's capability, which in turn provides a more robust reward for Actor. Experiments show that our method outperforms existing self-improvement training methods. SERL improves the LC win rate of Qwen3-8B on AlpacaEval 2 from 52.37% to 59.90%. To the best of our knowledge, our method achieves state-of-the-art performance among self-improving approaches. Furthermore, it achieves a performance comparable to significantly larger models like Qwen3-32B, demonstrating superior effectiveness and robustness on open-domain tasks.

**arXiv ID:** 2511.07922
</details>

<details>
<summary><strong>Efficient Cross-Domain Offline Reinforcement Learning with Dynamics- and Value-Aligned Data Filtering</strong> - Zhongjian Qiao, Rui Yang, Jiafei Lyu, Chenjia Bai, Xiu Li, Siyang Gao, Shuang Qiu - [[pdf]](https://arxiv.org/pdf/2512.02435)</summary>

**Abstract:** Cross-domain offline reinforcement learning (RL) aims to train a well-performing agent in the target environment, leveraging both a limited target domain dataset and a source domain dataset with (possibly) sufficient data coverage. Due to the underlying dynamics misalignment between source and target domains, naively merging the two datasets may incur inferior performance. Recent advances address this issue by selectively leveraging source domain samples whose dynamics align well with the target domain. However, our work demonstrates that dynamics alignment alone is insufficient, by examining the limitations of prior frameworks and deriving a new target domain sub-optimality bound for the policy learned on the source domain. More importantly, our theory underscores an additional need for \textit{value alignment}, i.e., selecting high-quality, high-value samples from the source domain, a critical dimension overlooked by existing works. Motivated by such theoretical insight, we propose \textbf{\underline{D}}ynamics- and \textbf{\underline{V}}alue-aligned \textbf{\underline{D}}ata \textbf{\underline{F}}iltering (DVDF) method, a novel unified cross-domain RL framework that selectively incorporates source domain samples exhibiting strong alignment in \textit{both dynamics and values}. We empirically study a range of dynamics shift scenarios, including kinematic and morphology shifts, and evaluate DVDF on various tasks and datasets, even in the challenging setting where the target domain dataset contains an extremely limited amount of data. Extensive experiments demonstrate that DVDF consistently outperforms strong baselines with significant improvements.

**arXiv ID:** 2512.02435
</details>

<details>
<summary><strong>WebGym: Scaling Training Environments for Visual Web Agents with Realistic Tasks</strong> - Hao Bai, Alexey Taymanov, Tong Zhang, Aviral Kumar, Spencer Whitehead - [[pdf]](https://arxiv.org/pdf/2601.02439)</summary>

**Abstract:** We present WebGym, the largest-to-date open-source environment for training realistic visual web agents. Real websites are non-stationary and diverse, making artificial or small-scale task sets insufficient for robust policy learning. WebGym contains nearly 300,000 tasks with rubric-based evaluations across diverse, real-world websites and difficulty levels. We train agents with a simple reinforcement learning (RL) recipe, which trains on the agent's own interaction traces (rollouts), using task rewards as feedback to guide learning. To enable scaling RL, we speed up sampling of trajectories in WebGym by developing a high-throughput asynchronous rollout system, designed specifically for web agents. Our system achieves a 4-5x rollout speedup compared to naive implementations. Second, we scale the task set breadth, depth, and size, which results in continued performance improvement. Fine-tuning a strong base vision-language model, Qwen-3-VL-8B-Instruct, on WebGym results in an improvement in success rate on an out-of-distribution test set from 26.2% to 42.9%, significantly outperforming agents based on proprietary models such as GPT-4o and GPT-5-Thinking that achieve 27.1% and 29.8%, respectively. This improvement is substantial because our test set consists only of tasks on websites never seen during training, unlike many other prior works on training visual web agents.

**arXiv ID:** 2601.02439
</details>

<details>
<summary><strong>Stagewise Reinforcement Learning and the Geometry of the Regret Landscape</strong> - Chris Elliott, Einar Urdshals, David Quarel, Matthew Farrugia-Roberts, Daniel Murfet - [[pdf]](https://arxiv.org/pdf/2601.07524)</summary>

**Abstract:** Singular learning theory characterizes Bayesian learning as an evolving tradeoff between accuracy and complexity, with transitions between qualitatively different solutions as sample size increases. We extend this theory to reinforcement learning, proving that the concentration of a generalized posterior over policies is governed by the local learning coefficient (LLC), an invariant of the geometry of the regret function. This theory predicts that deep reinforcement learning with SGD should proceed from simple policies with high regret to complex policies with low regret. We verify this prediction empirically in a gridworld environment exhibiting stagewise policy development: phase transitions over training manifest as "opposing staircases" where regret decreases sharply while the LLC increases.

**arXiv ID:** 2601.07524
</details>

<details>
<summary><strong>A Distributional Treatment of Real2Sim2Real for Object-Centric Agent Adaptation in Vision-Driven Deformable Linear Object Manipulation</strong> - Georgios Kamaras, Subramanian Ramamoorthy - [[pdf]](https://arxiv.org/pdf/2502.18615)</summary>

**Abstract:** We present an integrated (or end-to-end) framework for the Real2Sim2Real problem of manipulating deformable linear objects (DLOs) based on visual perception. Working with a parameterised set of DLOs, we use likelihood-free inference (LFI) to compute the posterior distributions for the physical parameters using which we can approximately simulate the behaviour of each specific DLO. We use these posteriors for domain randomisation while training, in simulation, object-specific visuomotor policies (i.e. assuming only visual and proprioceptive sensory) for a DLO reaching task, using model-free reinforcement learning. We demonstrate the utility of this approach by deploying sim-trained DLO manipulation policies in the real world in a zero-shot manner, i.e. without any further fine-tuning. In this context, we evaluate the capacity of a prominent LFI method to perform fine classification over the parametric set of DLOs, using only visual and proprioceptive data obtained in a dynamic manipulation trajectory. We then study the implications of the resulting domain distributions in sim-based policy learning and real-world performance.

**arXiv ID:** 2502.18615
</details>

<details>
<summary><strong>Self-Curriculum Model-based Reinforcement Learning for Shape Control of Deformable Linear Objects</strong> - Zhaowei Liang, Song Wang, Zhao Jin, Shirui Wu, Dan Wu - [[pdf]](https://arxiv.org/pdf/2602.21816)</summary>

**Abstract:** Precise shape control of Deformable Linear Objects (DLOs) is crucial in robotic applications such as industrial and medical fields. However, existing methods face challenges in handling complex large deformation tasks, especially those involving opposite curvatures, and lack efficiency and precision. To address this, we propose a two-stage framework combining Reinforcement Learning (RL) and online visual servoing. In the large-deformation stage, a model-based reinforcement learning approach using an ensemble of dynamics models is introduced to significantly improve sample efficiency. Additionally, we design a self-curriculum goal generation mechanism that dynamically selects intermediate-difficulty goals with high diversity through imagined evaluations, thereby optimizing the policy learning process. In the small-deformation stage, a Jacobian-based visual servo controller is deployed to ensure high-precision convergence. Simulation results show that the proposed method enables efficient policy learning and significantly outperforms mainstream baselines in shape control success rate and precision. Furthermore, the framework effectively transfers the policy trained in simulation to real-world tasks with zero-shot adaptation. It successfully completes all 30 cases with diverse initial and target shapes across DLOs of different sizes and materials. The project website is available at: this https URL

**arXiv ID:** 2602.21816
</details>

<details>
<summary><strong>Lang2Lift: A Language-Guided Autonomous Forklift System for Outdoor Industrial Pallet Handling</strong> - Huy Hoang Nguyen, Johannes Huemer, Markus Murschitz, Tobias Glueck, Minh Nhat Vu, Andreas Kugi - [[pdf]](https://arxiv.org/pdf/2508.15427)</summary>

**Abstract:** Automating pallet handling in outdoor logistics and construction environments remains challenging due to unstructured scenes, variable pallet configurations, and changing environmental conditions. In this paper, we present Lang2Lift, an end-to-end language-guided autonomous forklift system designed to support practical pallet pick-up operations in real-world outdoor settings. The system enables operators to specify target pallets using natural language instructions, allowing flexible selection among multiple pallets with different loads and spatial arrangements. Lang2Lift integrates foundation-model-based perception modules with motion planning and control in a closed-loop autonomy pipeline. Language-grounded visual perception is used to identify and segment target pallets, followed by 6D pose estimation and geometric refinement to generate manipulation-feasible insertion poses. The resulting pose estimates are directly coupled with the forklift planning and control modules to execute fully autonomous pallet pick-up maneuvers. We deploy and evaluate the proposed system on the ADAPT autonomous outdoor forklift platform across diverse real-world scenarios, including cluttered scenes, variable lighting, and different payload configurations. Tolerance-based pose evaluation further indicates accuracy sufficient for successful fork insertion. Timing and failure analyses highlight key deployment trade-offs and practical limitations, providing insights into integrating language-guided perception within industrial automation systems. Video demonstrations are available at this https URL

**arXiv ID:** 2508.15427
</details>

<details>
<summary><strong>ViSTAR: Virtual Skill Training with Augmented Reality with 3D Avatars and LLM coaching agent</strong> - Chunggi Lee, Hayato Saiki, Tica Lin, Eiji Ikeda, Kenji Suzuki, Chen Zhu-Tian, Hanspeter Pfister - [[pdf]](https://arxiv.org/pdf/2602.22077)</summary>

**Abstract:** We present ViSTAR, a Virtual Skill Training system in AR that supports self-guided basketball skill practice, with feedback on balance, posture, and timing. From a formative study with basketball players and coaches, the system addresses three challenges: understanding skills, identifying errors, and correcting mistakes. ViSTAR follows the Behavioral Skills Training (BST) framework-instruction, modeling, rehearsal, and feedback. It provides feedback through visual overlays, rhythm and timing cues, and an AI-powered coaching agent using 3D motion reconstruction. We generate verbal feedback by analyzing spatio-temporal joint data and mapping features to natural-language coaching cues via a Large Language Model (LLM). A key novelty is this feedback generation: motion features become concise coaching insights. In two studies (N=16), participants generally preferred our AI-generated feedback to coach feedback and reported that ViSTAR helped them notice posture and balance issues and refine movements beyond self-observation.

**arXiv ID:** 2602.22077
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
