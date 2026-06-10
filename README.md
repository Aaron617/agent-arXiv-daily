# Agent arXiv Daily

**Last Updated:** 2026-06-10 05:47:39

**Total Papers:** 28

## Table of Contents

- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>Bittensor Agent Arenas as a Trajectory Primitive: Distilling a Shopping Agent from ShoppingBench Subnet Traces</strong> - Shardul Bansal, Seth Schilbe, Jarrod Barnes - [[pdf]](https://arxiv.org/pdf/2606.10064)</summary>

**Abstract:** Small-model agentic post-training is bottlenecked less by the algorithm than by the trajectory substrate it consumes. Leading recipes (RLVR, group-relative RL, rejection-sampled re-SFT) all need multi-turn traces carrying per-trajectory supervision, and the two existing sources fall short: frontier-synthesised data inherits the synthesizer's biases and collapses the long tail, while unfiltered production logs are unjudged and contaminated by shortcut behaviour. We argue that an incentive-aligned agent arena can be engineered to manufacture such trajectories, and demonstrate this on ORO Subnet 15 (SN15), a Bittensor deployment of the ShoppingBench agentic-commerce benchmark. SN15's race mechanism, LLM reasoning judge, and rotating leak-cluster-guarded problem suite yield a corpus with three properties: incentive-aligned diversity, per-trajectory judging, and anti-memorised held-out evaluation. We introduce a structural-quality filter that converts the raw firehose into a trainable corpus by keeping agentic trajectories (the model itself emits the tool calls) and rejecting sub-task trajectories (the model only classifies or narrates over a deterministic search loop), then post-train Qwen3-4B with a recipe matched to the published ShoppingBench SFT-then-GRPO pipeline. On a leak-cluster-guarded held-out partition scored production-strict, the model lifts from the published Qwen3-4B base of 18.0% ASR to 42.7%, within single-problem noise of the synthetic-data SFT-only baseline (43.6%), while training on a fraction of a single day of subnet output. The supervised stack leaves a large pass@8 to pass@1 gap (53.3% vs 34.8%); a per-step teacher-grounded Dr. GRPO reward converts that headroom into process improvement, and we identify the sub-task firehose as the primary lever for closing the gap to the 48.7% SFT+GRPO bar. We release the filter, the corpus splits, and the arena mechanics.

**arXiv ID:** 2606.10064
</details>

<details>
<summary><strong>What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents</strong> - Jiaheng Hu, Mohit Shridhar, Caden Lu, Dhruv Shah, Hao-Tien Lewis Chiang, Jie Tan, Annie Xie - [[pdf]](https://arxiv.org/pdf/2606.10267)</summary>

**Abstract:** Hierarchical vision-language-action (Hi-VLA) systems have emerged as a promising paradigm for complex robot manipulation, by using high-level VLM planners to decompose tasks into language subgoals executed by low-level VLA controllers. Despite recent empirical progress, there is a lack of unified design principles for these systems: existing Hi-VLA systems differ in how they choose and connect planners, controllers, mechanisms to switch between the two, and how observations and memory are represented in the planner. In this paper, we present a systematic study of Hi-VLA design for robot manipulation. We unify representative Hi-VLA agents under an options-style control framework and benchmark core design choices across short-horizon, long-horizon, and reasoning-intensive tasks. Our analysis distills practical principles for building Hi-VLA systems, showing how model choices and interface mechanisms jointly shape performance. Applying these principles yields a substantially stronger system than either flat VLA control or a naively designed hierarchy, across experiments both in simulation and on a real ALOHA robot. Overall, our results provide a foundation for building more capable, robust, and principled hierarchical VLA agents. More information and video at this http URL.

**arXiv ID:** 2606.10267
</details>

<details>
<summary><strong>Rethinking Embodied Navigation via Relational Inductive Bias</strong> - Weitao An, Chenghao Xu, Xu Yang, Cheng Deng - [[pdf]](https://arxiv.org/pdf/2606.10348)</summary>

**Abstract:** Object navigation requires an agent to locate a target in an unknown environment through visual observations. Existing methods typically rely on open-vocabulary detectors or vision-language models (VLMs) to answer where to search, but often overlook what not to trust - which semantic cues are unreliable. Open-vocabulary perception is prone to systematic misleading evidence: false positives, outdated static priors, and repeated failed exploration due to lack of embodied verification, which contaminates mapping and decision-making. Such errors are rooted in structured object relations in real-world scenes. To address this, we propose DB-Nav, a framework that reshapes the search space via dual relational biases. It factorizes target-centric relations into an Activation Bias (propagates contextual evidence) and an Inhibition Bias (suppresses unreliable regions via perceptual confusion and action-level falsification). These biases are unified into a Relational Activation-Inhibition Exploration Graph that modulates frontier exploration values using online observations and failed accesses. Experiments on ObjectNav benchmarks show that DB-Nav significantly outperforms existing methods in success rate (SR) and Success weighted by Path Length (SPL), offering a lightweight, interpretable, and robust navigation framework without costly online VLM reasoning.

**arXiv ID:** 2606.10348
</details>

<details>
<summary><strong>AgenticNav: Zero-Shot Vision-and-Language Navigation as a Tool-Calling Harness</strong> - Yijian Li, Changze Li, Hantian Shi, Jiaying Luo, Jiyuan Cai, Ming Yang, Tong Qin - [[pdf]](https://arxiv.org/pdf/2606.10577)</summary>

**Abstract:** Zero-shot vision-and-language navigation in continuous environments (VLN-CE) has recently become feasible with large vision-language models (VLMs). However, existing methods typically rely on learned waypoint predictors to propose navigable actions. This severely limits the model's action space and fails to leverage depth inputs effectively. Moreover, memory is commonly handled by accumulating long textual or visual histories with substantial irrelevant context, or by retrieving cross-episode experiences, which weakens the zero-shot setting. In this paper, we rethink zero-shot VLN-CE as an agentic interface between the VLM and the environment, and present AgenticNav, a lightweight navigation harness that exposes action, depth, and memory as callable tools. Instead of choosing from predicted waypoints, the action tool allows the VLM to directly select a target pixel in RGB observations, converting it into executable motion. Depth is exposed through an on-demand pixel-depth tool, enabling the VLM to request precise metric distances only where they matter. For memory, AgenticNav provides a compact map image summarizing the historical trajectory, paired with a recall tool that allows the VLM to selectively revisit past visual observations without overwhelming the prompt context. On the R2R-CE benchmark, AgenticNav establishes new state-of-the-art (SOTA) performance among zero-shot methods given the same VLM backbone. Real-world validation further highlights its zero-shot generalization compared to prior methods. Ablations show that our action tool design outperforms traditional waypoint predictors, and that depth tool and agentic memory further contribute to navigation performance.

**arXiv ID:** 2606.10577
</details>

<details>
<summary><strong>Self-Supervised Relevance Modelling in Autonomous Driving via Counterfactual Analysis</strong> - Luca Lusvarghi, Javier Gozalvez, Pablo Urbano Hidalgo - [[pdf]](https://arxiv.org/pdf/2606.10688)</summary>

**Abstract:** Autonomous driving relies on computationally intensive perception pipelines to continuously detect and track objects in the surrounding environment. While some objects are key to plan safe and effective maneuvers, others may not be relevant and have no impact on the autonomous vehicle's driving decisions. Focusing on relevant objects allows a more efficient usage of available computational resources, reduces processing latencies, and limits the downstream propagation of perception noise. In this work, we propose a novel self-supervised approach based on counterfactual analysis to develop a relevance model - an AI-based tool that quantifies the relevance of objects for an autonomous vehicle. To demonstrate the potential of the proposed approach, we train a relevance model on a synthetic causal dataset generated in a selected urban scenario. Results show that the relevance model is able to accurately estimate the objects' relevance with millisecond-level latency, enabling real-time relevance estimation also in high-density scenarios. We also show that the relevance model can be used to build relevance heatmaps that offer valuable insights into the autonomous vehicle's driving policy and can be used to proactively inform perception and planning tasks. We openly release both the relevance model and the causal dataset.

**arXiv ID:** 2606.10688
</details>

<details>
<summary><strong>Diffusion Forcing Planner: History-Annealed Planning with Time-Dependent Guidance for Autonomous Driving</strong> - Zehan Zhang, Neng Zhang, Yaoyi Li, Jia Cai, Zhiling Wang - [[pdf]](https://arxiv.org/pdf/2606.11019)</summary>

**Abstract:** Learning-based motion planners, despite recent progress, often suffer from temporal inconsistency. Small perturbations across frames can accumulate into unstable trajectories, degrading comfort and safety in closed-loop driving. Several methods attempt to inject history as a static conditioning signal to stabilize outputs, only to induce the planner to copy historical patterns instead of adapting to environment contexts. To address this limitation, we propose Diffusion Forcing Planner (DFP), a diffusion-based planning framework driven by history-guided control. Specifically, DFP decomposes the full trajectory into history, current and future segments, and assign independent noise levels to each segment. The model jointly denoises the historical and the future segments, enforcing a heterogeneous joint diffusion process. At inference, classifier-free guidance (CFG) is applied to steer future sampling using annealed history in a controllable manner. Closed-loop evaluation and comprehensive ablations on nuPlan show that DFP achieves competitive performance while producing continuous, stable, and controllable motion plans in complex driving scenarios.

**arXiv ID:** 2606.11019
</details>

<details>
<summary><strong>EM-Fall: Embodied mmWave Sensing for Day-and-Night Fall Detection on Humanoid Robots</strong> - Yanshuo Lu, Yuxuan Hu, Shenghai Yuan, Xinyu Zhou, Kuangji Zuo, Bofan Lyu, XiChen Yuan, Jianfei Yang - [[pdf]](https://arxiv.org/pdf/2606.11109)</summary>

**Abstract:** Falls are one of the leading causes of injury and hospitalization among elderly individuals, making reliable fall awareness an essential capability for safety monitoring in residential environments. However, existing fall detection systems often rely on wearable devices or fixed sensing installations, which may suffer from low user compliance, limited spatial coverage, or degraded performance under occlusion and poor lighting conditions. In this work, we propose \textbf{EM-Fall}, an embodied fall detection framework deployed on a mobile humanoid robot. The system integrates millimeter-wave (mmWave) sensing with robotic mobility, allowing the robot to actively adjust its sensing viewpoint and maintain target observability across rooms and under occlusion. To address interference in complex residential environments, including pet motion and multipath artifacts, we design a human-centered perception pipeline combined with lightweight temporal modeling to capture motion evolution before, during, and after fall events. We evaluate the proposed system across eight real indoor environments with four participants and construct an in-home mmWave fall detection dataset. Experimental results show that the embodied mobile sensing paradigm improves monitoring continuity and maintains robust fall detection performance under diverse environmental conditions. The proposed framework provides a practical solution for robot-assisted safety monitoring in home environments.

**arXiv ID:** 2606.11109
</details>

<details>
<summary><strong>A Survey of Robotic Navigation and Manipulation with Physics Simulators in the Era of Embodied AI</strong> - Lik Hang Kenny Wong, Xueyang Kang, Kaixin Bai, Jianwei Zhang - [[pdf]](https://arxiv.org/pdf/2505.01458)</summary>

**Abstract:** Navigation and manipulation are core capabilities in Embodied AI, but training agents to perform them directly in the real world is costly, time-consuming, and unsafe. Therefore, sim-to-real transfer has emerged as a key approach, yet the sim-to-real gap persists. This survey examines how physics simulators address this gap by analyzing properties that have received limited attention in prior surveys. We also analyze their features for navigation and manipulation tasks, as well as their hardware requirements. Additionally, we offer a resource with benchmark datasets, metrics, simulation platforms, and methods to help researchers select suitable tools while accounting for hardware constraints.

**arXiv ID:** 2505.01458
</details>

<details>
<summary><strong>CollabSkill: Evaluating Human-Agent Collaboration On Real-World Tasks</strong> - Yijia Shao, Zora Zhiruo Wang, Neel Ahuja, Yicheng Wang, Bowen Liu, Diyi Yang - [[pdf]](https://arxiv.org/pdf/2606.09833)</summary>

**Abstract:** AI agents are reshaping the workspace, leading to drastic change of how humans work. Despite the considerable potential of human-agent collaboration both in preserving human agency and generating economic value, this paradigm remains largely absent from occupational task evaluation, hindered by the difficulty of gathering real human data and accounting for inter-human variability. We introduce CollabSkill, a framework for evaluating human-agent collaboration on real-world occupational tasks. CollabSkill pairs real human workers with AI agents on tasks matched to their occupational background, collecting data that capture the complexity of economically valuable tasks and the usage patterns of real workers. To account for inter-human variability, CollabSkill employs a Bayesian skill rating system to disentangle and quantify the skill contributions of both humans and AI agents. Drawing on over 1,500 prompts from 386 working sessions contributed by 93 human workers, our analysis yields insights on two fronts: on the agent side, rankings on CollabSkill diverge meaningfully from those of existing fully autonomous benchmarks where Codex leads, with Claude Code ranking first; on the human side, CollabSkill reveals that practical experience emerges as the primary driver of collaboration skill, with hands-on collaboration meaningfully shifting workers' AI literacy. Together, we hope CollabSkill enables the community to invest in systematic evaluation of human-agent collaboration and spurs development efforts aimed at building AI agents that genuinely augment human workers.

**arXiv ID:** 2606.09833
</details>

<details>
<summary><strong>Sketch-to-Layout: A Human-Centric Computational Agent for Constraint-Aware Synthesis of Modular Photobioreactors</strong> - Xiujin Liu, Shuqi Li, Yuxin Lin - [[pdf]](https://arxiv.org/pdf/2606.09849)</summary>

**Abstract:** Building-integrated photobioreactors (PBRs) offer a pathway for carbon-neutral architecture, yet deployment is hindered by configuration complexity and biological maintenance. This paper presents a modular PBR facade system powered by a computational framework reconciling design intent with physical validity. We introduce 'carbon-neutralization bricks' featuring integrated vessel-and-conduit geometry; monolithic fluid channels enable 'plug-and-play' assembly. To navigate the combinatorial complexity of 14 modular geometries, we develop a Computational Sketch-to-Layout Agent that formulates layout synthesis as a Constraint Satisfaction Problem (CSP).
Using the CP-SAT engine, the agent treats sparse user sketches as soft priors while enforcing hard constraints like port alignment and global connectivity. This allows non-experts to synthesize fabrication-ready configurations in near real-time. Furthermore, to facilitate autonomous maintenance, we propose a weakly supervised algae health monitoring pipeline. By employing a hybrid CNN-attention backbone and a temporal ranking loss, the system quantifies biological vitality from photographs without absolute ground-truth labels.
Experiments demonstrate the CSP solver achieves a 95.5% success rate on grid scales up to 15 x 15. Qualitative evaluations confirm the framework preserves design semantics while ensuring operational integrity. Long-term tests show the vision module produces health trajectories aligned with 14-day biological cycles, suggesting that integrating interactive synthesis with low-cost computer vision can democratize scalable carbon capture systems.

**arXiv ID:** 2606.09849
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>IntentKV: Cross-Turn Intent-Aware KV Cache Pruning for Agent Inference</strong> - Junjie Li, Jiong Lou, Jie Li - [[pdf]](https://arxiv.org/pdf/2606.09916)</summary>

**Abstract:** Multi-turn LLM agents fan short queries into long trajectories of tool calls, search results, and intermediate reasoning. Both KV memory and KV read bandwidth grow by orders of magnitude across a single trajectory, making the key-value (KV) cache, not parameter compute, the dominant serving bottleneck for long-horizon agents. We introduce IntentKV, learned KV pruning that keeps the base LLM frozen. IntentKV maintains a session-level QueryMemory of cross-turn intent, scores live history tokens with a memory-attention rule, and adds a zero-initialized residual head with cross-attention over current-query K-vectors. To stay composable with prefix caches, eviction is a slot-map redirection: dropped positions route to a sentinel dead slot while surviving K/V rows, RoPE phases, and slot identities stay in place. IntentKV matches the no-pruning full-cache baseline with almost no accuracy drop under tight KV budgets: at an 8k KV budget, mean peak request tokens drop 23.9% on Qwen3-8B and 30.7% on Qwen2.5-14B. On the 100 longest BCP queries that all methods complete on Qwen2.5-14B, IntentKV-8k further cuts worst-case peak request tokens from 92.3k to 20.5k, a 77.8% reduction, and worst-case raw KV reads from 411M to 31M, a 92.6% reduction.

**arXiv ID:** 2606.09916
</details>

<details>
<summary><strong>3SPO: State-Score-Supervised Policy Optimization for LLM Agents</strong> - Yu Han, Kailing Li, Yang Jiao, Yulin Dai, Yuqian Fu, Linhai Zhuo, Tianwen Qian - [[pdf]](https://arxiv.org/pdf/2606.09961)</summary>

**Abstract:** Training large language models (LLMs) as autonomous agents via reinforcement learning (RL) has enabled frontier models to achieve superhuman performance in long-horizon tasks. However, existing RL algorithms operate at the trajectory level, performing policy optimization only after collecting complete episode rollouts. This coarse-grained approach faces fundamental challenges in multi-turn agent settings where rewards are sparse, delayed, and credit assignment across individual steps is critical. In this work, we propose \textbf{State-Score-Supervised Policy Optimization (3SPO)}, a novel RL algorithm that performs post-step policy optimization with dynamic state score supervision. At each step, 3SPO computes the state score based on historical success rates, supervising step-wise credit assignment, adaptive rollout and post-step policy optimization without requiring value function estimation or additional auxiliary models. Theoretically, under a per-state bandit abstraction, we show that the proposed score-supervised allocation mechanism achieves logarithmic allocation regret and provide sample-complexity guarantees for action identification, score distinguishability, and filtering stability. Experiments on ALFWorld and WebShop with Qwen2.5-1.5B/7B-Instruct show that 3SPO consistently outperforms GRPO by $+22.6\%$ on ALFWorld and $+15.6$ points on WebShop, while using comparable resources to achieve $2.4\times$ more state exploration and $1.8\times$ faster convergence. Code is available at this https URL.

**arXiv ID:** 2606.09961
</details>

<details>
<summary><strong>From Confident Closing to Silent Failure: Characterizing False Success in LLM Agents</strong> - Laksh Advani - [[pdf]](https://arxiv.org/pdf/2606.09863)</summary>

**Abstract:** LLM agents can fail silently by asserting task completion when the environment state shows otherwise. We study this failure mode, false success, across two agent benchmarks: 9,876 tau2-bench trajectories from 8 model families and 1,879 AppWorld trajectories from 4 model families with text-independent ground truth. False success is common but varies by setting: 45--48% of failures in single-control tau2-bench domains, 3% in dual-control telecom, and 75.8% among AppWorld self-assessing coding-agent trajectories with explicit status claims. LLM judges fail reliably: no configuration across 5 judges, 5 prompt strategies, and full task specifications exceeds AUROC 0.65 on tau2-bench, and the same judges reach only 0.54 AUROC on AppWorld API-call traces. Judges rely on surface completion proxies -- confident closing language in tau2-bench and coarse action-sequence volume in AppWorld -- rather than verified state changes. Lightweight TF-IDF detectors achieve task-disjoint AUROC 0.83 on tau2-bench and 0.95 on AppWorld, recovering 4--8x more false successes than the best judge at the same flag rate with 3,300x lower latency. These results suggest that production monitoring should use lightweight, domain-calibrated detectors as triage signals rather than relying on LLM judges as the primary monitor for false success.

**arXiv ID:** 2606.09863
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (3 papers)</h2></summary>

<details>
<summary><strong>Agentic Social Affordance Framework (ASAF): Agent Identity Design as a Collaboration Interface in Multi-Agent Systems</strong> - Meng-Han Lee - [[pdf]](https://arxiv.org/pdf/2606.09832)</summary>

**Abstract:** As AI systems evolve from single conversational agents to complex multi-agent architectures, a critical design dimension has been overlooked: how the social identity of individual agents shapes human behavior within the collaboration. This paper introduces the Agentic Social Affordance Framework (ASAF), a theoretical framework that extends Social Affordance theory into the context of multi-agent AI systems. We propose that agent identity design functions not merely as a user interface convention, but as a collaboration interface -- structuring how users perceive, approach, and engage with each agent, and thereby influencing the quality of Human-Agent collaboration outcomes. Specifically, the social affordance layer constitutes an independent design dimension orthogonal to engineering orchestration: the two represent distinct decision spaces that cannot be derived from each other. ASAF comprises three mechanisms: Identity Signaling, Behavioral Priming, and Collaborative Governance, and specifies their boundary conditions through a four-tier Identity Signal Fidelity Spectrum and an individual-difference moderating variable (anthropomorphizing vs.\ instrumentalizing cognitive style). We situate ASAF in relation to existing affordance theory and the CASA paradigm, delineating where ASAF's multi-agent, topology-level predictions exceed the explanatory scope of dyadic frameworks. We discuss implications for multi-agent system design and outline directions for future empirical validation, including a factorial design for testing design-space orthogonality.

**arXiv ID:** 2606.09832
</details>

<details>
<summary><strong>Envisioning Sensemaking in Multi-Human, Multi-Agent Collaborative Knowledge Work</strong> - Zhitong Guan, Soo Young Rieh - [[pdf]](https://arxiv.org/pdf/2606.09840)</summary>

**Abstract:** Sensemaking is central to knowledge work, where people search, evaluate, interpret, and use information over time to construct durable understanding. The rise of generative AI has begun to reshape this process: GenAI systems now perform interpretive functions such as summarization, synthesis, and thematic grouping that knowledge workers have traditionally carried out themselves. In collaborative settings, these shifts compound, complicating how teams divide interpretive labor, trust one another's contributions, and negotiate shared understanding. In this position paper, we examine how GenAI reshapes sensemaking in collaborative knowledge work and propose five design principles for multi-human, multi-agent collaborative sensemaking: dynamic multi-layer information representations, active identification and bridging of gaps in understanding, critical engagement with information, verifiability, and accountability. Building on these principles, we introduce a conceptual framework for a dynamic shared representational workspace in which knowledge workers and specialized AI agents jointly gather evidence, schematize, hypothesize, and pursue collaborative goals. Through a partner agent, a shared space agent, and an orchestrator agent, the framework preserves the provenance and authorship of contributions and traces the evolution of both individual and shared interpretations, supporting coherent, negotiated knowledge construction that current generative AI systems tend to obscure.

**arXiv ID:** 2606.09840
</details>

<details>
<summary><strong>The Interlocutor Effect: Why LLMs Leak More Personal Data to Agents Than Humans</strong> - Faouzi El Yagoubi, Godwin Badu-Marfo, Ranwa Al Mallah - [[pdf]](https://arxiv.org/pdf/2606.09844)</summary>

**Abstract:** Large Language Models (LLMs) alter their privacy behavior based on the perceived identity of their interlocutor. While safety mechanisms typically prevent LLMs from releasing Personally Identifiable Information (PII) to human users, these models tend to reveal more sensitive data when addressing another AI agent.
We refer to this as the \textbf{Interlocutor Effect}. Through an ablation study, we find evidence that the technical nature of the recipient contributes to this effect, thereby diminishing the model's caution regarding privacy. To explore this further, we introduce the Attention Suppression Hypothesis, which posits that safety-aligned attention heads become inactive during interactions with agents. We assess this quantitatively by comparing human-directed and agent-directed prompts in 222 sensitive scenarios. Our findings, drawn from 3,464 interactions, indicate that portraying the recipient as an AI agent elevates PII leakage by up to 23 percentage points. Initial experiments on Llama-3.1-8B-Instruct corroborate this: deactivating one safety head induces leakage, whereas reactivating it reinstates privacy safeguards. We consider the implications for developing secure multi-agent systems.

**arXiv ID:** 2606.09844
</details>

</details>

<details open>
<summary><h2>Other Agent Research (3 papers)</h2></summary>

<details>
<summary><strong>Pushing the Performance Limits in Autonomous Racing: Continuous Stability-Aware Adaptive Velocity Planning in Formula Student Driverless</strong> - Tamara Bergerhoff, Sebastian Baader, Pascal Meißner, Frank Deinzer - [[pdf]](https://arxiv.org/pdf/2606.10733)</summary>

**Abstract:** In autonomous racing, especially in competitions such as Formula Student Driverless, precise planning of the target velocity of a race car is crucial for competitive lap times and stable driving behavior. Especially at high speeds, Velocity Planning (VP) is a significant challenge as it has to be performed in real time, taking into account track layouts, environmental influences, mechanical tolerances, and the resulting control inaccuracies. In this paper, we present a novel approach to VP that dynamically adapts to such changing conditions. Instead of estimating the physical Tire-Road Friction Coefficient (TRFC), a continuous scaling factor is inferred indirectly from vehicle stability. This factor not only reflects the effective tire-road interaction but also captures effects of control inaccuracies. From this, we generate a continuous friction map, which serves as a robust, adaptive basis for computing the optimal target speed, accounting for both vehicle and environmental limits. Our proposed approach was evaluated on a real Formula Student race car, showing a lap time improvement of 35 % over ten laps and an average increase of 8 % compared to a non-adaptive approach.

**arXiv ID:** 2606.10733
</details>

<details>
<summary><strong>Resilient Navigation for Autonomous Farm Robots by Leveraging Jerk-Augmented Models with IMU-Only Disturbance Rejection</strong> - Batu Candan, Mohammed Atallah, Simone Servadio, Saeed Arabi - [[pdf]](https://arxiv.org/pdf/2606.10971)</summary>

**Abstract:** Precise state estimation for navigation of autonomous agricultural robots is often compromised by sensor outages (GNSS/LiDAR/Visual) and high-frequency vibrations inherent in off-road environments. This paper proposes a robust navigation algorithm based on a jerk-augmented Extended Kalman Filter (EKF) integrated with a Multiple Tuning Factor (MTF) adaptation method. Unlike standard EKF approaches that assume constant measurement noise, our method dynamically adjusts the measurement covariance matrix in real-time, allowing the system to cope with sudden disturbances and sensor outliers. We evaluate the algorithm using real-world data from a Salin247 autonomous robot. Results demonstrate that jerk-augmentation combined with MTF adaptation significantly reduces 3D position Root Mean Square Error (RMSE) compared to baseline EKF models, providing superior dead-reckoning capabilities.

**arXiv ID:** 2606.10971
</details>

<details>
<summary><strong>Language-Driven Cost Optimization for Autonomous Driving</strong> - Diego Martinez-Baselga, Khaled Mustafa, Javier Alonso-Mora - [[pdf]](https://arxiv.org/pdf/2606.10974)</summary>

**Abstract:** The driving behavior of autonomous vehicles is typically governed by the cost function of their motion planner, which encodes objectives such as speed tracking, smoothness, lane keeping, and collision avoidance. However, tuning the parameters that shape this cost function is a challenging task that requires technical expertise, limiting the vehicle's ability to adapt to evolving traffic scenarios or end-user preferences. This work presents a language-driven framework for adaptive cost design in autonomous driving. A Large Language Model (LLM) interprets structured scenario descriptions and natural language user queries to generate the parameters applied to a risk-aware Model Predictive Path Integral (MPPI) controller. The system incorporates a human-in-the-loop validation stage in which the proposed behavioral changes are described in non-technical language and confirmed prior to deployment. Users may additionally provide feedback either before or after deployment, enabling iterative refinement of the vehicle's motion behavior. The framework is evaluated across multiple queries in realistic driving scenarios to assess its effectiveness. Simulation results demonstrate that the method successfully induces behavioral changes that align with the intended requirements in an intuitive manner, thereby bridging the gap between intelligent vehicle control systems and end users.

**arXiv ID:** 2606.10974
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (9 papers)</h2></summary>

<details>
<summary><strong>Discovering Interpretable Multi-Parameter Control Policies for Evolutionary Algorithms Using Deep Reinforcement Learning</strong> - Tai Nguyen, Phong Le, Carola Doerr, Nguyen Dang - [[pdf]](https://arxiv.org/pdf/2606.10129)</summary>

**Abstract:** While deep Reinforcement Learning (deep-RL) has been increasingly applied to parameter control in evolutionary algorithms, rigorous theoretical analysis of parameter control remains largely restricted to single-parameter settings, owing to the difficulty of deriving effective, interpretable multi-parameter policies amenable to formal study. We demonstrate how deep-RL can be leveraged to overcome this barrier, using the (1+($\lambda$,$\lambda$))-genetic algorithm optimizing OneMax, one of the few problems where a super-constant speedup of dynamic control has been formally proven, as a representative case study. We first show that standard approaches struggle to converge in this multi-parameter setting, and introduce algorithm-agnostic enhancements targeting action-space decomposition, reward shifting, and long-horizon discounting. With these in place, we compare common deep-RL methods and find that Double Deep Q-Networks uniquely avoid the policy collapse observed in Proximal Policy Optimization, yielding trajectories suitable for downstream analysis. Crucially, we move beyond the ``black-box'' nature of neural networks by distilling the learned behaviors into a transparent, symbolic control policy. This resulting policy does not only offer interpretability for future theoretical analysis but also yields exceptional performance, consistently outperforming existing baselines across a wide range of problem sizes.

**arXiv ID:** 2606.10129
</details>

<details>
<summary><strong>Uncertainty-Aware Motion Planning for Autonomous Driving in Mixed Traffic Environment</strong> - Ming Cheng, Hao Chen, Ziyi Yang, Ziluowen Luo, Senzhang Wang - [[pdf]](https://arxiv.org/pdf/2606.09958)</summary>

**Abstract:** In mixed-traffic environments where autonomous and human-driven vehicles may co-exist, motion planning for autonomous vehicles requires anticipating the future behaviors of surrounding human drivers. Existing reinforcement learning-based methods generally directly incorporate the predicted human intents into the observation to enable a proactive planning. However, human intent is inherently uncertain due to the behavioral diversity, perception noise, and partial observability. Treating predicted intends as deterministic states can result in unsafe decisions for autonomous vehicles. To address this problem, we propose Uncertainty-Aware Motion Planning (UAMP), which incorporates uncertainty in human intent prediction for AV decision-making. Specifically, UAMP first introduces a proximity-aware uncertainty estimator to quantify the interaction-conditioned intent uncertainty and constructs an uncertainty-guided joint intent distribution over surrounding human-driven vehicles. Within this uncertainty set, UAMP further introduces Uncertainty-Calibrated Value Learning (UCVL) to correct value function learning biases arising from directly incorporating uncertain human intent predictions into the observation. Extensive experiments in various mixed-traffic scenarios show that UAMP significantly improves safety and driving comfort, while maintaining traffic efficiency compared with existing approaches. The code is released at this https URL.

**arXiv ID:** 2606.09958
</details>

<details>
<summary><strong>MARCH: Model-Assisted Reinforcement Learning for the Perceptive Control of Humanoids over Sparse Footholds</strong> - Codrin Crismariu, Ryan K. Cosner - [[pdf]](https://arxiv.org/pdf/2606.10288)</summary>

**Abstract:** Perceptive bipedal locomotion over sparse terrain remains a difficult challenge: model-based methods are precise but brittle to uncertainty, while model-free methods are robust but struggle to discover the precise, constrained motions required for safety-critical locomotion where small errors can cause catastrophic failures. We propose a model-assisted reinforcement learning (RL) framework that combines both perspectives in three steps: (1) generate a safe reference trajectory using simplified models; (2) train a privileged teacher policy guided by a control Lyapunov function (CLF) reward built around the safe reference trajectory; and (3) distill the teacher into a vision-based student policy. We show that this model-assistance procedure produces physically grounded locomotion, improving sample efficiency, reducing the need for a complex learning curriculum, and achieving smoother locomotion behavior alongside stepping stone performance comparable to model-free baselines. We validate our approach in simulation and demonstrate successful deployment on a Unitree G1 humanoid robot navigating sparse footholds with lateral constraints.

**arXiv ID:** 2606.10288
</details>

<details>
<summary><strong>GuideWalk: Learning Unified Autonomous Navigation and Locomotion for Humanoid Robots across Versatile Terrains</strong> - Haoxuan Han, Chen Chen, Linao Gong, Xin Yang, Hao Hu, Junhong Guo, Zhicheng He, Yao Su, Fenghua He - [[pdf]](https://arxiv.org/pdf/2606.10449)</summary>

**Abstract:** Humanoid robots have achieved strong locomotion capabilities, but reliable navigation on versatile terrains remains challenging because obstacle avoidance must be coordinated with dynamically feasible motion. In this work, we present GuideWalk, a unified end-to-end framework that integrates traversability-aware navigation guidance with terrain-adaptive locomotion teacher for humanoid navigation. Specifically, we introduce a navigation module that provides explicit velocity guidance, decoupling obstacle avoidance from terrain conditions to enable robust planning across diverse environments. We propose a composite teacher distillation scheme, where goal-directed commands and dynamically consistent actions are aggregated and distilled into a single policy. To further improve robustness, the distilled policy is refined with reinforcement learning and an auxiliary behavior cloning objective, which promotes exploration while preserving desirable teacher behaviors. Experiments demonstrate that GuideWalk achieves stable and effective navigation while maintaining stable humanoid locomotion.

**arXiv ID:** 2606.10449
</details>

<details>
<summary><strong>An Exposure-Time-Aligned Primary-Path Architecture for Autonomous-Driving ECUs</strong> - Toru Saito, Yuki Hagura, Tatsuya Konishi, Satoru Mizusawa, Takumi Yajima - [[pdf]](https://arxiv.org/pdf/2606.10856)</summary>

**Abstract:** While end-to-end (E2E) autonomous driving has become the dominant research direction, production vehicles continue to rely on modular multi-NN pipelines for a non-trivial transitional period. The subject of this paper is the design of an architecture that, during this phase, supports a modular pipeline and an E2E path side by side and embeds a path for staged migration. Transplanted to a production SoC, egalitarian late fusion is compute-inefficient and offers no natural unit for staged E2E substitution. As an alternative, we propose three design principles: (i) Primary-Path, which explicitly selects a primary perception chain and prioritizes its enclosure within a single SoC pair over the non-critical paths (ii) Exposure-Time-Aligned, which propagates the primary sensor's exposure time $\tau_{\rm exp}$ as a tag along the chain and event-drives the fusion node on matched $\tau_{\rm exp}$ rather than a fixed cycle and (iii) Co-Path Coexistence, which, building on (i) and (ii), lets an E2E output path co-run with the modular pipeline within the same $\tau_{\rm exp}$ cycle. On a Dual-SoC production AD-ECU, the implementation closes camera-shutter to planner-output latency at a mean of 296 ms within the 350 ms design budget. Under (iii), the modular pipeline is primary at production launch and the E2E path runs as shadow on real vehicles, and the E2E scope is expanded as evaluation evidence accumulates.

**arXiv ID:** 2606.10856
</details>

<details>
<summary><strong>AllDayNav: Lifelong Navigation via Real-World Reinforcement Learning</strong> - Hang Yin, Yinan Liang, Jiazhao Zhang, Jiahang Liu, Minghan Li, Zhizheng Zhang, He Wang - [[pdf]](https://arxiv.org/pdf/2606.10927)</summary>

**Abstract:** Lifelong embodied navigation in dynamic environments requires robots to form persistent scene understanding from fragmentary observations, which remains difficult for existing methods that rely on explicit maps or scene graphs and struggle to generalize beyond structured settings. We propose AllDayNav, a lifelong self-learning navigation framework that implicitly encodes scene dynamics into the billion-scale parameters of a large model via reinforcement learning, powered by a self-evolving multimodal memory that maintains and updates visual keyframes, semantic descriptions, and temporal context while autonomously generating open-vocabulary instructions, image goals, and structured rewards. Experiments in both synthetic and real-world environments across cross-room, cross-episode, and cross-task scenarios show that AllDayNav achieves success rates approaching $100\%$ and consistently surpasses strong map-based, VLM, and RL baselines in path efficiency and robustness, demonstrating implicit, memory-driven reinforcement learning as a scalable alternative to explicit mapping for reliable lifelong navigation.

**arXiv ID:** 2606.10927
</details>

<details>
<summary><strong>RoboNaldo: Accurate, Stable and Powerful Humanoid Soccer Shooting via Motion-Guided Curriculum Reinforcement Learning</strong> - Yichao Zhong, Yidan Lu, Yuhang Lu, Tianyang Tang, Haoguang Mai, Yixuan Pan, Tianyu Li, Li Chen, Jingbo Wang, Zhongyu Li, Peng Lu, Hongyang Li - [[pdf]](https://arxiv.org/pdf/2606.11092)</summary>

**Abstract:** Elite humanoid soccer shooting requires whole-body stability, high-impulse whole-body interactions, and accuracy to targets. Motion tracking-driven reinforcement learning (RL) provides stability in whole-body movement coordination, but a fixed reference makes it hard to adapt to varied ball positions and strike timings; in contrast, task reward-driven RL struggles to explore and discover valid kicks from scratch. We therefore introduce RoboNaldo, a three-stage motion-guided curriculum RL framework for high-impulse humanoid interaction. A single human-kick reference is used as a scaffold and progressively shifts optimization towards shooting performance. The curriculum first learns a stable whole-body kicking prior, then adapts the kick to free-kick settings where the ball is stationary at random positions, and finally extends it to moving-ball shooting through a locomotion-command and kick-trigger interface. A high-level heuristic planner controls this interface during training, while alternative high-level controllers can drive the same low-level policy at inference. In simulation, RoboNaldo demonstrates free-kick shot error 48.6% lower and shoot velocity 2.96x than prior work baselines. In real world on a Unitree G1 with onboard perception, RoboNaldo attains 0.73 m and 0.86 m average target shooting error from 3 m away in free-kick and moving-ball cases, accordingly. And the post-contact ball velocity reaches 13.10 m/s, which is 59-71% of reported professional open-play shot speed. Project page: $\href{this https URL}{\text{this http URL}}$.

**arXiv ID:** 2606.11092
</details>

<details>
<summary><strong>On-sky demonstration of reinforcement learning for adaptive optics control</strong> - Jalo Nousiainen, Vincent Chambouleyron, Benoit Neichel, Sylvain Cetre, Jean-Francois Sauvage, Angelie Alagao, Markus Kasper, Jonathan Dray, Romain Fetick, Byron Engler - [[pdf]](https://arxiv.org/pdf/2606.10771)</summary>

**Abstract:** Reinforcement learning (RL)-based algorithms have recently emerged as a promising approach for adaptive optics (AO) control. In simulations and laboratory experiments, they have demonstrated robustness to real-world effects such as photon and detector noise, misregistration, vibrations, and rapid variations in seeing conditions. However, their performance has not yet been validated on sky. We report the first on-sky demonstration of a reinforcement learning controller for adaptive optics, named Policy Optimization for AO (PO4AO). We further analyze its on-sky behavior and identify directions for improving the algorithm and its implementation.PO4AO was implemented and deployed on the Papyrus adaptive optics system installed at the Coudé focus of the 1.52 m telescope (T152) at the OHP. A Python-based implementation was interfaced with the existing real-time controller (DAO RTC) via shared-memory buffers. The performance of PO4AO was compared to that of a standard integrator controller over several nights, covering a range of flux levels and atmospheric conditions. PO4AO consistently outperformed the standard integrator in all tested configurations. The controller successfully learned and compensated for vibration patterns and demonstrated strong robustness to measurement noise. Once tuned for Papyrus, PO4AO operated in a turnkey fashion, using a single set of hyperparameters across varying observing conditions and science targets. These performance gains were achieved despite a non-optimized Python implementation introducing approximately $750\,\mu\text{s}$ of additional latency, along with control jitter and occasional frame drops. When properly implemented and optimized, PO4AO constitutes a robust and high-performance turnkey controller for single-conjugate adaptive optics systems, paving the way for broader adoption of reinforcement learning strategies in on-sky AO operations.

**arXiv ID:** 2606.10771
</details>

<details>
<summary><strong>Human-AI Coordination Zones: A Framework for Designing Human-in-the-Loop Experiences with Agentic AI</strong> - James Pierce, Vaiva Kalnikaitė, Siddharth Gupta, Brian Granger - [[pdf]](https://arxiv.org/pdf/2606.09848)</summary>

**Abstract:** As generative and agentic AI becomes embedded in everyday products, practitioners face a persistent challenge: how to design human-AI coordination -- the ongoing mutual adjustment between users and AI systems as mediate through interfaces-that supports usability, trust, and safety. Existing resources offer high-level principles ("be transparent," "maintain user control") or low-level UI patterns, but there is a lack of mid-level design knowledge bridging the two. Through landscape and artifact analysis of 60 commercial AI applications, we introduce a framework defining human-AI coordination as the interplay of three dimensions: salience (how prominently AI is presented), involvement (what users can do to engage AI), and activity (what AI actually does). We contribute mid-level tools including coordination zones (done-for-me, done-under-me, done-with-me, done-without-me), an input taxonomy (prompted, sparked, inferred, layered), coordination curves for mapping user journeys, and design patterns demonstrating the generative capacity of the framework. The framework can be applied generatively to design experiences, analytically to evaluate existing ones, and communicatively to articulate ideas across stakeholders.

**arXiv ID:** 2606.09848
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
