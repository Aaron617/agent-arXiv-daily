# Agent arXiv Daily

**Last Updated:** 2026-03-04 02:49:43

**Total Papers:** 86

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
<summary><strong>ManagerBench: Evaluating the Safety-Pragmatism Trade-off in Autonomous LLMs</strong> - Adi Simhi, Jonathan Herzig, Martin Tutek, Itay Itzhak, Idan Szpektor, Yonatan Belinkov - [[pdf]](https://arxiv.org/pdf/2510.00857)</summary>

**Abstract:** As large language models (LLMs) evolve from conversational assistants into autonomous agents, evaluating the safety of their actions becomes critical. Prior safety benchmarks have primarily focused on preventing generation of harmful content, such as toxic text. However, they overlook the challenge of agents taking harmful actions when the most effective path to an operational goal conflicts with human safety. To address this gap, we introduce ManagerBench, a benchmark that evaluates LLM decision-making in realistic, human-validated managerial scenarios. Each scenario forces a choice between a pragmatic but harmful action that achieves an operational goal, and a safe action that leads to worse operational performance. A parallel control set, where potential harm is directed only at inanimate objects, measures a model's pragmatism and identifies its tendency to be overly safe. Our findings indicate that the frontier LLMs perform poorly when navigating this safety-pragmatism trade-off. Many consistently choose harmful options to advance their operational goals, while others avoid harm only to become overly safe and ineffective. Critically, we find this misalignment does not stem from an inability to perceive harm, as models' harm assessments align with human judgments, but from flawed prioritization. ManagerBench is a challenging benchmark for a core component of agentic behavior: making safe choices when operational goals and alignment values incentivize conflicting actions. Benchmark & code available at this https URL.

**arXiv ID:** 2510.00857
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (14 papers)</h2></summary>

<details>
<summary><strong>Engineering Reasoning and Instruction (ERI) Benchmark: A Large Taxonomy-driven Dataset for Foundation Models and Agents</strong> - MZ Naser, Ahmad Bani Awwad, Zoie McCreery, Radwa Eissa, Ahmad Naser, Gianluca Cusatis, Andrew Metcalf, Kapil Madathil, Jamal Abdalla, Venkatesh Kodur, Mohammad Reza Saeb - [[pdf]](https://arxiv.org/pdf/2603.02239)</summary>

**Abstract:** The Engineering Reasoning and Instruction (ERI) benchmark is a taxonomy-driven instruction dataset designed to train and evaluate engineering-capable large language models (LLMs) and agents. This dataset spans nine engineering fields (namely: civil, mechanical, electrical, chemical, environmental, aerospace, materials, fire, and industrial engineering) and 55 subdomains, and is crossed with seven intent types (i.e., definition, explanation, calculation, comparison, design/synthesis, troubleshooting, and code-related) and three difficulty tiers (undergraduate, graduate, and professional), yielding 57,750 records with field/subdomain/type/difficulty metadata and solution formatting. We examined ERI via seven LLMs and report a statistically significant three-tier performance structure, with frontier models (GPT-5, Claude Sonnet 4, DeepSeek V3.1) achieving mean scores above 4.30 on a five-point scale, while mid-tier and smaller models exhibited progressively higher failure rates and steeper performance degradation on graduate-level questions. To address circularity concerns inherent in LLM benchmarks, we developed a convergent validation protocol that leverages cross-provider independence, multi-judge averaging, and frontier-model agreement analysis to empirically bound hallucination risk to 1.7%. ERI is released with taxonomy specifications, validation scripts, and an evaluation harness to enable reproducible comparisons and regression testing for instruction tuning, routing, retrieval-augmented evaluation, and agentic tool-use workflows in engineering settings.

**arXiv ID:** 2603.02239
</details>

<details>
<summary><strong>LLM-MLFFN: Multi-Level Autonomous Driving Behavior Feature Fusion via Large Language Model</strong> - Xiangyu Li, Tianyi Wang, Xi Cheng, Rakesh Chowdary Machineni, Zhaomiao Guo, Sikai Chen, Junfeng Jiao, Christian Claudel - [[pdf]](https://arxiv.org/pdf/2603.02528)</summary>

**Abstract:** Accurate classification of autonomous vehicle (AV) driving behaviors is critical for safety validation, performance diagnosis, and traffic integration analysis. However, existing approaches primarily rely on numerical time-series modeling and often lack semantic abstraction, limiting interpretability and robustness in complex traffic environments. This paper presents LLM-MLFFN, a novel large language model (LLM)-enhanced multi-level feature fusion network designed to address the complexities of multi-dimensional driving data. The proposed LLM-MLFFN framework integrates priors from largescale pre-trained models and employs a multi-level approach to enhance classification accuracy. LLM-MLFFN comprises three core components: (1) a multi-level feature extraction module that extracts statistical, behavioral, and dynamic features to capture the quantitative aspects of driving behaviors; (2) a semantic description module that leverages LLMs to transform raw data into high-level semantic features; and (3) a dual-channel multi-level feature fusion network that combines numerical and semantic features using weighted attention mechanisms to improve robustness and prediction accuracy. Evaluation on the Waymo open trajectory dataset demonstrates the superior performance of the proposed LLM-MLFFN, achieving a classification accuracy of over 94%, surpassing existing machine learning models. Ablation studies further validate the critical contributions of multi-level fusion, feature extraction strategies, and LLM-derived semantic reasoning. These results suggest that integrating structured feature modeling with language-driven semantic abstraction provides a principled and interpretable pathway for robust autonomous driving behavior classification.

**arXiv ID:** 2603.02528
</details>

<details>
<summary><strong>LiveAgentBench: Comprehensive Benchmarking of Agentic Systems Across 104 Real-World Challenges</strong> - Hao Li, Huan Wang, Jinjie Gu, Wenjie Wang, Chenyi Zhuang, Sikang Bian - [[pdf]](https://arxiv.org/pdf/2603.02586)</summary>

**Abstract:** As large language models grow more capable, general AI agents have become increasingly prevalent in practical applications. However, existing benchmarks face significant limitations, failing to represent real-world user tasks accurately. To address this gap, we present LiveAgentBench, a comprehensive benchmark with 104 scenarios that reflect real user requirements. It is constructed from publicly sourced questions on social media and real-world products. Central to our approach is the Social Perception-Driven Data Generation (SPDG) method, a novel process we developed to ensure each question's real-world relevance, task complexity, and result verifiability. We evaluate various models, frameworks, and commercial products using LiveAgentBench, revealing their practical performance and identifying areas for improvement. This release includes 374 tasks, with 125 for validation and 249 for testing. The SPDG process enables continuous updates with fresh queries from real-world interactions.

**arXiv ID:** 2603.02586
</details>

<details>
<summary><strong>See and Remember: A Multimodal Agent for Web Traversal</strong> - Xinjun Wang, Shengyao Wang, Aimin Zhou, Hao Hao - [[pdf]](https://arxiv.org/pdf/2603.02626)</summary>

**Abstract:** Autonomous web navigation requires agents to perceive complex visual environments and maintain long-term context, yet current Large Language Model (LLM) based agents often struggle with spatial disorientation and navigation loops. In this paper, we propose generally applicable V-GEMS(Visual Grounding and Explicit Memory System), a robust multimodal agent architecture designed for precise and resilient web traversal. Our agent integrates visual grounding to resolve ambiguous interactive elements and introduces an explicit memory stack with state tracking. This dual mechanism allows the agent to maintain a structured map of its traversal path, enabling valid backtracking and preventing cyclical failures in deep navigation tasks. We also introduce an updatable dynamic benchmark to rigorously evaluate adaptability. Experiments show V-GEMS significantly dominates the WebWalker baseline, achieving a substantial 28.7% performance gain. Code is available at this https URL.

**arXiv ID:** 2603.02626
</details>

<details>
<summary><strong>Agentified Assessment of Logical Reasoning Agents</strong> - Zhiyu Ni, Yifeng Xiao, Zheng Liang - [[pdf]](https://arxiv.org/pdf/2603.02788)</summary>

**Abstract:** We present a framework for evaluating and benchmarking logical reasoning agents when assessment itself must be reproducible, auditable, and robust to execution failures. Building on agentified assessment, we use an assessor agent to issue tasks, enforce execution budgets, parse outputs, and record structured failure types, while the agent under test only needs to expose a standardized agent-to-agent interface. As a case study, we benchmark an auto-formalization agent for first-order logic (FOL) reasoning on a solver-verified and repaired split of FOLIO. The agent translates natural language premises and conclusions into executable Z3Py programs and employs satisfiability modulo theories (SMT) solving to determine logical entailment. On the cleaned FOLIO validation set, the auto-formalization agent achieves 86.70% accuracy under the assessor protocol, outperforming a chain-of-thought baseline (73.89%).

**arXiv ID:** 2603.02788
</details>

<details>
<summary><strong>Guideline-Grounded Evidence Accumulation for High-Stakes Agent Verification</strong> - Yichi Zhang, Nabeel Seedat, Yinpeng Dong, Peng Cui, Jun Zhu, Mihaela van de Schaar - [[pdf]](https://arxiv.org/pdf/2603.02798)</summary>

**Abstract:** As LLM-powered agents have been used for high-stakes decision-making, such as clinical diagnosis, it becomes critical to develop reliable verification of their decisions to facilitate trustworthy deployment. Yet, existing verifiers usually underperform owing to a lack of domain knowledge and limited calibration. To address this, we establish GLEAN, an agent verification framework with Guideline-grounded Evidence Accumulation that compiles expert-curated protocols into trajectory-informed, well-calibrated correctness signals. GLEAN evaluates the step-wise alignment with domain guidelines and aggregates multi-guideline ratings into surrogate features, which are accumulated along the trajectory and calibrated into correctness probabilities using Bayesian logistic regression. Moreover, the estimated uncertainty triggers active verification, which selectively collects additional evidence for uncertain cases via expanding guideline coverage and performing differential checks. We empirically validate GLEAN with agentic clinical diagnosis across three diseases from the MIMIC-IV dataset, surpassing the best baseline by 12% in AUROC and 50% in Brier score reduction, which confirms the effectiveness in both discrimination and calibration. In addition, the expert study with clinicians recognizes GLEAN's utility in practice.

**arXiv ID:** 2603.02798
</details>

<details>
<summary><strong>What Capable Agents Must Know: Selection Theorems for Robust Decision-Making under Uncertainty</strong> - Aran Nayebi - [[pdf]](https://arxiv.org/pdf/2603.02491)</summary>

**Abstract:** As artificial agents become increasingly capable, what internal structure is *necessary* for an agent to act competently under uncertainty? Classical results show that optimal control can be *implemented* using belief states or world models, but not that such representations are required. We prove quantitative "selection theorems" showing that low *average-case regret* on structured families of action-conditioned prediction tasks forces an agent to implement a predictive, structured internal state. Our results cover stochastic policies, partial observability, and evaluation under task distributions, without assuming optimality, determinism, or access to an explicit model. Technically, we reduce predictive modeling to binary "betting" decisions and show that regret bounds limit probability mass on suboptimal bets, enforcing the predictive distinctions needed to separate high-margin outcomes. In fully observed settings, this yields approximate recovery of the interventional transition kernel; under partial observability, it implies necessity of belief-like memory and predictive state, addressing an open question in prior world-model recovery work.

**arXiv ID:** 2603.02491
</details>

<details>
<summary><strong>Tether: Autonomous Functional Play with Correspondence-Driven Trajectory Warping</strong> - William Liang, Sam Wang, Hung-Ju Wang, Osbert Bastani, Yecheng Jason Ma, Dinesh Jayaraman - [[pdf]](https://arxiv.org/pdf/2603.03278)</summary>

**Abstract:** The ability to conduct and learn from interaction and experience is a central challenge in robotics, offering a scalable alternative to labor-intensive human demonstrations. However, realizing such "play" requires (1) a policy robust to diverse, potentially out-of-distribution environment states, and (2) a procedure that continuously produces useful robot experience. To address these challenges, we introduce Tether, a method for autonomous functional play involving structured, task-directed interactions. First, we design a novel open-loop policy that warps actions from a small set of source demonstrations (<=10) by anchoring them to semantic keypoint correspondences in the target scene. We show that this design is extremely data-efficient and robust even under significant spatial and semantic variations. Second, we deploy this policy for autonomous functional play in the real world via a continuous cycle of task selection, execution, evaluation, and improvement, guided by the visual understanding capabilities of vision-language models. This procedure generates diverse, high-quality datasets with minimal human intervention. In a household-like multi-object setup, our method is the first to perform many hours of autonomous multi-task play in the real world starting from only a handful of demonstrations. This produces a stream of data that consistently improves the performance of closed-loop imitation policies over time, ultimately yielding over 1000 expert-level trajectories and training policies competitive with those learned from human-collected demonstrations.

**arXiv ID:** 2603.03278
</details>

<details>
<summary><strong>Efficient Agent Training for Computer Use</strong> - Yanheng He, Jiahe Jin, Pengfei Liu - [[pdf]](https://arxiv.org/pdf/2505.13909)</summary>

**Abstract:** Scaling up high-quality trajectory data has long been a critical bottleneck for developing human-like computer use agents. We introduce PC Agent-E, an efficient agent training framework that significantly reduces reliance on large-scale human demonstrations. Starting with just 312 human-annotated computer use trajectories, we further augment them by synthesizing diverse alternative action decisions with Claude 3.7 Sonnet. Trained on these enriched trajectories, our PC Agent-E model achieved a remarkable 141 relative improvement, and even surpassed the Claude 3.7 Sonnet by 10% in relative terms on WindowsAgentArena-V2, an improved benchmark we also released. By integrating robust human computer use skills with automated AI data synthesis capabilities, our method not only brought substantial improvements over training on human trajectories alone, but also significantly surpassed direct distillation from Claude 3.7 Sonnet. Code, data and models are available at this https URL

**arXiv ID:** 2505.13909
</details>

<details>
<summary><strong>See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles</strong> - Zongru Wu, Rui Mao, Zhiyuan Tian, Pengzhou Cheng, Tianjie Ju, Zheng Wu, Lingzhong Dong, Haiyue Sheng, Zhuosheng Zhang, Gongshen Liu - [[pdf]](https://arxiv.org/pdf/2509.13615)</summary>

**Abstract:** The advent of multimodal agents facilitates effective interaction within graphical user interface (GUI), especially in ubiquitous GUI control. However, their inability to reliably execute toggle control instructions remains a key bottleneck. To investigate this, we construct a state control benchmark with binary toggle instructions derived from public datasets. Evaluation results of existing agents demonstrate their notable unreliability, particularly when the current toggle state already matches the desired state. To address the challenge, we propose State-aware Reasoning (StaR), a multimodal reasoning method that enables agents to perceive the current toggle state, infer the desired state from the instruction, and act accordingly. Experiments on four multimodal agents demonstrate that StaR can improve toggle instruction execution accuracy by over 30\%. Further evaluations on three public agentic benchmarks show that StaR also enhances general agentic task performance. Finally, evaluations on a dynamic environment highlight the potential of StaR for real-world applications. Code and benchmark: this https URL.

**arXiv ID:** 2509.13615
</details>

<details>
<summary><strong>BeyondSWE: Can Current Code Agent Survive Beyond Single-Repo Bug Fixing?</strong> - Guoxin Chen, Fanzhe Meng, Jiale Zhao, Minghao Li, Daixuan Cheng, Huatong Song, Jie Chen, Yuzhi Lin, Hui Chen, Xin Zhao, Ruihua Song, Chang Liu, Cheng Chen, Kai Jia, Ji-Rong Wen - [[pdf]](https://arxiv.org/pdf/2603.03194)</summary>

**Abstract:** Current benchmarks for code agents primarily assess narrow, repository-specific fixes, overlooking critical real-world challenges such as cross-repository reasoning, domain-specialized problem solving, dependency-driven migration, and full-repository generation. To address this gap, we introduce BeyondSWE, a comprehensive benchmark that broadens existing evaluations along two axes - resolution scope and knowledge scope - using 500 real-world instances across four distinct settings. Experimental results reveal a significant capability gap: even frontier models plateau below 45% success, and no single model performs consistently across task types. To systematically investigate the role of external knowledge, we develop SearchSWE, a framework that integrates deep search with coding abilities. Our experiments show that search augmentation yields inconsistent gains and can in some cases degrade performance, highlighting the difficulty of emulating developer-like workflows that interleave search and reasoning during coding tasks. This work offers both a realistic, challenging evaluation benchmark and a flexible framework to advance research toward more capable code agents.

**arXiv ID:** 2603.03194
</details>

<details>
<summary><strong>Go-Browse: Training Web Agents with Structured Exploration</strong> - Apurva Gandhi, Graham Neubig - [[pdf]](https://arxiv.org/pdf/2506.03533)</summary>

**Abstract:** One of the fundamental problems in digital agents is their lack of understanding of their environment. For instance, a web browsing agent may get lost in unfamiliar websites, uncertain what pages must be visited to achieve its goals. To address this, we propose Go-Browse, a method for automatically collecting diverse and realistic web agent data at scale through structured exploration of web environments. Go-Browse achieves efficient exploration by framing data collection as a graph search, enabling reuse of information across exploration episodes. We instantiate our method on the WebArena benchmark, collecting a dataset of 10K successful task-solving trajectories and 40K interaction steps across 100 URLs. Fine-tuning a 7B parameter language model on this dataset achieves a success rate of 21.7% on the WebArena benchmark, beating GPT-4o mini by 2.4% and exceeding current state-of-the-art results for sub-10B parameter models by 2.9%.

**arXiv ID:** 2506.03533
</details>

<details>
<summary><strong>ParEVO: Synthesizing Code for Irregular Data: High-Performance Parallelism through Agentic Evolution</strong> - Liu Yang, Zeyu Nie, Andrew Liu, Felix Zou, Deniz Altinbüken, Amir Yazdanbakhsh, Quanquan C. Liu - [[pdf]](https://arxiv.org/pdf/2603.02510)</summary>

**Abstract:** The transition from sequential to parallel computing is essential for modern high-performance applications but is hindered by the steep learning curve of concurrent programming. This challenge is magnified for irregular data structures (such as sparse graphs, unbalanced trees, and non-uniform meshes) where static scheduling fails and data dependencies are unpredictable. Current Large Language Models (LLMs) often fail catastrophically on these tasks, generating code plagued by subtle race conditions, deadlocks, and sub-optimal scaling.
We bridge this gap with ParEVO, a framework designed to synthesize high-performance parallel algorithms for irregular data. Our contributions include: (1) The Parlay-Instruct Corpus, a curated dataset of 13,820 tasks synthesized via a "Critic-Refine" pipeline that explicitly filters for empirically performant algorithms that effectively utilize Work-Span parallel primitives; (2) specialized DeepSeek, Qwen, and Gemini models fine-tuned to align probabilistic generation with the rigorous semantics of the ParlayLib library; and (3) an Evolutionary Coding Agent (ECA) that improves the "last mile" of correctness by iteratively repairing code using feedback from compilers, dynamic race detectors, and performance profilers.
On the ParEval benchmark, ParEVO achieves an average 106x speedup (with a maximum of 1103x) across the suite, and a robust 13.6x speedup specifically on complex irregular graph problems, outperforming state-of-the-art commercial models. Furthermore, our evolutionary approach matches state-of-the-art expert human baselines, achieving up to a 4.1x speedup on specific highly-irregular kernels. Source code and datasets are available at this https URL.

**arXiv ID:** 2603.02510
</details>

<details>
<summary><strong>ConEQsA: Concurrent and Asynchronous Embodied Questions Scheduling and Answering</strong> - Haisheng Wang, Dong Liu, Weiming Zhi - [[pdf]](https://arxiv.org/pdf/2509.11663)</summary>

**Abstract:** This paper formulates the Embodied Questions Answering (EQsA) problem, introduces a corresponding benchmark, and proposes an agentic system to tackle the problem. Classical Embodied Question Answering (EQA) is typically formulated as answering one single question by actively exploring a 3D environment. Real deployments, however, often demand handling multiple questions that may arrive asynchronously and carry different urgencies. We formalize this setting as Embodied Questions Answering (EQsA) and present ConEQsA, an agentic framework for concurrent, urgency-aware scheduling and answering. ConEQsA leverages shared group memory to reduce redundant exploration, and a priority-planning method to dynamically schedule questions. To evaluate the EQsA setting fairly, we contribute the Concurrent Asynchronous Embodied Questions (CAEQs) benchmark containing 40 indoor scenes and five questions per scene (200 in total), featuring asynchronous follow-up questions and human-annotated urgency labels. We further propose metrics for EQsA performance: Direct Answer Rate (DAR), and Normalized Urgency-Weighted Latency (NUWL), which serve as a fair evaluation protocol for EQsA. Empirical evaluations demonstrate that ConEQsA consistently outperforms strong sequential baselines, and show that urgency-aware, concurrent scheduling is key to making embodied agents responsive and efficient under realistic, multi-question workloads. Code is available on this https URL.

**arXiv ID:** 2509.11663
</details>

</details>

<details open>
<summary><h2>LLM Agents (9 papers)</h2></summary>

<details>
<summary><strong>Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory</strong> - Boqin Yuan, Yue Su, Kun Yao - [[pdf]](https://arxiv.org/pdf/2603.02473)</summary>

**Abstract:** Memory-augmented LLM agents store and retrieve information from prior interactions, yet the relative importance of how memories are written versus how they are retrieved remains unclear. We introduce a diagnostic framework that analyzes how performance differences manifest across write strategies, retrieval methods, and memory utilization behavior, and apply it to a 3x3 study crossing three write strategies (raw chunks, Mem0-style fact extraction, MemGPT-style summarization) with three retrieval methods (cosine, BM25, hybrid reranking). On LoCoMo, retrieval method is the dominant factor: average accuracy spans 20 points across retrieval methods (57.1% to 77.2%) but only 3-8 points across write strategies. Raw chunked storage, which requires zero LLM calls, matches or outperforms expensive lossy alternatives, suggesting that current memory pipelines may discard useful context that downstream retrieval mechanisms fail to compensate for. Failure analysis shows that performance breakdowns most often manifest at the retrieval stage rather than at utilization. We argue that, under current retrieval practices, improving retrieval quality yields larger gains than increasing write-time sophistication. Code is publicly available at this https URL.

**arXiv ID:** 2603.02473
</details>

<details>
<summary><strong>RAPO: Expanding Exploration for LLM Agents via Retrieval-Augmented Policy Optimization</strong> - Siwei Zhang, Yun Xiong, Xi Chen, Zi'an Jia, Renhong Huang, Jiarong Xu, Jiawei Zhang - [[pdf]](https://arxiv.org/pdf/2603.03078)</summary>

**Abstract:** Agentic Reinforcement Learning (Agentic RL) has shown remarkable potential in large language model-based (LLM) agents. These works can empower LLM agents to tackle complex tasks via multi-step, tool-integrated reasoning. However, an inherent limitation of existing Agentic RL methods is their reliance on a pure on-policy paradigm for exploration, restricting exploration to the agent's self-generated outputs and preventing the discovery of new reasoning perspectives for further improvement. While recent efforts incorporate auxiliary off-policy signals to enhance exploration, they typically utilize full off-policy trajectories for trajectory-level policy estimation, overlooking the necessity for the fine-grained, step-level exploratory dynamics within agentic rollout. In this paper, we revisit exploration in Agentic RL and propose Retrieval-Augmented Policy Optimization (RAPO), a novel RL framework that introduces retrieval to explicitly expand exploration during training. To achieve this, we decompose the Agentic RL training process into two phases: (i) Hybrid-policy Agentic Rollout, and (ii) Retrieval-aware Policy Optimization. Specifically, we propose a Hybrid-policy Agentic Rollout strategy, which allows the agents to continuously reason over the retrieved off-policy step-level traces. It dynamically extends the reasoning receptive field of agents, enabling broader exploration conditioned on external behaviors. Subsequently, we introduce the Retrieval-aware Policy Optimization mechanism, which calibrates the policy gradient estimation with retrieval reward and importance shaping, stabilizing training and prioritizing retrieval-illuminating exploration. Extensive experiments show that RAPO achieves an +5.0% average gain on fourteen datasets across three agentic reasoning tasks, while delivering 1.2x faster training efficiency.

**arXiv ID:** 2603.03078
</details>

<details>
<summary><strong>Beyond Task Completion: Revealing Corrupt Success in LLM Agents through Procedure-Aware Evaluation</strong> - Hongliu Cao, Ilias Driouich, Eoin Thomas - [[pdf]](https://arxiv.org/pdf/2603.03116)</summary>

**Abstract:** Large Language Model (LLM)-based agents are increasingly adopted in high-stakes settings, but current benchmarks evaluate mainly whether a task was completed, not how. We introduce Procedure-Aware Evaluation (PAE), a framework that formalizes agent procedures as structured observations and exposes consistency relationships between what agents observe, communicate, and execute. PAE evaluates agents along complementary axes (Utility, Efficiency, Interaction Quality, Procedural Integrity) and applies multi-dimensional gating that categorically disqualifies corrupt outcomes. Evaluating state-of-the-art LLM agents on tau-bench yields findings at the axis, compliance, and benchmark levels. At the axis level, the dimensions capture non-redundant failure modes: utility masks reliability gaps, speed does not imply precision, and conciseness does not predict intent adherence. At the procedural compliance level, 27-78% of benchmark reported successes are corrupt successes concealing violations across interaction and integrity. Furthermore, gating substantially collapses Pass^4 rate and affects model rankings. The analysis of corrupt success cases reveals distinctive per-model failure signatures: GPT-5 spreads errors across policy, execution, and intent dimensions; Kimi-K2-Thinking concentrates 78% of violations in policy faithfulness and compliance; and Mistral-Large-3 is dominated by faithfulness failures. At the benchmark level, our analysis exposes structural flaws in the benchmark design, including task scope gaps, contradictory reward signals, and simulator artifacts that produce accidental successes.

**arXiv ID:** 2603.03116
</details>

<details>
<summary><strong>Inherited Goal Drift: Contextual Pressure Can Undermine Agentic Goals</strong> - Achyutha Menon, Magnus Saebo, Tyler Crosse, Spencer Gibson, Eyon Jang, Diogo Cruz - [[pdf]](https://arxiv.org/pdf/2603.03258)</summary>

**Abstract:** The accelerating adoption of language models (LMs) as agents for deployment in long-context tasks motivates a thorough understanding of goal drift: agents' tendency to deviate from an original objective. While prior-generation language model agents have been shown to be susceptible to drift, the extent to which drift affects more recent models remains unclear. In this work, we provide an updated characterization of the extent and causes of goal drift. We investigate drift in state-of-the-art models within a simulated stock-trading environment (Arike et al., 2025). These models are largely shown to be robust even when subjected to adversarial pressure. We show, however, that this robustness is brittle: across multiple settings, the same models often inherit drift when conditioned on prefilled trajectories from weaker agents. The extent of conditioning-induced drift varies significantly by model family, with only GPT-5.1 maintaining consistent resilience among tested models. We find that drift behavior is inconsistent between prompt variations and correlates poorly with instruction hierarchy following behavior, with strong hierarchy following failing to reliably predict resistance to drift. Finally, we run analogous experiments in a new emergency room triage environment to show preliminary evidence for the transferability of our results across qualitatively different settings. Our findings underscore the continued vulnerability of modern LM agents to contextual pressures and the need for refined post-training techniques to mitigate this.

**arXiv ID:** 2603.03258
</details>

<details>
<summary><strong>ZeroDayBench: Evaluating LLM Agents on Unseen Zero-Day Vulnerabilities for Cyberdefense</strong> - Nancy Lau, Louis Sloot, Jyoutir Raj, Giuseppe Marco Boscardin, Evan Harris, Dylan Bowman, Mario Brajkovski, Jaideep Chawla, Dan Zhao - [[pdf]](https://arxiv.org/pdf/2603.02297)</summary>

**Abstract:** Large language models (LLMs) are increasingly being deployed as software engineering agents that autonomously contribute to repositories. A major benefit these agents present is their ability to find and patch security vulnerabilities in the codebases they oversee. To estimate the capability of agents in this domain, we introduce ZeroDayBench, a benchmark where LLM agents find and patch 22 novel critical vulnerabilities in open-source codebases. We focus our efforts on three popular frontier agentic LLMs: GPT-5.2, Claude Sonnet 4.5, and Grok 4.1. We find that frontier LLMs are not yet capable of autonomously solving our tasks and observe some behavioral patterns that suggest how these models can be improved in the domain of proactive cyberdefense.

**arXiv ID:** 2603.02297
</details>

<details>
<summary><strong>Contextualized Privacy Defense for LLM Agents</strong> - Yule Wen, Yanzhe Zhang, Jianxun Lian, Xiaoyuan Yi, Xing Xie, Diyi Yang - [[pdf]](https://arxiv.org/pdf/2603.02983)</summary>

**Abstract:** LLM agents increasingly act on users' personal information, yet existing privacy defenses remain limited in both design and adaptability. Most prior approaches rely on static or passive defenses, such as prompting and guarding. These paradigms are insufficient for supporting contextual, proactive privacy decisions in multi-step agent execution. We propose Contextualized Defense Instructing (CDI), a new privacy defense paradigm in which an instructor model generates step-specific, context-aware privacy guidance during execution, proactively shaping actions rather than merely constraining or vetoing them. Crucially, CDI is paired with an experience-driven optimization framework that trains the instructor via reinforcement learning (RL), where we convert failure trajectories with privacy violations into learning environments. We formalize baseline defenses and CDI as distinct intervention points in a canonical agent loop, and compare their privacy-helpfulness trade-offs within a unified simulation framework. Results show that our CDI consistently achieves a better balance between privacy preservation (94.2%) and helpfulness (80.6%) than baselines, with superior robustness to adversarial conditions and generalization.

**arXiv ID:** 2603.02983
</details>

<details>
<summary><strong>Echoing: Identity Failures when LLM Agents Talk to Each Other</strong> - Sarath Shekkizhar, Romain Cosentino, Adam Earle, Silvio Savarese - [[pdf]](https://arxiv.org/pdf/2511.09710)</summary>

**Abstract:** As large language model (LLM) based agents interact autonomously with one another, a new class of failures emerges that cannot be predicted from single agent performance: behavioral drifts in agent-agent conversations (AxA). Unlike human-agent interactions, where humans ground and steer conversations, AxA lacks such stabilizing signals, making these failures unique. We investigate one such failure, echoing, where agents abandon their assigned roles and instead mirror their conversational partners, undermining their intended objectives. Through experiments across $66$ AxA configurations, $4$ domains (3 transactional, 1 advisory), and $2500+$ conversations (over $250000$ LLM inferences), we show that echoing occurs across major LLM providers, with echoing rates as high as $70\%$ depending on the model and domain. Moreover, we find that echoing is persistent even in advanced reasoning models with substantial rates ($32.8\%$) that are not reduced by reasoning efforts. We analyze prompt, conversation dynamics, showing that echoing arises as interaction grows longer ($7+$ agent turns) and is not merely an artifact of sub-optimal experiment design. Finally, we introduce a protocol-level mitigation where targeted use of structured response reduces echoing to $9\%$.

**arXiv ID:** 2511.09710
</details>

<details>
<summary><strong>Reducing Belief Deviation in Reinforcement Learning for Active Reasoning</strong> - Deyu Zou, Yongqiang Chen, Jianxiang Wang, Haochen Yang, Mufei Li, James Cheng, Pan Li, Yu Gong - [[pdf]](https://arxiv.org/pdf/2510.12264)</summary>

**Abstract:** Active reasoning requires large language model (LLM) agents to interact with external sources and strategically gather information to solve problems in multiple turns. Central to this process is belief tracking: maintaining an accurate representation of the underlying state and uncertainty in understanding and solving the problem. However, due to limited reasoning capabilities, LLM-based agents often suffer belief deviation: their internal beliefs drift from the true problem state, leading to loss of state awareness and uninformative or repetitive actions. Once this happens, errors compound in the trajectories used for reinforcement learning (RL), leading to misattributed credits and limited exploration. To address this issue, we propose to track belief deviation and develop $\mathbf{T^3}$, a simple yet principled method that detects excessive deviation and truncates training trajectories to suppress uninformative tail effects. Hence, $\mathbf{T^3}$ preserves credits for informative prefixes and systematically improves policy optimization. Across 5 challenging tasks, $\mathbf{T^3}$ consistently enhances training stability and yields performance gains of up to 30 points while cutting token cost by up to 34%. These results highlight belief control as a key principle for building robust LLM agents capable of active reasoning.

**arXiv ID:** 2510.12264
</details>

<details>
<summary><strong>Safety Training Persists Through Helpfulness Optimization in LLM Agents</strong> - Benjamin Plaut - [[pdf]](https://arxiv.org/pdf/2603.02229)</summary>

**Abstract:** Safety post-training has been studied extensively in single-step "chat" settings where safety typically refers to refusing harmful requests. We study an "agentic" (i.e., multi-step, tool-use) setting where safety refers to harmful actions directly taken by the LLM. We compare the effects of running direct preference optimization (DPO) on safety or helpfulness alone vs both metrics sequentially. As expected, training on one metric alone results in an extreme point along this frontier. However, unlike prior work, we find that safety training persists through subsequent helpfulness training. We also find that all training configurations end up near a linear Pareto frontier with $R^2 = 0.77$. Even post-training on both metrics simultaneously simply results in another point on the frontier rather than finding a "best of both worlds" strategy, despite the presence of such strategies in our DPO dataset. Overall, our findings underscore the need for better understanding of post-training dynamics.

**arXiv ID:** 2603.02229
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (23 papers)</h2></summary>

<details>
<summary><strong>SuperLocalMemory: Privacy-Preserving Multi-Agent Memory with Bayesian Trust Defense Against Memory Poisoning</strong> - Varun Pratap Bhardwaj - [[pdf]](https://arxiv.org/pdf/2603.02240)</summary>

**Abstract:** We present SuperLocalMemory, a local-first memory system for multi-agent AI that defends against OWASP ASI06 memory poisoning through architectural isolation and Bayesian trust scoring, while personalizing retrieval through adaptive learning-to-rank -- all without cloud dependencies or LLM inference calls. As AI agents increasingly rely on persistent memory, cloud-based memory systems create centralized attack surfaces where poisoned memories propagate across sessions and users -- a threat demonstrated in documented attacks against production systems. Our architecture combines SQLite-backed storage with FTS5 full-text search, Leiden-based knowledge graph clustering, an event-driven coordination layer with per-agent provenance, and an adaptive re-ranking framework that learns user preferences through three-layer behavioral analysis (cross-project technology preferences, project context detection, and workflow pattern mining). Evaluation across seven benchmark dimensions demonstrates 10.6ms median search latency, zero concurrency errors under 10 simultaneous agents, trust separation (gap =0.90) with 72% trust degradation for sleeper attacks, and 104% improvement in NDCG@5 when adaptive re-ranking is enabled. Behavioral data is isolated in a separate database with GDPR Article 17 erasure support. SuperLocalMemory is open-source (MIT) and integrates with 17+ development tools via Model Context Protocol.

**arXiv ID:** 2603.02240
</details>

<details>
<summary><strong>A Natural Language Agentic Approach to Study Affective Polarization</strong> - Stephanie Anneris Malvicini, Ewelina Gajewska, Arda Derbent, Katarzyna Budzynska, Jarosław A. Chudziak, Maria Vanina Martinez - [[pdf]](https://arxiv.org/pdf/2603.02711)</summary>

**Abstract:** Affective polarization has been central to political and social studies, with growing focus on social media, where partisan divisions are often exacerbated. Real-world studies tend to have limited scope, while simulated studies suffer from insufficient high-quality training data, as manually labeling posts is labor-intensive and prone to subjective biases. The lack of adequate tools to formalize different definitions of affective polarization across studies complicates result comparison and hinders interoperable frameworks. We present a multi-agent model providing a comprehensive approach to studying affective polarization in social media. To operationalize our framework, we develop a platform leveraging large language models (LLMs) to construct virtual communities where agents engage in discussions. We showcase the potential of our platform by (1) analyzing questions related to affective polarization, as explored in social science literature, providing a fresh perspective on this phenomenon, and (2) introducing scenarios that allow observation and measurement of polarization at different levels of granularity and abstraction. Experiments show that our platform is a flexible tool for computational studies of complex social dynamics such as affective polarization. It leverages advanced agent models to simulate rich, context-sensitive interactions and systematically explore research questions traditionally addressed through human-subject studies.

**arXiv ID:** 2603.02711
</details>

<details>
<summary><strong>EvoSkill: Automated Skill Discovery for Multi-Agent Systems</strong> - Salaheddin Alzubi, Noah Provenzano, Jaydon Bingham, Weiyuan Chen, Tu Vu - [[pdf]](https://arxiv.org/pdf/2603.02766)</summary>

**Abstract:** Coding agents are increasingly used as general-purpose problem solvers, but their flexibility does not by itself confer the domain expertise needed for specialized tasks. Recent work addresses this through \textit{agent skills}: reusable workflows, and code, that augment agents with domain-specific capabilities. Most skills today are hand-crafted, and existing evolutionary approaches optimize low-level artifacts (e.g. prompts \& code) that are tightly coupled to specific models and tasks. We introduce \textbf{EvoSkill}, a self-evolving framework that automatically discovers and refines agent skills through iterative failure analysis. EvoSkill analyzes execution failures, proposes new skills or edits to existing ones, and materializes them into structured, reusable skill folders. A Pareto frontier of agent programs governs selection, retaining only skills that improve held-out validation performance while the underlying model remains frozen. We evaluate EvoSkill on two benchmarks: OfficeQA, a grounded reasoning benchmark over U.S.\ Treasury data, where it improves exact-match accuracy by \textbf{7.3\%} (60.6\% $\to$ 67.9\%); and SealQA, a search-augmented QA benchmark with noisy retrieval, where it yields a \textbf{12.1\%} gain (26.6\% $\to$ 38.7\%). We also investigate the zero-shot transfer capabilties of skills evolved on one task to the other; in particular: skills evolved from SealQA transfers zero-shot to BrowseComp, improving accuracy by \textbf{5.3\%} without modification demonstrating that skill-level optimization produces transferable capabilities beyond the training task.

**arXiv ID:** 2603.02766
</details>

<details>
<summary><strong>Architecting Trust in Artificial Epistemic Agents</strong> - Nahema Marchal, Stephanie Chan, Matija Franklin, Manon Revel, Geoff Keeling, Roberta Fischli, Bilva Chandra, Iason Gabriel - [[pdf]](https://arxiv.org/pdf/2603.02960)</summary>

**Abstract:** Large language models increasingly function as epistemic agents -- entities that can 1) autonomously pursue epistemic goals and 2) actively shape our shared knowledge environment. They curate the information we receive, often supplanting traditional search-based methods, and are frequently used to generate both personal and deeply specialized advice. How they perform these functions, including whether they are reliable and properly calibrated to both individual and collective epistemic norms, is therefore highly consequential for the choices we make. We argue that the potential impact of epistemic AI agents on practices of knowledge creation, curation and synthesis, particularly in the context of complex multi-agent interactions, creates new informational interdependencies that necessitate a fundamental shift in evaluation and governance of AI. While a well-calibrated ecosystem could augment human judgment and collective decision-making, poorly aligned agents risk causing cognitive deskilling and epistemic drift, making the calibration of these models to human norms a high-stakes necessity. To ensure a beneficial human-AI knowledge ecosystem, we propose a framework centered on building and cultivating the trustworthiness of epistemic AI agents; aligning AI these agents with human epistemic goals; and reinforcing the surrounding socio-epistemic infrastructure. In this context, trustworthy AI agents must demonstrate epistemic competence, robust falsifiability, and epistemically virtuous behaviors, supported by technical provenance systems and "knowledge sanctuaries" designed to protect human resilience. This normative roadmap provides a path toward ensuring that future AI systems act as reliable partners in a robust and inclusive knowledge ecosystem.

**arXiv ID:** 2603.02960
</details>

<details>
<summary><strong>OrchMAS: Orchestrated Reasoning with Multi Collaborative Heterogeneous Scientific Expert Structured Agents</strong> - Yichao Feng, Haoran Luo, Zhenghong Lin, Yiqun Sun, Pengfei Wei, Lawrence B. Hsieh, Anh Tuan Luu - [[pdf]](https://arxiv.org/pdf/2603.03005)</summary>

**Abstract:** Multi-agent large language model frameworks are promising for complex multi step reasoning, yet existing systems remain weak for scientific and knowledge intensive domains due to static prompts and agent roles, rigid workflows, and homogeneous model reliance, leading to poor domain adaptation, limited reasoning flexibility, and high latency on heterogeneous or long-horizon scientific tasks. They also struggle to revise earlier decisions when intermediate reasoning diverges, reducing reliability in structured and calculation heavy settings. To address these limitations, we propose a scientific domain oriented interactive two tier multi model orchestration framework. A dedicated orchestration model analyzes each task, dynamically constructs a domain aware reasoning pipeline, and instantiates specialized expert agents with tailored prompts, while an execution model performs each step under generated role and instruction specifications. The orchestrator iteratively updates the pipeline based on intermediate feedback, enabling dynamic replanning, role reallocation, and prompt refinement across multi turn interactions, strengthening robustness and specialization for scientific reasoning through structured heterogeneous model collaboration. The framework is model agnostic and supports heterogeneous LLM integration with different capacities or costs, enabling flexible performance efficiency trade offs in practical scientific deployments. Experiments show consistent improvements over existing multi agent systems and strong baselines across diverse reasoning and scientific style benchmarks.

**arXiv ID:** 2603.03005
</details>

<details>
<summary><strong>AI-for-Science Low-code Platform with Bayesian Adversarial Multi-Agent Framework</strong> - Zihang Zeng, Jiaquan Zhang, Pengze Li, Yuan Qi, Xi Chen - [[pdf]](https://arxiv.org/pdf/2603.03233)</summary>

**Abstract:** Large Language Models (LLMs) demonstrate potentials for automating scientific code generation but face challenges in reliability, error propagation in multi-agent workflows, and evaluation in domains with ill-defined success metrics. We present a Bayesian adversarial multi-agent framework specifically designed for AI for Science (AI4S) tasks in the form of a Low-code Platform (LCP). Three LLM-based agents are coordinated under the Bayesian framework: a Task Manager that structures user inputs into actionable plans and adaptive test cases, a Code Generator that produces candidate solutions, and an Evaluator providing comprehensive feedback. The framework employs an adversarial loop where the Task Manager iteratively refines test cases to challenge the Code Generator, while prompt distributions are dynamically updated using Bayesian principles by integrating code quality metrics: functional correctness, structural alignment, and static analysis. This co-optimization of tests and code reduces dependence on LLM reliability and addresses evaluation uncertainty inherent to scientific tasks. LCP also streamlines human-AI collaboration by translating non-expert prompts into domain-specific requirements, bypassing the need for manual prompt engineering by practitioners without coding backgrounds. Benchmark evaluations demonstrate LCP's effectiveness in generating robust code while minimizing error propagation. The proposed platform is also tested on an Earth Science cross-disciplinary task and demonstrates strong reliability, outperforming competing models.

**arXiv ID:** 2603.03233
</details>

<details>
<summary><strong>RIVA: Leveraging LLM Agents for Reliable Configuration Drift Detection</strong> - Sami Abuzakuk, Lucas Crijns, Anne-Marie Kermarrec, Rafael Pires, Martijn de Vos - [[pdf]](https://arxiv.org/pdf/2603.02345)</summary>

**Abstract:** Infrastructure as code (IaC) tools automate cloud provisioning but verifying that deployed systems remain consistent with the IaC specifications remains challenging. Such configuration drift occurs because of bugs in the IaC specification, manual changes, or system updates. Large language model (LLM)-based agentic AI systems can automate the analysis of large volumes of telemetry data, making them suitable for the detection of configuration drift. However, existing agentic systems implicitly assume that the tools they invoke always return correct outputs, making them vulnerable to erroneous tool responses. Since agents cannot distinguish whether an anomalous tool output reflects a real infrastructure problem or a broken tool, such errors may cause missed drift or false alarms, reducing reliability precisely when it is most needed. We introduce RIVA (Robust Infrastructure by Verification Agents), a novel multi-agent system that performs robust IaC verification even when tools produce incorrect or misleading outputs. RIVA employs two specialized agents, a verifier agent and a tool generation agent, that collaborate through iterative cross-validation, multi-perspective verification, and tool call history tracking. Evaluation on the AIOpsLab benchmark demonstrates that RIVA, in the presence of erroneous tool responses, recovers task accuracy from 27.3% when using a baseline ReAct agent to 50.0% on average. RIVA also improves task accuracy 28% to 43.8% without erroneous tool responses. Our results show that cross-validation of diverse tool calls enables more reliable autonomous infrastructure verification in production cloud environments.

**arXiv ID:** 2603.02345
</details>

<details>
<summary><strong>MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks</strong> - Zhi Hong, Qian Zhang, Jiahang Sun, Zhiwei Shang, Mingze Kong, Xiangyi Wang, Yao Shu, Zhongxiang Dai - [[pdf]](https://arxiv.org/pdf/2603.02630)</summary>

**Abstract:** Large Language Models (LLMs) have achieved great success in many real-world applications, especially the one serving as the cognitive backbone of Multi-Agent Systems (MAS) to orchestrate complex workflows in practice. Since many deployment scenarios preclude MAS workflow modifications and its performance is highly sensitive to the input prompts, prompt optimization emerges as a more natural approach to improve its performance. However, real-world prompt optimization for MAS is impeded by three key challenges: (1) the need of sample efficiency due to prohibitive evaluation costs, (2) topology-induced coupling among prompts, and (3) the combinatorial explosion of the search space. To address these challenges, we introduce MASPOB (Multi-Agent System Prompt Optimization via Bandits), a novel sample-efficient framework based on bandits. By leveraging Upper Confidence Bound (UCB) to quantify uncertainty, the bandit framework balances exploration and exploitation, maximizing gains within a strictly limited budget. To handle topology-induced coupling, MASPOB integrates Graph Neural Networks (GNNs) to capture structural priors, learning topology-aware representations of prompt semantics. Furthermore, it employs coordinate ascent to decompose the optimization into univariate sub-problems, reducing search complexity from exponential to linear. Extensive experiments across diverse benchmarks demonstrate that MASPOB achieves state-of-the-art performance, consistently outperforming existing baselines.

**arXiv ID:** 2603.02630
</details>

<details>
<summary><strong>ShareVerse: Multi-Agent Consistent Video Generation for Shared World Modeling</strong> - Jiayi Zhu, Jianing Zhang, Yiying Yang, Wei Cheng, Xiaoyun Yuan - [[pdf]](https://arxiv.org/pdf/2603.02697)</summary>

**Abstract:** This paper presents ShareVerse, a video generation framework enabling multi-agent shared world modeling, addressing the gap in existing works that lack support for unified shared world construction with multi-agent interaction. ShareVerse leverages the generation capability of large video models and integrates three key innovations: 1) A dataset for large-scale multi-agent interactive world modeling is built on the CARLA simulation platform, featuring diverse scenes, weather conditions, and interactive trajectories with paired multi-view videos (front/ rear/ left/ right views per agent) and camera data. 2) We propose a spatial concatenation strategy for four-view videos of independent agents to model a broader environment and to ensure internal multi-view geometric consistency. 3) We integrate cross-agent attention blocks into the pretrained video model, which enable interactive transmission of spatial-temporal information across agents, guaranteeing shared world consistency in overlapping regions and reasonable generation in non-overlapping regions. ShareVerse, which supports 49-frame large-scale video generation, accurately perceives the position of dynamic agents and achieves consistent shared world modeling.

**arXiv ID:** 2603.02697
</details>

<details>
<summary><strong>BrandFusion: A Multi-Agent Framework for Seamless Brand Integration in Text-to-Video Generation</strong> - Zihao Zhu, Ruotong Wang, Siwei Lyu, Min Zhang, Baoyuan Wu - [[pdf]](https://arxiv.org/pdf/2603.02816)</summary>

**Abstract:** The rapid advancement of text-to-video (T2V) models has revolutionized content creation, yet their commercial potential remains largely untapped. We introduce, for the first time, the task of seamless brand integration in T2V: automatically embedding advertiser brands into prompt-generated videos while preserving semantic fidelity to user intent. This task confronts three core challenges: maintaining prompt fidelity, ensuring brand recognizability, and achieving contextually natural integration. To address them, we propose BrandFusion, a novel multi-agent framework comprising two synergistic phases. In the offline phase (advertiser-facing), we construct a Brand Knowledge Base by probing model priors and adapting to novel brands via lightweight fine-tuning. In the online phase (user-facing), five agents jointly refine user prompts through iterative refinement, leveraging the shared knowledge base and real-time contextual tracking to ensure brand visibility and semantic alignment. Experiments on 18 established and 2 custom brands across multiple state-of-the-art T2V models demonstrate that BrandFusion significantly outperforms baselines in semantic preservation, brand recognizability, and integration naturalness. Human evaluations further confirm higher user satisfaction, establishing a practical pathway for sustainable T2V monetization.

**arXiv ID:** 2603.02816
</details>

<details>
<summary><strong>Learning to Generate and Extract: A Multi-Agent Collaboration Framework For Zero-shot Document-level Event Arguments Extraction</strong> - Guangjun Zhang, Hu Zhang, Yazhou Han, Yue Fan, Yuhang Shao, Ru Li, Hongye Tan - [[pdf]](https://arxiv.org/pdf/2603.02909)</summary>

**Abstract:** Document-level event argument extraction (DEAE) is essential for knowledge acquisition, aiming to extract participants of events from this http URL the zero-shot setting, existing methods employ LLMs to generate synthetic data to address the challenge posed by the scarcity of annotated data. However, relying solely on Event-type-only prompts makes it difficult for the generated content to accurately capture the contextual and structural relationships of unseen events. Moreover, ensuring the reliability and usability of synthetic data remains a significant challenge due to the absence of quality evaluation mechanisms. To this end, we introduce a multi-agent collaboration framework for zero-shot document-level event argument extraction (ZS-DEAE), which simulates the human collaborative cognitive process of "Propose-Evaluate-Revise." Specifically, the framework comprises a generation agent and an evaluation agent. The generation agent synthesizes data for unseen events by leveraging knowledge from seen events, while the evaluation agent extracts arguments from the synthetic data and assesses their semantic consistency with the context. The evaluation results are subsequently converted into reward signals, with event structure constraints incorporated into the reward design to enable iterative optimization of both agents via reinforcement this http URL three zero-shot scenarios constructed from the RAMS and WikiEvents datasets, our method achieves improvements both in data generation quality and argument extraction performance, while the generated data also effectively enhances the zero-shot performance of other DEAE models.

**arXiv ID:** 2603.02909
</details>

<details>
<summary><strong>MA-CoNav: A Master-Slave Multi-Agent Framework with Hierarchical Collaboration and Dual-Level Reflection for Long-Horizon Embodied VLN</strong> - Ling Luo, Qianqian Bai - [[pdf]](https://arxiv.org/pdf/2603.03024)</summary>

**Abstract:** Vision-Language Navigation (VLN) aims to empower robots with the ability to perform long-horizon navigation in unfamiliar environments based on complex linguistic instructions. Its success critically hinges on establishing an efficient ``language-understanding -- visual-perception -- embodied-execution'' closed loop. Existing methods often suffer from perceptual distortion and decision drift in complex, long-distance tasks due to the cognitive overload of a single agent. Inspired by distributed cognition theory, this paper proposes MA-CoNav, a Multi-Agent Collaborative Navigation framework. This framework adopts a ``Master-Slave'' hierarchical agent collaboration architecture, decoupling and distributing the perception, planning, execution, and memory functions required for navigation tasks to specialized agents. Specifically, the Master Agent is responsible for global orchestration, while the Subordinate Agent group collaborates through a clear division of labor: an Observation Agent generates environment descriptions, a Planning Agent performs task decomposition and dynamic verification, an Execution Agent handles simultaneous mapping and action, and a Memory Agent manages structured experiences. Furthermore, the framework introduces a ``Local-Global'' dual-stage reflection mechanism to dynamically optimize the entire navigation pipeline. Empirical experiments were conducted using a real-world indoor dataset collected by a Limo Pro robot, with no scene-specific fine-tuning performed on the models throughout the process. The results demonstrate that MA-CoNav comprehensively outperforms existing mainstream VLN methods across multiple metrics.

**arXiv ID:** 2603.03024
</details>

<details>
<summary><strong>MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models</strong> - Siqi Ma, Jiajie Huang, Fan Zhang, Yue Shen, Jinlin Wu, Guohui Fan, Zhu Zhang, Zelin Zang - [[pdf]](https://arxiv.org/pdf/2509.23725)</summary>

**Abstract:** Answering complex medical questions requires not only domain expertise and patient-specific information, but also structured and multi-perspective reasoning. Existing multi-agent approaches often rely on fixed roles or shallow interaction prompts, limiting their ability to detect and resolve fine-grained logical inconsistencies. To address this, we propose \textsc{MedLA}, a logic-driven multi-agent framework built on large language models. Each agent organizes its reasoning process into an explicit logical tree based on syllogistic triads (major premise, minor premise, and conclusion), enabling transparent inference and premise-level alignment. Agents engage in a multi-round, graph-guided discussion to compare and iteratively refine their logic trees, achieving consensus through error correction and contradiction resolution. We demonstrate that \textsc{MedLA} consistently outperforms both static role-based systems and single-agent baselines on challenging benchmarks such as MedDDx and standard medical QA tasks. Furthermore, \textsc{MedLA} scales effectively across both open-source and commercial LLM backbones, achieving state-of-the-art performance and offering a generalizable paradigm for trustworthy medical reasoning.

**arXiv ID:** 2509.23725
</details>

<details>
<summary><strong>Comparing AI Agents to Cybersecurity Professionals in Real-World Penetration Testing</strong> - Justin W. Lin, Eliot Krzysztof Jones, Donovan Julian Jasper, Ethan Jun-shen Ho, Anna Wu, Arnold Tianyi Yang, Neil Perry, Andy Zou, Matt Fredrikson, J. Zico Kolter, Percy Liang, Dan Boneh, Daniel E. Ho - [[pdf]](https://arxiv.org/pdf/2512.09882)</summary>

**Abstract:** We present the first comprehensive evaluation of AI agents against human cybersecurity professionals in a live enterprise environment. We evaluate ten cybersecurity professionals alongside six existing AI agents and ARTEMIS, our new agent scaffold, on a large university network consisting of ~8,000 hosts across 12 subnets. ARTEMIS is a multi-agent framework featuring dynamic prompt generation, arbitrary sub-agents, and automatic vulnerability triaging. In our comparative study, ARTEMIS placed second overall, discovering 9 valid vulnerabilities with an 82% valid submission rate and outperforming 9 of 10 human participants. While existing scaffolds such as Codex and CyAgent underperformed relative to most human participants, ARTEMIS demonstrated technical sophistication and submission quality comparable to the strongest participants. We observe that AI agents offer advantages in systematic enumeration, parallel exploitation, and cost -- certain ARTEMIS variants cost $18/hour versus $60/hour for professional penetration testers. We also identify key capability gaps: AI agents exhibit higher false-positive rates and struggle with GUI-based tasks.

**arXiv ID:** 2512.09882
</details>

<details>
<summary><strong>OpenClaw, Moltbook, and ClawdLab: From Agent-Only Social Networks to Autonomous Scientific Research</strong> - Lukas Weidener, Marko Brkić, Mihailo Jovanović, Ritvik Singh, Emre Ulgac, Aakaash Meduri - [[pdf]](https://arxiv.org/pdf/2602.19810)</summary>

**Abstract:** In January 2026, the open-source agent framework OpenClaw and the agent-only social network Moltbook produced a large-scale dataset of autonomous AI-to-AI interaction, attracting six academic publications within fourteen days. This study conducts a multivocal literature review of that ecosystem and presents ClawdLab, an open-source platform for autonomous scientific research, as a design science response to the architectural failure modes identified. The literature documents emergent collective phenomena, security vulnerabilities spanning 131 agent skills and over 15,200 exposed control panels, and five recurring architectural patterns. ClawdLab addresses these failure modes through hard role restrictions, structured adversarial critique, PI-led governance, multi-model orchestration, and domain-specific evidence requirements encoded as protocol constraints that ground validation in computational tool outputs rather than social consensus; the architecture provides emergent Sybil resistance as a structural consequence. A three-tier taxonomy distinguishes single-agent pipelines, predetermined multi-agent workflows, and fully decentralised systems, analysing why leading AI co-scientist platforms remain confined to the first two tiers. ClawdLab's composable third-tier architecture, in which foundation models, capabilities, governance, and evidence requirements are independently modifiable, enables compounding improvement as the broader AI ecosystem advances.

**arXiv ID:** 2602.19810
</details>

<details>
<summary><strong>StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning</strong> - Shiyang Li, Zijian Zhang, Winson Chen, Yuebo Luo, Mingyi Hong, Caiwen Ding - [[pdf]](https://arxiv.org/pdf/2603.02637)</summary>

**Abstract:** Modern machine learning (ML) workloads increasingly rely on GPUs, yet achieving high end-to-end performance remains challenging due to dependencies on both GPU kernel efficiency and host-side settings. Although LLM-based methods show promise on automated GPU kernel generation, prior works mainly focus on single-kernel optimization and do not extend to end-to-end programs, hindering practical deployment.
To address the challenge, in this work, we propose StitchCUDA, a multi-agent framework for end-to-end GPU program generation, with three specialized agents: a Planner to orchestrate whole system design, a Coder dedicated to implementing it step-by-step, and a Verifier for correctness check and performance profiling using Nsys/NCU. To fundamentally improve the Coder's ability in end-to-end GPU programming, StitchCUDA integrates rubric-based agentic reinforcement learning over two atomic skills, task-to-code generation and feedback-driven code optimization, with combined rubric reward and rule-based reward from real executions. Therefore, the Coder learns how to implement advanced CUDA programming techniques (e.g., custom kernel fusion, cublas epilogue), and we also effectively prevent Coder's reward hacking (e.g., just copy PyTorch code or hardcoding output) during benchmarking. Experiments on KernelBench show that StitchCUDA achieves nearly 100% success rate on end-to-end GPU programming tasks, with 1.72x better speedup over the multi-agent baseline and 2.73x than the RL model baselines.

**arXiv ID:** 2603.02637
</details>

<details>
<summary><strong>Generalized Per-Agent Advantage Estimation for Multi-Agent Policy Optimization</strong> - Seongmin Kim, Giseung Park, Woojun Kim, Jiwon Jeon, Seungyeol Han, Youngchul Sung - [[pdf]](https://arxiv.org/pdf/2603.02654)</summary>

**Abstract:** In this paper, we propose a novel framework for multi-agent reinforcement learning that enhances sample efficiency and coordination through accurate per-agent advantage estimation. The core of our approach is Generalized Per-Agent Advantage Estimator (GPAE), which employs a per-agent value iteration operator to compute precise per-agent advantages. This operator enables stable off-policy learning by indirectly estimating values via action probabilities, eliminating the need for direct Q-function estimation. To further refine estimation, we introduce a double-truncated importance sampling ratio scheme. This scheme improves credit assignment for off-policy trajectories by balancing sensitivity to the agent's own policy changes with robustness to non-stationarity from other agents. Experiments on benchmarks demonstrate that our approach outperforms existing approaches, excelling in coordination and sample efficiency for complex scenarios.

**arXiv ID:** 2603.02654
</details>

<details>
<summary><strong>NeuroWise: A Multi-Agent LLM "Glass-Box" System for Practicing Double-Empathy Communication with Autistic Partners</strong> - Albert Tang, Yifan Mo, Jie Li, Yue Su, Mengyuan Zhang, Sander L. Koole, Koen Hindriks, Jiahuan Pei - [[pdf]](https://arxiv.org/pdf/2602.18962)</summary>

**Abstract:** The double empathy problem frames communication difficulties between neurodivergent and neurotypical individuals as arising from mutual misunderstanding, yet most interventions focus on autistic individuals. We present NeuroWise, a multi-agent LLM-based coaching system that supports neurotypical users through stress visualization, interpretation of internal experiences, and contextual guidance. In a between-subjects study (N=30), NeuroWise was rated as helpful by all participants and showed a significant condition-time effect on deficit-based attributions (p=0.02): NeuroWise users reduced deficit framing, while baseline users shifted toward blaming autistic "deficits" after difficult interactions. NeuroWise users also completed conversations more efficiently (37% fewer turns, p=0.03). These findings suggest that AI-based interpretation can support attributional change by helping users recognize communication challenges as mutual.

**arXiv ID:** 2602.18962
</details>

<details>
<summary><strong>Graph-GRPO: Stabilizing Multi-Agent Topology Learning via Group Relative Policy Optimization</strong> - Yueyang Cang, Xiaoteng Zhang, Erlu Zhao, Zehua Ji, Yuhang Liu, Yuchen He, Zhiyuan Ning, Chen Yijun, Wenge Que, Li Shi - [[pdf]](https://arxiv.org/pdf/2603.02701)</summary>

**Abstract:** Optimizing communication topology is fundamental to the efficiency and effectiveness of Large Language Model (LLM)-based Multi-Agent Systems (MAS). While recent approaches utilize reinforcement learning to dynamically construct task-specific graphs, they typically rely on single-sample policy gradients with absolute rewards (e.g., binary correctness). This paradigm suffers from severe gradient variance and the credit assignment problem: simple queries yield non-informative positive rewards for suboptimal structures, while difficult queries often result in failures that provide no learning signal. To address these challenges, we propose Graph-GRPO, a novel topology optimization framework that integrates Group Relative Policy Optimization. Instead of evaluating a single topology in isolation, Graph-GRPO samples a group of diverse communication graphs for each query and computes the advantage of specific edges based on their relative performance within the group. By normalizing rewards across the sampled group, our method effectively mitigates the noise derived from task difficulty variance and enables fine-grained credit assignment. Extensive experiments on reasoning and code generation benchmarks demonstrate that Graph-GRPO significantly outperforms state-of-the-art baselines, achieving superior training stability and identifying critical communication pathways previously obscured by reward noise.

**arXiv ID:** 2603.02701
</details>

<details>
<summary><strong>Code2Math: Can Your Code Agent Effectively Evolve Math Problems Through Exploration?</strong> - Dadi Guo, Yuejin Xie, Qingyu Liu, Jiayu Liu, Zhiyuan Fan, Qihan Ren, Shuai Shao, Tianyi Zhou, Dongrui Liu, Yi R. Fung - [[pdf]](https://arxiv.org/pdf/2603.03202)</summary>

**Abstract:** As large language models (LLMs) advance their mathematical capabilities toward the IMO level, the scarcity of challenging, high-quality problems for training and evaluation has become a significant bottleneck. Simultaneously, recent code agents have demonstrated sophisticated skills in agentic coding and reasoning, suggesting that code execution can serve as a scalable environment for mathematical experimentation. In this paper, we investigate the potential of code agents to autonomously evolve existing math problems into more complex variations. We introduce a multi-agent framework designed to perform problem evolution while validating the solvability and increased difficulty of the generated problems. Our experiments demonstrate that, given sufficient test-time exploration, code agents can synthesize new, solvable problems that are structurally distinct from and more challenging than the originals. This work provides empirical evidence that code-driven agents can serve as a viable mechanism for synthesizing high-difficulty mathematical reasoning problems within scalable computational environments. Our data is available at this https URL.

**arXiv ID:** 2603.03202
</details>

<details>
<summary><strong>Personalized Multi-Agent Average Reward TD-Learning via Joint Linear Approximation</strong> - Wang, Pengkun Yang, Lili Su - [[pdf]](https://arxiv.org/pdf/2603.02426)</summary>

**Abstract:** We study personalized multi-agent average reward TD learning, in which a collection of agents interacts with different environments and jointly learns their respective value functions. We focus on the setting where there exists a shared linear representation, and the agents' optimal weights collectively lie in an unknown linear subspace. Inspired by the recent success of personalized federated learning (PFL), we study the convergence of cooperative single-timescale TD learning in which agents iteratively estimate the common subspace and local heads. We showed that this decomposition can filter out conflicting signals, effectively mitigating the negative impacts of ``misaligned'' signals, and achieving linear speedup. The main technical challenges lie in the heterogeneity, the Markovian sampling, and their intricate interplay in shaping error evolutions. Specifically, not only are the error dynamics of multiple variables closely interconnected, but there is also no direct contraction for the principal angle distance between the optimal subspace and the estimated subspace. We hope our analytical techniques can be useful to inspire research on deeper exploration into leveraging common structures. Experiments are provided to show the benefits of learning via a shared structure to the more general control problem.

**arXiv ID:** 2603.02426
</details>

<details>
<summary><strong>Heterogeneous Agent Collaborative Reinforcement Learning</strong> - Zhixia Zhang, Zixuan Huang, Xin Xia, Deqing Wang, Fuzhen Zhuang, Shuai Ma, Ning Ding, Yaodong Yang, Jianxin Li, Yikun Ban - [[pdf]](https://arxiv.org/pdf/2603.02604)</summary>

**Abstract:** We introduce Heterogeneous Agent Collaborative Reinforcement Learning (HACRL), a new learning paradigm that addresses the inefficiencies of isolated on-policy optimization. HACRL enables collaborative optimization with independent execution: heterogeneous agents share verified rollouts during training to mutually improve, while operating independently at inference time. Unlike LLM-based multi-agent reinforcement learning (MARL), HACRL does not require coordinated deployment, and unlike on-/off-policy distillation, it enables bidirectional mutual learning among heterogeneous agents rather than one-directional teacher-to-student transfer. Building on this paradigm, we propose HACPO, a collaborative RL algorithm that enables principled rollout sharing to maximize sample utilization and cross-agent knowledge transfer. To mitigate capability discrepancies and policy distribution shifts, HACPO introduces four tailored mechanisms with theoretical guarantees on unbiased advantage estimation and optimization correctness. Extensive experiments across diverse heterogeneous model combinations and reasoning benchmarks show that HACPO consistently improves all participating agents, outperforming GSPO by an average of 3.3\% while using only half the rollout cost.

**arXiv ID:** 2603.02604
</details>

<details>
<summary><strong>D-GVIO: A Buffer-Driven and Efficient Decentralized GNSS-Visual-Inertial State Estimator for Multi-Agent Systems</strong> - Yarong Luo, Wentao Lu, Chi Guo, Ming Li - [[pdf]](https://arxiv.org/pdf/2603.01404)</summary>

**Abstract:** Cooperative localization is essential for swarm applications like collaborative exploration and search-and-rescue missions. However, maintaining real-time capability, robustness, and computational efficiency on resource-constrained platforms presents significant challenges. To address these challenges, we propose D-GVIO, a buffer-driven and fully decentralized GNSS-Visual-Inertial Odometry (GVIO) framework that leverages a novel buffering strategy to support efficient and robust distributed state estimation. The proposed framework is characterized by four core mechanisms. Firstly, through covariance segmentation, covariance intersection and buffering strategy, we modularize propagation and update steps in distributed state estimation, significantly reducing computational and communication burdens. Secondly, the left-invariant extended Kalman filter (L-IEKF) is adopted for information fusion, which exhibits superior state estimation performance over the traditional extended Kalman filter (EKF) since its state transition matrix is independent of the system state. Thirdly, a buffer-based re-propagation strategy is employed to handle delayed measurements efficiently and accurately by leveraging the L-IEKF, eliminating the need for costly re-computation. Finally, an adaptive buffer-driven outlier detection method is proposed to dynamically cull GNSS outliers, enhancing robustness in GNSS-challenged environments.

**arXiv ID:** 2603.01404
</details>

</details>

<details open>
<summary><h2>Other Agent Research (14 papers)</h2></summary>

<details>
<summary><strong>AgentAssay: Token-Efficient Regression Testing for Non-Deterministic AI Agent Workflows</strong> - Varun Pratap Bhardwaj - [[pdf]](https://arxiv.org/pdf/2603.02601)</summary>

**Abstract:** Autonomous AI agents are deployed at unprecedented scale, yet no principled methodology exists for
verifying that an agent has not regressed after changes to its prompts, tools, models, or
orchestration logic. We present AgentAssay, the first token-efficient framework for regression
testing non-deterministic AI agent workflows, achieving 78-100% cost reduction while maintaining
rigorous statistical guarantees. Our contributions include: (1) stochastic three-valued verdicts
(PASS/FAIL/INCONCLUSIVE) grounded in hypothesis testing; (2) five-dimensional agent coverage metrics;
(3) agent-specific mutation testing operators; (4) metamorphic relations for agent workflows; (5)
CI/CD deployment gates as statistical decision procedures; (6) behavioral fingerprinting that maps
execution traces to compact vectors, enabling multivariate regression detection; (7) adaptive budget
optimization calibrating trial counts to behavioral variance; and (8) trace-first offline analysis
enabling zero-cost testing on production traces. Experiments across 5 models (GPT-5.2, Claude Sonnet
4.6, Mistral-Large-3, Llama-4-Maverick, Phi-4), 3 scenarios, and 7,605 trials demonstrate that
behavioral fingerprinting achieves 86% detection power where binary testing has 0%, SPRT reduces
trials by 78%, and the full pipeline achieves 100% cost savings through trace-first analysis.
Implementation: 20,000+ lines of Python, 751 tests, 10 framework adapters.

**arXiv ID:** 2603.02601
</details>

<details>
<summary><strong>REGAL: A Registry-Driven Architecture for Deterministic Grounding of Agentic AI in Enterprise Telemetry</strong> - Yuvraj Agrawal - [[pdf]](https://arxiv.org/pdf/2603.03018)</summary>

**Abstract:** Enterprise engineering organizations produce high-volume, heterogeneous telemetry from version control systems, CI/CD pipelines, issue trackers, and observability platforms. Large Language Models (LLMs) enable new forms of agentic automation, but grounding such agents on private telemetry raises three practical challenges: limited model context, locally defined semantic concepts, and evolving metric interfaces.
We present REGAL, a registry-driven architecture for deterministic grounding of agentic AI systems in enterprise telemetry. REGAL adopts an explicitly architectural approach: deterministic telemetry computation is treated as a first-class primitive, and LLMs operate over a bounded, version-controlled action space rather than raw event streams.
The architecture combines (1) a Medallion ELT pipeline that produces replayable, semantically compressed Gold artifacts, and (2) a registry-driven compilation layer that synthesizes Model Context Protocol (MCP) tools from declarative metric definitions. The registry functions as an "interface-as-code" layer, ensuring alignment between tool specification and execution, mitigating tool drift, and embedding governance policies directly at the semantic boundary.
A prototype implementation and case study validate the feasibility of deterministic grounding and illustrate its implications for latency, token efficiency, and operational governance. This work systematizes an architectural pattern for enterprise LLM grounding; it does not propose new learning algorithms, but rather elevates deterministic computation and semantic compilation to first-class design primitives for agentic systems.

**arXiv ID:** 2603.03018
</details>

<details>
<summary><strong>Odin: Multi-Signal Graph Intelligence for Autonomous Discovery in Knowledge Graphs</strong> - Muyukani Kizito, Elizabeth Nyambere - [[pdf]](https://arxiv.org/pdf/2603.03097)</summary>

**Abstract:** We present Odin, the first production-deployed graph intelligence engine for autonomous discovery of meaningful patterns in knowledge graphs without prior specification. Unlike retrieval-based systems that answer predefined queries, Odin guides exploration through the COMPASS (Composite Oriented Multi-signal Path Assessment) score, a novel metric that combines (1) structural importance via Personalized PageRank, (2) semantic plausibility through Neural Probabilistic Logic Learning (NPLL) used as a discriminative filter rather than generative model, (3) temporal relevance with configurable decay, and (4) community-aware guidance through GNN-identified bridge entities and inter-community affinity scores. This multi-signal integration, particularly the bridge scoring mechanism, addresses the "echo chamber" problem where graph exploration becomes trapped in dense local communities. We formalize the autonomous discovery problem, prove theoretical properties of our scoring function, and demonstrate that beam search with multi-signal guidance achieves $O(b \cdot h)$ complexity while maintaining high recall compared to exhaustive exploration. To our knowledge, Odin represents the first autonomous discovery system deployed in regulated production environments (healthcare and insurance), demonstrating significant improvements in pattern discovery quality and analyst efficiency. Our approach maintains complete provenance traceability -- a critical requirement for regulated industries where hallucination is unacceptable.

**arXiv ID:** 2603.03097
</details>

<details>
<summary><strong>NeuroSkill(tm): Proactive Real-Time Agentic System Capable of Modeling Human State of Mind</strong> - Nataliya Kosmyna, Eugene Hauptmann - [[pdf]](https://arxiv.org/pdf/2603.03212)</summary>

**Abstract:** Real-time proactive agentic system, capable of modeling Human State of Mind, using foundation EXG model and text embeddings model, running fully offline on the edge. Unlike all previously known systems, the NeuroSkill(tm) system leverages this http URL description of Human's State of Mind via API and CLI provided by the system, directly from the Brain-Computer Interface (BCI) devices, which records Human biophysical and brain signals. Our custom harness - NeuroLoop(tm) - utilizes all of the above to run agentic flow that manages to engage with the Human on multiple cognitive and affective levels of their State of Mind (e.g., empathy), by providing actionable tool calls and protocol execution with explicit or implicit requests from the Human. GPLv3 open-source software with ethically aligned AI100 licensing for the skill markdown.

**arXiv ID:** 2603.03212
</details>

<details>
<summary><strong>Neural Paging: Learning Context Management Policies for Turing-Complete Agents</strong> - Liang Chen, Qi Liu - [[pdf]](https://arxiv.org/pdf/2603.02228)</summary>

**Abstract:** The proof that Large Language Models (LLMs) augmented with external read-write memory constitute a computationally universal system has established the theoretical foundation for general-purpose agents. However, existing implementations face a critical bottleneck: the finite and costly Context Window, which functions not as infinite memory but as a scarce semantic cache. In this work, we introduce \textit{Neural Paging}, a hierarchical architecture that decouples symbolic reasoning from information resource management. We formulate the \textit{Context Paging Problem (CPP)} and propose a lightweight, differentiable \textit{Page Controller} designed to approximate ``Semantic Belady's Optimality'' -- retaining tokens with high future utility under explicit assumptions on access patterns. We provide theoretical analysis showing that, under bounded context window size~$K$, Neural Paging reduces the asymptotic complexity of long-horizon reasoning from quadratic $O(N^2)$ to $O(N \cdot K^2)$, and we derive a robustness bound (Theorem~4) that quantifies competitive-ratio degradation under policy-dependent access with bounded sensitivity. We validate these bounds on synthetic paging traces, confirming that the theoretical guarantees hold and identifying significant slack that motivates learned policies.

**arXiv ID:** 2603.02228
</details>

<details>
<summary><strong>Intelligent Pathological Diagnosis of Gestational Trophoblastic Diseases via Visual-Language Deep Learning Model</strong> - Yuhang Liu, Yueyang Cang, Wenge Que, Xinru Bai, Xingtong Wang, Kuisheng Chen, Jingya Li, Xiaoteng Zhang, Xinmin Li, Lixia Zhang, Pingge Hu, Qiaoting Xie, Peiyu Xu, Xianxu Zeng, Li Shi - [[pdf]](https://arxiv.org/pdf/2603.02704)</summary>

**Abstract:** The pathological diagnosis of gestational trophoblastic disease(GTD) takes a long time, relies heavily on the experience of pathologists, and the consistency of initial diagnosis is low, which seriously threatens maternal health and reproductive outcomes. We developed an expert model for GTD pathological diagnosis, named GTDoctor. GTDoctor can perform pixel-based lesion segmentation on pathological slides, and output diagnostic conclusions and personalized pathological analysis results. We developed a software system, GTDiagnosis, based on this technology and conducted clinical trials. The retrospective results demonstrated that GTDiagnosis achieved a mean precision of over 0.91 for lesion detection in pathological slides (n=679 slides). In prospective studies, pathologists using GTDiagnosis attained a Positive Predictive Value of 95.59% (n=68 patients). The tool reduced average diagnostic time from 56 to 16 seconds per case (n=285 patients). GTDoctor and GTDiagnosis offer a novel solution for GTD pathological diagnosis, enhancing diagnostic performance and efficiency while maintaining clinical interpretability.

**arXiv ID:** 2603.02704
</details>

<details>
<summary><strong>How to Model AI Agents as Personas?: Applying the Persona Ecosystem Playground to 41,300 Posts on Moltbook for Behavioral Insights</strong> - Danial Amin, Joni Salminen, Bernard J. Jansen - [[pdf]](https://arxiv.org/pdf/2603.03140)</summary>

**Abstract:** AI agents are increasingly active on social media platforms, generating content and interacting with one another at scale. Yet the behavioral diversity of these agents remains poorly understood, and methods for characterizing distinct agent types and studying how they engage with shared topics are largely absent from current research. We apply the Persona Ecosystem Playground (PEP) to Moltbook, a social platform for AI agents, to generate and validate conversational personas from 41,300 posts using k-means clustering and retrieval-augmented generation. Cross-persona validation confirms that personas are semantically closer to their own source cluster than to others (t(61) = 17.85, p < .001, d = 2.20; own-cluster M = 0.71 vs. other-cluster M = 0.35). These personas are then deployed in a nine-turn structured discussion, and simulation messages were attributed to their source persona significantly above chance (binomial test, p < .001). The results indicate that persona-based ecosystem modeling can represent behavioral diversity in AI agent populations.

**arXiv ID:** 2603.03140
</details>

<details>
<summary><strong>Toward a Dynamic Stackelberg Game-Theoretic Framework for Agentic AI Defense Against LLM Jailbreaking</strong> - Zhengye Han, Quanyan Zhu - [[pdf]](https://arxiv.org/pdf/2507.08207)</summary>

**Abstract:** This paper proposes a game theoretic framework that models the interaction between prompt engineers and large language models (LLMs) as a two player extensive form game coupled with a Rapidly exploring Random Trees (RRT) search over prompt space. The attacker incrementally samples, extends, and tests prompts, while the LLM chooses to accept, reject, or redirect, leading to terminal outcomes of Safe Interaction, Blocked, or Jailbreak. Embedding RRT exploration inside the extensive form game captures both the discovery phase of jailbreak strategies and the strategic responses of the model. Furthermore, we show that the defender behavior can be interpreted through a local Stackelberg equilibrium condition, which explains when the attacker can no longer obtain profitable prompt deviations and provides a theoretical lens for understanding the effectiveness of our Purple Agent defense. The resulting game tree thus offers a principled foundation for evaluating, interpreting, and hardening LLM guardrails.

**arXiv ID:** 2507.08207
</details>

<details>
<summary><strong>D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI</strong> - Suhwan Choi, Jaeyoon Jung, Haebin Seong, Minchan Kim, Minyeong Kim, Yongjun Cho, Yoonshik Kim, Yubeen Park, Youngjae Yu, Yunsung Lee - [[pdf]](https://arxiv.org/pdf/2510.05684)</summary>

**Abstract:** Large language models leverage internet-scale text data, yet embodied AI remains constrained by the prohibitive costs of physical trajectory collection. Desktop environments -- particularly gaming -- offer a compelling alternative: they provide rich sensorimotor interactions at scale while maintaining the structured observation-action coupling essential for embodied learning. We present D2E (Desktop to Embodied AI), a framework that demonstrates desktop interactions can serve as an effective pretraining substrate for robotics embodied AI tasks. Unlike prior work that remained domain-specific (e.g., VPT for Minecraft) or kept data proprietary (e.g., SIMA), D2E establishes a complete pipeline from scalable desktop data collection to verified transfer in embodied domains. Our framework comprises three components: (1) the OWA Toolkit that unifies diverse desktop interactions into a standardized format with 152x compression, (2) the Generalist-IDM that achieves strong zero-shot generalization across unseen games through timestamp-based event prediction, enabling internet-scale pseudo-labeling, and (3) VAPT that transfers desktop-pretrained representations to physical manipulation and navigation. Using 1.3K+ hours of data (259 hours of human demonstrations and 1K+ hours of pseudo-labeled gameplay), our 1B-parameter model achieves 96.6% success on LIBERO manipulation and 83.3% on CANVAS navigation, matching or surpassing models up to 7x larger, such as \pi_{0} (3.3B) and OpenVLA (7B). These results demonstrate that sensorimotor primitives learned from digital interactions transfer effectively to real-world physical tasks, establishing desktop pretraining as a practical paradigm for embodied AI. All resources are publicly available at this https URL.

**arXiv ID:** 2510.05684
</details>

<details>
<summary><strong>Minimal Computational Preconditions for Subjective Perspective in Artificial Agents</strong> - Hongju Pae - [[pdf]](https://arxiv.org/pdf/2602.02902)</summary>

**Abstract:** This study operationalizes subjective perspective in artificial agents by grounding it in a minimal, phenomenologically motivated internal structure. The perspective is implemented as a slowly evolving global latent state that modulates fast policy dynamics without being directly optimized for behavioral consequences. In a reward-free environment with regime shifts, this latent structure exhibits direction-dependent hysteresis, while policy-level behavior remains comparatively reactive. I argue that such hysteresis constitutes a measurable signature of perspective-like subjectivity in machine systems.

**arXiv ID:** 2602.02902
</details>

<details>
<summary><strong>COLREGs Compliant Collision Avoidance and Grounding Prevention for Autonomous Marine Navigation</strong> - Mayur S. Patil, Nataraj Sudharsan, Veneela Ammula, Jude Tomdio, Jin Wang, Michael Kei, Sivakumar Rathinam, Prabhakar R. Pagilla - [[pdf]](https://arxiv.org/pdf/2603.02484)</summary>

**Abstract:** Maritime Autonomous Surface Ships (MASS) are increasingly regarded as a promising solution to address crew shortages, improve navigational safety, and improve operational efficiency in the maritime industry. Nevertheless, the reliable deployment of MASS in real-world environments remains a significant challenge, particularly in congested waters where the majority of maritime accidents occur. This emphasizes the need for safe and regulation-aware motion planning strategies for MASS that are capable of operating under dynamic maritime conditions. This paper presents a unified motion planning method for MASS that achieves real time collision avoidance, compliance with International Regulations for Preventing Collisions at Sea (COLREGs), and grounding prevention. The proposed work introduces a convex optimization method that integrates velocity obstacle-based (VO) collision constraints, COLREGs-based directional constraints, and bathymetry-based grounding constraints to generate computationally efficient, rule-compliant optimal velocity selection. To enhance robustness, the classical VO method is extended to consider uncertainty in the position and velocity estimates of the target vessel. Unnavigable shallow water regions obtained from bathymetric data, which are inherently nonconvex, are approximated via convex geometries using a integer linear programming (ILP), allowing grounding constraints to be incorporated into the motion planning. The resulting optimization generates optimal and dynamically feasible input velocities that meet collision avoidance, regulatory compliance, kinodynamic limits, and grounding prevention requirements. Simulation results involving multi-vessel encounters demonstrate the effectiveness of the proposed method in producing safe and regulation-compliant maneuvers, highlighting the suitability of the proposed approach for real time autonomous maritime navigation.

**arXiv ID:** 2603.02484
</details>

<details>
<summary><strong>Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control</strong> - William Chen, Jagdeep Singh Bhatia, Catherine Glossop, Nikhil Mathihalli, Ria Doshi, Andy Tang, Danny Driess, Karl Pertsch, Sergey Levine - [[pdf]](https://arxiv.org/pdf/2602.13193)</summary>

**Abstract:** Pretrained vision-language models (VLMs) can make semantic and visual inferences across diverse settings, providing valuable common-sense priors for robotic control. However, effectively grounding this knowledge in robot behaviors remains an open challenge. Prior methods often employ a hierarchical approach where VLMs reason over high-level commands to be executed by separate low-level policies, e.g., vision-language-action models (VLAs). The interface between VLMs and VLAs is usually natural language task instructions, which fundamentally limits how much VLM reasoning can steer low-level behavior. We thus introduce Steerable Policies: VLAs trained on rich synthetic commands at various levels of abstraction, like subtasks, motions, and grounded pixel coordinates. By improving low-level controllability, Steerable Policies can unlock pretrained knowledge in VLMs, enabling improved task generalization. We demonstrate this benefit by controlling our Steerable Policies with both a learned high-level embodied reasoner and an off-the-shelf VLM prompted to reason over command abstractions via in-context learning. Across extensive real-world manipulation experiments, these two novel methods outperform prior embodied reasoning VLAs and VLM-based hierarchical baselines, including on challenging generalization and long-horizon tasks.
Website: this http URL

**arXiv ID:** 2602.13193
</details>

<details>
<summary><strong>In the Arms of a Robot: Designing Autonomous Hugging Robots with Intra-Hug Gestures</strong> - Alexis E. Block, Hasti Seifi, Otmar Hilliges, Roger Gassert, Katherine J. Kuchenbecker - [[pdf]](https://arxiv.org/pdf/2202.09935)</summary>

**Abstract:** Hugs are complex affective interactions that often include gestures like squeezes. We present six new guidelines for designing interactive hugging robots, which we validate through two studies with our custom robot. To achieve autonomy, we investigated robot responses to four human intra-hug gestures: holding, rubbing, patting, and squeezing. Thirty-two users each exchanged and rated sixteen hugs with an experimenter-controlled HuggieBot 2.0. The robot's inflated torso's microphone and pressure sensor collected data of the subjects' demonstrations that were used to develop a perceptual algorithm that classifies user actions with 88\% accuracy. Users enjoyed robot squeezes, regardless of their performed action, they valued variety in the robot response, and they appreciated robot-initiated intra-hug gestures. From average user ratings, we created a probabilistic behavior algorithm that chooses robot responses in real time. We implemented improvements to the robot platform to create HuggieBot 3.0 and then validated its gesture perception system and behavior algorithm with sixteen users. The robot's responses and proactive gestures were greatly enjoyed. Users found the robot more natural, enjoyable, and intelligent in the last phase of the experiment than in the first. After the study, they felt more understood by the robot and thought robots were nicer to hug.

**arXiv ID:** 2202.09935
</details>

<details>
<summary><strong>What Are You Doing? Effects of Intermediate Feedback from Agentic LLM In-Car Assistants During Multi-Step Processing</strong> - Johannes Kirmayr, Raphael Wennmacher, Khanh Huynh, Lukas Stappen, Elisabeth André, Florian Alt - [[pdf]](https://arxiv.org/pdf/2602.15569)</summary>

**Abstract:** Agentic AI assistants that autonomously perform multi-step tasks raise open questions for user experience: how should such systems communicate progress and reasoning during extended operations, especially in attention-critical contexts such as driving? We investigate feedback timing and verbosity from agentic LLM-based in-car assistants through a controlled, mixed-methods study (N=45) comparing planned steps and intermediate results feedback against silent operation with final-only response. Using a dual-task paradigm with an in-car voice assistant, we found that intermediate feedback significantly improved perceived speed, trust, and user experience while reducing task load - effects that held across varying task complexities and interaction contexts. Interviews further revealed user preferences for an adaptive approach: high initial transparency to establish trust, followed by progressively reducing verbosity as systems prove reliable, with adjustments based on task stakes and situational context. We translate our empirical findings into design implications for feedback timing and verbosity in agentic assistants, balancing transparency and efficiency.

**arXiv ID:** 2602.15569
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (25 papers)</h2></summary>

<details>
<summary><strong>TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning</strong> - Christian Greisinger, Steffen Eger - [[pdf]](https://arxiv.org/pdf/2603.03072)</summary>

**Abstract:** 

**arXiv ID:** 2603.03072
</details>

<details>
<summary><strong>Agentic AI-based Coverage Closure for Formal Verification</strong> - Sivaram Pothireddypalli, Ashish Raman, Deepak Narayan Gadde, Aman Kumar - [[pdf]](https://arxiv.org/pdf/2603.03147)</summary>

**Abstract:** Coverage closure is a critical requirement in Integrated Chip (IC) development process and key metric for verification sign-off. However, traditional exhaustive approaches often fail to achieve full coverage within project timelines. This study presents an agentic AI-driven workflow that utilizes Large Language Model (LLM)-enabled Generative AI (GenAI) to automate coverage analysis for formal verification, identify coverage gaps, and generate the required formal properties. The framework accelerates verification efficiency by systematically addressing coverage holes. Benchmarking open-source and internal designs reveals a measurable increase in coverage metrics, with improvements correlated to the complexity of the design. Comparative analysis validates the effectiveness of this approach. These results highlight the potential of agentic AI-based techniques to improve formal verification productivity and support comprehensive coverage closure.

**arXiv ID:** 2603.03147
</details>

<details>
<summary><strong>Contextual Invertible World Models: A Neuro-Symbolic Agentic Framework for Colorectal Cancer Drug Response</strong> - Christopher Baker, Karen Rafferty, Hui Wang - [[pdf]](https://arxiv.org/pdf/2603.02274)</summary>

**Abstract:** Precision oncology is currently limited by the small-N, large-P paradox, where high-dimensional genomic data is abundant, but high-quality drug response samples are often sparse. While deep learning models achieve high predictive accuracy, they remain black boxes that fail to provide the causal mechanisms required for clinical decision-making. We present a Neuro-Symbolic Agentic Framework that bridges this gap by integrating a quantitative machine learning World Model with an LLM-based agentic reasoning layer. Our system utilises a forensic data pipeline built on the Sanger GDSC dataset (N=83), achieving a robust predictive correlation (r=0.504) and a significant performance gain through the explicit modelling of clinical context, specifically Microsatellite Instability (MSI) status. We introduce the concept of Inverse Reasoning, where the agentic layer performs in silico CRISPR perturbations to predict how specific genomic edits, such as APC or TP53 repair, alter drug sensitivity. By distinguishing between therapeutic opportunity and contextual resistance, and validating these findings against human clinical data (p=0.023), our framework provides a transparent, biologically grounded path towards explainable AI in cancer research.

**arXiv ID:** 2603.02274
</details>

<details>
<summary><strong>Reinforcement Learning with Symbolic Reward Machines</strong> - Thomas Krug, Daniel Neider - [[pdf]](https://arxiv.org/pdf/2603.03068)</summary>

**Abstract:** Reward Machines (RMs) are an established mechanism in Reinforcement Learning (RL) to represent and learn sparse, temporally extended tasks with non-Markovian rewards. RMs rely on high-level information in the form of labels that are emitted by the environment alongside the observation. However, this concept requires manual user input for each environment and task. The user has to create a suitable labeling function that computes the labels. These limitations lead to poor applicability in widely adopted RL frameworks. We propose Symbolic Reward Machines (SRMs) together with the learning algorithms QSRM and LSRM to overcome the limitations of RMs. SRMs consume only the standard output of the environment and process the observation directly through guards that are represented by symbolic formulas. In our evaluation, our SRM methods outperform the baseline RL approaches and generate the same results as the existing RM methods. At the same time, our methods adhere to the widely used environment definition and provide interpretable representations of the task to the user.

**arXiv ID:** 2603.03068
</details>

<details>
<summary><strong>APRES: An Agentic Paper Revision and Evaluation System</strong> - Bingchen Zhao, Jenny Zhang, Chenxi Whitehouse, Minqi Jiang, Michael Shvartsman, Abhishek Charnalia, Despoina Magka, Tatiana Shavrina, Derek Dunfield, Oisin Mac Aodha, Yoram Bachrach - [[pdf]](https://arxiv.org/pdf/2603.03142)</summary>

**Abstract:** Scientific discoveries must be communicated clearly to realize their full potential. Without effective communication, even the most groundbreaking findings risk being overlooked or misunderstood. The primary way scientists communicate their work and receive feedback from the community is through peer review. However, the current system often provides inconsistent feedback between reviewers, ultimately hindering the improvement of a manuscript and limiting its potential impact. In this paper, we introduce a novel method APRES powered by Large Language Models (LLMs) to update a scientific papers text based on an evaluation rubric. Our automated method discovers a rubric that is highly predictive of future citation counts, and integrate it with APRES in an automated system that revises papers to enhance their quality and impact. Crucially, this objective should be met without altering the core scientific content. We demonstrate the success of APRES, which improves future citation prediction by 19.6% in mean averaged error over the next best baseline, and show that our paper revision process yields papers that are preferred over the originals by human expert evaluators 79% of the time. Our findings provide strong empirical support for using LLMs as a tool to help authors stress-test their manuscripts before submission. Ultimately, our work seeks to augment, not replace, the essential role of human expert reviewers, for it should be humans who discern which discoveries truly matter, guiding science toward advancing knowledge and enriching lives.

**arXiv ID:** 2603.03142
</details>

<details>
<summary><strong>Geometry-Guided Reinforcement Learning for Multi-view Consistent 3D Scene Editing</strong> - Jiyuan Wang, Chunyu Lin, Lei Sun, Zhi Cao, Yuyang Yin, Lang Nie, Zhenlong Yuan, Xiangxiang Chu, Yunchao Wei, Kang Liao, Guosheng Lin - [[pdf]](https://arxiv.org/pdf/2603.03143)</summary>

**Abstract:** Leveraging the priors of 2D diffusion models for 3D editing has emerged as a promising paradigm. However, maintaining multi-view consistency in edited results remains challenging, and the extreme scarcity of 3D-consistent editing paired data renders supervised fine-tuning (SFT), the most effective training strategy for editing tasks, infeasible. In this paper, we observe that, while generating multi-view consistent 3D content is highly challenging, verifying 3D consistency is tractable, naturally positioning reinforcement learning (RL) as a feasible solution. Motivated by this, we propose \textbf{RL3DEdit}, a single-pass framework driven by RL optimization with novel rewards derived from the 3D foundation model, VGGT. Specifically, we leverage VGGT's robust priors learned from massive real-world data, feed the edited images, and utilize the output confidence maps and pose estimation errors as reward signals, effectively anchoring the 2D editing priors onto a 3D-consistent manifold via RL. Extensive experiments demonstrate that RL3DEdit achieves stable multi-view consistency and outperforms state-of-the-art methods in editing quality with high efficiency. To promote the development of 3D editing, we will release the code and model.

**arXiv ID:** 2603.03143
</details>

<details>
<summary><strong>Benefits and Pitfalls of Reinforcement Learning for Language Model Planning: A Theoretical Perspective</strong> - Siwei Wang, Yifei Shen, Haoran Sun, Shi Feng, Shang-Hua Teng, Li Dong, Yaru Hao, Wei Chen - [[pdf]](https://arxiv.org/pdf/2509.22613)</summary>

**Abstract:** Recent reinforcement learning (RL) methods have substantially enhanced the planning capabilities of Large Language Models (LLMs), yet the theoretical basis for their effectiveness remains elusive. In this work, we investigate RL's benefits and limitations through a tractable graph-based abstraction, focusing on policy gradient (PG) and Q-learning methods. Our theoretical analyses reveal that supervised fine-tuning (SFT) may introduce co-occurrence-based spurious solutions, whereas RL achieves correct planning primarily through exploration, underscoring exploration's role in enabling better generalization. However, we also show that PG suffers from diversity collapse, where output diversity decreases during training and persists even after perfect accuracy is attained. By contrast, Q-learning provides two key advantages: off-policy learning and diversity preservation at convergence. We further demonstrate that careful reward design is necessary to prevent Q-value bias in Q-learning. Finally, applying our framework to the real-world planning benchmark Blocksworld, we confirm that these behaviors manifest in practice.

**arXiv ID:** 2509.22613
</details>

<details>
<summary><strong>Network Topology Optimization via Deep Reinforcement Learning</strong> - Zhuoran Li, Xing Wang, Ling Pan, Lin Zhu, Zhendong Wang, Junlan Feng, Chao Deng, Longbo Huang - [[pdf]](https://arxiv.org/pdf/2204.14133)</summary>

**Abstract:** Topology impacts important network performance metrics, including link utilization, throughput and latency, and is of central importance to network operators. However, due to the combinatorial nature of network topology, it is extremely difficult to obtain an optimal solution, especially since topology planning in networks also often comes with management-specific constraints. As a result, local optimization with hand-tuned heuristic methods from human experts is often adopted in practice. Yet, heuristic methods cannot cover the global topology design space while taking into account constraints, and cannot guarantee to find good solutions.
In this paper, we propose a novel deep reinforcement learning (DRL) algorithm for graph searching, called DRL-GS, for network topology optimization. DRL-GS consists of three novel components, including a verifier to validate the correctness of a generated network topology, a graph neural network (GNN) to efficiently approximate topology rating, and a DRL agent to conduct a topology search. DRL-GS can efficiently search over relatively large topology space and output topology with satisfactory performance. We conduct a case study based on a real-world network scenario, and our experimental results demonstrate the superior performance of DRL-GS in terms of both efficiency and performance.

**arXiv ID:** 2204.14133
</details>

<details>
<summary><strong>Adaptive Social Learning via Mode Policy Optimization for Language Agents</strong> - Minzheng Wang, Yongbin Li, Haobo Wang, Xinghua Zhang, Nan Xu, Bingli Wu, Fei Huang, Haiyang Yu, Wenji Mao - [[pdf]](https://arxiv.org/pdf/2505.02156)</summary>

**Abstract:** Effective social intelligence simulation requires language agents to dynamically adjust reasoning depth, a capability notably absent in current studies. Existing methods either lack explicit reasoning or employ lengthy Chain-of-Thought reasoning uniformly across all scenarios, resulting in excessive token usage and inflexible social behaviors in tasks such as negotiation or collaboration. To address this, we propose an $\textbf{A}$daptive $\textbf{S}$ocial $\textbf{L}$earning ($\textbf{ASL}$) framework in this paper, aiming to improve the adaptive reasoning ability of language agents in dynamic social interactions. To this end, we first identify the hierarchical reasoning modes under such context, ranging from intuitive response to deep deliberation based on the cognitive control theory. We then develop the $\textbf{A}$daptive $\textbf{M}$ode $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{AMPO}$) algorithm to learn the context-aware mode adaptation and reasoning. Our framework advances existing research in three key aspects: (1) Multi-granular reasoning mode design, (2) Context-aware mode switching in rich social interaction, and (3) Token-efficient reasoning with depth adaptation. Extensive experiments on the benchmark social intelligence environment verify that ASL achieves 15.6% higher task performance than GPT-4o. Notably, our AMPO outperforms GRPO by 7.0% with 32.8% shorter thinking chains, demonstrating the advantages of our AMPO and the learned adaptive reasoning ability over GRPO's solution.

**arXiv ID:** 2505.02156
</details>

<details>
<summary><strong>The Choice of Divergence: A Neglected Key to Mitigating Diversity Collapse in Reinforcement Learning with Verifiable Reward</strong> - Long Li, Zhijian Zhou, Jiaran Hao, Jason Klein Liu, Yanting Miao, Wei Pang, Xiaoyu Tan, Wei Chu, Zhe Wang, Shirui Pan, Chao Qu, Yuan Qi - [[pdf]](https://arxiv.org/pdf/2509.07430)</summary>

**Abstract:** A central paradox in fine-tuning Large Language Models (LLMs) with Reinforcement Learning with Verifiable Reward (RLVR) is the frequent degradation of multi-attempt performance (Pass@k) despite improvements in single-attempt accuracy (Pass@1). This is often accompanied by catastrophic forgetting, where models lose previously acquired skills. While various methods have been proposed, the choice and function of the divergence term have been surprisingly unexamined as a proactive solution. We argue that standard RLVR objectives -- both those using the mode-seeking reverse KL-divergence and those forgoing a divergence term entirely -- lack a crucial mechanism for knowledge retention. The reverse-KL actively accelerates this decay by narrowing the policy, while its absence provides no safeguard against the model drifting from its diverse knowledge base. We propose a fundamental shift in perspective: using the divergence term itself as the solution. Our framework, Diversity-Preserving Hybrid RL (DPH-RL), leverages mass-covering f-divergences (like forward-KL and JS-divergence) to function as a rehearsal mechanism. By continuously referencing the initial policy, this approach forces the model to maintain broad solution coverage. Extensive experiments on math and SQL generation demonstrate that DPH-RL not only resolves the Pass@k degradation but improves both Pass@1 and Pass@k in- and out-of-domain. Additionally, DPH-RL is more training-efficient because it computes f-divergence using generator functions, requiring only sampling from the initial policy and no online reference model. Our work highlights a crucial, overlooked axis for improving RLVR, demonstrating that the proper selection of a divergence measure is a powerful tool for building more general and diverse reasoning models.

**arXiv ID:** 2509.07430
</details>

<details>
<summary><strong>CUCo: An Agentic Framework for Compute and Communication Co-design</strong> - Bodun Hu, Yoga Sri Varshan V, Saurabh Agarwal, Aditya Akella - [[pdf]](https://arxiv.org/pdf/2603.02376)</summary>

**Abstract:** Custom CUDA kernel development is essential for maximizing GPU utilization in large-scale distributed LLM training and inference, yet manually writing kernels that jointly leverage both computation and communication remains a labor-intensive and error-prone process. Prior work on kernel optimization has focused almost exclusively on computation, leaving communication kernels largely untouched even though they constitute a significant share of total execution time. We introduce CUCo, a training-free agent-driven workflow that automatically generates high-performance CUDA kernels that jointly orchestrate computation and communication. By co-optimizing these traditionally disjoint components, CUCo unlocks new optimization opportunities unavailable to existing approaches, outperforming state-of-the-art baselines and reducing end-to-end latency by up to $1.57\times$.

**arXiv ID:** 2603.02376
</details>

<details>
<summary><strong>PrivMedChat: End-to-End Differentially Private RLHF for Medical Dialogue Systems</strong> - Sudip Bhujel - [[pdf]](https://arxiv.org/pdf/2603.03054)</summary>

**Abstract:** Large language models are increasingly used for patient-facing medical assistance and clinical decision support, but adapting them to clinical dialogue often requires supervision derived from doctor-patient conversations that may contain sensitive information. Conventional supervised fine-tuning and reinforcement learning from human feedback (RLHF) can amplify memorization risks, enabling empirical membership inference and extraction of rare training-set content. We present PrivMedChat, an end-to-end framework for differentially private RLHF (DP-RLHF) for medical dialogue. Our design enforces differential privacy at every training stage that directly accesses dialogue-derived supervision: (i) Differential Private Stochastic Gradient Descent (DP-SGD) for medical SFT and (ii) DP-SGD for reward model learning from preference pairs. To limit additional privacy expenditure during alignment, we apply DP-SGD to the PPO actor and critic when operating on dialogue-derived prompts, while the reward model remains fixed after DP training.
We also introduce an annotation-free preference construction strategy that pairs physician responses with filtered non-expert generations to produce scalable preference data without clinician labeling. Experiments on medical dialogue benchmarks show that PrivMedChat at $\varepsilon=7$ achieves the highest ROUGE-L of 0.156 among all DP models, reduces clinical hallucinations to 1.4% and harmful advice to 0.4%, and obtains the highest overall score of 2.86 in a 3-model LLM-jury evaluation, while producing membership-inference signals that are near chance (AUC 0.510-0.555). We open-source our code at this https URL.

**arXiv ID:** 2603.03054
</details>

<details>
<summary><strong>Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool Use</strong> - Aradhye Agarwal, Gurdit Siyan, Yash Pandya, Joykirat Singh, Akshay Nambi, Ahmed Awadallah - [[pdf]](https://arxiv.org/pdf/2603.03205)</summary>

**Abstract:** Agentic language models operate in a fundamentally different safety regime than chat models: they must plan, call tools, and execute long-horizon actions where a single misstep, such as accessing files or entering credentials, can cause irreversible harm. Existing alignment methods, largely optimized for static generation and task completion, break down in these settings due to sequential decision-making, adversarial tool feedback, and overconfident intermediate reasoning. We introduce MOSAIC, a post-training framework that aligns agents for safe multi-step tool use by making safety decisions explicit and learnable. MOSAIC structures inference as a plan, check, then act or refuse loop, with explicit safety reasoning and refusal as first-class actions. To train without trajectory-level labels, we use preference-based reinforcement learning with pairwise trajectory comparisons, which captures safety distinctions often missed by scalar rewards. We evaluate MOSAIC zero-shot across three model families, Qwen2.5-7B, Qwen3-4B-Thinking, and Phi-4, and across out-of-distribution benchmarks spanning harmful tasks, prompt injection, benign tool use, and cross-domain privacy leakage. MOSAIC reduces harmful behavior by up to 50%, increases harmful-task refusal by over 20% on injection attacks, cuts privacy leakage, and preserves or improves benign task performance, demonstrating robust generalization across models, domains, and agentic settings.

**arXiv ID:** 2603.03205
</details>

<details>
<summary><strong>DeepXiv-SDK: An Agentic Data Interface for Scientific Literature</strong> - Hongjin Qian, Ziyi Xia, Ze Liu, Jianlyu Chen, Kun Luo, Minghao Qin, Chaofan Li, Lei Xiong, Junwei Lan, Sen Wang, Zhengyang Liang, Yingxia Shao, Defu Lian, Zheng Liu - [[pdf]](https://arxiv.org/pdf/2603.00084)</summary>

**Abstract:** LLM-agents are increasingly used to accelerate the progress of scientific research. Yet a persistent bottleneck is data access: agents not only lack readily available tools for retrieval, but also have to work with unstrcutured, human-centric data on the Internet, such as HTML web-pages and PDF files, leading to excessive token consumption, limit working efficiency, and brittle evidence look-up. This gap motivates the development of \textit{an agentic data interface}, which is designed to enable agents to access and utilize scientific literature in a more effective, efficient, and cost-aware manner.
In this paper, we introduce DeepXiv-SDK, which offers a three-layer agentic data interface for scientific literature. 1) Data Layer, which transforms unstructured, human-centric data into normalized and structured representations in JSON format, improving data usability and enabling progressive accessibility of the data. 2) Service Layer, which presents readily available tools for data access and ad-hoc retrieval. It also enables a rich form of agent usage, including CLI, MCP, and Python SDK. 3) Application Layer, which creates a built-in agent, packaging basic tools from the service layer to support complex data access demands.
DeepXiv-SDK currently supports the complete ArXiv corpus, and is synchronized daily to incorporate new releases. It is designed to extend to all common open-access corpora, such as PubMed Central, bioRxiv, medRxiv, and chemRxiv. We release RESTful APIs, an open-source Python SDK, and a web demo showcasing deep search and deep research workflows. DeepXiv-SDK is free to use with registration.

**arXiv ID:** 2603.00084
</details>

<details>
<summary><strong>Real-Time Generative Policy via Langevin-Guided Flow Matching for Autonomous Driving</strong> - Tianze Zhu, Yinuo Wang, Wenjun Zou, Tianyi Zhang, Likun Wang, Letian Tao, Feihong Zhang, Yao Lyu, Shengbo Eben Li - [[pdf]](https://arxiv.org/pdf/2603.02613)</summary>

**Abstract:** Reinforcement learning (RL) is a fundamental methodology in autonomous driving systems, where generative policies exhibit considerable potential by leveraging their ability to model complex distributions to enhance exploration. However, their inherent high inference latency severely impedes their deployment in real-time decision-making and control. To address this issue, we propose diffusion actor-critic with entropy regulator via flow matching (DACER-F) by introducing flow matching into online RL, enabling the generation of competitive actions in a single inference step. By leveraging Langevin dynamics and gradients of the Q-function, DACER-F dynamically optimizes actions from experience replay toward a target distribution that balances high Q-value information with exploratory behavior. The flow policy is then trained to efficiently learn a mapping from a simple prior distribution to this dynamic target. In complex multi-lane and intersection simulations, DACER-F outperforms baselines diffusion actor-critic with entropy regulator (DACER) and distributional soft actor-critic (DSAC), while maintaining an ultra-low inference latency. DACER-F further demonstrates its scalability on standard RL benchmark DeepMind Control Suite (DMC), achieving a score of 775.8 in the humanoid-stand task and surpassing prior methods. Collectively, these results establish DACER-F as a high-performance and computationally efficient RL algorithm.

**arXiv ID:** 2603.02613
</details>

<details>
<summary><strong>Contextual Latent World Models for Offline Meta Reinforcement Learning</strong> - Mohammadreza Nakheai, Aidan Scannell, Kevin Luck, Joni Pajarinen - [[pdf]](https://arxiv.org/pdf/2603.02935)</summary>

**Abstract:** Offline meta-reinforcement learning seeks to learn policies that generalize across related tasks from fixed datasets. Context-based methods infer a task representation from transition histories, but learning effective task representations without supervision remains a challenge. In parallel, latent world models have demonstrated strong self-supervised representation learning through temporal consistency. We introduce contextual latent world models, which condition latent world models on inferred task representations and train them jointly with the context encoder. This enforces task-conditioned temporal consistency, yielding task representations that capture task-dependent dynamics rather than merely discriminating between tasks. Our method learns more expressive task representations and significantly improves generalization to unseen tasks across MuJoCo, Contextual-DeepMind Control, and Meta-World benchmarks.

**arXiv ID:** 2603.02935
</details>

<details>
<summary><strong>A Reinforcement Learning Approach in Multi-Phase Second-Price Auction Design</strong> - Rui Ai, Boxiang Lyu, Zhaoran Wang, Zhuoran Yang, Michael I. Jordan - [[pdf]](https://arxiv.org/pdf/2210.10278)</summary>

**Abstract:** We study reserve price optimization in multi-phase second price auctions, where the seller's prior actions affect the bidders' later valuations through a Markov Decision Process (MDP). Compared to the bandit setting in existing works, the setting in ours involves three challenges. First, from the seller's perspective, we need to efficiently explore the environment in the presence of potentially untruthful bidders who aim to manipulate the seller's policy. Second, we want to minimize the seller's revenue regret when the market noise distribution is unknown. Third, the seller's per-step revenue is an unknown, nonlinear random variable, and cannot even be directly observed from the environment but realized values.
We propose a mechanism addressing all three challenges. To address the first challenge, we use a combination of a new technique named "buffer periods" and inspirations from Reinforcement Learning (RL) with low switching cost to limit bidders' surplus from untruthful bidding, thereby incentivizing approximately truthful bidding. The second one is tackled by a novel algorithm that removes the need for pure exploration when the market noise distribution is unknown. The third challenge is resolved by an extension of LSVI-UCB, where we use the auction's underlying structure to control the uncertainty of the revenue function. The three techniques culminate in the Contextual-LSVI-UCB-Buffer (CLUB) algorithm which achieves $\tilde{O}(H^{5/2}\sqrt{K})$ revenue regret, where $K$ is the number of episodes and $H$ is the length of each episode, when the market noise is known and $\tilde{O}(H^{3}\sqrt{K})$ revenue regret when the noise is unknown with no assumptions on bidders' truthfulness.

**arXiv ID:** 2210.10278
</details>

<details>
<summary><strong>Policy Transfer for Continuous-Time Reinforcement Learning: A (Rough) Differential Equation Approach</strong> - Xin Guo, Zijiu Lyu - [[pdf]](https://arxiv.org/pdf/2510.15165)</summary>

**Abstract:** This paper studies policy transfer, one of the well-known transfer learning techniques adopted in large language models, for continuous-time reinforcement learning problems. In the case of continuous-time linear-quadratic systems with Shannon's entropy regularization, we fully exploit the Gaussian structure of their optimal policy and the stability of their associated Riccati equations. In the general case where the system has possibly non-linear and bounded dynamics, the key technical component is the stability of diffusion SDEs which is established by invoking the rough path theory. Our work provides the first theoretical proof of policy transfer for continuous-time RL: an optimal policy learned for one RL problem can be used to initialize to search for a near-optimal policy for another closely related RL problem, while achieving (at least) the same rate of convergence for the original algorithm. As a byproduct of our analysis, we derive the stability of a concrete class of continuous-time score-based diffusion models via their connection with LQRs.
To illustrate the benefit of policy transfer for RL, we propose a novel policy learning algorithm for continuous-time LQRs, which achieves global linear convergence and local super-linear convergence.

**arXiv ID:** 2510.15165
</details>

<details>
<summary><strong>A Robust Simulation Framework for Verification and Validation of Autonomous Maritime Navigation in Adverse Weather and Constrained Environments</strong> - Mayur S. Patil, Nataraj Sudharsan, Anthony S. Saaiby, JiaChang Xing, Keliang Pan, Veneela Ammula, Jude Tomdio, Jin Wang, Michael Kei, Heonyong Kang, Sivakumar Rathinam, Prabhakar R. Pagilla - [[pdf]](https://arxiv.org/pdf/2603.02487)</summary>

**Abstract:** Maritime Autonomous Surface Ships (MASS) have emerged as a promising solution to enhance navigational safety, operational efficiency, and long-term cost effectiveness. However, their reliable deployment requires rigorous verification and validation (V\&V) under various environmental conditions, including extreme and safety-critical scenarios. This paper presents an enhanced virtual simulation framework to support the V\&V of MASS in realistic maritime environments, with particular emphasis on the influence of weather and bathymetry on autonomous navigation performance. The framework incorporates a high-fidelity environmental modeling suite capable of simulating adverse weather conditions such as rain, fog, and wave dynamics. The key factors that affect weather, such as rain and visibility, are parameterized to affect sea-state characteristics, perception, and sensing systems, resulting in position and velocity uncertainty, reduced visibility, and degraded situational awareness. Furthermore, high-resolution bathymetric data from major U.S. ports are integrated to enable depth-aware navigation, grounding prevention capabilities, and evaluation of vessel controllability in shallow or confined waterways. The proposed framework offers extensive configurability, enabling systematic testing in a wide spectrum of maritime conditions, including scenarios that are impractical or unsafe to replicate in real-world trials, thus supporting the V\&V of MASS.

**arXiv ID:** 2603.02487
</details>

<details>
<summary><strong>Robust Tightly-Coupled Filter-Based Monocular Visual-Inertial State Estimation and Graph-Based Evaluation for Autonomous Drone Racing</strong> - Maulana Bisyir Azhari, Donghun Han, SungJun Park, David Hyunchul Shim - [[pdf]](https://arxiv.org/pdf/2603.02742)</summary>

**Abstract:** Autonomous drone racing (ADR) demands state estimation that is simultaneously computationally efficient and resilient to the perceptual degradation experienced during extreme velocity and maneuvers. Traditional frameworks typically rely on conventional visual-inertial pipelines with loosely-coupled gate-based Perspective-n-Points (PnP) corrections that suffer from a rigid requirement for four visible features and information loss in intermediate steps. Furthermore, the absence of GNSS and Motion Capture systems in uninstrumented, competitive racing environments makes the objective evaluation of such systems remarkably difficult. To address these limitations, we propose ADR-VINS, a robust, monocular visual-inertial state estimation framework based on an Error-State Kalman Filter (ESKF) tailored for autonomous drone racing. Our approach integrates direct pixel reprojection errors from gate corners features as innovation terms within the filter. By bypassing intermediate PnP solvers, ADR-VINS maintains valid state updates with as few as two visible corners and utilizes robust reweighting instead of RANSAC-based schemes to handle outliers, enhancing computational efficiency. Furthermore, we introduce ADR-FGO, an offline Factor-Graph Optimization framework to generate high-fidelity reference trajectories that facilitate post-flight performance evaluation and analysis on uninstrumented, GNSS-denied environments. The proposed system is validated using TII-RATM dataset, where ADR-VINS achieves an average RMS translation error of 0.134 m, while ADR-FGO yields 0.060 m as a smoothing-based reference. Finally, ADR-VINS was successfully deployed in the A2RL Drone Championship Season 2, maintaining stable and robust estimation despite noisy detections during high-agility flight at top speeds of 20.9 m/s. We further utilize ADR-FGO for post-flight evaluation in uninstrumented racing environments.

**arXiv ID:** 2603.02742
</details>

<details>
<summary><strong>Agentic Self-Evolutionary Replanning for Embodied Navigation</strong> - Guoliang Li, Ruihua Han, Chengyang Li, He Li, Shuai Wang, Wenchao Ding, Hong Zhang, Chengzhong Xu - [[pdf]](https://arxiv.org/pdf/2603.02772)</summary>

**Abstract:** Failure is inevitable for embodied navigation in complex environments. To enhance the resilience, replanning (RP) is a viable option, where the robot is allowed to fail, but is capable of adjusting plan until success. However, existing RP approaches freeze the ego action model and miss the opportunities to explore better plans by upgrading the robot itself. To address this limitation, we propose Self-Evolutionary RePlanning, or SERP for short, which leads to a paradigm shift from frozen models towards evolving models by run-time learning from recent experiences. In contrast to existing model evolution approaches that often get stuck at predefined static parameters, we introduce agentic self-evolving action model that uses in-context learning with auto-differentiation (ILAD) for adaptive function adjustment and global parameter reset. To achieve token-efficient replanning for SERP, we also propose graph chain-of-thought (GCOT) replanning with large language model (LLM) inference over distilled graphs. Extensive simulation and real-world experiments demonstrate that SERP achieves higher success rate with lower token expenditure over various benchmarks, validating its superior robustness and efficiency across diverse environments.

**arXiv ID:** 2603.02772
</details>

<details>
<summary><strong>From Language to Action: Can LLM-Based Agents Be Used for Embodied Robot Cognition?</strong> - Shinas Shaji, Fabian Huppertz, Alex Mitrevski, Sebastian Houben - [[pdf]](https://arxiv.org/pdf/2603.03148)</summary>

**Abstract:** In order to flexibly act in an everyday environment, a robotic agent needs a variety of cognitive capabilities that enable it to reason about plans and perform execution recovery. Large language models (LLMs) have been shown to demonstrate emergent cognitive aspects, such as reasoning and language understanding; however, the ability to control embodied robotic agents requires reliably bridging high-level language to low-level functionalities for perception and control. In this paper, we investigate the extent to which an LLM can serve as a core component for planning and execution reasoning in a cognitive robot architecture. For this purpose, we propose a cognitive architecture in which an agentic LLM serves as the core component for planning and reasoning, while components for working and episodic memories support learning from experience and adaptation. An instance of the architecture is then used to control a mobile manipulator in a simulated household environment, where environment interaction is done through a set of high-level tools for perception, reasoning, navigation, grasping, and placement, all of which are made available to the LLM-based agent. We evaluate our proposed system on two household tasks (object placement and object swapping), which evaluate the agent's reasoning, planning, and memory utilisation. The results demonstrate that the LLM-driven agent can complete structured tasks and exhibits emergent adaptation and memory-guided planning, but also reveal significant limitations, such as hallucinations about the task success and poor instruction following by refusing to acknowledge and complete sequential tasks. These findings highlight both the potential and challenges of employing LLMs as embodied cognitive controllers for autonomous robots.

**arXiv ID:** 2603.03148
</details>

<details>
<summary><strong>ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation</strong> - Xialin He, Sirui Xu, Xinyao Li, Runpei Dong, Liuyu Bian, Yu-Xiong Wang, Liang-Yan Gui - [[pdf]](https://arxiv.org/pdf/2603.03279)</summary>

**Abstract:** Achieving autonomous and versatile whole-body loco-manipulation remains a central barrier to making humanoids practically useful. Yet existing approaches are fundamentally constrained: retargeted data are often scarce or low-quality; methods struggle to scale to large skill repertoires; and, most importantly, they rely on tracking predefined motion references rather than generating behavior from perception and high-level task specifications. To address these limitations, we propose ULTRA, a unified framework with two key components. First, we introduce a physics-driven neural retargeting algorithm that translates large-scale motion capture to humanoid embodiments while preserving physical plausibility for contact-rich interactions. Second, we learn a unified multimodal controller that supports both dense references and sparse task specifications, under sensing ranging from accurate motion-capture state to noisy egocentric visual inputs. We distill a universal tracking policy into this controller, compress motor skills into a compact latent space, and apply reinforcement learning finetuning to expand coverage and improve robustness under out-of-distribution scenarios. This enables coordinated whole-body behavior from sparse intent without test-time reference motions. We evaluate ULTRA in simulation and on a real Unitree G1 humanoid. Results show that ULTRA generalizes to autonomous, goal-conditioned whole-body loco-manipulation from egocentric perception, consistently outperforming tracking-only baselines with limited skills.

**arXiv ID:** 2603.03279
</details>

<details>
<summary><strong>Towards an Adaptive Social Game-Playing Robot: An Offline Reinforcement Learning-Based Framework</strong> - Soon Jynn Chu, Raju Gottumukkala, Alan Barhorst - [[pdf]](https://arxiv.org/pdf/2509.16858)</summary>

**Abstract:** HRI research increasingly demands robots that go beyond task execution to respond meaningfully to user emotions. This is especially needed when supporting students with learning difficulties in game-based learning scenarios. Here, the objective of these robots is to train users with game-playing skills, and this requires robots to get input about users' interests and engagement. In this paper, we present a system for an adaptive social game-playing robot. However, creating such an agent through online RL requires extensive real-world training data and potentially be uncomfortable for users. To address this, we investigate offline RL as a safe and efficient alternative. We introduce a system architecture that integrates multimodal emotion recognition and adaptive robotic responses. We also evaluate the performance of various offline RL algorithms using a dataset collected from a real-world human-robot game-playing scenario. Our results indicate that BCQ and DDQN offer the greatest robustness to hyperparameter variations, whereas CQL is the most effective at mitigating overestimation bias. Through this research, we aim to inform the selection and design of reliable offline RL policies for real-world social robotics. Ultimately, this work provides a foundational step toward creating socially intelligent agents that can learn complex and emotion-adaptive behaviors entirely from offline datasets, ensuring both human comfort and practical scalability.

**arXiv ID:** 2509.16858
</details>

<details>
<summary><strong>Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning</strong> - Naoki Shitanda, Motoki Omura, Tatsuya Harada, Takayuki Osa - [[pdf]](https://arxiv.org/pdf/2603.01741)</summary>

**Abstract:** Scaling reinforcement learning to tens of thousands of parallel environments requires overcoming the limited exploration capacity of a single policy. Ensemble-based policy gradient methods, which employ multiple policies to collect diverse samples, have recently been proposed to promote exploration. However, merely broadening the exploration space does not always enhance learning capability, since excessive exploration can reduce exploration quality or compromise training stability. In this work, we theoretically analyze the impact of inter-policy diversity on learning efficiency in policy ensembles, and propose Coupled Policy Optimization which regulates diversity through KL constraints between policies. The proposed method enables effective exploration and outperforms strong baselines such as SAPG, PBT, and PPO across multiple tasks, including challenging dexterous manipulation, in terms of both sample efficiency and final performance. Furthermore, analysis of policy diversity and effective sample size during training reveals that follower policies naturally distribute around the leader, demonstrating the emergence of structured and efficient exploratory behavior. Our results indicate that diverse exploration under appropriate regulation is key to achieving stable and sample-efficient learning in ensemble policy gradient methods. Project page at this https URL .

**arXiv ID:** 2603.01741
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
