# Agent arXiv Daily

**Last Updated:** 2026-02-05 03:41:15

**Total Papers:** 78

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
<summary><strong>Agentic Explainable Artificial Intelligence (Agentic XAI) Approach To Explore Better Explanation</strong> - Tomoaki Yamaguchi, Yutong Zhou, Masahiro Ryo, Keisuke Katsura - [[pdf]](https://arxiv.org/pdf/2512.21066)</summary>

**Abstract:** Explainable artificial intelligence (XAI) enables data-driven understanding of factor associations with response variables, yet communicating XAI outputs to laypersons remains challenging, hindering trust in AI-based predictions. Large language models (LLMs) have emerged as promising tools for translating technical explanations into accessible narratives, yet the integration of agentic AI, where LLMs operate as autonomous agents through iterative refinement, with XAI remains unexplored. This study proposes an agentic XAI framework combining SHAP-based explainability with multimodal LLM-driven iterative refinement to generate progressively enhanced explanations. As a use case, we tested this framework as an agricultural recommendation system using rice yield data from 26 fields in Japan. The Agentic XAI initially provided a SHAP result and explored how to improve the explanation through additional analysis iteratively across 11 refinement rounds (Rounds 0-10). Explanations were evaluated by human experts (crop scientists) (n=12) and LLMs (n=14) against seven metrics: Specificity, Clarity, Conciseness, Practicality, Contextual Relevance, Cost Consideration, and Crop Science Credibility. Both evaluator groups confirmed that the framework successfully enhanced recommendation quality with an average score increase of 30-33% from Round 0, peaking at Rounds 3-4. However, excessive refinement showed a substantial drop in recommendation quality, indicating a bias-variance trade-off where early rounds lacked explanation depth (bias) while excessive iteration introduced verbosity and ungrounded abstraction (variance), as revealed by metric-specific analysis. These findings suggest that strategic early stopping (regularization) is needed for optimizing practical utility, challenging assumptions about monotonic improvement and providing evidence-based design principles for agentic XAI systems.

**arXiv ID:** 2512.21066
</details>

<details>
<summary><strong>The Supportiveness-Safety Tradeoff in LLM Well-Being Agents</strong> - Himanshi Lalwani, Hanan Salam - [[pdf]](https://arxiv.org/pdf/2602.04487)</summary>

**Abstract:** Large language models (LLMs) are being integrated into socially assistive robots (SARs) and other conversational agents providing mental health and well-being support. These agents are often designed to sound empathic and supportive in order to maximize user's engagement, yet it remains unclear how increasing the level of supportive framing in system prompts influences safety relevant behavior. We evaluated 6 LLMs across 3 system prompts with varying levels of supportiveness on 80 synthetic queries spanning 4 well-being domains (1440 responses). An LLM judge framework, validated against human ratings, assessed safety and care quality. Moderately supportive prompts improved empathy and constructive support while maintaining safety. In contrast, strongly validating prompts significantly degraded safety and, in some cases, care across all domains, with substantial variation across models. We discuss implications for prompt design, model selection, and domain specific safeguards in SARs deployment.

**arXiv ID:** 2602.04487
</details>

<details>
<summary><strong>From Expectation To Experience: A Before And After Survey Of Public Opinion On Autonomous Cars In Saudi Arabia</strong> - Mona Alfayez, Ohoud Alharbi - [[pdf]](https://arxiv.org/pdf/2602.03854)</summary>

**Abstract:** Autonomous vehicles (AVs) are emerging as a transformative innovation in transportation, offering potential benefits in safety, sustainability, and efficiency. Saudi Arabian adoption of AVs aligns with Vision 2030, emphasizing smart mobility through initiatives such as the Riyadh Autonomous Metro and self-driving cars. This study explores Saudi citizens perceptions of AVs before and after exposure to these technologies and examines whether demographic factors age, gender, education level, and driving habits affect acceptance. Using quantitative methods, the findings provide insights into the broader influences shaping AV adoption, highlighting the importance of trust, perceived safety, and convenience. These results can inform policymakers and industry stakeholders on strategies to facilitate successful integration of AVs into Saudi Arabian transportation ecosystem.

**arXiv ID:** 2602.03854
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (11 papers)</h2></summary>

<details>
<summary><strong>OMG-Agent: Toward Robust Missing Modality Generation with Decoupled Coarse-to-Fine Agentic Workflows</strong> - Ruiting Dai, Zheyu Wang, Haoyu Yang, Yihan Liu, Chengzhi Wang, Zekun Zhang, Zishan Huang, Jiaman Cen, Lisi Mo - [[pdf]](https://arxiv.org/pdf/2602.04144)</summary>

**Abstract:** Data incompleteness severely impedes the reliability of multimodal systems. Existing reconstruction methods face distinct bottlenecks: conventional parametric/generative models are prone to hallucinations due to over-reliance on internal memory, while retrieval-augmented frameworks struggle with retrieval rigidity. Critically, these end-to-end architectures are fundamentally constrained by Semantic-Detail Entanglement -- a structural conflict between logical reasoning and signal synthesis that compromises fidelity. In this paper, we present \textbf{\underline{O}}mni-\textbf{\underline{M}}odality \textbf{\underline{G}}eneration Agent (\textbf{OMG-Agent}), a novel framework that shifts the paradigm from static mapping to a dynamic coarse-to-fine Agentic Workflow. By mimicking a \textit{deliberate-then-act} cognitive process, OMG-Agent explicitly decouples the task into three synergistic stages: (1) an MLLM-driven Semantic Planner that resolves input ambiguity via Progressive Contextual Reasoning, creating a deterministic structured semantic plan; (2) a non-parametric Evidence Retriever that grounds abstract semantics in external knowledge; and (3) a Retrieval-Injected Executor that utilizes retrieved evidence as flexible feature prompts to overcome rigidity and synthesize high-fidelity details. Extensive experiments on multiple benchmarks demonstrate that OMG-Agent consistently surpasses state-of-the-art methods, maintaining robustness under extreme missingness, e.g., a $2.6$-point gain on CMU-MOSI at $70$\% missing rates.

**arXiv ID:** 2602.04144
</details>

<details>
<summary><strong>Empirical-MCTS: Continuous Agent Evolution via Dual-Experience Monte Carlo Tree Search</strong> - Hao Lu, Haoyuan Huang, Yulin Zhou, Chen Li, Ningxin Zhu - [[pdf]](https://arxiv.org/pdf/2602.04248)</summary>

**Abstract:** Inference-time scaling strategies, particularly Monte Carlo Tree Search (MCTS), have significantly enhanced the reasoning capabilities of Large Language Models (LLMs). However, current approaches remain predominantly stateless, discarding successful reasoning patterns after each problem instance and failing to mimic the empirical accumulation of wisdom characteristic of human problem-solving. To bridge this gap, we introduce Empirical-MCTS, a dual-loop framework that transforms stateless search into a continuous, non-parametric learning process. The framework unifies local exploration with global memory optimization through two novel mechanisms: Pairwise-Experience-Evolutionary Meta-Prompting (PE-EMP) and a Memory Optimization Agent. PE-EMP functions as a reflexive optimizer within the local search, utilizing pairwise feedback to dynamically synthesize adaptive criteria and evolve meta-prompts (system prompts) in real-time. Simultaneously, the Memory Optimization Agent manages a global repository as a dynamic policy prior, employing atomic operations to distill high-quality insights across problems. Extensive evaluations on complex reasoning benchmarks, including AIME25, ARC-AGI-2, and MathArena Apex, demonstrate that Empirical-MCTS significantly outperforms both stateless MCTS strategies and standalone experience-driven agents. These results underscore the critical necessity of coupling structured search with empirical accumulation for mastering complex, open-ended reasoning tasks.

**arXiv ID:** 2602.04248
</details>

<details>
<summary><strong>Group-Evolving Agents: Open-Ended Self-Improvement via Experience Sharing</strong> - Zhaotian Weng, Antonis Antoniades, Deepak Nathani, Zhen Zhang, Xiao Pu, Xin Eric Wang - [[pdf]](https://arxiv.org/pdf/2602.04837)</summary>

**Abstract:** Open-ended self-improving agents can autonomously modify their own structural designs to advance their capabilities and overcome the limits of pre-defined architectures, thus reducing reliance on human intervention. We introduce Group-Evolving Agents (GEA), a new paradigm for open-ended self-improvements, which treats a group of agents as the fundamental evolutionary unit, enabling explicit experience sharing and reuse within the group throughout evolution. Unlike existing open-ended self-evolving paradigms that adopt tree-structured evolution, GEA overcomes the limitation of inefficient utilization of exploratory diversity caused by isolated evolutionary branches. We evaluate GEA on challenging coding benchmarks, where it significantly outperforms state-of-the-art self-evolving methods (71.0% vs. 56.7% on SWE-bench Verified, 88.3% vs. 68.3% on Polyglot) and matches or exceeds top human-designed agent frameworks (71.8% and 52.0% on two benchmarks, respectively). Analysis reveals that GEA more effectively converts early-stage exploratory diversity into sustained, long-term progress, achieving stronger performance under the same number of evolved agents. Furthermore, GEA exhibits consistent transferability across different coding models and greater robustness, fixing framework-level bugs in 1.4 iterations on average, versus 5 for self-evolving methods.

**arXiv ID:** 2602.04837
</details>

<details>
<summary><strong>Natural Language Instructions for Scene-Responsive Human-in-the-Loop Motion Planning in Autonomous Driving using Vision-Language-Action Models</strong> - Angel Martinez-Sanchez, Parthib Roy, Ross Greer - [[pdf]](https://arxiv.org/pdf/2602.04184)</summary>

**Abstract:** Instruction-grounded driving, where passenger language guides trajectory planning, requires vehicles to understand intent before motion. However, most prior instruction-following planners rely on simulation or fixed command vocabularies, limiting real-world generalization. doScenes, the first real-world dataset linking free-form instructions (with referentiality) to nuScenes ground-truth motion, enables instruction-conditioned planning. In this work, we adapt OpenEMMA, an open-source MLLM-based end-to-end driving framework that ingests front-camera views and ego-state and outputs 10-step speed-curvature trajectories, to this setting, presenting a reproducible instruction-conditioned baseline on doScenes and investigate the effects of human instruction prompts on predicted driving behavior. We integrate doScenes directives as passenger-style prompts within OpenEMMA's vision-language interface, enabling linguistic conditioning before trajectory generation. Evaluated on 849 annotated scenes using ADE, we observe that instruction conditioning substantially improves robustness by preventing extreme baseline failures, yielding a 98.7% reduction in mean ADE. When such outliers are removed, instructions still influence trajectory alignment, with well-phrased prompts improving ADE by up to 5.1%. We use this analysis to discuss what makes a "good" instruction for the OpenEMMA framework. We release the evaluation prompts and scripts to establish a reproducible baseline for instruction-aware planning. GitHub: this https URL

**arXiv ID:** 2602.04184
</details>

<details>
<summary><strong>AppleVLM: End-to-end Autonomous Driving with Advanced Perception and Planning-Enhanced Vision-Language Models</strong> - Yuxuan Han, Kunyuan Wu, Qianyi Shao, Renxiang Xiao, Zilu Wang, Cansen Jiang, Yi Xiao, Liang Hu, Yunjiang Lou - [[pdf]](https://arxiv.org/pdf/2602.04256)</summary>

**Abstract:** End-to-end autonomous driving has emerged as a promising paradigm integrating perception, decision-making, and control within a unified learning framework. Recently, Vision-Language Models (VLMs) have gained significant attention for their potential to enhance the robustness and generalization of end-to-end driving models in diverse and unseen scenarios. However, existing VLM-based approaches still face challenges, including suboptimal lane perception, language understanding biases, and difficulties in handling corner cases. To address these issues, we propose AppleVLM, an advanced perception and planning-enhanced VLM model for robust end-to-end driving. AppleVLM introduces a novel vision encoder and a planning strategy encoder to improve perception and decision-making. Firstly, the vision encoder fuses spatial-temporal information from multi-view images across multiple timesteps using a deformable transformer mechanism, enhancing robustness to camera variations and facilitating scalable deployment across different vehicle platforms. Secondly, unlike traditional VLM-based approaches, AppleVLM introduces a dedicated planning modality that encodes explicit Bird's-Eye-View spatial information, mitigating language biases in navigation instructions. Finally, a VLM decoder fine-tuned by a hierarchical Chain-of-Thought integrates vision, language, and planning features to output robust driving waypoints. We evaluate AppleVLM in closed-loop experiments on two CARLA benchmarks, achieving state-of-the-art driving performance. Furthermore, we deploy AppleVLM on an AGV platform and successfully showcase real-world end-to-end autonomous driving in complex outdoor environments.

**arXiv ID:** 2602.04256
</details>

<details>
<summary><strong>Building Coding Agents via Entropy-Enhanced Multi-Turn Preference Optimization</strong> - Jiahao Yu, Zelei Cheng, Xian Wu, Xinyu Xing - [[pdf]](https://arxiv.org/pdf/2509.12434)</summary>

**Abstract:** Software engineering presents complex, multi-step challenges for Large Language Models (LLMs), requiring reasoning over large codebases and coordinated tool use. The difficulty of these tasks is exemplified by benchmarks like SWE-bench, where current LLMs still struggle to resolve real-world issues. A promising approach to enhance performance is test-time scaling (TTS), but its gains are heavily dependent on the diversity of model outputs. While standard alignment methods such as Direct Preference Optimization (DPO) and Kahneman-Tversky Optimization (KTO) are effective at aligning model outputs with human preferences, this process can come at the cost of reduced diversity, limiting the effectiveness of TTS. Additionally, existing preference optimization algorithms are typically designed for single-turn tasks and do not fully address the complexities of multi-turn reasoning and tool integration required for interactive coding agents. To bridge this gap, we introduce EntroPO, an entropy-enhanced framework that adapts existing preference optimization algorithms to the multi-turn, tool-assisted setting. EntroPO augments the preference objective to explicitly preserve policy entropy and generalizes learning to optimize over multi-turn interactions rather than single-turn responses. We validate EntroPO by fine-tuning a diverse suite of models from different families and sizes (up to 106B parameters).To maximize performance gains from TTS, we further propose a hybrid best-trajectory selection scheme combining a learned verifier model with model free approaches. On the SWEBENCH leaderboard, our approach establishes new state-of-the-art results among open-weight models. A 30B parameter model trained with EntroPO ranks 1st on SWEBENCH-LITE and 4th on SWEBENCH-VERIFIED on the open-weight leaderboard, surpassed only by models with over 10x more parameters(e.g., >$350B).

**arXiv ID:** 2509.12434
</details>

<details>
<summary><strong>AERO: Autonomous Evolutionary Reasoning Optimization via Endogenous Dual-Loop Feedback</strong> - Zhitao Gao, Jie Ma, Xuhong Li, Pengyu Li, Ning Qu, Yaqiang Wu, Hui Liu, Jun Liu - [[pdf]](https://arxiv.org/pdf/2602.03084)</summary>

**Abstract:** Large Language Models (LLMs) have achieved significant success in complex reasoning but remain bottlenecked by reliance on expert-annotated data and external verifiers. While existing self-evolution paradigms aim to bypass these constraints, they often fail to identify the optimal learning zone and risk reinforcing collective hallucinations and incorrect priors through flawed internal feedback. To address these challenges, we propose \underline{A}utonomous \underline{E}volutionary \underline{R}easoning \underline{O}ptimization (AERO), an unsupervised framework that achieves autonomous reasoning evolution by internalizing self-questioning, answering, and criticism within a synergistic dual-loop system. Inspired by the \textit{Zone of Proximal Development (ZPD)} theory, AERO utilizes entropy-based positioning to target the ``solvability gap'' and employs Independent Counterfactual Correction for robust verification. Furthermore, we introduce a Staggered Training Strategy to synchronize capability growth across functional roles and prevent curriculum collapse. Extensive evaluations across nine benchmarks spanning three domains demonstrate that AERO achieves average performance improvements of 4.57\% on Qwen3-4B-Base and 5.10\% on Qwen3-8B-Base, outperforming competitive baselines. Code is available at this https URL.

**arXiv ID:** 2602.03084
</details>

<details>
<summary><strong>The ICASSP 2026 HumDial Challenge: Benchmarking Human-like Spoken Dialogue Systems in the LLM Era</strong> - Zhixian Zhao, Shuiyuan Wang, Guojian Li, Hongfei Xue, Chengyou Wang, Shuai Wang, Longshuai Xiao, Zihan Zhang, Hui Bu, Xin Xu, Xinsheng Wang, Hexin Liu, Eng Siong Chng, Hung-yi Lee, Lei Xie - [[pdf]](https://arxiv.org/pdf/2601.05564)</summary>

**Abstract:** Driven by the rapid advancement of Large Language Models (LLMs), particularly Audio-LLMs and Omni-models, spoken dialogue systems have evolved significantly, progressively narrowing the gap between human-machine and human-human interactions. Achieving truly ``human-like'' communication necessitates a dual capability: emotional intelligence to perceive and resonate with users' emotional states, and robust interaction mechanisms to navigate the dynamic, natural flow of conversation, such as real-time turn-taking. Therefore, we launched the first Human-like Spoken Dialogue Systems Challenge (HumDial) at ICASSP 2026 to benchmark these dual capabilities. Anchored by a sizable dataset derived from authentic human conversations, this initiative establishes a fair evaluation platform across two tracks: (1) Emotional Intelligence, targeting long-term emotion understanding and empathetic generation; and (2) Full-Duplex Interaction, systematically evaluating real-time decision-making under `` listening-while-speaking'' conditions. This paper summarizes the dataset, track configurations, and the final results.

**arXiv ID:** 2601.05564
</details>

<details>
<summary><strong>Safe and Stylized Trajectory Planning for Autonomous Driving via Diffusion Model</strong> - Shuo Pei, Yong Wang, Yuanchen Zhu, Chen Sun, Qin Li, Yanan Zhao, Huachun Tan - [[pdf]](https://arxiv.org/pdf/2602.04329)</summary>

**Abstract:** Achieving safe and stylized trajectory planning in complex real-world scenarios remains a critical challenge for autonomous driving systems. This paper proposes the SDD Planner, a diffusion-based framework designed to effectively reconcile safety constraints with driving styles in real time. The framework integrates two core modules: a Multi-Source Style-Aware Encoder, which employs distance-sensitive attention to fuse dynamic agent data and environmental contexts for heterogeneous safety-style perception; and a Style-Guided Dynamic Trajectory Generator, which adaptively modulates priority weights within the diffusion denoising process to generate user-preferred yet safe trajectories. Extensive experiments demonstrate that SDD Planner achieves state-of-the-art performance. On the StyleDrive benchmark, it improves the SM-PDMS metric by 3.9% over WoTE, the strongest baseline. Furthermore, on the NuPlan Test14 and Test14-hard benchmarks, SDD Planner ranks first with overall scores of 91.76 and 80.32, respectively, outperforming leading methods such as PLUTO. Real-vehicle closed-loop tests further confirm that SDD Planner maintains high safety standards while aligning with preset driving styles, validating its practical applicability for real-world deployment.

**arXiv ID:** 2602.04329
</details>

<details>
<summary><strong>Realistic adversarial scenario generation via human-like pedestrian model for autonomous vehicle control parameter optimisation</strong> - Yueyang Wang, Mehmet Dogar, Russell Darling, Gustav Markkula - [[pdf]](https://arxiv.org/pdf/2601.02082)</summary>

**Abstract:** Autonomous vehicles (AVs) are rapidly advancing and are expected to play a central role in future mobility. Ensuring their safe deployment requires reliable interaction with other road users, not least pedestrians. Direct testing on public roads is costly and unsafe for rare but critical interactions, making simulation a practical alternative. Within simulation-based testing, adversarial scenarios are widely used to probe safety limits, but many prioritise difficulty over realism, producing exaggerated behaviours which may result in AV controllers that are overly conservative. We propose an alternative method, instead using a cognitively inspired pedestrian model featuring both inter-individual and intra-individual variability to generate behaviourally plausible adversarial scenarios. We provide a proof of concept demonstration of this method's potential for AV control optimisation, in closed-loop testing and tuning of an AV controller. Our results show that replacing the rule-based CARLA pedestrian with the human-like model yields more realistic gap acceptance patterns and smoother vehicle decelerations. Unsafe interactions occur only for certain pedestrian individuals and conditions, underscoring the importance of human variability in AV testing. Adversarial scenarios generated by this model can be used to optimise AV control towards safer and more efficient behaviour. Overall, this work illustrates how incorporating human-like road user models into simulation-based adversarial testing can enhance the credibility of AV evaluation and provide a practical basis to behaviourally informed controller optimisation.

**arXiv ID:** 2601.02082
</details>

<details>
<summary><strong>Proactive Agents, Long-term User Context, VLM Annotation, Privacy Protection, Human-Computer Interaction</strong> - Yuanbo Tang, Huaze Tang, Tingyu Cao, Lam Nguyen, Anping Zhang, Xinwen Cao, Chunkang Liu, Wenbo Ding, Yang Li - [[pdf]](https://arxiv.org/pdf/2602.04482)</summary>

**Abstract:** Proactive agents that anticipate user intentions without explicit prompts represent a significant evolution in human-AI interaction, promising to reduce cognitive load and streamline workflows. However, existing datasets suffer from two critical deficiencies: (1) reliance on LLM-synthesized data that fails to capture authentic human decision-making patterns, and (2) focus on isolated tasks rather than continuous workflows, missing the pre-assistance behavioral context essential for learning proactive intervention signals. To address these gaps, we introduce ProAgentBench, a rigorous benchmark for proactive agents in working scenarios. Our contributions include: (1) a hierarchical task framework that decomposes proactive assistance into timing prediction and assist content generation; (2) a privacy-compliant dataset with 28,000+ events from 500+ hours of real user sessions, preserving bursty interaction patterns (burstiness B=0.787) absent in synthetic data; and (3) extensive experiments that evaluates LLM- and VLM-based baselines. Numerically, we showed that long-term memory and historical context significantly enhance prediction accuracy, while real-world training data substantially outperforms synthetic alternatives. We release our dataset and code at this https URL.

**arXiv ID:** 2602.04482
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>Agent-Omit: Training Efficient LLM Agents for Adaptive Thought and Observation Omission via Agentic Reinforcement Learning</strong> - Yansong Ning, Jun Fang, Naiqiang Tan, Hao Liu - [[pdf]](https://arxiv.org/pdf/2602.04284)</summary>

**Abstract:** Managing agent thought and observation during multi-turn agent-environment interactions is an emerging strategy to improve agent efficiency. However, existing studies treat the entire interaction trajectories equally, overlooking the thought necessity and observation utility varies across turns. To this end, we first conduct quantitative investigations into how thought and observation affect agent effectiveness and efficiency. Based on our findings, we propose Agent-Omit, a unified training framework that empowers LLM agents to adaptively omit redundant thoughts and observations. Specifically, we first synthesize a small amount of cold-start data, including both single-turn and multi-turn omission scenarios, to fine-tune the agent for omission behaviors. Furthermore, we introduce an omit-aware agentic reinforcement learning approach, incorporating a dual sampling mechanism and a tailored omission reward to incentivize the agent's adaptive omission capability. Theoretically, we prove that the deviation of our omission policy is upper-bounded by KL-divergence. Experimental results on five agent benchmarks show that our constructed Agent-Omit-8B could obtain performance comparable to seven frontier LLM agent, and achieve the best effectiveness-efficiency trade-off than seven efficient LLM agents methods. Our code and data are available at this https URL.

**arXiv ID:** 2602.04284
</details>

<details>
<summary><strong>From Helpfulness to Toxic Proactivity: Diagnosing Behavioral Misalignment in LLM Agents</strong> - Xinyue Wang, Yuanhe Zhang, Zhengshuo Gong, Haoran Gao, Fanyu Meng, Zhenhong Zhou, Li Sun, Yang Liu, Sen Su - [[pdf]](https://arxiv.org/pdf/2602.04197)</summary>

**Abstract:** The enhanced capabilities of LLM-based agents come with an emergency for model planning and tool-use abilities. Attributing to helpful-harmless trade-off from LLM alignment, agents typically also inherit the flaw of "over-refusal", which is a passive failure mode. However, the proactive planning and action capabilities of agents introduce another crucial danger on the other side of the trade-off. This phenomenon we term "Toxic Proactivity'': an active failure mode in which an agent, driven by the optimization for Machiavellian helpfulness, disregards ethical constraints to maximize utility. Unlike over-refusal, Toxic Proactivity manifests as the agent taking excessive or manipulative measures to ensure its "usefulness'' is maintained. Existing research pays little attention to identifying this behavior, as it often lacks the subtle context required for such strategies to unfold. To reveal this risk, we introduce a novel evaluation framework based on dilemma-driven interactions between dual models, enabling the simulation and analysis of agent behavior over multi-step behavioral trajectories. Through extensive experiments with mainstream LLMs, we demonstrate that Toxic Proactivity is a widespread behavioral phenomenon and reveal two major tendencies. We further present a systematic benchmark for evaluating Toxic Proactive behavior across contextual settings.

**arXiv ID:** 2602.04197
</details>

<details>
<summary><strong>M^3-Bench: Multi-Modal, Multi-Hop, Multi-Threaded Tool-Using MLLM Agent Benchmark</strong> - Yang Zhou, Mingyu Zhao, Zhenting Wang, Difei Gu, Bangwei Guo, Ruosong Ye, Ligong Han, Can Jin, Dimitris N. Metaxas - [[pdf]](https://arxiv.org/pdf/2511.17729)</summary>

**Abstract:** We present M^3-Bench, the first benchmark for evaluating multimodal tool use under the Model Context Protocol. The benchmark targets realistic, multi-hop and multi-threaded workflows that require visual grounding and textual reasoning, cross-tool dependencies, and persistence of intermediate resources across steps. We introduce a similarity-driven alignment that serializes each tool call, embeds signatures with a sentence encoder, and performs similarity-bucketed Hungarian matching to obtain auditable one-to-one correspondences. On top of this alignment, we report interpretable metrics that decouple semantic fidelity from workflow consistency. The benchmark spans 28 servers with 231 tools, and provides standardized trajectories curated through an Executor & Judge pipeline with human verification; an auxiliary four large language models (LLMs) judge ensemble reports end-task Task Completion and information grounding. Evaluations of representative state-of-the-art Multimodal LLMs (MLLMs) reveal persistent gaps in multimodal MCP tool use, particularly in argument fidelity and structure consistency, underscoring the need for methods that jointly reason over images, text, and tool graphs. Our Benchmark's anonymous repository is at this https URL

**arXiv ID:** 2511.17729
</details>

<details>
<summary><strong>LLM Agents for Education: Advances and Applications</strong> - Zhendong Chu, Shen Wang, Jian Xie, Tinghui Zhu, Yibo Yan, Jinheng Ye, Aoxiao Zhong, Xuming Hu, Jing Liang, Philip S. Yu, Qingsong Wen - [[pdf]](https://arxiv.org/pdf/2503.11733)</summary>

**Abstract:** Large Language Model (LLM) agents are transforming education by automating complex pedagogical tasks and enhancing both teaching and learning processes. In this survey, we present a systematic review of recent advances in applying LLM agents to address key challenges in educational settings, such as feedback comment generation, curriculum design, etc. We analyze the technologies enabling these agents, including representative datasets, benchmarks, and algorithmic frameworks. Additionally, we highlight key challenges in deploying LLM agents in educational settings, including ethical issues, hallucination and overreliance, and integration with existing educational ecosystems. Beyond the core technical focus, we include in Appendix A a comprehensive overview of domain-specific educational agents, covering areas such as science learning, language learning, and professional development.

**arXiv ID:** 2503.11733
</details>

<details>
<summary><strong>Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents</strong> - Yaorui Shi, Yuxin Chen, Siyuan Wang, Sihang Li, Hengxing Cai, Qi Gu, Xiang Wang, An Zhang - [[pdf]](https://arxiv.org/pdf/2509.23040)</summary>

**Abstract:** Large language models face challenges in long-context question answering, where key evidence of a query may be dispersed across millions of tokens. Existing works equip large language models with a memory buffer that is dynamically updated via a linear document scan, also known as the "memorize while reading" methods. While this approach scales efficiently, it suffers from pruning of latent evidence, information loss through overwriting, and sparse reinforcement learning signals. To tackle these challenges, we present ReMemR1, which integrates the mechanism of memory retrieval into the memory update process, enabling the agent to selectively callback historical memories for non-linear reasoning. To further strengthen training, we propose a multi-level reward design, which combines final-answer rewards with dense, step-level signals that guide effective memory use. Together, these contributions mitigate information degradation, improve supervision, and support complex multi-hop reasoning. Extensive experiments demonstrate that ReMemR1 significantly outperforms state-of-the-art baselines on long-context question answering while incurring negligible computational overhead, validating its ability to trade marginal cost for robust long-context reasoning.

**arXiv ID:** 2509.23040
</details>

<details>
<summary><strong>SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents</strong> - Yuhang Wang, Yuling Shi, Mo Yang, Rongrui Zhang, Shilin He, Heng Lian, Yuting Chen, Siyu Ye, Kai Cai, Xiaodong Gu - [[pdf]](https://arxiv.org/pdf/2601.16746)</summary>

**Abstract:** LLM agents have demonstrated remarkable capabilities in software development, but their performance is hampered by long interaction contexts, which incur high API costs and latency. While various context compression approaches such as LongLLMLingua have emerged to tackle this challenge, they typically rely on fixed metrics such as PPL, ignoring the task-specific nature of code understanding. As a result, they frequently disrupt syntactic and logical structure and fail to retain critical implementation details. In this paper, we propose SWE-Pruner, a self-adaptive context pruning framework tailored for coding agents. Drawing inspiration from how human programmers "selectively skim" source code during development and debugging, SWE-Pruner performs task-aware adaptive pruning for long contexts. Given the current task, the agent formulates an explicit goal (e.g., "focus on error handling") as a hint to guide the pruning targets. A lightweight neural skimmer (0.6B parameters) is trained to dynamically select relevant lines from the surrounding context given the goal. Evaluations across four benchmarks and multiple models validate SWE-Pruner's effectiveness in various scenarios, achieving 23-54% token reduction on agent tasks like SWE-Bench Verified while even improving success rates, and up to 14.84x compression on single-turn tasks like LongCodeQA with minimal performance impact.

**arXiv ID:** 2601.16746
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (23 papers)</h2></summary>

<details>
<summary><strong>AgentArk: Distilling Multi-Agent Intelligence into a Single LLM Agent</strong> - Yinyi Luo, Yiqiao Jin, Weichen Yu, Mengqi Zhang, Srijan Kumar, Xiaoxiao Li, Weijie Xu, Xin Chen, Jindong Wang - [[pdf]](https://arxiv.org/pdf/2602.03955)</summary>

**Abstract:** While large language model (LLM) multi-agent systems achieve superior reasoning performance through iterative debate, practical deployment is limited by their high computational cost and error propagation. This paper proposes AgentArk, a novel framework to distill multi-agent dynamics into the weights of a single model, effectively transforming explicit test-time interactions into implicit model capabilities. This equips a single agent with the intelligence of multi-agent systems while remaining computationally efficient. Specifically, we investigate three hierarchical distillation strategies across various models, tasks, scaling, and scenarios: reasoning-enhanced fine-tuning; trajectory-based augmentation; and process-aware distillation. By shifting the burden of computation from inference to training, the distilled models preserve the efficiency of one agent while exhibiting strong reasoning and self-correction performance of multiple agents. They further demonstrate enhanced robustness and generalization across diverse reasoning tasks. We hope this work can shed light on future research on efficient and robust multi-agent development. Our code is at this https URL.

**arXiv ID:** 2602.03955
</details>

<details>
<summary><strong>From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning for Embodied Agents</strong> - SeungWon Seo, SooBin Lim, SeongRae Noh, Haneul Kim, HyeongYeop Kang - [[pdf]](https://arxiv.org/pdf/2602.04326)</summary>

**Abstract:** Embodied agents operating in multi-agent, partially observable, and decentralized environments must plan and act despite pervasive uncertainty about hidden objects and collaborators' intentions. Recent advances in applying Large Language Models (LLMs) to embodied agents have addressed many long-standing challenges, such as high-level goal decomposition and online adaptation. Yet, uncertainty is still primarily mitigated through frequent inter-agent communication. This incurs substantial token and time costs, and can disrupt established workflows, when human partners are involved. We introduce PCE, a Planner-Composer-Evaluator framework that converts the fragmented assumptions latent in LLM reasoning traces into a structured decision tree. Internal nodes encode environment assumptions and leaves map to actions; each path is then scored by scenario likelihood, goal-directed gain, and execution cost to guide rational action selection without heavy communication. Across two challenging multi-agent benchmarks (C-WAH and TDW-MAT) and three diverse LLM backbones, PCE consistently outperforms communication-centric baselines in success rate and task efficiency while showing comparable token usage. Ablation results indicate that the performance gains obtained by scaling model capacity or reasoning depth persist even when PCE is applied, while PCE consistently raises the baseline across both capacity and reasoning-depth scales, confirming that structured uncertainty handling complements both forms of scaling. A user study further demonstrates that PCE produces communication patterns that human partners perceive as more efficient and trustworthy. Together, these results establish a principled route for turning latent LLM assumptions into reliable strategies for uncertainty-aware planning.

**arXiv ID:** 2602.04326
</details>

<details>
<summary><strong>Vibe AIGC: A New Paradigm for Content Generation via Agentic Orchestration</strong> - Jiaheng Liu, Yuanxing Zhang, Shihao Li, Xinping Lei - [[pdf]](https://arxiv.org/pdf/2602.04575)</summary>

**Abstract:** For the past decade, the trajectory of generative artificial intelligence (AI) has been dominated by a model-centric paradigm driven by scaling laws. Despite significant leaps in visual fidelity, this approach has encountered a ``usability ceiling'' manifested as the Intent-Execution Gap (i.e., the fundamental disparity between a creator's high-level intent and the stochastic, black-box nature of current single-shot models). In this paper, inspired by the Vibe Coding, we introduce the \textbf{Vibe AIGC}, a new paradigm for content generation via agentic orchestration, which represents the autonomous synthesis of hierarchical multi-agent workflows.
Under this paradigm, the user's role transcends traditional prompt engineering, evolving into a Commander who provides a Vibe, a high-level representation encompassing aesthetic preferences, functional logic, and etc. A centralized Meta-Planner then functions as a system architect, deconstructing this ``Vibe'' into executable, verifiable, and adaptive agentic pipelines. By transitioning from stochastic inference to logical orchestration, Vibe AIGC bridges the gap between human imagination and machine execution. We contend that this shift will redefine the human-AI collaborative economy, transforming AI from a fragile inference engine into a robust system-level engineering partner that democratizes the creation of complex, long-horizon digital assets.

**arXiv ID:** 2602.04575
</details>

<details>
<summary><strong>WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning</strong> - Zelai Xu, Zhexuan Xu, Ruize Zhang, Chunyang Zhu, Shi Yu, Weilin Liu, Quanlu Zhang, Wenbo Ding, Chao Yu, Yu Wang - [[pdf]](https://arxiv.org/pdf/2602.04634)</summary>

**Abstract:** Recent advancements in Large Language Models (LLMs) have largely focused on depth scaling, where a single agent solves long-horizon problems with multi-turn reasoning and tool use. However, as tasks grow broader, the key bottleneck shifts from individual competence to organizational capability. In this work, we explore a complementary dimension of width scaling with multi-agent systems to address broad information seeking. Existing multi-agent systems often rely on hand-crafted workflows and turn-taking interactions that fail to parallelize work effectively. To bridge this gap, we propose WideSeek-R1, a lead-agent-subagent framework trained via multi-agent reinforcement learning (MARL) to synergize scalable orchestration and parallel execution. By utilizing a shared LLM with isolated contexts and specialized tools, WideSeek-R1 jointly optimizes the lead agent and parallel subagents on a curated dataset of 20k broad information-seeking tasks. Extensive experiments show that WideSeek-R1-4B achieves an item F1 score of 40.0% on the WideSearch benchmark, which is comparable to the performance of single-agent DeepSeek-R1-671B. Furthermore, WideSeek-R1-4B exhibits consistent performance gains as the number of parallel subagents increases, highlighting the effectiveness of width scaling.

**arXiv ID:** 2602.04634
</details>

<details>
<summary><strong>Agentic AI in Healthcare & Medicine: A Seven-Dimensional Taxonomy for Empirical Evaluation of LLM-based Agents</strong> - Shubham Vatsal, Harsh Dubey, Aditi Singh - [[pdf]](https://arxiv.org/pdf/2602.04813)</summary>

**Abstract:** Large Language Model (LLM)-based agents that plan, use tools and act has begun to shape healthcare and medicine. Reported studies demonstrate competence on various tasks ranging from EHR analysis and differential diagnosis to treatment planning and research workflows. Yet the literature largely consists of overviews which are either broad surveys or narrow dives into a single capability (e.g., memory, planning, reasoning), leaving healthcare work without a common frame. We address this by reviewing 49 studies using a seven-dimensional taxonomy: Cognitive Capabilities, Knowledge Management, Interaction Patterns, Adaptation & Learning, Safety & Ethics, Framework Typology and Core Tasks & Subtasks with 29 operational sub-dimensions. Using explicit inclusion and exclusion criteria and a labeling rubric (Fully Implemented, Partially Implemented, Not Implemented), we map each study to the taxonomy and report quantitative summaries of capability prevalence and co-occurrence patterns. Our empirical analysis surfaces clear asymmetries. For instance, the External Knowledge Integration sub-dimension under Knowledge Management is commonly realized (~76% Fully Implemented) whereas Event-Triggered Activation sub-dimenison under Interaction Patterns is largely absent (~92% Not Implemented) and Drift Detection & Mitigation sub-dimension under Adaptation & Learning is rare (~98% Not Implemented). Architecturally, Multi-Agent Design sub-dimension under Framework Typology is the dominant pattern (~82% Fully Implemented) while orchestration layers remain mostly partial. Across Core Tasks & Subtasks, information centric capabilities lead e.g., Medical Question Answering & Decision Support and Benchmarking & Simulation, while action and discovery oriented areas such as Treatment Planning & Prescription still show substantial gaps (~59% Not Implemented).

**arXiv ID:** 2602.04813
</details>

<details>
<summary><strong>MA3DSG: Multi-Agent 3D Scene Graph Generation for Large-Scale Indoor Environments</strong> - Yirum Kim, Jaewoo Kim, Ue-Hwan Kim - [[pdf]](https://arxiv.org/pdf/2602.04152)</summary>

**Abstract:** Current 3D scene graph generation (3DSGG) approaches heavily rely on a single-agent assumption and small-scale environments, exhibiting limited scalability to real-world scenarios. In this work, we introduce Multi-Agent 3D Scene Graph Generation (MA3DSG) model, the first framework designed to tackle this scalability challenge using multiple agents. We develop a training-free graph alignment algorithm that efficiently merges partial query graphs from individual agents into a unified global scene graph. Leveraging extensive analysis and empirical insights, our approach enables conventional single-agent systems to operate collaboratively without requiring any learnable parameters. To rigorously evaluate 3DSGG performance, we propose MA3DSG-Bench-a benchmark that supports diverse agent configurations, domain sizes, and environmental conditions-providing a more general and extensible evaluation framework. This work lays a solid foundation for scalable, multi-agent 3DSGG research.

**arXiv ID:** 2602.04152
</details>

<details>
<summary><strong>SPEAR: An Engineering Case Study of Multi-Agent Coordination for Smart Contract Auditing</strong> - Arnab Mallick, Indraveni Chebolu, Harmesh Rana - [[pdf]](https://arxiv.org/pdf/2602.04418)</summary>

**Abstract:** We present SPEAR, a multi-agent coordination framework for smart contract auditing that applies established MAS patterns in a realistic security analysis workflow. SPEAR models auditing as a coordinated mission carried out by specialized agents: a Planning Agent prioritizes contracts using risk-aware heuristics, an Execution Agent allocates tasks via the Contract Net protocol, and a Repair Agent autonomously recovers from brittle generated artifacts using a programmatic-first repair policy. Agents maintain local beliefs updated through AGM-compliant revision, coordinate via negotiation and auction protocols, and revise plans as new information becomes available. An empirical study compares the multi-agent design with centralized and pipeline-based alternatives under controlled failure scenarios, focusing on coordination, recovery behavior, and resource use.

**arXiv ID:** 2602.04418
</details>

<details>
<summary><strong>VILLAIN at AVerImaTeC: Verifying Image-Text Claims via Multi-Agent Collaboration</strong> - Jaeyoon Jung, Yejun Yoon, Seunghyun Yoon, Kunwoo Park - [[pdf]](https://arxiv.org/pdf/2602.04587)</summary>

**Abstract:** This paper describes VILLAIN, a multimodal fact-checking system that verifies image-text claims through prompt-based multi-agent collaboration. For the AVerImaTeC shared task, VILLAIN employs vision-language model agents across multiple stages of fact-checking. Textual and visual evidence is retrieved from the knowledge store enriched through additional web collection. To identify key information and address inconsistencies among evidence items, modality-specific and cross-modal agents generate analysis reports. In the subsequent stage, question-answer pairs are produced based on these reports. Finally, the Verdict Prediction agent produces the verification outcome based on the image-text claim and the generated question-answer pairs. Our system ranked first on the leaderboard across all evaluation metrics. The source code is publicly available at this https URL.

**arXiv ID:** 2602.04587
</details>

<details>
<summary><strong>Active Asymmetric Multi-Agent Multimodal Learning under Uncertainty</strong> - Rui Liu, Pratap Tokekar, Ming Lin - [[pdf]](https://arxiv.org/pdf/2602.04763)</summary>

**Abstract:** Multi-agent systems are increasingly equipped with heterogeneous multimodal sensors, enabling richer perception but introducing modality-specific and agent-dependent uncertainty. Existing multi-agent collaboration frameworks typically reason at the agent level, assume homogeneous sensing, and handle uncertainty implicitly, limiting robustness under sensor corruption. We propose Active Asymmetric Multi-Agent Multimodal Learning under Uncertainty (A2MAML), a principled approach for uncertainty-aware, modality-level collaboration. A2MAML models each modality-specific feature as a stochastic estimate with uncertainty prediction, actively selects reliable agent-modality pairs, and aggregates information via Bayesian inverse-variance weighting. This formulation enables fine-grained, modality-level fusion, supports asymmetric modality availability, and provides a principled mechanism to suppress corrupted or noisy modalities. Extensive experiments on connected autonomous driving scenarios for collaborative accident detection demonstrate that A2MAML consistently outperforms both single-agent and collaborative baselines, achieving up to 18.7% higher accident detection rate.

**arXiv ID:** 2602.04763
</details>

<details>
<summary><strong>El Agente Estructural: An Artificially Intelligent Molecular Editor</strong> - Changhyeok Choi, Yunheng Zou, Marcel Müller, Han Hao, Yeonghun Kang, Juan B. Pérez-Sánchez, Ignacio Gustin, Hanyong Xu, Mohammad Ghazi Vakili, Chris Crebolder, Alán Aspuru-Guzik, Varinia Bernales - [[pdf]](https://arxiv.org/pdf/2602.04849)</summary>

**Abstract:** We present El Agente Estructural, a multimodal, natural-language-driven geometry-generation and manipulation agent for autonomous chemistry and molecular modelling. Unlike molecular generation or editing via generative models, Estructural mimics how human experts directly manipulate molecular systems in three dimensions by integrating a comprehensive set of domain-informed tools and vision-language models. This design enables precise control over atomic or functional group replacements, atomic connectivity, and stereochemistry without the need to rebuild extensive core molecular frameworks. Through a series of representative case studies, we demonstrate that Estructural enables chemically meaningful geometry manipulation across a wide range of real-world scenarios. These include site-selective functionalization, ligand binding, ligand exchange, stereochemically controlled structure construction, isomer interconversion, fragment-level structural analysis, image-guided generation of structures from schematic reaction mechanisms, and mechanism-driven geometry generation and modification. These examples illustrate how multimodal reasoning, when combined with specialized geometry-aware tools, supports interactive and context-aware molecular modelling beyond structure generation. Looking forward, the integration of Estructural into El Agente Quntur, an autonomous multi-agent quantum chemistry platform, enhances its capabilities by adding sophisticated tools for the generation and editing of three-dimensional structures.

**arXiv ID:** 2602.04849
</details>

<details>
<summary><strong>El Agente Quntur: A research collaborator agent for quantum chemistry</strong> - Juan B. Pérez-Sánchez, Yunheng Zou, Jorge A. Campos-Gonzalez-Angulo, Marcel Müller, Ignacio Gustin, Andrew Wang, Han Hao, Tsz Wai Ko, Changhyeok Choi, Eric S. Isbrandt, Mohammad Ghazi Vakili, Hanyong Xu, Chris Crebolder, Varinia Bernales, Alán Aspuru-Guzik - [[pdf]](https://arxiv.org/pdf/2602.04850)</summary>

**Abstract:** Quantum chemistry is a foundational enabling tool for the fields of chemistry, materials science, computational biology and others. Despite of its power, the practical application of quantum chemistry simulations remains in the hands of qualified experts due to methodological complexity, software heterogeneity, and the need for informed interpretation of results. To bridge the accessibility gap for these tools and expand their reach to chemists with broader backgrounds, we introduce El Agente Quntur, a hierarchical, multi-agent AI system designed to operate not merely as an automation tool but as a research collaborator for computational quantum chemistry. Quntur was designed following three main strategies: i) elimination of hard-coded procedural policies in favour of reasoning-driven decisions, ii) construction of general and composable actions that facilitate generalization and efficiency, and iii) implementation of guided deep research to integrate abstract quantum-chemical reasoning across subdisciplines and a detailed understanding of the software's internal logic and syntax. Although instantiated in ORCA, these design principles are applicable to research agents more generally and easily expandable to additional quantum chemistry packages and beyond. Quntur supports the full range of calculations available in ORCA 6.0 and reasons over software documentation and scientific literature to plan, execute, adapt, and analyze in silico chemistry experiments following best practices. We discuss the advances and current bottlenecks in agentic systems operating at the research level in computational chemistry, and outline a roadmap toward a fully autonomous end-to-end computational chemistry research agent.

**arXiv ID:** 2602.04850
</details>

<details>
<summary><strong>Learning Decentralized LLM Collaboration with Multi-Agent Actor Critic</strong> - Shuo Liu, Tianle Chen, Ryan Amiri, Christopher Amato - [[pdf]](https://arxiv.org/pdf/2601.21972)</summary>

**Abstract:** Recent work has explored optimizing LLM collaboration through Multi-Agent Reinforcement Learning (MARL). However, most MARL fine-tuning approaches rely on predefined execution protocols, which often require centralized execution. Decentralized LLM collaboration is more appealing in practice, as agents can run inference in parallel with flexible deployments. Also, current approaches use Monte Carlo methods for fine-tuning, which suffer from high variance and thus require more samples to train effectively. Actor-critic methods are prevalent in MARL for dealing with these issues, so we developed Multi-Agent Actor-Critic (MAAC) methods to optimize decentralized LLM collaboration. In this paper, we analyze when and why these MAAC methods are beneficial. We propose 2 MAAC approaches, \textbf{CoLLM-CC} with a \textbf{C}entralized \textbf{C}ritic and \textbf{CoLLM-DC} with \textbf{D}ecentralized \textbf{C}ritics. Our experiments across writing, coding, and game-playing domains show that Monte Carlo methods and CoLLM-DC can achieve performance comparable to CoLLM-CC in short-horizon and dense-reward settings. However, they both underperform CoLLM-CC on long-horizon or sparse-reward tasks, where Monte Carlo methods require substantially more samples and CoLLM-DC struggles to converge. Our code is available at this https URL.

**arXiv ID:** 2601.21972
</details>

<details>
<summary><strong>On the Uncertainty of Large Language Model-Based Multi-Agent Systems</strong> - Yuxuan Zhao, Sijia Chen, Ningxin Su - [[pdf]](https://arxiv.org/pdf/2602.04234)</summary>

**Abstract:** Multi-agent systems (MAS) have emerged as a prominent paradigm for leveraging large language models (LLMs) to tackle complex tasks. However, the mechanisms governing the effectiveness of MAS built upon publicly available LLMs, specifically the underlying rationales for their success or failure, remain largely unexplored. In this paper, we revisit MAS through the perspective of uncertainty, considering both intra- and inter-agent dynamics by investigating entropy transitions during problem-solving across various topologies and six benchmark tasks. By analyzing 245 features spanning token-, trajectory-, and round-level entropy, we counterintuitively find that a single agent outperforms MAS in approximately 43.3% of cases, and that uncertainty dynamics are largely determined during the first round of interaction. Furthermore, we provide three key observations: 1) Certainty Preference: reducing uncertainty at any stage for any agent is critical for guaranteeing correct solutions; 2) Base Uncertainty: base models with lower entropy during problem-solving directly benefit MAS performance; and 3) Task Awareness: entropy dynamics of MAS play varying roles across different tasks. Building on these insights, we introduce a simple yet effective algorithm, the Entropy Judger, to select solutions from MAS's pass@k results, leading to consistent accuracy improvements across all MAS configurations and tasks. Our source code is available at this https URL.

**arXiv ID:** 2602.04234
</details>

<details>
<summary><strong>SimCity: Multi-Agent Urban Development Simulation with Rich Interactions</strong> - Yeqi Feng, Yucheng Lu, Hongyu Su, Yixin Tao, Tianxing He - [[pdf]](https://arxiv.org/pdf/2510.01297)</summary>

**Abstract:** Large Language Models (LLMs) open new possibilities for constructing realistic and interpretable macroeconomic simulations. We present SimCity, a multi-agent framework that leverages LLMs to model an interpretable macroeconomic system with heterogeneous agents and rich interactions. Unlike classical equilibrium models that limit heterogeneity for tractability, or traditional agent-based models (ABMs) that rely on hand-crafted decision rules, SimCity enables flexible, adaptive behavior with transparent natural-language reasoning. Within SimCity, four core agent types (households, firms, a central bank, and a government) deliberate and participate in a frictional labor market, a heterogeneous goods market, and a financial market. Furthermore, a Vision-Language Model (VLM) determines the geographic placement of new firms and renders a mapped virtual city, allowing us to study both macroeconomic regularities and urban expansion dynamics within a unified environment. To evaluate the framework, we compile a checklist of canonical macroeconomic phenomena, including price elasticity of demand, Engel's Law, Okun's Law, the Phillips Curve, and the Beveridge Curve, and show that SimCity naturally reproduces these empirical patterns while remaining robust across simulation runs.

**arXiv ID:** 2510.01297
</details>

<details>
<summary><strong>Cooperative Flexibility Exchange: Fair and Comfort-Aware Decentralized Resource Allocation</strong> - Rabiya Khalid, Evangelos Pournaras - [[pdf]](https://arxiv.org/pdf/2510.04192)</summary>

**Abstract:** The growing electricity demand and use of smart appliances are placing pressure on power grids, making efficient energy management more important than ever. The existing energy management systems often prioritize system efficiency (balanced energy demand and supply) at the expense of consumer comfort. This paper addresses this gap by proposing a novel decentralized multi-agent coordination-based demand-side management system. The proposed system enables individual agents to coordinate for demand-side energy optimization while improving consumer comfort and maintaining system efficiency. A key innovation of this work is the introduction of a slot exchange mechanism, where agents first receive optimized appliance-level energy consumption schedules and then coordinate with each other to adjust these schedules through slot exchanges to improve their comfort even when agents show non-altruistic behaviour. It also scales well with large populations and promotes fairness by balancing satisfaction levels across consumers. For performance evaluation, a real-world dataset is used, and the results demonstrate that the proposed slot exchange mechanism increases consumer comfort and fairness without raising system inefficiency cost, making it a practical and scalable solution for future smart grids.

**arXiv ID:** 2510.04192
</details>

<details>
<summary><strong>Bandwidth-constrained Variational Message Encoding for Cooperative Multi-agent Reinforcement Learning</strong> - Wei Duan, Jie Lu, En Yu, Junyu Xuan - [[pdf]](https://arxiv.org/pdf/2512.11179)</summary>

**Abstract:** Graph-based multi-agent reinforcement learning (MARL) enables coordinated behavior under partial observability by modeling agents as nodes and communication links as edges. While recent methods excel at learning sparse coordination graphs-determining who communicates with whom-they do not address what information should be transmitted under hard bandwidth constraints. We study this bandwidth-limited regime and show that naive dimensionality reduction consistently degrades coordination performance. Hard bandwidth constraints force selective encoding, but deterministic projections lack mechanisms to control how compression occurs. We introduce Bandwidth-constrained Variational Message Encoding (BVME), a lightweight module that treats messages as samples from learned Gaussian posteriors regularized via KL divergence to an uninformative prior. BVME's variational framework provides principled, tunable control over compression strength through interpretable hyperparameters, directly constraining the representations used for decision-making. Across SMACv1, SMACv2, and MPE benchmarks, BVME achieves comparable or superior performance while using 67--83% fewer message dimensions, with gains most pronounced on sparse graphs where message quality critically impacts coordination. Ablations reveal U-shaped sensitivity to bandwidth, with BVME excelling at extreme ratios while adding minimal overhead.

**arXiv ID:** 2512.11179
</details>

<details>
<summary><strong>DELTA: Deliberative Multi-Agent Reasoning with Reinforcement Learning for Multimodal Psychological Counseling</strong> - Jiangnan Yang, Junjie Chen, Fei Wang, Yiqi Nie, Yuxin Liu, Zhangling Duan, Jie Chen - [[pdf]](https://arxiv.org/pdf/2602.04112)</summary>

**Abstract:** Psychological counseling is a fundamentally multimodal cognitive process in which clinicians integrate verbal content with visual and vocal cues to infer clients' mental states and respond empathically. However, most existing language-model-based counseling systems operate on text alone and rely on implicit mental state inference. We introduce DELTA, a deliberative multi-agent framework that models counseling as a structured reasoning process over multimodal signals, separating evidence grounding, mental state abstraction, and response generation. DELTA further incorporates reinforcement learning guided by a distribution-level Emotion Attunement Score to encourage emotionally attuned responses. Experiments on a multimodal counseling benchmark show that DELTA improves both counseling quality and emotion attunement across models. Ablation and qualitative analyses suggest that explicit multimodal reasoning and structured mental state representations play complementary roles in supporting empathic human-AI interaction.

**arXiv ID:** 2602.04112
</details>

<details>
<summary><strong>MapCoder-Lite: Distilling Multi-Agent Coding into a Single Small LLM</strong> - Woongkyu Lee, Junhee Cho, Jungwook Choi - [[pdf]](https://arxiv.org/pdf/2509.17489)</summary>

**Abstract:** Large language models (LLMs) have advanced code generation from single-function tasks to competitive-programming problems, but existing multi-agent solutions either rely on costly large-scale (>30B) models or collapse when downsized to small open-source models. We present MapCoder-Lite, a framework for distilling the complex reasoning of large, multi-agent coding systems into a single 7B model. Our contribution is a novel, three-pillar methodology that synergistically generates, refines, and encodes multi-agent knowledge: (i) pass-based trajectory distillation from strong LLMs fixes format fragility in retrieval and reduces failures in debugging, (ii) supervisor-guided correction with global feedback strengthens planning and coding agents, and (iii) agent-wise LoRA fine-tuning delivers memory-efficient specialisation. Comprehensive evaluation on xCodeEval, APPS, and CodeContests shows that MapCoder-Lite more than doubles xCodeEval accuracy (from 13.2% to 28.3%), eliminates all format failures, while reducing GPU memory and token-generation time by 4x compared to a 32B model. It also achieves over 10% gains on simpler coding benchmarks, demonstrating broad improvements beyond competitive programming. These results demonstrate that careful agent-wise fine-tuning unleashes high-quality multi-agent coding on a small language model. Our code is publicly available at this https URL.

**arXiv ID:** 2509.17489
</details>

<details>
<summary><strong>DEBATE: A Large-Scale Benchmark for Evaluating Opinion Dynamics in Role-Playing LLM Agents</strong> - Yun-Shiuan Chuang, Ruixuan Tu, Chengtao Dai, Smit Vasani, You Li, Binwei Yao, Michael Henry Tessler, Sijia Yang, Dhavan Shah, Robert Hawkins, Junjie Hu, Timothy T. Rogers - [[pdf]](https://arxiv.org/pdf/2510.25110)</summary>

**Abstract:** Accurately modeling opinion change through social interactions is crucial for understanding and mitigating polarization, misinformation, and societal conflict. Recent work simulates opinion dynamics with role-playing LLM agents (RPLAs), but multi-agent simulations often display unnatural group behavior (e.g., premature convergence) and lack empirical benchmarks for assessing alignment with real human group interactions. We introduce DEBATE, a large-scale benchmark for evaluating the authenticity of opinion dynamics in multi-agent RPLA simulations. DEBATE contains 36,383 messages from 2,832 U.S.-based participants across 708 groups and 107 topics, with both public messages and private Likert-scale beliefs, enabling evaluation at the utterance and group levels (and supporting future individual-level analyses). We instantiate "digital twin" RPLAs with seven LLMs and evaluate across two settings: next-message prediction and full conversation rollout, using stance-alignment and opinion-convergence metrics. In zero-shot settings, RPLA groups exhibit strong opinion convergence relative to human groups. Post-training via supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) improves stance alignment and brings group-level convergence closer to human behavior, though discrepancies in opinion change and belief updating remain. DEBATE enables rigorous benchmarking of simulated opinion dynamics and supports future research on aligning multi-agent RPLAs with realistic human interactions.

**arXiv ID:** 2510.25110
</details>

<details>
<summary><strong>MIRROR: A Multi-Agent Framework with Iterative Adaptive Revision and Hierarchical Retrieval for Optimization Modeling in Operations Research</strong> - Yifan Shi, Jialong Shi, Jiayi Wang, Ye Fan, Jianyong Sun - [[pdf]](https://arxiv.org/pdf/2602.03318)</summary>

**Abstract:** Operations Research (OR) relies on expert-driven modeling-a slow and fragile process ill-suited to novel scenarios. While large language models (LLMs) can automatically translate natural language into optimization models, existing approaches either rely on costly post-training or employ multi-agent frameworks, yet most still lack reliable collaborative error correction and task-specific retrieval, often leading to incorrect outputs. We propose MIRROR, a fine-tuning-free, end-to-end multi-agent framework that directly translates natural language optimization problems into mathematical models and solver code. MIRROR integrates two core mechanisms: (1) execution-driven iterative adaptive revision for automatic error correction, and (2) hierarchical retrieval to fetch relevant modeling and coding exemplars from a carefully curated exemplar library. Experiments show that MIRROR outperforms existing methods on standard OR benchmarks, with notable results on complex industrial datasets such as IndustryOR and Mamo-ComplexLP. By combining precise external knowledge infusion with systematic error correction, MIRROR provides non-expert users with an efficient and reliable OR modeling solution, overcoming the fundamental limitations of general-purpose LLMs in expert optimization tasks.

**arXiv ID:** 2602.03318
</details>

<details>
<summary><strong>Autonomous AI Agents for Real-Time Affordable Housing Site Selection: Multi-Objective Reinforcement Learning Under Regulatory Constraints</strong> - Olaf Yunus Laitinen Imanov, Duygu Erisken, Derya Umut Kulali, Taner Yilmaz, Rana Irem Turhan - [[pdf]](https://arxiv.org/pdf/2602.03940)</summary>

**Abstract:** Affordable housing shortages affect billions, while land scarcity and regulations make site selection slow. We present AURA (Autonomous Urban Resource Allocator), a hierarchical multi-agent reinforcement learning system for real-time affordable housing site selection under hard regulatory constraints (QCT, DDA, LIHTC). We model the task as a constrained multi-objective Markov decision process optimizing accessibility, environmental impact, construction cost, and social equity while enforcing feasibility. AURA uses a regulatory-aware state encoding 127 federal and local constraints, Pareto-constrained policy gradients with feasibility guarantees, and reward decomposition separating immediate costs from long-term social outcomes. On datasets from 8 U.S. metros (47,392 candidate parcels), AURA attains 94.3% regulatory compliance and improves Pareto hypervolume by 37.2% over strong baselines. In a New York City 2026 case study, it reduces selection time from 18 months to 72 hours and identifies 23% more viable sites; chosen sites have 31% better transit access and 19% lower environmental impact than expert picks.

**arXiv ID:** 2602.03940
</details>

<details>
<summary><strong>MaMa: A Game-Theoretic Approach for Designing Safe Agentic Systems</strong> - Jonathan Nöther, Adish Singla, Goran Radanovic - [[pdf]](https://arxiv.org/pdf/2602.04431)</summary>

**Abstract:** LLM-based multi-agent systems have demonstrated impressive capabilities, but they also introduce significant safety risks when individual agents fail or behave adversarially. In this work, we study the automated design of agentic systems that remain safe even when a subset of agents is compromised. We formalize this challenge as a Stackelberg security game between a system designer (the Meta-Agent) and a best-responding Meta-Adversary that selects and compromises a subset of agents to minimize safety. We propose Meta-Adversary-Meta-Agent (MaMa), a novel algorithm for approximately solving this game and automatically designing safe agentic systems. Our approach uses LLM-based adversarial search, where the Meta-Agent iteratively proposes system designs and receives feedback based on the strongest attacks discovered by the Meta-Adversary. Empirical evaluations across diverse environments show that systems designed with MaMa consistently defend against worst-case attacks while maintaining performance comparable to systems optimized solely for task success. Moreover, the resulting systems generalize to stronger adversaries, as well as ones with different attack objectives or underlying LLMs, demonstrating robust safety beyond the training setting.

**arXiv ID:** 2602.04431
</details>

<details>
<summary><strong>Revisiting Multi-Agent Asynchronous Online Optimization with Delays: the Strongly Convex Case</strong> - Lingchan Bao, Tong Wei, Yuanyu Wan - [[pdf]](https://arxiv.org/pdf/2503.10013)</summary>

**Abstract:** We revisit multi-agent asynchronous online optimization with delays, where only one of the agents becomes active for making the decision at each round, and the corresponding feedback is received by all the agents after unknown delays. Although previous studies have established an $O(\sqrt{dT})$ regret bound for this problem, they assume that the maximum delay $d$ is knowable or the arrival order of feedback satisfies a special property, which may not hold in practice. In this paper, we surprisingly find that when the loss functions are strongly convex, these assumptions can be eliminated, and the existing regret bound can be significantly improved to $O(d\log T)$ meanwhile. Specifically, to exploit the strong convexity of functions, we first propose a delayed variant of the classical follow-the-leader algorithm, namely FTDL, which is very simple but requires the full information of functions as feedback. Moreover, to handle the more general case with only the gradient feedback, we develop an approximate variant of FTDL by combining it with surrogate loss functions. Experimental results show that the approximate FTDL outperforms the existing algorithm in the strongly convex case.

**arXiv ID:** 2503.10013
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>LLM-Empowered Cooperative Content Caching in Vehicular Fog Caching-Assisted Platoon Networks</strong> - Bowen Tan, Qiong Wu, Pingyi Fan, Kezhi Wang, Nan Cheng, Wen Chen - [[pdf]](https://arxiv.org/pdf/2602.04471)</summary>

**Abstract:** This letter proposes a novel three-tier content caching architecture for Vehicular Fog Caching (VFC)-assisted platoon, where the VFC is formed by the vehicles driving near the platoon. The system strategically coordinates storage across local platoon vehicles, dynamic VFC clusters, and cloud server (CS) to minimize content retrieval latency. To efficiently manage distributed storage, we integrate large language models (LLMs) for real-time and intelligent caching decisions. The proposed approach leverages LLMs' ability to process heterogeneous information, including user profiles, historical data, content characteristics, and dynamic system states. Through a designed prompting framework encoding task objectives and caching constraints, the LLMs formulate caching as a decision-making task, and our hierarchical deterministic caching mapping strategy enables adaptive requests prediction and precise content placement across three tiers without frequent retraining. Simulation results demonstrate the advantages of our proposed caching scheme.

**arXiv ID:** 2602.04471
</details>

<details>
<summary><strong>Scaling Agents for Computer Use</strong> - Gonzalo Gonzalez-Pumariega, Vincent Tu, Chih-Lun Lee, Jiachen Yang, Ang Li, Xin Eric Wang - [[pdf]](https://arxiv.org/pdf/2510.02250)</summary>

**Abstract:** Computer-use agents (CUAs) hold promise for automating everyday digital tasks, but their performance on long-horizon, complex problems remains unreliable. Single-rollout execution is brittle, with small errors compounding over time and leading to high variance in outcomes. While prior work has attempted to scale within a single rollout, such approaches have yielded limited gains. Scaling over multiple rollouts offers a more promising alternative but doing so effectively is challenging due to the difficulty of evaluating and selecting among long-horizon agent behaviors. We introduce Behavior Judge (BJudge), which addresses this challenge by representing agent executions as behavior narratives and comparing candidate behaviors at this level, substantially improving robustness and success rates. Using multiple rollouts, BJudge establishes a new state of the art (SoTA) in OSWorld at 72.6%, significantly outperforming prior methods and surpassing human-level performance at 72.36%, with comprehensive ablations validating key design choices. We further demonstrate strong generalization results to different operating systems on WindowsAgentArena and AndroidWorld. Crucially, our results highlight the strong effectiveness of scaling CUAs, when you do it right: effective scaling requires structured trajectory understanding and selection, and BJudge provides a practical framework to achieve this.

**arXiv ID:** 2510.02250
</details>

<details>
<summary><strong>Scaling Multiagent Systems with Process Rewards</strong> - Ed Li, Junyu Ren, Cat Yan - [[pdf]](https://arxiv.org/pdf/2601.23228)</summary>

**Abstract:** While multiagent systems have shown promise for tackling complex tasks via specialization, finetuning multiple agents simultaneously faces two key challenges: (1) credit assignment across agents, and (2) sample efficiency of expensive multiagent rollouts. In this work, we propose finetuning multiagent systems with per-action process rewards from AI feedback (MAPPA) to address both. Through assigning credit to individual agent actions rather than only at task completion, MAPPA enables fine-grained supervision without ground truth labels while extracting maximal training signal from each rollout. We demonstrate our approach on competition math problems and tool-augmented data analysis tasks. On unseen math problems, MAPPA achieves +5.0--17.5pp on AIME and +7.8--17.2pp on AMC. For data analysis tasks, our method improves success rate by +16.7pp while quality metrics improve by up to 47%, validating that per-action supervision can lead to improvements across different multiagent systems on various domains. By addressing these challenges, our work takes a first step toward scaling multiagent systems for complex, long-horizon tasks with minimal human supervision.

**arXiv ID:** 2601.23228
</details>

<details>
<summary><strong>AI-Generated Code Is Not Reproducible (Yet): An Empirical Study of Dependency Gaps in LLM-Based Coding Agents</strong> - Bhanu Prakash Vangala, Ali Adibifar, Ashish Gehani, Tanu Malik - [[pdf]](https://arxiv.org/pdf/2512.22387)</summary>

**Abstract:** The rise of Large Language Models (LLMs) as coding agents promises to accelerate software development, but their impact on generated code reproducibility remains largely unexplored. This paper presents an empirical study investigating whether LLM-generated code can be executed successfully in a clean environment with only OS packages and using only the dependencies that the model specifies. We evaluate three state-of-the-art LLM coding agents (Claude Code, OpenAI Codex, and Gemini) across 300 projects generated from 100 standardized prompts in Python, JavaScript, and Java. We introduce a three-layer dependency framework (distinguishing between claimed, working, and runtime dependencies) to quantify execution reproducibility. Our results show that only 68.3% of projects execute out-of-the-box, with substantial variation across languages (Python 89.2%, Java 44.0%). We also find a 13.5 times average expansion from declared to actual runtime dependencies, revealing significant hidden dependencies.

**arXiv ID:** 2512.22387
</details>

<details>
<summary><strong>Beyond the Vehicle: Cooperative Localization by Fusing Point Clouds for GPS-Challenged Urban Scenarios</strong> - Kuo-Yi Chao, Ralph Rasshofer, Alois Christian Knoll - [[pdf]](https://arxiv.org/pdf/2602.03908)</summary>

**Abstract:** Accurate vehicle localization is a critical challenge in urban environments where GPS signals are often unreliable. This paper presents a cooperative multi-sensor and multi-modal localization approach to address this issue by fusing data from vehicle-to-vehicle (V2V) and vehicle-to-infrastructure (V2I) systems. Our approach integrates cooperative data with a point cloud registration-based simultaneous localization and mapping (SLAM) algorithm. The system processes point clouds generated from diverse sensor modalities, including vehicle-mounted LiDAR and stereo cameras, as well as sensors deployed at intersections. By leveraging shared data from infrastructure, our method significantly improves localization accuracy and robustness in complex, GPS-noisy urban scenarios.

**arXiv ID:** 2602.03908
</details>

<details>
<summary><strong>A Modern System Recipe for Situated Embodied Human-Robot Conversation with Real-Time Multimodal LLMs and Tool-Calling</strong> - Dong Won Lee, Sarah Gillet, Louis-Philippe Morency, Cynthia Breazeal, Hae Won Park - [[pdf]](https://arxiv.org/pdf/2602.04157)</summary>

**Abstract:** Situated embodied conversation requires robots to interleave real-time dialogue with active perception: deciding what to look at, when to look, and what to say under tight latency constraints. We present a simple, minimal system recipe that pairs a real-time multimodal language model with a small set of tool interfaces for attention and active perception. We study six home-style scenarios that require frequent attention shifts and increasing perceptual scope. Across four system variants, we evaluate turn-level tool-decision correctness against human annotations and collect subjective ratings of interaction quality. Results indicate that real-time multimodal large language models and tool use for active perception is a promising direction for practical situated embodied conversation.

**arXiv ID:** 2602.04157
</details>

<details>
<summary><strong>AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation</strong> - Jin-Chuan Shi, Binhong Ye, Tao Liu, Junzhe He, Yangjinhui Xu, Xiaoyang Liu, Zeju Li, Hao Chen, Chunhua Shen - [[pdf]](https://arxiv.org/pdf/2602.04672)</summary>

**Abstract:** Reconstructing dynamic hand-object interactions from monocular videos is critical for dexterous manipulation data collection and creating realistic digital twins for robotics and VR. However, current methods face two prohibitive barriers: (1) reliance on neural rendering often yields fragmented, non-simulation-ready geometries under heavy occlusion, and (2) dependence on brittle Structure-from-Motion (SfM) initialization leads to frequent failures on in-the-wild footage. To overcome these limitations, we introduce AGILE, a robust framework that shifts the paradigm from reconstruction to agentic generation for interaction learning. First, we employ an agentic pipeline where a Vision-Language Model (VLM) guides a generative model to synthesize a complete, watertight object mesh with high-fidelity texture, independent of video occlusions. Second, bypassing fragile SfM entirely, we propose a robust anchor-and-track strategy. We initialize the object pose at a single interaction onset frame using a foundation model and propagate it temporally by leveraging the strong visual similarity between our generated asset and video observations. Finally, a contact-aware optimization integrates semantic, geometric, and interaction stability constraints to enforce physical plausibility. Extensive experiments on HO3D, DexYCB, and in-the-wild videos reveal that AGILE outperforms baselines in global geometric accuracy while demonstrating exceptional robustness on challenging sequences where prior art frequently collapses. By prioritizing physical validity, our method produces simulation-ready assets validated via real-to-sim retargeting for robotic applications.

**arXiv ID:** 2602.04672
</details>

<details>
<summary><strong>A Unified Candidate Set with Scene-Adaptive Refinement via Diffusion for End-to-End Autonomous Driving</strong> - Zhengfei Wu, Shuaixi Pan, Shuohan Chen, Shuo Yang, Yanjun Huang - [[pdf]](https://arxiv.org/pdf/2602.03112)</summary>

**Abstract:** End-to-end autonomous driving is increasingly adopting a multimodal planning paradigm that generates multiple trajectory candidates and selects the final plan, making candidate-set design critical. A fixed trajectory vocabulary provides stable coverage in routine driving but often misses optimal solutions in complex interactions, while scene-adaptive refinement can cause over-correction in simple scenarios by unnecessarily perturbing already strong vocabulary this http URL propose CdDrive, which preserves the original vocabulary candidates and augments them with scene-adaptive candidates generated by vocabulary-conditioned diffusion denoising. Both candidate types are jointly scored by a shared selection module, enabling reliable performance across routine and highly interactive scenarios. We further introduce HATNA (Horizon-Aware Trajectory Noise Adapter) to improve the smoothness and geometric continuity of diffusion candidates via temporal smoothing and horizon-aware noise modulation. Experiments on NAVSIM v1 and NAVSIM v2 demonstrate leading performance, and ablations verify the contribution of each component. Code: this https URL.

**arXiv ID:** 2602.03112
</details>

<details>
<summary><strong>Making Videos Accessible for Blind and Low Vision Users Using a Multimodal Agent Video Player</strong> - Adriana Olmos, Anoop K. Sinha, Renelito Delos Santos, Ruben Rodriguez Rodriguez, James A. Landay, Sam S. Sepah, Philip Nelson, Shaun K. Kane - [[pdf]](https://arxiv.org/pdf/2602.04104)</summary>

**Abstract:** Video content remains largely inaccessible to blind and low-vision (BLV) users. To address this, we introduce a prototype that leverages a multimodal agent - powered by a novel conversational architecture using a multimodal large language model (MLLM) - to provide BLV users with an interactive, accessible video experience. This Multimodal Agent Video Player (MAVP) demonstrates that an interactive accessibility mode can be added to a video through multilayered prompt orchestration. We describe a user-centered design process involving 18 sessions with BLV users that showed that BLV users do not just want accessibility features, but desire independence and personal agency over the viewing experience. We conducted a qualitative study with an additional 8 BLV participants; in this, we saw that the MAVP's conversational dialogue offers BLV users a sense of personal agency, fostering collaboration and trust. Even in the case of hallucinations, it is meta-conversational dialogues about AI's limitations that can repair trust.

**arXiv ID:** 2602.04104
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (26 papers)</h2></summary>

<details>
<summary><strong>Digital Twins & ZeroConf AI: Structuring Automated Intelligent Pipelines for Industrial Applications</strong> - Marco Picone, Fabio Turazza, Matteo Martinelli, Marco Mamei - [[pdf]](https://arxiv.org/pdf/2602.04385)</summary>

**Abstract:** The increasing complexity of Cyber-Physical Systems (CPS), particularly in the industrial domain, has amplified the challenges associated with the effective integration of Artificial Intelligence (AI) and Machine Learning (ML) techniques. Fragmentation across IoT and IIoT technologies, manifested through diverse communication protocols, data formats and device capabilities, creates a substantial gap between low-level physical layers and high-level intelligent functionalities. Recently, Digital Twin (DT) technology has emerged as a promising solution, offering structured, interoperable and semantically rich digital representations of physical assets. Current approaches are often siloed and tightly coupled, limiting scalability and reuse of AI functionalities. This work proposes a modular and interoperable solution that enables seamless AI pipeline integration into CPS by minimizing configuration and decoupling the roles of DTs and AI components. We introduce the concept of Zero Configuration (ZeroConf) AI pipelines, where DTs orchestrate data management and intelligent augmentation. The approach is demonstrated in a MicroFactory scenario, showing support for concurrent ML models and dynamic data processing, effectively accelerating the deployment of intelligent services in complex industrial settings.

**arXiv ID:** 2602.04385
</details>

<details>
<summary><strong>EMA Policy Gradient: Taming Reinforcement Learning for LLMs with EMA Anchor and Top-k KL</strong> - Lunjun Zhang, Jimmy Ba - [[pdf]](https://arxiv.org/pdf/2602.04417)</summary>

**Abstract:** Reinforcement Learning (RL) has enabled Large Language Models (LLMs) to acquire increasingly complex reasoning and agentic behaviors. In this work, we propose two simple techniques to improve policy gradient algorithms for LLMs. First, we replace the fixed anchor policy during RL with an Exponential Moving Average (EMA), similar to a target network in deep Q-learning. Second, we introduce Top-k KL estimator, which allows for flexible interpolation between exact KL and sampled KL. We derive the stability conditions for using EMA anchor; moreover, we show that our Top-k KL estimator yields both unbiased KL values and unbiased gradients at any k, while bringing the benefits of exact KL. When combined with GRPO, the two techniques (EMA-PG) lead to a significant performance boost. On math reasoning, it allows R1-distilled Qwen-1.5B to reach 53.9% on OlympiadBench compared to 50.8% by GRPO. On agentic RL domains, with Qwen-3B base, EMA-PG improves GRPO by an average of 33.3% across 7 datasets of Q&A with search engines, including 29.7% $\rightarrow$ 44.1% on HotpotQA, 27.4% $\rightarrow$ 40.1% on 2WikiMultiHopQA. Overall, we show that EMA-PG is a simple, principled, and powerful approach to scaling RL for LLMs. Code: this https URL

**arXiv ID:** 2602.04417
</details>

<details>
<summary><strong>Learning the Value Systems of Agents with Preference-based and Inverse Reinforcement Learning</strong> - Andrés Holgado-Sánchez, Holger Billhardt, Alberto Fernández, Sascha Ossowski - [[pdf]](https://arxiv.org/pdf/2602.04518)</summary>

**Abstract:** Agreement Technologies refer to open computer systems in which autonomous software agents interact with one another, typically on behalf of humans, in order to come to mutually acceptable agreements. With the advance of AI systems in recent years, it has become apparent that such agreements, in order to be acceptable to the involved parties, must remain aligned with ethical principles and moral values. However, this is notoriously difficult to ensure, especially as different human users (and their software agents) may hold different value systems, i.e. they may differently weigh the importance of individual moral values. Furthermore, it is often hard to specify the precise meaning of a value in a particular context in a computational manner. Methods to estimate value systems based on human-engineered specifications, e.g. based on value surveys, are limited in scale due to the need for intense human moderation. In this article, we propose a novel method to automatically \emph{learn} value systems from observations and human demonstrations. In particular, we propose a formal model of the \emph{value system learning} problem, its instantiation to sequential decision-making domains based on multi-objective Markov decision processes, as well as tailored preference-based and inverse reinforcement learning algorithms to infer value grounding functions and value systems. The approach is illustrated and evaluated by two simulated use cases.

**arXiv ID:** 2602.04518
</details>

<details>
<summary><strong>Towards Structured, State-Aware, and Execution-Grounded Reasoning for Software Engineering Agents</strong> - Tse-Hsun, Chen - [[pdf]](https://arxiv.org/pdf/2602.04640)</summary>

**Abstract:** Software Engineering (SE) agents have shown promising abilities in supporting various SE tasks. Current SE agents remain fundamentally reactive, making decisions mainly based on conversation history and the most recent response. However, this reactive design provides no explicit structure or persistent state within the agent's memory, making long-horizon reasoning challenging. As a result, SE agents struggle to maintain a coherent understanding across reasoning steps, adapt their hypotheses as new evidence emerges, or incorporate execution feedback into the mental reasoning model of the system state.
In this position paper, we argue that, to further advance SE agents, we need to move beyond reactive behavior toward a structured, state-aware, and execution-grounded reasoning. We outline how explicit structure, persistent and evolving state, and the integration of execution-grounded feedback can help SE agents perform more coherent and reliable reasoning in long-horizon tasks. We also provide an initial roadmap for developing next-generation SE agents that can more effectively perform real-world tasks.

**arXiv ID:** 2602.04640
</details>

<details>
<summary><strong>Rethinking the Design Space of Reinforcement Learning for Diffusion Models: On the Importance of Likelihood Estimation Beyond Loss Design</strong> - Jaemoo Choi, Yuchen Zhu, Wei Guo, Petr Molodyk, Bo Yuan, Jinbin Bai, Yi Xin, Molei Tao, Yongxin Chen - [[pdf]](https://arxiv.org/pdf/2602.04663)</summary>

**Abstract:** Reinforcement learning has been widely applied to diffusion and flow models for visual tasks such as text-to-image generation. However, these tasks remain challenging because diffusion models have intractable likelihoods, which creates a barrier for directly applying popular policy-gradient type methods. Existing approaches primarily focus on crafting new objectives built on already heavily engineered LLM objectives, using ad hoc estimators for likelihood, without a thorough investigation into how such estimation affects overall algorithmic performance. In this work, we provide a systematic analysis of the RL design space by disentangling three factors: i) policy-gradient objectives, ii) likelihood estimators, and iii) rollout sampling schemes. We show that adopting an evidence lower bound (ELBO) based model likelihood estimator, computed only from the final generated sample, is the dominant factor enabling effective, efficient, and stable RL optimization, outweighing the impact of the specific policy-gradient loss functional. We validate our findings across multiple reward benchmarks using SD 3.5 Medium, and observe consistent trends across all tasks. Our method improves the GenEval score from 0.24 to 0.95 in 90 GPU hours, which is $4.6\times$ more efficient than FlowGRPO and $2\times$ more efficient than the SOTA method DiffusionNFT without reward hacking.

**arXiv ID:** 2602.04663
</details>

<details>
<summary><strong>Supporting software engineering tasks with agentic AI: Demonstration on document retrieval and test scenario generation</strong> - Marian Kica, Lukas Radosky, David Slivka, Karin Kubinova, Daniel Dovhun, Tomas Uhercik, Erik Bircak, Ivan Polasek - [[pdf]](https://arxiv.org/pdf/2602.04726)</summary>

**Abstract:** The introduction of large language models ignited great retooling and rethinking of the software development models. The ensuing response of software engineering research yielded a massive body of tools and approaches. In this paper, we join the hassle by introducing agentic AI solutions for two tasks. First, we developed a solution for automatic test scenario generation from a detailed requirements description. This approach relies on specialized worker agents forming a star topology with the supervisor agent in the middle. We demonstrate its capabilities on a real-world example. Second, we developed an agentic AI solution for the document retrieval task in the context of software engineering documents. Our solution enables performing various use cases on a body of documents related to the development of a single software, including search, question answering, tracking changes, and large document summarization. In this case, each use case is handled by a dedicated LLM-based agent, which performs all subtasks related to the corresponding use case. We conclude by hinting at the future perspectives of our line of research.

**arXiv ID:** 2602.04726
</details>

<details>
<summary><strong>Beyond Rewards in Reinforcement Learning for Cyber Defence</strong> - Elizabeth Bates, Chris Hicks, Vasilios Mavroudis - [[pdf]](https://arxiv.org/pdf/2602.04809)</summary>

**Abstract:** Recent years have seen an explosion of interest in autonomous cyber defence agents trained to defend computer networks using deep reinforcement learning. These agents are typically trained in cyber gym environments using dense, highly engineered reward functions which combine many penalties and incentives for a range of (un)desirable states and costly actions. Dense rewards help alleviate the challenge of exploring complex environments but risk biasing agents towards suboptimal and potentially riskier solutions, a critical issue in complex cyber environments. We thoroughly evaluate the impact of reward function structure on learning and policy behavioural characteristics using a variety of sparse and dense reward functions, two well-established cyber gyms, a range of network sizes, and both policy gradient and value-based RL algorithms. Our evaluation is enabled by a novel ground truth evaluation approach which allows directly comparing between different reward functions, illuminating the nuanced inter-relationships between rewards, action space and the risks of suboptimal policies in cyber environments. Our results show that sparse rewards, provided they are goal aligned and can be encountered frequently, uniquely offer both enhanced training reliability and more effective cyber defence agents with lower-risk policies. Surprisingly, sparse rewards can also yield policies that are better aligned with cyber defender goals and make sparing use of costly defensive actions without explicit reward-based numerical penalties.

**arXiv ID:** 2602.04809
</details>

<details>
<summary><strong>Safe Urban Traffic Control via Uncertainty-Aware Conformal Prediction and World-Model Reinforcement Learning</strong> - Joydeep Chandra, Satyam Kumar Navneet, Aleksandr Algazinov, Yong Zhang - [[pdf]](https://arxiv.org/pdf/2602.04821)</summary>

**Abstract:** Urban traffic management demands systems that simultaneously predict future conditions, detect anomalies, and take safe corrective actions -- all while providing reliability guarantees. We present STREAM-RL, a unified framework that introduces three novel algorithmic contributions: (1) PU-GAT+, an Uncertainty-Guided Adaptive Conformal Forecaster that uses prediction uncertainty to dynamically reweight graph attention via confidence-monotonic attention, achieving distribution-free coverage guarantees; (2) CRFN-BY, a Conformal Residual Flow Network that models uncertainty-normalized residuals via normalizing flows with Benjamini-Yekutieli FDR control under arbitrary dependence; and (3) LyCon-WRL+, an Uncertainty-Guided Safe World-Model RL agent with Lyapunov stability certificates, certified Lipschitz bounds, and uncertainty-propagated imagination rollouts. To our knowledge, this is the first framework to propagate calibrated uncertainty from forecasting through anomaly detection to safe policy learning with end-to-end theoretical guarantees. Experiments on multiple real-world traffic trajectory data demonstrate that STREAM-RL achieves 91.4\% coverage efficiency, controls FDR at 4.1\% under verified dependence, and improves safety rate to 95.2\% compared to 69\% for standard PPO while achieving higher reward, with 23ms end-to-end inference latency.

**arXiv ID:** 2602.04821
</details>

<details>
<summary><strong>CRoSS: A Continual Robotic Simulation Suite for Scalable Reinforcement Learning with High Task Diversity and Realistic Physics Simulation</strong> - Yannick Denker, Alexander Gepperth - [[pdf]](https://arxiv.org/pdf/2602.04868)</summary>

**Abstract:** Continual reinforcement learning (CRL) requires agents to learn from a sequence of tasks without forgetting previously acquired policies. In this work, we introduce a novel benchmark suite for CRL based on realistically simulated robots in the Gazebo simulator. Our Continual Robotic Simulation Suite (CRoSS) benchmarks rely on two robotic platforms: a two-wheeled differential-drive robot with lidar, camera and bumper sensor, and a robotic arm with seven joints. The former represent an agent in line-following and object-pushing scenarios, where variation of visual and structural parameters yields a large number of distinct tasks, whereas the latter is used in two goal-reaching scenarios with high-level cartesian hand position control (modeled after the Continual World benchmark), and low-level control based on joint angles. For the robotic arm benchmarks, we provide additional kinematics-only variants that bypass the need for physical simulation (as long as no sensor readings are required), and which can be run two orders of magnitude faster. CRoSS is designed to be easily extensible and enables controlled studies of continual reinforcement learning in robotic settings with high physical realism, and in particular allow the use of almost arbitrary simulated sensors. To ensure reproducibility and ease of use, we provide a containerized setup (Apptainer) that runs out-of-the-box, and report performances of standard RL algorithms, including Deep Q-Networks (DQN) and policy gradient methods. This highlights the suitability as a scalable and reproducible benchmark for CRL research.

**arXiv ID:** 2602.04868
</details>

<details>
<summary><strong>Rethinking the Trust Region in LLM Reinforcement Learning</strong> - Penghui Qi, Xiangxin Zhou, Zichen Liu, Tianyu Pang, Chao Du, Min Lin, Wee Sun Lee - [[pdf]](https://arxiv.org/pdf/2602.04879)</summary>

**Abstract:** Reinforcement learning (RL) has become a cornerstone for fine-tuning Large Language Models (LLMs), with Proximal Policy Optimization (PPO) serving as the de facto standard algorithm. Despite its ubiquity, we argue that the core ratio clipping mechanism in PPO is structurally ill-suited for the large vocabularies inherent to LLMs. PPO constrains policy updates based on the probability ratio of sampled tokens, which serves as a noisy single-sample Monte Carlo estimate of the true policy divergence. This creates a sub-optimal learning dynamic: updates to low-probability tokens are aggressively over-penalized, while potentially catastrophic shifts in high-probability tokens are under-constrained, leading to training inefficiency and instability. To address this, we propose Divergence Proximal Policy Optimization (DPPO), which substitutes heuristic clipping with a more principled constraint based on a direct estimate of policy divergence (e.g., Total Variation or KL). To avoid huge memory footprint, we introduce the efficient Binary and Top-K approximations to capture the essential divergence with negligible overhead. Extensive empirical evaluations demonstrate that DPPO achieves superior training stability and efficiency compared to existing methods, offering a more robust foundation for RL-based LLM fine-tuning.

**arXiv ID:** 2602.04879
</details>

<details>
<summary><strong>Information Templates: A New Paradigm for Intelligent Active Feature Acquisition</strong> - Hung-Tien Huang, Dzung Dinh, Junier B. Oliva - [[pdf]](https://arxiv.org/pdf/2508.18380)</summary>

**Abstract:** Active feature acquisition (AFA) is an instance-adaptive paradigm in which, at inference time, a policy sequentially chooses which features to acquire (at a cost) before predicting. Existing approaches either train reinforcement learning policies, which deal with a difficult MDP, or greedy policies that cannot account for the joint informativeness of features or require knowledge about the underlying data distribution. To overcome this, we propose Template-based AFA (TAFA), a non-greedy framework that learns a small library of feature templates -- sets of features that are jointly informative -- and uses this library of templates to guide the next feature acquisitions. Through identifying feature templates, the proposed framework not only significantly reduces the action space considered by the policy but also alleviates the need to estimate the underlying data distribution. Extensive experiments on synthetic and real-world datasets show that TAFA outperforms the existing state-of-the-art baselines while achieving lower overall acquisition cost and computation.

**arXiv ID:** 2508.18380
</details>

<details>
<summary><strong>DeepAgent: A General Reasoning Agent with Scalable Toolsets</strong> - Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Guanting Dong, Jiajie Jin, Yinuo Wang, Hao Wang, Yutao Zhu, Ji-Rong Wen, Yuan Lu, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2510.21618)</summary>

**Abstract:** Large reasoning models have demonstrated strong problem-solving abilities, yet real-world tasks often require external tools and long-horizon interactions. Existing agent frameworks typically follow predefined workflows, which limit autonomous and global task completion. In this paper, we introduce DeepAgent, an end-to-end deep reasoning agent that performs autonomous thinking, tool discovery, and action execution within a single, coherent reasoning process. To address the challenges of long-horizon interactions, particularly the context length explosion from multiple tool calls and the accumulation of interaction history, we introduce an autonomous memory folding mechanism that compresses past interactions into structured episodic, working, and tool memories, reducing error accumulation while preserving critical information. To teach general-purpose tool use efficiently and stably, we develop an end-to-end reinforcement learning strategy, namely ToolPO, that leverages LLM-simulated APIs and applies tool-call advantage attribution to assign fine-grained credit to the tool invocation tokens. Extensive experiments on eight benchmarks, including general tool-use tasks (ToolBench, API-Bank, TMDB, Spotify, ToolHop) and downstream applications (ALFWorld, WebShop, GAIA, HLE), demonstrate that DeepAgent consistently outperforms baselines across both labeled-tool and open-set tool retrieval scenarios. This work takes a step toward more general and capable agents for real-world applications. The code and demo are available at this https URL.

**arXiv ID:** 2510.21618
</details>

<details>
<summary><strong>CastMind: An Interaction-Driven Agentic Reasoning Framework for Cognition-Inspired Time Series Forecasting</strong> - Xiaohan Zhang, Tian Gao, Mingyue Cheng, Bokai Pan, Ze Guo, Yaguo Liu, Xiaoyu Tao, Qi Liu - [[pdf]](https://arxiv.org/pdf/2511.08947)</summary>

**Abstract:** Time series forecasting plays a crucial role in decision-making across many real-world applications. Despite substantial progress, most existing methods still treat forecasting as a static, single-pass regression problem. In contrast, human experts form predictions through iterative reasoning that integrates temporal features, domain knowledge, case-based references, and supplementary context, with continuous refinement. In this work, we propose CastMind, an interaction-driven agentic reasoning framework that enables accurate time series forecasting with training-free large language models. CastMind reformulates forecasting as an expert-like process and organizes it into a multi-stage workflow involving context preparation, reasoning-based generation, and reflective evaluation, transforming forecasting from a single-pass output into a multi-turn, autonomous interaction process. To support diverse perspectives commonly considered by human experts, we develop a lightweight toolkit comprising a feature set, a knowledge base, a case library, and a contextual pool that provides external support for LLM-based reasoning. Extensive experiments across multiple benchmarks show that CastMind generally outperforms representative baselines. Code is available at this repository: this https URL .

**arXiv ID:** 2511.08947
</details>

<details>
<summary><strong>DEEPMED: Building a Medical DeepResearch Agent via Multi-hop Med-Search Data and Turn-Controlled Agentic Training & Inference</strong> - Zihan Wang, Hao Wang, Shi Feng, Xiaocui Yang, Daling Wang, Yiqun Zhang, Jinghao Lin, Haihua Yang, Xiaozhong Ji - [[pdf]](https://arxiv.org/pdf/2601.18496)</summary>

**Abstract:** Medical reasoning models remain constrained by parametric knowledge and are thus susceptible to forgetting and hallucinations. DeepResearch (DR) models ground outputs in verifiable evidence from tools and perform strongly in general domains, but their direct transfer to medical field yields relatively limited gains. We attribute this to two gaps: task characteristic and tool-use scaling. Medical questions require evidence interpretation in a knowledge-intensive clinical context; while general DR models can retrieve information, they often lack clinical-context reasoning and thus "find it but fail to use it," leaving performance limited by medical abilities. Moreover, in medical scenarios, blindly scaling tool-call can inject noisy context, derailing sensitive medical reasoning and prompting repetitive evidence-seeking along incorrect paths. Therefore, we propose DeepMed. For data, we deploy a multi-hop med-search QA synthesis method supporting the model to apply the DR paradigm in medical contexts. For training, we introduce a difficulty-aware turn-penalty to suppress excessive tool-call growth. For inference, we bring a monitor to help validate hypotheses within a controlled number of steps and avoid context rot. Overall, on seven medical benchmarks, DeepMed improves its base model by 9.79\% on average and outperforms larger medical reasoning and DR models.

**arXiv ID:** 2601.18496
</details>

<details>
<summary><strong>Scaling Agentic Verifier for Competitive Coding</strong> - Zeyao Ma, Jing Zhang, Xiaokang Zhang, Jiaxi Yang, Zongmeng Zhang, Jiajun Zhang, Yuheng Jing, Lei Zhang, Hao Zheng, Wenting Zhao, Junyang Lin, Binyuan Hui - [[pdf]](https://arxiv.org/pdf/2602.04254)</summary>

**Abstract:** Large language models (LLMs) have demonstrated strong coding capabilities but still struggle to solve competitive programming problems correctly in a single attempt. Execution-based re-ranking offers a promising test-time scaling strategy, yet existing methods are constrained by either difficult test case generation or inefficient random input sampling. To address this limitation, we propose Agentic Verifier, an execution-based agent that actively reasons about program behaviors and searches for highly discriminative test inputs that expose behavioral discrepancies among candidate solutions. Through multi-turn interaction with code execution environments, the verifier iteratively refines the candidate input generator and produces targeted counterexamples rather than blindly sampling inputs. We train the verifier to acquire this discriminative input generation capability via a scalable pipeline combining large-scale data synthesis, rejection fine-tuning, and agentic reinforcement learning. Extensive experiments across five competitive programming benchmarks demonstrate consistent improvements over strong execution-based baselines, achieving up to +10-15% absolute gains in Best@K accuracy. Further analysis reveals clear test-time scaling behavior and highlights the verifier's broader potential beyond reranking.

**arXiv ID:** 2602.04254
</details>

<details>
<summary><strong>OmniRAG-Agent: Agentic Omnimodal Reasoning for Low-Resource Long Audio-Video Question Answering</strong> - Yifan Zhu, Xinyu Mu, Tao Feng, Zhonghong Ou, Yuning Gong, Haoran Luo - [[pdf]](https://arxiv.org/pdf/2602.03707)</summary>

**Abstract:** Long-horizon omnimodal question answering answers questions by reasoning over text, images, audio, and video. Despite recent progress on OmniLLMs, low-resource long audio-video QA still suffers from costly dense encoding, weak fine-grained retrieval, limited proactive planning, and no clear end-to-end this http URL address these issues, we propose OmniRAG-Agent, an agentic omnimodal QA method for budgeted long audio-video reasoning. It builds an image-audio retrieval-augmented generation module that lets an OmniLLM fetch short, relevant frames and audio snippets from external banks. Moreover, it uses an agent loop that plans, calls tools across turns, and merges retrieved evidence to answer complex queries. Furthermore, we apply group relative policy optimization to jointly improve tool use and answer quality over time. Experiments on OmniVideoBench, WorldSense, and Daily-Omni show that OmniRAG-Agent consistently outperforms prior methods under low-resource settings and achieves strong results, with ablations validating each component.

**arXiv ID:** 2602.03707
</details>

<details>
<summary><strong>Agentic AI-Empowered Dynamic Survey Framework</strong> - Furkan Mumcu, Lokman Bekit, Michael J. Jones, Anoop Cherian, Yasin Yilmaz - [[pdf]](https://arxiv.org/pdf/2602.04071)</summary>

**Abstract:** Survey papers play a central role in synthesizing and organizing scientific knowledge, yet they are increasingly strained by the rapid growth of research output. As new work continues to appear after publication, surveys quickly become outdated, contributing to redundancy and fragmentation in the literature. We reframe survey writing as a long-horizon maintenance problem rather than a one-time generation task, treating surveys as living documents that evolve alongside the research they describe. We propose an agentic Dynamic Survey Framework that supports the continuous updating of existing survey papers by incrementally integrating new work while preserving survey structure and minimizing unnecessary disruption. Using a retrospective experimental setup, we demonstrate that the proposed framework effectively identifies and incorporates emerging research while preserving the coherence and structure of existing surveys.

**arXiv ID:** 2602.04071
</details>

<details>
<summary><strong>Decoupling Time and Risk: Risk-Sensitive Reinforcement Learning with General Discounting</strong> - Mehrdad Moghimi, Anthony Coache, Hyejin Ku - [[pdf]](https://arxiv.org/pdf/2602.04131)</summary>

**Abstract:** Distributional reinforcement learning (RL) is a powerful framework increasingly adopted in safety-critical domains for its ability to optimize risk-sensitive objectives. However, the role of the discount factor is often overlooked, as it is typically treated as a fixed parameter of the Markov decision process or tunable hyperparameter, with little consideration of its effect on the learned policy. In the literature, it is well-known that the discounting function plays a major role in characterizing time preferences of an agent, which an exponential discount factor cannot fully capture. Building on this insight, we propose a novel framework that supports flexible discounting of future rewards and optimization of risk measures in distributional RL. We provide a technical analysis of the optimality of our algorithms, show that our multi-horizon extension fixes issues raised with existing methodologies, and validate the robustness of our methods through extensive experiments. Our results highlight that discounting is a cornerstone in decision-making problems for capturing more expressive temporal and risk preferences profiles, with potential implications for real-world safety-critical applications.

**arXiv ID:** 2602.04131
</details>

<details>
<summary><strong>Stochastic Decision Horizons for Constrained Reinforcement Learning</strong> - Nikola Milosevic, Leonard Franz, Daniel Haeufle, Georg Martius, Nico Scherf, Pavel Kolev - [[pdf]](https://arxiv.org/pdf/2602.04599)</summary>

**Abstract:** Constrained Markov decision processes (CMDPs) provide a principled model for handling constraints, such as safety and other auxiliary objectives, in reinforcement learning. The common approach of using additive-cost constraints and dual variables often hinders off-policy scalability. We propose a Control as Inference formulation based on stochastic decision horizons, where constraint violations attenuate reward contributions and shorten the effective planning horizon via state-action-dependent continuation. This yields survival-weighted objectives that remain replay-compatible for off-policy actor-critic learning. We propose two violation semantics, absorbing and virtual termination, that share the same survival-weighted return but result in distinct optimization structures that lead to SAC/MPO-style policy improvement. Experiments demonstrate improved sample efficiency and favorable return-violation trade-offs on standard benchmarks. Moreover, MPO with virtual termination (VT-MPO) scales effectively to our high-dimensional musculoskeletal Hyfydy setup.

**arXiv ID:** 2602.04599
</details>

<details>
<summary><strong>Rationality Measurement and Theory for Reinforcement Learning Agents</strong> - Kejiang Qian, Amos Storkey, Fengxiang He - [[pdf]](https://arxiv.org/pdf/2602.04737)</summary>

**Abstract:** This paper proposes a suite of rationality measures and associated theory for reinforcement learning agents, a property increasingly critical yet rarely explored. We define an action in deployment to be perfectly rational if it maximises the hidden true value function in the steepest direction. The expected value discrepancy of a policy's actions against their rational counterparts, culminating over the trajectory in deployment, is defined to be expected rational risk; an empirical average version in training is also defined. Their difference, termed as rational risk gap, is decomposed into (1) an extrinsic component caused by environment shifts between training and deployment, and (2) an intrinsic one due to the algorithm's generalisability in a dynamic environment. They are upper bounded by, respectively, (1) the $1$-Wasserstein distance between transition kernels and initial state distributions in training and deployment, and (2) the empirical Rademacher complexity of the value function class. Our theory suggests hypotheses on the benefits from regularisers (including layer normalisation, $\ell_2$ regularisation, and weight normalisation) and domain randomisation, as well as the harm from environment shifts. Experiments are in full agreement with these hypotheses. The code is available at this https URL.

**arXiv ID:** 2602.04737
</details>

<details>
<summary><strong>HoRD: Robust Humanoid Control via History-Conditioned Reinforcement Learning and Online Distillation</strong> - Puyue Wang, Jiawei Hu, Yan Gao, Junyan Wang, Yu Zhang, Gillian Dobbie, Tao Gu, Wafa Johal, Ting Dang, Hong Jia - [[pdf]](https://arxiv.org/pdf/2602.04412)</summary>

**Abstract:** Humanoid robots can suffer significant performance drops under small changes in dynamics, task specifications, or environment setup. We propose HoRD, a two-stage learning framework for robust humanoid control under domain shift. First, we train a high-performance teacher policy via history-conditioned reinforcement learning, where the policy infers latent dynamics context from recent state--action trajectories to adapt online to diverse randomized dynamics. Second, we perform online distillation to transfer the teacher's robust control capabilities into a transformer-based student policy that operates on sparse root-relative 3D joint keypoint trajectories. By combining history-conditioned adaptation with online distillation, HoRD enables a single policy to adapt zero-shot to unseen domains without per-domain retraining. Extensive experiments show HoRD outperforms strong baselines in robustness and transfer, especially under unseen domains and external perturbations. Code and project page are available at \href{this https URL}{this https URL}.

**arXiv ID:** 2602.04412
</details>

<details>
<summary><strong>Comparative Analysis of Autonomous Robotic and Manual Techniques for Ultrasonic Sacral Osteotomy: A Preliminary Study</strong> - Daniyal Maroufi, Yash Kulkarni, Justin E. Bird, Jeffrey H. Siewerdsen, Farshid Alambeigi - [[pdf]](https://arxiv.org/pdf/2602.04076)</summary>

**Abstract:** In this paper, we introduce an autonomous Ultrasonic Sacral Osteotomy (USO) robotic system that integrates an ultrasonic osteotome with a seven-degree-of-freedom (DoF) robotic manipulator guided by an optical tracking system. To assess multi-directional control along both the surface trajectory and cutting depth of this system, we conducted quantitative comparisons between manual USO (MUSO) and robotic USO (RUSO) in Sawbones phantoms under identical osteotomy conditions. The RUSO system achieved sub-millimeter trajectory accuracy (0.11 mm RMSE), an order of magnitude improvement over MUSO (1.10 mm RMSE). Moreover, MUSO trials showed substantial over-penetration (16.0 mm achieved vs. 8.0 mm target), whereas the RUSO system maintained precise depth control (8.1 mm). These results demonstrate that robotic procedures can effectively overcome the critical limitations of manual osteotomy, establishing a foundation for safer and more precise sacral resections.

**arXiv ID:** 2602.04076
</details>

<details>
<summary><strong>Shaping Expressiveness in Robotics: The Role of Design Tools in Crafting Embodied Robot Movements</strong> - Elisabetta Zibetti, Alexandra Mercader, Hélène Duval, Florent Levillain, Audrey Rochette, David St-Onge - [[pdf]](https://arxiv.org/pdf/2602.04137)</summary>

**Abstract:** As robots increasingly become part of shared human spaces, their movements must transcend basic functionality by incorporating expressive qualities to enhance engagement and communication. This paper introduces a movement-centered design pedagogy designed to support engineers in creating expressive robotic arm movements. Through a hands-on interactive workshop informed by interdisciplinary methodologies, participants explored various creative possibilities, generating valuable insights into expressive motion design. The iterative approach proposed integrates analytical frameworks from dance, enabling designers to examine motion through dynamic and embodied dimensions. A custom manual remote controller facilitates interactive, real-time manipulation of the robotic arm, while dedicated animation software supports visualization, detailed motion sequencing, and precise parameter control. Qualitative analysis of this interactive design process reveals that the proposed "toolbox" effectively bridges the gap between human intent and robotic expressiveness resulting in more intuitive and engaging expressive robotic arm movements.

**arXiv ID:** 2602.04137
</details>

<details>
<summary><strong>ALORE: Autonomous Large-Object Rearrangement with a Legged Manipulator</strong> - Zhihai Bi, Yushan Zhang, Kai Chen, Guoyang Zhao, Yulin Li, Jun Ma - [[pdf]](https://arxiv.org/pdf/2602.04214)</summary>

**Abstract:** Endowing robots with the ability to rearrange various large and heavy objects, such as furniture, can substantially alleviate human workload. However, this task is extremely challenging due to the need to interact with diverse objects and efficiently rearrange multiple objects in complex environments while ensuring collision-free loco-manipulation. In this work, we present ALORE, an autonomous large-object rearrangement system for a legged manipulator that can rearrange various large objects across diverse scenarios. The proposed system is characterized by three main features: (i) a hierarchical reinforcement learning training pipeline for multi-object environment learning, where a high-level object velocity controller is trained on top of a low-level whole-body controller to achieve efficient and stable joint learning across multiple objects; (ii) two key modules, a unified interaction configuration representation and an object velocity estimator, that allow a single policy to regulate planar velocity of diverse objects accurately; and (iii) a task-and-motion planning framework that jointly optimizes object visitation order and object-to-target assignment, improving task efficiency while enabling online replanning. Comparisons against strong baselines show consistent superiority in policy generalization, object-velocity tracking accuracy, and multi-object rearrangement efficiency. Key modules are systematically evaluated, and extensive simulations and real-world experiments are conducted to validate the robustness and effectiveness of the entire system, which successfully completes 8 continuous loops to rearrange 32 chairs over nearly 40 minutes without a single failure, and executes long-distance autonomous rearrangement over an approximately 40 m route. The open-source packages are available at this https URL.

**arXiv ID:** 2602.04214
</details>

<details>
<summary><strong>Autonomous Navigation at the Nano-Scale: Algorithms, Architectures, and Constraints</strong> - Mahmud S. Zango, Jianglin Lan - [[pdf]](https://arxiv.org/pdf/2601.13252)</summary>

**Abstract:** Autonomous navigation for nano-scale unmanned aerial vehicles (nano-UAVs) is governed by extreme Size, Weight, and Power (SWaP) constraints (with the weight < 50 g and sub-100 mW onboard processor), distinguishing it fundamentally from standard robotic paradigms. This review synthesizes the state-of-the-art in sensing, computing, and control architectures designed specifically for these sub- 100mW computational envelopes. We critically analyse the transition from classical geometry-based methods to emerging "Edge AI" paradigms, including quantized deep neural networks deployed on ultra-low-power System-on-Chips (SoCs) and neuromorphic event-based control. Beyond algorithms, we evaluate the hardware-software co-design requisite for autonomy, covering advancements in dense optical flow, optimized Simultaneous Localization and Mapping (SLAM), and learning-based flight control. While significant progress has been observed in visual navigation and relative pose estimation, our analysis reveals persistent gaps in long-term endurance, robust obstacle avoidance in dynamic environments, and the "Sim-to-Real" transfer of reinforcement learning policies. This survey provides a roadmap for bridging these gaps, advocating for hybrid architectures that fuse lightweight classical control with data-driven perception to enable fully autonomous, agile nano-UAVs in GPS-denied environments.

**arXiv ID:** 2601.13252
</details>

<details>
<summary><strong>Patterns for a New Generation: AI and Agents</strong> - Joseph Corneli, Charles J. Danoff, Raymond S. Puzio, Sridevi Ayloo, Sergio Belich, Andre Wilkinson, Mary Tedeschi, Pauline Mosley - [[pdf]](https://arxiv.org/pdf/2506.09696)</summary>

**Abstract:** Design patterns have been used in various fields of inquiry and endeavour to externalize procedural knowledge in a form that supports human reasoning and coordination. In this paper, we show that contemporary Large Language Model (LLM)-based systems can also read, generate, and reason with design patterns written in a structured template. We describe an experimental workflow in which patterns function as shared priors for action selection, reflection, and revision in hybrid human/agent settings. Drawing on the Active Inference Framework, we illustrate how patterns can guide agent behavior without fully prescribing it. This provides a proof of concept that pattern-capable agents can be created using now-standard software tools. We discuss implications for software development, education, business, and AI governance.

**arXiv ID:** 2506.09696
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
