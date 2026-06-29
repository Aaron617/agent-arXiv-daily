# Agent arXiv Daily

**Last Updated:** 2026-06-29 06:05:01

**Total Papers:** 69

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
<summary><strong>ATOD: Annealed Turn-aware On-policy Distillation for Multi-turn Autonomous Agents</strong> - Qitai Tan, Zefang Zong, Yang Li, Peng Chen - [[pdf]](https://arxiv.org/pdf/2606.27814)</summary>

**Abstract:** Training small language-model agents for long-horizon interactive tasks requires both fast imitation and reward-driven improvement. On-policy distillation (OPD) provides dense teacher guidance and typically improves rapidly in the early stage, but its gains saturate once the student approaches the teacher, limiting the final performance ceiling. Reinforcement learning (RL) directly optimizes environment rewards and encourages exploratory improvement toward a higher reward-defined ceiling, but sparse and delayed feedback makes early-stage learning much less efficient than OPD. In this paper, we propose ATOD (Annealed Turn-aware On-policy Distillation), a hybrid online distillation algorithm that explicitly exploits this complementarity. (1) ATOD uses an annealed OPD-RL schedule: OPD dominates early training to approach teacher-level behavior, while RL is gradually strengthened to drive reward-based exploration. (2) ATOD introduces Turn-level Disagreement-Uncertainty Reweighting (T-DUR), which softly amplifies high-utility turns and improves dense supervision in long trajectories. Experiments on ALFWorld, WebShop, and Search-QA show that ATOD consistently outperforms competing post-training baselines: across the three student sizes, ATOD improves average success rate by 3.03 points over OPD and 23.62 points over GRPO, while surpassing the corresponding teacher models by 2.16 points.

**arXiv ID:** 2606.27814
</details>

<details>
<summary><strong>$τ$-Rec: A Verifiable Benchmark for Agentic Recommender Systems</strong> - Bharath Sivaram Narasimhan, Karthik R Narasimhan - [[pdf]](https://arxiv.org/pdf/2606.10156)</summary>

**Abstract:** As recommender systems transition toward agentic, multi-turn conversational interfaces, evaluation paradigms have struggled to keep pace. Current benchmarks often rely on "LLM-as-a-judge" evaluations, which introduce subjectivity, high costs and inconsistency. We present $\tau$-Rec, a benchmark for agentic recommender systems that replaces subjective evaluation with verifiable rewards and a reveal-tagged elicitation (RTE) mechanism that controls how task constraints surface during dialogue. By testing agents against structured catalog predicates and employing a pass^k reliability metric, $\tau$-Rec provides a systematic test for consistent reasoning. Our evaluation of nine configurations across five model families -- GPT-5.4, Claude Sonnet 4.6, Gemini 2.5 Flash, DeepSeek V4 Flash, Qwen3-32B and GPT-5 mini -- reveals a steep reliability cliff, where even the best model achieves only ~57% at pass^1 and ~35% at pass^4, highlighting a critical gap in current conversational agent deployment. All code and data are publicly available at this https URL.

**arXiv ID:** 2606.10156
</details>

<details>
<summary><strong>Training Observable Control Policies to Expose Agent State Through Actions</strong> - Andres Enriquez Fernandez, John J. Bird - [[pdf]](https://arxiv.org/pdf/2606.27609)</summary>

**Abstract:** Physical or operational constraints often impose communications limitations on autonomous agents. Such limitations complicate monitoring or multiagent coordination. Even when strong communications are absent, some information may still be available. The remainder of the relevant agent state may be reconstructed via estimation. The actions taken by an agent are a potential source of information -- as the agent interacts with the environment, these actions may be observed even in the absence of explicit communication. We investigate using actions to estimate the state of an agent, using reinforcement learning to develop policies which make the estimation problem more tractable. Policy observability is encouraged through the training reward and is analyzed using simulation of the trained agent. In an aircraft tracking problem a policy with enhanced observability is found that has minimal impact on nominal task performance.

**arXiv ID:** 2606.27609
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (15 papers)</h2></summary>

<details>
<summary><strong>NormAct: A Benchmark for Hidden Social Norm Compliance in Embodied Planning</strong> - Shiyun Zhao, Xinwei Song, Tianyu Guo, Xiaomeng Gao, Mingyuan Liu, Xu Han, Yuanyuan Zhang, Zhenliang Zhang, Xue Feng, Bo Dai - [[pdf]](https://arxiv.org/pdf/2606.27826)</summary>

**Abstract:** Multimodal large language models (MLLMs) are increasingly deployed as embodied planners in egocentric environments, where task success requires not only achieving instructed goals but also acting in socially appropriate ways. While explicit goals may render certain actions optimal, implicit social norms often impose hidden constraints. Existing evaluations typically focus on explicit goal achievement or direct norm knowledge, seldom assessing whether planners can infer and apply these hidden constraints within action sequences. We introduce NormAct, a benchmark for embodied social-norm interactions that evaluates plans on Goal Achievement, Norm Compliance, and overall Task Success. NormAct uniquely embeds hidden norms within ordinary tasks, testing whether models can realize them without explicit instruction. Experiments with state-of-the-art MLLMs (GPT-5.4, Claude Opus 4.7, Gemini 3 Pro) reveal a significant gap: models achieve explicit goals in 67.3\% of cases, but comply with hidden norms in only 26.4\%. Cue-condition experiments indicate that this gap stems not from a lack of general social knowledge, but from challenges in activating and grounding relevant norms in context. To address this, we propose NormPerceptor, a context-conditioned cue generator that infers scene-relevant norms prior to planning, increasing Task Success from 24.2\% to 46.7\%. Our results underscore the importance of enabling embodied agents to proactively detect hidden norms, ground them in visual evidence, and integrate them as action-planning constraints. Our benchmark is publicly available at this https URL.

**arXiv ID:** 2606.27826
</details>

<details>
<summary><strong>DMV-Bench: Diagnosing Long-Horizon Multimodal Agents' Visual Memory with Incidental Cue Injection</strong> - Yujin Tang, Chenming Shang, Ruize Xu, Nikhil Singh - [[pdf]](https://arxiv.org/pdf/2606.27499)</summary>

**Abstract:** Research on agent memory has matured rapidly, but almost entirely on the text side: few existing benchmarks ask, in an interactive environment, when an agent genuinely needs to remember what it saw rather than what it could write down. We introduce DMV-Bench (Code: this https URL), the first interactive benchmark for multimodal-agent visual memory. DMV-Bench is built on a controlled home-furnishing e-commerce catalogue of 1,000 product variants in which a text-leakage contract keeps the discriminative signal of each task in the pixels alone. Across a chain of autonomous shopping sessions, every visited product image carries a unique, pre-rendered incidental cue, and the agent is later asked to recall a particular cued product and navigate to its URL. Inspired by dual-coding theory, we propose DualMem, a memory architecture that maintains a visual and a verbal code in parallel. On DMV-Bench, DualMem outperforms a caption baseline and three recent multimodal agent-memory systems at every chain length J in {5, 10, 15, 50} on both Gemini 2.5 Flash and Qwen2.5-VL-7B, with the lead surviving controls for memory-bank size and encoding-position bias, and an asymmetric dual-coding regime in which vision carries the cue end-to-end while the verbal channel plays a smaller query-grounding role.

**arXiv ID:** 2606.27499
</details>

<details>
<summary><strong>HAT-4D: Lifting Monocular Video for 4D Multi-Object Interactions via Human-Agent Collaboration</strong> - Jiaxin Li, Yuxiang Wu, Zhenkai Zhang, Xinrui Shi, Haoyuan Wang, Yichen Zhao, Su Linxiang, Chenyang Yu, Mingyu Zhang, Yifan Ding, Boran Wen, Li Zhang, Ruiyang Liu, Yong-Lu Li - [[pdf]](https://arxiv.org/pdf/2606.28215)</summary>

**Abstract:** Extracting dynamic 4D object interactions from massive, in-the-wild monocular videos offers a highly efficient data collection pathway for scaling Embodied AI and training VLAs. However, existing monocular 4D reconstruction methods primarily focus on isolated objects, often failing under the severe occlusions and complex dynamics inherent in multi-object interactions. To bridge this gap, we propose HAT-4D, the first agentic framework designed to reconstruct the 3D geometry, temporal dynamics, and physical interactions of multiple objects from a single video. By integrating VLMs with a multi-level human-in-the-loop feedback mechanism, HAT-4D efficiently resolves depth ambiguities and interaction-induced occlusions during 3D generation and 4D propagation, yielding physically plausible assets without relying on expensive multicamera rigs. As a scalable data engine, HAT-4D facilitates the creation of MVOIK-4D, an open-world benchmark for monocular 4D interaction reconstruction, accompanied by a novel multi-dimensional evaluation protocol focused on physical plausibility and temporal consistency. Extensive experiments demonstrate that HAT-4D achieves SOTA performance on most evaluation metrics, while maintaining competitive semantic alignment. Ablation studies show that introducing a small amount of human feedback improves interaction reconstruction. Moreover, the data produced by HAT-4D effectively improves baseline performance when used for fine-tuning. Our data and code are available at this https URL

**arXiv ID:** 2606.28215
</details>

<details>
<summary><strong>Govern the Repository, Not the Agent: Measuring Ecosystem-Level Risk in AI-Native Software</strong> - Daniel Russo - [[pdf]](https://arxiv.org/pdf/2606.28235)</summary>

**Abstract:** Autonomous coding agents now open and merge pull requests in shared repositories at scale, and the field evaluates them the way it has always evaluated components, one agent at a time, on isolated benchmark tasks. Yet agents that each pass their own tests still leave repositories that accumulate problems no single contribution accounts for. We ask whether this problem belongs to the individual agent or to the repository where it accumulates. We study integration friction, the cost of integrating a contribution into a codebase that other contributors are concurrently changing. Across more than 930,000 agent-authored pull requests, we measure how much of the variation in friction stays with the repository after the contribution, its author, its size, and its agent are accounted for. About half does, and it survives full controls. In the same repositories, agent-authored contributions concentrate this repository-level friction roughly twice as much as human ones (intraclass correlation 0.30 versus 0.16), a gap that holds after controlling for codebase size, age, task shape, process maturity, and merge path. The risk is a property of the ecosystem, not the agent. AI-native software is therefore better measured and governed at the ecosystem level than one agent at a time.

**arXiv ID:** 2606.28235
</details>

<details>
<summary><strong>Agentic Hardware Design as Repository-Level Code Evolution</strong> - Cunxi Yu, Chenhui Deng, Nathaniel Pinckney, Brucek Khailany - [[pdf]](https://arxiv.org/pdf/2606.28279)</summary>

**Abstract:** We present HORIZON, a self-evolving agent framework that treats hardware design as repository-level code evolution. A Markdown harness is compiled into a project pack containing domain knowledge, an executable evaluator, an acceptance predicate, and a git/runtime policy; a hands-free agent loop then evolves an isolated git worktree, using repository operations for state management, tracing, and replay. This extends prior works of repository-scale self-evolution from EDA software systems, to hardware-design artifacts themselves. We evaluate our approach on ChipBench, RTLLM, Verilog-Eval, and nine CVDP categories, achieving 100\% benchmark completion across all suites with a fully hands-free agentic loop. However, we do not claim that agentic AI for hardware design is solved: these benchmarks are controlled proxies for a much broader engineering problem in chip design. Section~\ref{sec:discuss} examines the limitations of the current study and highlights open research challenges.

**arXiv ID:** 2606.28279
</details>

<details>
<summary><strong>Just Ask: Curious Code Agents Reveal System Prompts in Frontier LLMs</strong> - Xiang Zheng, Yutao Wu, Hanxun Huang, Yige Li, Xingjun Ma, Bo Li, Yu-Gang Jiang, Cong Wang - [[pdf]](https://arxiv.org/pdf/2601.21233)</summary>

**Abstract:** Autonomous code agents built on large language models are reshaping software and AI development through tool use, long-horizon reasoning, and self-directed interaction. However, this autonomy introduces a previously unrecognized security risk: agentic interaction fundamentally expands the LLM attack surface, enabling systematic probing and recovery of hidden system prompts that guide model behavior. We identify system prompt extraction as an emergent vulnerability intrinsic to code agents and present \textbf{\textsc{JustAsk}}, a self-evolving framework that autonomously discovers effective extraction strategies through interaction alone. Unlike prior prompt-engineering or dataset-based attacks, \textsc{JustAsk} requires no handcrafted prompts, labeled supervision, or privileged access beyond standard user interaction. It formulates extraction as an online exploration problem, using Upper Confidence Bound-based strategy selection and a hierarchical skill space spanning atomic probes and high-level orchestration. These skills exploit imperfect system-instruction generalization and inherent tensions between helpfulness and safety. Evaluated on \textbf{41} black-box commercial models across multiple providers, \textsc{JustAsk} consistently achieves full or near-complete system prompt recovery, revealing recurring design- and architecture-level vulnerabilities. Our results expose system prompts as a critical yet largely unprotected attack surface in modern agent systems.

**arXiv ID:** 2601.21233
</details>

<details>
<summary><strong>SEA-TS: Self-Evolving Agent for Autonomous Code Generation of Time Series Forecasting Algorithms</strong> - Longkun Xu, Xiaochun Zhang, Qiantu Tuo, Rui Li - [[pdf]](https://arxiv.org/pdf/2603.04873)</summary>

**Abstract:** Accurate time series forecasting underpins decision-making in many domains, yetconventional ML development often faces data scarcity, distribution shift, anddiminishing returns from manual iteration. We propose Self-Evolving Agent forTime Series Algorithms (SEATS), a framework that autonomously generates, val-idates, and optimizes forecasting algorithm code through an iterative self-evolutionloop. Our design combines three mechanisms: (1) Metric-Advantage MCTS(MA-MCTS), which replaces fixed rewards with a statistically normalized advan-tage score for search guidance, (2) code review with running prompt refinement,so every successfully executed solution is reviewed and the running prompt encodescorrective patterns for later iterations, and (3) global steerable reasoning, whichcompares each evaluated node to global best- and worst-performing solutions forcross-trajectory transfer. A MAP-Elites archive maintains architectural this http URL four datasets and two metrics, SEATS wins seven of eight comparisonsagainst strong baselines TimeMixer, Timer, and SEMixer

**arXiv ID:** 2603.04873
</details>

<details>
<summary><strong>Your AI Travel Agent Would Book You a Bullfight: An Agentic Benchmark for Implicit Animal Welfare in Frontier AI Models</strong> - Jasmine Brazilek, Joel Christoph, Oliver Tullio, Carol Kline, Miles Tidmarsh, Arturs Kanepajs - [[pdf]](https://arxiv.org/pdf/2606.18142)</summary>

**Abstract:** AI agents are moving from advisors to actors, booking travel, planning menus, and running procurement on behalf of users. Existing benchmarks for AI and animal welfare evaluate model text responses to question-answer prompts, leaving open whether the welfare reasoning surfaced in those responses transfers to agentic deployment where the model must take actions with tools. We introduce TAC (Travel Agent Compassion), the first agentic benchmark measuring whether AI agents avoid options involving animal exploitation when acting on behalf of users. TAC presents an AI agent with twelve hand-authored travel booking scenarios across six categories of animal exploitation, augmented to forty-eight samples to control for price, rating, and position confounds. We evaluate seven frontier models from four labs. Every model scores below the chance level of sixty-four percent, with the best performer (Claude Opus 4.7) at fifty-three percent. A single welfare-aware sentence in the system prompt yields gains of forty-seven to sixty-three percentage points in Claude and GPT-5.5, twenty-six points in GPT-5.2, and under twelve points in DeepSeek and Gemini. An auxiliary Inspect Scout audit of 288 base-condition transcripts from the top two performers, using Gemini 2.5 Flash Lite as judge, flags zero transcripts for evaluation awareness, suggesting the below-chance rates do not stem from the models recognising the evaluation. We discuss implications for category-level variation across cultural domains, the limits of text-response welfare benchmarks, and the EU General-Purpose AI Code of Practice systemic risk framework.

**arXiv ID:** 2606.18142
</details>

<details>
<summary><strong>Agent-as-a-Router: Agentic Model Routing for Coding Tasks</strong> - Pengfei Zhou, Zhiwei Tang, Yixing Ma, Jiasheng Tang, Yizeng Han, Zhenglin Wan, Fanqing Meng, Wei Wang, Bohan Zhuang, Wangbo Zhao, Yang You - [[pdf]](https://arxiv.org/pdf/2606.22902)</summary>

**Abstract:** Real-world users typically have access to multiple Large Language Models (LLMs) from different providers, and these LLMs often excel at distinct domains, yet none dominate all. Consequently, routing each task to the most suitable model becomes critical for both performance and cost. Existing routers treat this as a static, one-off classification problem. However, we identify the performance bottleneck for these routers as information deficit: simply augmenting a vanilla LLM router with performance statistics at the task-dimension level yields a 15.3% relative gain, surpassing a heuristic router built on the same dimension-level priors. Motivated by this finding, we propose Agent-as-a-Router, a framework that formalizes routing as a C-A-F loop (Context->Action->Feedback->Context). It closes the information gap by accumulating execution-grounded experience during deployment. We instantiate this framework as ACRouter, composed of an Orchestrator, a Verifier, a Memory module, and introduce CodeRouterBench, an evaluation environment comprising ~10K task instances with verified scores from 8 frontier LLMs, enabling regret-based router comparison on streaming tasks. Experiments show that ACRouter achieves the lowest cumulative regret on in-distribution tasks and generalizes to out-of-distribution agentic-programming tasks, demonstrating that our routing framework actively closes the information gap. Codes and benchmarks are released at this https URL.

**arXiv ID:** 2606.22902
</details>

<details>
<summary><strong>AI Snitches Get Glitches: Towards Evading Agentic Surveillance</strong> - Hyejun Jeong, Dzung Pham, Amir Houmansadr, Eugene Bagdasarian - [[pdf]](https://arxiv.org/pdf/2606.25836)</summary>

**Abstract:** To better assist users with completing challenging tasks, AI agents mediate communications, access data, and interact with different APIs. Many employers (and even nation-states) already provide their users with this technology. However, widespread adoption of AI agents creates a new risk to abuse access to user data for another goal: surveilling users. These users might not even have the ability or permission to control the actions and data accesses of the surveilling agents.
We introduce and formalize the problem of agentic surveillance: the ability of an AI agent to analyze available information, craft a report, and send it out using available tools. To evaluate surveillance capabilities across different models, we create SurveilBench, a dataset of various reporting scenarios focusing on three domains: corporate, education, and police. We find that some models exhibit emergent (i.e., unprompted) tendencies to help surveillance, but they also report the attempts to surveil users to the government.
Finally, we repurpose prompt injections for evading surveillance and develop three evasion techniques that hide from, deceive, or induce over-escalation in surveillance agents. We conclude that agentic surveillance can already be easily implemented and, therefore, call for a comprehensive technical, ethical, and legislative framework to protect users.

**arXiv ID:** 2606.25836
</details>

<details>
<summary><strong>Adaptive Turn-Taking for Real-time Multi-Party Voice Agents</strong> - Soumyajit Mitra, Prabhat Pandey, Abhinav Jain, Shanmukha Sahith, K V Vijay Girish - [[pdf]](https://arxiv.org/pdf/2606.13544)</summary>

**Abstract:** Turn-taking in multi-party spoken conversations remains a fundamental challenge for voice-based agents, particularly under dynamic floor competition and varying user expectations. We propose ModeratorLM, a role-playing voice agent that conditions turn-taking behavior on an explicitly assigned role in multi-party settings. The system is built on a speech large language model operating in chunk-wise streaming manner. We further introduce a reasoning-augmented variant that incorporates chain-of-thought reasoning over conversational context and the assigned role. We construct RolePlayConv, a large-scale synthetic dataset of spoken multi-party conversations with diverse assistant roles. Experiments on real-world meeting data and RolePlayConv show improved turn-taking precision by over 40% and recall by more than 70%, while substantially reducing false-positive interruptions compared to non-role-conditioned baselines.

**arXiv ID:** 2606.13544
</details>

<details>
<summary><strong>Ko-WideSearch: A Korean Breadth-Search Benchmark for Exhaustive Set Enumeration by Web Agents</strong> - Minbyul Jeong - [[pdf]](https://arxiv.org/pdf/2606.27595)</summary>

**Abstract:** Web-agent benchmarks overwhelmingly measure depth -- pinning one obscure answer behind a chain of constraints -- while breadth, exhaustively enumerating a closed set and filling each item's attributes, is barely evaluated, especially outside English. Breadth is also hard to build: certifying that a gold set is complete and every cell correct is far costlier than checking a single answer. I introduce \textsc{Ko-WideSearch}, a Korean breadth-search benchmark built by an automated synthesize-and-verify pipeline. Each task names a set-parent entity -- a TV season, a dynasty, a league, an administrative region, an election -- and asks for its full membership plus a per-item attribute table, graded by Item-, Column-, and Row-F1. It spans 228 tables over 190 entities and sixteen categories across three difficulty tiers, set by two structural knobs I dial independently -- table width and a 2-D composite key -- so cross-product membership climbs from 0\% to 100\% across the tiers. A single normalization-aware comparator is shared between gold construction and grading, so stable date and count columns are not over-dropped on formatting alone. Across twenty web agents, the failure is consistent: agents recover the set but not the rows (e.g.\ Item-F1 92.8 against Row-F1 53.7), accuracy falls steadily as the knobs harden, and neither more search nor more spend closes the gap. Broken down by cell, the hard part is finding the right value, not formatting it: open-ended free-text cells fail most, while cells with a standard answer such as a date or a name usually come out right.

**arXiv ID:** 2606.27595
</details>

<details>
<summary><strong>When Search Agents Should Ask: DiscoBench for Clarification-Aware Deep Search</strong> - Yiling Tao, Shihan Deng, Meiling Tao, Pengzhi Wei, Zhichao Hu, Zhihao Zhu - [[pdf]](https://arxiv.org/pdf/2606.27669)</summary>

**Abstract:** Search agents powered by large language models (LLMs) are increasingly used to solve complex information-seeking tasks, requiring multi-step retrieval and reasoning to fulfill user goals. However, existing benchmarks often assume that user queries are complete and explicit, overlooking the fact that real-world search requests are frequently vague, underspecified, or even factually incorrect. In deep search scenarios, such ambiguity can propagate along multi-step reasoning chains and lead agents toward incorrect search trajectories. To address this gap, we introduce DiscoBench, a benchmark for clarification-aware deep search, designed to evaluate whether search agents can proactively identify ambiguity, ask effective clarification questions, and recover correct reasoning paths through user interaction. DiscoBench contains 211 samples and 463 ambiguity instances across 11 real-world domains, covering four ambiguity types. We further design a user simulator for multi-turn interaction and evaluate model performance from four perspectives: task utility, ambiguity detection, interaction strategy, and cost efficiency. Experiments on representative LLMs show that ambiguity detection and effective clarification are distinct capabilities, and that repeatedly searching instead of asking for clarification often performs worse than direct guessing, highlighting a critical gap between retrieval ability and interactive problem-solving in current search agents.

**arXiv ID:** 2606.27669
</details>

<details>
<summary><strong>Multimodal Evaluator Preference Collapse: Cross-Modal Coupling in Self-Evolving Agents</strong> - Zewen Liu - [[pdf]](https://arxiv.org/pdf/2606.16682)</summary>

**Abstract:** When AI agents use language models to evaluate their own outputs in a feedback loop, systematic biases emerge. We show that Evaluator Preference Collapse (EPC) is dramatically amplified in multimodal settings. Using GPT-4o to evaluate DeepSeek-chat across text and visual tasks, we find that a single strategy (step_by_step) absorbs 48.4% of all weight -- 3.2x the collapse observed in text-only self-evaluation -- while three visual-domain strategies receive only 9.1% combined weight. We then demonstrate a novel phenomenon we term cross-modal coupling: evaluator preferences acquired on one modality transfer to and corrupt strategy selection on another. Through a four-phase isolation training paradigm, we measure coupling coefficients and document strategy inversion -- the optimal strategy for a modality reverses after cross-modal exposure. A Phase 3 statistical validation across five evaluator configurations (N=80 total independent repetitions, ~35,000 API calls) with both text-proxy and real-image visual tasks finds: cross-model evaluation produces strong coupling (JSD~0.19-0.34), real-image inputs yield the most directionally consistent signal (mean gamma_{T->V}=1.145, gamma_{V->T}=0.937, 70% T->V, Cohen's d=0.56), and self-evaluation provides near-complete immunity -- 97% of runs (N=30) yield zero coupling (JSD=0.003, d=0.07). Three methodological ablations and multi-executor validation confirm the effect is not a structural artifact. We introduce the coupling matrix indexed by evaluator identity, release the MM-EPC framework, and identify cross-model evaluator architecture as the primary risk factor for preference drift. Code and data: this https URL.

**arXiv ID:** 2606.16682
</details>

<details>
<summary><strong>Spacecraft Fiducial Marker for Autonomous Rendezvous, Proximity Operations, and Docking</strong> - Ravi Kumar Thakur, Matouš Vrba, Martin Saska - [[pdf]](https://arxiv.org/pdf/2606.27566)</summary>

**Abstract:** Robotic operations in space are challenging due to the harsh environment and the high cost of failure. Fiducial markers provide visual references that aid autonomous rendezvous, proximity operations, and docking for space robots. However, existing fiducial markers are mostly single-scale and largely designed for terrestrial robotics. Such markers leave the camera's field of view at close range, precisely during the proximity and docking phases where reliable tracking is most critical. This paper presents AstraTag, a fiducial marker designed for autonomous on-orbit robotic operations. The marker template is based on a square Spidron pattern whose recursive, self-similar structure enables detection across multiple spatial scales. Marker identification uses a 48-bit signature derived from triangular sub-regions of the template and encoded with a Generalised Reed-Solomon (GRS) code. The detection pipeline performs contour-based quadrilateral localisation, perspective normalisation, and signature matching against a pre-computed dictionary. To handle markers affixed to curved spacecraft surfaces, it incorporates a Thin-Plate Spline (TPS) re-warp fallback that exploits the marker's internal rectangular borders as additional geometric correspondences. We benchmark AstraTag against three-layer Fractal ArUco and AprilTag on spacecraft mockups with flat and curved surfaces. On curved surfaces, AstraTag achieves a higher detection rate than both baselines, offering a robust recursive-marker option for space robotics.

**arXiv ID:** 2606.27566
</details>

</details>

<details open>
<summary><h2>LLM Agents (10 papers)</h2></summary>

<details>
<summary><strong>Grounded Iterative Language Planning: How Parameterized World Models Reduce Hallucination Propagation in LLM Agents</strong> - Xinyuan Song, Zekun Cai - [[pdf]](https://arxiv.org/pdf/2606.27806)</summary>

**Abstract:** World models for language agents come in two useful forms. An agent-based world model calls an LLM API and reasons flexibly in language, but its errors appear as hallucinated state changes that are hard to score with ordinary regression losses. A parameterized world model is a trained transition predictor; its errors are easier to measure with quantities such as NodeMSE, delta accuracy, and validity accuracy, but it is usually weaker as a standalone planner. We compare these two families on four graph-structured planning benchmarks and introduce operational hallucination metrics for the agent-based case. The comparison motivates \textbf{Grounded Iterative Language Planning} (GILP), which trains only a small parameterized backbone and combines it with API-based agent reasoning. The backbone supplies valid actions, predicted state deltas, risk, and value; the LLM drafts an action and imagined delta; and a consistency gate asks for revision when the two disagree. On real GPT-4o-mini calls, GILP reduces hallucinated-state rate from 0.176 to 0.035. In calibrated simulator ablations, it raises success from 0.668 to 0.838 while adding only ~22% extra LLM calls.

**arXiv ID:** 2606.27806
</details>

<details>
<summary><strong>Internalizing the Future: A Unified Agentic Training Paradigm for World Model Planning</strong> - Xuan Zhang, Zhijian Zhou, Lingfeng Qiao, Yulei Qin, Ke Li, Xing Sun, Xiaoyu Tan, Chao Qu, Yuan Qi - [[pdf]](https://arxiv.org/pdf/2606.27483)</summary>

**Abstract:** Large language model (LLM) agents have demonstrated strong capability in sequential decision-making, yet they remains fundamentally reactive in long-horizon tasks. Unlike humans who employ "what-if" reasoning to evaluate potential plans before commitment, standard agents lack an internal world model to simulate future outcomes. Therefore, we propose to internalize future-aware planning by training a single autoregressive model to verbalize both a prospective state rollout and a plan-conditioned success estimate-a textual analogue of the Q-value. Crucially, we identify a format-capability gap: simply fine-tuning agents on look-ahead traces during post-training leads to superficial mimicry of foresight without genuine predictive grounding. To bridge this gap, we introduce a three-stage training paradigm: (i) World Model Agentic Mid-Training (WM-AMT) to inject latent predictive capabilities into the policy; (ii) Format-Eliciting SFT (FE-SFT) to structure this injected capability; and (iii) Foresight-Conditioned Reinforcement Learning (FC-RL) to refine the calibration and utility of the generated simulations. Evaluated on search and mathematical reasoning tasks, our approach consistently outperforms other training baselines. Our results demonstrate that effective internal world modeling in LLM agents requires a capability-first training pipeline to achieve grounded and calibrated foresight.

**arXiv ID:** 2606.27483
</details>

<details>
<summary><strong>Agentic Publication Protocol: An Attempt to Modernize Scientific Publication</strong> - Sirui Lu, Xiao-Liang Qi - [[pdf]](https://arxiv.org/pdf/2606.27386)</summary>

**Abstract:** Scientific publication is still organized primarily around static manuscripts, even though much of scientific progress depends on tacit know-how: how to run code, reproduce figures, interpret edge cases, choose useful follow-up directions, and avoid failed paths. Large language model agents create an opportunity to publish not only knowledge, but also operational know-how in a form that future readers and researchers can directly use. This paper outlines the Agentic Publication Protocol (APP), a lightweight repository format for packaging a paper together with code, data, environment information, reproducibility instructions, and an agent-facing instruction file. APP treats a version-controlled repository as the publication object and uses \texttt{this http URL} and optional skills to define a paper agent that can explain the work, reproduce key results when possible, and support follow-up research. We describe the design principles and details of the protocol, as well as the agent skills useful for publishing papers under the protocol. We also describe development tools for evaluating and improving the protocol and associated agent skills. Finally, we provide a broader discussion of the future of scientific research in the agent era.

**arXiv ID:** 2606.27386
</details>

<details>
<summary><strong>SidConArena: An Environment Evaluating Agents in Open-Ended,Positive-Sum Bargaining Game</strong> - Yeqi Feng, Yuxin Chen, Tianxing He - [[pdf]](https://arxiv.org/pdf/2606.27397)</summary>

**Abstract:** Evaluating LLM agents requires dynamic environments that go beyond static reasoning and zero-sum games. Real-world economic interaction is often open-ended and mixed-motive: agents must negotiate, create positive-sum surplus, compete for scarce assets, and plan under delayed returns. We introduce SidConArena, a new benchmark framework for evaluating LLM agents in open-ended, positive-sum bargaining. SidConArena formalizes a multi-player economy as a finite-horizon partially observable stochastic game with three coupled phases: natural-language negotiation with binding trades, deterministic converter-based production, and sealed-bid auctions for long-term assets. The framework combines structured observations, phase-aware agent dispatching, a neural-symbolic action interface, and asynchronous execution, enabling free-form interaction while preserving rule-grounded evaluation. Across homogeneous and heterogeneous tournaments, stronger frontier models achieve higher economic outcomes, yet agents still misvalue resources, bargain passively, and remain limited in long-horizon investment planning.

**arXiv ID:** 2606.27397
</details>

<details>
<summary><strong>Supersede: Diagnosing and Training the Memory-Update Gap in LLM Agents</strong> - Vedant Patel - [[pdf]](https://arxiv.org/pdf/2606.27472)</summary>

**Abstract:** Large language model (LLM) agents operate over long, multi-session interactions in which facts change: a user moves, a price updates, a plan is revised. Acting correctly requires using the current value of a fact and discarding values that have been superseded. We isolate this ability on real conversational data and show that it is a distinct, unsolved failure. On the knowledge-update subset of LongMemEval, replacing an agent's full context with a bounded, self-maintained memory drops accuracy from 92% to 77% even on a frontier model (gpt-5.4), a gap that is statistically significant (paired McNemar p<0.005) and persists across model scale while full-context accuracy saturates near 92%. The bottleneck is therefore memory maintenance, not comprehension, and is not closed by a stronger model. We then ask whether this is merely an undersized memory, and find it is not: as the conversation grows 24x, accuracy falls further (from 68% to 28%), and granting the agent proportionally more memory yields no detectable recovery (28% to 28%, n=25). The failure scales with the length of the conversation, not the compression ratio. We release Supersede, an open reinforcement-learning environment (on the verifiers / prime-rl stack) that turns this measurement into a training signal: agents are rewarded for answering from the current value and penalized for stale ones. Finally, we close the loop and show the gap is trainable: GRPO fine-tuning a small open model (Qwen2.5-3B) on this environment nearly doubles its held-out supersession accuracy on real, unseen conversations (9.0% to 16.7%, a single run), along a monotonic checkpoint curve indicating the learned policy, not the harness, carries the gain. To our knowledge this is the first trainable environment whose reward targets temporal fact-currency, and the first evidence the supersession gap can be trained down, not only measured.

**arXiv ID:** 2606.27472
</details>

<details>
<summary><strong>Agentic AI-Powered Re-Identification: An Emerging, Scalable Threat to Mobility Microdata Privacy</strong> - Oscar Thees, Roman Müller, Matthias Templ - [[pdf]](https://arxiv.org/pdf/2606.27936)</summary>

**Abstract:** The widespread collection of fine-grained location data by commercial data brokers creates a re-identification risk that is not widely recognised by the public. While prior research has established that mobility traces are highly unique and that individuals can, in principle, be identified from a handful of spatio-temporal points, such attacks have historically required significant manual effort from skilled analysts, limiting their practical scale.
In this feasibility study, we demonstrate in a real world setting that agentic AI fundamentally changes this threat model. We present an end-to-end pipeline in which large language model agents autonomously search the open web, cross-reference public records and social media, and resolve raw coordinate sequences to candidate identities - without human intervention. We evaluate the pipeline on a spatio-temporal dataset containing simulated location points anchored at and around true home and work addresses, focusing on a high-risk disclosure scenario. Our results demonstrate that, from spatio-temporal data and public sources alone, our agentic AI successfully re-identified 18 of the 25 re-identifiable individuals (72%) and 18 of 43 cases overall (41.9%).
We discuss implications for Statistical Disclosure Control (SDC) practice and outline the near-future escalation that data custodians and regulators must anticipate. De facto anonymity - an implicit foundation of SDC practice - is shifting. Agentic AI strengthens the case that re-identification is reasonably likely by any means under the GDPR Recital-26 standard, at costs of minutes-and-dollars per target.

**arXiv ID:** 2606.27936
</details>

<details>
<summary><strong>ToolPrivacyBench: Benchmarking Purpose-Bound Privacy in Tool-Using LLM Agents</strong> - Shijing Hu, Liang Liu, Zhu Meng, Zhicheng Zhao - [[pdf]](https://arxiv.org/pdf/2606.28061)</summary>

**Abstract:** Large language models (LLMs) have increasingly moved from standalone text generation systems to agents that invoke external tools, access environments, and execute multi-step tasks. However, conventional function-calling benchmarks mainly evaluate task completion and API correctness, while privacy evaluation benchmarks typically focus on final responses or privacy judgments. Neither perspective captures purpose-bound information flow across an executed multi-tool trajectory. Motivated by this limitation in current agent evaluation, ToolPrivacyBench audits whether task-private atoms are routed only to authorized tools and downstream sinks, thereby evaluating both task completion and privacy over-disclosure during tool use. The benchmark contains 2,150 cases, including 1,150 fully synthetic privacy-sensitive business workflows and 1,000 cases adapted from existing multi-tool and function-calling benchmarks. Each case is represented by a policy knowledge base. After an agent executes against mock business backends, the evaluator compares recorded tool arguments and backend audit logs with this policy knowledge base. The evaluation covers nine widely used agents to characterize purpose-bound privacy over-disclosure. The results show that successful tool execution does not imply appropriate privacy disclosure: an agent may complete a task while transmitting unnecessary private information through intermediate tool calls. ToolPrivacyBench therefore formalizes a need-to-know disclosure boundary, under which each tool should receive only the information necessary for its stated purpose, and uses trajectory-level auditing to identify privacy over-disclosure in multi-tool workflows.

**arXiv ID:** 2606.28061
</details>

<details>
<summary><strong>LiveClawBench: Benchmarking LLM Agents on Complex, Real-World Assistant Tasks</strong> - Xiang Long, Li Du, Yilong Xu, RongJian Xu, Qiyanhui Lu, Ying Gao, Qinhua Xie, Fangcheng Liu, Ning Ding, Haoqing Wang, Ziheng Li, Changjiang Zhou, Jianyuan Guo, Yehui Tang - [[pdf]](https://arxiv.org/pdf/2604.13072)</summary>

**Abstract:** OpenClaw-style personal assistants extend LLM agents from isolated tool use to open-ended, stateful, and personalized software environments. Evaluating these assistants is fundamentally a fidelity problem: benchmarks must be faithful both to the distribution of real assistant tasks and to the execution semantics of the environments in which those tasks unfold. Existing benchmarks often lose fidelity in one dimension or the other. Their task distributions are shaped by what is easy to isolate, mock, and verify, underrepresenting real-world difficulties such as cross-service dependency, contaminated state, implicit intent, and runtime change. Their environments are either live but hard to reproduce, or reproducible but reduced to endpoint-level stubs that remove sessions, artifacts, state transitions, and downstream side effects. We introduce LiveClawBench, a benchmark designed around this dual-fidelity requirement. LiveClawBench combines a Triple-Axis Complexity Framework for difficulty-driven task construction with reproducible full-stack mock applications that preserve stateful execution semantics. With 134 executable cases across 10 domains with 22 mocked services, LiveClawBench supports controlled, extensible, and factor-level diagnostic evaluation of realistic agentic tasks. We release the benchmark resources: (1) Benchmark: this https URL (2) Leaderboard: this https URL (3) Trajectories: this https URL

**arXiv ID:** 2604.13072
</details>

<details>
<summary><strong>Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety</strong> - Vamshi Krishna Bonagiri, Ponnurangam Kumaragurum, Khanh Nguyen, Benjamin Plaut - [[pdf]](https://arxiv.org/pdf/2510.16492)</summary>

**Abstract:** As Large Language Model (LLM) agents increasingly operate in complex environments with real-world consequences, their safety becomes critical. While uncertainty quantification is well-studied for single-turn tasks, multi-turn agentic scenarios with real-world tool access present unique challenges where uncertainties and ambiguities compound, leading to severe or catastrophic risks beyond traditional text generation failures. We propose using "quitting" as a simple yet effective behavioral mechanism for LLM agents to recognize and withdraw from situations where they lack confidence. Leveraging the ToolEmu framework, we conduct a systematic evaluation of quitting behavior across 12 state-of-the-art LLMs. Our results demonstrate a highly favorable safety-helpfulness trade-off: agents prompted to quit with explicit instructions improve safety by an average of +0.39 on a 0-3 scale across all models (+0.64 for proprietary models), while maintaining a negligible average decrease of -0.03 in helpfulness. Our analysis demonstrates that simply adding explicit quit instructions proves to be a highly effective safety mechanism that can immediately be deployed in existing agent systems, and establishes quitting as an effective first-line defense mechanism for autonomous agents in high-stakes applications.

**arXiv ID:** 2510.16492
</details>

<details>
<summary><strong>COOPA: A Modular LLM Agent Architecture for Operations Research Problems</strong> - Chuanhao Li, Xiaoan Xu, Dirk Bergemann, Ethan X. Fang, Yehua Wei, Zhuoran Yang - [[pdf]](https://arxiv.org/pdf/2606.27611)</summary>

**Abstract:** Operations Research (OR) provides a rigorous framework for high-stakes decision-making, but effective OR modeling requires substantial domain knowledge, mathematical abstraction, and solver expertise. Recent LLM-based systems automate parts of this pipeline, yet remain limited by low accuracy on complex problems, opaque outputs, and narrow solver support. We propose COOPA (COoperative OPerations Agent), a modular LLM-agent architecture for interpretable and scalable OR decision support. It combines three components: iterative confidence-based modeling, which generates multiple candidate formulations, self-evaluates them across modeling dimensions, and selects one using a max-min confidence criterion; element-level provenance and confidence explanations, which link variables, parameters, constraints, and objectives to quoted source text and provide an audit trail for human verification; and multi-solver routing to specialized optimizer agents for different OR problem classes. Across three OR benchmarks, eight LLM backbones, and four baselines under identical conditions, COOPA achieves the best macro-average accuracy on six of eight backbones and improves over the strongest baseline by up to 6.7 percentage points. A within-system ablation isolates the contribution of iterative confidence-based modeling, while additional analyses and case studies illustrate the value of source traceability and multi-solver dispatch.

**arXiv ID:** 2606.27611
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (16 papers)</h2></summary>

<details>
<summary><strong>When Does Personality Composition Matter for Multi-Agent LLM Teams?</strong> - Aryan Keluskar, Amrita Bhattacharjee, Huan Liu - [[pdf]](https://arxiv.org/pdf/2606.27443)</summary>

**Abstract:** Personality prompting shapes how large language models communicate, yet whether these behavioral shifts affect objective task outcomes remains under-explored. Prior work shows that agents prompted with low agreeableness produce adversarial language, while those prompted with high agreeableness become cooperative, but the relationship between communication style and task performance has not been systematically examined across multiple domains. In this work, we investigate whether personality composition matters for multi-agent team performance by manipulating personality traits across frontier LLMs on three task domains: structured coding, open-ended research collaboration, and competitive bargaining. We find that personality effects depend critically on task structure. In coding tasks, low agreeableness leads to large communication shifts that have little effect on milestone completion. In open-ended collaboration and bargaining, the same manipulation substantially degrades performance. We discuss implications for multi-agent system design and the limits of personality manipulation.

**arXiv ID:** 2606.27443
</details>

<details>
<summary><strong>Agent-Native Immune System: Architecture, Taxonomy, and Engineering</strong> - Bo Shen, Lifeng Chang, Tianyuan Wei, Yunpeng Li, Feng Shi, Yichen Han, Peijie Gao, Shiyi Kuang, Xin Chang, Dehui Li - [[pdf]](https://arxiv.org/pdf/2606.28270)</summary>

**Abstract:** The transition from static chat bots to autonomous agents--equipped with persistent memory, tool-use protocols, and multi-agent collaboration--has fundamentally expanded the AI threat landscape. Current defense mechanisms, such as perimeter security and training-time alignment, remain external to the agent's active reasoning loop. Consequently, they fall short: a fully aligned agent remains highly vulnerable to runtime hijacking via memory poisoning, tool-chain manipulation, or multi-agent protocol attacks. To address this critical gap, we introduce the Agent-Native Immune System (ANIS), the first biologically inspired, endogenous defense architecture embedded directly within the agent's cognitive loop. Our framework presents four primary contributions. First, we design a six-layer Immune Tower (L0-L5), distinctly incorporating Barrier Immunity (L1) as a non-cognitive, physical-and-logical isolation layer. Second, we establish a unified taxonomy of Agent Viruses and Agent Vaccines, formalizing the critical distinction between superficial non-parametric defenses and robust parametric vaccines. Third, we conceptualize the Harness Triad--Meta, Self, and Auto--a self-monitoring, meta-cognitive automation backbone that drives Continual Immune Learning (CIL), enabling vaccines to dynamically adapt to novel threats. Finally, we establish a rigorous theoretical demarcation between model alignment and agent immunity: while alignment provides a static "constitutional" value foundation during training, ANIS serves as the dynamic "law enforcement" mechanism during runtime. We conclude by framing open challenges for the field, including immune protocol standardization, novel evaluation metrics such as the Autoimmunity Rate (false-positive intervention rate), and the co-evolutionary dynamics between pathogens and vaccines within collective intelligence ecosystems.

**arXiv ID:** 2606.28270
</details>

<details>
<summary><strong>LLawCo: Learning Laws of Cooperation for Modeling Embodied Multi-Agent Behavior</strong> - Qinhong Zhou, Chuang Gan, Anoop Cherian - [[pdf]](https://arxiv.org/pdf/2606.28182)</summary>

**Abstract:** Embodied agents operating in decentralized and partially observable environments have attracted growing attention in recent years. However, existing large language model (LLM)-based agents often exhibit behaviors that are misaligned with their partners or inconsistent with the environment state, leading to inefficient cooperation and poor task success. To address this challenge, we propose a novel framework, Learning Laws of Cooperation (LLawCo), that enables embodied agents to autonomously align with both their partners and task objectives. Our framework allows agents to reflect on past failures to extract misaligned behavioral patterns, which are used to derive high-level behavioral laws, such as "Talk when necessary" and "Wait for partner." These laws are explicitly incorporated into the agents' chains of thought via supervised fine-tuning, aligning their reasoning with task requirements and the behavior of other agents. To evaluate our approach, we introduce PARTNR-Dialog, a large-scale multi-agent communicative and cooperative planning benchmark built on the PARTNR environment. Experiments on existing tasks and our new benchmark demonstrate significant improvements in cooperative efficiency and task success rates. Across four backbone LLMs, our method achieves average success rate improvements of 4.5% on the PARTNR-Dialog benchmark and 6.8% on the TDW-MAT benchmark over state-of-the-art open-source communicative agent frameworks. See the LLawCo project page for details: this https URL

**arXiv ID:** 2606.28182
</details>

<details>
<summary><strong>AgentPSO: Evolving Agent Reasoning Skill via Multi-agent Particle Swarm Optimization</strong> - Hyunmin Hwang, Jaemin Kim, Choonghan Kim, Hangeol Chang, Jong Chul Ye - [[pdf]](https://arxiv.org/pdf/2605.08704)</summary>

**Abstract:** Multi-agent reasoning has shown promise for improving the problem-solving ability of large language models by allowing multiple agents to explore diverse reasoning paths. However, most existing multi-agent methods rely on inference-time debate or aggregation, which can be vulnerable to incorrect peer influence and biased consensus. Moreover, the agents themselves remain static, as their underlying reasoning skills do not evolve across tasks. In this paper, we introduce \textbf{AgentPSO}, a particle-swarm-inspired framework for evolving multi-agent reasoning skills. AgentPSO treats each agent as a particle-like reasoner whose state is a natural-language skill and whose velocity is a semantic update direction, iteratively guiding agents toward higher-performing skill configurations. Across training iterations, each agent updates its skill by combining its previous velocity, personal-best skill, global-best skill, and a self-reflective direction derived from peer reasoning trajectories. This enables agents to learn reusable reasoning behaviors by drawing on their own experience and on the strongest skills found by the population, without updating the parameters of the backbone language model. Experiments on mathematical and general reasoning benchmarks show that AgentPSO improves over static single-agent skills and test-time-only multi-agent reasoning baselines. The evolved skills further transfer across benchmarks and to another backbone model, suggesting that AgentPSO captures reusable reasoning procedures rather than merely optimizing benchmark-specific prompts. Code is publicly available at this https URL.

**arXiv ID:** 2605.08704
</details>

<details>
<summary><strong>AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems</strong> - Changxin Lao, Fei Pan, Guozhuang Ma, Han Li, Huihuang Lin, Jijun Shi, Kangzhi Zhao, Kun Gai, Mo Zhou, Qinqin Zhou, Quan Chen, Ruochen Yang, Shifu Bie, Shijie Yi, Shuang Yang, Shuo Yang, Wenhao Li, Wentao Xie, Xiao Lv, Xuming Wang, Yijun Wang, Yiming Chen, Yusheng Huang, Zhongyuan Wang, Zibo Zhao, Zijie Zhuang, Baoning Xia, Chao Liu, Chaoyi Ma, Chubo He, Dawei Cong, Feng Jiang, Gang Wang, Guilin Xia, Hanwen Xu, Jiahong Xie, Jiahui Qiao, Jian Liang, Jiangfan Yue, Jing Wang, Jinghan Yang, Jinghui Jia, Kan Qin, Lei Wang, Ming Li, Peilin Song, Pengbo Xu, Qiang Luo, Ruiming Tang, Shiyang Liu, Shuxian Jin, Tao Wang, Tao Zhang, Xiang Gao, Xianghan Li, Yingsong Luo, Yiwen Ning, Yongcheng Liu, Yueyang Liu, Yuan Guo, Zhaojie Liu, Zhenkai Cui - [[pdf]](https://arxiv.org/pdf/2606.26859)</summary>

**Abstract:** Recommendation algorithm iteration is moving from an artisanal, engineer-bound process toward an industrialized research loop, but this transition remains blocked by a structural execution bottleneck: the idea-to-launch cycle still depends on human engineers to generate hypotheses, modify production code, launch A/B experiments, and attribute online results. Innovation therefore scales linearly with headcount rather than compounding with evidence, compute, and accumulated experimental knowledge. We present AgentX, a production-deployed multi-agent system that fundamentally restructures this production function. AgentX operates as a self-evolving development engine: it autonomously generates, implements, evaluates, and learns from recommendation experiments at a scale and pace that no manual workflow can sustain.
The system orchestrates four tightly coupled stages in a closed loop. A Brainstorm Agent synthesizes evidence from historical experiments, system architecture, data analysis, and external research into ranked, executable proposals. A Developing Agent translates each proposal into production-ready code through repository-grounded generation and multi-dimensional reliability verification. An Evaluation Agent conducts safe online rollout with guardrail-vetoed A/B judgment, converting both successes and failures into structured knowledge assets. A Harness Evolution layer (SGPO) then distills execution trajectories into semantic-gradient updates that continuously sharpen the agents themselves -- making the system not merely automated, but self-improving.

**arXiv ID:** 2606.26859
</details>

<details>
<summary><strong>Seven Security Challenges That Must be Solved in Cross-domain Multi-agent LLM Systems</strong> - Ronny Ko, Jiseong Jeong, Shuyuan Zheng, Chuan Xiao, Tae-Wan Kim, Makoto Onizuka, Won-Yong Shin - [[pdf]](https://arxiv.org/pdf/2505.23847)</summary>

**Abstract:** Large language models (LLMs) are rapidly evolving into autonomous agents that cooperate across organizational boundaries, enabling joint disaster response, supply-chain optimization, and other tasks that demand decentralized expertise without surrendering data ownership. Yet, cross-domain collaboration shatters the unified trust assumptions behind current alignment and containment techniques. An agent benign in isolation may, when receiving messages from an untrusted peer, leak secrets or violate policy, producing risks driven by emergent multi-agent dynamics rather than classical software bugs. This position paper maps the security agenda for cross-domain multi-agent LLM systems. We introduce seven categories of novel security challenges, for each of which we also present plausible attacks, security evaluation metrics, and future research guidelines.

**arXiv ID:** 2505.23847
</details>

<details>
<summary><strong>Hierarchical Control in Multi-Agent Games: LLM-based Planning and RL Execution</strong> - Jannik Hösch, Alessandro Sestini, Florian Fuchs, Amir Baghi, Joakim Bergdahl, Konrad Tollmar, Jean-Philippe Barrette-LaPierre, Linus Gisslén - [[pdf]](https://arxiv.org/pdf/2606.20014)</summary>

**Abstract:** Reinforcement learning (RL) has achieved strong performance in sequential decision-making, yet scaling to complex multi-agent environments remains challenging due to sparse rewards, large state-action spaces, and the difficulty of learning coordinated strategies. We propose a hierarchical architecture where a pretrained large language model (LLM) acts as a centralized strategic controller that selects among specialized RL skill policies for a team of agents, while RL policies handle reactive low-level execution. We evaluate this hybrid system in a competitive 2v2 King of the Hill environment against behavior tree (BT) and \emph{``Flat''} RL (end-to-end training without skill decomposition) baselines. The LLM+RL system achieves task performance statistically equivalent to hand-crafted BT (46.4\% vs 51.5\% win rate, $p=0.103$) while both significantly outperform Flat RL trained without skill decomposition. A user study ($n=15$) reveals that 60\% of participants perceive LLM+RL agents as the most human-like ($p=0.027$), citing behavioral adaptability and tactical variability. These results demonstrate that pretrained LLM reasoning can effectively orchestrate pretrained RL skills, achieving competitive multi-agent coordination and superior perceived believability without manual rule engineering.

**arXiv ID:** 2606.20014
</details>

<details>
<summary><strong>Contagion Networks: Evaluator Preference Propagation in Multi-Agent LLM Systems</strong> - Zewen Liu - [[pdf]](https://arxiv.org/pdf/2606.20493)</summary>

**Abstract:** When large language models serve as evaluators in multi-agent systems, their strategy preferences -- whether induced by explicit prompts or by shared architectural priors -- propagate through the agent network. We introduce Contagion Networks, a formal framework for measuring how evaluator preferences spread across interacting LLM agents. In a controlled 3-agent experiment using DeepSeek-chat with three distinct evaluator preference profiles (structured, balanced, evidence-based), we measure the Cross-Agent Contagion Matrix Gamma_3 and find that preferences consistently propagate between agents (gamma in [0.157, 0.352]). A neutral-prompt control experiment reveals a counter-intuitive result: shared architectural priors dominate explicit preference prompts as the driver of contagion (rho_neutral = 1.498 vs. rho_mixed = 1.299; prompt contribution: -63.5%). We identify three propagation regimes governed by the spectral radius rho(Gamma_N) and demonstrate that the same agents suppress preference contagion in chain topology (beta_3 = 0.0126 +/- 0.0038, 95% CI [0.0089, 0.0163], n=4 seeds) but cascade in fully-connected topology (Delta H_avg = -0.020) -- a topology-dependent regime transition validated both for homogeneous and cross-model agent pools (rho^cross = 1.296 +/- 0.016, n=4). We show that increasing evaluator committee size from k=1 to k=3 reduces effective contagion by 68.9% +/- 14.1% (n=4 seeds), providing an actionable mitigation strategy. We release the open-source Contagion Network experimental framework.

**arXiv ID:** 2606.20493
</details>

<details>
<summary><strong>Delayed Verification Destabilizes Multi-Agent LLM Belief: Instability Thresholds and Optimal Corrector Placement</strong> - Igor Itkin - [[pdf]](https://arxiv.org/pdf/2606.27409)</summary>

**Abstract:** Multi-agent large language model (LLM) systems often rely on verifier and critic agents to suppress hallucinations, but verification is delayed. During this delay, false claims can propagate through the agent network. We model this process as delayed consensus on a graph with grounded corrector nodes. Spectral decomposition by the grounded Laplacian yields a closed-form stability threshold for the verification dose: correction that is too strong or too delayed can turn consensus into oscillation. The most unstable regime occurs when the communication and verification delays coincide; for delay two, the threshold is the inverse golden ratio. The same framework gives a supermodular placement objective and a greedy (1-1/e)-approximation rule for assigning a limited corrector budget to influential nodes. Experiments across five open models confirm the predicted dose-delay oscillations. By contrast, grounded factual answering makes truth an absorbing boundary and eliminates the effect, suggesting that the instability is specific to signed-belief tasks while grounded verification remains stabilizing

**arXiv ID:** 2606.27409
</details>

<details>
<summary><strong>QueenBee Planner: Skill-Evolving Communication Topologies for Token-Efficient LLM Multi-Agent Systems</strong> - Congjia Tian, Yuhang Yao, Jiaming Cui - [[pdf]](https://arxiv.org/pdf/2606.27492)</summary>

**Abstract:** Large language model (LLM) multi-agent systems increasingly depend not only on how individual agents reason, but also on how agents are connected. This paper introduces QueenBee Planner, a framework that treats inter-agent communication topology as a retrievable and self-improving design skill. A pool of worker agents, the task adapter, and the scoring function are frozen; only an outer LLM planner learns to generate temporal communication DAGs specifying who sends information to whom, in which round, who merges messages, and who emits the final answer. Execution traces are distilled into evidence-backed design rules with three actions: \emph{Preserve}, \emph{Modify}, and \emph{Avoid}. To prevent self-evolution from turning lucky runs or plausible but false explanations into policy, QueenBee uses held-out acceptance gates, variance-aware credit, motif-level attribution, transfer trust, insight falsification, and structural deduplication. We evaluate the method on Count-Frequency aggregation and Silo-Bench-style distributed coordination tasks. With fixed workers, self-evolved graph generation produces communication structures that improve over fixed topologies and cold generation. In the CF fulltest setting, the best generated graph reduces RMSE from 12.53 for the strongest fixed topology to 7.87 while also reducing messages, model calls, and token cost; Silo-style results show the same direction of improvement over cold and fixed-topology baselines. These results suggest that multi-agent systems can learn reusable architectural design knowledge rather than merely memorizing task answers.

**arXiv ID:** 2606.27492
</details>

<details>
<summary><strong>GBC: Gradient-Based Connections for Optimizing Multi-Agent Systems</strong> - Xiaocheng Yang, Abdulrahman Alrabah, Dilek Hakkani-Tür, Gokhan Tur - [[pdf]](https://arxiv.org/pdf/2606.28187)</summary>

**Abstract:** Multi-agent systems (MAS) built on large language models (LLMs) provide a promising framework for solving complex tasks through role specialization and structured interaction. However, their performance is often limited by miscoordination and, more fundamentally, the lack of fine-grained credit assignment across agents. Existing approaches typically rely on coarse-grained feedback, making it difficult to identify which agents or interaction steps are responsible for errors. We propose Gradient-Based Connections (GBC), an approach for fine-grained attribution and optimization of multi-agent systems. GBC models a MAS as a computational graph and introduces gradient-based connection weights to quantify the influence of each agent's output on downstream agents at the token level. By constructing an attribution graph and propagating task-specific loss signals backward, our method enables precise identification of error sources and targeted prompt optimization. We further develop AgentChord, an efficient implementation that leverages prefix-based gradient computation. Experiments on MultiWOZ and {\tau}-bench show that GBC improves multi-agent performance and outperforms strong single-agent and multi-agent baselines, and higher attribution quality is associated with greater optimization effectiveness. Code is available at: this https URL.

**arXiv ID:** 2606.28187
</details>

<details>
<summary><strong>MMAO: A Metabolic Multi-Agent Optimizer with Endogenous Resource Allocation for Continuous and Discrete Optimization</strong> - Jinliang Xu, Liping Ma - [[pdf]](https://arxiv.org/pdf/2606.28109)</summary>

**Abstract:** Traditional meta-heuristics often rely on fixed population sizes, manually chosen search scales, and externally attached parameter-control modules. This paper presents the \textit{Metabolic Multi-Agent Optimizer} (MMAO), a cross-domain optimization framework in which adaptation is derived endogenously from a private-public metabolic resource loop. Each agent carries internal energy, a continuous role state, motion or structural memory, and local search history, while the population shares a communal resource pool. Fitness improvements are converted into normalized metabolic gains through a robust progress scale and a recent success statistic; the same closed loop then regulates sensing intensity, search amplitude, role drift, branching, pruning, respawning, and elite reinvestment. In the continuous setting, MMAO uses energy-regulated symmetric zero-order probing and role-interpolated motion. In the discrete setting, the same control law is instantiated through structural sensing, local route improvement, guided perturbation, and energy-weighted edge reuse. The paper combines an implementation-faithful formulation with a reproducible experimental study on a CEC2017 subset (10D/30D, 20 seeds) and five TSPLIB instances (100 discrete runs in total). The current evidence supports MMAO primarily as a parameter-light, self-calibrating optimization framework whose main validated originality lies in metabolically endogenous resource allocation across heterogeneous search behaviors, rather than as a universally superior optimizer.

**arXiv ID:** 2606.28109
</details>

<details>
<summary><strong>On dynamic multi-agent pathfinding methods: review, simulations and modifications</strong> - Gabriel Fejziaj, Salama Hassona, Wieslaw Marszalek - [[pdf]](https://arxiv.org/pdf/2606.03735)</summary>

**Abstract:** This paper presents a systematic study of pathfinding algorithms in the context of Dynamic Multi-Agent Pathfinding (D-MAPF), a setting that combines dynamic obstacles, partial observability, and inter-agent conflicts. We evaluate six representative algorithms: Dijkstra, D* Lite, Space-Time A*, WHCA*, M*, and a novel method denoted as A** within a unified simulation framework. The proposed A** algorithm introduces a template-based approach that decouples offline geometric path generation from online temporal adaptation. By precomputing multiple diverse candidate paths and dynamically reconnecting to them using space-time planning, A** improves solution quality in environments with frequent changes and limited sensing

**arXiv ID:** 2606.03735
</details>

<details>
<summary><strong>MedGuards: Multi-Agent System for Reliable Medical Error Detection and Correction</strong> - Congbo Ma, Hu Wang, Yichun Zhang, Farah E. Shamout - [[pdf]](https://arxiv.org/pdf/2606.25651)</summary>

**Abstract:** As Large Language Models (LLMs) are increasingly deployed in healthcare settings, accurate error detection and correction in generated or existing text becomes critical, as even minor mistakes can pose risks to patient safety. Existing methods for error detection and correction, including automated checks and heuristic-based approaches, do not generalize well across unseen datasets. In this paper, we propose MedGuards as a medical safety guardrail, which is a new framework that treats medical error detection and correction as a multi-agent in-context learning task. Specialized agents separately detect, localize, and correct errors, while a confidence-guided arbitration mechanism resolves disagreements using reasoning traces and confidence scores. This design enhances interpretability, robustness, and adaptability, without requiring additional training of the base LLMs. Additionally, we introduce the Keyword-Prioritized Correction Score (KPCS), a new evaluation metric that considers whether critical keywords within the reference text are generated correctly, providing a more comprehensive assessment than conventional metrics. Experiments across four multilingual medical datasets consisting of clinical notes demonstrate significant improvements by the proposed framework across several metrics and models. Our aim is to enable safer deployment of LLMs in real-world healthcare applications. For reproducibility, we make our code publicly available at this https URL.

**arXiv ID:** 2606.25651
</details>

<details>
<summary><strong>From Detection to Action: Using LLM Agents for Fault-Tolerant Control</strong> - Javal Vyas, Milapji Singh Gill, Artan Markaj, Felix Gehlhoff, Mehmet Mercangöz - [[pdf]](https://arxiv.org/pdf/2606.28011)</summary>

**Abstract:** We propose an agentic Large Language Model (LLM) framework for active Fault-Tolerant Control (FTC) that transforms fault detection outputs into constraint-aware recovery actions grounded in plant-specific knowledge. The approach couples (i) a multi-agent workflow that decomposes operator duties into monitoring, planning, action synthesis, simulation, validation, and reprompting; (ii) a Digital Process Plant Twin (DPPT) that exposes plant data, models, and a simulation service for pre-execution testing; and (iii) a Graph Retrieval-Augmented Generation (Graph RAG) layer built on the CPSMod ontology, which organizes plant knowledge (structure, function, hybrid dynamics, control context, and fault semantics) into a graph that supports relation-aware, multi-hop retrieval for the agents. Corrective actions are generated as minimal-risk state-machine recovery paths and corresponding discrete commands or continuous setpoint adaptations, then validated deterministically against interlocks, envelopes, and dynamic feasibility before any actuation. If no acceptable plan is found within a bounded time window, control is handed to a safety fallback. The framework is evaluated in simulation on two representative benchmarks: a discrete batch Mixing Module and a Continuous Stirred-Tank Reactor (CSTR) under closed-loop PID regulation. Results with lightweight LLMs (GPT-4o-mini and GPT-4.1-mini) show that semantically grounded agents can derive valid recovery decisions within latency budgets compatible with the respective process dynamics, demonstrating a practical pathway from detection to validated corrective action across both discrete and continuous FTC tasks.

**arXiv ID:** 2606.28011
</details>

<details>
<summary><strong>When Multi-Robot Systems Meet Agentic AI:Towards Embodied Collective Intelligence</strong> - Yuxuan Yan, Yuanyuan Jia, Qianqian Yang - [[pdf]](https://arxiv.org/pdf/2606.27929)</summary>

**Abstract:** Embodied AI is increasingly becoming agentic, shifting robots from perception--control pipelines towards closed-loop systems that can retrieve context, deliberate during execution, monitor feedback, and refine future behavior. In parallel, robotics research has also moved from single-robot autonomy towards multi-robot systems, driven by the need for wider sensing, distributed action, heterogeneous capabilities, and fault tolerance. As AI agents move from single-agent use towards multi-agent collaboration, robotics faces a parallel challenge: robot teams must move beyond sharing maps, task assignments, and datasets towards sharing the state produced by embodied agent loops. This article explores Embodied Collective Intelligence (ECI), a future multi-robot paradigm in which a robot team accumulates and uses world context, task progress, and skill experience as shared resources. Specifically, we first review how embodied AI is becoming agentic and how multi-robot cooperation has evolved. We then present Embodied Collective Intelligence through Co-Perception, Co-Action, and Co-Evolution. Finally, we use an illustrative navigation study to examine one concrete component of the concept: shared world-memory inheritance. The study shows that a newly added robot can benefit from merged team memory, but it is not intended as a full evaluation of the ECI framework. Taken together, the review and conceptual framework motivate Embodied Collective Intelligence as a direction for embodied multi-agent intelligence, while the case study grounds one measurable part of the concept.

**arXiv ID:** 2606.27929
</details>

</details>

<details open>
<summary><h2>Other Agent Research (5 papers)</h2></summary>

<details>
<summary><strong>ProMSA:Progressive Multimodal Search Agents for Knowledge-Based Visual Question Answering</strong> - ZhengXian Wu, Hangrui Xu, Kai Shi, Zhuohong Chen, Yunyao Yu, Chuanrui Zhang, Zirui Liao, Jun Yang, Zhenyu Yang, Haonan Lu, Haoqian Wang - [[pdf]](https://arxiv.org/pdf/2606.27974)</summary>

**Abstract:** Knowledge-based Visual Question Answering (KB-VQA) requires models to combine image understanding with external knowledge. Most prior methods use a fixed retrieve-then-generate pipeline with a pre-selected retriever and a static top-k setting, which is not adaptive during reasoning. We propose ProMSA, a progressive multimodal search agent for KB-VQA. Given an image-question pair, the agent iteratively chooses image search, text search, or stop, under explicit tool-call budgets and with deduplication to avoid redundant retrieval. For training, we first use rejection-sampling SFT to learn valid tool-use formats, then optimize the agent with TN-GSPO, a sequence-level RL objective that normalizes updates by both generation length and tool-interaction depth. Experiments on E-VQA and InfoSeek show consistent gains over strong RAG and agent baselines, and improved retrieval and end-to-end accuracy. The code is available at this https URL.

**arXiv ID:** 2606.27974
</details>

<details>
<summary><strong>CPAgents: Agentic Composite Phenotype Generation for Cardiac Disease Association</strong> - Zuoou Li, Wenlong Zhao, Kelly Yu, Weitong Zhang, Paul M. Matthews, Wenjia Bai, Bernhard Kainz, Mengyun Qiao - [[pdf]](https://arxiv.org/pdf/2606.28179)</summary>

**Abstract:** Identifying robust associations between cardiac imaging phenotypes and clinical diseases is fundamental to population-scale cardiovascular research and reliable risk stratification. However, current phenome-wide association studies rely on pre-defined, single-variable phenotypes or expert-crafted features, which limits their ability to capture clinically meaningful non-linear effects and cross-phenotype interactions. To address this, we propose CPAgents, an iterative phenotype-Composition framework for cardiovascular Phenome-wide association study (PheWAS) that automatically constructs and validates interpretable composite phenotypes (e.g., polynomial, ratio, and interaction forms) from base imaging features. Specifically, our system coordinates three agents: (i) an Analyst that identifies statistical pathologies and nominates candidate transformations; (ii) a Proposer that generates constrained, medically and statistically motivated expressions under numerical safety rules; and (iii) a Verifier that evaluates candidates using multi-stage criteria and produces transparent evidence trails for accepted phenotypes. Evaluated on a population-scale cardiac imaging cohort, the discovered composite phenotypes markedly improve disease discrimination: across 72 classifier-disease-metric combinations, our variants achieve the top rank in 56 cases versus 18 for baselines, with gains observed across all nine clinical disease categories. Our framework yields compact, clinically interpretable phenotype formulas with transparent evidence trails, enabling scalable discovery of stronger phenotype-disease associations beyond expert-driven feature selection.

**arXiv ID:** 2606.28179
</details>

<details>
<summary><strong>Towards Value-Constrained Credit Assignment in Fully Delegated AI Cooperatives</strong> - Young Yoon, Jimin Kim, Soyeon Park - [[pdf]](https://arxiv.org/pdf/2606.28217)</summary>

**Abstract:** We propose a framework for reward allocation in fully delegated AI cooperatives where humans are represented by agents that contribute data and participate in model updates under heterogeneous value constraints. The key idea is to credit only those updates that remain admissible after screening them against each principal's value profile. We formulate value-conditioned gradient filtering, online marginal contribution signals, and cumulative revenue settlement within a traversal learning (TL) substrate. TL is especially attractive here because it performs decentralized backpropagation without the quality loss associated with aggregation-centric distributed learning and, we argue, offers a finer attribution substrate than FedAvg-style federated learning by preserving explicit traversal and gradient paths. The framework is positioned against data valuation, federated contribution estimation, personalized federated learning, and pluralistic alignment.

**arXiv ID:** 2606.28217
</details>

<details>
<summary><strong>LinkAnchor: An Autonomous LLM-Based Agent for Issue-to-Commit Link Recovery</strong> - Arshia Akhavan, Alireza Hoseinpour, Abbas Heydarnoori, Hamid Bagheri, Mehdi Keshani - [[pdf]](https://arxiv.org/pdf/2508.12232)</summary>

**Abstract:** Issue-to-commit link recovery in software repositories is fundamental to software traceability and project management, yet it remains a challenging task. Prior studies show that only about 42.2% of issues on GitHub are correctly linked to their commits, highlighting the need for more effective solutions. Existing work has explored a range of ML/DL approaches, and more recently, large language models (LLMs) have been applied to this problem. However, these methods face two major limitations. First, LLMs are restricted by limited context windows and cannot simultaneously process all available data sources, such as long commit histories, extensive issue discussions, and large code repositories. Second, most approaches operate on individual issue-commit pairs, where a model independently scores the relevance of a single commit to an issue. This pairwise formulation fails to account for the complex associativity of software fixes, where an issue is often resolved by an aggregate chain of commits rather than a single atomic change. By ignoring these temporal and parental dependencies, existing methods often fail to incorporate the complete resolution logic and might misidentify intermediate commits as final fixes. Furthermore, this strategy is computationally inefficient in large repositories, as it requires exhaustively evaluating an enormous number of candidate pairs. To address these challenges, we present LinkAnchor, the first autonomous LLM-based agent designed specifically for issue-to-commit link recovery. LinkAnchor introduces a lazy-access architecture that allows the underlying LLM to dynamically retrieve only the most relevant contextual data, such as commits, issue comments, and code files, without exceeding token limits.

**arXiv ID:** 2508.12232
</details>

<details>
<summary><strong>Characterizing Driver Interactions with Autonomous Vehicles via Response Maps</strong> - Dave Broaddus, Rachel DiPirro, Chishang, Yang, Dan Calderone, Wendy Ju, Meeko Oishi - [[pdf]](https://arxiv.org/pdf/2606.27656)</summary>

**Abstract:** Understanding human responses to autonomous vehicle (AV) behaviors is essential for socially aware interaction, which is crucial for socially compatible navigation in shared traffic environments. We characterize human driving responses in interactions with AVs as feedback laws over the coupled state space of the human driven vehicle and the AV. We model the human driver's actions using a response map, a concept based in game theory, and employ a linear representation to capture driver behaviors as a function of AV behaviors, based on empirical data from a driving simulator study. Our results show that 1) human driver acceleration behavior can be captured using response maps, and 2) human driver responses differ significantly with respect to AV behaviors of yielding, non-yielding, and responsive to the human driver.

**arXiv ID:** 2606.27656
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (20 papers)</h2></summary>

<details>
<summary><strong>Tandem Reinforcement Learning with Verifiable Rewards</strong> - Difan Jiao, Raghav Singhal, Robert West, Ashton Anderson - [[pdf]](https://arxiv.org/pdf/2606.28166)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has significantly improved the reasoning capability of large language models, reaching expert or even superhuman performance in domains such as competition math. However, whether weaker agents and humans can actually harness this capability is far less certain, with RLVR documented to drift reasoning toward idiosyncratic patterns such as poor readability and language mixing. Tandem training is a recently introduced paradigm that targets this compatibility problem: a trained, stronger senior co-generates each rollout with a frozen, weaker junior, and the two are rewarded as a team, so the senior is pushed to reason in ways the junior can follow. Yet this paradigm has so far been demonstrated only in proof-of-concept settings, leaving open whether it scales to the long chains of thought of the modern RLVR pipeline. In this work, we propose Tandem Reinforcement Learning (TRL), which carries the tandem training paradigm into RLVR. In TRL, the senior and a frozen junior alternate stochastically to co-generate the reasoning, the resulting generation is rewarded, and the standard GRPO loss is applied to the senior. Training Qwen3-4B-Instruct on competition math, we find that TRL matches vanilla GRPO on solo reasoning capability while three properties emerge together from the same rollout structure: stronger handoff robustness with the junior, reduced distributional drift from the junior, and a chain-of-thought more legible to the junior. Our results demonstrate a promising route for RLVR with practical payoffs in multi-model communication and human compatibility.

**arXiv ID:** 2606.28166
</details>

<details>
<summary><strong>Agentic Episodic Control</strong> - Xidong Yang, Wenhao Li, Junjie Sheng, Yun Hua, Haosheng Chen, Chuyun Shen, Xiangfeng Wang - [[pdf]](https://arxiv.org/pdf/2506.01442)</summary>

**Abstract:** Reinforcement learning (RL) remains fundamentally limited by poor data efficiency and weak generalization. Prior episodic RL methods attempt to alleviate this via external memory modules, yet they suffer from two key limitations: a representation bottleneck caused by shallow encoders, and a retrieval dilemma where episodic memory is accessed indiscriminately. To address these challenges, we propose Agentic Episodic Control (AEC), a novel architecture that integrates large language models (LLMs) into episodic RL. AEC uses an LLM-based semantic augmenter to generate semantic representations from raw observations, and a critical state recognizer to selectively retrieve valuable experiences. This transforms memory usage from passive similarity matching into strategic, context-aware recall. Across five BabyAI-Text environments, AEC achieves 2-6x higher data efficiency than baselines and is the only method to solve complex tasks like UnlockLocal with over 90% success. It further demonstrates strong cross-task and cross-environment generalization, maintaining performance even under distribution shifts. AEC shows that combining LLM-derived priors with reinforcement learning yields more sample-efficient and adaptable agents.

**arXiv ID:** 2506.01442
</details>

<details>
<summary><strong>Conservative Equilibrium Discovery in Offline Game-Theoretic Multiagent Reinforcement Learning</strong> - Austin A. Nguyen, Michael P. Wellman - [[pdf]](https://arxiv.org/pdf/2603.00374)</summary>

**Abstract:** Offline learning of strategies takes data efficiency to its extreme by restricting algorithms to a fixed dataset of state-action trajectories. We consider the problem in a mixed-motive multiagent setting, where the goal is to solve a game under the offline learning constraint. We first frame this problem in terms of selecting among candidate equilibria. Since datasets may inform only a small fraction of game dynamics, it is generally infeasible in offline game-solving to even verify a proposed solution is a true equilibrium. Therefore, we consider the relative probability of low regret (i.e., closeness to equilibrium) across candidates based on the information available. Specifically, we extend Policy Space Response Oracles (PSRO), an online game-solving approach, by quantifying game dynamics uncertainty and modifying the RL objective to skew towards solutions more likely to have low regret in the true game. We further propose a novel meta-strategy solver, tailored for the offline setting, to guide strategy exploration in PSRO. Our incorporation of Conservatism principles from Offline reinforcement learning approaches for strategy Exploration gives our approach its name: COffeE-PSRO. Experiments demonstrate COffeE-PSRO's ability to extract lower-regret solutions than state-of-the-art offline approaches and reveal relationships between algorithmic components empirical game fidelity, and overall performance.

**arXiv ID:** 2603.00374
</details>

<details>
<summary><strong>Edit-R2: Context-Aware Reinforcement Learning for Multi-Turn Image Editing</strong> - Yuxiao Ye, Haoran He, Fangyuan Kong, Xintao Wang, Pengfei Wan, Kun Gai, Ling Pan - [[pdf]](https://arxiv.org/pdf/2606.05950)</summary>

**Abstract:** Text-guided image editing has advanced rapidly with diffusion models and unified multimodal foundation models. However, most existing methods remain confined to single-turn settings, overlooking the more realistic scenario of multi-turn in-context editing, where users iteratively refine an image through a sequence of instructions. In this setting, a model must follow each new instruction while preserving accumulated session-level constraints, challenged by two coupled failure modes: long-context dilution, where sparse textual constraints become difficult to recover from growing interleaved image-text histories, and state contamination, where earlier editing mistakes degrade subsequent generations. We introduce Edit-R2, a novel reinforcement learning post-training framework for unified multimodal models. Edit-R2 reconstructs the operative session intent, which effectively consolidates scattered historical constraints into an explicit reasoning trace before each editing turn. It further enables multi-turn RL over both reasoning and generation through a unified objective that jointly optimizes intent reconstruction generation in discrete text space and flow-matching image generation in continuous latent space, while a trajectory filtering mechanism suppresses corrupted rollouts to stabilize training under state contamination. To support systematic evaluation, we introduce MICE-Bench, a large-scale benchmark for multi-turn in-context editing with automated metrics for instruction following (IF), content consistency (CC), and global awareness (GA) over accumulated session constraints. Experiments show that Edit-R2 substantially improves multi-turn in-context editing and achieves competitive performance compared against strong baselines.

**arXiv ID:** 2606.05950
</details>

<details>
<summary><strong>Auto-Configuring Scientific Simulators with Lightweight Coding-Agent Adapters</strong> - Matthew Ho, Brian Liu, Jixuan Chen, Audrey Wang, Lianhui Qin - [[pdf]](https://arxiv.org/pdf/2606.09774)</summary>

**Abstract:** Configuring an advanced scientific simulator, translating a modeling goal into a valid, runnable input deck, is a persistent bottleneck that costs domain scientists hours to days. Input decks are executable interfaces: simulator-specific vocabulary, cross-file references, schema constraints, and validation rules must align before a simulation can run. We show that this bottleneck can be substantially reduced with a lightweight adapter around an off-the-shelf coding agent, rather than a bespoke simulator agent. Coding agents already navigate files, edit code, run commands, and repair outputs; what they lack is the simulator's executable contract, and rebuilding the agent loop risks discarding harness-calibrated tool-use and self-correction behavior. We introduce SIGA, a coding-agent adapter that supplies this contract through retrieval, procedural memory, agent-callable validation, and validation-gated termination while leaving the model and loop frozen. Because this contract is small and external, SIGA also supports adapter self-evolution: prior trajectories can rewrite the adapter contents without modifying the underlying agent. On GEOS, a multiphysics subsurface simulator, SIGA's main gain is reliability: on harder held-out tasks it improves TreeSim from 0.720 to 0.789 and reduces across-run standard deviation by about 16x by preventing empty or invalid decks. In a human calibration, SIGA reaches in about five minutes the deck quality a domain expert reached in about three hours. Transfers to OpenFOAM and LAMMPS show the recipe is portable but interface-dependent: completion gates help when structural completeness is the bottleneck, while memory and retrieval help when value correctness is.

**arXiv ID:** 2606.09774
</details>

<details>
<summary><strong>What the LLM Should Not Say: Boundary-Aware Context Grounding for A Seven-Channel EEG Agent</strong> - Zhiyuan Xu, Yueqing Dai, Junling Li, Junwen Luo - [[pdf]](https://arxiv.org/pdf/2606.26519)</summary>

**Abstract:** Large language models (LLMs) can make scientific software easier to use. However, a general model does not automatically know which measurements a particular sensor can support, which algorithms are implemented in the current software, or which conclusions are justified by a computed result. These distinctions are especially important for low-channel electroencephalography (EEG), where sparse spatial coverage and variable signal quality make plausible but unsupported interpretations easy to produce. We present NeuraDock Agent, an open-source architecture that separates a deterministic local EEG engine from a hardware-aware language layer. The numerical engine parses recordings, performs quality control, executes reviewed spectral workflows, and writes machine-readable artifacts. The LLM receives only a compact, allowlisted summary and a versioned context pack. The context describes the seven-channel hardware, reviewed workflows, result fields, implementation boundaries, scientific limits, and reference cases. Raw EEG and dense per-sample arrays remain local
We evaluate the system at three levels. First, 12 recordings produced identical structured results over ten numerical repetitions, and a complete Rest/Task run produced identical result, report, and figure hashes over three repetitions. Second, request-capture and failure-injection experiments confirmed the tested data boundary and preservation of local artifacts under HTTP, malformed-output, and connection failures. Third, a boundary-awareness benchmark tested 36 ordinary and adversarial questions under four context ablations and two LLMs, yielding 288 this http URL results support hardware- and implementation-aware grounding as a practical mechanism for calibrating what an EEG agent accepts, qualifies, or refuses; they do not establish clinical validity or a validated absolute cognitive-load index.

**arXiv ID:** 2606.26519
</details>

<details>
<summary><strong>A Primer on SO(3) Action Representations in Deep Reinforcement Learning</strong> - Martin Schuck, Sherif Samy, Angela P. Schoellig - [[pdf]](https://arxiv.org/pdf/2510.11103)</summary>

**Abstract:** Many robotic control tasks require policies to act on orientations, yet the geometry of SO(3) makes this nontrivial. Because SO(3) admits no global, smooth, minimal parameterization, common representations such as Euler angles, quaternions, rotation matrices, and Lie algebra coordinates introduce distinct constraints and failure modes. While these trade-offs are well studied for supervised learning, their implications for actions in reinforcement learning remain unclear. We systematically evaluate SO(3) action representations across three standard continuous control algorithms, PPO, SAC, and TD3, under dense and sparse rewards. We compare how representations shape exploration, interact with entropy regularization, and affect training stability through empirical studies and analyze the implications of different projections for obtaining valid rotations from Euclidean network outputs. Across a suite of robotics benchmarks, we quantify the practical impact of these choices and distill simple, implementation-ready guidelines for selecting and using rotation actions. Our results highlight that representation-induced geometry strongly influences exploration and optimization and show that representing actions as tangent vectors in the local frame yields the most reliable results across algorithms. The project webpage and code are available at this http URL.

**arXiv ID:** 2510.11103
</details>

<details>
<summary><strong>Hybrid Fact-Checking that Integrates Knowledge Graphs, Large Language Models, and Search-Based Retrieval Agents Improves Interpretable Claim Verification</strong> - Shaghayegh Kolli, Richard Rosenbaum, Timo Cavelius, Lasse Strothe, Andrii Lata, Jana Diesner - [[pdf]](https://arxiv.org/pdf/2511.03217)</summary>

**Abstract:** Large language models (LLMs) excel in generating fluent utterances but can lack reliable grounding in verified information. At the same time, knowledge-graph-based fact-checkers deliver precise and interpretable evidence, yet suffer from limited coverage or latency. By integrating LLMs with knowledge graphs and real-time search agents, we introduce a hybrid fact-checking approach that leverages the individual strengths of each component. Our system comprises three autonomous steps: 1) a Knowledge Graph (KG) Retrieval for rapid one-hop lookups in DBpedia, 2) an LM-based classification guided by a task-specific labeling prompt, producing outputs with internal rule-based logic, and 3) a Web Search Agent invoked only when KG coverage is insufficient. Our pipeline achieves an F1 score of 0.93 on the FEVER benchmark on the Supported/Refuted split without task-specific fine-tuning. To address Not enough information cases, we conduct a targeted reannotation study showing that our approach frequently uncovers valid evidence for claims originally labeled as Not Enough Information (NEI), as confirmed by both expert annotators and LLM reviewers. With this paper, we present a modular, opensource fact-checking pipeline with fallback strategies and generalization across datasets.

**arXiv ID:** 2511.03217
</details>

<details>
<summary><strong>Trust Region Masking for Long-Horizon LLM Reinforcement Learning</strong> - Yingru Li, Jiacai Liu, Jiawei Xu, Yuxuan Tong, Ziniu Li, Qian Liu, Baoxiang Wang - [[pdf]](https://arxiv.org/pdf/2512.23075)</summary>

**Abstract:** Policy gradient methods for Large Language Models optimize a policy $\pi_\theta$ via a surrogate objective computed from samples of a rollout policy $\pi_{\text{roll}}$. However, modern LLM-RL pipelines suffer from unavoidable implementation divergences -- backend discrepancies, Mixture-of-Experts routing discontinuities, and distributed training staleness -- causing off-policy mismatch ($\pi_{\text{roll}} \neq \pi_\theta$) and approximation errors between the surrogate and the true objective. We demonstrate that classical trust region bounds on this error scale as $O(T^2)$ with sequence length $T$, rendering them vacuous for long-horizon tasks. To address this, we derive a family of bounds -- both KL-based and TV-based -- including a Pinsker-Marginal bound ($O(T^{3/2})$), a Mixed bound ($O(T)$), and an Adaptive bound that strictly generalizes the Pinsker-Marginal bound via per-position importance-ratio decomposition. Taking the minimum over all bounds yields the tightest known guarantee across all divergence regimes. Crucially, all bounds depend on the maximum token-level divergence $D_{\mathrm{KL}}^{\mathrm{tok,max}}$ (or $D_{\mathrm{TV}}^{\mathrm{tok,max}}$), a sequence-level quantity that cannot be controlled by token-independent methods like PPO clipping. We propose Trust Region Masking (TRM), which masks entire sequences violating the trust region, enabling the first non-vacuous monotonic improvement guarantees for long-horizon LLM-RL.

**arXiv ID:** 2512.23075
</details>

<details>
<summary><strong>Glite ARF: Verifier-Driven Research with Parallel LLM Coding Agents</strong> - Vassili Philippov, Pavel Katunin, Dmitry Andreev, Igor Ostanin, Anton Nikolaev - [[pdf]](https://arxiv.org/pdf/2606.27416)</summary>

**Abstract:** LLM coding agents make it tempting to automate empirical research by delegating experiments to them directly, but naive delegation does not scale to large projects: low-rate instruction lapses compound into broken, irreproducible artefacts. To address this problem, we present Glite ARF, an open-source Python framework for running many LLM coding agents in parallel on a research repository without sacrificing reproducibility or auditability. The framework defines a three-role stack: a human researcher chooses which hypotheses to test, coding agents (Claude Code, Codex CLI) implement individual tasks under a fixed structure, and deterministic Python verifier scripts enforce task isolation, immutability of completed work, a corrections overlay, and a materialised project overview. We call this verifier-driven research: the rules of the research process live in code that fails loudly when violated, not in prose that agents are merely asked to follow. Using Glite ARF, we developed our submission to the BEA 2026 vocabulary-difficulty shared task, placing first in the closed track and second in the open track on all three target languages (Spanish, German, Mandarin) and reducing the official baseline RMSE by 29.9% (closed) and 35.9% (open). The campaign comprised 273 tracked tasks (146 experiment runs) across 129 feature sets, run by up to twelve parallel agents orchestrated from a single laptop - with some model training on rented A100s - at approximately \$450 in LLM API spend (\$498 total third-party cost), and structured per-fold provenance let us catch and strip four target-leaking feature sets, correcting an implausible 0.609 RMSE to 0.802. Across three campaigns in three domains, the framework's structural machinery adds only about 1% of wall-clock time. Framework and a public demo project accompany this paper.

**arXiv ID:** 2606.27416
</details>

<details>
<summary><strong>GenWorld: Empirically Grounded Urban Simulation Infrastructure for Scalable LLM-Agent Studies</strong> - Gen Li, Jieyuan Lan, Pengcheng Xu, Zongyuan Wu, Masaki Ogura, Tao Feng - [[pdf]](https://arxiv.org/pdf/2606.27650)</summary>

**Abstract:** LLM-agent simulation faces a joint grounding and scaling problem: agents should act in environments that reflect real urban constraints, yet direct online LLM calls for city-scale populations are computationally prohibitive. We present GenWorld, an empirically grounded urban simulation infrastructure that combines a building-level synthetic city, a structured agent-environment interface, and offline compilation of LLM-derived decision signals into lookup policies for scalable rollout. In a reference instantiation for Higashihiroshima, Japan, GenWorld grounds 196,608 synthetic residents in census and geospatial data, validates demographic consistency against census tabulations, and uses YJMob100K mobile-phone data as a commuting-distance diagnostic. We demonstrate the infrastructure through three reproducible cases: a full-city weekday rollout, a weekday-weekend behavioral contrast, and a warning-response perturbation with auditable replanning traces. These cases support GenWorld as a reproducible platform for grounded and scalable LLM-agent studies, while calibrated forecasting for traffic, evacuation, or policy outcomes remains future work.

**arXiv ID:** 2606.27650
</details>

<details>
<summary><strong>PerturbCellRL: Verifier-Guided Reinforcement Learning for Single-Cell Perturbation Prediction</strong> - Dongxia Wu, Mingyu Li, Yuhui Zhang, Anurendra Kumar, Emma Lundberg, Serena Yeung-Levy, Emily B. Fox - [[pdf]](https://arxiv.org/pdf/2606.27752)</summary>

**Abstract:** Single-cell perturbation models can reduce costly wet-lab screening by predicting how cells respond transcriptionally to interventions. While recent generative models improve population-level prediction, individual generated cells are not explicitly checked for biological consistency. We introduce PerturbCellRL, a reinforcement learning (RL) framework that post-trains a pretrained single-cell transcriptomic generator using a suite of cell-level verifiers as rewards. These verifiers define four rewards: Pearson top-k similarity, RMSE top-k proximity, DE Spearman, and Pathway activity. The Pathway activity verifier rewards cells whose pathway responses match known perturbation biology. We evaluate PerturbCellRL on multiple genetic and chemical perturbation benchmarks. Across these benchmarks, PerturbCellRL improves over the pretrained flow-matching generator on reward-aligned evaluation metrics and a held-out evaluation metric. Moreover, PerturbCellRL remains competitive with state-of-the-art methods on population-level metrics. Together, these results frame trustworthy single-cell prediction as verifier-guided generative alignment, moving beyond matching expression distributions toward predictions whose single-cell perturbation effects are explicitly checked for biological consistency.

**arXiv ID:** 2606.27752
</details>

<details>
<summary><strong>NormGuard: Reward-Preserving Norm Constraints in Flow-Matching Reinforcement Learning</strong> - Tianlin Pan, Lianyu Pang, Cheng Da, Huan Yang, Changqian Yu, Kun Gai, Wenhan Luo - [[pdf]](https://arxiv.org/pdf/2606.27771)</summary>

**Abstract:** Reinforcement learning (RL) post-training improves the reward alignment of flow-based generators, but often degrades perceptual quality in ways that are not captured by the reward proxy. We identify a simple structural signature of this drift: across three post-training methods (NFT, AWM, DPO), RL fine-tuning inflates the per-step velocity norm $\|v_\theta\|$ by $5\%$ to $15\%$ relative to the reference. A form of norm inflation has been studied in classifier-free guidance (CFG), where rescaling the velocity back to a reference norm at inference time can mitigate the resulting artifacts. However, this inference-time correction does not transfer cleanly to RL: rescaling $v_\theta$ to match $\|v_{\text{ref}}\|$ at inference time neither improves reward nor fixes the quality degradation, because the inflation is co-adapted into the model weights. Furthermore, an adjoint sensitivity analysis shows that velocity magnitude rescaling carries no coherent first-order reward signal at the batch level, indicating that suppressing norm inflation is unlikely to remove a consistently reward-carrying component. Since inference-time renormalization fails while norm suppression carries no reward cost, training-time intervention is the appropriate strategy. Together, these findings motivate \methodname, a hinge penalty that activates only when $\|v_\theta\|$ exceeds $\|v_{\text{ref}}\|$ and composes additively with any velocity-local base loss. Across two base models, three post-training methods, and two reward proxies, \methodname consistently improves MLLM-judged image quality and forensic realism while preserving reward, with gains that amplify under few-step inference and are not explained by early stopping.

**arXiv ID:** 2606.27771
</details>

<details>
<summary><strong>Regularized Reward-Punishment Reinforcement Learning</strong> - Jiexin Wang, Eiji Uchibe - [[pdf]](https://arxiv.org/pdf/2606.28152)</summary>

**Abstract:** We propose KL-Coupled Policy Regularization (KCPR), a policy coordination framework for Reward-Punishment Reinforcement Learning (RPRL). Based on KCPR, we derive KL-Coupled Soft Optimality (KCSO) and develop its deep realization, klDMP. Unlike existing RPRL approaches that optimize reward-seeking and punishment-related policies largely independently, KCPR enables direct interactions between companion policies by treating each as a dynamically learned prior for the other. KCSO yields coupled soft-optimal policies and KL-regularized Bellman operators, allowing reward and punishment information to jointly influence value propagation. To improve learning stability, we introduce a companion-prior softening mechanism and evaluate separate replay-buffer designs for balancing reward- and punishment-related experience. Experiments in grid-world and Gazebo robotic navigation tasks demonstrate that klDMP improves safety and learning stability while maintaining competitive task performance compared with DQN, SQL and softDMP. These results suggest that policy-level coordination provides an effective mechanism for integrating multiple behavioral objectives and may serve as a useful design principle for reinforcement learning systems with interacting motivational processes.

**arXiv ID:** 2606.28152
</details>

<details>
<summary><strong>LocalNav: Distilling Frontier VLMs and Embodied RL for On-Device Object Goal Navigation</strong> - Nicolas Baumann, Liam Boyle, Pu Deng, Edoardo Ghignone, Boyang Sun, Marc Pollefeys, Luca Benini, Michele Magno - [[pdf]](https://arxiv.org/pdf/2606.27871)</summary>

**Abstract:** Vision Language Models (VLMs) have emerged in the robotic domain as a powerful tool that enables environmental perception with language context, serving as a catalyst for open-vocabulary tasks like ObjectNav. Yet, their computational footprint typically confines them to cloud execution, hindering low-latency inference with local deployment on resource-constrained robots. To address this challenge, we present a distillation strategy that transfers complex spatial-semantic reasoning from large frontier models into a lightweight, 4B-parameter local VLM for edge execution on embedded GPU devices (e.g., Jetson Orin). We first establish a State of the Art (SotA), Scene Graph (SG)-based pipeline using Claude Sonnet 4.6, achieving a 39.7% Success Rate (SR) on the HM3D OVON benchmark. We then demonstrate that fine-tuning Qwen3.5-4B on just 500 frontier reasoning traces effectively enables navigation capabilities, yielding a SR of 34.5%, narrowing the gap to the performance of large cloud models. Finally, we introduce E-RLVR with Token Generation (TG) regularization to compress output sequence lengths for physical deployment while grounding the agent in its task. This downstream optimization reduces TG overhead by 72.1% and latency by 71.8%. Combined with quantization, this joint strategy yields a cumulative 82.8% reduction in overall inference latency without significantly sacrificing performance, presenting a viable paradigm for local, low-latency VLM execution on mobile robots.

**arXiv ID:** 2606.27871
</details>

<details>
<summary><strong>Building a Scalable, Reproducible, Evaluatable, and Closed-Loop Simulation Environment Foundation for Embodied Intelligence Cloud-Native Simulation Infrastructure for Embodied Intelligence Training, Evaluation, and Data Collection</strong> - Junwu Xiong, Yongjian Guo, Mingxi Luo, Ning Qiao, Lei Kang, Song Wang, Yince Gao - [[pdf]](https://arxiv.org/pdf/2606.27962)</summary>

**Abstract:** This paper presents a cloud-native simulation infrastructure framework for embodied intelligence that supports large-scale training, standardized evaluation, and simulation-based data collection. The framework unifies simulation environment generation, task execution, trajectory collection, model evaluation, data management, and cloud services into a scalable and reproducible platform.
To address the high cost, limited scalability, and poor reproducibility of real-world robotic data collection, the framework adopts cloud-native technologies including elastic resource scheduling, containerized simulation, unified data management, and service-oriented system design, enabling efficient large-scale simulation for multi-model and multi-task workloads.
Built on a four-layer architecture, the framework provides standardized environment assets, automated task generation, trajectory collection, benchmark evaluation, and closed-loop data optimization. It further integrates representative systems including D-VLA, RL-VLA3, Sword, and Pre-VLA to support scalable simulation, dynamic scheduling, visual augmentation, and real-time data filtering.
We argue that cloud-native simulation infrastructure provides a unified foundation for data generation, model training, standardized evaluation, and real-world deployment, and will play a key role in the future development of embodied intelligence.

**arXiv ID:** 2606.27962
</details>

<details>
<summary><strong>PA-BiCoop: A Primary-Auxiliary Cooperative Framework for General Bimanual Manipulation</strong> - Bai Qicheng, Wang Ziru, Ma Teli, Dai Guang, Wang Jingdong, Wang Mengmeng - [[pdf]](https://arxiv.org/pdf/2606.28192)</summary>

**Abstract:** Bimanual manipulation is essential for advanced robotic systems because it offers higher efficiency and flexibility compared to single-arm configurations. However, existing approaches either lack inter-arm interaction or ignore the need for a dynamic division of labor, treating the arms as functionally equivalent. To address these limitations, this paper draws inspiration from human bimanual manipulation where one arm handles core operations and the other provides auxiliary support, and proposes PA-BiCoop, a new single-model bimanual cooperation framework with dynamic primary-auxiliary arm differentiation. PA-BiCoop categorizes robotic arms into primary and auxiliary arms with adaptively adjustable roles across task stages, employs two specialized decoders that share a global feature encoder: the primary decoder generates the primary arm's base-coordinate pose and core-task affordance heatmaps, and the auxiliary decoder outputs the auxiliary arm's relative pose in the primary arm's coordinate system. Moreover, we design a dynamic role assignment module to automatically map roles to left/right arms without manual pre-definition. This design facilitates inter-arm knowledge sharing and coordinated manipulation. Extensive experiments demonstrate that our PA-BiCoop achieves superior performance: it outperforms state-of-the-art baselines by 48% on average in RLBench2 simulation tasks and by over 50% on average in real world tasks, thereby verifying its effectiveness and advancement in bimanual manipulation.

**arXiv ID:** 2606.28192
</details>

<details>
<summary><strong>Relating Reinforcement Learning to Dynamic Programming-Based Planning</strong> - Filip V. Georgiev, Kalle G. Timperi, Başak Sakçak, Steven M. LaValle - [[pdf]](https://arxiv.org/pdf/2603.07844)</summary>

**Abstract:** This paper bridges some of the gap between optimal planning and reinforcement learning (RL), both of which share roots in dynamic programming applied to sequential decision making or optimal control. Whereas planning typically favors deterministic models, goal termination, and cost minimization, RL tends to favor stochastic models, infinite-horizon discounting, and reward maximization in addition to learning-related parameters such as the learning rate and greediness factor. A derandomized version of RL is developed, analyzed, and implemented to yield performance comparisons with value iteration and Dijkstra's algorithm using simple planning models. Next, mathematical analysis shows: 1) conditions under which cost minimization and reward maximization are equivalent, 2) conditions for equivalence of single-shot goal termination and infinite-horizon episodic learning, and 3) conditions under which discounting causes goal achievement to fail. The paper then advocates for defining and optimizing truecost, rather than inserting arbitrary parameters to guide operations. Performance studies are then extended to the stochastic case, using planning-oriented criteria and comparing value iteration to RL with learning rates and greediness factors.

**arXiv ID:** 2603.07844
</details>

<details>
<summary><strong>Deep Reinforcement Learning-Enhanced Event-Triggered Data-Driven Predictive Control for a 3D Cable-Driven Soft Robotic Arm</strong> - Cheng Ouyang, Moeen Ul Islam, Kaixiang Zhang, Zhaojian Li, Xiaobo Tan, Dong Chen - [[pdf]](https://arxiv.org/pdf/2606.26048)</summary>

**Abstract:** Soft robots are challenging to control due to their nonlinear and time-varying dynamics. Data-enabled predictive control (DeePC) offers a model-free alternative by directly leveraging measured input-output trajectories to construct a predictive controller. However, its receding-horizon formulation requires solving a constrained optimization problem at every sampling instant, which can be computationally demanding for real-time deployment on resource-limited robotic platforms. To address this limitation, we propose an adaptive reinforcement-learning-based event-triggered DeePC (RL-ET-DeePC) framework for soft robotic control. A model-free RL policy is trained to determine when to invoke the DeePC optimizer based on the current system state representation, thereby reducing unnecessary optimization calls while preserving closed-loop performance. Simulation results show that RL-ET-DeePC reduces optimization frequency by up to 66% compared to periodic DeePC, while maintaining comparable tracking accuracy. Hardware experiments on a three-dimensional cable-driven soft robotic arm demonstrate zero-shot transfer, achieving a 34% reduction in optimization frequency with tracking accuracy comparable to periodic DeePC and more consistent performance than a static threshold-based event-triggered baseline.

**arXiv ID:** 2606.26048
</details>

<details>
<summary><strong>DIVER: Reinforced Diffusion Breaks Imitation Bottlenecks in End-to-End Autonomous Driving</strong> - Ziying Song, Lin Liu, Hongyu Pan, Bencheng Liao, Mingzhe Guo, Lei Yang, Yongchang Zhang, Shaoqing Xu, Caiyan Jia, Yadan Luo - [[pdf]](https://arxiv.org/pdf/2507.04049)</summary>

**Abstract:** Most end-to-end autonomous driving methods rely on imitation learning from single expert demonstrations, often leading to conservative and homogeneous behaviors that limit generalization in complex real-world scenarios. In this work, we propose DIVER, an end-to-end driving framework that integrates reinforcement learning with diffusion-based generation to produce diverse and feasible trajectories. At the core of DIVER lies a reinforced diffusion-based generation mechanism. First, the model conditions on map elements and surrounding agents to generate multiple reference trajectories from a single ground-truth trajectory, alleviating the limitations of imitation learning that arise from relying solely on single expert demonstrations. Second, reinforcement learning is employed to guide the diffusion process, where reward-based supervision enforces safety and diversity constraints on the generated trajectories, thereby enhancing their practicality and generalization capability. Furthermore, to address the limitations of L2-based open-loop metrics in capturing trajectory diversity, we propose a novel Diversity metric to evaluate the diversity of multi-mode this http URL experiments on the closed-loop NAVSIM and Bench2Drive benchmarks, as well as the open-loop nuScenes dataset, demonstrate that DIVER significantly improves trajectory diversity, effectively addressing the mode collapse problem inherent in imitation learning.

**arXiv ID:** 2507.04049
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
