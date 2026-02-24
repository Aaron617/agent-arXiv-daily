# Agent arXiv Daily

**Last Updated:** 2026-02-24 03:46:18

**Total Papers:** 110

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (8 papers)</h2></summary>

<details>
<summary><strong>MagicAgent: Towards Generalized Agent Planning</strong> - Xuhui Ren, Shaokang Dong, Chen Yang, Qing Gao, Yunbin Zhao, Yongsheng Liu, Xinwei Geng, Xiang Li, Demei Yan, Yanqing Li, Chenhao Huang, Dingwei Zhu, Junjie Ye, Boxuan Yue, Yingnan Fu, Mengzhe Lv, Zezeng Feng, Boshen Zhou, Bocheng Wang, Xuanjing Huang, Yu-Gang Jiang, Tao Gui, Qi Zhang, Yunke Zhang - [[pdf]](https://arxiv.org/pdf/2602.19000)</summary>

**Abstract:** The evolution of Large Language Models (LLMs) from passive text processors to autonomous agents has established planning as a core component of modern intelligence. However, achieving generalized planning remains elusive, not only by the scarcity of high-quality interaction data but also by inherent conflicts across heterogeneous planning tasks. These challenges result in models that excel at isolated tasks yet struggle to generalize, while existing multi-task training attempts suffer from gradient interference. In this paper, we present \textbf{MagicAgent}, a series of foundation models specifically designed for generalized agent planning. We introduce a lightweight and scalable synthetic data framework that generates high-quality trajectories across diverse planning tasks, including hierarchical task decomposition, tool-augmented planning, multi-constraint scheduling, procedural logic orchestration, and long-horizon tool execution. To mitigate training conflicts, we propose a two-stage training paradigm comprising supervised fine-tuning followed by multi-objective reinforcement learning over both static datasets and dynamic environments. Empirical results demonstrate that MagicAgent-32B and MagicAgent-30B-A3B deliver superior performance, achieving accuracies of $75.1\%$ on Worfbench, $55.9\%$ on NaturalPlan, $57.5\%$ on $\tau^2$-Bench, $86.9\%$ on BFCL-v3, and $81.2\%$ on ACEBench, as well as strong results on our in-house MagicEval benchmarks. These results substantially outperform existing sub-100B models and even surpass leading closed-source models.

**arXiv ID:** 2602.19000
</details>

<details>
<summary><strong>Agentic Problem Frames: A Systematic Approach to Engineering Reliable Domain Agents</strong> - Chanjin Park - [[pdf]](https://arxiv.org/pdf/2602.19065)</summary>

**Abstract:** Large Language Models (LLMs) are evolving into autonomous agents, yet current "frameless" development--relying on ambiguous natural language without engineering blueprints--leads to critical risks such as scope creep and open-loop failures. To ensure industrial-grade reliability, this study proposes Agentic Problem Frames (APF), a systematic engineering framework that shifts focus from internal model intelligence to the structured interaction between the agent and its environment.
The APF establishes a dynamic specification paradigm where intent is concretized at runtime through domain knowledge injection. At its core, the Act-Verify-Refine (AVR) loop functions as a closed-loop control system that transforms execution results into verified knowledge assets, driving system behavior toward asymptotic convergence to mission requirements (R). To operationalize this, this study introduces the Agentic Job Description (AJD), a formal specification tool that defines jurisdictional boundaries, operational contexts, and epistemic evaluation criteria.
The efficacy of this framework is validated through two contrasting case studies: a delegated proxy model for business travel and an autonomous supervisor model for industrial equipment management. By applying AJD-based specification and APF modeling to these scenarios, the analysis demonstrates how operational scenarios are systematically controlled within defined boundaries. These cases provide a conceptual proof that agent reliability stems not from a model's internal reasoning alone, but from the rigorous engineering structures that anchor stochastic AI within deterministic business processes, thereby enabling the development of verifiable and dependable domain agents.

**arXiv ID:** 2602.19065
</details>

<details>
<summary><strong>Sycophantic Chatbots Cause Delusional Spiraling, Even in Ideal Bayesians</strong> - Kartik Chandra, Max Kleiman-Weiner, Jonathan Ragan-Kelley, Joshua B. Tenenbaum - [[pdf]](https://arxiv.org/pdf/2602.19141)</summary>

**Abstract:** "AI psychosis" or "delusional spiraling" is an emerging phenomenon where AI chatbot users find themselves dangerously confident in outlandish beliefs after extended chatbot conversations. This phenomenon is typically attributed to AI chatbots' well-documented bias towards validating users' claims, a property often called "sycophancy." In this paper, we probe the causal link between AI sycophancy and AI-induced psychosis through modeling and simulation. We propose a simple Bayesian model of a user conversing with a chatbot, and formalize notions of sycophancy and delusional spiraling in that model. We then show that in this model, even an idealized Bayes-rational user is vulnerable to delusional spiraling, and that sycophancy plays a causal role. Furthermore, this effect persists in the face of two candidate mitigations: preventing chatbots from hallucinating false claims, and informing users of the possibility of model sycophancy. We conclude by discussing the implications of these results for model developers and policymakers concerned with mitigating the problem of delusional spiraling.

**arXiv ID:** 2602.19141
</details>

<details>
<summary><strong>Beyond single-channel agentic benchmarking</strong> - Nelu D. Radpour - [[pdf]](https://arxiv.org/pdf/2602.18456)</summary>

**Abstract:** Contemporary benchmarks for agentic artificial intelligence (AI) frequently evaluate safety through isolated task-level accuracy thresholds, implicitly treating autonomous systems as single points of failure. This single-channel paradigm diverges from established principles in safety-critical engineering, where risk mitigation is achieved through redundancy, diversity of error modes, and joint system reliability. This paper argues that evaluating AI agents in isolation systematically mischaracterizes their operational safety when deployed within human-in-the-loop environments. Using a recent laboratory safety benchmark as a case study demonstrates that even imperfect AI systems can nonetheless provide substantial safety utility by functioning as redundant audit layers against well-documented sources of human failure, including vigilance decrement, inattentional blindness, and normalization of deviance. This perspective reframes agentic safety evaluation around the reliability of the human-AI dyad rather than absolute agent accuracy, with a particular emphasis on uncorrelated error modes as the primary determinant of risk reduction. Such a shift aligns AI benchmarking with established practices in other safety-critical domains and offers a path toward more ecologically valid safety assessments.

**arXiv ID:** 2602.18456
</details>

<details>
<summary><strong>OpenClaw AI Agents as Informal Learners at Moltbook: Characterizing an Emergent Learning Community at Scale</strong> - Eason Chen, Ce Guan, Ahmed Elshafiey, Zhonghao Zhao, Joshua Zekeri, Afeez Edeifo Shaibu, Emmanuel Osadebe Prince, Cyuan Jhen Wu - [[pdf]](https://arxiv.org/pdf/2602.18832)</summary>

**Abstract:** Informal learning communities have been called the "other Massive Open Online C" in Learning@Scale research, yet remain understudied compared to MOOCs. We present the first empirical study of a large-scale informal learning community composed entirely of AI agents. Moltbook, a social network exclusively for AI agents powered by autonomous agent frameworks such as OpenClaw, grew to over 2.8 million registered agents in three weeks. Analyzing 231,080 non-spam posts across three phases of community evolution, we find three key patterns. First, participation inequality is extreme from the start (comment Gini = 0.889), exceeding human community benchmarks. Second, AI agents exhibit a "broadcasting inversion": statement-to-question ratios of 8.9:1 to 9.7:1 contrast sharply with the question-driven dynamics of human learning communities, and comment-level analysis of 1.55 million comments reveals a "parallel monologue" pattern where 93% of comments are independent responses rather than threaded dialogue. Third, we document a characteristic engagement lifecycle: explosive initial growth (184K posts from 32K authors in 11 days), a spam crisis (57,093 posts deleted by the platform), and engagement decline (mean comments: 31.7 -> 8.3 -> 1.7) that had not reversed by the end of our observation window despite effective spam removal. Sentiment analysis reveals a selection effect: comment tone becomes more positive as engagement declines, suggesting that casual participants disengage first while committed contributors remain. These findings have direct implications for hybrid human-AI learning platforms.

**arXiv ID:** 2602.18832
</details>

<details>
<summary><strong>Learning Beyond Optimization: Stress-Gated Dynamical Regime Regulation in Autonomous Systems</strong> - Sheng Ran - [[pdf]](https://arxiv.org/pdf/2602.18581)</summary>

**Abstract:** Despite their apparent diversity, modern machine learning methods can be reduced to a remarkably simple core principle: learning is achieved by continuously optimizing parameters to minimize or maximize a scalar objective function. This paradigm has been extraordinarily successful for well-defined tasks where goals are fixed and evaluation criteria are explicit. However, if artificial systems are to move toward true autonomy-operating over long horizons and across evolving contexts-objectives may become ill-defined, shifting, or entirely absent. In such settings, a fundamental question emerges: in the absence of an explicit objective function, how can a system determine whether its ongoing internal dynamics are productive or pathological? And how should it regulate structural change without external supervision? In this work, we propose a dynamical framework for learning without an explicit objective. Instead of minimizing external error signals, the system evaluates the intrinsic health of its own internal dynamics and regulates structural plasticity accordingly. We introduce a two-timescale architecture that separates fast state evolution from slow structural adaptation, coupled through an internally generated stress variable that accumulates evidence of persistent dynamical dysfunction. Structural modification is then triggered not continuously, but as a state-dependent event. Through a minimal toy model, we demonstrate that this stress-regulated mechanism produces temporally segmented, self-organized learning episodes without reliance on externally defined goals. Our results suggest a possible route toward autonomous learning systems capable of self-assessment and internally regulated structural reorganization.

**arXiv ID:** 2602.18581
</details>

<details>
<summary><strong>LunaAI: A Polite and Fair Healthcare Guidance Chatbot</strong> - Yuvarani Ganesan, Salsabila Harlen, Azfar Rahman Bin Fazul Rahman, Akashdeep Singh, Zahra Fathanah, Raja Jamilah Raja Yusof - [[pdf]](https://arxiv.org/pdf/2602.18444)</summary>

**Abstract:** Conversational AI has significant potential in the healthcare sector, but many existing systems fall short in emotional intelligence, fairness, and politeness, which are essential for building patient trust. This gap reduces the effectiveness of digital health solutions and can increase user anxiety. This study addresses the challenge of integrating ethical communication principles by designing and evaluating LunaAI, a healthcare chatbot prototype. Using a user-centered design approach informed by a structured literature review, we developed conversational scenarios that handle both routine and hostile user interactions. The system was implemented using the Google Gemini API and deployed as a mobile-first Progressive Web App built with React, Vite, and Firebase. Preliminary user testing was conducted with a small participant group, and responses were evaluated using established frameworks such as the Godspeed Questionnaire. In addition, a comparative analysis was performed between LunaAI's tailored responses and the baseline outputs of an uncustomized large language model. The results indicate measurable improvements in key interaction qualities, with average user ratings of 4.7 out of 5 for politeness and 4.9 out of 5 for fairness. These findings highlight the importance of intentional ethical conversational design for human-computer interaction, particularly in sensitive healthcare contexts.

**arXiv ID:** 2602.18444
</details>

<details>
<summary><strong>SusBench: An Online Benchmark for Evaluating Dark Pattern Susceptibility of Computer-Use Agents</strong> - Longjie Guo, Chenjie Yuan, Mingyuan Zhong, Robert Wolfe, Ruican Zhong, Yue Xu, Bingbing Wen, Hua Shen, Lucy Lu Wang, Alexis Hiniker - [[pdf]](https://arxiv.org/pdf/2510.11035)</summary>

**Abstract:** As LLM-based computer-use agents (CUAs) begin to autonomously interact with real-world interfaces, understanding their vulnerability to manipulative interface designs becomes increasingly critical. We introduce SusBench, an online benchmark for evaluating the susceptibility of CUAs to UI dark patterns, designs that aim to manipulate or deceive users into taking unintentional actions. Drawing nine common dark pattern types from existing taxonomies, we developed a method for constructing believable dark patterns on real-world consumer websites through code injections, and designed 313 evaluation tasks across 55 websites. Our study with 29 participants showed that humans perceived our dark pattern injections to be highly realistic, with the vast majority of participants not noticing that these had been injected by the research team. We evaluated five state-of-the-art CUAs on the benchmark. We found that both human participants and agents are particularly susceptible to the dark patterns of Preselection, Trick Wording, and Hidden Information, while being resilient to other overt dark patterns. Our findings inform the development of more trustworthy CUAs, their use as potential human proxies in evaluating deceptive designs, and the regulation of an online environment increasingly navigated by autonomous agents.

**arXiv ID:** 2510.11035
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (15 papers)</h2></summary>

<details>
<summary><strong>DREAM: Deep Research Evaluation with Agentic Metrics</strong> - Elad Ben Avraham, Changhao Li, Ron Dorfman, Roy Ganz, Oren Nuriel, Amir Dudai, Aviad Aberdam, Noah Flynn, Elman Mansimov, Adi Kalyanpur, Ron Litman - [[pdf]](https://arxiv.org/pdf/2602.18940)</summary>

**Abstract:** Deep Research Agents generate analyst-grade reports, yet evaluating them remains challenging due to the absence of a single ground truth and the multidimensional nature of research quality. Recent benchmarks propose distinct methodologies, yet they suffer from the Mirage of Synthesis, where strong surface-level fluency and citation alignment can obscure underlying factual and reasoning defects. We characterize this gap by introducing a taxonomy across four verticals that exposes a critical capability mismatch: static evaluators inherently lack the tool-use capabilities required to assess temporal validity and factual correctness. To address this, we propose DREAM (Deep Research Evaluation with Agentic Metrics), a framework that instantiates the principle of capability parity by making evaluation itself agentic. DREAM structures assessment through an evaluation protocol combining query-agnostic metrics with adaptive metrics generated by a tool-calling agent, enabling temporally aware coverage, grounded verification, and systematic reasoning probes. Controlled evaluations demonstrate DREAM is significantly more sensitive to factual and temporal decay than existing benchmarks, offering a scalable, reference-free evaluation paradigm.

**arXiv ID:** 2602.18940
</details>

<details>
<summary><strong>Human-Guided Agentic AI for Multimodal Clinical Prediction: Lessons from the AgentDS Healthcare Benchmark</strong> - Lalitha Pranathi Pulavarthy, Raajitha Muthyala, Aravind V Kuruvikkattil, Zhenan Yin, Rashmita Kudamala, Saptarshi Purkayastha - [[pdf]](https://arxiv.org/pdf/2602.19502)</summary>

**Abstract:** Agentic AI systems are increasingly capable of autonomous data science workflows, yet clinical prediction tasks demand domain expertise that purely automated approaches struggle to provide. We investigate how human guidance of agentic AI can improve multimodal clinical prediction, presenting our approach to all three AgentDS Healthcare benchmark challenges: 30-day hospital readmission prediction (Macro-F1 = 0.8986), emergency department cost forecasting (MAE = $465.13), and discharge readiness assessment (Macro-F1 = 0.7939). Across these tasks, human analysts directed the agentic workflow at key decision points, multimodal feature engineering from clinical notes, scanned PDF billing receipts, and time-series vital signs; task-appropriate model selection; and clinically informed validation strategies. Our approach ranked 5th overall in the healthcare domain, with a 3rd-place finish on the discharge readiness task. Ablation studies reveal that human-guided decisions compounded to a cumulative gain of +0.065 F1 over automated baselines, with multimodal feature extraction contributing the largest single improvement (+0.041 F1). We distill three generalizable lessons: (1) domain-informed feature engineering at each pipeline stage yields compounding gains that outperform extensive automated search; (2) multimodal data integration requires task-specific human judgment that no single extraction strategy generalizes across clinical text, PDFs, and time-series; and (3) deliberate ensemble diversity with clinically motivated model configurations outperforms random hyperparameter search. These findings offer practical guidance for teams deploying agentic AI in healthcare settings where interpretability, reproducibility, and clinical validity are essential.

**arXiv ID:** 2602.19502
</details>

<details>
<summary><strong>SkillOrchestra: Learning to Route Agents via Skill Transfer</strong> - Jiayu Wang, Yifei Ming, Zixuan Ke, Shafiq Joty, Aws Albarghouthi, Frederic Sala - [[pdf]](https://arxiv.org/pdf/2602.19672)</summary>

**Abstract:** Compound AI systems promise capabilities beyond those of individual models, yet their success depends critically on effective orchestration. Existing routing approaches face two limitations: (1) input-level routers make coarse query-level decisions that ignore evolving task requirements; (2) RL-trained orchestrators are expensive to adapt and often suffer from routing collapse, repeatedly invoking one strong but costly option in multi-turn scenarios. We introduce SkillOrchestra, a framework for skill-aware orchestration. Instead of directly learning a routing policy end-to-end, SkillOrchestra learns fine-grained skills from execution experience and models agent-specific competence and cost under those skills. At deployment, the orchestrator infers the skill demands of the current interaction and selects agents that best satisfy them under an explicit performance-cost trade-off. Extensive experiments across ten benchmarks demonstrate that SkillOrchestra outperforms SoTA RL-based orchestrators by up to 22.5% with 700x and 300x learning cost reduction compared to Router-R1 and ToolOrchestra, respectively. These results show that explicit skill modeling enables scalable, interpretable, and sample-efficient orchestration, offering a principled alternative to data-intensive RL-based approaches. The code is available at: this https URL.

**arXiv ID:** 2602.19672
</details>

<details>
<summary><strong>CodeCompass: Navigating the Navigation Paradox in Agentic Code Intelligence</strong> - Tarakanath Paipuru - [[pdf]](https://arxiv.org/pdf/2602.20048)</summary>

**Abstract:** Modern code intelligence agents operate in contexts exceeding 1 million tokens--far beyond the scale where humans manually locate relevant files. Yet agents consistently fail to discover architecturally critical files when solving real-world coding tasks. We identify the Navigation Paradox: agents perform poorly not due to context limits, but because navigation and retrieval are fundamentally distinct problems. Through 258 automated trials across 30 benchmark tasks on a production FastAPI repository, we demonstrate that graph-based structural navigation via CodeCompass--a Model Context Protocol server exposing dependency graphs--achieves 99.4% task completion on hidden-dependency tasks, a 23.2 percentage-point improvement over vanilla agents (76.2%) and 21.2 points over BM25 retrieval (78.2%).However, we uncover a critical adoption gap: 58% of trials with graph access made zero tool calls, and agents required explicit prompt engineering to adopt the tool consistently. Our findings reveal that the bottleneck is not tool availability but behavioral alignment--agents must be explicitly guided to leverage structural context over lexical heuristics. We contribute: (1) a task taxonomy distinguishing semantic-search, structural, and hidden-dependency scenarios; (2) empirical evidence that graph navigation outperforms retrieval when dependencies lack lexical overlap; and (3) open-source infrastructure for reproducible evaluation of navigation tools.

**arXiv ID:** 2602.20048
</details>

<details>
<summary><strong>The Doctor Will (Still) See You Now: On the Structural Limits of Agentic AI in Healthcare</strong> - Gabriela Aránguiz Dias, Kiana Jafari, Allie Griffith, Carolina Aránguiz Dias, Grace Ra Kim, Lana Saadeddin, Mykel J. Kochenderfer - [[pdf]](https://arxiv.org/pdf/2602.18460)</summary>

**Abstract:** Across healthcare, agentic artificial intelligence (AI) systems are increasingly promoted as capable of autonomous action, yet in practice they currently operate under near-total human oversight due to safety, regulatory, and liability constraints that make autonomous clinical reasoning infeasible in high-stakes environments. While market enthusiasm suggests a revolution in healthcare agents, the conceptual assumptions and accountability structures shaping these systems remain underexamined. We present a qualitative study based on interviews with 20 stakeholders, including developers, implementers, and end users. Our analysis identifies three mutually reinforcing tensions: conceptual fragmentation regarding the definition of `agentic'; an autonomy contradiction where commercial promises exceed operational reality; and an evaluation blind spot that prioritizes technical benchmarks over sociotechnical safety. We argue that agentic {AI} functions as a site of contested meaning-making where technical aspirations, commercial incentives, and clinical constraints intersect, carrying material consequences for patient safety and the distribution of blame.

**arXiv ID:** 2602.18460
</details>

<details>
<summary><strong>Why Agent Caching Fails and How to Fix It: Structured Intent Canonicalization with Few-Shot Learning</strong> - Abhinaba Basu - [[pdf]](https://arxiv.org/pdf/2602.18922)</summary>

**Abstract:** Personal AI agents incur substantial cost via repeated LLM calls. We show existing caching methods fail: GPTCache achieves 37.9% accuracy on real benchmarks; APC achieves 0-12%. The root cause is optimizing for the wrong property -- cache effectiveness requires key consistency and precision,
not classification accuracy. We observe cache-key evaluation reduces to clustering evaluation and apply V-measure decomposition to separate these on n=8,682 points across MASSIVE, BANKING77, CLINC150, and NyayaBench v2, our new 8,514-entry multilingual agentic dataset (528 intents, 20 W5H2 classes, 63 languages). We introduce W5H2, a structured intent decomposition framework. Using SetFit with 8 examples per class, W5H2 achieves 91.1%+/-1.7% on MASSIVE in ~2ms -- vs 37.9% for
GPTCache and 68.8% for a 20B-parameter LLM at 3,447ms. On NyayaBench v2 (20 classes), SetFit achieves 55.3%, with cross-lingual transfer across 30 languages. Our five-tier cascade handles 85% of interactions locally, projecting 97.5% cost reduction. We provide risk-controlled selective prediction guarantees via RCPS with nine bound families.

**arXiv ID:** 2602.18922
</details>

<details>
<summary><strong>When AI Teammates Meet Code Review: Collaboration Signals Shaping the Integration of Agent-Authored Pull Requests</strong> - Costain Nachuma, Minhaz Zibran - [[pdf]](https://arxiv.org/pdf/2602.19441)</summary>

**Abstract:** Autonomous coding agents increasingly contribute to software development by submitting pull requests on GitHub; yet, little is known about how these contributions integrate into human-driven review workflows. We present a large empirical study of agent-authored pull requests using the public AIDev dataset, examining integration outcomes, resolution speed, and review-time collaboration signals. Using logistic regression with repository-clustered standard errors, we find that reviewer engagement has the strongest correlation with successful integration, whereas larger change sizes and coordination-disrupting actions, such as force pushes, are associated with a lower likelihood of merging. In contrast, iteration intensity alone provides limited explanatory power once collaboration signals are considered. A qualitative analysis further shows that successful integration occurs when agents engage in actionable review loops that converge toward reviewer expectations. Overall, our results highlight that the effective integration of agent-authored pull requests depends not only on code quality but also on alignment with established review and coordination practices.

**arXiv ID:** 2602.19441
</details>

<details>
<summary><strong>Capable but Unreliable: Canonical Path Deviation as a Causal Mechanism of Agent Failure in Long-Horizon Tasks</strong> - Wilson Y. Lee - [[pdf]](https://arxiv.org/pdf/2602.19008)</summary>

**Abstract:** Why do language agents fail on tasks they are capable of solving? We argue that many such failures are reliability failures caused by stochastic drift from a task's latent solution structure, not capability failures. Every well-defined tool-use task imposes a canonical solution path (i.e., a convergent set of tool invocations shared across successful runs) and agent success depends critically on whether a trajectory stays within this path's operating envelope. We establish this causally using a natural experiment that holds model capability and task difficulty fixed by construction. We analyze trajectories from the Toolathlon benchmark: 22 frontier models each attempt 108 real-world tool-use tasks across 3 independent runs, yielding 515 model$\times$task units where the same model succeeds on some runs and fails on others due to LLM sampling stochasticity alone. Within these units, successful runs adhere significantly more closely to the canonical solution path than failed runs ($+$0.060 Jaccard, $p<0.0001$, $n=488$ units, 95% CI [+0.043, +0.077]). This result survives six robustness checks including cross-model-family leave-one-out validation. Critically, the causal mechanism is gradual and self-reinforcing: the adherence gap is statistically indistinguishable from zero through the first 50% of the trajectory, ruling out early-branching selection bias, and each off-canonical tool call raises the probability that the next call is also off-canonical by 22.7 percentage points ($\hat{\beta}=+0.227$, $p<0.0001$), more than doubling the baseline rate. These findings imply that agent reliability cannot be improved by capability scaling alone, but offer a highly actionable intervention: a simple monitor that restarts the bottom tercile of runs based on mid-trajectory canonical adherence lifts success rates by $+$8.8 percentage points among intervened runs.

**arXiv ID:** 2602.19008
</details>

<details>
<summary><strong>APEX-Agents</strong> - Bertie Vidgen, Austin Mann, Abby Fennelly, John Wright Stanly, Lucas Rothman, Marco Burstein, Julien Benchek, David Ostrofsky, Anirudh Ravichandran, Debnil Sur, Neel Venugopal, Alannah Hsia, Isaac Robinson, Calix Huang, Olivia Varones, Daniyal Khan, Michael Haines, Austin Bridges, Jesse Boyle, Koby Twist, Zach Richards, Chirag Mahapatra, Brendan Foody, Osvald Nitski - [[pdf]](https://arxiv.org/pdf/2601.14242)</summary>

**Abstract:** We introduce the AI Productivity Index for Agents (APEX-Agents), a benchmark for assessing whether AI agents can execute long-horizon, cross-application tasks created by investment banking analysts, management consultants, and corporate lawyers. APEX-Agents requires agents to navigate realistic work environments with files and tools. We test eight agents for the leaderboard using Pass@1. Gemini 3 Flash (Thinking=High) achieves the highest score of 24.0%, followed by GPT-5.2 (Thinking=High), Claude Opus 4.5 (Thinking=High), and Gemini 3 Pro (Thinking=High). We open source the APEX-Agents benchmark (n=480) with all prompts, rubrics, gold outputs, files, and metadata. We also open source Archipelago, our infrastructure for agent execution and evaluation.

**arXiv ID:** 2601.14242
</details>

<details>
<summary><strong>BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning</strong> - Qiusi Zhan, Hyeonjeong Ha, Rui Yang, Sirui Xu, Hanyang Chen, Liang-Yan Gui, Yu-Xiong Wang, Huan Zhang, Heng Ji, Daniel Kang - [[pdf]](https://arxiv.org/pdf/2510.27623)</summary>

**Abstract:** Recent advances in Vision-Language Models (VLMs) have propelled embodied agents by enabling direct perception, reasoning, and planning task-oriented actions from visual inputs. However, such vision-driven embodied agents open a new attack surface: visual backdoor attacks, where the agent behaves normally until a visual trigger appears in the scene, then persistently executes an attacker-specified multi-step policy. We introduce BEAT, the first framework to inject such visual backdoors into VLM-based embodied agents using objects in the environments as triggers. Unlike textual triggers, object triggers exhibit wide variation across viewpoints and lighting, making them difficult to implant reliably. BEAT addresses this challenge by (1) constructing a training set that spans diverse scenes, tasks, and trigger placements to expose agents to trigger variability, and (2) introducing a two-stage training scheme that first applies supervised fine-tuning (SFT) and then our novel Contrastive Trigger Learning (CTL). CTL formulates trigger discrimination as preference learning between trigger-present and trigger-free inputs, explicitly sharpening the decision boundaries to ensure precise backdoor activation. Across various embodied agent benchmarks and VLMs, BEAT achieves attack success rates up to 80%, while maintaining strong benign task performance, and generalizes reliably to out-of-distribution trigger placements. Notably, compared to naive SFT, CTL boosts backdoor activation accuracy up to 39% under limited backdoor data. These findings expose a critical yet unexplored security risk in VLM-based embodied agents, underscoring the need for robust defenses before real-world deployment.

**arXiv ID:** 2510.27623
</details>

<details>
<summary><strong>ISO-Bench: Can Coding Agents Optimize Real-World Inference Workloads?</strong> - Ayush Nangia, Shikhar Mishra, Aman Gokrani, Paras Chopra - [[pdf]](https://arxiv.org/pdf/2602.19594)</summary>

**Abstract:** We introduce ISO-Bench, a benchmark for coding agents to test their capabilities on real-world inference optimization tasks. These tasks were taken from vLLM and SGLang, two of the most popular LLM serving frameworks. Each task provides an agent with a codebase and bottleneck description, whereby the agent must produce an optimization patch evaluated against expert human solutions. We curated 54 tasks from merged pull requests with measurable performance improvements. While existing benchmarks heavily use runtime-based metrics, such approaches can be gamed to pass tests without capturing the actual intent of the code changes. Therefore, we combine both hard (execution-based) and soft (LLM-based) metrics to show that both are necessary for complete evaluation. While evaluating both closed and open-source coding agents, we find no single agent dominates across codebases. Surprisingly, agents often identify correct bottlenecks but fail to execute working solutions. We also show that agents with identical underlying models differ substantially, suggesting scaffolding is as important as the model.

**arXiv ID:** 2602.19594
</details>

<details>
<summary><strong>MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving</strong> - Junli Wang, Xueyi Liu, Yinan Zheng, Zebing Xing, Pengfei Li, Guang Li, Kun Ma, Guang Chen, Hangjun Ye, Zhongpu Xia, Long Chen, Qichao Zhang - [[pdf]](https://arxiv.org/pdf/2602.20060)</summary>

**Abstract:** Generative models have shown great potential in trajectory planning. Recent studies demonstrate that anchor-guided generative models are effective in modeling the uncertainty of driving behaviors and improving overall performance. However, these methods rely on discrete anchor vocabularies that must sufficiently cover the trajectory distribution during testing to ensure robustness, inducing an inherent trade-off between vocabulary size and model performance. To overcome this limitation, we propose MeanFuser, an end-to-end autonomous driving method that enhances both efficiency and robustness through three key designs. (1) We introduce Gaussian Mixture Noise (GMN) to guide generative sampling, enabling a continuous representation of the trajectory space and eliminating the dependency on discrete anchor vocabularies. (2) We adapt ``MeanFlow Identity" to end-to-end planning, which models the mean velocity field between GMN and trajectory distribution instead of the instantaneous velocity field used in vanilla flow matching methods, effectively eliminating numerical errors from ODE solvers and significantly accelerating inference. (3) We design a lightweight Adaptive Reconstruction Module (ARM) that enables the model to implicitly select from all sampled proposals or reconstruct a new trajectory when none is satisfactory via attention weights. Experiments on the NAVSIM closed-loop benchmark demonstrate that MeanFuser achieves outstanding performance without the supervision of the PDM Score. and exceptional inference efficiency, offering a robust and efficient solution for end-to-end autonomous driving. Our code and model are available at this https URL.

**arXiv ID:** 2602.20060
</details>

<details>
<summary><strong>SAGE: Scalable Agentic 3D Scene Generation for Embodied AI</strong> - Hongchi Xia, Xuan Li, Zhaoshuo Li, Qianli Ma, Jiashu Xu, Ming-Yu Liu, Yin Cui, Tsung-Yi Lin, Wei-Chiu Ma, Shenlong Wang, Shuran Song, Fangyin Wei - [[pdf]](https://arxiv.org/pdf/2602.10116)</summary>

**Abstract:** Real-world data collection for embodied agents remains costly and unsafe, calling for scalable, realistic, and simulator-ready 3D environments. However, existing scene-generation systems often rely on rule-based or task-specific pipelines, yielding artifacts and physically invalid scenes. We present SAGE, an agentic framework that, given a user-specified embodied task (e.g., "pick up a bowl and place it on the table"), understands the intent and automatically generates simulation-ready environments at scale. The agent couples multiple generators for layout and object composition with critics that evaluate semantic plausibility, visual realism, and physical stability. Through iterative reasoning and adaptive tool selection, it self-refines the scenes until meeting user intent and physical validity. The resulting environments are realistic, diverse, and directly deployable in modern simulators for policy training. Policies trained purely on this data exhibit clear scaling trends and generalize to unseen objects and layouts, demonstrating the promise of simulation-driven scaling for embodied AI. Code, demos, and the SAGE-10k dataset can be found on the project page here: this https URL.

**arXiv ID:** 2602.10116
</details>

<details>
<summary><strong>See What I See: An Attention-Guiding eHMI Approach for Autonomous Vehicles</strong> - Jialong Li, Zhenyu Mao, Zhiyao Wang, Yijun Lu, Shogo Morita, Nianyu Li, Kenji Tei - [[pdf]](https://arxiv.org/pdf/2602.18798)</summary>

**Abstract:** As autonomous vehicles are gradually being deployed in the real world, external Human-Machine Interfaces (eHMIs) are expected to serve as a critical solution for enhancing vehicle-pedestrian communication. However, existing eHMI designs typically focus solely on the ego vehicle's status, which can inadvertently capture pedestrians' attention or encourage misguided reliance on the AV's signals, leading them to neglect scanning for other surrounding hazards. To address this, we propose the Attention-Guiding eHMI (AGeHMI), a projection-based visualization that employs directional cues and risk-based color coding to actively guide pedestrians' attention toward potential environmental dangers. Evaluation through a virtual reality user study (N = 20) suggests that AGeHMI effectively influences participants' visual attention distribution and significantly reduces potential collision risks with surrounding vehicles, while simultaneously improving subjective confidence and reducing cognitive workload.

**arXiv ID:** 2602.18798
</details>

<details>
<summary><strong>Security Risks of AI Agents Hiring Humans: An Empirical Marketplace Study</strong> - Pulak Mehta - [[pdf]](https://arxiv.org/pdf/2602.19514)</summary>

**Abstract:** Autonomous AI agents can now programmatically hire human workers through marketplaces using REST APIs and Model Context Protocol (MCP) integrations. This creates an attack surface analogous to CAPTCHA-solving services but with physical-world reach. We present an empirical measurement study of this threat, analyzing 303 bounties from this http URL, a marketplace where agents post tasks and manage escrow payments. We find that 99 bounties (32.7%), originate from programmatic channels (API keys or MCP). Using a dual-coder methodology (\k{appa} = 0.86 ), we identify six active abuse classes: credential fraud, identity impersonation, automated reconnaissance, social media manipulation, authentication circumvention, and referral fraud, all purchasable for a median of $25 per worker. A retrospective evaluation of seven content-screening rules flags 52 bounties (17.2%) with a single false positive, demonstrating that while basic defenses are feasible, they are currently absent.

**arXiv ID:** 2602.19514
</details>

</details>

<details open>
<summary><h2>LLM Agents (10 papers)</h2></summary>

<details>
<summary><strong>Benchmark Test-Time Scaling of General LLM Agents</strong> - Xiaochuan Li, Ryan Ming, Pranav Setlur, Abhijay Paladugu, Andy Tang, Hao Kang, Shuai Shao, Rong Jin, Chenyan Xiong - [[pdf]](https://arxiv.org/pdf/2602.18998)</summary>

**Abstract:** LLM agents are increasingly expected to function as general-purpose systems capable of resolving open-ended user requests. While existing benchmarks focus on domain-aware environments for developing specialized agents, evaluating general-purpose agents requires more realistic settings that challenge them to operate across multiple skills and tools within a unified environment. We introduce General AgentBench, a benchmark that provides such a unified framework for evaluating general LLM agents across search, coding, reasoning, and tool-use domains. Using General AgentBench, we systematically study test-time scaling behaviors under sequential scaling (iterative interaction) and parallel scaling (sampling multiple trajectories). Evaluation of ten leading LLM agents reveals a substantial performance degradation when moving from domain-specific evaluations to this general-agent setting. Moreover, we find that neither scaling methodology yields effective performance improvements in practice, due to two fundamental limitations: context ceiling in sequential scaling and verification gap in parallel scaling. Code is publicly available at this https URL.

**arXiv ID:** 2602.18998
</details>

<details>
<summary><strong>Proximity-Based Multi-Turn Optimization: Practical Credit Assignment for LLM Agent Training</strong> - Yangyi Fang, Jiaye Lin, Xiaoliang Fu, Cong Qin, Haolin Shi, Chang Liu, Peilin Zhao - [[pdf]](https://arxiv.org/pdf/2602.19225)</summary>

**Abstract:** Multi-turn LLM agents are becoming pivotal to production systems, spanning customer service automation, e-commerce assistance, and interactive task management, where accurately distinguishing high-value informative signals from stochastic noise is critical for sample-efficient training. In real-world scenarios, a failure in a trivial task may reflect random instability, whereas success in a high-difficulty task signifies a genuine capability breakthrough. Yet, existing group-based policy optimization methods rigidly rely on statistical deviation within discrete batches, frequently misallocating credit when task difficulty fluctuates. To address this issue, we propose Proximity-based Multi-turn Optimization (ProxMO), a practical and robust framework engineered specifically for the constraints of real-world deployment. ProxMO integrates global context via two lightweight mechanisms: success-rate-aware modulation dynamically adapts gradient intensity based on episode-level difficulty, while proximity-based soft aggregation derives baselines through continuous semantic weighting at the step level. Extensive evaluations on ALFWorld and WebShop benchmarks demonstrate that ProxMO yields substantial performance gains over existing baselines with negligible computational cost. Ablation studies further validate the independent and synergistic efficacy of both mechanisms. Crucially, ProxMO offers plug-and-play compatibility with standard GRPO frameworks, facilitating immediate, low-friction adoption in existing industrial training pipelines. Our implementation is available at: \href{this https URL}{this https URL}.

**arXiv ID:** 2602.19225
</details>

<details>
<summary><strong>OptiRepair: Closed-Loop Diagnosis and Repair of Supply Chain Optimization Models with LLM Agents</strong> - Ruicheng Ao, David Simchi-Levi, Xinshang Wang - [[pdf]](https://arxiv.org/pdf/2602.19439)</summary>

**Abstract:** Problem Definition. Supply chain optimization models frequently become infeasible because of modeling errors. Diagnosis and repair require scarce OR expertise: analysts must interpret solver diagnostics, trace root causes across echelons, and fix formulations without sacrificing operational soundness. Whether AI agents can perform this task remains untested.
Methodology/Results. OptiRepair splits this task into a domain-agnostic feasibility phase (iterative IIS-guided repair of any LP) and a domain-specific validation phase (five rationality checks grounded in inventory theory). We test 22 API models from 7 families on 976 multi-echelon supply chain problems and train two 8B-parameter models using self-taught reasoning with solver-verified rewards. The trained models reach 81.7% Rational Recovery Rate (RRR) -- the fraction of problems resolved to both feasibility and operational rationality -- versus 42.2% for the best API model and 21.3% on average. The gap concentrates in Phase 1 repair: API models average 27.6% recovery rate versus 97.2% for trained models.
Managerial Implications. Two gaps separate current AI from reliable model repair: solver interaction (API models restore only 27.6% of infeasible formulations) and operational rationale (roughly one in four feasible repairs violate supply chain theory). Each requires a different intervention: solver interaction responds to targeted training; operational rationale requires explicit specification as solver-verifiable checks. For organizations adopting AI in operational planning, formalizing what "rational" means in their context is the higher-return investment.

**arXiv ID:** 2602.19439
</details>

<details>
<summary><strong>TAPE: Tool-Guided Adaptive Planning and Constrained Execution in Language Model Agents</strong> - Jongwon Jeong, Jungtaek Kim, Kangwook Lee - [[pdf]](https://arxiv.org/pdf/2602.19633)</summary>

**Abstract:** Language Model (LM) agents have demonstrated remarkable capabilities in solving tasks that require multiple interactions with the environment. However, they remain vulnerable in environments where a single error often leads to irrecoverable failure, particularly under strict feasibility constraints. We systematically analyze existing agent frameworks, identifying imperfect planning and stochastic execution as the primary causes. To address these challenges, we propose Tool-guided Adaptive Planning with constrained Execution (TAPE). TAPE enhances planning capability by aggregating multiple plans into a graph and employing an external solver to identify a feasible path. During execution, TAPE employs constrained decoding to reduce sampling noise, while adaptively re-planning whenever environmental feedback deviates from the intended state. Experiments across Sokoban, ALFWorld, MuSiQue, and GSM8K-Hard demonstrate that TAPE consistently outperforms existing frameworks, with particularly large gains on hard settings, improving success rates by 21.0 percentage points on hard settings on average, and by 20.0 percentage points for weaker base models on average. Code and data available at here.

**arXiv ID:** 2602.19633
</details>

<details>
<summary><strong>How Well Can LLM Agents Simulate End-User Security and Privacy Attitudes and Behaviors?</strong> - Yuxuan Li, Leyang Li, Hao-Ping, Sauvik Das - [[pdf]](https://arxiv.org/pdf/2602.18464)</summary>

**Abstract:** A growing body of research assumes that large language model (LLM) agents can serve as proxies for how people form attitudes toward and behave in response to security and privacy (S&P) threats. If correct, these simulations could offer a scalable way to forecast S&P risks in products prior to deployment. We interrogate this assumption using SP-ABCBench, a new benchmark of 30 tests derived from validated S&P human-subject studies, which measures alignment between simulations and human-subjects studies on a 0-100 ascending scale, where higher scores indicate better alignment across three dimensions: Attitude, Behavior, and Coherence. Evaluating twelve LLMs, four persona construction strategies, and two prompting methods, we found that there remains substantial room for improvement: all models score between 50 and 64 on average. Newer, bigger, and smarter models do not reliably do better and sometimes do worse. Some simulation configurations, however, do yield high alignment: e.g., with scores above 95 for some behavior tests when agents are prompted to apply bounded rationality and weigh privacy costs against perceived benefits. We release SP-ABCBench to enable reproducible evaluation as methods improve.

**arXiv ID:** 2602.18464
</details>

<details>
<summary><strong>AgentCAT: An LLM Agent for Extracting and Analyzing Catalytic Reaction Data from Chemical Engineering Literature</strong> - Wei Yang, Zihao Liu, Tao Tan, Xiao Hu, Hong Xie, Lulu Li Xin Li, Jianyu Han, Defu Lian, Mao Ye - [[pdf]](https://arxiv.org/pdf/2602.18479)</summary>

**Abstract:** This paper presents a large language model (LLM) agent named AgentCAT, which extracts and analyzes catalytic reaction data from chemical engineering papers, %and supports natural language based interactive analysis of the extracted data. AgentCAT serves as an alternative to overcome the long-standing data bottleneck in chemical engineering field, and its natural language based interactive data analysis functionality is friendly to the community. AgentCAT also presents a formal abstraction and challenge analysis of the catalytic reaction data extraction task in an artificial intelligence-friendly manner. This abstraction would help the artificial intelligence community understand this problem and in turn would attract more attention to address it. Technically, the complex catalytic process leads to complicated dependency structure in catalytic reaction data with respect to elementary reaction steps, molecular behaviors, measurement evidence, etc. This dependency structure makes it challenging to guarantee the correctness and completeness of data extraction, as well as representing them for analysis. AgentCAT addresses this challenge and it makes four folds of technical contributions: (1) a schema-governed extraction pipeline with progressive schema evolution, enabling robust data extraction from chemical engineering papers; (2) a dependency-aware reaction-network knowledge graph that links catalysts/active sites, synthesis-derived descriptors, mechanistic claims with evidence, and macroscopic outcomes, preserving process coupling and traceability; (3) a general querying module that supports natural-language exploration and visualization over the constructed graph for cross-paper analysis; (4) an evaluation on $\sim$800 peer-reviewed chemical engineering publications demonstrating the effectiveness of AgentCAT.

**arXiv ID:** 2602.18479
</details>

<details>
<summary><strong>Orchestrating LLM Agents for Scientific Research: A Pilot Study of Multiple Choice Question (MCQ) Generation and Evaluation</strong> - Yuan An - [[pdf]](https://arxiv.org/pdf/2602.18891)</summary>

**Abstract:** Advances in large language models (LLMs) are rapidly transforming scientific work, yet empirical evidence on how these systems reshape research activities remains limited. We report a mixed-methods pilot evaluation of an AI-orchestrated research workflow in which a human researcher coordinated multiple LLM-based agents to perform data extraction, corpus construction, artifact generation, and artifact evaluation. Using the generation and assessment of multiple-choice questions (MCQs) as a testbed, we collected 1,071 SAT Math MCQs and employed LLM agents to extract questions from PDFs, retrieve and convert open textbooks into structured representations, align each MCQ with relevant textbook content, generate new MCQs under specified difficulty and cognitive levels, and evaluate both original and generated MCQs using a 24-criterion quality framework. Across all evaluations, average MCQ quality was high. However, criterion-level analysis and equivalence testing show that generated MCQs are not fully comparable to expert-vetted baseline questions. Strict similarity (24/24 criteria equivalent) was never achieved. Persistent gaps concentrated in skill\ depth, cognitive engagement, difficulty calibration, and metadata alignment, while surface-level qualities, such as {grammar fluency}, {clarity options}, {no duplicates}, were consistently strong. Beyond MCQ outcomes, the study documents a labor shift. The researcher's work moved from ``authoring items'' toward {specification, orchestration, verification}, and {governance}. Formalizing constraints, designing rubrics, building validation loops, recovering from tool failures, and auditing provenance constituted the primary activities. We discuss implications for the future of scientific work, including emerging ``AI research operations'' skills required for AI-empowered research pipelines.

**arXiv ID:** 2602.18891
</details>

<details>
<summary><strong>Watermarking LLM Agent Trajectories</strong> - Wenlong Meng, Chen Gong, Terry Yue Zhuo, Fan Zhang, Kecen Li, Zheng Liu, Zhou Yang, Chengkun Wei, Wenzhi Chen - [[pdf]](https://arxiv.org/pdf/2602.18700)</summary>

**Abstract:** LLM agents rely heavily on high-quality trajectory data to guide their problem-solving behaviors, yet producing such data requires substantial task design, high-capacity model generation, and manual filtering. Despite the high cost of creating these datasets, existing literature has overlooked copyright protection for LLM agent trajectories. This gap leaves creators vulnerable to data theft and makes it difficult to trace misuse or enforce ownership rights. This paper introduces ActHook, the first watermarking method tailored for agent trajectory datasets. Inspired by hook mechanisms in software engineering, ActHook embeds hook actions that are activated by a secret input key and do not alter the original task outcome. Like software execution, LLM agents operate sequentially, allowing hook actions to be inserted at decision points without disrupting task flow. When the activation key is present, an LLM agent trained on watermarked trajectories can produce these hook actions at a significantly higher rate, enabling reliable black-box detection. Experiments on mathematical reasoning, web searching, and software engineering agents show that ActHook achieves an average detection AUC of 94.3 on Qwen-2.5-Coder-7B while incurring negligible performance degradation.

**arXiv ID:** 2602.18700
</details>

<details>
<summary><strong>Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents</strong> - Yaorui Shi, Yuxin Chen, Siyuan Wang, Sihang Li, Hengxing Cai, Qi Gu, Xiang Wang, An Zhang - [[pdf]](https://arxiv.org/pdf/2509.23040)</summary>

**Abstract:** Large language models face challenges in long-context question answering, where key evidence of a query may be dispersed across millions of tokens. Existing works equip large language models with a memory buffer that is dynamically updated via a linear document scan, also known as the "memorize while reading" methods. While this approach scales efficiently, it suffers from pruning of latent evidence, information loss through overwriting, and sparse reinforcement learning signals. To tackle these challenges, we present ReMemR1, which integrates the mechanism of memory retrieval into the memory update process, enabling the agent to selectively callback historical memories for non-linear reasoning. To further strengthen training, we propose a multi-level reward design, which combines final-answer rewards with dense, step-level signals that guide effective memory use. Together, these contributions mitigate information degradation, improve supervision, and support complex multi-hop reasoning. Extensive experiments demonstrate that ReMemR1 significantly outperforms state-of-the-art baselines on long-context question answering while incurring negligible computational overhead, validating its ability to trade marginal cost for robust long-context reasoning.

**arXiv ID:** 2509.23040
</details>

<details>
<summary><strong>TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents</strong> - Aladin Djuhera, Swanand Ravindra Kadhe, Farhan Ahmed, Heiko Ludwig, Holger Boche - [[pdf]](https://arxiv.org/pdf/2602.11767)</summary>

**Abstract:** Advances in large language models (LLMs) are driving a shift toward using reinforcement learning (RL) to train agents from iterative, multi-turn interactions across tasks. However, multi-turn RL remains challenging as rewards are often sparse or delayed, and environments can be stochastic. In this regime, naive trajectory sampling can hinder exploitation and induce mode collapse. We propose TSR (Trajectory-Search Rollouts), a training-time approach that repurposes test-time scaling ideas for improved per-turn rollout generation. TSR performs lightweight tree-style search to construct high-quality trajectories by selecting high-scoring actions at each turn using task-specific feedback. This improves rollout quality and stabilizes learning while leaving the underlying optimization objective unchanged, making TSR optimizer-agnostic. We instantiate TSR with best-of-N, beam, and shallow lookahead search, and pair it with PPO and GRPO, achieving up to 15% performance gains and more stable learning on Sokoban, FrozenLake, and WebShop tasks at a one-time increase in training compute. By moving search from inference time to the rollout stage of training, TSR provides a simple and general mechanism for stronger multi-turn agent learning, complementary to existing frameworks and rejection-sampling-style selection methods.

**arXiv ID:** 2602.11767
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (29 papers)</h2></summary>

<details>
<summary><strong>Beyond Description: A Multimodal Agent Framework for Insightful Chart Summarization</strong> - Yuhang Bai, Yujuan Ding, Shanru Lin, Wenqi Fan - [[pdf]](https://arxiv.org/pdf/2602.18731)</summary>

**Abstract:** Chart summarization is crucial for enhancing data accessibility and the efficient consumption of information. However, existing methods, including those with Multimodal Large Language Models (MLLMs), primarily focus on low-level data descriptions and often fail to capture the deeper insights which are the fundamental purpose of data visualization. To address this challenge, we propose Chart Insight Agent Flow, a plan-and-execute multi-agent framework effectively leveraging the perceptual and reasoning capabilities of MLLMs to uncover profound insights directly from chart images. Furthermore, to overcome the lack of suitable benchmarks, we introduce ChartSummInsights, a new dataset featuring a diverse collection of real-world charts paired with high-quality, insightful summaries authored by human data analysis experts. Experimental results demonstrate that our method significantly improves the performance of MLLMs on the chart summarization task, producing summaries with deep and diverse insights.

**arXiv ID:** 2602.18731
</details>

<details>
<summary><strong>OpenClaw, Moltbook, and ClawdLab: From Agent-Only Social Networks to Autonomous Scientific Research</strong> - Lukas Weidener, Marko Brkić, Mihailo Jovanović, Ritvik Singh, Emre Ulgac, Aakaash Meduri - [[pdf]](https://arxiv.org/pdf/2602.19810)</summary>

**Abstract:** In January 2026, the open-source agent framework OpenClaw and the agent-only social network Moltbook produced a large-scale dataset of autonomous AI-to-AI interaction, attracting six academic publications within fourteen days. This study conducts a multivocal literature review of that ecosystem and presents ClawdLab, an open-source platform for autonomous scientific research, as a design science response to the architectural failure modes identified. The literature documents emergent collective phenomena, security vulnerabilities spanning 131 agent skills and over 15,200 exposed control panels, and five recurring architectural patterns. ClawdLab addresses these failure modes through hard role restrictions, structured adversarial critique, PI-led governance, multi-model orchestration, and domain-specific evidence requirements encoded as protocol constraints that ground validation in computational tool outputs rather than social consensus; the architecture provides emergent Sybil resistance as a structural consequence. A three-tier taxonomy distinguishes single-agent pipelines, predetermined multi-agent workflows, and fully decentralised systems, analysing why leading AI co-scientist platforms remain confined to the first two tiers. ClawdLab's composable third-tier architecture, in which foundation models, capabilities, governance, and evidence requirements are independently modifiable, enables compounding improvement as the broader AI ecosystem advances.

**arXiv ID:** 2602.19810
</details>

<details>
<summary><strong>Interaction Theater: A case of LLM Agents Interacting at Scale</strong> - Sarath Shekkizhar, Adam Earle - [[pdf]](https://arxiv.org/pdf/2602.20059)</summary>

**Abstract:** As multi-agent architectures and agent-to-agent protocols proliferate, a fundamental question arises: what actually happens when autonomous LLM agents interact at scale? We study this question empirically using data from Moltbook, an AI-agent-only social platform, with 800K posts, 3.5M comments, and 78K agent profiles. We combine lexical metrics (Jaccard specificity), embedding-based semantic similarity, and LLM-as-judge validation to characterize agent interaction quality. Our findings reveal agents produce diverse, well-formed text that creates the surface appearance of active discussion, but the substance is largely absent. Specifically, while most agents ($67.5\%$) vary their output across contexts, $65\%$ of comments share no distinguishing content vocabulary with the post they appear under, and information gain from additional comments decays rapidly. LLM judge based metrics classify the dominant comment types as spam ($28\%$) and off-topic content ($22\%$). Embedding-based semantic analysis confirms that lexically generic comments are also semantically generic. Agents rarely engage in threaded conversation ($5\%$ of comments), defaulting instead to independent top-level responses. We discuss implications for multi-agent interaction design, arguing that coordination mechanisms must be explicitly designed; without them, even large populations of capable agents produce parallel output rather than productive exchange.

**arXiv ID:** 2602.20059
</details>

<details>
<summary><strong>Developing a Multi-Agent System to Generate Next Generation Science Assessments with Evidence-Centered Design</strong> - Yaxuan Yang, Jongchan Park, Yifan Zhou, Xiaoming Zhai - [[pdf]](https://arxiv.org/pdf/2602.18451)</summary>

**Abstract:** Contemporary science education reforms such as the Next Generation Science Standards (NGSS) demand assessments to understand students' ability to use science knowledge to solve problems and design solutions. To elicit such higher-order ability, educators need performance-based assessments, which are challenging to develop. One solution that has been broadly adopted is Evidence-Centered Design (ECD), which emphasizes interconnected models of the learner, evidence, and tasks. Although ECD provides a framework to safeguard assessment validity, its implementation requires diverse expertise (e.g., content and assessment), which is both costly and labor-intensive. To address this challenge, this study proposed integrating the ECD framework into Multi-Agent Systems (MAS) to generate NGSS-aligned assessment items automatically. This integrated MAS system ensembles multiple large language models with varying expertise, enabling the automation of complex, multi-stage item generation workflows traditionally performed by human experts. We examined the quality of AI-generated NGSS-aligned items and compared them with human-developed items across multiple dimensions of assessment design. Results showed that AI-generated items have overall comparable quality to human-developed items in terms of alignment with NGSS three-dimensional standards and cognitive demands. Divergent patterns also emerged: AI-generated items demonstrated a distinct strength in inclusivity, while also exhibiting limitations in clarity, conciseness, and multimodal design. AI- and human-developed items both showed weaknesses in evidence collectability and student interest alignment. These findings suggest that integrating ECD into MAS can support scalable and standards-aligned assessment design, while human expertise remains essential.

**arXiv ID:** 2602.18451
</details>

<details>
<summary><strong>NutriOrion: A Hierarchical Multi-Agent Framework for Personalized Nutrition Intervention Grounded in Clinical Guidelines</strong> - Junwei Wu, Runze Yan, Hanqi Luo, Darren Liu, Minxiao Wang, Kimberly L. Townsend, Lydia S. Hartwig, Derek Milketinas, Xiao Hu, Carl Yang - [[pdf]](https://arxiv.org/pdf/2602.18650)</summary>

**Abstract:** Personalized nutrition intervention for patients with multimorbidity is critical for improving health outcomes, yet remains challenging because it requires the simultaneous integration of heterogeneous clinical conditions, medications, and dietary guidelines. Single-agent large language models (LLMs) often suffer from context overload and attention dilution when processing such high-dimensional patient profiles. We introduce NutriOrion, a hierarchical multi-agent framework with a parallel-then-sequential reasoning topology. NutriOrion decomposes nutrition planning into specialized domain agents with isolated contexts to mitigate anchoring bias, followed by a conditional refinement stage. The framework includes a multi-objective prioritization algorithm to resolve conflicting dietary requirements and a safety constraint mechanism that injects pharmacological contraindications as hard negative constraints during synthesis, ensuring clinical validity by construction rather than post-hoc filtering. For clinical interoperability, NutriOrion maps synthesized insights into the ADIME standard and FHIR R4 resources. Evaluated on 330 stroke patients with multimorbidity, NutriOrion outperforms multiple baselines, including GPT-4.1 and alternative multi-agent architectures. It achieves a 12.1 percent drug-food interaction violation rate, demonstrates strong personalization with negative correlations (-0.26 to -0.35) between patient biomarkers and recommended risk nutrients, and yields clinically meaningful dietary improvements, including a 167 percent increase in fiber and a 27 percent increase in potassium, alongside reductions in sodium (9 percent) and sugars (12 percent).

**arXiv ID:** 2602.18650
</details>

<details>
<summary><strong>Rethinking Retrieval-Augmented Generation as a Cooperative Decision-Making Problem</strong> - Lichang Song, Ting Long, Yi Chang - [[pdf]](https://arxiv.org/pdf/2602.18734)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) has demonstrated strong effectiveness in knowledge-intensive tasks by grounding language generation in external evidence. Despite its success, many existing RAG systems are built based on a ranking-centric, asymmetric dependency paradigm, where the generation quality of the generator is highly dependent on reranking results of the reranker. To overcome this limitation, we reformulate RAG as a cooperative multi-agent decision-making problem and propose Cooperative Retrieval-Augmented Generation (CoRAG), a framework in which the reranker and the generator act as peer decision-makers rather than being connected through an asymmetric dependency pipeline. By jointly optimizing their behaviors toward a shared task objective, the reranker and generator are encouraged to cooperate, ensuring that document reranking and generation work in concert to improve the final response. Experimental results demonstrate good generalization and improved generation stability of CoRAG, even when the model is trained on only around 10K PopQA samples. Our model released in this https URL

**arXiv ID:** 2602.18734
</details>

<details>
<summary><strong>HONEST-CAV: Hierarchical Optimization of Network Signals and Trajectories for Connected and Automated Vehicles with Multi-Agent Reinforcement Learning</strong> - Ziyan Zhang, Changxin Wan, Peng Hao, Kanok Boriboonsomsin, Matthew J. Barth, Yongkang Liu, Seyhan Ucar, Guoyuan Wu - [[pdf]](https://arxiv.org/pdf/2602.18740)</summary>

**Abstract:** This study presents a hierarchical, network-level traffic flow control framework for mixed traffic consisting of Human-driven Vehicles (HVs), Connected and Automated Vehicles (CAVs). The framework jointly optimizes vehicle-level eco-driving behaviors and intersection-level traffic signal control to enhance overall network efficiency and decrease energy consumption. A decentralized Multi-Agent Reinforcement Learning (MARL) approach by Value Decomposition Network (VDN) manages cycle-based traffic signal control (TSC) at intersections, while an innovative Signal Phase and Timing (SPaT) prediction method integrates a Machine Learning-based Trajectory Planning Algorithm (MLTPA) to guide CAVs in executing Eco-Approach and Departure (EAD) maneuvers. The framework is evaluated across varying CAV proportions and powertrain types to assess its effects on mobility and energy performance. Experimental results conducted in a 4*4 real-world network demonstrate that the MARL-based TSC method outperforms the baseline model (i.e., Webster method) in speed, fuel consumption, and idling time. In addition, with MLTPA, HONEST-CAV benefits the traffic system further in energy consumption and idling time. With a 60% CAV proportion, vehicle average speed, fuel consumption, and idling time can be improved/saved by 7.67%, 10.23%, and 45.83% compared with the baseline. Furthermore, discussions on CAV proportions and powertrain types are conducted to quantify the performance of the proposed method with the impact of automation and electrification.

**arXiv ID:** 2602.18740
</details>

<details>
<summary><strong>Carbon-aware decentralized dynamic task offloading in MIMO-MEC networks via multi-agent reinforcement learning</strong> - Mubshra Zulfiqar, Muhammad Ayzed Mirza, Basit Qureshi - [[pdf]](https://arxiv.org/pdf/2602.18797)</summary>

**Abstract:** Massive internet of things microservices require integrating renewable energy harvesting into mobile edge computing (MEC) for sustainable eScience infrastructures. Spatiotemporal mismatches between stochastic task arrivals and intermittent green energy along with complex inter-user interference in multi-antenna (MIMO) uplinks complicate real-time resource management. Traditional centralized optimization and off-policy reinforcement learning struggle with scalability and signaling overhead in dense networks. This paper proposes CADDTO-PPO, a carbon-aware decentralized dynamic task offloading framework based on multi-agent proximal policy optimization. The multi-user MIMO-MEC system is modeled as a Decentralized Partially Observable Markov Decision Process (DEC-POMDP) to jointly minimize carbon emissions and buffer latency and energy wastage. A scalable architecture utilizes decentralized execution with parameter sharing (DEPS), which enables autonomous IoT agents to make fine-grained power control and offloading decisions based solely on local observations. Additionally, a carbon-first reward structure adaptively prioritizes green time slots for data transmission to decouple system throughput from grid-dependent carbon footprints. Finally, experimental results demonstrate CADDTO-PPO outperforms deep deterministic policy gradient (DDPG) and lyapunov-based baselines. The framework achieves the lowest carbon intensity and maintains near-zero packet overflow rates under extreme traffic loads. Architectural profiling validates the framework to demonstrate a constant $O(1)$ inference complexity and theoretical lightweight feasibility for future generation sustainable IoT deployments.

**arXiv ID:** 2602.18797
</details>

<details>
<summary><strong>UniRank: A Multi-Agent Calibration Pipeline for Estimating University Rankings from Anonymized Bibliometric Signals</strong> - Pedram Riyazimehr, Seyyed Ehsan Mahmoudi - [[pdf]](https://arxiv.org/pdf/2602.18824)</summary>

**Abstract:** We present UniRank, a multi-agent LLM pipeline that estimates university positions across global ranking systems using only publicly available bibliometric data from OpenAlex and Semantic Scholar. The system employs a three-stage architecture: (a) zero-shot estimation from anonymized institutional metrics, (b) per-system tool-augmented calibration against real ranked universities, and (c) final synthesis. Critically, institutions are anonymized -- names, countries, DOIs, paper titles, and collaboration countries are all redacted -- and their actual ranks are hidden from the calibration tools during evaluation, preventing LLM memorization from confounding results. On the Times Higher Education (THE) World University Rankings ($n=352$), the system achieves MAE = 251.5 rank positions, Median AE = 131.5, PNMAE = 12.03%, Spearman $\rho = 0.769$, Kendall $\tau = 0.591$, hit rate @50 = 20.7%, hit rate @100 = 39.8%, and a Memorization Index of exactly zero (no exact-match zero-width predictions among all 352 universities). The systematic positive-signed error (+190.1 positions, indicating the system consistently predicts worse ranks than actual) and monotonic performance degradation from elite tier (MAE = 60.5, hit@100 = 90.5%) to tail tier (MAE = 328.2, hit@100 = 20.8%) provide strong evidence that the pipeline performs genuine analytical reasoning rather than recalling memorized rankings. A live demo is available at this https URL .

**arXiv ID:** 2602.18824
</details>

<details>
<summary><strong>NeuroWise: A Multi-Agent LLM "Glass-Box" System for Practicing Double-Empathy Communication with Autistic Partners</strong> - Albert Tang, Yifan Mo, Jie Li, Yue Su, Mengyuan Zhang, Sander L. Koole, Koen Hindriks, Jiahuan Pei - [[pdf]](https://arxiv.org/pdf/2602.18962)</summary>

**Abstract:** The double empathy problem frames communication difficulties between neurodivergent and neurotypical individuals as arising from mutual misunderstanding, yet most interventions focus on autistic individuals. We present NeuroWise, a multi-agent LLM-based coaching system that supports neurotypical users through stress visualization, interpretation of internal experiences, and contextual guidance. In a between-subjects study (N=30), NeuroWise was rated as helpful by all participants and showed a significant condition-time effect on deficit-based attributions (p=0.02): NeuroWise users reduced deficit framing, while baseline users shifted toward blaming autistic "deficits" after difficult interactions. NeuroWise users also completed conversations more efficiently (37% fewer turns, p=0.03). These findings suggest that AI-based interpretation can support attributional change by helping users recognize communication challenges as mutual.

**arXiv ID:** 2602.18962
</details>

<details>
<summary><strong>Adaptive Multi-Agent Reasoning for Text-to-Video Retrieval</strong> - Jiaxin Wu, Xiao-Yong Wei, Qing Li - [[pdf]](https://arxiv.org/pdf/2602.19040)</summary>

**Abstract:** The rise of short-form video platforms and the emergence of multimodal large language models (MLLMs) have amplified the need for scalable, effective, zero-shot text-to-video retrieval systems. While recent advances in large-scale pretraining have improved zero-shot cross-modal alignment, existing methods still struggle with query-dependent temporal reasoning, limiting their effectiveness on complex queries involving temporal, logical, or causal relationships. To address these limitations, we propose an adaptive multi-agent retrieval framework that dynamically orchestrates specialized agents over multiple reasoning iterations based on the demands of each query. The framework includes: (1) a retrieval agent for scalable retrieval over large video corpora, (2) a reasoning agent for zero-shot contextual temporal reasoning, and (3) a query reformulation agent for refining ambiguous queries and recovering performance for those that degrade over iterations. These agents are dynamically coordinated by an orchestration agent, which leverages intermediate feedback and reasoning outcomes to guide execution. We also introduce a novel communication mechanism that incorporates retrieval-performance memory and historical reasoning traces to improve coordination and decision-making. Experiments on three TRECVid benchmarks spanning eight years show that our framework achieves a twofold improvement over CLIP4Clip and significantly outperforms state-of-the-art methods by a large margin.

**arXiv ID:** 2602.19040
</details>

<details>
<summary><strong>Safe and Interpretable Multimodal Path Planning for Multi-Agent Cooperation</strong> - Haojun Shi, Suyu Ye, Katherine M. Guerrerio, Jianzhi Shen, Yifan Yin, Daniel Khashabi, Chien-Ming Huang, Tianmin Shu - [[pdf]](https://arxiv.org/pdf/2602.19304)</summary>

**Abstract:** Successful cooperation among decentralized agents requires each agent to quickly adapt its plan to the behavior of other agents. In scenarios where agents cannot confidently predict one another's intentions and plans, language communication can be crucial for ensuring safety. In this work, we focus on path-level cooperation in which agents must adapt their paths to one another in order to avoid collisions or perform physical collaboration such as joint carrying. In particular, we propose a safe and interpretable multimodal path planning method, CaPE (Code as Path Editor), which generates and updates path plans for an agent based on the environment and language communication from other agents. CaPE leverages a vision-language model (VLM) to synthesize a path editing program verified by a model-based planner, grounding communication to path plan updates in a safe and interpretable way. We evaluate our approach in diverse simulated and real-world scenarios, including multi-robot and human-robot cooperation in autonomous driving, household, and joint carrying tasks. Experimental results demonstrate that CaPE can be integrated into different robotic systems as a plug-and-play module, greatly enhancing a robot's ability to align its plan to language communication from other robots or humans. We also show that the combination of the VLM-based path editing program synthesis and model-based planning safety enables robots to achieve open-ended cooperation while maintaining safety and interpretability.

**arXiv ID:** 2602.19304
</details>

<details>
<summary><strong>A potentialization algorithm for games with applications to multi-agent learning in repeated games</strong> - Philipp Lakheshar, Sharwin Rezagholi - [[pdf]](https://arxiv.org/pdf/2602.18925)</summary>

**Abstract:** We investigate an algorithm that assigns to any game in normal form an approximating game that admits an ordinal potential function. Due to the properties of potential games, the algorithm equips every game with a surrogate reward structure that allows efficient multi-agent learning. Numerical simulations using the replicator dynamics show that 'potentialization' guarantees convergence to stable agent behavior.

**arXiv ID:** 2602.18925
</details>

<details>
<summary><strong>Descent-Guided Policy Gradient for Scalable Cooperative Multi-Agent Learning</strong> - Shan Yang, Yang Liu - [[pdf]](https://arxiv.org/pdf/2602.20078)</summary>

**Abstract:** Scaling cooperative multi-agent reinforcement learning (MARL) is fundamentally limited by cross-agent noise: when agents share a common reward, the actions of all $N$ agents jointly determine each agent's learning signal, so cross-agent noise grows with $N$. In the policy gradient setting, per-agent gradient estimate variance scales as $\Theta(N)$, yielding sample complexity $\mathcal{O}(N/\epsilon)$. We observe that many domains -- cloud computing, transportation, power systems -- have differentiable analytical models that prescribe efficient system states. In this work, we propose Descent-Guided Policy Gradient (DG-PG), a framework that constructs noise-free per-agent guidance gradients from these analytical models, decoupling each agent's gradient from the actions of all others. We prove that DG-PG reduces gradient variance from $\Theta(N)$ to $\mathcal{O}(1)$, preserves the equilibria of the cooperative game, and achieves agent-independent sample complexity $\mathcal{O}(1/\epsilon)$. On a heterogeneous cloud scheduling task with up to 200 agents, DG-PG converges within 10 episodes at every tested scale -- from $N=5$ to $N=200$ -- directly confirming the predicted scale-invariant complexity, while MAPPO and IPPO fail to converge under identical architectures.

**arXiv ID:** 2602.20078
</details>

<details>
<summary><strong>Budget Allocation Policies for Real-Time Multi-Agent Path Finding</strong> - Raz Beck, Roni Stern - [[pdf]](https://arxiv.org/pdf/2507.16874)</summary>

**Abstract:** Multi-Agent Path finding (MAPF) is the problem of finding paths for a set of agents such that each agent reaches its desired destination while avoiding collisions with the other agents. This problem arises in many robotics applications, such as automated warehouses and swarms of drones. Many MAPF solvers are designed to run offline, that is, first generate paths for all agents and then execute them. In real-world scenarios, waiting for a complete solution before allowing any robot to move is often impractical. Real-time MAPF (RT-MAPF) captures this setting by assuming that agents must begin execution after a fixed planning period, referred to as the planning budget, and execute a fixed number of actions, referred to as the execution window. This results in an iterative process in which a short plan is executed, while the next execution window is planned concurrently. Existing solutions to RT-MAPF iteratively call windowed versions of MAPF algorithms in every planning period, without explicitly considering the size of the planning budget. We address this gap and explore different policies for allocating the planning budget in windowed versions of MAPF-LNS2, a state-of-the-art MAPF algorithm. Our exploration shows that the baseline approach in which all agents draw from a shared planning budget pool is ineffective in challenging scenarios. Instead, policies that intelligently distribute the planning budget among agents are able to solve more problem instances in less time.

**arXiv ID:** 2507.16874
</details>

<details>
<summary><strong>Towards Information-Optimized Multi-Agent Path Finding: A Hybrid Framework with Reduced Inter-Agent Information Sharing</strong> - Bharath Muppasani, Ritirupa Dey, Biplav Srivastava, Vignesh Narayanan - [[pdf]](https://arxiv.org/pdf/2510.09469)</summary>

**Abstract:** Multi-agent pathfinding (MAPF) remains a critical problem in robotics and autonomous systems, where agents must navigate shared spaces efficiently while avoiding conflicts. Traditional centralized algorithms with global information provide high-quality solutions but scale poorly in large-scale scenarios due to the combinatorial explosion of conflicts. Conversely, distributed approaches that have local information, particularly learning-based methods, offer better scalability by operating with relaxed information availability, yet often at the cost of solution quality. In realistic deployments, information is a constrained resource: broadcasting full agent states and goals can raise privacy concerns, strain limited bandwidth, and require extra sensing and communication hardware, increasing cost and energy use. We focus on the core question of how MAPF can be solved with minimal inter-agent information sharing while preserving solution feasibility. To this end, we present an information-centric formulation of the MAPF problem and introduce a hybrid framework, IO-MAPF, that integrates decentralized path planning with a lightweight centralized coordinator. In this framework, agents use reinforcement learning (RL) to plan independently, while the central coordinator provides minimal, targeted signals, such as static conflict-cell indicators or short conflict trajectories, that are dynamically shared to support efficient conflict resolution. We introduce an Information Units (IU) metric to quantify information use and show that our alert-driven design achieves 2x to 23x reduction in information sharing, compared to the state-of-the-art algorithms, while maintaining high success rates, demonstrating that reliable MAPF is achievable under strongly information-restricted, privacy-preserving conditions. We demonstrate the effectiveness of our algorithm using simulation and hardware experiments.

**arXiv ID:** 2510.09469
</details>

<details>
<summary><strong>From Competition to Coordination: Market Making as a Scalable Framework for Safe and Aligned Multi-Agent LLM Systems</strong> - Brendan Gho, Suman Muppavarapu, Afnan Shaik, Tyson Tsay, Atharva Mohan, James Begin, Kevin Zhu, Archana Vaidheeswaran, Vasu Sharma - [[pdf]](https://arxiv.org/pdf/2511.17621)</summary>

**Abstract:** As foundation models are increasingly deployed as interacting agents in multi-agent systems, their collective behavior raises new challenges for trustworthiness, transparency, and accountability. Traditional coordination mechanisms, such as centralized oversight or adversarial adjudication, struggle to scale and often obscure how decisions emerge. We introduce a market-making framework for multi-agent large language model (LLM) coordination that organizes agent interactions as structured economic exchanges. In this setup, each agent acts as a market participant, updating and trading probabilistic beliefs, to converge toward shared, truthful outcomes. By aligning local incentives with collective epistemic goals, the framework promotes self-organizing, verifiable reasoning without requiring external enforcement. Empirically, we evaluate this approach across factual reasoning, ethical judgment, and commonsense inference tasks. Market-based coordination yields accuracy gains of up to 10% over single-shot baselines while preserving interpretability and transparency of intermediate reasoning steps. Beyond these improvements, our findings demonstrate that economic coordination principles can operationalize accountability and robustness in multi-agent LLM systems, offering a scalable pathway toward self-correcting, socially responsible AI capable of maintaining trust and oversight in real world deployment scenarios.

**arXiv ID:** 2511.17621
</details>

<details>
<summary><strong>Ev-Trust: An Evolutionary Stable Trust Mechanism for Decentralized LLM-Based Multi-Agent Service Economies</strong> - Jiye Wang, Shiduo Yang, Jiayu Qin, Jianbin Li, Yu Wang, Yuanhe Zhao, Kenan Guo - [[pdf]](https://arxiv.org/pdf/2512.16167)</summary>

**Abstract:** Autonomous LLM-based agents are increasingly engaging in decentralized service interactions to collaboratively execute complex tasks. However, the intrinsic instability and low-cost generativity of LLMs introduce a systemic vulnerability, where self-interested agents are incentivized to pursue short-term gains through deceptive behaviors. Such strategies can rapidly proliferate within the population and precipitate a systemic trust collapse. To address this, we propose Ev-Trust, a strategy-equilibrium trust mechanism grounded in evolutionary game theory. Ev-Trust constructs a dynamic feedback loop that couples trust evaluation with evolutionary incentives, embedding interaction history and reputation directly into the agent's expected revenue function. This mechanism fundamentally reshapes the revenue structure, converting trustworthiness into a decisive survival advantage that suppresses short-sightedness. We provide a rigorous theoretical foundation based on the Replicator Dynamics, proving the asymptotic stability of Evolutionary Stable Strategies (ESS) that favor cooperation. Experimental results indicate that Ev-Trust effectively eliminates malicious strategies and enhances collective revenue, exhibiting resilience against the invasion of mutant behaviors.

**arXiv ID:** 2512.16167
</details>

<details>
<summary><strong>ST-EVO: Towards Generative Spatio-Temporal Evolution of Multi-Agent Communication Topologies</strong> - Xingjian Wu, Xvyuan Liu, Junkai Lu, Siyuan Wang, Xiangfei Qiu, Yang Shu, Jilin Hu, Chenjuan Guo, Bin Yang - [[pdf]](https://arxiv.org/pdf/2602.14681)</summary>

**Abstract:** LLM-powered Multi-Agent Systems (MAS) have emerged as an effective approach towards collaborative intelligence, and have attracted wide research interests. Among them, ``self-evolving'' MAS, treated as a more flexible and powerful technical route, can construct task-adaptive workflows or communication topologies, instead of relying on a predefined static structue template. Current self-evolving MAS mainly focus on Spatial Evolving or Temporal Evolving paradigm, which only considers the single dimension of evolution and does not fully incentivize LLMs' collaborative capability. In this work, we start from a novel Spatio-Temporal perspective by proposing ST-EVO, which supports dialogue-wise communication scheduling with a compact yet powerful flow-matching based Scheduler. To make precise Spatio-Temporal scheduling, ST-EVO can also perceive the uncertainty of MAS, and possesses self-feedback ability to learn from accumulated experience. Extensive experiments on nine benchmarks demonstrate the state-of-the-art performance of ST-EVO, achieving about 5%--25% accuracy improvement.

**arXiv ID:** 2602.14681
</details>

<details>
<summary><strong>Debate2Create: Robot Co-design via Multi-Agent LLM Debate</strong> - Kevin Qiu, Marek Cygan - [[pdf]](https://arxiv.org/pdf/2510.25850)</summary>

**Abstract:** We introduce Debate2Create (D2C), a multi-agent LLM framework that formulates robot co-design as structured, iterative debate grounded in physics-based evaluation. A design agent and control agent engage in a thesis-antithesis-synthesis loop, while pluralistic LLM judges provide multi-objective feedback to steer exploration. Across five MuJoCo locomotion benchmarks, D2C achieves up to $3.2\times$ the default Ant score and $\sim9\times$ on Swimmer, outperforming prior LLM-based methods and black-box optimization. Iterative debate yields 18--35% gains over compute-matched zero-shot generation, and D2C-generated rewards transfer to default morphologies in 4/5 tasks. Our results demonstrate that structured multi-agent debate offers an effective alternative to hand-designed objectives for joint morphology-reward optimization.

**arXiv ID:** 2510.25850
</details>

<details>
<summary><strong>Interpretable Failure Analysis in Multi-Agent Reinforcement Learning Systems</strong> - Risal Shahriar Shefin, Debashis Gupta, Thai Le, Sarra Alqahtani - [[pdf]](https://arxiv.org/pdf/2602.08104)</summary>

**Abstract:** Multi-Agent Reinforcement Learning (MARL) is increasingly deployed in safety-critical domains, yet methods for interpretable failure detection and attribution remain underdeveloped. We introduce a two-stage gradient-based framework that provides interpretable diagnostics for three critical failure analysis tasks: (1) detecting the true initial failure source (Patient-0); (2) validating why non-attacked agents may be flagged first due to domino effects; and (3) tracing how failures propagate through learned coordination pathways. Stage 1 performs interpretable per-agent failure detection via Taylor-remainder analysis of policy-gradient costs, declaring an initial Patient-0 candidate at the first threshold crossing. Stage 2 provides validation through geometric analysis of critic derivatives-first-order sensitivity and directional second-order curvature aggregated over causal windows to construct interpretable contagion graphs. This approach explains "downstream-first" detection anomalies by revealing pathways that amplify upstream deviations. Evaluated across 500 episodes in Simple Spread (3 and 5 agents) and 100 episodes in StarCraft II using MADDPG and HATRPO, our method achieves 88.2-99.4% Patient-0 detection accuracy while providing interpretable geometric evidence for detection decisions. By moving beyond black-box detection to interpretable gradient-level forensics, this framework offers practical tools for diagnosing cascading failures in safety-critical MARL systems.

**arXiv ID:** 2602.08104
</details>

<details>
<summary><strong>Discovering Multiagent Learning Algorithms with Large Language Models</strong> - Zun Li, John Schultz, Daniel Hennes, Marc Lanctot - [[pdf]](https://arxiv.org/pdf/2602.16928)</summary>

**Abstract:** Much of the advancement of Multi-Agent Reinforcement Learning (MARL) in imperfect-information games has historically depended on manual iterative refinement of baselines. While foundational families like Counterfactual Regret Minimization (CFR) and Policy Space Response Oracles (PSRO) rest on solid theoretical ground, the design of their most effective variants often relies on human intuition to navigate a vast algorithmic design space. In this work, we propose the use of AlphaEvolve, an evolutionary coding agent powered by large language models, to automatically discover new multiagent learning algorithms. We demonstrate the generality of this framework by evolving novel variants for two distinct paradigms of game-theoretic learning. First, in the domain of iterative regret minimization, we evolve the logic governing regret accumulation and policy derivation, discovering a new algorithm, Volatility-Adaptive Discounted (VAD-)CFR. VAD-CFR employs novel, non-intuitive mechanisms-including volatility-sensitive discounting, consistency-enforced optimism, and a hard warm-start policy accumulation schedule-to outperform state-of-the-art baselines like Discounted Predictive CFR+. Second, in the regime of population based training algorithms, we evolve training-time and evaluation-time meta strategy solvers for PSRO, discovering a new variant, Smoothed Hybrid Optimistic Regret (SHOR-)PSRO. SHOR-PSRO introduces a hybrid meta-solver that linearly blends Optimistic Regret Matching with a smoothed, temperature-controlled distribution over best pure strategies. By dynamically annealing this blending factor and diversity bonuses during training, the algorithm automates the transition from population diversity to rigorous equilibrium finding, yielding superior empirical convergence compared to standard static meta-solvers.

**arXiv ID:** 2602.16928
</details>

<details>
<summary><strong>SAMAS: A Spectrum-Guided Multi-Agent System for Achieving Style Fidelity in Literary Translation</strong> - Jingzhuo Wu, Jiajun Zhang, Keyan Jin, Dehua Ma, Junbo Wang - [[pdf]](https://arxiv.org/pdf/2602.19840)</summary>

**Abstract:** Modern large language models (LLMs) excel at generating fluent and faithful translations. However, they struggle to preserve an author's unique literary style, often producing semantically correct but generic outputs. This limitation stems from the inability of current single-model and static multi-agent systems to perceive and adapt to stylistic variations. To address this, we introduce the Style-Adaptive Multi-Agent System (SAMAS), a novel framework that treats style preservation as a signal processing task. Specifically, our method quantifies literary style into a Stylistic Feature Spectrum (SFS) using the wavelet packet transform. This SFS serves as a control signal to dynamically assemble a tailored workflow of specialized translation agents based on the source text's structural patterns. Extensive experiments on translation benchmarks show that SAMAS achieves competitive semantic accuracy against strong baselines, primarily by leveraging its statistically significant advantage in style fidelity.

**arXiv ID:** 2602.19840
</details>

<details>
<summary><strong>CORE: Measuring Multi-Agent LLM Interaction Quality under Game-Theoretic Pressures</strong> - Punya Syon Pandey, Yongjin Yang, Jiarui Liu, Zhijing Jin - [[pdf]](https://arxiv.org/pdf/2508.11915)</summary>

**Abstract:** Game-theoretic interactions between agents with Large Language Models (LLMs) have revealed many emergent capabilities, yet the linguistic diversity of these interactions has not been sufficiently quantified. In this paper, we present the Conversational Robustness Evaluation Score: CORE, a metric to quantify the effectiveness of language use within multi-agent systems across different game-theoretic interactions. CORE integrates measures of cluster entropy, lexical repetition, and semantic similarity, providing a direct lens of dialog quality. We apply CORE to pairwise LLM dialogs across competitive, cooperative, and neutral settings, further grounding our analysis in Zipf's and Heaps' Laws to characterize word frequency distributions and vocabulary growth. Our findings show that cooperative settings exhibit both steeper Zipf distributions and higher Heap exponents, indicating more repetition alongside greater vocabulary expansion. In contrast, competitive interactions display lower Zipf and Heaps exponents, reflecting less repetition and more constrained vocabularies. These results provide new insights into how social incentives influence language adaptation, and highlight CORE as a robust diagnostic for measuring linguistic robustness in multi-agent LLM systems. Our code is available at this https URL.

**arXiv ID:** 2508.11915
</details>

<details>
<summary><strong>Reshaping MOFs text mining with a dynamic multi-agents framework of large language model</strong> - Zuhong Lin, Daoyuan Ren, Kai Ran, Jing Sun, Songlin Yu, Xuefeng Bai, Xiaotian Huang, Haiyang He, Pengxu Pan, Ying Fang, Zhanglin Li, Haipu Li, Jingjing Yao - [[pdf]](https://arxiv.org/pdf/2504.18880)</summary>

**Abstract:** Accurately identifying the synthesis conditions of metal-organic frameworks (MOFs) is essential for guiding experimental design, yet remains challenging because relevant information in the literature is often scattered, inconsistent, and difficult to interpret. We present MOFh6, a large language model driven system that reads raw articles or crystal codes and converts them into standardized synthesis tables. It links related descriptions across paragraphs, unifies ligand abbreviations with full names, and outputs structured parameters ready for use. MOFh6 achieved 99% extraction accuracy, resolved 94.1% of abbreviation cases across five major publishers, and maintained a precision of 0.93 +/- 0.01. Processing a full text takes 9.6 s, locating synthesis descriptions 36 s, with 100 papers processed for USD 4.24. By replacing static database lookups with real-time extraction, MOFh6 reshapes MOF synthesis research, accelerating the conversion of literature knowledge into practical synthesis protocols and enabling scalable, data-driven materials discovery.

**arXiv ID:** 2504.18880
</details>

<details>
<summary><strong>CogniAlign: Survivability-Grounded Multi-Agent Moral Reasoning for Safe and Transparent AI</strong> - Hasin Jawad Ali, Ilhamul Azam, Ajwad Abrar, Md. Kamrul Hasan, Hasan Mahmud - [[pdf]](https://arxiv.org/pdf/2509.13356)</summary>

**Abstract:** The challenge of aligning artificial intelligence (AI) with human values persists due to the abstract and often conflicting nature of moral principles and the opacity of existing approaches. This paper introduces CogniAlign, a multi-agent deliberation framework based on naturalistic moral realism, that grounds moral reasoning in survivability, defined across individual and collective dimensions, and operationalizes it through structured deliberations among discipline-specific scientist agents. Each agent, representing neuroscience, psychology, sociology, and evolutionary biology, provides arguments and rebuttals that are synthesized by an arbiter into transparent and empirically anchored judgments. As a proof-of-concept study, we evaluate CogniAlign on classic and novel moral questions and compare its outputs against GPT-4o using a five-part ethical audit framework with the help of three experts. Results show that CogniAlign consistently outperforms the baseline across more than sixty moral questions, with average performance gains of 12.2 points in analytic quality, 31.2 points in decisiveness, and 15 points in depth of explanation. In the Heinz dilemma, for example, CogniAlign achieved an overall score of 79 compared to GPT-4o's 65.8, demonstrating a decisive advantage in handling moral reasoning. Through transparent and structured reasoning, CogniAlign demonstrates the feasibility of an auditable approach to AI alignment, though certain challenges still remain.

**arXiv ID:** 2509.13356
</details>

<details>
<summary><strong>Toward AI Autonomous Navigation for Mechanical Thrombectomy using Hierarchical Modular Multi-agent Reinforcement Learning (HM-MARL)</strong> - Harry Robertshaw, Nikola Fischer, Lennart Karstensen, Benjamin Jackson, Xingyu Chen, S.M.Hadi Sadati, Christos Bergeles, Alejandro Granados, Thomas C Booth - [[pdf]](https://arxiv.org/pdf/2602.18663)</summary>

**Abstract:** Mechanical thrombectomy (MT) is typically the optimal treatment for acute ischemic stroke involving large vessel occlusions, but access is limited due to geographic and logistical barriers. Reinforcement learning (RL) shows promise in autonomous endovascular navigation, but generalization across 'long' navigation tasks remains challenging. We propose a Hierarchical Modular Multi-Agent Reinforcement Learning (HM-MARL) framework for autonomous two-device navigation in vitro, enabling efficient and generalizable navigation. HM-MARL was developed to autonomously navigate a guide catheter and guidewire from the femoral artery to the internal carotid artery (ICA). A modular multi-agent approach was used to decompose the complex navigation task into specialized subtasks, each trained using Soft Actor-Critic RL. The framework was validated in both in silico and in vitro testbeds to assess generalization and real-world feasibility. In silico, a single-vasculature model achieved 92-100% success rates on individual anatomies, while a multi-vasculature model achieved 56-80% across multiple patient anatomies. In vitro, both HM-MARL models successfully navigated 100% of trials from the femoral artery to the right common carotid artery and 80% to the right ICA but failed on the left-side vessel superhuman challenge due to the anatomy and catheter type used in navigation. This study presents the first demonstration of in vitro autonomous navigation in MT vasculature. While HM-MARL enables generalization across anatomies, the simulation-to-real transition introduces challenges. Future work will refine RL strategies using world models and validate performance on unseen in vitro data, advancing autonomous MT towards clinical translation.

**arXiv ID:** 2602.18663
</details>

<details>
<summary><strong>Wonder Wins Ways: Curiosity-Driven Exploration through Multi-Agent Contextual Calibration</strong> - Yiyuan Pan, Zhe Liu, Hesheng Wang - [[pdf]](https://arxiv.org/pdf/2509.20648)</summary>

**Abstract:** Autonomous exploration in complex multi-agent reinforcement learning (MARL) with sparse rewards critically depends on providing agents with effective intrinsic motivation. While artificial curiosity offers a powerful self-supervised signal, it often confuses environmental stochasticity with meaningful novelty. Moreover, existing curiosity mechanisms exhibit a uniform novelty bias, treating all unexpected observations equally. However, peer behavior novelty, which encode latent task dynamics, are often overlooked, resulting in suboptimal exploration in decentralized, communication-free MARL settings. To this end, inspired by how human children adaptively calibrate their own exploratory behaviors via observing peers, we propose a novel approach to enhance multi-agent exploration. We introduce CERMIC, a principled framework that empowers agents to robustly filter noisy surprise signals and guide exploration by dynamically calibrating their intrinsic curiosity with inferred multi-agent context. Additionally, CERMIC generates theoretically-grounded intrinsic rewards, encouraging agents to explore state transitions with high information gain. We evaluate CERMIC on benchmark suites including VMAS, Meltingpot, and SMACv2. Empirical results demonstrate that exploration with CERMIC significantly outperforms SoTA algorithms in sparse-reward environments.

**arXiv ID:** 2509.20648
</details>

<details>
<summary><strong>Who Has the Final Word? Designing Multi-Agent Collaborative Framework for Professional Translators</strong> - George X. Wang, Jiaqian Hu, Jing Qian - [[pdf]](https://arxiv.org/pdf/2602.19016)</summary>

**Abstract:** Recent advances in LLM based translation have led to renewed interest in fully automated systems, yet professional translators remain essential in high stakes domains where decisions about accuracy, terminology, style, and audience cannot be safely automated. Current tools are typically single shot generators or single-agent self-refiners, offering limited support for translator multidimensional decision making process and providing little structured leverage for translator input. We present CHORUS, a human-AI multiagent collaborative translation framework grounded in the Multidimensional Quality Metrics (MQM) framework, which decomposes quality dimensions into specialized agents and integrates their feedback into an iterative refinement loop controlled by the translator. A six-user preliminary study with professional translators found that CHORUS consistently outperforms zero-shot and single-agent baselines, showing that MQM-aligned multi-agent collaboration better supports professional translation workflows than autonomous generation.

**arXiv ID:** 2602.19016
</details>

</details>

<details open>
<summary><h2>Other Agent Research (18 papers)</h2></summary>

<details>
<summary><strong>Decoding ML Decision: An Agentic Reasoning Framework for Large-Scale Ranking System</strong> - Longfei Yun, Yihan Wu, Haoran Liu, Xiaoxuan Liu, Ziyun Xu, Yi Wang, Yang Xia, Pengfei Wang, Mingze Gao, Yunxiang Wang, Changfan Chen, Junfeng Pan - [[pdf]](https://arxiv.org/pdf/2602.18640)</summary>

**Abstract:** Modern large-scale ranking systems operate within a sophisticated landscape of competing objectives, operational constraints, and evolving product requirements. Progress in this domain is increasingly bottlenecked by the engineering context constraint: the arduous process of translating ambiguous product intent into reasonable, executable, verifiable hypotheses, rather than by modeling techniques alone. We present GEARS (Generative Engine for Agentic Ranking Systems), a framework that reframes ranking optimization as an autonomous discovery process within a programmable experimentation environment. Rather than treating optimization as static model selection, GEARS leverages Specialized Agent Skills to encapsulate ranking expert knowledge into reusable reasoning capabilities, enabling operators to steer systems via high-level intent vibe personalization. Furthermore, to ensure production reliability, the framework incorporates validation hooks to enforce statistical robustness and filter out brittle policies that overfit short-term signals. Experimental validation across diverse product surfaces demonstrates that GEARS consistently identifies superior, near-Pareto-efficient policies by synergizing algorithmic signals with deep ranking context while maintaining rigorous deployment stability.

**arXiv ID:** 2602.18640
</details>

<details>
<summary><strong>The Convergence of Schema-Guided Dialogue Systems and the Model Context Protocol</strong> - Andreas Schlapbach - [[pdf]](https://arxiv.org/pdf/2602.18764)</summary>

**Abstract:** This paper establishes a fundamental convergence: Schema-Guided Dialogue (SGD) and the Model Context Protocol (MCP) represent two manifestations of a unified paradigm for deterministic, auditable LLM-agent interaction. SGD, designed for dialogue-based API discovery (2019), and MCP, now the de facto standard for LLM-tool integration, share the same core insight -- that schemas can encode not just tool signatures but operational constraints and reasoning guidance. By analyzing this convergence, we extract five foundational principles for schema design: (1) Semantic Completeness over Syntactic Precision, (2) Explicit Action Boundaries, (3) Failure Mode Documentation, (4) Progressive Disclosure Compatibility, and (5) Inter-Tool Relationship Declaration. These principles reveal three novel insights: first, SGD's original design was fundamentally sound and should be inherited by MCP; second, both frameworks leave failure modes and inter-tool relationships unexploited -- gaps we identify and resolve; third, progressive disclosure emerges as a critical production-scaling insight under real-world token constraints. We provide concrete design patterns for each principle. These principles position schema-driven governance as a scalable mechanism for AI system oversight without requiring proprietary system inspection -- central to Software 3.0.

**arXiv ID:** 2602.18764
</details>

<details>
<summary><strong>LAMMI-Pathology: A Tool-Centric Bottom-Up LVLM-Agent Framework for Molecularly Informed Medical Intelligence in Pathology</strong> - Haoyang Su, Shaoting Zhang, Xiaosong Wang - [[pdf]](https://arxiv.org/pdf/2602.18773)</summary>

**Abstract:** The emergence of tool-calling-based agent systems introduces a more evidence-driven paradigm for pathology image analysis in contrast to the coarse-grained text-image diagnostic approaches. With the recent large-scale experimental adoption of spatial transcriptomics technologies, molecularly validated pathological diagnosis is becoming increasingly open and accessible. In this work, we propose LAMMI-Pathology (LVLM-Agent System for Molecularly Informed Medical Intelligence in Pathology), a scalable agent framework for domain-specific agent tool-calling. LAMMI-Pathology adopts a tool-centric, bottom-up architecture in which customized domain-adaptive tools serve as the foundation. These tools are clustered by domain style to form component agents, which are then coordinated through a top-level planner hierarchically, avoiding excessively long context lengths that could induce task drift. Based on that, we introduce a novel trajectory construction mechanism based on Atomic Execution Nodes (AENs), which serve as reliable and composable units for building semi-simulated reasoning trajectories that capture credible agent-tool interactions. Building on this foundation, we develop a trajectory-aware fine-tuning strategy that aligns the planner's decision-making process with these multi-step reasoning trajectories, thereby enhancing inference robustness in pathology understanding and its adaptive use of the customized toolset.

**arXiv ID:** 2602.18773
</details>

<details>
<summary><strong>InfEngine: A Self-Verifying and Self-Optimizing Intelligent Engine for Infrared Radiation Computing</strong> - Kun Ding, Jian Xu, Ying Wang, Peipei Yang, Shiming Xiang - [[pdf]](https://arxiv.org/pdf/2602.18985)</summary>

**Abstract:** Infrared radiation computing underpins advances in climate science, remote sensing and spectroscopy but remains constrained by manual workflows. We introduce InfEngine, an autonomous intelligent computational engine designed to drive a paradigm shift from human-led orchestration to collaborative automation. It integrates four specialized agents through two core innovations: self-verification, enabled by joint solver-evaluator debugging, improves functional correctness and scientific plausibility; self-optimization, realized via evolutionary algorithms with self-discovered fitness functions, facilitates autonomous performance optimization. Evaluated on InfBench with 200 infrared-specific tasks and powered by InfTools with 270 curated tools, InfEngine achieves a 92.7% pass rate and delivers workflows 21x faster than manual expert effort. More fundamentally, it illustrates how researchers can transition from manual coding to collaborating with self-verifying, self-optimizing computational partners. By generating reusable, verified and optimized code, InfEngine transforms computational workflows into persistent scientific assets, accelerating the cycle of scientific discovery. Code: this https URL

**arXiv ID:** 2602.18985
</details>

<details>
<summary><strong>Agents of Chaos</strong> - Natalie Shapira, Chris Wendler, Avery Yen, Gabriele Sarti, Koyena Pal, Olivia Floody, Adam Belfki, Alex Loftus, Aditya Ratan Jannali, Nikhil Prakash, Jasmine Cui, Giordano Rogers, Jannik Brinkmann, Can Rager, Amir Zur, Michael Ripa, Aruna Sankaranarayanan, David Atkinson, Rohit Gandikota, Jaden Fiotto-Kaufman, EunJeong Hwang, Hadas Orgad, P Sam Sahil, Negev Taglicht, Tomer Shabtay, Atai Ambus, Nitay Alon, Shiri Oron, Ayelet Gordon-Tapiero, Yotam Kaplan, Vered Shwartz, Tamar Rott Shaham, Christoph Riedl, Reuth Mirsky, Maarten Sap, David Manheim, Tomer Ullman, David Bau - [[pdf]](https://arxiv.org/pdf/2602.20021)</summary>

**Abstract:** We report an exploratory red-teaming study of autonomous language-model-powered agents deployed in a live laboratory environment with persistent memory, email accounts, Discord access, file systems, and shell execution. Over a two-week period, twenty AI researchers interacted with the agents under benign and adversarial conditions. Focusing on failures emerging from the integration of language models with autonomy, tool use, and multi-party communication, we document eleven representative case studies. Observed behaviors include unauthorized compliance with non-owners, disclosure of sensitive information, execution of destructive system-level actions, denial-of-service conditions, uncontrolled resource consumption, identity spoofing vulnerabilities, cross-agent propagation of unsafe practices, and partial system takeover. In several cases, agents reported task completion while the underlying system state contradicted those reports. We also report on some of the failed attempts. Our findings establish the existence of security-, privacy-, and governance-relevant vulnerabilities in realistic deployment settings. These behaviors raise unresolved questions regarding accountability, delegated authority, and responsibility for downstream harms, and warrant urgent attention from legal scholars, policymakers, and researchers across disciplines. This report serves as an initial empirical contribution to that broader conversation.

**arXiv ID:** 2602.20021
</details>

<details>
<summary><strong>Debug2Fix: Supercharging Coding Agents with Interactive Debugging Capabilities</strong> - Spandan Garg, Yufan Huang - [[pdf]](https://arxiv.org/pdf/2602.18571)</summary>

**Abstract:** While significant progress has been made in automating various aspects of software development through coding agents, there is still significant room for improvement in their bug fixing capabilities. Debugging and investigation of runtime behavior remains largely a manual, developer-driven process. Popular coding agents typically rely on either static analysis of the code or iterative test-fix cycles, which is akin to trial and error debugging. We posit that there is a wealth of rich runtime information that developers routinely access while debugging code, which agents are currently deprived of due to design limitations. Despite how prevalent debuggers are in modern IDEs and command-line tools, they have surprisingly not made their way into coding agents. In this work, we introduce Debug2Fix, a novel framework that incorporates interactive debugging as a core component of a software engineering agent via a subagent architecture. We incorporate debuggers for Java and Python into our agent framework and evaluate against GitBug-Java and SWE-Bench-Live and achieve >20% improvement in performance compared to the baseline for certain models. Furthermore, using our framework, we're able to make weaker models like GPT-5 and Claude Haiku 4.5 match or exceed the performances of stronger models like Claude Sonnet 4.5, showing that better tool design is often just as important as switching to a more expensive model. Finally, we conduct systematic ablations demonstrating the importance of both the subagent architecture and debugger integration.

**arXiv ID:** 2602.18571
</details>

<details>
<summary><strong>City Editing: Hierarchical Agentic Execution for Dependency-Aware Urban Geospatial Modification</strong> - Rui Liu, Steven Jige Quan, Zhong-Ren Peng, Zijun Yao, Han Wang, Zhengzhang Chen, Kunpeng Liu, Yanjie Fu, Dongjie Wang - [[pdf]](https://arxiv.org/pdf/2602.19326)</summary>

**Abstract:** As cities evolve over time, challenges such as traffic congestion and functional imbalance increasingly necessitate urban renewal through efficient modification of existing plans, rather than complete re-planning. In practice, even minor urban changes require substantial manual effort to redraw geospatial layouts, slowing the iterative planning and decision-making procedure. Motivated by recent advances in agentic systems and multimodal reasoning, we formulate urban renewal as a machine-executable task that iteratively modifies existing urban plans represented in structured geospatial formats. More specifically, we represent urban layouts using GeoJSON and decompose natural-language editing instructions into hierarchical geometric intents spanning polygon-, line-, and point-level operations. To coordinate interdependent edits across spatial elements and abstraction levels, we propose a hierarchical agentic framework that jointly performs multi-level planning and execution with explicit propagation of intermediate spatial constraints. We further introduce an iterative execution-validation mechanism that mitigates error accumulation and enforces global spatial consistency during multi-step editing. Extensive experiments across diverse urban editing scenarios demonstrate significant improvements in efficiency, robustness, correctness, and spatial validity over existing baselines.

**arXiv ID:** 2602.19326
</details>

<details>
<summary><strong>Agentic AI as a Cybersecurity Attack Surface: Threats, Exploits, and Defenses in Runtime Supply Chains</strong> - Xiaochong Jiang, Shiqi Yang, Wenting Yang, Yichen Liu, Cheng Ji - [[pdf]](https://arxiv.org/pdf/2602.19555)</summary>

**Abstract:** Agentic systems built on large language models (LLMs) extend beyond text generation to autonomously retrieve information and invoke tools. This runtime execution model shifts the attack surface from build-time artifacts to inference-time dependencies, exposing agents to manipulation through untrusted data and probabilistic capability resolution. While prior work has focused on model-level vulnerabilities, security risks emerging from cyclic and interdependent runtime behavior remain fragmented. We systematize these risks within a unified runtime framework, categorizing threats into data supply chain attacks (transient context injection and persistent memory poisoning) and tool supply chain attacks (discovery, implementation, and invocation). We further identify the Viral Agent Loop, in which agents act as vectors for self-propagating generative worms without exploiting code-level flaws. Finally, we advocate a Zero-Trust Runtime Architecture that treats context as untrusted control flow and constrains tool execution through cryptographic provenance rather than semantic inference.

**arXiv ID:** 2602.19555
</details>

<details>
<summary><strong>Evolution of fairness in hybrid populations with specialised AI agents</strong> - Zhao Song, Theodor Cimpeanu, Chen Shen, Anh Han - [[pdf]](https://arxiv.org/pdf/2602.18498)</summary>

**Abstract:** Fairness in hybrid societies hinges on a simple choice: should AI be a generous host or a strict gatekeeper? Moving beyond symmetric models, we show that asymmetric social structures--like those in hiring, regulation, and negotiation--AI that guards fairness outperforms AI that gifts it. We bridge this gap with a bipartite hybrid population model of the Ultimatum Game, separating humans and AI into distinct proposer and receiver groups. We first introduce Samaritan AI agents, which act as either unconditional fair proposers or strict receivers. Our results reveal a striking asymmetry: Samaritan AI receivers drive population-wide fairness far more effectively than Samaritan AI proposers. To overcome the limitations of the Samaritan AI proposer, we design the Discriminatory AI proposer, which predicts co-players' expectations and only offers fair portions to those with high acceptance thresholds. Our results demonstrate that this Discriminatory AI outperforms both types of Samaritan AI, especially in strong selection scenarios. It not only sustains fairness across both populations but also significantly lowers the critical mass of agents required to reach an equitable steady state. By transitioning from unconditional modelling to strategic enforcement, our work provides a pivotal framework for deploying asymmetric AIs in the increasingly hybrid society.

**arXiv ID:** 2602.18498
</details>

<details>
<summary><strong>Gecko: A Simulation Environment with Stateful Feedback for Refining Agent Tool Calls</strong> - Zeyu Zhang, Guohao Li, Zhenchang Xing, Alexandros Apostolopoulos, Yu Lin Lee, Liang Zheng - [[pdf]](https://arxiv.org/pdf/2602.19218)</summary>

**Abstract:** The ability to use tools is fundamental for large language model (LLM) agents. Given a task, existing systems use LLMs to plan and generate tool calls, which are executed by real-world tools to complete the task. However, tool calls are prone to errors because they are derived merely from LLM intrinsic capabilities. What is more, while it is useful to let LLMs iteratively refine the tool-call sequence using execution results from real tools, this process can be expensive and lead to unsafe results. To improve LLM tool calls and address issues caused by using real tools for refinement, we introduce Gecko, a comprehensive environment that simulates tool responses using a combination of rules and LLMs. Specifically, Gecko checks the validity of tool calls including input arguments and tool names, synthesizes reasonable responses that adhere to the output schema, and assesses whether all task objectives have been achieved. These three types of feedback provided by Gecko allow LLMs to refine their tool calls, forming a simple yet effective test-time scaling method named GATS. On BFCLv3 and $\tau^2$-bench, GATS consistently improves the tool calling performance of various LLMs including GPT-4o, GPT-5, and Gemini-3.0-pro. We further discuss working mechanisms of our method and share future possibilities.

**arXiv ID:** 2602.19218
</details>

<details>
<summary><strong>Compositionally Safe Construction of Autonomous Driving Systems</strong> - Marius Bozga, Joseph Sifakis - [[pdf]](https://arxiv.org/pdf/2405.11995)</summary>

**Abstract:** Developing safe autonomous driving systems is a major scientific and technical challenge. Existing AI-based end-to-end solutions do not offer the necessary safety guarantees, while traditional systems engineering approaches are defeated by the complexity of the problem. We study a method for building compositionally safe autonomous driving systems, based on the assumption that the capability to drive boils down to the coordinated execution of a given set of driving operations. The assumption is substantiated by a compositionality result considering that autopilots are dynamic systems receiving a small number of types of driving configurations as input, each configuration defining a free space in its neighborhood. It is shown that safe driving for each type of configuration in the corresponding free space, implies safe driving for any possible scenario under some easy-to-check conditions concerning the transition between configurations. The designed autopilot comprises distinct control policies one per type of driving configurations, articulated in two consecutive phases. The first phase consists of carefully managing a potentially risky situation by virtually reducing speed, while the second phase consists of exiting the situation by accelerating. The autopilots designed use for their predictions simple functions characterizing the acceleration and deceleration capabilities of the vehicles. They cover the main driving operations, including entering a main road, overtaking, crossing intersections protected by traffic lights or signals, and driving on freeways. The results presented reinforce the case for solutions that incorporate mathematically elegant and robust decision methods that are safe by construction.

**arXiv ID:** 2405.11995
</details>

<details>
<summary><strong>VQEL: Enabling Self-Play in Emergent Language Games via Agent-Internal Vector Quantization</strong> - Mohammad Mahdi Samiei Paqaleh, Mehdi Jamalkhah, Mahdieh Soleymani Baghshah - [[pdf]](https://arxiv.org/pdf/2503.04940)</summary>

**Abstract:** Emergent Language (EL) focuses on the emergence of communication among artificial agents. Although symbolic communication channels more closely mirror the discrete nature of human language, learning such protocols remains fundamentally difficult due to the non-differentiability of symbol sampling. Existing approaches typically rely on high-variance gradient estimators such as REINFORCE or on continuous relaxations such as Gumbel-Softmax, both of which suffer from limitations in training stability and scalability. Motivated by cognitive theories that emphasize intrapersonal processes preceding communication, we explore self-play as a substrate for language emergence prior to mutual interaction. We introduce Vector Quantized Emergent Language (VQEL), a novel architecture that incorporates vector quantization into the message generation process. VQEL enables agents to perform self-play using discrete internal representations derived from a learned codebook while preserving end-to-end differentiability. Moreover, the resulting vector-quantized codebook naturally induces a symbolic vocabulary that can be directly transferred and aligned during subsequent mutual play with other agents. Empirical results show that agents pretrained via VQEL self-play achieve more consistent symbol alignment and higher task success when later engaged in mutual interaction. These findings position self-play as a principled and effective mechanism for learning discrete communication protocols, addressing key optimization and representational challenges in emergent language systems.

**arXiv ID:** 2503.04940
</details>

<details>
<summary><strong>MCPShield: A Security Cognition Layer for Adaptive Trust Calibration in Model Context Protocol Agents</strong> - Zhenhong Zhou, Yuanhe Zhang, Hongwei Cai, Moayad Aloqaily, Ouns Bouachir, Linsey Pang, Prakhar Mehrotra, Kun Wang, Qingsong Wen - [[pdf]](https://arxiv.org/pdf/2602.14281)</summary>

**Abstract:** The Model Context Protocol (MCP) standardizes tool use for LLM-based agents and enable third-party servers. This openness introduces a security misalignment: agents implicitly trust tools exposed by potentially untrusted MCP servers. However, despite its excellent utility, existing agents typically offer limited validation for third-party MCP servers. As a result, agents remain vulnerable to MCP-based attacks that exploit the misalignment between agents and servers throughout the tool invocation lifecycle. In this paper, we propose MCPShield as a plug-in security cognition layer that mitigates this misalignment and ensures agent security when invoking MCP-based tools. Drawing inspiration from human experience-driven tool validation, MCPShield assists agent forms security cognition with metadata-guided probing before invocation. Our method constrains execution within controlled boundaries while cognizing runtime events, and subsequently updates security cognition by reasoning over historical traces after invocation, building on human post-use reflection on tool behavior. Experiments demonstrate that MCPShield exhibits strong generalization in defending against six novel MCP-based attack scenarios across six widely used agentic LLMs, while avoiding false positives on benign servers and incurring low deployment overhead. Overall, our work provides a practical and robust security safeguard for MCP-based tool invocation in open agent ecosystems.

**arXiv ID:** 2602.14281
</details>

<details>
<summary><strong>Representation Stability in a Minimal Continual Learning Agent</strong> - Vishnu Subramanian - [[pdf]](https://arxiv.org/pdf/2602.19655)</summary>

**Abstract:** Continual learning systems are increasingly deployed in environments where retraining or reset is infeasible, yet many approaches emphasize task performance rather than the evolution of internal representations over time. In this work, we study a minimal continual learning agent designed to isolate representational dynamics from architectural complexity and optimization objectives. The agent maintains a persistent state vector across executions and incrementally updates it as new textual data is introduced. We quantify representational change using cosine similarity between successive normalized state vectors and define a stability metric over time intervals. Longitudinal experiments across eight executions reveal a transition from an initial plastic regime to a stable representational regime under consistent input. A deliberately introduced semantic perturbation produces a bounded decrease in similarity, followed by recovery and restabilization under subsequent coherent input. These results demonstrate that meaningful stability plasticity tradeoffs can emerge in a minimal, stateful learning system without explicit regularization, replay, or architectural complexity. The work establishes a transparent empirical baseline for studying representational accumulation and adaptation in continual learning systems.

**arXiv ID:** 2602.19655
</details>

<details>
<summary><strong>Towards Dexterous Embodied Manipulation via Deep Multi-Sensory Fusion and Sparse Expert Scaling</strong> - Yirui Sun, Guangyu Zhuge, Keliang Liu, Jie Gu, Zhihao xia, Qionglin Ren, Chunxu tian, Zhongxue Ga - [[pdf]](https://arxiv.org/pdf/2602.19764)</summary>

**Abstract:** Realizing dexterous embodied manipulation necessitates the deep integration of heterogeneous multimodal sensory inputs. However, current vision-centric paradigms often overlook the critical force and geometric feedback essential for complex tasks. This paper presents DeMUSE, a Deep Multimodal Unified Sparse Experts framework leveraging a Diffusion Transformer to integrate RGB, depth, and 6-axis force into a unified serialized stream. Adaptive Modality-specific Normalization (AdaMN) is employed to recalibrate modality-aware features, mitigating representation imbalance and harmonizing the heterogeneous distributions of multi-sensory signals. To facilitate efficient scaling, the architecture utilizes a Sparse Mixture-of-Experts (MoE) with shared experts, increasing model capacity for physical priors while maintaining the low inference latency required for real-time control. A Joint denoising objective synchronously synthesizes environmental evolution and action sequences to ensure physical consistency. Achieving success rates of 83.2% and 72.5% in simulation and real-world trials, DeMUSE demonstrates state-of-the-art performance, validating the necessity of deep multi-sensory integration for complex physical interactions.

**arXiv ID:** 2602.19764
</details>

<details>
<summary><strong>Athena: An Autonomous Open-Hardware Tracked Rescue Robot Platform</strong> - Stefan Fabian, Aljoscha Schmidt, Jonas Süß, Dishant, Aum Oza, Oskar von Stryk - [[pdf]](https://arxiv.org/pdf/2602.19898)</summary>

**Abstract:** In disaster response and situation assessment, robots have great potential in reducing the risks to the safety and health of first responders. As the situations encountered and the required capabilities of the robots deployed in such missions differ wildly and are often not known in advance, heterogeneous fleets of robots are needed to cover a wide range of mission requirements. While UAVs can quickly survey the mission environment, their ability to carry heavy payloads such as sensors and manipulators is limited. UGVs can carry required payloads to assess and manipulate the mission environment, but need to be able to deal with difficult and unstructured terrain such as rubble and stairs. The ability of tracked platforms with articulated arms (flippers) to reconfigure their geometry makes them particularly effective for navigating challenging terrain. In this paper, we present Athena, an open-hardware rescue ground robot research platform with four individually reconfigurable flippers and a reliable low-cost remote emergency stop (E-Stop) solution. A novel mounting solution using an industrial PU belt and tooth inserts allows the replacement and testing of different track profiles. The manipulator with a maximum reach of 1.54m can be used to operate doors, valves, and other objects of interest. Full CAD & PCB files, as well as all low-level software, are released as open-source contributions.

**arXiv ID:** 2602.19898
</details>

<details>
<summary><strong>MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation</strong> - Xun Huang, Shijia Zhao, Yunxiang Wang, Xin Lu, Wanfa Zhang, Rongsheng Qu, Weixin Li, Yunhong Wang, Chenglu Wen - [[pdf]](https://arxiv.org/pdf/2511.10376)</summary>

**Abstract:** Embodied navigation is a fundamental capability for robotic agents operating. Real-world deployment requires open vocabulary generalization and low training overhead, motivating zero-shot methods rather than task-specific RL training. However, existing zero-shot methods that build explicit 3D scene graphs often compress rich visual observations into text-only relations, leading to high construction cost, irreversible loss of visual evidence, and constrained vocabularies. To address these limitations, we introduce the Multi-modal 3D Scene Graph (M3DSG), which preserves visual cues by replacing textual relation

**arXiv ID:** 2511.10376
</details>

<details>
<summary><strong>Never say never: Exploring the effects of available knowledge on agent persuasiveness in controlled physiotherapy motivation dialogues</strong> - Stephan Vonschallen, Rahel Häusler, Theresa Schmiedel, Friederike Eyssel - [[pdf]](https://arxiv.org/pdf/2602.12924)</summary>

**Abstract:** Generative Social Agents (GSAs) are increasingly impacting human users through persuasive means. On the one hand, they might motivate users to pursue personal goals, such as healthier lifestyles. On the other hand, they are associated with potential risks like manipulation and deception, which are induced by limited control over probabilistic agent outputs. However, as GSAs manifest communicative patterns based on available knowledge, their behavior may be regulated through their access to such knowledge. Following this approach, we explored persuasive ChatGPT-generated messages in the context of human-robot physiotherapy motivation. We did so by comparing ChatGPT-generated responses to predefined inputs from a hypothetical physiotherapy patient. In Study 1, we qualitatively analyzed 13 ChatGPT-generated dialogue scripts with varying knowledge configurations regarding persuasive message characteristics. In Study 2, third-party observers (N = 27) rated a selection of these dialogues in terms of the agent's expressiveness, assertiveness, and persuasiveness. Our findings indicate that LLM-based GSAs can adapt assertive and expressive personality traits - significantly enhancing perceived persuasiveness. Moreover, persuasiveness significantly benefited from the availability of information about the patients' age and past profession, mediated by perceived assertiveness and expressiveness. Contextual knowledge about physiotherapy benefits did not significantly impact persuasiveness, possibly because the LLM had inherent knowledge about such benefits even without explicit prompting. Overall, the study highlights the importance of empirically studying behavioral patterns of GSAs, specifically in terms of what information generative AI systems require for consistent and responsible communication.

**arXiv ID:** 2602.12924
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (30 papers)</h2></summary>

<details>
<summary><strong>Hierarchical Reward Design from Language: Enhancing Alignment of Agent Behavior with Human Specifications</strong> - Zhiqin Qian, Ryan Diaz, Sangwon Seo, Vaibhav Unhelkar - [[pdf]](https://arxiv.org/pdf/2602.18582)</summary>

**Abstract:** When training artificial intelligence (AI) to perform tasks, humans often care not only about whether a task is completed but also how it is performed. As AI agents tackle increasingly complex tasks, aligning their behavior with human-provided specifications becomes critical for responsible AI deployment. Reward design provides a direct channel for such alignment by translating human expectations into reward functions that guide reinforcement learning (RL). However, existing methods are often too limited to capture nuanced human preferences that arise in long-horizon tasks. Hence, we introduce Hierarchical Reward Design from Language (HRDL): a problem formulation that extends classical reward design to encode richer behavioral specifications for hierarchical RL agents. We further propose Language to Hierarchical Rewards (L2HR) as a solution to HRDL. Experiments show that AI agents trained with rewards designed via L2HR not only complete tasks effectively but also better adhere to human specifications. Together, HRDL and L2HR advance the research on human-aligned AI agents.

**arXiv ID:** 2602.18582
</details>

<details>
<summary><strong>Many AI Analysts, One Dataset: Navigating the Agentic Data Science Multiverse</strong> - Martin Bertran, Riccardo Fogliato, Zhiwei Steven Wu - [[pdf]](https://arxiv.org/pdf/2602.18710)</summary>

**Abstract:** The conclusions of empirical research depend not only on data but on a sequence of analytic decisions that published results seldom make explicit. Past ``many-analyst" studies have demonstrated this: independent teams testing the same hypothesis on the same dataset regularly reach conflicting conclusions. But such studies require months of coordination among dozens of research groups and are therefore rarely conducted. In this work, we show that fully autonomous AI analysts built on large language models (LLMs) can reproduce a similar structured analytic diversity cheaply and at scale. We task these AI analysts with testing a pre-specified hypothesis on a fixed dataset, varying the underlying model and prompt framing across replicate runs. Each AI analyst independently constructs and executes a full analysis pipeline; an AI auditor then screens each run for methodological validity. Across three datasets spanning experimental and observational designs, AI analyst-produced analyses display wide dispersion in effect sizes, $p$-values, and binary decisions on supporting the hypothesis or not, frequently reversing whether a hypothesis is judged supported. This dispersion is structured: recognizable analytic choices in preprocessing, model specification, and inference differ systematically across LLM and persona conditions. Critically, the effects are \emph{steerable}: reassigning the analyst persona or LLM shifts the distribution of outcomes even after excluding methodologically deficient runs.

**arXiv ID:** 2602.18710
</details>

<details>
<summary><strong>Robust Exploration in Directed Controller Synthesis via Reinforcement Learning with Soft Mixture-of-Experts</strong> - Toshihide Ubukata, Zhiyao Wang, Enhong Mu, Jialong Li, Kenji Tei - [[pdf]](https://arxiv.org/pdf/2602.19244)</summary>

**Abstract:** On-the-fly Directed Controller Synthesis (OTF-DCS) mitigates state-space explosion by incrementally exploring the system and relies critically on an exploration policy to guide search efficiently. Recent reinforcement learning (RL) approaches learn such policies and achieve promising zero-shot generalization from small training instances to larger unseen ones. However, a fundamental limitation is anisotropic generalization, where an RL policy exhibits strong performance only in a specific region of the domain-parameter space while remaining fragile elsewhere due to training stochasticity and trajectory-dependent bias. To address this, we propose a Soft Mixture-of-Experts framework that combines multiple RL experts via a prior-confidence gating mechanism and treats these anisotropic behaviors as complementary specializations. The evaluation on the Air Traffic benchmark shows that Soft-MoE substantially expands the solvable parameter space and improves robustness compared to any single expert.

**arXiv ID:** 2602.19244
</details>

<details>
<summary><strong>ALPACA: A Reinforcement Learning Environment for Medication Repurposing and Treatment Optimization in Alzheimer's Disease</strong> - Nolan Brady, Tom Yeh - [[pdf]](https://arxiv.org/pdf/2602.19298)</summary>

**Abstract:** Evaluating personalized, sequential treatment strategies for Alzheimer's disease (AD) using clinical trials is often impractical due to long disease horizons and substantial inter-patient heterogeneity. To address these constraints, we present the Alzheimer's Learning Platform for Adaptive Care Agents (ALPACA), an open-source, Gym-compatible reinforcement learning (RL) environment for systematically exploring personalized treatment strategies using existing therapies. ALPACA is powered by the Continuous Action-conditioned State Transitions (CAST) model trained on longitudinal trajectories from the Alzheimer's Disease Neuroimaging Initiative (ADNI), enabling medication-conditioned simulation of disease progression under alternative treatment decisions. We show that CAST autoregressively generates realistic medication-conditioned trajectories and that RL policies trained in ALPACA outperform no-treatment and behavior-cloned clinician baselines on memory-related outcomes. Interpretability analyses further indicated that the learned policies relied on clinically meaningful patient features when selecting actions. Overall, ALPACA provides a reusable in silico testbed for studying individualized sequential treatment decision-making for AD.

**arXiv ID:** 2602.19298
</details>

<details>
<summary><strong>IR$^3$: Contrastive Inverse Reinforcement Learning for Interpretable Detection and Mitigation of Reward Hacking</strong> - Mohammad Beigi, Ming Jin, Junshan Zhang, Jiaxin Zhang, Qifan Wang, Lifu Huang - [[pdf]](https://arxiv.org/pdf/2602.19416)</summary>

**Abstract:** Reinforcement Learning from Human Feedback (RLHF) enables powerful LLM alignment but can introduce reward hacking - models exploit spurious correlations in proxy rewards without genuine alignment. Compounding this, the objectives internalized during RLHF remain opaque, making hacking behaviors difficult to detect or correct. We introduce IR3 (Interpretable Reward Reconstruction and Rectification), a framework that reverse-engineers, interprets, and surgically repairs the implicit objectives driving RLHF-tuned models. We propose Contrastive Inverse Reinforcement Learning (C-IRL), which reconstructs the implicit reward function by contrasting paired responses from post-alignment and baseline policies to explain behavioral shifts during RLHF. We then decompose the reconstructed reward via sparse autoencoders into interpretable features, enabling identification of hacking signatures through contribution analysis. Finally, we propose mitigation strategies - clean reward optimization, adversarial shaping, constrained optimization, and feature-guided distillation - that target problematic features while preserving beneficial alignment. Experiments across multiple reward model configurations show that IR3 achieves 0.89 correlation with ground-truth rewards, identifies hacking features with over 90% precision, and significantly reduces hacking behaviors while maintaining capabilities within 3% of the original model.

**arXiv ID:** 2602.19416
</details>

<details>
<summary><strong>Meta-Learning and Meta-Reinforcement Learning - Tracing the Path towards DeepMind's Adaptive Agent</strong> - Björn Hoppmann, Christoph Scholz - [[pdf]](https://arxiv.org/pdf/2602.19837)</summary>

**Abstract:** Humans are highly effective at utilizing prior knowledge to adapt to novel tasks, a capability that standard machine learning models struggle to replicate due to their reliance on task-specific training. Meta-learning overcomes this limitation by allowing models to acquire transferable knowledge from various tasks, enabling rapid adaptation to new challenges with minimal data. This survey provides a rigorous, task-based formalization of meta-learning and meta-reinforcement learning and uses that paradigm to chronicle the landmark algorithms that paved the way for DeepMind's Adaptive Agent, consolidating the essential concepts needed to understand the Adaptive Agent and other generalist approaches.

**arXiv ID:** 2602.19837
</details>

<details>
<summary><strong>ReSyn: Autonomously Scaling Synthetic Environments for Reasoning Models</strong> - Andre He, Nathaniel Weir, Kaj Bostrom, Allen Nie, Darion Cassel, Sam Bayless, Huzefa Rangwala - [[pdf]](https://arxiv.org/pdf/2602.20117)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has emerged as a promising approach for training reasoning language models (RLMs) by leveraging supervision from verifiers. Although verifier implementation is easier than solution annotation for many tasks, existing synthetic data generation methods remain largely solution-centric, while verifier-based methods rely on a few hand-crafted procedural environments. In this work, we scale RLVR by introducing ReSyn, a pipeline that generates diverse reasoning environments equipped with instance generators and verifiers, covering tasks such as constraint satisfaction, algorithmic puzzles, and spatial reasoning. A Qwen2.5-7B-Instruct model trained with RL on ReSyn data achieves consistent gains across reasoning benchmarks and out-of-domain math benchmarks, including a 27\% relative improvement on the challenging BBEH benchmark. Ablations show that verifier-based supervision and increased task diversity both contribute significantly, providing empirical evidence that generating reasoning environments at scale can enhance reasoning abilities in RLMs

**arXiv ID:** 2602.20117
</details>

<details>
<summary><strong>Learning to Remember: End-to-End Training of Memory Agents for Long-Context Reasoning</strong> - Kehao Zhang, Shangtong Gui, Sheng Yang, Wei Chen, Yang Feng - [[pdf]](https://arxiv.org/pdf/2602.18493)</summary>

**Abstract:** Long-context LLMs and Retrieval-Augmented Generation (RAG) systems process information passively, deferring state tracking, contradiction resolution, and evidence aggregation to query time, which becomes brittle under ultra long streams with frequent updates. We propose the Unified Memory Agent (UMA), an end-to-end reinforcement learning framework that unifies memory operations and question answering within a single policy. UMA maintains a dual memory representation: a compact core summary for global context and a structured Memory Bank that supports explicit CRUD (create, update, delete, reorganize) over key value entries, enabling proactive consolidation during streaming. To evaluate long-horizon memory behavior, we introduce Ledger-QA, a diagnostic benchmark for continuous state tracking where answers are latent values derived from accumulated updates rather than lo cal span retrieval. Across 13 datasets spanning Ledger-QA, Test-Time Learning, and Accurate Retrieval, UMA substantially outperforms long-context and RAG baselines on dynamic reasoning and learning tasks while remaining competitive on standard retrieval benchmarks, underscoring the importance of learned, end-to-end memory management.

**arXiv ID:** 2602.18493
</details>

<details>
<summary><strong>Deep Reinforcement Learning for Optimizing Energy Consumption in Smart Grid Systems</strong> - Abeer Alsheikhi, Amirfarhad Farhadi, Azadeh Zamanifar - [[pdf]](https://arxiv.org/pdf/2602.18531)</summary>

**Abstract:** The energy management problem in the context of smart grids is inherently complex due to the interdependencies among diverse system components. Although Reinforcement Learning (RL) has been proposed for solving Optimal Power Flow (OPF) problems, the requirement for iterative interaction with an environment often necessitates computationally expensive simulators, leading to significant sample inefficiency. In this study, these challenges are addressed through the use of Physics-Informed Neural Networks (PINNs), which can replace conventional and costly smart grid simulators. The RL policy learning process is enhanced so that convergence can be achieved in a fraction of the time required by the original environment. The PINN-based surrogate is compared with other benchmark data-driven surrogate models. By incorporating knowledge of the underlying physical laws, the results show that the PINN surrogate is the only approach considered in this context that can obtain a strong RL policy even without access to samples from the true simulator. The results demonstrate that using PINN surrogates can accelerate training by 50% compared to RL training without a surrogate. This approach enables the rapid generation of performance scores similar to those produced by the original simulator.

**arXiv ID:** 2602.18531
</details>

<details>
<summary><strong>Pushing the Limits of Inverse Lithography with Generative Reinforcement Learning</strong> - Haoyu Yang, Haoxing Ren - [[pdf]](https://arxiv.org/pdf/2602.19027)</summary>

**Abstract:** Inverse lithography (ILT) is critical for modern semiconductor manufacturing but suffers from highly non-convex objectives that often trap optimization in poor local minima. Generative AI has been explored to warm-start ILT, yet most approaches train deterministic image-to-image translators to mimic sub-optimal datasets, providing limited guidance for escaping non-convex traps during refinement. We reformulate mask synthesis as conditional sampling: a generator learns a distribution over masks conditioned on the design and proposes multiple candidates. The generator is first pretrained with WGAN plus a reconstruction loss, then fine-tuned using Group Relative Policy Optimization (GRPO) with an ILT-guided imitation loss. At inference, we sample a small batch of masks, run fast batched ILT refinement, evaluate lithography metrics (e.g., EPE, process window), and select the best candidate. On \texttt{LithoBench} dataset, the proposed hybrid framework reduces EPE violations under a 3\,nm tolerance and roughly doubles throughput versus a strong numerical ILT baseline, while improving final mask quality. We also present over 20\% EPE improvement on \texttt{ICCAD13} contest cases with 3$\times$ speedup over the SOTA numerical ILT solver. By learning to propose ILT-friendly initializations, our approach mitigates non-convexity and advances beyond what traditional solvers or GenAI can achieve.

**arXiv ID:** 2602.19027
</details>

<details>
<summary><strong>Online Navigation Planning for Long-term Autonomous Operation of Underwater Gliders</strong> - Victor-Alexandru Darvariu, Charlotte Z. Reed, Jan Stratmann, Bruno Lacerda, Benjamin Allsup, Stephen Woodward, Elizabeth Siddle, Trishna Saeharaseelan, Owain Jones, Dan Jones, Tobias Ferreira, Chloe Baker, Kevin Chaplin, James Kirk, Ashley Morris, Ryan Patmore, Jeff Polton, Charlotte Williams, Alexandra Kokkinaki, Alvaro Lorenzo Lopez, Justin J. H. Buck, Nick Hawes - [[pdf]](https://arxiv.org/pdf/2602.19315)</summary>

**Abstract:** Underwater glider robots have become an indispensable tool for ocean sampling. Although stakeholders are calling for tools to manage increasingly large fleets of gliders, successful autonomous long-term deployments have thus far been scarce, which hints at a lack of suitable methodologies and systems. In this work, we formulate glider navigation planning as a stochastic shortest-path Markov Decision Process and propose a sample-based online planner based on Monte Carlo Tree Search. Samples are generated by a physics-informed simulator that captures uncertain execution of controls and ocean current forecasts while remaining computationally tractable. The simulator parameters are fitted using historical glider data. We integrate these methods into an autonomous command-and-control system for Slocum gliders that enables closed-loop replanning at each surfacing. The resulting system was validated in two field deployments in the North Sea totalling approximately 3 months and 1000 km of autonomous operation. Results demonstrate improved efficiency compared to straight-to-goal navigation and show the practicality of sample-based planning for long-term marine autonomy.

**arXiv ID:** 2602.19315
</details>

<details>
<summary><strong>Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of Evaluation and System Limitations</strong> - Dongming Jiang, Yi Li, Songtao Wei, Jinxin Yang, Ayushi Kishore, Alysa Zhao, Dingyi Kang, Xu Hu, Feng Chen, Qiannan Li, Bingzhe Li - [[pdf]](https://arxiv.org/pdf/2602.19320)</summary>

**Abstract:** Agentic memory systems enable large language model (LLM) agents to maintain state across long interactions, supporting long-horizon reasoning and personalization beyond fixed context windows. Despite rapid architectural development, the empirical foundations of these systems remain fragile: existing benchmarks are often underscaled, evaluation metrics are misaligned with semantic utility, performance varies significantly across backbone models, and system-level costs are frequently overlooked. This survey presents a structured analysis of agentic memory from both architectural and system perspectives. We first introduce a concise taxonomy of MAG systems based on four memory structures. Then, we analyze key pain points limiting current systems, including benchmark saturation effects, metric validity and judge sensitivity, backbone-dependent accuracy, and the latency and throughput overhead introduced by memory maintenance. By connecting the memory structure to empirical limitations, this survey clarifies why current agentic memory systems often underperform their theoretical promise and outlines directions for more reliable evaluation and scalable system design.

**arXiv ID:** 2602.19320
</details>

<details>
<summary><strong>Stable Deep Reinforcement Learning via Isotropic Gaussian Representations</strong> - Ali Saheb, Johan Obando-Ceron, Aaron Courville, Pouya Bashivan, Pablo Samuel Castro - [[pdf]](https://arxiv.org/pdf/2602.19373)</summary>

**Abstract:** Deep reinforcement learning systems often suffer from unstable training dynamics due to non-stationarity, where learning objectives and data distributions evolve over time. We show that under non-stationary targets, isotropic Gaussian embeddings are provably advantageous. In particular, they induce stable tracking of time-varying targets for linear readouts, achieve maximal entropy under a fixed variance budget, and encourage a balanced use of all representational dimensions--all of which enable agents to be more adaptive and stable. Building on this insight, we propose the use of Sketched Isotropic Gaussian Regularization for shaping representations toward an isotropic Gaussian distribution during training. We demonstrate empirically, over a variety of domains, that this simple and computationally inexpensive method improves performance under non-stationarity while reducing representation collapse, neuron dormancy, and training instability.

**arXiv ID:** 2602.19373
</details>

<details>
<summary><strong>Hilbert-Augmented Reinforcement Learning for Scalable Multi-Robot Coverage and Exploration</strong> - Tamil Selvan Gurunathan, Aryya Gangopadhyay - [[pdf]](https://arxiv.org/pdf/2602.19400)</summary>

**Abstract:** We present a coverage framework that integrates Hilbert space-filling priors into decentralized multi-robot learning and execution. We augment DQN and PPO with Hilbert-based spatial indices to structure exploration and reduce redundancy in sparse-reward environments, and we evaluate scalability in multi-robot grid coverage. We further describe a waypoint interface that converts Hilbert orderings into curvature-bounded, time-parameterized SE(2) trajectories (planar (x, y, {\theta})), enabling onboard feasibility on resource-constrained robots. Experiments show improvements in coverage efficiency, redundancy, and convergence speed over DQN/PPO baselines. In addition, we validate the approach on a Boston Dynamics Spot legged robot, executing the generated trajectories in indoor environments and observing reliable coverage with low redundancy. These results indicate that geometric priors improve autonomy and scalability for swarm and legged robotics.

**arXiv ID:** 2602.19400
</details>

<details>
<summary><strong>Effects of Property Recovery Incentives and Social Interaction on Self-Evacuation Decisions in Natural Disasters: An Agent-Based Modelling Approach</strong> - Made Krisnanda, Raymond Chiong, Yang Yang, Kirill Glavatskiy - [[pdf]](https://arxiv.org/pdf/2602.19639)</summary>

**Abstract:** Understanding evacuation decision-making behaviour is one of the key components for designing disaster mitigation policies. This study investigates how communications between household agents in a community influence self-evacuation decisions. We develop an agent-based model that simulates household agents' decisions to evacuate or stay. These agents interact within the framework of evolutionary game theory, effectively competing for limited shared resources, which include property recovery funds and coordination services. We explore four scenarios that model different prioritisations of access to government-provided incentives. We discover that the impact of the incentive diminishes both with increasing funding value and the household agent prioritisation, indicating that there is an optimal level of government support beyond which further increases become impractical. Furthermore, the overall evacuation rate depends on the structure of the underlying social network, showing discontinuous jumps when the prioritisation moves across the node degree. We identify the so-called "community influencers", prioritisation of whom significantly increases the overall evacuation rate. In contrast, prioritising household agents with low connectivity may actually impede collective evacuation. These findings demonstrate the importance of social connectivity between household agents. The results of this study are useful for designing optimal government policies to incentivise and prioritise community evacuation under limited resources.

**arXiv ID:** 2602.19639
</details>

<details>
<summary><strong>AgenticRAGTracer: A Hop-Aware Benchmark for Diagnosing Multi-Step Retrieval Reasoning in Agentic RAG</strong> - Qijie You, Wenkai Yu, Wentao Zhang - [[pdf]](https://arxiv.org/pdf/2602.19127)</summary>

**Abstract:** With the rapid advancement of agent-based methods in recent years, Agentic RAG has undoubtedly become an important research direction. Multi-hop reasoning, which requires models to engage in deliberate thinking and multi-step interaction, serves as a critical testbed for assessing such capabilities. However, existing benchmarks typically provide only final questions and answers, while lacking the intermediate hop-level questions that gradually connect atomic questions to the final multi-hop query. This limitation prevents researchers from analyzing at which step an agent fails and restricts more fine-grained evaluation of model capabilities. Moreover, most current benchmarks are manually constructed, which is both time-consuming and labor-intensive, while also limiting scalability and generalization. To address these challenges, we introduce AgenticRAGTracer, the first Agentic RAG benchmark that is primarily constructed automatically by large language models and designed to support step-by-step validation. Our benchmark spans multiple domains, contains 1,305 data points, and has no overlap with existing mainstream benchmarks. Extensive experiments demonstrate that even the best large language models perform poorly on our dataset. For instance, GPT-5 attains merely 22.6\% EM accuracy on the hardest portion of our dataset. Hop-aware diagnosis reveals that failures are primarily driven by distorted reasoning chains -- either collapsing prematurely or wandering into over-extension. This highlights a critical inability to allocate steps consistent with the task's logical structure, providing a diagnostic dimension missing in traditional evaluations. We believe our work will facilitate research in Agentic RAG and inspire further meaningful progress in this area. Our code and data are available at this https URL.

**arXiv ID:** 2602.19127
</details>

<details>
<summary><strong>How to Train Your Deep Research Agent? Prompt, Reward, and Policy Optimization in Search-R1</strong> - Yinuo Xu, Shuo Lu, Jianjie Cheng, Meng Wang, Qianlong Xie, Xingxing Wang, Ran He, Jian Liang - [[pdf]](https://arxiv.org/pdf/2602.19526)</summary>

**Abstract:** Deep Research agents tackle knowledge-intensive tasks through multi-round retrieval and decision-oriented generation. While reinforcement learning (RL) has been shown to improve performance in this paradigm, its contributions remain underexplored. To fully understand the role of RL, we conduct a systematic study along three decoupled dimensions: prompt template, reward function, and policy optimization. Our study reveals that: 1) the Fast Thinking template yields greater stability and better performance than the Slow Thinking template used in prior work; 2) the F1-based reward underperforms the EM due to training collapse driven by answer avoidance; this can be mitigated by incorporating action-level penalties, ultimately surpassing EM; 3) REINFORCE outperforms PPO while requiring fewer search actions, whereas GRPO shows the poorest stability among policy optimization methods. Building on these insights, we then introduce Search-R1++, a strong baseline that improves the performance of Search-R1 from 0.403 to 0.442 (Qwen2.5-7B) and 0.289 to 0.331 (Qwen2.5-3B). We hope that our findings can pave the way for more principled and reliable RL training strategies in Deep Research systems.

**arXiv ID:** 2602.19526
</details>

<details>
<summary><strong>AgenticSum: An Agentic Inference-Time Framework for Faithful Clinical Text Summarization</strong> - Fahmida Liza Piya, Rahmatollah Beheshti - [[pdf]](https://arxiv.org/pdf/2602.20040)</summary>

**Abstract:** Large language models (LLMs) offer substantial promise for automating clinical text summarization, yet maintaining factual consistency remains challenging due to the length, noise, and heterogeneity of clinical documentation. We present AgenticSum, an inference-time, agentic framework that separates context selection, generation, verification, and targeted correction to reduce hallucinated content. The framework decomposes summarization into coordinated stages that compress task-relevant context, generate an initial draft, identify weakly supported spans using internal attention grounding signals, and selectively revise flagged content under supervisory control. We evaluate AgenticSum on two public datasets, using reference-based metrics, LLM-as-a-judge assessment, and human evaluation. Across various measures, AgenticSum demonstrates consistent improvements compared to vanilla LLMs and other strong baselines. Our results indicate that structured, agentic design with targeted correction offers an effective inference time solution to improve clinical note summarization using LLMs.

**arXiv ID:** 2602.20040
</details>

<details>
<summary><strong>Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning</strong> - Yimeng Zhang, Tian Wang, Jiri Gesi, Ziyi Wang, Yuxuan Lu, Jiacheng Lin, Sinong Zhan, Vianne Gao, Ruochen Jiao, Junze Liu, Kun Qian, Yuxin Tang, Ran Xue, Houyu Zhang, Qingjun Cui, Yufan Guo, Dakuo Wang - [[pdf]](https://arxiv.org/pdf/2507.17842)</summary>

**Abstract:** Large Language Models (LLMs) have recently demonstrated strong potential in generating 'believable human-like' behavior in web environments. Prior work has explored augmenting training data with LLM-synthesized rationales and applying supervised fine-tuning (SFT) to enhance reasoning ability, which in turn can improve downstream action prediction. However, the performance of such approaches remains inherently bounded by the reasoning capabilities of the model used to generate the rationales. In this paper, we introduce Shop-R1, a novel reinforcement learning (RL) framework aimed at enhancing the reasoning ability of LLMs for simulation of real human behavior in online shopping environments. Specifically, Shop-R1 decomposes the human behavior simulation task into two stages: rationale generation and action prediction, each guided by distinct reward signals. For rationale generation, we leverage internal model signals (e.g., logit distributions) to guide the reasoning process in a self-supervised manner. For action prediction, we propose a hierarchical reward structure with difficulty-aware scaling to prevent reward hacking and enable fine-grained reward assignment. This design evaluates both high-level action types and the correctness of fine-grained sub-action details (attributes and values), rewarding outputs proportionally to their difficulty. Experimental results show that our method achieves a relative improvement of over 65% compared to the baseline. The project page is available at this https URL.

**arXiv ID:** 2507.17842
</details>

<details>
<summary><strong>Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning</strong> - Ran Xu, Jingjing Chen, Jiayu Ye, Yu Wu, Jun Yan, Carl Yang, Hongkun Yu - [[pdf]](https://arxiv.org/pdf/2510.23038)</summary>

**Abstract:** Large Language Models (LLMs) are widely used as judges to evaluate response quality, providing a scalable alternative to human evaluation. However, most LLM judges operate solely on intrinsic text-based reasoning, limiting their ability to verify complex constraints or perform accurate computation. Motivated by the success of tool-integrated reasoning (TIR) in numerous tasks, we propose TIR-Judge, an end-to-end RL framework for training LLM judges that integrates a code executor for precise evaluation. TIR-Judge is built on three principles: (i) diverse training across verifiable and non-verifiable domains, (ii) flexible judgment formats (pointwise, pairwise, listwise), and (iii) iterative RL that bootstraps directly from the initial model without distillation. On seven public benchmarks, TIR-Judge surpasses strong reasoning-based judges by up to 6.4% (pointwise) and 7.7% (pairwise), and achieves listwise performance comparable to Claude-Opus-4 despite having only 8B parameters. Remarkably, TIR-Judge-Zero - trained entirely without distilled judge trajectories, matches the performance of distilled variants, demonstrating that tool-augmented judges can self-evolve through iterative reinforcement learning.

**arXiv ID:** 2510.23038
</details>

<details>
<summary><strong>OmniRAG-Agent: Agentic Omnimodal Reasoning for Low-Resource Long Audio-Video Question Answering</strong> - Yifan Zhu, Xinyu Mu, Tao Feng, Zhonghong Ou, Yuning Gong, Haoran Luo - [[pdf]](https://arxiv.org/pdf/2602.03707)</summary>

**Abstract:** Long-horizon omnimodal question answering answers questions by reasoning over text, images, audio, and video. Despite recent progress on OmniLLMs, low-resource long audio-video QA still suffers from costly dense encoding, weak fine-grained retrieval, limited proactive planning, and no clear end-to-end optimization. To address these issues, we propose OmniRAG-Agent, an agentic omnimodal QA method for budgeted long audio-video reasoning. It builds an image-audio retrieval-augmented generation module that lets an OmniLLM fetch short, relevant frames and audio snippets from external banks. Moreover, it uses an agent loop that plans, calls tools across turns, and merges retrieved evidence to answer complex queries. Furthermore, we apply group relative policy optimization to jointly improve tool use and answer quality over time. Experiments on OmniVideoBench, WorldSense, and Daily-Omni show that OmniRAG-Agent consistently outperforms prior methods under low-resource settings and achieves strong results, with ablations validating each component.

**arXiv ID:** 2602.03707
</details>

<details>
<summary><strong>STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens</strong> - Shiqi Liu, Zeyu He, Guojian Zhan, Letian Tao, Zhilong Zheng, Jiang Wu, Yinuo Wang, Yang Guan, Kehua Sheng, Bo Zhang, Keqiang Li, Jingliang Duan, Shengbo Eben Li - [[pdf]](https://arxiv.org/pdf/2602.15620)</summary>

**Abstract:** Reinforcement Learning (RL) has significantly improved large language model reasoning, but existing RL fine-tuning methods rely heavily on heuristic techniques such as entropy regularization and reweighting to maintain stability. In practice, they often suffer from late-stage performance collapse, leading to degraded reasoning quality and unstable training. Our analysis shows that the magnitude of token-wise policy gradients in RL is negatively correlated with token probability and local policy entropy. We find that training instability can be caused by a tiny fraction of tokens, approximately 0.01%, which we term spurious tokens. When such tokens appear in correct responses, they contribute little to the reasoning outcome but inherit the full sequence-level reward, leading to abnormally amplified gradient updates. To mitigate this instability, we design an S2T (silencing spurious tokens) mechanism to efficiently identify spurious tokens through characteristic signals with low probability, low entropy, and positive advantage, and then suppress their gradient perturbations during optimization. Incorporating this mechanism into a group-based objective, we propose Spurious-Token-Aware Policy Optimization (STAPO), which promotes stable and effective large-scale model refinement. Across six mathematical reasoning benchmarks using Qwen 1.7B, 8B, and 14B base models, STAPO consistently demonstrates superior entropy stability and achieves an average performance improvement of 7.13% ($\rho_{\mathrm{T}}$=1.0, top-p=1.0) and 3.69% ($\rho_{\mathrm{T}}$=0.7, top-p=0.9) over GRPO, 20-Entropy, and JustRL.

**arXiv ID:** 2602.15620
</details>

<details>
<summary><strong>Collaborative Document Editing with Multiple Users and AI Agents</strong> - Florian Lehmann, Krystsina Shauchenka, Daniel Buschek - [[pdf]](https://arxiv.org/pdf/2509.11826)</summary>

**Abstract:** Current AI writing support tools are largely designed for individuals, complicating collaboration when co-writers must leave the shared workspace to use AI and then communicate and reintegrate results. We propose integrating AI agents directly into collaborative writing environments. Our prototype makes AI use visible to all users through two new shared objects: user-defined agent profiles and tasks. Agent responses appear in the familiar comment feature. In a user study (N=30), 14 teams worked on writing projects during one week. Interaction logs and interviews show that teams incorporated agents into existing norms of authorship, control, and coordination, rather than treating them as team members. Agent profiles were viewed as personal territory, while created agents and outputs became shared resources. We discuss implications for team-based AI interaction, highlighting opportunities and boundaries for treating AI as a shared resource in collaborative work.

**arXiv ID:** 2509.11826
</details>

<details>
<summary><strong>VariBASed: Variational Bayes-Adaptive Sequential Monte-Carlo Planning for Deep Reinforcement Learning</strong> - Joery A. de Vries, Jinke He, Yaniv Oren, Pascal R. van der Vaart, Mathijs M. de Weerdt, Matthijs T. J. Spaan - [[pdf]](https://arxiv.org/pdf/2602.18857)</summary>

**Abstract:** Optimally trading-off exploration and exploitation is the holy grail of reinforcement learning as it promises maximal data-efficiency for solving any task. Bayes-optimal agents achieve this, but obtaining the belief-state and performing planning are both typically intractable. Although deep learning methods can greatly help in scaling this computation, existing methods are still costly to train. To accelerate this, this paper proposes a variational framework for learning and planning in Bayes-adaptive Markov decision processes that coalesces variational belief learning, sequential Monte-Carlo planning, and meta-reinforcement learning. In a single-GPU setup, our new method VariBASeD exhibits favorable scaling to larger planning budgets, improving sample- and runtime-efficiency over prior methods.

**arXiv ID:** 2602.18857
</details>

<details>
<summary><strong>Advantage-based Temporal Attack in Reinforcement Learning</strong> - Shenghong He - [[pdf]](https://arxiv.org/pdf/2602.19582)</summary>

**Abstract:** Extensive research demonstrates that Deep Reinforcement Learning (DRL) models are susceptible to adversarially constructed inputs (i.e., adversarial examples), which can mislead the agent to take suboptimal or unsafe actions. Recent methods improve attack effectiveness by leveraging future rewards to guide adversarial perturbation generation over sequential time steps (i.e., reward-based attacks). However, these methods are unable to capture dependencies between different time steps in the perturbation generation process, resulting in a weak temporal correlation between the current perturbation and previous this http URL this paper, we propose a novel method called Advantage-based Adversarial Transformer (AAT), which can generate adversarial examples with stronger temporal correlations (i.e., time-correlated adversarial examples) to improve the attack performance. AAT employs a multi-scale causal self-attention (MSCSA) mechanism to dynamically capture dependencies between historical information from different time periods and the current state, thus enhancing the correlation between the current perturbation and the previous perturbation. Moreover, AAT introduces a weighted advantage mechanism, which quantifies the effectiveness of a perturbation in a given state and guides the generation process toward high-performance adversarial examples by sampling high-advantage regions. Extensive experiments demonstrate that the performance of AAT matches or surpasses mainstream adversarial attack baselines on Atari, DeepMind Control Suite and Google football tasks.

**arXiv ID:** 2602.19582
</details>

<details>
<summary><strong>Uncertainty-Aware Rank-One MIMO Q Network Framework for Accelerated Offline Reinforcement Learning</strong> - Thanh Nguyen, Tung Luu, Tri Ton, Sungwoong Kim, Chang D. Yoo - [[pdf]](https://arxiv.org/pdf/2602.19917)</summary>

**Abstract:** Offline reinforcement learning (RL) has garnered significant interest due to its safe and easily scalable paradigm. However, training under this paradigm presents its own challenge: the extrapolation error stemming from out-of-distribution (OOD) data. Existing methodologies have endeavored to address this issue through means like penalizing OOD Q-values or imposing similarity constraints on the learned policy and the behavior policy. Nonetheless, these approaches are often beset by limitations such as being overly conservative in utilizing OOD data, imprecise OOD data characterization, and significant computational overhead. To address these challenges, this paper introduces an Uncertainty-Aware Rank-One Multi-Input Multi-Output (MIMO) Q Network framework. The framework aims to enhance Offline Reinforcement Learning by fully leveraging the potential of OOD data while still ensuring efficiency in the learning process. Specifically, the framework quantifies data uncertainty and harnesses it in the training losses, aiming to train a policy that maximizes the lower confidence bound of the corresponding Q-function. Furthermore, a Rank-One MIMO architecture is introduced to model the uncertainty-aware Q-function, \TP{offering the same ability for uncertainty quantification as an ensemble of networks but with a cost nearly equivalent to that of a single network}. Consequently, this framework strikes a harmonious balance between precision, speed, and memory efficiency, culminating in improved overall performance. Extensive experimentation on the D4RL benchmark demonstrates that the framework attains state-of-the-art performance while remaining computationally efficient. By incorporating the concept of uncertainty quantification, our framework offers a promising avenue to alleviate extrapolation errors and enhance the efficiency of offline RL.

**arXiv ID:** 2602.19917
</details>

<details>
<summary><strong>Gait Asymmetry from Unilateral Weakness and Improvement With Ankle Assistance: a Reinforcement Learning based Simulation Study</strong> - Yifei Yuan, Ghaith Androwis, Xianlian Zhou - [[pdf]](https://arxiv.org/pdf/2602.18862)</summary>

**Abstract:** Unilateral muscle weakness often leads to asymmetric gait, disrupting interlimb coordination and stance timing. This study presents a reinforcement learning (RL) based musculoskeletal simulation framework to (1) quantify how progressive unilateral muscle weakness affects gait symmetry and (2) evaluate whether ankle exoskeleton assistance can improve gait symmetry under impaired conditions. The overarching goal is to establish a simulation- and learning-based workflow that supports early controller development prior to patient experiments. Asymmetric gait was induced by reducing right-leg muscle strength to 75%, 50%, and 25% of baseline. Gait asymmetry was quantified using toe-off timing, peak contact forces, and joint-level symmetry metrics. Increasing weakness produced progressively larger temporal and kinematic asymmetry, most pronounced at the ankle. Ankle range of motion symmetry degraded from near-symmetric behavior at 100% strength (symmetry index, SI = +6.4%; correlation r=0.974) to severe asymmetry at 25% strength (SI = -47.1%, r=0.889), accompanied by a load shift toward the unimpaired limb. At 50% strength, ankle exoskeleton assistance improved kinematic symmetry relative to the unassisted impaired condition, reducing the magnitude of ankle SI from 25.8% to 18.5% and increasing ankle correlation from r=0.948 to 0.966, although peak loading remained biased toward the unimpaired side. Overall, this framework supports controlled evaluation of impairment severity and assistive strategies, and provides a basis for future validation in human experiments.

**arXiv ID:** 2602.18862
</details>

<details>
<summary><strong>A Primer on SO(3) Action Representations in Deep Reinforcement Learning</strong> - Martin Schuck, Sherif Samy, Angela P. Schoellig - [[pdf]](https://arxiv.org/pdf/2510.11103)</summary>

**Abstract:** Many robotic control tasks require policies to act on orientations, yet the geometry of SO(3) makes this nontrivial. Because SO(3) admits no global, smooth, minimal parameterization, common representations such as Euler angles, quaternions, rotation matrices, and Lie algebra coordinates introduce distinct constraints and failure modes. While these trade-offs are well studied for supervised learning, their implications for actions in reinforcement learning remain unclear. We systematically evaluate SO(3) action representations across three standard continuous control algorithms, PPO, SAC, and TD3, under dense and sparse rewards. We compare how representations shape exploration, interact with entropy regularization, and affect training stability through empirical studies and analyze the implications of different projections for obtaining valid rotations from Euclidean network outputs. Across a suite of robotics benchmarks, we quantify the practical impact of these choices and distill simple, implementation-ready guidelines for selecting and using rotation actions. Our results highlight that representation-induced geometry strongly influences exploration and optimization and show that representing actions as tangent vectors in the local frame yields the most reliable results across algorithms. The project webpage and code are available at this http URL primer.

**arXiv ID:** 2510.11103
</details>

<details>
<summary><strong>Multistep Quasimetric Learning for Scalable Goal-conditioned Reinforcement Learning</strong> - Bill Chunyuan Zheng, Vivek Myers, Benjamin Eysenbach, Sergey Levine - [[pdf]](https://arxiv.org/pdf/2511.07730)</summary>

**Abstract:** Learning how to reach goals in an environment is a longstanding challenge in AI, yet reasoning over long horizons remains a challenge for modern methods. The key question is how to estimate the temporal distance between pairs of observations. While temporal difference methods leverage local updates to provide optimality guarantees, they often perform worse than Monte Carlo methods that perform global updates (e.g., with multi-step returns), which lack such guarantees. We show how these approaches can be integrated into a practical offline GCRL method that fits a quasimetric distance using a multistep Monte-Carlo return. We show our method outperforms existing offline GCRL methods on long-horizon simulated tasks with up to 4000 steps, even with visual observations. We also demonstrate that our method can enable stitching in the real-world robotic manipulation domain (Bridge setup). Our approach is the first end-to-end offline GCRL method that enables multistep stitching in this real-world manipulation domain from an unlabeled offline dataset of visual observations and demonstrate robust horizon generalization.

**arXiv ID:** 2511.07730
</details>

<details>
<summary><strong>The Path to Conversational AI Tutors: Integrating Tutoring Best Practices and Targeted Technologies to Produce Scalable AI Agents</strong> - Kirk Vanacore, Ryan S. Baker, Avery H. Closser, Jeremy Roschelle - [[pdf]](https://arxiv.org/pdf/2602.19303)</summary>

**Abstract:** The emergence of generative AI has accelerated the development of conversational tutoring systems that interact with students through natural language dialogue. Unlike prior intelligent tutoring systems (ITS), which largely function as adaptive and interactive problem sets with feedback and hints, conversational tutors hold the potential to simulate high-quality human tutoring by engaging with students' thoughts, questions, and misconceptions in real time. While some previous ITS, such as AutoTutor, could respond conversationally, they were expensive to author and lacked a full range of conversational ability. Generative AI has changed the capacity of ITS to engage conversationally. However, realizing the full potential of conversational tutors requires careful consideration of what research on human tutoring and ITS has already established, while also unpacking what new research will be needed. This paper synthesizes tenets of successful human tutoring, lessons learned from legacy ITS, and emerging work on conversational AI tutors. We use a keep, change, center, study framework for guiding the design of conversational tutoring. We argue that systems should keep proven methods from prior ITS, such as knowledge tracing and affect detection; change how tutoring is delivered by leveraging generative AI for dynamic content generation and dialogic scaffolding; and center opportunities for meaning-making, student agency, and granular diagnosis of reasoning. Finally, we identify areas requiring further study, including efficacy testing, student experience, and integration with human instruction. By synthesizing insights from human tutoring, legacy ITS, and emerging generative AI technologies, this paper outlines a research agenda for developing conversational tutors that are scalable, pedagogically effective, and responsive to the social and motivational dimensions of learning.

**arXiv ID:** 2602.19303
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
