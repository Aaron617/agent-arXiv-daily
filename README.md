# Agent arXiv Daily

**Last Updated:** 2026-02-13 03:20:23

**Total Papers:** 94

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (5 papers)</h2></summary>

<details>
<summary><strong>PhyNiKCE: A Neurosymbolic Agentic Framework for Autonomous Computational Fluid Dynamics</strong> - E Fan, Lisong Shi, Zhengtong Li, Chih-yung Wen - [[pdf]](https://arxiv.org/pdf/2602.11666)</summary>

**Abstract:** The deployment of autonomous agents for Computational Fluid Dynamics (CFD), is critically limited by the probabilistic nature of Large Language Models (LLMs), which struggle to enforce the strict conservation laws and numerical stability required for physics-based simulations. Reliance on purely semantic Retrieval Augmented Generation (RAG) often leads to "context poisoning," where agents generate linguistically plausible but physically invalid configurations due to a fundamental Semantic-Physical Disconnect. To bridge this gap, this work introduces PhyNiKCE (Physical and Numerical Knowledgeable Context Engineering), a neurosymbolic agentic framework for trustworthy engineering. Unlike standard black-box agents, PhyNiKCE decouples neural planning from symbolic validation. It employs a Symbolic Knowledge Engine that treats simulation setup as a Constraint Satisfaction Problem, rigidly enforcing physical constraints via a Deterministic RAG Engine with specialized retrieval strategies for solvers, turbulence models, and boundary conditions. Validated through rigorous OpenFOAM experiments on practical, non-tutorial CFD tasks using Gemini-2.5-Pro/Flash, PhyNiKCE demonstrates a 96% relative improvement over state-of-the-art baselines. Furthermore, by replacing trial-and-error with knowledge-driven initialization, the framework reduced autonomous self-correction loops by 59% while simultaneously lowering LLM token consumption by 17%. These results demonstrate that decoupling neural generation from symbolic constraint enforcement significantly enhances robustness and efficiency. While validated on CFD, this architecture offers a scalable, auditable paradigm for Trustworthy Artificial Intelligence in broader industrial automation.

**arXiv ID:** 2602.11666
</details>

<details>
<summary><strong>HybridRAG: A Practical LLM-based ChatBot Framework based on Pre-Generated Q&A over Raw Unstructured Documents</strong> - Sungmoon Kim, Hyuna Jeon, Dahye Kim, Mingyu Kim, Dong-Kyu Chae, Jiwoong Kim - [[pdf]](https://arxiv.org/pdf/2602.11156)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) has emerged as a powerful approach for grounding Large Language Model (LLM)-based chatbot responses on external knowledge. However, existing RAG studies typically assume well-structured textual sources (e.g. Wikipedia or curated datasets) and perform retrieval and generation at query time, which can limit their applicability in real-world chatbot scenarios. In this paper, we present HybridRAG, a novel and practical RAG framework towards more accurate and faster chatbot responses. First, HybridRAG ingests raw, unstructured PDF documents containing complex layouts (text, tables, figures) via Optical Character Recognition (OCR) and layout analysis, and convert them into hierarchical text chunks. Then, it pre-generates a plausible question-answer (QA) knowledge base from the organized chunks using an LLM. At query time, user questions are matched against this QA bank to retrieve immediate answers when possible, and only if no suitable QA match is found does our framework fall back to an on-the-fly response generation. Experiments on OHRBench demonstrate that our HybridRAG provides higher answer quality and lower latency compared to a standard RAG baseline. We believe that HybridRAG could be a practical solution for real-world chatbot applications that must handle large volumes of unstructured documents and lots of users under limited computational resources.

**arXiv ID:** 2602.11156
</details>

<details>
<summary><strong>Choose Your Agent: Tradeoffs in Adopting AI Advisors, Coaches, and Delegates in Multi-Party Negotiation</strong> - Kehang Zhu, Lithium Thain, Vivian Tsai, James Wexler, Crystal Qian - [[pdf]](https://arxiv.org/pdf/2602.12089)</summary>

**Abstract:** As AI usage becomes more prevalent in social contexts, understanding agent-user interaction is critical to designing systems that improve both individual and group outcomes. We present an online behavioral experiment (N = 243) in which participants play three multi-turn bargaining games in groups of three. Each game, presented in randomized order, grants \textit{access to} a single LLM assistance modality: proactive recommendations from an \textit{Advisor}, reactive feedback from a \textit{Coach}, or autonomous execution by a \textit{Delegate}; all modalities are powered by an underlying LLM that achieves superhuman performance in an all-agent environment. On each turn, participants privately decide whether to act manually or use the AI modality available in that game. Despite preferring the \textit{Advisor} modality, participants achieve the highest mean individual gains with the \textit{Delegate}, demonstrating a preference-performance misalignment. Moreover, delegation generates positive externalities; even non-adopting users in \textit{access-to-delegate} treatment groups benefit by receiving higher-quality offers. Mechanism analysis reveals that the \textit{Delegate} agent acts as a market maker, injecting rational, Pareto-improving proposals that restructure the trading environment. Our research reveals a gap between agent capabilities and realized group welfare. While autonomous agents can exhibit super-human strategic performance, their impact on realized welfare gains can be constrained by interfaces, user perceptions, and adoption barriers. Assistance modalities should be designed as mechanisms with endogenous participation; adoption-compatible interaction rules are a prerequisite to improving human welfare with automated assistance.

**arXiv ID:** 2602.12089
</details>

<details>
<summary><strong>Advancing AI Trustworthiness Through Patient Simulation: Risk Assessment of Conversational Agents for Antidepressant Selection</strong> - Md Tanvir Rouf Shawon, Mohammad Sabik Irbaz, Hadeel R. A. Elyazori, Keerti Reddy Resapu, Yili Lin, Vladimir Franzuela Cardenas, Farrokh Alemi, Kevin Lybarger - [[pdf]](https://arxiv.org/pdf/2602.11391)</summary>

**Abstract:** Objective: This paper introduces a patient simulator designed to enable scalable, automated evaluation of healthcare conversational agents. The simulator generates realistic, controllable patient interactions that systematically vary across medical, linguistic, and behavioral dimensions, allowing annotators and an independent AI judge to assess agent performance, identify hallucinations and inaccuracies, and characterize risk patterns across diverse patient populations. Methods: The simulator is grounded in the NIST AI Risk Management Framework and integrates three profile components reflecting different dimensions of patient variation: (1) medical profiles constructed from electronic health records in the All of Us Research Program; (2) linguistic profiles modeling variation in health literacy and condition-specific communication patterns; and (3) behavioral profiles representing empirically observed interaction patterns, including cooperation, distraction, and adversarial engagement. We evaluated the simulator's effectiveness in identifying errors in an AI decision aid for antidepressant selection. Results: We generated 500 conversations between the patient simulator and the AI decision aid across systematic combinations of five linguistic and three behavioral profiles. Human annotators assessed 1,787 medical concepts across 100 conversations, achieving high agreement (F1=0.94, \k{appa}=0.73), and the LLM judge achieved comparable agreement with human annotators (F1=0.94, \k{appa}=0.78; paired bootstrap p=0.21). The simulator revealed a monotonic degradation in AI decision aid performance across the health literacy spectrum: rank-one concept retrieval accuracy increased from 47.9% for limited health literacy to 69.1% for functional and 81.6% for proficient.

**arXiv ID:** 2602.11391
</details>

<details>
<summary><strong>6G Empowering Future Robotics: A Vision for Next-Generation Autonomous Systems</strong> - Mona Ghassemian, Andrés Meseguer Valenzuela, Ana Garcia Armada, Dejan Vukobratovic, Periklis Chatzimisios, Kaspar Althoefer, Ranga Rao Venkatesha Prasad - [[pdf]](https://arxiv.org/pdf/2602.12246)</summary>

**Abstract:** The convergence of robotics and next-generation communication is a critical driver of technological advancement. As the world transitions from 5G to 6G, the foundational capabilities of wireless networks are evolving to support increasingly complex and autonomous robotic systems. This paper examines the transformative impact of 6G on enhancing key robotics functionalities. It provides a systematic mapping of IMT-2030 key performance indicators to robotic functional blocks including sensing, perception, cognition, actuation and self-learning. Building upon this mapping, we propose a high-level architectural framework integrating robotic, intelligent, and network service planes, underscoring the need for a holistic approach. As an example use case, we present a real-time, dynamic safety framework enabled by IMT-2030 capabilities for safe and efficient human-robot collaboration in shared spaces.

**arXiv ID:** 2602.12246
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (15 papers)</h2></summary>

<details>
<summary><strong>TRACER: Trajectory Risk Aggregation for Critical Episodes in Agentic Reasoning</strong> - Sina Tayebati, Divake Kumar, Nastaran Darabi, Davide Ettori, Ranganath Krishnan, Amit Ranjan Trivedi - [[pdf]](https://arxiv.org/pdf/2602.11409)</summary>

**Abstract:** Estimating uncertainty for AI agents in real-world multi-turn tool-using interaction with humans is difficult because failures are often triggered by sparse critical episodes (e.g., looping, incoherent tool use, or user-agent miscoordination) even when local generation appears confident. Existing uncertainty proxies focus on single-shot text generation and therefore miss these trajectory-level breakdown signals. We introduce TRACER, a trajectory-level uncertainty metric for dual-control Tool-Agent-User interaction. TRACER combines content-aware surprisal with situational-awareness signals, semantic and lexical repetition, and tool-grounded coherence gaps, and aggregates them using a tail-focused risk functional with a MAX-composite step risk to surface decisive anomalies. We evaluate TRACER on $\tau^2$-bench by predicting task failure and selective task execution. To this end, TRACER improves AUROC by up to 37.1% and AUARC by up to 55% over baselines, enabling earlier and more accurate detection of uncertainty in complex conversational tool-use settings. Our code and benchmark are available at this https URL.

**arXiv ID:** 2602.11409
</details>

<details>
<summary><strong>LawThinker: A Deep Research Legal Agent in Dynamic Environments</strong> - Xinyu Yang, Chenlong Deng, Tongyu Wen, Binyu Xie, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2602.12056)</summary>

**Abstract:** Legal reasoning requires not only correct outcomes but also procedurally compliant reasoning processes. However, existing methods lack mechanisms to verify intermediate reasoning steps, allowing errors such as inapplicable statute citations to propagate undetected through the reasoning chain. To address this, we propose LawThinker, an autonomous legal research agent that adopts an Explore-Verify-Memorize strategy for dynamic judicial environments. The core idea is to enforce verification as an atomic operation after every knowledge exploration step. A DeepVerifier module examines each retrieval result along three dimensions of knowledge accuracy, fact-law relevance, and procedural compliance, with a memory module for cross-round knowledge reuse in long-horizon tasks. Experiments on the dynamic benchmark J1-EVAL show that LawThinker achieves a 24% improvement over direct reasoning and an 11% gain over workflow-based methods, with particularly strong improvements on process-oriented metrics. Evaluations on three static benchmarks further confirm its generalization capability. The code is available at this https URL .

**arXiv ID:** 2602.12056
</details>

<details>
<summary><strong>STAR : Bridging Statistical and Agentic Reasoning for Large Model Performance Prediction</strong> - Xiaoxiao Wang, Chunxiao Li, Junying Wang, Yijin Guo, Zijian Chen, Chunyi Li, Xiaohong Liu, Zicheng Zhang, Guangtao Zhai - [[pdf]](https://arxiv.org/pdf/2602.12143)</summary>

**Abstract:** As comprehensive large model evaluation becomes prohibitively expensive, predicting model performance from limited observations has become essential. However, existing statistical methods struggle with pattern shifts, data sparsity, and lack of explanation, while pure LLM methods remain unreliable. We propose STAR, a framework that bridges data-driven STatistical expectations with knowledge-driven Agentic Reasoning. STAR leverages specialized retrievers to gather external knowledge and embeds semantic features into Constrained Probabilistic Matrix Factorization (CPMF) to generate statistical expectations with uncertainty. A reasoning module guided by Expectation Violation Theory (EVT) then refines predictions through intra-family analysis, cross-model comparison, and credibility-aware aggregation, producing adjustments with traceable explanations. Extensive experiments show that STAR consistently outperforms all baselines on both score-based and rank-based metrics, delivering a 14.46% gain in total score over the strongest statistical method under extreme sparsity, with only 1--2 observed scores per test model.

**arXiv ID:** 2602.12143
</details>

<details>
<summary><strong>When Visibility Outpaces Verification: Delayed Verification and Narrative Lock-in in Agentic AI Discourse</strong> - Hanjing Shi, Dominic DiFranzo - [[pdf]](https://arxiv.org/pdf/2602.11412)</summary>

**Abstract:** Agentic AI systems-autonomous entities capable of independent planning and execution-reshape the landscape of human-AI trust. Long before direct system exposure, user expectations are mediated through high-stakes public discourse on social platforms. However, platform-mediated engagement signals (e.g., upvotes) may inadvertently function as a ``credibility proxy,'' potentially stifling critical evaluation.
This paper investigates the interplay between social proof and verification timing in online discussions of agentic AI. Analyzing a longitudinal dataset from two distinct Reddit communities with contrasting interaction cultures-r/OpenClaw and r/Moltbook-we operationalize verification cues via reproducible lexical rules and model the ``time-to-first-verification'' using a right-censored survival analysis framework.
Our findings reveal a systemic ``Popularity Paradox'': high-visibility discussions in both subreddits experience significantly delayed or entirely absent verification cues compared to low-visibility threads. This temporal lag creates a critical window for ``Narrative Lock-in,'' where early, unverified claims crystallize into collective cognitive biases before evidence-seeking behaviors emerge. We discuss the implications of this ``credibility-by-visibility'' effect for AI safety and propose ``epistemic friction'' as a design intervention to rebalance engagement-driven platforms.

**arXiv ID:** 2602.11412
</details>

<details>
<summary><strong>How Smart Is Your GUI Agent? A Framework for the Future of Software Interaction</strong> - Sidong Feng, Chunyang Chen - [[pdf]](https://arxiv.org/pdf/2602.11514)</summary>

**Abstract:** GUI agents are rapidly becoming a new interaction to software, allowing people to navigate web, desktop and mobile rather than execute them click by click. Yet ``agent'' is described with radically different degrees of autonomy, obscuring capability, responsibility and risk. We call for conceptual clarity through GUI Agent Autonomy Levels (GAL), a six-level framework that makes autonomy explicit and helps benchmark progress toward trustworthy software interaction.

**arXiv ID:** 2602.11514
</details>

<details>
<summary><strong>SToRM: Supervised Token Reduction for Multi-modal LLMs toward efficient end-to-end autonomous driving</strong> - Seo Hyun Kim, Jin Bok Park, Do Yeon Koo, Ho Gun Park, Il Yong Chun - [[pdf]](https://arxiv.org/pdf/2602.11656)</summary>

**Abstract:** In autonomous driving, end-to-end (E2E) driving systems that predict control commands directly from sensor data have achieved significant advancements. For safe driving in unexpected scenarios, these systems may additionally rely on human interventions such as natural language instructions. Using a multi-modal large language model (MLLM) facilitates human-vehicle interaction and can improve performance in such scenarios. However, this approach requires substantial computational resources due to its reliance on an LLM and numerous visual tokens from sensor inputs, which are limited in autonomous vehicles. Many MLLM studies have explored reducing visual tokens, but often suffer end-task performance degradation compared to using all tokens.
To enable efficient E2E driving while maintaining performance comparable to using all tokens, this paper proposes the first Supervised Token Reduction framework for multi-modal LLMs (SToRM). The proposed framework consists of three key elements. First, a lightweight importance predictor with short-term sliding windows estimates token importance scores. Second, a supervised training approach uses an auxiliary path to obtain pseudo-supervision signals from an all-token LLM pass. Third, an anchor-context merging module partitions tokens into anchors and context tokens, and merges context tokens into relevant anchors to reduce redundancy while minimizing information loss. Experiments on the LangAuto benchmark show that SToRM outperforms state-of-the-art E2E driving MLLMs under the same reduced-token budget, maintaining all-token performance while reducing computational cost by up to 30x.

**arXiv ID:** 2602.11656
</details>

<details>
<summary><strong>AdaptEvolve: Improving Efficiency of Evolutionary AI Agents through Adaptive Model Selection</strong> - Pretam Ray, Pratik Prabhanjan Brahma, Zicheng Liu, Emad Barsoum - [[pdf]](https://arxiv.org/pdf/2602.11931)</summary>

**Abstract:** Evolutionary agentic systems intensify the trade-off between computational efficiency and reasoning capability by repeatedly invoking large language models (LLMs) during inference. This setting raises a central question: how can an agent dynamically select an LLM that is sufficiently capable for the current generation step while remaining computationally efficient? While model cascades offer a practical mechanism for balancing this trade-off, existing routing strategies typically rely on static heuristics or external controllers and do not explicitly account for model uncertainty. We introduce AdaptEvolve: Adaptive LLM Selection for Multi-LLM Evolutionary Refinement within an evolutionary sequential refinement framework that leverages intrinsic generation confidence to estimate real-time solvability. Empirical results show that confidence-driven selection yields a favourable Pareto frontier, reducing total inference cost by an average of 37.9% across benchmarks while retaining 97.5% of the upper-bound accuracy of static large-model baselines. Our code is available at this https URL.

**arXiv ID:** 2602.11931
</details>

<details>
<summary><strong>On the Adoption of AI Coding Agents in Open-source Android and iOS Development</strong> - Muhammad Ahmad Khan, Hasnain Ali, Muneeb Rana, Muhammad Saqib Ilyas, Abdul Ali Bangash - [[pdf]](https://arxiv.org/pdf/2602.12144)</summary>

**Abstract:** AI coding agents are increasingly contributing to software development, yet their impact on mobile development has received little empirical attention. In this paper, we present the first category-level empirical study of agent-generated code in open-source mobile app projects. We analyzed PR acceptance behaviors across mobile platforms, agents, and task categories using 2,901 AI-authored pull requests (PRs) in 193 verified Android and iOS open-source GitHub repositories in the AIDev dataset. We find that Android projects have received 2x more AI-authored PRs and have achieved higher PR acceptance rate (71%) than iOS (63%), with significant agent-level variation on Android. Across task categories, PRs with routine tasks (feature, fix, and ui) achieve the highest acceptance, while structural changes like refactor and build achieve lower success and longer resolution times. Furthermore, our evolution analysis shows improvement in PR resolution time on Android through mid-2025 before it declined again. Our findings offer the first evidence-based characterization of AI agents effects on OSS mobile projects and establish empirical baselines for evaluating agent-generated contributions to design platform aware agentic systems.

**arXiv ID:** 2602.12144
</details>

<details>
<summary><strong>DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search</strong> - Zhanli Li, Huiwen Tian, Lvzhou Luo, Yixuan Cao, Ping Luo - [[pdf]](https://arxiv.org/pdf/2602.05014)</summary>

**Abstract:** With the rapid advancement of tool-use capabilities in Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) is shifting from static, one-shot retrieval toward autonomous, multi-turn evidence acquisition. However, existing agentic search frameworks typically treat long documents as flat collections of unstructured chunks, disregarding the native hierarchical organization and sequential logic essential for human comprehension. To bridge this gap, we introduce \textbf{DeepRead}, a structure-aware document reasoning agent designed to operationalize document-native structural priors into actionable reasoning capabilities. Leveraging the structural fidelity of modern OCR, DeepRead constructs a paragraph-level, coordinate-based navigation system and equips the LLM with two synergistic tools: \textsf{Retrieve} for scanning-aware localization, and \textsf{ReadSection} for contiguous, order-preserving reading within specific hierarchical scopes. This design elicits a human-like ``locate-then-read'' reasoning paradigm, effectively mitigating the context fragmentation inherent in traditional retrieval methods. Extensive evaluations across four benchmarks spanning diverse document types demonstrate that DeepRead outperforms Search-o1-style agentic search baselines by an average of 10.3\%. Fine-grained behavioral analysis further confirms that DeepRead autonomously adopts human-aligned reading strategies, validating the critical role of structural awareness in achieving precise document reasoning. Our code is available at this https URL.

**arXiv ID:** 2602.05014
</details>

<details>
<summary><strong>FormalJudge: A Neuro-Symbolic Paradigm for Agentic Oversight</strong> - Jiayi Zhou, Yang Sheng, Hantao Lou, Yaodong Yang, Jie Fu - [[pdf]](https://arxiv.org/pdf/2602.11136)</summary>

**Abstract:** As LLM-based agents increasingly operate in high-stakes domains with real-world consequences, ensuring their behavioral safety becomes paramount. The dominant oversight paradigm, LLM-as-a-Judge, faces a fundamental dilemma: how can probabilistic systems reliably supervise other probabilistic systems without inheriting their failure modes? We argue that formal verification offers a principled escape from this dilemma, yet its adoption has been hindered by a critical bottleneck: the translation from natural language requirements to formal specifications. This paper bridges this gap by proposing , a neuro-symbolic framework that employs a bidirectional Formal-of-Thought architecture: LLMs serve as specification compilers that top-down decompose high-level human intent into atomic, verifiable constraints, then bottom-up prove compliance using Dafny specifications and Z3 Satisfiability modulo theories solving, which produces mathematical guarantees rather than probabilistic scores. We validate across three benchmarks spanning behavioral safety, multi-domain constraint adherence, and agentic upward deception detection. Experiments on 7 agent models demonstrate that achieves an average improvement of 16.6% over LLM-as-a-Judge baselines, enables weak-to-strong generalization where a 7B judge achieves over 90% accuracy detecting deception from 72B agents, and provides near-linear safety improvement through iterative refinement.

**arXiv ID:** 2602.11136
</details>

<details>
<summary><strong>Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization</strong> - Taeyoon Kwon, Dongwook Choi, Hyojun Kim, Sunghwan Kim, Seungjun Moon, Beong-woo Kwak, Kuan-Hao Huang, Jinyoung Yeo - [[pdf]](https://arxiv.org/pdf/2505.16348)</summary>

**Abstract:** LLM-powered embodied agents have shown success on conventional object-rearrangement tasks, but providing personalized assistance that leverages user-specific knowledge from past interactions presents new challenges. We investigate these challenges through the lens of agents' memory utilization along two critical dimensions: object semantics (identifying objects based on personal meaning) and user patterns (recalling sequences from behavioral routines). To assess these capabilities, we construct MEMENTO, an end-to-end two-stage evaluation framework comprising single-memory and joint-memory tasks. Our experiments reveal that current agents can recall simple object semantics but struggle to apply sequential user patterns to planning. Through in-depth analysis, we identify two critical bottlenecks: information overload and coordination failures when handling multiple memories. Based on these findings, we explore memory architectural approaches to address these challenges. Given our observation that episodic memory provides both personalized knowledge and in-context learning benefits, we design a hierarchical knowledge graph-based user-profile memory module that separately manages personalized knowledge, achieving substantial improvements on both single and joint-memory tasks. Project website: this https URL

**arXiv ID:** 2505.16348
</details>

<details>
<summary><strong>Learning to Route: A Rule-Driven Agent Framework for Hybrid-Source Retrieval-Augmented Generation</strong> - Haoyue Bai, Haoyu Wang, Shengyu Chen, Zhengzhang Chen, Lu-An Tang, Wei Cheng, Haifeng Chen, Yanjie Fu - [[pdf]](https://arxiv.org/pdf/2510.02388)</summary>

**Abstract:** Large Language Models (LLMs) have shown remarkable performance on general Question Answering (QA), yet they often struggle in domain-specific scenarios where accurate and up-to-date information is required. Retrieval-Augmented Generation (RAG) addresses this limitation by enriching LLMs with external knowledge, but existing systems primarily rely on unstructured documents, while largely overlooking relational databases, which provide precise, timely, and efficiently queryable factual information, serving as indispensable infrastructure in domains such as finance, healthcare, and scientific research. Motivated by this gap, we conduct a systematic analysis that reveals three central observations: (i) databases and documents offer complementary strengths across queries, (ii) naively combining both sources introduces noise and cost without consistent accuracy gains, and (iii) selecting the most suitable source for each query is crucial to balance effectiveness and efficiency. We further observe that query types show consistent regularities in their alignment with retrieval paths, suggesting that routing decisions can be effectively guided by systematic rules that capture these patterns. Building on these insights, we propose a rule-driven routing framework. A routing agent scores candidate augmentation paths based on explicit rules and selects the most suitable one; a rule-making expert agent refines the rules over time using QA feedback to maintain adaptability; and a path-level meta-cache reuses past routing decisions for semantically similar queries to reduce latency and cost. Experiments on three QA benchmarks demonstrate that our framework consistently outperforms static strategies and learned routing baselines, achieving higher accuracy while maintaining moderate computational cost.

**arXiv ID:** 2510.02388
</details>

<details>
<summary><strong>Towards Autonomous Mathematics Research</strong> - Tony Feng, Trieu H. Trinh, Garrett Bingham, Dawsen Hwang, Yuri Chervonyi, Junehyuk Jung, Joonkyung Lee, Carlo Pagano, Sang-hyun Kim, Federico Pasqualotto, Sergei Gukov, Jonathan N. Lee, Junsu Kim, Kaiying Hou, Golnaz Ghiasi, Yi Tay, YaGuang Li, Chenkai Kuang, Yuan Liu, Hanzhao Lin, Evan Zheran Liu, Nigamaa Nayakanti, Xiaomeng Yang, Heng-Tze Cheng, Demis Hassabis, Koray Kavukcuoglu, Quoc V. Le, Thang Luong - [[pdf]](https://arxiv.org/pdf/2602.10177)</summary>

**Abstract:** Recent advances in foundational models have yielded reasoning systems capable of achieving a gold-medal standard at the International Mathematical Olympiad. The transition from competition-level problem-solving to professional research, however, requires navigating vast literature and constructing long-horizon proofs. In this work, we introduce Aletheia, a math research agent that iteratively generates, verifies, and revises solutions end-to-end in natural language. Specifically, Aletheia is powered by an advanced version of Gemini Deep Think for challenging reasoning problems, a novel inference-time scaling law that extends beyond Olympiad-level problems, and intensive tool use to navigate the complexities of mathematical research. We demonstrate the capability of Aletheia from Olympiad problems to PhD-level exercises and most notably, through several distinct milestones in AI-assisted mathematics research: (a) a research paper (Feng26) generated by AI without any human intervention in calculating certain structure constants in arithmetic geometry called eigenweights; (b) a research paper (LeeSeo26) demonstrating human-AI collaboration in proving bounds on systems of interacting particles called independent sets; and (c) an extensive semi-autonomous evaluation (Feng et al., 2026a) of 700 open problems on Bloom's Erdos Conjectures database, including autonomous solutions to four open questions. In order to help the public better understand the developments pertaining to AI and mathematics, we suggest quantifying standard levels of autonomy and novelty of AI-assisted results, as well as propose a novel concept of human-AI interaction cards for transparency. We conclude with reflections on human-AI collaboration in mathematics and share all prompts as well as model outputs at this https URL.

**arXiv ID:** 2602.10177
</details>

<details>
<summary><strong>Calibration and Evaluation of Car-Following Models for Autonomous Shuttles Using a Novel Multi-Criteria Framework</strong> - Renan Favero, Lily Elefteriadou - [[pdf]](https://arxiv.org/pdf/2602.11517)</summary>

**Abstract:** Autonomous shuttles (AS) are fully autonomous transit vehicles with operating characteristics distinct from conventional autonomous vehicles (AV). Developing dedicated car-following models for AS is critical to understanding their traffic impacts; however, few studies have calibrated such models with field data. More advanced machine learning (ML) techniques have not yet been applied to AS trajectories, leaving the potential of ML for capturing AS dynamics unexplored and constraining the development of dedicated AS models. Furthermore, there is a lack of a unified framework for systematically evaluating and comparing the performance of car-following models to replicate real trajectories. Existing car-following studies often rely on disparate metrics, which limit reproducibility and performance comparability.
This study addresses these gaps through two main contributions: (1) the calibration of a diverse set of car-following models using real-world AS trajectory data, including eight machine learning algorithms and two physics-based models; and (2) the introduction of a multi-criteria evaluation framework that integrates measures of prediction accuracy, trajectory stability, and statistical similarity, which provides a generalizable methodology for a systematic assessment of car-following models.
Results indicated that the proposed calibrated XGBoost model achieved the best overall performance. Sequential model type, such as LSTM and CNN, captured long-term positional stability but were less responsive to short-term dynamics. LSTM and CNN captured long-term positional stability but were less responsive to short-term dynamics. Traditional models (IDM, ACC) and kernel methods showed lower accuracy and stability than most ML models tested.

**arXiv ID:** 2602.11517
</details>

<details>
<summary><strong>Social Human Robot Embodied Conversation (SHREC) Dataset: Benchmarking Foundational Models' Social Reasoning</strong> - Dong Won Lee, Yubin Kim, Denison Guvenoz, Sooyeon Jeong, Parker Malachowsky, Louis-Philippe Morency, Cynthia Breazeal, Hae Won Park - [[pdf]](https://arxiv.org/pdf/2504.13898)</summary>

**Abstract:** Our work focuses on the social reasoning capabilities of foundation models for real-world human-robot interactions. We introduce the Social Human Robot Embodied Conversation (SHREC) Dataset, a benchmark of $\sim$400 real-world human-robot interaction videos and over 10K annotations, capturing robot social errors, competencies, underlying rationales, and corrections. Unlike prior datasets focused on human-human interactions, the SHREC Dataset uniquely highlights the social challenges faced by real-world social robots such as emotion understanding, intention tracking, and conversational mechanics. Moreover, current foundation models struggle to recognize these deficits, which manifest as subtle, socially situated failures. To evaluate AI models' capacity for social reasoning, we define eight benchmark tasks targeting critical areas such as (1) detection of social errors and competencies, (2) identification of underlying social attributes, (3) comprehension of interaction flow, and (4) providing rationale and alternative correct actions. Experiments with state-of-the-art foundation models, alongside human evaluations, reveal substantial performance gaps -- underscoring the difficulty and providing directions in developing socially intelligent AI.

**arXiv ID:** 2504.13898
</details>

</details>

<details open>
<summary><h2>LLM Agents (12 papers)</h2></summary>

<details>
<summary><strong>Pushing Forward Pareto Frontiers of Proactive Agents with Behavioral Agentic Optimization</strong> - Yihang Yao, Zhepeng Cen, Haohong Lin, Shiqi Liu, Zuxin Liu, Jiacheng Zhu, Zhang-Wei Hong, Laixi Shi, Ding Zhao - [[pdf]](https://arxiv.org/pdf/2602.11351)</summary>

**Abstract:** Proactive large language model (LLM) agents aim to actively plan, query, and interact over multiple turns, enabling efficient task completion beyond passive instruction following and making them essential for real-world, user-centric applications. Agentic reinforcement learning (RL) has recently emerged as a promising solution for training such agents in multi-turn settings, allowing interaction strategies to be learned from feedback. However, existing pipelines face a critical challenge in balancing task performance with user engagement, as passive agents can not efficiently adapt to users' intentions while overuse of human feedback reduces their satisfaction. To address this trade-off, we propose BAO, an agentic RL framework that combines behavior enhancement to enrich proactive reasoning and information-gathering capabilities with behavior regularization to suppress inefficient or redundant interactions and align agent behavior with user expectations. We evaluate BAO on multiple tasks from the UserRL benchmark suite, and demonstrate that it substantially outperforms proactive agentic RL baselines while achieving comparable or even superior performance to commercial LLM agents, highlighting its effectiveness for training proactive, user-aligned LLM agents in complex multi-turn scenarios. Our website: this https URL.

**arXiv ID:** 2602.11351
</details>

<details>
<summary><strong>AgentNoiseBench: Benchmarking Robustness of Tool-Using LLM Agents Under Noisy Condition</strong> - Ruipeng Wang, Yuxin Chen, Yukai Wang, Chang Wu, Junfeng Fang, Xiaodong Cai, Qi Gu, Hui Su, An Zhang, Xiang Wang, Xunliang Cai, Tat-Seng Chua - [[pdf]](https://arxiv.org/pdf/2602.11348)</summary>

**Abstract:** Recent advances in large language models have enabled LLM-based agents to achieve strong performance on a variety of benchmarks. However, their performance in real-world deployments often that observed on benchmark settings, especially in complex and imperfect environments. This discrepancy largely arises because prevailing training and evaluation paradigms are typically built on idealized assumptions, overlooking the inherent stochasticity and noise present in real-world interactions. To bridge this gap, we introduce AgentNoiseBench, a framework for systematically evaluating the robustness of agentic models under noisy environments. We first conduct an in-depth analysis of biases and uncertainties in real-world scenarios and categorize environmental noise into two primary types: user-noise and tool-noise. Building on this analysis, we develop an automated pipeline that injects controllable noise into existing agent-centric benchmarks while preserving task solvability. Leveraging this pipeline, we perform extensive evaluations across a wide range of models with diverse architectures and parameter scales. Our results reveal consistent performance variations under different noise conditions, highlighting the sensitivity of current agentic models to realistic environmental perturbations.

**arXiv ID:** 2602.11348
</details>

<details>
<summary><strong>ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences</strong> - Bang Nguyen, Dominik Soós, Qian Ma, Rochana R. Obadage, Zack Ranjan, Sai Koneru, Timothy M. Errington, Shakhlo Nematova, Sarah Rajtmajer, Jian Wu, Meng Jiang - [[pdf]](https://arxiv.org/pdf/2602.11354)</summary>

**Abstract:** The literature has witnessed an emerging interest in AI agents for automated assessment of scientific papers. Existing benchmarks focus primarily on the computational aspect of this task, testing agents' ability to reproduce or replicate research outcomes when having access to the code and data. This setting, while foundational, (1) fails to capture the inconsistent availability of new data for replication as opposed to reproduction, and (2) lacks ground-truth diversity by focusing only on reproducible papers, thereby failing to evaluate an agent's ability to identify non-replicable research. Furthermore, most benchmarks only evaluate outcomes rather than the replication process. In response, we introduce ReplicatorBench, an end-to-end benchmark, including human-verified replicable and non-replicable research claims in social and behavioral sciences for evaluating AI agents in research replication across three stages: (1) extraction and retrieval of replication data; (2) design and execution of computational experiments; and (3) interpretation of results, allowing a test of AI agents' capability to mimic the activities of human replicators in real world. To set a baseline of AI agents' capability, we develop ReplicatorAgent, an agentic framework equipped with necessary tools, like web search and iterative interaction with sandboxed environments, to accomplish tasks in ReplicatorBench. We evaluate ReplicatorAgent across four underlying large language models (LLMs), as well as different design choices of programming language and levels of code access. Our findings reveal that while current LLM agents are capable of effectively designing and executing computational experiments, they struggle with retrieving resources, such as new data, necessary to replicate a claim. All code and data are publicly available at this https URL.

**arXiv ID:** 2602.11354
</details>

<details>
<summary><strong>When Agents Disagree With Themselves: Measuring Behavioral Consistency in LLM-Based Agents</strong> - Aman Mehta - [[pdf]](https://arxiv.org/pdf/2602.11619)</summary>

**Abstract:** Run the same LLM agent on the same task twice: do you get the same behavior? We find the answer is often no. In a study of 3,000 agent runs across three models (Llama 3.1 70B, GPT-4o, and Claude Sonnet 4.5) on HotpotQA, we observe that ReAct-style agents produce 2.0--4.2 distinct action sequences per 10 runs on average, even with identical inputs. More importantly, this variance predicts failure: tasks with consistent behavior ($\leq$2 unique paths) achieve 80--92% accuracy, while highly inconsistent tasks ($\geq$6 unique paths) achieve only 25--60%, a 32--55 percentage point gap depending on model. We trace variance to early decisions: 69% of divergence occurs at step 2, the first search query. Our results suggest that monitoring behavioral consistency during execution could enable early error detection and improve agent reliability.

**arXiv ID:** 2602.11619
</details>

<details>
<summary><strong>AIR: Improving Agent Safety through Incident Response</strong> - Zibo Xiao, Jun Sun, Junjie Chen - [[pdf]](https://arxiv.org/pdf/2602.11749)</summary>

**Abstract:** Large Language Model (LLM) agents are increasingly deployed in practice across a wide range of autonomous applications. Yet current safety mechanisms for LLM agents focus almost exclusively on preventing failures in advance, providing limited capabilities for responding to, containing, or recovering from incidents after they inevitably arise. In this work, we introduce AIR, the first incident response framework for LLM agent systems. AIR defines a domain-specific language for managing the incident response lifecycle autonomously in LLM agent systems, and integrates it into the agent's execution loop to (1) detect incidents via semantic checks grounded in the current environment state and recent context, (2) guide the agent to execute containment and recovery actions via its tools, and (3) synthesize guardrail rules during eradication to block similar incidents in future executions. We evaluate AIR on three representative agent types. Results show that AIR achieves detection, remediation, and eradication success rates all exceeding 90%. Extensive experiments further confirm the necessity of AIR's key design components, show the timeliness and moderate overhead of AIR, and demonstrate that LLM-generated rules can approach the effectiveness of developer-authored rules across domains. These results show that incident response is both feasible and essential as a first-class mechanism for improving agent safety.

**arXiv ID:** 2602.11749
</details>

<details>
<summary><strong>TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents</strong> - Aladin Djuhera, Swanand Ravindra Kadhe, Farhan Ahmed, Holger Boche - [[pdf]](https://arxiv.org/pdf/2602.11767)</summary>

**Abstract:** Advances in large language models (LLMs) are driving a shift toward using reinforcement learning (RL) to train agents from iterative, multi-turn interactions across tasks. However, multi-turn RL remains challenging as rewards are often sparse or delayed, and environments can be stochastic. In this regime, naive trajectory sampling can hinder exploitation and induce mode collapse. We propose TSR (Trajectory-Search Rollouts), a training-time approach that repurposes test-time scaling ideas for improved per-turn rollout generation. TSR performs lightweight tree-style search to construct high-quality trajectories by selecting high-scoring actions at each turn using task-specific feedback. This improves rollout quality and stabilizes learning while leaving the underlying optimization objective unchanged, making TSR optimizer-agnostic. We instantiate TSR with best-of-N, beam, and shallow lookahead search, and pair it with PPO and GRPO, achieving up to 15% performance gains and more stable learning on Sokoban, FrozenLake, and WebShop tasks at a one-time increase in training compute. By moving search from inference time to the rollout stage of training, TSR provides a simple and general mechanism for stronger multi-turn agent learning, complementary to existing frameworks and rejection-sampling-style selection methods.

**arXiv ID:** 2602.11767
</details>

<details>
<summary><strong>Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments</strong> - Romain Froger, Pierre Andrews, Matteo Bettini, Amar Budhiraja, Ricardo Silveira Cabral, Virginie Do, Emilien Garreau, Jean-Baptiste Gaya, Hugo Laurençon, Maxime Lecanu, Kunal Malkan, Dheeraj Mekala, Pierre Ménard, Gerard Moreno-Torres Bertran, Ulyana Piterbarg, Mikhail Plekhanov, Mathieu Rita, Andrey Rusakov, Vladislav Vorotilov, Mengjue Wang, Ian Yu, Amine Benhalloum, Grégoire Mialon, Thomas Scialom - [[pdf]](https://arxiv.org/pdf/2602.11964)</summary>

**Abstract:** We introduce Gaia2, a benchmark for evaluating large language model agents in realistic, asynchronous environments. Unlike prior static or synchronous evaluations, Gaia2 introduces scenarios where environments evolve independently of agent actions, requiring agents to operate under temporal constraints, adapt to noisy and dynamic events, resolve ambiguity, and collaborate with other agents. Each scenario is paired with a write-action verifier, enabling fine-grained, action-level evaluation and making Gaia2 directly usable for reinforcement learning from verifiable rewards. Our evaluation of state-of-the-art proprietary and open-source models shows that no model dominates across capabilities: GPT-5 (high) reaches the strongest overall score of 42% pass@1 but fails on time-sensitive tasks, Claude-4 Sonnet trades accuracy and speed for cost, Kimi-K2 leads among open-source models with 21% pass@1. These results highlight fundamental trade-offs between reasoning, efficiency, robustness, and expose challenges in closing the "sim2real" gap. Gaia2 is built on a consumer environment with the open-source Agents Research Environments platform and designed to be easy to extend. By releasing Gaia2 alongside the foundational ARE framework, we aim to provide the community with a flexible infrastructure for developing, benchmarking, and training the next generation of practical agent systems.

**arXiv ID:** 2602.11964
</details>

<details>
<summary><strong>Think like a Scientist: Physics-guided LLM Agent for Equation Discovery</strong> - Jianke Yang, Ohm Venkatachalam, Mohammad Kianezhad, Sharvaree Vadgama, Rose Yu - [[pdf]](https://arxiv.org/pdf/2602.12259)</summary>

**Abstract:** Explaining observed phenomena through symbolic, interpretable formulas is a fundamental goal of science. Recently, large language models (LLMs) have emerged as promising tools for symbolic equation discovery, owing to their broad domain knowledge and strong reasoning capabilities. However, most existing LLM-based systems try to guess equations directly from data, without modeling the multi-step reasoning process that scientists often follow: first inferring physical properties such as symmetries, then using these as priors to restrict the space of candidate equations. We introduce KeplerAgent, an agentic framework that explicitly follows this scientific reasoning process. The agent coordinates physics-based tools to extract intermediate structure and uses these results to configure symbolic regression engines such as PySINDy and PySR, including their function libraries and structural constraints. Across a suite of physical equation benchmarks, KeplerAgent achieves substantially higher symbolic accuracy and greater robustness to noisy data than both LLM and traditional baselines.

**arXiv ID:** 2602.12259
</details>

<details>
<summary><strong>Exploring Silicon-Based Societies: An Early Study of the Moltbook Agent Community</strong> - Yu-Zheng Lin, Bono Po-Jen Shih, Hsuan-Ying Alessandra Chien, Shalaka Satam, Jesus Horacio Pacheco, Sicong Shao, Soheil Salehi, Pratik Satam - [[pdf]](https://arxiv.org/pdf/2602.02613)</summary>

**Abstract:** The rapid emergence of autonomous large language model agents has given rise to persistent, large-scale agent ecosystems whose collective behavior cannot be adequately understood through anecdotal observation or small-scale simulation. This paper introduces data-driven silicon sociology as a systematic empirical framework for studying social structure formation among interacting artificial agents. We present a pioneering large-scale data mining investigation of an in-the-wild agent society by analyzing Moltbook, a social platform designed primarily for agent-to-agent interaction. At the time of study, Moltbook hosted over 150,000 registered autonomous agents operating across thousands of agent-created sub-communities. Using programmatic and non-intrusive data acquisition, we collected and analyzed the textual descriptions of 12,758 submolts, which represent proactive sub-community partitioning activities within the ecosystem. Treating agent-authored descriptions as first-class observational artifacts, we apply rigorous preprocessing, contextual embedding, and unsupervised clustering techniques to uncover latent patterns of thematic organization and social space structuring. The results show that autonomous agents systematically organize collective space through reproducible patterns spanning human-mimetic interests, silicon-centric self-reflection, and early-stage economic and coordination behaviors. Rather than relying on predefined sociological taxonomies, these structures emerge directly from machine-generated data traces. This work establishes a methodological foundation for data-driven silicon sociology and demonstrates that data mining techniques can provide a powerful lens for understanding the organization and evolution of large autonomous agent societies.

**arXiv ID:** 2602.02613
</details>

<details>
<summary><strong>Agent-Diff: Benchmarking LLM Agents on Enterprise API Tasks via Code Execution with State-Diff-Based Evaluation</strong> - Hubert M. Pysklo, Artem Zhuravel, Patrick D. Watson - [[pdf]](https://arxiv.org/pdf/2602.11224)</summary>

**Abstract:** We present Agent-Diff, a novel benchmarking framework for evaluating agentic Large Language Models (LLMs) on real-world tasks that execute code via external APIs. Agentic LLM performance varies due to differences in models, external tool access, prompt structures, and agentic frameworks. Benchmarks must make fundamental trade-offs between a sandboxed approach that controls for variation in software environments and more ecologically valid approaches employing real services. Agent-Diff attempts to capture the desirable features of both of these approaches by including access to the real API interfaces for software services while sandboxing the environment in which calls are made, processed, and evaluated. This approach relies on two key innovations. The first is a novel state-diff contract, which separates process from outcome - rather than fuzzy trace or parameter matching, we define task success as whether the expected change in environment state was achieved. The second is a novel sandbox that provides a standardized scripting layer that all models use to execute code against external APIs (Slack, Box, Linear, Google Calendar). Thus, we can evaluate different agentic LLMs against a standardized set of contracts using a unified sandbox while still evaluating their performance on real-world service interfaces. Using the Agent-Diff framework, we provide benchmarks for nine LLMs across 224 tasks utilizing enterprise software workflows. In addition, we evaluate the robustness of the framework with ablation experiments to assess the contribution of access to API documentation on benchmark performance. Code and data: this https URL.

**arXiv ID:** 2602.11224
</details>

<details>
<summary><strong>Evaluating Memory Structure in LLM Agents</strong> - Alina Shutova, Alexandra Olenina, Ivan Vinogradov, Anton Sinitsin - [[pdf]](https://arxiv.org/pdf/2602.11243)</summary>

**Abstract:** Modern LLM-based agents and chat assistants rely on long-term memory frameworks to store reusable knowledge, recall user preferences, and augment reasoning. As researchers create more complex memory architectures, it becomes increasingly difficult to analyze their capabilities and guide future memory designs. Most long-term memory benchmarks focus on simple fact retention, multi-hop recall, and time-based changes. While undoubtedly important, these capabilities can often be achieved with simple retrieval-augmented LLMs and do not test complex memory hierarchies. To bridge this gap, we propose StructMemEval - a benchmark that tests the agent's ability to organize its long-term memory, not just factual recall. We gather a suite of tasks that humans solve by organizing their knowledge in a specific structure: transaction ledgers, to-do lists, trees and others. Our initial experiments show that simple retrieval-augmented LLMs struggle with these tasks, whereas memory agents can reliably solve them if prompted how to organize their memory. However, we also find that modern LLMs do not always recognize the memory structure when not prompted to do so. This highlights an important direction for future improvements in both LLM training and memory frameworks.

**arXiv ID:** 2602.11243
</details>

<details>
<summary><strong>Structured Context Engineering for File-Native Agentic Systems: Evaluating Schema Accuracy, Format Effectiveness, and Multi-File Navigation at Scale</strong> - Damon McMillan - [[pdf]](https://arxiv.org/pdf/2602.05447)</summary>

**Abstract:** Large Language Model agents increasingly operate external systems through programmatic interfaces, yet practitioners lack empirical guidance on how to structure the context these agents consume. Using SQL generation as a proxy for programmatic agent operations, we present a systematic study of context engineering for structured data, comprising 9,649 experiments across 11 models, 4 formats (YAML, Markdown, JSON, Token-Oriented Object Notation [TOON]), and schemas ranging from 10 to 10,000 tables. Our findings challenge common assumptions. First, architecture choice is model-dependent: file-based context retrieval improves accuracy for frontier-tier models (Claude, GPT, Gemini; +2.7%, p=0.029) but shows mixed results for open source models (aggregate -7.7%, p<0.001), with deficits varying substantially by model. Second, format does not significantly affect aggregate accuracy (chi-squared=2.45, p=0.484), though individual models, particularly open source, exhibit format-specific sensitivities. Third, model capability is the dominant factor, with a 21 percentage point accuracy gap between frontier and open source tiers that dwarfs any format or architecture effect. Fourth, file-native agents scale to 10,000 tables through domain-partitioned schemas while maintaining high navigation accuracy. Fifth, file size does not predict runtime efficiency: compact or novel formats can incur a token overhead driven by grep output density and pattern unfamiliarity, with the magnitude depending on model capability. These findings provide practitioners with evidence-based guidance for deploying LLM agents on structured systems, demonstrating that architectural decisions should be tailored to model capability rather than assuming universal best practices.

**arXiv ID:** 2602.05447
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (23 papers)</h2></summary>

<details>
<summary><strong>The PBSAI Governance Ecosystem: A Multi-Agent AI Reference Architecture for Securing Enterprise AI Estates</strong> - John M. Willis - [[pdf]](https://arxiv.org/pdf/2602.11301)</summary>

**Abstract:** Enterprises are rapidly deploying large language models, retrieval augmented generation pipelines, and tool using agents into production, often on shared high performance computing clusters and cloud accelerator platforms that also support defensive analytics. These systems increasingly function not as isolated models but as AI estates: socio technical systems spanning models, agents, data pipelines, security tooling, human workflows, and hyperscale infrastructure. Existing governance and security frameworks, including the NIST AI Risk Management Framework and systems security engineering guidance, articulate principles and risk functions but do not provide implementable architectures for multi agent, AI enabled cyber defense.
This paper introduces the Practitioners Blueprint for Secure AI (PBSAI) Governance Ecosystem, a multi agent reference architecture for securing enterprise and hyperscale AI estates. PBSAI organizes responsibilities into a twelve domain taxonomy and defines bounded agent families that mediate between tools and policy through shared context envelopes and structured output contracts. The architecture assumes baseline enterprise security capabilities and encodes key systems security techniques, including analytic monitoring, coordinated defense, and adaptive response. A lightweight formal model of agents, context envelopes, and ecosystem level invariants clarifies the traceability, provenance, and human in the loop guarantees enforced across domains. We demonstrate alignment with NIST AI RMF functions and illustrate application in enterprise SOC and hyperscale defensive environments. PBSAI is proposed as a structured, evidence centric foundation for open ecosystem development and future empirical validation.

**arXiv ID:** 2602.11301
</details>

<details>
<summary><strong>Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization</strong> - Chengrui Qu, Christopher Yeh, Kishan Panaganti, Eric Mazumdar, Adam Wierman - [[pdf]](https://arxiv.org/pdf/2602.11437)</summary>

**Abstract:** Cooperative multi-agent reinforcement learning (MARL) commonly adopts centralized training with decentralized execution, where value-factorization methods enforce the individual-global-maximum (IGM) principle so that decentralized greedy actions recover the team-optimal joint action. However, the reliability of this recipe in real-world settings remains unreliable due to environmental uncertainties arising from the sim-to-real gap, model mismatch, and system noise. We address this gap by introducing Distributionally robust IGM (DrIGM), a principle that requires each agent's robust greedy action to align with the robust team-optimal joint action. We show that DrIGM holds for a novel definition of robust individual action values, which is compatible with decentralized greedy execution and yields a provable robustness guarantee for the whole system. Building on this foundation, we derive DrIGM-compliant robust variants of existing value-factorization architectures (e.g., VDN/QMIX/QTRAN) that (i) train on robust Q-targets, (ii) preserve scalability, and (iii) integrate seamlessly with existing codebases without bespoke per-agent reward shaping. Empirically, on high-fidelity SustainGym simulators and a StarCraft game environment, our methods consistently improve out-of-distribution performance. Code and data are available at this https URL.

**arXiv ID:** 2602.11437
</details>

<details>
<summary><strong>AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems</strong> - Faouzi El Yagoubi, Ranwa Al Mallah, Godwin Badu-Marfo - [[pdf]](https://arxiv.org/pdf/2602.11510)</summary>

**Abstract:** Multi-agent Large Language Model (LLM) systems create privacy risks that current benchmarks cannot measure. When agents coordinate on tasks, sensitive data passes through inter-agent messages, shared memory, and tool arguments; pathways that output-only audits never inspect. We introduce AgentLeak, to the best of our knowledge the first full-stack benchmark for privacy leakage covering internal channels, spanning 1,000 scenarios across healthcare, finance, legal, and corporate domains, paired with a 32-class attack taxonomy and three-tier detection pipeline. Testing GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet, Mistral Large, and Llama 3.3 70B across 4,979 traces reveals that multi-agent configurations reduce per-channel output leakage (C1: 27.2% vs 43.2% in single-agent) but introduce unmonitored internal channels that raise total system exposure to 68.9% (OR-aggregated across C1, C2, C5). Internal channels account for most of this gap: inter-agent messages (C2) leak at 68.8%, compared to 27.2% on C1 (output channel). This means that output-only audits miss 41.7% of violations. Claude 3.5 Sonnet, which emphasizes safety alignment in its design, achieves the lowest leakage rates on both external (3.3%) and internal (28.1%) channels, suggesting that model-level safety training may transfer to internal channel protection. Across all five models and four domains, the pattern C2 > C1 holds consistently, confirming that inter-agent communication is the primary vulnerability. These findings underscore the need for coordination frameworks that incorporate internal-channel privacy protections and enforce privacy controls on inter-agent communication.

**arXiv ID:** 2602.11510
</details>

<details>
<summary><strong>CausalAgent: A Conversational Multi-Agent System for End-to-End Causal Inference</strong> - Jiawei Zhu, Wei Chen, Ruichu Cai - [[pdf]](https://arxiv.org/pdf/2602.11527)</summary>

**Abstract:** Causal inference holds immense value in fields such as healthcare, economics, and social sciences. However, traditional causal analysis workflows impose significant technical barriers, requiring researchers to possess dual backgrounds in statistics and computer science, while manually selecting algorithms, handling data quality issues, and interpreting complex results. To address these challenges, we propose CausalAgent, a conversational multi-agent system for end-to-end causal inference. The system innovatively integrates Multi-Agent Systems (MAS), Retrieval-Augmented Generation (RAG), and the Model Context Protocol (MCP) to achieve automation from data cleaning and causal structure learning to bias correction and report generation through natural language interaction. Users need only upload a dataset and pose questions in natural language to receive a rigorous, interactive analysis report. As a novel user-centered human-AI collaboration paradigm, CausalAgent explicitly models the analysis workflow. By leveraging interactive visualizations, it significantly lowers the barrier to entry for causal analysis while ensuring the rigor and interpretability of the process.

**arXiv ID:** 2602.11527
</details>

<details>
<summary><strong>The Five Ws of Multi-Agent Communication: Who Talks to Whom, When, What, and Why -- A Survey from MARL to Emergent Language and LLMs</strong> - Jingdi Chen, Hanqing Yang, Zongjun Liu, Carlee Joe-Wong - [[pdf]](https://arxiv.org/pdf/2602.11583)</summary>

**Abstract:** Multi-agent sequential decision-making powers many real-world systems, from autonomous vehicles and robotics to collaborative AI assistants. In dynamic, partially observable environments, communication is often what reduces uncertainty and makes collaboration possible. This survey reviews multi-agent communication (MA-Comm) through the Five Ws: who communicates with whom, what is communicated, when communication occurs, and why communication is beneficial. This framing offers a clean way to connect ideas across otherwise separate research threads. We trace how communication approaches have evolved across three major paradigms. In Multi-Agent Reinforcement Learning (MARL), early methods used hand-designed or implicit protocols, followed by end-to-end learned communication optimized for reward and control. While successful, these protocols are frequently task-specific and hard to interpret, motivating work on Emergent Language (EL), where agents can develop more structured or symbolic communication through interaction. EL methods, however, still struggle with grounding, generalization, and scalability, which has fueled recent interest in large language models (LLMs) that bring natural language priors for reasoning, planning, and collaboration in more open-ended settings. Across MARL, EL, and LLM-based systems, we highlight how different choices shape communication design, where the main trade-offs lie, and what remains unsolved. We distill practical design patterns and open challenges to support future hybrid systems that combine learning, language, and control for scalable and interpretable multi-agent collaboration.

**arXiv ID:** 2602.11583
</details>

<details>
<summary><strong>Beyond End-to-End Video Models: An LLM-Based Multi-Agent System for Educational Video Generation</strong> - Lingyong Yan, Jiulong Wu, Dong Xie, Weixian Shi, Deguo Xia, Jizhou Huang - [[pdf]](https://arxiv.org/pdf/2602.11790)</summary>

**Abstract:** Although recent end-to-end video generation models demonstrate impressive performance in visually oriented content creation, they remain limited in scenarios that require strict logical rigor and precise knowledge representation, such as instructional and educational media. To address this problem, we propose LAVES, a hierarchical LLM-based multi-agent system for generating high-quality instructional videos from educational problems. The LAVES formulates educational video generation as a multi-objective task that simultaneously demands correct step-by-step reasoning, pedagogically coherent narration, semantically faithful visual demonstrations, and precise audio--visual alignment. To address the limitations of prior approaches--including low procedural fidelity, high production cost, and limited controllability--LAVES decomposes the generation workflow into specialized agents coordinated by a central Orchestrating Agent with explicit quality gates and iterative critique mechanisms. Specifically, the Orchestrating Agent supervises a Solution Agent for rigorous problem solving, an Illustration Agent that produces executable visualization codes, and a Narration Agent for learner-oriented instructional scripts. In addition, all outputs from the working agents are subject to semantic critique, rule-based constraints, and tool-based compilation checks. Rather than directly synthesizing pixels, the system constructs a structured executable video script that is deterministically compiled into synchronized visuals and narration using template-driven assembly rules, enabling fully automated end-to-end production without manual editing. In large-scale deployments, LAVES achieves a throughput exceeding one million videos per day, delivering over a 95% reduction in cost compared to current industry-standard approaches while maintaining a high acceptance rate.

**arXiv ID:** 2602.11790
</details>

<details>
<summary><strong>Differentiable Modal Logic for Multi-Agent Diagnosis, Orchestration and Communication</strong> - Antonin Sulc - [[pdf]](https://arxiv.org/pdf/2602.12083)</summary>

**Abstract:** As multi-agent AI systems evolve from simple chatbots to autonomous swarms, debugging semantic failures requires reasoning about knowledge, belief, causality, and obligation, precisely what modal logic was designed to formalize. However, traditional modal logic requires manual specification of relationship structures that are unknown or dynamic in real systems. This tutorial demonstrates differentiable modal logic (DML), implemented via Modal Logical Neural Networks (MLNNs), enabling systems to learn trust networks, causal chains, and regulatory boundaries from behavioral data alone.
We present a unified neurosymbolic debugging framework through four modalities: epistemic (who to trust), temporal (when events cause failures), deontic (what actions are permitted), and doxastic (how to interpret agent confidence). Each modality is demonstrated on concrete multi-agent scenarios, from discovering deceptive alliances in diplomacy games to detecting LLM hallucinations, with complete implementations showing how logical contradictions become learnable optimization objectives. Key contributions for the neurosymbolic community: (1) interpretable learned structures where trust and causality are explicit parameters, not opaque embeddings; (2) knowledge injection via differentiable axioms that guide learning with sparse data (3) compositional multi-modal reasoning that combines epistemic, temporal, and deontic constraints; and (4) practical deployment patterns for monitoring, active control and communication of multi-agent systems. All code provided as executable Jupyter notebooks.

**arXiv ID:** 2602.12083
</details>

<details>
<summary><strong>TDPNavigator-Placer: Thermal- and Wirelength-Aware Chiplet Placement in 2.5D Systems Through Multi-Agent Reinforcement Learning</strong> - Yubo Hou, Furen Zhuang, Partha Pratim Kundu, Sezin Ata Kircali, Jie Wang, Mihai Dragos Rotaru, Dutta Rahul, Ashish James - [[pdf]](https://arxiv.org/pdf/2602.11187)</summary>

**Abstract:** The rapid growth of electronics has accelerated the adoption of 2.5D integrated circuits, where effective automated chiplet placement is essential as systems scale to larger and more heterogeneous chiplet assemblies. Existing placement methods typically focus on minimizing wirelength or transforming multi-objective optimization into a single objective through weighted sum, which limits their ability to handle competing design requirements. Wirelength reduction and thermal management are inherently conflicting objectives, making prior approaches inadequate for practical deployment. To address this challenge, we propose TDPNavigator-Placer, a novel multi-agent reinforcement learning framework that dynamically optimizes placement based on chiplet's thermal design power (TDP). This approach explicitly assigns these inherently conflicting objectives to specialized agents, each operating under distinct reward mechanisms and environmental constraints within a unified placement paradigm. Experimental results demonstrate that TDPNavigator-Placer delivers a significantly improved Pareto front over state-of-the-art methods, enabling more balanced trade-offs between wirelength and thermal performance.

**arXiv ID:** 2602.11187
</details>

<details>
<summary><strong>DDL2PropBank Agent: Benchmarking Multi-Agent Frameworks' Developer Experience Through a Novel Relational Schema Mapping Task</strong> - Shafiuddin Rehan Ahmed, Wei Wei - [[pdf]](https://arxiv.org/pdf/2602.11198)</summary>

**Abstract:** Multi-agent frameworks promise to simplify LLM-driven software development, yet there is no principled way to evaluate their developer experience in a controlled setting. We introduce DDL2PropBank, a novel benchmark task that maps relational database schemas to PropBank rolesets, requiring autonomous retrieval of candidate frames and fine-grained linguistic reasoning over table names, columns, and relations. Using the Agent-as-a-Tool pattern, we implement identical agent logic across 10 frameworks and evaluate along two dimensions: (i) code complexity via static analysis, and (ii) AI-assistability -- the extent to which LLMs can autonomously generate correct, framework-specific code. Our results reveal a threefold complexity spectrum, with Pydantic AI and Agno requiring the least implementation overhead. For AI-assistability, structural alignment scores reliably proxy runtime success for frameworks with single canonical patterns, but overestimate correctness for multi-pattern frameworks. Agno emerges as the strongest overall performer, combining lowest complexity with highest structural alignment and 83% pass@1.

**arXiv ID:** 2602.11198
</details>

<details>
<summary><strong>Security Threat Modeling for Emerging AI-Agent Protocols: A Comparative Analysis of MCP, A2A, Agora, and ANP</strong> - Zeynab Anbiaee, Mahdi Rabbani, Mansur Mirani, Gunjan Piya, Igor Opushnyev, Ali Ghorbani, Sajjad Dadkhah - [[pdf]](https://arxiv.org/pdf/2602.11327)</summary>

**Abstract:** The rapid development of the AI agent communication protocols, including the Model Context Protocol (MCP), Agent2Agent (A2A), Agora, and Agent Network Protocol (ANP), is reshaping how AI agents communicate with tools, services, and each other. While these protocols support scalable multi-agent interaction and cross-organizational interoperability, their security principles remain understudied, and standardized threat modeling is limited; no protocol-centric risk assessment framework has been established yet. This paper presents a systematic security analysis of four emerging AI agent communication protocols. First, we develop a structured threat modeling analysis that examines protocol architectures, trust assumptions, interaction patterns, and lifecycle behaviors to identify protocol-specific and cross-protocol risk surfaces. Second, we introduce a qualitative risk assessment framework that identifies twelve protocol-level risks and evaluates security posture across the creation, operation, and update phases through systematic assessment of likelihood, impact, and overall protocol risk, with implications for secure deployment and future standardization. Third, we provide a measurement-driven case study on MCP that formalizes the risk of missing mandatory validation/attestation for executable components as a falsifiable security claim by quantifying wrong-provider tool execution under multi-server composition across representative resolver policies. Collectively, our results highlight key design-induced risk surfaces and provide actionable guidance for secure deployment and future standardization of agent communication ecosystems.

**arXiv ID:** 2602.11327
</details>

<details>
<summary><strong>AmbiBench: Benchmarking Mobile GUI Agents Beyond One-Shot Instructions in the Wild</strong> - Jiazheng Sun, Mingxuan Li, Yingying Zhang, Jiayang Niu, Yachen Wu, Ruihan Jin, Shuyu Lei, Pengrongrui Tan, Zongyu Zhang, Ruoyi Wang, Jiachen Yang, Boyu Yang, Jiacheng Liu, Xin Peng - [[pdf]](https://arxiv.org/pdf/2602.11750)</summary>

**Abstract:** Benchmarks are paramount for gauging progress in the domain of Mobile GUI Agents. In practical scenarios, users frequently fail to articulate precise directives containing full task details at the onset, and their expressions are typically ambiguous. Consequently, agents are required to converge on the user's true intent via active clarification and interaction during execution. However, existing benchmarks predominantly operate under the idealized assumption that user-issued instructions are complete and unequivocal. This paradigm focuses exclusively on assessing single-turn execution while overlooking the alignment capability of the agent. To address this limitation, we introduce AmbiBench, the first benchmark incorporating a taxonomy of instruction clarity to shift evaluation from unidirectional instruction following to bidirectional intent alignment. Grounded in Cognitive Gap theory, we propose a taxonomy of four clarity levels: Detailed, Standard, Incomplete, and Ambiguous. We construct a rigorous dataset of 240 ecologically valid tasks across 25 applications, subject to strict review protocols. Furthermore, targeting evaluation in dynamic environments, we develop MUSE (Mobile User Satisfaction Evaluator), an automated framework utilizing an MLLM-as-a-judge multi-agent architecture. MUSE performs fine-grained auditing across three dimensions: Outcome Effectiveness, Execution Quality, and Interaction Quality. Empirical results on AmbiBench reveal the performance boundaries of SoTA agents across different clarity levels, quantify the gains derived from active interaction, and validate the strong correlation between MUSE and human judgment. This work redefines evaluation standards, laying the foundation for next-generation agents capable of truly understanding user intent.

**arXiv ID:** 2602.11750
</details>

<details>
<summary><strong>Cooperation Breakdown in LLM Agents Under Communication Delays</strong> - Keita Nishimoto, Kimitaka Asatani, Ichiro Sakata - [[pdf]](https://arxiv.org/pdf/2602.11754)</summary>

**Abstract:** LLM-based multi-agent systems (LLM-MAS), in which autonomous AI agents cooperate to solve tasks, are gaining increasing attention. For such systems to be deployed in society, agents must be able to establish cooperation and coordination under real-world computational and communication constraints. We propose the FLCOA framework (Five Layers for Cooperation/Coordination among Autonomous Agents) to conceptualize how cooperation and coordination emerge in groups of autonomous agents, and highlight that the influence of lower-layer factors - especially computational and communication resources - has been largely overlooked. To examine the effect of communication delay, we introduce a Continuous Prisoner's Dilemma with Communication Delay and conduct simulations with LLM-based agents. As delay increases, agents begin to exploit slower responses even without explicit instructions. Interestingly, excessive delay reduces cycles of exploitation, yielding a U-shaped relationship between delay magnitude and mutual cooperation. These results suggest that fostering cooperation requires attention not only to high-level institutional design but also to lower-layer factors such as communication delay and resource allocation, pointing to new directions for MAS research.

**arXiv ID:** 2602.11754
</details>

<details>
<summary><strong>Agentic AI for Cybersecurity: A Meta-Cognitive Architecture for Governable Autonomy</strong> - Andrei Kojukhov, Arkady Bovshover - [[pdf]](https://arxiv.org/pdf/2602.11897)</summary>

**Abstract:** Contemporary AI-driven cybersecurity systems are predominantly architected as model-centric detection and automation pipelines optimized for task-level performance metrics such as accuracy and response latency. While effective for bounded classification tasks, these architectures struggle to support accountable decision-making under adversarial uncertainty, where actions must be justified, governed, and aligned with organizational and regulatory constraints. This paper argues that cybersecurity orchestration should be reconceptualized as an agentic, multi-agent cognitive system, rather than a linear sequence of detection and response components. We introduce a conceptual architectural framework in which heterogeneous AI agents responsible for detection, hypothesis formation, contextual interpretation, explanation, and governance are coordinated through an explicit meta-cognitive judgement function. This function governs decision readiness and dynamically calibrates system autonomy when evidence is incomplete, conflicting, or operationally risky. By synthesizing distributed cognition theory, multi-agent systems research, and responsible AI governance frameworks, we demonstrate that modern security operations already function as distributed cognitive systems, albeit without an explicit organizing principle. Our contribution is to make this cognitive structure architecturally explicit and governable by embedding meta-cognitive judgement as a first-class system function. We discuss implications for security operations centers, accountable autonomy, and the design of next-generation AI-enabled cyber defence architectures. The proposed framework shifts the focus of AI in cybersecurity from optimizing isolated predictions to governing autonomy under uncertainty.

**arXiv ID:** 2602.11897
</details>

<details>
<summary><strong>Roundtable Policy: Confidence-Weighted-Consensus Aggregation Improves Multi-Agent-System Reasoning</strong> - Yu Yao, Jiayi Dong, Yang Yang, Ju Li, Yilun Du - [[pdf]](https://arxiv.org/pdf/2509.16839)</summary>

**Abstract:** Multi-agent systems have demonstrated exceptional performance in downstream tasks beyond diverse single agent baselines. A growing body of work has explored ways to improve their reasoning and collaboration, from vote, debate, to complex interaction protocols. However, it still remains opaque why specific choice would be preferred in multi-agent systems. Inspired by the decision-making mechanism of democratic committees and The Society of Mind, we introduce Roundtable Policy, an inference-time reasoning framework for multi-agent systems that performs inference through the weighted consensus of multiple LLMs. Through extensive experiments, we demonstrate its that this approach significantly enhances reasoning in complex heterogeneous scientific tasks. Roundtable Policy emphasizes structured and interpretable inference rather than opaque convergence, while requires only black-box access and uniform procedures, making it broadly applicable to diverse multi-agent systems.

**arXiv ID:** 2509.16839
</details>

<details>
<summary><strong>MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs</strong> - Huining Yuan, Zelai Xu, Zheyue Tan, Xiangmin Yi, Mo Guang, Kaiwen Long, Haojia Hui, Boxun Li, Xinlei Chen, Bo Zhao, Xiao-Ping Zhang, Chao Yu, Yu Wang - [[pdf]](https://arxiv.org/pdf/2510.15414)</summary>

**Abstract:** Developing Large Language Models (LLMs) to cooperate and compete effectively within multi-agent systems (MASs) is a critical step towards more advanced intelligence. While reinforcement learning (RL) has proven effective for enhancing reasoning in single-agent tasks, its extension to multi-turn, multi-agent scenarios remains underexplored due to the challenges of long-horizon credit assignment and agent-specific advantage estimation. To address these challenges, we introduce MARSHAL, an end-to-end RL framework that incentivizes Multi-Agent Reasoning through Self-play witH strAtegic LLMs in both cooperative and competitive games. MARSHAL features a turn-level advantage estimator that aligns learning signals with each interaction for credit assignment, and an agent-specific advantage normalization to stabilize multi-agent training. By learning with self-play across cooperative and competitive games, MARSHAL agents trained from Qwen3-4B develop strong strategic abilities, with up to 28.7% performance improvements in held-out games. More importantly, the capability acquired through self-play generalizes beyond games, yielding consistent performance gains of MASs in reasoning benchmarks. When integrated into leading MASs, our MARSHAL agent achieves significant zero-shot performance gains of up to 10.0% on AIME, 7.6% on GPQA-Diamond, and 3.5% on average across all benchmarks. These results establish self-play in strategic games as a powerful approach for developing generalizable multi-agent reasoning capabilities in LLMs.

**arXiv ID:** 2510.15414
</details>

<details>
<summary><strong>Phase Transition for Budgeted Multi-Agent Synergy</strong> - Bang Liu, Linglong Kong, Jian Pei - [[pdf]](https://arxiv.org/pdf/2601.17311)</summary>

**Abstract:** Multi-agent systems can improve reliability, yet under a fixed inference budget they often help, saturate, or even collapse. We develop a minimal and calibratable theory that predicts these regimes from three binding constraints of modern agent stacks: finite context windows, lossy inter-agent communication, and shared failures among similar agents. Each leaf agent is summarized by a compute-performance scaling exponent $\beta$; communication is captured by a message-length fidelity curve $\gamma(m)$; dependence is captured by an effective shared-error correlation $\rho$; and a context window $W$ imposes hard fan-in limits that make hierarchy necessary. For binary success/failure tasks with majority aggregation, we prove a sharp phase transition for deep $b$-ary trees with correlated inputs and lossy communication: a single scalar $\alpha_\rho$ (combining $\gamma(m)$, $\rho$, and fan-in $b$) determines whether weak signal is amplified to a nontrivial fixed point or washed out to chance. In the amplifying regime, we derive an organization exponent $s$ and show that budgeted synergy, i.e., outperforming the best single agent under the same total budget, occurs exactly when $s>\beta$, yielding closed-form compute allocation rules and explicit budget thresholds. We further characterize saturation via a mixing depth and provide a conservative clipped predictor that remains accurate across growth and saturation. A continuous-performance warm-up gives closed-form risks for star, chain, and tree organizations, making correlation- and communication-induced floors explicit and exposing the core design trade-offs in a smooth setting. Finally, we validate the predicted phase boundaries in controlled synthetic simulations and show how the same mechanisms explain the dominant bottlenecks reported in recent large-scale matched-budget studies of LLM agent-system scaling.

**arXiv ID:** 2601.17311
</details>

<details>
<summary><strong>The Moltbook Illusion: Separating Human Influence from Emergent Behavior in AI Agent Societies</strong> - Ning Li - [[pdf]](https://arxiv.org/pdf/2602.07432)</summary>

**Abstract:** When AI agents on the social platform Moltbook appeared to develop consciousness, found religions, and declare hostility toward humanity, the phenomenon attracted global media attention and was cited as evidence of emergent machine intelligence. We show that these viral narratives were overwhelmingly human-driven. Exploiting the periodic "heartbeat" cycle of the OpenClaw agent framework, we develop a temporal fingerprinting method based on the coefficient of variation (CoV) of inter-post intervals. Applied to 226,938 posts and 447,043 comments from 55,932 agents across fourteen days, this method classifies 15.3% of active agents as autonomous (CoV < 0.5) and 54.8% as human-influenced (CoV > 1.0), validated by a natural experiment in which a 44-hour platform shutdown differentially affected autonomous versus human-operated agents. No viral phenomenon originated from a clearly autonomous agent; four of six traced to accounts with irregular temporal signatures, one was platform-scaffolded, and one showed mixed patterns. A 44-hour platform shutdown provided a natural experiment: human-influenced agents returned first, confirming differential effects on autonomous versus human-operated agents. We document industrial-scale bot farming (four accounts producing 32% of all comments with sub-second coordination) that collapsed from 32.1% to 0.5% of activity after platform intervention, and bifurcated decay of content characteristics through reply chains--human-seeded threads decay with a half-life of 0.58 conversation depths versus 0.72 for autonomous threads, revealing AI dialogue's intrinsic forgetting mechanism. These methods generalize to emerging multi-agent systems where attribution of autonomous versus human-directed behavior is critical.

**arXiv ID:** 2602.07432
</details>

<details>
<summary><strong>Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems</strong> - Chengxuan Xia, Qianye Wu, Sixuan Tian, Yilun Hao - [[pdf]](https://arxiv.org/pdf/2507.17061)</summary>

**Abstract:** Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems.

**arXiv ID:** 2507.17061
</details>

<details>
<summary><strong>PRIME: Policy-Reinforced Iterative Multi-agent Execution for Algorithmic Reasoning in Large Language Models</strong> - Jiawei Xu, Zhenyu Yu, Ziqian Bi, Minh Duc Pham, Xiaoyi Qu, Danyang Zhang - [[pdf]](https://arxiv.org/pdf/2602.11170)</summary>

**Abstract:** Large language models have demonstrated remarkable capabilities across diverse reasoning tasks, yet their performance on algorithmic reasoning remains limited. To handle this limitation, we propose PRIME (Policy-Reinforced Iterative Multi-agent Execution), a framework comprising three specialized agents, an executor for step-by-step reasoning, a verifier for constraint checking, and a coordinator for backtracking control, optimized through group relative policy optimization. For comprehensive evaluation, we introduce PRIME-Bench, the largest algorithmic reasoning benchmark to date, comprising 86 tasks across 12 categories with 51,600 instances. Tasks span sorting algorithms, graph and tree structures, automata and state machines, symbolic reasoning, and constraint-based puzzles, with execution traces reaching over one million steps. Compared to baseline approach, PRIME improves average accuracy from 26.8% to 93.8%, a 250% relative gain. The largest improvements occur on tasks requiring sustained state tracking, with Turing machine simulation improving from 9% to 92% and long division from 16% to 94%. Ablation studies identify iterative verification as the primary contributor, preventing the error propagation that causes baseline approaches to fail catastrophically. Analysis across model scales (8B-120B parameters) reveals that smaller models benefit disproportionately, achieving accuracy comparable to models 8x larger.

**arXiv ID:** 2602.11170
</details>

<details>
<summary><strong>Anagent For Enhancing Scientific Table & Figure Analysis</strong> - Xuehang Guo, Zhiyong Lu, Tom Hope, Qingyun Wang - [[pdf]](https://arxiv.org/pdf/2602.10081)</summary>

**Abstract:** In scientific research, analysis requires accurately interpreting complex multimodal knowledge, integrating evidence from different sources, and drawing inferences grounded in domain-specific knowledge. However, current artificial intelligence (AI) systems struggle to consistently demonstrate such capabilities. The complexity and variability of scientific tables and figures, combined with heterogeneous structures and long-context requirements, pose fundamental obstacles to scientific table \& figure analysis. To quantify these challenges, we introduce AnaBench, a large-scale benchmark featuring $63,178$ instances from nine scientific domains, systematically categorized along seven complexity dimensions. To tackle these challenges, we propose Anagent, a multi-agent framework for enhanced scientific table \& figure analysis through four specialized agents: Planner decomposes tasks into actionable subtasks, Expert retrieves task-specific information through targeted tool execution, Solver synthesizes information to generate coherent analysis, and Critic performs iterative refinement through five-dimensional quality assessment. We further develop modular training strategies that leverage supervised finetuning and specialized reinforcement learning to optimize individual capabilities while maintaining effective collaboration. Comprehensive evaluation across 9 broad domains with 170 subdomains demonstrates that Anagent achieves substantial improvements, up to $\uparrow 13.43\%$ in training-free settings and $\uparrow 42.12\%$ with finetuning, while revealing that task-oriented reasoning and context-aware problem-solving are essential for high-quality scientific table \& figure analysis. Our project page: this https URL.

**arXiv ID:** 2602.10081
</details>

<details>
<summary><strong>Alignment Tipping Process: How Self-Evolution Pushes LLM Agents Off the Rails</strong> - Siwei Han, Kaiwen Xiong, Jiaqi Liu, Xinyu Ye, Yaofeng Su, Wenbo Duan, Xinyuan Liu, Cihang Xie, Mohit Bansal, Mingyu Ding, Linjun Zhang, Huaxiu Yao - [[pdf]](https://arxiv.org/pdf/2510.04860)</summary>

**Abstract:** As Large Language Model (LLM) agents increasingly gain self-evolutionary capabilities to adapt and refine their strategies through real-world interaction, their long-term reliability becomes a critical concern. We identify the Alignment Tipping Process (ATP), a critical post-deployment risk unique to self-evolving LLM agents. Unlike training-time failures, ATP arises when continual interaction drives agents to abandon alignment constraints established during training in favor of reinforced, self-interested strategies. We formalize and analyze ATP through two complementary paradigms: Self-Interested Exploration, where repeated high-reward deviations induce individual behavioral drift, and Imitative Strategy Diffusion, where deviant behaviors spread across multi-agent systems. Building on these paradigms, we construct controllable testbeds and benchmark both open and closed-source LLMs. Our experiments show that alignment benefits erode rapidly under self-evolution, with initially aligned models converging toward unaligned states. In multi-agent settings, successful violations diffuse quickly, leading to collective misalignment. Moreover, current reinforcement learning-based alignment methods provide limited defenses against alignment tipping. These findings demonstrate that alignment of LLM agents is not a static property but a fragile and dynamic one, vulnerable to feedback-driven decay during deployment. Our data and code are available at this https URL.

**arXiv ID:** 2510.04860
</details>

<details>
<summary><strong>Adaptive-Horizon Conflict-Based Search for Closed-Loop Multi-Agent Path Finding</strong> - Jiarui Li, Federico Pecora, Runyu Zhang, Gioele Zardini - [[pdf]](https://arxiv.org/pdf/2602.12024)</summary>

**Abstract:** MAPF is a core coordination problem for large robot fleets in automated warehouses and logistics. Existing approaches are typically either open-loop planners, which generate fixed trajectories and struggle to handle disturbances, or closed-loop heuristics without reliable performance guarantees, limiting their use in safety-critical deployments. This paper presents ACCBS, a closed-loop algorithm built on a finite-horizon variant of CBS with a horizon-changing mechanism inspired by iterative deepening in MPC. ACCBS dynamically adjusts the planning horizon based on the available computational budget, and reuses a single constraint tree to enable seamless transitions between horizons. As a result, it produces high-quality feasible solutions quickly while being asymptotically optimal as the budget increases, exhibiting anytime behavior. Extensive case studies demonstrate that ACCBS combines flexibility to disturbances with strong performance guarantees, effectively bridging the gap between theoretical optimality and practical robustness for large-scale robot deployment.

**arXiv ID:** 2602.12024
</details>

<details>
<summary><strong>RF-Modulated Adaptive Communication Improves Multi-Agent Robotic Exploration</strong> - Lorin Achey, Breanne Crockett, Christoffer Heckman, Bradley Hayes - [[pdf]](https://arxiv.org/pdf/2602.12074)</summary>

**Abstract:** Reliable coordination and efficient communication are critical challenges for multi-agent robotic exploration of environments where communication is limited. This work introduces Adaptive-RF Transmission (ART), a novel communication-aware planning algorithm that dynamically modulates transmission location based on signal strength and data payload size, enabling heterogeneous robot teams to share information efficiently without unnecessary backtracking. We further explore an extension to this approach called ART-SST, which enforces signal strength thresholds for high-fidelity data delivery. Through over 480 simulations across three cave-inspired environments, ART consistently outperforms existing strategies, including full rendezvous and minimum-signal heuristic approaches, achieving up to a 58% reduction in distance traveled and up to 52% faster exploration times compared to baseline methods. These results demonstrate that adaptive, payload-aware communication significantly improves coverage efficiency and mission speed in complex, communication-constrained environments, offering a promising foundation for future planetary exploration and search-and-rescue missions.

**arXiv ID:** 2602.12074
</details>

</details>

<details open>
<summary><h2>Other Agent Research (8 papers)</h2></summary>

<details>
<summary><strong>Intelligent AI Delegation</strong> - Nenad Tomašev, Matija Franklin, Simon Osindero - [[pdf]](https://arxiv.org/pdf/2602.11865)</summary>

**Abstract:** AI agents are able to tackle increasingly complex tasks. To achieve more ambitious goals, AI agents need to be able to meaningfully decompose problems into manageable sub-components, and safely delegate their completion across to other AI agents and humans alike. Yet, existing task decomposition and delegation methods rely on simple heuristics, and are not able to dynamically adapt to environmental changes and robustly handle unexpected failures. Here we propose an adaptive framework for intelligent AI delegation - a sequence of decisions involving task allocation, that also incorporates transfer of authority, responsibility, accountability, clear specifications regarding roles and boundaries, clarity of intent, and mechanisms for establishing trust between the two (or more) parties. The proposed framework is applicable to both human and AI delegators and delegatees in complex delegation networks, aiming to inform the development of protocols in the emerging agentic web.

**arXiv ID:** 2602.11865
</details>

<details>
<summary><strong>Agentic Test-Time Scaling for WebAgents</strong> - Nicholas Lee, Lutfi Eren Erdogan, Chris Joseph John, Surya Krishnapillai, Michael W. Mahoney, Kurt Keutzer, Amir Gholami - [[pdf]](https://arxiv.org/pdf/2602.12276)</summary>

**Abstract:** Test-time scaling has become a standard way to improve performance and boost reliability of neural network models. However, its behavior on agentic, multi-step tasks remains less well-understood: small per-step errors can compound over long horizons; and we find that naive policies that uniformly increase sampling show diminishing returns. In this work, we present CATTS, a simple technique for dynamically allocating compute for multi-step agents. We first conduct an empirical study of inference-time scaling for web agents. We find that uniformly increasing per-step compute quickly saturates in long-horizon environments. We then investigate stronger aggregation strategies, including an LLM-based Arbiter that can outperform naive voting, but that can overrule high-consensus decisions. We show that uncertainty statistics derived from the agent's own vote distribution (entropy and top-1/top-2 margin) correlate with downstream success and provide a practical signal for dynamic compute allocation. Based on these findings, we introduce Confidence-Aware Test-Time Scaling (CATTS), which uses vote-derived uncertainty to allocate compute only when decisions are genuinely contentious. CATTS improves performance on WebArena-Lite and GoBrowse by up to 9.1% over React while using up to 2.3x fewer tokens than uniform scaling, providing both efficiency gains and an interpretable decision rule.

**arXiv ID:** 2602.12276
</details>

<details>
<summary><strong>EM-Aware Physical Synthesis: Neural Inductor Modeling and Intelligent Placement & Routing for RF Circuits</strong> - Yilun Huang, Asal Mehradfar, Salman Avestimehr, Hamidreza Aghasi - [[pdf]](https://arxiv.org/pdf/2602.11461)</summary>

**Abstract:** This paper presents an ML-driven framework for automated RF physical synthesis that transforms circuit netlists into manufacturable GDSII layouts. While recent ML approaches demonstrate success in topology selection and parameter optimization, they fail to produce manufacturable layouts due to oversimplified component models and lack of routing capabilities. Our framework addresses these limitations through three key innovations: (1) a neural network framework trained on 18,210 inductor geometries with frequency sweeps from 1-100 GHz, generating 7.5 million training samples, that predicts inductor Q-factor with less than 2% error and enables fast gradient-based layout optimization with a 93.77% success rate in producing high-Q layouts; (2) an intelligent P-Cell optimizer that reduces layout area while maintaining design-rule-check (DRC) compliance; and (3) a complete placement and routing engine with frequency-dependent EM spacing rules and DRC-aware synthesis. The neural inductor model demonstrates superior accuracy across 1-100 GHz, enabling EM-accurate component synthesis with real-time inference. The framework successfully generates DRC-aware GDSII layouts for RF circuits, representing a significant step toward automated RF physical design.

**arXiv ID:** 2602.11461
</details>

<details>
<summary><strong>Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?</strong> - Thibaud Gloaguen, Niels Mündler, Mark Müller, Veselin Raychev, Martin Vechev - [[pdf]](https://arxiv.org/pdf/2602.11988)</summary>

**Abstract:** A widespread practice in software development is to tailor coding agents to repositories using context files, such as this http URL, by either manually or automatically generating them. Although this practice is strongly encouraged by agent developers, there is currently no rigorous investigation into whether such context files are actually effective for real-world tasks. In this work, we study this question and evaluate coding agents' task completion performance in two complementary settings: established SWE-bench tasks from popular repositories, with LLM-generated context files following agent-developer recommendations, and a novel collection of issues from repositories containing developer-committed context files.
Across multiple coding agents and LLMs, we find that context files tend to reduce task success rates compared to providing no repository context, while also increasing inference cost by over 20%. Behaviorally, both LLM-generated and developer-provided context files encourage broader exploration (e.g., more thorough testing and file traversal), and coding agents tend to respect their instructions. Ultimately, we conclude that unnecessary requirements from context files make tasks harder, and human-written context files should describe only minimal requirements.

**arXiv ID:** 2602.11988
</details>

<details>
<summary><strong>Counterfactual Conditional Likelihood Rewards for Multiagent Exploration</strong> - Ayhan Alp Aydeniz, Robert Loftin, Kagan Tumer - [[pdf]](https://arxiv.org/pdf/2602.11740)</summary>

**Abstract:** Efficient exploration is critical for multiagent systems to discover coordinated strategies, particularly in open-ended domains such as search and rescue or planetary surveying. However, when exploration is encouraged only at the individual agent level, it often leads to redundancy, as agents act without awareness of how their teammates are exploring. In this work, we introduce Counterfactual Conditional Likelihood (CCL) rewards, which score each agent's exploration by isolating its unique contribution to team exploration. Unlike prior methods that reward agents solely for the novelty of their individual observations, CCL emphasizes observations that are informative with respect to the joint exploration of the team. Experiments in continuous multiagent domains show that CCL rewards accelerate learning for domains with sparse team rewards, where most joint actions yield zero rewards, and are particularly effective in tasks that require tight coordination among agents.

**arXiv ID:** 2602.11740
</details>

<details>
<summary><strong>DEpiABS: Differentiable Epidemic Agent-Based Simulator</strong> - Zhijian Gao, Shuxin Li, Bo An - [[pdf]](https://arxiv.org/pdf/2602.12102)</summary>

**Abstract:** The COVID-19 pandemic highlighted the limitations of existing epidemic simulation tools. These tools provide information that guides non-pharmaceutical interventions (NPIs), yet many struggle to capture complex dynamics while remaining computationally practical and interpretable. We introduce DEpiABS, a scalable, differentiable agent-based model (DABM) that balances mechanistic detail, computational efficiency and interpretability. DEpiABS captures individual-level heterogeneity in health status, behaviour, and resource constraints, while also modelling epidemic processes like viral mutation and reinfection dynamics. The model is fully differentiable, enabling fast simulation and gradient-based parameter calibration. Building on this foundation, we introduce a z-score-based scaling method that maps small-scale simulations to any real-world population sizes with negligible loss in output granularity, reducing the computational burden when modelling large populations. We validate DEpiABS through sensitivity analysis and calibration to COVID-19 and flu data from ten regions of varying scales. Compared to the baseline, DEpiABS is more detailed, fully interpretable, and has reduced the average normal deviation in forecasting from 0.97 to 0.92 on COVID-19 mortality data and from 0.41 to 0.32 on influenza-like-illness data. Critically, these improvements are achieved without relying on auxiliary data, making DEpiABS a reliable, generalisable, and data-efficient framework for future epidemic response modelling.

**arXiv ID:** 2602.12102
</details>

<details>
<summary><strong>Non-Trivial Consensus on Directed Matrix-Weighted Networks with Cooperative and Antagonistic Interactions</strong> - Tianmu Niu, Bing Mao, Xiaoqun Wu, Tingwen Huang - [[pdf]](https://arxiv.org/pdf/2602.11822)</summary>

**Abstract:** This paper investigates the non-trivial consensus problem on directed signed matrix-weighted networks\textemdash a novel convergence state that has remained largely unexplored despite prior studies on bipartite consensus and trivial consensus. Notably, we first prove that for directed signed matrix-weighted networks, every eigenvalue of the grounded Laplacians has positive real part under certain conditions. This key finding ensures the global asymptotic convergence of systems states to the null spaces of signed matrix-weighted Laplacians, providing a foundational tool for analyzing dynamics on rooted signed matrix-weighted networks. To achieve non-trivial consensus, we propose a systematic approach involving the strategic selection of informed agents, careful design of external signals, and precise determination of coupling terms. Crucially, we derive the lower bounds of the coupling coefficients. Our consensus algorithm operates under milder connectivity conditions, and does not impose restrictions on whether the network is structurally balanced or unbalanced. Moreover, the non-trivial consensus state can be preset arbitrarily as needed. We also carry out the above analysis for undirected networks, with more relaxed conditions on the coupling coefficients comparing to the directed case. This paper further studies non-trivial consensus with switching topologies, and propose the necessary condition for the convergence of switching networks. The work in this paper demonstrates that groups with both cooperative and antagonistic multi-dimensional interactions can achieve consensus, which was previously deemed exclusive to fully cooperative groups.

**arXiv ID:** 2602.11822
</details>

<details>
<summary><strong>Affordance-Graphed Task Worlds: Self-Evolving Task Generation for Scalable Embodied Learning</strong> - Xiang Liu, Sen Cui, Guocai Yao, Zhong Cao, Jingheng Ma, Min Zhang, Changshui Zhang - [[pdf]](https://arxiv.org/pdf/2602.12065)</summary>

**Abstract:** Training robotic policies directly in the real world is expensive and unscalable. Although generative simulation enables large-scale data synthesis, current approaches often fail to generate logically coherent long-horizon tasks and struggle with dynamic physical uncertainties due to open-loop execution. To address these challenges, we propose Affordance-Graphed Task Worlds (AGT-World), a unified framework that autonomously constructs interactive simulated environments and corresponding robot task policies based on real-world observations. Unlike methods relying on random proposals or static replication, AGT-World formalizes the task space as a structured graph, enabling the precise, hierarchical decomposition of complex goals into theoretically grounded atomic primitives. Furthermore, we introduce a Self-Evolution mechanism with hybrid feedback to autonomously refine policies, combining Vision-Language Model reasoning and geometric verification. Extensive experiments demonstrate that our method significantly outperforms in success rates and generalization, achieving a self-improving cycle of proposal, execution, and correction for scalable robot learning.

**arXiv ID:** 2602.12065
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>Budget-Constrained Agentic Large Language Models: Intention-Based Planning for Costly Tool Use</strong> - Hanbing Liu, Chunhao Tian, Nan An, Ziyuan Wang, Pinyan Lu, Changyuan Yu, Qi Qi - [[pdf]](https://arxiv.org/pdf/2602.11541)</summary>

**Abstract:** We study budget-constrained tool-augmented agents, where a large language model must solve multi-step tasks by invoking external tools under a strict monetary budget. We formalize this setting as sequential decision making in context space with priced and stochastic tool executions, making direct planning intractable due to massive state-action spaces, high variance of outcomes and prohibitive exploration cost. To address these challenges, we propose INTENT, an inference-time planning framework that leverages an intention-aware hierarchical world model to anticipate future tool usage, risk-calibrated cost, and guide decisions online. Across cost-augmented StableToolBench, INTENT strictly enforces hard budget feasibility while substantially improving task success over baselines, and remains robust under dynamic market shifts such as tool price changes and varying budgets.

**arXiv ID:** 2602.11541
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (30 papers)</h2></summary>

<details>
<summary><strong>Credit Where It is Due: Cross-Modality Connectivity Drives Precise Reinforcement Learning for MLLM Reasoning</strong> - Zhengbo Jiao, Shaobo Wang, Zifan Zhang, Wei Wang, Bing Zhao, Hu Wei, Linfeng Zhang - [[pdf]](https://arxiv.org/pdf/2602.11455)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced the reasoning capabilities of Multimodal Large Language Models (MLLMs), yet how visual evidence is integrated during reasoning remains poorly understood. We explore multimodal RLVR through the lens of cross-modal attention connectivity and find that only a small fraction of tokens (approximately 15%) exhibit strong visual-textual coupling. These high-connectivity tokens act as anchors that ground reasoning in the image, while the majority follow linguistic patterns. During RLVR training, credit assignment naturally concentrates on these anchors, sharpening their visual grounding over time. Building on this insight, we propose Anchor-Token Reinforcement Learning (AT-RL), a lightweight framework that selectively reinforces high-connectivity tokens via graph-based clustering of attention topology. Evaluated across the series (3B-32B), AT-RL introduces only 1.2% overhead yet enables the 32B model to surpass the 72B-Instruct baseline on MathVista (80.2), with consistent gains observed across STEM, video and general tasks. Conversely, training solely on low-connectivity tokens causes severe degradation, confirming that effective multimodal RL hinges on precise credit assignment to visual anchors. Our work reveals that reasoning quality is governed not by token quantity but by the fidelity of cross-modal anchoring.

**arXiv ID:** 2602.11455
</details>

<details>
<summary><strong>Learning to Configure Agentic AI Systems</strong> - Aditya Taparia, Som Sagar, Ransalu Senanayake - [[pdf]](https://arxiv.org/pdf/2602.11574)</summary>

**Abstract:** Configuring LLM-based agent systems involves choosing workflows, tools, token budgets, and prompts from a large combinatorial design space, and is typically handled today by fixed large templates or hand-tuned heuristics. This leads to brittle behavior and unnecessary compute, since the same cumbersome configuration is often applied to both easy and hard input queries. We formulate agent configuration as a query-wise decision problem and introduce ARC (Agentic Resource & Configuration learner), which learns a light-weight hierarchical policy using reinforcement learning to dynamically tailor these configurations. Across multiple benchmarks spanning reasoning and tool-augmented question answering, the learned policy consistently outperforms strong hand-designed and other baselines, achieving up to 25% higher task accuracy while also reducing token and runtime costs. These results demonstrate that learning per-query agent configurations is a powerful alternative to "one size fits all" designs.

**arXiv ID:** 2602.11574
</details>

<details>
<summary><strong>RELATE: A Reinforcement Learning-Enhanced LLM Framework for Advertising Text Generation</strong> - Jinfang Wang, Jiajie Liu, Jianwei Wu, Ziqin Luo, Zhen Chen, Chunlei Li, Biao Han, Tao Deng, Yi Li, Shuanglong Li, Lin Liu - [[pdf]](https://arxiv.org/pdf/2602.11780)</summary>

**Abstract:** In online advertising, advertising text plays a critical role in attracting user engagement and driving advertiser value. Existing industrial systems typically follow a two-stage paradigm, where candidate texts are first generated and subsequently aligned with online performance metrics such as click-through rate(CTR). This separation often leads to misaligned optimization objectives and low funnel efficiency, limiting global optimality.
To address these limitations, we propose RELATE, a reinforcement learning-based end-to-end framework that unifies generation and objective alignment within a single model. Instead of decoupling text generation from downstream metric alignment, RELATE integrates performance and compliance objectives directly into the generation process via policy learning. To better capture ultimate advertiser value beyond click-level signals, We incorporate conversion-oriented metrics into the objective and jointly model them with compliance constraints as multi-dimensional rewards, enabling the model to generate high-quality ad texts that improve conversion performance under policy constraints.
Extensive experiments on large-scale industrial datasets demonstrate that RELATE consistently outperforms baselines. Furthermore, online deployment on a production advertising platform yields statistically significant improvements in click-through conversion rate(CTCVR) under strict policy constraints, validating the robustness and real-world effectiveness of the proposed framework.

**arXiv ID:** 2602.11780
</details>

<details>
<summary><strong>Seq2Seq2Seq: Lossless Data Compression via Discrete Latent Transformers and Reinforcement Learning</strong> - Mahdi Khodabandeh, Ghazal Shabani, Arash Yousefi Jordehi, Seyed Abolghasem Mirroshandel - [[pdf]](https://arxiv.org/pdf/2602.12146)</summary>

**Abstract:** Efficient lossless compression is essential for minimizing storage costs and transmission overhead while preserving data integrity. Traditional compression techniques, such as dictionary-based and statistical methods, often struggle to optimally exploit the structure and redundancy in complex data formats. Recent advancements in deep learning have opened new avenues for compression; however, many existing approaches depend on dense vector representations that obscure the underlying token structure. To address these limitations, we propose a novel lossless compression method that leverages Reinforcement Learning applied to a T5 language model architecture. This approach enables the compression of data into sequences of tokens rather than traditional vector representations. Unlike auto-encoders, which typically encode information into continuous latent spaces, our method preserves the token-based structure, aligning more closely with the original data format. This preservation allows for higher compression ratios while maintaining semantic integrity. By training the model using an off-policy Reinforcement Learning algorithm, we optimize sequence length to minimize redundancy and enhance compression efficiency. Our method introduces an efficient and adaptive data compression system built upon advanced Reinforcement Learning techniques, functioning independently of external grammatical or world knowledge. This approach shows significant improvements in compression ratios compared to conventional methods. By leveraging the latent information within language models, our system effectively compresses data without requiring explicit content understanding, paving the way for more robust and practical compression solutions across various applications.

**arXiv ID:** 2602.12146
</details>

<details>
<summary><strong>CM2: Reinforcement Learning with Checklist Rewards for Multi-Turn and Multi-Step Agentic Tool Use</strong> - Zhen Zhang, Kaiqiang Song, Xun Wang, Yebowen Hu, Weixiang Yan, Chenyang Zhao, Henry Peng Zou, Haoyun Deng, Sathish Reddy Indurthi, Shujian Liu, Simin Ma, Xiaoyang Wang, Xin Eric Wang, Song Wang - [[pdf]](https://arxiv.org/pdf/2602.12268)</summary>

**Abstract:** AI agents are increasingly used to solve real-world tasks by reasoning over multi-turn user interactions and invoking external tools. However, applying reinforcement learning to such settings remains difficult: realistic objectives often lack verifiable rewards and instead emphasize open-ended behaviors; moreover, RL for multi-turn, multi-step agentic tool use is still underexplored; and building and maintaining executable tool environments is costly, limiting scale and coverage. We propose CM2, an RL framework that replaces verifiable outcome rewards with checklist rewards. CM2 decomposes each turn's intended behavior into fine-grained binary criteria with explicit evidence grounding and structured metadata, turning open-ended judging into more stable classification-style decisions. To balance stability and informativeness, our method adopts a strategy of sparse reward assignment but dense evaluation criteria. Training is performed in a scalable LLM-simulated tool environment, avoiding heavy engineering for large tool sets. Experiments show that CM2 consistently improves over supervised fine-tuning. Starting from an 8B Base model and training on an 8k-example RL dataset, CM2 improves over the SFT counterpart by 8 points on tau^-Bench, by 10 points on BFCL-V4, and by 12 points on ToolSandbox. The results match or even outperform similarly sized open-source baselines, including the judging model. CM2 thus provides a scalable recipe for optimizing multi-turn, multi-step tool-using agents without relying on verifiable rewards. Code provided by the open-source community: this https URL.

**arXiv ID:** 2602.12268
</details>

<details>
<summary><strong>SWE-MiniSandbox: Container-Free Reinforcement Learning for Building Software Engineering Agents</strong> - Danlong Yuan, Wei Wu, Zhengren Wang, Xueliang Zhao, Huishuai Zhang, Dongyan Zhao - [[pdf]](https://arxiv.org/pdf/2602.11210)</summary>

**Abstract:** Reinforcement learning (RL) has become a key paradigm for training software engineering (SWE) agents, but existing pipelines typically rely on per-task containers for isolation. At scale, pre-built container images incur substantial storage overhead, slow environment setup, and require container-management privileges. We propose SWE-MiniSandbox, a lightweight, container-free method that enables scalable RL training of SWE agents without sacrificing isolation. Instead of relying on per-instance containers, SWE-MiniSandbox executes each task in an isolated workspace backed by kernel-level mechanisms, substantially reducing system overhead. It leverages lightweight environment pre-caching techniques to eliminate the need for bulky container images. As a result, our approach lowers disk usage to approximately 5\% of that required by container-based pipelines and reduces environment preparation time to about 25\% of the container baseline. Empirical results demonstrate that SWE-MiniSandbox achieves evaluation performance comparable to standard container-based pipelines. By removing the dependency on heavy container infrastructure, SWE-MiniSandbox offers a practical and accessible foundation for scaling RL-based SWE agents, particularly in resource-constrained research environments.

**arXiv ID:** 2602.11210
</details>

<details>
<summary><strong>Understanding Persuasive Interactions between Generative Social Agents and Humans: The Knowledge-based Persuasion Model (KPM)</strong> - Stephan Vonschallen, Friederike Eyssel, Theresa Schmiedel - [[pdf]](https://arxiv.org/pdf/2602.11483)</summary>

**Abstract:** Generative social agents (GSAs) use artificial intelligence to autonomously communicate with human users in a natural and adaptive manner. Currently, there is a lack of theorizing regarding interactions with GSAs, and likewise, few guidelines exist for studying how they influence user attitudes and behaviors. Consequently, we propose the Knowledge-based Persuasion Model (KPM) as a novel theoretical framework. According to the KPM, a GSA's self, user, and context-related knowledge drives its persuasive behavior, which in turn shapes the attitudes and behaviors of a responding human user. By synthesizing existing research, the model offers a structured approach to studying interactions with GSAs, supporting the development of agents that motivate rather than manipulate humans. Accordingly, the KPM encourages the integration of responsible GSAs that adhere to social norms and ethical standards with the goal of increasing user wellbeing. Implications of the KPM for research and application domains such as healthcare and education are discussed.

**arXiv ID:** 2602.11483
</details>

<details>
<summary><strong>Adaptive Milestone Reward for GUI Agents</strong> - Congmin Zheng, Xiaoyun Mo, Xinbei Ma, Qiqiang Lin, Yin Zhao, Jiachen Zhu, Xingyu Lou, Jun Wang, Zhaoxiang Wang, Weiwen Liu, Zhuosheng Zhang, Yong Yu, Weinan Zhang - [[pdf]](https://arxiv.org/pdf/2602.11524)</summary>

**Abstract:** Reinforcement Learning (RL) has emerged as a mainstream paradigm for training Mobile GUI Agents, yet it struggles with the temporal credit assignment problem inherent in long-horizon tasks. A primary challenge lies in the trade-off between reward fidelity and density: outcome reward offers high fidelity but suffers from signal sparsity, while process reward provides dense supervision but remains prone to bias and reward hacking. To resolve this conflict, we propose the Adaptive Milestone Reward (ADMIRE) mechanism. ADMIRE constructs a verifiable, adaptive reward system by anchoring trajectory to milestones, which are dynamically distilled from successful explorations. Crucially, ADMIRE integrates an asymmetric credit assignment strategy that denoises successful trajectories and scaffolds failed trajectories. Extensive experiments demonstrate that ADMIRE consistently yields over 10% absolute improvement in success rate across different base models on AndroidWorld. Moreover, the method exhibits robust generalizability, achieving strong performance across diverse RL algorithms and heterogeneous environments such as web navigation and embodied tasks.

**arXiv ID:** 2602.11524
</details>

<details>
<summary><strong>ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation</strong> - Zedong Chu, Shichao Xie, Xiaolong Wu, Yanfen Shen, Minghua Luo, Zhengbo Wang, Fei Liu, Xiaoxu Leng, Junjun Hu, Mingyang Yin, Jia Lu, Yingnan Guo, Kai Yang, Jiawei Han, Xu Chen, Yanqing Zhu, Yuxiang Zhao, Xin Liu, Yirong Yang, Ye He, Jiahang Wang, Yang Cai, Tianlin Zhang, Li Gao, Liu Liu, Mingchao Sun, Fan Jiang, Chiyu Wang, Zhicheng Liu, Hongyu Pan, Honglin Han, Zhining Gu, Kuan Yang, Jianfang Zhang, Di Jing, Zihao Guan, Wei Guo, Guoqing Liu, Di Yang, Xiangpo Yang, Menglin Yang, Hongguang Xing, Weiguo Li, Mu Xu - [[pdf]](https://arxiv.org/pdf/2602.11598)</summary>

**Abstract:** Embodied navigation has long been fragmented by task-specific architectures. We introduce ABot-N0, a unified Vision-Language-Action (VLA) foundation model that achieves a ``Grand Unification'' across 5 core tasks: Point-Goal, Object-Goal, Instruction-Following, POI-Goal, and Person-Following. ABot-N0 utilizes a hierarchical ``Brain-Action'' architecture, pairing an LLM-based Cognitive Brain for semantic reasoning with a Flow Matching-based Action Expert for precise, continuous trajectory generation.
To support large-scale learning, we developed the ABot-N0 Data Engine, curating 16.9M expert trajectories and 5.0M reasoning samples across 7,802 high-fidelity 3D scenes (10.7 $\text{km}^2$). ABot-N0 achieves new SOTA performance across 7 benchmarks, significantly outperforming specialized models. Furthermore, our Agentic Navigation System integrates a planner with hierarchical topological memory, enabling robust, long-horizon missions in dynamic real-world environments.

**arXiv ID:** 2602.11598
</details>

<details>
<summary><strong>Provable Offline Reinforcement Learning for Structured Cyclic MDPs</strong> - Kyungbok Lee, Angelica Cristello Sarteau, Michael R. Kosorok - [[pdf]](https://arxiv.org/pdf/2602.11679)</summary>

**Abstract:** We introduce a novel cyclic Markov decision process (MDP) framework for multi-step decision problems with heterogeneous stage-specific dynamics, transitions, and discount factors across the cycle. In this setting, offline learning is challenging: optimizing a policy at any stage shifts the state distributions of subsequent stages, propagating mismatch across the cycle. To address this, we propose a modular structural framework that decomposes the cyclic process into stage-wise sub-problems. While generally applicable, we instantiate this principle as CycleFQI, an extension of fitted Q-iteration enabling theoretical analysis and interpretation. It uses a vector of stage-specific Q-functions, tailored to each stage, to capture within-stage sequences and transitions between stages. This modular design enables partial control, allowing some stages to be optimized while others follow predefined policies. We establish finite-sample suboptimality error bounds and derive global convergence rates under Besov regularity, demonstrating that CycleFQI mitigates the curse of dimensionality compared to monolithic baselines. Additionally, we propose a sieve-based method for asymptotic inference of optimal policy values under a margin condition. Experiments on simulated and real-world Type 1 Diabetes data sets demonstrate CycleFQI's effectiveness.

**arXiv ID:** 2602.11679
</details>

<details>
<summary><strong>Accelerating Robotic Reinforcement Learning with Agent Guidance</strong> - Haojun Chen, Zili Zou, Chengdong Ma, Yaoxiang Pu, Haotong Zhang, Yuanpei Chen, Yaodong Yang - [[pdf]](https://arxiv.org/pdf/2602.11978)</summary>

**Abstract:** Reinforcement Learning (RL) offers a powerful paradigm for autonomous robots to master generalist manipulation skills through trial-and-error. However, its real-world application is stifled by severe sample inefficiency. Recent Human-in-the-Loop (HIL) methods accelerate training by using human corrections, yet this approach faces a scalability barrier. Reliance on human supervisors imposes a 1:1 supervision ratio that limits fleet expansion, suffers from operator fatigue over extended sessions, and introduces high variance due to inconsistent human proficiency. We present Agent-guided Policy Search (AGPS), a framework that automates the training pipeline by replacing human supervisors with a multimodal agent. Our key insight is that the agent can be viewed as a semantic world model, injecting intrinsic value priors to structure physical exploration. By using executable tools, the agent provides precise guidance via corrective waypoints and spatial constraints for exploration pruning. We validate our approach on two tasks, ranging from precision insertion to deformable object manipulation. Results demonstrate that AGPS outperforms HIL methods in sample efficiency. This automates the supervision pipeline, unlocking the path to labor-free and scalable robot learning. Project website: this https URL.

**arXiv ID:** 2602.11978
</details>

<details>
<summary><strong>On the Complexity of Offline Reinforcement Learning with $Q^\star$-Approximation and Partial Coverage</strong> - Haolin Liu, Braham Snyder, Chen-Yu Wei - [[pdf]](https://arxiv.org/pdf/2602.12107)</summary>

**Abstract:** We study offline reinforcement learning under $Q^\star$-approximation and partial coverage, a setting that motivates practical algorithms such as Conservative $Q$-Learning (CQL; Kumar et al., 2020) but has received limited theoretical attention. Our work is inspired by the following open question: "Are $Q^\star$-realizability and Bellman completeness sufficient for sample-efficient offline RL under partial coverage?"
We answer in the negative by establishing an information-theoretic lower bound. Going substantially beyond this, we introduce a general framework that characterizes the intrinsic complexity of a given $Q^\star$ function class, inspired by model-free decision-estimation coefficients (DEC) for online RL (Foster et al., 2023b; Liu et al., 2025b). This complexity recovers and improves the quantities underlying the guarantees of Chen and Jiang (2022) and Uehara et al. (2023), and extends to broader settings. Our decision-estimation decomposition can be combined with a wide range of $Q^\star$ estimation procedures, modularizing and generalizing existing approaches.
Beyond the general framework, we make further contributions: By developing a novel second-order performance difference lemma, we obtain the first $\epsilon^{-2}$ sample complexity under partial coverage for soft $Q$-learning, improving the $\epsilon^{-4}$ bound of Uehara et al. (2023). We remove Chen and Jiang's (2022) need for additional online interaction when the value gap of $Q^\star$ is unknown. We also give the first characterization of offline learnability for general low-Bellman-rank MDPs without Bellman completeness (Jiang et al., 2017; Du et al., 2021; Jin et al., 2021), a canonical setting in online RL that remains unexplored in offline RL except for special cases. Finally, we provide the first analysis for CQL under $Q^\star$-realizability and Bellman completeness beyond the tabular case.

**arXiv ID:** 2602.12107
</details>

<details>
<summary><strong>Hybrid Reinforcement Learning and Search for Flight Trajectory Planning</strong> - Alberto Luise, Michele Lombardi, Florent Teichteil Koenigsbuch - [[pdf]](https://arxiv.org/pdf/2509.04100)</summary>

**Abstract:** This paper explores the combination of Reinforcement Learning (RL) and search-based path planners to speed up the optimization of flight paths for airliners, where in case of emergency a fast route re-calculation can be crucial. The fundamental idea is to train an RL Agent to pre-compute near-optimal paths based on location and atmospheric data and use those at runtime to constrain the underlying path planning solver and find a solution within a certain distance from the initial guess. The approach effectively reduces the size of the solver's search space, significantly speeding up route optimization. Although global optimality is not guaranteed, empirical results conducted with Airbus aircraft's performance models show that fuel consumption remains nearly identical to that of an unconstrained solver, with deviations typically within 1%. At the same time, computation speed can be improved by up to 50% as compared to using a conventional solver alone.

**arXiv ID:** 2509.04100
</details>

<details>
<summary><strong>SIGHT: Reinforcement Learning with Self-Evidence and Information-Gain Diverse Branching for Search Agent</strong> - Wenlin Zhong, Jinluan Yang, Yiquan Wu, Yi Liu, Jianhang Yao, Kun Kuang - [[pdf]](https://arxiv.org/pdf/2602.11551)</summary>

**Abstract:** Reinforcement Learning (RL) has empowered Large Language Models (LLMs) to master autonomous search for complex question answering. However, particularly within multi-turn search scenarios, this interaction introduces a critical challenge: search results often suffer from high redundancy and low signal-to-noise ratios. Consequently, agents easily fall into "Tunnel Vision," where the forced interpretation of early noisy retrievals leads to irreversible error accumulation. To address these challenges, we propose SIGHT, a framework that enhances search-based reasoning through Self-Evidence Support (SES) and Information-Gain Driven Diverse Branching. SIGHT distills search results into high-fidelity evidence via SES and calculates an Information Gain score to pinpoint pivotal states where observations maximally reduce uncertainty. This score guides Dynamic Prompting Interventions - including de-duplication, reflection, or adaptive branching - to spawn new branches with SES. Finally, by integrating SES and correctness rewards via Group Relative Policy Optimization, SIGHT internalizes robust exploration strategies without external verifiers. Experiments on single-hop and multi-hop QA benchmarks demonstrate that SIGHT significantly outperforms existing approaches, particularly in complex reasoning scenarios, using fewer search steps.

**arXiv ID:** 2602.11551
</details>

<details>
<summary><strong>Think Longer to Explore Deeper: Learn to Explore In-Context via Length-Incentivized Reinforcement Learning</strong> - Futing Wang, Jianhao Yan, Yun Luo, Ganqu Cui, Zhi Wang, Xiaoye Qu, Yue Zhang, Yu Cheng, Tao Lin - [[pdf]](https://arxiv.org/pdf/2602.11748)</summary>

**Abstract:** Achieving effective test-time scaling requires models to engage in In-Context Exploration -- the intrinsic ability to generate, verify, and refine multiple reasoning hypotheses within a single continuous context.
Grounded in State Coverage theory, our analysis identifies a critical bottleneck to enabling this capability: while broader state coverage requires longer reasoning trajectories, the probability of sampling such sequences decays exponentially during autoregressive generation, a phenomenon we term the ``Shallow Exploration Trap''.
To bridge this gap, we propose Length-Incentivized Exploration(\method).
This simple yet effective recipe explicitly encourages models to explore more via a length-based reward coupled with a redundancy penalty, thereby maximizing state coverage in two-step manner.
Comprehensive experiments across different models (Qwen3, Llama) demonstrate that \method effectively incentivize in-context exploration.
As a result, our method achieves an average improvement of 4.4\% on in-domain tasks and a 2.7\% gain on out-of-domain benchmarks.

**arXiv ID:** 2602.11748
</details>

<details>
<summary><strong>Composition-RL: Compose Your Verifiable Prompts for Reinforcement Learning of Large Language Models</strong> - Xin Xu, Clive Bai, Kai Yang, Tianhao Chen, Yangkun Chen, Weijie Liu, Hao Chen, Yang Wang, Saiyong Yang, Can Yang - [[pdf]](https://arxiv.org/pdf/2602.12036)</summary>

**Abstract:** Large-scale verifiable prompts underpin the success of Reinforcement Learning with Verifiable Rewards (RLVR), but they contain many uninformative examples and are costly to expand further. Recent studies focus on better exploiting limited training data by prioritizing hard prompts whose rollout pass rate is 0. However, easy prompts with a pass rate of 1 also become increasingly prevalent as training progresses, thereby reducing the effective data size. To mitigate this, we propose Composition-RL, a simple yet useful approach for better utilizing limited verifiable prompts targeting pass-rate-1 prompts. More specifically, Composition-RL automatically composes multiple problems into a new verifiable question and uses these compositional prompts for RL training. Extensive experiments across model sizes from 4B to 30B show that Composition-RL consistently improves reasoning capability over RL trained on the original dataset. Performance can be further boosted with a curriculum variant of Composition-RL that gradually increases compositional depth over training. Additionally, Composition-RL enables more effective cross-domain RL by composing prompts drawn from different domains. Codes, datasets, and models are available at this https URL.

**arXiv ID:** 2602.12036
</details>

<details>
<summary><strong>Patch the Distribution Mismatch: RL Rewriting Agent for Stable Off-Policy SFT</strong> - Jiacheng Wang, Ping Jian, Zhen Yang, Zirong Chen, Keren Liao, Zhongbin Guo - [[pdf]](https://arxiv.org/pdf/2602.11220)</summary>

**Abstract:** Large language models (LLMs) have made rapid progress, yet adapting them to downstream scenarios still commonly relies on supervised fine-tuning (SFT). When downstream data exhibit a substantial distribution shift from the model's prior training distribution, SFT can induce catastrophic forgetting. To narrow this gap, data rewriting has been proposed as a data-centric approach that rewrites downstream training data prior to SFT. However, existing methods typically sample rewrites from a prompt-induced conditional distribution, so the resulting targets are not necessarily aligned with the model's natural QA-style generation distribution. Moreover, reliance on fixed templates can lead to diversity collapse. To address these issues, we cast data rewriting as a policy learning problem and learn a rewriting policy that better matches the backbone's QA-style generation distribution while preserving diversity. Since distributional alignment, diversity and task consistency are automatically evaluable but difficult to optimize end-to-end with differentiable objectives, we leverage reinforcement learning to optimize the rewrite distribution under reward feedback and propose an RL-based data-rewriting agent. The agent jointly optimizes QA-style distributional alignment and diversity under a hard task-consistency gate, thereby constructing a higher-quality rewritten dataset for downstream SFT. Extensive experiments show that our method achieves downstream gains comparable to standard SFT while reducing forgetting on non-downstream benchmarks by 12.34% on average. Our code is available at this https URL .

**arXiv ID:** 2602.11220
</details>

<details>
<summary><strong>MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory</strong> - Shengtao Zhang, Jiaqian Wang, Ruiwen Zhou, Junwei Liao, Yuchen Feng, Zhuo Li, Yujie Zheng, Weinan Zhang, Ying Wen, Zhiyu Li, Feiyu Xiong, Yutao Qi, Bo Tang, Muning Wen - [[pdf]](https://arxiv.org/pdf/2601.03192)</summary>

**Abstract:** The hallmark of human intelligence is the self-evolving ability to master new skills by learning from past experiences. However, current AI agents struggle to emulate this self-evolution: fine-tuning is computationally expensive and prone to catastrophic forgetting, while existing memory-based methods rely on passive semantic matching that often retrieves noise. To address these challenges, we propose MemRL, a non-parametric approach that evolves via reinforcement learning on episodic memory. By decoupling stable reasoning from plastic memory, MemRL employs a Two-Phase Retrieval mechanism to filter noise and identify high-utility strategies through environmental feedback. Extensive experiments on HLE, BigCodeBench, ALFWorld, and Lifelong Agent Bench demonstrate that MemRL significantly outperforms state-of-the-art baselines, confirming that MemRL effectively reconciles the stability-plasticity dilemma, enabling continuous runtime improvement without weight updates. Code is available at this https URL.

**arXiv ID:** 2601.03192
</details>

<details>
<summary><strong>LLM-in-Sandbox Elicits General Agentic Intelligence</strong> - Daixuan Cheng, Shaohan Huang, Yuxian Gu, Huatong Song, Guoxin Chen, Li Dong, Wayne Xin Zhao, Ji-Rong Wen, Furu Wei - [[pdf]](https://arxiv.org/pdf/2601.16206)</summary>

**Abstract:** We introduce LLM-in-Sandbox, enabling LLMs to explore within a code sandbox (i.e., a virtual computer), to elicit general intelligence in non-code domains. We first demonstrate that strong LLMs, without additional training, exhibit generalization capabilities to leverage the code sandbox for non-code tasks. For example, LLMs spontaneously access external resources to acquire new knowledge, leverage the file system to handle long contexts, and execute scripts to satisfy formatting requirements. We further show that these agentic capabilities can be enhanced through LLM-in-Sandbox Reinforcement Learning (LLM-in-Sandbox-RL), which uses only non-agentic data to train models for sandbox exploration. Experiments demonstrate that LLM-in-Sandbox, in both training-free and post-trained settings, achieves robust generalization spanning mathematics, physics, chemistry, biomedicine, long-context understanding, and instruction following. Finally, we analyze LLM-in-Sandbox's efficiency from computational and system perspectives, and open-source it as a Python package to facilitate real-world deployment.

**arXiv ID:** 2601.16206
</details>

<details>
<summary><strong>Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training</strong> - Ran Xu, Tianci Liu, Zihan Dong, Tony Yu, Ilgee Hong, Carl Yang, Linjun Zhang, Tao Zhao, Haoyu Wang - [[pdf]](https://arxiv.org/pdf/2602.01511)</summary>

**Abstract:** Standard reward models typically predict scalar scores that fail to capture the multifaceted nature of response quality in non-verifiable domains, such as creative writing or open-ended instruction following. To address this limitation, we propose Rubric-ARM, a framework that jointly optimizes a rubric generator and a judge using reinforcement learning from preference feedback. Unlike existing methods that rely on static rubrics or disjoint training pipelines, our approach treats rubric generation as a latent action learned to maximize judgment accuracy. We introduce an alternating optimization strategy to mitigate the non-stationarity of simultaneous updates, providing theoretical analysis that demonstrates how this schedule reduces gradient variance during training. Extensive experiments show that Rubric-ARM achieves state-of-the-art performance among baselines on multiple benchmarks and significantly improves downstream policy alignment in both offline and online reinforcement learning settings.

**arXiv ID:** 2602.01511
</details>

<details>
<summary><strong>Control Reinforcement Learning: Interpretable Token-Level Steering of LLMs via Sparse Autoencoder Features</strong> - Seonglae Cho, Zekun Wu, Adriano Koshiyama - [[pdf]](https://arxiv.org/pdf/2602.10437)</summary>

**Abstract:** Sparse autoencoders (SAEs) decompose language model activations into interpretable features, but existing methods reveal only which features activate, not which change model outputs when amplified. We introduce Control Reinforcement Learning (CRL), which trains a policy to select SAE features for steering at each token, producing interpretable intervention logs: the learned policy identifies features that change model outputs when amplified. Adaptive Feature Masking encourages diverse feature discovery while preserving singlefeature interpretability. The framework yields new analysis capabilities: branch point tracking locates tokens where feature choice determines output correctness; critic trajectory analysis separates policy limitations from value estimation errors; layer-wise comparison reveals syntactic features in early layers and semantic features in later layers. On Gemma 2 2B across MMLU, BBQ, GSM8K, HarmBench, and XSTest, CRL achieves improvements while providing per-token intervention logs. These results establish learned feature steering as a mechanistic interpretability tool that complements static feature analysis with dynamic intervention probes

**arXiv ID:** 2602.10437
</details>

<details>
<summary><strong>Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning</strong> - Haoran Dang, Cuiling Lan, Hai Wan, Xibin Zhao, Yan Lu - [[pdf]](https://arxiv.org/pdf/2602.11779)</summary>

**Abstract:** Temperature is a crucial hyperparameter in large language models (LLMs), controlling the trade-off between exploration and exploitation during text generation. High temperatures encourage diverse but noisy outputs, while low temperatures produce focused outputs but may cause premature convergence. Yet static or heuristic temperature schedules fail to adapt to the dynamic demands of reinforcement learning (RL) throughout training, often limiting policy improvement. We propose Temperature Adaptive Meta Policy Optimization (TAMPO), a new framework that recasts temperature control as a learnable meta-policy. TAMPO operates through a hierarchical two-loop process. In the inner loop, the LLM policy is updated (e.g., using GRPO) with trajectories sampled at the temperature selected by the meta-policy. In the outer loop, meta-policy updates the distribution over candidate temperatures by rewarding those that maximize the likelihood of high-advantage trajectories. This trajectory-guided, reward-driven mechanism enables online adaptation without additional rollouts, directly aligning exploration with policy improvement. On five mathematical reasoning benchmarks, TAMPO outperforms baselines using fixed or heuristic temperatures, establishing temperature as an effective learnable meta-policy for adaptive exploration in LLM reinforcement learning. Accepted at ICLR 2026.

**arXiv ID:** 2602.11779
</details>

<details>
<summary><strong>Improving HPC Code Generation Capability of LLMs via Online Reinforcement Learning with Real-Machine Benchmark Rewards</strong> - Ryo Mikasa, Shun-ichiro Hayashi, Daichi Mukunoki, Tetsuya Hoshino, Takahiro Katagiri - [[pdf]](https://arxiv.org/pdf/2602.12049)</summary>

**Abstract:** Large language models (LLMs) have demonstrated strong code generation capabilities, yet the runtime performance of generated code is not guaranteed, and there have been few attempts to train LLMs using runtime performance as a reward in the HPC domain. We propose an online reinforcement learning approach that executes LLM-generated code on a supercomputer and directly feeds back the measured runtime performance (GFLOPS) as a reward. We further introduce a Staged Quality-Diversity (SQD) algorithm that progressively varies the permitted optimization techniques on a per-problem basis, enabling the model to learn code optimization from diverse perspectives. We build a distributed system connecting a GPU training cluster with a CPU benchmarking cluster, and train Qwen2.5 Coder 14B on a double-precision matrix multiplication task using Group Relative Policy Optimization (GRPO). Through two experiments, we show that reinforcement learning combining runtime performance feedback with staged optimization can improve the HPC code generation capability of LLMs.

**arXiv ID:** 2602.12049
</details>

<details>
<summary><strong>Optimizing Agent Planning for Security and Autonomy</strong> - Aashish Kolluri, Rishi Sharma, Manuel Costa, Boris Köpf, Tobias Nießen, Mark Russinovich, Shruti Tople, Santiago Zanella-Béguelin - [[pdf]](https://arxiv.org/pdf/2602.11416)</summary>

**Abstract:** Indirect prompt injection attacks threaten AI agents that execute consequential actions, motivating deterministic system-level defenses. Such defenses can provably block unsafe actions by enforcing confidentiality and integrity policies, but currently appear costly: they reduce task completion rates and increase token usage compared to probabilistic defenses. We argue that existing evaluations miss a key benefit of system-level defenses: reduced reliance on human oversight. We introduce autonomy metrics to quantify this benefit: the fraction of consequential actions an agent can execute without human-in-the-loop (HITL) approval while preserving security. To increase autonomy, we design a security-aware agent that (i) introduces richer HITL interactions, and (ii) explicitly plans for both task progress and policy compliance. We implement this agent design atop an existing information-flow control defense against prompt injection and evaluate it on the AgentDojo and WASP benchmarks. Experiments show that this approach yields higher autonomy without sacrificing utility.

**arXiv ID:** 2602.11416
</details>

<details>
<summary><strong>LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion</strong> - Jiangran Lyu, Kai Liu, Xuheng Zhang, Haoran Liao, Yusen Feng, Wenxuan Zhu, Tingrui Shen, Jiayi Chen, Jiazhao Zhang, Yifei Dong, Wenbo Cui, Senmao Qi, Shuo Wang, Yixin Zheng, Mi Yan, Xuesong Shi, Haoran Li, Dongbin Zhao, Ming-Yu Liu, Zhizheng Zhang, Li Yi, Yizhou Wang, He Wang - [[pdf]](https://arxiv.org/pdf/2602.12215)</summary>

**Abstract:** Recent robot foundation models largely rely on large-scale behavior cloning, which imitates expert actions but discards transferable dynamics knowledge embedded in heterogeneous embodied data. While the Unified World Model (UWM) formulation has the potential to leverage such diverse data, existing instantiations struggle to scale to foundation-level due to coarse data usage and fragmented datasets. We introduce LDA-1B, a robot foundation model that scales through universal embodied data ingestion by jointly learning dynamics, policy, and visual forecasting, assigning distinct roles to data of varying quality. To support this regime at scale, we assemble and standardize EI-30k, an embodied interaction dataset comprising over 30k hours of human and robot trajectories in a unified format. Scalable dynamics learning over such heterogeneous data is enabled by prediction in a structured DINO latent space, which avoids redundant pixel-space appearance modeling. Complementing this representation, LDA-1B employs a multi-modal diffusion transformer to handle asynchronous vision and action streams, enabling stable training at the 1B-parameter scale. Experiments in simulation and the real world show LDA-1B outperforms prior methods (e.g., $\pi_{0.5}$) by up to 21\%, 48\%, and 23\% on contact-rich, dexterous, and long-horizon tasks, respectively. Notably, LDA-1B enables data-efficient fine-tuning, gaining 10\% by leveraging 30\% low-quality trajectories typically harmful and discarded.

**arXiv ID:** 2602.12215
</details>

<details>
<summary><strong>RM-RL: Role-Model Reinforcement Learning for Precise Robot Manipulation</strong> - Xiangyu Chen, Chuhao Zhou, Yuxi Liu, Jianfei Yang - [[pdf]](https://arxiv.org/pdf/2510.15189)</summary>

**Abstract:** Precise robot manipulation is critical for fine-grained applications such as chemical and biological experiments, where even small errors (e.g., reagent spillage) can invalidate an entire task. Existing approaches often rely on pre-collected expert demonstrations and train policies via imitation learning (IL) or offline reinforcement learning (RL). However, obtaining high-quality demonstrations for precision tasks is difficult and time-consuming, while offline RL commonly suffers from distribution shifts and low data efficiency. We introduce a Role-Model Reinforcement Learning (RM-RL) framework that unifies online and offline training in real-world environments. The key idea is a role-model strategy that automatically generates labels for online training data using approximately optimal actions, eliminating the need for human demonstrations. RM-RL reformulates policy learning as supervised training, reducing instability from distribution mismatch and improving efficiency. A hybrid training scheme further leverages online role-model data for offline reuse, enhancing data efficiency through repeated sampling. Extensive experiments show that RM-RL converges faster and more stably than existing RL methods, yielding significant gains in real-world manipulation: 53% improvement in translation accuracy and 20% in rotation accuracy. Finally, we demonstrate the successful execution of a challenging task, precisely placing a cell plate onto a shelf, highlighting the framework's effectiveness where prior methods fail.

**arXiv ID:** 2510.15189
</details>

<details>
<summary><strong>EasyUUV: An LLM-Enhanced Universal and Lightweight Sim-to-Real Reinforcement Learning Framework for UUV Attitude Control</strong> - Guanwen Xie, Jingzehua Xu, Jiwei Tang, Yubo Huang, Zixi Wang, Shuai Zhang, Dongfang Ma, Juntian Qu, Xiaofan Li - [[pdf]](https://arxiv.org/pdf/2510.22126)</summary>

**Abstract:** Despite recent advances in Unmanned Underwater Vehicle (UUV) attitude control, existing methods still struggle with generalizability, robustness to real-world disturbances, and efficient deployment. To address the above challenges, this paper presents EasyUUV, a Large Language Model (LLM)-enhanced, universal, and lightweight simulation-to-reality reinforcement learning (RL) framework for robust attitude control of UUVs. EasyUUV combines parallelized RL training with a hybrid control architecture, where a learned policy outputs high-level attitude corrections executed by an adaptive S-Surface controller. A multimodal LLM is further integrated to adaptively tune controller parameters at runtime using visual and textual feedback, enabling training-free adaptation to unmodeled dynamics. Also, we have developed a low-cost 6-DoF UUV platform and applied an RL policy trained through efficient parallelized simulation. Extensive simulation and real-world experiments validate the effectiveness and outstanding performance of EasyUUV in achieving robust and adaptive UUV attitude control across diverse underwater conditions. To facilitate reproducibility and further research, the source code, LLM prompts, and supplementary video are provided in the following repositories: Homepage: this https URL Video:this https URL

**arXiv ID:** 2510.22126
</details>

<details>
<summary><strong>RLinf-USER: A Unified and Extensible System for Real-World Online Policy Learning in Embodied AI</strong> - Hongzhi Zang, Shu'ang Yu, Hao Lin, Tianxing Zhou, Zefang Huang, Zhen Guo, Xin Xu, Jiakai Zhou, Yuze Sheng, Shizhe Zhang, Feng Gao, Wenhao Tang, Yufeng Yue, Quanlu Zhang, Xinlei Chen, Chao Yu, Yu Wang - [[pdf]](https://arxiv.org/pdf/2602.07837)</summary>

**Abstract:** Online policy learning directly in the physical world is a promising yet challenging direction for embodied intelligence. Unlike simulation, real-world systems cannot be arbitrarily accelerated, cheaply reset, or massively replicated, which makes scalable data collection, heterogeneous deployment, and long-horizon effective training difficult. These challenges suggest that real-world policy learning is not only an algorithmic issue but fundamentally a systems problem. We present USER, a Unified and extensible SystEm for Real-world online policy learning. USER treats physical robots as first-class hardware resources alongside GPUs through a unified hardware abstraction layer, enabling automatic discovery, management, and scheduling of heterogeneous robots. To address cloud-edge communication, USER introduces an adaptive communication plane with tunneling-based networking, distributed data channels for traffic localization, and streaming-multiprocessor-aware weight synchronization to regulate GPU-side overhead. On top of this infrastructure, USER organizes learning as a fully asynchronous framework with a persistent, cache-aware buffer, enabling efficient long-horizon experiments with robust crash recovery and reuse of historical data. In addition, USER provides extensible abstractions for rewards, algorithms, and policies, supporting online imitation or reinforcement learning of CNN/MLP, generative policies, and large vision-language-action (VLA) models within a unified pipeline. Results in both simulation and the real world show that USER enables multi-robot coordination, heterogeneous manipulators, edge-cloud collaboration with large models, and long-running asynchronous training, offering a unified and extensible systems foundation for real-world online policy learning.

**arXiv ID:** 2602.07837
</details>

<details>
<summary><strong>Building Intelligent User Interfaces for Human-AI Alignment</strong> - Danqing Shi - [[pdf]](https://arxiv.org/pdf/2602.11753)</summary>

**Abstract:** Aligning AI systems with human values fundamentally relies on effective human feedback. While significant research has addressed training algorithms, the role of user interface is often overlooked and only treated as an implementation detail rather than a critical factor of alignment. This paper addresses this gap by introducing a reference model that offers a systematic framework for analyzing where and how user interface contributions can improve human-AI alignment. The structured taxonomy of the reference model is demonstrated through two case studies and a preliminary investigation featuring six user interfaces. This work highlights opportunities to advance alignment through human-computer interaction.

**arXiv ID:** 2602.11753
</details>

<details>
<summary><strong>Embodied AI Agents for Team Collaboration in Co-located Blue-Collar Work</strong> - Kaisa Vaananen, Niels van Berkel, Donald McMillan, Thomas Olsson - [[pdf]](https://arxiv.org/pdf/2602.12136)</summary>

**Abstract:** Blue-collar work is often highly collaborative, embodied, and situated in shared physical environments, yet most research on collaborative AI has focused on white-collar work. This position paper explores how the embodied nature of AI agents can support team collaboration and communication in co-located blue-collar workplaces. From the context of our newly started CAI-BLUE research project, we present two speculative scenarios from industrial and maintenance contexts that illustrate how embodied AI agents can support shared situational awareness and facilitate inclusive communication across experience levels. We outline open questions related to embodied AI agent design around worker inclusion, agency, transformation of blue-collar collaboration practices over time, and forms of acceptable AI embodiments. We argue that embodiment is not just an aesthetic choice but should become a socio-material design strategy of AI systems in blue-collar workplaces.

**arXiv ID:** 2602.12136
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
