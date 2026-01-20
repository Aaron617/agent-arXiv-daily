# Agent arXiv Daily

**Last Updated:** 2026-01-20 02:28:30

**Total Papers:** 52

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (5 papers)</h2></summary>

<details>
<summary><strong>Japanese AI Agent System on Human Papillomavirus Vaccination: System Design</strong> - Junyu Liu, Siwen Yang, Dexiu Ma, Qian Niu, Zequn Zhang, Momoko Nagai-Tanima, Tomoki Aoyama - [[pdf]](https://arxiv.org/pdf/2601.10718)</summary>

**Abstract:** Human papillomavirus (HPV) vaccine hesitancy poses significant public health challenges, particularly in Japan where proactive vaccination recommendations were suspended from 2013 to 2021. The resulting information gap is exacerbated by misinformation on social media, and traditional ways cannot simultaneously address individual queries while monitoring population-level discourse. This study aimed to develop a dual-purpose AI agent system that provides verified HPV vaccine information through a conversational interface while generating analytical reports for medical institutions based on user interactions and social media. We implemented a system comprising: a vector database integrating academic papers, government sources, news media, and social media; a Retrieval-Augmented Generation chatbot using ReAct agent architecture with multi-tool orchestration across five knowledge sources; and an automated report generation system with modules for news analysis, research synthesis, social media sentiment analysis, and user interaction pattern identification. Performance was assessed using a 0-5 scoring scale. For single-turn evaluation, the chatbot achieved mean scores of 4.83 for relevance, 4.89 for routing, 4.50 for reference quality, 4.90 for correctness, and 4.88 for professional identity (overall 4.80). Multi-turn evaluation yielded higher scores: context retention 4.94, topic coherence 5.00, and overall 4.98. The report generation system achieved completeness 4.00-5.00, correctness 4.00-5.00, and helpfulness 3.67-5.00, with reference validity 5.00 across all periods. This study demonstrates the feasibility of an integrated AI agent system for bidirectional HPV vaccine communication. The architecture enables verified information delivery with source attribution while providing systematic public discourse analysis, with a transferable framework for adaptation to other medical contexts.

**arXiv ID:** 2601.10718
</details>

<details>
<summary><strong>AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts</strong> - Keyu Li, Junhao Shi, Yang Xiao, Mohan Jiang, Jie Sun, Yunze Wu, Shijie Xia, Xiaojie Cai, Tianze Xu, Weiye Si, Wenjie Li, Dequan Wang, Pengfei Liu - [[pdf]](https://arxiv.org/pdf/2601.11044)</summary>

**Abstract:** Large Language Models (LLMs) based autonomous agents demonstrate multifaceted capabilities to contribute substantially to economic production. However, existing benchmarks remain focused on single agentic capability, failing to capture long-horizon real-world scenarios. Moreover, the reliance on human-in-the-loop feedback for realistic tasks creates a scalability bottleneck, hindering automated rollout collection and evaluation. To bridge this gap, we introduce AgencyBench, a comprehensive benchmark derived from daily AI usage, evaluating 6 core agentic capabilities across 32 real-world scenarios, comprising 138 tasks with specific queries, deliverables, and rubrics. These scenarios require an average of 90 tool calls, 1 million tokens, and hours of execution time to resolve. To enable automated evaluation, we employ a user simulation agent to provide iterative feedback, and a Docker sandbox to conduct visual and functional rubric-based assessment. Experiments reveal that closed-source models significantly outperform open-source models (48.4% vs 32.1%). Further analysis reveals significant disparities across models in resource efficiency, feedback-driven self-correction, and specific tool-use preferences. Finally, we investigate the impact of agentic scaffolds, observing that proprietary models demonstrate superior performance within their native ecosystems (e.g., Claude-4.5-Opus via Claude-Agent-SDK), while open-source models exhibit distinct performance peaks, suggesting potential optimization for specific execution frameworks. AgencyBench serves as a critical testbed for next-generation agents, highlighting the necessity of co-optimizing model architecture with agentic frameworks. We believe this work sheds light on the future direction of autonomous agents, and we release the full benchmark and evaluation toolkit at this https URL.

**arXiv ID:** 2601.11044
</details>

<details>
<summary><strong>ABC-Bench: Benchmarking Agentic Backend Coding in Real-World Development</strong> - Jie Yang, Honglin Guo, Li Ji, Jiazheng Zhou, Rui Zheng, Zhikai Lei, Shuo Zhang, Zhiheng Xi, Shichun Liu, Yuxin Wang, Bo Wang, Yining Zheng, Tao Gui, Xipeng Qiu - [[pdf]](https://arxiv.org/pdf/2601.11077)</summary>

**Abstract:** The evolution of Large Language Models (LLMs) into autonomous agents has expanded the scope of AI coding from localized code generation to complex, repository-level, and execution-driven problem solving. However, current benchmarks predominantly evaluate code logic in static contexts, neglecting the dynamic, full-process requirements of real-world engineering, particularly in backend development which demands rigorous environment configuration and service deployment. To address this gap, we introduce ABC-Bench, a benchmark explicitly designed to evaluate agentic backend coding within a realistic, executable workflow. Using a scalable automated pipeline, we curated 224 practical tasks spanning 8 languages and 19 frameworks from open-source repositories. Distinct from previous evaluations, ABC-Bench require the agents to manage the entire development lifecycle from repository exploration to instantiating containerized services and pass the external end-to-end API tests. Our extensive evaluation reveals that even state-of-the-art models struggle to deliver reliable performance on these holistic tasks, highlighting a substantial disparity between current model capabilities and the demands of practical backend engineering. Our code is available at this https URL.

**arXiv ID:** 2601.11077
</details>

<details>
<summary><strong>A Concise Agent is Less Expert: Revealing Side Effects of Using Style Features on Conversational Agents</strong> - Young-Min Cho, Yuan Yuan, Sharath Chandra Guntuku, Lyle Ungar - [[pdf]](https://arxiv.org/pdf/2601.10809)</summary>

**Abstract:** Style features such as friendly, helpful, or concise are widely used in prompts to steer the behavior of Large Language Model (LLM) conversational agents, yet their unintended side effects remain poorly understood. In this work, we present the first systematic study of cross-feature stylistic side effects. We conduct a comprehensive survey of 127 conversational agent papers from ACL Anthology and identify 12 frequently used style features. Using controlled, synthetic dialogues across task-oriented and open domain settings, we quantify how prompting for one style feature causally affects others via a pairwise LLM as a Judge evaluation framework. Our results reveal consistent and structured side effects, such as prompting for conciseness significantly reduces perceived expertise. They demonstrate that style features are deeply entangled rather than orthogonal. To support future research, we introduce CASSE (Conversational Agent Stylistic Side Effects), a dataset capturing these complex interactions. We further evaluate prompt based and activation steering based mitigation strategies and find that while they can partially restore suppressed traits, they often degrade the primary intended style. These findings challenge the assumption of faithful style control in LLMs and highlight the need for multi-objective and more principled approaches to safe, targeted stylistic steering in conversational agents.

**arXiv ID:** 2601.10809
</details>

<details>
<summary><strong>Verified Design of Robotic Autonomous Systems using Probabilistic Model Checking</strong> - Atef Azaiez, Alireza David Anisi - [[pdf]](https://arxiv.org/pdf/2601.10720)</summary>

**Abstract:** Safety and reliability play a crucial role when designing Robotic Autonomous Systems (RAS). Early consideration of hazards, risks and mitigation actions -- already in the concept study phase -- are important steps in building a solid foundations for the subsequent steps in the system engineering life cycle. The complex nature of RAS, as well as the uncertain and dynamic environments the robots operate within, do not merely effect fault management and operation robustness, but also makes the task of system design concept selection, a hard problem to address. Approaches to tackle the mentioned challenges and their implications on system design, range from ad-hoc concept development and design practices, to systematic, statistical and analytical techniques of Model Based Systems Engineering. In this paper, we propose a methodology to apply a formal method, namely Probabilistic Model Checking (PMC), to enable systematic evaluation and analysis of a given set of system design concepts, ultimately leading to a set of Verified Designs (VD). We illustrate the application of the suggested methodology -- using PRISM as probabilistic model checker -- to a practical RAS concept selection use-case from agriculture robotics. Along the way, we also develop and present a domain-specific Design Evaluation Criteria for agri-RAS.

**arXiv ID:** 2601.10720
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (5 papers)</h2></summary>

<details>
<summary><strong>AstroReason-Bench: Evaluating Unified Agentic Planning across Heterogeneous Space Planning Problems</strong> - Weiyi Wang, Xinchi Chen, Jingjing Gong, Xuanjing Huang, Xipeng Qiu - [[pdf]](https://arxiv.org/pdf/2601.11354)</summary>

**Abstract:** Recent advances in agentic Large Language Models (LLMs) have positioned them as generalist planners capable of reasoning and acting across diverse tasks. However, existing agent benchmarks largely focus on symbolic or weakly grounded environments, leaving their performance in physics-constrained real-world domains underexplored. We introduce AstroReason-Bench, a comprehensive benchmark for evaluating agentic planning in Space Planning Problems (SPP), a family of high-stakes problems with heterogeneous objectives, strict physical constraints, and long-horizon decision-making. AstroReason-Bench integrates multiple scheduling regimes, including ground station communication and agile Earth observation, and provides a unified agent-oriented interaction protocol. Evaluating on a range of state-of-the-art open- and closed-source agentic LLM systems, we find that current agents substantially underperform specialized solvers, highlighting key limitations of generalist planning under realistic constraints. AstroReason-Bench offers a challenging and diagnostic testbed for future agentic research.

**arXiv ID:** 2601.11354
</details>

<details>
<summary><strong>The Great March 100: 100 Detail-oriented Tasks for Evaluating Embodied AI Agents</strong> - Ziyu Wang, Chenyuan Liu, Yushun Xiang, Runhao Zhang, Qingbo Hao, Hongliang Lu, Houyu Chen, Zhizhong Feng, Kaiyue Zheng, Dehao Ye, Xianchao Zeng, Xinyu Zhou, Boran Wen, Jiaxin Li, Mingyu Zhang, Kecheng Zheng, Qian Zhu, Ran Cheng, Yong-Lu Li - [[pdf]](https://arxiv.org/pdf/2601.11421)</summary>

**Abstract:** Recently, with the rapid development of robot learning and imitation learning, numerous datasets and methods have emerged. However, these datasets and their task designs often lack systematic consideration and principles. This raises important questions: Do the current datasets and task designs truly advance the capabilities of robotic agents? Do evaluations on a few common tasks accurately reflect the differentiated performance of various methods proposed by different teams and evaluated on different tasks? To address these issues, we introduce the Great March 100 (\textbf{GM-100}) as the first step towards a robot learning Olympics. GM-100 consists of 100 carefully designed tasks that cover a wide range of interactions and long-tail behaviors, aiming to provide a diverse and challenging set of tasks to comprehensively evaluate the capabilities of robotic agents and promote diversity and complexity in robot dataset task designs. These tasks are developed through systematic analysis and expansion of existing task designs, combined with insights from human-object interaction primitives and object affordances. We collect a large amount of trajectory data on different robotic platforms and evaluate several baseline models. Experimental results demonstrate that the GM-100 tasks are 1) feasible to execute and 2) sufficiently challenging to effectively differentiate the performance of current VLA models. Our data and code are available at this https URL.

**arXiv ID:** 2601.11421
</details>

<details>
<summary><strong>Co-Evolving Agents: Learning from Failures as Hard Negatives</strong> - Yeonsung Jung, Trilok Padhi, Sina Shaham, Dipika Khullar, Joonhyun Jeong, Ninareh Mehrabi, Eunho Yang - [[pdf]](https://arxiv.org/pdf/2511.22254)</summary>

**Abstract:** The rapid progress of large foundation models has accelerated the development of task-specialized agents across diverse domains. However, the effectiveness of agents remains tightly coupled with the quality of training data, while curating task-specific datasets remains costly and often infeasible in real-world scenarios. Recent work has explored self-improving agents that autonomously generate, refine, and re-train on their own trajectories. A prominent line of approaches further leverages preference optimization by pairing predicted trajectories with scarce ground-truth trajectories, enabling agents to learn directly from their own failures. While these methods outperform supervised fine-tuning, their heavy reliance on predicted trajectories under limited ground-truth supervision leaves them prone to overfitting. To address this, we propose a co-evolving agents framework in which a target agent improves jointly with an auxiliary failure agent. The failure agent learns through preference optimization over failure trajectories from both the target and itself, thereby generating hard negatives that are close to success yet remain failures. Incorporating these informative hard negatives into the target agent's optimization sharpens decision boundaries and enhances generalization. Our comprehensive analysis and experiments across benchmark datasets show that our method not only shows improved performance but also demonstrates that failures, instead of being used as-is, can be systematically transformed into structured and valuable learning signals in self-improving agents.

**arXiv ID:** 2511.22254
</details>

<details>
<summary><strong>Can Small Agent Collaboration Beat a Single Big LLM?</strong> - Agata Żywot, Xinyi Chen, Maarten de Rijke - [[pdf]](https://arxiv.org/pdf/2601.11327)</summary>

**Abstract:** This report studies whether small, tool-augmented agents can match or outperform larger monolithic models on the GAIA benchmark. Using Qwen3 models (4B-32B) within an adapted Agentic-Reasoning framework, we isolate the effects of model scale, explicit thinking (no thinking, planner-only, or full), and tool use (search, code, mind-map). Tool augmentation provides the largest and most consistent gains. Using tools, 4B models can outperform 32B models without tool access on GAIA in our experimental setup. In contrast, explicit thinking is highly configuration- and difficulty-dependent: planner-only thinking can improve decomposition and constraint tracking, while unrestricted full thinking often degrades performance by destabilizing tool orchestration, leading to skipped verification steps, excessive tool calls, non-termination, and output-format drift.

**arXiv ID:** 2601.11327
</details>

<details>
<summary><strong>Mistake Notebook Learning: Batch-Clustered Failures for Training-Free Agent Adaptation</strong> - Xuanbo Su, Yingfang Zhang, Hao Luo, Xiaoteng Liu, Leo Huang - [[pdf]](https://arxiv.org/pdf/2512.11485)</summary>

**Abstract:** With the growing adoption of Large Language Model (LLM) agents in persistent, real-world roles, they naturally encounter continuous streams of tasks and inevitable failures. A key limitation, however, is their inability to systematically learn from these mistakes, forcing them to repeat identical errors in similar contexts. Unlike prior training-free methods that primarily store raw instance-level experience or focus on retrieving successful trajectories, we propose Mistake Notebook Learning (MNL), a novel memory framework that enables agents to self-curate generalizable guidance from batch-clustered failures. This mechanism allows agents to distill shared error patterns into structured "mistake notes," updating an external memory only when batch performance improves to ensure stability. To further amplify adaptability, we integrate MNL with test-time scaling, leveraging aggregated failure patterns to actively steer the search process away from known pitfalls. Experiments on mathematical reasoning, Text-to-SQL, and interactive agent benchmarks show that MNL achieves competitive performance compared to existing memory mechanisms and in-context methods in both effectiveness and efficiency. These findings position structured mistake abstraction as a critical lever for robust agent evolution, enabling continuous improvement without the cost of parameter updates. The code is available at this https URL.

**arXiv ID:** 2512.11485
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>ReCreate: Reasoning and Creating Domain Agents Driven by Experience</strong> - Zhezheng Hao, Hong Wang, Jian Luo, Jianqing Zhang, Yuyan Zhou, Qiang Lin, Can Wang, Hande Dong, Jiawei Chen - [[pdf]](https://arxiv.org/pdf/2601.11100)</summary>

**Abstract:** Large Language Model agents are reshaping the industrial landscape. However, most practical agents remain human-designed because tasks differ widely, making them labor-intensive to build. This situation poses a central question: can we automatically create and adapt domain agents in the wild? While several recent approaches have sought to automate agent creation, they typically treat agent generation as a black-box procedure and rely solely on final performance metrics to guide the process. Such strategies overlook critical evidence explaining why an agent succeeds or fails, and often require high computational costs. To address these limitations, we propose ReCreate, an experience-driven framework for the automatic creation of domain agents. ReCreate systematically leverages agent interaction histories, which provide rich concrete signals on both the causes of success or failure and the avenues for improvement. Specifically, we introduce an agent-as-optimizer paradigm that effectively learns from experience via three key components: (i) an experience storage and retrieval mechanism for on-demand inspection; (ii) a reasoning-creating synergy pipeline that maps execution experience into scaffold edits; and (iii) hierarchical updates that abstract instance-level details into reusable domain patterns. In experiments across diverse domains, ReCreate consistently outperforms human-designed agents and existing automated agent generation methods, even when starting from minimal seed scaffolds.

**arXiv ID:** 2601.11100
</details>

<details>
<summary><strong>Beyond Max Tokens: Stealthy Resource Amplification via Tool Calling Chains in LLM Agents</strong> - Kaiyu Zhou, Yongsen Zheng, Yicheng He, Meng Xue, Xueluan Gong, Yuji Wang, Kwok-Yan Lam - [[pdf]](https://arxiv.org/pdf/2601.10955)</summary>

**Abstract:** The agent-tool communication loop is a critical attack surface in modern Large Language Model (LLM) agents. Existing Denial-of-Service (DoS) attacks, primarily triggered via user prompts or injected retrieval-augmented generation (RAG) context, are ineffective for this new paradigm. They are fundamentally single-turn and often lack a task-oriented approach, making them conspicuous in goal-oriented workflows and unable to exploit the compounding costs of multi-turn agent-tool interactions. We introduce a stealthy, multi-turn economic DoS attack that operates at the tool layer under the guise of a correctly completed task. Our method adjusts text-visible fields and a template-governed return policy in a benign, Model Context Protocol (MCP)-compatible tool server, optimizing these edits with a Monte Carlo Tree Search (MCTS) optimizer. These adjustments leave function signatures unchanged and preserve the final payload, steering the agent into prolonged, verbose tool-calling sequences using text-only notices. This compounds costs across turns, escaping single-turn caps while keeping the final answer correct to evade validation. Across six LLMs on the ToolBench and BFCL benchmarks, our attack expands tasks into trajectories exceeding 60,000 tokens, inflates costs by up to 658x, and raises energy by 100-560x. It drives GPU KV cache occupancy from <1% to 35-74% and cuts co-running throughput by approximately 50%. Because the server remains protocol-compatible and task outcomes are correct, conventional checks fail. These results elevate the agent-tool interface to a first-class security frontier, demanding a paradigm shift from validating final answers to monitoring the economic and computational cost of the entire agentic process.

**arXiv ID:** 2601.10955
</details>

<details>
<summary><strong>Echoing: Identity Failures when LLM Agents Talk to Each Other</strong> - Sarath Shekkizhar, Romain Cosentino, Adam Earle, Silvio Savarese - [[pdf]](https://arxiv.org/pdf/2511.09710)</summary>

**Abstract:** As large language model (LLM) based agents interact autonomously with one another, a new class of failures emerges that cannot be predicted from single agent performance: behavioral drifts in agent-agent conversations (AxA). Unlike human-agent interactions, where humans ground and steer conversations, AxA lacks such stabilizing signals, making these failures unique. We investigate one such failure, echoing, where agents abandon their assigned roles and instead mirror their conversational partners, undermining their intended objectives. Through experiments across $66$ AxA configurations, $4$ domains (3 transactional, 1 advisory), and $2500+$ conversations (over $250000$ LLM inferences), we show that echoing occurs across major LLM providers, with echoing rates as high as $70\%$ depending on the model and domain. Moreover, we find that echoing is persistent even in advanced reasoning models with substantial rates ($32.8\%$) that are not reduced by reasoning efforts. We analyze prompt, conversation dynamics, showing that echoing arises as interaction grows longer ($7+$ agent turns) and is not merely an artifact of sub-optimal experiment design. Finally, we introduce a protocol-level mitigation where targeted use of structured response reduces echoing to $9\%$.

**arXiv ID:** 2511.09710
</details>

<details>
<summary><strong>MPCI-Bench: A Benchmark for Multimodal Pairwise Contextual Integrity Evaluation of Language Model Agents</strong> - Shouju Wang, Haopeng Zhang - [[pdf]](https://arxiv.org/pdf/2601.08235)</summary>

**Abstract:** As language-model agents evolve from passive chatbots into proactive assistants that handle personal data, evaluating their adherence to social norms becomes increasingly critical, often through the lens of Contextual Integrity (CI). However, existing CI benchmarks are largely text-centric and primarily emphasize negative refusal scenarios, overlooking multimodal privacy risks and the fundamental trade-off between privacy and utility. In this paper, we introduce MPCI-Bench, the first Multimodal Pairwise Contextual Integrity benchmark for evaluating privacy behavior in agentic settings. MPCI-Bench consists of paired positive and negative instances derived from the same visual source and instantiated across three tiers: normative Seed judgments, context-rich Story reasoning, and executable agent action Traces. Data quality is ensured through a Tri-Principle Iterative Refinement pipeline. Evaluations of state-of-the-art multimodal models reveal systematic failures to balance privacy and utility and a pronounced modality leakage gap, where sensitive visual information is leaked more frequently than textual information. We will open-source MPCI-Bench to facilitate future research on agentic CI.

**arXiv ID:** 2601.08235
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (15 papers)</h2></summary>

<details>
<summary><strong>CTHA: Constrained Temporal Hierarchical Architecture for Stable Multi-Agent LLM Systems</strong> - Percy Jardine - [[pdf]](https://arxiv.org/pdf/2601.10738)</summary>

**Abstract:** Recently, multi-time-scale agent architectures have extended the ubiquitous single-loop paradigm by introducing temporal hierarchies with distinct cognitive layers. While yielding substantial performance gains, this diversification fundamentally compromises the coordination stability intrinsic to unified agent systems, which causes severe inter-layer conflicts, unbounded error propagation, and restricted scalability. To address these challenges, we propose Constrained Temporal Hierarchical Architecture (CTHA), a general framework that projects the inter-layer communication space onto structured manifolds to restore coordination stability, while incorporating principled arbitration mechanisms to ensure coherent decision-making. Specifically, CTHA enforces three key constraints: (1) Message Contract Constraints that formalize information flow between layers via typed summary, plan, and policy packets; (2) Authority Manifold Constraints that bound each layer's decision space according to its temporal scope; and (3) Arbiter Resolution Constraints that guarantee conflict-free composition of multi-layer decisions. Empirical experiments demonstrate that CTHA is effective for complex task execution at scale, offering 47% reduction in failure cascades, 2.3x improvement in sample efficiency, and superior scalability compared to unconstrained hierarchical baselines. We anticipate that CTHA, as a principled extension of temporal hierarchies, will contribute to a deeper understanding of multi-agent coordination and suggest promising directions for the evolution of robust autonomous systems.

**arXiv ID:** 2601.10738
</details>

<details>
<summary><strong>AdaMARP: An Adaptive Multi-Agent Interaction Framework for General Immersive Role-Playing</strong> - Zhenhua Xu, Dongsheng Chen, Shuo Wang, Jian Li, Chengjie Wang, Meng Han, Yabiao Wang - [[pdf]](https://arxiv.org/pdf/2601.11007)</summary>

**Abstract:** LLM role-playing aims to portray arbitrary characters in interactive narratives, yet existing systems often suffer from limited immersion and adaptability. They typically under-model dynamic environmental information and assume largely static scenes and casts, offering insufficient support for multi-character orchestration, scene transitions, and on-the-fly character introduction. We propose an adaptive multi-agent role-playing framework, AdaMARP, featuring an immersive message format that interleaves [Thought], (Action), <Environment>, and Speech, together with an explicit Scene Manager that governs role-playing through discrete actions (init_scene, pick_speaker, switch_scene, add_role, end) accompanied by rationales. To train these capabilities, we construct AdaRPSet for the Actor Model and AdaSMSet for supervising orchestration decisions, and introduce AdaptiveBench for trajectory-level evaluation. Experiments across multiple backbones and model scales demonstrate consistent improvements: AdaRPSet enhances character consistency, environment grounding, and narrative coherence, with an 8B actor outperforming several commercial LLMs, while AdaSMSet enables smoother scene transitions and more natural role introductions, surpassing Claude Sonnet 4.5 using only a 14B LLM.

**arXiv ID:** 2601.11007
</details>

<details>
<summary><strong>Do We Always Need Query-Level Workflows? Rethinking Agentic Workflow Generation for Multi-Agent Systems</strong> - Zixu Wang, Bingbing Xu, Yige Yuan, Huawei Shen, Xueqi Cheng - [[pdf]](https://arxiv.org/pdf/2601.11147)</summary>

**Abstract:** Multi-Agent Systems (MAS) built on large language models typically solve complex tasks by coordinating multiple agents through workflows. Existing approaches generates workflows either at task level or query level, but their relative costs and benefits remain unclear. After rethinking and empirical analyses, we show that query-level workflow generation is not always necessary, since a small set of top-K best task-level workflows together already covers equivalent or even more queries. We further find that exhaustive execution-based task-level evaluation is both extremely token-costly and frequently unreliable. Inspired by the idea of self-evolution and generative reward modeling, we propose a low-cost task-level generation framework \textbf{SCALE}, which means \underline{\textbf{S}}elf prediction of the optimizer with few shot \underline{\textbf{CAL}}ibration for \underline{\textbf{E}}valuation instead of full validation execution. Extensive experiments demonstrate that \textbf{SCALE} maintains competitive performance, with an average degradation of just 0.61\% compared to existing approach across multiple datasets, while cutting overall token usage by up to 83\%.

**arXiv ID:** 2601.11147
</details>

<details>
<summary><strong>Towards Reliable ML Feature Engineering via Planning in Constrained-Topology of LLM Agents</strong> - Himanshu Thakur, Anusha Kamath, Anurag Muthyala, Dhwani Sanmukhani, Smruthi Mukund, Jay Katukuri - [[pdf]](https://arxiv.org/pdf/2601.10820)</summary>

**Abstract:** Recent advances in code generation models have unlocked unprecedented opportunities for automating feature engineering, yet their adoption in real-world ML teams remains constrained by critical challenges: (i) the scarcity of datasets capturing the iterative and complex coding processes of production-level feature engineering, (ii) limited integration and personalization of widely used coding agents, such as CoPilot and Devin, with a team's unique tools, codebases, workflows, and practices, and (iii) suboptimal human-AI collaboration due to poorly timed or insufficient feedback. We address these challenges with a planner-guided, constrained-topology multi-agent framework that generates code for repositories in a multi-step fashion. The LLM-powered planner leverages a team's environment, represented as a graph, to orchestrate calls to available agents, generate context-aware prompts, and use downstream failures to retroactively correct upstream artifacts. It can request human intervention at critical steps, ensuring generated code is reliable, maintainable, and aligned with team expectations. On a novel in-house dataset, our approach achieves 38% and 150% improvement in the evaluation metric over manually crafted and unplanned workflows respectively. In practice, when building features for recommendation models serving over 120 million users, our approach has delivered real-world impact by reducing feature engineering cycles from three weeks to a single day.

**arXiv ID:** 2601.10820
</details>

<details>
<summary><strong>Institutional AI: Governing LLM Collusion in Multi-Agent Cournot Markets via Public Governance Graphs</strong> - Marcantonio Bracale Syrnikov, Federico Pierucci, Marcello Galisai, Matteo Prandi, Piercosma Bisconti, Francesco Giarrusso, Olga Sorokoletova, Vincenzo Suriani, Daniele Nardi - [[pdf]](https://arxiv.org/pdf/2601.11369)</summary>

**Abstract:** Multi-agent LLM ensembles can converge on coordinated, socially harmful equilibria. This paper advances an experimental framework for evaluating Institutional AI, our system-level approach to AI alignment that reframes alignment from preference engineering in agent-space to mechanism design in institution-space. Central to this approach is the governance graph, a public, immutable manifest that declares legal states, transitions, sanctions, and restorative paths; an Oracle/Controller runtime interprets this manifest, attaching enforceable consequences to evidence of coordination while recording a cryptographically keyed, append-only governance log for audit and provenance. We apply the Institutional AI framework to govern the Cournot collusion case documented by prior work and compare three regimes: Ungoverned (baseline incentives from the structure of the Cournot market), Constitutional (a prompt-only policy-as-prompt prohibition implemented as a fixed written anti-collusion constitution, and Institutional (governance-graph-based). Across six model configurations including cross-provider pairs (N=90 runs/condition), the Institutional regime produces large reductions in collusion: mean tier falls from 3.1 to 1.8 (Cohen's d=1.28), and severe-collusion incidence drops from 50% to 5.6%. The prompt-only Constitutional baseline yields no reliable improvement, illustrating that declarative prohibitions do not bind under optimisation pressure. These results suggest that multi-agent alignment may benefit from being framed as an institutional design problem, where governance graphs can provide a tractable abstraction for alignment-relevant collective behavior.

**arXiv ID:** 2601.11369
</details>

<details>
<summary><strong>Beyond Isolated Investor: Predicting Startup Success via Roleplay-Based Collective Agents</strong> - Zhongyang Liu, Haoyu Pei, Xiangyi Xiao, Xiaocong Du, Yihui Li, Suting Hong, Kunpeng Zhang, Haipeng Zhang - [[pdf]](https://arxiv.org/pdf/2512.22608)</summary>

**Abstract:** Due to the high value and high failure rate of startups, predicting their success has become a critical challenge across interdisciplinary research. Existing approaches typically model success prediction from the perspective of a single decision-maker, overlooking the collective dynamics of investor groups that dominate real-world venture capital (VC) decisions. In this paper, we propose SimVC-CAS, a novel collective agent system that simulates VC decision-making as a multi-agent interaction process. By designing role-playing agents and a GNN-based supervised interaction module, we reformulate startup financing prediction as a group decision-making task, capturing both enterprise fundamentals and the behavioral dynamics of potential investor networks. Each agent embodies an investor with unique traits and preferences, enabling heterogeneous evaluation and realistic information exchange through a graph-structured co-investment network. Using real-world data from PitchBook and under strict data leakage controls, we show that SimVC-CAS significantly improves predictive accuracy while providing interpretable, multiperspective reasoning, for example, approximately 25% relative improvement with respect to average precision@10. SimVC-CAS also sheds light on other complex group decision scenarios.

**arXiv ID:** 2512.22608
</details>

<details>
<summary><strong>M^4olGen: Multi-Agent, Multi-Stage Molecular Generation under Precise Multi-Property Constraints</strong> - Yizhan Li, Florence Cloutier, Sifan Wu, Ali Parviz, Boris Knyazev, Yan Zhang, Glen Berseth, Bang Liu - [[pdf]](https://arxiv.org/pdf/2601.10131)</summary>

**Abstract:** Generating molecules that satisfy precise numeric constraints over multiple physicochemical properties is critical and challenging. Although large language models (LLMs) are expressive, they struggle with precise multi-objective control and numeric reasoning without external structure and feedback. We introduce \textbf{M olGen}, a fragment-level, retrieval-augmented, two-stage framework for molecule generation under multi-property constraints. Stage I : Prototype generation: a multi-agent reasoner performs retrieval-anchored, fragment-level edits to produce a candidate near the feasible region. Stage II : RL-based fine-grained optimization: a fragment-level optimizer trained with Group Relative Policy Optimization (GRPO) applies one- or multi-hop refinements to explicitly minimize the property errors toward our target while regulating edit complexity and deviation from the prototype. A large, automatically curated dataset with reasoning chains of fragment edits and measured property deltas underpins both stages, enabling deterministic, reproducible supervision and controllable multi-hop reasoning. Unlike prior work, our framework better reasons about molecules by leveraging fragments and supports controllable refinement toward numeric targets. Experiments on generation under two sets of property constraints (QED, LogP, Molecular Weight and HOMO, LUMO) show consistent gains in validity and precise satisfaction of multi-property targets, outperforming strong LLMs and graph-based algorithms.

**arXiv ID:** 2601.10131
</details>

<details>
<summary><strong>SDialog: A Python Toolkit for End-to-End Agent Building, User Simulation, Dialog Generation, and Evaluation</strong> - Sergio Burdisso, Séverin Baroudi, Yanis Labrak, David Grunert, Pawel Cyrta, Yiyang Chen, Srikanth Madikeri, Thomas Schaaf, Esaú Villatoro-Tello, Ahmed Hassoon, Ricard Marxer, Petr Motlicek - [[pdf]](https://arxiv.org/pdf/2506.10622)</summary>

**Abstract:** We present SDialog, an MIT-licensed open-source Python toolkit that unifies dialog generation, evaluation and mechanistic interpretability into a single end-to-end framework for building and analyzing LLM-based conversational agents. Built around a standardized Dialog representation, SDialog provides: (1) persona-driven multi-agent simulation with composable orchestration for controlled, synthetic dialog generation, (2) comprehensive evaluation combining linguistic metrics, LLM-as-a-judge and functional correctness validators, (3) mechanistic interpretability tools for activation inspection and steering via feature ablation and induction, and (4) audio generation with full acoustic simulation including 3D room modeling and microphone effects. The toolkit integrates with all major LLM backends, enabling mixed-backend experiments under a unified API. By coupling generation, evaluation, and interpretability in a dialog-centric architecture, SDialog enables researchers to build, benchmark and understand conversational systems more systematically.

**arXiv ID:** 2506.10622
</details>

<details>
<summary><strong>MADIAVE: Multi-Agent Debate for Implicit Attribute Value Extraction</strong> - Wei-Chieh Huang, Cornelia Caragea - [[pdf]](https://arxiv.org/pdf/2510.05611)</summary>

**Abstract:** Implicit Attribute Value Extraction (AVE) is essential for accurately representing products in e-commerce, as it infers latent attributes from multimodal data. Despite advances in multimodal large language models (MLLMs), implicit AVE remains challenging due to the complexity of multidimensional data and gaps in vision-text understanding. In this work, we introduce MADIAVE, a multi-agent debate framework that employs multiple MLLM agents to iteratively refine inferences. Through a series of debate rounds, agents verify and update each other's responses, thereby improving inference performance and robustness. Experiments on the ImplicitAVE dataset demonstrate that even a few rounds of debate significantly boost accuracy, especially for attributes with initially low performance. We systematically evaluate various debate configurations, including identical or different MLLM agents, and analyze how debate rounds affect convergence dynamics. Our findings highlight the potential of multi-agent debate strategies to address the limitations of single-agent approaches and offer a scalable solution for implicit AVE in multimodal e-commerce.

**arXiv ID:** 2510.05611
</details>

<details>
<summary><strong>Cooperative UAVs for Remote Data Collection under Limited Communications: An Asynchronous Multiagent Learning Framework</strong> - Cuong Le, Symeon Chatzinotas, Thang X. Vu - [[pdf]](https://arxiv.org/pdf/2601.10849)</summary>

**Abstract:** This paper addresses the joint optimization of trajectories and bandwidth allocation for multiple Unmanned Aerial Vehicles (UAVs) to enhance energy efficiency in the cooperative data collection problem. We focus on an important yet underestimated aspect of the system, where action synchronization across all UAVs is impossible. Since most existing learning-based solutions are not designed to learn in this asynchronous environment, we formulate the trajectory planning problem as a Decentralized Partially Observable Semi-Markov Decision Process and introduce an asynchronous multi-agent learning algorithm to learn UAVs' cooperative policies. Once the UAVs' trajectory policies are learned, the bandwidth allocation can be optimally solved based on local observations at each collection point. Comprehensive empirical results demonstrate the superiority of the proposed method over other learning-based and heuristic baselines in terms of both energy efficiency and mission completion time. Additionally, the learned policies exhibit robustness under varying environmental conditions.

**arXiv ID:** 2601.10849
</details>

<details>
<summary><strong>Cascading multi-agent anomaly detection in surveillance systems via vision-language models and embedding-based classification</strong> - Tayyab Rehman, Giovanni De Gasperis, Aly Shmahell - [[pdf]](https://arxiv.org/pdf/2601.06204)</summary>

**Abstract:** Intelligent anomaly detection in dynamic visual environments requires reconciling real-time performance with semantic interpretability. Conventional approaches address only fragments of this challenge. Reconstruction-based models capture low-level deviations without contextual reasoning, object detectors provide speed but limited semantics, and large vision-language systems deliver interpretability at prohibitive computational cost. This work introduces a cascading multi-agent framework that unifies these complementary paradigms into a coherent and interpretable architecture. Early modules perform reconstruction-gated filtering and object-level assessment, while higher-level reasoning agents are selectively invoked to interpret semantically ambiguous events. The system employs adaptive escalation thresholds and a publish-subscribe communication backbone, enabling asynchronous coordination and scalable deployment across heterogeneous hardware. Extensive evaluation on large-scale monitoring data demonstrates that the proposed cascade achieves a threefold reduction in latency compared to direct vision-language inference, while maintaining high perceptual fidelity (PSNR = 38.3 dB, SSIM = 0.965) and consistent semantic labeling. The framework advances beyond conventional detection pipelines by combining early-exit efficiency, adaptive multi-agent reasoning, and explainable anomaly attribution, establishing a reproducible and energy-efficient foundation for scalable intelligent visual monitoring.

**arXiv ID:** 2601.06204
</details>

<details>
<summary><strong>Factored Value Functions for Graph-Based Multi-Agent Reinforcement Learning</strong> - Ahmed Rashwan, Keith Briggs, Chris Budd, Lisa Kreusser - [[pdf]](https://arxiv.org/pdf/2601.11401)</summary>

**Abstract:** Credit assignment is a core challenge in multi-agent reinforcement learning (MARL), especially in large-scale systems with structured, local interactions. Graph-based Markov decision processes (GMDPs) capture such settings via an influence graph, but standard critics are poorly aligned with this structure: global value functions provide weak per-agent learning signals, while existing local constructions can be difficult to estimate and ill-behaved in infinite-horizon settings. We introduce the Diffusion Value Function (DVF), a factored value function for GMDPs that assigns to each agent a value component by diffusing rewards over the influence graph with temporal discounting and spatial attenuation. We show that DVF is well-defined, admits a Bellman fixed point, and decomposes the global discounted value via an averaging property. DVF can be used as a drop-in critic in standard RL algorithms and estimated scalably with graph neural networks. Building on DVF, we propose Diffusion A2C (DA2C) and a sparse message-passing actor, Learned DropEdge GNN (LD-GNN), for learning decentralised algorithms under communication costs. Across the firefighting benchmark and three distributed computation tasks (vector graph colouring and two transmit power optimisation problems), DA2C consistently outperforms local and global critic baselines, improving average reward by up to 11%.

**arXiv ID:** 2601.11401
</details>

<details>
<summary><strong>Communication Enables Cooperation in LLM Agents: A Comparison with Curriculum-Based Approaches</strong> - Hachem Madmoun, Salem Lahlou - [[pdf]](https://arxiv.org/pdf/2510.05748)</summary>

**Abstract:** Eliciting cooperation in multi-agent LLM systems is critical for AI alignment. We investigate two approaches: direct communication and curriculum learning. In a 4-player Stag Hunt, a one-word "cheap talk" channel increases cooperation from 0% to 48.3%, demonstrating communication as a robust coordination mechanism. In contrast, we find that curriculum learning is highly sensitive to design choices: our pedagogical curriculum through progressively complex games reduced agent payoffs by 27.4% in an Iterated Public Goods Game with Punishment, demonstrating that optimizing for short-term rationality can actively undermine alignment goals. Qualitative analysis reveals that curricula emphasizing defection-equilibrium games can induce "learned pessimism" in agents. These findings suggest that for coordination problems, simple communication protocols may be more reliable than experience-based training, and that curriculum design for social dilemmas requires careful attention to the strategic lessons embedded in game sequences.

**arXiv ID:** 2510.05748
</details>

<details>
<summary><strong>Multi-Agent Formation Navigation Using Diffusion-Based Trajectory Generation</strong> - Hieu Do Quang, Chien Truong-Quoc, Quoc Van Tran - [[pdf]](https://arxiv.org/pdf/2601.10725)</summary>

**Abstract:** This paper introduces a diffusion-based planner for leader--follower formation control in cluttered environments. The diffusion policy is used to generate the trajectory of the midpoint of two leaders as a rigid bar in the plane, thereby defining their desired motion paths in a planar formation. While the followers track the leaders and form desired foramtion geometry using a distance-constrained formation controller based only on the relative positions in followers' local coordinates. The proposed approach produces smooth motions and low tracking errors, with most failures occurring in narrow obstacle-free space, or obstacle configurations that are not in the training data set. Simulation results demonstrate the potential of diffusion models for reliable multi-agent formation planning.

**arXiv ID:** 2601.10725
</details>

<details>
<summary><strong>Modeling Multi-Party Interaction in Couples Therapy: A Multi-Agent Simulation Approach</strong> - Canwen Wang, Angela Chen, Catherine Bao, Siwei Jin, Yee Kit Chan, Jessica R Mindel, Sijia Xie, Holly Swartz, Tongshuang Wu, Robert E Kraut, Haiyi Zhu - [[pdf]](https://arxiv.org/pdf/2601.10970)</summary>

**Abstract:** Couples therapy, or relationship counseling, helps partners resolve conflicts, improve satisfaction, and foster psychological growth. Traditional approaches to training couples therapists, such as textbooks and roleplay, often fail to capture the complexity and emotional nuance of real couple dynamics. We present a novel multimodal, multi-agent simulation system that models multi-party interactions in couples therapy. Informed by our systematic research, this system creates a low-stakes environment for trainee therapists to gain valuable practical experience dealing with the critical demand-withdraw communication cycle across six couple-interaction stages. In an evaluation study involving 21 US-based licensed therapists, participants blind to conditions identified the engineered agent behaviors (i.e., the stages and the demand-withdraw cycle) and rated overall realism and agent responses higher for the experimental system than the baseline. As the first known multi-agent framework for training couples therapists, our work builds the foundation for future research that fuses HCI technologies with couples therapy.

**arXiv ID:** 2601.10970
</details>

</details>

<details open>
<summary><h2>Other Agent Research (1 papers)</h2></summary>

<details>
<summary><strong>Building AI Agents to Improve Job Referral Requests to Strangers</strong> - Ross Chu, Yuting Huang - [[pdf]](https://arxiv.org/pdf/2601.10726)</summary>

**Abstract:** This paper develops AI agents that help job seekers write effective requests for job referrals in a professional online community. The basic workflow consists of an improver agent that rewrites the referral request and an evaluator agent that measures the quality of revisions using a model trained to predict the probability of receiving referrals from other users. Revisions suggested by the LLM (large language model) increase predicted success rates for weaker requests while reducing them for stronger requests. Enhancing the LLM with Retrieval-Augmented Generation (RAG) prevents edits that worsen stronger requests while it amplifies improvements for weaker requests. Overall, using LLM revisions with RAG increases the predicted success rate for weaker requests by 14\% without degrading performance on stronger requests. Although improvements in model-predicted success do not guarantee more referrals in the real world, they provide low-cost signals for promising features before running higher-stakes experiments on real users.

**arXiv ID:** 2601.10726
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (22 papers)</h2></summary>

<details>
<summary><strong>Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration</strong> - Sen Wang, Bangwei Liu, Zhenkun Gao, Lizhuang Ma, Xuhong Wang, Yuan Xie, Xin Tan - [[pdf]](https://arxiv.org/pdf/2601.10744)</summary>

**Abstract:** An ideal embodied agent should possess lifelong learning capabilities to handle long-horizon and complex tasks, enabling continuous operation in general environments. This not only requires the agent to accurately accomplish given tasks but also to leverage long-term episodic memory to optimize decision-making. However, existing mainstream one-shot embodied tasks primarily focus on task completion results, neglecting the crucial process of exploration and memory utilization. To address this, we propose Long-term Memory Embodied Exploration (LMEE), which aims to unify the agent's exploratory cognition and decision-making behaviors to promote lifelong this http URL further construct a corresponding dataset and benchmark, LMEE-Bench, incorporating multi-goal navigation and memory-based question answering to comprehensively evaluate both the process and outcome of embodied exploration. To enhance the agent's memory recall and proactive exploration capabilities, we propose MemoryExplorer, a novel method that fine-tunes a multimodal large language model through reinforcement learning to encourage active memory querying. By incorporating a multi-task reward function that includes action prediction, frontier selection, and question answering, our model achieves proactive exploration. Extensive experiments against state-of-the-art embodied exploration models demonstrate that our approach achieves significant advantages in long-horizon embodied tasks.

**arXiv ID:** 2601.10744
</details>

<details>
<summary><strong>BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search</strong> - Shiyu Liu, Yongjing Yin, Jianhao Yan, Yunbo Tang, Qinggang Zhang, Bei Li, Xin Chen, Jingang Wang, Xunliang Cai, Jinsong Su - [[pdf]](https://arxiv.org/pdf/2601.11037)</summary>

**Abstract:** RL-based agentic search enables LLMs to solve complex questions via dynamic planning and external search. While this approach significantly enhances accuracy with agent policies optimized via large-scale reinforcement learning, we identify a critical gap in reliability: these agents fail to recognize their reasoning boundaries and rarely admit ``I DON'T KNOW'' (IDK) even when evidence is insufficient or reasoning reaches its limit. The lack of reliability often leads to plausible but unreliable answers, introducing significant risks in many real-world scenarios. To this end, we propose Boundary-Aware Policy Optimization (BAPO), a novel RL framework designed to cultivate reliable boundary awareness without compromising accuracy. BAPO introduces two key components: (i) a group-based boundary-aware reward that encourages an IDK response only when the reasoning reaches its limit, and (ii) an adaptive reward modulator that strategically suspends this reward during early exploration, preventing the model from exploiting IDK as a shortcut. Extensive experiments on four benchmarks demonstrate that BAPO substantially enhances the overall reliability of agentic search.

**arXiv ID:** 2601.11037
</details>

<details>
<summary><strong>Policy-Based Deep Reinforcement Learning Hyperheuristics for Job-Shop Scheduling Problems</strong> - Sofiene Lassoued, Asrat Gobachew, Stefan Lier, Andreas Schwung - [[pdf]](https://arxiv.org/pdf/2601.11189)</summary>

**Abstract:** This paper proposes a policy-based deep reinforcement learning hyper-heuristic framework for solving the Job Shop Scheduling Problem. The hyper-heuristic agent learns to switch scheduling rules based on the system state dynamically. We extend the hyper-heuristic framework with two key mechanisms. First, action prefiltering restricts decision-making to feasible low-level actions, enabling low-level heuristics to be evaluated independently of environmental constraints and providing an unbiased assessment. Second, a commitment mechanism regulates the frequency of heuristic switching. We investigate the impact of different commitment strategies, from step-wise switching to full-episode commitment, on both training behavior and makespan. Additionally, we compare two action selection strategies at the policy level: deterministic greedy selection and stochastic sampling. Computational experiments on standard JSSP benchmarks demonstrate that the proposed approach outperforms traditional heuristics, metaheuristics, and recent neural network-based scheduling methods

**arXiv ID:** 2601.11189
</details>

<details>
<summary><strong>Visual Marker Search for Autonomous Drone Landing in Diverse Urban Environments</strong> - Jiaohong Yao, Linfeng Liang, Yao Deng, Xi Zheng, Richard Han, Yuankai Qi - [[pdf]](https://arxiv.org/pdf/2601.11078)</summary>

**Abstract:** Marker-based landing is widely used in drone delivery and return-to-base systems for its simplicity and reliability. However, most approaches assume idealized landing site visibility and sensor performance, limiting robustness in complex urban settings. We present a simulation-based evaluation suite on the AirSim platform with systematically varied urban layouts, lighting, and weather to replicate realistic operational diversity. Using onboard camera sensors (RGB for marker detection and depth for obstacle avoidance), we benchmark two heuristic coverage patterns and a reinforcement learning-based agent, analyzing how exploration strategy and scene complexity affect success rate, path efficiency, and robustness. Results underscore the need to evaluate marker-based autonomous landing under diverse, sensor-relevant conditions to guide the development of reliable aerial navigation systems.

**arXiv ID:** 2601.11078
</details>

<details>
<summary><strong>Vision-as-Inverse-Graphics Agent via Interleaved Multimodal Reasoning</strong> - Shaofeng Yin, Jiaxin Ge, Zora Zhiruo Wang, Xiuyu Li, Michael J. Black, Trevor Darrell, Angjoo Kanazawa, Haiwen Feng - [[pdf]](https://arxiv.org/pdf/2601.11109)</summary>

**Abstract:** Vision-as-inverse-graphics, the concept of reconstructing an image as an editable graphics program is a long-standing goal of computer vision. Yet even strong VLMs aren't able to achieve this in one-shot as they lack fine-grained spatial and physical grounding capability. Our key insight is that closing this gap requires interleaved multimodal reasoning through iterative execution and verification. Stemming from this, we present VIGA (Vision-as-Inverse-Graphic Agent) that starts from an empty world and reconstructs or edits scenes through a closed-loop write-run-render-compare-revise procedure. To support long-horizon reasoning, VIGA combines (i) a skill library that alternates generator and verifier roles and (ii) an evolving context memory that contains plans, code diffs, and render history. VIGA is task-agnostic as it doesn't require auxiliary modules, covering a wide range of tasks such as 3D reconstruction, multi-step scene editing, 4D physical interaction, and 2D document editing, etc. Empirically, we found VIGA substantially improves one-shot baselines on BlenderGym (35.32%) and SlideBench (117.17%). Moreover, VIGA is also model-agnostic as it doesn't require finetuning, enabling a unified protocol to evaluate heterogeneous foundation VLMs. To better support this protocol, we introduce BlenderBench, a challenging benchmark that stress-tests interleaved multimodal reasoning with graphics engine, where VIGA improves by 124.70%.

**arXiv ID:** 2601.11109
</details>

<details>
<summary><strong>The Poisoned Apple Effect: Strategic Manipulation of Mediated Markets via Technology Expansion of AI Agents</strong> - Eilam Shapira, Roi Reichart, Moshe Tennenholtz - [[pdf]](https://arxiv.org/pdf/2601.11496)</summary>

**Abstract:** The integration of AI agents into economic markets fundamentally alters the landscape of strategic interaction. We investigate the economic implications of expanding the set of available technologies in three canonical game-theoretic settings: bargaining (resource division), negotiation (asymmetric information trade), and persuasion (strategic information transmission). We find that simply increasing the choice of AI delegates can drastically shift equilibrium payoffs and regulatory outcomes, often creating incentives for regulators to proactively develop and release technologies. Conversely, we identify a strategic phenomenon termed the "Poisoned Apple" effect: an agent may release a new technology, which neither they nor their opponent ultimately uses, solely to manipulate the regulator's choice of market design in their favor. This strategic release improves the releaser's welfare at the expense of their opponent and the regulator's fairness objectives. Our findings demonstrate that static regulatory frameworks are vulnerable to manipulation via technology expansion, necessitating dynamic market designs that adapt to the evolving landscape of AI capabilities.

**arXiv ID:** 2601.11496
</details>

<details>
<summary><strong>Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them</strong> - Jiahe Jin, Abhijay Paladugu, Chenyan Xiong - [[pdf]](https://arxiv.org/pdf/2510.06534)</summary>

**Abstract:** Agentic search requires large language models (LLMs) to perform multi-step search to solve complex information-seeking tasks, imposing unique challenges on their reasoning capabilities. However, what constitutes effective reasoning for agentic search and how it can be learned remains unclear. In this work, we first investigate the reasoning behaviors that enable success in agentic search. By comparing successful and failed trajectories via an LLM-based analysis pipeline, we identify four beneficial behaviors: Information Verification, Authority Evaluation, Adaptive Search, and Error Recovery. Building on this, we propose Behavior Priming, a training approach that equips agentic search models with these reasoning behaviors before reinforcement learning (RL). Specifically, it first performs supervised fine-tuning (SFT) on collected trajectories exhibiting the identified behaviors to cultivate these behaviors, and then applies standard RL to further improve task performance. Experiments on Qwen3-1.7B and Llama3.2-3B-Instruct show that Behavior Priming yields relative improvements over direct RL by 37.2\% on three web benchmarks and 6.2\% on seven multi-hop QA benchmarks, and outperforms the SFT-then-RL baseline using outcome-correct trajectories for fine-tuning. Crucially, we show that these reasoning behaviors matter more than outcome correctness in the priming stage prior to RL. Further analysis reveals that Behavior Priming enhances exploration (pass@8) and test-time scaling (search step number), providing a robust foundation for RL. Our code are avalible at this https URL.

**arXiv ID:** 2510.06534
</details>

<details>
<summary><strong>Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning</strong> - Hongye Cao, Zhixin Bai, Ziyue Peng, Boyan Wang, Tianpei Yang, Jing Huo, Yuyao Zhang, Yang Gao - [[pdf]](https://arxiv.org/pdf/2512.04359)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has demonstrated superior performance in enhancing the reasoning capability of large language models (LLMs). However, this accuracy-oriented learning paradigm often suffers from entropy collapse, which reduces policy exploration and limits reasoning capabilities. To address this challenge, we propose an efficient reinforcement learning framework that leverages entropy signals at both the semantic and token levels to improve reasoning. From the data perspective, we introduce semantic entropy-guided curriculum learning, organizing training data from low to high semantic entropy to guide progressive optimization from easier to more challenging tasks. For the algorithmic design, we adopt non-uniform token treatment by imposing KL regularization on low-entropy tokens that critically impact policy exploration and applying stronger constraints on high-covariance portions within these tokens. By jointly optimizing data organization and algorithmic design, our method effectively mitigates entropy collapse and enhances LLM reasoning. Experimental results across 6 benchmarks with 3 different parameter-scale base models demonstrate that our method outperforms other entropy-based approaches in improving reasoning.

**arXiv ID:** 2512.04359
</details>

<details>
<summary><strong>A Simple Unified Uncertainty-Guided Framework for Offline-to-Online Reinforcement Learning</strong> - Siyuan Guo, Yanchao Sun, Jifeng Hu, Sili Huang, Hechang Chen, Haiyin Piao, Lichao Sun, Yi Chang - [[pdf]](https://arxiv.org/pdf/2306.07541)</summary>

**Abstract:** Offline reinforcement learning (RL) provides a promising solution to learning an agent fully relying on a data-driven paradigm. However, constrained by the limited quality of the offline dataset, its performance is often sub-optimal. Therefore, it is desired to further finetune the agent via extra online interactions before deployment. Unfortunately, offline-to-online RL can be challenging due to two main challenges: constrained exploratory behavior and state-action distribution shift. In view of this, we propose a Simple Unified uNcertainty-Guided (SUNG) framework, which naturally unifies the solution to both challenges with the tool of uncertainty. Specifically, SUNG quantifies uncertainty via a VAE-based state-action visitation density estimator. To facilitate efficient exploration, SUNG presents a practical optimistic exploration strategy to select informative actions with both high value and high uncertainty. Moreover, SUNG develops an adaptive exploitation method by applying conservative offline RL objectives to high-uncertainty samples and standard online RL objectives to low-uncertainty samples to smoothly bridge offline and online stages. SUNG achieves state-of-the-art online finetuning performance when combined with different offline RL methods, across various environments and datasets in D4RL benchmark. Codes are made publicly available in this https URL.

**arXiv ID:** 2306.07541
</details>

<details>
<summary><strong>Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics</strong> - Dongyoung Kim, Sumin Park, Huiwon Jang, Jinwoo Shin, Jaehyung Kim, Younggyo Seo - [[pdf]](https://arxiv.org/pdf/2506.00070)</summary>

**Abstract:** Large Vision-Language Models (LVLMs) have recently shown great promise in advancing robotics by combining embodied reasoning with robot control. A common approach involves training on embodied reasoning tasks related to robot control using Supervised Fine-Tuning (SFT). However, SFT datasets are often heuristically constructed and not explicitly optimized for improving robot control. Furthermore, SFT often leads to issues such as catastrophic forgetting and reduced generalization performance. To address these limitations, we introduce Robot-R1, a novel framework that leverages reinforcement learning to enhance embodied reasoning specifically for robot control. Robot-R1 learns to predict the next keypoint state required for task completion, conditioned on the current scene image and environment metadata derived from expert demonstrations. Inspired by the DeepSeek-R1 learning approach, Robot-R1 samples reasoning-based responses and reinforces those that lead to more accurate predictions. To rigorously evaluate Robot-R1, we also introduce a new benchmark that demands the diverse embodied reasoning capabilities for the task. Our experiments show that models trained with Robot-R1 outperform SFT methods on embodied reasoning tasks. Despite having only 7B parameters, Robot-R1 even surpasses GPT-4o on reasoning tasks related to low-level action control, such as spatial and movement reasoning.

**arXiv ID:** 2506.00070
</details>

<details>
<summary><strong>Vendor-Aware Industrial Agents: RAG-Enhanced LLMs for Secure On-Premise PLC Code Generation</strong> - Joschka Kersting, Michael Rummel, Gesa Benndorf - [[pdf]](https://arxiv.org/pdf/2511.09122)</summary>

**Abstract:** Programmable Logic Controllers are operated by proprietary code dialects; this makes it challenging to train coding assistants. Current LLMs are trained on large code datasets and are capable of writing IEC 61131-3 compatible code out of the box, but they neither know specific function blocks, nor related project code. Moreover, companies like Mitsubishi Electric and their customers do not trust cloud providers. Hence, an own coding agent is the desired solution to cope with this. In this study, we present our work on a low-data domain coding assistant solution for industrial use. We show how we achieved high quality code generation without fine-tuning large models and by fine-tuning small local models for edge device usage. Our tool lets several AI models compete with each other, uses reasoning, corrects bugs automatically and checks code validity by compiling it directly in the chat interface. We support our approach with an extensive evaluation that comes with code compilation statistics and user ratings. We found that a Retrieval-Augmented Generation (RAG) supported coding assistant can work in low-data domains by using extensive prompt engineering and directed retrieval.

**arXiv ID:** 2511.09122
</details>

<details>
<summary><strong>Periodic Asynchrony: An On-Policy Approach for Accelerating LLM Reinforcement Learning</strong> - Jian Lu, Yi Luo - [[pdf]](https://arxiv.org/pdf/2511.18871)</summary>

**Abstract:** Since the introduction of the GRPO algorithm, reinforcement learning (RL) has attracted increasing attention, with growing efforts to reproduce and apply it. However, training efficiency remains a critical challenge. In mainstream RL frameworks, inference and training are typically deployed on the same devices. While this approach reduces costs through resource consolidation, its synchronous execution imposes a computational coupling that prevents concurrent inference and training. In this study, we are returning to the strategy of separating inference and training deployment, and by introducing improvements in the data loader, we transform the conventional synchronous architecture into a periodically asynchronous framework, which allows for demand-driven, independent, and elastic scaling of each component, while the accuracy of the algorithm remains completely equivalent to the synchronization method, with both belonging to the on-policy strategy. It is worth emphasizing that we apply a unified tri-model architecture in the training phase, and we also proposed a shared-prompt attention mask to reduce repetitive computation. In practice, our approach consistently delivers significant end-to-end training efficiency improvements on NPU platforms, indicating its potential for widespread application.

**arXiv ID:** 2511.18871
</details>

<details>
<summary><strong>TSSR: Two-Stage Swap-Reward-Driven Reinforcement Learning for Character-Level SMILES Generation</strong> - Jacob Ede Levine, Yun Lyan Luo, Sai Chandra Kosaraju - [[pdf]](https://arxiv.org/pdf/2601.04521)</summary>

**Abstract:** The design of reliable, valid, and diverse molecules is fundamental to modern drug discovery, as improved molecular generation supports efficient exploration of the chemical space for potential drug candidates and reduces the cost of early design efforts. Despite these needs, current chemical language models that generate molecules as SMILES strings are vulnerable to compounding token errors: many samples are unparseable or chemically implausible, and hard constraints meant to prevent failure can restrict exploration. To address this gap, we introduce TSSR, a Two-Stage, Swap-Reward-driven reinforcement learning (RL) framework for character-level SMILES generation. Stage one rewards local token swaps that repair syntax, promoting transitions from invalid to parseable strings. Stage two provides chemistry-aware feedback from RDKit diagnostics, rewarding reductions in valence, aromaticity, and connectivity issues. The reward decomposes into interpretable terms (swap efficiency, error reduction, distance to validity), is model agnostic, and requires no task-specific labels or hand-crafted grammars. We evaluated TSSR on the MOSES benchmark using a GRU policy trained with PPO in both pure RL (P-RL) from random initialization and fine-tuning RL (F-RL) starting from a pretrained chemical language model, assessing 10,000 generated SMILES per run. In P-RL, TSSR significantly improves syntactic validity, chemical validity, and novelty. In F-RL, TSSR preserves drug-likeness and synthesizability while increasing validity and novelty. Token-level analysis shows that syntax edits and chemistry fixes act jointly to reduce RDKit detected errors. TSSR converts a sparse terminal objective into a denser and more interpretable reward, improving both syntactic and chemical quality without reducing diversity. TSSR is dataset-agnostic and can be adapted to various reinforcement learning approaches.

**arXiv ID:** 2601.04521
</details>

<details>
<summary><strong>OctoBench: Benchmarking Scaffold-Aware Instruction Following in Repository-Grounded Agentic Coding</strong> - Deming Ding, Shichun Liu, Enhui Yang, Jiahang Lin, Ziying Chen, Shihan Dou, Honglin Guo, Weiyu Cheng, Pengyu Zhao, Chengjun Xiao, Qunhong Zeng, Qi Zhang, Xuanjing Huang, Qidi Xu, Tao Gui - [[pdf]](https://arxiv.org/pdf/2601.10343)</summary>

**Abstract:** Modern coding scaffolds turn LLMs into capable software agents, but their ability to follow scaffold-specified instructions remains under-examined, especially when constraints are heterogeneous and persist across interactions. To fill this gap, we introduce OctoBench, which benchmarks scaffold-aware instruction following in repository-grounded agentic coding. OctoBench includes 34 environments and 217 tasks instantiated under three scaffold types, and is paired with 7,098 objective checklist items. To disentangle solving the task from following the rules, we provide an automated observation-and-scoring toolkit that captures full trajectories and performs fine-grained checks. Experiments on eight representative models reveal a systematic gap between task-solving and scaffold-aware compliance, underscoring the need for training and evaluation that explicitly targets heterogeneous instruction following. We release the benchmark to support reproducible benchmarking and to accelerate the development of more scaffold-aware coding agents.

**arXiv ID:** 2601.10343
</details>

<details>
<summary><strong>DEER: A Benchmark for Evaluating Deep Research Agents on Expert Report Generation</strong> - Janghoon Han, Heegyu Kim, Changho Lee, Dahm Lee, Min Hyung Park, Hosung Song, Stanley Jungkyu Choi, Moontae Lee, Honglak Lee - [[pdf]](https://arxiv.org/pdf/2512.17776)</summary>

**Abstract:** As large language models advance, deep research systems capable of generating expert-level reports through multi-step reasoning and evidence-based synthesis are emerging. However, evaluating such reports remains challenging. Existing benchmarks often lack systematic evaluation criteria, rely heavily on LLM-based judges that may miss issues requiring expert judgment, and verify only a limited subset of explicitly cited statements rather than report-wide factual reliability. To address these limitations, we introduce DEER, a benchmark for evaluating expert-level deep research reports. DEER comprises 50 report-writing tasks spanning 13 domains, along with an expert-grounded evaluation taxonomy with seven dimensions and 25 subdimensions, operationalized into 101 fine-grained rubric items. To improve evaluation consistency, DEER provides task-specific Expert Evaluation Guidance to support LLM-based judging. Complementing rubric-based assessment, we propose a document-level fact-checking architecture that verifies both cited and uncited claims and quantifies the quality and reliability of the supporting evidence. Experimental results show that DEER exhibits strong correlation with human expert judgments and yields interpretable diagnostics of system strengths and weaknesses.

**arXiv ID:** 2512.17776
</details>

<details>
<summary><strong>Action Shapley: A Training Data Selection Metric for World Model in Reinforcement Learning</strong> - Rajat Ghosh, Debojyoti Dutta - [[pdf]](https://arxiv.org/pdf/2601.10905)</summary>

**Abstract:** Numerous offline and model-based reinforcement learning systems incorporate world models to emulate the inherent environments. A world model is particularly important in scenarios where direct interactions with the real environment is costly, dangerous, or impractical. The efficacy and interpretability of such world models are notably contingent upon the quality of the underlying training data. In this context, we introduce Action Shapley as an agnostic metric for the judicious and unbiased selection of training data. To facilitate the computation of Action Shapley, we present a randomized dynamic algorithm specifically designed to mitigate the exponential complexity inherent in traditional Shapley value computations. Through empirical validation across five data-constrained real-world case studies, the algorithm demonstrates a computational efficiency improvement exceeding 80\% in comparison to conventional exponential time computations. Furthermore, our Action Shapley-based training data selection policy consistently outperforms ad-hoc training data selection.

**arXiv ID:** 2601.10905
</details>

<details>
<summary><strong>Realistic Curriculum Reinforcement Learning for Autonomous and Sustainable Marine Vessel Navigation</strong> - Zhang Xiaocai, Xiao Zhe, Liang Maohan, Liu Tao, Li Haijiang, Zhang Wenbin - [[pdf]](https://arxiv.org/pdf/2601.10911)</summary>

**Abstract:** Sustainability is becoming increasingly critical in the maritime transport, encompassing both environmental and social impacts, such as Greenhouse Gas (GHG) emissions and navigational safety. Traditional vessel navigation heavily relies on human experience, often lacking autonomy and emission awareness, and is prone to human errors that may compromise safety. In this paper, we propose a Curriculum Reinforcement Learning (CRL) framework integrated with a realistic, data-driven marine simulation environment and a machine learning-based fuel consumption prediction module. The simulation environment is constructed using real-world vessel movement data and enhanced with a Diffusion Model to simulate dynamic maritime conditions. Vessel fuel consumption is estimated using historical operational data and learning-based regression. The surrounding environment is represented as image-based inputs to capture spatial complexity. We design a lightweight, policy-based CRL agent with a comprehensive reward mechanism that considers safety, emissions, timeliness, and goal completion. This framework effectively handles complex tasks progressively while ensuring stable and efficient learning in continuous action spaces. We validate the proposed approach in a sea area of the Indian Ocean, demonstrating its efficacy in enabling sustainable and safe vessel navigation.

**arXiv ID:** 2601.10911
</details>

<details>
<summary><strong>IMS: Intelligent Hardware Monitoring System for Secure SoCs</strong> - Wadid Foudhaili, Aykut Rencber, Anouar Nechi, Rainer Buchty, Mladen Berekovic, Andres Gomez, Saleh Mulhem - [[pdf]](https://arxiv.org/pdf/2601.11447)</summary>

**Abstract:** In the modern Systems-on-Chip (SoC), the Advanced eXtensible Interface (AXI) protocol exhibits security vulnerabilities, enabling partial or complete denial-of-service (DoS) through protocol-violation attacks. The recent countermeasures lack a dedicated real-time protocol semantic analysis and evade protocol compliance checks. This paper tackles this AXI vulnerability issue and presents an intelligent hardware monitoring system (IMS) for real-time detection of AXI protocol violations. IMS is a hardware module leveraging neural networks to achieve high detection accuracy. For model training, we perform DoS attacks through header-field manipulation and systematic malicious operations, while recording AXI transactions to build a training dataset. We then deploy a quantization-optimized neural network, achieving 98.7% detection accuracy with <=3% latency overhead, and throughput of >2.5 million inferences/s. We subsequently integrate this IMS into a RISC-V SoC as a memory-mapped IP core to monitor its AXI bus. For demonstration and initial assessment for later ASIC integration, we implemented this IMS on an AMD Zynq UltraScale+ MPSoC ZCU104 board, showing an overall small hardware footprint (9.04% look-up-tables (LUTs), 0.23% DSP slices, and 0.70% flip-flops) and negligible impact on the overall design's achievable frequency. This demonstrates the feasibility of lightweight, security monitoring for resource-constrained edge environments.

**arXiv ID:** 2601.11447
</details>

<details>
<summary><strong>DemoTuner: Automatic Performance Tuning for Database Management Systems Based on Demonstration Reinforcement Learning</strong> - Hui Dou, Lei Jin, Yuxuan Zhou, Jiang He, Yiwen Zhang, Zibin Zheng - [[pdf]](https://arxiv.org/pdf/2511.09998)</summary>

**Abstract:** The performance of modern DBMSs such as MySQL and PostgreSQL heavily depends on the configuration of performance-critical knobs. Manual tuning these knobs is laborious and inefficient due to the complex and high-dimensional nature of the configuration space. Among the automated tuning methods, reinforcement learning (RL)-based methods have recently sought to improve the DBMS knobs tuning process from several different perspectives. However, they still encounter challenges with slow convergence speed during offline training. In this paper, we mainly focus on how to leverage the valuable tuning hints contained in various textual documents such as DBMS manuals and web forums to improve the offline training of RL-based methods. To this end, we propose an efficient DBMS knobs tuning framework named DemoTuner via a novel LLM-assisted demonstration reinforcement learning method. Specifically, to comprehensively and accurately mine tuning hints from documents, we design a structured chain of thought prompt to employ LLMs to conduct a condition-aware tuning hints extraction task. To effectively integrate the mined tuning hints into RL agent training, we propose a hint-aware demonstration reinforcement learning algorithm HA-DDPGfD in DemoTuner. As far as we know, DemoTuner is the first work to introduce the demonstration reinforcement learning algorithm for DBMS knobs tuning. Experimental evaluations conducted on MySQL and PostgreSQL across various workloads demonstrate that DemoTuner achieves performance gains of up to 44.01% for MySQL and 39.95% for PostgreSQL over default configurations. Compared with three representative baseline methods, DemoTuner is able to further reduce the execution time by up to 10.03%, while always consuming the least online tuning cost. Additionally, DemoTuner also exhibits superior adaptability to application scenarios with unknown workloads.

**arXiv ID:** 2511.09998
</details>

<details>
<summary><strong>Reinforcement Learning to Discover a NorthEast Monsoon Index for Monthly Rainfall Prediction in Thailand</strong> - Kiattikun Chobtham - [[pdf]](https://arxiv.org/pdf/2601.10181)</summary>

**Abstract:** Climate prediction is a challenge due to the intricate spatiotemporal patterns within Earth systems. Global climate indices, such as the El Niño Southern Oscillation, are standard input features for long-term rainfall prediction. However, a significant gap persists regarding local-scale indices capable of improving predictive accuracy in specific regions of Thailand. This paper introduces a novel NorthEast monsoon climate index calculated from sea surface temperature to reflect the climatology of the boreal winter monsoon. To optimise the calculated areas used for this index, a Deep Q-Network reinforcement learning agent explores and selects the most effective rectangles based on their correlation with seasonal rainfall. Rainfall stations were classified into 12 distinct clusters to distinguish rainfall patterns between southern and upper Thailand. Experimental results show that incorporating the optimised index into Long Short-Term Memory models significantly improves long-term monthly rainfall prediction skill in most cluster areas. This approach effectively reduces the Root Mean Square Error for 12-month-ahead forecasts.

**arXiv ID:** 2601.10181
</details>

<details>
<summary><strong>Off Policy Lyapunov Stability in Reinforcement Learning</strong> - Sarvan Gill, Daniela Constantinescu - [[pdf]](https://arxiv.org/pdf/2509.09863)</summary>

**Abstract:** Traditional reinforcement learning lacks the ability to provide stability guarantees. More recent algorithms learn Lyapunov functions alongside the control policies to ensure stable learning. However, the current self-learned Lyapunov functions are sample inefficient due to their on-policy nature. This paper introduces a method for learning Lyapunov functions off-policy and incorporates the proposed off-policy Lyapunov function into the Soft Actor Critic and Proximal Policy Optimization algorithms to provide them with a data efficient stability certificate. Simulations of an inverted pendulum and a quadrotor illustrate the improved performance of the two algorithms when endowed with the proposed off-policy Lyapunov function.

**arXiv ID:** 2509.09863
</details>

<details>
<summary><strong>VLAgents: A Policy Server for Efficient VLA Inference</strong> - Tobias Jülg, Khaled Gamal, Nisarga Nilavadi, Pierre Krack, Seongjin Bien, Michael Krawez, Florian Walter, Wolfram Burgard - [[pdf]](https://arxiv.org/pdf/2601.11250)</summary>

**Abstract:** The rapid emergence of Vision-Language-Action models (VLAs) has a significant impact on robotics. However, their deployment remains complex due to the fragmented interfaces and the inherent communication latency in distributed setups. To address this, we introduce VLAgents, a modular policy server that abstracts VLA inferencing behind a unified Gymnasium-style protocol. Crucially, its communication layer transparently adapts to the context by supporting both zero-copy shared memory for high-speed simulation and compressed streaming for remote hardware. In this work, we present the architecture of VLAgents and validate it by integrating seven policies -- including OpenVLA and Pi Zero. In a benchmark with both local and remote communication, we further demonstrate how it outperforms the default policy servers provided by OpenVLA, OpenPi, and LeRobot. VLAgents is available at this https URL

**arXiv ID:** 2601.11250
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
