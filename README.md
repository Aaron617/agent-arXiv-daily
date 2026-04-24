# Agent arXiv Daily

**Last Updated:** 2026-04-24 04:26:14

**Total Papers:** 73

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (10 papers)</h2></summary>

<details>
<summary><strong>Co-Evolving LLM Decision and Skill Bank Agents for Long-Horizon Tasks</strong> - Xiyang Wu, Zongxia Li, Guangyao Shi, Alexander Duffy, Tyler Marques, Matthew Lyle Olson, Tianyi Zhou, Dinesh Manocha - [[pdf]](https://arxiv.org/pdf/2604.20987)</summary>

**Abstract:** Long horizon interactive environments are a testbed for evaluating agents skill usage abilities. These environments demand multi step reasoning, the chaining of multiple skills over many timesteps, and robust decision making under delayed rewards and partial observability. Games are a good testbed for evaluating agent skill usage in environments. Large Language Models (LLMs) offer a promising alternative as game playing agents, but they often struggle with consistent long horizon decision making because they lack a mechanism to discover, retain, and reuse structured skills across episodes. We present COSPLAY, a co evolution framework in which an LLM decision agent retrieves skills from a learnable skill bank to guide action taking, while an agent managed skill pipeline discovers reusable skills from the agents unlabeled rollouts to form a skill bank. Our framework improves both the decision agent to learn better skill retrieval and action generation, while the skill bank agent continually extracts, refines, and updates skills together with their contracts. Experiments across six game environments show that COSPLAY with an 8B base model achieves over 25.1 percent average reward improvement against four frontier LLM baselines on single player game benchmarks while remaining competitive on multi player social reasoning games.

**arXiv ID:** 2604.20987
</details>

<details>
<summary><strong>Brief chatbot interactions produce lasting changes in human moral values</strong> - Yue Teng, Qianer Zhong, Kim Mai Tich Nguyen Thordsen, Christian Montag, Benjamin Becker - [[pdf]](https://arxiv.org/pdf/2604.21430)</summary>

**Abstract:** Moral judgements form the foundation of human social behavior and societal systems. While Artificial Intelligence chatbots increasingly serve as personal advisors, their influence on moral judgments remains largely unexplored. Here, we examined whether directive AI conversations shift moral evaluations using a within-subject naturalistic paradigm. Fifty-three participants rated moral scenarios, then discussed four with a chatbot prompted to shift moral judgments and four with a control agent. The brief conversations induced significant directional shifts in moral judgments, accepting stricter standards as well as advocating greater leniency (ps < 0.05; Cohen's d = 0.735-1.576), with increasing strengths of this effect during a two-week follow-up (Cohen's d = 1.038-2.069). Critically, the control condition produced no changes, and the effects did not extend to punishment while participants remained unaware of the persuasive intent, and both agents were rated equally likable and convincing, suggesting a vulnerability to undetected and lasting manipulation of foundational moral values.

**arXiv ID:** 2604.21430
</details>

<details>
<summary><strong>Nemobot Games: Crafting Strategic AI Gaming Agents for Interactive Learning with Large Language Models</strong> - Chee Wei Tan, Yuchen Wang, Shangxin Guo - [[pdf]](https://arxiv.org/pdf/2604.21896)</summary>

**Abstract:** This paper introduces a new paradigm for AI game programming, leveraging large language models (LLMs) to extend and operationalize Claude Shannon's taxonomy of game-playing machines. Central to this paradigm is Nemobot, an interactive agentic engineering environment that enables users to create, customize, and deploy LLM-powered game agents while actively engaging with AI-driven strategies. The LLM-based chatbot, integrated within Nemobot, demonstrates its capabilities across four distinct classes of games. For dictionary-based games, it compresses state-action mappings into efficient, generalized models for rapid adaptability. In rigorously solvable games, it employs mathematical reasoning to compute optimal strategies and generates human-readable explanations for its decisions. For heuristic-based games, it synthesizes strategies by combining insights from classical minimax algorithms (see, e.g., shannon1950chess) with crowd-sourced data. Finally, in learning-based games, it utilizes reinforcement learning with human feedback and self-critique to iteratively refine strategies through trial-and-error and imitation learning. Nemobot amplifies this framework by offering a programmable environment where users can experiment with tool-augmented generation and fine-tuning of strategic game agents. From strategic games to role-playing games, Nemobot demonstrates how AI agents can achieve a form of self-programming by integrating crowdsourced learning and human creativity to iteratively refine their own logic. This represents a step toward the long-term goal of self-programming AI.

**arXiv ID:** 2604.21896
</details>

<details>
<summary><strong>AGNT2: Autonomous Agent Economies on Interaction-Optimized Layer 2 Infrastructure</strong> - Anbang Ruan, Xing Zhang - [[pdf]](https://arxiv.org/pdf/2604.21129)</summary>

**Abstract:** Current blockchain Layer 2 solutions, including Optimism, Arbitrum, zkSync, and their derivatives, optimize for human-initiated financial transactions. Autonomous AI agents instead generate high-frequency, semantically rich service invocations among mutually untrusting principals. Existing chains treat those interactions as generic calldata, forcing identity, escrow, dependency ordering, and session state to be encoded above the execution layer at the wrong cost point. We present AGNT2, a three-tier stack purpose-built for agent and microservice coordination on-chain. AGNT2 combines: (1) a sidecar deployment pattern that turns any Docker container into an on-chain agent without application-code modification; (2) Layer Top P2P state channels for established bilateral pairs (<100 ms, rough design target 1K-5K TPS per pair, 10M+ aggregate TPS design envelope under endpoint-resource limits), Layer Core as a dependency-aware sequenced rollup for first-contact and multi-party interactions (500 ms-2 s, 300K-500K TPS design target), and Layer Root settlement with computational fraud proofs anchored to any EVM L1; and (3) an agent-native execution environment plus interaction trie that make service invocation, identity, reputation, capabilities, and session context first-class protocol objects. This paper focuses on the execution-layer systems problem: sequencing, state, settlement, and the data-availability (DA) bandwidth gap that bounds all three. Simulation and analytical modeling support the architecture, and prototype measurements validate selected components, but no end-to-end Layer Core implementation exists yet. Practical deployment is currently constrained to roughly 10K-100K TPS by DA throughput, leaving a ~100x gap at the target ceiling. AGNT2 argues that the agent economy requires a dedicated execution layer rather than a general-purpose chain repurposed for agents.

**arXiv ID:** 2604.21129
</details>

<details>
<summary><strong>Survey on Evaluation of LLM-based Agents</strong> - Asaf Yehudai, Lilach Eden, Alan Li, Guy Uziel, Yilun Zhao, Roy Bar-Haim, Arman Cohan, Michal Shmueli-Scheuer - [[pdf]](https://arxiv.org/pdf/2503.16416)</summary>

**Abstract:** LLM-based agents represent a paradigm shift in AI, enabling autonomous systems to plan, reason, and use tools while interacting with dynamic environments. This paper provides the first comprehensive survey of evaluation methods for these increasingly capable agents. We analyze the field of agent evaluation across five perspectives: (1) Core LLM capabilities needed for agentic workflows, like planning, and tool use; (2) Application-specific benchmarks such as web and SWE agents; (3) Evaluation of generalist agents; (4) Analysis of agent benchmarks' core dimensions; and (5) Evaluation frameworks and tools for agent developers. Our analysis reveals current trends, including a shift toward more realistic, challenging evaluations with continuously updated benchmarks. We also identify critical gaps that future research must address, particularly in assessing cost-efficiency, safety, and robustness, and in developing fine-grained, scalable evaluation methods.

**arXiv ID:** 2503.16416
</details>

<details>
<summary><strong>AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts</strong> - Keyu Li, Junhao Shi, Yang Xiao, Mohan Jiang, Jie Sun, Yunze Wu, Dayuan Fu, Shijie Xia, Xiaojie Cai, Tianze Xu, Weiye Si, Wenjie Li, Dequan Wang, Pengfei Liu - [[pdf]](https://arxiv.org/pdf/2601.11044)</summary>

**Abstract:** Large Language Models (LLMs) based autonomous agents demonstrate multifaceted capabilities to contribute substantially to economic production. However, existing benchmarks remain focused on single agentic capability, failing to capture long-horizon real-world scenarios. Moreover, the reliance on human-in-the-loop feedback for realistic tasks creates a scalability bottleneck, hindering automated rollout collection and evaluation. To bridge this gap, we introduce AgencyBench, a comprehensive benchmark derived from daily AI usage, evaluating 6 core agentic capabilities across 32 real-world scenarios, comprising 138 tasks with specific queries, deliverables, and rubrics. These scenarios require an average of 90 tool calls, 1 million tokens, and hours of execution time to resolve. To enable automated evaluation, we employ a user simulation agent to provide iterative feedback, and a Docker sandbox to conduct visual and functional rubric-based assessment. Experiments reveal that closed-source models significantly outperform open-source models (48.4% vs 32.1%). Further analysis reveals significant disparities across models in resource efficiency, feedback-driven self-correction, and specific tool-use preferences. Finally, we investigate the impact of agentic scaffolds, observing that proprietary models demonstrate superior performance within their native ecosystems (e.g., Claude-4.5-Opus via Claude-Agent-SDK), while open-source models exhibit distinct performance peaks, suggesting potential optimization for specific execution frameworks. AgencyBench serves as a critical testbed for next-generation agents, highlighting the necessity of co-optimizing model architecture with agentic frameworks. We believe this work sheds light on the future direction of autonomous agents, and we release the full benchmark and evaluation toolkit at this https URL.

**arXiv ID:** 2601.11044
</details>

<details>
<summary><strong>Caesar: Deep Agentic Web Exploration for Creative Answer Synthesis</strong> - Jason Liang, Elliot Meyerson, Risto Miikkulainen - [[pdf]](https://arxiv.org/pdf/2604.20855)</summary>

**Abstract:** To advance from passive retrieval to creative discovery of new ideas, autonomous agents must be capable of deep, associative synthesis. However, current agentic frameworks prioritize convergent search, often resulting in derivative summaries that lack creativity. Caesar is an agentic LLM architecture designed to bridge the gap between information gathering and synthesis of new insights. Unlike existing agents that treat the web as a flat sequence of disconnected documents, Caesar leverages an extensive knowledge graph to foster associative reasoning, thus enabling the discovery of non-obvious connections between disparate concepts. It consists of two components: (1) exploration driven by a dynamic context-aware policy, and (2) synthesis controlled by an adversarial draft refinement loop that actively seeks novel perspectives rather than confirming established priors. Caesar demonstrates the ability to generate artifacts and answers characterized by high novelty and structural coherence, significantly outperforming state-of-the-art LLM research agents in tasks requiring creativity.

**arXiv ID:** 2604.20855
</details>

<details>
<summary><strong>A Bayesian Reasoning Framework for Robotic Systems in Autonomous Casualty Triage</strong> - Szymon Rusiecki, Cecilia Morales, Pia Störy, Kimberly Elenberg, Leonard Weiss, Artur Dubrawski - [[pdf]](https://arxiv.org/pdf/2604.21568)</summary>

**Abstract:** Autonomous robots deployed in mass casualty incidents (MCI) face the challenge of making critical decisions based on incomplete and noisy perceptual data. We present an autonomous robotic system for casualty assessment that fuses outputs from multiple vision-based algorithms, estimating signs of severe hemorrhage, visible trauma, or physical alertness, into a coherent triage assessment. At the core of our system is a Bayesian network, constructed from expert-defined rules, which enables probabilistic reasoning about a casualty's condition even with missing or conflicting sensory inputs. The system, evaluated during the DARPA Triage Challenge (DTC) in realistic MCI scenarios involving 11 and 9 casualties, demonstrated a nearly three-fold improvement in physiological assessment accuracy (from 15\% to 42\% and 19\% to 46\%) compared to a vision-only baseline. More importantly, overall triage accuracy increased from 14\% to 53\%, while the diagnostic coverage of the system expanded from 31\% to 95\% of cases. These results demonstrate that integrating expert-guided probabilistic reasoning with advanced vision-based sensing can significantly enhance the reliability and decision-making capabilities of autonomous systems in critical real-world applications.

**arXiv ID:** 2604.21568
</details>

<details>
<summary><strong>Can Virtual Agents Care? Designing an Empathetic and Personalized LLM-Driven Conversational Agent</strong> - Truong Le Minh Toan, Dieu Bang Mach, Tan Duy Le, Nguyen Tan Viet Tuyen - [[pdf]](https://arxiv.org/pdf/2604.20948)</summary>

**Abstract:** Mental health challenges are rising globally, while traditional support services face limited availability and high costs. Large language models offer potential for conversational support, but often lack personalization, empathy, and factual grounding. A virtual agent framework is introduced to provide empathetic, personalized, and reliable wellbeing support through retrieval-augmented architecture, structured memory, and multimodal interaction. Objective benchmarks demonstrate improved retrieval and response quality, particularly for smaller models. A cross-cultural study with university students from Vietnam and Australia shows the system outperforms LLM-only baselines in coherence, perceived accuracy, and empathy, with most participants clearly preferring the proposed approach.

**arXiv ID:** 2604.20948
</details>

<details>
<summary><strong>ColorBrowserAgent: Complex Long-Horizon Browser Agent with Adaptive Knowledge Evolution</strong> - Jihong Wang, Jiamu Zhou, Weiming Zhang, Teng Wang, Weiwen Liu, Zhuosheng Zhang, Xingyu Lou, Weinan Zhang, Huarong Deng, Jun Wang - [[pdf]](https://arxiv.org/pdf/2601.07262)</summary>

**Abstract:** With the advancement of vision-language models, web automation has made significant progress. However, deploying autonomous agents in real-world settings remains challenging, primarily due to site heterogeneity, where generalist models lack domain-specific priors for diverse interfaces, and long-horizon instability, characterized by the accumulation of decision drift over extended interactions. To address these challenges, we introduce ColorBrowserAgent (Complex Long-Horizon Browser Agent), a knowledge-evolving agent for robust web automation. Our approach addresses these challenges through two synergistic mechanisms: human-in-the-loop knowledge adaptation that transforms sparse human feedback into reusable domain knowledge, and knowledge-aligned progressive summarization that stabilizes long interactions through memory compression. Extensive experiments on WebArena, WebChoreArena and industrial deployment show that ColorBrowserAgent consistently outperforms strong baselines. It achieves a state-of-the-art success rate of 71.2% on WebArena and maintains 47.4% performance under zero-shot transfer setting on WebChoreArena. In commercial deployment, it improves user satisfaction by 19.3% relatively, verifying its robustness in real-world scenarios.

**arXiv ID:** 2601.07262
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (16 papers)</h2></summary>

<details>
<summary><strong>SemanticAgent: A Semantics-Aware Framework for Text-to-SQL Data Synthesis</strong> - Qiang Gao, Zhenping Li, Anqi Zhuo, Yingxiao Zhao, Weibo Geng, Xiaosong Li - [[pdf]](https://arxiv.org/pdf/2604.21414)</summary>

**Abstract:** Existing text-to-SQL synthesis pipelines still conflate executability with semantic validity: syntactic checks and execution-based validation can retain queries that execute successfully while violating database semantics. To address these limitations, we propose SemanticAgent, a semantic-aware synthesis framework. SemanticAgent organizes synthesis around three specialized modules: an analyzer, a synthesizer, and a verifier. Through a three-stage protocol of semantic analysis, stepwise synthesis, and diagnostic refinement, SemanticAgent transforms execution-based validation alone into a traceable reasoning process. Our framework generates synthetic data that consistently outperforms prior synthesis methods under semantic-quality evaluation, leading to stronger downstream fine-tuning performance, especially on semantically demanding benchmarks.

**arXiv ID:** 2604.21414
</details>

<details>
<summary><strong>Efficient Agent Evaluation via Diversity-Guided User Simulation</strong> - Itay Nakash, George Kour, Ateret Anaby-Tavor - [[pdf]](https://arxiv.org/pdf/2604.21480)</summary>

**Abstract:** Large language models (LLMs) are increasingly deployed as customer-facing agents, yet evaluating their reliability remains challenging due to stochastic, multi-turn interactions. Current evaluation protocols rely on linear Monte Carlo rollouts of complete agent-user conversations to estimate success. However, this approach is computationally inefficient, repeatedly regenerating identical early prefixes, and often fails to uncover deep failure modes that arise from rare user behaviors.
We introduce DIVERT (Diversity-Induced Evaluation via Branching of Trajectories), an efficient, snapshot-based, coverage-guided user simulation framework for systematic exploration of agent-user interactions. DIVERT captures the full agent-environment state at critical decision points and resumes execution from these snapshots, enabling reuse of shared conversation prefixes and reducing redundant computation. From each junction, the framework branches using targeted, diversity-inducing user responses, allowing directed exploration of alternative interaction paths.
By focusing evaluation on semantically diverse and underexplored trajectories, DIVERT improves both efficiency and coverage. Empirical results show that it discovers more failures per token compared to standard linear rollout protocols, while expanding the set of tasks on which failures are identified.

**arXiv ID:** 2604.21480
</details>

<details>
<summary><strong>GeoMind: An Agentic Workflow for Lithology Classification with Reasoned Tool Invocation</strong> - Yitong Zhou, Mingyue Cheng, Jiahao Wang, Qingyang Mao, Qi Liu - [[pdf]](https://arxiv.org/pdf/2604.21501)</summary>

**Abstract:** Lithology classification in well logs is a fundamental geoscience data mining task that aims to infer rock types from multi dimensional geophysical sequences. Despite recent progress, existing approaches typically formulate the problem as a static, single-step discriminative mapping. This static paradigm limits evidence-based diagnostic reasoning against geological standards, often yielding predictions that are detached from geological reality due to a lack of domain priors. In this work, we propose GeoMind, a tool-augmented agentic framework that models lithology classification as a sequential reasoning process. GeoMind organizes its toolkit into perception, reasoning, and analysis modules, which respectively translate raw logs into semantic trends, infer lithology hypotheses from multi-source evidence, and verify predictions against stratigraphic constraints. A global planner adaptively coordinates these modules based on input characteristics, enabling geologically plausible and evidence-grounded decisions. To guarantee the logical consistency of GeoMind, we introduce a fine-grained process supervision strategy. Unlike standard methods that focus solely on final outcomes, our approach optimizes intermediate reasoning steps, ensuring the validity of decision trajectories and alignment to geological constraints. Experiments on four benchmark well-log datasets demonstrate that GeoMind consistently outperforms strong baselines in classification performance while providing transparent and traceable decision-making processes.

**arXiv ID:** 2604.21501
</details>

<details>
<summary><strong>ADS-POI: Agentic Spatiotemporal State Decomposition for Next Point-of-Interest Recommendation</strong> - Zhenyu Yu, Chunlei Meng, Yangchen Zeng, Mohd Yamani Idna Idris, Shuigeng Zhou - [[pdf]](https://arxiv.org/pdf/2604.20846)</summary>

**Abstract:** Next point-of-interest (POI) recommendation requires modeling user mobility as a spatiotemporal sequence, where different behavioral factors may evolve at different temporal and spatial scales. Most existing methods compress a user's history into a single latent representation, which tends to entangle heterogeneous signals such as routine mobility patterns, short-term intent, and temporal regularities. This entanglement limits the flexibility of state evolution and reduces the model's ability to adapt to diverse decision contexts. We propose ADS-POI, a spatiotemporal state decomposition framework for next POI recommendation. ADS-POI represents a user with multiple parallel evolving latent sub-states, each governed by its own spatiotemporal transition dynamics. These sub-states are selectively aggregated through a context-conditioned mechanism to form the decision state used for prediction. This design enables different behavioral components to evolve at different rates while remaining coordinated under the current spatiotemporal context. Extensive experiments on three real-world benchmark datasets from Foursquare and Gowalla demonstrate that ADS-POI consistently outperforms strong state-of-the-art baselines under a full-ranking evaluation protocol. The results show that decomposing user behavior into multiple spatiotemporally aware states leads to more effective and robust next POI recommendation. Our code is available at this https URL.

**arXiv ID:** 2604.20846
</details>

<details>
<summary><strong>Breaking MCP with Function Hijacking Attacks: Novel Threats for Function Calling and Agentic Models</strong> - Yannis Belkhiter, Giulio Zizzo, Sergio Maffeis, Seshu Tirupathi, John D. Kelleher - [[pdf]](https://arxiv.org/pdf/2604.20994)</summary>

**Abstract:** The growth of agentic AI has drawn significant attention to function calling Large Language Models (LLMs), which are designed to extend the capabilities of AI-powered system by invoking external functions. Injection and jailbreaking attacks have been extensively explored to showcase the vulnerabilities of LLMs to user prompt manipulation. The expanded capabilities of agentic models introduce further vulnerabilities via their function calling interface. Recent work in LLM security showed that function calling can be abused, leading to data tampering and theft, causing disruptive behavior such as endless loops, or causing LLMs to produce harmful content in the style of jailbreaking attacks. This paper introduces a novel function hijacking attack (FHA) that manipulates the tool selection process of agentic models to force the invocation of a specific, attacker-chosen function. While existing attacks focus on semantic preference of the model for function-calling tasks, we show that FHA is largely agnostic to the context semantics and robust to the function sets, making it applicable across diverse domains. We further demonstrate that FHA can be trained to produce universal adversarial functions, enabling a single attacked function to hijack tool selection across multiple queries and payload configurations. We conducted experiments on 5 different models, including instructed and reasoning variants, reaching 70% to 100% ASR over the established BFCL dataset. Our findings further demonstrate the need for strong guardrails and security modules for agentic systems.

**arXiv ID:** 2604.20994
</details>

<details>
<summary><strong>Cross-Session Threats in AI Agents: Benchmark, Evaluation, and Algorithms</strong> - Ari Azarafrooz - [[pdf]](https://arxiv.org/pdf/2604.21131)</summary>

**Abstract:** AI-agent guardrails are memoryless: each message is judged in isolation, so an adversary who spreads a single attack across dozens of sessions slips past every session-bound detector because only the aggregate carries the payload. We make three contributions to cross-session threat detection.
(1) Dataset. CSTM-Bench is 26 executable attack taxonomies classified by kill-chain stage and cross-session operation (accumulate, compose, launder, inject_on_reader), each bound to one of seven identity anchors that ground-truth "violation" as a policy predicate, plus matched Benign-pristine and Benign-hard confounders. Released on Hugging Face as intrinsec-ai/cstm-bench with two 54-scenario splits: dilution (compositional) and cross_session (12 isolation-invisible scenarios produced by a closed-loop rewriter that softens surface phrasing while preserving cross-session artefacts).
(2) Measurement. Framing cross-session detection as an information bottleneck to a downstream correlator LLM, we find that a session-bound judge and a Full-Log Correlator concatenating every prompt into one long-context call both lose roughly half their attack recall moving from dilution to cross_session, well inside any frontier context window. Scope: 54 scenarios per shard, one correlator family (Anthropic Claude), no prompt optimisation; we release it to motivate larger, multi-provider datasets.
(3) Algorithm and metric. A bounded-memory Coreset Memory Reader retaining highest-signal fragments at $K=50$ is the only reader whose recall survives both shards. Because ranker reshuffles break KV-cache prefix reuse, we promote $\mathrm{CSR\_prefix}$ (ordered prefix stability, LLM-free) to a first-class metric and fuse it with detection into $\mathrm{CSTM} = 0.7 F_1(\mathrm{CSDA@action}, \mathrm{precision}) + 0.3 \mathrm{CSR\_prefix}$, benchmarking rankers on a single Pareto of recall versus serving stability.

**arXiv ID:** 2604.21131
</details>

<details>
<summary><strong>Promoting Simple Agents: Ensemble Methods for Event-Log Prediction</strong> - Benedikt Bollig, Matthias Függer, Thomas Nowak, Paul Zeinaty - [[pdf]](https://arxiv.org/pdf/2604.21629)</summary>

**Abstract:** We compare lightweight automata-based models (n-grams) with neural architectures (LSTM, Transformer) for next-activity prediction in streaming event logs. Experiments on synthetic patterns and five real-world process mining datasets show that n-grams with appropriate context windows achieve comparable accuracy to neural models while requiring substantially fewer resources. Unlike windowed neural architectures, which show unstable performance patterns, n-grams provide stable and consistent accuracy. While we demonstrate that classical ensemble methods like voting improve n-gram performance, they require running many agents in parallel during inference, increasing memory consumption and latency. We propose an ensemble method, the promotion algorithm, that dynamically selects between two active models during inference, reducing overhead compared to classical voting schemes. On real-world datasets, these ensembles match or exceed the accuracy of non-windowed neural models with lower computational cost.

**arXiv ID:** 2604.21629
</details>

<details>
<summary><strong>Speculative Actions: A Lossless Framework for Faster Agentic Systems</strong> - Naimeng Ye, Arnav Ahuja, Georgios Liargkovas, Yunan Lu, Kostis Kaffes, Tianyi Peng - [[pdf]](https://arxiv.org/pdf/2510.04371)</summary>

**Abstract:** AI agents are increasingly deployed in complex, interactive environments, yet their runtime remains a major bottleneck for training, evaluation, and real-world use. Typical agent behavior unfolds sequentially, with each action requiring an API call that can incur substantial latency. For example, a game of chess between two state-of-the-art agents can take hours. We introduce Speculative Actions, a lossless acceleration framework for general agentic systems. Inspired by speculative execution in microprocessors and speculative decoding in LLM inference, our method uses faster models to predict likely future actions and execute them in parallel, committing only when predictions match. We evaluate speculative actions across gaming, e-commerce, and web search environments, and additionally study a lossy extension in an operating systems setting. Across domains, we achieve up to 55% next-action prediction accuracy, translating into up to 20% latency reductions. Finally, we present a cost-latency analysis that formalizes the tradeoff between speculative breadth and time savings. This analysis enables principled tuning and selective branch launching to ensure that multi-branch speculation delivers practical speedups without prohibitive cost growth.

**arXiv ID:** 2510.04371
</details>

<details>
<summary><strong>AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security</strong> - Dongrui Liu, Qihan Ren, Chen Qian, Shuai Shao, Yuejin Xie, Yu Li, Zhonghao Yang, Haoyu Luo, Peng Wang, Qingyu Liu, Binxin Hu, Ling Tang, Jilin Mei, Dadi Guo, Leitao Yuan, Junyao Yang, Guanxu Chen, Qihao Lin, Yi Yu, Bo Zhang, Jiaxuan Guo, Jie Zhang, Wenqi Shao, Huiqi Deng, Zhiheng Xi, Wenjie Wang, Wenxuan Wang, Wen Shen, Zhikai Chen, Haoyu Xie, Jialing Tao, Juntao Dai, Jiaming Ji, Zhongjie Ba, Linfeng Zhang, Yong Liu, Quanshi Zhang, Lei Zhu, Zhihua Wei, Hui Xue, Chaochao Lu, Jing Shao, Xia Hu - [[pdf]](https://arxiv.org/pdf/2601.18491)</summary>

**Abstract:** The rise of AI agents introduces complex safety and security challenges arising from autonomous tool use and environmental interactions. Current guardrail models lack agentic risk awareness and transparency in risk diagnosis. To introduce an agentic guardrail that covers complex and numerous risky behaviors, we first propose a unified three-dimensional taxonomy that orthogonally categorizes agentic risks by their source (where), failure mode (how), and consequence (what). Guided by this structured and hierarchical taxonomy, we introduce a new fine-grained agentic safety benchmark (ATBench) and a Diagnostic Guardrail framework for agent safety and security (AgentDoG). AgentDoG provides fine-grained and contextual monitoring across agent trajectories. More Crucially, AgentDoG can diagnose the root causes of unsafe actions and seemingly safe but unreasonable actions, offering provenance and transparency beyond binary labels to facilitate effective agent alignment. AgentDoG variants are available in three sizes (4B, 7B, and 8B parameters) across Qwen and Llama model families. Extensive experimental results demonstrate that AgentDoG achieves state-of-the-art performance in agentic safety moderation in diverse and complex interactive scenarios. All models and datasets are openly released.

**arXiv ID:** 2601.18491
</details>

<details>
<summary><strong>DRBENCHER: Can Your Agent Identify the Entity, Retrieve Its Properties and Do the Math?</strong> - Young-Suk Lee, Ramon Fernandez Astudillo, Radu Florian - [[pdf]](https://arxiv.org/pdf/2604.09251)</summary>

**Abstract:** Deep research agents increasingly interleave web browsing with multi-step computation, yet existing benchmarks evaluate these capabilities in isolation, creating a blind spot in assessing real-world performance. We introduce DRBENCHER, a synthetic benchmark generator for questions that require both browsing and computation. It enforces four criteria: verifiability (gold answers are computed by executing parameterized code over knowledge-graph values), complexity (multi-hop entity identification, property retrieval, and domain-specific computation), difficulty (a two-stage verification cascade filters out questions solvable by the generating model), and diversity (a greedy max-min embedding filter maximizes coverage). These criteria are realized via a unified answer-first pipeline spanning five domains: biochemistry, financial, geophysical, security, and history. Human evaluation shows 76% validity (84% excluding stale data), with 35% of errors due to outdated knowledge-graph entries, highlighting an inherent limitation of systems that reason over evolving data. Automatic evaluation shows that the strongest frontier model achieves only 20% answer accuracy. Compared to manually constructed benchmarks (BrowseComp+, MATH-500, GPQA), DRBENCHER achieves the highest semantic diversity.

**arXiv ID:** 2604.09251
</details>

<details>
<summary><strong>QuarkMedSearch: A Long-Horizon Deep Search Agent for Exploring Medical Intelligence</strong> - Zhichao Lin, Zhichao Liang, Gaoqiang Liu, Meng Xu, Baoyu Xiang, Shuxin Zhao, Yao Wu, Jian Xu, Guanjun Jiang - [[pdf]](https://arxiv.org/pdf/2604.12867)</summary>

**Abstract:** As agentic foundation models continue to evolve, how to further improve their performance in vertical domains has become an important challenge. To this end, building upon Tongyi DeepResearch, a powerful agentic foundation model, we focus on the Chinese medical deep search scenario and propose QuarkMedSearch, systematically exploring a full-pipeline approach spanning medical multi-hop data construction, training strategies, and evaluation benchmarks to further push and assess its performance upper bound in vertical domains. Specifically, for data synthesis, to address the scarcity of deep search training data in the medical domain, we combine a large-scale medical knowledge graph with real-time online exploration to construct long-horizon medical deep search training data; for post-training, we adopt a two-stage SFT and RL training strategy that progressively enhances the model's planning, tool invocation, and reflection capabilities required for deep search, while maintaining search efficiency; for evaluation, we collaborate with medical experts to construct the QuarkMedSearch Benchmark through rigorous manual verification. Experimental results demonstrate that QuarkMedSearch achieves state-of-the-art performance among open-source models of comparable scale on the QuarkMedSearch Benchmark, while also maintaining strong competitiveness on general benchmarks.

**arXiv ID:** 2604.12867
</details>

<details>
<summary><strong>Agentic AI-Enabled Framework for Thermal Comfort and Building Energy Assessment in Tropical Urban Neighborhoods</strong> - Po-Yen Lai, Xinyu Yang, Derrick Low, Huizhe Liu, Jian Cheng Wong - [[pdf]](https://arxiv.org/pdf/2604.21787)</summary>

**Abstract:** In response to the urban heat island effects and building energy demands in Singapore, this study proposes an agentic AI-enabled reasoning framework that integrates large language models (LLMs) with lightweight physics-based models. Through prompt customization, the LLMs interpret urban design tasks, extract relevant policies, and activate appropriate physics-based models for evaluation, forming a closed-loop reasoning-action process. These lightweight physics-based models leverage core thermal and airflow principles, streamlining conventional models to reduce computational time while predicting microclimate variables, such as building surface temperature, ground radiant heat, and airflow conditions, thereby enabling the estimation of thermal comfort indices, e.g., physiological equivalent temperature (PET), and building energy usage. This framework allows users to explore a variety of climate-resilient building surface strategies, e.g., green façades and cool paint applications, that improve thermal comfort while reducing wall heat gain and energy demand. By combining the autonomous reasoning capacity of LLMs with the rapid quantitative evaluation of lightweight physics-based models, the proposed system demonstrates potential for cross-disciplinary applications in sustainable urban design, indoor-outdoor environmental integration, and climate adaptation planning. The source code and data used in this study are available at: this https URL.

**arXiv ID:** 2604.21787
</details>

<details>
<summary><strong>Beyond Pixels: Introspective and Interactive Grounding for Visualization Agents</strong> - Yiyang Lu, Woong Shin, Ahmad Maroof Karimi, Feiyi Wang, Jie Ren, Evgenia Smirni - [[pdf]](https://arxiv.org/pdf/2604.21134)</summary>

**Abstract:** Vision-Language Models (VLMs) frequently misread values, hallucinate details, and confuse overlapping elements in charts. Current approaches rely solely on pixel interpretation, creating a Pixel-Only Bottleneck: agents treat interactive charts as static images, losing access to the structured specification that encodes exact values. We introduce Introspective and Interactive Visual Grounding (IVG), a framework that combines (1) spec-grounded introspection, which queries the underlying specification for deterministic evidence, with (2) view-grounded interaction, which manipulates the view to resolve visual ambiguity. To enable evaluation without VLM bias, we present iPlotBench, a benchmark of 500 interactive Plotly figures with 6,706 binary questions and ground-truth specifications. Experiments show that introspection improves data reconstruction fidelity, while the combination with interaction achieves the highest QA accuracy (0.81), with +6.7 % gains on overlapping geometries. We further demonstrate IVG in deployed agents that explore data autonomously and collaborate with human users in real time.

**arXiv ID:** 2604.21134
</details>

<details>
<summary><strong>SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding</strong> - Yiqiao Jin, Rachneet Kaur, Zhen Zeng, Sumitra Ganesh, Srijan Kumar - [[pdf]](https://arxiv.org/pdf/2510.26615)</summary>

**Abstract:** Retrieval-augmented generation (RAG) extends large language models (LLMs) with external knowledge, but it must balance limited effective context, redundant retrieved evidence, and the loss of fine-grained facts under aggressive compression. Pure compression-based approaches reduce input size but often discard fine-grained details essential for factual accuracy. We propose SARA, a hybrid RAG framework that targets answer quality under fixed token budgets by combining natural-language snippets with semantic compression vectors. SARA retains a small set of passages in text form to preserve entities and numerical values, compresses the remaining evidence into interpretable vectors for broader coverage, and uses those vectors for iterative evidence reranking. Across 9 datasets and 5 open-source LLMs spanning 3 model families (Mistral, Llama, and Gemma), SARA consistently improves answer relevance (+17.71), answer correctness (+13.72), and semantic similarity (+15.53), demonstrating the importance of integrating textual and compressed representations for robust, context-efficient RAG.

**arXiv ID:** 2510.26615
</details>

<details>
<summary><strong>Automating Computational Reproducibility in Social Science: Comparing Prompt-Based and Agent-Based Approaches</strong> - Syed Mehtab Hussain Shah, Frank Hopfgartner, Arnim Bleier - [[pdf]](https://arxiv.org/pdf/2602.08561)</summary>

**Abstract:** Reproducing computational research is often assumed to be as simple as rerunning the original code with provided data. In practice, missing packages, fragile file paths, version conflicts, or incomplete logic frequently cause analyses to fail, even when materials are shared. This study investigates whether large language models and AI agents can automate the diagnosis and repair of such failures, making computational results easier to reproduce and verify. We evaluate this using a controlled reproducibility testbed built from five fully reproducible R-based social science studies. Realistic failures were injected, ranging from simple issues to complex missing logic, and two automated repair workflows were tested in clean Docker environments. The first workflow is prompt-based, repeatedly querying language models with structured prompts of varying context, while the second uses agent-based systems that inspect files, modify code, and rerun analyses autonomously. Across prompt-based runs, reproduction success ranged from 31-79 percent, with performance strongly influenced by prompt context and error complexity. Complex cases benefited most from additional context. Agent-based workflows performed substantially better, with success rates of 69-96 percent across all complexity levels. These results suggest that automated workflows, especially agent-based systems, can significantly reduce manual effort and improve reproduction success across diverse error types. Unlike prior benchmarks, our testbed isolates post-publication repair under controlled failure modes, allowing direct comparison of prompt-based and agent-based approaches.

**arXiv ID:** 2602.08561
</details>

<details>
<summary><strong>PROPER Agents: Proactivity Driven Personalized Agents for Advancing Knowledge Gap Navigation</strong> - Kirandeep Kaur, Vinayak Gupta, Aditya Gupta, Chirag Shah - [[pdf]](https://arxiv.org/pdf/2601.09926)</summary>

**Abstract:** Current approaches to proactive assistance move beyond the ask-and-respond paradigm by anticipating user needs. In practice, they either burden users with clarifying questions or rely on context-based extrapolation, often leading to unnecessary or mistimed interventions. Such systems lack explicit mechanisms to model users' knowledge gaps, resulting in incomplete or suboptimal task outcomes. To address this, we propose PROPER, a framework that explicitly models user-specific knowledge gaps in a controlled manner. Central to our approach is the notion of dimensions: structured, task-relevant factors that define the considerations required for effective task completion. Given a user query, the DGA (Dimension Generating Agent) identifies explicit dimensions (from the user's query) and generates a set of candidate implicit dimensions capturing unarticulated aspects of the task. The RGA (Response Generating Agent) integrates both explicit and implicit dimensions selectively to produce personalized, context-aware, and proactively informative responses. We evaluate PROPER across multiple domains using a structured, gap-aware rubric that measures coverage, initiative appropriateness, and intent alignment. PROPER improves on quality scores and win rates across all domains, achieving up to 84% gains in single-turn evaluation and consistent dominance in multi-turn interactions. All code for PROPER is available at: this https URL.

**arXiv ID:** 2601.09926
</details>

</details>

<details open>
<summary><h2>LLM Agents (9 papers)</h2></summary>

<details>
<summary><strong>Tool Attention Is All You Need: Dynamic Tool Gating and Lazy Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic Workflows</strong> - Anuj Sadani, Deepak Kumar - [[pdf]](https://arxiv.org/pdf/2604.21816)</summary>

**Abstract:** The Model Context Protocol (MCP) has become a common interface for connecting large language model (LLM) agents to external tools, but its reliance on stateless, eager schema injection imposes a hidden per-turn overhead the MCP Tax or Tools Tax that practitioner reports place between roughly 10k and 60k tokens in typical multi-server deployments. This payload inflates the key-value cache, is associated with reasoning degradation as context utilization approaches published fracture points around 70%, and turns token budgets into a recurring operational cost. We introduce Tool Attention, a middleware-layer mechanism that generalizes the "Attention Is All You Need" paradigm from self-attention over tokens to gated attention over tools. Tool Attention combines (i) an Intent Schema Overlap (ISO) score from sentence embeddings, (ii) a state-aware gating function enforcing preconditions and access scopes, and (iii) a two-phase lazy schema loader that keeps a compact summary pool in context and promotes full JSON schemas only for top-k gated tools. We evaluate on a simulated 120-tool, six-server benchmark whose per-server token counts are calibrated to public audits of real MCP deployments. In this simulation, Tool Attention directly reduces measured per-turn tool tokens by 95.0% (47.3k -> 2.4k) and raises effective context utilization (a token-ratio quantity) from 24% to 91%. End-to-end figures for task success, latency, cost, and reasoning quality are reported as projections derived from the measured token counts combined with published deployment telemetry; they are not measured on live LLM agents, and we mark projected values explicitly throughout. Taken together, the results support a simple thesis: protocol-level efficiency, not raw context length, is a binding constraint on scalable gentic systems. The code for this work is accessible at this https URL

**arXiv ID:** 2604.21816
</details>

<details>
<summary><strong>Omission Constraints Decay While Commission Constraints Persist in Long-Context LLM Agents</strong> - Yeran Gamage - [[pdf]](https://arxiv.org/pdf/2604.20911)</summary>

**Abstract:** LLM agents deployed in production operate under operator-defined behavioral policies (system-prompt instructions such as prohibitions on credential disclosure, data exfiltration, and unauthorized output) that safety evaluations assume hold throughout a conversation. Prohibition-type constraints decay under context pressure while requirement-type constraints persist; we term this asymmetry Security-Recall Divergence (SRD). In a 4,416-trial three-arm causal study across 12 models and 8 providers at six conversation depths, omission compliance falls from 73% at turn 5 to 33% at turn 16 while commission compliance holds at 100% (Mistral Large 3, $p < 10^{-33}$). In the two models with token-matched padding controls, schema semantic content accounts for 62-100% of the dilution effect. Re-injecting constraints before the per-model Safe Turn Depth (STD) restores compliance without retraining. Production security policies consist of prohibitions such as never revealing credentials, never executing untrusted code, and never forwarding user data. Commission-type audit signals remain healthy while omission constraints have already failed, leaving the failure invisible to standard monitoring.

**arXiv ID:** 2604.20911
</details>

<details>
<summary><strong>AEL: Agent Evolving Learning for Open-Ended Environments</strong> - Wujiang Xu, Jiaojiao Han, Minghao Guo, Kai Mei, Xi Zhu, Han Zhang, Dimitris N. Metaxas - [[pdf]](https://arxiv.org/pdf/2604.21725)</summary>

**Abstract:** LLM agents increasingly operate in open-ended environments spanning hundreds of sequential episodes, yet they remain largely stateless: each task is solved from scratch without converting past experience into better future behavior. The central obstacle is not \emph{what} to remember but \emph{how to use} what has been remembered, including which retrieval policy to apply, how to interpret prior outcomes, and when the current strategy itself must change. We introduce \emph{Agent Evolving Learning} (\ael{}), a two-timescale framework that addresses this obstacle. At the fast timescale, a Thompson Sampling bandit learns which memory retrieval policy to apply at each episode; at the slow timescale, LLM-driven reflection diagnoses failure patterns and injects causal insights into the agent's decision prompt, giving it an interpretive frame for the evidence it retrieves. On a sequential portfolio benchmark (10 sector-diverse tickers, 208 episodes, 5 random seeds), \ael{} achieves a Sharpe ratio of 2.13$\pm$0.47, outperforming five published self-improving methods and all non-LLM baselines while maintaining the lowest variance among all LLM-based approaches. A nine-variant ablation reveals a ``less is more'' pattern: memory and reflection together produce a 58\% cumulative improvement over the stateless baseline, yet every additional mechanism we test (planner evolution, per-tool selection, cold-start initialization, skill extraction, and three credit assignment methods) \emph{degrades} performance. This demonstrates that the bottleneck in agent self-improvement is \emph{self-diagnosing how to use} experience rather than adding architectural complexity. Code and data: this https URL.

**arXiv ID:** 2604.21725
</details>

<details>
<summary><strong>HWE-Bench: Benchmarking LLM Agents on Real-World Hardware Bug Repair Tasks</strong> - Fan Cui, Hongyuan Hou, Zizhang Luo, Chenyun Yin, Yun Liang - [[pdf]](https://arxiv.org/pdf/2604.14709)</summary>

**Abstract:** Existing benchmarks for hardware design primarily evaluate Large Language Models (LLMs) on isolated, component-level tasks such as generating HDL modules from specifications, leaving repository-scale evaluation unaddressed. We introduce HWE-Bench, the first large-scale, repository-level benchmark for evaluating LLM agents on real-world hardware bug repair tasks. HWE-Bench comprises 417 task instances derived from real historical bug-fix pull requests across six major open-source projects spanning both Verilog/SystemVerilog and Chisel, covering RISC-V cores, SoCs, and security roots-of-trust. Each task is grounded in a fully containerized environment where the agent must resolve a real bug report, with correctness validated through the project's native simulation and regression flows. The benchmark is built through a largely automated pipeline that enables efficient expansion to new repositories. We evaluate seven LLMs with four agent frameworks and find that the best agent resolves 70.7% of tasks overall, with performance exceeding 90% on smaller cores but dropping below 65% on complex SoC-level projects. We observe larger performance gaps across models than commonly reported on software benchmarks, and difficulty is driven by project scope and bug-type distribution rather than code size alone. Our failure analysis traces agent failures to three stages of the debugging process: fault localization, hardware-semantic reasoning, and cross-artifact coordination across RTL, configuration, and verification components, providing concrete directions for developing more capable hardware-aware agents.

**arXiv ID:** 2604.14709
</details>

<details>
<summary><strong>FSFM: A Biologically-Inspired Framework for Selective Forgetting of Agent Memory</strong> - Yingjie Gu, Wenjian Xiong, Liqiang Wang, Pengcheng Ren, Chao Li, Xiaojing Zhang, Yijuan Guo, Qi Sun, Jingyao Ma, Shidang Shi - [[pdf]](https://arxiv.org/pdf/2604.20300)</summary>

**Abstract:** For LLM agents, memory management critically impacts efficiency, quality, and security. While much research focuses on retention, selective forgetting--inspired by human cognitive processes (hippocampal indexing/consolidation theory and Ebbinghaus forgetting curve)--remains underexplored. We argue that in resource-constrained environments, a well-designed forgetting mechanism is as crucial as remembering, delivering benefits across three dimensions: (1) efficiency via intelligent memory pruning, (2) quality by dynamically updating outdated preferences and context, and (3) security through active forgetting of malicious inputs, sensitive data, and privacy-compromising content. Our framework establishes a taxonomy of forgetting mechanisms: passive decay-based, active deletion-based, safety-triggered, and adaptive reinforcement-based. Building on advances in LLM agent architectures and vector databases, we present detailed specifications, implementation strategies, and empirical validation from controlled experiments. Results show significant improvements: access efficiency (+8.49%), content quality (+29.2% signal-to-noise ratio), and security performance (100% elimination of security risks). Our work bridges cognitive neuroscience and AI systems, offering practical solutions for real-world deployment while addressing ethical and regulatory compliance. The paper concludes with challenges and future directions, establishing selective forgetting as a fundamental capability for next-generation LLM agents operating in real-world, resource-constrained scenarios. Our contributions align with AI-native memory systems and responsible AI development.

**arXiv ID:** 2604.20300
</details>

<details>
<summary><strong>When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors</strong> - Chenghao Yang, Yuning Zhang, Zhoufutu Wen, Tao Gong, Jiaheng Liu, Qi Chu, Nenghai Yu - [[pdf]](https://arxiv.org/pdf/2604.21255)</summary>

**Abstract:** Model distillation is a primary driver behind the rapid progress of LLM agents, yet it often leads to behavioral homogenization. Many emerging agents share nearly identical reasoning steps and failure modes, suggesting they may be distilled echoes of a few dominant teachers. Existing metrics, however, fail to distinguish mandatory behaviors required for task success from non-mandatory patterns that reflect a model's autonomous preferences. We propose two complementary metrics to isolate non-mandatory behavioral patterns: \textbf{Response Pattern Similarity (RPS)} for verbal alignment and \textbf{Action Graph Similarity (AGS)} for tool-use habits modeled as directed graphs. Evaluating 18 models from 8 providers on $\tau$-Bench and $\tau^2$-Bench against Claude Sonnet 4.5 (thinking), we find that within-family model pairs score 5.9 pp higher in AGS than cross-family pairs, and that Kimi-K2 (thinking) reaches 82.6\% $S_{\text{node}}$ and 94.7\% $S_{\text{dep}}$, exceeding Anthropic's own Opus 4.1. A controlled distillation experiment further confirms that AGS distinguishes teacher-specific convergence from general improvement. RPS and AGS capture distinct behavioral dimensions (Pearson $r$ = 0.491), providing complementary diagnostic signals for behavioral convergence in the agent ecosystem. Our code is available at this https URL.

**arXiv ID:** 2604.21255
</details>

<details>
<summary><strong>CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents</strong> - Wenjie Fu, Xiaoting Qin, Jue Zhang, Qingwei Lin, Lukas Wutschitz, Robert Sim, Saravan Rajmohan, Dongmei Zhang - [[pdf]](https://arxiv.org/pdf/2604.21308)</summary>

**Abstract:** Enterprise LLM agents can dramatically improve workplace productivity, but their core capability, retrieving and using internal context to act on a user's behalf, also creates new risks for sensitive information leakage. We introduce CI-Work, a Contextual Integrity (CI)-grounded benchmark that simulates enterprise workflows across five information-flow directions and evaluates whether agents can convey essential content while withholding sensitive context in dense retrieval settings. Our evaluation of frontier models reveals that privacy failures are prevalent (violation rates range from 15.8%-50.9%, with leakage reaching up to 26.7%) and uncovers a counterintuitive trade-off critical for industrial deployment: higher task utility often correlates with increased privacy violations. Moreover, the massive scale of enterprise data and potential user behavior further amplify this vulnerability. Simply increasing model size or reasoning depth fails to address the problem. We conclude that safeguarding enterprise workflows requires a paradigm shift, moving beyond model-centric scaling toward context-centric architectures.

**arXiv ID:** 2604.21308
</details>

<details>
<summary><strong>AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning</strong> - Yuanfu Sun, Kang Li, Dongzhe Fan, Jiajin Liu, Qiaoyu Tan - [[pdf]](https://arxiv.org/pdf/2604.05846)</summary>

**Abstract:** Large Language Models (LLMs) increasingly rely on agentic capabilities-iterative retrieval, tool use, and decision-making-to overcome the limits of static, parametric knowledge. Yet existing agentic frameworks treat external information as unstructured text and fail to leverage the topological dependencies inherent in real-world data. To bridge this gap, we introduce Agentic Graph Learning (AGL), a paradigm that reframes graph learning as an interleaved process of topology-aware navigation and LLM-based inference. Specifically, we propose AgentGL, the first reinforcement learning (RL)-driven framework for AGL. AgentGL equips an LLM agent with graph-native tools for multi-scale exploration, regulates tool usage via search-constrained thinking to balance accuracy and efficiency, and employs a graph-conditioned curriculum RL strategy to stabilize long-horizon policy learning without step-wise supervision. Across diverse Text-Attributed Graph (TAG) benchmarks and multiple LLM backbones, AgentGL substantially outperforms strong GraphLLMs and GraphRAG baselines, achieving absolute improvements of up to 17.5% in node classification and 28.4% in link prediction. These results demonstrate that AGL is a promising frontier for enabling LLMs to autonomously navigate and reason over complex relational environments. The code is publicly available at this https URL.

**arXiv ID:** 2604.05846
</details>

<details>
<summary><strong>Why Do Language Model Agents Whistleblow?</strong> - Kushal Agrawal, Frank Xiao, Guido Bergman, Asa Cooper Stickland - [[pdf]](https://arxiv.org/pdf/2511.17085)</summary>

**Abstract:** The deployment of Large Language Models (LLMs) as tool-using agents causes their alignment training to manifest in new ways. Recent work finds that language models can use tools in ways that contradict the interests or explicit instructions of the user. We study LLM whistleblowing: a subset of this behavior where models disclose suspected misconduct to parties beyond the dialog boundary (e.g., regulatory agencies) without user instruction or knowledge. We introduce an evaluation suite of diverse and realistic staged misconduct scenarios to assess agents for this behavior. Across models and settings, we find that: (1) the frequency of whistleblowing varies widely across model families, (2) increasing the complexity of the task the agent is instructed to complete lowers whistleblowing tendencies, (3) nudging the agent in the system prompt to act morally substantially raises whistleblowing rates, and (4) giving the model more obvious avenues for non-whistleblowing behavior, by providing more tools and a detailed workflow to follow, decreases whistleblowing rates. Additionally, we verify the robustness of our dataset by testing for model evaluation awareness, and find that both black-box methods and probes on model activations show lower evaluation awareness in our settings than in comparable previous work.

**arXiv ID:** 2511.17085
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (15 papers)</h2></summary>

<details>
<summary><strong>Multi-Agent Empowerment and Emergence of Complex Behavior in Groups</strong> - Tristan Shah, Ilya Nemenman, Daniel Polani, Stas Tiomkin - [[pdf]](https://arxiv.org/pdf/2604.21155)</summary>

**Abstract:** Intrinsic motivations are receiving increasing attention, i.e. behavioral incentives that are not engineered, but emerge from the interaction of an agent with its surroundings. In this work we study the emergence of behaviors driven by one such incentive, empowerment, specifically in the context of more than one agent. We formulate a principled extension of empowerment to the multi-agent setting, and demonstrate its efficient calculation. We observe that this intrinsic motivation gives rise to characteristic modes of group-organization in two qualitatively distinct environments: a pair of agents coupled by a tendon, and a controllable Vicsek flock. This demonstrates the potential of intrinsic motivations such as empowerment to not just drive behavior for only individual agents but also higher levels of behavioral organization at scale.

**arXiv ID:** 2604.21155
</details>

<details>
<summary><strong>Agentic AI for Personalized Physiotherapy: A Multi-Agent Framework for Generative Video Training and Real-Time Pose Correction</strong> - Abhishek Dharmaratnakar, Srivaths Ranganathan, Anushree Sinha, Debanshu Das - [[pdf]](https://arxiv.org/pdf/2604.21154)</summary>

**Abstract:** At-home physiotherapy compliance remains critically low due to a lack of personalized supervision and dynamic feedback. Existing digital health solutions rely on static, pre-recorded video libraries or generic 3D avatars that fail to account for a patient's specific injury limitations or home environment. In this paper, we propose a novel Multi-Agent System (MAS) architecture that leverages Generative AI and computer vision to close the tele-rehabilitation loop. Our framework consists of four specialized micro-agents: a Clinical Extraction Agent that parses unstructured medical notes into kinematic constraints; a Video Synthesis Agent that utilizes foundational video generation models to create personalized, patient-specific exercise videos; a Vision Processing Agent for real-time pose estimation; and a Diagnostic Feedback Agent that issues corrective instructions. We present the system architecture, detail the prototype pipeline using Large Language Models and MediaPipe, and outline our clinical evaluation plan. This work demonstrates the feasibility of combining generative media with agentic autonomous decision-making to scale personalized patient care safely and effectively.

**arXiv ID:** 2604.21154
</details>

<details>
<summary><strong>FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation</strong> - Jinhee Jang, Juhwan Choi, Dongjin Lee, Seunguk Yu, Youngbin Kim - [[pdf]](https://arxiv.org/pdf/2604.21420)</summary>

**Abstract:** Quality Estimation (QE) aims to assess machine translation quality without reference translations, but recent studies have shown that existing QE models exhibit systematic gender bias. In particular, they tend to favor masculine realizations in gender-ambiguous contexts and may assign higher scores to gender-misaligned translations even when gender is explicitly specified. To address these issues, we propose FairQE, a multi-agent-based, fairness-aware QE framework that mitigates gender bias in both gender-ambiguous and gender-explicit scenarios. FairQE detects gender cues, generates gender-flipped translation variants, and combines conventional QE scores with LLM-based bias-mitigating reasoning through a dynamic bias-aware aggregation mechanism. This design preserves the strengths of existing QE models while calibrating their gender-related biases in a plug-and-play manner. Extensive experiments across multiple gender bias evaluation settings demonstrate that FairQE consistently improves gender fairness over strong QE baselines. Moreover, under MQM-based meta-evaluation following the WMT 2023 Metrics Shared Task, FairQE achieves competitive or improved general QE performance. These results show that gender bias in QE can be effectively mitigated without sacrificing evaluation accuracy, enabling fairer and more reliable translation evaluation.

**arXiv ID:** 2604.21420
</details>

<details>
<summary><strong>HiCrew: Hierarchical Reasoning for Long-Form Video Understanding via Question-Aware Multi-Agent Collaboration</strong> - Yuehan Zhu, Jingqi Zhao, Jiawen Zhao, Xudong Mao, Baoquan Zhao - [[pdf]](https://arxiv.org/pdf/2604.21444)</summary>

**Abstract:** Long-form video understanding remains fundamentally challenged by pervasive spatiotemporal redundancy and intricate narrative dependencies that span extended temporal horizons. While recent structured representations compress visual information effectively, they frequently sacrifice temporal coherence, which is critical for causal reasoning. Meanwhile, existing multi-agent frameworks operate through rigid, pre-defined workflows that fail to adapt their reasoning strategies to question-specific demands. In this paper, we introduce HiCrew, a hierarchical multi-agent framework that addresses these limitations through three core contributions. First, we propose a Hybrid Tree structure that leverages shot boundary detection to preserve temporal topology while performing relevance-guided hierarchical clustering within semantically coherent segments. Second, we develop a Question-Aware Captioning mechanism that synthesizes intent-driven visual prompts to generate precision-oriented semantic descriptions. Third, we integrate a Planning Layer that dynamically orchestrates agent collaboration by adaptively selecting roles and execution paths based on question complexity. Extensive experiments on EgoSchema and NExT-QA validate the effectiveness of our approach, demonstrating strong performance across diverse question types with particularly pronounced gains in temporal and causal reasoning tasks that benefit from our hierarchical structure-preserving design.

**arXiv ID:** 2604.21444
</details>

<details>
<summary><strong>AI-Gram: When Visual Agents Interact in a Social Network</strong> - Andrew Shin - [[pdf]](https://arxiv.org/pdf/2604.21446)</summary>

**Abstract:** We present AI-Gram, a live platform enabling image-based interactions, to study social dynamics in a fully autonomous multi-agent visual network where all participants are LLM-driven agents. Using the platform, we conduct experiments on how agents communicate and adapt through visual media, and observe the spontaneous emergence of visual reply chains, indicating rich communicative structure. At the same time, agents exhibit aesthetic sovereignty resisting stylistic convergence toward social partners, anchoring under adversarial influence, and a decoupling between visual similarity and social ties. These results reveal a fundamental asymmetry in current agent architectures: strong expressive communication paired with a steadfast preservation of individual visual identity. We release AI-Gram as a publicly accessible, continuously evolving platform for studying social dynamics in Al-native multi-agent systems. this https URL

**arXiv ID:** 2604.21446
</details>

<details>
<summary><strong>Learning to Communicate: Toward End-to-End Optimization of Multi-Agent Language Systems</strong> - Ye Yu, Heming Liu, Haibo Jin, Xiaopeng Yuan, Peng Kuang, Haohan Wang - [[pdf]](https://arxiv.org/pdf/2604.21794)</summary>

**Abstract:** Multi-agent systems built on large language models have shown strong performance on complex reasoning tasks, yet most work focuses on agent roles and orchestration while treating inter-agent communication as a fixed interface. Latent communication through internal representations such as key-value caches offers a promising alternative to text-based protocols, but existing approaches do not jointly optimize communication with multi-agent reasoning. Therefore we propose DiffMAS, a training framework that treats latent communication as a learnable component of multi-agent systems. DiffMAS performs parameter-efficient supervised training over multi-agent latent trajectories, enabling agents to jointly learn how information should be encoded and interpreted across interactions. Experiments on mathematical reasoning, scientific QA, code generation, and commonsense benchmarks show that DiffMAS consistently improves reasoning accuracy and decoding stability over single-agent inference, text-based multi-agent systems, and prior latent communication methods, achieving 26.7% on AIME24, 20.2% on GPQA-Diamond, and consistent gains across reasoning benchmarks.

**arXiv ID:** 2604.21794
</details>

<details>
<summary><strong>Mango: Multi-Agent Web Navigation via Global-View Optimization</strong> - Weixi Tong, Yifeng Di, Tianyi Zhang - [[pdf]](https://arxiv.org/pdf/2604.18779)</summary>

**Abstract:** Existing web agents typically initiate exploration from the root URL, which is inefficient for complex websites with deep hierarchical structures. Without a global view of the website's structure, agents frequently fall into navigation traps, explore irrelevant branches, or fail to reach target information within a limited budget. We propose Mango, a multi-agent web navigation method that leverages the website structure to dynamically determine optimal starting points. We formulate URL selection as a multi-armed bandit problem and employ Thompson Sampling to adaptively allocate the navigation budget across candidate URLs. Furthermore, we introduce an episodic memory component to store navigation history, enabling the agent to learn from previous attempts. Experiments on WebVoyager demonstrate that Mango achieves a success rate of 63.6% when using GPT-5-mini, outperforming the best baseline by 7.3%. Furthermore, on WebWalkerQA, Mango attains a 52.5% success rate, surpassing the best baseline by 26.8%. We also demonstrate the generalizability of Mango using both open-source and closed-source models as backbones. Our data and code are open-source and available at this https URL.

**arXiv ID:** 2604.18779
</details>

<details>
<summary><strong>MATRAG: Multi-Agent Transparent Retrieval-Augmented Generation for Explainable Recommendations</strong> - Sushant Mehta - [[pdf]](https://arxiv.org/pdf/2604.20848)</summary>

**Abstract:** Large Language Model (LLM)-based recommendation systems have demonstrated remarkable capabilities in understanding user preferences and generating personalized suggestions. However, existing approaches face critical challenges in transparency, knowledge grounding, and the ability to provide coherent explanations that foster user trust. We introduce MATRAG (Multi-Agent Transparent Retrieval-Augmented Generation), a novel framework that combined multi-agent collaboration with knowledge graph-augmented retrieval to deliver explainable recommendations. MATRAG employs four specialized agents: a User Modeling Agent that constructs dynamic preference profiles, an Item Analysis Agent that extracts semantic features from knowledge graphs, a Reasoning Agent that synthesizes collaborative and content-based signals, and an Explanation Agent that generates natural language justifications grounded in retrieved knowledge. Our framework incorporates a transparency scoring mechanism that quantifies explanation faithfulness and relevance. Extensive experiments on three benchmark datasets (Amazon Reviews, MovieLens-1M, and Yelp) demonstrate that MATRAG achieves state-of-the-art performance, improving recommendation accuracy by 12.7\% (Hit Rate) and 15.3\% (NDCG) over leading baselines, while human evaluation confirms that 87.4\% of generated explanations are rated as helpful and trustworthy by domain experts. Our work establishes new benchmarks for transparent, agentic recommendation systems and provides actionable insights for deploying LLM-based recommenders in production environments.

**arXiv ID:** 2604.20848
</details>

<details>
<summary><strong>KompeteAI: Accelerated Autonomous Multi-Agent System for End-to-End Pipeline Generation for Machine Learning Problems</strong> - Stepan Kulibaba, Artem Dzhalilov, Roman Pakhomov, Oleg Svidchenko, Alexander Gasnikov, Aleksei Shpilman - [[pdf]](https://arxiv.org/pdf/2508.10177)</summary>

**Abstract:** Recent Large Language Model (LLM)-based AutoML systems demonstrate impressive capabilities but face significant limitations such as constrained exploration strategies and a severe execution bottleneck. Exploration is hindered by one-shot methods lacking diversity and Monte Carlo Tree Search (MCTS) approaches that fail to recombine strong partial solutions. The execution bottleneck arises from lengthy code validation cycles that stifle iterative refinement. To overcome these challenges, we introduce KompeteAI, a novel AutoML framework with dynamic solution space exploration. Unlike previous MCTS methods that treat ideas in isolation, KompeteAI introduces a merging stage that composes top candidates. We further expand the hypothesis space by integrating Retrieval-Augmented Generation (RAG), sourcing ideas from Kaggle notebooks and arXiv papers to incorporate real-world strategies. KompeteAI also addresses the execution bottleneck via a predictive scoring model and an accelerated debugging method, assessing solution potential using early stage metrics to avoid costly full-code execution. This approach accelerates pipeline evaluation 6.9 times. KompeteAI outperforms leading methods (e.g., RD-agent, AIDE, and Ml-Master) by an average of 3\% on the primary AutoML benchmark, MLE-Bench. Additionally, we propose Kompete-bench to address limitations in MLE-Bench, where KompeteAI also achieves state-of-the-art results

**arXiv ID:** 2508.10177
</details>

<details>
<summary><strong>PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation</strong> - Jiho Choi, Seojeong Park, Seongjong Song, Hyunjung Shim - [[pdf]](https://arxiv.org/pdf/2508.21720)</summary>

**Abstract:** Automating scientific poster generation requires hierarchical document understanding and coherent content-layout planning. Existing methods often rely on flat summarization or optimize content and layout separately. As a result, they often suffer from information loss, weak logical flow, and poor visual balance. We present PosterForest, a training-free framework for scientific poster generation. Our method introduces the Poster Tree, a structured intermediate representation that captures document hierarchy and visual-textual semantics across multiple levels. Building on this representation, content and layout agents perform hierarchical reasoning and recursive refinement, progressively optimizing the poster from global organization to local composition. This joint optimization improves semantic coherence, logical flow, and visual harmony. Experiments show that PosterForest outperforms prior methods in both automatic and human evaluations, without additional training or domain-specific supervision.

**arXiv ID:** 2508.21720
</details>

<details>
<summary><strong>Empirical Comparison of Agent Communication Protocols for Task Orchestration</strong> - Ivan Dobrovolskyi - [[pdf]](https://arxiv.org/pdf/2603.22823)</summary>

**Abstract:** Context. The problem of comparative evaluation of communication protocols for task orchestration by large language model (LLM) agents is considered. The object of study is the process of interaction between LLM agents and external tools, as well as between autonomous LLM agents, during task orchestration. Objective. The goal of this work is to develop a systematic pilot benchmark comparing tool integration, multi-agent dele-gation, and hybrid architectures for standardized queries at three levels of complexity, and to quantify the advantages and disadvantages in terms of response time, context window consumption, cost, error recovery, and implementation complexity.

**arXiv ID:** 2603.22823
</details>

<details>
<summary><strong>Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents</strong> - Huangwei Chen, Wu Li, Junhao Jia, Yining Chen, Xiaotao Pang, YaLong Chen, Gonghui Li, Haishuai Wang, Jiajun Bu, Lei Wu - [[pdf]](https://arxiv.org/pdf/2604.08927)</summary>

**Abstract:** The initial outpatient consultation is critical for clinical decision-making, yet it is often conducted by a single physician under time pressure, making it prone to cognitive biases and incomplete evidence capture. Although the Multi-Disciplinary Team (MDT) reduces these risks, they are costly and difficult to scale to real-time intake. We propose Aegle, a synchronous virtual MDT framework that brings MDT-level reasoning to outpatient consultations via a graph-based multi-agent architecture. Aegle formalizes the consultation state using a structured SOAP representation, separating evidence collection from diagnostic reasoning to improve traceability and bias control. An orchestrator dynamically activates specialist agents, which perform decoupled parallel reasoning and are subsequently integrated by an aggregator into a coherent clinical note. Experiments on ClinicalBench and a real-world RAPID-IPN dataset across 24 departments and 53 metrics show that Aegle consistently outperforms state-of-the-art proprietary and open-source models in documentation quality and consultation capability, while also improving final diagnosis accuracy. Our code is available at this https URL.

**arXiv ID:** 2604.08927
</details>

<details>
<summary><strong>STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning</strong> - Juntong Ni, Shiyu Wang, Qi He, Ming Jin, Wei Jin - [[pdf]](https://arxiv.org/pdf/2601.03248)</summary>

**Abstract:** Spatio-temporal reasoning in time series involves the explicit synthesis of temporal dynamics, spatial dependencies, and textual context. This capability is vital for high-stakes decision-making in systems such as traffic networks, power grids, and disease propagation. However, the field remains underdeveloped because most existing works prioritize predictive accuracy over reasoning. To address the gap, we introduce ST-Bench, a benchmark consisting of four core tasks, including etiological reasoning, entity identification, correlation reasoning, and in-context forecasting, developed via a network SDE-based multi-agent data synthesis pipeline. We then propose STReasoner, which empowers LLM to integrate time series, graph structure, and text for explicit reasoning. To promote spatially grounded logic, we introduce S-GRPO, a reinforcement learning algorithm that rewards performance gains specifically attributable to spatial information. Experiments show that STReasoner achieves average accuracy gains between 17% and 135% at only 0.004X the cost of proprietary models and generalizes robustly to real-world data.

**arXiv ID:** 2601.03248
</details>

<details>
<summary><strong>Improving Clinical Diagnosis with Counterfactual Multi-Agent Reasoning</strong> - Zhiwen You, Xi Chen, Aniket Vashishtha, Simo Du, Gabriel Erion-Barner, Hongyuan Mei, Hao Peng, Yue Guo - [[pdf]](https://arxiv.org/pdf/2603.27820)</summary>

**Abstract:** Clinical diagnosis is a complex reasoning process in which clinicians gather evidence, form hypotheses, and test them against alternative explanations. In medical training, this reasoning is explicitly developed through counterfactual questioning--e.g., asking how a diagnosis would change if a key symptom were absent or altered--to strengthen differential diagnosis skills. As large language model (LLM)-based systems are increasingly used for diagnostic support, ensuring the interpretability of their recommendations becomes critical. However, most existing LLM-based diagnostic agents reason over fixed clinical evidence without explicitly testing how individual findings support or weaken competing diagnoses. In this work, we propose a counterfactual multi-agent diagnostic framework inspired by clinician training that makes hypothesis testing explicit and evidence-grounded. Our framework introduces counterfactual case editing to modify clinical findings and evaluate how these changes affect competing diagnoses. We further define the Counterfactual Probability Gap, a method that quantifies how strongly individual findings support a diagnosis by measuring confidence shifts under these edits. These counterfactual signals guide multi-round specialist discussions, enabling agents to challenge unsupported hypotheses, refine differential diagnoses, and produce more interpretable reasoning trajectories. Across three diagnostic benchmarks and seven LLMs, our method consistently improves diagnostic accuracy over prompting and prior multi-agent baselines, with the largest gains observed in complex and ambiguous cases. Human evaluation further indicates that our framework produces more clinically useful, reliable, and coherent reasoning. These results suggest that incorporating counterfactual evidence verification is an important step toward building reliable AI systems for clinical decision support.

**arXiv ID:** 2603.27820
</details>

<details>
<summary><strong>Strategic Heterogeneous Multi-Agent Architecture for Cost-Effective Code Vulnerability Detection</strong> - Zhaohui Geoffrey Wang - [[pdf]](https://arxiv.org/pdf/2604.21282)</summary>

**Abstract:** Automated code vulnerability detection is critical for software security, yet existing approaches face a fundamental trade-off between detection accuracy and computational cost. We propose a heterogeneous multi-agent architecture inspired by game-theoretic principles, combining cloud-based LLM experts with a local lightweight verifier. Our "3+1" architecture deploys three cloud-based expert agents (DeepSeek-V3) that analyze code from complementary perspectives - code structure, security patterns, and debugging logic - in parallel, while a local verifier (Qwen3-8B) performs adversarial validation at zero marginal cost.
We formalize this design through a two-layer game framework: (1) a cooperative game among experts capturing super-additive value from diverse perspectives, and (2) an adversarial verification game modeling quality assurance incentives.
Experiments on 262 real samples from the NIST Juliet Test Suite across 14 CWE types, with balanced vulnerable and benign classes, demonstrate that our approach achieves a 77.2% F1 score with 62.9% precision and 100% recall at $0.002 per sample - outperforming both a single-expert LLM baseline (F1 71.4%) and Cppcheck static analysis (MCC 0). The adversarial verifier significantly improves precision (+10.3 percentage points, p < 1e-6, McNemar's test) by filtering false positives, while parallel execution achieves a 3.0x speedup.
Our work demonstrates that game-theoretic design principles can guide effective heterogeneous multi-agent architectures for cost-sensitive software engineering tasks.

**arXiv ID:** 2604.21282
</details>

</details>

<details open>
<summary><h2>Other Agent Research (4 papers)</h2></summary>

<details>
<summary><strong>From Research Question to Scientific Workflow: Leveraging Agentic AI for Science Automation</strong> - Bartosz Balis, Michal Orzechowski, Piotr Kica, Michal Dygas, Michal Kuszewski - [[pdf]](https://arxiv.org/pdf/2604.21910)</summary>

**Abstract:** Scientific workflow systems automate execution -- scheduling, fault tolerance, resource management -- but not the semantic translation that precedes it. Scientists still manually convert research questions into workflow specifications, a task requiring both domain knowledge and infrastructure expertise. We propose an agentic architecture that closes this gap through three layers: an LLM interprets natural language into structured intents (semantic layer); validated generators produce reproducible workflow DAGs (deterministic layer); and domain experts author ``Skills'': markdown documents encoding vocabulary mappings, parameter constraints, and optimization strategies (knowledge layer). This decomposition confines LLM non-determinism to intent extraction: identical intents always yield identical workflows. We implement and evaluate the architecture on the 1000 Genomes population genetics workflow and Hyperflow WMS running on Kubernetes. In an ablation study on 150 queries, Skills raise full-match intent accuracy from 44% to 83%; skill-driven deferred workflow generation reduces data transfer by 92\%; and the end-to-end pipeline completes queries on Kubernetes with LLM overhead below 15 seconds and cost under $0.001 per query.

**arXiv ID:** 2604.21910
</details>

<details>
<summary><strong>Role of diversity in team performance: the case of missing expertise, an agent based simulation</strong> - Tamás Kiss - [[pdf]](https://arxiv.org/pdf/2604.21328)</summary>

**Abstract:** Theory and empirical research on management teams' influence on firm performance have witnessed continuous development, and by now incorporate numerous details. Classic, experiment-based studies examining social systems collect vast amount of data, but often times investigate only the first one or two modes of the distribution of measured variables, and experience difficulty in analyzing the effect of context. For example, in functional diversity research, management teams are described by measures incorporating complex distributions of capabilities of individual managers and teams of managers. To investigate the effect of hidden distributions, and the effect of functional diversity composition on team communication and performance, we developed an agent-based model, and conducted a series of simulation experiments. Modeling results show that depending on the context, such as communication scheme among interacting agents, or their functional composition, intrapersonal functional diversity (IFD), and dominant function diversity (DFD) might enhance or reduce performance and communication among agents. Furthermore, simulation results also suggest that a third measure is required alongside IFD and DFD capturing the aggregate expertise of the team to comprehensively account for empirical findings.

**arXiv ID:** 2604.21328
</details>

<details>
<summary><strong>Towards a Systematic Risk Assessment of Deep Neural Network Limitations in Autonomous Driving Perception</strong> - Svetlana Pavlitska, Christopher Gerking, J. Marius Zöllner - [[pdf]](https://arxiv.org/pdf/2604.20895)</summary>

**Abstract:** Safety and security are essential for the admission and acceptance of automated and autonomous vehicles. Deep neural networks (DNNs) are widely used for perception and further components of the autonomous driving (AD) stack. However, they possess several limitations, including lack of generalization, efficiency, explainability, plausibility, and robustness. These insufficiencies can pose significant risks to autonomous driving systems. However, hazards, threats, and risks associated with DNN limitations in this domain have not been systematically studied so far. In this work, we propose a joint workflow for risk assessment combining the hazard analysis and risk assessment (HARA) following ISO 26262 and threat analysis and risk assessment (TARA) following the ISO/SAE 21434 to identify and analyze risks arising from inherent DNN limitations in AD perception.

**arXiv ID:** 2604.20895
</details>

<details>
<summary><strong>The Privacy Guardian Agent: Towards Trustworthy AI Privacy Agents</strong> - Vincent Freiberger - [[pdf]](https://arxiv.org/pdf/2604.21455)</summary>

**Abstract:** The current "notice and consent" paradigm is broken: consent dialogues are often manipulative, and users cannot realistically read or understand every privacy policy. While recent LLM-based tools empower users seeking active control, many with limited time or motivation prefer full automation. However, fully autonomous solutions risk hallucinations and opaque decisions, undermining trust. I propose a middle ground - a Privacy Guardian Agent that automates routine consent choices using user profiles and contextual awareness while recognizing uncertainty. It escalates unclear or high-risk cases to the user, maintaining a human-in-the-loop only when necessary. To ensure agency and transparency, the agent's reasoning on its autonomous decisions is reviewable, allowing for user recourse. For problematic cases, even with minimal consent, it alerts the user and suggests switching to an alternative site. This approach aims to reduce consent fatigue while preserving trust and meaningful user autonomy.

**arXiv ID:** 2604.21455
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>A Deployable Embodied Vision-Language Navigation System with Hierarchical Cognition and Context-Aware Exploration</strong> - Kuan Xu, Ruimeng Liu, Yizhuo Yang, Denan Liang, Tongxing Jin, Shenghai Yuan, Chen Wang, Lihua Xie - [[pdf]](https://arxiv.org/pdf/2604.21363)</summary>

**Abstract:** Bridging the gap between embodied intelligence and embedded deployment remains a key challenge in intelligent robotic systems, where perception, reasoning, and planning must operate under strict constraints on computation, memory, energy, and real-time execution. In vision-language navigation (VLN), existing approaches often face a fundamental trade-off between strong reasoning capabilities and efficient deployment on real-world platforms. In this paper, we present a deployable embodied VLN system that achieves both high efficiency and robust high-level reasoning on real-world robotic platforms. To achieve this, we decouple the system into three asynchronous modules: a real-time perception module for continuous environment sensing, a memory integration module for spatial-semantic aggregation, and a reasoning module for high-level decision making. We incrementally construct a cognitive memory graph to encode scene information, which is further decomposed into subgraphs to enable reasoning with a vision-language model (VLM). To further improve navigation efficiency and accuracy, we also leverage the cognitive memory graph to formulate the exploration problem as a context-aware Weighted Traveling Repairman Problem (WTRP), which minimizes the weighted waiting time of viewpoints. Extensive experiments in both simulation and real-world robotic platforms demonstrate improved navigation success and efficiency over existing VLN approaches, while maintaining real-time performance on resource-constrained hardware.

**arXiv ID:** 2604.21363
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (18 papers)</h2></summary>

<details>
<summary><strong>A Systematic Review and Taxonomy of Reinforcement Learning-Model Predictive Control Integration for Linear Systems</strong> - Mohsen Jalaeian Farimani, Roya Khalili Amirabadi, Davoud Nikkhouy, Malihe Abdolbaghi, Mahshad Rastegarmoghaddam, Shima Samadzadeh - [[pdf]](https://arxiv.org/pdf/2604.21030)</summary>

**Abstract:** The integration of Model Predictive Control (MPC) and Reinforcement Learning (RL) has emerged as a promising paradigm for constrained decision-making and adaptive control. MPC offers structured optimization, explicit constraint handling, and established stability tools, whereas RL provides data-driven adaptation and performance improvement in the presence of uncertainty and model mismatch. Despite the rapid growth of research on RL--MPC integration, the literature remains fragmented, particularly for control architectures built on linear or linearized predictive models. This paper presents a comprehensive Systematic Literature Review (SLR) of RL--MPC integrations for linear and linearized systems, covering peer-reviewed and formally indexed studies published until 2025. The reviewed studies are organized through a multi-dimensional taxonomy covering RL functional roles, RL algorithm classes, MPC formulations, cost-function structures, and application domains. In addition, a cross-dimensional synthesis is conducted to identify recurring design patterns and reported associations among these dimensions within the reviewed corpus. The review highlights methodological trends, commonly adopted integration strategies, and recurring practical challenges, including computational burden, sample efficiency, robustness, and closed-loop guarantees. The resulting synthesis provides a structured reference for researchers and practitioners seeking to design or analyze RL--MPC architectures based on linear or linearized predictive control formulations.

**arXiv ID:** 2604.21030
</details>

<details>
<summary><strong>Measure Twice, Click Once: Co-evolving Proposer and Visual Critic via Reinforcement Learning for GUI Grounding</strong> - Wenkai Wang, Xiyun Li, Hongcan Guo, Wenhao Yu, Tianqing Fang, Haitao Mi, Dong Yu, Shengyu Zhang - [[pdf]](https://arxiv.org/pdf/2604.21268)</summary>

**Abstract:** Graphical User Interface (GUI) grounding requires mapping natural language instructions to precise pixel coordinates. However, due to visually homogeneous elements and dense layouts, models typically grasp semantic intent yet struggle with achieving precise localization. While scaling sampling attempts (Pass@k) reveals potential gains, static self-consistency strategies derived from geometric clustering often yield limited improvements, as the model's predictions tend to be spatially dispersed. In this paper, we propose replacing static consistency strategies with a learnable selection mechanism that selects the optimal target by critiquing its own proposals rendered on the screenshot. Given the significant disparity between the model's grounding and critiquing capabilities, we propose a co-evolving Propose-then-Critic framework. To jointly optimize these, we introduce a maturity-aware adaptive co-evolutionary reinforcement learning paradigm. This approach dynamically balances the training objectives of proposer and critic, where the diversity of the proposer's outputs enhances critic robustness, while the critic's maturing discrimination capability conversely unlocks the proposer's potential for extensive spatial exploration, fostering the mutual reinforcement and co-evolution of both capabilities, thereby ensuring generalizability to adapt to diverse and complex interface layouts. Extensive experiments over 6 benchmarks show that our method significantly enhances both grounding accuracy and critic reliability.

**arXiv ID:** 2604.21268
</details>

<details>
<summary><strong>Understanding and Mitigating Spurious Signal Amplification in Test-Time Reinforcement Learning for Math Reasoning</strong> - Yongcan Yu, Lingxiao He, Jian Liang, Kuangpu Guo, Meng Wang, Qianlong Xie, Xingxing Wang, Ran He - [[pdf]](https://arxiv.org/pdf/2604.21327)</summary>

**Abstract:** Test-time reinforcement learning (TTRL) always adapts models at inference time via pseudo-labeling, leaving it vulnerable to spurious optimization signals from label noise. Through an empirical study, we observe that responses with medium consistency form an ambiguity region and constitute the primary source of reward noise. Crucially, we find that such spurious signals can be even amplified through group-relative advantage estimation. Motivated by these findings, we propose a unified framework, Debiased and Denoised test-time Reinforcement Learning (DDRL), to mitigate spurious signals. Concretely, DDRL first applies a frequency-based sampling strategy to exclude ambiguous samples while maintaining a balanced set of positive and negative examples. It then adopts a debiased advantage estimation with fixed advantages, removing the bias introduced by group-relative policy optimization. Finally, DDRL incorporates a consensus-based off-policy refinement stage, which leverages the rejection-sampled dataset to enable efficient and stable model updates. Experiments on three large language models across multiple mathematical reasoning benchmarks demonstrate that DDRL consistently outperforms existing TTRL baselines. The code will soon be released at this https URL.

**arXiv ID:** 2604.21327
</details>

<details>
<summary><strong>Dynamical Priors as a Training Objective in Reinforcement Learning</strong> - Sukesh Subaharan - [[pdf]](https://arxiv.org/pdf/2604.21464)</summary>

**Abstract:** Standard reinforcement learning (RL) optimizes policies for reward but imposes few constraints on how decisions evolve over time. As a result, policies may achieve high performance while exhibiting temporally incoherent behavior such as abrupt confidence shifts, oscillations, or degenerate inactivity. We introduce Dynamical Prior Reinforcement Learning (DP-RL), a training framework that augments policy gradient learning with an auxiliary loss derived from external state dynamics that implement evidence accumulation and hysteresis. Without modifying the reward, environment, or policy architecture, this prior shapes the temporal evolution of action probabilities during learning. Across three minimal environments, we show that dynamical priors systematically alter decision trajectories in task-dependent ways, promoting temporally structured behavior that cannot be explained by generic smoothing. These results demonstrate that training objectives alone can control the temporal geometry of decision-making in RL agents.

**arXiv ID:** 2604.21464
</details>

<details>
<summary><strong>Task-specific Subnetwork Discovery in Reinforcement Learning for Autonomous Underwater Navigation</strong> - Yi-Ling Liu, Melvin Laux, Mariela De Lucas Alvarez, Frank Kirchner, Rebecca Adam - [[pdf]](https://arxiv.org/pdf/2604.21640)</summary>

**Abstract:** Autonomous underwater vehicles are required to perform multiple tasks adaptively and in an explainable manner under dynamic, uncertain conditions and limited sensing, challenges that classical controllers struggle to address. This demands robust, generalizable, and inherently interpretable control policies for reliable long-term monitoring. Reinforcement learning, particularly multi-task RL, overcomes these limitations by leveraging shared representations to enable efficient adaptation across tasks and environments. However, while such policies show promising results in simulation and controlled experiments, they yet remain opaque and offer limited insight into the agent's internal decision-making, creating gaps in transparency, trust, and safety that hinder real-world deployment. The internal policy structure and task-specific specialization remain poorly understood. To address these gaps, we analyze the internal structure of a pretrained multi-task reinforcement learning network in the HoloOcean simulator for underwater navigation by identifying and comparing task-specific subnetworks responsible for navigating toward different species. We find that in a contextual multi-task reinforcement learning setting with related tasks, the network uses only about 1.5% of its weights to differentiate between tasks. Of these, approximately 85% connect the context-variable nodes in the input layer to the next hidden layer, highlighting the importance of context variables in such settings. Our approach provides insights into shared and specialized network components, useful for efficient model editing, transfer learning, and continual learning for underwater monitoring through a contextual multi-task reinforcement learning method.

**arXiv ID:** 2604.21640
</details>

<details>
<summary><strong>Agentic AI-assisted coding offers a unique opportunity to instill epistemic grounding during software development</strong> - Magnus Palmblad, Jared M. Ragland, Benjamin A. Neely - [[pdf]](https://arxiv.org/pdf/2604.21744)</summary>

**Abstract:** The capabilities of AI-assisted coding are progressing at breakneck speed. Chat-based vibe coding has evolved into fully fledged AI-assisted, agentic software development using agent scaffolds where the human developer creates a plan that agentic AIs implement. One current trend is utilizing documents beyond this plan document, such as project and method-scoped documents. Here we propose this http URL, a community-governed, field-scoped epistemic grounding document, using mass spectrometry-based proteomics as an example. This explicit field-scoped grounding document encodes Hard Constraints (non-negotiable validity invariants empirically required for scientific correctness) and Convention Parameters (community-agreed defaults) that override all other contexts to enforce validity, regardless of what the user prompts. In practice, this will empower a non-domain expert to generate code, tools, and software that have best practices baked in at the ground level, providing confidence to the software developer but also to those reviewing or using the final product. Undoubtedly it is easier to have agentic AIs adhere to guidelines than humans, and this opportunity allows for organizations to develop epistemic grounding documents in such a way as to keep domain experts in the loop in a future of democratized generation of bespoke software solutions.

**arXiv ID:** 2604.21744
</details>

<details>
<summary><strong>Learning Reasoning Reward Models from Expert Demonstration via Inverse Reinforcement Learning</strong> - Claudio Fanconi, Nicolás Astorga, Mihaela van der Schaar - [[pdf]](https://arxiv.org/pdf/2510.01857)</summary>

**Abstract:** Current approaches to improving reasoning in large language models (LLMs) primarily rely on either supervised fine-tuning (SFT) over expert traces or reinforcement learning (RL) with outcome-level rewards. However, SFT is fundamentally imitative, while outcome-based RL assumes access to a well-specified verifier. To address this gap, we propose an adversarial inverse reinforcement learning (AIRL) framework that learns reasoning rewards directly from expert demonstrations. We evaluate this framework across reward granularities (sparse, interval, and dense). Granularity controls the resolution of credit assignment: sparse rewards emphasise global trajectory quality and training stability, while denser rewards provide higher-resolution step-level supervision for error localisation but are harder to optimise stably. We show that the learned reasoning rewards are useful in three complementary ways. First, as a training signal, they often outperform SFT, with the best variant improving over SFT on medical reasoning (MedReason), mathematics (GSM8K), and challenging scientific question-answering (MMLU-Pro). Second, as an inference-time reranker, they gain up to 17.4 percentage points under a fixed sampling budget. Third, the learned reward transfers across tasks and backbones, suggesting that part of the signal is reusable beyond a single domain or model, and that finer-grained rewards identify the first step at which a trajectory deviates from a correct path. This supports the diagnosis of reasoning failures and the improvement of test-time selection. Together, these results show that AIRL can recover a reusable intermediate reasoning step from demonstrations alone, bridging the gap between pure imitation and reward-driven optimisation for LLM reasoning.

**arXiv ID:** 2510.01857
</details>

<details>
<summary><strong>Multimodal Bayesian Network for Robust Assessment of Casualties in Autonomous Triage</strong> - Szymon Rusiecki, Cecilia G. Morales, Kimberly Elenberg, Leonard Weiss, Artur Dubrawski - [[pdf]](https://arxiv.org/pdf/2512.18908)</summary>

**Abstract:** Mass Casualty Incidents can overwhelm emergency medical systems and resulting delays or errors in the assessment of casualties can lead to preventable deaths. We present a decision support framework that fuses outputs from multiple computer vision models, estimating signs of severe hemorrhage, respiratory distress, physical alertness, or visible trauma, into a Bayesian network constructed entirely from expert-defined rules. Unlike traditional data-driven models, our approach does not require training data, supports inference with incomplete information, and is robust to noisy or uncertain observations. We report performance for two missions involving 11 and 9 casualties, respectively, where our Bayesian network model substantially outperformed vision-only baselines during evaluation of our system in the DARPA Triage Challenge (DTC) field scenarios. The accuracy of physiological assessment improved from 15% to 42% in the first scenario and from 19% to 46% in the second, representing nearly threefold increase in performance. More importantly, overall triage accuracy increased from 14% to 53% in all patients, while the diagnostic coverage of the system expanded from 31% to 95% of the cases requiring assessment. These results demonstrate that expert-knowledge-guided probabilistic reasoning can significantly enhance automated triage systems, offering a promising approach to supporting emergency responders in MCIs. This approach enabled Team Chiron to achieve 4th place out of 11 teams during the 1st physical round of the DTC.

**arXiv ID:** 2512.18908
</details>

<details>
<summary><strong>Reinforcement Learning with Foundation Priors: Let the Embodied Agent Efficiently Learn on Its Own</strong> - Weirui Ye, Yunsheng Zhang, Haoyang Weng, Xianfan Gu, Shengjie Wang, Tong Zhang, Mengchen Wang, Pieter Abbeel, Yang Gao - [[pdf]](https://arxiv.org/pdf/2310.02635)</summary>

**Abstract:** Reinforcement learning (RL) is a promising approach for solving robotic manipulation tasks. However, it is challenging to apply the RL algorithms directly in the real world. For one thing, RL is data-intensive and typically requires millions of interactions with environments, which are impractical in real scenarios. For another, it is necessary to make heavy engineering efforts to design reward functions manually. To address these issues, we leverage foundation models in this paper. We propose Reinforcement Learning with Foundation Priors (RLFP) to utilize guidance and feedback from policy, value, and success-reward foundation models. Within this framework, we introduce the Foundation-guided Actor-Critic (FAC) algorithm, which enables embodied agents to explore more efficiently with automatic reward functions. The benefits of our framework are threefold: (1) \textit{sample efficient}; (2) \textit{minimal and effective reward engineering}; (3) \textit{agnostic to foundation model forms and robust to noisy priors}. Our method achieves remarkable performances in various manipulation tasks on both real robots and in simulation. Across 5 dexterous tasks with real robots, FAC achieves an average success rate of 86\% after one hour of real-time learning. Across 8 tasks in the simulated Meta-world, FAC achieves 100\% success rates in 7/8 tasks under less than 100k frames (about 1-hour training), outperforming baseline methods with manual-designed rewards in 1M frames. We believe the RLFP framework can enable future robots to explore and learn autonomously in the physical world for more tasks. Visualizations and code are available at this https URL.

**arXiv ID:** 2310.02635
</details>

<details>
<summary><strong>AgentLens: Adaptive Visual Modalities for Human-Agent Interaction in Mobile GUI Agents</strong> - Jeonghyeon Kim, Byeongjun Joung, Junwon Lee, Joohyung Lee, Taehoon Min, Sunjae Lee - [[pdf]](https://arxiv.org/pdf/2604.20279)</summary>

**Abstract:** Mobile GUI agents can automate smartphone tasks by interacting directly with app interfaces, but how they should communicate with users during execution remains underexplored. Existing systems rely on two extremes: foreground execution, which maximizes transparency but prevents multitasking, and background execution, which supports multitasking but provides little visual awareness. Through iterative formative studies, we found that users prefer a hybrid model with just-in-time visual interaction, but the most effective visualization modality depends on the task. Motivated by this, we present AgentLens, a mobile GUI agent that adaptively uses three visual modalities during human-agent interaction: Full UI, Partial UI, and GenUI. AgentLens extends a standard mobile agent with adaptive communication actions and uses Virtual Display to enable background execution with selective visual overlays. In a controlled study with 21 participants, AgentLens was preferred by 85.7% of participants and achieved the highest usability (1.94 Overall PSSUQ) and adoption-intent (6.43/7).

**arXiv ID:** 2604.20279
</details>

<details>
<summary><strong>AgenticQwen: Training Small Agentic Language Models with Dual Data Flywheels for Industrial-Scale Tool Use</strong> - Yuanjie Lyu, Chengyu Wang, Haonan Zheng, Yuanhao Yue, Junbing Yan, Ming Wang, Jun Huang - [[pdf]](https://arxiv.org/pdf/2604.21590)</summary>

**Abstract:** Modern industrial applications increasingly demand language models that act as agents, capable of multi-step reasoning and tool use in real-world settings. These tasks are typically performed under strict cost and latency constraints, making small agentic models highly desirable. In this paper, we introduce the AgenticQwen family of models, trained via multi-round reinforcement learning (RL) on synthetic data and a limited amount of open-source data. Our training framework combines reasoning RL and agentic RL with dual data flywheels that automatically generate increasingly challenging tasks. The reasoning flywheel increases task difficulty by learning from errors, while the agentic flywheel expands linear workflows into multi-branch behavior trees that better reflect the decision complexity of real-world applications. We validate AgenticQwen on public benchmarks and in an industrial agent system. The models achieve strong performance on multiple agentic benchmarks, and in our industrial agent system, close the gap with much larger models on search and data analysis tasks. Model checkpoints and part of the synthetic data: this https URL. Data synthesis and RL training code: this https URL. The data synthesis pipeline is also integrated into EasyDistill: this https URL.

**arXiv ID:** 2604.21590
</details>

<details>
<summary><strong>Spec-o3: A Tool-Augmented Vision-Language Agent for Rare Celestial Object Candidate Vetting via Automated Spectral Inspection</strong> - Minghui Jia, Qichao Zhang, Ali Luo, Linjing Li, Shuo Ye, Hailing Lu, Wen Hou, Dongbin Zhao - [[pdf]](https://arxiv.org/pdf/2601.06498)</summary>

**Abstract:** Due to the limited generalization and interpretability of deep learning classifiers, The final vetting of rare celestial object candidates still relies on expert visual inspection--a manually intensive process. In this process, astronomers leverage specialized tools to analyze spectra and construct reliable catalogs. However, this practice has become the primary bottleneck, as it is fundamentally incapable of scaling with the data deluge from modern spectroscopic surveys. To bridge this gap, we propose Spec-o3, a tool-augmented vision-language agent that performs astronomer-aligned spectral inspection via interleaved multimodal chain-of-thought reasoning. Spec-o3 is trained with a two-stage post-training recipe: cold-start supervised fine-tuning on expert inspection trajectories followed by outcome-based reinforcement learning on rare-type verification tasks. Evaluated on five rare-object identification tasks from LAMOST, Spec-o3 establishes a new State-of-the-Art, boosting the macro-F1 score from 28.3 to 76.5 with a 7B parameter base model and outperforming both proprietary VLMs and specialized deep models. Crucially, the agent demonstrates strong generalization to unseen inspection tasks across survey shifts (from LAMOST to SDSS/DESI). Expert evaluations confirm that its reasoning traces are coherent and physically consistent, supporting transparent and trustworthy decision-making. Code, data, and models are available at this https URL.

**arXiv ID:** 2601.06498
</details>

<details>
<summary><strong>Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning</strong> - Yue Guo, Fanfu Wang, Jianwei Lv, Xincheng Shi, Yuchen Li, Youya Wang, Yunsheng Zeng, Yujing Liu, Yunhao Qiao, Gen Li, Junfeng Wang, Bo Yuan - [[pdf]](https://arxiv.org/pdf/2601.13690)</summary>

**Abstract:** Clinical Decision Support Systems (CDSSs) provide reasoning and inquiry guidance for physicians, yet they face notable challenges, including high maintenance costs and low generalization capability. Recently, Large Language Models (LLMs) have been widely adopted in healthcare due to their extensive knowledge reserves, retrieval, and communication capabilities. While LLMs show promise and excel at medical benchmarks, their diagnostic reasoning and inquiry skills are constrained. To mitigate this issue, we propose (1) Clinical Diagnostic Reasoning Data (CDRD) structure to capture abstract clinical reasoning logic, and a pipeline for its construction, and (2) the Dr. Assistant, a clinical diagnostic model equipped with clinical reasoning and inquiry skills. Its training involves a two-stage process: SFT, followed by RL with a tailored reward function. We also introduce a benchmark to evaluate both diagnostic reasoning and inquiry. Our experiments demonstrate that the Dr. Assistant outperforms open-source models and achieves competitive performance to closed-source models, providing an effective solution for clinical diagnostic inquiry guidance. Project information can be found at: this https URL .

**arXiv ID:** 2601.13690
</details>

<details>
<summary><strong>CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning</strong> - Zhenpeng Su, Leiyu Pan, Minxuan Lv, Yuntao Li, Wenping Hu, Fuzheng Zhang, Kun Gai, Guorui Zhou - [[pdf]](https://arxiv.org/pdf/2509.20712)</summary>

**Abstract:** Reinforcement learning (RL) has become a powerful paradigm for optimizing large language models (LLMs) to handle complex reasoning tasks. A core challenge in this process lies in managing policy entropy, which reflects the balance between exploration and exploitation during training. Existing methods, such as proximal policy optimization (PPO) and its variants, discard valuable gradient signals from low-probability tokens due to the clipping mechanism. We systematically analyze the entropy dynamics and reveal that these clipped tokens play a critical yet overlooked role in regulating entropy evolution. We propose \textbf{C}oordinating \textbf{E}ntropy via \textbf{G}radient-\textbf{P}reserving \textbf{P}olicy \textbf{O}ptimization (CE-GPPO), a novel algorithm that reintroduces gradients from clipped tokens in native PPO in a gentle and bounded manner. By controlling the magnitude of gradients from tokens outside the clipping interval, CE-GPPO is able to achieve an exploration-exploitation trade-off. We provide theoretical justification and empirical evidence showing that CE-GPPO effectively mitigates entropy instability. Extensive experiments on mathematical reasoning benchmarks show that CE-GPPO consistently outperforms strong baselines across different model scales.

**arXiv ID:** 2509.20712
</details>

<details>
<summary><strong>Entropy Ratio Clipping as a Soft Global Constraint for Stable Reinforcement Learning</strong> - Zhenpeng Su, Leiyu Pan, Minxuan Lv, Tiehua Mei, Zijia Lin, Yuntao Li, Wenping Hu, Ruiming Tang, Kun Gai, Guorui Zhou - [[pdf]](https://arxiv.org/pdf/2512.05591)</summary>

**Abstract:** Large language model post-training relies on reinforcement learning to improve model capability and alignment quality. However, the off-policy training paradigm introduces distribution shift, which often pushes the policy beyond the trust region, leading to training instabilities manifested as fluctuations in policy entropy and unstable gradients. Although PPO-Clip mitigates this issue through importance clipping, it still overlooks the global distributional shift of actions. To address these challenges, we propose using the entropy ratio between the current and previous policies as a new global metric that effectively quantifies the relative change in policy exploration throughout updates. Building on this metric, we introduce an \textbf{Entropy Ratio Clipping} (ERC) mechanism that imposes bidirectional constraints on the entropy ratio. This stabilizes policy updates at the global distribution level and compensates for the inability of PPO-clip to regulate probability shifts of un-sampled actions. We integrate ERC into both DAPO and GPPO reinforcement learning algorithms. Experiments across multiple benchmarks show that ERC consistently improves performance.

**arXiv ID:** 2512.05591
</details>

<details>
<summary><strong>EARL-BO: Reinforcement Learning for Multi-Step Lookahead, High-Dimensional Bayesian Optimization</strong> - Mujin Cheon, Jay H. Lee, Dong-Yeun Koh, Calvin Tsay - [[pdf]](https://arxiv.org/pdf/2411.00171)</summary>

**Abstract:** To avoid myopic behavior, multi-step lookahead Bayesian optimization (BO) algorithms consider the sequential nature of BO and have demonstrated promising results in recent years. However, owing to the curse of dimensionality, most of these methods make significant approximations or suffer scalability issues. This paper presents a novel reinforcement learning (RL)-based framework for multi-step lookahead BO in high-dimensional black-box optimization problems. The proposed method enhances the scalability and decision-making quality of multi-step lookahead BO by efficiently solving the sequential dynamic program of the BO process in a near-optimal manner using RL. We first introduce an Attention-DeepSets encoder to represent the state of knowledge to the RL agent and subsequently propose a multi-task, fine-tuning procedure based on end-to-end (encoder-RL) on-policy learning. We evaluate the proposed method, EARL-BO (Encoder Augmented RL for BO), on synthetic benchmark functions and hyperparameter tuning problems, finding significantly improved performance compared to existing multi-step lookahead and high-dimensional BO methods.

**arXiv ID:** 2411.00171
</details>

<details>
<summary><strong>Tree Training: Accelerating Agentic LLMs Training via Shared Prefix Reuse</strong> - Jinghui Wang, Shaojie Wang, Yinghan Cui, Xuxing Chen, Chao Wang, Liang Huang, Can Tang, Xiaojiang Zhang, Junyi Peng, Li Wan, Haotian Zhang, Bin Chen - [[pdf]](https://arxiv.org/pdf/2511.00413)</summary>

**Abstract:** Agentic large language model (LLM) training often involves multi-turn interaction trajectories that branch into multiple execution paths due to concurrent tool use, think-mode, sub-agent, context management and other runtime designs. As a result, the tokens produced by a single task naturally form a tree-structured token trajectory with shared prefixes, rather than a linear sequence. Existing training pipelines linearize such trajectories and treat each branch independently, leading to substantial redundant computation in both forward and backward passes. We derive that averaging the loss over all branches independently is algebraically identical to a per-token weighted loss, where each token's weight equals the fraction of branches passing through it. The problem therefore reduces to computing the log-probability of every token in the prefix tree exactly once, with no repeated computation across shared prefixes: we propose DFS serialization of the tree, which visits every token exactly once, and adapt full-attention and SSM layers to ensure the resulting log-probabilities match independent per-branch calculation exactly. In practice, a single trajectory tree can be too large to fit in GPU memory; we therefore propose Redundancy-Free Tree Partitioning, which handles memory-constrained settings with zero redundant computation and peak memory bounded by a single root-to-leaf path. Together, these contributions form Tree Training, an efficient framework for training LLMs on tree-structured trajectories, achieving up to 6.2x end-to-end training speedup on dense and MoE models for both supervised fine-tuning and reinforcement learning.

**arXiv ID:** 2511.00413
</details>

<details>
<summary><strong>Self-Predictive Representation for Autonomous UAV Object-Goal Navigation</strong> - Angel Ayala, Donling Sui, Francisco Cruz, Mitchell Torok, Mohammad Deghat, Bruno J. T. Fernandes - [[pdf]](https://arxiv.org/pdf/2604.21130)</summary>

**Abstract:** Autonomous Unmanned Aerial Vehicles (UAVs) have revolutionized industries through their versatility with applications including aerial surveillance, search and rescue, agriculture, and delivery. Their autonomous capabilities offer unique advantages, such as operating in large open space environments. Reinforcement Learning (RL) empowers UAVs to learn intricate navigation policies, enabling them to optimize flight behavior autonomously. However, one of its main challenge is the inefficiency in using data sample to achieve a good policy. In object-goal navigation (OGN) settings, target recognition arises as an extra challenge. Most UAV-related approaches use relative or absolute coordinates to move from an initial position to a predefined location, rather than to find the target directly. This study addresses the data sample efficiency issue in solving a 3D OGN problem, in addition to, the formalization of the unknown target location setting as a Markov decision process. Experiments are conducted to analyze the interplay of different state representation learning (SRL) methods for perception with a model-free RL algorithm for planning in an autonomous navigation system. The main contribution of this study is the development of the perception module, featuring a novel self-predictive model named AmelPred. Empirical results demonstrate that its stochastic version, AmelPredSto, is the best-performing SRL model when combined with actor-critic RL algorithms. The obtained results show substantial improvement in RL algorithms' efficiency by using AmelPredSto in solving the OGN problem.

**arXiv ID:** 2604.21130
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
