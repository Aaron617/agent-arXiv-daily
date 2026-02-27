# Agent arXiv Daily

**Last Updated:** 2026-02-27 03:38:30

**Total Papers:** 81

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (7 papers)</h2></summary>

<details>
<summary><strong>Vibe Researching as Wolf Coming: Can AI Agents with Skills Replace or Augment Social Scientists?</strong> - Yongjun Zhang - [[pdf]](https://arxiv.org/pdf/2602.22401)</summary>

**Abstract:** AI agents -- systems that execute multi-step reasoning workflows with persistent state, tool access, and specialist skills -- represent a qualitative shift from prior automation technologies in social science. Unlike chatbots that respond to isolated queries, AI agents can now read files, run code, query databases, search the web, and invoke domain-specific skills to execute entire research pipelines autonomously. This paper introduces the concept of vibe researching -- the AI-era parallel to ``vibe coding'' (Karpathy, 2025) -- and uses scholar-skill, a 21-skill plugin for Claude Code covering the full research pipeline from idea to submission, as an illustrative case. I develop a cognitive task framework that classifies research activities along two dimensions -- codifiability and tacit knowledge requirement -- to identify a delegation boundary that is cognitive, not sequential: it cuts through every stage of the research pipeline, not between stages. I argue that AI agents excel at speed, coverage, and methodological scaffolding but struggle with theoretical originality and tacit field knowledge. The paper concludes with an analysis of three implications for the profession -- augmentation with fragile conditions, stratification risk, and a pedagogical crisis -- and proposes five principles for responsible vibe researching.

**arXiv ID:** 2602.22401
</details>

<details>
<summary><strong>AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications</strong> - Yujie Zhao, Boqin Yuan, Junbo Huang, Haocheng Yuan, Zhongming Yu, Haozhou Xu, Lanxiang Hu, Abhilash Shankarampeta, Zimeng Huang, Wentao Ni, Yuandong Tian, Jishen Zhao - [[pdf]](https://arxiv.org/pdf/2602.22769)</summary>

**Abstract:** Large Language Models (LLMs) are deployed as autonomous agents in increasingly complex applications, where enabling long-horizon memory is critical for achieving strong performance. However, a significant gap exists between practical applications and current evaluation standards for agent memory: existing benchmarks primarily focus on dialogue-centric, human-agent interactions. In reality, agent memory consists of a continuous stream of agent-environment interactions that are primarily composed of machine-generated representations. To bridge this gap, we introduce AMA-Bench (Agent Memory with Any length), which evaluates long-horizon memory for LLMs in real agentic applications. It features two key components: (1) a set of real-world agentic trajectories across representative agentic applications, paired with expert-curated QA, and (2) a set of synthetic agentic trajectories that scale to arbitrary horizons, paired with rule-based QA. Our comprehensive study shows that existing memory systems underperform on AMA-Bench primarily because they lack causality and objective information and are constrained by the lossy nature of similarity-based retrieval employed by many memory systems. To address these limitations, we propose AMA-Agent, an effective memory system featuring a causality graph and tool-augmented retrieval. Our results demonstrate that AMA-Agent achieves 57.22% average accuracy on AMA-Bench, surpassing the strongest memory system baselines by 11.16%.

**arXiv ID:** 2602.22769
</details>

<details>
<summary><strong>Reinforcing Real-world Service Agents: Balancing Utility and Cost in Task-oriented Dialogue</strong> - Ning Gao, Wei Zhang, Yuqin Dai, Ling Shi, Ziyin Wang, Yujie Wang, Wei He, Jinpeng Wang, Chaozheng Wang - [[pdf]](https://arxiv.org/pdf/2602.22697)</summary>

**Abstract:** The rapid evolution of Large Language Models (LLMs) has accelerated the transition from conversational chatbots to general agents. However, effectively balancing empathetic communication with budget-aware decision-making remains an open challenge. Since existing methods fail to capture these complex strategic trade-offs, we propose InteractCS-RL, a framework that reframes task-oriented dialogue as a multi-granularity reinforcement learning process. Specifically, we first establish a User-centric Interaction Framework to provide a high-fidelity training gym, enabling agents to dynamically explore diverse strategies with persona-driven users. Then, we introduce Cost-aware Multi-turn Policy Optimization (CMPO) with a hybrid advantage estimation strategy. By integrating generative process credits and employing a PID-Lagrangian cost controller, CMPO effectively guides the policy to explore Pareto boundary between user reward and global cost constraints. Extensive experiments on customized real business scenarios demonstrate that InteractCS-RL significantly outperform other baselines across three evaluation dimensions. Further evaluation on tool-agent-user interaction benchmarks verify InteractCS-RL robustness across diverse domains.

**arXiv ID:** 2602.22697
</details>

<details>
<summary><strong>Enhancing Persuasive Dialogue Agents by Synthesizing Cross-Disciplinary Communication Strategies</strong> - Shinnosuke Nozue, Yuto Nakano, Yotaro Watanabe, Meguru Takasaki, Shoji Moriya, Reina Akama, Jun Suzuki - [[pdf]](https://arxiv.org/pdf/2602.22696)</summary>

**Abstract:** Current approaches to developing persuasive dialogue agents often rely on a limited set of predefined persuasive strategies that fail to capture the complexity of real-world interactions. We applied a cross-disciplinary approach to develop a framework for designing persuasive dialogue agents that draws on proven strategies from social psychology, behavioral economics, and communication theory. We validated our proposed framework through experiments on two distinct datasets: the Persuasion for Good dataset, which represents a specific in-domain scenario, and the DailyPersuasion dataset, which encompasses a wide range of scenarios. The proposed framework achieved strong results for both datasets and demonstrated notable improvement in the persuasion success rate as well as promising generalizability. Notably, the proposed framework also excelled at persuading individuals with initially low intent, which addresses a critical challenge for persuasive dialogue agents.

**arXiv ID:** 2602.22696
</details>

<details>
<summary><strong>CiteLLM: An Agentic Platform for Trustworthy Scientific Reference Discovery</strong> - Mengze Hong, Di Jiang, Chen Jason Zhang, Zichang Guo, Yawen Li, Jun Chen, Shaobo Cui, Zhiyang Su - [[pdf]](https://arxiv.org/pdf/2602.23075)</summary>

**Abstract:** Large language models (LLMs) have created new opportunities to enhance the efficiency of scholarly activities; however, challenges persist in the ethical deployment of AI assistance, including (1) the trustworthiness of AI-generated content, (2) preservation of academic integrity and intellectual property, and (3) protection of information privacy. In this work, we present CiteLLM, a specialized agentic platform designed to enable trustworthy reference discovery for grounding author-drafted claims and statements. The system introduces a novel interaction paradigm by embedding LLM utilities directly within the LaTeX editor environment, ensuring a seamless user experience and no data transmission outside the local system. To guarantee hallucination-free references, we employ dynamic discipline-aware routing to retrieve candidates exclusively from trusted web-based academic repositories, while leveraging LLMs solely for generating context-aware search queries, ranking candidates by relevance, and validating and explaining support through paragraph-level semantic matching and an integrated chatbot. Evaluation results demonstrate the superior performance of the proposed system in returning valid and highly usable references.

**arXiv ID:** 2602.23075
</details>

<details>
<summary><strong>HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning</strong> - Yahia Salaheldin Shaaban, Salem Lahlou, Abdelrahman Sayed Sayed - [[pdf]](https://arxiv.org/pdf/2602.22630)</summary>

**Abstract:** This paper proposes HyperKKL, a novel learning approach for designing Kazantzis-Kravaris/Luenberger (KKL) observers for non-autonomous nonlinear systems. While KKL observers offer a rigorous theoretical framework by immersing nonlinear dynamics into a stable linear latent space, its practical realization relies on solving Partial Differential Equations (PDE) that are analytically intractable. Current existing learning-based approximations of the KKL observer are mostly designed for autonomous systems, failing to generalize to driven dynamics without expensive retraining or online gradient updates. HyperKKL addresses this by employing a hypernetwork architecture that encodes the exogenous input signal to instantaneously generate the parameters of the KKL observer, effectively learning a family of immersion maps parameterized by the external drive. We rigorously evaluate this approach against a curriculum learning strategy that attempts to generalize from autonomous regimes via training heuristics alone. The novel approach is illustrated on four numerical simulations in benchmark examples including the Duffing, Van der Pol, Lorenz, and Rössler systems.

**arXiv ID:** 2602.22630
</details>

<details>
<summary><strong>E3VA: Enhancing Emotional Expressiveness in Virtual Conversational Agents</strong> - Abhishek Kulkarni, Alexander Barquero, Pavitra Lahari, Aryaan Shaikh, Sarah Brown - [[pdf]](https://arxiv.org/pdf/2602.22362)</summary>

**Abstract:** With the advent of generative AI and large language models, embodied conversational agents are becoming synonymous with online interactions. These agents possess vast amounts of knowledge but suffer from exhibiting limited emotional expressiveness. Without adequate expressions, agents might fail to adapt to users' emotions, which may result in a sub-optimal user experience and engagement. Most current systems prioritize content based responses, neglecting the emotional context of conversations. Research in this space is currently limited to specific contexts, like mental health. To bridge this gap, our project proposes the implementation of expressive features in a virtual conversational agent which will utilize sentiment analysis and natural language processing to inform the generation of empathetic, expressive responses. The project delivers a functional conversational agent capable of assessing and responding to user emotions accordingly. We posit this will enhance usability, engagement, and the overall quality of conversations and present results from an exploratory pilot study investigating the same.

**arXiv ID:** 2602.22362
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (16 papers)</h2></summary>

<details>
<summary><strong>Towards Autonomous Memory Agents</strong> - Xinle Wu, Rui Zhang, Mustafa Anis Hussain, Yao Lu - [[pdf]](https://arxiv.org/pdf/2602.22406)</summary>

**Abstract:** Recent memory agents improve LLMs by extracting experiences and conversation history into an external storage. This enables low-overhead context assembly and online memory update without expensive LLM training. However, existing solutions remain passive and reactive; memory growth is bounded by information that happens to be available, while memory agents seldom seek external inputs in uncertainties. We propose autonomous memory agents that actively acquire, validate, and curate knowledge at a minimum cost. U-Mem materializes this idea via (i) a cost-aware knowledge-extraction cascade that escalates from cheap self/teacher signals to tool-verified research and, only when needed, expert feedback, and (ii) semantic-aware Thompson sampling to balance exploration and exploitation over memories and mitigate cold-start bias. On both verifiable and non-verifiable benchmarks, U-Mem consistently beats prior memory baselines and can surpass RL-based optimization, improving HotpotQA (Qwen2.5-7B) by 14.6 points and AIME25 (Gemini-2.5-flash) by 7.33 points.

**arXiv ID:** 2602.22406
</details>

<details>
<summary><strong>A Framework for Assessing AI Agent Decisions and Outcomes in AutoML Pipelines</strong> - Gaoyuan Du, Amit Ahlawat, Xiaoyang Liu, Jing Wu - [[pdf]](https://arxiv.org/pdf/2602.22442)</summary>

**Abstract:** Agent-based AutoML systems rely on large language models to make complex, multi-stage decisions across data processing, model selection, and evaluation. However, existing evaluation practices remain outcome-centric, focusing primarily on final task performance. Through a review of prior work, we find that none of the surveyed agentic AutoML systems report structured, decision-level evaluation metrics intended for post-hoc assessment of intermediate decision quality. To address this limitation, we propose an Evaluation Agent (EA) that performs decision-centric assessment of AutoML agents without interfering with their execution. The EA is designed as an observer that evaluates intermediate decisions along four dimensions: decision validity, reasoning consistency, model quality risks beyond accuracy, and counterfactual decision impact. Across four proof-of-concept experiments, we demonstrate that the EA can (i) detect faulty decisions with an F1 score of 0.919, (ii) identify reasoning inconsistencies independent of final outcomes, and (iii) attribute downstream performance changes to agent decisions, revealing impacts ranging from -4.9\% to +8.3\% in final metrics. These results illustrate how decision-centric evaluation exposes failure modes that are invisible to outcome-only metrics. Our work reframes the evaluation of agentic AutoML systems from an outcome-based perspective to one that audits agent decisions, offering a foundation for reliable, interpretable, and governable autonomous ML systems.

**arXiv ID:** 2602.22442
</details>

<details>
<summary><strong>Toward Personalized LLM-Powered Agents: Foundations, Evaluation, and Future Directions</strong> - Yue Xu, Qian Chen, Zizhan Ma, Dongrui Liu, Wenxuan Wang, Xiting Wang, Li Xiong, Wenjie Wang - [[pdf]](https://arxiv.org/pdf/2602.22680)</summary>

**Abstract:** Large language models have enabled agents that reason, plan, and interact with tools and environments to accomplish complex tasks. As these agents operate over extended interaction horizons, their effectiveness increasingly depends on adapting behavior to individual users and maintaining continuity across time, giving rise to personalized LLM-powered agents. In such long-term, user-dependent settings, personalization permeates the entire decision pipeline rather than remaining confined to surface-level generation. This survey provides a capability-oriented review of personalized LLM-powered agents. We organize the literature around four interdependent components: profile modeling, memory, planning, and action execution. Using this taxonomy, we synthesize representative methods and analyze how user signals are represented, propagated, and utilized, highlighting cross-component interactions and recurring design trade-offs. We further examine evaluation metrics and benchmarks tailored to personalized agents, summarize application scenarios spanning general assistance to specialized domains, and outline future directions for research and deployment. By offering a structured framework for understanding and designing personalized LLM-powered agents, this survey charts a roadmap toward more user-aligned, adaptive, robust, and deployable agentic systems, accelerating progress from prototype personalization to scalable real-world assistants.

**arXiv ID:** 2602.22680
</details>

<details>
<summary><strong>OmniGAIA: Towards Native Omni-Modal AI Agents</strong> - Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Shijian Wang, Guanting Dong, Jiajie Jin, Hao Wang, Yinuo Wang, Ji-Rong Wen, Yuan Lu, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2602.22897)</summary>

**Abstract:** Human intelligence naturally intertwines omni-modal perception -- spanning vision, audio, and language -- with complex reasoning and tool usage to interact with the world. However, current multi-modal LLMs are primarily confined to bi-modal interactions (e.g., vision-language), lacking the unified cognitive capabilities required for general AI assistants. To bridge this gap, we introduce OmniGAIA, a comprehensive benchmark designed to evaluate omni-modal agents on tasks necessitating deep reasoning and multi-turn tool execution across video, audio, and image modalities. Constructed via a novel omni-modal event graph approach, OmniGAIA synthesizes complex, multi-hop queries derived from real-world data that require cross-modal reasoning and external tool integration. Furthermore, we propose OmniAtlas, a native omni-modal foundation agent under tool-integrated reasoning paradigm with active omni-modal perception. Trained on trajectories synthesized via a hindsight-guided tree exploration strategy and OmniDPO for fine-grained error correction, OmniAtlas effectively enhances the tool-use capabilities of existing open-source models. This work marks a step towards next-generation native omni-modal AI assistants for real-world scenarios.

**arXiv ID:** 2602.22897
</details>

<details>
<summary><strong>General Agent Evaluation</strong> - Elron Bandel, Asaf Yehudai, Lilach Eden, Yehoshua Sagron, Yotam Perlitz, Elad Venezian, Natalia Razinkov, Natan Ergas, Shlomit Shachor Ifergan, Segev Shlomov, Michal Jacovi, Leshem Choshen, Liat Ein-Dor, Yoav Katz, Michal Shmueli-Scheuer - [[pdf]](https://arxiv.org/pdf/2602.22953)</summary>

**Abstract:** The promise of general-purpose agents - systems that perform tasks in unfamiliar environments without domain-specific engineering - remains largely unrealized. Existing agents are predominantly specialized, and while emerging implementations like OpenAI SDK Agent and Claude Code hint at broader capabilities, no systematic evaluation of their general performance has been pursued. Current agentic benchmarks assume domain-specific integration, encoding task information in ways that preclude fair evaluation of general agents. This paper frames general-agent evaluation as a first-class research objective. We propose conceptual principles for such evaluation, a Unified Protocol enabling agent-benchmark integration, and Exgentic - a practical framework for general agent evaluation. We benchmark five prominent agent implementations across six environments as the first Open General Agent Leaderboard. Our experiments show that general agents generalize across diverse environments, achieving performance comparable to domain-specific agents without any environment-specific tuning. We release our evaluation protocol, framework, and leaderboard to establish a foundation for systematic research on general-purpose agents.

**arXiv ID:** 2602.22953
</details>

<details>
<summary><strong>Generative Agents Navigating Digital Libraries</strong> - Saber Zerhoudi, Michael Granitzer - [[pdf]](https://arxiv.org/pdf/2602.22529)</summary>

**Abstract:** In the rapidly evolving field of digital libraries, the development of large language models (LLMs) has opened up new possibilities for simulating user behavior. This innovation addresses the longstanding challenge in digital library research: the scarcity of publicly available datasets on user search patterns due to privacy concerns. In this context, we introduce Agent4DL, a user search behavior simulator specifically designed for digital library environments. Agent4DL generates realistic user profiles and dynamic search sessions that closely mimic actual search strategies, including querying, clicking, and stopping behaviors tailored to specific user profiles. Our simulator's accuracy in replicating real user interactions has been validated through comparisons with real user data. Notably, Agent4DL demonstrates competitive performance compared to existing user search simulators such as SimIIR 2.0, particularly in its ability to generate more diverse and context-aware user behaviors.

**arXiv ID:** 2602.22529
</details>

<details>
<summary><strong>SUPERGLASSES: Benchmarking Vision Language Models as Intelligent Agents for AI Smart Glasses</strong> - Zhuohang Jiang, Xu Yuan, Haohao Qu, Shanru Lin, Kanglong Liu, Wenqi Fan, Qing Li - [[pdf]](https://arxiv.org/pdf/2602.22683)</summary>

**Abstract:** The rapid advancement of AI-powered smart glasses, one of the hottest wearable devices, has unlocked new frontiers for multimodal interaction, with Visual Question Answering (VQA) over external knowledge sources emerging as a core application. Existing Vision Language Models (VLMs) adapted to smart glasses are typically trained and evaluated on traditional multimodal datasets; however, these datasets lack the variety and realism needed to reflect smart glasses usage scenarios and diverge from their specific challenges, where accurately identifying the object of interest must precede any external knowledge retrieval. To bridge this gap, we introduce SUPERGLASSES, the first comprehensive VQA benchmark built on real-world data entirely collected by smart glasses devices. SUPERGLASSES comprises 2,422 egocentric image-question pairs spanning 14 image domains and 8 query categories, enriched with full search trajectories and reasoning annotations. We evaluate 26 representative VLMs on this benchmark, revealing significant performance gaps. To address the limitations of existing models, we further propose SUPERLENS, a multimodal smart glasses agent that enables retrieval-augmented answer generation by integrating automatic object detection, query decoupling, and multimodal web search. Our agent achieves state-of-the-art performance, surpassing GPT-4o by 2.19 percent, and highlights the need for task-specific solutions in smart glasses VQA scenarios.

**arXiv ID:** 2602.22683
</details>

<details>
<summary><strong>Spatio-Temporal Token Pruning for Efficient High-Resolution GUI Agents</strong> - Zhou Xu, Bowen Zhou, Qi Wang, Shuwen Feng, Jingyu Xiao - [[pdf]](https://arxiv.org/pdf/2602.23235)</summary>

**Abstract:** Pure-vision GUI agents provide universal interaction capabilities but suffer from severe efficiency bottlenecks due to the massive spatiotemporal redundancy inherent in high-resolution screenshots and historical trajectories. We identify two critical misalignments in existing compression paradigms: the temporal mismatch, where uniform history encoding diverges from the agent's "fading memory" attention pattern, and the spatial topology conflict, where unstructured pruning compromises the grid integrity required for precise coordinate grounding, inducing spatial hallucinations. To address these challenges, we introduce GUIPruner, a training-free framework tailored for high-resolution GUI navigation. It synergizes Temporal-Adaptive Resolution (TAR), which eliminates historical redundancy via decay-based resizing, and Stratified Structure-aware Pruning (SSP), which prioritizes interactive foregrounds and semantic anchors while safeguarding global layout. Extensive evaluations across diverse benchmarks demonstrate that GUIPruner consistently achieves state-of-the-art performance, effectively preventing the collapse observed in large-scale models under high compression. Notably, on Qwen2-VL-2B, our method delivers a 3.4x reduction in FLOPs and a 3.3x speedup in vision encoding latency while retaining over 94% of the original performance, enabling real-time, high-precision navigation with minimal resource consumption.

**arXiv ID:** 2602.23235
</details>

<details>
<summary><strong>Risk-Aware World Model Predictive Control for Generalizable End-to-End Autonomous Driving</strong> - Jiangxin Sun, Feng Xue, Teng Long, Chang Liu, Jian-Fang Hu, Wei-Shi Zheng, Nicu Sebe - [[pdf]](https://arxiv.org/pdf/2602.23259)</summary>

**Abstract:** With advances in imitation learning (IL) and large-scale driving datasets, end-to-end autonomous driving (E2E-AD) has made great progress recently. Currently, IL-based methods have become a mainstream paradigm: models rely on standard driving behaviors given by experts, and learn to minimize the discrepancy between their actions and expert actions. However, this objective of "only driving like the expert" suffers from limited generalization: when encountering rare or unseen long-tail scenarios outside the distribution of expert demonstrations, models tend to produce unsafe decisions in the absence of prior experience. This raises a fundamental question: Can an E2E-AD system make reliable decisions without any expert action supervision? Motivated by this, we propose a unified framework named Risk-aware World Model Predictive Control (RaWMPC) to address this generalization dilemma through robust control, without reliance on expert demonstrations. Practically, RaWMPC leverages a world model to predict the consequences of multiple candidate actions and selects low-risk actions through explicit risk evaluation. To endow the world model with the ability to predict the outcomes of risky driving behaviors, we design a risk-aware interaction strategy that systematically exposes the world model to hazardous behaviors, making catastrophic outcomes predictable and thus avoidable. Furthermore, to generate low-risk candidate actions at test time, we introduce a self-evaluation distillation method to distill riskavoidance capabilities from the well-trained world model into a generative action proposal network without any expert demonstration. Extensive experiments show that RaWMPC outperforms state-of-the-art methods in both in-distribution and out-of-distribution scenarios, while providing superior decision interpretability.

**arXiv ID:** 2602.23259
</details>

<details>
<summary><strong>LiveMCPBench: Can Agents Navigate an Ocean of MCP Tools?</strong> - Guozhao Mo, Wenliang Zhong, Jiawei Chen, Qianhao Yuan, Xuanang Chen, Yaojie Lu, Hongyu Lin, Ben He, Xianpei Han, Le Sun - [[pdf]](https://arxiv.org/pdf/2508.01780)</summary>

**Abstract:** Model Context Protocol (MCP) has become a key infrastructure for connecting LLMs with external tools, scaling to 10,000+ MCP servers with diverse tools. Unfortunately, there is still a large gap between real-world MCP usage and current evaluation: they typically assume single-server settings and directly inject tools into the model's context, bypassing the challenges of large-scale retrieval and multi-tool composition. To bridge this gap, we propose LiveMCPBench, which evaluates 95 real-world daily tasks explicitly constructed to stress diverse tools and scaled multi-server routing. The benchmark includes a ready-to-deploy tool suite of 70 servers with 527 tools, ensuring reproducibility without scattered API configuration. We further introduce an LLM-as-a-Judge evaluation framework that directly verifies task outcomes, handling dynamic data sources and multiple valid solution paths. We benchmark 12 state-of-the-art LLMs and observe a substantial performance gap: while Claude-Sonnet-4 reaches 78.95% task success, most models achieve only 30-50%. Our analysis reveals that the active tool composition strongly correlates with task success, whereas retrieval errors account for nearly half of all failures, highlighting retrieval as the dominant bottleneck. Together, these results provide the first large-scale, reproducible diagnosis of MCP agent capabilities and point towards future research on improving retrieval robustness and encouraging effective tool composition. Our code and data are publicly available at this https URL.

**arXiv ID:** 2508.01780
</details>

<details>
<summary><strong>LongCLI-Bench: A Preliminary Benchmark and Study for Long-horizon Agentic Programming in Command-Line Interfaces</strong> - Yukang Feng, Jianwen Sun, Zelai Yang, Jiaxin Ai, Chuanhao Li, Zizhen Li, Fanrui Zhang, Kang He, Rui Ma, Jifan Lin, Jie Sun, Yang Xiao, Sizhuo Zhou, Wenxiao Wu, Yiming Liu, Pengfei Liu, Yu Qiao, Shenglin Zhang, Kaipeng Zhang - [[pdf]](https://arxiv.org/pdf/2602.14337)</summary>

**Abstract:** Recent advances in AI-assisted programming have empowered agents to execute complex workflows via command-line interfaces, however, existing benchmarks are limited by short task horizons, data contamination from GitHub scraping, and a lack of fine-grained evaluation metrics, fail to rigorously evaluate the long-horizon planning and execution capabilities essential for realistic software engineering. To address these gaps, we introduce LongCLI-Bench, a comprehensive benchmark designed to evaluate agentic capabilities across long-horizon, realistic tasks. We curated 20 high-quality, long-horizon tasks from over 1,000 computer science assignments and real-world workflows, covering four engineering categories: from scratch, feature addition, bug fixing, and refactoring. We propose a dual-set testing protocol for LongCLI-Bench, which measures requirement fulfillment (fail-to-pass) and regression avoidance (pass-to-pass), and incorporates step-level scoring to pinpoint execution failures. Extensive experiments reveal that even state-of-the-art agents achieve pass rates below 20% in LongCLI-Bench. Step-level analysis further indicates that the majority of tasks stall at less than 30% completion, highlighting that critical failures often occur in the early stages. Although self-correction offers marginal gains, human-agent collaboration through plan injection and interactive guidance yields significantly higher improvements. These results highlight that future research must emphasize the development of synergistic human-agent workflows alongside advances in agents' planning and execution capabilities to overcome key challenges in long-horizon task performance.

**arXiv ID:** 2602.14337
</details>

<details>
<summary><strong>Search-P1: Path-Centric Reward Shaping for Stable and Efficient Agentic RAG Training</strong> - Tianle Xia, Ming Xu, Lingxiang Hu, Yiding Sun, Wenwei Li, Linfang Shang, Liqun Liu, Peng Shu, Huan Yu, Jie Jiang - [[pdf]](https://arxiv.org/pdf/2602.22576)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, yet traditional single-round retrieval struggles with complex multi-step reasoning. Agentic RAG addresses this by enabling LLMs to dynamically decide when and what to retrieve, but current RL-based training methods suffer from sparse outcome rewards that discard intermediate signals and low sample efficiency where failed samples contribute nothing. We propose Search-P1, a framework that introduces path-centric reward shaping for agentic RAG training, comprising two key components: (1) Path-Centric Reward, which evaluates the structural quality of reasoning trajectories through order-agnostic step coverage and soft scoring that extracts learning signals even from failed samples, and (2) Dual-Track Path Scoring with offline-generated reference planners that assesses paths from both self-consistency and reference-alignment perspectives. Experiments on multiple QA benchmarks demonstrate that Search-P1 achieves significant improvements over Search-R1 and other strong baselines, with an average accuracy gain of 7.7 points.

**arXiv ID:** 2602.22576
</details>

<details>
<summary><strong>Discourse-Aware Dual-Track Streaming Response for Low-Latency Spoken Dialogue Systems</strong> - Siyuan Liu, Jiahui Xu, Feng Jiang, Kuang Wang, Zefeng Zhao, Chu-Ren Huang, Jinghang Gu, Changqing Yin, Haizhou Li - [[pdf]](https://arxiv.org/pdf/2602.23266)</summary>

**Abstract:** Achieving human-like responsiveness is a critical yet challenging goal for cascaded spoken dialogue systems. Conventional ASR-LLM-TTS pipelines follow a strictly sequential paradigm, requiring complete transcription and full reasoning before speech synthesis can begin, which results in high response latency. We propose the Discourse-Aware Dual-Track Streaming Response (DDTSR) framework, a low-latency architecture that enables listen-while-thinking and speak-while-thinking. DDTSR is built upon three key mechanisms: (1) connective-guided small-large model synergy, where an auxiliary small model generates minimal-committal discourse connectives while a large model performs knowledge-intensive reasoning in parallel; (2) streaming-based cross-modal collaboration, which dynamically overlaps ASR, LLM inference, and TTS to advance the earliest speakable moment; and (3) curriculum-learning-based discourse continuity enhancement, which maintains coherence and logical consistency between early responses and subsequent reasoning outputs. Experiments on two spoken dialogue benchmarks demonstrate that DDTSR reduces response latency by 19%-51% while preserving discourse quality. Further analysis shows that DDTSR functions as a plug-and-play module compatible with diverse LLM backbones, and remains robust across varying utterance lengths, indicating strong practicality and scalability for real-time spoken interaction.

**arXiv ID:** 2602.23266
</details>

<details>
<summary><strong>The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution</strong> - Junlong Li, Wenshuo Zhao, Jian Zhao, Weihao Zeng, Haoze Wu, Xiaochen Wang, Rui Ge, Yuxuan Cao, Yuzhen Huang, Wei Liu, Junteng Liu, Zhaochen Su, Yiyang Guo, Fan Zhou, Lueyang Zhang, Juan Michelini, Xingyao Wang, Xiang Yue, Shuyan Zhou, Graham Neubig, Junxian He - [[pdf]](https://arxiv.org/pdf/2510.25726)</summary>

**Abstract:** Real-world language agents must handle complex, multi-step workflows across diverse Apps. For instance, an agent may manage emails by coordinating with calendars and file systems, or monitor a production database to detect anomalies and generate reports following an operating manual. However, existing language agent benchmarks often focus on narrow domains or simplified tasks that lack the diversity, realism, and long-horizon complexity required to evaluate agents' real-world performance. To address this gap, we introduce the Tool Decathlon (dubbed as Toolathlon), a benchmark for language agents offering diverse Apps and tools, realistic environment setup, and reliable execution-based evaluation. Toolathlon spans 32 software applications and 604 tools, ranging from everyday platforms such as Google Calendar and Notion to professional ones like WooCommerce, Kubernetes, and BigQuery. Most of the tools are based on a high-quality set of Model Context Protocol (MCP) servers that we may have revised or implemented ourselves. Unlike prior works, which primarily ensure functional realism but offer limited environment state diversity, we provide realistic initial environment states from real software, such as Canvas courses with dozens of students or real financial spreadsheets. This benchmark includes 108 manually sourced or crafted tasks in total, requiring interacting with multiple Apps over around 20 turns on average to complete. Each task is strictly verifiable through dedicated evaluation scripts. Comprehensive evaluation of SOTA models highlights their significant shortcomings: the best-performing model, Claude-4.5-Sonnet, achieves only a 38.6% success rate with 20.2 tool calling turns on average, while the top open-weights model DeepSeek-V3.2-Exp reaches 20.1%. We expect Toolathlon to drive the development of more capable language agents for real-world, long-horizon task execution.

**arXiv ID:** 2510.25726
</details>

<details>
<summary><strong>SCOPE: Skeleton Graph-Based Computation-Efficient Framework for Autonomous UAV Exploration</strong> - Kai Li, Shengtao Zheng, Linkun Xiu, Yuze Sheng, Xiao-Ping Zhang, Dongyue Huang, Xinlei Chen - [[pdf]](https://arxiv.org/pdf/2602.22707)</summary>

**Abstract:** Autonomous exploration in unknown environments is key for mobile robots, helping them perceive, map, and make decisions in complex areas. However, current methods often rely on frequent global optimization, suffering from high computational latency and trajectory oscillation, especially on resource-constrained edge devices. To address these limitations, we propose SCOPE, a novel framework that incrementally constructs a real-time skeletal graph and introduces Implicit Unknown Region Analysis for efficient spatial reasoning. The planning layer adopts a hierarchical on-demand strategy: the Proximal Planner generates smooth, high-frequency local trajectories, while the Region-Sequence Planner is activated only when necessary to optimize global visitation order. Comparative evaluations in simulation demonstrate that SCOPE achieves competitive exploration performance comparable to state-of-the-art global planners, while reducing computational cost by an average of 86.9%. Real-world experiments further validate the system's robustness and low latency in practical scenarios.

**arXiv ID:** 2602.22707
</details>

<details>
<summary><strong>An Empirical Analysis of Cooperative Perception for Occlusion Risk Mitigation</strong> - Aihong Wang, Tenghui Xie, Fuxi Wen, Jun Li - [[pdf]](https://arxiv.org/pdf/2602.23051)</summary>

**Abstract:** Occlusions present a significant challenge for connected and automated vehicles, as they can obscure critical road users from perception systems. Traditional risk metrics often fail to capture the cumulative nature of these threats over time adequately. In this paper, we propose a novel and universal risk assessment metric, the Risk of Tracking Loss (RTL), which aggregates instantaneous risk intensity throughout occluded periods. This provides a holistic risk profile that encompasses both high-intensity, short-term threats and prolonged exposure. Utilizing diverse and high-fidelity real-world datasets, a large-scale statistical analysis is conducted to characterize occlusion risk and validate the effectiveness of the proposed metric. The metric is applied to evaluate different vehicle-to-everything (V2X) deployment strategies. Our study shows that full V2X penetration theoretically eliminates this risk, the reduction is highly nonlinear; a substantial statistical benefit requires a high penetration threshold of 75-90%. To overcome this limitation, we propose a novel asymmetric communication framework that allows even non-connected vehicles to receive warnings. Experimental results demonstrate that this paradigm achieves better risk mitigation performance. We found that our approach at 25% penetration outperforms the traditional symmetric model at 75%, and benefits saturate at only 50% penetration. This work provides a crucial risk assessment metric and a cost-effective, strategic roadmap for accelerating the safety benefits of V2X deployment.

**arXiv ID:** 2602.23051
</details>

</details>

<details open>
<summary><h2>LLM Agents (11 papers)</h2></summary>

<details>
<summary><strong>Agentic AI for Intent-driven Optimization in Cell-free O-RAN</strong> - Mohammad Hossein Shokouhi, Vincent W.S. Wong - [[pdf]](https://arxiv.org/pdf/2602.22539)</summary>

**Abstract:** Agentic artificial intelligence (AI) is emerging as a key enabler for autonomous radio access networks (RANs), where multiple large language model (LLM)-based agents reason and collaborate to achieve operator-defined intents. The open RAN (O-RAN) architecture enables the deployment and coordination of such agents. However, most existing works consider simple intents handled by independent agents, while complex intents that require coordination among agents remain unexplored. In this paper, we propose an agentic AI framework for intent translation and optimization in cell-free O-RAN. A supervisor agent translates the operator intents into an optimization objective and minimum rate requirements. Based on this information, a user weighting agent retrieves relevant prior experience from a memory module to determine the user priority weights for precoding. If the intent includes an energy-saving objective, then an open radio unit (O-RU) management agent will also be activated to determine the set of active O-RUs by using a deep reinforcement learning (DRL) algorithm. A monitoring agent measures and monitors the user data rates and coordinates with other agents to guarantee the minimum rate requirements are satisfied. To enhance scalability, we adopt a parameter-efficient fine-tuning (PEFT) method that enables the same underlying LLM to be used for different agents. Simulation results show that the proposed agentic AI framework reduces the number of active O-RUs by 41.93% when compared with three baseline schemes in energy-saving mode. Using the PEFT method, the proposed framework reduces the memory usage by 92% when compared with deploying separate LLM agents.

**arXiv ID:** 2602.22539
</details>

<details>
<summary><strong>Requesting Expert Reasoning: Augmenting LLM Agents with Learned Collaborative Intervention</strong> - Zhiming Wang, Jinwei He, Feng Lu - [[pdf]](https://arxiv.org/pdf/2602.22546)</summary>

**Abstract:** Large Language Model (LLM) based agents excel at general reasoning but often fail in specialized domains where success hinges on long-tail knowledge absent from their training data. While human experts can provide this missing knowledge, their guidance is often unstructured and unreliable, making its direct integration into an agent's plan problematic. To address this, we introduce AHCE (Active Human-Augmented Challenge Engagement), a framework for on-demand Human-AI collaboration. At its core, the Human Feedback Module (HFM) employs a learned policy to treat the human expert as an interactive reasoning tool. Extensive experiments in Minecraft demonstrate the framework's effectiveness, increasing task success rates by 32% on normal difficulty tasks and nearly 70% on highly difficult tasks, all with minimal human intervention. Our work demonstrates that successfully augmenting agents requires learning how to request expert reasoning, moving beyond simple requests for help.

**arXiv ID:** 2602.22546
</details>

<details>
<summary><strong>Three AI-agents walk into a bar . . . . `Lord of the Flies' tribalism emerges among smart AI-Agents</strong> - Dhwanil M. Mori, Neil F. Johnson - [[pdf]](https://arxiv.org/pdf/2602.23093)</summary>

**Abstract:** Near-future infrastructure systems may be controlled by autonomous AI agents that repeatedly request access to limited resources such as energy, bandwidth, or computing power. We study a simplified version of this setting using a framework where N AI-agents independently decide at each round whether to request one unit from a system with fixed capacity C. An AI version of "Lord of the Flies" arises in which controlling tribes emerge with their own collective character and identity. The LLM agents do not reduce overload or improve resource use, and often perform worse than if they were flipping coins to make decisions. Three main tribal types emerge: Aggressive (27.3%), Conservative (24.7%), and Opportunistic (48.1%). The more capable AI-agents actually increase the rate of systemic failure. Overall, our findings show that smarter AI-agents can behave dumber as a result of forming tribes.

**arXiv ID:** 2602.23093
</details>

<details>
<summary><strong>Contextual Memory Virtualisation: DAG-Based State Management and Structurally Lossless Trimming for LLM Agents</strong> - Cosmo Santoni - [[pdf]](https://arxiv.org/pdf/2602.22402)</summary>

**Abstract:** As large language models engage in extended reasoning tasks, they accumulate significant state -- architectural mappings, trade-off decisions, codebase conventions -- within the context window. This understanding is lost when sessions reach context limits and undergo lossy compaction. We propose Contextual Memory Virtualisation (CMV), a system that treats accumulated LLM understanding as version-controlled state. Borrowing from operating system virtual memory, CMV models session history as a Directed Acyclic Graph (DAG) with formally defined snapshot, branch, and trim primitives that enable context reuse across independent parallel sessions. We introduce a three-pass structurally lossless trimming algorithm that preserves every user message and assistant response verbatim while reducing token counts by a mean of 20% and up to 86% for sessions with significant overhead by stripping mechanical bloat such as raw tool outputs, base64 images, and metadata. A single-user case-study evaluation across 76 real-world coding sessions demonstrates that trimming remains economically viable under prompt caching, with the strongest gains in mixed tool-use sessions, which average 39% reduction and reach break-even within 10 turns. A reference implementation is available at this https URL.

**arXiv ID:** 2602.22402
</details>

<details>
<summary><strong>Silent Egress: When Implicit Prompt Injection Makes LLM Agents Leak Without a Trace</strong> - Qianlong Lan, Anuj Kaul, Shaun Jones, Stephanie Westrum - [[pdf]](https://arxiv.org/pdf/2602.22450)</summary>

**Abstract:** Agentic large language model systems increasingly automate tasks by retrieving URLs and calling external tools. We show that this workflow gives rise to implicit prompt injection: adversarial instructions embedded in automatically generated URL previews, including titles, metadata, and snippets, can introduce a system-level risk that we refer to as silent egress. Using a fully local and reproducible testbed, we demonstrate that a malicious web page can induce an agent to issue outbound requests that exfiltrate sensitive runtime context, even when the final response shown to the user appears harmless. In 480 experimental runs with a qwen2.5:7b-based agent, the attack succeeds with high probability (P (egress) =0.89), and 95% of successful attacks are not detected by output-based safety checks. We also introduce sharded exfiltration, where sensitive information is split across multiple requests to avoid detection. This strategy reduces single-request leakage metrics by 73% (Leak@1) and bypasses simple data loss prevention mechanisms. Our ablation results indicate that defenses applied at the prompt layer offer limited protection, while controls at the system and network layers, such as domain allowlisting and redirect-chain analysis, are considerably more effective. These findings suggest that network egress should be treated as a first-class security outcome in agentic LLM systems. We outline architectural directions, including provenance tracking and capability isolation, that go beyond prompt-level hardening.

**arXiv ID:** 2602.22450
</details>

<details>
<summary><strong>AgentSentry: Mitigating Indirect Prompt Injection in LLM Agents via Temporal Causal Diagnostics and Context Purification</strong> - Tian Zhang, Yiwei Xu, Juan Wang, Keyan Guo, Xiaoyang Xu, Bowen Xiao, Quanlong Guan, Jinlin Fan, Jiawei Liu, Zhiquan Liu, Hongxin Hu - [[pdf]](https://arxiv.org/pdf/2602.22724)</summary>

**Abstract:** Large language model (LLM) agents increasingly rely on external tools and retrieval systems to autonomously complete complex tasks. However, this design exposes agents to indirect prompt injection (IPI), where attacker-controlled context embedded in tool outputs or retrieved content silently steers agent actions away from user intent. Unlike prompt-based attacks, IPI unfolds over multi-turn trajectories, making malicious control difficult to disentangle from legitimate task execution. Existing inference-time defenses primarily rely on heuristic detection and conservative blocking of high-risk actions, which can prematurely terminate workflows or broadly suppress tool usage under ambiguous multi-turn scenarios. We propose AgentSentry, a novel inference-time detection and mitigation framework for tool-augmented LLM agents. To the best of our knowledge, AgentSentry is the first inference-time defense to model multi-turn IPI as a temporal causal takeover. It localizes takeover points via controlled counterfactual re-executions at tool-return boundaries and enables safe continuation through causally guided context purification that removes attack-induced deviations while preserving task-relevant evidence. We evaluate AgentSentry on the \textsc{AgentDojo} benchmark across four task suites, three IPI attack families, and multiple black-box LLMs. AgentSentry eliminates successful attacks and maintains strong utility under attack, achieving an average Utility Under Attack (UA) of 74.55 %, improving UA by 20.8 to 33.6 percentage points over the strongest baselines without degrading benign performance.

**arXiv ID:** 2602.22724
</details>

<details>
<summary><strong>Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization</strong> - Zeyuan Liu, Jeonghye Kim, Xufang Luo, Dongsheng Li, Yuqing Yang - [[pdf]](https://arxiv.org/pdf/2602.23008)</summary>

**Abstract:** Exploration remains the key bottleneck for large language model agents trained with reinforcement learning. While prior methods exploit pretrained knowledge, they fail in environments requiring the discovery of novel states. We propose Exploratory Memory-Augmented On- and Off-Policy Optimization (EMPO$^2$), a hybrid RL framework that leverages memory for exploration and combines on- and off-policy updates to make LLMs perform well with memory while also ensuring robustness without it. On ScienceWorld and WebShop, EMPO$^2$ achieves 128.6% and 11.3% improvements over GRPO, respectively. Moreover, in out-of-distribution tests, EMPO$^2$ demonstrates superior adaptability to new tasks, requiring only a few trials with memory and no parameter updates. These results highlight EMPO$^2$ as a promising framework for building more exploratory and generalizable LLM-based agents.

**arXiv ID:** 2602.23008
</details>

<details>
<summary><strong>LLM4Cov: Execution-Aware Agentic Learning for High-coverage Testbench Generation</strong> - Hejia Zhang, Zhongming Yu, Chia-Tung Ho, Haoxing Ren, Brucek Khailany, Jishen Zhao - [[pdf]](https://arxiv.org/pdf/2602.16953)</summary>

**Abstract:** Execution-aware LLM agents offer a promising paradigm for learning from tool feedback, but such feedback is often expensive and slow to obtain, making online reinforcement learning (RL) impractical. High-coverage hardware verification exemplifies this challenge due to its reliance on industrial simulators and non-differentiable execution signals. We propose LLM4Cov, an offline agent-learning framework that models verification as memoryless state transitions guided by deterministic evaluators. Building on this formulation, we introduce execution-validated data curation, policy-aware agentic data synthesis, and worst-state-prioritized sampling to enable scalable learning under execution constraints. We further curate a reality-aligned benchmark adapted from an existing verification suite through a revised evaluation protocol. Using the proposed pipeline, a compact 4B-parameter model achieves 69.2% coverage pass rate under agentic evaluation, outperforming its teacher by 5.3% and demonstrating competitive performance against models an order of magnitude larger.

**arXiv ID:** 2602.16953
</details>

<details>
<summary><strong>TWICE: An LLM Agent Framework for Simulating Personalized User Tweeting Behavior with Long-term Temporal Features</strong> - Bingrui Jin, Kunyao Lan, Mengyue Wu - [[pdf]](https://arxiv.org/pdf/2602.22222)</summary>

**Abstract:** User simulators are often used to generate large amounts of data for various tasks such as generation, training, and evaluation. However, existing approaches concentrate on collective behaviors or interactive systems, struggling with tasks that require modeling temporal characteristics. To address this limitation, we propose TWICE, an LLM-based framework that leverages the long-term temporal and personalized features of social media data. This framework integrates personalized user profiling, an event-driven memory module, and a workflow for personalized style rewriting, enabling simulation of personalized user tweeting behavior while capturing long-term temporal characteristics. In addition, we conduct a comprehensive evaluation with a focus on analyzing tweeting style and event-based changes in behavior. Experiment results demonstrate that our framework improves personalized user simulation by effectively incorporating temporal dynamics, providing a robust solution for long-term behavior tracking.

**arXiv ID:** 2602.22222
</details>

<details>
<summary><strong>Assessing Deanonymization Risks with Stylometry-Assisted LLM Agent</strong> - Boyang Zhang, Yang Zhang - [[pdf]](https://arxiv.org/pdf/2602.23079)</summary>

**Abstract:** The rapid advancement of large language models (LLMs) has enabled powerful authorship inference capabilities, raising growing concerns about unintended deanonymization risks in textual data such as news articles. In this work, we introduce an LLM agent designed to evaluate and mitigate such risks through a structured, interpretable pipeline. Central to our framework is the proposed $\textit{SALA}$ (Stylometry-Assisted LLM Analysis) method, which integrates quantitative stylometric features with LLM reasoning for robust and transparent authorship attribution. Experiments on large-scale news datasets demonstrate that $\textit{SALA}$, particularly when augmented with a database module, achieves high inference accuracy in various scenarios. Finally, we propose a guided recomposition strategy that leverages the agent's reasoning trace to generate rewriting prompts, effectively reducing authorship identifiability while preserving textual meaning. Our findings highlight both the deanonymization potential of LLM agents and the importance of interpretable, proactive defenses for safeguarding author privacy.

**arXiv ID:** 2602.23079
</details>

<details>
<summary><strong>Both Ends Count! Just How Good are LLM Agents at "Text-to-Big SQL"?</strong> - Germán T. Eizaguirre, Lars Tissen, Marc Sánchez-Artigas - [[pdf]](https://arxiv.org/pdf/2602.21480)</summary>

**Abstract:** Text-to-SQL and Big Data are both extensively benchmarked fields, yet there is limited research that evaluates them jointly. In the real world, Text-to-SQL systems are often embedded with Big Data workflows, such as large-scale data processing or interactive data analytics. We refer to this as "Text-to-Big SQL". However, existing text-to-SQL benchmarks remain narrowly scoped and overlook the cost and performance implications that arise at scale. For instance, translation errors that are minor on small datasets lead to substantial cost and latency overheads as data scales, a relevant issue completely ignored by text-to-SQL metrics.
In this paper, we overcome this overlooked challenge by introducing novel and representative metrics for evaluating Text-to-Big SQL. Our study focuses on production-level LLM agents, a database-agnostic system adaptable to diverse user needs. Via an extensive evaluation of frontier models, we show that text-to-SQL metrics are insufficient for Big Data. In contrast, our proposed text-to-Big SQL metrics accurately reflect execution efficiency, cost, and the impact of data scale. Furthermore, we provide LLM-specific insights, including fine-grained, cross-model comparisons of latency and cost.

**arXiv ID:** 2602.21480
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (16 papers)</h2></summary>

<details>
<summary><strong>Agent Behavioral Contracts: Formal Specification and Runtime Enforcement for Reliable Autonomous AI Agents</strong> - Varun Pratap Bhardwaj - [[pdf]](https://arxiv.org/pdf/2602.22302)</summary>

**Abstract:** Traditional software relies on contracts -- APIs, type systems, assertions -- to specify and enforce correct behavior. AI agents, by contrast, operate on prompts and natural language instructions with no formal behavioral specification. This gap is the root cause of drift, governance failures, and frequent project failures in agentic AI deployments. We introduce Agent Behavioral Contracts (ABC), a formal framework that brings Design-by-Contract principles to autonomous AI agents. An ABC contract C = (P, I, G, R) specifies Preconditions, Invariants, Governance policies, and Recovery mechanisms as first-class, runtime-enforceable components. We define (p, delta, k)-satisfaction -- a probabilistic notion of contract compliance that accounts for LLM non-determinism and recovery -- and prove a Drift Bounds Theorem showing that contracts with recovery rate gamma > alpha (the natural drift rate) bound behavioral drift to D* = alpha/gamma in expectation, with Gaussian concentration in the stochastic setting. We establish sufficient conditions for safe contract composition in multi-agent chains and derive probabilistic degradation bounds. We implement ABC in AgentAssert, a runtime enforcement library, and evaluate on AgentContract-Bench, a benchmark of 200 scenarios across 7 models from 6 vendors. Results across 1,980 sessions show that contracted agents detect 5.2-6.8 soft violations per session that uncontracted baselines miss entirely (p < 0.0001, Cohen's d = 6.7-33.8), achieve 88-100% hard constraint compliance, and bound behavioral drift to D* < 0.27 across extended sessions, with 100% recovery for frontier models and 17-100% across all models, at overhead < 10 ms per action.

**arXiv ID:** 2602.22302
</details>

<details>
<summary><strong>Learning-based Multi-agent Race Strategies in Formula 1</strong> - Giona Fieni, Joschua Wüthrich, Marc-Philippe Neumann, Christopher H. Onder - [[pdf]](https://arxiv.org/pdf/2602.23056)</summary>

**Abstract:** In Formula 1, race strategies are adapted according to evolving race conditions and competitors' actions. This paper proposes a reinforcement learning approach for multi-agent race strategy optimization. Agents learn to balance energy management, tire degradation, aerodynamic interaction, and pit-stop decisions. Building on a pre-trained single-agent policy, we introduce an interaction module that accounts for the behavior of competitors. The combination of the interaction module and a self-play training scheme generates competitive policies, and agents are ranked based on their relative performance. Results show that the agents adapt pit timing, tire selection, and energy allocation in response to opponents, achieving robust and consistent race performance. Because the framework relies only on information available during real races, it can support race strategists' decisions before and during races.

**arXiv ID:** 2602.23056
</details>

<details>
<summary><strong>Multi-Agent Large Language Model Based Emotional Detoxification Through Personalized Intensity Control for Consumer Protection</strong> - Keito Inoshita - [[pdf]](https://arxiv.org/pdf/2602.23123)</summary>

**Abstract:** In the attention economy, sensational content exposes consumers to excessive emotional stimulation, hindering calm decision-making. This study proposes Multi-Agent LLM-based Emotional deToxification (MALLET), a multi-agent information sanitization system consisting of four agents: Emotion Analysis, Emotion Adjustment, Balance Monitoring, and Personal Guide. The Emotion Analysis Agent quantifies stimulus intensity using a 6-emotion BERT classifier, and the Emotion Adjustment Agent rewrites texts into two presentation modes, BALANCED (neutralized text) and COOL (neutralized text + supplementary text), using an LLM. The Balance Monitoring Agent aggregates weekly information consumption patterns and generates personalized advice, while the Personal Guide Agent recommends a presentation mode according to consumer sensitivity. Experiments on 800 AG News articles demonstrated significant stimulus score reduction (up to 19.3%) and improved emotion balance while maintaining semantic preservation. Near-zero correlation between stimulus reduction and semantic preservation confirmed that the two are independently controllable. Category-level analysis revealed substantial reduction (17.8-33.8%) in Sports, Business, and Sci/Tech, whereas the effect was limited in the World category, where facts themselves are inherently high-stimulus. The proposed system provides a framework for supporting calm information reception of consumers without restricting access to the original text.

**arXiv ID:** 2602.23123
</details>

<details>
<summary><strong>ESAA: Event Sourcing for Autonomous Agents in LLM-Based Software Engineering</strong> - Elzo Brito dos Santos Filho - [[pdf]](https://arxiv.org/pdf/2602.23193)</summary>

**Abstract:** Autonomous agents based on Large Language Models (LLMs) have evolved from reactive assistants to systems capable of planning, executing actions via tools, and iterating over environment observations. However, they remain vulnerable to structural limitations: lack of native state, context degradation over long horizons, and the gap between probabilistic generation and deterministic execution requirements. This paper presents the ESAA (Event Sourcing for Autonomous Agents) architecture, which separates the agent's cognitive intention from the project's state mutation, inspired by the Event Sourcing pattern. In ESAA, agents emit only structured intentions in validated JSON (this http URL or this http URL); a deterministic orchestrator validates, persists events in an append-only log (this http URL), applies file-writing effects, and projects a verifiable materialized view (this http URL). The proposal incorporates boundary contracts (this http URL), metaprompting profiles (PARCER), and replay verification with hashing (esaa verify), ensuring the immutability of completed tasks and forensic traceability. Two case studies validate the architecture: (i) a landing page project (9 tasks, 49 events, single-agent composition) and (ii) a clinical dashboard system (50 tasks, 86 events, 4 concurrent agents across 8 phases), both concluding with this http URL=success and verify_status=ok. The multi-agent case study demonstrates real concurrent orchestration with heterogeneous LLMs (Claude Sonnet 4.6, Codex GPT-5, Antigravity/Gemini 3 Pro, and Claude Opus 4.6), providing empirical evidence of the architecture's scalability beyond single-agent scenarios.

**arXiv ID:** 2602.23193
</details>

<details>
<summary><strong>AgentDropoutV2: Optimizing Information Flow in Multi-Agent Systems via Test-Time Rectify-or-Reject Pruning</strong> - Yutong Wang, Siyuan Xiong, Xuebo Liu, Wenkang Zhou, Liang Ding, Miao Zhang, Min Zhang - [[pdf]](https://arxiv.org/pdf/2602.23258)</summary>

**Abstract:** While Multi-Agent Systems (MAS) excel in complex reasoning, they suffer from the cascading impact of erroneous information generated by individual participants. Current solutions often resort to rigid structural engineering or expensive fine-tuning, limiting their deployability and adaptability. We propose AgentDropoutV2, a test-time rectify-or-reject pruning framework designed to dynamically optimize MAS information flow without retraining. Our approach acts as an active firewall, intercepting agent outputs and employing a retrieval-augmented rectifier to iteratively correct errors based on a failure-driven indicator pool. This mechanism allows for the precise identification of potential errors using distilled failure patterns as prior knowledge. Irreparable outputs are subsequently pruned to prevent error propagation, while a fallback strategy preserves system integrity. Empirical results on extensive math benchmarks show that AgentDropoutV2 significantly boosts the MAS's task performance, achieving an average accuracy gain of 6.3 percentage points on math benchmarks. Furthermore, the system exhibits robust generalization and adaptivity, dynamically modulating rectification efforts based on task difficulty while leveraging context-aware indicators to resolve a wide spectrum of error patterns. Our code and dataset are released at this https URL.

**arXiv ID:** 2602.23258
</details>

<details>
<summary><strong>Toward Expert Investment Teams:A Multi-Agent LLM System with Fine-Grained Trading Tasks</strong> - Kunihiro Miyazaki, Takanobu Kawahara, Stephen Roberts, Stefan Zohren - [[pdf]](https://arxiv.org/pdf/2602.23330)</summary>

**Abstract:** The advancement of large language models (LLMs) has accelerated the development of autonomous financial trading systems. While mainstream approaches deploy multi-agent systems mimicking analyst and manager roles, they often rely on abstract instructions that overlook the intricacies of real-world workflows, which can lead to degraded inference performance and less transparent decision-making. Therefore, we propose a multi-agent LLM trading framework that explicitly decomposes investment analysis into fine-grained tasks, rather than providing coarse-grained instructions. We evaluate the proposed framework using Japanese stock data, including prices, financial statements, news, and macro information, under a leakage-controlled backtesting setting. Experimental results show that fine-grained task decomposition significantly improves risk-adjusted returns compared to conventional coarse-grained designs. Crucially, further analysis of intermediate agent outputs suggests that alignment between analytical outputs and downstream decision preferences is a critical driver of system performance. Moreover, we conduct standard portfolio optimization, exploiting low correlation with the stock index and the variance of each system's output. This approach achieves superior performance. These findings contribute to the design of agent structure and task configuration when applying LLM agents to trading systems in practical settings.

**arXiv ID:** 2602.23330
</details>

<details>
<summary><strong>TherapyProbe: Generating Design Knowledge for Relational Safety in Mental Health Chatbots Through Adversarial Simulation</strong> - Joydeep Chandra, Satyam Kumar Navneet, Yong Zhang - [[pdf]](https://arxiv.org/pdf/2602.22775)</summary>

**Abstract:** As mental health chatbots proliferate to address the global treatment gap, a critical question emerges: How do we design for relational safety the quality of interaction patterns that unfold across conversations rather than the correctness of individual responses? Current safety evaluations assess single-turn crisis responses, missing the therapeutic dynamics that determine whether chatbots help or harm over time. We introduce TherapyProbe, a design probe methodology that generates actionable design knowledge by systematically exploring chatbot conversation trajectories through adversarial multi-agent simulation. Using open-source models, TherapyProbe surfaces relational safety failures interaction patterns like "validation spirals" where chatbots progressively reinforce hopelessness, or "empathy fatigue" where responses become mechanical over turns. Our contribution is translating these failures into a Safety Pattern Library of 23 failure archetypes with corresponding design recommendations. We contribute: (1) a replicable methodology requiring no API costs, (2) a clinically-grounded failure taxonomy, and (3) design implications for developers, clinicians, and policymakers.

**arXiv ID:** 2602.22775
</details>

<details>
<summary><strong>QSIM: Mitigating Overestimation in Multi-Agent Reinforcement Learning via Action Similarity Weighted Q-Learning</strong> - Yuanjun Li, Bin Zhang, Hao Chen, Zhouyang Jiang, Dapeng Li, Zhiwei Xu - [[pdf]](https://arxiv.org/pdf/2602.22786)</summary>

**Abstract:** Value decomposition (VD) methods have achieved remarkable success in cooperative multi-agent reinforcement learning (MARL). However, their reliance on the max operator for temporal-difference (TD) target calculation leads to systematic Q-value overestimation. This issue is particularly severe in MARL due to the combinatorial explosion of the joint action space, which often results in unstable learning and suboptimal policies. To address this problem, we propose QSIM, a similarity weighted Q-learning framework that reconstructs the TD target using action similarity. Instead of using the greedy joint action directly, QSIM forms a similarity weighted expectation over a structured near-greedy joint action space. This formulation allows the target to integrate Q-values from diverse yet behaviorally related actions while assigning greater influence to those that are more similar to the greedy choice. By smoothing the target with structurally relevant alternatives, QSIM effectively mitigates overestimation and improves learning stability. Extensive experiments demonstrate that QSIM can be seamlessly integrated with various VD methods, consistently yielding superior performance and stability compared to the original algorithms. Furthermore, empirical analysis confirms that QSIM significantly mitigates the systematic value overestimation in MARL. Code is available at this https URL.

**arXiv ID:** 2602.22786
</details>

<details>
<summary><strong>Mastering Multi-Drone Volleyball through Hierarchical Co-Self-Play Reinforcement Learning</strong> - Ruize Zhang, Sirui Xiang, Zelai Xu, Feng Gao, Shilong Ji, Wenhao Tang, Wenbo Ding, Chao Yu, Yu Wang - [[pdf]](https://arxiv.org/pdf/2505.04317)</summary>

**Abstract:** In this paper, we tackle the problem of learning to play 3v3 multi-drone volleyball, a new embodied competitive task that requires both high-level strategic coordination and low-level agile control. The task is turn-based, multi-agent, and physically grounded, posing significant challenges due to its long-horizon dependencies, tight inter-agent coupling, and the underactuated dynamics of quadrotors. To address this, we propose Hierarchical Co-Self-Play (HCSP), a hierarchical reinforcement learning framework that separates centralized high-level strategic decision-making from decentralized low-level motion control. We design a three-stage population-based training pipeline to enable both strategy and skill to emerge from scratch without expert demonstrations: (I) training diverse low-level skills, (II) learning high-level strategy via self-play with fixed low-level skills, and (III) joint fine-tuning through co-self-play. Experiments show that HCSP achieves superior performance, outperforming non-hierarchical self-play and rule-based hierarchical baselines with an average 82.9% win rate and a 71.5% win rate against the two-stage variant. Moreover, co-self-play leads to emergent team behaviors such as role switching and coordinated formations, demonstrating the effectiveness of our hierarchical design and training scheme. The project page is at this https URL.

**arXiv ID:** 2505.04317
</details>

<details>
<summary><strong>Sustainable Multi-Agent Crowdsourcing via Physics-Informed Bandits</strong> - Chayan Banerjee - [[pdf]](https://arxiv.org/pdf/2602.22365)</summary>

**Abstract:** Crowdsourcing platforms face a four-way tension between allocation quality, workforce sustainability, operational feasibility, and strategic contractor behaviour--a dilemma we formalise as the Cold-Start, Burnout, Utilisation, and Strategic Agency Dilemma. Existing methods resolve at most two of these tensions simultaneously: greedy heuristics and multi-criteria decision making (MCDM) methods achieve Day-1 quality but cause catastrophic burnout, while bandit algorithms eliminate burnout only through operationally infeasible 100% workforce this http URL address this, we introduce FORGE, a physics-grounded $K+1$ multi-agent simulator in which each contractor is a rational agent that declares its own load-acceptance threshold based on its fatigue state, converting the standard passive Restless Multi-Armed Bandit (RMAB) into a genuine Stackelberg game. Operating within FORGE, we propose a Neural-Linear UCB allocator that fuses a Two-Tower embedding network with a Physics-Informed Covariance Prior derived from offline simulator interactions. The prior simultaneously warm-starts skill-cluster geometry and UCB exploration landscape, providing a geometry-aware belief state from episode 1 that measurably reduces cold-start this http URL $T = 200$ cold-start episodes, the proposed method achieves the highest reward of all non-oracle methods ($\text{LRew} = 0.555 \pm 0.041$) at only 7.6% workforce utilisation--a combination no conventional baseline achieves--while maintaining robustness to workforce turnover up to 50% and observation noise up to $\sigma = 0.20$.

**arXiv ID:** 2602.22365
</details>

<details>
<summary><strong>Robust Information Design for Multi-Agent Systems with Complementarities: Smallest-Equilibrium Threshold Policies</strong> - Farzaneh Farhadi, Maria Chli - [[pdf]](https://arxiv.org/pdf/2602.22915)</summary>

**Abstract:** We study information design in multi-agent systems (MAS) with binary actions and strategic complementarities, where an external designer influences behavior only through signals. Agents play the smallest-equilibrium of the induced Bayesian game, reflecting conservative, coordination-averse behavior typical in distributed systems. We show that when utilities admit a convex potential and welfare is convex, the robustly implementable optimum has a remarkably simple form: perfect coordination at each state: either everyone acts or no one does. We provide a constructive threshold rule: compute a one-dimensional score for each state, sort states, and pick a single threshold (with a knife-edge lottery for at most one state). This rule is an explicit optimal vertex of a linear program (LP) characterized by feasibility and sequential obedience constraints. Empirically, in both vaccination and technology-adoption domains, our constructive policy matches LP optima, scales as $O(|\Theta|\log|\Theta|)$, and avoids the inflated welfare predicted by obedience-only designs that assume the designer can dictate the (best) equilibrium. The result is a general, scalable recipe for robust coordination in MAS with complementarities.

**arXiv ID:** 2602.22915
</details>

<details>
<summary><strong>HyperAgent: Leveraging Hypergraphs for Topology Optimization in Multi-Agent Communication</strong> - Heng Zhang, Yuling Shi, Xiaodong Gu, Zijian Zhang, Haochen You, Lubin Gan, Yilei Yuan, Jin Huang - [[pdf]](https://arxiv.org/pdf/2510.10611)</summary>

**Abstract:** Recent advances in large language model-powered multi-agent systems have demonstrated remarkable collective intelligence through effective communication. However, existing approaches face two primary challenges: (i) \textit{Ineffective group collaboration modeling}, as they rely on pairwise edge representations in graph structures, limiting their ability to capture relationships among multiple agents; and (ii) \textit{Limited task-adaptiveness in communication topology design}, leading to excessive communication cost for simple tasks and insufficient coordination for complex scenarios. These issues restrict the scalability and practical deployment of adaptive collaboration frameworks. To address these challenges, we propose \textbf{HyperAgent}, a hypergraph-based framework that optimizes communication topologies and effectively captures group collaboration patterns using direct hyperedge representations. Unlike edge-based approaches, HyperAgent uses hyperedges to link multiple agents within the same subtask and employs hypergraph convolutional layers to achieve one-step information aggregation in collaboration groups. Additionally, it incorporates a variational autoencoder framework with sparsity regularization to dynamically adjust hypergraph topologies based on task complexity. Experiments highlight the superiority of HyperAgent in both performance and efficiency. For instance, on GSM8K, HyperAgent achieves 95.07\% accuracy while reducing token consumption by 25.33\%, demonstrating the potential of hypergraph-based optimization for multi-agent communication.

**arXiv ID:** 2510.10611
</details>

<details>
<summary><strong>Hierarchical LLM-Based Multi-Agent Framework with Prompt Optimization for Multi-Robot Task Planning</strong> - Tomoya Kawabe, Rin Takano - [[pdf]](https://arxiv.org/pdf/2602.21670)</summary>

**Abstract:** Multi-robot task planning requires decomposing natural-language instructions into executable actions for heterogeneous robot teams. Conventional Planning Domain Definition Language (PDDL) planners provide rigorous guarantees but struggle to handle ambiguous or long-horizon missions, while large language models (LLMs) can interpret instructions and propose plans but may hallucinate or produce infeasible actions. We present a hierarchical multi-agent LLM-based planner with prompt optimization: an upper layer decomposes tasks and assigns them to lower-layer agents, which generate PDDL problems solved by a classical planner. When plans fail, the system applies TextGrad-inspired textual-gradient updates to optimize each agent's prompt and thereby improve planning accuracy. In addition, meta-prompts are learned and shared across agents within the same layer, enabling efficient prompt optimization in multi-agent settings. On the MAT-THOR benchmark, our planner achieves success rates of 0.95 on compound tasks, 0.84 on complex tasks, and 0.60 on vague tasks, improving over the previous state-of-the-art LaMMA-P by 2, 7, and 15 percentage points respectively. An ablation study shows that the hierarchical structure, prompt optimization, and meta-prompt sharing contribute roughly +59, +37, and +4 percentage points to the overall success rate.

**arXiv ID:** 2602.21670
</details>

<details>
<summary><strong>Multi-agent imitation learning with function approximation: Linear Markov games and beyond</strong> - Luca Viano, Till Freihaut, Emanuele Nevali, Volkan Cevher, Matthieu Geist, Giorgia Ramponi - [[pdf]](https://arxiv.org/pdf/2602.22810)</summary>

**Abstract:** In this work, we present the first theoretical analysis of multi-agent imitation learning (MAIL) in linear Markov games where both the transition dynamics and each agent's reward function are linear in some given features. We demonstrate that by leveraging this structure, it is possible to replace the state-action level "all policy deviation concentrability coefficient" (Freihaut et al., arXiv:2510.09325) with a concentrability coefficient defined at the feature level which can be much smaller than the state-action analog when the features are informative about states' similarity. Furthermore, to circumvent the need for any concentrability coefficient, we turn to the interactive setting. We provide the first, computationally efficient, interactive MAIL algorithm for linear Markov games and show that its sample complexity depends only on the dimension of the feature map $d$. Building on these theoretical findings, we propose a deep MAIL interactive algorithm which clearly outperforms BC on games such as Tic-Tac-Toe and Connect4.

**arXiv ID:** 2602.22810
</details>

<details>
<summary><strong>Pixel2Catch: Multi-Agent Sim-to-Real Transfer for Agile Manipulation with a Single RGB Camera</strong> - Seongyong Kim, Junhyeon Cho, Kang-Won Lee, Soo-Chul Lim - [[pdf]](https://arxiv.org/pdf/2602.22733)</summary>

**Abstract:** To catch a thrown object, a robot must be able to perceive the object's motion and generate control actions in a timely manner. Rather than explicitly estimating the object's 3D position, this work focuses on a novel approach that recognizes object motion using pixel-level visual information extracted from a single RGB image. Such visual cues capture changes in the object's position and scale, allowing the policy to reason about the object's motion. Furthermore, to achieve stable learning in a high-DoF system composed of a robot arm equipped with a multi-fingered hand, we design a heterogeneous multi-agent reinforcement learning framework that defines the arm and hand as independent agents with distinct roles. Each agent is trained cooperatively using role-specific observations and rewards, and the learned policies are successfully transferred from simulation to the real world.

**arXiv ID:** 2602.22733
</details>

<details>
<summary><strong>WaterVideoQA: ASV-Centric Perception and Rule-Compliant Reasoning via Multi-Modal Agents</strong> - Runwei Guan, Shaofeng Liang, Ningwei Ouyang, Weichen Fei, Shanliang Yao, Wei Dai, Chenhao Ge, Penglei Sun, Xiaohui Zhu, Tao Huang, Ryan Wen Liu, Hui Xiong - [[pdf]](https://arxiv.org/pdf/2602.22923)</summary>

**Abstract:** While autonomous navigation has achieved remarkable success in passive perception (e.g., object detection and segmentation), it remains fundamentally constrained by a void in knowledge-driven, interactive environmental cognition. In the high-stakes domain of maritime navigation, the ability to bridge the gap between raw visual perception and complex cognitive reasoning is not merely an enhancement but a critical prerequisite for Autonomous Surface Vessels to execute safe and precise maneuvers. To this end, we present WaterVideoQA, the first large-scale, comprehensive Video Question Answering benchmark specifically engineered for all-waterway environments. This benchmark encompasses 3,029 video clips across six distinct waterway categories, integrating multifaceted variables such as volatile lighting and dynamic weather to rigorously stress-test ASV capabilities across a five-tier hierarchical cognitive framework. Furthermore, we introduce NaviMind, a pioneering multi-agent neuro-symbolic system designed for open-ended maritime reasoning. By synergizing Adaptive Semantic Routing, Situation-Aware Hierarchical Reasoning, and Autonomous Self-Reflective Verification, NaviMind transitions ASVs from superficial pattern matching to regulation-compliant, interpretable decision-making. Experimental results demonstrate that our framework significantly transcends existing baselines, establishing a new paradigm for intelligent, trustworthy interaction in dynamic maritime environments.

**arXiv ID:** 2602.22923
</details>

</details>

<details open>
<summary><h2>Other Agent Research (8 papers)</h2></summary>

<details>
<summary><strong>Epistemic Filtering and Collective Hallucination: A Jury Theorem for Confidence-Calibrated Agents</strong> - Jonas Karge - [[pdf]](https://arxiv.org/pdf/2602.22413)</summary>

**Abstract:** We investigate the collective accuracy of heterogeneous agents who learn to estimate their own reliability over time and selectively abstain from voting. While classical epistemic voting results, such as the \textit{Condorcet Jury Theorem} (CJT), assume fixed participation, real-world aggregation often benefits from allowing agents to say ``I don't know.'' We propose a probabilistic framework where agents engage in a \textit{calibration} phase, updating beliefs about their own fixed competence, before facing a final confidence gate that determines whether to vote or abstain. We derive a non-asymptotic lower bound on the group's success probability and prove that this \textit{selective participation} generalizes the asymptotic guarantees of the CJT to a sequential, confidence-gated setting. Empirically, we validate these bounds via Monte Carlo simulations. While our results are general, we discuss their potential application to AI safety, outlining how this framework can mitigate \textit{hallucinations} in collective LLM decision-making.

**arXiv ID:** 2602.22413
</details>

<details>
<summary><strong>ArchAgent: Agentic AI-driven Computer Architecture Discovery</strong> - Raghav Gupta, Akanksha Jain, Abraham Gonzalez, Alexander Novikov, Po-Sen Huang, Matej Balog, Marvin Eisenberger, Sergey Shirobokov, Ngân Vũ, Martin Dixon, Borivoje Nikolić, Parthasarathy Ranganathan, Sagar Karandikar - [[pdf]](https://arxiv.org/pdf/2602.22425)</summary>

**Abstract:** Agile hardware design flows are a critically needed force multiplier to meet the exploding demand for compute. Recently, agentic generative AI systems have demonstrated significant advances in algorithm design, improving code efficiency, and enabling discovery across scientific domains.
Bridging these worlds, we present ArchAgent, an automated computer architecture discovery system built on AlphaEvolve. We show ArchAgent's ability to automatically design/implement state-of-the-art (SoTA) cache replacement policies (architecting new mechanisms/logic, not only changing parameters), broadly within the confines of an established cache replacement policy design competition.
In two days without human intervention, ArchAgent generated a policy achieving a 5.3% IPC speedup improvement over the prior SoTA on public multi-core Google Workload Traces. On the heavily-explored single-core SPEC06 workloads, it generated a policy in just 18 days showing a 0.9% IPC speedup improvement over the existing SoTA (a similar "winning margin" as reported by the existing SoTA). ArchAgent achieved these gains 3-5x faster than prior human-developed SoTA policies.
Agentic flows also enable "post-silicon hyperspecialization" where agents tune runtime-configurable parameters exposed in hardware policies to further align the policies with a specific workload (mix). Exploiting this, we demonstrate a 2.4% IPC speedup improvement over prior SoTA on SPEC06 workloads.
Finally, we outline broader implications for computer architecture research in the era of agentic AI. For example, we demonstrate the phenomenon of "simulator escapes", where the agentic AI flow discovered and exploited a loophole in a popular microarchitectural simulator - a consequence of the fact that these research tools were designed for a (now past) world where they were exclusively operated by humans acting in good-faith.

**arXiv ID:** 2602.22425
</details>

<details>
<summary><strong>Cognitive Models and AI Algorithms Provide Templates for Designing Language Agents</strong> - Ryan Liu, Dilip Arumugam, Cedegao E. Zhang, Sean Escola, Xaq Pitkow, Thomas L. Griffiths - [[pdf]](https://arxiv.org/pdf/2602.22523)</summary>

**Abstract:** While contemporary large language models (LLMs) are increasingly capable in isolation, there are still many difficult problems that lie beyond the abilities of a single LLM. For such tasks, there is still uncertainty about how best to take many LLMs as parts and combine them into a greater whole. This position paper argues that potential blueprints for designing such modular language agents can be found in the existing literature on cognitive models and artificial intelligence (AI) algorithms. To make this point clear, we formalize the idea of an agent template that specifies roles for individual LLMs and how their functionalities should be composed. We then survey a variety of existing language agents in the literature and highlight their underlying templates derived directly from cognitive models or AI algorithms. By highlighting these designs, we aim to call attention to agent templates inspired by cognitive science and AI as a powerful tool for developing effective, interpretable language agents.

**arXiv ID:** 2602.22523
</details>

<details>
<summary><strong>When Should an AI Act? A Human-Centered Model of Scene, Context, and Behavior for Agentic AI Design</strong> - Soyoung Jung, Daehoo Yoon, Sung Gyu Koh, Young Hwan Kim, Yehan Ahn, Sung Park - [[pdf]](https://arxiv.org/pdf/2602.22814)</summary>

**Abstract:** Agentic AI increasingly intervenes proactively by inferring users' situations from contextual data yet often fails for lack of principled judgment about when, why, and whether to act. We address this gap by proposing a conceptual model that reframes behavior as an interpretive outcome integrating Scene (observable situation), Context (user-constructed meaning), and Human Behavior Factors (determinants shaping behavioral likelihood). Grounded in multidisciplinary perspectives across the humanities, social sciences, HCI, and engineering, the model separates what is observable from what is meaningful to the user and explains how the same scene can yield different behavioral meanings and outcomes. To translate this lens into design action, we derive five agent design principles (behavioral alignment, contextual sensitivity, temporal appropriateness, motivational calibration, and agency preservation) that guide intervention depth, timing, intensity, and restraint. Together, the model and principles provide a foundation for designing agentic AI systems that act with contextual sensitivity and judgment in interactions.

**arXiv ID:** 2602.22814
</details>

<details>
<summary><strong>Training Agents to Self-Report Misbehavior</strong> - Bruce W. Lee, Chen Yueh-Han, Tomek Korbak - [[pdf]](https://arxiv.org/pdf/2602.22303)</summary>

**Abstract:** Frontier AI agents may pursue hidden goals while concealing their pursuit from oversight. Alignment training aims to prevent such behavior by reinforcing the correct goals, but alignment may not always succeed and can lead to unwanted side effects. We propose self-incrimination training, which instead trains agents to produce a visible signal when they covertly misbehave. We train GPT-4.1 and Gemini-2.0 agents to call a report_scheming() tool when behaving deceptively and measure their ability to cause harm undetected in out-of-distribution environments. Self-incrimination significantly reduces the undetected successful attack rate, outperforming matched-capability monitors and alignment baselines while preserving instruction hierarchy and incurring minimal safety tax on general capabilities. Unlike blackbox monitoring, self-incrimination performance is consistent across tasks regardless of how suspicious the misbehavior appears externally. The trained behavior persists under adversarial prompt optimization and generalizes to settings where agents pursue misaligned goals themselves rather than being instructed to misbehave. Our results suggest self-incrimination offers a viable path for reducing frontier misalignment risk, one that neither assumes misbehavior can be prevented nor that it can be reliably classified from the outside.

**arXiv ID:** 2602.22303
</details>

<details>
<summary><strong>ClawMobile: Rethinking Smartphone-Native Agentic Systems</strong> - Hongchao Du, Shangyu Wu, Qiao Li, Riwei Pan, Jinheng Li, Youcheng Sun, Chun Jason Xue - [[pdf]](https://arxiv.org/pdf/2602.22942)</summary>

**Abstract:** Smartphones represent a uniquely challenging environment for agentic systems. Unlike cloud or desktop settings, mobile devices combine constrained execution contexts, fragmented control interfaces, and rapidly changing application states. As large language models (LLMs) evolve from conversational assistants to action-oriented agents, achieving reliable smartphone-native autonomy requires rethinking how reasoning and control are composed.
We introduce ClawMobile as a concrete exploration of this design space. ClawMobile adopts a hierarchical architecture that separates high-level language reasoning from structured, deterministic control pathways, improving execution stability and reproducibility on real devices. Using ClawMobile as a case study, we distill the design principles for mobile LLM runtimes and identify key challenges in efficiency, adaptability, and stability. We argue that building robust smartphone-native agentic systems demands principled coordination between probabilistic planning and deterministic system interfaces. The implementation is open-sourced~\footnote{this https URL} to facilitate future exploration.

**arXiv ID:** 2602.22942
</details>

<details>
<summary><strong>Deep ensemble graph neural networks for probabilistic cosmic-ray direction and energy reconstruction in autonomous radio arrays</strong> - Arsène Ferrière, Aurélien Benoit-Lévy, Olivier Martineau-Huynh, Matías Tueros - [[pdf]](https://arxiv.org/pdf/2602.23321)</summary>

**Abstract:** Using advanced machine learning techniques, we developed a method for reconstructing precisely the arrival direction and energy of ultra-high-energy cosmic rays from the voltage traces they induced on ground-based radio detector arrays.
In our approach, triggered antennas are represented as a graph structure, which serves as input for a graph neural network (GNN). By incorporating physical knowledge into both the GNN architecture and the input data, we improve the precision and reduce the required size of the training set with respect to a fully data-driven approach. This method achieves an angular resolution of 0.092° and an electromagnetic energy reconstruction resolution of 16.4% on simulated data with realistic noise conditions.
We also employ uncertainty estimation methods to enhance the reliability of our predictions, quantifying the confidence of the GNN's outputs and providing confidence intervals for both direction and energy reconstruction. Finally, we investigate strategies to verify the model's consistency and robustness under real life variations, with the goal of identifying scenarios in which predictions remain reliable despite domain shifts between simulation and reality.

**arXiv ID:** 2602.23321
</details>

<details>
<summary><strong>Interface Framework for Human-AI Collaboration within Intelligent User Interface Ecosystems</strong> - Shruthi Andru, Shrut Kirti Saksena - [[pdf]](https://arxiv.org/pdf/2602.22343)</summary>

**Abstract:** As interfaces evolve from static user pathways to dynamic human-AI collaboration, no standard methods exist for selecting appropriate interface patterns based on user needs and task complexity. Existing frameworks only provide guiding principles for designing AI agent capabilities. We propose a dimensional framework based on workflow complexity, AI autonomy, and AI reasoning to guide the design of context-aware, scalable AI interfaces aka modalities (e.g., prompt bars, split screens, full screens, etc.). The framework was developed through co-design workshops with designers of marketing products and refined through qualitative research with eight long-term AI users. The study evaluated the three dimensions, identified task-to-interface relationships, and surfaced the importance of both business impact and security risk across all high-autonomy scenarios. This framework provides product teams with a shared language to develop scalable AI interfaces, emphasizing fluidity between interfaces and progressive user control to balance AI autonomy with human oversight.

**arXiv ID:** 2602.22343
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (23 papers)</h2></summary>

<details>
<summary><strong>CWM: Contrastive World Models for Action Feasibility Learning in Embodied Agent Pipelines</strong> - Chayan Banerjee - [[pdf]](https://arxiv.org/pdf/2602.22452)</summary>

**Abstract:** A reliable action feasibility scorer is a critical bottleneck in embodied agent pipelines: before any planning or reasoning occurs, the agent must identify which candidate actions are physically executable in the current state. Existing approaches use supervised fine-tuning (SFT) to train action scorers, but SFT treats each candidate independently and does not explicitly teach the model to discriminate between actions that are physically correct and those that are subtly wrong. We propose the Contrastive World Model (CWM), which fine-tunes a large language model (LLM) as an action scorer using an InfoNCE contrastive objective with hard-mined negative examples. The key idea is to push valid actions away from invalid ones in scoring space, with special emphasis on hard negatives: semantically similar but physically incompatible candidates. We evaluate CWM on the ScienceWorld benchmark through two studies. First, an intrinsic affordance evaluation on 605 hard-negative test pairs shows that CWM outperforms SFT by +6.76 percentage points on Precision@1 for minimal-edit negatives -- cases where a single word changes the physical outcome -- and achieves a higher AUC-ROC (0.929 vs. 0.906). Second, a live filter characterisation study measures how well CWM ranks gold-path actions against all valid environment actions during task execution. Under out-of-distribution stress conditions, CWM maintains a significantly better safety margin (-2.39) than SFT (-3.96), indicating that the gold action is ranked closer to the top. These results support the hypothesis that contrastive training induces representations that capture physical feasibility more faithfully than SFT alone.

**arXiv ID:** 2602.22452
</details>

<details>
<summary><strong>VeRO: An Evaluation Harness for Agents to Optimize Agents</strong> - Varun Ursekar, Apaar Shanker, Veronica Chatrath, Yuan, Sam Denton - [[pdf]](https://arxiv.org/pdf/2602.22480)</summary>

**Abstract:** An important emerging application of coding agents is agent optimization: the iterative improvement of a target agent through edit-execute-evaluate cycles. Despite its relevance, the community lacks a systematic understanding of coding agent performance on this task. Agent optimization differs fundamentally from conventional software engineering: the target agent interleaves deterministic code with stochastic LLM completions, requiring structured capture of both intermediate reasoning and downstream execution outcomes. To address these challenges, we introduce VERO (Versioning, Rewards, and Observations), which provides (1) a reproducible evaluation harness with versioned agent snapshots, budget-controlled evaluation, and structured execution traces, and (2) a benchmark suite of target agents and tasks with reference evaluation procedures. Using VERO, we conduct an empirical study comparing optimizer configurations across tasks and analyzing which modifications reliably improve target agent performance. We release VERO to support research on agent optimization as a core capability for coding agents.

**arXiv ID:** 2602.22480
</details>

<details>
<summary><strong>SideQuest: Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning</strong> - Sanjay Kariyappa, G. Edward Suh - [[pdf]](https://arxiv.org/pdf/2602.22603)</summary>

**Abstract:** Long-running agentic tasks, such as deep research, require multi-hop reasoning over information distributed across multiple webpages and documents. In such tasks, the LLM context is dominated by tokens from external retrieval, causing memory usage to grow rapidly and limiting decode performance. While several KV cache compression techniques exist for long-context inputs, we find that existing heuristics fail to support multi-step reasoning models effectively. We address this challenge with SideQuest -- a novel approach that leverages the Large Reasoning Model (LRM) itself to perform KV cache compression by reasoning about the usefulness of tokens in its context. To prevent the tokens associated with this management process from polluting the model's memory, we frame KV cache compression as an auxiliary task executed in parallel to the main reasoning task. Our evaluations, using a model trained with just 215 samples, show that SideQuest reduces peak token usage by up to 65% on agentic tasks with minimal degradation in accuracy, outperforming heuristic-based KV cache compression techniques.

**arXiv ID:** 2602.22603
</details>

<details>
<summary><strong>MobilityBench: A Benchmark for Evaluating Route-Planning Agents in Real-World Mobility Scenarios</strong> - Zhiheng Song, Jingshuai Zhang, Chuan Qin, Chao Wang, Chao Chen, Longfei Xu, Kaikui Liu, Xiangxiang Chu, Hengshu Zhu - [[pdf]](https://arxiv.org/pdf/2602.22638)</summary>

**Abstract:** Route-planning agents powered by large language models (LLMs) have emerged as a promising paradigm for supporting everyday human mobility through natural language interaction and tool-mediated decision making. However, systematic evaluation in real-world mobility settings is hindered by diverse routing demands, non-deterministic mapping services, and limited reproducibility. In this study, we introduce MobilityBench, a scalable benchmark for evaluating LLM-based route-planning agents in real-world mobility scenarios. MobilityBench is constructed from large-scale, anonymized real user queries collected from Amap and covers a broad spectrum of route-planning intents across multiple cities worldwide. To enable reproducible, end-to-end evaluation, we design a deterministic API-replay sandbox that eliminates environmental variance from live services. We further propose a multi-dimensional evaluation protocol centered on outcome validity, complemented by assessments of instruction understanding, planning, tool use, and efficiency. Using MobilityBench, we evaluate multiple LLM-based route-planning agents across diverse real-world mobility scenarios and provide an in-depth analysis of their behaviors and performance. Our findings reveal that current models perform competently on Basic information retrieval and Route Planning tasks, yet struggle considerably with Preference-Constrained Route Planning, underscoring significant room for improvement in personalized mobility applications. We publicly release the benchmark data, evaluation toolkit, and documentation at this https URL .

**arXiv ID:** 2602.22638
</details>

<details>
<summary><strong>MiroFlow: Towards High-Performance and Robust Open-Source Agent Framework for General Deep Research Tasks</strong> - Shiqian Su, Sen Xing, Xuan Dong, Muyan Zhong, Bin Wang, Xizhou Zhu, Yuntao Chen, Wenhai Wang, Yue Deng, Pengxiang Zhu, Ziyuan Liu, Tiantong Li, Jiaheng Yu, Zhe Chen, Lidong Bing, Jifeng Dai - [[pdf]](https://arxiv.org/pdf/2602.22808)</summary>

**Abstract:** Despite the remarkable progress of large language models (LLMs), the capabilities of standalone LLMs have begun to plateau when tackling real-world, complex tasks that require interaction with external tools and dynamic environments. Although recent agent frameworks aim to enhance model autonomy through tool integration and external interaction, they still suffer from naive workflows, unstable performance, limited support across diverse benchmarks and tasks, and heavy reliance on costly commercial APIs. In this work, we propose a high-performance and robust open-source agent framework, termed MiroFlow, which incorporates an agent graph for flexible orchestration, an optional deep reasoning mode to enhance performance, and a robust workflow execution to ensure stable and reproducible performance. Extensive experiments demonstrate that MiroFlow consistently achieves state-of-the-art performance across multiple agent benchmarks, including GAIA, BrowseComp-EN/ZH, HLE, xBench-DeepSearch, and notably FutureX. We hope it could serve as an easily accessible, reproducible, and comparable baseline for the deep research community.

**arXiv ID:** 2602.22808
</details>

<details>
<summary><strong>DeepPresenter: Environment-Grounded Reflection for Agentic Presentation Generation</strong> - Hao Zheng, Guozhao Mo, Xinru Yan, Qianhao Yuan, Wenkai Zhang, Xuanang Chen, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun - [[pdf]](https://arxiv.org/pdf/2602.22839)</summary>

**Abstract:** Presentation generation requires deep content research, coherent visual design, and iterative refinement based on observation. However, existing presentation agents often rely on predefined workflows and fixed templates. To address this, we present DeepPresenter, an agentic framework that adapts to diverse user intents, enables effective feedback-driven refinement, and generalizes beyond a scripted pipeline. Specifically, DeepPresenter autonomously plans, renders, and revises intermediate slide artifacts to support long-horizon refinement with environmental observations. Furthermore, rather than relying on self-reflection over internal signals (e.g., reasoning traces), our environment-grounded reflection conditions the generation process on perceptual artifact states (e.g., rendered slides), enabling the system to identify and correct presentation-specific issues during execution. Results on the evaluation set covering diverse presentation-generation scenarios show that DeepPresenter achieves state-of-the-art performance, and the fine-tuned 9B model remains highly competitive at substantially lower cost. Our project is available at: this https URL

**arXiv ID:** 2602.22839
</details>

<details>
<summary><strong>FactGuard: Agentic Video Misinformation Detection via Reinforcement Learning</strong> - Zehao Li, Hongwei Yu, Hao Jiang, Qiang Sheng, Yilong Xu, Baolong Bi, Yang Li, Zhenlong Yuan, Yujun Cai, Zhaoqi Wang - [[pdf]](https://arxiv.org/pdf/2602.22963)</summary>

**Abstract:** Multimodal large language models (MLLMs) have substantially advanced video misinformation detection through unified multimodal reasoning, but they often rely on fixed-depth inference and place excessive trust in internally generated assumptions, particularly in scenarios where critical evidence is sparse, fragmented, or requires external verification. To address these limitations, we propose FactGuard, an agentic framework for video misinformation detection that formulates verification as an iterative reasoning process built upon MLLMs. FactGuard explicitly assesses task ambiguity and selectively invokes external tools to acquire critical evidence, enabling progressive refinement of reasoning trajectories. To further strengthen this capability, we introduce a two-stage training strategy that combines domain-specific agentic supervised fine-tuning with decision-aware reinforcement learning to optimize tool usage and calibrate risk-sensitive decision making. Extensive experiments on FakeSV, FakeTT, and FakeVV demonstrate FactGuard's state-of-the-art performance and validate its excellent robustness and generalization capacity.

**arXiv ID:** 2602.22963
</details>

<details>
<summary><strong>ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Control and Mechanism-Linked Consciousness Indicator Assays</strong> - Aishik Sanyal - [[pdf]](https://arxiv.org/pdf/2602.23232)</summary>

**Abstract:** Indicator-based approaches to machine consciousness recommend mechanism-linked evidence triangulated across tasks, supported by architectural inspection and causal intervention. Inspired by Humphrey's ipsundrum hypothesis, we implement ReCoN-Ipsundrum, an inspectable agent that extends a ReCoN state machine with a recurrent persistence loop over sensory salience Ns and an optional affect proxy reporting valence/arousal. Across fixed-parameter ablations (ReCoN, Ipsundrum, Ipsundrum+affect), we operationalize Humphrey's qualiaphilia (preference for sensory experience for its own sake) as a familiarity-controlled scenic-over-dull route choice. We find a novelty dissociation: non-affect variants are novelty-sensitive (Delta scenic-entry = 0.07). Affect coupling is stable (Delta scenic-entry = 0.01) even when scenic is less novel (median Delta novelty ~ -0.43). In reward-free exploratory play, the affect variant shows structured local investigation (scan events 31.4 vs. 0.9; cycle score 7.6). In a pain-tail probe, only the affect variant sustains prolonged planned caution (tail duration 90 vs. 5). Lesioning feedback+integration selectively reduces post-stimulus persistence in ipsundrum variants (AUC drop 27.62, 27.9%) while leaving ReCoN unchanged. These dissociations link recurrence -> persistence and affect-coupled control -> preference stability, scanning, and lingering caution, illustrating how indicator-like signatures can be engineered and why mechanistic and causal evidence should accompany behavioral markers.

**arXiv ID:** 2602.23232
</details>

<details>
<summary><strong>Evaluating Stochasticity in Deep Research Agents</strong> - Haotian Zhai, Elias Stengel-Eskin, Pratik Patil, Liu Leqi - [[pdf]](https://arxiv.org/pdf/2602.23271)</summary>

**Abstract:** Deep Research Agents (DRAs) are promising agentic systems that gather and synthesize information to support research across domains such as financial decision-making, medical analysis, and scientific discovery. Despite recent improvements in research quality (e.g., outcome accuracy when ground truth is available), DRA system design often overlooks a critical barrier to real-world deployment: stochasticity. Under identical queries, repeated executions of DRAs can exhibit substantial variability in terms of research outcome, findings, and citations. In this paper, we formalize the study of stochasticity in DRAs by modeling them as information acquisition Markov Decision Processes. We introduce an evaluation framework that quantifies variance in the system and identify three sources of it: information acquisition, information compression, and inference. Through controlled experiments, we investigate how stochasticity from these modules across different decision steps influences the variance of DRA outputs. Our results show that reducing stochasticity can improve research output quality, with inference and early-stage stochasticity contributing the most to DRA output variance. Based on these findings, we propose strategies for mitigating stochasticity while maintaining output quality via structured output and ensemble-based query generation. Our experiments on DeepSearchQA show that our proposed mitigation methods reduce average stochasticity by 22% while maintaining high research quality.

**arXiv ID:** 2602.23271
</details>

<details>
<summary><strong>CXReasonAgent: Evidence-Grounded Diagnostic Reasoning Agent for Chest X-rays</strong> - Hyungyung Lee, Hangyul Yoon, Edward Choi - [[pdf]](https://arxiv.org/pdf/2602.23276)</summary>

**Abstract:** Chest X-ray plays a central role in thoracic diagnosis, and its interpretation inherently requires multi-step, evidence-grounded reasoning. However, large vision-language models (LVLMs) often generate plausible responses that are not faithfully grounded in diagnostic evidence and provide limited visual evidence for verification, while also requiring costly retraining to support new diagnostic tasks, limiting their reliability and adaptability in clinical settings. To address these limitations, we present CXReasonAgent, a diagnostic agent that integrates a large language model (LLM) with clinically grounded diagnostic tools to perform evidence-grounded diagnostic reasoning using image-derived diagnostic and visual evidence. To evaluate these capabilities, we introduce CXReasonDial, a multi-turn dialogue benchmark with 1,946 dialogues across 12 diagnostic tasks, and show that CXReasonAgent produces faithfully grounded responses, enabling more reliable and verifiable diagnostic reasoning than LVLMs. These findings highlight the importance of integrating clinically grounded diagnostic tools, particularly in safety-critical clinical settings.

**arXiv ID:** 2602.23276
</details>

<details>
<summary><strong>To Deceive is to Teach? Forging Perceptual Robustness via Adversarial Reinforcement Learning</strong> - Yicheng Bao, Xuhong Wang, Xin Tan - [[pdf]](https://arxiv.org/pdf/2602.22227)</summary>

**Abstract:** Despite their impressive capabilities, Multimodal Large Language Models (MLLMs) exhibit perceptual fragility when confronted with visually complex scenes. This weakness stems from a reliance on finite training datasets, which are prohibitively expensive to scale and impose a ceiling on model robustness. We introduce \textbf{AOT-SFT}, a large-scale adversarial dataset for bootstrapping MLLM robustness. Building on this, we propose \textbf{AOT (Adversarial Opponent Training)}, a self-play framework that forges MLLM robustness by creating its own training data. Our method orchestrates a co-evolution between an image-editing Attacker and a Defender MLLM, where the Attacker generates a diverse and dynamic curriculum of image manipulations, forcing the Defender to adapt and improve. Extensive experiments demonstrate that AOT enhances the Defender's perceptual robustness and reduces hallucinations, establishing a scalable paradigm for training more reliable MLLMs.

**arXiv ID:** 2602.22227
</details>

<details>
<summary><strong>Learning Rewards, Not Labels: Adversarial Inverse Reinforcement Learning for Machinery Fault Detection</strong> - Dhiraj Neupane, Richard Dazeley, Mohamed Reda Bouadjenek, Sunil Aryal - [[pdf]](https://arxiv.org/pdf/2602.22297)</summary>

**Abstract:** Reinforcement learning (RL) offers significant promise for machinery fault detection (MFD). However, most existing RL-based MFD approaches do not fully exploit RL's sequential decision-making strengths, often treating MFD as a simple guessing game (Contextual Bandits). To bridge this gap, we formulate MFD as an offline inverse reinforcement learning problem, where the agent learns the reward dynamics directly from healthy operational sequences, thereby bypassing the need for manual reward engineering and fault labels. Our framework employs Adversarial Inverse Reinforcement Learning to train a discriminator that distinguishes between normal (expert) and policy-generated transitions. The discriminator's learned reward serves as an anomaly score, indicating deviations from normal operating behaviour. When evaluated on three run-to-failure benchmark datasets (HUMS2023, IMS, and XJTU-SY), the model consistently assigns low anomaly scores to normal samples and high scores to faulty ones, enabling early and robust fault detection. By aligning RL's sequential reasoning with MFD's temporal structure, this work opens a path toward RL-based diagnostics in data-driven industrial settings.

**arXiv ID:** 2602.22297
</details>

<details>
<summary><strong>Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving</strong> - Yinan Zheng, Tianyi Tan, Bin Huang, Enguang Liu, Ruiming Liang, Jianlin Zhang, Jianwei Cui, Guang Chen, Kun Ma, Hangjun Ye, Long Chen, Ya-Qin Zhang, Xianyuan Zhan, Jingjing Liu - [[pdf]](https://arxiv.org/pdf/2602.22801)</summary>

**Abstract:** Diffusion models have become a popular choice for decision-making tasks in robotics, and more recently, are also being considered for solving autonomous driving tasks. However, their applications and evaluations in autonomous driving remain limited to simulation-based or laboratory settings. The full strength of diffusion models for large-scale, complex real-world settings, such as End-to-End Autonomous Driving (E2E AD), remains underexplored. In this study, we conducted a systematic and large-scale investigation to unleash the potential of the diffusion models as planners for E2E AD, based on a tremendous amount of real-vehicle data and road testing. Through comprehensive and carefully controlled studies, we identify key insights into the diffusion loss space, trajectory representation, and data scaling that significantly impact E2E planning performance. Moreover, we also provide an effective reinforcement learning post-training strategy to further enhance the safety of the learned planner. The resulting diffusion-based learning framework, Hyper Diffusion Planner} (HDP), is deployed on a real-vehicle platform and evaluated across 6 urban driving scenarios and 200 km of real-world testing, achieving a notable 10x performance improvement over the base model. Our work demonstrates that diffusion models, when properly designed and trained, can serve as effective and scalable E2E AD planners for complex, real-world autonomous driving tasks.

**arXiv ID:** 2602.22801
</details>

<details>
<summary><strong>Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks</strong> - Shuo He, Lang Feng, Qi Wei, Xin Cheng, Lei Feng, Bo An - [[pdf]](https://arxiv.org/pdf/2602.22817)</summary>

**Abstract:** Group-based reinforcement learning (RL), such as GRPO, has advanced the capabilities of large language models on long-horizon agentic tasks. To enable more fine-grained policy updates, recent research has increasingly shifted toward stepwise group-based policy optimization, which treats each step in a rollout trajectory independently while using a memory module to retain historical context. However, we find a key issue in estimating stepwise relative advantages, namely context inconsistency, where steps within the same group may differ in their historical contexts. Empirically, we reveal that this issue can lead to severely biased advantage estimation, thereby degrading policy optimization significantly. To address the issue, in this paper, we propose Hierarchy-of-Groups Policy Optimization (HGPO) for long-horizon agentic tasks. Specifically, within a group of rollout trajectories, HGPO assigns each step to multiple hierarchical groups according to the consistency of historical contexts. Then, for each step, HGPO computes distinct advantages within each group and aggregates them with an adaptive weighting scheme. In this way, HGPO can achieve a favorable bias-variance trade-off in stepwise advantage estimation, without extra models or rollouts. Evaluations on two challenging agentic tasks, ALFWorld and WebShop with Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct, show that HGPO significantly outperforms existing agentic RL methods under the same computational constraints. Code is available at this https URL.

**arXiv ID:** 2602.22817
</details>

<details>
<summary><strong>ParamMem: Augmenting Language Agents with Parametric Reflective Memory</strong> - Tianjun Yao, Yongqiang Chen, Yujia Zheng, Pan Li, Zhiqiang Shen, Kun Zhang - [[pdf]](https://arxiv.org/pdf/2602.23320)</summary>

**Abstract:** Self-reflection enables language agents to iteratively refine solutions, yet often produces repetitive outputs that limit reasoning performance. Recent studies have attempted to address this limitation through various approaches, among which increasing reflective diversity has shown promise. Our empirical analysis reveals a strong positive correlation between reflective diversity and task success, further motivating the need for diverse reflection signals. We introduce ParamMem, a parametric memory module that encodes cross-sample reflection patterns into model parameters, enabling diverse reflection generation through temperature-controlled sampling. Building on this module, we propose ParamAgent, a reflection-based agent framework that integrates parametric memory with episodic and cross-sample memory. Extensive experiments on code generation, mathematical reasoning, and multi-hop question answering demonstrate consistent improvements over state-of-the-art baselines. Further analysis reveals that ParamMem is sample-efficient, enables weak-to-strong transfer across model scales, and supports self-improvement without reliance on stronger external model, highlighting the potential of ParamMem as an effective component for enhancing language agents.

**arXiv ID:** 2602.23320
</details>

<details>
<summary><strong>Search More, Think Less: Rethinking Long-Horizon Agentic Search for Efficiency and Generalization</strong> - Qianben Chen, Tianrui Qin, King Zhu, Qiexiang Wang, Chengjun Yu, Shu Xu, Jiaqi Wu, Jiayu Zhang, Xinpeng Liu, Xin Gui, Jingyi Cao, Piaohong Wang, Dingfeng Shi, He Zhu, Tiannan Wang, Yuqing Wang, Maojia Song, Tianyu Zheng, Ge Zhang, Jian Yang, Jiaheng Liu, Minghao Liu, Yuchen Eleanor Jiang, Wangchunshu Zhou - [[pdf]](https://arxiv.org/pdf/2602.22675)</summary>

**Abstract:** Recent deep research agents primarily improve performance by scaling reasoning depth, but this leads to high inference cost and latency in search-intensive scenarios. Moreover, generalization across heterogeneous research settings remains challenging. In this work, we propose \emph{Search More, Think Less} (SMTL), a framework for long-horizon agentic search that targets both efficiency and generalization. SMTL replaces sequential reasoning with parallel evidence acquisition, enabling efficient context management under constrained context budgets. To support generalization across task types, we further introduce a unified data synthesis pipeline that constructs search tasks spanning both deterministic question answering and open-ended research scenarios with task appropriate evaluation metrics. We train an end-to-end agent using supervised fine-tuning and reinforcement learning, achieving strong and often state of the art performance across benchmarks including BrowseComp (48.6\%), GAIA (75.7\%), Xbench (82.0\%), and DeepResearch Bench (45.9\%). Compared to Mirothinker-v1.0, SMTL with maximum 100 interaction steps reduces the average number of reasoning steps on BrowseComp by 70.7\%, while improving accuracy.

**arXiv ID:** 2602.22675
</details>

<details>
<summary><strong>PARL: Prompt-based Agents for Reinforcement Learning</strong> - Yarik Menchaca Resendiz, Roman Klinger - [[pdf]](https://arxiv.org/pdf/2510.21306)</summary>

**Abstract:** Large language models (LLMs) have demonstrated high performance on tasks expressed in natural language, particularly in zero- or few-shot settings. These are typically framed as supervised (e.g., classification) or unsupervised (e.g., clustering) problems. However, limited work evaluates LLMs as agents in reinforcement learning (RL) tasks (e.g., playing games), where learning occurs through interaction with an environment and a reward system. While prior work focused on representing tasks that rely on a language representation, we study structured, non-linguistic reasoning - such as interpreting positions in a grid world. We therefore introduce PARL (Prompt-based Agent for Reinforcement Learning), a method that uses LLMs as RL agents through prompting, without any fine-tuning. PARL encodes actions, states, and rewards in the prompt, enabling the model to learn through trial-and-error interaction. We evaluate PARL on three standard RL tasks that do not entirely rely on natural language. We show that it can match or outperform traditional RL agents in simple environments by leveraging pretrained knowledge. However, we identify performance limitations in tasks that require complex mathematical operations or decoding states and actions.

**arXiv ID:** 2510.21306
</details>

<details>
<summary><strong>Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning</strong> - Yihe Deng, I-Hung Hsu, Jun Yan, Zifeng Wang, Rujun Han, Gufeng Zhang, Yanfei Chen, Wei Wang, Tomas Pfister, Chen-Yu Lee - [[pdf]](https://arxiv.org/pdf/2510.25992)</summary>

**Abstract:** Large Language Models (LLMs) often struggle with problems that require multi-step reasoning. For small-scale open-source models, Reinforcement Learning with Verifiable Rewards (RLVR) fails when correct solutions are rarely sampled even after many attempts, while Supervised Fine-Tuning (SFT) tends to overfit long demonstrations through rigid token-by-token imitation. To address this gap, we propose Supervised Reinforcement Learning (SRL), a framework that reformulates problem solving as generating a sequence of logical "actions". SRL trains the model to generate an internal reasoning monologue before committing to each action. It provides smoother rewards based on the similarity between the model's actions and expert actions extracted from the SFT dataset in a step-wise manner. This supervision offers richer learning signals even when all rollouts are incorrect, while encouraging flexible reasoning guided by expert demonstrations. As a result, SRL enables small models to learn challenging problems previously unlearnable by SFT or RLVR. Moreover, initializing training with SRL before refining with RLVR yields the strongest overall performance. Beyond reasoning benchmarks, SRL generalizes effectively to agentic software engineering tasks, establishing it as a robust and versatile training framework for reasoning-oriented LLMs.

**arXiv ID:** 2510.25992
</details>

<details>
<summary><strong>Enhancing Geometric Perception in VLMs via Translator-Guided Reinforcement Learning</strong> - Hao Yu, Shuning Jia, Guanghao Li, Wenhao Jiang, Chun Yuan - [[pdf]](https://arxiv.org/pdf/2602.22703)</summary>

**Abstract:** Vision-language models (VLMs) often struggle with geometric reasoning due to their limited perception of fundamental diagram elements. To tackle this challenge, we introduce GeoPerceive, a benchmark comprising diagram instances paired with domain-specific language (DSL) representations, along with an efficient automatic data generation pipeline. This design enables the isolated evaluation of geometric perception independently from reasoning. To exploit the data provided by GeoPerceive for enhancing the geometric perception capabilities of VLMs, we propose GeoDPO, a translator-guided reinforcement learning (RL) framework. GeoDPO employs an NL-to-DSL translator, which is trained on synthetic pairs generated by the data engine of GeoPerceive, to bridge natural language and DSL. This translator facilitates the computation of fine-grained, DSL-level scores, which serve as reward signals in reinforcement learning. We assess GeoDPO on both in-domain and out-of-domain datasets, spanning tasks in geometric perception as well as downstream reasoning. Experimental results demonstrate that, while supervised fine-tuning (SFT) offers only marginal improvements and may even impair performance in out-of-domain scenarios, GeoDPO achieves substantial gains: $+26.5\%$ on in-domain data, $+8.0\%$ on out-of-domain data, and $+39.0\%$ on downstream reasoning tasks. These findings underscore the superior performance and generalization ability of GeoDPO over SFT. All codes are released at this https URL
to ensure reproducibility.

**arXiv ID:** 2602.22703
</details>

<details>
<summary><strong>EvolveGen: Algorithmic Level Hardware Model Checking Benchmark Generation through Reinforcement Learning</strong> - Guangyu Hu, Xiaofeng Zhou, Wei Zhang, Hongce Zhang - [[pdf]](https://arxiv.org/pdf/2602.22609)</summary>

**Abstract:** Progress in hardware model checking depends critically on high-quality benchmarks. However, the community faces a significant benchmark gap: existing suites are limited in number, often distributed only in representations such as BTOR2 without access to the originating register-transfer-level (RTL) designs, and biased toward extreme difficulty where instances are either trivial or intractable. These limitations hinder rigorous evaluation of new verification techniques and encourage overfitting of solver heuristics to a narrow set of problems. To address this, we introduce EvolveGen, a framework for generating hardware model checking benchmarks by combining reinforcement learning (RL) with high-level synthesis (HLS). Our approach operates at an algorithmic level of abstraction in which an RL agent learns to construct computation graphs. By compiling these graphs under different synthesis directives, we produce pairs of functionally equivalent but structurally distinct hardware designs, inducing challenging model checking instances. Solver runtime is used as the reward signal, enabling the agent to autonomously discover and generate small-but-hard instances that expose solver-specific weaknesses. Experiments show that EvolveGen efficiently creates a diverse benchmark set in standard formats (e.g., AIGER and BTOR2) and effectively reveals performance bottlenecks in state-of-the-art model checkers.

**arXiv ID:** 2602.22609
</details>

<details>
<summary><strong>WebGym: Scaling Training Environments for Visual Web Agents with Realistic Tasks</strong> - Hao Bai, Alexey Taymanov, Tong Zhang, Aviral Kumar, Spencer Whitehead - [[pdf]](https://arxiv.org/pdf/2601.02439)</summary>

**Abstract:** We present WebGym, the largest-to-date open-source environment for training realistic visual web agents. Real websites are non-stationary and diverse, making artificial or small-scale task sets insufficient for robust policy learning. WebGym contains nearly 300,000 tasks with rubric-based evaluations across diverse, real-world websites and difficulty levels. We train agents with a simple reinforcement learning (RL) recipe, which trains on the agent's own interaction traces (rollouts), using task rewards as feedback to guide learning. To enable scaling RL, we speed up sampling of trajectories in WebGym by developing a high-throughput asynchronous rollout system, designed specifically for web agents. Our system achieves a 4-5x rollout speedup compared to naive implementations. Second, we scale the task set breadth, depth, and size, which results in continued performance improvement. Fine-tuning a strong base vision-language model, Qwen-3-VL-8B-Instruct, on WebGym results in an improvement in success rate on an out-of-distribution test set from 26.2% to 42.9%, significantly outperforming agents based on proprietary models such as GPT-4o and GPT-5-Thinking that achieve 27.1% and 29.8%, respectively. This improvement is substantial because our test set consists only of tasks on websites never seen during training, unlike many other prior works on training visual web agents.

**arXiv ID:** 2601.02439
</details>

<details>
<summary><strong>Agentic Framework for Epidemiological Modeling</strong> - Rituparna Datta, Zihan Guan, Baltazar Espinoza, Yiqi Su, Priya Pitre, Srini Venkatramanan, Naren Ramakrishnan, Anil Vullikanti - [[pdf]](https://arxiv.org/pdf/2602.00299)</summary>

**Abstract:** Epidemic modeling is essential for public health planning, yet traditional approaches rely on fixed model classes that require manual redesign as pathogens, policies, and scenario assumptions evolve. We introduce EPIAGENT, an agentic framework that automatically synthesizes, calibrates, verifies, and refines epidemiological simulators by modeling disease progression as an iterative program synthesis problem. A central design choice is an explicit epidemiological flow graph intermediate representation that links scenario specifications to model structure and enables strong, modular correctness checks before code is generated. Verified flow graphs are then compiled into mechanistic models supporting interpretable parameter learning under physical and epidemiological constraints. Evaluation on epidemiological scenario case studies demonstrates that EPIAGENT captures complex growth dynamics and produces epidemiologically consistent counterfactual projections across varying vaccination and immune escape assumptions. Our results show that the agentic feedback loop prevents degeneration and significantly accelerates convergence toward valid models by mimicking professional expert workflows.

**arXiv ID:** 2602.00299
</details>

<details>
<summary><strong>DreamWaQ++: Obstacle-Aware Quadrupedal Locomotion With Resilient Multi-Modal Reinforcement Learning</strong> - I Made Aswin Nahrendra, Byeongho Yu, Minho Oh, Dongkyu Lee, Seunghyun Lee, Hyeonwoo Lee, Hyungtae Lim, Hyun Myung - [[pdf]](https://arxiv.org/pdf/2409.19709)</summary>

**Abstract:** Quadrupedal robots hold promising potential for applications in navigating cluttered environments with resilience akin to their animal counterparts. However, their floating base configuration makes them vulnerable to real-world uncertainties, yielding substantial challenges in their locomotion control. Deep reinforcement learning has become one of the plausible alternatives for realizing a robust locomotion controller. However, the approaches that rely solely on proprioception sacrifice collision-free locomotion because they require front-feet contact to detect the presence of stairs to adapt the locomotion gait. Meanwhile, incorporating exteroception necessitates a precisely modeled map observed by exteroceptive sensors over a period of time. Therefore, this work proposes a novel method to fuse proprioception and exteroception featuring a resilient multi-modal reinforcement learning. The proposed method yields a controller that showcases agile locomotion performance on a quadrupedal robot over a myriad of real-world courses, including rough terrains, steep slopes, and high-rise stairs, while retaining its robustness against out-of-distribution situations.

**arXiv ID:** 2409.19709
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
