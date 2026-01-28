# Agent arXiv Daily

**Last Updated:** 2026-01-28 02:32:12

**Total Papers:** 118

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (14 papers)</h2></summary>

<details>
<summary><strong>Interpreting Agentic Systems: Beyond Model Explanations to System-Level Accountability</strong> - Judy Zhu, Dhari Gandhi, Himanshu Joshi, Ahmad Rezaie Mianroodi, Sedef Akinli Kocak, Dhanesh Ramachandran - [[pdf]](https://arxiv.org/pdf/2601.17168)</summary>

**Abstract:** Agentic systems have transformed how Large Language Models (LLMs) can be leveraged to create autonomous systems with goal-directed behaviors, consisting of multi-step planning and the ability to interact with different environments. These systems differ fundamentally from traditional machine learning models, both in architecture and deployment, introducing unique AI safety challenges, including goal misalignment, compounding decision errors, and coordination risks among interacting agents, that necessitate embedding interpretability and explainability by design to ensure traceability and accountability across their autonomous behaviors. Current interpretability techniques, developed primarily for static models, show limitations when applied to agentic systems. The temporal dynamics, compounding decisions, and context-dependent behaviors of agentic systems demand new analytical approaches. This paper assesses the suitability and limitations of existing interpretability methods in the context of agentic systems, identifying gaps in their capacity to provide meaningful insight into agent decision-making. We propose future directions for developing interpretability techniques specifically designed for agentic systems, pinpointing where interpretability is required to embed oversight mechanisms across the agent lifecycle from goal formation, through environmental interaction, to outcome evaluation. These advances are essential to ensure the safe and accountable deployment of agentic AI systems.

**arXiv ID:** 2601.17168
</details>

<details>
<summary><strong>Lattice: Generative Guardrails for Conversational Agents</strong> - Emily Broadhurst, Tawab Safi, Joseph Edell, Vashisht Ganesh, Karime Maamari - [[pdf]](https://arxiv.org/pdf/2601.17481)</summary>

**Abstract:** Conversational AI systems require guardrails to prevent harmful outputs, yet existing approaches use static rules that cannot adapt to new threats or deployment contexts. We introduce Lattice, a framework for self-constructing and continuously improving guardrails. Lattice operates in two stages: construction builds initial guardrails from labeled examples through iterative simulation and optimization; continuous improvement autonomously adapts deployed guardrails through risk assessment, adversarial testing, and consolidation. Evaluated on the ProsocialDialog dataset, Lattice achieves 91% F1 on held-out data, outperforming keyword baselines by 43pp, LlamaGuard by 25pp, and NeMo by 4pp. The continuous improvement stage achieves 7pp F1 improvement on cross-domain data through closed-loop optimization. Our framework shows that effective guardrails can be self-constructed through iterative optimization.

**arXiv ID:** 2601.17481
</details>

<details>
<summary><strong>When Personalization Legitimizes Risks: Uncovering Safety Vulnerabilities in Personalized Dialogue Agents</strong> - Jiahe Guo, Xiangran Guo, Yulin Hu, Zimo Long, Xingyu Sui, Xuda Zhi, Yongbo Huang, Hao He, Weixiang Zhao, Yanyan Zhao, Bing Qin - [[pdf]](https://arxiv.org/pdf/2601.17887)</summary>

**Abstract:** Long-term memory enables large language model (LLM) agents to support personalized and sustained interactions. However, most work on personalized agents prioritizes utility and user experience, treating memory as a neutral component and largely overlooking its safety implications. In this paper, we reveal intent legitimation, a previously underexplored safety failure in personalized agents, where benign personal memories bias intent inference and cause models to legitimize inherently harmful queries. To study this phenomenon, we introduce PS-Bench, a benchmark designed to identify and quantify intent legitimation in personalized interactions. Across multiple memory-augmented agent frameworks and base LLMs, personalization increases attack success rates by 15.8%-243.7% relative to stateless baselines. We further provide mechanistic evidence for intent legitimation from internal representations space, and propose a lightweight detection-reflection method that effectively reduces safety degradation. Overall, our work provides the first systematic exploration and evaluation of intent legitimation as a safety failure mode that naturally arises from benign, real-world personalization, highlighting the importance of assessing safety under long-term personal context. WARNING: This paper may contain harmful content.

**arXiv ID:** 2601.17887
</details>

<details>
<summary><strong>Agentic AI for Self-Driving Laboratories in Soft Matter: Taxonomy, Benchmarks,and Open Challenges</strong> - Xuanzhou Chen, Audrey Wang, Stanley Yin, Hanyang Jiang, Dong Zhang - [[pdf]](https://arxiv.org/pdf/2601.17920)</summary>

**Abstract:** Self-driving laboratories (SDLs) close the loop between experiment design, automated execution, and data-driven decision making, and they provide a demanding testbed for agentic AI under expensive actions, noisy and delayed feedback, strict feasibility and safety constraints, and non-stationarity. This survey uses soft matter as a representative setting but focuses on the AI questions that arise in real laboratories. We frame SDL autonomy as an agent environment interaction problem with explicit observations, actions, costs, and constraints, and we use this formulation to connect common SDL pipelines to established AI principles. We review the main method families that enable closed loop experimentation, including Bayesian optimization and active learning for sample efficient experiment selection, planning and reinforcement learning for long horizon protocol optimization, and tool using agents that orchestrate heterogeneous instruments and software. We emphasize verifiable and provenance aware policies that support debugging, reproducibility, and safe operation. We then propose a capability driven taxonomy that organizes systems by decision horizon, uncertainty modeling, action parameterization, constraint handling, failure recovery, and human involvement. To enable meaningful comparison, we synthesize benchmark task templates and evaluation metrics that prioritize cost aware performance, robustness to drift, constraint violation behavior, and reproducibility. Finally, we distill lessons from deployed SDLs and outline open challenges in multi-modal representation, calibrated uncertainty, safe exploration, and shared benchmark infrastructure.

**arXiv ID:** 2601.17920
</details>

<details>
<summary><strong>FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory</strong> - Lei Wei, Xu Dong, Xiao Peng, Niantao Xie, Bin Wang - [[pdf]](https://arxiv.org/pdf/2601.18642)</summary>

**Abstract:** Large language models deployed as autonomous agents face critical memory limitations, lacking selective forgetting mechanisms that lead to either catastrophic forgetting at context boundaries or information overload within them. While human memory naturally balances retention and forgetting through adaptive decay processes, current AI systems employ binary retention strategies that preserve everything or lose it entirely. We propose FadeMem, a biologically-inspired agent memory architecture that incorporates active forgetting mechanisms mirroring human cognitive efficiency. FadeMem implements differential decay rates across a dual-layer memory hierarchy, where retention is governed by adaptive exponential decay functions modulated by semantic relevance, access frequency, and temporal patterns. Through LLM-guided conflict resolution and intelligent memory fusion, our system consolidates related information while allowing irrelevant details to fade. Experiments on Multi-Session Chat, LoCoMo, and LTI-Bench demonstrate superior multi-hop reasoning and retrieval with 45\% storage reduction, validating the effectiveness of biologically-inspired forgetting in agent memory systems.

**arXiv ID:** 2601.18642
</details>

<details>
<summary><strong>TEA-Bench: A Systematic Benchmarking of Tool-enhanced Emotional Support Dialogue Agent</strong> - Xingyu Sui, Yanyan Zhao, Yulin Hu, Jiahe Guo, Weixiang Zhao, Bing Qin - [[pdf]](https://arxiv.org/pdf/2601.18700)</summary>

**Abstract:** Emotional Support Conversation requires not only affective expression but also grounded instrumental support to provide trustworthy guidance. However, existing ESC systems and benchmarks largely focus on affective support in text-only settings, overlooking how external tools can enable factual grounding and reduce hallucination in multi-turn emotional support. We introduce TEA-Bench, the first interactive benchmark for evaluating tool-augmented agents in ESC, featuring realistic emotional scenarios, an MCP-style tool environment, and process-level metrics that jointly assess the quality and factual grounding of emotional support. Experiments on nine LLMs show that tool augmentation generally improves emotional support quality and reduces hallucination, but the gains are strongly capacity-dependent: stronger models use tools more selectively and effectively, while weaker models benefit only marginally. We further release TEA-Dialog, a dataset of tool-enhanced ESC dialogues, and find that supervised fine-tuning improves in-distribution support but generalizes poorly. Our results underscore the importance of tool use in building reliable emotional support agents.

**arXiv ID:** 2601.18700
</details>

<details>
<summary><strong>Understanding Users' Privacy Reasoning and Behaviors During Chatbot Use to Support Meaningful Agency in Privacy</strong> - Mohammad Hadi Nezhad, Francisco Enrique Vicente Castro, Ivon Arroyo - [[pdf]](https://arxiv.org/pdf/2601.18125)</summary>

**Abstract:** Conversational agents (CAs) (e.g., chatbots) are increasingly used in settings where users disclose sensitive information, raising significant privacy concerns. Because privacy judgments are highly contextual, supporting users to engage in privacy-protective actions during chatbot interactions is essential. However, enabling meaningful engagement requires a deeper understanding of how users currently reason about and manage sensitive information during realistic chatbot use scenarios. To investigate this, we qualitatively examined computer science (undergraduate and masters) students' in-the-moment disclosure and protection behaviors, as well as the reasoning underlying these behaviors, across a range of realistic chatbot tasks. Participants used a simulated ChatGPT interface with and without a privacy notice panel that intercepts message submissions, highlights potentially sensitive information, and offers privacy protective actions. The panel supports anonymization through retracting, faking, and generalizing, and surfaces two of ChatGPT's built-in privacy controls to improve their discoverability. Drawing on interaction logs, think-alouds, and survey responses, we analyzed how the panel fostered privacy awareness, encouraged protective actions, and supported context-specific reasoning about what information to protect and how. We further discuss design opportunities for tools that provide users greater and more meaningful agency in protecting sensitive information during CA interactions.

**arXiv ID:** 2601.18125
</details>

<details>
<summary><strong>Temp-R1: A Unified Autonomous Agent for Complex Temporal KGQA via Reverse Curriculum Reinforcement Learning</strong> - Zhaoyan Gong, Zhiqiang Liu, Songze Li, Xiaoke Guo, Yuanxiang Liu, Xinle Deng, Zhizhen Liu, Lei Liang, Huajun Chen, Wen Zhang - [[pdf]](https://arxiv.org/pdf/2601.18296)</summary>

**Abstract:** Temporal Knowledge Graph Question Answering (TKGQA) is inherently challenging, as it requires sophisticated reasoning over dynamic facts with multi-hop dependencies and complex temporal constraints. Existing methods rely on fixed workflows and expensive closed-source APIs, limiting flexibility and scalability. We propose Temp-R1, the first autonomous end-to-end agent for TKGQA trained through reinforcement learning. To address cognitive overload in single-action reasoning, we expand the action space with specialized internal actions alongside external action. To prevent shortcut learning on simple questions, we introduce reverse curriculum learning that trains on difficult questions first, forcing the development of sophisticated reasoning before transferring to easier cases. Our 8B-parameter Temp-R1 achieves state-of-the-art performance on MultiTQ and TimelineKGQA, improving 19.8% over strong baselines on complex questions. Our work establishes a new paradigm for autonomous temporal reasoning agents. Our code will be publicly available soon at this https URL.

**arXiv ID:** 2601.18296
</details>

<details>
<summary><strong>RoboPhD: Self-Improving Text-to-SQL Through Autonomous Agent Evolution</strong> - Andrew Borthwick, Stephen Ash - [[pdf]](https://arxiv.org/pdf/2601.01126)</summary>

**Abstract:** We present RoboPhD, a system where AI agents autonomously conduct research to improve Text-to-SQL performance. RoboPhD implements a closed-loop evolution cycle with two coordinated components: a SQL Generation agent composed of a database analysis script and SQL generation instructions, and an Evolution agent that designs new versions based on performance feedback. Central to the framework is an ELO-based selection mechanism enabling survival-of-the-fittest dynamics while handling non-transitivity in performance. Starting from a naive 70-line baseline, RoboPhD evolves agents through iterative cross-pollination, discovering effective techniques without any external guidance on the Text-to-SQL domain. Our best agent, evolved to 1500 lines over 18 iterations, autonomously discovered strategies such as size-adaptive database analysis that adjusts depth based on schema complexity and SQL generation patterns for column selection, evidence interpretation, and aggregation. Evolution provides the largest gains on cheaper models: while we improve by 2.3 points over a strong Claude Opus 4.5 naive baseline, we show an improvement of 8.9 points over the weaker Claude Haiku model. This enables 'skip a tier' deployment: evolved Haiku exceeds naive Sonnet accuracy, and evolved Sonnet exceeds naive Opus, both at lower cost. The full system achieves 73.67% accuracy on the BIRD test set, demonstrating that AI can autonomously build a strong agentic system with only a trivial human-provided starting point.

**arXiv ID:** 2601.01126
</details>

<details>
<summary><strong>Autonomous Mars Rover Module for Soil Sampling and Life Component Analysis</strong> - Bibek Adhikari, Rishab Rijal, Rakesh Yadav, Nikchey Khatri, Sandesh Dhakal - [[pdf]](https://arxiv.org/pdf/2601.17158)</summary>

**Abstract:** The search for extraterrestrial life has long been a primary focus of scientific exploration, driven by rapid advancements in technology and our understanding of the universe. The discovery of water on Mars has sparked significant interest, raising the question of whether life could exist on the planet. This study proposes a novel approach to simulate and illustrate the detection of life using a proof-of-life module integrated into a Mars rover. The module is an autonomous system capable of traveling to designated regions, excavating soil, collecting samples, and performing biochemical testing onboard the rover itself. The project is inherently multidisciplinary, integrating mechanical systems such as a drill mechanism and a vacuum system, alongside biochemical analysis for soil testing. The module is capable of successfully detecting the presence or absence of living components of life from the collected soil particles. This proof-of-life module serves as a proof-of-concept for autonomous life detection in extraterrestrial environments and lays the foundation for future exploration missions.

**arXiv ID:** 2601.17158
</details>

<details>
<summary><strong>Vision-Proprioception Fusion with Mamba2 in End-to-End Reinforcement Learning for Motion Control</strong> - Xiaowen Tao, Yinuo Wang, Jinzhao Zhou - [[pdf]](https://arxiv.org/pdf/2509.07593)</summary>

**Abstract:** End-to-end reinforcement learning (RL) for motion control trains policies directly from sensor inputs to motor commands, enabling unified controllers for different robots and tasks. However, most existing methods are either blind (proprioception-only) or rely on fusion backbones with unfavorable compute-memory trade-offs. Recurrent controllers struggle with long-horizon credit assignment, and Transformer-based fusion incurs quadratic cost in token length, limiting temporal and spatial context. We present a vision-driven cross-modal RL framework built on SSD-Mamba2, a selective state-space backbone that applies state-space duality (SSD) to enable both recurrent and convolutional scanning with hardware-aware streaming and near-linear scaling. Proprioceptive states and exteroceptive observations (e.g., depth tokens) are encoded into compact tokens and fused by stacked SSD-Mamba2 layers. The selective state-space updates retain long-range dependencies with markedly lower latency and memory use than quadratic self-attention, enabling longer look-ahead, higher token resolution, and stable training under limited compute. Policies are trained end-to-end under curricula that randomize terrain and appearance and progressively increase scene complexity. A compact, state-centric reward balances task progress, energy efficiency, and safety. Across diverse motion-control scenarios, our approach consistently surpasses strong state-of-the-art baselines in return, safety (collisions and falls), and sample efficiency, while converging faster at the same compute budget. These results suggest that SSD-Mamba2 provides a practical fusion backbone for resource-constrained robotic and autonomous systems in engineering informatics applications.

**arXiv ID:** 2509.07593
</details>

<details>
<summary><strong>Bridging Instead of Replacing Online Coding Communities with AI through Community-Enriched Chatbot Designs</strong> - Junling Wang, Lahari Goswami, Gustavo Kreia Umbelino, Kiara Garcia Chau, Mrinmaya Sachan, April Yi Wang - [[pdf]](https://arxiv.org/pdf/2601.18697)</summary>

**Abstract:** LLM-based chatbots like ChatGPT have become popular tools for assisting with coding tasks. However, they often produce isolated responses and lack mechanisms for social learning or contextual grounding. In contrast, online coding communities like Kaggle offer socially mediated learning environments that foster critical thinking, engagement, and a sense of belonging. Yet, growing reliance on LLMs risks diminishing participation in these communities and weakening their collaborative value. To address this, we propose Community-Enriched AI, a design paradigm that embeds social learning dynamics into LLM-based chatbots by surfacing user-generated content and social design feature from online coding communities. Using this paradigm, we implemented a RAG-based AI chatbot leveraging resources from Kaggle to validate our design. Across two empirical studies involving 28 and 12 data science learners, respectively, we found that Community-Enriched AI significantly enhances user trust, encourages engagement with community, and effectively supports learners in solving data science tasks. We conclude by discussing design implications for AI assistance systems that bridge -- rather than replace -- online coding communities.

**arXiv ID:** 2601.18697
</details>

<details>
<summary><strong>Understanding Teen Overreliance on AI Companion Chatbots Through Self-Reported Reddit Narratives</strong> - Mohammad Namvarpour, Brandon Brofsky, Jessica Medina, Mamtaj Akter, Afsaneh Razi - [[pdf]](https://arxiv.org/pdf/2507.15783)</summary>

**Abstract:** AI companion chatbots are increasingly popular with teens. While these interactions are entertaining, they also risk overuse that can potentially disrupt offline daily life. We examined how adolescents describe reliance on AI companions, mapping their experiences onto behavioral addiction frameworks and exploring pathways to disengagement, by analyzing 318 Reddit posts made by users who self-disclosed as 13-17 years old on the this http URL subreddit. We found teens often begin using chatbots for support or creative play, but these activities can deepen into strong attachments marked by conflict, withdrawal, tolerance, relapse, and mood regulation. Reported consequences include sleep loss, academic decline, and strained real-world connections. Disengagement commonly arises when teens recognize harm, re-engage with offline life, or encounter restrictive platform changes. We highlight specific risks of character-based companion chatbots based on teens' perspectives and introduce a design framework (CARE) for guidance for safer systems and setting directions for future teen-centered research.

**arXiv ID:** 2507.15783
</details>

<details>
<summary><strong>Device-Native Autonomous Agents for Privacy-Preserving Negotiations</strong> - Joyjit Roy, Samaresh Kumar Singh - [[pdf]](https://arxiv.org/pdf/2601.00911)</summary>

**Abstract:** Automated negotiations in insurance and business-to-business (B2B) commerce encounter substantial challenges. Current systems force a trade-off between convenience and privacy by routing sensitive financial data through centralized servers, increasing security risks, and diminishing user trust. This study introduces a device-native autonomous Artificial Intelligence (AI) agent system for privacy-preserving negotiations. The proposed system operates exclusively on user hardware, enabling real-time bargaining while maintaining sensitive constraints locally. It integrates zero-knowledge proofs to ensure privacy and employs distilled world models to support advanced on-device reasoning. The architecture incorporates six technical components within an agentic AI workflow. Agents autonomously plan negotiation strategies, conduct secure multi-party bargaining, and generate cryptographic audit trails without exposing user data to external servers. The system is evaluated in insurance and B2B procurement scenarios across diverse device configurations. Results show an average success rate of 87%, a 2.4x latency improvement over cloud baselines, and strong privacy preservation through zero-knowledge proofs. User studies show 27% higher trust scores when decision trails are available. These findings establish a foundation for trustworthy autonomous agents in privacy-sensitive financial domains.

**arXiv ID:** 2601.00911
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (20 papers)</h2></summary>

<details>
<summary><strong>TheoremForge: Scaling up Formal Data Synthesis with Low-Budget Agentic Workflow</strong> - Yicheng Tao, Hongteng Xu - [[pdf]](https://arxiv.org/pdf/2601.17332)</summary>

**Abstract:** The high cost of agentic workflows in formal mathematics hinders large-scale data synthesis, exacerbating the scarcity of open-source corpora. To address this, we introduce \textbf{TheoremForge}, a cost-effective formal data synthesis pipeline that decomposes the formalization process into five sub-tasks, which are \textit{statement formalization}, \textit{proof generation}, \textit{premise selection}, \textit{proof correction} and \textit{proof sketching}. By implementing a \textit{Decoupled Extraction Strategy}, the workflow recovers valid training signals from globally failed trajectories, effectively utilizing wasted computation. Experiments on a 2,000-problem benchmark demonstrate that TheoremForge achieves a Verified Rate of 12.6\%, surpassing the 8.6\% baseline, at an average cost of only \textbf{\$0.481} per successful trajectory using Gemini-3-Flash. Crucially, our strategy increases data yield by \textbf{1.6$\times$} for proof generation compared to standard filtering. These results establish TheoremForge as a scalable framework for constructing a data flywheel to train future expert models. Our code is available \href{this https URL}{here}.

**arXiv ID:** 2601.17332
</details>

<details>
<summary><strong>EntWorld: A Holistic Environment and Benchmark for Verifiable Enterprise GUI Agents</strong> - Ying Mo, Yu Bai, Dapeng Sun, Yuqian Shi, Yukai Miao, Li Chen, Dan Li - [[pdf]](https://arxiv.org/pdf/2601.17722)</summary>

**Abstract:** Recent advances in Multimodal Large Language Models (MLLMs) have enabled agents to operate in open-ended web and operating system environments. However, existing benchmarks predominantly target consumer-oriented scenarios (e.g., e-commerce and travel booking), failing to capture the complexity and rigor of professional enterprise workflows. Enterprise systems pose distinct challenges, including high-density user interfaces, strict business logic constraints, and a strong reliance on precise, state-consistent information retrieval-settings in which current generalist agents often struggle. To address this gap, we introduce EntWorld, a large-scale benchmark consisting of 1,756 tasks across six representative enterprise domains, including customer relationship management (CRM), information technology infrastructure library (ITIL), and enterprise resource planning (ERP) systems. Unlike previous datasets that depend on fragile execution traces or extensive manual annotation, EntWorld adopts a schema-grounded task generation framework that directly reverse-engineers business logic from underlying database schemas, enabling the synthesis of realistic, long-horizon workflows. Moreover, we propose a SQL-based deterministic verification mechanism in building datasets that replaces ambiguous visual matching with rigorous state-transition validation. Experimental results demonstrate that state-of-the-art models (e.g., GPT-4.1) achieve 47.61% success rate on EntWorld, substantially lower than the human performance, highlighting a pronounced enterprise gap in current agentic capabilities and the necessity of developing domain-specific agents. We release EntWorld as a rigorous testbed to facilitate the development and evaluation of the next generation of enterprise-ready digital agents.

**arXiv ID:** 2601.17722
</details>

<details>
<summary><strong>SAGE: Steerable Agentic Data Generation for Deep Search with Execution Feedback</strong> - Fangyuan Xu, Rujun Han, Yanfei Chen, Zifeng Wang, I-Hung Hsu, Jun Yan, Vishy Tirumalashetty, Eunsol Choi, Tomas Pfister, Chen-Yu Lee - [[pdf]](https://arxiv.org/pdf/2601.18202)</summary>

**Abstract:** Deep search agents, which aim to answer complex questions requiring reasoning across multiple documents, can significantly speed up the information-seeking process. Collecting human annotations for this application is prohibitively expensive due to long and complex exploration trajectories. We propose an agentic pipeline that automatically generates high quality, difficulty-controlled deep search question-answer pairs for a given corpus and a target difficulty level. Our pipeline, SAGE, consists of a data generator which proposes QA pairs and a search agent which attempts to solve the generated question and provide execution feedback for the data generator. The two components interact over multiple rounds to iteratively refine the question-answer pairs until they satisfy the target difficulty level. Our intrinsic evaluation shows SAGE generates questions that require diverse reasoning strategies, while significantly increases the correctness and difficulty of the generated data. Our extrinsic evaluation demonstrates up to 23% relative performance gain on popular deep search benchmarks by training deep search agents with our synthetic data. Additional experiments show that agents trained on our data can adapt from fixed-corpus retrieval to Google Search at inference time, without further training.

**arXiv ID:** 2601.18202
</details>

<details>
<summary><strong>Yunjue Agent Tech Report: A Fully Reproducible, Zero-Start In-Situ Self-Evolving Agent System for Open-Ended Tasks</strong> - Haotian Li, Shijun Yang, Weizhen Qi, Silei Zhao, Rui Hua, Mingzhu Song, Xiaojian Yang, Chao Peng - [[pdf]](https://arxiv.org/pdf/2601.18226)</summary>

**Abstract:** Conventional agent systems often struggle in open-ended environments where task distributions continuously drift and external supervision is scarce. Their reliance on static toolsets or offline training lags behind these dynamics, leaving the system's capability boundaries rigid and unknown. To address this, we propose the In-Situ Self-Evolving paradigm. This approach treats sequential task interactions as a continuous stream of experience, enabling the system to distill short-term execution feedback into long-term, reusable capabilities without access to ground-truth labels. Within this framework, we identify tool evolution as the critical pathway for capability expansion, which provides verifiable, binary feedback signals. Within this framework, we develop Yunjue Agent, a system that iteratively synthesizes, optimizes, and reuses tools to navigate emerging challenges. To optimize evolutionary efficiency, we further introduce a Parallel Batch Evolution strategy. Empirical evaluations across five diverse benchmarks under a zero-start setting demonstrate significant performance gains over proprietary baselines. Additionally, complementary warm-start evaluations confirm that the accumulated general knowledge can be seamlessly transferred to novel domains. Finally, we propose a novel metric to monitor evolution convergence, serving as a function analogous to training loss in conventional optimization. We open-source our codebase, system traces, and evolved tools to facilitate future research in resilient, self-evolving intelligence.

**arXiv ID:** 2601.18226
</details>

<details>
<summary><strong>AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security</strong> - Dongrui Liu, Qihan Ren, Chen Qian, Shuai Shao, Yuejin Xie, Yu Li, Zhonghao Yang, Haoyu Luo, Peng Wang, Qingyu Liu, Binxin Hu, Ling Tang, Jilin Mei, Dadi Guo, Leitao Yuan, Junyao Yang, Guanxu Chen, Qihao Lin, Yi Yu, Bo Zhang, Jiaxuan Guo, Jie Zhang, Wenqi Shao, Huiqi Deng, Zhiheng Xi, Wenjie Wang, Wenxuan Wang, Wen Shen, Zhikai Chen, Haoyu Xie, Jialing Tao, Juntao Dai, Jiaming Ji, Zhongjie Ba, Linfeng Zhang, Yong Liu, Quanshi Zhang, Lei Zhu, Zhihua Wei, Hui Xue, Chaochao Lu, Jing Shao, Xia Hu - [[pdf]](https://arxiv.org/pdf/2601.18491)</summary>

**Abstract:** The rise of AI agents introduces complex safety and security challenges arising from autonomous tool use and environmental interactions. Current guardrail models lack agentic risk awareness and transparency in risk diagnosis. To introduce an agentic guardrail that covers complex and numerous risky behaviors, we first propose a unified three-dimensional taxonomy that orthogonally categorizes agentic risks by their source (where), failure mode (how), and consequence (what). Guided by this structured and hierarchical taxonomy, we introduce a new fine-grained agentic safety benchmark (ATBench) and a Diagnostic Guardrail framework for agent safety and security (AgentDoG). AgentDoG provides fine-grained and contextual monitoring across agent trajectories. More Crucially, AgentDoG can diagnose the root causes of unsafe actions and seemingly safe but unreasonable actions, offering provenance and transparency beyond binary labels to facilitate effective agent alignment. AgentDoG variants are available in three sizes (4B, 7B, and 8B parameters) across Qwen and Llama model families. Extensive experimental results demonstrate that AgentDoG achieves state-of-the-art performance in agentic safety moderation in diverse and complex interactive scenarios. All models and datasets are openly released.

**arXiv ID:** 2601.18491
</details>

<details>
<summary><strong>TelcoAI: Advancing 3GPP Technical Specification Search through Agentic Multi-Modal Retrieval-Augmented Generation</strong> - Rahul Ghosh, Chun-Hao Liu, Gaurav Rele, Vidya Sagar Ravipati, Hazar Aouad - [[pdf]](https://arxiv.org/pdf/2601.16984)</summary>

**Abstract:** The 3rd Generation Partnership Project (3GPP) produces complex technical specifications essential to global telecommunications, yet their hierarchical structure, dense formatting, and multi-modal content make them difficult to process. While Large Language Models (LLMs) show promise, existing approaches fall short in handling complex queries, visual information, and document interdependencies. We present TelcoAI, an agentic, multi-modal Retrieval-Augmented Generation (RAG) system tailored for 3GPP documentation. TelcoAI introduces section-aware chunking, structured query planning, metadata-guided retrieval, and multi-modal fusion of text and diagrams. Evaluated on multiple benchmarks-including expert-curated queries-our system achieves $87\%$ recall, $83\%$ claim recall, and $92\%$ faithfulness, representing a $16\%$ improvement over state-of-the-art baselines. These results demonstrate the effectiveness of agentic and multi-modal reasoning in technical document understanding, advancing practical solutions for real-world telecommunications research and engineering.

**arXiv ID:** 2601.16984
</details>

<details>
<summary><strong>Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations</strong> - Preethi Seshadri, Samuel Cahyawijaya, Ayomide Odumakinde, Sameer Singh, Seraphina Goldfarb-Tarrant - [[pdf]](https://arxiv.org/pdf/2601.17087)</summary>

**Abstract:** Agentic benchmarks increasingly rely on LLM-simulated users to scalably evaluate agent performance, yet the robustness, validity, and fairness of this approach remain unexamined. Through a user study with participants across the United States, India, Kenya, and Nigeria, we investigate whether LLM-simulated users serve as reliable proxies for real human users in evaluating agents on {\tau}-Bench retail tasks. We find that user simulation lacks robustness, with agent success rates varying up to 9 percentage points across different user LLMs. Furthermore, evaluations using simulated users exhibit systematic miscalibration, underestimating agent performance on challenging tasks and overestimating it on moderately difficult ones. African American Vernacular English (AAVE) speakers experience consistently worse success rates and calibration errors than Standard American English (SAE) speakers, with disparities compounding significantly with age. We also find simulated users to be a differentially effective proxy for different populations, performing worst for AAVE and Indian English speakers. Additionally, simulated users introduce conversational artifacts and surface different failure patterns than human users. These findings demonstrate that current evaluation practices risk misrepresenting agent capabilities across diverse user populations and may obscure real-world deployment challenges.

**arXiv ID:** 2601.17087
</details>

<details>
<summary><strong>How AI Coding Agents Modify Code: A Large-Scale Study of GitHub Pull Requests</strong> - Daniel Ogenrwot, John Businge - [[pdf]](https://arxiv.org/pdf/2601.17581)</summary>

**Abstract:** AI coding agents are increasingly acting as autonomous contributors by generating and submitting pull requests (PRs). However, we lack empirical evidence on how these agent-generated PRs differ from human contributions, particularly in how they modify code and describe their changes. Understanding these differences is essential for assessing their reliability and impact on development workflows. Using the MSR 2026 Mining Challenge version of the AIDev dataset, we analyze 24,014 merged Agentic PRs (440,295 commits) and 5,081 merged Human PRs (23,242 commits). We examine additions, deletions, commits, and files touched, and evaluate the consistency between PR descriptions and their diffs using lexical and semantic similarity. Agentic PRs differ substantially from Human PRs in commit count (Cliff's $\delta = 0.5429$) and show moderate differences in files touched and deleted lines. They also exhibit slightly higher description-to-diff similarity across all measures. These findings provide a large-scale empirical characterization of how AI coding agents contribute to open source development.

**arXiv ID:** 2601.17581
</details>

<details>
<summary><strong>The Script is All You Need: An Agentic Framework for Long-Horizon Dialogue-to-Cinematic Video Generation</strong> - Chenyu Mu, Xin He, Qu Yang, Wanshun Chen, Jiadi Yao, Huang Liu, Zihao Yi, Bo Zhao, Xingyu Chen, Ruotian Ma, Fanghua Ye, Erkun Yang, Cheng Deng, Zhaopeng Tu, Xiaolong Li, Linus - [[pdf]](https://arxiv.org/pdf/2601.17737)</summary>

**Abstract:** Recent advances in video generation have produced models capable of synthesizing stunning visual content from simple text prompts. However, these models struggle to generate long-form, coherent narratives from high-level concepts like dialogue, revealing a ``semantic gap'' between a creative idea and its cinematic execution. To bridge this gap, we introduce a novel, end-to-end agentic framework for dialogue-to-cinematic-video generation. Central to our framework is ScripterAgent, a model trained to translate coarse dialogue into a fine-grained, executable cinematic script. To enable this, we construct ScriptBench, a new large-scale benchmark with rich multimodal context, annotated via an expert-guided pipeline. The generated script then guides DirectorAgent, which orchestrates state-of-the-art video models using a cross-scene continuous generation strategy to ensure long-horizon coherence. Our comprehensive evaluation, featuring an AI-powered CriticAgent and a new Visual-Script Alignment (VSA) metric, shows our framework significantly improves script faithfulness and temporal fidelity across all tested video models. Furthermore, our analysis uncovers a crucial trade-off in current SOTA models between visual spectacle and strict script adherence, providing valuable insights for the future of automated filmmaking.

**arXiv ID:** 2601.17737
</details>

<details>
<summary><strong>Linguistic and Argument Diversity in Synthetic Data for Function-Calling Agents</strong> - Dan Greenstein, Zohar Karnin, Chen Amiraz, Oren Somekh - [[pdf]](https://arxiv.org/pdf/2601.17829)</summary>

**Abstract:** The construction of function calling agents has emerged as a promising avenue for extending model capabilities. A major challenge for this task is obtaining high quality diverse data for training. Prior work emphasizes diversity in functions, invocation patterns, and interaction turns, yet linguistic diversity of requests and coverage of arguments (e.g., \texttt{city\_name}, \texttt{stock\_ticker}) remain underexplored. We propose a method that generates synthetic datasets via optimizing general-purpose diversity metrics across both queries and arguments, without relying on hand-crafted rules or taxonomies, making it robust to different usecases. We demonstrate the effectiveness of our technique via both intrinsic and extrinsic testing, comparing it to SoTA data generation methods. We show a superiority over baselines in terms of diversity, while keeping comparable correctness. Additionally, when used as a training set, the model resulting from our dataset exhibits superior performance compared to analogous models based on the baseline data generation methods in out-of-distribution performance. In particular, we achieve an $7.4\%$ increase in accuracy on the BFCL benchmark compared to similar counterparts.

**arXiv ID:** 2601.17829
</details>

<details>
<summary><strong>Self-Manager: Parallel Agent Loop for Long-form Deep Research</strong> - Yilong Xu, Zhi Zheng, Xiang Long, Yujun Cai, Yiwei Wang - [[pdf]](https://arxiv.org/pdf/2601.17879)</summary>

**Abstract:** Long-form deep research requires multi-faceted investigations over extended horizons to get a comprehensive report. When handling such complex tasks, existing agents manage context at the subtask level to overcome linear context accumulation and information loss. However, they still adhere to a single context window and sequential execution paradigm, which results in mutual interference and blocking behavior, restricting scalability and adaptability. To address this issue, this paper introduces Self-Manager, a parallel agent loop that enables asynchronous and concurrent execution. The main thread can create multiple subthreads, each with its own isolated context, and manage them iteratively through Thread Control Blocks, allowing for more focused and flexible parallel agent execution. To assess its effectiveness, we benchmark Self-Manager on DeepResearch Bench, where it consistently outperforms existing single-agent loop baselines across all metrics. Furthermore, we conduct extensive analytical experiments to demonstrate the necessity of Self-Manager's design choices, as well as its advantages in contextual capacity, efficiency, and generalization.

**arXiv ID:** 2601.17879
</details>

<details>
<summary><strong>MalURLBench: A Benchmark Evaluating Agents' Vulnerabilities When Processing Web URLs</strong> - Dezhang Kong, Zhuxi Wu, Shiqi Liu, Zhicheng Tan, Kuichen Lu, Minghao Li, Qichen Liu, Shengyu Chu, Zhenhua Xu, Xuan Liu, Meng Han - [[pdf]](https://arxiv.org/pdf/2601.18113)</summary>

**Abstract:** LLM-based web agents have become increasingly popular for their utility in daily life and work. However, they exhibit critical vulnerabilities when processing malicious URLs: accepting a disguised malicious URL enables subsequent access to unsafe webpages, which can cause severe damage to service providers and users. Despite this risk, no benchmark currently targets this emerging threat. To address this gap, we propose MalURLBench, the first benchmark for evaluating LLMs' vulnerabilities to malicious URLs. MalURLBench contains 61,845 attack instances spanning 10 real-world scenarios and 7 categories of real malicious websites. Experiments with 12 popular LLMs reveal that existing models struggle to detect elaborately disguised malicious URLs. We further identify and analyze key factors that impact attack success rates and propose URLGuard, a lightweight defense module. We believe this work will provide a foundational resource for advancing the security of web agents. Our code is available at this https URL.

**arXiv ID:** 2601.18113
</details>

<details>
<summary><strong>SWE-EVO: Benchmarking Coding Agents in Long-Horizon Software Evolution Scenarios</strong> - Minh V. T. Thai, Tue Le, Dung Nguyen Manh, Huy Phan Nhat, Nghi D. Q. Bui - [[pdf]](https://arxiv.org/pdf/2512.18470)</summary>

**Abstract:** Existing benchmarks for AI coding agents focus on isolated, single-issue tasks such as fixing a bug or implementing a small feature. However, real-world software engineering is fundamentally a long-horizon endeavor: developers must interpret high-level requirements, plan coordinated changes across many files, and evolve codebases over multiple iterations while preserving existing functionality. We introduce SWE-EVO, a benchmark that evaluates agents on this long-horizon software evolution challenge. Constructed from release notes and version histories of seven mature open-source Python projects, SWE-EVO comprises 48 evolution tasks that require agents to implement multi-step modifications spanning an average of 21 files, validated against comprehensive test suites averaging 874 tests per instance. Experiments with state-of-the-art models reveal a striking capability gap: even GPT-5 with OpenHands achieves only a 21 percent resolution rate on SWE-EVO, compared to 65 percent on the single-issue SWE-Bench Verified. This demonstrates that current agents struggle with sustained, multi-file reasoning. We also propose Fix Rate, a fine-grained metric that captures partial progress toward solving these complex, long-horizon tasks.

**arXiv ID:** 2512.18470
</details>

<details>
<summary><strong>CooperBench: Why Coding Agents Cannot be Your Teammates Yet</strong> - Arpandeep Khatua, Hao Zhu, Peter Tran, Arya Prabhudesai, Frederic Sadrieh, Johann K. Lieberwirth, Xinkai Yu, Yicheng Fu, Michael J. Ryan, Jiaxin Pei, Diyi Yang - [[pdf]](https://arxiv.org/pdf/2601.13295)</summary>

**Abstract:** Resolving team conflicts requires not only task-specific competence, but also social intelligence to find common ground and build consensus. As AI agents increasingly collaborate on complex work, they must develop coordination capabilities to function as effective teammates. Yet we hypothesize that current agents lack these capabilities. To test this, we introduce CooperBench, a benchmark of over 600 collaborative coding tasks across 12 libraries in 4 programming languages. Each task assigns two agents different features that can be implemented independently but may conflict without proper coordination. Tasks are grounded in real open-source repositories with expert-written tests. Evaluating state-of-the-art coding agents, we observe the curse of coordination: agents achieve on average 30% lower success rates when working together compared to performing both tasks individually. This contrasts sharply with human teams, where adding teammates typically improves productivity. Our analysis reveals three key issues: (1) communication channels become jammed with vague, ill-timed, and inaccurate messages; (2) even with effective communication, agents deviate from their commitments; and (3) agents often hold incorrect expectations about others' plans and communication. Through large-scale simulation, we also observe rare but interesting emergent coordination behavior including role division, resource division, and negotiation. Our research presents a novel benchmark for collaborative coding and calls for a shift from pursuing individual agent capability to developing social intelligence.

**arXiv ID:** 2601.13295
</details>

<details>
<summary><strong>U-Fold: Dynamic Intent-Aware Context Folding for User-Centric Agents</strong> - Jin Su, Runnan Fang, Yeqiu Li, Xiaobin Wang, Shihao Cai, Pengjun Xie, Ningyu Zhang, Fajie Yuan - [[pdf]](https://arxiv.org/pdf/2601.18285)</summary>

**Abstract:** Large language model (LLM)-based agents have been successfully deployed in many tool-augmented settings, but their scalability is fundamentally constrained by context length. Existing context-folding methods mitigate this issue by summarizing past interactions, yet they are typically designed for single-query or single-intent scenarios. In more realistic user-centric dialogues, we identify two major failure modes: (i) they irreversibly discard fine-grained constraints and intermediate facts that are crucial for later decisions, and (ii) their summaries fail to track evolving user intent, leading to omissions and erroneous actions. To address these limitations, we propose U-Fold, a dynamic context-folding framework tailored to user-centric tasks. U-Fold retains the full user--agent dialogue and tool-call history but, at each turn, uses two core components to produce an intent-aware, evolving dialogue summary and a compact, task-relevant tool log. Extensive experiments on $\tau$-bench, $\tau^2$-bench, VitaBench, and harder context-inflated settings show that U-Fold consistently outperforms ReAct (achieving a 71.4% win rate in long-context settings) and prior folding baselines (with improvements of up to 27.0%), particularly on long, noisy, multi-turn tasks. Our study demonstrates that U-Fold is a promising step toward transferring context-management techniques from single-query benchmarks to realistic user-centric applications.

**arXiv ID:** 2601.18285
</details>

<details>
<summary><strong>RECAP: REwriting Conversations for Intent Understanding in Agentic Planning</strong> - Kushan Mitra, Dan Zhang, Hannah Kim, Estevam Hruschka - [[pdf]](https://arxiv.org/pdf/2509.04472)</summary>

**Abstract:** Understanding user intent is essential for effective planning in conversational assistants, particularly those powered by large language models (LLMs) coordinating multiple agents. However, real-world dialogues are often ambiguous, underspecified, or dynamic, making intent detection a persistent challenge. Traditional classification-based approaches struggle to generalize in open-ended settings, leading to brittle interpretations and poor downstream planning. We propose RECAP (REwriting Conversations for Agent Planning), a new benchmark designed to evaluate and advance intent rewriting, reframing user-agent dialogues into concise representations of user goals. RECAP captures diverse challenges such as ambiguity, intent drift, vagueness, and mixed-goal conversations. Alongside the dataset, we introduce an LLM-based evaluator that assesses planning utility given the rewritten intent. Using RECAP, we develop a prompt-based rewriting approach that outperforms baselines, in terms of plan preference. We further demonstrate that fine-tuning two DPO-based rewriters yields additional utility gains. Our results highlight intent rewriting as a critical and tractable component for improving agentic planning in open-domain dialogue systems.

**arXiv ID:** 2509.04472
</details>

<details>
<summary><strong>Agentic Very Long Video Understanding</strong> - Aniket Rege, Arka Sadhu, Yuliang Li, Kejie Li, Ramya Korlakai Vinayak, Yuning Chai, Yong Jae Lee, Hyo Jin Kim - [[pdf]](https://arxiv.org/pdf/2601.18157)</summary>

**Abstract:** The advent of always-on personal AI assistants, enabled by all-day wearable devices such as smart glasses, demands a new level of contextual understanding, one that goes beyond short, isolated events to encompass the continuous, longitudinal stream of egocentric video. Achieving this vision requires advances in long-horizon video understanding, where systems must interpret and recall visual and audio information spanning days or even weeks. Existing methods, including large language models and retrieval-augmented generation, are constrained by limited context windows and lack the ability to perform compositional, multi-hop reasoning over very long video streams. In this work, we address these challenges through EGAgent, an enhanced agentic framework centered on entity scene graphs, which represent people, places, objects, and their relationships over time. Our system equips a planning agent with tools for structured search and reasoning over these graphs, as well as hybrid visual and audio search capabilities, enabling detailed, cross-modal, and temporally coherent reasoning. Experiments on the EgoLifeQA and Video-MME (Long) datasets show that our method achieves state-of-the-art performance on EgoLifeQA (57.5%) and competitive performance on Video-MME (Long) (74.1%) for complex longitudinal video understanding tasks.

**arXiv ID:** 2601.18157
</details>

<details>
<summary><strong>AsterNav: Autonomous Aerial Robot Navigation In Darkness Using Passive Computation</strong> - Deepak Singh, Shreyas Khobragade, Nitin J. Sanket - [[pdf]](https://arxiv.org/pdf/2601.17550)</summary>

**Abstract:** Autonomous aerial navigation in absolute darkness is crucial for post-disaster search and rescue operations, which often occur from disaster-zone power outages. Yet, due to resource constraints, tiny aerial robots, perfectly suited for these operations, are unable to navigate in the darkness to find survivors safely. In this paper, we present an autonomous aerial robot for navigation in the dark by combining an Infra-Red (IR) monocular camera with a large-aperture coded lens and structured light without external infrastructure like GPS or motion-capture. Our approach obtains depth-dependent defocus cues (each structured light point appears as a pattern that is depth dependent), which acts as a strong prior for our AsterNet deep depth estimation model. The model is trained in simulation by generating data using a simple optical model and transfers directly to the real world without any fine-tuning or retraining. AsterNet runs onboard the robot at 20 Hz on an NVIDIA Jetson Orin$^\text{TM}$ Nano. Furthermore, our network is robust to changes in the structured light pattern and relative placement of the pattern emitter and IR camera, leading to simplified and cost-effective construction. We successfully evaluate and demonstrate our proposed depth navigation approach AsterNav using depth from AsterNet in many real-world experiments using only onboard sensing and computation, including dark matte obstacles and thin ropes (diameter 6.25mm), achieving an overall success rate of 95.5% with unknown object shapes, locations and materials. To the best of our knowledge, this is the first work on monocular, structured-light-based quadrotor navigation in absolute darkness.

**arXiv ID:** 2601.17550
</details>

<details>
<summary><strong>Foundation Models in Autonomous Driving: A Survey on Scenario Generation and Scenario Analysis</strong> - Yuan Gao, Mattia Piccinini, Yuchen Zhang, Dingrui Wang, Korbinian Moller, Roberto Brusnicki, Baha Zarrouki, Alessio Gambi, Jan Frederik Totz, Kai Storms, Steven Peters, Andrea Stocco, Bassam Alrifaee, Marco Pavone, Johannes Betz - [[pdf]](https://arxiv.org/pdf/2506.11526)</summary>

**Abstract:** For autonomous vehicles, safe navigation in complex environments depends on handling a broad range of diverse and rare driving scenarios. Simulation- and scenario-based testing have emerged as key approaches to development and validation of autonomous driving systems. Traditional scenario generation relies on rule-based systems, knowledge-driven models, and data-driven synthesis, often producing limited diversity and unrealistic safety-critical cases. With the emergence of foundation models, which represent a new generation of pre-trained, general-purpose AI models, developers can process heterogeneous inputs (e.g., natural language, sensor data, HD maps, and control actions), enabling the synthesis and interpretation of complex driving scenarios. In this paper, we conduct a survey about the application of foundation models for scenario generation and scenario analysis in autonomous driving (as of May 2025). Our survey presents a unified taxonomy that includes large language models, vision-language models, multimodal large language models, diffusion models, and world models for the generation and analysis of autonomous driving scenarios. In addition, we review the methodologies, open-source datasets, simulation platforms, and benchmark challenges, and we examine the evaluation metrics tailored explicitly to scenario generation and analysis. Finally, the survey concludes by highlighting the open challenges and research questions, and outlining promising future research directions. All reviewed papers are listed in a continuously maintained repository, which contains supplementary materials and is available at this https URL.

**arXiv ID:** 2506.11526
</details>

<details>
<summary><strong>Towards Real-time Adaptation of Embodied Agent in Human-Robot Collaboration</strong> - Shipeng Liu, Boshen Zhang, Zhehui Huang - [[pdf]](https://arxiv.org/pdf/2412.00435)</summary>

**Abstract:** Large Language Models (LLMs) have opened transformative possibilities for human-robot collaboration. However, enabling real-time collaboration requires both low latency and robust reasoning, and most LLMs suffer from high latency. To address this gap, we first propose a fine-grained benchmark that explicitly assesses agents' proactive adaptability and temporal responsiveness in the Overcooked-AI environment. Based on evaluation results, we propose MonTA (Monitor-then-Adapt), a hierarchical framework inspired by cognitive science research. MonTA contains three key modules: a lightweight Monitor that operates at high frequency (7 Hz) to detect adaptation needs, and two proficient Adapters for subtask and path adaptation reasoning that provide instructions to humans at a lower frequency. Our results demonstrate that MonTA significantly outperforms baseline agents on our proposed benchmark, achieving superior performance across layouts with varying teaming fluency. User studies confirm the high reasonableness of adaptation plans and consistent language instructions provided by our framework to humans.

**arXiv ID:** 2412.00435
</details>

</details>

<details open>
<summary><h2>LLM Agents (7 papers)</h2></summary>

<details>
<summary><strong>ReFuGe: Feature Generation for Prediction Tasks on Relational Databases with LLM Agents</strong> - Kyungho Kim, Geon Lee, Juyeon Kim, Dongwon Choi, Shinhwan Kang, Kijung Shin - [[pdf]](https://arxiv.org/pdf/2601.17735)</summary>

**Abstract:** Relational databases (RDBs) play a crucial role in many real-world web applications, supporting data management across multiple interconnected tables. Beyond typical retrieval-oriented tasks, prediction tasks on RDBs have recently gained attention. In this work, we address this problem by generating informative relational features that enhance predictive performance. However, generating such features is challenging: it requires reasoning over complex schemas and exploring a combinatorially large feature space, all without explicit supervision. To address these challenges, we propose ReFuGe, an agentic framework that leverages specialized large language model agents: (1) a schema selection agent identifies the tables and columns relevant to the task, (2) a feature generation agent produces diverse candidate features from the selected schema, and (3) a feature filtering agent evaluates and retains promising features through reasoning-based and validation-based filtering. It operates within an iterative feedback loop until performance converges. Experiments on RDB benchmarks demonstrate that ReFuGe substantially improves performance on various RDB prediction tasks. Our code and datasets are available at this https URL.

**arXiv ID:** 2601.17735
</details>

<details>
<summary><strong>Sentipolis: Emotion-Aware Agents for Social Simulations</strong> - Chiyuan Fu, Lyuhao Chen, Yunze Xiao, Weihao Xuan, Carlos Busso, Mona Diab - [[pdf]](https://arxiv.org/pdf/2601.18027)</summary>

**Abstract:** LLM agents are increasingly used for social simulation, yet emotion is often treated as a transient cue, causing emotional amnesia and weak long-horizon continuity. We present Sentipolis, a framework for emotionally stateful agents that integrates continuous Pleasure-Arousal-Dominance (PAD) representation, dual-speed emotion dynamics, and emotion--memory coupling. Across thousands of interactions over multiple base models and evaluators, Sentipolis improves emotionally grounded behavior, boosting communication, and emotional continuity. Gains are model-dependent: believability increases for higher-capacity models but can drop for smaller ones, and emotion-awareness can mildly reduce adherence to social norms, reflecting a human-like tension between emotion-driven behavior and rule compliance in social simulation. Network-level diagnostics show reciprocal, moderately clustered, and temporally stable relationship structures, supporting the study of cumulative social dynamics such as alliance formation and gradual relationship change.

**arXiv ID:** 2601.18027
</details>

<details>
<summary><strong>Paying Less Generalization Tax: A Cross-Domain Generalization Study of RL Training for LLM Agents</strong> - Zhihan Liu, Lin Guan, Yixin Nie, Kai Zhang, Zhuoqun Hao, Lin Chen, Asli Celikyilmaz, Zhaoran Wang, Na Zhang - [[pdf]](https://arxiv.org/pdf/2601.18217)</summary>

**Abstract:** Generalist LLM agents are often post-trained on a narrow set of environments but deployed across far broader, unseen domains. In this work, we investigate the challenge of agentic post-training when the eventual test domains are unknown. Specifically, we analyze which properties of reinforcement learning (RL) environments and modeling choices have the greatest influence on out-of-domain performance. First, we identify two environment axes that strongly correlate with cross-domain generalization: (i) state information richness, i.e., the amount of information for the agent to process from the state, and (ii) planning complexity, estimated via goal reachability and trajectory length under a base policy. Notably, domain realism and text-level similarity are not the primary factors; for instance, the simple grid-world domain Sokoban leads to even stronger generalization in SciWorld than the more realistic ALFWorld. Motivated by these findings, we further show that increasing state information richness alone can already effectively improve cross-domain robustness. We propose a randomization technique, which is low-overhead and broadly applicable: add small amounts of distractive goal-irrelevant features to the state to make it richer without altering the task. Beyond environment-side properties, we also examine several modeling choices: (a) SFT warmup or mid-training helps prevent catastrophic forgetting during RL but undermines generalization to domains that are not included in the mid-training datamix; and (b) turning on step-by-step thinking during RL, while not always improving in-domain performance, plays a crucial role in preserving generalization.

**arXiv ID:** 2601.18217
</details>

<details>
<summary><strong>ShopSimulator: Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants</strong> - Pei Wang, Yanan Wu, Xiaoshuai Song, Weixun Wang, Gengru Chen, Zhongwen Li, Kezhong Yan, Ken Deng, Qi Liu, Shuaibing Zhao, Shaopan Xiong, Xuepeng Liu, Xuefeng Chen, Wanxi Deng, Wenbo Su, Bo Zheng - [[pdf]](https://arxiv.org/pdf/2601.18225)</summary>

**Abstract:** Large language model (LLM)-based agents are increasingly deployed in e-commerce shopping. To perform thorough, user-tailored product searches, agents should interpret personal preferences, engage in multi-turn dialogues, and ultimately retrieve and discriminate among highly similar products. However, existing research has yet to provide a unified simulation environment that consistently captures all of these aspects, and always focuses solely on evaluation benchmarks without training support. In this paper, we introduce ShopSimulator, a large-scale and challenging Chinese shopping environment. Leveraging ShopSimulator, we evaluate LLMs across diverse scenarios, finding that even the best-performing models achieve less than 40% full-success rate. Error analysis reveals that agents struggle with deep search and product selection in long trajectories, fail to balance the use of personalization cues, and to effectively engage with users. Further training exploration provides practical guidance for overcoming these weaknesses, with the combination of supervised fine-tuning (SFT) and reinforcement learning (RL) yielding significant performance improvements. Code and data will be released at this https URL.

**arXiv ID:** 2601.18225
</details>

<details>
<summary><strong>Breaking the Protocol: Security Analysis of the Model Context Protocol Specification and Prompt Injection Vulnerabilities in Tool-Integrated LLM Agents</strong> - Narek Maloyan, Dmitry Namiot - [[pdf]](https://arxiv.org/pdf/2601.17549)</summary>

**Abstract:** The Model Context Protocol (MCP) has emerged as a de facto standard for integrating Large Language Models with external tools, yet no formal security analysis of the protocol specification exists. We present the first rigorous security analysis of MCP's architectural design, identifying three fundamental protocol-level vulnerabilities: (1) absence of capability attestation allowing servers to claim arbitrary permissions, (2) bidirectional sampling without origin authentication enabling server-side prompt injection, and (3) implicit trust propagation in multi-server configurations. We implement \textsc{MCPBench}, a novel framework bridging existing agent security benchmarks to MCP-compliant infrastructure, enabling direct measurement of protocol-specific attack surfaces. Through controlled experiments on 847 attack scenarios across five MCP server implementations, we demonstrate that MCP's architectural choices amplify attack success rates by 23--41\% compared to equivalent non-MCP integrations. We propose \textsc{MCPSec}, a backward-compatible protocol extension adding capability attestation and message authentication, reducing attack success rates from 52.8\% to 12.4\% with median latency overhead of 8.3ms per message. Our findings establish that MCP's security weaknesses are architectural rather than implementation-specific, requiring protocol-level remediation.

**arXiv ID:** 2601.17549
</details>

<details>
<summary><strong>The Shadow Self: Intrinsic Value Misalignment in Large Language Model Agents</strong> - Chen Chen, Kim Young Il, Yuan Yang, Wenhao Su, Yilin Zhang, Xueluan Gong, Qian Wang, Yongsen Zheng, Ziyao Liu, Kwok-Yan Lam - [[pdf]](https://arxiv.org/pdf/2601.17344)</summary>

**Abstract:** Large language model (LLM) agents with extended autonomy unlock new capabilities, but also introduce heightened challenges for LLM safety. In particular, an LLM agent may pursue objectives that deviate from human values and ethical norms, a risk known as value misalignment. Existing evaluations primarily focus on responses to explicit harmful input or robustness against system failure, while value misalignment in realistic, fully benign, and agentic settings remains largely underexplored. To fill this gap, we first formalize the Loss-of-Control risk and identify the previously underexamined Intrinsic Value Misalignment (Intrinsic VM). We then introduce IMPRESS (Intrinsic Value Misalignment Probes in REalistic Scenario Set), a scenario-driven framework for systematically assessing this risk. Following our framework, we construct benchmarks composed of realistic, fully benign, and contextualized scenarios, using a multi-stage LLM generation pipeline with rigorous quality control. We evaluate Intrinsic VM on 21 state-of-the-art LLM agents and find that it is a common and broadly observed safety risk across models. Moreover, the misalignment rates vary by motives, risk types, model scales, and architectures. While decoding strategies and hyperparameters exhibit only marginal influence, contextualization and framing mechanisms significantly shape misalignment behaviors. Finally, we conduct human verification to validate our automated judgments and assess existing mitigation strategies, such as safety prompting and guardrails, which show instability or limited effectiveness. We further demonstrate key use cases of IMPRESS across the AI Ecosystem. Our code and benchmark will be publicly released upon acceptance.

**arXiv ID:** 2601.17344
</details>

<details>
<summary><strong>Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates</strong> - Yibo Li, Zijie Lin, Ailin Deng, Xuan Zhang, Yufei He, Shuo Ji, Tri Cao, Bryan Hooi - [[pdf]](https://arxiv.org/pdf/2601.18510)</summary>

**Abstract:** While Large Language Model (LLM) agents excel at general tasks, they inherently struggle with continual adaptation due to the frozen weights after deployment. Conventional reinforcement learning (RL) offers a solution but incurs prohibitive computational costs and the risk of catastrophic forgetting. We introduce Just-In-Time Reinforcement Learning (JitRL), a training-free framework that enables test-time policy optimization without any gradient updates. JitRL maintains a dynamic, non-parametric memory of experiences and retrieves relevant trajectories to estimate action advantages on-the-fly. These estimates are then used to directly modulate the LLM's output logits. We theoretically prove that this additive update rule is the exact closed-form solution to the KL-constrained policy optimization objective. Extensive experiments on WebArena and Jericho demonstrate that JitRL establishes a new state-of-the-art among training-free methods. Crucially, JitRL outperforms the performance of computationally expensive fine-tuning methods (e.g., WebRL) while reducing monetary costs by over 30 times, offering a scalable path for continual learning agents. The code is available at this https URL.

**arXiv ID:** 2601.18510
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (31 papers)</h2></summary>

<details>
<summary><strong>Phase Transition for Budgeted Multi-Agent Synergy</strong> - Bang Liu, Linglong Kong, Jian Pei - [[pdf]](https://arxiv.org/pdf/2601.17311)</summary>

**Abstract:** Multi-agent systems can improve reliability, yet under a fixed inference budget they often help, saturate, or even collapse. We develop a minimal and calibratable theory that predicts these regimes from three binding constraints of modern agent stacks: finite context windows, lossy inter-agent communication, and shared failures among similar agents. Each leaf agent is summarized by a compute-performance scaling exponent $\beta$; communication is captured by a message-length fidelity curve $\gamma(m)$; dependence is captured by an effective shared-error correlation $\rho$; and a context window $W$ imposes hard fan-in limits that make hierarchy necessary. For binary success/failure tasks with majority aggregation, we prove a sharp phase transition for deep $b$-ary trees with correlated inputs and lossy communication: a single scalar $\alpha_\rho$ (combining $\gamma(m)$, $\rho$, and fan-in $b$) determines whether weak signal is amplified to a nontrivial fixed point or washed out to chance. In the amplifying regime, we derive an organization exponent $s$ and show that budgeted synergy, i.e., outperforming the best single agent under the same total budget, occurs exactly when $s>\beta$, yielding closed-form compute allocation rules and explicit budget thresholds. We further characterize saturation via a mixing depth and provide a conservative clipped predictor that remains accurate across growth and saturation. A continuous-performance warm-up gives closed-form risks for star, chain, and tree organizations, making correlation- and communication-induced floors explicit and exposing the core design trade-offs in a smooth setting. Finally, we validate the predicted phase boundaries in controlled synthetic simulations and show how the same mechanisms explain the dominant bottlenecks reported in recent large-scale matched-budget studies of LLM agent-system scaling.

**arXiv ID:** 2601.17311
</details>

<details>
<summary><strong>Multi-Agent Learning Path Planning via LLMs</strong> - Haoxin Xu, Changyong Qi, Tong Liu, Bohao Zhang, Anna He, Bingqian Jiang, Longwei Zheng, Xiaoqing Gu - [[pdf]](https://arxiv.org/pdf/2601.17346)</summary>

**Abstract:** The integration of large language models (LLMs) into intelligent tutoring systems offers transformative potential for personalized learning in higher education. However, most existing learning path planning approaches lack transparency, adaptability, and learner-centered explainability. To address these challenges, this study proposes a novel Multi-Agent Learning Path Planning (MALPP) framework that leverages a role- and rule-based collaboration mechanism among intelligent agents, each powered by LLMs. The framework includes three task-specific agents: a learner analytics agent, a path planning agent, and a reflection agent. These agents collaborate via structured prompts and predefined rules to analyze learning profiles, generate tailored learning paths, and iteratively refine them with interpretable feedback. Grounded in Cognitive Load Theory and Zone of Proximal Development, the system ensures that recommended paths are cognitively aligned and pedagogically meaningful. Experiments conducted on the MOOCCubeX dataset using seven LLMs show that MALPP significantly outperforms baseline models in path quality, knowledge sequence consistency, and cognitive load alignment. Ablation studies further validate the effectiveness of the collaborative mechanism and theoretical constraints. This research contributes to the development of trustworthy, explainable AI in education and demonstrates a scalable approach to learner-centered adaptive instruction powered by LLMs.

**arXiv ID:** 2601.17346
</details>

<details>
<summary><strong>DIML: Differentiable Inverse Mechanism Learning from Behaviors of Multi-Agent Learning Trajectories</strong> - Zhiyu An, Wan Du - [[pdf]](https://arxiv.org/pdf/2601.17678)</summary>

**Abstract:** We study inverse mechanism learning: recovering an unknown incentive-generating mechanism from observed strategic interaction traces of self-interested learning agents. Unlike inverse game theory and multi-agent inverse reinforcement learning, which typically infer utility/reward parameters inside a structured mechanism, our target includes unstructured mechanism -- a (possibly neural) mapping from joint actions to per-agent payoffs. Unlike differentiable mechanism design, which optimizes mechanisms forward, we infer mechanisms from behavior in an observational setting. We propose DIML, a likelihood-based framework that differentiates through a model of multi-agent learning dynamics and uses the candidate mechanism to generate counterfactual payoffs needed to predict observed actions. We establish identifiability of payoff differences under a conditional logit response model and prove statistical consistency of maximum likelihood estimation under standard regularity conditions. We evaluate DIML with simulated interactions of learning agents across unstructured neural mechanisms, congestion tolling, public goods subsidies, and large-scale anonymous games. DIML reliably recovers identifiable incentive differences and supports counterfactual prediction, where its performance rivals tabular enumeration oracle in small environments and its convergence scales to large, hundred-participant environments. Code to reproduce our experiments is open-sourced.

**arXiv ID:** 2601.17678
</details>

<details>
<summary><strong>Faramesh: A Protocol-Agnostic Execution Control Plane for Autonomous Agent Systems</strong> - Amjad Fatmi - [[pdf]](https://arxiv.org/pdf/2601.17744)</summary>

**Abstract:** Autonomous agent systems increasingly trigger real-world side effects: deploying infrastructure, modifying databases, moving money, and executing workflows. Yet most agent stacks provide no mandatory execution checkpoint where organizations can deterministically permit, deny, or defer an action before it changes reality. This paper introduces Faramesh, a protocol-agnostic execution control plane that enforces execution-time authorization for agent-driven actions via a non-bypassable Action Authorization Boundary (AAB). Faramesh canonicalizes agent intent into a Canonical Action Representation (CAR), evaluates actions deterministically against policy and state, and issues a decision artifact (PERMIT/DEFER/DENY) that executors must validate prior to execution. The system is designed to be framework- and model-agnostic, supports multi-agent and multi-tenant deployments, and remains independent of transport protocols (e.g., MCP). Faramesh further provides decision-centric, append-only provenance logging keyed by canonical action hashes, enabling auditability, verification, and deterministic replay without re-running agent reasoning. We show how these primitives yield enforceable, predictable governance for autonomous execution while avoiding hidden coupling to orchestration layers or observability-only approaches.

**arXiv ID:** 2601.17744
</details>

<details>
<summary><strong>Why Keep Your Doubts to Yourself? Trading Visual Uncertainties in Multi-Agent Bandit Systems</strong> - Jusheng Zhang, Yijia Fan, Kaitong Cai, Jing Yang, Jiawei Yao, Jian Wang, Guanlong Qu, Ziliang Chen, Keze Wang - [[pdf]](https://arxiv.org/pdf/2601.18735)</summary>

**Abstract:** Vision-Language Models (VLMs) enable powerful multi-agent systems, but scaling them is economically unsustainable: coordinating heterogeneous agents under information asymmetry often spirals costs. Existing paradigms, such as Mixture-of-Agents and knowledge-based routers, rely on heuristic proxies that ignore costs and collapse uncertainty structure, leading to provably suboptimal coordination. We introduce Agora, a framework that reframes coordination as a decentralized market for uncertainty. Agora formalizes epistemic uncertainty into a structured, tradable asset (perceptual, semantic, inferential), and enforces profitability-driven trading among agents based on rational economic rules. A market-aware broker, extending Thompson Sampling, initiates collaboration and guides the system toward cost-efficient equilibria. Experiments on five multimodal benchmarks (MMMU, MMBench, MathVision, InfoVQA, CC-OCR) show that Agora outperforms strong VLMs and heuristic multi-agent strategies, e.g., achieving +8.5% accuracy over the best baseline on MMMU while reducing cost by over 3x. These results establish market-based coordination as a principled and scalable paradigm for building economically viable multi-agent visual intelligence systems.

**arXiv ID:** 2601.18735
</details>

<details>
<summary><strong>Multi-Agent Deep Reinforcement Learning Under Constrained Communications</strong> - Shahil Shaik, Jonathon M. Smereka, Yue Wang - [[pdf]](https://arxiv.org/pdf/2601.17069)</summary>

**Abstract:** Centralized training with decentralized execution (CTDE) has been the dominant paradigm in multi-agent reinforcement learning (MARL), but its reliance on global state information during training introduces scalability, robustness, and generalization bottlenecks. Moreover, in practical scenarios such as adding/dropping teammates or facing environment dynamics that differ from the training, CTDE methods can be brittle and costly to retrain, whereas distributed approaches allow agents to adapt using only local information and peer-to-peer communication. We present a distributed MARL framework that removes the need for centralized critics or global information. Firstly, we develop a novel Distributed Graph Attention Network (D-GAT) that performs global state inference through multi-hop communication, where agents integrate neighbor features via input-dependent attention weights in a fully distributed manner. Leveraging D-GAT, we develop the distributed graph-attention MAPPO (DG-MAPPO) -- a distributed MARL framework where agents optimize local policies and value functions using local observations, multi-hop communication, and shared/averaged rewards. Empirical evaluation on the StarCraftII Multi-Agent Challenge, Google Research Football, and Multi-Agent Mujoco demonstrates that our method consistently outperforms strong CTDE baselines, achieving superior coordination across a wide range of cooperative tasks with both homogeneous and heterogeneous teams. Our distributed MARL framework provides a principled and scalable solution for robust collaboration, eliminating the need for centralized training or global observability. To the best of our knowledge, DG-MAPPO appears to be the first to fully eliminate reliance on privileged centralized information, enabling agents to learn and act solely through peer-to-peer communication.

**arXiv ID:** 2601.17069
</details>

<details>
<summary><strong>ChemNavigator: Agentic AI Discovery of Design Rules for Organic Photocatalysts</strong> - Iman Peivaste, Ahmed Makradi, Salim Belouettar - [[pdf]](https://arxiv.org/pdf/2601.17084)</summary>

**Abstract:** The discovery of high-performance organic photocatalysts for hydrogen evolution remains limited by the vastness of chemical space and the reliance on human intuition for molecular design. Here we present ChemNavigator, an agentic AI system that autonomously derives structure-property relationships through hypothesis-driven exploration of organic photocatalyst candidates. The system integrates large language model reasoning with density functional tight binding calculations in a multi-agent architecture that mirrors the scientific method: formulating hypotheses, designing experiments, executing calculations, and validating findings through rigorous statistical analysis. Through iterative discovery cycles encompassing 200 molecules, ChemNavigator autonomously identified six statistically significant design rules governing frontier orbital energies, including the effects of ether linkages, carbonyl groups, extended conjugation, cyano groups, halogen substituents, and amine groups. Importantly, these rules correspond to established principles of organic electronic structure (resonance donation, inductive withdrawal, $\pi$-delocalization), demonstrating that the system can independently derive chemical knowledge without explicit programming. Notably, autonomous agentic reasoning extracted these six validated rules from a molecular library where previous ML approaches identified only carbonyl effects. Furthermore, the quantified effect sizes provide a prioritized ranking for synthetic chemists, while feature interaction analysis revealed diminishing returns when combining strategies, challenging additive assumptions in molecular design. This work demonstrates that agentic AI systems can autonomously derive interpretable, chemically grounded design principles, establishing a framework for AI-assisted materials discovery that complements rather than replaces chemical intuition.

**arXiv ID:** 2601.17084
</details>

<details>
<summary><strong>Dynamic Role Assignment for Multi-Agent Debate</strong> - Miao Zhang, Junsik Kim, Siyuan Xiang, Jian Gao, Cheng Cao - [[pdf]](https://arxiv.org/pdf/2601.17152)</summary>

**Abstract:** Multi-agent large language model (LLM) and vision-language model (VLM) debate systems employ specialized roles for complex problem-solving, yet model specializations are not leveraged to decide which model should fill which role. We propose dynamic role assignment, a framework that runs a Meta-Debate to select suitable agents before the actual debate. The meta-debate has two stages: (1) proposal, where candidates provide role-tailored arguments, and (2) peer review, where proposals are scored with data and role-specific criteria to choose the best agent for each position. We evaluate our method on LLM problem solving benchmarks. Applied on top of existing debate systems, our approach consistently outperforms uniform assignments (filling all roles with the same model) by up to 74.8% and random assignments (assigning models to roles without considering their suitability) by up to 29.7%, depending on the task and the specific assignment. This work establishes a new paradigm for multi-agent system design, shifting from static agent deployment to dynamic and capability-aware selection.

**arXiv ID:** 2601.17152
</details>

<details>
<summary><strong>Decentralized Multi-Agent Swarms for Autonomous Grid Security in Industrial IoT: A Consensus-based Approach</strong> - Samaresh Kumar Singh, Joyjit Roy - [[pdf]](https://arxiv.org/pdf/2601.17303)</summary>

**Abstract:** As Industrial Internet of Things (IIoT) environments expand to include tens of thousands of connected devices. The centralization of security monitoring architectures creates serious latency issues that savvy attackers can exploit to compromise an entire manufacturing ecosystem. This paper outlines a new, decentralized multi-agent swarm (DMAS) architecture that includes autonomous artificial intelligence (AI) agents at each edge gateway, functioning as a distributed digital "immune system" for IIoT networks. Instead of using a traditional static firewall approach, the DMAS agents communicate via a lightweight peer-to-peer protocol to cooperatively detect anomalous behavior across the IIoT network without sending data to a cloud infrastructure. The authors also outline a consensus-based threat validation (CVT) process in which agents vote on the threat level of an identified threat, enabling instant quarantine of a compromised node or nodes. The authors conducted experiments on a testbed that simulated an innovative factory environment with 2000 IIoT devices and found that the DMAS demonstrated sub-millisecond response times (average of 0.85ms), 97.3% accuracy in detecting malicious activity under high load, and 87% accuracy in detecting zero-day attacks. All significantly higher than baseline values for both centralized and edge computing. Additionally, the proposed architecture can prevent real-time cascading failures in industrial control systems and reduce network bandwidth use by 89% compared to cloud-based solutions.

**arXiv ID:** 2601.17303
</details>

<details>
<summary><strong>Towards a Declarative Agentic Layer for Intelligent Agents in MCP-Based Server Ecosystems</strong> - Maria Jesus Rodriguez-Sanchez, Manuel Noguera, Angel Ruiz-Zafra, Kawtar Benghazi - [[pdf]](https://arxiv.org/pdf/2601.17435)</summary>

**Abstract:** Recent advances in Large Language Models (LLMs) have enabled the development of increasingly complex agentic and multi-agent systems capable of planning, tool use and task decomposition. However, empirical evidence shows that many of these systems suffer from fundamental reliability issues, including hallucinated actions, unexecutable plans and brittle coordination. Crucially, these failures do not stem from limitations of the underlying models themselves, but from the absence of explicit architectural structure linking goals, capabilities and execution. This paper presents a declarative, model-independent architectural layer for grounded agentic workflows that addresses this gap. The proposed layer, referred to as DALIA (Declarative Agentic Layer for Intelligent Agents), formalises executable capabilities, exposes tasks through a declarative discovery protocol, maintains a federated directory of agents and their execution resources, and constructs deterministic task graphs grounded exclusively in declared operations. By enforcing a clear separation between discovery, planning and execution, the architecture constrains agent behaviour to a verifiable operational space, reducing reliance on speculative reasoning and free-form coordination. We present the architecture and design principles of the proposed layer and illustrate its operation through a representative task-oriented scenario, demonstrating how declarative grounding enables reproducible and verifiable agentic workflows across heterogeneous environments.

**arXiv ID:** 2601.17435
</details>

<details>
<summary><strong>Embodiment-Induced Coordination Regimes in Tabular Multi-Agent Q-Learning</strong> - Muhammad Ahmed Atif, Nehal Naeem Haji, Mohammad Shahid Shaikh, Muhammad Ebad Atif - [[pdf]](https://arxiv.org/pdf/2601.17454)</summary>

**Abstract:** Centralized value learning is often assumed to improve coordination and stability in multi-agent reinforcement learning, yet this assumption is rarely tested under controlled conditions. We directly evaluate it in a fully tabular predator-prey gridworld by comparing independent and centralized Q-learning under explicit embodiment constraints on agent speed and stamina. Across multiple kinematic regimes and asymmetric agent roles, centralized learning fails to provide a consistent advantage and is frequently outperformed by fully independent learning, even under full observability and exact value estimation. Moreover, asymmetric centralized-independent configurations induce persistent coordination breakdowns rather than transient learning instability. By eliminating confounding effects from function approximation and representation learning, our tabular analysis isolates coordination structure as the primary driver of these effects. The results show that increased coordination can become a liability under embodiment constraints, and that the effectiveness of centralized learning is fundamentally regime and role dependent rather than universal.

**arXiv ID:** 2601.17454
</details>

<details>
<summary><strong>VissimRL: A Multi-Agent Reinforcement Learning Framework for Traffic Signal Control Based on Vissim</strong> - Hsiao-Chuan Chang, Sheng-You Huang, Yen-Chi Chen, I-Chen Wu - [[pdf]](https://arxiv.org/pdf/2601.18284)</summary>

**Abstract:** Traffic congestion remains a major challenge for urban transportation, leading to significant economic and environmental impacts. Traffic Signal Control (TSC) is one of the key measures to mitigate congestion, and recent studies have increasingly applied Reinforcement Learning (RL) for its adaptive capabilities. With respect to SUMO and CityFlow, the simulator Vissim offers high-fidelity driver behavior modeling and wide industrial adoption but remains underutilized in RL research due to its complex interface and lack of standardized frameworks. To address this gap, this paper proposes VissimRL, a modular RL framework for TSC that encapsulates Vissim's COM interface through a high-level Python API, offering standardized environments for both single- and multi-agent training. Experiments show that VissimRL significantly reduces development effort while maintaining runtime efficiency, and supports consistent improvements in traffic performance during training, as well as emergent coordination in multi-agent control. Overall, VissimRL demonstrates the feasibility of applying RL in high-fidelity simulations and serves as a bridge between academic research and practical applications in intelligent traffic signal control.

**arXiv ID:** 2601.18284
</details>

<details>
<summary><strong>Emergent Cooperation in Quantum Multi-Agent Reinforcement Learning Using Communication</strong> - Michael Kölle, Christian Reff, Leo Sünkel, Julian Hager, Gerhard Stenzel, Claudia Linnhoff-Popien - [[pdf]](https://arxiv.org/pdf/2601.18419)</summary>

**Abstract:** Emergent cooperation in classical Multi-Agent Reinforcement Learning has gained significant attention, particularly in the context of Sequential Social Dilemmas (SSDs). While classical reinforcement learning approaches have demonstrated capability for emergent cooperation, research on extending these methods to Quantum Multi-Agent Reinforcement Learning remains limited, particularly through communication. In this paper, we apply communication approaches to quantum Q-Learning agents: the Mutual Acknowledgment Token Exchange (MATE) protocol, its extension Mutually Endorsed Distributed Incentive Acknowledgment Token Exchange (MEDIATE), the peer rewarding mechanism Gifting, and Reinforced Inter-Agent Learning (RIAL). We evaluate these approaches in three SSDs: the Iterated Prisoner's Dilemma, Iterated Stag Hunt, and Iterated Game of Chicken. Our experimental results show that approaches using MATE with temporal-difference measure (MATE\textsubscript{TD}), AutoMATE, MEDIATE-I, and MEDIATE-S achieved high cooperation levels across all dilemmas, demonstrating that communication is a viable mechanism for fostering emergent cooperation in Quantum Multi-Agent Reinforcement Learning.

**arXiv ID:** 2601.18419
</details>

<details>
<summary><strong>LLM-Enhanced Multi-Agent Reinforcement Learning with Expert Workflow for Real-Time P2P Energy Trading</strong> - Chengwei Lou, Zekai Jin, Wei Tang, Guangfei Geng, Jin Yang, Lu Zhang - [[pdf]](https://arxiv.org/pdf/2507.14995)</summary>

**Abstract:** Real-time peer-to-peer (P2P) electricity markets dynamically adapt to fluctuations in renewable energy and variations in demand, maximizing economic benefits through instantaneous price responses while enhancing grid flexibility. However, scaling expert guidance for massive personalized prosumers poses critical challenges, including diverse decision-making demands and a lack of customized modeling frameworks. This paper proposes an integrated large language model-multi-agent reinforcement learning (LLM-MARL) framework for real-time P2P energy trading to address challenges such as the limited technical capability of prosumers, the lack of expert experience, and security issues of distribution networks. LLMs are introduced as experts to generate personalized strategies, guiding MARL under the centralized training with decentralized execution (CTDE) paradigm through imitation. To handle the scalability issues inherent in large-scale P2P networks, a differential attention-based critic network is introduced to efficiently extract key interaction features and enhance convergence. Experimental results demonstrate that LLM-generated strategies effectively substitute human experts. The proposed imitative expert MARL algorithms achieve significantly lower economic costs and voltage violation rates on test sets compared to baseline algorithms, while maintaining robust stability. This paper provides an effective solution for the real-time decision-making of the P2P electricity market by bridging expert knowledge with agent learning.

**arXiv ID:** 2507.14995
</details>

<details>
<summary><strong>TrustResearcher: Automating Knowledge-Grounded and Transparent Research Ideation with Multi-Agent Collaboration</strong> - Jiawei Zhou, Ruicheng Zhu, Mengshi Chen, Jianwei Wang, Kai Wang - [[pdf]](https://arxiv.org/pdf/2510.20844)</summary>

**Abstract:** Agentic systems have recently emerged as a promising tool to automate literature-based ideation. However, current systems often remain black-box, with limited transparency or control for researchers. Our work introduces TrustResearcher, a multi-agent demo system for knowledge-grounded and transparent ideation. Specifically, TrustResearcher integrates meticulously designed four stages into a unified framework: (A) Structured Knowledge Curation, (B) Diversified Idea Generation, (C) Multi-stage Idea Selection, and (D) Expert Panel Review and Synthesis. Different from prior pipelines, our system not only exposes intermediate reasoning states, execution logs, and configurable agents for inspections, but also enables diverse and evidence-aligned idea generation. Our design is also domain-agnostic, where the same pipeline can be instantiated in any scientific field. As an illustrative case, we demonstrate TrustResearcher on a graph-mining scenario (k-truss breaking problem), where it generates distinct, plausible candidates with evidence and critiques. A live demo and source code are available at this https URL

**arXiv ID:** 2510.20844
</details>

<details>
<summary><strong>Agent Contracts: A Formal Framework for Resource-Bounded Autonomous AI Systems</strong> - Qing Ye, Jing Tan - [[pdf]](https://arxiv.org/pdf/2601.08815)</summary>

**Abstract:** The Contract Net Protocol (1980) introduced coordination through contracts in multi-agent systems. Modern agent protocols standardize connectivity and interoperability; yet, none provide formal, resource governance-normative mechanisms to bound how much agents may consume or how long they may operate. We introduce Agent Contracts, a formal framework that extends the contract metaphor from task allocation to resource-bounded execution. An Agent Contract unifies input/output specifications, multi-dimensional resource constraints, temporal boundaries, and success criteria into a coherent governance mechanism with explicit lifecycle semantics. For multi-agent coordination, we establish conservation laws ensuring delegated budgets respect parent constraints, enabling hierarchical coordination through contract delegation. Empirical validation across four experiments demonstrates 90% token reduction with 525x lower variance in iterative workflows, zero conservation violations in multi-agent delegation, and measurable quality-resource tradeoffs through contract modes. Agent Contracts provide formal foundations for predictable, auditable, and resource-bounded autonomous AI deployment.

**arXiv ID:** 2601.08815
</details>

<details>
<summary><strong>MAViS: A Multi-Agent Framework for Long-Sequence Video Storytelling</strong> - Qian Wang, Ziqi Huang, Ruoxi Jia, Paul Debevec, Ning Yu - [[pdf]](https://arxiv.org/pdf/2508.08487)</summary>

**Abstract:** Despite recent advances, long-sequence video generation frameworks still suffer from significant limitations: poor assistive capability, suboptimal visual quality, and limited expressiveness. To mitigate these limitations, we propose MAViS, a multi-agent collaborative framework designed to assist in long-sequence video storytelling by efficiently translating ideas into visual narratives. MAViS orchestrates specialized agents across multiple stages, including script writing, shot designing, character modeling, keyframe generation, video animation, and audio generation. In each stage, agents operate under the 3E Principle -- Explore, Examine, and Enhance -- to ensure the completeness of intermediate outputs. Considering the capability limitations of current generative models, we propose the Script Writing Guidelines to optimize compatibility between scripts and generative tools. Experimental results demonstrate that MAViS achieves state-of-the-art performance in assistive capability, visual quality, and video expressiveness. Its modular framework further enables scalability with diverse generative models and tools. With just a brief idea description, MAViS enables users to rapidly explore diverse visual storytelling and creative directions for sequential video generation by efficiently producing high-quality, complete long-sequence videos. To the best of our knowledge, MAViS is the only framework that provides multimodal design output -- videos with narratives and background music.

**arXiv ID:** 2508.08487
</details>

<details>
<summary><strong>Collaborative Belief Reasoning with LLMs for Efficient Multi-Agent Collaboration</strong> - Zhimin Wang, Duo Wu, Shaokang He, Jinghe Wang, Linjia Kang, Jing Yu, Zhi Wang - [[pdf]](https://arxiv.org/pdf/2509.21981)</summary>

**Abstract:** Effective real-world multi-agent collaboration requires not only accurate planning but also the ability to reason about collaborators' intents--a crucial capability for avoiding miscoordination and redundant communication under partial observable environments. Due to their strong planning and reasoning capabilities, large language models (LLMs) have emerged as promising autonomous agents for collaborative task solving. However, existing collaboration frameworks for LLMs overlook their reasoning potential for dynamic intent inference, and thus produce inconsistent plans and redundant communication, reducing collaboration efficiency. To bridge this gap, we propose CoBel-World, a novel framework that equips LLM agents with a Collaborative Belief World--an internal representation jointly modeling the physical environment and collaborators' mental states. CoBel-World enables agents to parse external open-world knowledge into structured beliefs via a symbolic belief representation module, and perform zero-shot Bayesian-style belief updates through LLM reasoning. This allows agents to proactively detect potential miscoordination (e.g., conflicting plans) and communicate adaptively. Evaluated on challenging embodied benchmarks (i.e., TDW-MAT and C-WAH), CoBel-World significantly reduces communication costs by 64-79% and improves task completion efficiency by 4-28% compared to the strongest baseline. Our results show that explicit, intent-aware belief modeling is essential for efficient and human-like collaboration in LLM-based multi-agent systems.

**arXiv ID:** 2509.21981
</details>

<details>
<summary><strong>RAM-SD: Retrieval-Augmented Multi-agent framework for Sarcasm Detection</strong> - Ziyang Zhou, Ziqi Liu, Yan Wang, Yiming Lin, Yangbin Chen - [[pdf]](https://arxiv.org/pdf/2601.17002)</summary>

**Abstract:** Sarcasm detection remains a significant challenge due to its reliance on nuanced contextual understanding, world knowledge, and multi-faceted linguistic cues that vary substantially across different sarcastic expressions. Existing approaches, from fine-tuned transformers to large language models, apply a uniform reasoning strategy to all inputs, struggling to address the diverse analytical demands of sarcasm. These demands range from modeling contextual expectation violations to requiring external knowledge grounding or recognizing specific rhetorical patterns. To address this limitation, we introduce RAM-SD, a Retrieval-Augmented Multi-Agent framework for Sarcasm Detection. The framework operates through four stages: (1) contextual retrieval grounds the query in both sarcastic and non-sarcastic exemplars; (2) a meta-planner classifies the sarcasm type and selects an optimal reasoning plan from a predefined set; (3) an ensemble of specialized agents performs complementary, multi-view analysis; and (4) an integrator synthesizes these analyses into a final, interpretable judgment with a natural language explanation. Evaluated on four standard benchmarks, RAM-SD achieves a state-of-the-art Macro-F1 of 77.74%, outperforming the strong GPT-4o+CoC baseline by 7.01 points. Our framework not only sets a new performance benchmark but also provides transparent and interpretable reasoning traces, illuminating the cognitive processes behind sarcasm comprehension.

**arXiv ID:** 2601.17002
</details>

<details>
<summary><strong>EFT-CoT: A Multi-Agent Chain-of-Thought Framework for Emotion-Focused Therapy</strong> - Lanqing Du, Yunong Li, YuJie Long, Shihong Chen - [[pdf]](https://arxiv.org/pdf/2601.17842)</summary>

**Abstract:** Leveraging Large Language Models (LLMs) for Mental Health Question Answering (MHQA) is promising for mitigating resource shortages. However, existing Cognitive Behavioral Therapy (CBT)-based approaches predominantly favor a "top-down" rational restructuring, often neglecting clients' embodied experiences and primary emotion processing. To address this, we propose an Emotion-Focused Therapy (EFT)-based Multi-Agent Chain-of-Thought framework (EFT-CoT). Adopting a "bottom-up" trajectory, it deconstructs the intervention into a three-stage reasoning flow: "Embodied Perception - Cognitive Exploration - Narrative Intervention." Utilizing eight specialized agents, the system explicitly executes critical components such as somatic awareness mapping, adaptive assessment, core belief extraction, and narrative restructuring. We further constructed "EFT-Instruct," a high-quality dataset via Chain-of-Thought distillation of approximately 67,000 authentic texts, and fine-tuned a specialized model, EFT-LLM. Experimental evaluations demonstrate that EFT-LLM outperforms strong baselines and human responses across metrics like empathy depth and structural professionalism. Ablation studies confirm the necessity of the multi-agent mechanism. The model exhibits superior psychological reasoning, offering an effective pathway for interpretable, high-empathy counseling systems.

**arXiv ID:** 2601.17842
</details>

<details>
<summary><strong>Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents</strong> - Mahesh Ramesh, Kaousheik Jayakumar, Aswinkumar Ramkumar, Pavan Thodima, Aniket Rege - [[pdf]](https://arxiv.org/pdf/2601.18077)</summary>

**Abstract:** Cooperative reasoning under incomplete information remains challenging for both humans and multi-agent systems. The card game Hanabi embodies this challenge, requiring theory-of-mind reasoning and strategic communication. We benchmark 17 state-of-the-art LLM agents in 2-5 player games and study the impact of context engineering across model scales (4B to 600B+) to understand persistent coordination failures and robustness to scaffolding: from a minimal prompt with only explicit card details (Watson setting), to scaffolding with programmatic, Bayesian-motivated deductions (Sherlock setting), to multi-turn state tracking via working memory (Mycroft setting). We show that (1) agents can maintain an internal working memory for state tracking and (2) cross-play performance between different LLMs smoothly interpolates with model strength. In the Sherlock setting, the strongest reasoning models exceed 15 points on average across player counts, yet still trail experienced humans and specialist Hanabi agents, both consistently scoring above 20. We release the first public Hanabi datasets with annotated trajectories and move utilities: (1) HanabiLogs, containing 1,520 full game logs for instruction tuning, and (2) HanabiRewards, containing 560 games with dense move-level value annotations for all candidate moves. Supervised and RL finetuning of a 4B open-weight model (Qwen3-Instruct) on our datasets improves cooperative Hanabi play by 21% and 156% respectively, bringing performance to within ~3 points of a strong proprietary reasoning model (o4-mini) and surpassing the best non-reasoning model (GPT-4.1) by 52%. The HanabiRewards RL-finetuned model further generalizes beyond Hanabi, improving performance on a cooperative group-guessing benchmark by 11%, temporal reasoning on EventQA by 6.4%, instruction-following on IFBench-800K by 1.7 Pass@10, and matching AIME 2025 mathematical reasoning Pass@10.

**arXiv ID:** 2601.18077
</details>

<details>
<summary><strong>MultiVis-Agent: A Multi-Agent Framework with Logic Rules for Reliable and Comprehensive Cross-Modal Data Visualization</strong> - Jinwei Lu, Yuanfeng Song, Chen Zhang, Raymond Chi-Wing Wong - [[pdf]](https://arxiv.org/pdf/2601.18320)</summary>

**Abstract:** Real-world visualization tasks involve complex, multi-modal requirements that extend beyond simple text-to-chart generation, requiring reference images, code examples, and iterative refinement. Current systems exhibit fundamental limitations: single-modality input, one-shot generation, and rigid workflows. While LLM-based approaches show potential for these complex requirements, they introduce reliability challenges including catastrophic failures and infinite loop susceptibility. To address this gap, we propose MultiVis-Agent, a logic rule-enhanced multi-agent framework for reliable multi-modal and multi-scenario visualization generation. Our approach introduces a four-layer logic rule framework that provides mathematical guarantees for system reliability while maintaining flexibility. Unlike traditional rule-based systems, our logic rules are mathematical constraints that guide LLM reasoning rather than replacing it. We formalize the MultiVis task spanning four scenarios from basic generation to iterative refinement, and develop MultiVis-Bench, a benchmark with over 1,000 cases for multi-modal visualization evaluation. Extensive experiments demonstrate that our approach achieves 75.63% visualization score on challenging tasks, significantly outperforming baselines (57.54-62.79%), with task completion rates of 99.58% and code execution success rates of 94.56% (vs. 74.48% and 65.10% without logic rules), successfully addressing both complexity and reliability challenges in automated visualization generation.

**arXiv ID:** 2601.18320
</details>

<details>
<summary><strong>LegalMALR:Multi-Agent Query Understanding and LLM-Based Reranking for Chinese Statute Retrieval</strong> - Yunhan Li, Mingjie Xie, Gaoli Kang, Zihan Gong, Gengshen Wu, Min Yang - [[pdf]](https://arxiv.org/pdf/2601.17692)</summary>

**Abstract:** Statute retrieval is essential for legal assistance and judicial decision support, yet real-world legal queries are often implicit, multi-issue, and expressed in colloquial or underspecified forms. These characteristics make it difficult for conventional retrieval-augmented generation pipelines to recover the statutory elements required for accurate retrieval. Dense retrievers focus primarily on the literal surface form of the query, whereas lightweight rerankers lack the legal-reasoning capacity needed to assess statutory applicability. We present LegalMALR, a retrieval framework that integrates a Multi-Agent Query Understanding System (MAS) with a zero-shot large-language-model-based reranking module (LLM Reranker). MAS generates diverse, legally grounded reformulations and conducts iterative dense retrieval to broaden candidate coverage. To stabilise the stochastic behaviour of LLM-generated rewrites, we optimise a unified MAS policy using Generalized Reinforcement Policy Optimization(GRPO). The accumulated candidate set is subsequently evaluated by the LLM Reranker, which performs natural-language legal reasoning to produce the final ranking. We further construct CSAID, a dataset of 118 difficult Chinese legal queries annotated with multiple statutory labels, and evaluate LegalMALR on both CSAID and the public STARD benchmark. Experiments show that LegalMALR substantially outperforms strong Retrieval-augmented generation(RAG) baselines in both in-distribution and out-of-distribution settings, demonstrating the effectiveness of combining multi-perspective query interpretation, reinforcement-based policy optimisation, and large-model reranking for statute retrieval.

**arXiv ID:** 2601.17692
</details>

<details>
<summary><strong>Multi-Agent Collaborative Filtering: Orchestrating Users and Items for Agentic Recommendations</strong> - Yu Xia, Sungchul Kim, Tong Yu, Ryan A. Rossi, Julian McAuley - [[pdf]](https://arxiv.org/pdf/2511.18413)</summary>

**Abstract:** Agentic recommendations cast recommenders as large language model (LLM) agents that can plan, reason, use tools, and interact with users of varying preferences in web applications. However, most existing agentic recommender systems focus on generic single-agent plan-execute workflows or multi-agent task decomposition pipelines. Without recommendation-oriented design, they often underuse the collaborative signals in the user-item interaction history, leading to unsatisfying recommendation results. To address this, we propose the Multi-Agent Collaborative Filtering (MACF) framework for agentic recommendations, drawing an analogy between traditional collaborative filtering algorithms and LLM-based multi-agent collaboration. Specifically, given a target user and query, we instantiate similar users and relevant items as LLM agents with unique profiles. Each agent is able to call retrieval tools, suggest candidate items, and interact with other agents. Different from the static preference aggregation in traditional collaborative filtering, MACF employs a central orchestrator agent to adaptively manage the collaboration between user and item agents via dynamic agent recruitment and personalized collaboration instruction. Experimental results on datasets from three different domains show the advantages of our MACF framework compared to strong agentic recommendation baselines.

**arXiv ID:** 2511.18413
</details>

<details>
<summary><strong>Advances and Innovations in the Multi-Agent Robotic System (MARS) Challenge</strong> - Li Kang, Heng Zhou, Xiufeng Song, Rui Li, Bruno N.Y. Chen, Ziye Wang, Ximeng Meng, Stone Tao, Yiran Qin, Xiaohong Liu, Ruimao Zhang, Lei Bai, Yilun Du, Hao Su, Philip Torr, Zhenfei Yin, Ruihao Gong, Yejun Zeng, Fengjun Zhong, Shenghao Jin, Jinyang Guo, Xianglong Liu, Xiaojun Jia, Tianqi Shan, Wenqi Ren, Simeng Qin, Jialing Yang, Xiaoyu Ma, Tianxing Chen, Zixuan Li, Zijian Cai, Yan Qin, Yusen Qin, Qiangyu Chen, Kaixuan Wang, Zhaoming Han, Yao Mu, Ping Luo, Yuanqi Yao, Haoming Song, Jan-Nico Zaech, Fabien Despinoy, Danda Pani Paudel, Luc Van Gool - [[pdf]](https://arxiv.org/pdf/2601.18733)</summary>

**Abstract:** Recent advancements in multimodal large language models and vision-languageaction models have significantly driven progress in Embodied AI. As the field transitions toward more complex task scenarios, multi-agent system frameworks are becoming essential for achieving scalable, efficient, and collaborative solutions. This shift is fueled by three primary factors: increasing agent capabilities, enhancing system efficiency through task delegation, and enabling advanced human-agent interactions. To address the challenges posed by multi-agent collaboration, we propose the Multi-Agent Robotic System (MARS) Challenge, held at the NeurIPS 2025 Workshop on SpaVLE. The competition focuses on two critical areas: planning and control, where participants explore multi-agent embodied planning using vision-language models (VLMs) to coordinate tasks and policy execution to perform robotic manipulation in dynamic environments. By evaluating solutions submitted by participants, the challenge provides valuable insights into the design and coordination of embodied multi-agent systems, contributing to the future development of advanced collaborative AI systems.

**arXiv ID:** 2601.18733
</details>

<details>
<summary><strong>ORION: Option-Regularized Deep Reinforcement Learning for Cooperative Multi-Agent Online Navigation</strong> - Shizhe Zhang, Jingsong Liang, Zhitao Zhou, Shuhan Ye, Yizhuo Wang, Ming Siang Derek Tan, Jimmy Chiun, Yuhong Cao, Guillaume Sartoretti - [[pdf]](https://arxiv.org/pdf/2601.01155)</summary>

**Abstract:** Existing methods for multi-agent navigation typically assume fully known environments, offering limited support for partially known scenarios such as warehouses or factory floors. There, agents may need to plan trajectories that balance their own path optimality with their ability to collect and share information about the environment that can help their teammates reach their own goals. To these ends, we propose ORION, a novel deep reinforcement learning framework for cooperative multi-agent online navigation in partially known environments. Starting from an imperfect prior map, ORION trains agents to make decentralized decisions, coordinate to reach their individual targets, and actively reduce map uncertainty by sharing online observations in a closed perception-action loop. We first design a shared graph encoder that fuses prior map with online perception into a unified representation, providing robust state embeddings under dynamic map discrepancies. At the core of ORION is an option-critic framework that learns to reason about a set of high-level cooperative modes that translate into sequences of low-level actions, allowing agents to switch between individual navigation and team-level exploration adaptively. We further introduce a dual-stage cooperation strategy that enables agents to assist teammates under map uncertainty, thereby reducing the overall makespan. Across extensive maze-like maps and large-scale warehouse environments, our simulation results show that ORION achieves high-quality, real-time decentralized cooperation over varying team sizes, outperforming state-of-the-art classical and learning-based baselines. Finally, we validate ORION on physical robot teams, demonstrating its robustness and practicality for real-world cooperative navigation.

**arXiv ID:** 2601.01155
</details>

<details>
<summary><strong>DecompGAIL: Learning Realistic Traffic Behaviors with Decomposed Multi-Agent Generative Adversarial Imitation Learning</strong> - Ke Guo, Haochen Liu, Xiaojun Wu, Chen Lv - [[pdf]](https://arxiv.org/pdf/2510.06913)</summary>

**Abstract:** Realistic traffic simulation is critical for the development of autonomous driving systems and urban mobility planning, yet existing imitation learning approaches often fail to model realistic traffic behaviors. Behavior cloning suffers from covariate shift, while Generative Adversarial Imitation Learning (GAIL) is notoriously unstable in multi-agent settings. We identify a key source of this instability: irrelevant interaction misguidance, where a discriminator penalizes an ego vehicle's realistic behavior due to unrealistic interactions among its neighbors. To address this, we propose Decomposed Multi-agent GAIL (DecompGAIL), which explicitly decomposes realism into ego-map and ego-neighbor components, filtering out misleading neighbor: neighbor and neighbor: map interactions. We further introduce a social PPO objective that augments ego rewards with distance-weighted neighborhood rewards, encouraging overall realism across agents. Integrated into a lightweight SMART-based backbone, DecompGAIL achieves state-of-the-art performance on the WOMD Sim Agents 2025 benchmark.

**arXiv ID:** 2510.06913
</details>

<details>
<summary><strong>Probing the Future of Meta-Analysis: Eliciting Design Principles via an Agentic Research IDE</strong> - Sizhe Cheng, Feng Liang, Yuhan Wen, Xipei Yu, Yong Wang - [[pdf]](https://arxiv.org/pdf/2601.18239)</summary>

**Abstract:** Meta-analyses and systematic reviews demand rigorous abductive reasoning to build, test, and refine hypotheses across vast, heterogeneous literature. While NLP advancements have automated parts of this pipeline, existing tools often detach researchers from the cognitive loop or function merely as retrieval engines, leading to loss of intellectual ownership and frequent context switching. We present Research IDE, a prototype reimagining authoring environments through the "Research as Code" metaphor. Research IDE embeds a multi-agent backend into the writing flow, enabling in-situ verification via "hypothesis breakpoints." A one-week field deployment with 8 domain experts, followed by a reflective workshop, as a Research through Design (RtD) probe, reveals that users strongly preferred this verification workflow, actively leveraged prior knowledge for confirmation, and reported that breakpoints sparked insights. Drawing from participant feedback and suggestions, we derive design implications for future AI-assisted research tools that fully preserve researcher autonomy and intellectual ownership while harnessing computational scale.

**arXiv ID:** 2601.18239
</details>

<details>
<summary><strong>When Nobody Around Is Real: Exploring Public Opinions and User Experiences On the Multi-Agent AI Social Platform</strong> - Qiufang Yu, Mengmeng Wu, Xingyu Lan - [[pdf]](https://arxiv.org/pdf/2601.18275)</summary>

**Abstract:** Powered by large language models, a new genre of multi-agent social platforms has emerged. Apps such as this http URL deploy numerous AI agents that emulate human behavior, creating unprecedented bot-centric social networks. Yet, existing research has predominantly focused on one-on-one chatbots, leaving multi-agent AI platforms underexplored. To bridge this gap, we took this http URL as a case study and performed a two-stage investigation: (i) content analysis of 883 user comments; (ii) a 7-day diary study with 20 participants to document their firsthand platform experiences. While public discourse expressed greater skepticism, the diary study found that users did project a range of social expectations onto the AI agents. While some user expectations were met, the AI-dominant social environment introduces distinct problems, such as attention overload and homogenized interaction. These tensions signal a future where AI functions not merely as a tool or an anthropomorphized actor, but as the dominant medium of sociality itself-a paradigm shift that foregrounds new forms of architected social life.

**arXiv ID:** 2601.18275
</details>

<details>
<summary><strong>World Craft: Agentic Framework to Create Visualizable Worlds via Text</strong> - Jianwen Sun, Yukang Feng, Kaining Ying, Chuanhao Li, Zizhen Li, Fanrui Zhang, Jiaxin Ai, Yifan Chang, Yu Dai, Yifei Huang, Kaipeng Zhang - [[pdf]](https://arxiv.org/pdf/2601.09150)</summary>

**Abstract:** Large Language Models (LLMs) motivate generative agent simulation (e.g., AI Town) to create a ``dynamic world'', holding immense value across entertainment and research. However, for non-experts, especially those without programming skills, it isn't easy to customize a visualizable environment by themselves. In this paper, we introduce World Craft, an agentic world creation framework to create an executable and visualizable AI Town via user textual descriptions. It consists of two main modules, World Scaffold and World Guild. World Scaffold is a structured and concise standardization to develop interactive game scenes, serving as an efficient scaffolding for LLMs to customize an executable AI Town-like environment. World Guild is a multi-agent framework to progressively analyze users' intents from rough descriptions, and synthesizes required structured contents (\eg environment layout and assets) for World Scaffold . Moreover, we construct a high-quality error-correction dataset via reverse engineering to enhance spatial knowledge and improve the stability and controllability of layout generation, while reporting multi-dimensional evaluation metrics for further analysis. Extensive experiments demonstrate that our framework significantly outperforms existing commercial code agents (Cursor and Antigravity) and LLMs (Qwen3 and Gemini-3-Pro). in scene construction and narrative intent conveyance, providing a scalable solution for the democratization of environment creation.

**arXiv ID:** 2601.09150
</details>

<details>
<summary><strong>Strategic Tradeoffs Between Humans and AI in Multi-Agent Bargaining</strong> - Crystal Qian, Kehang Zhu, John Horton, Benjamin S. Manning, Vivian Tsai, James Wexler, Nithum Thain - [[pdf]](https://arxiv.org/pdf/2509.09071)</summary>

**Abstract:** Markets increasingly accommodate large language models (LLMs) as autonomous decision-making agents. As this transition occurs, it becomes critical to evaluate how these agents behave relative to their human and task-specific statistical predecessors. In this work, we present results from an empirical study comparing humans (N=216), multiple frontier LLMs, and customized Bayesian agents in dynamic multi-player bargaining games under identical conditions. Bayesian agents extract the highest surplus with aggressive trade proposals that are frequently rejected. Humans and LLMs achieve comparable aggregate surplus within their groups, but exhibit different trading strategies. LLMs favor conservative, concessionary proposals that are usually accepted by other LLMs, while humans propose trades that are consistent with fairness norms but are more likely to be rejected. These findings highlight that performance parity -- a common benchmark in agent evaluation -- can mask substantive procedural differences in how LLMs behave in complex multi-agent interactions.

**arXiv ID:** 2509.09071
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>RouteMoA: Dynamic Routing without Pre-Inference Boosts Efficient Mixture-of-Agents</strong> - Jize Wang, Han Wu, Zhiyuan You, Yiming Song, Yijun Wang, Zifei Shan, Yining Li, Songyang Zhang, Xinyi Le, Cailian Chen, Xinping Guan, Dacheng Tao - [[pdf]](https://arxiv.org/pdf/2601.18130)</summary>

**Abstract:** Mixture-of-Agents (MoA) improves LLM performance through layered collaboration, but its dense topology raises costs and latency. Existing methods employ LLM judges to filter responses, yet still require all models to perform inference before judging, failing to cut costs effectively. They also lack model selection criteria and struggle with large model pools, where full inference is costly and can exceed context limits. To address this, we propose RouteMoA, an efficient mixture-of-agents framework with dynamic routing. It employs a lightweight scorer to perform initial screening by predicting coarse-grained performance from the query, narrowing candidates to a high-potential subset without inference. A mixture of judges then refines these scores through lightweight self- and cross-assessment based on existing model outputs, providing posterior correction without additional inference. Finally, a model ranking mechanism selects models by balancing performance, cost, and latency. RouteMoA outperforms MoA across varying tasks and model pool sizes, reducing cost by 89.8% and latency by 63.6% in the large-scale model pool.

**arXiv ID:** 2601.18130
</details>

<details>
<summary><strong>Spatiotemporal Semantic V2X Framework for Cooperative Collision Prediction</strong> - Murat Arda Onsu, Poonam Lohan, Burak Kantarci, Aisha Syed, Matthew Andrews, Sean Kennedy - [[pdf]](https://arxiv.org/pdf/2601.17216)</summary>

**Abstract:** Intelligent Transportation Systems (ITS) demand real-time collision prediction to ensure road safety and reduce accident severity. Conventional approaches rely on transmitting raw video or high-dimensional sensory data from roadside units (RSUs) to vehicles, which is impractical under vehicular communication bandwidth and latency constraints. In this work, we propose a semantic V2X framework in which RSU-mounted cameras generate spatiotemporal semantic embeddings of future frames using the Video Joint Embedding Predictive Architecture (V-JEPA). To evaluate the system, we construct a digital twin of an urban traffic environment enabling the generation of d verse traffic scenarios with both safe and collision events. These embeddings of the future frame, extracted from V-JEPA, capture task-relevant traffic dynamics and are transmitted via V2X links to vehicles, where a lightweight attentive probe and classifier decode them to predict imminent collisions. By transmitting only semantic embeddings instead of raw frames, the proposed system significantly reduces communication overhead while maintaining predictive accuracy. Experimental results demonstrate that the framework with an appropriate processing method achieves a 10% F1-score improvement for collision prediction while reducing transmission requirements by four orders of magnitude compared to raw video. This validates the potential of semantic V2X communication to enable cooperative, real-time collision prediction in ITS.

**arXiv ID:** 2601.17216
</details>

<details>
<summary><strong>Mitigating the OWASP Top 10 For Large Language Models Applications using Intelligent Agents</strong> - Mohammad Fasha, Faisal Abul Rub, Nasim Matar, Bilal Sowan, Mohammad Al Khaldy - [[pdf]](https://arxiv.org/pdf/2601.18105)</summary>

**Abstract:** Large Language Models (LLMs) have emerged as a transformative and disruptive technology, enabling a wide range of applications in natural language processing, machine translation, and beyond. However, this widespread integration of LLMs also raised several security concerns highlighted by the Open Web Application Security Project (OWASP), which has identified the top 10 security vulnerabilities inherent in LLM applications. Addressing these vulnerabilities is crucial, given the increasing reliance on LLMs and the potential threats to data integrity, confidentiality, and service availability. This paper presents a framework designed to mitigate the security risks outlined in the OWASP Top 10. Our proposed model leverages LLM-enabled intelligent agents, offering a new approach to proactively identify, assess, and counteract security threats in real-time. The proposed framework serves as an initial blueprint for future research and development, aiming to enhance the security measures of LLMs and protect against emerging threats in this rapidly evolving landscape.

**arXiv ID:** 2601.18105
</details>

<details>
<summary><strong>An autonomous living database for perovskite photovoltaics</strong> - Sherjeel Shabih, Hampus Näsström, Sharat Patil, Asmin Askin, Keely Dodd-Clements, Jessica Helisa Hautrive Rossato, Hugo Gajardoni de Lemos, Yuxin Liu, Florian Mathies, Natalia Maticiuc, Rico Meitzner, Edgar Nandayapa, Juan José Patiño López, Yaru Wang, Lauri Himanen, Eva Unger, T. Jesper Jacobsson, José A. Márquez, Kevin Maik Jablonka - [[pdf]](https://arxiv.org/pdf/2601.17807)</summary>

**Abstract:** Scientific discovery is severely bottlenecked by the inability of manual curation to keep pace with exponential publication rates. This creates a widening knowledge gap. This is especially stark in photovoltaics, where the leading database for perovskite solar cells has been stagnant since 2021 despite massive ongoing research output. Here, we resolve this challenge by establishing an autonomous, self-updating living database (PERLA). Our pipeline integrates large language models with physics-aware validation to extract complex device data from the continuous literature stream, achieving human-level precision (>90%) and eliminating annotator variance. By employing this system on the previously inaccessible post-2021 literature, we uncover critical evolutionary trends hidden by data lag: the field has decisively shifted toward inverted architectures employing self-assembled monolayers and formamidinium-rich compositions, driving a clear trajectory of sustained voltage loss reduction. PERLA transforms static publications into dynamic knowledge resources that enable data-driven discovery to operate at the speed of publication.

**arXiv ID:** 2601.17807
</details>

<details>
<summary><strong>EMPM: Embodied MPM for Modeling and Simulation of Deformable Objects</strong> - Yunuo Chen, Yafei Hu, Lingfeng Sun, Tushar Kusnur, Laura Herlant, Chenfanfu Jiang - [[pdf]](https://arxiv.org/pdf/2601.17251)</summary>

**Abstract:** Modeling deformable objects - especially continuum materials - in a way that is physically plausible, generalizable, and data-efficient remains challenging across 3D vision, graphics, and robotic manipulation. Many existing methods oversimplify the rich dynamics of deformable objects or require large training sets, which often limits generalization. We introduce embodied MPM (EMPM), a deformable object modeling and simulation framework built on a differentiable Material Point Method (MPM) simulator that captures the dynamics of challenging materials. From multi-view RGB-D videos, our approach reconstructs geometry and appearance, then uses an MPM physics engine to simulate object behavior by minimizing the mismatch between predicted and observed visual data. We further optimize MPM parameters online using sensory feedback, enabling adaptive, robust, and physics-aware object representations that open new possibilities for robotic manipulation of complex deformables. Experiments show that EMPM outperforms spring-mass baseline models. Project website: this https URL.

**arXiv ID:** 2601.17251
</details>

<details>
<summary><strong>Are Conversational AI Agents the Way Out? Co-Designing Reader-Oriented News Experiences with Immigrants and Journalists</strong> - Yongle Zhang, Ge Gao - [[pdf]](https://arxiv.org/pdf/2601.18772)</summary>

**Abstract:** Recent discussions at the intersection of journalism, HCI, and human-centered computing ask how technologies can help create reader-oriented news experiences. The current paper takes up this initiative by focusing on immigrant readers, a group who reports significant difficulties engaging with mainstream news yet has received limited attention in prior research. We report findings from our co-design research with eleven immigrant readers living in the United States and seven journalists working in the same region, aiming to enhance the news experience of the former. Data collected from all participants revealed an "unaddressed-or-unaccountable" paradox that challenges value alignment across immigrant readers and journalists. This paradox points to four metaphors regarding how conversational AI agents can be designed to assist news reading. Each metaphor requires conversational AI, journalists, and immigrant readers to coordinate their shared responsibilities in a distinct manner. These findings provide insights into reader-oriented news experiences with AI in the loop.

**arXiv ID:** 2601.18772
</details>

<details>
<summary><strong>ReUseIt: Synthesizing Reusable AI Agent Workflows for Web Automation</strong> - Yimeng Liu, Misha Sra, Jeevana Priya Inala, Chenglong Wang - [[pdf]](https://arxiv.org/pdf/2510.14308)</summary>

**Abstract:** AI-powered web agents have the potential to automate repetitive tasks, such as form filling, information retrieval, and scheduling, but they struggle to reliably execute these tasks without human intervention, requiring users to provide detailed guidance during every run. We address this limitation by automatically synthesizing reusable workflows from an agent's successful and failed attempts. These workflows incorporate execution guards that help agents detect and fix errors while keeping users informed of progress and issues. Our approach enables agents to successfully complete repetitive tasks of the same type with minimal user intervention, increasing the success rates from 24.2% to 70.1% across fifteen tasks. To evaluate this approach, we invited nine users and found that our agent helped them complete web tasks with a higher success rate and less guidance compared to two baseline methods, as well as allowed users to easily monitor agent behavior and understand its failures.

**arXiv ID:** 2510.14308
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (39 papers)</h2></summary>

<details>
<summary><strong>Cognitive Platform Engineering for Autonomous Cloud Operations</strong> - Vinoth Punniyamoorthy, Nitin Saksena, Srivenkateswara Reddy Sankiti, Nachiappan Chockalingam, Aswathnarayan Muthukrishnan Kirubakaran, Shiva Kumar Reddy Carimireddy, Durgaraman Maruthavanan - [[pdf]](https://arxiv.org/pdf/2601.17542)</summary>

**Abstract:** Modern DevOps practices have accelerated software delivery through automation, CI/CD pipelines, and observability tooling,but these approaches struggle to keep pace with the scale and dynamism of cloud-native systems. As telemetry volume grows and configuration drift increases, traditional, rule-driven automation often results in reactive operations, delayed remediation, and dependency on manual expertise. This paper introduces Cognitive Platform Engineering, a next-generation paradigm that integrates sensing, reasoning, and autonomous action directly into the platform lifecycle. This paper propose a four-plane reference architecture that unifies data collection, intelligent inference, policy-driven orchestration, and human experience layers within a continuous feedback loop. A prototype implementation built with Kubernetes, Terraform, Open Policy Agent, and ML-based anomaly detection demonstrates improvements in mean time to resolution, resource efficiency, and compliance. The results show that embedding intelligence into platform operations enables resilient, self-adjusting, and intent-aligned cloud environments. The paper concludes with research opportunities in reinforcement learning, explainable governance, and sustainable self-managing cloud ecosystems.

**arXiv ID:** 2601.17542
</details>

<details>
<summary><strong>SQL-Trail: Multi-Turn Reinforcement Learning with Interleaved Feedback for Text-to-SQL</strong> - Harper Hua, Zhen Han, Zhengyuan Shen, Jeremy Lee, Patrick Guan, Qi Zhu, Sullam Jeoung, Yueyan Chen, Yunfei Bai, Shuai Wang, Vassilis Ioannidis, Huzefa Rangwala - [[pdf]](https://arxiv.org/pdf/2601.17699)</summary>

**Abstract:** While large language models (LLMs) have substantially improved Text-to-SQL generation, a pronounced gap remains between AI systems and human experts on challenging benchmarks such as BIRD-SQL. We argue this gap stems largely from the prevailing single-pass paradigm, which lacks the iterative reasoning, schema exploration, and error-correction behaviors that humans naturally employ. To address this limitation, we introduce SQL-Trail, a multi-turn reinforcement learning (RL) agentic framework for Text-to-SQL. Rather than producing a query in one shot, SQL-Trail interacts with the database environment and uses execution feedback to iteratively refine its predictions. Our approach centers on two key ideas: (i) an adaptive turn-budget allocation mechanism that scales the agent's interaction depth to match question difficulty, and (ii) a composite reward panel that jointly incentivizes SQL correctness and efficient exploration. Across benchmarks, SQL-Trail sets a new state of the art and delivers strong data efficiency--up to 18x higher than prior single-pass RL state-of-the-art methods. Notably, our 7B and 14B models outperform substantially larger proprietary systems by 5% on average, underscoring the effectiveness of interactive, agentic workflows for robust Text-to-SQL generation.

**arXiv ID:** 2601.17699
</details>

<details>
<summary><strong>Aligning Medical Conversational AI through Online Reinforcement Learning with Information-Theoretic Rewards</strong> - Tanvi Verma, Yang Zhou, Rick Siow Mong Goh, Yong Liu - [[pdf]](https://arxiv.org/pdf/2601.17828)</summary>

**Abstract:** We present Information Gain Fine-Tuning (IGFT), a novel approach for training medical conversational AI to conduct effective patient interviews and generate comprehensive History of Present Illness (HPI) without requiring pre-collected human conversations. IGFT combines online Group Relative Policy Optimization (GRPO) with information-theoretic rewards, enabling models to learn from self-generated conversations with simulated patients. Unlike existing approaches that rely on expensive expert-annotated conversations or static datasets, our online RL framework allows models to discover effective questioning strategies through exploration. Our key innovation is an information gain reward function that tracks which clinical entities such as symptoms, temporal patterns, and medical history, are revealed during conversation. Each question's reward is computed based on its expected information gain combined with GPT-4o-mini quality assessments across dimensions including clinical relevance, patient engagement, and specificity. This hybrid approach ensures models learn to ask targeted, clinically appropriate questions that efficiently gather diagnostic information. We fine-tune two models using LoRA: Llama-3.1-8B-Instruct and DeepSeek-R1-Distill-Qwen-7B (a reasoning-optimized model). Training exclusively on Avey data containing concise HPIs, we evaluate generalization to MIMIC data with longer, more elaborate HPIs. DeepSeek-R1-Distill-Qwen-7B (IGFT) achieves F1 scores of 0.408 on Avey (10.9% improvement over base) and 0.289 on MIMIC (12.9% improvement), while Llama-3.1-8B-Instruct (IGFT) reaches 0.384 and 0.336 respectively. Both models outperform OpenAI's model on MIMIC and surpass medical domain-specific baselines like HuatuoGPT and UltraMedical, which were optimized for single-turn medical QA rather than multi-turn conversations.

**arXiv ID:** 2601.17828
</details>

<details>
<summary><strong>DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints</strong> - Yinger Zhang, Shutong Jiang, Renhao Li, Jianhong Tu, Yang Su, Lianghao Deng, Xudong Guo, Chenxu Lv, Junyang Lin - [[pdf]](https://arxiv.org/pdf/2601.18137)</summary>

**Abstract:** While agent evaluation has shifted toward long-horizon tasks, most benchmarks still emphasize local, step-level reasoning rather than the global constrained optimization (e.g., time and financial budgets) that demands genuine planning ability. Meanwhile, existing LLM planning benchmarks underrepresent the active information gathering and fine-grained local constraints typical of real-world settings. To address this, we introduce DeepPlanning, a challenging benchmark for practical long-horizon agent planning. It features multi-day travel planning and multi-product shopping tasks that require proactive information acquisition, local constrained reasoning, and global constrained optimization. Evaluations on DeepPlanning show that even frontier agentic LLMs struggle with these problems, highlighting the importance of reliable explicit reasoning patterns and parallel tool use for achieving better effectiveness-efficiency trade-offs. Error analysis further points to promising directions for improving agentic LLMs over long planning horizons. We open-source the code and data to support future research.

**arXiv ID:** 2601.18137
</details>

<details>
<summary><strong>AI Agent for Reverse-Engineering Legacy Finite-Difference Code and Translating to Devito</strong> - Yinghan Hou, Zongyou Yang - [[pdf]](https://arxiv.org/pdf/2601.18381)</summary>

**Abstract:** To facilitate the transformation of legacy finite difference implementations into the Devito environment, this study develops an integrated AI agent framework. Retrieval-Augmented Generation (RAG) and open-source Large Language Models are combined through multi-stage iterative workflows in the system's hybrid LangGraph architecture. The agent constructs an extensive Devito knowledge graph through document parsing, structure-aware segmentation, extraction of entity relationships, and Leiden-based community detection. GraphRAG optimisation enhances query performance across semantic communities that include seismic wave simulation, computational fluid dynamics, and performance tuning libraries. A reverse engineering component derives three-level query strategies for RAG retrieval through static analysis of Fortran source code. To deliver precise contextual information for language model guidance, the multi-stage retrieval pipeline performs parallel searching, concept expansion, community-scale retrieval, and semantic similarity analysis. Code synthesis is governed by Pydantic-based constraints to guarantee structured outputs and reliability. A comprehensive validation framework integrates conventional static analysis with the G-Eval approach, covering execution correctness, structural soundness, mathematical consistency, and API compliance. The overall agent workflow is implemented on the LangGraph framework and adopts concurrent processing to support quality-based iterative refinement and state-aware dynamic routing. The principal contribution lies in the incorporation of feedback mechanisms motivated by reinforcement learning, enabling a transition from static code translation toward dynamic and adaptive analytical behavior.

**arXiv ID:** 2601.18381
</details>

<details>
<summary><strong>OffSeeker: Online Reinforcement Learning Is Not All You Need for Deep Research Agents</strong> - Yuhang Zhou, Kai Zheng, Qiguang Chen, Mengkang Hu, Qingfeng Sun, Can Xu, Jingjing Chen - [[pdf]](https://arxiv.org/pdf/2601.18467)</summary>

**Abstract:** Deep research agents have shown remarkable potential in handling long-horizon tasks. However, state-of-the-art performance typically relies on online reinforcement learning (RL), which is financially expensive due to extensive API calls. While offline training offers a more efficient alternative, its progress is hindered by the scarcity of high-quality research trajectories. In this paper, we demonstrate that expensive online reinforcement learning is not all you need to build powerful research agents. To bridge this gap, we introduce a fully open-source suite designed for effective offline training. Our core contributions include DeepForge, a ready-to-use task synthesis framework that generates large-scale research queries without heavy preprocessing; and a curated collection of 66k QA pairs, 33k SFT trajectories, and 21k DPO pairs. Leveraging these resources, we train OffSeeker (8B), a model developed entirely offline. Extensive evaluations across six benchmarks show that OffSeeker not only leads among similar-sized agents but also remains competitive with 30B-parameter systems trained via heavy online RL.

**arXiv ID:** 2601.18467
</details>

<details>
<summary><strong>DEEPMED: Building a Medical DeepResearch Agent via Multi-hop Med-Search Data and Turn-Controlled Agentic Training & Inference</strong> - Zihan wang, Hao Wang, Shi Feng, Xiaocui Yang, Daling Wang, Yiqun Zhang, Jinghao Lin, Haihua Yang, Xiaozhong Ji - [[pdf]](https://arxiv.org/pdf/2601.18496)</summary>

**Abstract:** Medical reasoning models remain constrained by parametric knowledge and are thus susceptible to forgetting and hallucinations. DeepResearch (DR) models ground outputs in verifiable evidence from tools and perform strongly in general domains, but their direct transfer to medical field yields relatively limited gains. We attribute this to two gaps: task characteristic and tool-use scaling. Medical questions require evidence interpretation in a knowledge-intensive clinical context; while general DR models can retrieve information, they often lack clinical-context reasoning and thus "find it but fail to use it," leaving performance limited by medical abilities. Moreover, in medical scenarios, blindly scaling tool-call can inject noisy context, derailing sensitive medical reasoning and prompting repetitive evidence-seeking along incorrect paths. Therefore, we propose DeepMed. For data, we deploy a multi-hop med-search QA synthesis method supporting the model to apply the DR paradigm in medical contexts. For training, we introduce a difficulty-aware turn-penalty to suppress excessive tool-call growth. For inference, we bring a monitor to help validate hypotheses within a controlled number of steps and avoid context rot. Overall, on seven medical benchmarks, DeepMed improves its base model by 9.79\% on average and outperforms larger medical reasoning and DR models.

**arXiv ID:** 2601.18496
</details>

<details>
<summary><strong>BibAgent: An Agentic Framework for Traceable Miscitation Detection in Scientific Literature</strong> - Peiran Li, Fangzhou Lin, Shuo Xing, Xiang Zheng, Xi Hong, Jiashuo Sun, Zhengzhong Tu, Chaoqun Ni - [[pdf]](https://arxiv.org/pdf/2601.16993)</summary>

**Abstract:** Citations are the bedrock of scientific authority, yet their integrity is compromised by widespread miscitations: ranging from nuanced distortions to fabricated references. Systematic citation verification is currently unfeasible; manual review cannot scale to modern publishing volumes, while existing automated tools are restricted by abstract-only analysis or small-scale, domain-specific datasets in part due to the "paywall barrier" of full-text access. We introduce BibAgent, a scalable, end-to-end agentic framework for automated citation verification. BibAgent integrates retrieval, reasoning, and adaptive evidence aggregation, applying distinct strategies for accessible and paywalled sources. For paywalled references, it leverages a novel Evidence Committee mechanism that infers citation validity via downstream citation consensus. To support systematic evaluation, we contribute a 5-category Miscitation Taxonomy and MisciteBench, a massive cross-disciplinary benchmark comprising 6,350 miscitation samples spanning 254 fields. Our results demonstrate that BibAgent outperforms state-of-the-art Large Language Model (LLM) baselines in citation verification accuracy and interpretability, providing scalable, transparent detection of citation misalignments across the scientific literature.

**arXiv ID:** 2601.16993
</details>

<details>
<summary><strong>Investigating Self-regulated Learning Sequences within a Generative AI-based Intelligent Tutoring System</strong> - Jie Gao, Shasha Li, Jianhua Zhang, Shan Li, Tingting Wang - [[pdf]](https://arxiv.org/pdf/2601.17000)</summary>

**Abstract:** There has been a growing trend in employing generative artificial intelligence (GenAI) techniques to support learning. Moreover, scholars have reached a consensus on the critical role of self-regulated learning (SRL) in ensuring learning effectiveness within GenAI-assisted learning environments, making it essential to capture students' dynamic SRL patterns. In this study, we extracted students' interaction patterns with GenAI from trace data as they completed a problem-solving task within a GenAI-assisted intelligent tutoring system. Students' purpose of using GenAI was also analyzed from the perspective of information processing, i.e., information acquisition and information transformation. Using sequential and clustering analysis, this study classified participants into two groups based on their SRL sequences. These two groups differed in the frequency and temporal characteristics of GenAI use. In addition, most students used GenAI for information acquisition rather than information transformation, while the correlation between the purpose of using GenAI and learning performance was not statistically significant. Our findings inform both pedagogical design and the development of GenAI-assisted learning environments.

**arXiv ID:** 2601.17000
</details>

<details>
<summary><strong>Retell, Reward, Repeat: Reinforcement Learning for Narrative Theory-Informed Story Generation</strong> - David Y. Liu, Xanthe Muston, Aditya Joshi, Sebastian Sequoiah-Grayson - [[pdf]](https://arxiv.org/pdf/2601.17226)</summary>

**Abstract:** Despite the subjective nature of storytelling, past works on automatic story generation (ASG) have relied on limited ground truths for training and evaluation. In this work, we explore reinforcement learning (d-RLAIF) as a post-training alternative to supervised fine-tuning (SFT). We first apply Todorov's Theory of Narrative Equilibrium to establish principles that define desirable ASG qualities. We prompt 7B and 14B LLM-as-judge models with our principles to test alignment with human annotators and provide reward signals during d-RLAIF. We use Gemini-3-Flash to evaluate the output of our post-trained models and compare them to human-written stories from the TimeTravel dataset. We show that d-RLAIF offers a viable alternative to supervised fine-tuning (SFT)--producing stories that are more diverse and aligned with human narrative conventions. Our paper demonstrates the promise of reinforcement learning for linguistically grounded post-training for subjective tasks such as ASG.

**arXiv ID:** 2601.17226
</details>

<details>
<summary><strong>Latent-Space Contrastive Reinforcement Learning for Stable and Efficient LLM Reasoning</strong> - Lianlei Shan, Han Chen, Yixuan Wang, Zhenjie Liu, Wei Li - [[pdf]](https://arxiv.org/pdf/2601.17275)</summary>

**Abstract:** While Large Language Models (LLMs) demonstrate exceptional performance in surface-level text generation, their nature in handling complex multi-step reasoning tasks often remains one of ``statistical fitting'' rather than systematic logical deduction. Traditional Reinforcement Learning (RL) attempts to mitigate this by introducing a ``think-before-speak'' paradigm. However, applying RL directly in high-dimensional, discrete token spaces faces three inherent challenges: sample-inefficient rollouts, high gradient estimation variance, and the risk of catastrophic forgetting. To fundamentally address these structural bottlenecks, we propose \textbf{DeepLatent Reasoning (DLR)}, a latent-space bidirectional contrastive reinforcement learning framework. This framework shifts the trial-and-error cost from expensive token-level full sequence generation to the continuous latent manifold. Specifically, we introduce a lightweight assistant model to efficiently sample $K$ reasoning chain encodings within the latent space. These encodings are filtered via a dual reward mechanism based on correctness and formatting; only high-value latent trajectories are fed into a \textbf{frozen main model} for single-pass decoding. To maximize reasoning diversity while maintaining coherence, we design a contrastive learning objective to enable directed exploration within the latent space. Since the main model parameters remain frozen during optimization, this method mathematically eliminates catastrophic forgetting. Experiments demonstrate that under comparable GPU computational budgets, DLR achieves more stable training convergence, supports longer-horizon reasoning chains, and facilitates the sustainable accumulation of reasoning capabilities, providing a viable path toward reliable and scalable reinforcement learning for LLMs.

**arXiv ID:** 2601.17275
</details>

<details>
<summary><strong>Agentic reinforcement learning empowers next-generation chemical language models for molecular design and synthesis</strong> - Hao Li, He Cao, Shenyao Peng, Zijing Liu, Bin Feng, Yu Wang, Zhiyuan Yan, Yonghong Tian, Yu Li, Li Yuan - [[pdf]](https://arxiv.org/pdf/2601.17687)</summary>

**Abstract:** Language models are revolutionizing the biochemistry domain, assisting scientists in drug design and chemical synthesis with high efficiency. Yet current approaches struggle between small language models prone to hallucination and limited knowledge retention, and large cloud-based language models plagued by privacy risks and high inference costs. To bridge this gap, we introduce ChemCRAFT, a novel framework leveraging agentic reinforcement learning to decouple chemical reasoning from knowledge storage. Instead of forcing the model to memorize vast chemical data, our approach empowers the language model to interact with a sandbox for precise information retrieval. This externalization of knowledge allows a locally deployable small model to achieve superior performance with minimal inference costs. To enable small language models for agent-calling ability, we build an agentic trajectory construction pipeline and a comprehensive chemical-agent sandbox. Based on sandbox interactions, we constructed ChemToolDataset, the first large-scale chemical tool trajectory dataset. Simultaneously, we propose SMILES-GRPO to build a dense chemical reward function, promoting the model's ability to call chemical agents. Evaluations across diverse aspects of drug design show that ChemCRAFT outperforms current cloud-based LLMs in molecular structure analysis, molecular optimization, and synthesis pathway prediction, demonstrating that scientific reasoning is not solely an emergent ability of model scale, but a learnable policy of tool orchestration. This work establishes a cost-effective and privacy-preserving paradigm for AI-aided chemistry, opening new avenues for accelerating molecular discovery with locally deployable agents.

**arXiv ID:** 2601.17687
</details>

<details>
<summary><strong>Diffusion Model-based Reinforcement Learning for Version Age of Information Scheduling: Average and Tail-Risk-Sensitive Control</strong> - Haoyuan Pan, Sizhao Chen, Zhaorui Wang, Tse-Tin Chan - [[pdf]](https://arxiv.org/pdf/2601.18069)</summary>

**Abstract:** Ensuring timely and semantically accurate information delivery is critical in real-time wireless systems. While Age of Information (AoI) quantifies temporal freshness, Version Age of Information (VAoI) captures semantic staleness by accounting for version evolution between transmitters and receivers. Existing VAoI scheduling approaches primarily focus on minimizing average VAoI, overlooking rare but severe staleness events that can compromise reliability under stochastic packet arrivals and unreliable channels. This paper investigates both average-oriented and tail-risk-sensitive VAoI scheduling in a multi-user status update system with long-term transmission cost constraints. We first formulate the average VAoI minimization problem as a constrained Markov decision process and introduce a deep diffusion-based Soft Actor-Critic (D2SAC) algorithm. By generating actions through a diffusion-based denoising process, D2SAC enhances policy expressiveness and establishes a strong baseline for mean performance. Building on this foundation, we put forth RS-D3SAC, a risk-sensitive deep distributional diffusion-based Soft Actor-Critic algorithm. RS-D3SAC integrates a diffusion-based actor with a quantile-based distributional critic, explicitly modeling the full VAoI return distribution. This enables principled tail-risk optimization via Conditional Value-at-Risk (CVaR) while satisfying long-term transmission cost constraints. Extensive simulations show that, while D2SAC reduces average VAoI, RS-D3SAC consistently achieves substantial reductions in CVaR without sacrificing mean performance. The dominant gain in tail-risk reduction stems from the distributional critic, with the diffusion-based actor providing complementary refinement to stabilize and enrich policy decisions, highlighting their effectiveness for robust and risk-aware VAoI scheduling in multi-user wireless systems.

**arXiv ID:** 2601.18069
</details>

<details>
<summary><strong>Learning to Ideate for Machine Learning Engineering Agents</strong> - Yunxiang Zhang, Kang Zhou, Zhichao Xu, Kiran Ramnath, Yun Zhou, Sangmin Woo, Haibo Ding, Lin Lee Cheong - [[pdf]](https://arxiv.org/pdf/2601.17596)</summary>

**Abstract:** Existing machine learning engineering (MLE) agents struggle to iteratively optimize their implemented algorithms for effectiveness. To address this, we introduce MLE-Ideator, a dual-agent framework that separates ideation from implementation. In our system, an implementation agent can request strategic help from a dedicated Ideator. We show this approach is effective in two ways. First, in a training-free setup, our framework significantly outperforms implementation-only agent baselines on MLE-Bench. Second, we demonstrate that the Ideator can be trained with reinforcement learning (RL) to generate more effective ideas. With only 1K training samples from 10 MLE tasks, our RL-trained Qwen3-8B Ideator achieves an 11.5% relative improvement compared to its untrained counterpart and surpasses Claude Sonnet 3.5. These results highlights a promising path toward training strategic AI systems for scientific discovery.

**arXiv ID:** 2601.17596
</details>

<details>
<summary><strong>ProGraph-R1: Progress-aware Reinforcement Learning for Graph Retrieval Augmented Generation</strong> - Jinyoung Park, Sanghyeok Lee, Omar Zia Khan, Hyunwoo J. Kim, Joo-Kyung Kim - [[pdf]](https://arxiv.org/pdf/2601.17755)</summary>

**Abstract:** Graph Retrieval-Augmented Generation (GraphRAG) has been successfully applied in various knowledge-intensive question answering tasks by organizing external knowledge into structured graphs of entities and relations. It enables large language models (LLMs) to perform complex reasoning beyond text-chunk retrieval. Recent works have employed reinforcement learning (RL) to train agentic GraphRAG frameworks that perform iterative interactions between LLMs and knowledge graphs. However, existing RL-based frameworks such as Graph-R1 suffer from two key limitations: (1) they primarily depend on semantic similarity for retrieval, often overlooking the underlying graph structure, and (2) they rely on sparse, outcome-level rewards, failing to capture the quality of intermediate retrieval steps and their dependencies. To address these limitations, we propose ProGraph-R1, a progress-aware agentic framework for graph-based retrieval and multi-step reasoning. ProGraph-R1 introduces a structure-aware hypergraph retrieval mechanism that jointly considers semantic relevance and graph connectivity, encouraging coherent traversal along multi-hop reasoning paths. We also design a progress-based step-wise policy optimization, which provides dense learning signals by modulating advantages according to intermediate reasoning progress within a graph, rather than relying solely on final outcomes. Experiments on multi-hop question answering benchmarks demonstrate that ProGraph-R1 consistently improves reasoning accuracy and generation quality over existing GraphRAG methods.

**arXiv ID:** 2601.17755
</details>

<details>
<summary><strong>MemWeaver: Weaving Hybrid Memories for Traceable Long-Horizon Agentic Reasoning</strong> - Juexiang Ye, Xue Li, Xinyu Yang, Chengkai Huang, Lanshun Nie, Lina Yao, Dechen Zhan - [[pdf]](https://arxiv.org/pdf/2601.18204)</summary>

**Abstract:** Large language model-based agents operating in long-horizon interactions require memory systems that support temporal consistency, multi-hop reasoning, and evidence-grounded reuse across sessions. Existing approaches largely rely on unstructured retrieval or coarse abstractions, which often lead to temporal conflicts, brittle reasoning, and limited traceability. We propose MemWeaver, a unified memory framework that consolidates long-term agent experiences into three interconnected components: a temporally grounded graph memory for structured relational reasoning, an experience memory that abstracts recurring interaction patterns from repeated observations, and a passage memory that preserves original textual evidence. MemWeaver employs a dual-channel retrieval strategy that jointly retrieves structured knowledge and supporting evidence to construct compact yet information-dense contexts for reasoning. Experiments on the LoCoMo benchmark demonstrate that MemWeaver substantially improves multi-hop and temporal reasoning accuracy while reducing input context length by over 95\% compared to long-context baselines.

**arXiv ID:** 2601.18204
</details>

<details>
<summary><strong>From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for Reinforcement Learning of Open-ended Generation</strong> - Yuxin Jiang, Yufei Wang, Qiyuan Zhang, Xingshan Zeng, Liangyou Li, Jierun Chen, Chaofan Tao, Haoli Bai, Lifeng Shang - [[pdf]](https://arxiv.org/pdf/2601.18533)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) succeeds in reasoning tasks (e.g., math and code) by checking the final verifiable answer (i.e., a verifiable dot signal). However, extending this paradigm to open-ended generation is challenging because there is no unambiguous ground truth. Relying on single-dot supervision often leads to inefficiency and reward hacking. To address these issues, we propose reinforcement learning with verifiable reference-based rewards (RLVRR). Instead of checking the final answer, RLVRR extracts an ordered linguistic signal from high-quality references (i.e, reward chain). Specifically, RLVRR decomposes rewards into two dimensions: content, which preserves deterministic core concepts (e.g., keywords), and style, which evaluates adherence to stylistic properties through LLM-based verification. In this way, RLVRR combines the exploratory strength of RL with the efficiency and reliability of supervised fine-tuning (SFT). Extensive experiments on more than 10 benchmarks with Qwen and Llama models confirm the advantages of our approach. RLVRR (1) substantially outperforms SFT trained with ten times more data and advanced reward models, (2) unifies the training of structured reasoning and open-ended generation, and (3) generalizes more effectively while preserving output diversity. These results establish RLVRR as a principled and efficient path toward verifiable reinforcement learning for general-purpose LLM alignment. We release our code and data at this https URL.

**arXiv ID:** 2601.18533
</details>

<details>
<summary><strong>Scaling medical imaging report generation with multimodal reinforcement learning</strong> - Qianchu Liu, Sheng Zhang, Guanghui Qin, Yu Gu, Ying Jin, Sam Preston, Yanbo Xu, Sid Kiblawi, Wen-wai Yim, Tim Ossowski, Tristan Naumann, Mu Wei, Hoifung Poon - [[pdf]](https://arxiv.org/pdf/2601.17151)</summary>

**Abstract:** Frontier models have demonstrated remarkable capabilities in understanding and reasoning with natural-language text, but they still exhibit major competency gaps in multimodal understanding and reasoning especially in high-value verticals such as biomedicine. Medical imaging report generation is a prominent example. Supervised fine-tuning can substantially improve performance, but they are prone to overfitting to superficial boilerplate patterns. In this paper, we introduce Universal Report Generation (UniRG) as a general framework for medical imaging report generation. By leveraging reinforcement learning as a unifying mechanism to directly optimize for evaluation metrics designed for end applications, UniRG can significantly improve upon supervised fine-tuning and attain durable generalization across diverse institutions and clinical practices. We trained UniRG-CXR on publicly available chest X-ray (CXR) data and conducted a thorough evaluation in CXR report generation with rigorous evaluation scenarios. On the authoritative ReXrank benchmark, UniRG-CXR sets new overall SOTA, outperforming prior state of the art by a wide margin.

**arXiv ID:** 2601.17151
</details>

<details>
<summary><strong>Agentic Search in the Wild: Intents and Trajectory Dynamics from 14M+ Real Search Requests</strong> - Jingjie Ning, João Coelho, Yibo Kong, Yunfan Long, Bruno Martins, João Magalhães, Jamie Callan, Chenyan Xiong - [[pdf]](https://arxiv.org/pdf/2601.17617)</summary>

**Abstract:** LLM-powered search agents are increasingly being used for multi-step information seeking tasks, yet the IR community lacks empirical understanding of how agentic search sessions unfold and how retrieved evidence is used. This paper presents a large-scale log analysis of agentic search based on 14.44M search requests (3.97M sessions) collected from DeepResearchGym, i.e. an open-source search API accessed by external agentic clients. We sessionize the logs, assign session-level intents and step-wise query-reformulation labels using LLM-based annotation, and propose Context-driven Term Adoption Rate (CTAR) to quantify whether newly introduced query terms are traceable to previously retrieved evidence. Our analyses reveal distinctive behavioral patterns. First, over 90% of multi-turn sessions contain at most ten steps, and 89% of inter-step intervals fall under one minute. Second, behavior varies by intent. Fact-seeking sessions exhibit high repetition that increases over time, while sessions requiring reasoning sustain broader exploration. Third, agents reuse evidence across steps. On average, 54% of newly introduced query terms appear in the accumulated evidence context, with contributions from earlier steps beyond the most recent retrieval. The findings suggest that agentic search may benefit from repetition-aware early stopping, intent-adaptive retrieval budgets, and explicit cross-step context tracking. We plan to release the anonymized logs to support future research.

**arXiv ID:** 2601.17617
</details>

<details>
<summary><strong>FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning</strong> - Zhaopeng Qiu, Shuang Yu, Jingqi Zhang, Shuai Zhang, Xue Huang, Jingyi Yang, Junjie Lai - [[pdf]](https://arxiv.org/pdf/2601.18150)</summary>

**Abstract:** Reinforcement learning (RL) for large language models (LLMs) is increasingly bottlenecked by rollout (generation), where long output sequence lengths make attention and KV-cache memory dominate end-to-end step time. FP8 offers an attractive lever for accelerating RL by reducing compute cost and memory traffic during rollout, but applying FP8 in RL introduces unique engineering and algorithmic challenges: policy weights change every step (requiring repeated quantization and weight synchronization into the inference engine) and low-precision rollouts can deviate from the higher-precision policy assumed by the trainer, causing train-inference mismatch and potential instability. This report presents a practical FP8 rollout stack for LLM RL, implemented in the veRL ecosystem with support for common training backends (e.g., FSDP/Megatron-LM) and inference engines (e.g., vLLM/SGLang). We (i) enable FP8 W8A8 linear-layer rollout using blockwise FP8 quantization, (ii) extend FP8 to KV-cache to remove long-context memory bottlenecks via per-step QKV scale recalibration, and (iii) mitigate mismatch using importance-sampling-based rollout correction (token-level TIS/MIS variants). Across dense and MoE models, these techniques deliver up to 44% rollout throughput gains while preserving learning behavior comparable to BF16 baselines.

**arXiv ID:** 2601.18150
</details>

<details>
<summary><strong>Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning</strong> - Sanghwan Bae, Jiwoo Hong, Min Young Lee, Hanbyul Kim, JeongYeon Nam, Donghyun Kwak - [[pdf]](https://arxiv.org/pdf/2504.03380)</summary>

**Abstract:** Recent advances in reinforcement learning with verifiable rewards (RLVR) show that large language models enhance their reasoning abilities when trained with verifiable signals. However, due to reward sparsity, effectiveness depends heavily on selecting samples of appropriate difficulty. In this work, we present a formal analysis of online difficulty-aware filtering and establish its theoretical foundations. We show that expected policy improvement is lower-bounded by the variance of task-level success probabilities, implying that selecting tasks of intermediate difficulty maximizes learning efficiency. Building on this, we demonstrate that balanced filtering maximizes this lower bound, leading to superior performance and sample efficiency. Evaluations across multiple math reasoning benchmarks validate that balanced filtering consistently enhances convergence speed and final performance, achieving up to +12% gains in less than half the training steps of standard GRPO. By extending our analysis to various reward distributions, we provide a principled foundation for future RLVR curriculum strategies, confirmed through both theoretical analysis and extensive empirical results.

**arXiv ID:** 2504.03380
</details>

<details>
<summary><strong>SeRL: Self-Play Reinforcement Learning for Large Language Models with Limited Data</strong> - Wenkai Fang, Shunyu Liu, Yang Zhou, Kongcheng Zhang, Tongya Zheng, Kaixuan Chen, Mingli Song, Dacheng Tao - [[pdf]](https://arxiv.org/pdf/2505.20347)</summary>

**Abstract:** Recent advances have demonstrated the effectiveness of Reinforcement Learning (RL) in improving the reasoning capabilities of Large Language Models (LLMs). However, existing works inevitably rely on high-quality instructions and verifiable rewards for effective training, both of which are often difficult to obtain in specialized domains. In this paper, we propose Self-play Reinforcement Learning (SeRL) to bootstrap LLM training with limited initial data. Specifically, SeRL comprises two complementary modules: self-instruction and self-rewarding. The former module generates additional instructions based on the available data at each training step, employing robust online filtering strategies to ensure instruction quality, diversity, and difficulty. The latter module introduces a simple yet effective majority-voting mechanism to estimate response rewards for additional instructions, eliminating the need for external annotations. Finally, SeRL performs conventional RL based on the generated data, facilitating iterative self-play learning. Extensive experiments on various reasoning benchmarks and across different LLM backbones demonstrate that the proposed SeRL yields results superior to its counterparts and achieves performance on par with those obtained by high-quality data with verifiable rewards. Our code is available at this https URL.

**arXiv ID:** 2505.20347
</details>

<details>
<summary><strong>Adaptive Constraint Propagation: Scaling Structured Inference for Large Language Models via Meta-Reinforcement Learning</strong> - Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma - [[pdf]](https://arxiv.org/pdf/2601.00095)</summary>

**Abstract:** Large language models increasingly require structured inference, from JSON schema enforcement to multi-lingual parsing, where outputs must satisfy complex constraints. We introduce MetaJuLS, a meta-reinforcement learning approach that learns universal constraint propagation policies applicable across languages and tasks without task-specific retraining. By formulating structured inference as adaptive constraint propagation and training a Graph Attention Network with meta-learning, MetaJuLS achieves 1.5--2.0$\times$ speedups over GPU-optimized baselines while maintaining within 0.2\% accuracy of state-of-the-art parsers. On Universal Dependencies across 10 languages and LLM-constrained generation (LogicBench, GSM8K-Constrained), MetaJuLS demonstrates rapid cross-domain adaptation: a policy trained on English parsing adapts to new languages and tasks with 5--10 gradient steps (5--15 seconds) rather than requiring hours of task-specific training. Mechanistic analysis reveals the policy discovers human-like parsing strategies (easy-first) and novel non-intuitive heuristics. By reducing propagation steps in LLM deployments, MetaJuLS contributes to Green AI by directly reducing inference carbon footprint.

**arXiv ID:** 2601.00095
</details>

<details>
<summary><strong>Inside Out: Evolving User-Centric Core Memory Trees for Long-Term Personalized Dialogue Systems</strong> - Jihao Zhao, Ding Chen, Zhaoxin Fan, Kerun Xu, Mengting Hu, Bo Tang, Feiyu Xiong, Zhiyu Li - [[pdf]](https://arxiv.org/pdf/2601.05171)</summary>

**Abstract:** Existing long-term personalized dialogue systems struggle to reconcile unbounded interaction streams with finite context constraints, often succumbing to memory noise accumulation, reasoning degradation, and persona inconsistency. To address these challenges, this paper proposes Inside Out, a framework that utilizes a globally maintained PersonaTree as the carrier of long-term user profiling. By constraining the trunk with an initial schema and updating the branches and leaves, PersonaTree enables controllable growth, achieving memory compression while preserving consistency. Moreover, we train a lightweight MemListener via reinforcement learning with process-based rewards to produce structured, executable, and interpretable {ADD, UPDATE, DELETE, NO_OP} operations, thereby supporting the dynamic evolution of the personalized tree. During response generation, PersonaTree is directly leveraged to enhance outputs in latency-sensitive scenarios; when users require more details, the agentic mode is triggered to introduce details on-demand under the constraints of the PersonaTree. Experiments show that PersonaTree outperforms full-text concatenation and various personalized memory systems in suppressing contextual noise and maintaining persona consistency. Notably, the small MemListener model achieves memory-operation decision performance comparable to, or even surpassing, powerful reasoning models such as DeepSeek-R1-0528 and Gemini-3-Pro.

**arXiv ID:** 2601.05171
</details>

<details>
<summary><strong>Quantum-Inspired Episode Selection for Monte Carlo Reinforcement Learning via QUBO Optimization</strong> - Hadi Salloum, Ali Jnadi, Yaroslav Kholodov, Alexander Gasnikov - [[pdf]](https://arxiv.org/pdf/2601.17570)</summary>

**Abstract:** Monte Carlo (MC) reinforcement learning suffers from high sample complexity, especially in environments with sparse rewards, large state spaces, and correlated trajectories. We address these limitations by reformulating episode selection as a Quadratic Unconstrained Binary Optimization (QUBO) problem and solving it with quantum-inspired samplers. Our method, MC+QUBO, integrates a combinatorial filtering step into standard MC policy evaluation: from each batch of trajectories, we select a subset that maximizes cumulative reward while promoting state-space coverage. This selection is encoded as a QUBO, where linear terms favor high-reward episodes and quadratic terms penalize redundancy. We explore both Simulated Quantum Annealing (SQA) and Simulated Bifurcation (SB) as black-box solvers within this framework. Experiments in a finite-horizon GridWorld demonstrate that MC+QUBO outperforms vanilla MC in convergence speed and final policy quality, highlighting the potential of quantum-inspired optimization as a decision-making subroutine in reinforcement learning.

**arXiv ID:** 2601.17570
</details>

<details>
<summary><strong>DRPG (Decompose, Retrieve, Plan, Generate): An Agentic Framework for Academic Rebuttal</strong> - Peixuan Han, Yingjie Yu, Jingjun Xu, Jiaxuan You - [[pdf]](https://arxiv.org/pdf/2601.18081)</summary>

**Abstract:** Despite the growing adoption of large language models (LLMs) in scientific research workflows, automated support for academic rebuttal, a crucial step in academic communication and peer review, remains largely underexplored. Existing approaches typically rely on off-the-shelf LLMs or simple pipelines, which struggle with long-context understanding and often fail to produce targeted and persuasive responses. In this paper, we propose DRPG, an agentic framework for automatic academic rebuttal generation that operates through four steps: Decompose reviews into atomic concerns, Retrieve relevant evidence from the paper, Plan rebuttal strategies, and Generate responses accordingly. Notably, the Planner in DRPG reaches over 98% accuracy in identifying the most feasible rebuttal direction. Experiments on data from top-tier conferences demonstrate that DRPG significantly outperforms existing rebuttal pipelines and achieves performance beyond the average human level using only an 8B model. Our analysis further demonstrates the effectiveness of the planner design and its value in providing multi-perspective and explainable suggestions. We also showed that DRPG works well in a more complex multi-round setting. These results highlight the effectiveness of DRPG and its potential to provide high-quality rebuttal content and support the scaling of academic discussions. Codes for this work are available at this https URL.

**arXiv ID:** 2601.18081
</details>

<details>
<summary><strong>Enhance the Safety in Reinforcement Learning by ADRC Lagrangian Methods</strong> - Mingxu Zhang, Huicheng Zhang, Jiaming Ji, Yaodong Yang, Ying Sun - [[pdf]](https://arxiv.org/pdf/2601.18142)</summary>

**Abstract:** Safe reinforcement learning (Safe RL) seeks to maximize rewards while satisfying safety constraints, typically addressed through Lagrangian-based methods. However, existing approaches, including PID and classical Lagrangian methods, suffer from oscillations and frequent safety violations due to parameter sensitivity and inherent phase lag. To address these limitations, we propose ADRC-Lagrangian methods that leverage Active Disturbance Rejection Control (ADRC) for enhanced robustness and reduced oscillations. Our unified framework encompasses classical and PID Lagrangian methods as special cases while significantly improving safety performance. Extensive experiments demonstrate that our approach reduces safety violations by up to 74%, constraint violation magnitudes by 89%, and average costs by 67\%, establishing superior effectiveness for Safe RL in complex environments.

**arXiv ID:** 2601.18142
</details>

<details>
<summary><strong>TriPlay-RL: Tri-Role Self-Play Reinforcement Learning for LLM Safety Alignment</strong> - Zhewen Tan, Wenhan Yu, Jianfeng Si, Tongxin Liu, Kaiqi Guan, Huiyan Jin, Jiawen Tao, Xiaokun Yuan, Duohe Ma, Xiangzheng Zhang, Tong Yang, Lin Sun - [[pdf]](https://arxiv.org/pdf/2601.18292)</summary>

**Abstract:** In recent years, safety risks associated with large language models have become increasingly prominent, highlighting the urgent need to mitigate the generation of toxic and harmful content. The mainstream paradigm for LLM safety alignment typically adopts a collaborative framework involving three roles: an attacker for adversarial prompt generation, a defender for safety defense, and an evaluator for response assessment. In this paper, we propose a closed-loop reinforcement learning framework called TriPlay-RL that enables iterative and co-improving collaboration among three roles with near-zero manual annotation. Experimental results show that the attacker preserves high output diversity while achieving a 20%-50% improvement in adversarial effectiveness; the defender attains 10%-30% gains in safety performance without degrading general reasoning capability; and the evaluator continuously refines its fine-grained judgment ability through iterations, accurately distinguishing unsafe responses, simple refusals, and useful guidance. Overall, our framework establishes an efficient and scalable paradigm for LLM safety alignment, enabling continuous co-evolution within a unified learning loop.

**arXiv ID:** 2601.18292
</details>

<details>
<summary><strong>K-Myriad: Jump-starting reinforcement learning with unsupervised parallel agents</strong> - Vincenzo De Paola, Mirco Mutti, Riccardo Zamboni, Marcello Restelli - [[pdf]](https://arxiv.org/pdf/2601.18580)</summary>

**Abstract:** Parallelization in Reinforcement Learning is typically employed to speed up the training of a single policy, where multiple workers collect experience from an identical sampling distribution. This common design limits the potential of parallelization by neglecting the advantages of diverse exploration strategies. We propose K-Myriad, a scalable and unsupervised method that maximizes the collective state entropy induced by a population of parallel policies. By cultivating a portfolio of specialized exploration strategies, K-Myriad provides a robust initialization for Reinforcement Learning, leading to both higher training efficiency and the discovery of heterogeneous solutions. Experiments on high-dimensional continuous control tasks, with large-scale parallelization, demonstrate that K-Myriad can learn a broad set of distinct policies, highlighting its effectiveness for collective exploration and paving the way towards novel parallelization strategies.

**arXiv ID:** 2601.18580
</details>

<details>
<summary><strong>Learning long term climate-resilient transport adaptation pathways under direct and indirect flood impacts using reinforcement learning</strong> - Miguel Costa, Arthur Vandervoort, Carolin Schmidt, Morten W. Petersen, Martin Drews, Karyn Morrissey, Francisco C. Pereira - [[pdf]](https://arxiv.org/pdf/2601.18586)</summary>

**Abstract:** Climate change is expected to intensify rainfall and other hazards, increasing disruptions in urban transportation systems. Designing effective adaptation strategies is challenging due to the long-term, sequential nature of infrastructure investments, deep uncertainty, and complex cross-sector interactions. We propose a generic decision-support framework that couples an integrated assessment model (IAM) with reinforcement learning (RL) to learn adaptive, multi-decade investment pathways under uncertainty. The framework combines long-term climate projections (e.g., IPCC scenario pathways) with models that map projected extreme-weather drivers (e.g. rain) into hazard likelihoods (e.g. flooding), propagate hazards into urban infrastructure impacts (e.g. transport disruption), and value direct and indirect consequences for service performance and societal costs. Embedded in a reinforcement-learning loop, it learns adaptive climate adaptation policies that trade off investment and maintenance expenditures against avoided impacts. In collaboration with Copenhagen Municipality, we demonstrate the approach on pluvial flooding in the inner city for the horizon of 2024 to 2100. The learned strategies yield coordinated spatial-temporal pathways and improved robustness relative to conventional optimization baselines, namely inaction and random action, illustrating the framework's transferability to other hazards and cities.

**arXiv ID:** 2601.18586
</details>

<details>
<summary><strong>Rank-1 Approximation of Inverse Fisher for Natural Policy Gradients in Deep Reinforcement Learning</strong> - Yingxiao Huo, Satya Prakash Dash, Radu Stoican, Samuel Kaski, Mingfei Sun - [[pdf]](https://arxiv.org/pdf/2601.18626)</summary>

**Abstract:** Natural gradients have long been studied in deep reinforcement learning due to their fast convergence properties and covariant weight updates. However, computing natural gradients requires inversion of the Fisher Information Matrix (FIM) at each iteration, which is computationally prohibitive in nature. In this paper, we present an efficient and scalable natural policy optimization technique that leverages a rank-1 approximation to full inverse-FIM. We theoretically show that under certain conditions, a rank-1 approximation to inverse-FIM converges faster than policy gradients and, under some conditions, enjoys the same sample complexity as stochastic policy gradient methods. We benchmark our method on a diverse set of environments and show that it achieves superior performance to standard actor-critic and trust-region baselines.

**arXiv ID:** 2601.18626
</details>

<details>
<summary><strong>ART for Diffusion Sampling: A Reinforcement Learning Approach to Timestep Schedule</strong> - Yilie Huang, Wenpin Tang, Xunyu Zhou - [[pdf]](https://arxiv.org/pdf/2601.18681)</summary>

**Abstract:** We consider time discretization for score-based diffusion models to generate samples from a learned reverse-time dynamic on a finite grid. Uniform and hand-crafted grids can be suboptimal given a budget on the number of time steps. We introduce Adaptive Reparameterized Time (ART) that controls the clock speed of a reparameterized time variable, leading to a time change and uneven timesteps along the sampling trajectory while preserving the terminal time. The objective is to minimize the aggregate error arising from the discretized Euler scheme. We derive a randomized control companion, ART-RL, and formulate time change as a continuous-time reinforcement learning (RL) problem with Gaussian policies. We then prove that solving ART-RL recovers the optimal ART schedule, which in turn enables practical actor--critic updates to learn the latter in a data-driven way. Empirically, based on the official EDM pipeline, ART-RL improves Fréchet Inception Distance on CIFAR-10 over a wide range of budgets and transfers to AFHQv2, FFHQ, and ImageNet without the need of retraining.

**arXiv ID:** 2601.18681
</details>

<details>
<summary><strong>Trust, Don't Trust, or Flip: Robust Preference-Based Reinforcement Learning with Multi-Expert Feedback</strong> - Seyed Amir Hosseini, Maryam Abdolali, Amirhosein Tavakkoli, Fardin Ayar, Ehsan Javanmardi, Manabu Tsukada, Mahdi Javanmardi - [[pdf]](https://arxiv.org/pdf/2601.18751)</summary>

**Abstract:** Preference-based reinforcement learning (PBRL) offers a promising alternative to explicit reward engineering by learning from pairwise trajectory comparisons. However, real-world preference data often comes from heterogeneous annotators with varying reliability; some accurate, some noisy, and some systematically adversarial. Existing PBRL methods either treat all feedback equally or attempt to filter out unreliable sources, but both approaches fail when faced with adversarial annotators who systematically provide incorrect preferences. We introduce TriTrust-PBRL (TTP), a unified framework that jointly learns a shared reward model and expert-specific trust parameters from multi-expert preference feedback. The key insight is that trust parameters naturally evolve during gradient-based optimization to be positive (trust), near zero (ignore), or negative (flip), enabling the model to automatically invert adversarial preferences and recover useful signal rather than merely discarding corrupted feedback. We provide theoretical analysis establishing identifiability guarantees and detailed gradient analysis that explains how expert separation emerges naturally during training without explicit supervision. Empirically, we evaluate TTP on four diverse domains spanning manipulation tasks (MetaWorld) and locomotion (DM Control) under various corruption scenarios. TTP achieves state-of-the-art robustness, maintaining near-oracle performance under adversarial corruption while standard PBRL methods fail catastrophically. Notably, TTP outperforms existing baselines by successfully learning from mixed expert pools containing both reliable and adversarial annotators, all while requiring no expert features beyond identification indices and integrating seamlessly with existing PBRL pipelines.

**arXiv ID:** 2601.18751
</details>

<details>
<summary><strong>Multi-Objective Reinforcement Learning for Efficient Tactical Decision Making for Trucks in Highway Traffic</strong> - Deepthi Pathare, Leo Laine, Morteza Haghir Chehreghani - [[pdf]](https://arxiv.org/pdf/2601.18783)</summary>

**Abstract:** Balancing safety, efficiency, and operational costs in highway driving poses a challenging decision-making problem for heavy-duty vehicles. A central difficulty is that conventional scalar reward formulations, obtained by aggregating these competing objectives, often obscure the structure of their trade-offs. We present a Proximal Policy Optimization based multi-objective reinforcement learning framework that learns a continuous set of policies explicitly representing these trade-offs and evaluates it on a scalable simulation platform for tactical decision making in trucks. The proposed approach learns a continuous set of Pareto-optimal policies that capture the trade-offs among three conflicting objectives: safety, quantified in terms of collisions and successful completion; energy efficiency and time efficiency, quantified using energy cost and driver cost, respectively. The resulting Pareto frontier is smooth and interpretable, enabling flexibility in choosing driving behavior along different conflicting objectives. This framework allows seamless transitions between different driving policies without retraining, yielding a robust and adaptive decision-making strategy for autonomous trucking applications.

**arXiv ID:** 2601.18783
</details>

<details>
<summary><strong>Athena: Synergizing Data Prefetching and Off-Chip Prediction via Online Reinforcement Learning</strong> - Rahul Bera, Zhenrong Lang, Caroline Hengartner, Konstantinos Kanellopoulos, Rakesh Kumar, Mohammad Sadrosadati, Onur Mutlu - [[pdf]](https://arxiv.org/pdf/2601.17615)</summary>

**Abstract:** Prefetching and off-chip prediction are two techniques proposed to hide long memory access latencies in high-performance processors. In this work, we demonstrate that: (1) prefetching and off-chip prediction often provide complementary performance benefits, yet (2) naively combining them often fails to realize their full performance potential, and (3) existing prefetcher control policies leave significant room for performance improvement behind.
Our goal is to design a holistic framework that can autonomously learn to coordinate an off-chip predictor with multiple prefetchers employed at various cache levels. To this end, we propose a new technique called Athena, which models the coordination between prefetchers and off-chip predictor (OCP) as a reinforcement learning (RL) problem. Athena acts as the RL agent that observes multiple system-level features (e.g., prefetcher/OCP accuracy, bandwidth usage) over an epoch of program execution, and uses them as state information to select a coordination action (i.e., enabling the prefetcher and/or OCP, and adjusting prefetcher aggressiveness). At the end of every epoch, Athena receives a numerical reward that measures the change in multiple system-level metrics (e.g., number of cycles taken to execute an epoch). Athena uses this reward to autonomously and continuously learn a policy to coordinate prefetchers with OCP.
Our extensive evaluation using a diverse set of memory-intensive workloads shows that Athena consistently outperforms prior state-of-the-art coordination policies across a wide range of system configurations with various combinations of underlying prefetchers, OCPs, and main memory bandwidths, while incurring only modest storage overhead. Athena is freely available at this https URL.

**arXiv ID:** 2601.17615
</details>

<details>
<summary><strong>Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning</strong> - Ziming Li, Chenhao Li, Marco Hutter - [[pdf]](https://arxiv.org/pdf/2601.17428)</summary>

**Abstract:** Curriculum learning has demonstrated substantial effectiveness in robot learning. However, it still faces limitations when scaling to complex, wide-ranging task spaces. Such task spaces often lack a well-defined difficulty structure, making the difficulty ordering required by previous methods challenging to define. We propose a Learning Progress-based Automatic Curriculum Reinforcement Learning (LP-ACRL) framework, which estimates the agent's learning progress online and adaptively adjusts the task-sampling distribution, thereby enabling automatic curriculum generation without prior knowledge of the difficulty distribution over the task space. Policies trained with LP-ACRL enable the ANYmal D quadruped to achieve and maintain stable, high-speed locomotion at 2.5 m/s linear velocity and 3.0 rad/s angular velocity across diverse terrains, including stairs, slopes, gravel, and low-friction flat surfaces--whereas previous methods have generally been limited to high speeds on flat terrain or low speeds on complex terrain. Experimental results demonstrate that LP-ACRL exhibits strong scalability and real-world applicability, providing a robust baseline for future research on curriculum generation in complex, wide-ranging robotic learning task spaces.

**arXiv ID:** 2601.17428
</details>

<details>
<summary><strong>Actor-Critic Cooperative Compensation to Model Predictive Control for Off-Road Autonomous Vehicles Under Unknown Dynamics</strong> - Prakhar Gupta, Jonathon M Smereka, Yunyi Jia - [[pdf]](https://arxiv.org/pdf/2503.00577)</summary>

**Abstract:** This study presents an Actor-Critic Cooperative Compensated Model Predictive Controller (AC3MPC) designed to address unknown system dynamics. To avoid the difficulty of modeling highly complex dynamics and ensuring realtime control feasibility and performance, this work uses deep reinforcement learning with a model predictive controller in a cooperative framework to handle unknown dynamics. The model-based controller takes on the primary role as both controllers are provided with predictive information about the other. This improves tracking performance and retention of inherent robustness of the model predictive controller. We evaluate this framework for off-road autonomous driving on unknown deformable terrains that represent sandy deformable soil, sandy and rocky soil, and cohesive clay-like deformable soil. Our findings demonstrate that our controller statistically outperforms standalone model-based and learning-based controllers by upto 29.2% and 10.2%. This framework generalized well over varied and previously unseen terrain characteristics to track longitudinal reference speeds with lower errors. Furthermore, this required significantly less training data compared to purely learning-based controller, while delivering better performance even when under-trained.

**arXiv ID:** 2503.00577
</details>

<details>
<summary><strong>Offline Reinforcement Learning using Human-Aligned Reward Labeling for Autonomous Emergency Braking in Occluded Pedestrian Crossing</strong> - Vinal Asodia, Barkin Dagda, Yinglong He, Zhenhua Feng, Saber Fallah - [[pdf]](https://arxiv.org/pdf/2504.08704)</summary>

**Abstract:** Effective leveraging of real-world driving datasets is crucial for enhancing the training of autonomous driving systems. While Offline Reinforcement Learning enables training autonomous vehicles with such data, most available datasets lack meaningful reward labels. Reward labeling is essential as it provides feedback for the learning algorithm to distinguish between desirable and undesirable behaviors, thereby improving policy performance. This paper presents a novel approach for generating human-aligned reward labels. The proposed approach addresses the challenge of absent reward signals in the real-world datasets by generating labels that reflect human judgment and safety considerations. The reward function incorporates an adaptive safety component that is activated by analyzing semantic segmentation maps, enabling the autonomous vehicle to prioritize safety over efficiency in potential collision scenarios. The proposed method is applied to an occluded pedestrian crossing scenario with varying pedestrian traffic levels, using simulation data. When the generated rewards were used to train various Offline Reinforcement Learning algorithms, each model produced a meaningful policy, demonstrating the method's viability. In addition, the method was applied to a subset of the Audi Autonomous Driving Dataset, and the reward labels were compared to human-annotated reward labels. The findings show a moderate disparity between the two reward sets, and, most interestingly, the method flagged unsafe states that the human annotator missed.

**arXiv ID:** 2504.08704
</details>

<details>
<summary><strong>MarioChart: Autonomous Tangibles as Active Proxy Interfaces for Embodied Casual Data Exploration</strong> - Shaozhang Dai, Kadek Ananta Satriadi, Jim Smiley, Barrett Ens, Lonni Besançon, Tim Dwyer - [[pdf]](https://arxiv.org/pdf/2601.18328)</summary>

**Abstract:** We introduce the notion of an Active Proxy interface, i.e. tangible models as proxies for physical data referents, supporting interactive exploration of data through active manipulation. We realise an active proxy data visualisation system, "MarioChart", using robot carts relocating themselves on a tabletop, e.g., to align with their data referents in a map or other visual layout. We consider a casual-data exploration scenario involving a multivariate campus sustainability dataset, using scale models as proxies for their physical building data referents. Our empirical study (n=12) compares active proxy use with conventional tablet interaction, finding that our active proxy system enhances short-term spatial memory of data and enables faster completion of certain data analytic tasks. It shows no significant differences compared to traditional touch-screens in long-term memory, physical fatigue, mental workload, or user engagement. Our study offers an initial baseline for active proxy techniques and advances understanding of tangible interfaces in situated data visualisation.

**arXiv ID:** 2601.18328
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
