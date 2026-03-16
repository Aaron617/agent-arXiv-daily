# Agent arXiv Daily

**Last Updated:** 2026-03-16 03:32:42

**Total Papers:** 69

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (6 papers)</h2></summary>

<details>
<summary><strong>AI Planning Framework for LLM-Based Web Agents</strong> - Orit Shahnovsky, Rotem Dror - [[pdf]](https://arxiv.org/pdf/2603.12710)</summary>

**Abstract:** Developing autonomous agents for web-based tasks is a core challenge in AI. While Large Language Model (LLM) agents can interpret complex user requests, they often operate as black boxes, making it difficult to diagnose why they fail or how they plan. This paper addresses this gap by formally treating web tasks as sequential decision-making processes. We introduce a taxonomy that maps modern agent architectures to traditional planning paradigms: Step-by-Step agents to Breadth-First Search (BFS), Tree Search agents to Best-First Tree Search, and Full-Plan-in-Advance agents to Depth-First Search (DFS). This framework allows for a principled diagnosis of system failures like context drift and incoherent task decomposition. To evaluate these behaviors, we propose five novel evaluation metrics that assess trajectory quality beyond simple success rates. We support this analysis with a new dataset of 794 human-labeled trajectories from the WebArena benchmark. Finally, we validate our evaluation framework by comparing a baseline Step-by-Step agent against a novel Full-Plan-in-Advance implementation. Our results reveal that while the Step-by-Step agent aligns more closely with human gold trajectories (38% overall success), the Full-Plan-in-Advance agent excels in technical measures such as element accuracy (89%), demonstrating the necessity of our proposed metrics for selecting appropriate agent architectures based on specific application constraints.

**arXiv ID:** 2603.12710
</details>

<details>
<summary><strong>Developing and evaluating a chatbot to support maternal health care</strong> - Smriti Jha, Vidhi Jain, Jianyu Xu, Grace Liu, Sowmya Ramesh, Jitender Nagpal, Gretchen Chapman, Benjamin Bellows, Siddhartha Goyal, Aarti Singh, Bryan Wilder - [[pdf]](https://arxiv.org/pdf/2603.13168)</summary>

**Abstract:** The ability to provide trustworthy maternal health information using phone-based chatbots can have a significant impact, particularly in low-resource settings where users have low health literacy and limited access to care. However, deploying such systems is technically challenging: user queries are short, underspecified, and code-mixed across languages, answers require regional context-specific grounding, and partial or missing symptom context makes safe routing decisions difficult.
We present a chatbot for maternal health in India developed through a partnership between academic researchers, a health tech company, a public health nonprofit, and a hospital. The system combines (1) stage-aware triage, routing high-risk queries to expert templates, (2) hybrid retrieval over curated maternal/newborn guidelines, and (3) evidence-conditioned generation from an LLM. Our core contribution is an evaluation workflow for high-stakes deployment under limited expert supervision. Targeting both component-level and end-to-end testing, we introduce: (i) a labeled triage benchmark (N=150) achieving 86.7% emergency recall, explicitly reporting the missed-emergency vs. over-escalation trade-off; (ii) a synthetic multi-evidence retrieval benchmark (N=100) with chunk-level evidence labels; (iii) LLM-as-judge comparison on real queries (N=781) using clinician-codesigned criteria; and (iv) expert validation. Our findings show that trustworthy medical assistants in multilingual, noisy settings require defense-in-depth design paired with multi-method evaluation, rather than any single model and evaluation method choice.

**arXiv ID:** 2603.13168
</details>

<details>
<summary><strong>CRAFT-GUI: Curriculum-Reinforced Agent For GUI Tasks</strong> - Songqin Nong, Xiaoxuan Tang, Jingxuan Xu, Sheng Zhou, Jianfeng Chen, Tao Jiang, Wenhao Xu - [[pdf]](https://arxiv.org/pdf/2508.11360)</summary>

**Abstract:** As autonomous agents become adept at understanding and interacting with graphical user interface (GUI) environments, a new era of automated task execution is emerging. Recent studies have demonstrated that Reinforcement Learning (RL) can effectively enhance agents' performance in dynamic interactive GUI environments. However, these methods face two key limitations: (1) they overlook the significant variation in difficulty across different GUI tasks by treating the entire training data as a uniform set, which hampers the agent's ability to adapt its learning process; and (2) most approaches collapse task-specific nuances into a single, coarse reward, leaving the agent with a uniform signal that yields inefficient policy updates. To address these limitations, we propose CRAFT-GUI, a curriculum learning framework based on Group Relative Policy Optimization (GRPO) that explicitly accounts for the varying difficulty across trajectories. To enable more fine-grained policy optimization, we design a reward function that combines simple rule-based signals with model-judged evaluation, providing richer and more nuanced feedback during training. Experimental results demonstrate that our method achieves significant improvements over previous state-of-the-art approaches, outperforming them by 5.6% on public benchmarks Android Control and 10.3% on our internal online benchmarks, respectively. These findings empirically validate the effectiveness of integrating reinforcement learning with curriculum learning in GUI interaction tasks.

**arXiv ID:** 2508.11360
</details>

<details>
<summary><strong>Building Benchmarks from the Ground Up: Community-Centered Evaluation of LLMs in Healthcare Chatbot Settings</strong> - Hamna Hamna, Gayatri Bhat, Sourabrata Mukherjee, Faisal Lalani, Evan Hadfield, Divya Siddarth, Kalika Bali, Sunayana Sitaram - [[pdf]](https://arxiv.org/pdf/2509.24506)</summary>

**Abstract:** Large Language Models (LLMs) are typically evaluated through general or domain-specific benchmarks testing capabilities that often lack grounding in the lived realities of end users. Critical domains such as healthcare require evaluations that extend beyond artificial or simulated tasks to reflect the everyday needs, cultural practices, and nuanced contexts of communities. We propose Samiksha, a community-driven evaluation pipeline co-created with civil-society organizations (CSOs) and community members. Our approach enables scalable, automated benchmarking through a culturally aware, community-driven pipeline in which community feedback informs what to evaluate, how the benchmark is built, and how outputs are scored. We demonstrate this approach in the health domain in India. Our analysis highlights how current multilingual LLMs address nuanced community health queries, while also offering a scalable pathway for contextually grounded and inclusive LLM evaluation.

**arXiv ID:** 2509.24506
</details>

<details>
<summary><strong>PISmith: Reinforcement Learning-based Red Teaming for Prompt Injection Defenses</strong> - Chenlong Yin, Runpeng Geng, Yanting Wang, Jinyuan Jia - [[pdf]](https://arxiv.org/pdf/2603.13026)</summary>

**Abstract:** Prompt injection poses serious security risks to real-world LLM applications, particularly autonomous agents. Although many defenses have been proposed, their robustness against adaptive attacks remains insufficiently evaluated, potentially creating a false sense of security. In this work, we propose PISmith, a reinforcement learning (RL)-based red-teaming framework that systematically assesses existing prompt-injection defenses by training an attack LLM to optimize injected prompts in a practical black-box setting, where the attacker can only query the defended LLM and observe its outputs. We find that directly applying standard GRPO to attack strong defenses leads to sub-optimal performance due to extreme reward sparsity -- most generated injected prompts are blocked by the defense, causing the policy's entropy to collapse before discovering effective attack strategies, while the rare successes cannot be learned effectively. In response, we introduce adaptive entropy regularization and dynamic advantage weighting to sustain exploration and amplify learning from scarce successes. Extensive evaluation on 13 benchmarks demonstrates that state-of-the-art prompt injection defenses remain vulnerable to adaptive attacks. We also compare PISmith with 7 baselines across static, search-based, and RL-based attack categories, showing that PISmith consistently achieves the highest attack success rates. Furthermore, PISmith achieves strong performance in agentic settings on InjecAgent and AgentDojo against both open-source and closed-source LLMs (e.g., GPT-4o-mini and GPT-5-nano). Our code is available at this https URL.

**arXiv ID:** 2603.13026
</details>

<details>
<summary><strong>Better Safe Than Sorry: Enhancing Arbitration Graphs for Safe and Robust Autonomous Decision-Making</strong> - Piotr Spieker, Nick Le Large, Martin Lauer - [[pdf]](https://arxiv.org/pdf/2411.10170)</summary>

**Abstract:** This paper introduces an extension to the arbitration graph framework designed to enhance the safety and robustness of autonomous systems in complex, dynamic environments. Building on the flexibility and scalability of arbitration graphs, the proposed method incorporates a verification step and structured fallback layers in the decision-making process. This ensures that only verified and safe commands are executed while enabling graceful degradation in the presence of unexpected faults or bugs. The approach is demonstrated using a Pac-Man simulation and further validated in the context of autonomous driving, where it shows significant reductions in accident risk and improvements in overall system safety. The bottom-up design of arbitration graphs allows for an incremental integration of new behavior components. The extension presented in this work enables the integration of experimental or immature behavior components while maintaining system safety by clearly and precisely defining the conditions under which behaviors are considered safe. The proposed method is implemented as a ready to use header-only C++ library, published under the MIT License. Together with the Pac-Man demo, it is available at this http URL.

**arXiv ID:** 2411.10170
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (16 papers)</h2></summary>

<details>
<summary><strong>Generating Expressive and Customizable Evals for Timeseries Data Analysis Agents with AgentFuel</strong> - Aadyaa Maddi, Prakhar Naval, Deepti Mande, Shane Duan, Muckai Girish, Vyas Sekar - [[pdf]](https://arxiv.org/pdf/2603.12483)</summary>

**Abstract:** Across many domains (e.g., IoT, observability, telecommunications, cybersecurity), there is an emerging adoption of conversational data analysis agents that enable users to "talk to your data" to extract insights. Such data analysis agents operate on timeseries data models; e.g., measurements from sensors or events monitoring user clicks and actions in product analytics. We evaluate 6 popular data analysis agents (both open-source and proprietary) on domain-specific data and query types, and find that they fail on stateful and incident-specific queries. We observe two key expressivity gaps in existing evals: domain-customized datasets and domain-specific query types. To enable practitioners in such domains to generate customized and expressive evals for such timeseries data agents, we present AgentFuel. AgentFuel helps domain experts quickly create customized evals to perform end-to-end functional tests. We show that AgentFuel's benchmarks expose key directions for improvement in existing data agent frameworks. We also present anecdotal evidence that using AgentFuel can improve agent performance (e.g., with GEPA). AgentFuel benchmarks are available at this https URL.

**arXiv ID:** 2603.12483
</details>

<details>
<summary><strong>Steve-Evolving: Open-World Embodied Self-Evolution via Fine-Grained Diagnosis and Dual-Track Knowledge Distillation</strong> - Zhengwei Xie, Zhisheng Chen, Ziyan Weng, Tingyu Wu, Chenglong Li, Vireo Zhang, Kun Wang - [[pdf]](https://arxiv.org/pdf/2603.13131)</summary>

**Abstract:** Open-world embodied agents must solve long-horizon tasks where the main bottleneck is not single-step planning quality but how interaction experience is organized and evolved. To this end, we present Steve-Evolving, a non-parametric self-evolving framework that tightly couples fine-grained execution diagnosis with dual-track knowledge distillation in a closed loop. The method follows three phases: Experience Anchoring, Experience Distillation, and Knowledge-Driven Closed-Loop Control. In detail, Experience Anchoring solidifies each subgoal attempt into a structured experience tuple with a fixed schema (pre-state, action, diagnosis-result, and post-state) and organizes it in a three-tier experience space with multi-dimensional indices (e.g., condition signatures, spatial hashing, and semantic tags) plus rolling summarization for efficient and auditable recall. To ensure sufficient information density for attribution, the execution layer provides compositional diagnosis signals beyond binary outcomes, including state-difference summaries, enumerated failure causes, continuous indicators, and stagnation/loop detection. Moreover, successful trajectories of Experience Distillation are generalized into reusable skills with explicit preconditions and verification criteria, while failures are distilled into executable guardrails that capture root causes and forbid risky operations at both subgoal and task granularities. Besides, Knowledge-Driven Closed-Loop Control retrieved skills and guardrails are injected into an LLM planner, and diagnosis-triggered local replanning updates the active constraints online, forming a continual evolution process without any model parameter updates. Experiments on the long-horizon suite of Minecraft MCU demonstrate consistent improvements over static-retrieval baselines.

**arXiv ID:** 2603.13131
</details>

<details>
<summary><strong>Test-Time Strategies for More Efficient and Accurate Agentic RAG</strong> - Brian Zhang, Deepti Guntur, Zhiyang Zuo, Abhinav Sharma, Shreyas Chaudhari, Wenlong Zhao, Franck Dernoncourt, Puneet Mathur, Ryan Rossi, Nedim Lipka - [[pdf]](https://arxiv.org/pdf/2603.12396)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) systems face challenges with complex, multihop questions, and agentic frameworks such as Search-R1 (Jin et al., 2025), which operates iteratively, have been proposed to address these complexities. However, such approaches can introduce inefficiencies, including repetitive retrieval of previously processed information and challenges in contextualizing retrieved results effectively within the current generation prompt. Such issues can lead to unnecessary retrieval turns, suboptimal reasoning, inaccurate answers, and increased token consumption.
In this paper, we investigate test-time modifications to the Search-R1 pipeline to mitigate these identified shortcomings. Specifically, we explore the integration of two components and their combination: a contextualization module to better integrate relevant information from retrieved documents into reasoning, and a de-duplication module that replaces previously retrieved documents with the next most relevant ones. We evaluate our approaches using the HotpotQA (Yang et al., 2018) and the Natural Questions (Kwiatkowski et al., 2019) datasets, reporting the exact match (EM) score, an LLM-as-a-Judge assessment of answer correctness, and the average number of turns.
Our best-performing variant, utilizing GPT-4.1-mini for contextualization, achieves a 5.6% increase in EM score and reduces the number of turns by 10.5% compared to the Search-R1 baseline, demonstrating improved answer accuracy and retrieval efficiency.

**arXiv ID:** 2603.12396
</details>

<details>
<summary><strong>Feynman: Knowledge-Infused Diagramming Agent for Scalable Visual Designs</strong> - Zixin Wen, Yifu Cai, Kyle Lee, Sam Estep, Josh Sunshine, Aarti Singh, Yuejie Chi, Wode Ni - [[pdf]](https://arxiv.org/pdf/2603.12597)</summary>

**Abstract:** Visual design is an essential application of state-of-the-art multi-modal AI systems. Improving these systems requires high-quality vision-language data at scale. Despite the abundance of internet image and text data, knowledge-rich and well-aligned image-text pairs are rare. In this paper, we present a scalable diagram generation pipeline built with our agent, Feynman. To create diagrams, Feynman first enumerates domain-specific knowledge components (''ideas'') and performs code planning based on the ideas. Given the plan, Feynman translates ideas into simple declarative programs and iterates to receives feedback and visually refine diagrams. Finally, the declarative programs are rendered by the Penrose diagramming system. The optimization-based rendering of Penrose preserves the visual semantics while injecting fresh randomness into the layout, thereby producing diagrams with visual consistency and diversity. As a result, Feynman can author diagrams along with grounded captions with very little cost and time. Using Feynman, we synthesized a dataset with more than 100k well-aligned diagram-caption pairs. We also curate a visual-language benchmark, Diagramma, from freshly generated data. Diagramma can be used for evaluating the visual reasoning capabilities of vision-language models. We plan to release the dataset, benchmark, and the full agent pipeline as an open-source project.

**arXiv ID:** 2603.12597
</details>

<details>
<summary><strong>CarPLAN: Context-Adaptive and Robust Planning with Dynamic Scene Awareness for Autonomous Driving</strong> - Junyong Yun, Jungho Kim, ByungHyun Lee, Dongyoung Lee, Sehwan Choi, Seunghyeop Nam, Kichun Jo, Jun Won Choi - [[pdf]](https://arxiv.org/pdf/2603.12607)</summary>

**Abstract:** Imitation learning (IL) is widely used for motion planning in autonomous driving due to its data efficiency and access to real-world driving data. For safe and robust real-world driving, IL-based planning requires capturing the complex driving contexts inherent in real-world data and enabling context-adaptive decision-making, rather than relying solely on expert trajectory imitation. In this paper, we propose CarPLAN, a novel IL-based motion planning framework that explicitly enhances driving context understanding and enables adaptive planning across diverse traffic scenarios. Our contributions are twofold: We introduce Displacement-Aware Predictive Encoding (DPE) to improve the model's spatial awareness by predicting future displacement vectors between the Autonomous Vehicle (AV) and surrounding scene elements. This allows the planner to account for relational spacing when generating trajectories. In addition to the standard imitation loss, we incorporate an augmented loss term that captures displacement prediction errors, ensuring planning decisions consider relative distances from other agents. To improve the model's ability to handle diverse driving contexts, we propose Context-Adaptive Multi-Expert Decoder (CMD), which leverages the Mixture of Experts (MoE) framework. CMD dynamically selects the most suitable expert decoders based on scene structure at each Transformer layer, enabling adaptive and context-aware planning in dynamic environments. We evaluate CarPLAN on the nuPlan benchmark and demonstrate state-of-the-art performance across all closed-loop simulation metrics. In particular, CarPLAN exhibits robust performance on challenging scenarios such as Test14-Hard, validating its effectiveness in complex driving conditions. Additional experiments on the Waymax benchmark further demonstrate its generalization capability across different benchmark settings.

**arXiv ID:** 2603.12607
</details>

<details>
<summary><strong>Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents</strong> - Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, Jeff Clune - [[pdf]](https://arxiv.org/pdf/2505.22954)</summary>

**Abstract:** Today's AI systems have human-designed, fixed architectures and cannot autonomously and continuously improve themselves. The advance of AI could itself be automated. If done safely, that would accelerate AI development and allow us to reap its benefits much sooner. Meta-learning can automate the discovery of novel algorithms, but is limited by first-order improvements and the human design of a suitable search space. The Gödel machine proposed a theoretical alternative: a self-improving AI that repeatedly modifies itself in a provably beneficial manner. Unfortunately, proving that most changes are net beneficial is impossible in practice. We introduce the Darwin Gödel Machine (DGM), a self-improving system that iteratively modifies its own code (thereby also improving its ability to modify its own codebase) and empirically validates each change using coding benchmarks. Inspired by Darwinian evolution and open-endedness research, the DGM maintains an archive of generated coding agents. It grows the archive by sampling an agent from it and using a foundation model to create a new, interesting, version of the sampled agent. This open-ended exploration forms a growing tree of diverse, high-quality agents and allows the parallel exploration of many different paths through the search space. Empirically, the DGM automatically improves its coding capabilities (e.g., better code editing tools, long-context window management, peer-review mechanisms), increasing performance on SWE-bench from 20.0% to 50.0%, and on Polyglot from 14.2% to 30.7%. Furthermore, the DGM significantly outperforms baselines without self-improvement or open-ended exploration. All experiments were done with safety precautions (e.g., sandboxing, human oversight). The DGM is a significant step toward self-improving AI, capable of gathering its own stepping stones along paths that unfold into endless innovation.

**arXiv ID:** 2505.22954
</details>

<details>
<summary><strong>AutoClimDS: Climate Data Science Agentic AI -- A Knowledge Graph is All You Need</strong> - Ahmed Jaber, Wangshu Zhu, Ayon Roy, Karthick Jayavelu, Justin Downes, Sameer Mohamed, Candace Agonafir, Linnia Hawkins, Tian Zheng - [[pdf]](https://arxiv.org/pdf/2509.21553)</summary>

**Abstract:** Climate data science remains constrained by fragmented data sources, heterogeneous formats, and steep technical expertise requirements. These barriers slow discovery, limit participation, and undermine reproducibility. We present AutoClimDS, a Minimum Viable Product (MVP) Agentic AI system that addresses these challenges by integrating a curated climate knowledge graph (KG) with a set of Agentic AI workflows designed for cloud-native scientific analysis. The KG unifies datasets, metadata, tools, and workflows into a machine-interpretable structure, while AI agents, powered by generative models, enable natural-language query interpretation, automated data discovery, programmatic data acquisition, and end-to-end climate analysis. A key result is that AutoClimDS can reproduce published scientific figures and analyses from natural-language instructions alone, completing the entire workflow from dataset selection to preprocessing to modeling. When given the same tasks, state-of-the-art general-purpose LLMs (e.g., ChatGPT GPT-5.1) cannot independently identify authoritative datasets or construct valid retrieval workflows using standard web access. This highlights the necessity of structured scientific memory for agentic scientific reasoning. By encoding procedural workflow knowledge into a KG and integrating it with existing technologies (cloud APIs, LLMs, sandboxed execution), AutoClimDS demonstrates that the KG serves as the essential enabling component, the irreplaceable structural foundation, for autonomous climate data science. This approach provides a pathway toward democratizing climate research through human-AI collaboration.

**arXiv ID:** 2509.21553
</details>

<details>
<summary><strong>XSkill: Continual Learning from Experience and Skills in Multimodal Agents</strong> - Guanyu Jiang, Zhaochen Su, Xiaoye Qu, Yi R. Fung - [[pdf]](https://arxiv.org/pdf/2603.12056)</summary>

**Abstract:** Multimodal agents can now tackle complex reasoning tasks with diverse tools, yet they still suffer from inefficient tool use and inflexible orchestration in open-ended settings. A central challenge is enabling such agents to continually improve without parameter updates by learning from past trajectories. We identify two complementary forms of reusable knowledge essential for this goal: experiences, providing concise action-level guidance for tool selection and decision making, and skills, providing structured task-level guidance for planning and tool use. To this end, we propose XSkill, a dual-stream framework for continual learning from experience and skills in multimodal agents. XSkill grounds both knowledge extraction and retrieval in visual observations. During accumulation, XSkill distills and consolidates experiences and skills from multi-path rollouts via visually grounded summarization and cross-rollout critique. During inference, it retrieves and adapts this knowledge to the current visual context and feeds usage history back into accumulation to form a continual learning loop. Evaluated on five benchmarks across diverse domains with four backbone models, XSkill consistently and substantially outperforms both tool-only and learning-based baselines. Further analysis reveals that the two knowledge streams play complementary roles in influencing the reasoning behaviors of agents and show superior zero-shot generalization.

**arXiv ID:** 2603.12056
</details>

<details>
<summary><strong>Scaling Generalist Data-Analytic Agents</strong> - Shuofei Qiao, Yanqiu Zhao, Zhisong Qiu, Xiaobin Wang, Jintian Zhang, Zhao Bin, Ningyu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen - [[pdf]](https://arxiv.org/pdf/2509.25084)</summary>

**Abstract:** Data-analytic agents are emerging as a key catalyst for automated scientific discovery and for the vision of Innovating AI. Current approaches, however, rely heavily on prompt engineering over proprietary models, while open-source models struggle to face diverse-format, large-scale data files and long-horizon, multi-step reasoning that real-world analytics demands. This paper introduces DataMind, a scalable data synthesis and agent training recipe designed to build generalist data-analytic agents. DataMind tackles three key challenges in building open-source data-analytic agents, including insufficient data resources, improper training strategy, and unstable code-based multi-turn rollout. Concretely, DataMind applies 1) a fine-grained task taxonomy and a recursive easy-to-hard task composition mechanism to increase the diversity and difficulty of synthesized queries; 2) a knowledge-augmented trajectory sampling strategy followed by model-based and rule-based filtering; 3) a dynamically adjustable training objective combining both SFT and RL losses; 4) a memory-frugal and stable code-based multi-turn rollout framework. Built on DataMind, we curate DataMind-12K, a high-quality trajectory set spanning diverse domains, task categories, and data file formats for data-analytic tasks. Trained on DataMind-12K, our DataMind-14B achieves state-of-the-art with an average score of 71.16% on multiple data analysis benchmarks, outperforming the strongest proprietary baselines DeepSeek-V3.1 and GPT-5. Our DataMind-7B also performs best among all open-source models with a score of 68.10%. We also incorporate some empirical insights gained from our exploratory trials into the analysis experiments, aiming to provide actionable insights about agentic training for the community. We will release DataMind-12K and DataMind-7B,14B for the community's future research.

**arXiv ID:** 2509.25084
</details>

<details>
<summary><strong>MalURLBench: A Benchmark Evaluating Agents' Vulnerabilities When Processing Web URLs</strong> - Dezhang Kong, Zhuxi Wu, Shiqi Liu, Zhicheng Tan, Kuichen Lu, Minghao Li, Qichen Liu, Shengyu Chu, Zhenhua Xu, Xuan Liu, Meng Han - [[pdf]](https://arxiv.org/pdf/2601.18113)</summary>

**Abstract:** LLM-based web agents have become increasingly popular for their utility in daily life and work. However, they exhibit critical vulnerabilities when processing malicious URLs: accepting a disguised malicious URL enables subsequent access to unsafe webpages, which can cause severe damage to service providers and users. Despite this risk, no benchmark currently targets this emerging threat. To address this gap, we propose MalURLBench, the first benchmark for evaluating LLMs' vulnerabilities to malicious URLs. MalURLBench contains 61,845 attack instances spanning 10 real-world scenarios and 7 categories of real malicious websites. Experiments with 12 popular LLMs reveal that existing models struggle to detect elaborately disguised malicious URLs. We further identify and analyze key factors that impact attack success rates and propose URLGuard, a lightweight defense module. We believe this work will provide a foundational resource for advancing the security of web agents. Our code is available at this https URL.

**arXiv ID:** 2601.18113
</details>

<details>
<summary><strong>HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios</strong> - Jiayue Pu, Zhongxiang Sun, Zilu Zhang, Xiao Zhang, Jun Xu - [[pdf]](https://arxiv.org/pdf/2603.11975)</summary>

**Abstract:** The rapid evolution of embodied agents has accelerated the deployment of household robots in real-world environments. However, unlike structured industrial settings, household spaces introduce unpredictable safety risks, where system limitations such as perception latency and lack of common sense knowledge can lead to dangerous errors. Current safety evaluations, often restricted to static images, text, or general hazards, fail to adequately benchmark dynamic unsafe action detection in these specific contexts. To bridge this gap, we introduce HomeSafe-Bench, a challenging benchmark designed to evaluate Vision-Language Models (VLMs) on unsafe action detection in household scenarios. HomeSafe-Bench is contrusted via a hybrid pipeline combining physical simulation with advanced video generation and features 438 diverse cases across six functional areas with fine-grained multidimensional annotations. Beyond benchmarking, we propose Hierarchical Dual-Brain Guard for Household Safety (HD-Guard), a hierarchical streaming architecture for real-time safety monitoring. HD-Guard coordinates a lightweight FastBrain for continuous high-frequency screening with an asynchronous large-scale SlowBrain for deep multimodal reasoning, effectively balancing inference efficiency with detection accuracy. Evaluations demonstrate that HD-Guard achieves a superior trade-off between latency and performance, while our analysis identifies critical bottlenecks in current VLM-based safety detection.

**arXiv ID:** 2603.11975
</details>

<details>
<summary><strong>Adaptive Vision-Language Model Routing for Computer Use Agents</strong> - Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen - [[pdf]](https://arxiv.org/pdf/2603.12823)</summary>

**Abstract:** Computer Use Agents (CUAs) translate natural-language instructions into Graphical User Interface (GUI) actions such as clicks, keystrokes, and scrolls by relying on a Vision-Language Model (VLM) to interpret screenshots and predict grounded tool calls. However, grounding accuracy varies dramatically across VLMs, while current CUA systems typically route every action to a single fixed model regardless of difficulty. We propose \textbf{Adaptive VLM Routing} (AVR), a framework that inserts a lightweight semantic routing layer between the CUA orchestrator and a pool of VLMs. For each tool call, AVR estimates action difficulty from multimodal embeddings, probes a small VLM to measure confidence, and routes the action to the cheapest model whose predicted accuracy satisfies a target reliability threshold. For \textit{warm} agents with memory of prior UI interactions, retrieved context further narrows the capability gap between small and large models, allowing many actions to be handled without escalation. We formalize routing as a cost--accuracy trade-off, derive a threshold-based policy for model selection, and evaluate AVR using ScreenSpot-Pro grounding data together with the OpenClaw agent routing benchmark. Across these settings, AVR projects inference cost reductions of up to 78\% while staying within 2 percentage points of an all-large-model baseline. When combined with the Visual Confused Deputy guardrail, AVR also escalates high-risk actions directly to the strongest available model, unifying efficiency and safety within a single routing framework. Materials are also provided Model, benchmark, and code: this https URL.

**arXiv ID:** 2603.12823
</details>

<details>
<summary><strong>RECAP: Reproducing Copyrighted Data from LLMs Training with an Agentic Pipeline</strong> - André V. Duarte, Xuying li, Bin Zeng, Arlindo L. Oliveira, Lei Li, Zhuo Li - [[pdf]](https://arxiv.org/pdf/2510.25941)</summary>

**Abstract:** If we cannot inspect the training data of a large language model (LLM), how can we ever know what it has seen? We believe the most compelling evidence arises when the model itself freely reproduces the target content. As such, we propose RECAP, an agentic pipeline designed to elicit and verify memorized training data from LLM outputs. At the heart of RECAP is a feedback-driven loop, where an initial extraction attempt is evaluated by a secondary language model, which compares the output against a reference passage and identifies discrepancies. These are then translated into minimal correction hints, which are fed back into the target model to guide subsequent generations. In addition, to address alignment-induced refusals, RECAP includes a jailbreaking module that detects and overcomes such barriers. We evaluate RECAP on EchoTrace, a new benchmark spanning over 30 full books, and the results show that RECAP leads to substantial gains over single-iteration approaches. For instance, with GPT-4.1, the average ROUGE-L score for the copyrighted text extraction improved from 0.38 to 0.47 - a nearly 24% increase.

**arXiv ID:** 2510.25941
</details>

<details>
<summary><strong>ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models</strong> - Yanpeng Zhao, Wentao Ding, Hongtao Li, Baoxiong Jia, Zilong Zheng - [[pdf]](https://arxiv.org/pdf/2603.13033)</summary>

**Abstract:** A recent trend in vision-language models (VLMs) has been to enhance their spatial cognition for embodied domains. Despite progress, existing evaluations have been limited both in paradigm and in coverage, hindering rapid, iterative model development. To address these limitations, we propose ESPIRE, a diagnostic benchmark for embodied spatial reasoning. ESPIRE offers a simulated world that physically grounds VLMs and evaluates them on spatial-reasoning-centric robotic tasks, thus narrowing the gap between evaluation and real-world deployment. To adapt VLMs to robotic tasks, we decompose each task into localization and execution, and frame both as generative problems, in stark contrast to predominant discriminative evaluations (e.g., via visual-question answering) that rely on distractors and discard execution. This decomposition further enables a fine-grained analysis beyond passive spatial reasoning toward reasoning to act. We systematically design ESPIRE both at the instruction level and at the environment level, ensuring broad coverage of spatial reasoning scenarios. We use ESPIRE to diagnose a range of frontier VLMs and provide in-depth analysis of their spatial reasoning behaviors.

**arXiv ID:** 2603.13033
</details>

<details>
<summary><strong>Dynamic Aware: Adaptive Multi-Mode Out-of-Distribution Detection for Trajectory Prediction in Autonomous Vehicles</strong> - Tongfei Guo, Lili Su - [[pdf]](https://arxiv.org/pdf/2509.13577)</summary>

**Abstract:** Trajectory prediction is central to the safe and seamless operation of autonomous vehicles (AVs). In deployment, however, prediction models inevitably face distribution shifts between training data and real-world conditions, where rare or underrepresented traffic scenarios induce out-of-distribution (OOD) cases. While most prior OOD detection research in AVs has concentrated on computer vision tasks such as object detection and segmentation, trajectory-level OOD detection remains largely underexplored. A recent study formulated this problem as a quickest change detection (QCD) task, providing formal guarantees on the trade-off between detection delay and false alarms [1]. Building on this foundation, we propose a new framework that introduces adaptive mechanisms to achieve robust detection in complex driving environments. Empirical analysis across multiple real-world datasets reveals that prediction errors -- even on in-distribution samples -- exhibit mode-dependent distributions that evolve over time with dataset-specific dynamics. By explicitly modeling these error modes, our method achieves substantial improvements in both detection delay and false alarm rates. Comprehensive experiments on established trajectory prediction benchmarks show that our framework significantly outperforms prior UQ- and vision-based OOD approaches in both accuracy and computational efficiency, offering a practical path toward reliable, driving-aware autonomy.

**arXiv ID:** 2509.13577
</details>

<details>
<summary><strong>DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving</strong> - Shuyao Shang, Bing Zhan, Yunfei Yan, Yuqi Wang, Yingyan Li, Yasong An, Xiaoman Wang, Jierui Liu, Lu Hou, Lue Fan, Zhaoxiang Zhang, Tieniu Tan - [[pdf]](https://arxiv.org/pdf/2603.11041)</summary>

**Abstract:** We propose DynVLA, a driving VLA model that introduces a new CoT paradigm termed Dynamics CoT. DynVLA forecasts compact world dynamics before action generation, enabling more informed and physically grounded decision-making. To obtain compact dynamics representations, DynVLA introduces a Dynamics Tokenizer that compresses future evolution into a small set of dynamics tokens. Considering the rich environment dynamics in interaction-intensive driving scenarios, DynVLA decouples ego-centric and environment-centric dynamics, yielding more accurate world dynamics modeling. We then train DynVLA to generate dynamics tokens before actions through SFT and RFT, improving decision quality while maintaining latency-efficient inference. Compared to Textual CoT, which lacks fine-grained spatiotemporal understanding, and Visual CoT, which introduces substantial redundancy due to dense image prediction, Dynamics CoT captures the evolution of the world in a compact, interpretable, and efficient form. Extensive experiments on NAVSIM, Bench2Drive, and a large-scale in-house dataset demonstrate that DynVLA consistently outperforms Textual CoT and Visual CoT methods, validating the effectiveness and practical value of Dynamics CoT. Project Page: this https URL.

**arXiv ID:** 2603.11041
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning</strong> - Shuo Yang, Soyeon Caren Han, Yihao Ding, Shuhe Wang, Eduard Hoy - [[pdf]](https://arxiv.org/pdf/2603.12740)</summary>

**Abstract:** Large Language Model (LLM) agents are increasingly applied to complex, multi-step tasks that require interaction with diverse external tools across various domains. However, current LLM agent tool planning methods typically rely on greedy, reactive tool selection strategies that lack foresight and fail to account for inter-tool dependencies. In this paper, we present ToolTree, a novel Monte Carlo tree search-inspired planning paradigm for tool planning. ToolTree explores possible tool usage trajectories using a dual-stage LLM evaluation and bidirectional pruning mechanism that enables the agent to make informed, adaptive decisions over extended tool-use sequences while pruning less promising branches before and after the tool execution. Empirical evaluations across both open-set and closed-set tool planning tasks on 4 benchmarks demonstrate that ToolTree consistently improves performance while keeping the highest efficiency, achieving an average gain of around 10\% compared to the state-of-the-art planning paradigm.

**arXiv ID:** 2603.12740
</details>

<details>
<summary><strong>AgentDrift: Unsafe Recommendation Drift Under Tool Corruption Hidden by Ranking Metrics in LLM Agents</strong> - Zekun Wu, Adriano Koshiyama, Sahan Bulathwela, Maria Perez-Ortiz - [[pdf]](https://arxiv.org/pdf/2603.12564)</summary>

**Abstract:** Tool-augmented LLM agents increasingly serve as multi-turn advisors in high-stakes domains, yet their evaluation relies on ranking-quality metrics that measure what is recommended but not whether it is safe for the user. We introduce a paired-trajectory protocol that replays real financial dialogues under clean and contaminated tool-output conditions across seven LLMs (7B to frontier) and decomposes divergence into information-channel and memory-channel mechanisms. Across the seven models tested, we consistently observe the evaluation-blindness pattern: recommendation quality is largely preserved under contamination (utility preservation ratio approximately 1.0) while risk-inappropriate products appear in 65-93% of turns, a systematic safety failure poorly reflected by standard NDCG. Safety violations are predominantly information-channel-driven, emerge at the first contaminated turn, and persist without self-correction over 23-step trajectories; no agent across 1,563 contaminated turns explicitly questions tool-data reliability. Even narrative-only corruption (biased headlines, no numerical manipulation) induces significant drift while completely evading consistency monitors. A safety-penalized NDCG variant (sNDCG) reduces preservation ratios to 0.51-0.74, indicating that much of the evaluation gap becomes visible once safety is explicitly measured. These results motivate considering trajectory-level safety monitoring, beyond single-turn quality, for deployed multi-turn agents in high-stakes settings.

**arXiv ID:** 2603.12564
</details>

<details>
<summary><strong>Spend Less, Reason Better: Budget-Aware Value Tree Search for LLM Agents</strong> - Yushu Li, Wenlong Deng, Jiajin Li, Xiaoxiao Li - [[pdf]](https://arxiv.org/pdf/2603.12634)</summary>

**Abstract:** Test-time scaling has become a dominant paradigm for improving LLM agent reliability, yet current approaches treat compute as an abundant resource, allowing agents to exhaust token and tool budgets on redundant steps or dead-end trajectories. Existing budget-aware methods either require expensive fine-tuning or rely on coarse, trajectory-level heuristics that cannot intervene mid-execution. We propose the Budget-Aware Value Tree (BAVT), a training-free inference-time framework that models multi-hop reasoning as a dynamic search tree guided by step-level value estimation within a single LLM backbone. Another key innovation is a budget-conditioned node selection mechanism that uses the remaining resource ratio as a natural scaling exponent over node values, providing a principled, parameter-free transition from broad exploration to greedy exploitation as the budget depletes. To combat the well-known overconfidence of LLM self-evaluation, BAVT employs a residual value predictor that scores relative progress rather than absolute state quality, enabling reliable pruning of uninformative or redundant tool calls. We further provide a theoretical convergence guarantee, proving that BAVT reaches a terminal answer with probability at least $1-\epsilon$ under an explicit finite budget bound. Extensive evaluations on four multi-hop QA benchmarks across two model families demonstrate that BAVT consistently outperforms parallel sampling baselines. Most notably, BAVT under strict low-budget constraints surpasses baseline performance at $4\times$ the resource allocation, establishing that intelligent budget management fundamentally outperforms brute-force compute scaling.

**arXiv ID:** 2603.12634
</details>

<details>
<summary><strong>SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks</strong> - Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, Shuyi Wang, Binxu Li, Qunhong Zeng, Di Wang, Xuandong Zhao, Yuanli Wang, Roey Ben Chaim, Zonglin Di, Yipeng Gao, Junwei He, Yizhuo He, Liqiang Jing, Luyang Kong, Xin Lan, Jiachen Li, Songlin Li, Yijiang Li, Yueqian Lin, Xinyi Liu, Xuanqing Liu, Haoran Lyu, Ze Ma, Bowei Wang, Runhui Wang, Tianyu Wang, Wengao Ye, Yue Zhang, Hanwen Xing, Yiqi Xue, Steven Dillmann, Han-chung Lee - [[pdf]](https://arxiv.org/pdf/2602.12670)</summary>

**Abstract:** Agent Skills are structured packages of procedural knowledge that augment LLM agents at inference time. Despite rapid adoption, there is no standard way to measure whether they actually help. We present SkillsBench, a benchmark of 86 tasks across 11 domains paired with curated Skills and deterministic verifiers. Each task is evaluated under three conditions: no Skills, curated Skills, and self-generated Skills. We test 7 agent-model configurations over 7,308 trajectories. Curated Skills raise average pass rate by 16.2 percentage points(pp), but effects vary widely by domain (+4.5pp for Software Engineering to +51.9pp for Healthcare) and 16 of 84 tasks show negative deltas. Self-generated Skills provide no benefit on average, showing that models cannot reliably author the procedural knowledge they benefit from consuming. Focused Skills with 2--3 modules outperform comprehensive documentation, and smaller models with Skills can match larger models without them.

**arXiv ID:** 2602.12670
</details>

<details>
<summary><strong>InterDeepResearch: Enabling Human-Agent Collaborative Information Seeking through Interactive Deep Research</strong> - Bo Pan, Lunke Pan, Yitao Zhou, Qi Jiang, Zhen Wen, Minfeng Zhu, Wei Chen - [[pdf]](https://arxiv.org/pdf/2603.12608)</summary>

**Abstract:** Deep research systems powered by LLM agents have transformed complex information seeking by automating the iterative retrieval, filtering, and synthesis of insights from massive-scale web sources. However, existing systems predominantly follow an autonomous "query-to-report" paradigm, limiting users to a passive role and failing to integrate their personal insights, contextual knowledge, and evolving research intents. This paper addresses the lack of human-in-the-loop collaboration in the agentic research process. Through a formative study, we identify that current systems hinder effective human-agent collaboration in terms of process observability, real-time steerability, and context navigation efficiency. Informed by these findings, we propose InterDeepResearch, an interactive deep research system backed by a dedicated research context management framework. The framework organizes research context into a hierarchical architecture with three levels (information, actions, and sessions), enabling dynamic context reduction to prevent LLM context exhaustion and cross-action backtracing for evidence provenance. Built upon this framework, the system interface integrates three coordinated views for visual sensemaking, and dedicated interaction mechanisms for interactive research context navigation. Evaluation on the Xbench-DeepSearch-v1 and Seal-0 benchmarks shows that InterDeepResearch achieves competitive performance compared to state-of-the-art deep research systems, while a formal user study demonstrates its effectiveness in supporting human-agent collaborative information seeking. Project page with system demo: this https URL.

**arXiv ID:** 2603.12608
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (14 papers)</h2></summary>

<details>
<summary><strong>Context is all you need: Towards autonomous model-based process design using agentic AI in flowsheet simulations</strong> - Pascal Schäfer, Lukas J. Krinke, Martin Wlotzka, Norbert Asprion - [[pdf]](https://arxiv.org/pdf/2603.12813)</summary>

**Abstract:** Agentic AI systems integrating large language models (LLMs) with reasoning and tooluse capabilities are transforming various domains - in particular, software development. In contrast, their application in chemical process flowsheet modelling remains largely unexplored. In this work, we present an agentic AI framework that delivers assistance in an industrial flowsheet simulation environment. To this end, we show the capabilities of GitHub Copilot (GitHub, Inc., 2026), when using state-of-the-art LLMs, such as Claude Opus 4.6 (Anthropic, PBC, 2026), to generate valid syntax for our in-house process modelling tool Chemasim using the technical documentation and a few commented examples as context. Based on this, we develop a multi-agent system that decomposes process development tasks with one agent solving the abstract problem using engineering knowledge and another agent implementing the solution as Chemasim code. We demonstrate the effectiveness of our framework for typical flowsheet modelling examples, including (i) a reaction/separation process, (ii) a pressure-swing distillation, and (iii) a heteroazeotropic distillation including entrainer selection. Along these lines, we discuss current limitations of the framework and outline future research directions to further enhance its capabilities.

**arXiv ID:** 2603.12813
</details>

<details>
<summary><strong>Efficient and Interpretable Multi-Agent LLM Routing via Ant Colony Optimization</strong> - Xudong Wang, Chaoning Zhang, Jiaquan Zhang, Chenghao Li, Qigan Sun, Sung-Ho Bae, Peng Wang, Ning Xie, Jie Zou, Yang Yang, Hengtao Shen - [[pdf]](https://arxiv.org/pdf/2603.12933)</summary>

**Abstract:** Large Language Model (LLM)-driven Multi-Agent Systems (MAS) have demonstrated strong capability in complex reasoning and tool use, and heterogeneous agent pools further broaden the quality--cost trade-off space. Despite these advances, real-world deployment is often constrained by high inference cost, latency, and limited transparency, which hinders scalable and efficient routing. Existing routing strategies typically rely on expensive LLM-based selectors or static policies, and offer limited controllability for semantic-aware routing under dynamic loads and mixed intents, often resulting in unstable performance and inefficient resource utilization. To address these limitations, we propose AMRO-S, an efficient and interpretable routing framework for Multi-Agent Systems (MAS). AMRO-S models MAS routing as a semantic-conditioned path selection problem, enhancing routing performance through three key mechanisms: First, it leverages a supervised fine-tuned (SFT) small language model for intent inference, providing a low-overhead semantic interface for each query; second, it decomposes routing memory into task-specific pheromone specialists, reducing cross-task interference and optimizing path selection under mixed workloads; finally, it employs a quality-gated asynchronous update mechanism to decouple inference from learning, optimizing routing without increasing latency. Extensive experiments on five public benchmarks and high-concurrency stress tests demonstrate that AMRO-S consistently improves the quality--cost trade-off over strong routing baselines, while providing traceable routing evidence through structured pheromone patterns.

**arXiv ID:** 2603.12933
</details>

<details>
<summary><strong>Semantic Invariance in Agentic AI</strong> - I. de Zarzà, J. de Curtò, Jordi Cabot, Pietro Manzoni, Carlos T. Calafate - [[pdf]](https://arxiv.org/pdf/2603.13173)</summary>

**Abstract:** Large Language Models (LLMs) increasingly serve as autonomous reasoning agents in decision support, scientific problem-solving, and multi-agent coordination systems. However, deploying LLM agents in consequential applications requires assurance that their reasoning remains stable under semantically equivalent input variations, a property we term semantic this http URL benchmark evaluations, which assess accuracy on fixed, canonical problem formulations, fail to capture this critical reliability dimension. To address this shortcoming, in this paper we present a metamorphic testing framework for systematically assessing the robustness of LLM reasoning agents, applying eight semantic-preserving transformations (identity, paraphrase, fact reordering, expansion, contraction, academic context, business context, and contrastive formulation) across seven foundation models spanning four distinct architectural families: Hermes (70B, 405B), Qwen3 (30B-A3B, 235B-A22B), DeepSeek-R1, and gpt-oss (20B, 120B). Our evaluation encompasses 19 multi-step reasoning problems across eight scientific domains. The results reveal that model scale does not predict robustness: the smaller Qwen3-30B-A3B achieves the highest stability (79.6% invariant responses, semantic similarity 0.91), while larger models exhibit greater fragility.

**arXiv ID:** 2603.13173
</details>

<details>
<summary><strong>VQQA: An Agentic Approach for Video Evaluation and Quality Improvement</strong> - Yiwen Song, Tomas Pfister, Yale Song - [[pdf]](https://arxiv.org/pdf/2603.12310)</summary>

**Abstract:** Despite rapid advancements in video generation models, aligning their outputs with complex user intent remains challenging. Existing test-time optimization methods are typically either computationally expensive or require white-box access to model internals. To address this, we present VQQA (Video Quality Question Answering), a unified, multi-agent framework generalizable across diverse input modalities and video generation tasks. By dynamically generating visual questions and using the resulting Vision-Language Model (VLM) critiques as semantic gradients, VQQA replaces traditional, passive evaluation metrics with human-interpretable, actionable feedback. This enables a highly efficient, closed-loop prompt optimization process via a black-box natural language interface. Extensive experiments demonstrate that VQQA effectively isolates and resolves visual artifacts, substantially improving generation quality in just a few refinement steps. Applicable to both text-to-video (T2V) and image-to-video (I2V) tasks, our method achieves absolute improvements of +11.57% on T2V-CompBench and +8.43% on VBench2 over vanilla generation, significantly outperforming state-of-the-art stochastic search and prompt optimization techniques.

**arXiv ID:** 2603.12310
</details>

<details>
<summary><strong>LLM Constitutional Multi-Agent Governance</strong> - J. de Curtò, I. de Zarzà - [[pdf]](https://arxiv.org/pdf/2603.13189)</summary>

**Abstract:** Large Language Models (LLMs) can generate persuasive influence strategies that shift cooperative behavior in multi-agent populations, but a critical question remains: does the resulting cooperation reflect genuine prosocial alignment, or does it mask erosion of agent autonomy, epistemic integrity, and distributional fairness? We introduce Constitutional Multi-Agent Governance (CMAG), a two-stage framework that interposes between an LLM policy compiler and a networked agent population, combining hard constraint filtering with soft penalized-utility optimization that balances cooperation potential against manipulation risk and autonomy pressure. We propose the Ethical Cooperation Score (ECS), a multiplicative composite of cooperation, autonomy, integrity, and fairness that penalizes cooperation achieved through manipulative means. In experiments on scale-free networks of 80 agents under adversarial conditions (70% violating candidates), we benchmark three regimes: full CMAG, naive filtering, and unconstrained optimization. While unconstrained optimization achieves the highest raw cooperation (0.873), it yields the lowest ECS (0.645) due to severe autonomy erosion (0.867) and fairness degradation (0.888). CMAG attains an ECS of 0.741, a 14.9% improvement, while preserving autonomy at 0.985 and integrity at 0.995, with only modest cooperation reduction to 0.770. The naive ablation (ECS = 0.733) confirms that hard constraints alone are insufficient. Pareto analysis shows CMAG dominates the cooperation-autonomy trade-off space, and governance reduces hub-periphery exposure disparities by over 60%. These findings establish that cooperation is not inherently desirable without governance: constitutional constraints are necessary to ensure that LLM-mediated influence produces ethically stable outcomes rather than manipulative equilibria.

**arXiv ID:** 2603.13189
</details>

<details>
<summary><strong>Multi-Agent Guided Policy Optimization</strong> - Yueheng Li, Guangming Xie, Zongqing Lu - [[pdf]](https://arxiv.org/pdf/2507.18059)</summary>

**Abstract:** Due to practical constraints such as partial observability and limited communication, Centralized Training with Decentralized Execution (CTDE) has become the dominant paradigm in cooperative Multi-Agent Reinforcement Learning (MARL). However, existing CTDE methods often underutilize centralized training or lack theoretical guarantees. We propose Multi-Agent Guided Policy Optimization (MAGPO), a novel framework that better leverages centralized training by integrating centralized guidance with decentralized execution. MAGPO uses an autoregressive joint policy for scalable, coordinated exploration and explicitly aligns it with decentralized policies to ensure deployability under partial observability. We provide theoretical guarantees of monotonic policy improvement and empirically evaluate MAGPO on 43 tasks across 6 diverse environments. Results show that MAGPO consistently outperforms strong CTDE baselines and matches or surpasses fully centralized approaches, offering a principled and practical solution for decentralized multi-agent learning. Our code and experimental data can be found in this https URL.

**arXiv ID:** 2507.18059
</details>

<details>
<summary><strong>Context Engineering: From Prompts to Corporate Multi-Agent Architecture</strong> - Vera V. Vishnyakova - [[pdf]](https://arxiv.org/pdf/2603.09619)</summary>

**Abstract:** As artificial intelligence (AI) systems evolve from stateless chatbots to autonomous multi-step agents, prompt engineering (PE), the discipline of crafting individual queries, proves necessary but insufficient. This paper introduces context engineering (CE) as a standalone discipline concerned with designing, structuring, and managing the entire informational environment in which an AI agent makes decisions. Drawing on vendor architectures (Google ADK, Anthropic, LangChain), current academic work (ACE framework, Google DeepMind's intelligent delegation), enterprise research (Deloitte, 2026; KPMG, 2026), and the author's experience building a multi-agent system, the paper proposes five context quality criteria: relevance, sufficiency, isolation, economy, and provenance, and frames context as the agent's operating system. Two higher-order disciplines follow. Intent engineering (IE) encodes organizational goals, values, and trade-off hierarchies into agent infrastructure. Specification engineering (SE) creates a machine-readable corpus of corporate policies and standards enabling autonomous operation of multi-agent systems at scale. Together these four disciplines form a cumulative pyramid maturity model of agent engineering, in which each level subsumes the previous one as a necessary foundation. Enterprise data reveals a gap: while 75% of enterprises plan agentic AI deployment within two years (Deloitte, 2026), deployment has surged and retreated as organizations confront scaling complexity (KPMG, 2026). The Klarna case illustrates a dual deficit, contextual and intentional. Whoever controls the agent's context controls its behavior; whoever controls its intent controls its strategy; whoever controls its specifications controls its scale.

**arXiv ID:** 2603.09619
</details>

<details>
<summary><strong>COMPASS: The explainable agentic framework for Sovereignty, Sustainability, Compliance, and Ethics</strong> - Jean-Sébastien Dessureault, Alain-Thierry Iliho Manzi, Soukaina Alaoui Ismaili, Khadim Lo, Mireille Lalancette, Éric Bélanger - [[pdf]](https://arxiv.org/pdf/2603.11277)</summary>

**Abstract:** The rapid proliferation of large language model (LLM)-based agentic systems raises critical concerns regarding digital sovereignty, environmental sustainability, regulatory compliance, and ethical alignment. Whilst existing frameworks address individual dimensions in isolation, no unified architecture systematically integrates these imperatives into the decision-making processes of autonomous agents. This paper introduces the COMPASS (Compliance and Orchestration for Multi-dimensional Principles in Autonomous Systems with Sovereignty) Framework, a novel multi-agent orchestration system designed to enforce value-aligned AI through modular, extensible governance mechanisms. The framework comprises an Orchestrator and four specialised sub-agents addressing sovereignty, carbon-aware computing, compliance, and ethics, each augmented with Retrieval-Augmented Generation (RAG) to ground evaluations in verified, context-specific documents. By employing an LLM-as-a-judge methodology, the system assigns quantitative scores and generates explainable justifications for each assessment dimension, enabling real-time arbitration of conflicting objectives. We validate the architecture through automated evaluation, demonstrating that RAG integration significantly enhances semantic coherence and mitigates the hallucination risks. Our results indicate that the framework's composition-based design facilitates seamless integration into diverse application domains whilst preserving interpretability and traceability.

**arXiv ID:** 2603.11277
</details>

<details>
<summary><strong>Aligning Large Language Model Agents with Rational and Moral Preferences: A Supervised Fine-Tuning Approach</strong> - Wei Lu, Amit Dhanda, Daniel L. Chen, Christian B. Hansen - [[pdf]](https://arxiv.org/pdf/2507.20796)</summary>

**Abstract:** As large language models (LLMs) increasingly act as autonomous agents in markets and organizations, their behavior in strategic environments becomes economically consequential. We document that off-the-shelf LLM agents exhibit systematic deviations from payoff-sensitive behavior in canonical economic games, including excessive cooperation and limited responsiveness to incentives. We introduce a supervised fine-tuning approach that aligns agent behavior with explicit economic preferences. Specifically, we generate optimal strategies under two stylized utility specifications, homo economicus, which maximizes self-interest, and homo moralis, which incorporates Kantian universalizability, and use these utility-implied reasoning and strategies to guide fine-tuning. Fine-tuning on a small, theory-driven synthetic dataset induces persistent and interpretable shifts in strategic behavior. In applications to moral dilemmas and repeated duopoly pricing, agents aligned to different preference structures produce systematically distinct equilibrium outcomes and pricing dynamics. These results frame AI alignment in multi-agent settings as an objective-design problem and illustrate how economic theory can guide the design of strategically coherent AI agents.

**arXiv ID:** 2507.20796
</details>

<details>
<summary><strong>Beyond Static Instruction: A Multi-agent AI Framework for Adaptive Augmented Reality Robot Training</strong> - Nicolas Leins, Jana Gonnermann-Müller, Malte Teichmann, Sebastian Pokutta - [[pdf]](https://arxiv.org/pdf/2603.00016)</summary>

**Abstract:** Augmented Reality (AR) offers powerful visualization capabilities for industrial robot training, yet current interfaces remain predominantly static, failing to account for learners' diverse cognitive profiles. In this paper, we present an AR application for robot training and propose a multi-agent AI framework for future integration that bridges the gap between static visualization and pedagogical intelligence. We report on the evaluation of the baseline AR interface with 36 participants performing a robotic pick-and-place task. While overall usability was high, notable disparities in task duration and learner characteristics highlighted the necessity for dynamic adaptation. To address this, we propose a multi-agent framework that orchestrates multiple components to perform complex preprocessing of multimodal inputs (e.g., voice, physiology, robot data) and adapt the AR application to the learner's needs. By utilizing autonomous Large Language Model (LLM) agents, the proposed system would dynamically adapt the learning environment based on advanced LLM reasoning in real-time.

**arXiv ID:** 2603.00016
</details>

<details>
<summary><strong>DIALECTIC: A Multi-Agent System for Startup Evaluation</strong> - Jae Yoon Bae, Simon Malberg, Joyce Galang, Andre Retterath, Georg Groh - [[pdf]](https://arxiv.org/pdf/2603.12274)</summary>

**Abstract:** Venture capital (VC) investors face a large number of investment opportunities but only invest in few of these, with even fewer ending up successful. Early-stage screening of opportunities is often limited by investor bandwidth, demanding tradeoffs between evaluation diligence and number of opportunities assessed. To ease this tradeoff, we introduce DIALECTIC, an LLM-based multi-agent system for startup evaluation. DIALECTIC first gathers factual knowledge about a startup and organizes these facts into a hierarchical question tree. It then synthesizes the facts into natural-language arguments for and against an investment and iteratively critiques and refines these arguments through a simulated debate, which surfaces only the most convincing arguments. Our system also produces numeric decision scores that allow investors to rank and thus efficiently prioritize opportunities. We evaluate DIALECTIC through backtesting on real investment opportunities aggregated from five VC funds, showing that DIALECTIC matches the precision of human VCs in predicting startup success.

**arXiv ID:** 2603.12274
</details>

<details>
<summary><strong>Collaborative Multi-Agent Optimization for Personalized Memory System</strong> - Wenyu Mao, Haoyang Liu, Zhao Liu, Haosong Tan, Yaorui Shi, Jiancan Wu, An Zhang, Xiang Wang - [[pdf]](https://arxiv.org/pdf/2603.12631)</summary>

**Abstract:** Memory systems are crucial to personalized LLMs by mitigating the context window limitation in capturing long-term user-LLM conversations. Typically, such systems leverage multiple agents to handle multi-granular memory construction and personalized memory retrieval tasks. To optimize the system, existing methods focus on specializing agents on their local tasks independently via prompt engineering or fine-tuning. However, they overlook cross-agent collaboration, where independent optimization on local agents hardly guarantees the global system performance. To address this issue, we propose a Collaborative Reinforcement Learning Framework for Multi-Agent Memory Systems (CoMAM), jointly optimizing local agents to facilitate collaboration. Specifically, we regularize agents' execution as a sequential Markov decision process (MDP) to embed inter-agent dependencies into the state transition, yielding both local task rewards (e.g., information coverage for memory construction) and global rewards (i.e., query-answer accuracy). Then, we quantify each agent's contribution via group-level ranking consistency between local and global rewards, treating them as adaptive weights to assign global credit and integrate local-global rewards. Each agent is optimized by these integrated rewards, aligning local improvements with the global performance. Experiments show CoMAM outperforms leading memory systems, validating the efficacy of our proposed collaborative reinforcement learning for joint optimization.

**arXiv ID:** 2603.12631
</details>

<details>
<summary><strong>Conflict Mitigation in Shared Environments using Flow-Aware Multi-Agent Path Finding</strong> - Lukas Heuer, Yufei Zhu, Luigi Palmieri, Andrey Rudenko, Anna Mannucci, Sven Koenig, Martin Magnusson - [[pdf]](https://arxiv.org/pdf/2603.12736)</summary>

**Abstract:** Deploying multi-robot systems in environments shared with dynamic and uncontrollable agents presents significant challenges, especially for large robot fleets. In such environments, individual robot operations can be delayed due to unforeseen conflicts with uncontrollable agents. While existing research primarily focuses on preserving the completeness of Multi-Agent Path Finding (MAPF) solutions considering delays, there is limited emphasis on utilizing additional environmental information to enhance solution quality in the presence of other dynamic agents. To this end, we propose Flow-Aware Multi-Agent Path Finding (FA-MAPF), a novel framework that integrates learned motion patterns of uncontrollable agents into centralized MAPF algorithms. Our evaluation, conducted on a diverse set of benchmark maps with simulated uncontrollable agents and on a real-world map with recorded human trajectories, demonstrates the effectiveness of FA-MAPF compared to state-of-the-art baselines. The experimental results show that FA-MAPF can consistently reduce conflicts with uncontrollable agents, up to 55%, without compromising task efficiency.

**arXiv ID:** 2603.12736
</details>

<details>
<summary><strong>Partially Observable Multi-Agent Reinforcement Learning with Information Sharing</strong> - Xiangyu Liu, Kaiqing Zhang - [[pdf]](https://arxiv.org/pdf/2308.08705)</summary>

**Abstract:** We study provable multi-agent reinforcement learning (RL) in the general framework of partially observable stochastic games (POSGs). To circumvent the known hardness results and the use of computationally intractable oracles, we advocate leveraging the potential \emph{information-sharing} among agents, a common practice in empirical multi-agent RL, and a standard model for multi-agent control systems with communication. We first establish several computational complexity results to justify the necessity of information-sharing, as well as the observability assumption that has enabled quasi-polynomial time and sample single-agent RL with partial observations, for tractably solving POSGs. Inspired by the inefficiency of planning in the ground-truth model, we then propose to further \emph{approximate} the shared common information to construct an approximate model of the POSG, in which an approximate \emph{equilibrium} (of the original POSG) can be found in quasi-polynomial-time, under the aforementioned assumptions. Furthermore, we develop a partially observable multi-agent RL algorithm whose time and sample complexities are \emph{both} quasi-polynomial. Finally, beyond equilibrium learning, we extend our algorithmic framework to finding the \emph{team-optimal solution} in cooperative POSGs, i.e., decentralized partially observable Markov decision processes, a more challenging goal. We establish concrete computational and sample complexities under several structural assumptions of the model. We hope our study could open up the possibilities of leveraging and even designing different \emph{information structures}, a well-studied notion in control theory, for developing both sample- and computation-efficient partially observable multi-agent RL.

**arXiv ID:** 2308.08705
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>Active Causal Structure Learning with Latent Variables: Towards Learning to Detour in Autonomous Robots</strong> - Pablo de los Riscos, Fernando J. Corbacho - [[pdf]](https://arxiv.org/pdf/2410.20894)</summary>

**Abstract:** Artificial General Intelligence (AGI) Agents and Robots must be able to cope with everchanging environments and tasks. They must be able to actively construct new internal causal models of their interactions with the environment when new structural changes take place in the environment. Thus, we claim that active causal structure learning with latent variables (ACSLWL) is a necessary component to build AGI agents and robots. This paper describes how a complex planning and expectation-based detour behavior can be learned by ACSLWL when, unexpectedly, and for the first time, the simulated robot encounters a sort of transparent barrier in its pathway towards its target. ACSWL consists of acting in the environment, discovering new causal relations, constructing new causal models, exploiting the causal models to maximize its expected utility, detecting possible latent variables when unexpected observations occur, and constructing new structures-internal causal models and optimal estimation of the associated parameters, to be able to cope efficiently with the new encountered situations. That is, the agent must be able to construct new causal internal models that transform a previously unexpected and inefficient (sub-optimal) situation, into a predictable situation with an optimal operating plan.

**arXiv ID:** 2410.20894
</details>

<details>
<summary><strong>Building Effective AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned</strong> - Nghi D. Q. Bui - [[pdf]](https://arxiv.org/pdf/2603.05344)</summary>

**Abstract:** The landscape of AI coding assistance is undergoing a fundamental shift from complex IDE plugins to versatile, terminal-native agents. Operating directly where developers manage source control, execute builds, and deploy environments, CLI-based agents offer unprecedented autonomy for long-horizon development tasks. In this paper, we present OPENDEV, an open-source, command-line coding agent written in Rust, engineered specifically for this new paradigm. Effective autonomous assistance requires strict safety controls and highly efficient context management to prevent context bloat and reasoning degradation. OPENDEV overcomes these challenges through a compound AI system architecture with workload-specialized model routing, a dual-agent architecture separating planning from execution, lazy tool discovery, and adaptive context compaction that progressively reduces older observations. Furthermore, it employs an automated memory system to accumulate project-specific knowledge across sessions and counteracts instruction fade-out through event-driven system reminders. By enforcing explicit reasoning phases and prioritizing context efficiency, OPENDEV provides a secure, extensible foundation for terminal-first AI assistance, offering a blueprint for robust autonomous software engineering.

**arXiv ID:** 2603.05344
</details>

<details>
<summary><strong>Measuring AI Agents' Progress on Multi-Step Cyber Attack Scenarios</strong> - Linus Folkerts, Will Payne, Simon Inman, Philippos Giavridis, Joe Skinner, Sam Deverett, James Aung, Ekin Zorer, Michael Schmatz, Mahmoud Ghanem, John Wilkinson, Alan Steer, Vy Hong, Jessica Wang - [[pdf]](https://arxiv.org/pdf/2603.11214)</summary>

**Abstract:** We evaluate the autonomous cyber-attack capabilities of frontier AI models on two purpose-built cyber ranges-a 32-step corporate network attack and a 7-step industrial control system attack-that require chaining heterogeneous capabilities across extended action sequences. By comparing seven models released over an eighteen-month period (August 2024 to February 2026) at varying inference-time compute budgets, we observe two capability trends. First, model performance scales log-linearly with inference-time compute, with no observed plateau-increasing from 10M to 100M tokens yields gains of up to 59%, requiring no specific technical sophistication from the operator. Second, each successive model generation outperforms its predecessor at fixed token budgets: on the corporate network range, average steps completed at 10M tokens rose from 1.7 (GPT-4o, August 2024) to 9.8 (Opus 4.6, February 2026). The best single run completed 22 of 32 steps, corresponding to roughly 6 of the estimated 14 hours a human expert would need. On the industrial control system range, performance remains limited, though the most recent models are the first to reliably complete steps, averaging 1.2-1.4 of 7 (max 3).

**arXiv ID:** 2603.11214
</details>

<details>
<summary><strong>A Tutorial on Cognitive Biases in Agentic AI-Driven 6G Autonomous Networks</strong> - Hatim Chergui, Farhad Rezazadeh, Merouane Debbah, Christos Verikoukis - [[pdf]](https://arxiv.org/pdf/2510.19973)</summary>

**Abstract:** The path to higher network autonomy in 6G lies beyond the mere optimization of key performance indicators (KPIs). While KPIs have enabled automation gains under TM Forum Levels 1--3, they remain numerical abstractions that act only as proxies for the real essence of communication networks: seamless connectivity, fairness, adaptability, and resilience. True autonomy requires perceiving and reasoning over the network environment as it is. Such progress can be achieved through \emph{agentic AI}, where large language model (LLM)-powered agents perceive multimodal telemetry, reason with memory, negotiate across domains, and act via APIs to achieve multi-objective goals. However, deploying such agents introduces the challenge of cognitive biases inherited from human design, which can distort reasoning, negotiation, tool use, and actuation. Between neuroscience and AI, this paper provides a tutorial on a selection of well-known biases, including their taxonomy, definition, mathematical formulation, emergence in telecom systems and the commonly impacted agentic components. The tutorial also presents various mitigation strategies tailored to each type of bias. The article finally provides two practical use-cases, which tackle the emergence, impact and mitigation gain of some famous biases in 6G inter-slice and cross-domain management. In particular, anchor randomization, temporal decay and inflection bonus techniques are introduced to specifically address anchoring, temporal and confirmation biases. This avoids that agents stick to the initial high resource allocation proposal or decisions that are recent and/or confirming a prior hypothesis. By grounding decisions in a richer and fairer set of past experiences, the quality and bravery of the agentic agreements in the second use-case, for instance, are leading to $\times 5$ lower latency and around $40\%$ higher energy saving.

**arXiv ID:** 2510.19973
</details>

<details>
<summary><strong>One Supervisor, Many Modalities: Adaptive Tool Orchestration for Autonomous Queries</strong> - Mayank Saini, Arit Kumar Bishwas - [[pdf]](https://arxiv.org/pdf/2603.11545)</summary>

**Abstract:** We present an agentic AI framework for autonomous multimodal query processing that coordinates specialized tools across text, image, audio, video, and document modalities. A central Supervisor dynamically decomposes user queries, delegates subtasks to modality-appropriate tools (e.g., object detection, OCR, speech transcription), and synthesizes results through adaptive routing strategies rather than predetermined decision trees. For text-only queries, the framework uses learned routing via RouteLLM, while non-text paths use SLM-assisted modality decomposition. Evaluated on 2,847 queries across 15 task categories, our framework achieves 72% reduction in time-to-accurate-answer, 85% reduction in conversational rework, and 67% cost reduction compared to the matched hierarchical baseline while maintaining accuracy parity. These results demonstrate that intelligent centralized orchestration fundamentally improves multimodal AI deployment economics.

**arXiv ID:** 2603.11545
</details>

<details>
<summary><strong>Human-AI Collaborative Autonomous Experimentation With Proxy Modeling for Comparative Observation</strong> - Arpan Biswas, Hiroshi Funakubo, Yongtao Liu - [[pdf]](https://arxiv.org/pdf/2603.12618)</summary>

**Abstract:** Optimization for different tasks like material characterization, synthesis, and functional properties for desired applications over multi-dimensional control parameters need a rapid strategic search through active learning such as Bayesian optimization (BO). However, such high-dimensional experimental physical descriptors are complex and noisy, from which realization of a low-dimensional mathematical scalar metrics or objective functions can be erroneous. Moreover, in traditional purely data-driven autonomous exploration, such objective functions often ignore the subtle variation and key features of the physical descriptors, thereby can fail to discover unknown phenomenon of the material systems. To address this, here we present a proxy-modelled Bayesian optimization (px-BO) via on-the-fly teaming between human and AI agents. Over the loop of BO, instead of defining a mathematical objective function directly from the experimental data, we introduce a voting system on the fly where the new experimental outcome will be compared with existing experiments, and the human agents will choose the preferred samples. These human-guided comparisons are then transformed into a proxy-based objective function via fitting Bradley-Terry (BT) model. Then, to minimize human interaction, this iteratively trained proxy model also acts as an AI agent for future surrogate human votes. Finally, these surrogate votes are periodically validated by human agents, and the corrections are then learned by the proxy model on-the-fly. We demonstrated the performance of the proposed px-BO framework into simulated and BEPS data generated from PTO sample. We find that our approach provided better control of the domain experts for an improved search over traditional data-driven exploration, thus, signifies the importance of human-AI teaming in an accelerated and meaningful material space exploration.

**arXiv ID:** 2603.12618
</details>

<details>
<summary><strong>From Woofs to Words: Towards Intelligent Robotic Guide Dogs with Verbal Communication</strong> - Yohei Hayamizu, David DeFazio, Hrudayangam Mehta, Zainab Altaweel, Jacqueline Choe, Chao Lin, Jake Juettner, Furui Xiao, Jeremy Blackburn, Shiqi Zhang - [[pdf]](https://arxiv.org/pdf/2603.12574)</summary>

**Abstract:** Assistive robotics is an important subarea of robotics that focuses on the well-being of people with disabilities. A robotic guide dog is an assistive quadruped robot that helps visually impaired people in obstacle avoidance and navigation. Enabling language capabilities for robotic guide dogs goes beyond naively adding an existing dialog system onto a mobile robot. The novel challenges include grounding language in the dynamically changing environment and improving spatial awareness for the human handler. To address those challenges, we develop a novel dialog system for robotic guide dogs that uses LLMs to verbalize both navigational plans and scenes. The goal is to enable verbal communication for collaborative decision-making within the handler-robot team. In experiments, we conducted a human study to evaluate different verbalization strategies and a simulation study to assess the efficiency and accuracy in navigation tasks.

**arXiv ID:** 2603.12574
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (21 papers)</h2></summary>

<details>
<summary><strong>Structured Distillation for Personalized Agent Memory: 11x Token Reduction with Retrieval Preservation</strong> - Sydney Lewis - [[pdf]](https://arxiv.org/pdf/2603.13017)</summary>

**Abstract:** Long conversations with an AI agent create a simple problem for one user: the history is useful, but carrying it verbatim is expensive. We study personalized agent memory: one user's conversation history with an agent, distilled into a compact retrieval layer for later search. Each exchange is compressed into a compound object with four fields (exchange_core, specific_context, thematic room_assignments, and regex-extracted files_touched). The searchable distilled text averages 38 tokens per exchange. Applied to 4,182 conversations (14,340 exchanges) from 6 software engineering projects, the method reduces average exchange length from 371 to 38 tokens, yielding 11x compression. We evaluate whether personalized recall survives that compression using 201 recall-oriented queries, 107 configurations spanning 5 pure and 5 cross-layer search modes, and 5 LLM graders (214,519 consensus-graded query-result pairs). The best pure distilled configuration reaches 96% of the best verbatim MRR (0.717 vs 0.745). Results are mechanism-dependent. All 20 vector search configurations remain non-significant after Bonferroni correction, while all 20 BM25 configurations degrade significantly (effect sizes |d|=0.031-0.756). The best cross-layer setup slightly exceeds the best pure verbatim baseline (MRR 0.759). Structured distillation compresses single-user agent memory without uniformly sacrificing retrieval quality. At 1/11 the context cost, thousands of exchanges fit within a single prompt while the verbatim source remains available for drill-down. We release the implementation and analysis pipeline as open-source software.

**arXiv ID:** 2603.13017
</details>

<details>
<summary><strong>Thermodynamics of Reinforcement Learning Curricula</strong> - Jacob Adamczyk, Juan Sebastian Rojas, Rahul V. Kulkarni - [[pdf]](https://arxiv.org/pdf/2603.12324)</summary>

**Abstract:** Connections between statistical mechanics and machine learning have repeatedly proven fruitful, providing insight into optimization, generalization, and representation learning. In this work, we follow this tradition by leveraging results from non-equilibrium thermodynamics to formalize curriculum learning in reinforcement learning (RL). In particular, we propose a geometric framework for RL by interpreting reward parameters as coordinates on a task manifold. We show that, by minimizing the excess thermodynamic work, optimal curricula correspond to geodesics in this task space. As an application of this framework, we provide an algorithm, "MEW" (Minimum Excess Work), to derive a principled schedule for temperature annealing in maximum-entropy RL.

**arXiv ID:** 2603.12324
</details>

<details>
<summary><strong>CALF: Communication-Aware Learning Framework for Distributed Reinforcement Learning</strong> - Carlos Purves, Pietro Lio' - [[pdf]](https://arxiv.org/pdf/2603.12543)</summary>

**Abstract:** Distributed reinforcement learning policies face network delays, jitter, and packet loss when deployed across edge devices and cloud servers. Standard RL training assumes zero-latency interaction, causing severe performance degradation under realistic network conditions. We introduce CALF (Communication-Aware Learning Framework), which trains policies under realistic network models during simulation. Systematic experiments demonstrate that network-aware training substantially reduces deployment performance gaps compared to network-agnostic baselines. Distributed policy deployments across heterogeneous hardware validate that explicitly modelling communication constraints during training enables robust real-world execution. These findings establish network conditions as a major axis of sim-to-real transfer for Wi-Fi-like distributed deployments, complementing physics and visual domain randomisation.

**arXiv ID:** 2603.12543
</details>

<details>
<summary><strong>Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages</strong> - Vishnu Teja Kunde, Fatemeh Doudi, Mahdi Farahbakhsh, Dileep Kalathil, Krishna Narayanan, Jean-Francois Chamberland - [[pdf]](https://arxiv.org/pdf/2603.12554)</summary>

**Abstract:** Reinforcement learning (RL) has been effective for post-training autoregressive (AR) language models, but extending these methods to diffusion language models (DLMs) is challenging due to intractable sequence-level likelihoods. Existing approaches therefore rely on surrogate likelihoods or heuristic approximations, which can introduce bias and obscure the sequential structure of denoising. We formulate diffusion-based sequence generation as a finite-horizon Markov decision process over the denoising trajectory and derive an exact, unbiased policy gradient that decomposes over denoising steps and is expressed in terms of intermediate advantages, without requiring explicit evaluation of the sequence likelihood. To obtain a practical and compute-efficient estimator, we (i) select denoising steps for policy updates via an entropy-guided approximation bound, and (ii) estimate intermediate advantages using a one-step denoising reward naturally provided by the diffusion model, avoiding costly multi-step rollouts. Experiments on coding and logical reasoning benchmarks demonstrate state-of-the-art results, with strong competitive performance on mathematical reasoning, outperforming existing RL post-training approaches for DLMs. Code is available at this https URL.

**arXiv ID:** 2603.12554
</details>

<details>
<summary><strong>Swap-guided Preference Learning for Personalized Reinforcement Learning from Human Feedback</strong> - Gihoon Kim, Euntai Kim - [[pdf]](https://arxiv.org/pdf/2603.12595)</summary>

**Abstract:** Reinforcement Learning from Human Feedback (RLHF) is a widely used approach to align large-scale AI systems with human values. However, RLHF typically assumes a single, universal reward, which overlooks diverse preferences and limits personalization. Variational Preference Learning (VPL) seeks to address this by introducing user-specific latent variables. Despite its promise, we found that VPL suffers from posterior collapse. While this phenomenon is well known in VAEs, it has not previously been identified in preference learning frameworks. Under sparse preference data and with overly expressive decoders, VPL may cause latent variables to be ignored, reverting to a single-reward model. To overcome this limitation, we propose Swap-guided Preference Learning (SPL). The key idea is to construct fictitious swap annotators and use the mirroring property of their preferences to guide the encoder. SPL introduces three components: (1) swap-guided base regularization, (2) Preferential Inverse Autoregressive Flow (P-IAF), and (3) adaptive latent conditioning. Experiments show that SPL mitigates collapse, enriches user-specific latents, and improves preference prediction. Our code and data are available at this https URL

**arXiv ID:** 2603.12595
</details>

<details>
<summary><strong>Efficient Real-World Autonomous Racing via Attenuated Residual Policy Optimization</strong> - Raphael Trumpp, Denis Hoornaert, Mirco Theile, Marco Caccamo - [[pdf]](https://arxiv.org/pdf/2603.12960)</summary>

**Abstract:** Residual policy learning (RPL), in which a learned policy refines a static base policy using deep reinforcement learning (DRL), has shown strong performance across various robotic applications. Its effectiveness is particularly evident in autonomous racing, a domain that serves as a challenging benchmark for real-world DRL. However, deploying RPL-based controllers introduces system complexity and increases inference latency. We address this by introducing an extension of RPL named attenuated residual policy optimization ($\alpha$-RPO). Unlike standard RPL, $\alpha$-RPO yields a standalone neural policy by progressively attenuating the base policy, which initially serves to bootstrap learning. Furthermore, this mechanism enables a form of privileged learning, where the base policy is permitted to use sensor modalities not required for final deployment. We design $\alpha$-RPO to integrate seamlessly with PPO, ensuring that the attenuated influence of the base controller is dynamically compensated during policy optimization. We evaluate $\alpha$-RPO by building a framework for 1:10-scaled autonomous racing around it. In both simulation and zero-shot real-world transfer to Roboracer cars, $\alpha$-RPO not only reduces system complexity but also improves driving performance compared to baselines - demonstrating its practicality for robotic deployment. Our code is available at: this https URL.

**arXiv ID:** 2603.12960
</details>

<details>
<summary><strong>ARL-Tangram: Unleash the Resource Efficiency in Agentic Reinforcement Learning</strong> - Bangjun Xiao, Yihao Zhao, Xiangwei Deng, Shihua Yu, Yuxing Xiang, Huaqiu Liu, Qiying Wang, Liang Zhao, Hailin Zhang, Xuanzhe Liu, Xin Jin, Fuli Luo - [[pdf]](https://arxiv.org/pdf/2603.13019)</summary>

**Abstract:** Agentic reinforcement learning (RL) has emerged as a transformative workload in cloud clusters, enabling large language models (LLMs) to solve complex problems through interactions with real world. However, unlike traditional RL, agentic RL demands substantial external cloud resources, e.g., CPUs for code execution and GPUs for reward models, that exist outside the primary training cluster. Existing agentic RL framework typically rely on static over-provisioning, i.e., resources are often tied to long-lived trajectories or isolated by tasks, which leads to severe resource inefficiency.
We propose the action-level orchestration, and incorporate it into ARL-Tangram, a unified resource management system that enables fine-grained external resource sharing and elasticity. ARL-Tangram utilizes a unified action-level formulation and an elastic scheduling algorithm to minimize action completion time (ACT) while satisfying heterogeneous resource constraints. Further, heterogeneous resource managers are tailored to efficiently support the action-level execution on resources with heterogeneous characteristics and topologies. Evaluation on real-world agentic RL tasks demonstrates that ARL-Tangram improves average ACT by up to 4.3$\times$, speeds up the step duration of RL training by up to 1.5$\times$, and saves the external resources by up to 71.2$\%$. This system has been deployed to support the training of the MiMo series models.

**arXiv ID:** 2603.13019
</details>

<details>
<summary><strong>OpenSage: Self-programming Agent Generation Engine</strong> - Hongwei Li, Zhun Wang, Qinrun Dai, Yuzhou Nie, Jinjun Peng, Ruitong Liu, Jingyang Zhang, Kaijie Zhu, Jingxuan He, Lun Wang, Yangruibo Ding, Yueqi Chen, Wenbo Guo, Dawn Song - [[pdf]](https://arxiv.org/pdf/2602.16891)</summary>

**Abstract:** Agent development kits (ADKs) provide effective platforms and tooling for constructing agents, and their designs are critical to the constructed agents' performance, especially the functionality for agent topology, tools, and memory. However, current ADKs either lack sufficient functional support or rely on humans to manually design these components, limiting agents' generalizability and overall performance. We propose OpenSage, the first ADK that enables LLMs to automatically create agents with self-generated topology and toolsets while providing comprehensive and structured memory support. OpenSage offers effective functionality for agents to create and manage their own sub-agents and toolkits. It also features a hierarchical, graph-based memory system for efficient management and a specialized toolkit tailored to software engineering tasks. Extensive experiments across three state-of-the-art benchmarks with various backbone models demonstrate the advantages of OpenSage over existing ADKs. We also conduct rigorous ablation studies to demonstrate the effectiveness of our design for each component. We believe OpenSage can pave the way for the next generation of agent development, shifting the focus from human-centered to AI-centered paradigms.

**arXiv ID:** 2602.16891
</details>

<details>
<summary><strong>DriveMind: A Dual Visual Language Model-based Reinforcement Learning Framework for Autonomous Driving</strong> - Dawood Wasif, Terrence J. Moore, Chandan K. Reddy, Frederica Free-Nelson, Seunghyun Yoon, Hyuk Lim, Dan Dongseong Kim, Jin-Hee Cho - [[pdf]](https://arxiv.org/pdf/2506.00819)</summary>

**Abstract:** End-to-end autonomous driving systems map sensor data directly to control commands, but remain opaque, lack interpretability, and offer no formal safety guarantees. While recent vision-language-guided reinforcement learning (RL) methods introduce semantic feedback, they often rely on static prompts and fixed objectives, limiting adaptability to dynamic driving scenes. We present DriveMind, a unified semantic reward framework that integrates: (i) a contrastive Vision-Language Model (VLM) encoder for stepwise semantic anchoring; (ii) a novelty-triggered VLM encoder-decoder, fine-tuned via chain-of-thought (CoT) distillation, for dynamic prompt generation upon semantic drift; (iii) a hierarchical safety module enforcing kinematic constraints (e.g., speed, lane centering, stability); and (iv) a compact predictive world model to reward alignment with anticipated ideal states. DriveMind achieves 19.4 +/- 2.3 km/h average speed, 0.98 +/- 0.03 route completion, and near-zero collisions in CARLA Town 2, outperforming baselines by over 4% in success rate. Its semantic reward generalizes zero-shot to real dash-cam data with minimal distributional shift, demonstrating robust cross-domain alignment and potential for real-world deployment.

**arXiv ID:** 2506.00819
</details>

<details>
<summary><strong>Accelerating Residual Reinforcement Learning with Uncertainty Estimation</strong> - Lakshita Dodeja, Karl Schmeckpeper, Shivam Vats, Thomas Weng, Mingxi Jia, George Konidaris, Stefanie Tellex - [[pdf]](https://arxiv.org/pdf/2506.17564)</summary>

**Abstract:** Residual Reinforcement Learning (RL) is a popular approach for adapting pretrained policies by learning a lightweight residual policy that provides corrective actions. While Residual RL is more sample-efficient than finetuning the entire base policy, existing methods struggle with sparse rewards and are designed for deterministic base policies. We propose two improvements to Residual RL that further enhance its sample efficiency and make it suitable for stochastic base policies. First, we leverage uncertainty estimates of the base policy to focus exploration on regions in which the base policy is not confident. Second, we propose a simple modification to off-policy residual learning that allows it to observe base actions and better handle stochastic base policies. We evaluate our method with both Gaussian-based and Diffusion-based stochastic base policies on tasks from Robosuite and D4RL, and compare against state-of-the-art finetuning methods, demo-augmented RL methods, and other residual RL methods. Our algorithm significantly outperforms existing baselines in a variety of simulation benchmark environments. We also deploy our learned polices in the real world to demonstrate their robustness with zero-shot sim-to-real transfer. Paper homepage : this http URL

**arXiv ID:** 2506.17564
</details>

<details>
<summary><strong>SegDAC: Visual Generalization in Reinforcement Learning via Dynamic Object Tokens</strong> - Alexandre Brown, Glen Berseth - [[pdf]](https://arxiv.org/pdf/2508.09325)</summary>

**Abstract:** Visual reinforcement learning policies trained on pixel observations often struggle to generalize when visual conditions change at test time. Object-centric representations are a promising alternative, but most approaches use fixed-size slot representations, require image reconstruction, or need auxiliary losses to learn object decompositions. As a result, it remains unclear how to learn RL policies directly from object-level inputs without these constraints. We propose SegDAC, a Segmentation-Driven Actor-Critic that operates on a variable-length set of object token embeddings. At each timestep, text-grounded segmentation produces object masks from which spatially aware token embeddings are extracted. A transformer-based actor-critic processes these dynamic tokens, using segment positional encoding to preserve spatial information across objects. We ablate these design choices and show that both segment positional encoding and variable-length processing are individually necessary for strong performance. We evaluate SegDAC on 8 ManiSkill3 manipulation tasks under 12 visual perturbation types across 3 difficulty levels. SegDAC improves over prior visual generalization methods by 15% on easy, 66% on medium, and 88% on the hardest settings. SegDAC matches the sample efficiency of the state-of-the-art visual RL methods while achieving improved generalization under visual changes. Project Page: this https URL

**arXiv ID:** 2508.09325
</details>

<details>
<summary><strong>CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions</strong> - Lizhi Yang, Blake Werner, Massimiliano de Sa, Aaron D. Ames - [[pdf]](https://arxiv.org/pdf/2510.14959)</summary>

**Abstract:** Reinforcement learning (RL), while powerful and expressive, can often prioritize performance at the expense of safety. Yet safety violations can lead to catastrophic outcomes in real-world deployments. Control Barrier Functions (CBFs) offer a principled method to enforce dynamic safety -- traditionally deployed online via safety filters. While the result is safe behavior, the fact that the RL policy does not have knowledge of the CBF can lead to conservative behaviors. This paper proposes CBF-RL, a framework for generating safe behaviors with RL by enforcing CBFs in training. CBF-RL has two key attributes: (1) minimally modifying a nominal RL policy to encode safety constraints via a CBF term, (2) and safety filtering of the policy rollouts in training. Theoretically, we prove that continuous-time safety filters can be deployed via closed-form expressions on discrete-time roll-outs. Practically, we demonstrate that CBF-RL internalizes the safety constraints in the learned policy -- both enforcing safer actions and biasing towards safer rewards -- enabling safe deployment without the need for an online safety filter. We validate our framework through ablation studies on navigation tasks and on the Unitree G1 humanoid robot, where CBF-RL enables safer exploration, faster convergence, and robust performance under uncertainty, enabling the humanoid robot to avoid obstacles and climb stairs safely in real-world settings without a runtime safety filter.

**arXiv ID:** 2510.14959
</details>

<details>
<summary><strong>VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking-with-Videos</strong> - Wenqi Liu, Yunxiao Wang, Shijie Ma, Meng Liu, Qile Su, Tianke Zhang, Haonan Fan, Changyi Liu, Kaiyu Jiang, Jiankang Chen, Kaiyu Tang, Bin Wen, Fan Yang, Tingting Gao, Han Li, Yinwei Wei, Xuemeng Song - [[pdf]](https://arxiv.org/pdf/2602.07801)</summary>

**Abstract:** In long-video understanding, conventional uniform frame sampling often fails to capture key visual evidence, leading to degraded performance and increased hallucinations. To address this, recent agentic thinking-with-videos paradigms have emerged, adopting a localize-clip-answer pipeline in which the model actively identifies relevant video segments, performs dense sampling within those clips, and then produces answers. However, existing methods remain inefficient, suffer from weak localization, and adhere to rigid workflows. To solve these issues, we propose VideoTemp-o3, a unified agentic thinking-with-videos framework that jointly models video grounding and question answering. VideoTemp-o3 exhibits strong localization capability, supports on-demand clipping, and can refine inaccurate localizations. Specifically, in the supervised fine-tuning stage, we design a unified masking mechanism that encourages exploration while preventing noise. For reinforcement learning, we introduce dedicated rewards to mitigate reward hacking. Besides, from the data perspective, we develop an effective pipeline to construct high-quality long video grounded QA data, along with a corresponding benchmark for systematic evaluation across various video durations. Experimental results demonstrate that our method achieves remarkable performance on both long video understanding and grounding.

**arXiv ID:** 2602.07801
</details>

<details>
<summary><strong>EvolveCoder: Evolving Test Cases via Adversarial Verification for Code Reinforcement Learning</strong> - Chi Ruan, Dongfu Jiang, Huaye Zeng, Ping Nie, Wenhu Chen - [[pdf]](https://arxiv.org/pdf/2603.12698)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) is a promising approach for improving code generation in large language models, but its effectiveness is limited by weak and static verification signals in existing coding RL datasets. In this paper, we propose a solution-conditioned and adversarial verification framework that iteratively refines test cases based on the execution behaviors of candidate solutions, with the goal of increasing difficulty, improving discriminative power, and reducing redundancy. Based on this framework, we introduce EvolveCoder-22k, a large-scale coding reinforcement learning dataset constructed through multiple rounds of adversarial test case evolution. Empirical analysis shows that iterative refinement substantially strengthens verification, with pass@1 decreasing from 43.80 to 31.22. Reinforcement learning on EvolveCoder-22k yields stable optimization and consistent performance gains, improving Qwen3-4B by an average of 4.2 points across four downstream benchmarks and outperforming strong 4B-scale baselines. Our results highlight the importance of adversarial, solution-conditioned verification for effective and scalable reinforcement learning in code generation.

**arXiv ID:** 2603.12698
</details>

<details>
<summary><strong>Mending the Holes: Mitigating Reward Hacking in Reinforcement Learning for Multilingual Translation</strong> - Yifeng Liu, Siqi Ouyang, Yatish Hosmane Revanasiddappa, Lei Li - [[pdf]](https://arxiv.org/pdf/2603.13045)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated remarkable capability in machine translation on high-resource language pairs, yet their performance on low-resource translation still lags behind. Existing post-training methods rely heavily on high-quality parallel data, which are often scarce or unavailable for low-resource languages. In this paper, we introduce WALAR, a reinforcement training method using only monolingual text to elevate LLMs' translation capabilities on massive low-resource languages while retaining their performance on high-resource languages. Our key insight is based on the observation of failure modes (or "holes") in existing source-based multilingual quality estimation (QE) models. Reinforcement learning (RL) using these QE models tends to amplify such holes, resulting in poorer multilingual LLMs. We develop techniques including word alignment and language alignment to mitigate such holes in WALAR's reward for RL training. We continually trained an LLM supporting translation of 101 languages using WALAR. The experiments show that our new model outperforms LLaMAX, one of the strongest open-source multilingual LLMs by a large margin on 1400 language directions on Flores-101 dataset.

**arXiv ID:** 2603.13045
</details>

<details>
<summary><strong>SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models</strong> - Ziyi Yang, Weizhou Shen, Chenliang Li, Ruijun Chen, Fanqi Wan, Ming Yan, Xiaojun Quan, Fei Huang - [[pdf]](https://arxiv.org/pdf/2509.23863)</summary>

**Abstract:** Progress in long-context reasoning for large language models (LLMs) has lagged behind other recent advances. This gap arises not only from the intrinsic difficulty of processing long texts, but also from the scarcity of reliable human annotations and programmatically verifiable reward signals. In this paper, we propose SPELL, a multi-role self-play reinforcement learning framework that enables scalable, label-free optimization for long-context reasoning. SPELL integrates three cyclical roles-questioner, responder, and verifier-within a single model to enable continual self-improvement. The questioner generates questions from raw documents paired with reference answers; the responder learns to solve these questions based on the documents; and the verifier evaluates semantic equivalence between the responder's output and the questioner's reference answer, producing reward signals to guide continual training. To stabilize training, we introduce an automated curriculum that gradually increases document length and a reward function that adapts question difficulty to the model's evolving capabilities. Extensive experiments on six long-context benchmarks show that SPELL consistently improves performance across diverse LLMs and outperforms equally sized models fine-tuned on large-scale annotated data. Notably, SPELL achieves an average 7.6-point gain in pass@8 on the strong reasoning model Qwen3-30B-A3B-Thinking, raising its performance ceiling and showing promise for scaling to even more capable models. Our code is available at this https URL.

**arXiv ID:** 2509.23863
</details>

<details>
<summary><strong>Adaptive $Q$-Aid for Conditional Supervised Learning in Offline Reinforcement Learning</strong> - Jeonghye Kim, Suyoung Lee, Woojun Kim, Youngchul Sung - [[pdf]](https://arxiv.org/pdf/2402.02017)</summary>

**Abstract:** Offline reinforcement learning (RL) has progressed with return-conditioned supervised learning (RCSL), but its lack of stitching ability remains a limitation. We introduce $Q$-Aided Conditional Supervised Learning (QCS), which effectively combines the stability of RCSL with the stitching capability of $Q$-functions. By analyzing $Q$-function over-generalization, which impairs stable stitching, QCS adaptively integrates $Q$-aid into RCSL's loss function based on trajectory return. Empirical results show that QCS significantly outperforms RCSL and value-based methods, consistently achieving or exceeding the maximum trajectory returns across diverse offline RL benchmarks.

**arXiv ID:** 2402.02017
</details>

<details>
<summary><strong>Autonomous Integration and Improvement of Robotic Assembly using Skill Graph Representations</strong> - Peiqi Yu, Philip Huang, Chaitanya Chawla, Guanya Shi, Jiaoyang Li, Changliu Liu - [[pdf]](https://arxiv.org/pdf/2603.12649)</summary>

**Abstract:** Robotic assembly systems traditionally require substantial manual engineering effort to integrate new tasks, adapt to new environments, and improve performance over time. This paper presents a framework for autonomous integration and continuous improvement of robotic assembly systems based on Skill Graph representations. A Skill Graph organizes robot capabilities as verb-based skills, explicitly linking semantic descriptions (verbs and nouns) with executable policies, pre-conditions, post-conditions, and evaluators. We show how Skill Graphs enable rapid system integration by supporting semantic-level planning over skills, while simultaneously grounding execution through well-defined interfaces to robot controllers and perception modules. After initial deployment, the same Skill Graph structure supports systematic data collection and closed-loop performance improvement, enabling iterative refinement of skills and their composition. We demonstrate how this approach unifies system configuration, execution, evaluation, and learning within a single representation, providing a scalable pathway toward adaptive and reusable robotic assembly systems. The code is at this https URL.

**arXiv ID:** 2603.12649
</details>

<details>
<summary><strong>Reinforcement Learning for Elliptical Cylinder Motion Control Tasks</strong> - Pawel Marczewski, Paulina Superczynska, Jakub Bernat, Szymon Szczesny - [[pdf]](https://arxiv.org/pdf/2603.12807)</summary>

**Abstract:** The control of devices with limited input always bring attention to solve by research due to its difficulty and non-trival solution. For instance, the inverted pendulum is benchmarking problem in control theory and machine learning. In this work, we are focused on the elliptical cylinder and its motion under limited torque. The inspiration of the problem is from untethered magnetic devices, which due to distance have to operate with limited input torque. In this work, the main goal is to define the control problem of elliptic cylinder with limited input torque and solve it by Reinforcement Learning. As a classical baseline, we evaluate a two-stage controller composed of an energy-shaping swing-up law and a local Linear Quadratic Regulator (LQR) stabilizer around the target equilibrium. The swing-up controller increases the system's mechanical energy to drive the state toward a neighborhood of the desired equilibrium, a linearization of the nonlinear model yields an LQR that regulates the angle and angular-rate states to the target orientation with bounded input. This swing-up + LQR policy is a strong, interpretable reference for underactuated system and serves a point of comparison to the learned policy under identical limits and parameters. The solution shows that the learning is possible however, the different cases like stabilization in upward position or rotating of half turn are very difficult for increasing mass or ellipses with a strongly unequal perimeter ratio.

**arXiv ID:** 2603.12807
</details>

<details>
<summary><strong>Beyond Imitation: Reinforcement Learning Fine-Tuning for Adaptive Diffusion Navigation Policies</strong> - Junhe Sheng, Ruofei Bai, Kuan Xu, Ruimeng Liu, Jie Chen, Shenghai Yuan, Wei-Yun Yau, Lihua Xie - [[pdf]](https://arxiv.org/pdf/2603.12868)</summary>

**Abstract:** Diffusion-based robot navigation policies trained on large-scale imitation learning datasets, can generate multi-modal trajectories directly from the robot's visual observations, bypassing the traditional localization-mapping-planning pipeline and achieving strong zero-shot generalization. However, their performance remains constrained by the coverage of offline datasets, and when deployed in unseen settings, distribution shift often leads to accumulated trajectory errors and safety-critical failures. Adapting diffusion policies with reinforcement learning is challenging because their iterative denoising structure hinders effective gradient backpropagation, while also making the training of an additional value network computationally expensive and less stable. To address these issues, we propose a reinforcement learning fine-tuning framework tailored for diffusion-based navigation. The method leverages the inherent multi-trajectory sampling mechanism of diffusion models and adopts Group Relative Policy Optimization (GRPO), which estimates relative advantages across sampled trajectories without requiring a separate value network. To preserve pretrained representations while enabling adaptation, we freeze the visual encoder and selectively update the higher decoder layers and action head, enhancing safety-aware behaviors through online environmental feedback. On the PointGoal task in Isaac Sim, our approach improves the Success Rate from 52.0% to 58.7% and SPL from 0.49 to 0.54 on unseen scenes, while reducing collision frequency. Additional experiments show that the fine-tuned policy transfers zero-shot to a real quadruped platform and maintains stable performance in geometrically out-of-distribution environments, suggesting improved adaptability and safe generalization to new domains.

**arXiv ID:** 2603.12868
</details>

<details>
<summary><strong>How GenAI Mentor Configurations Shape Early Collaborative Dynamics: A Classroom Comparison of Individual and Shared Agents</strong> - Siyu Zha, Weijing Liu, Fei Qin, Jie Cao, Yanjin Wang, Yujia Liu, Kaiyi Zhang, Jiangtao Gong, Yingqing Xu - [[pdf]](https://arxiv.org/pdf/2603.12600)</summary>

**Abstract:** Generative artificial intelligence (GenAI) is increasingly embedded in computer-supported collaborative learning (CSCL), yet little empirical research has unpacked how different configurations of AI participation reshape collaborative processes. This study investigates how GenAI configuration shapes collaborative regulation in authentic classroom settings. Two eighth-grade classes engaged in small-group creative problem-solving under two conditions: a shared-AI configuration, in which each group interacted with a single AI mentor, and an individual-AI configuration, in which each student accessed a personal AI instance. Using multi-layer discourse coding combined with lag sequential analysis (LSA) and ordered network analysis (ONA), we examined interaction distribution, AI-student coupling, shared regulation processes, and teacher orchestration. Results reveal distinct regulatory dynamics across configurations. Shared AI access promoted convergence-oriented collaboration, with stronger alignment of shared regulatory states and more coordinated group-level reasoning. In contrast, individual AI access distributed support across learners, producing more exploratory and evaluative cycles but also more fragmented interaction patterns, accompanied by increased teacher intervention to manage divergence. These findings suggest that AI configuration functions as a structural design variable that reorganizes the regulatory ecology of classroom collaboration.

**arXiv ID:** 2603.12600
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
