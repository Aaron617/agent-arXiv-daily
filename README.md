# Agent arXiv Daily

**Last Updated:** 2026-01-14 02:33:13

**Total Papers:** 118

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (12 papers)</h2></summary>

<details>
<summary><strong>HiMem: Hierarchical Long-Term Memory for LLM Long-Horizon Agents</strong> - Ningning Zhang, Xingxing Yang, Zhizhong Tan, Weiping Deng, Wenyong Wang - [[pdf]](https://arxiv.org/pdf/2601.06377)</summary>

**Abstract:** Although long-term memory systems have made substantial progress in recent years, they still exhibit clear limitations in adaptability, scalability, and self-evolution under continuous interaction settings. Inspired by cognitive theories, we propose HiMem, a hierarchical long-term memory framework for long-horizon dialogues, designed to support memory construction, retrieval, and dynamic updating during sustained interactions. HiMem constructs cognitively consistent Episode Memory via a Topic-Aware Event--Surprise Dual-Channel Segmentation strategy, and builds Note Memory that captures stable knowledge through a multi-stage information extraction pipeline. These two memory types are semantically linked to form a hierarchical structure that bridges concrete interaction events and abstract knowledge, enabling efficient retrieval without sacrificing information fidelity. HiMem supports both hybrid and best-effort retrieval strategies to balance accuracy and efficiency, and incorporates conflict-aware Memory Reconsolidation to revise and supplement stored knowledge based on retrieval feedback. This design enables continual memory self-evolution over long-term use. Experimental results on long-horizon dialogue benchmarks demonstrate that HiMem consistently outperforms representative baselines in accuracy, consistency, and long-term reasoning, while maintaining favorable efficiency. Overall, HiMem provides a principled and scalable design paradigm for building adaptive and self-evolving LLM-based conversational agents. The code is available at this https URL.

**arXiv ID:** 2601.06377
</details>

<details>
<summary><strong>SafePro: Evaluating the Safety of Professional-Level AI Agents</strong> - Kaiwen Zhou, Shreedhar Jangam, Ashwin Nagarajan, Tejas Polu, Suhas Oruganti, Chengzhi Liu, Ching-Chen Kuo, Yuting Zheng, Sravana Narayanaraju, Xin Eric Wang - [[pdf]](https://arxiv.org/pdf/2601.06663)</summary>

**Abstract:** Large language model-based agents are rapidly evolving from simple conversational assistants into autonomous systems capable of performing complex, professional-level tasks in various domains. While these advancements promise significant productivity gains, they also introduce critical safety risks that remain under-explored. Existing safety evaluations primarily focus on simple, daily assistance tasks, failing to capture the intricate decision-making processes and potential consequences of misaligned behaviors in professional settings. To address this gap, we introduce \textbf{SafePro}, a comprehensive benchmark designed to evaluate the safety alignment of AI agents performing professional activities. SafePro features a dataset of high-complexity tasks across diverse professional domains with safety risks, developed through a rigorous iterative creation and review process. Our evaluation of state-of-the-art AI models reveals significant safety vulnerabilities and uncovers new unsafe behaviors in professional contexts. We further show that these models exhibit both insufficient safety judgment and weak safety alignment when executing complex professional tasks. In addition, we investigate safety mitigation strategies for improving agent safety in these scenarios and observe encouraging improvements. Together, our findings highlight the urgent need for robust safety mechanisms tailored to the next generation of professional AI agents.

**arXiv ID:** 2601.06663
</details>

<details>
<summary><strong>From Values to Frameworks: A Qualitative Study of Ethical Reasoning in Agentic AI Practitioners</strong> - Theodore Roberts, Bahram Zarrin - [[pdf]](https://arxiv.org/pdf/2601.06062)</summary>

**Abstract:** Agentic artificial intelligence systems are autonomous technologies capable of pursuing complex goals with minimal human oversight and are rapidly emerging as the next frontier in AI. While these systems promise major gains in productivity, they also raise new ethical challenges. Prior research has examined how different populations prioritize Responsible AI values, yet little is known about how practitioners actually reason through the trade-offs inherent in designing these autonomous systems. This paper investigates the ethical reasoning of AI practitioners through qualitative interviews centered on structured dilemmas in agentic AI deployment. We find that the responses of practitioners do not merely reflect value preferences but rather align with three distinct reasoning frameworks. First is a Customer-Centric framework where choices are justified by business interests, legality, and user autonomy. Second is a Design-Centric framework emphasizing technical safeguards and system constraints. Third is an Ethics-Centric framework prioritizing social good and moral responsibility beyond compliance. We argue that these frameworks offer distinct and necessary insights for navigating ethical trade-offs. Consequently, providers of agentic AI must look beyond general principles and actively manage how these diverse reasoning frameworks are represented in their decision-making processes to ensure robust ethical outcomes.

**arXiv ID:** 2601.06062
</details>

<details>
<summary><strong>Islamic Chatbots in the Age of Large Language Models</strong> - Muhammad Aurangzeb Ahmad - [[pdf]](https://arxiv.org/pdf/2601.06092)</summary>

**Abstract:** Large Language Models (LLMs) are rapidly transforming how communities access, interpret, and circulate knowledge, and religious communities are no exception. Chatbots powered by LLMs are beginning to reshape authority, pedagogy, and everyday religious practice in Muslim communities. We analyze the landscape of LLM powered Islamic chatbots and how they are transforming Islamic religious practices e.g., democratizing access to religious knowledge but also running the risk of erosion of authority. We discuss what kind of challenges do these systems raise for Muslim communities and explore recommendations for the responsible design of these systems.

**arXiv ID:** 2601.06092
</details>

<details>
<summary><strong>Toward Safe and Responsible AI Agents: A Three-Pillar Model for Transparency, Accountability, and Trustworthiness</strong> - Edward C. Cheng, Jeshua Cheng, Alice Siu - [[pdf]](https://arxiv.org/pdf/2601.06223)</summary>

**Abstract:** This paper presents a conceptual and operational framework for developing and operating safe and trustworthy AI agents based on a Three-Pillar Model grounded in transparency, accountability, and trustworthiness. Building on prior work in Human-in-the-Loop systems, reinforcement learning, and collaborative AI, the framework defines an evolutionary path toward autonomous agents that balances increasing automation with appropriate human oversight. The paper argues that safe agent autonomy must be achieved through progressive validation, analogous to the staged development of autonomous driving, rather than through immediate full automation. Transparency and accountability are identified as foundational requirements for establishing user trust and for mitigating known risks in generative AI systems, including hallucinations, data bias, and goal misalignment, such as the inversion problem. The paper further describes three ongoing work streams supporting this framework: public deliberation on AI agents conducted by the Stanford Deliberative Democracy Lab, cross-industry collaboration through the Safe AI Agent Consortium, and the development of open tooling for an agent operating environment aligned with the Three-Pillar Model. Together, these contributions provide both conceptual clarity and practical guidance for enabling the responsible evolution of AI agents that operate transparently, remain aligned with human values, and sustain societal trust.

**arXiv ID:** 2601.06223
</details>

<details>
<summary><strong>Amory: Building Coherent Narrative-Driven Agent Memory through Agentic Reasoning</strong> - Yue Zhou, Xiaobo Guo, Belhassen Bayar, Srinivasan H. Sengamedu - [[pdf]](https://arxiv.org/pdf/2601.06282)</summary>

**Abstract:** Long-term conversational agents face a fundamental scalability challenge as interactions extend over time: repeatedly processing entire conversation histories becomes computationally prohibitive. Current approaches attempt to solve this through memory frameworks that predominantly fragment conversations into isolated embeddings or graph representations and retrieve relevant ones in a RAG style. While computationally efficient, these methods often treat memory formation minimally and fail to capture the subtlety and coherence of human memory. We introduce Amory, a working memory framework that actively constructs structured memory representations through enhancing agentic reasoning during offline time. Amory organizes conversational fragments into episodic narratives, consolidates memories with momentum, and semanticizes peripheral facts into semantic memory. At retrieval time, the system employs coherence-driven reasoning over narrative structures. Evaluated on the LOCOMO benchmark for long-term reasoning, Amory achieves considerable improvements over previous state-of-the-art, with performance comparable to full context reasoning while reducing response time by 50%. Analysis shows that momentum-aware consolidation significantly enhances response quality, while coherence-driven retrieval provides superior memory coverage compared to embedding-based approaches.

**arXiv ID:** 2601.06282
</details>

<details>
<summary><strong>LLMs Can't Play Hangman: On the Necessity of a Private Working Memory for Language Agents</strong> - Davide Baldelli, Ali Parviz, Amal Zouaq, Sarath Chandar - [[pdf]](https://arxiv.org/pdf/2601.06973)</summary>

**Abstract:** As LLMs move from text completion toward autonomous agents, they remain constrained by the standard chat interface, which lacks private working memory. This raises a fundamental question: can agents reliably perform interactive tasks that depend on hidden state? We define Private State Interactive Tasks (PSITs), which require agents to generate and maintain hidden information while producing consistent public responses. We show theoretically that any agent restricted to the public conversation history cannot simultaneously preserve secrecy and consistency in PSITs, yielding an impossibility theorem. To empirically validate this limitation, we introduce a self-consistency testing protocol that evaluates whether agents can maintain a hidden secret across forked dialogue branches. Standard chat-based LLMs and retrieval-based memory baselines fail this test regardless of scale, demonstrating that semantic retrieval does not enable true state maintenance. To address this, we propose a novel architecture incorporating an explicit private working memory; we demonstrate that this mechanism restores consistency, establishing private state as a necessary component for interactive language agents.

**arXiv ID:** 2601.06973
</details>

<details>
<summary><strong>The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents</strong> - Weihao Xuan, Qingcheng Zeng, Heli Qi, Yunze Xiao, Junjue Wang, Naoto Yokoya - [[pdf]](https://arxiv.org/pdf/2601.07264)</summary>

**Abstract:** Autonomous agents based on large language models (LLMs) are rapidly evolving to handle multi-turn tasks, but ensuring their trustworthiness remains a critical challenge. A fundamental pillar of this trustworthiness is calibration, which refers to an agent's ability to express confidence that reliably reflects its actual performance. While calibration is well-established for static models, its dynamics in tool-integrated agentic workflows remain underexplored. In this work, we systematically investigate verbalized calibration in tool-use agents, revealing a fundamental confidence dichotomy driven by tool type. Specifically, our pilot study identifies that evidence tools (e.g., web search) systematically induce severe overconfidence due to inherent noise in retrieved information, while verification tools (e.g., code interpreters) can ground reasoning through deterministic feedback and mitigate miscalibration. To robustly improve calibration across tool types, we propose a reinforcement learning (RL) fine-tuning framework that jointly optimizes task accuracy and calibration, supported by a holistic benchmark of reward designs. We demonstrate that our trained agents not only achieve superior calibration but also exhibit robust generalization from local training environments to noisy web settings and to distinct domains such as mathematical reasoning. Our results highlight the necessity of domain-specific calibration strategies for tool-use agents. More broadly, this work establishes a foundation for building self-aware agents that can reliably communicate uncertainty in high-stakes, real-world deployments.

**arXiv ID:** 2601.07264
</details>

<details>
<summary><strong>Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions</strong> - Yongqi Li, Hao Lang, Tieyun Qian, Yongbin Li - [[pdf]](https://arxiv.org/pdf/2601.07516)</summary>

**Abstract:** Vision-language models are increasingly employed as multimodal conversational agents (MCAs) for diverse conversational tasks. Recently, reinforcement learning (RL) has been widely explored for adapting MCAs to various human-AI interaction scenarios. Despite showing great enhancement in generalization performance, fine-tuning MCAs via RL still faces challenges in handling the extremely large text token space. To address this, we learn a compact latent action space for RL fine-tuning instead. Specifically, we adopt the learning from observation mechanism to construct the codebook for the latent action space, where future observations are leveraged to estimate current latent actions that could further be used to reconstruct future observations. However, the scarcity of paired image-text data hinders learning a codebook with sufficient coverage. Thus, we leverage both paired image-text data and text-only data to construct the latent action space, using a cross-modal projector for transforming text embeddings into image-text embeddings. We initialize the cross-modal projector on paired image-text data, and further train it on massive text-only data with a novel cycle consistency loss to enhance its robustness. We show that our latent action based method outperforms competitive baselines on two conversation tasks across various RL algorithms.

**arXiv ID:** 2601.07516
</details>

<details>
<summary><strong>ES-Mem: Event Segmentation-Based Memory for Long-Term Dialogue Agents</strong> - Huhai Zou, Tianhao Sun, Chuanjiang He, Yu Tian, Zhenyang Li, Li Jin, Nayu Liu, Jiang Zhong, Kaiwen Wei - [[pdf]](https://arxiv.org/pdf/2601.07582)</summary>

**Abstract:** Memory is critical for dialogue agents to maintain coherence and enable continuous adaptation in long-term interactions. While existing memory mechanisms offer basic storage and retrieval capabilities, they are hindered by two primary limitations: (1) rigid memory granularity often disrupts semantic integrity, resulting in fragmented and incoherent memory units; (2) prevalent flat retrieval paradigms rely solely on surface-level semantic similarity, neglecting the structural cues of discourse required to navigate and locate specific episodic contexts. To mitigate these limitations, drawing inspiration from Event Segmentation Theory, we propose ES-Mem, a framework incorporating two core components: (1) a dynamic event segmentation module that partitions long-term interactions into semantically coherent events with distinct boundaries; (2) a hierarchical memory architecture that constructs multi-layered memories and leverages boundary semantics to anchor specific episodic memory for precise context localization. Evaluations on two memory benchmarks demonstrate that ES-Mem yields consistent performance gains over baseline methods. Furthermore, the proposed event segmentation module exhibits robust applicability on dialogue segmentation datasets.

**arXiv ID:** 2601.07582
</details>

<details>
<summary><strong>Personality-Aware Reinforcement Learning for Persuasive Dialogue with LLM-Driven Simulation</strong> - Donghuo Zeng, Roberto Legaspi, Kazushi Ikeda - [[pdf]](https://arxiv.org/pdf/2601.06877)</summary>

**Abstract:** Effective persuasive dialogue agents adapt their strategies to individual users, accounting for the evolution of their psychological states and intentions throughout conversations. We present a personality-aware reinforcement learning approach comprising three main modules: (1) a Strategy-Oriented Interaction Framework, which serves as an agenda-based strategy controller that selects strategy-level actions and generate responses via Maximal Marginal Relevance (MMR) retrieval to ensure contextual relevance, diversity, and scalable data generation; (2) Personality-Aware User Representation Learning, which produces an 81-dimensional mixed-type embedding predicted at each turn from recent exchanges and appended to the reinforcement learning state; and (3) a Dueling Double DQN (D3QN) model and Reward Prediction, in which the policy is conditioned on dialogue history and turn-level personality estimates and trained using a composite reward incorporating agreement intent, donation amount, and changeof-mind penalties. We use an agenda-based LLM simulation pipeline to generate diverse interactions, from which personality estimation is inferred from the generated utterances. Experiments on the PersuasionForGood (P4G) dataset augmented with simulated dialogues reveal three main findings: (i) turn-level personality conditioning improves policy adaptability and cumulative persuasion rewards; (ii) LLM-driven simulation enhances generalization to unseen user behaviors; and (iii) incorporating a change-of-mind penalty reduces post-agreement retractions while slightly improving donation outcomes. These results demonstrate that structured interaction, dynamic personality estimation, and behaviorally informed rewards together yield more effective persuasive policies.

**arXiv ID:** 2601.06877
</details>

<details>
<summary><strong>ColorBrowserAgent: An Intelligent GUI Agent for Complex Long-Horizon Web Automation</strong> - Jiamu Zhou, Jihong Wang, Weiming Zhang, Weiwen Liu, Zhuosheng Zhang, Xingyu Lou, Weinan Zhang, Huarong Deng, Jun Wang - [[pdf]](https://arxiv.org/pdf/2601.07262)</summary>

**Abstract:** The web browser serves as a primary interface for daily human activities, making its automation a critical frontier for Human-Centred AI. While Large Language Models (LLMs) have enabled autonomous agents to interact with web GUIs, their reliability in real-world scenarios is hampered by long-horizon instability and the vast heterogeneity of site designs. In this paper, we introduce ColorBrowserAgent, a framework designed for Collaborative Autonomy in complex web tasks. Our approach integrates two human-centred mechanisms: (1) Progressive Progress Summarization, which mimics human short-term memory to maintain coherence over extended interactions; and (2) Human-in-the-Loop Knowledge Adaptation, which bridges the knowledge gap in diverse environments by soliciting expert intervention only when necessary. This symbiotic design allows the agent to learn from human tips without extensive retraining, effectively combining the scalability of AI with the adaptability of human cognition. Evaluated on the WebArena benchmark using GPT-5, ColorBrowserAgent achieves a state-of-the-art success rate of 71.2\%, demonstrating the efficacy of interactive human assistance in robust web automation.

**arXiv ID:** 2601.07262
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (14 papers)</h2></summary>

<details>
<summary><strong>Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection</strong> - Olivia Shanhong Liu, Pai Chet Ng, De Wen Soh, Konstantinos N. Plataniotis - [[pdf]](https://arxiv.org/pdf/2601.07232)</summary>

**Abstract:** Humorous memes blend visual and textual cues to convey irony, satire, or social commentary, posing unique challenges for AI systems that must interpret intent rather than surface correlations. Existing multimodal or prompting-based models generate explanations for humor but operate in an open loop,lacking the ability to critique or refine their reasoning once a prediction is made. We propose FLoReNce, an agentic feedback reasoning framework that treats meme understanding as a closed-loop process during learning and an open-loop process during inference. In the closed loop, a reasoning agent is critiqued by a judge; the error and semantic feedback are converted into control signals and stored in a feedback-informed, non-parametric knowledge base. At inference, the model retrieves similar judged experiences from this KB and uses them to modulate its prompt, enabling better, self-aligned reasoning without finetuning. On the PrideMM dataset, FLoReNce improves both predictive performance and explanation quality over static multimodal baselines, showing that feedback-regulated prompting is a viable path to adaptive meme humor understanding.

**arXiv ID:** 2601.07232
</details>

<details>
<summary><strong>JudgeFlow: Agentic Workflow Optimization via Block Judge</strong> - Zihan Ma, Zhikai Zhao, Chuanbo Hua, Federico Berto, Jinkyoo Park - [[pdf]](https://arxiv.org/pdf/2601.07477)</summary>

**Abstract:** Optimizing LLM-based agentic workflows is challenging for scaling AI capabilities. Current methods rely on coarse, end-to-end evaluation signals and lack fine-grained signals on where to refine, often resulting in inefficient or low-impact modifications. To address these limitations, we propose {\our{}}, an Evaluation-Judge-Optimization-Update pipeline. We incorporate reusable, configurable logic blocks into agentic workflows to capture fundamental forms of logic. On top of this abstraction, we design a dedicated Judge module that inspects execution traces -- particularly failed runs -- and assigns rank-based responsibility scores to problematic blocks. These fine-grained diagnostic signals are then leveraged by an LLM-based optimizer, which focuses modifications on the most problematic block in the workflow. Our approach improves sample efficiency, enhances interpretability through block-level diagnostics, and provides a scalable foundation for automating increasingly complex agentic workflows. We evaluate {\our{}} on mathematical reasoning and code generation benchmarks, where {\our{}} achieves superior performance and efficiency compared to existing methods. The source code is publicly available at this https URL.

**arXiv ID:** 2601.07477
</details>

<details>
<summary><strong>Active Evaluation of General Agents: Problem Definition and Comparison of Baseline Algorithms</strong> - Marc Lanctot, Kate Larson, Ian Gemp, Michael Kaisers - [[pdf]](https://arxiv.org/pdf/2601.07651)</summary>

**Abstract:** As intelligent agents become more generally-capable, i.e. able to master a wide variety of tasks, the complexity and cost of properly evaluating them rises significantly. Tasks that assess specific capabilities of the agents can be correlated and stochastic, requiring many samples for accurate comparisons, leading to added costs. In this paper, we propose a formal definition and a conceptual framework for active evaluation of agents across multiple tasks, which assesses the performance of ranking algorithms as a function of number of evaluation data samples. Rather than curating, filtering, or compressing existing data sets as a preprocessing step, we propose an online framing: on every iteration, the ranking algorithm chooses the task and agents to sample scores from. Then, evaluation algorithms report a ranking of agents on each iteration and their performance is assessed with respect to the ground truth ranking over time. Several baselines are compared under different experimental contexts, with synthetic generated data and simulated online access to real evaluation data from Atari game-playing agents. We find that the classical Elo rating system -- while it suffers from well-known failure modes, in theory -- is a consistently reliable choice for efficient reduction of ranking error in practice. A recently-proposed method, Soft Condorcet Optimization, shows comparable performance to Elo on synthetic data and significantly outperforms Elo on real Atari agent evaluation. When task variation from the ground truth is high, selecting tasks based on proportional representation leads to higher rate of ranking error reduction.

**arXiv ID:** 2601.07651
</details>

<details>
<summary><strong>Autonomous QA Agent: A Retrieval-Augmented Framework for Reliable Selenium Script Generation</strong> - Dudekula Kasim Vali - [[pdf]](https://arxiv.org/pdf/2601.06034)</summary>

**Abstract:** Software testing is critical in the software development lifecycle, yet translating requirements into executable test scripts remains manual and error-prone. While Large Language Models (LLMs) can generate code, they often hallucinate non-existent UI elements. We present the Autonomous QA Agent, a Retrieval-Augmented Generation (RAG) system that grounds Selenium script generation in project-specific documentation and HTML structure. By ingesting diverse formats (Markdown, PDF, HTML) into a vector database, our system retrieves relevant context before generation. Evaluation on 20 e-commerce test scenarios shows our RAG approach achieves 100% (20/20) syntax validity and 90% (18/20, 95% CI: [85%, 95%], p < 0.001) execution success, compared to 30% for standard LLM generation. While our evaluation is limited to a single domain, our method significantly reduces hallucinations by grounding generation in actual DOM structure, demonstrating RAG's potential for automated UI testing.

**arXiv ID:** 2601.06034
</details>

<details>
<summary><strong>Agentic AI Microservice Framework for Deepfake and Document Fraud Detection in KYC Pipelines</strong> - Chandra Sekhar Kubam - [[pdf]](https://arxiv.org/pdf/2601.06241)</summary>

**Abstract:** The rapid proliferation of synthetic media, presentation attacks, and document forgeries has created significant vulnerabilities in Know Your Customer (KYC) workflows across financial services, telecommunications, and digital-identity ecosystems. Traditional monolithic KYC systems lack the scalability and agility required to counter adaptive fraud. This paper proposes an Agentic AI Microservice Framework that integrates modular vision models, liveness assessment, deepfake detection, OCR-based document forensics, multimodal identity linking, and a policy driven risk engine. The system leverages autonomous micro-agents for task decomposition, pipeline orchestration, dynamic retries, and human-in-the-loop escalation. Experimental evaluations demonstrate improved detection accuracy, reduced latency, and enhanced resilience against adversarial inputs. The framework offers a scalable blueprint for regulated industries seeking robust, real-time, and privacy-preserving KYC verification.

**arXiv ID:** 2601.06241
</details>

<details>
<summary><strong>Automated QoR improvement in OpenROAD with coding agents</strong> - Amur Ghose, Junyeong Jang, Andrew B. Kahng, Jakang Lee - [[pdf]](https://arxiv.org/pdf/2601.06268)</summary>

**Abstract:** EDA development and innovation has been constrained by scarcity of expert engineering resources. While leading LLMs have demonstrated excellent performance in coding and scientific reasoning tasks, their capacity to advance EDA technology itself has been largely untested. We present AuDoPEDA, an autonomous, repository-grounded coding system built atop OpenAI models and a Codex-class agent that reads OpenROAD, proposes research directions, expands them into implementation steps, and submits executable diffs. Our contributions include (i) a closed-loop LLM framework for EDA code changes; (ii) a task suite and evaluation protocol on OpenROAD for PPA-oriented improvements; and (iii) end-to-end demonstrations with minimal human oversight. Experiments in OpenROAD achieve routed wirelength reductions of up to 5.9%, and effective clock period reductions of up to 10.0%.

**arXiv ID:** 2601.06268
</details>

<details>
<summary><strong>OS-Symphony: A Holistic Framework for Robust and Generalist Computer-Using Agent</strong> - Bowen Yang, Kaiming Jin, Zhenyu Wu, Zhaoyang Liu, Qiushi Sun, Zehao Li, JingJing Xie, Zhoumianze Liu, Fangzhi Xu, Kanzhi Cheng, Qingyun Li, Yian Wang, Yu Qiao, Zun Wang, Zichen Ding - [[pdf]](https://arxiv.org/pdf/2601.07779)</summary>

**Abstract:** While Vision-Language Models (VLMs) have significantly advanced Computer-Using Agents (CUAs), current frameworks struggle with robustness in long-horizon workflows and generalization in novel domains. These limitations stem from a lack of granular control over historical visual context curation and the absence of visual-aware tutorial retrieval. To bridge these gaps, we introduce OS-Symphony, a holistic framework that comprises an Orchestrator coordinating two key innovations for robust automation: (1) a Reflection-Memory Agent that utilizes milestone-driven long-term memory to enable trajectory-level self-correction, effectively mitigating visual context loss in long-horizon tasks; (2) Versatile Tool Agents featuring a Multimodal Searcher that adopts a SeeAct paradigm to navigate a browser-based sandbox to synthesize live, visually aligned tutorials, thereby resolving fidelity issues in unseen scenarios. Experimental results demonstrate that OS-Symphony delivers substantial performance gains across varying model scales, establishing new state-of-the-art results on three online benchmarks, notably achieving 65.84% on OSWorld.

**arXiv ID:** 2601.07779
</details>

<details>
<summary><strong>Calibrating Agent-Based Financial Markets Simulators with Pretrainable Automatic Posterior Transformation-Based Surrogates</strong> - Boquan Jiang, Zhenhua Yang, Chenkai Wang, Muyao Zhong, Heping Fang, Peng Yang - [[pdf]](https://arxiv.org/pdf/2601.06920)</summary>

**Abstract:** Calibrating Agent-Based Models (ABMs) is an important optimization problem for simulating the complex social systems, where the goal is to identify the optimal parameter of a given ABM by minimizing the discrepancy between the simulated data and the real-world observations. Unfortunately, it suffers from the extensive computational costs of iterative evaluations, which involves the expensive simulation with the candidate parameter. While Surrogate-Assisted Evolutionary Algorithms (SAEAs) have been widely adopted to alleviate the computational burden, existing methods face two key limitations: 1) surrogating the original evaluation function is hard due the nonlinear yet multi-modal nature of the ABMs, and 2) the commonly used surrogates cannot share the optimization experience among multiple calibration tasks, making the batched calibration less effective. To address these issues, this work proposes Automatic posterior transformation with Negatively Correlated Search and Adaptive Trust-Region (ANTR). ANTR first replaces the traditional surrogates with a pretrainable neural density estimator that directly models the posterior distribution of the parameters given observed data, thereby aligning the optimization objective with parameter-space accuracy. Furthermore, we incorporate a diversity-preserving search strategy to prevent premature convergence and an adaptive trust-region method to efficiently allocate computational resources. We take two representative ABM-based financial market simulators as the test bench as due to the high non-linearity. Experiments demonstrate that the proposed ANTR significantly outperforms conventional metaheuristics and state-of-the-art SAEAs in both calibration accuracy and computational efficiency, particularly in batch calibration scenarios across multiple market conditions.

**arXiv ID:** 2601.06920
</details>

<details>
<summary><strong>SWE-EVO: Benchmarking Coding Agents in Long-Horizon Software Evolution Scenarios</strong> - Minh V. T. Thai, Tue Le, Dung Nguyen Manh, Huy Phan Nhat, Nghi D. Q. Bui - [[pdf]](https://arxiv.org/pdf/2512.18470)</summary>

**Abstract:** Existing benchmarks for AI coding agents focus on isolated, single-issue tasks such as fixing a bug or implementing a small feature. However, real-world software engineering is fundamentally a long-horizon endeavor: developers must interpret high-level requirements, plan coordinated changes across many files, and evolve codebases over multiple iterations while preserving existing functionality. We introduce SWE-EVO, a benchmark that evaluates agents on this long-horizon software evolution challenge. Constructed from release notes and version histories of seven mature open-source Python projects, Tool comprises 48 evolution tasks that require agents to implement multi-step modifications spanning an average of 21 files, validated against comprehensive test suites averaging 874 tests per instance. Experiments with state-of-the-art models reveal a striking capability gap: even GPT-5 with OpenHands achieves only a 21 percent resolution rate on Tool, compared to 65 percent on the single-issue SWE-Bench Verified. This demonstrates that current agents struggle with sustained, multi-file reasoning. We also propose Fix Rate, a fine-grained metric that captures partial progress toward solving these complex, long-horizon tasks.

**arXiv ID:** 2512.18470
</details>

<details>
<summary><strong>From RAG to Agentic RAG for Faithful Islamic Question Answering</strong> - Gagan Bhatia, Hamdy Mubarak, Mustafa Jarrar, George Mikros, Fadi Zaraket, Mahmoud Alhirthani, Mutaz Al-Khatib, Logan Cochrane, Kareem Darwish, Rashid Yahiaoui, Firoj Alam - [[pdf]](https://arxiv.org/pdf/2601.07528)</summary>

**Abstract:** LLMs are increasingly used for Islamic question answering, where ungrounded responses may carry serious religious consequences. Yet standard MCQ/MRC-style evaluations do not capture key real-world failure modes, notably free-form hallucinations and whether models appropriately abstain when evidence is lacking. To shed a light on this aspect we introduce ISLAMICFAITHQA, a 3,810-item bilingual (Arabic/English) generative benchmark with atomic single-gold answers, which enables direct measurement of hallucination and abstention. We additionally developed an end-to-end grounded Islamic modelling suite consisting of (i) 25K Arabic text-grounded SFT reasoning pairs, (ii) 5K bilingual preference samples for reward-guided alignment, and (iii) a verse-level Qur'an retrieval corpus of $\sim$6k atomic verses (ayat). Building on these resources, we develop an agentic Quran-grounding framework (agentic RAG) that uses structured tool calls for iterative evidence seeking and answer revision. Experiments across Arabic-centric and multilingual LLMs show that retrieval improves correctness and that agentic RAG yields the largest gains beyond standard RAG, achieving state-of-the-art performance and stronger Arabic-English robustness even with a small model (i.e., Qwen3 4B). We will make the experimental resources and datasets publicly available for the community.

**arXiv ID:** 2601.07528
</details>

<details>
<summary><strong>Is Agentic RAG worth it? An experimental comparison of RAG approaches</strong> - Pietro Ferrazzi, Milica Cvjeticanin, Alessio Piraccini, Davide Giannuzzi - [[pdf]](https://arxiv.org/pdf/2601.07711)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) systems are usually defined by the combination of a generator and a retrieval component that extracts textual context from a knowledge base to answer user queries. However, such basic implementations exhibit several limitations, including noisy or suboptimal retrieval, misuse of retrieval for out-of-scope queries, weak query-document matching, and variability or cost associated with the generator. These shortcomings have motivated the development of "Enhanced" RAG, where dedicated modules are introduced to address specific weaknesses in the workflow. More recently, the growing self-reflective capabilities of Large Language Models (LLMs) have enabled a new paradigm, which we refer to as "Agentic" RAG. In this approach, the LLM orchestrates the entire process-deciding which actions to perform, when to perform them, and whether to iterate-thereby reducing reliance on fixed, manually engineered modules. Despite the rapid adoption of both paradigms, it remains unclear which approach is preferable under which conditions. In this work, we conduct an extensive, empirically driven evaluation of Enhanced and Agentic RAG across multiple scenarios and dimensions. Our results provide practical insights into the trade-offs between the two paradigms, offering guidance on selecting the most effective RAG design for real-world applications, considering both costs and performance.

**arXiv ID:** 2601.07711
</details>

<details>
<summary><strong>SeePerSea: Multi-modal Perception Dataset of In-water Objects for Autonomous Surface Vehicles</strong> - Mingi Jeong, Arihant Chadda, Ziang Ren, Luyang Zhao, Haowen Liu, Monika Roznere, Aiwei Zhang, Yitao Jiang, Sabriel Achong, Samuel Lensgraf, Alberto Quattrini Li - [[pdf]](https://arxiv.org/pdf/2404.18411)</summary>

**Abstract:** This paper introduces the first publicly accessible labeled multi-modal perception dataset for autonomous maritime navigation, focusing on in-water obstacles within the aquatic environment to enhance situational awareness for Autonomous Surface Vehicles (ASVs). This dataset, collected over 4 years and consisting of diverse objects encountered under varying environmental conditions, aims to bridge the research gap in ASVs by providing a multi-modal, annotated, and ego-centric perception dataset, for object detection and classification. We also show the applicability of the proposed dataset by training and testing current deep learning-based open-source perception algorithms that have shown success in the autonomous ground vehicle domain. With the training and testing results, we discuss open challenges for existing datasets and methods, identifying future research directions. We expect that our dataset will contribute to the development of future marine autonomy pipelines and marine (field) robotics. This dataset is open source and found at this https URL.

**arXiv ID:** 2404.18411
</details>

<details>
<summary><strong>A Vision-Language-Action Model with Visual Prompt for OFF-Road Autonomous Driving</strong> - Liangdong Zhang, Yiming Nie, Haoyang Li, Fanjie Kong, Baobao Zhang, Shunxin Huang, Kai Fu, Chen Min, Liang Xiao - [[pdf]](https://arxiv.org/pdf/2601.03519)</summary>

**Abstract:** Efficient trajectory planning in off-road terrains presents a formidable challenge for autonomous vehicles, often necessitating complex multi-step pipelines. However, traditional approaches exhibit limited adaptability in dynamic environments. To address these limitations, this paper proposes OFF-EMMA, a novel end-to-end multimodal framework designed to overcome the deficiencies of insufficient spatial perception and unstable reasoning in visual-language-action (VLA) models for off-road autonomous driving scenarios. The framework explicitly annotates input images through the design of a visual prompt block and introduces a chain-of-thought with self-consistency (COT-SC) reasoning strategy to enhance the accuracy and robustness of trajectory planning. The visual prompt block utilizes semantic segmentation masks as visual prompts, enhancing the spatial understanding ability of pre-trained visual-language models for complex terrains. The COT- SC strategy effectively mitigates the error impact of outliers on planning performance through a multi-path reasoning mechanism. Experimental results on the RELLIS-3D off-road dataset demonstrate that OFF-EMMA significantly outperforms existing methods, reducing the average L2 error of the Qwen backbone model by 13.3% and decreasing the failure rate from 16.52% to 6.56%.

**arXiv ID:** 2601.03519
</details>

<details>
<summary><strong>EZBlender: Efficient 3D Editing with Plan-and-ReAct Agent</strong> - Hao Wang, Wenhui Zhu, Shao Tang, Zhipeng Wang, Xuanzhao Dong, Xin Li, Xiwen Chen, Ashish Bastola, Xinhao Huang, Yalin Wang, Abolfazl Razi - [[pdf]](https://arxiv.org/pdf/2601.07143)</summary>

**Abstract:** As a cornerstone of the modern digital economy, 3D modeling and rendering demand substantial resources and manual effort when scene editing is performed in the traditional manner. Despite recent progress in VLM-based agents for 3D editing, the fundamental trade-off between editing precision and agent responsiveness remains unresolved. To overcome these limitations, we present EZBlender, a Blender agent with a hybrid framework that combines planning-based task decomposition and reactive local autonomy for efficient human AI collaboration and semantically faithful 3D editing. Specifically, this unexplored Plan-and-ReAct design not only preserves editing quality but also significantly reduces latency and computational cost. To further validate the efficiency and effectiveness of the proposed edge-autonomy architecture, we construct a dedicated multi-tasking benchmark that has not been systematically investigated in prior research. In addition, we provide a comprehensive analysis of language model preference, system responsiveness, and economic efficiency.

**arXiv ID:** 2601.07143
</details>

</details>

<details open>
<summary><h2>LLM Agents (12 papers)</h2></summary>

<details>
<summary><strong>ReliabilityBench: Evaluating LLM Agent Reliability Under Production-Like Stress Conditions</strong> - Aayush Gupta - [[pdf]](https://arxiv.org/pdf/2601.06112)</summary>

**Abstract:** Existing benchmarks for tool-using LLM agents primarily report single-run success rates and miss reliability properties required in production. We introduce \textbf{ReliabilityBench}, a benchmark for evaluating agent reliability across three dimensions: (i) consistency under repeated execution using $\mathrm{pass}^k$, (ii) robustness to semantically equivalent task perturbations at intensity $\epsilon$, and (iii) fault tolerance under controlled tool/API failures at intensity $\lambda$. ReliabilityBench contributes a unified reliability surface $R(k,\epsilon,\lambda)$, \textit{action metamorphic relations} that define correctness via end-state equivalence rather than text similarity, and a chaos-engineering-style fault injection framework (timeouts, rate limits, partial responses, schema drift). We evaluate two models (Gemini 2.0 Flash, GPT-4o) and two agent architectures (ReAct, Reflexion) across four domains (scheduling, travel, customer support, e-commerce) over 1,280 episodes. Perturbations alone reduce success from 96.9% at $\epsilon=0$ to 88.1% at $\epsilon=0.2$. Rate limiting is the most damaging fault in ablations. ReAct is more robust than Reflexion under combined stress, and Gemini 2.0 Flash achieves comparable reliability to GPT-4o at much lower cost. ReliabilityBench provides a systematic framework for assessing production readiness of LLM agents.

**arXiv ID:** 2601.06112
</details>

<details>
<summary><strong>ToolGym: an Open-world Tool-using Environment for Scalable Agent Testing and Data Curation</strong> - Ziqiao Xi, Shuang Liang, Qi Liu, Jiaqing Zhang, Letian Peng, Fang Nan, Meshal Nayim, Tianhui Zhang, Rishika Mundada, Lianhui Qin, Biwei Huang, Kun Zhou - [[pdf]](https://arxiv.org/pdf/2601.06328)</summary>

**Abstract:** Tool-using LLM agents still struggle in open-world settings with large tool pools, long-horizon objectives, wild constraints, and unreliable tool states. For scalable and realistic training and testing, we introduce an open-world tool-using environment, built on 5,571 format unified tools across 204 commonly used apps. It includes a task creation engine that synthesizes long-horizon, multi-tool workflows with wild constraints, and a state controller that injects interruptions and failures to stress-test robustness. On top of this environment, we develop a tool select-then-execute agent framework with a planner-actor decomposition to separate deliberate reasoning and self-correction from step-wise execution. Comprehensive evaluation of state-of-the-art LLMs reveals the misalignment between tool planning and execution abilities, the constraint following weakness of existing LLMs, and DeepSeek-v3.2's strongest robustness. Finally, we collect 1,170 trajectories from our environment to fine-tune LLMs, achieving superior performance to baselines using 119k samples, indicating the environment's value as both a realistic benchmark and a data engine for tool-using agents. Our code and data will be publicly released.

**arXiv ID:** 2601.06328
</details>

<details>
<summary><strong>No More Stale Feedback: Co-Evolving Critics for Open-World Agent Learning</strong> - Zhicong Li, Lingjie Jiang, Yulan Hu, Xingchen Zeng, Yixia Li, Xiangwen Zhang, Guanhua Chen, Zheng Pan, Xin Li, Yong Liu - [[pdf]](https://arxiv.org/pdf/2601.06794)</summary>

**Abstract:** Critique-guided reinforcement learning (RL) has emerged as a powerful paradigm for training LLM agents by augmenting sparse outcome rewards with natural-language feedback. However, current methods often rely on static or offline critic models, which fail to adapt as the policy evolves. In on-policy RL, the agent's error patterns shift over time, causing stationary critics to become stale and providing feedback of diminishing utility. To address this, we introduce ECHO (Evolving Critic for Hindsight-Guided Optimization)}, a framework that jointly optimizes the policy and critic through a synchronized co-evolutionary loop. ECHO utilizes a cascaded rollout mechanism where the critic generates multiple diagnoses for an initial trajectory, followed by policy refinement to enable group-structured advantage estimation. We address the challenge of learning plateaus via a saturation-aware gain shaping objective, which rewards the critic for inducing incremental improvements in high-performing trajectories. By employing dual-track GRPO updates, ECHO ensures the critic's feedback stays synchronized with the evolving policy. Experimental results show that ECHO yields more stable training and higher long-horizon task success across open-world environments.

**arXiv ID:** 2601.06794
</details>

<details>
<summary><strong>Active Context Compression: Autonomous Memory Management in LLM Agents</strong> - Nikhil Verma - [[pdf]](https://arxiv.org/pdf/2601.07190)</summary>

**Abstract:** Large Language Model (LLM) agents struggle with long-horizon software engineering tasks due to "Context Bloat." As interaction history grows, computational costs explode, latency increases, and reasoning capabilities degrade due to distraction by irrelevant past errors. Existing solutions often rely on passive, external summarization mechanisms that the agent cannot control. This paper proposes Focus, an agent-centric architecture inspired by the biological exploration strategies of Physarum polycephalum (slime mold). The Focus Agent autonomously decides when to consolidate key learnings into a persistent "Knowledge" block and actively withdraws (prunes) the raw interaction history. Using an optimized scaffold matching industry best practices (persistent bash + string-replacement editor), we evaluated Focus on N=5 context-intensive instances from SWE-bench Lite using Claude Haiku 4.5. With aggressive prompting that encourages frequent compression, Focus achieves 22.7% token reduction (14.9M -> 11.5M tokens) while maintaining identical accuracy (3/5 = 60% for both agents). Focus performed 6.0 autonomous compressions per task on average, with token savings up to 57% on individual instances. We demonstrate that capable models can autonomously self-regulate their context when given appropriate tools and prompting, opening pathways for cost-aware agentic systems without sacrificing task performance.

**arXiv ID:** 2601.07190
</details>

<details>
<summary><strong>ARM: Role-Conditioned Neuron Transplantation for Training-Free Generalist LLM Agent Merging</strong> - Zhuoka Feng, Kang Chen, Sihan Zhao, Kai Xiong, Yaoning Wang, Minshen Yu, Junjie Nian, Changyi Xiao, Yixin Cao, Yugang Jiang - [[pdf]](https://arxiv.org/pdf/2601.07309)</summary>

**Abstract:** Interactive large language model agents have advanced rapidly, but most remain specialized to a single environment and fail to adapt robustly to other environments. Model merging offers a training-free alternative by integrating multiple experts into a single model. In this paper, we propose Agent-Role Merging (ARM), an activation-guided, role-conditioned neuron transplantation method for model merging in LLM agents. ARM improves existing merging methods from static natural language tasks to multi-turn agent scenarios, and over the generalization ability across various interactive environments. This is achieved with a well designed 3-step framework: 1) constructing merged backbones, 2) selection based on its role-conditioned activation analysis, and 3) neuron transplantation for fine-grained refinements. Without gradient-based optimization, ARM improves cross-benchmark generalization while enjoying efficiency. Across diverse domains, the model obtained via ARM merging outperforms prior model merging methods and domain-specific expert models, while demonstrating strong out-of-domain generalization.

**arXiv ID:** 2601.07309
</details>

<details>
<summary><strong>Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents</strong> - Miao Su, Yucan Guo, Zhongni Hou, Long Bai, Zixuan Li, Yufei Zhang, Guojun Yin, Wei Lin, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng - [[pdf]](https://arxiv.org/pdf/2601.07468)</summary>

**Abstract:** Memory enables Large Language Model (LLM) agents to perceive, store, and use information from past dialogues, which is essential for personalization. However, existing methods fail to properly model the temporal dimension of memory in two aspects: 1) Temporal inaccuracy: memories are organized by dialogue time rather than their actual occurrence time; 2) Temporal fragmentation: existing methods focus on point-wise memory, losing durative information that captures persistent states and evolving patterns. To address these limitations, we propose Temporal Semantic Memory (TSM), a memory framework that models semantic time for point-wise memory and supports the construction and utilization of durative memory. During memory construction, it first builds a semantic timeline rather than a dialogue one. Then, it consolidates temporally continuous and semantically related information into a durative memory. During memory utilization, it incorporates the query's temporal intent on the semantic timeline, enabling the retrieval of temporally appropriate durative memories and providing time-valid, duration-consistent context to support response generation. Experiments on LongMemEval and LoCoMo show that TSM consistently outperforms existing methods and achieves up to 12.2% absolute improvement in accuracy, demonstrating the effectiveness of the proposed method.

**arXiv ID:** 2601.07468
</details>

<details>
<summary><strong>LLM Agents in Law: Taxonomy, Applications, and Challenges</strong> - Shuang Liu, Ruijia Zhang, Ruoyun Ma, Yujia Deng, Lanyi Zhu, Jiayu Li, Zelong Li, Zhibin Shen, Mengnan Du - [[pdf]](https://arxiv.org/pdf/2601.06216)</summary>

**Abstract:** Large language models (LLMs) have precipitated a dramatic improvement in the legal domain, yet the deployment of standalone models faces significant limitations regarding hallucination, outdated information, and verifiability. Recently, LLM agents have attracted significant attention as a solution to these challenges, utilizing advanced capabilities such as planning, memory, and tool usage to meet the rigorous standards of legal practice. In this paper, we present a comprehensive survey of LLM agents for legal tasks, analyzing how these architectures bridge the gap between technical capabilities and domain-specific needs. Our major contributions include: (1) systematically analyzing the technical transition from standard legal LLMs to legal agents; (2) presenting a structured taxonomy of current agent applications across distinct legal practice areas; (3) discussing evaluation methodologies specifically for agentic performance in law; and (4) identifying open challenges and outlining future directions for developing robust and autonomous legal assistants.

**arXiv ID:** 2601.06216
</details>

<details>
<summary><strong>ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking</strong> - Qiang Zhang, Boli Chen, Fanrui Zhang, Ruixue Ding, Shihang Wang, Qiuchen Wang, Yinfeng Huang, Haonan Zhang, Rongxiang Zhu, Pengyong Wang, Ailin Ren, Xin Li, Pengjun Xie, Jiawei Liu, Ning Guo, Jingren Zhou, Zheng-Jun Zha - [[pdf]](https://arxiv.org/pdf/2601.06487)</summary>

**Abstract:** Reinforcement learning has substantially improved the performance of LLM agents on tasks with verifiable outcomes, but it still struggles on open-ended agent tasks with vast solution spaces (e.g., complex travel planning). Due to the absence of objective ground-truth for these tasks, current RL algorithms largely rely on reward models that assign scalar scores to individual responses. We contend that such pointwise scoring suffers from an inherent discrimination collapse: the reward model struggles to distinguish subtle advantages among different trajectories, resulting in scores within a group being compressed into a narrow range. Consequently, the effective reward signal becomes dominated by noise from the reward model, leading to optimization stagnation. To address this, we propose ArenaRL, a reinforcement learning paradigm that shifts from pointwise scalar scoring to intra-group relative ranking. ArenaRL introduces a process-aware pairwise evaluation mechanism, employing multi-level rubrics to assign fine-grained relative scores to trajectories. Additionally, we construct an intra-group adversarial arena and devise a tournament-based ranking scheme to obtain stable advantage signals. Empirical results confirm that the built seeded single-elimination scheme achieves nearly equivalent advantage estimation accuracy to full pairwise comparisons with O(N^2) complexity, while operating with only O(N) complexity, striking an optimal balance between efficiency and precision. Furthermore, to address the lack of full-cycle benchmarks for open-ended agents, we build Open-Travel and Open-DeepResearch, two high-quality benchmarks featuring a comprehensive pipeline covering SFT, RL training, and multi-dimensional evaluation. Extensive experiments show that ArenaRL substantially outperforms standard RL baselines, enabling LLM agents to generate more robust solutions for complex real-world tasks.

**arXiv ID:** 2601.06487
</details>

<details>
<summary><strong>CEDAR: Context Engineering for Agentic Data Science</strong> - Rishiraj Saha Roy, Chris Hinze, Luzian Hahn, Fabian Kuech - [[pdf]](https://arxiv.org/pdf/2601.06606)</summary>

**Abstract:** We demonstrate CEDAR, an application for automating data science (DS) tasks with an agentic setup. Solving DS problems with LLMs is an underexplored area that has immense market value. The challenges are manifold: task complexities, data sizes, computational limitations, and context restrictions. We show that these can be alleviated via effective context engineering. We first impose structure into the initial prompt with DS-specific input fields, that serve as instructions for the agentic system. The solution is then materialized as an enumerated sequence of interleaved plan and code blocks generated by separate LLM agents, providing a readable structure to the context at any step of the workflow. Function calls for generating these intermediate texts, and for corresponding Python code, ensure that data stays local, and only aggregate statistics and associated instructions are injected into LLM prompts. Fault tolerance and context management are introduced via iterative code generation and smart history rendering. The viability of our agentic data scientist is demonstrated using canonical Kaggle challenges.

**arXiv ID:** 2601.06606
</details>

<details>
<summary><strong>Memory Poisoning Attack and Defense on Memory Based LLM-Agents</strong> - Balachandra Devarangadi Sunil, Isheeta Sinha, Piyush Maheshwari, Shantanu Todmal, Shreyan Mallik, Shuchi Mishra - [[pdf]](https://arxiv.org/pdf/2601.05504)</summary>

**Abstract:** Large language model agents equipped with persistent memory are vulnerable to memory poisoning attacks, where adversaries inject malicious instructions through query only interactions that corrupt the agents long term memory and influence future responses. Recent work demonstrated that the MINJA (Memory Injection Attack) achieves over 95 % injection success rate and 70 % attack success rate under idealized conditions. However, the robustness of these attacks in realistic deployments and effective defensive mechanisms remain understudied. This work addresses these gaps through systematic empirical evaluation of memory poisoning attacks and defenses in Electronic Health Record (EHR) agents. We investigate attack robustness by varying three critical dimensions: initial memory state, number of indication prompts, and retrieval parameters. Our experiments on GPT-4o-mini, Gemini-2.0-Flash and Llama-3.1-8B-Instruct models using MIMIC-III clinical data reveal that realistic conditions with pre-existing legitimate memories dramatically reduce attack effectiveness. We then propose and evaluate two novel defense mechanisms: (1) Input/Output Moderation using composite trust scoring across multiple orthogonal signals, and (2) Memory Sanitization with trust-aware retrieval employing temporal decay and pattern-based filtering. Our defense evaluation reveals that effective memory sanitization requires careful trust threshold calibration to prevent both overly conservative rejection (blocking all entries) and insufficient filtering (missing subtle attacks), establishing important baselines for future adaptive defense mechanisms. These findings provide crucial insights for securing memory-augmented LLM agents in production environments.

**arXiv ID:** 2601.05504
</details>

<details>
<summary><strong>Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception</strong> - Yize Cheng, Arshia Soltani Moakhar, Chenrui Fan, Parsa Hosseini, Kazem Faghih, Zahra Sodagar, Wenxiao Wang, Soheil Feizi - [[pdf]](https://arxiv.org/pdf/2510.23853)</summary>

**Abstract:** Large language model (LLM) agents are increasingly used to interact with and execute tasks in dynamic environments. However, a critical yet overlooked limitation of these agents is that they, by default, assume a stationary context, failing to account for the real-world time elapsed between messages. We refer to this as "temporal blindness". This limitation hinders decisions about when to invoke tools, leading agents to either over-rely on stale context and skip needed tool calls, or under-rely on it and redundantly repeat tool calls. To study this challenge, we constructed TicToc, a diverse dataset of multi-turn user-agent message trajectories across 76 scenarios, spanning dynamic environments with high, medium, and low time sensitivity. We collected human preferences between "calling a tool" and "directly answering" on each sample, and evaluated how well LLM tool-calling decisions align with human preferences under varying amounts of elapsed time. Our analysis reveals that existing models display poor alignment with human temporal perception, with no model achieving a normalized alignment rate better than 65% when given time stamp information. We also show that naive, prompt-based alignment techniques have limited effectiveness for most models, but specific post-training alignment can be a viable way to align multi-turn LLM tool use with human temporal perception. Our data and findings provide a first step toward understanding and mitigating temporal blindness, offering insights to foster the development of more time-aware and human-aligned agents.

**arXiv ID:** 2510.23853
</details>

<details>
<summary><strong>FROAV: A Framework for RAG Observation and Agent Verification -- Lowering the Barrier to LLM Agent Research</strong> - Tzu-Hsuan Lin, Chih-Hsuan Kao - [[pdf]](https://arxiv.org/pdf/2601.07504)</summary>

**Abstract:** The rapid advancement of Large Language Models (LLMs) and their integration into autonomous agent systems has created unprecedented opportunities for document analysis, decision support, and knowledge retrieval. However, the complexity of developing, evaluating, and iterating on LLM-based agent workflows presents significant barriers to researchers, particularly those without extensive software engineering expertise. We present FROAV (Framework for RAG Observation and Agent Verification), an open-source research platform that democratizes LLM agent research by providing a plug-and-play architecture combining visual workflow orchestration, a comprehensive evaluation framework, and extensible Python integration. FROAV implements a multi-stage Retrieval-Augmented Generation (RAG) pipeline coupled with a rigorous "LLM-as-a-Judge" evaluation system, all accessible through intuitive graphical interfaces. Our framework integrates n8n for no-code workflow design, PostgreSQL for granular data management, FastAPI for flexible backend logic, and Streamlit for human-in-the-loop interaction. Through this integrated ecosystem, researchers can rapidly prototype RAG strategies, conduct prompt engineering experiments, validate agent performance against human judgments, and collect structured feedback-all without writing infrastructure code. We demonstrate the framework's utility through its application to financial document analysis, while emphasizing its material-agnostic architecture that adapts to any domain requiring semantic analysis. FROAV represents a significant step toward making LLM agent research accessible to a broader scientific community, enabling researchers to focus on hypothesis testing and algorithmic innovation rather than system integration challenges.

**arXiv ID:** 2601.07504
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (25 papers)</h2></summary>

<details>
<summary><strong>Dreaming Is Not a Bug: A Jung-Inspired Dream Layer for Multi-Agent LLM Companions</strong> - V. Cheung - [[pdf]](https://arxiv.org/pdf/2601.06115)</summary>

**Abstract:** Inspired by a personal dream about knowledge-sharing barriers in an everyday hardware project, this paper proposes a Jung-inspired "Dream Layer" for LLM companions, reframing controlled offline hallucinations as a resource for learning and relationship-building rather than a mere reliability bug. Drawing on Jung's notion of the collective unconscious as a shared repository of archetypal forms, we introduce an Artificial Collective Unconscious (ACU): a shared dream pool where agents contribute de-identified, abstract Interaction Templates that are later re-instantiated as idiosyncratic Dream Narratives. The Dream Layer runs strictly offline: logic-enforcing modules are relaxed and sampling temperature is increased, yielding safe but deliberately bizarre narratives (e.g., travel sequences with mismatched currencies) that augment data for rare events and edge-case safety tests; to harness risk productively, we add a governance stack of strict abstraction, temporal delays, and ephemeral memory. Through behavioural simulations of everyday dialogue and long-horizon adaptation tasks, we show that the Dream Layer enables a critical decoupling: agents remain firm on safety constraints (e.g., security policies) while becoming flexible in narrative strategy (e.g., using shared archetypal metaphors to resolve deadlocks), conceptually reframing hallucination so that online, unmarked instances remain bugs, whereas bounded, marked, and delayed ones become a goldmine for synthetic scenarios and deepened companionship, echoing anti-overfitting dream mechanisms proposed in contemporary neuroscience.

**arXiv ID:** 2601.06115
</details>

<details>
<summary><strong>ConSensus: Multi-Agent Collaboration for Multimodal Sensing</strong> - Hyungjun Yoon, Mohammad Malekzadeh, Sung-Ju Lee, Fahim Kawsar, Lorena Qendro - [[pdf]](https://arxiv.org/pdf/2601.06453)</summary>

**Abstract:** Large language models (LLMs) are increasingly grounded in sensor data to perceive and reason about human physiology and the physical world. However, accurately interpreting heterogeneous multimodal sensor data remains a fundamental challenge. We show that a single monolithic LLM often fails to reason coherently across modalities, leading to incomplete interpretations and prior-knowledge bias. We introduce ConSensus, a training-free multi-agent collaboration framework that decomposes multimodal sensing tasks into specialized, modality-aware agents. To aggregate agent-level interpretations, we propose a hybrid fusion mechanism that balances semantic aggregation, which enables cross-modal reasoning and contextual understanding, with statistical consensus, which provides robustness through agreement across modalities. While each approach has complementary failure modes, their combination enables reliable inference under sensor noise and missing data. We evaluate ConSensus on five diverse multimodal sensing benchmarks, demonstrating an average accuracy improvement of 7.1% over the single-agent baseline. Furthermore, ConSensus matches or exceeds the performance of iterative multi-agent debate methods while achieving a 12.7 times reduction in average fusion token cost through a single-round hybrid fusion protocol, yielding a robust and efficient solution for real-world multimodal sensing tasks.

**arXiv ID:** 2601.06453
</details>

<details>
<summary><strong>Agentic AI Empowered Intent-Based Networking for 6G</strong> - Genze Jiang, Kezhi Wang, Xiaomin Chen, Yizhou Huang - [[pdf]](https://arxiv.org/pdf/2601.06640)</summary>

**Abstract:** The transition towards sixth-generation (6G) wireless networks necessitates autonomous orchestration mechanisms capable of translating high-level operational intents into executable network configurations. Existing approaches to Intent-Based Networking (IBN) rely upon either rule-based systems that struggle with linguistic variation or end-to-end neural models that lack interpretability and fail to enforce operational constraints. This paper presents a hierarchical multi-agent framework where Large Language Model (LLM) based agents autonomously decompose natural language intents, consult domain-specific specialists, and synthesise technically feasible network slice configurations through iterative reasoning-action (ReAct) cycles. The proposed architecture employs an orchestrator agent coordinating two specialist agents, i.e., Radio Access Network (RAN) and Core Network agents, via ReAct-style reasoning, grounded in structured network state representations. Experimental evaluation across diverse benchmark scenarios shows that the proposed system outperforms rule-based systems and direct LLM prompting, with architectural principles applicable to Open RAN (O-RAN) deployments. The results also demonstrate that whilst contemporary LLMs possess general telecommunications knowledge, network automation requires careful prompt engineering to encode context-dependent decision thresholds, advancing autonomous orchestration capabilities for next-generation wireless systems.

**arXiv ID:** 2601.06640
</details>

<details>
<summary><strong>From Text to Simulation: A Multi-Agent LLM Workflow for Automated Chemical Process Design</strong> - Xufei Tian, Wenli Du, Shaoyi Yang, Han Hu, Hui Xin, Shifeng Qu, Ke Ye - [[pdf]](https://arxiv.org/pdf/2601.06776)</summary>

**Abstract:** Process simulation is a critical cornerstone of chemical engineering design. Current automated chemical design methodologies focus mainly on various representations of process flow diagrams. However, transforming these diagrams into executable simulation flowsheets remains a time-consuming and labor-intensive endeavor, requiring extensive manual parameter configuration within simulation software. In this work, we propose a novel multi-agent workflow that leverages the semantic understanding capabilities of large language models(LLMs) and enables iterative interactions with chemical process simulation software, achieving end-to-end automated simulation from textual process specifications to computationally validated software configurations for design enhancement. Our approach integrates four specialized agents responsible for task understanding, topology generation, parameter configuration, and evaluation analysis, respectively, coupled with Enhanced Monte Carlo Tree Search to accurately interpret semantics and robustly generate configurations. Evaluated on Simona, a large-scale process description dataset, our method achieves a 31.1% improvement in the simulation convergence rate compared to state-of-the-art baselines and reduces the design time by 89. 0% compared to the expert manual design. This work demonstrates the potential of AI-assisted chemical process design, which bridges the gap between conceptual design and practical implementation. Our workflow is applicable to diverse process-oriented industries, including pharmaceuticals, petrochemicals, food processing, and manufacturing, offering a generalizable solution for automated process design.

**arXiv ID:** 2601.06776
</details>

<details>
<summary><strong>OpenTinker: Separating Concerns in Agentic Reinforcement Learning</strong> - Siqi Zhu, Jiaxuan You - [[pdf]](https://arxiv.org/pdf/2601.07376)</summary>

**Abstract:** We introduce OpenTinker, an infrastructure for reinforcement learning (RL) of large language model (LLM) agents built around a separation of concerns across algorithm design, execution, and agent-environment interaction. Rather than relying on monolithic, end-to-end RL pipelines, OpenTinker decomposes agentic learning systems into lightweight, composable components with clearly defined abstraction boundaries. Users specify agents, environments, and interaction protocols, while inference and training are delegated to a managed execution runtime. OpenTinker introduces a centralized scheduler for managing training and inference workloads, including LoRA-based and full-parameter RL, supervised fine-tuning, and inference, over shared resources. We further discuss design principles for extending OpenTinker to multi-agent training. Finally, we present a set of RL use cases that demonstrate the effectiveness of the framework in practical agentic learning scenarios.

**arXiv ID:** 2601.07376
</details>

<details>
<summary><strong>Puzzle it Out: Local-to-Global World Model for Offline Multi-Agent Reinforcement Learning</strong> - Sijia li, Xinran Li, Shibo Chen, Jun Zhang - [[pdf]](https://arxiv.org/pdf/2601.07463)</summary>

**Abstract:** Offline multi-agent reinforcement learning (MARL) aims to solve cooperative decision-making problems in multi-agent systems using pre-collected datasets. Existing offline MARL methods primarily constrain training within the dataset distribution, resulting in overly conservative policies that struggle to generalize beyond the support of the data. While model-based approaches offer a promising solution by expanding the original dataset with synthetic data generated from a learned world model, the high dimensionality, non-stationarity, and complexity of multi-agent systems make it challenging to accurately estimate the transitions and reward functions in offline MARL. Given the difficulty of directly modeling joint dynamics, we propose a local-to-global (LOGO) world model, a novel framework that leverages local predictions-which are easier to estimate-to infer global state dynamics, thus improving prediction accuracy while implicitly capturing agent-wise dependencies. Using the trained world model, we generate synthetic data to augment the original dataset, expanding the effective state-action space. To ensure reliable policy learning, we further introduce an uncertainty-aware sampling mechanism that adaptively weights synthetic data by prediction uncertainty, reducing approximation error propagation to policies. In contrast to conventional ensemble-based methods, our approach requires only an additional encoder for uncertainty estimation, significantly reducing computational overhead while maintaining accuracy. Extensive experiments across 8 scenarios against 8 baselines demonstrate that our method surpasses state-of-the-art baselines on standard offline MARL benchmarks, establishing a new model-based baseline for generalizable offline multi-agent learning.

**arXiv ID:** 2601.07463
</details>

<details>
<summary><strong>VirtualEnv: A Platform for Embodied AI Research</strong> - Kabir Swain, Sijie Han, Ayush Raina, Jin Zhang, Shuang Li, Michael Stopa, Antonio Torralba - [[pdf]](https://arxiv.org/pdf/2601.07553)</summary>

**Abstract:** As large language models (LLMs) continue to improve in reasoning and decision-making, there is a growing need for realistic and interactive environments where their abilities can be rigorously evaluated. We present VirtualEnv, a next-generation simulation platform built on Unreal Engine 5 that enables fine-grained benchmarking of LLMs in embodied and interactive scenarios. VirtualEnv supports rich agent-environment interactions, including object manipulation, navigation, and adaptive multi-agent collaboration, as well as game-inspired mechanics like escape rooms and procedurally generated environments. We provide a user-friendly API built on top of Unreal Engine, allowing researchers to deploy and control LLM-driven agents using natural language instructions. We integrate large-scale LLMs and vision-language models (VLMs), such as GPT-based models, to generate novel environments and structured tasks from multimodal inputs. Our experiments benchmark the performance of several popular LLMs across tasks of increasing complexity, analyzing differences in adaptability, planning, and multi-agent coordination. We also describe our methodology for procedural task generation, task validation, and real-time environment control. VirtualEnv is released as an open-source platform, we aim to advance research at the intersection of AI and gaming, enable standardized evaluation of LLMs in embodied AI settings, and pave the way for future developments in immersive simulations and interactive entertainment.

**arXiv ID:** 2601.07553
</details>

<details>
<summary><strong>DIAGPaper: Diagnosing Valid and Specific Weaknesses in Scientific Papers via Multi-Agent Reasoning</strong> - Zhuoyang Zou, Abolfazl Ansari, Delvin Ce Zhang, Dongwon Lee, Wenpeng Yin - [[pdf]](https://arxiv.org/pdf/2601.07611)</summary>

**Abstract:** Paper weakness identification using single-agent or multi-agent LLMs has attracted increasing attention, yet existing approaches exhibit key limitations. Many multi-agent systems simulate human roles at a surface level, missing the underlying criteria that lead experts to assess complementary intellectual aspects of a paper. Moreover, prior methods implicitly assume identified weaknesses are valid, ignoring reviewer bias, misunderstanding, and the critical role of author rebuttals in validating review quality. Finally, most systems output unranked weakness lists, rather than prioritizing the most consequential issues for users. In this work, we propose DIAGPaper, a novel multi-agent framework that addresses these challenges through three tightly integrated modules. The customizer module simulates human-defined review criteria and instantiates multiple reviewer agents with criterion-specific expertise. The rebuttal module introduces author agents that engage in structured debate with reviewer agents to validate and refine proposed weaknesses. The prioritizer module learns from large-scale human review practices to assess the severity of validated weaknesses and surfaces the top-K severest ones to users. Experiments on two benchmarks, AAAR and ReviewCritique, demonstrate that DIAGPaper substantially outperforms existing methods by producing more valid and more paper-specific weaknesses, while presenting them in a user-oriented, prioritized manner.

**arXiv ID:** 2601.07611
</details>

<details>
<summary><strong>GenAITEd Ghana: A First-of-Its-Kind Context-Aware and Curriculum-Aligned Conversational AI Agent for Teacher Education</strong> - Matthew Nyaaba, Patrick Kyeremeh, Macharious Nabang, Bismark Nyaaba Akanzire, Sakina Acquah, Cyril Ababio Titty, Kotor Asare, Jerry Etornam Kudaya - [[pdf]](https://arxiv.org/pdf/2601.06093)</summary>

**Abstract:** Global frameworks increasingly advocate for Responsible Artificial Intelligence (AI) in education, yet they provide limited guidance on how ethical, culturally responsive, and curriculum-aligned AI can be operationalized within functioning teacher education systems, particularly in the Global South. This study addresses this gap through the design and evaluation of GenAITEd Ghana, a context-aware, region-specific conversational AI prototype developed to support teacher education in Ghana. Guided by a Design Science Research approach, the system was developed as a school-mimetic digital infrastructure aligned with the organizational logic of Ghanaian Colleges of Education and the National Council for Curriculum and Assessment (NaCCA) framework. GenAITEd Ghana operates as a multi-agent, retrieval-augmented conversational AI that coordinates multiple models for curriculum-grounded dialogue, automatic speech recognition, voice synthesis, and multimedia interaction. Two complementary prompt pathways were embedded: system-level prompts that enforce curriculum boundaries, ethical constraints, and teacher-in-the-loop oversight, and interaction-level semi-automated prompts that structure live pedagogical dialogue through clarification, confirmation, and guided response generation. Evaluation findings show that the system effectively enacted key Responsible AI principles, including transparency, accountability, cultural responsiveness, privacy, and human oversight. Human expert evaluations further indicated that GenAITEd Ghana is pedagogically appropriate for Ghanaian teacher education, promoting student engagement while preserving educators' professional authority. Identified challenges highlight the need for continued model integration, professional development, and critical AI literacy to mitigate risks of over-reliance.

**arXiv ID:** 2601.06093
</details>

<details>
<summary><strong>Multi-Agent Framework for Controllable and Protected Generative Content Creation: Addressing Copyright and Provenance in AI-Generated Media</strong> - Haris Khan, Sadia Asif, Shumaila Asif - [[pdf]](https://arxiv.org/pdf/2601.06232)</summary>

**Abstract:** The proliferation of generative AI systems creates unprecedented opportunities for content creation while raising critical concerns about controllability, copyright infringement, and content provenance. Current generative models operate as "black boxes" with limited user control and lack built-in mechanisms to protect intellectual property or trace content origin. We propose a novel multi-agent framework that addresses these challenges through specialized agent roles and integrated watermarking. Our system orchestrates Director, Generator, Reviewer, Integration, and Protection agents to ensure user intent alignment while embedding digital provenance markers. We demonstrate feasibility through two case studies: creative content generation with iterative refinement and copyright protection for AI-generated art in commercial contexts. Preliminary feasibility evidence from prior work indicates up to 23\% improvement in semantic alignment and 95\% watermark recovery rates. This work contributes to responsible generative AI deployment, positioning multi-agent systems as a solution for trustworthy creative workflows in legal and commercial applications.

**arXiv ID:** 2601.06232
</details>

<details>
<summary><strong>An Intelligent AI glasses System with Multi-Agent Architecture for Real-Time Voice Processing and Task Execution</strong> - Sheng-Kai Chen, Jyh-Horng Wu, Ching-Yao Lin, Yen-Ting Lin - [[pdf]](https://arxiv.org/pdf/2601.06235)</summary>

**Abstract:** This paper presents an AI glasses system that integrates real-time voice processing, artificial intelligence(AI) agents, and cross-network streaming capabilities. The system employs dual-agent architecture where Agent 01 handles Automatic Speech Recognition (ASR) and Agent 02 manages AI processing through local Large Language Models (LLMs), Model Context Protocol (MCP) tools, and Retrieval-Augmented Generation (RAG). The system supports real-time RTSP streaming for voice and video data transmission, eye tracking data collection, and remote task execution through RabbitMQ messaging. Implementation demonstrates successful voice command processing with multilingual support and cross-platform task execution capabilities.

**arXiv ID:** 2601.06235
</details>

<details>
<summary><strong>HiDVFS: A Hierarchical Multi-Agent DVFS Scheduler for OpenMP DAG Workloads</strong> - Mohammad Pivezhandi, Abusayeed Saifullah, Ali Jannesari - [[pdf]](https://arxiv.org/pdf/2601.06425)</summary>

**Abstract:** With advancements in multicore embedded systems, leakage power, exponentially tied to chip temperature, has surpassed dynamic power consumption. Energy-aware solutions use dynamic voltage and frequency scaling (DVFS) to mitigate overheating in performance-intensive scenarios, while software approaches allocate high-utilization tasks across core configurations in parallel systems to reduce power. However, existing heuristics lack per-core frequency monitoring, failing to address overheating from uneven core activity, and task assignments without detailed profiling overlook irregular execution patterns. We target OpenMP DAG workloads. Because makespan, energy, and thermal goals often conflict within a single benchmark, this work prioritizes performance (makespan) while reporting energy and thermal as secondary outcomes. To overcome these issues, we propose HiDVFS (a hierarchical multi-agent, performance-aware DVFS scheduler) for parallel systems that optimizes task allocation based on profiling data, core temperatures, and makespan-first objectives. It employs three agents: one selects cores and frequencies using profiler data, another manages core combinations via temperature sensors, and a third sets task priorities during resource contention. A makespan-focused reward with energy and temperature regularizers estimates future states and enhances sample efficiency. Experiments on the NVIDIA Jetson TX2 using the BOTS suite (9 benchmarks) compare HiDVFS against state-of-the-art approaches. With multi-seed validation (seeds 42, 123, 456), HiDVFS achieves the best finetuned performance with 4.16 plus/minus 0.58s average makespan (L10), representing a 3.44x speedup over GearDVFS (14.32 plus/minus 2.61s) and 50.4% energy reduction (63.7 kJ vs 128.4 kJ). Across all BOTS benchmarks, HiDVFS achieves an average 3.95x speedup and 47.1% energy reduction.

**arXiv ID:** 2601.06425
</details>

<details>
<summary><strong>Logic-Driven Semantic Communication for Resilient Multi-Agent Systems</strong> - Tamara Alshammari, Mehdi Bennis - [[pdf]](https://arxiv.org/pdf/2601.06733)</summary>

**Abstract:** The advent of 6G networks is accelerating autonomy and intelligence in large-scale, decentralized multi-agent systems (MAS). While this evolution enables adaptive behavior, it also heightens vulnerability to stressors such as environmental changes and adversarial behavior. Existing literature on resilience in decentralized MAS largely focuses on isolated aspects, such as fault tolerance, without offering a principled unified definition of multi-agent resilience. This gap limits the ability to design systems that can continuously sense, adapt, and recover under dynamic conditions. This article proposes a formal definition of MAS resilience grounded in two complementary dimensions: epistemic resilience, wherein agents recover and sustain accurate knowledge of the environment, and action resilience, wherein agents leverage that knowledge to coordinate and sustain goals under disruptions. We formalize resilience via temporal epistemic logic and quantify it using recoverability time (how quickly desired properties are re-established after a disturbance) and durability time (how long accurate beliefs and goal-directed behavior are sustained after recovery). We design an agent architecture and develop decentralized algorithms to achieve both epistemic and action resilience. We provide formal verification guarantees, showing that our specifications are sound with respect to the metric bounds and admit finite-horizon verification, enabling design-time certification and lightweight runtime monitoring. Through a case study on distributed multi-agent decision-making under stressors, we show that our approach outperforms baseline methods. Our formal verification analysis and simulation results highlight that the proposed framework enables resilient, knowledge-driven decision-making and sustained operation, laying the groundwork for resilient decentralized MAS in next-generation communication systems.

**arXiv ID:** 2601.06733
</details>

<details>
<summary><strong>DemMA: Dementia Multi-Turn Dialogue Agent with Expert-Guided Reasoning and Action Simulation</strong> - Yutong Song, Jiang Wu, Kazi Sharif, Honghui Xu, Nikil Dutt, Amir Rahmani - [[pdf]](https://arxiv.org/pdf/2601.06373)</summary>

**Abstract:** Simulating dementia patients with large language models (LLMs) is challenging due to the need to jointly model cognitive impairment, emotional dynamics, and nonverbal behaviors over long conversations. We present DemMA, an expert-guided dementia dialogue agent for high-fidelity multi-turn patient simulation. DemMA constructs clinically grounded dementia personas by integrating pathology information, personality traits, and subtype-specific memory-status personas informed by clinical experts. To move beyond text-only simulation, DemMA explicitly models nonverbal behaviors, including motion, facial expressions, and vocal cues. We further introduce a Chain-of-Thought distillation framework that trains a single LLM to jointly generate reasoning traces, patient utterances, and aligned behavioral actions within one forward pass, enabling efficient deployment without multi-agent inference. Extensive evaluations with experts, medical students, and LLM judges demonstrate that DemMA significantly outperforms strong baselines across multiple metrics.

**arXiv ID:** 2601.06373
</details>

<details>
<summary><strong>The Axiom of Consent: Friction Dynamics in Multi-Agent Coordination</strong> - Murad Farzulla - [[pdf]](https://arxiv.org/pdf/2601.06692)</summary>

**Abstract:** Multi-agent systems face a fundamental coordination problem: agents must coordinate despite heterogeneous preferences, asymmetric stakes, and imperfect information. When coordination fails, friction emerges: measurable resistance manifesting as deadlock, thrashing, communication overhead, or outright conflict. This paper derives a formal framework for analyzing coordination friction from a single axiom: actions affecting agents require authorization from those agents in proportion to stakes.
From this axiom of consent, we establish the kernel triple $({\alpha}, {\sigma}, {\epsilon})$ (alignment, stake, and entropy) characterizing any resource allocation configuration. The friction equation $F = {\sigma} (1 + {\epsilon})/(1 + {\alpha})$ predicts coordination difficulty as a function of preference alignment ${\alpha}$, stake magnitude ${\sigma}$, and communication entropy ${\epsilon}$. The Replicator-Optimization Mechanism (ROM) governs evolutionary selection over coordination strategies: configurations generating less friction persist longer, establishing consent-respecting arrangements as dynamical attractors rather than normative ideals.
We develop formal definitions for resource consent, coordination legitimacy, and friction-aware allocation in multi-agent systems. The framework yields testable predictions: MARL systems with higher reward alignment exhibit faster convergence; distributed allocations accounting for stake asymmetry generate lower coordination failure; AI systems with interpretability deficits produce friction proportional to the human-AI alignment gap. Applications to cryptocurrency governance and political systems demonstrate that the same equations govern friction dynamics across domains, providing a complexity science perspective on coordination under preference heterogeneity.

**arXiv ID:** 2601.06692
</details>

<details>
<summary><strong>SwarmFoam: An OpenFOAM Multi-Agent System Based on Multiple Types of Large Language Models</strong> - Chunwei Yang, Yankai Wang, Jianxiang Tang, Haojie Qu, Ziqiang Zou, YuLiu, Chunrui Deng, Zhifang Qiu, Ming Ding - [[pdf]](https://arxiv.org/pdf/2601.07252)</summary>

**Abstract:** Numerical simulation is one of the mainstream methods in scientific research, typically performed by professional engineers. With the advancement of multi-agent technology, using collaborating agents to replicate human behavior shows immense potential for intelligent Computational Fluid Dynamics (CFD) simulations. Some muti-agent systems based on Large Language Models have been proposed. However, they exhibit significant limitations when dealing with complex geometries. This paper introduces a new multi-agent simulation framework, SwarmFoam. SwarmFoam integrates functionalities such as Multi-modal perception, Intelligent error correction, and Retrieval-Augmented Generation, aiming to achieve more complex simulations through dual parsing of images and high-level instructions. Experimental results demonstrate that SwarmFoam has good adaptability to simulation inputs from different modalities. The overall pass rate for 25 test cases was 84%, with natural language and multi-modal input cases achieving pass rates of 80% and 86.7%, respectively. The work presented by SwarmFoam will further promote the development of intelligent agent methods for CFD.

**arXiv ID:** 2601.07252
</details>

<details>
<summary><strong>Agents of Diffusion: Enhancing Diffusion Language Models with Multi-Agent Reinforcement Learning for Structured Data Generation (Extended Version)</strong> - Aja Khanal, Kaushik T. Ranade, Rishabh Agrawal, Kalyan S. Basu, Apurva Narayan - [[pdf]](https://arxiv.org/pdf/2601.07152)</summary>

**Abstract:** Generating high-quality structured data such as JSON records, remains a fundamental challenge for large language models (LLMs), particularly when semantic richness must coexist with strict schema adherence. While autoregressive LLMs offer strong structural consistency, they often struggle with semantic variation and output diversity. In contrast, diffusion language models (DLMs) introduce powerful mechanisms for semantic richness and bidirectional decoding, yet lack the inductive biases needed for reliable structure preservation. We present Agents of Diffusion (AoD), a novel framework that unifies the generative flexibility of DLMs with the reasoning capabilities of autoregressive models through language-mediated reinforcement learning. AoD frames structured text generation as a multi-agent alignment process, where a prompt optimization agent collaborates with a judge agent to iteratively guide a DLM using natural language feedback. This approach enables controllable, schema-consistent generation without modifying model parameters or relying on handcrafted constraints. AoD advances the state of controllable generation by demonstrating that diffusion models, when supervised by cooperative agents, can achieve both high semantic novelty and structural fidelity. Across multiple structured data benchmarks, AoD consistently outperforms diffusion and autoregressive baselines, establishing a new path forward for structure-aware, diversity-enhanced text synthesis.

**arXiv ID:** 2601.07152
</details>

<details>
<summary><strong>VLM-CAD: VLM-Optimized Collaborative Agent Design Workflow for Analog Circuit Sizing</strong> - Guanyuan Pan, Yugui Lin, Tiansheng Zhou, Pietro Liò, Shuai Wang, Yaqi Wang - [[pdf]](https://arxiv.org/pdf/2601.07315)</summary>

**Abstract:** Analog mixed-signal circuit sizing involves complex trade-offs within high-dimensional design spaces. Existing automatic analog circuit sizing approaches often underutilize circuit schematics and lack the explainability required for industry adoption. To tackle these challenges, we propose a Vision Language Model-optimized collaborative agent design workflow (VLM-CAD), which analyzes circuits, optimizes DC operating points, performs inference-based sizing and executes external sizing optimization. We integrate Image2Net to annotate circuit schematics and generate a structured JSON description for precise interpretation by Vision Language Models. Furthermore, we propose an Explainable Trust Region Bayesian Optimization method (ExTuRBO) that employs collaborative warm-starting from agent-generated seeds and offers dual-granularity sensitivity analysis for external sizing optimization, supporting a comprehensive final design report. Experiment results on amplifier sizing tasks using 180nm, 90nm, and 45nm Predictive Technology Models demonstrate that VLM-CAD effectively balances power and performance, achieving a 100% success rate in optimizing an amplifier with a complementary input and a class-AB output stage, while maintaining total runtime under 43 minutes across all experiments.

**arXiv ID:** 2601.07315
</details>

<details>
<summary><strong>Cascading multi-agent anomaly detection in surveillance systems via vision-language models and embedding-based classification</strong> - Tayyab Rehman, Giovanni De Gasperis, Aly Shmahell - [[pdf]](https://arxiv.org/pdf/2601.06204)</summary>

**Abstract:** Intelligent anomaly detection in dynamic visual environments requires reconciling real-time performance with semantic interpretability. Conventional approaches address only fragments of this challenge. Reconstruction-based models capture low-level deviations without contextual reasoning, object detectors provide speed but limited semantics, and large vision-language systems deliver interpretability at prohibitive computational cost. This work introduces a cascading multi-agent framework that unifies these complementary paradigms into a coherent and interpretable architecture. Early modules perform reconstruction-gated filtering and object-level assessment, while higher-level reasoning agents are selectively invoked to interpret semantically ambiguous events. The system employs adaptive escalation thresholds and a publish-subscribe communication backbone, enabling asynchronous coordination and scalable deployment across heterogeneous hardware. Extensive evaluation on large-scale monitoring data demonstrates that the proposed cascade achieves a threefold reduction in latency compared to direct vision-language inference, while maintaining high perceptual fidelity (PSNR = 38.3 dB, SSIM = 0.965) and consistent semantic labeling. The framework advances beyond conventional detection pipelines by combining early-exit efficiency, adaptive multi-agent reasoning, and explainable anomaly attribution, establishing a reproducible and energy-efficient foundation for scalable intelligent visual monitoring.

**arXiv ID:** 2601.06204
</details>

<details>
<summary><strong>Modeling Descriptive Norms in Multi-Agent Systems: An Auto-Aggregation PDE Framework with Adaptive Perception Kernels</strong> - Chao Li, Ilia Derevitskii, Sergey Kovalchuk - [[pdf]](https://arxiv.org/pdf/2601.06557)</summary>

**Abstract:** This paper presents a PDE-based auto-aggregation model for simulating descriptive norm dynamics in autonomous multi-agent systems, capturing convergence and violation through non-local perception kernels and external potential fields. Extending classical transport equations, the framework represents opinion popularity as a continuous distribution, enabling direct interactions without Bayesian guessing of beliefs. Applied to a real-world COVID-19 dataset from a major medical center, the experimental results demonstrate that: when clinical guidelines serve as a top-down constraint mechanism, it effectively generates convergence of novel descriptive norms consistent with the dataset; in the bottom-up experiment, potential field guidance successfully promotes the system's reconstruction of descriptive norms aligned with the dataset through violation-and-recoupling; whereas fully autonomous interaction leads to the emergence of multi-centric normative structures independent of the dataset.

**arXiv ID:** 2601.06557
</details>

<details>
<summary><strong>The Subtle Art of Defection: Understanding Uncooperative Behaviors in LLM based Multi-Agent Systems</strong> - Devang Kulshreshtha, Wanyu Du, Raghav Jain, Srikanth Doss, Hang Su, Sandesh Swamy, Yanjun Qi - [[pdf]](https://arxiv.org/pdf/2511.15862)</summary>

**Abstract:** This paper introduces a novel framework for simulating and analyzing how uncooperative behaviors can destabilize or collapse LLM-based multi-agent systems. Our framework includes two key components: (1) a game theory-based taxonomy of uncooperative agent behaviors, addressing a notable gap in the existing literature; and (2) a structured, multi-stage simulation pipeline that dynamically generates and refines uncooperative behaviors as agents' states evolve. We evaluate the framework via a collaborative resource management setting, measuring system stability using metrics such as survival time and resource overuse rate. Empirically, our framework achieves 96.7% accuracy in generating realistic uncooperative behaviors, validated by human evaluations. Our results reveal a striking contrast: cooperative agents maintain perfect system stability (100% survival over 12 rounds with 0% resource overuse), while any uncooperative behavior can trigger rapid system collapse within 1 to 7 rounds. We also evaluate LLM-based defense methods, finding they detect some uncooperative behaviors, but some behaviors remain largely undetectable. These gaps highlight how uncooperative agents degrade collective outcomes and underscore the need for more resilient multi-agent systems.

**arXiv ID:** 2511.15862
</details>

<details>
<summary><strong>KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment</strong> - Yuxing Lu, Wei Wu, Xukai Zhao, Rui Peng, Jinzhuo Wang - [[pdf]](https://arxiv.org/pdf/2502.06472)</summary>

**Abstract:** Maintaining comprehensive and up-to-date knowledge graphs (KGs) is critical for modern AI systems, but manual curation struggles to scale with the rapid growth of scientific literature. This paper presents KARMA, a novel framework employing multi-agent large language models (LLMs) to automate KG enrichment through structured analysis of unstructured text. Our approach employs nine collaborative agents, spanning entity discovery, relation extraction, schema alignment, and conflict resolution that iteratively parse documents, verify extracted knowledge, and integrate it into existing graph structures while adhering to domain-specific schema. Experiments on 1,200 PubMed articles from three different domains demonstrate the effectiveness of KARMA in knowledge graph enrichment, with the identification of up to 38,230 new entities while achieving 83.1\% LLM-verified correctness and reducing conflict edges by 18.6\% through multi-layer assessments.

**arXiv ID:** 2502.06472
</details>

<details>
<summary><strong>Enhancing Cloud Network Resilience via a Robust LLM-Empowered Multi-Agent Reinforcement Learning Framework</strong> - Yixiao Peng, Hao Hu, Feiyang Li, Xinye Cao, Yingchang Jiang, Jipeng Tang, Guoshun Nan, Yuling Liu - [[pdf]](https://arxiv.org/pdf/2601.07122)</summary>

**Abstract:** While virtualization and resource pooling empower cloud networks with structural flexibility and elastic scalability, they inevitably expand the attack surface and challenge cyber resilience. Reinforcement Learning (RL)-based defense strategies have been developed to optimize resource deployment and isolation policies under adversarial conditions, aiming to enhance system resilience by maintaining and restoring network availability. However, existing approaches lack robustness as they require retraining to adapt to dynamic changes in network structure, node scale, attack strategies, and attack intensity. Furthermore, the lack of Human-in-the-Loop (HITL) support limits interpretability and flexibility. To address these limitations, we propose CyberOps-Bots, a hierarchical multi-agent reinforcement learning framework empowered by Large Language Models (LLMs). Inspired by MITRE ATT&CK's Tactics-Techniques model, CyberOps-Bots features a two-layer architecture: (1) An upper-level LLM agent with four modules--ReAct planning, IPDRR-based perception, long-short term memory, and action/tool integration--performs global awareness, human intent recognition, and tactical planning; (2) Lower-level RL agents, developed via heterogeneous separated pre-training, execute atomic defense actions within localized network regions. This synergy preserves LLM adaptability and interpretability while ensuring reliable RL execution. Experiments on real cloud datasets show that, compared to state-of-the-art algorithms, CyberOps-Bots maintains network availability 68.5% higher and achieves a 34.7% jumpstart performance gain when shifting the scenarios without retraining. To our knowledge, this is the first study to establish a robust LLM-RL framework with HITL support for cloud defense. We will release our framework to the community, facilitating the advancement of robust and autonomous defense in cloud networks.

**arXiv ID:** 2601.07122
</details>

<details>
<summary><strong>Integrating Symbolic RL Planning into a BDI-based Autonomous UAV Framework: System Integration and SIL Validation</strong> - Sangwoo Jeon, Juchul Shin, YeonJe Cho, Gyeong-Tae Kim, Seongwoo Kim - [[pdf]](https://arxiv.org/pdf/2508.11890)</summary>

**Abstract:** Modern autonomous drone missions increasingly require software frameworks capable of seamlessly integrating structured symbolic planning with adaptive reinforcement learning (RL). Although traditional rule-based architectures offer robust structured reasoning for drone autonomy, their capabilities fall short in dynamically complex operational environments that require adaptive symbolic planning. Symbolic RL (SRL), using the Planning Domain Definition Language (PDDL), explicitly integrates domain-specific knowledge and operational constraints, significantly improving the reliability and safety of unmanned aerial vehicle (UAV) decision making. In this study, we propose the AMAD-SRL framework, an extended and refined version of the Autonomous Mission Agents for Drones (AMAD) cognitive multi-agent architecture, enhanced with symbolic reinforcement learning for dynamic mission planning and execution. We validated our framework in a Software-in-the-Loop (SIL) environment structured identically to an intended Hardware-In-the-Loop Simulation (HILS) platform, ensuring seamless transition to real hardware. Experimental results demonstrate stable integration and interoperability of modules, successful transitions between BDI-driven and symbolic RL-driven planning phases, and consistent mission performance. Specifically, we evaluate a target acquisition scenario in which the UAV plans a surveillance path followed by a dynamic reentry path to secure the target while avoiding threat zones. In this SIL evaluation, mission efficiency improved by approximately 75% over a coverage-based baseline, measured by travel distance reduction. This study establishes a robust foundation for handling complex UAV missions and discusses directions for further enhancement and validation.

**arXiv ID:** 2508.11890
</details>

<details>
<summary><strong>Role-Playing LLM-Based Multi-Agent Support Framework for Detecting and Addressing Family Communication Bias</strong> - Rushia Harada, Yuken Kimura, Keito Inoshita - [[pdf]](https://arxiv.org/pdf/2507.11210)</summary>

**Abstract:** Well-being in family settings involves subtle psychological dynamics that conventional metrics often overlook. In particular, unconscious parental expectations, termed ideal parent bias, can suppress children's emotional expression and autonomy. This suppression, referred to as suppressed emotion, often stems from well-meaning but value-driven communication, which is difficult to detect or address from outside the family. Focusing on these latent dynamics, this study explores Large Language Model (LLM)-based support for psychologically safe family communication. We constructed a Japanese parent-child dialogue corpus of 30 scenarios, each annotated with metadata on ideal parent bias and suppressed emotion. Based on this corpus, we developed a Role-Playing LLM-based multi-agent dialogue support framework that analyzes dialogue and generates feedback. Specialized agents detect suppressed emotion, describe implicit ideal parent bias in parental speech, and infer contextual attributes such as the child's age and background. A meta-agent compiles these outputs into a structured report, which is then passed to five selected expert agents. These agents collaboratively generate empathetic and actionable feedback through a structured four-step discussion process. Experiments show that the system can detect categories of suppressed emotion with moderate accuracy and produce feedback rated highly in empathy and practicality. Moreover, simulated follow-up dialogues incorporating this feedback exhibited signs of improved emotional expression and mutual understanding, suggesting the framework's potential in supporting positive transformation in family interactions.

**arXiv ID:** 2507.11210
</details>

</details>

<details open>
<summary><h2>Other Agent Research (17 papers)</h2></summary>

<details>
<summary><strong>PsyAgent: Constructing Human-like Agents Based on Psychological Modeling and Contextual Interaction</strong> - Zibin Meng, Kani Chen - [[pdf]](https://arxiv.org/pdf/2601.06158)</summary>

**Abstract:** Human-like agents require modeling how dispositions interact with social structure. We present PsyAgent, which couples a Big Five trait prior with Bourdieu's cognitive-social co-structure. PsyAgent comprises: (i) Individual Structure (IS), a machine-usable profile encoding traits and facets, cognitive style, values, cultural and educational capital, and salient life episodes; and (ii) Multi-Scenario Contexting (MSC), role-relationship-norm frames spanning eight arenas (work, family, friendship, strangers and civic life, solitude and self-regulation, romance, learning, and public expression). At inference, fixed structured prompts bind the active scenario to the agent profile, yielding behavior that is stable yet context-sensitive. We instantiate IS and MSC to synthesize supervision (role-play dialogues, decision probes, feedback trajectories) and then fine-tune a small LLM. The resulting model produces consistent, identifiable persona-aligned behaviors for specified Big Five configurations and matches or exceeds several larger untuned LLMs and other untuned baselines on our metrics: persona consistency, contextual appropriateness, style matching, trait identifiability, and long-horizon stability. Ablations show IS chiefly improves trait fidelity and stylistic stability, while MSC drives norm awareness and decision fit; both are necessary for cross-scenario performance. PsyAgent offers a precise, data-efficient architecture for personality-grounded agents.

**arXiv ID:** 2601.06158
</details>

<details>
<summary><strong>Agentic Diagnostic Reasoning over Telecom and Datacenter Infrastructure</strong> - Nicolas Tacheny - [[pdf]](https://arxiv.org/pdf/2601.07342)</summary>

**Abstract:** Large-scale telecom and datacenter infrastructures rely on multi-layered service and resource models, where failures propagate across physical and logical components and affect multiple customers. Traditional approaches to root cause analysis(RCA) rely on hard-coded graph traversal algorithms or rule-based correlation engines, which are costly to maintain and tightly coupled to the infrastructure model.
In this work, we introduce an agentic diagnostic framework where a Large Language Model (LLM) performs step-wise investigation using a constrained tool space exposed through the Model Context Protocol (MCP). Instead of embedding causal logic or traversal algorithms into the application, the agent autonomously navigates the infrastructure model by invoking tools for service lookup, dependency retrieval, structured and unstructured data, and event analysis, and impact discovery. We define an investigation protocol that structures the agent's reasoning and ensures grounding, reproducibility, and safe handling of missing or ambiguous information.
This work lays the foundation for autonomous incident resolution and change impact mitigation. Future systems will not only diagnose and remediate infrastructure failures, but also predict the impact of planned changes on services and customers, enabling operators to mitigate risks before executing maintenance operations.

**arXiv ID:** 2601.07342
</details>

<details>
<summary><strong>Learning How to Remember: A Meta-Cognitive Management Method for Structured and Transferable Agent Memory</strong> - Sirui Liang, Pengfei Cao, Jian Zhao, Wenhao Teng, Xiangwen Liao, Jun Zhao, Kang Liu - [[pdf]](https://arxiv.org/pdf/2601.07470)</summary>

**Abstract:** Large language model (LLM) agents increasingly rely on accumulated memory to solve long-horizon decision-making tasks. However, most existing approaches store memory in fixed representations and reuse it at a single or implicit level of abstraction, which limits generalization and often leads to negative transfer when distribution shift. This paper proposes the Meta-Cognitive Memory Abstraction method (MCMA), which treats memory abstraction as a learnable cognitive skill rather than a fixed design choice. MCMA decouples task execution from memory management by combining a frozen task model with a learned memory copilot. The memory copilot is trained using direct preference optimization, it determines how memories should be structured, abstracted, and reused. Memories are further organized into a hierarchy of abstraction levels, enabling selective reuse based on task similarity. When no memory is transferable, MCMA transfers the ability to abstract and manage memory by transferring the memory copilot. Experiments on ALFWorld, ScienceWorld, and BabyAI demonstrate substantial improvements in performance, out-of-distribution generalization, and cross-task transfer over several baselines.

**arXiv ID:** 2601.07470
</details>

<details>
<summary><strong>Beyond Entangled Planning: Task-Decoupled Planning for Long-Horizon Agents</strong> - Yunfan Li, Bingbing Xu, Xueyun Tian, Xiucheng Xu, Huawei Shen - [[pdf]](https://arxiv.org/pdf/2601.07577)</summary>

**Abstract:** Recent advances in large language models (LLMs) have enabled agents to autonomously execute complex, long-horizon tasks, yet planning remains a primary bottleneck for reliable task execution. Existing methods typically fall into two paradigms: step-wise planning, which is reactive but often short-sighted; and one-shot planning, which generates a complete plan upfront yet is brittle to execution errors. Crucially, both paradigms suffer from entangled contexts, where the agent must reason over a monolithic history spanning multiple sub-tasks. This entanglement increases cognitive load and lets local errors propagate across otherwise independent decisions, making recovery computationally expensive. To address this, we propose Task-Decoupled Planning (TDP), a training-free framework that replaces entangled reasoning with task decoupling. TDP decomposes tasks into a directed acyclic graph (DAG) of sub-goals via a Supervisor. Using a Planner and Executor with scoped contexts, TDP confines reasoning and replanning to the active sub-task. This isolation prevents error propagation and corrects deviations locally without disrupting the workflow. Results on TravelPlanner, ScienceWorld, and HotpotQA show that TDP outperforms strong baselines while reducing token consumption by up to 82%, demonstrating that sub-task decoupling improves both robustness and efficiency for long-horizon agents.

**arXiv ID:** 2601.07577
</details>

<details>
<summary><strong>Socio-technical aspects of Agentic AI</strong> - Praveen Kumar Donta, Alaa Saleh, Ying Li, Shubham Vaishnav, Kai Fang, Hailin Feng, Yuchao Xia, Thippa Reddy Gadekallu, Qiyang Zhang, Xiaodan Shi, Ali Beikmohammadi, Sindri Magnússon, Ilir Murturi, Chinmaya Kumar Dehury, Marcin Paprzycki, Lauri Loven, Sasu Tarkoma, Schahram Dustdar - [[pdf]](https://arxiv.org/pdf/2601.06064)</summary>

**Abstract:** Agentic Artificial Intelligence (AI) represents a fundamental shift in the design of intelligent systems, characterized by interconnected components that collectively enable autonomous perception, reasoning, planning, action, and learning. Recent research on agentic AI has largely focused on technical foundations, including system architectures, reasoning and planning mechanisms, coordination strategies, and application-level performance across domains. However, the societal, ethical, economic, environmental, and governance implications of agentic AI remain weakly integrated into these technical treatments. This paper addresses this gap by presenting a socio-technical analysis of agentic AI that explicitly connects core technical components with societal context. We examine how architectural choices in perception, cognition, planning, execution, and memory introduce dependencies related to data governance, accountability, transparency, safety, and sustainability. To structure this analysis, we adopt the MAD-BAD-SAD construct as an analytical lens, capturing motivations, applications, and moral dilemmas (MAD); biases, accountability, and dangers (BAD); and societal impact, adoption, and design considerations (SAD). Using this lens, we analyze ethical considerations, implications, and challenges arising from contemporary agentic AI systems and assess their manifestation across emerging applications, including healthcare, education, industry, smart and sustainable cities, social services, communications and networking, and earth observation and satellite communications. The paper further identifies open challenges and suggests future research directions, framing agentic AI as an integrated socio-technical system whose behavior and impact are co-produced by algorithms, data, organizational practices, regulatory frameworks, and social norms.

**arXiv ID:** 2601.06064
</details>

<details>
<summary><strong>MemGovern: Enhancing Code Agents through Learning from Governed Human Experiences</strong> - Qihao Wang, Ziming Cheng, Shuo Zhang, Fan Liu, Rui Xu, Heng Lian, Kunyi Wang, Xiaoming Yu, Jianghao Yin, Sen Hu, Yue Hu, Shaolei Zhang, Yanbing Liu, Ronghao Chen, Huacan Wang - [[pdf]](https://arxiv.org/pdf/2601.06789)</summary>

**Abstract:** While autonomous software engineering (SWE) agents are reshaping programming paradigms, they currently suffer from a "closed-world" limitation: they attempt to fix bugs from scratch or solely using local context, ignoring the immense historical human experience available on platforms like GitHub. Accessing this open-world experience is hindered by the unstructured and fragmented nature of real-world issue-tracking data. In this paper, we introduce MemGovern, a framework designed to govern and transform raw GitHub data into actionable experiential memory for agents. MemGovern employs experience governance to convert human experience into agent-friendly experience cards and introduces an agentic experience search strategy that enables logic-driven retrieval of human expertise. By producing 135K governed experience cards, MemGovern achieves a significant performance boost, improving resolution rates on the SWE-bench Verified by 4.65%. As a plug-in approach, MemGovern provides a solution for agent-friendly memory infrastructure.

**arXiv ID:** 2601.06789
</details>

<details>
<summary><strong>The Practicality of Normalizing Flow Test-Time Training in Bayesian Inference for Agent-Based Models</strong> - Junyao Zhang, Jinglai Li, Junqi Tang - [[pdf]](https://arxiv.org/pdf/2601.07413)</summary>

**Abstract:** Agent-Based Models (ABMs) are gaining great popularity in economics and social science because of their strong flexibility to describe the realistic and heterogeneous decisions and interaction rules between individual agents. In this work, we investigate for the first time the practicality of test-time training (TTT) of deep models such as normalizing flows, in the parameters posterior estimations of ABMs. We propose several practical TTT strategies for fine-tuning the normalizing flow against distribution shifts. Our numerical study demonstrates that TTT schemes are remarkably effective, enabling real-time adjustment of flow-based inference for ABM parameters.

**arXiv ID:** 2601.07413
</details>

<details>
<summary><strong>Characterizing Agent-Based Model Dynamics via $ε$-Machines and Kolmogorov-Style Complexity</strong> - Roberto Garrone - [[pdf]](https://arxiv.org/pdf/2510.12729)</summary>

**Abstract:** We propose a two-level information-theoretic framework for characterizing the informational organization of Agent-Based Model (ABM) dynamics within the broader paradigm of Complex Adaptive Systems (CAS). At the macro level, a pooled $\varepsilon$-machine is reconstructed as a reference model summarizing the system-wide informational regime. At the micro level, $\varepsilon$-machines are reconstructed for each caregiver--elder dyad and variable, complemented by algorithm-agnostic Kolmogorov-style measures, including normalized LZ78 complexity and bits per symbol from lossless compression. The resulting feature set, $\{h_{\mu}, C_{\mu}, E, \mathrm{LZ78}, \mathrm{bps}\}$, enables distributional analysis, stratified comparisons, and unsupervised clustering across agents and scenarios. Empirical results show that coupling $\varepsilon$-machines with compression diagnostics yields a coherent picture of where predictive information resides in the caregiving ABM. Global reconstructions provide a memoryless baseline ($L{=}0$ under coarse symbolizations), whereas per-dyad models reveal localized structure, particularly for walkability under ordinal encodings ($m{=}3$). Compression metrics corroborate these patterns: dictionary compressors agree on algorithmic redundancy, while normalized LZ78 captures statistical novelty. Socioeconomic variables display cross-sectional heterogeneity and near-memoryless dynamics, whereas spatial interaction induces bounded temporal memory and recurrent regimes. The framework thus distinguishes semantic organization (predictive causation and memory) from syntactic simplicity (description length) and clarifies how emergence manifests at different system layers. It is demonstrated on a caregiver--elder case study with dyad-level $\varepsilon$-machine reconstructions and compression-based diagnostics.

**arXiv ID:** 2510.12729
</details>

<details>
<summary><strong>A Graph-Theoretical Perspective on Law Design for Multiagent Systems</strong> - Qi Shi, Pavel Naumov - [[pdf]](https://arxiv.org/pdf/2511.06361)</summary>

**Abstract:** A law in a multiagent system is a set of constraints imposed on agents' behaviours to avoid undesirable outcomes. The paper considers two types of laws: useful laws that, if followed, completely eliminate the undesirable outcomes and gap-free laws that guarantee that at least one agent can be held responsible each time an undesirable outcome occurs. In both cases, we study the problem of finding a law that achieves the desired result by imposing the minimum restrictions.
We prove that, for both types of laws, the minimisation problem is NP-hard even in the simple case of one-shot concurrent interactions. We also show that the approximation algorithm for the vertex cover problem in hypergraphs could be used to efficiently approximate the minimum laws in both cases.

**arXiv ID:** 2511.06361
</details>

<details>
<summary><strong>Value of Information: A Framework for Human-Agent Communication</strong> - Yijiang River Dong, Tiancheng Hu, Zheng Hui, Caiqi Zhang, Ivan Vulić, Andreea Bobu, Nigel Collier - [[pdf]](https://arxiv.org/pdf/2601.06407)</summary>

**Abstract:** Large Language Model (LLM) agents deployed for real-world tasks face a fundamental dilemma: user requests are underspecified, yet agents must decide whether to act on incomplete information or interrupt users for clarification. Existing approaches either rely on brittle confidence thresholds that require task-specific tuning, or fail to account for the varying stakes of different decisions. We introduce a decision-theoretic framework that resolves this trade-off through the Value of Information (VoI), enabling agents to dynamically weigh the expected utility gain from asking questions against the cognitive cost imposed on users. Our inference-time method requires no hyperparameter tuning and adapts seamlessly across contexts-from casual games to medical diagnosis. Experiments across four diverse domains (20 Questions, medical diagnosis, flight booking, and e-commerce) show that VoI consistently matches or exceeds the best manually-tuned baselines, achieving up to 1.36 utility points higher in high-cost settings. This work provides a parameter-free framework for adaptive agent communication that explicitly balances task risk, query ambiguity, and user effort.

**arXiv ID:** 2601.06407
</details>

<details>
<summary><strong>Can a Unimodal Language Agent Provide Preferences to Tune a Multimodal Vision-Language Model?</strong> - Sazia Tabasum Mim, Jack Morris, Manish Dhakal, Yanming Xiu, Maria Gorlatova, Yi Ding - [[pdf]](https://arxiv.org/pdf/2601.06424)</summary>

**Abstract:** To explore a more scalable path for adding multimodal capabilities to existing LLMs, this paper addresses a fundamental question: Can a unimodal LLM, relying solely on text, reason about its own informational needs and provide effective feedback to optimize a multimodal model? To answer this, we propose a method that enables a language agent to give feedback to a vision-language model (VLM) to adapt text generation to the agent's preferences. Our results from different experiments affirm this hypothesis, showing that LLM preference feedback significantly enhances VLM descriptions. Using our proposed method, we find that the VLM can generate multimodal scene descriptions to help the LLM better understand multimodal context, leading to improvements of maximum 13% in absolute accuracy compared to the baseline multimodal approach. Furthermore, a human study validated our AI-driven feedback, showing a 64.6% preference alignment rate between the LLM's choices and human judgments. Extensive experiments provide insights on how and why the method works and its limitations.

**arXiv ID:** 2601.06424
</details>

<details>
<summary><strong>Precision Meets Art: Autonomous Multi-UAV System for Large Scale Mural Drawing</strong> - Andrei A. Korigodskii, Artem E. Vasiunik, Georgii A. Varin, Adilia M. Zukhurova, Matvei V. Urvantsev, Semen A. Osipenkov, Igor S. Efremov, Georgii E. Bondar - [[pdf]](https://arxiv.org/pdf/2601.06508)</summary>

**Abstract:** The integration of autonomous unmanned aerial vehicles (UAVs) into large-scale artistic projects has emerged as a new application in robotics. This paper presents the design, deployment, and testing of a novel multi-drone system for automated mural painting in outdoor settings. This technology makes use of new software that coordinates multiple drones simultaneously, utilizing state-machine algorithms for task execution. Key advancements are the complex positioning system that combines 2D localization using a single motion tracking camera with onboard LiDAR for precise positioning, and a novel flight control algorithm, which works differently along the trajectory and normally to it, ensuring smoothness and high precision of the drawings at the same time. A 100 square meters mural was created using the developed multi-drone system, validating the system's efficacy. Compared to single-drone approaches, our multi-UAV solution significantly improves scalability and operational speed while maintaining high stability even in harsh weather conditions. The findings highlight the potential of autonomous robotic swarms in creative applications, paving the way for further advancements in large-scale robotic art.

**arXiv ID:** 2601.06508
</details>

<details>
<summary><strong>Large-Scale Autonomous Gas Monitoring for Volcanic Environments: A Legged Robot on Mount Etna</strong> - Julia Richter, Turcan Tuna, Manthan Patel, Takahiro Miki, Devon Higgins, James Fox, Cesar Cadena, Andres Diaz, Marco Hutter - [[pdf]](https://arxiv.org/pdf/2601.07362)</summary>

**Abstract:** Volcanic gas emissions are key precursors of eruptive activity. Yet, obtaining accurate near-surface measurements remains hazardous and logistically challenging, motivating the need for autonomous solutions. Limited mobility in rough volcanic terrain has prevented wheeled systems from performing reliable in situ gas measurements, reducing their usefulness as sensing platforms. We present a legged robotic system for autonomous volcanic gas analysis, utilizing the quadruped ANYmal, equipped with a quadrupole mass spectrometer system. Our modular autonomy stack integrates a mission planning interface, global planner, localization framework, and terrain-aware local navigation. We evaluated the system on Mount Etna across three autonomous missions in varied terrain, achieving successful gas-source detections with autonomy rates of 93-100%. In addition, we conducted a teleoperated mission in which the robot measured natural fumaroles, detecting sulfur dioxide and carbon dioxide. We discuss lessons learned from the gas-analysis and autonomy perspectives, emphasizing the need for adaptive sensing strategies, tighter integration of global and local planning, and improved hardware design.

**arXiv ID:** 2601.07362
</details>

<details>
<summary><strong>LOONG: Online Time-Optimal Autonomous Flight for MAVs in Cluttered Environments</strong> - Xin Guan, Fangguo Zhao, Qianyi Wang, Chengcheng Zhao, Jiming Chen, Shuo Li - [[pdf]](https://arxiv.org/pdf/2601.07434)</summary>

**Abstract:** Autonomous flight of micro air vehicles (MAVs) in unknown, cluttered environments remains challenging for time-critical missions due to conservative maneuvering strategies. This article presents an integrated planning and control framework for high-speed, time-optimal autonomous flight of MAVs in cluttered environments. In each replanning cycle (100 Hz), a time-optimal trajectory under polynomial presentation is generated as a reference, with the time-allocation process accelerated by imitation learning. Subsequently, a time-optimal model predictive contouring control (MPCC) incorporates safe flight corridor (SFC) constraints at variable horizon steps to enable aggressive yet safe maneuvering, while fully exploiting the MAV's dynamics. We validate the proposed framework extensively on a custom-built LiDAR-based MAV platform. Simulation results demonstrate superior aggressiveness compared to the state of the art, while real-world experiments achieve a peak speed of 18 m/s in a cluttered environment and succeed in 10 consecutive trials from diverse start points. The video is available at the following link: this https URL.

**arXiv ID:** 2601.07434
</details>

<details>
<summary><strong>NanoCockpit: Performance-optimized Application Framework for AI-based Autonomous Nanorobotics</strong> - Elia Cereda, Alessandro Giusti, Daniele Palossi - [[pdf]](https://arxiv.org/pdf/2601.07476)</summary>

**Abstract:** Autonomous nano-drones, powered by vision-based tiny machine learning (TinyML) models, are a novel technology gaining momentum thanks to their broad applicability and pushing scientific advancement on resource-limited embedded systems. Their small form factor, i.e., a few 10s grams, severely limits their onboard computational resources to sub-\SI{100}{\milli\watt} microcontroller units (MCUs). The Bitcraze Crazyflie nano-drone is the \textit{de facto} standard, offering a rich set of programmable MCUs for low-level control, multi-core processing, and radio transmission. However, roboticists very often underutilize these onboard precious resources due to the absence of a simple yet efficient software layer capable of time-optimal pipelining of multi-buffer image acquisition, multi-core computation, intra-MCUs data exchange, and Wi-Fi streaming, leading to sub-optimal control performances. Our \textit{NanoCockpit} framework aims to fill this gap, increasing the throughput and minimizing the system's latency, while simplifying the developer experience through coroutine-based multi-tasking. In-field experiments on three real-world TinyML nanorobotics applications show our framework achieves ideal end-to-end latency, i.e. zero overhead due to serialized tasks, delivering quantifiable improvements in closed-loop control performance ($-$30\% mean position error, mission success rate increased from 40\% to 100\%).

**arXiv ID:** 2601.07476
</details>

<details>
<summary><strong>FlyCo: Foundation Model-Empowered Drones for Autonomous 3D Structure Scanning in Open-World Environments</strong> - Chen Feng, Guiyong Zheng, Tengkai Zhuang, Yongqian Wu, Fangzhan He, Haojia Li, Juepeng Zheng, Shaojie Shen, Boyu Zhou - [[pdf]](https://arxiv.org/pdf/2601.07558)</summary>

**Abstract:** Autonomous 3D scanning of open-world target structures via drones remains challenging despite broad applications. Existing paradigms rely on restrictive assumptions or effortful human priors, limiting practicality, efficiency, and adaptability. Recent foundation models (FMs) offer great potential to bridge this gap. This paper investigates a critical research problem: What system architecture can effectively integrate FM knowledge for this task? We answer it with FlyCo, a principled FM-empowered perception-prediction-planning loop enabling fully autonomous, prompt-driven 3D target scanning in diverse unknown open-world environments. FlyCo directly translates low-effort human prompts (text, visual annotations) into precise adaptive scanning flights via three coordinated stages: (1) perception fuses streaming sensor data with vision-language FMs for robust target grounding and tracking; (2) prediction distills FM knowledge and combines multi-modal cues to infer the partially observed target's complete geometry; (3) planning leverages predictive foresight to generate efficient and safe paths with comprehensive target coverage. Building on this, we further design key components to boost open-world target grounding efficiency and robustness, enhance prediction quality in terms of shape accuracy, zero-shot generalization, and temporal stability, and balance long-horizon flight efficiency with real-time computability and online collision avoidance. Extensive challenging real-world and simulation experiments show FlyCo delivers precise scene understanding, high efficiency, and real-time safety, outperforming existing paradigms with lower human effort and verifying the proposed architecture's practicality. Comprehensive ablations validate each component's contribution. FlyCo also serves as a flexible, extensible blueprint, readily leveraging future FM and robotics advances. Code will be released.

**arXiv ID:** 2601.07558
</details>

<details>
<summary><strong>AURA-CVC: Autonomous Ultrasound-guided Robotic Assistance for Central Venous Catheterization</strong> - Deepak Raina, Lidia Al-Zogbi, Brian Teixeira, Vivek Singh, Ankur Kapoor, Thorsten Fleiter, Muyinatu A. Lediju Bell, Vinciya Pandian, Axel Krieger - [[pdf]](https://arxiv.org/pdf/2507.05979)</summary>

**Abstract:** Purpose: Central venous catheterization (CVC) is a critical medical procedure for vascular access, hemodynamic monitoring, and life-saving interventions. Its success remains challenging due to the need for continuous ultrasound-guided visualization of a target vessel and approaching needle, which is further complicated by anatomical variability and operator dependency. Errors in needle placement can lead to life-threatening complications. While robotic systems offer a potential solution, achieving full autonomy remains challenging. In this work, we propose an end-to-end robotic-ultrasound-guided CVC pipeline, from scan initialization to needle insertion. Methods: We introduce a deep-learning model to identify clinically relevant anatomical landmarks from a depth image of the patient's neck, obtained using RGB-D camera, to autonomously define the scanning region and paths. Then, a robot motion planning framework is proposed to scan, segment, reconstruct, and localize vessels (veins and arteries), followed by the identification of the optimal insertion zone. Finally, a needle guidance module plans the insertion under ultrasound guidance with operator's feedback. This pipeline was validated on a high-fidelity commercial phantom across 10 simulated clinical scenarios. Results: The proposed pipeline achieved 10 out of 10 successful needle placements on the first attempt. Vessels were reconstructed with a mean error of 2.15 \textit{mm}, and autonomous needle insertion was performed with an error less than or close to 1 \textit{mm}. Conclusion: To our knowledge, this is the first robotic CVC system demonstrated on a high-fidelity phantom with integrated planning, scanning, and insertion. Experimental results show its potential for clinical translation.

**arXiv ID:** 2507.05979
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>ET-Agent: Incentivizing Effective Tool-Integrated Reasoning Agent via Behavior Calibration</strong> - Yifei Chen, Guanting Dong, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2601.06860)</summary>

**Abstract:** Large Language Models (LLMs) can extend their parameter knowledge limits by adopting the Tool-Integrated Reasoning (TIR) paradigm. However, existing LLM-based agent training framework often focuses on answers' accuracy, overlooking specific alignment for behavior patterns. Consequently, agent often exhibits ineffective actions during TIR tasks, such as redundant and insufficient tool calls. How to calibrate erroneous behavioral patterns when executing TIR tasks, thereby exploring effective trajectories, remains an open-ended problem. In this paper, we propose ET-Agent, a training framework for calibrating agent's tool-use behavior through two synergistic perspectives: Self-evolving Data Flywheel and Behavior Calibration Training. Specifically, we introduce a self-evolutionary data flywheel to generate enhanced data, used to fine-tune LLM to improve its exploration ability. Based on this, we implement an two-phases behavior-calibration training framework. It is designed to progressively calibrate erroneous behavioral patterns to optimal behaviors. Further in-depth experiments confirm the superiority of \ourmodel{} across multiple dimensions, including correctness, efficiency, reasoning conciseness, and tool execution accuracy. Our ET-Agent framework provides practical insights for research in the TIR field. Codes can be found in this https URL

**arXiv ID:** 2601.06860
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (37 papers)</h2></summary>

<details>
<summary><strong>LSRIF: Logic-Structured Reinforcement Learning for Instruction Following</strong> - Qingyu Ren, Qianyu He, Jingwen Chang, Jie Zeng, Jiaqing Liang, Yanghua Xiao, Han Xia, Zeye Sun, Fei Yu - [[pdf]](https://arxiv.org/pdf/2601.06431)</summary>

**Abstract:** Instruction-following is critical for large language models, but real-world instructions often contain logical structures such as sequential dependencies and conditional branching. Existing methods typically construct datasets with parallel constraints and optimize average rewards, ignoring logical dependencies and yielding noisy signals. We propose a logic-structured training framework LSRIF that explicitly models instruction logic. We first construct a dataset LSRInstruct with constraint structures such as parallel, sequential, and conditional types, and then design structure-aware rewarding method LSRIF including average aggregation for parallel structures, failure-penalty propagation for sequential structures, and selective rewards for conditional branches. Experiments show LSRIF brings significant improvements in instruction-following (in-domain and out-of-domain) and general reasoning. Analysis reveals that learning with explicit logic structures brings parameter updates in attention layers and sharpens token-level attention to constraints and logical operators.

**arXiv ID:** 2601.06431
</details>

<details>
<summary><strong>DRAGON: LLM-Driven Decomposition and Reconstruction Agents for Large-Scale Combinatorial Optimization</strong> - Shengkai Chen, Zhiguang Cao, Jianan Zhou, Yaoxin Wu, Senthilnath Jayavelu, Zhuoyi Lin, Xiaoli Li, Shili Xiang - [[pdf]](https://arxiv.org/pdf/2601.06502)</summary>

**Abstract:** Large Language Models (LLMs) have recently shown promise in addressing combinatorial optimization problems (COPs) through prompt-based strategies. However, their scalability and generalization remain limited, and their effectiveness diminishes as problem size increases, particularly in routing problems involving more than 30 nodes. We propose DRAGON, which stands for Decomposition and Reconstruction Agents Guided OptimizatioN, a novel framework that combines the strengths of metaheuristic design and LLM reasoning. Starting from an initial global solution, DRAGON autonomously identifies regions with high optimization potential and strategically decompose large-scale COPs into manageable subproblems. Each subproblem is then reformulated as a concise, localized optimization task and solved through targeted LLM prompting guided by accumulated experiences. Finally, the locally optimized solutions are systematically reintegrated into the original global context to yield a significantly improved overall outcome. By continuously interacting with the optimization environment and leveraging an adaptive experience memory, the agents iteratively learn from feedback, effectively coupling symbolic reasoning with heuristic search. Empirical results show that, unlike existing LLM-based solvers limited to small-scale instances, DRAGON consistently produces feasible solutions on TSPLIB, CVRPLIB, and Weibull-5k bin packing benchmarks, and achieves near-optimal results (0.16% gap) on knapsack problems with over 3M variables. This work shows the potential of feedback-driven language agents as a new paradigm for generalizable and interpretable large-scale optimization.

**arXiv ID:** 2601.06502
</details>

<details>
<summary><strong>GDEPO: Group Dual-dynamic and Equal-right-advantage Policy Optimization with Enhanced Training Data Utilization for Sample-Constrained Reinforcement Learning</strong> - Zhengqing Yan, Xinyang Liu, Yi Zhang, Fan Guo, Yao Liu, Junchen Wan, Kang Song - [[pdf]](https://arxiv.org/pdf/2601.06795)</summary>

**Abstract:** Automated Theorem Proving (ATP) represents a fundamental challenge in Artificial Intelligence (AI), requiring the construction of machine-verifiable proofs in formal languages such as Lean to evaluate AI reasoning capabilities. Reinforcement learning (RL), particularly the high-performance Group Relative Policy Optimization (GRPO) algorithm, has emerged as a mainstream approach for this task. However, in ATP scenarios, GRPO faces two critical issues: when composite rewards are used, its relative advantage estimation may conflict with the binary feedback from the formal verifier; meanwhile, its static sampling strategy may discard entire batches of data if no valid proof is found, resulting in zero contribution to model updates and significant data waste. To address these limitations, we propose Group Dual-dynamic and Equal-right-advantage Policy Optimization (GDEPO), a method incorporating three core mechanisms: 1) dynamic additional sampling, which resamples invalid batches until a valid proof is discovered; 2) equal-right advantage, decoupling the sign of the advantage function (based on correctness) from its magnitude (modulated by auxiliary rewards) to ensure stable and correct policy updates; and 3) dynamic additional iterations, applying extra gradient steps to initially failed but eventually successful samples to accelerate learning on challenging cases. Experiments conducted on three datasets of varying difficulty (MinF2F-test, MathOlympiadBench, PutnamBench) confirm the effectiveness of GDEPO, while ablation studies validate the necessity of its synergistic components. The proposed method enhances data utilization and optimization efficiency, offering a novel training paradigm for ATP.

**arXiv ID:** 2601.06795
</details>

<details>
<summary><strong>Thinking with Deltas: Incentivizing Reinforcement Learning via Differential Visual Reasoning Policy</strong> - Shujian Gao, Yuan Wang, Jiangtao Yan, Zuxuan Wu, Yu-Gang Jiang - [[pdf]](https://arxiv.org/pdf/2601.06801)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced reasoning capabilities in Large Language Models. However, adapting RLVR to multimodal domains suffers from a critical \textit{perception-reasoning decoupling}. Existing paradigms, driven by text-centric outcome rewards, reasoning in language medium, inadvertently encourage models to bypass visual perception. We empirically validate this through blind experiments: state-of-the-art policies maintain or surprisingly improve performance even when visual inputs are entirely removed. This reveals that these models degenerate into \textit{blind reasoners}, exploiting linguistic priors to generate plausible answers instead of attending to visual evidence. In response, we propose \textbf{Thinking with Deltas}, a framework driven by a \textbf{Differential Visual Reasoning Policy (DVRP)}. DVRP introduces intrinsic supervision via visual triplets, comprising original, masked, and perturbed inputs. It optimizes the model to maximize reasoning divergence from masked inputs (enforcing \textit{visual sensitivity}) while minimizing divergence from perturbed inputs (ensuring \textit{visual robustness}). By aligning reasoning variations strictly with the \textit{Delta} of visual information, DVRP inherently bolsters visual understanding capabilities and significantly outperforms state-of-the-art methods on both general and medical benchmarks, without requiring external annotations or auxiliary tools.

**arXiv ID:** 2601.06801
</details>

<details>
<summary><strong>Dr. Zero: Self-Evolving Search Agents without Training Data</strong> - Zhenrui Yue, Kartikeya Upasani, Xianjun Yang, Suyu Ge, Shaoliang Nie, Yuning Mao, Zhe Liu, Dong Wang - [[pdf]](https://arxiv.org/pdf/2601.07055)</summary>

**Abstract:** As high-quality data becomes increasingly difficult to obtain, data-free self-evolution has emerged as a promising paradigm. This approach allows large language models (LLMs) to autonomously generate and solve complex problems, thereby improving their reasoning capabilities. However, multi-turn search agents struggle in data-free self-evolution due to the limited question diversity and the substantial compute required for multi-step reasoning and tool using. In this work, we introduce Dr. Zero, a framework enabling search agents to effectively self-evolve without any training data. In particular, we design a self-evolution feedback loop where a proposer generates diverse questions to train a solver initialized from the same base model. As the solver evolves, it incentivizes the proposer to produce increasingly difficult yet solvable tasks, thus establishing an automated curriculum to refine both agents. To enhance training efficiency, we also introduce hop-grouped relative policy optimization (HRPO). This method clusters structurally similar questions to construct group-level baselines, effectively minimizing the sampling overhead in evaluating each query's individual difficulty and solvability. Consequently, HRPO significantly reduces the compute requirements for solver training without compromising performance or stability. Extensive experiment results demonstrate that the data-free Dr. Zero matches or surpasses fully supervised search agents, proving that complex reasoning and search capabilities can emerge solely through self-evolution.

**arXiv ID:** 2601.07055
</details>

<details>
<summary><strong>Rewarding Creativity: A Human-Aligned Generative Reward Model for Reinforcement Learning in Storytelling</strong> - Zhaoyan Li, Hang Lei, Yujia Wang, Lanbo Liu, Hao Liu, Liang Yu - [[pdf]](https://arxiv.org/pdf/2601.07149)</summary>

**Abstract:** While Large Language Models (LLMs) can generate fluent text, producing high-quality creative stories remains challenging. Reinforcement Learning (RL) offers a promising solution but faces two critical obstacles: designing reliable reward signals for subjective storytelling quality and mitigating training instability. This paper introduces the Reinforcement Learning for Creative Storytelling (RLCS) framework to systematically address both challenges. First, we develop a Generative Reward Model (GenRM) that provides multi-dimensional analysis and explicit reasoning about story preferences, trained through supervised fine-tuning on demonstrations with reasoning chains distilled from strong teacher models, followed by GRPO-based refinement on expanded preference data. Second, we introduce an entropy-based reward shaping strategy that dynamically prioritizes learning on confident errors and uncertain correct predictions, preventing overfitting on already-mastered patterns. Experiments demonstrate that GenRM achieves 68\% alignment with human creativity judgments, and RLCS significantly outperforms strong baselines including Gemini-2.5-Pro in overall story quality. This work provides a practical pipeline for applying RL to creative domains, effectively navigating the dual challenges of reward modeling and training stability.

**arXiv ID:** 2601.07149
</details>

<details>
<summary><strong>LRAS: Advanced Legal Reasoning with Agentic Search</strong> - Yujin Zhou, Chuxue Cao, Jinluan Yang, Lijun Wu, Conghui He, Sirui Han, Yike Guo - [[pdf]](https://arxiv.org/pdf/2601.07296)</summary>

**Abstract:** While Large Reasoning Models (LRMs) have demonstrated exceptional logical capabilities in mathematical domains, their application to the legal field remains hindered by the strict requirements for procedural rigor and adherence to legal logic. Existing legal LLMs, which rely on "closed-loop reasoning" derived solely from internal parametric knowledge, frequently suffer from lack of self-awareness regarding their knowledge boundaries, leading to confident yet incorrect conclusions. To address this challenge, we present Legal Reasoning with Agentic Search (LRAS), the first framework designed to transition legal LLMs from static and parametric "closed-loop thinking" to dynamic and interactive "Active Inquiry". By integrating Introspective Imitation Learning and Difficulty-aware Reinforcement Learning, LRAS enables LRMs to identify knowledge boundaries and handle legal reasoning complexity. Empirical results demonstrate that LRAS outperforms state-of-the-art baselines by 8.2-32\%, with the most substantial gains observed in tasks requiring deep reasoning with reliable knowledge. We will release our data and models for further exploration soon.

**arXiv ID:** 2601.07296
</details>

<details>
<summary><strong>TeleMem: Building Long-Term and Multimodal Memory for Agentic AI</strong> - Chunliang Chen, Ming Guan, Xiao Lin, Jiaxu Li, Qiyi Wang, Xiangyu Chen, Jixiang Luo, Changzhi Sun, Dell Zhang, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2601.06037)</summary>

**Abstract:** Large language models (LLMs) excel at many NLP tasks but struggle to sustain long-term interactions due to limited attention over extended dialogue histories. Retrieval-augmented generation (RAG) mitigates this issue but lacks reliable mechanisms for updating or refining stored memories, leading to schema-driven hallucinations, inefficient write operations, and minimal support for multimodal this http URL address these challenges, we propose TeleMem, a unified long-term and multimodal memory system that maintains coherent user profiles through narrative dynamic extraction, ensuring that only dialogue-grounded information is preserved. TeleMem further introduces a structured writing pipeline that batches, retrieves, clusters, and consolidates memory entries, substantially improving storage efficiency, reducing token usage, and accelerating memory operations. Additionally, a multimodal memory module combined with ReAct-style reasoning equips the system with a closed-loop observe, think, and act process that enables accurate understanding of complex video content in long-term contexts. Experimental results show that TeleMem surpasses the state-of-the-art Mem0 baseline with 19% higher accuracy, 43% fewer tokens, and a 2.1x speedup on the ZH-4O long-term role-play gaming benchmark.

**arXiv ID:** 2601.06037
</details>

<details>
<summary><strong>Reinforcement Learning for Chain of Thought Compression with One-Domain-to-All Generalization</strong> - Hanyu Li, Jiangshan Duo, Bofei Gao, Hailin Zhang, Sujian Li, Xiaotie Deng, Liang Zhao - [[pdf]](https://arxiv.org/pdf/2601.06052)</summary>

**Abstract:** Chain-of-thought reasoning in large language models often creates an "overthinking trap," leading to excessive computational cost and latency for unreliable accuracy gains. Prior work has typically relied on global, static controls that risk penalizing necessary reasoning. We introduce a sample-level, soft reinforcement learning compression method that penalizes inefficiently long rollouts, but only on problems where the model has already mastered and already produced a more concise rollout. Our experiments show that this method reduces average response length by 20-40% with comparable or higher accuracy. Crucially, the compression exhibits strong cross-domain generalization; a model trained on math spontaneously shortens responses on unseen tasks like code, instruction following, and general knowledge QA, with stable or improved accuracy. We demonstrate a stable post-training curriculum (accuracy-compression-accuracy) that can ultimately produce models that are more accurate and reason more concisely, arguing that such compression method should be a standard phase in developing efficient reasoning models.

**arXiv ID:** 2601.06052
</details>

<details>
<summary><strong>COVR:Collaborative Optimization of VLMs and RL Agent for Visual-Based Control</strong> - Canming Xia, Peixi Peng, Guang Tan, Zhan Su, Haoran Xu, Zhenxian Liu, Luntong Li - [[pdf]](https://arxiv.org/pdf/2601.06122)</summary>

**Abstract:** Visual reinforcement learning (RL) suffers from poor sample efficiency due to high-dimensional observations in complex tasks. While existing works have shown that vision-language models (VLMs) can assist RL, they often focus on knowledge distillation from the VLM to RL, overlooking the potential of RL-generated interaction data to enhance the VLM. To address this, we propose COVR, a collaborative optimization framework that enables the mutual enhancement of the VLM and RL policies. Specifically, COVR fine-tunes the VLM with RL-generated data to enhance the semantic reasoning ability consistent with the target task, and uses the enhanced VLM to further guide policy learning via action priors. To improve fine-tuning efficiency, we introduce two key modules: (1) an Exploration-Driven Dynamic Filter module that preserves valuable exploration samples using adaptive thresholds based on the degree of exploration, and (2) a Return-Aware Adaptive Loss Weight module that improves the stability of training by quantifying the inconsistency of sampling actions via return signals of RL. We further design a progressive fine-tuning strategy to reduce resource consumption. Extensive experiments show that COVR achieves strong performance across various challenging visual control tasks.

**arXiv ID:** 2601.06122
</details>

<details>
<summary><strong>Reinforcement Learning-Guided Dynamic Multi-Graph Fusion for Evacuation Traffic Prediction</strong> - Md Nafees Fuad Rafi, Samiul Hasan - [[pdf]](https://arxiv.org/pdf/2601.06664)</summary>

**Abstract:** Real-time traffic prediction is critical for managing transportation systems during hurricane evacuations. Although data-driven graph-learning models have demonstrated strong capabilities in capturing the complex spatiotemporal dynamics of evacuation traffic at a network level, they mostly consider a single dimension (e.g., travel-time or distance) to construct the underlying graph. Furthermore, these models often lack interpretability, offering little insight into which input variables contribute most to their predictive performance. To overcome these limitations, we develop a novel Reinforcement Learning-guided Dynamic Multi-Graph Fusion (RL-DMF) framework for evacuation traffic prediction. We construct multiple dynamic graphs at each time step to represent heterogeneous spatiotemporal relationships between traffic detectors. A dynamic multi-graph fusion (DMF) module is employed to adaptively learn and combine information from these graphs. To enhance model interpretability, we introduce RL-based intelligent feature selection and ranking (RL-IFSR) method that learns to mask irrelevant features during model training. The model is evaluated using a real-world dataset of 12 hurricanes affecting Florida from 2016 to 2024. For an unseen hurricane (Milton, 2024), the model achieves a 95% accuracy (RMSE = 293.9) for predicting the next 1-hour traffic flow. Moreover, the model can forecast traffic flow for up to next 6 hours with 90% accuracy (RMSE = 426.4). The RL-DMF framework outperforms several state-of-the-art traffic prediction models. Furthermore, ablation experiments confirm the effectiveness of dynamic multi-graph fusion and RL-IFSR approaches for improving model performance. This research provides a generalized and interpretable model for real-time evacuation traffic forecasting, with significant implications for evacuation traffic management.

**arXiv ID:** 2601.06664
</details>

<details>
<summary><strong>Bi-Mem: Bidirectional Construction of Hierarchical Memory for Personalized LLMs via Inductive-Reflective Agents</strong> - Wenyu Mao, Haosong Tan, Shuchang Liu, Haoyang Liu, Yifan Xu, Huaxiang Ji, Xiang Wang - [[pdf]](https://arxiv.org/pdf/2601.06490)</summary>

**Abstract:** Constructing memory from users' long-term conversations overcomes LLMs' contextual limitations and enables personalized interactions. Recent studies focus on hierarchical memory to model users' multi-granular behavioral patterns via clustering and aggregating historical conversations. However, conversational noise and memory hallucinations can be amplified during clustering, causing locally aggregated memories to misalign with the user's global persona. To mitigate this issue, we propose Bi-Mem, an agentic framework ensuring hierarchical memory fidelity through bidirectional construction. Specifically, we deploy an inductive agent to form the hierarchical memory: it extracts factual information from raw conversations to form fact-level memory, aggregates them into thematic scenes (i.e., local scene-level memory) using graph clustering, and infers users' profiles as global persona-level memory. Simultaneously, a reflective agent is designed to calibrate local scene-level memories using global constraints derived from the persona-level memory, thereby enforcing global-local alignment. For coherent memory recall, we propose an associative retrieval mechanism: beyond initial hierarchical search, a spreading activation process allows facts to evoke contextual scenes, while scene-level matches retrieve salient supporting factual information. Empirical evaluations demonstrate that Bi-Mem achieves significant improvements in question answering performance on long-term personalized conversational tasks.

**arXiv ID:** 2601.06490
</details>

<details>
<summary><strong>Predefined-time One-Shot Cooperative Estimation, Guidance, and Control for Simultaneous Target Interception</strong> - Lohitvel Gopikannan, Shashi Ranjan Kumar, Abhinav Sinha - [[pdf]](https://arxiv.org/pdf/2601.07744)</summary>

**Abstract:** This work develops a unified nonlinear estimation-guidance-control framework for cooperative simultaneous interception of a stationary target under a heterogeneous sensing topology, where sensing capabilities are non-uniform across interceptors. Specifically, only a subset of agents is instrumented with onboard seekers (informed/seeker-equipped agents), whereas the rest of them (seeker-less agents) acquire the information about the target indirectly via the informed agents and execute a distributed cooperative guidance for simultaneous target interception. To address the resulting partial observability, a predefined-time distributed observer is leveraged, guaranteeing convergence of the target state estimates for seeker-less agents through information exchange with seeker-equipped neighbors over a directed communication graph. Thereafter, an improved time-to-go estimate accounting for wide launch envelopes is utilized to design the distributed cooperative guidance commands. This estimate is coupled with a predefined-time consensus protocol, ensuring consensus in the agents' time-to-go values. The temporal upper bounds within which both observer error and time-to-go consensus error converge to zero can be prescribed as design parameters. Furthermore, the cooperative guidance commands are realized by means of an autopilot, wherein the interceptor is steered by canard actuation. The corresponding fin deflection commands are generated using a predefined-time convergent sliding mode control law. This enables the autopilot to precisely track the commanded lateral acceleration within a design-specified time, while maintaining non-singularity of the overall design. Theoretical guarantees are supported by numerical simulations across diverse engagement geometries, verifying the estimation accuracy, the cooperative interception performance, and the autopilot response using the proposed scheme.

**arXiv ID:** 2601.07744
</details>

<details>
<summary><strong>Spec-o3: A Tool-Augmented Vision-Language Agent for Rare Celestial Object Candidate Vetting via Automated Spectral Inspection</strong> - Minghui Jia, Qichao Zhang, Ali Luo, Linjing Li, Shuo Ye, Hailing Lu, Wen Hou, Dongbin Zhao - [[pdf]](https://arxiv.org/pdf/2601.06498)</summary>

**Abstract:** Due to the limited generalization and interpretability of deep learning classifiers, The final vetting of rare celestial object candidates still relies on expert visual inspection--a manually intensive process. In this process, astronomers leverage specialized tools to analyze spectra and construct reliable catalogs. However, this practice has become the primary bottleneck, as it is fundamentally incapable of scaling with the data deluge from modern spectroscopic surveys. To bridge this gap, we propose Spec-o3, a tool-augmented vision-language agent that performs astronomer-aligned spectral inspection via interleaved multimodal chain-of-thought reasoning. Spec-o3 is trained with a two-stage post-training recipe: cold-start supervised fine-tuning on expert inspection trajectories followed by outcome-based reinforcement learning on rare-type verification tasks. Evaluated on five rare-object identification tasks from LAMOST, Spec-o3 establishes a new State-of-the-Art, boosting the macro-F1 score from 28.3 to 76.5 with a 7B parameter base model and outperforming both proprietary VLMs and specialized deep models. Crucially, the agent demonstrates strong generalization to unseen inspection tasks across survey shifts (from LAMOST to SDSS/DESI). Expert evaluations confirm that its reasoning traces are coherent and physically consistent, supporting transparent and trustworthy decision-making. Code, data, and models are available at \href{this https URL}{Project HomePage}.

**arXiv ID:** 2601.06498
</details>

<details>
<summary><strong>AgentHallu: Benchmarking Automated Hallucination Attribution of LLM-based Agents</strong> - Xuannan Liu, Xiao Yang, Zekun Li, Peipei Li, Ran He - [[pdf]](https://arxiv.org/pdf/2601.06818)</summary>

**Abstract:** As LLM-based agents operate over sequential multi-step reasoning, hallucinations arising at intermediate steps risk propagating along the trajectory, thus degrading overall reliability. Unlike hallucination detection in single-turn responses, diagnosing hallucinations in multi-step workflows requires identifying which step causes the initial divergence. To fill this gap, we propose a new research task, automated hallucination attribution of LLM-based agents, aiming to identify the step responsible for the hallucination and explain why. To support this task, we introduce AgentHallu, a comprehensive benchmark with: (1) 693 high-quality trajectories spanning 7 agent frameworks and 5 domains, (2) a hallucination taxonomy organized into 5 categories (Planning, Retrieval, Reasoning, Human-Interaction, and Tool-Use) and 14 sub-categories, and (3) multi-level annotations curated by humans, covering binary labels, hallucination-responsible steps, and causal explanations. We evaluate 13 leading models, and results show the task is challenging even for top-tier models (like GPT-5, Gemini-2.5-Pro). The best-performing model achieves only 41.1\% step localization accuracy, where tool-use hallucinations are the most challenging at just 11.6\%. We believe AgentHallu will catalyze future research into developing robust, transparent, and reliable agentic systems.

**arXiv ID:** 2601.06818
</details>

<details>
<summary><strong>TreePS-RAG: Tree-based Process Supervision for Reinforcement Learning in Agentic RAG</strong> - Tianhua Zhang, Kun Li, Junan Li, Yunxiang Li, Hongyin Luo, Xixin Wu, James Glass, Helen Meng - [[pdf]](https://arxiv.org/pdf/2601.06922)</summary>

**Abstract:** Agentic retrieval-augmented generation (RAG) formulates question answering as a multi-step interaction between reasoning and information retrieval, and has recently been advanced by reinforcement learning (RL) with outcome-based supervision. While effective, relying solely on sparse final rewards limits step-wise credit assignment and provides weak guidance for intermediate reasoning and actions. Recent efforts explore process-level supervision, but typically depend on offline constructed training data, which risks distribution shift, or require costly intermediate annotations. We present TreePS-RAG, an online, tree-based RL framework for agentic RAG that enables step-wise credit assignment while retaining standard outcome-only rewards. Our key insight is to model agentic RAG reasoning as a rollout tree, where each reasoning step naturally maps to a node. This tree structure allows step utility to be estimated via Monte Carlo estimation over its descendant outcomes, yielding fine-grained process advantages without requiring intermediate labels. To make this paradigm practical, we introduce an efficient online tree construction strategy that preserves exploration diversity under a constrained computational budget. With a rollout cost comparable to strong baselines like Search-R1, experiments on seven multi-hop and general QA benchmarks across multiple model scales show that TreePS-RAG consistently and significantly outperforms both outcome-supervised and leading process-supervised RL methods.

**arXiv ID:** 2601.06922
</details>

<details>
<summary><strong>ReinPool: Reinforcement Learning Pooling Multi-Vector Embeddings for Retrieval System</strong> - Sungguk Cha, DongWook Kim, Mintae Kim, Youngsub Han, Byoung-Ki Jeon, Sangyeob Lee - [[pdf]](https://arxiv.org/pdf/2601.07125)</summary>

**Abstract:** Multi-vector embedding models have emerged as a powerful paradigm for document retrieval, preserving fine-grained visual and textual details through token-level representations. However, this expressiveness comes at a staggering cost: storing embeddings for every token inflates index sizes by over $1000\times$ compared to single-vector approaches, severely limiting scalability. We introduce \textbf{ReinPool}, a reinforcement learning framework that learns to dynamically filter and pool multi-vector embeddings into compact, retrieval-optimized representations. By training with an inverse retrieval objective and NDCG-based rewards, ReinPool identifies and retains only the most discriminative vectors without requiring manual importance annotations. On the Vidore V2 benchmark across three vision-language embedding models, ReinPool compresses multi-vector representations by $746$--$1249\times$ into single vectors while recovering 76--81\% of full multi-vector retrieval performance. Compared to static mean pooling baselines, ReinPool achieves 22--33\% absolute NDCG@3 improvement, demonstrating that learned selection significantly outperforms heuristic aggregation.

**arXiv ID:** 2601.07125
</details>

<details>
<summary><strong>Agent Bain vs. Agent McKinsey: A New Text-to-SQL Benchmark for the Business Domain</strong> - Yue Li, Ran Tao, Derek Hommel, Yusuf Denizay Dönder, Sungyong Chang, David Mimno, Unso Eun Seo Jo - [[pdf]](https://arxiv.org/pdf/2510.07309)</summary>

**Abstract:** Text-to-SQL benchmarks have traditionally only tested simple data access as a translation task of natural language to SQL queries. But in reality, users tend to ask diverse questions that require more complex responses including data-driven predictions or recommendations. Using the business domain as a motivating example, we introduce CORGI, a new benchmark that expands text-to-SQL to reflect practical database queries encountered by end users. CORGI is composed of synthetic databases inspired by enterprises such as DoorDash, Airbnb, and Lululemon. It provides questions across four increasingly complicated categories of business queries: descriptive, explanatory, predictive, and recommendational. This challenge calls for causal reasoning, temporal forecasting, and strategic recommendation, reflecting multi-level and multi-step agentic intelligence. We find that LLM performance degrades on higher-level questions as question complexity increases. CORGI also introduces and encourages the text-to-SQL community to consider new automatic methods for evaluating open-ended, qualitative responses in data access tasks. Our experiments show that LLMs exhibit an average 33.12% lower success execution rate (SER) on CORGI compared to existing benchmarks such as BIRD, highlighting the substantially higher complexity of real-world business needs. We release the CORGI dataset, an evaluation framework, and a submission website to support future research.

**arXiv ID:** 2510.07309
</details>

<details>
<summary><strong>Instructions are all you need: Self-supervised Reinforcement Learning for Instruction Following</strong> - Qingyu Ren, Qianyu He, Bowei Zhang, Jie Zeng, Jiaqing Liang, Yanghua Xiao, Weikang Zhou, Zeye Sun, Fei Yu - [[pdf]](https://arxiv.org/pdf/2510.14420)</summary>

**Abstract:** Language models often struggle to follow multi-constraint instructions that are crucial for real-world applications. Existing reinforcement learning (RL) approaches suffer from dependency on external supervision and sparse reward signals from multi-constraint tasks. We propose a label-free self-supervised RL framework that eliminates dependency on external supervision by deriving reward signals directly from instructions and generating pseudo-labels for reward model training. Our approach introduces constraint decomposition strategies and efficient constraint-wise binary classification to address sparse reward challenges while maintaining computational efficiency. Experiments show that our approach generalizes well, achieving strong improvements across 3 in-domain and 5 out-of-domain datasets, including challenging agentic and multi-turn instruction following. The data and code are publicly available at this https URL

**arXiv ID:** 2510.14420
</details>

<details>
<summary><strong>GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning</strong> - Jinchang Luo, Mingquan Cheng, Fan Wan, Ni Li, Xiaoling Xia, Shuangshuang Tian, Tingcheng Bian, Haiwei Wang, Haohuan Fu, Yan Tao - [[pdf]](https://arxiv.org/pdf/2510.20548)</summary>

**Abstract:** Reinforcement learning has recently shown promise in improving retrieval-augmented generation (RAG). Despite these advances, its effectiveness in multi-hop question answering (QA) remains limited by two fundamental limitations: (i) global planning absence to structure multi-step reasoning, and (ii) unfaithful execution, which hinders effective query formulation and consistent use of retrieved evidence. We propose GlobalRAG, a reinforcement learning framework designed to enhance global reasoning in multi-hop QA. GlobalRAG decomposes questions into subgoals, coordinates retrieval with reasoning, and refines evidence iteratively. To guide this process, we introduce Planning Quality Reward and SubGoal Completion Reward, which encourage coherent planning and reliable subgoal execution. In addition, a progressive weight annealing strategy balances process-oriented and outcome-based objectives. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that GlobalRAG significantly outperforms strong baselines while using only 8k training data (42% of the training data used by strong baselines), achieving average improvements of 14.2% in both EM and F1.

**arXiv ID:** 2510.20548
</details>

<details>
<summary><strong>Revisiting Entropy in Reinforcement Learning for Large Reasoning Models</strong> - Renren Jin, Pengzhi Gao, Yuqi Ren, Zhuowen Han, Tongxuan Zhang, Wuwei Huang, Wei Liu, Jian Luan, Deyi Xiong - [[pdf]](https://arxiv.org/pdf/2511.05993)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has emerged as a prominent paradigm for enhancing the reasoning capabilities of large language models (LLMs). However, the entropy of LLMs usually collapses during RLVR training, leading to premature convergence to suboptimal local minima and hindering further performance improvement. Although various approaches have been proposed to mitigate entropy collapse, a comprehensive study of entropy in RLVR remains lacking. To bridge this gap, we conduct extensive experiments to investigate the entropy dynamics of LLMs trained with RLVR and analyze how model entropy correlates with response diversity, calibration, and performance across various benchmarks. Our results identify three key factors that influence entropy: the clipping thresholds in the optimization objective, the number of off-policy updates, and the diversity of the training data. Furthermore, through both theoretical analysis and empirical validation, we demonstrate that tokens with positive advantages are the primary drivers of entropy collapse. Motivated by this insight, we propose Positive-Advantage Reweighting, a simple yet effective approach that regulates model entropy by adjusting the loss weights assigned to tokens with positive advantages during RLVR training, while maintaining competitive performance.

**arXiv ID:** 2511.05993
</details>

<details>
<summary><strong>Reward-Preserving Attacks For Robust Reinforcement Learning</strong> - Lucas Schott, Elies Gherbi, Hatem Hajri, Sylvain Lamprier - [[pdf]](https://arxiv.org/pdf/2601.07118)</summary>

**Abstract:** Adversarial robustness in RL is difficult because perturbations affect entire trajectories: strong attacks can break learning, while weak attacks yield little robustness, and the appropriate strength varies by state. We propose $\alpha$-reward-preserving attacks, which adapt the strength of the adversary so that an $\alpha$ fraction of the nominal-to-worst-case return gap remains achievable at each state. In deep RL, we use a gradient-based attack direction and learn a state-dependent magnitude $\eta\le \eta_{\mathcal B}$ selected via a critic $Q^{\pi}_\alpha((s,a),\eta)$ trained off-policy over diverse radii. This adaptive tuning calibrates attack strength and, with intermediate $\alpha$, improves robustness across radii while preserving nominal performance, outperforming fixed- and random-radius baselines.

**arXiv ID:** 2601.07118
</details>

<details>
<summary><strong>Generating readily synthesizable small molecule fluorophore scaffolds with reinforcement learning</strong> - Ruhi Sayana, Kate Callon, Jennifer Xu, Jonathan Deutsch, Steven Chu, James Zou, John Janetzko, Rabindra V. Shivnaraine, Kyle Swanson - [[pdf]](https://arxiv.org/pdf/2601.07145)</summary>

**Abstract:** Developing new fluorophores for advanced imaging techniques requires exploring new chemical space. While generative AI approaches have shown promise in designing novel dye scaffolds, prior efforts often produced synthetically intractable candidates due to a lack of reaction constraints. Here, we developed SyntheFluor-RL, a generative AI model that employs known reaction libraries and molecular building blocks to create readily synthesizable fluorescent molecule scaffolds via reinforcement learning. To guide the generation of fluorophores, SyntheFluor-RL employs a scoring function built on multiple graph neural networks (GNNs) that predict key photophysical properties, including photoluminescence quantum yield, absorption, and emission wavelengths. These outputs are dynamically weighted and combined with a computed pi-conjugation score to prioritize candidates with desirable optical characteristics and synthetic feasibility. SyntheFluor-RL generated 11,590 candidate molecules, which were filtered to 19 structures predicted to possess dye-like properties. Of the 19 molecules, 14 were synthesized and 13 were experimentally confirmed. The top three were characterized, with the lead compound featuring a benzothiadiazole chromophore and exhibiting strong fluorescence (PLQY = 0.62), a large Stokes shift (97 nm), and a long excited-state lifetime (11.5 ns). These results demonstrate the effectiveness of SyntheFluor-RL in the identification of synthetically accessible fluorophores for further development.

**arXiv ID:** 2601.07145
</details>

<details>
<summary><strong>Offline Meta-Reinforcement Learning with Flow-Based Task Inference and Adaptive Correction of Feature Overgeneralization</strong> - Min Wang, Xin Li, Mingzhong Wang, Hasnaa Bennis - [[pdf]](https://arxiv.org/pdf/2601.07164)</summary>

**Abstract:** Offline meta-reinforcement learning (OMRL) combines the strengths of learning from diverse datasets in offline RL with the adaptability to new tasks of meta-RL, promising safe and efficient knowledge acquisition by RL agents. However, OMRL still suffers extrapolation errors due to out-of-distribution (OOD) actions, compromised by broad task distributions and Markov Decision Process (MDP) ambiguity in meta-RL setups. Existing research indicates that the generalization of the $Q$ network affects the extrapolation error in offline RL. This paper investigates this relationship by decomposing the $Q$ value into feature and weight components, observing that while decomposition enhances adaptability and convergence in the case of high-quality data, it often leads to policy degeneration or collapse in complex tasks. We observe that decomposed $Q$ values introduce a large estimation bias when the feature encounters OOD samples, a phenomenon we term ''feature overgeneralization''. To address this issue, we propose FLORA, which identifies OOD samples by modeling feature distributions and estimating their uncertainties. FLORA integrates a return feedback mechanism to adaptively adjust feature components. Furthermore, to learn precise task representations, FLORA explicitly models the complex task distribution using a chain of invertible transformations. We theoretically and empirically demonstrate that FLORA achieves rapid adaptation and meta-policy improvement compared to baselines across various environments.

**arXiv ID:** 2601.07164
</details>

<details>
<summary><strong>On the Non-decoupling of Supervised Fine-tuning and Reinforcement Learning in Post-training</strong> - Xueyan Niu, Bo Bai, Wei Han, Weixi Zhang - [[pdf]](https://arxiv.org/pdf/2601.07389)</summary>

**Abstract:** Post-training of large language models routinely interleaves supervised fine-tuning (SFT) with reinforcement learning (RL). These two methods have different objectives: SFT minimizes the cross-entropy loss between model outputs and expert responses, while RL maximizes reward signals derived from human preferences or rule-based verifiers. Modern reasoning models have widely adopted the practice of alternating SFT and RL training. However, there is no theoretical account of whether they can be decoupled. We prove that decoupling is impossible in either order: (1) SFT-then-RL coupling: RL increases SFT loss under SFT optimality and (2) RL-then-SFT coupling: SFT lowers the reward achieved by RL. Experiments on Qwen3-0.6B confirm the predicted degradation, verifying that SFT and RL cannot be separated without loss of prior performance in the post-training

**arXiv ID:** 2601.07389
</details>

<details>
<summary><strong>Stagewise Reinforcement Learning and the Geometry of the Regret Landscape</strong> - Chris Elliott, Einar Urdshals, David Quarel, Matthew Farrugia-Roberts, Daniel Murfet - [[pdf]](https://arxiv.org/pdf/2601.07524)</summary>

**Abstract:** Singular learning theory characterizes Bayesian learning as an evolving tradeoff between accuracy and complexity, with transitions between qualitatively different solutions as sample size increases. We extend this theory to deep reinforcement learning, proving that the concentration of the generalized posterior over policies is governed by the local learning coefficient (LLC), an invariant of the geometry of the regret function. This theory predicts that Bayesian phase transitions in reinforcement learning should proceed from simple policies with high regret to complex policies with low regret. We verify this prediction empirically in a gridworld environment exhibiting stagewise policy development: phase transitions over SGD training manifest as "opposing staircases" where regret decreases sharply while the LLC increases. Notably, the LLC detects phase transitions even when estimated on a subset of states where the policies appear identical in terms of regret, suggesting it captures changes in the underlying algorithm rather than just performance.

**arXiv ID:** 2601.07524
</details>

<details>
<summary><strong>Reinforcement Learning for Micro-Level Claims Reserving</strong> - Benjamin Avanzi, Ronald Richman, Bernard Wong, Mario Wüthrich, Yagebu Xie - [[pdf]](https://arxiv.org/pdf/2601.07637)</summary>

**Abstract:** Outstanding claim liabilities are revised repeatedly as claims develop, yet most modern reserving models are trained as one-shot predictors and typically learn only from settled claims. We formulate individual claims reserving as a claim-level Markov decision process in which an agent sequentially updates outstanding claim liability (OCL) estimates over development, using continuous actions and a reward design that balances accuracy with stable reserve revisions. A key advantage of this reinforcement learning (RL) approach is that it can learn from all observed claim trajectories, including claims that remain open at valuation, thereby avoiding the reduced sample size and selection effects inherent in supervised methods trained on ultimate outcomes only. We also introduce practical components needed for actuarial use -- initialisation of new claims, temporally consistent tuning via a rolling-settlement scheme, and an importance-weighting mechanism to mitigate portfolio-level underestimation driven by the rarity of large claims. On CAS and SPLICE synthetic general insurance datasets, the proposed Soft Actor-Critic implementation delivers competitive claim-level accuracy and strong aggregate OCL performance, particularly for the immature claim segments that drive most of the liability.

**arXiv ID:** 2601.07637
</details>

<details>
<summary><strong>On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning</strong> - Changyu Liu, Yiyang Liu, Taowen Wang, Qiao Zhuang, James Chenhao Liang, Wenhao Yang, Renjing Xu, Qifan Wang, Dongfang Liu, Cheng Han - [[pdf]](https://arxiv.org/pdf/2601.06748)</summary>

**Abstract:** Vision-Language-Action models have recently emerged as a powerful paradigm for general-purpose robot learning, enabling agents to map visual observations and natural-language instructions into executable robotic actions. Though popular, they are primarily trained via supervised fine-tuning or training-time reinforcement learning, requiring explicit fine-tuning phases, human interventions, or controlled data collection. Consequently, existing methods remain unsuitable for challenging simulated- or physical-world deployments, where robots must respond autonomously and flexibly to evolving environments. To address this limitation, we introduce a Test-Time Reinforcement Learning for VLAs (TT-VLA), a framework that enables on-the-fly policy adaptation during inference. TT-VLA formulates a dense reward mechanism that leverages step-by-step task-progress signals to refine action policies during test time while preserving the SFT/RL-trained priors, making it an effective supplement to current VLA models. Empirical results show that our approach enhances overall adaptability, stability, and task success in dynamic, previously unseen scenarios under simulated and real-world settings. We believe TT-VLA offers a principled step toward self-improving, deployment-ready VLAs.

**arXiv ID:** 2601.06748
</details>

<details>
<summary><strong>Heterogeneous Multi-Expert Reinforcement Learning for Long-Horizon Multi-Goal Tasks in Autonomous Forklifts</strong> - Yun Chen, Bowei Huang, Fan Guo, Kang Song - [[pdf]](https://arxiv.org/pdf/2601.07304)</summary>

**Abstract:** Autonomous mobile manipulation in unstructured warehouses requires a balance between efficient large-scale navigation and high-precision object interaction. Traditional end-to-end learning approaches often struggle to handle the conflicting demands of these distinct phases. Navigation relies on robust decision-making over large spaces, while manipulation needs high sensitivity to fine local details. Forcing a single network to learn these different objectives simultaneously often causes optimization interference, where improving one task degrades the other. To address these limitations, we propose a Heterogeneous Multi-Expert Reinforcement Learning (HMER) framework tailored for autonomous forklifts. HMER decomposes long-horizon tasks into specialized sub-policies controlled by a Semantic Task Planner. This structure separates macro-level navigation from micro-level manipulation, allowing each expert to focus on its specific action space without interference. The planner coordinates the sequential execution of these experts, bridging the gap between task planning and continuous control. Furthermore, to solve the problem of sparse exploration, we introduce a Hybrid Imitation-Reinforcement Training Strategy. This method uses expert demonstrations to initialize the policy and Reinforcement Learning for fine-tuning. Experiments in Gazebo simulations show that HMER significantly outperforms sequential and end-to-end baselines. Our method achieves a task success rate of 94.2\% (compared to 62.5\% for baselines), reduces operation time by 21.4\%, and maintains placement error within 1.5 cm, validating its efficacy for precise material handling.

**arXiv ID:** 2601.07304
</details>

<details>
<summary><strong>Failure-Aware RL: Reliable Offline-to-Online Reinforcement Learning with Self-Recovery for Real-World Manipulation</strong> - Huanyu Li, Kun Lei, Sheng Zang, Kaizhe Hu, Yongyuan Liang, Bo An, Xiaoli Li, Huazhe Xu - [[pdf]](https://arxiv.org/pdf/2601.07821)</summary>

**Abstract:** Post-training algorithms based on deep reinforcement learning can push the limits of robotic models for specific objectives, such as generalizability, accuracy, and robustness. However, Intervention-requiring Failures (IR Failures) (e.g., a robot spilling water or breaking fragile glass) during real-world exploration happen inevitably, hindering the practical deployment of such a paradigm. To tackle this, we introduce Failure-Aware Offline-to-Online Reinforcement Learning (FARL), a new paradigm minimizing failures during real-world reinforcement learning. We create FailureBench, a benchmark that incorporates common failure scenarios requiring human intervention, and propose an algorithm that integrates a world-model-based safety critic and a recovery policy trained offline to prevent failures during online exploration. Extensive simulation and real-world experiments demonstrate the effectiveness of FARL in significantly reducing IR Failures while improving performance and generalization during online reinforcement learning post-training. FARL reduces IR Failures by 73.1% while elevating performance by 11.3% on average during real-world RL post-training. Videos and code are available at this https URL.

**arXiv ID:** 2601.07821
</details>

<details>
<summary><strong>Benchmarking Autonomy in Scientific Experiments: A Hierarchical Taxonomy for Autonomous Large-Scale Facilities</strong> - James Le Houx - [[pdf]](https://arxiv.org/pdf/2601.06978)</summary>

**Abstract:** The transition from automated data collection to fully autonomous discovery requires a shared vocabulary to benchmark progress. While the automotive industry relies on the SAE J3016 standard, current taxonomies for autonomous science presuppose an owner-operator model that is incompatible with the operational rigidities of Large-Scale User Facilities. Here, we propose the Benchmarking Autonomy in Scientific Experiments (BASE) Scale, a 6-level taxonomy (Levels 0-5) specifically adapted for these unique constraints. Unlike owner-operator models, User Facilities require zero-shot deployment where agents must operate immediately without extensive training periods. We define the specific technical requirements for each tier, identifying the Inference Barrier (Level 3) as the critical latency threshold where decisions shift from scalar feedback to semantic digital twins. Fundamentally, this level extends the decision manifold from spatial exploration to temporal gating, enabling the agent to synchronise acquisition with the onset of transient physical events. By establishing these operational definitions, the BASE Scale provides facility directors, funding bodies, and beamline scientists with a standardised metric to assess risk, define liability, and quantify the intelligence of experimental workflows.

**arXiv ID:** 2601.06978
</details>

<details>
<summary><strong>Autonomous Driving in Unstructured Environments: How Far Have We Come?</strong> - Chen Min, Shubin Si, Xu Wang, Hanzhang Xue, Weizhong Jiang, Zitong Chen, Mengmeng Li, Jilin Mei, Erke Shang, Zhipeng Xiao, Bin Dai, Qi Zhu, Hao Fu, Dawei Zhao, Liang Xiao, Yiming Nie, Yu Hu - [[pdf]](https://arxiv.org/pdf/2410.07701)</summary>

**Abstract:** Research on autonomous driving in unstructured outdoor environments is less advanced than in structured urban settings due to challenges like environmental diversities and scene complexity. These environments-such as rural areas and rugged terrains-pose unique obstacles that are not common in structured urban areas. Despite these difficulties, autonomous driving in unstructured outdoor environments is crucial for applications in agriculture, mining, and military operations. Our survey reviews over 250 papers for autonomous driving in unstructured outdoor environments, covering offline mapping, pose estimation, environmental perception, path planning, end-to-end autonomous driving, datasets, and relevant challenges. We also discuss emerging trends and future research directions. This review aims to consolidate knowledge and encourage further research for autonomous driving in unstructured environments. To support ongoing work, we maintain an active repository with up-to-date literature and open-source projects at: this https URL.

**arXiv ID:** 2410.07701
</details>

<details>
<summary><strong>Cross-Platform Learnable Fuzzy Gain-Scheduled Proportional-Integral-Derivative Controller Tuning via Physics-Constrained Meta-Learning and Reinforcement Learning Adaptation</strong> - JiaHao Wu, ShengWen Yu - [[pdf]](https://arxiv.org/pdf/2511.06500)</summary>

**Abstract:** Motivation and gap: PID-family controllers remain a pragmatic choice for many robotic systems due to their simplicity and interpretability, but tuning stable, high-performing gains is time-consuming and typically non-transferable across robot morphologies, payloads, and deployment conditions. Fuzzy gain scheduling can provide interpretable online adjustment, yet its per-joint scaling and consequent parameters are platform-dependent and difficult to tune systematically.
Proposed approach: We propose a hierarchical framework for cross-platform tuning of a learnable fuzzy gain-scheduled PID (LF-PID). The controller uses shared fuzzy membership partitions to preserve common error semantics, while learning per-joint scaling and Takagi-Sugeno consequent parameters that schedule PID gains online. Combined with physics-constrained virtual robot synthesis, meta-learning provides cross-platform initialization from robot physical features, and a lightweight reinforcement learning (RL) stage performs deployment-specific refinement under dynamics mismatch. Starting from three base simulated platforms, we generate 232 physically valid training variants via bounded perturbations of mass (+/-10%), inertia (+/-15%), and friction (+/-20%).
Results and insight: We evaluate cross-platform generalization on two distinct systems (a 9-DOF serial manipulator and a 12-DOF quadruped) under multiple disturbance scenarios. The RL adaptation stage improves tracking performance on top of the meta-initialized controller, with up to 80.4% error reduction in challenging high-load joints (12.36 degrees to 2.42 degrees) and 19.2% improvement under parameter uncertainty. We further identify an optimization ceiling effect: online refinement yields substantial gains when the meta-initialized baseline exhibits localized deficiencies, but provides limited improvement when baseline quality is already uniformly strong.

**arXiv ID:** 2511.06500
</details>

<details>
<summary><strong>Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions</strong> - Kehan Long, Jorge Cortés, Nikolay Atanasov - [[pdf]](https://arxiv.org/pdf/2505.10947)</summary>

**Abstract:** Establishing stability certificates for closed-loop systems under reinforcement learning (RL) policies is essential to move beyond empirical performance and offer guarantees of system behavior. Classical Lyapunov methods require a strict stepwise decrease in the Lyapunov function but such certificates are difficult to construct for learned policies. The RL value function is a natural candidate but it is not well understood how it can be adapted for this purpose. To gain intuition, we first study the linear quadratic regulator (LQR) problem and make two key observations. First, a Lyapunov function can be obtained from the value function of an LQR policy by augmenting it with a residual term related to the system dynamics and stage cost. Second, the classical Lyapunov decrease requirement can be relaxed to a generalized Lyapunov condition requiring only decrease on average over multiple time steps. Using this intuition, we consider the nonlinear setting and formulate an approach to learn generalized Lyapunov functions by augmenting RL value functions with neural network residual terms. Our approach successfully certifies the stability of RL policies trained on Gymnasium and DeepMind Control benchmarks. We also extend our method to jointly train neural controllers and stability certificates using a multi-step Lyapunov loss, resulting in larger certified inner approximations of the region of attraction compared to the classical Lyapunov approach. Overall, our formulation enables stability certification for a broad class of systems with learned policies by making certificates easier to construct, thereby bridging classical control theory and modern learning-based methods.

**arXiv ID:** 2505.10947
</details>

<details>
<summary><strong>SegDAC: Improving Visual Reinforcement Learning by Extracting Dynamic Object-Centric Representations from Pretrained Vision Models</strong> - Alexandre Brown, Glen Berseth - [[pdf]](https://arxiv.org/pdf/2508.09325)</summary>

**Abstract:** Visual reinforcement learning (RL) is challenging due to the need to extract useful representations from high-dimensional inputs while learning effective control from sparse and noisy rewards. Although large perception models exist, integrating them effectively into RL for visual generalization and improved sample efficiency remains difficult. We propose SegDAC, a Segmentation-Driven Actor-Critic method. SegDAC uses Segment Anything (SAM) for object-centric decomposition and YOLO-World to ground the image segmentation process via text inputs. It includes a novel transformer-based architecture that supports a dynamic number of segments at each time step and effectively learns which segments to focus on using online RL, without using human labels. By evaluating SegDAC over a challenging visual generalization benchmark using Maniskill3, which covers diverse manipulation tasks under strong visual perturbations, we demonstrate that SegDAC achieves significantly better visual generalization, doubling prior performance on the hardest setting and matching or surpassing prior methods in sample efficiency across all evaluated tasks. Project Page: this https URL

**arXiv ID:** 2508.09325
</details>

<details>
<summary><strong>MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control</strong> - Xue Bin Peng - [[pdf]](https://arxiv.org/pdf/2510.13794)</summary>

**Abstract:** MimicKit is an open-source framework for training motion controllers using motion imitation and reinforcement learning. The codebase provides implementations of commonly-used motion-imitation techniques and RL algorithms. This framework is intended to support research and applications in computer graphics and robotics by providing a unified training framework, along with standardized environment, agent, and data structures. The codebase is designed to be modular and easily configurable, enabling convenient modification and extension to new characters and tasks. The open-source codebase is available at: this https URL.

**arXiv ID:** 2510.13794
</details>

<details>
<summary><strong>DiSCo: Making Absence Visible in Intelligent Summarization Interfaces</strong> - Eran Fainman, Hagit Ben Shoshan, Adir Solomon, Osnat Mokryn - [[pdf]](https://arxiv.org/pdf/2601.07229)</summary>

**Abstract:** Intelligent interfaces increasingly use large language models to summarize user-generated content, yet these summaries emphasize what is mentioned while overlooking what is missing. This presence bias can mislead users who rely on summaries to make decisions. We present Domain Informed Summarization through Contrast (DiSCo), an expectation-based computational approach that makes absences visible by comparing each entity's content with domain topical expectations captured in reference distributions of aspects typically discussed in comparable accommodations. This comparison identifies aspects that are either unusually emphasized or missing relative to domain norms and integrates them into the generated text. In a user study across three accommodation domains, namely ski, beach, and city center, DiSCo summaries were rated as more detailed and useful for decision making than baseline large language model summaries, although slightly harder to read. The findings show that modeling expectations reduces presence bias and improves both transparency and decision support in intelligent summarization interfaces.

**arXiv ID:** 2601.07229
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
