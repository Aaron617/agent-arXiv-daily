# Agent arXiv Daily

**Last Updated:** 2026-01-23 03:08:12

**Total Papers:** 61

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
<summary><strong>Agentic Confidence Calibration</strong> - Jiaxin Zhang, Caiming Xiong, Chien-Sheng Wu - [[pdf]](https://arxiv.org/pdf/2601.15778)</summary>

**Abstract:** AI agents are rapidly advancing from passive language models to autonomous systems executing complex, multi-step tasks. Yet their overconfidence in failure remains a fundamental barrier to deployment in high-stakes settings. Existing calibration methods, built for static single-turn outputs, cannot address the unique challenges of agentic systems, such as compounding errors along trajectories, uncertainty from external tools, and opaque failure modes. To address these challenges, we introduce, for the first time, the problem of Agentic Confidence Calibration and propose Holistic Trajectory Calibration (HTC), a novel diagnostic framework that extracts rich process-level features ranging from macro dynamics to micro stability across an agent's entire trajectory. Powered by a simple, interpretable model, HTC consistently surpasses strong baselines in both calibration and discrimination, across eight benchmarks, multiple LLMs, and diverse agent frameworks. Beyond performance, HTC delivers three essential advances: it provides interpretability by revealing the signals behind failure, enables transferability by applying across domains without retraining, and achieves generalization through a General Agent Calibrator (GAC) that achieves the best calibration (lowest ECE) on the out-of-domain GAIA benchmark. Together, these contributions establish a new process-centric paradigm for confidence calibration, providing a framework for diagnosing and enhancing the reliability of AI agents.

**arXiv ID:** 2601.15778
</details>

<details>
<summary><strong>A Checklist for Trustworthy, Safe, and User-Friendly Mental Health Chatbots</strong> - Shreya Haran, Samiha Thatikonda, Dong Whi Yoo, Koustuv Saha - [[pdf]](https://arxiv.org/pdf/2601.15412)</summary>

**Abstract:** Mental health concerns are rising globally, prompting increased reliance on technology to address the demand-supply gap in mental health services. In particular, mental health chatbots are emerging as a promising solution, but these remain largely untested, raising concerns about safety and potential harms. In this paper, we dive into the literature to identify critical gaps in the design and implementation of mental health chatbots. We contribute an operational checklist to help guide the development and design of more trustworthy, safe, and user-friendly chatbots. The checklist serves as both a developmental framework and an auditing tool to ensure ethical and effective chatbot design. We discuss how this checklist is a step towards supporting more responsible design practices and supporting new standards for sociotechnically sound digital mental health tools.

**arXiv ID:** 2601.15412
</details>

<details>
<summary><strong>PromptHelper: A Prompt Recommender System for Encouraging Creativity in AI Chatbot Interactions</strong> - Jason Kim, Maria Teleki, James Caverlee - [[pdf]](https://arxiv.org/pdf/2601.15575)</summary>

**Abstract:** Prompting is central to interaction with AI systems, yet many users struggle to explore alternative directions, articulate creative intent, or understand how variations in prompts shape model outputs. We introduce prompt recommender systems (PRS) as an interaction approach that supports exploration, suggesting contextually relevant follow-up prompts. We present PromptHelper, a PRS prototype integrated into an AI chatbot that surfaces semantically diverse prompt suggestions while users work on real writing tasks. We evaluate PromptHelper in a 2x2 fully within-subjects study (N=32) across creative and academic writing tasks. Results show that PromptHelper significantly increases users' perceived exploration and expressiveness without increasing cognitive workload. Qualitative findings illustrate how prompt recommendations help users branch into new directions, overcome uncertainty about what to ask next, and better articulate their intent. We discuss implications for designing AI interfaces that scaffold exploratory interaction while preserving user agency, and release open-source resources to support research on prompt recommendation.

**arXiv ID:** 2601.15575
</details>

<details>
<summary><strong>Do You Feel Comfortable? Detecting Hidden Conversational Escalation in AI Chatbots</strong> - Jihyung Park, Saleh Afroogh, David Atkinson, Junfeng Jiao - [[pdf]](https://arxiv.org/pdf/2512.06193)</summary>

**Abstract:** Large Language Models (LLM) are increasingly integrated into everyday interactions, serving not only as information assistants but also as emotional companions. Even in the absence of explicit toxicity, repeated emotional reinforcement or affective drift can gradually escalate distress in a form of \textit{implicit harm} that traditional toxicity filters fail to detect. Existing guardrail mechanisms often rely on external classifiers or clinical rubrics that may lag behind the nuanced, real-time dynamics of a developing conversation. To address this gap, we propose GAUGE (Guarding Affective Utterance Generation Escalation), logit-based framework for the real-time detection of hidden conversational escalation. GAUGE measures how an LLM's output probabilistically shifts the affective state of a dialogue.

**arXiv ID:** 2512.06193
</details>

<details>
<summary><strong>Cloning the Self for Mental Well-Being: A Framework for Designing Safe and Therapeutic Self-Clone Chatbots</strong> - Mehrnoosh Sadat Shirvani, Jackie Crowley, Cher Peng, Jackie Liu, Thomas Chao, Suky Martinez, Laura Brandt, Ig-Jae Kim, Dongwook Yoon - [[pdf]](https://arxiv.org/pdf/2601.15465)</summary>

**Abstract:** As digital tools increasingly mediate mental health care, self-clone chatbots can offer a uniquely novel approach to intra-personal exploration and self-derived support. Trained to replicate users' conversational patterns, self-clones allow users to talk to themselves through their digital replicas. Despite the promises, these systems may carry risks around identity confusion, negative reinforcement, and blurred user agency. Through interviews with 16 mental health professionals and 6 general users, we aim to uncover tensions and design opportunities in this emerging space to guide responsible self-clone design. Our analysis produces a design framework organized around three priorities: (1) defining goals and grounding the approach in existing therapeutic models, (2) design dimensions including the self-clone persona and user-clone relationship dynamics, and (3) considerations for minimizing potential emotional and ethical harms. This framework contributes an interdisciplinary foundation for designing self-clone chatbots as AI-mediated self-interaction tools that are emotionally and ethically attuned in mental health contexts.

**arXiv ID:** 2601.15465
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (11 papers)</h2></summary>

<details>
<summary><strong>MiRAGE: A Multiagent Framework for Generating Multimodal Multihop Question-Answer Dataset for RAG Evaluation</strong> - Chandan Kumar Sahu, Premith Kumar Chilukuri, Matthew Hetrich - [[pdf]](https://arxiv.org/pdf/2601.15487)</summary>

**Abstract:** The rapid evolution of Retrieval-Augmented Generation (RAG) toward multimodal, high-stakes enterprise applications has outpaced the development of domain specific evaluation benchmarks. Existing datasets often rely on general-domain corpora or purely textual retrieval, failing to capture the complexity of specialized technical documents where information is inextricably multimodal and reasoning requires synthesizing disjoint evidence. We address this gap by introducing MiRAGE, a Multiagent framework for RAG systems Evaluation, that leverages a collaborative swarm of specialized agents to generate verified, domain-specific, multimodal, and multi-hop Question-Answer datasets. MiRAGE orchestrates a swarm of specialized agents: a recursive context optimization loop to aggregate scattered evidence, an adversarial verifier agent to guarantee factual grounding, and an agent to recognize the expert persona and the relevant domain to mimic expert cognitive workflows. Extensive empirical evaluation across four distinct domains (regulations, finance, quantitative biology, and journalism) demonstrates that MiRAGE generates datasets with significantly higher reasoning complexity (>2.3 average hops) and factual faithfulness. Our ablation studies point that MiRAGE can be powered by LLMs if textual descriptions of the images are available. Visual grounding still remains a frontier. By automating the creation of gold standard evaluation datasets that reflect the latent thematic structure of proprietary corpora, MiRAGE provides the necessary infrastructure to rigorously benchmark the next generation information retrieval systems.

**arXiv ID:** 2601.15487
</details>

<details>
<summary><strong>Autonomous Business System via Neuro-symbolic AI</strong> - Cecil Pang, Hiroki Sayama - [[pdf]](https://arxiv.org/pdf/2601.15599)</summary>

**Abstract:** Current business environments require organizations to continuously reconfigure cross-functional processes, yet enterprise systems are still organized around siloed departments, rigid workflows, and hard-coded automation. Meanwhile large language models (LLMs) excel at interpreting natural language and unstructured data but lack deterministic, verifiable execution of complex business logic. To address this gap, here we introduce AUTOBUS, an Autonomous Business System that integrates LLM-based AI agents, predicate-logic programming, and business-semantics-centric enterprise data into a coherent neuro-symbolic AI architecture for orchestrating end-to-end business initiatives. AUTOBUS models an initiative as a network of tasks with explicit pre/post conditions, required data, evaluation rules, and API-level actions. Enterprise data is organized as a knowledge graph whose entities, relationships, and constraints are translated into logic facts and foundational rules, providing the semantic grounding for task reasoning. Core AI agents synthesize task instructions, enterprise semantics, and available tools into task-specific logic programs, which are executed by a logic engine that enforces constraints, coordinates auxiliary tools, and orchestrate execution of actions and outcomes. Humans define and maintain the semantics, policies and task instructions, curate tools, and supervise high-impact or ambiguous decisions, ensuring accountability and adaptability. We detail the AUTOBUS architecture, the anatomy of the AI agent generated logic programs, and the role of humans and auxiliary tools in the lifecycle of a business initiative.

**arXiv ID:** 2601.15599
</details>

<details>
<summary><strong>Improving Methodologies for Agentic Evaluations Across Domains: Leakage of Sensitive Information, Fraud and Cybersecurity Threats</strong> - Ee Wei Seah, Yongsen Zheng, Naga Nikshith, Mahran Morsidi, Gabriel Waikin Loh Matienzo, Nigel Gay, Akriti Vij, Benjamin Chua, En Qi Ng, Sharmini Johnson, Vanessa Wilfred, Wan Sie Lee, Anna Davidson, Catherine Devine, Erin Zorer, Gareth Holvey, Harry Coppock, James Walpole, Jerome Wynee, Magda Dubois, Michael Schmatz, Patrick Keane, Sam Deverett, Bill Black, Bo Yan, Bushra Sabir, Frank Sun, Hao Zhang, Harriet Farlow, Helen Zhou, Lingming Dong, Qinghua Lu, Seung Jang, Sharif Abuadbba, Simon O'Callaghan, Suyu Ma, Tom Howroyd, Cyrus Fung, Fatemeh Azadi, Isar Nejadgholi, Krishnapriya Vishnubhotla, Pulei Xiong, Saeedeh Lohrasbi, Scott Buffett, Shahrear Iqbal, Sowmya Vajjala, Anna Safont-Andreu, Luca Massarelli, Oskar van der Wal, Simon Möller, Agnes Delaborde, Joris Duguépéroux, Nicolas Rolin, Romane Gallienne, Sarah Behanzin, Tom Seimandi, Akiko Murakami, Takayuki Semitsu, Teresa Tsukiji, Angela Kinuthia, Michael Michie, Stephanie Kasaon, Jean Wangari, Hankyul Baek, Jaewon Noh, Kihyuk Nam, Sang Seo, Sungpil Shin, Taewhi Lee, Yongsu Kim - [[pdf]](https://arxiv.org/pdf/2601.15679)</summary>

**Abstract:** The rapid rise of autonomous AI systems and advancements in agent capabilities are introducing new risks due to reduced oversight of real-world interactions. Yet agent testing remains nascent and is still a developing science. As AI agents begin to be deployed globally, it is important that they handle different languages and cultures accurately and securely.
To address this, participants from The International Network for Advanced AI Measurement, Evaluation and Science, including representatives from Singapore, Japan, Australia, Canada, the European Commission, France, Kenya, South Korea, and the United Kingdom have come together to align approaches to agentic evaluations.
This is the third exercise, building on insights from two earlier joint testing exercises conducted by the Network in November 2024 and February 2025. The objective is to further refine best practices for testing advanced AI systems.
The exercise was split into two strands: (1) common risks, including leakage of sensitive information and fraud, led by Singapore AISI; and (2) cybersecurity, led by UK AISI. A mix of open and closed-weight models were evaluated against tasks from various public agentic benchmarks. Given the nascency of agentic testing, our primary focus was on understanding methodological issues in conducting such tests, rather than examining test results or model capabilities. This collaboration marks an important step forward as participants work together to advance the science of agentic evaluations.

**arXiv ID:** 2601.15679
</details>

<details>
<summary><strong>Agentic Uncertainty Quantification</strong> - Jiaxin Zhang, Prafulla Kumar Choubey, Kung-Hsiang Huang, Caiming Xiong, Chien-Sheng Wu - [[pdf]](https://arxiv.org/pdf/2601.15703)</summary>

**Abstract:** Although AI agents have demonstrated impressive capabilities in long-horizon reasoning, their reliability is severely hampered by the ``Spiral of Hallucination,'' where early epistemic errors propagate irreversibly. Existing methods face a dilemma: uncertainty quantification (UQ) methods typically act as passive sensors, only diagnosing risks without addressing them, while self-reflection mechanisms suffer from continuous or aimless corrections. To bridge this gap, we propose a unified Dual-Process Agentic UQ (AUQ) framework that transforms verbalized uncertainty into active, bi-directional control signals. Our architecture comprises two complementary mechanisms: System 1 (Uncertainty-Aware Memory, UAM), which implicitly propagates verbalized confidence and semantic explanations to prevent blind decision-making; and System 2 (Uncertainty-Aware Reflection, UAR), which utilizes these explanations as rational cues to trigger targeted inference-time resolution only when necessary. This enables the agent to balance efficient execution and deep deliberation dynamically. Extensive experiments on closed-loop benchmarks and open-ended deep research tasks demonstrate that our training-free approach achieves superior performance and trajectory-level calibration. We believe this principled framework AUQ represents a significant step towards reliable agents.

**arXiv ID:** 2601.15703
</details>

<details>
<summary><strong>AgentSM: Semantic Memory for Agentic Text-to-SQL</strong> - Asim Biswal, Chuan Lei, Xiao Qin, Aodong Li, Balakrishnan Narayanaswamy, Tim Kraska - [[pdf]](https://arxiv.org/pdf/2601.15709)</summary>

**Abstract:** Recent advances in LLM-based Text-to-SQL have achieved remarkable gains on public benchmarks such as BIRD and Spider. Yet, these systems struggle to scale in realistic enterprise settings with large, complex schemas, diverse SQL dialects, and expensive multi-step reasoning. Emerging agentic approaches show potential for adaptive reasoning but often suffer from inefficiency and instability-repeating interactions with databases, producing inconsistent outputs, and occasionally failing to generate valid answers. To address these challenges, we introduce Agent Semantic Memory (AgentSM), an agentic framework for Text-to-SQL that builds and leverages interpretable semantic memory. Instead of relying on raw scratchpads or vector retrieval, AgentSM captures prior execution traces-or synthesizes curated ones-as structured programs that directly guide future reasoning. This design enables systematic reuse of reasoning paths, which allows agents to scale to larger schemas, more complex questions, and longer trajectories efficiently and reliably. Compared to state-of-the-art systems, AgentSM achieves higher efficiency by reducing average token usage and trajectory length by 25% and 35%, respectively, on the Spider 2.0 benchmark. It also improves execution accuracy, reaching a state-of-the-art accuracy of 44.8% on the Spider 2.0 Lite benchmark.

**arXiv ID:** 2601.15709
</details>

<details>
<summary><strong>FormGym: Doing Paperwork with Agents</strong> - Matthew Toles, Rattandeep Singh, Isaac Song, Zhou Yu - [[pdf]](https://arxiv.org/pdf/2506.14079)</summary>

**Abstract:** Completing paperwork is a challenging and time-consuming problem. Form filling is especially challenging in the pure-image domain without access to OCR, typeset PDF text, or a DOM. For computer agents, it requires multiple abilities, including multi-modal understanding, information retrieval, and tool-use. We present a novel form-filling benchmark consisting of 432 fields spread across 55 documents and 3 tasks, requiring knowledge of 236 features per user. We find that baseline VLAs achieve less than 1% accuracy in most cases, primarily due to poor localization ability. GUI agents also struggle, scoring between 10.6-68.0% despite high cost and latency. Therefore, we also contribute FieldFinder, a tool to assist LLMs in identifying where to place text on a form. With FieldFinder, all models achieve equal or better performance in all six study conditions, with a maximum increase from 2% to 56%.

**arXiv ID:** 2506.14079
</details>

<details>
<summary><strong>Gaming the Judge: Unfaithful Chain-of-Thought Can Undermine Agent Evaluation</strong> - Muhammad Khalifa, Lajanugen Logeswaran, Jaekyeom Kim, Sungryull Sohn, Yunxiang Zhang, Moontae Lee, Hao Peng, Lu Wang, Honglak Lee - [[pdf]](https://arxiv.org/pdf/2601.14691)</summary>

**Abstract:** Large language models (LLMs) are increasingly used as judges to evaluate agent performance, particularly in non-verifiable settings where judgments rely on agent trajectories including chain-of-thought (CoT) reasoning. This paradigm implicitly assumes that the agent's CoT faithfully reflects both its internal reasoning and the underlying environment state. We show this assumption is brittle: LLM judges are highly susceptible to manipulation of agent reasoning traces. By systematically rewriting agent CoTs while holding actions and observations fixed, we demonstrate that manipulated reasoning alone can inflate false positive rates of state-of-the-art VLM judges by up to 90% across 800 trajectories spanning diverse web tasks. We study manipulation strategies spanning style-based approaches that alter only the presentation of reasoning and content-based approaches that fabricate signals of task progress, and find that content-based manipulations are consistently more effective. We evaluate prompting-based techniques and scaling judge-time compute, which reduce but do not fully eliminate susceptibility to manipulation. Our findings reveal a fundamental vulnerability in LLM-based evaluation and highlight the need for judging mechanisms that verify reasoning claims against observable evidence.

**arXiv ID:** 2601.14691
</details>

<details>
<summary><strong>Emerging from Ground: Addressing Intent Deviation in Tool-Using Agents via Deriving Real Calls into Virtual Trajectories</strong> - Qian Xiong, Yuekai Huang, Bo Yang, Yujia Zheng, Tianhao Li, Ziyou Jiang, Zhiyuan Chang, Zhaoyang Li, Huanxiang Feng, Mingyang Li - [[pdf]](https://arxiv.org/pdf/2601.15120)</summary>

**Abstract:** LLMs have advanced tool-using agents for real-world applications, yet they often lead to unexpected behaviors or results. Beyond obvious failures, the subtle issue of "intent deviation" severely hinders reliable evaluation and performance improvement. Existing post-training methods generally leverage either real system samples or virtual data simulated by LLMs. However, the former is costly due to reliance on hand-crafted user requests, while the latter suffers from distribution shift from the real tools in the wild. Additionally, both methods lack negative samples tailored to intent deviation scenarios, hindering effective guidance on preference learning. We introduce RISE, a "Real-to-Virtual" method designed to mitigate intent deviation. Anchoring on verified tool primitives, RISE synthesizes virtual trajectories and generates diverse negative samples through mutation on critical parameters. With synthetic data, RISE fine-tunes backbone LLMs via the two-stage training for intent alignment. Evaluation results demonstrate that data synthesized by RISE achieve promising results in eight metrics covering user requires, execution trajectories and agent responses. Integrating with training, RISE achieves an average 35.28% improvement in Acctask (task completion) and 23.27% in Accintent (intent alignment), outperforming SOTA baselines by 1.20--42.09% and 1.17--54.93% respectively.

**arXiv ID:** 2601.15120
</details>

<details>
<summary><strong>EmbedAgent: Benchmarking Large Language Models in Embedded System Development</strong> - Ruiyang Xu, Jialun Cao, Mingyuan Wu, Wenliang Zhong, Yaojie Lu, Ben He, Xianpei Han, Shing-Chi Cheung, Le Sun - [[pdf]](https://arxiv.org/pdf/2506.11003)</summary>

**Abstract:** Large Language Models (LLMs) have shown promise in various tasks, yet few benchmarks assess their capabilities in embedded system this http URL this paper, we introduce EmbedAgent, a paradigm designed to simulate real-world roles in embedded system development, such as Embedded System Programmer, Architect, and Integrator. This paradigm enables LLMs to be tested in tasks that bridge the gap between digital and physical systems, allowing for a more comprehensive assessment of their capabilities. To evaluate LLMs on these tasks, we propose Embedbench, the first comprehensive benchmark for embedded system programming, circuit design, and cross-platform this http URL consists of 126 cases, covering 9 electronic components across 3 hardware platforms. Through extensive experiments on 10 mainstream LLMs, we uncover several key findings. Surprisingly, despite the simplicity of the cases, DeepSeek-R1 achieves only a 55.6% pass@1 rate when provided with schematic information, and 50.0% when tasked with generating the schematics itself. In the cross-platform migration tasks, LLMs show relatively strong performance with MicroPython on the Raspberry Pi Pico (with the top model achieving 73.8% pass@1), but perform poorly on ESP-IDF, where the best model reaches only 29.4% pass@1.Interestingly, we observe that general-purpose chat LLMs like DeepSeek-V3 often fail to utilize relevant pre-trained knowledge in this domain, while reasoning LLMs tend to overthink and overlook efficient knowledge during pretraining. Based on these insights, we propose two strategies: retrieval augmented generation and compiler feedback-to enhance LLM performance. These strategies result in significant improvements, with Deepseek-R1 reaching a 65.1% pass@1 with correct schematics, and 53.1% without. Additionally, the accuracy of the Arduino to ESP32 migration task improves from 21.4% to 27.8%.

**arXiv ID:** 2506.11003
</details>

<details>
<summary><strong>DUET: Agentic Design Understanding via Experimentation and Testing</strong> - Gus Henry Smith, Sandesh Adhikary, Vineet Thumuluri, Karthik Suresh, Vivek Pandit, Kartik Hegde, Hamid Shojaei, Chandra Bhagavatula - [[pdf]](https://arxiv.org/pdf/2512.06247)</summary>

**Abstract:** AI agents powered by large language models (LLMs) are being used to solve increasingly complex software engineering challenges, but struggle with hardware design tasks. Register Transfer Level (RTL) code presents a unique challenge for LLMs, as it encodes complex, dynamic, time-evolving behaviors using the low-level language features of SystemVerilog. LLMs struggle to infer these complex behaviors from the syntax of RTL alone, which limits their ability to complete all downstream tasks like code completion, documentation, or verification. In response to this issue, we present DUET: a general methodology for developing Design Understanding via Experimentation and Testing. DUET mimics how hardware design experts develop an understanding of complex designs: not just via a one-off readthrough of the RTL, but via iterative experimentation using a number of tools. DUET iteratively generates hypotheses, tests them with EDA tools (e.g., simulation, waveform inspection, and formal verification), and integrates the results to build a bottom-up understanding of the design. In our evaluations, we show that DUET improves AI agent performance on formal verification, when compared to a baseline flow without experimentation.

**arXiv ID:** 2512.06247
</details>

<details>
<summary><strong>I-MCTS: Enhancing Agentic AutoML via Introspective Monte Carlo Tree Search</strong> - Zujie Liang, Feng Wei, Wujiang Xu, Lin Chen, Yuxi Qian, Xinhui Wu - [[pdf]](https://arxiv.org/pdf/2502.14693)</summary>

**Abstract:** Recent advancements in large language models (LLMs) have shown remarkable potential in automating machine learning tasks. However, existing LLM-based agents often struggle with low-diversity and suboptimal code generation. While recent work has introduced Monte Carlo Tree Search (MCTS) to address these issues, limitations persist in the quality and diversity of thoughts generated, as well as in the scalar value feedback mechanisms used for node selection. In this study, we introduce Introspective Monte Carlo Tree Search (I-MCTS), a novel approach that iteratively expands tree nodes through an introspective process that meticulously analyzes solutions and results from parent and sibling nodes. This facilitates a continuous refinement of the node in the search tree, thereby enhancing the overall decision-making process. Furthermore, we integrate a Large Language Model (LLM)-based value model to facilitate direct evaluation of each node's solution prior to conducting comprehensive computational rollouts. A hybrid rewarding mechanism is implemented to seamlessly transition the Q-value from LLM-estimated scores to actual performance scores. This allows higher-quality nodes to be traversed earlier. Applied to the various ML tasks, our approach demonstrates a 4% absolute improvement in performance compared to the strong open-source AutoML agents, showcasing its effectiveness in enhancing agentic AutoML systems. Resource available at this https URL

**arXiv ID:** 2502.14693
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>Aeon: High-Performance Neuro-Symbolic Memory Management for Long-Horizon LLM Agents</strong> - Mustafa Arslan - [[pdf]](https://arxiv.org/pdf/2601.15311)</summary>

**Abstract:** Large Language Models (LLMs) are fundamentally constrained by the quadratic computational cost of self-attention and the "Lost in the Middle" phenomenon, where reasoning capabilities degrade as context windows expand. Existing solutions, primarily "Flat RAG" architectures relying on vector databases, treat memory as an unstructured bag of embeddings. This approach fails to capture the hierarchical and temporal structure of long-horizon interactions, leading to "Vector Haze", the retrieval of disjointed facts lacking episodic continuity. We propose Aeon, a Neuro-Symbolic Cognitive Operating System that redefines memory not as a static store, but as a managed OS resource. Aeon structures memory into a Memory Palace (a spatial index implemented via Atlas, a SIMD-accelerated Page-Clustered Vector Index that combines small-world graph navigation with B+ Tree-style disk locality to minimize read amplification) and a Trace (a neuro-symbolic episodic graph). We introduce the Semantic Lookaside Buffer (SLB), a predictive caching mechanism that exploits conversational locality to achieve sub-millisecond retrieval latencies. Benchmarks demonstrate that Aeon achieves < 1ms retrieval latency on conversational workloads while ensuring state consistency via a zero-copy C++/Python bridge, effectively enabling persistent, structured memory for autonomous agents.

**arXiv ID:** 2601.15311
</details>

<details>
<summary><strong>Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents</strong> - Raffi Khatchadourian - [[pdf]](https://arxiv.org/pdf/2601.15322)</summary>

**Abstract:** LLM agents struggle with regulatory audit replay: when asked to reproduce a flagged transaction decision with identical inputs, most deployments fail to return consistent results. This paper introduces the Determinism-Faithfulness Assurance Harness (DFAH), a framework for measuring trajectory determinism and evidence-conditioned faithfulness in tool-using agents deployed in financial services.
Across 74 configurations (12 models, 4 providers, 8-24 runs each at T=0.0) in non-agentic baseline experiments, 7-20B parameter models achieved 100% determinism, while 120B+ models required 3.7x larger validation samples to achieve equivalent statistical reliability. Agentic tool-use introduces additional variance (see Tables 4-7). Contrary to the assumed reliability-capability trade-off, a positive Pearson correlation emerged (r = 0.45, p < 0.01, n = 51 at T=0.0) between determinism and faithfulness; models producing consistent outputs also tended to be more evidence-aligned.
Three financial benchmarks are provided (compliance triage, portfolio constraints, DataOps exceptions; 50 cases each) along with an open-source stress-test harness. In these benchmarks and under DFAH evaluation settings, Tier 1 models with schema-first architectures achieved determinism levels consistent with audit replay requirements.

**arXiv ID:** 2601.15322
</details>

<details>
<summary><strong>Controlling Long-Horizon Behavior in Language Model Agents with Explicit State Dynamics</strong> - Sukesh Subaharan - [[pdf]](https://arxiv.org/pdf/2601.16087)</summary>

**Abstract:** 

**arXiv ID:** 2601.16087
</details>

<details>
<summary><strong>VideoThinker: Building Agentic VideoLLMs with LLM-Guided Tool Reasoning</strong> - Chenglin Li, Qianglong Chen, Feng Han, Yikun Wang, Xingxi Yin, Yan Gong, Ruilin Li, Yin Zhang, Jiaqi Wang - [[pdf]](https://arxiv.org/pdf/2601.15724)</summary>

**Abstract:** Long-form video understanding remains a fundamental challenge for current Video Large Language Models. Most existing models rely on static reasoning over uniformly sampled frames, which weakens temporal localization and leads to substantial information loss in long videos. Agentic tools such as temporal retrieval, spatial zoom, and temporal zoom offer a natural way to overcome these limitations by enabling adaptive exploration of key moments. However, constructing agentic video understanding data requires models that already possess strong long-form video comprehension, creating a circular dependency. We address this challenge with VideoThinker, an agentic Video Large Language Model trained entirely on synthetic tool interaction trajectories. Our key idea is to convert videos into rich captions and employ a powerful agentic language model to generate multi-step tool use sequences in caption space. These trajectories are subsequently grounded back to video by replacing captions with the corresponding frames, yielding a large-scale interleaved video and tool reasoning dataset without requiring any long-form understanding from the underlying model. Training on this synthetic agentic dataset equips VideoThinker with dynamic reasoning capabilities, adaptive temporal exploration, and multi-step tool use. Remarkably, VideoThinker significantly outperforms both caption-only language model agents and strong video model baselines across long-video benchmarks, demonstrating the effectiveness of tool augmented synthetic data and adaptive retrieval and zoom reasoning for long-form video understanding.

**arXiv ID:** 2601.15724
</details>

<details>
<summary><strong>ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking</strong> - Qiang Zhang, Boli Chen, Fanrui Zhang, Ruixue Ding, Shihang Wang, Qiuchen Wang, Yinfeng Huang, Haonan Zhang, Rongxiang Zhu, Pengyong Wang, Ailin Ren, Xin Li, Pengjun Xie, Jiawei Liu, Ning Guo, Jingren Zhou, Zheng-Jun Zha - [[pdf]](https://arxiv.org/pdf/2601.06487)</summary>

**Abstract:** Reinforcement learning has substantially improved the performance of LLM agents on tasks with verifiable outcomes, but it still struggles on open-ended agent tasks with vast solution spaces (e.g., complex travel planning). Due to the absence of objective ground-truth for these tasks, current RL algorithms largely rely on reward models that assign scalar scores to individual responses. We contend that such pointwise scoring suffers from an inherent discrimination collapse: the reward model struggles to distinguish subtle advantages among different trajectories, resulting in scores within a group being compressed into a narrow range. Consequently, the effective reward signal becomes dominated by noise from the reward model, leading to optimization stagnation. To address this, we propose ArenaRL, a reinforcement learning paradigm that shifts from pointwise scalar scoring to intra-group relative ranking. ArenaRL introduces a process-aware pairwise evaluation mechanism, employing multi-level rubrics to assign fine-grained relative scores to trajectories. Additionally, we construct an intra-group adversarial arena and devise a tournament-based ranking scheme to obtain stable advantage signals. Empirical results confirm that the built seeded single-elimination scheme achieves nearly equivalent advantage estimation accuracy to full pairwise comparisons with O(N^2) complexity, while operating with only O(N) complexity, striking an optimal balance between efficiency and precision. Furthermore, to address the lack of full-cycle benchmarks for open-ended agents, we build Open-Travel and Open-DeepResearch, two high-quality benchmarks featuring a comprehensive pipeline covering SFT, RL training, and multi-dimensional evaluation. Extensive experiments show that ArenaRL substantially outperforms standard RL baselines, enabling LLM agents to generate more robust solutions for complex real-world tasks.

**arXiv ID:** 2601.06487
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (11 papers)</h2></summary>

<details>
<summary><strong>TransportAgents: a multi-agents LLM framework for traffic accident severity prediction</strong> - Zhichao Yang, Jiashu He, Jinxuan Fan, Cirillo Cinzia - [[pdf]](https://arxiv.org/pdf/2601.15519)</summary>

**Abstract:** Accurate prediction of traffic crash severity is critical for improving emergency response and public safety planning. Although recent large language models (LLMs) exhibit strong reasoning capabilities, their single-agent architectures often struggle with heterogeneous, domain-specific crash data and tend to generate biased or unstable predictions. To address these limitations, this paper proposes TransportAgents, a hybrid multi-agent framework that integrates category-specific LLM reasoning with a multilayer perceptron (MLP) integration module. Each specialized agent focuses on a particular subset of traffic information, such as demographics, environmental context, or incident details, to produce intermediate severity assessments that are subsequently fused into a unified prediction. Extensive experiments on two complementary U.S. datasets, the Consumer Product Safety Risk Management System (CPSRMS) and the National Electronic Injury Surveillance System (NEISS), demonstrate that TransportAgents consistently outperforms both traditional machine learning and advanced LLM-based baselines. Across three representative backbones, including closed-source models such as GPT-3.5 and GPT-4o, as well as open-source models such as LLaMA-3.3, the framework exhibits strong robustness, scalability, and cross-dataset generalizability. A supplementary distributional analysis further shows that TransportAgents produces more balanced and well-calibrated severity predictions than standard single-agent LLM approaches, highlighting its interpretability and reliability for safety-critical decision support applications.

**arXiv ID:** 2601.15519
</details>

<details>
<summary><strong>ALIGNAgent: Adaptive Learner Intelligence for Gap Identification and Next-step guidance</strong> - Bismack Tokoli, Luis Jaimes, Ayesha S. Dina - [[pdf]](https://arxiv.org/pdf/2601.15551)</summary>

**Abstract:** Personalized learning systems have emerged as a promising approach to enhance student outcomes by tailoring educational content, pacing, and feedback to individual needs. However, most existing systems remain fragmented, specializing in either knowledge tracing, diagnostic modeling, or resource recommendation, but rarely integrating these components into a cohesive adaptive cycle. In this paper, we propose ALIGNAgent (Adaptive Learner Intelligence for Gap Identification and Next-step guidance), a multi-agent educational framework designed to deliver personalized learning through integrated knowledge estimation, skill-gap identification, and targeted resource this http URL begins by processing student quiz performance, gradebook data, and learner preferences to generate topic-level proficiency estimates using a Skill Gap Agent that employs concept-level diagnostic reasoning to identify specific misconceptions and knowledge deficiencies. After identifying skill gaps, the Recommender Agent retrieves preference-aware learning materials aligned with diagnosed deficiencies, implementing a continuous feedback loop where interventions occur before advancing to subsequent topics. Extensive empirical evaluation on authentic datasets from two undergraduate computer science courses demonstrates ALIGNAgent's effectiveness, with GPT-4o-based agents achieving precision of 0.87-0.90 and F1 scores of 0.84-0.87 in knowledge proficiency estimation validated against actual exam performance.

**arXiv ID:** 2601.15551
</details>

<details>
<summary><strong>Agentic Persona Control and Task State Tracking for Realistic User Simulation in Interactive Scenarios</strong> - Hareeshwar Karthikeyan - [[pdf]](https://arxiv.org/pdf/2601.15290)</summary>

**Abstract:** Testing conversational AI systems at scale across diverse domains necessitates realistic and diverse user interactions capturing a wide array of behavioral patterns. We present a novel multi-agent framework for realistic, explainable human user simulation in interactive scenarios, using persona control and task state tracking to mirror human cognitive processes during goal-oriented conversations. Our system employs three specialized AI agents: (1) a User Agent to orchestrate the overall interaction, (2) a State Tracking Agent to maintain structured task state, and (3) a Message Attributes Generation Agent that controls conversational attributes based on task progress and assigned persona. To validate our approach, we implement and evaluate the framework for guest ordering at a restaurant with scenarios rich in task complexity, behavioral diversity, and conversational ambiguity. Through systematic ablations, we evaluate the contributory efficacy of each agentic component to overall simulation quality in terms of persona adherence, task completion accuracy, explainability, and realism. Our experiments demonstrate that the complete multi-agent system achieves superior simulation quality compared to single-LLM baselines, with significant gains across all evaluation metrics. This framework establishes a powerful environment for orchestrating agents to simulate human users with cognitive plausibility, decomposing the simulation into specialized sub-agents that reflect distinct aspects of human thought processes applicable across interactive domains.

**arXiv ID:** 2601.15290
</details>

<details>
<summary><strong>FARM: Field-Aware Resolution Model for Intelligent Trigger-Action Automation</strong> - Khusrav Badalov, Young Yoon - [[pdf]](https://arxiv.org/pdf/2601.15687)</summary>

**Abstract:** Trigger-Action Programming (TAP) platforms such as IFTTT and Zapier enable Web of Things (WoT) automation by composing event-driven rules across heterogeneous services. A TAP applet links a trigger to an action and must bind trigger outputs (ingredients) to action inputs (fields) to be executable. Prior work largely treats TAP as service-level prediction from natural language, which often yields non-executable applets that still require manual configuration. We study the function-level configuration problem: generating complete applets with correct ingredient-to-field bindings. We propose FARM (Field-Aware Resolution Model), a two-stage architecture for automated applet generation with full configuration. Stage 1 trains contrastive dual encoders with selective layer freezing over schema-enriched representations, retrieving candidates from 1,724 trigger functions and 1,287 action functions (2.2M possible trigger-action pairs). Stage 2 performs selection and configuration using an LLM-based multi-agent pipeline. It includes intent analysis, trigger selection, action selection via cross-schema scoring, and configuration verification. Agents coordinate through shared state and agreement-based selection. FARM achieves 81% joint accuracy on Gold (62% Noisy, 70% One-shot) at the function level, where both trigger and action functions must match the ground truth. For comparison with service-level baselines, we map functions to their parent services and evaluate at the service level. FARM reaches 81% joint accuracy and improves over TARGE by 23 percentage points. FARM also generates ingredient-to-field bindings, producing executable automation configurations.

**arXiv ID:** 2601.15687
</details>

<details>
<summary><strong>VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning</strong> - Li Kang, Xiufeng Song, Heng Zhou, Yiran Qin, Jie Yang, Xiaohong Liu, Philip Torr, Lei Bai, Zhenfei Yin - [[pdf]](https://arxiv.org/pdf/2506.09049)</summary>

**Abstract:** Coordinating multiple embodied agents in dynamic environments remains a core challenge in artificial intelligence, requiring both perception-driven reasoning and scalable cooperation strategies. While recent works have leveraged large language models (LLMs) for multi-agent planning, a few have begun to explore vision-language models (VLMs) for visual reasoning. However, these VLM-based approaches remain limited in their support for diverse embodiment types. In this work, we introduce VIKI-Bench, the first hierarchical benchmark tailored for embodied multi-agent cooperation, featuring three structured levels: agent activation, task planning, and trajectory perception. VIKI-Bench includes diverse robot embodiments, multi-view visual observations, and structured supervision signals to evaluate reasoning grounded in visual inputs. To demonstrate the utility of VIKI-Bench, we propose VIKI-R, a two-stage framework that fine-tunes a pretrained vision-language model (VLM) using Chain-of-Thought annotated demonstrations, followed by reinforcement learning under multi-level reward signals. Our extensive experiments show that VIKI-R significantly outperforms baselines method across all task levels. Furthermore, we show that reinforcement learning enables the emergence of compositional cooperation patterns among heterogeneous agents. Together, VIKI-Bench and VIKI-R offer a unified testbed and method for advancing multi-agent, visual-driven cooperation in embodied AI systems.

**arXiv ID:** 2506.09049
</details>

<details>
<summary><strong>How malicious AI swarms can threaten democracy: The fusion of agentic AI and LLMs marks a new frontier in information warfare</strong> - Daniel Thilo Schroeder, Meeyoung Cha, Andrea Baronchelli, Nick Bostrom, Nicholas A. Christakis, David Garcia, Amit Goldenberg, Yara Kyrychenko, Kevin Leyton-Brown, Nina Lutz, Gary Marcus, Filippo Menczer, Gordon Pennycook, David G. Rand, Maria Ressa, Frank Schweitzer, Dawn Song, Christopher Summerfield, Audrey Tang, Jay J. Van Bavel, Sander van der Linden, Jonas R. Kunst - [[pdf]](https://arxiv.org/pdf/2506.06299)</summary>

**Abstract:** Advances in AI offer the prospect of manipulating beliefs and behaviors on a population-wide level. Large language models and autonomous agents now let influence campaigns reach unprecedented scale and precision. Generative tools can expand propaganda output without sacrificing credibility and inexpensively create falsehoods that are rated as more human-like than those written by humans. Techniques meant to refine AI reasoning, such as chain-of-thought prompting, can just as effectively be used to generate more convincing falsehoods. Enabled by these capabilities, a disruptive threat is emerging: swarms of collaborative, malicious AI agents. Fusing LLM reasoning with multi-agent architectures, these systems are capable of coordinating autonomously, infiltrating communities, and fabricating consensus efficiently. By adaptively mimicking human social dynamics, they threaten democracy. Because the resulting harms stem from design, commercial incentives, and governance, we prioritize interventions at multiple leverage points, focusing on pragmatic mechanisms over voluntary compliance.

**arXiv ID:** 2506.06299
</details>

<details>
<summary><strong>Collaborate, Deliberate, Evaluate: How LLM Alignment Affects Coordinated Multi-Agent Outcomes</strong> - Abhijnan Nath, Carine Graff, Nikhil Krishnaswamy - [[pdf]](https://arxiv.org/pdf/2509.05882)</summary>

**Abstract:** As Large Language Models (LLMs) get integrated into diverse workflows, they are increasingly being regarded as "collaborators" with humans, and required to work in coordination with other AI systems. If such AI collaborators are to reliably coordinate their actions and behaviors with humans or other AIs, their properties and behaviors over multi-turn interactions must be known and predictable. This paper examines how different alignment methods affect LLM agents' effectiveness as partners in multi-turn, multi-party collaborations. We study this question through the lens of intervention agents that insert themselves into group dialogues not to provide answers, but to encourage the collaborative group to slow down and reflect upon their reasoning for deliberative decision-making. Common alignment techniques are typically developed under simplified single-user settings and assume the optimality of the underlying token MDP. Using the theoretical lens of the modified-action MDP, we show how they do not account for the dynamics of long-horizon multi-party interactions. We present a novel roleplay simulation methodology, where we align LLMs according to different methods and then deploy them in collaborative task dialogues to quantify how interventions affect the trajectory of group collaboration, belief alignment, and coordination. Our results show that an intervention agent that is robust to action modification significantly outperforms common alignment baselines in supporting correct task outcomes.

**arXiv ID:** 2509.05882
</details>

<details>
<summary><strong>VLM-CAD: VLM-Optimized Collaborative Agent Design Workflow for Analog Circuit Sizing</strong> - Guanyuan Pan, Shuai Wang, Yugui Lin, Tiansheng Zhou, Pietro Liò, Yaqi Wang, Zhenxin Zhao - [[pdf]](https://arxiv.org/pdf/2601.07315)</summary>

**Abstract:** Analog mixed-signal circuit sizing involves complex trade-offs within high-dimensional design spaces. Existing automatic analog circuit sizing approaches rely solely on netlists, ignoring the circuit schematic, which hinders the cognitive link between the schematic and its performance. Furthermore, the black-box nature of machine learning methods and hallucination risks in large language models fail to provide the necessary ground-truth explainability required for industrial sign-off. To address these challenges, we propose a Vision Language Model-optimized collaborative agent design workflow (VLM-CAD), which analyzes circuits, optimizes DC operating points, performs inference-based sizing, and executes external sizing optimization. We integrate Image2Net to annotate circuit schematics and generate a structured JSON description for precise interpretation by Vision Language Models. Furthermore, we propose an Explainable Trust Region Bayesian Optimization method (ExTuRBO) that employs collaborative warm-start from agent-generated seeds and offers dual-granularity sensitivity analysis for external sizing optimization, supporting a comprehensive final design report. Experiment results on amplifier sizing tasks using 180nm, 90nm, and 45nm Predictive Technology Models demonstrate that VLM-CAD effectively balances power and performance while maintaining physics-based explainability. VLM-CAD meets all specification requirements while maintaining low power consumption in optimizing an amplifier with a complementary input and a class-AB output stage, with a total runtime under 66 minutes across all experiments on two amplifiers.

**arXiv ID:** 2601.07315
</details>

<details>
<summary><strong>MALTopic: Multi-Agent LLM Topic Modeling Framework</strong> - Yash Sharma - [[pdf]](https://arxiv.org/pdf/2601.15299)</summary>

**Abstract:** Topic modeling is a crucial technique for extracting latent themes from unstructured text data, particularly valuable in analyzing survey responses. However, traditional methods often only consider free-text responses and do not natively incorporate structured or categorical survey responses for topic modeling. And they produce abstract topics, requiring extensive human interpretation. To address these limitations, we propose the Multi-Agent LLM Topic Modeling Framework (MALTopic). This framework decomposes topic modeling into specialized tasks executed by individual LLM agents: an enrichment agent leverages structured data to enhance textual responses, a topic modeling agent extracts latent themes, and a deduplication agent refines the results. Comparative analysis on a survey dataset demonstrates that MALTopic significantly improves topic coherence, diversity, and interpretability compared to LDA and BERTopic. By integrating structured data and employing a multi-agent approach, MALTopic generates human-readable topics with enhanced contextual relevance, offering a more effective solution for analyzing complex survey data.

**arXiv ID:** 2601.15299
</details>

<details>
<summary><strong>DISPATCH -- Decentralized Informed Spatial Planning and Assignment of Tasks for Cooperative Heterogeneous Agents</strong> - Yao Liu, Sampad Mohanty, Elizabeth Ondula, Bhaskar Krishnamachari - [[pdf]](https://arxiv.org/pdf/2511.17915)</summary>

**Abstract:** Spatial task allocation in systems such as multi-robot delivery or ride-sharing requires balancing efficiency with fair service across tasks. Greedy assignment policies that match each agent to its highest-preference or lowest-cost task can maximize efficiency but often create inequities: some tasks receive disproportionately favorable service (e.g., shorter delays or better matches), while others face long waits or poor allocations.
We study fairness in heterogeneous multi-agent systems where tasks vary in preference alignment and urgency. Most existing approaches either assume centralized coordination or largely ignore fairness under partial observability. Distinct from this prior work, we establish a connection between the Eisenberg-Gale (EG) equilibrium convex program and decentralized, partially observable multi-agent learning. Building on this connection, we develop two equilibrium-informed algorithms that integrate fairness and efficiency: (i) a multi-agent reinforcement learning (MARL) framework, EG-MARL, whose training is guided by a centralized EG equilibrium assignment algorithm; and (ii) a stochastic online optimization mechanism that performs guided exploration and subset-based fair assignment as tasks are discovered.
We evaluate on Multi-Agent Particle Environment (MPE) simulations across varying team sizes against centralized EG, Hungarian, and Min-Max distance baselines, and also present a Webots-based warehouse proof-of-concept with heterogeneous robots. Both methods preserve the fairness-efficiency balance of the EG solution under partial observability, with EG-MARL achieving near-centralized coordination and reduced travel distances, and the online mechanism enabling real-time allocation with competitive fairness.

**arXiv ID:** 2511.17915
</details>

<details>
<summary><strong>Exploring Implicit Perspectives on Autism in Large Language Models Through Multi-Agent Simulations</strong> - Sohyeon Park, Jesus Armando Beltran, Aehong Min, Anamara Ritt-Olson, Gillian R. Hayes - [[pdf]](https://arxiv.org/pdf/2601.15437)</summary>

**Abstract:** Large Language Models (LLMs) like ChatGPT offer potential support for autistic people, but this potential requires understanding the implicit perspectives these models might carry, including their biases and assumptions about autism. Moving beyond single-agent prompting, we utilized LLM-based multi-agent systems to investigate complex social scenarios involving autistic and non-autistic agents. In our study, agents engaged in group-task conversations and answered structured interview questions, which we analyzed to examine ChatGPT's biases and how it conceptualizes autism. We found that ChatGPT assumes autistic people are socially dependent, which may affect how it interacts with autistic users or conveys information about autism. To address these challenges, we propose adopting the double empathy problem, which reframes communication breakdowns as a mutual challenge. We describe how future LLMs could address the biases we observed and improve interactions involving autistic people by incorporating the double empathy problem into their design.

**arXiv ID:** 2601.15437
</details>

</details>

<details open>
<summary><h2>Other Agent Research (6 papers)</h2></summary>

<details>
<summary><strong>DualShield: Safe Model Predictive Diffusion via Reachability Analysis for Interactive Autonomous Driving</strong> - Rui Yang, Lei Zheng, Ruoyu Yao, Jun Ma - [[pdf]](https://arxiv.org/pdf/2601.15729)</summary>

**Abstract:** Diffusion models have emerged as a powerful approach for multimodal motion planning in autonomous driving. However, their practical deployment is typically hindered by the inherent difficulty in enforcing vehicle dynamics and a critical reliance on accurate predictions of other agents, making them prone to safety issues under uncertain interactions. To address these limitations, we introduce DualShield, a planning and control framework that leverages Hamilton-Jacobi (HJ) reachability value functions in a dual capacity. First, the value functions act as proactive guidance, steering the diffusion denoising process towards safe and dynamically feasible regions. Second, they form a reactive safety shield using control barrier-value functions (CBVFs) to modify the executed actions and ensure safety. This dual mechanism preserves the rich exploration capabilities of diffusion models while providing principled safety assurance under uncertain and even adversarial interactions. Simulations in challenging unprotected U-turn scenarios demonstrate that DualShield significantly improves both safety and task efficiency compared to leading methods from different planning paradigms under uncertainty.

**arXiv ID:** 2601.15729
</details>

<details>
<summary><strong>TDFlow: Agentic Workflows for Test Driven Development</strong> - Kevin Han, Siddharth Maddikayala, Tim Knappe, Om Patel, Austen Liao, Amir Barati Farimani - [[pdf]](https://arxiv.org/pdf/2510.23761)</summary>

**Abstract:** We introduce TDFlow, a novel test-driven agentic workflow that frames repository-scale software engineering as a test-resolution task, specifically designed to solve human-written tests. Given a set of tests, TDFlow repeatedly proposes, revises, and debugs repository-scale patches using precisely engineered sub-agents and tightly constrained tools. The workflow decomposes software engineering program repair into four components governed by respective sub-agents. This simple, forced decoupling of patch proposing, debugging, patch revision, and optional test generation (1) reduces long-context burden on any individual sub-agent, (2) focuses each sub-agent on specific, pre-defined sub-tasks, and (3) allows for specialized performance improvement on specific sub-tasks. When provided human-written tests, TDFlow attains 88.8% pass rate on SWE-Bench Lite (an absolute improvement of 27.8% over the next best system) and 94.3% on SWE-Bench Verified. Manual inspection of the 800 TDFlow runs within SWE-Bench Lite and Verified uncover only 7 instances of test hacking, which were subsequently counted as failures. Furthermore, we show that the primary obstacle to human-level software engineering performance lies within writing successful reproduction tests. We envision a human-LLM interactive system powered by TDFlow where human developers write tests solved by LLM systems. Together, these results indicate that modern LLMs, when embedded in a narrowly engineered, test-driven workflow, already achieve human-level test resolution -- with the final frontier for fully autonomous repository repair being the accurate generation of valid reproduction tests.

**arXiv ID:** 2510.23761
</details>

<details>
<summary><strong>Dynamics of Agentic Loops in Large Language Models: A Geometric Theory of Trajectories</strong> - Nicolas Tacheny - [[pdf]](https://arxiv.org/pdf/2512.10350)</summary>

**Abstract:** Agentic systems built on large language models operate through recursive feedback loops, where each output becomes the next input. Yet the geometric behavior of these agentic loops (whether they converge, diverge, or exhibit more complex dynamics) remains poorly understood. This paper introduces a geometric framework for analyzing agentic trajectories in semantic embedding space, treating iterative transformations as discrete dynamical systems. We distinguish the artifact space, where linguistic transformations occur, from the embedding space, where geometric measurements are performed. Because cosine similarity is biased by embedding anisotropy, we introduce an isotonic calibration that eliminates systematic bias and aligns similarities with human semantic judgments while preserving high local stability. This enables rigorous measurement of trajectories, clusters and attractors. Through controlled experiments on singular agentic loops, we identify two fundamental regimes. A contractive rewriting loop converges toward a stable attractor with decreasing dispersion, while an exploratory summarize and negate loop produces unbounded divergence with no cluster formation. These regimes display qualitatively distinct geometric signatures of contraction and expansion. Our results show that prompt design directly governs the dynamical regime of an agentic loop, enabling systematic control of convergence, divergence and trajectory structure in iterative LLM transformations.

**arXiv ID:** 2512.10350
</details>

<details>
<summary><strong>Deaf and Hard of Hearing Access to Intelligent Personal Assistants: Comparison of Voice-Based Options with an LLM-Powered Touch Interface</strong> - Paige S. DeVries, Michaela Okosi, Ming Li, Nora Dunphy, Gidey Gezae, Dante Conway, Abraham Glasser, Raja Kushalnagar, Christian Vogler - [[pdf]](https://arxiv.org/pdf/2601.15209)</summary>

**Abstract:** We investigate intelligent personal assistants (IPAs) accessibility for deaf and hard of hearing (DHH) people who can use their voice in everyday communication. The inability of IPAs to understand diverse accents including deaf speech renders them largely inaccessible to non-signing and speaking DHH individuals. Using an Echo Show, we compare the usability of natural language input via spoken English; with Alexa's automatic speech recognition and a Wizard-of-Oz setting with a trained facilitator re-speaking commands against that of a large language model (LLM)-assisted touch interface in a mixed-methods study. The touch method was navigated through an LLM-powered "task prompter," which integrated the user's history and smart environment to suggest contextually-appropriate commands. Quantitative results showed no significant differences across both spoken English conditions vs LLM-assisted touch. Qualitative results showed variability in opinions on the usability of each method. Ultimately, it will be necessary to have robust deaf-accented speech recognized natively by IPAs.

**arXiv ID:** 2601.15209
</details>

<details>
<summary><strong>DextER: Language-driven Dexterous Grasp Generation with Embodied Reasoning</strong> - Junha Lee, Eunha Park, Minsu Cho - [[pdf]](https://arxiv.org/pdf/2601.16046)</summary>

**Abstract:** Language-driven dexterous grasp generation requires the models to understand task semantics, 3D geometry, and complex hand-object interactions. While vision-language models have been applied to this problem, existing approaches directly map observations to grasp parameters without intermediate reasoning about physical interactions. We present DextER, Dexterous Grasp Generation with Embodied Reasoning, which introduces contact-based embodied reasoning for multi-finger manipulation. Our key insight is that predicting which hand links contact where on the object surface provides an embodiment-aware intermediate representation bridging task semantics with physical constraints. DextER autoregressively generates embodied contact tokens specifying which finger links contact where on the object surface, followed by grasp tokens encoding the hand configuration. On DexGYS, DextER achieves 67.14% success rate, outperforming state-of-the-art by 3.83%p with 96.4% improvement in intention alignment. We also demonstrate steerable generation through partial contact specification, providing fine-grained control over grasp synthesis.

**arXiv ID:** 2601.16046
</details>

<details>
<summary><strong>Reflective Motion and a Physical Canvas: Exploring Embodied Journaling in Virtual Reality</strong> - Michael Yin, Robert Xiao, Nadine Wagener - [[pdf]](https://arxiv.org/pdf/2601.15656)</summary>

**Abstract:** In traditional journaling practices, authors express and process their thoughts by writing them down. We propose a somaesthetic-inspired alternative that uses the human body, rather than written words, as the medium of expression. We coin this embodied journaling, as people's isolated body movements and spoken words become the canvas of reflection. We implemented embodied journaling in virtual reality and conducted a within-subject user study (n=20) to explore the emergent behaviours from the process and to compare its expressive and reflective qualities to those of written journaling. When writing-based norms and affordances were absent, we found that participants defaulted towards unfiltered emotional expression, often forgoing words altogether. Rather, subconscious body motion and paralinguistic acoustic qualities unveiled deeper, sometimes hidden feelings, prompting reflection that happens after emotional expression rather than during it. We discuss both the capabilities and pitfalls of embodied journaling, ultimately challenging the idea that reflection culminates in linguistic reasoning.

**arXiv ID:** 2601.15656
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (23 papers)</h2></summary>

<details>
<summary><strong>Agentic AI Governance and Lifecycle Management in Healthcare</strong> - Chandra Prakash, Mary Lind, Avneesh Sisodia - [[pdf]](https://arxiv.org/pdf/2601.15630)</summary>

**Abstract:** Healthcare organizations are beginning to embed agentic AI into routine workflows, including clinical documentation support and early-warning monitoring. As these capabilities diffuse across departments and vendors, health systems face agent sprawl, causing duplicated agents, unclear accountability, inconsistent controls, and tool permissions that persist beyond the original use case. Existing AI governance frameworks emphasize lifecycle risk management but provide limited guidance for the day-to-day operations of agent fleets. We propose a Unified Agent Lifecycle Management (UALM) blueprint derived from a rapid, practice-oriented synthesis of governance standards, agent security literature, and healthcare compliance requirements. UALM maps recurring gaps onto five control-plane layers: (1) an identity and persona registry, (2) orchestration and cross-domain mediation, (3) PHI-bounded context and memory, (4) runtime policy enforcement with kill-switch triggers, and (5) lifecycle management and decommissioning linked to credential revocation and audit logging. A companion maturity model supports staged adoption. UALM offers healthcare CIOs, CISOs, and clinical leaders an implementable pattern for audit-ready oversight that preserves local innovation and enables safer scaling across clinical and administrative domains.

**arXiv ID:** 2601.15630
</details>

<details>
<summary><strong>Inference-Time Scaling of Verification: Self-Evolving Deep Research Agents via Test-Time Rubric-Guided Verification</strong> - Yuxuan Wan, Tianqing Fang, Zaitang Li, Yintong Huo, Wenxuan Wang, Haitao Mi, Dong Yu, Michael R. Lyu - [[pdf]](https://arxiv.org/pdf/2601.15808)</summary>

**Abstract:** Recent advances in Deep Research Agents (DRAs) are transforming automated knowledge discovery and problem-solving. While the majority of existing efforts focus on enhancing policy capabilities via post-training, we propose an alternative paradigm: self-evolving the agent's ability by iteratively verifying the policy model's outputs, guided by meticulously crafted rubrics. This approach gives rise to the inference-time scaling of verification, wherein an agent self-improves by evaluating its generated answers to produce iterative feedback and refinements. We derive the rubrics based on an automatically constructed DRA Failure Taxonomy, which systematically classifies agent failures into five major categories and thirteen sub-categories. We present DeepVerifier, a rubrics-based outcome reward verifier that leverages the asymmetry of verification and outperforms vanilla agent-as-judge and LLM judge baselines by 12%-48% in meta-evaluation F1 score. To enable practical self-evolution, DeepVerifier integrates as a plug-and-play module during test-time inference. The verifier produces detailed rubric-based feedback, which is fed back to the agent for iterative bootstrapping, refining responses without additional training. This test-time scaling delivers 8%-11% accuracy gains on challenging subsets of GAIA and XBench-DeepResearch when powered by capable closed-source LLMs. Finally, to support open-source advancement, we release DeepVerifier-4K, a curated supervised fine-tuning dataset of 4,646 high-quality agent steps focused on DRA verification. These examples emphasize reflection and self-critique, enabling open models to develop robust verification capabilities.

**arXiv ID:** 2601.15808
</details>

<details>
<summary><strong>EvoCUA: Evolving Computer Use Agents via Learning from Scalable Synthetic Experience</strong> - Taofeng Xue, Chong Peng, Mianqiu Huang, Linsen Guo, Tiancheng Han, Haozhe Wang, Jianing Wang, Xiaocheng Zhang, Xin Yang, Dengchang Zhao, Jinrui Ding, Xiandi Ma, Yuchen Xie, Peng Pei, Xunliang Cai, Xipeng Qiu - [[pdf]](https://arxiv.org/pdf/2601.15876)</summary>

**Abstract:** The development of native computer-use agents (CUA) represents a significant leap in multimodal AI. However, their potential is currently bottlenecked by the constraints of static data scaling. Existing paradigms relying primarily on passive imitation of static datasets struggle to capture the intricate causal dynamics inherent in long-horizon computer tasks. In this work, we introduce EvoCUA, a native computer use agentic model. Unlike static imitation, EvoCUA integrates data generation and policy optimization into a self-sustaining evolutionary cycle. To mitigate data scarcity, we develop a verifiable synthesis engine that autonomously generates diverse tasks coupled with executable validators. To enable large-scale experience acquisition, we design a scalable infrastructure orchestrating tens of thousands of asynchronous sandbox rollouts. Building on these massive trajectories, we propose an iterative evolving learning strategy to efficiently internalize this experience. This mechanism dynamically regulates policy updates by identifying capability boundaries -- reinforcing successful routines while transforming failure trajectories into rich supervision through error analysis and self-correction. Empirical evaluations on the OSWorld benchmark demonstrate that EvoCUA achieves a success rate of 56.7%, establishing a new open-source state-of-the-art. Notably, EvoCUA significantly outperforms the previous best open-source model, OpenCUA-72B (45.0%), and surpasses leading closed-weights models such as UI-TARS-2 (53.1%). Crucially, our results underscore the generalizability of this approach: the evolving paradigm driven by learning from experience yields consistent performance gains across foundation models of varying scales, establishing a robust and scalable path for advancing native agent capabilities.

**arXiv ID:** 2601.15876
</details>

<details>
<summary><strong>Large Language Models as Simulative Agents for Neurodivergent Adult Psychometric Profiles</strong> - Francesco Chiappone, Davide Marocco, Nicola Milano - [[pdf]](https://arxiv.org/pdf/2601.15319)</summary>

**Abstract:** Adult neurodivergence, including Attention-Deficit/Hyperactivity Disorder (ADHD), high-functioning Autism Spectrum Disorder (ASD), and Cognitive Disengagement Syndrome (CDS), is marked by substantial symptom overlap that limits the discriminant sensitivity of standard psychometric instruments. While recent work suggests that Large Language Models (LLMs) can simulate human psychometric responses from qualitative data, it remains unclear whether they can accurately and stably model neurodevelopmental traits rather than broad personality characteristics. This study examines whether LLMs can generate psychometric responses that approximate those of real individuals when grounded in a structured qualitative interview, and whether such simulations are sensitive to variations in trait intensity. Twenty-six adults completed a 29-item open-ended interview and four standardized self-report measures (ASRS, BAARS-IV, AQ, RAADS-R). Two LLMs (GPT-4o and Qwen3-235B-A22B) were prompted to infer an individual psychological profile from interview content and then respond to each questionnaire in-role. Accuracy, reliability, and sensitivity were assessed using group-level comparisons, error metrics, exact-match scoring, and a randomized baseline. Both models outperformed random responses across instruments, with GPT-4o showing higher accuracy and reproducibility. Simulated responses closely matched human data for ASRS, BAARS-IV, and RAADS-R, while the AQ revealed subscale-specific limitations, particularly in Attention to Detail. Overall, the findings indicate that interview-grounded LLMs can produce coherent and above-chance simulations of neurodevelopmental traits, supporting their potential use as synthetic participants in early-stage psychometric research, while highlighting clear domain-specific constraints.

**arXiv ID:** 2601.15319
</details>

<details>
<summary><strong>Q-Probe: Scaling Image Quality Assessment to High Resolution via Context-Aware Agentic Probing</strong> - Xiang Li, XueHeng Li, Yu Wang, XuanHua He, ZhangChi Hu, WeiWei Yu, ChengJun Xie - [[pdf]](https://arxiv.org/pdf/2601.15356)</summary>

**Abstract:** Reinforcement Learning (RL) has empowered Multimodal Large Language Models (MLLMs) to achieve superior human preference alignment in Image Quality Assessment (IQA). However, existing RL-based IQA models typically rely on coarse-grained global views, failing to capture subtle local degradations in high-resolution scenarios. While emerging "Thinking with Images" paradigms enable multi-scale visual perception via zoom-in mechanisms, their direct adaptation to IQA induces spurious "cropping-implies-degradation" biases and misinterprets natural depth-of-field as artifacts. To address these challenges, we propose Q-Probe, the first agentic IQA framework designed to scale IQA to high resolution via context-aware probing. First, we construct Vista-Bench, a pioneering benchmark tailored for fine-grained local degradation analysis in high-resolution IQA settings. Furthermore, we propose a three-stage training paradigm that progressively aligns the model with human preferences, while simultaneously eliminating causal bias through a novel context-aware cropping strategy. Extensive experiments demonstrate that Q-Probe achieves state-of-the-art performance in high-resolution settings while maintaining superior efficacy across resolution scales.

**arXiv ID:** 2601.15356
</details>

<details>
<summary><strong>A Beacon Based Solution for Autonomous UUVs GNSS-Denied Stealthy Navigation</strong> - Alexandre Albore, Humbert Fiorino, Damien Pellier - [[pdf]](https://arxiv.org/pdf/2601.15802)</summary>

**Abstract:** Autonomous Unmanned Underwater Vehicles (UUVs) enable military and civilian covert operations in coastal areas without relying on support vessels or Global Navigation Satellite Systems (GNSS). Such operations are critical when surface access is not possible and stealthy navigation is required in restricted environments such as protected zones or dangerous areas under access ban. GNSS denied navigation is then essential to maintaining concealment as surfacing could expose UUVs to detection. To ensure a precise fleet positioning a constellation of beacons deployed by aerial or surface drones establish a synthetic landmark network that will guide the fleet of UUVs along an optimized path from the continental shelf to the goal on the shore. These beacons either submerged or floating emit acoustic signals for UUV localisation and navigation. A hierarchical planner generates an adaptive route for the drones executing primitive actions while continuously monitoring and replanning as needed to maintain trajectory accuracy.

**arXiv ID:** 2601.15802
</details>

<details>
<summary><strong>LLM-in-Sandbox Elicits General Agentic Intelligence</strong> - Daixuan Cheng, Shaohan Huang, Yuxian Gu, Huatong Song, Guoxin Chen, Li Dong, Wayne Xin Zhao, Ji-Rong Wen, Furu Wei - [[pdf]](https://arxiv.org/pdf/2601.16206)</summary>

**Abstract:** We introduce LLM-in-Sandbox, enabling LLMs to explore within a code sandbox (i.e., a virtual computer), to elicit general intelligence in non-code domains. We first demonstrate that strong LLMs, without additional training, exhibit generalization capabilities to leverage the code sandbox for non-code tasks. For example, LLMs spontaneously access external resources to acquire new knowledge, leverage the file system to handle long contexts, and execute scripts to satisfy formatting requirements. We further show that these agentic capabilities can be enhanced through LLM-in-Sandbox Reinforcement Learning (LLM-in-Sandbox-RL), which uses only non-agentic data to train models for sandbox exploration. Experiments demonstrate that LLM-in-Sandbox, in both training-free and post-trained settings, achieves robust generalization spanning mathematics, physics, chemistry, biomedicine, long-context understanding, and instruction following. Finally, we analyze LLM-in-Sandbox's efficiency from computational and system perspectives, and open-source it as a Python package to facilitate real-world deployment.

**arXiv ID:** 2601.16206
</details>

<details>
<summary><strong>Evolving in Tasks: Empowering the Multi-modality Large Language Model as the Computer Use Agent</strong> - Yuhao Cheng, Liang Tang, Shuxian Li, Yukang Huo, Tiaonan Duan, Kaer Huang, Yanzhe Jing, Yiqiang Yan - [[pdf]](https://arxiv.org/pdf/2508.04037)</summary>

**Abstract:** Computer use agents represent an emerging area in artificial intelligence, aiming to operate computers autonomously to fulfill user tasks, attracting significant attention from both industry and academia. However, the performance of existing agents remains insufficient for practical deployment. In this paper, we propose the Self-Evolution Agent (SEA) for computer operation, alongside three core innovations in data generation, reinforcement learning, and model enhancement to develop this agent. Specifically, we first design an automatic pipeline to generate verifiable task trajectories for training. Second, we propose Efficient Step-wise Reinforcement Learning to reduce the substantial computational overhead of long-horizon training. Finally, we introduce a model enhancement method that integrates grounding and planning capabilities into a single model without additional training. Leveraging these innovations, our SEA (with only 7B parameters) outperforms existing models of the same parameter scale and achieves performance comparable to larger models (e.g., 32B/72B parameters) on computer use tasks. We plan to release the model weights and related code as open-source resources in the future.

**arXiv ID:** 2508.04037
</details>

<details>
<summary><strong>Graph Neural Networks, Deep Reinforcement Learning and Probabilistic Topic Modeling for Strategic Multiagent Settings</strong> - Georgios Chalkiadakis, Charilaos Akasiadis, Gerasimos Koresis, Stergios Plataniotis, Leonidas Bakopoulos - [[pdf]](https://arxiv.org/pdf/2511.10501)</summary>

**Abstract:** This paper provides a comprehensive review of mainly GNN, DRL, and PTM methods with a focus on their potential incorporation in strategic multiagent settings. We draw interest in (i) ML methods currently utilized for uncovering unknown model structures adaptable to the task of strategic opponent modeling, and (ii) the integration of these methods with Game Theoretic concepts that avoid relying on assumptions often invalid in real-world scenarios, such as the Common Prior Assumption (CPA) and the Self-Interest Hypothesis (SIH). We analyze the ability to handle uncertainty and heterogeneity, two characteristics that are very common in real-world application cases, as well as scalability. As a potential answer to effectively modeling relationships and interactions in multiagent settings, we champion the use of GNN. Such approaches are designed to operate upon graph-structured data, and have been shown to be a very powerful tool for performing tasks such as node classification and link prediction. Next, we review the domain of RL, and in particular that of multiagent deep reinforcement learning. Single-agent deep RL has been widely used for decision making in demanding game settings. Its application in multiagent settings though is hindered due to, e.g., varying relationships between agents, and non-stationarity of the environment. We describe existing relevant game theoretic solution concepts, and consider properties such as fairness and stability. Our review comes complete with a note on the literature that utilizes probabilistic topic modeling (PTM) in domains other than that of document analysis and classification. Finally, we identify certain open challenges -- specifically, the need to (i) fit non-stationary environments, (ii) balance the degrees of stability and adaptation, (iii) tackle uncertainty and heterogeneity, (iv) guarantee scalability and solution tractability.

**arXiv ID:** 2511.10501
</details>

<details>
<summary><strong>GDEPO: Group Dual-dynamic and Equal-right Advantage Policy Optimization with Enhanced Training Data Utilization for Sample-Constrained Reinforcement Learning</strong> - Zhengqing Yan, Xinyang Liu, Yi Zhang, Fan Guo, ChengXun Jia, Junchen Wan, Yao Liu, Qi Liu, Jihao Huang, Kang Song - [[pdf]](https://arxiv.org/pdf/2601.06795)</summary>

**Abstract:** Automated Theorem Proving (ATP) represents a fundamental challenge in Artificial Intelligence (AI), requiring the construction of machine-verifiable proofs in formal languages such as Lean to evaluate AI reasoning capabilities. Reinforcement learning (RL), particularly the high-performance Group Relative Policy Optimization (GRPO) algorithm, has emerged as a mainstream approach for this task. However, in ATP scenarios, GRPO faces two critical issues: when composite rewards are used, its relative advantage estimation may conflict with the binary feedback from the formal verifier; meanwhile, its static sampling strategy may discard entire batches of data if no valid proof is found, resulting in zero contribution to model updates and significant data waste. To address these limitations, we propose Group Dual-dynamic and Equal-right-advantage Policy Optimization (GDEPO), a method incorporating three core mechanisms: 1) dynamic additional sampling, which resamples invalid batches until a valid proof is discovered; 2) equal-right advantage, decoupling the sign of the advantage function (based on correctness) from its magnitude (modulated by auxiliary rewards) to ensure stable and correct policy updates; and 3) dynamic additional iterations, applying extra gradient steps to initially failed but eventually successful samples to accelerate learning on challenging cases. Experiments conducted on three datasets of varying difficulty (MinF2F-test, MathOlympiadBench, PutnamBench) confirm the effectiveness of GDEPO, while ablation studies validate the necessity of its synergistic components. The proposed method enhances data utilization and optimization efficiency, offering a novel training paradigm for ATP.

**arXiv ID:** 2601.06795
</details>

<details>
<summary><strong>Representation-Driven Reinforcement Learning</strong> - Ofir Nabati, Guy Tennenholtz, Shie Mannor - [[pdf]](https://arxiv.org/pdf/2305.19922)</summary>

**Abstract:** We present a representation-driven framework for reinforcement learning. By representing policies as estimates of their expected values, we leverage techniques from contextual bandits to guide exploration and exploitation. Particularly, embedding a policy network into a linear feature space allows us to reframe the exploration-exploitation problem as a representation-exploitation problem, where good policy representations enable optimal exploration. We demonstrate the effectiveness of this framework through its application to evolutionary and policy gradient-based approaches, leading to significantly improved performance compared to traditional methods. Our framework provides a new perspective on reinforcement learning, highlighting the importance of policy representation in determining optimal exploration-exploitation strategies.

**arXiv ID:** 2305.19922
</details>

<details>
<summary><strong>Robust Reinforcement Learning in Finance: Modeling Market Impact with Elliptic Uncertainty Sets</strong> - Shaocong Ma, Heng Huang - [[pdf]](https://arxiv.org/pdf/2510.19950)</summary>

**Abstract:** In financial applications, reinforcement learning (RL) agents are commonly trained on historical data, where their actions do not influence prices. However, during deployment, these agents trade in live markets where their own transactions can shift asset prices, a phenomenon known as market impact. This mismatch between training and deployment environments can significantly degrade performance. Traditional robust RL approaches address this model misspecification by optimizing the worst-case performance over a set of uncertainties, but typically rely on symmetric structures that fail to capture the directional nature of market impact. To address this issue, we develop a novel class of elliptic uncertainty sets. We establish both implicit and explicit closed-form solutions for the worst-case uncertainty under these sets, enabling efficient and tractable robust policy evaluation. Experiments on single-asset and multi-asset trading tasks demonstrate that our method achieves superior Sharpe ratio and remains robust under increasing trade volumes, offering a more faithful and scalable approach to RL in financial markets.

**arXiv ID:** 2510.19950
</details>

<details>
<summary><strong>TeleMem: Building Long-Term and Multimodal Memory for Agentic AI</strong> - Chunliang Chen, Ming Guan, Xiao Lin, Jiaxu Li, Luxi Lin, Qiyi Wang, Xiangyu Chen, Jixiang Luo, Changzhi Sun, Dell Zhang, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2601.06037)</summary>

**Abstract:** Large language models (LLMs) excel at many NLP tasks but struggle to sustain long-term interactions due to limited attention over extended dialogue histories. Retrieval-augmented generation (RAG) mitigates this issue but lacks reliable mechanisms for updating or refining stored memories, leading to schema-driven hallucinations, inefficient write operations, and minimal support for multimodal this http URL address these challenges, we propose TeleMem, a unified long-term and multimodal memory system that maintains coherent user profiles through narrative dynamic extraction, ensuring that only dialogue-grounded information is preserved. TeleMem further introduces a structured writing pipeline that batches, retrieves, clusters, and consolidates memory entries, substantially improving storage efficiency, reducing token usage, and accelerating memory operations. Additionally, a multimodal memory module combined with ReAct-style reasoning equips the system with a closed-loop observe, think, and act process that enables accurate understanding of complex video content in long-term contexts. Experimental results show that TeleMem surpasses the state-of-the-art Mem0 baseline with 19% higher accuracy, 43% fewer tokens, and a 2.1x speedup on the ZH-4O long-term role-play gaming benchmark.

**arXiv ID:** 2601.06037
</details>

<details>
<summary><strong>Words to Describe What I'm Feeling: Exploring the Potential of AI Agents for High Subjectivity Decisions in Advance Care Planning</strong> - Kellie Yu Hui Sim, Pin Sym Foong, Chenyu Zhao, Melanie Yi Ning Quek, Swarangi Subodh Mehta, Kenny Tsu Wei Choo - [[pdf]](https://arxiv.org/pdf/2512.11276)</summary>

**Abstract:** Loss of decisional capacity, coupled with the increasing absence of reliable human proxies, raises urgent questions about how individuals' values can be represented in Advance Care Planning (ACP). To probe this fraught design space of high-risk, high-subjectivity decision support, we built an experience prototype (\acpagent{}) and asked 15 participants in 4 workshops to train it to be their personal ACP proxy. We analysed their coping strategies and feature requests and mapped the results onto axes of agent autonomy and human control. Our findings show a surprising 86.7\% agreement with \acpagent{}, arguing for a potential new role of AI in ACP where agents act as personal advocates for individuals, building mutual intelligibility over time. We propose that the key areas of future risk that must be addressed are the moderation of users' expectations and designing accountability and oversight over agent deployment and cutoffs.

**arXiv ID:** 2512.11276
</details>

<details>
<summary><strong>Vision-as-Inverse-Graphics Agent via Interleaved Multimodal Reasoning</strong> - Shaofeng Yin, Jiaxin Ge, Zora Zhiruo Wang, Xiuyu Li, Michael J. Black, Trevor Darrell, Angjoo Kanazawa, Haiwen Feng - [[pdf]](https://arxiv.org/pdf/2601.11109)</summary>

**Abstract:** Vision-as-inverse-graphics, the concept of reconstructing an image as an editable graphics program is a long-standing goal of computer vision. Yet even strong VLMs aren't able to achieve this in one-shot as they lack fine-grained spatial and physical grounding capability. Our key insight is that closing this gap requires interleaved multimodal reasoning through iterative execution and verification. Stemming from this, we present VIGA (Vision-as-Inverse-Graphic Agent) that starts from an empty world and reconstructs or edits scenes through a closed-loop write-run-render-compare-revise procedure. To support long-horizon reasoning, VIGA combines (i) a skill library that alternates generator and verifier roles and (ii) an evolving context memory that contains plans, code diffs, and render history. VIGA is task-agnostic as it doesn't require auxiliary modules, covering a wide range of tasks such as 3D reconstruction, multi-step scene editing, 4D physical interaction, and 2D document editing, etc. Empirically, we found VIGA substantially improves one-shot baselines on BlenderGym (35.32%) and SlideBench (117.17%). Moreover, VIGA is also model-agnostic as it doesn't require finetuning, enabling a unified protocol to evaluate heterogeneous foundation VLMs. To better support this protocol, we introduce BlenderBench, a challenging benchmark that stress-tests interleaved multimodal reasoning with graphics engine, where VIGA improves by 124.70%.

**arXiv ID:** 2601.11109
</details>

<details>
<summary><strong>Statistical Reinforcement Learning in the Real World: A Survey of Challenges and Future Directions</strong> - Asim H. Gazi, Yongyi Guo, Daiqi Gao, Ziping Xu, Kelly W. Zhang, Susan A. Murphy - [[pdf]](https://arxiv.org/pdf/2601.15353)</summary>

**Abstract:** Reinforcement learning (RL) has achieved remarkable success in real-world decision-making across diverse domains, including gaming, robotics, online advertising, public health, and natural language processing. Despite these advances, a substantial gap remains between RL research and its deployment in many practical settings. Two recurring challenges often underlie this gap. First, many settings offer limited opportunity for the agent to interact extensively with the target environment due to practical constraints. Second, many target environments often undergo substantial changes, requiring redesign and redeployment of RL systems (e.g., advancements in science and technology that change the landscape of healthcare delivery). Addressing these challenges and bridging the gap between basic research and application requires theory and methodology that directly inform the design, implementation, and continual improvement of RL systems in real-world settings.
In this paper, we frame the application of RL in practice as a three-component process: (i) online learning and optimization during deployment, (ii) post- or between-deployment offline analyses, and (iii) repeated cycles of deployment and redeployment to continually improve the RL system. We provide a narrative review of recent advances in statistical RL that address these components, including methods for maximizing data utility for between-deployment inference, enhancing sample efficiency for online learning within-deployment, and designing sequences of deployments for continual improvement. We also outline future research directions in statistical RL that are use-inspired -- aiming for impactful application of RL in practice.

**arXiv ID:** 2601.15353
</details>

<details>
<summary><strong>GenPO: Generative Diffusion Models Meet On-Policy Reinforcement Learning</strong> - Shutong Ding, Ke Hu, Shan Zhong, Haoyang Luo, Weinan Zhang, Jingya Wang, Jun Wang, Ye Shi - [[pdf]](https://arxiv.org/pdf/2505.18763)</summary>

**Abstract:** Recent advances in reinforcement learning (RL) have demonstrated the powerful exploration capabilities and multimodality of generative diffusion-based policies. While substantial progress has been made in offline RL and off-policy RL settings, integrating diffusion policies into on-policy frameworks like PPO remains underexplored. This gap is particularly significant given the widespread use of large-scale parallel GPU-accelerated simulators, such as IsaacLab, which are optimized for on-policy RL algorithms and enable rapid training of complex robotic tasks. A key challenge lies in computing state-action log-likelihoods under diffusion policies, which is straightforward for Gaussian policies but intractable for flow-based models due to irreversible forward-reverse processes and discretization errors (e.g., Euler-Maruyama approximations). To bridge this gap, we propose GenPO, a generative policy optimization framework that leverages exact diffusion inversion to construct invertible action mappings. GenPO introduces a novel doubled dummy action mechanism that enables invertibility via alternating updates, resolving log-likelihood computation barriers. Furthermore, we also use the action log-likelihood for unbiased entropy and KL divergence estimation, enabling KL-adaptive learning rates and entropy regularization in on-policy updates. Extensive experiments on eight IsaacLab benchmarks, including legged locomotion (Ant, Humanoid, Anymal-D, Unitree H1, Go2), dexterous manipulation (Shadow Hand), aerial control (Quadcopter), and robotic arm tasks (Franka), demonstrate GenPO's superiority over existing RL baselines. Notably, GenPO is the first method to successfully integrate diffusion policies into on-policy RL, unlocking their potential for large-scale parallelized training and real-world robotic deployment.

**arXiv ID:** 2505.18763
</details>

<details>
<summary><strong>Integrating Neural Differential Forecasting with Safe Reinforcement Learning for Blood Glucose Regulation</strong> - Yushen Liu, Yanfu Zhang, Xugui Zhou - [[pdf]](https://arxiv.org/pdf/2511.12417)</summary>

**Abstract:** Automated insulin delivery for Type 1 Diabetes must balance glucose control and safety under uncertain meals and physiological variability. While reinforcement learning (RL) enables adaptive personalization, existing approaches struggle to simultaneously guarantee safety, leaving a gap in achieving both personalized and risk-aware glucose control, such as overdosing before meals or stacking corrections. To bridge this gap, we propose TSODE, a safety-aware controller that integrates Thompson Sampling RL with a Neural Ordinary Differential Equation (NeuralODE) forecaster to address this challenge. Specifically, the NeuralODE predicts short-term glucose trajectories conditioned on proposed insulin doses, while a conformal calibration layer quantifies predictive uncertainty to reject or scale risky actions. In the FDA-approved UVa/Padova simulator (adult cohort), TSODE achieved 87.9% time-in-range with less than 10% time below 70 mg/dL, outperforming relevant baselines. These results demonstrate that integrating adaptive RL with calibrated NeuralODE forecasting enables interpretable, safe, and robust glucose regulation.

**arXiv ID:** 2511.12417
</details>

<details>
<summary><strong>A Mobile Magnetic Manipulation Platform for Gastrointestinal Navigation with Deep Reinforcement Learning Control</strong> - Zhifan Yan, Chang Liu, Yiyang Jiang, Wenxuan Zheng, Xinhao Chen, Axel Krieger - [[pdf]](https://arxiv.org/pdf/2601.15545)</summary>

**Abstract:** Targeted drug delivery in the gastrointestinal (GI) tract using magnetic robots offers a promising alternative to systemic treatments. However, controlling these robots is a major challenge. Stationary magnetic systems have a limited workspace, while mobile systems (e.g., coils on a robotic arm) suffer from a "model-calibration bottleneck", requiring complex, pre-calibrated physical models that are time-consuming to create and computationally expensive. This paper presents a compact, low-cost mobile magnetic manipulation platform that overcomes this limitation using Deep Reinforcement Learning (DRL). Our system features a compact four-electromagnet array mounted on a UR5 collaborative robot. A Soft Actor-Critic (SAC)-based control strategy is trained through a sim-to-real pipeline, enabling effective policy deployment within 15 minutes and significantly reducing setup time. We validated the platform by controlling a 7-mm magnetic capsule along 2D trajectories. Our DRL-based controller achieved a root-mean-square error (RMSE) of 1.18~mm for a square path and 1.50~mm for a circular path. We also demonstrated successful tracking over a clinically relevant, 30 cm * 20 cm workspace. This work demonstrates a rapidly deployable, model-free control framework capable of precise magnetic manipulation in a large workspace,validated using a 2D GI phantom.

**arXiv ID:** 2601.15545
</details>

<details>
<summary><strong>AION: Aerial Indoor Object-Goal Navigation Using Dual-Policy Reinforcement Learning</strong> - Zichen Yan, Yuchen Hou, Shenao Wang, Yichao Gao, Rui Huang, Lin Zhao - [[pdf]](https://arxiv.org/pdf/2601.15614)</summary>

**Abstract:** Object-Goal Navigation (ObjectNav) requires an agent to autonomously explore an unknown environment and navigate toward target objects specified by a semantic label. While prior work has primarily studied zero-shot ObjectNav under 2D locomotion, extending it to aerial platforms with 3D locomotion capability remains underexplored. Aerial robots offer superior maneuverability and search efficiency, but they also introduce new challenges in spatial perception, dynamic control, and safety assurance. In this paper, we propose AION for vision-based aerial ObjectNav without relying on external localization or global maps. AION is an end-to-end dual-policy reinforcement learning (RL) framework that decouples exploration and goal-reaching behaviors into two specialized policies. We evaluate AION on the AI2-THOR benchmark and further assess its real-time performance in IsaacSim using high-fidelity drone models. Experimental results show that AION achieves superior performance across comprehensive evaluation metrics in exploration, navigation efficiency, and safety. The video can be found at this https URL.

**arXiv ID:** 2601.15614
</details>

<details>
<summary><strong>D-Optimality-Guided Reinforcement Learning for Efficient Open-Loop Calibration of a 3-DOF Ankle Rehabilitation Robot</strong> - Qifan Hu, Branko Celler, Weidong Mu, Steven W. Su - [[pdf]](https://arxiv.org/pdf/2601.15707)</summary>

**Abstract:** Accurate alignment of multi-degree-of-freedom rehabilitation robots is essential for safe and effective patient training. This paper proposes a two-stage calibration framework for a self-designed three-degree-of-freedom (3-DOF) ankle rehabilitation robot. First, a Kronecker-product-based open-loop calibration method is developed to cast the input-output alignment into a linear parameter identification problem, which in turn defines the associated experimental design objective through the resulting information matrix. Building on this formulation, calibration posture selection is posed as a combinatorial design-of-experiments problem guided by a D-optimality criterion, i.e., selecting a small subset of postures that maximises the determinant of the information matrix. To enable practical selection under constraints, a Proximal Policy Optimization (PPO) agent is trained in simulation to choose 4 informative postures from a candidate set of 50. Across simulation and real-robot evaluations, the learned policy consistently yields substantially more informative posture combinations than random selection: the mean determinant of the information matrix achieved by PPO is reported to be more than two orders of magnitude higher with reduced variance. In addition, real-world results indicate that a parameter vector identified from only four D-optimality-guided postures provides stronger cross-episode prediction consistency than estimates obtained from a larger but unstructured set of 50 postures. The proposed framework therefore improves calibration efficiency while maintaining robust parameter estimation, offering practical guidance for high-precision alignment of multi-DOF rehabilitation robots.

**arXiv ID:** 2601.15707
</details>

<details>
<summary><strong>Reinforcement Learning Compensated Model Predictive Control for Off-road Driving on Unknown Deformable Terrain</strong> - Prakhar Gupta, Jonathon M. Smereka, Yunyi Jia - [[pdf]](https://arxiv.org/pdf/2408.09253)</summary>

**Abstract:** This study presents an Actor-Critic reinforcement learning Compensated Model Predictive Controller (AC2MPC) designed for high-speed, off-road autonomous driving on deformable terrains. Addressing the difficulty of modeling unknown tire-terrain interaction and ensuring real-time control feasibility and performance, this framework integrates deep reinforcement learning with a model predictive controller to manage unmodeled nonlinear dynamics. We evaluate the controller framework over constant and varying velocity profiles using high-fidelity simulator Project Chrono. Our findings demonstrate that our controller statistically outperforms standalone model-based and learning-based controllers over three unknown terrains that represent sandy deformable track, sandy and rocky track and cohesive clay-like deformable soil track. Despite varied and previously unseen terrain characteristics, this framework generalized well enough to track longitudinal reference speeds with the least error. Furthermore, this framework required significantly less training data compared to purely learning based controller, converging in fewer steps while delivering better performance. Even when under-trained, this controller outperformed the standalone controllers, highlighting its potential for safer and more efficient real-world deployment.

**arXiv ID:** 2408.09253
</details>

<details>
<summary><strong>UXCascade: Scalable Usability Testing with Simulated User Agents</strong> - Steffen Holter, Eunyee Koh, Mustafa Doga Dogan, Gromit Yeuk-Yin Chan - [[pdf]](https://arxiv.org/pdf/2601.15777)</summary>

**Abstract:** Simulated user agents are increasingly used in usability testing to support fast, iterative UX workflows, as they generate rich data such as action logs and think-aloud reasoning, but the unstructured nature of this output often obscures actionable insights. We present UXCascade, an interactive tool for extracting, aggregating, and presenting agent-generated usability feedback at scale. Our core contribution is a multi-level analysis workflow that (1) highlights patterns across persona traits, goals, and outcomes, (2) links agent reasoning to specific issues, and (3) supports actionable design improvements. UXCascade operationalizes this approach by listing agent goals, traits, and issues in a structured overview. Practitioners can explore detailed reasoning traces and annotated views, propose interface edits, and assess their impact across personas. This enables a top-down, exploration-driven analysis from patterns to concrete UX interventions. A user study with eight UX professionals demonstrates that UXCascade integrates into existing workflows, enabling iterative feedback during early-stage interface development.

**arXiv ID:** 2601.15777
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
