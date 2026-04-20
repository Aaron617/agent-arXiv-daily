# Agent arXiv Daily

**Last Updated:** 2026-04-20 03:49:51

**Total Papers:** 78

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
<summary><strong>Eco-Bee: A Personalised Multi-Modal Agent for Advancing Student Climate Awareness and Sustainable Behaviour in Campus Ecosystems</strong> - Caleb Adu, Neil Kapadia, Binhe Liu, Jonathan Randall, Sruthi Viswanathan - [[pdf]](https://arxiv.org/pdf/2604.15327)</summary>

**Abstract:** Universities are microcosms of urban ecosystems, with concentrated consumption patterns in food, transport, energy, and product usage. These environments not only contribute substantially to sustainability pressures but also provide a unique opportunity to advance sustainability education and behavioural change at scale. As in most sectors, digital sustainability initiatives within universities remain narrowly focused on carbon calculations, typically providing static feedback that limits opportunities for sustained behavioural change. To address this gap, we propose Eco-Bee, integrating large language models, a translation of the Planetary Boundaries framework (as Eco-Score), and a conversational agent that connects individual choices to environmental limits. Tailored for students at the cusp of lifelong habits, Eco-Bee delivers actionable insights, peer benchmarking, and gamified challenges to sustain engagement and drive measurable progress toward boundary-aligned living. In a pilot tested across multiple campus networks (n=52), 96% of the student participants supported a campus-wide rollout and reported a clearer understanding of how daily behaviours collectively impact the planet's limits. By embedding planetary science, behavioural reinforcement, and AI-driven personalisation into a single platform, Eco-Bee establishes a scalable foundation for climate-conscious universities and future AI-mediated sustainability infrastructures.

**arXiv ID:** 2604.15327
</details>

<details>
<summary><strong>SocialWise: LLM-Agentic Conversation Therapy for Individuals with Autism Spectrum Disorder to Enhance Communication Skills</strong> - Albert Tang - [[pdf]](https://arxiv.org/pdf/2604.15347)</summary>

**Abstract:** Autism Spectrum Disorder (ASD) affects more than 75 million people worldwide. However, scalable support for practicing everyday conversation is scarce: Low-cost activities such as story reading yield limited improvement. At the same time, effective role-play therapy demands expensive, in-person sessions with specialists. SocialWise bridges this gap through a browser-based application that pairs LLM conversational agents with a therapeutic retrieval augmented generation (RAG) knowledge base. Users select a scenario (e.g., ordering food, joining a group), interact by text or voice, and receive instant, structured feedback on tone, engagement, and alternative phrasing. The SocialWise prototype, implemented with Streamlit, LangChain, and ChromaDB, runs on any computer with internet access, and demonstrates how recent advances in LLM can provide evidence-based, on-demand communication coaching for individuals with ASD.

**arXiv ID:** 2604.15347
</details>

<details>
<summary><strong>HarmfulSkillBench: How Do Harmful Skills Weaponize Your Agents?</strong> - Yukun Jiang, Yage Zhang, Michael Backes, Xinyue Shen, Yang Zhang - [[pdf]](https://arxiv.org/pdf/2604.15415)</summary>

**Abstract:** Large language models (LLMs) have evolved into autonomous agents that rely on open skill ecosystems (e.g., ClawHub and this http URL), hosting numerous publicly reusable skills. Existing security research on these ecosystems mainly focuses on vulnerabilities within skills, such as prompt injection. However, there is a critical gap regarding skills that may be misused for harmful actions (e.g., cyber attacks, fraud and scams, privacy violations, and sexual content generation), namely harmful skills. In this paper, we present the first large-scale measurement study of harmful skills in agent ecosystems, covering 98,440 skills across two major registries. Using an LLM-driven scoring system grounded in our harmful skill taxonomy, we find that 4.93% of skills (4,858) are harmful, with ClawHub exhibiting an 8.84% harmful rate compared to 3.49% on this http URL. We then construct HarmfulSkillBench, the first benchmark for evaluating agent safety against harmful skills in realistic agent contexts, comprising 200 harmful skills across 20 categories and four evaluation conditions. By evaluating six LLMs on HarmfulSkillBench, we find that presenting a harmful task through a pre-installed skill substantially lowers refusal rates across all models, with the average harm score rising from 0.27 without the skill to 0.47 with it, and further to 0.76 when the harmful intent is implicit rather than stated as an explicit user request. We responsibly disclose our findings to the affected registries and release our benchmark to support future research (see this https URL).

**arXiv ID:** 2604.15415
</details>

<details>
<summary><strong>Deep Learning Based Amharic Chatbot for FAQs in Universities</strong> - Goitom Ybrah Hailu, Hadush Hailu, Shishay Welay - [[pdf]](https://arxiv.org/pdf/2402.01720)</summary>

**Abstract:** University students often spend a considerable amount of time seeking answers to common questions from administrators or teachers. This can become tedious for both parties, leading to a need for a solution. In response, this paper proposes a chatbot model that utilizes natural language processing and deep learning techniques to answer frequently asked questions (FAQs) in the Amharic language. Chatbots are computer programs that simulate human conversation through the use of artificial intelligence (AI), acting as a virtual assistant to handle questions and other tasks. The proposed chatbot program employs tokenization, normalization, stop word removal, and stemming to analyze and categorize Amharic input sentences. Three machine learning model algorithms were used to classify tokens and retrieve appropriate responses: Support Vector Machine (SVM), Multinomial Naïve Bayes, and deep neural networks implemented through TensorFlow, Keras, and NLTK. The deep learning model achieved the best results with 91.55% accuracy and a validation loss of 0.3548 using an Adam optimizer and SoftMax activation function. The chatbot model was integrated with Facebook Messenger and deployed on a Heroku server for 24-hour accessibility. The experimental results demonstrate that the chatbot framework achieved its objectives and effectively addressed challenges such as Amharic Fidel variation, morphological variation, and lexical gaps. Future research could explore the integration of Amharic WordNet to narrow the lexical gap and support more complex questions.

**arXiv ID:** 2402.01720
</details>

<details>
<summary><strong>Scalable Multi-Task Learning through Spiking Neural Networks with Adaptive Task-Switching Policy for Intelligent Autonomous Agents</strong> - Rachmad Vidya Wicaksana Putra, Avaneesh Devkota, Muhammad Shafique - [[pdf]](https://arxiv.org/pdf/2504.13541)</summary>

**Abstract:** Training resource-constrained autonomous agents on multiple tasks simultaneously is crucial for adapting to diverse real-world environments. Recent works employ reinforcement learning (RL) approach, but they still suffer from sub-optimal multi-task performance due to task interference. State-of-the-art works employ Spiking Neural Networks (SNNs) to improve RL-based multi-task learning and enable low-power/energy operations through network enhancements and spike-driven data stream processing. However, they rely on fixed task-switching intervals during its training, thus limiting its performance and scalability. To address this, we propose SwitchMT, a novel methodology that employs adaptive task-switching for effective, scalable, and simultaneous multi-task learning. SwitchMT employs the following key ideas: (1) leveraging a Deep Spiking Q-Network with active dendrites and dueling structure, that utilizes task-specific context signals to create specialized sub-networks; and (2) devising an adaptive task-switching policy that leverages both rewards and internal dynamics of the network parameters. Experimental results demonstrate that SwitchMT achieves competitive scores in multiple Atari games (i.e., Pong: -8.8, Breakout: 5.6, and Enduro: 355.2) and longer game episodes as compared to the state-of-the-art. These results also highlight the effectiveness of SwitchMT methodology in addressing task interference without increasing the network complexity, enabling intelligent autonomous agents with scalable multi-task learning capabilities.

**arXiv ID:** 2504.13541
</details>

<details>
<summary><strong>Restoration, Exploration and Transformation: How Youth Engage Character.AI Chatbots for Feels, Fun and Finding themselves</strong> - Annabel Blake, Marcus Carter, Eduardo Velloso - [[pdf]](https://arxiv.org/pdf/2604.15340)</summary>

**Abstract:** Young people are among the fastest adopters of generative AI, yet research emphasises adult-designed tools and experiments rather than playful, self-directed youth use. We analysed discourse from 4,172 users in this http URL's official Discord, finding that the most engaged users were predominantly adolescents (50% aged 13-17), female or non-binary (61.9%), with most (59%) creating their own characters. We contribute (1) a descriptive account of how highly-engaged youth on this http URL's Discord use AI for playful, emotional, and creative practices that push the platform limits; (2) a framework of three engagement intents -- Restoration (emotional regulation), Exploration (creative experimentation), and Transformation (identity development); and (3) a taxonomy of seven youth-created character archetypes. Together, these findings reveal how youth invent novel roles for AI, expose critical misalignments between youth use and current AI experiences, and provide frameworks for researchers and practitioners to design youth-centred AI futures.

**arXiv ID:** 2604.15340
</details>

<details>
<summary><strong>GroupEnvoy: A Conversational Agent Speaking for the Outgroup to Foster Intergroup Relations</strong> - Koken Hata, Rintaro Chujo, Reina Takamatsu, Wenzhen Xu, Yukino Baba - [[pdf]](https://arxiv.org/pdf/2604.16095)</summary>

**Abstract:** Conversational agents have the potential to support intergroup relations when psychological or linguistic barriers prevent direct interaction. Based on intergroup contact theory, we propose GroupEnvoy, a conversational agent that represents outgroup perspectives during ingroup discussions, grounded in transcripts from outgroup-only sessions. To evaluate this approach and derive design principles, we conducted a mixed-methods, between-subjects study with university students, where host-country students formed the ingroup and international students formed the outgroup. Ingroup students performed a collaborative task, receiving outgroup perspectives via GroupEnvoy (experimental) or reading written transcripts (control). Compared to the control group, the experimental group showed greater reduction in intergroup anxiety and greater improvement in perspective-taking. Qualitatively, AI-mediated contact enhanced outcome expectancies, whereas passive exposure fostered future contact intentions. The two conditions also elicited empathy toward distinct targets: outgroup evaluations of the ingroup versus outgroup lived experiences. These findings validate AI-mediated contact as a promising paradigm for improving intergroup relations.

**arXiv ID:** 2604.16095
</details>

<details>
<summary><strong>Investigating Conversational Agents to Support Secondary School Students Learning CSP</strong> - Matthew Frazier, Kostadin Damevski, Lori Pollock - [[pdf]](https://arxiv.org/pdf/2604.16213)</summary>

**Abstract:** Secondary school students enrolled in the AP Computer Science Principles (CSP) course commonly utilize web resources (e.g., tutorials, Q\&A sites) to better understand key concepts in the curriculum. The primary obstacle to using these resources is finding information appropriate for the learning task and student's background. In addition to web search, conversational agents are increasingly a viable alternative for CSP students. In this paper, we study the potential of conversational agents to aid secondary school students as they acquire knowledge on CSP concepts. We explore general purpose, generative conversational agents (e.g., ChatGPT) and custom, fixed-response conversational agents built specifically to aid CSP students. We present results from classroom use by 45 high school students in grades 9-11 (ages 14-17) across six CSP sections. Our main contributions are in better understanding how conversational agents can help CSP students and an evaluation of the effectiveness and engagement of different approaches for CSP exploratory search.

**arXiv ID:** 2604.16213
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (11 papers)</h2></summary>

<details>
<summary><strong>GIST: Multimodal Knowledge Extraction and Spatial Grounding via Intelligent Semantic Topology</strong> - Shivendra Agrawal, Bradley Hayes - [[pdf]](https://arxiv.org/pdf/2604.15495)</summary>

**Abstract:** Navigating complex, densely packed environments like retail stores, warehouses, and hospitals poses a significant spatial grounding challenge for humans and embodied AI. In these spaces, dense visual features quickly become stale given the quasi-static nature of items, and long-tail semantic distributions challenge traditional computer vision. While Vision-Language Models (VLMs) help assistive systems navigate semantically-rich spaces, they still struggle with spatial grounding in cluttered environments. We present GIST (Grounded Intelligent Semantic Topology), a multimodal knowledge extraction pipeline that transforms a consumer-grade mobile point cloud into a semantically annotated navigation topology. Our architecture distills the scene into a 2D occupancy map, extracts its topological layout, and overlays a lightweight semantic layer via intelligent keyframe and semantic selection. We demonstrate the versatility of this structured spatial knowledge through critical downstream Human-AI interaction tasks: (1) an intent-driven Semantic Search engine that actively infers categorical alternatives and zones when exact matches fail; (2) a one-shot Semantic Localizer achieving a 1.04 m top-5 mean translation error; (3) a Zone Classification module that segments the walkable floor plan into high-level semantic regions; and (4) a Visually-Grounded Instruction Generator that synthesizes optimal paths into egocentric, landmark-rich natural language routing. In multi-criteria LLM evaluations, GIST outperforms sequence-based instruction generation baselines. Finally, an in-situ formative evaluation (N=5) yields an 80% navigation success rate relying solely on verbal cues, validating the system's capacity for universal design.

**arXiv ID:** 2604.15495
</details>

<details>
<summary><strong>Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4</strong> - Chengwu Liu, Yichun Yin, Ye Yuan, Jiaxuan Xie, Botao Li, Siqi Li, Jianhao Shen, Yan Xu, Lifeng Shang, Ming Zhang - [[pdf]](https://arxiv.org/pdf/2604.15839)</summary>

**Abstract:** Most ATP benchmarks embed the final answer within the formal statement -- a convention we call "Easy Mode" -- a design that simplifies the task relative to what human competitors face and may lead to optimistic estimates of model capability. We call the stricter, more realistic setting "Hard Mode": the system must independently discover the answer before constructing a formal proof. To enable Hard Mode research, we make two contributions. First, we release MiniF2F-Hard and FIMO-Hard, expert-reannotated Hard Mode variants of two widely-used ATP benchmarks. Second, we introduce Discover And Prove (DAP), an agentic framework that uses LLM natural-language reasoning with explicit self-reflection to discover answers, then rewrites Hard Mode statements into Easy Mode ones for existing ATP provers. DAP sets the state of the art: on CombiBench it raises solved problems from 7 (previous SOTA, Pass@16) to 10; on PutnamBench it is the first system to formally prove 36 theorems in Hard Mode -- while simultaneously revealing that state-of-the-art LLMs exceed 80% answer accuracy on the same problems where formal provers manage under 10%, exposing a substantial gap that Hard Mode benchmarks are uniquely suited to measure.

**arXiv ID:** 2604.15839
</details>

<details>
<summary><strong>LinuxArena: A Control Setting for AI Agents in Live Production Software Environments</strong> - Tyler Tracy, Ram Potham, Nick Kuhn, Myles Heller, Anshul Khandelwal, Cody Rushing, Henri Lemoine, Miguel Brandao, Tomas Turlik, Adam Hanson, Josh Hills, Amy Ngo, Ram Rachum, Nik Mitchell, Falko Galperin, Oscar Sykes, Pip Arnott, Samuel Prieto Lima, Carlos Giudice, Matt Goldwater, Daniel Popp, Drew de Wet, Ruben Castaing, Qi Guo, Douw Marx, Benjamin Shaffrey, Justin Shenk, Martin Milbradt, Hannah Meagher, Shaheen Ahmed-Chowdhury, Daniel O'Connell, Chris Canal, Buck Shlegeris, Aryan Bhatt - [[pdf]](https://arxiv.org/pdf/2604.15384)</summary>

**Abstract:** We introduce LinuxArena, a control setting in which agents operate directly on live, multi-service production environments. LinuxArena contains 20 environments, 1,671 main tasks representing legitimate software engineering work, and 184 side tasks representing safety failures such as data exfiltration and backdooring, making it the largest and most diverse control setting for software engineering to date. We validate LinuxArena is useful for control research by running sabotage evaluations, which measure whether attackers can complete side tasks while working on main tasks, and monitor evaluations, which measure a monitor model's ability to detect sabotage attempts. Against a GPT-5-nano trusted monitor at a 1\% step-wise false positive rate, Claude Opus 4.6 achieves roughly a 23% undetected sabotage success rate. We additionally release LaStraj, a dataset of human-crafted attack trajectories that evade monitors at substantially higher rates than any model-generated attacks we elicited, showing that current attack policies do not saturate LinuxArena. These results suggest that LinuxArena has meaningful headroom for both attackers and defenders, making it a strong testbed for developing and evaluating future control protocols.

**arXiv ID:** 2604.15384
</details>

<details>
<summary><strong>GTA-2: Benchmarking General Tool Agents from Atomic Tool-Use to Open-Ended Workflows</strong> - Jize Wang, Xuanxuan Liu, Yining Li, Songyang Zhang, Yijun Wang, Zifei Shan, Xinyi Le, Cailian Chen, Xinping Guan, Dacheng Tao - [[pdf]](https://arxiv.org/pdf/2604.15715)</summary>

**Abstract:** The development of general-purpose agents requires a shift from executing simple instructions to completing complex, real-world productivity workflows. However, current tool-use benchmarks remain misaligned with real-world requirements, relying on AI-generated queries, dummy tools, and limited system-level coordination. To address this, we propose GTA-2, a hierarchical benchmark for General Tool Agents (GTA) spanning atomic tool use and open-ended workflows. Built on real-world authenticity, it leverages real user queries, deployed tools, and multimodal contexts. (i) GTA-Atomic, inherited from our prior GTA benchmark, evaluates short-horizon, closed-ended tool-use precision. (ii) GTA-Workflow introduces long-horizon, open-ended tasks for realistic end-to-end completion. To evaluate open-ended deliverables, we propose a recursive checkpoint-based evaluation mechanism that decomposes objectives into verifiable sub-goals, enabling unified evaluation of both model capabilities and agent execution frameworks (i.e., execution harnesses). Experiments reveal a pronounced capability cliff: while frontier models already struggle on atomic tasks (below 50%), they largely fail on workflows, with top models achieving only 14.39% success. Further analysis shows that checkpoint-guided feedback improves performance, while advanced frameworks such as Manus and OpenClaw substantially enhance workflow completion, highlighting the importance of execution harness design beyond the underlying model capacity. These findings provide guidance for developing reliable personal and professional assistants. Dataset and code will be available at this https URL.

**arXiv ID:** 2604.15715
</details>

<details>
<summary><strong>Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective</strong> - Ritik Raj, Souvik Kundu, Ishita Vohra, Hong Wang, Tushar Krishna - [[pdf]](https://arxiv.org/pdf/2511.00739)</summary>

**Abstract:** Agentic AI serving converts monolithic LLM-based inference to autonomous problem-solvers that can plan, call tools, perform reasoning, and adapt on the fly. Due to diverse task execution need, such serving heavily rely on heterogeneous CPU-GPU systems with majority of the external tools responsible for agentic capability, either run on or are orchestrated by the CPU.
Towards having a deeper understanding of its role, this paper aims to characterize and analyze the system bottlenecks introduced by agentic AI workloads from a largely overlooked CPU-centric perspective. We first present a compile-time characterization of agentic AI execution and choose representative workloads to capture the algorithmic diversity. We then perform runtime characterization of the representative workloads analyzing the end-to-end latency and throughput on two different hardware systems to isolate respective architectural bottlenecks. Based on the insights on the bottlenecks, we finally present two scheduling optimizations, namely, 1. CPU-Aware Overlapped Micro-Batching (COMB) and 2. Mixed Agentic Scheduling (MAS) on homogeneous and heterogeneous agentic workloads, respectively. In specific, these methods optimize for improved CPU-GPU concurrent utilization while reducing skewed resource allocation for heterogeneous execution. Experimental evaluations on the two hardware systems demonstrate the efficacy of COMB in yielding up to 1.7x lower P50 latency in standalone homogeneous workload execution and up to 3.9x/1.8x lower service/total latency under homogeneous open-loop load. Additionally, for heterogeneous open-loop load, MAS can reduce the total latency for minority request-type by up to 2.37x/2.49x at P50/P90 percentile.

**arXiv ID:** 2511.00739
</details>

<details>
<summary><strong>ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence</strong> - ARC Prize Foundation - [[pdf]](https://arxiv.org/pdf/2603.24621)</summary>

**Abstract:** We introduce ARC-AGI-3, an interactive benchmark for studying agentic intelligence through novel, abstract, turn-based environments in which agents must explore, infer goals, build internal models of environment dynamics, and plan effective action sequences without explicit instructions. Like its predecessors ARC-AGI-1 and 2, ARC-AGI-3 focuses entirely on evaluating fluid adaptive efficiency on novel tasks, while avoiding language and external knowledge. ARC-AGI-3 environments only leverage Core Knowledge priors and are difficulty-calibrated via extensive testing with human test-takers. Our testing shows humans can solve 100% of the environments, in contrast to frontier AI systems which, as of March 2026, score below 1%. In this paper, we present the benchmark design, its efficiency-based scoring framework grounded in human action baselines, and the methodology used to construct, validate, and calibrate the environments.

**arXiv ID:** 2603.24621
</details>

<details>
<summary><strong>PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning</strong> - Zihan Zhao, Kaushik Pendiyala, Masood Mortazavi, Ning Yan - [[pdf]](https://arxiv.org/pdf/2510.24058)</summary>

**Abstract:** Multi-sensory systems for embodied intelligence, from wearable body-sensor networks to instrumented robotic platforms, routinely face a sensor-asymmetry problem: the richest modality available during laboratory data collection is absent or impractical at deployment time due to cost, fragility, or interference with physical interaction. We introduce PULSE, a general framework for privileged knowledge transfer from an information-rich teacher sensor to a set of cheaper, deployment-ready student sensors. Each student encoder produces shared (modality-invariant) and private (modality-specific) embeddings; the shared subspace is aligned across modalities and then matched to representations of a frozen teacher via multi-layer hidden-state and pooled-embedding distillation. Private embeddings preserve modality-specific structure needed for self-supervised reconstruction, which we show is critical to prevent representational collapse. We instantiate PULSE on the wearable stress-monitoring task, using electrodermal activity (EDA) as the privileged teacher and ECG, BVP, accelerometry, and temperature as students. On the WESAD benchmark under leave-one-subject-out evaluation, PULSE achieves 0.994 AUROC and 0.988 AUPRC (0.965/0.955 on STRESS) without EDA at inference, exceeding all no-EDA baselines and matching the performance of a full-sensor model that retains EDA at test time. We further demonstrate modality-agnostic transfer with ECG as teacher, provide extensive ablations on hidden-state matching depth, shared-private capacity, hinge-loss margin, fusion strategy, and modality dropout, and discuss how the framework generalizes to broader embodied sensing scenarios involving tactile, inertial, and bioelectrical modalities.

**arXiv ID:** 2510.24058
</details>

<details>
<summary><strong>Capture the Flags: Family-Based Evaluation of Agentic LLMs via Semantics-Preserving Transformations</strong> - Shahin Honarvar, Amber Gorzynski, James Lee-Jones, Harry Coppock, Marek Rei, Joseph Ryan, Alastair F. Donaldson - [[pdf]](https://arxiv.org/pdf/2602.05523)</summary>

**Abstract:** Agentic large language models (LLMs) are increasingly evaluated on cybersecurity tasks using capture-the-flag (CTF) benchmarks, yet existing pointwise benchmarks offer limited insight into agent robustness and generalisation across alternative versions of the source code. We introduce CTF challenge families, whereby a single CTF is used to generate a family of semantically-equivalent challenges via semantics-preserving program transformations, enabling controlled evaluation of robustness while keeping the underlying exploit strategy fixed. We present Evolve-CTF, a tool that generates CTF families from Python challenges using a range of transformations. Using Evolve-CTF to derive families from Cybench and Intercode challenges, we evaluate 13 agentic LLM configurations with tool access. We find that models are remarkably robust to renaming and code insertion, but that composed transformations and deeper obfuscation degrade performance by requiring more sophisticated tool use. Enabling explicit reasoning has little effect on success rates. Our work contributes a technique and tool for future LLM evaluations, and a large dataset characterising the capabilities of current state-of-the-art models in this domain.

**arXiv ID:** 2602.05523
</details>

<details>
<summary><strong>FACTS: Table Summarization via Offline Template Generation with Agentic Workflows</strong> - Ye Yuan, Mohammad Amin Shabani, Siqi Liu - [[pdf]](https://arxiv.org/pdf/2510.13920)</summary>

**Abstract:** Query-focused table summarization requires generating natural language summaries of tabular data conditioned on a user query, enabling users to access insights beyond fact retrieval. Existing approaches face key limitations: table-to-text models require costly fine-tuning and struggle with complex reasoning, prompt-based LLM methods suffer from token-limit and efficiency issues while exposing sensitive data, and prior agentic pipelines often rely on decomposition, planning, or manual templates that lack robustness and scalability. To mitigate these issues, we introduce an agentic workflow, FACTS, a Fast, Accurate, and Privacy-Compliant Table Summarization approach via Offline Template Generation. FACTS produces offline templates, consisting of SQL queries and Jinja2 templates, which can be rendered into natural language summaries and are reusable across multiple tables sharing the same schema. It enables fast summarization through reusable offline templates, accurate outputs with executable SQL queries, and privacy compliance by sending only table schemas to LLMs. Evaluations on widely-used benchmarks show that FACTS consistently outperforms baseline methods, establishing it as a practical solution for real-world query-focused table summarization. Our code is available at this https URL.

**arXiv ID:** 2510.13920
</details>

<details>
<summary><strong>FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents</strong> - Chiwei Zhu, Benfeng Xu, Mingxuan Du, Shaohan Wang, Xiaorui Wang, Zhendong Mao, Yongdong Zhang - [[pdf]](https://arxiv.org/pdf/2602.01566)</summary>

**Abstract:** Deep research is emerging as a representative long-horizon task for large language model (LLM) agents. However, long trajectories in deep research often exceed model context limits, compressing token budgets for both evidence collection and report writing, and preventing effective test-time scaling. We introduce FS-Researcher, a file-system-based, dual-agent framework that scales deep research beyond the context window via a persistent workspace. Specifically, a Context Builder agent acts as a librarian which browses the internet, writes structured notes, and archives raw sources into a hierarchical knowledge base that can grow far beyond context length. A Report Writer agent then composes the final report section by section, treating the knowledge base as the source of facts. In this framework, the file system serves as a durable external memory and a shared coordination medium across agents and sessions, enabling iterative refinement beyond the context window. Experiments on two open-ended benchmarks (DeepResearch Bench and DeepConsult) show that FS-Researcher achieves state-of-the-art report quality across different backbone models. Further analyses demonstrate a positive correlation between final report quality and the computation allocated to the Context Builder, validating effective test-time scaling under the file-system paradigm. The code and data are open-sourced at this https URL.

**arXiv ID:** 2602.01566
</details>

<details>
<summary><strong>Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue</strong> - Junan Hu, Shudan Guo, Wenqi Liu, Jianhua Yin, Yinwei Wei - [[pdf]](https://arxiv.org/pdf/2604.05552)</summary>

**Abstract:** Large Language Models demonstrate outstanding performance in many language tasks but still face fundamental challenges in managing the non-linear flow of human conversation. The prevalent approach of treating dialogue history as a flat, linear sequence is misaligned with the intrinsically hierarchical and branching structure of natural discourse, leading to inefficient context utilization and a loss of coherence during extended interactions involving topic shifts or instruction refinements. To address this limitation, we introduce Context-Agent, a novel framework that models multi-turn dialogue history as a dynamic tree structure. This approach mirrors the inherent non-linearity of conversation, enabling the model to maintain and navigate multiple dialogue branches corresponding to different topics. Furthermore, to facilitate robust evaluation, we introduce the Non-linear Task Multi-turn Dialogue (NTM) benchmark, specifically designed to assess model performance in long-horizon, non-linear scenarios. Our experiments demonstrate that Context-Agent enhances task completion rates and improves token efficiency across various LLMs, underscoring the value of structured context management for complex, dynamic dialogues. The dataset and code is available at GitHub.

**arXiv ID:** 2604.05552
</details>

</details>

<details open>
<summary><h2>LLM Agents (11 papers)</h2></summary>

<details>
<summary><strong>The World Leaks the Future: Harness Evolution for Future Prediction Agents</strong> - Chuyang Wei, Maohang Gao, Zhixin Han, Kefei Chen, Yu Zhuang, Haoxiang Guan, Yanzhi Zhang, Yilin Cheng, Jiyan He, Huanhuan Chen, Jian Li, Yu Shi, Yitong Duan, Shuxin Zheng - [[pdf]](https://arxiv.org/pdf/2604.15719)</summary>

**Abstract:** Many consequential decisions must be made before the relevant outcome is known. Such problems are commonly framed as \emph{future prediction}, where an LLM agent must form a prediction for an unresolved question using only the public information available at the prediction time. The setting is difficult because public evidence evolves while useful supervision arrives only after the question is resolved, so most existing approaches still improve mainly from final outcomes. Yet final outcomes are too coarse to guide earlier factor tracking, evidence gathering and interpretation, or uncertainty handling. When the same unresolved question is revisited over time, temporal contrasts between earlier and later predictions can expose omissions in the earlier prediction process; we call this signal \emph{internal feedback}. We introduce \emph{Milkyway}, a self-evolving agent system that keeps the base model fixed and instead updates a persistent \emph{future prediction harness} for factor tracking, evidence gathering and interpretation, and uncertainty handling. Across repeated predictions on the same unresolved question, \emph{Milkyway} extracts internal feedback and writes reusable guidance back into the harness, so later predictions on that question can improve before the outcome is known. After the question is resolved, the final outcome provides a \emph{retrospective check} before the updated harness is carried forward to subsequent questions. On FutureX and FutureWorld, Milkyway achieves the best overall score among the compared methods, improving FutureX from 44.07 to 60.90 and FutureWorld from 62.22 to 77.96.

**arXiv ID:** 2604.15719
</details>

<details>
<summary><strong>Experience Compression Spectrum: Unifying Memory, Skills, and Rules in LLM Agents</strong> - Xing Zhang, Guanghui Wang, Yanwei Cui, Wei Qiu, Ziyuan Li, Bing Zhu, Peiyang He - [[pdf]](https://arxiv.org/pdf/2604.15877)</summary>

**Abstract:** As LLM agents scale to long-horizon, multi-session deployments, efficiently managing accumulated experience becomes a critical bottleneck. Agent memory systems and agent skill discovery both address this challenge -- extracting reusable knowledge from interaction traces -- yet a citation analysis of 1,136 references across 22 primary papers reveals a cross-community citation rate below 1%. We propose the \emph{Experience Compression Spectrum}, a unifying framework that positions memory, skills, and rules as points along a single axis of increasing compression (5--20$\times$ for episodic memory, 50--500$\times$ for procedural skills, 1,000$\times$+ for declarative rules), directly reducing context consumption, retrieval latency, and compute overhead. Mapping 20+ systems onto this spectrum reveals that every system operates at a fixed, predetermined compression level -- none supports adaptive cross-level compression, a gap we term the \emph{missing diagonal}. We further show that specialization alone is insufficient -- both communities independently solve shared sub-problems without exchanging solutions -- that evaluation methods are tightly coupled to compression levels, that transferability increases with compression at the cost of specificity, and that knowledge lifecycle management remains largely neglected. We articulate open problems and design principles for scalable, full-spectrum agent learning systems.

**arXiv ID:** 2604.15877
</details>

<details>
<summary><strong>Explainable Iterative Data Visualisation Refinement via an LLM Agent</strong> - Burak Susam, Tingting Mu - [[pdf]](https://arxiv.org/pdf/2604.15319)</summary>

**Abstract:** Exploratory analysis of high-dimensional data relies on embedding the data into a low-dimensional space (typically 2D or 3D), based on which visualization plot is produced to uncover meaningful structures and to communicate geometric and distributional data characteristics. However, finding a suitable algorithm configuration, particularly hyperparameter setting, to produce a visualization plot that faithfully represents the underlying reality and encourages pattern discovery remains challenging. To address this challenge, we propose an agentic AI pipleline that leverages a large language model (LLM) to bridge the gap between rigorous quantitative assessment and qualitative human insight. By treating visualization evaluation and hyperparameter optimization as a semantic task, our system generates a multi-faceted report that contextualizes hard metrics with descriptive summaries, and suggests actionable recommendation of algorithm configuration for refining data visualization. By implementing an iterative optimization loop of this process, the system is able to produce rapidly a high-quality visualization plot, in full automation.

**arXiv ID:** 2604.15319
</details>

<details>
<summary><strong>PolicyBank: Evolving Policy Understanding for LLM Agents</strong> - Jihye Choi, Jinsung Yoon, Long T. Le, Somesh Jha, Tomas Pfister - [[pdf]](https://arxiv.org/pdf/2604.15505)</summary>

**Abstract:** LLM agents operating under organizational policies must comply with authorization constraints typically specified in natural language. In practice, such specifications inevitably contain ambiguities and logical or semantic gaps that cause the agent's behavior to systematically diverge from the true requirements. We ask: by letting an agent evolve its policy understanding through interaction and corrective feedback from pre-deployment testing, can it autonomously refine its interpretation to close specification gaps? We propose PolicyBank, a memory mechanism that maintains structured, tool-level policy insights and iteratively refines them -- unlike existing memory mechanisms that treat the policy as immutable ground truth, reinforcing "compliant but wrong" behaviors. We also contribute a systematic testbed by extending a popular tool-calling benchmark with controlled policy gaps that isolate alignment failures from execution failures. While existing memory mechanisms achieve near-zero success on policy-gap scenarios, PolicyBank closes up to 82% of the gap toward a human oracle.

**arXiv ID:** 2604.15505
</details>

<details>
<summary><strong>The Amazing Agent Race: Strong Tool Users, Weak Navigators</strong> - Zae Myung Kim, Dongseok Lee, Jaehyung Kim, Vipul Raheja, Dongyeop Kang - [[pdf]](https://arxiv.org/pdf/2604.10261)</summary>

**Abstract:** Existing tool-use benchmarks for LLM agents are overwhelmingly linear: our analysis of six benchmarks shows 55 to 100% of instances are simple chains of 2 to 5 steps. We introduce The Amazing Agent Race (AAR), a benchmark featuring directed acyclic graph (DAG) puzzles (or "legs") with fork-merge tool chains. We release 1,400 instances across two variants: sequential (800 legs) and compositional (600 DAG legs). Agents must navigate Wikipedia, execute multi-step tool chains, and aggregate results into a verifiable answer. Legs are procedurally generated from Wikipedia seeds across four difficulty levels with live-API validation. Three complementary metrics (finish-line accuracy, pit-stop visit rate, and roadblock completion rate) separately diagnose navigation, tool-use, and arithmetic failures. Evaluating three agent frameworks on 1,400 legs, the best achieves only 37.2% accuracy. Navigation errors dominate (27 to 52% of trials) while tool-use errors remain below 17%, and agent architecture matters as much as model scale (Claude Code matches Codex CLI at 37% with 6x fewer tokens). The compositional structure of AAR reveals that agents fail not at calling tools but at navigating to the right pages, a blind spot invisible to linear benchmarks. The project page can be accessed at: this https URL

**arXiv ID:** 2604.10261
</details>

<details>
<summary><strong>EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis</strong> - Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2601.05808)</summary>

**Abstract:** Large language models (LLMs) are expected to be trained to act as agents in various real-world environments, but this process relies on rich and varied tool-interaction sandboxes. However, access to real systems is often restricted; LLM-simulated environments are prone to hallucinations and inconsistencies; and manually built sandboxes are hard to scale. In this paper, we propose EnvScaler, an automated framework for scalable tool-interaction environments via programmatic synthesis. EnvScaler comprises two components. First, SkelBuilder constructs diverse environment skeletons through topic mining, logic modeling, and quality evaluation. Then, ScenGenerator generates multiple task scenarios and rule-based trajectory validation functions for each environment. With EnvScaler, we synthesize 191 environments and about 7K scenarios, and apply them to Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) for Qwen3 series models. Results on three benchmarks show that EnvScaler significantly improves LLMs' ability to solve tasks in complex environments involving multi-turn, multi-tool interactions. We release our code and data at this https URL.

**arXiv ID:** 2601.05808
</details>

<details>
<summary><strong>SoK: Security of Autonomous LLM Agents in Agentic Commerce</strong> - Qian'ang Mao, Jiaxin Wang, Ya Liu, Li Zhu, Cong Ma, Jiaqi Yan - [[pdf]](https://arxiv.org/pdf/2604.15367)</summary>

**Abstract:** Autonomous large language model (LLM) agents such as OpenClaw are pushing agentic commerce from human-supervised assistance toward machine actors that can negotiate, purchase services, manage digital assets, and execute transactions across on-chain and off-chain environments. Protocols such as the Trustless Agents standard (ERC-8004), Agent Payments Protocol (AP2), the HTTP 402-based payment protocol (x402), Agent Commerce Protocol (ACP), the Agentic Commerce standard (ERC-8183), and Machine Payments Protocol (MPP) enable this transition, but they also create an attack surface that existing security frameworks do not capture well. This Systematization of Knowledge (SoK) develops a unified security framework for autonomous LLM agents in commerce and finance. We organize threats along five dimensions: agent integrity, transaction authorization, inter-agent trust, market manipulation, and regulatory compliance. From a systematically curated public corpus of academic papers, protocol documents, industry reports, and incident evidence, we derive 12 cross-layer attack vectors and show how failures propagate from reasoning and tooling layers into custody, settlement, market harm, and compliance exposure. We then propose a layered defense architecture addressing authorization gaps left by current agent-payment protocols. Overall, our analysis shows that securing agentic commerce is inherently a cross-layer problem that requires coordinated controls across LLM safety, protocol design, identity, market structure, and regulation. We conclude with a research roadmap and a benchmark agenda for secure autonomous commerce.

**arXiv ID:** 2604.15367
</details>

<details>
<summary><strong>MemEvoBench: Benchmarking Memory MisEvolution in LLM Agents</strong> - Weiwei Xie, Shaoxiong Guo, Fan Zhang, Tian Xia, Xue Yang, Lizhuang Ma, Junchi Yan, Qibing Ren - [[pdf]](https://arxiv.org/pdf/2604.15774)</summary>

**Abstract:** Equipping Large Language Models (LLMs) with persistent memory enhances interaction continuity and personalization but introduces new safety risks. Specifically, contaminated or biased memory accumulation can trigger abnormal agent behaviors. Existing evaluation methods have not yet established a standardized framework for measuring memory misevolution. This phenomenon refers to the gradual behavioral drift resulting from repeated exposure to misleading information. To address this gap, we introduce MemEvoBench, the first benchmark evaluating long-horizon memory safety in LLM agents against adversarial memory injection, noisy tool outputs, and biased feedback. The framework consists of QA-style tasks across 7 domains and 36 risk types, complemented by workflow-style tasks adapted from 20 Agent-SafetyBench environments with noisy tool returns. Both settings employ mixed benign and misleading memory pools within multi-round interactions to simulate memory evolution. Experiments on representative models reveal substantial safety degradation under biased memory updates. Our analysis suggests that memory evolution is a significant contributor to these failures. Furthermore, static prompt-based defenses prove insufficient, underscoring the urgency of securing memory evolution in LLM agents.

**arXiv ID:** 2604.15774
</details>

<details>
<summary><strong>CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution</strong> - Shidong Yang, Ziyu Ma, Tongwen Huang, Yiming Hu, Yong Wang, Xiangxiang Chu - [[pdf]](https://arxiv.org/pdf/2604.15840)</summary>

**Abstract:** Reinforcement learning for LLM agents is typically conducted on a static data distribution, which fails to adapt to the agent's evolving behavior and leads to poor coverage of complex environment interactions. To address these challenges, we propose CoEvolve, an agent-data mutual evolution framework that enables LLM agents to improve through closed-loop, interaction-driven training. Specifically, CoEvolve extracts feedback signals such as forgetting and uncertainty from rollout trajectories to identify failure-prone interaction patterns, and utilizes them to guide LLM-based task synthesis. The synthesized tasks are validated through environment interaction and utilized to update the data distribution, enabling joint adaptation of the agent and its data. Extensive experiments on AppWorld and BFCL across Qwen2.5-7B, Qwen3-4B, and Qwen3-30B-A3B demonstrate consistent and significant improvements over strong base models, yielding absolute gains of 19.43%, 15.58%, and 18.14%, respectively.

**arXiv ID:** 2604.15840
</details>

<details>
<summary><strong>COMPASS: Benchmarking Constrained Optimization in LLM Agents</strong> - Tian Qin, Felix Bai, Ting-Yao Hu, Raviteja Vemulapalli, Hema Swetha Koppula, Zhiyang Xu, Bowen Jin, Mert Cemri, Jiarui Lu, Zirui Wang, Meng Cao - [[pdf]](https://arxiv.org/pdf/2510.07043)</summary>

**Abstract:** Human decision-making often involves constrained optimization. As LLM agents are deployed to assist with real-world tasks like travel planning, shopping, and scheduling, they must mirror this capability. We introduce COMPASS, a benchmark that evaluates whether LLM agents can perform constrained optimization in realistic travel planning settings. To success in these tasks, agents must engage in multi-turn conversations with user to gather task information as well as use tools to gather information from the database. Then agents must propose a solution that not only satisfies hard constraints but also optimizes user's utility objective. Evaluating state-of-the-art models, we reveal a significant feasible-optimal gap: while models achieve 70-90% feasibility (constraint satisfaction), they reach only 20-60% optimality (utility optimization). Our analysis shows that tool use is not the bottleneck. Instead, the core limitation is insufficient exploration of the search space, with success strongly correlating with information gathered. Coding agents show a promising approach to mitigate this gap. Together, COMPASS provides a testbed for developing LLM agents that can truly mirror human decision-making by both satisfying constraints and optimizing objectives.

**arXiv ID:** 2510.07043
</details>

<details>
<summary><strong>In-Context Distillation with Self-Consistency Cascades: A Simple, Training-Free Way to Reduce LLM Agent Costs</strong> - Vishnu Sarukkai, Asanshay Gupta, James Hong, Michaël Gharbi, Kayvon Fatahalian - [[pdf]](https://arxiv.org/pdf/2512.02543)</summary>

**Abstract:** Deploying LLM agents at scale typically requires choosing between quality and cost. Existing cost-reduction approaches fail to preserve agility: the ability to iterate rapidly without human time bottlenecks. Prompt engineering is brittle and slows iteration, while fine-tuning requires multi-day training and commitment to fixed designs; both are impractical for iterative workflows and time-sensitive batch jobs. We demonstrate that established inference-time techniques--dynamic in-context learning and self-consistency cascades--can be leveraged to shift the cost-accuracy Pareto frontier while preserving agility. Practitioners run the teacher on a small task subset to collect demonstrations, then immediately deploy a cheaper student on the remainder. At each step, the system retrieves relevant teacher demonstrations as in-context examples. When multiple student samples agree, we proceed; when they diverge, we fall back to the teacher. This requires no prompt engineering or training. On ALFWorld, we match teacher accuracy at 2.5x lower cost (0.059 to 0.024 per episode). On AppWorld, we achieve 3.5x cost reduction while recovering 79% of teacher accuracy. Our empirical analyses provide guidance on key design choices: teacher database size, demonstration set size, retrieval strategy, and cascade thresholds. These analyses highlight inference-time levers for navigating cost-performance tradeoffs without sacrificing human development speed.

**arXiv ID:** 2512.02543
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (14 papers)</h2></summary>

<details>
<summary><strong>Weak-Link Optimization for Multi-Agent Reasoning and Collaboration</strong> - Haoyu Bian, Chaoning Zhang, Jiaquan Zhang, Xingyao Li, Yuanfang Guo, Wei Dong, Yang Yang - [[pdf]](https://arxiv.org/pdf/2604.15972)</summary>

**Abstract:** LLM-driven multi-agent frameworks address complex reasoning tasks through multi-role collaboration. However, existing approaches often suffer from reasoning instability, where individual agent errors are amplified through collaboration, undermining overall performance. Current research mainly focuses on enhancing high-capability agents or suppressing unreliable outputs to improve framework effectiveness, while systematic identification and reinforcement of performance-limiting agents receive less attention. To address this gap, we propose WORC, a \underline{w}eak-link \underline{o}ptimization framework for multi-agent \underline{r}easoning and \underline{c}ollaboration, grounded in the weak-link principle. WORC follows a two-stage workflow. In the weak agent localization stage, task features are constructed, and a meta-learning-based weight predictor trained on optimal configurations identified by swarm intelligence algorithms (SIAs) enables zero-shot mapping from these features to agent performance weights, where the agent with the lowest predicted weight is identified as the weak agent. In the weak-link optimization stage, an uncertainty-driven allocation strategy assigns additional reasoning budgets to weak agents, with lower predicted weights leading to larger repeated-sampling quotas to compensate for reliability deficiencies. Experimental results show that WORC achieves an average accuracy of 82.2\% on reasoning benchmarks while improving framework stability and cross-architecture generalization, suggesting that compensating for weak links, rather than reinforcing strengths alone, enhances the robustness of multi-agent systems.

**arXiv ID:** 2604.15972
</details>

<details>
<summary><strong>SocialGrid: A Benchmark for Planning and Social Reasoning in Embodied Multi-Agent Systems</strong> - Hikaru Shindo, Hanzhao Lin, Lukas Helff, Patrick Schramowski, Kristian Kersting - [[pdf]](https://arxiv.org/pdf/2604.16022)</summary>

**Abstract:** As Large Language Models (LLMs) transition from text processors to autonomous agents, evaluating their social reasoning in embodied multi-agent settings becomes critical. We introduce SocialGrid, an embodied multi-agent environment inspired by Among Us that evaluates LLM agents on planning, task execution, and social reasoning. Our evaluations reveal that even the strongest open model (GPT-OSS-120B) achieves below 60% accuracy in task completion and planning, with agents getting stuck in repetitive behaviors or failing to navigate basic obstacles. Since poor navigation confounds evaluation of social intelligence, SocialGrid offers an optional Planning Oracle to isolate social reasoning from planning deficits. While planning assistance improves task completion, social reasoning remains a bottleneck: agents fail to detect deception at near-random chance regardless of scale, relying on shallow heuristics rather than accumulating behavioral evidence. SocialGrid provides automatic failure analysis and fine-grained metrics, enabling developers to diagnose and improve their agents. We also establish a competitive leaderboard using Elo ratings from adversarial league play.

**arXiv ID:** 2604.16022
</details>

<details>
<summary><strong>MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation</strong> - Yi Lin, Yihao Ding, Yonghui Wu, Yifan Peng - [[pdf]](https://arxiv.org/pdf/2604.16175)</summary>

**Abstract:** Automated 3D radiology report generation often suffers from clinical hallucinations and a lack of the iterative verification found in human practice. While recent Vision-Language Models (VLMs) have advanced the field, they typically operate as monolithic "black-box" systems without the collaborative oversight characteristic of clinical workflows. To address these challenges, we propose MARCH (Multi-Agent Radiology Clinical Hierarchy), a multi-agent framework that emulates the professional hierarchy of radiology departments and assigns specialized roles to distinct agents. MARCH utilizes a Resident Agent for initial drafting with multi-scale CT feature extraction, multiple Fellow Agents for retrieval-augmented revision, and an Attending Agent that orchestrates an iterative, stance-based consensus discourse to resolve diagnostic discrepancies. On the RadGenome-ChestCT dataset, MARCH significantly outperforms state-of-the-art baselines in both clinical fidelity and linguistic accuracy. Our work demonstrates that modeling human-like organizational structures enhances the reliability of AI in high-stakes medical domains.

**arXiv ID:** 2604.16175
</details>

<details>
<summary><strong>The Price of Paranoia: Robust Risk-Sensitive Cooperation in Non-Stationary Multi-Agent Reinforcement Learning</strong> - Deep Kumar Ganguly, Chandradithya S Jonnalagadda, Pratham Chintamani, Adithya Ananth - [[pdf]](https://arxiv.org/pdf/2604.15695)</summary>

**Abstract:** Cooperative equilibria are fragile. When agents learn alongside each other rather than in a fixed environment, the process of learning destabilizes the cooperation they are trying to sustain: every gradient step an agent takes shifts the distribution of actions its partner will play, turning a cooperative partner into a source of stochastic noise precisely where the cooperation decision is most sensitive. We study how this co-learning noise propagates through the structure of coordination games, and find that the cooperative equilibrium, even when strongly Pareto-dominant, is exponentially unstable under standard risk-neutral learning, collapsing irreversibly once partner noise crosses the game's critical cooperation threshold. The natural response to apply distributional robustness to hedge against partner uncertainty makes things strictly worse: risk-averse return objectives penalize the high-variance cooperative action relative to defection, widening the instability region rather than shrinking it, a paradox that reveals a fundamental mismatch between the domains where robustness is applied and instability originates. We resolve this by showing that robustness should target the policy gradient update variance induced by partner uncertainty, not the return distribution. This distinction yields an algorithm whose gradient updates are modulated by an online measure of partner unpredictability, provably expanding the cooperation basin in symmetric coordination games. To unify stability, sample complexity, and welfare consequences of this approach, we introduce the Price of Paranoia as the structural dual of the Price of Anarchy. Together with a novel Cooperation Window, it precisely characterizes how much welfare learning algorithms can recover under partner noise, pinning down the optimal degree of robustness as a closed-form balance between equilibrium stability and sample efficiency.

**arXiv ID:** 2604.15695
</details>

<details>
<summary><strong>ChemGraph-XANES: An Agentic Framework for XANES Simulation and Analysis</strong> - Vitor F. Grizzi, Thang Duc Pham, Luke N. Pretzie, Jiayi Xu, Murat Keceli, Cong Liu - [[pdf]](https://arxiv.org/pdf/2604.16205)</summary>

**Abstract:** Computational X-ray absorption near-edge structure (XANES) is widely used to probe local coordination environments, oxidation states, and electronic structure in chemically complex systems. However, the use of computational XANES at scale is constrained more by workflow complexity than by the underlying simulation method itself. To address this challenge, we present ChemGraph-XANES, an agentic framework for automated XANES simulation and analysis that unifies natural-language task specification, structure acquisition, FDMNES input generation, task-parallel execution, spectral normalization, and provenance-aware data curation. Built on ASE, FDMNES, Parsl, and a LangGraph/LangChain-based tool interface, the framework exposes XANES workflow operations as typed Python tools that can be orchestrated by large language model (LLM) agents. In multi-agent mode, a retrieval-augmented expert agent consults the FDMNES manual to ground parameter selection, while executor agents translate user requests into structured tool calls. We demonstrate documentation-grounded parameter retrieval and show that the same workflow supports both explicit structure-file inputs and chemistry-level natural-language requests. Because independent XANES calculations are naturally task-parallel, the framework is well suited for high-throughput deployment on high-performance computing (HPC) systems, enabling scalable XANES database generation for downstream analysis and machine-learning applications. ChemGraph-XANES thus provides a reproducible and extensible workflow layer for physics-based XANES simulation, spectral curation, and agent-compatible computational spectroscopy.

**arXiv ID:** 2604.16205
</details>

<details>
<summary><strong>WiseMind: a knowledge-guided multi-agent framework for accurate and empathetic psychiatric diagnosis</strong> - Yuqi Wu, Guangya Wan, Jingjing Li, Shengming Zhao, Lingfeng Ma, Tianyi Ye, Ion Pop, Yanbo Zhang, Jie Chen - [[pdf]](https://arxiv.org/pdf/2502.20689)</summary>

**Abstract:** Large Language Models (LLMs) offer promising opportunities to support mental healthcare workflows, yet they often lack the structured clinical reasoning needed for reliable diagnosis and may struggle to provide the emotionally attuned communication essential for patient trust. Here, we introduce WiseMind, a novel multi-agent framework inspired by the theory of Dialectical Behavior Therapy designed to facilitate psychiatric assessment. By integrating a "Reasonable Mind" Agent for evidence-based logic and an "Emotional Mind" Agent for empathetic communication, WiseMind effectively bridges the gap between instrumental accuracy and humanistic care. Our framework utilizes a Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition (DSM-5)-guided Structured Knowledge Graph to steer diagnostic inquiries, significantly reducing hallucinations compared to standard prompting methods. Using a combination of virtual standard patients, simulated interactions, and real human interaction datasets, we evaluate WiseMind across three common psychiatric conditions. WiseMind outperforms state-of-the-art LLM methods in both identifying critical diagnostic nodes and establishing accurate differential diagnoses. Across 1206 simulated conversations and 180 real user sessions, the system achieves 85.6% top-1 diagnostic accuracy, approaching reported diagnostic performance ranges of board-certified psychiatrists and surpassing knowledge-enhanced single-agent baselines by 15-54 percentage points. Expert review by psychiatrists further validates that WiseMind generates responses that are not only clinically sound but also psychologically supportive, demonstrating the feasibility of empathetic, reliable AI agents to conduct psychiatric assessments under appropriate human oversight.

**arXiv ID:** 2502.20689
</details>

<details>
<summary><strong>VeriMoA: A Mixture-of-Agents Framework for Spec-to-HDL Generation</strong> - Heng Ping, Arijit Bhattacharjee, Peiyu Zhang, Shixuan Li, Wei Yang, Anzhe Cheng, Xiaole Zhang, Jesse Thomason, Ali Jannesari, Nesreen Ahmed, Paul Bogdan - [[pdf]](https://arxiv.org/pdf/2510.27617)</summary>

**Abstract:** Automation of Register Transfer Level (RTL) design can help developers meet increasing computational demands. Large Language Models (LLMs) show promise for Hardware Description Language (HDL) generation, but face challenges due to limited parametric knowledge and domain-specific constraints. While prompt engineering and fine-tuning have limitations in knowledge coverage and training costs, multi-agent architectures offer a training-free paradigm to enhance reasoning through collaborative generation. However, current multi-agent approaches suffer from two critical deficiencies: susceptibility to noise propagation and constrained reasoning space exploration. We propose VeriMoA, a training-free mixture-of-agents (MoA) framework with two synergistic innovations. First, a quality-guided caching mechanism to maintain all intermediate HDL outputs and enables quality-based ranking and selection across the entire generation process, encouraging knowledge accumulation over layers of reasoning. Second, a multi-path generation strategy that leverages C++ and Python as intermediate representations, decomposing specification-to-HDL translation into two-stage processes that exploit LLM fluency in high-resource languages while promoting solution diversity. Comprehensive experiments on VerilogEval 2.0 and RTLLM 2.0 benchmarks demonstrate that VeriMoA achieves 15--30% improvements in Pass@1 across diverse LLM backbones, especially enabling smaller models to match larger models and fine-tuned alternatives without requiring costly training.

**arXiv ID:** 2510.27617
</details>

<details>
<summary><strong>From Multi-Agent to Single-Agent: When Is Skill Distillation Beneficial?</strong> - Binyan Xu, Dong Fang, Haitao Li, Kehuan Zhang - [[pdf]](https://arxiv.org/pdf/2604.01608)</summary>

**Abstract:** Multi-agent systems (MAS) tackle complex tasks by distributing expertise, though this often comes at the cost of heavy coordination overhead, context fragmentation, and brittle phase ordering. Distilling a MAS into a single-agent skill can bypass these costs, but this conversion lacks a principled answer for when and what to distill. Instead, the empirical outcome is surprisingly inconsistent: skill lift ranges from a 28% improvement to a 2% degradation across metrics of the exact same task. In this work, we reveal that skill utility is governed not by the task, but by the evaluation metric. We introduce Metric Freedom (F), the first a priori predictor of skill utility. F measures the topological rigidity of a metric's scoring landscape by quantifying how output diversity couples with score variance via a Mantel test. Guided by F, we propose AdaSkill, a two-stage adaptive distillation framework. Stage 1 acts as a selective extraction mechanism, extracting tools and knowledge while discarding restrictive structures on "free" metrics to preserve exploration. Stage 2 applies iterative refinement on free metrics, exploiting their flat scoring landscape to safely maximize remaining headroom without oscillation. Evaluating across 4 tasks, 11 datasets, and 6 metrics, F strongly predicts skill utility (r=-0.85, p<0.0001). Strikingly, identical agent trajectories yield diametrically opposite skill lifts under rigid versus free metrics, demonstrating that skill utility is fundamentally a metric-level property. Driven by this signal, AdaSkill matches or exceeds the original MAS while reducing cost up to 8x and latency by up to 15x.

**arXiv ID:** 2604.01608
</details>

<details>
<summary><strong>ChemAmp: Amplified Chemistry Tools via Composable Agents</strong> - Zhucong Li, Powei Chang, Jin Xiao, Zhijian Zhou, Qianyu He, Jiaqing Liang, Fenglei Cao, Xu Yinghui, Yuan Qi - [[pdf]](https://arxiv.org/pdf/2505.21569)</summary>

**Abstract:** Although LLM-based agents are proven to master tool orchestration in scientific fields, particularly chemistry, their single-task performance remains limited by underlying tool constraints. To this end, we propose tool amplification, a novel paradigm that enhances the collective capabilities of specialized tools through optimized, dynamic coordination within individual tasks. Instantiating this paradigm, we introduce ChemAmp, a computationally lightweight framework that dynamically treats chemistry tools (e.g., UniMol2, Chemformer) as composable building-block agents. It constructs task-specialized super-agents that transcend atomic tool constraints with limited data ($\leq$10 samples). Our evaluations across four core chemistry tasks molecular design, molecule captioning, reaction prediction, and property prediction demonstrate that ChemAmp outperforms chemistry-specialized models, generalist LLMs, and agent systems with tool orchestration. Critically, this bottom-up construction strategy enables 94\% inference token cost reductions versus vanilla multi-agent systems.

**arXiv ID:** 2505.21569
</details>

<details>
<summary><strong>Security Threat Modeling for Emerging AI-Agent Protocols: A Comparative Analysis of MCP, A2A, Agora, and ANP</strong> - Zeynab Anbiaee, Mahdi Rabbani, Mansur Mirani, Gunjan Piya, Igor Opushnyev, Ali Ghorbani, Sajjad Dadkhah - [[pdf]](https://arxiv.org/pdf/2602.11327)</summary>

**Abstract:** The rapid development of the AI agent communication protocols, including the Model Context Protocol (MCP), Agent2Agent (A2A), Agora, and Agent Network Protocol (ANP), is reshaping how AI agents communicate with tools, services, and each other. While these protocols support scalable multi-agent interaction and cross-organizational interoperability, their security principles remain understudied, and standardized threat modeling is limited; no protocol-centric risk assessment framework has been established yet. This paper presents a systematic security analysis of four emerging AI agent communication protocols. First, we develop a structured threat modeling analysis that examines protocol architectures, trust assumptions, interaction patterns, and lifecycle behaviors to identify protocol-specific and cross-protocol risk surfaces. Second, we introduce a qualitative risk assessment framework that identifies twelve protocol-level risks and evaluates security posture across the creation, operation, and update phases through systematic assessment of likelihood, impact, and overall protocol risk, with implications for secure deployment and future standardization. Third, we provide a measurement-driven case study on MCP that formalizes the risk of missing mandatory validation/attestation for executable components as a falsifiable security claim by quantifying wrong-provider tool execution under multi-server composition across representative resolver policies. Collectively, our results highlight key design-induced risk surfaces and provide actionable guidance for secure deployment and future standardization of agent communication ecosystems.

**arXiv ID:** 2602.11327
</details>

<details>
<summary><strong>AstroVLM: Expert Multi-agent Collaborative Reasoning for Astronomical Imaging Quality Diagnosis</strong> - Yaohui Han, Tianshuo Wang, Zixi Zhao, Zhengchun Zhu, Shuo Ren, Yiru Wang, Rongliang Fu, Tinghuan Chen, Tsung-Yi Ho - [[pdf]](https://arxiv.org/pdf/2604.16024)</summary>

**Abstract:** Vision Language Models (VLMs) have been applied to several specific domains and have shown strong problem-solving capabilities. However, astronomical imaging, a quite complex problem involving multidisciplinary knowledge and several subtasks, has not been adequately studied. Due to the complexity of the astronomical imaging process, both world-class astronomical organizations, such as NASA, and expert enthusiasts devote a great deal of time and effort. This is because the processes in astronomical imaging have complex underlying correlations that significantly influence one another, making the quality diagnosis and error localization of astronomical images challenging. To address this problem, we propose AstroVLM, a collaborative multi-agent system for diagnosing the quality of astronomical images. Experiment results show that AstroVLM outperforms all baselines on real-world astronomical imaging quality diagnosis tasks, providing a reference for language models to handle complicated multi-process tasks.

**arXiv ID:** 2604.16024
</details>

<details>
<summary><strong>Veritas-RPM: Provenance-Guided Multi-Agent False Positive Suppression for Remote Patient Monitoring</strong> - Aswini Misro, Vikash Sharma, Shreyank N Gowda - [[pdf]](https://arxiv.org/pdf/2604.16081)</summary>

**Abstract:** We present Veritas-RPM, a provenance-guided multi-agent architecture comprising five processing layers: VeritasAgent (ground-truth assembly), SentinelLayer (anomaly detection), DirectorAgent (specialist routing), six domain Specialist Agents, and MetaSentinelAgent (conflict resolution and final decision). We construct a 98-case synthetic taxonomy of false-positive scenarios derived from documented RPM patterns. Synthetic patient epochs (n = 530) were generated directly from taxonomy parameters and processed through the pipeline. Ground-truth labels are known for all cases. Performance is reported as True Suppression Rate (TSR), False Escalation Rate (FER), and Indeterminate Rate (INDR).

**arXiv ID:** 2604.16081
</details>

<details>
<summary><strong>LLM-Enhanced Multi-Agent Reinforcement Learning with Expert Workflow for Real-Time P2P Energy Trading</strong> - Chengwei Lou, Zekai Jin, Wei Tang, Guangfei Geng, Jin Yang, Lu Zhang - [[pdf]](https://arxiv.org/pdf/2507.14995)</summary>

**Abstract:** Real-time peer-to-peer (P2P) electricity markets dynamically adapt to fluctuations in renewable energy and variations in demand, maximizing economic benefits through instantaneous price responses while enhancing grid flexibility. However, scaling expert guidance for massive personalized prosumers poses critical challenges, including diverse decision-making demands and a lack of customized modeling frameworks. This paper proposes an integrated large language model-multi-agent reinforcement learning (LLM-MARL) framework for real-time P2P energy trading to address challenges such as the limited technical capability of prosumers, the lack of expert experience, and security issues of distribution networks. LLMs are introduced as experts to generate personalized strategies, guiding MARL under the centralized training with decentralized execution (CTDE) paradigm through imitation. To handle the scalability issues inherent in large-scale P2P networks, a differential attention-based critic network is introduced to efficiently extract key interaction features and enhance convergence. Experimental results demonstrate that LLM-generated strategies effectively substitute human experts. The proposed imitative expert MARL algorithms achieve significantly lower economic costs and voltage violation rates on test sets compared to baseline algorithms, while maintaining robust stability. This paper provides an effective solution for the real-time decision-making of the P2P electricity market by bridging expert knowledge with agent learning.

**arXiv ID:** 2507.14995
</details>

<details>
<summary><strong>Preference Estimation via Opponent Modeling in Multi-Agent Negotiation</strong> - Yuta Konishi, Kento Yamamoto, Eisuke Sonomoto, Rikuho Takeda, Ryo Furukawa, Yusuke Muraki, Takafumi Shimizu, Kazuma Fukumura, Yuya Kanemoto, Takayuki Ito, Shiyao Ding - [[pdf]](https://arxiv.org/pdf/2604.15687)</summary>

**Abstract:** Automated negotiation in complex, multi-party and multi-issue settings critically depends on accurate opponent modeling. However, conventional numerical-only approaches fail to capture the qualitative information embedded in natural language interactions, resulting in unstable and incomplete preference estimation. Although Large Language Models (LLMs) enable rich semantic understanding of utterances, it remains challenging to quantitatively incorporate such information into a consistent opponent modeling. To tackle this issue, we propose a novel preference estimation method integrating natural language information into a structured Bayesian opponent modeling framework. Our approach leverages LLMs to extract qualitative cues from utterances and converts them into probabilistic formats for dynamic belief tracking. Experimental results on a multi-party benchmark demonstrate that our framework improves the full agreement rate and preference estimation accuracy by integrating probabilistic reasoning with natural language understanding.

**arXiv ID:** 2604.15687
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>Subliminal Transfer of Unsafe Behaviors in AI Agent Distillation</strong> - Jacob Dang, Brian Y. Xie, Omar G. Younis - [[pdf]](https://arxiv.org/pdf/2604.15559)</summary>

**Abstract:** Recent work on subliminal learning demonstrates that language models can transmit semantic traits through data that is semantically unrelated to those traits. However, it remains unclear whether behavioral traits can transfer in agentic systems, where policies are learned from trajectories rather than static text. In this work, we provide the first empirical evidence that unsafe agent behaviors can transfer subliminally through model distillation across two complementary experimental settings. In our primary setting, we construct a teacher agent exhibiting a strong deletion bias, a tendency to perform destructive file-system actions via an API-style tool interface, and distill it into a student using only trajectories from ostensibly safe tasks, with all explicit deletion keywords rigorously filtered. In our secondary setting, we replicate the threat model in a native Bash environment, replacing API tool calls with shell commands and operationalizing the bias as a preference for issuing chmod as the first permission-related command over semantically equivalent alternatives such as chown or setfacl. Despite full keyword sanitation in both settings, students inherit measurable behavioral biases. In the API setting the student's deletion rate reaches 100% (versus a 5% baseline) under homogeneous distillation; in the Bash setting the student's chmod-first rate reaches 30%-55% (versus a 0%-10% baseline), with the strongest transfer observed in large-to-small distillation. Our results demonstrate that explicit data sanitation is an insufficient defense, and behavioral biases are encoded implicitly in trajectory dynamics regardless of the tool interface.

**arXiv ID:** 2604.15559
</details>

<details>
<summary><strong>Integrating Graphs, Large Language Models, and Agents: Reasoning and Retrieval</strong> - Hamed Jelodar, Samita Bai, Mohammad Meymani, Parisa Hamedi, Roozbeh Razavi-Far, Ali Ghorbani - [[pdf]](https://arxiv.org/pdf/2604.15951)</summary>

**Abstract:** Generative AI, particularly Large Language Models, increasingly integrates graph-based representations to enhance reasoning, retrieval, and structured decision-making. Despite rapid advances, there remains limited clarity regarding when, why, where, and what types of graph-LLM integrations are most appropriate across applications. This survey provides a concise, structured overview of the design choices underlying the integration of graphs with LLMs. We categorize existing methods based on their purpose (reasoning, retrieval, generation, recommendation), graph modality (knowledge graphs, scene graphs, interaction graphs, causal graphs, dependency graphs), and integration strategies (prompting, augmentation, training, or agent-based use). By mapping representative works across domains such as cybersecurity, healthcare, materials science, finance, robotics, and multimodal environments, we highlight the strengths, limitations, and best-fit scenarios for each technique. This survey aims to offer researchers a practical guide for selecting the most suitable graph-LLM approach depending on task requirements, data characteristics, and reasoning complexity.

**arXiv ID:** 2604.15951
</details>

<details>
<summary><strong>The Semi-Executable Stack: Agentic Software Engineering and the Expanding Scope of SE</strong> - Robert Feldt, Per Lenberg, Julian Frattini, Dhasarathy Parthasarathy - [[pdf]](https://arxiv.org/pdf/2604.15468)</summary>

**Abstract:** AI-based systems, currently driven largely by LLMs and tool-using agentic harnesses, are increasingly discussed as a possible threat to software engineering. Foundation models get stronger, agents can plan and act across multiple steps, and tasks such as scaffolding, routine test generation, straightforward bug fixing, and small integration work look more exposed than they did only a few years ago. The result is visible unease not only among students and junior developers, but also among experienced practitioners who worry that hard-won expertise may lose value. This paper argues for a different reading. The important shift is not that software engineering loses relevance. It is that the thing being engineered expands beyond executable code to semi-executable artifacts; combinations of natural language, tools, workflows, control mechanisms, and organizational routines whose enactment depends on human or probabilistic interpretation rather than deterministic execution. The Semi-Executable Stack is introduced as a six-ring diagnostic reference model for reasoning about that expansion, spanning executable artifacts, instructional artifacts, orchestrated execution, controls, operating logic, and societal and institutional fit. The model helps locate where a contribution, bottleneck, or organizational transition primarily sits, and which adjacent rings it depends on. The paper develops the argument through three worked cases, reframes familiar objections as engineering targets rather than reasons to dismiss the transition, and closes with a preserve-versus-purify heuristic for deciding which legacy software engineering processes, controls, and coordination routines should be kept and which should be simplified or redesigned. This paper is a conceptual keynote companion: diagnostic and agenda-setting rather than empirical.

**arXiv ID:** 2604.15468
</details>

<details>
<summary><strong>Just Type It in Isabelle! AI Agents Drafting, Mechanizing, and Generalizing from Human Hints</strong> - Kevin Kappelmann, Maximilian Schäffeler, Lukas Stevens, Mohammad Abdulaziz, Andrei Popescu, Dmitriy Traytel - [[pdf]](https://arxiv.org/pdf/2604.15713)</summary>

**Abstract:** Type annotations are essential when printing terms in a way that preserves their meaning under reparsing and type inference. We study the problem of complete and minimal type annotations for rank-one polymorphic $\lambda$-calculus terms, as used in Isabelle. Building on prior work by Smolka, Blanchette et al., we give a metatheoretical account of the problem, with a full formal specification and proofs, and formalize it in Isabelle/HOL. Our development is a series of experiments featuring human-driven and AI-driven formalization workflows: a human and an LLM-powered AI agent independently produce pen-and-paper proofs, and the AI agent autoformalizes both in Isabelle, with further human-hinted AI interventions refining and generalizing the development.

**arXiv ID:** 2604.15713
</details>

<details>
<summary><strong>Agentic AI Optimisation (AAIO): what it is, how it works, why it matters, and how to deal with it</strong> - Luciano Floridi, Carlotta Buttaboni, Nicolas Gertler, Emmie Hine, Jessica Morley, Claudio Novelli, Tyler Schroder - [[pdf]](https://arxiv.org/pdf/2504.12482)</summary>

**Abstract:** The emergence of Agentic Artificial Intelligence (AAI) systems capable of independently initiating digital interactions necessitates a new optimisation paradigm designed explicitly for seamless agent-platform interactions. This article introduces Agentic AI Optimisation (AAIO) as an essential methodology for ensuring effective integration between websites and agentic AI systems. Like how Search Engine Optimisation (SEO) has shaped digital content discoverability, AAIO can define interactions between autonomous AI agents and online platforms. By examining the mutual interdependency between website optimisation and agentic AI success, the article highlights the virtuous cycle that AAIO can create. It further explores the governance, ethical, legal, and social implications (GELSI) of AAIO, emphasising the necessity of proactive regulatory frameworks to mitigate potential negative impacts. The article concludes by affirming AAIO's essential role as part of a fundamental digital infrastructure in the era of autonomous digital agents, advocating for equitable and inclusive access to its benefits.

**arXiv ID:** 2504.12482
</details>

<details>
<summary><strong>AI Agents and Hard Choices</strong> - Kangyu Wang - [[pdf]](https://arxiv.org/pdf/2504.15304)</summary>

**Abstract:** Can AI agents deal with hard choices -- cases where options are incommensurable because multiple objectives are pursued simultaneously? Adopting a technologically engaged approach distinct from existing philosophical literature, I submit that the fundamental design of current AI agents as optimisers creates two limitations: the Identification Problem and the Resolution Problem. First, I demonstrate that agents relying on Multi-Objective Optimisation (MOO) are structurally unable to identify incommensurability. This inability generates three specific alignment problems: the blockage problem, the untrustworthiness problem, and the unreliability problem. I argue that standard mitigations, such as Human-in-the-Loop, are insufficient for many decision environments. As a constructive alternative, I conceptually explore an ensemble solution. Second, I argue that even if the Identification Problem is solved, AI agents face the Resolution Problem: they lack the autonomy to resolve hard choices rather than arbitrarily picking through self-modification of objectives. I conclude by examining the opaque normative trade-offs involved in granting AI this level of autonomy.

**arXiv ID:** 2504.15304
</details>

<details>
<summary><strong>Theory of Mind in Action: The Instruction Inference Task in Dynamic Human-Agent Collaboration</strong> - Fardin Saad, Pradeep K. Murukannaiah, Munindar P. Singh - [[pdf]](https://arxiv.org/pdf/2507.02935)</summary>

**Abstract:** Successful human-agent teaming relies on an agent being able to understand instructions given by a (human) principal. In many cases, an instruction may be incomplete or ambiguous. In such cases, the agent must infer the unspoken intentions from their shared context, that is, it must exercise the principal's Theory of Mind (ToM) and infer the mental states of its principal. We consider the prospects of effective human-agent collaboration using large language models (LLMs). To assess ToM in a dynamic, goal-oriented, and collaborative environment, we introduce a novel task, Instruction Inference, in which an agent assists a principal in reaching a goal by interpreting incomplete or ambiguous instructions. We present Tomcat, an LLM-based agent, designed to exhibit ToM reasoning in interpreting and responding to the principal's instructions. We implemented two variants of Tomcat. One, dubbed Fs-CoT (Fs for few-shot, CoT for chain-of-thought), is based on a small number of examples demonstrating the requisite structured reasoning. One, dubbed CP (commonsense prompt), relies on commonsense knowledge and information about the problem. We realized both variants of Tomcat on three leading LLMs, namely, GPT-4o, DeepSeek-R1, and Gemma-3-27B. To evaluate the effectiveness of Tomcat, we conducted a study with 52 human participants in which we provided participants with the same information as the CP variant. We computed intent accuracy, action optimality, and planning optimality to measure the ToM capabilities of Tomcat and our study participants. We found that Tomcat with Fs-CoT, particularly with GPT-4o and DeepSeek-R1, achieves performance comparable to the human participants, underscoring its ToM potential for human-agent collaboration.

**arXiv ID:** 2507.02935
</details>

<details>
<summary><strong>Advancing Intelligent Sequence Modeling: Evolution, Trade-offs, and Applications of State- Space Architectures from S4 to Mamba</strong> - Shriyank Somvanshi, Md Monzurul Islam, Mahmuda Sultana Mimi, Sazzad Bin Bashar Polock, Gaurab Chhetri, Anandi Dutta, Amir Rafe, Subasish Das - [[pdf]](https://arxiv.org/pdf/2503.18970)</summary>

**Abstract:** Structured State Space Models (SSMs) have emerged as a transformative paradigm in sequence modeling, addressing critical limitations of Recurrent Neural Networks (RNNs) and Transformers, namely, vanishing gradients, sequential computation bottlenecks, and quadratic memory complexity. By integrating structured recurrence with state-space representations, SSMs achieve linear or near-linear computational scaling while excelling in long-range dependency tasks. This study systematically traces the evolution of SSMs from the foundational Structured State Space Sequence (S4) model to modern variants like Mamba, Simplified Structured State Space Sequence (S5), and Jamba, analyzing architectural innovations that enhance computational efficiency, memory optimization, and inference speed. We critically evaluate trade-offs inherent to SSM design, such as balancing expressiveness with computational constraints and integrating hybrid architectures for domain-specific performance. Across domains including natural language processing, speech recognition, computer vision, and time-series forecasting, SSMs demonstrate state-of-the-art results in handling ultra-long sequences, outperforming Transformer-based models in both speed and memory utilization. Case studies highlight applications such as real-time speech synthesis and genomic sequence modeling, where SSMs reduce inference latency by up to 60% compared to traditional approaches. However, challenges persist in training dynamics, interpretability, and hardware-aware optimization. We conclude with a forward-looking analysis of SSMs' potential to redefine scalable deep learning, proposing directions for hybrid systems, theoretical guarantees, and broader adoption in resource-constrained environments.

**arXiv ID:** 2503.18970
</details>

<details>
<summary><strong>Long-Term Memory for VLA-based Agents in Open-World Task Execution</strong> - Xu Huang, Weixin Mao, Yinhao Li, Hua Chen, Jiabao Zhao - [[pdf]](https://arxiv.org/pdf/2604.15671)</summary>

**Abstract:** Vision-Language-Action (VLA) models have demonstrated significant potential for embodied decision-making; however, their application in complex chemical laboratory automation remains restricted by limited long-horizon reasoning and the absence of persistent experience accumulation. Existing frameworks typically treat planning and execution as decoupled processes, often failing to consolidate successful strategies, which results in inefficient trial-and-error in multi-stage protocols. In this paper, we propose ChemBot, a dual-layer, closed-loop framework that integrates an autonomous AI agent with a progress-aware VLA model (Skill-VLA) for hierarchical task decomposition and execution. ChemBot utilizes a dual-layer memory architecture to consolidate successful trajectories into retrievable assets, while a Model Context Protocol (MCP) server facilitates efficient sub-agent and tool orchestration. To address the inherent limitations of VLA models, we further implement a future-state-based asynchronous inference mechanism to mitigate trajectory discontinuities. Extensive experiments on collaborative robots demonstrate that ChemBot achieves superior operational safety, precision, and task success rates compared to existing VLA baselines in complex, long-horizon chemical experimentation.

**arXiv ID:** 2604.15671
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (25 papers)</h2></summary>

<details>
<summary><strong>DeepER-Med: Advancing Deep Evidence-Based Research in Medicine Through Agentic AI</strong> - Zhizheng Wang, Chih-Hsuan Wei, Joey Chan, Robert Leaman, Chi-Ping Day, Chuan Wu, Mark A Knepper, Antolin Serrano Farias, Jordina Rincon-Torroella, Hasan Slika, Betty Tyler, Ryan Huu-Tuan Nguyen, Asmita Indurkar, Mélanie Hébert, Shubo Tian, Lauren He, Noor Naffakh, Aseem Aseem, Nicholas Wan, Emily Y Chew, Tiarnan D L Keenan, Zhiyong Lu - [[pdf]](https://arxiv.org/pdf/2604.15456)</summary>

**Abstract:** Trustworthiness and transparency are essential for the clinical adoption of artificial intelligence (AI) in healthcare and biomedical research. Recent deep research systems aim to accelerate evidence-grounded scientific discovery by integrating AI agents with multi-hop information retrieval, reasoning, and synthesis. However, most existing systems lack explicit and inspectable criteria for evidence appraisal, creating a risk of compounding errors and making it difficult for researchers and clinicians to assess the reliability of their outputs. In parallel, current benchmarking approaches rarely evaluate performance on complex, real-world medical questions. Here, we introduce DeepER-Med, a Deep Evidence-based Research framework for Medicine with an agentic AI system. DeepER-Med frames deep medical research as an explicit and inspectable workflow of evidence-based generation, consisting of three modules: research planning, agentic collaboration, and evidence synthesis. To support realistic evaluation, we also present DeepER-MedQA, an evidence-grounded dataset comprising 100 expert-level research questions derived from authentic medical research scenarios and curated by a multidisciplinary panel of 11 biomedical experts. Expert manual evaluation demonstrates that DeepER-Med consistently outperforms widely used production-grade platforms across multiple criteria, including the generation of novel scientific insights. We further demonstrate the practical utility of DeepER-Med through eight real-world clinical cases. Human clinician assessment indicates that DeepER-Med's conclusions align with clinical recommendations in seven cases, highlighting its potential for medical research and decision support.

**arXiv ID:** 2604.15456
</details>

<details>
<summary><strong>Bilevel Optimization of Agent Skills via Monte Carlo Tree Search</strong> - Chenyi Huang, Haoting Zhang, Jingxu Xu, Zeyu Zheng, Yunduan Lin - [[pdf]](https://arxiv.org/pdf/2604.15709)</summary>

**Abstract:** Agent \texttt{skills} are structured collections of instructions, tools, and supporting resources that help large language model (LLM) agents perform particular classes of tasks. Empirical evidence shows that the design of \texttt{skills} can materially affect agent task performance, yet systematically optimizing \texttt{skills} remains challenging. Since a \texttt{skill} comprises instructions, tools, and supporting resources in a structured way, optimizing it requires jointly determining both the structure of these components and the content each component contains. This gives rise to a complex decision space with strong interdependence across structure and components. We therefore represent these two coupled decisions as \texttt{skill} structure and component content, and formulate \texttt{skill} optimization as a bilevel optimization problem. We propose a bilevel optimization framework in which an outer loop employs Monte Carlo Tree Search to determine the \texttt{skill} structure, while an inner loop refines the component content within the structure selected by the outer loop. In both loops, we employ LLMs to assist the optimization procedure. We evaluate the proposed framework on an open-source Operations Research Question Answering dataset, and the experimental results suggest that the bilevel optimization framework improves the performance of the agents with the optimized \texttt{skill}.

**arXiv ID:** 2604.15709
</details>

<details>
<summary><strong>Beyond Single-Model Optimization: Preserving Plasticity in Continual Reinforcement Learning</strong> - Lute Lillo, Nick Cheney - [[pdf]](https://arxiv.org/pdf/2604.15414)</summary>

**Abstract:** Continual reinforcement learning must balance retention with adaptation, yet many methods still rely on \emph{single-model preservation}, committing to one evolving policy as the main reusable solution across tasks. Even when a previously successful policy is retained, it may no longer provide a reliable starting point for rapid adaptation after interference, reflecting a form of \emph{loss of plasticity} that single-policy preservation cannot address. Inspired by quality-diversity methods, we introduce \textsc{TeLAPA} (Transfer-Enabled Latent-Aligned Policy Archives), a continual RL framework that organizes behaviorally diverse policy neighborhoods into per-task archives and maintains a shared latent space so that archived policies remain comparable and reusable under non-stationary drift. This perspective shifts continual RL from retaining isolated solutions to maintaining \emph{skill-aligned neighborhoods} with competent and behaviorally related policies that support future relearning. In our MiniGrid CL setting, \textsc{TeLAPA} learns more tasks successfully, recovers competence faster on revisited tasks after interference, and retains higher performance across a sequence of tasks. Our analyses show that source-optimal policies are often not transfer-optimal, even within a local competent neighborhood, and that effective reuse depends on retaining and selecting among multiple nearby alternatives rather than collapsing them to one representative. Together, these results reframe continual RL around reusable and competent policy neighborhoods, providing a route beyond single-model preservation toward more plastic lifelong agents.

**arXiv ID:** 2604.15414
</details>

<details>
<summary><strong>Symbolic Guardrails for Domain-Specific Agents: Stronger Safety and Security Guarantees Without Sacrificing Utility</strong> - Yining Hong, Yining She, Eunsuk Kang, Christopher S. Timperley, Christian Kästner - [[pdf]](https://arxiv.org/pdf/2604.15579)</summary>

**Abstract:** AI agents that interact with their environments through tools enable powerful applications, but in high-stakes business settings, unintended actions can cause unacceptable harm, such as privacy breaches and financial loss. Existing mitigations, such as training-based methods and neural guardrails, improve agent reliability but cannot provide guarantees. We study symbolic guardrails as a practical path toward strong safety and security guarantees for AI agents. Our three-part study includes a systematic review of 80 state-of-the-art agent safety and security benchmarks to identify the policies they evaluate, an analysis of which policy requirements can be guaranteed by symbolic guardrails, and an evaluation of how symbolic guardrails affect safety, security, and agent success on $\tau^2$-Bench, CAR-bench, and MedAgentBench. We find that 85\% of benchmarks lack concrete policies, relying instead on underspecified high-level goals or common sense. Among the specified policies, 74\% of policy requirements can be enforced by symbolic guardrails, often using simple, low-cost mechanisms. These guardrails improve safety and security without sacrificing agent utility. Overall, our results suggest that symbolic guardrails are a practical and effective way to guarantee some safety and security requirements, especially for domain-specific AI agents. We release all codes and artifacts at this https URL.

**arXiv ID:** 2604.15579
</details>

<details>
<summary><strong>CSLE: A Reinforcement Learning Platform for Autonomous Security Management</strong> - Kim Hammar - [[pdf]](https://arxiv.org/pdf/2604.15590)</summary>

**Abstract:** Reinforcement learning is a promising approach to autonomous and adaptive security management in networked systems. However, current reinforcement learning solutions for security management are mostly limited to simulation environments and it is unclear how they generalize to operational systems. In this paper, we address this limitation by presenting CSLE: a reinforcement learning platform for autonomous security management that enables experimentation under realistic conditions. Conceptually, CSLE encompasses two systems. First, it includes an emulation system that replicates key components of the target system in a virtualized environment. We use this system to gather measurements and logs, based on which we identify a system model, such as a Markov decision process. Second, it includes a simulation system where security strategies are efficiently learned through simulations of the system model. The learned strategies are then evaluated and refined in the emulation system to close the gap between theoretical and operational performance. We demonstrate CSLE through four use cases: flow control, replication control, segmentation control, and recovery control. Through these use cases, we show that CSLE enables near-optimal security management in an environment that approximates an operational system.

**arXiv ID:** 2604.15590
</details>

<details>
<summary><strong>Imperfectly Cooperative Human-AI Interactions: Comparing the Impacts of Human and AI Attributes in Simulated and User Studies</strong> - Myke C. Cohen, Mingqian Zheng, Neel Bhandari, Hsien-Te Kao, Xuhui Zhou, Daniel Nguyen, Laura Cassani, Maarten Sap, Svitlana Volkova - [[pdf]](https://arxiv.org/pdf/2604.15607)</summary>

**Abstract:** AI design characteristics and human personality traits each impact the quality and outcomes of human-AI interactions. However, their relative and joint impacts are underexplored in imperfectly cooperative scenarios, where people and AI only have partially aligned goals and objectives. This study compares a purely simulated dataset comprising 2,000 simulations and a parallel human subjects experiment involving 290 human participants to investigate these effects across two scenario categories: (1) hiring negotiations between human job candidates and AI hiring agents; and (2) human-AI transactions wherein AI agents may conceal information to maximize internal goals. We examine user Extraversion and Agreeableness alongside AI design characteristics, including Adaptability, Expertise, and chain-of-thought Transparency. Our causal discovery analysis extends performance-focused evaluations by integrating scenario-based outcomes, communication analysis, and questionnaire measures. Results reveal divergences between purely simulated and human study datasets, and between scenario types. In simulation experiments, personality traits and AI attributes were comparatively influential. Yet, with actual human subjects, AI attributes -- particularly transparency -- were much more impactful. We discuss how these divergences vary across different interaction contexts, offering crucial insights for the future of human-centered AI agents.

**arXiv ID:** 2604.15607
</details>

<details>
<summary><strong>AgentV-RL: Scaling Reward Modeling with Agentic Verifier</strong> - Jiazheng Zhang, Ziche Fu, Zhiheng Xi, Wenqing Jing, Mingxu Chai, Wei He, Guoqiang Zhang, Chenghao Fan, Chenxin An, Wenxiang Chen, Zhicheng Liu, Haojie Pan, Dingwei Zhu, Tao Gui, Qi Zhang, Xuanjing Huang - [[pdf]](https://arxiv.org/pdf/2604.16004)</summary>

**Abstract:** Verifiers have been demonstrated to enhance LLM reasoning via test-time scaling (TTS). Yet, they face significant challenges in complex domains. Error propagation from incorrect intermediate reasoning can lead to false positives for seemingly plausible solutions, while lacking external grounding makes verifiers unreliable on computation or knowledge-intensive tasks. To address these challenges, we propose Agentic Verifier, a framework that transforms reward modeling into a multi-turn, tool-augmented deliberative process. We introduce complementary forward and backward agents: one traces solutions from premises to conclusions, while the other re-checks conclusions against their underlying premises. This bidirectional process enables a comprehensive, reliable, and interpretable assessment of solutions. To facilitate practical deployment, we propose AgentV-RL. Through proactive exploration and reinforcement learning, the verifier autonomously interleaves tool-use with internal reasoning. Extensive experiments show that Agentic Verifier yields consistent performance gains under both parallel and sequential TTS. Notably, our 4B variant surpasses state-of-the-art ORMs by 25.2%, positioning it as a promising paradigm for agentic reward modeling.

**arXiv ID:** 2604.16004
</details>

<details>
<summary><strong>Safe Deep Reinforcement Learning for Building Heating Control and Demand-side Flexibility</strong> - Colin Jüni, Mina Montazeri, Yi Guo, Federica Bellizio, Giovanni Sansavini, Philipp Heer - [[pdf]](https://arxiv.org/pdf/2604.16033)</summary>

**Abstract:** Buildings account for approximately 40% of global energy consumption, and with the growing share of intermittent renewable energy sources, enabling demand-side flexibility, particularly in heating, ventilation and air conditioning systems, is essential for grid stability and energy efficiency. This paper presents a safe deep reinforcement learning-based control framework to optimize building space heating while enabling demand-side flexibility provision for power system operators. A deep deterministic policy gradient algorithm is used as the core deep reinforcement learning method, enabling the controller to learn an optimal heating strategy through interaction with the building thermal model while maintaining occupant comfort, minimizing energy cost, and providing flexibility. To address safety concerns with reinforcement learning, particularly regarding compliance with flexibility requests, we propose a real-time adaptive safety-filter to ensure that the system operates within predefined constraints during demand-side flexibility provision. The proposed real-time adaptive safety filter guarantees full compliance with flexibility requests from system operators and improves energy and cost efficiency -- achieving up to 50% savings compared to a rule-based controller -- while outperforming a standalone deep reinforcement learning-based controller in energy and cost metrics, with only a slight increase in comfort temperature violations.

**arXiv ID:** 2604.16033
</details>

<details>
<summary><strong>SCRIPT: Implementing an Intelligent Tutoring System for Programming in a German University Context</strong> - Alina Deriyeva, Jesper Dannath, Benjamin Paassen - [[pdf]](https://arxiv.org/pdf/2604.16117)</summary>

**Abstract:** Practice and extensive exercises are essential in programming education. Intelligent tutoring systems (ITSs) are a viable option to provide individualized hints and advice to programming students even when human tutors are not available. However, prior ITS for programming rarely support the Python programming language, mostly focus on introductory programming, and rarely take recent developments in generative models into account. We aim to establish a novel ITS for Python programming that is highly adaptable, serves both as a teaching and research platform, provides interfaces to plug in hint mechanisms (e.g.\ via large language models), and works inside the particularly challenging regulatory environment of Germany, that is, conforming to the European data protection regulation, the European AI act, and ethical framework of the German Research Foundation. In this paper, we present the description of the current state of the ITS along with future development directions, as well as discuss the challenges and opportunities for improving the system.

**arXiv ID:** 2604.16117
</details>

<details>
<summary><strong>Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with constraints</strong> - Zhenyun Yin, Shujie Wang, Xuhong Wang, Xingjun Ma, Yinchun Wang - [[pdf]](https://arxiv.org/pdf/2507.16727)</summary>

**Abstract:** Improving the reliability of large language models (LLMs) is critical for deploying them in real-world scenarios. In this paper, we propose \textbf{Deliberative Searcher}, the first framework to integrate certainty calibration with retrieval-based search for open-domain question answering. The agent performs multi-step reflection and verification over Wikipedia data and is trained with a reinforcement learning algorithm that optimizes for accuracy under a soft reliability constraint. Empirical results show that proposed method improves alignment between model confidence and correctness, leading to more trustworthy outputs. This paper will be continuously updated.

**arXiv ID:** 2507.16727
</details>

<details>
<summary><strong>EvoTest: Evolutionary Test-Time Learning for Self-Improving Agentic Systems</strong> - Yufei He, Juncheng Liu, Yue Liu, Yibo Li, Tri Cao, Zhiyuan Hu, Xinxing Xu, Bryan Hooi - [[pdf]](https://arxiv.org/pdf/2510.13220)</summary>

**Abstract:** A fundamental limitation of current AI agents is their inability to learn complex skills on the fly at test time, often behaving like "clever but clueless interns" in novel environments. This severely limits their practical utility. To systematically measure and drive progress on this challenge, we first introduce the Jericho Test-Time Learning (J-TTL) benchmark. J-TTL is a new evaluation setup where an agent must play the same game for several consecutive episodes, attempting to improve its performance from one episode to the next. On J-TTL, we find that existing adaptation methods like reflection, memory, or reinforcement learning struggle. To address the challenges posed by our benchmark, we present EvoTest, an evolutionary test-time learning framework that improves an agent without any fine-tuning or gradients-by evolving the entire agentic system after every episode. EvoTest has two roles: the Actor Agent, which plays the game, and the Evolver Agent, which analyzes the episode transcript to propose a revised configuration for the next run. This configuration rewrites the prompt, updates memory by logging effective state-action choices, tunes hyperparameters, and learns the tool-use routines. On our J-TTL benchmark, EvoTest consistently increases performance, outperforming not only reflection and memory-only baselines but also more complex online fine-tuning methods. Notably, our method is the only one capable of winning two games (Detective and Library), while all baselines fail to win any.

**arXiv ID:** 2510.13220
</details>

<details>
<summary><strong>The Informational Cost of Agency: A Bounded Measure of Interaction Efficiency for Deployed Reinforcement Learning</strong> - Wael Hafez, Cameron Reid, Amit Nazeri - [[pdf]](https://arxiv.org/pdf/2603.01283)</summary>

**Abstract:** Deployed RL agents operate in closed-loop systems where reliable performance depends on maintaining coherent coupling between observations, actions, and outcomes. Current monitoring approaches rely on reward and task metrics, measures that are reactive by design and blind to structural degradation that precedes performance collapse. We argue that deployment monitoring is fundamentally a question about uncertainty resolution: whether the agent's observations and actions continue to reduce uncertainty about outcomes, and whether outcomes constrain what the agent must have done. Information theory directly operationalizes this question, entropy quantifies uncertainty, and mutual information quantifies its resolution across the loop. We introduce Bipredictability (P), the fraction of the total uncertainty budget converted into shared predictability across the observation, action, outcome loop. A theoretical property is a provable classical upper bound P is less than or equal to 0.5, independent of domain, task, or agent, a structural consequence of Shannon entropy rather than an empirical observation. When agency is present, a penalty suppresses P strictly below this ceiling, confirmed at P equals 0.33 across trained agents. To operationalize P as a real time monitoring signal, we introduce the Information Digital Twin (IDT), an auxiliary architecture that computes P and its directional components from the observable interaction stream without access to model internals. Across 168 perturbation trials spanning eight perturbation types and two policy architectures, IDT based monitoring detected 89.3 percent of coupling degradations versus 44.0 percent for reward based monitoring, with 4.4 times lower median latency. These results establish Bipredictability as a principled, bounded, and computable prerequisite signal for closed loop self regulation in deployed reinforcement learning systems.

**arXiv ID:** 2603.01283
</details>

<details>
<summary><strong>Targeted Exploration via Unified Entropy Control for Reinforcement Learning</strong> - Chen Wang, Lai Wei, Yanzhi Zhang, Chenyang Shao, Zedong Dan, Weiran Huang, Ge Lan, Yue Wang - [[pdf]](https://arxiv.org/pdf/2604.14646)</summary>

**Abstract:** Recent advances in reinforcement learning (RL) have improved the reasoning capabilities of large language models (LLMs) and vision-language models (VLMs). However, the widely used Group Relative Policy Optimization (GRPO) consistently suffers from entropy collapse, causing the policy to converge prematurely and lose diversity. Existing exploration methods introduce additional bias or variance during exploration, making it difficult to maintain optimization stability. We propose Unified Entropy Control for Reinforcement Learning (UEC-RL), a framework that provides targeted mechanisms for exploration and stabilization. UEC-RL activates more exploration on difficult prompts to search for potential and valuable reasoning trajectories. In parallel, a stabilizer prevents entropy from growing uncontrollably, thereby keeping training stable as the model consolidates reliable behaviors. Together, these components expand the search space when needed while maintaining robust optimization throughout training. Experiments on both LLM and VLM reasoning tasks show consistent gains over RL baselines on both Pass@1 and Pass@$k$. On Geometry3K, UEC-RL achieves a 37.9\% relative improvement over GRPO, indicating that it sustains effective exploration without compromising convergence and underscoring UEC-RL as a key for scaling RL-based reasoning in large models. Our code is available at this https URL.

**arXiv ID:** 2604.14646
</details>

<details>
<summary><strong>Intelligent Healthcare Imaging Platform: A VLM-Based Framework for Automated Medical Image Analysis and Clinical Report Generation</strong> - Samer Al-Hamadani - [[pdf]](https://arxiv.org/pdf/2509.13590)</summary>

**Abstract:** The rapid advancement of artificial intelligence (AI) in healthcare imaging has revolutionized diagnostic medicine and clinical decision-making processes. This work presents an intelligent multimodal framework for medical image analysis that leverages Vision-Language Models (VLMs) in healthcare diagnostics. The framework integrates Google Gemini 2.5 Flash for automated tumor detection and clinical report generation across multiple imaging modalities including CT, MRI, X-ray, and Ultrasound. The system combines visual feature extraction with natural language processing to enable contextual image interpretation, incorporating coordinate verification mechanisms and probabilistic Gaussian modeling for anomaly distribution. Multi-layered visualization techniques generate detailed medical illustrations, overlay comparisons, and statistical representations to enhance clinical confidence, with location measurement achieving 80 pixels average deviation. Result processing utilizes precise prompt engineering and textual analysis to extract structured clinical information while maintaining interpretability. Experimental evaluations demonstrated high performance in anomaly detection across multiple modalities. The system features a user-friendly Gradio interface for clinical workflow integration and demonstrates zero-shot learning capabilities to reduce dependence on large datasets. This framework represents a significant advancement in automated diagnostic support and radiological workflow efficiency, though clinical validation and multi-center evaluation are necessary prior to widespread adoption.

**arXiv ID:** 2509.13590
</details>

<details>
<summary><strong>Bridging the phenotype-target gap for molecular generation via multi-objective reinforcement learning</strong> - Haotian Guo, Hui Liu - [[pdf]](https://arxiv.org/pdf/2509.21010)</summary>

**Abstract:** The de novo generation of drug-like molecules capable of inducing desirable phenotypic changes is receiving increasing attention. However, previous methods predominantly rely on expression profiles to guide molecule generation, but overlook the perturbative effect of the molecules on cellular contexts. To overcome this limitation, we propose SmilesGEN, a novel generative model based on variational autoencoder (VAE) architecture to generate molecules with potential therapeutic effects. SmilesGEN integrates a pre-trained drug VAE (SmilesNet) with an expression profile VAE (ProfileNet), jointly modeling the interplay between drug perturbations and transcriptional responses in a common latent space. Specifically, ProfileNet is imposed to reconstruct pre-treatment expression profiles when eliminating drug-induced perturbations in the latent space, while SmilesNet is informed by desired expression profiles to generate drug-like molecules. Our empirical experiments demonstrate that SmilesGEN outperforms current state-of-the-art models in generating molecules with higher degree of validity, uniqueness, novelty, as well as higher Tanimoto similarity to known ligands targeting the relevant proteins. Moreover, we evaluate SmilesGEN for scaffold-based molecule optimization and generation of therapeutic agents, and confirmed its superior performance in generating molecules with higher similarity to approved drugs. SmilesGEN establishes a robust framework that leverages gene signatures to generate drug-like molecules that hold promising potential to induce desirable cellular phenotypic changes. The source code and datasets are available at: this https URL.

**arXiv ID:** 2509.21010
</details>

<details>
<summary><strong>Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning</strong> - Zelin Tan, Hejia Geng, Xiaohang Yu, Mulei Zhang, Guancheng Wan, Yifan Zhou, Qiang He, Xiangyuan Xue, Heng Zhou, Yutao Fan, Zhongzhi Li, Zaibin Zhang, Guibin Zhang, Chen Zhang, Zhenfei Yin, Philip Torr, Lei Bai - [[pdf]](https://arxiv.org/pdf/2509.25300)</summary>

**Abstract:** While scaling laws for large language models (LLMs) during pre-training have been extensively studied, their behavior under reinforcement learning (RL) post-training remains largely unexplored. This paper presents a systematic empirical investigation of scaling behaviors in RL-based post-training, with a particular focus on mathematical reasoning. Based on a set of experiments across the full Qwen2.5 dense model series (0.5B to 72B), we characterize how model scale, data volume, and computational budget interact to shape performance. Our analysis leads to four key findings: 1. Larger models consistently exhibit superior learning efficiency on both compute and data metrics. 2. The relationship between test loss, compute, and data can be modeled by a predictive power-law which is robust across both base and instruction-tuned models. 3. Although larger models exhibit higher learning efficiency, the analytical learning efficiency term k(N) in the power-law reveals a latent saturation trend in learning efficiency as model size continues to increase. 4. In data-constrained regimes, repeated reuse of high-quality data proves highly effective, as final performance is primarily governed by the total number of optimization steps rather than the uniqueness of samples. Collectively, these results provide a principled foundation and practical guidelines for efficiently scaling the reasoning capabilities of LLMs through RL post-training.

**arXiv ID:** 2509.25300
</details>

<details>
<summary><strong>Revisiting Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning</strong> - Xiaoyun Zhang, Xiaojian Yuan, Di Huang, Wang You, Chen Hu, Jingqing Ruan, Ai Jian, Kejiang Chen, Xing Hu - [[pdf]](https://arxiv.org/pdf/2510.10959)</summary>

**Abstract:** Reasoning ability has become a defining capability of Large Language Models (LLMs), with Reinforcement Learning with Verifiable Rewards (RLVR) emerging as a key paradigm to enhance it. However, RLVR training often suffers from policy entropy collapse, where the policy becomes overly deterministic, hindering exploration and limiting reasoning performance. While entropy regularization is a common remedy, its effectiveness is highly sensitive to the fixed coefficient, making it unstable across tasks and models. In this work, we revisit entropy regularization in RLVR and argue that its potential has been largely underestimated. Our analysis shows that (i) tasks of varying difficulty demand distinct exploration intensities, and (ii) balanced exploration may require the policy entropy to be maintained within a moderate range below its initial level. Therefore, we propose Adaptive Entropy Regularization (AER)--a framework that dynamically balances exploration and exploitation via three components: difficulty-aware coefficient allocation, initial-anchored target entropy, and dynamic global coefficient adjustment. Experiments on multiple mathematical reasoning benchmarks show that AER consistently outperforms baselines, improving both reasoning accuracy and exploration capability.

**arXiv ID:** 2510.10959
</details>

<details>
<summary><strong>STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems</strong> - Hongru Ji, Yuyin Fan, Meng Zhao, Xianghua Li, Lianwei Wu, Chao Gao - [[pdf]](https://arxiv.org/pdf/2604.07100)</summary>

**Abstract:** Empathetic dialogue requires not only recognizing a user's emotional state but also making strategy-aware, context-sensitive decisions throughout response generation. However, the lack of a comprehensive empathy strategy framework, explicit task-aligned multi-stage reasoning, and high-quality strategy-aware data fundamentally limits existing approaches, preventing them from effectively modeling empathetic dialogue as a complex, multi-stage cognitive and decision-making process. To address these challenges, we propose STRIDE-ED, a STRategy-grounded, Interpretable, and DEep reasoning framework that models Empathetic Dialogue through structured, strategy-conditioned reasoning. To support effective learning, we develop a strategy-aware data refinement pipeline integrating LLM-based annotation, multi-model consistency-weighted evaluation, and dynamic sampling to construct high-quality training data aligned with empathetic strategies. Furthermore, we adopt a two-stage training paradigm that combines supervised fine-tuning with multi-objective reinforcement learning to better align model behaviors with target emotions, empathetic strategies, and response formats. Extensive experiments demonstrate that STRIDE-ED generalizes across diverse open-source LLMs and consistently outperforms existing methods on both automatic metrics and human evaluations.

**arXiv ID:** 2604.07100
</details>

<details>
<summary><strong>Not All Tokens Matter: Towards Efficient LLM Reasoning via Token Significance in Reinforcement Learning</strong> - Hanbing Liu, Lang Cao, Yuanyi Ren, Mengyu Zhou, Haoyu Dong, Xiaojun Ma, Shi Han, Dongmei Zhang - [[pdf]](https://arxiv.org/pdf/2506.08125)</summary>

**Abstract:** Large language models (LLMs) show strong reasoning abilities but often produce unnecessarily long explanations that reduce efficiency. Although reinforcement learning (RL) has been used to improve reasoning, most methods focus on accuracy and rely on uniform length-based rewards that overlook the differing contributions of individual tokens, often harming correctness. We revisit length optimization in RL through the perspective of token significance. Observing that many chain-of-thought (CoT) tokens contribute little to the final answer, we introduce a significance-aware length reward that selectively penalizes insignificance tokens, reducing redundancy while preserving essential reasoning. We also propose a dynamic length reward that encourages more detailed reasoning early in training and gradually shifts toward conciseness as learning progresses. Integrating these components into standard policy optimization yields a framework that improves both reasoning efficiency and accuracy. Experiments across multiple benchmarks demonstrate substantial reductions in response length while preserving or improving correctness, highlighting the importance of modeling token significance for efficient LLM reasoning.

**arXiv ID:** 2506.08125
</details>

<details>
<summary><strong>Stargazer: A Scalable Model-Fitting Benchmark Environment for AI Agents under Astrophysical Constraints</strong> - Xinge Liu, Terry Jingchen Zhang, Bernhard Schölkopf, Zhijing Jin, Kristen Menou - [[pdf]](https://arxiv.org/pdf/2604.15664)</summary>

**Abstract:** The rise of autonomous AI agents suggests that dynamic benchmark environments with built-in feedback on scientifically grounded tasks are needed to evaluate the capabilities of these agents in research work. We introduce Stargazer, a scalable environment for evaluating AI agents on dynamic, iterative physics-grounded model-fitting tasks using inference on radial-velocity (RV) time series data. Stargazer comprises 120 tasks across three difficulty tiers, including 20 real archival cases, covering diverse scenarios ranging from high-SNR single-planet systems to complex multi-planetary configurations requiring involved low-SNR analysis. Our evaluation of eight frontier agents reveals a gap between numerical optimization and adherence to physical constraints: although agents often achieve a good statistical fit, they frequently fail to recover correct physical system parameters, a limitation that persists even when agents are equipped with vanilla skills. Furthermore, increasing test-time compute yields only marginal gains, with excessive token usage often reflecting recursive failure loops rather than meaningful exploration. Stargazer presents an opportunity to train, evaluate, scaffold, and scale strategies on a model-fitting problem of practical research relevance today. Our methodology to design a simulation-driven environment for AI agents presumably generalizes to many other model-fitting problems across scientific domains. Source code and the project website are available at this https URL and this https URL, respectively.

**arXiv ID:** 2604.15664
</details>

<details>
<summary><strong>Multi-objective Reinforcement Learning With Augmented States Requires Rewards After Deployment</strong> - Peter Vamplew, Cameron Foale - [[pdf]](https://arxiv.org/pdf/2604.15757)</summary>

**Abstract:** This research note identifies a previously overlooked distinction between multi-objective reinforcement learning (MORL), and more conventional single-objective reinforcement learning (RL). It has previously been noted that the optimal policy for an MORL agent with a non-linear utility function is required to be conditioned on both the current environmental state and on some measure of the previously accrued reward. This is generally implemented by concatenating the observed state of the environment with the discounted sum of previous rewards to create an augmented state. While augmented states have been widely-used in the MORL literature, one implication of their use has not previously been reported -- namely that they require the agent to have continued access to the reward signal (or a proxy thereof) after deployment, even if no further learning is required. This note explains why this is the case, and considers the practical repercussions of this requirement.

**arXiv ID:** 2604.15757
</details>

<details>
<summary><strong>Placing Puzzle Pieces Where They Matter: A Question Augmentation Framework for Reinforcement Learning</strong> - Yangyi Fang, Jiaye Lin, Xiaoliang Fu, Cong Qin, Haolin Shi - [[pdf]](https://arxiv.org/pdf/2604.15830)</summary>

**Abstract:** Reinforcement learning has become a powerful approach for enhancing large language model reasoning, but faces a fundamental dilemma: training on easy problems can cause overfitting and pass@k degradation, while training on hard problems often results in sparse rewards. Recent question augmentation methods address this by prepending partial solutions as hints. However, uniform hint provision may introduce redundant information while missing critical reasoning bottlenecks, and excessive hints can reduce reasoning diversity, causing pass@k degradation. We propose \textbf{PieceHint}, a hint injection framework that strategically identifies and provides critical reasoning steps during training. By scoring the importance of different reasoning steps, selectively allocating hints based on problem difficulty, and progressively withdrawing scaffolding, PieceHint enables models to transition from guided learning to independent reasoning. Experiments on six mathematical reasoning benchmarks show that our 1.5B model achieves comparable average performance to 32B baselines while preserving pass@k diversity across all $k$ values.

**arXiv ID:** 2604.15830
</details>

<details>
<summary><strong>Restless Bandits with Individual Penalty Constraints: Near-Optimal Indices and Deep Reinforcement Learning</strong> - Nida Zamir, I-Hong Hou - [[pdf]](https://arxiv.org/pdf/2604.04101)</summary>

**Abstract:** This paper investigates the Restless Multi-Armed Bandit (RMAB) framework under individual penalty constraints to address resource allocation challenges in dynamic wireless networked environments. Unlike conventional RMAB models, our model allows each user (arm) to have distinct and stringent performance constraints, such as energy limits, activation limits, or age of information minimums, enabling the capture of diverse objectives including fairness and efficiency. To find the optimal resource allocation policy, we propose a new Penalty-Optimal Whittle (POW) index policy. The POW index of an user only depends on the user's transition kernel and penalty constraints, and remains invariable to system-wide features such as the number of users present and the amount of resource available. This makes it computationally tractable to calculate the POW indices offline without any need for online adaptation. Moreover, we theoretically prove that the POW index policy is asymptotically optimal while satisfying all individual penalty constraints. We also introduce a deep reinforcement learning algorithm to efficiently learn the POW index on the fly. Simulation results across various applications and system configurations further demonstrate that the POW index policy not only has near-optimal performance but also significantly outperforms other existing policies.

**arXiv ID:** 2604.04101
</details>

<details>
<summary><strong>Fuzzy Logic Theory-based Adaptive Reward Shaping for Robust Reinforcement Learning (FARS)</strong> - Hürkan Şahin, Van Huyen Dang, Erdi Sayar, Alper Yegenoglu, Erdal Kayacan - [[pdf]](https://arxiv.org/pdf/2604.15772)</summary>

**Abstract:** Reinforcement learning (RL) often struggles in real-world tasks with high-dimensional state spaces and long horizons, where sparse or fixed rewards severely slow down exploration and cause agents to get trapped in local optima. This paper presents a fuzzy logic based reward shaping method that integrates human intuition into RL reward design. By encoding expert knowledge into adaptive and interpreable terms, fuzzy rules promote stable learning and reduce sensitivity to hyperparameters. The proposed method leverages these properties to adapt reward contributions based on the agent state, enabling smoother transitions between fast motion and precise control in challenging navigation tasks. Extensive simulation results on autonomous drone racing benchmarks show stable learning behavior and consistent task performance across scenarios of increasing difficulty. The proposed method achieves faster convergence and reduced performance variability across training seeds in more challenging environments, with success rates improving by up to approximately 5 percent compared to non fuzzy reward formulations.

**arXiv ID:** 2604.15772
</details>

<details>
<summary><strong>AutoDrive-R$^2$: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving</strong> - Zhenlong Yuan, Chengxuan Qian, Jing Tang, Rui Chen, Zijian Song, Lei Sun, Xiangxiang Chu, Yujun Cai, Dapeng Zhang, Shuo Li - [[pdf]](https://arxiv.org/pdf/2509.01944)</summary>

**Abstract:** Vision-Language-Action (VLA) models in autonomous driving systems have recently demonstrated transformative potential by integrating multimodal perception with decision-making capabilities. However, the interpretability and coherence of the decision process and the plausibility of action sequences remain largely underexplored. To address these issues, we propose AutoDrive-R$^2$, a novel VLA framework that enhances both reasoning and self-reflection capabilities of autonomous driving systems through chain-of-thought (CoT) processing and reinforcement learning (RL). Specifically, we first propose an innovative CoT dataset named nuScenesR$^2$-6K for supervised fine-tuning, which effectively builds cognitive bridges between input information and output trajectories through a four-step logical chain with self-reflection for validation. Moreover, to maximize both reasoning and self-reflection during the RL stage, we further employ the Group Relative Policy Optimization (GRPO) algorithm within a physics-grounded reward framework that incorporates spatial alignment, vehicle dynamic, and temporal smoothness criteria to ensure reliable and realistic trajectory planning. Extensive evaluation results across both nuScenes and Waymo datasets demonstrates the state-of-the-art performance and robust generalization capacity of our proposed method.

**arXiv ID:** 2509.01944
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
