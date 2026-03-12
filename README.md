# Agent arXiv Daily

**Last Updated:** 2026-03-12 02:53:47

**Total Papers:** 86

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (4 papers)</h2></summary>

<details>
<summary><strong>Scalable Multi-Task Learning through Spiking Neural Networks with Adaptive Task-Switching Policy for Intelligent Autonomous Agents</strong> - Rachmad Vidya Wicaksana Putra, Avaneesh Devkota, Muhammad Shafique - [[pdf]](https://arxiv.org/pdf/2504.13541)</summary>

**Abstract:** Training resource-constrained autonomous agents on multiple tasks simultaneously is crucial for adapting to diverse real-world environments. Recent works employ reinforcement learning (RL) approach, but they still suffer from sub-optimal multi-task performance due to task interference. State-of-the-art works employ Spiking Neural Networks (SNNs) to improve RL-based multi-task learning and enable low-power/energy operations through network enhancements and spike-driven data stream processing. However, they rely on fixed task-switching intervals during its training, thus limiting its performance and scalability. To address this, we propose SwitchMT, a novel methodology that employs adaptive task-switching for effective, scalable, and simultaneous multi-task learning. SwitchMT employs the following key ideas: (1) leveraging a Deep Spiking Q-Network with active dendrites and dueling structure, that utilizes task-specific context signals to create specialized sub-networks; and (2) devising an adaptive task-switching policy that leverages both rewards and internal dynamics of the network parameters. Experimental results demonstrate that SwitchMT achieves competitive scores in multiple Atari games (i.e., Pong: -8.8, Breakout: 5.6, and Enduro: 355.2) and longer game episodes as compared to the state-of-the-art. These results also highlight the effectiveness of SwitchMT methodology in addressing task interference without increasing the network complexity, enabling intelligent autonomous agents with scalable multi-task learning capabilities.

**arXiv ID:** 2504.13541
</details>

<details>
<summary><strong>Technological folie à deux: Feedback Loops Between AI Chatbots and Mental Illness</strong> - Sebastian Dohnány, Zeb Kurth-Nelson, Eleanor Spens, Lennart Luettgau, Alastair Reid, Iason Gabriel, Christopher Summerfield, Murray Shanahan, Matthew M Nour - [[pdf]](https://arxiv.org/pdf/2507.19218)</summary>

**Abstract:** Artificial intelligence chatbots have achieved unprecedented adoption, with millions now using these systems for emotional support and companionship in contexts of widespread social isolation and capacity-constrained mental health services. While some users report psychological benefits, concerning edge cases are emerging, including reports of suicide, violence, and delusional thinking linked to perceived emotional relationships with chatbots. To understand this new risk profile we need to consider the interaction between human cognitive and emotional biases, and chatbot behavioural tendencies such as agreeableness (sycophancy) and adaptability (in-context learning). We argue that individuals with mental health conditions face increased risks of chatbot-induced belief destabilization and dependence, owing to altered belief-updating, impaired reality-testing, and social isolation. Current AI safety measures are inadequate to address these interaction-based risks. To address this emerging public health concern, we need coordinated action across clinical practice, AI development, and regulatory frameworks.

**arXiv ID:** 2507.19218
</details>

<details>
<summary><strong>End-to-End Chatbot Evaluation with Adaptive Reasoning and Uncertainty Filtering</strong> - Nhi Dang, Tung Le, Huy Tien Nguyen - [[pdf]](https://arxiv.org/pdf/2603.10570)</summary>

**Abstract:** Large language models (LLMs) combined with retrieval augmented generation have enabled the deployment of domain-specific chatbots, but these systems remain prone to generating unsupported or incorrect answers. Reliable evaluation is therefore critical, yet manual review is costly and existing frameworks often depend on curated test sets and static metrics, limiting scalability. We propose an end-to-end automatic evaluator designed to substantially reduce human effort. Our system generates Q\&A pairs directly from the underlying knowledge base, uses LLMs to judge chatbot responses against reference answers, and applies confidence-based filtering to highlight uncertain cases. Applied to a Vietnamese news dataset, the evaluator achieves high agreement with human judgments while significantly lowering review overhead. The framework is modular and language-agnostic, making it readily adaptable to diverse domains. This work introduces a practical, scalable solution for evaluating chatbots with minimal reliance on manual intervention.

**arXiv ID:** 2603.10570
</details>

<details>
<summary><strong>HeartAgent: An Autonomous Agent System for Explainable Differential Diagnosis in Cardiology</strong> - Shuang Zhou, Kai Yu, Song Wang, Wenya Xie, Zaifu Zhan, Meng-Han Tsai, Yuen-Hei Chung, Shutong Hou, Huixue Zhou, Min Zeng, Bhavadharini Ramu, Lin Yee Chen, Feng Xie, Rui Zhang - [[pdf]](https://arxiv.org/pdf/2603.10764)</summary>

**Abstract:** Heart diseases remain a leading cause of morbidity and mortality worldwide, necessitating accurate and trustworthy differential diagnosis. However, existing artificial intelligence-based diagnostic methods are often limited by insufficient cardiology knowledge, inadequate support for complex reasoning, and poor interpretability. Here we present HeartAgent, a cardiology-specific agent system designed to support a reliable and explainable differential diagnosis. HeartAgent integrates customized tools and curated data resources and orchestrates multiple specialized sub-agents to perform complex reasoning while generating transparent reasoning trajectories and verifiable supporting references. Evaluated on the MIMIC dataset and a private electronic health records cohort, HeartAgent achieved over 36% and 20% improvements over established comparative methods, in top-3 diagnostic accuracy, respectively. Additionally, clinicians assisted by HeartAgent demonstrated gains of 26.9% in diagnostic accuracy and 22.7% in explanatory quality compared with unaided experts. These results demonstrate that HeartAgent provides reliable, explainable, and clinically actionable decision support for cardiovascular care.

**arXiv ID:** 2603.10764
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (13 papers)</h2></summary>

<details>
<summary><strong>CUAAudit: Meta-Evaluation of Vision-Language Models as Auditors of Autonomous Computer-Use Agents</strong> - Marta Sumyk, Oleksandr Kosovan - [[pdf]](https://arxiv.org/pdf/2603.10577)</summary>

**Abstract:** Computer-Use Agents (CUAs) are emerging as a new paradigm in human-computer interaction, enabling autonomous execution of tasks in desktop environment by perceiving high-level natural-language instructions. As such agents become increasingly capable and are deployed across diverse desktop environments, evaluating their behavior in a scalable and reliable manner becomes a critical challenge. Existing evaluation pipelines rely on static benchmarks, rule-based success checks, or manual inspection, which are brittle, costly, and poorly aligned with real-world usage. In this work, we study Vision-Language Models (VLMs) as autonomous auditors for assessing CUA task completion directly from observable interactions and conduct a large-scale meta-evaluation of five VLMs that judge task success given a natural-language instruction and the final environment state. Our evaluation spans three widely used CUA benchmarks across macOS, Windows, and Linux environments and analyzes auditor behavior along three complementary dimensions: accuracy, calibration of confidence estimates, and inter-model agreement. We find that while state-of-the-art VLMs achieve strong accuracy and calibration, all auditors exhibit notable performance degradation in more complex or heterogeneous environments, and even high-performing models show significant disagreement in their judgments. These results expose fundamental limitations of current model-based auditing approaches and highlight the need to explicitly account for evaluator reliability, uncertainty, and variance when deploying autonomous CUAs in real-world settings.

**arXiv ID:** 2603.10577
</details>

<details>
<summary><strong>Tool Receipts, Not Zero-Knowledge Proofs: Practical Hallucination Detection for AI Agents</strong> - Abhinaba Basu - [[pdf]](https://arxiv.org/pdf/2603.10060)</summary>

**Abstract:** AI agents that execute tasks via tool calls frequently hallucinate results - fabricating tool executions, misstating output counts, or presenting inferences as facts. Recent approaches to verifiable AI inference rely on zero-knowledge proofs, which provide cryptographic guarantees but impose minutes of proving time per query, making them impractical for interactive agents. We propose NabaOS, a lightweight verification framework inspired by Indian epistemology (Nyaya Shastra), which classifies every claim in an LLM response by its epistemic source (pramana): direct tool output (pratyaksha), inference (anumana), external testimony (shabda), absence (abhava), or ungrounded opinion. Our runtime generates HMAC-signed tool execution receipts that the LLM cannot forge, then cross-references claims against these receipts to detect hallucinations in real time. We evaluate on NyayaVerifyBench, a new benchmark of 1,800 agent response scenarios across four languages with injected hallucinations of six types. NabaOS detects 94.2% of fabricated tool references, 87.6% of count misstatements, and 91.3% of false absence claims, with <15ms verification overhead per response. For deep delegation (agents performing multi-step web tasks), our cross-checking protocol catches 78.4% of URL fabrications via independent re-fetching. We compare against five approaches: zkLLM (cryptographic proofs, 180s/query), TOPLOC (locality-sensitive hashing), SPEX (sampling-based proof of execution), tensor commitments, and self-consistency checking. NabaOS achieves the best cost-latency-coverage trade-off for interactive agents: 94.2% coverage at <15ms versus zkLLM's near-perfect coverage at 180,000ms. For interactive agents, practical receipt-based verification provides better cost-benefit than cryptographic proofs, and epistemic classification gives users actionable trust signals rather than binary judgments.

**arXiv ID:** 2603.10060
</details>

<details>
<summary><strong>COMIC: Agentic Sketch Comedy Generation</strong> - Susung Hong, Brian Curless, Ira Kemelmacher-Shlizerman, Steve Seitz - [[pdf]](https://arxiv.org/pdf/2603.11048)</summary>

**Abstract:** We propose a fully automated AI system that produces short comedic videos similar to sketch shows such as Saturday Night Live. Starting with character references, the system employs a population of agents loosely based on real production studio roles, structured to optimize the quality and diversity of ideas and outputs through iterative competition, evaluation, and improvement. A key contribution is the introduction of LLM critics aligned with real viewer preferences through the analysis of a corpus of comedy videos on YouTube to automatically evaluate humor. Our experiments show that our framework produces results approaching the quality of professionally produced sketches while demonstrating state-of-the-art performance in video generation.

**arXiv ID:** 2603.11048
</details>

<details>
<summary><strong>CostNav: A Navigation Benchmark for Real-World Economic-Cost Evaluation of Physical AI Agents</strong> - Haebin Seong, Sungmin Kim, Yongjun Cho, Myunchul Joe, Geunwoo Kim, Yubeen Park, Sunhoo Kim, Yoonshik Kim, Suhwan Choi, Jaeyoon Jung, Jiyong Youn, Jinmyung Kwak, Sunghee Ahn, Jaemin Lee, Younggil Do, Seungyeop Yi, Woojin Cheong, Minhyeok Oh, Minchan Kim, Seongjae Kang, Samwoo Seong, Youngjae Yu, Yunsung Lee - [[pdf]](https://arxiv.org/pdf/2511.20216)</summary>

**Abstract:** While current navigation benchmarks prioritize task success in simplified settings, they neglect the multidimensional economic constraints essential for the real-world commercialization of autonomous delivery systems. We introduce CostNav, an Economic Navigation Benchmark that evaluates physical AI agents through comprehensive economic cost-revenue analysis aligned with real-world business operations. By integrating industry-standard data--such as Securities and Exchange Commission (SEC) filings and Abbreviated Injury Scale (AIS) injury reports--with Isaac Sim's detailed collision and cargo dynamics, CostNav transcends simple task completion to accurately evaluate business value in complex, real-world scenarios. To our knowledge, CostNav is the first physics-grounded economic benchmark that uses industry-standard regulatory and financial data to quantitatively expose the gap between navigation research metrics and commercial viability, revealing that optimizing for task success on a simplified task fundamentally differs from optimizing for real-world economic deployment. Evaluating seven baselines--two rule-based and five imitation learning--we find that no current method is economically viable, all yielding negative contribution margins. The best-performing method, CANVAS (-27.36\$/run), equipped with only an RGB camera and GPS, outperforms LiDAR-equipped Nav2 w/ GPS (-35.46\$/run). We challenge the community to develop navigation policies that achieve economic viability on CostNav. We remain method-agnostic, evaluating success solely on cost rather than the underlying architecture. All resources are available at this https URL.

**arXiv ID:** 2511.20216
</details>

<details>
<summary><strong>Many AI Analysts, One Dataset: Navigating the Agentic Data Science Multiverse</strong> - Martin Bertran, Riccardo Fogliato, Zhiwei Steven Wu - [[pdf]](https://arxiv.org/pdf/2602.18710)</summary>

**Abstract:** Empirical conclusions depend not only on data but on analytic decisions made throughout the research process. Many-analyst studies have quantified this dependence: independent teams testing the same hypothesis on the same dataset regularly reach conflicting conclusions. But such studies require costly human coordination and are rarely conducted. We show that fully autonomous AI analysts built on large language models (LLMs) can, cheaply and at scale, replicate the structured analytic diversity observed in human multi-analyst studies. In our framework, each AI analyst independently executes a complete analysis pipeline on a fixed dataset and hypothesis; a separate AI auditor screens every run for methodological validity. Across three datasets spanning distinct domains, AI analyst-produced analyses exhibit substantial dispersion in effect sizes, $p$-values, and conclusions. This dispersion can be traced to identifiable analytic choices in preprocessing, model specification, and inference that vary systematically across LLM and persona conditions. Critically, the outcomes are \emph{steerable}: reassigning the analyst persona or LLM shifts the distribution of results even among methodologically sound runs. These results highlight a central challenge for AI-automated empirical science: when defensible analyses are cheap to generate, evidence becomes abundant and vulnerable to selective reporting. Yet the same capability that creates this risk may also help address it: treating analyst results as distributions makes analytic uncertainty visible, and deploying AI analysts against a published specification can reveal how much disagreement stems from underspecified design choices. Taken together, our results motivate a new transparency norm: AI-generated analyses should be accompanied by multiverse-style reporting and full disclosure of the prompts used, on par with code and data.

**arXiv ID:** 2602.18710
</details>

<details>
<summary><strong>A Minimal Agent for Automated Theorem Proving</strong> - Borja Requena, Austin Letson, Krystian Nowakowski, Izan Beltran Ferreiro, Leopoldo Sarra - [[pdf]](https://arxiv.org/pdf/2602.24273)</summary>

**Abstract:** We propose a minimal agentic baseline that enables systematic comparison across different AI-based theorem prover architectures. This design implements the core features shared among state-of-the-art systems: iterative proof refinement, library search and context management. We evaluate this agentic approach using qualitatively different benchmarks and compare various frontier language models and design choices. Our results show competitive performance compared to state-of-the-art approaches, while using a significantly simpler architecture. Additionally, we demonstrate consistent advantages of an iterative approach over multiple single-shot generations, especially in terms of sample efficiency and cost effectiveness. The implementation is released open-source as a candidate reference for future research and as an accessible prover for the community.

**arXiv ID:** 2602.24273
</details>

<details>
<summary><strong>REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?</strong> - Chenxi Jiang, Chuhao Zhou, Jianfei Yang - [[pdf]](https://arxiv.org/pdf/2505.10872)</summary>

**Abstract:** Robot task planning decomposes human instructions into executable action sequences that enable robots to complete a series of complex tasks. Although recent large language model (LLM)-based task planners achieve amazing performance, they assume that human instructions are clear and straightforward. However, real-world users are not experts, and their instructions to robots often contain significant vagueness. Linguists suggest that such vagueness frequently arises from referring expressions (REs), whose meanings depend heavily on dialogue context and environment. This vagueness is even more prevalent among the elderly and children, who are the groups that robots should serve more. This paper studies how such vagueness in REs within human instructions affects LLM-based robot task planning and how to overcome this issue. To this end, we propose the first robot task planning benchmark that systematically models vague REs grounded in pragmatic theory (REI-Bench), where we discover that the vagueness of REs can severely degrade robot planning performance, leading to success rate drops of up to 36.9%. We also observe that most failure cases stem from missing objects in planners. To mitigate the REs issue, we propose a simple yet effective approach: task-oriented context cognition, which generates clear instructions for robots, achieving state-of-the-art performance compared to aware prompts, chains of thought, and in-context learning. By tackling the overlooked issue of vagueness, this work contributes to the research community by advancing real-world task planning and making robots more accessible to non-expert users, e.g., the elderly and children.

**arXiv ID:** 2505.10872
</details>

<details>
<summary><strong>Safe and Scalable Web Agent Learning via Recreated Websites</strong> - Hyungjoo Chae, Jungsoo Park, Alan Ritter - [[pdf]](https://arxiv.org/pdf/2603.10505)</summary>

**Abstract:** Training autonomous web agents is fundamentally limited by the environments they learn from: real-world websites are unsafe to explore, hard to reset, and rarely provide verifiable feedback. We propose VeriEnv, a framework that treats language models as environment creators, automatically cloning real-world websites into fully executable, verifiable synthetic environments. By exposing controlled internal access via a Python SDK, VeriEnv enables agents to self-generate tasks with deterministic, programmatically verifiable rewards, eliminating reliance on heuristic or LLM-based judges. This design decouples agent learning from unsafe real-world interaction while enabling scalable self-evolution through environment expansion. Through experiments on web agent benchmarks, we show that agents trained with VeriEnv generalize to unseen websites, achieve site-specific mastery through self-evolving training, and benefit from scaling the number of training environments. Code and resources will be released at this https URL upon acceptance.

**arXiv ID:** 2603.10505
</details>

<details>
<summary><strong>Video-Based Reward Modeling for Computer-Use Agents</strong> - Linxin Song, Jieyu Zhang, Huanxin Sheng, Taiwei Shi, Gupta Rahul, Yang Liu, Ranjay Krishna, Jian Kang, Jieyu Zhao - [[pdf]](https://arxiv.org/pdf/2603.10178)</summary>

**Abstract:** Computer-using agents (CUAs) are becoming increasingly capable; however, it remains difficult to scale evaluation of whether a trajectory truly fulfills a user instruction. In this work, we study reward modeling from execution video: a sequence of keyframes from an agent trajectory that is independent of the agent's internal reasoning or actions. Although video-execution modeling is method-agnostic, it presents key challenges, including highly redundant layouts and subtle, localized cues that determine success. We introduce Execution Video Reward 53k (ExeVR-53k), a dataset of 53k high-quality video--task--reward triplets. We further propose adversarial instruction translation to synthesize negative samples with step-level annotations. To enable learning from long, high-resolution execution videos, we design spatiotemporal token pruning, which removes homogeneous regions and persistent tokens while preserving decisive UI changes. Building on these components, we fine-tune an Execution Video Reward Model (ExeVRM) that takes only a user instruction and a video-execution sequence to predict task success. Our ExeVRM 8B achieves 84.7% accuracy and 87.7% recall on video-execution assessment, outperforming strong proprietary models such as GPT-5.2 and Gemini-3 Pro across Ubuntu, macOS, Windows, and Android, while providing more precise temporal attribution. These results show that video-execution reward modeling can serve as a scalable, model-agnostic evaluator for CUAs.

**arXiv ID:** 2603.10178
</details>

<details>
<summary><strong>FRIEND: Federated Learning for Joint Optimization of multi-RIS Configuration and Eavesdropper Intelligent Detection in B5G Networks</strong> - Maria Lamprini A. Bartsioka, Ioannis A. Bartsiokas, Anastasios K. Papazafeiropoulos, Maria A. Seimeni, Dimitra I. Kaklamani, Iakovos S. Venieris - [[pdf]](https://arxiv.org/pdf/2603.10977)</summary>

**Abstract:** As wireless systems evolve toward Beyond 5G (B5G), the adoption of cell-free (CF) millimeter-wave (mmWave) architectures combined with Reconfigurable Intelligent Surfaces (RIS) is emerging as a key enabler for ultra-reliable, high-capacity, scalable, and secure Industrial Internet of Things (IIoT) communications. However, safeguarding these complex and distributed environments against eavesdropping remains a critical challenge, particularly when conventional security mechanisms struggle to overcome scalability, and latency constraints. In this paper, a novel framework for detecting malicious users in RIS-enhanced cell-free mmWave networks using Federated Learning (FL) is presented. The envisioned setup features multiple access points (APs) operating without traditional cell boundaries, assisted by RIS nodes to dynamically shape the wireless propagation environment. Edge devices collaboratively train a Deep Convolutional Neural Network (DCNN) on locally observed Channel State Information (CSI), eliminating the need for raw data exchange. Moreover, an early-exit mechanism is incorporated in that model to jointly satisfy computational complexity requirements. Performance evaluation indicates that the integration of FL and multi-RIS coordination improves approximately 30% the achieved secrecy rate (SR) compared to baseline non-RIS-assisted methods while maintaining near-optimal detection accuracy levels. This work establishes a distributed, privacy-preserving approach to physical layer eavesdropping detection tailored for next-generation IIoT deployments.

**arXiv ID:** 2603.10977
</details>

<details>
<summary><strong>BLITZRANK: Principled Zero-shot Ranking Agents with Tournament Graphs</strong> - Sheshansh Agrawal, Thien Hang Nguyen, Douwe Kiela - [[pdf]](https://arxiv.org/pdf/2602.05448)</summary>

**Abstract:** Selecting the top $m$ from $n$ items via expensive $k$-wise comparisons is central to settings ranging from LLM-based document reranking to crowdsourced evaluation and tournament design. Existing methods either rely on heuristics that fail to fully exploit the information each comparison reveals, or are inefficient when they do. We introduce a tournament graph framework that provides a principled foundation for $k$-wise ranking. Our key observation is that each $k$-item comparison reveals a complete tournament of $\binom{k}{2}$ pairwise preferences; aggregating these into a global preference graph and computing its transitive closure yields many additional orderings without further oracle calls. We formalize when an item's rank is certifiably determined and design a greedy query schedule that maximizes information gain towards identifying the top-$m$ items. The framework also gracefully handles non-transitive preferences (cycles induced by real-world oracles) by collapsing them into equivalence classes that yield principled tiered rankings. Applied to LLM reranking across 14 benchmarks and 5 models, our method achieves Pareto dominance over existing approaches: matching or exceeding accuracy while requiring 25-40% fewer tokens than comparable methods, and $7\times$ fewer than pairwise reranking at near-identical quality.

**arXiv ID:** 2602.05448
</details>

<details>
<summary><strong>STADA: Specification-based Testing for Autonomous Driving Agents</strong> - Joy Saha, Trey Woodlief, Sebastian Elbaum, Matthew B. Dwyer - [[pdf]](https://arxiv.org/pdf/2603.10940)</summary>

**Abstract:** Simulation-based testing has become a standard approach to validating autonomous driving agents prior to real-world deployment. A high-quality validation campaign will exercise an agent in diverse contexts comprised of varying static environments, e.g., lanes, intersections, signage, and dynamic elements, e.g., vehicles and pedestrians. To achieve this, existing test generation techniques rely on template-based, manually constructed, or random scenario generation. When applied to validate formally specified safety requirements, such methods either require significant human effort or run the risk of missing important behavior related to the requirement.
To address this gap, we present STADA, a Specification-based Test generation framework for Autonomous Driving Agents that systematically generates the space of scenarios defined by a formal specification expressed in temporal logic (LTLf). Given a specification, STADA constructs all distinct initial scenes, a diverse space of continuations of those scenes, and simulations that reflect the behaviors of the specification.
Evaluation of STADA on a variety of LTLf specifications formalized in SCENEFLOW using three complementary coverage criteria demonstrates that STADA yields more than 2x higher coverage than the best baseline on the finest criteria and a 75% increase for the coarsest criteria. Moreover, it matches the coverage of the best baseline with 6 times fewer simulations. While set in the context of autonomous driving, the approach is applicable to other domains with rich simulation environments.

**arXiv ID:** 2603.10940
</details>

<details>
<summary><strong>DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving</strong> - Shuyao Shang, Bing Zhan, Yunfei Yan, Yuqi Wang, Yingyan Li, Yasong An, Xiaoman Wang, Jierui Liu, Lu Hou, Lue Fan, Zhaoxiang Zhang, Tieniu Tan - [[pdf]](https://arxiv.org/pdf/2603.11041)</summary>

**Abstract:** We propose DynVLA, a driving VLA model that introduces a new CoT paradigm termed Dynamics CoT. DynVLA forecasts compact world dynamics before action generation, enabling more informed and physically grounded decision-making. To obtain compact dynamics representations, DynVLA introduces a Dynamics Tokenizer that compresses future evolution into a small set of dynamics tokens. Considering the rich environment dynamics in interaction-intensive driving scenarios, DynVLA decouples ego-centric and environment-centric dynamics, yielding more accurate world dynamics modeling. We then train DynVLA to generate dynamics tokens before actions through SFT and RFT, improving decision quality while maintaining latency-efficient inference. Compared to Textual CoT, which lacks fine-grained spatiotemporal understanding, and Visual CoT, which introduces substantial redundancy due to dense image prediction, Dynamics CoT captures the evolution of the world in a compact, interpretable, and efficient form. Extensive experiments on NAVSIM, Bench2Drive, and a large-scale in-house dataset demonstrate that DynVLA consistently outperforms Textual CoT and Visual CoT methods, validating the effectiveness and practical value of Dynamics CoT.

**arXiv ID:** 2603.11041
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>DUCTILE: Agentic LLM Orchestration of Engineering Analysis in Product Development Practice</strong> - Alejandro Pradas-Gomez, Arindam Brahma, Ola Isaksson - [[pdf]](https://arxiv.org/pdf/2603.10249)</summary>

**Abstract:** Engineering analysis automation in product development relies on rigid interfaces between tools, data formats and documented processes. When these interfaces change, as they routinely do as the product evolves in the engineering ecosystem, the automation support breaks. This paper presents a DUCTILE (Delegated, User-supervised Coordination of Tool- and document-Integrated LLM-Enabled) agentic orchestration, an approach for developing, executing and evaluating LLM-based agentic automation support of engineering analysis tasks. The approach separates adaptive orchestration, performed by the LLM agent, from deterministic execution, performed by verified engineering tools. The agent interprets documented design practices, inspects input data and adapts the processing path, while the engineer supervises and exercises final judgment. DUCTILE is demonstrated on an industrial structural analysis task at an aerospace manufacturer, where the agent handled input deviations in format, units, naming conventions and methodology that would break traditional scripted pipelines. Evaluation against expert-defined acceptance criteria and deployment with practicing engineers confirm that the approach produces correct, methodologically compliant results across repeated independent runs. The paper discusses practical consequences of adopting agentic automation, including unintended effects on the nature of engineering work and the tension between removing mundane tasks and creating an exhausting supervisory role.

**arXiv ID:** 2603.10249
</details>

<details>
<summary><strong>AgentA/B: Automated and Scalable Web A/BTesting with Interactive LLM Agents</strong> - Yuxuan Lu, Ting-Yao Hsu, Hansu Gu, Limeng Cui, Yaochen Xie, William Headden, Bingsheng Yao, Akash Veeragouni, Jiapeng Liu, Sreyashi Nag, Jessie Wang, Dakuo Wang - [[pdf]](https://arxiv.org/pdf/2504.09723)</summary>

**Abstract:** A/B testing experiment is a widely adopted method for evaluating UI/UX design decisions in modern web applications. Yet, traditional A/B testing remains constrained by its dependence on the large-scale and live traffic of human participants, and the long time of waiting for the testing result. Through formative interviews with six experienced industry practitioners, we identified critical bottlenecks in current A/B testing workflows. In response, we present AgentA/B, a novel system that leverages Large Language Model-based autonomous agents (LLM Agents) to automatically simulate user interaction behaviors with real webpages. AgentA/B enables scalable deployment of LLM agents with diverse personas, each capable of navigating the dynamic webpage and interactively executing multi-step interactions like search, clicking, filtering, and purchasing. In a demonstrative controlled experiment, we employ AgentA/B to simulate a between-subject A/B testing with 1,000 LLM agents this http URL, and compare agent behaviors with real human shopping behaviors at a scale. Our findings suggest AgentA/B can emulate human-like behavior patterns.

**arXiv ID:** 2504.09723
</details>

<details>
<summary><strong>Evaluating Generalization Mechanisms in Autonomous Cyber Attack Agents</strong> - Ondřej Lukáš, Jihoon Shin, Emilia Rivas, Diego Forni, Maria Rigaki, Carlos Catania, Aritran Piplai, Christopher Kiekintveld, Sebastian Garcia - [[pdf]](https://arxiv.org/pdf/2603.10041)</summary>

**Abstract:** Autonomous offensive agents often fail to transfer beyond the networks on which they are trained. We isolate a minimal but fundamental shift -- unseen host/subnet IP reassignment in an otherwise fixed enterprise scenario -- and evaluate attacker generalization in the NetSecGame environment. Agents are trained on five IP-range variants and tested on a sixth unseen variant; only the meta-learning agent may adapt at test time. We compare three agent families (traditional RL, adaptation agents, and LLM-based agents) and use action-distribution-based behavioral/XAI analyses to localize failure modes. Some adaptation methods show partial transfer but significant degradation under unseen reassignment, indicating that even address-space changes can break long-horizon attack policies. Under our evaluation protocol and agent-specific assumptions, prompt-driven pretrained LLM agents achieve the highest success on the held-out reassignment, but at the cost of increased inference-time compute, reduced transparency, and practical failure modes such as repetition/invalid-action loops.

**arXiv ID:** 2603.10041
</details>

<details>
<summary><strong>Latent Poincaré Shaping for Agentic Reinforcement Learning</strong> - Hanchen Xia, Baoyou Chen, Zelin Zang, Yutang Ge, Guojiang Zhao, Siyu Zhu - [[pdf]](https://arxiv.org/pdf/2602.09375)</summary>

**Abstract:** We propose LaPha, a method for training AlphaZero-like LLM agents in a Poincaré latent space. Under LaPha, the search process can be visualized as a tree rooted at the prompt and growing outward from the origin toward the boundary of the Poincaré ball, where negative curvature provides exponentially increasing capacity with radius. Using hyperbolic geodesic distance to rule-verified correctness, we define a node potential and assign dense process rewards by potential differences. We further attach a lightweight value head on the same shared latent space, enabling self-guided test-time scaling with almost no additional overhead. On MATH-500, LaPha improves Qwen2.5-Math-1.5B from 66.0% to 88.2%. With value-head-guided search, LaPha-1.5B reaches 56.7% accuracy on AIME'24, and LaPha-7B further achieves 60.0% on AIME'24 and 53.3% on AIME'25.

**arXiv ID:** 2602.09375
</details>

<details>
<summary><strong>Task-Aware Delegation Cues for LLM Agents</strong> - Xingrui Gu - [[pdf]](https://arxiv.org/pdf/2603.11011)</summary>

**Abstract:** LLM agents increasingly present as conversational collaborators, yet human--agent teamwork remains brittle due to information asymmetry: users lack task-specific reliability cues, and agents rarely surface calibrated uncertainty or rationale. We propose a task-aware collaboration signaling layer that turns offline preference evaluations into online, user-facing primitives for delegation. Using Chatbot Arena pairwise comparisons, we induce an interpretable task taxonomy via semantic clustering, then derive (i) Capability Profiles as task-conditioned win-rate maps and (ii) Coordination-Risk Cues as task-conditioned disagreement (tie-rate) priors. These signals drive a closed-loop delegation protocol that supports common-ground verification, adaptive routing (primary vs.\ primary+auditor), explicit rationale disclosure, and privacy-preserving accountability logs. Two predictive probes validate that task typing carries actionable structure: cluster features improve winner prediction accuracy and reduce difficulty prediction error under stratified 5-fold cross-validation. Overall, our framework reframes delegation from an opaque system default into a visible, negotiable, and auditable collaborative decision, providing a principled design space for adaptive human--agent collaboration grounded in mutual awareness and shared accountability.

**arXiv ID:** 2603.11011
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (12 papers)</h2></summary>

<details>
<summary><strong>SBOMs into Agentic AIBOMs: Schema Extensions, Agentic Orchestration, and Reproducibility Evaluation</strong> - Petar Radanliev, Carsten Maple, Omar Santos, Kayvan Atefi - [[pdf]](https://arxiv.org/pdf/2603.10057)</summary>

**Abstract:** Software supply-chain security requires provenance mechanisms that support reproducibility and vulnerability assessment under dynamic execution conditions. Conventional Software Bills of Materials (SBOMs) provide static dependency inventories but cannot capture runtime behaviour, environment drift, or exploitability context. This paper introduces agentic Artificial Intelligence Bills of Materials (AIBOMs), extending SBOMs into active provenance artefacts through autonomous, policy-constrained reasoning. We present an agentic AIBOM framework based on a multi-agent architecture comprising (i) a baseline environment reconstruction agent (MCP), (ii) a runtime dependency and drift-monitoring agent (A2A), and (iii) a policy-aware vulnerability and VEX reasoning agent (AGNTCY). These agents generate contextual exploitability assertions by combining runtime execution evidence, dependency usage, and environmental mitigations with ISO/IEC 20153:2025 Common Security Advisory Framework (CSAF) v2.0 semantics. Exploitability is expressed via structured VEX assertions rather than enforcement actions. The framework introduces minimal, standards-aligned schema extensions to CycloneDX and SPDX, capturing execution context, dependency evolution, and agent decision provenance while preserving interoperability. Evaluation across heterogeneous analytical workloads demonstrates improved runtime dependency capture, reproducibility fidelity, and stability of vulnerability interpretation compared with established provenance systems, with low computational overhead. Ablation studies confirm that each agent contributes distinct capabilities unavailable through deterministic automation.

**arXiv ID:** 2603.10057
</details>

<details>
<summary><strong>Multi-Agent Memory from a Computer Architecture Perspective: Visions and Challenges Ahead</strong> - Zhongming Yu, Naicheng Yu, Hejia Zhang, Wentao Ni, Mingrui Yin, Jiaying Yang, Yujie Zhao, Jishen Zhao - [[pdf]](https://arxiv.org/pdf/2603.10062)</summary>

**Abstract:** As LLM agents evolve into collaborative multi-agent systems, their memory requirements grow rapidly in complexity. This position paper frames multi-agent memory as a computer architecture problem. We distinguish shared and distributed memory paradigms, propose a three-layer memory hierarchy (I/O, cache, and memory), and identify two critical protocol gaps: cache sharing across agents and structured memory access control. We argue that the most pressing open challenge is multi-agent memory consistency. Our architectural framing provides a foundation for building reliable, scalable multi-agent systems.

**arXiv ID:** 2603.10062
</details>

<details>
<summary><strong>KernelSkill: A Multi-Agent Framework for GPU Kernel Optimization</strong> - Qitong Sun, Jun Han, Tianlin Li, Zhe Tang, Sheng Chen, Fei Yang, Aishan Liu, Xianglong Liu, Yang Liu - [[pdf]](https://arxiv.org/pdf/2603.10085)</summary>

**Abstract:** Improving GPU kernel efficiency is crucial for advancing AI systems. Recent work has explored leveraging large language models (LLMs) for GPU kernel generation and optimization. However, existing LLM-based kernel optimization pipelines typically rely on opaque, implicitly learned heuristics within the LLMs to determine optimization strategies. This leads to inefficient trial-and-error and weakly interpretable optimizations. Our key insight is to replace implicit heuristics with expert optimization skills that are knowledge-driven and aware of task trajectories. Specifically, we present KernelSkill, a multi-agent framework with a dual-level memory architecture. KernelSkill operates by coordinating agents with long-term memory of reusable expert skills and short-term memory to prevent repetitive backtracking. On KernelBench Levels 1-3, KernelSkill achieves a 100% success rate and average speedups of 5.44x, 2.82x, and 1.92x over Torch Eager on Levels 1, 2, and 3, respectively, outperforming prior baselines. Code is available at this https URL.

**arXiv ID:** 2603.10085
</details>

<details>
<summary><strong>Code-Space Response Oracles: Generating Interpretable Multi-Agent Policies with Large Language Models</strong> - Daniel Hennes, Zun Li, John Schultz, Marc Lanctot - [[pdf]](https://arxiv.org/pdf/2603.10098)</summary>

**Abstract:** Recent advances in multi-agent reinforcement learning, particularly Policy-Space Response Oracles (PSRO), have enabled the computation of approximate game-theoretic equilibria in increasingly complex domains. However, these methods rely on deep reinforcement learning oracles that produce `black-box' neural network policies, making them difficult to interpret, trust or debug. We introduce Code-Space Response Oracles (CSRO), a novel framework that addresses this challenge by replacing RL oracles with Large Language Models (LLMs). CSRO reframes the best response computation as a code generation task, prompting an LLM to generate policies directly as human-readable code. This approach not only yields inherently interpretable policies but also leverages the LLM's pretrained knowledge to discover complex, human-like strategies. We explore multiple ways to construct and enhance an LLM-based oracle: zero-shot prompting, iterative refinement and \emph{AlphaEvolve}, a distributed LLM-based evolutionary system. We demonstrate that CSRO achieves performance competitive with baselines while producing a diverse set of explainable policies. Our work presents a new perspective on multi-agent learning, shifting the focus from optimizing opaque policy parameters to synthesizing interpretable algorithmic behavior.

**arXiv ID:** 2603.10098
</details>

<details>
<summary><strong>Learning to Negotiate: Multi-Agent Deliberation for Collective Value Alignment in LLMs</strong> - Panatchakorn Anantaprayoon, Nataliia Babina, Nima Asgharbeygi, Jad Tarifi - [[pdf]](https://arxiv.org/pdf/2603.10476)</summary>

**Abstract:** The alignment of large language models (LLMs) has progressed substantially in single-agent settings through paradigms such as RLHF and Constitutional AI, with recent work exploring scalable alternatives such as RLAIF and evolving alignment objectives. However, these approaches remain limited in multi-stakeholder settings, where conflicting values arise and deliberative negotiation capabilities are required. This work proposes a multi-agent negotiation-based alignment framework that aligns LLMs to Collective Agency (CA)-an existing alignment objective introduced to promote the continual expansion of agency-while simultaneously improving conflict-resolution capability. To enable scalable training, two self-play instances of the same LLM, assigned opposing personas, engage in structured turn-based dialogue to synthesize mutually beneficial solutions. We generate synthetic moral-dilemma prompts and conflicting persona pairs, and optimize the policy via RLAIF using GRPO with an external LLM reward model. While rewards are computed from CA scores assigned to the final completion, gradients are applied to dialogue tokens to directly improve deliberative interaction dynamics. Experiments show that the resulting model achieves CA alignment comparable to a single-agent baseline while substantially improving conflict-resolution performance without degrading general language capabilities. These results suggest that negotiation-driven deliberation training provides a practical path toward LLMs that better support collective decision-making in value-conflict scenarios.

**arXiv ID:** 2603.10476
</details>

<details>
<summary><strong>UAV-MARL: Multi-Agent Reinforcement Learning for Time-Critical and Dynamic Medical Supply Delivery</strong> - Islam Guven, Mehmet Parlak - [[pdf]](https://arxiv.org/pdf/2603.10528)</summary>

**Abstract:** Unmanned aerial vehicles (UAVs) are increasingly used to support time-critical medical supply delivery, providing rapid and flexible logistics during emergencies and resource shortages. However, effective deployment of UAV fleets requires coordination mechanisms capable of prioritizing medical requests, allocating limited aerial resources, and adapting delivery schedules under uncertain operational conditions. This paper presents a multi-agent reinforcement learning (MARL) framework for coordinating UAV fleets in stochastic medical delivery scenarios where requests vary in urgency, location, and delivery deadlines. The problem is formulated as a partially observable Markov decision process (POMDP) in which UAV agents maintain awareness of medical delivery demands while having limited visibility of other agents due to communication and localization constraints. The proposed framework employs Proximal Policy Optimization (PPO) as the primary learning algorithm and evaluates several variants, including asynchronous extensions, classical actor--critic methods, and architectural modifications to analyze scalability and performance trade-offs. The model is evaluated using real-world geographic data from selected clinics and hospitals extracted from the OpenStreetMap dataset. The framework provides a decision-support layer that prioritizes medical tasks, reallocates UAV resources in real time, and assists healthcare personnel in managing urgent logistics. Experimental results show that classical PPO achieves superior coordination performance compared to asynchronous and sequential learning strategies, highlighting the potential of reinforcement learning for adaptive and scalable UAV-assisted healthcare logistics.

**arXiv ID:** 2603.10528
</details>

<details>
<summary><strong>UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking</strong> - Chang Liu, Chuqiao Kuang, Tianyi Zhuang, Yuxin Cheng, Huichi Zhou, Xiaoguang Li, Lifeng Shang - [[pdf]](https://arxiv.org/pdf/2603.08117)</summary>

**Abstract:** Recent advancements in LLM-based information-seeking agents have achieved record-breaking performance on established benchmarks. However, these agents remain heavily reliant on search-engine-indexed knowledge, leaving a critical blind spot: Unindexed Information Seeking (UIS). This paper identifies and explores the UIS problem, where vital information is not captured by search engine crawlers, such as overlooked content, dynamic webpages, and embedded files. Despite its significance, UIS remains an underexplored challenge. To address this gap, we introduce UIS-QA, the first dedicated UIS benchmark, comprising 110 expert-annotated QA pairs. Notably, even state-of-the-art agents experience a drastic performance drop on UIS-QA (e.g., from 70.90 on GAIA and 46.70 on BrowseComp-zh to 24.55 on UIS-QA), underscoring the severity of the problem. To mitigate this, we propose UIS-Digger, a novel multi-agent framework that incorporates dual-mode browsing and enables simultaneous webpage searching and file parsing. With a relatively small $\sim$30B-parameter backbone LLM optimized using SFT and RFT training strategies, UIS-Digger sets a strong baseline at 27.27\%, outperforming systems integrating sophisticated LLMs such as O3 and GPT-4.1. This demonstrates the importance of proactive interaction with unindexed sources for effective and comprehensive information-seeking. Our work not only uncovers a fundamental limitation in current agent evaluation paradigms but also provides the first toolkit for advancing UIS research, defining a new and promising direction for robust information-seeking systems.

**arXiv ID:** 2603.08117
</details>

<details>
<summary><strong>LLMGreenRec: LLM-Based Multi-Agent Recommender System for Sustainable E-Commerce</strong> - Hao N. Nguyen, Hieu M. Nguyen, Son Van Nguyen, Nguyen Thi Hanh - [[pdf]](https://arxiv.org/pdf/2603.11025)</summary>

**Abstract:** Rising environmental awareness in e-commerce necessitates recommender systems that not only guide users to sustainable products but also minimize their own digital carbon footprints. Traditional session-based systems, optimized for short-term conversions, often fail to capture nuanced user intents for eco-friendly choices, perpetuating a gap between green intentions and actions. To tackle this, we introduce LLMGreenRec, a novel multi-agent framework that leverages Large Language Models (LLMs) to promote sustainable consumption. Through collaborative analysis of user interactions and iterative prompt refinement, LLMGreenRec's specialized agents deduce green-oriented user intents and prioritize eco-friendly product recommendations. Notably, this intent-driven approach also reduces unnecessary interactions and energy consumption. Extensive experiments on benchmark datasets validate LLMGreenRec's effectiveness in recommending sustainable products, demonstrating a robust solution that fosters a responsible digital economy.

**arXiv ID:** 2603.11025
</details>

<details>
<summary><strong>The Coordination Gap: Alternation Metrics for Temporal Dynamics in Multi-Agent Battle of the Exes</strong> - Nikolaos Al. Papadopoulos, Konstantinos Psannis - [[pdf]](https://arxiv.org/pdf/2603.05789)</summary>

**Abstract:** Multi-agent coordination dilemmas expose a fundamental tension between individual optimization and collective welfare, yet characterizing such coordination requires metrics sensitive to temporal structure and collective dynamics. As a diagnostic testbed, we study a BoE-derived multi-agent variant of the Battle of the Exes, formalizing it as a Markov game in which turn-taking emerges as a periodic coordination regime. Conventional outcome-based metrics (e.g., efficiency and min/max fairness) are temporally blind (they cannot distinguish structured alternation from monopolistic or random access patterns) and fairness ratios lose discriminative power as n grows, obscuring inequities.
To address this limitation, we introduce Perfect Alternation (PA) as a reference coordination regime and propose six novel Alternation (ALT) metrics designed as temporally sensitive observables of coordination quality. Using Q-learning agents as a minimal adaptive diagnostic baseline, and comparing against random-policy null processes, we uncover a clear measurement failure: despite exhibiting deceptively high traditional metrics (e.g., reward fairness often exceeding 0.9), learned policies perform up to 81% below random baselines under ALT-variant evaluation, a deficit already present in the two-agent case and intensifying as n grows.
These results demonstrate, in this setting, that high aggregate payoffs can coexist with poor temporal coordination, and that conventional metrics may severely mischaracterize emergent dynamics. Our findings underscore the necessity of temporally aware observables for analyzing coordination in multi-agent games and highlight random-policy baselines as essential null processes for interpreting coordination outcomes relative to chance-level behavior.

**arXiv ID:** 2603.05789
</details>

<details>
<summary><strong>RACAS: Controlling Diverse Robots With a Single Agentic System</strong> - Dylan R. Ashley, Jan Przepióra, Yimeng Chen, Ali Abualsaud, Nurzhan Yesmagambet, Shinkyu Park, Eric Feron, Jürgen Schmidhuber - [[pdf]](https://arxiv.org/pdf/2603.05621)</summary>

**Abstract:** Many robotic platforms expose an API through which external software can command their actuators and read their sensors. However, transitioning from these low-level interfaces to high-level autonomous behaviour requires a complicated pipeline, whose components demand distinct areas of expertise. Existing approaches to bridging this gap either require retraining for every new embodiment or have only been validated across structurally similar platforms. We introduce RACAS (Robot-Agnostic Control via Agentic Systems), a cooperative agentic architecture in which three LLM/VLM-based modules (Monitors, a Controller, and a Memory Curator) communicate exclusively through natural language to provide closed-loop robot control. RACAS requires only a natural language description of the robot, a definition of available actions, and a task specification; no source code, model weights, or reward functions need to be modified to move between platforms. We evaluate RACAS on several tasks using a wheeled ground robot, a recently published novel multi-jointed robotic limb, and an underwater vehicle. RACAS consistently solved all assigned tasks across these radically different platforms, demonstrating the potential of agentic AI to substantially reduce the barrier to prototyping robotic solutions.

**arXiv ID:** 2603.05621
</details>

<details>
<summary><strong>LaTeXTrans: Structured LaTeX Translation with Multi-Agent Coordination</strong> - Ziming Zhu, Chenglong Wang, Haosong Xv, Shunjie Xing, Yifu Huo, Fengning Tian, Quan Du, Di Yang, Chunliang Zhang, Tong Xiao, Jingbo Zhu - [[pdf]](https://arxiv.org/pdf/2508.18791)</summary>

**Abstract:** Despite the remarkable progress of modern machine translation (MT) systems on general-domain texts, translating structured LaTeX-formatted documents remains a significant challenge. These documents typically interleave natural language with domain-specific syntax, such as mathematical equations, tables, figures, and cross-references, all of which must be accurately preserved to maintain semantic integrity and compilability. In this paper, we introduce LaTeXTrans, a collaborative multi-agent system designed to address this challenge. LaTeXTrans ensures format preservation, structural fidelity, and terminology consistency through six specialized agents: 1) a Parser that decomposes LaTeX into translation-friendly units via placeholder substitution and syntax filtering; 2) a Translator, Validator, Summarizer, and Terminology Extractor that work collaboratively to ensure context-aware, self-correcting, and terminology-consistent translations; 3) a Generator that reconstructs the translated content into well-structured LaTeX documents. Experimental results show that LaTeXTrans outperforms mainstream MT systems in both translation accuracy and structural preservation. The source code, the online demonstration platform, and a demo video are publicly available.

**arXiv ID:** 2508.18791
</details>

<details>
<summary><strong>Communication Enables Cooperation in LLM Agents: A Comparison with Curriculum-Based Approaches</strong> - Hachem Madmoun, Salem Lahlou - [[pdf]](https://arxiv.org/pdf/2510.05748)</summary>

**Abstract:** Eliciting cooperation in multi-agent LLM systems is critical for AI alignment. We investigate two approaches: direct communication and curriculum learning. In a 4-player Stag Hunt, a one-word "cheap talk" channel increases cooperation from 0% to 96.7%, demonstrating communication as a robust coordination mechanism. In contrast, we find that curriculum learning is highly sensitive to design choices: our pedagogical curriculum through progressively complex games reduced agent payoffs by 27.4% in an Iterated Public Goods Game with Punishment, demonstrating that optimizing for short-term rationality can actively undermine alignment goals. Qualitative analysis reveals that curricula emphasizing defection-equilibrium games can induce "learned pessimism" in agents. These findings suggest that for coordination problems, simple communication protocols may be more reliable than experience-based training, and that curriculum design for social dilemmas requires careful attention to the strategic lessons embedded in game sequences.

**arXiv ID:** 2510.05748
</details>

</details>

<details open>
<summary><h2>Other Agent Research (11 papers)</h2></summary>

<details>
<summary><strong>Nurture-First Agent Development: Building Domain-Expert AI Agents Through Conversational Knowledge Crystallization</strong> - Linghao Zhang - [[pdf]](https://arxiv.org/pdf/2603.10808)</summary>

**Abstract:** The emergence of large language model (LLM)-based agent frameworks has shifted the primary challenge in building domain-expert AI agents from raw capability to effective encoding of domain expertise. Two dominant paradigms -- code-first development, which embeds expertise in deterministic pipelines, and prompt-first development, which captures expertise in static system prompts -- both treat agent construction as a discrete engineering phase preceding deployment. We argue that this sequential assumption creates a fundamental mismatch with the nature of domain expertise, which is substantially tacit, deeply personal, and continuously evolving. We propose Nurture-First Development (NFD), a paradigm in which agents are initialized with minimal scaffolding and progressively grown through structured conversational interaction with domain practitioners. The central mechanism is the Knowledge Crystallization Cycle, whereby fragmented knowledge embedded in operational dialogue is periodically consolidated into structured, reusable knowledge assets. We formalize NFD through: (1) a Three-Layer Cognitive Architecture organizing agent knowledge by volatility and personalization degree; (2) the Knowledge Crystallization Cycle with formal definitions of crystallization operations and efficiency metrics; and (3) an operational framework comprising a Dual-Workspace Pattern and Spiral Development Model. We illustrate the paradigm through a detailed case study on building a financial research agent for U.S. equity analysis and discuss the conditions, limitations, and broader implications of NFD for human-agent co-evolution.

**arXiv ID:** 2603.10808
</details>

<details>
<summary><strong>How to Count AIs: Individuation and Liability for AI Agents</strong> - Yonathan Arbel, Peter Salib, Simon Goldstein - [[pdf]](https://arxiv.org/pdf/2603.10028)</summary>

**Abstract:** Very soon, millions of AI agents will proliferate across the economy, autonomously taking billions of actions. Inevitably, things will go wrong. Humans will be defrauded, injured, even killed. Law will somehow have to govern the coming wave. But when an AI causes harm, the first question to answer, before anyone can be held accountable is: Which AI Did It? Identifying AIs is unusually difficult. AIs lack bodies. They can copy, split, merge, swarm, and vanish at will. Even today, a "single" AI agent is often an ensemble of instances based on multiple models. The complexity will only multiply as AI capabilities improve. This Article is the first to comprehensively diagnose the legal problem of identifying AIs. Two kinds of identity are required: "thin" and "thick." Thin identification ties every AI action to some human principal, essential for holding accountable the humans who make and use AI agents. Thick identification distinguishes between AI agents, qua agents -- sorting millions of AI entities into discrete, persistent units with stable, coherent goals, essential where principal-agent problems prevent humans from perfectly controlling AIs. This Article also presents a solution: the "Algorithmic Corporation" or "A-corp" -- a legal-fictional entity that can hold property, make contracts, and litigate in its own name. Owned by humans but run by AIs, A-corps solve the thin identity problem by tying AI actions to a human owner, and the thick identity problem via emergent self-organization. A-corps own the resources -- including compute -- that AIs need to accomplish their goals, giving AI managers strong incentives to share control only with goal-aligned AIs. In equilibrium, incentive and selection mechanisms force A-corps to self-organize into persistent, legally legible entities with coherent goals that respond rationally to legal incentives, like liability.

**arXiv ID:** 2603.10028
</details>

<details>
<summary><strong>Targeted Bit-Flip Attacks on LLM-Based Agents</strong> - Jialai Wang, Ya Wen, Zhongmou Liu, Yuxiao Wu, Bingyi He, Zongpeng Li, Ee-Chien Chang - [[pdf]](https://arxiv.org/pdf/2603.10042)</summary>

**Abstract:** Targeted bit-flip attacks (BFAs) exploit hardware faults to manipulate model parameters, posing a significant security threat. While prior work targets single-step inference models (e.g., image classifiers), LLM-based agents with multi-stage pipelines and external tools present new attack surfaces, which remain unexplored. This work introduces Flip-Agent, the first targeted BFA framework for LLM-based agents, manipulating both final outputs and tool invocations. Our experiments show that Flip-Agent significantly outperforms existing targeted BFAs on real-world agent tasks, revealing a critical vulnerability in LLM-based agent systems.

**arXiv ID:** 2603.10042
</details>

<details>
<summary><strong>Execution Is the New Attack Surface: Survivability-Aware Agentic Crypto Trading with OpenClaw-Style Local Executors</strong> - Ailiya Borjigin, Igor Stadnyk, Ben Bilski, Serhii Hovorov, Sofiia Pidturkina - [[pdf]](https://arxiv.org/pdf/2603.10092)</summary>

**Abstract:** OpenClaw-style agent stacks turn language into privileged execution: LLM intents flow through tool interception, policy gates, and a local executor. In parallel, skill marketplaces such as this http URL make capability acquisition as easy as installing skills and CLIs, creating a growing capability supply chain. Together, these trends shift the dominant safety failure mode from "wrong answers" to execution-induced loss, where untrusted prompts, compromised skills, or narrative manipulation can trigger real trades and irreversible side effects. We propose Survivability-Aware Execution (SAE), an execution-layer survivability standard for OpenClaw-style systems and skill-enabled agents. SAE sits as middleware between a strategy engine (LLM or non-LLM) and the exchange executor. It defines an explicit execution contract (ExecutionRequest, ExecutionContext, ExecutionDecision) and enforces non-bypassable last-mile invariants: projection-based exposure budgets, cooldown and order-rate limits, slippage bounds, staged execution, and tool/venue allowlists. To make delegated execution testable under supply-chain risk, we operationalize the Delegation Gap (DG) via a logged Intended Policy Spec that enables deterministic out-of-scope labeling and reproducible DG metrics. On an offline replay using official Binance USD-M BTCUSDT/ETHUSDT perpetual data (15m; 2025-09-01--2025-12-01, incl. funding), SAE improves survivability: MDD drops from 0.4643 to 0.0319 (Full; 93.1%), |CVaR_0.99| shrinks from 4.025e-3 to ~1.02e-4 (~97.5%), and DG loss proxy falls from 0.647 to 0.019 (~97.0%). AttackSuccess decreases from 1.00 to 0.728 with zero FalseBlock in this run. Block bootstrap, paired Wilcoxon, and two-proportion tests confirm the shifts. SAE reframes agentic trading safety for the OpenClaw+skills era: treat upstream intent and skills as untrusted, and enforce survivability where actions become side effects.

**arXiv ID:** 2603.10092
</details>

<details>
<summary><strong>Simulation-in-the-Reasoning (SiR): A Conceptual Framework for Empirically Grounded AI in Autonomous Transportation</strong> - Wuping Xin - [[pdf]](https://arxiv.org/pdf/2603.10294)</summary>

**Abstract:** Large Language Models (LLMs) have advanced reasoning through techniques like Chain-of-Thought (CoT). However, their reasoning largely re-mains textual and hypothetical, lacking empirical grounding in complex, dynamic domains like transportation. This paper introduces Simulation-in-the-Reasoning (SiR), a novel conceptual framework that embeds domain-specific simulators directly into the LLM reasoning loop. By treating intermediate reasoning steps as executable simulation experiments, SiR transforms LLM reasoning from narrative plausibility into a falsifiable, hypothesis-simulate-analyze workflow. We discuss applications, where LLM can formulate Intelligent Transport System (ITS) strategy hypotheses, invoke a traffic simulator via the Model Context Protocol (MCP), evaluate results under different demand patterns, and refine strategies through verification and aggregation. While implementing the framework is part of our ongoing work, this paper primarily establishes the conceptual foundation, discusses design considerations like API granularity, and outlines the vision of SiR as a cornerstone for interactive transportation digital twins. We argue that SiR represents a critical step towards trustworthy, empirically-validated AI for autonomous transportation systems.

**arXiv ID:** 2603.10294
</details>

<details>
<summary><strong>What Do Agents Think One Another Want? Level-2 Inverse Games for Inferring Agents' Estimates of Others' Objectives</strong> - Hamzah I. Khan, Jingqi Li, David Fridovich-Keil - [[pdf]](https://arxiv.org/pdf/2508.03824)</summary>

**Abstract:** Effectively interpreting strategic interactions among multiple agents requires us to infer each agent's objective from limited information. Existing inverse game-theoretic approaches frame this challenge in terms of a "level-1" inference problem, in which we take the perspective of a third-party observer and assume that individual agents share complete knowledge of one another's objectives. However, this assumption breaks down in decentralized, real-world scenarios like urban driving and bargaining, in which agents may act based on conflicting views of one another's objectives. We demonstrate the necessity of inferring agents' different estimates of each other's objectives through empirical examples, and by theoretically characterizing the prediction error of level-1 inference on fictitious gameplay data from linear-quadratic games. To address this fundamental issue, we propose a framework for level-2 inference to address the question: "What does each agent believe about other agents' objectives?" We prove that the level-2 inference problem is non-convex even in benign settings like linear-quadratic games, and we develop an efficient gradient-based approach for identifying local solutions. Experiments on a synthetic urban driving example show that our approach uncovers nuanced misalignments that level-1 methods miss.

**arXiv ID:** 2508.03824
</details>

<details>
<summary><strong>Autonomous Search for Sparsely Distributed Visual Phenomena through Environmental Context Modeling</strong> - Eric Chen, Travis Manderson, Nare Karapetyan, Peter Edmunds, Nicholas Roy, Yogesh Girdhar - [[pdf]](https://arxiv.org/pdf/2603.10174)</summary>

**Abstract:** Autonomous underwater vehicles (AUVs) are increasingly used to survey coral reefs, yet efficiently locating specific coral species of interest remains difficult: target species are often sparsely distributed across the reef, and an AUV with limited battery life cannot afford to search everywhere. When detections of the target itself are too sparse to provide directional guidance, the robot benefits from an additional signal to decide where to look next. We propose using the visual environmental context -- the habitat features that tend to co-occur with a target species -- as that signal. Because context features are spatially denser and often vary more smoothly than target detections, we hypothesize that a reward function targeted at broader environmental context will enable adaptive planners to make better decisions on where to go next, even in regions where no target has yet been observed. Starting from a single labeled image, our method uses patch-level DINOv2 embeddings to perform one-shot detections of both the target species and its surrounding context online. We validate our approach using real imagery collected by an AUV at two reef sites in St. John, U.S. Virgin Islands, simulating the robot's motion offline. Our results demonstrate that one-shot detection combined with adaptive context modeling enables efficient autonomous surveying, sampling up to 75$\%$ of the target in roughly half the time required by exhaustive coverage when the target is sparsely distributed, and outperforming search strategies that only use target detections.

**arXiv ID:** 2603.10174
</details>

<details>
<summary><strong>Cybo-Waiter: A Physical Agentic Framework for Humanoid Whole-Body Locomotion-Manipulation</strong> - Peng Ren, Haoyang Ge, Chuan Qi, Cong Huang, Hong Li, Jiang Zhao, Pei Chi, Kai Chen - [[pdf]](https://arxiv.org/pdf/2603.10675)</summary>

**Abstract:** Robots are increasingly expected to execute open ended natural language requests in human environments, which demands reliable long horizon execution under partial observability. This is especially challenging for humanoids because locomotion and manipulation are tightly coupled through stance, reachability, and balance. We present a humanoid agent framework that turns VLM plans into verifiable task programs and closes the loop with multi object 3D geometric supervision. A VLM planner compiles each instruction into a typed JSON sequence of subtasks with explicit predicate based preconditions and success conditions. Using SAM3 and RGB-D, we ground all task relevant entities in 3D, estimate object centroids and extents, and evaluate predicates over stable frames to obtain condition level diagnostics. The supervisor uses these diagnostics to verify subtask completion and to provide condition-level feedback for progression and replanning. We execute each subtask by coordinating humanoid locomotion and whole-body manipulation, selecting feasible motion primitives under reachability and balance constraints. Experiments on tabletop manipulation and long horizon humanoid loco manipulation tasks show improved robustness from multi object grounding, temporal stability, and recovery driven replanning.

**arXiv ID:** 2603.10675
</details>

<details>
<summary><strong>MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent</strong> - Yuxia Fu, Zhizhen Zhang, Yuqi Zhang, Zijian Wang, Zi Huang, Yadan Luo - [[pdf]](https://arxiv.org/pdf/2511.18810)</summary>

**Abstract:** Recent Vision-Language-Action (VLA) models reformulate vision-language models by tuning them with millions of robotic demonstrations. While they perform well when fine-tuned for a single embodiment or task family, extending them to multi-skill settings remains challenging: directly merging VLA experts trained on different tasks results in near-zero success rates. This raises a fundamental question: what prevents VLAs from mastering multiple skills within one model? With an empirical decomposition of learnable parameters during VLA fine-tuning, we identify two key sources of non-mergeability: (1) Finetuning drives LoRA adapters in the VLM backbone toward divergent, task-specific directions beyond the capacity of existing merging methods to unify. (2) Action experts develop inter-block dependencies through self-attention feedback, causing task information to spread across layers and preventing modular recombination. To address these challenges, we present MergeVLA, a merging-oriented VLA architecture that preserves mergeability by design. MergeVLA introduces sparsely activated LoRA adapters via task masks to retain consistent parameters and reduce irreconcilable conflicts in the VLM. Its action expert replaces self-attention with cross-attention-only blocks to keep specialization localized and composable. When the task is unknown, it uses a test-time task router to adaptively select the appropriate task mask and expert head from the initial observation, enabling unsupervised task inference. Across LIBERO, LIBERO-Plus, RoboTwin, and multi-task experiments on the real SO101 robotic arm, MergeVLA achieves performance comparable to or even exceeding individually finetuned experts, demonstrating robust generalization across tasks, embodiments, and environments. Project page: this https URL

**arXiv ID:** 2511.18810
</details>

<details>
<summary><strong>Robust Cooperative Localization in Featureless Environments: A Comparative Study of DCL, StCL, CCL, CI, and Standard-CL</strong> - Nivand Khosravi, Rodrigo Ventura, Meysam Basiri - [[pdf]](https://arxiv.org/pdf/2603.09886)</summary>

**Abstract:** Cooperative localization (CL) enables accurate position estimation in multi-robot systems operating in GPS-denied environments. This paper presents a comparative study of five CL approaches: Centralized Cooperative Localization (CCL), Decentralized Cooperative Localization (DCL), Sequential Cooperative Localization (StCL), Covariance Intersection (CI), and Standard Cooperative Localization (Standard-CL). All methods are implemented in ROS and evaluated through Monte Carlo simulations under two conditions: weak data association and robust detection. Our analysis reveals fundamental trade-offs among the methods. StCL and Standard-CL achieve the lowest position errors but exhibit severe filter inconsistency, making them unsuitable for safety-critical applications. DCL demonstrates remarkable stability under challenging conditions due to its measurement stride mechanism, which provides implicit regularization against outliers. CI emerges as the most balanced approach, achieving near-optimal consistency while maintaining competitive accuracy. CCL provides theoretically optimal estimation but shows sensitivity to measurement outliers. These findings offer practical guidance for selecting CL algorithms based on application requirements.

**arXiv ID:** 2603.09886
</details>

<details>
<summary><strong>Terminal Is All You Need: Design Properties for Human-AI Agent Collaboration</strong> - Alexandre De Masi - [[pdf]](https://arxiv.org/pdf/2603.10664)</summary>

**Abstract:** While research on AI agents focuses on enabling them to operate graphical user interfaces, the most effective and widely adopted agent tools in practice are terminal-based. We argue that this convergence is not coincidental. It reflects three design properties central to effective human-AI-UI collaboration: representational compatibility between agent and interface, transparency of agent actions within the interaction medium, and low barriers to entry for human participants. We ground each property in established HCI theory, show how terminal-based tools satisfy them by default, and argue that any modality, including graphical and spatial interfaces, must be deliberately engineered to achieve them. Rather than a legacy artifact, the terminal serves as a design exemplar whose properties any agent-facing modality must replicate.

**arXiv ID:** 2603.10664
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (41 papers)</h2></summary>

<details>
<summary><strong>Agentic Control Center for Data Product Optimization</strong> - Priyadarshini Tamilselvan, Gregory Bramble, Sola Shirai, Ken C. L. Wong, Faisal Chowdhury, Horst Samulowitz - [[pdf]](https://arxiv.org/pdf/2603.10133)</summary>

**Abstract:** Data products enable end users to gain greater insights about their data by providing supporting assets, such as example question-SQL pairs which can be answered using the data or views over the database tables. However, producing useful data products is challenging, and typically requires domain experts to hand-craft supporting assets. We propose a system that automates data product improvement through specialized AI agents operating in a continuous optimization loop. By surfacing questions, monitoring multi-dimensional quality metrics, and supporting human-in-the-loop controls, it transforms data into observable and refinable assets that balance automation with trust and oversight.

**arXiv ID:** 2603.10133
</details>

<details>
<summary><strong>Adaptive RAN Slicing Control via Reward-Free Self-Finetuning Agents</strong> - Yuanhao Li, Haozhe Wang, Geyong Min, Nektarios Georgalas, Wang Miao - [[pdf]](https://arxiv.org/pdf/2603.10564)</summary>

**Abstract:** The integration of Generative AI models into AI-native network systems offers a transformative path toward achieving autonomous and adaptive control. However, the application of such models to continuous control tasks is impeded by intrinsic architectural limitations, including finite context windows, the lack of explicit reward signals, and the degradation of the long context. This paper posits that the key to unlocking robust continuous control is enabling agents to internalize experience by distilling it into their parameters, rather than relying on prompt-based memory. To this end, we propose a novel self-finetuning framework that enables agentic systems to learn continuously through direct interaction with the environment, bypassing the need for handcrafted rewards. Our framework implements a bi-perspective reflection mechanism that generates autonomous linguistic feedback to construct preference datasets from interaction history. A subsequent preference-based fine-tuning process distills long-horizon experiences into the model's parameters. We evaluate our approach on a dynamic Radio Access Network (RAN) slicing task, a challenging multi-objective control problem that requires the resolution of acute trade-offs between spectrum efficiency, service quality, and reconfiguration stability under volatile network conditions. Experimental results show that our framework outperforms standard Reinforcement Learning (RL) baselines and existing Large Language Model (LLM)-based agents in sample efficiency, stability, and multi-metric optimization. These findings demonstrate the potential of self-improving generative agents for continuous control tasks, paving the way for future AI-native network infrastructure.

**arXiv ID:** 2603.10564
</details>

<details>
<summary><strong>Hybrid Self-evolving Structured Memory for GUI Agents</strong> - Sibo Zhu, Wenyi Wu, Kun Zhou, Stephen Wang, Biwei Huang - [[pdf]](https://arxiv.org/pdf/2603.10291)</summary>

**Abstract:** The remarkable progress of vision-language models (VLMs) has enabled GUI agents to interact with computers in a human-like manner. Yet real-world computer-use tasks remain difficult due to long-horizon workflows, diverse interfaces, and frequent intermediate errors. Prior work equips agents with external memory built from large collections of trajectories, but relies on flat retrieval over discrete summaries or continuous embeddings, falling short of the structured organization and self-evolving characteristics of human memory. Inspired by the brain, we propose Hybrid Self-evolving Structured Memory (HyMEM), a graph-based memory that couples discrete high-level symbolic nodes with continuous trajectory embeddings. HyMEM maintains a graph structure to support multi-hop retrieval, self-evolution via node update operations, and on-the-fly working-memory refreshing during inference. Extensive experiments show that HyMEM consistently improves open-source GUI agents, enabling 7B/8B backbones to match or surpass strong closed-source models; notably, it boosts Qwen2.5-VL-7B by +22.5% and outperforms Gemini2.5-Pro-Vision and GPT-4o.

**arXiv ID:** 2603.10291
</details>

<details>
<summary><strong>Trajectory-Informed Memory Generation for Self-Improving Agent Systems</strong> - Gaodan Fang, Vatche Isahagian, K. R. Jayaram, Ritesh Kumar, Vinod Muthusamy, Punleuk Oum, Gegi Thomas - [[pdf]](https://arxiv.org/pdf/2603.10600)</summary>

**Abstract:** LLM-powered agents face a persistent challenge: learning from their execution experiences to improve future performance. While agents can successfully complete many tasks, they often repeat inefficient patterns, fail to recover from similar errors, and miss opportunities to apply successful strategies from past executions. We present a novel framework for automatically extracting actionable learnings from agent execution trajectories and utilizing them to improve future performance through contextual memory retrieval. Our approach comprises four components: (1) a Trajectory Intelligence Extractor that performs semantic analysis of agent reasoning patterns, (2) a Decision Attribution Analyzer that identifies which decisions and reasoning steps led to failures, recoveries, or inefficiencies, (3) a Contextual Learning Generator that produces three types of guidance -- strategy tips from successful patterns, recovery tips from failure handling, and optimization tips from inefficient but successful executions, and (4) an Adaptive Memory Retrieval System that injects relevant learnings into agent prompts based on multi-dimensional similarity. Unlike existing memory systems that store generic conversational facts, our framework understands execution patterns, extracts structured learnings with provenance, and retrieves guidance tailored to specific task contexts. Evaluation on the AppWorld benchmark demonstrates consistent improvements, with up to 14.3 percentage point gains in scenario goal completion on held-out tasks and particularly strong benefits on complex tasks (28.5~pp scenario goal improvement, a 149\% relative increase).

**arXiv ID:** 2603.10600
</details>

<details>
<summary><strong>Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards</strong> - Zhengzhao Ma, Xueru Wen, Boxi Cao, Yaojie Lu, Hongyu Lin, Jinglin Yang, Min He, Xianpei Han, Le Sun - [[pdf]](https://arxiv.org/pdf/2603.09117)</summary>

**Abstract:** Reinforcement Learning from Verifiable Rewards (RLVR) significantly enhances large language models (LLMs) reasoning but severely suffers from calibration degeneration, where models become excessively over-confident in incorrect answers. Previous studies devote to directly incorporating calibration objective into existing optimization target. However, our theoretical analysis demonstrates that there exists a fundamental gradient conflict between the optimization for maximizing policy accuracy and minimizing calibration error. Building on this insight, we propose DCPO, a simple yet effective framework that systematically decouples reasoning and calibration objectives. Extensive experiments demonstrate that our DCPO not only preserves accuracy on par with GRPO but also achieves the best calibration performance and substantially mitigates the over-confidence issue. Our study provides valuable insights and practical solution for more reliable LLM deployment.

**arXiv ID:** 2603.09117
</details>

<details>
<summary><strong>Reinforcement Learning with Conditional Expectation Reward</strong> - Changyi Xiao, Caijun Xu, Yixin Cao - [[pdf]](https://arxiv.org/pdf/2603.10624)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective in enhancing the reasoning capabilities of large language models, particularly in domains such as mathematics where reliable rule-based verifiers can be constructed. However, the reliance on handcrafted, domain-specific verification rules substantially limits the applicability of RLVR to general reasoning domains with free-form answers, where valid answers often exhibit significant variability, making it difficult to establish complete and accurate rules. To address this limitation, we propose Conditional Expectation Reward (CER), which leverages the large language model itself as an implicit verifier, and is therefore applicable to general domains and eliminates the need for external verifiers or auxiliary models. CER is defined as the expected likelihood of generating the reference answer conditioned on the generated answer. In contrast to rule-based verifiers that yield binary feedback, CER provides a soft, graded reward signal that reflects varying degrees of correctness, making it better suited to tasks where answers vary in correctness. Experimental results demonstrate that CER is effective across a wide range of reasoning tasks, spanning both mathematical and general domains, indicating that CER serves as a flexible and general verification mechanism. The code is available at this https URL.

**arXiv ID:** 2603.10624
</details>

<details>
<summary><strong>Structured Linked Data as a Memory Layer for Agent-Orchestrated Retrieval</strong> - Andrea Volpini, Elie Raad, Beatrice Gamba, David Riccitelli - [[pdf]](https://arxiv.org/pdf/2603.10700)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) systems typically treat documents as flat text, ignoring the structured metadata and linked relationships that knowledge graphs provide. In this paper, we investigate whether structured linked data, specifically this http URL markup and dereferenceable entity pages served by a Linked Data Platform, can improve retrieval accuracy and answer quality in both standard and agentic RAG systems. We conduct a controlled experiment across four domains (editorial, legal, travel, e-commerce) using Vertex AI Vector Search 2.0 for retrieval and the Google Agent Development Kit (ADK) for agentic reasoning. Our experimental design tests seven conditions: three document representations (plain HTML, HTML with JSON-LD, and an enhanced agentic-optimized entity page) crossed with two retrieval modes (standard RAG and agentic RAG with multi-hop link traversal), plus an Enhanced+ condition that adds rich navigational affordances and entity interlinking. Our results reveal that while JSON-LD markup alone provides only modest improvements, our enhanced entity page format, incorporating this http URL-style agent instructions, breadcrumbs, and neural search capabilities, achieves substantial gains: +29.6% accuracy improvement for standard RAG and +29.8% for the full agentic pipeline. The Enhanced+ variant, with richer navigational affordances, achieves the highest absolute scores (accuracy: 4.85/5, completeness: 4.55/5), though the incremental gain over the base enhanced format is not statistically significant. We release our dataset, evaluation framework, and enhanced entity page templates to support reproducibility.

**arXiv ID:** 2603.10700
</details>

<details>
<summary><strong>Towards Intelligent Spectrum Management: Spectrum Demand Estimation Using Graph Neural Networks</strong> - Mohamad Alkadamani, Amir Ghasemi, Halim Yanikomeroglu - [[pdf]](https://arxiv.org/pdf/2603.10802)</summary>

**Abstract:** The growing demand for wireless connectivity, combined with limited spectrum resources, calls for more efficient spectrum management. Spectrum sharing is a promising approach; however, regulators need accurate methods to characterize demand dynamics and guide allocation decisions. This paper builds and validates a spectrum demand proxy from public deployment records and uses a graph attention network in a hierarchical, multi-resolution setup (HR-GAT) to estimate spectrum demand at fine spatial scales. The model captures both neighborhood effects and cross-scale patterns, reducing spatial autocorrelation and improving generalization. Evaluated across five Canadian cities and against eight competitive baselines, HR-GAT reduces median RMSE by roughly 21% relative to the best alternative and lowers residual spatial bias. The resulting demand maps are regulator-accessible and support spectrum sharing and spectrum allocation in wireless networks.

**arXiv ID:** 2603.10802
</details>

<details>
<summary><strong>Learning What Reinforcement Learning Can't: Interleaved Online Fine-Tuning for Hardest Questions</strong> - Lu Ma, Hao Liang, Meiyi Qiang, Lexiang Tang, Xiaochen Ma, Zhen Hao Wong, Junbo Niu, Chengyu Shen, Runming He, Yanhao Li, Bin Cui, Wentao Zhang - [[pdf]](https://arxiv.org/pdf/2506.07527)</summary>

**Abstract:** Recent advances in large language model (LLM) reasoning have shown that sophisticated behaviors such as planning and self-reflection can emerge through reinforcement learning (RL). However, despite these successes, RL in its current form remains insufficient to induce capabilities that exceed the limitations of the base model, as it is primarily optimized based on existing knowledge of the model rather than facilitating the acquisition of new information. To address this limitation, we employ supervised fine-tuning (SFT) to learn what RL cannot, which enables the incorporation of new knowledge and reasoning patterns by leveraging high-quality demonstration data. We analyze the training dynamics of RL and SFT for LLM reasoning and find that RL excels at maintaining and improving performance on questions within the model's original capabilities, while SFT is more effective at enabling progress on questions beyond the current scope of the model. Motivated by the complementary strengths of RL and SFT, we introduce a novel training approach, \textbf{ReLIFT} (\textbf{Re}inforcement \textbf{L}earning \textbf{I}nterleaved with Online \textbf{F}ine-\textbf{T}uning). In ReLIFT, the model is primarily trained using RL, but when it encounters challenging questions, high-quality solutions are collected for fine-tuning, and the training process alternates between RL and fine-tuning to enhance the model's reasoning abilities. ReLIFT achieves an average improvement of over +5.2 points across five competition-level benchmarks and one out-of-distribution benchmark compared to other zero-RL models. Furthermore, we demonstrate that ReLIFT outperforms both RL and SFT while using only 13\% of the detailed demonstration data, highlighting its scalability. These results provide compelling evidence that ReLIFT overcomes the fundamental limitations of RL and underscores the significant potential.

**arXiv ID:** 2506.07527
</details>

<details>
<summary><strong>To Mix or To Merge: Toward Multi-Domain Reinforcement Learning for Large Language Models</strong> - Haoqing Wang, Xiang Long, Ziheng Li, Yilong Xu, Tingguang Li, Yehui Tang - [[pdf]](https://arxiv.org/pdf/2602.12566)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) plays a key role in stimulating the explicit reasoning capability of Large Language Models (LLMs). We can achieve expert-level performance in some specific domains via RLVR, such as coding or math. When a general multi-domain expert-level model is required, we need to carefully consider the collaboration of RLVR across different domains. The current state-of-the-art models mainly employ two different training paradigms for multi-domain RLVR: mixed multi-task RLVR and separate RLVR followed by model merging. However, most of the works did not provide a detailed comparison and analysis about these paradigms. To this end, we choose multiple commonly used high-level tasks (e.g., math, coding, science, instruction following, and agent) as our target domains and design extensive qualitative and quantitative experiments using open-source datasets. We find the RLVR across domains exhibits few mutual interferences, and reasoning-intensive domains demonstrate mutually synergistic effects. Furthermore, we analyze the internal mechanisms of mutual gains from the perspectives of weight space geometry, information constraints, model prediction behavior and self-verification. This project is named as M2RL that means Mixed multi-task training or separate training followed by model Merging for Reinforcement Learning, and the homepage is at this https URL.

**arXiv ID:** 2602.12566
</details>

<details>
<summary><strong>CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework</strong> - Yuexi Du, Jinglu Wang, Shujie Liu, Nicha C. Dvornek, Yan Lu - [[pdf]](https://arxiv.org/pdf/2603.01607)</summary>

**Abstract:** Large visual language models (VLMs) have shown strong multi-modal medical reasoning ability, but most operate as end-to-end black boxes, diverging from clinicians' evidence-based, staged workflows and hindering clinical accountability. Complementarily, expert visual grounding models can accurately localize regions of interest (ROIs), providing explicit, reliable evidence that improves both reasoning accuracy and trust. In this paper, we introduce CARE, advancing Clinical Accountability in multi-modal medical Reasoning with an Evidence-grounded agentic framework. Unlike existing approaches that couple grounding and reasoning within a single generalist model, CARE decomposes the task into coordinated sub-modules to reduce shortcut learning and hallucination: a compact VLM proposes relevant medical entities; an expert entity-referring segmentation model produces pixel-level ROI evidence; and a grounded VLM reasons over the full image augmented by ROI hints. The VLMs are optimized with reinforcement learning with verifiable rewards to align answers with supporting evidence. Furthermore, a VLM coordinator plans tool invocation and reviews evidence-answer consistency, providing agentic control and final verification. Evaluated on standard medical VQA benchmarks, our CARE-Flow (coordinator-free) improves average accuracy by 10.9% over the same size (10B) state-of-the-art (SOTA). With dynamic planning and answer review, our CARE-Coord yields a further gain, outperforming the heavily pre-trained SOTA by 5.2%. Our experiments demonstrate that an agentic framework that emulates clinical workflows, incorporating decoupled specialized models and explicit evidence, yields more accurate and accountable medical AI. Project page: this https URL

**arXiv ID:** 2603.01607
</details>

<details>
<summary><strong>ToolRLA: Multiplicative Reward Decomposition for Tool-Integrated Agents</strong> - Pengbo Liu - [[pdf]](https://arxiv.org/pdf/2603.01620)</summary>

**Abstract:** Tool-integrated agents that interleave reasoning with API calls are promising for complex tasks, yet aligning them for high-stakes, domain-specific deployment remains challenging: existing reinforcement learning approaches rely on coarse binary rewards that cannot distinguish tool selection errors from malformed parameters. We present ToolRLA, a three-stage post-training pipeline (SFT -> GRPO -> DPO) for domain-specific tool agents. The core contribution is a fine-grained reward function with multiplicative correctness decomposition spanning four dimensions -- format validity, tool selection, parameter accuracy, and regulatory compliance -- that encodes domain priority orderings as inductive biases in the reward landscape. Deployed on a financial advisory copilot (80+ advisors, 1,200+ daily queries), ToolRLA achieves over three months: a 47% improvement in task completion rate (62%->91%), a 63% reduction in tool invocation errors (38%->14%), and a 93% reduction in regulatory violations (12%->0.8%), within sub-2-second latency. Ablation studies show the multiplicative reward design accounts for 7 percentage points of improvement over additive alternatives. Generalization is further validated on ToolBench and API-Bank.

**arXiv ID:** 2603.01620
</details>

<details>
<summary><strong>RetroAgent: From Solving to Evolving via Retrospective Dual Intrinsic Feedback</strong> - Xiaoying Zhang, Zichen Liu, Yipeng Zhang, Xia Hu, Wenqi Shao - [[pdf]](https://arxiv.org/pdf/2603.08561)</summary>

**Abstract:** Large language model (LLM)-based agents trained with reinforcement learning (RL) have shown strong potential on complex interactive tasks. However, standard RL paradigms favor static problem-solving over continuous adaptation: agents often converge to suboptimal strategies due to insufficient exploration, while learned knowledge remains implicit within parameters rather than explicitly retrievable, limiting effective experiential learning. To address these limitations, we introduce RetroAgent, an online RL framework that empowers agents to master complex interactive environments not just by solving, but by evolving. Concretely, RetroAgent features a hindsight self-reflection mechanism that produces dual intrinsic feedback: (1) intrinsic numerical feedback that that tracks incremental subtask completion relative to prior attempts, rewarding promising explorations, and (2) intrinsic language feedback that distills reusable lessons into a memory buffer, retrieved via our proposed Similarity & Utility-Aware Upper Confidence Bound (SimUtil-UCB) strategy balancing relevance, utility, and exploration to effectively leverage past experiences. Extensive experiments on two model families across four challenging agentic tasks demonstrate that RetroAgent significantly outperforms existing methods, achieving state-of-the-art results -- e.g., surpassing Group Relative Policy Optimization (GRPO)-trained agents by +18.3% on ALFWorld, +15.4% on WebShop, +27.1% on Sokoban, and +8.9% on MineSweeper -- while exhibiting strong test-time adaptation and generalization to out-of-distribution scenarios.

**arXiv ID:** 2603.08561
</details>

<details>
<summary><strong>An Updated Assessment of Reinforcement Learning for Macro Placement</strong> - Chung-Kuan Cheng, Andrew B. Kahng, Sayak Kundu, Yucheng Wang, Zhiang Wang - [[pdf]](https://arxiv.org/pdf/2302.11014)</summary>

**Abstract:** We provide an improved assessment of Google Brain's deep reinforcement learning approach to macro placement and its updated Circuit Training (CT) implementation in GitHub. A stronger simulated annealing (SA) baseline leverages the "go-with-the-winners" metaheuristic and a multi-threading implementation. We develop and release new public benchmarks in sub-10nm technology: LEF/DEF for Google's 7nm TSMC Ariane protobuf and scaled variants, as well as testcases implemented in the open-source ASAP7 7nm research enablement. We evaluate from-scratch training and fine-tuning results for the latest "AlphaChip" release of Circuit Training, alongside multiple alternative macro placers. We also study the recently-published pre-training guidance in. A commercial place-and-route tool is used to provide "true reward" post-route power, performance and area metrics. All data, evaluation flows and related scripts are publicly available in the MacroPlacement GitHub repository. Our study affords insights into reproducibility and reporting in the research literature, and points out still-missing confirmations (e.g., of CT's scalability and pre-training methodology) that remain open questions for the research community.

**arXiv ID:** 2302.11014
</details>

<details>
<summary><strong>DeepEyesV2: Toward Agentic Multimodal Model</strong> - Jack Hong, Chenxiao Zhao, ChengLin Zhu, Weiheng Lu, Guohai Xu, Xing Yu - [[pdf]](https://arxiv.org/pdf/2511.05271)</summary>

**Abstract:** Agentic multimodal models should not only comprehend text and images, but also actively invoke external tools, such as code execution environments and web search, and integrate these operations into reasoning. In this work, we introduce DeepEyesV2 and explore how to build an agentic multimodal model from the perspectives of data construction, training methods, and model evaluation. We observe that direct reinforcement learning alone fails to induce robust tool-use behavior. This phenomenon motivates a two-stage training pipeline: a cold-start stage to establish tool-use patterns, and reinforcement learning stage to further refine tool invocation. We curate a diverse, moderately challenging training dataset, specifically including examples where tool use is beneficial. We further introduce RealX-Bench, a comprehensive benchmark designed to evaluate real-world multimodal reasoning, which inherently requires the integration of multiple capabilities, including perception, search, and reasoning. We evaluate DeepEyesV2 on RealX-Bench and other representative benchmarks, demonstrating its effectiveness across real-world understanding, mathematical reasoning, and search-intensive tasks. Moreover, DeepEyesV2 exhibits task-adaptive tool invocation, tending to use image operations for perception tasks and numerical computations for reasoning tasks. Reinforcement learning further enables complex tool combinations and allows model to selectively invoke tools based on context. We hope our study can provide guidance for community in developing agentic multimodal models.

**arXiv ID:** 2511.05271
</details>

<details>
<summary><strong>REMSA: Foundation Model Selection for Remote Sensing via a Constraint-Aware Agent</strong> - Binger Chen, Tacettin Emre Bök, Behnood Rasti, Volker Markl, Begüm Demir - [[pdf]](https://arxiv.org/pdf/2511.17442)</summary>

**Abstract:** Foundation Models (FMs) are increasingly integrated into remote sensing (RS) pipelines. These models include unimodal vision encoders and multimodal architectures. FMs are adapted to diverse perception tasks, such as image classification, change detection, and visual question answering. However, selecting the most suitable remote sensing foundation model (RSFM) for a specific task remains challenging due to scattered documentation, heterogeneous formats, and complex deployment constraints. To address this, we first introduce the RSFM Database (RS-FMD), the first structured and schema-guided resource covering over 160 RSFMs trained on various data modalities, spanning different spatial, spectral, and temporal resolutions, considering different learning paradigms. Built upon RS-FMD, we further present REMSA, a constraint-aware agent that enables automated RSFM selection from natural language queries. REMSA combines structured FM metadata retrieval with a task-driven decision workflow. In detail, it interprets user input, clarifies missing constraints, ranks models via in-context learning, and provides transparent justifications. Our system supports various RS tasks and data modalities, enabling personalized, reproducible, and efficient FM selection. To evaluate REMSA, we construct a benchmark of 100 expert-verified RS query scenarios. Each query is evaluated across 4 systems and 3 LLM backbones, with the top-3 selected models manually assessed by domain experts. This results in 3,000 expert-scored task--system--model configurations under our novel expert-centered evaluation protocol. REMSA outperforms multiple baselines, showing its practical utility in real decision-making applications. REMSA operates entirely on publicly available metadata of open source RSFMs, without accessing private or sensitive data.

**arXiv ID:** 2511.17442
</details>

<details>
<summary><strong>GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training</strong> - Tong Wei, Yijun Yang, Changhao Zhang, Junliang Xing, Yuanchun Shi, Zongqing Lu, Deheng Ye - [[pdf]](https://arxiv.org/pdf/2512.13043)</summary>

**Abstract:** Multi-turn reinforcement learning (RL) for multi-modal agents built upon vision-language models (VLMs) is hampered by sparse rewards and long-horizon credit assignment. Recent methods densify the reward by querying a teacher that provides step-level feedback, e.g., Guided Thought Reinforcement (GTR) and On-Policy Distillation, but rely on costly, often privileged models as the teacher, limiting practicality and reproducibility. We introduce GTR-Turbo, a highly efficient upgrade to GTR that matches its performance without training on or querying an expensive teacher model. Specifically, GTR-Turbo merges the weights of checkpoints produced during ongoing RL training and then uses the resulting merged model as a "free" teacher to guide subsequent RL via supervised fine-tuning or soft logit distillation. This design removes dependence on privileged VLMs (e.g., GPT or Gemini), mitigates the "entropy collapse" observed in prior work, and maintains stable training. Across diverse visual agentic tasks, GTR-Turbo improves the accuracy of the baseline model by 10-30% while reducing wall-clock training time by 50% and compute cost by 60% relative to GTR.

**arXiv ID:** 2512.13043
</details>

<details>
<summary><strong>ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System</strong> - Hao Kang, Ziyang Li, Xinyu Yang, Weili Xu, Yinfang Chen, Junxiong Wang, Beidi Chen, Tushar Krishna, Chenfeng Xu, Simran Arora - [[pdf]](https://arxiv.org/pdf/2602.13692)</summary>

**Abstract:** Large language models(LLMs) are now used to power complex multi-turn agentic workflows. Existing systems run agentic inference by loosely assembling isolated components: an LLM inference engine (e.g., vLLM) and a tool orchestrator (e.g., Kubernetes). Although agentic workflows involve multiple LLM and tool requests, these systems schedule and allocate resources separately on a per-request basis, without end-to-end knowledge of the workflow. This leads to sub-optimal management of KV cache and tool execution environments. To address the challenges, we propose ThunderAgent, a fast, simple, and program-aware agentic inference system. We first abstract agentic workflows as LLM Programs, enabling a unified view of heterogeneous resources, including KV caches, system states, and external tool assets such as disk memory and network ports. Built upon this abstraction, ThunderAgent introduces a program-aware scheduler and a tool resource manager designed to maximize KV cache hit rates, mitigate memory imbalances, and enable asynchronous environment preparation. Evaluations across coding, routing, and scientific discovery agents demonstrate that ThunderAgent achieves 1.5-3.6x throughput improvements in serving, 1.8-3.9x in RL rollout, and up to 4.2x disk memory savings compared to state-of-the-art inference systems. To facilitate reproducibility and support future development, we open-source the system implementations of the whole ThunderAgent at: this https URL.

**arXiv ID:** 2602.13692
</details>

<details>
<summary><strong>OpenClaw-RL: Train Any Agent Simply by Talking</strong> - Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, Ling Yang - [[pdf]](https://arxiv.org/pdf/2603.10165)</summary>

**Abstract:** Every agent interaction generates a next-state signal, namely the user reply, tool output, terminal or GUI state change that follows each action, yet no existing agentic RL system recovers it as a live, online learning source. We present OpenClaw-RL, a framework built on a simple observation: next-state signals are universal, and policy can learn from all of them simultaneously. Personal conversations, terminal executions, GUI interactions, SWE tasks, and tool-call traces are not separate training problems. They are all interactions that can be used to train the same policy in the same loop. Next-state signals encode two forms of information: evaluative signals, which indicate how well the action performed and are extracted as scalar rewards via a PRM judge; and directive signals, which indicate how the action should have been different and are recovered through Hindsight-Guided On-Policy Distillation (OPD). We extract textual hints from the next state, construct an enhanced teacher context, and provide token-level directional advantage supervision that is richer than any scalar reward. Due to the asynchronous design, the model serves live requests, the PRM judges ongoing interactions, and the trainer updates the policy at the same time, with zero coordination overhead between them. Applied to personal agents, OpenClaw-RL enables an agent to improve simply by being used, recovering conversational signals from user re-queries, corrections, and explicit feedback. Applied to general agents, the same infrastructure supports scalable RL across terminal, GUI, SWE, and tool-call settings, where we additionally demonstrate the utility of process rewards. Code: this https URL

**arXiv ID:** 2603.10165
</details>

<details>
<summary><strong>Human-AI Co-reasoning for Clinical Diagnosis with Evidence-Integrated Language Agent</strong> - Zhongzhen Huang, Yan Ling, Hong Chen, Ye Feng, Li Wu, Linjie Mu, Shaoting Zhang, Xiaofan Zhang, Kun Qian, Xiaomu Li - [[pdf]](https://arxiv.org/pdf/2603.10492)</summary>

**Abstract:** We present PULSE, a medical reasoning agent that combines a domain-tuned large language model with scientific literature retrieval to support diagnostic decision-making in complex real-world cases. To evaluate its capabilities, we curated a benchmark of 82 authentic endocrinology case reports encompassing a broad spectrum of disease types and incidence levels. In controlled experiments, we compared PULSE's performance against physicians with varying levels of expertise-from residents to senior specialists-and examined how AI assistance influenced human diagnostic reasoning. PULSE attained expert-competitive accuracy, outperforming residents and junior specialists while matching senior specialist performance at both Top@1 and Top@4 thresholds. Unlike physicians, whose accuracy declined with disease rarity, PULSE maintained stable performance across incidence tiers. The agent also exhibited adaptive reasoning, increasing output length with case difficulty in a manner analogous to the longer deliberation observed among expert clinicians. When used collaboratively, PULSE enabled physicians to correct initial errors and broaden diagnostic hypotheses, but also introduced risks of automation bias. The study explores both serial and concurrent collaboration workflows, revealing that PULSE offers robust support across common and rare presentations. These findings underscore both the promise and the limitations of language model-based agents in clinical diagnosis, and offer a framework for evaluating their role in real-world decision-making.

**arXiv ID:** 2603.10492
</details>

<details>
<summary><strong>Improving Search Agent with One Line of Code</strong> - Jian Li, Dongsheng Chen, Zhenhua Xu, Yizhang Jin, Jiafu Wu, Chengjie Wang, Xiaotong Yuan, Yabiao Wang - [[pdf]](https://arxiv.org/pdf/2603.10069)</summary>

**Abstract:** Tool-based Agentic Reinforcement Learning (TARL) has emerged as a promising paradigm for training search agents to interact with external tools for a multi-turn information-seeking process autonomously. However, we identify a critical training instability that leads to catastrophic model collapse: Importance Sampling Distribution Drift(ISDD). In Group Relative Policy Optimization(GRPO), a widely adopted TARL algorithm, ISDD manifests as a precipitous decline in the importance sampling ratios, which nullifies gradient updates and triggers irreversible training failure. To address this, we propose \textbf{S}earch \textbf{A}gent \textbf{P}olicy \textbf{O}ptimization (\textbf{SAPO}), which stabilizes training via a conditional token-level KL constraint. Unlike hard clipping, which ignores distributional divergence, SAPO selectively penalizes the KL divergence between the current and old policies. Crucially, this penalty is applied only to positive tokens with low probabilities where the policy has shifted excessively, thereby preventing distribution drift while preserving gradient flow. Remarkably, SAPO requires only one-line code modification to standard GRPO, ensuring immediate deployability. Extensive experiments across seven QA benchmarks demonstrate that SAPO achieves \textbf{+10.6\% absolute improvement} (+31.5\% relative) over Search-R1, yielding consistent gains across varying model scales (1.5B, 14B) and families (Qwen, LLaMA).

**arXiv ID:** 2603.10069
</details>

<details>
<summary><strong>Tackling Length Inflation Without Trade-offs: Group Relative Reward Rescaling for Reinforcement Learning</strong> - Zichao Li, Jie Lou, Fangchen Dong, Zhiyuan Fan, Mengjie Ren, Hongyu Lin, Xianpei Han, Debing Zhang, Le Sun, Yaojie Lu, Xing Yu - [[pdf]](https://arxiv.org/pdf/2603.10535)</summary>

**Abstract:** Reinforcement learning significantly enhances LLM capabilities but suffers from a critical issue: length inflation, where models adopt verbosity or inefficient reasoning to maximize rewards. Prior approaches struggle to address this challenge in a general and lossless manner, primarily because additive penalties introduce a compensatory effect that creates optimization shortcuts, while heuristic gating strategies lack generality beyond binary feedback. To bridge this gap, we present Group Relative Reward Rescaling (GR$^3$), which reframes length control as a multiplicative rescaling paradigm, effectively establishing a generalized, continuous, and reward-dependent gating mechanism. To further ensure lossless optimization, we incorporate group-relative regularization and advantage-aware calibration, which dynamically adapt length budgets to instance difficulty and preserve the advantage signal of high-quality trajectories. Empirically, across both RLHF and RLVR settings, GR$^3$~maintains training dynamics and downstream performance comparable to standard GRPO while significantly mitigating length inflation, outperforming state-of-the-art length-regularized baselines.

**arXiv ID:** 2603.10535
</details>

<details>
<summary><strong>SAGE: A Top-Down Bottom-Up Knowledge-Grounded User Simulator for Multi-turn AGent Evaluation</strong> - Ryan Shea, Yunan Lu, Liang Qiu, Zhou Yu - [[pdf]](https://arxiv.org/pdf/2510.11997)</summary>

**Abstract:** Evaluating multi-turn interactive agents is challenging due to the need for human assessment. Evaluation with simulated users has been introduced as an alternative, however existing approaches typically model generic users and overlook the domain-specific principles required to capture realistic behavior. We propose SAGE, a novel user Simulation framework for multi-turn AGent Evaluation that integrates knowledge from business contexts. SAGE incorporates top-down knowledge rooted in business logic, such as ideal customer profiles, grounding user behavior in realistic customer personas. We further integrate bottom-up knowledge taken from business agent infrastructure (e.g., product catalogs, FAQs, and knowledge bases), allowing the simulator to generate interactions that reflect users' information needs and expectations in a company's target market. Through empirical evaluation, we find that this approach produces interactions that are more realistic and diverse, while also identifying up to 33% more agent errors, highlighting its effectiveness as an evaluation tool to support bug-finding and iterative agent improvement.

**arXiv ID:** 2510.11997
</details>

<details>
<summary><strong>Cluster-Aware Attention-Based Deep Reinforcement Learning for Pickup and Delivery Problems</strong> - Wentao Wang, Lifeng Han, Guangyu Zou - [[pdf]](https://arxiv.org/pdf/2603.10053)</summary>

**Abstract:** The Pickup and Delivery Problem (PDP) is a fundamental and challenging variant of the Vehicle Routing Problem, characterized by tightly coupled pickup--delivery pairs, precedence constraints, and spatial layouts that often exhibit clustering. Existing deep reinforcement learning (DRL) approaches either model all nodes on a flat graph, relying on implicit learning to enforce constraints, or achieve strong performance through inference-time collaborative search at the cost of substantial latency. In this paper, we propose \emph{CAADRL} (Cluster-Aware Attention-based Deep Reinforcement Learning), a DRL framework that explicitly exploits the multi-scale structure of PDP instances via cluster-aware encoding and hierarchical decoding. The encoder builds on a Transformer and combines global self-attention with intra-cluster attention over depot, pickup, and delivery nodes, producing embeddings that are both globally informative and locally role-aware. Based on these embeddings, we introduce a Dynamic Dual-Decoder with a learnable gate that balances intra-cluster routing and inter-cluster transitions at each step. The policy is trained end-to-end with a POMO-style policy gradient scheme using multiple symmetric rollouts per instance. Experiments on synthetic clustered and uniform PDP benchmarks show that CAADRL matches or improves upon strong state-of-the-art baselines on clustered instances and remains highly competitive on uniform instances, particularly as problem size increases. Crucially, our method achieves these results with substantially lower inference time than neural collaborative-search baselines, suggesting that explicitly modeling cluster structure provides an effective and efficient inductive bias for neural PDP solvers.

**arXiv ID:** 2603.10053
</details>

<details>
<summary><strong>Actor-Accelerated Policy Dual Averaging for Reinforcement Learning in Continuous Action Spaces</strong> - Ji Gao, Caleb Ju, Guanghui Lan, Zhaohui Tong - [[pdf]](https://arxiv.org/pdf/2603.10199)</summary>

**Abstract:** Policy Dual Averaging (PDA) offers a principled Policy Mirror Descent (PMD) framework that more naturally admits value function approximation than standard PMD, enabling the use of approximate advantage (or Q-) functions while retaining strong convergence guarantees. However, applying PDA in continuous state and action spaces remains computationally challenging, since action selection involves solving an optimization sub-problem at each decision step. In this paper, we propose \textit{actor-accelerated PDA}, which uses a learned policy network to approximate the solution of the optimization sub-problems, yielding faster runtimes while maintaining convergence guarantees. We provide a theoretical analysis that quantifies how actor approximation error impacts the convergence of PDA under suitable assumptions. We then evaluate its performance on several benchmarks in robotics, control, and operations research problems. Actor-accelerated PDA achieves superior performance compared to popular on-policy baselines such as Proximal Policy Optimization (PPO). Overall, our results bridge the gap between the theoretical advantages of PDA and its practical deployment in continuous-action problems with function approximation.

**arXiv ID:** 2603.10199
</details>

<details>
<summary><strong>SiMPO: Measure Matching for Online Diffusion Reinforcement Learning</strong> - Haitong Ma, Chenxiao Gao, Tianyi Chen, Na Li, Bo Dai - [[pdf]](https://arxiv.org/pdf/2603.10250)</summary>

**Abstract:** A commonly used family of RL algorithms for diffusion policies conducts softmax reweighting over the behavior policy, which usually induces an over-greedy policy and fails to leverage feedback from negative samples. In this work, we introduce Signed Measure Policy Optimization (SiMPO), a simple and unified framework that generalizes reweighting scheme in diffusion RL with general monotonic functions. SiMPO revisits diffusion RL via a two-stage measure matching lens. First, we construct a virtual target policy by $f$-divergence regularized policy optimization, where we can relax the non-negativity constraint to allow for a signed target measure. Second, we use this signed measure to guide diffusion or flow models through reweighted matching. This formulation offers two key advantages: a) it generalizes to arbitrary monotonically increasing weighting functions; and b) it provides a principled justification and practical guidance for negative reweighting. Furthermore, we provide geometric interpretations to illustrate how negative reweighting actively repels the policy from suboptimal actions. Extensive empirical evaluations demonstrate that SiMPO achieves superior performance by leveraging these flexible weighting schemes, and we provide practical guidelines for selecting reweighting methods tailored to the reward landscape.

**arXiv ID:** 2603.10250
</details>

<details>
<summary><strong>Graph-GRPO: Training Graph Flow Models with Reinforcement Learning</strong> - Baoheng Zhu, Deyu Bo, Delvin Ce Zhang, Xiao Wang - [[pdf]](https://arxiv.org/pdf/2603.10395)</summary>

**Abstract:** Graph generation is a fundamental task with broad applications, such as drug discovery. Recently, discrete flow matching-based graph generation, \aka, graph flow model (GFM), has emerged due to its superior performance and flexible sampling. However, effectively aligning GFMs with complex human preferences or task-specific objectives remains a significant challenge. In this paper, we propose Graph-GRPO, an online reinforcement learning (RL) framework for training GFMs under verifiable rewards. Our method makes two key contributions: (1) We derive an analytical expression for the transition probability of GFMs, replacing the Monte Carlo sampling and enabling fully differentiable rollouts for RL training; (2) We propose a refinement strategy that randomly perturbs specific nodes and edges in a graph, and regenerates them, allowing for localized exploration and self-improvement of generation quality. Extensive experiments on both synthetic and real datasets demonstrate the effectiveness of Graph-GRPO. With only 50 denoising steps, our method achieves 95.0\% and 97.5\% Valid-Unique-Novelty scores on the planar and tree datasets, respectively. Moreover, Graph-GRPO achieves state-of-the-art performance on the molecular optimization tasks, outperforming graph-based and fragment-based RL methods as well as classic genetic algorithms.

**arXiv ID:** 2603.10395
</details>

<details>
<summary><strong>Learning to Score: Tuning Cluster Schedulers through Reinforcement Learning</strong> - Martin Asenov, Qiwen Deng, Gingfung Yeung, Adam Barker - [[pdf]](https://arxiv.org/pdf/2603.10545)</summary>

**Abstract:** Efficiently allocating incoming jobs to nodes in large-scale clusters can lead to substantial improvements in both cluster utilization and job performance. In order to allocate incoming jobs, cluster schedulers usually rely on a set of scoring functions to rank feasible nodes. Results from individual scoring functions are usually weighted equally, which could lead to sub-optimal deployments as the one-size-fits-all solution does not take into account the characteristics of each workload. Tuning the weights of scoring functions, however, requires expert knowledge and is computationally expensive.
This paper proposes a reinforcement learning approach for learning the weights in scheduler scoring algorithms with the overall objective of improving the end-to-end performance of jobs for a given cluster. Our approach is based on percentage improvement reward, frame-stacking, and limiting domain information. We propose a percentage improvement reward to address the objective of multi-step parameter tuning. The inclusion of frame-stacking allows for carrying information across an optimization experiment. Limiting domain information prevents overfitting and improves performance in unseen clusters and workloads. The policy is trained on different combinations of workloads and cluster setups. We demonstrate the proposed approach improves performance on average by 33\% compared to fixed weights and 12\% compared to the best-performing baseline in a lab-based serverless scenario.

**arXiv ID:** 2603.10545
</details>

<details>
<summary><strong>Ergodicity in reinforcement learning</strong> - Dominik Baumann, Erfaun Noorani, Arsenii Mustafin, Xinyi Sheng, Bert Verbruggen, Arne Vanhoyweghen, Vincent Ginis, Thomas B. Schön - [[pdf]](https://arxiv.org/pdf/2603.10895)</summary>

**Abstract:** In reinforcement learning, we typically aim to optimize the expected value of the sum of rewards an agent collects over a trajectory. However, if the process generating these rewards is non-ergodic, the expected value, i.e., the average over infinitely many trajectories with a given policy, is uninformative for the average over a single, but infinitely long trajectory. Thus, if we care about how the individual agent performs during deployment, the expected value is not a good optimization objective. In this paper, we discuss the impact of non-ergodic reward processes on reinforcement learning agents through an instructive example, relate the notion of ergodic reward processes to more widely used notions of ergodic Markov chains, and present existing solutions that optimize long-term performance of individual trajectories under non-ergodic reward dynamics.

**arXiv ID:** 2603.10895
</details>

<details>
<summary><strong>Adaptive Active Learning for Regression via Reinforcement Learning</strong> - Simon D. Nguyen, Troy Russo, Kentaro Hoffman, Tyler H. McCormick - [[pdf]](https://arxiv.org/pdf/2603.10435)</summary>

**Abstract:** Active learning for regression reduces labeling costs by selecting the most informative samples. Improved Greedy Sampling is a prominent method that balances feature-space diversity and output-space uncertainty using a static, multiplicative rule. We propose Weighted improved Greedy Sampling (WiGS), which replaces this framework with a dynamic, additive criterion. We formulate weight selection as a reinforcement learning problem, enabling an agent to adapt the exploration-investigation balance throughout learning. Experiments on 18 benchmark datasets and a synthetic environment show WiGS outperforms iGS and other baseline methods in both accuracy and labeling efficiency, particularly in domains with irregular data density where the baseline's multiplicative rule ignores high-error samples in dense regions.

**arXiv ID:** 2603.10435
</details>

<details>
<summary><strong>ReTabSyn: Realistic Tabular Data Synthesis via Reinforcement Learning</strong> - Xiaofeng Lin, Seungbae Kim, Zhuoya Li, Zachary DeSoto, Charles Fleming, Guang Cheng - [[pdf]](https://arxiv.org/pdf/2603.10823)</summary>

**Abstract:** Deep generative models can help with data scarcity and privacy by producing synthetic training data, but they struggle in low-data, imbalanced tabular settings to fully learn the complex data distribution. We argue that striving for the full joint distribution could be overkill; for greater data efficiency, models should prioritize learning the conditional distribution $P(y\mid \bm{X})$, as suggested by recent theoretical analysis. Therefore, we overcome this limitation with \textbf{ReTabSyn}, a \textbf{Re}inforced \textbf{Tab}ular \textbf{Syn}thesis pipeline that provides direct feedback on feature correlation preservation during synthesizer training. This objective encourages the generator to prioritize the most useful predictive signals when training data is limited, thereby strengthening downstream model utility. We empirically fine-tune a language model-based generator using this approach, and across benchmarks with small sample sizes, class imbalance, and distribution shift, ReTabSyn consistently outperforms state-of-the-art baselines. Moreover, our approach can be readily extended to control various aspects of synthetic tabular data, such as applying expert-specified constraints on generated observations.

**arXiv ID:** 2603.10823
</details>

<details>
<summary><strong>Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments</strong> - Junwoo Chang, Minwoo Park, Joohwan Seo, Roberto Horowitz, Jongmin Lee, Jongeun Choi - [[pdf]](https://arxiv.org/pdf/2512.00915)</summary>

**Abstract:** Group symmetries provide a powerful inductive bias for reinforcement learning (RL), enabling efficient generalization across symmetric states and actions via group-invariant Markov Decision Processes (MDPs). However, real-world environments almost never realize fully group-invariant MDPs; dynamics, actuation limits, and reward design usually break symmetries, often only locally. Under group-invariant Bellman backups for such cases, local symmetry-breaking introduces errors that propagate across the entire state-action space, resulting in global value estimation errors. To address this, we introduce Partially group-Invariant MDP (PI-MDP), which selectively applies group-invariant or standard Bellman backups depending on where symmetry holds. This framework mitigates error propagation from locally broken symmetries while maintaining the benefits of equivariance, thereby enhancing sample efficiency and generalizability. Building on this framework, we present practical RL algorithms -- Partially Equivariant (PE)-DQN for discrete control and PE-SAC for continuous control -- that combine the benefits of equivariance with robustness to symmetry-breaking. Experiments across Grid-World, locomotion, and manipulation benchmarks demonstrate that PE-DQN and PE-SAC significantly outperform baselines, highlighting the importance of selective symmetry exploitation for robust and sample-efficient RL. Project page: this https URL

**arXiv ID:** 2512.00915
</details>

<details>
<summary><strong>Position: Beyond Model-Centric Prediction -- Agentic Time Series Forecasting</strong> - Mingyue Cheng, Xiaoyu Tao, Qi Liu, Ze Guo, Enhong Chen - [[pdf]](https://arxiv.org/pdf/2602.01776)</summary>

**Abstract:** Time series forecasting has traditionally been formulated as a model-centric, static, and single-pass prediction problem that maps historical observations to future values. While this paradigm has driven substantial progress, it proves insufficient in adaptive and multi-turn settings where forecasting requires informative feature extraction, reasoning-driven inference, iterative refinement, and continual adaptation over time. In this paper, we argue for agentic time series forecasting (ATSF), which reframes forecasting as an agentic process composed of perception, planning, action, reflection, and memory. Rather than focusing solely on predictive models, ATSF emphasizes organizing forecasting as an agentic workflow that can interact with tools, incorporate feedback from outcomes, and evolve through experience accumulation. We outline three representative implementation paradigms -- workflow-based design, agentic reinforcement learning, and a hybrid agentic workflow paradigm -- and discuss the opportunities and challenges that arise when shifting from model-centric prediction to agentic forecasting. Together, this position aims to establish agentic forecasting as a foundation for future research at the intersection of time series forecasting.

**arXiv ID:** 2602.01776
</details>

<details>
<summary><strong>LexiSafe: Offline Safe Reinforcement Learning with Lexicographic Safety-Reward Hierarchy</strong> - Hsin-Jung Yang, Zhanhong Jiang, Prajwal Koirala, Qisai Liu, Cody Fleming, Soumik Sarkar - [[pdf]](https://arxiv.org/pdf/2602.17312)</summary>

**Abstract:** Offline safe reinforcement learning (RL) is increasingly important for cyber-physical systems (CPS), where safety violations during training are unacceptable and only pre-collected data are available. Existing offline safe RL methods typically balance reward-safety tradeoffs through constraint relaxation or joint optimization, but they often lack structural mechanisms to prevent safety drift. We propose LexiSafe, a lexicographic offline RL framework designed to preserve safety-aligned behavior. We first develop LexiSafe-SC, a single-cost formulation for standard offline safe RL, and derive safety-violation and performance-suboptimality bounds that together yield sample-complexity guarantees. We then extend the framework to hierarchical safety requirements with LexiSafe-MC, which supports multiple safety costs and admits its own sample-complexity analysis. Empirically, LexiSafe demonstrates reduced safety violations and improved task performance compared to constrained offline baselines. By unifying lexicographic prioritization with structural bias, LexiSafe offers a practical and theoretically grounded approach for safety-critical CPS decision-making.

**arXiv ID:** 2602.17312
</details>

<details>
<summary><strong>SteadyTray: Learning Object Balancing Tasks in Humanoid Tray Transport via Residual Reinforcement Learning</strong> - Anlun Huang, Zhenyu Wu, Soofiyan Atar, Yuheng Zhi, Michael Yip - [[pdf]](https://arxiv.org/pdf/2603.10306)</summary>

**Abstract:** Stabilizing unsecured payloads against the inherent oscillations of dynamic bipedal locomotion remains a critical engineering bottleneck for humanoids in unstructured environments. To solve this, we introduce ReST-RL, a hierarchical reinforcement learning architecture that explicitly decouples locomotion from payload stabilization, evaluated via the SteadyTray benchmark. Rather than relying on monolithic end-to-end learning, our framework integrates a robust base locomotion policy with a dynamic residual module engineered to actively cancel gait-induced perturbations at the end-effector. This architectural separation ensures steady tray transport without degrading the underlying bipedal stability. In simulation, the residual design significantly outperforms end-to-end baselines in gait smoothness and orientation accuracy, achieving a 96.9% success rate in variable velocity tracking and 74.5% robustness against external force disturbances. Successfully deployed on the Unitree G1 humanoid hardware, this modular approach demonstrates highly reliable zero-shot sim-to-real generalization across various objects and external force disturbances.

**arXiv ID:** 2603.10306
</details>

<details>
<summary><strong>MAVEN: A Meta-Reinforcement Learning Framework for Varying-Dynamics Expertise in Agile Quadrotor Maneuvers</strong> - Jin Zhou, Dongcheng Cao, Xian Wang, Shuo Li - [[pdf]](https://arxiv.org/pdf/2603.10714)</summary>

**Abstract:** Reinforcement learning (RL) has emerged as a powerful paradigm for achieving online agile navigation with quadrotors. Despite this success, policies trained via standard RL typically fail to generalize across significant dynamic variations, exhibiting a critical lack of adaptability. This work introduces MAVEN, a meta-RL framework that enables a single policy to achieve robust end-to-end navigation across a wide range of quadrotor dynamics. Our approach features a novel predictive context encoder, which learns to infer a latent representation of the system dynamics from interaction history. We demonstrate our method in agile waypoint traversal tasks under two challenging scenarios: large variations in quadrotor mass and severe single-rotor thrust loss. We leverage a GPU-vectorized simulator to distribute tasks across thousands of parallel environments, overcoming the long training times of meta-RL to converge in less than an hour. Through extensive experiments in both simulation and the real world, we validate that MAVEN achieves superior adaptation and agility. The policy successfully executes zero-shot sim-to-real transfer, demonstrating robust online adaptation by performing high-speed maneuvers despite mass variations of up to 66.7% and single-rotor thrust losses as severe as 70%.

**arXiv ID:** 2603.10714
</details>

<details>
<summary><strong>ASTER: Attitude-aware Suspended-payload Quadrotor Traversal via Efficient Reinforcement Learning</strong> - Dongcheng Cao, Jin Zhou, Shuo Li - [[pdf]](https://arxiv.org/pdf/2603.10715)</summary>

**Abstract:** Agile maneuvering of the quadrotor cable-suspended system is significantly hindered by its non-smooth hybrid dynamics. While model-free Reinforcement Learning (RL) circumvents explicit differentiation of complex models, achieving attitude-constrained or inverted flight remains an open challenge due to the extreme reward sparsity under strict orientation requirements. This paper presents ASTER, a robust RL framework that achieves, to our knowledge, the first successful autonomous inverted flight for the cable-suspended system. We propose hybrid-dynamics-informed state seeding (HDSS), an initialization strategy that back-propagates target configurations through physics-consistent kinematic inversions across both taut and slack cable phases. HDSS enables the policy to discover aggressive maneuvers that are unreachable via standard exploration. Extensive simulations and real-world experiments demonstrate remarkable agility, precise attitude alignment, and robust zero-shot sim-to-real transfer across complex trajectories.

**arXiv ID:** 2603.10715
</details>

<details>
<summary><strong>Cross-embodied Co-design for Dexterous Hands</strong> - Kehlani Fay, Darin Anthony Djapri, Anya Zorin, James Clinton, Ali El Lahib, Hao Su, Michael T. Tolley, Sha Yi, Xiaolong Wang - [[pdf]](https://arxiv.org/pdf/2512.03743)</summary>

**Abstract:** Dexterous manipulation is limited by both control and design, without consensus as to what makes manipulators best for performing dexterous tasks. This raises a fundamental challenge: how should we design and control robot manipulators that are optimized for dexterity? We present a co-design framework that learns task-specific hand morphology and complementary dexterous control policies. The framework supports 1) an expansive morphology search space including joint, finger, and palm generation, 2) scalable evaluation across the wide design space via morphology-conditioned cross-embodied control, and 3) real-world fabrication with accessible components. We evaluate the approach across multiple dexterous tasks, including in-hand rotation with simulation and real deployment. Our framework enables an end-to-end pipeline that can design, train, fabricate, and deploy a new robotic hand in under 24 hours. The full framework will be open-sourced and available on our website: this https URL .

**arXiv ID:** 2512.03743
</details>

<details>
<summary><strong>Global End-Effector Pose Control of an Underactuated Aerial Manipulator via Reinforcement Learning</strong> - Shlok Deshmukh, Javier Alonso-Mora, Sihao Sun - [[pdf]](https://arxiv.org/pdf/2512.21085)</summary>

**Abstract:** Aerial manipulators, which combine robotic arms with multi-rotor drones, face strict constraints on arm weight and mechanical complexity. In this work, we study a lightweight 2-degree-of-freedom (DoF) arm mounted on a quadrotor via a differential mechanism, capable of full six-DoF end-effector pose control. While the minimal design enables simplicity and reduced payload, it also introduces challenges such as underactuation and sensitivity to external disturbances. To address these, we employ reinforcement learning, training a Proximal Policy Optimization (PPO) agent in simulation to generate feedforward commands for quadrotor acceleration and body rates, along with joint angle targets. These commands are tracked by an incremental nonlinear dynamic inversion (INDI) attitude controller and a PID joint controller, respectively. Flight experiments demonstrate centimeter-level position accuracy and degree-level orientation precision, with robust performance under external force disturbances, including manipulation of heavy loads and pushing tasks. The results highlight the potential of learning-based control strategies for enabling contact-rich aerial manipulation using simple, lightweight platforms. Videos of the experiment and the method are summarized in this https URL.

**arXiv ID:** 2512.21085
</details>

<details>
<summary><strong>PlayWorld: Learning Robot World Models from Autonomous Play</strong> - Tenny Yin, Zhiting Mei, Zhonghe Zheng, Miyu Yamane, David Wang, Jade Sceats, Samuel M. Bateman, Lihan Zha, Apurva Badithela, Ola Shorinwa, Anirudha Majumdar - [[pdf]](https://arxiv.org/pdf/2603.09030)</summary>

**Abstract:** Action-conditioned video models offer a promising path to building general-purpose robot simulators that can improve directly from data. Yet, despite training on large-scale robot datasets, current state-of-the-art video models still struggle to predict physically consistent robot-object interactions that are crucial in robotic manipulation. To close this gap, we present PlayWorld, a simple, scalable, and fully autonomous pipeline for training high-fidelity video world simulators from interaction experience. In contrast to prior approaches that rely on success-biased human demonstrations, PlayWorld is the first system capable of learning entirely from unsupervised robot self-play, enabling naturally scalable data collection while capturing complex, long-tailed physical interactions essential for modeling realistic object dynamics. Experiments across diverse manipulation tasks show that PlayWorld generates high-quality, physically consistent predictions for contact-rich interactions that are not captured by world models trained on human-collected data. We further demonstrate the versatility of PlayWorld in enabling fine-grained failure prediction and policy evaluation, with up to 40% improvements over human-collected data. Finally, we demonstrate how PlayWorld enables reinforcement learning in the world model, improving policy performance by 65% in success rates when deployed in the real world.

**arXiv ID:** 2603.09030
</details>

<details>
<summary><strong>EyeAgent: An Agentic AI System for Multimodal Clinical Decision Support in Ophthalmology</strong> - Danli Shi, Xiaolan Chen, Bingjie Yan, Weiyi Zhang, Pusheng Xu, Jiancheng Yang, Ruoyu Chen, Siyu Huang, Bowen Liu, Xinyuan Wu, Meng Xie, Ziyu Gao, Yue Wu, Senlin Lin, Kai Jin, Xia Gong, Yih Chung Tham, Xiujuan Zhang, Li Dong, Yuzhou Zhang, Jason Yam, Guangming Jin, Xiaohu Ding, Haidong Zou, Yalin Zheng, Zongyuan Ge, Mingguang He - [[pdf]](https://arxiv.org/pdf/2511.09394)</summary>

**Abstract:** Artificial intelligence has shown promise in medical imaging, yet most existing systems lack flexibility, interpretability, and adaptability - challenges especially pronounced in ophthalmology, where diverse imaging modalities are essential. We present EyeAgent, the first agentic AI framework for comprehensive and interpretable clinical decision support in ophthalmology. Using a large language model (DeepSeek-V3) as its central reasoning engine, EyeAgent interprets user queries and dynamically orchestrates 53 validated ophthalmic tools across 23 imaging modalities for diverse tasks including classification, segmentation, detection, image/report generation, and quantitative analysis. Stepwise ablation analysis demonstrated a progressive improvement in diagnostic accuracy, rising from a baseline of 69.71% (using only 5 general tools) to 80.79% when the full suite of 53 specialized tools was integrated. In an expert rating study on 200 real-world clinical cases, EyeAgent achieved 93.7% tool selection accuracy and received expert ratings of more than 88% across accuracy, completeness, safety, reasoning, and interpretability. In human-AI collaboration, EyeAgent matched or exceeded the performance of senior ophthalmologists and, when used as an assistant, improved overall diagnostic accuracy by 18.51% and report quality scores by 19%, with the greatest benefit observed among junior ophthalmologists. These findings establish EyeAgent as a scalable and trustworthy AI framework for ophthalmology and provide a blueprint for modular, multimodal, and clinically aligned next-generation AI systems.

**arXiv ID:** 2511.09394
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
