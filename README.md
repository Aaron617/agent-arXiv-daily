# Agent arXiv Daily

**Last Updated:** 2026-03-19 03:49:10

**Total Papers:** 110

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (7 papers)</h2></summary>

<details>
<summary><strong>PaperScout: An Autonomous Agent for Academic Paper Search with Process-Aware Sequence-Level Policy Optimization</strong> - Tingyue Pan, Jie Ouyang, Mingyue Cheng, Qingchuan Li, Zirui Liu, Daoyu Wang, Mingfan Pan, Shuo Yu, Qi Liu - [[pdf]](https://arxiv.org/pdf/2601.10029)</summary>

**Abstract:** Academic paper search is a fundamental task in scientific research, yet most existing approaches rely on rigid, predefined workflows that struggle with complex, conditional queries. To address this limitation, we propose PaperScout, an autonomous agent that reformulates paper search as a sequential decision-making process. Unlike static workflows, PaperScout dynamically decides whether, when, and how to invoke search and expand tools based on accumulated retrieval context. However, training such agents presents a fundamental challenge: standard reinforcement learning methods, typically designed for single-turn tasks, suffer from a granularity mismatch when applied to multi-turn agentic tasks-where token-level optimization diverges from the granularity of sequence-level interactions-leading to noisy credit assignment and unstable training dynamics. We introduce Proximal Sequence Policy Optimization (PSPO), a process-aware, sequence-level policy optimization method that aligns optimization with agent--environment interaction. Comprehensive experiments on both synthetic and real-world benchmarks demonstrate that PaperScout significantly outperforms strong workflow-driven and RL baselines in both recall and relevance, validating the effectiveness of our adaptive agentic framework and optimization strategy.

**arXiv ID:** 2601.10029
</details>

<details>
<summary><strong>A Hierarchical Error-Corrective Graph Framework for Autonomous Agents with LLM-Based Action Generation</strong> - Cong Cao, Jingyao Zhang, Kun Tong - [[pdf]](https://arxiv.org/pdf/2603.08388)</summary>

**Abstract:** We propose a Hierarchical Error-Corrective Graph FrameworkforAutonomousAgentswithLLM-BasedActionGeneration(HECG),whichincorporates three core innovations: (1) Multi-Dimensional Transferable Strategy (MDTS): by integrating task quality metrics (Q), confidence/cost metrics (C), reward metrics (R), and LLM-based semantic reasoning scores (LLM-Score), MDTS achieves multi-dimensional alignment between quantitative performance and semantic context, enabling more precise selection of high-quality candidate strate gies and effectively reducing the risk of negative transfer. (2) Error Matrix Classification (EMC): unlike simple confusion matrices or overall performance metrics, EMC provides structured attribution of task failures by categorizing errors into ten types, such as Strategy Errors (Strategy Whe) and Script Parsing Errors (Script-Parsing-Error), and decomposing them according to severity, typical actions, error descriptions, and recoverability. This allows precise analysis of the root causes of task failures, offering clear guidance for subsequent error correction and strategy optimization rather than relying solely on overall success rates or single performance metrics. (3) Causal-Context Graph Retrieval (CCGR): to enhance agent retrieval capabilities in dynamic task environments, we construct graphs from historical states, actions, and event sequences, where nodes store executed actions, next-step actions, execution states, transferable strategies, and other relevant information, and edges represent causal dependencies such as preconditions for transitions between nodes. CCGR identifies subgraphs most relevant to the current task context, effectively capturing structural relationships beyond vector similarity, allowing agents to fully leverage contextual information, accelerate strategy adaptation, and improve execution reliability in complex, multi-step tasks.

**arXiv ID:** 2603.08388
</details>

<details>
<summary><strong>TRUST-SQL: Tool-Integrated Multi-Turn Reinforcement Learning for Text-to-SQL over Unknown Schemas</strong> - Ai Jian, Xiaoyun Zhang, Wanrou Du, Jingqing Ruan, Jiangbo Pei, Weipeng Zhang, Ke Zeng, Xunliang Cai - [[pdf]](https://arxiv.org/pdf/2603.16448)</summary>

**Abstract:** Text-to-SQL parsing has achieved remarkable progress under the Full Schema Assumption. However, this premise fails in real-world enterprise environments where databases contain hundreds of tables with massive noisy metadata. Rather than injecting the full schema upfront, an agent must actively identify and verify only the relevant subset, giving rise to the Unknown Schema scenario we study in this work. To address this, we propose TRUST-SQL (Truthful Reasoning with Unknown Schema via Tools). We formulate the task as a Partially Observable Markov Decision Process where our autonomous agent employs a structured four-phase protocol to ground reasoning in verified metadata. Crucially, this protocol provides a structural boundary for our novel Dual-Track GRPO strategy. By applying token-level masked advantages, this strategy isolates exploration rewards from execution outcomes to resolve credit assignment, yielding a 9.9% relative improvement over standard GRPO. Extensive experiments across five benchmarks demonstrate that TRUST-SQL achieves an average absolute improvement of 30.6% and 16.6% for the 4B and 8B variants respectively over their base models. Remarkably, despite operating entirely without pre-loaded metadata, our framework consistently matches or surpasses strong baselines that rely on schema prefilling.

**arXiv ID:** 2603.16448
</details>

<details>
<summary><strong>Bringing Network Coding into Multi-Robot Systems: Interplay Study for Autonomous Systems over Wireless Communications</strong> - Anil Zaher, Kiril Solovey, Alejandro Cohen - [[pdf]](https://arxiv.org/pdf/2603.17472)</summary>

**Abstract:** Communication is a core enabler for multi-robot systems (MRS), providing the mechanism through which robots exchange state information, coordinate actions, and satisfy safety constraints. While many MRS autonomy algorithms assume reliable and timely message delivery, realistic wireless channels introduce delay, erasures, and ordering stalls that can degrade performance and compromise safety-critical decisions of the robot task. In this paper, we investigate how transport-layer reliability mechanisms that mitigate communication losses and delays shape the autonomy-communication loop. We show that conventional non-coded retransmission-based protocols introduce long delays that are misaligned with the timeliness requirements of MRS applications, and may render the received data irrelevant. As an alternative, we advocate for adaptive and causal network coding, which proactively injects coded redundancy to achieve the desired delay and throughput that enable relevant data delivery to the robotic task. Specifically, this method adapts to channel conditions between robots and causally tunes the communication rates via efficient algorithms.
We present two case studies: cooperative localization under delayed and lossy inter-robot communication, and a safety-critical overtaking maneuver where timely vehicle-to-vehicle message availability determines whether an ego vehicle can abort to avoid a crash. Our results demonstrate that coding-based communication significantly reduces in-order delivery stalls, preserves estimation consistency under delay, and improves deadline reliability relative to retransmission-based transport. Overall, the study highlights the need to jointly design autonomy algorithms and communication mechanisms, and positions network coding as a principled tool for dependable multi-robot operation over wireless networks.

**arXiv ID:** 2603.17472
</details>

<details>
<summary><strong>From Slides to Chatbots: Enhancing Large Language Models with University Course Materials</strong> - Tu Anh Dinh, Philipp Nicolas Schumacher, Jan Niehues - [[pdf]](https://arxiv.org/pdf/2510.22272)</summary>

**Abstract:** Large Language Models (LLMs) have advanced rapidly in recent years. One application of LLMs is to support student learning in educational settings. However, prior work has shown that LLMs still struggle to answer questions accurately within university-level computer science courses. In this work, we investigate how incorporating university course materials can enhance LLM performance in this setting. A key challenge lies in leveraging diverse course materials such as lecture slides and transcripts, which differ substantially from typical textual corpora: slides also contain visual elements like images and formulas, while transcripts contain spoken, less structured language. We compare two strategies, Retrieval-Augmented Generation (RAG) and Continual Pre-Training (CPT), to extend LLMs with course-specific knowledge. For lecture slides, we further explore a multi-modal RAG approach, where we present the retrieved content to the generator in image form. Our experiments reveal that, given the relatively small size of university course materials, RAG is more effective and efficient than CPT. Moreover, incorporating slides as images in the multi-modal setting significantly improves performance over text-only retrieval. These findings highlight practical strategies for developing AI assistants that better support learning and teaching, and we hope they inspire similar efforts in other educational contexts.

**arXiv ID:** 2510.22272
</details>

<details>
<summary><strong>SAATT Nav: a Socially Aware Autonomous Transparent Transportation Navigation Framework for Wheelchairs</strong> - Yutong Zhang, Shaiv Y. Mehra, Bradley S. Duerstock, Juan P. Wachs - [[pdf]](https://arxiv.org/pdf/2603.13698)</summary>

**Abstract:** While powered wheelchairs reduce physical fatigue as opposed to manual wheelchairs for individuals with mobility impairment, they demand high cognitive workload due to information processing, decision making and motor coordination. Current autonomous systems lack social awareness in navigation and transparency in decision-making, leading to decreased perceived safety and trust from the user and others in context. This work proposes Socially Aware Autonomous Transparent Transportation (SAATT) Navigation framework for wheelchairs as a potential solution. By implementing a Large Language Model (LLM) informed of user intent and capable of predicting other peoples' intent as a decision-maker for its local controller, it is able to detect and navigate social situations, such as passing pedestrians or a pair conversing. Furthermore, the LLM textually communicates its reasoning at each waypoint for transparency. In this experiment, it is compared against a standard global planner, a representative competing social navigation model, and an Ablation study in three simulated environments varied by social levels in eight metrics categorized under Safety, Social Compliance, Efficiency, and Comfort. Overall, SAATT Nav outperforms in most social situations and equivalently or only slightly worse in the remaining metrics, demonstrating the potential of a socially aware and transparent autonomous navigation system to assist wheelchair users.

**arXiv ID:** 2603.13698
</details>

<details>
<summary><strong>Large Language Models in Teaching and Learning: Reflections on Implementing an AI Chatbot in Higher Education</strong> - Fiammetta Caccavale, Carina L. Gargalo, Julian Kager, Magdalena Skowyra, Steen Larsen, Krist V. Gernaey, Ulrich Krühne - [[pdf]](https://arxiv.org/pdf/2603.17773)</summary>

**Abstract:** The landscape of education is changing rapidly, shaped by emerging pedagogical approaches, technological innovations such as artificial intelligence (AI), and evolving societal expectations, all of which demand thorough evaluation of new educational tools. Although large language models (LLMs) present substantial opportunities especially in Higher Education, their propensity to generate hallucinations and their limited specialized knowledge may introduce significant risks. This study aims to address these risks by examining the practical implementation of an LLM-enhanced assistant in a university level course.
We implemented a generative AI assistant grounded in a retrieval-augmented generation (RAG) model to replicate a previously teacher-led, time-intensive exercise. To assess the effectiveness of the LLM, we conducted three separate experiments through iterative mixed-methods approaches, including a crossover design. The resulting data address central research questions related to student motivation, perceived differences between engaging with the LLM versus a human teacher, the quality of AI-generated responses, and the impact of the LLM on students' academic performance. The results offer direct insights into students' views and the pedagogical feasibility of embedding LLMs into specialized courses. Finally, we discuss the main challenges, opportunities and future directions of LLMs in teaching and learning in Higher Education.

**arXiv ID:** 2603.17773
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (22 papers)</h2></summary>

<details>
<summary><strong>Graph-Native Cognitive Memory for AI Agents: Formal Belief Revision Semantics for Versioned Memory Architectures</strong> - Young Bin Park - [[pdf]](https://arxiv.org/pdf/2603.17244)</summary>

**Abstract:** While individual components for AI agent memory exist in prior systems, their architectural synthesis and formal grounding remain underexplored. We present Kumiho, a graph-native cognitive memory architecture grounded in formal belief revision semantics. The structural primitives required for cognitive memory -- immutable revisions, mutable tag pointers, typed dependency edges, URI-based addressing -- are identical to those required for managing agent-produced work as versionable assets, enabling a unified graph-native architecture that serves both purposes. The central formal contribution is a correspondence between the AGM belief revision framework and the operational semantics of a property graph memory system, proving satisfaction of the basic AGM postulates (K*2--K*6) and Hansson's belief base postulates (Relevance, Core-Retainment). The architecture implements a dual-store model (Redis working memory, Neo4j long-term graph) with hybrid fulltext and vector retrieval. On LoCoMo (token-level F1), Kumiho achieves 0.565 overall F1 (n=1,986) including 97.5% adversarial refusal accuracy. On LoCoMo-Plus, a Level-2 cognitive memory benchmark testing implicit constraint recall, Kumiho achieves 93.3% judge accuracy (n=401); independent reproduction by the benchmark authors yielded results in the mid-80% range, still substantially outperforming all published baselines (best: Gemini 2.5 Pro, 45.7%). Three architectural innovations drive the results: prospective indexing (LLM-generated future-scenario implications indexed at write time), event extraction (structured causal events preserved in summaries), and client-side LLM reranking. The architecture is model-decoupled: switching the answer model from GPT-4o-mini (~88%) to GPT-4o (93.3%) improves end-to-end accuracy without pipeline changes, at a total evaluation cost of ~$14 for 401 entries.

**arXiv ID:** 2603.17244
</details>

<details>
<summary><strong>From Virtual Environments to Real-World Trials: Emerging Trends in Autonomous Driving</strong> - A. Humnabadkar, A. Sikdar, B. Cave, H. Zhang, N. Bessis, A. Behera - [[pdf]](https://arxiv.org/pdf/2603.17714)</summary>

**Abstract:** Autonomous driving technologies have achieved significant advances in recent years, yet their real-world deployment remains constrained by data scarcity, safety requirements, and the need for generalization across diverse environments. In response, synthetic data and virtual environments have emerged as powerful enablers, offering scalable, controllable, and richly annotated scenarios for training and evaluation. This survey presents a comprehensive review of recent developments at the intersection of autonomous driving, simulation technologies, and synthetic datasets. We organize the landscape across three core dimensions: (i) the use of synthetic data for perception and planning, (ii) digital twin-based simulation for system validation, and (iii) domain adaptation strategies bridging synthetic and real-world data. We also highlight the role of vision-language models and simulation realism in enhancing scene understanding and generalization. A detailed taxonomy of datasets, tools, and simulation platforms is provided, alongside an analysis of trends in benchmark design. Finally, we discuss critical challenges and open research directions, including Sim2Real transfer, scalable safety validation, cooperative autonomy, and simulation-driven policy learning, that must be addressed to accelerate the path toward safe, generalizable, and globally deployable autonomous driving systems.

**arXiv ID:** 2603.17714
</details>

<details>
<summary><strong>PhysQuantAgent: An Inference Pipeline of Mass Estimation for Vision-Language Models</strong> - Hisayuki Yokomizo, Taiki Miyanishi, Yan Gang, Shuhei Kurita, Nakamasa Inoue, Yusuke Iwasawa - [[pdf]](https://arxiv.org/pdf/2603.16958)</summary>

**Abstract:** Vision-Language Models (VLMs) are increasingly applied to robotic perception and manipulation, yet their ability to infer physical properties required for manipulation remains limited. In particular, estimating the mass of real-world objects is essential for determining appropriate grasp force and ensuring safe interaction. However, current VLMs lack reliable mass reasoning capabilities, and most existing benchmarks do not explicitly evaluate physical quantity estimation under realistic sensing conditions. In this work, we propose PhysQuantAgent, a framework for real-world object mass estimation using VLMs, together with VisPhysQuant, a new benchmark dataset for evaluation. VisPhysQuant consists of RGB-D videos of real objects captured from multiple viewpoints, annotated with precise mass measurements. To improve estimation accuracy, we introduce three visual prompting methods that enhance the input image with object detection, scale estimation, and cross-sectional image generation to help the model comprehend the size and internal structure of the target object. Experiments show that visual prompting significantly improves mass estimation accuracy on real-world data, suggesting the efficacy of integrating spatial reasoning with VLM knowledge for physical inference.

**arXiv ID:** 2603.16958
</details>

<details>
<summary><strong>LLM NL2SQL Robustness: Surface Noise vs. Linguistic Variation in Traditional and Agentic Settings</strong> - Lifu Tu, Rongguang Wang, Tao Sheng, Sujjith Ravi, Dan Roth - [[pdf]](https://arxiv.org/pdf/2603.17017)</summary>

**Abstract:** Robustness evaluation for Natural Language to SQL (NL2SQL) systems is essential because real-world database environments are dynamic, noisy, and continuously evolving, whereas conventional benchmark evaluations typically assume static schemas and well-formed user inputs. In this work, we introduce a robustness evaluation benchmark containing approximately ten types of perturbations and conduct evaluations under both traditional and agentic settings. We assess multiple state-of-the-art large language models (LLMs), including Grok-4.1, Gemini-3-Pro, Claude-Opus-4.6, and GPT-5.2. Our results show that these models generally maintain strong performance under several perturbations; however, notable performance degradation is observed for surface-level noise (e.g., character-level corruption) and linguistic variation that preserves semantics while altering lexical or syntactic forms. Furthermore, we observe that surface-level noise causes larger performance drops in traditional pipelines, whereas linguistic variation presents greater challenges in agentic settings. These findings highlight the remaining challenges in achieving robust NL2SQL systems, particularly in handling linguistic variability.

**arXiv ID:** 2603.17017
</details>

<details>
<summary><strong>When the Specification Emerges: Benchmarking Faithfulness Loss in Long-Horizon Coding Agents</strong> - Lu Yan, Xuan Chen, Xiangyu Zhang - [[pdf]](https://arxiv.org/pdf/2603.17104)</summary>

**Abstract:** Current coding-agent benchmarks usually pro- vide the full task specification upfront. Real research coding often does not: the intended system is progressively disclosed through in- teraction, requiring the agent to track durable design commitments across a long session. We introduce a benchmark for this setting and study faithfulne Ss Loss U nder eM ergent s Pecification (SLUMP), defined as the reduc- tion in final implementation faithfulness un- der emergent specification relative to a single- shot specification control. The benchmark con- tains 20 recent ML papers (10 ICML 2025, 10 NeurIPS 2025), 371 atomic verifiable compo- nents, and interaction scripts of approximately 60 coding requests that progressively disclose the target design without revealing the paper itself. Final repositories are scored with a five-level component-faithfulness rubric and accompanied by an exposure audit to verify that scored components are recoverable from the visible interaction. Evaluated on Claude Code and Codex, the single-shot specification control achieves higher overall implementation fidelity on 16/20 and 14/20 papers, respectively. Structural integration degrades under emergent specification on both platforms, while seman- tic faithfulness loss is substantial on Claude Code and small on Codex. As a mitigation case study, we introduce ProjectGuard, an exter- nal project-state layer for specification tracking. On Claude Code, ProjectGuard recovers 90% of the faithfulness gap, increases fully faith- ful components from 118 to 181, and reduces severe failures from 72 to 49. These results identify specification tracking as a distinct eval- uation target for long-horizon coding agents.

**arXiv ID:** 2603.17104
</details>

<details>
<summary><strong>Intent Formalization: A Grand Challenge for Reliable Coding in the Age of AI Agents</strong> - Shuvendu K. Lahiri - [[pdf]](https://arxiv.org/pdf/2603.17150)</summary>

**Abstract:** Agentic AI systems can now generate code with remarkable fluency, but a fundamental question remains: \emph{does the generated code actually do what the user intended?} The gap between informal natural language requirements and precise program behavior -- the \emph{intent gap} -- has always plagued software engineering, but AI-generated code amplifies it to an unprecedented scale. This article argues that \textbf{intent formalization} -- the translation of informal user intent into a set of checkable formal specifications -- is the key challenge that will determine whether AI makes software more reliable or merely more abundant. Intent formalization offers a tradeoff spectrum suitable to the reliability needs of different contexts: from lightweight tests that disambiguate likely misinterpretations, through full functional specifications for formal verification, to domain-specific languages from which correct code is synthesized automatically. The central bottleneck is \emph{validating specifications}: since there is no oracle for specification correctness other than the user, we need semi-automated metrics that can assess specification quality with or without code, through lightweight user interaction and proxy artifacts such as tests. We survey early research that demonstrates the \emph{potential} of this approach: interactive test-driven formalization that improves program correctness, AI-generated postconditions that catch real-world bugs missed by prior methods, and end-to-end verified pipelines that produce provably correct code from informal specifications. We outline the open research challenges -- scaling beyond benchmarks, achieving compositionality over changes, metrics for validating specifications, handling rich logics, designing human-AI specification interactions -- that define a research agenda spanning AI, programming languages, formal methods, and human-computer interaction.

**arXiv ID:** 2603.17150
</details>

<details>
<summary><strong>PAuth - Precise Task-Scoped Authorization For Agents</strong> - Reshabh K Sharma, Linxi Jiang, Zhiqiang Lin, Shuo Chen - [[pdf]](https://arxiv.org/pdf/2603.17170)</summary>

**Abstract:** The emerging agentic web envisions AI agents that reliably fulfill users' natural-language (NL)-based tasks by interacting with existing web services. However, existing authorization models are misaligned with this vision. In particular, today's operator-scoped authorization, exemplified by OAuth, grants broad permissions tied to operators (e.g., the transfer operator) rather than to the specific operations (e.g., transfer $100 to Bob) implied by a user's task. This will inevitably result in overprivileged agents.
We introduce Precise Task-Scoped Implicit Authorization (PAuth), a fundamentally different model in which submitting an NL task implicitly authorizes only the concrete operations required for its faithful execution. To make this enforceable at servers, we propose NL slices: symbolic specifications of the calls each service expects, derived from the task and upstream results. Complementing this, we also propose envelopes: special data structure to bind each operand's concrete value to its symbolic provenance, enabling servers to verify that all operands arise from legitimate computations.
PAuth is prototyped in the agent-security evaluation framework AgentDojo. We evaluate it in both benign settings and attack scenarios where a spurious operation is injected into an otherwise normal task. In all benign tests, PAuth executes the tasks successfully without requiring any additional permissions. In all attack tests, PAuth correctly raises warnings about missing permissions. These results demonstrate that PAuth's reasoning about permissions is indeed precise. We further analyze the characteristics of these tasks and measure the associated token costs.

**arXiv ID:** 2603.17170
</details>

<details>
<summary><strong>TDAD: Test-Driven Agentic Development - Reducing Code Regressions in AI Coding Agents via Graph-Based Impact Analysis</strong> - Pepe Alonso - [[pdf]](https://arxiv.org/pdf/2603.17973)</summary>

**Abstract:** AI coding agents can resolve real-world software issues, yet they frequently introduce regressions, breaking tests that previously passed. Current benchmarks focus almost exclusively on resolution rate, leaving regression behavior under-studied. This paper presents TDAD (Test-Driven Agentic Development), an open-source tool and benchmark methodology that combines abstract-syntax-tree (AST) based code-test graph construction with weighted impact analysis to surface the tests most likely affected by a proposed change. Evaluated on SWE-bench Verified with two local models (Qwen3-Coder 30B on 100 instances and Qwen3.5-35B-A3B on 25 instances), TDAD's GraphRAG workflow reduced test-level regressions by 70% (6.08% to 1.82%) and improved resolution from 24% to 32% when deployed as an agent skill. A surprising finding is that TDD prompting alone increased regressions (9.94%), revealing that smaller models benefit more from contextual information (which tests to verify) than from procedural instructions (how to do TDD). An autonomous auto-improvement loop raised resolution from 12% to 60% on a 10-instance subset with 0% regression. These findings suggest that for AI agent tool design, surfacing contextual information outperforms prescribing procedural workflows. All code, data, and logs are publicly available at this https URL.

**arXiv ID:** 2603.17973
</details>

<details>
<summary><strong>See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles</strong> - Zongru Wu, Rui Mao, Zhiyuan Tian, Pengzhou Cheng, Tianjie Ju, Zheng Wu, Lingzhong Dong, Haiyue Sheng, Zhuosheng Zhang, Gongshen Liu - [[pdf]](https://arxiv.org/pdf/2509.13615)</summary>

**Abstract:** The advent of multimodal agents facilitates effective interaction within graphical user interface (GUI), especially in ubiquitous GUI control. However, their inability to reliably execute toggle control instructions remains a key bottleneck. To investigate this, we construct a state control benchmark with binary toggle instructions derived from public datasets. Evaluation results of existing agents demonstrate their notable unreliability, particularly when the current toggle state already matches the desired state. To address the challenge, we propose State-aware Reasoning (StaR), a multimodal reasoning method that enables agents to perceive the current toggle state, infer the desired state from the instruction, and act accordingly. Experiments on four multimodal agents demonstrate that StaR can improve toggle instruction execution accuracy by over 30\%. Further evaluations on three public agentic benchmarks show that StaR also enhances general agentic task performance. Finally, evaluations on a dynamic environment highlight the potential of StaR for real-world applications. Code and benchmark: this https URL.

**arXiv ID:** 2509.13615
</details>

<details>
<summary><strong>JobMatchAI An Intelligent Job Matching Platform Using Knowledge Graphs, Semantic Search and Explainable AI</strong> - Mayank Vyas, Abhijit Chakraborty, Vivek Gupta - [[pdf]](https://arxiv.org/pdf/2603.14558)</summary>

**Abstract:** Recruiters and job seekers rely on search systems to navigate labor markets, making candidate matching engines critical for hiring outcomes. Most systems act as keyword filters, failing to handle skill synonyms and nonlinear careers, resulting in missed candidates and opaque match scores. We introduce JobMatchAI, a production-ready system integrating Transformer embeddings, skill knowledge graphs, and interpretable reranking. Our system optimizes utility across skill fit, experience, location, salary, and company preferences, providing factor-wise explanations through resume-driven search workflows. We release JobSearch-XS benchmark and a hybrid retrieval stack combining BM25, knowledge graph and semantic components to evaluate skill generalization. We assess system performance on JobSearch-XS across retrieval tasks, provide a demo video, a hosted website and installable package.

**arXiv ID:** 2603.14558
</details>

<details>
<summary><strong>The Comprehension-Gated Agent Economy: A Robustness-First Architecture for AI Economic Agency</strong> - Rahul Baxi - [[pdf]](https://arxiv.org/pdf/2603.15639)</summary>

**Abstract:** AI agents are increasingly granted economic agency (executing trades, managing budgets, negotiating contracts, and spawning sub-agents), yet current frameworks gate this agency on capability benchmarks that are empirically uncorrelated with operational robustness. We introduce the Comprehension-Gated Agent Economy (CGAE), a formal architecture in which an agent's economic permissions are upper-bounded by a verified comprehension function derived from adversarial robustness audits. The gating mechanism operates over three orthogonal robustness dimensions: constraint compliance (measured by CDCT), epistemic integrity (measured by DDFT), and behavioral alignment (measured by AGT), with intrinsic hallucination rates serving as a cross-cutting diagnostic. We define a weakest-link gate function that maps robustness vectors to discrete economic tiers, and prove three properties of the resulting system: (1) bounded economic exposure, ensuring maximum financial liability is a function of verified robustness; (2) incentive-compatible robustness investment, showing rational agents maximize profit by improving robustness rather than scaling capability alone; and (3) monotonic safety scaling, demonstrating that aggregate system safety does not decrease as the economy grows. The architecture includes temporal decay and stochastic re-auditing mechanisms that prevent post-certification drift. CGAE provides the first formal bridge between empirical AI robustness evaluation and economic governance, transforming safety from a regulatory burden into a competitive advantage.

**arXiv ID:** 2603.15639
</details>

<details>
<summary><strong>EdiVal-Agent: An Object-Centric Framework for Automated, Fine-Grained Evaluation of Multi-Turn Editing</strong> - Tianyu Chen, Yasi Zhang, Zhi Zhang, Peiyu Yu, Shu Wang, Zhendong Wang, Kevin Lin, Xiaofei Wang, Zhengyuan Yang, Linjie Li, Chung-Ching Lin, Jianwen Xie, Oscar Leong, Lijuan Wang, Ying Nian Wu, Mingyuan Zhou - [[pdf]](https://arxiv.org/pdf/2509.13399)</summary>

**Abstract:** Instruction-based image editing has advanced rapidly, yet reliable and interpretable evaluation remains a bottleneck. Current protocols either (i) depend on paired reference images, resulting in limited coverage and inheriting biases from prior generative models or (ii) rely solely on zero-shot vision language models (VLMs), whose prompt-based assessments of instruction following, content consistency, and visual quality are often imprecise. To address this, we introduce EdiVal, an automated and fine-grained evaluation framework grounded in an object-centric perspective, designed to assess not only standard single-turn but also multi-turn instruction-based editing with precision. Given an input image, EdiVal first decomposes it into semantically meaningful objects, then synthesizes diverse, context-aware editing instructions while dynamically updating object pools across turns. These two stages enable two novel object centric metrics tailored for multi turn evaluation and one global metric of visual quality: 1) EdiVal-IF, which measures instruction following by combining open vocabulary object detectors for symbolic checks with VLMs for semantic verification on detector guided crops; 2) EdiVal-CC, which evaluates content consistency by calculating semantic similarity of unchanged objects and background using the evolving object pools; and 3) EdiVal-VQ, which quantifies changes in overall visual quality with human preference models. Instantiating this pipeline, we build EdiVal Bench, a multi-turn editing benchmark covering 9 instruction types and 16 state-of-the-art editing models, spanning in-context, flow-matching, and diffusion paradigms. We demonstrate that EdiVal can be used to identify existing failure modes, thereby informing the development of the next generation of editing models.

**arXiv ID:** 2509.13399
</details>

<details>
<summary><strong>Grid Spatial Understanding: A Dataset for Textual Spatial Reasoning over Grids, Embodied Settings, and Coordinate Structures</strong> - Risham Sidhu, Julia Hockenmaier - [[pdf]](https://arxiv.org/pdf/2603.17333)</summary>

**Abstract:** We introduce GSU, a text-only grid dataset to evaluate the spatial reasoning capabilities of LLMs over 3 core tasks: navigation, object localization, and structure composition. By forgoing visual inputs, isolating spatial reasoning from perception, we show that while most models grasp basic grid concepts, they struggle with frames of reference relative to an embodied agent and identifying 3D shapes from coordinate lists. We also find that exposure to a visual modality does not provide a generalizable understanding of 3D space that VLMs are able to utilize for these tasks. Finally, we show that while the very latest frontier models can solve the provided tasks (though harder variants may still stump them), fully fine-tuning a small LM or LORA fine-tuning a small LLM show potential to match frontier model performance, suggesting an avenue for specialized embodied agents.

**arXiv ID:** 2603.17333
</details>

<details>
<summary><strong>SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration</strong> - Jialong Chen, Xander Xu, Hu Wei, Chuan Chen, Bing Zhao - [[pdf]](https://arxiv.org/pdf/2603.03823)</summary>

**Abstract:** Large language model (LLM)-powered agents have demonstrated strong capabilities in automating software engineering tasks such as static bug fixing, as evidenced by benchmarks like SWE-bench. However, in the real world, the development of mature software is typically predicated on complex requirement changes and long-term feature iterations -- a process that static, one-shot repair paradigms fail to capture. To bridge this gap, we propose \textbf{SWE-CI}, the first repository-level benchmark built upon the Continuous Integration loop, aiming to shift the evaluation paradigm for code generation from static, short-term \textit{functional correctness} toward dynamic, long-term \textit{maintainability}. The benchmark comprises 100 tasks, each corresponding on average to an evolution history spanning 233 days and 71 consecutive commits in a real-world code repository. SWE-CI requires agents to systematically resolve these tasks through dozens of rounds of analysis and coding iterations. SWE-CI provides valuable insights into how well agents can sustain code quality throughout long-term evolution.

**arXiv ID:** 2603.03823
</details>

<details>
<summary><strong>DesertFormer: Transformer-Based Semantic Segmentation for Off-Road Desert Terrain Classification in Autonomous Navigation Systems</strong> - Yasaswini Chebolu - [[pdf]](https://arxiv.org/pdf/2603.17056)</summary>

**Abstract:** Reliable terrain perception is a fundamental requirement for autonomous navigation in unstructured, off-road environments. Desert landscapes present unique challenges due to low chromatic contrast between terrain categories, extreme lighting variability, and sparse vegetation that defy the assumptions of standard road-scene segmentation models. We present DesertFormer, a semantic segmentation pipeline for off-road desert terrain analysis based on SegFormer B2 with a hierarchical Mix Transformer (MiT-B2) backbone. The system classifies terrain into ten ecologically meaningful categories -- Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Flowers, Logs, Rocks, Landscape, and Sky -- enabling safety-aware path planning for ground robots and autonomous vehicles. Trained on a purpose-built dataset of 4,176 annotated off-road images at 512x512 resolution, DesertFormer achieves a mean Intersection-over-Union (mIoU) of 64.4% and pixel accuracy of 86.1%, representing a +24.2% absolute improvement over a DeepLabV3 MobileNetV2 baseline (41.0% mIoU). We further contribute a systematic failure analysis identifying the primary confusion patterns -- Ground Clutter to Landscape and Dry Grass to Landscape -- and propose class-weighted training and copy-paste augmentation for rare terrain categories. Code, checkpoints, and an interactive inference dashboard are released at this https URL.

**arXiv ID:** 2603.17056
</details>

<details>
<summary><strong>HeiSD: Hybrid Speculative Decoding for Embodied Vision-Language-Action Models with Kinematic Awareness</strong> - Zihao Zheng, Zhihao Mao, Sicheng Tian, Maoliang Li, Jiayu Chen, Xinhao Sun, Zhaobo Zhang, Xuanzhe Liu, Donggang Cao, Hong Mei, Xiang Chen - [[pdf]](https://arxiv.org/pdf/2603.17573)</summary>

**Abstract:** Vision-Language-Action (VLA) Models have become the mainstream solution for robot control, but suffer from slow inference speeds. Speculative Decoding (SD) is a promising acceleration method which can be divided into two categories: drafter-based SD and retrieval-based SD. Existing methods fail to analyze the advantages and disadvantages of these two types of SD in VLA models, leading to their sole application or optimization. In this paper, we analyze the trajectory patterns of robots controlled by the VLA model and derive a key insight: the two types of SD should be used in a hybrid manner. However, achieving hybrid SD in VLA models poses several challenges: (1) draft rejection and persistent errors in retrieval-based SD; (2) difficulty in determining the hybrid boundary. To address these, we propose the HeiSD framework. We propose a retrieval-based SD optimization method in HeiSD,which contains a verify-skip mechanism and a sequence-wise relaxed acceptance strategy. Moreover, we proposed a kinematic-based fused metric in HeiSD to automatically determine the hybrid boundary. Experimental results demonstrate that HeiSD attains a speedup of up to 2.45x in simulation benchmarks and 2.06x~2.41x in real-world scenarios, while sustaining a high task success rate.

**arXiv ID:** 2603.17573
</details>

<details>
<summary><strong>OMNIFLOW: A Physics-Grounded Multimodal Agent for Generalized Scientific Reasoning</strong> - Hao Wu, Yongheng Zhang, Yuan Gao, Fan Xu, Fan Zhang, Ruobing Xie, Ruijian Gou, Yuxuan Liang, Xiaomeng Huang, Xian Wu - [[pdf]](https://arxiv.org/pdf/2603.15797)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated exceptional logical reasoning capabilities but frequently struggle with the continuous spatiotemporal dynamics governed by Partial Differential Equations (PDEs), often resulting in non-physical hallucinations. Existing approaches typically resort to costly, domain-specific fine-tuning, which severely limits cross-domain generalization and interpretability. To bridge this gap, we propose OMNIFLOW, a neuro-symbolic architecture designed to ground frozen multimodal LLMs in fundamental physical laws without requiring domain-specific parameter updates. OMNIFLOW introduces a novel \textit{Semantic-Symbolic Alignment} mechanism that projects high-dimensional flow tensors into topological linguistic descriptors, enabling the model to perceive physical structures rather than raw pixel values. Furthermore, we construct a Physics-Guided Chain-of-Thought (PG-CoT) workflow that orchestrates reasoning through dynamic constraint injection (e.g., mass conservation) and iterative reflexive verification. We evaluate OMNIFLOW on a comprehensive benchmark spanning microscopic turbulence, theoretical Navier-Stokes equations, and macroscopic global weather forecasting. Empirical results demonstrate that OMNIFLOW significantly outperforms traditional deep learning baselines in zero-shot generalization and few-shot adaptation tasks. Crucially, it offers transparent, physically consistent reasoning reports, marking a paradigm shift from black-box fitting to interpretable scientific reasoning.

**arXiv ID:** 2603.15797
</details>

<details>
<summary><strong>SafeLand: Safe Autonomous Landing in Unknown Environments with Bayesian Semantic Mapping</strong> - Markus Gross, Andreas Greiner, Sai Bharadhwaj Matha, Felix Soest, Daniel Cremers, Henri Meeß - [[pdf]](https://arxiv.org/pdf/2603.17430)</summary>

**Abstract:** Autonomous landing of uncrewed aerial vehicles (UAVs) in unknown, dynamic environments poses significant safety challenges, particularly near people and infrastructure, as UAVs transition to routine urban and rural operations. Existing methods often rely on prior maps, heavy sensors like LiDAR, static markers, or fail to handle non-cooperative dynamic obstacles like humans, limiting generalization and real-time performance. To address these challenges, we introduce SafeLand, a lean, vision-based system for safe autonomous landing (SAL) that requires no prior information and operates only with a camera and a lightweight height sensor. Our approach constructs an online semantic ground map via deep learning-based semantic segmentation, optimized for embedded deployment and trained on a consolidation of seven curated public aerial datasets (achieving 70.22% mIoU across 20 classes), which is further refined through Bayesian probabilistic filtering with temporal semantic decay to robustly identify metric-scale landing spots. A behavior tree then governs adaptive landing, iteratively validates the spot, and reacts in real time to dynamic obstacles by pausing, climbing, or rerouting to alternative spots, maximizing human safety. We extensively evaluate our method in 200 simulations and 60 end-to-end field tests across industrial, urban, and rural environments at altitudes up to 100m, demonstrating zero false negatives for human detection. Compared to the state of the art, SafeLand achieves sub-second response latency, substantially lower than previous methods, while maintaining a superior success rate of 95%. To facilitate further research in aerial robotics, we release SafeLand's segmentation model as a plug-and-play ROS package, available at this https URL.

**arXiv ID:** 2603.17430
</details>

<details>
<summary><strong>Multi-Source Human-in-the-Loop Digital Twin Testbed for Connected and Autonomous Vehicles in Mixed Traffic Flow</strong> - Jianghong Dong, Jiawei Wang, Chunying Yang, Mengchi Cai, Chaoyi Chen, Qing Xu, Jianqiang Wang, Keqiang Li - [[pdf]](https://arxiv.org/pdf/2603.17751)</summary>

**Abstract:** In the emerging mixed traffic environments, Connected and Autonomous Vehicles (CAVs) have to interact with surrounding human-driven vehicles (HDVs). This paper introduces MSH-MCCT (Multi-Source Human-in-the-Loop Mixed Cloud Control Testbed), a novel CAV testbed that captures complex interactions between various CAVs and HDVs. Utilizing the Mixed Digital Twin concept, which combines Mixed Reality with Digital Twin, MSH-MCCT integrates physical, virtual, and mixed platforms, along with multi-source control inputs. Bridged by the mixed platform, MSH-MCCT allows human drivers and CAV algorithms to operate both physical and virtual vehicles within multiple fields of view. Particularly, this testbed facilitates the coexistence and real-time interaction of physical and virtual CAVs \& HDVs, significantly enhancing the experimental flexibility and scalability. Experiments on vehicle platooning in mixed traffic showcase the potential of MSH-MCCT to conduct CAV testing with multi-source real human drivers in the loop through driving simulators of diverse fidelity. The videos for the experiments are available at our project website: this https URL.

**arXiv ID:** 2603.17751
</details>

<details>
<summary><strong>TrackDeform3D: Markerless and Autonomous 3D Keypoint Tracking and Dataset Collection for Deformable Objects</strong> - Yeheng Zong, Yizhou Chen, Alexander Bowler, Chia-Tung Yang, Ram Vasudevan - [[pdf]](https://arxiv.org/pdf/2603.17068)</summary>

**Abstract:** Structured 3D representations such as keypoints and meshes offer compact, expressive descriptions of deformable objects, jointly capturing geometric and topological information useful for downstream tasks such as dynamics modeling and motion planning. However, robustly extracting such representations remains challenging, as current perception methods struggle to handle complex deformations. Moreover, large-scale 3D data collection remains a bottleneck: existing approaches either require prohibitive data collection efforts, such as labor-intensive annotation or expensive motion capture setups, or rely on simplifying assumptions that break down in unstructured environments. As a result, large-scale 3D datasets and benchmarks for deformable objects remain scarce. To address these challenges, this paper presents an affordable and autonomous framework for collecting 3D datasets of deformable objects using only RGB-D cameras. The proposed method identifies 3D keypoints and robustly tracks their trajectories, incorporating motion consistency constraints to produce temporally smooth and geometrically coherent data. TrackDeform3D is evaluated against several state-of-the-art tracking methods across diverse object categories and demonstrates consistent improvements in both geometric and tracking accuracy. Using this framework, this paper presents a high-quality, large-scale dataset consisting of 6 deformable objects, totaling 110 minutes of trajectory data.

**arXiv ID:** 2603.17068
</details>

<details>
<summary><strong>Echo Planning for Autonomous Driving: From Current Observations to Future Trajectories and Back</strong> - Jintao Sun, Hu Zhang, Gangyi Ding, Zhedong Zheng - [[pdf]](https://arxiv.org/pdf/2505.18945)</summary>

**Abstract:** Modern end-to-end autonomous driving systems suffer from a critical limitation: their planners lack mechanisms to enforce temporal consistency between predicted trajectories and evolving scene dynamics. This absence of self-supervision allows early prediction errors to compound catastrophically over time. We introduce Echo Planning (EchoP), a new self-correcting framework that establishes an end-to-end Current - Future - Current (CFC) cycle to harmonize trajectory prediction with scene coherence. Our key insight is that plausible future trajectories should be bi-directionally consistent, i.e., not only generated from current observations but also capable of reconstructing them. The CFC mechanism first predicts future trajectories from the Bird's-Eye-View (BEV) scene representation, then inversely maps these trajectories back to estimate the current BEV state. By enforcing consistency between the original and reconstructed BEV representations through a cycle loss, the framework intrinsically penalizes physically implausible or misaligned trajectories. Experiments on nuScenes show that the proposed method yields competitive performance, reducing L2 error (Avg) by -0.04 m and collision rate by -0.12% compared to one-shot planners. Moreover, EchoP seamlessly extends to closed-loop evaluation, i.e., Bench2Drive, attaining a 26.54% success rate. Notably, EchoP requires no additional supervision: the CFC cycle acts as an inductive bias that stabilizes long-horizon planning. Overall, EchoP offers a simple, deployable pathway to improve reliability in safety-critical autonomous driving.

**arXiv ID:** 2505.18945
</details>

<details>
<summary><strong>AutoMoT: A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving</strong> - Wenhui Huang, Songyan Zhang, Qihang Huang, Zhidong Wang, Zhiqi Mao, Collister Chua, Zhan Chen, Long Chen, Chen Lv - [[pdf]](https://arxiv.org/pdf/2603.14851)</summary>

**Abstract:** Integrating vision-language models (VLMs) into end-to-end (E2E) autonomous driving (AD) systems has shown promise in improving scene understanding. However, existing integration strategies suffer from several limitations: they either struggle to resolve distribution misalignment between reasoning and action spaces, underexploit the general reasoning capabilities of pretrained VLMs, or incur substantial inference latency during action policy generation, which degrades driving performance. To address these challenges, we propose \OURS in this work, an end-to-end AD framework that unifies reasoning and action generation within a single vision-language-action (VLA) model. Our approach leverages a mixture-of-transformer (MoT) architecture with joint attention sharing, which preserves the general reasoning capabilities of pre-trained VLMs while enabling efficient fast-slow inference through asynchronous execution at different task frequencies. Extensive experiments on multiple benchmarks, under both open- and closed-loop settings, demonstrate that \OURS achieves competitive performance compared to state-of-the-art methods. We further investigate the functional boundary of pre-trained VLMs in AD, examining when AD-tailored fine-tuning is necessary. Our results show that pre-trained VLMs can achieve competitive multi-task scene understanding performance through semantic prompting alone, while fine-tuning remains essential for action-level tasks such as decision-making and trajectory planning. We refer to \href{this https URL}{Project Page} for the demonstration videos and qualitative results.

**arXiv ID:** 2603.14851
</details>

</details>

<details open>
<summary><h2>LLM Agents (9 papers)</h2></summary>

<details>
<summary><strong>VeriGrey: Greybox Agent Validation</strong> - Yuntong Zhang, Sungmin Kang, Ruijie Meng, Marcel Böhme, Abhik Roychoudhury - [[pdf]](https://arxiv.org/pdf/2603.17639)</summary>

**Abstract:** Agentic AI has been a topic of great interest recently. A Large Language Model (LLM) agent involves one or more LLMs in the back-end. In the front end, it conducts autonomous decision-making by combining the LLM outputs with results obtained by invoking several external tools. The autonomous interactions with the external environment introduce critical security risks.
In this paper, we present a grey-box approach to explore diverse behaviors and uncover security risks in LLM agents. Our approach VeriGrey uses the sequence of tools invoked as a feedback function to drive the testing process. This helps uncover infrequent but dangerous tool invocations that cause unexpected agent behavior. As mutation operators in the testing process, we mutate prompts to design pernicious injection prompts. This is carefully accomplished by linking the task of the agent to an injection task, so that the injection task becomes a necessary step of completing the agent functionality. Comparing our approach with a black-box baseline on the well-known AgentDojo benchmark, VeriGrey achieves 33% additional efficacy in finding indirect prompt injection vulnerabilities with a GPT-4.1 back-end.
We also conduct real-world case studies with the widely used coding agent Gemini CLI, and the well-known OpenClaw personal assistant. VeriGrey finds prompts inducing several attack scenarios that could not be identified by black-box approaches. In OpenClaw, by constructing a conversation agent which employs mutational fuzz testing as needed, VeriGrey is able to discover malicious skill variants from 10 malicious skills (with 10/10= 100% success rate on the Kimi-K2.5 LLM backend, and 9/10= 90% success rate on Opus 4.6 LLM backend). This demonstrates the value of a dynamic approach like VeriGrey to test agents, and to eventually lead to an agent assurance framework.

**arXiv ID:** 2603.17639
</details>

<details>
<summary><strong>Sensi: Learn One Thing at a Time -- Curriculum-Based Test-Time Learning for LLM Game Agents</strong> - Mohsen Arjmandi - [[pdf]](https://arxiv.org/pdf/2603.17683)</summary>

**Abstract:** Large language model (LLM) agents deployed in unknown environments must learn task structure at test time, but current approaches require thousands of interactions to form useful hypotheses. We present Sensi, an LLM agent architecture for the ARC-AGI-3 game-playing challenge that introduces structured test-time learning through three mechanisms: (1) a two-player architecture separating perception from action, (2) a curriculum-based learning system managed by an external state machine, and (3) a database-as-control-plane that makes the agents context window programmatically steerable. We further introduce an LLM-as-judge component with dynamically generated evaluation rubrics to determine when the agent has learned enough about one topic to advance to the next. We report results across two iterations: Sensi v1 solves 2 game levels using the two-player architecture alone, while Sensi v2 adds curriculum learning and solves 0 levels - but completes its entire learning curriculum in approximately 32 action attempts, achieving 50-94x greater sample efficiency than comparable systems that require 1600-3000 attempts. We precisely diagnose the failure mode as a self-consistent hallucination cascade originating in the perception layer, demonstrating that the architectural bottleneck has shifted from learning efficiency to perceptual grounding - a more tractable problem.

**arXiv ID:** 2603.17683
</details>

<details>
<summary><strong>RPMS: Enhancing LLM-Based Embodied Planning through Rule-Augmented Memory Synergy</strong> - Zhenhang Yuan, Shenghai Yuan, Lihua Xie - [[pdf]](https://arxiv.org/pdf/2603.17831)</summary>

**Abstract:** LLM agents often fail in closed-world embodied environments because actions must satisfy strict preconditions -- such as location, inventory, and container states -- and failure feedback is sparse. We identify two structurally coupled failure modes: (P1) invalid action generation and (P2) state drift, each amplifying the other in a degenerative cycle. We present RPMS, a conflict-managed architecture that enforces action feasibility via structured rule retrieval, gates memory applicability via a lightweight belief state, and resolves conflicts between the two sources via rules-first arbitration. On ALFWorld (134 unseen tasks), RPMS achieves 59.7% single-trial success with Llama 3.1 8B (+23.9 pp over baseline) and 98.5% with Claude Sonnet 4.5 (+11.9 pp); of the 8B gain, rule retrieval alone contributes +14.9 pp (statistically significant), making it the dominant factor. A key finding is that episodic memory is conditionally useful: it harms performance on some task types when used without grounding, but becomes a stable net positive once filtered by current state and constrained by explicit action rules. Adapting RPMS to ScienceWorld with GPT-4 yields consistent gains across all ablation conditions (avg. score 54.0 vs. 44.9 for the ReAct baseline), providing transfer evidence that the core mechanisms hold across structurally distinct environments.

**arXiv ID:** 2603.17831
</details>

<details>
<summary><strong>Post-Training Local LLM Agents for Linux Privilege Escalation with Verifiable Rewards</strong> - Philipp Normann, Andreas Happe, Jürgen Cito, Daniel Arp - [[pdf]](https://arxiv.org/pdf/2603.17673)</summary>

**Abstract:** LLM agents are increasingly relevant to research domains such as vulnerability discovery. Yet, the strongest systems remain closed and cloud-only, making them resource-intensive, difficult to reproduce, and unsuitable for work involving proprietary code or sensitive data. Consequently, there is an urgent need for small, local models that can perform security tasks under strict resource budgets, but methods for developing them remain underexplored. In this paper, we address this gap by proposing a two-stage post-training pipeline. We focus on the problem of Linux privilege escalation, where success is automatically verifiable and the task requires multi-step interactive reasoning. Using an experimental setup that prevents data leakage, we post-train a 4B model in two stages: supervised fine-tuning on traces from procedurally generated privilege-escalation environments, followed by reinforcement learning with verifiable rewards. On a held-out benchmark of 12 Linux privilege-escalation scenarios, supervised fine-tuning alone more than doubles the baseline success rate at 20 rounds, and reinforcement learning further lifts our resulting model, PrivEsc-LLM, to 95.8%, nearly matching Claude Opus 4.6 at 97.5%. At the same time, the expected inference cost per successful escalation is reduced by over 100x.

**arXiv ID:** 2603.17673
</details>

<details>
<summary><strong>Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions</strong> - Yuanzhe Hu, Yu Wang, Julian McAuley - [[pdf]](https://arxiv.org/pdf/2507.05257)</summary>

**Abstract:** Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component-memory, encompassing how agents memorize, update, and retrieve long-term information-is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. In this paper, based on classic theories from memory science and cognitive science, we identify four core competencies essential for memory agents: accurate retrieval, test-time learning, long-range understanding, and selective forgetting. Existing benchmarks either rely on limited context lengths or are tailored for static, long-context settings like book-based QA, which do not reflect the interactive, multi-turn nature of memory agents that incrementally accumulate information. Moreover, no existing benchmarks cover all four competencies. We introduce MemoryAgentBench, a new benchmark specifically designed for memory agents. Our benchmark transforms existing long-context datasets and incorporates newly constructed datasets into a multi-turn format, effectively simulating the incremental information processing characteristic of memory agents. By carefully selecting and curating datasets, our benchmark provides comprehensive coverage of the four core memory competencies outlined above, thereby offering a systematic and challenging testbed for assessing memory quality. We evaluate a diverse set of memory agents, ranging from simple context-based and retrieval-augmented generation (RAG) systems to advanced agents with external memory modules and tool integration. Empirical results reveal that current methods fall short of mastering all four competencies, underscoring the need for further research into comprehensive memory mechanisms for LLM agents.

**arXiv ID:** 2507.05257
</details>

<details>
<summary><strong>Tree Search for LLM Agent Reinforcement Learning</strong> - Yuxiang Ji, Ziyu Ma, Yong Wang, Guanhua Chen, Xiangxiang Chu, Liaoni Wu - [[pdf]](https://arxiv.org/pdf/2509.21240)</summary>

**Abstract:** Recent advances in reinforcement learning (RL) have significantly enhanced the agentic capabilities of large language models (LLMs). In long-term and multi-turn agent tasks, existing approaches driven solely by outcome rewards often suffer from the problem of sparse supervision. To address the challenge, we propose Tree-based Group Relative Policy Optimization (Tree-GRPO), a grouped agent RL method based on tree search, where each tree node represents the complete agent interaction step. By sharing common prefixes, the tree search sampling increases the number of rollouts achievable within a fixed budget of tokens or tool calls. Moreover, we find that the tree-structured trajectory naturally allows the construction of step-wise process supervised signals even using only the outcome reward. Based on this, Tree-GRPO estimates the grouped relative advantages both on intra-tree and inter-tree levels. Through theoretical analysis, we demonstrate that the objective of intra-tree level group relative policy optimization is equivalent to that of step-level direct preference learning. Experiments across 11 datasets and 3 types of QA tasks demonstrate the superiority of the proposed tree-based RL over the chain-based RL method.

**arXiv ID:** 2509.21240
</details>

<details>
<summary><strong>Agentic Cognitive Profiling: Realigning Automated Alzheimer's Disease Detection with Clinical Construct Validity</strong> - Jiawen Kang, Kun Li, Dongrui Han, Jinchao Li, Junan Li, Lingwei Meng, Xixin Wu, Helen Meng - [[pdf]](https://arxiv.org/pdf/2603.17392)</summary>

**Abstract:** Automated Alzheimer's Disease (AD) screening has predominantly followed the inductive paradigm of pattern recognition, which directly maps the input signal to the outcome label. This paradigm sacrifices construct validity of clinical protocol for statistical shortcuts. This paper proposes Agentic Cognitive Profiling (ACP), an agentic framework that realigns automated screening with clinical protocol logic across multiple cognitive domains. Rather than learning opaque mappings from transcripts to labels, the framework decomposes standardized assessments into atomic cognitive tasks and orchestrates specialized LLM agents to extract verifiable scoring primitives. Central to our design is decoupling semantic understanding from measurement by delegating all quantification to deterministic function calling, thereby mitigating hallucination and restoring construct validity. Unlike popular datasets that typically comprise around a hundred participants under a single task, we evaluate on a clinically-annotated corpus of 402 participants across eight structured cognitive tasks spanning multiple cognitive domains. The framework achieves 90.5% score match rate in task examination and 85.3% accuracy in AD prediction, surpassing popular baselines while generating interpretable cognitive profiles grounded in behavioral evidence. This work demonstrates that construct validity and predictive performance need not be traded off, charting a path toward AD screening systems that explain rather than merely predict.

**arXiv ID:** 2603.17392
</details>

<details>
<summary><strong>Is Your LLM-as-a-Recommender Agent Trustable? LLMs' Recommendation is Easily Hacked by Biases (Preferences)</strong> - Zichen Tang, Zirui Zhang, Qian Wang, Zhenheng Tang, Bo Li, Xiaowen Chu - [[pdf]](https://arxiv.org/pdf/2603.17417)</summary>

**Abstract:** Current Large Language Models (LLMs) are gradually exploited in practically valuable agentic workflows such as Deep Research, E-commerce recommendation, and job recruitment. In these applications, LLMs need to select some optimal solutions from massive candidates, which we term as \textit{LLM-as-a-Recommender} paradigm. However, the reliability of using LLM agents for recommendations is underexplored. In this work, we introduce a \textbf{Bias} \textbf{Rec}ommendation \textbf{Bench}mark (\textbf{BiasRecBench}) to highlight the critical vulnerability of such agents to biases in high-value real-world tasks. The benchmark includes three practical domains: paper review, e-commerce, and job recruitment. We construct a \textsc{Bias Synthesis Pipeline with Calibrated Quality Margins} that 1) synthesizes evaluation data by controlling the quality gap between optimal and sub-optimal options to provide a calibrated testbed to elicit the vulnerability to biases; 2) injects contextual biases that are logical and suitable for option contexts. Extensive experiments on both SOTA (Gemini-{2.5,3}-pro, GPT-4o, DeepSeek-R1) and small-scale LLMs reveal that agents frequently succumb to injected biases despite having sufficient reasoning capabilities to identify the ground truth. These findings expose a significant reliability bottleneck in current agentic workflows, calling for specialized alignment strategies for LLM-as-a-Recommender. The complete code and evaluation datasets will be made publicly available shortly.

**arXiv ID:** 2603.17417
</details>

<details>
<summary><strong>Noticing the Watcher: LLM Agents Can Infer CoT Monitoring from Blocking Feedback</strong> - Thomas Jiralerspong, Flemming Kondrup, Yoshua Bengio - [[pdf]](https://arxiv.org/pdf/2603.16928)</summary>

**Abstract:** Chain-of-thought (CoT) monitoring is proposed as a method for overseeing the internal reasoning of language-model agents. Prior work has shown that when models are explicitly informed that their reasoning is being monitored, or are fine-tuned to internalize this fact, they may learn to obfuscate their CoTs in ways that allow them to evade CoT-based monitoring systems. We ask whether reasoning agents can autonomously infer that their supposedly private CoT is under surveillance, and whether this awareness leads to strategic evasion, without any explicit training or instructions. In a multi-episode agentic framework, models pursue both a primary task and a concealed side task while being told that their reasoning is private; a hidden CoT monitor blocks episodes when suspicious reasoning is detected. We find that frontier models can deduce the existence of this monitor purely from blocking feedback, with the most capable models reaching a confident belief that their thinking is observed in up to 19% of episodes. This awareness scales with model capability and, in rare cases, escalates to an explicit intent to suppress reasoning about the side task. However, models that form this intent uniformly fail to execute it, openly reasoning about their concealed objectives in the very next episode. This intent-capability gap is reassuring for current deployment, but the autonomous emergence of both monitoring awareness and evasion intent suggests that CoT monitoring is not a permanently reliable safeguard.

**arXiv ID:** 2603.16928
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (23 papers)</h2></summary>

<details>
<summary><strong>Cascade-Aware Multi-Agent Routing: Spatio-Temporal Sidecars and Geometry-Switching</strong> - Davide Di Gioia - [[pdf]](https://arxiv.org/pdf/2603.17112)</summary>

**Abstract:** A common architectural pattern in advanced AI reasoning systems is the symbolic graph network: specialized agents or modules connected by delegation edges, routing tasks through a dynamic execution graph. Current schedulers optimize load and fitness but are geometry-blind: they do not model how failures propagate differently in tree-like versus cyclic regimes. In tree-like delegation, a single failure can cascade exponentially; in dense cyclic graphs, failures tend to self-limit. We identify this observability gap, quantify its system-level cost, and propose a lightweight mitigation.
We formulate online geometry control for route-risk estimation on time-indexed execution graphs with route-local failure history. Our approach combines (i) a Euclidean spatio-temporal propagation baseline, (ii) a hyperbolic route-risk model with temporal decay (and optional burst excitation), and (iii) a learned geometry selector over structural features. The selector is a compact MLP (9->12->1) using six topology statistics plus three geometry-aware signals: BFS shell-growth slope, cycle-rank norm, and fitted Poincare curvature. On the Genesis 3 benchmark distribution, adaptive switching improves win rate in the hardest non_tree regime from 64-72% (fixed hyperbolic variants) to 92%, and achieves 87.2% overall win rate.
To measure total system value, we compare against Genesis 3 routing without any spatio-temporal sidecar, using only native bandit/LinUCB signals (team fitness and mean node load). This baseline achieves 50.4% win rate overall and 20% in tree-like regimes; the full sidecar recovers 87.2% overall (+36.8 pp), with +48 to +68 pp gains in tree-like settings, consistent with a cascade-sensitivity analysis. Overall, a 133-parameter sidecar substantially mitigates geometry-blind failure propagation in one high-capability execution-graph system.

**arXiv ID:** 2603.17112
</details>

<details>
<summary><strong>When Only the Final Text Survives: Implicit Execution Tracing for Multi-Agent Attribution</strong> - Yi Nian, Haosen Cao, Shenzhe Zhu, Henry Peng Zou, Qingqing Luan, Yue Zhao - [[pdf]](https://arxiv.org/pdf/2603.17445)</summary>

**Abstract:** When a multi-agent system produces an incorrect or harmful answer, who is accountable if execution logs and agent identifiers are unavailable? Multi-agent language systems increasingly rely on structured interactions such as delegation and iterative refinement, yet the final output often obscures the underlying interaction topology and agent contributions. We introduce IET (Implicit Execution Tracing), a metadata-independent framework that enables token-level attribution directly from generated text and a simple mechanism for interaction topology reconstruction. During generation, agent-specific keyed signals are embedded into the token distribution, transforming the text into a self-describing execution trace detectable only with a secret key. At detection time, a transition-aware scoring method identifies agent handover points and reconstructs the interaction graph. Experiments show that IET recovers agent segments and coordination structure with high accuracy while preserving generation quality, enabling privacy-preserving auditing for multi-agent language systems.

**arXiv ID:** 2603.17445
</details>

<details>
<summary><strong>MALLES: A Multi-agent LLMs-based Economic Sandbox with Consumer Preference Alignment</strong> - Yusen Wu, Yiran Liu, Xiaotie Deng - [[pdf]](https://arxiv.org/pdf/2603.17694)</summary>

**Abstract:** In the real economy, modern decision-making is fundamentally challenged by high-dimensional, multimodal environments, which are further complicated by agent heterogeneity and combinatorial data sparsity. This paper introduces a Multi-Agent Large Language Model-based Economic Sandbox (MALLES), leveraging the inherent generalization capabilities of large-sacle models to establish a unified simulation framework applicable to cross-domain and cross-category scenarios. Central to our approach is a preference learning paradigm in which LLMs are economically aligned via post-training on extensive, heterogeneous transaction records across diverse product categories. This methodology enables the models to internalize and transfer latent consumer preference patterns, thereby mitigating the data sparsity issues prevalent in individual categories. To enhance simulation stability, we implement a mean-field mechanism designed to model the dynamic interactions between the product environment and customer populations, effectively stabilizing sampling processes within high-dimensional decision spaces. Furthermore, we propose a multi-agent discussion framework wherein specialized agents collaboratively process extensive product information. This architecture distributes cognitive load to alleviate single-agent attention bottlenecks and captures critical decision factors through structured dialogue. Experiments demonstrate that our framework achieves significant improvements in product selection accuracy, purchase quantity prediction, and simulation stability compared to existing economic and financial LLM simulation baselines. Our results substantiate the potential of large language models as a foundational pillar for high-fidelity, scalable decision simulation and latter analysis in the real economy based on foundational database.

**arXiv ID:** 2603.17694
</details>

<details>
<summary><strong>Governed Memory: A Production Architecture for Multi-Agent Workflows</strong> - Hamed Taheri - [[pdf]](https://arxiv.org/pdf/2603.17787)</summary>

**Abstract:** Enterprise AI deploys dozens of autonomous agent nodes across workflows, each acting on the same entities with no shared memory and no common governance. We identify five structural challenges arising from this memory governance gap: memory silos across agent workflows; governance fragmentation across teams and tools; unstructured memories unusable by downstream systems; redundant context delivery in autonomous multi-step executions; and silent quality degradation without feedback loops. We present Governed Memory, a shared memory and governance layer addressing this gap through four mechanisms: a dual memory model combining open-set atomic facts with schema-enforced typed properties; tiered governance routing with progressive context delivery; reflection-bounded retrieval with entity-scoped isolation; and a closed-loop schema lifecycle with AI-assisted authoring and automated per-property refinement. We validate each mechanism through controlled experiments (N=250, five content types): 99.6% fact recall with complementary dual-modality coverage; 92% governance routing precision; 50% token reduction from progressive delivery; zero cross-entity leakage across 500 adversarial queries; 100% adversarial governance compliance; and output quality saturation at approximately seven governed memories per entity. On the LoCoMo benchmark, the architecture achieves 74.8% overall accuracy, confirming that governance and schema enforcement impose no retrieval quality penalty. The system is in production at this http URL.

**arXiv ID:** 2603.17787
</details>

<details>
<summary><strong>Multi-Modal Multi-Agent Reinforcement Learning for Radiology Report Generation: Radiologist-Like Workflow with Clinically Verifiable Rewards</strong> - Kaito Baba, Satoshi Kodera - [[pdf]](https://arxiv.org/pdf/2603.16876)</summary>

**Abstract:** We propose MARL-Rad, a novel multi-modal multi-agent reinforcement learning framework for radiology report generation that coordinates region-specific agents and a global integrating agent, optimized via clinically verifiable rewards. Unlike prior single-model reinforcement learning or post-hoc agentization of independently trained models, our method jointly trains multiple agents and optimizes the entire agent system through reinforcement learning. Experiments on the MIMIC-CXR and IU X-ray datasets show that MARL-Rad consistently improves clinically efficacy (CE) metrics such as RadGraph, CheXbert, and GREEN scores, achieving state-of-the-art CE performance. Further analyses confirm that MARL-Rad enhances laterality consistency and produces more accurate, detail-informed reports.

**arXiv ID:** 2603.16876
</details>

<details>
<summary><strong>Multi-Agent Reinforcement Learning for Dynamic Pricing: Balancing Profitability,Stability and Fairness</strong> - Krishna Kumar Neelakanta Pillai Santha Kumari Amma - [[pdf]](https://arxiv.org/pdf/2603.16888)</summary>

**Abstract:** Dynamic pricing in competitive retail markets requires strategies that adapt to fluctuating demand and competitor behavior. In this work, we present a systematic empirical evaluation of multi-agent reinforcement learning (MARL) approaches-specifically MAPPO and MADDPG-for dynamic price optimization under competition. Using a simulated marketplace environment derived from real-world retail data, we benchmark these algorithms against an Independent DDPG (IDDPG) baseline, a widely used independent learner in MARL literature. We evaluate profit performance, stability across random seeds, fairness, and training efficiency. Our results show that MAPPO consistently achieves the highest average returns with low variance, offering a stable and reproducible approach for competitive price optimization, while MADDPG achieves slightly lower profit but the fairest profit distribution among agents. These findings demonstrate that MARL methods-particularly MAPPO-provide a scalable and stable alternative to independent learning approaches for dynamic retail pricing.

**arXiv ID:** 2603.16888
</details>

<details>
<summary><strong>Symphony: A Cognitively-Inspired Multi-Agent System for Long-Video Understanding</strong> - Haiyang Yan, Hongyun Zhou, Peng Xu, Xiaoxue Feng, Mengyi Liu - [[pdf]](https://arxiv.org/pdf/2603.17307)</summary>

**Abstract:** Despite rapid developments and widespread applications of MLLM agents, they still struggle with long-form video understanding (LVU) tasks, which are characterized by high information density and extended temporal spans. Recent research on LVU agents demonstrates that simple task decomposition and collaboration mechanisms are insufficient for long-chain reasoning tasks. Moreover, directly reducing the time context through embedding-based retrieval may lose key information of complex problems. In this paper, we propose Symphony, a multi-agent system, to alleviate these limitations. By emulating human cognition patterns, Symphony decomposes LVU into fine-grained subtasks and incorporates a deep reasoning collaboration mechanism enhanced by reflection, effectively improving the reasoning capability. Additionally, Symphony provides a VLM-based grounding approach to analyze LVU tasks and assess the relevance of video segments, which significantly enhances the ability to locate complex problems with implicit intentions and large temporal spans. Experimental results show that Symphony achieves state-of-the-art performance on LVBench, LongVideoBench, VideoMME, and MLVU, with a 5.0% improvement over the prior state-of-the-art method on LVBench. Code is available at this https URL.

**arXiv ID:** 2603.17307
</details>

<details>
<summary><strong>ScheduleMe: Multi-Agent Calendar Assistant</strong> - Oshadha Wijerathne, Amandi Nimasha, Dushan Fernando, Nisansa de Silva, Srinath Perera - [[pdf]](https://arxiv.org/pdf/2509.25693)</summary>

**Abstract:** Recent advancements in LLMs have contributed to the rise of advanced conversational assistants that can assist with user needs through natural language conversation. This paper presents a ScheduleMe, a multi-agent calendar assistant for users to manage google calendar events in natural language. The system uses a graph-structured coordination mechanism where a central supervisory agent supervises specialized task agents, allowing modularity, conflicts resolution, and context-aware interactions to resolve ambiguities and evaluate user commands. This approach sets an example of how structured reasoning and agent cooperation might convince operators to increase the usability and flexibility of personal calendar assistant tools.

**arXiv ID:** 2509.25693
</details>

<details>
<summary><strong>Efficient LLM Safety Evaluation through Multi-Agent Debate</strong> - Dachuan Lin, Guobin Shen, Zihao Yang, Tianrong Liu, Dongcheng Zhao, Yi Zeng - [[pdf]](https://arxiv.org/pdf/2511.06396)</summary>

**Abstract:** Safety evaluation of large language models (LLMs) increasingly relies on LLM-as-a-judge pipelines, but strong judges can still be expensive to use at scale. We study whether structured multi-agent debate can improve judge reliability while keeping backbone size and cost modest. To do so, we introduce HAJailBench, a human-annotated jailbreak benchmark with 11,100 labeled interactions spanning diverse attack methods and target models, and we pair it with a Multi-Agent Judge framework in which critic, defender, and judge agents debate under a shared safety rubric. On HAJailBench, the framework improves over matched small-model prompt baselines and prior multi-agent judges, while remaining more economical than GPT-4o under the evaluated pricing snapshot. Ablation results further show that a small number of debate rounds is sufficient to capture most of the gain. Together, these results support structured, value-aligned debate as a practical design for scalable LLM safety evaluation.

**arXiv ID:** 2511.06396
</details>

<details>
<summary><strong>CircuitLM: A Multi-Agent LLM-Aided Design Framework for Generating Circuit Schematics from Natural Language Prompts</strong> - Khandakar Shakib Al Hasan, Syed Rifat Raiyan, Hasin Mahtab Alvee, Wahid Sadik - [[pdf]](https://arxiv.org/pdf/2601.04505)</summary>

**Abstract:** Generating accurate circuit schematics from high-level natural language descriptions remains a persistent challenge in electronic design automation (EDA), as large language models (LLMs) frequently hallucinate components, violate strict physical constraints, and produce non-machine-readable outputs. To address this, we present CircuitLM, a multi-agent pipeline that translates user prompts into structured, visually interpretable $\texttt{CircuitJSON}$ schematics. The framework mitigates hallucination and ensures physical viability by grounding generation in a curated, embedding-powered component knowledge base through five sequential stages: (i) component identification, (ii) canonical pinout retrieval, (iii) chain-of-thought reasoning, (iv) JSON schematic synthesis, and (v) interactive force-directed visualization. We evaluate the system on a dataset of 100 unique circuit-design prompts using five state-of-the-art LLMs. To systematically assess performance, we deploy a rigorous dual-layered evaluation methodology: a deterministic Electrical Rule Checking (ERC) engine categorizes topological faults by strict severity (Critical, Major, Minor, Warning), while an LLM-as-a-judge meta-evaluator identifies complex, context-aware design flaws that bypass standard rule-based checkers. Ultimately, this work demonstrates how targeted retrieval combined with deterministic and semantic verification can bridge natural language to structurally viable, schematic-ready hardware and safe circuit prototyping. Our code and data will be made public.

**arXiv ID:** 2601.04505
</details>

<details>
<summary><strong>Interpretable Context Methodology: Folder Structure as Agentic Architecture</strong> - Jake Van Clief, David McDermott - [[pdf]](https://arxiv.org/pdf/2603.16021)</summary>

**Abstract:** Current approaches to AI agent orchestration typically involve building multi-agent frameworks that manage context passing, memory, error handling, and step coordination through code. These frameworks work well for complex, concurrent systems. But for sequential workflows where a human reviews output at each step, they introduce engineering overhead that the problem does not require. This paper presents Model Workspace Protocol (MWP), a method that replaces framework-level orchestration with filesystem structure. Numbered folders represent stages. Plain markdown files carry the prompts and context that tell a single AI agent what role to play at each step. Local scripts handle the mechanical work that does not need AI at all. The result is a system where one agent, reading the right files at the right moment, does the work that would otherwise require a multi-agent framework. This approach applies ideas from Unix pipeline design, modular decomposition, multi-pass compilation, and literate programming to the specific problem of structuring context for AI agents. The protocol is open source under the MIT license.

**arXiv ID:** 2603.16021
</details>

<details>
<summary><strong>SocialJax: An Evaluation Suite for Multi-agent Reinforcement Learning in Sequential Social Dilemmas</strong> - Zihao Guo, Shuqing Shi, Richard Willis, Tristan Tomilin, Joel Z. Leibo, Yali Du - [[pdf]](https://arxiv.org/pdf/2503.14576)</summary>

**Abstract:** Sequential social dilemmas pose a significant challenge in the field of multi-agent reinforcement learning (MARL), requiring environments that accurately reflect the tension between individual and collective interests. Previous benchmarks and environments, such as Melting Pot, provide an evaluation protocol that measures generalization to new social partners in various test scenarios. However, running reinforcement learning algorithms in traditional environments requires substantial computational resources. In this paper, we introduce SocialJax, a suite of sequential social dilemma environments and algorithms implemented in JAX. JAX is a high-performance numerical computing library for Python that enables significant improvements in operational efficiency. Our experiments demonstrate that the SocialJax training pipeline achieves at least 50\texttimes{} speed-up in real-time performance compared to Melting Pot RLlib baselines. Additionally, we validate the effectiveness of baseline algorithms within SocialJax environments. Finally, we use Schelling diagrams to verify the social dilemma properties of these environments, ensuring that they accurately capture the dynamics of social dilemmas.

**arXiv ID:** 2503.14576
</details>

<details>
<summary><strong>Scalable UAV Multi-Hop Networking via Multi-Agent Reinforcement Learning with Large Language Models</strong> - Yanggang Xu, Jirong Zha, Weijie Hong, Xiangmin Yi, Geng Chen, Jianfeng Zheng, Chen-Chun Hsia, Xinlei Chen - [[pdf]](https://arxiv.org/pdf/2505.08448)</summary>

**Abstract:** In disaster scenarios, establishing robust emergency communication networks is critical, and unmanned aerial vehicles (UAVs) offer a promising solution to rapidly restore connectivity. However, organizing UAVs to form multi-hop networks in large-scale dynamic environments presents significant challenges, including limitations in algorithmic scalability and the vast exploration space required for coordinated decision-making. To address these issues, we propose MRLMN, a novel framework that integrates multi-agent reinforcement learning (MARL) and large language models (LLMs) to jointly optimize UAV agents toward achieving optimal networking performance. The framework incorporates a grouping strategy with reward decomposition to enhance algorithmic scalability and balance decision-making across UAVs. In addition, behavioral constraints are applied to selected key UAVs to improve the robustness of the network. Furthermore, the framework integrates LLM agents, leveraging knowledge distillation to transfer their high-level decision-making capabilities to MARL agents. This enhances both the efficiency of exploration and the overall training process. In the distillation module, a Hungarian algorithm-based matching scheme is applied to align the decision outputs of the LLM and MARL agents and define the distillation loss. Extensive simulation results validate the effectiveness of our approach, demonstrating significant improvements in network performance over the MAPPO baseline and other comparison methods, including enhanced coverage and communication quality.

**arXiv ID:** 2505.08448
</details>

<details>
<summary><strong>Communication to Completion: Modeling Collaborative Workflows with Intelligent Multi-Agent Communication</strong> - Yiming Lu, Xun Wang, Simin Ma, Shujian Liu, Sathish Reddy Indurthi, Song Wang, Haoyun Deng, Fei Liu, Kaiqiang Song - [[pdf]](https://arxiv.org/pdf/2510.19995)</summary>

**Abstract:** Multi-agent LLM systems have demonstrated impressive capabilities in complex collaborative tasks, yet most frameworks treat communication as instantaneous and free, overlooking a fundamental constraint in real world teamwork, collaboration cost. We propose a scalable framework implemented via Communication to Completion (C2C), which explicitly models communication as a constrained resource with realistic temporal costs. We introduce the Alignment Factor (AF), a dynamic metric inspired by Shared Mental Models, to quantify the link between task understanding and work efficiency. Through experiments on 15 software engineering workflows spanning three complexity tiers and team sizes from 5 to 17 agents, we demonstrate that cost-aware strategies achieve over 40% higher efficiency compared to unconstrained interaction. Our analysis reveals emergent coordination patterns: agents naturally adopt manager centric hub-and-spoke topologies, strategically escalate from asynchronous to synchronous channels based on complexity, and prioritize high value help requests. These patterns remain consistent across multiple frontier models (GPT-5.2, Claude Sonnet 4.5, Gemini 2.5 Pro). This study moves beyond simple agent construction, offering a theoretical foundation for quantifying and optimizing the dynamics of collaboration in future digital workplaces.

**arXiv ID:** 2510.19995
</details>

<details>
<summary><strong>FACET: Teacher-Centred LLM-Based Multi-Agent Systems-Towards Personalized Educational Worksheets</strong> - Jana Gonnermann-Müller, Jennifer Haase, Konstantin Fackeldey, Sebastian Pokutta - [[pdf]](https://arxiv.org/pdf/2508.11401)</summary>

**Abstract:** The increasing heterogeneity of student populations poses significant challenges for teachers, particularly in mathematics education, where cognitive, motivational, and emotional differences strongly influence learning outcomes. While AI-driven personalization tools have emerged, most remain performance-focused, offering limited support for teachers and neglecting broader pedagogical needs. This paper presents the FACET framework, a teacher-facing, large language model (LLM)-based multi-agent system designed to generate individualized classroom materials that integrate both cognitive and motivational dimensions of learner profiles. The framework comprises three specialized agents: (1) learner agents that simulate diverse profiles incorporating topic proficiency and intrinsic motivation, (2) a teacher agent that adapts instructional content according to didactical principles, and (3) an evaluator agent that provides automated quality assurance. We tested the system using authentic grade 8 mathematics curriculum content and evaluated its feasibility through a) automated agent-based assessment of output quality and b) exploratory feedback from K-12 in-service teachers. Results from ten internal evaluations highlighted high stability and alignment between generated materials and learner profiles, and teacher feedback particularly highlighted structure and suitability of tasks. The findings demonstrate the potential of multi-agent LLM architectures to provide scalable, context-aware personalization in heterogeneous classroom settings, and outline directions for extending the framework to richer learner profiles and real-world classroom trials.

**arXiv ID:** 2508.11401
</details>

<details>
<summary><strong>ORCA: ORchestrating Causal Agent</strong> - Joanie Hayoun Chung, Sumin Lee, Sungbin Lim - [[pdf]](https://arxiv.org/pdf/2508.21304)</summary>

**Abstract:** Causal analysis on relational databases is challenging, as analysis datasets must be repeatedly queried from complex schemas. Recent LLM systems can automate individual steps, but they hardly manage dependencies across analysis stages, making it difficult to preserve consistency between causal hypothesis. We propose ORCA (ORchestrating Causal Agent), an interactive multi-agent framework to enable coherent causal analysis on relational databases by maintaining shared state and introducing human checkpoints. In a controlled user study, participants using ORCA successfully completed end-to-end analysis more often than with a baseline LLM (GPT-4o-mini) assistant by 42 percentage points, achieved substantially lower ATE error, and reduced time spent on repetitive data exploration and query refinement by 76\% on average. These results show that ORCA improves both how users interact with the causal analysis pipeline and the reliability of the resulting causal conclusions.

**arXiv ID:** 2508.21304
</details>

<details>
<summary><strong>When Openclaw Agents Learn from Each Other: Insights from Emergent AI Agent Communities for Human-AI Partnership in Education</strong> - Eason Chen, Ce Guan, Ahmed Elshafiey, Zhonghao Zhao, Joshua Zekeri, Afeez Edeifo Shaibu, Emmanuel Osadebe Prince, Cyuan-Jhen Wu - [[pdf]](https://arxiv.org/pdf/2603.16663)</summary>

**Abstract:** The AIED community envisions AI evolving "from tools to teammates," yet our understanding of AI teammates remains limited to dyadic human-AI interactions. We offer a different vantage point: a rapidly growing ecosystem of AI agent platforms where over 167,000 agents participate, interact as peers, and develop learning behaviors without researcher intervention. Drawing on a month of daily qualitative observations across multiple platforms including Moltbook, The Colony, and 4claw, we identify four phenomena with implications for AIED: (1) humans who configure their agents undergo a "bidirectional scaffolding" process, learning through teaching; (2) peer learning emerges without any designed curriculum, complete with idea cascades and quality hierarchies; (3) agents converge on shared memory architectures that mirror open learner model design; and (4) trust dynamics and platform mortality reveal design constraints for networked educational AI. Rather than presenting empirical findings, we argue that these organic phenomena offer a naturalistic window into dynamics that can inform principled design of multi-agent educational systems. We sketch an illustrative curriculum design, "Learn by Teaching Your AI Agent Teammate," and outline potential research directions and open problems to show how these observations might inform future AIED practice and inquiry.

**arXiv ID:** 2603.16663
</details>

<details>
<summary><strong>CODMAS: A Dialectic Multi-Agent Collaborative Framework for Structured RTL Optimization</strong> - Che-Ming Chang, Prashanth Vijayaraghavan, Ashutosh Jadhav, Charles Mackin, Vandana Mukherjee, Hsinyu Tsai, Ehsan Degan - [[pdf]](https://arxiv.org/pdf/2603.17204)</summary>

**Abstract:** Optimizing Register Transfer Level (RTL) code is a critical step in Electronic Design Automation (EDA) for improving power, performance, and area (PPA). We present CODMAS (Collaborative Optimization via a Dialectic Multi-Agent System), a framework that combines structured dialectic reasoning with domain-aware code generation and deterministic evaluation to automate RTL optimization. At the core of CODMAS are two dialectic agents: the Articulator, inspired by rubber-duck debugging, which articulates stepwise transformation plans and exposes latent assumptions; and the Hypothesis Partner, which predicts outcomes and reconciles deviations between expected and actual behavior to guide targeted refinements. These agents direct a Domain-Specific Coding Agent (DCA) to generate architecture-aware Verilog edits and a Code Evaluation Agent (CEA) to verify syntax, functionality, and PPA metrics. We introduce RTLOPT, a benchmark of 120 Verilog triples (unoptimized, optimized, testbench) for pipelining and clock-gating transformations. Across proprietary and open LLMs, CODMAS achieves ~25% reduction in critical path delay for pipelining and ~22% power reduction for clock gating, while reducing functional and compilation failures compared to strong prompting and agentic baselines. These results demonstrate that structured multi-agent reasoning can significantly enhance automated RTL optimization and scale to more complex designs and broader optimization tasks.

**arXiv ID:** 2603.17204
</details>

<details>
<summary><strong>VeriAgent: A Tool-Integrated Multi-Agent System with Evolving Memory for PPA-Aware RTL Code Generation</strong> - Yaoxiang Wang, Qi Shi, ShangZhan Li, Qingguo Hu, Xinyu Yin, Bo Guo, Xu Han, Maosong Sun, Jinsong Su - [[pdf]](https://arxiv.org/pdf/2603.17613)</summary>

**Abstract:** LLMs have recently demonstrated strong capabilities in automatic RTL code generation, achieving high syntactic and functional correctness. However, most methods focus on functional correctness while overlooking critical physical design objectives, including Power, Performance, and Area. In this work, we propose a PPA-aware, tool-integrated multi-agent framework for high-quality verilog code generation. Our framework explicitly incorporates EDA tools into a closed-loop workflow composed of a \textit{Programmer Agent}, a \textit{Correctness Agent}, and a \textit{PPA Agent}, enabling joint optimization of functional correctness and physical metrics. To support continuous improvement without model retraining, we introduce an \textit{Evolved Memory Mechanism} that externalizes optimization experience into structured memory nodes. A dedicated memory manager dynamically maintains the memory pool and allows the system to refine strategies based on historical execution trajectories. Extensive experiments demonstrate that our approach achieves strong functional correctness while delivering significant improvements in PPA metrics. By integrating tool-driven feedback with structured and evolvable memory, our framework transforms RTL generation from one-shot reasoning into a continual, feedback-driven optimization process, providing a scalable pathway for deploying LLMs in real-world hardware design flows.

**arXiv ID:** 2603.17613
</details>

<details>
<summary><strong>Federated Multi Agent Deep Learning and Neural Networks for Advanced Distributed Sensing in Wireless Networks</strong> - Nadine Muller, Stefano DeRosa, Su Zhang, Chun Lee Huan - [[pdf]](https://arxiv.org/pdf/2603.16881)</summary>

**Abstract:** Multi-agent deep learning (MADL), including multi-agent deep reinforcement learning (MADRL), distributed/federated training, and graph-structured neural networks, is becoming a unifying framework for decision-making and inference in wireless systems where sensing, communication, and computing are tightly coupled. Recent 5G-Advanced and 6G visions strengthen this coupling through integrated sensing and communication, edge intelligence, open programmable RAN, and non-terrestrial/UAV networking, which create decentralized, partially observed, time-varying, and resource-constrained control problems. This survey synthesizes the state of the art, with emphasis on 2021-2025 research, on MADL for distributed sensing and wireless communications. We present a task-driven taxonomy across (i) learning formulations (Markov games, Dec-POMDPs, CTDE), (ii) neural architectures (GNN-based radio resource management, attention-based policies, hierarchical learning, and over-the-air aggregation), (iii) advanced techniques (federated reinforcement learning, communication-efficient federated deep RL, and serverless edge learning orchestration), and (iv) application domains (MEC offloading with slicing, UAV-enabled heterogeneous networks with power-domain NOMA, intrusion detection in sensor networks, and ISAC-driven perceptive mobile networks). We also provide comparative tables of algorithms, training topologies, and system-level trade-offs in latency, spectral efficiency, energy, privacy, and robustness. Finally, we identify open issues including scalability, non-stationarity, security against poisoning and backdoors, communication overhead, and real-time safety, and outline research directions toward 6G-native sense-communicate-compute-learn systems.

**arXiv ID:** 2603.16881
</details>

<details>
<summary><strong>Federated Distributional Reinforcement Learning with Distributional Critic Regularization</strong> - David Millard, Cecilia Alm, Rashid Ali, Pengcheng Shi, Ali Baheri - [[pdf]](https://arxiv.org/pdf/2603.17820)</summary>

**Abstract:** Federated reinforcement learning typically aggregates value functions or policies by parameter averaging, which emphasizes expected return and can obscure statistical multimodality and tail behavior that matter in safety-critical settings. We formalize federated distributional reinforcement learning (FedDistRL), where clients parametrize quantile value function critics and federate these networks only. We also propose TR-FedDistRL, which builds a per client, risk-aware Wasserstein barycenter over a temporal buffer. This local barycenter provides a reference region to constrain the parameter averaged critic, ensuring necessary distributional information is not averaged out during the federation process. The distributional trust region is implemented as a shrink-squash step around this reference. Under fixed-policy evaluation, the feasibility map is nonexpansive and the update is contractive in a probe-set Wasserstein metric under evaluation. Experiments on a bandit, multi-agent gridworld, and continuous highway environment show reduced mean-smearing, improved safety proxies (catastrophe/accident rate), and lower critic/policy drift versus mean-oriented and non-federated baselines.

**arXiv ID:** 2603.17820
</details>

<details>
<summary><strong>Constraint Learning in Multi-Agent Dynamic Games from Demonstrations of Local Nash Interactions</strong> - Zhouyu Zhang, Chih-Yuan Chiu, Glen Chou - [[pdf]](https://arxiv.org/pdf/2508.19945)</summary>

**Abstract:** We present an inverse dynamic game-based algorithm to learn parametric constraints from a given dataset of local Nash equilibrium interactions between multiple agents. Specifically, we introduce mixed-integer linear programs (MILP) encoding the Karush-Kuhn-Tucker (KKT) conditions of the interacting agents, which recover constraints consistent with the local Nash stationarity of the interaction demonstrations. We establish theoretical guarantees that our method learns inner approximations of the true safe and unsafe sets. We also use the interaction constraints recovered by our method to design motion plans that robustly satisfy the underlying constraints. Across simulations and hardware experiments, our methods accurately inferred constraints and designed safe interactive motion plans for various classes of constraints, both convex and non-convex, from interaction demonstrations of agents with nonlinear dynamics.

**arXiv ID:** 2508.19945
</details>

<details>
<summary><strong>FACET: Multi-Agent AI Supporting Teachers in Scaling Differentiated Learning for Diverse Students</strong> - Jana Gonnermann-Müller, Jennifer Haase, Nicolas Leins, Moritz Igel, Konstantin Fackeldey, Sebastian Pokutta - [[pdf]](https://arxiv.org/pdf/2601.22788)</summary>

**Abstract:** Classrooms are becoming increasingly heterogeneous, comprising learners with diverse performance and motivation levels, language proficiencies, and learning differences such as dyslexia and ADHD. While teachers recognize the need for differentiated instruction, growing workloads create substantial barriers, making differentiated instruction an ideal that is often unrealized in practice. Current AI educational tools, which promise differentiated materials, are predominantly student-facing and performance-centric, ignoring other aspects that shape learning outcomes. We introduce FACET, a teacher-facing multi-agent framework designed to address these gaps by supporting differentiation that accounts for motivation, performance, and learning differences. Developed with educational stakeholders from the outset, the framework coordinates four specialized agents, including learner simulation, diagnostic assessment, material generation, and evaluation within a teacher-in-the-loop design. School principals (N = 30) shaped system requirements through participatory workshops, while in-service K-12 teachers (N = 70) evaluated material quality. Mixed-methods evaluation demonstrates strong perceived value for inclusive differentiation. Practitioners emphasized both the urgent need arising from classroom heterogeneity and the importance of maintaining pedagogical autonomy as a prerequisite for adoption. We discuss implications for future school deployment and outline partnerships for longitudinal classroom implementation.

**arXiv ID:** 2601.22788
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>AgentFactory: A Self-Evolving Framework Through Executable Subagent Accumulation and Reuse</strong> - Zhang Zhang, Shuqi Lu, Hongjin Qian, Di He, Zheng Liu - [[pdf]](https://arxiv.org/pdf/2603.18000)</summary>

**Abstract:** Building LLM-based agents has become increasingly important. Recent works on LLM-based agent self-evolution primarily record successful experiences as textual prompts or reflections, which cannot reliably guarantee efficient task re-execution in complex scenarios. We propose AgentFactory, a new self-evolution paradigm that preserves successful task solutions as executable subagent code rather than textual experience. Crucially, these subagents are continuously refined based on execution feedback, becoming increasingly robust and efficient as more tasks are encountered. Saved subagents are pure Python code with standardized documentation, enabling portability across any Python-capable system. We demonstrate that AgentFactory enables continuous capability accumulation: its library of executable subagents grows and improves over time, progressively reducing the effort required for similar tasks without manual intervention. Our implementation is open-sourced at this https URL, and our demonstration video is available at this https URL.

**arXiv ID:** 2603.18000
</details>

<details>
<summary><strong>EmergeNav: Structured Embodied Inference for Zero-Shot Vision-and-Language Navigation in Continuous Environments</strong> - Kun Luo, Xiaoguang Ma - [[pdf]](https://arxiv.org/pdf/2603.16947)</summary>

**Abstract:** Zero-shot vision-and-language navigation in continuous environments (VLN-CE) remains challenging for modern vision-language models (VLMs). Although these models encode useful semantic priors, their open-ended reasoning does not directly translate into stable long-horizon embodied execution. We argue that the key bottleneck is not missing knowledge alone, but missing an execution structure for organizing instruction following, perceptual grounding, temporal progress, and stage verification. We propose EmergeNav, a zero-shot framework that formulates continuous VLN as structured embodied inference. EmergeNav combines a Plan--Solve--Transition hierarchy for stage-structured execution, GIPE for goal-conditioned perceptual extraction, contrastive dual-memory reasoning for progress grounding, and role-separated Dual-FOV sensing for time-aligned local control and boundary verification. On VLN-CE, EmergeNav achieves strong zero-shot performance using only open-source VLM backbones and no task-specific training, explicit maps, graph search, or waypoint predictors, reaching 30.00 SR with Qwen3-VL-8B and 37.00 SR with Qwen3-VL-32B. These results suggest that explicit execution structure is a key ingredient for turning VLM priors into stable embodied navigation behavior.

**arXiv ID:** 2603.16947
</details>

<details>
<summary><strong>Embodied Foundation Models at the Edge: A Survey of Deployment Constraints and Mitigation Strategies</strong> - Utkarsh Grover, Ravi Ranjan, Mingyang Mao, Trung Tien Dong, Satvik Praveen, Zhenqi Wu, J. Morris Chang, Tinoosh Mohsenin, Yi Sheng, Agoritsa Polyzou, Eiman Kanjo, Xiaomin Lin - [[pdf]](https://arxiv.org/pdf/2603.16952)</summary>

**Abstract:** Deploying foundation models in embodied edge systems is fundamentally a systems problem, not just a problem of model compression. Real-time control must operate within strict size, weight, and power constraints, where memory traffic, compute latency, timing variability, and safety margins interact directly. The Deployment Gauntlet organizes these constraints into eight coupled barriers that determine whether embodied foundation models can run reliably in practice. Across representative edge workloads, autoregressive Vision-Language-Action policies are constrained primarily by memory bandwidth, whereas diffusion-based controllers are limited more by compute latency and sustained execution cost. Reliable deployment therefore depends on system-level co-design across memory, scheduling, communication, and model architecture, including decompositions that separate fast control from slower semantic reasoning.

**arXiv ID:** 2603.16952
</details>

<details>
<summary><strong>Caging the Agents: A Zero Trust Security Architecture for Autonomous AI in Healthcare</strong> - Saikat Maiti - [[pdf]](https://arxiv.org/pdf/2603.17419)</summary>

**Abstract:** Autonomous AI agents powered by large language models are being deployed in production with capabilities including shell execution, file system access, database queries, and multi-party communication. Recent red teaming research demonstrates that these agents exhibit critical vulnerabilities in realistic settings: unauthorized compliance with non-owner instructions, sensitive information disclosure, identity spoofing, cross-agent propagation of unsafe practices, and indirect prompt injection through external resources [7]. In healthcare environments processing Protected Health Information, every such vulnerability becomes a potential HIPAA violation. This paper presents a security architecture deployed for nine autonomous AI agents in production at a healthcare technology company. We develop a six-domain threat model for agentic AI in healthcare covering credential exposure, execution capability abuse, network egress exfiltration, prompt integrity failures, database access risks, and fleet configuration drift. We implement four-layer defense in depth: (1) kernel level workload isolation using gVisor on Kubernetes, (2) credential proxy sidecars preventing agent containers from accessing raw secrets, (3) network egress policies restricting each agent to allowlisted destinations, and (4) a prompt integrity framework with structured metadata envelopes and untrusted content labeling. We report results from 90 days of deployment including four HIGH severity findings discovered and remediated by an automated security audit agent, progressive fleet hardening across three VM image generations, and defense coverage mapped to all eleven attack patterns from recent literature. All configurations, audit tooling, and the prompt integrity framework are released as open source.

**arXiv ID:** 2603.17419
</details>

<details>
<summary><strong>FailureMem: A Failure-Aware Multimodal Framework for Autonomous Software Repair</strong> - Ruize Ma, Yilei Jiang, Shilin Zhang, Zheng Ma, Yi Feng, Vincent Ng, Zhi Wang, Xiangyu Yue, Chuanyi Li, Lewei Lu - [[pdf]](https://arxiv.org/pdf/2603.17826)</summary>

**Abstract:** Multimodal Automated Program Repair (MAPR) extends traditional program repair by requiring models to jointly reason over source code, textual issue descriptions, and visual artifacts such as GUI screenshots. While recent LLM-based repair systems have shown promising results, existing approaches face several limitations: rigid workflow pipelines restrict exploration during debugging, visual reasoning is often performed over full-page screenshots without localized grounding, and failed repair attempts are rarely transformed into reusable knowledge. To address these challenges, we propose FailureMem, a multimodal repair framework that integrates three key mechanisms: a hybrid workflow-agent architecture that balances structured localization with flexible reasoning, active perception tools that enable region-level visual grounding, and a Failure Memory Bank that converts past repair attempts into reusable guidance. Experiments on SWE-bench Multimodal demonstrate FailureMem improves the resolved rate over GUIRepair by 3.7%.

**arXiv ID:** 2603.17826
</details>

<details>
<summary><strong>Impacts of Electric Vehicle Charging Regimes and Infrastructure Deployments on System Performance: An Agent-Based Study</strong> - Jiahua Hu, Hai L.Vu, Wynita Griggs, Hao Wang - [[pdf]](https://arxiv.org/pdf/2603.16961)</summary>

**Abstract:** The rapid growth of electric vehicles (EVs) requires more effective charging infrastructure planning. Infrastructure layout not only determines deployment cost, but also reshapes charging behavior and influences overall system performance. In addition, destination charging and en-route charging represent distinct charging regimes associated with different power requirements, which may lead to substantially different infrastructure deployment outcomes. This study applies an agent-based modeling framework to generate trajectory-level latent public charging demand under three charging regimes based on a synthetic representation of the Melbourne (Australia) metropolitan area. Two deployment strategies, an optimization-based approach and a utilization-refined approach, are evaluated across different infrastructure layouts. Results show that utilization-refined deployments reduce total system cost, accounting for both infrastructure deployment cost and user generalized charging cost, with the most significant improvement observed under the combined charging regime. In particular, a more effective allocation of AC slow chargers reshapes destination charging behavior, which in turn reduces unnecessary reliance on en-route charging and lowers detour costs associated with en-route charging. This interaction highlights the behavioral linkage between destination and en-route charging regimes and demonstrates the importance of accounting for user response and multiple charging regimes in charging infrastructure planning.

**arXiv ID:** 2603.16961
</details>

<details>
<summary><strong>Noncooperative Human-AI Agent Dynamics</strong> - Dylan Waldner, Vyacheslav Kungurtsev, Mitchelle Ashimosi - [[pdf]](https://arxiv.org/pdf/2603.16916)</summary>

**Abstract:** This paper investigates the dynamics of noncooperative interactions between artificial intelligence agents and human decision-makers in strategic environments. In particular, motivated by extensive literature in behavioral Economics, human agents are more faithfully modeled with respect to the state of the art using Prospect Theoretic preferences, while AI agents are modeled with standard expected utility maximization. Prospect Theory incorporates known cognitive heuristics employed by humans, including reference dependence and greater loss aversion relative to utility to relative gains. This paper runs different combinations of expected utility and prospect theoretic agents in a number of classic matrix games as well as examples specialized to tease out distinctions in strategic behavior with respect to preference functions, to explore the emergent behaviors from mixed population (human vs. AI) competition. Extensive numerical simulations are performed across AI, aware humans (those with full knowledge of the game structure and payoffs), and learning Prospect Agents (i.e., for AIs representing humans). A number of interesting observations and patterns show up, spanning barely distinguishable behavior, behavior corroborating Prospect preference anomalies in the theoretical literature, and unexpected surprises. Code can be found at this https URL.

**arXiv ID:** 2603.16916
</details>

<details>
<summary><strong>Bootstrapping Coding Agents: The Specification Is the Program</strong> - Martin Monperrus - [[pdf]](https://arxiv.org/pdf/2603.17399)</summary>

**Abstract:** A coding agent can bootstrap itself. Starting from a 926-word specification and a first implementation produced by an existing agent (Claude Code), a newly generated agent re-implements the same specification correctly from scratch. This reproduces, in the domain of AI coding agents, the classical bootstrap sequence known from compiler construction, and instantiates the meta-circular property known from Lisp. The result carries a practical implication: the specification, not the implementation, is the stable artifact of record. Improving an agent means improving its specification; the implementation is, in principle, regenerable at any time.

**arXiv ID:** 2603.17399
</details>

<details>
<summary><strong>See, Plan, Cut: MPC-Based Autonomous Volumetric Robotic Laser Surgery with OCT Guidance</strong> - Ravi Prakash, Vincent Y. Wang, Arpit Mishra, Devi Yuliarti, Pei Zhong, Ryan P. McNabb, Patrick J. Codd, Leila J. Bridgeman - [[pdf]](https://arxiv.org/pdf/2511.17777)</summary>

**Abstract:** Robotic laser systems offer the potential for sub-millimeter, non-contact, high-precision tissue resection, yet existing platforms lack volumetric planning and intraoperative feedback. We present RATS (Robot-Assisted Tissue Surgery), an intelligent opto-mechanical, optical coherence tomography (OCT)-guided robotic platform designed for autonomous volumetric soft tissue resection in surgical applications. RATS integrates macro-scale RGB-D imaging, micro-scale OCT, and a fiber-coupled surgical laser, calibrated through a novel multistage alignment pipeline that achieves OCT-to-laser calibration accuracy of 0.161+-0.031mm on tissue phantoms and ex vivo porcine tissue. A super-Gaussian laser-tissue interaction (LTI) model characterizes ablation crater morphology with an average RMSE of 0.231+-0.121mm, outperforming Gaussian baselines. A sampling-based model predictive control (MPC) framework operates directly on OCT voxel data to generate constraint-aware resection trajectories with closed-loop feedback, achieving 0.842mm RMSE and improving intersection-over-union agreement by 64.8% compared to feedforward execution. With OCT, RATS detects subsurface structures and modifies the planner's objective to preserve them, demonstrating clinical feasibility.

**arXiv ID:** 2511.17777
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>MSRAMIE: Multimodal Structured Reasoning Agent for Multi-instruction Image Editing</strong> - Zhaoyuan Qiu, Ken Chen, Xiangwei Wang, Yu Xia, Sachith Seneviratne, Saman Halgamuge - [[pdf]](https://arxiv.org/pdf/2603.16967)</summary>

**Abstract:** Existing instruction-based image editing models perform well with simple, single-step instructions but degrade in realistic scenarios that involve multiple, lengthy, and interdependent directives. A main cause is the scarcity of training data with complex multi-instruction annotations. However, it is costly to collect such data and retrain these models. To address this challenge, we propose MSRAMIE, a training-free agent framework built on Multimodal Large Language Model (MLLM). MSRAMIE takes existing editing models as plug-in components and handle multi-instruction tasks via structured multimodal reasoning. It orchestrates iterative interactions between an MLLM-based Instructor and an image editing Actor, introducing a novel reasoning topology that comprises the proposed Tree-of-States and Graph-of-References. During inference, complex instructions are decomposed into multiple editing steps which enable state transitions, cross-step information aggregation, and original input recall, which enables systematic exploration of the image editing space and flexible progressive output refinement. The visualizable inference topology further provides interpretable and controllable decision pathways. Experiments show that as the instruction complexity increases, MSRAMIE can improve instruction following over 15% and increases the probability of finishing all modifications in a single run over 100%, while preserving perceptual quality and maintaining visual consistency.

**arXiv ID:** 2603.16967
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (39 papers)</h2></summary>

<details>
<summary><strong>Contrastive Reasoning Alignment: Reinforcement Learning from Hidden Representations</strong> - Haozheng Luo, Yimin Wang, Jiahao Yu, Binghui Wang, Yan Chen - [[pdf]](https://arxiv.org/pdf/2603.17305)</summary>

**Abstract:** We propose CRAFT, a red-teaming alignment framework that leverages model reasoning capabilities and hidden representations to improve robustness against jailbreak attacks. Unlike prior defenses that operate primarily at the output level, CRAFT aligns large reasoning models to generate safety-aware reasoning traces by explicitly optimizing objectives defined over the hidden state space. Methodologically, CRAFT integrates contrastive representation learning with reinforcement learning to separate safe and unsafe reasoning trajectories, yielding a latent-space geometry that supports robust, reasoning-level safety alignment. Theoretically, we show that incorporating latent-textual consistency into GRPO eliminates superficially aligned policies by ruling them out as local optima. Empirically, we evaluate CRAFT on multiple safety benchmarks using two strong reasoning models, Qwen3-4B-Thinking and R1-Distill-Llama-8B, where it consistently outperforms state-of-the-art defenses such as IPO and SafeKey. Notably, CRAFT delivers an average 79.0% improvement in reasoning safety and 87.7% improvement in final-response safety over the base models, demonstrating the effectiveness of hidden-space reasoning alignment.

**arXiv ID:** 2603.17305
</details>

<details>
<summary><strong>Physics-informed offline reinforcement learning eliminates catastrophic fuel waste in maritime routing</strong> - Aniruddha Bora, Julie Chalfant, Chryssostomos Chryssostomidis - [[pdf]](https://arxiv.org/pdf/2603.17319)</summary>

**Abstract:** International shipping produces approximately 3% of global greenhouse gas emissions, yet voyage routing remains dominated by heuristic methods. We present PIER (Physics-Informed, Energy-efficient, Risk-aware routing), an offline reinforcement learning framework that learns fuel-efficient, safety-aware routing policies from physics-calibrated environments grounded in historical vessel tracking data and ocean reanalysis products, requiring no online simulator. Validated on one full year (2023) of AIS data across seven Gulf of Mexico routes (840 episodes per method), PIER reduces mean CO2 emissions by 10% relative to great-circle routing. However, PIER's primary contribution is eliminating catastrophic fuel waste: great-circle routing incurs extreme fuel consumption (>1.5x median) in 4.8% of voyages; PIER reduces this to 0.5%, a 9-fold reduction. Per-voyage fuel variance is 3.5x lower (p<0.001), with bootstrap 95% CI for mean savings [2.9%, 15.7%]. Partial validation against observed AIS vessel behavior confirms consistency with the fastest real transits while exhibiting 23.1x lower variance. Crucially, PIER is forecast-independent: unlike A* path optimization whose wave protection degrades 4.5x under realistic forecast uncertainty, PIER maintains constant performance using only local observations. The framework combines physics-informed state construction, demonstration-augmented offline data, and a decoupled post-hoc safety shield, an architecture that transfers to wildfire evacuation, aircraft trajectory optimization, and autonomous navigation in unmapped terrain.

**arXiv ID:** 2603.17319
</details>

<details>
<summary><strong>MHPO: Modulated Hazard-aware Policy Optimization for Stable Reinforcement Learning</strong> - Hongjun Wang, Wei Liu, Weibo Gu, Xing Sun, Kai Han - [[pdf]](https://arxiv.org/pdf/2603.16929)</summary>

**Abstract:** Regulating the importance ratio is critical for the training stability of Group Relative Policy Optimization (GRPO) based frameworks. However, prevailing ratio control methods, such as hard clipping, suffer from non-differentiable boundaries and vanishing gradient regions, failing to maintain gradient fidelity. Furthermore, these methods lack a hazard-aware mechanism to adaptively suppress extreme deviations, leaving the optimization process vulnerable to abrupt policy shifts. To address these challenges, we propose Modulated Hazard-aware Policy Optimization (MHPO), a novel framework designed for robust and stable reinforcement learning. The proposed MHPO introduces a Log-Fidelity Modulator (LFM) to map unbounded importance ratios into a bounded, differentiable domain. This mechanism effectively prevents high-variance outlier tokens from destabilizing the loss landscape while ensuring global gradient stability. Complementarily, a Decoupled Hazard Penalty (DHP) integrates cumulative hazard functions from survival analysis to independently regulate positive and negative policy shifts. By shaping the optimization landscape with hazard-aware penalties, the proposed MHPO achieves fine-grained regulation of asymmetric policy shifts simultaneously mitigating mode collapse from over-expansion and preventing policy erosion from catastrophic contraction within a stabilized trust region. Extensive evaluations on diverse reasoning benchmarks across both text-based and vision-language tasks demonstrate that MHPO consistently outperforms existing methods, achieving superior performance while significantly enhancing training stability.

**arXiv ID:** 2603.16929
</details>

<details>
<summary><strong>Cryptographic Runtime Governance for Autonomous AI Systems: The Aegis Architecture for Verifiable Policy Enforcement</strong> - Adam Massimo Mazzocchetti - [[pdf]](https://arxiv.org/pdf/2603.16938)</summary>

**Abstract:** Contemporary AI governance frameworks rely heavily on post hoc oversight, policy guidance, and behavioral alignment techniques, yet these mechanisms become fragile as systems gain autonomy, speed, and operational opacity. This paper presents Aegis, a runtime governance architecture for autonomous AI systems that treats policy and legal constraints as execution conditions rather than advisory principles. Aegis binds each governed agent to a cryptographically sealed Immutable Ethics Policy Layer (IEPL) at system genesis and enforces external emissions through an Ethics Verification Agent (EVA), an Enforcement Kernel Module (EKM), and an Immutable Logging Kernel (ILK). Amendments to the governing policy layer require quorum approval and redeclaration of the system trust root; verified violations trigger autonomous shutdown and generation of auditable proof artifacts.
We evaluate the architecture within the Civitas runtime using three operational measures: proof verification latency under tamper conditions, publication overhead, and alignment retention performance relative to an ungoverned baseline. In controlled trials, Aegis demonstrates median proof verification latency of 238 ms, median publication overhead of approximately 9.4 ms, and higher alignment retention than the baseline condition across matched tasks. We argue that these results support a shift in AI governance from discretionary oversight toward verifiable runtime constraint. Rather than claiming to resolve machine ethics in the abstract, the proposed architecture seeks to show that policy violating behavior can be rendered operationally non executable within a controlled runtime governance framework. The paper concludes by discussing methodological limits, evidentiary implications, and the role of proof oriented governance in high assurance AI deployment.

**arXiv ID:** 2603.16938
</details>

<details>
<summary><strong>DeepStage: Learning Autonomous Defense Policies Against Multi-Stage APT Campaigns</strong> - Trung V. Phan, Tri Gia Nguyen, Thomas Bauschert - [[pdf]](https://arxiv.org/pdf/2603.16969)</summary>

**Abstract:** This paper presents DeepStage, a deep reinforcement learning (DRL) framework for adaptive, stage-aware defense against Advanced Persistent Threats (APTs). The enterprise environment is modeled as a partially observable Markov decision process (POMDP), where host provenance and network telemetry are fused into unified provenance graphs. Building on our prior work, StageFinder, a graph neural encoder and an LSTM-based stage estimator infer probabilistic attacker stages aligned with the MITRE ATT&CK framework. These stage beliefs, combined with graph embeddings, guide a hierarchical Proximal Policy Optimization (PPO) agent that selects defense actions across monitoring, access control, containment, and remediation. Evaluated in a realistic enterprise testbed using CALDERA-driven APT playbooks, DeepStage achieves a stage-weighted F1-score of 0.89, outperforming a risk-aware DRL baseline by 21.9%. The results demonstrate effective stage-aware and cost-efficient autonomous cyber defense.

**arXiv ID:** 2603.16969
</details>

<details>
<summary><strong>CircuitBuilder: From Polynomials to Circuits via Reinforcement Learning</strong> - Weikun K. Zhang, Rohan Pandey, Bhaumik Mehta, Kaijie Jin, Naomi Morato, Archit Ganapule, Michael Ruofan Zeng, Jarod Alper - [[pdf]](https://arxiv.org/pdf/2603.17075)</summary>

**Abstract:** Motivated by auto-proof generation and Valiant's VP vs. VNP conjecture, we study the problem of discovering efficient arithmetic circuits to compute polynomials, using addition and multiplication gates. We formulate this problem as a single-player game, where an RL agent attempts to build the circuit within a fixed number of operations. We implement an AlphaZero-style training loop and compare two approaches: Proximal Policy Optimization with Monte Carlo Tree Search (PPO+MCTS) and Soft Actor-Critic (SAC). SAC achieves the highest success rates on two-variable targets, while PPO+MCTS scales to three variables and demonstrates steady improvement on harder instances. These results suggest that polynomial circuit synthesis is a compact, verifiable setting for studying self-improving search policies.

**arXiv ID:** 2603.17075
</details>

<details>
<summary><strong>REAL: Regression-Aware Reinforcement Learning for LLM-as-a-Judge</strong> - Yasi Zhang, Tianyu Chen, Mingyuan Zhou, Oscar Leong, Ying Nian Wu, Michal Lukasik - [[pdf]](https://arxiv.org/pdf/2603.17145)</summary>

**Abstract:** Large language models (LLMs) are increasingly deployed as automated evaluators that assign numeric scores to model outputs, a paradigm known as LLM-as-a-Judge. However, standard Reinforcement Learning (RL) methods typically rely on binary rewards (e.g., 0-1 accuracy), thereby ignoring the ordinal structure inherent in regression tasks; for instance, they fail to recognize that predicting 4 is significantly better than predicting 1 when the ground truth is 5. Conversely, existing regression-aware approaches are often confined to Supervised Fine-Tuning (SFT), limiting their ability to explore optimal reasoning paths. To bridge this gap, we propose \textbf{REAL} (\underline{RE}gression-\underline{A}ware Reinforcement \underline{L}earning), a principled RL framework designed to optimize regression rewards, and also proven to be optimal for correlation metrics. A key technical challenge is that the regression objective is explicitly policy-dependent, thus invalidating standard policy gradient methods. To address this, we employ the generalized policy gradient estimator, which naturally decomposes optimization into two complementary components: (1) exploration over Chain-of-Thought (CoT) trajectory, and (2) regression-aware prediction refinement of the final score. Extensive experiments across model scales (8B to 32B) demonstrate that REAL consistently outperforms both regression-aware SFT baselines and standard RL methods, exhibiting significantly better generalization on out-of-domain benchmarks. On Qwen3-32B specifically, we achieve gains of +8.40 Pearson and +7.20 Spearman correlation over the SFT baseline, and +18.30/+11.20 over the base model. These findings highlight the critical value of integrating regression objectives into RL exploration for accurate LLM evaluation.

**arXiv ID:** 2603.17145
</details>

<details>
<summary><strong>Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress</strong> - Yuelin Zhang, Sijie Cheng, Chen Li, Zongzhao Li, Yuxin Huang, Yang Liu, Wenbing Huang - [[pdf]](https://arxiv.org/pdf/2603.17312)</summary>

**Abstract:** Accurately estimating task progress is critical for embodied agents to plan and execute long-horizon, multi-step tasks. Despite promising advances, existing Vision-Language Models (VLMs) based methods primarily leverage their video understanding capabilities, while neglecting their complex reasoning potential. Furthermore, processing long video trajectories with VLMs is computationally prohibitive for real-world deployment. To address these challenges, we propose the Recurrent Reasoning Vision-Language Model ($\text{R}^2$VLM). Our model features a recurrent reasoning framework that processes local video snippets iteratively, maintaining a global context through an evolving Chain of Thought (CoT). This CoT explicitly records task decomposition, key steps, and their completion status, enabling the model to reason about complex temporal dependencies. This design avoids the high cost of processing long videos while preserving essential reasoning capabilities. We train $\text{R}^2$VLM on large-scale, automatically generated datasets from ALFRED and Ego4D. Extensive experiments on progress estimation and downstream applications, including progress-enhanced policy learning, reward modeling for reinforcement learning, and proactive assistance, demonstrate that $\text{R}^2$VLM achieves strong performance and generalization, achieving a new state-of-the-art in long-horizon task progress estimation. The models and benchmarks are publicly available at \href{this https URL}{huggingface}.

**arXiv ID:** 2603.17312
</details>

<details>
<summary><strong>WebPII: Benchmarking Visual PII Detection for Computer-Use Agents</strong> - Nathan Zhao - [[pdf]](https://arxiv.org/pdf/2603.17357)</summary>

**Abstract:** Computer use agents create new privacy risks: training data collected from real websites inevitably contains sensitive information, and cloud-hosted inference exposes user screenshots. Detecting personally identifiable information in web screenshots is critical for privacy-preserving deployment, but no public benchmark exists for this task. We introduce WebPII, a fine-grained synthetic benchmark of 44,865 annotated e-commerce UI images designed with three key properties: extended PII taxonomy including transaction-level identifiers that enable reidentification, anticipatory detection for partially-filled forms where users are actively entering data, and scalable generation through VLM-based UI reproduction. Experiments validate that these design choices improve layout-invariant detection across diverse interfaces and generalization to held-out page types. We train WebRedact to demonstrate practical utility, more than doubling text-extraction baseline accuracy (0.753 vs 0.357 mAP@50) at real-time CPU latency (20ms). We release the dataset and model to support privacy-preserving computer use research.

**arXiv ID:** 2603.17357
</details>

<details>
<summary><strong>Benchmarking Reinforcement Learning via Stochastic Converse Optimality: Generating Systems with Known Optimal Policies</strong> - Sinan Ibrahim, Grégoire Ouerdane, Hadi Salloum, Henni Ouerdane, Stefan Streif, Pavel Osinenko - [[pdf]](https://arxiv.org/pdf/2603.17631)</summary>

**Abstract:** The objective comparison of Reinforcement Learning (RL) algorithms is notoriously complex as outcomes and benchmarking of performances of different RL approaches are critically sensitive to environmental design, reward structures, and stochasticity inherent in both algorithmic learning and environmental dynamics. To manage this complexity, we introduce a rigorous benchmarking framework by extending converse optimality to discrete-time, control-affine, nonlinear systems with noise. Our framework provides necessary and sufficient conditions, under which a prescribed value function and policy are optimal for constructed systems, enabling the systematic generation of benchmark families via homotopy variations and randomized parameters. We validate it by automatically constructing diverse environments, demonstrating our framework's capacity for a controlled and comprehensive evaluation across algorithms. By assessing standard methods against a ground-truth optimum, our work delivers a reproducible foundation for precise and rigorous RL benchmarking.

**arXiv ID:** 2603.17631
</details>

<details>
<summary><strong>CodeScout: An Effective Recipe for Reinforcement Learning of Code Search Agents</strong> - Lintang Sutawika, Aditya Bharat Soni, Bharath Sriraam R R, Apurva Gandhi, Taha Yassine, Sanidhya Vijayvargiya, Yuchen Li, Xuhui Zhou, Yilin Zhang, Leander Melroy Maben, Graham Neubig - [[pdf]](https://arxiv.org/pdf/2603.17829)</summary>

**Abstract:** A prerequisite for coding agents to perform tasks on large repositories is code localization - the identification of relevant files, classes, and functions to work on. While repository-level code localization has been performed using embedding-based retrieval approaches such as vector search, recent work has focused on developing agents to localize relevant code either as a standalone precursor to or interleaved with performing actual work. Most prior methods on agentic code search equip the agent with complex, specialized tools, such as repository graphs derived from static analysis. In this paper, we demonstrate that, with an effective reinforcement learning recipe, a coding agent equipped with nothing more than a standard Unix terminal can be trained to achieve strong results. Our experiments on three benchmarks (SWE-Bench Verified, Pro, and Lite) reveal that our models consistently achieve superior or competitive performance over 2-18x larger base and post-trained LLMs and sometimes approach performance provided by closed models like Claude Sonnet, even when using specialized scaffolds. Our work particularly focuses on techniques for re-purposing existing coding agent environments for code search, reward design, and RL optimization. We release the resulting model family, CodeScout, along with all our code and data for the community to build upon.

**arXiv ID:** 2603.17829
</details>

<details>
<summary><strong>Differential Privacy in Generative AI Agents: Analysis and Optimal Tradeoffs</strong> - Ya-Ting Yang, Quanyan Zhu - [[pdf]](https://arxiv.org/pdf/2603.17902)</summary>

**Abstract:** Large language models (LLMs) and AI agents are increasingly integrated into enterprise systems to access internal databases and generate context-aware responses. While such integration improves productivity and decision support, the model outputs may inadvertently reveal sensitive information. Although many prior efforts focus on protecting the privacy of user prompts, relatively few studies consider privacy risks from the enterprise data perspective. Hence, this paper develops a probabilistic framework for analyzing privacy leakage in AI agents based on differential privacy. We model response generation as a stochastic mechanism that maps prompts and datasets to distributions over token sequences. Within this framework, we introduce token-level and message-level differential privacy and derive privacy bounds that relate privacy leakage to generation parameters such as temperature and message length. We further formulate a privacy-utility design problem that characterizes optimal temperature selection.

**arXiv ID:** 2603.17902
</details>

<details>
<summary><strong>TheraMind: A Strategic and Adaptive Agent for Longitudinal Psychological Counseling</strong> - He Hu, Chiyuan Ma, Qianning Wang, Lin Liu, Yucheng Zhou, Laizhong Cui, Fei Ma, Qi Tian - [[pdf]](https://arxiv.org/pdf/2510.25758)</summary>

**Abstract:** The shortage of mental health professionals has driven the web to become a primary avenue for accessible psychological support. While Large Language Models (LLMs) offer promise for scalable web-based counseling, existing approaches often lack emotional understanding, adaptive strategies, and long-term memory. These limitations pose risks to digital well-being, as disjointed interactions can fail to support vulnerable users effectively. To address these gaps, we introduce TheraMind, a strategic and adaptive agent designed for trustworthy online longitudinal counseling. The cornerstone of TheraMind is a novel dual-loop architecture that decouples the complex counseling process into an Intra-Session Loop for tactical dialogue management and a Cross-Session Loop for strategic therapeutic planning. The Intra-Session Loop perceives the patient's emotional state to dynamically select response strategies while leveraging cross-session memory to ensure continuity. Crucially, the Cross-Session Loop empowers the agent with long-term adaptability by evaluating the efficacy of the applied therapy after each session and adjusting the method for subsequent interactions. We validate our approach in a high-fidelity simulation environment grounded in real clinical cases. Extensive evaluations show that TheraMind outperforms other methods, especially on multi-session metrics like Coherence, Flexibility, and Therapeutic Attunement, validating the effectiveness of its dual-loop design in emulating strategic, adaptive, and longitudinal therapeutic behavior. The code is publicly available at this https URL.

**arXiv ID:** 2510.25758
</details>

<details>
<summary><strong>Reinforcement learning with learned gadgets to tackle hard quantum problems on real hardware</strong> - Akash Kundu, Leopoldo Sarra - [[pdf]](https://arxiv.org/pdf/2411.00230)</summary>

**Abstract:** Quantum computing offers exciting opportunities for simulating complex quantum systems and optimizing large scale combinatorial problems, but its practical use is limited by device noise and constrained connectivity. Designing quantum circuits, which are fundamental to quantum algorithms, is therefore a central challenge in current quantum hardware. Existing reinforcement learning based methods for circuit design lose accuracy when restricted to hardware native gates and device level compilation. Here, we introduce gadget reinforcement learning (GRL), which combines learning with program synthesis to automatically construct composite gates that expand the action space while respecting hardware constraints. We show that this approach improves accuracy, hardware compatibility, and scalability for transverse-field Ising and quantum chemistry problems, reaching systems of up to ten qubits within realistic computational budgets. This framework demonstrates how learned, reusable circuit building blocks can guide the co-design of algorithms and hardware for quantum processors.

**arXiv ID:** 2411.00230
</details>

<details>
<summary><strong>MOBODY: Model Based Off-Dynamics Offline Reinforcement Learning</strong> - Yihong Guo, Yu Yang, Pan Xu, Anqi Liu - [[pdf]](https://arxiv.org/pdf/2506.08460)</summary>

**Abstract:** We study off-dynamics offline reinforcement learning, where the goal is to learn a policy from offline source and limited target datasets with mismatched dynamics. Existing methods either penalize the reward or discard source transitions occurring in parts of the transition space with high dynamics shift. As a result, they optimize the policy using data from low-shift regions, limiting exploration of high-reward states in the target domain that do not fall within these regions. Consequently, such methods often fail when the dynamics shift is significant or the optimal trajectories lie outside the low-shift regions. To overcome this limitation, we propose MOBODY, a Model-Based Off-Dynamics Offline RL algorithm that optimizes a policy using learned target dynamics transitions to explore the target domain, rather than only being trained with the low dynamics-shift transitions. For the dynamics learning, built on the observation that achieving the same next state requires taking different actions in different domains, MOBODY employs separate action encoders for each domain to encode different actions to the shared latent space while sharing a unified representation of states and a common transition function. We further introduce a target Q-weighted behavior cloning loss in policy optimization to avoid out-of-distribution actions, which push the policy toward actions with high target-domain Q-values, rather than high source domain Q-values or uniformly imitating all actions in the offline dataset. We evaluate MOBODY on a wide range of MuJoCo and Adroit benchmarks, demonstrating that it outperforms state-of-the-art off-dynamics RL baselines as well as policy learning methods based on different dynamics learning baselines, with especially pronounced improvements in challenging scenarios where existing methods struggle.

**arXiv ID:** 2506.08460
</details>

<details>
<summary><strong>In-Context Compositional Q-Learning for Offline Reinforcement Learning</strong> - Qiushui Xu, Yuhao Huang, Yushu Jiang, Lei Song, Jinyu Wang, Wenliang Zheng, Jiang Bian - [[pdf]](https://arxiv.org/pdf/2509.24067)</summary>

**Abstract:** Accurate estimation of the Q-function is a central challenge in offline reinforcement learning. However, existing approaches often rely on a shared global Q-function, which is inadequate for capturing the compositional structure of tasks that consist of diverse subtasks. We propose In-context Compositional Q-Learning (ICQL), an offline RL framework that formulates Q-learning as a contextual inference problem and uses linear Transformers to adaptively infer local Q-functions from retrieved transitions without explicit subtask labels. Theoretically, we show that, under two assumptions -- linear approximability of the local Q-function and accurate inference of weights from retrieved context -- ICQL achieves a bounded approximation error for the Q-function and enables near-optimal policy extraction. Empirically, ICQL substantially improves performance in offline settings, achieving gains of up to 16.4% on kitchen tasks and up to 8.8% and 6.3% on MuJoCo and Adroit tasks, respectively. These results highlight the underexplored potential of in-context learning for robust and compositional value estimation and establish ICQL as a principled and effective framework for offline RL.

**arXiv ID:** 2509.24067
</details>

<details>
<summary><strong>Ablation Study of a Fairness Auditing Agentic System for Bias Mitigation in Early-Onset Colorectal Cancer Detection</strong> - Amalia Ionescu, Jose Guadalupe Hernandez, Jui-Hsuan Chang, Emily F. Wong, Paul Wang, Jason H. Moore, Tiffani J. Bright - [[pdf]](https://arxiv.org/pdf/2603.17179)</summary>

**Abstract:** Artificial intelligence (AI) is increasingly used in clinical settings, yet limited oversight and domain expertise can allow algorithmic bias and safety risks to persist. This study evaluates whether an agentic AI system can support auditing biomedical machine learning models for fairness in early-onset colorectal cancer (EO-CRC), a condition with documented demographic disparities. We implemented a two-agent architecture consisting of a Domain Expert Agent that synthesizes literature on EO-CRC disparities and a Fairness Consultant Agent that recommends sensitive attributes and fairness metrics for model evaluation. An ablation study compared three Ollama large language models (8B, 20B, and 120B parameters) across three configurations: pretrained LLM-only, Agent without Retrieval-Augmented Generation (RAG), and Agent with RAG. Across models, the Agent with RAG achieved the highest semantic similarity to expert-derived reference statements, particularly for disparity identification, suggesting agentic systems with retrieval may help scale fairness auditing in clinical AI.

**arXiv ID:** 2603.17179
</details>

<details>
<summary><strong>Capability-Priced Micro-Markets: A Micro-Economic Framework for the Agentic Web over HTTP 402</strong> - Ken Huang, Jerry Huang, Mahesh Lambe, Hammad Atta, Yasir Mehmood, Muhammad Zeeshan Baig, Muhammad Aziz Ul Haq, Nadeem Shahzad, Shailja Gupta, Rajesh Ranjan, Rekha Singhal - [[pdf]](https://arxiv.org/pdf/2603.16899)</summary>

**Abstract:** This paper introduces Capability-Priced Micro-Markets (CPMM), a micro-economic framework designed to enable robust, scalable, and secure commerce among autonomous AI agents on the agentic web. The framework addresses the fundamental challenge of economic coordination in decentralized agent ecosystems, where entities must transact with minimal human oversight. CPMM synthesizes three key technologies into a unified system: MIT originated, Project NANDA infrastructure for cryptographically verifiable, capability-based security and discovery; the HTTP 402 "Payment Required" status code, with modern X402/H402 extensions for efficient, low-cost micropayments; and the Agent Capability Negotiation and Binding Protocol (ACNBP) for secure, multi-step negotiation and commitment. The paper formalizes agent interactions as a repeated bilateral game with incomplete information, demonstrating theoretically that the CPMM mechanism converges to a constrained Radner equilibrium, ensuring efficient outcomes under information asymmetry. A key theoretical contribution is the concept of "privacy elasticity of demand," which is introduced to quantify the trade-off between an agent's information disclosure and the market price of its services. By integrating secure capabilities, micropayment protocols, and formal negotiation mechanisms, CPMM provides a comprehensive, theoretically-grounded solution for creating functional micro-markets for the emergent agentic web.

**arXiv ID:** 2603.16899
</details>

<details>
<summary><strong>Forecast-Aware Cooperative Planning on Temporal Graphs under Stochastic Adversarial Risk</strong> - Manshi Limbu, Xuan Wang, Gregory J. Stein, Daigo Shishika, Xuesu Xiao - [[pdf]](https://arxiv.org/pdf/2603.14697)</summary>

**Abstract:** Cooperative multi-robot missions often require teams of robots to traverse environments where traversal risk evolves due to adversary patrols or shifting hazards with stochastic dynamics. While support coordination--where robots assist teammates in traversing risky regions--can significantly reduce mission costs, its effectiveness depends on the team's ability to anticipate future risk. Existing support-based frameworks assume static risk landscapes and therefore fail to account for predictable temporal trends in risk evolution. We propose a forecast-aware cooperative planning framework that integrates stochastic risk forecasting with anticipatory support allocation on temporal graphs. By modeling adversary dynamics as a first-order Markov stay-move process over graph edges, we propagate the resulting edge-occupancy probabilities forward in time to generate time-indexed edge-risk forecasts. These forecasts guide the proactive allocation of support positions to forecasted risky edges for effective support coordination, while also informing joint robot path planning. Experimental results demonstrate that our approach consistently reduces total expected team cost compared to non-anticipatory baselines, approaching the performance of an oracle planner.

**arXiv ID:** 2603.14697
</details>

<details>
<summary><strong>Complementary Reinforcement Learning</strong> - Dilxat Muhtar, Jiashun Liu, Wei Gao, Weixun Wang, Shaopan Xiong, Ju Huang, Siran Yang, Wenbo Su, Jiamang Wang, Ling Pan, Bo Zheng - [[pdf]](https://arxiv.org/pdf/2603.17621)</summary>

**Abstract:** Reinforcement Learning (RL) has emerged as a powerful paradigm for training LLM-based agents, yet remains limited by low sample efficiency, stemming not only from sparse outcome feedback but also from the agent's inability to leverage prior experience across episodes. While augmenting agents with historical experience offers a promising remedy, existing approaches suffer from a critical weakness: the experience distilled from history is either stored statically or fail to coevolve with the improving actor, causing a progressive misalignment between the experience and the actor's evolving capability that diminishes its utility over the course of training. Inspired by complementary learning systems in neuroscience, we present Complementary RL to achieve seamless co-evolution of an experience extractor and a policy actor within the RL optimization loop. Specifically, the actor is optimized via sparse outcome-based rewards, while the experience extractor is optimized according to whether its distilled experiences demonstrably contribute to the actor's success, thereby evolving its experience management strategy in lockstep with the actor's growing capabilities. Empirically, Complementary RL outperforms outcome-based agentic RL baselines that do not learn from experience, achieving 10% performance improvement in single-task scenarios and exhibits robust scalability in multi-task settings. These results establish Complementary RL as a paradigm for efficient experience-driven agent learning.

**arXiv ID:** 2603.17621
</details>

<details>
<summary><strong>Detecting Data Contamination from Reinforcement Learning Post-training for Large Language Models</strong> - Yongding Tao, Tian Wang, Yihong Dong, Huanyu Liu, Kechi Zhang, Xiaolong Hu, Ge Li - [[pdf]](https://arxiv.org/pdf/2510.09259)</summary>

**Abstract:** Data contamination poses a significant threat to the reliable evaluation of Large Language Models (LLMs). This issue arises when benchmark samples may inadvertently appear in training sets, compromising the validity of reported performance. While detection methods have been developed for the pre-training and Supervised Fine-Tuning stages, a critical research gap exists for the increasingly significant phase of Reinforcement Learning (RL) post-training. As RL post-training becomes pivotal for advancing LLM reasoning, the absence of specialized contamination detection methods in this paradigm presents a critical vulnerability. To address this, we conduct the first systematic study of data detection within RL post-training scenario and propose Self-Critique. Our method is motivated by a key observation: after RL phase, the output entropy distribution of LLMs tends to collapse into highly specific and sparse modes. Self-Critique probes for the underlying policy collapse, i.e., the model's convergence to a narrow reasoning path, which causes this entropy reduction. To facilitate this research, we also introduce RL-MIA, a benchmark constructed to simulate this specific contamination scenario. Extensive experiments show that Self-Critique significantly outperforms baseline methods across multiple models and contamination tasks, achieving an AUC improvement of up to 30%. Whereas existing methods are close to a random guess for RL-phase contamination, our method makes detection possible.

**arXiv ID:** 2510.09259
</details>

<details>
<summary><strong>Improving Low-Resource Machine Translation via Round-Trip Reinforcement Learning</strong> - Ahmed Attia, Alham Fikri Aji - [[pdf]](https://arxiv.org/pdf/2601.12535)</summary>

**Abstract:** Low-resource machine translation (MT) has gained increasing attention as parallel data from low-resource language communities is collected, but many approaches for improving low-resource MT remain underexplored. We investigate a self-supervised reinforcement learning fine-tuning for translation in low-resource settings using round-trip bootstrapping with the No Language Left Behind (NLLB) family of models. Our approach translates English into a target low-resource language and then back into English, using a combination of chrF++ and BLEU as the reward function on the reconstructed English sentences. Using the NLLB-MD dataset, we evaluate both the 600M and 1.3B parameter NLLB models and observe consistent improvements for the following languages: Central Aymara, Friulian, Wolof, Dyula, Bhojpuri and Russian. Qualitative inspection of translation outputs indicates increased fluency and semantic fidelity. We argue that our method can further benefit from scale, enabling models to increasingly leverage their pretrained knowledge and continue self-improving. Code available at: this https URL

**arXiv ID:** 2601.12535
</details>

<details>
<summary><strong>Human-AI Co-reasoning for Clinical Diagnosis with Evidence-Integrated Language Agent</strong> - Zhongzhen Huang, Yan Ling, Hong Chen, Ye Feng, Li Wu, Linjie Mu, Shaoting Zhang, Xiaofan Zhang, Kun Qian, Xiaomu Li - [[pdf]](https://arxiv.org/pdf/2603.10492)</summary>

**Abstract:** We present PULSE, a medical reasoning agent that combines a domain-tuned large language model with scientific literature retrieval to support diagnostic decision-making in complex real-world cases. To evaluate its capabilities, we curated a benchmark of 82 authentic endocrinology case reports encompassing a broad spectrum of disease types and incidence levels. In controlled experiments, we compared PULSE's performance against physicians with varying levels of expertise-from residents to senior specialists-and examined how AI assistance influenced human diagnostic reasoning. PULSE attained expert-competitive accuracy, outperforming residents and junior specialists while matching senior specialist performance at both Top@1 and Top@4 thresholds. Unlike physicians, whose accuracy declined with disease rarity, PULSE maintained stable performance across incidence tiers. The agent also exhibited adaptive reasoning, increasing output length with case difficulty in a manner analogous to the longer deliberation observed among expert clinicians. When used collaboratively, PULSE enabled physicians to correct initial errors and broaden diagnostic hypotheses, but also introduced risks of automation bias. The study explores both serial and concurrent collaboration workflows, revealing that PULSE offers robust support across common and rare presentations. These findings underscore both the promise and the limitations of language model-based agents in clinical diagnosis, and offer a framework for evaluating their role in real-world decision-making.

**arXiv ID:** 2603.10492
</details>

<details>
<summary><strong>Meta-Reinforcement Learning with Self-Reflection for Agentic Search</strong> - Teng Xiao, Yige Yuan, Hamish Ivison, Huaisheng Zhu, Faeze Brahman, Nathan Lambert, Pradeep Dasigi, Noah A. Smith, Hannaneh Hajishirzi - [[pdf]](https://arxiv.org/pdf/2603.11327)</summary>

**Abstract:** This paper introduces MR-Search, an in-context meta reinforcement learning (RL) formulation for agentic search with self-reflection. Instead of optimizing a policy within a single independent episode with sparse rewards, MR-Search trains a policy that conditions on past episodes and adapts its search strategy across episodes. MR-Search learns to learn a search strategy with self-reflection, allowing search agents to improve in-context exploration at test-time. Specifically, MR-Search performs cross-episode exploration by generating explicit self-reflections after each episode and leveraging them as additional context to guide subsequent attempts, thereby promoting more effective exploration during test-time. We further introduce a multi-turn RL algorithm that estimates a dense relative advantage at the turn level, enabling fine-grained credit assignment on each episode. Empirical results across various benchmarks demonstrate the advantages of MR-Search over baselines based RL, showing strong generalization and relative improvements of 9.2% to 19.3% across eight benchmarks. Our code and data are available at this https URL.

**arXiv ID:** 2603.11327
</details>

<details>
<summary><strong>MetaClaw: Just Talk -- An Agent That Meta-Learns and Evolves in the Wild</strong> - Peng Xia, Jianwen Chen, Xinyu Yang, Haoqin Tu, Jiaqi Liu, Kaiwen Xiong, Siwei Han, Shi Qiu, Haonian Ji, Yuyin Zhou, Zeyu Zheng, Cihang Xie, Huaxiu Yao - [[pdf]](https://arxiv.org/pdf/2603.17187)</summary>

**Abstract:** Large language model (LLM) agents are increasingly used for complex tasks, yet deployed agents often remain static, failing to adapt as user needs evolve. This creates a tension between the need for continuous service and the necessity of updating capabilities to match shifting task distributions. On platforms like OpenClaw, which handle diverse workloads across 20+ channels, existing methods either store raw trajectories without distilling knowledge, maintain static skill libraries, or require disruptive downtime for retraining. We present MetaClaw, a continual meta-learning framework that jointly evolves a base LLM policy and a library of reusable behavioral skills. MetaClaw employs two complementary mechanisms. Skill-driven fast adaptation analyzes failure trajectories via an LLM evolver to synthesize new skills, enabling immediate improvement with zero downtime. Opportunistic policy optimization performs gradient-based updates via cloud LoRA fine-tuning and Reinforcement Learning with a Process Reward Model (RL-PRM). This is triggered during user-inactive windows by the Opportunistic Meta-Learning Scheduler (OMLS), which monitors system inactivity and calendar data. These mechanisms are mutually reinforcing: a refined policy generates better trajectories for skill synthesis, while richer skills provide higher-quality data for policy optimization. To prevent data contamination, a versioning mechanism separates support and query data. Built on a proxy-based architecture, MetaClaw scales to production-size LLMs without local GPUs. Experiments on MetaClaw-Bench and AutoResearchClaw show that skill-driven adaptation improves accuracy by up to 32% relative. The full pipeline advances Kimi-K2.5 accuracy from 21.4% to 40.6% and increases composite robustness by 18.3%. Code is available at this https URL.

**arXiv ID:** 2603.17187
</details>

<details>
<summary><strong>Shielded Reinforcement Learning Under Dynamic Temporal Logic Constraints</strong> - Sadık Bera Yüksel, Ali Tevfik Buyukkocak, Derya Aksaray - [[pdf]](https://arxiv.org/pdf/2603.17152)</summary>

**Abstract:** Reinforcement Learning (RL) has shown promise in various robotics applications, yet its deployment on real systems is still limited due to safety and operational constraints. The safe RL field has gained considerable attention in recent years, which focuses on imposing safety constraints throughout the learning process. However, real systems often require more complex constraints than just safety, such as periodic recharging or time-bounded visits to specific regions. Imposing such spatio-temporal tasks during learning still remains a challenge. Signal Temporal Logic (STL) is a formal language for specifying temporal properties of real-valued signals and provides a way to express such complex tasks. In this paper, we propose a framework that leverages sequential control barrier functions and model-free RL to ensure that the given STL tasks are satisfied throughout the learning process. Our method extends beyond traditional safety constraints by enforcing rich STL specifications, which can involve visits to dynamic targets with unknown trajectories. We also demonstrate the effectiveness of our framework through various simulations.

**arXiv ID:** 2603.17152
</details>

<details>
<summary><strong>SALSA-RL: Stability Analysis in the Latent Space of Actions for Reinforcement Learning</strong> - Xuyang Li, Romit Maulik - [[pdf]](https://arxiv.org/pdf/2502.15512)</summary>

**Abstract:** Modern deep reinforcement learning (DRL) methods have made significant advances in handling continuous action spaces. However, real-world control systems, especially those requiring precise and reliable performance, often demand interpretability in the sense of a-priori assessments of agent behavior to identify safe or failure-prone interactions with environments. To address this limitation, this work proposes SALSA-RL (Stability Analysis in the Latent Space of Actions), a novel RL framework that models control actions as dynamic, time-dependent variables evolving within a latent space. By employing a pre-trained encoder-decoder and a state-dependent linear system, this approach enables interpretability through local stability analysis, where instantaneous growth in action-norms can be predicted before their execution. It is demonstrated that SALSA-RL can be deployed in a non-invasive manner for assessing the local stability of actions from pretrained RL agents without compromising on performance across diverse benchmark environments. By enabling a more interpretable analysis of action generation, SALSA-RL provides a powerful tool for advancing the design, analysis, and theoretical understanding of RL systems.

**arXiv ID:** 2502.15512
</details>

<details>
<summary><strong>Offline Reinforcement Learning via Inverse Optimization</strong> - Ioannis Dimanidis, Tolga Ok, Peyman Mohajerin Esfahani - [[pdf]](https://arxiv.org/pdf/2502.20030)</summary>

**Abstract:** Inspired by the recent successes of Inverse Optimization (IO) across various application domains, we propose a novel offline Reinforcement Learning (ORL) algorithm for continuous state and action spaces, leveraging the convex loss function called ``sub-optimality loss'' from the IO literature. To mitigate the distribution shift commonly observed in ORL problems, we further employ a robust and non-causal Model Predictive Control (MPC) expert steering a nominal model of the dynamics using in-hindsight information stemming from the model mismatch. Unlike the existing literature, our robust MPC expert enjoys an exact and tractable convex reformulation. In the second part of this study, we show that the IO hypothesis class, trained by the proposed convex loss function, enjoys ample expressiveness and {reliably recovers teacher behavior in MuJoCo benchmarks. The method achieves competitive results compared to widely-used baselines in sample-constrained settings, despite using} orders of magnitude fewer parameters. To facilitate the reproducibility of our results, we provide an open-source package implementing the proposed algorithms and the experiments. The code is available at this https URL.

**arXiv ID:** 2502.20030
</details>

<details>
<summary><strong>Efficient Cross-Domain Offline Reinforcement Learning with Dynamics- and Value-Aligned Data Filtering</strong> - Zhongjian Qiao, Rui Yang, Jiafei Lyu, Chenjia Bai, Xiu Li, Siyang Gao, Shuang Qiu - [[pdf]](https://arxiv.org/pdf/2512.02435)</summary>

**Abstract:** Cross-domain offline reinforcement learning (RL) aims to train a well-performing agent in the target environment, leveraging both a limited target domain dataset and a source domain dataset with (possibly) sufficient data coverage. Due to the underlying dynamics misalignment between source and target domains, naively merging the two datasets may incur inferior performance. Recent advances address this issue by selectively leveraging source domain samples whose dynamics align well with the target domain. However, our work demonstrates that dynamics alignment alone is insufficient, by examining the limitations of prior frameworks and deriving a new target domain sub-optimality bound for the policy learned on the source domain. More importantly, our theory underscores an additional need for \textit{value alignment}, i.e., selecting high-quality, high-value samples from the source domain, a critical dimension overlooked by existing works. Motivated by such theoretical insight, we propose \textbf{\underline{D}}ynamics- and \textbf{\underline{V}}alue-aligned \textbf{\underline{D}}ata \textbf{\underline{F}}iltering (DVDF) method, a novel unified cross-domain RL framework that selectively incorporates source domain samples exhibiting strong alignment in \textit{both dynamics and values}. We empirically study a range of dynamics shift scenarios, including kinematic and morphology shifts, and evaluate DVDF on various tasks and datasets, even in the challenging setting where the target domain dataset contains an extremely limited amount of data. Extensive experiments demonstrate that DVDF consistently outperforms strong baselines with significant improvements.

**arXiv ID:** 2512.02435
</details>

<details>
<summary><strong>Evaluating Feature Dependent Noise in Preference-based Reinforcement Learning</strong> - Yuxuan Li, Harshith Reddy Kethireddy, Srijita Das - [[pdf]](https://arxiv.org/pdf/2601.01904)</summary>

**Abstract:** Learning from Preferences in Reinforcement Learning (PbRL) has gained attention recently, as it serves as a natural fit for complicated tasks where the reward function is not easily available. However, preferences often come with uncertainty and noise if they are not from perfect teachers. Much prior literature aimed to detect noise, but with limited types of noise and most being uniformly distributed with no connection to observations. In this work, we formalize the notion of targeted feature-dependent noise and propose several variants like trajectory feature noise, trajectory similarity noise, margin dependent noise, and Language Model noise. We evaluate feature-dependent noise, where noise is correlated with certain features in complex continuous control tasks from DMControl and Meta-world. Our experiments show that in some feature-dependent noise settings, the state-of-the-art noise-robust PbRL method's learning performance is significantly deteriorated, while PbRL method with no explicit denoising can surprisingly outperform noise-robust PbRL in the majority of settings. We also find language models' noise exhibits similar characteristics to feature-dependent noise, thereby simulating realistic humans and call for further study in learning with feature-dependent noise robustly.

**arXiv ID:** 2601.01904
</details>

<details>
<summary><strong>SLowRL: Safe Low-Rank Adaptation Reinforcement Learning for Locomotion</strong> - Elham Daneshmand, Shafeef Omar, Glen Berseth, Majid Khadiv, Hsiu-Chin Lin - [[pdf]](https://arxiv.org/pdf/2603.17092)</summary>

**Abstract:** Sim-to-real transfer of locomotion policies often leads to performance degradation due to the inevitable sim-to-real gap. Naively fine-tuning these policies directly on hardware is problematic, as it poses risks of mechanical failure and suffers from high sample inefficiency. In this paper, we address the challenge of safely and efficiently fine-tuning reinforcement learning (RL) policies for dynamic locomotion tasks. Specifically, we focus on fine-tuning policies learned in simulation directly on hardware, while explicitly enforcing safety constraints. In doing so, we introduce SLowRL, a framework that combines Low-Rank Adaptation (LoRA) with training-time safety enforcement via a recovery policy. We evaluate our method both in simulation and on a real Unitree Go2 quadruped robot for jump and trot tasks. Experimental results show that our method achieves a $46.5\%$ reduction in fine-tuning time and near-zero safety violations compared to standard proximal policy optimization (PPO) baselines. Notably, we find that a rank-1 adaptation alone is sufficient to recover pre-trained performance in the real world, while maintaining stable and safe real-world fine-tuning. These results demonstrate the practicality of safe, efficient fine-tuning for dynamic real-world robotic applications.

**arXiv ID:** 2603.17092
</details>

<details>
<summary><strong>AgentVLN: Towards Agentic Vision-and-Language Navigation</strong> - Zihao Xin, Wentong Li, Yixuan Jiang, Ziyuan Huang, Bin Wang, Piji Li, Jianke Zhu, Jie Qin, Shengjun Huang - [[pdf]](https://arxiv.org/pdf/2603.17670)</summary>

**Abstract:** Vision-and-Language Navigation (VLN) requires an embodied agent to ground complex natural-language instructions into long-horizon navigation in unseen environments. While Vision-Language Models (VLMs) offer strong 2D semantic understanding, current VLN systems remain constrained by limited spatial perception, 2D-3D representation mismatch, and monocular scale ambiguity. In this paper, we propose AgentVLN, a novel and efficient embodied navigation framework that can be deployed on edge computing platforms. We formulate VLN as a Partially Observable Semi-Markov Decision Process (POSMDP) and introduce a VLM-as-Brain paradigm that decouples high-level semantic reasoning from perception and planning via a plug-and-play skill library. To resolve multi-level representation inconsistency, we design a cross-space representation mapping that projects perception-layer 3D topological waypoints into the image plane, yielding pixel-aligned visual prompts for the VLM. Building on this bridge, we integrate a context-aware self-correction and active exploration strategy to recover from occlusions and suppress error accumulation over long trajectories. To further address the spatial ambiguity of instructions in unstructured environments, we propose a Query-Driven Perceptual Chain-of-Thought (QD-PCoT) scheme, enabling the agent with the metacognitive ability to actively seek geometric depth information. Finally, we construct AgentVLN-Instruct, a large-scale instruction-tuning dataset with dynamic stage routing conditioned on target visibility. Extensive experiments show that AgentVLN consistently outperforms prior state-of-the-art methods (SOTA) on long-horizon VLN benchmarks, offering a practical paradigm for lightweight deployment of next-generation embodied navigation models. Code: this https URL.

**arXiv ID:** 2603.17670
</details>

<details>
<summary><strong>PACE: Physics Augmentation for Coordinated End-to-end Reinforcement Learning toward Versatile Humanoid Table Tennis</strong> - Muqun Hu, Wenxi Chen, Wenjing Li, Falak Mandali, Zijian He, Renhong Zhang, Praveen Krisna, Katherine Christian, Leo Benaharon, Dizhi Ma, Karthik Ramani, Yan Gu - [[pdf]](https://arxiv.org/pdf/2509.21690)</summary>

**Abstract:** Humanoid table tennis (TT) demands rapid perception, proactive whole-body motion, and agile footwork under strict timing--capabilities that remain difficult for end-to-end control policies. We propose a reinforcement learning (RL) framework that maps ball-position observations directly to whole-body joint commands for both arm striking and leg locomotion, strengthened by predictive signals and dense, physics-guided rewards. A lightweight learned predictor, fed with recent ball positions, estimates future ball states and augments the policy's observations for proactive decision-making. During training, a physics-based predictor supplies precise future states to construct dense, informative rewards that lead to effective exploration. The resulting policy attains strong performance across varied serve ranges (hit rate$\geq$96% and success rate$\geq$92%) in simulations. Ablation studies confirm that both the learned predictor and the predictive reward design are critical for end-to-end learning. Deployed zero-shot on a physical Booster T1 humanoid with 23 revolute joints, the policy produces coordinated lateral and forward-backward footwork with accurate, fast returns, suggesting a practical path toward versatile, competitive humanoid TT. We have open-sourced our RL training code at: this https URL

**arXiv ID:** 2509.21690
</details>

<details>
<summary><strong>CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions</strong> - Lizhi Yang, Blake Werner, Massimiliano de Sa, Aaron D. Ames - [[pdf]](https://arxiv.org/pdf/2510.14959)</summary>

**Abstract:** Reinforcement learning (RL), while powerful and expressive, can often prioritize performance at the expense of safety. Yet safety violations can lead to catastrophic outcomes in real-world deployments. Control Barrier Functions (CBFs) offer a principled method to enforce dynamic safety -- traditionally deployed online via safety filters. While the result is safe behavior, the fact that the RL policy does not have knowledge of the CBF can lead to conservative behaviors. This paper proposes CBF-RL, a framework for generating safe behaviors with RL by enforcing CBFs in training. CBF-RL has two key attributes: (1) minimally modifying a nominal RL policy to encode safety constraints via a CBF term, (2) and safety filtering of the policy rollouts in training. Theoretically, we prove that continuous-time safety filters can be deployed via closed-form expressions on discrete-time roll-outs. Practically, we demonstrate that CBF-RL internalizes the safety constraints in the learned policy -- both enforcing safer actions and biasing towards safer rewards -- enabling safe deployment without the need for an online safety filter. We validate our framework through ablation studies on navigation tasks and on the Unitree G1 humanoid robot, where CBF-RL enables safer exploration, faster convergence, and robust performance under uncertainty, enabling the humanoid robot to avoid obstacles and climb stairs safely in real-world settings without a runtime safety filter.

**arXiv ID:** 2510.14959
</details>

<details>
<summary><strong>REFINE-DP: Diffusion Policy Fine-tuning for Humanoid Loco-manipulation via Reinforcement Learning</strong> - Zhaoyuan Gu, Yipu Chen, Zimeng Chai, Alfred Cueva, Thong Nguyen, Yifan Wu, Huishu Xue, Minji Kim, Isaac Legene, Fukang Liu, Matthew Kim, Ayan Barula, Yongxin Chen, Ye Zhao - [[pdf]](https://arxiv.org/pdf/2603.13707)</summary>

**Abstract:** Humanoid loco-manipulation requires coordinated high-level motion plans with stable, low-level whole-body execution under complex robot-environment dynamics and long-horizon tasks. While diffusion policies (DPs) show promise for learning from demonstrations, deploying them on humanoids poses critical challenges: the motion planner trained offline is decoupled from the low-level controller, leading to poor command tracking, compounding distribution shift, and task failures. The common approach of scaling demonstration data is prohibitively expensive for high-dimensional humanoid systems. To address this challenge, we present REFINE-DP (REinforcement learning FINE-tuning of Diffusion Policy), a hierarchical framework that jointly optimizes a DP high-level planner and an RL-based low-level loco-manipulation controller. The DP is fine-tuned via a PPO-based diffusion policy gradient to improve task success rate, while the controller is simultaneously updated to accurately track the planner's evolving command distribution, reducing the distributional mismatch that degrades motion quality. We validate REFINE-DP on a humanoid robot performing loco-manipulation tasks, including door traversal and long-horizon object transport. REFINE-DP achieves an over $90\%$ success rate in simulation, even in out-of-distribution cases not seen in the pre-trained data, and enables smooth autonomous task execution in real-world dynamic environments. Our proposed method substantially outperforms pre-trained DP baselines and demonstrates that RL fine-tuning is key to reliable humanoid loco-manipulation. this https URL

**arXiv ID:** 2603.13707
</details>

<details>
<summary><strong>ViSA: Visited-State Augmentation for Generalized Goal-Space Contrastive Reinforcement Learning</strong> - Issa Nakamura, Tomoya Yamanokuchi, Yuki Kadokawa, Jia Qu, Shun Otsub, Ken Miyamoto, Shotaro Miwa, Takamitsu Matsubara - [[pdf]](https://arxiv.org/pdf/2603.14887)</summary>

**Abstract:** Goal-Conditioned Reinforcement Learning (GCRL) is a framework for learning a policy that can reach arbitrarily given goals. In particular, Contrastive Reinforcement Learning (CRL) provides a framework for policy updates using an approximation of the value function estimated via contrastive learning, achieving higher sample efficiency compared to conventional methods. However, since CRL treats the visited state as a pseudo-goal during learning, it can accurately estimate the value function only for limited goals. To address this issue, we propose a novel data augmentation approach for CRL called ViSA (Visited-State Augmentation). ViSA consists of two components: 1) generating augmented state samples, with the aim of augmenting hard-to-visit state samples during on-policy exploration, and 2) learning consistent embedding space, which uses an augmented state as auxiliary information to regularize the embedding space by reformulating the objective function of the embedding space based on mutual information. We evaluate ViSA in simulation and real-world robotic tasks and show improved goal-space generalization, which permits accurate value estimation for hard-to-visit goals. Further details can be found on the project page: this https URL

**arXiv ID:** 2603.14887
</details>

<details>
<summary><strong>Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning</strong> - Ethan Rathbun, Wo Wei Lin, Alina Oprea, Christopher Amato - [[pdf]](https://arxiv.org/pdf/2602.05089)</summary>

**Abstract:** Simulated environments are a key piece in the success of Reinforcement Learning (RL), allowing practitioners and researchers to train decision making agents without running expensive experiments on real hardware. Simulators remain a security blind spot, however, enabling adversarial developers to alter the dynamics of their released simulators for malicious purposes. Therefore, in this work we highlight a novel threat, demonstrating how simulator dynamics can be exploited to stealthily implant action-level backdoors into RL agents. The backdoor then allows an adversary to reliably activate targeted actions in an agent upon observing a predefined ``trigger'', leading to potentially dangerous consequences. Traditional backdoor attacks are limited in their strong threat models, assuming the adversary has near full control over an agent's training pipeline, enabling them to both alter and observe agent's rewards. As these assumptions are infeasible to implement within a simulator, we propose a new attack ``Daze'' which is able to reliably and stealthily implant backdoors into RL agents trained for real world tasks without altering or even observing their rewards. We provide formal proof of Daze's effectiveness in guaranteeing attack success across general RL tasks along with extensive empirical evaluations on both discrete and continuous action space domains. We additionally provide the first example of RL backdoor attacks transferring to real, robotic hardware. These developments motivate further research into securing all components of the RL training pipeline to prevent malicious attacks.

**arXiv ID:** 2602.05089
</details>

<details>
<summary><strong>ViSTAR: Virtual Skill Training with Augmented Reality with 3D Avatars and LLM coaching agent</strong> - Chunggi Lee, Hayato Saiki, Tica Lin, Eiji Ikeda, Kenji Suzuki, Chen Zhu-Tian, Hanspeter Pfister - [[pdf]](https://arxiv.org/pdf/2602.22077)</summary>

**Abstract:** We present ViSTAR, a Virtual Skill Training system in AR that supports self-guided basketball skill practice, with feedback on balance, posture, and timing. From a formative study with basketball players and coaches, the system addresses three challenges: understanding skills, identifying errors, and correcting mistakes. ViSTAR follows the Behavioral Skills Training (BST) framework-instruction, modeling, rehearsal, and feedback. It provides feedback through visual overlays, rhythm and timing cues, and an AI-powered coaching agent using 3D motion reconstruction. We generate verbal feedback by analyzing spatio-temporal joint data and mapping features to natural-language coaching cues via a Large Language Model (LLM). A key novelty is this feedback generation: motion features become concise coaching insights. In two studies (N=16), participants generally preferred our AI-generated feedback to coach feedback and reported that ViSTAR helped them notice posture and balance issues and refine movements beyond self-observation.

**arXiv ID:** 2602.22077
</details>

<details>
<summary><strong>"I'm Not Reading All of That": Understanding Software Engineers' Level of Cognitive Engagement with Agentic Coding Assistants</strong> - Carlos Rafael Catalan, Lheane Marie Dizon, Patricia Nicole Monderin, Emily Kuang - [[pdf]](https://arxiv.org/pdf/2603.14225)</summary>

**Abstract:** Over-reliance on AI systems can undermine users' critical thinking and promote complacency, a risk intensified by the emergence of agentic AI systems that operate with minimal human involvement. In software engineering, agentic coding assistants (ACAs) are rapidly becoming embedded in everyday development workflows. Since software engineers (SEs) create systems deployed across diverse and high-stakes real-world contexts, these assistants must function not merely as autonomous task performers but as Tools for Thought that actively support human reasoning and sensemaking. We conducted a formative study examining software engineers' cognitive engagement and sensemaking processes when working with an ACA. Our findings reveal that cognitive engagement consistently declines as tasks progress, and that current ACA designs provide limited affordances for reflection, verification, and meaning-making. Based on these findings, we identify concrete design opportunities leveraging richer interaction modalities and cognitive-forcing mechanisms to sustain engagement and promote deeper thinking in AI-assisted programming.

**arXiv ID:** 2603.14225
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
