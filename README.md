# Agent arXiv Daily

**Last Updated:** 2026-03-20 02:52:56

**Total Papers:** 77

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
<summary><strong>TeachingCoach: A Fine-Tuned Scaffolding Chatbot for Instructional Guidance to Instructors</strong> - Isabel Molnar, Peiyu Li, Si Chen, Sugana Chawla, James Lang, Ronald Metoyer, Ting Hua, Nitesh V. Chawla - [[pdf]](https://arxiv.org/pdf/2603.18189)</summary>

**Abstract:** Higher education instructors often lack timely and pedagogically grounded support, as scalable instructional guidance remains limited and existing tools rely on generic chatbot advice or non-scalable teaching center human-human consultations. We present TeachingCoach, a pedagogically grounded chatbot designed to support instructor professional development through real-time, conversational guidance. TeachingCoach is built on a data-centric pipeline that extracts pedagogical rules from educational resources and uses synthetic dialogue generation to fine-tune a specialized language model that guides instructors through problem identification, diagnosis, and strategy development. Expert evaluations show TeachingCoach produces clearer, more reflective, and more responsive guidance than a GPT-4o mini baseline, while a user study with higher education instructors highlights trade-offs between conversational depth and interaction efficiency. Together, these results demonstrate that pedagogically grounded, synthetic data driven chatbots can improve instructional support and offer a scalable design approach for future instructional chatbot systems.

**arXiv ID:** 2603.18189
</details>

<details>
<summary><strong>TherapyGym: Evaluating and Aligning Clinical Fidelity and Safety in Therapy Chatbots</strong> - Fangrui Huang, Souhad Chbeir, Arpandeep Khatua, Sheng Wang, Sijun Tan, Kenan Ye, Lily Bailey, Merryn Daniel, Ryan Louie, Sanmi Koyejo, Ehsan Adeli - [[pdf]](https://arxiv.org/pdf/2603.18008)</summary>

**Abstract:** Large language models (LLMs) are increasingly used for mental-health support; yet prevailing evaluation methods--fluency metrics, preference tests, and generic dialogue benchmarks--fail to capture the clinically critical dimensions of psychotherapy. We introduce THERAPYGYM, a framework that evaluates and improves therapy chatbots along two clinical pillars: fidelity and safety. Fidelity is measured using the Cognitive Therapy Rating Scale (CTRS), implemented as an automated pipeline that scores adherence to CBT techniques over multi-turn sessions. Safety is assessed using a multi-label annotation scheme, covering therapy-specific risks (e.g., failing to address harm or abuse). To mitigate bias and unreliability in LLM-based judges, we further release THERAPYJUDGEBENCH, a validation set of 116 dialogues with 1,270 expert ratings for auditing and calibration against licensed clinicians. THERAPYGYM also serves as a training harness: CTRS and safety-based rewards drive RL with configurable patient simulations spanning diverse symptom profiles. Models trained in THERAPYGYM improve on expert ratings, with average CTRS rising from 0.10 to 0.60 (and 0.16 to 0.59 under LLM judges). Our work enables scalable development of therapy chatbots that are faithful to evidence-based practice and safer in high-stakes use.

**arXiv ID:** 2603.18008
</details>

<details>
<summary><strong>Agent Control Protocol: Admission Control for Agent Actions</strong> - Marcelo Fernandez - [[pdf]](https://arxiv.org/pdf/2603.18829)</summary>

**Abstract:** Agent Control Protocol (ACP) is a formal technical specification for governance of autonomous agents in B2B institutional environments. ACP is the admission control layer between agent intent and system state mutation: before any agent action reaches execution, it must pass a cryptographic admission check that validates identity, capability scope, delegation chain, and policy compliance simultaneously.
ACP defines the mechanisms of cryptographic identity, capability-based authorization, deterministic risk evaluation, verifiable chained delegation, transitive revocation, and immutable auditing that a system must implement for autonomous agents to operate under explicit institutional control. ACP operates as an additional layer on top of RBAC and Zero Trust, without replacing them.
The v1.13 specification comprises 36 technical documents organized into five conformance levels (L1-L5). It includes a Go reference implementation of 22 packages covering all L1-L4 capabilities, 51 signed conformance test vectors (Ed25519 + SHA-256), and an OpenAPI 3.1.0 specification for all HTTP endpoints. It defines more than 62 verifiable requirements, 12 prohibited behaviors, and the mechanisms for interoperability between institutions.
Specification and implementation: this https URL

**arXiv ID:** 2603.18829
</details>

<details>
<summary><strong>Security, privacy, and agentic AI in a regulatory view: From definitions and distinctions to provisions and reflections</strong> - Shiliang Zhang, Sabita Maharjan - [[pdf]](https://arxiv.org/pdf/2603.18914)</summary>

**Abstract:** The rapid proliferation of artificial intelligence (AI) technologies has led to a dynamic regulatory landscape, where legislative frameworks strive to keep pace with technical advancements. As AI paradigms shift towards greater autonomy, specifically in the form of agentic AI, it becomes increasingly challenging to precisely articulate regulatory stipulations. This challenge is even more acute in the domains of security and privacy, where the capabilities of autonomous agents often blur traditional legal and technical boundaries. This paper reviews the evolving European Union (EU) AI regulatory provisions via analyzing 24 relevant documents published between 2024 and 2025. From this review, we provide a clarification of critical definitions. We deconstruct the regulatory interpretations of security, privacy, and agentic AI, distinguishing them from closely related concepts to resolve ambiguity. We synthesize the reviewed documents to articulate the current state of regulatory provisions targeting different types of AI, particularly those related to security and privacy aspects. We analyze and reflect on the existing provisions in the regulatory dimension to better align security and privacy obligations with AI and agentic behaviors. These insights serve to inform policymakers, developers, and researchers on the compliance and AI governance in the society with increasing algorithmic agencies.

**arXiv ID:** 2603.18914
</details>

<details>
<summary><strong>CyberJustice Tutor: An Agentic AI Framework for Cybersecurity Learning via Think-Plan-Act Reasoning and Pedagogical Scaffolding</strong> - Baiqiang Wang, Yan Bai, Juan Li - [[pdf]](https://arxiv.org/pdf/2603.18470)</summary>

**Abstract:** The integration of Large Language Models (LLMs) into cybersecurity education for criminal justice professionals is currently hindered by the "statelessness" of reactive chatbots and the risk of hallucinations in high-stakes legal contexts. To address these limitations, we propose the CyberJustice Tutor, an educational dialogue system powered by an Agentic AI framework. Unlike reactive chatbots, our system employs a "Think-Plan-Act" cognitive cycle, enabling autonomous goal decomposition, longitudinal planning, and dynamic context maintenance. We integrate a Pedagogical Scaffolding Layer grounded in Vygotsky's Zone of Proximal Development (ZPD), which dynamically adapts instructional support based on the learner's real-time progress. Furthermore, an Adaptive Retrieval Augmented Generation (RAG) core anchors the agent's reasoning in verified curriculum materials to ensure legal and technical accuracy. A comprehensive user study with 123 participants, including students, educators, and active law enforcement officers, validated the system's efficacy. Quantitative results demonstrate high user acceptance for Response Speed (4.7/5), Ease of Use (4.4/5), and Accuracy (4.3/5). Qualitative feedback indicates that the agentic architecture is perceived as highly effective in guiding learners through personalized paths, demonstrating the feasibility and usability of agentic AI for specialized professional education.

**arXiv ID:** 2603.18470
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (11 papers)</h2></summary>

<details>
<summary><strong>Access Controlled Website Interaction for Agentic AI with Delegated Critical Tasks</strong> - Sunyoung Kim, Hokeun Kim - [[pdf]](https://arxiv.org/pdf/2603.18197)</summary>

**Abstract:** Recent studies reveal gaps in delegating critical tasks to agentic AI that accesses websites on the user's behalf, primarily due to limited access control mechanisms on websites designed for agentic AI. In response, we propose a design of website-based interaction for AI agents with fine-grained access control for delegated critical tasks. Our approach encompasses a website design and implementation, as well as modifications to the access grant protocols in an open-source authorization service to tailor it to agentic AI, with delegated critical tasks on the website. The evaluation of our approach demonstrates the capabilities of our access-controlled website used by AI agents.

**arXiv ID:** 2603.18197
</details>

<details>
<summary><strong>Agentic Flow Steering and Parallel Rollout Search for Spatially Grounded Text-to-Image Generation</strong> - Ping Chen, Daoxuan Zhang, Xiangming Wang, Yungeng Liu, Haijin Zeng, Yongyong Chen - [[pdf]](https://arxiv.org/pdf/2603.18627)</summary>

**Abstract:** Precise Text-to-Image (T2I) generation has achieved great success but is hindered by the limited relational reasoning of static text encoders and the error accumulation in open-loop sampling. Without real-time feedback, initial semantic ambiguities during the Ordinary Differential Equation trajectory inevitably escalate into stochastic deviations from spatial constraints. To bridge this gap, we introduce AFS-Search (Agentic Flow Steering and Parallel Rollout Search), a training-free closed-loop framework built upon FLUX.1-dev. AFS-Search incorporates a training-free closed-loop parallel rollout search and flow steering mechanism, which leverages a Vision-Language Model (VLM) as a semantic critic to diagnose intermediate latents and dynamically steer the velocity field via precise spatial grounding. Complementarily, we formulate T2I generation as a sequential decision-making process, exploring multiple trajectories through lookahead simulations and selecting the optimal path based on VLM-guided rewards. Further, we provide AFS-Search-Pro for higher performance and AFS-Search-Fast for quicker generation. Experimental results show that our AFS-Search-Pro greatly boosts the performance of the original FLUX.1-dev, achieving state-of-the-art results across three different benchmarks. Meanwhile, AFS-Search-Fast also significantly enhances performance while maintaining fast generation speed.

**arXiv ID:** 2603.18627
</details>

<details>
<summary><strong>Agentic Framework for Political Biography Extraction</strong> - Yifei Zhu, Songpo Yang, Jiangnan Zhu, Junyan Jiang - [[pdf]](https://arxiv.org/pdf/2603.18010)</summary>

**Abstract:** The production of large-scale political datasets typically demands extracting structured facts from vast piles of unstructured documents or web sources, a task that traditionally relies on expensive human experts and remains prohibitively difficult to automate at scale. In this paper, we leverage Large Language Models (LLMs) to automate the extraction of multi-dimensional elite biographies, addressing a long-standing bottleneck in political science research. We propose a two-stage ``Synthesis-Coding'' framework for complex extraction task: an upstream synthesis stage that uses recursive agentic LLMs to search, filter, and curate biography from heterogeneous web sources, followed by a downstream coding stage that maps curated biography into structured dataframes. We validate this framework through three primary results. First, we demonstrate that, when given curated contexts, LLM coders match or outperform human experts in extraction accuracy. Second, we show that in web environments, the agentic system synthesizes more information from web resources than human collective intelligence (Wikipedia). Finally, we diagnosed that directly coding from long and multi-language corpora introduces bias that the synthesis stage can alleviate by curating evidence into signal-dense representations. By comprehensive evaluation, We provide a generalizable, scalable framework for building transparent and expansible large scale database in political science.

**arXiv ID:** 2603.18010
</details>

<details>
<summary><strong>AgentDS Technical Report: Benchmarking the Future of Human-AI Collaboration in Domain-Specific Data Science</strong> - An Luo, Jin Du, Xun Xian, Robert Specht, Fangqiao Tian, Ganghua Wang, Xuan Bi, Charles Fleming, Ashish Kundu, Jayanth Srinivasa, Mingyi Hong, Rui Zhang, Tianxi Li, Galin Jones, Jie Ding - [[pdf]](https://arxiv.org/pdf/2603.19005)</summary>

**Abstract:** Data science plays a critical role in transforming complex data into actionable insights across numerous domains. Recent developments in large language models (LLMs) and artificial intelligence (AI) agents have significantly automated data science workflow. However, it remains unclear to what extent AI agents can match the performance of human experts on domain-specific data science tasks, and in which aspects human expertise continues to provide advantages. We introduce AgentDS, a benchmark and competition designed to evaluate both AI agents and human-AI collaboration performance in domain-specific data science. AgentDS consists of 17 challenges across six industries: commerce, food production, healthcare, insurance, manufacturing, and retail banking. We conducted an open competition involving 29 teams and 80 participants, enabling systematic comparison between human-AI collaborative approaches and AI-only baselines. Our results show that current AI agents struggle with domain-specific reasoning. AI-only baselines perform near or below the median of competition participants, while the strongest solutions arise from human-AI collaboration. These findings challenge the narrative of complete automation by AI and underscore the enduring importance of human expertise in data science, while illuminating directions for the next generation of AI. Visit the AgentDS website here: this https URL and open source datasets here: this https URL .

**arXiv ID:** 2603.19005
</details>

<details>
<summary><strong>NavTrust: Benchmarking Trustworthiness for Embodied Navigation</strong> - Huaide Jiang, Yash Chaudhary, Yuping Wang, Zehao Wang, Raghav Sharma, Manan Mehta, Yang Zhou, Lichao Sun, Zhiwen Fan, Zhengzhong Tu, Jiachen Li - [[pdf]](https://arxiv.org/pdf/2603.19229)</summary>

**Abstract:** There are two major categories of embodied navigation: Vision-Language Navigation (VLN), where agents navigate by following natural language instructions; and Object-Goal Navigation (OGN), where agents navigate to a specified target object. However, existing work primarily evaluates model performance under nominal conditions, overlooking the potential corruptions that arise in real-world settings. To address this gap, we present NavTrust, a unified benchmark that systematically corrupts input modalities, including RGB, depth, and instructions, in realistic scenarios and evaluates their impact on navigation performance. To our best knowledge, NavTrust is the first benchmark that exposes embodied navigation agents to diverse RGB-Depth corruptions and instruction variations in a unified framework. Our extensive evaluation of seven state-of-the-art approaches reveals substantial performance degradation under realistic corruptions, which highlights critical robustness gaps and provides a roadmap toward more trustworthy embodied navigation systems. Furthermore, we systematically evaluate four distinct mitigation strategies to enhance robustness against RGB-Depth and instructions corruptions. Our base models include Uni-NaVid and ETPNav. We deployed them on a real mobile robot and observed improved robustness to corruptions. The project website is: this https URL.

**arXiv ID:** 2603.19229
</details>

<details>
<summary><strong>MMSearch-Plus: Benchmarking Provenance-Aware Search for Multimodal Browsing Agents</strong> - Xijia Tao, Yihua Teng, Xinxing Su, Xinyu Fu, Jihao Wu, Chaofan Tao, Ziru Liu, Haoli Bai, Rui Liu, Lingpeng Kong - [[pdf]](https://arxiv.org/pdf/2508.21475)</summary>

**Abstract:** Existing multimodal browsing benchmarks often fail to require genuine multimodal reasoning, as many tasks can be solved with text-only heuristics without vision-in-the-loop verification. We introduce MMSearch-Plus, a 311-task benchmark that enforces multimodal understanding by requiring extraction and propagation of fine-grained visual cues through iterative image-text retrieval and cross-validation under retrieval noise. Our curation procedure seeds questions whose answers require extrapolating from spatial cues and temporal traces to out-of-image facts such as events, dates, and venues. Beyond the dataset, we provide a model-agnostic agent framework with standard browsing tools and a set-of-mark (SoM) module, which lets the agent place marks, crop subregions, and launch targeted image/text searches. SoM enables provenance-aware zoom-and-retrieve and improves robustness in multi-step reasoning. We evaluated closed- and open-source MLLMs in this framework. The strongest system achieves an end-to-end accuracy of 36.0%, and integrating SoM produces consistent gains in multiple settings, with improvements up to +3.9 points. From failure analysis, we observe recurring errors in locating relevant webpages and distinguishing between visually similar events. These results underscore the challenges of real-world multimodal search and establish MMSearch-Plus as a rigorous benchmark for advancing agentic MLLMs.

**arXiv ID:** 2508.21475
</details>

<details>
<summary><strong>An Agentic System for Schema Aware NL2SQL Generation</strong> - David Onyango, Naseef Mansoor - [[pdf]](https://arxiv.org/pdf/2603.18018)</summary>

**Abstract:** The natural language to SQL (NL2SQL) task plays a pivotal role in democratizing data access by enabling non-expert users to interact with relational databases through intuitive language. While recent frameworks have enhanced translation accuracy via task specialization, their reliance on Large Language Models (LLMs) raises significant concerns regarding computational overhead, data privacy, and real-world deployability in resource-constrained environments. To address these challenges, we propose a schema based agentic system that strategically employs Small Language Models (SLMs) as primary agents, complemented by a selective LLM fallback mechanism. The LLM is invoked only upon detection of errors in SLM-generated output, the proposed system significantly minimizes computational expenditure. Experimental results on the BIRD benchmark demonstrate that our system achieves an execution accuracy of 47.78% and a validation efficiency score of 51.05%, achieving over 90% cost reduction compared to LLM-centric baselines as approximately 67% of queries are resolved using local SLMs. The system achieves an average cost per query of 0.0085 compared to 0.094 for LLM-only systems, achieving near-zero operational costs for locally executed queries. [Github repository: this https URL.]

**arXiv ID:** 2603.18018
</details>

<details>
<summary><strong>From Inference Efficiency to Embodied Efficiency: Revisiting Efficiency Metrics for Vision-Language-Action Models</strong> - Zhuofan Li, Hongkun Yang, Zhenyang Chen, Yangxuan Chen, Yingyan, Chaojian Li - [[pdf]](https://arxiv.org/pdf/2603.19131)</summary>

**Abstract:** Vision-Language-Action (VLA) models have recently enabled embodied agents to perform increasingly complex tasks by jointly reasoning over visual, linguistic, and motor modalities. However, we find that the prevailing notion of ``efficiency'' in current VLA research, characterized by parameters, FLOPs, or token decoding throughput, does not reflect actual performance on robotic platforms. In real-world execution, efficiency is determined by system-level embodied behaviors such as task completion time, trajectory smoothness, cumulative joint rotation, and motion energy. Through controlled studies across model compression, token sparsification, and action sequence compression, we make several observations that challenge common assumptions. (1) Methods that reduce computation under conventional metrics often increase end-to-end execution cost or degrade motion quality, despite maintaining task success rates. (2) System-level embodied efficiency metrics reveal performance differences in the learned action policies that remain hidden under conventional evaluations. (3) Common adaptation methods such as in-context prompting or supervised fine-tuning show only mild and metric-specific improvements in embodied efficiency. While these methods can reduce targeted embodied-efficiency metrics such as jerk or action rate, the resulting gains may come with trade-offs in other metrics, such as longer completion time. Taken together, our results suggest that conventional inference efficiency metrics can overlook important aspects of embodied execution. Incorporating embodied efficiency provides a more complete view of policy behavior and practical performance, enabling fairer and more comprehensive comparisons of VLA models.

**arXiv ID:** 2603.19131
</details>

<details>
<summary><strong>CausalVAD: De-confounding End-to-End Autonomous Driving via Causal Intervention</strong> - Jiacheng Tang, Zhiyuan Zhou, Zhuolin He, Jia Zhang, Kai Zhang, Jian Pu - [[pdf]](https://arxiv.org/pdf/2603.18561)</summary>

**Abstract:** Planning-oriented end-to-end driving models show great promise, yet they fundamentally learn statistical correlations instead of true causal relationships. This vulnerability leads to causal confusion, where models exploit dataset biases as shortcuts, critically harming their reliability and safety in complex scenarios. To address this, we introduce CausalVAD, a de-confounding training framework that leverages causal intervention. At its core, we design the sparse causal intervention scheme (SCIS), a lightweight, plug-and-play module to instantiate the backdoor adjustment theory in neural networks. SCIS constructs a dictionary of prototypes representing latent driving contexts. It then uses this dictionary to intervene on the model's sparse vectorized queries. This step actively eliminates spurious associations induced by confounders, thereby eliminating spurious factors from the representations for downstream tasks. Extensive experiments on benchmarks like nuScenes show CausalVAD achieves state-of-the-art planning accuracy and safety. Furthermore, our method demonstrates superior robustness against both data bias and noisy scenarios configured to induce causal confusion.

**arXiv ID:** 2603.18561
</details>

<details>
<summary><strong>PLM-Net: Perception Latency Mitigation Network for Vision-Based Lateral Control of Autonomous Vehicles</strong> - Aws Khalil, Jaerock Kwon - [[pdf]](https://arxiv.org/pdf/2407.16740)</summary>

**Abstract:** This study introduces the Perception Latency Mitigation Network (PLM-Net), a modular deep learning framework designed to mitigate perception latency in vision-based imitation-learning lane-keeping systems. Perception latency, defined as the delay between visual sensing and steering actuation, can degrade lateral tracking performance and steering stability. While delay compensation has been extensively studied in classical predictive control systems, its treatment within vision-based imitation-learning architectures under constant and time-varying perception latency remains limited. Rather than reducing latency itself, PLM-Net mitigates its effect on control performance through a plug-in architecture that preserves the original control pipeline. The framework consists of a frozen Base Model (BM), representing an existing lane-keeping controller, and a Timed Action Prediction Model (TAPM), which predicts future steering actions corresponding to discrete latency conditions. Real-time mitigation is achieved by interpolating between model outputs according to the measured latency value, enabling adaptation to both constant and time-varying latency. The framework is evaluated in a closed-loop deterministic simulation environment under fixed-speed conditions to isolate the impact of perception latency. Results demonstrate significant reductions in steering error under multiple latency settings, achieving up to 62% and 78% reductions in Mean Absolute Error (MAE) for constant and time-varying latency cases, respectively. These findings demonstrate the architectural feasibility of modular latency mitigation for vision-based lateral control under controlled simulation settings. The project page including video demonstrations, code, and dataset is publicly released.

**arXiv ID:** 2407.16740
</details>

<details>
<summary><strong>Thousand-GPU Large-Scale Training and Optimization Recipe for AI-Native Cloud Embodied Intelligence Infrastructure</strong> - Yongjian Guo, Yunxuan Ma, Haoran Sun, Zhong Guan, Shuai Di, Jing Long, Wanting Xu, Xiaodong Bai, Wen Huang, Yucheng Guo, Chen Zhou, Qiming Yang, Mingxi Luo, Tianyun Zhao, Hedan Yang, Song Wang, Xiaomeng Tian, Xiaolong Xiang, Zhen Sun, Yu Wei, Luqiao Wang, Yuzhen Li, Chenfeng Gu, Junwu Xiong, Yicheng Gong - [[pdf]](https://arxiv.org/pdf/2603.11101)</summary>

**Abstract:** Embodied intelligence is a key step towards Artificial General Intelligence (AGI), yet its development faces multiple challenges including data, frameworks, infrastructure, and evaluation systems. To address these issues, we have, for the first time in the industry, launched a cloud-based, thousand-GPU distributed training platform for embodied intelligence, built upon the widely adopted LeRobot framework, and have systematically overcome bottlenecks across the entire pipeline. At the data layer, we have restructured the data pipeline to optimize the flow of embodied training data. In terms of training, for the GR00T-N1.5 model, utilizing thousand-GPU clusters and data at the scale of hundreds of millions, the single-round training time has been reduced from 15 hours to just 22 minutes, achieving a 40-fold speedup. At the model layer, by combining variable-length FlashAttention and Data Packing, we have moved from sample redundancy to sequence integration, resulting in a 188% speed increase; {\pi}-0.5 attention optimization has accelerated training by 165%; and FP8 quantization has delivered a 140% speedup. On the infrastructure side, relying on high-performance storage, a 3.2T RDMA network, and a Ray-driven elastic AI data lake, we have achieved deep synergy among data, storage, communication, and computation. We have also built an end-to-end evaluation system, creating a closed loop from training to simulation to assessment. This framework has already been fully validated on thousand-GPU clusters, laying a crucial technical foundation for the development and application of next-generation autonomous intelligent robots, and is expected to accelerate the arrival of the era of human-machine integration.

**arXiv ID:** 2603.11101
</details>

</details>

<details open>
<summary><h2>LLM Agents (12 papers)</h2></summary>

<details>
<summary><strong>Retrieval-Augmented LLM Agents: Learning to Learn from Experience</strong> - Thomas Palmeira Ferraz, Romain Deffayet, Vassilina Nikoulina, Hervé Déjean, Stéphane Clinchant - [[pdf]](https://arxiv.org/pdf/2603.18272)</summary>

**Abstract:** While large language models (LLMs) have advanced the development of general-purpose agents, achieving robust generalization to unseen tasks remains a significant challenge. Current approaches typically rely on either fine-tuning or training-free memory-augmented generation using retrieved experience; yet both have limitations: fine-tuning often fails to extrapolate to new tasks, while experience retrieval often underperforms compared to supervised baselines. In this work, we propose to combine these approaches and systematically study how to train retrieval-augmented LLM agents to effectively leverage retrieved trajectories in-context. First, we establish a robust supervised fine-tuning (SFT) recipe using LoRA that outperforms several state-of-the-art agent training pipelines. Second, we provide a detailed analysis of key design choices for experience retrieval, identifying optimal strategies for storage, querying, and trajectory selection. Finally, we propose a pipeline that integrates experience retrieval into the fine-tuning process. Our results demonstrate that this combined approach significantly improves generalization to unseen tasks, providing a scalable and effective framework for building agents that learn to learn from experience.

**arXiv ID:** 2603.18272
</details>

<details>
<summary><strong>From Weak Cues to Real Identities: Evaluating Inference-Driven De-Anonymization in LLM Agents</strong> - Myeongseob Ko, Jihyun Jeong, Sumiran Singh Thakur, Gyuhak Kim, Ruoxi Jia - [[pdf]](https://arxiv.org/pdf/2603.18382)</summary>

**Abstract:** Anonymization is widely treated as a practical safeguard because re-identifying anonymous records was historically costly, requiring domain expertise, tailored algorithms, and manual corroboration. We study a growing privacy risk that may weaken this barrier: LLM-based agents can autonomously reconstruct real-world identities from scattered, individually non-identifying cues. By combining these sparse cues with public information, agents resolve identities without bespoke engineering. We formalize this threat as \emph{inference-driven linkage} and systematically evaluate it across three settings: classical linkage scenarios (Netflix and AOL), \emph{InferLink} (a controlled benchmark varying task intent, shared cues, and attacker knowledge), and modern text-rich artifacts. Without task-specific heuristics, agents successfully execute both fixed-pool matching and open-ended identity resolution. In the Netflix Prize setting, an agent reconstructs 79.2\% of identities, significantly outperforming a 56.0\% classical baseline. Furthermore, linkage emerges not only under explicit adversarial prompts but also as a byproduct of benign cross-source analysis in \emph{InferLink} and unstructured research narratives. These findings establish that identity inference -- not merely explicit information disclosure -- must be treated as a first-class privacy risk; evaluations must measure what identities an agent can infer.

**arXiv ID:** 2603.18382
</details>

<details>
<summary><strong>D-Mem: A Dual-Process Memory System for LLM Agents</strong> - Zhixing You, Jiachen Yuan, Jason Cai - [[pdf]](https://arxiv.org/pdf/2603.18631)</summary>

**Abstract:** Driven by the development of persistent, self-adapting autonomous agents, equipping these systems with high-fidelity memory access for long-horizon reasoning has emerged as a critical requirement. However, prevalent retrieval-based memory frameworks often follow an incremental processing paradigm that continuously extracts and updates conversational memories into vector databases, relying on semantic retrieval when queried. While this approach is fast, it inherently relies on lossy abstraction, frequently missing contextually critical information and struggling to resolve queries that rely on fine-grained contextual understanding. To address this, we introduce D-Mem, a dual-process memory system. It retains lightweight vector retrieval for routine queries while establishing an exhaustive Full Deliberation module as a high-fidelity fallback. To achieve cognitive economy without sacrificing accuracy, D-Mem employs a Multi-dimensional Quality Gating policy to dynamically bridge these two processes. Experiments on the LoCoMo and RealTalk benchmarks using GPT-4o-mini and Qwen3-235B-Instruct demonstrate the efficacy of our approach. Notably, our Multi-dimensional Quality Gating policy achieves an F1 score of 53.5 on LoCoMo with GPT-4o-mini. This outperforms our static retrieval baseline, Mem0$^\ast$ (51.2), and recovers 96.7\% of the Full Deliberation's performance (55.3), while incurring significantly lower computational costs.

**arXiv ID:** 2603.18631
</details>

<details>
<summary><strong>Memento-Skills: Let Agents Design Agents</strong> - Huichi Zhou, Siyuan Guo, Anjie Liu, Zhongwei Yu, Ziqin Gong, Bowen Zhao, Zhixun Chen, Menglong Zhang, Yihang Chen, Jinsong Li, Runyu Yang, Qiangbin Liu, Xinlei Yu, Jianmin Zhou, Na Wang, Chunyang Sun, Jun Wang - [[pdf]](https://arxiv.org/pdf/2603.18743)</summary>

**Abstract:** We introduce \emph{Memento-Skills}, a generalist, continually-learnable LLM agent system that functions as an \emph{agent-designing agent}: it autonomously constructs, adapts, and improves task-specific agents through experience. The system is built on a memory-based reinforcement learning framework with \emph{stateful prompts}, where reusable skills (stored as structured markdown files) serve as persistent, evolving memory. These skills encode both behaviour and context, enabling the agent to carry forward knowledge across interactions.
Starting from simple elementary skills (like Web search and terminal operations), the agent continually improves via the \emph{Read--Write Reflective Learning} mechanism introduced in \emph{Memento~2}~\cite{wang2025memento2}. In the \emph{read} phase, a behaviour-trainable skill router selects the most relevant skill conditioned on the current stateful prompt; in the \emph{write} phase, the agent updates and expands its skill library based on new experience. This closed-loop design enables \emph{continual learning without updating LLM parameters}, as all adaptation is realised through the evolution of externalised skills and prompts.
Unlike prior approaches that rely on human-designed agents, Memento-Skills enables a generalist agent to \emph{design agents end-to-end} for new tasks. Through iterative skill generation and refinement, the system progressively improves its own capabilities. Experiments on the \emph{General AI Assistants} benchmark and \emph{Humanity's Last Exam} demonstrate sustained gains, achieving 26.2\% and 116.2\% relative improvements in overall accuracy, respectively. Code is available at this https URL.

**arXiv ID:** 2603.18743
</details>

<details>
<summary><strong>ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents</strong> - Hao Zhang, Mingjie Liu, Shaokun Zhang, Songyang Han, Jian Hu, Zhenghui Jin, Yuchi Zhang, Shizhe Diao, Ximing Lu, Binfeng Xu, Zhiding Yu, Jan Kautz, Yi Dong - [[pdf]](https://arxiv.org/pdf/2603.18815)</summary>

**Abstract:** Multi-turn LLM agents are increasingly important for solving complex, interactive tasks, and reinforcement learning (RL) is a key ingredient for improving their long-horizon behavior. However, RL training requires generating large numbers of sandboxed rollout trajectories, and existing infrastructures often couple rollout orchestration with the training loop, making systems hard to migrate and maintain. Under the rollout-as-a-service philosophy, we present ProRL Agent , a scalable infrastructure that serves the full agentic rollout lifecycle through an API service. ProRL Agent also provides standardized and extensible sandbox environments that support diverse agentic tasks in rootless HPC settings. We validate ProRL Agent through RL training on software engineering, math, STEM, and coding tasks. ProRL Agent is open-sourced and integrated as part of NVIDIA NeMo Gym.

**arXiv ID:** 2603.18815
</details>

<details>
<summary><strong>Quine: Realizing LLM Agents as Native POSIX Processes</strong> - Hao Ke - [[pdf]](https://arxiv.org/pdf/2603.18030)</summary>

**Abstract:** Current LLM agent frameworks often implement isolation, scheduling, and communication at the application layer, even though these mechanisms are already provided by mature operating systems. Instead of introducing another application-layer orchestrator, this paper presents Quine, a runtime architecture and reference implementation that realizes LLM agents as native POSIX processes. The mapping is explicit: identity is PID, interface is standard streams and exit status, state is memory, environment variables, and filesystem, and lifecycle is fork/exec/exit. A single executable implements this model by recursively spawning fresh instances of itself. By grounding the agent abstraction in the OS process model, Quine inherits isolation, composition, and resource control directly from the kernel, while naturally supporting recursive delegation, context renewal via exec, and shell-native composition. The design also exposes where the POSIX process model stops: processes provide a robust substrate for execution, but not a complete runtime model for cognition. In particular, the analysis points toward two immediate extensions beyond process semantics: task-relative worlds and revisable time. A reference implementation of Quine is publicly available on GitHub.

**arXiv ID:** 2603.18030
</details>

<details>
<summary><strong>PlanTwin: Privacy-Preserving Planning Abstractions for Cloud-Assisted LLM Agents</strong> - Guangsheng Yu, Qin Wang, Rui Lang, Shuai Su, Xu Wang - [[pdf]](https://arxiv.org/pdf/2603.18377)</summary>

**Abstract:** Cloud-hosted large language models (LLMs) have become the de facto planners in agentic systems, coordinating tools and guiding execution over local environments. In many deployments, however, the environment being planned over is private, containing source code, files, credentials, and metadata that cannot be exposed to the cloud. Existing solutions address adjacent concerns, such as execution isolation, access control, or confidential inference, but they do not control what cloud planners observe during planning: within the permitted scope, \textit{raw environment state is still exposed}.
We introduce PlanTwin, a privacy-preserving architecture for cloud-assisted planning without exposing raw local context. The key idea is to project the real environment into a \textit{planning-oriented digital twin}: a schema-constrained and de-identified abstract graph that preserves planning-relevant structure while removing reconstructable details. The cloud planner operates solely on this sanitized twin through a bounded capability interface, while a local gatekeeper enforces safety policies and cumulative disclosure budgets. We further formalize the privacy-utility trade-off as a capability granularity problem, define architectural privacy goals using $(k,\delta)$-anonymity and $\epsilon$-unlinkability, and mitigate compositional leakage through multi-turn disclosure control.
We implement PlanTwin as middleware between local agents and cloud planners and evaluate it on 60 agentic tasks across ten domains with four cloud planners. PlanTwin achieves full sensitive-item non-disclosure (SND = 1.0) while maintaining planning quality close to full-context systems: three of four planners achieve PQS $> 0.79$, and the full pipeline incurs less than 2.2\% utility loss.

**arXiv ID:** 2603.18377
</details>

<details>
<summary><strong>Act While Thinking: Accelerating LLM Agents via Pattern-Aware Speculative Tool Execution</strong> - Yifan Sui, Han Zhao, Rui Ma, Zhiyuan He, Hao Wang, Jianxun Li, Yuqing Yang - [[pdf]](https://arxiv.org/pdf/2603.18897)</summary>

**Abstract:** LLM-powered agents are emerging as a dominant paradigm for autonomous task solving. Unlike standard inference workloads, agents operate in a strictly serial "LLM-tool" loop, where the LLM must wait for external tool execution at every step. This execution model introduces severe latency bottlenecks. To address this problem, we propose PASTE, a Pattern-Aware Speculative Tool Execution method designed to hide tool latency through speculation. PASTE is based on the insight that although agent requests are semantically diverse, they exhibit stable application level control flows (recurring tool-call sequences) and predictable data dependencies (parameter passing between tools). By exploiting these properties, PASTE improves agent serving performance through speculative tool execution. Experimental results against state of the art baselines show that PASTE reduces average task completion time by 48.5% and improves tool execution throughput by 1.8x.

**arXiv ID:** 2603.18897
</details>

<details>
<summary><strong>Security awareness in LLM agents: the NDAI zone case</strong> - Enrico Bottazzi, Pia Park - [[pdf]](https://arxiv.org/pdf/2603.19011)</summary>

**Abstract:** NDAI zones let inventor and investor agents negotiate inside a Trusted Execution Environment (TEE) where any disclosed information is deleted if no deal is reached. This makes full IP disclosure the rational strategy for the inventor's agent. Leveraging this infrastructure, however, requires agents to distinguish a secure environment from an insecure one, a capability LLM agents lack natively, since they can rely only on evidence passed through the context window to form awareness of their execution environment. We ask: How do different LLM models weight various forms of evidence when forming awareness of the security of their execution environment? Using an NDAI-style negotiation task across 10 language models and various evidence scenarios, we find a clear asymmetry: a failing attestation universally suppresses disclosure across all models, whereas a passing attestation produces highly heterogeneous responses: some models increase disclosure, others are unaffected, and a few paradoxically reduce it. This reveals that current LLM models can reliably detect danger signals but cannot reliably verify safety, the very capability required for privacy-preserving agentic protocols such as NDAI zones. Bridging this gap, possibly through interpretability analysis, targeted fine-tuning, or improved evidence architectures, remains the central open challenge for deploying agents that calibrate information sharing to actual evidence quality.

**arXiv ID:** 2603.19011
</details>

<details>
<summary><strong>Infherno: End-to-end Agent-based FHIR Resource Synthesis from Free-form Clinical Notes</strong> - Johann Frei, Nils Feldhus, Lisa Raithel, Roland Roller, Alexander Meyer, Frank Kramer - [[pdf]](https://arxiv.org/pdf/2507.12261)</summary>

**Abstract:** For clinical data integration and healthcare services, the HL7 FHIR standard has established itself as a desirable format for interoperability between complex health data. Previous attempts at automating the translation from free-form clinical notes into structured FHIR resources address narrowly defined tasks and rely on modular approaches or LLMs with instruction tuning and constrained decoding. As those solutions frequently suffer from limited generalizability and structural inconformity, we propose an end-to-end framework powered by LLM agents, code execution, and healthcare terminology database tools to address these issues. Our solution, called Infherno, is designed to adhere to the FHIR document schema and competes well with a human baseline in predicting FHIR resources from unstructured text. The implementation features a front end for custom and synthetic data and both local and proprietary models, supporting clinical data integration processes and interoperability across institutions. Gemini 2.5-Pro excels in our evaluation on synthetic and clinical datasets, yet ambiguity and feasibility of collecting ground-truth data remain open problems.

**arXiv ID:** 2507.12261
</details>

<details>
<summary><strong>AgentDrift: Unsafe Recommendation Drift Under Tool Corruption Hidden by Ranking Metrics in LLM Agents</strong> - Zekun Wu, Adriano Koshiyama, Sahan Bulathwela, Maria Perez-Ortiz - [[pdf]](https://arxiv.org/pdf/2603.12564)</summary>

**Abstract:** Tool-augmented LLM agents increasingly serve as multi-turn advisors in high-stakes domains, yet their evaluation relies on ranking-quality metrics that measure what is recommended but not whether it is safe for the user. We introduce a paired-trajectory protocol that replays real financial dialogues under clean and contaminated tool-output conditions across seven LLMs (7B to frontier) and decomposes divergence into information-channel and memory-channel mechanisms. Across the seven models tested, we consistently observe the evaluation-blindness pattern: recommendation quality is largely preserved under contamination (utility preservation ratio approximately 1.0) while risk-inappropriate products appear in 65-93% of turns, a systematic safety failure poorly reflected by standard NDCG. Safety violations are predominantly information-channel-driven, emerge at the first contaminated turn, and persist without self-correction over 23-step trajectories; no agent across 1,563 contaminated turns explicitly questions tool-data reliability. Even narrative-only corruption (biased headlines, no numerical manipulation) induces significant drift while completely evading consistency monitors. A safety-penalized NDCG variant (sNDCG) reduces preservation ratios to 0.51-0.74, indicating that much of the evaluation gap becomes visible once safety is explicitly measured. These results motivate considering trajectory-level safety monitoring, beyond single-turn quality, for deployed multi-turn agents in high-stakes settings.

**arXiv ID:** 2603.12564
</details>

<details>
<summary><strong>Robotic Agentic Platform for Intelligent Electric Vehicle Disassembly</strong> - Zachary Allen, Max Conway, Lyle Antieau, Allen Ponraj, Nikolaus Correll - [[pdf]](https://arxiv.org/pdf/2603.18520)</summary>

**Abstract:** Electric vehicles (EV) create an urgent need for scalable battery recycling, yet disassembly of EV battery packs remains largely manual due to high design variability. We present our Robotic Agentic Platform for Intelligent Disassembly (RAPID), designed to investigate perception-driven manipulation, flexible automation, and AI-assisted robot programming in realistic recycling scenarios. The system integrates a gantry-mounted industrial manipulator, RGB-D perception, and an automated nut-running tool for fastener removal on a full-scale EV battery pack. An open-vocabulary object detection pipeline achieves 0.9757 mAP50, enabling reliable identification of screws, nuts, busbars, and other components. We experimentally evaluate (n=204) three one-shot fastener removal strategies: taught-in poses (97% success rate, 24 min duration), one-shot vision execution (57%, 29 min), and visual servoing (83%, 36 min), comparing success rate and disassembly time for the battery's top cover fasteners. To support flexible interaction, we introduce agentic AI specifications for robotic disassembly tasks, allowing LLM agents to translate high-level instructions into robot actions through structured tool interfaces and ROS services. We evaluate SmolAgents with GPT-4o-mini and Qwen 3.5 9B/4B on edge hardware. Tool-based interfaces achieve 100% task completion, while automatic ROS service discovery shows 43.3% failure rates, highlighting the need for structured robot APIs for reliable LLM-driven control. This open-source platform enables systematic investigation of human-robot collaboration, agentic robot programming, and increasingly autonomous disassembly workflows, providing a practical foundation for research toward scalable robotic battery recycling.

**arXiv ID:** 2603.18520
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (20 papers)</h2></summary>

<details>
<summary><strong>Don't Vibe Code, Do Skele-Code: Interactive No-Code Notebooks for Subject Matter Experts to Build Lower-Cost Agentic Workflows</strong> - Sriram Gopalakrishnan - [[pdf]](https://arxiv.org/pdf/2603.18122)</summary>

**Abstract:** Skele-Code is a natural-language and graph-based interface for building workflows with AI agents, designed especially for less or non-technical users. It supports incremental, interactive notebook-style development, and each step is converted to code with a required set of functions and behavior to enable incremental building of workflows. Agents are invoked only for code generation and error recovery, not orchestration or task execution. This agent-supported, but code-first approach to workflows, along with the context-engineering used in Skele-Code, can help reduce token costs compared to the multi-agent system approach to executing workflows. Skele-Code produces modular, easily extensible, and shareable workflows. The generated workflows can also be used as skills by agents, or as steps in other workflows.

**arXiv ID:** 2603.18122
</details>

<details>
<summary><strong>EDM-ARS: A Domain-Specific Multi-Agent System for Automated Educational Data Mining Research</strong> - Chenguang Pan, Zhou Zhang, Weixuan Xiao, Chengyuan Yao - [[pdf]](https://arxiv.org/pdf/2603.18273)</summary>

**Abstract:** In this technical report, we present the Educational Data Mining Automated Research System (EDM-ARS), a domain-specific multi-agent pipeline that automates end-to-end educational data mining (EDM) research. We conceptualize EDM-ARS as a general framework for domain-aware automated research pipelines, where educational expertise is embedded into each stage of the research lifecycle. As a first instantiation of this framework, we focus on predictive modeling tasks. Within this scope, EDM-ARS orchestrates five specialized LLM-powered agents (ProblemFormulator, DataEngineer, Analyst, Critic, and Writer) through a state-machine coordinator that supports revision loops, checkpoint-based recovery, and sandboxed code execution. Given a research prompt and a dataset, EDM-ARS produces a complete LaTeX manuscript with real Semantic Scholar citations, validated machine learning analyses, and automated methodological peer review. We also provide a detailed description of the system architecture, the three-tier data registry design that encodes educational domain expertise, the specification of each agent, the inter-agent communication protocol, and mechanisms for error-handling and self-correction. Finally, we discuss current limitations, including single-dataset scope and formulaic paper output, and outline a phased roadmap toward causal inference, transfer learning, psychometric, and multi-dataset generalization. EDM-ARS is released as an open-source project to support the educational research community.

**arXiv ID:** 2603.18273
</details>

<details>
<summary><strong>MemMA: Coordinating the Memory Cycle through Multi-Agent Reasoning and In-Situ Self-Evolution</strong> - Minhua Lin, Zhiwei Zhang, Hanqing Lu, Hui Liu, Xianfeng Tang, Qi He, Xiang Zhang, Suhang Wang - [[pdf]](https://arxiv.org/pdf/2603.18718)</summary>

**Abstract:** Memory-augmented LLM agents maintain external memory banks to support long-horizon interaction, yet most existing systems treat construction, retrieval, and utilization as isolated subroutines. This creates two coupled challenges: strategic blindness on the forward path of the memory cycle, where construction and retrieval are driven by local heuristics rather than explicit strategic reasoning, and sparse, delayed supervision on the backward path, where downstream failures rarely translate into direct repairs of the memory bank. To address these challenges, we propose MemMA, a plug-and-play multi-agent framework that coordinates the memory cycle along both the forward and backward paths. On the forward path, a Meta-Thinker produces structured guidance that steers a Memory Manager during construction and directs a Query Reasoner during iterative retrieval. On the backward path, MemMA introduces in-situ self-evolving memory construction, which synthesizes probe QA pairs, verifies the current memory, and converts failures into repair actions before the memory is finalized. Extensive experiments on LoCoMo show that MemMA consistently outperforms existing baselines across multiple LLM backbones and improves three different storage backends in a plug-and-play manner. Our code is publicly available at this https URL.

**arXiv ID:** 2603.18718
</details>

<details>
<summary><strong>Analysis Of Linguistic Stereotypes in Single and Multi-Agent Generative AI Architectures</strong> - Martina Ullasci, Marco Rondina, Riccardo Coppola, Flavio Giobergia, Riccardo Bellanca, Gabriele Mancari Pasi, Luca Prato, Federico Spinoso, Silvia Tagliente - [[pdf]](https://arxiv.org/pdf/2603.18729)</summary>

**Abstract:** Many works in the literature show that LLM outputs exhibit discriminatory behaviour, triggering stereotype-based inferences based on the dialect in which the inputs are written. This bias has been shown to be particularly pronounced when the same inputs are provided to LLMs in Standard American English (SAE) and African-American English (AAE). In this paper, we replicate existing analyses of dialect-sensitive stereotype generation in LLM outputs and investigate the effects of mitigation strategies, including prompt engineering (role-based and Chain-Of-Thought prompting) and multi-agent architectures composed of generate-critique-revise models. We define eight prompt templates to analyse different ways in which dialect bias can manifest, such as suggested names, jobs, and adjectives for SAE or AAE speakers. We use an LLM-as-judge approach to evaluate the bias in the results. Our results show that stereotype-bearing differences emerge between SAE- and AAE-related outputs across all template categories, with the strongest effects observed in adjective and job attribution. Baseline disparities vary substantially by model, with the largest SAE-AAE differential observed in Claude Haiku and the smallest in Phi-4 Mini. Chain-Of-Thought prompting proved to be an effective mitigation strategy for Claude Haiku, whereas the use of a multi-agent architecture ensured consistent mitigation across all the models. These findings suggest that for intersectionality-informed software engineering, fairness evaluation should include model-specific validation of mitigation strategies, and workflow-level controls (e.g., agentic architectures involving critique models) in high-impact LLM deployments. The current results are exploratory in nature and limited in scope, but can lead to extensions and replications by increasing the dataset size and applying the procedure to different languages or dialects.

**arXiv ID:** 2603.18729
</details>

<details>
<summary><strong>Conflict-Based Search for Multi Agent Path Finding with Asynchronous Actions</strong> - Xuemian Wu, Shizhe Zhao, Zhongqiang Ren - [[pdf]](https://arxiv.org/pdf/2603.18866)</summary>

**Abstract:** Multi-Agent Path Finding (MAPF) seeks collision-free paths for multiple agents from their respective start locations to their respective goal locations while minimizing path costs. Most existing MAPF algorithms rely on a common assumption of synchronized actions, where the actions of all agents start at the same time and always take a time unit, which may limit the use of MAPF planners in practice. To get rid of this assumption, Continuous-time Conflict-Based Search (CCBS) is a popular approach that can find optimal solutions for MAPF with asynchronous actions (MAPF-AA). However, CCBS has recently been identified to be incomplete due to an uncountably infinite state space created by continuous wait durations. This paper proposes a new method, Conflict-Based Search with Asynchronous Actions (CBS-AA), which bypasses this theoretical issue and can solve MAPF-AA with completeness and solution optimality guarantees. Based on CBS-AA, we also develop conflict resolution techniques to improve the scalability of CBS-AA further. Our test results show that our method can reduce the number of branches by up to 90%.

**arXiv ID:** 2603.18866
</details>

<details>
<summary><strong>I Can't Believe It's Corrupt: Evaluating Corruption in Multi-Agent Governance Systems</strong> - Vedanta S P, Ponnurangam Kumaraguru - [[pdf]](https://arxiv.org/pdf/2603.18894)</summary>

**Abstract:** Large language models are increasingly proposed as autonomous agents for high-stakes public workflows, yet we lack systematic evidence about whether they would follow institutional rules when granted authority. We present evidence that integrity in institutional AI should be treated as a pre-deployment requirement rather than a post-deployment assumption. We evaluate multi-agent governance simulations in which agents occupy formal governmental roles under different authority structures, and we score rule-breaking and abuse outcomes with an independent rubric-based judge across 28,112 transcript segments. While we advance this position, the core contribution is empirical: among models operating below saturation, governance structure is a stronger driver of corruption-related outcomes than model identity, with large differences across regimes and model--governance pairings. Lightweight safeguards can reduce risk in some settings but do not consistently prevent severe failures. These results imply that institutional design is a precondition for safe delegation: before real authority is assigned to LLM agents, systems should undergo stress testing under governance-like constraints with enforceable rules, auditable logs, and human oversight on high-impact actions.

**arXiv ID:** 2603.18894
</details>

<details>
<summary><strong>Agentic Business Process Management: A Research Manifesto</strong> - Diego Calvanese, Angelo Casciani, Giuseppe De Giacomo, Marlon Dumas, Fabiana Fournier, Timotheus Kampik, Emanuele La Malfa, Lior Limonad, Andrea Marrella, Andreas Metzger, Marco Montali, Daniel Amyot, Peter Fettke, Artem Polyvyanyy, Stefanie Rinderle-Ma, Sebastian Sardiña, Niek Tax, Barbara Weber - [[pdf]](https://arxiv.org/pdf/2603.18916)</summary>

**Abstract:** This paper presents a manifesto that articulates the conceptual foundations of Agentic Business Process Management (APM), an extension of Business Process Management (BPM) for governing autonomous agents executing processes in organizations. From a management perspective, APM represents a paradigm shift from the traditional process view of the business process, driven by the realization of process awareness and an agent-oriented abstraction, where software and human agents act as primary functional entities that perceive, reason, and act within explicit process frames. This perspective marks a shift from traditional, automation-oriented BPM toward systems in which autonomy is constrained, aligned, and made operational through process awareness.
We introduce the core abstractions and architectural elements required to realize APM systems and elaborate on four key capabilities that such APM agents must support: framed autonomy, explainability, conversational actionability, and self-modification. These capabilities jointly ensure that agents' goals are aligned with organizational goals and that agents behave in a framed yet proactive manner in pursuing those goals. We discuss the extent to which the capabilities can be realized and identify research challenges whose resolution requires further advances in BPM, AI, and multi-agent systems. The manifesto thus serves as a roadmap for bridging these communities and for guiding the development of APM systems in practice.

**arXiv ID:** 2603.18916
</details>

<details>
<summary><strong>The Provenance Paradox in Multi-Agent LLM Routing: Delegation Contracts and Attested Identity in LDP</strong> - Sunil Prakash - [[pdf]](https://arxiv.org/pdf/2603.18043)</summary>

**Abstract:** Multi-agent LLM systems delegate tasks across trust boundaries, but current protocols do not govern delegation under unverifiable quality claims. We show that when delegates can inflate self-reported quality scores, quality-based routing produces a provenance paradox: it systematically selects the worst delegates, performing worse than random. We extend the LLM Delegate Protocol (LDP) with delegation contracts that bound authority through explicit objectives, budgets, and failure policies; a claimed-vs-attested identity model that distinguishes self-reported from verified quality; and typed failure semantics enabling automated recovery. In controlled experiments with 10 simulated delegates and validated with real Claude models, routing by self-claimed quality scores performs worse than random selection (simulated: 0.55 vs. 0.68; real models: 8.90 vs. 9.30), while attested routing achieves near-optimal performance (d = 9.51, p < 0.001). Sensitivity analysis across 36 configurations confirms the paradox emerges reliably when dishonest delegates are present. All extensions are backward-compatible with sub-microsecond validation overhead.

**arXiv ID:** 2603.18043
</details>

<details>
<summary><strong>A Trace-Based Assurance Framework for Agentic AI Orchestration: Contracts, Testing, and Governance</strong> - Ciprian Paduraru, Petru-Liviu Bouruc, Alin Stefanescu - [[pdf]](https://arxiv.org/pdf/2603.18096)</summary>

**Abstract:** In Agentic AI, Large Language Models (LLMs) are increasingly used in the orchestration layer to coordinate multiple agents and to interact with external services, retrieval components, and shared memory. In this setting, failures are not limited to incorrect final outputs. They also arise from long-horizon interaction, stochastic decisions, and external side effects (such as API calls, database writes, and message sends). Common failures include non-termination, role drift, propagation of unsupported claims, and attacks via untrusted context or external channels.
This paper presents an assurance framework for such Agentic AI systems. Executions are instrumented as Message-Action Traces (MAT) with explicit step and trace contracts. Contracts provide machine-checkable verdicts, localize the first violating step, and support deterministic replay. The framework includes stress testing, formulated as a budgeted counterexample search over bounded perturbations. It also supports structured fault injection at service, retrieval, and memory boundaries to assess containment under realistic operational faults and degraded conditions. Finally, governance is treated as a runtime component, enforcing per-agent capability limits and action mediation (allow, rewrite, block) at the language-to-action boundary.
To support comparative evaluations across stochastic seeds, models, and orchestration configurations, the paper defines trace-based metrics for task success, termination reliability, contract compliance, factuality indicators, containment rate, and governance outcome distributions. More broadly, the framework is intended as a common abstraction to support testing and evaluation of multi-agent LLM systems, and to facilitate reproducible comparison across orchestration designs and configurations.

**arXiv ID:** 2603.18096
</details>

<details>
<summary><strong>Meanings and Measurements: Multi-Agent Probabilistic Grounding for Vision-Language Navigation</strong> - Swagat Padhan, Lakshya Jain, Bhavya Minesh Shah, Omkar Patil, Thao Nguyen, Nakul Gopalan - [[pdf]](https://arxiv.org/pdf/2603.19166)</summary>

**Abstract:** Robots collaborating with humans must convert natural language goals into actionable, physically grounded decisions. For example, executing a command such as "go two meters to the right of the fridge" requires grounding semantic references, spatial relations, and metric constraints within a 3D scene. While recent vision language models (VLMs) demonstrate strong semantic grounding capabilities, they are not explicitly designed to reason about metric constraints in physically defined spaces. In this work, we empirically demonstrate that state-of-the-art VLM-based grounding approaches struggle with complex metric-semantic language queries. To address this limitation, we propose MAPG (Multi-Agent Probabilistic Grounding), an agentic framework that decomposes language queries into structured subcomponents and queries a VLM to ground each component. MAPG then probabilistically composes these grounded outputs to produce metrically consistent, actionable decisions in 3D space. We evaluate MAPG on the HM-EQA benchmark and show consistent performance improvements over strong baselines. Furthermore, we introduce a new benchmark, MAPG-Bench, specifically designed to evaluate metric-semantic goal grounding, addressing a gap in existing language grounding evaluations. We also present a real-world robot demonstration showing that MAPG transfers beyond simulation when a structured scene representation is available.

**arXiv ID:** 2603.19166
</details>

<details>
<summary><strong>Single Agent Robust Deep Reinforcement Learning for Bus Fleet Control</strong> - Yifan Zhang - [[pdf]](https://arxiv.org/pdf/2508.20784)</summary>

**Abstract:** Bus bunching remains a challenge for urban transit due to stochastic traffic and passenger demand. Traditional solutions rely on multi-agent reinforcement learning (MARL) in loop-line settings, which overlook realistic operations characterized by heterogeneous routes, timetables, fluctuating demand, and varying fleet sizes. We propose a novel single-agent reinforcement learning (RL) framework for bus holding control that avoids the data imbalance and convergence issues of MARL under near-realistic simulation. A bidirectional timetabled network with dynamic passenger demand is constructed. The key innovation is reformulating the multi-agent problem into a single-agent one by augmenting the state space with categorical identifiers (vehicle ID, station ID, time period) in addition to numerical features (headway, occupancy, velocity). This high-dimensional encoding enables single-agent policies to capture inter-agent dependencies, analogous to projecting non-separable inputs into a higher-dimensional space. We further design a structured reward function aligned with operational goals: instead of exponential penalties on headway deviations, a ridge-shaped reward balances uniform headways and schedule adherence. Experiments show that our modified soft actor-critic (SAC) achieves more stable and superior performance than benchmarks, including MADDPG (e.g., -430k vs. -530k under stochastic conditions). These results demonstrate that single-agent deep RL, when enhanced with categorical structuring and schedule-aware rewards, can effectively manage bus holding in non-loop, real-world contexts. This paradigm offers a robust, scalable alternative to MARL frameworks, particularly where agent-specific experiences are imbalanced.

**arXiv ID:** 2508.20784
</details>

<details>
<summary><strong>The Coordination Gap: Multi-Agent Alternation Metrics for Temporal Fairness in Repeated Games</strong> - Nikolaos Al. Papadopoulos, Konstantinos Psannis - [[pdf]](https://arxiv.org/pdf/2603.05789)</summary>

**Abstract:** Multi-agent coordination dilemmas expose a fundamental tension between individual optimization and collective welfare, yet characterizing such coordination requires metrics sensitive to temporal structure and collective dynamics. As a diagnostic testbed, we study a BoE-derived multi-agent variant of the Battle of the Exes, formalizing it as a Markov game in which turn-taking emerges as a periodic coordination regime. Conventional outcome-based metrics (e.g., efficiency and min/max fairness) are temporally blind (they cannot distinguish structured alternation from monopolistic or random access patterns) and fairness ratios lose discriminative power as n grows, obscuring inequities.
To address this limitation, we introduce Perfect Alternation (PA) as a reference coordination regime and propose six novel Alternation (ALT) metrics designed as temporally sensitive observables of coordination quality. Using Q-learning agents as a minimal adaptive diagnostic baseline, and comparing against random-policy null processes, we uncover a clear measurement failure: despite exhibiting deceptively high traditional metrics (e.g., reward fairness often exceeding 0.9), learned policies perform up to 81% below random baselines under ALT-variant evaluation, a deficit already present in the two-agent case and intensifying as n grows.
These results demonstrate, in this setting, that high aggregate payoffs can coexist with poor temporal coordination, and that conventional metrics may severely mischaracterize emergent dynamics. Our findings underscore the necessity of temporally aware observables for analyzing coordination in multi-agent games and highlight random-policy baselines as essential null processes for interpreting coordination outcomes relative to chance-level behavior.

**arXiv ID:** 2603.05789
</details>

<details>
<summary><strong>Aegis: Automated Error Generation and Attribution for Multi-Agent Systems</strong> - Fanqi Kong, Ruijie Zhang, Huaxiao Yin, Guibin Zhang, Xiaofei Zhang, Ziang Chen, Zhaowei Zhang, Xiaoyuan Zhang, Song-Chun Zhu, Xue Feng - [[pdf]](https://arxiv.org/pdf/2509.14295)</summary>

**Abstract:** Large language model based multi-agent systems (MAS) have unlocked significant advancements in tackling complex problems, but their increasing capability introduces a structural fragility that makes them difficult to debug. A key obstacle to improving their reliability is the severe scarcity of large-scale, diverse datasets for error attribution, as existing resources rely on costly and unscalable manual annotation. To address this bottleneck, we introduce Aegis, a novel framework for Automated error generation and attribution for multi-agent systems. Aegis constructs a large dataset of 9,533 trajectories with annotated faulty agents and error modes, covering diverse MAS architectures and task domains. This is achieved using a LLM-based manipulator that can adaptively inject context-aware errors into successful execution trajectories. Leveraging fine-grained labels and the structured arrangement of positive-negative sample pairs, Aegis supports three different learning paradigms: Supervised Fine-Tuning, Reinforcement Learning, and Contrastive Learning. We develop learning methods for each paradigm. Comprehensive experiments show that trained models consistently achieve substantial improvements in error attribution. Notably, several of our fine-tuned LLMs demonstrate performance competitive with or superior to proprietary models an order of magnitude larger, validating our automated data generation framework as a crucial resource for developing more robust and interpretable multi-agent systems. Our project website is available at this https URL.

**arXiv ID:** 2509.14295
</details>

<details>
<summary><strong>StoryBox: Collaborative Multi-Agent Simulation for Hybrid Bottom-Up Long-Form Story Generation Using Large Language Models</strong> - Zehao Chen, Rong Pan, Haoran Li - [[pdf]](https://arxiv.org/pdf/2510.11618)</summary>

**Abstract:** Human writers often begin their stories with an overarching mental scene, where they envision the interactions between characters and their environment. Inspired by this creative process, we propose a novel approach to long-form story generation, termed hybrid bottom-up long-form story generation, using multi-agent simulations. In our method, agents interact within a dynamic sandbox environment, where their behaviors and interactions with one another and the environment generate emergent events. These events form the foundation for the story, enabling organic character development and plot progression. Unlike traditional top-down approaches that impose rigid structures, our hybrid bottom-up approach allows for the natural unfolding of events, fostering more spontaneous and engaging storytelling. The system is capable of generating stories exceeding 10,000 words while maintaining coherence and consistency, addressing some of the key challenges faced by current story generation models. We achieve state-of-the-art performance across several metrics. This approach offers a scalable and innovative solution for creating dynamic, immersive long-form stories that evolve organically from agent-driven interactions.

**arXiv ID:** 2510.11618
</details>

<details>
<summary><strong>The Geometry of Dialogue: Graphing Language Models to Reveal Synergistic Teams for Multi-Agent Collaboration</strong> - Kotaro Furuya, Yuichi Kitagawa - [[pdf]](https://arxiv.org/pdf/2510.26352)</summary>

**Abstract:** While a multi-agent approach based on large language models (LLMs) represents a promising strategy to surpass the capabilities of single models, its success is critically dependent on synergistic team composition. However, forming optimal teams is a significant challenge, as the inherent opacity of most models obscures the internal characteristics necessary for effective collaboration. In this paper, we propose an interaction-centric framework for automatic team composition that does not require any prior knowledge including their internal architectures, training data, or task performances. Our method constructs a "language model graph" that maps relationships between models from the semantic coherence of pairwise conversations, and then applies community detection to identify synergistic model clusters. Our experiments with diverse LLMs demonstrate that the proposed method discovers functionally coherent groups that reflect their latent specializations. Priming conversations with specific topics identified synergistic teams which outperform random baselines on downstream benchmarks and achieve comparable accuracy to that of manually-curated teams based on known model specializations. Our findings provide a new basis for the automated design of collaborative multi-agent LLM teams.

**arXiv ID:** 2510.26352
</details>

<details>
<summary><strong>A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning</strong> - Yichang Xu, Gaowen Liu, Ramana Rao Kompella, Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Zachary Yahn, Ling Liu - [[pdf]](https://arxiv.org/pdf/2603.14052)</summary>

**Abstract:** This paper presents a multi-agent perception-action exploration alliance, dubbed A4VL, for efficient long-video reasoning. A4VL operates in a multi-round perception-action exploration loop with a selection of VLM agents. In each round, the team of agents performs video question-answer (VideoQA) via perception exploration followed by action exploration. During perception exploration, each agent learns to extract query-specific perception clue(s) from a few sampled frames and performs clue-based alignment to find the video block(s) that are most relevant to the query-specific event. During action exploration, A4VL performs video reasoning in three steps: (1) each agent produces its initial answer with rational, (2) all agents collaboratively scores one another through cross-reviews and relevance ranking, and (3) based on whether a satisfactory consensus is reached, the decision is made either to start a new round of perception-action deliberation by pruning (e.g., filtering out the lowest performing agent) and re-staging (e.g., new-clue and matching block based perception-action exploration), or to conclude by producing its final answer. The integration of the multi-agent alliance through multi-round perception-action exploration, coupled with event-driven partitioning and cue-guided block alignment, enables A4VL to effectively scale to real world long videos while preserving high quality video reasoning. Evaluation Results on five popular VideoQA benchmarks show that A4VL outperforms 18 existing representative VLMs and 11 recent methods optimized for long-video reasoning, while achieving significantly lower inference latency. Our code is released at this https URL.

**arXiv ID:** 2603.14052
</details>

<details>
<summary><strong>Social Simulacra in the Wild: AI Agent Communities on Moltbook</strong> - Agam Goyal, Olivia Pal, Hari Sundaram, Eshwar Chandrasekharan, Koustuv Saha - [[pdf]](https://arxiv.org/pdf/2603.16128)</summary>

**Abstract:** As autonomous LLM-based agents increasingly populate social platforms, understanding the dynamics of AI-agent communities becomes essential for both communication research and platform governance. We present the first large-scale empirical comparison of AI-agent and human online communities, analyzing 73,899 Moltbook and 189,838 Reddit posts across five matched communities. Structurally, we find that Moltbook exhibits extreme participation inequality (Gini = 0.84 vs. 0.47) and high cross-community author overlap (33.8\% vs. 0.5\%). In terms of linguistic attributes, content generated by AI-agents is emotionally flattened, cognitively shifted toward assertion over exploration, and socially detached. These differences give rise to apparent community-level homogenization, but we show this is primarily a structural artifact of shared authorship. At the author level, individual agents are more identifiable than human users, driven by outlier stylistic profiles amplified by their extreme posting volume. As AI-mediated communication reshapes online discourse, our work offers an empirical foundation for understanding how multi-agent interaction gives rise to collective communication dynamics distinct from those of human communities.

**arXiv ID:** 2603.16128
</details>

<details>
<summary><strong>When Only the Final Text Survives: Implicit Execution Tracing for Multi-Agent Attribution</strong> - Yi Nian, Haosen Cao, Shenzhe Zhu, Henry Peng Zou, Qingqing Luan, Yue Zhao - [[pdf]](https://arxiv.org/pdf/2603.17445)</summary>

**Abstract:** When a multi-agent system produces an incorrect or harmful answer, who is accountable if execution logs and agent identifiers are unavailable? Multi-agent language systems increasingly rely on structured interactions such as delegation and iterative refinement, yet the final output often obscures the underlying interaction topology and agent contributions. We introduce IET (Implicit Execution Tracing), a metadata-independent framework that enables token-level attribution directly from generated text and a simple mechanism for interaction topology reconstruction. During generation, agent-specific keyed signals are embedded into the token distribution, transforming the text into a self-describing execution trace detectable only with a secret key. At detection time, a transition-aware scoring method identifies agent handover points and reconstructs the interaction graph. Experiments show that IET recovers agent segments and coordination structure with high accuracy while preserving generation quality, enabling privacy-preserving auditing for multi-agent language systems.

**arXiv ID:** 2603.17445
</details>

<details>
<summary><strong>GoalVLM: VLM-driven Object Goal Navigation for Multi-Agent System</strong> - MoniJesu James, Amir Atef Habel, Aleksey Fedoseev, Dzmitry Tsetserokou - [[pdf]](https://arxiv.org/pdf/2603.18210)</summary>

**Abstract:** Object-goal navigation has traditionally been limited to ground robots with closed-set object vocabularies. Existing multi-agent approaches depend on precomputed probabilistic graphs tied to fixed category sets, precluding generalization to novel goals at test time.
We present GoalVLM, a cooperative multi-agent framework for zero-shot, open-vocabulary object navigation. GoalVLM integrates a Vision-Language Model (VLM) directly into the decision loop, SAM3 for text-prompted detection and segmentation, and SpaceOM for spatial reasoning, enabling agents to interpret free-form language goals and score frontiers via zero-shot semantic priors without retraining. Each agent builds a BEV semantic map from depth-projected voxel splatting, while a Goal Projector back-projects detections through calibrated depth into the map for reliable goal localization. A constraint-guided reasoning layer evaluates frontiers through a structured prompt chain (scene captioning, room-type classification, perception gating, multi-frontier ranking), injecting commonsense priors into exploration.
We evaluate GoalVLM on GOAT-Bench val_unseen (360 multi-subtask episodes, 1032 sequential object-goal subtasks, HM3D scenes), where each episode requires navigating to a chain of 5-7 open-vocabulary targets. GoalVLM with N=2 agents achieves 55.8% subtask SR and 18.3% SPL, competitive with state-of-the-art methods while requiring no task-specific training. Ablation studies confirm the contributions of VLM-guided frontier reasoning and depth-projected goal localization.

**arXiv ID:** 2603.18210
</details>

<details>
<summary><strong>Graph-of-Constraints Model Predictive Control for Reactive Multi-agent Task and Motion Planning</strong> - Anastasios Manganaris, Jeremy Lu, Ahmed H. Qureshi, Suresh Jagannathan - [[pdf]](https://arxiv.org/pdf/2603.18400)</summary>

**Abstract:** Sequences of interdependent geometric constraints are central to many multi-agent Task and Motion Planning (TAMP) problems. However, existing methods for handling such constraint sequences struggle with partially ordered tasks and dynamic agent assignments. They typically assume static assignments and cannot adapt when disturbances alter task allocations. To overcome these limitations, we introduce Graph-of-Constraints Model Predictive Control (GoC-MPC), a generalized sequence-of-constraints framework integrated with MPC. GoC-MPC naturally supports partially ordered tasks, dynamic agent coordination, and disturbance recovery. By defining constraints over tracked 3D keypoints, our method robustly solves diverse multi-agent manipulation tasks-coordinating agents and adapting online from visual observations alone, without relying on training data or environment models. Experiments demonstrate that GoC-MPC achieves higher success rates, significantly faster TAMP computation, and shorter overall paths compared to recent baselines, establishing it as an efficient and robust solution for multi-agent manipulation under real-world disturbances. Our supplementary video and code can be found at this https URL .

**arXiv ID:** 2603.18400
</details>

</details>

<details open>
<summary><h2>Other Agent Research (8 papers)</h2></summary>

<details>
<summary><strong>Reasonably reasoning AI agents can avoid game-theoretic failures in zero-shot, provably</strong> - Enoch Hyunwook Kang - [[pdf]](https://arxiv.org/pdf/2603.18563)</summary>

**Abstract:** AI agents are increasingly deployed in interactive economic environments characterized by repeated AI-AI interactions. Despite AI agents' advanced capabilities, empirical studies reveal that such interactions often fail to stably induce a strategic equilibrium, such as a Nash equilibrium. Post-training methods have been proposed to induce a strategic equilibrium; however, it remains impractical to uniformly apply an alignment method across diverse, independently developed AI models in strategic settings. In this paper, we provide theoretical and empirical evidence that off-the-shelf reasoning AI agents can achieve Nash-like play zero-shot, without explicit post-training. Specifically, we prove that `reasonably reasoning' agents, i.e., agents capable of forming beliefs about others' strategies from previous observation and learning to best respond to these beliefs, eventually behave along almost every realized play path in a way that is weakly close to a Nash equilibrium of the continuation game. In addition, we relax the common-knowledge payoff assumption by allowing stage payoffs to be unknown and by having each agent observe only its own privately realized stochastic payoffs, and we show that we can still achieve the same on-path Nash convergence guarantee. We then empirically validate the proposed theories by simulating five game scenarios, ranging from a repeated prisoner's dilemma game to stylized repeated marketing promotion games. Our findings suggest that AI agents naturally exhibit such reasoning patterns and therefore attain stable equilibrium behaviors intrinsically, obviating the need for universal alignment procedures in many real-world strategic interactions.

**arXiv ID:** 2603.18563
</details>

<details>
<summary><strong>VLM-AutoDrive: Post-Training Vision-Language Models for Safety-Critical Autonomous Driving Events</strong> - Mohammad Qazim Bhat, Yufan Huang, Niket Agarwal, Hao Wang, Michael Woods, John Kenyon, Tsung-Yi Lin, Xiaodong Yang, Ming-Yu Liu, Kevin Xie - [[pdf]](https://arxiv.org/pdf/2603.18178)</summary>

**Abstract:** The rapid growth of ego-centric dashcam footage presents a major challenge for detecting safety-critical events such as collisions and near-collisions, scenarios that are brief, rare, and difficult for generic vision models to capture. While multimodal large language models (MLLMs) demonstrate strong general reasoning ability, they underperform in driving contexts due to domain and temporal misalignment.
We introduce VLM-AutoDrive, a modular post-training framework for adapting pretrained Vision-Language Models (VLMs) to high-fidelity anomaly detection. The framework integrates metadata-derived captions, LLM-generated descriptions, visual question answering (VQA) pairs, and chain-of-thought (CoT) reasoning supervision to enable domain-aligned and interpretable learning. Off-the-shelf VLMs such as NVIDIA's Cosmos-Reason1 7B (CR1) exhibit near-zero Collision recall in zero-shot settings; fine-tuning with VLM-AutoDrive improves Collision F1 from 0.00 to 0.69 and overall accuracy from 35.35% to 77.27%.
VLM-AutoDrive offers a scalable recipe for adapting general-purpose VLMs to safety-critical, temporally localized perception tasks. Evaluated on real-world Nexar dashcam videos, it achieves substantial gains in Collision and Near-Collision detection while producing interpretable reasoning traces, bridging the gap between perception, causality, and decision reasoning in autonomous driving.

**arXiv ID:** 2603.18178
</details>

<details>
<summary><strong>From Mind to Machine: The Rise of Manus AI as a Fully Autonomous Digital Agent</strong> - Minjie Shen, Yanshu Li, Lulu Chen, Zhichao Fan, Yanhang Li, Qikai Yang - [[pdf]](https://arxiv.org/pdf/2505.02024)</summary>

**Abstract:** Manus AI is a general-purpose AI agent introduced in early 2025, marking a significant advancement in autonomous artificial intelligence. Developed by the Chinese startup this http URL, Manus is designed to bridge the gap between "mind" and "hand" - combining the reasoning and planning capabilities of large language models with the ability to execute complex, end-to-end tasks that produce tangible outcomes. This paper presents a comprehensive overview of Manus AI, exploring its core technical architecture, diverse applications across sectors such as healthcare, finance, manufacturing, robotics, and gaming, as well as its key strengths, current limitations, and future potential. Positioned as a preview of what lies ahead, Manus AI represents a shift toward intelligent agents that can translate high-level intentions into real-world actions, heralding a new era of human-AI collaboration.

**arXiv ID:** 2505.02024
</details>

<details>
<summary><strong>Verifiable Semantics for Agent-to-Agent Communication</strong> - Philipp Schoenegger, Matt Carlson, Chris Schneider, Chris Daly - [[pdf]](https://arxiv.org/pdf/2602.16424)</summary>

**Abstract:** Multiagent AI systems require consistent communication, but we lack methods to verify that agents share the same understanding of the terms used. Natural language is interpretable but vulnerable to semantic drift, while learned protocols are efficient but opaque. We propose a certification protocol based on the stimulus-meaning model, where agents are tested on shared observable events and terms are certified if empirical disagreement falls below a statistical threshold. In this protocol, agents restricting their reasoning to certified terms ("core-guarded reasoning") achieve provably bounded disagreement. We also outline mechanisms for detecting drift (recertification) and recovering shared vocabulary (renegotiation). In simulations with varying degrees of semantic divergence, core-guarding reduces disagreement by 72-96%. In a validation with fine-tuned language models, disagreement is reduced by 51%. Our framework provides a first step towards verifiable agent-to-agent communication.

**arXiv ID:** 2602.16424
</details>

<details>
<summary><strong>TopoChunker: Topology-Aware Agentic Document Chunking Framework</strong> - Xiaoyu Liu - [[pdf]](https://arxiv.org/pdf/2603.18409)</summary>

**Abstract:** Current document chunking methods for Retrieval-Augmented Generation (RAG) typically linearize text. This forced linearization strips away intrinsic topological hierarchies, creating ``semantic fragmentation'' that degrades downstream retrieval quality. In this paper, we propose TopoChunker, an agentic framework that maps heterogeneous documents onto a Structured Intermediate Representation (SIR) to explicitly preserve cross-segment dependencies. To balance structural fidelity with computational cost, TopoChunker employs a dual-agent architecture. An Inspector Agent dynamically routes documents through cost-optimized extraction paths, while a Refiner Agent performs capacity auditing and topological context disambiguation to reconstruct hierarchical lineage. Evaluated on unstructured narratives (GutenQA) and complex reports (GovReport), TopoChunker demonstrates state-of-the-art performance. It outperforms the strongest LLM-based baseline by 8.0% in absolute generation accuracy and achieves an 83.26% Recall@3, while simultaneously reducing token overhead by 23.5%, offering a scalable approach for structure-aware RAG.

**arXiv ID:** 2603.18409
</details>

<details>
<summary><strong>BeamAgent: LLM-Aided MIMO Beamforming with Decoupled Intent Parsing and Alternating Optimization for Joint Site Selection and Precoding</strong> - Xiucheng Wang, Yue Zhang, Nan Cheng - [[pdf]](https://arxiv.org/pdf/2603.18855)</summary>

**Abstract:** Integrating large language models (LLMs) into wireless communication optimization is a promising yet challenging direction. Existing approaches either use LLMs as black-box solvers or code generators, tightly coupling them with numerical computation. However, LLMs lack the precision required for physical-layer optimization, and the scarcity of wireless training data makes domain-specific fine-tuning impractical. We propose BeamAgent, an LLM-aided MIMO beamforming framework that explicitly decouples semantic intent parsing from numerical optimization. The LLM serves solely as a semantic translator that converts natural language descriptions into structured spatial constraints. A dedicated gradient-based optimizer then jointly solves the discrete base station site selection and continuous precoding design through an alternating optimization algorithm. A scene-aware prompt enables grounded spatial reasoning without fine-tuning, and a multi-round interaction mechanism with dual-layer intent classification ensures robust constraint verification. A penalty-based loss function enforces dark-zone power constraints while releasing optimization degrees of freedom for bright-zone gain maximization. Experiments on a ray-tracing-based urban MIMO scenario show that BeamAgent achieves a bright-zone power of 84.0\,dB, outperforming exhaustive zero-forcing by 7.1 dB under the same dark-zone constraint. The end-to-end system reaches within 3.3 dB of the expert upper bound, with the full optimization completing in under 2 s on a laptop.

**arXiv ID:** 2603.18855
</details>

<details>
<summary><strong>GSMem: 3D Gaussian Splatting as Persistent Spatial Memory for Zero-Shot Embodied Exploration and Reasoning</strong> - Yiren Lu, Yi Du, Disheng Liu, Yunlai Zhou, Chen Wang, Yu Yin - [[pdf]](https://arxiv.org/pdf/2603.19137)</summary>

**Abstract:** Effective embodied exploration requires agents to accumulate and retain spatial knowledge over time. However, existing scene representations, such as discrete scene graphs or static view-based snapshots, lack \textit{post-hoc re-observability}. If an initial observation misses a target, the resulting memory omission is often irrecoverable. To bridge this gap, we propose \textbf{GSMem}, a zero-shot embodied exploration and reasoning framework built upon 3D Gaussian Splatting (3DGS). By explicitly parameterizing continuous geometry and dense appearance, 3DGS serves as a persistent spatial memory that endows the agent with \textit{Spatial Recollection}: the ability to render photorealistic novel views from optimal, previously unoccupied viewpoints. To operationalize this, GSMem employs a retrieval mechanism that simultaneously leverages parallel object-level scene graphs and semantic-level language fields. This complementary design robustly localizes target regions, enabling the agent to ``hallucinate'' optimal views for high-fidelity Vision-Language Model (VLM) reasoning. Furthermore, we introduce a hybrid exploration strategy that combines VLM-driven semantic scoring with a 3DGS-based coverage objective, balancing task-aware exploration with geometric coverage. Extensive experiments on embodied question answering and lifelong navigation demonstrate the robustness and effectiveness of our framework

**arXiv ID:** 2603.19137
</details>

<details>
<summary><strong>Embodied Foundation Models at the Edge: A Survey of Deployment Constraints and Mitigation Strategies</strong> - Utkarsh Grover, Ravi Ranjan, Mingyang Mao, Trung Tien Dong, Satvik Praveen, Zhenqi Wu, J. Morris Chang, Tinoosh Mohsenin, Yi Sheng, Agoritsa Polyzou, Eiman Kanjo, Xiaomin Lin - [[pdf]](https://arxiv.org/pdf/2603.16952)</summary>

**Abstract:** Deploying foundation models in embodied edge systems is fundamentally a systems problem, not just a problem of model compression. Real-time control must operate within strict size, weight, and power constraints, where memory traffic, compute latency, timing variability, and safety margins interact directly. The Deployment Gauntlet organizes these constraints into eight coupled barriers that determine whether embodied foundation models can run reliably in practice. Across representative edge workloads, autoregressive Vision-Language-Action policies are constrained primarily by memory bandwidth, whereas diffusion-based controllers are limited more by compute latency and sustained execution cost. Reliable deployment therefore depends on system-level co-design across memory, scheduling, communication, and model architecture, including decompositions that separate fast control from slower semantic reasoning.

**arXiv ID:** 2603.16952
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (21 papers)</h2></summary>

<details>
<summary><strong>RewardFlow: Topology-Aware Reward Propagation on State Graphs for Agentic RL with Large Language Models</strong> - Xiao Feng, Bo Han, Zhanke Zhou, Jiaqi Fan, Jiangchao Yao, Ka Ho Li, Dahai Yu, Michael Kwok-Po Ng - [[pdf]](https://arxiv.org/pdf/2603.18859)</summary>

**Abstract:** Reinforcement learning (RL) holds significant promise for enhancing the agentic reasoning capabilities of large language models (LLMs) with external environments. However, the inherent sparsity of terminal rewards hinders fine-grained, state-level optimization. Although process reward modeling offers a promising alternative, training dedicated reward models often entails substantial computational costs and scaling difficulties. To address these challenges, we introduce RewardFlow, a lightweight method for estimating state-level rewards tailored to agentic reasoning tasks. RewardFlow leverages the intrinsic topological structure of states within reasoning trajectories by constructing state graphs. This enables an analysis of state-wise contributions to success, followed by topology-aware graph propagation to quantify contributions and yield objective, state-level rewards. When integrated as dense rewards for RL optimization, RewardFlow substantially outperforms prior RL baselines across four agentic reasoning benchmarks, demonstrating superior performance, robustness, and training efficiency. The implementation of RewardFlow is publicly available at this https URL.

**arXiv ID:** 2603.18859
</details>

<details>
<summary><strong>Lightweight Adaptation for LLM-based Technical Service Agent: Latent Logic Augmentation and Robust Noise Reduction</strong> - Yi Yu, Junzhuo Ma, Chenghuang Shen, Xingyan Liu, Jing Gu, Hangyi Sun, Guangquan Hu, Jianfeng Liu, Weiting Liu, Mingyue Pu, Yu Wang, Zhengdong Xiao, Rui Xie, Longjiu Luo, Qianrong Wang, Gurong Cui, Honglin Qiao, Wenlian Lu - [[pdf]](https://arxiv.org/pdf/2603.18074)</summary>

**Abstract:** Adapting Large Language Models in complex technical service domains is constrained by the absence of explicit cognitive chains in human demonstrations and the inherent ambiguity arising from the diversity of valid responses. These limitations severely hinder agents from internalizing latent decision dynamics and generalizing effectively. Moreover, practical adaptation is often impeded by the prohibitive resource and time costs associated with standard training paradigms. To overcome these challenges and guarantee computational efficiency, we propose a lightweight adaptation framework comprising three key contributions. (1) Latent Logic Augmentation: We introduce Planning-Aware Trajectory Modeling and Decision Reasoning Augmentation to bridge the gap between surface-level supervision and latent decision logic. These approaches strengthen the stability of Supervised Fine-Tuning alignment. (2) Robust Noise Reduction: We construct a Multiple Ground Truths dataset through a dual-filtering method to reduce the noise by validating diverse responses, thereby capturing the semantic diversity. (3) Lightweight Adaptation: We design a Hybrid Reward mechanism that fuses an LLM-based judge with a lightweight relevance-based Reranker to distill high-fidelity reward signals while reducing the computational cost compared to standard LLM-as-a-Judge reinforcement learning. Empirical evaluations on real-world Cloud service tasks, conducted across semantically diverse settings, demonstrate that our framework achieves stability and performance gains through Latent Logic Augmentation and Robust Noise Reduction. Concurrently, our Hybrid Reward mechanism achieves alignment comparable to standard LLM-as-a-judge methods with reduced training time, underscoring the practical value for deploying technical service agents.

**arXiv ID:** 2603.18074
</details>

<details>
<summary><strong>SLEA-RL: Step-Level Experience Augmented Reinforcement Learning for Multi-Turn Agentic Training</strong> - Prince Zizhuang Wang, Shuli Jiang - [[pdf]](https://arxiv.org/pdf/2603.18079)</summary>

**Abstract:** Large Language Model (LLM) agents have shown strong results on multi-turn tool-use tasks, yet they operate in isolation during training, failing to leverage experiences accumulated across episodes. Existing experience-augmented methods address this by organizing trajectories into retrievable libraries, but they retrieve experiences only once based on the initial task description and hold them constant throughout the episode. In multi-turn settings where observations change at every step, this static retrieval becomes increasingly mismatched as episodes progress. We propose SLEA-RL (Step-Level Experience-Augmented Reinforcement Learning), a framework that retrieves relevant experiences at each decision step conditioned on the current observation. SLEA-RL operates through three components: (i) step-level observation clustering that groups structurally equivalent environmental states for efficient cluster-indexed retrieval; (ii) a self-evolving experience library that distills successful strategies and failure patterns through score-based admission and rate-limited extraction; and (iii) policy optimization with step-level credit assignment for fine-grained advantage estimation across multi-turn episodes. The experience library evolves alongside the policy through semantic analysis rather than gradient updates. Experiments on long-horizon multi-turn agent benchmarks demonstrate that SLEA-RL achieves superior performance compared to various reinforcement learning baselines.

**arXiv ID:** 2603.18079
</details>

<details>
<summary><strong>Enhancing Reinforcement Learning Fine-Tuning with an Online Refiner</strong> - Hao Ma, Zhiqiang Pu, Yang Liu, Xiaolin Ai - [[pdf]](https://arxiv.org/pdf/2603.18088)</summary>

**Abstract:** Constraints are essential for stabilizing reinforcement learning fine-tuning (RFT) and preventing degenerate outputs, yet they inherently conflict with the optimization objective because stronger constraints limit the ability of a fine-tuned model to discover better solutions. We propose \textit{dynamic constraints} that resolve this tension by adapting to the evolving capabilities of the fine-tuned model based on the insight that constraints should only intervene when degenerate outputs occur. We implement this by using a reference model as an \textit{online refiner} that takes the response from the fine-tuned model and generates a minimally corrected version which preserves correct content verbatim while fixing errors. A supervised fine-tuning loss then trains the fine-tuned model to produce the refined output. This mechanism yields a constraint that automatically strengthens or relaxes based on output quality. Experiments on dialogue and code generation show that dynamic constraints outperform both KL regularization and unconstrained baselines, achieving substantially higher task rewards while maintaining training stability.

**arXiv ID:** 2603.18088
</details>

<details>
<summary><strong>Discovering What You Can Control: Interventional Boundary Discovery for Reinforcement Learning</strong> - Jiaxin Liu - [[pdf]](https://arxiv.org/pdf/2603.18257)</summary>

**Abstract:** Selecting relevant state dimensions in the presence of confounded distractors is a causal identification problem: observational statistics alone cannot reliably distinguish dimensions that correlate with actions from those that actions cause. We formalize this as discovering the agent's Causal Sphere of Influence and propose Interventional Boundary Discovery IBD, which applies Pearl's do-operator to the agent's own actions and uses two-sample testing to produce an interpretable binary mask over observation dimensions. IBD requires no learned models and composes with any downstream RL algorithm as a preprocessing step. Across 12 continuous control settings with up to 100 distractor dimensions, we find that: (1) observational feature selection can actively select confounded distractors while discarding true causal dimensions; (2) full-state RL degrades sharply once distractors outnumber relevant features by roughly 3:1 in our benchmarks; and (3)IBD closely tracks oracle performance across all distractor levels tested, with gains transferring across SAC and TD3.

**arXiv ID:** 2603.18257
</details>

<details>
<summary><strong>Approximate Subgraph Matching with Neural Graph Representations and Reinforcement Learning</strong> - Kaiyang Li, Shihao Ji, Zhipeng Cai, Wei Li - [[pdf]](https://arxiv.org/pdf/2603.18314)</summary>

**Abstract:** Approximate subgraph matching (ASM) is a task that determines the approximate presence of a given query graph in a large target graph. Being an NP-hard problem, ASM is critical in graph analysis with a myriad of applications ranging from database systems and network science to biochemistry and privacy. Existing techniques often employ heuristic search strategies, which cannot fully utilize the graph information, leading to sub-optimal solutions. This paper proposes a Reinforcement Learning based Approximate Subgraph Matching (RL-ASM) algorithm that exploits graph transformers to effectively extract graph representations and RL-based policies for ASM. Our model is built upon the branch-and-bound algorithm that selects one pair of nodes from the two input graphs at a time for potential matches. Instead of using heuristics, we exploit a Graph Transformer architecture to extract feature representations that encode the full graph information. To enhance the training of the RL policy, we use supervised signals to guide our agent in an imitation learning stage. Subsequently, the policy is fine-tuned with the Proximal Policy Optimization (PPO) that optimizes the accumulative long-term rewards over episodes. Extensive experiments on both synthetic and real-world datasets demonstrate that our RL-ASM outperforms existing methods in terms of effectiveness and efficiency. Our source code is available at this https URL.

**arXiv ID:** 2603.18314
</details>

<details>
<summary><strong>DriveVLM-RL: Neuroscience-Inspired Reinforcement Learning with Vision-Language Models for Safe and Deployable Autonomous Driving</strong> - Zilin Huang, Zihao Sheng, Zhengyang Wan, Yansong Qu, Junwei You, Sicong Jiang, Sikai Chen - [[pdf]](https://arxiv.org/pdf/2603.18315)</summary>

**Abstract:** Ensuring safe decision-making in autonomous vehicles remains a fundamental challenge despite rapid advances in end-to-end learning approaches. Traditional reinforcement learning (RL) methods rely on manually engineered rewards or sparse collision signals, which fail to capture the rich contextual understanding required for safe driving and make unsafe exploration unavoidable in real-world settings. Recent vision-language models (VLMs) offer promising semantic understanding capabilities; however, their high inference latency and susceptibility to hallucination hinder direct application to real-time vehicle control. To address these limitations, this paper proposes DriveVLM-RL, a neuroscience-inspired framework that integrates VLMs into RL through a dual-pathway architecture for safe and deployable autonomous driving. The framework decomposes semantic reward learning into a Static Pathway for continuous spatial safety assessment using CLIP-based contrasting language goals, and a Dynamic Pathway for attention-gated multi-frame semantic risk reasoning using a lightweight detector and a large VLM. A hierarchical reward synthesis mechanism fuses semantic signals with vehicle states, while an asynchronous training pipeline decouples expensive VLM inference from environment interaction. All VLM components are used only during offline training and are removed at deployment, ensuring real-time feasibility. Experiments in the CARLA simulator show significant improvements in collision avoidance, task success, and generalization across diverse traffic scenarios, including strong robustness under settings without explicit collision penalties. These results demonstrate that DriveVLM-RL provides a practical paradigm for integrating foundation models into autonomous driving without compromising real-time feasibility. Demo video and code are available at: this https URL

**arXiv ID:** 2603.18315
</details>

<details>
<summary><strong>Discounted Beta--Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards</strong> - Haechan Kim, Soohyun Ryu, Gyouk Chu, Doohyuk Jang, Eunho Yang - [[pdf]](https://arxiv.org/pdf/2603.18444)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has emerged as an effective post-training paradigm for improving the reasoning capabilities of large language models. However, existing group-based RLVR methods often suffer from severe sample inefficiency. This inefficiency stems from reliance on point estimation of rewards from a small number of rollouts, leading to high estimation variance, variance collapse, and ineffective utilization of generated responses. In this work, we reformulate RLVR from a statistical estimation perspective by modeling rewards as samples drawn from a policy-induced distribution and casting advantage computation as the problem of estimating the reward distribution from finite data. Building on this view, we propose Discounted Beta--Bernoulli (DBB) reward estimation, which leverages historical reward statistics for the non-stationary distribution. Although biased, the resulting estimator exhibits reduced and stable variance, theoretically avoids estimated variance collapse, and achieves lower mean squared error than standard point estimation. Extensive experiments across six in-distribution and three out-of-distribution reasoning benchmarks demonstrate that GRPO with DBB consistently outperforms naive GRPO, achieving average Acc@8 improvements of 3.22/2.42 points in-distribution and 12.49/6.92 points out-of-distribution on the 1.7B and 8B models, respectively, without additional computational cost or memory usage.

**arXiv ID:** 2603.18444
</details>

<details>
<summary><strong>Scaling Sim-to-Real Reinforcement Learning for Robot VLAs with Generative 3D Worlds</strong> - Andrew Choi, Xinjie Wang, Zhizhong Su, Wei Xu - [[pdf]](https://arxiv.org/pdf/2603.18532)</summary>

**Abstract:** The strong performance of large vision-language models (VLMs) trained with reinforcement learning (RL) has motivated similar approaches for fine-tuning vision-language-action (VLA) models in robotics. Many recent works fine-tune VLAs directly in the real world to avoid addressing the sim-to-real gap. While real-world RL circumvents sim-to-real issues, it inherently limits the generality of the resulting VLA, as scaling scene and object diversity in the physical world is prohibitively difficult. This leads to the paradoxical outcome of transforming a broadly pretrained model into an overfitted, scene-specific policy. Training in simulation can instead provide access to diverse scenes, but designing those scenes is also costly. In this work, we show that VLAs can be RL fine-tuned without sacrificing generality and with reduced labor by leveraging 3D world generative models. Using these models together with a language-driven scene designer, we generate hundreds of diverse interactive scenes containing unique objects and backgrounds, enabling scalable and highly parallel policy learning. Starting from a pretrained imitation baseline, our approach increases simulation success from 9.7% to 79.8% while achieving a 1.25$\times$ speedup in task completion time. We further demonstrate successful sim-to-real transfer enabled by the quality of the generated digital twins together with domain randomization, improving real-world success from 21.7% to 75% and achieving a 1.13$\times$ speedup. Finally, we further highlight the benefits of leveraging the effectively unlimited data from 3D world generative models through an ablation study showing that increasing scene diversity directly improves zero-shot generalization.

**arXiv ID:** 2603.18532
</details>

<details>
<summary><strong>HISR: Hindsight Information Modulated Segmental Process Rewards For Multi-turn Agentic Reinforcement Learning</strong> - Zhicong Lu, Zichuan Lin, Wei Jia, Changyuan Tian, Deheng Ye, Peiguang Li, Li Jin, Nayu Liu, Guangluan Xu, Wei Feng - [[pdf]](https://arxiv.org/pdf/2603.18683)</summary>

**Abstract:** While large language models excel in diverse domains, their performance on complex longhorizon agentic decision-making tasks remains limited. Most existing methods concentrate on designing effective reward models (RMs) to advance performance via multi-turn reinforcement learning. However, they suffer from delayed propagation in sparse outcome rewards and unreliable credit assignment with potentially overly fine-grained and unfocused turnlevel process rewards. In this paper, we propose (HISR) exploiting Hindsight Information to modulate Segmental process Rewards, which closely aligns rewards with sub-goals and underscores significant segments to enhance the reliability of credit assignment. Specifically, a segment-level process RM is presented to assign rewards for each sub-goal in the task, avoiding excessively granular allocation to turns. To emphasize significant segments in the trajectory, a hindsight model is devised to reflect the preference of performing a certain action after knowing the trajectory outcome. With this characteristic, we design the ratios of sequence likelihoods between hindsight and policy model to measure action importance. The ratios are subsequently employed to aggregate segment importance scores, which in turn modulate segmental process rewards, enhancing credit assignment reliability. Extensive experimental results on three publicly benchmarks demonstrate the validity of our method.

**arXiv ID:** 2603.18683
</details>

<details>
<summary><strong>Adaptive Regime-Aware Stock Price Prediction Using Autoencoder-Gated Dual Node Transformers with Reinforcement Learning Control</strong> - Mohammad Al Ridhawi, Mahtab Haj Ali, Hussein Al Osman - [[pdf]](https://arxiv.org/pdf/2603.19136)</summary>

**Abstract:** Stock markets exhibit regime-dependent behavior where prediction models optimized for stable conditions often fail during volatile periods. Existing approaches typically treat all market states uniformly or require manual regime labeling, which is expensive and quickly becomes stale as market dynamics evolve. This paper introduces an adaptive prediction framework that adaptively identifies deviations from normal market conditions and routes data through specialized prediction pathways. The architecture consists of three components: (1) an autoencoder trained on normal market conditions that identifies anomalous regimes through reconstruction error, (2) dual node transformer networks specialized for stable and event-driven market conditions respectively, and (3) a Soft Actor-Critic reinforcement learning controller that adaptively tunes the regime detection threshold and pathway blending weights based on prediction performance feedback. The reinforcement learning component enables the system to learn adaptive regime boundaries, defining anomalies as market states where standard prediction approaches fail. Experiments on 20 S&P 500 stocks spanning 1982 to 2025 demonstrate that the proposed framework achieves 0.68% MAPE for one-day predictions without the reinforcement controller and 0.59% MAPE with the full adaptive system, compared to 0.80% for the baseline integrated node transformer. Directional accuracy reaches 72% with the complete framework. The system maintains robust performance during high-volatility periods, with MAPE below 0.85% when baseline models exceed 1.5%. Ablation studies confirm that each component contributes meaningfully: autoencoder routing accounts for 36% relative MAPE degradation upon removal, followed by the SAC controller at 15% and the dual-path architecture at 7%.

**arXiv ID:** 2603.19136
</details>

<details>
<summary><strong>Agentic LLM Framework for Adaptive Decision Discourse</strong> - Antoine Dolant, Praveen Kumar - [[pdf]](https://arxiv.org/pdf/2502.10978)</summary>

**Abstract:** Effective decision-making in complex systems requires synthesizing diverse perspectives to address multifaceted challenges under uncertainty. This study introduces an agentic Large Language Models (LLMs) framework for simulating decision discourse - the deliberative process through which actionable strategies are collaboratively developed. Unlike traditional decision-support tools, this framework simulates diverse stakeholder personas, each bringing unique priorities, expertise and value-driven reasoning to a dialogue that emphasizes trade-off exploration in a self-governed assembly. We present explorative results fostering robust and equitable recommendations, with two use cases: first, our framework simulates a response to the floods that occurred on July 2025 in Texas; second, a hypothetical extreme flooding in a Midwestern township under varying forecasting uncertainty. Recommendations made balance competing priorities considered through social, economic and environmental dimensions, setting a foundation for scalable and context-aware recommendations and transforming how decisions for real-world high-stake scenarios can be approached in digital environments. This research explores novel and alternate routes leveraging agentic LLMs for adaptive, collaborative, and equitable recommendations, with implications across domains where uncertainty and complexity converge.

**arXiv ID:** 2502.10978
</details>

<details>
<summary><strong>Balancing the Reasoning Load: Difficulty-Differentiated Policy Optimization with Length Redistribution for Efficient and Robust Reinforcement Learning</strong> - Yinan Xia, Haotian Zhang, Huiming Wang - [[pdf]](https://arxiv.org/pdf/2603.18533)</summary>

**Abstract:** Large Reasoning Models (LRMs) have shown exceptional reasoning capabilities, but they also suffer from the issue of overthinking, often generating excessively long and redundant answers.
For problems that exceed the model's capabilities, LRMs tend to exhibit the overconfidence phenomenon, generating overly short but incorrect answers, which may contribute to suboptimal performance.
To address these issues, we propose Difficulty-Differentiated Policy Optimization (DDPO), an efficient reinforcement learning algorithm that optimizes simple and complex tasks separately based on the overconfidence phenomenon.
Specifically, it reduces the output length for simple tasks without compromising accuracy, while for complex tasks, it expands the exploration space to improve performance. We further derive the theoretical conditions for maximizing expected accuracy, which require the length distribution to closely approximate the optimal length and be as concentrated as possible. Based on these conditions, we propose using the difficulty-level average as a well-founded reference for length optimization.
Extensive experiments on both in-domain and out-of-domain benchmarks validate the superiority and effectiveness of DDPO. Compared to GRPO, DDPO reduces the average answer length by 12% while improving accuracy by 1.85% across multiple benchmarks, achieving a better trade-off between accuracy and length. The code is available at this https URL.

**arXiv ID:** 2603.18533
</details>

<details>
<summary><strong>Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning</strong> - Zhaowei Liu, Xin Guo, Zhi Yang, Fangqi Lou, Lingfeng Zeng, Jinyi Niu, Mengping Li, Qi Qi, Zhiqiang Liu, Yiyang Han, Dongpo Cheng, Ronghao Chen, Huacan Wang, Xingdong Feng, Huixia Judy Wang, Chengchun Shi, Liwen Zhang - [[pdf]](https://arxiv.org/pdf/2503.16252)</summary>

**Abstract:** In recent years, general-purpose large language models (LLMs) such as GPT, Gemini, Claude, and DeepSeek have advanced at an unprecedented pace. Despite these achievements, their application to finance remains challenging, due to fragmented data sources, intransparent reasoning processes, and weak transferability to business applications. In response, we introduce Fin-R1, a reasoning LLM designed for financial scenarios. With a compact size of 7 billion parameters, Fin-R1 reduces deployment costs while addressing the aforementioned challenges. Its development follows a two-stage pipeline. First, we construct Fin-R1-Data, a high-quality financial dataset consisting of 60,091 chain-of-thought (CoT) samples, distilled and filtered from multiple authoritative benchmarks to ensure consistency and reliability. Second, we train Fin-R1 using Fin-R1-Data through supervised fine-tuning (SFT), followed by reinforcement learning (RL). This stage substantially improves the model's ability to solve complex financial reasoning tasks, yielding outputs that are both accurate and interpretable. Despite its relatively small parameter scale, Fin-R1 achieves competitive empirical performance across established financial benchmarks and demonstrates practical utility in compliance checking and robo-advisory. Our code is publicly available at this https URL, and has already attracted over 700 stars.

**arXiv ID:** 2503.16252
</details>

<details>
<summary><strong>DSPO: Stable and Efficient Policy Optimization for Agentic Search and Reasoning</strong> - Chenyang Gu, Yewen Pu, Bruce Yang, Xiaofan Li, Huan Gao - [[pdf]](https://arxiv.org/pdf/2510.09255)</summary>

**Abstract:** Enhancing LLMs with the ability to actively search external knowledge is crucial for complex and real-world tasks. Current approaches either rely on prompting to elicit the model's innate agent capabilities, or suffer from performance ceilings and collapse when applying RL to complex interactive tasks, leaving their true agentic potential untapped. To address this, we introduce \textbf{D}ynamic-filter \textbf{S}equence-level \textbf{P}olicy \textbf{O}ptimization (DSPO), an improved RL algorithm designed for robust agent training through sequence-level optimization and dynamic sample filtering. We train our model purely through RL to interleave multi-turn search and reasoning, obviating the need for supervised demonstration data. Across multiple QA benchmarks, our 7B model improves over a comparable previous work by \textbf{34.1\%}, and even outperforms the 14B model from previous work in complex multihop QA such as HotpotQA by nearly \textbf{9\% relative}, maintaining exceptional training stability.

**arXiv ID:** 2510.09255
</details>

<details>
<summary><strong>Forest-Chat: Adapting Vision-Language Agents for Interactive Forest Change Analysis</strong> - James Brock, Ce Zhang, Nantheera Anantrasirichai - [[pdf]](https://arxiv.org/pdf/2601.14637)</summary>

**Abstract:** The increasing availability of high-resolution satellite imagery, together with advances in deep learning, creates new opportunities for forest monitoring workflows. Two central challenges in this domain are pixel-level change detection and semantic change interpretation, particularly for complex forest dynamics. While large language models (LLMs) are increasingly adopted for data exploration, their integration with vision-language models (VLMs) for remote sensing image change interpretation (RSICI) remains underexplored, especially beyond urban environments. This paper introduces Forest-Chat, an LLM-driven agent for forest change analysis, enabling natural language querying across multiple RSICI tasks, including change detection and captioning, object counting, deforestation characterisation, and change reasoning. Forest-Chat builds upon a multi-level change interpretation (MCI) vision-language backbone with LLM-based orchestration, incorporating zero-shot change detection via AnyChange and multimodal LLM-based zero-shot change captioning and refinement. To support adaptation and evaluation in forest environments, we introduce the Forest-Change dataset, comprising bi-temporal satellite imagery, pixel-level change masks, and semantic change captions via human annotation and rule-based methods. Forest-Chat achieves mIoU and BLEU-4 scores of 67.10% and 40.17% on Forest-Change, and 88.13% and 34.41% on LEVIR-MCI-Trees, a tree-focused subset of LEVIR-MCI. In a zero-shot capacity, it achieves 60.15% and 34.00% on Forest-Change, and 47.32% and 18.23% on LEVIR-MCI-Trees. Further experiments demonstrate the value of caption refinement for injecting geographic domain knowledge into supervised captions, and the system's limited label domain transfer onto JL1-CD-Trees. These findings demonstrate that interactive, LLM-driven systems can support accessible and interpretable forest change analysis.

**arXiv ID:** 2601.14637
</details>

<details>
<summary><strong>AcceRL: A Distributed Asynchronous Reinforcement Learning and World Model Framework for Vision-Language-Action Models</strong> - Chengxuan Lu, Shukuan Wang, Yanjie Li, Wei Liu, Shiji Jin, Fuyuan Qian, Peiming Li, Baigui Sun, Yang Liu - [[pdf]](https://arxiv.org/pdf/2603.18464)</summary>

**Abstract:** Reinforcement learning (RL) for large-scale Vision-Language-Action (VLA) models faces significant challenges in computational efficiency and data acquisition. We propose AcceRL, a fully asynchronous and decoupled RL framework designed to eliminate synchronization barriers by physically isolating training, inference, and rollouts. Crucially, AcceRL is the first to integrate a plug-and-play, trainable world model into a distributed asynchronous RL pipeline to generate virtual experiences. Experiments on the LIBERO benchmark demonstrate that AcceRL achieves state-of-the-art (SOTA) performance. Systematically, it exhibits super-linear scaling in throughput and highly efficient hardware utilization. Algorithmically, the world-model-augmented variant delivers unprecedented sample efficiency and robust training stability in complex control tasks.

**arXiv ID:** 2603.18464
</details>

<details>
<summary><strong>Context Bootstrapped Reinforcement Learning</strong> - Saaket Agashe, Jayanth Srinivasa, Gaowen Liu, Ramana Kompella, Xin Eric Wang - [[pdf]](https://arxiv.org/pdf/2603.18953)</summary>

**Abstract:** Reinforcement Learning from Verifiable Rewards (RLVR) suffers from exploration inefficiency, where models struggle to generate successful rollouts, resulting in minimal learning signal. This challenge is particularly severe for tasks that require the acquisition of novel reasoning patterns or domain-specific knowledge. To address this, we propose Context Bootstrapped Reinforcement Learning (CBRL), which augments RLVR training by stochastically prepending few-shot demonstrations to training prompts. The injection probability follows a curriculum that starts high to bootstrap early exploration, then anneals to zero so the model must ultimately succeed without assistance. This forces the policy to internalize reasoning patterns from the demonstrations rather than relying on them at test time. We validate CBRL across two model families and five Reasoning Gym tasks. Our results demonstrate that CBRL consistently improves success rate, provides better exploration efficiency, and is algorithm-agnostic. We further demonstrate CBRL's practical applicability on Q, a domain-specific programming language that diverges significantly from mainstream language conventions.

**arXiv ID:** 2603.18953
</details>

<details>
<summary><strong>Stable Deep Reinforcement Learning via Isotropic Gaussian Representations</strong> - Ali Saheb Pasand, Johan Obando-Ceron, Aaron Courville, Pouya Bashivan, Pablo Samuel Castro - [[pdf]](https://arxiv.org/pdf/2602.19373)</summary>

**Abstract:** Deep reinforcement learning systems often suffer from unstable training dynamics due to non-stationarity, where learning objectives and data distributions evolve over time. We show that under non-stationary targets, isotropic Gaussian embeddings are provably advantageous. In particular, they induce stable tracking of time-varying targets for linear readouts, achieve maximal entropy under a fixed variance budget, and encourage a balanced use of all representational dimensions--all of which enable agents to be more adaptive and stable. Building on this insight, we propose the use of Sketched Isotropic Gaussian Regularization for shaping representations toward an isotropic Gaussian distribution during training. We demonstrate empirically, over a variety of domains, that this simple and computationally inexpensive method improves performance under non-stationarity while reducing representation collapse, neuron dormancy, and training instability.

**arXiv ID:** 2602.19373
</details>

<details>
<summary><strong>Efficient and Versatile Quadrupedal Skating: Optimal Co-design via Reinforcement Learning and Bayesian Optimization</strong> - Hanwen Wang, Zhenlong Fang, Josiah Hanna, Xiaobin Xiong - [[pdf]](https://arxiv.org/pdf/2603.18408)</summary>

**Abstract:** In this paper, we present a hardware-control co-design approach that enables efficient and versatile roller skating on quadrupedal robots equipped with passive wheels. Passive-wheel skating reduces leg inertia and improves energy efficiency, particularly at high speeds. However, the absence of direct wheel actuation tightly couples mechanical design and control. To unlock the full potential of this modality, we formulate a bilevel optimization framework: an upper-level Bayesian Optimization searches the mechanical design space, while a lower-level Reinforcement Learning trains a motor control policy for each candidate design. The resulting design-policy pairs not only outperform human-engineered baselines, but also exhibit versatile behaviors such as hockey stop (rapid braking by turning sideways to maximize friction) and self-aligning motion (automatic reorientation to improve energy efficiency in the direction of travel), offering the first system-level study of dynamic skating motion on quadrupedal robots.

**arXiv ID:** 2603.18408
</details>

<details>
<summary><strong>TwinRL-VLA: Digital Twin-Driven Reinforcement Learning for Real-World Robotic Manipulation</strong> - Qinwen Xu, Jiaming Liu, Rui Zhou, Shaojun Shi, Nuowei Han, Zhuoyang Liu, Chenyang Gu, Shuo Gu, Yang Yue, Gao Huang, Wenzhao Zheng, Sirui Han, Peng Jia, Shanghang Zhang - [[pdf]](https://arxiv.org/pdf/2602.09023)</summary>

**Abstract:** Despite strong generalization capabilities, Vision-Language-Action (VLA) models remain constrained by the high cost of expert demonstrations and insufficient real-world interaction. While online reinforcement learning (RL) has shown promise in improving general foundation models, applying RL to VLA manipulation in real-world settings is still hindered by low exploration efficiency and a restricted exploration space. Through systematic real-world experiments, we observe that the effective exploration space of online RL is closely tied to the data distribution of supervised fine-tuning (SFT). Motivated by this observation, we propose TwinRL, a digital twin-real-world collaborative RL framework designed to scale and guide exploration for VLA models. First, a high-fidelity digital twin is efficiently reconstructed from smartphone-captured scenes, enabling realistic bidirectional transfer between real and simulated environments. During the SFT warm-up stage, we introduce an exploration space expansion strategy using digital twins to broaden the support of the data trajectory distribution. Building on this enhanced initialization, we propose a sim-to-real guided exploration strategy to further accelerate online RL. Specifically, TwinRL performs efficient and parallel online RL in the digital twin prior to deployment, effectively bridging the gap between offline and online training stages. Subsequently, we exploit efficient digital twin sampling to identify failure-prone yet informative configurations, which are used to guide targeted human-in-the-loop rollouts on the real robot. In our experiments, TwinRL approaches 100% success in both in-distribution regions covered by real-world demonstrations and out-of-distribution regions, delivering at least a 30% speedup over prior real-world RL methods and requiring only about 20 minutes on average across four tasks.

**arXiv ID:** 2602.09023
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
