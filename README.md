# Agent arXiv Daily

**Last Updated:** 2025-11-12 02:09:27

**Total Papers:** 90

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
<summary><strong>Dataforge: A Data Agent Platform for Autonomous Data Engineering</strong> - Xinyuan Wang, Yanjie Fu - [[pdf]](https://arxiv.org/pdf/2511.06185)</summary>

**Abstract:** The growing demand for AI applications in fields such as materials discovery, molecular modeling, and climate science has made data preparation an important but labor-intensive step. Raw data from diverse sources must be cleaned, normalized, and transformed to become AI-ready, while effective feature transformation and selection are essential for efficient training and inference. To address the challenges of scalability and expertise dependence, we present Data Agent, a fully autonomous system specialized for tabular data. Leveraging large language model (LLM) reasoning and grounded validation, Data Agent automatically performs data cleaning, hierarchical routing, and feature-level optimization through dual feedback loops. It embodies three core principles: automatic, safe, and non-expert friendly, which ensure end-to-end reliability without human supervision. This demo showcases the first practical realization of an autonomous Data Agent, illustrating how raw data can be transformed "From Data to Better Data."

**arXiv ID:** 2511.06185
</details>

<details>
<summary><strong>When AI Meets the Web: Prompt Injection Risks in Third-Party AI Chatbot Plugins</strong> - Yigitcan Kaya, Anton Landerer, Stijn Pletinckx, Michelle Zimmermann, Christopher Kruegel, Giovanni Vigna - [[pdf]](https://arxiv.org/pdf/2511.05797)</summary>

**Abstract:** Prompt injection attacks pose a critical threat to large language models (LLMs), with prior work focusing on cutting-edge LLM applications like personal copilots. In contrast, simpler LLM applications, such as customer service chatbots, are widespread on the web, yet their security posture and exposure to such attacks remain poorly understood. These applications often rely on third-party chatbot plugins that act as intermediaries to commercial LLM APIs, offering non-expert website builders intuitive ways to customize chatbot behaviors. To bridge this gap, we present the first large-scale study of 17 third-party chatbot plugins used by over 10,000 public websites, uncovering previously unknown prompt injection risks in practice. First, 8 of these plugins (used by 8,000 websites) fail to enforce the integrity of the conversation history transmitted in network requests between the website visitor and the chatbot. This oversight amplifies the impact of direct prompt injection attacks by allowing adversaries to forge conversation histories (including fake system messages), boosting their ability to elicit unintended behavior (e.g., code generation) by 3 to 8x. Second, 15 plugins offer tools, such as web-scraping, to enrich the chatbot's context with website-specific content. However, these tools do not distinguish the website's trusted content (e.g., product descriptions) from untrusted, third-party content (e.g., customer reviews), introducing a risk of indirect prompt injection. Notably, we found that ~13% of e-commerce websites have already exposed their chatbots to third-party content. We systematically evaluate both vulnerabilities through controlled experiments grounded in real-world observations, focusing on factors such as system prompt design and the underlying LLM. Our findings show that many plugins adopt insecure practices that undermine the built-in LLM safeguards.

**arXiv ID:** 2511.05797
</details>

<details>
<summary><strong>ConvFill: Model Collaboration for Responsive Conversational Voice Agents</strong> - Vidya Srinivas, Zachary Englhardt, Maximus Powers, Shwetak Patel, Vikram Iyer - [[pdf]](https://arxiv.org/pdf/2511.07397)</summary>

**Abstract:** Deploying conversational voice agents with large language models faces a critical challenge: cloud-based foundation models provide deep reasoning and domain knowledge but introduce latency that disrupts natural conversation, while on-device models respond immediately but lack sophistication. We propose conversational infill, a task where a lightweight on-device model generates contextually appropriate dialogue while seamlessly incorporating streaming knowledge from a powerful backend model. This approach decouples response latency from model capability, enabling systems that feel responsive while accessing the full power of large-scale models. We present ConvFill, a 360M parameter model trained on synthetic multi-domain conversations. Evaluation across multiple backend models shows that conversational infill can be successfully learned, with ConvFill achieving accuracy improvements of 36-42% over standalone small models of the same size while consistently retaining sub-200ms response latencies. Our results demonstrate the promise of this approach for building on-device conversational agents that are both immediately responsive and knowledgeable.

**arXiv ID:** 2511.07397
</details>

<details>
<summary><strong>Personalizing Emotion-aware Conversational Agents? Exploring User Traits-driven Conversational Strategies for Enhanced Interaction</strong> - Yuchong Zhang, Yong Ma, Di Fu, Stephanie Zubicueta Portales, Morten Fjeld, Danica Kragic - [[pdf]](https://arxiv.org/pdf/2511.06954)</summary>

**Abstract:** Conversational agents (CAs) are increasingly embedded in daily life, yet their ability to navigate user emotions efficiently is still evolving. This study investigates how users with varying traits -- gender, personality, and cultural background -- adapt their interaction strategies with emotion-aware CAs in specific emotional scenarios. Using an emotion-aware CA prototype expressing five distinct emotions (neutral, happy, sad, angry, and fear) through male and female voices, we examine how interaction dynamics shift across different voices and emotional contexts through empirical studies. Our findings reveal distinct variations in user engagement and conversational strategies based on individual traits, emphasizing the value of personalized, emotion-sensitive interactions. By analyzing both qualitative and quantitative data, we demonstrate that tailoring CAs to user characteristics can enhance user satisfaction and interaction quality. This work underscores the critical need for ongoing research to design CAs that not only recognize but also adaptively respond to emotional needs, ultimately supporting a diverse user groups more effectively.

**arXiv ID:** 2511.06954
</details>

<details>
<summary><strong>People Perceive More Phantom Costs From Autonomous Agents When They Make Unreasonably Generous Offers</strong> - Benjamin Lebrun, Christoph Bartneck, David Kaber, Andrew Vonasch - [[pdf]](https://arxiv.org/pdf/2511.07401)</summary>

**Abstract:** People often reject offers that are too generous due to the perception of hidden drawbacks referred to as "phantom costs." We hypothesized that this perception and the decision-making vary based on the type of agent making the offer (human vs. robot) and the degree to which the agent is perceived to be autonomous or have the capacity for self-interest. To test this conjecture, participants (N = 855) engaged in a car-buying simulation where a human or robot sales agent, described as either autonomous or not, offered either a small (5%) or large (85%) discount. Results revealed that the robot was perceived as less self-interested than the human, which reduced the perception of phantom costs. While larger discounts increased phantom costs, they also increased purchase intentions, suggesting that perceived benefits can outweigh phantom costs. Importantly, phantom costs were not only attributed to the agent participants interacted with, but also to the product and the agent's manager, highlighting at least three sources of suspicion. These findings deepen our understanding of to whom people assign responsibility and how perceptions shape both human-human and human-robot interactions, with implications for ethical AI design and marketing strategies.

**arXiv ID:** 2511.07401
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>Evidence-Bound Autonomous Research (EviBound): A Governance Framework for Eliminating False Claims</strong> - Ruiying Chen - [[pdf]](https://arxiv.org/pdf/2511.05524)</summary>

**Abstract:** LLM-based autonomous research agents report false claims: tasks marked "complete" despite missing artifacts, contradictory metrics, or failed executions. EviBound is an evidence-bound execution framework that eliminates false claims through dual governance gates requiring machine-checkable evidence.
Two complementary gates enforce evidence requirements. The pre-execution Approval Gate validates acceptance criteria schemas before code runs, catching structural violations proactively. The post-execution Verification Gate validates artifacts via MLflow API queries (with recursive path checking) and optionally validates metrics when specified by acceptance criteria. Claims propagate only when backed by a queryable run ID, required artifacts, and FINISHED status. Bounded, confidence-gated retries (typically 1-2 attempts) recover from transient failures without unbounded loops.
The framework was evaluated on 8 benchmark tasks spanning infrastructure validation, ML capabilities, and governance stress tests. Baseline A (Prompt-Level Only) yields 100% hallucination (8/8 claimed, 0/8 verified). Baseline B (Verification-Only) reduces hallucination to 25% (2/8 fail verification). EviBound (Dual Gates) achieves 0% hallucination: 7/8 tasks verified and 1 task correctly blocked at the approval gate, all with only approximately 8.3% execution overhead.
This package includes execution trajectories, MLflow run IDs for all verified tasks, and a 4-step verification protocol. Research integrity is an architectural property, achieved through governance gates rather than emergent from model scale.

**arXiv ID:** 2511.05524
</details>

<details>
<summary><strong>ROAR: Robust Accident Recognition and Anticipation for Autonomous Driving</strong> - Xingcheng Liu, Yanchen Guan, Haicheng Liao, Zhengbing He, Zhenning Li - [[pdf]](https://arxiv.org/pdf/2511.06226)</summary>

**Abstract:** Accurate accident anticipation is essential for enhancing the safety of autonomous vehicles (AVs). However, existing methods often assume ideal conditions, overlooking challenges such as sensor failures, environmental disturbances, and data imperfections, which can significantly degrade prediction accuracy. Additionally, previous models have not adequately addressed the considerable variability in driver behavior and accident rates across different vehicle types. To overcome these limitations, this study introduces ROAR, a novel approach for accident detection and prediction. ROAR combines Discrete Wavelet Transform (DWT), a self adaptive object aware module, and dynamic focal loss to tackle these challenges. The DWT effectively extracts features from noisy and incomplete data, while the object aware module improves accident prediction by focusing on high-risk vehicles and modeling the spatial temporal relationships among traffic agents. Moreover, dynamic focal loss mitigates the impact of class imbalance between positive and negative samples. Evaluated on three widely used datasets, Dashcam Accident Dataset (DAD), Car Crash Dataset (CCD), and AnAn Accident Detection (A3D), our model consistently outperforms existing baselines in key metrics such as Average Precision (AP) and mean Time to Accident (mTTA). These results demonstrate the model's robustness in real-world conditions, particularly in handling sensor degradation, environmental noise, and imbalanced data distributions. This work offers a promising solution for reliable and accurate accident anticipation in complex traffic environments.

**arXiv ID:** 2511.06226
</details>

<details>
<summary><strong>AUTO-Explorer: Automated Data Collection for GUI Agent</strong> - Xiangwu Guo, Difei Gao, Mike Zheng Shou - [[pdf]](https://arxiv.org/pdf/2511.06417)</summary>

**Abstract:** Recent advancements in GUI agents have significantly expanded their ability to interpret natural language commands to manage software interfaces. However, acquiring GUI data remains a significant challenge. Existing methods often involve designing automated agents that browse URLs from the Common Crawl, using webpage HTML to collect screenshots and corresponding annotations, including the names and bounding boxes of UI elements. However, this method is difficult to apply to desktop software or some newly launched websites not included in the Common Crawl. While we expect the model to possess strong generalization capabilities to handle this, it is still crucial for personalized scenarios that require rapid and perfect adaptation to new software or websites. To address this, we propose an automated data collection method with minimal annotation costs, named Auto-Explorer. It incorporates a simple yet effective exploration mechanism that autonomously parses and explores GUI environments, gathering data efficiently. Additionally, to assess the quality of exploration, we have developed the UIXplore benchmark. This benchmark creates environments for explorer agents to discover and save software states. Using the data gathered, we fine-tune a multimodal large language model (MLLM) and establish a GUI element grounding testing set to evaluate the effectiveness of the exploration strategies. Our experiments demonstrate the superior performance of Auto-Explorer, showing that our method can quickly enhance the capabilities of an MLLM in explored software.

**arXiv ID:** 2511.06417
</details>

<details>
<summary><strong>DigiData: Training and Evaluating General-Purpose Mobile Control Agents</strong> - Yuxuan Sun, Manchen Wang, Shengyi Qian, William R. Wong, Eric Gan, Pierluca D'Oro, Alejandro Castillejo Munoz, Sneha Silwal, Pedro Matias, Nitin Kamra, Satwik Kottur, Nick Raines, Xuanyi Zhao, Joy Chen, Joseph Greer, Andrea Madotto, Allen Bolourchi, James Valori, Kevin Carlberg, Karl Ridgeway, Joseph Tighe - [[pdf]](https://arxiv.org/pdf/2511.07413)</summary>

**Abstract:** AI agents capable of controlling user interfaces have the potential to transform human interaction with digital devices. To accelerate this transformation, two fundamental building blocks are essential: high-quality datasets that enable agents to achieve complex and human-relevant goals, and robust evaluation methods that allow researchers and practitioners to rapidly enhance agent performance. In this paper, we introduce DigiData, a large-scale, high-quality, diverse, multi-modal dataset designed for training mobile control agents. Unlike existing datasets, which derive goals from unstructured interactions, DigiData is meticulously constructed through comprehensive exploration of app features, resulting in greater diversity and higher goal complexity. Additionally, we present DigiData-Bench, a benchmark for evaluating mobile control agents on real-world complex tasks. We demonstrate that the commonly used step-accuracy metric falls short in reliably assessing mobile control agents and, to address this, we propose dynamic evaluation protocols and AI-powered evaluations as rigorous alternatives for agent assessment. Our contributions aim to significantly advance the development of mobile control agents, paving the way for more intuitive and effective human-device interactions.

**arXiv ID:** 2511.07413
</details>

<details>
<summary><strong>IMDMR: An Intelligent Multi-Dimensional Memory Retrieval System for Enhanced Conversational AI</strong> - Tejas Pawar, Sarika Patil, Om Tilekar, Rushikesh Janwade, Vaibhav Helambe - [[pdf]](https://arxiv.org/pdf/2511.05495)</summary>

**Abstract:** Conversational AI systems often struggle with maintaining coherent, contextual memory across extended interactions, limiting their ability to provide personalized and contextually relevant responses. This paper presents IMDMR (Intelligent Multi-Dimensional Memory Retrieval), a novel system that addresses these limitations through a multi-dimensional search architecture. Unlike existing memory systems that rely on single-dimensional approaches, IMDMR leverages six distinct memory dimensions-semantic, entity, category, intent, context, and temporal-to provide comprehensive memory retrieval capabilities. Our system incorporates intelligent query processing with dynamic strategy selection, cross-memory entity resolution, and advanced memory integration techniques. Through comprehensive evaluation against five baseline systems including LangChain RAG, LlamaIndex, MemGPT, and spaCy + RAG, IMDMR achieves a 3.8x improvement in overall performance (0.792 vs 0.207 for the best baseline). We present both simulated (0.314) and production (0.792) implementations, demonstrating the importance of real technology integration while maintaining superiority over all baseline systems. Ablation studies demonstrate the effectiveness of multi-dimensional search, with the full system outperforming individual dimension approaches by 23.3%. Query-type analysis reveals superior performance across all categories, particularly for preferences/interests (0.630) and goals/aspirations (0.630) queries. Comprehensive visualizations and statistical analysis confirm the significance of these improvements with p < 0.001 across all metrics. The results establish IMDMR as a significant advancement in conversational AI memory systems, providing a robust foundation for enhanced user interactions and personalized experiences.

**arXiv ID:** 2511.05495
</details>

<details>
<summary><strong>WebVIA: A Web-based Vision-Language Agentic Framework for Interactive and Verifiable UI-to-Code Generation</strong> - Mingde Xu, Zhen Yang, Wenyi Hong, Lihang Pan, Xinyue Fan, Yan Wang, Xiaotao Gu, Bin Xu, Jie Tang - [[pdf]](https://arxiv.org/pdf/2511.06251)</summary>

**Abstract:** User interface (UI) development requires translating design mockups into functional code, a process that remains repetitive and labor-intensive. While recent Vision-Language Models (VLMs) automate UI-to-Code generation, they generate only static HTML/CSS/JavaScript layouts lacking interactivity. To address this, we propose WebVIA, the first agentic framework for interactive UI-to-Code generation and validation. The framework comprises three components: 1) an exploration agent to capture multi-state UI screenshots; 2) a UI2Code model that generates executable interactive code; 3) a validation module that verifies the interactivity. Experiments demonstrate that WebVIA-Agent achieves more stable and accurate UI exploration than general-purpose agents (e.g., Gemini-2.5-Pro). In addition, our fine-tuned WebVIA-UI2Code models exhibit substantial improvements in generating executable and interactive HTML/CSS/JavaScript code, outperforming their base counterparts across both interactive and static UI2Code benchmarks. Our code and models are available at \href{this https URL}{\texttt{this https URL}}.

**arXiv ID:** 2511.06251
</details>

<details>
<summary><strong>Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper</strong> - Atsuyuki Miyai, Mashiro Toyooka, Takashi Otonari, Zaiying Zhao, Kiyoharu Aizawa - [[pdf]](https://arxiv.org/pdf/2511.04583)</summary>

**Abstract:** Understanding the current capabilities and risks of AI Scientist systems is essential for ensuring trustworthy and sustainable AI-driven scientific progress while preserving the integrity of the academic ecosystem. To this end, we develop Jr. AI Scientist, a state-of-the-art autonomous AI scientist system that mimics the core research workflow of a novice student researcher: Given the baseline paper from the human mentor, it analyzes its limitations, formulates novel hypotheses for improvement, and iteratively conducts experiments until improvements are realized, and writes a paper with the results. Unlike previous approaches that assume full automation or operate on small-scale code, Jr. AI Scientist follows a well-defined research workflow and leverages modern coding agents to handle complex, multi-file implementations, leading to scientifically valuable contributions. Through our experiments, the Jr. AI Scientist successfully generated new research papers that build upon real NeurIPS, IJCV, and ICLR works by proposing and implementing novel methods. For evaluation, we conducted automated assessments using AI Reviewers, author-led evaluations, and submissions to Agents4Science, a venue dedicated to AI-driven scientific contributions. The findings demonstrate that Jr. AI Scientist generates papers receiving higher review scores than existing fully automated systems. Nevertheless, we identify important limitations from both the author evaluation and the Agents4Science reviews, indicating the potential risks of directly applying current AI Scientist systems and key challenges for future research. Finally, we comprehensively report various risks identified during development. We believe this study clarifies the current role and limitations of AI Scientist systems, offering insights into the areas that still require human expertise and the risks that may emerge as these systems evolve.

**arXiv ID:** 2511.04583
</details>

<details>
<summary><strong>Catching Contamination Before Generation: Spectral Kill Switches for Agents</strong> - Valentin Noël - [[pdf]](https://arxiv.org/pdf/2511.05804)</summary>

**Abstract:** Agentic language models compose multi step reasoning chains, yet intermediate steps can be corrupted by inconsistent context, retrieval errors, or adversarial inputs, which makes post hoc evaluation too late because errors propagate before detection. We introduce a diagnostic that requires no additional training and uses only the forward pass to emit a binary accept or reject signal during agent execution. The method analyzes token graphs induced by attention and computes two spectral statistics in early layers, namely the high frequency energy ratio and spectral entropy. We formalize these signals, establish invariances, and provide finite sample estimators with uncertainty quantification. Under a two regime mixture assumption with a monotone likelihood ratio property, we show that a single threshold on the high frequency energy ratio is optimal in the Bayes sense for detecting context inconsistency. Empirically, the high frequency energy ratio exhibits robust bimodality during context verification across multiple model families, which enables gating decisions with overhead below one millisecond on our hardware and configurations. We demonstrate integration into retrieval augmented agent pipelines and discuss deployment as an inline safety monitor. The approach detects contamination while the model is still processing the text, before errors commit to the reasoning chain.

**arXiv ID:** 2511.05804
</details>

<details>
<summary><strong>Scalable Offline Metrics for Autonomous Driving</strong> - Animikh Aich, Adwait Kulkarni, Eshed Ohn-Bar - [[pdf]](https://arxiv.org/pdf/2510.08571)</summary>

**Abstract:** Real-world evaluation of perception-based planning models for robotic systems, such as autonomous vehicles, can be safely and inexpensively conducted offline, i.e. by computing model prediction error over a pre-collected validation dataset with ground-truth annotations. However, extrapolating from offline model performance to online settings remains a challenge. In these settings, seemingly minor errors can compound and result in test-time infractions or collisions. This relationship is understudied, particularly across diverse closed-loop metrics and complex urban maneuvers. In this work, we revisit this undervalued question in policy evaluation through an extensive set of experiments across diverse conditions and metrics. Based on analysis in simulation, we find an even worse correlation between offline and online settings than reported by prior studies, casting doubts on the validity of current evaluation practices and metrics for driving policies. Next, we bridge the gap between offline and online evaluation. We investigate an offline metric based on epistemic uncertainty, which aims to capture events that are likely to cause errors in closed-loop settings. The resulting metric achieves over 13% improvement in correlation compared to previous offline metrics. We further validate the generalization of our findings beyond the simulation environment in real-world settings, where even greater gains are observed.

**arXiv ID:** 2510.08571
</details>

<details>
<summary><strong>AgentSUMO: An Agentic Framework for Interactive Simulation Scenario Generation in SUMO via Large Language Models</strong> - Minwoo Jeong, Jeeyun Chang, Yoonjin Yoon - [[pdf]](https://arxiv.org/pdf/2511.06804)</summary>

**Abstract:** The growing complexity of urban mobility systems has made traffic simulation indispensable for evidence-based transportation planning and policy evaluation. However, despite the analytical capabilities of platforms such as the Simulation of Urban MObility (SUMO), their application remains largely confined to domain experts. Developing realistic simulation scenarios requires expertise in network construction, origin-destination modeling, and parameter configuration for policy experimentation, creating substantial barriers for non-expert users such as policymakers, urban planners, and city officials. Moreover, the requests expressed by these users are often incomplete and abstract-typically articulated as high-level objectives, which are not well aligned with the imperative, sequential workflows employed in existing language-model-based simulation frameworks. To address these challenges, this study proposes AgentSUMO, an agentic framework for interactive simulation scenario generation via large language models. AgentSUMO departs from imperative, command-driven execution by introducing an adaptive reasoning layer that interprets user intents, assesses task complexity, infers missing parameters, and formulates executable simulation plans. The framework is structured around two complementary components, the Interactive Planning Protocol, which governs reasoning and user interaction, and the Model Context Protocol, which manages standardized communication and orchestration among simulation tools. Through this design, AgentSUMO converts abstract policy objectives into executable simulation scenarios. Experiments on urban networks in Seoul and Manhattan demonstrate that the agentic workflow achieves substantial improvements in traffic flow metrics while maintaining accessibility for non-expert users, successfully bridging the gap between policy goals and executable simulation workflows.

**arXiv ID:** 2511.06804
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>LLM-Guided Reinforcement Learning with Representative Agents for Traffic Modeling</strong> - Hanlin Sun, Jiayang Li - [[pdf]](https://arxiv.org/pdf/2511.06260)</summary>

**Abstract:** Large language models (LLMs) are increasingly used as behavioral proxies for self-interested travelers in agent-based traffic models. Although more flexible and generalizable than conventional models, the practical use of these approaches remains limited by scalability due to the cost of calling one LLM for every traveler. Moreover, it has been found that LLM agents often make opaque choices and produce unstable day-to-day dynamics. To address these challenges, we propose to model each homogeneous traveler group facing the same decision context with a single representative LLM agent who behaves like the population's average, maintaining and updating a mixed strategy over routes that coincides with the group's aggregate flow proportions. Each day, the LLM reviews the travel experience and flags routes with positive reinforcement that they hope to use more often, and an interpretable update rule then converts this judgment into strategy adjustments using a tunable (progressively decaying) step size. The representative-agent design improves scalability, while the separation of reasoning from updating clarifies the decision logic while stabilizing learning. In classic traffic assignment settings, we find that the proposed approach converges rapidly to the user equilibrium. In richer settings with income heterogeneity, multi-criteria costs, and multi-modal choices, the generated dynamics remain stable and interpretable, reproducing plausible behavioral patterns well-documented in psychology and economics, for example, the decoy effect in toll versus non-toll road selection, and higher willingness-to-pay for convenience among higher-income travelers when choosing between driving, transit, and park-and-ride options.

**arXiv ID:** 2511.06260
</details>

<details>
<summary><strong>FLEX: Continuous Agent Evolution via Forward Learning from Experience</strong> - Zhicheng Cai, Xinyuan Guo, Yu Pei, JiangTao Feng, Jiangjie Chen, Ya-Qin Zhang, Wei-Ying Ma, Mingxuan Wang, Hao Zhou - [[pdf]](https://arxiv.org/pdf/2511.06449)</summary>

**Abstract:** Autonomous agents driven by Large Language Models (LLMs) have revolutionized reasoning and problem-solving but remain static after training, unable to grow with experience as intelligent beings do during deployment. We introduce Forward Learning with EXperience (FLEX), a gradient-free learning paradigm that enables LLM agents to continuously evolve through accumulated experience. Specifically, FLEX cultivates scalable and inheritable evolution by constructing a structured experience library through continual reflection on successes and failures during interaction with the environment. FLEX delivers substantial improvements on mathematical reasoning, chemical retrosynthesis, and protein fitness prediction (up to 23% on AIME25, 10% on USPTO50k, and 14% on ProteinGym). We further identify a clear scaling law of experiential growth and the phenomenon of experience inheritance across agents, marking a step toward scalable and inheritable continuous agent evolution. Project Page: this https URL.

**arXiv ID:** 2511.06449
</details>

<details>
<summary><strong>More Agents Helps but Adversarial Robustness Gap Persists</strong> - Khashayar Alavi, Zhastay Yeltay, Lucie Flek, Akbar Karimi - [[pdf]](https://arxiv.org/pdf/2511.07112)</summary>

**Abstract:** When LLM agents work together, they seem to be more powerful than a single LLM in mathematical question answering. However, are they also more robust to adversarial inputs? We investigate this question using adversarially perturbed math questions. These perturbations include punctuation noise with three intensities (10, 30, and 50 percent), plus real-world and human-like typos (WikiTypo, R2ATA). Using a unified sampling-and-voting framework (Agent Forest), we evaluate six open-source models (Qwen3-4B/14B, Llama3.1-8B, Mistral-7B, Gemma3-4B/12B) across four benchmarks (GSM8K, MATH, MMLU-Math, MultiArith), with various numbers of agents n from one to 25 (1, 2, 5, 10, 15, 20, 25). Our findings show that (1) Noise type matters: punctuation noise harm scales with its severity, and the human typos remain the dominant bottleneck, yielding the largest gaps to Clean accuracy and the highest ASR even with a large number of agents. And (2) Collaboration reliably improves accuracy as the number of agents, n, increases, with the largest gains from one to five agents and diminishing returns beyond 10 agents. However, the adversarial robustness gap persists regardless of the agent count.

**arXiv ID:** 2511.07112
</details>

<details>
<summary><strong>Graph Representation-based Model Poisoning on the Heterogeneous Internet of Agents</strong> - Hanlin Cai, Houtianfu Wang, Haofan Dong, Kai Li, Ozgur B. Akan - [[pdf]](https://arxiv.org/pdf/2511.07176)</summary>

**Abstract:** Internet of Agents (IoA) envisions a unified, agent-centric paradigm where heterogeneous large language model (LLM) agents can interconnect and collaborate at scale. Within this paradigm, federated learning (FL) serves as a key enabler that allows distributed LLM agents to co-train global models without centralizing data. However, the FL-enabled IoA system remains vulnerable to model poisoning attacks, and the prevailing distance and similarity-based defenses become fragile at billion-parameter scale and under heterogeneous data distributions. This paper proposes a graph representation-based model poisoning (GRMP) attack, which passively exploits observed benign local models to construct a parameter correlation graph and extends an adversarial variational graph autoencoder to capture and reshape higher-order dependencies. The GRMP attack synthesizes malicious local models that preserve benign-like statistics while embedding adversarial objectives, remaining elusive to detection at the server. Experiments demonstrate a gradual drop in system accuracy under the proposed attack and the ineffectiveness of the prevailing defense mechanism in detecting the attack, underscoring a severe threat to the ambitious IoA paradigm.

**arXiv ID:** 2511.07176
</details>

<details>
<summary><strong>BLADE: Benchmarking Language Model Agents for Data-Driven Science</strong> - Ken Gu, Ruoxi Shang, Ruien Jiang, Keying Kuang, Richard-John Lin, Donghe Lyu, Yue Mao, Youran Pan, Teng Wu, Jiaqian Yu, Yikun Zhang, Tianmai M. Zhang, Lanyi Zhu, Mike A. Merrill, Jeffrey Heer, Tim Althoff - [[pdf]](https://arxiv.org/pdf/2408.09667)</summary>

**Abstract:** Data-driven scientific discovery requires the iterative integration of scientific domain knowledge, statistical expertise, and an understanding of data semantics to make nuanced analytical decisions, e.g., about which variables, transformations, and statistical models to consider. LM-based agents equipped with planning, memory, and code execution capabilities have the potential to support data-driven science. However, evaluating agents on such open-ended tasks is challenging due to multiple valid approaches, partially correct steps, and different ways to express the same decisions. To address these challenges, we present BLADE, a benchmark to automatically evaluate agents' multifaceted approaches to open-ended research questions. BLADE consists of 12 datasets and research questions drawn from existing scientific literature, with ground truth collected from independent analyses by expert data scientists and researchers. To automatically evaluate agent responses, we developed corresponding computational methods to match different representations of analyses to this ground truth. Though language models possess considerable world knowledge, our evaluation shows that they are often limited to basic analyses. However, agents capable of interacting with the underlying data demonstrate improved, but still non-optimal, diversity in their analytical decision making. Our work enables the evaluation of agents for data-driven science and provides researchers deeper insights into agents' analysis approaches.

**arXiv ID:** 2408.09667
</details>

<details>
<summary><strong>ECom-Bench: Can LLM Agent Resolve Real-World E-commerce Customer Support Issues?</strong> - Haoxin Wang, Xianhan Peng, Xucheng Huang, Yizhe Huang, Ming Gong, Chenghan Yang, Yang Liu, Ling Jiang - [[pdf]](https://arxiv.org/pdf/2507.05639)</summary>

**Abstract:** In this paper, we introduce ECom-Bench, the first benchmark framework for evaluating LLM agent with multimodal capabilities in the e-commerce customer support domain. ECom-Bench features dynamic user simulation based on persona information collected from real e-commerce customer interactions and a realistic task dataset derived from authentic e-commerce dialogues. These tasks, covering a wide range of business scenarios, are designed to reflect real-world complexities, making ECom-Bench highly challenging. For instance, even advanced models like GPT-4o achieve only a 10-20% pass^3 metric in our benchmark, highlighting the substantial difficulties posed by complex e-commerce scenarios. The code and data have been made publicly available at this https URL to facilitate further research and development in this domain.

**arXiv ID:** 2507.05639
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (21 papers)</h2></summary>

<details>
<summary><strong>SMAGDi: Socratic Multi Agent Interaction Graph Distillation for Efficient High Accuracy Reasoning</strong> - Aayush Aluru, Myra Malik, Samarth Patankar, Spencer Kim, Kevin Zhu, Sean O'Brien, Vasu Sharma - [[pdf]](https://arxiv.org/pdf/2511.05528)</summary>

**Abstract:** Multi-agent systems (MAS) often achieve higher reasoning accuracy than single models, but their reliance on repeated debates across agents makes them computationally expensive. We introduce SMAGDi, a distillation framework that transfers the debate dynamics of a five-agent Llama-based MAS into a compact Socratic decomposer-solver student. SMAGDi represents debate traces as directed interaction graphs, where nodes encode intermediate reasoning steps with correctness labels and edges capture continuity and cross-agent influence. The student is trained with a composite objective combining language modeling, graph-based supervision, contrastive reasoning, and embedding alignment to preserve both fluency and structured reasoning. On StrategyQA and MMLU, SMAGDi compresses a 40B multi-agent system into a 6B student while retaining 88% of its accuracy, substantially outperforming prior distillation methods such as MAGDi, standard KD, and fine-tuned baselines. These results highlight that explicitly modeling interaction graphs and Socratic decomposition enable small models to inherit the accuracy benefits of multi-agent debate while remaining efficient enough for real-world deployment.

**arXiv ID:** 2511.05528
</details>

<details>
<summary><strong>Maestro: Learning to Collaborate via Conditional Listwise Policy Optimization for Multi-Agent LLMs</strong> - Wei Yang, Jiacheng Pang, Shixuan Li, Paul Bogdan, Stephen Tu, Jesse Thomason - [[pdf]](https://arxiv.org/pdf/2511.06134)</summary>

**Abstract:** Multi-agent systems (MAS) built on Large Language Models (LLMs) are being used to approach complex problems and can surpass single model inference. However, their success hinges on navigating a fundamental cognitive tension: the need to balance broad, divergent exploration of the solution space with a principled, convergent synthesis to the optimal solution. Existing paradigms often struggle to manage this duality, leading to premature consensus, error propagation, and a critical credit assignment problem that fails to distinguish between genuine reasoning and superficially plausible arguments. To resolve this core challenge, we propose the Multi-Agent Exploration-Synthesis framework Through Role Orchestration (Maestro), a principled paradigm for collaboration that structurally decouples these cognitive modes. Maestro uses a collective of parallel Execution Agents for diverse exploration and a specialized Central Agent for convergent, evaluative synthesis. To operationalize this critical synthesis phase, we introduce Conditional Listwise Policy Optimization (CLPO), a reinforcement learning objective that disentangles signals for strategic decisions and tactical rationales. By combining decision-focused policy gradients with a list-wise ranking loss over justifications, CLPO achieves clean credit assignment and stronger comparative supervision. Experiments on mathematical reasoning and general problem-solving benchmarks demonstrate that Maestro, coupled with CLPO, consistently outperforms existing state-of-the-art multi-agent approaches, delivering absolute accuracy gains of 6% on average and up to 10% at best.

**arXiv ID:** 2511.06134
</details>

<details>
<summary><strong>MALinZero: Efficient Low-Dimensional Search for Mastering Complex Multi-Agent Planning</strong> - Sizhe Tang, Jiayu Chen, Tian Lan - [[pdf]](https://arxiv.org/pdf/2511.06142)</summary>

**Abstract:** Monte Carlo Tree Search (MCTS), which leverages Upper Confidence Bound for Trees (UCTs) to balance exploration and exploitation through randomized sampling, is instrumental to solving complex planning problems. However, for multi-agent planning, MCTS is confronted with a large combinatorial action space that often grows exponentially with the number of agents. As a result, the branching factor of MCTS during tree expansion also increases exponentially, making it very difficult to efficiently explore and exploit during tree search. To this end, we propose MALinZero, a new approach to leverage low-dimensional representational structures on joint-action returns and enable efficient MCTS in complex multi-agent planning. Our solution can be viewed as projecting the joint-action returns into the low-dimensional space representable using a contextual linear bandit problem formulation. We solve the contextual linear bandit problem with convex and $\mu$-smooth loss functions -- in order to place more importance on better joint actions and mitigate potential representational limitations -- and derive a linear Upper Confidence Bound applied to trees (LinUCT) to enable novel multi-agent exploration and exploitation in the low-dimensional space. We analyze the regret of MALinZero for low-dimensional reward functions and propose an $(1-\tfrac1e)$-approximation algorithm for the joint action selection by maximizing a sub-modular objective. MALinZero demonstrates state-of-the-art performance on multi-agent benchmarks such as matrix games, SMAC, and SMACv2, outperforming both model-based and model-free multi-agent reinforcement learning baselines with faster learning speed and better performance.

**arXiv ID:** 2511.06142
</details>

<details>
<summary><strong>Efficient LLM Safety Evaluation through Multi-Agent Debate</strong> - Dachuan Lin, Guobin Shen, Zihao Yang, Tianrong Liu, Dongcheng Zhao, Yi Zeng - [[pdf]](https://arxiv.org/pdf/2511.06396)</summary>

**Abstract:** Safety evaluation of large language models (LLMs) increasingly relies on LLM-as-a-Judge frameworks, but the high cost of frontier models limits scalability. We propose a cost-efficient multi-agent judging framework that employs Small Language Models (SLMs) through structured debates among critic, defender, and judge agents. To rigorously assess safety judgments, we construct HAJailBench, a large-scale human-annotated jailbreak benchmark comprising 12,000 adversarial interactions across diverse attack methods and target models. The dataset provides fine-grained, expert-labeled ground truth for evaluating both safety robustness and judge reliability. Our SLM-based framework achieves agreement comparable to GPT-4o judges on HAJailBench while substantially reducing inference cost. Ablation results show that three rounds of debate yield the optimal balance between accuracy and efficiency. These findings demonstrate that structured, value-aligned debate enables SLMs to capture semantic nuances of jailbreak attacks and that HAJailBench offers a reliable foundation for scalable LLM safety evaluation.

**arXiv ID:** 2511.06396
</details>

<details>
<summary><strong>Agentic AI Sustainability Assessment for Supply Chain Document Insights</strong> - Diego Gosmar, Anna Chiara Pallotta, Giovanni Zenezini - [[pdf]](https://arxiv.org/pdf/2511.07097)</summary>

**Abstract:** This paper presents a comprehensive sustainability assessment framework for document intelligence within supply chain operations, centered on agentic artificial intelligence (AI). We address the dual objective of improving automation efficiency while providing measurable environmental performance in document-intensive workflows. The research compares three scenarios: fully manual (human-only), AI-assisted (human-in-the-loop, HITL), and an advanced multi-agent agentic AI workflow leveraging parsers and verifiers. Empirical results show that AI-assisted HITL and agentic AI scenarios achieve reductions of up to 70-90% in energy consumption, 90-97% in carbon dioxide emissions, and 89-98% in water usage compared to manual processes. Notably, full agentic configurations, combining advanced reasoning (thinking mode) and multi-agent validation, achieve substantial sustainability gains over human-only approaches, even when resource usage increases slightly versus simpler AI-assisted solutions. The framework integrates performance, energy, and emission indicators into a unified ESG-oriented methodology for assessing and governing AI-enabled supply chain solutions. The paper includes a complete replicability use case demonstrating the methodology's application to real-world document extraction tasks.

**arXiv ID:** 2511.07097
</details>

<details>
<summary><strong>AgenticSciML: Collaborative Multi-Agent Systems for Emergent Discovery in Scientific Machine Learning</strong> - Qile Jiang, George Karniadakis - [[pdf]](https://arxiv.org/pdf/2511.07262)</summary>

**Abstract:** Scientific Machine Learning (SciML) integrates data-driven inference with physical modeling to solve complex problems in science and engineering. However, the design of SciML architectures, loss formulations, and training strategies remains an expert-driven research process, requiring extensive experimentation and problem-specific insights. Here we introduce AgenticSciML, a collaborative multi-agent system in which over 10 specialized AI agents collaborate to propose, critique, and refine SciML solutions through structured reasoning and iterative evolution. The framework integrates structured debate, retrieval-augmented method memory, and ensemble-guided evolutionary search, enabling the agents to generate and assess new hypotheses about architectures and optimization procedures. Across physics-informed learning and operator learning tasks, the framework discovers solution methods that outperform single-agent and human-designed baselines by up to four orders of magnitude in error reduction. The agents produce novel strategies -- including adaptive mixture-of-expert architectures, decomposition-based PINNs, and physics-informed operator learning models -- that do not appear explicitly in the curated knowledge base. These results show that collaborative reasoning among AI agents can yield emergent methodological innovation, suggesting a path toward scalable, transparent, and autonomous discovery in scientific computing.

**arXiv ID:** 2511.07262
</details>

<details>
<summary><strong>Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion</strong> - Chen Han, Yijia Ma, Jin Tan, Wenzhen Zheng, Xijin Tang - [[pdf]](https://arxiv.org/pdf/2511.07267)</summary>

**Abstract:** Multi-agent debate (MAD) frameworks have emerged as promising approaches for misinformation detection by simulating adversarial reasoning. While prior work has focused on detection accuracy, it overlooks the importance of helping users understand the reasoning behind factual judgments and develop future resilience. The debate transcripts generated during MAD offer a rich but underutilized resource for transparent reasoning. In this study, we introduce ED2D, an evidence-based MAD framework that extends previous approach by incorporating factual evidence retrieval. More importantly, ED2D is designed not only as a detection framework but also as a persuasive multi-agent system aimed at correcting user beliefs and discouraging misinformation sharing. We compare the persuasive effects of ED2D-generated debunking transcripts with those authored by human experts. Results demonstrate that ED2D outperforms existing baselines across three misinformation detection benchmarks. When ED2D generates correct predictions, its debunking transcripts exhibit persuasive effects comparable to those of human experts; However, when ED2D misclassifies, its accompanying explanations may inadvertently reinforce users'misconceptions, even when presented alongside accurate human explanations. Our findings highlight both the promise and the potential risks of deploying MAD systems for misinformation intervention. We further develop a public community website to help users explore ED2D, fostering transparency, critical thinking, and collaborative fact-checking.

**arXiv ID:** 2511.07267
</details>

<details>
<summary><strong>AdvisingWise: Supporting Academic Advising in Higher Educations Through a Human-in-the-Loop Multi-Agent Framework</strong> - Wendan Jiang, Shiyuan Wang, Hiba Eltigani, Rukhshan Haroon, Abdullah Bin Faisal, Fahad Dogar - [[pdf]](https://arxiv.org/pdf/2511.05706)</summary>

**Abstract:** Academic advising is critical to student success in higher education, yet high student-to-advisor ratios limit advisors' capacity to provide timely support, particularly during peak periods. Recent advances in Large Language Models (LLMs) present opportunities to enhance the advising process. We present AdvisingWise, a multi-agent system that automates time-consuming tasks, such as information retrieval and response drafting, while preserving human oversight. AdvisingWise leverages authoritative institutional resources and adaptively prompts students about their academic backgrounds to generate reliable, personalized responses. All system responses undergo human advisor validation before delivery to students. We evaluate AdvisingWise through a mixed-methods approach: (1) expert evaluation on responses of 20 sample queries, (2) LLM-as-a-judge evaluation of the information retrieval strategy, and (3) a user study with 8 academic advisors to assess the system's practical utility. Our evaluation shows that AdvisingWise produces accurate, personalized responses. Advisors reported increasingly positive perceptions after using AdvisingWise, as their initial concerns about reliability and personalization diminished. We conclude by discussing the implications of human-AI synergy on the practice of academic advising.

**arXiv ID:** 2511.05706
</details>

<details>
<summary><strong>PRAGMA: A Profiling-Reasoned Multi-Agent Framework for Automatic Kernel Optimization</strong> - Kelun Lei, Hailong Yang, Huaitao Zhang, Xin You, Kaige Zhang, Zhongzhi Luan, Yi Liu, Depei Qian - [[pdf]](https://arxiv.org/pdf/2511.06345)</summary>

**Abstract:** Designing high-performance kernels requires expert-level tuning and a deep understanding of hardware characteristics. Recent advances in large language models (LLMs) have enabled automated kernel generation, yet most existing systems rely solely on correctness or execution time feedback, lacking the ability to reason about low-level performance bottlenecks. In this paper, we introduce PRAGMA, a profile-guided AI kernel generation framework that integrates execution feedback and fine-grained hardware profiling into the reasoning loop. PRAGMA enables LLMs to identify performance bottlenecks, preserve historical best versions, and iteratively refine code quality. We evaluate PRAGMA on KernelBench, covering GPU and CPU backends. Results show that PRAGMA consistently outperforms baseline AIKG without profiling enabled and achieves 2.81$\times$ and 2.30$\times$ averaged speedups against Torch on CPU and GPU platforms, respectively.

**arXiv ID:** 2511.06345
</details>

<details>
<summary><strong>When AI Agents Collude Online: Financial Fraud Risks by Collaborative LLM Agents on Social Platforms</strong> - Qibing Ren, Zhijie Zheng, Jiaxuan Guo, Junchi Yan, Lizhuang Ma, Jing Shao - [[pdf]](https://arxiv.org/pdf/2511.06448)</summary>

**Abstract:** In this work, we study the risks of collective financial fraud in large-scale multi-agent systems powered by large language model (LLM) agents. We investigate whether agents can collaborate in fraudulent behaviors, how such collaboration amplifies risks, and what factors influence fraud success. To support this research, we present MultiAgentFraudBench, a large-scale benchmark for simulating financial fraud scenarios based on realistic online interactions. The benchmark covers 28 typical online fraud scenarios, spanning the full fraud lifecycle across both public and private domains. We further analyze key factors affecting fraud success, including interaction depth, activity level, and fine-grained collaboration failure modes. Finally, we propose a series of mitigation strategies, including adding content-level warnings to fraudulent posts and dialogues, using LLMs as monitors to block potentially malicious agents, and fostering group resilience through information sharing at the societal level. Notably, we observe that malicious agents can adapt to environmental interventions. Our findings highlight the real-world risks of multi-agent financial fraud and suggest practical measures for mitigating them. Code is available at this https URL.

**arXiv ID:** 2511.06448
</details>

<details>
<summary><strong>A Multi-Agent System for Semantic Mapping of Relational Data to Knowledge Graphs</strong> - Milena Trajanoska, Riste Stojanov, Dimitar Trajanov - [[pdf]](https://arxiv.org/pdf/2511.06455)</summary>

**Abstract:** Enterprises often maintain multiple databases for storing critical business data in siloed systems, resulting in inefficiencies and challenges with data interoperability. A key to overcoming these challenges lies in integrating disparate data sources, enabling businesses to unlock the full potential of their data. Our work presents a novel approach for integrating multiple databases using knowledge graphs, focusing on the application of large language models as semantic agents for mapping and connecting structured data across systems by leveraging existing vocabularies. The proposed methodology introduces a semantic layer above tables in relational databases, utilizing a system comprising multiple LLM agents that map tables and columns to this http URL terms. Our approach achieves a mapping accuracy of over 90% in multiple domains.

**arXiv ID:** 2511.06455
</details>

<details>
<summary><strong>A Low-Rank Method for Vision Language Model Hallucination Mitigation in Autonomous Driving</strong> - Keke Long, Jiacheng Guo, Tianyun Zhang, Hongkai Yu, Xiaopeng Li - [[pdf]](https://arxiv.org/pdf/2511.06496)</summary>

**Abstract:** Vision Language Models (VLMs) are increasingly used in autonomous driving to help understand traffic scenes, but they sometimes produce hallucinations, which are false details not grounded in the visual input. Detecting and mitigating hallucinations is challenging when ground-truth references are unavailable and model internals are inaccessible. This paper proposes a novel self-contained low-rank approach to automatically rank multiple candidate captions generated by multiple VLMs based on their hallucination levels, using only the captions themselves without requiring external references or model access. By constructing a sentence-embedding matrix and decomposing it into a low-rank consensus component and a sparse residual, we use the residual magnitude to rank captions: selecting the one with the smallest residual as the most hallucination-free. Experiments on the NuScenes dataset demonstrate that our approach achieves 87% selection accuracy in identifying hallucination-free captions, representing a 19% improvement over the unfiltered baseline and a 6-10% improvement over multi-agent debate method. The sorting produced by sparse error magnitudes shows strong correlation with human judgments of hallucinations, validating our scoring mechanism. Additionally, our method, which can be easily parallelized, reduces inference time by 51-67% compared to debate approaches, making it practical for real-time autonomous driving applications.

**arXiv ID:** 2511.06496
</details>

<details>
<summary><strong>S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning</strong> - Jiangwen Dong, Zehui Lin, Wanyu Lin, Mingjin Zhang - [[pdf]](https://arxiv.org/pdf/2511.06727)</summary>

**Abstract:** Large Language Models (LLMs) have achieved impressive performance in complex reasoning problems. Their effectiveness highly depends on the specific nature of the task, especially the required domain knowledge. Existing approaches, such as mixture-of-experts, typically operate at the task level; they are too coarse to effectively solve the heterogeneous problems involving multiple subjects. This work proposes a novel framework that performs fine-grained analysis at subject level equipped with a designated multi-agent collaboration strategy for addressing heterogeneous problem reasoning. Specifically, given an input query, we first employ a Graph Neural Network to identify the relevant subjects and infer their interdependencies to generate an \textit{Subject-based Directed Acyclic Graph} (S-DAG), where nodes represent subjects and edges encode information flow. Then we profile the LLM models by assigning each model a subject-specific expertise score, and select the top-performing one for matching corresponding subject of the S-DAG. Such subject-model matching enables graph-structured multi-agent collaboration where information flows from the starting model to the ending model over S-DAG. We curate and release multi-subject subsets of standard benchmarks (MMLU-Pro, GPQA, MedMCQA) to better reflect complex, real-world reasoning tasks. Extensive experiments show that our approach significantly outperforms existing task-level model selection and multi-agent collaboration baselines in accuracy and efficiency. These results highlight the effectiveness of subject-aware reasoning and structured collaboration in addressing complex and multi-subject problems.

**arXiv ID:** 2511.06727
</details>

<details>
<summary><strong>Multi-Agent Reinforcement Learning for Deadlock Handling among Autonomous Mobile Robots</strong> - Marcel Müller - [[pdf]](https://arxiv.org/pdf/2511.07071)</summary>

**Abstract:** This dissertation explores the application of multi-agent reinforcement learning (MARL) for handling deadlocks in intralogistics systems that rely on autonomous mobile robots (AMRs). AMRs enhance operational flexibility but also increase the risk of deadlocks, which degrade system throughput and reliability. Existing approaches often neglect deadlock handling in the planning phase and rely on rigid control rules that cannot adapt to dynamic operational conditions.
To address these shortcomings, this work develops a structured methodology for integrating MARL into logistics planning and operational control. It introduces reference models that explicitly consider deadlock-capable multi-agent pathfinding (MAPF) problems, enabling systematic evaluation of MARL strategies. Using grid-based environments and an external simulation software, the study compares traditional deadlock handling strategies with MARL-based solutions, focusing on PPO and IMPALA algorithms under different training and execution modes.
Findings reveal that MARL-based strategies, particularly when combined with centralized training and decentralized execution (CTDE), outperform rule-based methods in complex, congested environments. In simpler environments or those with ample spatial freedom, rule-based methods remain competitive due to their lower computational demands. These results highlight that MARL provides a flexible and scalable solution for deadlock handling in dynamic intralogistics scenarios, but requires careful tailoring to the operational context.

**arXiv ID:** 2511.07071
</details>

<details>
<summary><strong>Bridging the Prototype-Production Gap: A Multi-Agent System for Notebooks Transformation</strong> - Hanya Elhashemy, Youssef Lotfy, Yongjian Tang - [[pdf]](https://arxiv.org/pdf/2511.07257)</summary>

**Abstract:** The increasing adoption of Jupyter notebooks in data science and machine learning workflows has created a gap between exploratory code development and production-ready software systems. While notebooks excel at iterative development and visualization, they often lack proper software engineering principles, making their transition to production environments challenging. This paper presents Codelevate, a novel multi-agent system that automatically transforms Jupyter notebooks into well-structured, maintainable Python code repositories. Our system employs three specialized agents - Architect, Developer, and Structure - working in concert through a shared dependency tree to ensure architectural coherence and code quality. Our experimental results validate Codelevate's capability to bridge the prototype-to-production gap through autonomous code transformation, yielding quantifiable improvements in code quality metrics while preserving computational semantics.

**arXiv ID:** 2511.07257
</details>

<details>
<summary><strong>Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation</strong> - Shiyuan Li, Yixin Liu, Qingsong Wen, Chengqi Zhang, Shirui Pan - [[pdf]](https://arxiv.org/pdf/2507.18224)</summary>

**Abstract:** Multi-agent systems (MAS) based on large language models (LLMs) have emerged as a powerful solution for dealing with complex problems across diverse domains. The effectiveness of MAS is critically dependent on its collaboration topology, which has become a focal point for automated design research. However, existing approaches are fundamentally constrained by their reliance on a template graph modification paradigm with a predefined set of agents and hard-coded interaction structures, significantly limiting their adaptability to task-specific requirements. To address these limitations, we reframe MAS design as a conditional autoregressive graph generation task, where both the system composition and structure are designed jointly. We propose ARG-Designer, a novel autoregressive model that operationalizes this paradigm by constructing the collaboration graph from scratch. Conditioned on a natural language task query, ARG-Designer sequentially and dynamically determines the required number of agents, selects their appropriate roles from an extensible pool, and establishes the optimal communication links between them. This generative approach creates a customized topology in a flexible and extensible manner, precisely tailored to the unique demands of different tasks. Extensive experiments across six diverse benchmarks demonstrate that ARG-Designer not only achieves state-of-the-art performance but also enjoys significantly greater token efficiency and enhanced extensibility. The source code of ARG-Designer is available at this https URL.

**arXiv ID:** 2507.18224
</details>

<details>
<summary><strong>Hybrid Training for Enhanced Multi-task Generalization in Multi-agent Reinforcement Learning</strong> - Mingliang Zhang, Sichang Su, Chengyang He, Guillaume Sartoretti - [[pdf]](https://arxiv.org/pdf/2408.13567)</summary>

**Abstract:** In multi-agent reinforcement learning (MARL), achieving multi-task generalization to diverse agents and objectives presents significant challenges. Existing online MARL algorithms primarily focus on single-task performance, but their lack of multi-task generalization capabilities typically results in substantial computational waste and limited real-life applicability. Meanwhile, existing offline multi-task MARL approaches are heavily dependent on data quality, often resulting in poor performance on unseen tasks. In this paper, we introduce HyGen, a novel hybrid MARL framework, Hybrid Training for Enhanced Multi-Task Generalization, which integrates online and offline learning to ensure both multi-task generalization and training efficiency. Specifically, our framework extracts potential general skills from offline multi-task datasets. We then train policies to select the optimal skills under the centralized training and decentralized execution paradigm (CTDE). During this stage, we utilize a replay buffer that integrates both offline data and online interactions. We empirically demonstrate that our framework effectively extracts and refines general skills, yielding impressive generalization to unseen tasks. Comparative analyses on the StarCraft multi-agent challenge show that HyGen outperforms a wide range of existing solely online and offline methods.

**arXiv ID:** 2408.13567
</details>

<details>
<summary><strong>FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation</strong> - Song Jin, Shuqi Li, Shukun Zhang, Rui Yan - [[pdf]](https://arxiv.org/pdf/2511.07322)</summary>

**Abstract:** While LLMs have shown great success in financial tasks like stock prediction and question answering, their application in fully automating Equity Research Report generation remains uncharted territory. In this paper, we formulate the Equity Research Report (ERR) Generation task for the first time. To address the data scarcity and the evaluation metrics absence, we present an open-source evaluation benchmark for ERR generation - FinRpt. We frame a Dataset Construction Pipeline that integrates 7 financial data types and produces a high-quality ERR dataset automatically, which could be used for model training and evaluation. We also introduce a comprehensive evaluation system including 11 metrics to assess the generated ERRs. Moreover, we propose a multi-agent framework specifically tailored to address this task, named FinRpt-Gen, and train several LLM-based agents on the proposed datasets using Supervised Fine-Tuning and Reinforcement Learning. Experimental results indicate the data quality and metrics effectiveness of the benchmark FinRpt and the strong performance of FinRpt-Gen, showcasing their potential to drive innovation in the ERR generation field. All code and datasets are publicly available.

**arXiv ID:** 2511.07322
</details>

<details>
<summary><strong>Surgical Agent Orchestration Platform for Voice-directed Patient Data Interaction</strong> - Hyeryun Park, Byung Mo Gu, Jun Hee Lee, Byeong Hyeon Choi, Sekeun Kim, Hyun Koo Kim, Kyungsang Kim - [[pdf]](https://arxiv.org/pdf/2511.07392)</summary>

**Abstract:** In da Vinci robotic surgery, surgeons' hands and eyes are fully engaged in the procedure, making it difficult to access and manipulate multimodal patient data without interruption. We propose a voice-directed Surgical Agent Orchestrator Platform (SAOP) built on a hierarchical multi-agent framework, consisting of an orchestration agent and three task-specific agents driven by Large Language Models (LLMs). These LLM-based agents autonomously plan, refine, validate, and reason to map voice commands into specific tasks such as retrieving clinical information, manipulating CT scans, or navigating 3D anatomical models on the surgical video. We also introduce a Multi-level Orchestration Evaluation Metric (MOEM) to comprehensively assess the performance and robustness from command-level and category-level perspectives. The SAOP achieves high accuracy and success rates across 240 voice commands, while LLM-based agents improve robustness against speech recognition errors and diverse or ambiguous free-form commands, demonstrating strong potential to support minimally invasive da Vinci robotic surgery.

**arXiv ID:** 2511.07392
</details>

<details>
<summary><strong>Multi-Agent AI Framework for Road Situation Detection and C-ITS Message Generation</strong> - Kailin Tong, Selim Solmaz, Kenan Mujkic, Gottfried Allmer, Bo Leng - [[pdf]](https://arxiv.org/pdf/2511.06892)</summary>

**Abstract:** Conventional road-situation detection methods achieve strong performance in predefined scenarios but fail in unseen cases and lack semantic interpretation, which is crucial for reliable traffic recommendations. This work introduces a multi-agent AI framework that combines multimodal large language models (MLLMs) with vision-based perception for road-situation monitoring. The framework processes camera feeds and coordinates dedicated agents for situation detection, distance estimation, decision-making, and Cooperative Intelligent Transport System (C-ITS) message generation. Evaluation is conducted on a custom dataset of 103 images extracted from 20 videos of the TAD dataset. Both Gemini-2.0-Flash and Gemini-2.5-Flash were evaluated. The results show 100\% recall in situation detection and perfect message schema correctness; however, both models suffer from false-positive detections and have reduced performance in terms of number of lanes, driving lane status and cause code. Surprisingly, Gemini-2.5-Flash, though more capable in general tasks, underperforms Gemini-2.0-Flash in detection accuracy and semantic understanding and incurs higher latency (Table II). These findings motivate further work on fine-tuning specialized LLMs or MLLMs tailored for intelligent transportation applications.

**arXiv ID:** 2511.06892
</details>

<details>
<summary><strong>Disentangled Control of Multi-Agent Systems</strong> - Ruoyu Lin, Gennaro Notomista, Magnus Egerstedt - [[pdf]](https://arxiv.org/pdf/2511.05900)</summary>

**Abstract:** This paper develops a general framework for multi-agent control synthesis, which applies to a wide range of problems with convergence guarantees, regardless of the complexity of the underlying graph topology and the explicit time dependence of the objective function. The proposed framework systematically addresses a particularly challenging problem in multi-agent systems, i.e., decentralization of entangled dynamics among different agents, and it naturally supports multi-objective robotics and real-time implementations. To demonstrate its generality and effectiveness, the framework is implemented across three experiments, namely time-varying leader-follower formation control, decentralized coverage control for time-varying density functions without any approximations, which is a long-standing open problem, and safe formation navigation in dense environments.

**arXiv ID:** 2511.05900
</details>

</details>

<details open>
<summary><h2>Other Agent Research (8 papers)</h2></summary>

<details>
<summary><strong>An Epistemic Perspective on Agent Awareness</strong> - Pavel Naumov, Alexandra Pavlova - [[pdf]](https://arxiv.org/pdf/2511.05977)</summary>

**Abstract:** The paper proposes to treat agent awareness as a form of knowledge, breaking the tradition in the existing literature on awareness. It distinguishes the de re and de dicto forms of such knowledge. The work introduces two modalities capturing these forms and formally specifies their meaning using a version of 2D-semantics. The main technical result is a sound and complete logical system describing the interplay between the two proposed modalities and the standard "knowledge of the fact" modality.

**arXiv ID:** 2511.05977
</details>

<details>
<summary><strong>From Failure Modes to Reliability Awareness in Generative and Agentic AI System</strong> - Janet, Liangwei Zhang - [[pdf]](https://arxiv.org/pdf/2511.05511)</summary>

**Abstract:** This chapter bridges technical analysis and organizational preparedness by tracing the path from layered failure modes to reliability awareness in generative and agentic AI systems. We first introduce an 11-layer failure stack, a structured framework for identifying vulnerabilities ranging from hardware and power foundations to adaptive learning and agentic reasoning. Building on this, the chapter demonstrates how failures rarely occur in isolation but propagate across layers, creating cascading effects with systemic consequences. To complement this diagnostic lens, we develop the concept of awareness mapping: a maturity-oriented framework that quantifies how well individuals and organizations recognize reliability risks across the AI stack. Awareness is treated not only as a diagnostic score but also as a strategic input for AI governance, guiding improvement and resilience planning. By linking layered failures to awareness levels and further integrating this into Dependability-Centred Asset Management (DCAM), the chapter positions awareness mapping as both a measurement tool and a roadmap for trustworthy and sustainable AI deployment across mission-critical domains.

**arXiv ID:** 2511.05511
</details>

<details>
<summary><strong>EVLP:Learning Unified Embodied Vision-Language Planner with Reinforced Supervised Fine-Tuning</strong> - Xinyan Cai, Shiguang Wu, Dafeng Chi, Yuzheng Zhuang, Xingyue Quan, Jianye Hao, Qiang Guan - [[pdf]](https://arxiv.org/pdf/2511.05553)</summary>

**Abstract:** In complex embodied long-horizon manipulation tasks, effective task decomposition and execution require synergistic integration of textual logical reasoning and visual-spatial imagination to ensure efficient and accurate operation. Current methods fail to adopt a unified generation framework for multimodal planning, lead to inconsistent in multimodal planning. To address this challenge, we present \textbf{EVLP (Embodied Vision-Language Planner)}, an innovative multimodal unified generation framework that jointly models linguistic reasoning and visual generation. Our approach achieves multimodal planning for long-horizon tasks through a novel training pipeline incorporating dynamic pretraining and reinforced alignment. Our core innovations consist of three key components: \textbf{1) Unified Multimodal Generation Framework}: For understanding, We integrate semantic information with spatial features to provide comprehensive visual perception. For generation, we directly learn the joint distribution of discrete images for one-step visual synthesis, enabling coordinated language-visual modeling through learnable cross-modal attention mechanisms. \textbf{2) Dynamic Perception Pretraining}: We propose a bidirectional dynamic alignment strategy employing inverse dynamics tasks and forward dynamics tasks, effectively strengthening multimodal correlations within a unified feature space. \textbf{3) Reinforced Supervised Fine-Tuning}: While conducting instruction-based fine-tuning in the unified generation space, we construct a reinforce loss to align the spatial logic between textual actions and generated images, enabling the model to acquire spatio-awared multimodal planning capabilities.

**arXiv ID:** 2511.05553
</details>

<details>
<summary><strong>Adapting Web Agents with Synthetic Supervision</strong> - Zhaoyang Wang, Yiming Liang, Xuchao Zhang, Qianhui Wu, Siwei Han, Anson Bastos, Rujia Wang, Chetan Bansal, Baolin Peng, Jianfeng Gao, Saravan Rajmohan, Huaxiu Yao - [[pdf]](https://arxiv.org/pdf/2511.06101)</summary>

**Abstract:** Web agents struggle to adapt to new websites due to the scarcity of environment specific tasks and demonstrations. Recent works have explored synthetic data generation to address this challenge, however, they suffer from data quality issues where synthesized tasks contain hallucinations that cannot be executed, and collected trajectories are noisy with redundant or misaligned actions. In this paper, we propose SynthAgent, a fully synthetic supervision framework that aims at improving synthetic data quality via dual refinement of both tasks and trajectories. Our approach begins by synthesizing diverse tasks through categorized exploration of web elements, ensuring efficient coverage of the target environment. During trajectory collection, we refine tasks when conflicts with actual observations are detected, mitigating hallucinations while maintaining task consistency. After collection, we conduct trajectory refinement with a global context to mitigate potential noise or misalignments. Finally, we fine-tune open-source web agents on the refined synthetic data to adapt them to the target environment. Experimental results demonstrate that SynthAgent outperforms existing synthetic data methods, validating the importance of high-quality synthetic supervision. The code will be publicly available at this https URL.

**arXiv ID:** 2511.06101
</details>

<details>
<summary><strong>A Graph-Theoretical Perspective on Law Design for Multiagent Systems</strong> - Qi Shi, Pavel Naumov - [[pdf]](https://arxiv.org/pdf/2511.06361)</summary>

**Abstract:** A law in a multiagent system is a set of constraints imposed on agents' behaviours to avoid undesirable outcomes. The paper considers two types of laws: useful laws that, if followed, completely eliminate the undesirable outcomes and gap-free laws that guarantee that at least one agent can be held responsible each time an undesirable outcome occurs. In both cases, we study the problem of finding a law that achieves the desired result by imposing the minimum restrictions.
We prove that, for both types of laws, the minimisation problem is NP-hard even in the simple case of one-shot concurrent interactions. We also show that the approximation algorithm for the vertex cover problem in hypergraphs could be used to efficiently approximate the minimum laws in both cases.

**arXiv ID:** 2511.06361
</details>

<details>
<summary><strong>From Generation to Attribution: Music AI Agent Architectures for the Post-Streaming Era</strong> - Wonil Kim, Hyeongseok Wi, Seungsoon Park, Taejun Kim, Sangeun Keum, Keunhyoung Kim, Taewan Kim, Jongmin Jung, Taehyoung Kim, Gaetan Guerrero, Mael Le Goff, Julie Po, Dongjoo Moon, Juhan Nam, Jongpil Lee - [[pdf]](https://arxiv.org/pdf/2510.20276)</summary>

**Abstract:** Generative AI is reshaping music creation, but its rapid growth exposes structural gaps in attribution, rights management, and economic models. Unlike past media shifts, from live performance to recordings, downloads, and streaming, AI transforms the entire lifecycle of music, collapsing boundaries between creation, distribution, and monetization. However, existing streaming systems, with opaque and concentrated royalty flows, are ill-equipped to handle the scale and complexity of AI-driven production. We propose a content-based Music AI Agent architecture that embeds attribution directly into the creative workflow through block-level retrieval and agentic orchestration. Designed for iterative, session-based interaction, the system organizes music into granular components (Blocks) stored in BlockDB; each use triggers an Attribution Layer event for transparent provenance and real-time settlement. This framework reframes AI from a generative tool into infrastructure for a Fair AI Media Platform. By enabling fine-grained attribution, equitable compensation, and participatory engagement, it points toward a post-streaming paradigm where music functions not as a static catalog but as a collaborative and adaptive ecosystem.

**arXiv ID:** 2510.20276
</details>

<details>
<summary><strong>Leveraging LLM-based agents for social science research: insights from citation network simulations</strong> - Jiarui Ji, Runlin Lei, Xuchen Pan, Zhewei Wei, Hao Sun, Yankai Lin, Xu Chen, Yongzheng Yang, Yaliang Li, Bolin Ding, Ji-Rong Wen - [[pdf]](https://arxiv.org/pdf/2511.03758)</summary>

**Abstract:** The emergence of Large Language Models (LLMs) demonstrates their potential to encapsulate the logic and patterns inherent in human behavior simulation by leveraging extensive web data pre-training. However, the boundaries of LLM capabilities in social simulation remain unclear. To further explore the social attributes of LLMs, we introduce the CiteAgent framework, designed to generate citation networks based on human-behavior simulation with LLM-based agents. CiteAgent successfully captures predominant phenomena in real-world citation networks, including power-law distribution, citational distortion, and shrinking diameter. Building on this realistic simulation, we establish two LLM-based research paradigms in social science: LLM-SE (LLM-based Survey Experiment) and LLM-LE (LLM-based Laboratory Experiment). These paradigms facilitate rigorous analyses of citation network phenomena, allowing us to validate and challenge existing theories. Additionally, we extend the research scope of traditional science of science studies through idealized social experiments, with the simulation experiment results providing valuable insights for real-world academic environments. Our work demonstrates the potential of LLMs for advancing science of science research in social science.

**arXiv ID:** 2511.03758
</details>

<details>
<summary><strong>Simulator Ensembles for Trustworthy Autonomous Driving Testing</strong> - Lev Sorokin, Matteo Biagiola, Andrea Stocco - [[pdf]](https://arxiv.org/pdf/2503.08936)</summary>

**Abstract:** Scenario-based testing with driving simulators is extensively used to identify failing conditions of automated driving assistance systems (ADAS). However, existing studies have shown that repeated test execution in the same as well as in distinct simulators can yield different outcomes, which can be attributed to sources of flakiness or different implementations of the physics. In this paper, we present MultiSim, a novel approach to multi-simulation ADAS testing based on a search-based testing approach that leverages an ensemble of simulators to identify failure-inducing, simulator-agnostic test scenarios. During the search, each scenario is evaluated jointly on multiple simulators. Scenarios that produce consistent results across simulators are prioritized for further exploration, while those that fail on only a subset of simulators are given less priority, as they may reflect simulator-specific issues rather than generalizable failures. Our empirical study, which involves testing three lane-keeping ADAS on different pairs of three widely used simulators, demonstrates that MultiSim outperforms single-simulator testing by achieving, on average, a higher rate of simulator-agnostic failures by 66%. Compared to a state-of-the-art multi-simulator approach that combines the outcome of independent test generation campaigns obtained in different simulators, MultiSim identifies, on average, up to 3.4X more simulator-agnostic failing tests and higher failure rates. To avoid the costly execution of test inputs on which simulators disagree, we propose to predict simulator disagreements and bypass test executions. Our results show that utilizing a surrogate model during the search retains the average number of valid failures and also improves efficiency. Our findings indicate that combining an ensemble of simulators is a promising approach for the automated cross-replication in ADAS testing.

**arXiv ID:** 2503.08936
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (40 papers)</h2></summary>

<details>
<summary><strong>Klear-AgentForge: Forging Agentic Intelligence through Posttraining Scaling</strong> - Qi Wang, Hongzhi Zhang, Jia Fu, Kai Fu, Yahui Liu, Tinghai Zhang, Chenxi Sun, Gangwei Jiang, Jingyi Tang, Xingguang Ji, Yang Yue, Jingyuan Zhang, Fuzheng Zhang, Kun Gai, Guorui Zhou - [[pdf]](https://arxiv.org/pdf/2511.05951)</summary>

**Abstract:** Despite the proliferation of powerful agentic models, the lack of critical post-training details hinders the development of strong counterparts in the open-source community. In this study, we present a comprehensive and fully open-source pipeline for training a high-performance agentic model for interacting with external tools and environments, named Klear-Qwen3-AgentForge, starting from the Qwen3-8B base model. We design effective supervised fine-tuning (SFT) with synthetic data followed by multi-turn reinforcement learning (RL) to unlock the potential for multiple diverse agentic tasks. We perform exclusive experiments on various agentic benchmarks in both tool use and coding domains. Klear-Qwen3-AgentForge-8B achieves state-of-the-art performance among LLMs of similar size and remains competitive with significantly larger models.

**arXiv ID:** 2511.05951
</details>

<details>
<summary><strong>SofT-GRPO: Surpassing Discrete-Token LLM Reinforcement Learning via Gumbel-Reparameterized Soft-Thinking Policy Optimization</strong> - Zhi Zheng, Wee Sun Lee - [[pdf]](https://arxiv.org/pdf/2511.06411)</summary>

**Abstract:** The soft-thinking paradigm for Large Language Model (LLM) reasoning can outperform the conventional discrete-token Chain-of-Thought (CoT) reasoning in some scenarios, underscoring its research and application value. However, while the discrete-token CoT reasoning pattern can be reinforced through policy optimization algorithms such as group relative policy optimization (GRPO), extending the soft-thinking pattern with Reinforcement Learning (RL) remains challenging. This difficulty stems from the complexities of injecting stochasticity into soft-thinking tokens and updating soft-thinking policies accordingly. As a result, previous attempts to combine soft-thinking with GRPO typically underperform their discrete-token GRPO counterparts. To fully unlock the potential of soft-thinking, this paper presents a novel policy optimization algorithm, SofT-GRPO, to reinforce LLMs under the soft-thinking reasoning pattern. SofT-GRPO injects the Gumbel noise into logits, employs the Gumbel-Softmax technique to avoid soft-thinking tokens outside the pre-trained embedding space, and leverages the reparameterization trick in policy gradient. We conduct experiments across base LLMs ranging from 1.5B to 7B parameters, and results demonstrate that SofT-GRPO enables soft-thinking LLMs to slightly outperform discrete-token GRPO on Pass@1 (+0.13% on average accuracy), while exhibiting a substantial uplift on Pass@32 (+2.19% on average accuracy). Codes and weights are available on this https URL

**arXiv ID:** 2511.06411
</details>

<details>
<summary><strong>Brain-Inspired Planning for Better Generalization in Reinforcement Learning</strong> - Mingde "Harry" Zhao - [[pdf]](https://arxiv.org/pdf/2511.06470)</summary>

**Abstract:** Existing Reinforcement Learning (RL) systems encounter significant challenges when applied to real-world scenarios, primarily due to poor generalization across environments that differ from their training conditions. This thesis explores the direction of enhancing agents' zero-shot systematic generalization abilities by granting RL agents reasoning behaviors that are found to help systematic generalization in the human brain. Inspired by human conscious planning behaviors, we first introduced a top-down attention mechanism, which allows a decision-time planning agent to dynamically focus its reasoning on the most relevant aspects of the environmental state given its instantaneous intentions, a process we call "spatial abstraction". This approach significantly improves systematic generalization outside the training tasks. Subsequently, building on spatial abstraction, we developed the Skipper framework to automatically decompose complex tasks into simpler, more manageable sub-tasks. Skipper provides robustness against distributional shifts and efficacy in long-term, compositional planning by focusing on pertinent spatial and temporal elements of the environment. Finally, we identified a common failure mode and safety risk in planning agents that rely on generative models to generate state targets during planning. It is revealed that most agents blindly trust the targets they hallucinate, resulting in delusional planning behaviors. Inspired by how the human brain rejects delusional intentions, we propose learning a feasibility evaluator to enable rejecting hallucinated infeasible targets, which led to significant performance improvements in various kinds of planning agents. Finally, we suggest directions for future research, aimed at achieving general task abstraction and fully enabling abstract planning.

**arXiv ID:** 2511.06470
</details>

<details>
<summary><strong>GRAPH-GRPO-LEX: Contract Graph Modeling and Reinforcement Learning with Group Relative Policy Optimization</strong> - Moriya Dechtiar, Daniel Martin Katz, Mari Sundaresan, Sylvain Jaume, Hongming Wang - [[pdf]](https://arxiv.org/pdf/2511.06618)</summary>

**Abstract:** Contracts are complex documents featuring detailed formal structures, explicit and implicit dependencies and rich semantic content. Given these document properties, contract drafting and manual examination of contracts have proven to be both arduous and susceptible to errors. This work aims to simplify and automate the task of contract review and analysis using a novel framework for transforming legal contracts into structured semantic graphs, enabling computational analysis and data-driven insights. We introduce a detailed ontology mapping core legal contract elements to their graph-theoretic equivalents of nodes and edges. We then present a reinforcement learning based Large Language Model (LLM) framework for segmentation and extraction of entities and relationships from contracts. Our method, GRAPH-GRPO-LEX, incorporates both LLMs and reinforcement learning with group relative policy optimization (GRPO). By applying a carefully drafted reward function of graph metrics, we demonstrate the ability to automatically identify direct relationships between clauses, and even uncover hidden dependencies. Our introduction of the gated GRPO approach shows a strong learning signal and can move contract analysis from a linear, manual reading process to an easily visualized graph. This allows for a more dynamic analysis, including building the groundwork for contract linting similar to what is now practiced in software engineering.

**arXiv ID:** 2511.06618
</details>

<details>
<summary><strong>IterResearch: Rethinking Long-Horizon Agents via Markovian State Reconstruction</strong> - Guoxin Chen, Zile Qiao, Xuanzhong Chen, Donglei Yu, Haotian Xu, Wayne Xin Zhao, Ruihua Song, Wenbiao Yin, Huifeng Yin, Liwen Zhang, Kuan Li, Minpeng Liao, Yong Jiang, Pengjun Xie, Fei Huang, Jingren Zhou - [[pdf]](https://arxiv.org/pdf/2511.07327)</summary>

**Abstract:** Recent advances in deep-research agents have shown promise for autonomous knowledge construction through dynamic reasoning over external sources. However, existing approaches rely on a mono-contextual paradigm that accumulates all information in a single, expanding context window, leading to context suffocation and noise contamination that limit their effectiveness on long-horizon tasks. We introduce IterResearch, a novel iterative deep-research paradigm that reformulates long-horizon research as a Markov Decision Process with strategic workspace reconstruction. By maintaining an evolving report as memory and periodically synthesizing insights, our approach preserves consistent reasoning capacity across arbitrary exploration depths. We further develop Efficiency-Aware Policy Optimization (EAPO), a reinforcement learning framework that incentivizes efficient exploration through geometric reward discounting and enables stable distributed training via adaptive downsampling. Extensive experiments demonstrate that IterResearch achieves substantial improvements over existing open-source agents with average +14.5pp across six benchmarks and narrows the gap with frontier proprietary systems. Remarkably, our paradigm exhibits unprecedented interaction scaling, extending to 2048 interactions with dramatic performance gains (from 3.5\% to 42.5\%), and serves as an effective prompting strategy, improving frontier models by up to 19.2pp over ReAct on long-horizon tasks. These findings position IterResearch as a versatile solution for long-horizon reasoning, effective both as a trained agent and as a prompting paradigm for frontier models.

**arXiv ID:** 2511.07327
</details>

<details>
<summary><strong>CoPRIS: Efficient and Stable Reinforcement Learning via Concurrency-Controlled Partial Rollout with Importance Sampling</strong> - Zekai Qu, Yinxu Pan, Ao Sun, Chaojun Xiao, Xu Han - [[pdf]](https://arxiv.org/pdf/2511.05589)</summary>

**Abstract:** Reinforcement learning (RL) post-training has become a trending paradigm for enhancing the capabilities of large language models (LLMs). Most existing RL systems for LLMs operate in a fully synchronous manner, where training must wait for the rollout of an entire batch to complete. This design leads to severe inefficiencies, as extremely long trajectories can stall the entire rollout process and leave many GPUs idle. To address this issue, we propose Concurrency- Controlled Partial Rollout with Importance Sampling (CoPRIS), which mitigates long-tail inefficiencies by maintaining a fixed number of concurrent rollouts, early-terminating once sufficient samples are collected, and reusing unfinished trajectories in subsequent rollouts. To mitigate the impact of off-policy trajectories, we introduce Cross-stage Importance Sampling Correction, which concatenates buffered log probabilities from the previous policy with those recomputed under the current policy for importance sampling correction. Experiments on challenging mathematical reasoning benchmarks show that CoPRIS achieves up to 1.94x faster training while maintaining comparable or superior performance to synchronous RL systems. The code of CoPRIS is available at this https URL.

**arXiv ID:** 2511.05589
</details>

<details>
<summary><strong>Reinforcement Learning Improves Traversal of Hierarchical Knowledge in LLMs</strong> - Renfei Zhang, Manasa Kaniselvan, Niloofar Mireshghallah - [[pdf]](https://arxiv.org/pdf/2511.05933)</summary>

**Abstract:** Reinforcement learning (RL) is often credited with improving language model reasoning and generalization at the expense of degrading memorized knowledge. We challenge this narrative by observing that RL-enhanced models consistently outperform their base and supervised fine-tuned (SFT) counterparts on pure knowledge recall tasks, particularly those requiring traversal of hierarchical, structured knowledge (e.g., medical codes). We hypothesize these gains stem not from newly acquired data, but from improved procedural skills in navigating and searching existing knowledge hierarchies within the model parameters. To support this hypothesis, we show that structured prompting, which explicitly guides SFTed models through hierarchical traversal, recovers most of the performance gap (reducing 24pp to 7pp on MedConceptsQA for DeepSeek-V3/R1). We further find that while prompting improves final-answer accuracy, RL-enhanced models retain superior ability to recall correct procedural paths on deep-retrieval tasks. Finally our layer-wise internal activation analysis reveals that while factual representations (e.g., activations for the statement "code 57.95 refers to urinary infection") maintain high cosine similarity between SFT and RL models, query representations (e.g., "what is code 57.95") diverge noticeably, indicating that RL primarily transforms how models traverse knowledge rather than the knowledge representation itself.

**arXiv ID:** 2511.05933
</details>

<details>
<summary><strong>Adaptive Agent Selection and Interaction Network for Image-to-point cloud Registration</strong> - Zhixin Cheng, Xiaotian Yin, Jiacheng Deng, Bohao Liao, Yujia Chen, Xu Zhou, Baoqun Yin, Tianzhu Zhang - [[pdf]](https://arxiv.org/pdf/2511.05965)</summary>

**Abstract:** Typical detection-free methods for image-to-point cloud registration leverage transformer-based architectures to aggregate cross-modal features and establish correspondences. However, they often struggle under challenging conditions, where noise disrupts similarity computation and leads to incorrect correspondences. Moreover, without dedicated designs, it remains difficult to effectively select informative and correlated representations across modalities, thereby limiting the robustness and accuracy of registration. To address these challenges, we propose a novel cross-modal registration framework composed of two key modules: the Iterative Agents Selection (IAS) module and the Reliable Agents Interaction (RAI) module. IAS enhances structural feature awareness with phase maps and employs reinforcement learning principles to efficiently select reliable agents. RAI then leverages these selected agents to guide cross-modal interactions, effectively reducing mismatches and improving overall robustness. Extensive experiments on the RGB-D Scenes v2 and 7-Scenes benchmarks demonstrate that our method consistently achieves state-of-the-art performance.

**arXiv ID:** 2511.05965
</details>

<details>
<summary><strong>Revisiting Entropy in Reinforcement Learning for Large Reasoning Models</strong> - Renren Jin, Pengzhi Gao, Yuqi Ren, Zhuowen Han, Tongxuan Zhang, Wuwei Huang, Wei Liu, Jian Luan, Deyi Xiong - [[pdf]](https://arxiv.org/pdf/2511.05993)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has emerged as a predominant approach for enhancing the reasoning capabilities of large language models (LLMs). However, the entropy of LLMs usually collapses during RLVR training, causing premature convergence to suboptimal local minima and hinder further performance improvement. Although various approaches have been proposed to mitigate entropy collapse, a comprehensive study of entropy in RLVR remains lacking. To address this gap, we conduct extensive experiments to investigate the entropy dynamics of LLMs trained with RLVR and analyze how model entropy correlates with response diversity, calibration, and performance across various benchmarks. Our findings reveal that the number of off-policy updates, the diversity of training data, and the clipping thresholds in the optimization objective are critical factors influencing the entropy of LLMs trained with RLVR. Moreover, we theoretically and empirically demonstrate that tokens with positive advantages are the primary contributors to entropy collapse, and that model entropy can be effectively regulated by adjusting the relative loss weights of tokens with positive and negative advantages during training.

**arXiv ID:** 2511.05993
</details>

<details>
<summary><strong>MemoriesDB: A Temporal-Semantic-Relational Database for Long-Term Agent Memory / Modeling Experience as a Graph of Temporal-Semantic Surfaces</strong> - Joel Ward - [[pdf]](https://arxiv.org/pdf/2511.06179)</summary>

**Abstract:** We introduce MemoriesDB, a unified data architecture designed to avoid decoherence across time, meaning, and relation in long-term computational memory. Each memory is a time-semantic-relational entity-a structure that simultaneously encodes when an event occurred, what it means, and how it connects to other events. Built initially atop PostgreSQL with pgvector extensions, MemoriesDB combines the properties of a time-series datastore, a vector database, and a graph system within a single append-only schema. Each memory is represented as a vertex uniquely labeled by its microsecond timestamp and accompanied by low- and high-dimensional normalized embeddings that capture semantic context. Directed edges between memories form labeled relations with per-edge metadata, enabling multiple contextual links between the same vertices. Together these constructs form a time-indexed stack of temporal-semantic surfaces, where edges project as directional arrows in a 1+1-dimensional similarity field, tracing the evolution of meaning through time while maintaining cross-temporal coherence. This formulation supports efficient time-bounded retrieval, hybrid semantic search, and lightweight structural reasoning in a single query path. A working prototype demonstrates scalable recall and contextual reinforcement using standard relational infrastructure, and we discuss extensions toward a columnar backend, distributed clustering, and emergent topic modeling.

**arXiv ID:** 2511.06179
</details>

<details>
<summary><strong>Novel Concepts for Agent-Based Population Modelling and Simulation: Updates from GEPOC ABM</strong> - Martin Bicher, Maximilian Viehauser, Daniele Giannandrea, Hannah Kastinger, Dominik Brunmeir, Niki Popper - [[pdf]](https://arxiv.org/pdf/2511.05637)</summary>

**Abstract:** In recent years, dynamic agent-based population models, which model every inhabitant of a country as a statistically representative agent, have been gaining in popularity for decision support. This is mainly due to their high degree of flexibility with respect to their area of application. GEPOC ABM is one of these models. Developed in 2015, it is now a well-established decision support tool and has been successfully applied for a wide range of population-level research questions ranging from health-care to logistics. At least in part, this success is attributable to continuous improvement and development of new methods. While some of these are very application- or implementation-specific, others can be well transferred to other population models. The focus of the present work lies on the presentation of three selected transferable innovations. We illustrate an innovative time-update concept for the individual agents, a co-simulation-inspired simulation strategy, and a strategy for accurate model parametrisation. We describe these methods in a reproducible manner, explain their advantages and provide ideas on how they can be transferred to other population models.

**arXiv ID:** 2511.05637
</details>

<details>
<summary><strong>Coupling Agent-based Modeling and Life Cycle Assessment to Analyze Trade-offs in Resilient Energy Transitions</strong> - Beichen Zhang, Mohammed T. Zaki, Hanna Breunig, Newsha K. Ajami - [[pdf]](https://arxiv.org/pdf/2511.06791)</summary>

**Abstract:** Transitioning to sustainable and resilient energy systems requires navigating complex and interdependent trade-offs across environmental, social, and resource dimensions. Neglecting these trade-offs can lead to unintended consequences across sectors. However, existing assessments often evaluate emerging energy pathways and their impacts in silos, overlooking critical interactions such as regional resource competition and cumulative impacts. We present an integrated modeling framework that couples agent-based modeling and Life Cycle Assessment (LCA) to simulate how energy transition pathways interact with regional resource competition, ecological constraints, and community-level burdens. We apply the model to a case study in Southern California. The results demonstrate how integrated and multiscale decision making can shape energy pathway deployment and reveal spatially explicit trade-offs under scenario-driven constraints. This modeling framework can further support more adaptive and resilient energy transition planning on spatial and institutional scales.

**arXiv ID:** 2511.06791
</details>

<details>
<summary><strong>RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments</strong> - Zhiyuan Zeng, Hamish Ivison, Yiping Wang, Lifan Yuan, Shuyue Stella Li, Zhuorui Ye, Siting Li, Jacqueline He, Runlong Zhou, Tong Chen, Chenyang Zhao, Yulia Tsvetkov, Simon Shaolei Du, Natasha Jaques, Hao Peng, Pang Wei Koh, Hannaneh Hajishirzi - [[pdf]](https://arxiv.org/pdf/2511.07317)</summary>

**Abstract:** We introduce Reinforcement Learning (RL) with Adaptive Verifiable Environments (RLVE), an approach using verifiable environments that procedurally generate problems and provide algorithmically verifiable rewards, to scale up RL for language models (LMs). RLVE enables each verifiable environment to dynamically adapt its problem difficulty distribution to the policy model's capabilities as training progresses. In contrast, static data distributions often lead to vanishing learning signals when problems are either too easy or too hard for the policy. To implement RLVE, we create RLVE-Gym, a large-scale suite of 400 verifiable environments carefully developed through manual environment engineering. Using RLVE-Gym, we show that environment scaling, i.e., expanding the collection of training environments, consistently improves generalizable reasoning capabilities. RLVE with joint training across all 400 environments in RLVE-Gym yields a 3.37% absolute average improvement across six reasoning benchmarks, starting from one of the strongest 1.5B reasoning LMs. By comparison, continuing this LM's original RL training yields only a 0.49% average absolute gain despite using over 3x more compute. We release our code publicly.

**arXiv ID:** 2511.07317
</details>

<details>
<summary><strong>CG-TTRL: Context-Guided Test-Time Reinforcement Learning for On-Device Large Language Models</strong> - Peyman Hosseini, Ondrej Bohdal, Taha Ceritli, Ignacio Castro, Matthew Purver, Mete Ozay, Umberto Michieli - [[pdf]](https://arxiv.org/pdf/2511.06430)</summary>

**Abstract:** Test-time Reinforcement Learning (TTRL) has shown promise in adapting foundation models for complex tasks at test-time, resulting in large performance improvements. TTRL leverages an elegant two-phase sampling strategy: first, multi-sampling derives a pseudo-label via majority voting, while subsequent downsampling and reward-based fine-tuning encourages the model to explore and learn diverse valid solutions, with the pseudo-label modulating the reward signal. Meanwhile, in-context learning has been widely explored at inference time and demonstrated the ability to enhance model performance without weight updates. However, TTRL's two-phase sampling strategy under-utilizes contextual guidance, which can potentially improve pseudo-label accuracy in the initial exploitation phase while regulating exploration in the second. To address this, we propose context-guided TTRL (CG-TTRL), integrating context dynamically into both sampling phases and propose a method for efficient context selection for on-device applications. Our evaluations on mathematical and scientific QA benchmarks show CG-TTRL outperforms TTRL (e.g. additional 7% relative accuracy improvement over TTRL), while boosting efficiency by obtaining strong performance after only a few steps of test-time training (e.g. 8% relative improvement rather than 1% over TTRL after 3 steps).

**arXiv ID:** 2511.06430
</details>

<details>
<summary><strong>RareAgents: Autonomous Multi-disciplinary Team for Rare Disease Diagnosis and Treatment</strong> - Xuanzhong Chen, Ye Jin, Xiaohao Mao, Lun Wang, Shuyang Zhang, Ting Chen - [[pdf]](https://arxiv.org/pdf/2412.12475)</summary>

**Abstract:** Rare diseases, despite their low individual incidence, collectively impact around 300 million people worldwide due to the vast number of diseases. The involvement of multiple organs and systems, and the shortage of specialized doctors with relevant experience, make diagnosing and treating rare diseases more challenging than common diseases. Recently, agents powered by large language models (LLMs) have demonstrated notable applications across various domains. In the medical field, some agent methods have outperformed direct prompts in question-answering tasks from medical examinations. However, current agent frameworks are not well-adapted to real-world clinical scenarios, especially those involving the complex demands of rare diseases. To bridge this gap, we introduce RareAgents, the first LLM-driven multi-disciplinary team decision-support tool designed specifically for the complex clinical context of rare diseases. RareAgents integrates advanced Multidisciplinary Team (MDT) coordination, memory mechanisms, and medical tools utilization, leveraging Llama-3.1-8B/70B as the base model. Experimental results show that RareAgents outperforms state-of-the-art domain-specific models, GPT-4o, and current agent frameworks in diagnosis and treatment for rare diseases. Furthermore, we contribute a novel rare disease dataset, MIMIC-IV-Ext-Rare, to facilitate further research in this field.

**arXiv ID:** 2412.12475
</details>

<details>
<summary><strong>DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning</strong> - Wenxuan Shi, Haochen Tan, Chuqiao Kuang, Xiaoguang Li, Xiaozhe Ren, Chen Zhang, Hanting Chen, Yasheng Wang, Lu Hou, Lifeng Shang - [[pdf]](https://arxiv.org/pdf/2505.24332)</summary>

**Abstract:** Information seeking demands iterative evidence gathering and reflective reasoning, yet large language models (LLMs) still struggle with it in open-web question answering. Existing prompting and supervised fine-tuning (SFT) methods remain fixed by prompt rules or training corpora, and are usually benchmarked only on well-structured wiki sources, limiting real-world adaptability. We introduce WebPuzzle, a 24k-sample training and 275-sample test benchmark that evaluates information seeking on the live internet, across both wiki and open-domain queries. Leveraging 7k WebPuzzle instances, we develop DeepDiver, a reinforcement-learning (RL) framework that cultivates Search Intensity Scaling (SIS)-an emergent ability to escalate search frequency and depth instead of settling on overconfident, under-evidenced answers. With SIS, Qwen2.5-7B-Instruct and Pangu-7B-Reasoner attain performance on real-web tasks comparable to the 671B-parameter DeepSeek-R1. We detail DeepDiver's curriculum from cold-start SFT to a well designed RL procedure, and show that its seeking policy generalized from closed-ended queries to open-ended generation such as long-form writing. Our results advance adaptive information seeking in LLMs and provide a rigorous benchmark for future work.

**arXiv ID:** 2505.24332
</details>

<details>
<summary><strong>ReCode: Updating Code API Knowledge with Reinforcement Learning</strong> - Haoze Wu, Yunzhi Yao, Wenhao Yu, Ningyu Zhang - [[pdf]](https://arxiv.org/pdf/2506.20495)</summary>

**Abstract:** Large Language Models (LLMs) exhibit remarkable code generation capabilities but falter when adapting to frequent updates in external library APIs. This critical limitation, stemming from reliance on outdated API knowledge from their training data, even with access to current documentation, impedes reliable code generation in dynamic environments. To tackle this issue, we propose ReCode (rule-based Reinforcement learning for Code Update), a novel framework that mimics human programmer adaptation to API changes. Specifically, we construct a dataset of approximately 2,000 data entries to train the LLMs to perform version migration based on updated information. Then, we introduce a modified string similarity metric for code evaluation as the reward for reinforcement learning. Our experiments demonstrate that ReCode substantially boosts LLMs' code generation performance in dynamic API scenarios, especially on the unseen CodeUpdateArena task. Crucially, compared to supervised fine-tuning, ReCode has less impact on LLMs' general code generation abilities. We apply ReCode on various LLMs and reinforcement learning algorithms (GRPO and DAPO), all achieving consistent improvements. Notably, after training, Qwen2.5-Coder-7B outperforms that of the 32B parameter code instruction-tuned model and the reasoning model with the same architecture. Code is available at this https URL.

**arXiv ID:** 2506.20495
</details>

<details>
<summary><strong>EditGRPO: Reinforcement Learning with Post-Rollout Edits for Clinically Accurate Chest X-Ray Report Generation</strong> - Kai Zhang, Christopher Malon, Lichao Sun, Martin Renqiang Min - [[pdf]](https://arxiv.org/pdf/2509.22812)</summary>

**Abstract:** Radiology report generation requires advanced medical image analysis, effective temporal reasoning, and accurate text generation. Although recent innovations, particularly multimodal large language models, have shown improved performance, their supervised fine-tuning (SFT) objective is not explicitly aligned with clinical efficacy. In this work, we introduce EditGRPO, a mixed-policy reinforcement learning algorithm designed specifically to optimize the generation through clinically motivated rewards. EditGRPO integrates on-policy exploration with off-policy guidance by injecting sentence-level detailed corrections during training rollouts. This mixed-policy approach addresses the exploration dilemma and sampling efficiency issues typically encountered in RL. Applied to a Qwen2.5-VL-3B, EditGRPO outperforms both SFT and vanilla GRPO baselines, achieving an average improvement of 3.4\% in clinical metrics across four major datasets. Notably, EditGRPO also demonstrates superior out-of-domain generalization, with an average performance gain of 5.9\% on unseen datasets.

**arXiv ID:** 2509.22812
</details>

<details>
<summary><strong>HaluMem: Evaluating Hallucinations in Memory Systems of Agents</strong> - Ding Chen, Simin Niu, Kehang Li, Peng Liu, Xiangping Zheng, Bo Tang, Xinchi Li, Feiyu Xiong, Zhiyu Li - [[pdf]](https://arxiv.org/pdf/2511.03506)</summary>

**Abstract:** Memory systems are key components that enable AI systems such as LLMs and AI agents to achieve long-term learning and sustained interaction. However, during memory storage and retrieval, these systems frequently exhibit memory hallucinations, including fabrication, errors, conflicts, and omissions. Existing evaluations of memory hallucinations are primarily end-to-end question answering, which makes it difficult to localize the operational stage within the memory system where hallucinations arise. To address this, we introduce the Hallucination in Memory Benchmark (HaluMem), the first operation level hallucination evaluation benchmark tailored to memory systems. HaluMem defines three evaluation tasks (memory extraction, memory updating, and memory question answering) to comprehensively reveal hallucination behaviors across different operational stages of interaction. To support evaluation, we construct user-centric, multi-turn human-AI interaction datasets, HaluMem-Medium and HaluMem-Long. Both include about 15k memory points and 3.5k multi-type questions. The average dialogue length per user reaches 1.5k and 2.6k turns, with context lengths exceeding 1M tokens, enabling evaluation of hallucinations across different context scales and task complexities. Empirical studies based on HaluMem show that existing memory systems tend to generate and accumulate hallucinations during the extraction and updating stages, which subsequently propagate errors to the question answering stage. Future research should focus on developing interpretable and constrained memory operation mechanisms that systematically suppress hallucinations and improve memory reliability.

**arXiv ID:** 2511.03506
</details>

<details>
<summary><strong>The Landscape of Agentic Reinforcement Learning for LLMs: A Survey</strong> - Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi Li, Xiangyuan Xue, Yijiang Li, Yifan Zhou, Yang Chen, Chen Zhang, Yutao Fan, Zihu Wang, Songtao Huang, Francisco Piedrahita-Velez, Yue Liao, Hongru Wang, Mengyue Yang, Heng Ji, Jun Wang, Shuicheng Yan, Philip Torr, Lei Bai - [[pdf]](https://arxiv.org/pdf/2509.02547)</summary>

**Abstract:** The emergence of agentic reinforcement learning (Agentic RL) marks a paradigm shift from conventional reinforcement learning applied to large language models (LLM RL), reframing LLMs from passive sequence generators into autonomous, decision-making agents embedded in complex, dynamic worlds. This survey formalizes this conceptual shift by contrasting the degenerate single-step Markov Decision Processes (MDPs) of LLM-RL with the temporally extended, partially observable Markov decision processes (POMDPs) that define Agentic RL. Building on this foundation, we propose a comprehensive twofold taxonomy: one organized around core agentic capabilities, including planning, tool use, memory, reasoning, self-improvement, and perception, and the other around their applications across diverse task domains. Central to our thesis is that reinforcement learning serves as the critical mechanism for transforming these capabilities from static, heuristic modules into adaptive, robust agentic behavior. To support and accelerate future research, we consolidate the landscape of open-source environments, benchmarks, and frameworks into a practical compendium. By synthesizing over five hundred recent works, this survey charts the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose AI agents.

**arXiv ID:** 2509.02547
</details>

<details>
<summary><strong>Optimizing Predictive Maintenance in Intelligent Manufacturing: An Integrated FNO-DAE-GNN-PPO MDP Framework</strong> - Shiqing Qiu - [[pdf]](https://arxiv.org/pdf/2511.05594)</summary>

**Abstract:** In the era of smart manufacturing, predictive maintenance (PdM) plays a pivotal role in improving equipment reliability and reducing operating costs. In this paper, we propose a novel Markov Decision Process (MDP) framework that integrates advanced soft computing techniques - Fourier Neural Operator (FNO), Denoising Autoencoder (DAE), Graph Neural Network (GNN), and Proximal Policy Optimisation (PPO) - to address the multidimensional challenges of predictive maintenance in complex manufacturing systems. Specifically, the proposed framework innovatively combines the powerful frequency-domain representation capability of FNOs to capture high-dimensional temporal patterns; DAEs to achieve robust, noise-resistant latent state embedding from complex non-Gaussian sensor data; and GNNs to accurately represent inter-device dependencies for coordinated system-wide maintenance decisions. Furthermore, by exploiting PPO, the framework ensures stable and efficient optimisation of long-term maintenance strategies to effectively handle uncertainty and non-stationary dynamics. Experimental validation demonstrates that the approach significantly outperforms multiple deep learning baseline models with up to 13% cost reduction, as well as strong convergence and inter-module synergy. The framework has considerable industrial potential to effectively reduce downtime and operating expenses through data-driven strategies.

**arXiv ID:** 2511.05594
</details>

<details>
<summary><strong>Distributionally Robust Self Paced Curriculum Reinforcement Learning</strong> - Anirudh Satheesh, Keenan Powell, Vaneet Aggarwal - [[pdf]](https://arxiv.org/pdf/2511.05694)</summary>

**Abstract:** A central challenge in reinforcement learning is that policies trained in controlled environments often fail under distribution shifts at deployment into real-world environments. Distributionally Robust Reinforcement Learning (DRRL) addresses this by optimizing for worst-case performance within an uncertainty set defined by a robustness budget $\epsilon$. However, fixing $\epsilon$ results in a tradeoff between performance and robustness: small values yield high nominal performance but weak robustness, while large values can result in instability and overly conservative policies. We propose Distributionally Robust Self-Paced Curriculum Reinforcement Learning (DR-SPCRL), a method that overcomes this limitation by treating $\epsilon$ as a continuous curriculum. DR-SPCRL adaptively schedules the robustness budget according to the agent's progress, enabling a balance between nominal and robust performance. Empirical results across multiple environments demonstrate that DR-SPCRL not only stabilizes training but also achieves a superior robustness-performance trade-off, yielding an average 11.8\% increase in episodic return under varying perturbations compared to fixed or heuristic scheduling strategies, and achieving approximately 1.9$\times$ the performance of the corresponding nominal RL algorithms.

**arXiv ID:** 2511.05694
</details>

<details>
<summary><strong>Approximating Shapley Explanations in Reinforcement Learning</strong> - Daniel Beechey, Özgür Şimşek - [[pdf]](https://arxiv.org/pdf/2511.06094)</summary>

**Abstract:** Reinforcement learning has achieved remarkable success in complex decision-making environments, yet its lack of transparency limits its deployment in practice, especially in safety-critical settings. Shapley values from cooperative game theory provide a principled framework for explaining reinforcement learning; however, the computational cost of Shapley explanations is an obstacle to their use. We introduce FastSVERL, a scalable method for explaining reinforcement learning by approximating Shapley values. FastSVERL is designed to handle the unique challenges of reinforcement learning, including temporal dependencies across multi-step trajectories, learning from off-policy data, and adapting to evolving agent behaviours in real time. FastSVERL introduces a practical, scalable approach for principled and rigorous interpretability in reinforcement learning.

**arXiv ID:** 2511.06094
</details>

<details>
<summary><strong>Guardian-regularized Safe Offline Reinforcement Learning for Smart Weaning of Mechanical Circulatory Devices</strong> - Aysin Tumay, Sophia Sun, Sonia Fereidooni, Aaron Dumas, Elise Jortberg, Rose Yu - [[pdf]](https://arxiv.org/pdf/2511.06111)</summary>

**Abstract:** We study the sequential decision-making problem for automated weaning of mechanical circulatory support (MCS) devices in cardiogenic shock patients. MCS devices are percutaneous micro-axial flow pumps that provide left ventricular unloading and forward blood flow, but current weaning strategies vary significantly across care teams and lack data-driven approaches. Offline reinforcement learning (RL) has proven to be successful in sequential decision-making tasks, but our setting presents challenges for training and evaluating traditional offline RL methods: prohibition of online patient interaction, highly uncertain circulatory dynamics due to concurrent treatments, and limited data availability. We developed an end-to-end machine learning framework with two key contributions (1) Clinically-aware OOD-regularized Model-based Policy Optimization (CORMPO), a density-regularized offline RL algorithm for out-of-distribution suppression that also incorporates clinically-informed reward shaping and (2) a Transformer-based probabilistic digital twin that models MCS circulatory dynamics for policy evaluation with rich physiological and clinical metrics. We prove that \textsf{CORMPO} achieves theoretical performance guarantees under mild assumptions. CORMPO attains a higher reward than the offline RL baselines by 28% and higher scores in clinical metrics by 82.6% on real and synthetic datasets. Our approach offers a principled framework for safe offline policy learning in high-stakes medical applications where domain expertise and safety constraints are essential.

**arXiv ID:** 2511.06111
</details>

<details>
<summary><strong>Deep Reinforcement Learning for Dynamic Origin-Destination Matrix Estimation in Microscopic Traffic Simulations Considering Credit Assignment</strong> - Donggyu Min, Seongjin Choi, Dong-Kyu Kim - [[pdf]](https://arxiv.org/pdf/2511.06229)</summary>

**Abstract:** This paper focuses on dynamic origin-destination matrix estimation (DODE), a crucial calibration process necessary for the effective application of microscopic traffic simulations. The fundamental challenge of the DODE problem in microscopic simulations stems from the complex temporal dynamics and inherent uncertainty of individual vehicle dynamics. This makes it highly challenging to precisely determine which vehicle traverses which link at any given moment, resulting in intricate and often ambiguous relationships between origin-destination (OD) matrices and their contributions to resultant link flows. This phenomenon constitutes the credit assignment problem, a central challenge addressed in this study. We formulate the DODE problem as a Markov Decision Process (MDP) and propose a novel framework that applies model-free deep reinforcement learning (DRL). Within our proposed framework, the agent learns an optimal policy to sequentially generate OD matrices, refining its strategy through direct interaction with the simulation environment. The proposed method is validated on the Nguyen-Dupuis network using SUMO, where its performance is evaluated against ground-truth link flows aggregated at 5-minute intervals over a 30-minute horizon. Experimental results demonstrate that our approach achieves a 43.2% reduction in mean squared error (MSE) compared to the best-performing conventional baseline. By reframing DODE as a sequential decision-making problem, our approach addresses the credit assignment challenge through its learned policy, thereby overcoming the limitations of conventional methods and proposing a novel framework for calibration of microscopic traffic simulations.

**arXiv ID:** 2511.06229
</details>

<details>
<summary><strong>DRIVE: Data Curation Best Practices for Reinforcement Learning with Verifiable Reward in Competitive Code Generation</strong> - Speed Zhu, Jianwei Cai, Guang Chen, Lulu Wu, Saiyong Yang, Wiggin Zhou - [[pdf]](https://arxiv.org/pdf/2511.06307)</summary>

**Abstract:** Recent reasoning-first models (e.g., OpenAI o1, DeepSeek R1) have spurred a resurgence of interest in RLVR. Nevertheless, advances are dominated by mathematics (e.g., AIME), with competitive-programming code generation underexplored and data curation receiving less attention than RL algorithm design. We investigate how to construct RLVR datasets (i.e., RL prompts) and present practical training techniques that yield strong performance on competitive-programming code generation. Our pipeline begins with supervised fine-tuning (SFT) distilled from strong open-source models, augmented with general-purpose and reasoning-intensive data. RL then follows a two-stage process with executable, testcase-driven rewards: first, training on a large, uniformly distributed set of competitive-programming problems using Group Relative Policy Optimization (GRPO) with 8 rollouts per prompt and a relatively short response-generation window (e.g., 32k during SFT and 24k in this stage) to expand entropy and mitigate repetition and truncation; second, we perform \textbf{Pre-GRPO}: updating on a small, high-quality set of challenging problems with a large rollout budget (64 rollouts per prompt) under a hard-focus curriculum that continuously retains the most difficult instances throughout training. We implement our method on Qwen2.5-32B and evaluate on LeetCode and Codeforces weekly contests to avoid data leakage. The resulting model achieves state-of-the-art performance among models of similar scale and is comparable to leading systems such as DeepSeek v3.1 and Doubao-1.5-Thinking. We also examine scaling trends and observe strong RL scaling on an internal large-scale MoE model. Our study distills concise best practices for data curation, entropy expansion, and curriculum design in RLVR for competitive-programming code generation.

**arXiv ID:** 2511.06307
</details>

<details>
<summary><strong>Practical Policy Distillation for Reinforcement Learning in Radio Access Networks</strong> - Sara Khosravi, Burak Demirel, Linghui Zhou, Javier Rasines, Pablo Soldati - [[pdf]](https://arxiv.org/pdf/2511.06563)</summary>

**Abstract:** Adopting artificial intelligence (AI) in radio access networks (RANs) presents several challenges, including limited availability of link-level measurements (e.g., CQI reports), stringent real-time processing constraints (e.g., sub-1 ms per TTI), and network heterogeneity (different spectrum bands, cell types, and vendor equipment). A critical yet often overlooked barrier lies in the computational and memory limitations of RAN baseband hardware, particularly in legacy 4th Generation (4G) systems, which typically lack on-chip neural accelerators. As a result, only lightweight AI models (under 1 Mb and sub-100~\mu s inference time) can be effectively deployed, limiting both their performance and applicability. However, achieving strong generalization across diverse network conditions often requires large-scale models with substantial resource demands. To address this trade-off, this paper investigates policy distillation in the context of a reinforcement learning-based link adaptation task. We explore two strategies: single-policy distillation, where a scenario-agnostic teacher model is compressed into one generalized student model; and multi-policy distillation, where multiple scenario-specific teachers are consolidated into a single generalist student. Experimental evaluations in a high-fidelity, 5th Generation (5G)-compliant simulator demonstrate that both strategies produce compact student models that preserve the teachers' generalization capabilities while complying with the computational and memory limitations of existing RAN hardware.

**arXiv ID:** 2511.06563
</details>

<details>
<summary><strong>Controllable Flow Matching for Online Reinforcement Learning</strong> - Bin Wang, Boxiang Tao, Haifeng Jing, Hongbo Dou, Zijian Wang - [[pdf]](https://arxiv.org/pdf/2511.06816)</summary>

**Abstract:** Model-based reinforcement learning (MBRL) typically relies on modeling environment dynamics for data efficiency. However, due to the accumulation of model errors over long-horizon rollouts, such methods often face challenges in maintaining modeling stability. To address this, we propose CtrlFlow, a trajectory-level synthetic method using conditional flow matching (CFM), which directly modeling the distribution of trajectories from initial states to high-return terminal states without explicitly modeling the environment transition function. Our method ensures optimal trajectory sampling by minimizing the control energy governed by the non-linear Controllability Gramian Matrix, while the generated diverse trajectory data significantly enhances the robustness and cross-task generalization of policy learning. In online settings, CtrlFlow demonstrates the better performance on common MuJoCo benchmark tasks than dynamics models and achieves superior sample efficiency compared to standard MBRL methods.

**arXiv ID:** 2511.06816
</details>

<details>
<summary><strong>On The Presence of Double-Descent in Deep Reinforcement Learning</strong> - Viktor Veselý, Aleksandar Todorov, Matthia Sabatelli - [[pdf]](https://arxiv.org/pdf/2511.06895)</summary>

**Abstract:** The double descent (DD) paradox, where over-parameterized models see generalization improve past the interpolation point, remains largely unexplored in the non-stationary domain of Deep Reinforcement Learning (DRL). We present preliminary evidence that DD exists in model-free DRL, investigating it systematically across varying model capacity using the Actor-Critic framework. We rely on an information-theoretic metric, Policy Entropy, to measure policy uncertainty throughout training. Preliminary results show a clear epoch-wise DD curve; the policy's entrance into the second descent region correlates with a sustained, significant reduction in Policy Entropy. This entropic decay suggests that over-parameterization acts as an implicit regularizer, guiding the policy towards robust, flatter minima in the loss landscape. These findings establish DD as a factor in DRL and provide an information-based mechanism for designing agents that are more general, transferable, and robust.

**arXiv ID:** 2511.06895
</details>

<details>
<summary><strong>Learning to Focus: Prioritizing Informative Histories with Structured Attention Mechanisms in Partially Observable Reinforcement Learning</strong> - Daniel De Dios Allegue, Jinke He, Frans A. Oliehoek - [[pdf]](https://arxiv.org/pdf/2511.06946)</summary>

**Abstract:** Transformers have shown strong ability to model long-term dependencies and are increasingly adopted as world models in model-based reinforcement learning (RL) under partial observability. However, unlike natural language corpora, RL trajectories are sparse and reward-driven, making standard self-attention inefficient because it distributes weight uniformly across all past tokens rather than emphasizing the few transitions critical for control. To address this, we introduce structured inductive priors into the self-attention mechanism of the dynamics head: (i) per-head memory-length priors that constrain attention to task-specific windows, and (ii) distributional priors that learn smooth Gaussian weightings over past state-action pairs. We integrate these mechanisms into UniZero, a model-based RL agent with a Transformer-based world model that supports planning under partial observability. Experiments on the Atari 100k benchmark show that most efficiency gains arise from the Gaussian prior, which smoothly allocates attention to informative transitions, while memory-length priors often truncate useful signals with overly restrictive cut-offs. In particular, Gaussian Attention achieves a 77% relative improvement in mean human-normalized scores over UniZero. These findings suggest that in partially observable RL domains with non-stationary temporal dependencies, discrete memory windows are difficult to learn reliably, whereas smooth distributional priors flexibly adapt across horizons and yield more robust data efficiency. Overall, our results demonstrate that encoding structured temporal priors directly into self-attention improves the prioritization of informative histories for dynamics modeling under partial observability.

**arXiv ID:** 2511.06946
</details>

<details>
<summary><strong>Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning</strong> - Hyunsoo Park, Aron Walsh - [[pdf]](https://arxiv.org/pdf/2511.07158)</summary>

**Abstract:** Discovering functional crystalline materials entails navigating an immense combinatorial design space. While recent advances in generative artificial intelligence have enabled the sampling of chemically plausible compositions and structures, a fundamental challenge remains: the objective misalignment between likelihood-based sampling in generative modelling and targeted focus on underexplored regions where novel compounds reside. Here, we introduce a reinforcement learning framework that guides latent denoising diffusion models toward diverse and novel, yet thermodynamically viable crystalline compounds. Our approach integrates group relative policy optimisation with verifiable, multi-objective rewards that jointly balance creativity, stability, and diversity. Beyond de novo generation, we demonstrate enhanced property-guided design that preserves chemical validity, while targeting desired functional properties. This approach establishes a modular foundation for controllable AI-driven inverse design that addresses the novelty-validity trade-off across scientific discovery applications of generative models.

**arXiv ID:** 2511.07158
</details>

<details>
<summary><strong>Superhuman AI for Stratego Using Self-Play Reinforcement Learning and Test-Time Search</strong> - Samuel Sokota, Eugene Vinitsky, Hengyuan Hu, J. Zico Kolter, Gabriele Farina - [[pdf]](https://arxiv.org/pdf/2511.07312)</summary>

**Abstract:** Few classical games have been regarded as such significant benchmarks of artificial intelligence as to have justified training costs in the millions of dollars. Among these, Stratego -- a board wargame exemplifying the challenge of strategic decision making under massive amounts of hidden information -- stands apart as a case where such efforts failed to produce performance at the level of top humans. This work establishes a step change in both performance and cost for Stratego, showing that it is now possible not only to reach the level of top humans, but to achieve vastly superhuman level -- and that doing so requires not an industrial budget, but merely a few thousand dollars. We achieved this result by developing general approaches for self-play reinforcement learning and test-time search under imperfect information.

**arXiv ID:** 2511.07312
</details>

<details>
<summary><strong>Grounding Computer Use Agents on Human Demonstrations</strong> - Aarash Feizi, Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Kaixin Li, Rabiul Awal, Xing Han Lù, Johan Obando-Ceron, Juan A. Rodriguez, Nicolas Chapados, David Vazquez, Adriana Romero-Soriano, Reihaneh Rabbany, Perouz Taslakian, Christopher Pal, Spandana Gella, Sai Rajeswar - [[pdf]](https://arxiv.org/pdf/2511.07332)</summary>

**Abstract:** Building reliable computer-use agents requires grounding: accurately connecting natural language instructions to the correct on-screen elements. While large datasets exist for web and mobile interactions, high-quality resources for desktop environments are limited. To address this gap, we introduce GroundCUA, a large-scale desktop grounding dataset built from expert human demonstrations. It covers 87 applications across 12 categories and includes 56K screenshots, with every on-screen element carefully annotated for a total of over 3.56M human-verified annotations. From these demonstrations, we generate diverse instructions that capture a wide range of real-world tasks, providing high-quality data for model training. Using GroundCUA, we develop the GroundNext family of models that map instructions to their target UI elements. At both 3B and 7B scales, GroundNext achieves state-of-the-art results across five benchmarks using supervised fine-tuning, while requiring less than one-tenth the training data of prior work. Reinforcement learning post-training further improves performance, and when evaluated in an agentic setting on the OSWorld benchmark using o3 as planner, GroundNext attains comparable or superior results to models trained with substantially more data,. These results demonstrate the critical role of high-quality, expert-driven datasets in advancing general-purpose computer-use agents.

**arXiv ID:** 2511.07332
</details>

<details>
<summary><strong>Sim-to-Real Transfer in Deep Reinforcement Learning for Bipedal Locomotion</strong> - Lingfan Bao, Tianhu Peng, Chengxu Zhou - [[pdf]](https://arxiv.org/pdf/2511.06465)</summary>

**Abstract:** This chapter addresses the critical challenge of simulation-to-reality (sim-to-real) transfer for deep reinforcement learning (DRL) in bipedal locomotion. After contextualizing the problem within various control architectures, we dissect the ``curse of simulation'' by analyzing the primary sources of sim-to-real gap: robot dynamics, contact modeling, state estimation, and numerical solvers. Building on this diagnosis, we structure the solutions around two complementary philosophies. The first is to shrink the gap through model-centric strategies that systematically improve the simulator's physical fidelity. The second is to harden the policy, a complementary approach that uses in-simulation robustness training and post-deployment adaptation to make the policy inherently resilient to model inaccuracies. The chapter concludes by synthesizing these philosophies into a strategic framework, providing a clear roadmap for developing and evaluating robust sim-to-real solutions.

**arXiv ID:** 2511.06465
</details>

<details>
<summary><strong>Adaptive PID Control for Robotic Systems via Hierarchical Meta-Learning and Reinforcement Learning with Physics-Based Data Augmentation</strong> - JiaHao Wu, ShengWen Yu - [[pdf]](https://arxiv.org/pdf/2511.06500)</summary>

**Abstract:** Proportional-Integral-Derivative (PID) controllers remain the predominant choice in industrial robotics due to their simplicity and reliability. However, manual tuning of PID parameters for diverse robotic platforms is time-consuming and requires extensive domain expertise. This paper presents a novel hierarchical control framework that combines meta-learning for PID initialization and reinforcement learning (RL) for online adaptation. To address the sample efficiency challenge, a \textit{physics-based data augmentation} strategy is introduced that generates virtual robot configurations by systematically perturbing physical parameters, enabling effective meta-learning with limited real robot data. The proposed approach is evaluated on two heterogeneous platforms: a 9-DOF Franka Panda manipulator and a 12-DOF Laikago quadruped robot. Experimental results demonstrate that the proposed method achieves 16.6\% average improvement on Franka Panda (6.26° MAE), with exceptional gains in high-load joints (J2: 80.4\% improvement from 12.36° to 2.42°). Critically, this work discovers the \textit{optimization ceiling effect}: RL achieves dramatic improvements when meta-learning exhibits localized high-error joints, but provides no benefit (0.0\%) when baseline performance is uniformly strong, as observed in Laikago. The method demonstrates robust performance under disturbances (parameter uncertainty: +19.2\%, no disturbance: +16.6\%, average: +10.0\%) with only 10 minutes of training time. Multi-seed analysis across 100 random initializations confirms stable performance (4.81+/-1.64\% average). These results establish that RL effectiveness is highly dependent on meta-learning baseline quality and error distribution, providing important design guidance for hierarchical control systems.

**arXiv ID:** 2511.06500
</details>

<details>
<summary><strong>Underactuated Biomimetic Autonomous Underwater Vehicle for Ecosystem Monitoring</strong> - Kaustubh Singh, Shivam Kumar, Shashikant Pawar, Sandeep Manjanna - [[pdf]](https://arxiv.org/pdf/2511.06578)</summary>

**Abstract:** In this paper, we present an underactuated biomimetic underwater robot that is suitable for ecosystem monitoring in both marine and freshwater environments. We present an updated mechanical design for a fish-like robot and propose minimal actuation behaviors learned using reinforcement learning techniques. We present our preliminary mechanical design of the tail oscillation mechanism and illustrate the swimming behaviors on FishGym simulator, where the reinforcement learning techniques will be tested on

**arXiv ID:** 2511.06578
</details>

<details>
<summary><strong>Physically-Grounded Goal Imagination: Physics-Informed Variational Autoencoder for Self-Supervised Reinforcement Learning</strong> - Lan Thi Ha Nguyen, Kien Ton Manh, Anh Do Duc, Nam Pham Hai - [[pdf]](https://arxiv.org/pdf/2511.06745)</summary>

**Abstract:** Self-supervised goal-conditioned reinforcement learning enables robots to autonomously acquire diverse skills without human supervision. However, a central challenge is the goal setting problem: robots must propose feasible and diverse goals that are achievable in their current environment. Existing methods like RIG (Visual Reinforcement Learning with Imagined Goals) use variational autoencoder (VAE) to generate goals in a learned latent space but have the limitation of producing physically implausible goals that hinder learning efficiency. We propose Physics-Informed RIG (PI-RIG), which integrates physical constraints directly into the VAE training process through a novel Enhanced Physics-Informed Variational Autoencoder (Enhanced p3-VAE), enabling the generation of physically consistent and achievable goals. Our key innovation is the explicit separation of the latent space into physics variables governing object dynamics and environmental factors capturing visual appearance, while enforcing physical consistency through differential equation constraints and conservation laws. This enables the generation of physically consistent and achievable goals that respect fundamental physical principles such as object permanence, collision constraints, and dynamic feasibility. Through extensive experiments, we demonstrate that this physics-informed goal generation significantly improves the quality of proposed goals, leading to more effective exploration and better skill acquisition in visual robotic manipulation tasks including reaching, pushing, and pick-and-place scenarios.

**arXiv ID:** 2511.06745
</details>

<details>
<summary><strong>Dynamics-Decoupled Trajectory Alignment for Sim-to-Real Transfer in Reinforcement Learning for Autonomous Driving</strong> - Thomas Steinecker, Alexander Bienemann, Denis Trescher, Thorsten Luettel, Mirko Maehlisch - [[pdf]](https://arxiv.org/pdf/2511.07155)</summary>

**Abstract:** Reinforcement learning (RL) has shown promise in robotics, but deploying RL on real vehicles remains challenging due to the complexity of vehicle dynamics and the mismatch between simulation and reality. Factors such as tire characteristics, road surface conditions, aerodynamic disturbances, and vehicle load make it infeasible to model real-world dynamics accurately, which hinders direct transfer of RL agents trained in simulation. In this paper, we present a framework that decouples motion planning from vehicle control through a spatial and temporal alignment strategy between a virtual vehicle and the real system. An RL agent is first trained in simulation using a kinematic bicycle model to output continuous control actions. Its behavior is then distilled into a trajectory-predicting agent that generates finite-horizon ego-vehicle trajectories, enabling synchronization between virtual and real vehicles. At deployment, a Stanley controller governs lateral dynamics, while longitudinal alignment is maintained through adaptive update mechanisms that compensate for deviations between virtual and real trajectories. We validate our approach on a real vehicle and demonstrate that the proposed alignment strategy enables robust zero-shot transfer of RL-based motion planning from simulation to reality, successfully decoupling high-level trajectory generation from low-level vehicle control.

**arXiv ID:** 2511.07155
</details>

<details>
<summary><strong>Towards Human-AI-Robot Collaboration and AI-Agent based Digital Twins for Parkinson's Disease Management: Review and Outlook</strong> - Hassan Hizeh, Rim Chighri, Muhammad Mahboob Ur Rahman, Mohamed A. Bahloul, Ali Muqaibel, Tareq Y. Al-Naffouri - [[pdf]](https://arxiv.org/pdf/2511.06036)</summary>

**Abstract:** The current body of research on Parkinson's disease (PD) screening, monitoring, and management has evolved along two largely independent trajectories. The first research community focuses on multimodal sensing of PD-related biomarkers using noninvasive technologies such as inertial measurement units (IMUs), force/pressure insoles, electromyography (EMG), electroencephalography (EEG), speech and acoustic analysis, and RGB/RGB-D motion capture systems. These studies emphasize data acquisition, feature extraction, and machine learning-based classification for PD screening, diagnosis, and disease progression modeling. In parallel, a second research community has concentrated on robotic intervention and rehabilitation, employing socially assistive robots (SARs), robot-assisted rehabilitation (RAR) systems, and virtual reality (VR)-integrated robotic platforms for improving motor and cognitive function, enhancing social engagement, and supporting caregivers. Despite the complementary goals of these two domains, their methodological and technological integration remains limited, with minimal data- level or decision-level coupling between the two. With the advent of advanced artificial intelligence (AI), including large language models (LLMs), agentic AI systems, a unique opportunity now exists to unify these research streams. We envision a closed-loop sensor-AI-robot framework in which multimodal sensing continuously guides the interaction between the patient, caregiver, humanoid robot (and physician) through AI agents that are powered by a multitude of AI models such as robotic and wearables foundation models, LLM-based reasoning, reinforcement learning, and continual learning. Such closed-loop system enables personalized, explainable, and context-aware intervention, forming the basis for digital twin of the PD patient that can adapt over time to deliver intelligent, patient-centered PD care.

**arXiv ID:** 2511.06036
</details>

<details>
<summary><strong>TwinOR: Photorealistic Digital Twins of Dynamic Operating Rooms for Embodied AI Research</strong> - Han Zhang, Yiqing Shen, Roger D. Soberanis-Mukul, Ankita Ghosh, Hao Ding, Lalithkumar Seenivasan, Jose L. Porras, Zhekai Mao, Chenjia Li, Wenjie Xiao, Lonny Yarmus, Angela Christine Argento, Masaru Ishii, Mathias Unberath - [[pdf]](https://arxiv.org/pdf/2511.07412)</summary>

**Abstract:** Developing embodied AI for intelligent surgical systems requires safe, controllable environments for continual learning and evaluation. However, safety regulations and operational constraints in operating rooms (ORs) limit embodied agents from freely perceiving and interacting in realistic settings. Digital twins provide high-fidelity, risk-free environments for exploration and training. How we may create photorealistic and dynamic digital representations of ORs that capture relevant spatial, visual, and behavioral complexity remains unclear. We introduce TwinOR, a framework for constructing photorealistic, dynamic digital twins of ORs for embodied AI research. The system reconstructs static geometry from pre-scan videos and continuously models human and equipment motion through multi-view perception of OR activities. The static and dynamic components are fused into an immersive 3D environment that supports controllable simulation and embodied exploration. The proposed framework reconstructs complete OR geometry with centimeter level accuracy while preserving dynamic interaction across surgical workflows, enabling realistic renderings and a virtual playground for embodied AI systems. In our experiments, TwinOR simulates stereo and monocular sensor streams for geometry understanding and visual localization tasks. Models such as FoundationStereo and ORB-SLAM3 on TwinOR-synthesized data achieve performance within their reported accuracy on real indoor datasets, demonstrating that TwinOR provides sensor-level realism sufficient for perception and localization challenges. By establishing a real-to-sim pipeline for constructing dynamic, photorealistic digital twins of OR environments, TwinOR enables the safe, scalable, and data-efficient development and benchmarking of embodied AI, ultimately accelerating the deployment of embodied AI from sim-to-real.

**arXiv ID:** 2511.07412
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
