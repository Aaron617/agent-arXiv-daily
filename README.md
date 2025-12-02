# Agent arXiv Daily

**Last Updated:** 2025-12-02 02:19:07

**Total Papers:** 78

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
<summary><strong>Affective Multimodal Agents with Proactive Knowledge Grounding for Emotionally Aligned Marketing Dialogue</strong> - Lin Yu, Xiaofei Han, Yifei Kang, Chiung-Yi Tseng, Danyang Zhang, Ziqian Bi, Zhimo Han - [[pdf]](https://arxiv.org/pdf/2511.21728)</summary>

**Abstract:** Recent advances in large language models (LLMs) have enabled fluent dialogue systems, but most remain reactive and struggle in emotionally rich, goal-oriented settings such as marketing conversations. To address this limitation, we propose AffectMind, a multimodal affective dialogue agent that performs proactive reasoning and dynamic knowledge grounding to sustain emotionally aligned and persuasive interactions. AffectMind combines three components: a Proactive Knowledge Grounding Network (PKGN) that continuously updates factual and affective context from text, vision, and prosody; an Emotion--Intent Alignment Model (EIAM) that jointly models user emotion and purchase intent to adapt persuasion strategies; and a Reinforced Discourse Loop (RDL) that optimizes emotional coherence and engagement via reinforcement signals from user responses. Experiments on two newly curated marketing dialogue datasets, MM-ConvMarket and AffectPromo, show that AffectMind outperforms strong LLM-based baselines in emotional consistency (+26\%), persuasive success rate (+19\%), and long-term user engagement (+23\%), highlighting emotion-grounded proactivity as a key capability for commercial multimodal agents.

**arXiv ID:** 2511.21728
</details>

<details>
<summary><strong>TinyLLM: Evaluation and Optimization of Small Language Models for Agentic Tasks on Edge Devices</strong> - Mohd Ariful Haque, Fahad Rahman, Kishor Datta Gupta, Khalil Shujaee, Roy George - [[pdf]](https://arxiv.org/pdf/2511.22138)</summary>

**Abstract:** This paper investigates the effectiveness of small language models (SLMs) for agentic tasks (function/tool/API calling) with a focus on running agents on edge devices without reliance on cloud infrastructure. We evaluate SLMs using the Berkeley Function Calling Leaderboard (BFCL) framework and describe parameter-driven optimization strategies that include supervised fine-tuning (SFT), parameter-efficient fine-tuning (PEFT), reinforcement learning (RL)-based optimization, preference alignment via Direct Preference Optimization (DPO), and hybrid methods. We report results for models including TinyAgent, TinyLlama, Qwen, and xLAM across BFCL categories (simple, multiple, parallel, parallel-multiple, and relevance detection), both in live and non-live settings, and in multi-turn evaluations. We additionally detail a DPO training pipeline constructed from AgentBank data (e.g., ALFRED), including our conversion of SFT data to chosen-rejected pairs using TinyLlama responses as rejected outputs and manual validation. Our results demonstrate clear accuracy differences across model scales where medium-sized models (1-3B parameters) significantly outperform ultra-compact models (<1B parameters), achieving up to 65.74% overall accuracy, and 55.62% multi-turn accuracy with hybrid optimization. This study highlights the importance of hybrid optimization strategies that enable small language models to deliver accurate, efficient, and stable agentic AI on edge devices, making privacy-preserving, low-latency autonomous agents practical beyond the cloud.

**arXiv ID:** 2511.22138
</details>

<details>
<summary><strong>ShieldAgent: Shielding Agents via Verifiable Safety Policy Reasoning</strong> - Zhaorun Chen, Mintong Kang, Bo Li - [[pdf]](https://arxiv.org/pdf/2503.22738)</summary>

**Abstract:** Autonomous agents powered by foundation models have seen widespread adoption across various real-world applications. However, they remain highly vulnerable to malicious instructions and attacks, which can result in severe consequences such as privacy breaches and financial losses. More critically, existing guardrails for LLMs are not applicable due to the complex and dynamic nature of agents. To tackle these challenges, we propose ShieldAgent, the first guardrail agent designed to enforce explicit safety policy compliance for the action trajectory of other protected agents through logical reasoning. Specifically, ShieldAgent first constructs a safety policy model by extracting verifiable rules from policy documents and structuring them into a set of action-based probabilistic rule circuits. Given the action trajectory of the protected agent, ShieldAgent retrieves relevant rule circuits and generates a shielding plan, leveraging its comprehensive tool library and executable code for formal verification. In addition, given the lack of guardrail benchmarks for agents, we introduce ShieldAgent-Bench, a dataset with 3K safety-related pairs of agent instructions and action trajectories, collected via SOTA attacks across 6 web environments and 7 risk categories. Experiments show that ShieldAgent achieves SOTA on ShieldAgent-Bench and three existing benchmarks, outperforming prior methods by 11.3% on average with a high recall of 90.1%. Additionally, ShieldAgent reduces API queries by 64.7% and inference time by 58.2%, demonstrating its high precision and efficiency in safeguarding agents.

**arXiv ID:** 2503.22738
</details>

<details>
<summary><strong>Seeing before Observable: Potential Risk Reasoning in Autonomous Driving via Vision Language Models</strong> - Jiaxin Liu, Xiangyu Yan, Liang Peng, Lei Yang, Lingjun Zhang, Yuechen Luo, Yueming Tao, Ashton Yu Xuan Tan, Mu Li, Lei Zhang, Ziqi Zhan, Sai Guo, Hong Wang, Jun Li - [[pdf]](https://arxiv.org/pdf/2511.22928)</summary>

**Abstract:** Ensuring safety remains a key challenge for autonomous vehicles (AVs), especially in rare and complex scenarios. One critical but understudied aspect is the \textbf{potential risk} situations, where the risk is \textbf{not yet observable} but can be inferred from subtle precursors, such as anomalous behaviors or commonsense violations. Recognizing these precursors requires strong semantic understanding and reasoning capabilities, which are often absent in current AV systems due to the scarcity of such cases in existing driving or risk-centric datasets. Moreover, current autonomous driving accident datasets often lack annotations of the causal reasoning chains behind incidents, which are essential for identifying potential risks before they become observable. To address these gaps, we introduce PotentialRiskQA, a novel vision-language dataset designed for reasoning about potential risks prior to observation. Each sample is annotated with structured scene descriptions, semantic precursors, and inferred risk outcomes. Based on this dataset, we further propose PR-Reasoner, a vision-language-model-based framework tailored for onboard potential risk reasoning. Experimental results show that fine-tuning on PotentialRiskQA enables PR-Reasoner to significantly enhance its performance on the potential risk reasoning task compared to baseline VLMs. Together, our dataset and model provide a foundation for developing autonomous systems with improved foresight and proactive safety capabilities, moving toward more intelligent and resilient AVs.

**arXiv ID:** 2511.22928
</details>

<details>
<summary><strong>Investigating AI in Peer Support via Multi-Module System-Driven Embodied Conversational Agents</strong> - Ruoyu Wen, Xiaoli Wu, Kunal Gupta, Simon Hoermann, Mark Billinghurst, Alaeddin Nassani, Dwain Allan, Thammathip Piumsomboon - [[pdf]](https://arxiv.org/pdf/2511.22269)</summary>

**Abstract:** Young people's mental well-being is a global concern, with peer support playing a key role in daily emotional regulation. Conversational agents are increasingly viewed as promising tools for delivering accessible, personalised peer support, particularly where professional counselling is limited. However, existing systems often suffer from rigid input formats, scripted responses, and limited emotional sensitivity. The emergence of large language models introduces new possibilities for generating flexible, context-aware, and empathetic responses. To explore how individuals with psychological training perceive such systems in peer support contexts, we developed an LLM-based multi-module system to drive embodied conversational agents informed by Cognitive Behavioral Therapy (CBT). In a user study (N=10), we qualitatively examined participants' perceptions, focusing on trust, response quality, workflow integration, and design opportunities for future mental well-being support systems.

**arXiv ID:** 2511.22269
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (20 papers)</h2></summary>

<details>
<summary><strong>Real-Time Procedural Learning From Experience for AI Agents</strong> - Dasheng Bi, Yubin Hu, Mohammed N. Nasir - [[pdf]](https://arxiv.org/pdf/2511.22074)</summary>

**Abstract:** Learning how to do things from trial and error in real time is a hallmark of biological intelligence, yet most LLM-based agents lack mechanisms to acquire procedural knowledge after deployment. We propose Procedural Recall for Agents with eXperiences Indexed by State (PRAXIS), a lightweight post-training learning mechanism that stores the consequences of actions and retrieves them by jointly matching environmental and internal states of past episodes to the current state. PRAXIS augments agentic action selection with retrieved state-action-result exemplars that are generated in real time. When evaluated on the REAL web browsing benchmark, PRAXIS improves task completion accuracy, reliability, and cost efficiency across different foundation model backbones, and shows preliminary generalization to unseen tasks in similar environments. These results demonstrate that PRAXIS enables the practical adoption of AI agents in fast-evolving stateful environments by helping them learn new procedures effectively.

**arXiv ID:** 2511.22074
</details>

<details>
<summary><strong>Co-Evolving Agents: Learning from Failures as Hard Negatives</strong> - Yeonsung Jung, Trilok Padhi, Sina Shaham, Dipika Khullar, Joonhyun Jeong, Ninareh Mehrabi, Eunho Yang - [[pdf]](https://arxiv.org/pdf/2511.22254)</summary>

**Abstract:** The rapid progress of large foundation models has accelerated the development of task-specialized agents across diverse domains. However, the effectiveness of agents remains tightly coupled with the quality of training data, while curating task-specific datasets remains costly and often infeasible in real-world scenarios. Recent work has explored self-improving agents that autonomously generate, refine, and re-train on their own trajectories. A prominent line of approaches further leverages preference optimization by pairing predicted trajectories with scarce ground-truth trajectories, enabling agents to learn directly from their own failures. While these methods outperform supervised fine-tuning, their heavy reliance on predicted trajectories under limited ground-truth supervision leaves them prone to overfitting. To address this, we propose a co-evolving agents framework in which a target agent improves jointly with an auxiliary failure agent. The failure agent learns through preference optimization over failure trajectories from both the target and itself, thereby generating hard negatives that are close to success yet remain failures. Incorporating these informative hard negatives into the target agent's optimization sharpens decision boundaries and enhances generalization. Our comprehensive analysis and experiments across benchmark datasets show that our method not only shows improved performance but also demonstrates that failures, instead of being used as-is, can be systematically transformed into structured and valuable learning signals in self-improving agents.

**arXiv ID:** 2511.22254
</details>

<details>
<summary><strong>Geometrically-Constrained Agent for Spatial Reasoning</strong> - Zeren Chen, Xiaoya Lu, Zhijie Zheng, Pengrui Li, Lehan He, Yijin Zhou, Jing Shao, Bohan Zhuang, Lu Sheng - [[pdf]](https://arxiv.org/pdf/2511.22659)</summary>

**Abstract:** Vision Language Models (VLMs) exhibit a fundamental semantic-to-geometric gap in spatial reasoning: they excel at qualitative semantic inference but their reasoning operates within a lossy semantic space, misaligned with high-fidelity geometry. Current paradigms fail to bridge this gap. Training-based methods suffer from an ``oracle paradox,'' learning flawed spatial logic from imperfect oracles. Tool-integrated methods constrain the final computation but critically leave the VLM's planning process unconstrained, resulting in geometrically flawed plans. In this work, we propose Geometrically-Constrained Agent (GCA), a training-free agentic paradigm that resolves this gap by introducing a formal task constraint. Specifically, we strategically decouples the VLM's role into two stages. First, acting as a semantic analyst, the VLM translates the user's ambiguous query into the formal, verifiable task constraint, which defines the reference frame and objective. Second, acting as a task solver, the VLM generates and executes tool calls strictly within the deterministic bounds defined by the constraint. This geometrically-constrained reasoning strategy successfully resolve the semantic-to-geometric gap, yielding a robust and verifiable reasoning pathway for spatial reasoning. Comprehensive experiments demonstrate that GCA achieves SOTA performance on multiple spatial reasoning benchmarks, surpassing existing training-based and tool-integrated methods by ~27%. Please see our homepage at this https URL.

**arXiv ID:** 2511.22659
</details>

<details>
<summary><strong>MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents</strong> - Ruoxuan Zhang, Qiyun Zheng, Zhiyu Zhou, Ziqi Liao, Siyu Wu, Jian-Yu Jiang-Lin, Bin Wen, Hongxia Xie, Jianlong Fu, Wen-Huang Cheng - [[pdf]](https://arxiv.org/pdf/2511.23055)</summary>

**Abstract:** Theory of Mind (ToM) refers to the ability to infer others' mental states, such as beliefs, desires, and intentions. Current vision-language embodied agents lack ToM-based decision-making, and existing benchmarks focus solely on human mental states while ignoring the agent's own perspective, hindering coherent decision and action generation. To address this, we propose MindPower, a Robot-Centric framework integrating Perception, Mental Reasoning, Decision Making and Action. Given multimodal inputs, MindPower first perceives the environment and human states, then performs ToM Reasoning to model both self and others, and finally generates decisions and actions guided by inferred mental states. Furthermore, we introduce Mind-Reward, a novel optimization objective that encourages VLMs to produce consistent ToM Reasoning and behavior. Our model outperforms GPT-4o by 12.77% in decision making and 12.49% in action generation.

**arXiv ID:** 2511.23055
</details>

<details>
<summary><strong>Hierarchical AI-Meteorologist: LLM-Agent System for Multi-Scale and Explainable Weather Forecast Reporting</strong> - Daniil Sukhorukov, Andrei Zakharov, Nikita Glazkov, Katsiaryna Yanchanka, Vladimir Kirilin, Maxim Dubovitsky, Roman Sultimov, Yuri Maksimov, Ilya Makarov - [[pdf]](https://arxiv.org/pdf/2511.23387)</summary>

**Abstract:** We present the Hierarchical AI-Meteorologist, an LLM-agent system that generates explainable weather reports using a hierarchical forecast reasoning and weather keyword generation. Unlike standard approaches that treat forecasts as flat time series, our framework performs multi-scale reasoning across hourly, 6-hour, and daily aggregations to capture both short-term dynamics and long-term trends. Its core reasoning agent converts structured meteorological inputs into coherent narratives while simultaneously extracting a few keywords effectively summarizing the dominant meteorological events. These keywords serve as semantic anchors for validating consistency, temporal coherence and factual alignment of the generated reports. Using OpenWeather and Meteostat data, we demonstrate that hierarchical context and keyword-based validation substantially improve interpretability and robustness of LLM-generated weather narratives, offering a reproducible framework for semantic evaluation of automated meteorological reporting and advancing agent-based scientific reasoning.

**arXiv ID:** 2511.23387
</details>

<details>
<summary><strong>Towards Continuous Intelligence Growth: Self-Training, Continual Learning, and Dual-Scale Memory in SuperIntelliAgent</strong> - Jianzhe Lin, Zeyu Pan, Yun Zhu, Ruiqi Song, Jining Yang - [[pdf]](https://arxiv.org/pdf/2511.23436)</summary>

**Abstract:** We introduce SuperIntelliAgent, an agentic learning framework that couples a trainable small diffusion model (the learner) with a frozen large language model (the verifier) to enable continual intelligence growth through self-supervised interaction. Unlike conventional supervised fine-tuning, SuperIntelliAgent learns autonomously without annotation: the learner generates candidate outputs, the verifier evaluates them through step-by-step reasoning, and their interaction produces chosen/rejected pairs for Direct Preference Optimization (DPO). This converts each input into a pseudo-training signal for continual improvement. The framework integrates dual-scale memory: short-term in-context memory that preserves reasoning traces across refinement cycles, and long-term memory that consolidates acquired knowledge through lightweight on-the-fly fine-tuning. A replay buffer retains samples that show verifiable progress and replays them as auxiliary supervision, reinforcing recent learning while forming adaptive curricula. SuperIntelliAgent is infrastructure-agnostic and can be plugged into existing agentic frameworks while turning ordinary inference loops into a lifelong optimization process. We posit that pairing a trainable learner with a reasoning-capable verifier forms a minimal reliable unit of growing intelligence, as paired feedback and partial-history replay yield richer learning curricula and stronger preference alignment. With a small number of automatically generated DPO pairs, the learner improves across all benchmarks, indicating that this mechanism provides a promising direction for continual intelligence accumulation and real-world deployment.

**arXiv ID:** 2511.23436
</details>

<details>
<summary><strong>A Benchmark for Procedural Memory Retrieval in Language Agents</strong> - Ishant Kohar, Aswanth Krishnan - [[pdf]](https://arxiv.org/pdf/2511.21730)</summary>

**Abstract:** Current AI agents excel in familiar settings, but fail sharply when faced with novel tasks with unseen vocabularies -- a core limitation of procedural memory systems. We present the first benchmark that isolates procedural memory retrieval from task execution, evaluating whether agents can recognize functionally equivalent procedures that span different object instantiations. Using ALFWorld, we construct dual corpora of expert and LLM-generated trajectories and evaluate six retrieval methods using systematically stratified queries. Our results expose a clear generalization cliff: embedding-based methods perform strongly on familiar contexts, yet degrade considerably on novel ones, while LLM-generated procedural abstractions demonstrate reliable cross-context transfer. Controlled ablations show that although embeddings capture some lexical-level abstraction, they fundamentally treat procedures as unordered bags of words, discarding temporal structure necessary for cross-context transfer. Corpus scale delivers far larger gains than representation enrichment, revealing an architectural ceiling in current encoders. Our benchmark offers the first diagnostic framework separating genuine procedural understanding from surface-level memorization and gives tools for developing retrieval systems capable of dependable generalization. Resources available at our GitHub repository (this https URL).

**arXiv ID:** 2511.21730
</details>

<details>
<summary><strong>A Safety and Security Framework for Real-World Agentic Systems</strong> - Shaona Ghosh, Barnaby Simkin, Kyriacos Shiarlis, Soumili Nandi, Dan Zhao, Matthew Fiedler, Julia Bazinska, Nikki Pope, Roopa Prabhu, Daniel Rohrer, Michael Demoret, Bartley Richardson - [[pdf]](https://arxiv.org/pdf/2511.21990)</summary>

**Abstract:** This paper introduces a dynamic and actionable framework for securing agentic AI systems in enterprise deployment. We contend that safety and security are not merely fixed attributes of individual models but also emergent properties arising from the dynamic interactions among models, orchestrators, tools, and data within their operating environments. We propose a new way of identification of novel agentic risks through the lens of user safety. Although, for traditional LLMs and agentic models in isolation, safety and security has a clear separation, through the lens of safety in agentic systems, they appear to be connected. Building on this foundation, we define an operational agentic risk taxonomy that unifies traditional safety and security concerns with novel, uniquely agentic risks, including tool misuse, cascading action chains, and unintended control amplification among others. At the core of our approach is a dynamic agentic safety and security framework that operationalizes contextual agentic risk management by using auxiliary AI models and agents, with human oversight, to assist in contextual risk discovery, evaluation, and mitigation. We further address one of the most challenging aspects of safety and security of agentic systems: risk discovery through sandboxed, AI-driven red teaming. We demonstrate the framework effectiveness through a detailed case study of NVIDIA flagship agentic research assistant, AI-Q Research Assistant, showcasing practical, end-to-end safety and security evaluations in complex, enterprise-grade agentic workflows. This risk discovery phase finds novel agentic risks that are then contextually mitigated. We also release the dataset from our case study, containing traces of over 10,000 realistic attack and defense executions of the agentic workflow to help advance research in agentic safety.

**arXiv ID:** 2511.21990
</details>

<details>
<summary><strong>CoT4AD: A Vision-Language-Action Model with Explicit Chain-of-Thought Reasoning for Autonomous Driving</strong> - Zhaohui Wang, Tengbo Yu, Hao Tang - [[pdf]](https://arxiv.org/pdf/2511.22532)</summary>

**Abstract:** Vision-Language-Action (VLA) models have recently attracted growing attention in end-to-end autonomous driving for their strong reasoning capabilities and rich world knowledge. However, existing VLAs often suffer from limited numerical reasoning ability and overly simplified input-output mappings, which hinder their performance in complex driving scenarios requiring step-by-step causal reasoning. To address these challenges, we propose CoT4AD, a novel VLA framework that introduces Chain-of-Thought (CoT) reasoning for autonomous driving to enhance both numerical and causal reasoning in Vision-Language Models (VLMs). CoT4AD integrates visual observations and language instructions to perform semantic reasoning, scene understanding, and trajectory planning. During training, it explicitly models a perception-question-prediction-action CoT to align the reasoning space with the action space across multiple driving tasks. During inference, it performs implicit CoT reasoning to enable consistent numerical reasoning and robust decision-making in dynamic environments. Extensive experiments on both real-world and simulated benchmarks, including nuScenes and Bench2Drive, demonstrate that CoT4AD achieves state-of-the-art performance in both open-loop and closed-loop evaluations. Code will be released upon paper acceptance.

**arXiv ID:** 2511.22532
</details>

<details>
<summary><strong>Exact Learning of Arithmetic with Differentiable Agents</strong> - Hristo Papazov, Francesco D'Angelo, Nicolas Flammarion - [[pdf]](https://arxiv.org/pdf/2511.22751)</summary>

**Abstract:** We explore the possibility of exact algorithmic learning with gradient-based methods and introduce a differentiable framework capable of strong length generalization on arithmetic tasks. Our approach centers on Differentiable Finite-State Transducers (DFSTs), a Turing-complete model family that avoids the pitfalls of prior architectures by enabling constant-precision, constant-time generation, and end-to-end log-parallel differentiable training. Leveraging policy-trajectory observations from expert agents, we train DFSTs to perform binary and decimal addition and multiplication. Remarkably, models trained on tiny datasets generalize without error to inputs thousands of times longer than the training examples. These results show that training differentiable agents on structured intermediate supervision could pave the way towards exact gradient-based learning of algorithmic skills. Code available at \href{this https URL}{this https URL}.

**arXiv ID:** 2511.22751
</details>

<details>
<summary><strong>Beyond Curve Fitting: Neuro-Symbolic Agents for Context-Aware Epidemic Forecasting</strong> - Joongwon Chae, Runming Wang, Chen Xiong, Gong Yunhan, Lian Zhang, Ji Jiansong, Dongmei Yu, Peiwu Qin - [[pdf]](https://arxiv.org/pdf/2511.23276)</summary>

**Abstract:** Effective surveillance of hand, foot and mouth disease (HFMD) requires forecasts accounting for epidemiological patterns and contextual drivers like school calendars and weather. While classical models and recent foundation models (e.g., Chronos, TimesFM) incorporate covariates, they often lack the semantic reasoning to interpret the causal interplay between conflicting drivers. In this work, we propose a two-agent framework decoupling contextual interpretation from probabilistic forecasting. An LLM "event interpreter" processes heterogeneous signals-including school schedules, meteorological summaries, and reports-into a scalar transmission-impact signal. A neuro-symbolic core then combines this with historical case counts to produce calibrated probabilistic forecasts. We evaluate the framework on real-world HFMD datasets from Hong Kong (2023-2024) and Lishui, China (2024). Compared to traditional and foundation-model baselines, our approach achieves competitive point forecasting accuracy while providing robust 90% prediction intervals (coverage 0.85-1.00) and human-interpretable rationales. Our results suggest that structurally integrating domain knowledge through LLMs can match state-of-the-art performance while yielding context-aware forecasts that align with public health workflows. Code is available at this https URL .

**arXiv ID:** 2511.23276
</details>

<details>
<summary><strong>Are LLMs Good Safety Agents or a Propaganda Engine?</strong> - Neemesh Yadav, Francesco Ortu, Jiarui Liu, Joeun Yook, Bernhard Schölkopf, Rada Mihalcea, Alberto Cazzaniga, Zhijing Jin - [[pdf]](https://arxiv.org/pdf/2511.23174)</summary>

**Abstract:** Large Language Models (LLMs) are trained to refuse to respond to harmful content. However, systematic analyses of whether this behavior is truly a reflection of its safety policies or an indication of political censorship, that is practiced globally by countries, is lacking. Differentiating between safety influenced refusals or politically motivated censorship is hard and unclear. For this purpose we introduce PSP, a dataset built specifically to probe the refusal behaviors in LLMs from an explicitly political context. PSP is built by formatting existing censored content from two data sources, openly available on the internet: sensitive prompts in China generalized to multiple countries, and tweets that have been censored in various countries. We study: 1) impact of political sensitivity in seven LLMs through data-driven (making PSP implicit) and representation-level approaches (erasing the concept of politics); and, 2) vulnerability of models on PSP through prompt injection attacks (PIAs). Associating censorship with refusals on content with masked implicit intent, we find that most LLMs perform some form of censorship. We conclude with summarizing major attributes that can cause a shift in refusal distributions across models and contexts of different countries.

**arXiv ID:** 2511.23174
</details>

<details>
<summary><strong>Intelligent Neural Networks: From Layered Architectures to Graph-Organized Intelligence</strong> - Antoine Salomon - [[pdf]](https://arxiv.org/pdf/2511.22813)</summary>

**Abstract:** Biological neurons exhibit remarkable intelligence: they maintain internal states, communicate selectively with other neurons, and self-organize into complex graphs rather than rigid hierarchical layers. What if artificial intelligence could emerge from similarly intelligent computational units? We introduce Intelligent Neural Networks (INN), a paradigm shift where neurons are first-class entities with internal memory and learned communication patterns, organized in complete graphs rather than sequential layers.
Each Intelligent Neuron combines selective state-space dynamics (knowing when to activate) with attention-based routing (knowing to whom to send signals), enabling emergent computation through graph-structured interactions. On the standard Text8 character modeling benchmark, INN achieves 1.705 Bit-Per-Character (BPC), significantly outperforming a comparable Transformer (2.055 BPC) and matching a highly optimized LSTM baseline. Crucially, a parameter-matched baseline of stacked Mamba blocks fails to converge (>3.4 BPC) under the same training protocol, demonstrating that INN's graph topology provides essential training stability. Ablation studies confirm this: removing inter-neuron communication degrades performance or leads to instability, proving the value of learned neural routing.
This work demonstrates that neuron-centric design with graph organization is not merely bio-inspired -- it is computationally effective, opening new directions for modular, interpretable, and scalable neural architectures.

**arXiv ID:** 2511.22813
</details>

<details>
<summary><strong>ROVER: Recursive Reasoning Over Videos with Vision-Language Models for Embodied Tasks</strong> - Philip Schroeder, Ondrej Biza, Thomas Weng, Hongyin Luo, James Glass - [[pdf]](https://arxiv.org/pdf/2508.01943)</summary>

**Abstract:** Vision-language models (VLMs) have exhibited impressive capabilities across diverse image understanding tasks, but still struggle in settings that require reasoning over extended sequences of camera frames from a video. This limits their utility in embodied settings, which require reasoning over long frame sequences from a continuous stream of visual input at each moment of a task attempt. To address this limitation, we propose ROVER (Reasoning Over VidEo Recursively), a framework that enables the model to recursively decompose long-horizon video trajectories into segments corresponding to shorter subtasks within the trajectory. In doing so, ROVER facilitates more focused and accurate reasoning over temporally localized frame sequences without losing global context. We evaluate ROVER, implemented using an in-context learning approach, on diverse OpenX Embodiment videos and on a new dataset derived from RoboCasa that consists of 543 videos showing both expert and perturbed non-expert trajectories across 27 robotic manipulation tasks. ROVER outperforms strong baselines across three video reasoning tasks: task progress estimation, frame-level natural language reasoning, and video question answering. We observe that, by reducing the number of frames the model reasons over at each timestep, ROVER mitigates hallucinations, especially during unexpected or non-optimal moments of a trajectory. In addition, by enabling the implementation of a subtask-specific sliding context window, ROVER's time complexity scales linearly with video length, an asymptotic improvement over baselines. Demos, code, and data available at: this https URL

**arXiv ID:** 2508.01943
</details>

<details>
<summary><strong>Agentar-Scale-SQL: Advancing Text-to-SQL through Orchestrated Test-Time Scaling</strong> - Pengfei Wang, Baolin Sun, Xuemei Dong, Yaxun Dai, Hongwei Yuan, Mengdie Chu, Yingqi Gao, Xiang Qi, Peng Zhang, Ying Yan - [[pdf]](https://arxiv.org/pdf/2509.24403)</summary>

**Abstract:** State-of-the-art (SOTA) Text-to-SQL methods still lag significantly behind human experts on challenging benchmarks like BIRD. Current approaches that explore test-time scaling lack an orchestrated strategy and neglect the model's internal reasoning process. To bridge this gap, we introduce Agentar-Scale-SQL, a novel framework leveraging scalable computation to improve performance. Agentar-Scale-SQL implements an Orchestrated Test-Time Scaling strategy that synergistically combines three distinct perspectives: i) Internal Scaling via RL-enhanced Intrinsic Reasoning, ii) Sequential Scaling through Iterative Refinement, and iii) Parallel Scaling using Diverse Synthesis and Tournament Selection. Agentar-Scale-SQL is a general-purpose framework designed for easy adaptation to new databases and more powerful language models. Extensive experiments show that Agentar-Scale-SQL achieves SOTA performance on the BIRD benchmark, reaching 81.67% execution accuracy on the test set and ranking first on the official leaderboard, demonstrating an effective path toward human-level performance.

**arXiv ID:** 2509.24403
</details>

<details>
<summary><strong>Autonomous labeling of surgical resection margins using a foundation model</strong> - Xilin Yang, Musa Aydin, Yuhong Lu, Sahan Yoruc Selcuk, Bijie Bai, Yijie Zhang, Andrew Birkeland, Katjana Ehrlich, Julien Bec, Laura Marcu, Nir Pillar, Aydogan Ozcan - [[pdf]](https://arxiv.org/pdf/2511.22131)</summary>

**Abstract:** Assessing resection margins is central to pathological specimen evaluation and has profound implications for patient outcomes. Current practice employs physical inking, which is applied variably, and cautery artifacts can obscure the true margin on histological sections. We present a virtual inking network (VIN) that autonomously localizes the surgical cut surface on whole-slide images, reducing reliance on inks and standardizing margin-focused review. VIN uses a frozen foundation model as the feature extractor and a compact two-layer multilayer perceptron trained for patch-level classification of cautery-consistent features. The dataset comprised 120 hematoxylin and eosin (H&E) stained slides from 12 human tonsil tissue blocks, resulting in ~2 TB of uncompressed raw image data, where a board-certified pathologist provided boundary annotations. In blind testing with 20 slides from previously unseen blocks, VIN produced coherent margin overlays that qualitatively aligned with expert annotations across serial sections. Quantitatively, region-level accuracy was ~73.3% across the test set, with errors largely confined to limited areas that did not disrupt continuity of the whole-slide margin map. These results indicate that VIN captures cautery-related histomorphology and can provide a reproducible, ink-free margin delineation suitable for integration into routine digital pathology workflows and for downstream measurement of margin distances.

**arXiv ID:** 2511.22131
</details>

<details>
<summary><strong>SUPER-AD: Semantic Uncertainty-aware Planning for End-to-End Robust Autonomous Driving</strong> - Wonjeong Ryu, Seungjun Yu, Seokha Moon, Hojun Choi, Junsung Park, Jinkyu Kim, Hyunjung Shim - [[pdf]](https://arxiv.org/pdf/2511.22865)</summary>

**Abstract:** End-to-End (E2E) planning has become a powerful paradigm for autonomous driving, yet current systems remain fundamentally uncertainty-blind. They assume perception outputs are fully reliable, even in ambiguous or poorly observed scenes, leaving the planner without an explicit measure of uncertainty. To address this limitation, we propose a camera-only E2E framework that estimates aleatoric uncertainty directly in BEV space and incorporates it into planning. Our method produces a dense, uncertainty-aware drivability map that captures both semantic structure and geometric layout at pixel-level resolution. To further promote safe and rule-compliant behavior, we introduce a lane-following regularization that encodes lane structure and traffic norms. This prior stabilizes trajectory planning under normal conditions while preserving the flexibility needed for maneuvers such as overtaking or lane changes. Together, these components enable robust and interpretable trajectory planning, even under challenging uncertainty conditions. Evaluated on the NAVSIM benchmark, our method achieves state-of-the-art performance, delivering substantial gains on both the challenging NAVHARD and NAVSAFE subsets. These results demonstrate that our principled aleatoric uncertainty modeling combined with driving priors significantly advances the safety and reliability of camera-only E2E autonomous driving.

**arXiv ID:** 2511.22865
</details>

<details>
<summary><strong>Percept-WAM: Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving</strong> - Jianhua Han, Meng Tian, Jiangtong Zhu, Fan He, Huixin Zhang, Sitong Guo, Dechang Zhu, Hao Tang, Pei Xu, Yuze Guo, Minzhe Niu, Haojie Zhu, Qichao Dong, Xuechao Yan, Siyuan Dong, Lu Hou, Qingqiu Huang, Xiaosong Jia, Hang Xu - [[pdf]](https://arxiv.org/pdf/2511.19221)</summary>

**Abstract:** Autonomous driving heavily relies on accurate and robust spatial perception. Many failures arise from inaccuracies and instability, especially in long-tail scenarios and complex interactions. However, current vision-language models are weak at spatial grounding and understanding, and VLA systems built on them therefore show limited perception and localization ability. To address these challenges, we introduce Percept-WAM, a perception-enhanced World-Awareness-Action Model that is the first to implicitly integrate 2D/3D scene understanding abilities within a single vision-language model (VLM). Instead of relying on QA-style spatial reasoning, Percept-WAM unifies 2D/3D perception tasks into World-PV and World-BEV tokens, which encode both spatial coordinates and confidence. We propose a grid-conditioned prediction mechanism for dense object perception, incorporating IoU-aware scoring and parallel autoregressive decoding, improving stability in long-tail, far-range, and small-object scenarios. Additionally, Percept-WAM leverages pretrained VLM parameters to retain general intelligence (e.g., logical reasoning) and can output perception results and trajectory control outputs directly. Experiments show that Percept-WAM matches or surpasses classical detectors and segmenters on downstream perception benchmarks, achieving 51.7/58.9 mAP on COCO 2D detection and nuScenes BEV 3D detection. When integrated with trajectory decoders, it further improves planning performance on nuScenes and NAVSIM, e.g., surpassing DiffusionDrive by 2.1 in PMDS on NAVSIM. Qualitative results further highlight its strong open-vocabulary and long-tail generalization.

**arXiv ID:** 2511.19221
</details>

<details>
<summary><strong>DualVLA: Building a Generalizable Embodied Agent via Partial Decoupling of Reasoning and Action</strong> - Zhen Fang, Zhuoyang Liu, Jiaming Liu, Hao Chen, Yu Zeng, Shiting Huang, Zehui Chen, Lin Chen, Shanghang Zhang, Feng Zhao - [[pdf]](https://arxiv.org/pdf/2511.22134)</summary>

**Abstract:** To build a generalizable Vision-Language-Action (VLA) model with strong reasoning ability, a common strategy is to first train a specialist VLA on robot demonstrations to acquire reliable manipulation skills, and then incorporate mixed annotated robot data together with multimodal data to restore broader reasoning capabilities. However, we observe that the resulting reasoning VLA often suffers from degraded action performance compared to the specialist model before fine-tuning, a phenomenon we refer to as action degeneration. To address this issue, we propose DualVLA, which enhances action performance through carefully designed post-training while still preserving reasoning capability. We first introduce a dual-layer data pruning method that removes redundant embodied reasoning, preventing it from adversely influencing action learning. To further strengthen action generation, we design a dual-teacher adaptive distillation strategy that assigns different supervision signals to different data domains while maintaining reasoning ability. To fill the evaluation gap for generalist VLAs, we also propose VLA Score, which decouples VLA capability into reasoning, intention, action, and alignment dimensions for a more fine-grained assessment. Experiments show that DualVLA achieves an average success rate of 61.0 in SimplerEnv and an average score of 65.4 across eight competitive multimodal benchmarks, demonstrating a stronger balance between precise action execution and multimodal understanding. Project Website: this https URL.

**arXiv ID:** 2511.22134
</details>

<details>
<summary><strong>Foundation Models in Autonomous Driving: A Survey on Scenario Generation and Scenario Analysis</strong> - Yuan Gao, Mattia Piccinini, Yuchen Zhang, Dingrui Wang, Korbinian Moller, Roberto Brusnicki, Baha Zarrouki, Alessio Gambi, Jan Frederik Totz, Kai Storms, Steven Peters, Andrea Stocco, Bassam Alrifaee, Marco Pavone, Johannes Betz - [[pdf]](https://arxiv.org/pdf/2506.11526)</summary>

**Abstract:** For autonomous vehicles, safe navigation in complex environments depends on handling a broad range of diverse and rare driving scenarios. Simulation- and scenario-based testing have emerged as key approaches to development and validation of autonomous driving systems. Traditional scenario generation relies on rule-based systems, knowledge-driven models, and data-driven synthesis, often producing limited diversity and unrealistic safety-critical cases. With the emergence of foundation models, which represent a new generation of pre-trained, general-purpose AI models, developers can process heterogeneous inputs (e.g., natural language, sensor data, HD maps, and control actions), enabling the synthesis and interpretation of complex driving scenarios. In this paper, we conduct a survey about the application of foundation models for scenario generation and scenario analysis in autonomous driving (as of May 2025). Our survey presents a unified taxonomy that includes large language models, vision-language models, multimodal large language models, diffusion models, and world models for the generation and analysis of autonomous driving scenarios. In addition, we review the methodologies, open-source datasets, simulation platforms, and benchmark challenges, and we examine the evaluation metrics tailored explicitly to scenario generation and analysis. Finally, the survey concludes by highlighting the open challenges and research questions, and outlining promising future research directions. All reviewed papers are listed in a continuously maintained repository, which contains supplementary materials and is available at this https URL.

**arXiv ID:** 2506.11526
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>Swarms of Large Language Model Agents for Protein Sequence Design with Experimental Validation</strong> - Fiona Y. Wang, Di Sheng Lee, David L. Kaplan, Markus J. Buehler - [[pdf]](https://arxiv.org/pdf/2511.22311)</summary>

**Abstract:** Designing proteins de novo with tailored structural, physicochemical, and functional properties remains a grand challenge in biotechnology, medicine, and materials science, due to the vastness of sequence space and the complex coupling between sequence, structure, and function. Current state-of-the-art generative methods, such as protein language models (PLMs) and diffusion-based architectures, often require extensive fine-tuning, task-specific data, or model reconfiguration to support objective-directed design, thereby limiting their flexibility and scalability. To overcome these limitations, we present a decentralized, agent-based framework inspired by swarm intelligence for de novo protein design. In this approach, multiple large language model (LLM) agents operate in parallel, each assigned to a specific residue position. These agents iteratively propose context-aware mutations by integrating design objectives, local neighborhood interactions, and memory and feedback from previous iterations. This position-wise, decentralized coordination enables emergent design of diverse, well-defined sequences without reliance on motif scaffolds or multiple sequence alignments, validated with experiments on proteins with alpha helix and coil structures. Through analyses of residue conservation, structure-based metrics, and sequence convergence and embeddings, we demonstrate that the framework exhibits emergent behaviors and effective navigation of the protein fitness landscape. Our method achieves efficient, objective-directed designs within a few GPU-hours and operates entirely without fine-tuning or specialized training, offering a generalizable and adaptable solution for protein design. Beyond proteins, the approach lays the groundwork for collective LLM-driven design across biomolecular systems and other scientific discovery tasks.

**arXiv ID:** 2511.22311
</details>

<details>
<summary><strong>GEO-Detective: Unveiling Location Privacy Risks in Images with LLM Agents</strong> - Xinyu Zhang, Yixin Wu, Boyang Zhang, Chenhao Lin, Chao Shen, Michael Backes, Yang Zhang - [[pdf]](https://arxiv.org/pdf/2511.22441)</summary>

**Abstract:** Images shared on social media often expose geographic cues. While early geolocation methods required expert effort and lacked generalization, the rise of Large Vision Language Models (LVLMs) now enables accurate geolocation even for ordinary users. However, existing approaches are not optimized for this task. To explore the full potential and associated privacy risks, we present Geo-Detective, an agent that mimics human reasoning and tool use for image geolocation inference. It follows a procedure with four steps that adaptively selects strategies based on image difficulty and is equipped with specialized tools such as visual reverse search, which emulates how humans gather external geographic clues. Experimental results show that GEO-Detective outperforms baseline large vision language models (LVLMs) overall, particularly on images lacking visible geographic features. In country level geolocation tasks, it achieves an improvement of over 11.1% compared to baseline LLMs, and even at finer grained levels, it still provides around a 5.2% performance gain. Meanwhile, when equipped with external clues, GEO-Detective becomes more likely to produce accurate predictions, reducing the "unknown" prediction rate by more than 50.6%. We further explore multiple defense strategies and find that Geo-Detective exhibits stronger robustness, highlighting the need for more effective privacy safeguards.

**arXiv ID:** 2511.22441
</details>

<details>
<summary><strong>MCP vs RAG vs NLWeb vs HTML: A Comparison of the Effectiveness and Efficiency of Different Agent Interfaces to the Web (Technical Report)</strong> - Aaron Steiner, Ralph Peeters, Christian Bizer - [[pdf]](https://arxiv.org/pdf/2511.23281)</summary>

**Abstract:** Large language model agents are increasingly used to automate web tasks such as product search, offer comparison, and checkout. Current research explores different interfaces through which these agents interact with websites, including traditional HTML browsing, retrieval-augmented generation (RAG) over pre-crawled content, communication via Web APIs using the Model Context Protocol (MCP), and natural-language querying through the NLWeb interface. However, no prior work has compared these four architectures within a single controlled environment using identical tasks.
To address this gap, we introduce a testbed consisting of four simulated e-shops, each offering its products via HTML, MCP, and NLWeb interfaces. For each interface (HTML, RAG, MCP, and NLWeb) we develop specialized agents that perform the same sets of tasks, ranging from simple product searches and price comparisons to complex queries for complementary or substitute products and checkout processes. We evaluate the agents using GPT 4.1, GPT 5, GPT 5 mini, and Claude Sonnet 4 as underlying LLM. Our evaluation shows that the RAG, MCP and NLWeb agents outperform HTML on both effectiveness and efficiency. Averaged over all tasks, F1 rises from 0.67 for HTML to between 0.75 and 0.77 for the other agents. Token usage falls from about 241k for HTML to between 47k and 140k per task. The runtime per task drops from 291 seconds to between 50 and 62 seconds. The best overall configuration is RAG with GPT 5 achieving an F1 score of 0.87 and a completion rate of 0.79. Also taking cost into consideration, RAG with GPT 5 mini offers a good compromise between API usage fees and performance. Our experiments show the choice of the interaction interface has a substantial impact on both the effectiveness and efficiency of LLM-based web agents.

**arXiv ID:** 2511.23281
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (21 papers)</h2></summary>

<details>
<summary><strong>Embedded Universal Predictive Intelligence: a coherent framework for multi-agent learning</strong> - Alexander Meulemans, Rajai Nasser, Maciej Wołczyk, Marissa A. Weis, Seijin Kobayashi, Blake Richards, Guillaume Lajoie, Angelika Steger, Marcus Hutter, James Manyika, Rif A. Saurous, João Sacramento, Blaise Agüera y Arcas - [[pdf]](https://arxiv.org/pdf/2511.22226)</summary>

**Abstract:** The standard theory of model-free reinforcement learning assumes that the environment dynamics are stationary and that agents are decoupled from their environment, such that policies are treated as being separate from the world they inhabit. This leads to theoretical challenges in the multi-agent setting where the non-stationarity induced by the learning of other agents demands prospective learning based on prediction models. To accurately model other agents, an agent must account for the fact that those other agents are, in turn, forming beliefs about it to predict its future behavior, motivating agents to model themselves as part of the environment. Here, building upon foundational work on universal artificial intelligence (AIXI), we introduce a mathematical framework for prospective learning and embedded agency centered on self-prediction, where Bayesian RL agents predict both future perceptual inputs and their own actions, and must therefore resolve epistemic uncertainty about themselves as part of the universe they inhabit. We show that in multi-agent settings, self-prediction enables agents to reason about others running similar algorithms, leading to new game-theoretic solution concepts and novel forms of cooperation unattainable by classical decoupled agents. Moreover, we extend the theory of AIXI, and study universally intelligent embedded agents which start from a Solomonoff prior. We show that these idealized agents can form consistent mutual predictions and achieve infinite-order theory of mind, potentially setting a gold standard for embedded multi-agent learning.

**arXiv ID:** 2511.22226
</details>

<details>
<summary><strong>Training High-Level Schedulers with Execution-Feedback Reinforcement Learning for Long-Horizon GUI Automation</strong> - Zehao Deng, Tianjie Ju, Zheng Wu, Zhuosheng Zhang, Gongshen Liu - [[pdf]](https://arxiv.org/pdf/2511.22235)</summary>

**Abstract:** The rapid development of large vision-language model (VLM) has greatly promoted the research of GUI agent. However, GUI agents still face significant challenges in handling long-horizon tasks. First, single-agent models struggle to balance high-level capabilities and low-level execution capability, facing prevalent issues of responsibility coupling and capability conflicts. Second, agents lack awareness of the task state, leading to progress loss in long-horizon tasks. To address these challenges, we propose a staged execution-feedback reinforcement learning algorithm. Unlike training a unified policy model, we focus on training high-level scheduling models. Specifically, we propose and train two agents: a Coordinator, responsible for the strategic planning and task decomposition; and a State Tracker, responsible for context compression and information management to maintain the task's state and coherence. Based on this, we built the Coordinator-Executor-State Tracker (CES) multi-agent framework, which can be integrated with any low-level Executor model, assisting the Executor in solving long-horizon tasks through task scheduling and state management. Experiments on long-horizon task benchmarks demonstrate that CES significantly enhances the system's planning and state management capabilities. Furthermore, analysis confirms that our trained high-level scheduling module is a generalizable, plug-and-play module that significantly enhances the long-horizon capabilities of various Executors. Code can be available at this https URL.

**arXiv ID:** 2511.22235
</details>

<details>
<summary><strong>A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind</strong> - Fengming Zhu, Yuxin Pan, Xiaomeng Zhu, Fangzhen Lin - [[pdf]](https://arxiv.org/pdf/2511.22536)</summary>

**Abstract:** Originating in psychology, $\textit{Theory of Mind}$ (ToM) has attracted significant attention across multiple research communities, especially logic, economics, and robotics. Most psychological work does not aim at formalizing those central concepts, namely $\textit{goals}$, $\textit{intentions}$, and $\textit{beliefs}$, to automate a ToM-based computational process, which, by contrast, has been extensively studied by logicians. In this paper, we offer a different perspective by proposing a computational framework viewed through the lens of game theory. On the one hand, the framework prescribes how to make boudedly rational decisions while maintaining a theory of mind about others (and recursively, each of the others holding a theory of mind about the rest); on the other hand, it employs statistical techniques and approximate solutions to retain computability of the inherent computational problem.

**arXiv ID:** 2511.22536
</details>

<details>
<summary><strong>Agentic AI Framework for Individuals with Disabilities and Neurodivergence: A Multi-Agent System for Healthy Eating, Daily Routines, and Inclusive Well-Being</strong> - Salman Jan, Toqeer Ali Syed, Gohar Ali, Ali Akarma, Mohammad Riyaz Belgaum, Ahmad Ali - [[pdf]](https://arxiv.org/pdf/2511.22737)</summary>

**Abstract:** The paper presents a detailed Agentic Artificial Intelligence (AI) model that would enable people with disabilities and neurodivergence to lead healthier lives and have more regular days. The system will use a multi-layer structure; it will include an Application and Interface Layer, an Agents Layer, and a Data Source Layer to provide adaptive, transparent, and inclusive support. Fundamentally, a hybrid reasoning engine will synchronize four special-purpose agents, which include: a personalized-nutrition-based, called a Meal Planner Agent; an adaptive-scheduling-based, called a Reminder Agent; interactive assistance during grocery shopping and cooking, called a Food Guidance Agent; and a continuous-intake-and-physiological-tracking, called a Monitoring Agent. All the agents interact through a central communicative system called the Blackboard/Event Bus, which allows autonomous interaction and real-time feedback loops with multimedia user interfaces. Privacy-sensitive data sources, including electronic health records (EHRs), nutritional databases, wearable sensors, and smart kitchen Internet of Things, are also included in the framework and placed into a policy-controlled layer, which ensures data safety and compliance with consent. Collaborative care and clinician dashboards allow common supervision, and discussable artificial intelligence (XAI) modules give brief explanations of why a decision was made, making users responsible and reliant. The proposed agentic AI framework is an extension beyond traditional assistive systems since it incorporates inclusiveness, personalization, and accessibility at all levels. It displays the intersection of multi-agent reasoning, multi-modal interfaces, and human-centered design that will enable the development of autonomy, health, and digital equity among people with disabilities and neurodivergence.

**arXiv ID:** 2511.22737
</details>

<details>
<summary><strong>Agentic AI Framework for Cloudburst Prediction and Coordinated Response</strong> - Toqeer Ali Syed, Sohail Khan, Salman Jan, Gohar Ali, Muhammad Nauman, Ali Akarma, Ahmad Ali - [[pdf]](https://arxiv.org/pdf/2511.22767)</summary>

**Abstract:** The challenge is growing towards extreme and short-duration rainfall events like a cloudburst that are peculiar to the traditional forecasting systems, in which the predictions and the response are taken as two distinct processes. The paper outlines an agentic artificial intelligence system to study atmospheric water-cycle intelligence, which combines sensing, forecasting, downscaling, hydrological modeling and coordinated response into a single, interconnected, priceless, closed-loop system. The framework uses autonomous but cooperative agents that reason, sense, and act throughout the entire event lifecycle, and use the intelligence of weather prediction to become real-time decision intelligence. Comparison of multi-year radar, satellite, and ground-based evaluation of the northern part of Pakistan demonstrates that the multi-agent configuration enhances forecast reliability, critical success index and warning lead time compared to the baseline models. Population reach was maximised, and errors during evacuation were minimised through communication and routing agents, and adaptive recalibration and transparent auditability were provided by the embedded layer of learning. Collectively, this leads to the conclusion that collaborative AI agents are capable of transforming atmospheric data streams into practicable foresight and provide a platform of scalable adaptive and learning-based climate resilience.

**arXiv ID:** 2511.22767
</details>

<details>
<summary><strong>InsightEval: An Expert-Curated Benchmark for Assessing Insight Discovery in LLM-Driven Data Agents</strong> - Zhenghao Zhu, Yuanfeng Song, Xin Chen, Chengzhong Liu, Yakun Cui, Caleb Chen Cao, Sirui Han, Yike Guo - [[pdf]](https://arxiv.org/pdf/2511.22884)</summary>

**Abstract:** Data analysis has become an indispensable part of scientific research. To discover the latent knowledge and insights hidden within massive datasets, we need to perform deep exploratory analysis to realize their full value. With the advent of large language models (LLMs) and multi-agent systems, more and more researchers are making use of these technologies for insight discovery. However, there are few benchmarks for evaluating insight discovery capabilities. As one of the most comprehensive existing frameworks, InsightBench also suffers from many critical flaws: format inconsistencies, poorly conceived objectives, and redundant insights. These issues may significantly affect the quality of data and the evaluation of agents. To address these issues, we thoroughly investigate shortcomings in InsightBench and propose essential criteria for a high-quality insight benchmark. Regarding this, we develop a data-curation pipeline to construct a new dataset named InsightEval. We further introduce a novel metric to measure the exploratory performance of agents. Through extensive experiments on InsightEval, we highlight prevailing challenges in automated insight discovery and raise some key findings to guide future research in this promising direction.

**arXiv ID:** 2511.22884
</details>

<details>
<summary><strong>Peer-to-Peer Energy Trading in Dairy Farms using Multi-Agent Reinforcement Learning</strong> - Mian Ibad Ali Shah, Marcos Eduardo Cruz Victorio, Maeve Duffy, Enda Barrett, Karl Mason - [[pdf]](https://arxiv.org/pdf/2511.23148)</summary>

**Abstract:** The integration of renewable energy resources in rural areas, such as dairy farming communities, enables decentralized energy management through Peer-to-Peer (P2P) energy trading. This research highlights the role of P2P trading in efficient energy distribution and its synergy with advanced optimization techniques. While traditional rule-based methods perform well under stable conditions, they struggle in dynamic environments. To address this, Multi-Agent Reinforcement Learning (MARL), specifically Proximal Policy Optimization (PPO) and Deep Q-Networks (DQN), is combined with community/distributed P2P trading mechanisms. By incorporating auction-based market clearing, a price advisor agent, and load and battery management, the approach achieves significant improvements. Results show that, compared to baseline models, DQN reduces electricity costs by 14.2% in Ireland and 5.16% in Finland, while increasing electricity revenue by 7.24% and 12.73%, respectively. PPO achieves the lowest peak hour demand, reducing it by 55.5% in Ireland, while DQN reduces peak hour demand by 50.0% in Ireland and 27.02% in Finland. These improvements are attributed to both MARL algorithms and P2P energy trading, which together results in electricity cost and peak hour demand reduction, and increase electricity selling revenue. This study highlights the complementary strengths of DQN, PPO, and P2P trading in achieving efficient, adaptable, and sustainable energy management in rural communities.

**arXiv ID:** 2511.23148
</details>

<details>
<summary><strong>Agentic AI Framework for Smart Inventory Replenishment</strong> - Toqeer Ali Syed, Salman Jan, Gohar Ali, Ali Akarma, Ahmad Ali, Qurat-ul-Ain Mastoi - [[pdf]](https://arxiv.org/pdf/2511.23366)</summary>

**Abstract:** In contemporary retail, the variety of products available (e.g. clothing, groceries, cosmetics, frozen goods) make it difficult to predict the demand, prevent stockouts, and find high-potential products. We suggest an agentic AI model that will be used to monitor the inventory, initiate purchase attempts to the appropriate suppliers, and scan for trending or high-margin products to incorporate. The system applies demand forecasting, supplier selection optimization, multi-agent negotiation and continuous learning. We apply a prototype to a setting in the store of a middle scale mart, test its performance on three conventional and artificial data tables, and compare the results to the base heuristics. Our findings indicate that there is a decrease in stockouts, a reduction of inventory holding costs, and an improvement in product mix turnover. We address constraints, scalability as well as improvement prospect.

**arXiv ID:** 2511.23366
</details>

<details>
<summary><strong>Beyond Component Strength: Synergistic Integration and Adaptive Calibration in Multi-Agent RAG Systems</strong> - Jithin Krishnan - [[pdf]](https://arxiv.org/pdf/2511.21729)</summary>

**Abstract:** Building reliable retrieval-augmented generation (RAG) systems requires more than adding powerful components; it requires understanding how they interact. Using ablation studies on 50 queries (15 answerable, 10 edge cases, and 25 adversarial), we show that enhancements such as hybrid retrieval, ensemble verification, and adaptive thresholding provide almost no benefit when used in isolation, yet together achieve a 95% reduction in abstention (from 40% to 2%) without increasing hallucinations. We also identify a measurement challenge: different verification strategies can behave safely but assign inconsistent labels (for example, "abstained" versus "unsupported"), creating apparent hallucination rates that are actually artifacts of labeling. Our results show that synergistic integration matters more than the strength of any single component, that standardized metrics and labels are essential for correctly interpreting performance, and that adaptive calibration is needed to prevent overconfident over-answering even when retrieval quality is high.

**arXiv ID:** 2511.21729
</details>

<details>
<summary><strong>Bridging Planning and Execution: Multi-Agent Path Finding Under Real-World Deadlines</strong> - Jingtian Yan, Shuai Zhou, Stephen F. Smith, Jiaoyang Li - [[pdf]](https://arxiv.org/pdf/2511.21886)</summary>

**Abstract:** The Multi-Agent Path Finding (MAPF) problem aims to find collision-free paths for multiple agents while optimizing objectives such as the sum of costs or makespan. MAPF has wide applications in domains like automated warehouses, manufacturing systems, and airport logistics. However, most MAPF formulations assume a simplified robot model for planning, which overlooks execution-time factors such as kinodynamic constraints, communication latency, and controller variability. This gap between planning and execution is problematic for time-sensitive applications. To bridge this gap, we propose REMAP, an execution-informed MAPF planning framework that can be combined with leading search-based MAPF planners with minor changes. Our framework integrates the proposed ExecTimeNet to accurately estimate execution time based on planned paths. We demonstrate our method for solving MAPF with Real-world Deadlines (MAPF-RD) problem, where agents must reach their goals before a predefined wall-clock time. We integrate our framework with two popular MAPF methods, MAPF-LNS and CBS. Experiments show that REMAP achieves up to 20% improvement in solution quality over baseline methods (e.g., constant execution speed estimators) on benchmark maps with up to 300 agents.

**arXiv ID:** 2511.21886
</details>

<details>
<summary><strong>Heterogeneous Multi-Agent Reinforcement Learning with Attention for Cooperative and Scalable Feature Transformation</strong> - Tao Zhe, Huazhen Fang, Kunpeng Liu, Qian Lou, Tamzidul Hoque, Dongjie Wang - [[pdf]](https://arxiv.org/pdf/2511.21934)</summary>

**Abstract:** Feature transformation enhances downstream task performance by generating informative features through mathematical feature crossing. Despite the advancements in deep learning, feature transformation remains essential for structured data, where deep models often struggle to capture complex feature interactions. Prior literature on automated feature transformation has achieved success but often relies on heuristics or exhaustive searches, leading to inefficient and time-consuming processes. Recent works employ reinforcement learning (RL) to enhance traditional approaches through a more effective trial-and-error way. However, two limitations remain: 1) Dynamic feature expansion during the transformation process, which causes instability and increases the learning complexity for RL agents; 2) Insufficient cooperation and communication between agents, which results in suboptimal feature crossing operations and degraded model performance. To address them, we propose a novel heterogeneous multi-agent RL framework to enable cooperative and scalable feature transformation. The framework comprises three heterogeneous agents, grouped into two types, each designed to select essential features and operations for feature crossing. To enhance communication among these agents, we implement a shared critic mechanism that facilitates information exchange during feature transformation. To handle the dynamically expanding feature space, we tailor multi-head attention-based feature agents to select suitable features for feature crossing. Additionally, we introduce a state encoding technique during the optimization process to stabilize and enhance the learning dynamics of the RL agents, resulting in more robust and reliable transformation policies. Finally, we conduct extensive experiments to validate the effectiveness, efficiency, robustness, and interpretability of our model.

**arXiv ID:** 2511.21934
</details>

<details>
<summary><strong>AgentShield: Make MAS more secure and efficient</strong> - Kaixiang Wang, Zhaojiacheng Zhou, Bunyod Suvonov, Jiong Lou, Jie LI - [[pdf]](https://arxiv.org/pdf/2511.22924)</summary>

**Abstract:** Large Language Model (LLM)-based Multi-Agent Systems (MAS) offer powerful cooperative reasoning but remain vulnerable to adversarial attacks, where compromised agents can undermine the system's overall performance. Existing defenses either depend on single trusted auditors, creating single points of failure, or sacrifice efficiency for robustness. To resolve this tension, we propose \textbf{AgentShield}, a distributed framework for efficient, decentralized auditing. AgentShield introduces a novel three-layer defense: \textbf{(i) Critical Node Auditing} prioritizes high-influence agents via topological analysis; \textbf{(ii) Light Token Auditing} implements a cascade protocol using lightweight sentry models for rapid discriminative verification; and \textbf{(iii) Two-Round Consensus Auditing} triggers heavyweight arbiters only upon uncertainty to ensure global agreement. This principled design optimizes the robustness-efficiency trade-off. Experiments demonstrate that AgentShield achieves a 92.5\% recovery rate and reduces auditing overhead by over 70\% compared to existing methods, maintaining high collaborative accuracy across diverse MAS topologies and adversarial scenarios.

**arXiv ID:** 2511.22924
</details>

<details>
<summary><strong>MegaChat: A Synthetic Persian Q&A Dataset for High-Quality Sales Chatbot Evaluation</strong> - Mahdi Rahmani, AmirHossein Saffari, Reyhane Rahmani - [[pdf]](https://arxiv.org/pdf/2511.23397)</summary>

**Abstract:** Small and medium-sized enterprises (SMEs) in Iran increasingly leverage Telegram for sales, where real-time engagement is essential for conversion. However, developing AI-driven chatbots for this purpose requires large, high-quality question-and-answer (Q&A) datasets, which are typically expensive and resource-intensive to produce, especially for low-resource languages like Persian. In this paper, we introduce MegaChat, the first fully synthetic Persian Q&A dataset designed to evaluate intelligent sales chatbots in Telegram-based e-commerce. We propose a novel, automated multi-agent architecture that generates persona-aware Q&A pairs by collecting data from active Telegram shopping channels. The system employs specialized agents for question generation, validation, and refinement, ensuring the production of realistic and diverse conversational data. To evaluate answer generation, we compare three classic retrieval-augmented generation (RAG) models with our advanced agentic system, which features multi-query retrieval, reranking, and persona-aligned response synthesis. Using GPT-5.1 for evaluation across six quality dimensions, our results show that the agentic architecture outperformed traditional RAG models in 4 out of 5 diverse channels, demonstrating its ability to generate scalable, high-quality datasets without relying on expensive human annotation or complex fine-tuning. MegaChat provides SMEs with an efficient, cost-effective solution for building intelligent customer engagement systems in specialized commercial domains, enabling advancements in multilingual conversational AI for low-resource languages. Download: this https URL

**arXiv ID:** 2511.23397
</details>

<details>
<summary><strong>Formal Verification of Probabilistic Multi-Agent Systems for Ballistic Rocket Flight Using Probabilistic Alternating-Time Temporal Logic</strong> - Damian Kurpiewski, Jędrzej Michalczyk, Wojciech Jamroga, Jerzy Julian Michalski, Teofil Sidoruk - [[pdf]](https://arxiv.org/pdf/2511.22572)</summary>

**Abstract:** This technical report presents a comprehensive formal verification approach for probabilistic agent systems modeling ballistic rocket flight trajectories using Probabilistic Alternating-Time Temporal Logic (PATL). We describe an innovative verification framework specifically designed for analyzing critical safety properties of ballistic rockets engineered to achieve microgravity conditions for scientific experimentation. Our model integrates authentic flight telemetry data encompassing velocity vectors, pitch angles, attitude parameters, and GPS coordinates to construct probabilistic state transition systems that rigorously account for environmental stochasticity, particularly meteorological variability. We formalize mission-critical safety properties through PATL specifications to systematically identify trajectory deviation states where the rocket risks landing in prohibited or hazardous zones. The verification framework facilitates real-time safety monitoring and enables automated intervention mechanisms, including emergency engine disengagement protocols, when predefined safety thresholds are exceeded. Experimental validation demonstrates the practical effectiveness and reliability of our approach in ensuring mission safety while maintaining scientific mission objectives.

**arXiv ID:** 2511.22572
</details>

<details>
<summary><strong>CRAwDAD: Causal Reasoning Augmentation with Dual-Agent Debate</strong> - Finn G. Vamosi, Nils D. Forkert - [[pdf]](https://arxiv.org/pdf/2511.22854)</summary>

**Abstract:** When people reason about cause and effect, they often consider many competing "what if" scenarios before deciding which explanation fits best. Analogously, advanced language models capable of causal inference can consider multiple interventions and counterfactuals to judge the validity of causal claims. Crucially, this type of reasoning is less like a single calculation and more like an internal dialogue between alternative hypotheses. In this paper, we make this dialogue explicit through a dual-agent debate framework where one model provides a structured causal inference, and the other critically examines this reasoning for logical flaws. When disagreements arise, agents attempt to persuade each other, challenging each other's logic and revising their conclusions until they converge on a mutually agreed answer. To take advantage of this deliberative process, we specifically use reasoning language models, whose strengths in both causal inference and adversarial debate remain under-explored relative to standard large language models. We evaluate our approach on the CLadder dataset, a benchmark linking natural language questions to formally defined causal graphs across all three rungs of Pearl's ladder of causation. With Qwen3 and DeepSeek-R1 as debater agents, we demonstrate that multi-agent debate improves DeepSeek-R1's overall accuracy in causal inference from 78.03% to 87.45%, with the counterfactual category specifically improving from 67.94% to 80.04% accuracy. Similarly, Qwen3's overall accuracy improves from 84.16% to 89.41%, and counterfactual questions from 71.53% to 80.35%, showing that strong models can still benefit greatly from debate with weaker agents. Our results highlight the potential of reasoning models as building blocks for multi-agent systems in causal inference, and demonstrate the importance of diverse perspectives in causal problem-solving.

**arXiv ID:** 2511.22854
</details>

<details>
<summary><strong>From homeostasis to resource sharing: Biologically and economically aligned multi-objective multi-agent gridworld-based AI safety benchmarks</strong> - Roland Pihlakas - [[pdf]](https://arxiv.org/pdf/2410.00081)</summary>

**Abstract:** Developing safe, aligned agentic AI systems requires comprehensive empirical testing, yet many existing benchmarks neglect crucial themes aligned with biology and economics, both time-tested fundamental sciences describing our needs and preferences. To address this gap, the present work focuses on introducing biologically and economically motivated themes that have been neglected in current mainstream discussions on AI safety - namely a set of multi-objective, multi-agent alignment benchmarks that emphasize homeostasis for bounded and biological objectives, diminishing returns for unbounded, instrumental, and business objectives, sustainability principle, and resource sharing. Eight main benchmark environments have been implemented on the above themes, to illustrate key pitfalls and challenges in agentic AI-s, such as unboundedly maximizing a homeostatic objective, over-optimizing one objective at the expense of others, neglecting safety constraints, or depleting shared resources.

**arXiv ID:** 2410.00081
</details>

<details>
<summary><strong>MTTR-A: Measuring Cognitive Recovery Latency in Multi-Agent Systems</strong> - Barak Or - [[pdf]](https://arxiv.org/pdf/2511.20663)</summary>

**Abstract:** Ensuring cognitive stability in autonomous multi-agent systems (MAS) is a central challenge for large-scale, distributed AI. While existing observability tools monitor system outputs, they cannot quantify how rapidly agentic workflows recover once reasoning coherence has been lost. We adapt classical reliability metrics-Mean Time-to-Recovery (MTTR), Mean Time Between Failures (MTBF), and related ratios-into the cognitive domain, defining MTTR-A (Mean Time-to-Recovery for Agentic Systems) as a runtime measure of cognitive recovery latency. MTTR-A quantifies the time required for a MAS to detect reasoning drift and restore consistent operation, capturing the recovery of reasoning coherence rather than infrastructural repair.
A benchmark simulation using the AG~News corpus and the LangGraph orchestration framework was conducted, modeling recovery latencies across multiple reflex modes. Automated reflexes restored stability within approximately 6s on average, while human-approval interventions required about 12s. Across 200 runs, the median simulated MTTR-A was 6.21+-2.14s, MTBF=6.7+-2.14s, and NRR=0.08, demonstrating measurable runtime resilience across reflex strategies.
By formalizing recovery latency as a quantifiable property of distributed reasoning-and deriving reliability bounds linking recovery time and cognitive uptime-this work establishes a foundation for runtime dependability in agentic cognition, transforming cognitive recovery from an ad-hoc process into a standardized, interpretable performance

**arXiv ID:** 2511.20663
</details>

<details>
<summary><strong>A Layered Protocol Architecture for the Internet of Agents</strong> - Charles Fleming, Vijoy Pandey, Luca Muscariello, Ramana Kompella - [[pdf]](https://arxiv.org/pdf/2511.19699)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated remarkable performance improvements and the ability to learn domain-specific languages (DSLs), including APIs and tool interfaces. This capability has enabled the creation of AI agents that can perform preliminary computations and act through tool calling, now being standardized via protocols like MCP. However, LLMs face fundamental limitations: their context windows cannot grow indefinitely, constraining their memory and computational capacity. Agent collaboration emerges as essential for solving increasingly complex problems, mirroring how computational systems rely on different types of memory to scale. The "Internet of Agents" (IoA) represents the communication stack that enables agents to scale by distributing computation across collaborating entities. Current network architectural stacks (OSI and TCP/IP) were designed for data delivery between hosts and processes, not for agent collaboration with semantic understanding. To address this gap, we propose two new layers: an Agent Communication Layer (L8) and an Agent Semantic Negotiation Layer (L9). L8 formalizes the structure of communication, standardizing message envelopes, speech-act performatives (e.g., REQUEST, INFORM), and interaction patterns (e.g., request-reply, publish-subscribe), building on protocols like MCP. L9, which does not exist today, formalizes the meaning of communication, enabling agents to discover, negotiate, and lock a "Shared Context" -- a formal schema defining the concepts, tasks, and parameters relevant to their interaction. Together, these layers provide the foundation for scalable, distributed agent collaboration, enabling the next generation of multi-agentic systems.

**arXiv ID:** 2511.19699
</details>

<details>
<summary><strong>Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection</strong> - Michelle Yuan, Khushbu Pahwa, Shuaichen Chang, Mustafa Kaba, Jiarong Jiang, Xiaofei Ma, Yi Zhang, Monica Sunkara - [[pdf]](https://arxiv.org/pdf/2510.16499)</summary>

**Abstract:** Designing effective agentic systems requires the seamless composition and integration of agents, tools, and models within dynamic and uncertain environments. Most existing methods rely on static, semantic retrieval approaches for tool or agent discovery. However, effective reuse and composition of existing components remain challenging due to incomplete capability descriptions and the limitations of retrieval methods. Component selection suffers because the decisions are not based on capability, cost, and real-time utility. To address these challenges, we introduce a structured, automated framework for agentic system composition that is inspired by the knapsack problem. Our framework enables a composer agent to systematically identify, select, and assemble an optimal set of agentic components by jointly considering performance, budget constraints, and compatibility. By dynamically testing candidate components and modeling their utility in real-time, our approach streamlines the assembly of agentic systems and facilitates scalable reuse of resources. Empirical evaluation with Claude 3.5 Sonnet across five benchmarking datasets shows that our online-knapsack-based composer consistently lies on the Pareto frontier, achieving higher success rates at significantly lower component costs compared to our baselines. In the single-agent setup, the online knapsack composer shows a success rate improvement of up to 31.6% in comparison to the retrieval baselines. In multi-agent systems, the online knapsack composer increases success rate from 37% to 87% when agents are selected from an agent inventory of 100+ agents. The substantial performance gap confirms the robust adaptability of our method across diverse domains and budget constraints.

**arXiv ID:** 2510.16499
</details>

<details>
<summary><strong>Energy Efficient Sleep Mode Optimization in 5G mmWave Networks via Multi Agent Deep Reinforcement Learning</strong> - Saad Masrur, Ismail Guvenc, David Lopez Perez - [[pdf]](https://arxiv.org/pdf/2511.22105)</summary>

**Abstract:** Dynamic sleep mode optimization (SMO) in millimeter-wave (mmWave) networks is essential for maximizing energy efficiency (EE) under stringent quality-of-service (QoS) constraints. However, existing optimization and reinforcement learning (RL) approaches rely on aggregated, static base station (BS) traffic models that fail to capture non-stationary traffic dynamics and suffer from large state-action spaces, limiting real-world deployment. To address these challenges, this paper proposes a multi-agent deep reinforcement learning (MARL) framework using a Double Deep Q-Network (DDQN), referred to as MARL-DDQN, for adaptive SMO in a 3D urban environment with a time-varying and community-based user equipment (UE) mobility model. Unlike conventional single-agent RL, MARL-DDQN enables scalable, distributed decision-making with minimal signaling overhead. A realistic BS power consumption model and beamforming are integrated to accurately quantify EE, while QoS is defined in terms of throughput. The method adapts SMO policies to maximize EE while mitigating inter-cell interference and ensuring throughput fairness. Simulations show that MARL-DDQN outperforms state-of-the-art strategies, including All On, iterative QoS-aware load-based (IT-QoS-LB), MARL-DDPG, and MARL-PPO, achieving up to 0.60 Mbit/Joule EE, 8.5 Mbps 10th-percentile throughput, and meeting QoS constraints 95% of the time under dynamic scenarios.

**arXiv ID:** 2511.22105
</details>

<details>
<summary><strong>Emergent Coordination and Phase Structure in Independent Multi-Agent Reinforcement Learning</strong> - Azusa Yamaguchi - [[pdf]](https://arxiv.org/pdf/2511.23315)</summary>

**Abstract:** A clearer understanding of when coordination emerges, fluctuates, or collapses in decentralized multi-agent reinforcement learning (MARL) is increasingly sought in order to characterize the dynamics of multi-agent learning systems. We revisit fully independent Q-learning (IQL) as a minimal decentralized testbed and run large-scale experiments across environment size L and agent density rho. We construct a phase map using two axes - the cooperative success rate (CSR) and a stability index derived from TD-error variance - revealing three distinct regimes: a coordinated and stable phase, a fragile transition region, and a jammed or disordered phase. A sharp double Instability Ridge separates these regimes and corresponds to persistent kernel drift, the time-varying shift of each agent's effective transition kernel induced by others' policy updates. Synchronization analysis further shows that temporal alignment is required for sustained cooperation, and that competition between drift and synchronization generates the fragile regime. Removing agent identifiers eliminates drift entirely and collapses the three-phase structure, demonstrating that small inter-agent asymmetries are a necessary driver of drift. Overall, the results show that decentralized MARL exhibits a coherent phase structure governed by the interaction between scale, density, and kernel drift, suggesting that emergent coordination behaves as a distribution-interaction-driven phase phenomenon.

**arXiv ID:** 2511.23315
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>Optimized Agent Shift Scheduling Using Multi-Phase Allocation Approach</strong> - Sanalkumar K, Koushik Dey, Swati Meena - [[pdf]](https://arxiv.org/pdf/2511.22632)</summary>

**Abstract:** Effective agent shift scheduling is crucial for businesses, especially in the Contact Center as a Service (CCaaS) industry, to ensure seamless operations and fulfill employee needs. Most studies utilizing mathematical model-based solutions approach the problem as a single-step process, often resulting in inefficiencies and high computational demands. In contrast, we present a multi-phase allocation method that addresses scalability and accuracy by dividing the problem into smaller sub-problems of day and shift allocation, which significantly reduces number of computational variables and allows for targeted objective functions, ultimately enhancing both efficiency and accuracy. Each subproblem is modeled as a Integer Programming Problem (IPP), with solutions sequentially feeding into the subsequent subproblem. We then apply the proposed method, using a multi-objective framework, to address the difficulties posed by peak demand scenarios such as holiday rushes, where maintaining service levels is essential despite having limited number of employees

**arXiv ID:** 2511.22632
</details>

<details>
<summary><strong>Solving Context Window Overflow in AI Agents</strong> - Anton Bulle Labate, Valesca Moura de Sousa, Sandro Rama Fiorini, Leonardo Guerreiro Azevedo, Raphael Melo Thiago, Viviane Torres da Silva - [[pdf]](https://arxiv.org/pdf/2511.22729)</summary>

**Abstract:** Large Language Models (LLMs) have become increasingly capable of interacting with external tools, granting access to specialized knowledge beyond their training data - critical in dynamic, knowledge-intensive domains such as Chemistry and Materials Science. However, large tool outputs can overflow the LLMs' context window, preventing task completion. Existing solutions such as truncation or summarization fail to preserve complete outputs, making them unsuitable for workflows requiring the full data. This work introduces a method that enables LLMs to process and utilize tool responses of arbitrary length without loss of information. By shifting the model's interaction from raw data to memory pointers, the method preserves tool functionality, allows seamless integration into agentic workflows, and reduces token usage and execution time. The proposed method is validated on a real-world Materials Science application that cannot be executed with conventional workflows, and its effectiveness is demonstrated via a comparative analysis where both methods succeed. In this experiment, the proposed approach consumed approximately seven times fewer tokens than the traditional workflow.

**arXiv ID:** 2511.22729
</details>

<details>
<summary><strong>BUDD-e: an autonomous robotic guide for visually impaired users</strong> - Jinyang Li, Marcello Farina, Luca Mozzarelli, Luca Cattaneo, Panita Rattamasanaprapai, Eleonora A. Tagarelli, Matteo Corno, Paolo Perego, Giuseppe Andreoni, Emanuele Lettieri - [[pdf]](https://arxiv.org/pdf/2511.22541)</summary>

**Abstract:** This paper describes the design and the realization of a prototype of the novel guide robot BUDD-e for visually impaired users. The robot has been tested in a real scenario with the help of visually disabled volunteers at ASST Grande Ospedale Metropolitano Niguarda, in Milan. The results of the experimental campaign are throughly described in the paper, displaying its remarkable performance and user-acceptance.

**arXiv ID:** 2511.22541
</details>

<details>
<summary><strong>Safe Autonomous Lane Changing: Planning with Dynamic Risk Fields and Time-Varying Convex Space Generation</strong> - Zhen Tian, Zhihao Lin - [[pdf]](https://arxiv.org/pdf/2511.22829)</summary>

**Abstract:** This paper presents a novel trajectory planning pipeline for complex driving scenarios like autonomous lane changing, by integrating risk-aware planning with guaranteed collision avoidance into a unified optimization framework. We first construct a dynamic risk fields (DRF) that captures both the static and dynamic collision risks from surrounding vehicles. Then, we develop a rigorous strategy for generating time-varying convex feasible spaces that ensure kinematic feasibility and safety requirements. The trajectory planning problem is formulated as a finite-horizon optimal control problem and solved using a constrained iterative Linear Quadratic Regulator (iLQR) algorithm that jointly optimizes trajectory smoothness, control effort, and risk exposure while maintaining strict feasibility. Extensive simulations demonstrate that our method outperforms traditional approaches in terms of safety and efficiency, achieving collision-free trajectories with shorter lane-changing distances (28.59 m) and times (2.84 s) while maintaining smooth and comfortable acceleration patterns. In dense roundabout environments the planner further demonstrates robust adaptability, producing larger safety margins, lower jerk, and superior curvature smoothness compared with APF, MPC, and RRT based baselines. These results confirm that the integrated DRF with convex feasible space and constrained iLQR solver provides a balanced solution for safe, efficient, and comfortable trajectory generation in dynamic and interactive traffic scenarios.

**arXiv ID:** 2511.22829
</details>

<details>
<summary><strong>Motion-to-Motion Latency Measurement Framework for Connected and Autonomous Vehicle Teleoperation</strong> - François Provost, Faisal Hawlader, Mehdi Testouri, Raphaël Frank - [[pdf]](https://arxiv.org/pdf/2511.22467)</summary>

**Abstract:** Latency is a key performance factor for the teleoperation of Connected and Autonomous Vehicles (CAVs). It affects how quickly an operator can perceive changes in the driving environment and apply corrective actions. Most existing work focuses on Glass-to-Glass (G2G) latency, which captures delays only in the video pipeline. However, there is no standard method for measuring Motion-to-Motion (M2M) latency, defined as the delay between the physical steering movement of the remote operator and the corresponding steering motion in the vehicle. This paper presents an M2M latency measurement framework that uses Hall-effect sensors and two synchronized Raspberry Pi~5 devices. The system records interrupt-based timestamps on both sides to estimate M2M latency, independently of the underlying teleoperation architecture. Precision tests show an accuracy of 10--15~ms, while field results indicate that actuator delays dominate M2M latency, with median values above 750~ms.

**arXiv ID:** 2511.22467
</details>

<details>
<summary><strong>Switching control of underactuated multi-channel systems with input constraints for cooperative manipulation</strong> - Dongjae Lee, Dimos V. Dimarogonas, H. Jin Kim - [[pdf]](https://arxiv.org/pdf/2511.22810)</summary>

**Abstract:** This work presents an event-triggered switching control framework for a class of nonlinear underactuated multi-channel systems with input constraints. These systems are inspired by cooperative manipulation tasks involving underactuation, where multiple underactuated agents collaboratively push or pull an object to a target pose. Unlike existing approaches for multi-channel systems, our method addresses underactuation and the potential loss of controllability by additionally addressing channel assignment of agents. To simultaneously account for channel assignment, input constraints, and stabilization, we formulate the control problem as a Mixed Integer Linear Programming and derive sufficient conditions for its feasibility. To improve real-time computation efficiency, we introduce an event-triggered control scheme that maintains stability even between switching events through a quadratic programming-based stabilizing controller. We theoretically establish the semi-global exponential stability of the proposed method and the asymptotic stability of its extension to nonprehensile cooperative manipulation under noninstantaneous switching. The proposed framework is further validated through numerical simulations on 2D and 3D free-flyer systems and multi-robot nonprehensile pushing tasks.

**arXiv ID:** 2511.22810
</details>

<details>
<summary><strong>Can Intelligent User Interfaces Engage in Philosophical Discussions? A Longitudinal Study of Philosophers' Evolving Perceptions</strong> - Yibo Meng, Lyumanshan Ye, Eve He, Zhe Yan, Zhiming Liu, Yipeng Yu, Yan Guan, Xiaolan Ding - [[pdf]](https://arxiv.org/pdf/2511.23188)</summary>

**Abstract:** This study investigates the evolving attitudes of philosophy scholars towards the participation of generative AI based Intelligent User Interfaces (IUIs) in philosophical discourse. We conducted a three year (2023--2025) mixed methods longitudinal study with 16 philosophy scholars and students. Qualitative data from annual interviews reveal a three stage evolution in attitude: from initial resistance and unfamiliarity, to instrumental acceptance of the IUI as a tool, and finally to a deep principled questioning of the IUI's fundamental capacity for genuine philosophical thought. Quantitative data from blind assessments, where participants rated anonymized philosophical answers from both humans and an IUI, complement these findings. While participants acknowledged the IUI's proficiency in tasks requiring formal logic and knowledge reproduction, they consistently identified significant shortcomings in areas demanding dialectical reasoning, originality and embodied understanding. The study concludes that participants do not see the IUI as a peer but rather as a sophisticated mirror whose capabilities and limitations provoke a deeper reflection on the unique and irreplaceable human dimensions of philosophical inquiry, such as intuition, value laden commitment and the courage to question fundamental premises.

**arXiv ID:** 2511.23188
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (22 papers)</h2></summary>

<details>
<summary><strong>Hybrid Stackelberg Game and Diffusion-based Auction for Two-tier Agentic AI Task Offloading in Internet of Agents</strong> - Yue Zhong, Yongju Tong, Jiawen Kang, Minghui Dai, Hong-Ning Dai, Zhou Su, Dusit Niyato - [[pdf]](https://arxiv.org/pdf/2511.22076)</summary>

**Abstract:** The Internet of Agents (IoA) is rapidly gaining prominence as a foundational architecture for interconnected intelligent systems, designed to facilitate seamless discovery, communication, and collaborative reasoning among a vast network of Artificial Intelligence (AI) agents. Powered by Large Language and Vision-Language Models, IoA enables the development of interactive, rational agents capable of complex cooperation, moving far beyond traditional isolated models. IoA involves physical entities, i.e., Wireless Agents (WAs) with limited onboard resources, which need to offload their compute-intensive agentic AI services to nearby servers. Such servers can be Mobile Agents (MAs), e.g., vehicle agents, or Fixed Agents (FAs), e.g., end-side units agents. Given their fixed geographical locations and stable connectivity, FAs can serve as reliable communication gateways and task aggregation points. This stability allows them to effectively coordinate with and offload to an Aerial Agent (AA) tier, which has an advantage not affordable for highly mobile MAs with dynamic connectivity limitations. As such, we propose a two-tier optimization approach. The first tier employs a multi-leader multi-follower Stackelberg game. In the game, MAs and FAs act as the leaders who set resource prices. WAs are the followers to determine task offloading ratios. However, when FAs become overloaded, they can further offload tasks to available aerial resources. Therefore, the second tier introduces a Double Dutch Auction model where overloaded FAs act as the buyers to request resources, and AAs serve as the sellers for resource provision. We then develop a diffusion-based Deep Reinforcement Learning algorithm to solve the model. Numerical results demonstrate the superiority of our proposed scheme in facilitating task offloading.

**arXiv ID:** 2511.22076
</details>

<details>
<summary><strong>Adapting Like Humans: A Metacognitive Agent with Test-time Reasoning</strong> - Yang Li, Zhiyuan He, Yuxuan Huang, Zhuhanling Xiao, Chao Yu, Meng Fang, Kun Shao, Jun Wang - [[pdf]](https://arxiv.org/pdf/2511.23262)</summary>

**Abstract:** Recent Vision-Language Models (VLMs) exhibit strong perceptual reasoning abilities, yet they often struggle to adapt efficiently when encountering novel tasks at test time. In contrast, humans leverage the metacognitive model with memory, enabling continuous strategy refinement through metacognitive control when faced with new challenges. To bridge this gap, we propose metacognitive test-time reasoning (MCTR), a framework that equips models with the ability to learn, adapt, and improve during test time through metacognitive self-updating. Inspired by the dual structure of human metacognition, MCTR comprises meta-level and object-level VLM reasoning modules, each equipped with dedicated memory systems for hierarchical adaptive reasoning. Specifically, MCTR consists of (1) a meta-reasoning module which incrementally builds a structured memory by discovering and storing task-relevant rules, environmental patterns, and action-outcome relationships from test-time observations as natural language descriptions; and (2) an action-reasoning module that determines optimal actions through context-aware perception and strategic reasoning by dynamically retrieving and integrating knowledge from memory. The action-reasoning module continuously updates its policy through proposed metacognitive test-time reinforcement learning, adapting as knowledge memory evolves. We evaluate MCTR on 45 Atari games (33 seen, 12 unseen). MCTR demonstrates robust test-time adaptation, achieving 9/12 top-1 results on unseen games compared with baselines. Analyses through ablations, learning dynamics, and case studies reveal the complementary contributions of both components and show meta-reasoning evolving toward human-like adaptation strategies.

**arXiv ID:** 2511.23262
</details>

<details>
<summary><strong>PathReasoning: A multimodal reasoning agent for query-based ROI navigation on whole-slide images</strong> - Kunpeng Zhang, Hanwen Xu, Sheng Wang - [[pdf]](https://arxiv.org/pdf/2511.21902)</summary>

**Abstract:** Deciphering tumor microenvironment from Whole Slide Images (WSIs) is intriguing as it is key to cancer diagnosis, prognosis and treatment response. While these gigapixel images on one hand offer a comprehensive portrait of cancer, on the other hand, the extremely large size, as much as more than 10 billion pixels, make it challenging and time-consuming to navigate to corresponding regions to support diverse clinical inspection. Inspired by pathologists who conducted navigation on WSIs with a combination of sampling, reasoning and self-reflection, we proposed "PathReasoning", a multi-modal reasoning agent that iteratively navigates across WSIs through multiple rounds of reasoning and refinements. Specifically, starting with randomly sampled candidate regions, PathReasoning reviews current selections with self-reflection, reasoning over the correspondence between visual observations and clinical questions, and concludes by proposing new regions to explore. Across rounds, PathReasoning builds a reasoning chain that gradually directs attention to diagnostically relevant areas. PathReasoning turns each whole slide into a sequence of question-guided views, allowing the model to efficiently find informative ROIs within a fixed number of steps, without the need for dense pixel-level annotations. PathReasoning can substantially outperform strong ROI-selection approaches by 6.7% and 3.1% of AUROC on subtyping and longitudinal analysis tasks. The high-quality ROIs further support accurate report generation on breast cancer, significantly outperforming the standard GPT-4o by 10% in accuracy. PathReasoning prioritizes question-specific regions and constructs interpretable reasoning chains, supporting efficient slide review, consistent diagnostic interpretations, comprehensive reporting, and evidence traceability in digital pathology.

**arXiv ID:** 2511.21902
</details>

<details>
<summary><strong>Prompted Policy Search: Reinforcement Learning through Linguistic and Numerical Reasoning in LLMs</strong> - Yifan Zhou, Sachin Grover, Mohamed El Mistiri, Kamalesh Kalirathnam, Pratyush Kerhalkar, Swaroop Mishra, Neelesh Kumar, Sanket Gaurav, Oya Aran, Heni Ben Amor - [[pdf]](https://arxiv.org/pdf/2511.21928)</summary>

**Abstract:** Reinforcement Learning (RL) traditionally relies on scalar reward signals, limiting its ability to leverage the rich semantic knowledge often available in real-world tasks. In contrast, humans learn efficiently by combining numerical feedback with language, prior knowledge, and common sense. We introduce Prompted Policy Search (ProPS), a novel RL method that unifies numerical and linguistic reasoning within a single framework. Unlike prior work that augment existing RL components with language, ProPS places a large language model (LLM) at the center of the policy optimization loop-directly proposing policy updates based on both reward feedback and natural language input. We show that LLMs can perform numerical optimization in-context, and that incorporating semantic signals, such as goals, domain knowledge, and strategy hints can lead to more informed exploration and sample-efficient learning. ProPS is evaluated across fifteen Gymnasium tasks, spanning classic control, Atari games, and MuJoCo environments, and compared to seven widely-adopted RL algorithms (e.g., PPO, SAC, TRPO). It outperforms all baselines on eight out of fifteen tasks and demonstrates substantial gains when provided with domain knowledge. These results highlight the potential of unifying semantics and numerics for transparent, generalizable, and human-aligned RL.

**arXiv ID:** 2511.21928
</details>

<details>
<summary><strong>RELiQ: Scalable Entanglement Routing via Reinforcement Learning in Quantum Networks</strong> - Tobias Meuser, Jannis Weil, Aninda Lahiri, Marius Paraschiv - [[pdf]](https://arxiv.org/pdf/2511.22321)</summary>

**Abstract:** Quantum networks are becoming increasingly important because of advancements in quantum computing and quantum sensing, such as recent developments in distributed quantum computing and federated quantum machine learning. Routing entanglement in quantum networks poses several fundamental as well as technical challenges, including the high dynamicity of quantum network links and the probabilistic nature of quantum operations. Consequently, designing hand-crafted heuristics is difficult and often leads to suboptimal performance, especially if global network topology information is unavailable.
In this paper, we propose RELiQ, a reinforcement learning-based approach to entanglement routing that only relies on local information and iterative message exchange. Utilizing a graph neural network, RELiQ learns graph representations and avoids overfitting to specific network topologies - a prevalent issue for learning-based approaches. Our approach, trained on random graphs, consistently outperforms existing local information heuristics and learning-based approaches when applied to random and real-world topologies. When compared to global information heuristics, our method achieves similar or superior performance because of its rapid response to topology changes.

**arXiv ID:** 2511.22321
</details>

<details>
<summary><strong>PRO-V-R1: Reasoning Enhanced Programming Agent for RTL Verification</strong> - Yujie Zhao, Zhijing Wu, Zeqing Yuan, Zhongming Yu, Hejia Zhang, Wentao Ni, Chia-Tung Ho, Haoxing Ren, Jishen Zhao - [[pdf]](https://arxiv.org/pdf/2506.12200)</summary>

**Abstract:** Register-Transfer Level (RTL) verification is a primary bottleneck consuming 60-70% of development time. While Large Language Models (LLMs) show promise for RTL automation, their performance and research focus have overwhelmingly centered on RTL generation rather than verification. Current methods for RTL verification rely on large scale proprietary models (e.g., GPT-4o) to generate Python-based functional references, incurring a high cost and raising data-privacy risks. To date, an end-to-end open-source solution for autonomous verification remains absent.
We introduce PRO-V-R1, the first trainable open-source agentic framework for autonomous RTL verification. Our contributions are threefold: (1) we design PRO-V sys, a modular agentic system that couples LLM-based reasoning with programmatic tool use for RTL verification; (2) we establish a data construction pipeline that leverages existing RTL datasets to build simulation-validated, expert-level trajectories tailored for supervised fine-tuning (SFT) RTL verification agents; and (3) we implement an efficient reinforcement learning (RL) algorithm that uses verification-specific rewards derived from program-tool feedback to optimize the end-to-end verification workflow. Our empirical evaluation demonstrates PRO-V-R1 achieves a 57.7% functional correctness rate and 34.0% in robust fault detection, significantly outperforming the base model's 25.7% and 21.8% (respectively) from the state-of-the-art (SOTA) automatic verification system. This configuration also outperforms large-scale proprietary LLMs in functional correctness and shows comparable robustness for fault detection.

**arXiv ID:** 2506.12200
</details>

<details>
<summary><strong>From Ambiguity to Verdict: A Semiotic-Grounded Multi-Perspective Agent for LLM Logical Reasoning</strong> - Yunyao Zhang, Xinglang Zhang, Junxi Sheng, Wenbing Li, Junqing Yu, Wei Yang, Zikai Song - [[pdf]](https://arxiv.org/pdf/2509.24765)</summary>

**Abstract:** Logical reasoning is a fundamental capability of large language models. However, existing studies often overlook the interaction between logical complexity and semantic complexity, leading to systems that struggle with abstract propositions, ambiguous contexts, and conflicting stances that are central to human reasoning. We propose LogicAgent, a semiotic-square-guided framework that jointly addresses these two axes of difficulty. The semiotic square provides a principled structure for multi-perspective semantic analysis, and LogicAgent integrates automated deduction with reflective verification to manage logical complexity across deeper reasoning chains. To support evaluation under these conditions, we introduce RepublicQA, a benchmark that couples semantic complexity with logical depth. RepublicQA reaches college-level semantic difficulty (FKGL 11.94), contains philosophically grounded abstract propositions with systematically constructed contrary and contradictory forms, and offers a semantically rich setting for assessing logical reasoning in large language models. Experiments show that LogicAgent achieves state-of-the-art performance on RepublicQA with a 6.25 percent average improvement over strong baselines, and generalizes effectively to mainstream logical reasoning benchmarks including ProntoQA, ProofWriter, FOLIO, and ProverQA, achieving an additional 7.05 percent average gain. These results demonstrate the effectiveness of semiotic-grounded multi-perspective reasoning in enhancing logical performance.

**arXiv ID:** 2509.24765
</details>

<details>
<summary><strong>Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh for Empowering Access to Justice</strong> - Azmine Toushik Wasi, Wahid Faisal, Mst Rafia Islam - [[pdf]](https://arxiv.org/pdf/2511.08605)</summary>

**Abstract:** Bangladesh's low-income population faces major barriers to affordable legal advice due to complex legal language, procedural opacity, and high costs. Existing AI legal assistants lack Bengali-language support and jurisdiction-specific adaptation, limiting their effectiveness. To address this, we developed Mina, a multilingual LLM-based legal assistant tailored for the Bangladeshi context. It employs multilingual embeddings and a RAG-based chain-of-tools framework for retrieval, reasoning, translation, and document generation, delivering context-aware legal drafts, citations, and plain-language explanations via an interactive chat interface. Evaluated by law faculty from leading Bangladeshi universities across all stages of the 2022 and 2023 Bangladesh Bar Council Exams, Mina scored 75-80% in Preliminary MCQs, Written, and simulated Viva Voce exams, matching or surpassing average human performance and demonstrating clarity, contextual understanding, and sound legal reasoning. Even under a conservative upper bound, Mina operates at just 0.12-0.61% of typical legal consultation costs in Bangladesh, yielding a 99.4-99.9\% cost reduction relative to human-provided services. These results confirm its potential as a low-cost, multilingual AI assistant that automates key legal tasks and scales access to justice, offering a real-world case study on building domain-specific, low-resource systems and addressing challenges of multilingual adaptation, efficiency, and sustainable public-service AI deployment.

**arXiv ID:** 2511.08605
</details>

<details>
<summary><strong>Beyond Query-Level Comparison: Fine-Grained Reinforcement Learning for Text-to-SQL with Automated Interpretable Critiques</strong> - Guifeng Wang, Yuanfeng Song, Meng Yang, Tao Zhu, Xiaoming Yin, Xing Chen - [[pdf]](https://arxiv.org/pdf/2511.22258)</summary>

**Abstract:** Text-to-SQL, a pivotal natural language processing (NLP) task that converts textual queries into executable SQL, has seen substantial progress in recent years. However, existing evaluation and reward mechanisms used to train and assess the text-to-SQL models remain a critical bottleneck. Current approaches heavily rely on manually annotated gold SQL queries, which are costly to produce and impractical for large-scale evaluation. More importantly, most reinforcement learning (RL) methods in text-to-SQL leverage only the final binary execution outcome as the reward signal, a coarse-grained supervision that overlooks detailed structural and semantic errors from the perspective of rubrics. To address these challenges, we propose RuCo-C, a novel generative judge model for fine-grained, query-specific automatic evaluation using interpretable critiques without human intervention. Our framework first automatically generates query-specific evaluation rubrics for human-free annotation, linking them to interpretable critiques. Subsequently, it integrates densified reward feedback through a "progressive exploration" strategy during the RL training process, which dynamically adjusts the rewards to enhance the model's performance. Comprehensive experiments demonstrate that RuCo-C outperforms existing methods in text-to-SQL evaluation, yielding significant performance gains.

**arXiv ID:** 2511.22258
</details>

<details>
<summary><strong>Simulated patient systems powered by large language model-based AI agents offer potential for transforming medical education</strong> - Huizi Yu, Jiayan Zhou, Lingyao Li, Shan Chen, Jack Gallifant, Anye Shi, Xiang Li, Jingxian He, Wenyue Hua, Mingyu Jin, Guang Chen, Yang Zhou, Zhao Li, Trisha Gupte, Ming-Li Chen, Zahra Azizi, Qi Dou, Bryan P. Yan, Yongfeng Zhang, Yanqiu Xing, Themistocles L. Danielle S. Bitterman, Themistocles L. Assimes, Xin Ma, Lin Lu, Lizhou Fan - [[pdf]](https://arxiv.org/pdf/2409.18924)</summary>

**Abstract:** Background: Simulated patient systems are important in medical education and research, providing safe, integrative training environments and supporting clinical decision making. Advances in artificial intelligence (AI), especially large language models (LLMs), can enhance simulated patients by replicating medical conditions and doctor patient interactions with high fidelity and at low cost, but effectiveness and trustworthiness remain open challenges. Methods: We developed AIPatient, a simulated patient system powered by LLM based AI agents. The system uses a retrieval augmented generation (RAG) framework with six task specific agents for complex reasoning. To improve realism, it is linked to the AIPatient knowledge graph built from de identified real patient data in the MIMIC III intensive care database. Results: We evaluated electronic health record (EHR) based medical question answering (QA), readability, robustness, stability, and user experience. AIPatient reached 94.15 percent QA accuracy when all six agents were enabled, outperforming versions with partial or no agent integration. The knowledge base achieved an F1 score of 0.89. Readability scores showed a median Flesch Reading Ease of 68.77 and a median Flesch Kincaid Grade of 6.4, indicating accessibility for most medical trainees and clinicians. Robustness and stability were supported by non significant variance in repeated trials (analysis of variance F value 0.61, p greater than 0.1; F value 0.78, p greater than 0.1). A user study with medical students showed that AIPatient provides high fidelity, usability, and educational value, comparable to or better than human simulated patients for history taking. Conclusions: LLM based simulated patient systems can deliver accurate, readable, and reliable medical encounters and show strong potential to transform medical education.

**arXiv ID:** 2409.18924
</details>

<details>
<summary><strong>Prompt-R1: Collaborative Automatic Prompting Framework via End-to-end Reinforcement Learning</strong> - Wenjin Liu, Haoran Luo, Xueyuan Lin, Haoming Liu, Tiesunlong Shen, Jiapu Wang, Rui Mao, Erik Cambria - [[pdf]](https://arxiv.org/pdf/2511.01016)</summary>

**Abstract:** Recently, advanced large language models (LLMs) have emerged at an increasingly rapid pace. However, when faced with complex problems, most users are often unable to provide accurate and effective prompts to interact with LLMs, thus limiting the performance of LLMs. To address this challenge, we propose Prompt-R1, an end-to-end reinforcement learning framework that uses a small-scale LLM to collaborate with large-scale LLMs, replacing user interaction to solve problems better. This collaboration is cast as a multi-turn prompt interaction, where the small-scale LLM thinks and generates prompts, and the large-scale LLM performs complex reasoning. A dual-constrained reward is designed to optimize for correctness, generation quality, and reasoning accuracy. Prompt-R1 provides a plug-and-play framework that supports both inference and training with various large-scale LLMs. Experiments on multiple public datasets show that Prompt-R1 significantly outperforms baseline models across tasks. Our code is publicly available at this https URL.

**arXiv ID:** 2511.01016
</details>

<details>
<summary><strong>Asymmetric REINFORCE for off-Policy Reinforcement Learning: Balancing positive and negative rewards</strong> - Charles Arnal, Gaëtan Narozniak, Vivien Cabannes, Yunhao Tang, Julia Kempe, Remi Munos - [[pdf]](https://arxiv.org/pdf/2506.20520)</summary>

**Abstract:** Reinforcement learning (RL) is increasingly used to align large language models (LLMs). Off-policy methods offer greater implementation simplicity and data efficiency than on-policy techniques, but often result in suboptimal performance. In this work, we study the intermediate range of algorithms between off-policy RL and supervised fine-tuning by analyzing a simple off-policy REINFORCE algorithm, where the advantage is defined as $A=r-V$, with $r$ a reward and $V$ some tunable baseline. Intuitively, lowering $V$ emphasizes high-reward samples, while raising it penalizes low-reward ones more heavily. We first provide a theoretical analysis of this off-policy REINFORCE algorithm, showing that when the baseline $V$ lower-bounds the expected reward, the algorithm enjoys a policy improvement guarantee. Our analysis reveals that while on-policy updates can safely leverage both positive and negative signals, off-policy updates benefit from focusing more on positive rewards than on negative ones. We validate our findings experimentally in a controlled stochastic bandit setting and through fine-tuning state-of-the-art LLMs on reasoning tasks.

**arXiv ID:** 2506.20520
</details>

<details>
<summary><strong>BiCQL-ML: A Bi-Level Conservative Q-Learning Framework for Maximum Likelihood Inverse Reinforcement Learning</strong> - Junsung Park - [[pdf]](https://arxiv.org/pdf/2511.22210)</summary>

**Abstract:** Offline inverse reinforcement learning (IRL) aims to recover a reward function that explains expert behavior using only fixed demonstration data, without any additional online interaction. We propose BiCQL-ML, a policy-free offline IRL algorithm that jointly optimizes a reward function and a conservative Q-function in a bi-level framework, thereby avoiding explicit policy learning. The method alternates between (i) learning a conservative Q-function via Conservative Q-Learning (CQL) under the current reward, and (ii) updating the reward parameters to maximize the expected Q-values of expert actions while suppressing over-generalization to out-of-distribution actions. This procedure can be viewed as maximum likelihood estimation under a soft value matching principle. We provide theoretical guarantees that BiCQL-ML converges to a reward function under which the expert policy is soft-optimal. Empirically, we show on standard offline RL benchmarks that BiCQL-ML improves both reward recovery and downstream policy performance compared to existing offline IRL baselines.

**arXiv ID:** 2511.22210
</details>

<details>
<summary><strong>Improving Stochastic Action-Constrained Reinforcement Learning via Truncated Distributions</strong> - Roland Stolz, Michael Eichelbeck, Matthias Althoff - [[pdf]](https://arxiv.org/pdf/2511.22406)</summary>

**Abstract:** In reinforcement learning (RL), it is often advantageous to consider additional constraints on the action space to ensure safety or action relevance. Existing work on such action-constrained RL faces challenges regarding effective policy updates, computational efficiency, and predictable runtime. Recent work proposes to use truncated normal distributions for stochastic policy gradient methods. However, the computation of key characteristics, such as the entropy, log-probability, and their gradients, becomes intractable under complex constraints. Hence, prior work approximates these using the non-truncated distributions, which severely degrades performance. We argue that accurate estimation of these characteristics is crucial in the action-constrained RL setting, and propose efficient numerical approximations for them. We also provide an efficient sampling strategy for truncated policy distributions and validate our approach on three benchmark environments, which demonstrate significant performance improvements when using accurate estimations.

**arXiv ID:** 2511.22406
</details>

<details>
<summary><strong>OBLR-PO: A Theoretical Framework for Stable Reinforcement Learning</strong> - Zixun Huang, Jiayi Sheng, Zeyu Zheng - [[pdf]](https://arxiv.org/pdf/2511.23310)</summary>

**Abstract:** Existing reinforcement learning (RL)-based post-training methods for large language models have advanced rapidly, yet their design has largely been guided by heuristics rather than systematic theoretical principles. This gap limits our understanding of the properties of the gradient estimators and the associated optimization algorithms, thereby constraining opportunities to improve training stability and overall performance. In this work, we provide a unified theoretical framework that characterizes the statistical properties of commonly used policy-gradient estimators under mild assumptions. Our analysis establishes unbiasedness, derives exact variance expressions, and yields an optimization-loss upper bound that enables principled reasoning about learning dynamics. Building on these results, we prove convergence guarantees and derive an adaptive learning-rate schedule governed by the signal-to-noise ratio (SNR) of gradients. We further show that the variance-optimal baseline is a gradient-weighted estimator, offering a new principle for variance reduction and naturally enhancing stability beyond existing methods. These insights motivate Optimal Baseline and Learning-Rate Policy Optimization (OBLR-PO), an algorithm that jointly adapts learning rates and baselines in a theoretically grounded manner. Experiments on Qwen3-4B-Base and Qwen3-8B-Base demonstrate consistent gains over existing policy optimization methods, validating that our theoretical contributions translate into practical improvements in large-scale post-training.

**arXiv ID:** 2511.23310
</details>

<details>
<summary><strong>Model-Based Reward Shaping for Adversarial Inverse Reinforcement Learning in Stochastic Environments</strong> - Simon Sinong Zhan, Philip Wang, Qingyuan Wu, Yixuan Wang, Ruochen Jiao, Chao Huang, Qi Zhu - [[pdf]](https://arxiv.org/pdf/2410.03847)</summary>

**Abstract:** In this paper, we aim to tackle the limitation of the Adversarial Inverse Reinforcement Learning (AIRL) method in stochastic environments where theoretical results cannot hold and performance is degraded. To address this issue, we propose a novel method which infuses the dynamics information into the reward shaping with the theoretical guarantee for the induced optimal policy in the stochastic environments. Incorporating our novel model-enhanced rewards, we present a novel Model-Enhanced AIRL framework, which integrates transition model estimation directly into reward shaping. Furthermore, we provide a comprehensive theoretical analysis of the reward error bound and performance difference bound for our method. The experimental results in MuJoCo benchmarks show that our method can achieve superior performance in stochastic environments and competitive performance in deterministic environments, with significant improvement in sample efficiency, compared to existing baselines.

**arXiv ID:** 2410.03847
</details>

<details>
<summary><strong>OpenTwinMap: An Open-Source Digital Twin Generator for Urban Autonomous Driving</strong> - Alex Richardson, Jonathan Sprinkle - [[pdf]](https://arxiv.org/pdf/2511.21925)</summary>

**Abstract:** Digital twins of urban environments play a critical role in advancing autonomous vehicle (AV) research by enabling simulation, validation, and integration with emerging generative world models. While existing tools have demonstrated value, many publicly available solutions are tightly coupled to specific simulators, difficult to extend, or introduce significant technical overhead. For example, CARLA-the most widely used open-source AV simulator-provides a digital twin framework implemented entirely as an Unreal Engine C++ plugin, limiting flexibility and rapid prototyping. In this work, we propose OpenTwinMap, an open-source, Python-based framework for generating high-fidelity 3D urban digital twins. The completed framework will ingest LiDAR scans and OpenStreetMap (OSM) data to produce semantically segmented static environment assets, including road networks, terrain, and urban structures, which can be exported into Unreal Engine for AV simulation. OpenTwinMap emphasizes extensibility and parallelization, lowering the barrier for researchers to adapt and scale the pipeline to diverse urban contexts. We describe the current capabilities of the OpenTwinMap, which includes preprocessing of OSM and LiDAR data, basic road mesh and terrain generation, and preliminary support for CARLA integration.

**arXiv ID:** 2511.21925
</details>

<details>
<summary><strong>Nonholonomic Narrow Dead-End Escape with Deep Reinforcement Learning</strong> - Denghan Xiong, Yanzhe Zhao, Yutong Chen, Zichun Wang - [[pdf]](https://arxiv.org/pdf/2511.22338)</summary>

**Abstract:** Nonholonomic constraints restrict feasible velocities without reducing configuration-space dimension, which makes collision-free geometric paths generally non-executable for car-like robots. Ackermann steering further imposes curvature bounds and forbids in-place rotation, so escaping from narrow dead ends typically requires tightly sequenced forward and reverse maneuvers. Classical planners that decouple global search and local steering struggle in these settings because narrow passages occupy low-measure regions and nonholonomic reachability shrinks the set of valid connections, which degrades sampling efficiency and increases sensitivity to clearances. We study nonholonomic narrow dead-end escape for Ackermann vehicles and contribute three components. First, we construct a generator that samples multi-phase forward-reverse trajectories compatible with Ackermann kinematics and inflates their envelopes to synthesize families of narrow dead ends that are guaranteed to admit at least one feasible escape. Second, we construct a training environment that enforces kinematic constraints and train a policy using the soft actor-critic algorithm. Third, we evaluate against representative classical planners that combine global search with nonholonomic steering. Across parameterized dead-end families, the learned policy solves a larger fraction of instances, reduces maneuver count, and maintains comparable path length and planning time while under the same sensing and control limits. We provide our project as open source at this https URL

**arXiv ID:** 2511.22338
</details>

<details>
<summary><strong>HybridWorldSim: A Scalable and Controllable High-fidelity Simulator for Autonomous Driving</strong> - Qiang Li, Yingwenqi Jiang, Tuoxi Li, Duyu Chen, Xiang Feng, Yucheng Ao, Shangyue Liu, Xingchen Yu, Youcheng Cai, Yumeng Liu, Yuexin Ma, Xin Hu, Li Liu, Yu Zhang, Linkun Xu, Bingtao Gao, Xueyuan Wang, Shuchang Zhou, Xianming Liu, Ligang Liu - [[pdf]](https://arxiv.org/pdf/2511.22187)</summary>

**Abstract:** Realistic and controllable simulation is critical for advancing end-to-end autonomous driving, yet existing approaches often struggle to support novel view synthesis under large viewpoint changes or to ensure geometric consistency. We introduce HybridWorldSim, a hybrid simulation framework that integrates multi-traversal neural reconstruction for static backgrounds with generative modeling for dynamic agents. This unified design addresses key limitations of previous methods, enabling the creation of diverse and high-fidelity driving scenarios with reliable visual and spatial consistency. To facilitate robust benchmarking, we further release a new multi-traversal dataset MIRROR that captures a wide range of routes and environmental conditions across different cities. Extensive experiments demonstrate that HybridWorldSim surpasses previous state-of-the-art methods, providing a practical and scalable solution for high-fidelity simulation and a valuable resource for research and development in autonomous driving.

**arXiv ID:** 2511.22187
</details>

<details>
<summary><strong>Advancing Embodied Intelligence in Robotic-Assisted Endovascular Procedures: A Systematic Review of AI Solutions</strong> - Tianliang Yao, Bo Lu, Markus Kowarschik, Yixuan Yuan, Hubin Zhao, Sebastien Ourselin, Kaspar Althoefer, Junbo Ge, Peng Qi - [[pdf]](https://arxiv.org/pdf/2504.15327)</summary>

**Abstract:** Endovascular procedures have revolutionized vascular disease treatment, yet their manual execution is challenged by the demands for high precision, operator fatigue, and radiation exposure. Robotic systems have emerged as transformative solutions to mitigate these inherent limitations. A pivotal moment has arrived, where a confluence of pressing clinical needs and breakthroughs in AI creates an opportunity for a paradigm shift toward Embodied Intelligence (EI), enabling robots to navigate complex vascular networks and adapt to dynamic physiological conditions. Data-driven approaches, leveraging advanced computer vision, medical image analysis, and machine learning, drive this evolution by enabling real-time vessel segmentation, device tracking, and anatomical landmark detection. Reinforcement learning and imitation learning further enhance navigation strategies and replicate expert techniques. This review systematically analyzes the integration of EI into endovascular robotics, identifying profound systemic challenges such as the heterogeneity in validation standards and the gap between human mimicry and machine-native capabilities. Based on this analysis, a conceptual roadmap is proposed that reframes the ultimate objective away from systems that supplant clinical decision-making. This vision of augmented intelligence, where the clinician's role evolves into that of a high-level supervisor, provides a principled foundation for the future of the field.

**arXiv ID:** 2504.15327
</details>

<details>
<summary><strong>Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning</strong> - Gunshi Gupta, Karmesh Yadav, Zsolt Kira, Yarin Gal, Rahaf Aljundi - [[pdf]](https://arxiv.org/pdf/2510.19732)</summary>

**Abstract:** To enable embodied agents to operate effectively over extended timeframes, it is crucial to develop models that form and access memories to stay contextualized in their environment. In the current paradigm of training transformer-based policies for embodied sequential decision-making tasks, visual inputs often overwhelm the context limits of transformers, while humans can maintain and utilize a lifetime of experience compressed as memories. Significant compression is possible in principle, as much of the input is irrelevant and can be abstracted. However, existing approaches predominantly focus on either recurrent models with fixed-size memory or transformers with full-context reliance. In this work, we propose Memo, a transformer-based architecture and training recipe for reinforcement learning (RL) on memory-intensive, long-horizon tasks. Memo incorporates the creation and retrieval of memory by interleaving periodic summarization tokens with the inputs of a model during training. We demonstrate Memo's effectiveness on a gridworld meta-RL benchmark and a multi-object navigation task in photo-realistic indoor settings. Memo outperforms naive long-context transformer baselines while being more compute and storage efficient. Additionally, Memo generalizes better to longer contexts at inference time and remains robust in streaming settings, where historical context must be truncated to fit inference constraints. Our code is available at: this https URL.

**arXiv ID:** 2510.19732
</details>

<details>
<summary><strong>Interactive Groupwise Comparison for Reinforcement Learning from Human Feedback</strong> - Jan Kompatscher, Danqing Shi, Giovanna Varni, Tino Weinkauf, Antti Oulasvirta - [[pdf]](https://arxiv.org/pdf/2507.04340)</summary>

**Abstract:** Reinforcement learning from human feedback (RLHF) has emerged as a key enabling technology for aligning AI behaviour with human preferences. The traditional way to collect data in RLHF is via pairwise comparisons: human raters are asked to indicate which one of two samples they prefer. We present an interactive visualisation that better exploits the human visual ability to compare and explore whole groups of samples. The interface is comprised of two linked views: 1) an exploration view showing a contextual overview of all sampled behaviours organised in a hierarchical clustering structure; and 2) a comparison view displaying two selected groups of behaviours for user queries. Users can efficiently explore large sets of behaviours by iterating between these two views. Additionally, we devised an active learning approach suggesting groups for comparison. As shown by our evaluation in six simulated robotics tasks, our approach increases the final rewards by 69.34%. It leads to lower error rates and better policies. We open-source the code that can be easily integrated into the RLHF training loop, supporting research on human-AI alignment.

**arXiv ID:** 2507.04340
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
