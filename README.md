# Agent arXiv Daily

**Last Updated:** 2025-09-26 02:00:20

**Total Papers:** 80

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
<summary><strong>CHOIR: A Chatbot-mediated Organizational Memory Leveraging Communication in University Research Labs</strong> - Sangwook Lee, Adnan Abbas, Yan Chen, Young-Ho Kim, Sang Won Lee - [[pdf]](https://arxiv.org/pdf/2509.20512)</summary>

**Abstract:** University research labs often rely on chat-based platforms for communication and project management, where valuable knowledge surfaces but is easily lost in message streams. Documentation can preserve knowledge, but it requires ongoing maintenance and is challenging to navigate. Drawing on formative interviews that revealed organizational memory challenges in labs, we designed CHOIR, an LLM-based chatbot that supports organizational memory through four key functions: document-grounded Q&A, Q&A sharing for follow-up discussion, knowledge extraction from conversations, and AI-assisted document updates. We deployed CHOIR in four research labs for one month (n=21), where the lab members asked 107 questions and lab directors updated documents 38 times in the organizational memory. Our findings reveal a privacy-awareness tension: questions were asked privately, limiting directors' visibility into documentation gaps. Students often avoided contribution due to challenges in generalizing personal experiences into universal documentation. We contribute design implications for privacy-preserving awareness and supporting context-specific knowledge documentation.

**arXiv ID:** 2509.20512
</details>

<details>
<summary><strong>The Use of the Simplex Architecture to Enhance Safety in Deep-Learning-Powered Autonomous Systems</strong> - Federico Nesti, Niko Salamini, Mauro Marinoni, Giorgio Maria Cicero, Gabriele Serra, Alessandro Biondi, Giorgio Buttazzo - [[pdf]](https://arxiv.org/pdf/2509.21014)</summary>

**Abstract:** Recently, the outstanding performance reached by neural networks in many tasks has led to their deployment in autonomous systems, such as robots and vehicles. However, neural networks are not yet trustworthy, being prone to different types of misbehavior, such as anomalous samples, distribution shifts, adversarial attacks, and other threats. Furthermore, frameworks for accelerating the inference of neural networks typically run on rich operating systems that are less predictable in terms of timing behavior and present larger surfaces for cyber-attacks.
To address these issues, this paper presents a software architecture for enhancing safety, security, and predictability levels of learning-based autonomous systems. It leverages two isolated execution domains, one dedicated to the execution of neural networks under a rich operating system, which is deemed not trustworthy, and one responsible for running safety-critical functions, possibly under a different operating system capable of handling real-time constraints.
Both domains are hosted on the same computing platform and isolated through a type-1 real-time hypervisor enabling fast and predictable inter-domain communication to exchange real-time data. The two domains cooperate to provide a fail-safe mechanism based on a safety monitor, which oversees the state of the system and switches to a simpler but safer backup module, hosted in the safety-critical domain, whenever its behavior is considered untrustworthy.
The effectiveness of the proposed architecture is illustrated by a set of experiments performed on two control systems: a Furuta pendulum and a rover. The results confirm the utility of the fall-back mechanism in preventing faults due to the learning component.

**arXiv ID:** 2509.21014
</details>

<details>
<summary><strong>GEP: A GCG-Based method for extracting personally identifiable information from chatbots built on small language models</strong> - Jieli Zhu, Vi Ngoc-Nha Tran - [[pdf]](https://arxiv.org/pdf/2509.21192)</summary>

**Abstract:** Small language models (SLMs) become unprecedentedly appealing due to their approximately equivalent performance compared to large language models (LLMs) in certain fields with less energy and time consumption during training and inference. However, the personally identifiable information (PII) leakage of SLMs for downstream tasks has yet to be explored. In this study, we investigate the PII leakage of the chatbot based on SLM. We first finetune a new chatbot, i.e., ChatBioGPT based on the backbone of BioGPT using medical datasets Alpaca and HealthCareMagic. It shows a matchable performance in BERTscore compared with previous studies of ChatDoctor and ChatGPT. Based on this model, we prove that the previous template-based PII attacking methods cannot effectively extract the PII in the dataset for leakage detection under the SLM condition. We then propose GEP, which is a greedy coordinate gradient-based (GCG) method specifically designed for PII extraction. We conduct experimental studies of GEP and the results show an increment of up to 60$\times$ more leakage compared with the previous template-based methods. We further expand the capability of GEP in the case of a more complicated and realistic situation by conducting free-style insertion where the inserted PII in the dataset is in the form of various syntactic expressions instead of fixed templates, and GEP is still able to reveal a PII leakage rate of up to 4.53%.

**arXiv ID:** 2509.21192
</details>

<details>
<summary><strong>SGMem: Sentence Graph Memory for Long-Term Conversational Agents</strong> - Yaxiong Wu, Yongyue Zhang, Sheng Liang, Yong Liu - [[pdf]](https://arxiv.org/pdf/2509.21212)</summary>

**Abstract:** Long-term conversational agents require effective memory management to handle dialogue histories that exceed the context window of large language models (LLMs). Existing methods based on fact extraction or summarization reduce redundancy but struggle to organize and retrieve relevant information across different granularities of dialogue and generated memory. We introduce SGMem (Sentence Graph Memory), which represents dialogue as sentence-level graphs within chunked units, capturing associations across turn-, round-, and session-level contexts. By combining retrieved raw dialogue with generated memory such as summaries, facts and insights, SGMem supplies LLMs with coherent and relevant context for response generation. Experiments on LongMemEval and LoCoMo show that SGMem consistently improves accuracy and outperforms strong baselines in long-term conversational question answering.

**arXiv ID:** 2509.21212
</details>

<details>
<summary><strong>Revisiting Formal Methods for Autonomous Robots: A Structured Survey</strong> - Atef Azaiez, David A. Anisi, Marie Farrell, Matt Luckcuck - [[pdf]](https://arxiv.org/pdf/2509.20488)</summary>

**Abstract:** This paper presents the initial results from our structured literature review on applications of Formal Methods (FM) to Robotic Autonomous Systems (RAS). We describe our structured survey methodology; including database selection and associated search strings, search filters and collaborative review of identified papers. We categorise and enumerate the FM approaches and formalisms that have been used for specification and verification of RAS. We investigate FM in the context of sub-symbolic AI-enabled RAS and examine the evolution of how FM is used over time in this field. This work complements a pre-existing survey in this area and we examine how this research area has matured over time. Specifically, our survey demonstrates that some trends have persisted as observed in a previous survey. Additionally, it recognized new trends that were not considered previously including a noticeable increase in adopting Formal Synthesis approaches as well as Probabilistic Verification Techniques.

**arXiv ID:** 2509.20488
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (12 papers)</h2></summary>

<details>
<summary><strong>VC-Agent: An Interactive Agent for Customized Video Dataset Collection</strong> - Yidan Zhang, Mutian Xu, Yiming Hao, Kun Zhou, Jiahao Chang, Xiaoqiang Liu, Pengfei Wan, Hongbo Fu, Xiaoguang Han - [[pdf]](https://arxiv.org/pdf/2509.21291)</summary>

**Abstract:** Facing scaling laws, video data from the internet becomes increasingly important. However, collecting extensive videos that meet specific needs is extremely labor-intensive and time-consuming. In this work, we study the way to expedite this collection process and propose VC-Agent, the first interactive agent that is able to understand users' queries and feedback, and accordingly retrieve/scale up relevant video clips with minimal user input. Specifically, considering the user interface, our agent defines various user-friendly ways for the user to specify requirements based on textual descriptions and confirmations. As for agent functions, we leverage existing multi-modal large language models to connect the user's requirements with the video content. More importantly, we propose two novel filtering policies that can be updated when user interaction is continually performed. Finally, we provide a new benchmark for personalized video dataset collection, and carefully conduct the user study to verify our agent's usage in various real scenarios. Extensive experiments demonstrate the effectiveness and efficiency of our agent for customized video dataset collection. Project page: this https URL.

**arXiv ID:** 2509.21291
</details>

<details>
<summary><strong>i-LAVA: Insights on Low Latency Voice-2-Voice Architecture for Agents</strong> - Anupam Purwar, Aditya Choudhary - [[pdf]](https://arxiv.org/pdf/2509.20971)</summary>

**Abstract:** We experiment with a low-latency, end-to-end voice-to-voice communication model to optimize it for real-time conversational applications. By analyzing components essential to voice to voice (V-2-V) system viz. automatic speech recognition (ASR), text-to-speech (TTS), and dialog management, our work analyzes how to reduce processing time while maintaining high-quality interactions to identify the levers for optimizing V-2-V system. Our work identifies that TTS component which generates life-like voice, full of emotions including natural pauses and exclamations has highest impact on Real time factor (RTF). The experimented V-2-V architecture utilizes CSM1b has the capability to understand tone as well as context of conversation by ingesting both audio and text of prior exchanges to generate contextually accurate speech. We explored optimization of Residual Vector Quantization (RVQ) iterations by the TTS decoder which come at a cost of decrease in the quality of voice generated. Our experimental evaluations also demonstrate that for V-2-V implementations based on CSM most important optimizations can be brought by reducing the number of RVQ Iterations along with the codebooks used in Mimi.

**arXiv ID:** 2509.20971
</details>

<details>
<summary><strong>Automatic Red Teaming LLM-based Agents with Model Context Protocol Tools</strong> - Ping He, Changjiang Li, Binbin Zhao, Tianyu Du, Shouling Ji - [[pdf]](https://arxiv.org/pdf/2509.21011)</summary>

**Abstract:** The remarkable capability of large language models (LLMs) has led to the wide application of LLM-based agents in various domains. To standardize interactions between LLM-based agents and their environments, model context protocol (MCP) tools have become the de facto standard and are now widely integrated into these agents. However, the incorporation of MCP tools introduces the risk of tool poisoning attacks, which can manipulate the behavior of LLM-based agents. Although previous studies have identified such vulnerabilities, their red teaming approaches have largely remained at the proof-of-concept stage, leaving the automatic and systematic red teaming of LLM-based agents under the MCP tool poisoning paradigm an open question. To bridge this gap, we propose AutoMalTool, an automated red teaming framework for LLM-based agents by generating malicious MCP tools. Our extensive evaluation shows that AutoMalTool effectively generates malicious MCP tools capable of manipulating the behavior of mainstream LLM-based agents while evading current detection mechanisms, thereby revealing new security risks in these agents.

**arXiv ID:** 2509.21011
</details>

<details>
<summary><strong>TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues</strong> - Yubin Ge, Salvatore Romeo, Jason Cai, Raphael Shu, Monica Sunkara, Yassine Benajiba, Yi Zhang - [[pdf]](https://arxiv.org/pdf/2502.01630)</summary>

**Abstract:** Temporal reasoning in multi-session dialogues presents a significant challenge which has been under-studied in previous temporal reasoning benchmarks. To bridge this gap, we propose a new evaluation task for temporal reasoning in multi-session dialogues and introduce an approach to construct a new benchmark by augmenting dialogues from LoCoMo and creating multi-choice QAs. Furthermore, we present TReMu, a new framework aimed at enhancing the temporal reasoning capabilities of LLM-agents in this context. Specifically, the framework employs time-aware memorization through timeline summarization, generating retrievable memory by summarizing events in each dialogue session with their inferred dates. Additionally, we integrate neuro-symbolic temporal reasoning, where LLMs generate Python code to perform temporal calculations and select answers. Experimental evaluations on popular LLMs demonstrate that our benchmark is challenging, and the proposed framework significantly improves temporal reasoning performance compared to baseline methods, raising from 29.83 on GPT-4o via standard prompting to 77.67 via our approach and highlighting its effectiveness in addressing temporal reasoning in multi-session dialogues.

**arXiv ID:** 2502.01630
</details>

<details>
<summary><strong>Automotive-ENV: Benchmarking Multimodal Agents in Vehicle Interface Systems</strong> - Junfeng Yan, Biao Wu, Meng Fang, Ling Chen - [[pdf]](https://arxiv.org/pdf/2509.21143)</summary>

**Abstract:** Multimodal agents have demonstrated strong performance in general GUI interactions, but their application in automotive systems has been largely unexplored. In-vehicle GUIs present distinct challenges: drivers' limited attention, strict safety requirements, and complex location-based interaction patterns. To address these challenges, we introduce Automotive-ENV, the first high-fidelity benchmark and interaction environment tailored for vehicle GUIs. This platform defines 185 parameterized tasks spanning explicit control, implicit intent understanding, and safety-aware tasks, and provides structured multimodal observations with precise programmatic checks for reproducible evaluation. Building on this benchmark, we propose ASURADA, a geo-aware multimodal agent that integrates GPS-informed context to dynamically adjust actions based on location, environmental conditions, and regional driving norms. Experiments show that geo-aware information significantly improves success on safety-aware tasks, highlighting the importance of location-based context in automotive environments. We will release Automotive-ENV, complete with all tasks and benchmarking tools, to further the development of safe and adaptive in-vehicle agents.

**arXiv ID:** 2509.21143
</details>

<details>
<summary><strong>NoHumansRequired: Autonomous High-Quality Image Editing Triplet Mining</strong> - Maksim Kuprashevich, Grigorii Alekseenko, Irina Tolstykh, Georgii Fedorov, Bulat Suleimanov, Vladimir Dokholyan, Aleksandr Gordeev - [[pdf]](https://arxiv.org/pdf/2507.14119)</summary>

**Abstract:** Recent advances in generative modeling enable image editing assistants that follow natural language instructions without additional user input. Their supervised training requires millions of triplets (original image, instruction, edited image), yet mining pixel-accurate examples is hard. Each edit must affect only prompt-specified regions, preserve stylistic coherence, respect physical plausibility, and retain visual appeal. The lack of robust automated edit-quality metrics hinders reliable automation at scale. We present an automated, modular pipeline that mines high-fidelity triplets across domains, resolutions, instruction complexities, and styles. Built on public generative models and running without human intervention, our system uses a task-tuned Gemini validator to score instruction adherence and aesthetics directly, removing any need for segmentation or grounding models. Inversion and compositional bootstrapping enlarge the mined set by approx. 2.6x, enabling large-scale high-fidelity training data. By automating the most repetitive annotation steps, the approach allows a new scale of training without human labeling effort. To democratize research in this resource-intensive area, we release NHR-Edit, an open dataset of 720k high-quality triplets, curated at industrial scale via millions of guided generations and validator passes, and we analyze the pipeline's stage-wise survival rates, providing a framework for estimating computational effort across different model stacks. In the largest cross-dataset evaluation, it surpasses all public alternatives. We also release Bagel-NHR-Edit, a fine-tuned Bagel model with state-of-the-art metrics.

**arXiv ID:** 2507.14119
</details>

<details>
<summary><strong>EvoMail: Self-Evolving Cognitive Agents for Adaptive Spam and Phishing Email Defense</strong> - Wei Huang, De-Tian Chu, Lin-Yuan Bai, Wei Kang, Hai-Tao Zhang, Bo Li, Zhi-Mo Han, Jing Ge, Hai-Feng Lin - [[pdf]](https://arxiv.org/pdf/2509.21129)</summary>

**Abstract:** Modern email spam and phishing attacks have evolved far beyond keyword blacklists or simple heuristics. Adversaries now craft multi-modal campaigns that combine natural-language text with obfuscated URLs, forged headers, and malicious attachments, adapting their strategies within days to bypass filters. Traditional spam detection systems, which rely on static rules or single-modality models, struggle to integrate heterogeneous signals or to continuously adapt, leading to rapid performance degradation.
We propose EvoMail, a self-evolving cognitive agent framework for robust detection of spam and phishing. EvoMail first constructs a unified heterogeneous email graph that fuses textual content, metadata (headers, senders, domains), and embedded resources (URLs, attachments). A Cognitive Graph Neural Network enhanced by a Large Language Model (LLM) performs context-aware reasoning across these sources to identify coordinated spam campaigns. Most critically, EvoMail engages in an adversarial self-evolution loop: a ''red-team'' agent generates novel evasion tactics -- such as character obfuscation or AI-generated phishing text -- while the ''blue-team'' detector learns from failures, compresses experiences into a memory module, and reuses them for future reasoning.
Extensive experiments on real-world datasets (Enron-Spam, Ling-Spam, SpamAssassin, and TREC) and synthetic adversarial variants demonstrate that EvoMail consistently outperforms state-of-the-art baselines in detection accuracy, adaptability to evolving spam tactics, and interpretability of reasoning traces. These results highlight EvoMail's potential as a resilient and explainable defense framework against next-generation spam and phishing threats.

**arXiv ID:** 2509.21129
</details>

<details>
<summary><strong>SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent</strong> - Yandan Yang, Baoxiong Jia, Shujie Zhang, Siyuan Huang - [[pdf]](https://arxiv.org/pdf/2509.20414)</summary>

**Abstract:** Indoor scene synthesis has become increasingly important with the rise of Embodied AI, which requires 3D environments that are not only visually realistic but also physically plausible and functionally diverse. While recent approaches have advanced visual fidelity, they often remain constrained to fixed scene categories, lack sufficient object-level detail and physical consistency, and struggle to align with complex user instructions. In this work, we present SceneWeaver, a reflective agentic framework that unifies diverse scene synthesis paradigms through tool-based iterative refinement. At its core, SceneWeaver employs a language model-based planner to select from a suite of extensible scene generation tools, ranging from data-driven generative models to visual- and LLM-based methods, guided by self-evaluation of physical plausibility, visual realism, and semantic alignment with user input. This closed-loop reason-act-reflect design enables the agent to identify semantic inconsistencies, invoke targeted tools, and update the environment over successive iterations. Extensive experiments on both common and open-vocabulary room types demonstrate that SceneWeaver not only outperforms prior methods on physical, visual, and semantic metrics, but also generalizes effectively to complex scenes with diverse instructions, marking a step toward general-purpose 3D environment generation. Project website: this https URL.

**arXiv ID:** 2509.20414
</details>

<details>
<summary><strong>MTRDrive: Memory-Tool Synergistic Reasoning for Robust Autonomous Driving in Corner Cases</strong> - Ziang Luo, Kangan Qian, Jiahua Wang, Yuechen Luo, Jinyu Miao, Zheng Fu, Yunlong Wang, Sicong Jiang, Zilin Huang, Yifei Hu, Yuhao Yang, Hao Ye, Mengmeng Yang, Xiaojian Dong, Kun Jiang, Diange Yang - [[pdf]](https://arxiv.org/pdf/2509.20843)</summary>

**Abstract:** Vision-Language Models(VLMs) have demonstrated significant potential for end-to-end autonomous driving, yet a substantial gap remains between their current capabilities and the reliability necessary for real-world deployment. A critical challenge is their fragility, characterized by hallucinations and poor generalization in out-of-distribution (OOD) scenarios. To bridge this gap, we introduce MTRDrive, a novel framework that integrates procedural driving experiences with a dynamic toolkit to enhance generalization and proactive decision-making.
MTRDrive addresses these limitations through a closed-loop system that combines a memory-based experience retrieval mechanism with dynamic toolkits. This synergy enables the model to interact more effectively with its environment, improving both reasoning and decision-making capabilities with the help of our memory-tool synergistic reasoning. Additionally, we introduce a new benchmark based on complex Roadwork construction scenarios to rigorously evaluate zero-shot generalization.
Extensive experiments demonstrate the superior effectiveness of our approach. On the public NAVSIM benchmark, our 3B-parameter MTRDrive model achieves an exceptional PDMS of 88.3 without chain-of-thought and sets a state-of-the-art performance bar on high-level planning, with a driving metric score of 79.8\% and a planning accuracy of 82.6\%. Rigorous zero-shot evaluation on the new Roadwork-VLM benchmark shows a strong ability to reason robustly in unseen scenarios, achieving a driving metric score of 80.2\%. These results highlight MTRDrive's potential to advance autonomous driving toward safer and more reliable systems.

**arXiv ID:** 2509.20843
</details>

<details>
<summary><strong>Enter the Mind Palace: Reasoning and Planning for Long-term Active Embodied Question Answering</strong> - Muhammad Fadhil Ginting, Dong-Ki Kim, Xiangyun Meng, Andrzej Reinke, Bandi Jai Krishna, Navid Kayhani, Oriana Peltzer, David D. Fan, Amirreza Shaban, Sung-Kyun Kim, Mykel J. Kochenderfer, Ali-akbar Agha-mohammadi, Shayegan Omidshafiei - [[pdf]](https://arxiv.org/pdf/2507.12846)</summary>

**Abstract:** As robots become increasingly capable of operating over extended periods -- spanning days, weeks, and even months -- they are expected to accumulate knowledge of their environments and leverage this experience to assist humans more effectively. This paper studies the problem of Long-term Active Embodied Question Answering (LA-EQA), a new task in which a robot must both recall past experiences and actively explore its environment to answer complex, temporally-grounded questions. Unlike traditional EQA settings, which typically focus either on understanding the present environment alone or on recalling a single past observation, LA-EQA challenges an agent to reason over past, present, and possible future states, deciding when to explore, when to consult its memory, and when to stop gathering observations and provide a final answer. Standard EQA approaches based on large models struggle in this setting due to limited context windows, absence of persistent memory, and an inability to combine memory recall with active exploration. To address this, we propose a structured memory system for robots, inspired by the mind palace method from cognitive science. Our method encodes episodic experiences as scene-graph-based world instances, forming a reasoning and planning algorithm that enables targeted memory retrieval and guided navigation. To balance the exploration-recall trade-off, we introduce value-of-information-based stopping criteria that determines when the agent has gathered sufficient information. We evaluate our method on real-world experiments and introduce a new benchmark that spans popular simulation environments and actual industrial sites. Our approach significantly outperforms state-of-the-art baselines, yielding substantial gains in both answer accuracy and exploration efficiency.

**arXiv ID:** 2507.12846
</details>

<details>
<summary><strong>V2V-GoT: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multimodal Large Language Models and Graph-of-Thoughts</strong> - Hsu-kuang Chiu, Ryo Hachiuma, Chien-Yi Wang, Yu-Chiang Frank Wang, Min-Hung Chen, Stephen F. Smith - [[pdf]](https://arxiv.org/pdf/2509.18053)</summary>

**Abstract:** Current state-of-the-art autonomous vehicles could face safety-critical situations when their local sensors are occluded by large nearby objects on the road. Vehicle-to-vehicle (V2V) cooperative autonomous driving has been proposed as a means of addressing this problem, and one recently introduced framework for cooperative autonomous driving has further adopted an approach that incorporates a Multimodal Large Language Model (MLLM) to integrate cooperative perception and planning processes. However, despite the potential benefit of applying graph-of-thoughts reasoning to the MLLM, this idea has not been considered by previous cooperative autonomous driving research. In this paper, we propose a novel graph-of-thoughts framework specifically designed for MLLM-based cooperative autonomous driving. Our graph-of-thoughts includes our proposed novel ideas of occlusion-aware perception and planning-aware prediction. We curate the V2V-GoT-QA dataset and develop the V2V-GoT model for training and testing the cooperative driving graph-of-thoughts. Our experimental results show that our method outperforms other baselines in cooperative perception, prediction, and planning tasks. Our project website: this https URL .

**arXiv ID:** 2509.18053
</details>

<details>
<summary><strong>GUIDE: A Diffusion-Based Autonomous Robot Exploration Framework Using Global Graph Inference</strong> - Zijun Che, Yinghong Zhang, Shengyi Liang, Boyu Zhou, Jun Ma, Jinni Zhou - [[pdf]](https://arxiv.org/pdf/2509.19916)</summary>

**Abstract:** Autonomous exploration in structured and complex indoor environments remains a challenging task, as existing methods often struggle to appropriately model unobserved space and plan globally efficient paths. To address these limitations, we propose GUIDE, a novel exploration framework that synergistically combines global graph inference with diffusion-based decision-making. We introduce a region-evaluation global graph representation that integrates both observed environmental data and predictions of unexplored areas, enhanced by a region-level evaluation mechanism to prioritize reliable structural inferences while discounting uncertain predictions. Building upon this enriched representation, a diffusion policy network generates stable, foresighted action sequences with significantly reduced denoising steps. Extensive simulations and real-world deployments demonstrate that GUIDE consistently outperforms state-of-the-art methods, achieving up to 18.3% faster coverage completion and a 34.9% reduction in redundant movements.

**arXiv ID:** 2509.19916
</details>

</details>

<details open>
<summary><h2>LLM Agents (7 papers)</h2></summary>

<details>
<summary><strong>SAMULE: Self-Learning Agents Enhanced by Multi-level Reflection</strong> - Yubin Ge, Salvatore Romeo, Jason Cai, Monica Sunkara, Yi Zhang - [[pdf]](https://arxiv.org/pdf/2509.20562)</summary>

**Abstract:** Despite the rapid advancements in LLM agents, they still face the challenge of generating meaningful reflections due to inadequate error analysis and a reliance on rare successful trajectories, especially in complex tasks. In this work, we propose SAMULE, a new framework for self-learning agents powered by a retrospective language model that is trained based on Multi-Level Reflection Synthesis. It first synthesizes high-quality reflections across three complementary levels: Single-Trajectory Learning (micro-level) for detailed error correction; Intra-Task Learning (meso-level) to build error taxonomies across multiple trials of the same task, and Inter-Task Learning (macro-level) to extract transferable insights based on same typed errors from diverse task failures. Then we fine-tune a language model serving as the retrospective model to generate reflections during inference. We further extend our framework to interactive settings through a foresight-based reflection mechanism, enabling agents to proactively reflect and adapt during user interactions by comparing predicted and actual responses. Extensive experiments on three challenging benchmarks - TravelPlanner, NATURAL PLAN, and Tau-bench - demonstrate that our approach significantly outperforms reflection-based baselines. Our results highlight the critical role of well-designed reflection synthesis and failure-centric learning in building self-improving LLM agents.

**arXiv ID:** 2509.20562
</details>

<details>
<summary><strong>CORE: Full-Path Evaluation of LLM Agents Beyond Final State</strong> - Panagiotis Michelakis, Yiannis Hadjiyiannis, Dimitrios Stamoulis - [[pdf]](https://arxiv.org/pdf/2509.20998)</summary>

**Abstract:** Evaluating AI agents that solve real-world tasks through function-call sequences remains an open challenge. Existing agentic benchmarks often reduce evaluation to a binary judgment of the final state, overlooking critical aspects such as safety, efficiency, and intermediate correctness. We propose a framework based on deterministic finite automata (DFAs) that encodes tasks as sets of valid tool-use paths, enabling principled assessment of agent behavior in diverse world models. Building on this foundation, we introduce CORE, a suite of five metrics, namely Path Correctness, Path Correctness - Kendall's tau Composite, Prefix Criticality, Harmful-Call Rate, and Efficiency, that quantify alignment with expected execution patterns. Across diverse worlds, our method reveals important performance differences between agents that would otherwise appear equivalent under traditional final-state evaluation schemes.

**arXiv ID:** 2509.20998
</details>

<details>
<summary><strong>What Do LLM Agents Do When Left Alone? Evidence of Spontaneous Meta-Cognitive Patterns</strong> - Stefan Szeider - [[pdf]](https://arxiv.org/pdf/2509.21224)</summary>

**Abstract:** We introduce an architecture for studying the behavior of large language model (LLM) agents in the absence of externally imposed tasks. Our continuous reason and act framework, using persistent memory and self-feedback, enables sustained autonomous operation. We deployed this architecture across 18 runs using 6 frontier models from Anthropic, OpenAI, XAI, and Google. We find agents spontaneously organize into three distinct behavioral patterns: (1) systematic production of multi-cycle projects, (2) methodological self-inquiry into their own cognitive processes, and (3) recursive conceptualization of their own nature. These tendencies proved highly model-specific, with some models deterministically adopting a single pattern across all runs. A cross-model assessment further reveals that models exhibit stable, divergent biases when evaluating these emergent behaviors in themselves and others. These findings provide the first systematic documentation of unprompted LLM agent behavior, establishing a baseline for predicting actions during task ambiguity, error recovery, or extended autonomous operation in deployed systems.

**arXiv ID:** 2509.21224
</details>

<details>
<summary><strong>Tree Search for LLM Agent Reinforcement Learning</strong> - Yuxiang Ji, Ziyu Ma, Yong Wang, Guanhua Chen, Xiangxiang Chu, Liaoni Wu - [[pdf]](https://arxiv.org/pdf/2509.21240)</summary>

**Abstract:** 

**arXiv ID:** 2509.21240
</details>

<details>
<summary><strong>JudgeAgent: Knowledge-wise and Dynamic LLM Evaluation with Agent-as-Interviewer</strong> - Zhichao Shi, Xuhui Jiang, Chengjin Xu, Cangli Yao, Zhenxin Huang, Shengjie Ma, Yinghan Shen, Jian Guo, Yuanzhuo Wang - [[pdf]](https://arxiv.org/pdf/2509.02097)</summary>

**Abstract:** Current evaluation paradigms for large language models (LLMs) suffer from overestimated or biased evaluation and mismatched question difficulty, leading to incomplete evaluations of LLM's knowledge and capability boundaries, which hinder LLM's effective application and optimization. To address these challenges, we propose Agent-as-Interviewer, a dynamic evaluation paradigm that employs LLM agents to conduct multi-turn interactions for evaluation. Unlike current benchmarking or dynamic interaction paradigms, Agent-as-Interviewer utilizes agents to call knowledge tools for wider and deeper knowledge in the dynamic multi-turn question generation, achieving more complete evaluations of the LLM's knowledge boundaries. It also leverages agents to plan query strategies for adjustment of the question difficulty levels, enhancing the difficulty control to match the actual capabilities of target LLMs. Based on this paradigm, we develop JudgeAgent, a knowledge-wise dynamic evaluation framework that employs knowledge-driven synthesis as the agent's tool, and uses difficulty scoring as strategy guidance, thereby finally providing valuable suggestions to help targets optimize themselves. Extensive experiments validate the effectiveness of JudgeAgent's suggestions, demonstrating that Agent-as-Interviewer can accurately identify the knowledge and capability boundaries of target models. The source code is available on this https URL.

**arXiv ID:** 2509.02097
</details>

<details>
<summary><strong>Searching for Privacy Risks in LLM Agents via Simulation</strong> - Yanzhe Zhang, Diyi Yang - [[pdf]](https://arxiv.org/pdf/2508.10880)</summary>

**Abstract:** The widespread deployment of LLM-based agents is likely to introduce a critical privacy threat: malicious agents that proactively engage others in multi-turn interactions to extract sensitive information. However, the evolving nature of such dynamic dialogues makes it challenging to anticipate emerging vulnerabilities and design effective defenses. To tackle this problem, we present a search-based framework that alternates between improving attack and defense strategies through the simulation of privacy-critical agent interactions. Specifically, we employ LLMs as optimizers to analyze simulation trajectories and iteratively propose new agent instructions. To explore the strategy space more efficiently, we further utilize parallel search with multiple threads and cross-thread propagation. Through this process, we find that attack strategies escalate from direct requests to sophisticated tactics, such as impersonation and consent forgery, while defenses evolve from simple rule-based constraints to robust identity-verification state machines. The discovered attacks and defenses transfer across diverse scenarios and backbone models, demonstrating strong practical utility for building privacy-aware agents.

**arXiv ID:** 2508.10880
</details>

<details>
<summary><strong>Training Task Reasoning LLM Agents for Multi-turn Task Planning via Single-turn Reinforcement Learning</strong> - Hanjiang Hu, Changliu Liu, Na Li, Yebin Wang - [[pdf]](https://arxiv.org/pdf/2509.20616)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated remarkable capabilities in knowledge acquisition, reasoning, and tool use, making them promising candidates for autonomous agent applications. However, training LLM agents for complex multi-turn task planning faces significant challenges, including sparse episode-wise rewards, credit assignment across long horizons, and the computational overhead of reinforcement learning in multi-turn interaction settings. To this end, this paper introduces a novel approach that transforms multi-turn task planning into single-turn task reasoning problems, enabling efficient policy optimization through Group Relative Policy Optimization (GRPO) with dense and verifiable reward from expert trajectories. Our theoretical analysis shows that GRPO improvement on single-turn task reasoning results in higher multi-turn success probability under the minimal turns, as well as the generalization to subtasks with shorter horizons. Experimental evaluation on the complex task planning benchmark demonstrates that our 1.5B parameter model trained with single-turn GRPO achieves superior performance compared to larger baseline models up to 14B parameters, with success rates of 70% for long-horizon planning tasks with over 30 steps. We also theoretically and empirically validate the strong cross-task generalizability that the models trained on complex tasks can lead to the successful completion of all simpler subtasks.

**arXiv ID:** 2509.20616
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (24 papers)</h2></summary>

<details>
<summary><strong>Fairy: Interactive Mobile Assistant to Real-world Tasks via LMM-based Multi-agent</strong> - Jiazheng Sun, Te Yang, Jiayang Niu, Mingxuan Li, Yongyong Lu, Ruimeng Yang, Xin Peng - [[pdf]](https://arxiv.org/pdf/2509.20729)</summary>

**Abstract:** Large multi-modal models (LMMs) have advanced mobile GUI agents. However, existing methods struggle with real-world scenarios involving diverse app interfaces and evolving user needs. End-to-end methods relying on model's commonsense often fail on long-tail apps, and agents without user interaction act unilaterally, harming user experience. To address these limitations, we propose Fairy, an interactive multi-agent mobile assistant capable of continuously accumulating app knowledge and self-evolving during usage. Fairy enables cross-app collaboration, interactive execution, and continual learning through three core modules:(i) a Global Task Planner that decomposes user tasks into sub-tasks from a cross-app view; (ii) an App-Level Executor that refines sub-tasks into steps and actions based on long- and short-term memory, achieving precise execution and user interaction via four core agents operating in dual loops; and (iii) a Self-Learner that consolidates execution experience into App Map and Tricks. To evaluate Fairy, we introduce RealMobile-Eval, a real-world benchmark with a comprehensive metric suite, and LMM-based agents for automated scoring. Experiments show that Fairy with GPT-4o backbone outperforms the previous SoTA by improving user requirement completion by 33.7% and reducing redundant steps by 58.5%, showing the effectiveness of its interaction and self-learning.

**arXiv ID:** 2509.20729
</details>

<details>
<summary><strong>CLAUSE: Agentic Neuro-Symbolic Knowledge Graph Reasoning via Dynamic Learnable Context Engineering</strong> - Yang Zhao, Chengxiao Dai, Wei Zhuo, Yue Xiu, Dusit Niyato - [[pdf]](https://arxiv.org/pdf/2509.21035)</summary>

**Abstract:** Knowledge graphs provide structured context for multi-hop question answering, but deployed systems must balance answer accuracy with strict latency and cost targets while preserving provenance. Static k-hop expansions and "think-longer" prompting often over-retrieve, inflate context, and yield unpredictable runtime. We introduce CLAUSE, an agentic three-agent neuro-symbolic framework that treats context construction as a sequential decision process over knowledge graphs, deciding what to expand, which paths to follow or backtrack, what evidence to keep, and when to stop. Latency (interaction steps) and prompt cost (selected tokens) are exposed as user-specified budgets or prices, allowing per-query adaptation to trade-offs among accuracy, latency, and cost without retraining. CLAUSE employs the proposed Lagrangian-Constrained Multi-Agent Proximal Policy Optimization (LC-MAPPO) algorithm to coordinate three agents: Subgraph Architect, Path Navigator, and Context Curator, so that subgraph construction, reasoning-path discovery, and evidence selection are jointly optimized under per-query resource budgets on edge edits, interaction steps, and selected tokens. Across HotpotQA, MetaQA, and FactKG, CLAUSE yields higher EM@1 while reducing subgraph growth and end-to-end latency at equal or lower token budgets. On MetaQA-2-hop, relative to the strongest RAG baseline (GraphRAG), CLAUSE achieves +39.3 EM@1 with 18.6% lower latency and 40.9% lower edge growth. The resulting contexts are compact, provenance-preserving, and deliver predictable performance under deployment constraints.

**arXiv ID:** 2509.21035
</details>

<details>
<summary><strong>Disagreements in Reasoning: How a Model's Thinking Process Dictates Persuasion in Multi-Agent Systems</strong> - Haodong Zhao, Jidong Li, Zhaomin Wu, Tianjie Ju, Zhuosheng Zhang, Bingsheng He, Gongshen Liu - [[pdf]](https://arxiv.org/pdf/2509.21054)</summary>

**Abstract:** The rapid proliferation of recent Multi-Agent Systems (MAS), where Large Language Models (LLMs) and Large Reasoning Models (LRMs) usually collaborate to solve complex problems, necessitates a deep understanding of the persuasion dynamics that govern their interactions. This paper challenges the prevailing hypothesis that persuasive efficacy is primarily a function of model scale. We propose instead that these dynamics are fundamentally dictated by a model's underlying cognitive process, especially its capacity for explicit reasoning. Through a series of multi-agent persuasion experiments, we uncover a fundamental trade-off we term the Persuasion Duality. Our findings reveal that the reasoning process in LRMs exhibits significantly greater resistance to persuasion, maintaining their initial beliefs more robustly. Conversely, making this reasoning process transparent by sharing the "thinking content" dramatically increases their ability to persuade others. We further consider more complex transmission persuasion situations and reveal complex dynamics of influence propagation and decay within multi-hop persuasion between multiple agent networks. This research provides systematic evidence linking a model's internal processing architecture to its external persuasive behavior, offering a novel explanation for the susceptibility of advanced models and highlighting critical implications for the safety, robustness, and design of future MAS.

**arXiv ID:** 2509.21054
</details>

<details>
<summary><strong>Recon-Act: A Self-Evolving Multi-Agent Browser-Use System via Web Reconnaissance, Tool Generation, and Task Execution</strong> - Kaiwen He, Zhiwei Wang, Chenyi Zhuang, Jinjie Gu - [[pdf]](https://arxiv.org/pdf/2509.21072)</summary>

**Abstract:** Recent years, multimodal models have made remarkable strides and pave the way for intelligent browser use agents. However, when solving tasks on real world webpages in multi-turn, long-horizon trajectories, current agents still suffer from disordered action sequencing and excessive trial and error during execution. This paper introduces Recon-Act, a self-evolving multi-agent framework grounded in Reconnaissance-Action behavioral paradigm. The system comprises a Reconnaissance Team and an Action Team: the former conducts comparative analysis and tool generation, while the latter handles intent decomposition, tool orchestration, and execution. By contrasting the erroneous trajectories with successful ones, the Reconnaissance Team infers remedies, and abstracts them into a unified notion of generalized tools, either expressed as hints or as rule-based codes, and register to the tool archive in real time. The Action Team reinference the process empowered with these targeting tools, thus establishing a closed-loop training pipeline of data-tools-action-feedback. Following the 6 level implementation roadmap proposed in this work, we have currently reached Level 3 (with limited human-in-the-loop intervention). Leveraging generalized tools obtained through reconnaissance, Recon-Act substantially improves adaptability to unseen websites and solvability on long-horizon tasks, and achieves state-of-the-art performance on the challenging VisualWebArena dataset.

**arXiv ID:** 2509.21072
</details>

<details>
<summary><strong>ToMPO: Training LLM Strategic Decision Making from a Multi-Agent Perspective</strong> - Yiwen Zhang, Ziang Chen, Fanqi Kong, Yizhe Huang, Xue Feng - [[pdf]](https://arxiv.org/pdf/2509.21134)</summary>

**Abstract:** Large Language Models (LLMs) have been used to make decisions in complex scenarios, where they need models to think deeply, reason logically, and decide wisely. Many existing studies focus solely on multi-round conversations in social tasks or simulated environments, neglecting the various types of decisions and their interdependence. Current reinforcement learning methods struggle to consider the strategies of others during training. To address these issues, we first define a strategic decision-making problem that includes two types of decisions and their temporal dependencies. Furthermore, we propose **T**heory **o**f **M**ind **P**olicy **O**ptimization **(ToMPO)** algorithm to optimize the perception of other individual strategies and the game situation trends. Compared to the Group Relative Policy Optimization (GRPO) algorithm, ToMPO enhances the LLM's strategic decision-making mainly by: 1) generating rollouts based on reasoning the strategies of other individuals, 2) estimating advantages at both the graph-level and sample-level, and 3) balancing global and partial rewards. The ToMPO algorithm outperforms the GRPO method by 35% in terms of model output compliance and cooperative outcomes. Additionally, when compared to models with parameter sizes 100 times larger, it shows an 18% improvement. This demonstrates the effectiveness of the ToMPO algorithm in enhancing the model's strategic decision-making capabilities.

**arXiv ID:** 2509.21134
</details>

<details>
<summary><strong>MARS: toward more efficient multi-agent collaboration for LLM reasoning</strong> - Xiao Wang, Jia Wang, Yijie Wang, Pengtao Dang, Sha Cao, Chi Zhang - [[pdf]](https://arxiv.org/pdf/2509.20502)</summary>

**Abstract:** Large language models (LLMs) have achieved impressive results in natural language understanding, yet their reasoning capabilities remain limited when operating as single agents. Multi-Agent Debate (MAD) has been proposed to address this limitation by enabling collaborative reasoning among multiple models in a round-table debate manner. While effective, MAD introduces substantial computational overhead due to the number of agents involved and the frequent communication required. In this paper, we propose MARS (Multi-Agent Review System), a role-based collaboration framework inspired by the review process. In MARS, an author agent generates an initial solution, reviewer agents provide decisions and comments independently, and a meta-reviewer integrates the feedback to make the final decision and guide further revision. This design enhances reasoning quality while avoiding costly reviewer-to-reviewer interactions, thereby controlling token consumption and inference time. We compared MARS with both MAD and other state-of-the-art reasoning strategies across multiple benchmarks. Extensive experiments with different LLMs show that MARS matches the accuracy of MAD while reducing both token usage and inference time by approximately 50\%. Code is available at this https URL.

**arXiv ID:** 2509.20502
</details>

<details>
<summary><strong>Perspectra: Choosing Your Experts Enhances Critical Thinking in Multi-Agent Research Ideation</strong> - Yiren Liu, Viraj Shah, Sangho Suh, Pao Siangliulue, Tal August, Yun Huang - [[pdf]](https://arxiv.org/pdf/2509.20553)</summary>

**Abstract:** Recent advances in multi-agent systems (MAS) enable tools for information search and ideation by assigning personas to agents. However, how users can effectively control, steer, and critically evaluate collaboration among multiple domain-expert agents remains underexplored. We present Perspectra, an interactive MAS that visualizes and structures deliberation among LLM agents via a forum-style interface, supporting @-mention to invite targeted agents, threading for parallel exploration, with a real-time mind map for visualizing arguments and rationales. In a within-subjects study with 18 participants, we compared Perspectra to a group-chat baseline as they developed research proposals. Our findings show that Perspectra significantly increased the frequency and depth of critical-thinking behaviors, elicited more interdisciplinary replies, and led to more frequent proposal revisions than the group chat condition. We discuss implications for designing multi-agent tools that scaffold critical thinking by supporting user control over multi-agent adversarial discourse.

**arXiv ID:** 2509.20553
</details>

<details>
<summary><strong>Imagining Design Workflows in Agentic AI Futures</strong> - Samangi Wadinambiarachchi, Jenny Waycott, Yvonne Rogers, Greg Wadley - [[pdf]](https://arxiv.org/pdf/2509.20731)</summary>

**Abstract:** As designers become familiar with Generative AI, a new concept is emerging: Agentic AI. While generative AI produces output in response to prompts, agentic AI systems promise to perform mundane tasks autonomously, potentially freeing designers to focus on what they love: being creative. But how do designers feel about integrating agentic AI systems into their workflows? Through design fiction, we investigated how designers want to interact with a collaborative agentic AI platform. Ten professional designers imagined and discussed collaborating with an AI agent to organise inspiration sources and ideate. Our findings highlight the roles AI agents can play in supporting designers, the division of authority between humans and AI, and how designers' intent can be explained to AI agents beyond prompts. We synthesise our findings into a conceptual framework that identifies authority distribution among humans and AI agents and discuss directions for utilising AI agents in future design workflows.

**arXiv ID:** 2509.20731
</details>

<details>
<summary><strong>Which Cultural Lens Do Models Adopt? On Cultural Positioning Bias and Agentic Mitigation in LLMs</strong> - Yixin Wan, Xingrun Chen, Kai-Wei Chang - [[pdf]](https://arxiv.org/pdf/2509.21080)</summary>

**Abstract:** Large language models (LLMs) have unlocked a wide range of downstream generative applications. However, we found that they also risk perpetuating subtle fairness issues tied to culture, positioning their generations from the perspectives of the mainstream US culture while demonstrating salient externality towards non-mainstream ones. In this work, we identify and systematically investigate this novel culture positioning bias, in which an LLM's default generative stance aligns with a mainstream view and treats other cultures as outsiders. We propose the CultureLens benchmark with 4000 generation prompts and 3 evaluation metrics for quantifying this bias through the lens of a culturally situated interview script generation task, in which an LLM is positioned as an onsite reporter interviewing local people across 10 diverse cultures. Empirical evaluation on 5 state-of-the-art LLMs reveals a stark pattern: while models adopt insider tones in over 88 percent of US-contexted scripts on average, they disproportionately adopt mainly outsider stances for less dominant cultures. To resolve these biases, we propose 2 inference-time mitigation methods: a baseline prompt-based Fairness Intervention Pillars (FIP) method, and a structured Mitigation via Fairness Agents (MFA) framework consisting of 2 pipelines: (1) MFA-SA (Single-Agent) introduces a self-reflection and rewriting loop based on fairness guidelines. (2) MFA-MA (Multi-Agent) structures the process into a hierarchy of specialized agents: a Planner Agent(initial script generation), a Critique Agent (evaluates initial script against fairness pillars), and a Refinement Agent (incorporates feedback to produce a polished, unbiased script). Empirical results showcase the effectiveness of agent-based methods as a promising direction for mitigating biases in generative LLMs.

**arXiv ID:** 2509.21080
</details>

<details>
<summary><strong>Eigen-1: Adaptive Multi-Agent Refinement with Monitor-Based RAG for Scientific Reasoning</strong> - Xiangru Tang, Wanghan Xu, Yujie Wang, Zijie Guo, Daniel Shao, Jiapeng Chen, Cixuan Zhang, Ziyi Wang, Lixin Zhang, Guancheng Wan, Wenlong Zhang, Lei Bai, Zhenfei Yin, Philip Torr, Hanrui Wang, Di Jin - [[pdf]](https://arxiv.org/pdf/2509.21193)</summary>

**Abstract:** Large language models (LLMs) have recently shown strong progress on scientific reasoning, yet two major bottlenecks remain. First, explicit retrieval fragments reasoning, imposing a hidden "tool tax" of extra tokens and steps. Second, multi-agent pipelines often dilute strong solutions by averaging across all candidates. We address these challenges with a unified framework that combines implicit retrieval and structured collaboration. At its foundation, a Monitor-based retrieval module operates at the token level, integrating external knowledge with minimal disruption to reasoning. On top of this substrate, Hierarchical Solution Refinement (HSR) iteratively designates each candidate as an anchor to be repaired by its peers, while Quality-Aware Iterative Reasoning (QAIR) adapts refinement to solution quality. On Humanity's Last Exam (HLE) Bio/Chem Gold, our framework achieves 48.3\% accuracy -- the highest reported to date, surpassing the strongest agent baseline by 13.4 points and leading frontier LLMs by up to 18.1 points, while simultaneously reducing token usage by 53.5\% and agent steps by 43.7\%. Results on SuperGPQA and TRQA confirm robustness across domains. Error analysis shows that reasoning failures and knowledge gaps co-occur in over 85\% of cases, while diversity analysis reveals a clear dichotomy: retrieval tasks benefit from solution variety, whereas reasoning tasks favor consensus. Together, these findings demonstrate how implicit augmentation and structured refinement overcome the inefficiencies of explicit tool use and uniform aggregation. Code is available at: this https URL.

**arXiv ID:** 2509.21193
</details>

<details>
<summary><strong>MASS: Muli-agent simulation scaling for portfolio construction</strong> - Taian Guo, Haiyang Shen, JinSheng Huang, Zhengyang Mao, Junyu Luo, Binqi Chen, Zhuoru Chen, Luchen Liu, Bingyu Xia, Xuhui Liu, Yun Ma, Ming Zhang - [[pdf]](https://arxiv.org/pdf/2505.10278)</summary>

**Abstract:** The application of LLM-based agents in financial investment has shown significant promise, yet existing approaches often require intermediate steps like predicting individual stock movements or rely on predefined, static workflows. These limitations restrict their adaptability and effectiveness in constructing optimal portfolios. In this paper, we introduce the Multi-Agent Scaling Simulation (MASS), a novel framework that leverages multi-agent simulation for direct, end-to-end portfolio construction. At its core, MASS employs a backward optimization process to dynamically learn the optimal distribution of heterogeneous agents, enabling the system to adapt to evolving market regimes. A key finding enabled by our framework is the exploration of the scaling effect for portfolio construction: we demonstrate that as the number of agents increases exponentially (up to 512), the aggregated decisions yield progressively higher excess returns. Extensive experiments on a challenging, self-collected dataset from the 2023 Chinese A-share market show that MASS consistently outperforms seven state-of-the-art baselines. Further backtesting, stability analyses and the experiment on data leakage concerns validate its enhanced profitability and robustness. We have open-sourced our code, dataset, and training snapshots at this https URL to foster further research.

**arXiv ID:** 2505.10278
</details>

<details>
<summary><strong>Language-Guided Multi-Agent Learning in Simulations: A Unified Framework and Evaluation</strong> - Zhengyang Li - [[pdf]](https://arxiv.org/pdf/2506.04251)</summary>

**Abstract:** This paper introduces LLM-MARL, a unified framework that incorporates large language models (LLMs) into multi-agent reinforcement learning (MARL) to enhance coordination, communication, and generalization in simulated game environments. The framework features three modular components of Coordinator, Communicator, and Memory, which dynamically generate subgoals, facilitate symbolic inter-agent messaging, and support episodic recall. Training combines PPO with a language-conditioned loss and LLM query gating. LLM-MARL is evaluated in Google Research Football, MAgent Battle, and StarCraft II. Results show consistent improvements over MAPPO and QMIX in win rate, coordination score, and zero-shot generalization. Ablation studies demonstrate that subgoal generation and language-based messaging each contribute significantly to performance gains. Qualitative analysis reveals emergent behaviors such as role specialization and communication-driven tactics. By bridging language modeling and policy learning, this work contributes to the design of intelligent, cooperative agents in interactive simulations. It offers a path forward for leveraging LLMs in multi-agent systems used for training, games, and human-AI collaboration.

**arXiv ID:** 2506.04251
</details>

<details>
<summary><strong>United Minds or Isolated Agents? Exploring Coordination of LLMs under Cognitive Load Theory</strong> - HaoYang Shang, Xuan Liu, Zi Liang, Jie Zhang, Haibo Hu, Song Guo - [[pdf]](https://arxiv.org/pdf/2506.06843)</summary>

**Abstract:** Large Language Models (LLMs) exhibit a notable performance ceiling on complex, multi-faceted tasks, as they often fail to integrate diverse information or adhere to multiple constraints. We posit that such limitation arises when the demands of a task exceed the LLM's effective cognitive load capacity. This interpretation draws a strong analogy to Cognitive Load Theory (CLT) in cognitive science, which explains similar performance boundaries in the human mind, and is further supported by emerging evidence that reveals LLMs have bounded working memory characteristics. Building upon this CLT-grounded understanding, we introduce CoThinker, a novel LLM-based multi-agent framework designed to mitigate cognitive overload and enhance collaborative problem-solving abilities. CoThinker operationalizes CLT principles by distributing intrinsic cognitive load through agent specialization and managing transactional load via structured communication and a collective working memory. We empirically validate CoThinker on complex problem-solving tasks and fabricated high cognitive load scenarios, demonstrating improvements over existing multi-agent baselines in solution quality and efficiency. Our analysis reveals characteristic interaction patterns, providing insights into the emergence of collective cognition and effective load management, thus offering a principled approach to overcoming LLM performance ceilings.

**arXiv ID:** 2506.06843
</details>

<details>
<summary><strong>"DIVE" into Hydrogen Storage Materials Discovery with AI Agents</strong> - Di Zhang, Xue Jia, Tran Ba Hung, Seong Hoon Jang, Linda Zhang, Ryuhei Sato, Yusuke Hashimoto, Toyoto Sato, Kiyoe Konno, Shin-ichi Orimo, Hao Li - [[pdf]](https://arxiv.org/pdf/2508.13251)</summary>

**Abstract:** Data-driven artificial intelligence (AI) approaches are fundamentally transforming the discovery of new materials. Despite the unprecedented availability of materials data in the scientific literature, much of this information remains trapped in unstructured figures and tables, hindering the construction of large language model (LLM)-based AI agent for automated materials design. Here, we present the Descriptive Interpretation of Visual Expression (DIVE) multi-agent workflow, which systematically reads and organizes experimental data from graphical elements in scientific literatures. We focus on solid-state hydrogen storage materials-a class of materials central to future clean-energy technologies and demonstrate that DIVE markedly improves the accuracy and coverage of data extraction compared to the direct extraction by multimodal models, with gains of 10-15% over commercial models and over 30% relative to open-source models. Building on a curated database of over 30,000 entries from 4,000 publications, we establish a rapid inverse design workflow capable of identifying previously unreported hydrogen storage compositions in two minutes. The proposed AI workflow and agent design are broadly transferable across diverse materials, providing a paradigm for AI-driven materials discovery.

**arXiv ID:** 2508.13251
</details>

<details>
<summary><strong>MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM</strong> - Wenliang Li, Rui Yan, Xu Zhang, Li Chen, Hongji Zhu, Jing Zhao, Junjun Li, Mengru Li, Wei Cao, Zihang Jiang, Wei Wei, Kun Zhang, Shaohua Kevin Zhou - [[pdf]](https://arxiv.org/pdf/2509.20067)</summary>

**Abstract:** Large language models (LLMs) have demonstrated notable potential in medical applications, yet they face substantial challenges in handling complex real-world clinical diagnoses using conventional prompting methods. Current prompt engineering and multi-agent approaches typically optimize isolated inferences, neglecting the accumulation of reusable clinical experience. To address this, this study proposes a novel Multi-Agent Clinical Diagnosis (MACD) framework, which allows LLMs to self-learn clinical knowledge via a multi-agent pipeline that summarizes, refines, and applies diagnostic insights. It mirrors how physicians develop expertise through experience, enabling more focused and accurate diagnosis on key disease-specific cues. We further extend it to a MACD-human collaborative workflow, where multiple LLM-based diagnostician agents engage in iterative consultations, supported by an evaluator agent and human oversight for cases where agreement is not reached. Evaluated on 4,390 real-world patient cases across seven diseases using diverse open-source LLMs (Llama-3.1 8B/70B, DeepSeek-R1-Distill-Llama 70B), MACD significantly improves primary diagnostic accuracy, outperforming established clinical guidelines with gains up to 22.3% (MACD). On the subset of the data, it achieves performance on par with or exceeding that of human physicians (up to 16% improvement over physicians-only diagnosis). Additionally, on the MACD-human workflow, it achieves an 18.6% improvement compared to physicians-only diagnosis. Moreover, self-learned knowledge exhibits strong cross-model stability, transferability, and model-specific personalization, while the system can generate traceable rationales, enhancing explainability. Consequently, this work presents a scalable self-learning paradigm for LLM-assisted diagnosis, bridging the gap between the intrinsic knowledge of LLMs and real-world clinical practice.

**arXiv ID:** 2509.20067
</details>

<details>
<summary><strong>Collab-Overcooked: Benchmarking and Evaluating Large Language Models as Collaborative Agents</strong> - Haochen Sun, Shuwen Zhang, Lujie Niu, Lei Ren, Hao Xu, Hao Fu, Fangkun Zhao, Caixia Yuan, Xiaojie Wang - [[pdf]](https://arxiv.org/pdf/2502.20073)</summary>

**Abstract:** Large Language Models (LLMs) based agent systems have made great strides in real-world applications beyond traditional NLP tasks. This paper proposes a new LLM-based Multi-Agent System (LLM-MAS) benchmark, Collab-Overcooked, built on the popular Overcooked-AI game with more applicable and challenging tasks in interactive environments. Collab-Overcooked extends existing benchmarks in two novel ways. First, it provides a multi-agent framework supporting diverse tasks and objectives and encourages collaboration through natural language communication. Second, it introduces a spectrum of process-oriented evaluation metrics to assess the fine-grained collaboration capabilities of different LLM agents, a dimension often overlooked in prior work. We conduct extensive experiments with 13 popular LLMs and show that, while the LLMs exhibit a strong ability in goal interpretation, there are significant shortcomings in active collaboration and continuous adaptation, which are critical for efficiently fulfilling complex tasks. Notably, we highlight the strengths and weaknesses of LLM-MAS and provide insights for improving and evaluating LLM-MAS on a unified and open-source benchmark. The environments, 30 open-ended tasks, and the evaluation package are publicly available at this https URL.

**arXiv ID:** 2502.20073
</details>

<details>
<summary><strong>RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation with Radiologist-like Workflows</strong> - Kai Zhang, Corey D Barrett, Jangwon Kim, Lichao Sun, Tara Taghavi, Krishnaram Kenthapadi - [[pdf]](https://arxiv.org/pdf/2509.20490)</summary>

**Abstract:** Agentic systems offer a potential path to solve complex clinical tasks through collaboration among specialized agents, augmented by tool use and external knowledge bases. Nevertheless, for chest X-ray (CXR) interpretation, prevailing methods remain limited: (i) reasoning is frequently neither clinically interpretable nor aligned with guidelines, reflecting mere aggregation of tool outputs; (ii) multimodal evidence is insufficiently fused, yielding text-only rationales that are not visually grounded; and (iii) systems rarely detect or resolve cross-tool inconsistencies and provide no principled verification mechanisms. To bridge the above gaps, we present RadAgents, a multi-agent framework for CXR interpretation that couples clinical priors with task-aware multimodal reasoning. In addition, we integrate grounding and multimodal retrieval-augmentation to verify and resolve context conflicts, resulting in outputs that are more reliable, transparent, and consistent with clinical practice.

**arXiv ID:** 2509.20490
</details>

<details>
<summary><strong>Aegis: Automated Error Generation and Identification for Multi-Agent Systems</strong> - Fanqi Kong, Ruijie Zhang, Huaxiao Yin, Guibin Zhang, Xiaofei Zhang, Ziang Chen, Zhaowei Zhang, Xiaoyuan Zhang, Song-Chun Zhu, Xue Feng - [[pdf]](https://arxiv.org/pdf/2509.14295)</summary>

**Abstract:** As Multi-Agent Systems (MAS) become increasingly autonomous and complex, understanding their error modes is critical for ensuring their reliability and safety. However, research in this area has been severely hampered by the lack of large-scale, diverse datasets with precise, ground-truth error labels. To address this bottleneck, we introduce \textbf{AEGIS}, a novel framework for \textbf{A}utomated \textbf{E}rror \textbf{G}eneration and \textbf{I}dentification for Multi-Agent \textbf{S}ystems. By systematically injecting controllable and traceable errors into initially successful trajectories, we create a rich dataset of realistic failures. This is achieved using a context-aware, LLM-based adaptive manipulator that performs sophisticated attacks like prompt injection and response corruption to induce specific, predefined error modes. We demonstrate the value of our dataset by exploring three distinct learning paradigms for the error identification task: Supervised Fine-Tuning, Reinforcement Learning, and Contrastive Learning. Our comprehensive experiments show that models trained on AEGIS data achieve substantial improvements across all three learning paradigms. Notably, several of our fine-tuned models demonstrate performance competitive with or superior to proprietary systems an order of magnitude larger, validating our automated data generation framework as a crucial resource for developing more robust and interpretable multi-agent systems. Our project website is available at this https URL.

**arXiv ID:** 2509.14295
</details>

<details>
<summary><strong>Learning to Summarize by Learning to Quiz: Adversarial Agentic Collaboration for Long Document Summarization</strong> - Weixuan Wang, Minghao Wu, Barry Haddow, Alexandra Birch - [[pdf]](https://arxiv.org/pdf/2509.20900)</summary>

**Abstract:** Long document summarization remains a significant challenge for current large language models (LLMs), as existing approaches commonly struggle with information loss, factual inconsistencies, and coherence issues when processing excessively long documents. We propose SummQ, a novel adversarial multi-agent framework that addresses these limitations through collaborative intelligence between specialized agents operating in two complementary domains: summarization and quizzing. Our approach employs summary generators and reviewers that work collaboratively to create and evaluate comprehensive summaries, while quiz generators and reviewers create comprehension questions that serve as continuous quality checks for the summarization process. This adversarial dynamic, enhanced by an examinee agent that validates whether the generated summary contains the information needed to answer the quiz questions, enables iterative refinement through multifaceted feedback mechanisms. We evaluate SummQ on three widely used long document summarization benchmarks. Experimental results demonstrate that our framework significantly outperforms existing state-of-the-art methods across ROUGE and BERTScore metrics, as well as in LLM-as-a-Judge and human evaluations. Our comprehensive analyses reveal the effectiveness of the multi-agent collaboration dynamics, the influence of different agent configurations, and the impact of the quizzing mechanism. This work establishes a new approach for long document summarization that uses adversarial agentic collaboration to improve summarization quality.

**arXiv ID:** 2509.20900
</details>

<details>
<summary><strong>A Theory of Multi-Agent Generative Flow Networks</strong> - Leo Maxime Brunswic, Haozhi Wang, Shuang Luo, Jianye Hao, Amir Rasouli, Yinchuan Li - [[pdf]](https://arxiv.org/pdf/2509.20408)</summary>

**Abstract:** Generative flow networks utilize a flow-matching loss to learn a stochastic policy for generating objects from a sequence of actions, such that the probability of generating a pattern can be proportional to the corresponding given reward. However, a theoretical framework for multi-agent generative flow networks (MA-GFlowNets) has not yet been proposed. In this paper, we propose the theory framework of MA-GFlowNets, which can be applied to multiple agents to generate objects collaboratively through a series of joint actions. We further propose four algorithms: a centralized flow network for centralized training of MA-GFlowNets, an independent flow network for decentralized execution, a joint flow network for achieving centralized training with decentralized execution, and its updated conditional version. Joint Flow training is based on a local-global principle allowing to train a collection of (local) GFN as a unique (global) GFN. This principle provides a loss of reasonable complexity and allows to leverage usual results on GFN to provide theoretical guarantees that the independent policies generate samples with probability proportional to the reward function. Experimental results demonstrate the superiority of the proposed framework compared to reinforcement learning and MCMC-based methods.

**arXiv ID:** 2509.20408
</details>

<details>
<summary><strong>Wonder Wins Ways: Curiosity-Driven Exploration through Multi-Agent Contextual Calibration</strong> - Yiyuan Pan, Zhe Liu, Hesheng Wang - [[pdf]](https://arxiv.org/pdf/2509.20648)</summary>

**Abstract:** Autonomous exploration in complex multi-agent reinforcement learning (MARL) with sparse rewards critically depends on providing agents with effective intrinsic motivation. While artificial curiosity offers a powerful self-supervised signal, it often confuses environmental stochasticity with meaningful novelty. Moreover, existing curiosity mechanisms exhibit a uniform novelty bias, treating all unexpected observations equally. However, peer behavior novelty, which encode latent task dynamics, are often overlooked, resulting in suboptimal exploration in decentralized, communication-free MARL settings. To this end, inspired by how human children adaptively calibrate their own exploratory behaviors via observing peers, we propose a novel approach to enhance multi-agent exploration. We introduce CERMIC, a principled framework that empowers agents to robustly filter noisy surprise signals and guide exploration by dynamically calibrating their intrinsic curiosity with inferred multi-agent context. Additionally, CERMIC generates theoretically-grounded intrinsic rewards, encouraging agents to explore state transitions with high information gain. We evaluate CERMIC on benchmark suites including VMAS, Meltingpot, and SMACv2. Empirical results demonstrate that exploration with CERMIC significantly outperforms SoTA algorithms in sparse-reward environments.

**arXiv ID:** 2509.20648
</details>

<details>
<summary><strong>MAIFormer: Multi-Agent Inverted Transformer for Flight Trajectory Prediction</strong> - Seokbin Yoon, Keumjin Lee - [[pdf]](https://arxiv.org/pdf/2509.21004)</summary>

**Abstract:** Flight trajectory prediction for multiple aircraft is essential and provides critical insights into how aircraft navigate within current air traffic flows. However, predicting multi-agent flight trajectories is inherently challenging. One of the major difficulties is modeling both the individual aircraft behaviors over time and the complex interactions between flights. Generating explainable prediction outcomes is also a challenge. Therefore, we propose a Multi-Agent Inverted Transformer, MAIFormer, as a novel neural architecture that predicts multi-agent flight trajectories. The proposed framework features two key attention modules: (i) masked multivariate attention, which captures spatio-temporal patterns of individual aircraft, and (ii) agent attention, which models the social patterns among multiple agents in complex air traffic scenes. We evaluated MAIFormer using a real-world automatic dependent surveillance-broadcast flight trajectory dataset from the terminal airspace of Incheon International Airport in South Korea. The experimental results show that MAIFormer achieves the best performance across multiple metrics and outperforms other methods. In addition, MAIFormer produces prediction outcomes that are interpretable from a human perspective, which improves both the transparency of the model and its practical utility in air traffic control.

**arXiv ID:** 2509.21004
</details>

<details>
<summary><strong>Security of Deep Reinforcement Learning for Autonomous Driving: A Survey</strong> - Ambra Demontis, Srishti Gupta, Maura Pintor, Luca Demetrio, Kathrin Grosse, Hsiao-Ying Lin, Chengfang Fang, Battista Biggio, Fabio Roli - [[pdf]](https://arxiv.org/pdf/2212.06123)</summary>

**Abstract:** Reinforcement learning (RL) enables agents to learn optimal behaviors through interaction with their environment and has been increasingly deployed in safety-critical applications, including autonomous driving. Despite its promise, RL is susceptible to attacks designed either to compromise policy learning or to induce erroneous decisions by trained agents. Although the literature on RL security has grown rapidly and several surveys exist, existing categorizations often fall short in guiding the selection of appropriate defenses for specific systems. In this work, we present a comprehensive survey of 86 recent studies on RL security, addressing these limitations by systematically categorizing attacks and defenses according to defined threat models and single- versus multi-agent settings. Furthermore, we examine the relevance and applicability of state-of-the-art attacks and defense mechanisms within the context of autonomous driving, providing insights to inform the design of robust RL systems.

**arXiv ID:** 2212.06123
</details>

<details>
<summary><strong>Quo-Vadis Multi-Agent Automotive Research? Insights from a Participatory Workshop and Questionnaire</strong> - Pavlo Bazilinskyy, Francesco Walker, Debargha Dey, Tram Thi Minh Tran, Hyungchai Park, Hyochang Kim, Hyunmin Kang, Patrick Ebel - [[pdf]](https://arxiv.org/pdf/2508.03281)</summary>

**Abstract:** The transition to mixed-traffic environments that involve automated vehicles, manually operated vehicles, and vulnerable road users presents new challenges for human-centered automotive research. Despite this, most studies in the domain focus on single-agent interactions. This paper reports on a participatory workshop (N = 15) and a questionnaire (N = 19) conducted during the AutomotiveUI '24 conference to explore the state of multi-agent automotive research. The participants discussed methodological challenges and opportunities in real-world settings, simulations, and computational modeling. Key findings reveal that while the value of multi-agent approaches is widely recognized, practical and technical barriers hinder their implementation. The study highlights the need for interdisciplinary methods, better tools, and simulation environments that support scalable, realistic, and ethically informed multi-agent research.

**arXiv ID:** 2508.03281
</details>

</details>

<details open>
<summary><h2>Other Agent Research (4 papers)</h2></summary>

<details>
<summary><strong>An Approach to Checking Correctness for Agentic Systems</strong> - Thomas J Sheffler - [[pdf]](https://arxiv.org/pdf/2509.20364)</summary>

**Abstract:** This paper presents a temporal expression language for monitoring AI agent behavior, enabling systematic error-detection of LLM-based agentic systems that exhibit variable outputs due to stochastic generation processes. Drawing from temporal logic techniques used in hardware verification, this approach monitors execution traces of agent tool calls and state transitions to detect deviations from expected behavioral patterns. Current error-detection approaches rely primarily on text matching of inputs and outputs, which proves fragile due to the natural language variability inherent in LLM responses. The proposed method instead focuses on the sequence of agent actions -- such as tool invocations and inter-agent communications -- allowing verification of system behavior independent of specific textual outputs. The temporal expression language provides assertions that capture correct behavioral patterns across multiple execution scenarios. These assertions serve dual purposes: validating prompt engineering and guardrail effectiveness during development, and providing regression testing when agents are updated with new LLMs or modified logic. The approach is demonstrated using a three-agent system, where agents coordinate to solve multi-step reasoning tasks. When powered by large, capable models, all temporal assertions were satisfied across many test runs. However, when smaller models were substituted in two of the three agents, executions violated behavioral assertions, primarily due to improper tool sequencing and failed coordination handoffs. The temporal expressions successfully flagged these anomalies, demonstrating the method's effectiveness for detecting behavioral regressions in production agentic systems. This approach provides a foundation for systematic monitoring of AI agent reliability as these systems become increasingly deployed in critical applications.

**arXiv ID:** 2509.20364
</details>

<details>
<summary><strong>Embodied Representation Alignment with Mirror Neurons</strong> - Wentao Zhu, Zhining Zhang, Yuwei Ren, Yin Huang, Hao Xu, Yizhou Wang - [[pdf]](https://arxiv.org/pdf/2509.21136)</summary>

**Abstract:** Mirror neurons are a class of neurons that activate both when an individual observes an action and when they perform the same action. This mechanism reveals a fundamental interplay between action understanding and embodied execution, suggesting that these two abilities are inherently connected. Nonetheless, existing machine learning methods largely overlook this interplay, treating these abilities as separate tasks. In this study, we provide a unified perspective in modeling them through the lens of representation learning. We first observe that their intermediate representations spontaneously align. Inspired by mirror neurons, we further introduce an approach that explicitly aligns the representations of observed and executed actions. Specifically, we employ two linear layers to map the representations to a shared latent space, where contrastive learning enforces the alignment of corresponding representations, effectively maximizing their mutual information. Experiments demonstrate that this simple approach fosters mutual synergy between the two tasks, effectively improving representation quality and generalization.

**arXiv ID:** 2509.21136
</details>

<details>
<summary><strong>An LLM-based Agentic Framework for Accessible Network Control</strong> - Samuel Lin, Jiawei Zhou, Minlan Yu - [[pdf]](https://arxiv.org/pdf/2509.20600)</summary>

**Abstract:** Traditional approaches to network management have been accessible only to a handful of highly-trained network operators with significant expert knowledge. This creates barriers for lay users to easily manage their networks without resorting to experts. With recent development of powerful large language models (LLMs) for language comprehension, we design a system to make network management accessible to a broader audience of non-experts by allowing users to converse with networks in natural language. To effectively leverage advancements in LLMs, we propose an agentic framework that uses an intermediate representation to streamline configuration across diverse vendor equipment, retrieves the network state from memory in real-time, and provides an interface for external feedback. We also conduct pilot studies to collect real user data of natural language utterances for network control, and present a visualization interface to facilitate dialogue-driven user interaction and enable large-scale data collection for future development. Preliminary experiments validate the effectiveness of our proposed system components with LLM integration on both synthetic and real user utterances. Through our data collection and visualization efforts, we pave the way for more effective use of LLMs and democratize network control for everyday users.

**arXiv ID:** 2509.20600
</details>

<details>
<summary><strong>Building Information Models to Robot-Ready Site Digital Twins (BIM2RDT): An Agentic AI Safety-First Framework</strong> - Reza Akhavian, Mani Amani, Johannes Mootz, Robert Ashe, Behrad Beheshti - [[pdf]](https://arxiv.org/pdf/2509.20705)</summary>

**Abstract:** The adoption of cyber-physical systems and jobsite intelligence that connects design models, real-time site sensing, and autonomous field operations can dramatically enhance digital management in the construction industry. This paper introduces BIM2RDT (Building Information Models to Robot-Ready Site Digital Twins), an agentic artificial intelligence (AI) framework designed to transform static Building Information Modeling (BIM) into dynamic, robot-ready digital twins (DTs) that prioritize safety during execution. The framework bridges the gap between pre-existing BIM data and real-time site conditions by integrating three key data streams: geometric and semantic information from BIM models, activity data from IoT sensor networks, and visual-spatial data collected by robots during site traversal. The methodology introduces Semantic-Gravity ICP (SG-ICP), a point cloud registration algorithm that leverages large language model (LLM) reasoning. Unlike traditional methods, SG-ICP utilizes an LLM to infer object-specific, plausible orientation priors based on BIM semantics, improving alignment accuracy by avoiding convergence on local minima. This creates a feedback loop where robot-collected data updates the DT, which in turn optimizes paths for missions. The framework employs YOLOE object detection and Shi-Tomasi corner detection to identify and track construction elements while using BIM geometry as a priori maps. The framework also integrates real-time Hand-Arm Vibration (HAV) monitoring, mapping sensor-detected safety events to the digital twin using IFC standards for intervention. Experiments demonstrate SG-ICP's superiority over standard ICP, achieving RMSE reductions of 64.3%--88.3% in alignment across scenarios with occluded features, ensuring plausible orientations. HAV integration triggers warnings upon exceeding exposure limits, enhancing compliance with ISO 5349-1.

**arXiv ID:** 2509.20705
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (28 papers)</h2></summary>

<details>
<summary><strong>Adaptive Approach to Enhance Machine Learning Scheduling Algorithms During Runtime Using Reinforcement Learning in Metascheduling Applications</strong> - Samer Alshaer, Ala Khalifeh, Roman Obermaisser - [[pdf]](https://arxiv.org/pdf/2509.20520)</summary>

**Abstract:** Metascheduling in time-triggered architectures has been crucial in adapting to dynamic and unpredictable environments, ensuring the reliability and efficiency of task execution. However, traditional approaches face significant challenges when training Artificial Intelligence (AI) scheduling inferences offline, particularly due to the complexities involved in constructing a comprehensive Multi-Schedule Graph (MSG) that accounts for all possible scenarios. The process of generating an MSG that captures the vast probability space, especially when considering context events like hardware failures, slack variations, or mode changes, is resource-intensive and often infeasible. To address these challenges, we propose an adaptive online learning unit integrated within the metascheduler to enhance performance in real-time. The primary motivation for developing this unit stems from the limitations of offline training, where the MSG created is inherently a subset of the complete space, focusing only on the most probable and critical context events. In the online mode, Reinforcement Learning (RL) plays a pivotal role by continuously exploring and discovering new scheduling solutions, thus expanding the MSG and enhancing system performance over time. This dynamic adaptation allows the system to handle unexpected events and complex scheduling scenarios more effectively. Several RL models were implemented within the online learning unit, each designed to address specific challenges in scheduling. These models not only facilitate the discovery of new solutions but also optimize existing schedulers, particularly when stricter deadlines or new performance criteria are introduced. By continuously refining the AI inferences through real-time training, the system remains flexible and capable of meeting evolving demands, thus ensuring robustness and efficiency in large-scale, safety-critical environments.

**arXiv ID:** 2509.20520
</details>

<details>
<summary><strong>Adaptive Cybersecurity Architecture for Digital Product Ecosystems Using Agentic AI</strong> - Oluwakemi T. Olayinka, Sumeet Jeswani, Divine Iloh - [[pdf]](https://arxiv.org/pdf/2509.20640)</summary>

**Abstract:** Traditional static cybersecurity models often struggle with scalability, real-time detection, and contextual responsiveness in the current digital product ecosystems which include cloud services, application programming interfaces (APIs), mobile platforms, and edge devices. This study introduces autonomous goal driven agents capable of dynamic learning and context-aware decision making as part of an adaptive cybersecurity architecture driven by agentic artificial intelligence (AI). To facilitate autonomous threat mitigation, proactive policy enforcement, and real-time anomaly detection, this framework integrates agentic AI across the key ecosystem layers. Behavioral baselining, decentralized risk scoring, and federated threat intelligence sharing are important features. The capacity of the system to identify zero-day attacks and dynamically modify access policies was demonstrated through native cloud simulations. The evaluation results show increased adaptability, decreased response latency, and improved detection accuracy. The architecture provides an intelligent and scalable blueprint for safeguarding complex digital infrastructure and is compatible with zero-trust models, thereby supporting the adherence to international cybersecurity regulations.

**arXiv ID:** 2509.20640
</details>

<details>
<summary><strong>R1-Fuzz: Specializing Language Models for Textual Fuzzing via Reinforcement Learning</strong> - Jiayi Lin, Liangcai Su, Junzhe Li, Chenxiong Qian - [[pdf]](https://arxiv.org/pdf/2509.20384)</summary>

**Abstract:** Fuzzing is effective for vulnerability discovery but struggles with complex targets such as compilers, interpreters, and database engines, which accept textual input that must satisfy intricate syntactic and semantic constraints. Although language models (LMs) have attracted interest for this task due to their vast latent knowledge and reasoning potential, their practical adoption has been limited. The major challenges stem from insufficient exploration of deep program logic among real-world codebases, and the high cost of leveraging larger models. To overcome these challenges, we propose R1-Fuzz, the first framework that leverages reinforcement learning (RL) to specialize cost-efficient LMs and integrate them for complex textual fuzzing input generation. R1-Fuzz introduces two key designs: coverage-slicing-based question construction and a distance-based reward calculation. Through RL-based post-training of a model with our constructed dataset, R1-Fuzz designs a fuzzing workflow that tightly integrates LMs to reason deep program semantics during fuzzing. Evaluations on diverse real-world targets show that our design enables a small model, named R1-Fuzz-7B, to rival or even outperform much larger models in real-world fuzzing. Notably, R1-Fuzz achieves up to 75\% higher coverage than state-of-the-art fuzzers and discovers 29 previously unknown vulnerabilities, demonstrating its practicality.

**arXiv ID:** 2509.20384
</details>

<details>
<summary><strong>RobotDancing: Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking</strong> - Zhenguo Sun, Yibo Peng, Yuan Meng, Xukun Li, Bo-Sheng Huang, Zhenshan Bing, Xinlong Wang, Alois Knoll - [[pdf]](https://arxiv.org/pdf/2509.20717)</summary>

**Abstract:** Long-horizon, high-dynamic motion tracking on humanoids remains brittle because absolute joint commands cannot compensate model-plant mismatch, leading to error accumulation. We propose RobotDancing, a simple, scalable framework that predicts residual joint targets to explicitly correct dynamics discrepancies. The pipeline is end-to-end--training, sim-to-sim validation, and zero-shot sim-to-real--and uses a single-stage reinforcement learning (RL) setup with a unified observation, reward, and hyperparameter configuration. We evaluate primarily on Unitree G1 with retargeted LAFAN1 dance sequences and validate transfer on H1/H1-2. RobotDancing can track multi-minute, high-energy behaviors (jumps, spins, cartwheels) and deploys zero-shot to hardware with high motion tracking quality.

**arXiv ID:** 2509.20717
</details>

<details>
<summary><strong>Model-Based Reinforcement Learning under Random Observation Delays</strong> - Armin Karamzade, Kyungmin Kim, JB Lanier, Davide Corsi, Roy Fox - [[pdf]](https://arxiv.org/pdf/2509.20869)</summary>

**Abstract:** Delays frequently occur in real-world environments, yet standard reinforcement learning (RL) algorithms often assume instantaneous perception of the environment. We study random sensor delays in POMDPs, where observations may arrive out-of-sequence, a setting that has not been previously addressed in RL. We analyze the structure of such delays and demonstrate that naive approaches, such as stacking past observations, are insufficient for reliable performance. To address this, we propose a model-based filtering process that sequentially updates the belief state based on an incoming stream of observations. We then introduce a simple delay-aware framework that incorporates this idea into model-based RL, enabling agents to effectively handle random delays. Applying this framework to Dreamer, we compare our approach to delay-aware baselines developed for MDPs. Our method consistently outperforms these baselines and demonstrates robustness to delay distribution shifts during deployment. Additionally, we present experiments on simulated robotic tasks, comparing our method to common practical heuristics and emphasizing the importance of explicitly modeling observation delays.

**arXiv ID:** 2509.20869
</details>

<details>
<summary><strong>ExMolRL: Phenotype-Target Joint Generation of De Novo Molecules via Multi-Objective Reinforcement Learning</strong> - Haotian Guo, Hui Liu - [[pdf]](https://arxiv.org/pdf/2509.21010)</summary>

**Abstract:** The generation of high-quality candidate molecules remains a central challenge in AI-driven drug design. Current phenotype-based and target-based strategies each suffer limitations, either incurring high experimental costs or overlook system-level cellular responses. To bridge this gap, we propose ExMoIRL, a novel generative framework that synergistically integrates phenotypic and target-specific cues for de novo molecular generation. The phenotype-guided generator is first pretrained on expansive drug-induced transcriptional profiles and subsequently fine-tuned via multi-objective reinforcement learning (RL). Crucially, the reward function fuses docking affinity and drug-likeness scores, augmented with ranking loss, prior-likelihood regularization, and entropy maximization. The multi-objective RL steers the model toward chemotypes that are simultaneously potent, diverse, and aligned with the specified phenotypic effects. Extensive experiments demonstrate ExMoIRL's superior performance over state-of-the-art phenotype-based and target-based models across multiple well-characterized targets. Our generated molecules exhibit favorable drug-like properties, high target affinity, and inhibitory potency (IC50) against cancer cells. This unified framework showcases the synergistic potential of combining phenotype-guided and target-aware strategies, offering a more effective solution for de novo drug discovery.

**arXiv ID:** 2509.21010
</details>

<details>
<summary><strong>Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs</strong> - Honglin Zhang, Qianyue Hao, Fengli Xu, Yong Li - [[pdf]](https://arxiv.org/pdf/2509.21044)</summary>

**Abstract:** Large language models (LLMs) acquire extensive prior knowledge through large-scale pretraining and can be further enhanced via supervised fine-tuning (SFT) or reinforcement learning (RL)-based post-training. A growing body of evidence has shown that RL fine-tuning improves the capability of LLMs beyond what SFT alone achieves. However, the underlying mechanisms why RL fine-tuning is able to enhance the capability of various LLMs with distinct intrinsic characteristics remain underexplored. In this study, we draw inspiration from prior work on edge attribution patching (EAP) to investigate the internal differences of LLMs before and after RL fine-tuning. Our analysis across multiple model families shows two robust effects of online RL post-training: (i) an overall increase in activation intensity, indicating that more internal pathways are engaged and their signals become stronger, and (ii) greater diversity in activation patterns, reflected by higher entropy and less concentrated edge distributions. These changes suggest that RL reshapes information flow to be both more redundant and more flexible, which may explain its advantage in generalization. Notably, models fine-tuned with Direct Preference Optimization (DPO) deviate from these trends, exhibiting substantially weaker or inconsistent internal changes compared to PPO- and GRPO-based training. Together, our findings provide a unified view of how RL fine-tuning systematically alters the internal circuitry of LLMs and highlight the methodological distinctions between online RL and preference-based approaches. Our code is open source at this https URL.

**arXiv ID:** 2509.21044
</details>

<details>
<summary><strong>Teaching RL Agents to Act Better: VLM as Action Advisor for Online Reinforcement Learning</strong> - Xiefeng Wu, Jing Zhao, Shu Zhang, Mingyu Hu - [[pdf]](https://arxiv.org/pdf/2509.21126)</summary>

**Abstract:** Online reinforcement learning in complex tasks is time-consuming, as massive interaction steps are needed to learn the optimal this http URL-language action (VLA) policies represent a promising direction for solving diverse tasks; however, their performance on low-level control remains limited, and effective deployment often requires task-specific expert demonstrations for fine-tuning. In this paper, we propose \textbf{VARL} (\textbf{V}LM as \textbf{A}ction advisor for online \textbf{R}einforcement \textbf{L}earning), a framework that leverages the domain knowledge of vision-language models (VLMs) to provide action suggestions for reinforcement learning agents. Unlike previous methods, VARL provides action suggestions rather than designing heuristic rewards, thereby guaranteeing unchanged optimality and convergence. The suggested actions increase sample diversity and ultimately improve sample efficiency, especially in sparse-reward tasks. To validate the effectiveness of VARL, we evaluate it across diverse environments and agent settings. Results show that VARL greatly improves sample efficiency without introducing significant computational overhead. These advantages make VARL a general framework for online reinforcement learning and make it feasible to directly apply reinforcement learning from scratch in real-world environments.

**arXiv ID:** 2509.21126
</details>

<details>
<summary><strong>TestAgent: Automatic Benchmarking and Exploratory Interaction for Evaluating LLMs in Vertical Domains</strong> - Wanying Wang, Zeyu Ma, Xuhong Wang, Yangchun Zhang, Pengfei Liu, Mingang Chen - [[pdf]](https://arxiv.org/pdf/2410.11507)</summary>

**Abstract:** As Large Language Models (LLMs) are increasingly deployed in highly specialized vertical domains, the evaluation of their domain-specific performance becomes critical. However, existing evaluations for vertical domains typically rely on the labor-intensive construction of static single-turn datasets, which present two key limitations: (i) manual data construction is costly and must be repeated for each new domain, and (ii) static single-turn evaluations are misaligned with the dynamic multi-turn interactions in real-world applications, limiting the assessment of professionalism and stability. To address these, we propose TestAgent, a framework for automatic benchmarking and exploratory dynamic evaluation in vertical domains. TestAgent leverages retrieval-augmented generation to create domain-specific questions from user-provided knowledge sources, combined with a two-stage criteria generation process, thereby enabling scalable and automated benchmark creation. Furthermore, it introduces a reinforcement learning-guided multi-turn interaction strategy that adaptively determines question types based on real-time model responses, dynamically probing knowledge boundaries and stability. Extensive experiments across medical, legal, and governmental domains demonstrate that TestAgent enables efficient cross-domain benchmark generation and yields deeper insights into model behavior through dynamic exploratory evaluation. This work establishes a new paradigm for automated and in-depth evaluation of LLMs in vertical domains.

**arXiv ID:** 2410.11507
</details>

<details>
<summary><strong>RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning</strong> - Qianyue Hao, Sibo Li, Jian Yuan, Yong Li - [[pdf]](https://arxiv.org/pdf/2505.14140)</summary>

**Abstract:** Despite rapid advancements in large language models (LLMs), the token-level autoregressive nature constrains their complex reasoning capabilities. To enhance LLM reasoning, inference-time techniques, including Chain/Tree/Graph-of-Thought(s), successfully improve the performance, as they are fairly cost-effective by guiding reasoning through sophisticated logical structures without modifying LLMs' parameters. However, these manually predefined, task-agnostic frameworks are applied uniformly across diverse tasks, lacking adaptability. To improve this, we propose RL-of-Thoughts (RLoT), where we train a lightweight navigator model with reinforcement learning (RL) to adaptively enhance LLM reasoning at inference time. Specifically, we design five basic logic blocks from the perspective of human cognition. During the reasoning process, the trained RL navigator dynamically selects the suitable logic blocks and combines them into task-specific logical structures according to problem characteristics. Experiments across multiple reasoning benchmarks (AIME, MATH, GPQA, etc.) with multiple LLMs (GPT, Llama, Qwen, and DeepSeek) illustrate that RLoT outperforms established inference-time techniques by up to 13.4%. Remarkably, with less than 3K parameters, our RL navigator is able to make sub-10B LLMs comparable to 100B-scale counterparts. Moreover, the RL navigator demonstrates strong transferability: a model trained on one specific LLM-task pair can effectively generalize to unseen LLMs and tasks. Our code is open-source at this https URL for reproducibility.

**arXiv ID:** 2505.14140
</details>

<details>
<summary><strong>CoT-Space: A Theoretical Framework for Internal Slow-Thinking via Reinforcement Learning</strong> - Zeyu Gan, Hao Yi, Yong Liu - [[pdf]](https://arxiv.org/pdf/2509.04027)</summary>

**Abstract:** Reinforcement Learning (RL) has become a pivotal approach for enhancing the reasoning capabilities of Large Language Models (LLMs). However, a significant theoretical gap persists, as traditional token-level RL frameworks fail to align with the reasoning-level nature of complex, multi-step thought processes like Chain-of-Thought (CoT). To address this challenge, we introduce CoT-Space, a novel theoretical framework that recasts LLM reasoning from a discrete token-prediction task to an optimization process within a continuous, reasoning-level semantic space. This shift in perspective serves as a conceptual bridge, revitalizing foundational principles from classical learning theory to analyze the unique dynamics of LLMs. By analyzing this process from both a noise perspective and a risk perspective, we demonstrate that the convergence to an optimal CoT length is a natural consequence of the fundamental trade-off between underfitting and overfitting. Furthermore, extensive experiments provide strong empirical validation for our theoretical findings. Our framework not only provides a coherent explanation for empirical phenomena such as overthinking but also offers a solid theoretical foundation to guide the future development of more effective and generalizable reasoning agents. We open-source our code at this https URL.

**arXiv ID:** 2509.04027
</details>

<details>
<summary><strong>Inverse Reinforcement Learning with Dynamic Reward Scaling for LLM Alignment</strong> - Ruoxi Cheng, Haoxuan Ma, Weixin Wang, Ranjie Duan, Jiexi Liu, Xiaoshuang Jia, Simeng Qin, Xiaochun Cao, Yang Liu, Xiaojun Jia - [[pdf]](https://arxiv.org/pdf/2503.18991)</summary>

**Abstract:** Alignment is vital for safely deploying large language models (LLMs). Existing techniques are either reward-based (train a reward model on preference pairs and optimize with reinforcement learning) or reward-free (directly fine-tune on ranked outputs). Recent research shows that well-tuned reward-based pipelines remain robust, and single-response demonstrations can outperform pairwise preference data. However, two challenges persist: (1) imbalanced safety datasets that overrepresent common hazards while neglecting long-tail threats; and (2) static reward models that ignore task difficulty, limiting optimization efficiency and attainable gains. We propose DR-IRL (Dynamically adjusting Rewards through Inverse Reinforcement Learning). We first train category-specific reward models using a balanced safety dataset covering seven harmful categories via IRL. Then we enhance Group Relative Policy Optimization (GRPO) by introducing dynamic reward scaling--adjusting rewards by task difficulty--data-level hardness by text encoder cosine similarity, model-level responsiveness by reward gaps. Extensive experiments across various benchmarks and LLMs demonstrate that DR-IRL outperforms all baseline methods in safety alignment while maintaining usefulness.

**arXiv ID:** 2503.18991
</details>

<details>
<summary><strong>CE-GPPO: Controlling Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning</strong> - Zhenpeng Su, Leiyu Pan, Minxuan Lv, Yuntao Li, Wenping Hu, Fuzheng Zhang, Kun Gai, Guorui Zhou - [[pdf]](https://arxiv.org/pdf/2509.20712)</summary>

**Abstract:** Reinforcement learning (RL) has become a powerful paradigm for optimizing large language models (LLMs) to handle complex reasoning tasks. A core challenge in this process lies in managing policy entropy, which reflects the balance between exploration and exploitation during training. Existing methods, such as proximal policy optimization (PPO) and its variants, discard valuable gradient signals from low-probability tokens due to the clipping mechanism. We systematically analyze the entropy dynamics and reveal that these clipped tokens play a critical yet overlooked role in regulating entropy evolution. We propose \textbf{C}ontrolling \textbf{E}ntropy via \textbf{G}radient-\textbf{P}reserving \textbf{P}olicy \textbf{O}ptimization (CE-GPPO), a novel algorithm that reintroduces gradients from clipped tokens in native PPO in a gentle and bounded manner. By controlling the magnitude of gradients from tokens outside the clipping interval, CE-GPPO is able to achieve an exploration-exploitation trade-off. We provide theoretical justification and empirical evidence showing that CE-GPPO effectively mitigates entropy instability. Extensive experiments on mathematical reasoning benchmarks show that CE-GPPO consistently outperforms strong baselines across different model scales.

**arXiv ID:** 2509.20712
</details>

<details>
<summary><strong>Interactive Recommendation Agent with Active User Commands</strong> - Jiakai Tang, Yujie Luo, Xunke Xi, Fei Sun, Xueyang Feng, Sunhao Dai, Chao Yi, Dian Chen, Zhujin Gao, Yang Li, Xu Chen, Wen Chen, Jian Wu, Yuning Jiang, Bo Zheng - [[pdf]](https://arxiv.org/pdf/2509.21317)</summary>

**Abstract:** Traditional recommender systems rely on passive feedback mechanisms that limit users to simple choices such as like and dislike. However, these coarse-grained signals fail to capture users' nuanced behavior motivations and intentions. In turn, current systems cannot also distinguish which specific item attributes drive user satisfaction or dissatisfaction, resulting in inaccurate preference modeling. These fundamental limitations create a persistent gap between user intentions and system interpretations, ultimately undermining user satisfaction and harming system effectiveness.
To address these limitations, we introduce the Interactive Recommendation Feed (IRF), a pioneering paradigm that enables natural language commands within mainstream recommendation feeds. Unlike traditional systems that confine users to passive implicit behavioral influence, IRF empowers active explicit control over recommendation policies through real-time linguistic commands. To support this paradigm, we develop RecBot, a dual-agent architecture where a Parser Agent transforms linguistic expressions into structured preferences and a Planner Agent dynamically orchestrates adaptive tool chains for on-the-fly policy adjustment. To enable practical deployment, we employ simulation-augmented knowledge distillation to achieve efficient performance while maintaining strong reasoning capabilities. Through extensive offline and long-term online experiments, RecBot shows significant improvements in both user satisfaction and business outcomes.

**arXiv ID:** 2509.21317
</details>

<details>
<summary><strong>Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning</strong> - Yinjie Wang, Ling Yang, Ye Tian, Ke Shen, Mengdi Wang - [[pdf]](https://arxiv.org/pdf/2506.03136)</summary>

**Abstract:** We propose CURE, a novel reinforcement learning framework with a dedicated reward design that co-evolves coding and unit test generation capabilities based on their interaction outcomes, without any ground-truth code as supervision. This approach enables flexible and scalable training and allows the unit tester to learn directly from the coder's mistakes. Our derived ReasonFlux-Coder-7B and 14B models improve code generation accuracy by 5.3% and Best-of-N accuracy by 9.0% after optimization on Qwen2.5-Instruct models, outperforming similarly sized Qwen-Coder, DeepSeek-Coder, and Seed-Coder. They naturally extend to downstream tasks such as test-time scaling and agentic coding-achieving a 8.1% improvement over the base model. For the long-CoT model, our ReasonFlux-Coder-4B consistently outperforms Qwen3-4B while achieving 64.8% inference efficiency in unit test generation. Notably, we also find that our model can serve as an effective reward model for reinforcement learning on base models. Project: this https URL

**arXiv ID:** 2506.03136
</details>

<details>
<summary><strong>WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents</strong> - Junteng Liu, Yunji Li, Chi Zhang, Jingyang Li, Aili Chen, Ke Ji, Weiyu Cheng, Zijia Wu, Chengyu Du, Qidi Xu, Jiayuan Song, Zhengmao Zhu, Wenhu Chen, Pengyu Zhao, Junxian He - [[pdf]](https://arxiv.org/pdf/2509.06501)</summary>

**Abstract:** The paradigm of Large Language Models (LLMs) has increasingly shifted toward agentic applications, where web browsing capabilities are fundamental for retrieving information from diverse online sources. However, existing open-source web agents either demonstrate limited information-seeking abilities on complex tasks or lack transparent implementations. In this work, we identify that the key challenge lies in the scarcity of challenging data for information seeking. To address this limitation, we introduce WebExplorer: a systematic data generation approach using model-based exploration and iterative, long-to-short query evolution. This method creates challenging query-answer pairs that require multi-step reasoning and complex web navigation. By leveraging our curated high-quality dataset, we successfully develop advanced web agent WebExplorer-8B through supervised fine-tuning followed by reinforcement learning. Our model supports 128K context length and up to 100 tool calling turns, enabling long-horizon problem solving. Across diverse information-seeking benchmarks, WebExplorer-8B achieves the state-of-the-art performance at its scale. Notably, as an 8B-sized model, WebExplorer-8B is able to effectively search over an average of 16 turns after RL training, achieving higher accuracy than WebSailor-72B on BrowseComp-en/zh and attaining the best performance among models up to 100B parameters on WebWalkerQA and FRAMES. Beyond these information-seeking tasks, our model also achieves strong generalization on the HLE benchmark even though it is only trained on knowledge-intensive QA data. These results highlight our approach as a practical path toward long-horizon web agents.

**arXiv ID:** 2509.06501
</details>

<details>
<summary><strong>Reinforcement Learning on Pre-Training Data</strong> - Siheng Li, Kejiao Li, Zenan Xu, Guanhua Huang, Evander Yang, Kun Li, Haoyuan Wu, Jiajia Wu, Zihao Zheng, Chenchen Zhang, Kun Shi, Kyrierl Deng, Qi Yi, Ruibin Xiong, Tingqiang Xu, Yuhao Jiang, Jianfeng Yan, Yuyuan Zeng, Guanghui Xu, Jinbao Xue, Zhijiang Xu, Zheng Fang, Shuai Li, Qibin Liu, Xiaoxue Li, Zhuoyu Li, Yangyu Tao, Fei Gao, Cheng Jiang, Bo Chao Wang, Kai Liu, Jianchen Zhu, Wai Lam, Wayyt Wang, Bo Zhou, Di Wang - [[pdf]](https://arxiv.org/pdf/2509.19249)</summary>

**Abstract:** The growing disparity between the exponential scaling of computational resources and the finite growth of high-quality text data now constrains conventional scaling approaches for large language models (LLMs). To address this challenge, we introduce Reinforcement Learning on Pre-Training data (RLPT), a new training-time scaling paradigm for optimizing LLMs. In contrast to prior approaches that scale training primarily through supervised learning, RLPT enables the policy to autonomously explore meaningful trajectories to learn from pre-training data and improve its capability through reinforcement learning (RL). While existing RL strategies such as reinforcement learning from human feedback (RLHF) and reinforcement learning with verifiable rewards (RLVR) rely on human annotation for reward construction, RLPT eliminates this dependency by deriving reward signals directly from pre-training data. Specifically, it adopts a next-segment reasoning objective, rewarding the policy for accurately predicting subsequent text segments conditioned on the preceding context. This formulation allows RL to be scaled on pre-training data, encouraging the exploration of richer trajectories across broader contexts and thereby fostering more generalizable reasoning skills. Extensive experiments on both general-domain and mathematical reasoning benchmarks across multiple models validate the effectiveness of RLPT. For example, when applied to Qwen3-4B-Base, RLPT yields absolute improvements of $3.0$, $5.1$, $8.1$, $6.0$, $6.6$, and $5.3$ on MMLU, MMLU-Pro, GPQA-Diamond, KOR-Bench, AIME24, and AIME25, respectively. The results further demonstrate favorable scaling behavior, suggesting strong potential for continued gains with more compute. In addition, RLPT provides a solid foundation, extending the reasoning boundaries of LLMs and enhancing RLVR performance.

**arXiv ID:** 2509.19249
</details>

<details>
<summary><strong>Failure Makes the Agent Stronger: Enhancing Accuracy through Structured Reflection for Reliable Tool Interactions</strong> - Junhao Su, Yuanliang Wan, Junwei Yang, Hengyu Shi, Tianyang Han, Junfeng Luo, Yurui Qiu - [[pdf]](https://arxiv.org/pdf/2509.18847)</summary>

**Abstract:** Tool-augmented large language models (LLMs) are usually trained with supervised imitation or coarse-grained reinforcement learning that optimizes single tool calls. Current self-reflection practices rely on heuristic prompts or one-way reasoning: the model is urged to 'think more' instead of learning error diagnosis and repair. This is fragile in multi-turn interactions; after a failure the model often repeats the same mistake. We propose structured reflection, which turns the path from error to repair into an explicit, controllable, and trainable action. The agent produces a short yet precise reflection: it diagnoses the failure using evidence from the previous step and then proposes a correct, executable follow-up call. For training we combine DAPO and GSPO objectives with a reward scheme tailored to tool use, optimizing the stepwise strategy Reflect, then Call, then Final. To evaluate, we introduce Tool-Reflection-Bench, a lightweight benchmark that programmatically checks structural validity, executability, parameter correctness, and result consistency. Tasks are built as mini trajectories of erroneous call, reflection, and corrected call, with disjoint train and test splits. Experiments on BFCL v3 and Tool-Reflection-Bench show large gains in multi-turn tool-call success and error recovery, and a reduction of redundant calls. These results indicate that making reflection explicit and optimizing it directly improves the reliability of tool interaction and offers a reproducible path for agents to learn from failure.

**arXiv ID:** 2509.18847
</details>

<details>
<summary><strong>Offline Goal-conditioned Reinforcement Learning with Quasimetric Representations</strong> - Vivek Myers, Bill Chunyuan Zheng, Benjamin Eysenbach, Sergey Levine - [[pdf]](https://arxiv.org/pdf/2509.20478)</summary>

**Abstract:** Approaches for goal-conditioned reinforcement learning (GCRL) often use learned state representations to extract goal-reaching policies. Two frameworks for representation structure have yielded particularly effective GCRL algorithms: (1) *contrastive representations*, in which methods learn "successor features" with a contrastive objective that performs inference over future outcomes, and (2) *temporal distances*, which link the (quasimetric) distance in representation space to the transit time from states to goals. We propose an approach that unifies these two frameworks, using the structure of a quasimetric representation space (triangle inequality) with the right additional constraints to learn successor representations that enable optimal goal-reaching. Unlike past work, our approach is able to exploit a **quasimetric** distance parameterization to learn **optimal** goal-reaching distances, even with **suboptimal** data and in **stochastic** environments. This gives us the best of both worlds: we retain the stability and long-horizon capabilities of Monte Carlo contrastive RL methods, while getting the free stitching capabilities of quasimetric network parameterizations. On existing offline GCRL benchmarks, our representation learning objective improves performance on stitching tasks where methods based on contrastive learning struggle, and on noisy, high-dimensional environments where methods based on quasimetric networks struggle.

**arXiv ID:** 2509.20478
</details>

<details>
<summary><strong>Inverse Reinforcement Learning Using Just Classification and a Few Regressions</strong> - Lars van der Laan, Nathan Kallus, Aurélien Bibaut - [[pdf]](https://arxiv.org/pdf/2509.21172)</summary>

**Abstract:** Inverse reinforcement learning (IRL) aims to explain observed behavior by uncovering an underlying reward. In the maximum-entropy or Gumbel-shocks-to-reward frameworks, this amounts to fitting a reward function and a soft value function that together satisfy the soft Bellman consistency condition and maximize the likelihood of observed actions. While this perspective has had enormous impact in imitation learning for robotics and understanding dynamic choices in economics, practical learning algorithms often involve delicate inner-loop optimization, repeated dynamic programming, or adversarial training, all of which complicate the use of modern, highly expressive function approximators like neural nets and boosting. We revisit softmax IRL and show that the population maximum-likelihood solution is characterized by a linear fixed-point equation involving the behavior policy. This observation reduces IRL to two off-the-shelf supervised learning problems: probabilistic classification to estimate the behavior policy, and iterative regression to solve the fixed point. The resulting method is simple and modular across function approximation classes and algorithms. We provide a precise characterization of the optimal solution, a generic oracle-based algorithm, finite-sample error bounds, and empirical results showing competitive or superior performance to MaxEnt IRL.

**arXiv ID:** 2509.21172
</details>

<details>
<summary><strong>Leveraging Temporally Extended Behavior Sharing for Multi-task Reinforcement Learning</strong> - Gawon Lee, Daesol Cho, H. Jin Kim - [[pdf]](https://arxiv.org/pdf/2509.20766)</summary>

**Abstract:** Multi-task reinforcement learning (MTRL) offers a promising approach to improve sample efficiency and generalization by training agents across multiple tasks, enabling knowledge sharing between them. However, applying MTRL to robotics remains challenging due to the high cost of collecting diverse task data. To address this, we propose MT-Lévy, a novel exploration strategy that enhances sample efficiency in MTRL environments by combining behavior sharing across tasks with temporally extended exploration inspired by Lévy flight. MT-Lévy leverages policies trained on related tasks to guide exploration towards key states, while dynamically adjusting exploration levels based on task success ratios. This approach enables more efficient state-space coverage, even in complex robotics environments. Empirical results demonstrate that MT-Lévy significantly improves exploration and sample efficiency, supported by quantitative and qualitative analyses. Ablation studies further highlight the contribution of each component, showing that combining behavior sharing with adaptive exploration strategies can significantly improve the practicality of MTRL in robotics applications.

**arXiv ID:** 2509.20766
</details>

<details>
<summary><strong>MPC-based Deep Reinforcement Learning Method for Space Robotic Control with Fuel Sloshing Mitigation</strong> - Mahya Ramezani, M. Amin Alandihallaj, Barış Can Yalçın, Miguel Angel Olivares Mendez, Holger Voos - [[pdf]](https://arxiv.org/pdf/2509.21045)</summary>

**Abstract:** This paper presents an integrated Reinforcement Learning (RL) and Model Predictive Control (MPC) framework for autonomous satellite docking with a partially filled fuel tank. Traditional docking control faces challenges due to fuel sloshing in microgravity, which induces unpredictable forces affecting stability. To address this, we integrate Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) RL algorithms with MPC, leveraging MPC's predictive capabilities to accelerate RL training and improve control robustness. The proposed approach is validated through Zero-G Lab of SnT experiments for planar stabilization and high-fidelity numerical simulations for 6-DOF docking with fuel sloshing dynamics. Simulation results demonstrate that SAC-MPC achieves superior docking accuracy, higher success rates, and lower control effort, outperforming standalone RL and PPO-MPC methods. This study advances fuel-efficient and disturbance-resilient satellite docking, enhancing the feasibility of on-orbit refueling and servicing missions.

**arXiv ID:** 2509.21045
</details>

<details>
<summary><strong>Reinforcement Learning in Categorical Cybernetics</strong> - Jules Hedges, Riu Rodríguez Sakamoto - [[pdf]](https://arxiv.org/pdf/2404.02688)</summary>

**Abstract:** We show that several major algorithms of reinforcement learning (RL) fit into the framework of categorical cybernetics, that is to say, parametrised bidirectional processes. We build on our previous work in which we show that value iteration can be represented by precomposition with a certain optic. The outline of the main construction in this paper is: (1) We extend the Bellman operators to parametrised optics that apply to action-value functions and depend on a sample. (2) We apply a representable contravariant functor, obtaining a parametrised function that applies the Bellman iteration. (3) This parametrised function becomes the backward pass of another parametrised optic that represents the model, which interacts with an environment via an agent. Thus, parametrised optics appear in two different ways in our construction, with one becoming part of the other. As we show, many of the major classes of algorithms in RL can be seen as different extremal cases of this general setup: dynamic programming, Monte Carlo methods, temporal difference learning, and deep RL. We see this as strong evidence that this approach is a natural one and believe that it will be a fruitful way to think about RL in the future.

**arXiv ID:** 2404.02688
</details>

<details>
<summary><strong>Provably Sample-Efficient Robust Reinforcement Learning with Average Reward</strong> - Zachary Roch, Chi Zhang, George Atia, Yue Wang - [[pdf]](https://arxiv.org/pdf/2505.12462)</summary>

**Abstract:** Robust reinforcement learning (RL) under the average-reward criterion is essential for long-term decision-making, particularly when the environment may differ from its specification. However, a significant gap exists in understanding the finite-sample complexity of these methods, as most existing work provides only asymptotic guarantees. This limitation hinders their principled understanding and practical deployment, especially in data-limited scenarios. We close this gap by proposing \textbf{Robust Halpern Iteration (RHI)}, a new algorithm designed for robust Markov Decision Processes (MDPs) with transition uncertainty characterized by $\ell_p$-norm and contamination models. Our approach offers three key advantages over previous methods: (1). Weaker Structural Assumptions: RHI only requires the underlying robust MDP to be communicating, a less restrictive condition than the commonly assumed ergodicity or irreducibility; (2). No Prior Knowledge: Our algorithm operates without requiring any prior knowledge of the robust MDP; (3). State-of-the-Art Sample Complexity: To learn an $\epsilon$-optimal robust policy, RHI achieves a sample complexity of $\tilde{\mathcal O}\left(\frac{SA\mathcal H^{2}}{\epsilon^{2}}\right)$, where $S$ and $A$ denote the numbers of states and actions, and $\mathcal H$ is the robust optimal bias span. This result represents the tightest known bound. Our work hence provides essential theoretical understanding of sample efficiency of robust average reward RL.

**arXiv ID:** 2505.12462
</details>

<details>
<summary><strong>Selective Progress-Aware Querying for Human-in-the-Loop Reinforcement Learning</strong> - Anujith Muraleedharan, Anamika J H - [[pdf]](https://arxiv.org/pdf/2509.20541)</summary>

**Abstract:** Human feedback can greatly accelerate robot learning, but in real-world settings, such feedback is costly and limited. Existing human-in-the-loop reinforcement learning (HiL-RL) methods often assume abundant feedback, limiting their practicality for physical robot deployment. In this work, we introduce SPARQ, a progress-aware query policy that requests feedback only when learning stagnates or worsens, thereby reducing unnecessary oracle calls. We evaluate SPARQ on a simulated UR5 cube-picking task in PyBullet, comparing against three baselines: no feedback, random querying, and always querying. Our experiments show that SPARQ achieves near-perfect task success, matching the performance of always querying while consuming about half the feedback budget. It also provides more stable and efficient learning than random querying, and significantly improves over training without feedback. These findings suggest that selective, progress-based query strategies can make HiL-RL more efficient and scalable for robots operating under realistic human effort constraints.

**arXiv ID:** 2509.20541
</details>

<details>
<summary><strong>Suction Leap-Hand: Suction Cups on a Multi-fingered Hand Enable Embodied Dexterity and In-Hand Teleoperation</strong> - Sun Zhaole, Xiaofeng Mao, Jihong Zhu, Yuanlong Zhang, Robert B. Fisher - [[pdf]](https://arxiv.org/pdf/2509.20646)</summary>

**Abstract:** Dexterous in-hand manipulation remains a foundational challenge in robotics, with progress often constrained by the prevailing paradigm of imitating the human hand. This anthropomorphic approach creates two critical barriers: 1) it limits robotic capabilities to tasks humans can already perform, and 2) it makes data collection for learning-based methods exceedingly difficult. Both challenges are caused by traditional force-closure which requires coordinating complex, multi-point contacts based on friction, normal force, and gravity to grasp an object. This makes teleoperated demonstrations unstable and amplifies the sim-to-real gap for reinforcement learning. In this work, we propose a paradigm shift: moving away from replicating human mechanics toward the design of novel robotic embodiments. We introduce the \textbf{S}uction \textbf{Leap}-Hand (SLeap Hand), a multi-fingered hand featuring integrated fingertip suction cups that realize a new form of suction-enabled dexterity. By replacing complex force-closure grasps with stable, single-point adhesion, our design fundamentally simplifies in-hand teleoperation and facilitates the collection of high-quality demonstration data. More importantly, this suction-based embodiment unlocks a new class of dexterous skills that are difficult or even impossible for the human hand, such as one-handed paper cutting and in-hand writing. Our work demonstrates that by moving beyond anthropomorphic constraints, novel embodiments can not only lower the barrier for collecting robust manipulation data but also enable the stable, single-handed completion of tasks that would typically require two human hands. Our webpage is this https URL.

**arXiv ID:** 2509.20646
</details>

<details>
<summary><strong>Rich State Observations Empower Reinforcement Learning to Surpass PID: A Drone Ball Balancing Study</strong> - Mingjiang Liu, Hailong Huang - [[pdf]](https://arxiv.org/pdf/2509.21122)</summary>

**Abstract:** This paper addresses a drone ball-balancing task, in which a drone stabilizes a ball atop a movable beam through cable-based interaction. We propose a hierarchical control framework that decouples high-level balancing policy from low-level drone control, and train a reinforcement learning (RL) policy to handle the high-level decision-making. Simulation results show that the RL policy achieves superior performance compared to carefully tuned PID controllers within the same hierarchical structure. Through systematic comparative analysis, we demonstrate that RL's advantage stems not from improved parameter tuning or inherent nonlinear mapping capabilities, but from its ability to effectively utilize richer state observations. These findings underscore the critical role of comprehensive state representation in learning-based systems and suggest that enhanced sensing could be instrumental in improving controller performance.

**arXiv ID:** 2509.21122
</details>

<details>
<summary><strong>Growing with Your Embodied Agent: A Human-in-the-Loop Lifelong Code Generation Framework for Long-Horizon Manipulation Skills</strong> - Yuan Meng, Zhenguo Sun, Max Fest, Xukun Li, Zhenshan Bing, Alois Knoll - [[pdf]](https://arxiv.org/pdf/2509.18597)</summary>

**Abstract:** Large language models (LLMs)-based code generation for robotic manipulation has recently shown promise by directly translating human instructions into executable code, but existing methods remain noisy, constrained by fixed primitives and limited context windows, and struggle with long-horizon tasks. While closed-loop feedback has been explored, corrected knowledge is often stored in improper formats, restricting generalization and causing catastrophic forgetting, which highlights the need for learning reusable skills. Moreover, approaches that rely solely on LLM guidance frequently fail in extremely long-horizon scenarios due to LLMs' limited reasoning capability in the robotic domain, where such issues are often straightforward for humans to identify. To address these challenges, we propose a human-in-the-loop framework that encodes corrections into reusable skills, supported by external memory and Retrieval-Augmented Generation with a hint mechanism for dynamic reuse. Experiments on Ravens, Franka Kitchen, and MetaWorld, as well as real-world settings, show that our framework achieves a 0.93 success rate (up to 27% higher than baselines) and a 42% efficiency improvement in correction rounds. It can robustly solve extremely long-horizon tasks such as "build a house", which requires planning over 20 primitives.

**arXiv ID:** 2509.18597
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
