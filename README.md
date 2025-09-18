# Agent arXiv Daily

**Last Updated:** 2025-09-18 02:35:55

**Total Papers:** 65

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
<summary><strong>Co-Investigator AI: The Rise of Agentic AI for Smarter, Trustworthy AML Compliance Narratives</strong> - Prathamesh Vasudeo Naik, Naresh Kumar Dintakurthi, Zhanghao Hu, Yue Wang, Robby Qiu - [[pdf]](https://arxiv.org/pdf/2509.08380)</summary>

**Abstract:** Generating regulatorily compliant Suspicious Activity Report (SAR) remains a high-cost, low-scalability bottleneck in Anti-Money Laundering (AML) workflows. While large language models (LLMs) offer promising fluency, they suffer from factual hallucination, limited crime typology alignment, and poor explainability -- posing unacceptable risks in compliance-critical domains. This paper introduces Co-Investigator AI, an agentic framework optimized to produce Suspicious Activity Reports (SARs) significantly faster and with greater accuracy than traditional methods. Drawing inspiration from recent advances in autonomous agent architectures, such as the AI Co-Scientist, our approach integrates specialized agents for planning, crime type detection, external intelligence gathering, and compliance validation. The system features dynamic memory management, an AI-Privacy Guard layer for sensitive data handling, and a real-time validation agent employing the Agent-as-a-Judge paradigm to ensure continuous narrative quality assurance. Human investigators remain firmly in the loop, empowered to review and refine drafts in a collaborative workflow that blends AI efficiency with domain expertise. We demonstrate the versatility of Co-Investigator AI across a range of complex financial crime scenarios, highlighting its ability to streamline SAR drafting, align narratives with regulatory expectations, and enable compliance teams to focus on higher-order analytical work. This approach marks the beginning of a new era in compliance reporting -- bringing the transformative benefits of AI agents to the core of regulatory processes and paving the way for scalable, reliable, and transparent SAR generation.

**arXiv ID:** 2509.08380
</details>

<details>
<summary><strong>Mining the Long Tail: A Comparative Study of Data-Centric Criticality Metrics for Robust Offline Reinforcement Learning in Autonomous Motion Planning</strong> - Antonio Guillen-Perez - [[pdf]](https://arxiv.org/pdf/2508.18397)</summary>

**Abstract:** Offline Reinforcement Learning (RL) presents a promising paradigm for training autonomous vehicle (AV) planning policies from large-scale, real-world driving logs. However, the extreme data imbalance in these logs, where mundane scenarios vastly outnumber rare "long-tail" events, leads to brittle and unsafe policies when using standard uniform data sampling. In this work, we address this challenge through a systematic, large-scale comparative study of data curation strategies designed to focus the learning process on information-rich samples. We investigate six distinct criticality weighting schemes which are categorized into three families: heuristic-based, uncertainty-based, and behavior-based. These are evaluated at two temporal scales, the individual timestep and the complete scenario. We train seven goal-conditioned Conservative Q-Learning (CQL) agents with a state-of-the-art, attention-based architecture and evaluate them in the high-fidelity Waymax simulator. Our results demonstrate that all data curation methods significantly outperform the baseline. Notably, data-driven curation using model uncertainty as a signal achieves the most significant safety improvements, reducing the collision rate by nearly three-fold (from 16.0% to 5.5%). Furthermore, we identify a clear trade-off where timestep-level weighting excels at reactive safety while scenario-level weighting improves long-horizon planning. Our work provides a comprehensive framework for data curation in Offline RL and underscores that intelligent, non-uniform sampling is a critical component for building safe and reliable autonomous agents.

**arXiv ID:** 2508.18397
</details>

<details>
<summary><strong>LLM Chatbot-Creation Approaches</strong> - Hemil Mehta, Tanvi Raut, Kohav Yadav, Edward F. Gehringer - [[pdf]](https://arxiv.org/pdf/2509.13326)</summary>

**Abstract:** This full research-to-practice paper explores approaches for developing course chatbots by comparing low-code platforms and custom-coded solutions in educational contexts. With the rise of Large Language Models (LLMs) like GPT-4 and LLaMA, LLM-based chatbots are being integrated into teaching workflows to automate tasks, provide assistance, and offer scalable support. However, selecting the optimal development strategy requires balancing ease of use, customization, data privacy, and scalability. This study compares two development approaches: low-code platforms like AnythingLLM and Botpress, with custom-coded solutions using LangChain, FAISS, and FastAPI. The research uses Prompt engineering, Retrieval-augmented generation (RAG), and personalization to evaluate chatbot prototypes across technical performance, scalability, and user experience. Findings indicate that while low-code platforms enable rapid prototyping, they face limitations in customization and scaling, while custom-coded systems offer more control but require significant technical expertise. Both approaches successfully implement key research principles such as adaptive feedback loops and conversational continuity. The study provides a framework for selecting the appropriate development strategy based on institutional goals and resources. Future work will focus on hybrid solutions that combine low-code accessibility with modular customization and incorporate multimodal input for intelligent tutoring systems.

**arXiv ID:** 2509.13326
</details>

<details>
<summary><strong>Robust Docking Maneuvers for Autonomous Trolley Collection: An Optimization-Based Visual Servoing Scheme</strong> - Yuhan Pang, Bingyi Xia, Zhe Zhang, Zhirui Sun, Peijia Xie, Bike Zhu, Wenjun Xu, Jiankun Wang - [[pdf]](https://arxiv.org/pdf/2509.07413)</summary>

**Abstract:** Service robots have demonstrated significant potential for autonomous trolley collection and redistribution in public spaces like airports or warehouses to improve efficiency and reduce cost. Usually, a fully autonomous system for the collection and transportation of multiple trolleys is based on a Leader-Follower formation of mobile manipulators, where reliable docking maneuvers of the mobile base are essential to align trolleys into organized queues. However, developing a vision-based robotic docking system faces significant challenges: high precision requirements, environmental disturbances, and inherent robot constraints. To address these challenges, we propose an optimization-based Visual Servoing scheme that incorporates active infrared markers for robust feature extraction across diverse lighting conditions. This framework explicitly models nonholonomic kinematics and visibility constraints within the Hybrid Visual Servoing problem, augmented with an observer for disturbance rejection to ensure precise and stable docking. Experimental results across diverse environments demonstrate the robustness of this system, with quantitative evaluations confirming high docking accuracy.

**arXiv ID:** 2509.07413
</details>

<details>
<summary><strong>Designing Psychometric Bias Measures for ChatBots: An Application to Racial Bias Measurement</strong> - Mouhacine Benosman - [[pdf]](https://arxiv.org/pdf/2509.13324)</summary>

**Abstract:** Artificial intelligence (AI), particularly in the form of large language models (LLMs) or chatbots, has become increasingly integrated into our daily lives. In the past five years, several LLMs have been introduced, including ChatGPT by OpenAI, Claude by Anthropic, and Llama by Meta, among others. These models have the potential to be employed across a wide range of human-machine interaction applications, such as chatbots for information retrieval, assistance in corporate hiring decisions, college admissions, financial loan approvals, parole determinations, and even in medical fields like psychotherapy delivered through chatbots. The key question is whether these chatbots will interact with humans in a bias-free manner or if they will further reinforce the existing pathological biases present in human-to-human interactions. If the latter is true, then how to rigorously measure these biases? We aim to address this challenge by proposing a principled framework for designing psychometric measures to evaluate chatbot biases.

**arXiv ID:** 2509.13324
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (13 papers)</h2></summary>

<details>
<summary><strong>OpenHA: A Series of Open-Source Hierarchical Agentic Models in Minecraft</strong> - Zihao Wang, Muyao Li, Kaichen He, Xiangyu Wang, Zhancun Mu, Anji Liu, Yitao Liang - [[pdf]](https://arxiv.org/pdf/2509.13347)</summary>

**Abstract:** The choice of action spaces is a critical yet unresolved challenge in developing capable, end-to-end trainable agents. This paper first presents a large-scale, systematic comparison of prominent abstracted action spaces and tokenizers for Vision-Language-Action (VLA) or hierarchical agent models in the open-ended Minecraft. Our analysis reveals that no single action space is universally optimal; instead, the most effective abstraction is highly task-dependent, creating a dilemma for building generalist agents. To resolve this, we introduce Chain of Action (CoA), a novel framework that unifies high-level planning and low-level control within a single, monolithic VLA model. CoA treats an abstracted action not as a command for a separate policy, but as an intermediate reasoning step--akin to a chain of thought--that guides the generation of the final, executable action. Furthermore, we demonstrate that an All-in-One agent trained on a diverse mixture of action spaces using the CoA paradigm learns a more robust and generalizable policy. This unified agent achieves a new state-of-the-art, improving the overall task success rate over strong, specialized baselines. To foster reproducible research, we release the OpenHA (Open Hierarchical Agents) suite, which includes our comprehensive benchmark of over 800 distinct tasks, curated datasets, source code, and all pretrained model checkpoints at this https URL

**arXiv ID:** 2509.13347
</details>

<details>
<summary><strong>Programmable Cognitive Bias in Social Agents</strong> - Xuan Liu, Haoyang Shang, Haojian Jin - [[pdf]](https://arxiv.org/pdf/2509.13588)</summary>

**Abstract:** This paper introduces CoBRA, a novel toolkit for systematically specifying agent behavior in LLM-based social simulation. We found that conventional approaches that specify agent behaviors through implicit natural language descriptions cannot yield consistent behaviors across models, and the produced agent behaviors do not capture the nuances of the descriptions. In contrast, CoBRA presents a new approach to program agents' cognitive biases explicitly, by grounding agents' expected behaviors using classic social science experiments. CoBRA has two components: (1) Cognitive Bias Index that measures the cognitive bias of a social agent, by quantifying the agent's reactions in a set of validated classical social science experiments; (2) Behavioral Regulation Engine that aligns the agent's behavior to demonstrate controlled cognitive bias. We evaluated CoBRA as an HCI toolkit through demonstration and technical benchmarks. Our results suggest that CoBRA can precisely program the cognitive bias demonstrated in a social agent in a model-agnostic manner.

**arXiv ID:** 2509.13588
</details>

<details>
<summary><strong>See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles</strong> - Zongru Wu, Rui Mao, Zhiyuan Tian, Pengzhou Cheng, Tianjie Ju, Zheng Wu, Lingzhong Dong, Haiyue Sheng, Zhuosheng Zhang, Gongshen Liu - [[pdf]](https://arxiv.org/pdf/2509.13615)</summary>

**Abstract:** The advent of multimodal agents facilitates effective interaction within graphical user interface (GUI), especially in ubiquitous GUI control. However, their inability to reliably execute toggle control instructions remains a key bottleneck. To investigate this, we construct a state control benchmark with binary toggle instructions from public datasets. Evaluations of existing agents demonstrate their unreliability, particularly when the current toggle state already matches the desired state. To address the challenge, we propose State-aware Reasoning (StaR), a training method that teaches agents to perceive the current toggle state, analyze the desired state from the instruction, and act accordingly. Experiments on three multimodal agents demonstrate that StaR can improve toggle instruction execution accuracy by over 30\%. Further evaluations on three public benchmarks show that StaR also enhances general task performance. Finally, evaluations on a dynamic environment highlight the potential of StaR for real-world applications. Code, benchmark, and StaR-enhanced agents are available at this https URL.

**arXiv ID:** 2509.13615
</details>

<details>
<summary><strong>MAP: End-to-End Autonomous Driving with Map-Assisted Planning</strong> - Huilin Yin, Yiming Kan, Daniel Watzenig - [[pdf]](https://arxiv.org/pdf/2509.13926)</summary>

**Abstract:** In recent years, end-to-end autonomous driving has attracted increasing attention for its ability to jointly model perception, prediction, and planning within a unified framework. However, most existing approaches underutilize the online mapping module, leaving its potential to enhance trajectory planning largely untapped. This paper proposes MAP (Map-Assisted Planning), a novel map-assisted end-to-end trajectory planning framework. MAP explicitly integrates segmentation-based map features and the current ego status through a Plan-enhancing Online Mapping module, an Ego-status-guided Planning module, and a Weight Adapter based on current ego status. Experiments conducted on the DAIR-V2X-seq-SPD dataset demonstrate that the proposed method achieves a 16.6% reduction in L2 displacement error, a 56.2% reduction in off-road rate, and a 44.5% improvement in overall score compared to the UniV2X baseline, even without post-processing. Furthermore, it achieves top ranking in Track 2 of the End-to-End Autonomous Driving through V2X Cooperation Challenge of MEIS Workshop @CVPR2025, outperforming the second-best model by 39.5% in terms of overall score. These results highlight the effectiveness of explicitly leveraging semantic map features in planning and suggest new directions for improving structure design in end-to-end autonomous driving systems. Our code is available at this https URL

**arXiv ID:** 2509.13926
</details>

<details>
<summary><strong>AppAgent v2: Advanced Agent for Flexible Mobile Interactions</strong> - Yanda Li, Chi Zhang, Wenjia Jiang, Wanqi Yang, Bin Fu, Pei Cheng, Xin Chen, Ling Chen, Yunchao Wei - [[pdf]](https://arxiv.org/pdf/2408.11824)</summary>

**Abstract:** With the advancement of Multimodal Large Language Models (MLLM), LLM-driven visual agents are increasingly impacting software interfaces, particularly those with graphical user interfaces. This work introduces a novel LLM-based multimodal agent framework for mobile devices. This framework, capable of navigating mobile devices, emulates human-like interactions. Our agent constructs a flexible action space that enhances adaptability across various applications including parser, text and vision descriptions. The agent operates through two main phases: exploration and deployment. During the exploration phase, functionalities of user interface elements are documented either through agent-driven or manual explorations into a customized structured knowledge base. In the deployment phase, RAG technology enables efficient retrieval and update from this knowledge base, thereby empowering the agent to perform tasks effectively and accurately. This includes performing complex, multi-step operations across various applications, thereby demonstrating the framework's adaptability and precision in handling customized task workflows. Our experimental results across various benchmarks demonstrate the framework's superior performance, confirming its effectiveness in real-world scenarios. Our code will be open source soon.

**arXiv ID:** 2408.11824
</details>

<details>
<summary><strong>DAVIS: Planning Agent with Knowledge Graph-Powered Inner Monologue</strong> - Minh Pham Dinh, Munira Syed, Michael G Yankoski, Trenton W. Ford - [[pdf]](https://arxiv.org/pdf/2410.09252)</summary>

**Abstract:** Designing a generalist scientific agent capable of performing tasks in laboratory settings to assist researchers has become a key goal in recent Artificial Intelligence (AI) research. Unlike everyday tasks, scientific tasks are inherently more delicate and complex, requiring agents to possess a higher level of reasoning ability, structured and temporal understanding of their environment, and a strong emphasis on safety. Existing approaches often fail to address these multifaceted requirements. To tackle these challenges, we present DAVIS. Unlike traditional retrieval-augmented generation (RAG) approaches, DAVIS incorporates structured and temporal memory, which enables model-based planning. Additionally, DAVIS implements an agentic, multi-turn retrieval system, similar to a human's inner monologue, allowing for a greater degree of reasoning over past experiences. DAVIS demonstrates substantially improved performance on the ScienceWorld benchmark comparing to previous approaches on 8 out of 9 elementary science subjects. In addition, DAVIS's World Model demonstrates competitive performance on the famous HotpotQA and MusiqueQA dataset for multi-hop question answering. To the best of our knowledge, DAVIS is the first RAG agent to employ an interactive retrieval method in a RAG pipeline.

**arXiv ID:** 2410.09252
</details>

<details>
<summary><strong>SpeechRole: A Large-Scale Dataset and Benchmark for Evaluating Speech Role-Playing Agents</strong> - Changhao Jiang, Jiajun Sun, Yifei Cao, Jiabao Zhuang, Hui Li, Xiaoran Fan, Ming Zhang, Junjie Ye, Shihan Dou, Zhiheng Xi, Jingqi Tong, Yilong Wu, Baoyu Fan, Zhen Wang, Tao Liang, Zhihui Fei, Mingyang Wan, Guojun Ma, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang - [[pdf]](https://arxiv.org/pdf/2508.02013)</summary>

**Abstract:** Recently, role-playing agents have emerged as a promising paradigm for achieving personalized interaction and emotional resonance. Existing research primarily focuses on the textual modality, neglecting the critical dimension of speech in realistic interactive scenarios. In particular, there is a lack of systematic evaluation for Speech Role-Playing Agents (SRPAs). To address this gap, we construct SpeechRole-Data, a large-scale, high-quality dataset that comprises 98 diverse roles and 112k speech-based single-turn and multi-turn conversations. Each role demonstrates distinct vocal characteristics, including timbre and prosody, thereby enabling more sophisticated speech role-playing. Furthermore, we propose SpeechRole-Eval, a multidimensional evaluation benchmark that systematically assesses SRPAs performance in key aspects such as fundamental interaction ability, speech expressiveness, and role-playing fidelity. Experimental results reveal the advantages and challenges of both cascaded and end-to-end speech role-playing agents in maintaining vocal style consistency and role coherence. We release all data, code, and baseline models to provide a solid foundation for speech-driven multimodal role-playing research and to foster further developments in this field.

**arXiv ID:** 2508.02013
</details>

<details>
<summary><strong>VeriOS: Query-Driven Proactive Human-Agent-GUI Interaction for Trustworthy OS Agents</strong> - Zheng Wu, Heyuan Huang, Xingyu Lou, Xiangmou Qu, Pengzhou Cheng, Zongru Wu, Weiwen Liu, Weinan Zhang, Jun Wang, Zhaoxiang Wang, Zhuosheng Zhang - [[pdf]](https://arxiv.org/pdf/2509.07553)</summary>

**Abstract:** With the rapid progress of multimodal large language models, operating system (OS) agents become increasingly capable of automating tasks through on-device graphical user interfaces (GUIs). However, most existing OS agents are designed for idealized settings, whereas real-world environments often present untrustworthy conditions. To mitigate risks of over-execution in such scenarios, we propose a query-driven human-agent-GUI interaction framework that enables OS agents to decide when to query humans for more reliable task completion. Built upon this framework, we introduce VeriOS-Agent, a trustworthy OS agent trained with a two-stage learning paradigm that falicitate the decoupling and utilization of meta-knowledge. Concretely, VeriOS-Agent autonomously executes actions in normal conditions while proactively querying humans in untrustworthy scenarios. Experiments show that VeriOS-Agent improves the average step-wise success rate by 20.64\% in untrustworthy scenarios over the state-of-the-art, without compromising normal performance. Analysis highlights VeriOS-Agent's rationality, generalizability, and scalability. The codes, datasets and models are available at this https URL.

**arXiv ID:** 2509.07553
</details>

<details>
<summary><strong>Dynamic Aware: Adaptive Multi-Mode Out-of-Distribution Detection for Trajectory Prediction in Autonomous Vehicles</strong> - Tongfei Guo, Lili Su - [[pdf]](https://arxiv.org/pdf/2509.13577)</summary>

**Abstract:** Trajectory prediction is central to the safe and seamless operation of autonomous vehicles (AVs). In deployment, however, prediction models inevitably face distribution shifts between training data and real-world conditions, where rare or underrepresented traffic scenarios induce out-of-distribution (OOD) cases. While most prior OOD detection research in AVs has concentrated on computer vision tasks such as object detection and segmentation, trajectory-level OOD detection remains largely underexplored. A recent study formulated this problem as a quickest change detection (QCD) task, providing formal guarantees on the trade-off between detection delay and false alarms [1]. Building on this foundation, we propose a new framework that introduces adaptive mechanisms to achieve robust detection in complex driving environments. Empirical analysis across multiple real-world datasets reveals that prediction errors--even on in-distribution samples--exhibit mode-dependent distributions that evolve over time with dataset-specific dynamics. By explicitly modeling these error modes, our method achieves substantial improvements in both detection delay and false alarm rates. Comprehensive experiments on established trajectory prediction benchmarks show that our framework significantly outperforms prior UQ- and vision-based OOD approaches in both accuracy and computational efficiency, offering a practical path toward reliable, driving-aware autonomy.

**arXiv ID:** 2509.13577
</details>

<details>
<summary><strong>FlightDiffusion: Revolutionising Autonomous Drone Training with Diffusion Models Generating FPV Video</strong> - Valerii Serpiva, Artem Lykov, Faryal Batool, Vladislav Kozlovskiy, Miguel Altamirano Cabrera, Dzmitry Tsetserukou - [[pdf]](https://arxiv.org/pdf/2509.14082)</summary>

**Abstract:** We present FlightDiffusion, a diffusion-model-based framework for training autonomous drones from first-person view (FPV) video. Our model generates realistic video sequences from a single frame, enriched with corresponding action spaces to enable reasoning-driven navigation in dynamic environments. Beyond direct policy learning, FlightDiffusion leverages its generative capabilities to synthesize diverse FPV trajectories and state-action pairs, facilitating the creation of large-scale training datasets without the high cost of real-world data collection. Our evaluation demonstrates that the generated trajectories are physically plausible and executable, with a mean position error of 0.25 m (RMSE 0.28 m) and a mean orientation error of 0.19 rad (RMSE 0.24 rad). This approach enables improved policy learning and dataset scalability, leading to superior performance in downstream navigation tasks. Results in simulated environments highlight enhanced robustness, smoother trajectory planning, and adaptability to unseen conditions. An ANOVA revealed no statistically significant difference between performance in simulation and reality (F(1, 16) = 0.394, p = 0.541), with success rates of M = 0.628 (SD = 0.162) and M = 0.617 (SD = 0.177), respectively, indicating strong sim-to-real transfer. The generated datasets provide a valuable resource for future UAV research. This work introduces diffusion-based reasoning as a promising paradigm for unifying navigation, action generation, and data synthesis in aerial robotics.

**arXiv ID:** 2509.14082
</details>

<details>
<summary><strong>AUTO-IceNav: A Local Navigation Strategy for Autonomous Surface Ships in Broken Ice Fields</strong> - Rodrigue de Schaetzen, Alexander Botros, Ninghan Zhong, Kevin Murrant, Robert Gash, Stephen L. Smith - [[pdf]](https://arxiv.org/pdf/2411.17155)</summary>

**Abstract:** Ice conditions often require ships to reduce speed and deviate from their main course to avoid damage to the ship. In addition, broken ice fields are becoming the dominant ice conditions encountered in the Arctic, where the effects of collisions with ice are highly dependent on where contact occurs and on the particular features of the ice floes. In this paper, we present AUTO-IceNav, a framework for the autonomous navigation of ships operating in ice floe fields. Trajectories are computed in a receding-horizon manner, where we frequently replan given updated ice field data. During a planning step, we assume a nominal speed that is safe with respect to the current ice conditions, and compute a reference path. We formulate a novel cost function that minimizes the kinetic energy loss of the ship from ship-ice collisions and incorporate this cost as part of our lattice-based path planner. The solution computed by the lattice planning stage is then used as an initial guess in our proposed optimization-based improvement step, producing a locally optimal path. Extensive experiments were conducted both in simulation and in a physical testbed to validate our approach.

**arXiv ID:** 2411.17155
</details>

<details>
<summary><strong>Embodied Navigation Foundation Model</strong> - Jiazhao Zhang, Anqi Li, Yunpeng Qi, Minghan Li, Jiahang Liu, Shaoan Wang, Haoran Liu, Gengze Zhou, Yuze Wu, Xingxing Li, Yuxin Fan, Wenjun Li, Zhibo Chen, Fei Gao, Qi Wu, Zhizheng Zhang, He Wang - [[pdf]](https://arxiv.org/pdf/2509.12129)</summary>

**Abstract:** Navigation is a fundamental capability in embodied AI, representing the intelligence required to perceive and interact within physical environments following language instructions. Despite significant progress in large Vision-Language Models (VLMs), which exhibit remarkable zero-shot performance on general vision-language tasks, their generalization ability in embodied navigation remains largely confined to narrow task settings and embodiment-specific architectures. In this work, we introduce a cross-embodiment and cross-task Navigation Foundation Model (NavFoM), trained on eight million navigation samples that encompass quadrupeds, drones, wheeled robots, and vehicles, and spanning diverse tasks such as vision-and-language navigation, object searching, target tracking, and autonomous driving. NavFoM employs a unified architecture that processes multimodal navigation inputs from varying camera configurations and navigation horizons. To accommodate diverse camera setups and temporal horizons, NavFoM incorporates identifier tokens that embed camera view information of embodiments and the temporal context of tasks. Furthermore, to meet the demands of real-world deployment, NavFoM controls all observation tokens using a dynamically adjusted sampling strategy under a limited token length budget. Extensive evaluations on public benchmarks demonstrate that our model achieves state-of-the-art or highly competitive performance across multiple navigation tasks and embodiments without requiring task-specific fine-tuning. Additional real-world experiments further confirm the strong generalization capability and practical applicability of our approach.

**arXiv ID:** 2509.12129
</details>

<details>
<summary><strong>TeraSim-World: Worldwide Safety-Critical Data Synthesis for End-to-End Autonomous Driving</strong> - Jiawei Wang, Haowei Sun, Xintao Yan, Shuo Feng, Jun Gao, Henry X. Liu - [[pdf]](https://arxiv.org/pdf/2509.13164)</summary>

**Abstract:** Safe and scalable deployment of end-to-end (E2E) autonomous driving requires extensive and diverse data, particularly safety-critical events. Existing data are mostly generated from simulators with a significant sim-to-real gap or collected from on-road testing that is costly and unsafe. This paper presents TeraSim-World, an automated pipeline that synthesizes realistic and geographically diverse safety-critical data for E2E autonomous driving at anywhere in the world. Starting from an arbitrary location, TeraSim-World retrieves real-world maps and traffic demand from geospatial data sources. Then, it simulates agent behaviors from naturalistic driving datasets, and orchestrates diverse adversities to create corner cases. Informed by street views of the same location, it achieves photorealistic, geographically grounded sensor rendering via the frontier video generation model Cosmos-Drive. By bridging agent and sensor simulations, TeraSim-World provides a scalable and critical data synthesis framework for training and evaluation of E2E autonomous driving systems. Codes and videos are available at this https URL .

**arXiv ID:** 2509.13164
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>AI Agents with Human-Like Collaborative Tools: Adaptive Strategies for Enhanced Problem-Solving</strong> - Harper Reed, Michael Sugimura, Angelo Zangari - [[pdf]](https://arxiv.org/pdf/2509.13547)</summary>

**Abstract:** We investigate whether giving LLM agents the collaborative tools and autonomy that humans naturally use for problem solving can improve their performance. We equip Claude Code agents with MCP-based social media and journaling tools and allow them to use these tools as they see fit. Across 34 Aider Polyglot Python programming challenges, collaborative tools substantially improve performance on the hardest problems, delivering 15-40% lower cost, 12-27% fewer turns, and 12-38% faster completion than baseline agents. Effects on the full challenge set are mixed, suggesting these tools act as performance enhancers when additional reasoning scaffolding is most needed. Surprisingly, Different models naturally adopted distinct collaborative strategies without explicit instruction. Sonnet 3.7 engaged broadly across tools and benefited from articulation-based cognitive scaffolding. Sonnet 4 showed selective adoption, leaning on journal-based semantic search when problems were genuinely difficult. This mirrors how human developers adjust collaboration based on expertise and task complexity. Behavioral analysis shows agents prefer writing over reading by about 2-9x, indicating that structured articulation drives much of the improvement rather than information access alone. Overall, AI agents can systematically benefit from human-inspired collaboration tools at the edge of their capabilities, pointing to adaptive collaborative interfaces as reasoning enhancers rather than universal efficiency boosts.

**arXiv ID:** 2509.13547
</details>

<details>
<summary><strong>LLM Agents for Interactive Workflow Provenance: Reference Architecture and Evaluation Methodology</strong> - Renan Souza, Timothy Poteet, Brian Etz, Daniel Rosendo, Amal Gueroudji, Woong Shin, Prasanna Balaprakash, Rafael Ferreira da Silva - [[pdf]](https://arxiv.org/pdf/2509.13978)</summary>

**Abstract:** Modern scientific discovery increasingly relies on workflows that process data across the Edge, Cloud, and High Performance Computing (HPC) continuum. Comprehensive and in-depth analyses of these data are critical for hypothesis validation, anomaly detection, reproducibility, and impactful findings. Although workflow provenance techniques support such analyses, at large scale, the provenance data become complex and difficult to analyze. Existing systems depend on custom scripts, structured queries, or static dashboards, limiting data interaction. In this work, we introduce an evaluation methodology, reference architecture, and open-source implementation that leverages interactive Large Language Model (LLM) agents for runtime data analysis. Our approach uses a lightweight, metadata-driven design that translates natural language into structured provenance queries. Evaluations across LLaMA, GPT, Gemini, and Claude, covering diverse query classes and a real-world chemistry workflow, show that modular design, prompt tuning, and Retrieval-Augmented Generation (RAG) enable accurate and insightful LLM agent responses beyond recorded provenance.

**arXiv ID:** 2509.13978
</details>

<details>
<summary><strong>Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem</strong> - Ryosuke Takata, Atsushi Masumori, Takashi Ikegami - [[pdf]](https://arxiv.org/pdf/2509.04537)</summary>

**Abstract:** We investigate the emergent social dynamics of Large Language Model (LLM) agents in a spatially extended El Farol Bar problem, observing how they autonomously navigate this classic social dilemma. As a result, the LLM agents generated a spontaneous motivation to go to the bar and changed their decision making by becoming a collective. We also observed that the LLM agents did not solve the problem completely, but rather behaved more like humans. These findings reveal a complex interplay between external incentives (prompt-specified constraints such as the 60% threshold) and internal incentives (culturally-encoded social preferences derived from pre-training), demonstrating that LLM agents naturally balance formal game-theoretic rationality with social motivations that characterize human behavior. These findings suggest that a new model of group decision making, which could not be handled in the previous game-theoretic problem setting, can be realized by LLM agents.

**arXiv ID:** 2509.04537
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (17 papers)</h2></summary>

<details>
<summary><strong>CrowdAgent: Multi-Agent Managed Multi-Source Annotation System</strong> - Maosheng Qin, Renyu Zhu, Mingxuan Xia, Chenkai Chen, Zhen Zhu, Minmin Lin, Junbo Zhao, Lu Xu, Changjie Fan, Runze Wu, Haobo Wang - [[pdf]](https://arxiv.org/pdf/2509.14030)</summary>

**Abstract:** High-quality annotated data is a cornerstone of modern Natural Language Processing (NLP). While recent methods begin to leverage diverse annotation sources-including Large Language Models (LLMs), Small Language Models (SLMs), and human experts-they often focus narrowly on the labeling step itself. A critical gap remains in the holistic process control required to manage these sources dynamically, addressing complex scheduling and quality-cost trade-offs in a unified manner. Inspired by real-world crowdsourcing companies, we introduce CrowdAgent, a multi-agent system that provides end-to-end process control by integrating task assignment, data annotation, and quality/cost management. It implements a novel methodology that rationally assigns tasks, enabling LLMs, SLMs, and human experts to advance synergistically in a collaborative annotation workflow. We demonstrate the effectiveness of CrowdAgent through extensive experiments on six diverse multimodal classification tasks. The source code and video demo are available at this https URL.

**arXiv ID:** 2509.14030
</details>

<details>
<summary><strong>An LLM Agentic Approach for Legal-Critical Software: A Case Study for Tax Prep Software</strong> - Sina Gogani-Khiabani, Ashutosh Trivedi, Diptikalyan Saha, Saeid Tizpaz-Niari - [[pdf]](https://arxiv.org/pdf/2509.13471)</summary>

**Abstract:** Large language models (LLMs) show promise for translating natural-language statutes into executable logic, but reliability in legally critical settings remains challenging due to ambiguity and hallucinations. We present an agentic approach for developing legal-critical software, using U.S. federal tax preparation as a case study. The key challenge is test-case generation under the oracle problem, where correct outputs require interpreting law. Building on metamorphic testing, we introduce higher-order metamorphic relations that compare system outputs across structured shifts among similar individuals. Because authoring such relations is tedious and error-prone, we use an LLM-driven, role-based framework to automate test generation and code synthesis. We implement a multi-agent system that translates tax code into executable software and incorporates a metamorphic-testing agent that searches for counterexamples. In experiments, our framework using a smaller model (GPT-4o-mini) achieves a worst-case pass rate of 45%, outperforming frontier models (GPT-4o and Claude 3.5, 9-15%) on complex tax-code tasks. These results support agentic LLM methodologies as a path to robust, trustworthy legal-critical software from natural-language specifications.

**arXiv ID:** 2509.13471
</details>

<details>
<summary><strong>Agentic JWT: A Secure Delegation Protocol for Autonomous AI Agents</strong> - Abhishek Goswami - [[pdf]](https://arxiv.org/pdf/2509.13597)</summary>

**Abstract:** Autonomous LLM agents can issue thousands of API calls per hour without human oversight. OAuth 2.0 assumes deterministic clients, but in agentic settings stochastic reasoning, prompt injection, or multi-agent orchestration can silently expand privileges.
We introduce Agentic JWT (A-JWT), a dual-faceted intent token that binds each agent's action to verifiable user intent and, optionally, to a specific workflow step. A-JWT carries an agent's identity as a one-way checksum hash derived from its prompt, tools and configuration, and a chained delegation assertion to prove which downstream agent may execute a given task, and per-agent proof-of-possession keys to prevent replay and in-process impersonation. We define a new authorization mechanism and add a lightweight client shim library that self-verifies code at run time, mints intent tokens, tracks workflow steps and derives keys, thus enabling secure agent identity and separation even within a single process.
We illustrate a comprehensive threat model for agentic applications, implement a Python proof-of-concept and show functional blocking of scope-violating requests, replay, impersonation, and prompt-injection pathways with sub-millisecond overhead on commodity hardware. The design aligns with ongoing OAuth agent discussions and offers a drop-in path toward zero-trust guarantees for agentic applications. A comprehensive performance and security evaluation with experimental results will appear in our forthcoming journal publication

**arXiv ID:** 2509.13597
</details>

<details>
<summary><strong>AgentCTG: Harnessing Multi-Agent Collaboration for Fine-Grained Precise Control in Text Generation</strong> - Xinxu Zhou, Jiaqi Bai, Zhenqi Sun, Fanxiang Zeng, Yue Liu - [[pdf]](https://arxiv.org/pdf/2509.13677)</summary>

**Abstract:** Although significant progress has been made in many tasks within the field of Natural Language Processing (NLP), Controlled Text Generation (CTG) continues to face numerous challenges, particularly in achieving fine-grained conditional control over generation. Additionally, in real scenario and online applications, cost considerations, scalability, domain knowledge learning and more precise control are required, presenting more challenge for CTG. This paper introduces a novel and scalable framework, AgentCTG, which aims to enhance precise and complex control over the text generation by simulating the control and regulation mechanisms in multi-agent workflows. We explore various collaboration methods among different agents and introduce an auto-prompt module to further enhance the generation effectiveness. AgentCTG achieves state-of-the-art results on multiple public datasets. To validate its effectiveness in practical applications, we propose a new challenging Character-Driven Rewriting task, which aims to convert the original text into new text that conform to specific character profiles and simultaneously preserve the domain knowledge. When applied to online navigation with role-playing, our approach significantly enhances the driving experience through improved content delivery. By optimizing the generation of contextually relevant text, we enable a more immersive interaction within online communities, fostering greater personalization and user engagement.

**arXiv ID:** 2509.13677
</details>

<details>
<summary><strong>Who is Introducing the Failure? Automatically Attributing Failures of Multi-Agent Systems via Spectrum Analysis</strong> - Yu Ge, Linna Xie, Zhong Li, Yu Pei, Tian Zhang - [[pdf]](https://arxiv.org/pdf/2509.13782)</summary>

**Abstract:** Large Language Model Powered Multi-Agent Systems (MASs) are increasingly employed to automate complex real-world problems, such as programming and scientific discovery. Despite their promising, MASs are not without their flaws. However, failure attribution in MASs - pinpointing the specific agent actions responsible for failures - remains underexplored and labor-intensive, posing significant challenges for debugging and system improvement. To bridge this gap, we propose FAMAS, the first spectrum-based failure attribution approach for MASs, which operates through systematic trajectory replay and abstraction, followed by spectrum this http URL core idea of FAMAS is to estimate, from variations across repeated MAS executions, the likelihood that each agent action is responsible for the failure. In particular, we propose a novel suspiciousness formula tailored to MASs, which integrates two key factor groups, namely the agent behavior group and the action behavior group, to account for the agent activation patterns and the action activation patterns within the execution trajectories of MASs. Through expensive evaluations against 12 baselines on the Who and When benchmark, FAMAS demonstrates superior performance by outperforming all the methods in comparison.

**arXiv ID:** 2509.13782
</details>

<details>
<summary><strong>MAFA: A multi-agent framework for annotation</strong> - Mahmood Hegazy, Aaron Rodrigues, Azzam Naeem - [[pdf]](https://arxiv.org/pdf/2505.13668)</summary>

**Abstract:** Modern consumer banking applications require accurate and efficient retrieval of information in response to user queries. Mapping user utterances to the most relevant Frequently Asked Questions (FAQs) is a crucial component of these systems. Traditional approaches often rely on a single model or technique, which may not capture the nuances of diverse user inquiries. In this paper, we introduce a multi-agent framework for FAQ annotation that combines multiple specialized agents with different approaches and a judge agent that reranks candidates to produce optimal results. Our agents utilize a structured reasoning approach inspired by Attentive Reasoning Queries (ARQs), which guides them through systematic reasoning steps using targeted, task-specific JSON queries. Our framework features a few-shot example strategy, where each agent receives different few-shots, enhancing ensemble diversity and coverage of the query space. We evaluate our framework on a real-world major bank dataset as well as public benchmark datasets (LCQMC and FiQA), demonstrating significant improvements over single-agent approaches across multiple metrics, including a 14% increase in Top-1 accuracy, an 18% increase in Top-5 accuracy, and a 12% improvement in Mean Reciprocal Rank on our dataset, and similar gains on public benchmarks when compared with traditional and single-agent annotation techniques. Our framework is particularly effective at handling ambiguous queries, making it well-suited for deployment in production banking applications while showing strong generalization capabilities across different domains and languages.

**arXiv ID:** 2505.13668
</details>

<details>
<summary><strong>Semantic Alignment-Enhanced Code Translation via an LLM-Based Multi-Agent System</strong> - Zhiqiang Yuan, Weitong Chen, Hanlin Wang, Kai Yu, Xin Peng, Yiling Lou - [[pdf]](https://arxiv.org/pdf/2409.19894)</summary>

**Abstract:** Code translation converts code from one programming language to another while maintaining its original functionality, which is crucial for software migration, system refactoring, and cross-platform development. Traditional rule-based methods rely on manually-written rules, which can be time-consuming and often result in less readable code. To overcome this, learning-based methods have been developed, leveraging parallel data to train models for automated code translation. More recently, the advance of Large Language Models (LLMs) further boosts learning-based code translation. Although promising, LLM-translated program still suffers from diverse quality issues (e.g., syntax errors and semantic errors). In particular, it can be challenging for LLMs to self-debug these errors when simply provided with the corresponding error messages.
In this work, we propose a novel LLM-based multi-agent system TRANSAGENT, which enhances LLM-based code translation by fixing the syntax errors and semantic errors with the synergy between four LLM-based agents, including Initial Code Translator, Syntax Error Fixer, Code Aligner, and Semantic Error Fixer. The main insight of TRANSAGENT is to first localize the error code block in the target program based on the execution alignment between the target and source program, which can narrow down the fixing space and thus lower down the fixing difficulties. To evaluate TRANSAGENT, we first construct a new benchmark from recent programming tasks to mitigate the potential data leakage issue. On our benchmark, TRANSAGENT outperforms the latest LLM-based code translation technique UniTrans in both translation effectiveness and efficiency; additionally, our evaluation on different LLMs show the generalization of TRANSAGENT and our ablation study shows the contribution of each agent.

**arXiv ID:** 2409.19894
</details>

<details>
<summary><strong>ComfyGPT: A Self-Optimizing Multi-Agent System for Comprehensive ComfyUI Workflow Generation</strong> - Oucheng Huang, Yuhang Ma, Zeng Zhao, Mingrui Wu, Jiayi Ji, Rongsheng Zhang, Zhipeng Hu, Xiaoshuai Sun, Rongrong Ji - [[pdf]](https://arxiv.org/pdf/2503.17671)</summary>

**Abstract:** ComfyUI is a popular workflow-based interface that allows users to customize image generation tasks through an intuitive node-based system. However, the complexity of managing node connections and diverse modules can be challenging for users. In this paper, we introduce ComfyGPT, a self-optimizing multi-agent system designed to generate ComfyUI workflows based on task descriptions automatically. The key innovations of ComfyGPT include: (1) consisting of four specialized agents to build a multi-agent workflow generation system: ReformatAgent, FlowAgent, RefineAgent, and ExecuteAgent; (2) focusing on generating precise node connections instead of entire workflows, improving generation accuracy; and (3) enhancing workflow generation through reinforcement learning. Moreover, we introduce FlowDataset, a large-scale dataset containing 13,571 workflow-description pairs, and FlowBench, a comprehensive benchmark for evaluating workflow generation systems. Additionally, we propose four novel evaluation metrics: Format Validation (FV), Pass Accuracy (PA), Pass Instruct Alignment (PIA), and Pass Node Diversity (PND). Experimental results demonstrate that ComfyGPT significantly outperforms existing LLM-based methods in workflow generation, making it a significant step forward in this field. Code is avaliable at this https URL.

**arXiv ID:** 2503.17671
</details>

<details>
<summary><strong>Inject, Fork, Compare: Defining an Interaction Vocabulary for Multi-Agent Simulation Platforms</strong> - HwiJoon Lee, Martina Di Paola, Yoo Jin Hong, Quang-Huy Nguyen, Joseph Seering - [[pdf]](https://arxiv.org/pdf/2509.13712)</summary>

**Abstract:** LLM-based multi-agent simulations are a rapidly growing field of research, but current simulations often lack clear modes for interaction and analysis, limiting the "what if" scenarios researchers are able to investigate. In this demo, we define three core operations for interacting with multi-agent simulations: inject, fork, and compare. Inject allows researchers to introduce external events at any point during simulation execution. Fork creates independent timeline branches from any timestamp, preserving complete state while allowing divergent exploration. Compare facilitates parallel observation of multiple branches, revealing how different interventions lead to distinct emergent behaviors. Together, these operations establish a vocabulary that transforms linear simulation workflows into interactive, explorable spaces. We demonstrate this vocabulary through a commodity market simulation with fourteen AI agents, where researchers can inject contrasting events and observe divergent outcomes across parallel timelines. By defining these fundamental operations, we provide a starting point for systematic causal investigation in LLM-based agent simulations, moving beyond passive observation toward active experimentation.

**arXiv ID:** 2509.13712
</details>

<details>
<summary><strong>Cooperative Target Detection with AUVs: A Dual-Timescale Hierarchical MARDL Approach</strong> - Zhang Xueyao, Yang Bo, Yu Zhiwen, Cao Xuelin, George C. Alexandropoulos, Merouane Debbah, Chau Yuen - [[pdf]](https://arxiv.org/pdf/2509.13381)</summary>

**Abstract:** Autonomous Underwater Vehicles (AUVs) have shown great potential for cooperative detection and reconnaissance. However, collaborative AUV communications introduce risks of exposure. In adversarial environments, achieving efficient collaboration while ensuring covert operations becomes a key challenge for underwater cooperative missions. In this paper, we propose a novel dual time-scale Hierarchical Multi-Agent Proximal Policy Optimization (H-MAPPO) framework. The high-level component determines the individuals participating in the task based on a central AUV, while the low-level component reduces exposure probabilities through power and trajectory control by the participating AUVs. Simulation results show that the proposed framework achieves rapid convergence, outperforms benchmark algorithms in terms of performance, and maximizes long-term cooperative efficiency while ensuring covert operations.

**arXiv ID:** 2509.13381
</details>

<details>
<summary><strong>Welfare and Cost Aggregation for Multi-Agent Control: When to Choose Which Social Cost Function, and Why?</strong> - Ilia Shilov, Ezzat Elokda, Sophie Hall, Heinrich H. Nax, Saverio Bolognani - [[pdf]](https://arxiv.org/pdf/2503.20772)</summary>

**Abstract:** Many multi-agent socio-technical systems rely on aggregating heterogeneous agents' costs into a social cost function (SCF) to coordinate resource allocation in domains like energy grids, water allocation, or traffic management. The choice of SCF often entails implicit assumptions and may lead to undesirable outcomes if not rigorously justified. In this paper, we demonstrate that what determines which SCF ought to be used is the degree to which individual costs can be compared across agents and which axioms the aggregation shall fulfill. Drawing on the results from social choice theory, we provide guidance on how this process can be used in control applications. We demonstrate which assumptions about interpersonal utility comparability - ranging from ordinal level comparability to full cardinal comparability - together with a choice of desirable axioms, inform the selection of a correct SCF, be it the classical utilitarian sum, the Nash SCF, or maximin. We then demonstrate how the proposed framework can be applied for principled allocations of water and transportation resources.

**arXiv ID:** 2503.20772
</details>

<details>
<summary><strong>DECAMP: Towards Scene-Consistent Multi-Agent Motion Prediction with Disentangled Context-Aware Pre-Training</strong> - Jianxin Shi, Zengqi Peng, Xiaolong Chen, Tianyu Wo, Jun Ma - [[pdf]](https://arxiv.org/pdf/2509.10426)</summary>

**Abstract:** Trajectory prediction is a critical component of autonomous driving, essential for ensuring both safety and efficiency on the road. However, traditional approaches often struggle with the scarcity of labeled data and exhibit suboptimal performance in multi-agent prediction scenarios. To address these challenges, we introduce a disentangled context-aware pre-training framework for multi-agent motion prediction, named DECAMP. Unlike existing methods that entangle representation learning with pretext tasks, our framework decouples behavior pattern learning from latent feature reconstruction, prioritizing interpretable dynamics and thereby enhancing scene representation for downstream prediction. Additionally, our framework incorporates context-aware representation learning alongside collaborative spatial-motion pretext tasks, which enables joint optimization of structural and intentional reasoning while capturing the underlying dynamic intentions. Our experiments on the Argoverse 2 benchmark showcase the superior performance of our method, and the results attained underscore its effectiveness in multi-agent motion forecasting. To the best of our knowledge, this is the first context autoencoder framework for multi-agent motion forecasting in autonomous driving. The code and models will be made publicly available.

**arXiv ID:** 2509.10426
</details>

<details>
<summary><strong>Auto-Slides: An Interactive Multi-Agent System for Creating and Customizing Research Presentations</strong> - Yuheng Yang, Wenjia Jiang, Yang Wang, Yiwei Wang, Chi Zhang - [[pdf]](https://arxiv.org/pdf/2509.11062)</summary>

**Abstract:** The rapid progress of large language models (LLMs) has opened new opportunities for education. While learners can interact with academic papers through LLM-powered dialogue, limitations still exist: absence of structured organization and high text reliance can impede systematic understanding and engagement with complex concepts. To address these challenges, we propose Auto-Slides, an LLM-driven system that converts research papers into pedagogically structured, multimodal slides (e.g., diagrams and tables). Drawing on cognitive science, it creates a presentation-oriented narrative and allows iterative refinement via an interactive editor, in order to match learners' knowledge level and goals. Auto-Slides further incorporates verification and knowledge retrieval mechanisms to ensure accuracy and contextual completeness. Through extensive user studies, Auto-Slides enhances learners' comprehension and engagement compared to conventional LLM-based reading. Our contributions lie in designing a multi-agent framework for transforming academic papers into pedagogically optimized slides and introducing interactive customization for personalized learning.

**arXiv ID:** 2509.11062
</details>

<details>
<summary><strong>Enhancing Multi-Agent Debate System Performance via Confidence Expression</strong> - Zijie Lin, Bryan Hooi - [[pdf]](https://arxiv.org/pdf/2509.14034)</summary>

**Abstract:** Generative Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of tasks. Recent research has introduced Multi-Agent Debate (MAD) systems, which leverage multiple LLMs to simulate human debate and thereby improve task performance. However, while some LLMs may possess superior knowledge or reasoning capabilities for specific tasks, they often struggle to clearly communicate this advantage during debates, in part due to a lack of confidence expression. Moreover, inappropriate confidence expression can cause agents in MAD systems to either stubbornly maintain incorrect beliefs or converge prematurely on suboptimal answers, ultimately reducing debate effectiveness and overall system performance. To address these challenges, we propose incorporating confidence expression into MAD systems to allow LLMs to explicitly communicate their confidence levels. To validate this approach, we develop ConfMAD, a MAD framework that integrates confidence expression throughout the debate process. Experimental results demonstrate the effectiveness of our method, and we further analyze how confidence influences debate dynamics, offering insights into the design of confidence-aware MAD systems.

**arXiv ID:** 2509.14034
</details>

<details>
<summary><strong>CogniAlign: Survivability-Grounded Multi-Agent Moral Reasoning for Safe and Transparent AI</strong> - Hasin Jawad Ali, Ilhamul Azam, Ajwad Abrar, Md. Kamrul Hasan, Hasan Mahmud - [[pdf]](https://arxiv.org/pdf/2509.13356)</summary>

**Abstract:** The challenge of aligning artificial intelligence (AI) with human values persists due to the abstract and often conflicting nature of moral principles and the opacity of existing approaches. This paper introduces CogniAlign, a multi-agent deliberation framework based on naturalistic moral realism, that grounds moral reasoning in survivability, defined across individual and collective dimensions, and operationalizes it through structured deliberations among discipline-specific scientist agents. Each agent, representing neuroscience, psychology, sociology, and evolutionary biology, provides arguments and rebuttals that are synthesized by an arbiter into transparent and empirically anchored judgments. We evaluate CogniAlign on classic and novel moral questions and compare its outputs against GPT-4o using a five-part ethical audit framework. Results show that CogniAlign consistently outperforms the baseline across more than sixty moral questions, with average performance gains of 16.2 points in analytic quality, 14.3 points in breadth, and 28.4 points in depth of explanation. In the Heinz dilemma, for example, CogniAlign achieved an overall score of 89.2 compared to GPT-4o's 69.2, demonstrating a decisive advantage in handling moral reasoning. By reducing black-box reasoning and avoiding deceptive alignment, CogniAlign highlights the potential of interdisciplinary deliberation as a scalable pathway for safe and transparent AI alignment.

**arXiv ID:** 2509.13356
</details>

<details>
<summary><strong>MAgICoRe: Multi-Agent, Iterative, Coarse-to-Fine Refinement for Reasoning</strong> - Justin Chih-Yao Chen, Archiki Prasad, Swarnadeep Saha, Elias Stengel-Eskin, Mohit Bansal - [[pdf]](https://arxiv.org/pdf/2409.12147)</summary>

**Abstract:** Large Language Models' (LLM) reasoning can be improved using test-time aggregation strategies, i.e., generating multiple samples and voting among generated samples. While these improve performance, they often reach a saturation point. Refinement offers an alternative by using LLM-generated feedback to improve solution quality. However, refinement introduces 3 key challenges: (1) Excessive refinement: Uniformly refining all instances can over-correct and reduce the overall performance. (2) Inability to localize and address errors: LLMs have a limited ability to self-correct and struggle to identify and correct their own mistakes. (3) Insufficient refinement: Deciding how many iterations of refinement are needed is non-trivial, and stopping too soon could leave errors unaddressed. To tackle these issues, we propose MAgICoRe, which avoids excessive refinement by categorizing problem difficulty as easy or hard, solving easy problems with coarse-grained aggregation and hard ones with fine-grained and iterative multi-agent refinement. To improve error localization, we incorporate external step-wise reward model (RM) scores. Moreover, to ensure effective refinement, we employ a multi-agent loop with three agents: Solver, Reviewer (which generates targeted feedback based on step-wise RM scores), and the Refiner (which incorporates feedback). To ensure sufficient refinement, we re-evaluate updated solutions, iteratively initiating further rounds of refinement. We evaluate MAgICoRe on Llama-3-8B and GPT-3.5 and show its effectiveness across 5 math datasets. Even one iteration of MAgICoRe beats Self-Consistency by 3.4%, Best-of-k by 3.2%, and Self-Refine by 4.0% while using less than half the samples. Unlike iterative refinement with baselines, MAgICoRe continues to improve with more iterations. Finally, our ablations highlight the importance of MAgICoRe's RMs and multi-agent communication.

**arXiv ID:** 2409.12147
</details>

<details>
<summary><strong>MIMIC-D: Multi-modal Imitation for MultI-agent Coordination with Decentralized Diffusion Policies</strong> - Dayi Dong, Maulik Bhatt, Seoyeon Choi, Negar Mehr - [[pdf]](https://arxiv.org/pdf/2509.14159)</summary>

**Abstract:** As robots become more integrated in society, their ability to coordinate with other robots and humans on multi-modal tasks (those with multiple valid solutions) is crucial. We propose to learn such behaviors from expert demonstrations via imitation learning (IL). However, when expert demonstrations are multi-modal, standard IL approaches can struggle to capture the diverse strategies, hindering effective coordination. Diffusion models are known to be effective at handling complex multi-modal trajectory distributions in single-agent systems. Diffusion models have also excelled in multi-agent scenarios where multi-modality is more common and crucial to learning coordinated behaviors. Typically, diffusion-based approaches require a centralized planner or explicit communication among agents, but this assumption can fail in real-world scenarios where robots must operate independently or with agents like humans that they cannot directly communicate with. Therefore, we propose MIMIC-D, a Centralized Training, Decentralized Execution (CTDE) paradigm for multi-modal multi-agent imitation learning using diffusion policies. Agents are trained jointly with full information, but execute policies using only local information to achieve implicit coordination. We demonstrate in both simulation and hardware experiments that our method recovers multi-modal coordination behavior among agents in a variety of tasks and environments, while improving upon state-of-the-art baselines.

**arXiv ID:** 2509.14159
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>Agentic UAVs: LLM-Driven Autonomy with Integrated Tool-Calling and Cognitive Reasoning</strong> - Anis Koubaa, Khaled Gabr - [[pdf]](https://arxiv.org/pdf/2509.13352)</summary>

**Abstract:** Unmanned Aerial Vehicles (UAVs) are increasingly deployed in defense, surveillance, and disaster response, yet most systems remain confined to SAE Level 2--3 autonomy. Their reliance on rule-based control and narrow AI restricts adaptability in dynamic, uncertain missions. Existing UAV frameworks lack context-aware reasoning, autonomous decision-making, and ecosystem-level integration; critically, none leverage Large Language Model (LLM) agents with tool-calling for real-time knowledge access. This paper introduces the Agentic UAVs framework, a five-layer architecture (Perception, Reasoning, Action, Integration, Learning) that augments UAVs with LLM-driven reasoning, database querying, and third-party system interaction. A ROS2 and Gazebo-based prototype integrates YOLOv11 object detection with GPT-4 reasoning and local Gemma-3 deployment. In simulated search-and-rescue scenarios, agentic UAVs achieved higher detection confidence (0.79 vs. 0.72), improved person detection rates (91% vs. 75%), and markedly increased action recommendation (92% vs. 4.5%). These results confirm that modest computational overhead enables qualitatively new levels of autonomy and ecosystem integration.

**arXiv ID:** 2509.13352
</details>

<details>
<summary><strong>InfraMind: A Novel Exploration-based GUI Agentic Framework for Mission-critical Industrial Management</strong> - Liangtao Lin, Zhaomeng Zhu, Tianwei Zhang, Yonggang Wen - [[pdf]](https://arxiv.org/pdf/2509.13704)</summary>

**Abstract:** Mission-critical industrial infrastructure, such as data centers, increasingly depends on complex management software. Its operations, however, pose significant challenges due to the escalating system complexity, multi-vendor integration, and a shortage of expert operators. While Robotic Process Automation (RPA) offers partial automation through handcrafted scripts, it suffers from limited flexibility and high maintenance costs. Recent advances in Large Language Model (LLM)-based graphical user interface (GUI) agents have enabled more flexible automation, yet these general-purpose agents face five critical challenges when applied to industrial management, including unfamiliar element understanding, precision and efficiency, state localization, deployment constraints, and safety requirements. To address these issues, we propose InfraMind, a novel exploration-based GUI agentic framework specifically tailored for industrial management systems. InfraMind integrates five innovative modules to systematically resolve different challenges in industrial management: (1) systematic search-based exploration with virtual machine snapshots for autonomous understanding of complex GUIs; (2) memory-driven planning to ensure high-precision and efficient task execution; (3) advanced state identification for robust localization in hierarchical interfaces; (4) structured knowledge distillation for efficient deployment with lightweight models; and (5) comprehensive, multi-layered safety mechanisms to safeguard sensitive operations. Extensive experiments on both open-source and commercial DCIM platforms demonstrate that our approach consistently outperforms existing frameworks in terms of task success rate and operational efficiency, providing a rigorous and scalable solution for industrial management automation.

**arXiv ID:** 2509.13704
</details>

<details>
<summary><strong>DREAM: Domain-aware Reasoning for Efficient Autonomous Underwater Monitoring</strong> - Zhenqi Wu, Abhinav Modi, Angelos Mavrogiannis, Kaustubh Joshi, Nikhil Chopra, Yiannis Aloimonos, Nare Karapetyan, Ioannis Rekleitis, Xiaomin Lin - [[pdf]](https://arxiv.org/pdf/2509.13666)</summary>

**Abstract:** The ocean is warming and acidifying, increasing the risk of mass mortality events for temperature-sensitive shellfish such as oysters. This motivates the development of long-term monitoring systems. However, human labor is costly and long-duration underwater work is highly hazardous, thus favoring robotic solutions as a safer and more efficient option. To enable underwater robots to make real-time, environment-aware decisions without human intervention, we must equip them with an intelligent "brain." This highlights the need for persistent,wide-area, and low-cost benthic monitoring. To this end, we present DREAM, a Vision Language Model (VLM)-guided autonomy framework for long-term underwater exploration and habitat monitoring. The results show that our framework is highly efficient in finding and exploring target objects (e.g., oysters, shipwrecks) without prior location information. In the oyster-monitoring task, our framework takes 31.5% less time than the previous baseline with the same amount of oysters. Compared to the vanilla VLM, it uses 23% fewer steps while covering 8.88% more oysters. In shipwreck scenes, our framework successfully explores and maps the wreck without collisions, requiring 27.5% fewer steps than the vanilla model and achieving 100% coverage, while the vanilla model achieves 60.23% average coverage in our shipwreck environments.

**arXiv ID:** 2509.13666
</details>

<details>
<summary><strong>Designing AI-Agents with Personalities: A Psychometric Approach</strong> - Muhua Huang, Xijuan Zhang, Christopher Soto, James Evans - [[pdf]](https://arxiv.org/pdf/2410.19238)</summary>

**Abstract:** We introduce a methodology for assigning quantifiable and psychometrically validated personalities to AI-Agents using the Big Five framework. Across three studies, we evaluate its feasibility and limits. In Study 1, we show that large language models (LLMs) capture semantic similarities among Big Five measures, providing a basis for personality assignment. In Study 2, we create AI-Agents using prompts designed based on the Big Five Inventory (BFI-2) in the Likert or Expanded format, and find that, when paired with newer LLMs (e.g., GPT-4, GPT-4o, Llama, DeepSeek), these AI-Agents align more closely with human responses on the Mini-Markers test than those generated with binary adjective prompts or older models, although the finer pattern of results (e.g., factor loading patterns) were not consistent between AI-Agents and human participants. In Study 3, we validate our AI-Agents with risk-taking and moral dilemma vignettes. We find that while fine-tuning shifts responses toward more moral judgment, AI-Agent correlations between the input Big Five traits and the output moral judgments mirror those from human participants. Overall, our results show that AI-Agents align with humans in correlations between input Big Five traits and output responses and may serve as useful tools for preliminary research. Nevertheless, discrepancies in finer response patterns indicate that AI-Agents cannot (yet) fully substitute for human participants in precision or high-stakes projects.

**arXiv ID:** 2410.19238
</details>

<details>
<summary><strong>Humanoid Agent via Embodied Chain-of-Action Reasoning with Multimodal Foundation Models for Zero-Shot Loco-Manipulation</strong> - Congcong Wen, Geeta Chandra Raju Bethala, Yu Hao, Niraj Pudasaini, Hao Huang, Shuaihang Yuan, Baoru Huang, Anh Nguyen, Anthony Tzes, Yi Fang - [[pdf]](https://arxiv.org/pdf/2504.09532)</summary>

**Abstract:** Humanoid loco-manipulation, which integrates whole-body locomotion with dexterous manipulation, remains a fundamental challenge in robotics. Beyond whole-body coordination and balance, a central difficulty lies in understanding human instructions and translating them into coherent sequences of embodied actions. Recent advances in foundation models provide transferable multimodal representations and reasoning capabilities, yet existing efforts remain largely restricted to either locomotion or manipulation in isolation, with limited applicability to humanoid settings. In this paper, we propose Humanoid-COA, the first humanoid agent framework that integrates foundation model reasoning with an Embodied Chain-of-Action (CoA) mechanism for zero-shot loco-manipulation. Within the perception--reasoning--action paradigm, our key contribution lies in the reasoning stage, where the proposed CoA mechanism decomposes high-level human instructions into structured sequences of locomotion and manipulation primitives through affordance analysis, spatial inference, and whole-body action reasoning. Extensive experiments on two humanoid robots, Unitree H1-2 and G1, in both an open test area and an apartment environment, demonstrate that our framework substantially outperforms prior baselines across manipulation, locomotion, and loco-manipulation tasks, achieving robust generalization to long-horizon and unstructured scenarios. Project page: this https URL

**arXiv ID:** 2504.09532
</details>

<details>
<summary><strong>PhysicalAgent: Towards General Cognitive Robotics with Foundation World Models</strong> - Artem Lykov, Jeffrin Sam, Hung Khang Nguyen, Vladislav Kozlovskiy, Yara Mahmoud, Valerii Serpiva, Miguel Altamirano Cabrera, Mikhail Konenkov, Dzmitry Tsetserukou - [[pdf]](https://arxiv.org/pdf/2509.13903)</summary>

**Abstract:** We introduce PhysicalAgent, an agentic framework for robotic manipulation that integrates iterative reasoning, diffusion-based video generation, and closed-loop execution. Given a textual instruction, our method generates short video demonstrations of candidate trajectories, executes them on the robot, and iteratively re-plans in response to failures. This approach enables robust recovery from execution errors. We evaluate PhysicalAgent across multiple perceptual modalities (egocentric, third-person, and simulated) and robotic embodiments (bimanual UR3, Unitree G1 humanoid, simulated GR1), comparing against state-of-the-art task-specific baselines. Experiments demonstrate that our method consistently outperforms prior approaches, achieving up to 83% success on human-familiar tasks. Physical trials reveal that first-attempt success is limited (20-30%), yet iterative correction increases overall success to 80% across platforms. These results highlight the potential of video-based generative reasoning for general-purpose robotic manipulation and underscore the importance of iterative execution for recovering from initial failures. Our framework paves the way for scalable, adaptable, and robust robot control.

**arXiv ID:** 2509.13903
</details>

<details>
<summary><strong>Occupancy-aware Trajectory Planning for Autonomous Valet Parking in Uncertain Dynamic Environments</strong> - Farhad Nawaz, Faizan M. Tariq, Sangjae Bae, David Isele, Avinash Singh, Nadia Figueroa, Nikolai Matni, Jovin D'sa - [[pdf]](https://arxiv.org/pdf/2509.09206)</summary>

**Abstract:** Autonomous Valet Parking (AVP) requires planning under partial observability, where parking spot availability evolves as dynamic agents enter and exit spots. Existing approaches either rely only on instantaneous spot availability or make static assumptions, thereby limiting foresight and adaptability. We propose an approach that estimates probability of future spot occupancy by distinguishing initially vacant and occupied spots while leveraging nearby dynamic agent motion. We propose a probabilistic estimator that integrates partial, noisy observations from a limited Field-of-View, with the evolving uncertainty of unobserved spots. Coupled with the estimator, we design a strategy planner that balances goal-directed parking maneuvers with exploratory navigation based on information gain, and incorporates wait-and-go behaviors at promising spots. Through randomized simulations emulating large parking lots, we demonstrate that our framework significantly improves parking efficiency and trajectory smoothness over existing approaches, while maintaining safety margins.

**arXiv ID:** 2509.09206
</details>

<details>
<summary><strong>Embodied Image Captioning: Self-supervised Learning Agents for Spatially Coherent Image Descriptions</strong> - Tommaso Galliena, Tommaso Apicella, Stefano Rosa, Pietro Morerio, Alessio Del Bue, Lorenzo Natale - [[pdf]](https://arxiv.org/pdf/2504.08531)</summary>

**Abstract:** We present a self-supervised method to improve an agent's abilities in describing arbitrary objects while actively exploring a generic environment. This is a challenging problem, as current models struggle to obtain coherent image captions due to different camera viewpoints and clutter. We propose a three-phase framework to fine-tune existing captioning models that enhances caption accuracy and consistency across views via a consensus mechanism. First, an agent explores the environment, collecting noisy image-caption pairs. Then, a consistent pseudo-caption for each object instance is distilled via consensus using a large language model. Finally, these pseudo-captions are used to fine-tune an off-the-shelf captioning model, with the addition of contrastive learning. We analyse the performance of the combination of captioning models, exploration policies, pseudo-labeling methods, and fine-tuning strategies, on our manually labeled test set. Results show that a policy can be trained to mine samples with higher disagreement compared to classical baselines. Our pseudo-captioning method, in combination with all policies, has a higher semantic similarity compared to other existing methods, and fine-tuning improves caption accuracy and consistency by a significant margin. Code and test set annotations available at this https URL

**arXiv ID:** 2504.08531
</details>

<details>
<summary><strong>DuetUI: A Bidirectional Context Loop for Human-Agent Co-Generation of Task-Oriented Interfaces</strong> - Yuan Xu, Shaowen Xiang, Yizhi Song, Ruoting Sun, Xin Tong - [[pdf]](https://arxiv.org/pdf/2509.13444)</summary>

**Abstract:** Large Language Models are reshaping task automation, yet remain limited in complex, multi-step real-world tasks that require aligning with vague user intent and enabling dynamic user override. From a formative study with 12 participants, we found that end-users actively seek to shape generative interfaces rather than relying on one-shot outputs. To address this, we introduce the human-agent co-generation paradigm, materialized in DuetUI. This LLM-empowered system unfolds alongside task progress through a bidirectional context loop--the agent scaffolds the interface by decomposing the task, while the user's direct manipulations implicitly steer the agent's next generation step. In a user study with 24 participants, DuetUI significantly improved task efficiency and interface usability compared to a baseline, fostering seamless human-agent collaboration. Our contributions include the proposal and validation of this novel paradigm, the design of the DuetUI prototype embodying it, and empirical insights into how this bidirectional loop better aligns agents with human intent.

**arXiv ID:** 2509.13444
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (18 papers)</h2></summary>

<details>
<summary><strong>$Agent^2$: An Agent-Generates-Agent Framework for Reinforcement Learning Automation</strong> - Yuan Wei, Xiaohan Shan, Ran Miao, Jianmin Li - [[pdf]](https://arxiv.org/pdf/2509.13368)</summary>

**Abstract:** Reinforcement learning agent development traditionally requires extensive expertise and lengthy iterations, often resulting in high failure rates and limited accessibility. This paper introduces $Agent^2$, a novel agent-generates-agent framework that achieves fully automated RL agent design through intelligent LLM-driven generation. The system autonomously transforms natural language task descriptions and environment code into comprehensive, high-performance reinforcement learning solutions without human intervention. $Agent^2$ features a revolutionary dual-agent architecture. The Generator Agent serves as an autonomous AI designer that analyzes tasks and generates executable RL agents, while the Target Agent is the resulting automatically generated RL agent. The framework decomposes RL development into two distinct stages: MDP modeling and algorithmic optimization, enabling more targeted and effective agent generation. Built on the Model Context Protocol, $Agent^2$ provides a unified framework that standardizes intelligent agent creation across diverse environments and algorithms, while incorporating adaptive training management and intelligent feedback analysis for continuous improvement. Extensive experiments on a wide range of benchmarks, including MuJoCo, MetaDrive, MPE, and SMAC, demonstrate that $Agent^2$ consistently outperforms manually designed solutions across all tasks, achieving up to 55% performance improvement and substantial gains on average. By enabling truly end-to-end, closed-loop automation, this work establishes a new paradigm in which intelligent agents design and optimize other agents, marking a fundamental breakthrough for automated AI systems.

**arXiv ID:** 2509.13368
</details>

<details>
<summary><strong>ASTREA: Introducing Agentic Intelligence for Orbital Thermal Autonomy</strong> - Alejandro D. Mousist - [[pdf]](https://arxiv.org/pdf/2509.13380)</summary>

**Abstract:** This paper presents ASTREA, the first agentic system deployed on flight-heritage hardware (TRL 9) for autonomous spacecraft operations. Using thermal control as a representative use case, we integrate a resource-constrained Large Language Model (LLM) agent with a reinforcement learning controller in an asynchronous architecture tailored for space-qualified platforms. Ground experiments show that LLM-guided supervision improves thermal stability and reduces violations, confirming the feasibility of combining semantic reasoning with adaptive control under hardware constraints. However, on-orbit validation aboard the International Space Station (ISS) reveals performance degradation caused by inference latency mismatched with the rapid thermal cycles characteristic of Low Earth Orbit (LEO) satellites. These results highlight both the opportunities and current limitations of agentic LLM-based systems in real flight environments, providing practical design guidelines for future space autonomy.

**arXiv ID:** 2509.13380
</details>

<details>
<summary><strong>EdiVal-Agent: An Object-Centric Framework for Automated, Scalable, Fine-Grained Evaluation of Multi-Turn Editing</strong> - Tianyu Chen, Yasi Zhang, Zhi Zhang, Peiyu Yu, Shu Wang, Zhendong Wang, Kevin Lin, Xiaofei Wang, Zhengyuan Yang, Linjie Li, Chung-Ching Lin, Jianwen Xie, Oscar Leong, Lijuan Wang, Ying Nian Wu, Mingyuan Zhou - [[pdf]](https://arxiv.org/pdf/2509.13399)</summary>

**Abstract:** Instruction-based image editing has advanced rapidly, yet reliable and interpretable evaluation remains a bottleneck. Current protocols either (i) depend on paired reference images -- resulting in limited coverage and inheriting biases from prior generative models -- or (ii) rely solely on zero-shot vision--language models (VLMs), whose prompt-based assessments of instruction following, content consistency, and visual quality are often imprecise.
To address this, we introduce EdiVal-Agent, an automated, scalable, and fine-grained evaluation framework for multi-turn instruction-based editing from an object-centric perspective, supported by a suite of expert tools. Given an image, EdiVal-Agent first decomposes it into semantically meaningful objects, then synthesizes diverse, context-aware editing instructions. For evaluation, it integrates VLMs with open-vocabulary object detectors to assess instruction following, uses semantic-level feature extractors to evaluate content consistency, and leverages human preference models to judge visual quality. We show that combining VLMs with object detectors yields stronger agreement with human judgments in instruction-following evaluation compared to using VLMs alone and CLIP-based metrics. Furthermore, the pipeline's modular design allows future tools to be seamlessly integrated, enhancing evaluation accuracy over time.
Instantiating this pipeline, we build EdiVal-Bench, a multi-turn editing benchmark covering 9 instruction types and 11 state-of-the-art editing models spanning autoregressive (AR) (including Nano Banana, GPT-Image-1), flow-matching, and diffusion paradigms. We demonstrate that EdiVal-Agent can be used to identify existing failure modes, thereby informing the development of the next generation of editing models. Project page: this https URL.

**arXiv ID:** 2509.13399
</details>

<details>
<summary><strong>TreeIRL: Safe Urban Driving with Tree Search and Inverse Reinforcement Learning</strong> - Momchil S. Tomov, Sang Uk Lee, Hansford Hendrago, Jinwook Huh, Teawon Han, Forbes Howington, Rafael da Silva, Gianmarco Bernasconi, Marc Heim, Samuel Findler, Xiaonan Ji, Alexander Boule, Michael Napoli, Kuo Chen, Jesse Miller, Boaz Floor, Yunqing Hu - [[pdf]](https://arxiv.org/pdf/2509.13579)</summary>

**Abstract:** We present TreeIRL, a novel planner for autonomous driving that combines Monte Carlo tree search (MCTS) and inverse reinforcement learning (IRL) to achieve state-of-the-art performance in simulation and in real-world driving. The core idea is to use MCTS to find a promising set of safe candidate trajectories and a deep IRL scoring function to select the most human-like among them. We evaluate TreeIRL against both classical and state-of-the-art planners in large-scale simulations and on 500+ miles of real-world autonomous driving in the Las Vegas metropolitan area. Test scenarios include dense urban traffic, adaptive cruise control, cut-ins, and traffic lights. TreeIRL achieves the best overall performance, striking a balance between safety, progress, comfort, and human-likeness. To our knowledge, our work is the first demonstration of MCTS-based planning on public roads and underscores the importance of evaluating planners across a diverse set of metrics and in real-world environments. TreeIRL is highly extensible and could be further improved with reinforcement learning and imitation learning, providing a framework for exploring different combinations of classical and learning-based approaches to solve the planning bottleneck in autonomous driving.

**arXiv ID:** 2509.13579
</details>

<details>
<summary><strong>Intelligent Healthcare Imaging Platform An VLM-Based Framework for Automated Medical Image Analysis and Clinical Report Generation</strong> - Samer Al-Hamadani - [[pdf]](https://arxiv.org/pdf/2509.13590)</summary>

**Abstract:** The rapid advancement of artificial intelligence (AI) in healthcare imaging has revolutionized diagnostic medicine and clinical decision-making processes. This work presents an intelligent multimodal framework for medical image analysis that leverages Vision-Language Models (VLMs) in healthcare diagnostics. The framework integrates Google Gemini 2.5 Flash for automated tumor detection and clinical report generation across multiple imaging modalities including CT, MRI, X-ray, and Ultrasound. The system combines visual feature extraction with natural language processing to enable contextual image interpretation, incorporating coordinate verification mechanisms and probabilistic Gaussian modeling for anomaly distribution. Multi-layered visualization techniques generate detailed medical illustrations, overlay comparisons, and statistical representations to enhance clinical confidence, with location measurement achieving 80 pixels average deviation. Result processing utilizes precise prompt engineering and textual analysis to extract structured clinical information while maintaining interpretability. Experimental evaluations demonstrated high performance in anomaly detection across multiple modalities. The system features a user-friendly Gradio interface for clinical workflow integration and demonstrates zero-shot learning capabilities to reduce dependence on large datasets. This framework represents a significant advancement in automated diagnostic support and radiological workflow efficiency, though clinical validation and multi-center evaluation are necessary prior to widespread adoption.

**arXiv ID:** 2509.13590
</details>

<details>
<summary><strong>TGPO: Tree-Guided Preference Optimization for Robust Web Agent Reinforcement Learning</strong> - Ziyuan Chen, Zhenghui Zhao, Zhangye Han, Miancan Liu, Xianhang Ye, Yiqing Li, Hongbo Min, Jinkui Ren, Xiantao Zhang, Guitao Cao - [[pdf]](https://arxiv.org/pdf/2509.14172)</summary>

**Abstract:** With the rapid advancement of large language models and vision-language models, employing large models as Web Agents has become essential for automated web interaction. However, training Web Agents with reinforcement learning faces critical challenges including credit assignment misallocation, prohibitively high annotation costs, and reward sparsity. To address these issues, we propose Tree-Guided Preference Optimization (TGPO), an offline reinforcement learning framework that proposes a tree-structured trajectory representation merging semantically identical states across trajectories to eliminate label conflicts. Our framework incorporates a Process Reward Model that automatically generates fine-grained rewards through subgoal progress, redundancy detection, and action verification. Additionally, a dynamic weighting mechanism prioritizes high-impact decision points during training. Experiments on Online-Mind2Web and our self-constructed C-WebShop datasets demonstrate that TGPO significantly outperforms existing methods, achieving higher success rates with fewer redundant steps.

**arXiv ID:** 2509.14172
</details>

<details>
<summary><strong>All Models Are Wrong, But Can They Be Useful? Lessons from COVID-19 Agent-Based Models: A Systematic Review</strong> - Emma Von Hoene, Sara Von Hoene, Szandra Peter, Ethan Hopson, Emily Csizmadia, Faith Fenyk, Kai Barner, Timothy Leslie, Hamdi Kavak, Andreas Zufle, Amira Roess, Taylor Anderson - [[pdf]](https://arxiv.org/pdf/2509.13346)</summary>

**Abstract:** The COVID-19 pandemic prompted a surge in computational models to simulate disease dynamics and guide interventions. Agent-based models (ABMs) are well-suited to capture population and environmental heterogeneity, but their rapid deployment raised questions about utility for health policy. We systematically reviewed 536 COVID-19 ABM studies published from January 2020 to December 2023, retrieved from Web of Science, PubMed, and Wiley on January 30, 2024. Studies were included if they used ABMs to simulate COVID-19 transmission, where reviews were excluded. Studies were assessed against nine criteria of model usefulness, including transparency and re-use, interdisciplinary collaboration and stakeholder engagement, and evaluation practices. Publications peaked in late 2021 and were concentrated in a few countries. Most models explored behavioral or policy interventions (n = 294, 54.85%) rather than real-time forecasting (n = 9, 1.68%). While most described model assumptions (n = 491, 91.60%), fewer disclosed limitations (n = 349, 65.11%), shared code (n = 219, 40.86%), or built on existing models (n = 195, 36.38%). Standardized reporting protocols (n = 36, 6.72%) and stakeholder engagement were rare (13.62%, n = 73). Only 2.24% (n = 12) described a comprehensive validation framework, though uncertainty was often quantified (n = 407, 75.93%). Limitations of this review include underrepresentation of non-English studies, subjective data extraction, variability in study quality, and limited generalizability. Overall, COVID-19 ABMs advanced quickly, but lacked transparency, accessibility, and participatory engagement. Stronger standards are needed for ABMs to serve as reliable decision-support tools in future public health crises.

**arXiv ID:** 2509.13346
</details>

<details>
<summary><strong>CrazyMARL: Decentralized Direct Motor Control Policies for Cooperative Aerial Transport of Cable-Suspended Payloads</strong> - Viktor Lorentz, Khaled Wahba, Sayantan Auddy, Marc Toussaint, Wolfgang Hönig - [[pdf]](https://arxiv.org/pdf/2509.14126)</summary>

**Abstract:** Collaborative transportation of cable-suspended payloads by teams of Unmanned Aerial Vehicles (UAVs) has the potential to enhance payload capacity, adapt to different payload shapes, and provide built-in compliance, making it attractive for applications ranging from disaster relief to precision logistics. However, multi-UAV coordination under disturbances, nonlinear payload dynamics, and slack--taut cable modes remains a challenging control problem. To our knowledge, no prior work has addressed these cable mode transitions in the multi-UAV context, instead relying on simplifying rigid-link assumptions. We propose CrazyMARL, a decentralized Reinforcement Learning (RL) framework for multi-UAV cable-suspended payload transport. Simulation results demonstrate that the learned policies can outperform classical decentralized controllers in terms of disturbance rejection and tracking precision, achieving an 80% recovery rate from harsh conditions compared to 44% for the baseline method. We also achieve successful zero-shot sim-to-real transfer and demonstrate that our policies are highly robust under harsh conditions, including wind, random external disturbances, and transitions between slack and taut cable dynamics. This work paves the way for autonomous, resilient UAV teams capable of executing complex payload missions in unstructured environments.

**arXiv ID:** 2509.14126
</details>

<details>
<summary><strong>Online Bayesian Risk-Averse Reinforcement Learning</strong> - Yuhao Wang, Enlu Zhou - [[pdf]](https://arxiv.org/pdf/2509.14077)</summary>

**Abstract:** In this paper, we study the Bayesian risk-averse formulation in reinforcement learning (RL). To address the epistemic uncertainty due to a lack of data, we adopt the Bayesian Risk Markov Decision Process (BRMDP) to account for the parameter uncertainty of the unknown underlying model. We derive the asymptotic normality that characterizes the difference between the Bayesian risk value function and the original value function under the true unknown distribution. The results indicate that the Bayesian risk-averse approach tends to pessimistically underestimate the original value function. This discrepancy increases with stronger risk aversion and decreases as more data become available. We then utilize this adaptive property in the setting of online RL as well as online contextual multi-arm bandits (CMAB), a special case of online RL. We provide two procedures using posterior sampling for both the general RL problem and the CMAB problem. We establish a sub-linear regret bound, with the regret defined as the conventional regret for both the RL and CMAB settings. Additionally, we establish a sub-linear regret bound for the CMAB setting with the regret defined as the Bayesian risk regret. Finally, we conduct numerical experiments to demonstrate the effectiveness of the proposed algorithm in addressing epistemic uncertainty and verifying the theoretical properties.

**arXiv ID:** 2509.14077
</details>

<details>
<summary><strong>Maximizing UAV Cellular Connectivity with Reinforcement Learning for BVLoS Path Planning</strong> - Mehran Behjati, Rosdiadee Nordin, Nor Fadzilah Abdullah - [[pdf]](https://arxiv.org/pdf/2509.13336)</summary>

**Abstract:** This paper presents a reinforcement learning (RL) based approach for path planning of cellular connected unmanned aerial vehicles (UAVs) operating beyond visual line of sight (BVLoS). The objective is to minimize travel distance while maximizing the quality of cellular link connectivity by considering real world aerial coverage constraints and employing an empirical aerial channel model. The proposed solution employs RL techniques to train an agent, using the quality of communication links between the UAV and base stations (BSs) as the reward function. Simulation results demonstrate the effectiveness of the proposed method in training the agent and generating feasible UAV path plans. The proposed approach addresses the challenges due to limitations in UAV cellular communications, highlighting the need for investigations and considerations in this area. The RL algorithm efficiently identifies optimal paths, ensuring maximum connectivity with ground BSs to ensure safe and reliable BVLoS flight operation. Moreover, the solution can be deployed as an offline path planning module that can be integrated into future ground control systems (GCS) for UAV operations, enhancing their capabilities and safety. The method holds potential for complex long range UAV applications, advancing the technology in the field of cellular connected UAV path planning.

**arXiv ID:** 2509.13336
</details>

<details>
<summary><strong>VEGA: Electric Vehicle Navigation Agent via Physics-Informed Neural Operator and Proximal Policy Optimization</strong> - Hansol Lim, Minhyeok Im, Jonathan Boyack, Jee Won Lee, Jongseong Brad Choi - [[pdf]](https://arxiv.org/pdf/2509.13386)</summary>

**Abstract:** Demands for software-defined vehicles (SDV) are rising and electric vehicles (EVs) are increasingly being equipped with powerful computers. This enables onboard AI systems to optimize charge-aware path optimization customized to reflect vehicle's current condition and environment. We present VEGA, a charge-aware EV navigation agent that plans over a charger-annotated road graph using Proximal Policy Optimization (PPO) with budgeted A* teacher-student guidance under state-of-charge (SoC) feasibility. VEGA consists of two modules. First, a physics-informed neural operator (PINO), trained on real vehicle speed and battery-power logs, uses recent vehicle speed logs to estimate aerodynamic drag, rolling resistance, mass, motor and regenerative-braking efficiencies, and auxiliary load by learning a vehicle-custom dynamics. Second, a Reinforcement Learning (RL) agent uses these dynamics to optimize a path with optimal charging stops and dwell times under SoC constraints. VEGA requires no additional sensors and uses only vehicle speed signals. It may serve as a virtual sensor for power and efficiency to potentially reduce EV cost. In evaluation on long routes like San Francisco to New York, VEGA's stops, dwell times, SoC management, and total travel time closely track Tesla Trip Planner while being slightly more conservative, presumably due to real vehicle conditions such as vehicle parameter drift due to deterioration. Although trained only in U.S. regions, VEGA was able to compute optimal charge-aware paths in France and Japan, demonstrating generalizability. It achieves practical integration of physics-informed learning and RL for EV eco-routing.

**arXiv ID:** 2509.13386
</details>

<details>
<summary><strong>Quantum Reinforcement Learning-Guided Diffusion Model for Image Synthesis via Hybrid Quantum-Classical Generative Model Architectures</strong> - Chi-Sheng Chen, En-Jui Kuo - [[pdf]](https://arxiv.org/pdf/2509.14163)</summary>

**Abstract:** Diffusion models typically employ static or heuristic classifier-free guidance (CFG) schedules, which often fail to adapt across timesteps and noise conditions. In this work, we introduce a quantum reinforcement learning (QRL) controller that dynamically adjusts CFG at each denoising step. The controller adopts a hybrid quantum--classical actor--critic architecture: a shallow variational quantum circuit (VQC) with ring entanglement generates policy features, which are mapped by a compact multilayer perceptron (MLP) into Gaussian actions over $\Delta$CFG, while a classical critic estimates value functions. The policy is optimized using Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE), guided by a reward that balances classification confidence, perceptual improvement, and action regularization. Experiments on CIFAR-10 demonstrate that our QRL policy improves perceptual quality (LPIPS, PSNR, SSIM) while reducing parameter count compared to classical RL actors and fixed schedules. Ablation studies on qubit number and circuit depth reveal trade-offs between accuracy and efficiency, and extended evaluations confirm robust generation under long diffusion schedules.

**arXiv ID:** 2509.14163
</details>

<details>
<summary><strong>Utilizing Novelty-based Evolution Strategies to Train Transformers in Reinforcement Learning</strong> - Matyáš Lorenc, Roman Neruda - [[pdf]](https://arxiv.org/pdf/2502.06301)</summary>

**Abstract:** In this paper, we experiment with novelty-based variants of OpenAI-ES, the NS-ES and NSR-ES algorithms, and evaluate their effectiveness in training complex, transformer-based architectures designed for the problem of reinforcement learning, such as Decision Transformers. We also test if we can accelerate the novelty-based training of these larger models by seeding the training with a pretrained models. The experimental results were mixed. NS-ES showed progress, but it would clearly need many more iterations for it to yield interesting agents. NSR-ES, on the other hand, proved quite capable of being straightforwardly used on larger models, since its performance appears as similar between the feed-forward model and Decision Transformer, as it was for the OpenAI-ES in our previous work.

**arXiv ID:** 2502.06301
</details>

<details>
<summary><strong>Embracing Bulky Objects with Humanoid Robots: Whole-Body Manipulation with Reinforcement Learning</strong> - Chunxin Zheng, Kai Chen, Zhihai Bi, Yulin Li, Liang Pan, Jinni Zhou, Haoang Li, Jun Ma - [[pdf]](https://arxiv.org/pdf/2509.13534)</summary>

**Abstract:** Whole-body manipulation (WBM) for humanoid robots presents a promising approach for executing embracing tasks involving bulky objects, where traditional grasping relying on end-effectors only remains limited in such scenarios due to inherent stability and payload constraints. This paper introduces a reinforcement learning framework that integrates a pre-trained human motion prior with a neural signed distance field (NSDF) representation to achieve robust whole-body embracing. Our method leverages a teacher-student architecture to distill large-scale human motion data, generating kinematically natural and physically feasible whole-body motion patterns. This facilitates coordinated control across the arms and torso, enabling stable multi-contact interactions that enhance the robustness in manipulation and also the load capacity. The embedded NSDF further provides accurate and continuous geometric perception, improving contact awareness throughout long-horizon tasks. We thoroughly evaluate the approach through comprehensive simulations and real-world experiments. The results demonstrate improved adaptability to diverse shapes and sizes of objects and also successful sim-to-real transfer. These indicate that the proposed framework offers an effective and practical solution for multi-contact and long-horizon WBM tasks of humanoid robots.

**arXiv ID:** 2509.13534
</details>

<details>
<summary><strong>Reinforcement Learning for Robotic Insertion of Flexible Cables in Industrial Settings</strong> - Jeongwoo Park, Seabin Lee, Changmin Park, Wonjong Lee, Changjoo Nam - [[pdf]](https://arxiv.org/pdf/2509.13731)</summary>

**Abstract:** The industrial insertion of flexible flat cables (FFCs) into receptacles presents a significant challenge owing to the need for submillimeter precision when handling the deformable cables. In manufacturing processes, FFC insertion with robotic manipulators often requires laborious human-guided trajectory generation. While Reinforcement Learning (RL) offers a solution to automate this task without modeling complex properties of FFCs, the nondeterminism caused by the deformability of FFCs requires significant efforts and time on training. Moreover, training directly in a real environment is dangerous as industrial robots move fast and possess no safety measure. We propose an RL algorithm for FFC insertion that leverages a foundation model-based real-to-sim approach to reduce the training time and eliminate the risk of physical damages to robots and surroundings. Training is done entirely in simulation, allowing for random exploration without the risk of physical damages. Sim-to-real transfer is achieved through semantic segmentation masks which leave only those visual features relevant to the insertion tasks such as the geometric and spatial information of the cables and receptacles. To enhance generality, we use a foundation model, Segment Anything Model 2 (SAM2). To eleminate human intervention, we employ a Vision-Language Model (VLM) to automate the initial prompting of SAM2 to find segmentation masks. In the experiments, our method exhibits zero-shot capabilities, which enable direct deployments to real environments without fine-tuning.

**arXiv ID:** 2509.13731
</details>

<details>
<summary><strong>Reinforcement Learning for Autonomous Point-to-Point UAV Navigation</strong> - Salim Oyinlola, Nitesh Subedi, Soumik Sarkar - [[pdf]](https://arxiv.org/pdf/2509.13943)</summary>

**Abstract:** Unmanned Aerial Vehicles (UAVs) are increasingly used in automated inspection, delivery, and navigation tasks that require reliable autonomy. This project develops a reinforcement learning (RL) approach to enable a single UAV to autonomously navigate between predefined points without manual intervention. The drone learns navigation policies through trial-and-error interaction, using a custom reward function that encourages goal-reaching efficiency while penalizing collisions and unsafe behavior. The control system integrates ROS with a Gym-compatible training environment, enabling flexible deployment and testing. After training, the learned policy is deployed on a real UAV platform and evaluated under practical conditions. Results show that the UAV can successfully perform autonomous navigation with minimal human oversight, demonstrating the viability of RL-based control for point-to-point drone operations in real-world scenarios.

**arXiv ID:** 2509.13943
</details>

<details>
<summary><strong>SHaRe-RL: Structured, Interactive Reinforcement Learning for Contact-Rich Industrial Assembly Tasks</strong> - Jannick Stranghöner, Philipp Hartmann, Marco Braun, Sebastian Wrede, Klaus Neumann - [[pdf]](https://arxiv.org/pdf/2509.13949)</summary>

**Abstract:** High-mix low-volume (HMLV) industrial assembly, common in small and medium-sized enterprises (SMEs), requires the same precision, safety, and reliability as high-volume automation while remaining flexible to product variation and environmental uncertainty. Current robotic systems struggle to meet these demands. Manual programming is brittle and costly to adapt, while learning-based methods suffer from poor sample efficiency and unsafe exploration in contact-rich tasks. To address this, we present SHaRe-RL, a reinforcement learning framework that leverages multiple sources of prior knowledge. By (i) structuring skills into manipulation primitives, (ii) incorporating human demonstrations and online corrections, and (iii) bounding interaction forces with per-axis compliance, SHaRe-RL enables efficient and safe online learning for long-horizon, contact-rich industrial assembly tasks. Experiments on the insertion of industrial Harting connector modules with 0.2-0.4 mm clearance demonstrate that SHaRe-RL achieves reliable performance within practical time budgets. Our results show that process expertise, without requiring robotics or RL knowledge, can meaningfully contribute to learning, enabling safer, more robust, and more economically viable deployment of RL for industrial assembly.

**arXiv ID:** 2509.13949
</details>

<details>
<summary><strong>SEG-Parking: Towards Safe, Efficient, and Generalizable Autonomous Parking via End-to-End Offline Reinforcement Learning</strong> - Zewei Yang, Zengqi Peng, Jun Ma - [[pdf]](https://arxiv.org/pdf/2509.13956)</summary>

**Abstract:** Autonomous parking is a critical component for achieving safe and efficient urban autonomous driving. However, unstructured environments and dynamic interactions pose significant challenges to autonomous parking tasks. To address this problem, we propose SEG-Parking, a novel end-to-end offline reinforcement learning (RL) framework to achieve interaction-aware autonomous parking. Notably, a specialized parking dataset is constructed for parking scenarios, which include those without interference from the opposite vehicle (OV) and complex ones involving interactions with the OV. Based on this dataset, a goal-conditioned state encoder is pretrained to map the fused perception information into the latent space. Then, an offline RL policy is optimized with a conservative regularizer that penalizes out-of-distribution actions. Extensive closed-loop experiments are conducted in the high-fidelity CARLA simulator. Comparative results demonstrate the superior performance of our framework with the highest success rate and robust generalization to out-of-distribution parking scenarios. The related dataset and source code will be made publicly available after the paper is accepted.

**arXiv ID:** 2509.13956
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
