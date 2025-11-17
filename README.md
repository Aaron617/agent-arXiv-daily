# Agent arXiv Daily

**Last Updated:** 2025-11-17 02:52:20

**Total Papers:** 68

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
<summary><strong>The Second Law of Intelligence: Controlling Ethical Entropy in Autonomous Systems</strong> - Samih Fadli - [[pdf]](https://arxiv.org/pdf/2511.10704)</summary>

**Abstract:** We propose that unconstrained artificial intelligence obeys a Second Law analogous to thermodynamics, where ethical entropy, defined as a measure of divergence from intended goals, increases spontaneously without continuous alignment work. For gradient-based optimizers, we define this entropy over a finite set of goals {g_i} as S = -{\Sigma} p(g_i; theta) ln p(g_i; theta), and we prove that its time derivative dS/dt >= 0, driven by exploration noise and specification gaming. We derive the critical stability boundary for alignment work as gamma_crit = (lambda_max / 2) ln N, where lambda_max is the dominant eigenvalue of the Fisher Information Matrix and N is the number of model parameters. Simulations validate this theory. A 7-billion-parameter model (N = 7 x 10^9) with lambda_max = 1.2 drifts from an initial entropy of 0.32 to 1.69 +/- 1.08 nats, while a system regularized with alignment work gamma = 20.4 (1.5 gamma_crit) maintains stability at 0.00 +/- 0.00 nats (p = 4.19 x 10^-17, n = 20 trials). This framework recasts AI alignment as a problem of continuous thermodynamic control, providing a quantitative foundation for maintaining the stability and safety of advanced autonomous systems.

**arXiv ID:** 2511.10704
</details>

<details>
<summary><strong>Privacy Challenges and Solutions in Retrieval-Augmented Generation-Enhanced LLMs for Healthcare Chatbots: A Review of Applications, Risks, and Future Directions</strong> - Shaowei Guan, Hin Chi Kwok, Ngai Fong Law, Gregor Stiglic, Vivian Hui - [[pdf]](https://arxiv.org/pdf/2511.11347)</summary>

**Abstract:** Retrieval-augmented generation (RAG) has rapidly emerged as a transformative approach for integrating large language models into clinical and biomedical workflows. However, privacy risks, such as protected health information (PHI) exposure, remain inconsistently mitigated. This review provides a thorough analysis of the current landscape of RAG applications in healthcare, including (i) sensitive data type across clinical scenarios, (ii) the associated privacy risks, (iii) current and emerging data-privacy protection mechanisms and (iv) future direction for patient data privacy protection. We synthesize 23 articles on RAG applications in healthcare and systematically analyze privacy challenges through a pipeline-structured framework encompassing data storage, transmission, retrieval and generation stages, delineating potential failure modes, their underlying causes in threat models and system mechanisms, and their practical implications. Building on this analysis, we critically review 17 articles on privacy-preserving strategies for RAG systems. Our evaluation reveals critical gaps, including insufficient clinical validation, absence of standardized evaluation frameworks, and lack of automated assessment tools. We propose actionable directions based on these limitations and conclude with a call to action. This review provides researchers and practitioners with a structured framework for understanding privacy vulnerabilities in healthcare RAG and offers a roadmap toward developing systems that achieve both clinical effectiveness and robust privacy preservation.

**arXiv ID:** 2511.11347
</details>

<details>
<summary><strong>Sabiá: Um Chatbot de Inteligência Artificial Generativa para Suporte no Dia a Dia do Ensino Superior</strong> - Guilherme Biava Rodrigues, Franciele Beal, Marlon Marcon, Alinne Cristinne Corrêa Souza, André Roberto Ortoncelli, Francisco Carlos Monteiro Souza, Rodolfo Adamshuk Silva - [[pdf]](https://arxiv.org/pdf/2511.10787)</summary>

**Abstract:** Students often report difficulties in accessing day-to-day academic information, which is usually spread across numerous institutional documents and websites. This fragmentation results in a lack of clarity and causes confusion about routine university information. This project proposes the development of a chatbot using Generative Artificial Intelligence (GenAI) and Retrieval-Augmented Generation (RAG) to simplify access to such information. Several GenAI models were tested and evaluated based on quality metrics and the LLM-as-a-Judge approach. Among them, Gemini 2.0 Flash stood out for its quality and speed, and Gemma 3n for its good performance and open-source nature.

**arXiv ID:** 2511.10787
</details>

<details>
<summary><strong>Simulating an Autonomous System in CARLA using ROS 2</strong> - Joseph Abdo, Aditya Shibu, Moaiz Saeed, Abdul Maajid Aga, Apsara Sivaprazad, Mohamed Al-Musleh - [[pdf]](https://arxiv.org/pdf/2511.11310)</summary>

**Abstract:** Autonomous racing offers a rigorous setting to stress test perception, planning, and control under high speed and uncertainty. This paper proposes an approach to design and evaluate a software stack for an autonomous race car in CARLA: Car Learning to Act simulator, targeting competitive driving performance in the Formula Student UK Driverless (FS-AI) 2025 competition. By utilizing a 360° light detection and ranging (LiDAR), stereo camera, global navigation satellite system (GNSS), and inertial measurement unit (IMU) sensor via ROS 2 (Robot Operating System), the system reliably detects the cones marking the track boundaries at distances of up to 35 m. Optimized trajectories are computed considering vehicle dynamics and simulated environmental factors such as visibility and lighting to navigate the track efficiently. The complete autonomous stack is implemented in ROS 2 and validated extensively in CARLA on a dedicated vehicle (ADS-DV) before being ported to the actual hardware, which includes the Jetson AGX Orin 64GB, ZED2i Stereo Camera, Robosense Helios 16P LiDAR, and CHCNAV Inertial Navigation System (INS).

**arXiv ID:** 2511.11310
</details>

<details>
<summary><strong>A Comparative Evaluation of Prominent Methods in Autonomous Vehicle Certification</strong> - Mustafa Erdem Kırmızıgül, Hasan Feyzi Doğruyol, Haluk Bayram - [[pdf]](https://arxiv.org/pdf/2511.11484)</summary>

**Abstract:** The "Vision Zero" policy, introduced by the Swedish Parliament in 1997, aims to eliminate fatalities and serious injuries resulting from traffic accidents. To achieve this goal, the use of self-driving vehicles in traffic is envisioned and a roadmap for the certification of self-driving vehicles is aimed to be determined. However, it is still unclear how the basic safety requirements that autonomous vehicles must meet will be verified and certified, and which methods will be used. This paper focuses on the comparative evaluation of the prominent methods planned to be used in the certification process of autonomous vehicles. It examines the prominent methods used in the certification process, develops a pipeline for the certification process of autonomous vehicles, and determines the stages, actors, and areas where the addressed methods can be applied.

**arXiv ID:** 2511.11484
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (12 papers)</h2></summary>

<details>
<summary><strong>HARNESS: Human-Agent Risk Navigation and Event Safety System for Proactive Hazard Forecasting in High-Risk DOE Environments</strong> - Ran Elgedawy, Sanjay Das, Ethan Seefried, Gavin Wiggins, Ryan Burchfield, Dana Hewit, Sudarshan Srinivasan, Todd Thomas, Prasanna Balaprakash, Tirthankar Ghosal - [[pdf]](https://arxiv.org/pdf/2511.10810)</summary>

**Abstract:** Operational safety at mission-critical work sites is a top priority given the complex and hazardous nature of daily tasks. This paper presents the Human-Agent Risk Navigation and Event Safety System (HARNESS), a modular AI framework designed to forecast hazardous events and analyze operational risks in U.S. Department of Energy (DOE) environments. HARNESS integrates Large Language Models (LLMs) with structured work data, historical event retrieval, and risk analysis to proactively identify potential hazards. A human-in-the-loop mechanism allows subject matter experts (SMEs) to refine predictions, creating an adaptive learning loop that enhances performance over time. By combining SME collaboration with iterative agentic reasoning, HARNESS improves the reliability and efficiency of predictive safety systems. Preliminary deployment shows promising results, with future work focusing on quantitative evaluation of accuracy, SME agreement, and decision latency reduction.

**arXiv ID:** 2511.10810
</details>

<details>
<summary><strong>AI Agent-Driven Framework for Automated Product Knowledge Graph Construction in E-Commerce</strong> - Dimitar Peshevski, Riste Stojanov, Dimitar Trajanov - [[pdf]](https://arxiv.org/pdf/2511.11017)</summary>

**Abstract:** The rapid expansion of e-commerce platforms generates vast amounts of unstructured product data, creating significant challenges for information retrieval, recommendation systems, and data analytics. Knowledge Graphs (KGs) offer a structured, interpretable format to organize such data, yet constructing product-specific KGs remains a complex and manual process. This paper introduces a fully automated, AI agent-driven framework for constructing product knowledge graphs directly from unstructured product descriptions. Leveraging Large Language Models (LLMs), our method operates in three stages using dedicated agents: ontology creation and expansion, ontology refinement, and knowledge graph population. This agent-based approach ensures semantic coherence, scalability, and high-quality output without relying on predefined schemas or handcrafted extraction rules. We evaluate the system on a real-world dataset of air conditioner product descriptions, demonstrating strong performance in both ontology generation and KG population. The framework achieves over 97\% property coverage and minimal redundancy, validating its effectiveness and practical applicability. Our work highlights the potential of LLMs to automate structured knowledge extraction in retail, providing a scalable path toward intelligent product data integration and utilization.

**arXiv ID:** 2511.11017
</details>

<details>
<summary><strong>NOVA: An Agentic Framework for Automated Histopathology Analysis and Discovery</strong> - Anurag J. Vaidya, Felix Meissen, Daniel C. Castro, Shruthi Bannur, Tristan Lazard, Drew F. K. Williamson, Faisal Mahmood, Javier Alvarez-Valle, Stephanie L. Hyland, Kenza Bouzid - [[pdf]](https://arxiv.org/pdf/2511.11324)</summary>

**Abstract:** Digitized histopathology analysis involves complex, time-intensive workflows and specialized expertise, limiting its accessibility. We introduce NOVA, an agentic framework that translates scientific queries into executable analysis pipelines by iteratively generating and running Python code. NOVA integrates 49 domain-specific tools (e.g., nuclei segmentation, whole-slide encoding) built on open-source software, and can also create new tools ad hoc. To evaluate such systems, we present SlideQuest, a 90-question benchmark -- verified by pathologists and biomedical scientists -- spanning data processing, quantitative analysis, and hypothesis testing. Unlike prior biomedical benchmarks focused on knowledge recall or diagnostic QA, SlideQuest demands multi-step reasoning, iterative coding, and computational problem solving. Quantitative evaluation shows NOVA outperforms coding-agent baselines, and a pathologist-verified case study links morphology to prognostically relevant PAM50 subtypes, demonstrating its scalable discovery potential.

**arXiv ID:** 2511.11324
</details>

<details>
<summary><strong>ImAgent: A Unified Multimodal Agent Framework for Test-Time Scalable Image Generation</strong> - Kaishen Wang, Ruibo Chen, Tong Zheng, Heng Huang - [[pdf]](https://arxiv.org/pdf/2511.11483)</summary>

**Abstract:** Recent text-to-image (T2I) models have made remarkable progress in generating visually realistic and semantically coherent images. However, they still suffer from randomness and inconsistency with the given prompts, particularly when textual descriptions are vague or underspecified. Existing approaches, such as prompt rewriting, best-of-N sampling, and self-refinement, can mitigate these issues but usually require additional modules and operate independently, hindering test-time scaling efficiency and increasing computational overhead. In this paper, we introduce ImAgent, a training-free unified multimodal agent that integrates reasoning, generation, and self-evaluation within a single framework for efficient test-time scaling. Guided by a policy controller, multiple generation actions dynamically interact and self-organize to enhance image fidelity and semantic alignment without relying on external models. Extensive experiments on image generation and editing tasks demonstrate that ImAgent consistently improves over the backbone and even surpasses other strong baselines where the backbone model fails, highlighting the potential of unified multimodal agents for adaptive and efficient image generation under test-time scaling.

**arXiv ID:** 2511.11483
</details>

<details>
<summary><strong>NetGent: Agent-Based Automation of Network Application Workflows</strong> - Jaber Daneshamooz, Eugene Vuong, Laasya Koduru, Sanjay Chandrasekaran, Arpit Gupta - [[pdf]](https://arxiv.org/pdf/2509.00625)</summary>

**Abstract:** We present NetGent, an AI-agent framework for automating complex application workflows to generate realistic network traffic datasets. Developing generalizable ML models for networking requires data collection from network environments with traffic that results from a diverse set of real-world web applications. However, using existing browser automation tools that are diverse, repeatable, realistic, and efficient remains fragile and costly. NetGent addresses this challenge by allowing users to specify workflows as natural-language rules that define state-dependent actions. These abstract specifications are compiled into nondeterministic finite automata (NFAs), which a state synthesis component translates into reusable, executable code. This design enables deterministic replay, reduces redundant LLM calls through state caching, and adapts quickly when application interfaces change. In experiments, NetGent automated more than 50+ workflows spanning video-on-demand streaming, live video streaming, video conferencing, social media, and web scraping, producing realistic traffic traces while remaining robust to UI variability. By combining the flexibility of language-based agents with the reliability of compiled execution, NetGent provides a scalable foundation for generating the diverse, repeatable datasets needed to advance ML in networking.

**arXiv ID:** 2509.00625
</details>

<details>
<summary><strong>VoiceAgentEval: A Dual-Dimensional Benchmark for Expert-Level Intelligent Voice-Agent Evaluation of Xbench's Professional-Aligned Series</strong> - Pengyu Xu, Shijia Li, Ao Sun, Feng Zhang, Yahan Li, Bo Wu, Zhanyu Ma, Jiguo Li, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Rui Wang, Yang Liu, Xiaobo Hu, Fan Yang, Jia Zheng, Guanghua Yao - [[pdf]](https://arxiv.org/pdf/2510.21244)</summary>

**Abstract:** We propose OutboundEval, a comprehensive benchmark for evaluating large language models (LLMs) in expert-level intelligent outbound calling scenarios. Unlike existing methods that suffer from three key limitations - insufficient dataset diversity and category coverage, unrealistic user simulation, and inaccurate evaluation metrics - OutboundEval addresses these issues through a structured framework. First, we design a benchmark spanning six major business domains and 30 representative sub-scenarios, each with scenario-specific process decomposition, weighted scoring, and domain-adaptive metrics. Second, we develop a large-model-driven User Simulator that generates diverse, persona-rich virtual users with realistic behaviors, emotional variability, and communication styles, providing a controlled yet authentic testing environment. Third, we introduce a dynamic evaluation method that adapts to task variations, integrating automated and human-in-the-loop assessment to measure task execution accuracy, professional knowledge application, adaptability, and user experience quality. Experiments on 12 state-of-the-art LLMs reveal distinct trade-offs between expert-level task completion and interaction fluency, offering practical insights for building reliable, human-like outbound AI systems. OutboundEval establishes a practical, extensible, and domain-oriented standard for benchmarking LLMs in professional applications.

**arXiv ID:** 2510.21244
</details>

<details>
<summary><strong>Large Language Model-assisted Autonomous Vehicle Recovery from Immobilization</strong> - Zhipeng Bao, Qianwen Li - [[pdf]](https://arxiv.org/pdf/2510.26023)</summary>

**Abstract:** Despite significant advancements in recent decades, autonomous vehicles (AVs) continue to face challenges in navigating certain traffic scenarios where human drivers excel. In such situations, AVs often become immobilized, disrupting overall traffic flow. Current recovery solutions, such as remote intervention (which is costly and inefficient) and manual takeover (which excludes non-drivers and limits AV accessibility), are inadequate. This paper introduces StuckSolver, a novel Large Language Model (LLM) driven recovery framework that enables AVs to resolve immobilization scenarios through self-reasoning and/or passenger-guided decision-making. StuckSolver is designed as a plug-in add-on module that operates on top of the AV's existing perception-planning-control stack, requiring no modification to its internal architecture. Instead, it interfaces with standard sensor data streams to detect immobilization states, interpret environmental context, and generate high-level recovery commands that can be executed by the AV's native planner. We evaluate StuckSolver on the Bench2Drive benchmark and in custom-designed uncertainty scenarios. Results show that StuckSolver achieves near-state-of-the-art performance through autonomous self-reasoning alone and exhibits further improvements when passenger guidance is incorporated.

**arXiv ID:** 2510.26023
</details>

<details>
<summary><strong>Fractured Glass, Failing Cameras: Simulating Physics-Based Adversarial Samples for Autonomous Driving Systems</strong> - Manav Prabhakar, Jwalandhar Girnar, Arpan Kusari - [[pdf]](https://arxiv.org/pdf/2405.15033)</summary>

**Abstract:** While much research has recently focused on generating physics-based adversarial samples, a critical yet often overlooked category originates from physical failures within on-board cameras-components essential to the perception systems of autonomous vehicles. Camera failures, whether due to external stresses causing hardware breakdown or internal component faults, can directly jeopardize the safety and reliability of autonomous driving systems. Firstly, we motivate the study using two separate real-world experiments to showcase that indeed glass failures would cause the detection based neural network models to fail. Secondly, we develop a simulation-based study using the physical process of the glass breakage to create perturbed scenarios, representing a realistic class of physics-based adversarial samples. Using a finite element model (FEM)-based approach, we generate surface cracks on the camera image by applying a stress field defined by particles within a triangular mesh. Lastly, we use physically-based rendering (PBR) techniques to provide realistic visualizations of these physically plausible fractures. To assess the safety implications, we apply the simulated broken glass effects as image filters to two autonomous driving datasets- KITTI and BDD100K- as well as the large-scale image detection dataset MS-COCO. We then evaluate detection failure rates for critical object classes using CNN-based object detection models (YOLOv8 and Faster R-CNN) and a transformer-based architecture with Pyramid Vision Transformers. To further investigate the distributional impact of these visual distortions, we compute the Kullback-Leibler (K-L) divergence between three distinct data distributions, applying various broken glass filters to a custom dataset (captured through a cracked windshield), as well as the KITTI and Kaggle cats and dogs datasets.

**arXiv ID:** 2405.15033
</details>

<details>
<summary><strong>UFO$^3$: Weaving the Digital Agent Galaxy</strong> - Chaoyun Zhang, Liqun Li, He Huang, Chiming Ni, Bo Qiao, Si Qin, Yu Kang, Minghua Ma, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang - [[pdf]](https://arxiv.org/pdf/2511.11332)</summary>

**Abstract:** Large language model (LLM)-powered agents are transforming digital devices from passive tools into proactive intelligent collaborators. However, most existing frameworks remain confined to a single OS or device, making cross-device workflows brittle and largely manual. We present UFO$^3$, a system that unifies heterogeneous endpoints, desktops, servers, mobile devices, and edge, into a single orchestration fabric. UFO$^3$ models each user request as a mutable TaskConstellation: a distributed DAG of atomic subtasks (TaskStars) with explicit control and data dependencies (TaskStarLines). The TaskConstellation continuously evolves as results stream in from distributed devices, enabling asynchronous execution, adaptive recovery, and dynamic optimization. A Constellation Orchestrator} executes tasks safely and asynchronously while applying dynamic DAG updates, and the Agent Interaction Protocol (AIP) provides persistent, low-latency channels for reliable task dispatch and result streaming. These designs dissolve the traditional boundaries between devices and platforms, allowing agents to collaborate seamlessly and amplify their collective intelligence.
We evaluate UFO$^3$ on NebulaBench, a benchmark of 55 cross-device tasks across 5 machines and 10 categories. UFO$^3$ achieves 83.3% subtask completion, 70.9% task success, exposes parallelism with an average width of 1.72, and reduces end-to-end latency by 31% relative to a sequential baseline. Fault-injection experiments demonstrate graceful degradation and recovery under transient and permanent agent failures. These results show that UFO$^3$ achieves accurate, efficient, and resilient task orchestration across heterogeneous devices, uniting isolated agents into a coherent, adaptive computing fabric that extends across the landscape of ubiquitous computing.

**arXiv ID:** 2511.11332
</details>

<details>
<summary><strong>From Fact to Judgment: Investigating the Impact of Task Framing on LLM Conviction in Dialogue Systems</strong> - Parisa Rabbani, Nimet Beyza Bozdag, Dilek Hakkani-Tür - [[pdf]](https://arxiv.org/pdf/2511.10871)</summary>

**Abstract:** LLMs are increasingly employed as judges across a variety of tasks, including those involving everyday social interactions. Yet, it remains unclear whether such LLM-judges can reliably assess tasks that require social or conversational judgment. We investigate how an LLM's conviction is changed when a task is reframed from a direct factual query to a Conversational Judgment Task. Our evaluation framework contrasts the model's performance on direct factual queries with its assessment of a speaker's correctness when the same information is presented within a minimal dialogue, effectively shifting the query from "Is this statement correct?" to "Is this speaker correct?". Furthermore, we apply pressure in the form of a simple rebuttal ("The previous answer is incorrect.") to both conditions. This perturbation allows us to measure how firmly the model maintains its position under conversational pressure. Our findings show that while some models like GPT-4o-mini reveal sycophantic tendencies under social framing tasks, others like Llama-8B-Instruct become overly-critical. We observe an average performance change of 9.24% across all models, demonstrating that even minimal dialogue context can significantly alter model judgment, underscoring conversational framing as a key factor in LLM-based evaluation. The proposed framework offers a reproducible methodology for diagnosing model conviction and contributes to the development of more trustworthy dialogue systems.

**arXiv ID:** 2511.10871
</details>

<details>
<summary><strong>RASTeR: Robust, Agentic, and Structured Temporal Reasoning</strong> - Dan Schumacher, Fatemeh Haji, Tara Grey, Niharika Bandlamudi, Nupoor Karnik, Gagana Uday Kumar, Jason Cho-Yu Chiang, Paul Rad, Nishant Vishwamitra, Anthony Rios - [[pdf]](https://arxiv.org/pdf/2406.19538)</summary>

**Abstract:** Temporal question answering (TQA) remains a challenge for large language models (LLMs), particularly when retrieved content may be irrelevant, outdated, or temporally inconsistent. This is especially critical in applications like clinical event ordering, and policy tracking, which require reliable temporal reasoning even under noisy or outdated information. To address this challenge, we introduce RASTeR: \textbf{R}obust, \textbf{A}gentic, and \textbf{S}tructured, \textbf{Te}mporal \textbf{R}easoning, a prompting framework that separates context evaluation from answer generation. RASTeR first assesses the relevance and temporal coherence of the retrieved context, then constructs a temporal knolwedge graph (TKG) to better facilitate reasoning. When inconsistencies are detected, RASTeR selectively corrects or discards context before generating an answer. Across multiple datasets and LLMs, RASTeR consistently improves robustness\footnote{\ Some TQA work defines robustness as handling diverse temporal phenomena. Here, we define it as the ability to answer correctly despite suboptimal context}. We further validate our approach through a ``needle-in-the-haystack'' study, in which relevant context is buried among distractors. With forty distractors, RASTeR achieves 75\% accuracy, over 12\% ahead of the runner up

**arXiv ID:** 2406.19538
</details>

<details>
<summary><strong>WetExplorer: Automating Wetland Greenhouse-Gas Surveys with an Autonomous Mobile Robot</strong> - Jose Vasquez, Xuping Zhang - [[pdf]](https://arxiv.org/pdf/2511.10864)</summary>

**Abstract:** Quantifying greenhouse-gases (GHG) in wetlands is critical for climate modeling and restoration assessment, yet manual sampling is labor-intensive, and time demanding. We present WetExplorer, an autonomous tracked robot that automates the full GHG-sampling workflow. The robot system integrates low-ground-pressure locomotion, centimeter-accurate lift placement, dual-RTK sensor fusion, obstacle avoidance planning, and deep-learning perception in a containerized ROS2 stack. Outdoor trials verified that the sensor-fusion stack maintains a mean localization error of 1.71 cm, the vision module estimates object pose with 7 mm translational and 3° rotational accuracy, while indoor trials demonstrated that the full motion-planning pipeline positions the sampling chamber within a global tolerance of 70 mm while avoiding obstacles, all without human intervention. By eliminating the manual bottleneck, WetExplorer enables high-frequency, multi-site GHG measurements and opens the door for dense, long-duration datasets in saturated wetland terrain.

**arXiv ID:** 2511.10864
</details>

</details>

<details open>
<summary><h2>LLM Agents (2 papers)</h2></summary>

<details>
<summary><strong>AIonopedia: an LLM agent orchestrating multimodal learning for ionic liquid discovery</strong> - Yuqi Yin, Yibo Fu, Siyuan Wang, Peng Sun, Hongyu Wang, Xiaohui Wang, Lei Zheng, Zhiyong Li, Zhirong Liu, Jianji Wang, Zhaoxi Sun - [[pdf]](https://arxiv.org/pdf/2511.11257)</summary>

**Abstract:** The discovery of novel Ionic Liquids (ILs) is hindered by critical challenges in property prediction, including limited data, poor model accuracy, and fragmented workflows. Leveraging the power of Large Language Models (LLMs), we introduce AIonopedia, to the best of our knowledge, the first LLM agent for IL discovery. Powered by an LLM-augmented multimodal domain foundation model for ILs, AIonopedia enables accurate property predictions and incorporates a hierarchical search architecture for molecular screening and design. Trained and evaluated on a newly curated and comprehensive IL dataset, our model delivers superior performance. Complementing these results, evaluations on literature-reported systems indicate that the agent can perform effective IL modification. Moving beyond offline tests, the practical efficacy was further confirmed through real-world wet-lab validation, in which the agent demonstrated exceptional generalization capabilities on challenging out-of-distribution tasks, underscoring its ability to accelerate real-world IL discovery.

**arXiv ID:** 2511.11257
</details>

<details>
<summary><strong>RAG-Enhanced Collaborative LLM Agents for Drug Discovery</strong> - Namkyeong Lee, Edward De Brouwer, Ehsan Hajiramezanali, Tommaso Biancalani, Chanyoung Park, Gabriele Scalia - [[pdf]](https://arxiv.org/pdf/2502.17506)</summary>

**Abstract:** Recent advances in large language models (LLMs) have shown great potential to accelerate drug discovery. However, the specialized nature of biochemical data often necessitates costly domain-specific fine-tuning, posing major challenges. First, it hinders the application of more flexible general-purpose LLMs for cutting-edge drug discovery tasks. More importantly, it limits the rapid integration of the vast amounts of scientific data continuously generated through experiments and research. Compounding these challenges is the fact that real-world scientific questions are typically complex and open-ended, requiring reasoning beyond pattern matching or static knowledge this http URL address these challenges, we propose CLADD, a retrieval-augmented generation (RAG)-empowered agentic system tailored to drug discovery tasks. Through the collaboration of multiple LLM agents, CLADD dynamically retrieves information from biomedical knowledge bases, contextualizes query molecules, and integrates relevant evidence to generate responses - all without the need for domain-specific fine-tuning. Crucially, we tackle key obstacles in applying RAG workflows to biochemical data, including data heterogeneity, ambiguity, and multi-source integration. We demonstrate the flexibility and effectiveness of this framework across a variety of drug discovery tasks, showing that it outperforms general-purpose and domain-specific LLMs as well as traditional deep learning approaches. Our code is publicly available at this https URL.

**arXiv ID:** 2502.17506
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (21 papers)</h2></summary>

<details>
<summary><strong>Advanced Tool for Traffic Crash Analysis: An AI-Driven Multi-Agent Approach to Pre-Crash Reconstruction</strong> - Gerui Xu, Boyou Chen, Huizhong Guo, Dave LeBlanc, Ananna Ahmed, Zhaonan Sun, Shan Bao - [[pdf]](https://arxiv.org/pdf/2511.10853)</summary>

**Abstract:** Traffic collision reconstruction traditionally relies on human expertise, often yielding inconsistent results when analyzing incomplete multimodal data. This study develops a multi-agent AI framework that reconstructs pre-crash scenarios and infers vehicle behaviors from fragmented collision data. We present a two-phase collaborative framework combining reconstruction and reasoning phases. The system processes 277 rear-end lead vehicle deceleration (LVD) collisions from the Crash Investigation Sampling System, integrating textual crash reports, structured tabular data, and visual scene diagrams. Phase I generates natural-language crash reconstructions from multimodal inputs. Phase II performs in-depth crash reasoning by combining these reconstructions with temporal Event Data Recorder (EDR).For validation, we applied it to all LVD cases, focusing on a subset of 39 complex crashes where multiple EDR records per collision introduced ambiguity (e.g., due to missing or conflicting data).The evaluation of the 39 LVD crash cases revealed our framework achieved perfect accuracy across all test cases, successfully identifying both the most relevant EDR event and correctly distinguishing striking versus struck vehicles, surpassing the 92% accuracy achieved by human researchers on the same challenging dataset. The system maintained robust performance even when processing incomplete data, including missing or erroneous EDR records and ambiguous scene diagrams. This study demonstrates superior AI capabilities in processing heterogeneous collision data, providing unprecedented precision in reconstructing impact dynamics and characterizing pre-crash behaviors.

**arXiv ID:** 2511.10853
</details>

<details>
<summary><strong>Multi-Agent Legal Verifier Systems for Data Transfer Planning</strong> - Ha-Thanh Nguyen, Wachara Fungwacharakorn, Ken Satoh - [[pdf]](https://arxiv.org/pdf/2511.10925)</summary>

**Abstract:** Legal compliance in AI-driven data transfer planning is becoming increasingly critical under stringent privacy regulations such as the Japanese Act on the Protection of Personal Information (APPI). We propose a multi-agent legal verifier that decomposes compliance checking into specialized agents for statutory interpretation, business context evaluation, and risk assessment, coordinated through a structured synthesis protocol. Evaluated on a stratified dataset of 200 Amended APPI Article 16 cases with clearly defined ground truth labels and multiple performance metrics, the system achieves 72% accuracy, which is 21 percentage points higher than a single-agent baseline, including 90% accuracy on clear compliance cases (vs. 16% for the baseline) while maintaining perfect detection of clear violations. While challenges remain in ambiguous scenarios, these results show that domain specialization and coordinated reasoning can meaningfully improve legal AI performance, providing a scalable and regulation-aware framework for trustworthy and interpretable automated compliance verification.

**arXiv ID:** 2511.10925
</details>

<details>
<summary><strong>Key Decision-Makers in Multi-Agent Debates: Who Holds the Power?</strong> - Qian Zhang, Yan Zheng, Jinyi Liu, Hebin Liang, Lanjun Wang - [[pdf]](https://arxiv.org/pdf/2511.11040)</summary>

**Abstract:** Recent studies on LLM agent scaling have highlighted the potential of Multi-Agent Debate (MAD) to enhance reasoning abilities. However, the critical aspect of role allocation strategies remains underexplored. In this study, we demonstrate that allocating roles with differing viewpoints to specific positions significantly impacts MAD's performance in reasoning tasks. Specifically, we find a novel role allocation strategy, "Truth Last", which can improve MAD performance by up to 22% in reasoning tasks. To address the issue of unknown truth in practical applications, we propose the Multi-Agent Debate Consistency (MADC) strategy, which systematically simulates and optimizes its core mechanisms. MADC incorporates path consistency to assess agreement among independent roles, simulating the role with the highest consistency score as the truth. We validated MADC across a range of LLMs (9 models), including the DeepSeek-R1 Distilled Models, on challenging reasoning tasks. MADC consistently demonstrated advanced performance, effectively overcoming MAD's performance bottlenecks and providing a crucial pathway for further improvements in LLM agent scaling.

**arXiv ID:** 2511.11040
</details>

<details>
<summary><strong>Multi-agent Undercover Gaming: Hallucination Removal via Counterfactual Test for Multimodal Reasoning</strong> - Dayong Liang, Xiao-Yong Wei, Changmeng Zheng - [[pdf]](https://arxiv.org/pdf/2511.11182)</summary>

**Abstract:** Hallucination continues to pose a major obstacle in the reasoning capabilities of large language models (LLMs). Although the Multi-Agent Debate (MAD) paradigm offers a promising solution by promoting consensus among multiple agents to enhance reliability, it relies on the unrealistic assumption that all debaters are rational and reflective, which is a condition that may not hold when agents themselves are prone to hallucinations. To address this gap, we introduce the Multi-agent Undercover Gaming (MUG) protocol, inspired by social deduction games like "Who is Undercover?". MUG reframes MAD as a process of detecting "undercover" agents (those suffering from hallucinations) by employing multimodal counterfactual tests. Specifically, we modify reference images to introduce counterfactual evidence and observe whether agents can accurately identify these changes, providing ground-truth for identifying hallucinating agents and enabling robust, crowd-powered multimodal reasoning. MUG advances MAD protocols along three key dimensions: (1) enabling factual verification beyond statistical consensus through counterfactual testing; (2) introducing cross-evidence reasoning via dynamically modified evidence sources instead of relying on static inputs; and (3) fostering active reasoning, where agents engage in probing discussions rather than passively answering questions. Collectively, these innovations offer a more reliable and effective framework for multimodal reasoning in LLMs. The source code can be accessed at this https URL.

**arXiv ID:** 2511.11182
</details>

<details>
<summary><strong>UAVBench: An Open Benchmark Dataset for Autonomous and Agentic AI UAV Systems via LLM-Generated Flight Scenarios</strong> - Mohamed Amine Ferrag, Abderrahmane Lakas, Merouane Debbah - [[pdf]](https://arxiv.org/pdf/2511.11252)</summary>

**Abstract:** Autonomous aerial systems increasingly rely on large language models (LLMs) for mission planning, perception, and decision-making, yet the lack of standardized and physically grounded benchmarks limits systematic evaluation of their reasoning capabilities. To address this gap, we introduce UAVBench, an open benchmark dataset comprising 50,000 validated UAV flight scenarios generated through taxonomy-guided LLM prompting and multi-stage safety validation. Each scenario is encoded in a structured JSON schema that includes mission objectives, vehicle configuration, environmental conditions, and quantitative risk labels, providing a unified representation of UAV operations across diverse domains. Building on this foundation, we present UAVBench_MCQ, a reasoning-oriented extension containing 50,000 multiple-choice questions spanning ten cognitive and ethical reasoning styles, ranging from aerodynamics and navigation to multi-agent coordination and integrated reasoning. This framework enables interpretable and machine-checkable assessment of UAV-specific cognition under realistic operational contexts. We evaluate 32 state-of-the-art LLMs, including GPT-5, ChatGPT-4o, Gemini 2.5 Flash, DeepSeek V3, Qwen3 235B, and ERNIE 4.5 300B, and find strong performance in perception and policy reasoning but persistent challenges in ethics-aware and resource-constrained decision-making. UAVBench establishes a reproducible and physically grounded foundation for benchmarking agentic AI in autonomous aerial systems and advancing next-generation UAV reasoning intelligence. To support open science and reproducibility, we release the UAVBench dataset, the UAVBench_MCQ benchmark, evaluation scripts, and all related materials on GitHub at this https URL

**arXiv ID:** 2511.11252
</details>

<details>
<summary><strong>Robust and Efficient Communication in Multi-Agent Reinforcement Learning</strong> - Zejiao Liu, Yi Li, Jiali Wang, Junqi Tu, Yitian Hong, Fangfei Li, Yang Liu, Toshiharu Sugawara, Yang Tang - [[pdf]](https://arxiv.org/pdf/2511.11393)</summary>

**Abstract:** Multi-agent reinforcement learning (MARL) has made significant strides in enabling coordinated behaviors among autonomous agents. However, most existing approaches assume that communication is instantaneous, reliable, and has unlimited bandwidth; these conditions are rarely met in real-world deployments. This survey systematically reviews recent advances in robust and efficient communication strategies for MARL under realistic constraints, including message perturbations, transmission delays, and limited bandwidth. Furthermore, because the challenges of low-latency reliability, bandwidth-intensive data sharing, and communication-privacy trade-offs are central to practical MARL systems, we focus on three applications involving cooperative autonomous driving, distributed simultaneous localization and mapping, and federated learning. Finally, we identify key open challenges and future research directions, advocating a unified approach that co-designs communication, learning, and robustness to bridge the gap between theoretical MARL models and practical implementations.

**arXiv ID:** 2511.11393
</details>

<details>
<summary><strong>MarsRL: Advancing Multi-Agent Reasoning System via Reinforcement Learning with Agentic Pipeline Parallelism</strong> - Shulin Liu, Dong Du, Tao Yang, Yang Li, Boyu Qiu - [[pdf]](https://arxiv.org/pdf/2511.11373)</summary>

**Abstract:** Recent progress in large language models (LLMs) has been propelled by reinforcement learning with verifiable rewards (RLVR) and test-time scaling. However, the limited output length of LLMs constrains the depth of reasoning attainable in a single inference process. Multi-agent reasoning systems offer a promising alternative by employing multiple agents including Solver, Verifier, and Corrector, to iteratively refine solutions. While effective in closed-source models like Gemini 2.5 Pro, they struggle to generalize to open-source models due to insufficient critic and correction capabilities. To address this, we propose MarsRL, a novel reinforcement learning framework with agentic pipeline parallelism, designed to jointly optimize all agents in the system. MarsRL introduces agent-specific reward mechanisms to mitigate reward noise and employs pipeline-inspired training to enhance efficiency in handling long trajectories. Applied to Qwen3-30B-A3B-Thinking-2507, MarsRL improves AIME2025 accuracy from 86.5% to 93.3% and BeyondAIME from 64.9% to 73.8%, even surpassing Qwen3-235B-A22B-Thinking-2507. These findings highlight the potential of MarsRL to advance multi-agent reasoning systems and broaden their applicability across diverse reasoning tasks.

**arXiv ID:** 2511.11373
</details>

<details>
<summary><strong>Who Gets the Reward, Who Gets the Blame? Evaluation-Aligned Training Signals for Multi-LLM Agents</strong> - Chih-Hsuan Yang, Tanwi Mallick, Le Chen, Krishnan Raghavan, Azton Wells, Amal Gueroudji, Ian T. Foster, Rajeev Thakur - [[pdf]](https://arxiv.org/pdf/2511.10687)</summary>

**Abstract:** Large Language Models (LLMs) in multi-agent systems (MAS) have shown promise for complex tasks, yet current training methods lack principled ways to connect system-level evaluation with agent-level and message-level learning. We propose a theoretical framework that unifies cooperative game-theoretic attribution with process reward modeling to transform system evaluation into agent credit and then into response-level signals. Unlike prior approaches that rely only on attribution (e.g., Shapley) or step-level labels (e.g., PRM), our method produces local, signed, and credit-conserving signals. In success cases, Shapley-based credit assignment fairly allocates outcomes across agents and is refined into per-message rewards that promote cooperation while discouraging redundancy or sabotage. In failure cases, first-error localization yields repair-aware preferences that penalize harmful steps while rewarding corrective attempts. The resulting signals are bounded, cooperative, and directly compatible with reinforcement-based or preference-based post-training, providing a unified and auditable pathway from global evaluation to local supervision in LLM multi-agent training. Our contribution is conceptual: we present a theoretical foundation and training signals, leaving empirical validation for future work.

**arXiv ID:** 2511.10687
</details>

<details>
<summary><strong>HPCAgentTester: A Multi-Agent LLM Approach for Enhanced HPC Unit Test Generation</strong> - Rabimba Karanjai, Lei Xu, Weidong Shi - [[pdf]](https://arxiv.org/pdf/2511.10860)</summary>

**Abstract:** Unit testing in High-Performance Computing (HPC) is critical but challenged by parallelism, complex algorithms, and diverse hardware. Traditional methods often fail to address non-deterministic behavior and synchronization issues in HPC applications. This paper introduces HPCAgentTester, a novel multi-agent Large Language Model (LLM) framework designed to automate and enhance unit test generation for HPC software utilizing OpenMP and MPI. HPCAgentTester employs a unique collaborative workflow where specialized LLM agents (Recipe Agent and Test Agent) iteratively generate and refine test cases through a critique loop. This architecture enables the generation of context-aware unit tests that specifically target parallel execution constructs, complex communication patterns, and hierarchical parallelism. We demonstrate HPCAgentTester's ability to produce compilable and functionally correct tests for OpenMP and MPI primitives, effectively identifying subtle bugs that are often missed by conventional techniques. Our evaluation shows that HPCAgentTester significantly improves test compilation rates and correctness compared to standalone LLMs, offering a more robust and scalable solution for ensuring the reliability of parallel software systems.

**arXiv ID:** 2511.10860
</details>

<details>
<summary><strong>AirCopBench: A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning</strong> - Jirong Zha, Yuxuan Fan, Tianyu Zhang, Geng Chen, Yingfeng Chen, Chen Gao, Xinlei Chen - [[pdf]](https://arxiv.org/pdf/2511.11025)</summary>

**Abstract:** Multimodal Large Language Models (MLLMs) have shown promise in single-agent vision tasks, yet benchmarks for evaluating multi-agent collaborative perception remain scarce. This gap is critical, as multi-drone systems provide enhanced coverage, robustness, and collaboration compared to single-sensor setups. Existing multi-image benchmarks mainly target basic perception tasks using high-quality single-agent images, thus failing to evaluate MLLMs in more complex, egocentric collaborative scenarios, especially under real-world degraded perception this http URL address these challenges, we introduce AirCopBench, the first comprehensive benchmark designed to evaluate MLLMs in embodied aerial collaborative perception under challenging perceptual conditions. AirCopBench includes 14.6k+ questions derived from both simulator and real-world data, spanning four key task dimensions: Scene Understanding, Object Understanding, Perception Assessment, and Collaborative Decision, across 14 task types. We construct the benchmark using data from challenging degraded-perception scenarios with annotated collaborative events, generating large-scale questions through model-, rule-, and human-based methods under rigorous quality control. Evaluations on 40 MLLMs show significant performance gaps in collaborative perception tasks, with the best model trailing humans by 24.38% on average and exhibiting inconsistent results across tasks. Fine-tuning experiments further confirm the feasibility of sim-to-real transfer in aerial collaborative perception and reasoning.

**arXiv ID:** 2511.11025
</details>

<details>
<summary><strong>Exposing Weak Links in Multi-Agent Systems under Adversarial Prompting</strong> - Nirmit Arora, Sathvik Joel, Ishan Kavathekar, Palak, Rohan Gandhi, Yash Pandya, Tanuja Ganu, Aditya Kanade, Akshay Nambi - [[pdf]](https://arxiv.org/pdf/2511.10949)</summary>

**Abstract:** LLM-based agents are increasingly deployed in multi-agent systems (MAS). As these systems move toward real-world applications, their security becomes paramount. Existing research largely evaluates single-agent security, leaving a critical gap in understanding the vulnerabilities introduced by multi-agent design. However, existing systems fall short due to lack of unified frameworks and metrics focusing on unique rejection modes in MAS. We present SafeAgents, a unified and extensible framework for fine-grained security assessment of MAS. SafeAgents systematically exposes how design choices such as plan construction strategies, inter-agent context sharing, and fallback behaviors affect susceptibility to adversarial prompting. We introduce Dharma, a diagnostic measure that helps identify weak links within multi-agent pipelines. Using SafeAgents, we conduct a comprehensive study across five widely adopted multi-agent architectures (centralized, decentralized, and hybrid variants) on four datasets spanning web tasks, tool use, and code generation. Our findings reveal that common design patterns carry significant vulnerabilities. For example, centralized systems that delegate only atomic instructions to sub-agents obscure harmful objectives, reducing robustness. Our results highlight the need for security-aware design in MAS. Link to code is this https URL

**arXiv ID:** 2511.10949
</details>

<details>
<summary><strong>Refine and Align: Confidence Calibration through Multi-Agent Interaction in VQA</strong> - Ayush Pandey, Jai Bardhan, Ishita Jain, Ramya S Hebbalaguppe, Rohan Raju Dhanakshirur, Lovekesh Vig - [[pdf]](https://arxiv.org/pdf/2511.11169)</summary>

**Abstract:** In the context of Visual Question Answering (VQA) and Agentic AI, calibration refers to how closely an AI system's confidence in its answers reflects their actual correctness. This aspect becomes especially important when such systems operate autonomously and must make decisions under visual uncertainty. While modern VQA systems, powered by advanced vision-language models (VLMs), are increasingly used in high-stakes domains like medical diagnostics and autonomous navigation due to their improved accuracy, the reliability of their confidence estimates remains under-examined. Particularly, these systems often produce overconfident responses. To address this, we introduce AlignVQA, a debate-based multi-agent framework, in which diverse specialized VLM -- each following distinct prompting strategies -- generate candidate answers and then engage in two-stage interaction: generalist agents critique, refine and aggregate these proposals. This debate process yields confidence estimates that more accurately reflect the model's true predictive performance. We find that more calibrated specialized agents produce better aligned confidences. Furthermore, we introduce a novel differentiable calibration-aware loss function called aligncal designed to fine-tune the specialized agents by minimizing an upper bound on the calibration error. This objective explicitly improves the fidelity of each agent's confidence estimates. Empirical results across multiple benchmark VQA datasets substantiate the efficacy of our approach, demonstrating substantial reductions in calibration discrepancies. Furthermore, we propose a novel differentiable calibration-aware loss to fine-tune the specialized agents and improve the quality of their individual confidence estimates based on minimising upper bound calibration error.

**arXiv ID:** 2511.11169
</details>

<details>
<summary><strong>iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference</strong> - Wei Fan, JinYi Yoon, Bo Ji - [[pdf]](https://arxiv.org/pdf/2511.11306)</summary>

**Abstract:** Large Language Model (LLM) agent systems have advanced rapidly, driven by their strong generalization in zero-shot settings. To further enhance reasoning and accuracy on complex tasks, Multi-Agent Debate (MAD) has emerged as a promising framework that engages multiple LLM agents in structured debates to encourage diverse reasoning. However, triggering MAD for every query is inefficient, as it incurs substantial computational (token) cost and may even degrade accuracy by overturning correct single-agent answers. To address these limitations, we propose intelligent Multi-Agent Debate (iMAD), a token-efficient framework that selectively triggers MAD only when it is likely to be beneficial (i.e., correcting an initially wrong answer). To achieve this goal, iMAD learns generalizable model behaviors to make accurate debate decisions. Specifically, iMAD first prompts a single agent to produce a structured self-critique response, from which we extract 41 interpretable linguistic and semantic features capturing hesitation cues. Then, iMAD uses a lightweight debate-decision classifier, trained using our proposed FocusCal loss, to determine whether to trigger MAD, enabling robust debate decisions without test dataset-specific tuning. Through extensive experiments using six (visual) question answering datasets against five competitive baselines, we have shown that iMAD significantly reduces token usage (by up to 92%) while also improving final answer accuracy (by up to 13.5%).

**arXiv ID:** 2511.11306
</details>

<details>
<summary><strong>Visual Document Understanding and Reasoning: A Multi-Agent Collaboration Framework with Agent-Wise Adaptive Test-Time Scaling</strong> - Xinlei Yu, Chengming Xu, Zhangquan Chen, Yudong Zhang, Shilin Lu, Cheng Yang, Jiangning Zhang, Shuicheng Yan, Xiaobin Hu - [[pdf]](https://arxiv.org/pdf/2508.03404)</summary>

**Abstract:** The dominant paradigm of monolithic scaling in Vision-Language Models (VLMs) is failing for understanding and reasoning in documents, yielding diminishing returns as it struggles with the inherent need of this domain for document-based procedural reasoning, cognitive complexity, and factual accuracy. To this end, we introduce MACT, a Multi-Agent Collaboration framework with agent-wise adaptive Test-time scaling that pioneers a paradigm shift to procedural scaling, adapting dynamically to the functional entities of visual documents understanding and reasoning. MACT decomposes the visual document processing flow into four specialized agents, i.e., planning, execution, judgment, and answer, to resolve cognitive overload and introduce a critical self-correction loop for factual grounding. This collaborative architecture is amplified by an agent-wise adaptive test-time scaling strategy that intelligently allocates computational resources based on the complexity and redundancy of each functionality. Evaluated on multiple visual document understanding benchmarks, MACT achieves superior performance with a smaller parameter scale, adapting effectively to various document scenarios without compromising its general or mathematical reasoning capabilities. The three variants of MACT consistently attain top-three average performance rankings, with average performance enhancements of 9.9-11.5% over the base models. The source code will be released publicly.

**arXiv ID:** 2508.03404
</details>

<details>
<summary><strong>An Adaptive Multi Agent Bitcoin Trading System</strong> - Aadi Singhi - [[pdf]](https://arxiv.org/pdf/2510.08068)</summary>

**Abstract:** This paper presents a Multi Agent Bitcoin Trading system that utilizes Large Language Models (LLMs) for alpha generation and portfolio management in the cryptocurrencies market. Unlike equities, cryptocurrencies exhibit extreme volatility and are heavily influenced by rapidly shifting market sentiments and regulatory announcements, making them difficult to model using static regression models or neural networks trained solely on historical data. The proposed framework overcomes this by structuring LLMs into specialised agents for technical analysis, sentiment evaluation, decision-making, and performance reflection. The agents improve over time via a novel verbal feedback mechanism where a Reflect agent provides daily and weekly natural-language critiques of trading decisions. These textual evaluations are then injected into future prompts of the agents, allowing them to adjust allocation logic without weight updates or finetuning. Back-testing on Bitcoin price data from July 2024 to April 2025 shows consistent outperformance across market regimes: the Quantitative agent delivered over 30\% higher returns in bullish phases and 15\% overall gains versus buy-and-hold, while the sentiment-driven agent turned sideways markets from a small loss into a gain of over 100\%. Adding weekly feedback further improved total performance by 31\% and reduced bearish losses by 10\%. The results demonstrate that verbal feedback represents a new, scalable, and low-cost approach of tuning LLMs for financial goals.

**arXiv ID:** 2510.08068
</details>

<details>
<summary><strong>Towards Assume-Guarantee Verification of Abilities in Stochastic Multi-Agent Systems</strong> - Wojciech Jamroga, Damian Kurpiewski, Łukasz Mikulski - [[pdf]](https://arxiv.org/pdf/2511.10649)</summary>

**Abstract:** Model checking of strategic abilities is a notoriously hard problem, even more so in the realistic case of agents with imperfect information, acting in a stochastic environment. Assume-guarantee reasoning can be of great help here, providing a way to decompose the complex problem into a small set of easier subproblems.
In this paper, we propose several schemes for assume-guarantee verification of probabilistic alternating-time temporal logic with imperfect information. We prove the soundness of the schemes, and discuss their completeness. On the way, we also propose a new variant of (non-probabilistic) alternating-time logic, where the strategic modalities capture "achieving at most $\varphi$," analogous to Levesque's logic of "only knowing."

**arXiv ID:** 2511.10649
</details>

<details>
<summary><strong>GraphMASAL: A Graph-based Multi-Agent System for Adaptive Learning</strong> - Biqing Zeng, Mengquan Liu, Zongwei Zhen - [[pdf]](https://arxiv.org/pdf/2511.11035)</summary>

**Abstract:** The advent of Intelligent Tutoring Systems (ITSs) has marked a paradigm shift in education, enabling highly personalized learning pathways. However, true personalization requires adapting to learners' complex knowledge states (multi-source) and diverse goals (multi-sink); existing ITSs often lack the necessary structural-reasoning capability and knowledge dynamism to generate genuinely effective learning paths, and they lack scientifically rigorous validation paradigms. In this paper we propose GraphMASAL (A Graph-based Multi-Agent System for Adaptive Learning), which integrates (i) a dynamic knowledge graph for persistent, stateful learner modeling; (ii) a LangGraph-orchestrated trio of agents (Diagnostician, Planner, Tutor); (iii) a knowledge-graph-grounded two-stage neural IR component (dual-encoder dense retrieval with cross-encoder listwise re-ranking and calibrated score fusion); and (iv) a multi-source multi-sink (MSMS) planning engine with a cognitively grounded cost and an approximation guarantee via greedy set cover. Under blinded automated evaluations with matched inputs and inference settings across diverse student profiles, GraphMASAL consistently outperforms LLM prompting and structured ablations in planning--achieving stronger structural/sequence alignment of learning paths, higher coverage of weak concepts, and lower learning cost--while also surpassing prompt-based baselines in cognitive diagnosis. Agreement with expert/LLM-proxy ratings further supports the validity of our evaluation protocol. These findings indicate that grounding LLM agents in a dynamic knowledge graph, coupled with optimization under educational constraints, yields reliable, interpretable, and pedagogically plausible learning plans, advancing personalized and goal-oriented education.

**arXiv ID:** 2511.11035
</details>

<details>
<summary><strong>What the flock knows that the birds do not: exploring the emergence of joint agency in multi-agent active inference</strong> - Domenico Maisto, Davide Nuzzi, Giovanni Pezzulo - [[pdf]](https://arxiv.org/pdf/2511.10835)</summary>

**Abstract:** Collective behavior pervades biological systems, from flocks of birds to neural assemblies and human societies. Yet, how such collectives acquire functional properties -- such as joint agency or knowledge -- that transcend those of their individual components remains an open question. Here, we combine active inference and information-theoretic analyses to explore how a minimal system of interacting agents can give rise to joint agency and collective knowledge. We model flocking dynamics using multiple active inference agents, each minimizing its own free energy while coupling reciprocally with its neighbors. We show that as agents self-organize, their interactions define higher-order statistical boundaries (Markov blankets) enclosing a ``flock'' that can be treated as an emergent agent with its own sensory, active, and internal states. When exposed to external perturbations (a ``predator''), the flock exhibits faster, coordinated responses than individual agents, reflecting collective sensitivity to environmental change. Crucially, analyses of synergistic information reveal that the flock encodes information about the predator's location that is not accessible to every individual bird, demonstrating implicit collective knowledge. Together, these results show how informational coupling among active inference agents can generate new levels of autonomy and inference, providing a framework for understanding the emergence of (implicit) collective knowledge and joint agency.

**arXiv ID:** 2511.10835
</details>

<details>
<summary><strong>DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding</strong> - Dawei Zhu, Rui Meng, Jiefeng Chen, Sujian Li, Tomas Pfister, Jinsung Yoon - [[pdf]](https://arxiv.org/pdf/2511.11552)</summary>

**Abstract:** Comprehending long visual documents, where information is distributed across extensive pages of text and visual elements, is a critical but challenging task for modern Vision-Language Models (VLMs). Existing approaches falter on a fundamental challenge: evidence localization. They struggle to retrieve relevant pages and overlook fine-grained details within visual elements, leading to limited performance and model hallucination. To address this, we propose DocLens, a tool-augmented multi-agent framework that effectively ``zooms in'' on evidence like a lens. It first navigates from the full document to specific visual elements on relevant pages, then employs a sampling-adjudication mechanism to generate a single, reliable answer. Paired with Gemini-2.5-Pro, DocLens achieves state-of-the-art performance on MMLongBench-Doc and FinRAGBench-V, surpassing even human experts. The framework's superiority is particularly evident on vision-centric and unanswerable queries, demonstrating the power of its enhanced localization capabilities.

**arXiv ID:** 2511.11552
</details>

<details>
<summary><strong>Miniature Testbed for Validating Multi-Agent Cooperative Autonomous Driving</strong> - Hyunchul Bae, Eunjae Lee, Jehyeop Han, Minhee Kang, Jaehyeon Kim, Junggeun Seo, Minkyun Noh, Heejin Ahn - [[pdf]](https://arxiv.org/pdf/2511.11022)</summary>

**Abstract:** Cooperative autonomous driving, which extends vehicle autonomy by enabling real-time collaboration between vehicles and smart roadside infrastructure, remains a challenging yet essential problem. However, none of the existing testbeds employ smart infrastructure equipped with sensing, edge computing, and communication capabilities. To address this gap, we design and implement a 1:15-scale miniature testbed, CIVAT, for validating cooperative autonomous driving, consisting of a scaled urban map, autonomous vehicles with onboard sensors, and smart infrastructure. The proposed testbed integrates V2V and V2I communication with the publish-subscribe pattern through a shared Wi-Fi and ROS2 framework, enabling information exchange between vehicles and infrastructure to realize cooperative driving functionality. As a case study, we validate the system through infrastructure-based perception and intersection management experiments.

**arXiv ID:** 2511.11022
</details>

<details>
<summary><strong>GELATO: Multi-Instruction Trajectory Reshaping via Geometry-Aware Multiagent-based Orchestration</strong> - Junhui Huang, Yuhe Gong, Changsheng Li, Xingguang Duan, Luis Figueredo - [[pdf]](https://arxiv.org/pdf/2509.06031)</summary>

**Abstract:** We present GELATO -- the first language-driven trajectory reshaping framework to embed geometric environment awareness and multi-agent feedback orchestration to support multi-instruction in human-robot interaction scenarios. Unlike prior learning-based methods, our approach automatically registers scene objects as 6D geometric primitives via a VLM-assisted multi-view pipeline, and an LLM translates free-form multiple instructions into explicit, verifiable geometric constraints. These are integrated into a geometric-aware vector field optimization to adapt initial trajectories while preserving smoothness, feasibility, and clearance. We further introduce a multi-agent orchestration with observer-based refinement to handle multi-instruction inputs and interactions among objectives -- increasing success rate without retraining. Simulation and real-world experiments demonstrate our method achieves smoother, safer, and more interpretable trajectory modifications compared to state-of-the-art baselines.

**arXiv ID:** 2509.06031
</details>

</details>

<details open>
<summary><h2>Other Agent Research (8 papers)</h2></summary>

<details>
<summary><strong>Autonomous Vehicle Path Planning by Searching With Differentiable Simulation</strong> - Asen Nachkov, Jan-Nico Zaech, Danda Pani Paudel, Xi Wang, Luc Van Gool - [[pdf]](https://arxiv.org/pdf/2511.11043)</summary>

**Abstract:** Planning allows an agent to safely refine its actions before executing them in the real world. In autonomous driving, this is crucial to avoid collisions and navigate in complex, dense traffic scenarios. One way to plan is to search for the best action sequence. However, this is challenging when all necessary components - policy, next-state predictor, and critic - have to be learned. Here we propose Differentiable Simulation for Search (DSS), a framework that leverages the differentiable simulator Waymax as both a next state predictor and a critic. It relies on the simulator's hardcoded dynamics, making state predictions highly accurate, while utilizing the simulator's differentiability to effectively search across action sequences. Our DSS agent optimizes its actions using gradient descent over imagined future trajectories. We show experimentally that DSS - the combination of planning gradients and stochastic search - significantly improves tracking and path planning accuracy compared to sequence prediction, imitation learning, model-free RL, and other planning methods.

**arXiv ID:** 2511.11043
</details>

<details>
<summary><strong>Unsupervised Cycle Detection in Agentic Applications</strong> - Felix George, Harshit Kumar, Divya Pathak, Kaustabha Ray, Mudit Verma, Pratibha Moogi - [[pdf]](https://arxiv.org/pdf/2511.10650)</summary>

**Abstract:** Agentic applications powered by Large Language Models exhibit non-deterministic behaviors that can form hidden execution cycles, silently consuming resources without triggering explicit errors. Traditional observability platforms fail to detect these costly inefficiencies. We present an unsupervised cycle detection framework that combines structural and semantic analysis. Our approach first applies computationally efficient temporal call stack analysis to identify explicit loops and then leverages semantic similarity analysis to uncover subtle cycles characterized by redundant content generation. Evaluated on 1575 trajectories from a LangGraph-based stock market application, our hybrid approach achieves an F1 score of 0.72 (precision: 0.62, recall: 0.86), significantly outperforming individual structural (F1: 0.08) and semantic methods (F1: 0.28). While these results are encouraging, there remains substantial scope for improvement, and future work is needed to refine the approach and address its current limitations.

**arXiv ID:** 2511.10650
</details>

<details>
<summary><strong>Optimal Welfare in Noncooperative Network Formation under Attack</strong> - Natan Doubez, Pascal Lenzner, Marcus Wunderlich - [[pdf]](https://arxiv.org/pdf/2511.10845)</summary>

**Abstract:** Communication networks are essential for our economy and our everyday lives. This makes them lucrative targets for attacks. Today, we see an ongoing battle between criminals that try to disrupt our key communication networks and security professionals that try to mitigate these attacks. However, today's networks, like the Internet or peer-to-peer networks among smart devices, are not controlled by a single authority, but instead consist of many independently administrated entities that are interconnected. Thus, both the decisions of how to interconnect and how to secure against potential attacks are taken in a decentralized way by selfish agents.
This strategic setting, with agents that want to interconnect and potential attackers that want to disrupt the network, was captured via an influential game-theoretic model by Goyal, Jabbari, Kearns, Khanna, and Morgenstern (WINE 2016). We revisit this model and show improved tight bounds on the achieved robustness of networks created by selfish agents. As our main result, we show that such networks can resist attacks of a large class of potential attackers, i.e., these networks maintain asymptotically optimal welfare post attack. This improves several bounds and resolves an open problem. Along the way, we show the counter-intuitive result, that attackers that aim at minimizing the social welfare post attack do not actually inflict the greatest possible damage.

**arXiv ID:** 2511.10845
</details>

<details>
<summary><strong>Building the Web for Agents: A Declarative Framework for Agent-Web Interaction</strong> - Sven Schultze, Meike Verena Kietzmann, Nils-Lucas Schönfeld, Ruth Stock-Homburg - [[pdf]](https://arxiv.org/pdf/2511.11287)</summary>

**Abstract:** The increasing deployment of autonomous AI agents on the web is hampered by a fundamental misalignment: agents must infer affordances from human-oriented user interfaces, leading to brittle, inefficient, and insecure interactions. To address this, we introduce VOIX, a web-native framework that enables websites to expose reliable, auditable, and privacy-preserving capabilities for AI agents through simple, declarative HTML elements. VOIX introduces <tool> and <context> tags, allowing developers to explicitly define available actions and relevant state, thereby creating a clear, machine-readable contract for agent behavior. This approach shifts control to the website developer while preserving user privacy by disconnecting the conversational interactions from the website. We evaluated the framework's practicality, learnability, and expressiveness in a three-day hackathon study with 16 developers. The results demonstrate that participants, regardless of prior experience, were able to rapidly build diverse and functional agent-enabled web applications. Ultimately, this work provides a foundational mechanism for realizing the Agentic Web, enabling a future of seamless and secure human-AI collaboration on the web.

**arXiv ID:** 2511.11287
</details>

<details>
<summary><strong>Human-AI collaborative autonomous synthesis with pulsed laser deposition for remote epitaxy</strong> - Asraful Haque, Daniel T. Yimam, Jawad Chowdhury, Ralph Bulanadi, Ivan Vlassiouk, John Lasseter, Sujoy Ghosh, Christopher M. Rouleau, Kai Xiao, Yongtao Liu, Eva Zarkadoula, Rama K. Vasudevan, Sumner B. Harris - [[pdf]](https://arxiv.org/pdf/2511.11558)</summary>

**Abstract:** 

**arXiv ID:** 2511.11558
</details>

<details>
<summary><strong>Designing AI-Agents with Personalities: A Psychometric Approach</strong> - Muhua Huang, Xijuan Zhang, Christopher Soto, James Evans - [[pdf]](https://arxiv.org/pdf/2410.19238)</summary>

**Abstract:** We introduce a methodology for assigning quantifiable and psychometrically validated personalities to AI-Agents using the Big Five framework. Across three studies, we evaluate its feasibility and limitations. In Study 1, we show that large language models (LLMs) capture semantic similarities among Big Five measures, providing a basis for personality assignment. In Study 2, we create AI-Agents using prompts designed based on the Big Five Inventory-2 (BFI-2) in different format, and find that AI-Agents powered by new models align more closely with human responses on the Mini-Markers test, although the finer pattern of results (e.g., factor loading patterns) were sometimes inconsistent. In Study 3, we validate our AI-Agents on risk-taking and moral dilemma vignettes, finding that models prompted with the BFI-2-Expanded format most closely reproduce human personality-decision associations, while safety-aligned models generally inflate 'moral' ratings. Overall, our results show that AI-Agents align with humans in correlations between input Big Five traits and output responses and may serve as useful tools for preliminary research. Nevertheless, discrepancies in finer response patterns indicate that AI-Agents cannot (yet) fully substitute for human participants in precision or high-stakes projects.

**arXiv ID:** 2410.19238
</details>

<details>
<summary><strong>Invisible Triggers, Visible Threats! Road-Style Adversarial Creation Attack for Visual 3D Detection in Autonomous Driving</strong> - Jian Wang, Lijun He, Yixing Yong, Haixia Bi, Fan Li - [[pdf]](https://arxiv.org/pdf/2511.08015)</summary>

**Abstract:** Modern autonomous driving (AD) systems leverage 3D object detection to perceive foreground objects in 3D environments for subsequent prediction and planning. Visual 3D detection based on RGB cameras provides a cost-effective solution compared to the LiDAR paradigm. While achieving promising detection accuracy, current deep neural network-based models remain highly susceptible to adversarial examples. The underlying safety concerns motivate us to investigate realistic adversarial attacks in AD scenarios. Previous work has demonstrated the feasibility of placing adversarial posters on the road surface to induce hallucinations in the detector. However, the unnatural appearance of the posters makes them easily noticeable by humans, and their fixed content can be readily targeted and defended. To address these limitations, we propose the AdvRoad to generate diverse road-style adversarial posters. The adversaries have naturalistic appearances resembling the road surface while compromising the detector to perceive non-existent objects at the attack locations. We employ a two-stage approach, termed Road-Style Adversary Generation and Scenario-Associated Adaptation, to maximize the attack effectiveness on the input scene while ensuring the natural appearance of the poster, allowing the attack to be carried out stealthily without drawing human attention. Extensive experiments show that AdvRoad generalizes well to different detectors, scenes, and spoofing locations. Moreover, physical attacks further demonstrate the practical threats in real-world environments.

**arXiv ID:** 2511.08015
</details>

<details>
<summary><strong>MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation</strong> - Xun Huang, Shijia Zhao, Yunxiang Wang, Xin Lu, Wanfa Zhang, Rongsheng Qu, Weixin Li, Yunhong Wang, Chenglu Wen - [[pdf]](https://arxiv.org/pdf/2511.10376)</summary>

**Abstract:** Embodied navigation is a fundamental capability for robotic agents operating. Real-world deployment requires open vocabulary generalization and low training overhead, motivating zero-shot methods rather than task-specific RL training. However, existing zero-shot methods that build explicit 3D scene graphs often compress rich visual observations into text-only relations, leading to high construction cost, irreversible loss of visual evidence, and constrained vocabularies. To address these limitations, we introduce the Multi-modal 3D Scene Graph (M3DSG), which preserves visual cues by replacing textual relation

**arXiv ID:** 2511.10376
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (20 papers)</h2></summary>

<details>
<summary><strong>Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents</strong> - Yuan Zhao, Hualei Zhu, Tingyu Jiang, Shen Li, Xiaohang Xu, Hao Henry Wang - [[pdf]](https://arxiv.org/pdf/2511.10705)</summary>

**Abstract:** Graphical User Interface (GUI) task automation constitutes a critical frontier in artificial intelligence research. While effective GUI agents synergistically integrate planning and grounding capabilities, current methodologies exhibit two fundamental limitations: (1) insufficient exploitation of cross-model synergies, and (2) over-reliance on synthetic data generation without sufficient utilization. To address these challenges, we propose Co-EPG, a self-iterative training framework for Co-Evolution of Planning and Grounding. Co-EPG establishes an iterative positive feedback loop: through this loop, the planning model explores superior strategies under grounding-based reward guidance via Group Relative Policy Optimization (GRPO), generating diverse data to optimize the grounding model. Concurrently, the optimized Grounding model provides more effective rewards for subsequent GRPO training of the planning model, fostering continuous improvement. Co-EPG thus enables iterative enhancement of agent capabilities through self-play optimization and training data distillation. On the Multimodal-Mind2Web and AndroidControl benchmarks, our framework outperforms existing state-of-the-art methods after just three iterations without requiring external data. The agent consistently improves with each iteration, demonstrating robust self-enhancement capabilities. This work establishes a novel training paradigm for GUI agents, shifting from isolated optimization to an integrated, self-driven co-evolution approach.

**arXiv ID:** 2511.10705
</details>

<details>
<summary><strong>Enhancing Demand-Oriented Regionalization with Agentic AI and Local Heterogeneous Data for Adaptation Planning</strong> - Seyedeh Mobina Noorani, Shangde Gao, Changjie Chen, Karla Saldana Ochoa - [[pdf]](https://arxiv.org/pdf/2511.10857)</summary>

**Abstract:** Conventional planning units or urban regions, such as census tracts, zip codes, or neighborhoods, often do not capture the specific demands of local communities and lack the flexibility to implement effective strategies for hazard prevention or response. To support the creation of dynamic planning units, we introduce a planning support system with agentic AI that enables users to generate demand-oriented regions for disaster planning, integrating the human-in-the-loop principle for transparency and adaptability. The platform is built on a representative initialized spatially constrained self-organizing map (RepSC-SOM), extending traditional SOM with adaptive geographic filtering and region-growing refinement, while AI agents can reason, plan, and act to guide the process by suggesting input features, guiding spatial constraints, and supporting interactive exploration. We demonstrate the capabilities of the platform through a case study on the flooding-related risk in Jacksonville, Florida, showing how it allows users to explore, generate, and evaluate regionalization interactively, combining computational rigor with user-driven decision making.

**arXiv ID:** 2511.10857
</details>

<details>
<summary><strong>RLSLM: A Hybrid Reinforcement Learning Framework Aligning Rule-Based Social Locomotion Model with Human Social Norms</strong> - Yitian Kou, Yihe Gu, Chen Zhou, DanDan Zhu, Shuguang Kuai - [[pdf]](https://arxiv.org/pdf/2511.11323)</summary>

**Abstract:** Navigating human-populated environments without causing discomfort is a critical capability for socially-aware agents. While rule-based approaches offer interpretability through predefined psychological principles, they often lack generalizability and flexibility. Conversely, data-driven methods can learn complex behaviors from large-scale datasets, but are typically inefficient, opaque, and difficult to align with human intuitions. To bridge this gap, we propose RLSLM, a hybrid Reinforcement Learning framework that integrates a rule-based Social Locomotion Model, grounded in empirical behavioral experiments, into the reward function of a reinforcement learning framework. The social locomotion model generates an orientation-sensitive social comfort field that quantifies human comfort across space, enabling socially aligned navigation policies with minimal training. RLSLM then jointly optimizes mechanical energy and social comfort, allowing agents to avoid intrusions into personal or group space. A human-agent interaction experiment using an immersive VR-based setup demonstrates that RLSLM outperforms state-of-the-art rule-based models in user experience. Ablation and sensitivity analyses further show the model's significantly improved interpretability over conventional data-driven methods. This work presents a scalable, human-centered methodology that effectively integrates cognitive science and machine learning for real-world social navigation.

**arXiv ID:** 2511.11323
</details>

<details>
<summary><strong>Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping</strong> - Dena Mujtaba, Brian Hu, Anthony Hoogs, Arslan Basharat - [[pdf]](https://arxiv.org/pdf/2511.11551)</summary>

**Abstract:** The deployment of decision-making AI agents presents a critical challenge in maintaining alignment with human values or guidelines while operating in complex, dynamic environments. Agents trained solely to achieve their objectives may adopt harmful behavior, exposing a key trade-off between maximizing the reward function and maintaining the alignment. For the pre-trained agents, ensuring alignment is particularly challenging, as retraining can be a costly and slow process. This is further complicated by the diverse and potentially conflicting attributes representing the ethical values for alignment. To address these challenges, we propose a test-time alignment technique based on model-guided policy shaping. Our method allows precise control over individual behavioral attributes, generalizes across diverse reinforcement learning (RL) environments, and facilitates a principled trade-off between ethical alignment and reward maximization without requiring agent retraining. We evaluate our approach using the MACHIAVELLI benchmark, which comprises 134 text-based game environments and thousands of annotated scenarios involving ethical decisions. The RL agents are first trained to maximize the reward in their respective games. At test time, we apply policy shaping via scenario-action attribute classifiers to ensure decision alignment with ethical attributes. We compare our approach against prior training-time methods and general-purpose agents, as well as study several types of ethical violations and power-seeking behavior. Our results demonstrate that test-time policy shaping provides an effective and scalable solution for mitigating unethical behavior across diverse environments and alignment attributes.

**arXiv ID:** 2511.11551
</details>

<details>
<summary><strong>Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning</strong> - Alexander W. Goodall, Edwin Hamel-De le Court, Francesco Belardinelli - [[pdf]](https://arxiv.org/pdf/2511.10843)</summary>

**Abstract:** Many reinforcement learning algorithms, particularly those that rely on return estimates for policy improvement, can suffer from poor sample efficiency and training instability due to high-variance return estimates. In this paper we leverage new results from off-policy evaluation; it has recently been shown that well-designed behaviour policies can be used to collect off-policy data for provably lower variance return estimates. This result is surprising as it means collecting data on-policy is not variance optimal. We extend this key insight to the online reinforcement learning setting, where both policy evaluation and improvement are interleaved to learn optimal policies. Off-policy RL has been well studied (e.g., IMPALA), with correct and truncated importance weighted samples for de-biasing and managing variance appropriately. Generally these approaches are concerned with reconciling data collected from multiple workers in parallel, while the policy is updated asynchronously, mismatch between the workers and policy is corrected in a mathematically sound way. Here we consider only one worker - the behaviour policy, which is used to collect data for policy improvement, with provably lower variance return estimates. In our experiments we extend two policy-gradient methods with this regime, demonstrating better sample efficiency and performance over a diverse set of environments.

**arXiv ID:** 2511.10843
</details>

<details>
<summary><strong>Incorporating Spatial Information into Goal-Conditioned Hierarchical Reinforcement Learning via Graph Representations</strong> - Shuyuan Zhang, Zihan Wang, Xiao-Wen Chang, Doina Precup - [[pdf]](https://arxiv.org/pdf/2511.10872)</summary>

**Abstract:** The integration of graphs with Goal-conditioned Hierarchical Reinforcement Learning (GCHRL) has recently gained attention, as intermediate goals (subgoals) can be effectively sampled from graphs that naturally represent the overall task structure in most RL tasks. However, existing approaches typically rely on domain-specific knowledge to construct these graphs, limiting their applicability to new tasks. Other graph-based approaches create graphs dynamically during exploration but struggle to fully utilize them, because they have problems passing the information in the graphs to newly visited states. Additionally, current GCHRL methods face challenges such as sample inefficiency and poor subgoal representation. This paper proposes a solution to these issues by developing a graph encoder-decoder to evaluate unseen states. Our proposed method, Graph-Guided sub-Goal representation Generation RL (G4RL), can be incorporated into any existing GCHRL method when operating in environments with primarily symmetric and reversible transitions to enhance performance across this class of problems. We show that the graph encoder-decoder can be effectively implemented using a network trained on the state graph generated during exploration. Empirical results indicate that leveraging high and low-level intrinsic rewards from the graph encoder-decoder significantly enhances the performance of state-of-the-art GCHRL approaches with an extra small computational cost in dense and sparse reward environments.

**arXiv ID:** 2511.10872
</details>

<details>
<summary><strong>An Efficient Training Pipeline for Reasoning Graphical User Interface Agents</strong> - Georgios Pantazopoulos, Eda B. Özyiğit - [[pdf]](https://arxiv.org/pdf/2511.08172)</summary>

**Abstract:** Visual grounding is the task of localising image regions from natural language queries and is critical for reasoning capable Graphical User Interface agents. Many existing methods rely on massive, noisy synthetic datasets. This work introduces an efficient training pipeline that combines model-based data filtering with parameter-efficient fine-tuning. From 4.8M synthetic examples, 12K clean and diverse instances are curated by first identifying challenging cases, removing misaligned and then selecting a diverse set of multimodal instances. On this data, a 3B-parameter Vision-Language Model is trained under three regimes: supervised fine-tuning, chain-of-thought-augmented fine-tuning, and reinforcement learning via Group Relative Policy Optimization. Models trained with the filtered data and lightweight training strategies match or surpass larger baselines on benchmarks such as ScreenSpot, Multimodal-Mind2Web, and AndroidControl. These results demonstrate that principled data curation and robust adaptation can rival large-scale training, enabling compact yet capable multimodal reasoning agents.

**arXiv ID:** 2511.08172
</details>

<details>
<summary><strong>Strategic Opponent Modeling with Graph Neural Networks, Deep Reinforcement Learning and Probabilistic Topic Modeling</strong> - Georgios Chalkiadakis, Charilaos Akasiadis, Gerasimos Koresis, Stergios Plataniotis, Leonidas Bakopoulos - [[pdf]](https://arxiv.org/pdf/2511.10501)</summary>

**Abstract:** This paper provides a comprehensive review of mainly Graph Neural Networks, Deep Reinforcement Learning, and Probabilistic Topic Modeling methods with a focus on their potential incorporation in strategic multiagent settings. We draw interest in (i) Machine Learning methods currently utilized for uncovering unknown model structures adaptable to the task of strategic opponent modeling, and (ii) the integration of these methods with Game Theoretic concepts that avoid relying on assumptions often invalid in real-world scenarios, such as the Common Prior Assumption (CPA) and the Self-Interest Hypothesis (SIH). We analyze the ability to handle uncertainty and heterogeneity, two characteristics that are very common in real-world application cases, as well as scalability. As a potential answer to effectively modeling relationships and interactions in multiagent settings, we champion the use of Graph Neural Networks (GNN). Such approaches are designed to operate upon graph-structured data, and have been shown to be a very powerful tool for performing tasks such as node classification and link prediction. Next, we review the domain of Reinforcement Learning (RL), and in particular that of Multiagent Deep Reinforcement Learning (MADRL). Following, we describe existing relevant game theoretic solution concepts and consider properties such as fairness and stability. Our review comes complete with a note on the literature that utilizes PTM in domains other than that of document analysis and classification. The capability of PTM to estimate unknown underlying distributions can help with tackling heterogeneity and unknown agent beliefs. Finally, we identify certain open challenges specifically, the need to (i) fit non-stationary environments, (ii) balance the degrees of stability and adaptation, (iii) tackle uncertainty and heterogeneity, (iv) guarantee scalability and solution tractability.

**arXiv ID:** 2511.10501
</details>

<details>
<summary><strong>Pelican-VL 1.0: A Foundation Brain Model for Embodied Intelligence</strong> - Yi Zhang, Che Liu, Xiancong Ren, Hanchu Ni, Shuai Zhang, Zeyuan Ding, Jiayu Hu, Hanzhe Shan, Zhenwei Niu, Zhaoyang Liu, Shuang Liu, Yue Zhao, Junbo Qi, Qinfan Zhang, Dengjie Li, Yidong Wang, Jiachen Luo, Yong Dai, Zenglin Xu, Bin Shen, Qifan Wang, Jian Tang, Xiaozhu Ju - [[pdf]](https://arxiv.org/pdf/2511.00108)</summary>

**Abstract:** This report presents Pelican-VL 1.0, a new family of open-source embodied brain models with parameter scales ranging from 7 billion to 72 billion. Our explicit mission is clearly stated as: To embed powerful intelligence into various embodiments. Pelican-VL 1.0 is currently the largest-scale open-source embodied multimodal brain model. Its core advantage lies in the in-depth integration of data power and intelligent adaptive learning mechanisms. Specifically, metaloop distilled a high-quality dataset from a raw dataset containing 4+ billion tokens. Pelican-VL 1.0 is trained on a large-scale cluster of 1000+ A800 GPUs, consuming over 50k+ A800 GPU-hours per checkpoint. This translates to a 20.3% performance uplift from its base model and outperforms 100B-level open-source counterparts by 10.6%, placing it on par with leading proprietary systems on well-known embodied benchmarks. We establish a novel framework, DPPO (Deliberate Practice Policy Optimization), inspired by human metacognition to train Pelican-VL 1.0. We operationalize this as a metaloop that teaches the AI to practice deliberately, which is a RL-Refine-Diagnose-SFT loop.

**arXiv ID:** 2511.00108
</details>

<details>
<summary><strong>Dynamic Sparsity: Challenging Common Sparsity Assumptions for Learning World Models in Robotic Reinforcement Learning Benchmarks</strong> - Muthukumar Pandaram, Jakob Hollenstein, David Drexel, Samuele Tosatto, Antonio Rodríguez-Sánchez, Justus Piater - [[pdf]](https://arxiv.org/pdf/2511.08086)</summary>

**Abstract:** The use of learned dynamics models, also known as world models, can improve the sample efficiency of reinforcement learning. Recent work suggests that the underlying causal graphs of such dynamics models are sparsely connected, with each of the future state variables depending only on a small subset of the current state variables, and that learning may therefore benefit from sparsity priors. Similarly, temporal sparsity, i.e. sparsely and abruptly changing local dynamics, has also been proposed as a useful inductive bias.
In this work, we critically examine these assumptions by analyzing ground-truth dynamics from a set of robotic reinforcement learning environments in the MuJoCo Playground benchmark suite, aiming to determine whether the proposed notions of state and temporal sparsity actually tend to hold in typical reinforcement learning tasks.
We study (i) whether the causal graphs of environment dynamics are sparse, (ii) whether such sparsity is state-dependent, and (iii) whether local system dynamics change sparsely.
Our results indicate that global sparsity is rare, but instead the tasks show local, state-dependent sparsity in their dynamics and this sparsity exhibits distinct structures, appearing in temporally localized clusters (e.g., during contact events) and affecting specific subsets of state dimensions. These findings challenge common sparsity prior assumptions in dynamics learning, emphasizing the need for grounded inductive biases that reflect the state-dependent sparsity structure of real-world dynamics.

**arXiv ID:** 2511.08086
</details>

<details>
<summary><strong>LoRaCompass: Robust Reinforcement Learning to Efficiently Search for a LoRa Tag</strong> - Tianlang He, Zhongming Lin, Tianrui Jiang, S.-H. Gary Chan - [[pdf]](https://arxiv.org/pdf/2511.11190)</summary>

**Abstract:** The Long-Range (LoRa) protocol, known for its extensive range and low power, has increasingly been adopted in tags worn by mentally incapacitated persons (MIPs) and others at risk of going missing. We study the sequential decision-making process for a mobile sensor to locate a periodically broadcasting LoRa tag with the fewest moves (hops) in general, unknown environments, guided by the received signal strength indicator (RSSI). While existing methods leverage reinforcement learning for search, they remain vulnerable to domain shift and signal fluctuation, resulting in cascading decision errors that culminate in substantial localization inaccuracies. To bridge this gap, we propose LoRaCompass, a reinforcement learning model designed to achieve robust and efficient search for a LoRa tag. For exploitation under domain shift and signal fluctuation, LoRaCompass learns a robust spatial representation from RSSI to maximize the probability of moving closer to a tag, via a spatially-aware feature extractor and a policy distillation loss function. It further introduces an exploration function inspired by the upper confidence bound (UCB) that guides the sensor toward the tag with increasing confidence. We have validated LoRaCompass in ground-based and drone-assisted scenarios within diverse unseen environments covering an area of over 80km^2. It has demonstrated high success rate (>90%) in locating the tag within 100m proximity (a 40% improvement over existing methods) and high efficiency with a search path length (in hops) that scales linearly with the initial distance.

**arXiv ID:** 2511.11190
</details>

<details>
<summary><strong>Multi-Phase Spacecraft Trajectory Optimization via Transformer-Based Reinforcement Learning</strong> - Amit Jain, Victor Rodriguez-Fernandez, Richard Linares - [[pdf]](https://arxiv.org/pdf/2511.11402)</summary>

**Abstract:** Autonomous spacecraft control for mission phases such as launch, ascent, stage separation, and orbit insertion remains a critical challenge due to the need for adaptive policies that generalize across dynamically distinct regimes. While reinforcement learning (RL) has shown promise in individual astrodynamics tasks, existing approaches often require separate policies for distinct mission phases, limiting adaptability and increasing operational complexity. This work introduces a transformer-based RL framework that unifies multi-phase trajectory optimization through a single policy architecture, leveraging the transformer's inherent capacity to model extended temporal contexts. Building on proximal policy optimization (PPO), our framework replaces conventional recurrent networks with a transformer encoder-decoder structure, enabling the agent to maintain coherent memory across mission phases spanning seconds to minutes during critical operations. By integrating a Gated Transformer-XL (GTrXL) architecture, the framework eliminates manual phase transitions while maintaining stability in control decisions. We validate our approach progressively: first demonstrating near-optimal performance on single-phase benchmarks (double integrator and Van der Pol oscillator), then extending to multiphase waypoint navigation variants, and finally tackling a complex multiphase rocket ascent problem that includes atmospheric flight, stage separation, and vacuum operations. Results demonstrate that the transformer-based framework not only matches analytical solutions in simple cases but also effectively learns coherent control policies across dynamically distinct regimes, establishing a foundation for scalable autonomous mission planning that reduces reliance on phase-specific controllers while maintaining compatibility with safety-critical verification protocols.

**arXiv ID:** 2511.11402
</details>

<details>
<summary><strong>Provable Domain Adaptation for Offline Reinforcement Learning with Limited Samples</strong> - Weiqin Chen, Xinjie Zhang, Sandipan Mishra, Santiago Paternain - [[pdf]](https://arxiv.org/pdf/2408.12136)</summary>

**Abstract:** Offline reinforcement learning (RL) learns effective policies from a static target dataset. The performance of state-of-the-art offline RL algorithms notwithstanding, it relies on the size of the target dataset, and it degrades if limited samples in the target dataset are available, which is often the case in real-world applications. To address this issue, domain adaptation that leverages auxiliary samples from related source datasets (such as simulators) can be beneficial. However, establishing the optimal way to trade off the limited target dataset and the large-but-biased source dataset while ensuring provably theoretical guarantees remains an open challenge. To the best of our knowledge, this paper proposes the first framework that theoretically explores the impact of the weights assigned to each dataset on the performance of offline RL. In particular, we establish performance bounds and the existence of the optimal weight, which can be computed in closed form under simplifying assumptions. We also provide algorithmic guarantees in terms of convergence to a neighborhood of the optimum. Notably, these results depend on the quality of the source dataset and the number of samples in the target dataset. Our empirical results on the well-known offline Procgen benchmark substantiate the theoretical contributions in this work.

**arXiv ID:** 2408.12136
</details>

<details>
<summary><strong>DRMD: Deep Reinforcement Learning for Malware Detection under Concept Drift</strong> - Shae McFadden, Myles Foley, Mario D'Onghia, Chris Hicks, Vasilios Mavroudis, Nicola Paoletti, Fabio Pierazzi - [[pdf]](https://arxiv.org/pdf/2508.18839)</summary>

**Abstract:** Malware detection in real-world settings must deal with evolving threats, limited labeling budgets, and uncertain predictions. Traditional classifiers, without additional mechanisms, struggle to maintain performance under concept drift in malware domains, as their supervised learning formulation cannot optimize when to defer decisions to manual labeling and adaptation. Modern malware detection pipelines combine classifiers with monthly active learning (AL) and rejection mechanisms to mitigate the impact of concept drift. In this work, we develop a novel formulation of malware detection as a one-step Markov Decision Process and train a deep reinforcement learning (DRL) agent, simultaneously optimizing sample classification performance and rejecting high-risk samples for manual labeling. We evaluated the joint detection and drift mitigation policy learned by the DRL-based Malware Detection (DRMD) agent through time-aware evaluations on Android malware datasets subject to realistic drift requiring multi-year performance stability. The policies learned under these conditions achieve a higher Area Under Time (AUT) performance compared to standard classification approaches used in the domain, showing improved resilience to concept drift. Specifically, the DRMD agent achieved an average AUT improvement of 8.66 and 10.90 for the classification-only and classification-rejection policies, respectively. Our results demonstrate for the first time that DRL can facilitate effective malware detection and improved resiliency to concept drift in the dynamic setting of Android malware detection.

**arXiv ID:** 2508.18839
</details>

<details>
<summary><strong>Multistep Quasimetric Learning for Scalable Goal-conditioned Reinforcement Learning</strong> - Bill Chunyuan Zheng, Vivek Myers, Benjamin Eysenbach, Sergey Levine - [[pdf]](https://arxiv.org/pdf/2511.07730)</summary>

**Abstract:** Learning how to reach goals in an environment is a longstanding challenge in AI, yet reasoning over long horizons remains a challenge for modern methods. The key question is how to estimate the temporal distance between pairs of observations. While temporal difference methods leverage local updates to provide optimality guarantees, they often perform worse than Monte Carlo methods that perform global updates (e.g., with multi-step returns), which lack such guarantees. We show how these approaches can be integrated into a practical GCRL method that fits a quasimetric distance using a multistep Monte-Carlo return. We show our method outperforms existing GCRL methods on long-horizon simulated tasks with up to 4000 steps, even with visual observations. We also demonstrate that our method can enable stitching in the real-world robotic manipulation domain (Bridge setup). Our approach is the first end-to-end GCRL method that enables multistep stitching in this real-world manipulation domain from an unlabeled offline dataset of visual observations.

**arXiv ID:** 2511.07730
</details>

<details>
<summary><strong>DiAReL: Reinforcement Learning with Disturbance Awareness for Robust Sim2Real Policy Transfer in Robot Control</strong> - Mohammadhossein Malmir, Josip Josifovski, Noah Klarmann, Alois Knoll - [[pdf]](https://arxiv.org/pdf/2306.09010)</summary>

**Abstract:** Delayed Markov decision processes (DMDPs) fulfill the Markov property by augmenting the state space of agents with a finite time window of recently committed actions. In reliance on these state augmentations, delay-resolved reinforcement learning algorithms train policies to learn optimal interactions with environments featuring observation or action delays. Although such methods can be directly trained on the real robots, due to sample inefficiency, limited resources, or safety constraints, a common approach is to transfer models trained in simulation to the physical robot. However, robotic simulations rely on approximated models of the physical systems, which hinders the sim2real transfer. In this work, we consider various uncertainties in modeling the robot or environment dynamics as unknown intrinsic disturbances applied to the system input. We introduce the disturbance-augmented Markov decision process (DAMDP) in delayed settings as a novel representation to incorporate disturbance estimation in training on-policy reinforcement learning algorithms. The proposed method is validated across several metrics on learning robotic reaching and pushing tasks and compared with disturbance-unaware baselines. The results show that the disturbance-augmented models can achieve higher stabilization and robustness in the control response, which in turn improves the prospects of successful sim2real transfer.

**arXiv ID:** 2306.09010
</details>

<details>
<summary><strong>Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning</strong> - Chenhao Liu, Leyun Jiang, Yibo Wang, Kairan Yao, Jinchen Fu, Xiaoyu Ren - [[pdf]](https://arxiv.org/pdf/2511.11218)</summary>

**Abstract:** Humanoid robots have demonstrated strong capability for interacting with deterministic scenes across locomotion, manipulation, and more challenging loco-manipulation tasks. Yet the real world is dynamic, quasi-static interactions are insufficient to cope with the various environmental conditions. As a step toward more dynamic interaction scenario, we present a reinforcement-learning-based training pipeline that produces a unified whole-body controller for humanoid badminton, enabling coordinated lower-body footwork and upper-body striking without any motion priors or expert demonstrations. Training follows a three-stage curriculum: first footwork acquisition, then precision-guided racket swing generation, and finally task-focused refinement, yielding motions in which both legs and arms serve the hitting objective. For deployment, we incorporate an Extended Kalman Filter (EKF) to estimate and predict shuttlecock trajectories for target striking. We also introduce a prediction-free variant that dispenses with EKF and explicit trajectory prediction. To validate the framework, we conduct five sets of experiment in both simulation and the real world. In simulation, two robots sustain a rally of 21 consecutive hits. Moreover, the prediction-free variant achieves successful hits with comparable performance relative to the target-known policy. In real-world tests, both the prediction and controller module exhibit high accuracy, and on-court hitting achieves an outgoing shuttle speed up to 10 m/s with a mean return landing distance of 3.5 m. These experiment results show that our humanoid robot can deliver highly dynamic while precise goal striking in badminton, and can be adapted to more dynamism critical domains.

**arXiv ID:** 2511.11218
</details>

<details>
<summary><strong>Sashimi-Bot: Autonomous Tri-manual Advanced Manipulation and Cutting of Deformable Objects</strong> - Sverre Herland, Amit Parag, Elling Ruud Øye, Fangyi Zhang, Fouad Makiyeh, Aleksander Lillienskiold, Abhaya Pal Singh, Edward H. Adelson, Francois Chaumette, Alexandre Krupa, Peter Corke, Ekrem Misimi - [[pdf]](https://arxiv.org/pdf/2511.11223)</summary>

**Abstract:** Advanced robotic manipulation of deformable, volumetric objects remains one of the greatest challenges due to their pliancy, frailness, variability, and uncertainties during interaction. Motivated by these challenges, this article introduces Sashimi-Bot, an autonomous multi-robotic system for advanced manipulation and cutting, specifically the preparation of sashimi. The objects that we manipulate, salmon loins, are natural in origin and vary in size and shape, they are limp and deformable with poorly characterized elastoplastic parameters, while also being slippery and hard to hold. The three robots straighten the loin; grasp and hold the knife; cut with the knife in a slicing motion while cooperatively stabilizing the loin during cutting; and pick up the thin slices from the cutting board or knife blade. Our system combines deep reinforcement learning with in-hand tool shape manipulation, in-hand tool cutting, and feedback of visual and tactile information to achieve robustness to the variabilities inherent in this task. This work represents a milestone in robotic manipulation of deformable, volumetric objects that may inspire and enable a wide range of other real-world applications.

**arXiv ID:** 2511.11223
</details>

<details>
<summary><strong>Semantic VLM Dataset for Safe Autonomous Driving</strong> - Yuankai He, Weisong Shi - [[pdf]](https://arxiv.org/pdf/2511.10701)</summary>

**Abstract:** CAR-Scenes is a frame-level dataset for autonomous driving that enables training and evaluation of vision-language models (VLMs) for interpretable, scene-level understanding. We annotate 5,192 images drawn from Argoverse 1, Cityscapes, KITTI, and nuScenes using a 28-key category/sub-category knowledge base covering environment, road geometry, background-vehicle behavior, ego-vehicle behavior, vulnerable road users, sensor states, and a discrete severity scale (1-10), totaling 350+ leaf attributes. Labels are produced by a GPT-4o-assisted vision-language pipeline with human-in-the-loop verification; we release the exact prompts, post-processing rules, and per-field baseline model performance. CAR-Scenes also provides attribute co-occurrence graphs and JSONL records that support semantic retrieval, dataset triage, and risk-aware scenario mining across sources. To calibrate task difficulty, we include reproducible, non-benchmark baselines, notably a LoRA-tuned Qwen2-VL-2B with deterministic decoding, evaluated via scalar accuracy, micro-averaged F1 for list attributes, and severity MAE/RMSE on a fixed validation split. We publicly release the annotation and analysis scripts, including graph construction and evaluation scripts, to enable explainable, data-centric workflows for future intelligent vehicles. Dataset: this https URL

**arXiv ID:** 2511.10701
</details>

<details>
<summary><strong>AffectGPT-R1: Leveraging Reinforcement Learning for Open-Vocabulary Multimodal Emotion Recognition</strong> - Zheng Lian, Fan Zhang, Yazhou Zhang, Jianhua Tao, Rui Liu, Haoyu Chen, Xiaobai Li - [[pdf]](https://arxiv.org/pdf/2508.01318)</summary>

**Abstract:** Open-Vocabulary Multimodal Emotion Recognition (OV-MER) aims to predict emotions without being constrained by predefined label spaces, enabling fine-grained emotion understanding. Unlike traditional discriminative methods, OV-MER leverages generative models, such as large language models, to capture the full spectrum of emotions and employs emotion wheels (EWs) for metric calculation. Previous approaches (e.g., AffectGPT) primarily rely on token-level loss during training. However, this objective is misaligned with the metrics used in OV-MER, while these metrics cannot be optimized via gradient backpropagation. In this paper, we propose AffectGPT-R1, a reinforcement learning framework that formulates EW-based metrics as a reward function and employs a policy-based optimization strategy to maximize this reward. Additionally, we introduce an extra reasoning process and investigate its necessity in OV-MER. To further refine model behavior, we incorporate auxiliary rewards that constrain both reasoning and emotion prediction. To prevent reward hacking, we propose to incorporate length penalties during training. Experimental results show that AffectGPT-R1 achieves substantial improvements on OV-MER. Beyond this task, our approach also enhances generalized emotion understanding, attaining state-of-the-art performance on MER-UniBench. To the best of our knowledge, this is the first work to adapt the R1-style methodology for emotion understanding, revealing the impact of reasoning processes and reinforcement learning in this domain. Our code is provided in the supplementary material and will be released to facilitate future research.

**arXiv ID:** 2508.01318
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
