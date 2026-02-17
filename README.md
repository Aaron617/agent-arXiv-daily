# Agent arXiv Daily

**Last Updated:** 2026-02-17 03:44:59

**Total Papers:** 62

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
<summary><strong>BrowseComp-$V^3$: A Visual, Vertical, and Verifiable Benchmark for Multimodal Browsing Agents</strong> - Huanyao Zhang, Jiepeng Zhou, Bo Li, Bowen Zhou, Yanzhe Dan, Haishan Lu, Zhiyong Cao, Jiaoyang Chen, Yuqian Han, Zinan Sheng, Zhengwei Tao, Hao Liang, Jialong Wu, Yang Shi, Yuanpeng He, Jiaye Lin, Qintong Zhang, Guochen Yan, Runhao Zhao, Zhengpin Li, Xiaohan Yu, Lang Mei, Chong Chen, Wentao Zhang, Bin Cui - [[pdf]](https://arxiv.org/pdf/2602.12876)</summary>

**Abstract:** Multimodal large language models (MLLMs), equipped with increasingly advanced planning and tool-use capabilities, are evolving into autonomous agents capable of performing multimodal web browsing and deep search in open-world environments. However, existing benchmarks for multimodal browsing remain limited in task complexity, evidence accessibility, and evaluation granularity, hindering comprehensive and reproducible assessments of deep search capabilities. To address these limitations, we introduce BrowseComp-$V^3$, a novel benchmark consisting of 300 carefully curated and challenging questions spanning diverse domains. The benchmark emphasizes deep, multi-level, and cross-modal multi-hop reasoning, where critical evidence is interleaved across textual and visual modalities within and across web pages. All supporting evidence is strictly required to be publicly searchable, ensuring fairness and reproducibility. Beyond final-answer accuracy, we incorporate an expert-validated, subgoal-driven process evaluation mechanism that enables fine-grained analysis of intermediate reasoning behaviors and systematic characterization of capability boundaries. In addition, we propose OmniSeeker, a unified multimodal browsing agent framework integrating diverse web search and visual perception tools. Comprehensive experiments demonstrate that even state-of-the-art models achieve only 36% accuracy on our benchmark, revealing critical bottlenecks in multimodal information integration and fine-grained perception. Our results highlight a fundamental gap between current model capabilities and robust multimodal deep search in real-world settings.

**arXiv ID:** 2602.12876
</details>

<details>
<summary><strong>Robustness of Object Detection of Autonomous Vehicles in Adverse Weather Conditions</strong> - Fox Pettersen, Hong Zhu - [[pdf]](https://arxiv.org/pdf/2602.12902)</summary>

**Abstract:** As self-driving technology advances toward widespread adoption, determining safe operational thresholds across varying environmental conditions becomes critical for public safety. This paper proposes a method for evaluating the robustness of object detection ML models in autonomous vehicles under adverse weather conditions. It employs data augmentation operators to generate synthetic data that simulates different severance degrees of the adverse operation conditions at progressive intensity levels to find the lowest intensity of the adverse conditions at which the object detection model fails. The robustness of the object detection model is measured by the average first failure coefficients (AFFC) over the input images in the benchmark. The paper reports an experiment with four object detection models: YOLOv5s, YOLOv11s, Faster R-CNN, and Detectron2, utilising seven data augmentation operators that simulate weather conditions fog, rain, and snow, and lighting conditions of dark, bright, flaring, and shadow. The experiment data show that the method is feasible, effective, and efficient to evaluate and compare the robustness of object detection models in various adverse operation conditions. In particular, the Faster R-CNN model achieved the highest robustness with an overall average AFFC of 71.9% over all seven adverse conditions, while YOLO variants showed the AFFC values of 43%. The method is also applied to assess the impact of model training that targets adverse operation conditions using synthetic data on model robustness. It is observed that such training can improve robustness in adverse conditions but may suffer from diminishing returns and forgetting phenomena (i.e., decline in robustness) if overtrained.

**arXiv ID:** 2602.12902
</details>

<details>
<summary><strong>Blind Gods and Broken Screens: Architecting a Secure, Intent-Centric Mobile Agent Operating System</strong> - Zhenhua Zou, Sheng Guo, Qiuyang Zhan, Lepeng Zhao, Shuo Li, Qi Li, Ke Xu, Mingwei Xu, Zhuotao Liu - [[pdf]](https://arxiv.org/pdf/2602.10915)</summary>

**Abstract:** The evolution of Large Language Models (LLMs) has shifted mobile computing from App-centric interactions to system-level autonomous agents. Current implementations predominantly rely on a "Screen-as-Interface" paradigm, which inherits structural vulnerabilities and conflicts with the mobile ecosystem's economic foundations. In this paper, we conduct a systematic security analysis of state-of-the-art mobile agents using Doubao Mobile Assistant as a representative case. We decompose the threat landscape into four dimensions - Agent Identity, External Interface, Internal Reasoning, and Action Execution - revealing critical flaws such as fake App identity, visual spoofing, indirect prompt injection, and unauthorized privilege escalation stemming from a reliance on unstructured visual data.
To address these challenges, we propose Aura, an Agent Universal Runtime Architecture for a clean-slate secure agent OS. Aura replaces brittle GUI scraping with a structured, agent-native interaction model. It adopts a Hub-and-Spoke topology where a privileged System Agent orchestrates intent, sandboxed App Agents execute domain-specific tasks, and the Agent Kernel mediates all communication. The Agent Kernel enforces four defense pillars: (i) cryptographic identity binding via a Global Agent Registry; (ii) semantic input sanitization through a multilayer Semantic Firewall; (iii) cognitive integrity via taint-aware memory and plan-trajectory alignment; and (iv) granular access control with non-deniable auditing. Evaluation on MobileSafetyBench shows that, compared to Doubao, Aura improves low-risk Task Success Rate from roughly 75% to 94.3%, reduces high-risk Attack Success Rate from roughly 40% to 4.4%, and achieves near-order-of-magnitude latency gains. These results demonstrate Aura as a viable, secure alternative to the "Screen-as-Interface" paradigm.

**arXiv ID:** 2602.10915
</details>

<details>
<summary><strong>Choose Your Agent: Tradeoffs in Adopting AI Advisors, Coaches, and Delegates in Multi-Party Negotiation</strong> - Kehang Zhu, Nithum Thain, Vivian Tsai, James Wexler, Crystal Qian - [[pdf]](https://arxiv.org/pdf/2602.12089)</summary>

**Abstract:** As AI usage becomes more prevalent in social contexts, understanding agent-user interaction is critical to designing systems that improve both individual and group outcomes. We present an online behavioral experiment (N = 243) in which participants play three multi-turn bargaining games in groups of three. Each game, presented in randomized order, grants access to a single LLM assistance modality: proactive recommendations from an Advisor, reactive feedback from a Coach, or autonomous execution by a Delegate; all modalities are powered by an underlying LLM that achieves superhuman performance in an all-agent environment. On each turn, participants privately decide whether to act manually or use the AI modality available in that game. Despite preferring the Advisor modality, participants achieve the highest mean individual gains with the Delegate, demonstrating a preference-performance misalignment. Moreover, delegation generates positive externalities; even non-adopting users in access-to-delegate treatment groups benefit by receiving higher-quality offers. Mechanism analysis reveals that the Delegate agent acts as a market maker, injecting rational, Pareto-improving proposals that restructure the trading environment. Our research reveals a gap between agent capabilities and realized group welfare. While autonomous agents can exhibit super-human strategic performance, their impact on realized welfare gains can be constrained by interfaces, user perceptions, and adoption barriers. Assistance modalities should be designed as mechanisms with endogenous participation; adoption-compatible interaction rules are a prerequisite to improving human welfare with automated assistance.

**arXiv ID:** 2602.12089
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>Scaling Web Agent Training through Automatic Data Generation and Fine-grained Evaluation</strong> - Lajanugen Logeswaran, Jaekyeom Kim, Sungryull Sohn, Creighton Glasscock, Honglak Lee - [[pdf]](https://arxiv.org/pdf/2602.12544)</summary>

**Abstract:** We present a scalable pipeline for automatically generating high-quality training data for web agents. In particular, a major challenge in identifying high-quality training instances is trajectory evaluation - quantifying how much progress was made towards task completion. We introduce a novel constraint-based evaluation framework that provides fine-grained assessment of progress towards task completion. This enables us to leverage partially successful trajectories, which significantly expands the amount of usable training data. We evaluate our method on a new benchmark we propose called BookingArena, which consists of complex booking tasks across 20 popular websites, and demonstrate that our distilled student model outperforms open-source approaches and matches or exceeds commercial systems, while being a significantly smaller model. Our work addresses the challenge of efficiently creating diverse, realistic web interaction datasets and provides a systematic evaluation methodology for complex structured web tasks.

**arXiv ID:** 2602.12544
</details>

<details>
<summary><strong>GeoAgent: Learning to Geolocate Everywhere with Reinforced Geographic Characteristics</strong> - Modi Jin, Yiming Zhang, Boyuan Sun, Dingwen Zhang, MingMing Cheng, Qibin Hou - [[pdf]](https://arxiv.org/pdf/2602.12617)</summary>

**Abstract:** This paper presents GeoAgent, a model capable of reasoning closely with humans and deriving fine-grained address conclusions. Previous RL-based methods have achieved breakthroughs in performance and interpretability but still remain concerns because of their reliance on AI-generated chain-of-thought (CoT) data and training strategies, which conflict with geographic characteristics. To address these issues, we first introduce GeoSeek, a new geolocation dataset comprising CoT data annotated by geographic experts and professional players. We further thoroughly explore the inherent characteristics of geographic tasks and propose a geo-similarity reward and a consistency reward assessed by a consistency agent to assist training. This encourages the model to converge towards correct answers from a geographic perspective while ensuring the integrity and consistency of its reasoning process. Experimental results show that GeoAgent outperforms existing methods and a series of general VLLMs across multiple grains, while generating reasoning that closely aligns with humans.

**arXiv ID:** 2602.12617
</details>

<details>
<summary><strong>AI Agents for Inventory Control: Human-LLM-OR Complementarity</strong> - Jackie Baek, Yaopeng Fu, Will Ma, Tianyi Peng - [[pdf]](https://arxiv.org/pdf/2602.12631)</summary>

**Abstract:** Inventory control is a fundamental operations problem in which ordering decisions are traditionally guided by theoretically grounded operations research (OR) algorithms. However, such algorithms often rely on rigid modeling assumptions and can perform poorly when demand distributions shift or relevant contextual information is unavailable. Recent advances in large language models (LLMs) have generated interest in AI agents that can reason flexibly and incorporate rich contextual signals, but it remains unclear how best to incorporate LLM-based methods into traditional decision-making pipelines.
We study how OR algorithms, LLMs, and humans can interact and complement each other in a multi-period inventory control setting. We construct InventoryBench, a benchmark of over 1,000 inventory instances spanning both synthetic and real-world demand data, designed to stress-test decision rules under demand shifts, seasonality, and uncertain lead times. Through this benchmark, we find that OR-augmented LLM methods outperform either method in isolation, suggesting that these methods are complementary rather than substitutes.
We further investigate the role of humans through a controlled classroom experiment that embeds LLM recommendations into a human-in-the-loop decision pipeline. Contrary to prior findings that human-AI collaboration can degrade performance, we show that, on average, human-AI teams achieve higher profits than either humans or AI agents operating alone. Beyond this population-level finding, we formalize an individual-level complementarity effect and derive a distribution-free lower bound on the fraction of individuals who benefit from AI collaboration; empirically, we find this fraction to be substantial.

**arXiv ID:** 2602.12631
</details>

<details>
<summary><strong>AgenticShop: Benchmarking Agentic Product Curation for Personalized Web Shopping</strong> - Sunghwan Kim, Ryang Heo, Yongsik Seo, Jinyoung Yeo, Dongha Lee - [[pdf]](https://arxiv.org/pdf/2602.12315)</summary>

**Abstract:** The proliferation of e-commerce has made web shopping platforms key gateways for customers navigating the vast digital marketplace. Yet this rapid expansion has led to a noisy and fragmented information environment, increasing cognitive burden as shoppers explore and purchase products online. With promising potential to alleviate this challenge, agentic systems have garnered growing attention for automating user-side tasks in web shopping. Despite significant advancements, existing benchmarks fail to comprehensively evaluate how well agentic systems can curate products in open-web settings. Specifically, they have limited coverage of shopping scenarios, focusing only on simplified single-platform lookups rather than exploratory search. Moreover, they overlook personalization in evaluation, leaving unclear whether agents can adapt to diverse user preferences in realistic shopping contexts. To address this gap, we present AgenticShop, the first benchmark for evaluating agentic systems on personalized product curation in open-web environment. Crucially, our approach features realistic shopping scenarios, diverse user profiles, and a verifiable, checklist-driven personalization evaluation framework. Through extensive experiments, we demonstrate that current agentic systems remain largely insufficient, emphasizing the need for user-side systems that effectively curate tailored products across the modern web.

**arXiv ID:** 2602.12315
</details>

<details>
<summary><strong>A Microservice-Based Platform for Sustainable and Intelligent SLO Fulfilment and Service Management</strong> - Juan Luis Herrera, Daniel Wang, Schahram Dustdar - [[pdf]](https://arxiv.org/pdf/2602.12875)</summary>

**Abstract:** The Microservices Architecture (MSA) design pattern has become a staple for modern applications, allowing functionalities to be divided across fine-grained microservices, fostering reusability, distribution, and interoperability. As MSA-based applications are deployed to the Computing Continuum (CC), meeting their Service Level Objectives (SLOs) becomes a challenge. Trading off performance and sustainability SLOs is especially challenging. This challenge can be addressed with intelligent decision systems, able to reconfigure the services during runtime to meet the SLOs. However, developing these agents while adhering to the MSA pattern is complex, especially because CC providers, who have key know-how and information to fulfill these SLOs, must comply with the privacy requirements of application developers. This work presents the Carbon-Aware SLO and Control plAtform (CASCA), an open-source MSA-based platform that allows CC providers to reconfigure services and fulfill their SLOs while maintaining the privacy of developers. CASCA is architected to be highly reusable, distributable, and easy to use, extend, and modify. CASCA has been evaluated in a real CC testbed for a media streaming service, where decision systems implemented in Bash, Rust, and Python successfully reconfigured the service, unaffected by upholding privacy.

**arXiv ID:** 2602.12875
</details>

<details>
<summary><strong>VoiceAgentBench: Are Voice Assistants ready for agentic tasks?</strong> - Dhruv Jain, Harshit Shukla, Gautam Rajeev, Ashish Kulkarni, Chandra Khatri, Shubham Agarwal - [[pdf]](https://arxiv.org/pdf/2510.07978)</summary>

**Abstract:** Large scale Speech Language Models have enabled voice assistants capable of understanding natural spoken queries and performing complex tasks. However, existing speech benchmarks largely focus on isolated capabilities such as transcription or question answering and do not systematically evaluate agentic behavior or adversarial robustness. To address this, we introduce VoiceAgentBench, a comprehensive benchmark for evaluating SpeechLMs in realistic spoken agentic settings, comprising 6,000+ synthetic spoken queries spanning single-tool invocations, multi-tool workflows, multi-turn dialogue, and safety evaluations across English and six Indic languages. To ensure speaker diversity, we further simulate speaker variability using a novel sampling strategy that selects audios for TTS voice conversion based on speaker embeddings to maximize acoustic diversity. Our evaluation measures tool selection accuracy, structural consistency, and the correctness of tool invocations, including adversarial robustness. Across agentic tasks, ASR-LLM pipelines outperform end-to-end SpeechLMs, achieving up to 60.6% average parameter-filling accuracy on English, while SpeechLMs exhibit lower performance and sharper degradation on Indic languages. All models struggle in sequential workflows and safety evaluations, highlighting persistent limitations in tool orchestration, multilingual generalization, and safety robustness. VoiceAgentBench is publicly available on Hugging Face at this https URL, and the codebase is released at this https URL.

**arXiv ID:** 2510.07978
</details>

<details>
<summary><strong>Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization</strong> - Taeyoon Kwon, Dongwook Choi, Hyojun Kim, Sunghwan Kim, Seungjun Moon, Beong-woo Kwak, Kuan-Hao Huang, Jinyoung Yeo - [[pdf]](https://arxiv.org/pdf/2505.16348)</summary>

**Abstract:** LLM-powered embodied agents have shown success on conventional object-rearrangement tasks, but providing personalized assistance that leverages user-specific knowledge from past interactions presents new challenges. We investigate these challenges through the lens of agents' memory utilization along two critical dimensions: object semantics (identifying objects based on personal meaning) and user patterns (recalling sequences from behavioral routines). To assess these capabilities, we construct MEMENTO, an end-to-end two-stage evaluation framework comprising single-memory and joint-memory tasks. Our experiments reveal that current agents can recall simple object semantics but struggle to apply sequential user patterns to planning. Through in-depth analysis, we identify two critical bottlenecks: information overload and coordination failures when handling multiple memories. Based on these findings, we explore memory architectural approaches to address these challenges. Given our observation that episodic memory provides both personalized knowledge and in-context learning benefits, we design a hierarchical knowledge graph-based user-profile memory module that separately manages personalized knowledge, achieving substantial improvements on both single and joint-memory tasks. Project website: this https URL

**arXiv ID:** 2505.16348
</details>

<details>
<summary><strong>Quantization-Aware Collaborative Inference for Large Embodied AI Models</strong> - Zhonghao Lyu, Ming Xiao, Mikael Skoglund, Merouane Debbah, H. Vincent Poor - [[pdf]](https://arxiv.org/pdf/2602.13052)</summary>

**Abstract:** Large artificial intelligence models (LAIMs) are increasingly regarded as a core intelligence engine for embodied AI applications. However, the massive parameter scale and computational demands of LAIMs pose significant challenges for resource-limited embodied agents. To address this issue, we investigate quantization-aware collaborative inference (co-inference) for embodied AI systems. First, we develop a tractable approximation for quantization-induced inference distortion. Based on this approximation, we derive lower and upper bounds on the quantization rate-inference distortion function, characterizing its dependence on LAIM statistics, including the quantization bit-width. Next, we formulate a joint quantization bit-width and computation frequency design problem under delay and energy constraints, aiming to minimize the distortion upper bound while ensuring tightness through the corresponding lower bound. Extensive evaluations validate the proposed distortion approximation, the derived rate-distortion bounds, and the effectiveness of the proposed joint design. Particularly, simulations and real-world testbed experiments demonstrate the effectiveness of the proposed joint design in balancing inference quality, latency, and energy consumption in edge embodied AI systems.

**arXiv ID:** 2602.13052
</details>

<details>
<summary><strong>An Autonomous, End-to-End, Convex-Based Framework for Close-Range Rendezvous Trajectory Design and Guidance with Hardware Testbed Validation</strong> - Minduli C. Wijayatunga, Julian Guinane, Nathan D. Wallace, Xiaofeng Wu - [[pdf]](https://arxiv.org/pdf/2602.12421)</summary>

**Abstract:** Autonomous satellite servicing missions must execute close-range rendezvous under stringent safety and operational constraints while remaining computationally tractable for onboard use and robust to uncertainty in sensing, actuation, and dynamics. This paper presents CORTEX (Convex Optimization for Rendezvous Trajectory Execution), an autonomous, perception-enabled, real-time trajectory design and guidance framework for close-range rendezvous. CORTEX integrates a deep-learning perception pipeline with convex-optimisation-based trajectory design and guidance, including reference regeneration and abort-to-safe-orbit logic to recover from large deviations caused by sensor faults and engine failures.
CORTEX is validated in high-fidelity software simulation and hardware-in-the-loop experiments. The software pipeline (Basilisk) models high-fidelity relative dynamics, realistic thruster execution, perception, and attitude control. Hardware testing uses (i) an optical navigation testbed to assess perception-to-estimation performance and (ii) a planar air-bearing testbed to evaluate the end-to-end guidance loop under representative actuation and subsystem effects. A Monte-Carlo campaign in simulation includes initial-state uncertainty, thrust-magnitude errors, and missed-thrust events; under the strongest case investigated, CORTEX achieves terminal docking errors of $36.85 \pm 44.46$ mm in relative position and $1.25 \pm 2.26$ mm/s in relative velocity. On the planar air-bearing testbed, 18 cases are executed (10 nominal; 8 off-nominal requiring recomputation and/or abort due to simulated engine failure and sensor malfunctions), yielding terminal errors of $8.09 \pm 5.29$ mm in position and $2.23 \pm 1.72$ mm/s in velocity.

**arXiv ID:** 2602.12421
</details>

<details>
<summary><strong>SKYSURF: A Self-learning Framework for Persistent Surveillance using Cooperative Aerial Gliders</strong> - Houssem Eddine Mohamadi, Nadjia Kara - [[pdf]](https://arxiv.org/pdf/2602.12838)</summary>

**Abstract:** The success of surveillance applications involving small unmanned aerial vehicles (UAVs) depends on how long the limited on-board power would persist. To cope with this challenge, alternative renewable sources of lift are sought. One promising solution is to extract energy from rising masses of buoyant air. This paper proposes a local-global behavioral management and decision-making approach for the autonomous deployment of soaring-capable UAVs. The cooperative UAVs are modeled as non-deterministic finite state-based rational agents. In addition to a mission planning module for assigning tasks and issuing dynamic navigation waypoints for a new path planning scheme, in which the concepts of visibility and prediction are applied to avoid the collisions. Moreover, a delayed learning and tuning strategy is employed optimize the gains of the path tracking controller. Rigorous comparative analyses carried out with three benchmarking baselines and 15 evolutionary algorithms highlight the adequacy of the proposed approach for maintaining the surveillance persistency (staying aloft for longer periods without landing) and maximizing the detection of targets (two times better than non-cooperative and semi-cooperative approaches) with less power consumption (almost 6% of battery consumed in six hours).

**arXiv ID:** 2602.12838
</details>

</details>

<details open>
<summary><h2>LLM Agents (10 papers)</h2></summary>

<details>
<summary><strong>Think Fast and Slow: Step-Level Cognitive Depth Adaptation for LLM Agents</strong> - Ruihan Yang, Fanghua Ye, Xiang We, Ruoqing Zhao, Kang Luo, Xinbo Xu, Bo Zhao, Ruotian Ma, Shanyi Wang, Zhaopeng Tu, Xiaolong Li, Deqing Yang, Linus - [[pdf]](https://arxiv.org/pdf/2602.12662)</summary>

**Abstract:** Large language models (LLMs) are increasingly deployed as autonomous agents for multi-turn decision-making tasks. However, current agents typically rely on fixed cognitive patterns: non-thinking models generate immediate responses, while thinking models engage in deep reasoning uniformly. This rigidity is inefficient for long-horizon tasks, where cognitive demands vary significantly from step to step, with some requiring strategic planning and others only routine execution. In this paper, we introduce CogRouter, a framework that trains agents to dynamically adapt cognitive depth at each step. Grounded in ACT-R theory, we design four hierarchical cognitive levels ranging from instinctive responses to strategic planning. Our two-stage training approach includes Cognition-aware Supervised Fine-tuning (CoSFT) to instill stable level-specific patterns, and Cognition-aware Policy Optimization (CoPO) for step-level credit assignment via confidence-aware advantage reweighting. The key insight is that appropriate cognitive depth should maximize the confidence of the resulting action. Experiments on ALFWorld and ScienceWorld demonstrate that CogRouter achieves state-of-the-art performance with superior efficiency. With Qwen2.5-7B, it reaches an 82.3% success rate, outperforming GPT-4o (+40.3%), OpenAI-o3 (+18.3%), and GRPO (+14.0%), while using 62% fewer tokens.

**arXiv ID:** 2602.12662
</details>

<details>
<summary><strong>SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks</strong> - Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, Shuyi Wang, Qunhong Zeng, Di Wang, Xuandong Zhao, Yuanli Wang, Roey Ben Chaim, Zonglin Di, Yipeng Gao, Junwei He, Yizhuo He, Liqiang Jing, Luyang Kong, Xin Lan, Jiachen Li, Songlin Li, Yijiang Li, Yueqian Lin, Xinyi Liu, Xuanqing Liu, Haoran Lyu, Ze Ma, Bowei Wang, Runhui Wang, Tianyu Wang, Wengao Ye, Yue Zhang, Hanwen Xing, Yiqi Xue, Steven Dillmann, Han-chung Lee - [[pdf]](https://arxiv.org/pdf/2602.12670)</summary>

**Abstract:** Agent Skills are structured packages of procedural knowledge that augment LLM agents at inference time. Despite rapid adoption, there is no standard way to measure whether they actually help. We present SkillsBench, a benchmark of 86 tasks across 11 domains paired with curated Skills and deterministic verifiers. Each task is evaluated under three conditions: no Skills, curated Skills, and self-generated Skills. We test 7 agent-model configurations over 7,308 trajectories. Curated Skills raise average pass rate by 16.2 percentage points(pp), but effects vary widely by domain (+4.5pp for Software Engineering to +51.9pp for Healthcare) and 16 of 84 tasks show negative deltas. Self-generated Skills provide no benefit on average, showing that models cannot reliably author the procedural knowledge they benefit from consuming. Focused Skills with 2--3 modules outperform comprehensive documentation, and smaller models with Skills can match larger models without them.

**arXiv ID:** 2602.12670
</details>

<details>
<summary><strong>From Biased Chatbots to Biased Agents: Examining Role Assignment Effects on LLM Agent Robustness</strong> - Linbo Cao, Lihao Sun, Yang Yue - [[pdf]](https://arxiv.org/pdf/2602.12285)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly deployed as autonomous agents capable of actions with real-world impacts beyond text generation. While persona-induced biases in text generation are well documented, their effects on agent task performance remain largely unexplored, even though such effects pose more direct operational risks. In this work, we present the first systematic case study showing that demographic-based persona assignments can alter LLM agents' behavior and degrade performance across diverse domains. Evaluating widely deployed models on agentic benchmarks spanning strategic reasoning, planning, and technical operations, we uncover substantial performance variations - up to 26.2% degradation, driven by task-irrelevant persona cues. These shifts appear across task types and model architectures, indicating that persona conditioning and simple prompt injections can distort an agent's decision-making reliability. Our findings reveal an overlooked vulnerability in current LLM agentic systems: persona assignments can introduce implicit biases and increase behavioral volatility, raising concerns for the safe and robust deployment of LLM agents.

**arXiv ID:** 2602.12285
</details>

<details>
<summary><strong>Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward</strong> - Renjun Xu, Yang Yan - [[pdf]](https://arxiv.org/pdf/2602.12430)</summary>

**Abstract:** The transition from monolithic language models to modular, skill-equipped agents marks a defining shift in how large language models (LLMs) are deployed in practice. Rather than encoding all procedural knowledge within model weights, agent skills -- composable packages of instructions, code, and resources that agents load on demand -- enable dynamic capability extension without retraining. It is formalized in a paradigm of progressive disclosure, portable skill definitions, and integration with the Model Context Protocol (MCP). This survey provides a comprehensive treatment of the agent skills landscape, as it has rapidly evolved during the last few months. We organize the field along four axes: (i) architectural foundations, examining the {this http URL} specification, progressive context loading, and the complementary roles of skills and MCP; (ii) skill acquisition, covering reinforcement learning with skill libraries, autonomous skill discovery (SEAgent), and compositional skill synthesis; (iii) deployment at scale, including the computer-use agent (CUA) stack, GUI grounding advances, and benchmark progress on OSWorld and SWE-bench; and (iv) security, where recent empirical analyses reveal that 26.1% of community-contributed skills contain vulnerabilities, motivating our proposed Skill Trust and Lifecycle Governance Framework -- a four-tier, gate-based permission model that maps skill provenance to graduated deployment capabilities. We identify seven open challenges -- from cross-platform skill portability to capability-based permission models -- and propose a research agenda for realizing trustworthy, self-improving skill ecosystems. Unlike prior surveys that broadly cover LLM agents or tool use, this work focuses specifically on the emerging skill abstraction layer and its implications for the next generation of agentic systems. Project repo: this https URL

**arXiv ID:** 2602.12430
</details>

<details>
<summary><strong>Favia: Forensic Agent for Vulnerability-fix Identification and Analysis</strong> - André Storhaug, Jiamou Sun, Jingyue Li - [[pdf]](https://arxiv.org/pdf/2602.12500)</summary>

**Abstract:** Identifying vulnerability-fixing commits corresponding to disclosed CVEs is essential for secure software maintenance but remains challenging at scale, as large repositories contain millions of commits of which only a small fraction address security issues. Existing automated approaches, including traditional machine learning techniques and recent large language model (LLM)-based methods, often suffer from poor precision-recall trade-offs. Frequently evaluated on randomly sampled commits, we uncover that they are substantially underestimating real-world difficulty, where candidate commits are already security-relevant and highly similar. We propose Favia, a forensic, agent-based framework for vulnerability-fix identification that combines scalable candidate ranking with deep and iterative semantic reasoning. Favia first employs an efficient ranking stage to narrow the search space of commits. Each commit is then rigorously evaluated using a ReAct-based LLM agent. By providing the agent with a pre-commit repository as environment, along with specialized tools, the agent tries to localize vulnerable components, navigates the codebase, and establishes causal alignment between code changes and vulnerability root causes. This evidence-driven process enables robust identification of indirect, multi-file, and non-trivial fixes that elude single-pass or similarity-based methods. We evaluate Favia on CVEVC, a large-scale dataset we made that comprises over 8 million commits from 3,708 real-world repositories, and show that it consistently outperforms state-of-the-art traditional and LLM-based baselines under realistic candidate selection, achieving the strongest precision-recall trade-offs and highest F1-scores.

**arXiv ID:** 2602.12500
</details>

<details>
<summary><strong>In-Context Autonomous Network Incident Response: An End-to-End Large Language Model Agent Approach</strong> - Yiran Gao, Kim Hammar, Tao Li - [[pdf]](https://arxiv.org/pdf/2602.13156)</summary>

**Abstract:** Rapidly evolving cyberattacks demand incident response systems that can autonomously learn and adapt to changing threats. Prior work has extensively explored the reinforcement learning approach, which involves learning response strategies through extensive simulation of the incident. While this approach can be effective, it requires handcrafted modeling of the simulator and suppresses useful semantics from raw system logs and alerts. To address these limitations, we propose to leverage large language models' (LLM) pre-trained security knowledge and in-context learning to create an end-to-end agentic solution for incident response planning. Specifically, our agent integrates four functionalities, perception, reasoning, planning, and action, into one lightweight LLM (14b model). Through fine-tuning and chain-of-thought reasoning, our LLM agent is capable of processing system logs and inferring the underlying network state (perception), updating its conjecture of attack models (reasoning), simulating consequences under different response strategies (planning), and generating an effective response (action). By comparing LLM-simulated outcomes with actual observations, the LLM agent repeatedly refines its attack conjecture and corresponding response, thereby demonstrating in-context adaptation. Our agentic approach is free of modeling and can run on commodity hardware. When evaluated on incident logs reported in the literature, our agent achieves recovery up to 23% faster than those of frontier LLMs.

**arXiv ID:** 2602.13156
</details>

<details>
<summary><strong>ATLAS : Adaptive Self-Evolutionary Research Agent with Task-Distributed Multi-LLM Supporters</strong> - Ujin Jeon, Jiyong Kwon, Madison Ann Sullivan, Caleb Eunho Lee, Guang Lin - [[pdf]](https://arxiv.org/pdf/2602.02709)</summary>

**Abstract:** Recent multi-LLM agent systems perform well in prompt optimization and automated problem-solving, but many either keep the solver frozen after fine-tuning or rely on a static preference-optimization loop, which becomes intractable for long-horizon tasks. We propose ATLAS (Adaptive Task-distributed Learning for Agentic Self-evolution), a task-distributed framework that iteratively develops a lightweight research agent while delegating complementary roles to specialized supporter agents for exploration, hyperparameter tuning, and reference policy management. Our core algorithm, Evolving Direct Preference Optimization (EvoDPO), adaptively updates the phase-indexed reference policy. We provide a theoretical regret analysis for a preference-based contextual bandit under concept drift. In addition, experiments were conducted on non-stationary linear contextual bandits and scientific machine learning (SciML) loss reweighting for the 1D Burgers' equation. Both results show that ATLAS improves stability and performance over a static single-agent baseline.

**arXiv ID:** 2602.02709
</details>

<details>
<summary><strong>ToolACE-MT: Non-Autoregressive Generation for Agentic Multi-Turn Interaction</strong> - Xingshan Zeng, Weiwen Liu, Lingzhi Wang, Liangyou Li, Fei Mi, Yasheng Wang, Lifeng Shang, Xin Jiang, Qun Liu - [[pdf]](https://arxiv.org/pdf/2508.12685)</summary>

**Abstract:** Agentic task-solving with Large Language Models (LLMs) requires multi-turn, multi-step interactions, often involving complex function calls and dynamic user-agent exchanges. Existing simulation-based data generation methods for such scenarios rely heavily on costly autoregressive interactions between multiple LLM agents, thereby compromising the practical efficiency of agentic data generation. In this paper, we propose ToolACE-MT, a novel Non-Autoregressive Iterative Generation framework for constructing high-quality multi-turn agentic dialogues. ToolACE-MT generates full conversational trajectories through three stages: coarse-grained initialization, iterative refinement, and offline verification. The initialization phase builds a structurally complete yet semantically coarse dialogue skeleton; the iterative refinement phase introduces realistic complexities and continued refinement via mask-and-fill operations; and the offline verification phase ensures correctness and coherence via rule- and model-based checks. Experiments demonstrate that ToolACE-MT enables efficient, effective and generalizable agentic data generation, offering a new paradigm for high-quality data construction in tool-augmented LLM scenarios.

**arXiv ID:** 2508.12685
</details>

<details>
<summary><strong>SciAgentGym: Benchmarking Multi-Step Scientific Tool-use in LLM Agents</strong> - Yujiong Shen, Yajie Yang, Zhiheng Xi, Binze Hu, Huayu Sha, Jiazheng Zhang, Qiyuan Peng, Junlin Shang, Jixuan Huang, Yutao Fan, Jingqi Tong, Shihan Dou, Ming Zhang, Lei Bai, Zhenfei Yin, Tao Gui, Xingjun Ma, Qi Zhang, Xuanjing Huang, Yu-Gang Jiang - [[pdf]](https://arxiv.org/pdf/2602.12984)</summary>

**Abstract:** Scientific reasoning inherently demands integrating sophisticated toolkits to navigate domain-specific knowledge. Yet, current benchmarks largely overlook agents' ability to orchestrate tools for such rigorous workflows. To bridge this gap, we introduce SciAgentGym, a scalable interactive environment featuring 1,780 domain-specific tools across four natural science disciplines, supported by a robust execution infrastructure. Complementing this, we present SciAgentBench, a tiered evaluation suite designed to stress-test agentic capabilities from elementary actions to long-horizon workflows. Our evaluation identifies a critical bottleneck: state-of-the-art models struggle with complex scientific tool-use. Even for a leading model like GPT-5, success rates drop sharply from 60.6% to 30.9% as interaction horizons extend, primarily due to failures in multi-step workflow execution. To address this, we propose SciForge, a data synthesis method that models the tool action space as a dependency graph to generate logic-aware training trajectories. By fine-tuning on these trajectories, our SciAgent-8B outperforms the significantly larger Qwen3-VL-235B-Instruct while exhibiting positive cross-domain transfer of scientific tool-use capabilities. These results underscore the promising potential of next-generation autonomous scientific agents.

**arXiv ID:** 2602.12984
</details>

<details>
<summary><strong>Memory Injection Attacks on LLM Agents via Query-Only Interaction</strong> - Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, Zhen Xiang - [[pdf]](https://arxiv.org/pdf/2503.03704)</summary>

**Abstract:** Agents powered by large language models (LLMs) have demonstrated strong capabilities in a wide range of complex, real-world applications. However, LLM agents with a compromised memory bank may easily produce harmful outputs when the past records retrieved for demonstration are malicious. In this paper, we propose a novel Memory INJection Attack, MINJA, without assuming that the attacker can directly modify the memory bank of the agent. The attacker injects malicious records into the memory bank by only interacting with the agent via queries and output observations. These malicious records are designed to elicit a sequence of malicious reasoning steps corresponding to a different target query during the agent's execution of the victim user's query. Specifically, we introduce a sequence of bridging steps to link victim queries to the malicious reasoning steps. During the memory injection, we propose an indication prompt that guides the agent to autonomously generate similar bridging steps, with a progressive shortening strategy that gradually removes the indication prompt, such that the malicious record will be easily retrieved when processing later victim queries. Our extensive experiments across diverse agents demonstrate the effectiveness of MINJA in compromising agent memory. With minimal requirements for execution, MINJA enables any user to influence agent memory, highlighting the risk.

**arXiv ID:** 2503.03704
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (11 papers)</h2></summary>

<details>
<summary><strong>Perceptual Self-Reflection in Agentic Physics Simulation Code Generation</strong> - Prashant Shende, Bradley Camburn - [[pdf]](https://arxiv.org/pdf/2602.12311)</summary>

**Abstract:** We present a multi-agent framework for generating physics simulation code from natural language descriptions, featuring a novel perceptual self-reflection mechanism for validation. The system employs four specialized agents: a natural language interpreter that converts user requests into physics-based descriptions; a technical requirements generator that produces scaled simulation parameters; a physics code generator with automated self-correction; and a physics validator that implements perceptual self-reflection. The key innovation is perceptual validation, which analyzes rendered animation frames using a vision-capable language model rather than inspecting code structure directly. This approach addresses the ``oracle gap'' where syntactically correct code produces physically incorrect behavior--a limitation that conventional testing cannot detect. We evaluate the system across seven domains including classical mechanics, fluid dynamics, thermodynamics, electromagnetics, wave physics, reaction-diffusion systems, and non-physics data visualization. The perceptual self-reflection architecture demonstrates substantial improvement over single-shot generation baselines, with the majority of tested scenarios achieving target physics accuracy thresholds. The system exhibits robust pipeline stability with consistent code self-correction capability, operating at approximately \$0.20 per animation. These results validate our hypothesis that feeding visual simulation outputs back to a vision-language model for iterative refinement significantly outperforms single-shot code generation for physics simulation tasks and highlights the potential of agentic AI to support engineering workflows and physics data generation pipelines.

**arXiv ID:** 2602.12311
</details>

<details>
<summary><strong>A Survey on Hypergame Theory: Modeling Misaligned Perceptions and Nested Beliefs for Multi-agent Systems</strong> - Vince Trencsenyi, Agnieszka Mensfelt, Kostas Stathis - [[pdf]](https://arxiv.org/pdf/2507.19593)</summary>

**Abstract:** Classical game-theoretic models typically assume rational agents, complete information, and common knowledge of payoffs - assumptions that are often violated in real-world MAS characterized by uncertainty, misaligned perceptions, and nested beliefs. To overcome these limitations, researchers have proposed extensions that incorporate models of cognitive constraints, subjective beliefs, and heterogeneous reasoning. Among these, hypergame theory extends the classical paradigm by explicitly modeling agents' subjective perceptions of the strategic scenario, known as perceptual games, in which agents may hold divergent beliefs about the structure, payoffs, or available actions. We present a systematic review of agent-compatible applications of hypergame theory, examining how its descriptive capabilities have been adapted to dynamic and interactive MAS contexts. We analyze 44 selected studies from cybersecurity, robotics, social simulation, communications, and general game-theoretic modeling. Building on a formal introduction to hypergame theory and its two major extensions - hierarchical hypergames and HNF - we develop agent-compatibility criteria and an agent-based classification framework to assess integration patterns and practical applicability. Our analysis reveals prevailing tendencies, including the prevalence of hierarchical and graph-based models in deceptive reasoning and the simplification of extensive theoretical frameworks in practical applications. We identify structural gaps, including the limited adoption of HNF-based models, the lack of formal hypergame languages, and unexplored opportunities for modeling human-agent and agent-agent misalignment. By synthesizing trends, challenges, and open research directions, this review provides a new roadmap for applying hypergame theory to enhance the realism and effectiveness of strategic modeling in dynamic multi-agent environments.

**arXiv ID:** 2507.19593
</details>

<details>
<summary><strong>Difficulty-Aware Agentic Orchestration for Query-Specific Multi-Agent Workflows</strong> - Jinwei Su, Qizhen Lan, Yinghui Xia, Lifan Sun, Weiyou Tian, Tianyu Shi, Xinyuan Song, Lewei He, Yang Jingsong - [[pdf]](https://arxiv.org/pdf/2509.11079)</summary>

**Abstract:** Large Language Model (LLM)-based agentic systems have shown strong capabilities across various tasks. However, existing multi-agent frameworks often rely on static or task-level workflows, which either over-process simple queries or underperform on complex ones, while also neglecting the efficiency-performance trade-offs across heterogeneous LLMs. To address these limitations, we propose Difficulty-Aware Agentic Orchestration (DAAO), which can dynamically generate query-specific multi-agent workflows guided by predicted query difficulty. DAAO comprises three interdependent modules: a variational autoencoder (VAE) for difficulty estimation, a modular operator allocator, and a cost- and performance-aware LLM router. A self-adjusting policy updates difficulty estimates based on workflow success, enabling simpler workflows for easy queries and more complex strategies for harder ones. Experiments on six benchmarks demonstrate that DAAO surpasses prior multi-agent systems in both accuracy and inference efficiency, validating its effectiveness for adaptive, difficulty-aware reasoning.

**arXiv ID:** 2509.11079
</details>

<details>
<summary><strong>WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning</strong> - Zelai Xu, Zhexuan Xu, Ruize Zhang, Chunyang Zhu, Shi Yu, Weilin Liu, Quanlu Zhang, Wenbo Ding, Chao Yu, Yu Wang - [[pdf]](https://arxiv.org/pdf/2602.04634)</summary>

**Abstract:** Recent advancements in Large Language Models (LLMs) have largely focused on depth scaling, where a single agent solves long-horizon problems with multi-turn reasoning and tool use. However, as tasks grow broader, the key bottleneck shifts from individual competence to organizational capability. In this work, we explore a complementary dimension of width scaling with multi-agent systems to address broad information seeking. Existing multi-agent systems often rely on hand-crafted workflows and turn-taking interactions that fail to parallelize work effectively. To bridge this gap, we propose WideSeek-R1, a lead-agent-subagent framework trained via multi-agent reinforcement learning (MARL) to synergize scalable orchestration and parallel execution. By utilizing a shared LLM with isolated contexts and specialized tools, WideSeek-R1 jointly optimizes the lead agent and parallel subagents on a curated dataset of 20k broad information-seeking tasks. Extensive experiments show that WideSeek-R1-4B achieves an item F1 score of 40.0% on the WideSearch benchmark, which is comparable to the performance of single-agent DeepSeek-R1-671B. Furthermore, WideSeek-R1-4B exhibits consistent performance gains as the number of parallel subagents increases, highlighting the effectiveness of width scaling.

**arXiv ID:** 2602.04634
</details>

<details>
<summary><strong>MASPRM: Multi-Agent System Process Reward Model</strong> - Milad Yazdani, Mahdi Mostajabdaveh, Zirui Zhou, Ying Xiong - [[pdf]](https://arxiv.org/pdf/2510.24803)</summary>

**Abstract:** Practical deployment of multi-agent systems (MAS) demands strong performance at test time, motivating methods that guide search during inference and selectively spend compute to improve quality. We present the Multi-Agent System Process Reward Model (MASPRM). It assigns values to partial inter-agent transcripts for each action and each agent, and acts as a controller during inference. MASPRM is trained from multi-agent Monte Carlo Tree Search (MCTS) rollouts labeled only with terminal outcome rewards, without requiring human step-level annotations, by propagating returns to local targets. During inference, MASPRM guides step-level beam search (SBS) and MCTS, focusing computation on promising branches and pruning unpromising ones. We train and test MASPRM across different tasks and domains, using GSM8K, MATH, MMLU, and LogiQA as benchmarks. Averaged across these benchmarks, MASPRM improves Hit@1 over policy likelihood by up to $+13.4$ points and improves ranking quality, reducing Hit@1$->$Hit@5 gaps by up to $10.3$ points. MASPRM complements inference-time search by scoring intermediate routed transcripts to guide rollouts in MAS with fixed schedules. Code: this https URL

**arXiv ID:** 2510.24803
</details>

<details>
<summary><strong>Multi-Agent Model-Based Reinforcement Learning with Joint State-Action Learned Embeddings</strong> - Zhizun Wang, David Meger - [[pdf]](https://arxiv.org/pdf/2602.12520)</summary>

**Abstract:** Learning to coordinate many agents in partially observable and highly dynamic environments requires both informative representations and data-efficient training. To address this challenge, we present a novel model-based multi-agent reinforcement learning framework that unifies joint state-action representation learning with imaginative roll-outs. We design a world model trained with variational auto-encoders and augment the model using the state-action learned embedding (SALE). SALE is injected into both the imagination module that forecasts plausible future roll-outs and the joint agent network whose individual action values are combined through a mixing network to estimate the joint action-value function. By coupling imagined trajectories with SALE-based action values, the agents acquire a richer understanding of how their choices influence collective outcomes, leading to improved long-term planning and optimization under limited real-environment interactions. Empirical studies on well-established multi-agent benchmarks, including StarCraft II Micro-Management, Multi-Agent MuJoCo, and Level-Based Foraging challenges, demonstrate consistent gains of our method over baseline algorithms and highlight the effectiveness of joint state-action learned embeddings within a multi-agent model-based paradigm.

**arXiv ID:** 2602.12520
</details>

<details>
<summary><strong>Bayesian Ego-graph Inference for Networked Multi-Agent Reinforcement Learning</strong> - Wei Duan, Jie Lu, Junyu Xuan - [[pdf]](https://arxiv.org/pdf/2509.16606)</summary>

**Abstract:** In networked multi-agent reinforcement learning (Networked-MARL), decentralized agents must act under local observability and constrained communication over fixed physical graphs. Existing methods often assume static neighborhoods, limiting adaptability to dynamic or heterogeneous environments. While centralized frameworks can learn dynamic graphs, their reliance on global state access and centralized infrastructure is impractical in real-world decentralized systems. We propose a stochastic graph-based policy for Networked-MARL, where each agent conditions its decision on a sampled subgraph over its local physical neighborhood. Building on this formulation, we introduce BayesG, a decentralized actor-framework that learns sparse, context-aware interaction structures via Bayesian variational inference. Each agent operates over an ego-graph and samples a latent communication mask to guide message passing and policy computation. The variational distribution is trained end-to-end alongside the policy using an evidence lower bound (ELBO) objective, enabling agents to jointly learn both interaction topology and decision-making strategies. BayesG outperforms strong MARL baselines on large-scale traffic control tasks with up to 167 agents, demonstrating superior scalability, efficiency, and performance.

**arXiv ID:** 2509.16606
</details>

<details>
<summary><strong>TraceBack: Multi-Agent Decomposition for Fine-Grained Table Attribution</strong> - Tejas Anvekar, Junha Park, Rajat Jha, Devanshu Gupta, Poojah Ganesan, Puneeth Mathur, Vivek Gupta - [[pdf]](https://arxiv.org/pdf/2602.13059)</summary>

**Abstract:** Question answering (QA) over structured tables requires not only accurate answers but also transparency about which cells support them. Existing table QA systems rarely provide fine-grained attribution, so even correct answers often lack verifiable grounding, limiting trust in high-stakes settings. We address this with TraceBack, a modular multi-agent framework for scalable, cell-level attribution in single-table QA. TraceBack prunes tables to relevant rows and columns, decomposes questions into semantically coherent sub-questions, and aligns each answer span with its supporting cells, capturing both explicit and implicit evidence used in intermediate reasoning steps. To enable systematic evaluation, we release CITEBench, a benchmark with phrase-to-cell annotations drawn from ToTTo, FetaQA, and AITQA. We further propose FairScore, a reference-less metric that compares atomic facts derived from predicted cells and answers to estimate attribution precision and recall without human cell labels. Experiments show that TraceBack substantially outperforms strong baselines across datasets and granularities, while FairScore closely tracks human judgments and preserves relative method rankings, supporting interpretable and scalable evaluation of table-based QA.

**arXiv ID:** 2602.13059
</details>

<details>
<summary><strong>B3C: A Minimalist Approach to Offline Multi-Agent Reinforcement Learning</strong> - Woojun Kim, Katia Sycara - [[pdf]](https://arxiv.org/pdf/2501.18138)</summary>

**Abstract:** Overestimation arising from selecting unseen actions during policy evaluation is a major challenge in offline reinforcement learning (RL). A minimalist approach in the single-agent setting -- adding behavior cloning (BC) regularization to existing online RL algorithms -- has been shown to be effective; however, this approach is understudied in multi-agent settings. In particular, overestimation becomes worse in multi-agent settings due to the presence of multiple actions, resulting in the BC regularization-based approach easily suffering from either over-regularization or critic divergence. To address this, we propose a simple yet effective method, Behavior Cloning regularization with Critic Clipping (B3C), which clips the target critic value in policy evaluation based on the maximum return in the dataset and pushes the limit of the weight on the RL objective over BC regularization, thereby improving performance. Additionally, we leverage existing value factorization techniques, particularly non-linear factorization, which is understudied in offline settings. Integrated with non-linear value factorization, B3C outperforms state-of-the-art algorithms on various offline multi-agent benchmarks.

**arXiv ID:** 2501.18138
</details>

<details>
<summary><strong>Multi-Agent Stage-wise Conservative Linear Bandits</strong> - Amirhossein Afsharrad, Ahmadreza Moradipari, Sanjay Lall - [[pdf]](https://arxiv.org/pdf/2510.00602)</summary>

**Abstract:** In many real-world applications such as recommendation systems, multiple learning agents must balance exploration and exploitation while maintaining safety guarantees to avoid catastrophic failures. We study the stochastic linear bandit problem in a multi-agent networked setting where agents must satisfy stage-wise conservative constraints. A network of $N$ agents collaboratively maximizes cumulative reward while ensuring that the expected reward at every round is no less than $(1-\alpha)$ times that of a baseline policy. Each agent observes local rewards with unknown parameters, but the network optimizes for the global parameter (average of local parameters). Agents communicate only with immediate neighbors, and each communication round incurs additional regret. We propose MA-SCLUCB (Multi-Agent Stage-wise Conservative Linear UCB), an episodic algorithm alternating between action selection and consensus-building phases. We prove that MA-SCLUCB achieves regret $\tilde{O}\left(\frac{d}{\sqrt{N}}\sqrt{T}\cdot\frac{\log(NT)}{\sqrt{\log(1/|\lambda_2|)}}\right)$ with high probability, where $d$ is the dimension, $T$ is the horizon, and $|\lambda_2|$ is the network's second largest eigenvalue magnitude. Our analysis shows: (i) collaboration yields $\frac{1}{\sqrt{N}}$ improvement despite local communication, (ii) communication overhead grows only logarithmically for well-connected networks, and (iii) stage-wise safety adds only lower-order regret. Thus, distributed learning with safety guarantees achieves near-optimal performance in reasonably connected networks.

**arXiv ID:** 2510.00602
</details>

<details>
<summary><strong>Automating UI Optimization through Multi-Agentic Reasoning</strong> - Zhipeng Li, Christoph Gebhardt, Yi-Chi Liao, Christian Holz - [[pdf]](https://arxiv.org/pdf/2602.13126)</summary>

**Abstract:** We present AutoOptimization, a novel multi-objective optimization framework for adapting user interfaces. From a user's verbal preferences for changing a UI, our framework guides a prioritization-based Pareto frontier search over candidate layouts. It selects suitable objective functions for UI placement while simultaneously parameterizing them according to the user's instructions to define the optimization problem. A solver then generates a series of optimal UI layouts, which our framework validates against the user's instructions to adapt the UI with the final solution. Our approach thus overcomes the previous need for manual inspection of layouts and the use of population averages for objective parameters. We integrate multiple agents sequentially within our framework, enabling the system to leverage their reasoning capabilities to interpret user preferences, configure the optimization problem, and validate optimization outcomes.

**arXiv ID:** 2602.13126
</details>

</details>

<details open>
<summary><h2>Other Agent Research (6 papers)</h2></summary>

<details>
<summary><strong>WebClipper: Efficient Evolution of Web Agents with Graph-based Trajectory Pruning</strong> - Junjie Wang, Zequn Xie, Dan Yang, Jie Feng, Yue Shen, Duolin Sun, Meixiu Long, Yihan Jiao, Zhehao Tan, Jian Wang, Peng Wei, Jinjie Gu - [[pdf]](https://arxiv.org/pdf/2602.12852)</summary>

**Abstract:** Deep Research systems based on web agents have shown strong potential in solving complex information-seeking tasks, yet their search efficiency remains underexplored. We observe that many state-of-the-art open-source web agents rely on long tool-call trajectories with cyclic reasoning loops and exploration of unproductive branches. To address this, we propose WebClipper, a framework that compresses web agent trajectories via graph-based pruning. Concretely, we model the agent's search process as a state graph and cast trajectory optimization as a minimum-necessary Directed Acyclic Graph (DAG) mining problem, yielding pruned trajectories that preserve essential reasoning while eliminating redundant steps. Continued training on these refined trajectories enables the agent to evolve toward more efficient search patterns and reduces tool-call rounds by about 20% while improving accuracy. Furthermore, we introduce a new metric called F-AE Score to measure the model's overall performance in balancing accuracy and efficiency. Experiments demonstrate that WebClipper compresses tool-call rounds under excellent performance, providing practical insight into balancing effectiveness and efficiency in web agent design.

**arXiv ID:** 2602.12852
</details>

<details>
<summary><strong>Never say never: Exploring the effects of available knowledge on agent persuasiveness in controlled physiotherapy motivation dialogues</strong> - Stephan Vonschallen, Rahel Häusler, Theresa Schmiedel, Friederike Eyssel - [[pdf]](https://arxiv.org/pdf/2602.12924)</summary>

**Abstract:** Generative Social Agents (GSAs) are increasingly impacting human users through persuasive means. On the one hand, they might motivate users to pursue personal goals, such as healthier lifestyles. On the other hand, they are associated with potential risks like manipulation and deception, which are induced by limited control over probabilistic agent outputs. However, as GSAs manifest communicative patterns based on available knowledge, their behavior may be regulated through their access to such knowledge. Following this approach, we explored persuasive ChatGPT-generated messages in the context of human-robot physiotherapy motivation. We did so by comparing ChatGPT-generated responses to predefined inputs from a hypothetical physiotherapy patient. In Study 1, we qualitatively analyzed 13 ChatGPT-generated dialogue scripts with varying knowledge configurations regarding persuasive message characteristics. In Study 2, third-party observers (N = 27) rated a selection of these dialogues in terms of the agent's expressiveness, assertiveness, and persuasiveness. Our findings indicate that LLM-based GSAs can adapt assertive and expressive personality traits -- significantly enhancing perceived persuasiveness. Moreover, persuasiveness significantly benefited from the availability of information about the patients' age and past profession, mediated by perceived assertiveness and expressiveness. Contextual knowledge about physiotherapy benefits did not significantly impact persuasiveness, possibly because the LLM had inherent knowledge about such benefits even without explicit prompting. Overall, the study highlights the importance of empirically studying behavioral patterns of GSAs, specifically in terms of what information generative AI systems require for consistent and responsible communication.

**arXiv ID:** 2602.12924
</details>

<details>
<summary><strong>UniManip: General-Purpose Zero-Shot Robotic Manipulation with Agentic Operational Graph</strong> - Haichao Liu, Yuanjiang Xue, Yuheng Zhou, Haoyuan Deng, Yinan Liang, Lihua Xie, Ziwei Wang - [[pdf]](https://arxiv.org/pdf/2602.13086)</summary>

**Abstract:** Achieving general-purpose robotic manipulation requires robots to seamlessly bridge high-level semantic intent with low-level physical interaction in unstructured environments. However, existing approaches falter in zero-shot generalization: end-to-end Vision-Language-Action (VLA) models often lack the precision required for long-horizon tasks, while traditional hierarchical planners suffer from semantic rigidity when facing open-world variations. To address this, we present UniManip, a framework grounded in a Bi-level Agentic Operational Graph (AOG) that unifies semantic reasoning and physical grounding. By coupling a high-level Agentic Layer for task orchestration with a low-level Scene Layer for dynamic state representation, the system continuously aligns abstract planning with geometric constraints, enabling robust zero-shot execution. Unlike static pipelines, UniManip operates as a dynamic agentic loop: it actively instantiates object-centric scene graphs from unstructured perception, parameterizes these representations into collision-free trajectories via a safety-aware local planner, and exploits structured memory to autonomously diagnose and recover from execution failures. Extensive experiments validate the system's robust zero-shot capability on unseen objects and tasks, demonstrating a 22.5% and 25.0% higher success rate compared to state-of-the-art VLA and hierarchical baselines, respectively. Notably, the system enables direct zero-shot transfer from fixed-base setups to mobile manipulation without fine-tuning or reconfiguration. Our open-source project page can be found at this https URL.

**arXiv ID:** 2602.13086
</details>

<details>
<summary><strong>Temporally-Sampled Efficiently Adaptive State Lattices for Autonomous Ground Robot Navigation in Partially Observed Environments</strong> - Ashwin Satish Menon, Eric R. Damm, Eli S. Lancaster, Felix A. Sanchez, Jason M. Gregory, Thomas M. Howard - [[pdf]](https://arxiv.org/pdf/2602.13159)</summary>

**Abstract:** Due to sensor limitations, environments that off-road mobile robots operate in are often only partially observable. As the robots move throughout the environment and towards their goal, the optimal route is continuously revised as the sensors perceive new information. In traditional autonomous navigation architectures, a regional motion planner will consume the environment map and output a trajectory for the local motion planner to use as a reference. Due to the continuous revision of the regional plan guidance as a result of changing map information, the reference trajectories which are passed down to the local planner can differ significantly across sequential planning cycles. This rapidly changing guidance can result in unsafe navigation behavior, often requiring manual safety interventions during autonomous traversals in off-road environments. To remedy this problem, we propose Temporally-Sampled Efficiently Adaptive State Lattices (TSEASL), which is a regional planner arbitration architecture that considers updated and optimized versions of previously generated trajectories against the currently generated trajectory. When tested on a Clearpath Robotics Warthog Unmanned Ground Vehicle as well as real map data collected from the Warthog, results indicate that when running TSEASL, the robot did not require manual interventions in the same locations where the robot was running the baseline planner. Additionally, higher levels of planner stability were recorded with TSEASL over the baseline. The paper concludes with a discussion of further improvements to TSEASL in order to make it more generalizable to various off-road autonomy scenarios.

**arXiv ID:** 2602.13159
</details>

<details>
<summary><strong>Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control</strong> - William Chen, Jagdeep Singh Bhatia, Catherine Glossop, Nikhil Mathihalli, Ria Doshi, Andy Tang, Danny Driess, Karl Pertsch, Sergey Levine - [[pdf]](https://arxiv.org/pdf/2602.13193)</summary>

**Abstract:** Pretrained vision-language models (VLMs) can make semantic and visual inferences across diverse settings, providing valuable common-sense priors for robotic control. However, effectively grounding this knowledge in robot behaviors remains an open challenge. Prior methods often employ a hierarchical approach where VLMs reason over high-level commands to be executed by separate low-level policies, e.g., vision-language-action models (VLAs). The interface between VLMs and VLAs is usually natural language task instructions, which fundamentally limits how much VLM reasoning can steer low-level behavior. We thus introduce Steerable Policies: VLAs trained on rich synthetic commands at various levels of abstraction, like subtasks, motions, and grounded pixel coordinates. By improving low-level controllability, Steerable Policies can unlock pretrained knowledge in VLMs, enabling improved task generalization. We demonstrate this benefit by controlling our Steerable Policies with both a learned high-level embodied reasoner and an off-the-shelf VLM prompted to reason over command abstractions via in-context learning. Across extensive real-world manipulation experiments, these two novel methods outperform prior embodied reasoning VLAs and VLM-based hierarchical baselines, including on challenging generalization and long-horizon tasks.
Website: this http URL

**arXiv ID:** 2602.13193
</details>

<details>
<summary><strong>Human Tool: An MCP-Style Framework for Human-Agent Collaboration</strong> - Yuanrong Tang, Huiling Peng, Bingxi Zhao, Hengyang Ding, Hanchao Song, Tianhong Wang, Chen Zhong, Jiangtao Gong - [[pdf]](https://arxiv.org/pdf/2602.12953)</summary>

**Abstract:** Human-AI collaboration faces growing challenges as AI systems increasingly outperform humans on complex tasks, while humans remain responsible for orchestration, validation, and decision oversight. To address this imbalance, we introduce Human Tool, an MCP-style interface abstraction, building on recent Model Context Protocol designs, that exposes humans as callable tools within AI-led, proactive workflows. Here, "tool" denotes a coordination abstraction, not a reduction of human authority or responsibility. Building on LLM-based agent architectures, we operationalize Human Tool by modeling human contributions through structured tool schemas of capabilities, information, and authority. These schemas enable agents to dynamically invoke human input based on relative strengths and reintegrate it through efficient, natural interaction protocols. We validate the framework through controlled studies in both decision-making and creative tasks, demonstrating improved task performance, reduced human workload, and more balanced collaboration dynamics compared to baseline systems. Finally, we discuss implications for human-centered AI design, highlighting how MCP-style human tools enable strong AI leadership while amplifying uniquely human strengths.

**arXiv ID:** 2602.12953
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (21 papers)</h2></summary>

<details>
<summary><strong>To Mix or To Merge: Toward Multi-Domain Reinforcement Learning for Large Language Models</strong> - Haoqing Wang, Xiang Long, Ziheng Li, Yilong Xu, Tingguang Li, Yehui Tang - [[pdf]](https://arxiv.org/pdf/2602.12566)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) plays a key role in stimulating the explicit reasoning capability of Large Language Models (LLMs). We can achieve expert-level performance in some specific domains via RLVR, such as coding or math. When a general multi-domain expert-level model is required, we need to carefully consider the collaboration of RLVR across different domains. The current state-of-the-art models mainly employ two different training paradigms for multi-domain RLVR: mixed multi-task RLVR and separate RLVR followed by model merging. However, most of the works did not provide a detailed comparison and analysis about these paradigms. To this end, we choose multiple commonly used high-level tasks (e.g., math, coding, science, and instruction following) as our target domains and design extensive qualitative and quantitative experiments using open-source datasets. We find the RLVR across domains exhibits few mutual interferences, and reasoning-intensive domains demonstrate mutually synergistic effects. Furthermore, we analyze the internal mechanisms of mutual gains from the perspectives of weight space geometry, model prediction behavior, and information constraints. This project is named as M2RL that means Mixed multi-task training or separate training followed by model Merging for Reinforcement Learning, and the homepage is at this https URL

**arXiv ID:** 2602.12566
</details>

<details>
<summary><strong>Energy-Aware Reinforcement Learning for Robotic Manipulation of Articulated Components in Infrastructure Operation and Maintenance</strong> - Xiaowen Tao, Yinuo Wang, Haitao Ding, Yuanyang Qi, Ziyu Song - [[pdf]](https://arxiv.org/pdf/2602.12288)</summary>

**Abstract:** With the growth of intelligent civil infrastructure and smart cities, operation and maintenance (O&M) increasingly requires safe, efficient, and energy-conscious robotic manipulation of articulated components, including access doors, service drawers, and pipeline valves. However, existing robotic approaches either focus primarily on grasping or target object-specific articulated manipulation, and they rarely incorporate explicit actuation energy into multi-objective optimisation, which limits their scalability and suitability for long-term deployment in real O&M settings. Therefore, this paper proposes an articulation-agnostic and energy-aware reinforcement learning framework for robotic manipulation in intelligent infrastructure O&M. The method combines part-guided 3D perception, weighted point sampling, and PointNet-based encoding to obtain a compact geometric representation that generalises across heterogeneous articulated objects. Manipulation is formulated as a Constrained Markov Decision Process (CMDP), in which actuation energy is explicitly modelled and regulated via a Lagrangian-based constrained Soft Actor-Critic scheme. The policy is trained end-to-end under this CMDP formulation, enabling effective articulated-object operation while satisfying a long-horizon energy budget. Experiments on representative O&M tasks demonstrate 16%-30% reductions in energy consumption, 16%-32% fewer steps to success, and consistently high success rates, indicating a scalable and sustainable solution for infrastructure O&M manipulation.

**arXiv ID:** 2602.12288
</details>

<details>
<summary><strong>Value Bonuses using Ensemble Errors for Exploration in Reinforcement Learning</strong> - Abdul Wahab, Raksha Kumaraswamy, Martha White - [[pdf]](https://arxiv.org/pdf/2602.12375)</summary>

**Abstract:** Optimistic value estimates provide one mechanism for directed exploration in reinforcement learning (RL). The agent acts greedily with respect to an estimate of the value plus what can be seen as a value bonus. The value bonus can be learned by estimating a value function on reward bonuses, propagating local uncertainties around rewards. However, this approach only increases the value bonus for an action retroactively, after seeing a higher reward bonus from that state and action. Such an approach does not encourage the agent to visit a state and action for the first time. In this work, we introduce an algorithm for exploration called Value Bonuses with Ensemble errors (VBE), that maintains an ensemble of random action-value functions (RQFs). VBE uses the errors in the estimation of these RQFs to design value bonuses that provide first-visit optimism and deep exploration. The key idea is to design the rewards for these RQFs in such a way that the value bonus can decrease to zero. We show that VBE outperforms Bootstrap DQN and two reward bonus approaches (RND and ACB) on several classic environments used to test exploration and provide demonstrative experiments that it can scale easily to more complex environments like Atari.

**arXiv ID:** 2602.12375
</details>

<details>
<summary><strong>AstRL: Analog and Mixed-Signal Circuit Synthesis with Deep Reinforcement Learning</strong> - Felicia B. Guo, Ken T. Ho, Andrei Vladimirescu, Borivoje Nikolic - [[pdf]](https://arxiv.org/pdf/2602.12402)</summary>

**Abstract:** Analog and mixed-signal (AMS) integrated circuits (ICs) lie at the core of modern computing and communications systems. However, despite the continued rise in design complexity, advances in AMS automation remain limited. This reflects the central challenge in developing a generalized optimization method applicable across diverse circuit design spaces, many of which are distinct, constrained, and non-differentiable. To address this, our work casts circuit design as a graph generation problem and introduces a novel method of AMS synthesis driven by deep reinforcement learning (AstRL). Based on a policy-gradient approach, AstRL generates circuits directly optimized for user-specified targets within a simulator-embedded environment that provides ground-truth feedback during training. Through behavioral-cloning and discriminator-based similarity rewards, our method demonstrates, for the first time, an expert-aligned paradigm for generalized circuit generation validated in simulation. Importantly, the proposed approach operates at the level of individual transistors, enabling highly expressive, fine-grained topology generation. Strong inductive biases encoded in the action space and environment further drive structurally consistent and valid generation. Experimental results for three realistic design tasks illustrate substantial improvements in conventional design metrics over state-of-the-art baselines, with 100% of generated designs being structurally correct and over 90% demonstrating required functionality.

**arXiv ID:** 2602.12402
</details>

<details>
<summary><strong>Safe Reinforcement Learning via Recovery-based Shielding with Gaussian Process Dynamics Models</strong> - Alexander W. Goodall, Francesco Belardinelli - [[pdf]](https://arxiv.org/pdf/2602.12444)</summary>

**Abstract:** Reinforcement learning (RL) is a powerful framework for optimal decision-making and control but often lacks provable guarantees for safety-critical applications. In this paper, we introduce a novel recovery-based shielding framework that enables safe RL with a provable safety lower bound for unknown and non-linear continuous dynamical systems. The proposed approach integrates a backup policy (shield) with the RL agent, leveraging Gaussian process (GP) based uncertainty quantification to predict potential violations of safety constraints, dynamically recovering to safe trajectories only when necessary. Experience gathered by the 'shielded' agent is used to construct the GP models, with policy optimization via internal model-based sampling - enabling unrestricted exploration and sample efficient learning, without compromising safety. Empirically our approach demonstrates strong performance and strict safety-compliance on a suite of continuous control environments.

**arXiv ID:** 2602.12444
</details>

<details>
<summary><strong>TRACE: Temporal Reasoning via Agentic Context Evolution for Streaming Electronic Health Records (EHRs)</strong> - Zhan Qu, Michael Färber - [[pdf]](https://arxiv.org/pdf/2602.12833)</summary>

**Abstract:** Large Language Models (LLMs) encode extensive medical knowledge but struggle to apply it reliably to longitudinal patient trajectories, where evolving clinical states, irregular timing, and heterogeneous events degrade performance over time. Existing adaptation strategies rely on fine-tuning or retrieval-based augmentation, which introduce computational overhead, privacy constraints, or instability under long contexts. We introduce TRACE (Temporal Reasoning via Agentic Context Evolution), a framework that enables temporal clinical reasoning with frozen LLMs by explicitly structuring and maintaining context rather than extending context windows or updating parameters. TRACE operates over a dual-memory architecture consisting of a static Global Protocol encoding institutional clinical rules and a dynamic Individual Protocol tracking patient-specific state. Four agentic components, Router, Reasoner, Auditor, and Steward, coordinate over this structured memory to support temporal inference and state evolution. The framework maintains bounded inference cost via structured state compression and selectively audits safety-critical clinical decisions. Evaluated on longitudinal clinical event streams from MIMIC-IV, TRACE significantly improves next-event prediction accuracy, protocol adherence, and clinical safety over long-context and retrieval-augmented baselines, while producing interpretable and auditable reasoning traces.

**arXiv ID:** 2602.12833
</details>

<details>
<summary><strong>How to Train Your LLM Web Agent: A Statistical Diagnosis</strong> - Dheeraj Vattikonda, Santhoshi Ravichandran, Emiliano Penaloza, Hadi Nekoei, Megh Thakkar, Thibault Le Sellier de Chezelles, Nicolas Gontier, Miguel Muñoz-Mármol, Sahar Omidi Shayegan, Stefania Raimondo, Xue Liu, Alexandre Drouin, Laurent Charlin, Alexandre Piché, Alexandre Lacoste, Massimo Caccia - [[pdf]](https://arxiv.org/pdf/2507.04103)</summary>

**Abstract:** LLM-based web agents have recently made significant progress, but much of it has occurred in closed-source systems, widening the gap with open-source alternatives. Progress has been held back by two key challenges: first, a narrow focus on single-step tasks that overlooks the complexity of multi-step web interactions; and second, the high compute costs required to post-train LLM-based web agents. To address this, we present the first statistically grounded study on compute allocation for LLM web-agent post-training. Our approach uses a two-stage pipeline, training a Llama 3.1 8B student to imitate a Llama 3.3 70B teacher via supervised fine-tuning (SFT), followed by on-policy reinforcement learning. We find this process highly sensitive to hyperparameter choices, making exhaustive sweeps impractical. To spare others from expensive trial-and-error, we sample 1,370 configurations and use bootstrapping to estimate effective hyperparameters. Our results show that combining SFT with on-policy RL consistently outperforms either approach alone on both WorkArena and MiniWob++. Further, this strategy requires only 55% of the compute to match the peak performance of pure SFT on MiniWob++, effectively pushing the compute-performance Pareto frontier, and is the only strategy that can close the gap with closed-source models.

**arXiv ID:** 2507.04103
</details>

<details>
<summary><strong>Agentic AI Security: Threats, Defenses, Evaluation, and Open Challenges</strong> - Anshuman Chhabra, Shrestha Datta, Shahriar Kabir Nahin, Prasant Mohapatra - [[pdf]](https://arxiv.org/pdf/2510.23883)</summary>

**Abstract:** Agentic AI systems powered by large language models (LLMs) and endowed with planning, tool use, memory, and autonomy, are emerging as powerful, flexible platforms for automation. Their ability to autonomously execute tasks across web, software, and physical environments creates new and amplified security risks, distinct from both traditional AI safety and conventional software security. This survey outlines a taxonomy of threats specific to agentic AI, reviews recent benchmarks and evaluation methodologies, and discusses defense strategies from both technical and governance perspectives. We synthesize current research and highlight open challenges, aiming to support the development of secure-by-design agent systems.

**arXiv ID:** 2510.23883
</details>

<details>
<summary><strong>From Prompt to Product: A Human-Centered Benchmark of Agentic App Generation Systems</strong> - Marcos Ortiz, Justin Hill, Collin Overbay, Ingrida Semenec, Frederic Sauve-Hoover, Jim Schwoebel, Joel Shor - [[pdf]](https://arxiv.org/pdf/2512.18080)</summary>

**Abstract:** Agentic AI systems capable of generating full-stack web applications from natural language prompts ("prompt- to-app") represent a significant shift in software development. However, evaluating these systems remains challenging, as visual polish, functional correctness, and user trust are often misaligned. As a result, it is unclear how existing prompt-to-app tools compare under realistic, human-centered evaluation criteria. In this paper, we introduce a human-centered benchmark for evaluating prompt-to-app systems and conduct a large-scale comparative study of three widely used platforms: Replit, Bolt, and Firebase Studio. Using a diverse set of 96 prompts spanning common web application tasks, we generate 288 unique application artifacts. We evaluate these systems through a large-scale human-rater study involving 205 participants and 1,071 quality-filtered pairwise comparisons, assessing task-based ease of use, visual appeal, perceived completeness, and user trust. Our results show that these systems are not interchangeable: Firebase Studio consistently outperforms competing platforms across all human-evaluated dimensions, achieving the highest win rates for ease of use, trust, visual appeal, and visual appropriateness. Bolt performs competitively on visual appeal but trails Firebase on usability and trust, while Replit underperforms relative to both across most metrics. These findings highlight a persistent gap between visual polish and functional reliability in prompt-to-app systems and demonstrate the necessity of interactive, task-based evaluation. We release our benchmark framework, prompt set, and generated artifacts to support reproducible evaluation and future research in agentic application generation.

**arXiv ID:** 2512.18080
</details>

<details>
<summary><strong>Flow-Factory: A Unified Framework for Reinforcement Learning in Flow-Matching Models</strong> - Bowen Ping, Chengyou Jia, Minnan Luo, Hangwei Qian, Ivor Tsang - [[pdf]](https://arxiv.org/pdf/2602.12529)</summary>

**Abstract:** Reinforcement learning has emerged as a promising paradigm for aligning diffusion and flow-matching models with human preferences, yet practitioners face fragmented codebases, model-specific implementations, and engineering complexity. We introduce Flow-Factory, a unified framework that decouples algorithms, models, and rewards through through a modular, registry-based architecture. This design enables seamless integration of new algorithms and architectures, as demonstrated by our support for GRPO, DiffusionNFT, and AWM across Flux, Qwen-Image, and WAN video models. By minimizing implementation overhead, Flow-Factory empowers researchers to rapidly prototype and scale future innovations with ease. Flow-Factory provides production-ready memory optimization, flexible multi-reward training, and seamless distributed training support. The codebase is available at this https URL.

**arXiv ID:** 2602.12529
</details>

<details>
<summary><strong>Dual-Granularity Contrastive Reward via Generated Episodic Guidance for Efficient Embodied RL</strong> - Xin Liu, Yixuan Li, Yuhui Chen, Yuxing Qin, Haoran Li, Dongbin Zhao - [[pdf]](https://arxiv.org/pdf/2602.12636)</summary>

**Abstract:** Designing suitable rewards poses a significant challenge in reinforcement learning (RL), especially for embodied manipulation. Trajectory success rewards are suitable for human judges or model fitting, but the sparsity severely limits RL sample efficiency. While recent methods have effectively improved RL via dense rewards, they rely heavily on high-quality human-annotated data or abundant expert supervision. To tackle these issues, this paper proposes Dual-granularity contrastive reward via generated Episodic Guidance (DEG), a novel framework to seek sample-efficient dense rewards without requiring human annotations or extensive supervision. Leveraging the prior knowledge of large video generation models, DEG only needs a small number of expert videos for domain adaptation to generate dedicated task guidance for each RL episode. Then, the proposed dual-granularity reward that balances coarse-grained exploration and fine-grained matching, will guide the agent to efficiently approximate the generated guidance video sequentially in the contrastive self-supervised latent space, and finally complete the target task. Extensive experiments on 18 diverse tasks across both simulation and real-world settings show that DEG can not only serve as an efficient exploration stimulus to help the agent quickly discover sparse success rewards, but also guide effective RL and stable policy convergence independently.

**arXiv ID:** 2602.12636
</details>

<details>
<summary><strong>ADEPT: RL-Aligned Agentic Decoding of Emotion via Evidence Probing Tools -- From Consensus Learning to Ambiguity-Driven Emotion Reasoning</strong> - Esther Sun, Bo-Hao Su, Abinay Reddy Naini, Shinji Watanabe, Carlos Busso - [[pdf]](https://arxiv.org/pdf/2602.12714)</summary>

**Abstract:** Speech Large Language Models (SLLMs) enable high-level emotion reasoning but often produce ungrounded, text-biased judgments without verifiable acoustic evidence. In contrast, self-supervised speech encoders such as WavLM provide strong acoustic representations yet remain opaque discriminative models with limited interpretability. To bridge this gap, we introduce ADEPT (Agentic Decoding of Emotion via Evidence Probing Tools), a framework that reframes emotion recognition as a multi-turn inquiry process rather than a single-pass prediction. ADEPT transforms an SLLM into an agent that maintains an evolving candidate emotion set and adaptively invokes dedicated semantic and acoustic probing tools within a structured pipeline of candidate generation, evidence collection, and adjudication. Crucially, ADEPT enables a paradigm shift from consensus learning to ambiguity-driven emotion reasoning. Since human affect exhibits inherent complexity and frequent co-occurrence of emotions, we treat minority annotations as informative perceptual signals rather than discarding them as noise. Finally, we integrate Group Relative Policy Optimization (GRPO) with an Evidence Trust Gate to explicitly couple tool-usage behaviors with prediction quality and enforce evidence-grounded reasoning. Experiments show that ADEPT improves primary emotion accuracy in most settings while substantially improving minor emotion characterization, producing explanations grounded in auditable acoustic and semantic evidence.

**arXiv ID:** 2602.12714
</details>

<details>
<summary><strong>TCRL: Temporal-Coupled Adversarial Training for Robust Constrained Reinforcement Learning in Worst-Case Scenarios</strong> - Wentao Xu, Zhongming Yao, Weihao Li, Zhenghang Song, Yumeng Song, Tianyi Li, Yushuai Li - [[pdf]](https://arxiv.org/pdf/2602.13040)</summary>

**Abstract:** Constrained Reinforcement Learning (CRL) aims to optimize decision-making policies under constraint conditions, making it highly applicable to safety-critical domains such as autonomous driving, robotics, and power grid management. However, existing robust CRL approaches predominantly focus on single-step perturbations and temporally independent adversarial models, lacking explicit modeling of robustness against temporally coupled perturbations. To tackle these challenges, we propose TCRL, a novel temporal-coupled adversarial training framework for robust constrained reinforcement learning (TCRL) in worst-case scenarios. First, TCRL introduces a worst-case-perceived cost constraint function that estimates safety costs under temporally coupled perturbations without the need to explicitly model adversarial attackers. Second, TCRL establishes a dual-constraint defense mechanism on the reward to counter temporally coupled adversaries while maintaining reward unpredictability. Experimental results demonstrate that TCRL consistently outperforms existing methods in terms of robustness against temporally coupled perturbation attacks across a variety of CRL tasks.

**arXiv ID:** 2602.13040
</details>

<details>
<summary><strong>Online reinforcement learning via sparse Gaussian mixture model Q-functions</strong> - Minh Vu, Konstantinos Slavakis - [[pdf]](https://arxiv.org/pdf/2509.14585)</summary>

**Abstract:** This paper introduces a structured and interpretable online policy-iteration framework for reinforcement learning (RL), built around the novel class of sparse Gaussian mixture model Q-functions (S-GMM-QFs). Extending earlier work that trained GMM-QFs offline, the proposed framework develops an online scheme that leverages streaming data to encourage exploration. Model complexity is regulated through sparsification by Hadamard overparametrization, which mitigates overfitting while preserving expressiveness. The parameter space of S-GMM-QFs is naturally endowed with a Riemannian manifold structure, allowing for principled parameter updates via online gradient descent on a smooth objective. Numerical experiments show that S-GMM-QFs match or even outperform dense deep RL (DeepRL) methods on standard benchmarks while using significantly fewer parameters. Moreover, they maintain strong performance even in low-parameter regimes where sparsified DeepRL methods fail to generalize.

**arXiv ID:** 2509.14585
</details>

<details>
<summary><strong>PISHYAR: A Socially Intelligent Smart Cane for Indoor Social Navigation and Multimodal Human-Robot Interaction for Visually Impaired People</strong> - Mahdi Haghighat Joo, Maryam Karimi Jafari, Alireza Taheri - [[pdf]](https://arxiv.org/pdf/2602.12597)</summary>

**Abstract:** This paper presents PISHYAR, a socially intelligent smart cane designed by our group to combine socially aware navigation with multimodal human-AI interaction to support both physical mobility and interactive assistance. The system consists of two components: (1) a social navigation framework implemented on a Raspberry Pi 5 that integrates real-time RGB-D perception using an OAK-D Lite camera, YOLOv8-based object detection, COMPOSER-based collective activity recognition, D* Lite dynamic path planning, and haptic feedback via vibration motors for tasks such as locating a vacant seat; and (2) an agentic multimodal LLM-VLM interaction framework that integrates speech recognition, vision language models, large language models, and text-to-speech, with dynamic routing between voice-only and vision-only modes to enable natural voice-based communication, scene description, and object localization from visual input. The system is evaluated through a combination of simulation-based tests, real-world field experiments, and user-centered studies. Results from simulated and real indoor environments demonstrate reliable obstacle avoidance and socially compliant navigation, achieving an overall system accuracy of approximately 80% under different social conditions. Group activity recognition further shows robust performance across diverse crowd scenarios. In addition, a preliminary exploratory user study with eight visually impaired and low-vision participants evaluates the agentic interaction framework through structured tasks and a UTAUT-based questionnaire reveals high acceptance and positive perceptions of usability, trust, and perceived sociability during our experiments. The results highlight the potential of PISHYAR as a multimodal assistive mobility aid that extends beyond navigation to provide socially interactive support for such users.

**arXiv ID:** 2602.12597
</details>

<details>
<summary><strong>RLinf-Co: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models</strong> - Liangzhi Shi, Shuaihang Chen, Feng Gao, Yinuo Chen, Kang Chen, Tonghe Zhang, Hongzhi Zhang, Weinan Zhang, Chao Yu, Yu Wang - [[pdf]](https://arxiv.org/pdf/2602.12628)</summary>

**Abstract:** Simulation offers a scalable and low-cost way to enrich vision-language-action (VLA) training, reducing reliance on expensive real-robot demonstrations. However, most sim-real co-training methods rely on supervised fine-tuning (SFT), which treats simulation as a static source of demonstrations and does not exploit large-scale closed-loop interaction. Consequently, real-world gains and generalization are often limited. In this paper, we propose an \underline{\textit{RL}}-based sim-real \underline{\textit{Co}}-training \modify{(RL-Co)} framework that leverages interactive simulation while preserving real-world capabilities. Our method follows a generic two-stage design: we first warm-start the policy with SFT on a mixture of real and simulated demonstrations, then fine-tune it with reinforcement learning in simulation while adding an auxiliary supervised loss on real-world data to anchor the policy and mitigate catastrophic forgetting. We evaluate our framework on four real-world tabletop manipulation tasks using two representative VLA architectures, OpenVLA and $\pi_{0.5}$, and observe consistent improvements over real-only fine-tuning and SFT-based co-training, including +24% real-world success on OpenVLA and +20% on $\pi_{0.5}$. Beyond higher success rates, RL co-training yields stronger generalization to unseen task variations and substantially improved real-world data efficiency, providing a practical and scalable pathway for leveraging simulation to enhance real-robot deployment.

**arXiv ID:** 2602.12628
</details>

<details>
<summary><strong>TRANS: Terrain-aware Reinforcement Learning for Agile Navigation of Quadruped Robots under Social Interactions</strong> - Wei Zhu, Irfan Tito Kurniawan, Ye Zhao, Mistuhiro Hayashibe - [[pdf]](https://arxiv.org/pdf/2602.12724)</summary>

**Abstract:** This study introduces TRANS: Terrain-aware Reinforcement learning for Agile Navigation under Social interactions, a deep reinforcement learning (DRL) framework for quadrupedal social navigation over unstructured terrains. Conventional quadrupedal navigation typically separates motion planning from locomotion control, neglecting whole-body constraints and terrain awareness. On the other hand, end-to-end methods are more integrated but require high-frequency sensing, which is often noisy and computationally costly. In addition, most existing approaches assume static environments, limiting their use in human-populated settings. To address these limitations, we propose a two-stage training framework with three DRL pipelines. (1) TRANS-Loco employs an asymmetric actor-critic (AC) model for quadrupedal locomotion, enabling traversal of uneven terrains without explicit terrain or contact observations. (2) TRANS-Nav applies a symmetric AC framework for social navigation, directly mapping transformed LiDAR data to ego-agent actions under differential-drive kinematics. (3) A unified pipeline, TRANS, integrates TRANS-Loco and TRANS-Nav, supporting terrain-aware quadrupedal navigation in uneven and socially interactive environments. Comprehensive benchmarks against locomotion and social navigation baselines demonstrate the effectiveness of TRANS. Hardware experiments further confirm its potential for sim-to-real transfer.

**arXiv ID:** 2602.12724
</details>

<details>
<summary><strong>Agentic AI for Robot Control: Flexible but still Fragile</strong> - Oscar Lima, Marc Vinci, Martin Günther, Marian Renz, Alexander Sung, Sebastian Stock, Johannes Brust, Lennart Niecksch, Zongyao Yi, Felix Igelbrink, Benjamin Kisliuk, Martin Atzmueller, Joachim Hertzberg - [[pdf]](https://arxiv.org/pdf/2602.13081)</summary>

**Abstract:** Recent work leverages the capabilities and commonsense priors of generative models for robot control. In this paper, we present an agentic control system in which a reasoning-capable language model plans and executes tasks by selecting and invoking robot skills within an iterative planner and executor loop. We deploy the system on two physical robot platforms in two settings: (i) tabletop grasping, placement, and box insertion in indoor mobile manipulation (Mobipick) and (ii) autonomous agricultural navigation and sensing (Valdemar). Both settings involve uncertainty, partial observability, sensor noise, and ambiguous natural-language commands. The system exposes structured introspection of its planning and decision process, reacts to exogenous events via explicit event checks, and supports operator interventions that modify or redirect ongoing execution. Across both platforms, our proof-of-concept experiments reveal substantial fragility, including non-deterministic suboptimal behavior, instruction-following errors, and high sensitivity to prompt specification. At the same time, the architecture is flexible: transfer to a different robot and task domain largely required updating the system prompt (domain model, affordances, and action catalogue) and re-binding the same tool interface to the platform-specific skill API.

**arXiv ID:** 2602.13081
</details>

<details>
<summary><strong>Assessing Vision-Language Models for Perception in Autonomous Underwater Robotic Software</strong> - Muhammad Yousaf, Aitor Arrieta, Shaukat Ali, Paolo Arcaini, Shuai Wang - [[pdf]](https://arxiv.org/pdf/2602.10655)</summary>

**Abstract:** Autonomous Underwater Robots (AURs) operate in challenging underwater environments, including low visibility and harsh water conditions. Such conditions present challenges for software engineers developing perception modules for the AUR software. To successfully carry out these tasks, deep learning has been incorporated into the AUR software to support its operations. However, the unique challenges of underwater environments pose difficulties for deep learning models, which often rely on labeled data that is scarce and noisy. This may undermine the trustworthiness of AUR software that relies on perception modules. Vision-Language Models (VLMs) offer promising solutions for AUR software as they generalize to unseen objects and remain robust in noisy conditions by inferring information from contextual cues. Despite this potential, their performance and uncertainty in underwater environments remain understudied from a software engineering perspective. Motivated by the needs of an industrial partner in assurance and risk management for maritime systems to assess the potential use of VLMs in this context, we present an empirical evaluation of VLM-based perception modules within the AUR software. We assess their ability to detect underwater trash by computing performance, uncertainty, and their relationship, to enable software engineers to select appropriate VLMs for their AUR software.

**arXiv ID:** 2602.10655
</details>

<details>
<summary><strong>GatheringSense: AI-Generated Imagery and Embodied Experiences for Understanding Literati Gatherings</strong> - You Zhou, Bingyuan Wang, Hongcheng Guo, Rui Cao, Zeyu Wang - [[pdf]](https://arxiv.org/pdf/2602.12565)</summary>

**Abstract:** Chinese literati gatherings (Wenren Yaji), as a situated form of Chinese traditional culture, remain underexplored in depth. Although generative AI supports powerful multimodal generation, current cultural applications largely emphasize aesthetic reproduction and struggle to convey the deeper meanings of cultural rituals and social frameworks. Based on embodied cognition, we propose an AI-driven dual-path framework for cultural understanding, which we instantiate through GatheringSense, a literati-gathering experience. We conduct a mixed-methods study (N=48) to compare how AI-generated multimodal content and embodied participation complement each other in supporting the understanding of literati gatherings and fostering cultural resonance. Our results show that AI-generated content effectively improves the readability of cultural symbols and initial emotional attraction, yet limitations in physical coherence and micro-level credibility may affect users' satisfaction. In contrast, embodied experience significantly deepens participants' understanding of ritual rules and social roles, and increases their psychological closeness and presence. Based on these findings, we offer empirical evidence and five transferable design implications for generative experience in cultural heritage.

**arXiv ID:** 2602.12565
</details>

<details>
<summary><strong>DuetUI: A Bidirectional Context Loop for Human-Agent Co-Generation of Task-Oriented Interfaces</strong> - Yuan Xu, Shaowen Xiang, Yizhi Song, Ruoting Sun, Xin Tong - [[pdf]](https://arxiv.org/pdf/2509.13444)</summary>

**Abstract:** Large Language Models are reshaping task automation, yet remain limited in complex, multi-step real-world tasks that require aligning with vague user intent and enabling dynamic user override. From a formative study with 12 participants, we found that end-users actively seek to shape task-oriented interfaces rather than relying on one-shot outputs. To address this, we introduce the human-agent co-generation paradigm, materialized in DuetUI. This LLM-empowered system unfolds alongside task progress through a bidirectional context loop-the agent scaffolds the interface by decomposing the task, while the user's direct manipulations implicitly steer the agent's next generation step. In a technical ablation study and a user study with 24 participants, DuetUI improved task efficiency and interface usability, supporting more seamless human-agent collaboration. Our contributions include the proposal of this novel paradigm, the design of a proof-of-concept DuetUI prototype embodying it, and empirical and technical insights from an initial evaluation of how this bidirectional loop may help align agents with human intent and inform future development.

**arXiv ID:** 2509.13444
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
