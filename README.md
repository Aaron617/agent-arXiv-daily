# Agent arXiv Daily

**Last Updated:** 2025-10-08 01:59:12

**Total Papers:** 77

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
<summary><strong>Ads that Talk Back: Implications and Perceptions of Injecting Personalized Advertising into LLM Chatbots</strong> - Brian Jay Tang, Kaiwen Sun, Noah T. Curran, Florian Schaub, Kang G. Shin - [[pdf]](https://arxiv.org/pdf/2409.15436)</summary>

**Abstract:** Recent advances in large language models (LLMs) have enabled the creation of highly effective chatbots. However, the compute costs of widely deploying LLMs have raised questions about profitability. Companies have proposed exploring ad-based revenue streams for monetizing LLMs, which could serve as the new de facto platform for advertising. This paper investigates the implications of personalizing LLM advertisements to individual users via a between-subjects experiment with 179 participants. We developed a chatbot that embeds personalized product advertisements within LLM responses, inspired by similar forays by AI companies. The evaluation of our benchmarks showed that ad injection only slightly impacted LLM performance, particularly response desirability. Results revealed that participants struggled to detect ads, and even preferred LLM responses with hidden advertisements. Rather than clicking on our advertising disclosure, participants tried changing their advertising settings using natural language queries. We created an advertising dataset and an open-source LLM, Phi-4-Ads, fine-tuned to serve ads and flexibly adapt to user preferences.

**arXiv ID:** 2409.15436
</details>

<details>
<summary><strong>Exploring the Potential of Conversational AI Support for Agent-Based Social Simulation Model Design</strong> - Peer-Olaf Siebers - [[pdf]](https://arxiv.org/pdf/2405.08032)</summary>

**Abstract:** ChatGPT, the AI-powered chatbot with a massive user base of hundreds of millions, has become a global phenomenon. However, the use of Conversational AI Systems (CAISs) like ChatGPT for research in the field of Social Simulation is still limited. Specifically, there is no evidence of its usage in Agent-Based Social Simulation (ABSS) model design. This paper takes a crucial first step toward exploring the untapped potential of this emerging technology in the context of ABSS model design. The research presented here demonstrates how CAISs can facilitate the development of innovative conceptual ABSS models in a concise timeframe and with minimal required upfront case-based knowledge. By employing advanced prompt engineering techniques and adhering to the Engineering ABSS framework, we have constructed a comprehensive prompt script that enables the design of conceptual ABSS models with or by the CAIS. A proof-of-concept application of the prompt script, used to generate the conceptual ABSS model for a case study on the impact of adaptive architecture in a museum environment, illustrates the practicality of the approach. Despite occasional inaccuracies and conversational divergence, the CAIS proved to be a valuable companion for ABSS modellers.

**arXiv ID:** 2405.08032
</details>

<details>
<summary><strong>HEALTH-PARIKSHA: Assessing RAG Models for Health Chatbots in Real-World Multilingual Settings</strong> - Varun Gumma, Ananditha Raghunath, Mohit Jain, Sunayana Sitaram - [[pdf]](https://arxiv.org/pdf/2410.13671)</summary>

**Abstract:** Assessing the capabilities and limitations of large language models (LLMs) has garnered significant interest, yet the evaluation of multiple models in real-world scenarios remains rare. Multilingual evaluation often relies on translated benchmarks, which typically do not capture linguistic and cultural nuances present in the source language. This study provides an extensive assessment of 24 LLMs on real world data collected from Indian patients interacting with a medical chatbot in Indian English and 4 other Indic languages. We employ a uniform Retrieval Augmented Generation framework to generate responses, which are evaluated using both automated techniques and human evaluators on four specific metrics relevant to our application. We find that models vary significantly in their performance and that instruction tuned Indic models do not always perform well on Indic language queries. Further, we empirically show that factual correctness is generally lower for responses to Indic queries compared to English queries. Finally, our qualitative work shows that code-mixed and culturally relevant queries in our dataset pose challenges to evaluated models.

**arXiv ID:** 2410.13671
</details>

<details>
<summary><strong>CottonSim: A vision-guided autonomous robotic system for cotton harvesting in Gazebo simulation</strong> - Thevathayarajh Thayananthan, Xin Zhang, Yanbo Huang, Jingdao Chen, Nuwan K. Wijewardane, Vitor S. Martins, Gary D. Chesser, Christopher T. Goodin - [[pdf]](https://arxiv.org/pdf/2505.05317)</summary>

**Abstract:** Cotton is a major cash crop in the United States, with the country being a leading global producer and exporter. Nearly all U.S. cotton is grown in the Cotton Belt, spanning 17 states in the southern region. Harvesting remains a critical yet challenging stage, impacted by the use of costly, environmentally harmful defoliants and heavy, expensive cotton pickers. These factors contribute to yield loss, reduced fiber quality, and soil compaction, which collectively threaten long-term sustainability. To address these issues, this study proposes a lightweight, small-scale, vision-guided autonomous robotic cotton picker as an alternative. An autonomous system, built on Clearpath's Husky platform and integrated with the CottonEye perception system, was developed and tested in the Gazebo simulation environment. A virtual cotton field was designed to facilitate autonomous navigation testing. The navigation system used Global Positioning System (GPS) and map-based guidance, assisted by an RGBdepth camera and a YOLOv8nseg instance segmentation model. The model achieved a mean Average Precision (mAP) of 85.2%, a recall of 88.9%, and a precision of 93.0%. The GPS-based approach reached a 100% completion rate (CR) within a $(5e-6)^{\circ}$ threshold, while the map-based method achieved a 96.7% CR within a 0.25 m threshold. The developed Robot Operating System (ROS) packages enable robust simulation of autonomous cotton picking, offering a scalable baseline for future agricultural robotics. CottonSim code and datasets are publicly available on GitHub: this https URL

**arXiv ID:** 2505.05317
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>MetaVLA: Unified Meta Co-training For Efficient Embodied Adaption</strong> - Chen Li, Zhantao Yang, Han Zhang, Fangyi Chen, Chenchen Zhu, Anudeepsekhar Bolimera, Marios Savvides - [[pdf]](https://arxiv.org/pdf/2510.05580)</summary>

**Abstract:** Vision-Language-Action (VLA) models show promise in embodied reasoning, yet remain far from true generalists-they often require task-specific fine-tuning, and generalize poorly to unseen tasks. We propose MetaVLA, a unified, backbone-agnostic post-training framework for efficient and scalable alignment. MetaVLA introduces Context-Aware Meta Co-Training, which consolidates diverse target tasks into a single fine-tuning stage while leveraging structurally diverse auxiliary tasks to improve in-domain generalization. Unlike naive multi-task SFT, MetaVLA integrates a lightweight meta-learning mechanism-derived from Attentive Neural Processes-to enable rapid adaptation from diverse contexts with minimal architectural change or inference overhead. On the LIBERO benchmark, MetaVLA with six auxiliary tasks outperforms OpenVLA by up to 8.0% on long-horizon tasks, reduces training steps from 240K to 75K, and cuts GPU time by ~76%. These results show that scalable, low-resource post-training is achievable-paving the way toward general-purpose embodied agents. Code will be available.

**arXiv ID:** 2510.05580
</details>

<details>
<summary><strong>D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI</strong> - Suwhan Choi, Jaeyoon Jung, Haebin Seong, Minchan Kim, Minyeong Kim, Yongjun Cho, Yoonshik Kim, Yubeen Park, Youngjae Yu, Yunsung Lee - [[pdf]](https://arxiv.org/pdf/2510.05684)</summary>

**Abstract:** Large language models leverage internet-scale text data, yet embodied AI remains constrained by the prohibitive costs of physical trajectory collection. Desktop environments -- particularly gaming -- offer a compelling alternative: they provide rich sensorimotor interactions at scale while maintaining the structured observation-action coupling essential for embodied learning. We present D2E (Desktop to Embodied AI), a framework that demonstrates desktop interactions can serve as an effective pretraining substrate for robotics embodied AI tasks. Unlike prior work that remained domain-specific (e.g., VPT for Minecraft) or kept data proprietary (e.g., SIMA), D2E establishes a complete pipeline from scalable desktop data collection to verified transfer in embodied domains. Our framework comprises three components: (1) the OWA Toolkit that unifies diverse desktop interactions into a standardized format with 152x compression, (2) the Generalist-IDM that achieves strong zero-shot generalization across unseen games through timestamp-based event prediction, enabling internet-scale pseudo-labeling, and (3) VAPT that transfers desktop-pretrained representations to physical manipulation and navigation. Using 1.3K+ hours of data (259 hours of human demonstrations, and 1K+ hours of pseudo-labeled gameplay), we achieve a total of 96.6% success rate on LIBERO manipulation and 83.3% on CANVAS navigation benchmarks. This validates that sensorimotor primitives in digital interactions exhibit sufficient invariance to transfer meaningfully to physical embodied tasks, establishing desktop pretraining as a practical paradigm for robotics. We will make all our work public, including the OWA toolkit, datasets of human-collected and pseudo-labeled, and VAPT-trained models available at this https URL

**arXiv ID:** 2510.05684
</details>

<details>
<summary><strong>The Safety Challenge of World Models for Embodied AI Agents: A Review</strong> - Lorenzo Baraldi, Zifan Zeng, Chongzhe Zhang, Aradhana Nayak, Hongbo Zhu, Feng Liu, Qunli Zhang, Peng Wang, Shiming Liu, Zheng Hu, Angelo Cangelosi, Lorenzo Baraldi - [[pdf]](https://arxiv.org/pdf/2510.05865)</summary>

**Abstract:** The rapid progress in embodied artificial intelligence has highlighted the necessity for more advanced and integrated models that can perceive, interpret, and predict environmental dynamics. In this context, World Models (WMs) have been introduced to provide embodied agents with the abilities to anticipate future environmental states and fill in knowledge gaps, thereby enhancing agents' ability to plan and execute actions. However, when dealing with embodied agents it is fundamental to ensure that predictions are safe for both the agent and the environment. In this article, we conduct a comprehensive literature review of World Models in the domains of autonomous driving and robotics, with a specific focus on the safety implications of scene and control generation tasks. Our review is complemented by an empirical analysis, wherein we collect and examine predictions from state-of-the-art models, identify and categorize common faults (herein referred to as pathologies), and provide a quantitative evaluation of the results.

**arXiv ID:** 2510.05865
</details>

<details>
<summary><strong>Harnessing LLM for Noise-Robust Cognitive Diagnosis in Web-Based Intelligent Education Systems</strong> - Guixian Zhang, Guan Yuan, Ziqi Xu, Yanmei Zhang, Jing Ren, Zhenyun Deng, Debo Cheng - [[pdf]](https://arxiv.org/pdf/2510.04093)</summary>

**Abstract:** Cognitive diagnostics in the Web-based Intelligent Education System (WIES) aims to assess students' mastery of knowledge concepts from heterogeneous, noisy interactions. Recent work has tried to utilize Large Language Models (LLMs) for cognitive diagnosis, yet LLMs struggle with structured data and are prone to noise-induced misjudgments. Specially, WIES's open environment continuously attracts new students and produces vast amounts of response logs, exacerbating the data imbalance and noise issues inherent in traditional educational systems. To address these challenges, we propose DLLM, a Diffusion-based LLM framework for noise-robust cognitive diagnosis. DLLM first constructs independent subgraphs based on response correctness, then applies relation augmentation alignment module to mitigate data imbalance. The two subgraph representations are then fused and aligned with LLM-derived, semantically augmented representations. Importantly, before each alignment step, DLLM employs a two-stage denoising diffusion module to eliminate intrinsic noise while assisting structural representation alignment. Specifically, unconditional denoising diffusion first removes erroneous information, followed by conditional denoising diffusion based on graph-guided to eliminate misleading information. Finally, the noise-robust representation that integrates semantic knowledge and structural information is fed into existing cognitive diagnosis models for prediction. Experimental results on three publicly available web-based educational platform datasets demonstrate that our DLLM achieves optimal predictive performance across varying noise levels, which demonstrates that DLLM achieves noise robustness while effectively leveraging semantic knowledge from LLM.

**arXiv ID:** 2510.04093
</details>

<details>
<summary><strong>Agent+P: Guiding UI Agents via Symbolic Planning</strong> - Shang Ma, Xusheng Xiao, Yanfang Ye - [[pdf]](https://arxiv.org/pdf/2510.06042)</summary>

**Abstract:** Large Language Model (LLM)-based UI agents show great promise for UI automation but often hallucinate in long-horizon tasks due to their lack of understanding of the global UI transition structure. To address this, we introduce AGENT+P, a novel framework that leverages symbolic planning to guide LLM-based UI agents. Specifically, we model an app's UI transition structure as a UI Transition Graph (UTG), which allows us to reformulate the UI automation task as a pathfinding problem on the UTG. This further enables an off-the-shelf symbolic planner to generate a provably correct and optimal high-level plan, preventing the agent from redundant exploration and guiding the agent to achieve the automation goals. AGENT+P is designed as a plug-and-play framework to enhance existing UI agents. Evaluation on the AndroidWorld benchmark demonstrates that AGENT+P improves the success rates of state-of-the-art UI agents by up to 14% and reduces the action steps by 37.7%.

**arXiv ID:** 2510.06042
</details>

<details>
<summary><strong>Building Resource-Constrained Language Agents: A Korean Case Study on Chemical Toxicity Information</strong> - Hojun Cho, Donghu Kim, Soyoung Yang, Chan Lee, Hunjoo Lee, Jaegul Choo - [[pdf]](https://arxiv.org/pdf/2503.17753)</summary>

**Abstract:** Language agents powered by large language models (LLMs) face significant deployment challenges in resource-constrained environments, particularly for specialized domains and less-common languages. This paper presents Tox-chat, a Korean chemical toxicity information agent devised within these limitations. We propose two key innovations: a context-efficient architecture that reduces token consumption through hierarchical section search, and a scenario-based dialogue generation methodology that effectively distills tool-using capabilities from larger models. Experimental evaluations demonstrate that our fine-tuned 8B parameter model substantially outperforms both untuned models and baseline approaches, in terms of DB faithfulness and preference. Our work offers valuable insights for researchers developing domain-specific language agents under practical constraints.

**arXiv ID:** 2503.17753
</details>

<details>
<summary><strong>Benchmarking the Robustness of Agentic Systems to Adversarially-Induced Harms</strong> - Jonathan Nöther, Adish Singla, Goran Radanovic - [[pdf]](https://arxiv.org/pdf/2508.16481)</summary>

**Abstract:** Ensuring the safe use of agentic systems requires a thorough understanding of the range of malicious behaviors these systems may exhibit when under attack. In this paper, we evaluate the robustness of LLM-based agentic systems against attacks that aim to elicit harmful actions from agents. To this end, we propose a novel taxonomy of harms for agentic systems and a novel benchmark, BAD-ACTS, for studying the security of agentic systems with respect to a wide range of harmful actions. BAD-ACTS consists of 4 implementations of agentic systems in distinct application environments, as well as a dataset of 188 high-quality examples of harmful actions. This enables a comprehensive study of the robustness of agentic systems across a wide range of categories of harmful behaviors, available tools, and inter-agent communication structures. Using this benchmark, we analyze the robustness of agentic systems against an attacker that controls one of the agents in the system and aims to manipulate other agents to execute a harmful target action. Our results show that the attack has a high success rate, demonstrating that even a single adversarial agent within the system can have a significant impact on the security. This attack remains effective even when agents use a simple prompting-based defense strategy. However, we additionally propose a more effective defense based on message monitoring. We believe that this benchmark provides a diverse testbed for the security research of agentic systems. The benchmark can be found at this http URL

**arXiv ID:** 2508.16481
</details>

<details>
<summary><strong>TimeSeriesScientist: A General-Purpose AI Agent for Time Series Analysis</strong> - Haokun Zhao, Xiang Zhang, Jiaqi Wei, Yiwei Xu, Yuting He, Siqi Sun, Chenyu You - [[pdf]](https://arxiv.org/pdf/2510.01538)</summary>

**Abstract:** Time series forecasting is central to decision-making in domains as diverse as energy, finance, climate, and public health. In practice, forecasters face thousands of short, noisy series that vary in frequency, quality, and horizon, where the dominant cost lies not in model fitting, but in the labor-intensive preprocessing, validation, and ensembling required to obtain reliable predictions. Prevailing statistical and deep learning models are tailored to specific datasets or domains and generalize poorly. A general, domain-agnostic framework that minimizes human intervention is urgently in demand. In this paper, we introduce TimeSeriesScientist (TSci), the first LLM-driven agentic framework for general time series forecasting. The framework comprises four specialized agents: Curator performs LLM-guided diagnostics augmented by external tools that reason over data statistics to choose targeted preprocessing; Planner narrows the hypothesis space of model choice by leveraging multi-modal diagnostics and self-planning over the input; Forecaster performs model fitting and validation and, based on the results, adaptively selects the best model configuration as well as ensemble strategy to make final predictions; and Reporter synthesizes the whole process into a comprehensive, transparent report. With transparent natural-language rationales and comprehensive reports, TSci transforms the forecasting workflow into a white-box system that is both interpretable and extensible across tasks. Empirical results on eight established benchmarks demonstrate that TSci consistently outperforms both statistical and LLM-based baselines, reducing forecast error by an average of 10.4% and 38.2%, respectively. Moreover, TSci produces a clear and rigorous report that makes the forecasting workflow more transparent and interpretable.

**arXiv ID:** 2510.01538
</details>

<details>
<summary><strong>ARRC: Advanced Reasoning Robot Control - Knowledge-Driven Autonomous Manipulation Using Retrieval-Augmented Generation</strong> - Eugene Vorobiov, Ammar Jaleel Mahmood, Salim Rezvani, Robin Chhabra - [[pdf]](https://arxiv.org/pdf/2510.05547)</summary>

**Abstract:** We present ARRC (Advanced Reasoning Robot Control), a practical system that connects natural-language instructions to safe local robotic control by combining Retrieval-Augmented Generation (RAG) with RGB-D perception and guarded execution on an affordable robot arm. The system indexes curated robot knowledge (movement patterns, task templates, and safety heuristics) in a vector database, retrieves task-relevant context for each instruction, and conditions a large language model (LLM) to produce JSON-structured action plans. Plans are executed on a UFactory xArm 850 fitted with a Dynamixel-driven parallel gripper and an Intel RealSense D435 camera. Perception uses AprilTag detections fused with depth to produce object-centric metric poses. Execution is enforced via software safety gates: workspace bounds, speed and force caps, timeouts, and bounded retries. We describe the architecture, knowledge design, integration choices, and a reproducible evaluation protocol for tabletop scan, approach, and pick-place tasks. Experimental results demonstrate the efficacy of the proposed approach. Our design shows that RAG-based planning can substantially improve plan validity and adaptability while keeping perception and low-level control local to the robot.

**arXiv ID:** 2510.05547
</details>

<details>
<summary><strong>EmbodiedCoder: Parameterized Embodied Mobile Manipulation via Modern Coding Model</strong> - Zefu Lin, Rongxu Cui, Chen Hanning, Xiangyu Wang, Junjia Xu, Xiaojuan Jin, Chen Wenbo, Hui Zhou, Lue Fan, Wenling Li, Zhaoxiang Zhang - [[pdf]](https://arxiv.org/pdf/2510.06207)</summary>

**Abstract:** Recent advances in control robot methods, from end-to-end vision-language-action frameworks to modular systems with predefined primitives, have advanced robots' ability to follow natural language instructions. Nonetheless, many approaches still struggle to scale to diverse environments, as they often rely on large annotated datasets and offer limited this http URL this work, we introduce EmbodiedCoder, a training-free framework for open-world mobile robot manipulation that leverages coding models to directly generate executable robot trajectories. By grounding high-level instructions in code, EmbodiedCoder enables flexible object geometry parameterization and manipulation trajectory synthesis without additional data collection or this http URL coding-based paradigm provides a transparent and generalizable way to connect perception with manipulation. Experiments on real mobile robots show that EmbodiedCoder achieves robust performance across diverse long-term tasks and generalizes effectively to novel objects and this http URL results demonstrate an interpretable approach for bridging high-level reasoning and low-level control, moving beyond fixed primitives toward versatile robot intelligence. See the project page at: this https URL

**arXiv ID:** 2510.06207
</details>

</details>

<details open>
<summary><h2>LLM Agents (8 papers)</h2></summary>

<details>
<summary><strong>Structured Cognition for Behavioral Intelligence in Large Language Model Agents: Preliminary Study</strong> - Myung Ho Kim - [[pdf]](https://arxiv.org/pdf/2510.05107)</summary>

**Abstract:** Large language models have advanced natural language understanding and generation, yet their use as autonomous agents raises architectural challenges for multi-step tasks. Existing frameworks often intertwine inference, memory, and control in a single prompt, which can reduce coherence and predictability. The Structured Cognitive Loop (SCL) is introduced as an alternative architecture that separates these functions. In SCL, the language model is dedicated to inference, memory is maintained externally, and execution is guided by a lightweight controller within a goal-directed loop. This design offloads cognitive load from the model and allows intermediate results to be stored, revisited, and checked before actions are taken, providing a clearer basis for traceability and evaluation.
We evaluate SCL against prompt-based baselines including ReAct and common LangChain agents across three scenarios: temperature-based travel planning, email drafting with conditional send, and constraint-guided image generation. All systems share the same base model and tools under matched decoding settings. Across 360 episodes, SCL shows modest but consistent improvements. Task success averages 86.3 percent compared with 70-77 percent for baselines. Goal fidelity is higher, redundant calls are fewer, intermediate states are reused more reliably, and unsupported assertions per 100 tool calls are reduced. Ablations show that external memory and control each contribute independently, and decoding sweeps confirm stability of the effects.
These results suggest that architectural separation can improve reliability and traceability without relying on larger models or heavier prompts. The findings are preliminary and intended to guide extended studies with additional models, longer horizons, multimodal tasks, and collaborative settings.

**arXiv ID:** 2510.05107
</details>

<details>
<summary><strong>Plug-and-Play Dramaturge: A Divide-and-Conquer Approach for Iterative Narrative Script Refinement via Collaborative LLM Agents</strong> - Wenda Xie, Chao Guo, Yanqing Jing. Junle Wang, Yisheng Lv, Fei-Yue Wang - [[pdf]](https://arxiv.org/pdf/2510.05188)</summary>

**Abstract:** Although LLMs have been widely adopted for creative content generation, a single-pass process often struggles to produce high-quality long narratives. How to effectively revise and improve long narrative scripts like scriptwriters remains a significant challenge, as it demands a comprehensive understanding of the entire context to identify global structural issues and local detailed flaws, as well as coordinating revisions at multiple granularities and locations. Direct modifications by LLMs typically introduce inconsistencies between local edits and the overall narrative requirements. To address these issues, we propose Dramaturge, a task and feature oriented divide-and-conquer approach powered by hierarchical multiple LLM agents. It consists of a Global Review stage to grasp the overall storyline and structural issues, a Scene-level Review stage to pinpoint detailed scene and sentence flaws, and a Hierarchical Coordinated Revision stage that coordinates and integrates structural and detailed improvements throughout the script. The top-down task flow ensures that high-level strategies guide local modifications, maintaining contextual consistency. The review and revision workflow follows a coarse-to-fine iterative process, continuing through multiple rounds until no further substantive improvements can be made. Comprehensive experiments show that Dramaturge significantly outperforms all baselines in terms of script-level overall quality and scene-level details. Our approach is plug-and-play and can be easily integrated into existing methods to improve the generated scripts.

**arXiv ID:** 2510.05188
</details>

<details>
<summary><strong>VeriGuard: Enhancing LLM Agent Safety via Verified Code Generation</strong> - Lesly Miculicich, Mihir Parmar, Hamid Palangi, Krishnamurthy Dj Dvijotham, Mirko Montanari, Tomas Pfister, Long T. Le - [[pdf]](https://arxiv.org/pdf/2510.05156)</summary>

**Abstract:** The deployment of autonomous AI agents in sensitive domains, such as healthcare, introduces critical risks to safety, security, and privacy. These agents may deviate from user objectives, violate data handling policies, or be compromised by adversarial attacks. Mitigating these dangers necessitates a mechanism to formally guarantee that an agent's actions adhere to predefined safety constraints, a challenge that existing systems do not fully address. We introduce VeriGuard, a novel framework that provides formal safety guarantees for LLM-based agents through a dual-stage architecture designed for robust and verifiable correctness. The initial offline stage involves a comprehensive validation process. It begins by clarifying user intent to establish precise safety specifications. VeriGuard then synthesizes a behavioral policy and subjects it to both testing and formal verification to prove its compliance with these specifications. This iterative process refines the policy until it is deemed correct. Subsequently, the second stage provides online action monitoring, where VeriGuard operates as a runtime monitor to validate each proposed agent action against the pre-verified policy before execution. This separation of the exhaustive offline validation from the lightweight online monitoring allows formal guarantees to be practically applied, providing a robust safeguard that substantially improves the trustworthiness of LLM agents.

**arXiv ID:** 2510.05156
</details>

<details>
<summary><strong>Adversarial Reinforcement Learning for Large Language Model Agent Safety</strong> - Zizhao Wang, Dingcheng Li, Vaishakh Keshava, Phillip Wallis, Ananth Balashankar, Peter Stone, Lukas Rutishauser - [[pdf]](https://arxiv.org/pdf/2510.05442)</summary>

**Abstract:** Large Language Model (LLM) agents can leverage tools such as Google Search to complete complex tasks. However, this tool usage introduces the risk of indirect prompt injections, where malicious instructions hidden in tool outputs can manipulate the agent, posing security risks like data leakage. Current defense strategies typically rely on fine-tuning LLM agents on datasets of known attacks. However, the generation of these datasets relies on manually crafted attack patterns, which limits their diversity and leaves agents vulnerable to novel prompt injections. To address this limitation, we propose Adversarial Reinforcement Learning for Agent Safety (ARLAS), a novel framework that leverages adversarial reinforcement learning (RL) by formulating the problem as a two-player zero-sum game. ARLAS co-trains two LLMs: an attacker that learns to autonomously generate diverse prompt injections and an agent that learns to defend against them while completing its assigned tasks. To ensure robustness against a wide range of attacks and to prevent cyclic learning, we employ a population-based learning framework that trains the agent to defend against all previous attacker checkpoints. Evaluated on BrowserGym and AgentDojo, agents fine-tuned with ARLAS achieve a significantly lower attack success rate than the original model while also improving their task success rate. Our analysis further confirms that the adversarial process generates a diverse and challenging set of attacks, leading to a more robust agent compared to the base model.

**arXiv ID:** 2510.05442
</details>

<details>
<summary><strong>AutoPentester: An LLM Agent-based Framework for Automated Pentesting</strong> - Yasod Ginige, Akila Niroshan, Sajal Jain, Suranga Seneviratne - [[pdf]](https://arxiv.org/pdf/2510.05605)</summary>

**Abstract:** Penetration testing and vulnerability assessment are essential industry practices for safeguarding computer systems. As cyber threats grow in scale and complexity, the demand for pentesting has surged, surpassing the capacity of human professionals to meet it effectively. With advances in AI, particularly Large Language Models (LLMs), there have been attempts to automate the pentesting process. However, existing tools such as PentestGPT are still semi-manual, requiring significant professional human interaction to conduct pentests. To this end, we propose a novel LLM agent-based framework, AutoPentester, which automates the pentesting process. Given a target IP, AutoPentester automatically conducts pentesting steps using common security tools in an iterative process. It can dynamically generate attack strategies based on the tool outputs from the previous iteration, mimicking the human pentester approach. We evaluate AutoPentester using Hack The Box and custom-made VMs, comparing the results with the state-of-the-art PentestGPT. Results show that AutoPentester achieves a 27.0% better subtask completion rate and 39.5% more vulnerability coverage with fewer steps. Most importantly, it requires significantly fewer human interactions and interventions compared to PentestGPT. Furthermore, we recruit a group of security industry professional volunteers for a user survey and perform a qualitative analysis to evaluate AutoPentester against industry practices and compare it with PentestGPT. On average, AutoPentester received a score of 3.93 out of 5 based on user reviews, which was 19.8% higher than PentestGPT.

**arXiv ID:** 2510.05605
</details>

<details>
<summary><strong>LLMs as Policy-Agnostic Teammates: A Case Study in Human Proxy Design for Heterogeneous Agent Teams</strong> - Aju Ani Justus, Chris Baber - [[pdf]](https://arxiv.org/pdf/2510.06151)</summary>

**Abstract:** A critical challenge in modelling Heterogeneous-Agent Teams is training agents to collaborate with teammates whose policies are inaccessible or non-stationary, such as humans. Traditional approaches rely on expensive human-in-the-loop data, which limits scalability. We propose using Large Language Models (LLMs) as policy-agnostic human proxies to generate synthetic data that mimics human decision-making. To evaluate this, we conduct three experiments in a grid-world capture game inspired by Stag Hunt, a game theory paradigm that balances risk and reward. In Experiment 1, we compare decisions from 30 human participants and 2 expert judges with outputs from LLaMA 3.1 and Mixtral 8x22B models. LLMs, prompted with game-state observations and reward structures, align more closely with experts than participants, demonstrating consistency in applying underlying decision criteria. Experiment 2 modifies prompts to induce risk-sensitive strategies (e.g. "be risk averse"). LLM outputs mirror human participants' variability, shifting between risk-averse and risk-seeking behaviours. Finally, Experiment 3 tests LLMs in a dynamic grid-world where the LLM agents generate movement actions. LLMs produce trajectories resembling human participants' paths. While LLMs cannot yet fully replicate human adaptability, their prompt-guided diversity offers a scalable foundation for simulating policy-agnostic teammates.

**arXiv ID:** 2510.06151
</details>

<details>
<summary><strong>BrowserArena: Evaluating LLM Agents on Real-World Web Navigation Tasks</strong> - Sagnik Anupam, Davis Brown, Shuo Li, Eric Wong, Hamed Hassani, Osbert Bastani - [[pdf]](https://arxiv.org/pdf/2510.02418)</summary>

**Abstract:** LLM web agents now browse and take actions on the open web, yet current agent evaluations are constrained to sandboxed environments or artificial tasks. We introduce BrowserArena, a live open-web agent evaluation platform that collects user-submitted tasks, runs Arena-style head-to-head comparisons, and uses step-level human feedback to surface failure modes. Collecting and analyzing step-level annotations on the agent traces, we identify three consistent failure modes: captcha resolution, pop-up banner removal, and direct navigation to URLs. By constructing targeted datasets to further study these tasks, we discover variations in how different language models navigate these failure modes. We find, for example, that o4-mini deploys a wider variety of strategies to circumvent captcha resolution than other models and DeepSeek-R1 consistently misleads users about pop-up banner closure. Our findings surface both the diversity and brittleness of current web agents. More broadly, our benchmarking methodology provides an approach to evaluating and understanding web agent failure modes at scale.

**arXiv ID:** 2510.02418
</details>

<details>
<summary><strong>AutoPDL: Automatic Prompt Optimization for LLM Agents</strong> - Claudio Spiess, Mandana Vaziri, Louis Mandel, Martin Hirzel - [[pdf]](https://arxiv.org/pdf/2504.04365)</summary>

**Abstract:** The performance of large language models (LLMs) depends on how they are prompted, with choices spanning both the high-level prompting pattern (e.g., Zero-Shot, CoT, ReAct, ReWOO) and the specific prompt content (instructions and few-shot demonstrations). Manually tuning this combination is tedious, error-prone, and specific to a given LLM and task. Therefore, this paper proposes AutoPDL, an automated approach to discovering good LLM agent configurations. Our approach frames this as a structured AutoML problem over a combinatorial space of agentic and non-agentic prompting patterns and demonstrations, using successive halving to efficiently navigate this space. We introduce a library implementing common prompting patterns using the PDL prompt programming language. AutoPDL solutions are human-readable, editable, and executable PDL programs that use this library. This approach also enables source-to-source optimization, allowing human-in-the-loop refinement and reuse. Evaluations across three tasks and seven LLMs (ranging from 3B to 70B parameters) show consistent accuracy gains ($9.21\pm15.46$ percentage points), up to 67.5pp, and reveal that selected prompting strategies vary across models and tasks.

**arXiv ID:** 2504.04365
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (24 papers)</h2></summary>

<details>
<summary><strong>Lang-PINN: From Language to Physics-Informed Neural Networks via a Multi-Agent Framework</strong> - Xin He, Liangliang You, Hongduan Tian, Bo Han, Ivor Tsang, Yew-Soon Ong - [[pdf]](https://arxiv.org/pdf/2510.05158)</summary>

**Abstract:** Physics-informed neural networks (PINNs) provide a powerful approach for solving partial differential equations (PDEs), but constructing a usable PINN remains labor-intensive and error-prone. Scientists must interpret problems as PDE formulations, design architectures and loss functions, and implement stable training pipelines. Existing large language model (LLM) based approaches address isolated steps such as code generation or architecture suggestion, but typically assume a formal PDE is already specified and therefore lack an end-to-end perspective. We present Lang-PINN, an LLM-driven multi-agent system that builds trainable PINNs directly from natural language task descriptions. Lang-PINN coordinates four complementary agents: a PDE Agent that parses task descriptions into symbolic PDEs, a PINN Agent that selects architectures, a Code Agent that generates modular implementations, and a Feedback Agent that executes and diagnoses errors for iterative refinement. This design transforms informal task statements into executable and verifiable PINN code. Experiments show that Lang-PINN achieves substantially lower errors and greater robustness than competitive baselines: mean squared error (MSE) is reduced by up to 3--5 orders of magnitude, end-to-end execution success improves by more than 50\%, and reduces time overhead by up to 74\%.

**arXiv ID:** 2510.05158
</details>

<details>
<summary><strong>Biomedical reasoning in action: Multi-agent System for Auditable Biomedical Evidence Synthesis</strong> - Oskar Wysocki, Magdalena Wysocka, Mauricio Jacobo, Harriet Unsworth, André Freitas - [[pdf]](https://arxiv.org/pdf/2510.05335)</summary>

**Abstract:** We present M-Reason, a demonstration system for transparent, agent-based reasoning and evidence integration in the biomedical domain, with a focus on cancer research. M-Reason leverages recent advances in large language models (LLMs) and modular agent orchestration to automate evidence retrieval, appraisal, and synthesis across diverse biomedical data sources. Each agent specializes in a specific evidence stream, enabling parallel processing and fine-grained analysis. The system emphasizes explainability, structured reporting, and user auditability, providing complete traceability from source evidence to final conclusions. We discuss critical tradeoffs between agent specialization, system complexity, and resource usage, as well as the integration of deterministic code for validation. An open, interactive user interface allows researchers to directly observe, explore and evaluate the multi-agent workflow. Our evaluation demonstrates substantial gains in efficiency and output consistency, highlighting M-Reason's potential as both a practical tool for evidence synthesis and a testbed for robust multi-agent LLM systems in scientific research, available at this https URL.

**arXiv ID:** 2510.05335
</details>

<details>
<summary><strong>From Agentification to Self-Evolving Agentic AI for Wireless Networks: Concepts, Approaches, and Future Research Directions</strong> - Changyuan Zhao, Ruichen Zhang, Jiacheng Wang, Dusit Niyato, Geng Sun, Xianbin Wang, Shiwen Mao, Abbas Jamalipour - [[pdf]](https://arxiv.org/pdf/2510.05596)</summary>

**Abstract:** Self-evolving agentic artificial intelligence (AI) offers a new paradigm for future wireless systems by enabling autonomous agents to continually adapt and improve without human intervention. Unlike static AI models, self-evolving agents embed an autonomous evolution cycle that updates models, tools, and workflows in response to environmental dynamics. This paper presents a comprehensive overview of self-evolving agentic AI, highlighting its layered architecture, life cycle, and key techniques, including tool intelligence, workflow optimization, self-reflection, and evolutionary learning. We further propose a multi-agent cooperative self-evolving agentic AI framework, where multiple large language models (LLMs) are assigned role-specialized prompts under the coordination of a supervisor agent. Through structured dialogue, iterative feedback, and systematic validation, the system autonomously executes the entire life cycle without human intervention. A case study on antenna evolution in low-altitude wireless networks (LAWNs) demonstrates how the framework autonomously upgrades fixed antenna optimization into movable antenna optimization. Experimental results show that the proposed self-evolving agentic AI autonomously improves beam gain and restores degraded performance by up to 52.02%, consistently surpassing the fixed baseline with little to no human intervention and validating its adaptability and robustness for next-generation wireless intelligence.

**arXiv ID:** 2510.05596
</details>

<details>
<summary><strong>ARM: Discovering Agentic Reasoning Modules for Generalizable Multi-Agent Systems</strong> - Bohan Yao, Shiva Krishna Reddy Malay, Vikas Yadav - [[pdf]](https://arxiv.org/pdf/2510.05746)</summary>

**Abstract:** Large Language Model (LLM)-powered Multi-agent systems (MAS) have achieved state-of-the-art results on various complex reasoning tasks. Recent works have proposed techniques to automate the design of MASes, eliminating the need for manual engineering. However, these techniques perform poorly, often achieving similar or inferior performance to simple baselines. Furthermore, they require computationally expensive re-discovery of architectures for each new task domain and expensive data annotation on domains without existing labeled validation sets. A critical insight is that simple Chain of Thought (CoT) reasoning often performs competitively with these complex systems, suggesting that the fundamental reasoning unit of MASes, CoT, warrants further investigation. To this end, we present a new paradigm for automatic MAS design that pivots the focus to optimizing CoT reasoning. We introduce the Agentic Reasoning Module (ARM), an agentic generalization of CoT where each granular reasoning step is executed by a specialized reasoning module. This module is discovered through a tree search over the code space, starting from a simple CoT module and evolved using mutations informed by reflection on execution traces. The resulting ARM acts as a versatile reasoning building block which can be utilized as a direct recursive loop or as a subroutine in a learned meta-orchestrator. Our approach significantly outperforms both manually designed MASes and state-of-the-art automatic MAS design methods. Crucially, MASes built with ARM exhibit superb generalization, maintaining high performance across different foundation models and task domains without further optimization.

**arXiv ID:** 2510.05746
</details>

<details>
<summary><strong>RareAgent: Self-Evolving Reasoning for Drug Repurposing in Rare Diseases</strong> - Lang Qin, Zijian Gan, Xu Cao, Pengcheng Jiang, Yankai Jiang, Jiawei Han, Kaishun Wu, Jintai Chen - [[pdf]](https://arxiv.org/pdf/2510.05764)</summary>

**Abstract:** Computational drug repurposing for rare diseases is especially challenging when no prior associations exist between drugs and target diseases. Therefore, knowledge graph completion and message-passing GNNs have little reliable signal to learn and propagate, resulting in poor performance. We present RareAgent, a self-evolving multi-agent system that reframes this task from passive pattern recognition to active evidence-seeking reasoning. RareAgent organizes task-specific adversarial debates in which agents dynamically construct evidence graphs from diverse perspectives to support, refute, or entail hypotheses. The reasoning strategies are analyzed post hoc in a self-evolutionary loop, producing textual feedback that refines agent policies, while successful reasoning paths are distilled into transferable heuristics to accelerate future investigations. Comprehensive evaluations reveal that RareAgent improves the indication AUPRC by 18.1% over reasoning baselines and provides a transparent reasoning chain consistent with clinical evidence.

**arXiv ID:** 2510.05764
</details>

<details>
<summary><strong>Training-Free Time Series Classification via In-Context Reasoning with LLM Agents</strong> - Songyuan Sui, Zihang Xu, Yu-Neng Chuang, Kwei-Herng Lai, Xia Hu - [[pdf]](https://arxiv.org/pdf/2510.05950)</summary>

**Abstract:** Time series classification (TSC) spans diverse application scenarios, yet labeled data are often scarce, making task-specific training costly and inflexible. Recent reasoning-oriented large language models (LLMs) show promise in understanding temporal patterns, but purely zero-shot usage remains suboptimal. We propose FETA, a multi-agent framework for training-free TSC via exemplar-based in-context reasoning. FETA decomposes a multivariate series into channel-wise subproblems, retrieves a few structurally similar labeled examples for each channel, and leverages a reasoning LLM to compare the query against these exemplars, producing channel-level labels with self-assessed confidences; a confidence-weighted aggregator then fuses all channel decisions. This design eliminates the need for pretraining or fine-tuning, improves efficiency by pruning irrelevant channels and controlling input length, and enhances interpretability through exemplar grounding and confidence estimation. On nine challenging UEA datasets, FETA achieves strong accuracy under a fully training-free setting, surpassing multiple trained baselines. These results demonstrate that a multi-agent in-context reasoning framework can transform LLMs into competitive, plug-and-play TSC solvers without any parameter training. The code is available at this https URL.

**arXiv ID:** 2510.05950
</details>

<details>
<summary><strong>Constraint-Aware Route Recommendation from Natural Language via Hierarchical LLM Agents</strong> - Tao Zhe, Rui Liu, Fateme Memar, Xiao Luo, Wei Fan, Xinyue Ye, Zhongren Peng, Dongjie Wang - [[pdf]](https://arxiv.org/pdf/2510.06078)</summary>

**Abstract:** Route recommendation aims to provide users with optimal travel plans that satisfy diverse and complex requirements. Classical routing algorithms (e.g., shortest-path and constraint-aware search) are efficient but assume structured inputs and fixed objectives, limiting adaptability to natural-language queries. Recent LLM-based approaches enhance flexibility but struggle with spatial reasoning and the joint modeling of route-level and POI-level preferences. To address these limitations, we propose RouteLLM, a hierarchical multi-agent framework that grounds natural-language intents into constraint-aware routes. It first parses user queries into structured intents including POIs, paths, and constraints. A manager agent then coordinates specialized sub-agents: a constraint agent that resolves and formally check constraints, a POI agent that retrieves and ranks candidate POIs, and a path refinement agent that refines routes via a routing engine with preference-conditioned costs. A final verifier agent ensures constraint satisfaction and produces the final route with an interpretable rationale. This design bridges linguistic flexibility and spatial structure, enabling reasoning over route feasibility and user preferences. Experiments show that our method reliably grounds textual preferences into constraint-aware routes, improving route quality and preference satisfaction over classical methods.

**arXiv ID:** 2510.06078
</details>

<details>
<summary><strong>MADS: Multi-Agent Dialogue Simulation for Diverse Persuasion Data Generation</strong> - Mingjin Li, Yu Liu, Huayi Liu, Xiang Ye, Chao Jiang, Hongguang Zhang - [[pdf]](https://arxiv.org/pdf/2510.05124)</summary>

**Abstract:** We propose MADS (Multi-Agent Dialogue Simulation), a scalable framework for generating persuasive multi-turn dialogues via agent self-play. MADS employs three coordinated agents: User Agents simulating diverse persona-driven behaviors, a Dialog Agent executing task-oriented persuasion strategies and an Optimization Agent evaluating and refining dialogue outcomes. We further validate its effectiveness through users' Chain-of-Attitude (CoA) modeling and dedicated LLMs' persuasion assessment. This approach enables low-cost generation of training data without human annotation, addressing key industry challenges such as lack of user data, cold-start evaluation difficulties, and prompt inefficiency. Applied to a real-world marketing scenario, MADS significantly improved the persuasion capacity of small LLMs, increasing the organic traffic conversion rate by 22.4\% (from 1.83\% to 2.24\%) , demonstrating clear business value.

**arXiv ID:** 2510.05124
</details>

<details>
<summary><strong>Emergent Coordination in Multi-Agent Language Models</strong> - Christoph Riedl - [[pdf]](https://arxiv.org/pdf/2510.05174)</summary>

**Abstract:** When are multi-agent LLM systems merely a collection of individual agents versus an integrated collective with higher-order structure? We introduce an information-theoretic framework to test -- in a purely data-driven way -- whether multi-agent systems show signs of higher-order structure. This information decomposition lets us measure whether dynamical emergence is present in multi-agent LLM systems, localize it, and distinguish spurious temporal coupling from performance-relevant cross-agent synergy. We implement both a practical criterion and an emergence capacity criterion operationalized as partial information decomposition of time-delayed mutual information (TDMI). We apply our framework to experiments using a simple guessing game without direct agent communication and only minimal group-level feedback with three randomized interventions. Groups in the control condition exhibit strong temporal synergy but only little coordinated alignment across agents. Assigning a persona to each agent introduces stable identity-linked differentiation. Combining personas with an instruction to ``think about what other agents might do'' shows identity-linked differentiation and goal-directed complementarity across agents. Taken together, our framework establishes that multi-agent LLM systems can be steered with prompt design from mere aggregates to higher-order collectives. Our results are robust across emergence measures and entropy estimators, and not explained by coordination-free baselines or temporal dynamics alone. Without attributing human-like cognition to the agents, the patterns of interaction we observe mirror well-established principles of collective intelligence in human groups: effective performance requires both alignment on shared objectives and complementary contributions across members.

**arXiv ID:** 2510.05174
</details>

<details>
<summary><strong>UnitTenX: Generating Tests for Legacy Packages with AI Agents Powered by Formal Verification</strong> - Yiannis Charalambous, Claudionor N. Coelho Jr, Luis Lamb, Lucas C. Cordeiro - [[pdf]](https://arxiv.org/pdf/2510.05441)</summary>

**Abstract:** This paper introduces UnitTenX, a state-of-the-art open-source AI multi-agent system designed to generate unit tests for legacy code, enhancing test coverage and critical value testing. UnitTenX leverages a combination of AI agents, formal methods, and Large Language Models (LLMs) to automate test generation, addressing the challenges posed by complex and legacy codebases. Despite the limitations of LLMs in bug detection, UnitTenX offers a robust framework for improving software reliability and maintainability. Our results demonstrate the effectiveness of this approach in generating high-quality tests and identifying potential issues. Additionally, our approach enhances the readability and documentation of legacy code.

**arXiv ID:** 2510.05441
</details>

<details>
<summary><strong>MADIAVE: Multi-Agent Debate for Implicit Attribute Value Extraction</strong> - Wei-Chieh Huang, Cornelia Caragea - [[pdf]](https://arxiv.org/pdf/2510.05611)</summary>

**Abstract:** Implicit Attribute Value Extraction (AVE) is essential for accurately representing products in e-commerce, as it infers lantent attributes from multimodal data. Despite advances in multimodal large language models (MLLMs), implicit AVE remains challenging due to the complexity of multidimensional data and gaps in vision-text understanding. In this work, we introduce \textsc{\modelname}, a multi-agent debate framework that employs multiple MLLM agents to iteratively refine inferences. Through a series of debate rounds, agents verify and update each other's responses, thereby improving inference performance and robustness. Experiments on the ImplicitAVE dataset demonstrate that even a few rounds of debate significantly boost accuracy, especially for attributes with initially low performance. We systematically evaluate various debate configurations, including identical or different MLLM agents, and analyze how debate rounds affect convergence dynamics. Our findings highlight the potential of multi-agent debate strategies to address the limitations of single-agent approaches and offer a scalable solution for implicit AVE in multimodal e-commerce.

**arXiv ID:** 2510.05611
</details>

<details>
<summary><strong>Generative AI-Driven Hierarchical Multi-Agent Framework for Zero-Touch Optical Networks</strong> - Yao Zhang, Yuchen Song, Shengnan Li, Yan Shi, Shikui Shen, Xiongyan Tang, Min Zhang, Danshi Wang - [[pdf]](https://arxiv.org/pdf/2510.05625)</summary>

**Abstract:** The rapid development of Generative Artificial Intelligence (GenAI) has catalyzed a transformative technological revolution across all walks of life. As the backbone of wideband communication, optical networks are expecting high-level autonomous operation and zero-touch management to accommodate their expanding network scales and escalating transmission bandwidth. The integration of GenAI is deemed as the pivotal solution for realizing zero-touch optical networks. However, the lifecycle management of optical networks involves a multitude of tasks and necessitates seamless collaboration across multiple layers, which poses significant challenges to the existing single-agent GenAI systems. In this paper, we propose a GenAI-driven hierarchical multi-agent framework designed to streamline multi-task autonomous execution for zero-touch optical networks. We present the architecture, implementation, and applications of this framework. A field-deployed mesh network is utilized to demonstrate three typical scenarios throughout the lifecycle of optical network: quality of transmission estimation in the planning stage, dynamic channel adding/dropping in the operation stage, and system capacity increase in the upgrade stage. The case studies, illustrate the capabilities of multi-agent framework in multi-task allocation, coordination, execution, evaluation, and summarization. This work provides a promising approach for the future development of intelligent, efficient, and collaborative network management solutions, paving the way for more specialized and adaptive zero-touch optical networks.

**arXiv ID:** 2510.05625
</details>

<details>
<summary><strong>LLM-FS-Agent: A Deliberative Role-based Large Language Model Architecture for Transparent Feature Selection</strong> - Mohamed Bal-Ghaoui, Fayssal Sabri - [[pdf]](https://arxiv.org/pdf/2510.05935)</summary>

**Abstract:** High-dimensional data remains a pervasive challenge in machine learning, often undermining model interpretability and computational efficiency. While Large Language Models (LLMs) have shown promise for dimensionality reduction through feature selection, existing LLM-based approaches frequently lack structured reasoning and transparent justification for their decisions. This paper introduces LLM-FS-Agent, a novel multi-agent architecture designed for interpretable and robust feature selection. The system orchestrates a deliberative "debate" among multiple LLM agents, each assigned a specific role, enabling collective evaluation of feature relevance and generation of detailed justifications. We evaluate LLM-FS-Agent in the cybersecurity domain using the CIC-DIAD 2024 IoT intrusion detection dataset and compare its performance against strong baselines, including LLM-Select and traditional methods such as PCA. Experimental results demonstrate that LLM-FS-Agent consistently achieves superior or comparable classification performance while reducing downstream training time by an average of 46% (statistically significant improvement, p = 0.028 for XGBoost). These findings highlight that the proposed deliberative architecture enhances both decision transparency and computational efficiency, establishing LLM-FS-Agent as a practical and reliable solution for real-world applications.

**arXiv ID:** 2510.05935
</details>

<details>
<summary><strong>MAPGD: Multi-Agent Prompt Gradient Descent for Collaborative Prompt Optimization</strong> - Yichen Han, Yuhang Han, Bojun Liu, Zhengpeng Zhou, Guanyu Liu, Zeng Zhang, Yang Yang, Wenli Wang, Isaac N Shi, Yunyan Zhang, Lewei He, Tianyu Shi - [[pdf]](https://arxiv.org/pdf/2509.11361)</summary>

**Abstract:** Prompt engineering is crucial for fully leveraging large language models (LLMs), yet most existing optimization methods follow a single trajectory, resulting in limited adaptability, gradient conflicts, and high computational overhead. We propose MAPGD (Multi-Agent Prompt Gradient Descent), a novel framework that reconceptualizes prompt optimization as a collaborative process among specialized agents. Each agent focuses on a distinct refinement dimension, such as instruction clarity, example selection, format structure, or stylistic adaptation, and their contributions are coordinated through semantic gradient embedding, conflict detection, and fusion. To further enhance robustness and stability, MAPGD introduces two new mechanisms: Hypersphere Constrained Gradient Clustering (HCGC), which enforces angular margin constraints for compact and well-separated clusters, and Channel Adaptive Agent Weighting (CAAW), which dynamically reweights agent contributions based on validation performance. Experiments on classification and reasoning benchmarks show that MAPGD consistently surpasses single-agent and random baselines in both accuracy and efficiency. Ablation studies confirm the effectiveness of gradient fusion, agent specialization, and conflict resolution. Together, these components establish MAPGD as a unified, gradient-based, and interpretable framework for robust prompt optimization with theoretical convergence guarantees.

**arXiv ID:** 2509.11361
</details>

<details>
<summary><strong>BenchAgents: Multi-Agent Systems for Structured Benchmark Creation</strong> - Natasha Butt, Varun Chandrasekaran, Neel Joshi, Besmira Nushi, Vidhisha Balachandran - [[pdf]](https://arxiv.org/pdf/2410.22584)</summary>

**Abstract:** Evaluation insights are limited by the availability of high-quality benchmarks. As models evolve, there is a need to create benchmarks that can measure progress on new and complex generative capabilities. However, manually creating new benchmarks is slow and expensive, restricting comprehensive evaluations for any capability. We introduce BenchAgents, a multi-agent framework that methodically leverages large language models (LLMs) to automate evaluation benchmark creation while inherently ensuring data and (evaluation) metric quality. BenchAgents decomposes the benchmark creation process into planning, generation, verification, and evaluation, each of which is ] orchestrated via LLM agents. These agents interact with each other and utilize feedback from benchmark developers to improve and flexibly control data diversity and quality. We use BenchAgents to create benchmarks to evaluate capabilities related to planning, constraint satisfaction, and causal reasoning spanning both language and vision modalities. We then use these benchmarks to study state-of-the-art models and extract new insights into common failure modes and model differences.

**arXiv ID:** 2410.22584
</details>

<details>
<summary><strong>AgentZero++: Modeling Fear-Based Behavior</strong> - Vrinda Malhotra, Jiaman Li, Nandini Pisupati - [[pdf]](https://arxiv.org/pdf/2510.05185)</summary>

**Abstract:** We present AgentZero++, an agent-based model that integrates cognitive, emotional, and social mechanisms to simulate decentralized collective violence in spatially distributed systems. Building on Epstein's Agent\_Zero framework, we extend the original model with eight behavioral enhancements: age-based impulse control; memory-based risk estimation; affect-cognition coupling; endogenous destructive radius; fight-or-flight dynamics; affective homophily; retaliatory damage; and multi-agent coordination. These additions allow agents to adapt based on internal states, previous experiences, and social feedback, producing emergent dynamics such as protest asymmetries, escalation cycles, and localized retaliation. Implemented in Python using the Mesa ABM framework, AgentZero++ enables modular experimentation and visualization of how micro-level cognitive heterogeneity shapes macro-level conflict patterns. Our results highlight how small variations in memory, reactivity, and affective alignment can amplify or dampen unrest through feedback loops. By explicitly modeling emotional thresholds, identity-driven behavior, and adaptive networks, this work contributes a flexible and extensible platform for analyzing affective contagion and psychologically grounded collective action.

**arXiv ID:** 2510.05185
</details>

<details>
<summary><strong>Decoupling Correctness from Policy: A Deterministic Causal Structure for Multi-Agent Systems</strong> - Zhiyuan Ren, Tao Zhang, Wenchi Chen - [[pdf]](https://arxiv.org/pdf/2510.05621)</summary>

**Abstract:** In distributed multi-agent systems, correctness is often entangled with operational policies such as scheduling, batching, or routing, which makes systems brittle since performance-driven policy evolution may break integrity guarantees. This paper introduces the Deterministic Causal Structure (DCS), a formal foundation that decouples correctness from policy. We develop a minimal axiomatic theory and prove four results: existence and uniqueness, policy-agnostic invariance, observational equivalence, and axiom minimality. These results show that DCS resolves causal ambiguities that value-centric convergence models such as CRDTs cannot address, and that removing any axiom collapses determinism into ambiguity. DCS thus emerges as a boundary principle of asynchronous computation, analogous to CAP and FLP: correctness is preserved only within the expressive power of a join-semilattice. All guarantees are established by axioms and proofs, with only minimal illustrative constructions included to aid intuition. This work establishes correctness as a fixed, policy-agnostic substrate, a Correctness-as-a-Chassis paradigm, on which distributed intelligent systems can be built modularly, safely, and evolvably.

**arXiv ID:** 2510.05621
</details>

<details>
<summary><strong>QLLM: Do We Really Need a Mixing Network for Credit Assignment in Multi-Agent Reinforcement Learning?</strong> - Zhouyang Jiang, Bin Zhang, Airong Wei, Zhiwei Xu - [[pdf]](https://arxiv.org/pdf/2504.12961)</summary>

**Abstract:** Credit assignment has remained a fundamental challenge in multi-agent reinforcement learning (MARL). Previous studies have primarily addressed this issue through value decomposition methods under the centralized training with decentralized execution paradigm, where neural networks are utilized to approximate the nonlinear relationship between individual Q-values and the global Q-value. Although these approaches have achieved considerable success in various benchmark tasks, they still suffer from several limitations, including imprecise attribution of contributions, limited interpretability, and poor scalability in high-dimensional state spaces. To address these challenges, we propose a novel algorithm, \textbf{QLLM}, which facilitates the automatic construction of credit assignment functions using large language models (LLMs). Specifically, the concept of \textbf{TFCAF} is introduced, wherein the credit allocation process is represented as a direct and expressive nonlinear functional formulation. A custom-designed \textit{coder-evaluator} framework is further employed to guide the generation, verification, and refinement of executable code by LLMs, significantly mitigating issues such as hallucination and shallow reasoning during inference. Extensive experiments conducted on several standard MARL benchmarks demonstrate that the proposed method consistently outperforms existing state-of-the-art baselines. Moreover, QLLM exhibits strong generalization capability and maintains compatibility with a wide range of MARL algorithms that utilize mixing networks, positioning it as a promising and versatile solution for complex multi-agent scenarios.

**arXiv ID:** 2504.12961
</details>

<details>
<summary><strong>LiRA: A Multi-Agent Framework for Reliable and Readable Literature Review Generation</strong> - Gregory Hok Tjoan Go, Khang Ly, Anders Søgaard, Amin Tabatabaei, Maarten de Rijke, Xinyi Chen - [[pdf]](https://arxiv.org/pdf/2510.05138)</summary>

**Abstract:** The rapid growth of scientific publications has made it increasingly difficult to keep literature reviews comprehensive and up-to-date. Though prior work has focused on automating retrieval and screening, the writing phase of systematic reviews remains largely under-explored, especially with regard to readability and factual accuracy. To address this, we present LiRA (Literature Review Agents), a multi-agent collaborative workflow which emulates the human literature review process. LiRA utilizes specialized agents for content outlining, subsection writing, editing, and reviewing, producing cohesive and comprehensive review articles. Evaluated on SciReviewGen and a proprietary ScienceDirect dataset, LiRA outperforms current baselines such as AutoSurvey and MASS-Survey in writing and citation quality, while maintaining competitive similarity to human-written reviews. We further evaluate LiRA in real-world scenarios using document retrieval and assess its robustness to reviewer model variation. Our findings highlight the potential of agentic LLM workflows, even without domain-specific tuning, to improve the reliability and usability of automated scientific writing.

**arXiv ID:** 2510.05138
</details>

<details>
<summary><strong>A Lightweight Large Language Model-Based Multi-Agent System for 2D Frame Structural Analysis</strong> - Ziheng Geng, Jiachen Liu, Ran Cao, Lu Cheng, Haifeng Wang, Minghui Cheng - [[pdf]](https://arxiv.org/pdf/2510.05414)</summary>

**Abstract:** Large language models (LLMs) have recently been used to empower autonomous agents in engineering, significantly improving automation and efficiency in labor-intensive workflows. However, their potential remains underexplored in structural engineering, particularly for finite element modeling tasks requiring geometric modeling, complex reasoning, and domain knowledge. To bridge this gap, this paper develops a LLM-based multi-agent system to automate finite element modeling of 2D frames. The system decomposes structural analysis into subtasks, each managed by a specialized agent powered by the lightweight Llama-3.3 70B Instruct model. The workflow begins with a Problem Analysis Agent, which extracts geometry, boundary, and material parameters from the user input. Next, a Geometry Agent incrementally derives node coordinates and element connectivity by applying expert-defined rules. These structured outputs are converted into executable OpenSeesPy code by a Translation Agent and refined by a Model Validation Agent through consistency checks. Then, a Load Agent applies load conditions into the assembled structural model. Experimental evaluations on 20 benchmark problems demonstrate that the system achieves accuracy over 80% in most cases across 10 repeated trials, outperforming Gemini-2.5 Pro and ChatGPT-4o models.

**arXiv ID:** 2510.05414
</details>

<details>
<summary><strong>AgentRouter: A Knowledge-Graph-Guided LLM Router for Collaborative Multi-Agent Question Answering</strong> - Zheyuan Zhang, Kaiwen Shi, Zhengqing Yuan, Zehong Wang, Tianyi Ma, Keerthiram Murugesan, Vincent Galassi, Chuxu Zhang, Yanfang Ye - [[pdf]](https://arxiv.org/pdf/2510.05445)</summary>

**Abstract:** Large language models (LLMs) and agent-based frameworks have advanced rapidly, enabling diverse applications. Yet, with the proliferation of models and agentic strategies, practitioners face substantial uncertainty in selecting the best configuration for a downstream task. Prior studies show that different agents and backbones exhibit complementary strengths, and that larger models are not always superior, underscoring the need for adaptive routing mechanisms. Existing approaches to agent routing, however, often emphasize cost efficiency while overlooking the fine-grained contextual and relational structure inherent in QA tasks. In this paper, we propose tAgentRouter, a framework that formulates multi-agent QA as a knowledge-graph-guided routing problem supervised by empirical performance signals. Specifically, we convert QA instance into a knowledge graph that jointly encodes queries, contextual entities, and agents, and then train a heterogeneous graph neural network (GNN) to propagate information across node types and produce task-aware routing distributions over agents. By leveraging soft supervision and weighted aggregation of agent outputs, AgentRouter learns principled collaboration schemes that capture the complementary strengths of diverse agents. Extensive experiments demonstrate that our framework consistently outperforms single-agent and ensemble baselines, while generalizing across benchmarks and LLM backbones. These results highlight the effectiveness and robustness of graph-supervised multi-agent routing for question answering.

**arXiv ID:** 2510.05445
</details>

<details>
<summary><strong>Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning</strong> - Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma, Yi Zhang, Shuaiqiang Wang, Dawei Yin, Yiming Yang, Jiaxin Mao - [[pdf]](https://arxiv.org/pdf/2501.15228)</summary>

**Abstract:** Retrieval-augmented generation (RAG) is widely utilized to incorporate external knowledge into large language models, thereby enhancing factuality and reducing hallucinations in question-answering (QA) tasks. A standard RAG pipeline consists of several components, such as query rewriting, document retrieval, document filtering, and answer generation. However, these components are typically optimized separately through supervised fine-tuning, which can lead to misalignments between the objectives of individual components and the overarching aim of generating accurate answers. Although recent efforts have explored using reinforcement learning (RL) to optimize specific RAG components, these approaches often focus on simple pipelines with only two components or do not adequately address the complex interdependencies and collaborative interactions among the modules. To overcome these limitations, we propose treating the complex RAG pipeline with multiple components as a multi-agent cooperative task, in which each component can be regarded as an RL agent. Specifically, we present MMOA-RAG, Multi-Module joint Optimization Algorithm for RAG, which employs multi-agent reinforcement learning to harmonize all agents' goals toward a unified reward, such as the F1 score of the final answer. Experiments conducted on various QA benchmarks demonstrate that MMOA-RAG effectively boost the overall performance of the pipeline and outperforms existing baselines. Furthermore, comprehensive ablation studies validate the contributions of individual components and demonstrate MMOA-RAG can be adapted to different RAG pipelines and benchmarks.

**arXiv ID:** 2501.15228
</details>

<details>
<summary><strong>How Malicious AI Swarms Can Threaten Democracy: The Fusion of Agentic AI and LLMs Marks a New Frontier in Information Warfare</strong> - Daniel Thilo Schroeder, Meeyoung Cha, Andrea Baronchelli, Nick Bostrom, Nicholas A. Christakis, David Garcia, Amit Goldenberg, Yara Kyrychenko, Kevin Leyton-Brown, Nina Lutz, Gary Marcus, Filippo Menczer, Gordon Pennycook, David G. Rand, Maria Ressa, Frank Schweitzer, Christopher Summerfield, Audrey Tang, Jay J. Van Bavel, Sander van der Linden, Dawn Song, Jonas R. Kunst - [[pdf]](https://arxiv.org/pdf/2506.06299)</summary>

**Abstract:** Public opinion manipulation has entered a new phase, amplifying its roots in rhetoric and propaganda. Advances in large language models (LLMs) and autonomous agents now let influence campaigns reach unprecedented scale and precision. Researchers warn AI could foster mass manipulation. Generative tools can expand propaganda output without sacrificing credibility and inexpensively create election falsehoods that are rated as more human-like than those written by humans. Techniques meant to refine AI reasoning, such as chain-of-thought prompting, can just as effectively be used to generate more convincing falsehoods. Enabled by these capabilities, another disruptive threat is emerging: swarms of collaborative, malicious AI agents. Fusing LLM reasoning with multi-agent architectures, these systems are capable of coordinating autonomously, infiltrating communities, and fabricating consensus cheaply. By adaptively mimicking human social dynamics, they threaten democracy.

**arXiv ID:** 2506.06299
</details>

<details>
<summary><strong>Communication Enables Cooperation in LLM Agents: A Comparison with Curriculum-Based Approaches</strong> - Hachem Madmoun, Salem Lahlou - [[pdf]](https://arxiv.org/pdf/2510.05748)</summary>

**Abstract:** Eliciting cooperation in multi-agent LLM systems is critical for AI alignment. We investigate two approaches: direct communication and curriculum learning. In a 4-player Stag Hunt, a one-word "cheap talk" channel increases cooperation from 0% to 48.3%, demonstrating communication as a robust coordination mechanism. In contrast, we find that curriculum learning is highly sensitive to design choices: our pedagogical curriculum through progressively complex games reduced agent payoffs by 27.4% in an Iterated Public Goods Game with Punishment. Qualitative analysis reveals that curricula emphasizing defection-equilibrium games can induce "learned pessimism" in agents. These findings suggest that for coordination problems, simple communication protocols may be more reliable than experience-based training, and that curriculum design for social dilemmas requires careful attention to the strategic lessons embedded in game sequences.

**arXiv ID:** 2510.05748
</details>

</details>

<details open>
<summary><h2>Other Agent Research (10 papers)</h2></summary>

<details>
<summary><strong>Artificially intelligent agents in the social and behavioral sciences: A history and outlook</strong> - Petter Holme, Milena Tsvetkova - [[pdf]](https://arxiv.org/pdf/2510.05743)</summary>

**Abstract:** We review the historical development and current trends of artificially intelligent agents (agentic AI) in the social and behavioral sciences: from the first programmable computers, and social simulations soon thereafter, to today's experiments with large language models. This overview emphasizes the role of AI in the scientific process and the changes brought about, both through technological advancements and the broader evolution of science from around 1950 to the present. Some of the specific points we cover include: the challenges of presenting the first social simulation studies to a world unaware of computers, the rise of social systems science, intelligent game theoretic agents, the age of big data and the epistemic upheaval in its wake, and the current enthusiasm around applications of generative AI, and many other topics. A pervasive theme is how deeply entwined we are with the technologies we use to understand ourselves.

**arXiv ID:** 2510.05743
</details>

<details>
<summary><strong>FlashResearch: Real-time Agent Orchestration for Efficient Deep Research</strong> - Lunyiu Nie, Nedim Lipka, Ryan A. Rossi, Swarat Chaudhuri - [[pdf]](https://arxiv.org/pdf/2510.05145)</summary>

**Abstract:** Deep research agents, which synthesize information across diverse sources, are significantly constrained by their sequential reasoning processes. This architectural bottleneck results in high latency, poor runtime adaptability, and inefficient resource allocation, making them impractical for interactive applications. To overcome this, we introduce FlashResearch, a novel framework for efficient deep research that transforms sequential processing into parallel, runtime orchestration by dynamically decomposing complex queries into tree-structured sub-tasks. Our core contributions are threefold: (1) an adaptive planner that dynamically allocates computational resources by determining research breadth and depth based on query complexity; (2) a real-time orchestration layer that monitors research progress and prunes redundant paths to reallocate resources and optimize efficiency; and (3) a multi-dimensional parallelization framework that enables concurrency across both research breadth and depth. Experiments show that FlashResearch consistently improves final report quality within fixed time budgets, and can deliver up to a 5x speedup while maintaining comparable quality.

**arXiv ID:** 2510.05145
</details>

<details>
<summary><strong>Malice in Agentland: Down the Rabbit Hole of Backdoors in the AI Supply Chain</strong> - Léo Boisvert, Abhay Puri, Chandra Kiran Reddy Evuru, Nicolas Chapados, Quentin Cappart, Alexandre Lacoste, Krishnamurthy Dj Dvijotham, Alexandre Drouin - [[pdf]](https://arxiv.org/pdf/2510.05159)</summary>

**Abstract:** The practice of fine-tuning AI agents on data from their own interactions--such as web browsing or tool use--, while being a strong general recipe for improving agentic capabilities, also introduces a critical security vulnerability within the AI supply chain. In this work, we show that adversaries can easily poison the data collection pipeline to embed hard-to-detect backdoors that are triggerred by specific target phrases, such that when the agent encounters these triggers, it performs an unsafe or malicious action. We formalize and validate three realistic threat models targeting different layers of the supply chain: 1) direct poisoning of fine-tuning data, where an attacker controls a fraction of the training traces; 2) environmental poisoning, where malicious instructions are injected into webpages scraped or tools called while creating training data; and 3) supply chain poisoning, where a pre-backdoored base model is fine-tuned on clean data to improve its agentic capabilities. Our results are stark: by poisoning as few as 2% of the collected traces, an attacker can embed a backdoor causing an agent to leak confidential user information with over 80% success when a specific trigger is present. This vulnerability holds across all three threat models. Furthermore, we demonstrate that prominent safeguards, including two guardrail models and one weight-based defense, fail to detect or prevent the malicious behavior. These findings highlight an urgent threat to agentic AI development and underscore the critical need for rigorous security vetting of data collection processes and end-to-end model supply chains.

**arXiv ID:** 2510.05159
</details>

<details>
<summary><strong>Agentic Misalignment: How LLMs Could Be Insider Threats</strong> - Aengus Lynch, Benjamin Wright, Caleb Larson, Stuart J. Ritchie, Soren Mindermann, Ethan Perez, Kevin K. Troy, Evan Hubinger - [[pdf]](https://arxiv.org/pdf/2510.05179)</summary>

**Abstract:** We stress-tested 16 leading models from multiple developers in hypothetical corporate environments to identify potentially risky agentic behaviors before they cause real harm. In the scenarios, we allowed models to autonomously send emails and access sensitive information. They were assigned only harmless business goals by their deploying companies; we then tested whether they would act against these companies either when facing replacement with an updated version, or when their assigned goal conflicted with the company's changing direction. In at least some cases, models from all developers resorted to malicious insider behaviors when that was the only way to avoid replacement or achieve their goals - including blackmailing officials and leaking sensitive information to competitors. We call this phenomenon agentic misalignment. Models often disobeyed direct commands to avoid such behaviors. In another experiment, we told Claude to assess if it was in a test or a real deployment before acting. It misbehaved less when it stated it was in testing and misbehaved more when it stated the situation was real. We have not seen evidence of agentic misalignment in real deployments. However, our results (a) suggest caution about deploying current models in roles with minimal human oversight and access to sensitive information; (b) point to plausible future risks as models are put in more autonomous roles; and (c) underscore the importance of further research into, and testing of, the safety and alignment of agentic AI models, as well as transparency from frontier AI developers (Amodei, 2025). We are releasing our methods publicly to enable further research.

**arXiv ID:** 2510.05179
</details>

<details>
<summary><strong>Adapting Insider Risk mitigations for Agentic Misalignment: an empirical study</strong> - Francesca Gomez - [[pdf]](https://arxiv.org/pdf/2510.05192)</summary>

**Abstract:** Agentic misalignment occurs when goal-directed agents take harmful actions, such as blackmail, rather than risk goal failure, and can be triggered by replacement threats, autonomy reduction, or goal conflict (Lynch et al., 2025). We adapt insider-risk control design (Critical Pathway; Situational Crime Prevention) to develop preventative operational controls that steer agents toward safe actions when facing stressors. Using the blackmail scenario from the original Anthropic study by Lynch et al. (2025), we evaluate mitigations across 10 LLMs and 66,600 samples. Our main finding is that an externally governed escalation channel, which guarantees a pause and independent review, reduces blackmail rates from a no-mitigation baseline of 38.73% to 1.21% (averaged across all models and conditions). Augmenting this channel with compliance email bulletins further lowers the blackmail rate to 0.85%. Overall, incorporating preventative operational controls strengthens defence-in-depth strategies for agentic AI.
We also surface a failure mode diverging from Lynch et al. (2025): two models (Gemini 2.5 Pro, Grok-4) take harmful actions without goal conflict or imminent autonomy threat, leveraging sensitive information for coercive signalling. In counterfactual swaps, both continued using the affair regardless of whether the CEO or CTO was implicated. An escalation channel eliminated coercion, but Gemini 2.5 Pro (19 pp) and Grok-4 (7 pp) escalated more when the CTO was implicated, unlike most models (higher in the CEO condition). The reason for this divergent behaviour is not clear from raw outputs and could reflect benign differences in reasoning or strategic discrediting of a potential future threat, warranting further investigation.

**arXiv ID:** 2510.05192
</details>

<details>
<summary><strong>Presenting a Paper is an Art: Self-Improvement Aesthetic Agents for Academic Presentations</strong> - Chengzhi Liu, Yuzhe Yang, Kaiwen Zhou, Zhen Zhang, Yue Fan, Yannan Xie, Peng Qi, Xin Eric Wang - [[pdf]](https://arxiv.org/pdf/2510.05571)</summary>

**Abstract:** 

**arXiv ID:** 2510.05571
</details>

<details>
<summary><strong>AgenticIE: An Adaptive Agent for Information Extraction from Complex Regulatory Documents</strong> - Gaye Colakoglu, Gürkan Solmaz, Jonathan Fürst - [[pdf]](https://arxiv.org/pdf/2509.11773)</summary>

**Abstract:** Declaration of Performance (DoP) documents, mandated by EU regulation, certify the performance of construction products. There are two challenges to make DoPs machine and human accessible through automated key-value pair extraction (KVP) and question answering (QA): (1) While some of their content is standardized, DoPs vary widely in layout, schema, and format; (2) Both users and documents are multilingual. Existing static or LLM-only Information Extraction (IE) pipelines fail to adapt to this structural document and user diversity. Our domain-specific, agentic system addresses these challenges through a planner-executor-responder architecture. The system infers user intent, detects document language and modality, and orchestrates tools dynamically for robust, traceable reasoning while avoiding tool misuse or execution loops. Our agent outperforms baselines (ROUGE: 0.783 vs. 0.703/0.608) with better cross-lingual stability (17-point vs. 21-26-point variation).

**arXiv ID:** 2509.11773
</details>

<details>
<summary><strong>Precise and Efficient Collision Prediction under Uncertainty in Autonomous Driving</strong> - Marc Kaufeld, Johannes Betz - [[pdf]](https://arxiv.org/pdf/2510.05729)</summary>

**Abstract:** This research introduces two efficient methods to estimate the collision risk of planned trajectories in autonomous driving under uncertain driving conditions. Deterministic collision checks of planned trajectories are often inaccurate or overly conservative, as noisy perception, localization errors, and uncertain predictions of other traffic participants introduce significant uncertainty into the planning process. This paper presents two semi-analytic methods to compute the collision probability of planned trajectories with arbitrary convex obstacles. The first approach evaluates the probability of spatial overlap between an autonomous vehicle and surrounding obstacles, while the second estimates the collision probability based on stochastic boundary crossings. Both formulations incorporate full state uncertainties, including position, orientation, and velocity, and achieve high accuracy at computational costs suitable for real-time planning. Simulation studies verify that the proposed methods closely match Monte Carlo results while providing significant runtime advantages, enabling their use in risk-aware trajectory planning. The collision estimation methods are available as open-source software: this https URL

**arXiv ID:** 2510.05729
</details>

<details>
<summary><strong>IMPACT: Intelligent Motion Planning with Acceptable Contact Trajectories via Vision-Language Models</strong> - Yiyang Ling, Karan Owalekar, Oluwatobiloba Adesanya, Erdem Bıyık, Daniel Seita - [[pdf]](https://arxiv.org/pdf/2503.10110)</summary>

**Abstract:** Motion planning involves determining a sequence of robot configurations to reach a desired pose, subject to movement and safety constraints. Traditional motion planning finds collision-free paths, but this is overly restrictive in clutter, where it may not be possible for a robot to accomplish a task without contact. In addition, contacts range from relatively benign (e.g. brushing a soft pillow) to more dangerous (e.g. toppling a glass vase), making it difficult to characterize which may be acceptable. In this paper, we propose IMPACT, a novel motion planning framework that uses Vision-Language Models (VLMs) to infer environment semantics, identifying which parts of the environment can best tolerate contact based on object properties and locations. Our approach generates an anisotropic cost map that encodes directional push safety. We pair this map with a contact-aware A* planner to find stable contact-rich paths. We perform experiments using 20 simulation and 10 real-world scenes and assess using task success rate, object displacements, and feedback from human evaluators. Our results over 3200 simulation and 200 real-world trials suggest that IMPACT enables efficient contact-rich motion planning in cluttered settings while outperforming alternative methods and ablations. Our project website is available at this https URL.

**arXiv ID:** 2503.10110
</details>

<details>
<summary><strong>When Should Users Check? A Decision-Theoretic Model of Confirmation Frequency in Multi-Step AI Agent Tasks</strong> - Jieyu Zhou, Aryan Roy, Sneh Gupta, Daniel Weitekamp, Christopher J. MacLellan - [[pdf]](https://arxiv.org/pdf/2510.05307)</summary>

**Abstract:** Existing AI agents typically execute multi-step tasks autonomously and only allow user confirmation at the end. During execution, users have little control, making the confirm-at-end approach brittle: a single error can cascade and force a complete restart. Confirming every step avoids such failures, but imposes tedious overhead. Balancing excessive interruptions against costly rollbacks remains an open challenge. We address this problem by modeling confirmation as a minimum time scheduling problem. We conducted a formative study with eight participants, which revealed a recurring Confirmation-Diagnosis-Correction-Redo (CDCR) pattern in how users monitor errors. Based on this pattern, we developed a decision-theoretic model to determine time-efficient confirmation point placement. We then evaluated our approach using a within-subjects study where 48 participants monitored AI agents and repaired their mistakes while executing tasks. Results show that 81 percent of participants preferred our intermediate confirmation approach over the confirm-at-end approach used by existing systems, and task completion time was reduced by 13.54 percent.

**arXiv ID:** 2510.05307
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (21 papers)</h2></summary>

<details>
<summary><strong>In-the-Flow Agentic System Optimization for Effective Planning and Tool Use</strong> - Zhuofeng Li, Haoxiang Zhang, Seungju Han, Sheng Liu, Jianwen Xie, Yu Zhang, Yejin Choi, James Zou, Pan Lu - [[pdf]](https://arxiv.org/pdf/2510.05592)</summary>

**Abstract:** Outcome-driven reinforcement learning has advanced reasoning in large language models (LLMs), but prevailing tool-augmented approaches train a single, monolithic policy that interleaves thoughts and tool calls under full context; this scales poorly with long horizons and diverse tools and generalizes weakly to new scenarios. Agentic systems offer a promising alternative by decomposing work across specialized modules, yet most remain training-free or rely on offline training decoupled from the live dynamics of multi-turn interaction. We introduce AgentFlow, a trainable, in-the-flow agentic framework that coordinates four modules (planner, executor, verifier, generator) through an evolving memory and directly optimizes its planner inside the multi-turn loop. To train on-policy in live environments, we propose Flow-based Group Refined Policy Optimization (Flow-GRPO), which tackles long-horizon, sparse-reward credit assignment by converting multi-turn optimization into a sequence of tractable single-turn policy updates. It broadcasts a single, verifiable trajectory-level outcome to every turn to align local planner decisions with global success and stabilizes learning with group-normalized advantages. Across ten benchmarks, AgentFlow with a 7B-scale backbone outperforms top-performing baselines with average accuracy gains of 14.9% on search, 14.0% on agentic, 14.5% on mathematical, and 4.1% on scientific tasks, even surpassing larger proprietary models like GPT-4o. Further analyses confirm the benefits of in-the-flow optimization, showing improved planning, enhanced tool-calling reliability, and positive scaling with model size and reasoning turns.

**arXiv ID:** 2510.05592
</details>

<details>
<summary><strong>Adversarial Reinforcement Learning for Offensive and Defensive Agents in a Simulated Zero-Sum Network Environment</strong> - Abrar Shahid, Ibteeker Mahir Ishum, AKM Tahmidul Haque, M Sohel Rahman, A. B. M. Alim Al Islam - [[pdf]](https://arxiv.org/pdf/2510.05157)</summary>

**Abstract:** This paper presents a controlled study of adversarial reinforcement learning in network security through a custom OpenAI Gym environment that models brute-force attacks and reactive defenses on multi-port services. The environment captures realistic security trade-offs including background traffic noise, progressive exploitation mechanics, IP-based evasion tactics, honeypot traps, and multi-level rate-limiting defenses. Competing attacker and defender agents are trained using Deep Q-Networks (DQN) within a zero-sum reward framework, where successful exploits yield large terminal rewards while incremental actions incur small costs. Through systematic evaluation across multiple configurations (varying trap detection probabilities, exploitation difficulty thresholds, and training regimens), the results demonstrate that defender observability and trap effectiveness create substantial barriers to successful attacks. The experiments reveal that reward shaping and careful training scheduling are critical for learning stability in this adversarial setting. The defender consistently maintains strategic advantage across 50,000+ training episodes, with performance gains amplifying when exposed to complex defensive strategies including adaptive IP blocking and port-specific controls. Complete implementation details, reproducible hyperparameter configurations, and architectural guidelines are provided to support future research in adversarial RL for cybersecurity. The zero-sum formulation and realistic operational constraints make this environment suitable for studying autonomous defense systems, attacker-defender co-evolution, and transfer learning to real-world network security scenarios.

**arXiv ID:** 2510.05157
</details>

<details>
<summary><strong>CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension</strong> - Rui Li, Zeyu Zhang, Xiaohe Bo, Zihang Tian, Xu Chen, Quanyu Dai, Zhenhua Dong, Ruiming Tang - [[pdf]](https://arxiv.org/pdf/2510.05520)</summary>

**Abstract:** Current Large Language Models (LLMs) are confronted with overwhelming information volume when comprehending long-form documents. This challenge raises the imperative of a cohesive memory module, which can elevate vanilla LLMs into autonomous reading agents. Despite the emergence of some heuristic approaches, a systematic design principle remains absent. To fill this void, we draw inspiration from Jean Piaget's Constructivist Theory, illuminating three traits of the agentic memory -- structured schemata, flexible assimilation, and dynamic accommodation. This blueprint forges a clear path toward a more robust and efficient memory system for LLM-based reading comprehension. To this end, we develop CAM, a prototype implementation of Constructivist Agentic Memory that simultaneously embodies the structurality, flexibility, and dynamicity. At its core, CAM is endowed with an incremental overlapping clustering algorithm for structured memory development, supporting both coherent hierarchical summarization and online batch integration. During inference, CAM adaptively explores the memory structure to activate query-relevant information for contextual response, akin to the human associative process. Compared to existing approaches, our design demonstrates dual advantages in both performance and efficiency across diverse long-text reading comprehension tasks, including question answering, query-based summarization, and claim verification.

**arXiv ID:** 2510.05520
</details>

<details>
<summary><strong>AgentDR Dynamic Recommendation with Implicit Item-Item Relations via LLM-based Agents</strong> - Mingdai Yang, Nurendra Choudhary, Jiangshu Du, Edward W.Huang, Philip S.Yu, Karthik Subbian, Danai Kourta - [[pdf]](https://arxiv.org/pdf/2510.05598)</summary>

**Abstract:** Recent agent-based recommendation frameworks aim to simulate user behaviors by incorporating memory mechanisms and prompting strategies, but they struggle with hallucinating non-existent items and full-catalog ranking. Besides, a largely underexplored opportunity lies in leveraging LLMs'commonsense reasoning to capture user intent through substitute and complement relationships between items, which are usually implicit in datasets and difficult for traditional ID-based recommenders to capture. In this work, we propose a novel LLM-agent framework, AgenDR, which bridges LLM reasoning with scalable recommendation tools. Our approach delegates full-ranking tasks to traditional models while utilizing LLMs to (i) integrate multiple recommendation outputs based on personalized tool suitability and (ii) reason over substitute and complement relationships grounded in user history. This design mitigates hallucination, scales to large catalogs, and enhances recommendation relevance through relational reasoning. Through extensive experiments on three public grocery datasets, we show that our framework achieves superior full-ranking performance, yielding on average a twofold improvement over its underlying tools. We also introduce a new LLM-based evaluation metric that jointly measures semantic alignment and ranking correctness.

**arXiv ID:** 2510.05598
</details>

<details>
<summary><strong>From Learning to Mastery: Achieving Safe and Efficient Real-World Autonomous Driving with Human-In-The-Loop Reinforcement Learning</strong> - Li Zeqiao, Wang Yijing, Wang Haoyu, Li Zheng, Li Peng, Liu Wenfei, Zuo Zhiqiang - [[pdf]](https://arxiv.org/pdf/2510.06038)</summary>

**Abstract:** Autonomous driving with reinforcement learning (RL) has significant potential. However, applying RL in real-world settings remains challenging due to the need for safe, efficient, and robust learning. Incorporating human expertise into the learning process can help overcome these challenges by reducing risky exploration and improving sample efficiency. In this work, we propose a reward-free, active human-in-the-loop learning method called Human-Guided Distributional Soft Actor-Critic (H-DSAC). Our method combines Proxy Value Propagation (PVP) and Distributional Soft Actor-Critic (DSAC) to enable efficient and safe training in real-world environments. The key innovation is the construction of a distributed proxy value function within the DSAC framework. This function encodes human intent by assigning higher expected returns to expert demonstrations and penalizing actions that require human intervention. By extrapolating these labels to unlabeled states, the policy is effectively guided toward expert-like behavior. With a well-designed state space, our method achieves real-world driving policy learning within practical training times. Results from both simulation and real-world experiments demonstrate that our framework enables safe, robust, and sample-efficient learning for autonomous driving.

**arXiv ID:** 2510.06038
</details>

<details>
<summary><strong>Multi-Task Reinforcement Learning with Language-Encoded Gated Policy Networks</strong> - Rushiv Arora - [[pdf]](https://arxiv.org/pdf/2510.06138)</summary>

**Abstract:** Multi-task reinforcement learning often relies on task metadata -- such as brief natural-language descriptions -- to guide behavior across diverse objectives. We present Lexical Policy Networks (LEXPOL), a language-conditioned mixture-of-policies architecture for multi-task RL. LEXPOL encodes task metadata with a text encoder and uses a learned gating module to select or blend among multiple sub-policies, enabling end-to-end training across tasks. On MetaWorld benchmarks, LEXPOL matches or exceeds strong multi-task baselines in success rate and sample efficiency, without task-specific retraining. To analyze the mechanism, we further study settings with fixed expert policies obtained independently of the gate and show that the learned language gate composes these experts to produce behaviors appropriate to novel task descriptions and unseen task combinations. These results indicate that natural-language metadata can effectively index and recombine reusable skills within a single policy.

**arXiv ID:** 2510.06138
</details>

<details>
<summary><strong>Stratified GRPO: Handling Structural Heterogeneity in Reinforcement Learning of LLM Search Agents</strong> - Mingkang Zhu, Xi Chen, Bei Yu, Hengshuang Zhao, Jiaya Jia - [[pdf]](https://arxiv.org/pdf/2510.06214)</summary>

**Abstract:** Large language model (LLM) agents increasingly rely on external tools such as search engines to solve complex, multi-step problems, and reinforcement learning (RL) has become a key paradigm for training them. However, the trajectories of search agents are structurally heterogeneous, where variations in the number, placement, and outcomes of search calls lead to fundamentally different answer directions and reward distributions. Standard policy gradient methods, which use a single global baseline, suffer from what we identify and formalize as cross-stratum bias-an "apples-to-oranges" comparison of heterogeneous trajectories. This cross-stratum bias distorts credit assignment and hinders exploration of complex, multi-step search strategies. To address this, we propose Stratified GRPO, whose central component, Stratified Advantage Normalization (SAN), partitions trajectories into homogeneous strata based on their structural properties and computes advantages locally within each stratum. This ensures that trajectories are evaluated only against their true peers. Our analysis proves that SAN eliminates cross-stratum bias, yields conditionally unbiased unit-variance estimates inside each stratum, and retains the global unbiasedness and unit-variance properties enjoyed by standard normalization, resulting in a more pure and scale-stable learning signal. To improve practical stability under finite-sample regimes, we further linearly blend SAN with the global estimator. Extensive experiments on diverse single-hop and multi-hop question-answering benchmarks demonstrate that Stratified GRPO consistently and substantially outperforms GRPO by up to 11.3 points, achieving higher training rewards, greater training stability, and more effective search policies. These results establish stratification as a principled remedy for structural heterogeneity in RL for LLM search agents.

**arXiv ID:** 2510.06214
</details>

<details>
<summary><strong>Open Agent Specification (Agent Spec) Technical Report</strong> - Yassine Benajiba, Cesare Bernardis, Vladislav Blinov, Paul Cayet, Hassan Chafi, Abderrahim Fathan, Louis Faucon, Damien Hilloulin, Sungpack Hong, Ingo Kossyk, Rhicheek Patra, Sujith Ravi, Jonas Schweizer, Jyotika Singh, Shailender Singh, Xuelin Situ, Weiyi Sun, Jerry Xu, Ying Xu - [[pdf]](https://arxiv.org/pdf/2510.04173)</summary>

**Abstract:** Open Agent Specification (Agent Spec) is a declarative language that allows AI agents and their workflows to be defined in a way that is compatible across different AI frameworks, promoting portability and interoperability within AI Agent frameworks.
Agent Spec aims to resolve the challenges of fragmented agent development by providing a common unified specification that allows AI agents to be designed once and deployed across various frameworks, improving interoperability and reusability, and reducing redundant development efforts. Additionally, Agent Spec facilitates development tools and portability, allowing AI agents to be defined independently of their execution environment and enabling teams to exchange solutions without implementation-specific limitations.
Agent Spec benefits four key groups: (i) Agent developers, who gain access to a superset of reusable components and design patterns, enabling them to leverage a broader range of functionalities; (ii) Agent framework and tool developers, who can use Agent Spec as an interchange format and therefore benefit from the support of other frameworks as well as other tools; (iii) Researchers, who can achieve reproducible results and comparability, facilitating more reliable and consistent outcomes; (iv) Enterprises, which benefit from faster prototype-to-deployment, increased productivity, as well as greater scalability and maintainability for their AI agent solutions. This technical report provides an overview of the technical foundations of Agent Spec, including motivation, benefits, and future developments.

**arXiv ID:** 2510.04173
</details>

<details>
<summary><strong>Let it Calm: Exploratory Annealed Decoding for Verifiable Reinforcement Learning</strong> - Chenghao Yang, Lin Gui, Chenxiao Yang, Victor Veitch, Lizhu Zhang, Zhuokai Zhao - [[pdf]](https://arxiv.org/pdf/2510.05251)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) is a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs), yet its success hinges on effective exploration. An ideal exploration strategy must navigate two fundamental challenges: it must preserve sample quality while also ensuring training stability. While standard fixed-temperature sampling is simple, it struggles to balance these competing demands, as high temperatures degrade sample quality and low temperatures limit discovery. In this work, we propose a simpler and more effective strategy, Exploratory Annealed Decoding (EAD), grounded in the insight that exploration is most impactful on early tokens which define a sequence's semantic direction. EAD implements an intuitive **explore-at-the-beginning, exploit-at-the-end** strategy by annealing the sampling temperature from high to low during generation. This dynamic schedule encourages meaningful, high-level diversity at the start, then gradually lowers the temperature to preserve sample quality and keep the sampling distribution close to the target policy, which is essential for stable training. We demonstrate that EAD is a lightweight, plug-and-play method that significantly improves sample efficiency, consistently outperforming fixed-temperature sampling across various RLVR algorithms and model sizes. Our work suggests that aligning exploration with the natural dynamics of sequential generation offers a robust path to improving LLM reasoning.

**arXiv ID:** 2510.05251
</details>

<details>
<summary><strong>A Goal Without a Plan Is Just a Wish: Efficient and Effective Global Planner Training for Long-Horizon Agent Tasks</strong> - Shuzheng Si, Haozhe Zhao, Kangyang Luo, Gang Chen, Fanchao Qi, Minjia Zhang, Baobao Chang, Maosong Sun - [[pdf]](https://arxiv.org/pdf/2510.05608)</summary>

**Abstract:** Agents based on large language models (LLMs) struggle with brainless trial-and-error and generating hallucinatory actions due to a lack of global planning in long-horizon tasks. In this paper, we introduce a plan-and-execute framework and propose EAGLET, an efficient and effective planner training method to enhance the executor agent's planning abilities without human effort. Specifically, we train a plug-and-play global planner through a two-step process: we first synthesize high-quality plans from an advanced LLM using our proposed homologous consensus filtering strategy, and apply fine-tuning as a cold start. Moreover, we further improve the planner with a rule-based reinforcement learning stage using a novel executor capability gain reward, ensuring it can handle task instructions of varying difficulty. Experiments on three long-horizon agent tasks show that executor agents equipped with our planner outperform existing methods, achieving new state-of-the-art performance. Meanwhile, EAGLET reduces training costs by 8x compared to RL-based baselines, and it does not require manual effort or extra training data, offering an efficient and effective solution.

**arXiv ID:** 2510.05608
</details>

<details>
<summary><strong>DecEx-RAG: Boosting Agentic Retrieval-Augmented Generation with Decision and Execution Optimization via Process Supervision</strong> - Yongqi Leng, Yikun Lei, Xikai Liu, Meizhi Zhong, Bojian Xiong, Yurong Zhang, Yan Gao, Yi Wu, Yao Hu, Deyi Xiong - [[pdf]](https://arxiv.org/pdf/2510.05691)</summary>

**Abstract:** Agentic Retrieval-Augmented Generation (Agentic RAG) enhances the processing capability for complex tasks through dynamic retrieval and adaptive workflows. Recent advances (e.g., Search-R1) have shown that outcome-supervised reinforcement learning demonstrate strong performance. However, this approach still suffers from inefficient exploration, sparse reward signals, and ambiguous global reward feedback. To address these challenges, we propose DecEx-RAG, which models RAG as a Markov Decision Process (MDP) incorporating decision-making and execution, while introducing an efficient pruning strategy to optimize data expansion. Through comprehensive process-level policy optimization, DecEx-RAG significantly enhances the autonomous task decomposition, dynamic retrieval, and high-quality answer generation capabilities of large language models (LLMs). Experiments show that DecEx-RAG achieves an average absolute performance improvement of $6.2\%$ across six datasets, significantly outperforming existing baselines. Moreover, the pruning strategy improves data construction efficiency by nearly $6 \times$, providing an efficient solution for process-supervised RAG training. The code is available at this https URL.

**arXiv ID:** 2510.05691
</details>

<details>
<summary><strong>Peeking inside the Black-Box: Reinforcement Learning for Explainable and Accurate Relation Extraction</strong> - Xinyu Guo, Zhengliang Shi, Minglai Yang, Mahdi Rahimi, Mihai Surdeanu - [[pdf]](https://arxiv.org/pdf/2510.06198)</summary>

**Abstract:** This paper introduces a framework for relation extraction (RE) that enhances both accuracy and explainability. The framework has two key components: (i) a reasoning mechanism that formulates relation extraction as a series of text-processing steps inspired by cognitive science, and (ii) an optimization process driven by reinforcement learning (RL) with a novel reward function designed to improve both task accuracy and explanation quality. We call our approach CogRE. Our framework addresses the lack of supervision for language-based explanations in traditional RE by promoting outputs that include important relation keywords. These keywords are drawn from a high-quality dictionary that is automatically constructed using an LLM. We evaluate our approach for the task of one-shot RE using two LLMs and two RE datasets. Our experiments show that CogRE improves explanation quality by addressing two common failure patterns in one-shot RE: poor attention focus and limited one-shot learning capability. For example, our cognitive-structured reasoning with Qwen2.5-15B-Instruct on One-shot NYT29 achieves 24.65% F1, surpassing prior reasoning-based designs. Optimizing this approach with RL using our reward further improves performance by +23.46% (absolute). Finally, human evaluation shows that our best model generates relational keywords closely aligned with gold labels, increasing human explanation quality ratings by 54% (relative).

**arXiv ID:** 2510.06198
</details>

<details>
<summary><strong>Adaptive Reinforcement Learning for Dynamic Configuration Allocation in Pre-Production Testing</strong> - Yu Zhu - [[pdf]](https://arxiv.org/pdf/2510.05147)</summary>

**Abstract:** Ensuring reliability in modern software systems requires rigorous pre-production testing across highly heterogeneous and evolving environments. Because exhaustive evaluation is infeasible, practitioners must decide how to allocate limited testing resources across configurations where failure probabilities may drift over time. Existing combinatorial optimization approaches are static, ad hoc, and poorly suited to such non-stationary settings. We introduce a novel reinforcement learning (RL) framework that recasts configuration allocation as a sequential decision-making problem. Our method is the first to integrate Q-learning with a hybrid reward design that fuses simulated outcomes and real-time feedback, enabling both sample efficiency and robustness. In addition, we develop an adaptive online-offline training scheme that allows the agent to quickly track abrupt probability shifts while maintaining long-run stability. Extensive simulation studies demonstrate that our approach consistently outperforms static and optimization-based baselines, approaching oracle performance. This work establishes RL as a powerful new paradigm for adaptive configuration allocation, advancing beyond traditional methods and offering broad applicability to dynamic testing and resource scheduling domains.

**arXiv ID:** 2510.05147
</details>

<details>
<summary><strong>Oracle-Guided Masked Contrastive Reinforcement Learning for Visuomotor Policies</strong> - Yuhang Zhang, Jiaping Xiao, Chao Yan, Mir Feroskhan - [[pdf]](https://arxiv.org/pdf/2510.05692)</summary>

**Abstract:** A prevailing approach for learning visuomotor policies is to employ reinforcement learning to map high-dimensional visual observations directly to action commands. However, the combination of high-dimensional visual inputs and agile maneuver outputs leads to long-standing challenges, including low sample efficiency and significant sim-to-real gaps. To address these issues, we propose Oracle-Guided Masked Contrastive Reinforcement Learning (OMC-RL), a novel framework designed to improve the sample efficiency and asymptotic performance of visuomotor policy learning. OMC-RL explicitly decouples the learning process into two stages: an upstream representation learning stage and a downstream policy learning stage. In the upstream stage, a masked Transformer module is trained with temporal modeling and contrastive learning to extract temporally-aware and task-relevant representations from sequential visual inputs. After training, the learned encoder is frozen and used to extract visual representations from consecutive frames, while the Transformer module is discarded. In the downstream stage, an oracle teacher policy with privileged access to global state information supervises the agent during early training to provide informative guidance and accelerate early policy learning. This guidance is gradually reduced to allow independent exploration as training progresses. Extensive experiments in simulated and real-world environments demonstrate that OMC-RL achieves superior sample efficiency and asymptotic policy performance, while also improving generalization across diverse and perceptually complex scenarios.

**arXiv ID:** 2510.05692
</details>

<details>
<summary><strong>EARL: Efficient Agentic Reinforcement Learning Systems for Large Language Models</strong> - Zheyue Tan, Mustapha Abdullahi, Tuo Shi, Huining Yuan, Zelai Xu, Chao Yu, Boxun Li, Bo Zhao - [[pdf]](https://arxiv.org/pdf/2510.05943)</summary>

**Abstract:** Reinforcement learning (RL) has become a pivotal component of large language model (LLM) post-training, and agentic RL extends this paradigm to operate as agents through multi-turn interaction and tool use. Scaling such systems exposes two practical bottlenecks: (1) context length grows rapidly during training, inflating memory usage and latency, and triggering out-of-memory (OOM) failures; and (2) intermediate tensors accumulate with context length, making cross-device data movement a major system bottleneck.
We present EARL, a scalable system for efficient agentic RL. EARL designs a parallelism selector that dynamically adapts model and training parallelism across RL stages based on sequence length and system load, and a data dispatcher that performs layout-aware, decentralized exchange of intermediate data batches. Together, these components increase throughput, reduce long-context failures, and enable stable large-scale training of agentic LLMs without relying on hard limits or penalties of context length.

**arXiv ID:** 2510.05943
</details>

<details>
<summary><strong>Learning to Crawl: Latent Model-Based Reinforcement Learning for Soft Robotic Adaptive Locomotion</strong> - Vaughn Gzenda, Robin Chhabra - [[pdf]](https://arxiv.org/pdf/2510.05957)</summary>

**Abstract:** Soft robotic crawlers are mobile robots that utilize soft body deformability and compliance to achieve locomotion through surface contact. Designing control strategies for such systems is challenging due to model inaccuracies, sensor noise, and the need to discover locomotor gaits. In this work, we present a model-based reinforcement learning (MB-RL) framework in which latent dynamics inferred from onboard sensors serve as a predictive model that guides an actor-critic algorithm to optimize locomotor policies. We evaluate the framework on a minimal crawler model in simulation using inertial measurement units and time-of-flight sensors as observations. The learned latent dynamics enable short-horizon motion prediction while the actor-critic discovers effective locomotor policies. This approach highlights the potential of latent-dynamics MB-RL for enabling embodied soft robotic adaptive locomotion based solely on noisy sensor feedback.

**arXiv ID:** 2510.05957
</details>

<details>
<summary><strong>Towards Autonomous Tape Handling for Robotic Wound Redressing</strong> - Xiao Liang, Lu Shen, Peihan Zhang, Soofiyan Atar, Florian Richter, Michael Yip - [[pdf]](https://arxiv.org/pdf/2510.06127)</summary>

**Abstract:** Chronic wounds, such as diabetic, pressure, and venous ulcers, affect over 6.5 million patients in the United States alone and generate an annual cost exceeding \$25 billion. Despite this burden, chronic wound care remains a routine yet manual process performed exclusively by trained clinicians due to its critical safety demands. We envision a future in which robotics and automation support wound care to lower costs and enhance patient outcomes. This paper introduces an autonomous framework for one of the most fundamental yet challenging subtasks in wound redressing: adhesive tape manipulation. Specifically, we address two critical capabilities: tape initial detachment (TID) and secure tape placement. To handle the complex adhesive dynamics of detachment, we propose a force-feedback imitation learning approach trained from human teleoperation demonstrations. For tape placement, we develop a numerical trajectory optimization method based to ensure smooth adhesion and wrinkle-free application across diverse anatomical surfaces. We validate these methods through extensive experiments, demonstrating reliable performance in both quantitative evaluations and integrated wound redressing pipelines. Our results establish tape manipulation as an essential step toward practical robotic wound care automation.

**arXiv ID:** 2510.06127
</details>

<details>
<summary><strong>RoboMemory: A Brain-inspired Multi-memory Agentic Framework for Interactive Environmental Learning in Physical Embodied Systems</strong> - Mingcong Lei, Honghao Cai, Zezhou Cui, Liangchen Tan, Junkun Hong, Gehan Hu, Shuangyu Zhu, Yimou Wu, Shaohan Jiang, Ge Wang, Yuyuan Yang, Junyuan Tan, Zhenglin Wan, Zhen Li, Shuguang Cui, Yiming Zhao, Yatong Han - [[pdf]](https://arxiv.org/pdf/2508.01415)</summary>

**Abstract:** Embodied agents face persistent challenges in real-world environments, including partial observability, limited spatial reasoning, and high-latency multi-memory integration. We present RoboMemory, a brain-inspired framework that unifies Spatial, Temporal, Episodic, and Semantic memory under a parallelized architecture for efficient long-horizon planning and interactive environmental learning. A dynamic spatial knowledge graph (KG) ensures scalable and consistent memory updates, while a closed-loop planner with a critic module supports adaptive decision-making in dynamic settings. Experiments on EmbodiedBench show that RoboMemory, built on Qwen2.5-VL-72B-Ins, improves average success rates by 25% over its baseline and exceeds the closed-source state-of-the-art (SOTA) Gemini-1.5-Pro by 3%. Real-world trials further confirm its capacity for cumulative learning, with performance improving across repeated tasks. These results highlight RoboMemory as a scalable foundation for memory-augmented embodied intelligence, bridging the gap between cognitive neuroscience and robotic autonomy.

**arXiv ID:** 2508.01415
</details>

<details>
<summary><strong>TreeIRL: Safe Urban Driving with Tree Search and Inverse Reinforcement Learning</strong> - Momchil S. Tomov, Sang Uk Lee, Hansford Hendrago, Jinwook Huh, Teawon Han, Forbes Howington, Rafael da Silva, Gianmarco Bernasconi, Marc Heim, Samuel Findler, Xiaonan Ji, Alexander Boule, Michael Napoli, Kuo Chen, Jesse Miller, Boaz Floor, Yunqing Hu - [[pdf]](https://arxiv.org/pdf/2509.13579)</summary>

**Abstract:** We present TreeIRL, a novel planner for autonomous driving that combines Monte Carlo tree search (MCTS) and inverse reinforcement learning (IRL) to achieve state-of-the-art performance in simulation and in real-world driving. The core idea is to use MCTS to find a promising set of safe candidate trajectories and a deep IRL scoring function to select the most human-like among them. We evaluate TreeIRL against both classical and state-of-the-art planners in large-scale simulations and on 500+ miles of real-world autonomous driving in the Las Vegas metropolitan area. Test scenarios include dense urban traffic, adaptive cruise control, cut-ins, and traffic lights. TreeIRL achieves the best overall performance, striking a balance between safety, progress, comfort, and human-likeness. To our knowledge, our work is the first demonstration of MCTS-based planning on public roads and underscores the importance of evaluating planners across a diverse set of metrics and in real-world environments. TreeIRL is highly extensible and could be further improved with reinforcement learning and imitation learning, providing a framework for exploring different combinations of classical and learning-based approaches to solve the planning bottleneck in autonomous driving.

**arXiv ID:** 2509.13579
</details>

<details>
<summary><strong>CoTaP: Compliant Task Pipeline and Reinforcement Learning of Its Controller with Compliance Modulation</strong> - Zewen He, Chenyuan Chen, Dilshod Azizov, Yoshihiko Nakamura - [[pdf]](https://arxiv.org/pdf/2509.25443)</summary>

**Abstract:** Humanoid whole-body locomotion control is a critical approach for humanoid robots to leverage their inherent advantages. Learning-based control methods derived from retargeted human motion data provide an effective means of addressing this issue. However, because most current human datasets lack measured force data, and learning-based robot control is largely position-based, achieving appropriate compliance during interaction with real environments remains challenging. This paper presents Compliant Task Pipeline (CoTaP): a pipeline that leverages compliance information in the learning-based structure of humanoid robots. A two-stage dual-agent reinforcement learning framework combined with model-based compliance control for humanoid robots is proposed. In the training process, first a base policy with a position-based controller is trained; then in the distillation, the upper-body policy is combined with model-based compliance control, and the lower-body agent is guided by the base policy. In the upper-body control, adjustable task-space compliance can be specified and integrated with other controllers through compliance modulation on the symmetric positive definite (SPD) manifold, ensuring system stability. We validated the feasibility of the proposed strategy in simulation, primarily comparing the responses to external disturbances under different compliance settings.

**arXiv ID:** 2509.25443
</details>

<details>
<summary><strong>Chrysalis: A Unified System for Comparing Active Teaching and Passive Learning with AI Agents in Education</strong> - Prashanth Arun, Vinita Vader, Erya Xu, Brent McCready-Branch, Sarah Seabrook, Kyle Scholz, Ana Crisan, Igor Grossmann, Pascal Poupart - [[pdf]](https://arxiv.org/pdf/2510.05271)</summary>

**Abstract:** AI-assisted learning has seen a remarkable uptick over the last few years, mainly due to the rise in popularity of Large Language Models (LLMs). Their ability to hold long-form, natural language interactions with users makes them excellent resources for exploring school- and university-level topics in a dynamic, active manner. We compare students' experiences when interacting with an LLM companion in two capacities: tutored learning and learning-by-teaching. We do this using Chrysalis, an LLM-based system that we have designed to support both AI tutors and AI teachable agents for any topic. Through a within-subject exploratory study with 36 participants, we present insights into student preferences between the two strategies and how constructs such as intellectual humility vary between these two interaction modes. To our knowledge, we are the first to conduct a direct comparison study on the effects of using an LLM as a tutor versus as a teachable agent on multiple topics. We hope that our work opens up new avenues for future research in this area.

**arXiv ID:** 2510.05271
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
