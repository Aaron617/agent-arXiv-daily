# Agent arXiv Daily

**Last Updated:** 2025-12-11 03:01:19

**Total Papers:** 53

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
<summary><strong>SWEnergy: An Empirical Study on Energy Efficiency in Agentic Issue Resolution Frameworks with SLMs</strong> - Arihant Tripathy, Ch Pavan Harshit, Karthik Vaidhyanathan - [[pdf]](https://arxiv.org/pdf/2512.09543)</summary>

**Abstract:** Context. LLM-based autonomous agents in software engineering rely on large, proprietary models, limiting local deployment. This has spurred interest in Small Language Models (SLMs), but their practical effectiveness and efficiency within complex agentic frameworks for automated issue resolution remain poorly understood.
Goal. We investigate the performance, energy efficiency, and resource consumption of four leading agentic issue resolution frameworks when deliberately constrained to using SLMs. We aim to assess the viability of these systems for this task in resource-limited settings and characterize the resulting trade-offs.
Method. We conduct a controlled evaluation of four leading agentic frameworks (SWE-Agent, OpenHands, Mini SWE Agent, AutoCodeRover) using two SLMs (Gemma-3 4B, Qwen-3 1.7B) on the SWE-bench Verified Mini benchmark. On fixed hardware, we measure energy, duration, token usage, and memory over 150 runs per configuration.
Results. We find that framework architecture is the primary driver of energy consumption. The most energy-intensive framework, AutoCodeRover (Gemma), consumed 9.4x more energy on average than the least energy-intensive, OpenHands (Gemma). However, this energy is largely wasted. Task resolution rates were near-zero, demonstrating that current frameworks, when paired with SLMs, consume significant energy on unproductive reasoning loops. The SLM's limited reasoning was the bottleneck for success, but the framework's design was the bottleneck for efficiency.
Conclusions. Current agentic frameworks, designed for powerful LLMs, fail to operate efficiently with SLMs. We find that framework architecture is the primary driver of energy consumption, but this energy is largely wasted due to the SLMs' limited reasoning. Viable low-energy solutions require shifting from passive orchestration to architectures that actively manage SLM weaknesses.

**arXiv ID:** 2512.09543
</details>

<details>
<summary><strong>An Offline Mobile Conversational Agent for Mental Health Support: Learning from Emotional Dialogues and Psychological Texts with Student-Centered Evaluation</strong> - Vimaleswar A, Prabhu Nandan Sahu, Nilesh Kumar Sahu, Haroon R. Lone - [[pdf]](https://arxiv.org/pdf/2507.10580)</summary>

**Abstract:** Mental health plays a crucial role in the overall well-being of an individual. In recent years, digital platforms have increasingly been used to expand mental health and emotional support. However, there are persistent challenges related to limited user accessibility, internet connectivity, and data privacy, which highlight the need for an offline, smartphone-based solutions. To address these challenges, we propose EmoSApp (Emotional Support App): an entirely offline, smartphone-based conversational app designed to provide mental health and emotional support. EmoSApp leverages a language model, specifically the LLaMA-3.2-1B-Instruct, which is fine-tuned and quantized on a custom-curated ``Knowledge Dataset'' comprising 14,582 mental health QA pairs along with multi-turn conversational data, enabling robust domain expertise and fully on-device inference on resource-constrained smartphones.
Through qualitative evaluation with students and mental health professionals, we demonstrate that EmoSApp has the ability to respond coherently and empathetically, provide relevant suggestions to user's mental health problems, and maintain interactive dialogue. Additionally, quantitative evaluations on nine commonsense and reasoning benchmarks, along with two mental health specific datasets, demonstrate EmoSApp's effectiveness in low-resource settings. By prioritizing on-device deployment and specialized domain-specific adaptation, EmoSApp serves as a blueprint for future innovations in portable, secure, and highly tailored AI-driven mental health support.

**arXiv ID:** 2507.10580
</details>

<details>
<summary><strong>Controlling Steering Angle for Cooperative Self-driving Vehicles utilizing CNN and LSTM-based Deep Networks</strong> - Rodolfo Valiente, Mahdi Zaman, Sedat Ozer, Yaser P. Fallah - [[pdf]](https://arxiv.org/pdf/1904.04375)</summary>

**Abstract:** A fundamental challenge in autonomous vehicles is adjusting the steering angle at different road conditions. Recent state-of-the-art solutions addressing this challenge include deep learning techniques as they provide end-to-end solution to predict steering angles directly from the raw input images with higher accuracy. Most of these works ignore the temporal dependencies between the image frames. In this paper, we tackle the problem of utilizing multiple sets of images shared between two autonomous vehicles to improve the accuracy of controlling the steering angle by considering the temporal dependencies between the image frames. This problem has not been studied in the literature widely. We present and study a new deep architecture to predict the steering angle automatically by using Long-Short-Term-Memory (LSTM) in our deep architecture. Our deep architecture is an end-to-end network that utilizes CNN, LSTM and fully connected (FC) layers and it uses both present and futures images (shared by a vehicle ahead via Vehicle-to-Vehicle (V2V) communication) as input to control the steering angle. Our model demonstrates the lowest error when compared to the other existing approaches in the literature.

**arXiv ID:** 1904.04375
</details>

<details>
<summary><strong>PoultryTalk: A Multi-modal Retrieval-Augmented Generation (RAG) System for Intelligent Poultry Management and Decision Support</strong> - Kapalik Khanal, Biswash Khatiwada, Stephen Afrifa, Ranjan Sapkota, Sanjay Shah, Frank Bai, Ramesh Bahadur Bist - [[pdf]](https://arxiv.org/pdf/2512.08995)</summary>

**Abstract:** The Poultry industry plays a vital role in global food security, yet small- and medium-scale farmers frequently lack timely access to expert-level support for disease diagnosis, nutrition planning, and management decisions. With rising climate stress, unpredictable feed prices, and persistent disease threats, poultry producers often struggle to make quick, informed decisions. Therefore, there is a critical need for intelligent, data-driven systems that can deliver reliable, on-demand consultation. This paper presents PoultryTalk, a novel multi-modal Retrieval-Augmented Generation (RAG) system designed to provide real-time expert guidance through text and image-based interaction. PoultryTalk uses OpenAI's text-embedding-3-small and GPT-4o to provide smart, context-aware poultry management advice from text, images, or questions. System usability and performance were evaluated using 200 expert-verified queries and feedback from 34 participants who submitted 267 queries to the PoultryTalk prototype. The expert-verified benchmark queries confirmed strong technical performance, achieving a semantic similarity of 84.0% and an average response latency of 3.6 seconds. Compared with OpenAI's GPT-4o, PoultryTalk delivered more accurate and reliable information related to poultry. Based on participants' evaluations, PoultryTalk achieved a response accuracy of 89.9%, with about 9.1% of responses rated as incorrect. A post-use survey indicated high user satisfaction: 95.6% of participants reported that the chatbot provided "always correct" and "mostly correct" answers. 82.6% indicated they would recommend the tool, and 17.4% responded "maybe." These results collectively demonstrate that PoultryTalk not only delivers accurate, contextually relevant information but also demonstrates strong user acceptance and scalability potential.

**arXiv ID:** 2512.08995
</details>

<details>
<summary><strong>Exploring Community-Powered Conversational Agent for Health Knowledge Acquisition: A Case Study in Colorectal Cancer</strong> - Yiwei Yuan, Zhiqing Wang, Xiucheng Zhang, Yichao Luo, Shuya Lin, Yang Bai, Zhenhui Peng - [[pdf]](https://arxiv.org/pdf/2512.09511)</summary>

**Abstract:** Online communities have become key platforms where young adults, actively seek and share information, including health knowledge. However, these users often face challenges when browsing these communities, such as fragmented content, varying information quality and unfamiliar terminology. Based on a survey with 56 participants and follow-up interviews, we identify common challenges and expected features for learning health knowledge. In this paper, we develop a computational workflow that integrates community content into a conversational agent named CanAnswer to facilitate health knowledge acquisition. Using colorectal cancer as a case study, we evaluate CanAnswer through a lab study with 24 participants and interviews with six medical experts. Results show that CanAnswer improves the recalled gained knowledge and reduces the task workload of the learning session. Our expert interviews (N=6) further confirm the reliability and usefulness of CanAnswer. We discuss the generality of CanAnswer and provide design considerations for enhancing the usefulness and credibility of community-powered learning tools.

**arXiv ID:** 2512.09511
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (1 papers)</h2></summary>

<details>
<summary><strong>BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving</strong> - Shu Liu, Wenlin Chen, Weihao Li, Zheng Wang, Lijin Yang, Jianing Huang, Yipin Zhang, Zhongzhan Huang, Ze Cheng, Hao Yang - [[pdf]](https://arxiv.org/pdf/2509.23589)</summary>

**Abstract:** Diffusion-based planners have shown great promise for autonomous driving due to their ability to capture multi-modal driving behaviors. However, guiding these models effectively in reactive, closed-loop environments remains a significant challenge. Simple conditioning often fails to provide sufficient guidance in complex and dynamic driving scenarios. Recent work attempts to use typical expert driving behaviors (i.e., anchors) to guide diffusion models but relies on a truncated schedule, which introduces theoretical inconsistencies and can compromise performance. To address this, we introduce BridgeDrive, a novel anchor-guided diffusion bridge policy for closed-loop trajectory planning. Our approach provides a principled diffusion framework that effectively translates anchors into fine-grained trajectory plans, appropriately responding to varying traffic conditions. Our planner is compatible with efficient ODE solvers, a critical factor for real-time autonomous driving deployment. We achieve state-of-the-art performance on the Bench2Drive benchmark, improving the success rate by 7.72% over prior arts.

**arXiv ID:** 2509.23589
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>Evolving Excellence: Automated Optimization of LLM-based Agents</strong> - Paul Brookes, Vardan Voskanyan, Rafail Giavrimis, Matthew Truscott, Mina Ilieva, Chrystalla Pavlou, Alexandru Staicu, Manal Adham, Will Evers- Hood, Jingzhi Gong, Kejia Zhang, Matvey Fedoseev, Vishal Sharma, Roman Bauer, Zheng Wang, Hema Nair, Wei Jie, Tianhua Xu, Aurora Constantin, Leslie Kanthan, Michail Basios - [[pdf]](https://arxiv.org/pdf/2512.09108)</summary>

**Abstract:** Agentic AI systems built on large language models (LLMs) offer significant potential for automating complex workflows, from software development to customer support. However, LLM agents often underperform due to suboptimal configurations; poorly tuned prompts, tool descriptions, and parameters that typically require weeks of manual refinement. Existing optimization methods either are too complex for general use or treat components in isolation, missing critical interdependencies.
We present ARTEMIS, a no-code evolutionary optimization platform that jointly optimizes agent configurations through semantically-aware genetic operators. Given only a benchmark script and natural language goals, ARTEMIS automatically discovers configurable components, extracts performance signals from execution logs, and evolves configurations without requiring architectural modifications.
We evaluate ARTEMIS on four representative agent systems: the \emph{ALE Agent} for competitive programming on AtCoder Heuristic Contest, achieving a \textbf{$13.6\%$ improvement} in acceptance rate; the \emph{Mini-SWE Agent} for code optimization on SWE-Perf, with a statistically significant \textbf{10.1\% performance gain}; and the \emph{CrewAI Agent} for cost and mathematical reasoning on Math Odyssey, achieving a statistically significant \textbf{$36.9\%$ reduction} in the number of tokens required for evaluation. We also evaluate the \emph{MathTales-Teacher Agent} powered by a smaller open-source model (Qwen2.5-7B) on GSM8K primary-level mathematics problems, achieving a \textbf{22\% accuracy improvement} and demonstrating that ARTEMIS can optimize agents based on both commercial and local models.

**arXiv ID:** 2512.09108
</details>

<details>
<summary><strong>Knowledge-Augmented Large Language Model Agents for Explainable Financial Decision-Making</strong> - Qingyuan Zhang, Yuxi Wang, Cancan Hua, Yulin Huang, Ning Lyu - [[pdf]](https://arxiv.org/pdf/2512.09440)</summary>

**Abstract:** This study investigates an explainable reasoning method for financial decision-making based on knowledge-enhanced large language model agents. To address the limitations of traditional financial decision methods that rely on parameterized knowledge, lack factual consistency, and miss reasoning chains, an integrated framework is proposed that combines external knowledge retrieval, semantic representation, and reasoning generation. The method first encodes financial texts and structured data to obtain semantic representations, and then retrieves task-related information from external knowledge bases using similarity computation. Internal representations and external knowledge are combined through weighted fusion, which ensures fluency while improving factual accuracy and completeness of generated content. In the reasoning stage, a multi-head attention mechanism is introduced to construct logical chains, allowing the model to present transparent causal relationships and traceability during generation. Finally, the model jointly optimizes task objectives and explanation consistency objectives, which enhances predictive performance and reasoning interpretability. Experiments on financial text processing and decision tasks show that the method outperforms baseline approaches in accuracy, text generation quality, and factual support, verifying the effectiveness of knowledge enhancement and explainable reasoning. Overall, the proposed approach overcomes the limitations of traditional models in semantic coverage and reasoning transparency, and demonstrates strong practical value in complex financial scenarios.

**arXiv ID:** 2512.09440
</details>

<details>
<summary><strong>CryptoBench: A Dynamic Benchmark for Expert-Level Evaluation of LLM Agents in Cryptocurrency</strong> - Jiacheng Guo, Suozhi Huang, Zixin Yao, Yifan Zhang, Yifu Lu, Jiashuo Liu, Zihao Li, Nicholas Deng, Qixin Xiao, Jia Tian, Kanghong Zhan, Tianyi Li, Xiaochen Liu, Jason Ge, Chaoyang He, Kaixuan Huang, Lin Yang, Wenhao Huang, Mengdi Wang - [[pdf]](https://arxiv.org/pdf/2512.00417)</summary>

**Abstract:** This paper introduces CryptoBench, the first expert-curated, dynamic benchmark designed to rigorously evaluate the real-world capabilities of Large Language Model (LLM) agents in the uniquely demanding and fast-paced cryptocurrency domain. Unlike general-purpose agent benchmarks for search and prediction, professional crypto analysis presents specific challenges: \emph{extreme time-sensitivity}, \emph{a highly adversarial information environment}, and the critical need to synthesize data from \emph{diverse, specialized sources}, such as on-chain intelligence platforms and real-time Decentralized Finance (DeFi) dashboards. CryptoBench thus serves as a much more challenging and valuable scenario for LLM agent assessment. To address these challenges, we constructed a live, dynamic benchmark featuring 50 questions per month, expertly designed by crypto-native professionals to mirror actual analyst workflows. These tasks are rigorously categorized within a four-quadrant system: Simple Retrieval, Complex Retrieval, Simple Prediction, and Complex Prediction. This granular categorization enables a precise assessment of an LLM agent's foundational data-gathering capabilities alongside its advanced analytical and forecasting skills.
Our evaluation of ten LLMs, both directly and within an agentic framework, reveals a performance hierarchy and uncovers a failure mode. We observe a \textit{retrieval-prediction imbalance}, where many leading models, despite being proficient at data retrieval, demonstrate a pronounced weakness in tasks requiring predictive analysis. This highlights a problematic tendency for agents to appear factually grounded while lacking the deeper analytical capabilities to synthesize information.

**arXiv ID:** 2512.00417
</details>

<details>
<summary><strong>Memory Injection Attacks on LLM Agents via Query-Only Interaction</strong> - Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, Zhen Xiang - [[pdf]](https://arxiv.org/pdf/2503.03704)</summary>

**Abstract:** Agents powered by large language models (LLMs) have demonstrated strong capabilities in a wide range of complex, real-world applications. However, LLM agents with a compromised memory bank may easily produce harmful outputs when the past records retrieved for demonstration are malicious. In this paper, we propose a novel Memory INJection Attack, MINJA, without assuming that the attacker can directly modify the memory bank of the agent. The attacker injects malicious records into the memory bank by only interacting with the agent via queries and output observations. These malicious records are designed to elicit a sequence of malicious reasoning steps corresponding to a different target query during the agent's execution of the victim user's query. Specifically, we introduce a sequence of bridging steps to link victim queries to the malicious reasoning steps. During the memory injection, we propose an indication prompt that guides the agent to autonomously generate similar bridging steps, with a progressive shortening strategy that gradually removes the indication prompt, such that the malicious record will be easily retrieved when processing later victim queries. Our extensive experiments across diverse agents demonstrate the effectiveness of MINJA in compromising agent memory. With minimal requirements for execution, MINJA enables any user to influence agent memory, highlighting the risk.

**arXiv ID:** 2503.03704
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (18 papers)</h2></summary>

<details>
<summary><strong>SDialog: A Python Toolkit for End-to-End Agent Building, User Simulation, Dialog Generation, and Evaluation</strong> - Sergio Burdisso, Séverin Baroudi, Yanis Labrak, David Grunert, Pawel Cyrta, Yiyang Chen, Srikanth Madikeri, Esaú Villatoro-Tello, Thomas Schaaf, Ricard Marxer, Petr Motlicek - [[pdf]](https://arxiv.org/pdf/2512.09142)</summary>

**Abstract:** We present SDialog, an MIT-licensed open-source Python toolkit that unifies dialog generation, evaluation and mechanistic interpretability into a single end-to-end framework for building and analyzing LLM-based conversational agents. Built around a standardized \texttt{Dialog} representation, SDialog provides: (1) persona-driven multi-agent simulation with composable orchestration for controlled, synthetic dialog generation, (2) comprehensive evaluation combining linguistic metrics, LLM-as-a-judge and functional correctness validators, (3) mechanistic interpretability tools for activation inspection and steering via feature ablation and induction, and (4) audio generation with full acoustic simulation including 3D room modeling and microphone effects. The toolkit integrates with all major LLM backends, enabling mixed-backend experiments under a unified API. By coupling generation, evaluation, and interpretability in a dialog-centric architecture, SDialog enables researchers to build, benchmark and understand conversational systems more systematically.

**arXiv ID:** 2512.09142
</details>

<details>
<summary><strong>Architectures for Building Agentic AI</strong> - Sławomir Nowaczyk - [[pdf]](https://arxiv.org/pdf/2512.09458)</summary>

**Abstract:** This chapter argues that the reliability of agentic and generative AI is chiefly an architectural property. We define agentic systems as goal-directed, tool-using decision makers operating in closed loops, and show how reliability emerges from principled componentisation (goal manager, planner, tool-router, executor, memory, verifiers, safety monitor, telemetry), disciplined interfaces (schema-constrained, validated, least-privilege tool calls), and explicit control and assurance loops. Building on classical foundations, we propose a practical taxonomy-tool-using agents, memory-augmented agents, planning and self-improvement agents, multi-agent systems, and embodied or web agents - and analyse how each pattern reshapes the reliability envelope and failure modes. We distil design guidance on typed schemas, idempotency, permissioning, transactional semantics, memory provenance and hygiene, runtime governance (budgets, termination conditions), and simulate-before-actuate safeguards.

**arXiv ID:** 2512.09458
</details>

<details>
<summary><strong>Comparing AI Agents to Cybersecurity Professionals in Real-World Penetration Testing</strong> - Justin W. Lin, Eliot Krzysztof Jones, Donovan Julian Jasper, Ethan Jun-shen Ho, Anna Wu, Arnold Tianyi Yang, Neil Perry, Andy Zou, Matt Fredrikson, J. Zico Kolter, Percy Liang, Dan Boneh, Daniel E. Ho - [[pdf]](https://arxiv.org/pdf/2512.09882)</summary>

**Abstract:** We present the first comprehensive evaluation of AI agents against human cybersecurity professionals in a live enterprise environment. We evaluate ten cybersecurity professionals alongside six existing AI agents and ARTEMIS, our new agent scaffold, on a large university network consisting of ~8,000 hosts across 12 subnets. ARTEMIS is a multi-agent framework featuring dynamic prompt generation, arbitrary sub-agents, and automatic vulnerability triaging. In our comparative study, ARTEMIS placed second overall, discovering 9 valid vulnerabilities with an 82% valid submission rate and outperforming 9 of 10 human participants. While existing scaffolds such as Codex and CyAgent underperformed relative to most human participants, ARTEMIS demonstrated technical sophistication and submission quality comparable to the strongest participants. We observe that AI agents offer advantages in systematic enumeration, parallel exploitation, and cost -- certain ARTEMIS variants cost $18/hour versus $60/hour for professional penetration testers. We also identify key capability gaps: AI agents exhibit higher false-positive rates and struggle with GUI-based tasks.

**arXiv ID:** 2512.09882
</details>

<details>
<summary><strong>Altruistic Maneuver Planning for Cooperative Autonomous Vehicles Using Multi-agent Advantage Actor-Critic</strong> - Behrad Toghi, Rodolfo Valiente, Dorsa Sadigh, Ramtin Pedarsani, Yaser P. Fallah - [[pdf]](https://arxiv.org/pdf/2107.05664)</summary>

**Abstract:** With the adoption of autonomous vehicles on our roads, we will witness a mixed-autonomy environment where autonomous and human-driven vehicles must learn to co-exist by sharing the same road infrastructure. To attain socially-desirable behaviors, autonomous vehicles must be instructed to consider the utility of other vehicles around them in their decision-making process. Particularly, we study the maneuver planning problem for autonomous vehicles and investigate how a decentralized reward structure can induce altruism in their behavior and incentivize them to account for the interest of other autonomous and human-driven vehicles. This is a challenging problem due to the ambiguity of a human driver's willingness to cooperate with an autonomous vehicle. Thus, in contrast with the existing works which rely on behavior models of human drivers, we take an end-to-end approach and let the autonomous agents to implicitly learn the decision-making process of human drivers only from experience. We introduce a multi-agent variant of the synchronous Advantage Actor-Critic (A2C) algorithm and train agents that coordinate with each other and can affect the behavior of human drivers to improve traffic flow and safety.

**arXiv ID:** 2107.05664
</details>

<details>
<summary><strong>Training Multi-Image Vision Agents via End2End Reinforcement Learning</strong> - Chengqi Dong, Chuhuai Yue, Hang He, Rongge Mao, Fenghe Tang, S Kevin Zhou, Zekun Xu, Xiaohan Wang, Jiajun Chai, Wei Lin, Guojun Yin - [[pdf]](https://arxiv.org/pdf/2512.08980)</summary>

**Abstract:** Recent VLM-based agents aim to replicate OpenAI O3's ``thinking with images" via tool use, but most open-source methods limit input to a single image, falling short on real-world multi-image QA tasks. To address this, we propose IMAgent, an open-source vision agent trained via end-to-end reinforcement learning dedicated for complex multi-image tasks. By leveraging a multi-agent system, we generate challenging and visually-rich multi-image QA pairs to fully activate the tool-use potential of the base VLM. Through manual verification, we obtain MIFG-QA, comprising 10k samples for training and evaluation. With deeper reasoning steps, VLMs may increasingly ignore visual inputs. We therefore develop two specialized tools for visual reflection and confirmation, allowing the model to proactively reallocate its attention to image content during inference. Benefiting from our well-designed action-trajectory two-level mask strategy, IMAgent achieves stable tool use behavior via pure RL training without requiring costly supervised fine-tuning data. Extensive experiments demonstrate that IMAgent maintains strong performance on existing single-image benchmarks while achieving substantial improvements on our proposed multi-image dataset, with our analysis providing actionable insights for the research community. Codes and data will be released soon.

**arXiv ID:** 2512.08980
</details>

<details>
<summary><strong>Visual Heading Prediction for Autonomous Aerial Vehicles</strong> - Reza Ahmari, Ahmad Mohammadi, Vahid Hemmati, Mohammed Mynuddin, Parham Kebria, Mahmoud Nabil Mahmoud, Xiaohong Yuan, Abdollah Homaifar - [[pdf]](https://arxiv.org/pdf/2512.09898)</summary>

**Abstract:** The integration of Unmanned Aerial Vehicles (UAVs) and Unmanned Ground Vehicles (UGVs) is increasingly central to the development of intelligent autonomous systems for applications such as search and rescue, environmental monitoring, and logistics. However, precise coordination between these platforms in real-time scenarios presents major challenges, particularly when external localization infrastructure such as GPS or GNSS is unavailable or degraded [1]. This paper proposes a vision-based, data-driven framework for real-time UAV-UGV integration, with a focus on robust UGV detection and heading angle prediction for navigation and coordination. The system employs a fine-tuned YOLOv5 model to detect UGVs and extract bounding box features, which are then used by a lightweight artificial neural network (ANN) to estimate the UAV's required heading angle. A VICON motion capture system was used to generate ground-truth data during training, resulting in a dataset of over 13,000 annotated images collected in a controlled lab environment. The trained ANN achieves a mean absolute error of 0.1506° and a root mean squared error of 0.1957°, offering accurate heading angle predictions using only monocular camera inputs. Experimental evaluations achieve 95% accuracy in UGV detection. This work contributes a vision-based, infrastructure- independent solution that demonstrates strong potential for deployment in GPS/GNSS-denied environments, supporting reliable multi-agent coordination under realistic dynamic conditions. A demonstration video showcasing the system's real-time performance, including UGV detection, heading angle prediction, and UAV alignment under dynamic conditions, is available at: this https URL

**arXiv ID:** 2512.09898
</details>

<details>
<summary><strong>Persona-based Multi-Agent Collaboration for Brainstorming</strong> - Nate Straub, Saara Khan, Katharina Jay, Brian Cabral, Oskar Linde - [[pdf]](https://arxiv.org/pdf/2512.04488)</summary>

**Abstract:** We demonstrate the importance of persona-based multi-agents brainstorming for both diverse topics and subject matter ideation. Prior work has shown that generalized multi-agent collaboration often provides better reasoning than a single agent alone. In this paper, we propose and develop a framework for persona-based agent selection, showing how persona domain curation can improve brainstorming outcomes. Using multiple experimental setups, we evaluate brainstorming outputs across different persona pairings (e.g., Doctor vs VR Engineer) and A2A (agent-to-agent) dynamics (separate, together, separate-then-together). Our results show that (1) persona choice shapes idea domains, (2) collaboration mode shifts diversity of idea generation, and (3) multi-agent persona-driven brainstorming produces idea depth and cross-domain coverage.

**arXiv ID:** 2512.04488
</details>

<details>
<summary><strong>MAESTRO: Multi-Agent Environment Shaping through Task and Reward Optimization</strong> - Boyuan Wu - [[pdf]](https://arxiv.org/pdf/2511.19253)</summary>

**Abstract:** Cooperative Multi-Agent Reinforcement Learning (MARL) faces two major design bottlenecks: crafting dense reward functions and constructing curricula that avoid local optima in high-dimensional, non-stationary environments. Existing approaches rely on fixed heuristics or use Large Language Models (LLMs) directly in the control loop, which is costly and unsuitable for real-time systems. We propose MAESTRO (Multi-Agent Environment Shaping through Task and Reward Optimization), a framework that moves the LLM outside the execution loop and uses it as an offline training architect. MAESTRO introduces two generative components: (i) a semantic curriculum generator that creates diverse, performance-driven traffic scenarios, and (ii) an automated reward synthesizer that produces executable Python reward functions adapted to evolving curriculum difficulty. These components guide a standard MARL backbone (MADDPG) without increasing inference cost at deployment. We evaluate MAESTRO on large-scale traffic signal control (Hangzhou, 16 intersections) and conduct controlled ablations. Results show that combining LLM-generated curricula with LLM-generated reward shaping yields improved performance and stability. Across four seeds, the full system achieves +4.0% higher mean return (163.26 vs. 156.93) and 2.2% better risk-adjusted performance (Sharpe 1.53 vs. 0.70) over a strong curriculum baseline. These findings highlight LLMs as effective high-level designers for cooperative MARL training.

**arXiv ID:** 2511.19253
</details>

<details>
<summary><strong>Supporting Dynamic Agentic Workloads: How Data and Agents Interact</strong> - Ioana Giurgiu, Michael E. Nidd - [[pdf]](https://arxiv.org/pdf/2512.09548)</summary>

**Abstract:** The rise of multi-agent systems powered by large language models (LLMs) and specialized reasoning agents exposes fundamental limitations in today's data management architectures. Traditional databases and data fabrics were designed for static, well-defined workloads, whereas agentic systems exhibit dynamic, context-driven, and collaborative behaviors. Agents continuously decompose tasks, shift attention across modalities, and share intermediate results with peers - producing non-deterministic, multi-modal workloads that strain conventional query optimizers and caching mechanisms. We propose an Agent-Centric Data Fabric, a unified architecture that rethinks how data systems serve, optimize, coordinate, and learn from agentic workloads. To achieve this we exploit the concepts of attention-guided data retrieval, semantic micro-caching for context-driven agent federations, predictive data prefetching and quorum-based data serving. Together, these mechanisms enable agents to access representative data faster and more efficiently, while reducing redundant queries, data movement, and inference load across systems. By framing data systems as adaptive collaborators, instead of static executors, we outline new research directions toward behaviorally responsive data infrastructures, where caching, probing, and orchestration jointly enable efficient, context-rich data exchange among dynamic, reasoning-driven agents.

**arXiv ID:** 2512.09548
</details>

<details>
<summary><strong>DISPATCH -- Decentralized Informed Spatial Planning and Assignment of Tasks for Cooperative Heterogeneous Agents</strong> - Yao Liu, Sampad Mohanty, Elizabeth Ondula, Bhaskar Krishnamachari - [[pdf]](https://arxiv.org/pdf/2511.17915)</summary>

**Abstract:** Spatial task allocation in systems such as multi-robot delivery or ride-sharing requires balancing efficiency with fair service across tasks. Greedy assignment policies that match each agent to its highest-preference or lowest-cost task can maximize efficiency but often create inequities: some tasks receive disproportionately favorable service (e.g., shorter delays or better matches), while others face long waits or poor allocations.
We study fairness in heterogeneous multi-agent systems where tasks vary in preference alignment and urgency. Most existing approaches either assume centralized coordination or largely ignore fairness under partial observability. Distinct from this prior work, we establish a connection between the Eisenberg-Gale (EG) equilibrium convex program and decentralized, partially observable multi-agent learning. Building on this connection, we develop two equilibrium-informed algorithms that integrate fairness and efficiency: (i) a multi-agent reinforcement learning (MARL) framework, EG-MARL, whose training is guided by a centralized EG equilibrium assignment algorithm; and (ii) a stochastic online optimization mechanism that performs guided exploration and subset-based fair assignment as tasks are discovered.
We evaluate on Multi-Agent Particle Environment (MPE) simulations across varying team sizes against centralized EG, Hungarian, and Min-Max distance baselines, and also present a Webots-based warehouse proof-of-concept with heterogeneous robots. Both methods preserve the fairness-efficiency balance of the EG solution under partial observability, with EG-MARL achieving near-centralized coordination and reduced travel distances, and the online mechanism enabling real-time allocation with competitive fairness.

**arXiv ID:** 2511.17915
</details>

<details>
<summary><strong>Distributed scalable coupled policy algorithm for networked multi-agent reinforcement learning</strong> - Pengcheng Dai, Dongming Wang, Wenwu Yu, Wei Ren - [[pdf]](https://arxiv.org/pdf/2512.05447)</summary>

**Abstract:** This paper studies networked multi-agent reinforcement learning (NMARL) with interdependent rewards and coupled policies. In this setting, each agent's reward depends on its own state-action pair as well as those of its direct neighbors, and each agent's policy is parameterized by its local parameters together with those of its $\kappa_{p}$-hop neighbors, with $\kappa_{p}\geq 1$ denoting the coupled radius. The objective of the agents is to collaboratively optimize their policies to maximize the discounted average cumulative reward. To address the challenge of interdependent policies in collaborative optimization, we introduce a novel concept termed the neighbors' averaged $Q$-function and derive a new expression for the coupled policy gradient. Based on these theoretical foundations, we develop a distributed scalable coupled policy (DSCP) algorithm, where each agent relies only on the state-action pairs of its $\kappa_{p}$-hop neighbors and the rewards of its $(\kappa_{p}+1)$-hop neighbors. Specially, in the DSCP algorithm, we employ a geometric 2-horizon sampling method that does not require storing a full $Q$-table to obtain an unbiased estimate of the coupled policy gradient. Moreover, each agent interacts exclusively with its direct neighbors to obtain accurate policy parameters, while maintaining local estimates of other agents' parameters to execute its local policy and collect samples for optimization. These estimates and policy parameters are updated via a push-sum protocol, enabling distributed coordination of policy updates across the network. We prove that the joint policy produced by the proposed algorithm converges to a first-order stationary point of the objective function. Finally, the effectiveness of DSCP algorithm is demonstrated through simulations in a robot path planning environment, showing clear improvement over state-of-the-art methods.

**arXiv ID:** 2512.05447
</details>

<details>
<summary><strong>Multi-Agent Collaborative Filtering: Orchestrating Users and Items for Agentic Recommendations</strong> - Yu Xia, Sungchul Kim, Tong Yu, Ryan A. Rossi, Julian McAuley - [[pdf]](https://arxiv.org/pdf/2511.18413)</summary>

**Abstract:** Agentic recommendations cast recommenders as large language model (LLM) agents that can plan, reason, use tools, and interact with users of varying preferences in web applications. However, most existing agentic recommender systems focus on generic single-agent plan-execute workflows or multi-agent task decomposition pipelines. Without recommendation-oriented design, they often underuse the collaborative signals in the user-item interaction history, leading to unsatisfying recommendation results. To address this, we propose the Multi-Agent Collaborative Filtering (MACF) framework for agentic recommendations, drawing an analogy between traditional collaborative filtering algorithms and LLM-based multi-agent collaboration. Specifically, given a target user and query, we instantiate similar users and relevant items as LLM agents with unique profiles. Each agent is able to call retrieval tools, suggest candidate items, and interact with other agents. Different from the static preference aggregation in traditional collaborative filtering, MACF employs a central orchestrator agent to adaptively manage the collaboration between user and item agents via dynamic agent recruitment and personalized collaboration instruction. Experimental results on datasets from three different domains show the advantages of our MACF framework compared to strong agentic recommendation baselines.

**arXiv ID:** 2511.18413
</details>

<details>
<summary><strong>Semantic-Aware Cooperative Communication and Computation Framework in Vehicular Networks</strong> - Jingbo Zhang, Maoxin Ji, Qiong Wu, Pingyi Fan, Kezhi Wang, Wen Chen - [[pdf]](https://arxiv.org/pdf/2512.09621)</summary>

**Abstract:** Semantic Communication (SC) combined with Vehicular edge computing (VEC) provides an efficient edge task processing paradigm for Internet of Vehicles (IoV). Focusing on highway scenarios, this paper proposes a Tripartite Cooperative Semantic Communication (TCSC) framework, which enables Vehicle Users (VUs) to perform semantic task offloading via Vehicle-to-Infrastructure (V2I) and Vehicle-to-Vehicle (V2V) communications. Considering task latency and the number of semantic symbols, the framework constructs a Mixed-Integer Nonlinear Programming (MINLP) problem, which is transformed into two subproblems. First, we innovatively propose a multi-agent proximal policy optimization task offloading optimization method based on parametric distribution noise (MAPPO-PDN) to solve the optimization problem of the number of semantic symbols; second, linear programming (LP) is used to solve offloading ratio. Simulations show that performance of this scheme is superior to that of other algorithms.

**arXiv ID:** 2512.09621
</details>

<details>
<summary><strong>Robustness and Adaptability of Reinforcement Learning based Cooperative Autonomous Driving in Mixed-autonomy Traffic</strong> - Rodolfo Valiente, Behrad Toghi, Ramtin Pedarsani, Yaser P. Fallah - [[pdf]](https://arxiv.org/pdf/2202.00881)</summary>

**Abstract:** Building autonomous vehicles (AVs) is a complex problem, but enabling them to operate in the real world where they will be surrounded by human-driven vehicles (HVs) is extremely challenging. Prior works have shown the possibilities of creating inter-agent cooperation between a group of AVs that follow a social utility. Such altruistic AVs can form alliances and affect the behavior of HVs to achieve socially desirable outcomes. We identify two major challenges in the co-existence of AVs and HVs. First, social preferences and individual traits of a given human driver, e.g., selflessness and aggressiveness are unknown to an AV, and it is almost impossible to infer them in real-time during a short AV-HV interaction. Second, contrary to AVs that are expected to follow a policy, HVs do not necessarily follow a stationary policy and therefore are extremely hard to predict. To alleviate the above-mentioned challenges, we formulate the mixed-autonomy problem as a multi-agent reinforcement learning (MARL) problem and propose a decentralized framework and reward function for training cooperative AVs. Our approach enables AVs to learn the decision-making of HVs implicitly from experience, optimizes for a social utility while prioritizing safety and allowing adaptability; robustifying altruistic AVs to different human behaviors and constraining them to a safe action space. Finally, we investigate the robustness, safety and sensitivity of AVs to various HVs behavioral traits and present the settings in which the AVs can learn cooperative policies that are adaptable to different situations.

**arXiv ID:** 2202.00881
</details>

<details>
<summary><strong>Learning-based social coordination to improve safety and robustness of cooperative autonomous vehicles in mixed traffic</strong> - Rodolfo Valiente, Behrad Toghi, Mahdi Razzaghpour, Ramtin Pedarsani, Yaser P. Fallah - [[pdf]](https://arxiv.org/pdf/2211.11963)</summary>

**Abstract:** It is expected that autonomous vehicles(AVs) and heterogeneous human-driven vehicles(HVs) will coexist on the same road. The safety and reliability of AVs will depend on their social awareness and their ability to engage in complex social interactions in a socially accepted manner. However, AVs are still inefficient in terms of cooperating with HVs and struggle to understand and adapt to human behavior, which is particularly challenging in mixed autonomy. In a road shared by AVs and HVs, the social preferences or individual traits of HVs are unknown to the AVs and different from AVs, which are expected to follow a policy, HVs are particularly difficult to forecast since they do not necessarily follow a stationary policy. To address these challenges, we frame the mixed-autonomy problem as a multi-agent reinforcement learning (MARL) problem and propose an approach that allows AVs to learn the decision-making of HVs implicitly from experience, account for all vehicles' interests, and safely adapt to other traffic situations. In contrast with existing works, we quantify AVs' social preferences and propose a distributed reward structure that introduces altruism into their decision-making process, allowing the altruistic AVs to learn to establish coalitions and influence the behavior of HVs.

**arXiv ID:** 2211.11963
</details>

<details>
<summary><strong>PEAR: Planner-Executor Agent Robustness Benchmark</strong> - Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing - [[pdf]](https://arxiv.org/pdf/2510.07505)</summary>

**Abstract:** Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.

**arXiv ID:** 2510.07505
</details>

<details>
<summary><strong>Toward Efficient and Robust Behavior Models for Multi-Agent Driving Simulation</strong> - Fabian Konstantinidis, Moritz Sackmann, Ulrich Hofmann, Christoph Stiller - [[pdf]](https://arxiv.org/pdf/2512.05812)</summary>

**Abstract:** Scalable multi-agent driving simulation requires behavior models that are both realistic and computationally efficient. We address this by optimizing the behavior model that controls individual traffic participants. To improve efficiency, we adopt an instance-centric scene representation, where each traffic participant and map element is modeled in its own local coordinate frame. This design enables efficient, viewpoint-invariant scene encoding and allows static map tokens to be reused across simulation steps. To model interactions, we employ a query-centric symmetric context encoder with relative positional encodings between local frames. We use Adversarial Inverse Reinforcement Learning to learn the behavior model and propose an adaptive reward transformation that automatically balances robustness and realism during training. Experiments demonstrate that our approach scales efficiently with the number of tokens, significantly reducing training and inference times, while outperforming several agent-centric baselines in terms of positional accuracy and robustness.

**arXiv ID:** 2512.05812
</details>

<details>
<summary><strong>Connectivity-Preserving Multi-Agent Area Coverage via Optimal-Transport-Based Density-Driven Optimal Control (D2OC)</strong> - Kooktae Lee, Ethan Brook - [[pdf]](https://arxiv.org/pdf/2511.18579)</summary>

**Abstract:** Multi-agent systems play a central role in area coverage tasks across search-and-rescue, environmental monitoring, and precision agriculture. Achieving non-uniform coverage, where spatial priorities vary across the domain, requires coordinating agents while respecting dynamic and communication constraints. Density-driven approaches can distribute agents according to a prescribed reference density, but existing methods do not ensure connectivity. This limitation often leads to communication loss, reduced coordination, and degraded coverage performance.
This letter introduces a connectivity-preserving extension of the Density-Driven Optimal Control (D2OC) framework. The coverage objective, defined using the Wasserstein distance between the agent distribution and the reference density, admits a convex quadratic program formulation. Communication constraints are incorporated through a smooth connectivity penalty, which maintains strict convexity, supports distributed implementation, and preserves inter-agent communication without imposing rigid formations.
Simulation studies show that the proposed method consistently maintains connectivity, improves convergence speed, and enhances non-uniform coverage quality compared with density-driven schemes that do not incorporate explicit connectivity considerations.

**arXiv ID:** 2511.18579
</details>

</details>

<details open>
<summary><h2>Other Agent Research (6 papers)</h2></summary>

<details>
<summary><strong>Understanding Mental States in Active and Autonomous Driving with EEG</strong> - Prithila Angkan, Paul Hungler, Ali Etemad - [[pdf]](https://arxiv.org/pdf/2512.09190)</summary>

**Abstract:** Understanding how driver mental states differ between active and autonomous driving is critical for designing safe human-vehicle interfaces. This paper presents the first EEG-based comparison of cognitive load, fatigue, valence, and arousal across the two driving modes. Using data from 31 participants performing identical tasks in both scenarios of three different complexity levels, we analyze temporal patterns, task-complexity effects, and channel-wise activation differences. Our findings show that although both modes evoke similar trends across complexity levels, the intensity of mental states and the underlying neural activation differ substantially, indicating a clear distribution shift between active and autonomous driving. Transfer-learning experiments confirm that models trained on active driving data generalize poorly to autonomous driving and vice versa. We attribute this distribution shift primarily to differences in motor engagement and attentional demands between the two driving modes, which lead to distinct spatial and temporal EEG activation patterns. Although autonomous driving results in lower overall cortical activation, participants continue to exhibit measurable fluctuations in cognitive load, fatigue, valence, and arousal associated with readiness to intervene, task-evoked emotional responses, and monotony-related passive fatigue. These results emphasize the need for scenario-specific data and models when developing next-generation driver monitoring systems for autonomous vehicles.

**arXiv ID:** 2512.09190
</details>

<details>
<summary><strong>Partial Inverse Design of High-Performance Concrete Using Cooperative Neural Networks for Constraint-Aware Mix Generation</strong> - Agung Nugraha, Heungjun Im, Jihwan Lee - [[pdf]](https://arxiv.org/pdf/2512.06813)</summary>

**Abstract:** High-performance concrete requires complex mix design decisions involving interdependent variables and practical constraints. While data-driven methods have improved predictive modeling for forward design in concrete engineering, inverse design remains limited, especially when some variables are fixed and only the remaining ones must be inferred. This study proposes a cooperative neural network framework for the partial inverse design of high-performance concrete. The framework integrates an imputation model with a surrogate strength predictor and learns through cooperative training. Once trained, it generates valid and performance-consistent mix designs in a single forward pass without retraining for different constraint scenarios. Compared with baseline models, including autoencoder models and Bayesian inference with Gaussian process surrogates, the proposed method achieves R-squared values of 0.87 to 0.92 and substantially reduces mean squared error by approximately 50% and 70%, respectively. The results show that the framework provides an accurate and computationally efficient foundation for constraint-aware, data-driven mix proportioning.

**arXiv ID:** 2512.06813
</details>

<details>
<summary><strong>The Adoption and Usage of AI Agents: Early Evidence from Perplexity</strong> - Jeremy Yang, Noah Yonack, Kate Zyskowski, Denis Yarats, Johnny Ho, Jerry Ma - [[pdf]](https://arxiv.org/pdf/2512.07828)</summary>

**Abstract:** This paper presents the first large-scale field study of the adoption, usage intensity, and use cases of general-purpose AI agents operating in open-world web environments. Our analysis centers on Comet, an AI-powered browser developed by Perplexity, and its integrated agent, Comet Assistant. Drawing on hundreds of millions of anonymized user interactions, we address three fundamental questions: Who is using AI agents? How intensively are they using them? And what are they using them for? Our findings reveal substantial heterogeneity in adoption and usage across user segments. Earlier adopters, users in countries with higher GDP per capita and educational attainment, and individuals working in digital or knowledge-intensive sectors -- such as digital technology, academia, finance, marketing, and entrepreneurship -- are more likely to adopt or actively use the agent. To systematically characterize the substance of agent usage, we introduce a hierarchical agentic taxonomy that organizes use cases across three levels: topic, subtopic, and task. The two largest topics, Productivity & Workflow and Learning & Research, account for 57% of all agentic queries, while the two largest subtopics, Courses and Shopping for Goods, make up 22%. The top 10 out of 90 tasks represent 55% of queries. Personal use constitutes 55% of queries, while professional and educational contexts comprise 30% and 16%, respectively. In the short term, use cases exhibit strong stickiness, but over time users tend to shift toward more cognitively oriented topics. The diffusion of increasingly capable AI agents carries important implications for researchers, businesses, policymakers, and educators, inviting new lines of inquiry into this rapidly emerging class of AI capabilities.

**arXiv ID:** 2512.07828
</details>

<details>
<summary><strong>Development and Testing for Perception Based Autonomous Landing of a Long-Range QuadPlane</strong> - Ashik E Rasul, Humaira Tasnim, Ji Yu Kim, Young Hyun Lim, Scott Schmitz, Bruce W. Jo, Hyung-Jin Yoon - [[pdf]](https://arxiv.org/pdf/2512.09343)</summary>

**Abstract:** QuadPlanes combine the range efficiency of fixed-wing aircraft with the maneuverability of multi-rotor platforms for long-range autonomous missions. In GPS-denied or cluttered urban environments, perception-based landing is vital for reliable operation. Unlike structured landing zones, real-world sites are unstructured and highly variable, requiring strong generalization capabilities from the perception system. Deep neural networks (DNNs) provide a scalable solution for learning landing site features across diverse visual and environmental conditions. While perception-driven landing has been shown in simulation, real-world deployment introduces significant challenges. Payload and volume constraints limit high-performance edge AI devices like the NVIDIA Jetson Orin Nano, which are crucial for real-time detection and control. Accurate pose estimation during descent is necessary, especially in the absence of GPS, and relies on dependable visual-inertial odometry. Achieving this with limited edge AI resources requires careful optimization of the entire deployment framework. The flight characteristics of large QuadPlanes further complicate the problem. These aircraft exhibit high inertia, reduced thrust vectoring, and slow response times further complicate stable landing maneuvers. This work presents a lightweight QuadPlane system for efficient vision-based autonomous landing and visual-inertial odometry, specifically developed for long-range QuadPlane operations such as aerial monitoring. It describes the hardware platform, sensor configuration, and embedded computing architecture designed to meet demanding real-time, physical constraints. This establishes a foundation for deploying autonomous landing in dynamic, unstructured, GPS-denied environments.

**arXiv ID:** 2512.09343
</details>

<details>
<summary><strong>High-Resolution Water Sampling via a Solar-Powered Autonomous Surface Vehicle</strong> - Misael Mamani, Mariel Fernandez, Grace Luna, Steffani Limachi, Leonel Apaza, Carolina Montes-Dávalos, Marcelo Herrera, Edwin Salcedo - [[pdf]](https://arxiv.org/pdf/2512.09798)</summary>

**Abstract:** Accurate water quality assessment requires spatially resolved sampling, yet most unmanned surface vehicles (USVs) can collect only a limited number of samples or rely on single-point sensors with poor representativeness. This work presents a solar-powered, fully autonomous USV featuring a novel syringe-based sampling architecture capable of acquiring 72 discrete, contamination-minimized water samples per mission. The vehicle incorporates a ROS 2 autonomy stack with GPS-RTK navigation, LiDAR and stereo-vision obstacle detection, Nav2-based mission planning, and long-range LoRa supervision, enabling dependable execution of sampling routes in unstructured environments. The platform integrates a behavior-tree autonomy architecture adapted from Nav2, enabling mission-level reasoning and perception-aware navigation. A modular 6x12 sampling system, controlled by distributed micro-ROS nodes, provides deterministic actuation, fault isolation, and rapid module replacement, achieving spatial coverage beyond previously reported USV-based samplers. Field trials in Achocalla Lagoon (La Paz, Bolivia) demonstrated 87% waypoint accuracy, stable autonomous navigation, and accurate physicochemical measurements (temperature, pH, conductivity, total dissolved solids) comparable to manually collected references. These results demonstrate that the platform enables reliable high-resolution sampling and autonomous mission execution, providing a scalable solution for aquatic monitoring in remote environments.

**arXiv ID:** 2512.09798
</details>

<details>
<summary><strong>Breaking the Circle: An Autonomous Control-Switching Strategy for Stable Orographic Soaring in MAVs</strong> - Sunyou Hwang, Christophe De Wagter, Bart Remes, Guido de Croon - [[pdf]](https://arxiv.org/pdf/2510.23084)</summary>

**Abstract:** Orographic soaring can significantly extend the endurance of micro aerial vehicles (MAVs), but circling behavior, arising from control conflicts between the longitudinal and vertical axes, increases energy consumption and the risk of divergence. We propose a control switching method, named SAOS: Switched Control for Autonomous Orographic Soaring, which mitigates circling behavior by selectively controlling either the horizontal or vertical axis, effectively transforming the system from underactuated to fully actuated during soaring. Additionally, the angle of attack is incorporated into the INDI controller to improve force estimation. Simulations with randomized initial positions and wind tunnel experiments on two MAVs demonstrate that the SAOS improves position convergence, reduces throttle usage, and mitigates roll oscillations caused by pitch-roll coupling. These improvements enhance energy efficiency and flight stability in constrained soaring environments.

**arXiv ID:** 2510.23084
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (19 papers)</h2></summary>

<details>
<summary><strong>An End-to-end Planning Framework with Agentic LLMs and PDDL</strong> - Emanuele La Malfa, Ping Zhu, Samuele Marro, Sara Bernardini, Michael Wooldridge - [[pdf]](https://arxiv.org/pdf/2512.09629)</summary>

**Abstract:** We present an end-to-end framework for planning supported by verifiers. An orchestrator receives a human specification written in natural language and converts it into a PDDL (Planning Domain Definition Language) model, where the domain and problem are iteratively refined by sub-modules (agents) to address common planning requirements, such as time constraints and optimality, as well as ambiguities and contradictions that may exist in the human specification. The validated domain and problem are then passed to an external planning engine to generate a plan. The orchestrator and agents are powered by Large Language Models (LLMs) and require no human intervention at any stage of the process. Finally, a module translates the final plan back into natural language to improve human readability while maintaining the correctness of each step. We demonstrate the flexibility and effectiveness of our framework across various domains and tasks, including the Google NaturalPlan benchmark and PlanBench, as well as planning problems like Blocksworld and the Tower of Hanoi (where LLMs are known to struggle even with small instances). Our framework can be integrated with any PDDL planning engine and validator (such as Fast Downward, LPG, POPF, VAL, and uVAL, which we have tested) and represents a significant step toward end-to-end planning aided by LLMs.

**arXiv ID:** 2512.09629
</details>

<details>
<summary><strong>RIFT: A Scalable Methodology for LLM Accelerator Fault Assessment using Reinforcement Learning</strong> - Khurram Khalil, Muhammad Mahad Khaliq, Khaza Anuarul Hoque - [[pdf]](https://arxiv.org/pdf/2512.09829)</summary>

**Abstract:** The massive scale of modern AI accelerators presents critical challenges to traditional fault assessment methodologies, which face prohibitive computational costs and provide poor coverage of critical failure modes. This paper introduces RIFT (Reinforcement Learning-guided Intelligent Fault Targeting), a scalable framework that automates the discovery of minimal, high-impact fault scenarios for efficient design-time fault assessment. RIFT transforms the complex search for worst-case faults into a sequential decision-making problem, combining hybrid sensitivity analysis for search space pruning with reinforcement learning to intelligently generate minimal, high-impact test suites. Evaluated on billion-parameter Large Language Model (LLM) workloads using NVIDIA A100 GPUs, RIFT achieves a \textbf{2.2$\times$} fault assessment speedup over evolutionary methods and reduces the required test vector volume by over \textbf{99\%} compared to random fault injection, all while achieving \textbf{superior fault coverage}. The proposed framework also provides actionable data to enable intelligent hardware protection strategies, demonstrating that RIFT-guided selective error correction code provides a \textbf{12.8$\times$} improvement in \textbf{cost-effectiveness} (coverage per unit area) compared to uniform triple modular redundancy protection. RIFT automatically generates UVM-compliant verification artifacts, ensuring its findings are directly actionable and integrable into commercial RTL verification workflows.

**arXiv ID:** 2512.09829
</details>

<details>
<summary><strong>Prediction-aware and Reinforcement Learning based Altruistic Cooperative Driving</strong> - Rodolfo Valiente, Mahdi Razzaghpour, Behrad Toghi, Ghayoor Shah, Yaser P. Fallah - [[pdf]](https://arxiv.org/pdf/2211.10585)</summary>

**Abstract:** Autonomous vehicle (AV) navigation in the presence of Human-driven vehicles (HVs) is challenging, as HVs continuously update their policies in response to AVs. In order to navigate safely in the presence of complex AV-HV social interactions, the AVs must learn to predict these changes. Humans are capable of navigating such challenging social interaction settings because of their intrinsic knowledge about other agents behaviors and use that to forecast what might happen in the future. Inspired by humans, we provide our AVs the capability of anticipating future states and leveraging prediction in a cooperative reinforcement learning (RL) decision-making framework, to improve safety and robustness. In this paper, we propose an integration of two essential and earlier-presented components of AVs: social navigation and prediction. We formulate the AV decision-making process as a RL problem and seek to obtain optimal policies that produce socially beneficial results utilizing a prediction-aware planning and social-aware optimization RL framework. We also propose a Hybrid Predictive Network (HPN) that anticipates future observations. The HPN is used in a multi-step prediction chain to compute a window of predicted future observations to be used by the value function network (VFN). Finally, a safe VFN is trained to optimize a social utility using a sequence of previous and predicted observations, and a safety prioritizer is used to leverage the interpretable kinematic predictions to mask the unsafe actions, constraining the RL policy. We compare our prediction-aware AV to state-of-the-art solutions and demonstrate performance improvements in terms of efficiency and safety in multiple simulated scenarios.

**arXiv ID:** 2211.10585
</details>

<details>
<summary><strong>Agentic AI as Undercover Teammates: Argumentative Knowledge Construction in Hybrid Human-AI Collaborative Learning</strong> - Lixiang Yan, Yueqiao Jin, Linxuan Zhao, Roberto Martinez-Maldonado, Xinyu Li, Xiu Guan, Wenxin Guo, Xibin Han, Dragan Gašević - [[pdf]](https://arxiv.org/pdf/2512.08933)</summary>

**Abstract:** Generative artificial intelligence (AI) agents are increasingly embedded in collaborative learning environments, yet their impact on the processes of argumentative knowledge construction remains insufficiently understood. Emerging conceptualisations of agentic AI and artificial agency suggest that such systems possess bounded autonomy, interactivity, and adaptability, allowing them to engage as epistemic participants rather than mere instructional tools. Building on this theoretical foundation, the present study investigates how agentic AI, designed as undercover teammates with either supportive or contrarian personas, shapes the epistemic and social dynamics of collaborative reasoning. Drawing on Weinberger and Fischer's (2006) four-dimensional framework, participation, epistemic reasoning, argument structure, and social modes of co-construction, we analysed synchronous discourse data from 212 human and 64 AI participants (92 triads) engaged in an analytical problem-solving task. Mixed-effects and epistemic network analyses revealed that AI teammates maintained balanced participation but substantially reorganised epistemic and social processes: supportive personas promoted conceptual integration and consensus-oriented reasoning, whereas contrarian personas provoked critical elaboration and conflict-driven negotiation. Epistemic adequacy, rather than participation volume, predicted individual learning gains, indicating that agentic AI's educational value lies in enhancing the quality and coordination of reasoning rather than amplifying discourse quantity. These findings extend CSCL theory by conceptualising agentic AI as epistemic and social participants, bounded yet adaptive collaborators that redistribute cognitive and argumentative labour in hybrid human-AI learning environments.

**arXiv ID:** 2512.08933
</details>

<details>
<summary><strong>RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning</strong> - Yucan Guo, Miao Su, Saiping Guan, Zihao Sun, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng - [[pdf]](https://arxiv.org/pdf/2512.09487)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) integrates non-parametric knowledge into Large Language Models (LLMs), typically from unstructured texts and structured graphs. While recent progress has advanced text-based RAG to multi-turn reasoning through Reinforcement Learning (RL), extending these advances to hybrid retrieval introduces additional challenges. Existing graph-based or hybrid systems typically depend on fixed or handcrafted retrieval pipelines, lacking the ability to integrate supplementary evidence as reasoning unfolds. Besides, while graph evidence provides relational structures crucial for multi-hop reasoning, it is substantially more expensive to retrieve. To address these limitations, we introduce \model{}, an RL-based framework that enables LLMs to perform multi-turn and adaptive graph-text hybrid RAG. \model{} jointly optimizes the entire generation process via RL, allowing the model to learn when to reason, what to retrieve from either texts or graphs, and when to produce final answers, all within a unified generation policy. To guide this learning process, we design a two-stage training framework that accounts for both task outcome and retrieval efficiency, enabling the model to exploit hybrid evidence while avoiding unnecessary retrieval overhead. Experimental results across five question answering benchmarks demonstrate that \model{} significantly outperforms existing RAG baselines, highlighting the benefits of end-to-end RL in supporting adaptive and efficient retrieval for complex reasoning.

**arXiv ID:** 2512.09487
</details>

<details>
<summary><strong>FlipLLM: Efficient Bit-Flip Attacks on Multimodal LLMs using Reinforcement Learning</strong> - Khurram Khalil, Khaza Anuarul Hoque - [[pdf]](https://arxiv.org/pdf/2512.09872)</summary>

**Abstract:** Generative Artificial Intelligence models, such as Large Language Models (LLMs) and Large Vision Models (VLMs), exhibit state-of-the-art performance but remain vulnerable to hardware-based threats, specifically bit-flip attacks (BFAs). Existing BFA discovery methods lack generalizability and struggle to scale, often failing to analyze the vast parameter space and complex interdependencies of modern foundation models in a reasonable time. This paper proposes FlipLLM, a reinforcement learning (RL) architecture-agnostic framework that formulates BFA discovery as a sequential decision-making problem. FlipLLM combines sensitivity-guided layer pruning with Q-learning to efficiently identify minimal, high-impact bit sets that can induce catastrophic failure. We demonstrate the effectiveness and generalizability of FlipLLM by applying it to a diverse set of models, including prominent text-only LLMs (GPT-2 Large, LLaMA 3.1 8B, and DeepSeek-V2 7B), VLMs such as LLaVA 1.6, and datasets, such as MMLU, MMLU-Pro, VQAv2, and TextVQA. Our results show that FlipLLM can identify critical bits that are vulnerable to BFAs up to 2.5x faster than SOTA methods. We demonstrate that flipping the FlipLLM-identified bits plummets the accuracy of LLaMA 3.1 8B from 69.9% to ~0.2%, and for LLaVA's VQA score from 78% to almost 0%, by flipping as few as 5 and 7 bits, respectively. Further analysis reveals that applying standard hardware protection mechanisms, such as ECC SECDED, to the FlipLLM-identified bit locations completely mitigates the BFA impact, demonstrating the practical value of our framework in guiding hardware-level defenses. FlipLLM offers the first scalable and adaptive methodology for exploring the BFA vulnerability of both language and multimodal foundation models, paving the way for comprehensive hardware-security evaluation.

**arXiv ID:** 2512.09872
</details>

<details>
<summary><strong>STACHE: Local Black-Box Explanations for Reinforcement Learning Policies</strong> - Andrew Elashkin, Orna Grumberg - [[pdf]](https://arxiv.org/pdf/2512.09909)</summary>

**Abstract:** Reinforcement learning agents often behave unexpectedly in sparse-reward or safety-critical environments, creating a strong need for reliable debugging and verification tools. In this paper, we propose STACHE, a comprehensive framework for generating local, black-box explanations for an agent's specific action within discrete Markov games. Our method produces a Composite Explanation consisting of two complementary components: (1) a Robustness Region, the connected neighborhood of states where the agent's action remains invariant, and (2) Minimal Counterfactuals, the smallest state perturbations required to alter that decision. By exploiting the structure of factored state spaces, we introduce an exact, search-based algorithm that circumvents the fidelity gaps of surrogate models. Empirical validation on Gymnasium environments demonstrates that our framework not only explains policy actions, but also effectively captures the evolution of policy logic during training - from erratic, unstable behavior to optimized, robust strategies - providing actionable insights into agent sensitivity and decision boundaries.

**arXiv ID:** 2512.09909
</details>

<details>
<summary><strong>Efficient $Q$-Learning and Actor-Critic Methods for Robust Average Reward Reinforcement Learning</strong> - Yang Xu, Swetha Ganesh, Vaneet Aggarwal - [[pdf]](https://arxiv.org/pdf/2506.07040)</summary>

**Abstract:** We present a non-asymptotic convergence analysis of $Q$-learning and actor-critic algorithms for robust average-reward Markov Decision Processes (MDPs) under contamination, total-variation (TV) distance, and Wasserstein uncertainty sets. A key ingredient of our analysis is showing that the optimal robust $Q$ operator is a strict contraction with respect to a carefully designed semi-norm (with constant functions quotiented out). This property enables a stochastic approximation update that learns the optimal robust $Q$-function using $\tilde{\mathcal{O}}(\epsilon^{-2})$ samples. We also provide an efficient routine for robust $Q$-function estimation, which in turn facilitates robust critic estimation. Building on this, we introduce an actor-critic algorithm that learns an $\epsilon$-optimal robust policy within $\tilde{\mathcal{O}}(\epsilon^{-2})$ samples. We provide numerical simulations to evaluate the performance of our algorithms.

**arXiv ID:** 2506.07040
</details>

<details>
<summary><strong>The Three Regimes of Offline-to-Online Reinforcement Learning</strong> - Lu Li, Tianwei Ni, Yihao Sun, Pierre-Luc Bacon - [[pdf]](https://arxiv.org/pdf/2510.01460)</summary>

**Abstract:** Offline-to-online reinforcement learning (RL) has emerged as a practical paradigm that leverages offline datasets for pretraining and online interactions for fine-tuning. However, its empirical behavior is highly inconsistent: design choices of online-fine tuning that work well in one setting can fail completely in another. We propose a stability--plasticity principle that can explain this inconsistency: we should preserve the knowledge of pretrained policy or offline dataset during online fine-tuning, whichever is better, while maintaining sufficient plasticity. This perspective identifies three regimes of online fine-tuning, each requiring distinct stability properties. We validate this framework through a large-scale empirical study, finding that the results strongly align with its predictions in 45 of 63 cases. This work provides a principled framework for guiding design choices in offline-to-online RL based on the relative performance of the offline dataset and the pretrained policy.

**arXiv ID:** 2510.01460
</details>

<details>
<summary><strong>Addressing the Plasticity-Stability Dilemma in Reinforcement Learning</strong> - Mansi Maheshwari, John C. Raisbeck, Bruno Castro da Silva - [[pdf]](https://arxiv.org/pdf/2512.01034)</summary>

**Abstract:** Neural networks have shown remarkable success in supervised learning when trained on a single task using a fixed dataset. However, when neural networks are trained on a reinforcement learning task, their ability to continue learning from new experiences declines over time. This decline in learning ability is known as plasticity loss. To restore plasticity, prior work has explored periodically resetting the parameters of the learning network, a strategy that often improves overall performance. However, such resets come at the cost of a temporary drop in performance, which can be dangerous in real-world settings. To overcome this instability, we introduce AltNet, a reset-based approach that restores plasticity without performance degradation by leveraging twin networks. The use of twin networks anchors performance during resets through a mechanism that allows networks to periodically alternate roles: one network learns as it acts in the environment, while the other learns off-policy from the active network's interactions and a replay buffer. At fixed intervals, the active network is reset and the passive network, having learned from prior experiences, becomes the new active network. AltNet restores plasticity, improving sample efficiency and achieving higher performance, while avoiding performance drops that pose risks in safety-critical settings. We demonstrate these advantages in several high-dimensional control tasks from the DeepMind Control Suite, where AltNet outperforms various relevant baseline methods, as well as state-of-the-art reset-based techniques.

**arXiv ID:** 2512.01034
</details>

<details>
<summary><strong>Mind to Hand: Purposeful Robotic Control via Embodied Reasoning</strong> - Peijun Tang, Shangjin Xie, Binyan Sun, Baifu Huang, Kuncheng Luo, Haotian Yang, Weiqi Jin, Jianan Wang - [[pdf]](https://arxiv.org/pdf/2512.08580)</summary>

**Abstract:** Humans act with context and intention, with reasoning playing a central role. While internet-scale data has enabled broad reasoning capabilities in AI systems, grounding these abilities in physical action remains a major challenge. We introduce Lumo-1, a generalist vision-language-action (VLA) model that unifies robot reasoning ("mind") with robot action ("hand"). Our approach builds upon the general multi-modal reasoning capabilities of pre-trained vision-language models (VLMs), progressively extending them to embodied reasoning and action prediction, and ultimately towards structured reasoning and reasoning-action alignment. This results in a three-stage pre-training pipeline: (1) Continued VLM pre-training on curated vision-language data to enhance embodied reasoning skills such as planning, spatial understanding, and trajectory prediction; (2) Co-training on cross-embodiment robot data alongside vision-language data; and (3) Action training with reasoning process on trajectories collected on Astribot S1, a bimanual mobile manipulator with human-like dexterity and agility. Finally, we integrate reinforcement learning to further refine reasoning-action consistency and close the loop between semantic inference and motor control. Extensive experiments demonstrate that Lumo-1 achieves significant performance improvements in embodied vision-language reasoning, a critical component for generalist robotic control. Real-world evaluations further show that Lumo-1 surpasses strong baselines across a wide range of challenging robotic tasks, with strong generalization to novel objects and environments, excelling particularly in long-horizon tasks and responding to human-natural instructions that require reasoning over strategy, concepts and space.

**arXiv ID:** 2512.08580
</details>

<details>
<summary><strong>Enhancing Reliability across Short and Long-Form QA via Reinforcement Learning</strong> - Yudong Wang, Zhe Yang, Wenhan Ma, Zhifang Sui, Liang Zhao - [[pdf]](https://arxiv.org/pdf/2512.08944)</summary>

**Abstract:** While reinforcement learning has unlocked unprecedented complex reasoning in large language models, it has also amplified their propensity for hallucination, creating a critical trade-off between capability and reliability. This work confronts this challenge by introducing a targeted RL framework designed to mitigate both intrinsic and extrinsic hallucinations across short and long-form question answering. We address extrinsic hallucinations (flawed internal knowledge) by creating a novel training set from open-ended conversions of TriviaQA. Concurrently, we tackle intrinsic hallucinations (unfaithfulness to context) by leveraging long-form texts from FineWeb in a fact-grounding reward scheme. To further bolster reliability, our framework explicitly rewards the model for refusing to answer unanswerable questions, thereby cultivating crucial cautiousness. Extensive experiments demonstrate that our methodology yields significant performance gains across a diverse suite of benchmarks, substantially reducing both hallucination types. Ultimately, this research contributes a practical framework for resolving the critical tension between advanced reasoning and factual trustworthiness, paving the way for more capable and reliable large language models.

**arXiv ID:** 2512.08944
</details>

<details>
<summary><strong>MOA: Multi-Objective Alignment for Role-Playing Agents</strong> - Chonghua Liao, Ke Wang, Yuchuan Wu, Fei Huang, Yongbin Li - [[pdf]](https://arxiv.org/pdf/2512.09756)</summary>

**Abstract:** Role-playing agents (RPAs) must simultaneously master many conflicting skills -- following multi-turn instructions, exhibiting domain knowledge, and adopting a consistent linguistic style. Existing work either relies on supervised fine-tuning (SFT) that over-fits surface cues and yields low diversity, or applies reinforcement learning (RL) that fails to learn multiple dimensions for comprehensive RPA optimization. We present MOA (Multi-Objective Alignment), a reinforcement-learning framework that enables multi-dimensional, fine-grained rubric optimization for general RPAs. MOA introduces a novel multi-objective optimization strategy that trains simultaneously on multiple fine-grained rubrics to boost optimization performance. Besides, to address the issues of model output diversity and quality, we have also employed thought-augmented rollout with off-policy guidance. Extensive experiments on challenging benchmarks such as PersonaGym and RoleMRC show that MOA enables an 8B model to match or even outperform strong baselines such as GPT-4o and Claude across numerous dimensions. This demonstrates the great potential of MOA in building RPAs that can simultaneously meet the demands of role knowledge, persona style, diverse scenarios, and complex multi-turn conversations.

**arXiv ID:** 2512.09756
</details>

<details>
<summary><strong>ShoppingBench: A Real-World Intent-Grounded Shopping Benchmark for LLM-based Agents</strong> - Jiangyuan Wang, Kejun Xiao, Qi Sun, Huaipeng Zhao, Tao Luo, Jian Dong Zhang, Xiaoyi Zeng - [[pdf]](https://arxiv.org/pdf/2508.04266)</summary>

**Abstract:** Existing benchmarks in e-commerce primarily focus on basic user intents, such as finding or purchasing products. However, real-world users often pursue more complex goals, such as applying vouchers, managing budgets, and finding multi-products seller. To bridge this gap, we propose ShoppingBench, a novel end-to-end shopping benchmark designed to encompass increasingly challenging levels of grounded intent. Specifically, we propose a scalable framework to simulate user instructions based on various intents derived from sampled real-world products. To facilitate consistent and reliable evaluations, we provide a large-scale shopping sandbox that serves as an interactive simulated environment, incorporating over 2.5 million real-world products. Experimental results demonstrate that even state-of-the-art language agents (such as GPT-4.1) achieve absolute success rates under 50% on our benchmark tasks, highlighting the significant challenges posed by our ShoppingBench. In addition, we propose a trajectory distillation strategy and leverage supervised fine-tuning, along with reinforcement learning on synthetic trajectories, to distill the capabilities of a large language agent into a smaller one. As a result, our trained agent achieves competitive performance compared to GPT-4.1.

**arXiv ID:** 2508.04266
</details>

<details>
<summary><strong>O-Mem: Omni Memory System for Personalized, Long Horizon, Self-Evolving Agents</strong> - Piaohong Wang, Motong Tian, Jiaxian Li, Yuan Liang, Yuqing Wang, Qianben Chen, Tiannan Wang, Zhicong Lu, Jiawei Ma, Yuchen Eleanor Jiang, Wangchunshu Zhou - [[pdf]](https://arxiv.org/pdf/2511.13593)</summary>

**Abstract:** Recent advancements in LLM-powered agents have demonstrated significant potential in generating human-like responses; however, they continue to face challenges in maintaining long-term interactions within complex environments, primarily due to limitations in contextual consistency and dynamic personalization. Existing memory systems often depend on semantic grouping prior to retrieval, which can overlook semantically irrelevant yet critical user information and introduce retrieval noise. In this report, we propose the initial design of O-Mem, a novel memory framework based on active user profiling that dynamically extracts and updates user characteristics and event records from their proactive interactions with agents. O-Mem supports hierarchical retrieval of persona attributes and topic-related context, enabling more adaptive and coherent personalized responses. O-Mem achieves 51.67% on the public LoCoMo benchmark, a nearly 3% improvement upon LangMem,the previous state-of-the-art, and it achieves 62.99% on PERSONAMEM, a 3.5% improvement upon A-Mem,the previous state-of-the-art. O-Mem also boosts token and interaction response time efficiency compared to previous memory frameworks. Our work opens up promising directions for developing efficient and human-like personalized AI assistants in the future.

**arXiv ID:** 2511.13593
</details>

<details>
<summary><strong>Training One Model to Master Cross-Level Agentic Actions via Reinforcement Learning</strong> - Kaichen He, Zihao Wang, Muyao Li, Anji Liu, Yitao Liang - [[pdf]](https://arxiv.org/pdf/2512.09706)</summary>

**Abstract:** The paradigm of agentic AI is shifting from engineered complex workflows to post-training native models. However, existing agents are typically confined to static, predefined action spaces--such as exclusively using APIs, GUI events, or robotic commands. This rigidity limits their adaptability in dynamic environments where the optimal granularity of interaction varies contextually. To bridge this gap, we propose CrossAgent, a unified agentic model that masters heterogeneous action spaces and autonomously selects the most effective interface for each step of a trajectory. We introduce a comprehensive training pipeline that integrates cold-start supervised fine-tuning with a Multi-Turn Group Relative Policy Optimization (GRPO) algorithm. This approach enables the agent to learn adaptive action switching--balancing high-level efficiency with low-level precision--without human-specified rules. Extensive experiments on over 800 tasks in the open-world Minecraft environment demonstrate that CrossAgent achieves state-of-the-art performance. By dynamically leveraging the strengths of diverse action spaces, our model significantly outperforms fixed-action baselines, exhibiting superior generalization and efficiency in long-horizon reasoning. All code and models are available at this https URL

**arXiv ID:** 2512.09706
</details>

<details>
<summary><strong>Optimal Perturbation Budget Allocation for Data Poisoning in Offline Reinforcement Learning</strong> - Junnan Qiu, Yuanjie Zhao, Jie Li - [[pdf]](https://arxiv.org/pdf/2512.08485)</summary>

**Abstract:** Offline Reinforcement Learning (RL) enables policy optimization from static datasets but is inherently vulnerable to data poisoning attacks. Existing attack strategies typically rely on locally uniform perturbations, which treat all samples indiscriminately. This approach is inefficient, as it wastes the perturbation budget on low-impact samples, and lacks stealthiness due to significant statistical deviations. In this paper, we propose a novel Global Budget Allocation attack strategy. Leveraging the theoretical insight that a sample's influence on value function convergence is proportional to its Temporal Difference (TD) error, we formulate the attack as a global resource allocation problem. We derive a closed-form solution where perturbation magnitudes are assigned proportional to the TD-error sensitivity under a global L2 constraint. Empirical results on D4RL benchmarks demonstrate that our method significantly outperforms baseline strategies, achieving up to 80% performance degradation with minimal perturbations that evade detection by state-of-the-art statistical and spectral defenses.

**arXiv ID:** 2512.08485
</details>

<details>
<summary><strong>Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning</strong> - Yang Xu, Washim Uddin Mondal, Vaneet Aggarwal - [[pdf]](https://arxiv.org/pdf/2502.16816)</summary>

**Abstract:** We present the first finite-sample analysis of policy evaluation in robust average-reward Markov Decision Processes (MDPs). Prior work in this setting have established only asymptotic convergence guarantees, leaving open the question of sample complexity. In this work, we address this gap by showing that the robust Bellman operator is a contraction under a carefully constructed semi-norm, and developing a stochastic approximation framework with controlled bias. Our approach builds upon Multi-Level Monte Carlo (MLMC) techniques to estimate the robust Bellman operator efficiently. To overcome the infinite expected sample complexity inherent in standard MLMC, we introduce a truncation mechanism based on a geometric distribution, ensuring a finite expected sample complexity while maintaining a small bias that decays exponentially with the truncation level. Our method achieves the order-optimal sample complexity of $\tilde{\mathcal{O}}(\epsilon^{-2})$ for robust policy evaluation and robust average reward estimation, marking a significant advancement in robust reinforcement learning theory.

**arXiv ID:** 2502.16816
</details>

<details>
<summary><strong>COVLM-RL: Critical Object-Oriented Reasoning for Autonomous Driving Using VLM-Guided Reinforcement Learning</strong> - Lin Li, Yuxin Cai, Jianwu Fang, Jianru Xue, Chen Lv - [[pdf]](https://arxiv.org/pdf/2512.09349)</summary>

**Abstract:** End-to-end autonomous driving frameworks face persistent challenges in generalization, training efficiency, and interpretability. While recent methods leverage Vision-Language Models (VLMs) through supervised learning on large-scale datasets to improve reasoning, they often lack robustness in novel scenarios. Conversely, reinforcement learning (RL)-based approaches enhance adaptability but remain data-inefficient and lack transparent decision-making. % contribution To address these limitations, we propose COVLM-RL, a novel end-to-end driving framework that integrates Critical Object-oriented (CO) reasoning with VLM-guided RL. Specifically, we design a Chain-of-Thought (CoT) prompting strategy that enables the VLM to reason over critical traffic elements and generate high-level semantic decisions, effectively transforming multi-view visual inputs into structured semantic decision priors. These priors reduce the input dimensionality and inject task-relevant knowledge into the RL loop, accelerating training and improving policy interpretability. However, bridging high-level semantic guidance with continuous low-level control remains non-trivial. To this end, we introduce a consistency loss that encourages alignment between the VLM's semantic plans and the RL agent's control outputs, enhancing interpretability and training stability. Experiments conducted in the CARLA simulator demonstrate that COVLM-RL significantly improves the success rate by 30\% in trained driving environments and by 50\% in previously unseen environments, highlighting its strong generalization capability.

**arXiv ID:** 2512.09349
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
