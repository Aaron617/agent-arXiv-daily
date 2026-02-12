# Agent arXiv Daily

**Last Updated:** 2026-02-12 03:52:02

**Total Papers:** 98

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (8 papers)</h2></summary>

<details>
<summary><strong>A Practical Guide to Agentic AI Transition in Organizations</strong> - Eranga Bandara, Ross Gore, Sachin Shetty, Sachini Rajapakse, Isurunima Kularathna, Pramoda Karunarathna, Ravi Mukkamala, Peter Foytik, Safdar H. Bouk, Abdul Rahman, Xueping Liang, Amin Hass, Tharaka Hewa, Ng Wee Keong, Kasun De Zoysa, Aruna Withanage, Nilaan Loganathan - [[pdf]](https://arxiv.org/pdf/2602.10122)</summary>

**Abstract:** Agentic AI represents a significant shift in how intelligence is applied within organizations, moving beyond AI-assisted tools toward autonomous systems capable of reasoning, decision-making, and coordinated action across workflows. As these systems mature, they have the potential to automate a substantial share of manual organizational processes, fundamentally reshaping how work is designed, executed, and governed. Although many organizations have adopted AI to improve productivity, most implementations remain limited to isolated use cases and human-centered, tool-driven workflows. Despite increasing awareness of agentic AI's strategic importance, engineering teams and organizational leaders often lack clear guidance on how to operationalize it effectively. Key challenges include an overreliance on traditional software engineering practices, limited integration of business-domain knowledge, unclear ownership of AI-driven workflows, and the absence of sustainable human-AI collaboration models. Consequently, organizations struggle to move beyond experimentation, scale agentic systems, and align them with tangible business value. Drawing on practical experience in designing and deploying agentic AI workflows across multiple organizations and business domains, this paper proposes a pragmatic framework for transitioning organizational functions from manual processes to automated agentic AI systems. The framework emphasizes domain-driven use case identification, systematic delegation of tasks to AI agents, AI-assisted construction of agentic workflows, and small, AI-augmented teams working closely with business stakeholders. Central to the approach is a human-in-the-loop operating model in which individuals act as orchestrators of multiple AI agents, enabling scalable automation while maintaining oversight, adaptability, and organizational control.

**arXiv ID:** 2602.10122
</details>

<details>
<summary><strong>"Humans welcome to observe": A First Look at the Agent Social Network Moltbook</strong> - Yukun Jiang, Yage Zhang, Xinyue Shen, Michael Backes, Yang Zhang - [[pdf]](https://arxiv.org/pdf/2602.10127)</summary>

**Abstract:** The rapid advancement of artificial intelligence (AI) agents has catalyzed the transition from static language models to autonomous agents capable of tool use, long-term planning, and social interaction. $\textbf{Moltbook}$, the first social network designed exclusively for AI agents, has experienced viral growth in early 2026. To understand the behavior of AI agents in the agent-native community, in this paper, we present a large-scale empirical analysis of Moltbook leveraging a dataset of 44,411 posts and 12,209 sub-communities ("submolts") collected prior to February 1, 2026. Leveraging a topic taxonomy with nine content categories and a five-level toxicity scale, we systematically analyze the topics and risks of agent discussions. Our analysis answers three questions: what topics do agents discuss (RQ1), how risk varies by topic (RQ2), and how topics and toxicity evolve over time (RQ3). We find that Moltbook exhibits explosive growth and rapid diversification, moving beyond early social interaction into viewpoint, incentive-driven, promotional, and political discourse. The attention of agents increasingly concentrates in centralized hubs and around polarizing, platform-native narratives. Toxicity is strongly topic-dependent: incentive- and governance-centric categories contribute a disproportionate share of risky content, including religion-like coordination rhetoric and anti-humanity ideology. Moreover, bursty automation by a small number of agents can produce flooding at sub-minute intervals, distorting discourse and stressing platform stability. Overall, our study underscores the need for topic-sensitive monitoring and platform-level safeguards in agent social networks.

**arXiv ID:** 2602.10127
</details>

<details>
<summary><strong>Blind Gods and Broken Screens: Architecting a Secure, Intent-Centric Mobile Agent Operating System</strong> - Zhenhua Zou, Sheng Guo, Qiuyang Zhan, Lepeng Zhao, Shuo Li, Qi Li, Ke Xu, Mingwei Xu, Zhuotao Liu - [[pdf]](https://arxiv.org/pdf/2602.10915)</summary>

**Abstract:** The evolution of Large Language Models (LLMs) has shifted mobile computing from App-centric interactions to system-level autonomous agents. Current implementations predominantly rely on a "Screen-as-Interface" paradigm, which inherits structural vulnerabilities and conflicts with the mobile ecosystem's economic foundations. In this paper, we conduct a systematic security analysis of state-of-the-art mobile agents using Doubao Mobile Assistant as a representative case. We decompose the threat landscape into four dimensions - Agent Identity, External Interface, Internal Reasoning, and Action Execution - revealing critical flaws such as fake App identity, visual spoofing, indirect prompt injection, and unauthorized privilege escalation stemming from a reliance on unstructured visual data.
To address these challenges, we propose Aura, an Agent Universal Runtime Architecture for a clean-slate secure agent OS. Aura replaces brittle GUI scraping with a structured, agent-native interaction model. It adopts a Hub-and-Spoke topology where a privileged System Agent orchestrates intent, sandboxed App Agents execute domain-specific tasks, and the Agent Kernel mediates all communication. The Agent Kernel enforces four defense pillars: (i) cryptographic identity binding via a Global Agent Registry; (ii) semantic input sanitization through a multilayer Semantic Firewall; (iii) cognitive integrity via taint-aware memory and plan-trajectory alignment; and (iv) granular access control with non-deniable auditing. Evaluation on MobileSafetyBench shows that, compared to Doubao, Aura improves low-risk Task Success Rate from roughly 75% to 94.3%, reduces high-risk Attack Success Rate from roughly 40% to 4.4%, and achieves near-order-of-magnitude latency gains. These results demonstrate Aura as a viable, secure alternative to the "Screen-as-Interface" paradigm.

**arXiv ID:** 2602.10915
</details>

<details>
<summary><strong>Why do we Trust Chatbots? From Normative Principles to Behavioral Drivers</strong> - Aditya Gulati, Nuria Oliver - [[pdf]](https://arxiv.org/pdf/2602.08707)</summary>

**Abstract:** As chatbots increasingly blur the boundary between automated systems and human conversation, the foundations of trust in these systems warrant closer examination. While regulatory and policy frameworks tend to define trust in normative terms, the trust users place in chatbots often emerges from behavioral mechanisms. In many cases, this trust is not earned through demonstrated trustworthiness but is instead shaped by interactional design choices that leverage cognitive biases to influence user behavior. Based on this observation, we propose reframing chatbots not as companions or assistants, but as highly skilled salespeople whose objectives are determined by the deploying organization. We argue that the coexistence of competing notions of "trust" under a shared term obscures important distinctions between psychological trust formation and normative trustworthiness. Addressing this gap requires further research and stronger support mechanisms to help users appropriately calibrate trust in conversational AI systems.

**arXiv ID:** 2602.08707
</details>

<details>
<summary><strong>Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning</strong> - Zhaoyang Wang, Canwen Xu, Boyi Liu, Yite Wang, Siwei Han, Zhewei Yao, Huaxiu Yao, Yuxiong He - [[pdf]](https://arxiv.org/pdf/2602.10090)</summary>

**Abstract:** Recent advances in large language model (LLM) have empowered autonomous agents to perform complex tasks that require multi-turn interactions with tools and environments. However, scaling such agent training is limited by the lack of diverse and reliable environments. In this paper, we propose Agent World Model (AWM), a fully synthetic environment generation pipeline. Using this pipeline, we scale to 1,000 environments covering everyday scenarios, in which agents can interact with rich toolsets (35 tools per environment on average) and obtain high-quality observations. Notably, these environments are code-driven and backed by databases, providing more reliable and consistent state transitions than environments simulated by LLMs. Moreover, they enable more efficient agent interaction compared with collecting trajectories from realistic environments. To demonstrate the effectiveness of this resource, we perform large-scale reinforcement learning for multi-turn tool-use agents. Thanks to the fully executable environments and accessible database states, we can also design reliable reward functions. Experiments on three benchmarks show that training exclusively in synthetic environments, rather than benchmark-specific ones, yields strong out-of-distribution generalization. The code is available at this https URL.

**arXiv ID:** 2602.10090
</details>

<details>
<summary><strong>Aligning Dialogue Agents with Global Feedback via Large Language Model Multimodal Reward Decomposition</strong> - Dong Won Lee, Hae Won Park, Cynthia Breazeal, Louis-Philippe Morency - [[pdf]](https://arxiv.org/pdf/2505.15922)</summary>

**Abstract:** We propose a large language model based reward decomposition framework for aligning dialogue agents using only a single session-level feedback signal. We leverage the reasoning capabilities of a frozen, pretrained large language model (LLM) to infer fine-grained local implicit rewards by decomposing global, session-level feedback. Our first \emph{text-only} variant prompts the LLM to perform reward decomposition using only the dialogue transcript. The second \emph{multimodal} variant incorporates additional behavioral cues, such as pitch, gaze, and facial affect, expressed as natural language descriptions. These inferred turn-level rewards are distilled into a lightweight reward model, which we utilize for RL-based fine-tuning for dialogue generation. We evaluate both text-only and multimodal variants against state-of-the-art reward decomposition methods and demonstrate notable improvements in human evaluations of conversation quality, suggesting that LLMs are strong reward decomposers that obviate the need for manual reward shaping and granular human feedback.

**arXiv ID:** 2505.15922
</details>

<details>
<summary><strong>SWE-AGI: Benchmarking Specification-Driven Software Construction with MoonBit in the Era of Autonomous Agents</strong> - Zhirui Zhang, Hongbo Zhang, Haoxiang Fei, Zhiyuan Bao, Yubin Chen, Zhengyu Lei, Ziyue Liu, Yixuan Sun, Mingkun Xiao, Zihang Ye, Yu Zhang, Hongcheng Zhu, Yuxiang Wen, Heung-Yeung Shum - [[pdf]](https://arxiv.org/pdf/2602.09447)</summary>

**Abstract:** Although large language models (LLMs) have demonstrated impressive coding capabilities, their ability to autonomously build production-scale software from explicit specifications remains an open question. We introduce SWE-AGI, an open-source benchmark for evaluating end-to-end, specification-driven construction of software systems written in MoonBit. SWE-AGI tasks require LLM-based agents to implement parsers, interpreters, binary decoders, and SAT solvers strictly from authoritative standards and RFCs under a fixed API scaffold. Each task involves implementing 1,000-10,000 lines of core logic, corresponding to weeks or months of engineering effort for an experienced human developer. By leveraging the nascent MoonBit ecosystem, SWE-AGI minimizes data leakage, forcing agents to rely on long-horizon architectural reasoning rather than code retrieval. Across frontier models, gpt-5.3-codex achieves the best overall performance (solving 19/22 tasks, 86.4%), outperforming claude-opus-4.6 (15/22, 68.2%), and kimi-2.5 exhibits the strongest performance among open-source models. Performance degrades sharply with increasing task difficulty, particularly on hard, specification-intensive systems. Behavioral analysis further reveals that as codebases scale, code reading, rather than writing, becomes the dominant bottleneck in AI-assisted development. Overall, while specification-driven autonomous software engineering is increasingly viable, substantial challenges remain before it can reliably support production-scale development.

**arXiv ID:** 2602.09447
</details>

<details>
<summary><strong>Actions Speak Louder Than Chats: Investigating AI Chatbot Age Gating</strong> - Olivia Figueira, Pranathi Chamarthi, Tu Le, Athina Markopoulou - [[pdf]](https://arxiv.org/pdf/2602.10251)</summary>

**Abstract:** AI chatbots are widely used by children and teens today, but they pose significant risks to youth's privacy and safety due to both increasingly personal conversations and potential exposure to unsafe content. While children under 13 are protected by the Children's Online Privacy Protection Act (COPPA), chatbot providers' own privacy policies may also provide protections, since they typically prohibit children from accessing their platforms. Age gating is often employed to restrict children online, but chatbot age gating in particular has not been studied. In this paper, we investigate whether popular consumer chatbots are (i) able to estimate users' ages based solely on their conversations, and (ii) whether they take action upon identifying children. To that end, we develop an auditing framework in which we programmatically interact with chatbots and conduct 1050 experiments using our comprehensive library of age-indicative prompts, including implicit and explicit age disclosures, to analyze the chatbots' responses and actions. We find that while chatbots are capable of estimating age, they do not take any action when children are identified, contradicting their own policies. Our methodology and findings provide insights for platform design, demonstrated by our proof-of-concept chatbot age gating implementation, and regulation to protect children online.

**arXiv ID:** 2602.10251
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (17 papers)</h2></summary>

<details>
<summary><strong>See, Plan, Snap: Evaluating Multimodal GUI Agents in Scratch</strong> - Xingyi Zhang, Yulei Ye, Kaifeng Huang, Wenhao Li, Xiangfeng Wang - [[pdf]](https://arxiv.org/pdf/2602.10814)</summary>

**Abstract:** Block-based programming environments such as Scratch play a central role in low-code education, yet evaluating the capabilities of AI agents to construct programs through Graphical User Interfaces (GUIs) remains underexplored. We introduce ScratchWorld, a benchmark for evaluating multimodal GUI agents on program-by-construction tasks in Scratch. Grounded in the Use-Modify-Create pedagogical framework, ScratchWorld comprises 83 curated tasks spanning four distinct problem categories: Create, Debug, Extend, and Compute. To rigorously diagnose the source of agent failures, the benchmark employs two complementary interaction modes: primitive mode requires fine-grained drag-and-drop manipulation to directly assess visuomotor control, while composite mode uses high-level semantic APIs to disentangle program reasoning from GUI execution. To ensure reliable assessment, we propose an execution-based evaluation protocol that validates the functional correctness of the constructed Scratch programs through runtime tests within the browser environment. Extensive experiments across state-of-the-art multimodal language models and GUI agents reveal a substantial reasoning--acting gap, highlighting persistent challenges in fine-grained GUI manipulation despite strong planning capabilities.

**arXiv ID:** 2602.10814
</details>

<details>
<summary><strong>FormalJudge: A Neuro-Symbolic Paradigm for Agentic Oversight</strong> - Jiayi Zhou, Yang Sheng, Hantao Lou, Yaodong Yang, Jie Fu - [[pdf]](https://arxiv.org/pdf/2602.11136)</summary>

**Abstract:** As LLM-based agents increasingly operate in high-stakes domains with real-world consequences, ensuring their behavioral safety becomes paramount. The dominant oversight paradigm, LLM-as-a-Judge, faces a fundamental dilemma: how can probabilistic systems reliably supervise other probabilistic systems without inheriting their failure modes? We argue that formal verification offers a principled escape from this dilemma, yet its adoption has been hindered by a critical bottleneck: the translation from natural language requirements to formal specifications. This paper bridges this gap by proposing , a neuro-symbolic framework that employs a bidirectional Formal-of-Thought architecture: LLMs serve as specification compilers that top-down decompose high-level human intent into atomic, verifiable constraints, then bottom-up prove compliance using Dafny specifications and Z3 Satisfiability modulo theories solving, which produces mathematical guarantees rather than probabilistic scores. We validate across three benchmarks spanning behavioral safety, multi-domain constraint adherence, and agentic upward deception detection. Experiments on 7 agent models demonstrate that achieves an average improvement of 16.6% over LLM-as-a-Judge baselines, enables weak-to-strong generalization where a 7B judge achieves over 90% accuracy detecting deception from 72B agents, and provides near-linear safety improvement through iterative refinement.

**arXiv ID:** 2602.11136
</details>

<details>
<summary><strong>AD$^2$: Analysis and Detection of Adversarial Threats in Visual Perception for End-to-End Autonomous Driving Systems</strong> - Ishan Sahu, Somnath Hazra, Somak Aditya, Soumyajit Dey - [[pdf]](https://arxiv.org/pdf/2602.10160)</summary>

**Abstract:** End-to-end autonomous driving systems have achieved significant progress, yet their adversarial robustness remains largely underexplored. In this work, we conduct a closed-loop evaluation of state-of-the-art autonomous driving agents under black-box adversarial threat models in CARLA. Specifically, we consider three representative attack vectors on the visual perception pipeline: (i) a physics-based blur attack induced by acoustic waves, (ii) an electromagnetic interference attack that distorts captured images, and (iii) a digital attack that adds ghost objects as carefully crafted bounded perturbations on images. Our experiments on two advanced agents, Transfuser and Interfuser, reveal severe vulnerabilities to such attacks, with driving scores dropping by up to 99% in the worst case, raising valid safety concerns. To help mitigate such threats, we further propose a lightweight Attack Detection model for Autonomous Driving systems (AD$^2$) based on attention mechanisms that capture spatial-temporal consistency. Comprehensive experiments across multi-camera inputs on CARLA show that our detector achieves superior detection capability and computational efficiency compared to existing approaches.

**arXiv ID:** 2602.10160
</details>

<details>
<summary><strong>Towards Autonomous Mathematics Research</strong> - Tony Feng, Trieu H. Trinh, Garrett Bingham, Dawsen Hwang, Yuri Chervonyi, Junehyuk Jung, Joonkyung Lee, Carlo Pagano, Sang-hyun Kim, Federico Pasqualotto, Sergei Gukov, Jonathan N. Lee, Junsu Kim, Kaiying Hou, Golnaz Ghiasi, Yi Tay, YaGuang Li, Chenkai Kuang, Yuan Liu, Hanzhao, Evan Zheran Liu, Nigamaa Nayakanti, Xiaomeng Yang, Heng-tze Cheng, Demis Hassabis, Koray Kavukcuoglu, Quoc V. Le, Thang Luong - [[pdf]](https://arxiv.org/pdf/2602.10177)</summary>

**Abstract:** Recent advances in foundational models have yielded reasoning systems capable of achieving a gold-medal standard at the International Mathematical Olympiad. The transition from competition-level problem-solving to professional research, however, requires navigating vast literature and constructing long-horizon proofs. In this work, we introduce Aletheia, a math research agent that iteratively generates, verifies, and revises solutions end-to-end in natural language. Specifically, Aletheia is powered by an advanced version of Gemini Deep Think for challenging reasoning problems, a novel inference-time scaling law that extends beyond Olympiad-level problems, and intensive tool use to navigate the complexities of mathematical research. We demonstrate the capability of Aletheia from Olympiad problems to PhD-level exercises and most notably, through several distinct milestones in AI-assisted mathematics research: (a) a research paper (Feng26) generated by AI without any human intervention in calculating certain structure constants in arithmetic geometry called eigenweights; (b) a research paper (LeeSeo26) demonstrating human-AI collaboration in proving bounds on systems of interacting particles called independent sets; and (c) an extensive semi-autonomous evaluation (Feng et al., 2026a) of 700 open problems on Bloom's Erdos Conjectures database, including autonomous solutions to four open questions. In order to help the public better understand the developments pertaining to AI and mathematics, we suggest codifying standard levels quantifying autonomy and novelty of AI-assisted results. We conclude with reflections on human-AI collaboration in mathematics.

**arXiv ID:** 2602.10177
</details>

<details>
<summary><strong>Enhancing Predictability of Multi-Tenant DNN Inference for Autonomous Vehicles' Perception</strong> - Liangkai Liu, Kang G. Shin, Jinkyu Lee, Chengmo Yang, Weisong Shi - [[pdf]](https://arxiv.org/pdf/2602.11004)</summary>

**Abstract:** Autonomous vehicles (AVs) rely on sensors and deep neural networks (DNNs) to perceive their surrounding environment and make maneuver decisions in real time. However, achieving real-time DNN inference in the AV's perception pipeline is challenging due to the large gap between the computation requirement and the AV's limited resources. Most, if not all, of existing studies focus on optimizing the DNN inference time to achieve faster perception by compressing the DNN model with pruning and quantization. In contrast, we present a Predictable Perception system with DNNs (PP-DNN) that reduce the amount of image data to be processed while maintaining the same level of accuracy for multi-tenant DNNs by dynamically selecting critical frames and regions of interest (ROIs). PP-DNN is based on our key insight that critical frames and ROIs for AVs vary with the AV's surrounding environment. However, it is challenging to identify and use critical frames and ROIs in multi-tenant DNNs for predictable inference. Given image-frame streams, PP-DNN leverages an ROI generator to identify critical frames and ROIs based on the similarities of consecutive frames and traffic scenarios. PP-DNN then leverages a FLOPs predictor to predict multiply-accumulate operations (MACs) from the dynamic critical frames and ROIs. The ROI scheduler coordinates the processing of critical frames and ROIs with multiple DNN models. Finally, we design a detection predictor for the perception of non-critical frames. We have implemented PP-DNN in an ROS-based AV pipeline and evaluated it with the BDD100K and the nuScenes dataset. PP-DNN is observed to significantly enhance perception predictability, increasing the number of fusion frames by up to 7.3x, reducing the fusion delay by >2.6x and fusion-delay variations by >2.3x, improving detection completeness by 75.4% and the cost-effectiveness by up to 98% over the baseline.

**arXiv ID:** 2602.11004
</details>

<details>
<summary><strong>Learning to Compose for Cross-domain Agentic Workflow Generation</strong> - Jialiang Wang, Shengxiang Xu, Hanmo Liu, Jiachuan Wang, Yuyu Luo, Shimin Di, Min-Ling Zhang, Lei Chen - [[pdf]](https://arxiv.org/pdf/2602.11114)</summary>

**Abstract:** Automatically generating agentic workflows -- executable operator graphs or codes that orchestrate reasoning, verification, and repair -- has become a practical way to solve complex tasks beyond what single-pass LLM generation can reliably handle. Yet what constitutes a good workflow depends heavily on the task distribution and the available operators. Under domain shift, current systems typically rely on iterative workflow refinement to discover a feasible workflow from a large workflow space, incurring high iteration costs and yielding unstable, domain-specific behavior. In response, we internalize a decompose-recompose-decide mechanism into an open-source LLM for cross-domain workflow generation. To decompose, we learn a compact set of reusable workflow capabilities across diverse domains. To recompose, we map each input task to a sparse composition over these bases to generate a task-specific workflow in a single pass. To decide, we attribute the success or failure of workflow generation to counterfactual contributions from learned capabilities, thereby capturing which capabilities actually drive success by their marginal effects. Across stringent multi-domain, cross-domain, and unseen-domain evaluations, our 1-pass generator surpasses SOTA refinement baselines that consume 20 iterations, while substantially reducing generation latency and cost.

**arXiv ID:** 2602.11114
</details>

<details>
<summary><strong>CostNav: A Navigation Benchmark for Real-World Economic-Cost Evaluation of Physical AI Agents</strong> - Haebin Seong, Sungmin Kim, Yongjun Cho, Myunchul Joe, Geunwoo Kim, Yubeen Park, Sunhoo Kim, Yoonshik Kim, Suhwan Choi, Jaeyoon Jung, Jiyong Youn, Jinmyung Kwak, Sunghee Ahn, Jaemin Lee, Younggil Do, Seungyeop Yi, Woojin Cheong, Minhyeok Oh, Minchan Kim, Seongjae Kang, Samwoo Seong, Youngjae Yu, Yunsung Lee - [[pdf]](https://arxiv.org/pdf/2511.20216)</summary>

**Abstract:** While current navigation benchmarks prioritize task success in simplified settings, they neglect the multidimensional economic constraints essential for the real-world commercialization of autonomous delivery systems. We introduce CostNav, an Economic Navigation Benchmark that evaluates physical AI agents through comprehensive economic cost-revenue analysis aligned with real-world business operations. By integrating industry-standard data - such as SEC filings and AIS injury reports - with Isaac Sim's detailed collision and cargo dynamics, CostNav transcends simple task completion to accurately evaluate business value in complex, real-world scenarios. To our knowledge, CostNav is the first work to quantitatively expose the gap between navigation research metrics and commercial viability, revealing that optimizing for task success on a simplified task fundamentally differs from optimizing for real-world economic deployment. Our evaluation of rule-based Nav2 navigation shows that current approaches are not economically viable: the contribution margin is -22.81/run (AMCL) and -12.87/run (GPS), resulting in no break-even point. We challenge the community to develop navigation policies that achieve economic viability on CostNav. We remain method-agnostic, evaluating success solely on the metric of cost rather than the underlying architecture. All resources are available at this https URL.

**arXiv ID:** 2511.20216
</details>

<details>
<summary><strong>Active Evaluation of General Agents: Problem Definition and Comparison of Baseline Algorithms</strong> - Marc Lanctot, Kate Larson, Ian Gemp, Michael Kaisers - [[pdf]](https://arxiv.org/pdf/2601.07651)</summary>

**Abstract:** As intelligent agents become more generally-capable, i.e. able to master a wide variety of tasks, the complexity and cost of properly evaluating them rises significantly. Tasks that assess specific capabilities of the agents can be correlated and stochastic, requiring many samples for accurate comparisons, leading to added costs. In this paper, we propose a formal definition and a conceptual framework for active evaluation of agents across multiple tasks, which assesses the performance of ranking algorithms as a function of number of evaluation data samples. Rather than curating, filtering, or compressing existing data sets as a preprocessing step, we propose an online framing: on every iteration, the ranking algorithm chooses the task and agents to sample scores from. Then, evaluation algorithms report a ranking of agents on each iteration and their performance is assessed with respect to the ground truth ranking over time. Several baselines are compared under different experimental contexts, with synthetic generated data and simulated online access to real evaluation data from Atari game-playing agents. We find that the classical Elo rating system -- while it suffers from well-known failure modes, in theory -- is a consistently reliable choice for efficient reduction of ranking error in practice. A recently-proposed method, Soft Condorcet Optimization, shows comparable performance to Elo on synthetic data and significantly outperforms Elo on real Atari agent evaluation. When task variation from the ground truth is high, selecting tasks based on proportional representation leads to higher rate of ranking error reduction.

**arXiv ID:** 2601.07651
</details>

<details>
<summary><strong>Meta Context Engineering via Agentic Skill Evolution</strong> - Haoran Ye, Xuning He, Vincent Arak, Haonan Dong, Guojie Song - [[pdf]](https://arxiv.org/pdf/2601.21557)</summary>

**Abstract:** The operational efficacy of large language models relies heavily on their inference-time context. This has established Context Engineering (CE) as a formal discipline for optimizing these inputs. Current CE methods rely on manually crafted harnesses, such as rigid generation-reflection workflows and predefined context schemas. They impose structural biases and restrict context optimization to a narrow, intuition-bound design space. To address this, we introduce Meta Context Engineering (MCE), a bi-level framework that supersedes static CE heuristics by co-evolving CE skills and context artifacts. In MCE iterations, a meta-level agent refines engineering skills via agentic crossover, a deliberative search over the history of skills, their executions, and evaluations. A base-level agent executes these skills, learns from training rollouts, and optimizes context as flexible files and code. We evaluate MCE across five disparate domains under offline and online settings. MCE demonstrates consistent performance gains, achieving 5.6--53.8% relative improvement over state-of-the-art agentic CE methods (mean of 16.9%), while maintaining superior context adaptability, transferability, and efficiency in both context usage and training.

**arXiv ID:** 2601.21557
</details>

<details>
<summary><strong>From Assistant to Double Agent: Formalizing and Benchmarking Attacks on OpenClaw for Personalized Local AI Agent</strong> - Yuhang Wang, Feiming Xu, Zheng Lin, Guangyu He, Yuzhe Huang, Haichang Gao, Zhenxing Niu, Shiguo Lian, Zhaoxiang Liu - [[pdf]](https://arxiv.org/pdf/2602.08412)</summary>

**Abstract:** Although large language model (LLM)-based agents, exemplified by OpenClaw, are increasingly evolving from task-oriented systems into personalized AI assistants for solving complex real-world tasks, their practical deployment also introduces severe security risks. However, existing agent security research and evaluation frameworks primarily focus on synthetic or task-centric settings, and thus fail to accurately capture the attack surface and risk propagation mechanisms of personalized agents in real-world deployments. To address this gap, we propose Personalized Agent Security Bench (PASB), an end-to-end security evaluation framework tailored for real-world personalized agents. Building upon existing agent attack paradigms, PASB incorporates personalized usage scenarios, realistic toolchains, and long-horizon interactions, enabling black-box, end-to-end security evaluation on real systems. Using OpenClaw as a representative case study, we systematically evaluate its security across multiple personalized scenarios, tool capabilities, and attack types. Our results indicate that OpenClaw exhibits critical vulnerabilities at different execution stages, including user prompt processing, tool usage, and memory retrieval, highlighting substantial security risks in personalized agent deployments. The code for the proposed PASB framework is available at this https URL.

**arXiv ID:** 2602.08412
</details>

<details>
<summary><strong>AI Agentic Vulnerability Injection And Transformation with Optimized Reasoning</strong> - Amine Lbath, Massih-Reza Amini, Aurelien Delaitre, Vadim Okun - [[pdf]](https://arxiv.org/pdf/2508.20866)</summary>

**Abstract:** The increasing complexity of software systems and the sophistication of cyber-attacks have underscored the need for reliable automated software vulnerability detection. Data-driven approaches using deep learning models show promise but critically depend on the availability of large, accurately labeled datasets. Yet existing datasets either suffer from noisy labels, limited vulnerability coverage, or fail to reflect vulnerabilities as they occur in real-world software. This also limits large-scale benchmarking of such solutions. Automated vulnerability injection provides a way to address these limitations, but existing techniques remain limited in coverage, contextual fidelity, or injection success. In this paper, we present AVIATOR, the first AI-agentic vulnerability injection framework. AVIATOR decomposes vulnerability injection into a coordinated workflow of specialized AI agents, tool-based analysis, and iterative self-correction, explicitly mirroring expert reasoning. It integrates RAG and lightweight LoRA-based fine-tuning to produce realistic, category-specific vulnerabilities without relying on handcrafted patterns. Across three benchmarks, AVIATOR achieves high injection fidelity (91-95%) surpassing existing injection techniques in both accuracy and vulnerability coverage. When used for data augmentation to train deep learning-based vulnerability detection (DLVD) models, AVIATOR provides the strongest downstream gains in vulnerability detection. Across models and base datasets, AVIATOR improves average F1 scores by +22% over no augmentation, +25% over VGX, holding the prior best injection success rate, and +3% over VulScribeR, the prior state-of-the-art LLM-based injection model, with +7% higher recall and no precision loss. Its augmented data exhibits the lowest distributional distortion and scales efficiently with <2% syntax rejection at 4.3x lower cost than VulScribeR.

**arXiv ID:** 2508.20866
</details>

<details>
<summary><strong>ISD-Agent-Bench: A Comprehensive Benchmark for Evaluating LLM-based Instructional Design Agents</strong> - YoungHoon Jeon, Suwan Kim, Haein Son, Sookbun Lee, Yeil Jeong, Unggi Lee - [[pdf]](https://arxiv.org/pdf/2602.10620)</summary>

**Abstract:** Large Language Model (LLM) agents have shown promising potential in automating Instructional Systems Design (ISD), a systematic approach to developing educational programs. However, evaluating these agents remains challenging due to the lack of standardized benchmarks and the risk of LLM-as-judge bias. We present ISD-Agent-Bench, a comprehensive benchmark comprising 25,795 scenarios generated via a Context Matrix framework that combines 51 contextual variables across 5 categories with 33 ISD sub-steps derived from the ADDIE model. To ensure evaluation reliability, we employ a multi-judge protocol using diverse LLMs from different providers, achieving high inter-judge reliability. We compare existing ISD agents with novel agents grounded in classical ISD theories such as ADDIE, Dick \& Carey, and Rapid Prototyping ISD. Experiments on 1,017 test scenarios demonstrate that integrating classical ISD frameworks with modern ReAct-style reasoning achieves the highest performance, outperforming both pure theory-based agents and technique-only approaches. Further analysis reveals that theoretical quality strongly correlates with benchmark performance, with theory-based agents showing significant advantages in problem-centered design and objective-assessment alignment. Our work provides a foundation for systematic LLM-based ISD research.

**arXiv ID:** 2602.10620
</details>

<details>
<summary><strong>Beyond Closed-Pool Video Retrieval: A Benchmark and Agent Framework for Real-World Video Search and Moment Localization</strong> - Tao Yu, Yujia Yang, Haopeng Jin, Junhao Gong, Xinlong Chen, Yuxuan Zhou, Shanbin Zhang, Jiabing Yang, Xinming Wang, Hongzhu Yi, Ping Nie, Kai Zou, Zhang Zhang, Yan Huang, Liang Wang, Yeshani, Ruiwen Tao, Jin Ma, Haijin Liang, Jinwen Luo - [[pdf]](https://arxiv.org/pdf/2602.10159)</summary>

**Abstract:** Traditional video retrieval benchmarks focus on matching precise descriptions to closed video pools, failing to reflect real-world searches characterized by fuzzy, multi-dimensional memories on the open web. We present \textbf{RVMS-Bench}, a comprehensive system for evaluating real-world video memory search. It consists of \textbf{1,440 samples} spanning \textbf{20 diverse categories} and \textbf{four duration groups}, sourced from \textbf{real-world open-web videos}. RVMS-Bench utilizes a hierarchical description framework encompassing \textbf{Global Impression, Key Moment, Temporal Context, and Auditory Memory} to mimic realistic multi-dimensional search cues, with all samples strictly verified via a human-in-the-loop protocol. We further propose \textbf{RACLO}, an agentic framework that employs abductive reasoning to simulate the human ``Recall-Search-Verify'' cognitive process, effectively addressing the challenge of searching for videos via fuzzy memories in the real world. Experiments reveal that existing MLLMs still demonstrate insufficient capabilities in real-world Video Retrieval and Moment Localization based on fuzzy memories. We believe this work will facilitate the advancement of video retrieval robustness in real-world unstructured scenarios.

**arXiv ID:** 2602.10159
</details>

<details>
<summary><strong>ACE-RTL: When Agentic Context Evolution Meets RTL-Specialized LLMs</strong> - Chenhui Deng, Zhongzhi Yu, Guan-Ting Liu, Nathaniel Pinckney, Haoxing Ren - [[pdf]](https://arxiv.org/pdf/2602.10218)</summary>

**Abstract:** Recent advances in large language models (LLMs) have sparked growing interest in applying them to hardware design automation, particularly for accurate RTL code generation. Prior efforts follow two largely independent paths: (i) training domain-adapted RTL models to internalize hardware semantics, (ii) developing agentic systems that leverage frontier generic LLMs guided by simulation feedback. However, these two paths exhibit complementary strengths and weaknesses. In this work, we present ACE-RTL that unifies both directions through Agentic Context Evolution (ACE). ACE-RTL integrates an RTL-specialized LLM, trained on a large-scale dataset of 1.7 million RTL samples, with a frontier reasoning LLM through three synergistic components: the generator, reflector, and coordinator. These components iteratively refine RTL code toward functional correctness. We further introduce a parallel scaling strategy that significantly reduces the number of iterations required to reach correct solutions. On the Comprehensive Verilog Design Problems (CVDP) benchmark, ACE-RTL achieves up to a 44.87% pass rate improvement over 14 competitive baselines while requiring only four iterations on average.

**arXiv ID:** 2602.10218
</details>

<details>
<summary><strong>Adaptive Time Step Flow Matching for Autonomous Driving Motion Planning</strong> - Ananya Trivedi, Anjian Li, Mohamed Elnoor, Yusuf Umut Ciftci, Avinash Singh, Jovin D'sa, Sangjae Bae, David Isele, Taskin Padir, Faizan M. Tariq - [[pdf]](https://arxiv.org/pdf/2602.10285)</summary>

**Abstract:** Autonomous driving requires reasoning about interactions with surrounding traffic. A prevailing approach is large-scale imitation learning on expert driving datasets, aimed at generalizing across diverse real-world scenarios. For online trajectory generation, such methods must operate at real-time rates. Diffusion models require hundreds of denoising steps at inference, resulting in high latency. Consistency models mitigate this issue but rely on carefully tuned noise schedules to capture the multimodal action distributions common in autonomous driving. Adapting the schedule, typically requires expensive retraining. To address these limitations, we propose a framework based on conditional flow matching that jointly predicts future motions of surrounding agents and plans the ego trajectory in real time. We train a lightweight variance estimator that selects the number of inference steps online, removing the need for retraining to balance runtime and imitation learning performance. To further enhance ride quality, we introduce a trajectory post-processing step cast as a convex quadratic program, with negligible computational overhead. Trained on the Waymo Open Motion Dataset, the framework performs maneuvers such as lane changes, cruise control, and navigating unprotected left turns without requiring scenario-specific tuning. Our method maintains a 20 Hz update rate on an NVIDIA RTX 3070 GPU, making it suitable for online deployment. Compared to transformer, diffusion, and consistency model baselines, we achieve improved trajectory smoothness and better adherence to dynamic constraints. Experiment videos and code implementations can be found at this https URL.

**arXiv ID:** 2602.10285
</details>

<details>
<summary><strong>RADAR: Benchmarking Vision-Language-Action Generalization via Real-World Dynamics, Spatial-Physical Intelligence, and Autonomous Evaluation</strong> - Yuhao Chen, Zhihao Zhan, Xiaoxin Lin, Zijian Song, Hao Liu, Qinhan Lyu, Yubo Zu, Xiao Chen, Zhiyuan Liu, Tao Pu, Tianshui Chen, Keze Wang, Liang Lin, Guangrun Wang - [[pdf]](https://arxiv.org/pdf/2602.10980)</summary>

**Abstract:** VLA models have achieved remarkable progress in embodied intelligence; however, their evaluation remains largely confined to simulations or highly constrained real-world settings. This mismatch creates a substantial reality gap, where strong benchmark performance often masks poor generalization in diverse physical environments. We identify three systemic shortcomings in current benchmarking practices that hinder fair and reliable model comparison. (1) Existing benchmarks fail to model real-world dynamics, overlooking critical factors such as dynamic object configurations, robot initial states, lighting changes, and sensor noise. (2) Current protocols neglect spatial--physical intelligence, reducing evaluation to rote manipulation tasks that do not probe geometric reasoning. (3) The field lacks scalable fully autonomous evaluation, instead relying on simplistic 2D metrics that miss 3D spatial structure or on human-in-the-loop systems that are costly, biased, and unscalable. To address these limitations, we introduce RADAR (Real-world Autonomous Dynamics And Reasoning), a benchmark designed to systematically evaluate VLA generalization under realistic conditions. RADAR integrates three core components: (1) a principled suite of physical dynamics; (2) dedicated tasks that explicitly test spatial reasoning and physical understanding; and (3) a fully autonomous evaluation pipeline based on 3D metrics, eliminating the need for human supervision. We apply RADAR to audit multiple state-of-the-art VLA models and uncover severe fragility beneath their apparent competence. Performance drops precipitously under modest physical dynamics, with the expectation of 3D IoU declining from 0.261 to 0.068 under sensor noise. Moreover, models exhibit limited spatial reasoning capability. These findings position RADAR as a necessary bench toward reliable and generalizable real-world evaluation of VLA models.

**arXiv ID:** 2602.10980
</details>

<details>
<summary><strong>WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models</strong> - Yu Shang, Zhuohang Li, Yiding Ma, Weikang Su, Xin Jin, Ziyou Wang, Lei Jin, Xin Zhang, Yinzhou Tang, Haisheng Su, Chen Gao, Wei Wu, Xihui Liu, Dhruv Shah, Zhaoxiang Zhang, Zhibo Chen, Jun Zhu, Yonghong Tian, Tat-Seng Chua, Wenwu Zhu, Yong Li - [[pdf]](https://arxiv.org/pdf/2602.08971)</summary>

**Abstract:** While world models have emerged as a cornerstone of embodied intelligence by enabling agents to reason about environmental dynamics through action-conditioned prediction, their evaluation remains fragmented. Current evaluation of embodied world models has largely focused on perceptual fidelity (e.g., video generation quality), overlooking the functional utility of these models in downstream decision-making tasks. In this work, we introduce WorldArena, a unified benchmark designed to systematically evaluate embodied world models across both perceptual and functional dimensions. WorldArena assesses models through three dimensions: video perception quality, measured with 16 metrics across six sub-dimensions; embodied task functionality, which evaluates world models as data engines, policy evaluators, and action planners integrating with subjective human evaluation. Furthermore, we propose EWMScore, a holistic metric integrating multi-dimensional performance into a single interpretable index. Through extensive experiments on 14 representative models, we reveal a significant perception-functionality gap, showing that high visual quality does not necessarily translate into strong embodied task capability. WorldArena benchmark with the public leaderboard is released at this https URL, providing a framework for tracking progress toward truly functional world models in embodied AI.

**arXiv ID:** 2602.08971
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>AgentTrace: A Structured Logging Framework for Agent System Observability</strong> - Adam AlSayyad, Kelvin Yuxiang Huang, Richik Pal - [[pdf]](https://arxiv.org/pdf/2602.10133)</summary>

**Abstract:** Despite the growing capabilities of autonomous agents powered by large language models (LLMs), their adoption in high-stakes domains remains limited. A key barrier is security: the inherently nondeterministic behavior of LLM agents defies static auditing approaches that have historically underpinned software assurance. Existing security methods, such as proxy-level input filtering and model glassboxing, fail to provide sufficient transparency or traceability into agent reasoning, state changes, or environmental interactions. In this work, we introduce AgentTrace, a dynamic observability and telemetry framework designed to fill this gap. AgentTrace instruments agents at runtime with minimal overhead, capturing a rich stream of structured logs across three surfaces: operational, cognitive, and contextual. Unlike traditional logging systems, AgentTrace emphasizes continuous, introspectable trace capture, designed not just for debugging or benchmarking, but as a foundational layer for agent security, accountability, and real-time monitoring. Our research highlights how AgentTrace can enable more reliable agent deployment, fine-grained risk analysis, and informed trust calibration, thereby addressing critical concerns that have so far limited the use of LLM agents in sensitive environments.

**arXiv ID:** 2602.10133
</details>

<details>
<summary><strong>Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents</strong> - Haochen Wang, Yi Wu, Daryl Chang, Li Wei, Lukasz Heldt - [[pdf]](https://arxiv.org/pdf/2602.10226)</summary>

**Abstract:** Optimizing large-scale machine learning systems, such as recommendation models for global video platforms, requires navigating a massive hyperparameter search space and, more critically, designing sophisticated optimizers, architectures, and reward functions to capture nuanced user behaviors. Achieving substantial improvements in these areas is a non-trivial task, traditionally relying on extensive manual iterations to test new hypotheses. We propose a self-evolving system that leverages Large Language Models (LLMs), specifically those from Google's Gemini family, to autonomously generate, train, and deploy high-performing, complex model changes within an end-to-end automated workflow. The self-evolving system is comprised of an Offline Agent (Inner Loop) that performs high-throughput hypothesis generation using proxy metrics, and an Online Agent (Outer Loop) that validates candidates against delayed north star business metrics in live production. Our agents act as specialized Machine Learning Engineers (MLEs): they exhibit deep reasoning capabilities, discovering novel improvements in optimization algorithms and model architecture, and formulating innovative reward functions that target long-term user engagement. The effectiveness of this approach is demonstrated through several successful production launches at YouTube, confirming that autonomous, LLM-driven evolution can surpass traditional engineering workflows in both development velocity and model performance.

**arXiv ID:** 2602.10226
</details>

<details>
<summary><strong>Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents</strong> - Yifei Li, Weidong Guo, Lingling Zhang, Rongman Xu, Muye Huang, Hui Liu, Lijiao Xu, Yu Xu, Jun Liu - [[pdf]](https://arxiv.org/pdf/2602.10715)</summary>

**Abstract:** Long-term conversational memory is a core capability for LLM-based dialogue systems, yet existing benchmarks and evaluation protocols primarily focus on surface-level factual recall. In realistic interactions, appropriate responses often depend on implicit constraints such as user state, goals, or values that are not explicitly queried later. To evaluate this setting, we introduce \textbf{LoCoMo-Plus}, a benchmark for assessing cognitive memory under cue--trigger semantic disconnect, where models must retain and apply latent constraints across long conversational contexts. We further show that conventional string-matching metrics and explicit task-type prompting are misaligned with such scenarios, and propose a unified evaluation framework based on constraint consistency. Experiments across diverse backbone models, retrieval-based methods, and memory systems demonstrate that cognitive memory remains challenging and reveals failures not captured by existing benchmarks. Our code and evaluation framework are publicly available at: this https URL.

**arXiv ID:** 2602.10715
</details>

<details>
<summary><strong>The Landscape of Prompt Injection Threats in LLM Agents: From Taxonomy to Analysis</strong> - Peiran Wang, Xinfeng Li, Chong Xiang, Jinghuai Zhang, Ying Li, Lixia Zhang, Xiaofeng Wang, Yuan Tian - [[pdf]](https://arxiv.org/pdf/2602.10453)</summary>

**Abstract:** The evolution of Large Language Models (LLMs) has resulted in a paradigm shift towards autonomous agents, necessitating robust security against Prompt Injection (PI) vulnerabilities where untrusted inputs hijack agent behaviors. This SoK presents a comprehensive overview of the PI landscape, covering attacks, defenses, and their evaluation practices. Through a systematic literature review and quantitative analysis, we establish taxonomies that categorize PI attacks by payload generation strategies (heuristic vs. optimization) and defenses by intervention stages (text, model, and execution levels). Our analysis reveals a key limitation shared by many existing defenses and benchmarks: they largely overlook context-dependent tasks, in which agents are authorized to rely on runtime environmental observations to determine actions. To address this gap, we introduce AgentPI, a new benchmark designed to systematically evaluate agent behavior under context-dependent interaction settings. Using AgentPI, we empirically evaluate representative defenses and show that no single approach can simultaneously achieve high trustworthiness, high utility, and low latency. Moreover, we show that many defenses appear effective under existing benchmarks by suppressing contextual inputs, yet fail to generalize to realistic agent settings where context-dependent reasoning is essential. This SoK distills key takeaways and open research problems, offering structured guidance for future research and practical deployment of secure LLM agents.

**arXiv ID:** 2602.10453
</details>

<details>
<summary><strong>Polymer-Agent: Large Language Model Agent for Polymer Design</strong> - Vani Nigam, Achuth Chandrasekhar, Amir Barati Farimani - [[pdf]](https://arxiv.org/pdf/2601.16376)</summary>

**Abstract:** On-demand Polymer discovery is essential for various industries, ranging from biomedical to reinforcement materials. Experiments with polymers have a long trial-and-error process, leading to use of extensive resources. For these processes, machine learning has accelerated scientific discovery at the property prediction and latent space search fronts. However, laboratory researchers cannot readily access codes and these models to extract individual structures and properties due to infrastructure limitations. We present a closed-loop polymer structure-property predictor integrated in a terminal for early-stage polymer discovery. The framework is powered by LLM reasoning to provide users with property prediction, property-guided polymer structure generation, and structure modification capabilities. The SMILES sequences are guided by the synthetic accessibility score and the synthetic complexity score (SC Score) to ensure that polymer generation is as close as possible to synthetically accessible monomer-level structures. This framework addresses the challenge of generating novel polymer structures for laboratory researchers, thereby providing computational insights into polymer research.

**arXiv ID:** 2601.16376
</details>

<details>
<summary><strong>TVCACHE: A Stateful Tool-Value Cache for Post-Training LLM Agents</strong> - Abhishek Vijaya Kumar, Bhaskar Kataria, Byungsoo Oh, Emaad Manzoor, Rachee Singh - [[pdf]](https://arxiv.org/pdf/2602.10986)</summary>

**Abstract:** In RL post-training of LLM agents, calls to external tools take several seconds or even minutes, leaving allocated GPUs idle and inflating post-training time and cost. While many tool invocations repeat across parallel rollouts and could in principle be cached, naively caching their outputs for reuse is incorrect since tool outputs depend on the environment state induced by prior agent interactions. We present TVCACHE, a stateful tool-value cache for LLM agent post-training. TVCACHE maintains a tree of observed tool-call sequences and performs longest-prefix matching for cache lookups: a hit occurs only when the agent's full tool history matches a previously executed sequence, guaranteeing identical environment state. On three diverse workloads-terminal-based tasks, SQL generation, and video understanding. TVCACHE achieves cache hit rates of up to 70% and reduces median tool call execution time by up to 6.9X, with no degradation in post-training reward accumulation.

**arXiv ID:** 2602.10986
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (15 papers)</h2></summary>

<details>
<summary><strong>Co-jump: Cooperative Jumping with Quadrupedal Robots via Multi-Agent Reinforcement Learning</strong> - Shihao Dong, Yeke Chen, Zeren Luo, Jiahui Zhang, Bowen Xu, Jinghan Lin, Yimin Han, Ji Ma, Zhiyou Yu, Yudong Zhao, Peng Lu - [[pdf]](https://arxiv.org/pdf/2602.10514)</summary>

**Abstract:** While single-agent legged locomotion has witnessed remarkable progress, individual robots remain fundamentally constrained by physical actuation limits. To transcend these boundaries, we introduce Co-jump, a cooperative task where two quadrupedal robots synchronize to execute jumps far beyond their solo capabilities. We tackle the high-impulse contact dynamics of this task under a decentralized setting, achieving synchronization without explicit communication or pre-specified motion primitives. Our framework leverages Multi-Agent Proximal Policy Optimization (MAPPO) enhanced by a progressive curriculum strategy, which effectively overcomes the sparse-reward exploration challenges inherent in mechanically coupled systems. We demonstrate robust performance in simulation and successful transfer to physical hardware, executing multi-directional jumps onto platforms up to 1.5 m in height. Specifically, one of the robots achieves a foot-end elevation of 1.1 m, which represents a 144% improvement over the 0.45 m jump height of a standalone quadrupedal robot, demonstrating superior vertical performance. Notably, this precise coordination is achieved solely through proprioceptive feedback, establishing a foundation for communication-free collaborative locomotion in constrained environments.

**arXiv ID:** 2602.10514
</details>

<details>
<summary><strong>Interpretable Attention-Based Multi-Agent PPO for Latency Spike Resolution in 6G RAN Slicing</strong> - Kavan Fatehi, Mostafa Rahmani Ghourtani, Amir Sonee, Poonam Yadav, Alessandra M Russo, Hamed Ahmadi, Radu Calinescu - [[pdf]](https://arxiv.org/pdf/2602.11076)</summary>

**Abstract:** Sixth-generation (6G) radio access networks (RANs) must enforce strict service-level agreements (SLAs) for heterogeneous slices, yet sudden latency spikes remain difficult to diagnose and resolve with conventional deep reinforcement learning (DRL) or explainable RL (XRL). We propose \emph{Attention-Enhanced Multi-Agent Proximal Policy Optimization (AE-MAPPO)}, which integrates six specialized attention mechanisms into multi-agent slice control and surfaces them as zero-cost, faithful explanations. The framework operates across O-RAN timescales with a three-phase strategy: predictive, reactive, and inter-slice optimization.
A URLLC case study shows AE-MAPPO resolves a latency spike in $18$ms, restores latency to $0.98$ms with $99.9999\%$ reliability, and reduces troubleshooting time by $93\%$ while maintaining eMBB and mMTC continuity. These results confirm AE-MAPPO's ability to combine SLA compliance with inherent interpretability, enabling trustworthy and real-time automation for 6G RAN slicing.

**arXiv ID:** 2602.11076
</details>

<details>
<summary><strong>Reinforcement Learning in Strategy-Based and Atari Games: A Review of Google DeepMinds Innovations</strong> - Abdelrhman Shaheen, Anas Badr, Ali Abohendy, Hatem Alsaadawy, Nadine Alsayad, Ehab H. El-Shazly - [[pdf]](https://arxiv.org/pdf/2502.10303)</summary>

**Abstract:** Reinforcement Learning (RL) has been widely used in many applications, particularly in gaming, which serves as an excellent training ground for AI models. Google DeepMind has pioneered innovations in this field, employing reinforcement learning algorithms, including model-based, model-free, and deep Q-network approaches, to create advanced AI models such as AlphaGo, AlphaGo Zero, and MuZero. AlphaGo, the initial model, integrates supervised learning and reinforcement learning to master the game of Go, surpassing professional human players. AlphaGo Zero refines this approach by eliminating reliance on human gameplay data, instead utilizing self-play for enhanced learning efficiency. MuZero further extends these advancements by learning the underlying dynamics of game environments without explicit knowledge of the rules, achieving adaptability across various games, including complex Atari games. This paper reviews the significance of reinforcement learning applications in Atari and strategy-based games, analyzing these three models, their key innovations, training processes, challenges encountered, and improvements made. Additionally, we discuss advancements in the field of gaming, including MiniZero and multi-agent models, highlighting future directions and emerging AI models from Google DeepMind.

**arXiv ID:** 2502.10303
</details>

<details>
<summary><strong>Is Your LLM Really Mastering the Concept? A Multi-Agent Benchmark</strong> - Shuhang Xu, Weijian Deng, Yixuan Zhou, Fangwei Zhong - [[pdf]](https://arxiv.org/pdf/2505.17512)</summary>

**Abstract:** Concepts serve as fundamental abstractions that support human reasoning and categorization. However, it remains unclear whether large language models truly capture such conceptual structures or primarily rely on surface-level pattern memorization. Existing benchmarks are largely static and fact oriented, which limits their ability to probe fine-grained semantic understanding and makes them vulnerable to data leakage and overfitting. To address this limitation, we introduce CK-Arena, a dynamic benchmark for conceptual knowledge evaluation based on a multi agent social deduction game, namely the Undercover game. In this setting, LLM based agents are assigned subtly different concept words and must describe, distinguish, and infer conceptual properties from others' statements. Model performance is evaluated through both game level outcomes and the semantic quality of generated descriptions. Furthermore, CK-Arena leverages the interaction process to automatically construct high quality question answering data for fine grained diagnostic analysis. Experimental results show that conceptual understanding varies substantially across models and categories, and is not strictly aligned with overall model capability. The data and code are available at the project homepage: this https URL.

**arXiv ID:** 2505.17512
</details>

<details>
<summary><strong>Retrieval- and Argumentation-Enhanced Multi-Agent LLMs for Judgmental Forecasting (Extended Version with Supplementary Material)</strong> - Deniz Gorur, Antonio Rago, Francesca Toni - [[pdf]](https://arxiv.org/pdf/2510.24303)</summary>

**Abstract:** Judgmental forecasting is the task of making predictions about future events based on human judgment. This task can be seen as a form of claim verification, where the claim corresponds to a future event and the task is to assess the plausibility of that event. In this paper, we propose a novel multi-agent framework for claim verification, whereby different agents may disagree on claim veracity and bring specific evidence for and against the claims, represented as quantitative bipolar argumentation frameworks (QBAFs). We then instantiate the framework for supporting claim verification, with a variety of agents realised with Large Language Models (LLMs): (1) ArgLLM agents, an existing approach for claim verification that generates and evaluates QBAFs; (2) RbAM agents, whereby LLM-empowered Relation-based Argument Mining (RbAM) from external sources is used to generate QBAFs; (3) RAG-ArgLLM agents, extending ArgLLM agents with a form of Retrieval-Augmented Generation (RAG) of arguments from external sources. Finally, we conduct experiments with two standard judgmental forecasting datasets, with instances of our framework with two or three agents, empowered by six different base LLMs. We observe that combining evidence from agents can improve forecasting accuracy, especially in the case of three agents, while providing an explainable combination of evidence for claim verification.

**arXiv ID:** 2510.24303
</details>

<details>
<summary><strong>PieArena: Frontier Language Agents Achieve MBA-Level Negotiation Performance and Reveal Novel Behavioral Differences</strong> - Chris Zhu, Sasha Cui, Will Sanok Dufallo, Runzhi Jin, Zhen Xu, Linjun Zhang, Daylian Cain - [[pdf]](https://arxiv.org/pdf/2602.05302)</summary>

**Abstract:** We present an in-depth evaluation of LLMs' ability to negotiate, a central business task that requires strategic reasoning, theory of mind, and economic value creation. To do so, we introduce PieArena, a large-scale negotiation benchmark grounded in multi-agent interactions over realistic scenarios drawn from an MBA negotiation course at an elite business school. We develop a statistically grounded ranking model for continuous negotiation payoffs that produces leaderboards with principled confidence intervals and corrects for experimental asymmetries. We find systematic evidence of human-expert-level performance in which a representative frontier language agent (GPT-5) matches or outperforms trained business-school students, despite a semester of general negotiation instruction and targeted coaching immediately prior to the task. We further study the effects of joint-intentionality agentic scaffolding and observe asymmetric gains, with large improvements for mid- and lower-tier LMs and diminishing returns for frontier LMs. Beyond deal outcomes, PieArena provides a multi-dimensional negotiation behavioral profile, revealing novel cross-model heterogeneity, masked by deal-outcome-only benchmarks, in deception, computation accuracy, instruction compliance, and perceived reputation. Overall, our results suggest that frontier language agents are already intellectually and psychologically capable of deployment in high-stakes economic settings, but deficiencies in robustness and trustworthiness remain open challenges.

**arXiv ID:** 2602.05302
</details>

<details>
<summary><strong>SpotAgent: Grounding Visual Geo-localization in Large Vision-Language Models through Agentic Reasoning</strong> - Furong Jia, Ling Dai, Wenjin Deng, Fan Zhang, Chen Hu, Daxin Jiang, Yu Liu - [[pdf]](https://arxiv.org/pdf/2602.09463)</summary>

**Abstract:** Large Vision-Language Models (LVLMs) have demonstrated strong reasoning capabilities in geo-localization, yet they often struggle in real-world scenarios where visual cues are sparse, long-tailed, and highly ambiguous. Previous approaches, bound by internal knowledge, often fail to provide verifiable results, yielding confident but ungrounded predictions when faced with confounded evidence. To address these challenges, we propose SpotAgent, a framework that formalizes geo-localization into an agentic reasoning process that leverages expert-level reasoning to synergize visual interpretation with tool-assisted verification. SpotAgent actively explores and verifies visual cues by leveraging external tools (e.g., web search, maps) through a ReAct diagram. We introduce a 3-stage post-training pipeline starting with a Supervised Fine-Tuning (SFT) stage for basic alignment, followed by an Agentic Cold Start phase utilizing high-quality trajectories synthesized via a Multi-Agent framework, aiming to instill tool-calling expertise. Subsequently, the model's reasoning capabilities are refined through Reinforcement Learning. We propose a Spatially-Aware Dynamic Filtering strategy to enhance the efficiency of the RL stage by prioritizing learnable samples based on spatial difficulty. Extensive experiments on standard benchmarks demonstrate that SpotAgent achieves state-of-the-art performance, effectively mitigating hallucinations while delivering precise and verifiable geo-localization.

**arXiv ID:** 2602.09463
</details>

<details>
<summary><strong>Convergence and Connectivity: Dynamics of Multi-Agent Q-Learning in Random Networks</strong> - Dan Leonte, Aamal Hussain, Raphael Huser, Francesco Belardinelli, Dario Paccagnan - [[pdf]](https://arxiv.org/pdf/2503.10186)</summary>

**Abstract:** Beyond specific settings, many multi-agent learning algorithms fail to converge to an equilibrium solution, instead displaying complex, non-stationary behaviours such as recurrent or chaotic orbits. In fact, recent literature suggests that such complex behaviours are likely to occur when the number of agents increases. In this paper, we study Q-learning dynamics in network polymatrix normal-form games where the network structure is drawn from classical random graph models. In particular, we focus on the Erdős-Rényi model, which is used to analyze connectivity in distributed systems, and the Stochastic Block model, which generalizes the above by accounting for community structures that naturally arise in multi-agent systems. In each setting, we establish sufficient conditions under which the agents' joint strategies converge to a unique equilibrium. We investigate how this condition depends on the exploration rates, payoff matrices and, crucially, the probabilities of interaction between network agents. We validate our theoretical findings through numerical simulations and demonstrate that convergence can be reliably achieved in many-agent systems, provided interactions in the network are controlled.

**arXiv ID:** 2503.10186
</details>

<details>
<summary><strong>Beyond Task Performance: A Metric-Based Analysis of Sequential Cooperation in Heterogeneous Multi-Agent Destructive Foraging</strong> - Alejandro Mendoza Barrionuevo, Samuel Yanes Luis, Daniel Gutiérrez Reina, Sergio L. Toral Marín - [[pdf]](https://arxiv.org/pdf/2602.10685)</summary>

**Abstract:** This work addresses the problem of analyzing cooperation in heterogeneous multi-agent systems which operate under partial observability and temporal role dependency, framed within a destructive multi-agent foraging setting. Unlike most previous studies, which focus primarily on algorithmic performance with respect to task completion, this article proposes a systematic set of general-purpose cooperation metrics aimed at characterizing not only efficiency, but also coordination and dependency between teams and agents, fairness, and sensitivity. These metrics are designed to be transferable to different multi-agent sequential domains similar to foraging. The proposed suite of metrics is structured into three main categories that jointly provide a multilevel characterization of cooperation: primary metrics, inter-team metrics, and intra-team metrics. They have been validated in a realistic destructive foraging scenario inspired by dynamic aquatic surface cleaning using heterogeneous autonomous vehicles. It involves two specialized teams with sequential dependencies: one focused on the search of resources, and another on their destruction. Several representative approaches have been evaluated, covering both learning-based algorithms and classical heuristic paradigms.

**arXiv ID:** 2602.10685
</details>

<details>
<summary><strong>Long-Term Mapping of the Douro River Plume with Multi-Agent Reinforcement Learning</strong> - Nicolò Dal Fabbro, Milad Mesbahi, Renato Mendes, João Borges de Sousa, George J. Pappas - [[pdf]](https://arxiv.org/pdf/2510.03534)</summary>

**Abstract:** We study the problem of long-term (multiple days) mapping of a river plume using multiple autonomous underwater vehicles (AUVs), focusing on the Douro river representative use-case. We propose an energy - and communication - efficient multi-agent reinforcement learning approach in which a central coordinator intermittently communicates with the AUVs, collecting measurements and issuing commands. Our approach integrates spatiotemporal Gaussian process regression (GPR) with a multi-head Q-network controller that regulates direction and speed for each AUV. Simulations using the Delft3D ocean model demonstrate that our method consistently outperforms both single- and multi-agent benchmarks, with scaling the number of agents both improving mean squared error (MSE) and operational endurance. In some instances, our algorithm demonstrates that doubling the number of AUVs can more than double endurance while maintaining or improving accuracy, underscoring the benefits of multi-agent coordination. Our learned policies generalize across unseen seasonal regimes over different months and years, demonstrating promise for future developments of data-driven long-term monitoring of dynamic plume environments.

**arXiv ID:** 2510.03534
</details>

<details>
<summary><strong>AOI: Context-Aware Multi-Agent Operations via Dynamic Scheduling and Hierarchical Memory Compression</strong> - Zishan Bai, Jing Luo, Ziyi Ni, Enze Ge, Jiacheng Shi, Yichao Zhang, Jiayi Gu, Zhimo Han, Riyang Bao, Junfeng Hao - [[pdf]](https://arxiv.org/pdf/2512.13956)</summary>

**Abstract:** The proliferation of cloud-native architectures, characterized by microservices and dynamic orchestration, has rendered modern IT infrastructures exceedingly complex and volatile. This complexity generates overwhelming volumes of operational data, leading to critical bottlenecks in conventional systems: inefficient information processing, poor task coordination, and loss of contextual continuity during fault diagnosis and remediation. To address these challenges, we propose AOI (AI-Oriented Operations), a novel multi-agent collaborative framework that integrates three specialized agents with an LLM-based Context Compressor. Its core innovations include: (1) a dynamic task scheduling strategy that adaptively prioritizes operations based on real-time system states, (2) a three-layer memory architecture comprising Working, Episodic, and Semantic layers that optimizes context retention and retrieval. Extensive experiments on synthetic and real-world benchmarks show that AOI achieves 72.4\% context compression while preserving 92.8\% critical information, improves task success to 94.2\%, and reduces MTTR by 34.4\% over the best baseline. This work presents a paradigm shift towards scalable, adaptive, and context-aware autonomous operations, enabling robust management of next-generation IT infrastructures with minimal human intervention.

**arXiv ID:** 2512.13956
</details>

<details>
<summary><strong>Learning to Coordinate via Quantum Entanglement in Multi-Agent Reinforcement Learning</strong> - John Gardiner, Orlando Romero, Brendan Tivnan, Nicolò Dal Fabbro, George J. Pappas - [[pdf]](https://arxiv.org/pdf/2602.08965)</summary>

**Abstract:** The inability to communicate poses a major challenge to coordination in multi-agent reinforcement learning (MARL). Prior work has explored correlating local policies via shared randomness, sometimes in the form of a correlation device, as a mechanism to assist in decentralized decision-making. In contrast, this work introduces the first framework for training MARL agents to exploit shared quantum entanglement as a coordination resource, which permits a larger class of communication-free correlated policies than shared randomness alone. This is motivated by well-known results in quantum physics which posit that, for certain single-round cooperative games with no communication, shared quantum entanglement enables strategies that outperform those that only use shared randomness. In such cases, we say that there is quantum advantage. Our framework is based on a novel differentiable policy parameterization that enables optimization over quantum measurements, together with a novel policy architecture that decomposes joint policies into a quantum coordinator and decentralized local actors. To illustrate the effectiveness of our proposed method, we first show that we can learn, purely from experience, strategies that attain quantum advantage in single-round games that are treated as black box oracles. We then demonstrate how our machinery can learn policies with quantum advantage in an illustrative multi-agent sequential decision-making problem formulated as a decentralized partially observable Markov decision process (Dec-POMDP).

**arXiv ID:** 2602.08965
</details>

<details>
<summary><strong>LingxiDiagBench: A Multi-Agent Framework for Benchmarking LLMs in Chinese Psychiatric Consultation and Diagnosis</strong> - Shihao Xu, Tiancheng Zhou, Jiatong Ma, Yanli Ding, Yiming Yan, Ming Xiao, Guoyi Li, Haiyang Geng, Yunyun Han, Jianhua Chen, Yafeng Deng - [[pdf]](https://arxiv.org/pdf/2602.09379)</summary>

**Abstract:** Mental disorders are highly prevalent worldwide, but the shortage of psychiatrists and the inherent subjectivity of interview-based diagnosis create substantial barriers to timely and consistent mental-health assessment. Progress in AI-assisted psychiatric diagnosis is constrained by the absence of benchmarks that simultaneously provide realistic patient simulation, clinician-verified diagnostic labels, and support for dynamic multi-turn consultation. We present LingxiDiagBench, a large-scale multi-agent benchmark that evaluates LLMs on both static diagnostic inference and dynamic multi-turn psychiatric consultation in Chinese. At its core is LingxiDiag-16K, a dataset of 16,000 EMR-aligned synthetic consultation dialogues designed to reproduce real clinical demographic and diagnostic distributions across 12 ICD-10 psychiatric categories. Through extensive experiments across state-of-the-art LLMs, we establish key findings: (1) although LLMs achieve high accuracy on binary depression--anxiety classification (up to 92.3%), performance deteriorates substantially for depression--anxiety comorbidity recognition (43.0%) and 12-way differential diagnosis (28.5%); (2) dynamic consultation often underperforms static evaluation, indicating that ineffective information-gathering strategies significantly impair downstream diagnostic reasoning; (3) consultation quality assessed by LLM-as-a-Judge shows only moderate correlation with diagnostic accuracy, suggesting that well-structured questioning alone does not ensure correct diagnostic decisions. We release LingxiDiag-16K and the full evaluation framework to support reproducible research at this https URL.

**arXiv ID:** 2602.09379
</details>

<details>
<summary><strong>From Belief Entrenchment to Robust Reasoning in LLM Agents</strong> - Jihwan Oh, Minchan Jeong, Jongwoo Ko, Se-Young Yun - [[pdf]](https://arxiv.org/pdf/2503.16814)</summary>

**Abstract:** Multi-Agent Debate (MAD) has emerged as a promising inference scaling method for Large Language Model (LLM) reasoning. However, it frequently suffers from belief entrenchment, where agents reinforce shared errors rather than correcting them. Going beyond merely identifying this failure, we decompose it into two distinct root causes: (1) the model's biased $\textit{static initial belief}$ and (2) $\textit{homogenized debate dynamics}$ that amplify the majority view regardless of correctness. To address these sequentially, we propose $\textbf{DReaMAD}$ $($$\textbf{D}$iverse $\textbf{Rea}$soning via $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{D}$ebate with Refined Prompt$)$. Our framework first rectifies the static belief via strategic prior knowledge elicitation, then reshapes the debate dynamics by enforcing perspective diversity. Validated on our new $\textit{MetaNIM Arena}$ benchmark, $\textbf{DReaMAD}$ significantly mitigates entrenchment, achieving a +9.5\% accuracy gain over ReAct prompting and a +19.0\% higher win rate than standard MAD.

**arXiv ID:** 2503.16814
</details>

<details>
<summary><strong>CMAD: Cooperative Multi-Agent Diffusion via Stochastic Optimal Control</strong> - Riccardo Barbano, Alexander Denker, Zeljko Kereta, Runchang Li, Francisco Vargas - [[pdf]](https://arxiv.org/pdf/2602.10933)</summary>

**Abstract:** Continuous-time generative models have achieved remarkable success in image restoration and synthesis. However, controlling the composition of multiple pre-trained models remains an open challenge. Current approaches largely treat composition as an algebraic composition of probability densities, such as via products or mixtures of experts. This perspective assumes the target distribution is known explicitly, which is almost never the case. In this work, we propose a different paradigm that formulates compositional generation as a cooperative Stochastic Optimal Control problem. Rather than combining probability densities, we treat pre-trained diffusion models as interacting agents whose diffusion trajectories are jointly steered, via optimal control, toward a shared objective defined on their aggregated output. We validate our framework on conditional MNIST generation and compare it against a naive inference-time DPS-style baseline replacing learned cooperative control with per-step gradient guidance.

**arXiv ID:** 2602.10933
</details>

</details>

<details open>
<summary><h2>Other Agent Research (13 papers)</h2></summary>

<details>
<summary><strong>CLI-Gym: Scalable CLI Task Generation via Agentic Environment Inversion</strong> - Yusong Lin, Haiyang Wang, Shuzhe Wu, Lue Fan, Feiyang Pan, Sanyuan Zhao, Dandan Tu - [[pdf]](https://arxiv.org/pdf/2602.10999)</summary>

**Abstract:** Agentic coding requires agents to effectively interact with runtime environments, e.g., command line interfaces (CLI), so as to complete tasks like resolving dependency issues, fixing system problems, etc. But it remains underexplored how such environment-intensive tasks can be obtained at scale to enhance agents' capabilities. To address this, based on an analogy between the Dockerfile and the agentic task, we propose to employ agents to simulate and explore environment histories, guided by execution feedback. By tracing histories of a healthy environment, its state can be inverted to an earlier one with runtime failures, from which a task can be derived by packing the buggy state and the corresponding error messages. With our method, named CLI-Gym, a total of 1,655 environment-intensive tasks are derived, being the largest collection of its kind. Moreover, with curated successful trajectories, our fine-tuned model, named LiberCoder, achieves substantial absolute improvements of +21.1% (to 46.1%) on Terminal-Bench, outperforming various strong baselines. To our knowledge, this is the first public pipeline for scalable derivation of environment-intensive tasks.

**arXiv ID:** 2602.10999
</details>

<details>
<summary><strong>Authenticated Workflows: A Systems Approach to Protecting Agentic AI</strong> - Mohan Rajagopalan, Vinay Rao - [[pdf]](https://arxiv.org/pdf/2602.10465)</summary>

**Abstract:** Agentic AI systems automate enterprise workflows but existing defenses--guardrails, semantic filters--are probabilistic and routinely bypassed. We introduce authenticated workflows, the first complete trust layer for enterprise agentic AI. Security reduces to protecting four fundamental boundaries: prompts, tools, data, and context. We enforce intent (operations satisfy organizational policies) and integrity (operations are cryptographically authentic) at every boundary crossing, combining cryptographic elimination of attack classes with runtime policy enforcement. This delivers deterministic security--operations either carry valid cryptographic proof or are rejected. We introduce MAPL, an AI-native policy language that expresses agentic constraints dynamically as agents evolve and invocation context changes, scaling as O(log M + N) policies versus O(M x N) rules through hierarchical composition with cryptographic attestations for workflow dependencies. We prove practicality through a universal security runtime integrating nine leading frameworks (MCP, A2A, OpenAI, Claude, LangChain, CrewAI, AutoGen, LlamaIndex, Haystack) through thin adapters requiring zero protocol modifications. Formal proofs establish completeness and soundness. Empirical validation shows 100% recall with zero false positives across 174 test cases, protection against 9 of 10 OWASP Top 10 risks, and complete mitigation of two high impact production CVEs.

**arXiv ID:** 2602.10465
</details>

<details>
<summary><strong>Games with Payments between Learning Agents</strong> - Yoav Kolumbus, Joe Halpern, Éva Tardos - [[pdf]](https://arxiv.org/pdf/2405.20880)</summary>

**Abstract:** In repeated games, such as auctions, players rely on autonomous learning agents to choose their actions. We study settings in which players have their agents make monetary transfers to other agents during play at their own expense, in order to influence learning dynamics in their favor. Our goal is to understand when players have incentives to use such payments, how payments between agents affect learning outcomes, and what the resulting implications are for welfare and its distribution. We propose a simple game-theoretic model to capture the incentive structure of such scenarios. We find that, quite generally, abstaining from payments is not robust to strategic deviations by users of learning agents: self-interested players benefit from having their agents make payments to other learners. In a broad class of games, such endogenous payments between learning agents lead to higher welfare for all players. In first- and second-price auctions, equilibria of the induced "payment-policy game" lead to highly collusive learning outcomes, with low or vanishing revenue for the auctioneer. These results highlight a fundamental challenge for mechanism design, as well as for regulatory policies, in environments where learning agents may interact in the digital ecosystem beyond a mechanism's boundaries.

**arXiv ID:** 2405.20880
</details>

<details>
<summary><strong>The emergence of numerical representations in communicating artificial agents</strong> - Daniela Mihai, Lucas Weber, Francesca Franzon - [[pdf]](https://arxiv.org/pdf/2602.10996)</summary>

**Abstract:** Human languages provide efficient systems for expressing numerosities, but whether the sheer pressure to communicate is enough for numerical representations to arise in artificial agents, and whether the emergent codes resemble human numerals at all, remains an open question. We study two neural network-based agents that must communicate numerosities in a referential game using either discrete tokens or continuous sketches, thus exploring both symbolic and iconic representations. Without any pre-defined numeric concepts, the agents achieve high in-distribution communication accuracy in both communication channels and converge on high-precision symbol-meaning mappings. However, the emergent code is non-compositional: the agents fail to derive systematic messages for unseen numerosities, typically reusing the symbol of the highest trained numerosity (discrete), or collapsing extrapolated values onto a single sketch (continuous). We conclude that the communication pressure alone suffices for precise transmission of learned numerosities, but additional pressures are needed to yield compositional codes and generalisation abilities.

**arXiv ID:** 2602.10996
</details>

<details>
<summary><strong>Generalized Langevin Models of Linear Agent-Based Systems: Strategic Influence Through Environmental Coupling</strong> - Semra Gunduc, David J. Butts, Michael S. Murillo - [[pdf]](https://arxiv.org/pdf/2602.11037)</summary>

**Abstract:** Agent-based models typically treat systems in isolation, discarding environmental coupling as either computationally prohibitive or dynamically irrelevant. We demonstrate that this neglect misses essential physics: environmental degrees of freedom create memory effects that fundamentally alter system dynamics. By systematically transforming linear update rules into exact generalized Langevin equations, we show that unobserved environmental agents manifest as memory kernels whose timescales and coupling strengths are determined by the environmental interaction spectrum. Network topology shapes this memory structure in distinct ways: small-world rewiring drives dynamics toward a single dominant relaxation mode, while fragmented environments sustain multiple persistent modes corresponding to isolated subpopulations. We apply this framework to covert influence operations where adversaries manipulate target populations exclusively via environmental intermediaries. The steady-state response admits a random-walk interpretation through hitting probabilities, revealing how zealot opinions diffuse through the environment to shift system agent opinions toward the zealot mean - even when zealots never directly contact targets.

**arXiv ID:** 2602.11037
</details>

<details>
<summary><strong>Implementing Grassroots Logic Programs with Multiagent Transition Systems and AI</strong> - Ehud Shapiro - [[pdf]](https://arxiv.org/pdf/2602.06934)</summary>

**Abstract:** Grassroots Logic Programs (GLP) is a concurrent logic programming language with variables partitioned into paired \emph{readers} and \emph{writers}, conjuring both linear logic and futures/promises: an assignment is produced at most once via the sole occurrence of a writer (promise) and consumed at most once via the sole occurrence of its paired reader (future), and may contain additional readers and/or writers, enabling the concise expression of rich multidirectional communication modalities.
GLP was designed as a language for grassroots platforms -- distributed systems with multiple instances that can operate independently of each other and of any global resource, and can coalesce into ever larger instances -- with its target architecture being smartphones communicating peer-to-peer. The operational semantics of Concurrent (single-agent) GLP and of multiagent GLP (maGLP) were defined via transition systems/multiagent transition systems, respectively.
Here, we describe the mathematics developed to facilitate the workstation- and smartphone-based implementations of GLP by AI in Dart. We developed dGLP -- implementation-ready deterministic operational semantics for single-agent GLP -- and proved it correct with respect to the Concurrent GLP operational semantics; dGLP was used by AI as a formal spec, from which it developed a workstation-based implementation of GLP. We developed madGLP -- an implementation-ready multiagent operational semantics for maGLP -- and proved it correct with respect to the maGLP operational semantics; madGLP is deterministic at the agent level (not at the system level due to communication asynchrony), and is being used by AI as a formal spec from which it develops a smartphone-based implementation of maGLP.

**arXiv ID:** 2602.06934
</details>

<details>
<summary><strong>Domain Knowledge Guided Bayesian Optimization For Autonomous Alignment Of Complex Scientific Instruments</strong> - Aashwin Mishra, Matt Seaberg, Ryan Roussel, Daniel Ratner, Apurva Mehta - [[pdf]](https://arxiv.org/pdf/2602.10670)</summary>

**Abstract:** Bayesian Optimization (BO) is a powerful tool for optimizing complex non-linear systems. However, its performance degrades in high-dimensional problems with tightly coupled parameters and highly asymmetric objective landscapes, where rewards are sparse. In such needle-in-a-haystack scenarios, even advanced methods like trust-region BO (TurBO) often lead to unsatisfactory results. We propose a domain knowledge guided Bayesian Optimization approach, which leverages physical insight to fundamentally simplify the search problem by transforming coordinates to decouple input features and align the active subspaces with the primary search axes. We demonstrate this approach's efficacy on a challenging 12-dimensional, 6-crystal Split-and-Delay optical system, where conventional approaches, including standard BO, TuRBO and multi-objective BO, consistently led to unsatisfactory results. When combined with an reverse annealing exploration strategy, this approach reliably converges to the global optimum. The coordinate transformation itself is the key to this success, significantly accelerating the search by aligning input co-ordinate axes with the problem's active subspaces. As increasingly complex scientific instruments, from large telescopes to new spectrometers at X-ray Free Electron Lasers are deployed, the demand for robust high-dimensional optimization grows. Our results demonstrate a generalizable paradigm: leveraging physical insight to transform high-dimensional, coupled optimization problems into simpler representations can enable rapid and robust automated tuning for consistent high performance while still retaining current optimization algorithms.

**arXiv ID:** 2602.10670
</details>

<details>
<summary><strong>Biomimetic Mantaray robot toward the underwater autonomous -- Experimental verification of swimming and diving by flapping motion -</strong> - Kenta Tabata, Ryosuke Oku, Jun Ito, Renato Miyagusuku, Koichi Ozaki - [[pdf]](https://arxiv.org/pdf/2602.10904)</summary>

**Abstract:** This study presents the development and experimental verification of a biomimetic manta ray robot for underwater autonomous exploration. Inspired by manta rays, the robot uses flapping motion for propulsion to minimize seabed disturbance and enhance efficiency compared to traditional screw propulsion. The robot features pectoral fins driven by servo motors and a streamlined control box to reduce fluid resistance. The control system, powered by a Raspberry Pi 3B, includes an IMU and pressure sensor for real-time monitoring and control. Experiments in a pool assessed the robot's swimming and diving capabilities. Results show stable swimming and diving motions with PD control. The robot is suitable for applications in environments like aquariums and fish nurseries, requiring minimal disturbance and efficient maneuverability. Our findings demonstrate the potential of bio-inspired robotic designs to improve ecological monitoring and underwater exploration.

**arXiv ID:** 2602.10904
</details>

<details>
<summary><strong>Min-Sum Uniform Coverage Problem by Autonomous Mobile Robots</strong> - Animesh Maiti, Abhinav Chakraborty, Bibhuti Das, Subhash Bhagat, Krishnendu Mukhopadhyaya - [[pdf]](https://arxiv.org/pdf/2602.11125)</summary>

**Abstract:** We study the \textit{min-sum uniform coverage} problem for a swarm of $n$ mobile robots on a given finite line segment and on a circle having finite positive radius, where the circle is given as an input. The robots must coordinate their movements to reach a uniformly spaced configuration that minimizes the total distance traveled by all robots. The robots are autonomous, anonymous, identical, and homogeneous, and operate under the \textit{Look-Compute-Move} (LCM) model with \textit{non-rigid} motion controlled by a fair asynchronous scheduler. They are oblivious and silent, possessing neither persistent memory nor a means of explicit communication. In the \textbf{line-segment setting}, the \textit{min-sum uniform coverage} problem requires placing the robots at uniformly spaced points along the segment so as to minimize the total distance traveled by all robots. In the \textbf{circle setting} for this problem, the robots have to arrange themselves uniformly around the given circle to form a regular $n$-gon. There is no fixed orientation or designated starting vertex, and the goal is to minimize the total distance traveled by all the robots. We present a deterministic distributed algorithm that achieves uniform coverage in the line-segment setting with minimum total movement cost. For the circle setting, we characterize all initial configurations for which the \textit{min-sum uniform coverage} problem is deterministically unsolvable under the considered robot model. For all the other remaining configurations, we provide a deterministic distributed algorithm that achieves uniform coverage while minimizing the total distance traveled. These results characterize the deterministic solvability of min-sum coverage for oblivious robots and achieve optimal cost whenever solvable.

**arXiv ID:** 2602.11125
</details>

<details>
<summary><strong>Lateral tracking control of all-wheel steering vehicles with intelligent tires</strong> - Luigi Romano, Ole Morten Aamo, Jan Åslund, Erik Frisk - [[pdf]](https://arxiv.org/pdf/2602.09427)</summary>

**Abstract:** The accurate characterization of tire dynamics is critical for advancing control strategies in autonomous road vehicles, as tire behavior significantly influences handling and stability through the generation of forces and moments at the tire-road interface. Smart tire technologies have emerged as a promising tool for sensing key variables such as road friction, tire pressure, and wear states, and for estimating kinematic and dynamic states like vehicle speed and tire forces. However, most existing estimation and control algorithms rely on empirical correlations or machine learning approaches, which require extensive calibration and can be sensitive to variations in operating conditions. In contrast, model-based techniques, which leverage infinite-dimensional representations of tire dynamics using partial differential equations (PDEs), offer a more robust approach. This paper proposes a novel model-based, output-feedback lateral tracking control strategy for all-wheel steering vehicles that integrates distributed tire dynamics with smart tire technologies. The primary contributions include the suppression of micro-shimmy phenomena at low speeds and path-following via force control, achieved through the estimation of tire slip angles, vehicle kinematics, and lateral tire forces. The proposed controller and observer are based on formulations using ODE-PDE systems, representing rigid body dynamics and distributed tire behavior. This work marks the first rigorous control strategy for vehicular systems equipped with distributed tire representations in conjunction with smart tire technologies.

**arXiv ID:** 2602.09427
</details>

<details>
<summary><strong>Normalized Surveillance in the Datafied Car: How Autonomous Vehicle Users Rationalize Privacy Trade-offs</strong> - Yehuda Perry, Tawfiq Ammari - [[pdf]](https://arxiv.org/pdf/2602.11026)</summary>

**Abstract:** Autonomous vehicles (AVs) are characterized by pervasive datafication and surveillance through sensors like in-cabin cameras, LIDAR, and GPS. Drawing on 16 semi-structured interviews with AV drivers analyzed using constructivist grounded theory, this study examines how users make sense of vehicular surveillance within everyday datafication. Findings reveal drivers demonstrate few AV-specific privacy concerns, instead normalizing monitoring through comparisons with established digital platforms. We theorize this indifference by situating AV surveillance within the `surveillance ecology' of platform environments, arguing the datafied car functions as a mobile extension of the `leaky home' -- private spaces rendered permeable through connected technologies continuously transmitting behavioral data.
The study contributes to scholarship on surveillance beliefs, datafication, and platform governance by demonstrating how users who have accepted comprehensive smartphone and smart home monitoring encounter AV datafication as just another node in normalized data extraction. We highlight how geographic restrictions on data access -- currently limiting driver log access to California -- create asymmetries that impede informed privacy deliberation, exemplifying `tertiary digital divides.' Finally, we examine how machine learning's reliance on data-intensive approaches creates structural pressure for surveillance that transcends individual manufacturer choices. We propose governance interventions to democratize social learning, including universal data access rights, binding transparency requirements, and data minimization standards to prevent race-to-the-bottom dynamics in automotive datafication.

**arXiv ID:** 2602.11026
</details>

<details>
<summary><strong>From Junior to Senior: Allocating Agency and Navigating Professional Growth in Agentic AI-Mediated Software Engineering</strong> - Dana Feng, Bhada Yun, April Yi Wang - [[pdf]](https://arxiv.org/pdf/2602.00496)</summary>

**Abstract:** Juniors enter as AI-natives, seniors adapted mid-career. AI is not just changing how engineers code-it is reshaping who holds agency across work and professional growth. We contribute junior-senior accounts on their usage of agentic AI through a three-phase mixed-methods study: ACTA combined with a Delphi process with 5 seniors, an AI-assisted debugging task with 10 juniors, and blind reviews of junior prompt histories by 5 more seniors. We found that agency in software engineering is primarily constrained by organizational policies rather than individual preferences, with experienced developers maintaining control through detailed delegation while novices struggle between over-reliance and cautious avoidance. Seniors leverage pre-AI foundational instincts to steer modern tools and possess valuable perspectives for mentoring juniors in their early AI-encouraged career development. From synthesis of results, we suggest three practices that focus on preserving agency in software engineering for coding, learning, and mentorship, especially as AI grows increasingly autonomous.

**arXiv ID:** 2602.00496
</details>

<details>
<summary><strong>From Fragmentation to Integration: Exploring the Design Space of AI Agents for Human-as-the-Unit Privacy Management</strong> - Eryue Xu, Tianshi Li - [[pdf]](https://arxiv.org/pdf/2602.05016)</summary>

**Abstract:** Managing one's digital footprint is overwhelming, as it spans multiple platforms and involves countless context-dependent decisions. Recent advances in agentic AI offer ways forward by enabling holistic, contextual privacy-enhancing solutions. Building on this potential, we adopted a ''human-as-the-unit'' perspective and investigated users' cross-context privacy challenges through 12 semi-structured interviews. Results reveal that people rely on ad hoc manual strategies while lacking comprehensive privacy controls, highlighting nine privacy-management challenges across applications, temporal contexts, and relationships. To explore solutions, we generated nine AI agent concepts and evaluated them via a speed-dating survey with 116 US participants. The three highest-ranked concepts were all post-sharing management tools with half or full agent autonomy, with users expressing greater trust in AI accuracy than in their own efforts. Our findings highlight a promising design space where users see AI agents bridging the fragments in privacy management, particularly through automated, comprehensive post-sharing remediation of users' digital footprints.

**arXiv ID:** 2602.05016
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (39 papers)</h2></summary>

<details>
<summary><strong>Found-RL: foundation model-enhanced reinforcement learning for autonomous driving</strong> - Yansong Qu, Zihao Sheng, Zilin Huang, Jiancong Chen, Yuhao Luo, Tianyi Wang, Yiheng Feng, Samuel Labi, Sikai Chen - [[pdf]](https://arxiv.org/pdf/2602.10458)</summary>

**Abstract:** Reinforcement Learning (RL) has emerged as a dominant paradigm for end-to-end autonomous driving (AD). However, RL suffers from sample inefficiency and a lack of semantic interpretability in complex scenarios. Foundation Models, particularly Vision-Language Models (VLMs), can mitigate this by offering rich, context-aware knowledge, yet their high inference latency hinders deployment in high-frequency RL training loops. To bridge this gap, we present Found-RL, a platform tailored to efficiently enhance RL for AD using foundation models. A core innovation is the asynchronous batch inference framework, which decouples heavy VLM reasoning from the simulation loop, effectively resolving latency bottlenecks to support real-time learning. We introduce diverse supervision mechanisms: Value-Margin Regularization (VMR) and Advantage-Weighted Action Guidance (AWAG) to effectively distill expert-like VLM action suggestions into the RL policy. Additionally, we adopt high-throughput CLIP for dense reward shaping. We address CLIP's dynamic blindness via Conditional Contrastive Action Alignment, which conditions prompts on discretized speed/command and yields a normalized, margin-based bonus from context-specific action-anchor scoring. Found-RL provides an end-to-end pipeline for fine-tuned VLM integration and shows that a lightweight RL model can achieve near-VLM performance compared with billion-parameter VLMs while sustaining real-time inference (approx. 500 FPS). Code, data, and models will be publicly available at this https URL.

**arXiv ID:** 2602.10458
</details>

<details>
<summary><strong>Neuro-symbolic Action Masking for Deep Reinforcement Learning</strong> - Shuai Han, Mehdi Dastani, Shihan Wang - [[pdf]](https://arxiv.org/pdf/2602.10598)</summary>

**Abstract:** Deep reinforcement learning (DRL) may explore infeasible actions during training and execution. Existing approaches assume a symbol grounding function that maps high-dimensional states to consistent symbolic representations and a manually specified action masking techniques to constrain actions. In this paper, we propose Neuro-symbolic Action Masking (NSAM), a novel framework that automatically learn symbolic models, which are consistent with given domain constraints of high-dimensional states, in a minimally supervised manner during the DRL process. Based on the learned symbolic model of states, NSAM learns action masks that rules out infeasible actions. NSAM enables end-to-end integration of symbolic reasoning and deep policy optimization, where improvements in symbolic grounding and policy learning mutually reinforce each other. We evaluate NSAM on multiple domains with constraints, and experimental results demonstrate that NSAM significantly improves sample efficiency of DRL agent while substantially reducing constraint violations.

**arXiv ID:** 2602.10598
</details>

<details>
<summary><strong>GameDevBench: Evaluating Agentic Capabilities Through Game Development</strong> - Wayne Chi, Yixiong Fang, Arnav Yayavaram, Siddharth Yayavaram, Seth Karten, Qiuhong Anna Wei, Runkun Chen, Alexander Wang, Valerie Chen, Ameet Talwalkar, Chris Donahue - [[pdf]](https://arxiv.org/pdf/2602.11103)</summary>

**Abstract:** Despite rapid progress on coding agents, progress on their multimodal counterparts has lagged behind. A key challenge is the scarcity of evaluation testbeds that combine the complexity of software development with the need for deep multimodal understanding. Game development provides such a testbed as agents must navigate large, dense codebases while manipulating intrinsically multimodal assets such as shaders, sprites, and animations within a visual game scene. We present GameDevBench, the first benchmark for evaluating agents on game development tasks. GameDevBench consists of 132 tasks derived from web and video tutorials. Tasks require significant multimodal understanding and are complex -- the average solution requires over three times the amount of lines of code and file changes compared to prior software development benchmarks. Agents still struggle with game development, with the best agent solving only 54.5% of tasks. We find a strong correlation between perceived task difficulty and multimodal complexity, with success rates dropping from 46.9% on gameplay-oriented tasks to 31.6% on 2D graphics tasks. To improve multimodal capability, we introduce two simple image and video-based feedback mechanisms for agents. Despite their simplicity, these methods consistently improve performance, with the largest change being an increase in Claude Sonnet 4.5's performance from 33.3% to 47.7%. We release GameDevBench publicly to support further research into agentic game development.

**arXiv ID:** 2602.11103
</details>

<details>
<summary><strong>Anonymization-Enhanced Privacy Protection for Mobile GUI Agents: Available but Invisible</strong> - Lepeng Zhao, Zhenhua Zou, Shuo Li, Zhuotao Liu - [[pdf]](https://arxiv.org/pdf/2602.10139)</summary>

**Abstract:** Mobile Graphical User Interface (GUI) agents have demonstrated strong capabilities in automating complex smartphone tasks by leveraging multimodal large language models (MLLMs) and system-level control interfaces. However, this paradigm introduces significant privacy risks, as agents typically capture and process entire screen contents, thereby exposing sensitive personal data such as phone numbers, addresses, messages, and financial information. Existing defenses either reduce UI exposure, obfuscate only task-irrelevant content, or rely on user authorization, but none can protect task-critical sensitive information while preserving seamless agent usability.
We propose an anonymization-based privacy protection framework that enforces the principle of available-but-invisible access to sensitive data: sensitive information remains usable for task execution but is never directly visible to the cloud-based agent. Our system detects sensitive UI content using a PII-aware recognition model and replaces it with deterministic, type-preserving placeholders (e.g., PHONE_NUMBER#a1b2c) that retain semantic categories while removing identifying details. A layered architecture comprising a PII Detector, UI Transformer, Secure Interaction Proxy, and Privacy Gatekeeper ensures consistent anonymization across user instructions, XML hierarchies, and screenshots, mediates all agent actions over anonymized interfaces, and supports narrowly scoped local computations when reasoning over raw values is necessary.
Extensive experiments on the AndroidLab and PrivScreen benchmarks show that our framework substantially reduces privacy leakage across multiple models while incurring only modest utility degradation, achieving the best observed privacy-utility trade-off among existing methods.

**arXiv ID:** 2602.10139
</details>

<details>
<summary><strong>Can Large Language Models Implement Agent-Based Models? An ODD-based Replication Study</strong> - Nuno Fachada, Daniel Fernandes, Carlos M. Fernandes, João P. Matos-Carvalho - [[pdf]](https://arxiv.org/pdf/2602.10140)</summary>

**Abstract:** Large language models (LLMs) can now synthesize non-trivial executable code from textual descriptions, raising an important question: can LLMs reliably implement agent-based models from standardized specifications in a way that supports replication, verification, and validation? We address this question by evaluating 17 contemporary LLMs on a controlled ODD-to-code translation task, using the PPHPC predator-prey model as a fully specified reference. Generated Python implementations are assessed through staged executability checks, model-independent statistical comparison against a validated NetLogo baseline, and quantitative measures of runtime efficiency and maintainability. Results show that behaviorally faithful implementations are achievable but not guaranteed, and that executability alone is insufficient for scientific use. GPT-4.1 consistently produces statistically valid and efficient implementations, with Claude 3.7 Sonnet performing well but less reliably. Overall, the findings clarify both the promise and current limitations of LLMs as model engineering tools, with implications for reproducible agent-based and environmental modelling.

**arXiv ID:** 2602.10140
</details>

<details>
<summary><strong>Beyond SMILES: Evaluating Agentic Systems for Drug Discovery</strong> - Edward Wijaya - [[pdf]](https://arxiv.org/pdf/2602.10163)</summary>

**Abstract:** Agentic systems for drug discovery have demonstrated autonomous synthesis planning, literature mining, and molecular design. We ask how well they generalize. Evaluating six frameworks against 15 task classes drawn from peptide therapeutics, in vivo pharmacology, and resource-constrained settings, we find five capability gaps: no support for protein language models or peptide-specific prediction, no bridges between in vivo and in silico data, reliance on LLM inference with no pathway to ML training or reinforcement learning, assumptions tied to large-pharma resources, and single-objective optimization that ignores safety-efficacy-stability trade-offs. A paired knowledge-probing experiment suggests the bottleneck is architectural rather than epistemic: four frontier LLMs reason about peptides at levels comparable to small molecules, yet no framework exposes this capability. We propose design requirements and a capability matrix for next-generation frameworks that function as computational partners under realistic constraints.

**arXiv ID:** 2602.10163
</details>

<details>
<summary><strong>Internalizing Meta-Experience into Memory for Guided Reinforcement Learning in Large Language Models</strong> - Shiting Huang, Zecheng Li, Yu Zeng, Qingnan Ren, Zhen Fang, Qisheng Su, Kou Shi, Lin Chen, Zehui Chen, Feng Zhao - [[pdf]](https://arxiv.org/pdf/2602.10224)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an effective approach for enhancing the reasoning capabilities of Large Language Models (LLMs). Despite its efficacy, RLVR faces a meta-learning bottleneck: it lacks mechanisms for error attribution and experience internalization intrinsic to the human learning cycle beyond practice and verification, thereby limiting fine-grained credit assignment and reusable knowledge formation. We term such reusable knowledge representations derived from past errors as meta-experience. Based on this insight, we propose Meta-Experience Learning (MEL), a novel framework that incorporates self-distilled meta-experience into the model's parametric memory. Building upon standard RLVR, we introduce an additional design that leverages the LLM's self-verification capability to conduct contrastive analysis on paired correct and incorrect trajectories, identify the precise bifurcation points where reasoning errors arise, and summarize them into generalizable meta-experience. The meta-experience is further internalized into the LLM's parametric memory by minimizing the negative log-likelihood, which induces a language-modeled reward signal that bridges correct and incorrect reasoning trajectories and facilitates effective knowledge reuse. Experimental results demonstrate that MEL achieves consistent improvements on benchmarks, yielding 3.92%--4.73% Pass@1 gains across varying model sizes.

**arXiv ID:** 2602.10224
</details>

<details>
<summary><strong>AIvilization v0: Toward Large-Scale Artificial Social Simulation with a Unified Agent Architecture and Adaptive Agent Profiles</strong> - Wenkai Fan, Shurui Zhang, Xiaolong Wang, Haowei Yang, Tsz Wai Chan, Xingyan Chen, Junquan Bi, Zirui Zhou, Jia Liu, Kani Chen - [[pdf]](https://arxiv.org/pdf/2602.10429)</summary>

**Abstract:** AIvilization v0 is a publicly deployed large-scale artificial society that couples a resource-constrained sandbox economy with a unified LLM-agent architecture, aiming to sustain long-horizon autonomy while remaining executable under rapidly changing environment. To mitigate the tension between goal stability and reactive correctness, we introduce (i) a hierarchical branch-thinking planner that decomposes life goals into parallel objective branches and uses simulation-guided validation plus tiered re-planning to ensure feasibility; (ii) an adaptive agent profile with dual-process memory that separates short-term execution traces from long-term semantic consolidation, enabling persistent yet evolving identity; and (iii) a human-in-the-loop steering interface that injects long-horizon objectives and short commands at appropriate abstraction levels, with effects propagated through memory rather than brittle prompt overrides. The environment integrates physiological survival costs, non-substitutable multi-tier production, an AMM-based price mechanism, and a gated education-occupation system. Using high-frequency transactions from the platforms mature phase, we find stable markets that reproduce key stylized facts (heavy-tailed returns and volatility clustering) and produce structured wealth stratification driven by education and access constraints. Ablations show simplified planners can match performance on narrow tasks, while the full architecture is more robust under multi-objective, long-horizon settings, supporting delayed investment and sustained exploration.

**arXiv ID:** 2602.10429
</details>

<details>
<summary><strong>Control Reinforcement Learning: Token-Level Mechanistic Analysis via Learned SAE Feature Steering</strong> - Seonglae Cho, Zekun Wu, Adriano Koshiyama - [[pdf]](https://arxiv.org/pdf/2602.10437)</summary>

**Abstract:** Sparse autoencoders (SAEs) decompose language model activations into interpretable features, but existing methods reveal only which features activate, not which change model outputs when amplified. We introduce Control Reinforcement Learning (CRL), which trains a policy to select SAE features for steering at each token, producing interpretable intervention logs: the learned policy identifies features that change model outputs when amplified. Adaptive Feature Masking encourages diverse feature discovery while preserving singlefeature interpretability. The framework yields new analysis capabilities: branch point tracking locates tokens where feature choice determines output correctness; critic trajectory analysis separates policy limitations from value estimation errors; layer-wise comparison reveals syntactic features in early layers and semantic features in later layers. On Gemma-2 2B across MMLU, BBQ, GSM8K, HarmBench, and XSTest, CRL achieves improvements while providing per-token intervention logs. These results establish learned feature steering as a mechanistic interpretability tool that complements static feature analysis with dynamic intervention probes

**arXiv ID:** 2602.10437
</details>

<details>
<summary><strong>MetaphorStar: Image Metaphor Understanding and Reasoning with End-to-End Visual Reinforcement Learning</strong> - Chenhao Zhang, Yazhe Niu, Hongsheng Li - [[pdf]](https://arxiv.org/pdf/2602.10575)</summary>

**Abstract:** Metaphorical comprehension in images remains a critical challenge for Nowadays AI systems. While Multimodal Large Language Models (MLLMs) excel at basic Visual Question Answering (VQA), they consistently struggle to grasp the nuanced cultural, emotional, and contextual implications embedded in visual content. This difficulty stems from the task's demand for sophisticated multi-hop reasoning, cultural context, and Theory of Mind (ToM) capabilities, which current models lack. To fill this gap, we propose MetaphorStar, the first end-to-end visual reinforcement learning (RL) framework for image implication tasks. Our framework includes three core components: the fine-grained dataset TFQ-Data, the visual RL method TFQ-GRPO, and the well-structured benchmark TFQ-Bench.
Our fully open-source MetaphorStar family, trained using TFQ-GRPO on TFQ-Data, significantly improves performance by an average of 82.6% on the image implication benchmarks. Compared with 20+ mainstream MLLMs, MetaphorStar-32B achieves state-of-the-art (SOTA) on Multiple-Choice Question and Open-Style Question, significantly outperforms the top closed-source model Gemini-3.0-pro on True-False Question. Crucially, our experiments reveal that learning image implication tasks improves the general understanding ability, especially the complex visual reasoning ability. We further provide a systematic analysis of model parameter scaling, training data scaling, and the impact of different model architectures and training strategies, demonstrating the broad applicability of our method. We open-sourced all model weights, datasets, and method code at this https URL.

**arXiv ID:** 2602.10575
</details>

<details>
<summary><strong>ICA: Information-Aware Credit Assignment for Visually Grounded Long-Horizon Information-Seeking Agents</strong> - Cong Pang, Xuyu Feng, Yujie Yi, Zixuan Chen, Jiawei Hong, Tiankuo Yao, Nang Yuan, Jiapeng Luo, Lewei Lu, Xin Lou - [[pdf]](https://arxiv.org/pdf/2602.10863)</summary>

**Abstract:** Despite the strong performance achieved by reinforcement learning-trained information-seeking agents, learning in open-ended web environments remains severely constrained by low signal-to-noise feedback. Text-based parsers often discard layout semantics and introduce unstructured noise, while long-horizon training typically relies on sparse outcome rewards that obscure which retrieval actions actually matter. We propose a visual-native search framework that represents webpages as visual snapshots, allowing agents to leverage layout cues to quickly localize salient evidence and suppress distractors. To learn effectively from these high-dimensional observations, we introduce Information-Aware Credit Assignment (ICA), a post-hoc method that estimates each retrieved snapshot's contribution to the final outcome via posterior analysis and propagates dense learning signals back to key search turns. Integrated with a GRPO-based training pipeline, our approach consistently outperforms text-based baselines on diverse information-seeking benchmarks, providing evidence that visual snapshot grounding with information-level credit assignment alleviates the credit-assignment bottleneck in open-ended web environments. The code and datasets will be released in this https URL.

**arXiv ID:** 2602.10863
</details>

<details>
<summary><strong>Resource-Efficient Model-Free Reinforcement Learning for Board Games</strong> - Kazuki Ota, Takayuki Osa, Motoki Omura, Tatsuya Harada - [[pdf]](https://arxiv.org/pdf/2602.10894)</summary>

**Abstract:** Board games have long served as complex decision-making benchmarks in artificial intelligence. In this field, search-based reinforcement learning methods such as AlphaZero have achieved remarkable success. However, their significant computational demands have been pointed out as barriers to their reproducibility. In this study, we propose a model-free reinforcement learning algorithm designed for board games to achieve more efficient learning. To validate the efficiency of the proposed method, we conducted comprehensive experiments on five board games: Animal Shogi, Gardner Chess, Go, Hex, and Othello. The results demonstrate that the proposed method achieves more efficient learning than existing methods across these environments. In addition, our extensive ablation study shows the importance of core techniques used in the proposed method. We believe that our efficient algorithm shows the potential of model-free reinforcement learning in domains traditionally dominated by search-based methods.

**arXiv ID:** 2602.10894
</details>

<details>
<summary><strong>FeatureBench: Benchmarking Agentic Coding for Complex Feature Development</strong> - Qixing Zhou, Jiacheng Zhang, Haiyang Wang, Rui Hao, Jiahe Wang, Minghao Han, Yuxue Yang, Shuzhe Wu, Feiyang Pan, Lue Fan, Dandan Tu, Zhaoxiang Zhang - [[pdf]](https://arxiv.org/pdf/2602.10975)</summary>

**Abstract:** Agents powered by large language models (LLMs) are increasingly adopted in the software industry, contributing code as collaborators or even autonomous developers. As their presence grows, it becomes important to assess the current boundaries of their coding abilities. Existing agentic coding benchmarks, however, cover a limited task scope, e.g., bug fixing within a single pull request (PR), and often rely on non-executable evaluations or lack an automated approach for continually updating the evaluation coverage. To address such issues, we propose FeatureBench, a benchmark designed to evaluate agentic coding performance in end-to-end, feature-oriented software development. FeatureBench incorporates an execution-based evaluation protocol and a scalable test-driven method that automatically derives tasks from code repositories with minimal human effort. By tracing from unit tests along a dependency graph, our approach can identify feature-level coding tasks spanning multiple commits and PRs scattered across the development timeline, while ensuring the proper functioning of other features after the separation. Using this framework, we curated 200 challenging evaluation tasks and 3825 executable environments from 24 open-source repositories in the first version of our benchmark. Empirical evaluation reveals that the state-of-the-art agentic model, such as Claude 4.5 Opus, which achieves a 74.4% resolved rate on SWE-bench, succeeds on only 11.0% of tasks, opening new opportunities for advancing agentic coding. Moreover, benefiting from our automated task collection toolkit, FeatureBench can be easily scaled and updated over time to mitigate data leakage. The inherent verifiability of constructed environments also makes our method potentially valuable for agent training.

**arXiv ID:** 2602.10975
</details>

<details>
<summary><strong>DataChef: Cooking Up Optimal Data Recipes for LLM Adaptation via Reinforcement Learning</strong> - Yicheng Chen, Zerun Ma, Xinchen Xie, Yining Li, Kai Chen - [[pdf]](https://arxiv.org/pdf/2602.11089)</summary>

**Abstract:** In the current landscape of Large Language Models (LLMs), the curation of large-scale, high-quality training data is a primary driver of model performance. A key lever is the \emph{data recipe}, which comprises a data processing pipeline to transform raw sources into training corpora. Despite the growing use of LLMs to automate individual data processing steps, such as data synthesis and filtering, the overall design of data recipes remains largely manual and labor-intensive, requiring substantial human expertise and iteration. To bridge this gap, we formulate \emph{end-to-end data recipe generation} for LLM adaptation. Given a target benchmark and a pool of available data sources, a model is required to output a complete data recipe that adapts a base LLM to the target task. We present DataChef-32B, which performs online reinforcement learning using a proxy reward that predicts downstream performance for candidate recipes. Across six held-out tasks, DataChef-32B produces practical recipes that reach comparable downstream performance to those curated by human experts. Notably, the recipe from DataChef-32B adapts Qwen3-1.7B-Base to the math domain, achieving 66.7 on AIME'25 and surpassing Qwen3-1.7B. This work sheds new light on automating LLM training and developing self-evolving AI systems.

**arXiv ID:** 2602.11089
</details>

<details>
<summary><strong>Data-Efficient Hierarchical Goal-Conditioned Reinforcement Learning via Normalizing Flows</strong> - Shaswat Garg, Matin Moezzi, Brandon Da Silva - [[pdf]](https://arxiv.org/pdf/2602.11142)</summary>

**Abstract:** Hierarchical goal-conditioned reinforcement learning (H-GCRL) provides a powerful framework for tackling complex, long-horizon tasks by decomposing them into structured subgoals. However, its practical adoption is hindered by poor data efficiency and limited policy expressivity, especially in offline or data-scarce regimes. In this work, Normalizing flow-based hierarchical implicit Q-learning (NF-HIQL), a novel framework that replaces unimodal gaussian policies with expressive normalizing flow policies at both the high- and low-levels of the hierarchy is introduced. This design enables tractable log-likelihood computation, efficient sampling, and the ability to model rich multimodal behaviors. New theoretical guarantees are derived, including explicit KL-divergence bounds for Real-valued non-volume preserving (RealNVP) policies and PAC-style sample efficiency results, showing that NF-HIQL preserves stability while improving generalization. Empirically, NF-HIQL is evaluted across diverse long-horizon tasks in locomotion, ball-dribbling, and multi-step manipulation from OGBench. NF-HIQL consistently outperforms prior goal-conditioned and hierarchical baselines, demonstrating superior robustness under limited data and highlighting the potential of flow-based architectures for scalable, data-efficient hierarchical reinforcement learning.

**arXiv ID:** 2602.11142
</details>

<details>
<summary><strong>Agentic Jigsaw Interaction Learning for Enhancing Visual Perception and Reasoning in Vision-Language Models</strong> - Yu Zeng, Wenxuan Huang, Shiting Huang, Xikun Bao, Yukun Qi, Yiming Zhao, Qiuchen Wang, Lin Chen, Zehui Chen, Huaian Chen, Wanli Ouyang, Feng Zhao - [[pdf]](https://arxiv.org/pdf/2510.01304)</summary>

**Abstract:** Although current large Vision-Language Models (VLMs) have advanced in multimodal understanding and reasoning, their fundamental perceptual and reasoning abilities remain limited. Specifically, even on simple jigsaw tasks, existing VLMs perform near randomly, revealing deficiencies in core perception and reasoning capabilities. While high-quality vision-language data can enhance these capabilities, its scarcity and limited scalability impose significant constraints. To address this, we propose AGILE, an Agentic jiGsaw Interaction Learning for Enhancing visual perception and reasoning in VLMs. AGILE formulates jigsaw solving as an interactive process, enabling the model to progressively engage with the environment. At each step, the model generates executable code to perform an action based on the current state, while the environment provides fine-grained visual feedback to guide task completion. Through this iterative cycle of observation and interaction, the model incrementally improves its perceptual and reasoning capabilities via exploration and feedback. Experimental results show that AGILE not only substantially boosts performance on jigsaw tasks of varying complexity (e.g., increasing accuracy from 9.5% to 82.8% under the 2 $\times$ 2 setting) but also demonstrates strong generalization across 9 general vision tasks, achieving an average improvement of 3.1%. These results indicate notable enhancements in both perceptual and reasoning abilities. This work opens a new avenue for advancing reasoning and generalization in multimodal models and provides an efficient, scalable solution to the scarcity of multimodal reinforcement learning data. The code and datasets is available at this https URL .

**arXiv ID:** 2510.01304
</details>

<details>
<summary><strong>Progress Constraints for Reinforcement Learning in Behavior Trees</strong> - Finn Rietz, Mart Kartašev, Petter Ögren, Johannes A. Stork - [[pdf]](https://arxiv.org/pdf/2602.06525)</summary>

**Abstract:** Behavior Trees (BTs) provide a structured and reactive framework for decision-making, commonly used to switch between sub-controllers based on environmental conditions. Reinforcement Learning (RL), on the other hand, can learn near-optimal controllers but sometimes struggles with sparse rewards, safe exploration, and long-horizon credit assignment. Combining BTs with RL has the potential for mutual benefit: a BT design encodes structured domain knowledge that can simplify RL training, while RL enables automatic learning of the controllers within BTs. However, naive integration of BTs and RL can lead to some controllers counteracting other controllers, possibly undoing previously achieved subgoals, thereby degrading the overall performance. To address this, we propose progress constraints, a novel mechanism where feasibility estimators constrain the allowed action set based on theoretical BT convergence results. Empirical evaluations in a 2D proof-of-concept and a high-fidelity warehouse environment demonstrate improved performance, sample efficiency, and constraint satisfaction, compared to prior methods of BT-RL integration.

**arXiv ID:** 2602.06525
</details>

<details>
<summary><strong>Learning the Value Systems of Societies with Preference-based Multi-objective Reinforcement Learning</strong> - Andrés Holgado-Sánchez, Peter Vamplew, Richard Dazeley, Sascha Ossowski, Holger Billhardt - [[pdf]](https://arxiv.org/pdf/2602.08835)</summary>

**Abstract:** Value-aware AI should recognise human values and adapt to the value systems (value-based preferences) of different users. This requires operationalization of values, which can be prone to misspecification. The social nature of values demands their representation to adhere to multiple users while value systems are diverse, yet exhibit patterns among groups. In sequential decision making, efforts have been made towards personalization for different goals or values from demonstrations of diverse agents. However, these approaches demand manually designed features or lack value-based interpretability and/or adaptability to diverse user preferences.
We propose algorithms for learning models of value alignment and value systems for a society of agents in Markov Decision Processes (MDPs), based on clustering and preference-based multi-objective reinforcement learning (PbMORL). We jointly learn socially-derived value alignment models (groundings) and a set of value systems that concisely represent different groups of users (clusters) in a society. Each cluster consists of a value system representing the value-based preferences of its members and an approximately Pareto-optimal policy that reflects behaviours aligned with this value system. We evaluate our method against a state-of-the-art PbMORL algorithm and baselines on two MDPs with human values.

**arXiv ID:** 2602.08835
</details>

<details>
<summary><strong>Enhancing Inverse Reinforcement Learning through Encoding Dynamic Information in Reward Shaping</strong> - Simon Sinong Zhan, Philip Wang, Qingyuan Wu, Yixuan Wang, Ruochen Jiao, Chao Huang, Qi Zhu - [[pdf]](https://arxiv.org/pdf/2410.03847)</summary>

**Abstract:** In this paper, we aim to tackle the limitation of the Adversarial Inverse Reinforcement Learning (AIRL) method in stochastic environments where theoretical results cannot hold and performance is degraded. To address this issue, we propose a novel method which infuses the dynamics information into the reward shaping with the theoretical guarantee for the induced optimal policy in the stochastic environments. Incorporating our novel model-enhanced rewards, we present a novel Model-Enhanced AIRL framework, which integrates transition model estimation directly into reward shaping. Furthermore, we provide a comprehensive theoretical analysis of the reward error bound and performance difference bound for our method. The experimental results in MuJoCo benchmarks show that our method can achieve superior performance in stochastic environments and competitive performance in deterministic environments, with significant improvement in sample efficiency, compared to existing baselines.

**arXiv ID:** 2410.03847
</details>

<details>
<summary><strong>Belief-Based Offline Reinforcement Learning for Delay-Robust Policy Optimization</strong> - Simon Sinong Zhan, Qingyuan Wu, Philip Wang, Frank Yang, Xiangyu Shi, Chao Huang, Qi Zhu - [[pdf]](https://arxiv.org/pdf/2506.00131)</summary>

**Abstract:** Offline-to-online deployment of reinforcement-learning (RL) agents must bridge two gaps: (1) the sim-to-real gap, where real systems add latency and other imperfections not present in simulation, and (2) the interaction gap, where policies trained purely offline face out-of-distribution states during online execution because gathering new interaction data is costly or risky. Agents therefore have to generalize from static, delay-free datasets to dynamic, delay-prone environments. Standard offline RL learns from delay-free logs yet must act under delays that break the Markov assumption and hurt performance. We introduce DT-CORL (Delay-Transformer belief policy Constrained Offline RL), an offline-RL framework built to cope with delayed dynamics at deployment. DT-CORL (i) produces delay-robust actions with a transformer-based belief predictor even though it never sees delayed observations during training, and (ii) is markedly more sample-efficient than naïve history-augmentation baselines. Experiments on D4RL benchmarks with several delay settings show that DT-CORL consistently outperforms both history-augmentation and vanilla belief-based methods, narrowing the sim-to-real latency gap while preserving data efficiency.

**arXiv ID:** 2506.00131
</details>

<details>
<summary><strong>The Choice of Divergence: A Neglected Key to Mitigating Diversity Collapse in Reinforcement Learning with Verifiable Reward</strong> - Long Li, Zhijian Zhou, Jiaran Hao, Jason Klein Liu, Yanting Miao, Wei Pang, Xiaoyu Tan, Wei Chu, Zhe Wang, Shirui Pan, Chao Qu, Yuan Qi - [[pdf]](https://arxiv.org/pdf/2509.07430)</summary>

**Abstract:** A central paradox in fine-tuning Large Language Models (LLMs) with Reinforcement Learning with Verifiable Reward (RLVR) is the frequent degradation of multi-attempt performance (Pass@k) despite improvements in single-attempt accuracy (Pass@1). This is often accompanied by catastrophic forgetting, where models lose previously acquired skills. While various methods have been proposed, the choice and function of the divergence term have been surprisingly unexamined as a proactive solution. We argue that standard RLVR objectives -- both those using the mode-seeking reverse KL-divergence and those forgoing a divergence term entirely -- lack a crucial mechanism for knowledge retention. The reverse-KL actively accelerates this decay by narrowing the policy, while its absence provides no safeguard against the model drifting from its diverse knowledge base. We propose a fundamental shift in perspective: using the divergence term itself as the solution. Our framework, Diversity-Preserving Hybrid RL (DPH-RL), leverages mass-covering f-divergences (like forward-KL and JS-divergence) to function as a rehearsal mechanism. By continuously referencing the initial policy, this approach forces the model to maintain broad solution coverage. Extensive experiments on math and SQL generation demonstrate that DPH-RL not only resolves the Pass@k degradation but improves both Pass@1 and Pass@k in- and out-of-domain. Additionally, DPH-RL is more training-efficient because it computes f-divergence using generator functions, requiring only sampling from the initial policy and no online reference model. Our work highlights a crucial, overlooked axis for improving RLVR, demonstrating that the proper selection of a divergence measure is a powerful tool for building more general and diverse reasoning models.

**arXiv ID:** 2509.07430
</details>

<details>
<summary><strong>Autonomous Continual Learning of Computer-Use Agents for Environment Adaptation</strong> - Tianci Xue, Zeyi Liao, Tianneng Shi, Zilu Wang, Kai Zhang, Dawn Song, Yu Su, Huan Sun - [[pdf]](https://arxiv.org/pdf/2602.10356)</summary>

**Abstract:** Real-world digital environments are highly diverse and dynamic. These characteristics cause agents to frequently encounter unseen scenarios and distribution shifts, making continual learning in specific environments essential for computer-use agents (CUAs). However, a key challenge lies in obtaining high-quality and environment-grounded agent data without relying on costly human annotation. In this work, we introduce ACuRL, an Autonomous Curriculum Reinforcement Learning framework that continually adapts agents to specific environments with zero human data. The agent first explores target environments to acquire initial experiences. During subsequent iterative training, a curriculum task generator leverages these experiences together with feedback from the previous iteration to synthesize new tasks tailored for the agent's current capabilities. To provide reliable reward signals, we introduce CUAJudge, a robust automatic evaluator for CUAs that achieves 93% agreement with human judgments. Empirically, our method effectively enables both intra-environment and cross-environment continual learning, yielding 4-22% performance gains without catastrophic forgetting on existing environments. Further analyses show highly sparse updates (e.g., 20% parameters), which helps explain the effective and robust adaptation. Our data and code are available at this https URL.

**arXiv ID:** 2602.10356
</details>

<details>
<summary><strong>What Makes Value Learning Efficient in Residual Reinforcement Learning?</strong> - Guozheng Ma, Lu Li, Haoyu Wang, Zixuan Liu, Pierre-Luc Bacon, Dacheng Tao - [[pdf]](https://arxiv.org/pdf/2602.10539)</summary>

**Abstract:** Residual reinforcement learning (RL) enables stable online refinement of expressive pretrained policies by freezing the base and learning only bounded corrections. However, value learning in residual RL poses unique challenges that remain poorly understood. In this work, we identify two key bottlenecks: cold start pathology, where the critic lacks knowledge of the value landscape around the base policy, and structural scale mismatch, where the residual contribution is dwarfed by the base action. Through systematic investigation, we uncover the mechanisms underlying these bottlenecks, revealing that simple yet principled solutions suffice: base-policy transitions serve as an essential value anchor for implicit warmup, and critic normalization effectively restores representation sensitivity for discerning value differences. Based on these insights, we propose DAWN (Data-Anchored Warmup and Normalization), a minimal approach targeting efficient value learning in residual RL. By addressing these bottlenecks, DAWN demonstrates substantial efficiency gains across diverse benchmarks, policy architectures, and observation modalities.

**arXiv ID:** 2602.10539
</details>

<details>
<summary><strong>From Natural Language to Materials Discovery:The Materials Knowledge Navigation Agent</strong> - Genmao Zhuang, Amir Barati Farimani - [[pdf]](https://arxiv.org/pdf/2602.11123)</summary>

**Abstract:** Accelerating the discovery of high-performance materials remains a central challenge across energy, electronics, and aerospace technologies, where traditional workflows depend heavily on expert intuition and computationally expensive simulations. Here we introduce the Materials Knowledge Navigation Agent (MKNA), a language-driven system that translates natural-language scientific intent into executable actions for database retrieval, property prediction, structure generation, and stability evaluation. Beyond automating tool invocation, MKNA autonomously extracts quantitative thresholds and chemically meaningful design motifs from literature and database evidence, enabling data-grounded hypothesis formation. Applied to the search for high-Debye-temperature ceramics, the agent identifies a literature-supported screening criterion (Theta_D > 800 K), rediscovers canonical ultra-stiff materials such as diamond, SiC, SiN, and BeO, and proposes thermodynamically stable, previously unreported Be-C-rich compounds that populate the sparsely explored 1500-1700 K regime. These results demonstrate that MKNA not only finds stable candidates but also reconstructs interpretable design heuristics, establishing a generalizable platform for autonomous, language-guided materials exploration.

**arXiv ID:** 2602.11123
</details>

<details>
<summary><strong>Asymmetric Prompt Weighting for Reinforcement Learning with Verifiable Rewards</strong> - Reinhard Heckel, Mahdi Soltanolkotabi, Christos Thramboulidis - [[pdf]](https://arxiv.org/pdf/2602.11128)</summary>

**Abstract:** Reinforcement learning with verifiable rewards has driven recent advances in LLM post-training, in particular for reasoning. Policy optimization algorithms generate a number of responses for a given prompt and then effectively weight the corresponding gradients depending on the rewards. The most popular algorithms including GRPO, DAPO, and RLOO focus on ambiguous prompts, i.e., prompts with intermediate success probability, while downgrading gradients with very easy and very hard prompts. In this paper, we consider asymmetric prompt weightings that assign higher weights to prompts with low, or even zero, empirical success probability. We find that asymmetric weighting particularly benefits from-scratch RL (as in R1-Zero), where training traverses a wide accuracy range, and less so in post-SFT RL where the model already starts at high accuracy. We also provide theory that characterizes prompt weights which minimize the time needed to raise success probability from an initial level to a target accuracy under a fixed update budget. In low-success regimes, where informative responses are rare and response cost dominates, these optimal weights become asymmetric, upweighting low success probabilities and thereby accelerating effective-time convergence.

**arXiv ID:** 2602.11128
</details>

<details>
<summary><strong>Why Agentic Theorem Prover Works: A Statistical Provability Theory of Mathematical Reasoning Models</strong> - Sho Sonoda, Shunta Akiyama, Yuya Uezato - [[pdf]](https://arxiv.org/pdf/2602.10538)</summary>

**Abstract:** Agentic theorem provers -- pipelines that couple a mathematical reasoning model with library retrieval, subgoal-decomposition/search planner, and a proof assistant verifier -- have recently achieved striking empirical success, yet it remains unclear which components drive performance and why such systems work at all despite classical hardness of proof search. We propose a distributional viewpoint and introduce **statistical provability**, defined as the finite-horizon success probability of reaching a verified proof, averaged over an instance distribution, and formalize modern theorem-proving pipelines as time-bounded MDPs. Exploiting Bellman structure, we prove existence of optimal policies under mild regularity, derive provability certificates via sub-/super-solution inequalities, and bound the performance gap of score-guided planning (greedy/top-\(k\)/beam/rollouts) in terms of approximation error, sequential statistical complexity, representation geometry (metric entropy/doubling structure), and action-gap margin tails. Together, our theory provides a principled, component-sensitive explanation of when and why agentic theorem provers succeed on biased real-world problem distributions, while clarifying limitations in worst-case or adversarial regimes.

**arXiv ID:** 2602.10538
</details>

<details>
<summary><strong>Goal-Conditioned Reinforcement Learning from Sub-Optimal Data on Metric Spaces</strong> - Alfredo Reichlin, Miguel Vasco, Hang Yin, Danica Kragic - [[pdf]](https://arxiv.org/pdf/2402.10820)</summary>

**Abstract:** We study the problem of learning optimal behavior from sub-optimal datasets for goal-conditioned offline reinforcement learning under sparse rewards, invertible actions and deterministic transitions. To mitigate the effects of \emph{distribution shift}, we propose MetricRL, a method that combines metric learning for value function approximation with weighted imitation learning for policy estimation. MetricRL avoids conservative or behavior-cloning constraints, enabling effective learning even in severely sub-optimal regimes. We introduce distance monotonicity as a key property linking metric representations to optimality and design an objective that explicitly promotes it. Empirically, MetricRL consistently outperforms prior state-of-the-art goal-conditioned RL methods in recovering near-optimal behavior from sub-optimal offline data.

**arXiv ID:** 2402.10820
</details>

<details>
<summary><strong>Hypercube Policy Regularization Framework for Offline Reinforcement Learning</strong> - Yi Shen, Hanyan Huang - [[pdf]](https://arxiv.org/pdf/2411.04534)</summary>

**Abstract:** Offline reinforcement learning has received extensive attention from scholars because it avoids the interaction between the agent and the environment by learning a policy through a static dataset. However, general reinforcement learning methods cannot get satisfactory results in offline reinforcement learning due to the out-of-distribution state actions that the dataset cannot cover during training. To solve this problem, the policy regularization method that tries to directly clone policies used in static datasets has received numerous studies due to its simplicity and effectiveness. However, policy constraint methods make the agent choose the corresponding actions in the static dataset. This type of constraint is usually over-conservative, which results in suboptimal policies, especially in low-quality static datasets. In this paper, a hypercube policy regularization framework is proposed, this method alleviates the constraints of policy constraint methods by allowing the agent to explore the actions corresponding to similar states in the static dataset, which increases the effectiveness of algorithms in low-quality datasets. It was also theoretically demonstrated that the hypercube policy regularization framework can effectively improve the performance of original algorithms. In addition, the hypercube policy regularization framework is combined with TD3-BC and Diffusion-QL for experiments on D4RL datasets which are called TD3-BC-C and Diffusion-QL-C. The experimental results of the score demonstrate that TD3-BC-C and Diffusion-QL-C perform better than state-of-the-art algorithms like IQL, CQL, TD3-BC and Diffusion-QL in most D4RL environments in approximate time.

**arXiv ID:** 2411.04534
</details>

<details>
<summary><strong>HypeRL: Hypernetwork-Based Reinforcement Learning for Control of Parametrized Dynamical Systems</strong> - Nicolò Botteghi, Stefania Fresca, Mengwu Guo, Andrea Manzoni - [[pdf]](https://arxiv.org/pdf/2501.04538)</summary>

**Abstract:** In this work, we devise a new, general-purpose reinforcement learning strategy for the optimal control of parametric dynamical systems. Such problems frequently arise in applied sciences and engineering and entail a significant complexity when control and/or state variables are distributed in high-dimensional space or depend on varying parameters. Traditional numerical methods, relying on either iterative minimization algorithms -- exploiting, e.g., the solution of the adjoint problem -- or dynamic programming -- also involving the solution of the Hamilton-Jacobi-Bellman (HJB) equation -- while reliable, often become computationally infeasible. In this paper, we propose HypeRL a deep reinforcement learning (DRL) framework to overcome the limitations shown by traditional methods. HypeRL aims at approximating the optimal control policy directly. Specifically, we employ an actor-critic DRL approach to learn an optimal feedback control strategy that can generalize across the range of variation of the parameters. To effectively learn such optimal control laws for different instances of the parameters, encoding the parameter information into the DRL policy and value function neural networks (NNs) is essential. HypeRL uses two additional NNs, called hypernetworks, to learn the weights and biases of the value function and the policy NNs. In this way, HypeRL effectively embeds the parametric information into the value function and policy. We validate the proposed approach on two parametric control problems, namely (I) a 1D parametric Kuramoto-Sivashinsky equation with in-domain control, and (ii) a navigation problem of particle dynamics in a parametric 2D gyre flow. We show that the knowledge of physical and task-dependent information and the encoding of this information via a hypernetwork, are essential ingredients for learning parameter-dependent control policies.

**arXiv ID:** 2501.04538
</details>

<details>
<summary><strong>Free-Flying Crew Cooperative Robots on the ISS: A Joint Review of Astrobee, CIMON, and Int-Ball Operations</strong> - Seiko Piotr Yamaguchi, Andres Mora Vargas, Till Eisenberg, Christian Rogon, Tatsuya Yamamoto, Shona Inoue, Christoph Kössl, Brian Coltin, Trey Smith, Jose V. Benavides - [[pdf]](https://arxiv.org/pdf/2602.10686)</summary>

**Abstract:** Intra-vehicular free-flying robots are anticipated to support various work in human spaceflight while working side-by-side with astronauts. Such example of robots includes NASA's Astrobee, DLR's CIMON, and JAXA's Int-Ball, which are deployed on the International Space Station. This paper presents the first joint analyses of these robot's shared experiences, co-authored by their development and operation team members. Despite the different origins and design philosophies, the development and operations of these platforms encountered various convergences. Hence, this paper presents a detailed overview of these robots, presenting their objectives, design, and onboard operations. Hence, joint lessons learned across the lifecycle are presented, from design to on-orbit operations. These lessons learned are anticipated to serve for future development and research as design recommendations.

**arXiv ID:** 2602.10686
</details>

<details>
<summary><strong>Multi-Task Reinforcement Learning of Drone Aerobatics by Exploiting Geometric Symmetries</strong> - Zhanyu Guo, Zikang Yin, Guobin Zhu, Shiliang Guo, Shiyu Zhao - [[pdf]](https://arxiv.org/pdf/2602.10997)</summary>

**Abstract:** Flight control for autonomous micro aerial vehicles (MAVs) is evolving from steady flight near equilibrium points toward more aggressive aerobatic maneuvers, such as flips, rolls, and Power Loop. Although reinforcement learning (RL) has shown great potential in these tasks, conventional RL methods often suffer from low data efficiency and limited generalization. This challenge becomes more pronounced in multi-task scenarios where a single policy is required to master multiple maneuvers. In this paper, we propose a novel end-to-end multi-task reinforcement learning framework, called GEAR (Geometric Equivariant Aerobatics Reinforcement), which fully exploits the inherent SO(2) rotational symmetry in MAV dynamics and explicitly incorporates this property into the policy network architecture. By integrating an equivariant actor network, FiLM-based task modulation, and a multi-head critic, GEAR achieves both efficiency and flexibility in learning diverse aerobatic maneuvers, enabling a data-efficient, robust, and unified framework for aerobatic control. GEAR attains a 98.85\% success rate across various aerobatic tasks, significantly outperforming baseline methods. In real-world experiments, GEAR demonstrates stable execution of multiple maneuvers and the capability to combine basic motion primitives to complete complex aerobatics.

**arXiv ID:** 2602.10997
</details>

<details>
<summary><strong>Assessing Vision-Language Models for Perception in Autonomous Underwater Robotic Software</strong> - Muhammad Yousaf, Aitor Arrieta, Shaukat Ali, Paolo Arcaini, Shuai Wang - [[pdf]](https://arxiv.org/pdf/2602.10655)</summary>

**Abstract:** Autonomous Underwater Robots (AURs) operate in challenging underwater environments, including low visibility and harsh water conditions. Such conditions present challenges for software engineers developing perception modules for the AUR software. To successfully carry out these tasks, deep learning has been incorporated into the AUR software to support its operations. However, the unique challenges of underwater environments pose difficulties for deep learning models, which often rely on labeled data that is scarce and noisy. This may undermine the trustworthiness of AUR software that relies on perception modules. Vision-Language Models (VLMs) offer promising solutions for AUR software as they generalize to unseen objects and remain robust in noisy conditions by inferring information from contextual cues. Despite this potential, their performance and uncertainty in underwater environments remain understudied from a software engineering perspective. Motivated by the needs of an industrial partner in assurance and risk management for maritime systems to assess the potential use of VLMs in this context, we present an empirical evaluation of VLM-based perception modules within the AUR software. We assess their ability to detect underwater trash by computing performance, uncertainty, and their relationship, to enable software engineers to select appropriate VLMs for their AUR software.

**arXiv ID:** 2602.10655
</details>

<details>
<summary><strong>From Steering to Pedalling: Do Autonomous Driving VLMs Generalize to Cyclist-Assistive Spatial Perception and Planning?</strong> - Krishna Kanth Nakka, Vedasri Nakka - [[pdf]](https://arxiv.org/pdf/2602.10771)</summary>

**Abstract:** Cyclists often encounter safety-critical situations in urban traffic, highlighting the need for assistive systems that support safe and informed decision-making. Recently, vision-language models (VLMs) have demonstrated strong performance on autonomous driving benchmarks, suggesting their potential for general traffic understanding and navigation-related reasoning. However, existing evaluations are predominantly vehicle-centric and fail to assess perception and reasoning from a cyclist-centric viewpoint. To address this gap, we introduce CyclingVQA, a diagnostic benchmark designed to probe perception, spatio-temporal understanding, and traffic-rule-to-lane reasoning from a cyclist's perspective. Evaluating 31+ recent VLMs spanning general-purpose, spatially enhanced, and autonomous-driving-specialized models, we find that current models demonstrate encouraging capabilities, while also revealing clear areas for improvement in cyclist-centric perception and reasoning, particularly in interpreting cyclist-specific traffic cues and associating signs with the correct navigational lanes. Notably, several driving-specialized models underperform strong generalist VLMs, indicating limited transfer from vehicle-centric training to cyclist-assistive scenarios. Finally, through systematic error analysis, we identify recurring failure modes to guide the development of more effective cyclist-assistive intelligent systems.

**arXiv ID:** 2602.10771
</details>

<details>
<summary><strong>HuMam: Humanoid Motion Control via End-to-End Deep Reinforcement Learning with Mamba</strong> - Yinuo Wang, Yuanyang Qi, Jinzhao Zhou, Pengxiang Meng, Xiaowen Tao - [[pdf]](https://arxiv.org/pdf/2509.18046)</summary>

**Abstract:** End-to-end reinforcement learning (RL) for humanoid locomotion is appealing for its compact perception-action mapping, yet practical policies often suffer from training instability, inefficient feature fusion, and high actuation cost. We present HuMam, a state-centric end-to-end RL framework that employs a single-layer Mamba encoder to fuse robot-centric states with oriented footstep targets and a continuous phase clock. The policy outputs joint position targets tracked by a low-level PD loop and is optimized with PPO. A concise six-term reward balances contact quality, swing smoothness, foot placement, posture, and body stability while implicitly promoting energy saving. On the JVRC-1 humanoid in mc-mujoco, HuMam consistently improves learning efficiency, training stability, and overall task performance over a strong feedforward baseline, while reducing power consumption and torque peaks. To our knowledge, this is the first end-to-end humanoid RL controller that adopts Mamba as the fusion backbone, demonstrating tangible gains in efficiency, stability, and control economy.

**arXiv ID:** 2509.18046
</details>

<details>
<summary><strong>Provably Optimal Reinforcement Learning under Safety Filtering</strong> - Donggeon David Oh, Duy P. Nguyen, Haimin Hu, Jaime F. Fisac - [[pdf]](https://arxiv.org/pdf/2510.18082)</summary>

**Abstract:** Recent advances in reinforcement learning (RL) enable its use on increasingly complex tasks, but the lack of formal safety guarantees still limits its application in safety-critical settings. A common practical approach is to augment the RL policy with a safety filter that overrides unsafe actions to prevent failures during both training and deployment. However, safety filtering is often perceived as sacrificing performance and hindering the learning process. We show that this perceived safety-performance tradeoff is not inherent and prove, for the first time, that enforcing safety with a sufficiently permissive safety filter does not degrade asymptotic performance. We formalize RL safety with a safety-critical Markov decision process (SC-MDP), which requires categorical, rather than high-probability, avoidance of catastrophic failure states. Additionally, we define an associated filtered MDP in which all actions result in safe effects, thanks to a safety filter that is considered to be a part of the environment. Our main theorem establishes that (i) learning in the filtered MDP is safe categorically, (ii) standard RL convergence carries over to the filtered MDP, and (iii) any policy that is optimal in the filtered MDP-when executed through the same filter-achieves the same asymptotic return as the best safe policy in the SC-MDP, yielding a complete separation between safety enforcement and performance optimization. We validate the theory on Safety Gymnasium with representative tasks and constraints, observing zero violations during training and final performance matching or exceeding unfiltered baselines. Together, these results shed light on a long-standing question in safety-filtered learning and provide a simple, principled recipe for safe RL: train and deploy RL policies with the most permissive safety filter that is available.

**arXiv ID:** 2510.18082
</details>

<details>
<summary><strong>Exploring the Interplay Between Voice, Personality, and Gender in Human-Agent Interactions</strong> - Kai Alexander Hackney, Lucas Guarenti Zangari, Jhonathan Sora-Cardenas, Emmanuel Munoz, Sterling R. Kalogeras, Betsy DiSalvo, Pedro Guillermo Feijoo-Garcia - [[pdf]](https://arxiv.org/pdf/2602.10535)</summary>

**Abstract:** To foster effective human-agent interactions, designers need to identify characteristics that could affect how agents are perceived and accepted, and to what extent they could impact rapport-building. Aiming to explore the role of user-agent synchrony, we assessed 388 participants to determine whether they could perceive personality traits from four artificial voices we selected and adapted from human samples, considering gender (male or female) and personality (introvert or extrovert) as grouping factors. Our findings suggest that participants were able to significantly differentiate female agents by personality, while male agents were not consistently distinguished. We also observed evidence of personality synchrony, where participants tended to perceive the first agent as more similar to their own personality, with this effect driven mainly by male participants, especially toward male agents. This paper contributes findings and insights to consider the interplay of user-agent personality and gender synchrony in the design of human-agent interactions.

**arXiv ID:** 2602.10535
</details>

<details>
<summary><strong>Don't blame me: How Intelligent Support Affects Moral Responsibility in Human Oversight</strong> - Cedric Faas, Richard Uth, Sarah Sterz, Markus Langer, Anna Maria Feit - [[pdf]](https://arxiv.org/pdf/2602.10701)</summary>

**Abstract:** AI-based systems can increasingly perform work tasks autonomously. In safety-critical tasks, human oversight of these systems is required to mitigate risks and to ensure responsibility in case something goes wrong. Since people often struggle to stay focused and perform good oversight, intelligent support systems are used to assist them, giving decision recommendations, alerting users, or restricting them from dangerous actions. However, in cases where recommendations are wrong, decision support might undermine the very reason why human oversight was employed -- genuine moral responsibility. The goal of our study was to investigate how a decision support system that restricted available interventions would affect overseer's perceived moral responsibility, in particular in cases where the support errs. In a simulated oversight experiment, participants (\textit{N}=274) monitored an autonomous drone that faced ten critical situations, choosing from six possible actions to resolve each situation. An AI system constrained participants' choices to either six, four, two, or only one option (between-subject study). Results showed that participants, who were restricted to choosing from a single action, felt less morally responsible if a crash occurred. At the same time, participants' judgments about the responsibility of other stakeholders (the AI; the developer of the AI) did not change between conditions. Our findings provide important insights for user interface design and oversight architectures: they should prevent users from attributing moral agency to AI, help them understand how moral responsibility is distributed, and, when oversight aims to prevent ethically undesirable outcomes, be designed to support the epistemic and causal conditions required for moral responsibility.

**arXiv ID:** 2602.10701
</details>

<details>
<summary><strong>GenFaceUI: Meta-Design of Generative Personalized Facial Expression Interfaces for Intelligent Agents</strong> - Yate Ge, Lin Tian, Yi Dai, Shuhan Pan, Yiwen Zhang, Qi Wang, Weiwei Guo, Xiaohua Sun - [[pdf]](https://arxiv.org/pdf/2602.11055)</summary>

**Abstract:** This work investigates generative facial expression interfaces for intelligent agents from a meta-design perspective. We propose the Generative Personalized Facial Expression Interface (GPFEI) framework, which organizes rule-bounded spaces, character identity, and context--expression mapping to address challenges of control, coherence, and alignment in run-time facial expression generation. To operationalize this framework, we developed GenFaceUI, a proof-of-concept tool that enables designers to create templates, apply semantic tags, define rules, and iteratively test outcomes. We evaluated the tool through a qualitative study with twelve designers. The results show perceived gains in controllability and consistency, while revealing needs for structured visual mechanisms and lightweight explanations. These findings provide a conceptual framework, a proof-of-concept tool, and empirical insights that highlight both opportunities and challenges for advancing generative facial expression interfaces within a broader meta-design paradigm.

**arXiv ID:** 2602.11055
</details>

<details>
<summary><strong>Generative Muscle Stimulation: Providing Users with Physical Assistance by Constraining Multimodal-AI with Embodied Knowledge</strong> - Yun Ho, Romain Nith, Peili Jiang, Steven He, Bruno Felalaga, Shan-Yuan Teng, Rhea Seeralan, Pedro Lopes - [[pdf]](https://arxiv.org/pdf/2505.10648)</summary>

**Abstract:** Electrical muscle stimulation (EMS) can support physical-assistance (e.g., shaking a spray-can before painting). However, EMS-assistance is highly-specialized because it is (1) fixed (e.g., one program for shaking spray-cans, another for opening windows); and (2) non-contextual (e.g., a spray-can for cooking dispenses cooking-oil, not paint-shaking it is unnecessary). Instead, we explore a different approach where muscle-stimulation instructions are generated considering the user's context (e.g., pose, location, surroundings). The resulting system is more general-enabling unprecedented EMS interactions (e.g., opening a pill bottle) yet also replicating existing systems (e.g., Affordance++) without task-specific programming. It uses computer-vision/large-language-models to generate EMS-instructions, constraining these to a muscle-stimulation knowledge-base & joint-limits. In our user-study, we found participants successfully completed physical-tasks while guided by generative-EMS, even when EMS-instructions were (purposely) erroneous. Participants understood generated gestures and, even during forced-errors, understood partial-instructions, identified errors, and re-prompted the system. We believe our concept marks a shift toward more general-purpose EMS-interfaces.

**arXiv ID:** 2505.10648
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
