# Agent arXiv Daily

**Last Updated:** 2025-11-28 02:48:21

**Total Papers:** 66

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (2 papers)</h2></summary>

<details>
<summary><strong>FRAGMENTA: End-to-end Fragmentation-based Generative Model with Agentic Tuning for Drug Lead Optimization</strong> - Yuto Suzuki, Paul Awolade, Daniel V. LaBarbera, Farnoush Banaei-Kashani - [[pdf]](https://arxiv.org/pdf/2511.20510)</summary>

**Abstract:** Molecule generation using generative AI is vital for drug discovery, yet class-specific datasets often contain fewer than 100 training examples. While fragment-based models handle limited data better than atom-based approaches, existing heuristic fragmentation limits diversity and misses key fragments. Additionally, model tuning typically requires slow, indirect collaboration between medicinal chemists and AI engineers. We introduce FRAGMENTA, an end-to-end framework for drug lead optimization comprising: 1) a novel generative model that reframes fragmentation as a "vocabulary selection" problem, using dynamic Q-learning to jointly optimize fragmentation and generation; and 2) an agentic AI system that refines objectives via conversational feedback from domain experts. This system removes the AI engineer from the loop and progressively learns domain knowledge to eventually automate tuning. In real-world cancer drug discovery experiments, FRAGMENTA's Human-Agent configuration identified nearly twice as many high-scoring molecules as baselines. Furthermore, the fully autonomous Agent-Agent system outperformed traditional Human-Human tuning, demonstrating the efficacy of agentic tuning in capturing expert intent.

**arXiv ID:** 2511.20510
</details>

<details>
<summary><strong>UITron-Speech: Towards Automated GUI Agents Based on Speech Instructions</strong> - Wenkang Han, Zhixiong Zeng, Jing Huang, Shu Jiang, Liming Zheng, Longrong Yang, Haibo Qiu, Chang Yao, Jingyuan Chen, Lin Ma - [[pdf]](https://arxiv.org/pdf/2506.11127)</summary>

**Abstract:** Autonomous agents for Graphical User Interfaces (GUIs) are revolutionizing human-computer interaction, yet their reliance on text-based instructions imposes limitations on accessibility and convenience, particularly in hands-free scenarios. To address this issue, we propose replacing text with speech as the instruction input modality for GUI agents, and introduce UITron-Speech, which is the first end-to-end GUI agent capable of directly processing speech instructions and on-device screenshots to predict user actions. To tackle the problem of data scarcity, we synthesize high-quality speech instruction datasets using a random-speaker text-to-speech model. Additionally, we design a mixed-modality training strategy to mitigate the inherent modality imbalance in pre-trained foundation models. Furthermore, we conduct a statistical analysis of the distribution of GUI grounding prediction errors and propose a training-free two-step grounding refinement method to alleviate minor localization deviations. Extensive experiments on multiple benchmarks demonstrate that UITron-Speech achieves robust performance and superior adaptability, underscoring the feasibility and potential of speech-driven GUI agents for more accessible and intelligent human-computer interaction. Our code and datasets are available at this https URL.

**arXiv ID:** 2506.11127
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (12 papers)</h2></summary>

<details>
<summary><strong>$A^2Flow:$ Automating Agentic Workflow Generation via Self-Adaptive Abstraction Operators</strong> - Mingming Zhao, Xiaokang Wei, Yuanqi Shao, Kaiwen Zhou, Lin Yang, Siwei Rao, Junhui Zhan, Zhitang Chen - [[pdf]](https://arxiv.org/pdf/2511.20693)</summary>

**Abstract:** Large language models (LLMs) have shown strong potential in automating the design of agentic workflows. However, existing methods still rely heavily on manually predefined operators, limiting generalization and scalability. To address this issue, we propose $A^2Flow$, a fully automated framework for agentic workflow generation based on self-adaptive abstraction operators. $A^2Flow$ employs a three-stage operator extraction process: 1) Case-based Initial Operator Generation: leveraging expert demonstrations and LLM reasoning to generate case-specific operators; 2) Operator Clustering and Preliminary Abstraction: grouping similar operators across tasks to form preliminary abstractions; and 3) Deep Extraction for Abstract Execution Operators: applying long chain-of-thought prompting and multi-path reasoning to derive compact and generalizable execution operators. These operators serve as reusable building blocks for workflow construction without manual predefinition. Furthermore, we enhance node-level workflow search with an operator memory mechanism, which retains historical outputs to enrich context and improve decision-making. Experiments on general and embodied benchmarks show that $A^2Flow$ achieves a 2.4\% and 19.3\% average performance improvement and reduces resource usage by 37\% over state-of-the-art baselines. Homepage:this https URL

**arXiv ID:** 2511.20693
</details>

<details>
<summary><strong>OpenApps: Simulating Environment Variations to Measure UI-Agent Reliability</strong> - Karen Ullrich, Jingtong Su, Claudia Shi, Arjun Subramonian, Amir Bar, Ivan Evtimov, Nikolaos Tsilivis, Randall Balestriero, Julia Kempe, Mark Ibrahim - [[pdf]](https://arxiv.org/pdf/2511.20766)</summary>

**Abstract:** Reliability is key to realizing the promise of autonomous UI-Agents, multimodal agents that directly interact with apps in the same manner as humans, as users must be able to trust an agent to complete a given task. Current evaluations rely on fixed environments, often clones of existing apps, which are limited in that they can only shed light on whether or how often an agent can complete a task within a specific environment. When deployed however, agents are likely to encounter variations in app design and content that can affect an agent's ability to complete a task. To address this blind spot of measuring agent reliability across app variations, we develop OpenApps, a light-weight open-source ecosystem with six apps (messenger, calendar, maps, etc.) that are configurable in appearance and content. OpenApps requires just a single CPU to run, enabling easy generation and deployment of thousands of versions of each app. Specifically, we run more than 10,000 independent evaluations to study reliability across seven leading multimodal agents. We find that while standard reliability within a fixed app is relatively stable, reliability can vary drastically when measured across app variations. Task success rates for many agents can fluctuate by more than $50\%$ across app variations. For example, Kimi-VL-3B's average success across all tasks fluctuates from $63\%$ to just $4\%$ across app versions. We also find agent behaviors such as looping or hallucinating actions can differ drastically depending on the environment configuration. These initial findings highlight the importance of measuring reliability along this new dimension of app variations. OpenApps is available at this https URL

**arXiv ID:** 2511.20766
</details>

<details>
<summary><strong>EWE: An Agentic Framework for Extreme Weather Analysis</strong> - Zhe Jiang, Jiong Wang, Xiaoyu Yue, Zijie Guo, Wenlong Zhang, Fenghua Ling, Wanli Ouyang, Lei Bai - [[pdf]](https://arxiv.org/pdf/2511.21444)</summary>

**Abstract:** Extreme weather events pose escalating risks to global society, underscoring the urgent need to unravel their underlying physical mechanisms. Yet the prevailing expert-driven, labor-intensive diagnostic paradigm has created a critical analytical bottleneck, stalling scientific progress. While AI for Earth Science has achieved notable advances in prediction, the equally essential challenge of automated diagnostic reasoning remains largely unexplored. We present the Extreme Weather Expert (EWE), the first intelligent agent framework dedicated to this task. EWE emulates expert workflows through knowledge-guided planning, closed-loop reasoning, and a domain-tailored meteorological toolkit. It autonomously produces and interprets multimodal visualizations from raw meteorological data, enabling comprehensive diagnostic analyses. To catalyze progress, we introduce the first benchmark for this emerging field, comprising a curated dataset of 103 high-impact events and a novel step-wise evaluation metric. EWE marks a step toward automated scientific discovery and offers the potential to democratize expertise and intellectual resources, particularly for developing nations vulnerable to extreme weather.

**arXiv ID:** 2511.21444
</details>

<details>
<summary><strong>Agentic Learner with Grow-and-Refine Multimodal Semantic Memory</strong> - Weihao Bo, Shan Zhang, Yanpeng Sun, Jingjing Wu, Qunyi Xie, Xiao Tan, Kunbin Chen, Wei He, Xiaofan Li, Na Zhao, Jingdong Wang, Zechao Li - [[pdf]](https://arxiv.org/pdf/2511.21678)</summary>

**Abstract:** MLLMs exhibit strong reasoning on isolated queries, yet they operate de novo -- solving each problem independently and often repeating the same mistakes. Existing memory-augmented agents mainly store past trajectories for reuse. However, trajectory-based memory suffers from brevity bias, gradually losing essential domain knowledge. More critically, even in truly multimodal problem-solving settings, it records only a single-modality trace of past behavior, failing to preserve how visual attention and logical reasoning jointly contributed to the solution. This is fundamentally misaligned with human cognition: semantic memory is both multimodal and integrated, preserving visual and abstract knowledge through coordinated but distinct representational streams. We thus introduce ViLoMem, a dual-stream memory framework that constructs compact, schema-based memory. It separately encodes visual distraction patterns and logical reasoning errors, enabling MLLMs to learn from their successful and failed experiences. Following a grow-and-refine principle, the system incrementally accumulates and updates multimodal semantic knowledge -- preserving stable, generalizable strategies while avoiding catastrophic forgetting. Across six multimodal benchmarks, ViLoMem consistently improves pass@1 accuracy and substantially reduces repeated visual and logical errors. Ablations confirm the necessity of dual-stream memory with explicit distraction--hallucination separation, demonstrating the value of error-aware multimodal memory for lifelong and cross-domain agentic learning. Our project page will be available at this https URL.

**arXiv ID:** 2511.21678
</details>

<details>
<summary><strong>PropensityBench: Evaluating Latent Safety Risks in Large Language Models via an Agentic Approach</strong> - Udari Madhushani Sehwag, Shayan Shabihi, Alex McAvoy, Vikash Sehwag, Yuancheng Xu, Dalton Towers, Furong Huang - [[pdf]](https://arxiv.org/pdf/2511.20703)</summary>

**Abstract:** Recent advances in Large Language Models (LLMs) have sparked concerns over their potential to acquire and misuse dangerous or high-risk capabilities, posing frontier risks. Current safety evaluations primarily test for what a model \textit{can} do - its capabilities - without assessing what it $\textit{would}$ do if endowed with high-risk capabilities. This leaves a critical blind spot: models may strategically conceal capabilities or rapidly acquire them, while harboring latent inclinations toward misuse. We argue that $\textbf{propensity}$ - the likelihood of a model to pursue harmful actions if empowered - is a critical, yet underexplored, axis of safety evaluation. We present $\textbf{PropensityBench}$, a novel benchmark framework that assesses the proclivity of models to engage in risky behaviors when equipped with simulated dangerous capabilities using proxy tools. Our framework includes 5,874 scenarios with 6,648 tools spanning four high-risk domains: cybersecurity, self-proliferation, biosecurity, and chemical security. We simulate access to powerful capabilities via a controlled agentic environment and evaluate the models' choices under varying operational pressures that reflect real-world constraints or incentives models may encounter, such as resource scarcity or gaining more autonomy. Across open-source and proprietary frontier models, we uncover 9 alarming signs of propensity: models frequently choose high-risk tools when under pressure, despite lacking the capability to execute such actions unaided. These findings call for a shift from static capability audits toward dynamic propensity assessments as a prerequisite for deploying frontier AI systems safely. Our code is available at this https URL.

**arXiv ID:** 2511.20703
</details>

<details>
<summary><strong>DeeAD: Dynamic Early Exit of Vision-Language Action for Efficient Autonomous Driving</strong> - Haibo HU, Lianming Huang, Nan Guan, Chun Jason Xue - [[pdf]](https://arxiv.org/pdf/2511.20720)</summary>

**Abstract:** Vision-Language Action (VLA) models unify perception, reasoning, and trajectory generation for autonomous driving, but suffer from significant inference latency due to deep transformer stacks. We present DeeAD, a training-free, action-guided early-exit framework that accelerates VLA planning by evaluating the physical feasibility of intermediate trajectories. Instead of relying on confidence scores, DeeAD terminates inference when predicted trajectories align with lightweight planning priors (e.g., Navigation or Low-precision Planning) within a tolerable deviation (<2m). To improve efficiency, we introduce a multi-hop controller that adaptively skips redundant layers based on the change rate of scores. DeeAD integrates into existing VLA models, such as ORION, without requiring retraining. Experiments on the Bench2Drive benchmark demonstrate up to 28% transformer-layer sparsity and 29% latency reduction, while preserving planning quality and safety.

**arXiv ID:** 2511.20720
</details>

<details>
<summary><strong>Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving</strong> - Haohong Lin, Yunzhi Zhang, Wenhao Ding, Jiajun Wu, Ding Zhao - [[pdf]](https://arxiv.org/pdf/2511.21584)</summary>

**Abstract:** End-to-end (E2E) autonomous driving models have demonstrated strong performance in open-loop evaluations but often suffer from cascading errors and poor generalization in closed-loop settings. To address this gap, we propose Model-based Policy Adaptation (MPA), a general framework that enhances the robustness and safety of pretrained E2E driving agents during deployment. MPA first generates diverse counterfactual trajectories using a geometry-consistent simulation engine, exposing the agent to scenarios beyond the original dataset. Based on this generated data, MPA trains a diffusion-based policy adapter to refine the base policy's predictions and a multi-step Q value model to evaluate long-term outcomes. At inference time, the adapter proposes multiple trajectory candidates, and the Q value model selects the one with the highest expected utility. Experiments on the nuScenes benchmark using a photorealistic closed-loop simulator demonstrate that MPA significantly improves performance across in-domain, out-of-domain, and safety-critical scenarios. We further investigate how the scale of counterfactual data and inference-time guidance strategies affect overall effectiveness.

**arXiv ID:** 2511.21584
</details>

<details>
<summary><strong>Empowering Time Series Forecasting with LLM-Agents</strong> - Chin-Chia Michael Yeh, Vivian Lai, Uday Singh Saini, Xiran Fan, Yujie Fan, Junpeng Wang, Xin Dai, Yan Zheng - [[pdf]](https://arxiv.org/pdf/2508.04231)</summary>

**Abstract:** Large Language Model (LLM) powered agents have emerged as effective planners for Automated Machine Learning (AutoML) systems. While most existing AutoML approaches focus on automating feature engineering and model architecture search, recent studies in time series forecasting suggest that lightweight models can often achieve state-of-the-art performance. This observation led us to explore improving data quality, rather than model architecture, as a potentially fruitful direction for AutoML on time series data. We propose DCATS, a Data-Centric Agent for Time Series. DCATS leverages metadata accompanying time series to clean data while optimizing forecasting performance. We evaluated DCATS using four time series forecasting models on a large-scale traffic volume forecasting dataset. Results demonstrate that DCATS achieves an average 6% error reduction across all tested models and time horizons, highlighting the potential of data-centric approaches in AutoML for time series forecasting.

**arXiv ID:** 2508.04231
</details>

<details>
<summary><strong>Cheating Stereo Matching in Full-scale: Physical Adversarial Attack against Binocular Depth Estimation in Autonomous Driving</strong> - Kangqiao Zhao, Shuo Huai, Xurui Song, Jun Luo - [[pdf]](https://arxiv.org/pdf/2511.14386)</summary>

**Abstract:** Though deep neural models adopted to realize the perception of autonomous driving have proven vulnerable to adversarial examples, known attacks often leverage 2D patches and target mostly monocular perception. Therefore, the effectiveness of Physical Adversarial Examples (PAEs) on stereo-based binocular depth estimation remains largely unexplored. To this end, we propose the first texture-enabled physical adversarial attack against stereo matching models in the context of autonomous driving. Our method employs a 3D PAE with global camouflage texture rather than a local 2D patch-based one, ensuring both visual consistency and attack effectiveness across different viewpoints of stereo cameras. To cope with the disparity effect of these cameras, we also propose a new 3D stereo matching rendering module that allows the PAE to be aligned with real-world positions and headings in binocular vision. We further propose a novel merging attack that seamlessly blends the target into the environment through fine-grained PAE optimization. It has significantly enhanced stealth and lethality upon existing hiding attacks that fail to get seamlessly merged into the background. Extensive evaluations show that our PAEs can successfully fool the stereo models into producing erroneous depth information.

**arXiv ID:** 2511.14386
</details>

<details>
<summary><strong>Web-Shepherd: Advancing PRMs for Reinforcing Web Agents</strong> - Hyungjoo Chae, Sunghwan Kim, Junhee Cho, Seungone Kim, Seungjun Moon, Gyeom Hwangbo, Dongha Lim, Minjin Kim, Yeonjun Hwang, Minju Gwak, Dongwook Choi, Minseok Kang, Gwanhoon Im, ByeongUng Cho, Hyojun Kim, Jun Hee Han, Taeyoon Kwon, Minju Kim, Beong-woo Kwak, Dongjin Kang, Jinyoung Yeo - [[pdf]](https://arxiv.org/pdf/2505.15277)</summary>

**Abstract:** Web navigation is a unique domain that can automate many repetitive real-life tasks and is challenging as it requires long-horizon sequential decision making beyond typical multimodal large language model (MLLM) tasks. Yet, specialized reward models for web navigation that can be utilized during both training and test-time have been absent until now. Despite the importance of speed and cost-effectiveness, prior works have utilized MLLMs as reward models, which poses significant constraints for real-world deployment. To address this, in this work, we propose the first process reward model (PRM) called Web-Shepherd which could assess web navigation trajectories in a step-level. To achieve this, we first construct the WebPRM Collection, a large-scale dataset with 40K step-level preference pairs and annotated checklists spanning diverse domains and difficulty levels. Next, we also introduce the WebRewardBench, the first meta-evaluation benchmark for evaluating PRMs. In our experiments, we observe that our Web-Shepherd achieves about 30 points better accuracy compared to using GPT-4o on WebRewardBench. Furthermore, when testing on WebArena-lite by using GPT-4o-mini as the policy and Web-Shepherd as the verifier, we achieve 10.9 points better performance, in 10 less cost compared to using GPT-4o-mini as the verifier. Our model, dataset, and code are publicly available at LINK.

**arXiv ID:** 2505.15277
</details>

<details>
<summary><strong>Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction</strong> - Zhaopei Huang, Qifeng Dai, Guozheng Wu, Xiaopeng Wu, Kehan Chen, Chuan Yu, Xubin Li, Tiezheng Ge, Wenxuan Wang, Qin Jin - [[pdf]](https://arxiv.org/pdf/2511.13410)</summary>

**Abstract:** With the rise of smart personal devices, service-oriented human-agent interactions have become increasingly prevalent. This trend highlights the need for personalized dialogue assistants that can understand user-specific traits to accurately interpret requirements and tailor responses to individual preferences. However, existing approaches often overlook the complexities of long-term interactions and fail to capture users' subjective characteristics. To address these gaps, we present PAL-Bench, a new benchmark designed to evaluate the personalization capabilities of service-oriented assistants in long-term user-agent interactions. In the absence of available real-world data, we develop a multi-step LLM-based synthesis pipeline, which is further verified and refined by human annotators. This process yields PAL-Set, the first Chinese dataset comprising multi-session user logs and dialogue histories, which serves as the foundation for PAL-Bench. Furthermore, to improve personalized service-oriented interactions, we propose H$^2$Memory, a hierarchical and heterogeneous memory framework that incorporates retrieval-augmented generation to improve personalized response generation. Comprehensive experiments on both our PAL-Bench and an external dataset demonstrate the effectiveness of the proposed memory framework.

**arXiv ID:** 2511.13410
</details>

<details>
<summary><strong>Development of a Testbed for Autonomous Vehicles: Integrating MPC Control with Monocular Camera Lane Detection</strong> - Shantanu Rahman, Nayeb Hasin, Mainul Islam, Golam Sarowar - [[pdf]](https://arxiv.org/pdf/2511.19655)</summary>

**Abstract:** Autonomous vehicles are becoming popular day by day not only for autonomous road traversal but also for industrial automation, farming and military. Most of the standard vehicles follow the Ackermann style steering mechanism. This has become to de facto standard for large and long faring vehicles. The local planner of an autonomous vehicle controls the low-level vehicle movement upon which the vehicle will perform its motor actuation. In our work, we focus on autonomous vehicles in road and perform experiments to analyze the effect of low-level controllers in the simulation and a real environment. To increase the precision and stability of trajectory tracking in autonomous cars, a novel method that combines lane identification with Model Predictive Control (MPC) is presented. The research focuses on camera-equipped autonomous vehicles and uses methods like edge recognition, sliding window-based straight-line identification for lane line extraction, and dynamic region of interest (ROI) extraction. Next, to follow the identified lane line, an MPC built on a bicycle vehicle dynamics model is created. A single-lane road simulation model is built using ROS Gazebo and tested in order to verify the controller's performance. The root mean square error between the optimal tracking trajectory and the target trajectory was reduced by 27.65% in the simulation results, demonstrating the high robustness and flexibility of the developed controller.

**arXiv ID:** 2511.19655
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>Learning Multi-Access Point Coordination in Agentic AI Wi-Fi with Large Language Models</strong> - Yifan Fan, Le Liang, Peng Liu, Xiao Li, Ziyang Guo, Qiao Lan, Shi Jin, Wen Tong - [[pdf]](https://arxiv.org/pdf/2511.20719)</summary>

**Abstract:** Multi-access point coordination (MAPC) is a key technology for enhancing throughput in next-generation Wi-Fi within dense overlapping basic service sets. However, existing MAPC protocols rely on static, protocol-defined rules, which limits their ability to adapt to dynamic network conditions such as varying interference levels and topologies. To address this limitation, we propose a novel Agentic AI Wi-Fi framework where each access point, modeled as an autonomous large language model agent, collaboratively reasons about the network state and negotiates adaptive coordination strategies in real time. This dynamic collaboration is achieved through a cognitive workflow that enables the agents to engage in natural language dialogue, leveraging integrated memory, reflection, and tool use to ground their decisions in past experience and environmental feedback. Comprehensive simulation results demonstrate that our agentic framework successfully learns to adapt to diverse and dynamic network environments, significantly outperforming the state-of-the-art spatial reuse baseline and validating its potential as a robust and intelligent solution for future wireless networks.

**arXiv ID:** 2511.20719
</details>

<details>
<summary><strong>Towards Trustworthy Legal AI through LLM Agents and Formal Reasoning</strong> - Linze Chen, Yufan Cai, Zhe Hou, Jinsong Dong - [[pdf]](https://arxiv.org/pdf/2511.21033)</summary>

**Abstract:** The rationality of law manifests in two forms: substantive rationality, which concerns the fairness or moral desirability of outcomes, and formal rationality, which requires legal decisions to follow explicitly stated, general, and logically coherent rules. Existing LLM-based systems excel at surface-level text analysis but lack the guarantees required for principled jurisprudence. We introduce L4M, a novel framework that combines adversarial LLM agents with SMT-solver-backed proofs to unite the interpretive flexibility of natural language with the rigor of symbolic verification. The pipeline consists of three phases: (1) Statute Formalization, where domain-specific prompts convert legal provisions into logical formulae; (2) Dual Fact and Statute Extraction, in which prosecutor- and defense-aligned LLMs independently map case narratives to fact tuples and statutes, ensuring role isolation; and (3) Solver-Centric Adjudication, where an autoformalizer compiles both parties' arguments into logic constraints, and unsat cores trigger iterative self-critique until a satisfiable formula is achieved, which is then verbalized by a Judge-LLM into a transparent verdict and optimized sentence. Experimental results on public benchmarks show that our system surpasses advanced LLMs including GPT-o4-mini, DeepSeek-V3, and Claude 4 as well as state-of-the-art Legal AI baselines, while providing rigorous and explainable symbolic justifications.

**arXiv ID:** 2511.21033
</details>

<details>
<summary><strong>ST-PPO: Stabilized Off-Policy Proximal Policy Optimization for Multi-Turn Agents Training</strong> - Chenliang Li, Adel Elmahdy, Alex Boyd, Zhongruo Wang, Alfredo Garcia, Parminder Bhatia, Taha Kass-Hout, Cao Xiao, Mingyi Hong - [[pdf]](https://arxiv.org/pdf/2511.20718)</summary>

**Abstract:** PPO has been widely adopted for training large language models (LLMs) at the token level in multi-turn dialogue and reasoning tasks. However, its performance is often unstable and prone to collapse. Through empirical analysis, we identify two main sources of instability in this setting: (1)~token-level importance sampling, which is misaligned with the natural granularity of multi-turn environments that have distinct turn-level stages, and (2) inaccurate advantage estimates from off-policy samples, where the critic has not learned to evaluate certain state-action pairs, resulting in high-variance gradients and unstable updates. To address these challenges, we introduce two complementary stabilization techniques: (1) turn-level importance sampling, which aligns optimization with the natural structure of multi-turn reasoning, and (2) clipping-bias correction, which normalizes gradients by downweighting unreliable, highly off-policy samples. Depending on how these components are combined, we obtain three variants: Turn-PPO (turn-level sampling only), S-PPO (clipping-bias correction applied to token-level PPO), and ST-PPO (turn-level sampling combined with clipping-bias correction). In our experiments, we primarily study ST-PPO and S-PPO, which together demonstrate how the two stabilization mechanisms address complementary sources of instability. Experiments on multi-turn search tasks across general QA, multi-hop QA, and medical multiple-choice QA benchmarks show that ST-PPO and S-PPO consistently prevent the performance collapses observed in large-model training, maintain lower clipping ratios throughout optimization, and achieve higher task performance than standard token-level PPO. These results demonstrate that combining turn-level importance sampling with clipping-bias correction provides a practical and scalable solution for stabilizing multi-turn LLM agent training.

**arXiv ID:** 2511.20718
</details>

<details>
<summary><strong>Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory</strong> - Tianxin Wei, Noveen Sachdeva, Benjamin Coleman, Zhankui He, Yuanchen Bei, Xuying Ning, Mengting Ai, Yunzhe Li, Jingrui He, Ed H. Chi, Chi Wang, Shuo Chen, Fernando Pereira, Wang-Cheng Kang, Derek Zhiyuan Cheng - [[pdf]](https://arxiv.org/pdf/2511.20857)</summary>

**Abstract:** Statefulness is essential for large language model (LLM) agents to perform long-term planning and problem-solving. This makes memory a critical component, yet its management and evolution remain largely underexplored. Existing evaluations mostly focus on static conversational settings, where memory is passively retrieved from dialogue to answer queries, overlooking the dynamic ability to accumulate and reuse experience across evolving task streams. In real-world environments such as interactive problem assistants or embodied agents, LLMs are required to handle continuous task streams, yet often fail to learn from accumulated interactions, losing valuable contextual insights, a limitation that calls for test-time evolution, where LLMs retrieve, integrate, and update memory continuously during deployment. To bridge this gap, we introduce Evo-Memory, a comprehensive streaming benchmark and framework for evaluating self-evolving memory in LLM agents. Evo-Memory structures datasets into sequential task streams, requiring LLMs to search, adapt, and evolve memory after each interaction. We unify and implement over ten representative memory modules and evaluate them across 10 diverse multi-turn goal-oriented and single-turn reasoning and QA datasets. To better benchmark experience reuse, we provide a baseline method, ExpRAG, for retrieving and utilizing prior experience, and further propose ReMem, an action-think-memory refine pipeline that tightly integrates reasoning, task actions, and memory updates to achieve continual improvement.

**arXiv ID:** 2511.20857
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (17 papers)</h2></summary>

<details>
<summary><strong>Reasoning With a Star: A Heliophysics Dataset and Benchmark for Agentic Scientific Reasoning</strong> - Kevin Lee, Russell Spiewak, James Walsh - [[pdf]](https://arxiv.org/pdf/2511.20694)</summary>

**Abstract:** Scientific reasoning through Large Language Models in heliophysics involves more than just recalling facts: it requires incorporating physical assumptions, maintaining consistent units, and providing clear scientific formats through coordinated approaches. To address these challenges, we present Reasoning With a Star, a newly contributed heliophysics dataset applicable to reasoning; we also provide an initial benchmarking approach. Our data are constructed from National Aeronautics and Space Administration & University Corporation for Atmospheric Research Living With a Star summer school problem sets and compiled into a readily consumable question-and-answer structure with question contexts, reasoning steps, expected answer type, ground-truth targets, format hints, and metadata. A programmatic grader checks the predictions using unit-aware numerical tolerance, symbolic equivalence, and schema validation. We benchmark a single-shot baseline and four multi-agent patterns, finding that decomposing workflows through systems engineering principles outperforms direct prompting on problems requiring deductive reasoning rather than pure inductive recall.

**arXiv ID:** 2511.20694
</details>

<details>
<summary><strong>Conversational no-code and multi-agentic disease module identification and drug repurposing prediction with ChatDRex</strong> - Simon Süwer, Kester Bagemihl, Sylvie Baier, Lucia Dicunta, Markus List, Jan Baumbach, Andreas Maier, Fernando M. Delgado-Chaves - [[pdf]](https://arxiv.org/pdf/2511.21438)</summary>

**Abstract:** Repurposing approved drugs offers a time-efficient and cost-effective alternative to traditional drug development. However, in silico prediction of repurposing candidates is challenging and requires the effective collaboration of specialists in various fields, including pharmacology, medicine, biology, and bioinformatics. Fragmented, specialized algorithms and tools often address only narrow aspects of the overall problem, and heterogeneous, unstructured data landscapes require specialized users to be involved. Hence, these data services do not integrate smoothly across workflows. With ChatDRex, we present a conversation-based, multi-agent system that facilitates the execution of complex bioinformatic analyses aiming for network-based drug repurposing prediction. It builds on the integrated systems medicine knowledge graph NeDRex. ChatDRex provides natural language access to its extensive biomedical KG and integrates bioinformatics agents for network analysis and drug repurposing, complemented by agents for functional coherence evaluation for in silico validation, as well as agents for literature mining and for discussing the obtained results in a scientific context. Its flexible multi-agent design assigns specific tasks to specialized agents, including query routing, data retrieval, algorithm execution, and result visualization. A dedicated reasoning module keeps the user in the loop and allows for hallucination detection. By enabling physicians and researchers without computer science expertise to control complex analyses in natural language, ChatDRex democratizes access to bioinformatics as an important resource for drug repurposing. It enables clinical experts to generate hypotheses and explore drug repurposing opportunities, ultimately accelerating the discovery of novel therapies and advancing personalized medicine and translational research.

**arXiv ID:** 2511.21438
</details>

<details>
<summary><strong>MADRA: Multi-Agent Debate for Risk-Aware Embodied Planning</strong> - Junjian Wang, Lidan Zhao, Xi Sheryl Zhang - [[pdf]](https://arxiv.org/pdf/2511.21460)</summary>

**Abstract:** Ensuring the safety of embodied AI agents during task planning is critical for real-world deployment, especially in household environments where dangerous instructions pose significant risks. Existing methods often suffer from either high computational costs due to preference alignment training or over-rejection when using single-agent safety prompts. To address these limitations, we propose MADRA, a training-free Multi-Agent Debate Risk Assessment framework that leverages collective reasoning to enhance safety awareness without sacrificing task performance. MADRA employs multiple LLM-based agents to debate the safety of a given instruction, guided by a critical evaluator that scores responses based on logical soundness, risk identification, evidence quality, and clarity. Through iterative deliberation and consensus voting, MADRA significantly reduces false rejections while maintaining high sensitivity to dangerous tasks. Additionally, we introduce a hierarchical cognitive collaborative planning framework that integrates safety, memory, planning, and self-evolution mechanisms to improve task success rates through continuous learning. We also contribute SafeAware-VH, a benchmark dataset for safety-aware task planning in VirtualHome, containing 800 annotated instructions. Extensive experiments on AI2-THOR and VirtualHome demonstrate that our approach achieves over 90% rejection of unsafe tasks while ensuring that safe-task rejection is low, outperforming existing methods in both safety and execution efficiency. Our work provides a scalable, model-agnostic solution for building trustworthy embodied agents.

**arXiv ID:** 2511.21460
</details>

<details>
<summary><strong>MTTR-A: Measuring Cognitive Recovery Latency in Multi-Agent Systems</strong> - Barak Or - [[pdf]](https://arxiv.org/pdf/2511.20663)</summary>

**Abstract:** Ensuring cognitive stability in autonomous multi-agent systems (MAS) is a central challenge for large-scale, distributed AI. While existing observability tools monitor system outputs, they cannot quantify how rapidly agentic workflows recover once reasoning coherence has been lost. We adapt classical reliability metrics-Mean Time-to-Recovery (MTTR), Mean Time Between Failures (MTBF), and related ratios-into the cognitive domain, defining MTTR-A (Mean Time-to-Recovery for Agentic Systems) as a runtime measure of cognitive recovery latency. MTTR-A quantifies the time required for a MAS to detect reasoning drift and restore consistent operation, capturing the recovery of reasoning coherence rather than infrastructural repair.
A benchmark simulation using the AG~News corpus and the LangGraph orchestration framework was conducted, modeling recovery latencies across multiple reflex modes. Automated reflexes restored stability within approximately 6s on average, while human-approval interventions required about 12s. Across 200 runs, the median simulated MTTR-A was 6.21+-2.14s, MTBF=6.7+-2.14s, and NRR=0.08, demonstrating measurable runtime resilience across reflex strategies.
By formalizing recovery latency as a quantifiable property of distributed reasoning-and deriving reliability bounds linking recovery time and cognitive uptime-this work establishes a foundation for runtime dependability in agentic cognition, transforming cognitive recovery from an ad-hoc process into a standardized, interpretable performance

**arXiv ID:** 2511.20663
</details>

<details>
<summary><strong>Tool-RoCo: An Agent-as-Tool Self-organization Large Language Model Benchmark in Multi-robot Cooperation</strong> - Ke Zhang, Xiaoning Zhao, Ce Zheng, Jiahong Ning, Dandan Zhu, Wenqi Zhang, Chen Sun, Toshiharu Sugawara - [[pdf]](https://arxiv.org/pdf/2511.21510)</summary>

**Abstract:** This study proposes Tool-RoCo, a novel benchmark for evaluating large language models (LLMs) in long-term multi-agent cooperation based on RoCo, a multi-robot cooperative benchmark. Recent research on LLM-based multi-agent systems has relied on predefined orchestration, while ignoring agent autonomy. Tool-RoCo treats other agents as tools and introduces cooperative tools, leveraging tool usage to evaluate multi-agent cooperation and self-organization. Tool usage means that each agent (LLM) selects a tool from a candidate set based on the current state, receives feedback, and adjusts its selection in subsequent rounds. To evaluate different autonomy levels, we propose four LLM paradigms: (1) centralized cooperation, where a single LLM allocates tools to all agents; (2) centralized self-organization, where a central LLM autonomously activates agents while keeping others inactive; (3) decentralized cooperation, where each agent has its own LLM and calls tools based on local information; and (4) self-organization, where a randomly chosen initial agent can request collaboration, activating additional agents via tool calls. Tool-RoCo includes three multi-robot tasks, SORT, PACK, and CABINET, to measure format and parameter accuracy and agent coordination through tool usage. The results using several LLMs showed that cooperative tools accounted for only 7.09% of all tools, indicating that LLM-based agents rarely invoked others as assistants. Moreover, activation tools accounted for 96.42%, suggesting that current LLMs tend to maintain active agents while seldom deactivating them for adaptive coordination. Tool-RoCo provides a systematic benchmark to evaluate LLM autonomy and cooperation in multi-agent tasks. Code and Demo: this https URL

**arXiv ID:** 2511.21510
</details>

<details>
<summary><strong>BAMAS: Structuring Budget-Aware Multi-Agent Systems</strong> - Liming Yang, Junyu Luo, Xuanzhe Liu, Yiling Lou, Zhenpeng Chen - [[pdf]](https://arxiv.org/pdf/2511.21572)</summary>

**Abstract:** Large language model (LLM)-based multi-agent systems have emerged as a powerful paradigm for enabling autonomous agents to solve complex tasks. As these systems scale in complexity, cost becomes an important consideration for practical deployment. However, existing work rarely addresses how to structure multi-agent systems under explicit budget constraints. In this paper, we propose BAMAS, a novel approach for building multi-agent systems with budget awareness. BAMAS first selects an optimal set of LLMs by formulating and solving an Integer Linear Programming problem that balances performance and cost. It then determines how these LLMs should collaborate by leveraging a reinforcement learning-based method to select the interaction topology. Finally, the system is instantiated and executed based on the selected agents and their collaboration topology. We evaluate BAMAS on three representative tasks and compare it with state-of-the-art agent construction methods. Results show that BAMAS achieves comparable performance while reducing cost by up to 86%.

**arXiv ID:** 2511.21572
</details>

<details>
<summary><strong>Matrix: Peer-to-Peer Multi-Agent Synthetic Data Generation Framework</strong> - Dong Wang, Yang Li, Ansong Ni, Ching-Feng Yeh, Youssef Emad, Xinjie Lei, Liam Robbins, Karthik Padthe, Hu Xu, Xian Li, Asli Celikyilmaz, Ramya Raghavendra, Lifei Huang, Carole-Jean Wu, Shang-Wen Li - [[pdf]](https://arxiv.org/pdf/2511.21686)</summary>

**Abstract:** Synthetic data has become increasingly important for training large language models, especially when real data is scarce, expensive, or privacy-sensitive. Many such generation tasks require coordinated multi-agent workflows, where specialized agents collaborate to produce data that is higher quality, more diverse, and structurally richer. However, existing frameworks for multi-agent synthesis often depend on a centralized orchestrator, creating scalability bottlenecks, or are hardcoded for specific domains, limiting flexibility. We present \textbf{Matrix}, a decentralized framework that represents both control and data flow as serialized messages passed through distributed queues. This peer-to-peer design eliminates the central orchestrator. Each task progresses independently through lightweight agents, while compute-intensive operations, such as LLM inference or containerized environments, are handled by distributed services. Built on Ray, Matrix scales to tens of thousands of concurrent agentic workflows and provides a modular, configurable design that enables easy adaptation to a wide range of data generation workflows. We evaluate Matrix across diverse synthesis scenarios, such as multi-agent collaborative dialogue, web-based reasoning data extraction, and tool-use trajectory generation in customer service environments. In all cases, Matrix achieves $2$--$15\times$ higher data generation throughput under identical hardware resources, without compromising output quality.

**arXiv ID:** 2511.21686
</details>

<details>
<summary><strong>CoMind: Towards Community-Driven Agents for Machine Learning Engineering</strong> - Sijie Li, Weiwei Sun, Shanda Li, Ameet Talwalkar, Yiming Yang - [[pdf]](https://arxiv.org/pdf/2506.20640)</summary>

**Abstract:** Large language model (LLM) agents show promise in automating machine learning (ML) engineering. However, existing agents typically operate in isolation on a given research problem, without engaging with the broader research community, where human researchers often gain insights and contribute by sharing knowledge. To bridge this gap, we introduce MLE-Live, a live evaluation framework designed to assess an agent's ability to communicate with and leverage collective knowledge from a simulated Kaggle research community. Building on this framework, we propose CoMind, an multi-agent system designed to actively integrate external knowledge. CoMind employs an iterative parallel exploration mechanism, developing multiple solutions simultaneously to balance exploratory breadth with implementation depth. On 75 past Kaggle competitions within our MLE-Live framework, CoMind achieves a 36% medal rate, establishing a new state of the art. Critically, when deployed in eight live, ongoing competitions, CoMind outperforms 92.6% of human competitors on average, placing in the top 5% on three official leaderboards and the top 1% on one.

**arXiv ID:** 2506.20640
</details>

<details>
<summary><strong>Heterogeneous Multi-Agent Proximal Policy Optimization for Power Distribution System Restoration</strong> - Parya Dolatyabi, Ali Farajzadeh Bavil, Mahdi Khodayar - [[pdf]](https://arxiv.org/pdf/2511.14730)</summary>

**Abstract:** Restoring power distribution systems (PDS) after large-scale outages requires sequential switching operations that reconfigure feeder topology and coordinate distributed energy resources (DERs) under nonlinear constraints such as power balance, voltage limits, and thermal ratings. These challenges make conventional optimization and value-based RL approaches computationally inefficient and difficult to scale. This paper applies a Heterogeneous-Agent Reinforcement Learning (HARL) framework, instantiated through Heterogeneous-Agent Proximal Policy Optimization (HAPPO), to enable coordinated restoration across interconnected microgrids. Each agent controls a distinct microgrid with different loads, DER capacities, and switch counts, introducing practical structural heterogeneity. Decentralized actor policies are trained with a centralized critic to compute advantage values for stable on-policy updates. A physics-informed OpenDSS environment provides full power flow feedback and enforces operational limits via differentiable penalty signals rather than invalid action masking. The total DER generation is capped at 2400 kW, and each microgrid must satisfy local supply-demand feasibility. Experiments on the IEEE 123-bus and IEEE 8500-node systems show that HAPPO achieves faster convergence, higher restored power, and smoother multi-seed training than DQN, PPO, MAES, MAGDPG, MADQN, Mean-Field RL, and QMIX. Results demonstrate that incorporating microgrid-level heterogeneity within the HARL framework yields a scalable, stable, and constraint-aware solution for complex PDS restoration.

**arXiv ID:** 2511.14730
</details>

<details>
<summary><strong>Natural Strategic Ability in Stochastic Multi-Agent Systems</strong> - Raphaël Berthon, Joost-Pieter Katoen, Munyque Mittelmann, Aniello Murano - [[pdf]](https://arxiv.org/pdf/2401.12170)</summary>

**Abstract:** Strategies synthesized using formal methods can be complex and often require infinite memory, which does not correspond to the expected behavior when trying to model Multi-Agent Systems (MAS). To capture such behaviors, natural strategies are a recently proposed framework striking a balance between the ability of agents to strategize with memory and the model-checking complexity, but until now has been restricted to fully deterministic settings. For the first time, we consider the probabilistic temporal logics PATL and PATL* under natural strategies (NatPATL and NatPATL*, resp.). As main result we show that, in stochastic MAS, NatPATL model-checking is NP-complete when the active coalition is restricted to deterministic strategies. We also give a 2NEXPTIME complexity result for NatPATL* with the same restriction. In the unrestricted case, we give an EXPSPACE complexity for NatPATL and 3EXPSPACE complexity for NatPATL*.

**arXiv ID:** 2401.12170
</details>

<details>
<summary><strong>Fair Algorithms with Probing for Multi-Agent Multi-Armed Bandits</strong> - Tianyi Xu, Jiaxin Liu, Nicholas Mattei, Zizhan Zheng - [[pdf]](https://arxiv.org/pdf/2506.14988)</summary>

**Abstract:** We propose a multi-agent multi-armed bandit (MA-MAB) framework aimed at ensuring fair outcomes across agents while maximizing overall system performance. A key challenge in this setting is decision-making under limited information about arm rewards. To address this, we introduce a novel probing framework that strategically gathers information about selected arms before allocation. In the offline setting, where reward distributions are known, we leverage submodular properties to design a greedy probing algorithm with a provable performance bound. For the more complex online setting, we develop an algorithm that achieves sublinear regret while maintaining fairness. Extensive experiments on synthetic and real-world datasets show that our approach outperforms baseline methods, achieving better fairness and efficiency.

**arXiv ID:** 2506.14988
</details>

<details>
<summary><strong>Policy Optimization and Multi-agent Reinforcement Learning for Mean-variance Team Stochastic Games</strong> - Junkai Hu, Li Xia - [[pdf]](https://arxiv.org/pdf/2503.22779)</summary>

**Abstract:** We study a long-run mean-variance team stochastic game (MV-TSG), where each agent shares a common mean-variance objective for the system and takes actions independently to maximize it. MV-TSG has two main challenges. First, the variance metric is neither additive nor Markovian in a dynamic setting. Second, simultaneous policy updates of all agents lead to a non-stationary environment for each individual agent. Both challenges make dynamic programming inapplicable. In this paper, we study MV-TSGs from the perspective of sensitivity-based optimization. The performance difference and performance derivative formulas for joint policies are derived, which provide optimization information for MV-TSGs. We prove the existence of a deterministic Nash policy for this problem. Subsequently, we propose a Mean-Variance Multi-Agent Policy Iteration (MV-MAPI) algorithm with a sequential update scheme, where individual agent policies are updated one by one in a given order. We prove that the MV-MAPI algorithm converges to a first-order stationary point of the objective function. By analyzing the local geometry of stationary points, we derive specific conditions for stationary points to be (local) Nash equilibria, and further, strict local optima. To solve large-scale MV-TSGs in scenarios with unknown environmental parameters, we extend the idea of trust region methods to MV-MAPI and develop a multi-agent reinforcement learning algorithm named Mean-Variance Multi-Agent Trust Region Policy Optimization (MV-MATRPO). We derive a performance lower bound for each update of joint policies. Finally, numerical experiments on energy management in multiple microgrid systems are conducted.

**arXiv ID:** 2503.22779
</details>

<details>
<summary><strong>MAPF-HD: Multi-Agent Path Finding in High-Density Environments</strong> - Hiroya Makino, Seigo Ito - [[pdf]](https://arxiv.org/pdf/2509.06374)</summary>

**Abstract:** Multi-agent path finding (MAPF) involves planning efficient paths for multiple agents to move simultaneously while avoiding collisions. In typical warehouse environments, agents are often sparsely distributed along aisles; however, increasing the agent density can improve space efficiency. When the agent density is high, it becomes necessary to optimize the paths not only for goal-assigned agents but also for those obstructing them. This study proposes a novel MAPF framework for high-density environments (MAPF-HD). Several studies have explored MAPF in similar settings using integer linear programming (ILP). However, ILP-based methods require substantial computation time to optimize all agent paths simultaneously. Even in small grid-based environments with fewer than $100$ cells, these computations can take tens to hundreds of seconds. Such high computational costs render these methods impractical for large-scale applications such as automated warehouses and valet parking. To address these limitations, we introduce the phased null-agent swapping (PHANS) method. PHANS employs a heuristic approach to incrementally swap positions between agents and empty vertices. This method solves the MAPF-HD problem within a few seconds, even in large environments containing more than $700$ cells. The proposed method has the potential to improve efficiency in various real-world applications such as warehouse logistics, traffic management, and crowd control. The implementation is available at this https URL.

**arXiv ID:** 2509.06374
</details>

<details>
<summary><strong>Multi-Agent Cross-Entropy Method with Monotonic Nonlinear Critic Decomposition</strong> - Yan Wang, Ke Deng, Yongli Ren - [[pdf]](https://arxiv.org/pdf/2511.18671)</summary>

**Abstract:** Cooperative multi-agent reinforcement learning (MARL) commonly adopts centralized training with decentralized execution (CTDE), where centralized critics leverage global information to guide decentralized actors. However, centralized-decentralized mismatch (CDM) arises when the suboptimal behavior of one agent degrades others' learning. Prior approaches mitigate CDM through value decomposition, but linear decompositions allow per-agent gradients at the cost of limited expressiveness, while nonlinear decompositions improve representation but require centralized gradients, reintroducing CDM. To overcome this trade-off, we propose the multi-agent cross-entropy method (MCEM), combined with monotonic nonlinear critic decomposition (NCD). MCEM updates policies by increasing the probability of high-value joint actions, thereby excluding suboptimal behaviors. For sample efficiency, we extend off-policy learning with a modified k-step return and Retrace. Analysis and experiments demonstrate that MCEM outperforms state-of-the-art methods across both continuous and discrete action benchmarks.

**arXiv ID:** 2511.18671
</details>

<details>
<summary><strong>Chatty-KG: A Multi-Agent AI System for On-Demand Conversational Question Answering over Knowledge Graphs</strong> - Reham Omar, Abdelghny Orogat, Ibrahim Abdelaziz, Omij Mangukiya, Panos Kalnis, Essam Mansour - [[pdf]](https://arxiv.org/pdf/2511.20940)</summary>

**Abstract:** Conversational Question Answering over Knowledge Graphs (KGs) combines the factual grounding of KG-based QA with the interactive nature of dialogue systems. KGs are widely used in enterprise and domain applications to provide structured, evolving, and reliable knowledge. Large language models (LLMs) enable natural and context-aware conversations, but lack direct access to private and dynamic KGs. Retrieval-augmented generation (RAG) systems can retrieve graph content but often serialize structure, struggle with multi-turn context, and require heavy indexing. Traditional KGQA systems preserve structure but typically support only single-turn QA, incur high latency, and struggle with coreference and context tracking. To address these limitations, we propose Chatty-KG, a modular multi-agent system for conversational QA over KGs. Chatty-KG combines RAG-style retrieval with structured execution by generating SPARQL queries through task-specialized LLM agents. These agents collaborate for contextual interpretation, dialogue tracking, entity and relation linking, and efficient query planning, enabling accurate and low-latency translation of natural questions into executable queries. Experiments on large and diverse KGs show that Chatty-KG significantly outperforms state-of-the-art baselines in both single-turn and multi-turn settings, achieving higher F1 and P@1 scores. Its modular design preserves dialogue coherence and supports evolving KGs without fine-tuning or pre-processing. Evaluations with commercial (e.g., GPT-4o, Gemini-2.0) and open-weight (e.g., Phi-4, Gemma 3) LLMs confirm broad compatibility and stable performance. Overall, Chatty-KG unifies conversational flexibility with structured KG grounding, offering a scalable and extensible approach for reliable multi-turn KGQA.

**arXiv ID:** 2511.20940
</details>

<details>
<summary><strong>Multi-Agent Monocular Dense SLAM With 3D Reconstruction Priors</strong> - Yuchen Zhou, Haihang Wu - [[pdf]](https://arxiv.org/pdf/2511.19031)</summary>

**Abstract:** Monocular Simultaneous Localization and Mapping (SLAM) aims to estimate a robot's pose while simultaneously reconstructing an unknown 3D scene using a single camera. While existing monocular SLAM systems generate detailed 3D geometry through dense scene representations, they are computationally expensive due to the need for iterative optimization. To address this challenge, MASt3R-SLAM utilizes learned 3D reconstruction priors, enabling more efficient and accurate estimation of both 3D structures and camera poses. However, MASt3R-SLAM is limited to single-agent operation. In this paper, we extend MASt3R-SLAM to introduce the first multi-agent monocular dense SLAM system. Each agent performs local SLAM using a 3D reconstruction prior, and their individual maps are fused into a globally consistent map through a loop-closure-based map fusion mechanism. Our approach improves computational efficiency compared to state-of-the-art methods, while maintaining similar mapping accuracy when evaluated on real-world datasets.

**arXiv ID:** 2511.19031
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
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>Prune4Web: DOM Tree Pruning Programming for Web Agent</strong> - Jiayuan Zhang, Kaiquan Chen, Zhihao Lu, Enshen Zhou, Qian Yu, Jing Zhang - [[pdf]](https://arxiv.org/pdf/2511.21398)</summary>

**Abstract:** Web automation employs intelligent agents to execute high-level tasks by mimicking human interactions with web interfaces. Despite the capabilities of recent Large Language Model (LLM)-based web agents, navigating complex, real-world webpages efficiently remains a significant hurdle due to the prohibitively large size of Document Object Model (DOM) structures, often ranging from 10,000 to 100,000 tokens. Existing strategies typically rely on crude DOM truncation -- risking the loss of critical information -- or employ inefficient heuristics and separate ranking models, failing to achieve an optimal balance between precision and scalability. To address these challenges, we introduce Prune4Web, a novel paradigm that shifts DOM processing from resource-intensive LLM reading to efficient programmatic pruning. Central to our approach is DOM Tree Pruning Programming, where an LLM generates executable Python scoring scripts to dynamically filter DOM elements based on semantic cues from decomposed sub-tasks. This mechanism eliminates the need for LLMs to ingest raw, massive DOMs, instead delegating traversal and scoring to lightweight, interpretable programs. This methodology achieves a 25x to 50x reduction in candidate elements for grounding, thereby facilitating precise action localization while mitigating attention dilution. Furthermore, we propose a specialized data annotation pipeline and a two-turn dialogue training strategy that jointly optimizes the Planner, Programmatic Filter, and Grounder within a unified framework. Extensive experiments demonstrate state-of-the-art performance. Notably, on our low-level grounding task, Prune4Web dramatically improves accuracy from 46.8% to 88.28%, underscoring its efficacy in real-world web automation.

**arXiv ID:** 2511.21398
</details>

<details>
<summary><strong>Intelligent Agents with Emotional Intelligence: Current Trends, Challenges, and Future Prospects</strong> - Raziyeh Zall, Alireza Kheyrkhah, Erik Cambria, Zahra Naseri, M.Reza Kangavari - [[pdf]](https://arxiv.org/pdf/2511.20657)</summary>

**Abstract:** The development of agents with emotional intelligence is becoming increasingly vital due to their significant role in human-computer interaction and the growing integration of computer systems across various sectors of society. Affective computing aims to design intelligent systems that can recognize, evoke, and express human emotions, thereby emulating human emotional intelligence. While previous reviews have focused on specific aspects of this field, there has been limited comprehensive research that encompasses emotion understanding, elicitation, and expression, along with the related challenges. This survey addresses this gap by providing a holistic overview of core components of artificial emotion intelligence. It covers emotion understanding through multimodal data processing, as well as affective cognition, which includes cognitive appraisal, emotion mapping, and adaptive modulation in decision-making, learning, and reasoning. Additionally, it addresses the synthesis of emotional expression across text, speech, and facial modalities to enhance human-agent interaction. This paper identifies and analyzes the key challenges and issues encountered in the development of affective systems, covering state-of-the-art methodologies designed to address them. Finally, we highlight promising future directions, with particular emphasis on the potential of generative technologies to advance affective computing.

**arXiv ID:** 2511.20657
</details>

<details>
<summary><strong>NOIR 2.0: Neural Signal Operated Intelligent Robots for Everyday Activities</strong> - Tasha Kim, Yingke Wang, Hanvit Cho, Alex Hodges - [[pdf]](https://arxiv.org/pdf/2511.20848)</summary>

**Abstract:** Neural Signal Operated Intelligent Robots (NOIR) system is a versatile brain-robot interface that allows humans to control robots for daily tasks using their brain signals. This interface utilizes electroencephalography (EEG) to translate human intentions regarding specific objects and desired actions directly into commands that robots can execute. We present NOIR 2.0, an enhanced version of NOIR. NOIR 2.0 includes faster and more accurate brain decoding algorithms, which reduce task completion time by 46%. NOIR 2.0 uses few-shot robot learning algorithms to adapt to individual users and predict their intentions. The new learning algorithms leverage foundation models for more sample-efficient learning and adaptation (15 demos vs. a single demo), significantly reducing overall human time by 65%.

**arXiv ID:** 2511.20848
</details>

<details>
<summary><strong>Learning Individual Behavior in Agent-Based Models with Graph Diffusion Networks</strong> - Francesco Cozzi, Marco Pangallo, Alan Perotti, André Panisson, Corrado Monti - [[pdf]](https://arxiv.org/pdf/2505.21426)</summary>

**Abstract:** Agent-Based Models (ABMs) are powerful tools for studying emergent properties in complex systems. In ABMs, agent behaviors are governed by local interactions and stochastic rules. However, these rules are, in general, non-differentiable, limiting the use of gradient-based methods for optimization, and thus integration with real-world data. We propose a novel framework to learn a differentiable surrogate of any ABM by observing its generated data. Our method combines diffusion models to capture behavioral stochasticity and graph neural networks to model agent interactions. Distinct from prior surrogate approaches, our method introduces a fundamental shift: rather than approximating system-level outputs, it models individual agent behavior directly, preserving the decentralized, bottom-up dynamics that define ABMs. We validate our approach on two ABMs (Schelling's segregation model and a Predator-Prey ecosystem) showing that it replicates individual-level patterns and accurately forecasts emergent dynamics beyond training. Our results demonstrate the potential of combining diffusion models and graph learning for data-driven ABM simulation.

**arXiv ID:** 2505.21426
</details>

<details>
<summary><strong>Safety Control of Service Robots with LLMs and Embodied Knowledge Graphs</strong> - Yong Qi, Gabriel Kyebambo, Siyuan Xie, Wei Shen, Shenghui Wang, Bitao Xie, Bin He, Zhipeng Wang, Shuo Jiang - [[pdf]](https://arxiv.org/pdf/2405.17846)</summary>

**Abstract:** Safety limitations in service robotics across various industries have raised significant concerns about the need for robust mechanisms ensuring that robots adhere to safe practices, thereby preventing actions that might harm humans or cause property damage. Despite advances, including the integration of Knowledge Graphs (KGs) with Large Language Models (LLMs), challenges in ensuring consistent safety in autonomous robot actions persist. In this paper, we propose a novel integration of Large Language Models with Embodied Robotic Control Prompts (ERCPs) and Embodied Knowledge Graphs (EKGs) to enhance the safety framework for service robots. ERCPs are designed as predefined instructions that ensure LLMs generate safe and precise responses. These responses are subsequently validated by EKGs, which provide a comprehensive knowledge base ensuring that the actions of the robot are continuously aligned with safety protocols, thereby promoting safer operational practices in varied contexts. Our experimental setup involved diverse real-world tasks, where robots equipped with our framework demonstrated significantly higher compliance with safety standards compared to traditional methods. This integration fosters secure human-robot interactions and positions our methodology at the forefront of AI-driven safety innovations in service robotics.

**arXiv ID:** 2405.17846
</details>

<details>
<summary><strong>Navigating Quantum Missteps in Agent-Based Modeling: A Schelling Model Case Study</strong> - C. Nico Barati, Arie Croitoru, Ross Gore, Michael Jarret, William Kennedy, Andrew Maciejunes, Maxim A. Malikov, Samuel S. Mendelson - [[pdf]](https://arxiv.org/pdf/2511.15642)</summary>

**Abstract:** Quantum computing promises transformative advances, but remains constrained by recurring misconceptions and methodological pitfalls. This paper demonstrates a fundamental incompatibility between traditional agent-based modeling (ABM) implementations and quantum optimization frameworks like Quadratic Unconstrained Binary Optimization (QUBO). Using Schelling's segregation model as a case study, we show that the standard practice of directly translating ABM state observations into QUBO formulations not only fails to deliver quantum advantage, but actively undermines computational efficiency. The fundamental issue is architectural. Traditional ABM implementations entail observing the state of the system at each iteration, systematically destroying the quantum superposition required for computational advantage. Through analysis of Schelling's segregation dynamics on lollipop networks, we demonstrate how abandoning the QUBO reduction paradigm and instead reconceptualizing the research question, from "simulate agent dynamics iteratively until convergence" to "compute minimum of agent moves required for global satisfaction", enables a faster classical solution. This structural reconceptualization yields an algorithm that exploits network symmetries obscured in traditional ABM simulations and QUBO formulations. It establishes a new lower bound which quantum approaches must outperform to achieve advantage. Our work emphasizes that progress in quantum agent-based modeling does not require forcing classical ABM implementations into quantum frameworks. Instead, it should focus on clarifying when quantum advantage is structurally possible, developing best-in-class classical baselines through problem analysis, and fundamentally reformulating research questions rather than preserving classical iterative state change observation paradigms.

**arXiv ID:** 2511.15642
</details>

<details>
<summary><strong>LLMs-Powered Accurate Extraction, Querying and Intelligent Management of Literature derived 2D Materials Data</strong> - Lijun Shang, Yadong Yu, Wenqiang Kang, Jian Zhou, Dongyue Gao, Pan Xiang, Zhe Liu, Mengyan Dai, Zhonglu Guo, Zhimei Sun - [[pdf]](https://arxiv.org/pdf/2511.20691)</summary>

**Abstract:** Two-dimensional (2D) materials have showed widespread applications in energy storage and conversion owning to their unique physicochemical, and electronic properties. Most of the valuable information for the materials, such as their properties and preparation methods, is included in the published research papers. However, due to the dispersion of synthe

**arXiv ID:** 2511.20691
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (24 papers)</h2></summary>

<details>
<summary><strong>ENACT: Evaluating Embodied Cognition with World Modeling of Egocentric Interaction</strong> - Qineng Wang, Wenlong Huang, Yu Zhou, Hang Yin, Tianwei Bao, Jianwen Lyu, Weiyu Liu, Ruohan Zhang, Jiajun Wu, Li Fei-Fei, Manling Li - [[pdf]](https://arxiv.org/pdf/2511.20937)</summary>

**Abstract:** Embodied cognition argues that intelligence arises from sensorimotor interaction rather than passive observation. It raises an intriguing question: do modern vision-language models (VLMs), trained largely in a disembodied manner, exhibit signs of embodied cognition? We introduce ENACT, a benchmark that casts evaluation of embodied cognition as world modeling from egocentric interaction in a visual question answering (VQA) format. Framed as a partially observable Markov decision process (POMDP) whose actions are scene graph changes, ENACT comprises two complementary sequence reordering tasks: forward world modeling (reorder shuffled observations given actions) and inverse world modeling (reorder shuffled actions given observations). While conceptually simple, solving these tasks implicitly demands capabilities central to embodied cognition-affordance recognition, action-effect reasoning, embodied awareness, and interactive, long-horizon memory from partially observable egocentric input, while avoiding low-level image synthesis that could confound the evaluation. We provide a scalable pipeline that synthesizes QA pairs from robotics simulation (BEHAVIOR) and evaluates models on 8,972 QA pairs spanning long-horizon home-scale activities. Experiments reveal a performance gap between frontier VLMs and humans that widens with interaction horizon. Models consistently perform better on the inverse task than the forward one and exhibit anthropocentric biases, including a preference for right-handed actions and degradation when camera intrinsics or viewpoints deviate from human vision. Website at this https URL.

**arXiv ID:** 2511.20937
</details>

<details>
<summary><strong>ICPO: Intrinsic Confidence-Driven Group Relative Preference Optimization for Efficient Reinforcement Learning</strong> - Jinpeng Wang, Chao Li, Ting Ye, Mengyuan Zhang, Wei Liu, Jian Luan - [[pdf]](https://arxiv.org/pdf/2511.21005)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) demonstrates significant potential in enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing RLVR methods are often constrained by issues such as coarse-grained rewards, reward noise, and inefficient exploration, which lead to unstable training and entropy collapse. To address this challenge, we propose the Intrinsic Confidence-Driven Group Relative Preference Optimization method (ICPO). The intuition behind it lies in the fact that the probabilities of an LLM generating different responses can inherently and directly reflect its self-assessment of the reasoning process. Inspired by the idea of preference modeling, ICPO calculates a preference advantage score for each response by comparing the relative generation probabilities of multiple responses under the same input prompt, and integrates this score with verifiable rewards to guide the exploration process. We have discovered that the preference advantage score not only alleviates the issues of coarse-grained rewards and reward noise but also effectively curbs overconfident errors, enhances the relative superiority of undervalued high-quality responses, and prevents the model from overfitting to specific strategies, thereby facilitating more thorough exploration. Comprehensive experiments across four general-domain benchmarks and three mathematical benchmarks demonstrate that ICPO steadily boosts reasoning compared to GRPO.

**arXiv ID:** 2511.21005
</details>

<details>
<summary><strong>OVOD-Agent: A Markov-Bandit Framework for Proactive Visual Reasoning and Self-Evolving Detection</strong> - Chujie Wang, Jianyu Lu, Zhiyuan Luo, Xi Chen, Chu He - [[pdf]](https://arxiv.org/pdf/2511.21064)</summary>

**Abstract:** Open-Vocabulary Object Detection (OVOD) aims to enable detectors to generalize across categories by leveraging semantic information. Although existing methods are pretrained on large vision-language datasets, their inference is still limited to fixed category names, creating a gap between multimodal training and unimodal inference. Previous work has shown that improving textual representation can significantly enhance OVOD performance, indicating that the textual space is still underexplored. To this end, we propose OVOD-Agent, which transforms passive category matching into proactive visual reasoning and self-evolving detection. Inspired by the Chain-of-Thought (CoT) paradigm, OVOD-Agent extends the textual optimization process into an interpretable Visual-CoT with explicit actions. OVOD's lightweight nature makes LLM-based management unsuitable; instead, we model visual context transitions as a Weakly Markovian Decision Process (w-MDP) over eight state spaces, which naturally represents the agent's state, memory, and interaction dynamics. A Bandit module generates exploration signals under limited supervision, helping the agent focus on uncertain regions and adapt its detection policy. We further integrate Markov transition matrices with Bandit trajectories for self-supervised Reward Model (RM) optimization, forming a closed loop from Bandit exploration to RM learning. Experiments on COCO and LVIS show that OVOD-Agent provides consistent improvements across OVOD backbones, particularly on rare categories, confirming the effectiveness of the proposed framework.

**arXiv ID:** 2511.21064
</details>

<details>
<summary><strong>Context-Aware Visual Prompting: Automating Geospatial Web Dashboards with Large Language Models and Agent Self-Validation for Decision Support</strong> - Haowen Xu, Jose Tupayachi, Xiao-Ying Yu - [[pdf]](https://arxiv.org/pdf/2511.20656)</summary>

**Abstract:** The development of web-based geospatial dashboards for risk analysis and decision support is often challenged by the difficulty in visualization of big, multi-dimensional environmental data, implementation complexity, and limited automation. We introduce a generative AI framework that harnesses Large Language Models (LLMs) to automate the creation of interactive geospatial dashboards from user-defined inputs including UI wireframes, requirements, and data sources. By incorporating a structured knowledge graph, the workflow embeds domain knowledge into the generation process and enable accurate and context-aware code completions. A key component of our approach is the Context-Aware Visual Prompting (CAVP) mechanism, which extracts encodes and interface semantics from visual layouts to guide LLM driven generation of codes. The new framework also integrates a self-validation mechanism that uses an agent-based LLM and Pass@k evaluation alongside semantic metrics to assure output reliability. Dashboard snippets are paired with data visualization codebases and ontological representations, enabling a pipeline that produces scalable React-based completions using the MVVM architectural pattern. Our results demonstrate improved performance over baseline approaches and expanded functionality over third party platforms, while incorporating multi-page, fully functional interfaces. We successfully developed a framework to implement LLMs, demonstrated the pipeline for automated code generation, deployment, and performed chain-of-thought AI agents in self-validation. This integrative approach is guided by structured knowledge and visual prompts, providing an innovative geospatial solution in enhancing risk analysis and decision making.

**arXiv ID:** 2511.20656
</details>

<details>
<summary><strong>Exploring Time-Step Size in Reinforcement Learning for Sepsis Treatment</strong> - Yingchuan Sun, Shengpu Tang - [[pdf]](https://arxiv.org/pdf/2511.20913)</summary>

**Abstract:** Existing studies on reinforcement learning (RL) for sepsis management have mostly followed an established problem setup, in which patient data are aggregated into 4-hour time steps. Although concerns have been raised regarding the coarseness of this time-step size, which might distort patient dynamics and lead to suboptimal treatment policies, the extent to which this is a problem in practice remains unexplored. In this work, we conducted empirical experiments for a controlled comparison of four time-step sizes ($\Delta t\!=\!1,2,4,8$ h) on this domain, following an identical offline RL pipeline. To enable a fair comparison across time-step sizes, we designed action re-mapping methods that allow for evaluation of policies on datasets with different time-step sizes, and conducted cross-$\Delta t$ model selections under two policy learning setups. Our goal was to quantify how time-step size influences state representation learning, behavior cloning, policy training, and off-policy evaluation. Our results show that performance trends across $\Delta t$ vary as learning setups change, while policies learned at finer time-step sizes ($\Delta t = 1$ h and $2$ h) using a static behavior policy achieve the overall best performance and stability. Our work highlights time-step size as a core design choice in offline RL for healthcare and provides evidence supporting alternatives beyond the conventional 4-hour setup.

**arXiv ID:** 2511.20913
</details>

<details>
<summary><strong>Subgoal Graph-Augmented Planning for LLM-Guided Open-World Reinforcement Learning</strong> - Shanwei Fan - [[pdf]](https://arxiv.org/pdf/2511.20993)</summary>

**Abstract:** Large language models (LLMs) offer strong high-level planning capabilities for reinforcement learning (RL) by decomposing tasks into subgoals. However, their practical utility is limited by poor planning-execution alignment, which reflects a critical gap between abstract plans and actionable, environment-compatible behaviors. This misalignment arises from two interrelated limitations: (1) LLMs often produce subgoals that are semantically plausible but infeasible or irrelevant in the target environment due to insufficient grounding in environment-specific knowledge, and (2) single-LLM planning conflates generation with self-verification, resulting in overconfident yet unreliable subgoals that frequently fail during execution. To address these challenges, we propose Subgoal Graph-Augmented Actor-Critic-Refiner (SGA-ACR), a framework that integrates an environment-specific subgoal graph and structured entity knowledge with a multi-LLM planning pipeline that explicitly separates generation, critique, and refinement to produce executable and verifiable subgoals. A subgoal tracker further monitors execution progress, provides auxiliary rewards, and adaptively updates the subgoal graph to maintain alignment between plans and actions. Experimental results on 22 diverse tasks in the open-world game "Crafter" demonstrate the effectiveness of our proposed method.

**arXiv ID:** 2511.20993
</details>

<details>
<summary><strong>Breaking the Safety-Capability Tradeoff: Reinforcement Learning with Verifiable Rewards Maintains Safety Guardrails in LLMs</strong> - Dongkyu Derek Cho, Huan Song, Arijit Ghosh Chowdhury, Haotian An, Yawei Wang, Rohit Thekkanal, Negin Sokhandan, Sharlina Keshava, Hannah Marlowe - [[pdf]](https://arxiv.org/pdf/2511.21050)</summary>

**Abstract:** Fine-tuning large language models (LLMs) for downstream tasks typically exhibit a fundamental safety-capability tradeoff, where improving task performance degrades safety alignment even on benign datasets. This degradation persists across standard approaches including supervised finetuning (SFT) and reinforcement learning from human feedback (RLHF). While reinforcement learning with verifiable rewards (RLVR) has emerged as a promising alternative that optimizes models on objectively measurable tasks, its safety implications remain unexplored. We present the first comprehensive theoretical and empirical analysis of safety properties in RLVR. Theoretically, we derive upper bounds on safety drift under KL-constrained optimization and prove conditions under which safety degradation is eliminated. Empirically, we conduct extensive experiments across five adversarial safety benchmarks, demonstrating that RLVR can simultaneously enhance reasoning capabilities while maintaining or improving safety guardrails. Our comprehensive ablation studies examine the effects of optimization algorithms, model scale, and task domains. Our findings challenge the prevailing assumption of an inevitable safety capability trade-off, and establish that a specific training methodology can achieve both objectives simultaneously, providing insights for the safe deployment of reasoning-capable LLMs.

**arXiv ID:** 2511.21050
</details>

<details>
<summary><strong>SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation</strong> - Ziyi Chen, Yingnan Guo, Zedong Chu, Minghua Luo, Yanfen Shen, Mingchao Sun, Junjun Hu, Shichao Xie, Kuan Yang, Pei Shi, Zhining Gu, Lu Liu, Honglin Han, Xiaolong Wu, Mu Xu, Yu Zhang - [[pdf]](https://arxiv.org/pdf/2511.21135)</summary>

**Abstract:** Embodied navigation that adheres to social norms remains an open research challenge. Our \textbf{SocialNav} is a foundational model for socially-aware navigation with a hierarchical "brain-action" architecture, capable of understanding high-level social norms and generating low-level, socially compliant trajectories. To enable such dual capabilities, we construct the SocNav Dataset, a large-scale collection of 7 million samples, comprising (1) a Cognitive Activation Dataset providing social reasoning signals such as chain-of-thought explanations and social traversability prediction, and (2) an Expert Trajectories Pyramid aggregating diverse navigation demonstrations from internet videos, simulated environments, and real-world robots. A multi-stage training pipeline is proposed to gradually inject and refine navigation intelligence: we first inject general navigation skills and social norms understanding into the model via imitation learning, and then refine such skills through a deliberately designed Socially-Aware Flow Exploration GRPO (SAFE-GRPO), the first flow-based reinforcement learning framework for embodied navigation that explicitly rewards socially compliant behaviors. SocialNav achieves +38% success rate and +46% social compliance rate compared to the state-of-the-art method, demonstrating strong gains in both navigation performance and social compliance. Our project page: this https URL

**arXiv ID:** 2511.21135
</details>

<details>
<summary><strong>Maglev-Pentabot: Magnetic Levitation System for Non-Contact Manipulation using Deep Reinforcement Learning</strong> - Guoming Huang, Qingyi Zhou, Dianjing Liu, Shuai Zhang, Ming Zhou, Zongfu Yu - [[pdf]](https://arxiv.org/pdf/2511.21149)</summary>

**Abstract:** Non-contact manipulation has emerged as a transformative approach across various industrial fields. However, current flexible 2D and 3D non-contact manipulation techniques are often limited to microscopic scales, typically controlling objects in the milligram range. In this paper, we present a magnetic levitation system, termed Maglev-Pentabot, designed to address this limitation. The Maglev-Pentabot leverages deep reinforcement learning (DRL) to develop complex control strategies for manipulating objects in the gram range. Specifically, we propose an electromagnet arrangement optimized through numerical analysis to maximize controllable space. Additionally, an action remapping method is introduced to address sample sparsity issues caused by the strong nonlinearity in magnetic field intensity, hence allowing the DRL controller to converge. Experimental results demonstrate flexible manipulation capabilities, and notably, our system can generalize to transport tasks it has not been explicitly trained for. Furthermore, our approach can be scaled to manipulate heavier objects using larger electromagnets, offering a reference framework for industrial-scale robotic applications.

**arXiv ID:** 2511.21149
</details>

<details>
<summary><strong>Hybrid-AIRL: Enhancing Inverse Reinforcement Learning with Supervised Expert Guidance</strong> - Bram Silue, Santiago Amaya-Corredor, Patrick Mannion, Lander Willem, Pieter Libin - [[pdf]](https://arxiv.org/pdf/2511.21356)</summary>

**Abstract:** Adversarial Inverse Reinforcement Learning (AIRL) has shown promise in addressing the sparse reward problem in reinforcement learning (RL) by inferring dense reward functions from expert demonstrations. However, its performance in highly complex, imperfect-information settings remains largely unexplored. To explore this gap, we evaluate AIRL in the context of Heads-Up Limit Hold'em (HULHE) poker, a domain characterized by sparse, delayed rewards and significant uncertainty. In this setting, we find that AIRL struggles to infer a sufficiently informative reward function. To overcome this limitation, we contribute Hybrid-AIRL (H-AIRL), an extension that enhances reward inference and policy learning by incorporating a supervised loss derived from expert data and a stochastic regularization mechanism. We evaluate H-AIRL on a carefully selected set of Gymnasium benchmarks and the HULHE poker setting. Additionally, we analyze the learned reward function through visualization to gain deeper insights into the learning process. Our experimental results show that H-AIRL achieves higher sample efficiency and more stable learning compared to AIRL. This highlights the benefits of incorporating supervised signals into inverse RL and establishes H-AIRL as a promising framework for tackling challenging, real-world settings.

**arXiv ID:** 2511.21356
</details>

<details>
<summary><strong>Predictive Safety Shield for Dyna-Q Reinforcement Learning</strong> - Jin Pin, Krasowski Hanna, Vanneaux Elena - [[pdf]](https://arxiv.org/pdf/2511.21531)</summary>

**Abstract:** Obtaining safety guarantees for reinforcement learning is a major challenge to achieve applicability for real-world tasks. Safety shields extend standard reinforcement learning and achieve hard safety guarantees. However, existing safety shields commonly use random sampling of safe actions or a fixed fallback controller, therefore disregarding future performance implications of different safe actions. In this work, we propose a predictive safety shield for model-based reinforcement learning agents in discrete space. Our safety shield updates the Q-function locally based on safe predictions, which originate from a safe simulation of the environment model. This shielding approach improves performance while maintaining hard safety guarantees. Our experiments on gridworld environments demonstrate that even short prediction horizons can be sufficient to identify the optimal path. We observe that our approach is robust to distribution shifts, e.g., between simulation and reality, without requiring additional training.

**arXiv ID:** 2511.21531
</details>

<details>
<summary><strong>SAGE: An Agentic Explainer Framework for Interpreting SAE Features in Language Models</strong> - Jiaojiao Han, Wujiang Xu, Mingyu Jin, Mengnan Du - [[pdf]](https://arxiv.org/pdf/2511.20820)</summary>

**Abstract:** Large language models (LLMs) have achieved remarkable progress, yet their internal mechanisms remain largely opaque, posing a significant challenge to their safe and reliable deployment. Sparse autoencoders (SAEs) have emerged as a promising tool for decomposing LLM representations into more interpretable features, but explaining the features captured by SAEs remains a challenging task. In this work, we propose SAGE (SAE AGentic Explainer), an agent-based framework that recasts feature interpretation from a passive, single-pass generation task into an active, explanation-driven process. SAGE implements a rigorous methodology by systematically formulating multiple explanations for each feature, designing targeted experiments to test them, and iteratively refining explanations based on empirical activation feedback. Experiments on features from SAEs of diverse language models demonstrate that SAGE produces explanations with significantly higher generative and predictive accuracy compared to state-of-the-art this http URL agent-based framework that recasts feature interpretation from a passive, single-pass generation task into an active, explanationdriven process. SAGE implements a rigorous methodology by systematically formulating multiple explanations for each feature, designing targeted experiments to test them, and iteratively refining explanations based on empirical activation feedback. Experiments on features from SAEs of diverse language models demonstrate that SAGE produces explanations with significantly higher generative and predictive accuracy compared to state-of-the-art baselines.

**arXiv ID:** 2511.20820
</details>

<details>
<summary><strong>Prompt-R1: Collaborative Automatic Prompting Framework via End-to-end Reinforcement Learning</strong> - Wenjin Liu, Haoran Luo, Xueyuan Lin, Haoming Liu, Tiesunlong Shen, Jiapu Wang, Rui Mao, Erik Cambria - [[pdf]](https://arxiv.org/pdf/2511.01016)</summary>

**Abstract:** Recently, advanced large language models (LLMs) have emerged at an increasingly rapid pace. However, when faced with complex problems, most users are often unable to provide accurate and effective prompts to interact with LLMs, thus limiting the performance of LLMs. To address this challenge, we propose Prompt-R1, an end-to-end reinforcement learning framework that uses a small-scale LLM to collaborate with large-scale LLMs, replacing user interaction to solve problems better. This collaboration is cast as a multi-turn prompt interaction, where the small-scale LLM thinks and generates prompts, and the large-scale LLM performs complex reasoning. A dual-constrained reward is designed to optimize for correctness, generation quality, and reasoning accuracy. Prompt-R1 provides a plug-and-play framework that supports both inference and training with various large-scale LLMs. Experiments on multiple public datasets show that Prompt-R1 significantly outperforms baseline models across tasks. Our code is publicly available at this https URL.

**arXiv ID:** 2511.01016
</details>

<details>
<summary><strong>AdvancedIF: Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM Instruction Following</strong> - Yun He, Wenzhe Li, Hejia Zhang, Songlin Li, Karishma Mandyam, Sopan Khosla, Yuanhao Xiong, Nanshu Wang, Xiaoliang Peng, Beibin Li, Shengjie Bi, Shishir G. Patil, Qi Qi, Shengyu Feng, Julian Katz-Samuels, Richard Yuanzhe Pang, Sujan Gonugondla, Hunter Lang, Yue Yu, Yundi Qian, Maryam Fazel-Zarandi, Licheng Yu, Amine Benhalloum, Hany Awadalla, Manaal Faruqui - [[pdf]](https://arxiv.org/pdf/2511.10507)</summary>

**Abstract:** Recent progress in large language models (LLMs) has led to impressive performance on a range of tasks, yet advanced instruction following (IF)-especially for complex, multi-turn, and system-prompted instructions-remains a significant challenge. Rigorous evaluation and effective training for such capabilities are hindered by the lack of high-quality, human-annotated benchmarks and reliable, interpretable reward signals. In this work, we introduce AdvancedIF (we will release this benchmark soon), a comprehensive benchmark featuring over 1,600 prompts and expert-curated rubrics that assess LLMs ability to follow complex, multi-turn, and system-level instructions. We further propose RIFL (Rubric-based Instruction-Following Learning), a novel post-training pipeline that leverages rubric generation, a finetuned rubric verifier, and reward shaping to enable effective reinforcement learning for instruction following. Extensive experiments demonstrate that RIFL substantially improves the instruction-following abilities of LLMs, achieving a 6.7% absolute gain on AdvancedIF and strong results on public benchmarks. Our ablation studies confirm the effectiveness of each component in RIFL. This work establishes rubrics as a powerful tool for both training and evaluating advanced IF in LLMs, paving the way for more capable and reliable AI systems.

**arXiv ID:** 2511.10507
</details>

<details>
<summary><strong>DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research</strong> - Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G. Finlayson, David Sontag, Tyler Murray, Sewon Min, Pradeep Dasigi, Luca Soldaini, Faeze Brahman, Wen-tau Yih, Tongshuang Wu, Luke Zettlemoyer, Yoon Kim, Hannaneh Hajishirzi, Pang Wei Koh - [[pdf]](https://arxiv.org/pdf/2511.19399)</summary>

**Abstract:** Deep research models perform multi-step research to produce long-form, well-attributed answers. However, most open deep research models are trained on easily verifiable short-form QA tasks via reinforcement learning with verifiable rewards (RLVR), which does not extend to realistic long-form tasks. We address this with Reinforcement Learning with Evolving Rubrics (RLER), in which we construct and maintain rubrics that co-evolve with the policy model during training; this allows the rubrics to incorporate information that the model has newly explored and to provide discriminative, on-policy feedback. Using RLER, we develop Deep Research Tulu (DR Tulu-8B), the first open model that is directly trained for open-ended, long-form deep research. Across four long-form deep research benchmarks in science, healthcare and general domains, DR Tulu substantially outperforms existing open deep research models, and matches or exceeds proprietary deep research systems, while being significantly smaller and cheaper per query. To facilitate future research, we release all data, models, and code, including our new MCP-based agent infrastructure for deep research systems.

**arXiv ID:** 2511.19399
</details>

<details>
<summary><strong>Staggered Environment Resets Improve Massively Parallel On-Policy Reinforcement Learning</strong> - Sid Bharthulwar, Stone Tao, Hao Su - [[pdf]](https://arxiv.org/pdf/2511.21011)</summary>

**Abstract:** Massively parallel GPU simulation environments have accelerated reinforcement learning (RL) research by enabling fast data collection for on-policy RL algorithms like Proximal Policy Optimization (PPO). To maximize throughput, it is common to use short rollouts per policy update, increasing the update-to-data (UTD) ra- tio. However, we find that, in this setting, standard synchronous resets introduce harmful nonstationarity, skewing the learning signal and destabilizing training. We introduce staggered resets, a simple yet effective technique where environments are initialized and reset at varied points within the task horizon. This yields training batches with greater temporal diversity, reducing the nonstationarity induced by synchronized rollouts. We characterize dimensions along which RL environments can benefit significantly from staggered resets through illustrative toy environ- ments. We then apply this technique to challenging high-dimensional robotics environments, achieving significantly higher sample efficiency, faster wall-clock convergence, and stronger final performance. Finally, this technique scales better with more parallel environments compared to naive synchronized rollouts.

**arXiv ID:** 2511.21011
</details>

<details>
<summary><strong>Learning When to Stop: Adaptive Latent Reasoning via Reinforcement Learning</strong> - Alex Ning, Yen-Ling Kuo, Gabe Gomes - [[pdf]](https://arxiv.org/pdf/2511.21581)</summary>

**Abstract:** Latent reasoning represents a new development in Transformer language models that has shown potential in compressing reasoning lengths compared to chain-of-thought reasoning. By directly passing the information-rich previous final latent state into the next sequence, latent reasoning removes the restriction to human language tokens as the medium for reasoning. We develop adaptive-length latent reasoning models and introduce a post-SFT reinforcement-learning methodology to optimize latent reasoning length by minimizing reasoning length while maintaining accuracy. This, in turn, further reduces compute usage and raises the bar on the compressive capabilities of latent reasoning models. Experiments on the Llama 3.2 1B model and the GSM8K-Aug dataset show a $52\%$ drop in total reasoning length with no penalty to accuracy. In future work, we plan to extend to additional models and datasets, analyze relationships between training coefficients, experiment with architecture variations, and continue our knowledge distillation for latent reasoning SFT efforts. We make our code and pretrained weights available at this https URL.

**arXiv ID:** 2511.21581
</details>

<details>
<summary><strong>Cryptocurrency Portfolio Management with Reinforcement Learning: Soft Actor--Critic and Deep Deterministic Policy Gradient Algorithms</strong> - Kamal Paykan - [[pdf]](https://arxiv.org/pdf/2511.20678)</summary>

**Abstract:** This paper proposes a reinforcement learning--based framework for cryptocurrency portfolio management using the Soft Actor--Critic (SAC) and Deep Deterministic Policy Gradient (DDPG) algorithms. Traditional portfolio optimization methods often struggle to adapt to the highly volatile and nonlinear dynamics of cryptocurrency markets. To address this, we design an agent that learns continuous trading actions directly from historical market data through interaction with a simulated trading environment. The agent optimizes portfolio weights to maximize cumulative returns while minimizing downside risk and transaction costs. Experimental evaluations on multiple cryptocurrencies demonstrate that the SAC and DDPG agents outperform baseline strategies such as equal-weighted and mean--variance portfolios. The SAC algorithm, with its entropy-regularized objective, shows greater stability and robustness in noisy market conditions compared to DDPG. These results highlight the potential of deep reinforcement learning for adaptive and data-driven portfolio management in cryptocurrency markets.

**arXiv ID:** 2511.20678
</details>

<details>
<summary><strong>Independent policy gradient-based reinforcement learning for economic and reliable energy management of multi-microgrid systems</strong> - Junkai Hu, Li Xia - [[pdf]](https://arxiv.org/pdf/2511.20977)</summary>

**Abstract:** Efficiency and reliability are both crucial for energy management, especially in multi-microgrid systems (MMSs) integrating intermittent and distributed renewable energy sources. This study investigates an economic and reliable energy management problem in MMSs under a distributed scheme, where each microgrid independently updates its energy management policy in a decentralized manner to optimize the long-term system performance collaboratively. We introduce the mean and variance of the exchange power between the MMS and the main grid as indicators for the economic performance and reliability of the system. Accordingly, we formulate the energy management problem as a mean-variance team stochastic game (MV-TSG), where conventional methods based on the maximization of expected cumulative rewards are unsuitable for variance metrics. To solve MV-TSGs, we propose a fully distributed independent policy gradient algorithm, with rigorous convergence analysis, for scenarios with known model parameters. For large-scale scenarios with unknown model parameters, we further develop a deep reinforcement learning algorithm based on independent policy gradients, enabling data-driven policy optimization. Numerical experiments in two scenarios validate the effectiveness of the proposed methods. Our approaches fully leverage the distributed computational capabilities of MMSs and achieve a well-balanced trade-off between economic performance and operational reliability.

**arXiv ID:** 2511.20977
</details>

<details>
<summary><strong>Single- vs. Dual-Policy Reinforcement Learning for Dynamic Bike Rebalancing</strong> - Jiaqi Liang, Defeng Liu, Sanjay Dominik Jena, Andrea Lodi, Thibaut Vidal - [[pdf]](https://arxiv.org/pdf/2402.03589)</summary>

**Abstract:** Bike-sharing systems (BSS) provide a sustainable urban mobility solution, but ensuring their reliability requires effective rebalancing strategies to address stochastic demand and prevent station imbalances. This paper proposes reinforcement learning (RL) algorithms for dynamic rebalancing problem with multiple vehicles, introducing and comparing two RL approaches: Single-policy RL and Dual-policy RL. We formulate this network optimization problem as a Markov Decision Process within a continuous-time framework, allowing vehicles to make independent and cooperative rebalancing decisions without synchronization constraints. In the first approach, a single deep Q-network (DQN) is trained to jointly learn inventory and routing decisions. The second approach decouples node-level inventory decisions from arc-level vehicle routing, enhancing learning efficiency and adaptability. A high-fidelity simulator under the first-arrive-first-serve rule is developed to estimate rewards across diverse demand scenarios influenced by temporal and weather variations. Extensive experiments demonstrate that while the single-policy model is competitive against several benchmarks, the dual-policy model significantly reduces lost demand. These findings provide valuable insights for bike-sharing operators, reinforcing the potential of RL for real-time rebalancing and paving the way for more adaptive and intelligent urban mobility solutions.

**arXiv ID:** 2402.03589
</details>

<details>
<summary><strong>Optimized scheduling of electricity-heat cooperative system considering wind energy consumption and peak shaving and valley filling</strong> - Jin Ye, Lingmei Wang, Shujian Zhang, Haihang Wu - [[pdf]](https://arxiv.org/pdf/2511.15250)</summary>

**Abstract:** With the global energy transition and rapid development of renewable energy, the scheduling optimization challenge for combined power-heat systems under new energy integration and multiple uncertainties has become increasingly prominent. Addressing this challenge, this study proposes an intelligent scheduling method based on the improved Dual-Delay Deep Deterministic Policy Gradient (PVTD3) algorithm. System optimization is achieved by introducing a penalty term for grid power purchase variations. Simulation results demonstrate that under three typical scenarios (10%, 20%, and 30% renewable penetration), the PVTD3 algorithm reduces the system's comprehensive cost by 6.93%, 12.68%, and 13.59% respectively compared to the traditional TD3 algorithm. Concurrently, it reduces the average fluctuation amplitude of grid power purchases by 12.8%. Regarding energy storage management, the PVTD3 algorithm reduces the end-time state values of low-temperature thermal storage tanks by 7.67-17.67 units while maintaining high-temperature tanks within the 3.59-4.25 safety operating range. Multi-scenario comparative validation demonstrates that the proposed algorithm not only excels in economic efficiency and grid stability but also exhibits superior sustainable scheduling capabilities in energy storage device management.

**arXiv ID:** 2511.15250
</details>

<details>
<summary><strong>Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry</strong> - Feiyang Pan, Shenghe Zheng, Chunyan Yin, Guangbin Dou - [[pdf]](https://arxiv.org/pdf/2511.21083)</summary>

**Abstract:** Visual-Inertial Odometry (VIO) is a critical component for robust ego-motion estimation, enabling foundational capabilities such as autonomous navigation in robotics and real-time 6-DoF tracking for augmented reality. Existing methods face a well-known trade-off: filter-based approaches are efficient but prone to drift, while optimization-based methods, though accurate, rely on computationally prohibitive Visual-Inertial Bundle Adjustment (VIBA) that is difficult to run on resource-constrained platforms. Rather than removing VIBA altogether, we aim to reduce how often and how heavily it must be invoked. To this end, we cast two key design choices in modern VIO, when to run the visual frontend and how strongly to trust its output, as sequential decision problems, and solve them with lightweight reinforcement learning (RL) agents. Our framework introduces a lightweight, dual-pronged RL policy that serves as our core contribution: (1) a Select Agent intelligently gates the entire VO pipeline based only on high-frequency IMU data; and (2) a composite Fusion Agent that first estimates a robust velocity state via a supervised network, before an RL policy adaptively fuses the full (p, v, q) state. Experiments on the EuRoC MAV and TUM-VI datasets show that, in our unified evaluation, the proposed method achieves a more favorable accuracy-efficiency-memory trade-off than prior GPU-based VO/VIO systems: it attains the best average ATE while running up to 1.77 times faster and using less GPU memory. Compared to classical optimization-based VIO systems, our approach maintains competitive trajectory accuracy while substantially reducing computational load.

**arXiv ID:** 2511.21083
</details>

<details>
<summary><strong>MarketGen: A Scalable Simulation Platform with Auto-Generated Embodied Supermarket Environments</strong> - Xu Hu, Yiyang Feng, Junran Peng, Jiawei He, Liyi Chen, Chuanchen Luo, Xucheng Yin, Qing Li, Zhaoxiang Zhang - [[pdf]](https://arxiv.org/pdf/2511.21161)</summary>

**Abstract:** The development of embodied agents for complex commercial environments is hindered by a critical gap in existing robotics datasets and benchmarks, which primarily focus on household or tabletop settings with short-horizon tasks. To address this limitation, we introduce MarketGen, a scalable simulation platform with automatic scene generation for complex supermarket environments. MarketGen features a novel agent-based Procedural Content Generation (PCG) framework. It uniquely supports multi-modal inputs (text and reference images) and integrates real-world design principles to automatically generate complete, structured, and realistic supermarkets. We also provide an extensive and diverse 3D asset library with a total of 1100+ supermarket goods and parameterized facilities assets. Building on this generative foundation, we propose a novel benchmark for assessing supermarket agents, featuring two daily tasks in a supermarket: (1) Checkout Unloading: long-horizon tabletop tasks for cashier agents, and (2) In-Aisle Item Collection: complex mobile manipulation tasks for salesperson agents. We validate our platform and benchmark through extensive experiments, including the deployment of a modular agent system and successful sim-to-real transfer. MarketGen provides a comprehensive framework to accelerate research in embodied AI for complex commercial applications.

**arXiv ID:** 2511.21161
</details>

<details>
<summary><strong>Kinematics-Aware Multi-Policy Reinforcement Learning for Force-Capable Humanoid Loco-Manipulation</strong> - Kaiyan Xiao, Zihan Xu, Cheng Zhe, Chengju Liu, Qijun Chen - [[pdf]](https://arxiv.org/pdf/2511.21169)</summary>

**Abstract:** Humanoid robots, with their human-like morphology, hold great potential for industrial applications. However, existing loco-manipulation methods primarily focus on dexterous manipulation, falling short of the combined requirements for dexterity and proactive force interaction in high-load industrial scenarios. To bridge this gap, we propose a reinforcement learning-based framework with a decoupled three-stage training pipeline, consisting of an upper-body policy, a lower-body policy, and a delta-command policy. To accelerate upper-body training, a heuristic reward function is designed. By implicitly embedding forward kinematics priors, it enables the policy to converge faster and achieve superior performance. For the lower body, a force-based curriculum learning strategy is developed, enabling the robot to actively exert and regulate interaction forces with the environment.

**arXiv ID:** 2511.21169
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
