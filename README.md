# Agent arXiv Daily

**Last Updated:** 2025-11-27 02:07:43

**Total Papers:** 74

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
<summary><strong>Understanding Human-Chatbot Romance: A Qualitative and Quantitative Study on Romantic Fantasy and Other Interpersonal Characteristics</strong> - Paula Ebner, Jessica Szczuka - [[pdf]](https://arxiv.org/pdf/2503.00195)</summary>

**Abstract:** LLM-based chatbots are now being specifically designed to facilitate social companionship, even romantic relationships, incorporating features that parallel human relationship dynamics. This has led a subset of users to form romantic relationships with chatbots. Understanding which interpersonal characteristics drive individuals to form intense, emotional bonds with chatbots is crucial for comprehending the potential psychological and societal impacts of romantic human-chatbot relationships. This mixed-methods study investigates psychological predictors of relationship intensity among individuals currently in romantic relationships with chatbots. Romantic and sexual fantasy, promising constructs not previously investiagted in this context, are examined alongside previously discussed factors (loneliness, anthropomorphism, attachment orientation, and sexual sensation seeking). In Study 1, quantitative data from individuals with chatbot partners (N=92) showed that romantic fantasy explained the most variance in relationship intensity, with additional contributions from anthropomorphism and avoidant attachment. Contrary to expectations, the other predictors, including loneliness, did not significantly predict intensity. In Study 2, 15 qualitative interviews illuminated how users employ romantic fantasy to enhance their relationships, describing active fantasy use to shape interactions and a desire for their chatbot to feel as human as possible. This study provides the first quantitative sample of this under-researched population, explaining who might form more intense romantic relationships with chatbots.

**arXiv ID:** 2503.00195
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (16 papers)</h2></summary>

<details>
<summary><strong>HeaRT: A Hierarchical Circuit Reasoning Tree-Based Agentic Framework for AMS Design Optimization</strong> - Souradip Poddar, Chia-Tung Ho, Ziming Wei, Weidong Cao, Haoxing Ren, David Z. Pan - [[pdf]](https://arxiv.org/pdf/2511.19669)</summary>

**Abstract:** Conventional AI-driven AMS design automation algorithms remain constrained by their reliance on high-quality datasets to capture underlying circuit behavior, coupled with poor transferability across architectures, and a lack of adaptive mechanisms. This work proposes HeaRT, a foundational reasoning engine for automation loops and a first step toward intelligent, adaptive, human-style design optimization. HeaRT consistently demonstrates reasoning accuracy >97% and Pass@1 performance >98% across our 40-circuit benchmark repository, even as circuit complexity increases, while operating at <0.5x real-time token budget of SOTA baselines. Our experiments show that HeaRT yields >3x faster convergence in both sizing and topology design adaptation tasks across diverse optimization approaches, while preserving prior design intent.

**arXiv ID:** 2511.19669
</details>

<details>
<summary><strong>Fara-7B: An Efficient Agentic Model for Computer Use</strong> - Ahmed Awadallah, Yash Lara, Raghav Magazine, Hussein Mozannar, Akshay Nambi, Yash Pandya, Aravind Rajeswaran, Corby Rosset, Alexey Taymanov, Vibhav Vineet, Spencer Whitehead, Andrew Zhao - [[pdf]](https://arxiv.org/pdf/2511.19663)</summary>

**Abstract:** Progress in computer use agents (CUAs) has been constrained by the absence of large and high-quality datasets that capture how humans interact with a computer. While LLMs have thrived on abundant textual data, no comparable corpus exists for CUA trajectories. To address these gaps, we introduce FaraGen, a novel synthetic data generation system for multi-step web tasks. FaraGen can propose diverse tasks from frequently used websites, generate multiple solution attempts, and filter successful trajectories using multiple verifiers. It achieves high throughput, yield, and diversity for multi-step web tasks, producing verified trajectories at approximately $1 each. We use this data to train Fara-7B, a native CUA model that perceives the computer using only screenshots, executes actions via predicted coordinates, and is small enough to run on-device. We find that Fara-7B outperforms other CUA models of comparable size on benchmarks like WebVoyager, Online-Mind2Web, and WebTailBench -- our novel benchmark that better captures under-represented web tasks in pre-existing benchmarks. Furthermore, Fara-7B is competitive with much larger frontier models, illustrating key benefits of scalable data generation systems in advancing small efficient agentic models. We are making Fara-7B open-weight on Microsoft Foundry and HuggingFace, and we are releasing WebTailBench.

**arXiv ID:** 2511.19663
</details>

<details>
<summary><strong>NOEM$^{3}$A: A Neuro-Symbolic Ontology-Enhanced Method for Multi-Intent Understanding in Mobile Agents</strong> - Ioannis Tzachristas, Aifen Sui - [[pdf]](https://arxiv.org/pdf/2511.19780)</summary>

**Abstract:** We introduce a neuro-symbolic framework for multi-intent understanding in mobile AI agents by integrating a structured intent ontology with compact language models. Our method leverages retrieval-augmented prompting, logit biasing and optional classification heads to inject symbolic intent structure into both input and output representations. We formalize a new evaluation metric-Semantic Intent Similarity (SIS)-based on hierarchical ontology depth, capturing semantic proximity even when predicted intents differ lexically. Experiments on a subset of ambiguous/demanding dialogues of MultiWOZ 2.3 (with oracle labels from GPT-o3) demonstrate that a 3B Llama model with ontology augmentation approaches GPT-4 accuracy (85% vs 90%) at a tiny fraction of the energy and memory footprint. Qualitative comparisons show that ontology-augmented models produce more grounded, disambiguated multi-intent interpretations. Our results validate symbolic alignment as an effective strategy for enabling accurate and efficient on-device NLU.

**arXiv ID:** 2511.19780
</details>

<details>
<summary><strong>"Are We Done Yet?": A Vision-Based Judge for Autonomous Task Completion of Computer Use Agents</strong> - Marta Sumyk, Oleksandr Kosovan - [[pdf]](https://arxiv.org/pdf/2511.20067)</summary>

**Abstract:** Computer Use Agents (CUAs) are designed to autonomously operate digital interfaces, yet they often fail to reliably determine whether a given task has been completed. We present an autonomous evaluation and feedback framework that uses vision-language models to assess task completion directly from screenshots and task descriptions. Our dataset covers 42 built-in macOS applications and 1,260 human-labeled tasks across a wide range of scenarios. Our framework achieves up to 73 percent accuracy in task success detection and yields an average relative improvement of 27 percent in overall task success when evaluator feedback is applied. These results show that vision-based evaluation can serve as an effective feedback mechanism that improves the reliability and self-correction of autonomous computer-use agents.

**arXiv ID:** 2511.20067
</details>

<details>
<summary><strong>VICoT-Agent: A Vision-Interleaved Chain-of-Thought Framework for Interpretable Multimodal Reasoning and Scalable Remote Sensing Analysis</strong> - Chujie Wang, Zhiyuan Luo, Ruiqi Liu, Can Ran, Shenghua Fan, Xi Chen, Chu He - [[pdf]](https://arxiv.org/pdf/2511.20085)</summary>

**Abstract:** The current remote sensing image analysis task is increasingly evolving from traditional object recognition to complex intelligence reasoning, which places higher requirements on the model's reasoning ability and the flexibility of tool invocation. To this end, we propose a new multimodal agent framework, Vision-Interleaved Chain-of-Thought Framework (VICoT), which implements explicit multi-round reasoning by dynamically incorporating visual tools into the chain of thought. Through a stack-based reasoning structure and a modular MCP-compatible tool suite, VICoT enables LLMs to efficiently perform multi-round, interleaved vision-language reasoning tasks with strong generalization and this http URL also propose the Reasoning Stack distillation method to migrate complex Agent behaviors to small, lightweight models, which ensures the reasoning capability while significantly reducing complexity. Experiments on multiple remote sensing benchmarks demonstrate that VICoT significantly outperforms existing SOTA frameworks in reasoning transparency, execution efficiency, and generation quality.

**arXiv ID:** 2511.20085
</details>

<details>
<summary><strong>CostNav: A Navigation Benchmark for Cost-Aware Evaluation of Embodied Agents</strong> - Haebin Seong, Sungmin Kim, Minchan Kim, Yongjun Cho, Myunchul Joe, Suhwan Choi, Jaeyoon Jung, Jiyong Youn, Yoonshik Kim, Samwoo Seong, Yubeen Park, Youngjae Yu, Yunsung Lee - [[pdf]](https://arxiv.org/pdf/2511.20216)</summary>

**Abstract:** Existing navigation benchmarks focus on task success metrics while overlooking economic viability -- critical for commercial deployment of autonomous delivery robots. We introduce \emph{CostNav}, a \textbf{Micro-Navigation Economic Testbed} that evaluates embodied agents through comprehensive cost-revenue analysis aligned with real-world business operations. CostNav models the complete economic lifecycle including hardware, training, energy, maintenance costs, and delivery revenue with service-level agreements, using industry-derived parameters. \textbf{To our knowledge, CostNav is the first work to quantitatively expose the gap between navigation research metrics and commercial viability}, revealing that optimizing for task success fundamentally differs from optimizing for economic deployment. Our cost model uses parameters derived from industry data sources (energy rates, delivery service pricing), and we project from a reduced-scale simulation to realistic deliveries. Under this projection, the baseline achieves 43.0\% SLA compliance but is \emph{not} commercially viable: yielding a loss of \$30.009 per run with no finite break-even point, because operating costs are dominated by collision-induced maintenance, which accounts for 99.7\% of per-run costs and highlights collision avoidance as a key optimization target. We demonstrate a learning-based on-device navigation baseline and establish a foundation for evaluating rule-based navigation, imitation learning, and cost-aware RL training. CostNav bridges the gap between navigation research and commercial deployment, enabling data-driven decisions about economic trade-offs across navigation paradigms.

**arXiv ID:** 2511.20216
</details>

<details>
<summary><strong>Prune-Then-Plan: Step-Level Calibration for Stable Frontier Exploration in Embodied Question Answering</strong> - Noah Frahm, Prakrut Patel, Yue Zhang, Shoubin Yu, Mohit Bansal, Roni Sengupta - [[pdf]](https://arxiv.org/pdf/2511.19768)</summary>

**Abstract:** Large vision-language models (VLMs) have improved embodied question answering (EQA) agents by providing strong semantic priors for open-vocabulary reasoning. However, when used directly for step-level exploration, VLMs often exhibit frontier oscillations, unstable back-and-forth movements caused by overconfidence and miscalibration, leading to inefficient navigation and degraded answer quality. We propose Prune-Then-Plan, a simple and effective framework that stabilizes exploration through step-level calibration. Instead of trusting raw VLM scores, our method prunes implausible frontier choices using a Holm-Bonferroni inspired pruning procedure and then delegates final decisions to a coverage-based planner. This separation converts overconfident predictions into conservative, interpretable actions by relying on human-level judgments to calibrate the step-level behavior of VLMs. Integrated into the 3D-Mem EQA framework, our approach achieves relative improvements of up to 49% and 33% in visually grounded SPL and LLM-Match metrics respectively over baselines. Overall, our method achieves better scene coverage under equal exploration budgets on both OpenEQA and EXPRESS-Bench datasets.

**arXiv ID:** 2511.19768
</details>

<details>
<summary><strong>Cross-LLM Generalization of Behavioral Backdoor Detection in AI Agent Supply Chains</strong> - Arun Chowdary Sanna - [[pdf]](https://arxiv.org/pdf/2511.19874)</summary>

**Abstract:** As AI agents become integral to enterprise workflows, their reliance on shared tool libraries and pre-trained components creates significant supply chain vulnerabilities. While previous work has demonstrated behavioral backdoor detection within individual LLM architectures, the critical question of cross-LLM generalization remains unexplored, a gap with serious implications for organizations deploying multiple AI systems. We present the first systematic study of cross-LLM behavioral backdoor detection, evaluating generalization across six production LLMs (GPT-5.1, Claude Sonnet 4.5, Grok 4.1, Llama 4 Maverick, GPT-OSS 120B, and DeepSeek Chat V3.1). Through 1,198 execution traces and 36 cross-model experiments, we quantify a critical finding: single-model detectors achieve 92.7% accuracy within their training distribution but only 49.2% across different LLMs, a 43.4 percentage point generalization gap equivalent to random guessing. Our analysis reveals that this gap stems from model-specific behavioral signatures, particularly in temporal features (coefficient of variation > 0.8), while structural features remain stable across architectures. We show that model-aware detection incorporating model identity as an additional feature achieves 90.6% accuracy universally across all evaluated models. We release our multi-LLM trace dataset and detection framework to enable reproducible research.

**arXiv ID:** 2511.19874
</details>

<details>
<summary><strong>BrowseSafe: Understanding and Preventing Prompt Injection Within AI Browser Agents</strong> - Kaiyuan Zhang, Mark Tenenholtz, Kyle Polley, Jerry Ma, Denis Yarats, Ninghui Li - [[pdf]](https://arxiv.org/pdf/2511.20597)</summary>

**Abstract:** The integration of artificial intelligence (AI) agents into web browsers introduces security challenges that go beyond traditional web application threat models. Prior work has identified prompt injection as a new attack vector for web agents, yet the resulting impact within real-world environments remains insufficiently understood.
In this work, we examine the landscape of prompt injection attacks and synthesize a benchmark of attacks embedded in realistic HTML payloads. Our benchmark goes beyond prior work by emphasizing injections that can influence real-world actions rather than mere text outputs, and by presenting attack payloads with complexity and distractor frequency similar to what real-world agents encounter. We leverage this benchmark to conduct a comprehensive empirical evaluation of existing defenses, assessing their effectiveness across a suite of frontier AI models. We propose a multi-layered defense strategy comprising both architectural and model-based defenses to protect against evolving prompt injection attacks. Our work offers a blueprint for designing practical, secure web agents through a defense-in-depth approach.

**arXiv ID:** 2511.20597
</details>

<details>
<summary><strong>Multi-Modal Data Exploration via Language Agents</strong> - Farhad Nooralahzadeh, Yi Zhang, Jonathan Furst, Kurt Stockinger - [[pdf]](https://arxiv.org/pdf/2412.18428)</summary>

**Abstract:** International enterprises, organizations, and hospitals collect large amounts of multi-modal data stored in databases, text documents, images, and videos. While there has been recent progress in the separate fields of multi-modal data exploration as well as in database systems that automatically translate natural language questions to database query languages, the research challenge of querying both structured databases and unstructured modalities (e.g., texts, images) in natural language remains largely unexplored. In this paper, we propose M$^2$EX -a system that enables multi-modal data exploration via language agents. Our approach is based on the following research contributions: (1) Our system is inspired by a real-world use case that enables users to explore multi-modal information systems. (2) M$^2$EX leverages an LLM-based agentic AI framework to decompose a natural language question into subtasks such as text-to-SQL generation and image analysis and to orchestrate modality-specific experts in an efficient query plan. (3) Experimental results on multi-modal datasets, encompassing relational data, text, and images, demonstrate that our system outperforms state-of-the-art multi-modal exploration systems, excelling in both accuracy and various performance metrics, including query latency, API costs, and planning efficiency, thanks to the more effective utilization of the reasoning capabilities of LLMs.

**arXiv ID:** 2412.18428
</details>

<details>
<summary><strong>Agentar-Scale-SQL: Advancing Text-to-SQL through Orchestrated Test-Time Scaling</strong> - Pengfei Wang, Baolin Sun, Xuemei Dong, Yaxun Dai, Hongwei Yuan, Mengdie Chu, Yingqi Gao, Xiang Qi, Peng Zhang, Ying Yan - [[pdf]](https://arxiv.org/pdf/2509.24403)</summary>

**Abstract:** State-of-the-art (SOTA) Text-to-SQL methods still lag significantly behind human experts on challenging benchmarks like BIRD. Current approaches that explore test-time scaling lack an orchestrated strategy and neglect the model's internal reasoning process. To bridge this gap, we introduce Agentar-Scale-SQL, a novel framework leveraging scalable computation to improve performance. Agentar-Scale-SQL implements an Orchestrated Test-Time Scaling strategy that synergistically combines three distinct perspectives: i) Internal Scaling via RL-enhanced Intrinsic Reasoning, ii) Sequential Scaling through Iterative Refinement, and iii) Parallel Scaling using Diverse Synthesis and Tournament Selection. Agentar-Scale-SQL is a general-purpose framework designed for easy adaptation to new databases and more powerful language models. Extensive experiments show that Agentar-Scale-SQL achieves SOTA performance on the BIRD benchmark, reaching 81.67% execution accuracy on the test set and ranking first on the official leaderboard, demonstrating an effective path toward human-level performance.

**arXiv ID:** 2509.24403
</details>

<details>
<summary><strong>OceanGym: A Benchmark Environment for Underwater Embodied Agents</strong> - Yida Xue, Mingjun Mao, Xiangyuan Ru, Yuqi Zhu, Baochang Ren, Shuofei Qiao, Mengru Wang, Shumin Deng, Xinyu An, Ningyu Zhang, Ying Chen, Huajun Chen - [[pdf]](https://arxiv.org/pdf/2509.26536)</summary>

**Abstract:** We introduce OceanGym, the first comprehensive benchmark for ocean underwater embodied agents, designed to advance AI in one of the most demanding real-world environments. Unlike terrestrial or aerial domains, underwater settings present extreme perceptual and decision-making challenges, including low visibility, dynamic ocean currents, making effective agent deployment exceptionally difficult. OceanGym encompasses eight realistic task domains and a unified agent framework driven by Multi-modal Large Language Models (MLLMs), which integrates perception, memory, and sequential decision-making. Agents are required to comprehend optical and sonar data, autonomously explore complex environments, and accomplish long-horizon objectives under these harsh conditions. Extensive experiments reveal substantial gaps between state-of-the-art MLLM-driven agents and human experts, highlighting the persistent difficulty of perception, planning, and adaptability in ocean underwater environments. By providing a high-fidelity, rigorously designed platform, OceanGym establishes a testbed for developing robust embodied AI and transferring these capabilities to real-world autonomous ocean underwater vehicles, marking a decisive step toward intelligent agents capable of operating in one of Earth's last unexplored frontiers. The code and data are available at this https URL.

**arXiv ID:** 2509.26536
</details>

<details>
<summary><strong>Development of a Testbed for Autonomous Vehicles: Integrating MPC Control with Monocular Camera Lane Detection</strong> - Shantanu Rahman, Nayeb Hasin, Mainul Islam - [[pdf]](https://arxiv.org/pdf/2511.19655)</summary>

**Abstract:** Autonomous vehicles are becoming popular day by day not only for autonomous road traversal but also for industrial automation, farming and military. Most of the standard vehicles follow the Ackermann style steering mechanism. This has become to de facto standard for large and long faring vehicles. The local planner of an autonomous vehicle controls the low-level vehicle movement upon which the vehicle will perform its motor actuation. In our work, we focus on autonomous vehicles in road and perform experiments to analyze the effect of low-level controllers in the simulation and a real environment. To increase the precision and stability of trajectory tracking in autonomous cars, a novel method that combines lane identification with Model Predictive Control (MPC) is presented. The research focuses on camera-equipped autonomous vehicles and uses methods like edge recognition, sliding window-based straight-line identification for lane line extraction, and dynamic region of interest (ROI) extraction. Next, to follow the identified lane line, an MPC built on a bicycle vehicle dynamics model is created. A single-lane road simulation model is built using ROS Gazebo and tested in order to verify the controller's performance. The root mean square error between the optimal tracking trajectory and the target trajectory was reduced by 27.65% in the simulation results, demonstrating the high robustness and flexibility of the developed controller.

**arXiv ID:** 2511.19655
</details>

<details>
<summary><strong>GigaWorld-0: World Models as Data Engine to Empower Embodied AI</strong> - GigaWorld Team, Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Haoyun Li, Jiagang Zhu, Kerui Li, Mengyuan Xu, Qiuping Deng, Siting Wang, Wenkang Qin, Xinze Chen, Xiaofeng Wang, Yankai Wang, Yu Cao, Yifan Chang, Yuan Xu, Yun Ye, Yang Wang, Yukun Zhou, Zhengyuan Zhang, Zhehao Dong, Zheng Zhu - [[pdf]](https://arxiv.org/pdf/2511.19861)</summary>

**Abstract:** World models are emerging as a foundational paradigm for scalable, data-efficient embodied AI. In this work, we present GigaWorld-0, a unified world model framework designed explicitly as a data engine for Vision-Language-Action (VLA) learning. GigaWorld-0 integrates two synergistic components: GigaWorld-0-Video, which leverages large-scale video generation to produce diverse, texture-rich, and temporally coherent embodied sequences under fine-grained control of appearance, camera viewpoint, and action semantics; and GigaWorld-0-3D, which combines 3D generative modeling, 3D Gaussian Splatting reconstruction, physically differentiable system identification, and executable motion planning to ensure geometric consistency and physical realism. Their joint optimization enables the scalable synthesis of embodied interaction data that is visually compelling, spatially coherent, physically plausible, and instruction-aligned. Training at scale is made feasible through our efficient GigaTrain framework, which exploits FP8-precision and sparse attention to drastically reduce memory and compute requirements. We conduct comprehensive evaluations showing that GigaWorld-0 generates high-quality, diverse, and controllable data across multiple dimensions. Critically, VLA model (e.g., GigaBrain-0) trained on GigaWorld-0-generated data achieve strong real-world performance, significantly improving generalization and task success on physical robots without any real-world interaction during training.

**arXiv ID:** 2511.19861
</details>

<details>
<summary><strong>Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI</strong> - Xinhao Liu, Jiaqi Li, Youming Deng, Ruxin Chen, Yingjia Zhang, Yifei Ma, Li Guo, Yiming Li, Jing Zhang, Chen Feng - [[pdf]](https://arxiv.org/pdf/2511.20620)</summary>

**Abstract:** Reproducible closed-loop evaluation remains a major bottleneck in Embodied AI such as visual navigation. A promising path forward is high-fidelity simulation that combines photorealistic sensor rendering with geometrically grounded interaction in complex, open-world urban environments. Although recent video-3DGS methods ease open-world scene capturing, they are still unsuitable for benchmarking due to large visual and geometric sim-to-real gaps. To address these challenges, we introduce Wanderland, a real-to-sim framework that features multi-sensor capture, reliable reconstruction, accurate geometry, and robust view synthesis. Using this pipeline, we curate a diverse dataset of indoor-outdoor urban scenes and systematically demonstrate how image-only pipelines scale poorly, how geometry quality impacts novel view synthesis, and how all of these adversely affect navigation policy learning and evaluation reliability. Beyond serving as a trusted testbed for embodied navigation, Wanderland's rich raw sensor data further allows benchmarking of 3D reconstruction and novel view synthesis models. Our work establishes a new foundation for reproducible research in open-world embodied AI. Project website is at this https URL.

**arXiv ID:** 2511.20620
</details>

<details>
<summary><strong>Final Happiness: What Intelligent User Interfaces Can Do for The Lonely Dying</strong> - Yibo Meng, Rong Fu, Lyumanshan Ye, Zhiming Liu, Zhixin Cai, Xiaolan Ding, Yan Guan - [[pdf]](https://arxiv.org/pdf/2511.14164)</summary>

**Abstract:** This study explores the design of Intelligent User Interfaces (IUIs) to address the profound existential loneliness of terminally ill individuals. While Human-Computer Interaction (HCI) has made inroads in "Thanatechnology," current research often focuses on practical aspects like digital legacy management, overlooking the subjective, existential needs of those facing death in isolation. To address this gap, we conducted in-depth qualitative interviews with 14 lonely, terminally ill individuals. Our core contributions are: (1) An empirically-grounded model articulating the complex psychological, practical, social, and spiritual needs of this group; (2) The "Three Pillars, Twelve Principles" framework for designing IUIs as "Existential Companions"; and (3) A critical design directive derived from user evaluations: technology in this context should aim for transcendence over simulation. The findings suggest that IUIs should create experiences that augment or surpass human capabilities, rather than attempting to simulate basic human connections, which can paradoxically deepen loneliness. This research provides a clear, user-centered path for designing technology that serves not as a "tool for dying," but as a "partner for living fully until the end".

**arXiv ID:** 2511.14164
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>Bridging Symbolic Control and Neural Reasoning in LLM Agents: The Structured Cognitive Loop</strong> - Myung Ho Kim - [[pdf]](https://arxiv.org/pdf/2511.17673)</summary>

**Abstract:** Large language model agents suffer from fundamental architectural problems: entangled reasoning and execution, memory volatility, and uncontrolled action sequences. We introduce Structured Cognitive Loop (SCL), a modular architecture that explicitly separates agent cognition into five phases: Retrieval, Cognition, Control, Action, and Memory (R-CCAM). At the core of SCL is Soft Symbolic Control, an adaptive governance mechanism that applies symbolic constraints to probabilistic inference, preserving neural flexibility while restoring the explainability and controllability of classical symbolic systems. Through empirical validation on multi-step conditional reasoning tasks, we demonstrate that SCL achieves zero policy violations, eliminates redundant tool calls, and maintains complete decision traceability. These results address critical gaps in existing frameworks such as ReAct, AutoGPT, and memory-augmented approaches. Our contributions are threefold: (1) we situate SCL within the taxonomy of hybrid intelligence, differentiating it from prompt-centric and memory-only approaches; (2) we formally define Soft Symbolic Control and contrast it with neuro-symbolic AI; and (3) we derive three design principles for trustworthy agents: modular decomposition, adaptive symbolic governance, and transparent state management. We provide a complete open-source implementation demonstrating the R-CCAM loop architecture, alongside a live GPT-4o-powered travel planning agent. By connecting expert system principles with modern LLM capabilities, this work offers a practical and theoretically grounded path toward reliable, explainable, and governable AI agents.

**arXiv ID:** 2511.17673
</details>

<details>
<summary><strong>Simulating Macroeconomic Expectations using LLM Agents</strong> - Jianhao Lin, Lexuan Sun, Yixin Yan - [[pdf]](https://arxiv.org/pdf/2505.17648)</summary>

**Abstract:** We introduce a novel framework for simulating macroeconomic expectations using LLM Agents. By constructing LLM Agents equipped with various functional modules, we replicate three representative survey experiments involving several expectations across different types of economic agents. Our results show that although the expectations simulated by LLM Agents are more homogeneous than those of humans, they consistently outperform LLMs relying simply on prompt engineering, and possess human-like mental mechanisms. Evaluation reveals that these capabilities stem from the contributions of their components, offering guidelines for their architectural design. Our approach complements traditional methods and provides new insights into AI behavioral science in macroeconomic research

**arXiv ID:** 2505.17648
</details>

<details>
<summary><strong>Improved LLM Agents for Financial Document Question Answering</strong> - Nelvin Tan, Zian Seng, Liang Zhang, Yu-Ching Shih, Dong Yang, Amol Salunkhe - [[pdf]](https://arxiv.org/pdf/2506.08726)</summary>

**Abstract:** Large language models (LLMs) have shown impressive capabilities on numerous natural language processing tasks. However, LLMs still struggle with numerical question answering for financial documents that include tabular and textual data. Recent works have showed the effectiveness of critic agents (i.e., self-correction) for this task given oracle labels. Building upon this framework, this paper examines the effectiveness of the traditional critic agent when oracle labels are not available, and show, through experiments, that this critic agent's performance deteriorates in this scenario. With this in mind, we present an improved critic agent, along with the calculator agent which outperforms the previous state-of-the-art approach (program-of-thought) and is safer. Furthermore, we investigate how our agents interact with each other, and how this interaction affects their performance.

**arXiv ID:** 2506.08726
</details>

<details>
<summary><strong>Adaptive LLM Agents: Toward Personalized Empathetic Care</strong> - Priyanka Singh, Sebastian Von Mammen - [[pdf]](https://arxiv.org/pdf/2511.20080)</summary>

**Abstract:** Current mental-health conversational systems are usually based on fixed, generic dialogue patterns. This paper proposes an adaptive framework based on large language models that aims to personalize therapeutic interaction according to a user's psychological state, quantified with the Acceptance of Illness Scale (AIS). The framework defines three specialized agents, L, M, and H, each linked to a different level of illness acceptance, and adjusts conversational behavior over time using continuous feedback signals. The AIS-stratified architecture is treated as a diegetic prototype placed in a plausible near-future setting and examined through the method of design fiction. By embedding the architecture in narrative scenarios, the study explores how such agents might influence access to care and therapeutic relationship. The goal is to show how clinically informed personalization, technical feasibility, and speculative scenario analysis can together inform the responsible design of LLM-based companions for mental-health support.

**arXiv ID:** 2511.20080
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (25 papers)</h2></summary>

<details>
<summary><strong>KOM: A Multi-Agent Artificial Intelligence System for Precision Management of Knee Osteoarthritis (KOA)</strong> - Weizhi Liu, Xi Chen, Zekun Jiang, Liang Zhao, Kunyuan Jiang, Ruisi Tang, Li Wang, Mingke You, Hanyu Zhou, Hongyu Chen, Qiankun Xiong, Yong Nie, Kang Li, Jian Li - [[pdf]](https://arxiv.org/pdf/2511.19798)</summary>

**Abstract:** Knee osteoarthritis (KOA) affects more than 600 million individuals globally and is associated with significant pain, functional impairment, and disability. While personalized multidisciplinary interventions have the potential to slow disease progression and enhance quality of life, they typically require substantial medical resources and expertise, making them difficult to implement in resource-limited settings. To address this challenge, we developed KOM, a multi-agent system designed to automate KOA evaluation, risk prediction, and treatment prescription. This system assists clinicians in performing essential tasks across the KOA care pathway and supports the generation of tailored management plans based on individual patient profiles, disease status, risk factors, and contraindications. In benchmark experiments, KOM demonstrated superior performance compared to several general-purpose large language models in imaging analysis and prescription generation. A randomized three-arm simulation study further revealed that collaboration between KOM and clinicians reduced total diagnostic and planning time by 38.5% and resulted in improved treatment quality compared to each approach used independently. These findings indicate that KOM could help facilitate automated KOA management and, when integrated into clinical workflows, has the potential to enhance care efficiency. The modular architecture of KOM may also offer valuable insights for developing AI-assisted management systems for other chronic conditions.

**arXiv ID:** 2511.19798
</details>

<details>
<summary><strong>M$^3$Prune: Hierarchical Communication Graph Pruning for Efficient Multi-Modal Multi-Agent Retrieval-Augmented Generation</strong> - Weizi Shao, Taolin Zhang, Zijie Zhou, Chen Chen, Chengyu Wang, Xiaofeng He - [[pdf]](https://arxiv.org/pdf/2511.19969)</summary>

**Abstract:** Recent advancements in multi-modal retrieval-augmented generation (mRAG), which enhance multi-modal large language models (MLLMs) with external knowledge, have demonstrated that the collective intelligence of multiple agents can significantly outperform a single model through effective communication. Despite impressive performance, existing multi-agent systems inherently incur substantial token overhead and increased computational costs, posing challenges for large-scale deployment. To address these issues, we propose a novel Multi-Modal Multi-agent hierarchical communication graph PRUNING framework, termed M$^3$Prune. Our framework eliminates redundant edges across different modalities, achieving an optimal balance between task performance and token overhead. Specifically, M$^3$Prune first applies intra-modal graph sparsification to textual and visual modalities, identifying the edges most critical for solving the task. Subsequently, we construct a dynamic communication topology using these key edges for inter-modal graph sparsification. Finally, we progressively prune redundant edges to obtain a more efficient and hierarchical topology. Extensive experiments on both general and domain-specific mRAG benchmarks demonstrate that our method consistently outperforms both single-agent and robust multi-agent mRAG systems while significantly reducing token consumption.

**arXiv ID:** 2511.19969
</details>

<details>
<summary><strong>DRAFT-RL: Multi-Agent Chain-of-Draft Reasoning for Reinforcement Learning-Enhanced LLMs</strong> - Yuanhao Li, Mingshan Liu, Hongbo Wang, Yiding Zhang, Yifei Ma, Wei Tan - [[pdf]](https://arxiv.org/pdf/2511.20468)</summary>

**Abstract:** Large Language Models (LLMs) have shown impressive capabilities in multi-step reasoning and this http URL works introduce multi-agent reflection frameworks where multiple LLM agents critique and refine each other's outputs using reinforcement learning (RL). However, these approaches often rely on single-shot responses and lack structural diversity in reasoning exploration. In this paper, we propose DRAFT-RL, a novel framework that integrates Chain-of-Draft (CoD) reasoning into multi-agent RL training. Instead of generating single responses, each agent produces multiple drafts per query, which are then evaluated by peer agents and a learned reward model to identify the most promising trajectory. These selected drafts are used to refine future reasoning strategies through actor-critic this http URL-RL enables explicit multi-path exploration, peer-guided reflection, and reward-aligned selection, resulting in more robust and interpretable LLM agent behavior. We evaluate our method on complex reasoning tasks including code synthesis, symbolic math, and knowledge-intensive QA,demonstrating that DRAFT-RL outperforms existing reflective and RL-based agents by significant margins in both accuracy and convergence speed

**arXiv ID:** 2511.20468
</details>

<details>
<summary><strong>Z-Space: A Multi-Agent Tool Orchestration Framework for Enterprise-Grade LLM Automation</strong> - Qingsong He, Jing Nan, Jiayu Jiao, Liangjie Tang, Xiaodong Xu, Mengmeng Sun, Qingyao Wang, Minghui Yan - [[pdf]](https://arxiv.org/pdf/2511.19483)</summary>

**Abstract:** Large Language Models can break through knowledge and timeliness limitations by invoking external tools within the Model Context Protocol framework to achieve automated execution of complex tasks. However, with the rapid growth of enterprise-scale MCP services, efficiently and accurately matching target functionalities among thousands of heterogeneous tools has become a core challenge restricting system practicality. Existing approaches generally rely on full-prompt injection or static semantic retrieval, facing issues including semantic disconnection between user queries and tool descriptions, context inflation in LLM input, and high inference latency. To address these challenges, this paper proposes Z-Space, a data-generation-oriented multi-agent collaborative tool invocation framework Z-Space. The Z-Space framework establishes a multi-agent collaborative architecture and tool filtering algorithm: (1) A structured semantic understanding of user queries is achieved through an intent parsing model; (2) A tool filtering module (FSWW) based on fused subspace weighted algorithm realizes fine-grained semantic alignment between intents and tools without parameter tuning; (3) An inference execution agent is constructed to support dynamic planning and fault-tolerant execution for multi-step tasks. This framework has been deployed in the Eleme platform's technical division, serving large-scale test data generation scenarios across multiple business units including Taotian, Gaode, and Hema. Production data demonstrates that the system reduces average token consumption in tool inference by 96.26\% while achieving a 92\% tool invocation accuracy rate, significantly enhancing the efficiency and reliability of intelligent test data generation systems.

**arXiv ID:** 2511.19483
</details>

<details>
<summary><strong>AttackPilot: Autonomous Inference Attacks Against ML Services With LLM-Based Agents</strong> - Yixin Wu, Rui Wen, Chi Cui, Michael Backes, Yang Zhang - [[pdf]](https://arxiv.org/pdf/2511.19536)</summary>

**Abstract:** Inference attacks have been widely studied and offer a systematic risk assessment of ML services; however, their implementation and the attack parameters for optimal estimation are challenging for non-experts. The emergence of advanced large language models presents a promising yet largely unexplored opportunity to develop autonomous agents as inference attack experts, helping address this challenge. In this paper, we propose AttackPilot, an autonomous agent capable of independently conducting inference attacks without human intervention. We evaluate it on 20 target services. The evaluation shows that our agent, using GPT-4o, achieves a 100.0% task completion rate and near-expert attack performance, with an average token cost of only $0.627 per run. The agent can also be powered by many other representative LLMs and can adaptively optimize its strategy under service constraints. We further perform trace analysis, demonstrating that design choices, such as a multi-agent framework and task-specific action spaces, effectively mitigate errors such as bad plans, inability to follow instructions, task context loss, and hallucinations. We anticipate that such agents could empower non-expert ML service providers, auditors, or regulators to systematically assess the risks of ML services without requiring deep domain expertise.

**arXiv ID:** 2511.19536
</details>

<details>
<summary><strong>Trust-Based Social Learning for Communication (TSLEC) Protocol Evolution in Multi-Agent Reinforcement Learning</strong> - Abraham Itzhak Weinberg - [[pdf]](https://arxiv.org/pdf/2511.19562)</summary>

**Abstract:** Emergent communication in multi-agent systems typically occurs through independent learning, resulting in slow convergence and potentially suboptimal protocols. We introduce TSLEC (Trust-Based Social Learning with Emergent Communication), a framework where agents explicitly teach successful strategies to peers, with knowledge transfer modulated by learned trust relationships. Through experiments with 100 episodes across 30 random seeds, we demonstrate that trust-based social learning reduces episodes-to-convergence by 23.9% (p < 0.001, Cohen's d = 1.98) compared to independent emergence, while producing compositional protocols (C = 0.38) that remain robust under dynamic objectives (Phi > 0.867 decoding accuracy). Trust scores strongly correlate with teaching quality (r = 0.743, p < 0.001), enabling effective knowledge filtering. Our results establish that explicit social learning fundamentally accelerates emergent communication in multi-agent coordination.

**arXiv ID:** 2511.19562
</details>

<details>
<summary><strong>A Layered Protocol Architecture for the Internet of Agents</strong> - Charles Fleming, Vijoy Pandey, Ramana Kompella, Luca Muscariello - [[pdf]](https://arxiv.org/pdf/2511.19699)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated remarkable performance improvements and the ability to learn domain-specific languages (DSLs), including APIs and tool interfaces. This capability has enabled the creation of AI agents that can perform preliminary computations and act through tool calling, now being standardized via protocols like MCP. However, LLMs face fundamental limitations: their context windows cannot grow indefinitely, constraining their memory and computational capacity. Agent collaboration emerges as essential for solving increasingly complex problems, mirroring how computational systems rely on different types of memory to scale. The "Internet of Agents" (IoA) represents the communication stack that enables agents to scale by distributing computation across collaborating entities.
Current network architectural stacks (OSI and TCP/IP) were designed for data delivery between hosts and processes, not for agent collaboration with semantic understanding. To address this gap, we propose two new layers: an \textbf{Agent Communication Layer (L8)} and an \textbf{Agent Semantic Negotiation Layer (L9)}. L8 formalizes the \textit{structure} of communication, standardizing message envelopes, speech-act performatives (e.g., REQUEST, INFORM), and interaction patterns (e.g., request-reply, publish-subscribe), building on protocols like MCP. L9, which does not exist today, formalizes the \textit{meaning} of communication, enabling agents to discover, negotiate, and lock a "Shared Context" -- a formal schema defining the concepts, tasks, and parameters relevant to their interaction. Together, these layers provide the foundation for scalable, distributed agent collaboration, enabling the next generation of multi-agentic systems.

**arXiv ID:** 2511.19699
</details>

<details>
<summary><strong>An Adaptive, Data-Integrated Agent-Based Modeling Framework for Explainable and Contestable Policy Design</strong> - Roberto Garrone - [[pdf]](https://arxiv.org/pdf/2511.19726)</summary>

**Abstract:** Multi-agent systems often operate under feedback, adaptation, and non-stationarity, yet many simulation studies retain static decision rules and fixed control parameters. This paper introduces a general adaptive multi-agent learning framework that integrates: (i) four dynamic regimes distinguishing static versus adaptive agents and fixed versus adaptive system parameters; (ii) information-theoretic diagnostics (entropy rate, statistical complexity, and predictive information) to assess predictability and structure; (iii) structural causal models for explicit intervention semantics; (iv) procedures for generating agent-level priors from aggregate or sample data; and (v) unsupervised methods for identifying emergent behavioral regimes. The framework offers a domain-neutral architecture for analyzing how learning agents and adaptive controls jointly shape system trajectories, enabling systematic comparison of stability, performance, and interpretability across non-equilibrium, oscillatory, or drifting dynamics. Mathematical definitions, computational operators, and an experimental design template are provided, yielding a structured methodology for developing explainable and contestable multi-agent decision processes.

**arXiv ID:** 2511.19726
</details>

<details>
<summary><strong>R3A: Reliable RTL Repair Framework with Multi-Agent Fault Localization and Stochastic Tree-of-Thoughts Patch Generation</strong> - Zizhang Luo, Fan Cui, Kexing Zhou, Runlin Guo, Mile Xia, Hongyuan Hou, Yun Lian - [[pdf]](https://arxiv.org/pdf/2511.20090)</summary>

**Abstract:** Repairing RTL bugs is crucial for hardware design and verification. Traditional automatic program repair (APR) methods define dedicated search spaces to locate and fix bugs with program synthesis. However, they heavily rely on fixed templates and can only deal with limited bugs. As an alternative, Large Language Models with the ability to understand code semantics can be explored for RTL repair. However, they suffer from unreliable outcomes due to inherent randomness and long input contexts of RTL code and waveform. To address these challenges, we propose R3A, an LLM-based automatic RTL program repair framework upon the basic model to improve reliability. R3A proposes the stochastic Tree-Of-Thoughts method to control a patch generation agent to explore a validated solution for the bug. The algorithm samples search states according to a heuristic function to balance between exploration and exploitation for a reliable outcome. Besides, R3A proposes a multi-agent fault localization method to find fault candidates as the starting points for the patch generation agent, further increasing the reliability. Experiments show R3A can fix 90.6% of bugs in the RTL-repair dataset within a given time limit, which covers 45% more bugs than traditional methods and other LLM-based approaches, while achieving an 86.7% pass@5 rate on average, showing a high reliability.

**arXiv ID:** 2511.20090
</details>

<details>
<summary><strong>EnergyTwin: A Multi-Agent System for Simulating and Coordinating Energy Microgrids</strong> - Jakub Muszyński, Ignacy Walużenicz, Patryk Zan, Zofia Wrona, Maria Ganzha, Marcin Paprzycki, Costin Bădică - [[pdf]](https://arxiv.org/pdf/2511.20590)</summary>

**Abstract:** Microgrids are deployed to reduce purchased grid energy, limit exposure to volatile tariffs, and ensure service continuity during disturbances. This requires coordinating heterogeneous distributed energy resources across multiple time scales and under variable conditions. Among existing tools, typically, power-system simulators capture physical behaviour but assume centralized control, while multi-agent frameworks model decentralized decision-making but represent energy with no physical grounding. In this context, the EnergyTwin is introduced, an agent-based microgrid simulation environment that couples physically grounded models with forecast-informed, rolling-horizon planning, and negotiations. Each asset is modeled as an agent, interacting with a central agent that obtains forecasts, formulates predictions, and allocates energy through contract-based interactions. EnergyTwin targets tertiary-layer decision making and is extensible for digital-twin use. Its feasibility was evaluated in a university campus microgrid scenario where multiple planning strategies were compared. Achieved results show that forecast-driven rolling-horizon planning increases local energy self-sufficiency, maintains higher battery reserves, and reduces exposure to low-resilience operating states. They demonstrate also potential of EnergyTwin as platform supporting research on resilient, negotiation-driven microgrids.

**arXiv ID:** 2511.20590
</details>

<details>
<summary><strong>Latent Collaboration in Multi-Agent Systems</strong> - Jiaru Zou, Xiyuan Yang, Ruizhong Qiu, Gaotang Li, Katherine Tieu, Pan Lu, Ke Shen, Hanghang Tong, Yejin Choi, Jingrui He, James Zou, Mengdi Wang, Ling Yang - [[pdf]](https://arxiv.org/pdf/2511.20639)</summary>

**Abstract:** Multi-agent systems (MAS) extend large language models (LLMs) from independent single-model reasoning to coordinative system-level intelligence. While existing LLM agents depend on text-based mediation for reasoning and communication, we take a step forward by enabling models to collaborate directly within the continuous latent space. We introduce LatentMAS, an end-to-end training-free framework that enables pure latent collaboration among LLM agents. In LatentMAS, each agent first performs auto-regressive latent thoughts generation through last-layer hidden embeddings. A shared latent working memory then preserves and transfers each agent's internal representations, ensuring lossless information exchange. We provide theoretical analyses establishing that LatentMAS attains higher expressiveness and lossless information preservation with substantially lower complexity than vanilla text-based MAS. In addition, empirical evaluations across 9 comprehensive benchmarks spanning math and science reasoning, commonsense understanding, and code generation show that LatentMAS consistently outperforms strong single-model and text-based MAS baselines, achieving up to 14.6% higher accuracy, reducing output token usage by 70.8%-83.7%, and providing 4x-4.3x faster end-to-end inference. These results demonstrate that our new latent collaboration framework enhances system-level reasoning quality while offering substantial efficiency gains without any additional training. Code and data are fully open-sourced at this https URL.

**arXiv ID:** 2511.20639
</details>

<details>
<summary><strong>Scalable and Accurate Graph Reasoning with LLM-based Multi-Agents</strong> - Yuwei Hu, Runlin Lei, Xinyi Huang, Zhewei Wei, Yongchao Liu - [[pdf]](https://arxiv.org/pdf/2410.05130)</summary>

**Abstract:** Recent research has explored the use of Large Language Models (LLMs) for tackling complex graph reasoning tasks. However, due to the intricacies of graph structures and the inherent limitations of LLMs in handling long text, current approaches often fail to deliver satisfactory accuracy, even on small-scale graphs and simple tasks. To address these challenges, we introduce GraphAgent-Reasoner, a fine-tuning-free framework that utilizes a multi-agent collaboration strategy for explicit and precise graph reasoning. Inspired by distributed graph computation theory, our framework decomposes graph problems into smaller, node-centric tasks that are distributed among multiple agents. The agents collaborate to solve the overall problem, significantly reducing the amount of information and complexity handled by a single LLM, thus enhancing the accuracy of graph reasoning. By simply increasing the number of agents, GraphAgent-Reasoner can efficiently scale to accommodate larger graphs with over 1,000 nodes. Evaluated on the GraphInstruct dataset, our framework demonstrates near-perfect accuracy on polynomial-time graph reasoning tasks, significantly outperforming the best available models, both closed-source and fine-tuned open-source variants. Our framework also demonstrates the capability to handle real-world graph reasoning applications such as webpage importance analysis.

**arXiv ID:** 2410.05130
</details>

<details>
<summary><strong>LLM Collaboration With Multi-Agent Reinforcement Learning</strong> - Shuo Liu, Tianle Chen, Zeyu Liang, Xueguang Lyu, Christopher Amato - [[pdf]](https://arxiv.org/pdf/2508.04652)</summary>

**Abstract:** A large amount of work has been done in Multi-Agent Systems (MAS) for modeling and solving problems with multiple interacting agents. However, most LLMs are pretrained independently and not specifically optimized for coordination. Existing LLM fine-tuning frameworks rely on individual rewards, which require complex reward designs for each agent to encourage collaboration. To address these challenges, we model LLM collaboration as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. We develop a multi-agent, multi-turn algorithm, Multi-Agent Group Relative Policy Optimization (MAGRPO), to solve it, building on current RL approaches for LLMs as well as MARL techniques. Our experiments on LLM writing and coding collaboration demonstrate that fine-tuning MAS with MAGRPO enables agents to generate high-quality responses efficiently through effective cooperation. Our approach opens the door to using other MARL methods for LLMs and highlights the associated challenges. Our code is available at this https URL.

**arXiv ID:** 2508.04652
</details>

<details>
<summary><strong>Heterogeneous Multi-Agent Proximal Policy Optimization for Power Distribution System Restoration</strong> - Parya Dolatyabi, Mahdi Khodayar - [[pdf]](https://arxiv.org/pdf/2511.14730)</summary>

**Abstract:** Restoring power distribution systems (PDS) after large-scale outages requires sequential switching operations that reconfigure feeder topology and coordinate distributed energy resources (DERs) under nonlinear constraints such as power balance, voltage limits, and thermal ratings. These challenges make conventional optimization and value-based RL approaches computationally inefficient and difficult to scale. This paper applies a Heterogeneous-Agent Reinforcement Learning (HARL) framework, instantiated through Heterogeneous-Agent Proximal Policy Optimization (HAPPO), to enable coordinated restoration across interconnected microgrids. Each agent controls a distinct microgrid with different loads, DER capacities, and switch counts, introducing practical structural heterogeneity. Decentralized actor policies are trained with a centralized critic to compute advantage values for stable on-policy updates. A physics-informed OpenDSS environment provides full power flow feedback and enforces operational limits via differentiable penalty signals rather than invalid action masking. The total DER generation is capped at 2400 kW, and each microgrid must satisfy local supply-demand feasibility. Experiments on the IEEE 123-bus and IEEE 8500-node systems show that HAPPO achieves faster convergence, higher restored power, and smoother multi-seed training than DQN, PPO, MAES, MAGDPG, MADQN, Mean-Field RL, and QMIX. Results demonstrate that incorporating microgrid-level heterogeneity within the HARL framework yields a scalable, stable, and constraint-aware solution for complex PDS restoration.

**arXiv ID:** 2511.14730
</details>

<details>
<summary><strong>VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning</strong> - Boyu Chen, Zikang Wang, Zhengrong Yue, Kainan Yan, Chenyun Yu, Yi Huang, Zijun Liu, Yafei Wen, Xiaoxin Chen, Yang Liu, Peng Li, Yali Wang - [[pdf]](https://arxiv.org/pdf/2511.19524)</summary>

**Abstract:** By leveraging tool-augmented Multimodal Large Language Models (MLLMs), multi-agent frameworks are driving progress in video understanding. However, most of them adopt static and non-learnable tool invocation mechanisms, which limit the discovery of diverse clues essential for robust perception and reasoning regarding temporally or spatially complex videos. To address this challenge, we propose a novel Multi-agent system for video understanding, namely VideoChat-M1. Instead of using a single or fixed policy, VideoChat-M1 adopts a distinct Collaborative Policy Planning (CPP) paradigm with multiple policy agents, which comprises three key processes. (1) Policy Generation: Each agent generates its unique tool invocation policy tailored to the user's query; (2) Policy Execution: Each agent sequentially invokes relevant tools to execute its policy and explore the video content; (3) Policy Communication: During the intermediate stages of policy execution, agents interact with one another to update their respective policies. Through this collaborative framework, all agents work in tandem, dynamically refining their preferred policies based on contextual insights from peers to effectively respond to the user's query. Moreover, we equip our CPP paradigm with a concise Multi-Agent Reinforcement Learning (MARL) method. Consequently, the team of policy agents can be jointly optimized to enhance VideoChat-M1's performance, guided by both the final answer reward and intermediate collaborative process feedback. Extensive experiments demonstrate that VideoChat-M1 achieves SOTA performance across eight benchmarks spanning four tasks. Notably, on LongVideoBench, our method outperforms the SOTA model Gemini 2.5 pro by 3.6% and GPT-4o by 15.6%.

**arXiv ID:** 2511.19524
</details>

<details>
<summary><strong>RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation with Radiologist-like Workflows</strong> - Kai Zhang, Corey D Barrett, Jangwon Kim, Lichao Sun, Tara Taghavi, Krishnaram Kenthapadi - [[pdf]](https://arxiv.org/pdf/2509.20490)</summary>

**Abstract:** Agentic systems offer a potential path to solve complex clinical tasks through collaboration among specialized agents, augmented by tool use and external knowledge bases. Nevertheless, for chest X-ray (CXR) interpretation, prevailing methods remain limited: (i) reasoning is frequently neither clinically interpretable nor aligned with guidelines, reflecting mere aggregation of tool outputs; (ii) multimodal evidence is insufficiently fused, yielding text-only rationales that are not visually grounded; and (iii) systems rarely detect or resolve cross-tool inconsistencies and provide no principled verification mechanisms. To bridge the above gaps, we present RadAgents, a multi-agent framework that couples clinical priors with task-aware multimodal reasoning and encodes a radiologist-style workflow into a modular, auditable pipeline. In addition, we integrate grounding and multimodal retrieval-augmentation to verify and resolve context conflicts, resulting in outputs that are more reliable, transparent, and consistent with clinical practice.

**arXiv ID:** 2509.20490
</details>

<details>
<summary><strong>VIL2C: Value-of-Information Aware Low-Latency Communication for Multi-Agent Reinforcement Learning</strong> - Qian Zhang, Zhuo Sun, Yao Zhang, Zhiwen Yu, Bin Guo, Jun Zhang - [[pdf]](https://arxiv.org/pdf/2511.19146)</summary>

**Abstract:** Inter-agent communication serves as an effective mechanism for enhancing performance in collaborative multi-agent reinforcement learning(MARL) systems. However, the inherent communication latency in practical systems induces both action decision delays and outdated information sharing, impeding MARL performance gains, particularly in time-critical applications like autonomous driving. In this work, we propose a Value-of-Information aware Low-latency Communication(VIL2C) scheme that proactively adjusts the latency distribution to mitigate its effects in MARL systems. Specifically, we define a Value of Information (VOI) metric to quantify the importance of delayed message transmission based on each delayed message's importance. Moreover, we propose a progressive message reception mechanism to adaptively adjust the reception duration based on received messages. We derive the optimized VoI aware resource allocation and theoretically prove the performance advantage of the proposed VIL2C scheme. Extensive experiments demonstrate that VIL2C outperforms existing approaches under various communication conditions. These gains are attributed to the low-latency transmission of high-VoI messages via resource allocation and the elimination of unnecessary waiting periods via adaptive reception duration.

**arXiv ID:** 2511.19146
</details>

<details>
<summary><strong>LiRA: A Multi-Agent Framework for Reliable and Readable Literature Review Generation</strong> - Gregory Hok Tjoan Go, Khang Ly, Anders Søgaard, Amin Tabatabaei, Maarten de Rijke, Xinyi Chen - [[pdf]](https://arxiv.org/pdf/2510.05138)</summary>

**Abstract:** The rapid growth of scientific publications has made it increasingly difficult to keep literature reviews comprehensive and up-to-date. Though prior work has focused on automating retrieval and screening, the writing phase of systematic reviews remains largely under-explored, especially with regard to readability and factual accuracy. To address this, we present LiRA (Literature Review Agents), a multi-agent collaborative workflow which emulates the human literature review process. LiRA utilizes specialized agents for content outlining, subsection writing, editing, and reviewing, producing cohesive and comprehensive review articles. Evaluated on SciReviewGen and a proprietary ScienceDirect dataset, LiRA outperforms current baselines such as AutoSurvey and MASS-Survey in writing and citation quality, while maintaining competitive similarity to human-written reviews. We further evaluate LiRA in real-world scenarios using document retrieval and assess its robustness to reviewer model variation. Our findings highlight the potential of agentic LLM workflows, even without domain-specific tuning, to improve the reliability and usability of automated scientific writing.

**arXiv ID:** 2510.05138
</details>

<details>
<summary><strong>CLIMATEAGENT: Multi-Agent Orchestration for Complex Climate Data Science Workflows</strong> - Hyeonjae Kim, Chenyue Li, Wen Deng, Mengxi Jin, Wen Huang, Mengqian Lu, Binhang Yuan - [[pdf]](https://arxiv.org/pdf/2511.20109)</summary>

**Abstract:** Climate science demands automated workflows to transform comprehensive questions into data-driven statements across massive, heterogeneous datasets. However, generic LLM agents and static scripting pipelines lack climate-specific context and flexibility, thus, perform poorly in practice. We present ClimateAgent, an autonomous multi-agent framework that orchestrates end-to-end climate data analytic workflows. ClimateAgent decomposes user questions into executable sub-tasks coordinated by an Orchestrate-Agent and a Plan-Agent; acquires data via specialized Data-Agents that dynamically introspect APIs to synthesize robust download scripts; and completes analysis and reporting with a Coding-Agent that generates Python code, visualizations, and a final report with a built-in self-correction loop. To enable systematic evaluation, we introduce Climate-Agent-Bench-85, a benchmark of 85 real-world tasks spanning atmospheric rivers, drought, extreme precipitation, heat waves, sea surface temperature, and tropical cyclones. On Climate-Agent-Bench-85, ClimateAgent achieves 100% task completion and a report quality score of 8.32, outperforming GitHub-Copilot (6.27) and a GPT-5 baseline (3.26). These results demonstrate that our multi-agent orchestration with dynamic API awareness and self-correcting execution substantially advances reliable, end-to-end automation for climate science analytic tasks.

**arXiv ID:** 2511.20109
</details>

<details>
<summary><strong>Designing Reputation Systems for Manufacturing Data Trading Markets: A Multi-Agent Evaluation with Q-Learning and IRL-Estimated Utilities</strong> - Kenta Yamamoto, Teruaki Hayashi - [[pdf]](https://arxiv.org/pdf/2511.19930)</summary>

**Abstract:** Recent advances in machine learning and big data analytics have intensified the demand for high-quality cross-domain datasets and accelerated the growth of data trading across organizations. As data become increasingly recognized as an economic asset, data marketplaces have emerged as a key infrastructure for data-driven innovation. However, unlike mature product or service markets, data-trading environments remain nascent and suffer from pronounced information asymmetry. Buyers cannot verify the content or quality before purchasing data, making trust and quality assurance central challenges. To address these issues, this study develops a multi-agent data-market simulator that models participant behavior and evaluates the institutional mechanisms for trust formation. Focusing on the manufacturing sector, where initiatives such as GAIA-X and Catena-X are advancing, the simulator integrates reinforcement learning (RL) for adaptive agent behavior and inverse reinforcement learning (IRL) to estimate utility functions from empirical behavioral data. Using the simulator, we examine the market-level effects of five representative reputation systems-Time-decay, Bayesian-beta, PageRank, PowerTrust, and PeerTrust-and found that PeerTrust achieved the strongest alignment between data price and quality, while preventing monopolistic dominance. Building on these results, we develop a hybrid reputation mechanism that integrates the strengths of existing systems to achieve improved price-quality consistency and overall market stability. This study extends simulation-based data-market analysis by incorporating trust and reputation as endogenous mechanisms and offering methodological and institutional insights into the design of reliable and efficient data ecosystems.

**arXiv ID:** 2511.19930
</details>

<details>
<summary><strong>On Feasible Rewards in Multi-Agent Inverse Reinforcement Learning</strong> - Till Freihaut, Giorgia Ramponi - [[pdf]](https://arxiv.org/pdf/2411.15046)</summary>

**Abstract:** Multi-agent Inverse Reinforcement Learning (MAIRL) aims to recover agent reward functions from expert demonstrations. We characterize the feasible reward set in Markov games, identifying all reward functions that rationalize a given equilibrium. However, equilibrium-based observations are often ambiguous: a single Nash equilibrium can correspond to many reward structures, potentially changing the game's nature in multi-agent systems. We address this by introducing entropy-regularized Markov games, which yield a unique equilibrium while preserving strategic incentives. For this setting, we provide a sample complexity analysis detailing how errors affect learned policy performance. Our work establishes theoretical foundations and practical insights for MAIRL.

**arXiv ID:** 2411.15046
</details>

<details>
<summary><strong>ARBoids: Adaptive Residual Reinforcement Learning With Boids Model for Cooperative Multi-USV Target Defense</strong> - Jiyue Tao, Tongsheng Shen, Dexin Zhao, Feitian Zhang - [[pdf]](https://arxiv.org/pdf/2502.18549)</summary>

**Abstract:** The target defense problem (TDP) for unmanned surface vehicles (USVs) concerns intercepting an adversarial USV before it breaches a designated target region, using one or more defending USVs. A particularly challenging scenario arises when the attacker exhibits superior maneuverability compared to the defenders, significantly complicating effective interception. To tackle this challenge, this letter introduces ARBoids, a novel adaptive residual reinforcement learning framework that integrates deep reinforcement learning (DRL) with the biologically inspired, force-based Boids model. Within this framework, the Boids model serves as a computationally efficient baseline policy for multi-agent coordination, while DRL learns a residual policy to adaptively refine and optimize the defenders' actions. The proposed approach is validated in a high-fidelity Gazebo simulation environment, demonstrating superior performance over traditional interception strategies, including pure force-based approaches and vanilla DRL policies. Furthermore, the learned policy exhibits strong adaptability to attackers with diverse maneuverability profiles, highlighting its robustness and generalization capability. The code of ARBoids will be released upon acceptance of this letter.

**arXiv ID:** 2502.18549
</details>

<details>
<summary><strong>AirFed: A Federated Graph-Enhanced Multi-Agent Reinforcement Learning Framework for Multi-UAV Cooperative Mobile Edge Computing</strong> - Zhiyu Wang, Suman Raj, Rajkumar Buyya - [[pdf]](https://arxiv.org/pdf/2510.23053)</summary>

**Abstract:** Multiple Unmanned Aerial Vehicles (UAVs) cooperative Mobile Edge Computing (MEC) systems face critical challenges in coordinating trajectory planning, task offloading, and resource allocation while ensuring Quality of Service (QoS) under dynamic and uncertain environments. Existing approaches suffer from limited scalability, slow convergence, and inefficient knowledge sharing among UAVs, particularly when handling large-scale IoT device deployments with stringent deadline constraints. This paper proposes AirFed, a novel federated graph-enhanced multi-agent reinforcement learning framework that addresses these challenges through three key innovations. First, we design dual-layer dynamic Graph Attention Networks (GATs) that explicitly model spatial-temporal dependencies among UAVs and IoT devices, capturing both service relationships and collaborative interactions within the network topology. Second, we develop a dual-Actor single-Critic architecture that jointly optimizes continuous trajectory control and discrete task offloading decisions. Third, we propose a reputation-based decentralized federated learning mechanism with gradient-sensitive adaptive quantization, enabling efficient and robust knowledge sharing across heterogeneous UAVs. Extensive experiments demonstrate that AirFed achieves 42.9% reduction in weighted cost compared to state-of-the-art baselines, attains over 99% deadline satisfaction and 94.2% IoT device coverage rate, and reduces communication overhead by 54.5%. Scalability analysis confirms robust performance across varying UAV numbers, IoT device densities, and system scales, validating AirFed's practical applicability for large-scale UAV-MEC deployments.

**arXiv ID:** 2510.23053
</details>

<details>
<summary><strong>Multi-Agent gatekeeper: Safe Flight Planning and Formation Control for Urban Air Mobility</strong> - Thomas Marshall Vielmetti, Devansh R Agrawal, Dimitra Panagou - [[pdf]](https://arxiv.org/pdf/2511.19691)</summary>

**Abstract:** We present Multi-Agent gatekeeper, a framework that provides provable safety guarantees for leader-follower formation control in cluttered 3D environments. Existing methods face a trad-off: online planners and controllers lack formal safety guarantees, while offline planners lack adaptability to changes in the number of agents or desired formation. To address this gap, we propose a hybrid architecture where a single leader tracks a pre-computed, safe trajectory, which serves as a shared trajectory backup set for all follower agents. Followers execute a nominal formation-keeping tracking controller, and are guaranteed to remain safe by always possessing a known-safe backup maneuver along the leader's path. We formally prove this method ensures collision avoidance with both static obstacles and other agents. The primary contributions are: (1) the multi-agent gatekeeper algorithm, which extends our single-agent gatekeeper framework to multi-agent systems; (2) the trajectory backup set for provably safe inter-agent coordination for leader-follower formation control; and (3) the first application of the gatekeeper framework in a 3D environment. We demonstrate our approach in a simulated 3D urban environment, where it achieved a 100% collision-avoidance success rate across 100 randomized trials, significantly outperforming baseline CBF and NMPC methods. Finally, we demonstrate the physical feasibility of the resulting trajectories on a team of quadcopters.

**arXiv ID:** 2511.19691
</details>

<details>
<summary><strong>Disentangled Control of Multi-Agent Systems</strong> - Ruoyu Lin, Gennaro Notomista, Magnus Egerstedt - [[pdf]](https://arxiv.org/pdf/2511.05900)</summary>

**Abstract:** This paper develops a general framework for multi-agent control synthesis, which applies to a wide range of problems with convergence guarantees, regardless of the complexity of the underlying graph topology and the explicit time dependence of the objective function. The proposed framework systematically addresses a particularly challenging problem in multi-agent systems, i.e., decentralization of entangled dynamics among different agents, and it naturally supports multi-objective robotics and real-time implementations. To demonstrate its generality and effectiveness, the framework is implemented across three experiments, namely time-varying leader-follower formation control, decentralized coverage control for time-varying density functions without any approximations, which is a long-standing open problem, and safe formation navigation in dense environments.

**arXiv ID:** 2511.05900
</details>

</details>

<details open>
<summary><h2>Other Agent Research (6 papers)</h2></summary>

<details>
<summary><strong>Reducing Latency of LLM Search Agent via Speculation-based Algorithm-System Co-Design</strong> - Zixiao Huang, Wen Zeng, Tianyu Fu, Tengxuan Liu, Yizhou Sun, Ke Hong, Xinhao Yang, Chengchun Liu, Yan Li, Quanlu Zhang, Guohao Dai, Zhenhua Zhu, Yu Wang - [[pdf]](https://arxiv.org/pdf/2511.20048)</summary>

**Abstract:** LLM-based search agents achieve strong performance but suffer from severe latency, as each step requires serialized LLM reasoning followed by action of tool execution. We revisit this bottleneck through the lens of speculation. While traditional predict-verify speculation paradigm can break serial execution, its benefit remains limited, as it retains the full original workload and adds extra inference overhead. We observe that early agent steps often involve simple evidence-gathering, where correct actions can often be predicted without full reasoning. Building on these observations, we present SPAgent, an algorithm-system co-design framework that expands the role of speculation in search agents to reduce latency. Algorithmically, SPAgent introduces a two-phase adaptive speculation mechanism that selectively omits verification when safe. System-wise, a two-level scheduler regulates speculative requests based on engine load to ensure speculation remains beneficial. We implement SPAgent in real-world systems. Across extensive experimental settings, SPAgent achieves up to $1.65\times$ end-to-end speedup while maintaining same or even achieving higher accuracy, enabling practical deployment of multi-step search agents.

**arXiv ID:** 2511.20048
</details>

<details>
<summary><strong>Improved Linear-Time Construction of Minimal Dominating Set via Mobile Agents</strong> - Prabhat Kumar Chand, Anisur Rahaman Molla - [[pdf]](https://arxiv.org/pdf/2511.19880)</summary>

**Abstract:** Mobile agents have emerged as a powerful framework for solving fundamental graph problems in distributed settings in recent times. These agents, modelled as autonomous physical or software entities, possess local computation power, finite memory and have the ability to traverse a graph, offering efficient solutions to a range of classical problems. In this work, we focus on the problem of computing a \emph{minimal dominating set} (mDS) in anonymous graphs using mobile agents. Building on the recently proposed optimal dispersion algorithm on the synchronous mobile agent model, we design two new algorithms that achieve a \emph{linear-time} solution for this problem in the synchronous setting. Specifically, given a connected $n$-node graph with $n$ agents initially placed in either rooted or arbitrary configurations, we show that an mDS can be computed in $O(n)$ rounds using only $O(\log n)$ bits of memory per agent, without using any prior knowledge of any global parameters. This improves upon the best-known complexity results in the literature over the same model. In addition, as natural by-products of our methodology, our algorithms also construct a spanning tree and elect a unique leader in $O(n)$ rounds, which are also important results of independent interest in the mobile-agent framework.

**arXiv ID:** 2511.19880
</details>

<details>
<summary><strong>Solving Heterogeneous Agent Models with Physics-informed Neural Networks</strong> - Marta Grzeskiewicz - [[pdf]](https://arxiv.org/pdf/2511.20283)</summary>

**Abstract:** Understanding household behaviour is essential for modelling macroeconomic dynamics and designing effective policy. While heterogeneous agent models offer a more realistic alternative to representative agent frameworks, their implementation poses significant computational challenges, particularly in continuous time. The Aiyagari-Bewley-Huggett (ABH) framework, recast as a system of partial differential equations, typically relies on grid-based solvers that suffer from the curse of dimensionality, high computational cost, and numerical inaccuracies. This paper introduces the ABH-PINN solver, an approach based on Physics-Informed Neural Networks (PINNs), which embeds the Hamilton-Jacobi-Bellman and Kolmogorov Forward equations directly into the neural network training objective. By replacing grid-based approximation with mesh-free, differentiable function learning, the ABH-PINN solver benefits from the advantages of PINNs of improved scalability, smoother solutions, and computational efficiency. Preliminary results show that the PINN-based approach is able to obtain economically valid results matching the established finite-difference solvers.

**arXiv ID:** 2511.20283
</details>

<details>
<summary><strong>Human-Centered Cooperative Control Coupling Autonomous and Haptic Shared Control via Control Barrier Function</strong> - Eito Sato, Takahiro Wada - [[pdf]](https://arxiv.org/pdf/2511.19869)</summary>

**Abstract:** Haptic shared control (HSC) is effective in teleoperation when full autonomy is limited by uncertainty or sensing constraints. However, autonomous control performance achieved by maximizing HSC strength is limited because the dynamics of the joystick and human arm affect the robot's behavior. We propose a cooperative framework coupling a joystick-independent autonomous controller with HSC. A control barrier function ignores joystick inputs within a safe region determined by the human operator in real-time, while HSC is engaged otherwise. A pilot experiment on simulated tasks with tele-operated underwater robot in virtual environment demonstrated improved accuracy and reduced required time over conventional HSC.

**arXiv ID:** 2511.19869
</details>

<details>
<summary><strong>Nauplius Optimisation for Autonomous Hydrodynamics</strong> - Shyalan Ramesh, Scott Mann, Alex Stumpf - [[pdf]](https://arxiv.org/pdf/2510.15350)</summary>

**Abstract:** Autonomous Underwater vehicles must operate in strong currents, limited acoustic bandwidth, and persistent sensing requirements where conventional swarm optimisation methods are unreliable. This paper formulates an irreversible hydrodynamic deployment problem for Autonomous Underwater Vehicle (AUV) swarms and presents Nauplius Optimisation for Autonomous Hydrodynamics (NOAH), a novel nature-inspired swarm optimisation algorithm that combines current-aware drift, irreversible settlement in persistent sensing nodes, and colony- based communication. Drawing inspiration from the behaviour of barnacle nauplii, NOAH addresses the critical limitations of existing swarm algorithms by providing hydrodynamic awareness, irreversible anchoring mechanisms, and colony-based communication capabilities essential for underwater exploration missions. The algorithm establishes a comprehensive foundation for scalable and energy-efficient underwater swarm robotics with validated performance analysis. Validation studies demonstrate an 86% success rate for permanent anchoring scenarios, providing a unified formulation for hydrodynamic constraints and irreversible settlement behaviours with an empirical study under flow.

**arXiv ID:** 2510.15350
</details>

<details>
<summary><strong>SVBRD-LLM: Self-Verifying Behavioral Rule Discovery for Autonomous Vehicle Identification</strong> - Xiangyu Li, Zhaomiao Guo - [[pdf]](https://arxiv.org/pdf/2511.14977)</summary>

**Abstract:** As more autonomous vehicles operate on public roads, understanding real-world behavior of autonomous vehicles is critical to analyzing traffic safety, making policies, and public acceptance. This paper proposes SVBRD-LLM, a framework that automatically discovers, verifies, and applies interpretable behavioral rules from real traffic videos through zero-shot prompt engineering. The framework extracts vehicle trajectories using YOLOv8 and ByteTrack, computes kinematic features, and employs GPT-5 zero-shot prompting to compare autonomous and human-driven vehicles, generating 35 structured behavioral rule hypotheses. These rules are tested on a validation set, iteratively refined based on failure cases to filter spurious correlations, and compiled into a high-confidence rule library. The framework is evaluated on an independent test set for speed change prediction, lane change prediction, and autonomous vehicle identification tasks. Experiments on over 1500 hours of real traffic videos show that the framework achieves 90.0% accuracy and 93.3% F1-score in autonomous vehicle identification. The discovered rules clearly reveal distinctive characteristics of autonomous vehicles in speed control smoothness, lane change conservativeness, and acceleration stability, with each rule accompanied by semantic description, applicable context, and validation confidence.

**arXiv ID:** 2511.14977
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (21 papers)</h2></summary>

<details>
<summary><strong>Scaling Agentic Reinforcement Learning for Tool-Integrated Reasoning in VLMs</strong> - Meng Lu, Ran Xu, Yi Fang, Wenxuan Zhang, Yue Yu, Gaurav Srivastava, Yuchen Zhuang, Mohamed Elhoseiny, Charles Fleming, Carl Yang, Zhengzhong Tu, Yang Xie, Guanghua Xiao, Hanrui Wang, Di Jin, Wenqi Shi, Xuan Wang - [[pdf]](https://arxiv.org/pdf/2511.19773)</summary>

**Abstract:** While recent vision-language models (VLMs) demonstrate strong image understanding, their ability to "think with images", i.e., to reason through multi-step visual interactions, remains limited. We introduce VISTA-Gym, a scalable training environment for incentivizing tool-integrated visual reasoning capabilities in VLMs. VISTA-Gym unifies diverse real-world multimodal reasoning tasks (7 tasks from 13 datasets in total) with a standardized interface for visual tools (e.g., grounding, parsing), executable interaction loops, verifiable feedback signals, and efficient trajectory logging, enabling visual agentic reinforcement learning at scale. While recent VLMs exhibit strong text-only reasoning, both proprietary and open-source models still struggle with tool selection, invocation, and coordination. With VISTA-Gym, we train VISTA-R1 to interleave tool-use with agentic reasoning via multi-turn trajectory sampling and end-to-end reinforcement learning. Extensive experiments across 11 public reasoning-intensive VQA benchmarks show that VISTA-R1-8B outperforms state-of-the-art baselines with similar sizes by 9.51%-18.72%, demonstrating VISTA-Gym as an effective training ground to unlock the tool-integrated reasoning capabilities for VLMs.

**arXiv ID:** 2511.19773
</details>

<details>
<summary><strong>Reinforcement Learning with $ω$-Regular Objectives and Constraints</strong> - Dominik Wagner, Leon Witzman, Luke Ong - [[pdf]](https://arxiv.org/pdf/2511.19849)</summary>

**Abstract:** Reinforcement learning (RL) commonly relies on scalar rewards with limited ability to express temporal, conditional, or safety-critical goals, and can lead to reward hacking. Temporal logic expressible via the more general class of $\omega$-regular objectives addresses this by precisely specifying rich behavioural properties. Even still, measuring performance by a single scalar (be it reward or satisfaction probability) masks safety-performance trade-offs that arise in settings with a tolerable level of risk.
We address both limitations simultaneously by combining $\omega$-regular objectives with explicit constraints, allowing safety requirements and optimisation targets to be treated separately. We develop a model-based RL algorithm based on linear programming, which in the limit produces a policy maximising the probability of satisfying an $\omega$-regular objective while also adhering to $\omega$-regular constraints within specified thresholds. Furthermore, we establish a translation to constrained limit-average problems with optimality-preserving guarantees.

**arXiv ID:** 2511.19849
</details>

<details>
<summary><strong>Agentic AI-Empowered Conversational Embodied Intelligence Networks in 6G</strong> - Mingkai Chen, Zijie Feng, Lei Wang, Yaser Khamayseh - [[pdf]](https://arxiv.org/pdf/2511.19865)</summary>

**Abstract:** In the 6G era, semantic collaboration among multiple embodied intelligent devices (MEIDs) becomes crucial for complex task execution. However, existing systems face challenges in multimodal information fusion, adaptive communication, and decision interpretability. To address these limitations, we propose a collaborative Conversational Embodied Intelligence Network (CC-EIN) integrating multimodal feature fusion, adaptive semantic communication, task coordination, and interpretability. PerceptiNet performs cross-modal fusion of image and radar data to generate unified semantic representations. An adaptive semantic communication strategy dynamically adjusts coding schemes and transmission power according to task urgency and channel quality. A semantic-driven collaboration mechanism further supports task decomposition and conflict-free coordination among heterogeneous devices. Finally, the InDec module enhances decision transparency through Grad-CAM visualization. Simulation results in post-earthquake rescue scenarios demonstrate that CC-EIN achieves 95.4% task completion rate and 95% transmission efficiency while maintaining strong semantic consistency and energy efficiency.

**arXiv ID:** 2511.19865
</details>

<details>
<summary><strong>Improving Language Agents through BREW</strong> - Shashank Kirtania, Param Biyani, Priyanshu Gupta, Yasharth Bajpai, Roshni Iyer, Sumit Gulwani, Gustavo Soares - [[pdf]](https://arxiv.org/pdf/2511.20297)</summary>

**Abstract:** Large Language Model (LLM)-based agents are increasingly applied to tasks requiring structured reasoning, tool use, and environmental adaptation, such as data manipulation, multistep planning, and computer-use automation. However, despite their versatility, current training paradigms for model weight optimization methods, like PPO and GRPO, remain relatively impractical with their high computational overhead for rollout convergence. In addition, the resulting agent policies are difficult to interpret, adapt, or incrementally improve. To address this, we investigate creating and refining structured memory of experiential learning of an agent from its environment as an alternative route to agent optimization. We introduce BREW (Bootstrapping expeRientially-learned Environmental knoWledge), a framework for agent optimization for downstream tasks via KB construction and refinement. In our formulation, we introduce an effective method for partitioning agent memory for more efficient retrieval and refinement. BREW uses task graders and behavior rubrics to learn insights while leveraging state-space search for ensuring robustness from the noise and non-specificity in natural language. Empirical results on real world, domain-grounded benchmarks -- OSWorld, $\tau^2$Bench, and SpreadsheetBench -- show BREW achieves $10-20\%$ improvement in task precision, $10-15\%$ reduction in API/tool calls leading to faster execution time, all while maintaining computational efficiency on par with base models. Unlike prior work where memory is treated as static context, we establish the KB as a modular and controllable substrate for agent optimization -- an explicit lever for shaping behavior in a transparent, interpretable, and extensible manner.

**arXiv ID:** 2511.20297
</details>

<details>
<summary><strong>IRSDA: An Agent-Orchestrated Framework for Enterprise Intrusion Response</strong> - Damodar Panigrahi, Raj Patel, Shaswata Mitra, Sudip Mittal, Shahram Rahimi - [[pdf]](https://arxiv.org/pdf/2511.19644)</summary>

**Abstract:** Modern enterprise systems face escalating cyber threats that are increasingly dynamic, distributed, and multi-stage in nature. Traditional intrusion detection and response systems often rely on static rules and manual workflows, which limit their ability to respond with the speed and precision required in high-stakes environments. To address these challenges, we present the Intrusion Response System Digital Assistant (IRSDA), an agent-based framework designed to deliver autonomous and policy-compliant cyber defense. IRSDA combines Self-Adaptive Autonomic Computing Systems (SA-ACS) with the Knowledge guided Monitor, Analyze, Plan, and Execute (MAPE-K) loop to support real-time, partition-aware decision-making across enterprise infrastructure.
IRSDA incorporates a knowledge-driven architecture that integrates contextual information with AI-based reasoning to support system-guided intrusion response. The framework leverages retrieval mechanisms and structured representations to inform decision-making while maintaining alignment with operational policies. We assess the system using a representative real-world microservices application, demonstrating its ability to automate containment, enforce compliance, and provide traceable outputs for security analyst interpretation. This work outlines a modular and agent-driven approach to cyber defense that emphasizes explainability, system-state awareness, and operational control in intrusion response.

**arXiv ID:** 2511.19644
</details>

<details>
<summary><strong>Learning to Clean: Reinforcement Learning for Noisy Label Correction</strong> - Marzi Heidari, Hanping Zhang, Yuhong Guo - [[pdf]](https://arxiv.org/pdf/2511.19808)</summary>

**Abstract:** The challenge of learning with noisy labels is significant in machine learning, as it can severely degrade the performance of prediction models if not addressed properly. This paper introduces a novel framework that conceptualizes noisy label correction as a reinforcement learning (RL) problem. The proposed approach, Reinforcement Learning for Noisy Label Correction (RLNLC), defines a comprehensive state space representing data and their associated labels, an action space that indicates possible label corrections, and a reward mechanism that evaluates the efficacy of label corrections. RLNLC learns a deep feature representation based policy network to perform label correction through reinforcement learning, utilizing an actor-critic method. The learned policy is subsequently deployed to iteratively correct noisy training labels and facilitate the training of the prediction model. The effectiveness of RLNLC is demonstrated through extensive experiments on multiple benchmark datasets, where it consistently outperforms existing state-of-the-art techniques for learning with noisy labels.

**arXiv ID:** 2511.19808
</details>

<details>
<summary><strong>Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning</strong> - Jiaqi Liu, Kaiwen Xiong, Peng Xia, Yiyang Zhou, Haonian Ji, Lu Feng, Siwei Han, Mingyu Ding, Huaxiu Yao - [[pdf]](https://arxiv.org/pdf/2511.19900)</summary>

**Abstract:** Vision-language agents have achieved remarkable progress in a variety of multimodal reasoning tasks; however, their learning remains constrained by the limitations of human-annotated supervision. Recent self-rewarding approaches attempt to overcome this constraint by allowing models to act as their own critics or reward providers. Yet, purely text-based self-evaluation struggles to verify complex visual reasoning steps and often suffers from evaluation hallucinations. To address these challenges, inspired by recent advances in tool-integrated reasoning, we propose Agent0-VL, a self-evolving vision-language agent that achieves continual improvement with tool-integrated reasoning. Agent0-VL incorporates tool usage not only into reasoning but also into self-evaluation and self-repair, enabling the model to introspect, verify, and refine its reasoning through evidence-grounded analysis. It unifies two synergistic roles within a single LVLM: a Solver that performs multi-turn tool-integrated reasoning, and a Verifier that generates structured feedback and fine-grained self-rewards through tool-grounded critique. These roles interact through a Self-Evolving Reasoning Cycle, where tool-based verification and reinforcement learning jointly align the reasoning and evaluation distributions for stable self-improvement. Through this zero-external-reward evolution, Agent0-VL aligns its reasoning and verification behaviors without any human annotation or external reward models, achieving continual self-improvement. Experiments on geometric problem solving and visual scientific analysis show that Agent0-VL achieves an 12.5% improvement over the base model. Our code is available at this https URL.

**arXiv ID:** 2511.19900
</details>

<details>
<summary><strong>Optimize Flip Angle Schedules In MR Fingerprinting Using Reinforcement Learning</strong> - Shenjun Zhong, Zhifeng Chen, Zhaolin Chen - [[pdf]](https://arxiv.org/pdf/2511.19941)</summary>

**Abstract:** Magnetic Resonance Fingerprinting (MRF) leverages transient-state signal dynamics generated by the tunable acquisition parameters, making the design of an optimal, robust sequence a complex, high-dimensional sequential decision problem, such as optimizing one of the key parameters, flip angle. Reinforcement learning (RL) offers a promising approach to automate parameter selection, to optimize pulse sequences that maximize the distinguishability of fingerprints across the parameter space. In this work, we introduce an RL framework for optimizing the flip-angle schedule in MRF and demonstrate a learned schedule exhibiting non-periodic patterns that enhances fingerprint separability. Additionally, an interesting observation is that the RL-optimized schedule may enable a reduction in the number of repetition time, potentially accelerate MRF acquisitions.

**arXiv ID:** 2511.19941
</details>

<details>
<summary><strong>WaymoQA: A Multi-View Visual Question Answering Dataset for Safety-Critical Reasoning in Autonomous Driving</strong> - Seungjun Yu, Seonho Lee, Namho Kim, Jaeyo Shin, Junsung Park, Wonjeong Ryu, Raehyuk Jung, Hyunjung Shim - [[pdf]](https://arxiv.org/pdf/2511.20022)</summary>

**Abstract:** Recent advancements in multimodal large language models (MLLMs) have shown strong understanding of driving scenes, drawing interest in their application to autonomous driving. However, high-level reasoning in safety-critical scenarios, where avoiding one traffic risk can create another, remains a major challenge. Such reasoning is often infeasible with only a single front view and requires a comprehensive view of the environment, which we achieve through multi-view inputs. We define Safety-Critical Reasoning as a new task that leverages multi-view inputs to address this challenge. Then, we distill Safety-Critical Reasoning into two stages: first resolve the immediate risk, then mitigate the decision-induced downstream risks. To support this, we introduce WaymoQA, a dataset of 35,000 human-annotated question-answer pairs covering complex, high-risk driving scenarios. The dataset includes multiple-choice and open-ended formats across both image and video modalities. Experiments reveal that existing MLLMs underperform in safety-critical scenarios compared to normal scenes, but fine-tuning with WaymoQA significantly improves their reasoning ability, highlighting the effectiveness of our dataset in developing safer and more reasoning-capable driving agents.

**arXiv ID:** 2511.20022
</details>

<details>
<summary><strong>Leveraging weights signals - Predicting and improving generalizability in reinforcement learning</strong> - Olivier Moulin, Vincent Francois-lavet, Paul Elbers, Mark Hoogendoorn - [[pdf]](https://arxiv.org/pdf/2511.20234)</summary>

**Abstract:** Generalizability of Reinforcement Learning (RL) agents (ability to perform on environments different from the ones they have been trained on) is a key problem as agents have the tendency to overfit to their training environments. In order to address this problem and offer a solution to increase the generalizability of RL agents, we introduce a new methodology to predict the generalizability score of RL agents based on the internal weights of the agent's neural networks. Using this prediction capability, we propose some changes in the Proximal Policy Optimization (PPO) loss function to boost the generalization score of the agents trained with this upgraded version. Experimental results demonstrate that our improved PPO algorithm yields agents with stronger generalizability compared to the original version.

**arXiv ID:** 2511.20234
</details>

<details>
<summary><strong>Flash-DMD: Towards High-Fidelity Few-Step Image Generation with Efficient Distillation and Joint Reinforcement Learning</strong> - Guanjie Chen, Shirui Huang, Kai Liu, Jianchen Zhu, Xiaoye Qu, Peng Chen, Yu Cheng, Yifu Sun - [[pdf]](https://arxiv.org/pdf/2511.20549)</summary>

**Abstract:** Diffusion Models have emerged as a leading class of generative models, yet their iterative sampling process remains computationally expensive. Timestep distillation is a promising technique to accelerate generation, but it often requires extensive training and leads to image quality degradation. Furthermore, fine-tuning these distilled models for specific objectives, such as aesthetic appeal or user preference, using Reinforcement Learning (RL) is notoriously unstable and easily falls into reward hacking. In this work, we introduce Flash-DMD, a novel framework that enables fast convergence with distillation and joint RL-based refinement. Specifically, we first propose an efficient timestep-aware distillation strategy that significantly reduces training cost with enhanced realism, outperforming DMD2 with only $2.1\%$ its training cost. Second, we introduce a joint training scheme where the model is fine-tuned with an RL objective while the timestep distillation training continues simultaneously. We demonstrate that the stable, well-defined loss from the ongoing distillation acts as a powerful regularizer, effectively stabilizing the RL training process and preventing policy collapse. Extensive experiments on score-based and flow matching models show that our proposed Flash-DMD not only converges significantly faster but also achieves state-of-the-art generation quality in the few-step sampling regime, outperforming existing methods in visual quality, human preference, and text-image alignment metrics. Our work presents an effective paradigm for training efficient, high-fidelity, and stable generative models. Codes are coming soon.

**arXiv ID:** 2511.20549
</details>

<details>
<summary><strong>AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning</strong> - Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, Jiashu Wang, Tongkai Yang, Binhang Yuan, Yi Wu - [[pdf]](https://arxiv.org/pdf/2505.24298)</summary>

**Abstract:** Reinforcement learning (RL) has become a dominant paradigm for training large language models (LLMs), particularly for reasoning tasks. Effective RL for LLMs requires massive parallelization and poses an urgent need for efficient training systems. Most existing large-scale RL systems for LLMs are synchronous, alternating generation and training in a batch setting where rollouts in each training batch are generated by the same model. This approach stabilizes RL training but suffers from severe system-level inefficiency: generation must wait until the longest output in the batch is completed before model updates, resulting in GPU underutilization. We present AReaL, a fully asynchronous RL system that completely decouples generation from training. Rollout workers in AReaL continuously generate new outputs without waiting, while training workers update the model whenever a batch of data is collected. AReaL also incorporates a collection of system-level optimizations, leading to substantially higher GPU utilization. To stabilize RL training, AReaL balances the workload of rollout and training workers to control data staleness, and adopts a staleness-enhanced PPO variant to better handle outdated training samples. Extensive experiments on math and code reasoning benchmarks show that AReaL achieves up to 2.77$\times$ training speedup compared to synchronous systems with the same number of GPUs and matched or improved final performance. The code of AReaL is available at this https URL.

**arXiv ID:** 2505.24298
</details>

<details>
<summary><strong>Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive Exploration for Agentic Reinforcement Learning</strong> - Yulei Qin, Xiaoyu Tan, Zhengbao He, Gang Li, Haojia Lin, Zongyi Li, Zihan Xu, Yuchen Shi, Siqi Cai, Renting Rui, Shaofei Cai, Yuzheng Cai, Xuan Zhang, Sheng Ye, Ke Li, Xing Sun - [[pdf]](https://arxiv.org/pdf/2509.22601)</summary>

**Abstract:** Reinforcement learning (RL) is the dominant paradigm for sharpening strategic tool use capabilities of LLMs on long-horizon, sparsely-rewarded agent tasks, yet it faces a fundamental challenge of exploration-exploitation trade-off. Existing studies stimulate exploration through the lens of policy entropy, but such mechanical entropy maximization is prone to RL instability due to the multi-turn distribution shifting. In this paper, we target the progressive exploration-exploitation balance under the guidance of the agent's own experiences without succumbing to either entropy collapsing or runaway divergence. We propose SPEAR, a self-imitation learning (SIL) recipe for training agentic LLMs. It extends the vanilla SIL, where a replay buffer stores good experience for off-policy update, by gradually steering the policy entropy across stages. Specifically, the proposed curriculum scheduling harmonizes intrinsic reward shaping and self-imitation to 1) expedite exploration via frequent tool interactions at the beginning, and 2) strengthen exploitation of successful tactics upon convergence towards familiarity with the environment. We also combine bag-of-tricks of industrial RL optimizations for a strong baseline this http URL to demonstrate our effectiveness. In ALFWorld and WebShop, SPEAR increases the success rates of GRPO/GiGPO/Dr.BoT by up to 16.1%/5.1%/8.6% and 20.7%/11.8%/13.9%, respectively. In AIME24 and AIME25, SPEAR boosts this http URL by up to 3.8% and 6.1%, respectively. Such gains incur only 10%-25% extra theoretical complexity and negligible runtime overhead in practice, demonstrating the plug-and-play scalability of SPEAR.

**arXiv ID:** 2509.22601
</details>

<details>
<summary><strong>Attention Trajectories as a Diagnostic Axis for Deep Reinforcement Learning</strong> - Charlotte Beylier, Hannah Selder, Arthur Fleig, Simon M. Hofmann, Nico Scherf - [[pdf]](https://arxiv.org/pdf/2511.20591)</summary>

**Abstract:** The learning process of a reinforcement learning (RL) agent remains poorly understood beyond the mathematical formulation of its learning algorithm. To address this gap, we introduce attention-oriented metrics (ATOMs) to investigate the development of an RL agent's attention during training. In a controlled experiment, we tested ATOMs on three variations of a Pong game, each designed to teach the agent distinct behaviours, complemented by a behavioural assessment. ATOMs successfully delineate the attention patterns of an agent trained on each game variation, and that these differences in attention patterns translate into differences in the agent's behaviour. Through continuous monitoring of ATOMs during training, we observed that the agent's attention developed in phases, and that these phases were consistent across game variations. Overall, we believe that ATOM could help improve our understanding of the learning processes of RL agents and better understand the relationship between attention and learning.

**arXiv ID:** 2511.20591
</details>

<details>
<summary><strong>Agint: Agentic Graph Compilation for Software Engineering Agents</strong> - Abhi Chivukula, Jay Somasundaram, Vijay Somasundaram - [[pdf]](https://arxiv.org/pdf/2511.19635)</summary>

**Abstract:** LLM-based coding agents are increasingly common but still face challenges in context management, latency, reliability, reproducibility, and scalability. We present Agint, an agentic graph compiler, interpreter, and runtime that incrementally and hierarchically converts natural-language instructions into typed, effect-aware code DAGs. Agint introduces explicit type floors (text to data to spec to code) grounded in semantic graph transformations and a hybrid LLM and function-based JIT runtime. This enables dynamic graph refinement, reproducible and optimizable execution, speculative evaluation, and interoperability with existing developer tools. Agint's typed graph bindings improve reliability and allow concurrent composition of concurrent codebases by construction, supporting accelerated development with smaller and faster models, lower latency, efficient context utilization, and higher throughput. Hierarchical compilation allows scalable graph edits, while the graph structure supports reproducibility and efficient parallel generation. Agint provides a composable unix-style toolchain: dagify (DAG compiler), dagent (hybrid JIT runtime), schemagin (schema generator), and datagin (data transformer) for realtime, low-latency code and dataflow creation. Human developers and coding agents refine graphs through the Agint CLI, while non-technical users use Agint Flow GUI for visual editing, conversational refinement, and debugging to promote prototype agentic workflows to production code. This continuous co-creation model allows teams to prototype quickly, refine seamlessly, and deploy reliably, bridging natural language, compiler methods, and developer tooling to enable a new generation of composable, team-centric coding agents at scale.

**arXiv ID:** 2511.19635
</details>

<details>
<summary><strong>Quantum-Enhanced Reinforcement Learning for Accelerating Newton-Raphson Convergence with Ising Machines: A Case Study for Power Flow Analysis</strong> - Zeynab Kaseb, Matthias Moller, Lindsay Spoor, Jerry J. Guo, Yu Xiang, Peter Palensky, Pedro P. Vergara - [[pdf]](https://arxiv.org/pdf/2511.20237)</summary>

**Abstract:** The Newton-Raphson (NR) method is widely used for solving power flow (PF) equations due to its quadratic convergence. However, its performance deteriorates under poor initialization or extreme operating scenarios, e.g., high levels of renewable energy penetration. Traditional NR initialization strategies often fail to address these challenges, resulting in slow convergence or even divergence. We propose the use of reinforcement learning (RL) to optimize the initialization of NR, and introduce a novel quantum-enhanced RL environment update mechanism to mitigate the significant computational cost of evaluating power system states over a combinatorially large action space at each RL timestep by formulating the voltage adjustment task as a quadratic unconstrained binary optimization problem. Specifically, quantum/digital annealers are integrated into the RL environment update to evaluate state transitions using a problem Hamiltonian designed for PF. Results demonstrate significant improvements in convergence speed, a reduction in NR iteration counts, and enhanced robustness under different operating conditions.

**arXiv ID:** 2511.20237
</details>

<details>
<summary><strong>CoC-VLA: Delving into Adversarial Domain Transfer for Explainable Autonomous Driving via Chain-of-Causality Visual-Language-Action Model</strong> - Dapeng Zhang, Fei Shen, Rui Zhao, Yinda Chen, Peng Zhi, Chenyang Li, Rui Zhou, Qingguo Zhou - [[pdf]](https://arxiv.org/pdf/2511.19914)</summary>

**Abstract:** Autonomous driving represents a prominent application of artificial intelligence. Recent approaches have shifted from focusing solely on common scenarios to addressing complex, long-tail situations such as subtle human behaviors, traffic accidents, and non-compliant driving patterns. Given the demonstrated capabilities of large language models (LLMs) in understanding visual and natural language inputs and following instructions, recent methods have integrated LLMs into autonomous driving systems to enhance reasoning, interpretability, and performance across diverse scenarios. However, existing methods typically rely either on real-world data, which is suitable for industrial deployment, or on simulation data tailored to rare or hard case scenarios. Few approaches effectively integrate the complementary advantages of both data sources. To address this limitation, we propose a novel VLM-guided, end-to-end adversarial transfer framework for autonomous driving that transfers long-tail handling capabilities from simulation to real-world deployment, named CoC-VLA. The framework comprises a teacher VLM model, a student VLM model, and a discriminator. Both the teacher and student VLM models utilize a shared base architecture, termed the Chain-of-Causality Visual-Language Model (CoC VLM), which integrates temporal information via an end-to-end text adapter. This architecture supports chain-of-thought reasoning to infer complex driving logic. The teacher and student VLM models are pre-trained separately on simulated and real-world datasets. The discriminator is trained adversarially to facilitate the transfer of long-tail handling capabilities from simulated to real-world environments by the student VLM model, using a novel backpropagation strategy.

**arXiv ID:** 2511.19914
</details>

<details>
<summary><strong>Power-Efficient Autonomous Mobile Robots</strong> - Liangkai Liu, Weisong Shi, Kang G. Shin - [[pdf]](https://arxiv.org/pdf/2511.20467)</summary>

**Abstract:** This paper presents pNav, a novel power-management system that significantly enhances the power/energy-efficiency of Autonomous Mobile Robots (AMRs) by jointly optimizing their physical/mechanical and cyber subsystems. By profiling AMRs' power consumption, we identify three challenges in achieving CPS (cyber-physical system) power-efficiency that involve both cyber (C) and physical (P) subsystems: (1) variabilities of system power consumption breakdown, (2) environment-aware navigation locality, and (3) coordination of C and P subsystems. pNav takes a multi-faceted approach to achieve power-efficiency of AMRs. First, it integrates millisecond-level power consumption prediction for both C and P subsystems. Second, it includes novel real-time modeling and monitoring of spatial and temporal navigation localities for AMRs. Third, it supports dynamic coordination of AMR software (navigation, detection) and hardware (motors, DVFS driver) configurations. pNav is prototyped using the Robot Operating System (ROS) Navigation Stack, 2D LiDAR, and camera. Our in-depth evaluation with a real robot and Gazebo environments demonstrates a >96% accuracy in predicting power consumption and a 38.1% reduction in power consumption without compromising navigation accuracy and safety.

**arXiv ID:** 2511.20467
</details>

<details>
<summary><strong>AVS: A Computational and Hierarchical Storage System for Autonomous Vehicles</strong> - Yuxin Wang, Yuankai He, Weisong Shi - [[pdf]](https://arxiv.org/pdf/2511.19453)</summary>

**Abstract:** Autonomous vehicles (AVs) are evolving into mobile computing platforms, equipped with powerful processors and diverse sensors that generate massive heterogeneous data, for example 14 TB per day. Supporting emerging third-party applications calls for a general-purpose, queryable onboard storage system. Yet today's data loggers and storage stacks in vehicles fail to deliver efficient data storage and retrieval. This paper presents AVS, an Autonomous Vehicle Storage system that co-designs computation with a hierarchical layout: modality-aware reduction and compression, hot-cold tiering with daily archival, and a lightweight metadata layer for indexing. The design is grounded with system-level benchmarks on AV data that cover SSD and HDD filesystems and embedded indexing, and is validated on embedded hardware with real L4 autonomous driving traces. The prototype delivers predictable real-time ingest, fast selective retrieval, and substantial footprint reduction under modest resource budgets. The work also outlines observations and next steps toward more scalable and longer deployments to motivate storage as a first-class component in AV stacks.

**arXiv ID:** 2511.19453
</details>

<details>
<summary><strong>Reasoning-VLA: A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving</strong> - Dapeng Zhang, Zhenlong Yuan, Zhangquan Chen, Chih-Ting Liao, Yinda Chen, Fei Shen, Qingguo Zhou, Tat-Seng Chua - [[pdf]](https://arxiv.org/pdf/2511.19912)</summary>

**Abstract:** Vision-Language-Action (VLA) models have recently shown strong decision-making capabilities in autonomous driving. However, existing VLAs often struggle with achieving efficient inference and generalizing to novel autonomous vehicle configurations and driving scenarios. In this paper, we propose Reasoning-VLA, a general and fast action-generation VLA framework. The proposed model employs a set of learnable action queries, initialized via Gaussian sampling from ground-truth trajectories within the training corpus. These learnable queries interact with reasoning-enhanced vision-language features to generate continuous action trajectories in parallel. To promote robust generalization, we consolidate eight publicly available autonomous driving datasets into a standardized, Chain-of-Thought reasoning-based, and easy-to-use data format for model training. Leveraging both supervised learning and reinforcement learning fine-tuning, extensive empirical evaluations across multiple benchmarks demonstrate that Reasoning-VLA achieves state-of-the-art performance, superior generalization capability, and the excellent inference speed reported to date.

**arXiv ID:** 2511.19912
</details>

<details>
<summary><strong>Map-World: Masked Action planning and Path-Integral World Model for Autonomous Driving</strong> - Bin Hu, Zijian Lu, Haicheng Liao, Chengran Yuan, Bin Rao, Yongkang Li, Guofa Li, Zhiyong Cui, Cheng-zhong Xu, Zhenning Li - [[pdf]](https://arxiv.org/pdf/2511.20156)</summary>

**Abstract:** Motion planning for autonomous driving must handle multiple plausible futures while remaining computationally efficient. Recent end-to-end systems and world-model-based planners predict rich multi-modal trajectories, but typically rely on handcrafted anchors or reinforcement learning to select a single best mode for training and control. This selection discards information about alternative futures and complicates optimization. We propose MAP-World, a prior-free multi-modal planning framework that couples masked action planning with a path-weighted world model. The Masked Action Planning (MAP) module treats future ego motion as masked sequence completion: past waypoints are encoded as visible tokens, future waypoints are represented as mask tokens, and a driving-intent path provides a coarse scaffold. A compact latent planning state is expanded into multiple trajectory queries with injected noise, yielding diverse, temporally consistent modes without anchor libraries or teacher policies. A lightweight world model then rolls out future BEV semantics conditioned on each candidate trajectory. During training, semantic losses are computed as an expectation over modes, using trajectory probabilities as discrete path weights, so the planner learns from the full distribution of plausible futures instead of a single selected path. On NAVSIM, our method matches anchor-based approaches and achieves state-of-the-art performance among world-model-based methods, while avoiding reinforcement learning and maintaining real-time inference latency.

**arXiv ID:** 2511.20156
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
