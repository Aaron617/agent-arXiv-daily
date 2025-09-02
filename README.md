# Agent Research Papers Daily

**Last Updated:** 2025-09-02 14:01:39

**Total Papers:** 50

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (7 papers)</h2></summary>

<details>
<summary><strong>A Survey of Scientific Large Language Models: From Data Foundations to Agent Frontiers</strong> - Ming Hu, Chenglong Ma, Wei Li, Wanghan Xu, Jiamin Wu, Jucheng Hu, Tianbin Li, Guohang Zhuang, Jiaqi Liu, Yingzhou Lu, Ying Chen, Chaoyang Zhang, Cheng Tan, Jie Ying, Guocheng Wu, Shujian Gao, Pengcheng Chen, Jiashi Lin, Haitao Wu, Lulu Chen, Fengxiang Wang, Yuanyuan Zhang, Xiangyu Zhao, Feilong Tang, Encheng Su, Junzhi Ning, Xinyao Liu, Ye Du, Changkai Ji, Cheng Tang, Huihui Xu, Ziyang Chen, Ziyan Huang, Jiyao Liu, Pengfei Jiang, Yizhou Wang, Chen Tang, Jianyu Wu, Yuchen Ren, Siyuan Yan, Zhonghua Wang, Zhongxing Xu, Shiyan Su, Shangquan Sun, Runkai Zhao, Zhisheng Zhang, Yu Liu, Fudi Wang, Yuanfeng Ji, Yanzhou Su, Hongming Shan, Chunmei Feng, Jiahao Xu, Jiangtao Yan, Wenhao Tang, Diping Song, Lihao Liu, Yanyan Huang, Lequan Yu, Bin Fu, Shujun Wang, Xiaomeng Li, Xiaowei Hu, Yun Gu, Ben Fei, Zhongying Deng, Benyou Wang, Yuewen Cao, Minjie Shen, Haodong Duan, Jie Xu, Yirong Chen, Fang Yan, Hongxia Hao, Jielan Li, Jiajun Du, Yanbo Wang, Imran Razzak, Chi Zhang, Lijun Wu, Conghui He, Zhaohui Lu, Jinhai Huang, Yihao Liu, Fenghua Ling, Yuqiang Li, Aoran Wang, Qihao Zheng, Nanqing Dong, Tianfan Fu, Dongzhan Zhou, Yan Lu, Wenlong Zhang, Jin Ye, Jianfei Cai, Wanli Ouyang, Yu Qiao, Zongyuan Ge, Shixiang Tang, Junjun He - [[pdf]](https://arxiv.org/pdf/2508.21148)</summary>

**Abstract:** Scientific Large Language Models (Sci-LLMs) are transforming how knowledge is represented, integrated, and applied in scientific research, yet their progress is shaped by the complex nature of scientific data. This survey presents a comprehensive, data-centric synthesis that reframes the development of Sci-LLMs as a co-evolution between models and their underlying data substrate. We formulate a unified taxonomy of scientific data and a hierarchical model of scientific knowledge, emphasizing the multimodal, cross-scale, and domain-specific challenges that differentiate scientific corpora from general natural language processing datasets. We systematically review recent Sci-LLMs, from general-purpose foundations to specialized models across diverse scientific disciplines, alongside an extensive analysis of over 270 pre-/post-training datasets, showing why Sci-LLMs pose distinct demands -- heterogeneous, multi-scale, uncertainty-laden corpora that require representations preserving domain invariance and enabling cross-modal reasoning. On evaluation, we examine over 190 benchmark datasets and trace a shift from static exams toward process- and discovery-oriented assessments with advanced evaluation protocols. These data-centric analyses highlight persistent issues in scientific data development and discuss emerging solutions involving semi-automated annotation pipelines and expert validation. Finally, we outline a paradigm shift toward closed-loop systems where autonomous agents based on Sci-LLMs actively experiment, validate, and contribute to a living, evolving knowledge base. Collectively, this work provides a roadmap for building trustworthy, continually evolving artificial intelligence (AI) systems that function as a true partner in accelerating scientific discovery.

**arXiv ID:** 2508.21148
</details>

<details>
<summary><strong>BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design</strong> - Deepro Choudhury, Sinead Williamson, Adam Goliński, Ning Miao, Freddie Bickford Smith, Michael Kirchhof, Yizhe Zhang, Tom Rainforth - [[pdf]](https://arxiv.org/pdf/2508.21184)</summary>

**Abstract:** We propose a general-purpose approach for improving the ability of Large Language Models (LLMs) to intelligently and adaptively gather information from a user or other external source using the framework of sequential Bayesian experimental design (BED). This enables LLMs to act as effective multi-turn conversational agents and interactively interface with external environments. Our approach, which we call BED-LLM (Bayesian Experimental Design with Large Language Models), is based on iteratively choosing questions or queries that maximize the expected information gain (EIG) about the task of interest given the responses gathered previously. We show how this EIG can be formulated in a principled way using a probabilistic model derived from the LLM's belief distribution and provide detailed insights into key decisions in its construction. Further key to the success of BED-LLM are a number of specific innovations, such as a carefully designed estimator for the EIG, not solely relying on in-context updates for conditioning on previous responses, and a targeted strategy for proposing candidate queries. We find that BED-LLM achieves substantial gains in performance across a wide range of tests based on the 20-questions game and using the LLM to actively infer user preferences, compared to direct prompting of the LLM and other adaptive design strategies.

**arXiv ID:** 2508.21184
</details>

<details>
<summary><strong>SAGA: A Security Architecture for Governing AI Agentic Systems</strong> - Georgios Syros, Anshuman Suri, Jacob Ginesin, Cristina Nita-Rotaru, Alina Oprea - [[pdf]](https://arxiv.org/pdf/2504.21034)</summary>

**Abstract:** Large Language Model (LLM)-based agents increasingly interact, collaborate, and delegate tasks to one another autonomously with minimal human interaction. Industry guidelines for agentic system governance emphasize the need for users to maintain comprehensive control over their agents, mitigating potential damage from malicious agents. Several proposed agentic system designs address agent identity, authorization, and delegation, but remain purely theoretical, without concrete implementation and evaluation. Most importantly, they do not provide user-controlled agent management.
To address this gap, we propose SAGA, a scalable Security Architecture for Governing Agentic systems, that offers user oversight over their agents' lifecycle. In our design, users register their agents with a central entity, the Provider, that maintains agent contact information, user-defined access control policies, and helps agents enforce these policies on inter-agent communication. We introduce a cryptographic mechanism for deriving access control tokens, that offers fine-grained control over an agent's interaction with other agents, providing formal security guarantees. We evaluate SAGA on several agentic tasks, using agents in different geolocations, and multiple on-device and cloud LLMs, demonstrating minimal performance overhead with no impact on underlying task utility in a wide range of conditions. Our architecture enables secure and trustworthy deployment of autonomous agents, accelerating the responsible adoption of this technology in sensitive environments.

**arXiv ID:** 2504.21034
</details>

<details>
<summary><strong>Designing Smarter Conversational Agents for Kids: Lessons from Cognitive Work and Means-Ends Analyses</strong> - Vanessa Figueiredo - [[pdf]](https://arxiv.org/pdf/2508.21209)</summary>

**Abstract:** This paper presents two studies on how Brazilian children (ages 9--11) use conversational agents (CAs) for schoolwork, discovery, and entertainment, and how structured scaffolds can enhance these interactions. In Study 1, a seven-week online investigation with 23 participants (children, parents, teachers) employed interviews, observations, and Cognitive Work Analysis to map children's information-processing flows, the role of more knowledgeable others, functional uses, contextual goals, and interaction patterns to inform conversation-tree design. We identified three CA functions: School, Discovery, Entertainment, and derived ``recipe'' scaffolds mirroring parent-child support. In Study 2, we prompted GPT-4o-mini on 1,200 simulated child-CA exchanges, comparing conversation-tree recipes based on structured-prompting to an unstructured baseline. Quantitative evaluation of readability, question count/depth/diversity, and coherence revealed gains for the recipe approach. Building on these findings, we offer design recommendations: scaffolded conversation-trees, child-dedicated profiles for personalized context, and caregiver-curated content. Our contributions include the first CWA application with Brazilian children, an empirical framework of child-CA information flows, and an LLM-scaffolding ``recipe'' (i.e., structured-prompting) for effective, scaffolded learning.

**arXiv ID:** 2508.21209
</details>

<details>
<summary><strong>Mini Autonomous Car Driving based on 3D Convolutional Neural Networks</strong> - Pablo Moraes, Monica Rodriguez, Kristofer S. Kappel, Hiago Sodre, Santiago Fernandez, Igor Nunes, Bruna Guterres, Ricardo Grando - [[pdf]](https://arxiv.org/pdf/2508.21271)</summary>

**Abstract:** Autonomous driving applications have become increasingly relevant in the automotive industry due to their potential to enhance vehicle safety, efficiency, and user experience, thereby meeting the growing demand for sophisticated driving assistance features. However, the development of reliable and trustworthy autonomous systems poses challenges such as high complexity, prolonged training periods, and intrinsic levels of uncertainty. Mini Autonomous Cars (MACs) are used as a practical testbed, enabling validation of autonomous control methodologies on small-scale setups. This simplified and cost-effective environment facilitates rapid evaluation and comparison of machine learning models, which is particularly useful for algorithms requiring online training. To address these challenges, this work presents a methodology based on RGB-D information and three-dimensional convolutional neural networks (3D CNNs) for MAC autonomous driving in simulated environments. We evaluate the proposed approach against recurrent neural networks (RNNs), with architectures trained and tested on two simulated tracks with distinct environmental features. Performance was assessed using task completion success, lap-time metrics, and driving consistency. Results highlight how architectural modifications and track complexity influence the models' generalization capability and vehicle control performance. The proposed 3D CNN demonstrated promising results when compared with RNNs.

**arXiv ID:** 2508.21271
</details>

<details>
<summary><strong>2COOOL: 2nd Workshop on the Challenge Of Out-Of-Label Hazards in Autonomous Driving</strong> - Ali K. AlShami, Ryan Rabinowitz, Maged Shoman, Jianwu Fang, Lukas Picek, Shao-Yuan Lo, Steve Cruz, Khang Nhut Lam, Nachiket Kamod, Lei-Lei Li, Jugal Kalita, Terrance E. Boult - [[pdf]](https://arxiv.org/pdf/2508.21080)</summary>

**Abstract:** As the computer vision community advances autonomous driving algorithms, integrating vision-based insights with sensor data remains essential for improving perception, decision making, planning, prediction, simulation, and control. Yet we must ask: Why don't we have entirely safe self-driving cars yet? A key part of the answer lies in addressing novel scenarios, one of the most critical barriers to real-world deployment. Our 2COOOL workshop provides a dedicated forum for researchers and industry experts to push the state of the art in novelty handling, including out-of-distribution hazard detection, vision-language models for hazard understanding, new benchmarking and methodologies, and safe autonomous driving practices. The 2nd Workshop on the Challenge of Out-of-Label Hazards in Autonomous Driving (2COOOL) will be held at the International Conference on Computer Vision (ICCV) 2025 in Honolulu, Hawaii, on October 19, 2025. We aim to inspire the development of new algorithms and systems for hazard avoidance, drawing on ideas from anomaly detection, open-set recognition, open-vocabulary modeling, domain adaptation, and related fields. Building on the success of its inaugural edition at the Winter Conference on Applications of Computer Vision (WACV) 2025, the workshop will feature a mix of academic and industry participation.

**arXiv ID:** 2508.21080
</details>

<details>
<summary><strong>AI Chatbots as Professional Service Agents: Developing a Professional Identity</strong> - Wenwen Li, Kangwei Shi, Yidong Chai - [[pdf]](https://arxiv.org/pdf/2501.14179)</summary>

**Abstract:** With the rapid expansion of large language model (LLM) applications, there is an emerging shift in the role of LLM-based AI chatbots from serving merely as general inquiry tools to acting as professional service agents. However, current studies often overlook a critical aspect of professional service agents: the act of communicating in a manner consistent with their professional identities. This is of particular importance in the healthcare sector, where effective communication with patients is essential for achieving professional goals, such as promoting patient well-being by encouraging healthy behaviors. To bridge this gap, we propose LAPI (LLM-based Agent with a Professional Identity), a novel framework for designing professional service agent tailored for medical question-and-answer (Q\&A) services, ensuring alignment with a specific professional identity. Our method includes a theory-guided task planning process that decomposes complex professional tasks into manageable subtasks aligned with professional objectives and a pragmatic entropy method designed to generate professional and ethical responses with low uncertainty. Experiments on various LLMs show that the proposed approach outperforms baseline methods, including few-shot prompting, chain-of-thought prompting, across key metrics such as fluency, naturalness, empathy, patient-centricity, and ROUGE-L scores. Additionally, the ablation study underscores the contribution of each component to the overall effectiveness of the approach.

**arXiv ID:** 2501.14179
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (6 papers)</h2></summary>

<details>
<summary><strong>MMSearch-Plus: A Simple Yet Challenging Benchmark for Multimodal Browsing Agents</strong> - Xijia Tao, Yihua Teng, Xinxing Su, Xinyu Fu, Jihao Wu, Chaofan Tao, Ziru Liu, Haoli Bai, Rui Liu, Lingpeng Kong - [[pdf]](https://arxiv.org/pdf/2508.21475)</summary>

**Abstract:** Large multimodal language models (MLLMs) are increasingly deployed as web agents, yet many multimodal browsing benchmarks can be solved by shallow, fixed workflows that lean on high-recall image search and nearby text-masking the genuinely multimodal challenges of fine-grained visual reasoning, provenance verification, and long-horizon tool use. We introduce MMSearch-Plus, a benchmark of 311 tasks that highly demand multimodal understanding while preserving the difficulty profile of strong text-only browsing suites. Each item is constructed to contain multiple weak, localized visual signals that must be extracted, propagated through iterative text-image search, and cross-validated under retrieval noise before answering. Our curation procedure, Spatial-Temporal Extrapolation, seeds questions whose answers require extrapolating from spatial cues (micro-text, part-level appearance, layouts, signage) and temporal traces (broadcast overlays, seasonal context) to out-of-image facts such as events, dates, and venues. We provide a model-agnostic agent framework with browsing tools and evaluate a range of closed and open MLLMs. The strongest agent (o3) attains 15.1% without search and 36.0% accuracy with rollout under our framework, while a strong open-source model (Qwen-2.5-VL-72B-Instruct) achieves 0.0% without search and 6.9% after 20 rounds of search. Beyond answer accuracy, we assess bounding-box production and cropped-image search, and conduct an error analysis that surfaces failures in source verification, part-based reasoning, and long-horizon planning.

**arXiv ID:** 2508.21475
</details>

<details>
<summary><strong>EmbodiedOneVision: Interleaved Vision-Text-Action Pretraining for General Robot Control</strong> - Delin Qu, Haoming Song, Qizhi Chen, Zhaoqing Chen, Xianqiang Gao, Xinyi Ye, Qi Lv, Modi Shi, Guanghui Ren, Cheng Ruan, Maoqing Yao, Haoran Yang, Jiacheng Bao, Bin Zhao, Dong Wang - [[pdf]](https://arxiv.org/pdf/2508.21112)</summary>

**Abstract:** The human ability to seamlessly perform multimodal reasoning and physical interaction in the open world is a core goal for general-purpose embodied intelligent systems. Recent vision-language-action (VLA) models, which are co-trained on large-scale robot and visual-text data, have demonstrated notable progress in general robot control. However, they still fail to achieve human-level flexibility in interleaved reasoning and interaction. In this work, introduce EO-Robotics, consists of EO-1 model and EO-Data1.5M dataset. EO-1 is a unified embodied foundation model that achieves superior performance in multimodal embodied reasoning and robot control through interleaved vision-text-action pre-training. The development of EO-1 is based on two key pillars: (i) a unified architecture that processes multimodal inputs indiscriminately (image, text, video, and action), and (ii) a massive, high-quality multimodal embodied reasoning dataset, EO-Data1.5M, which contains over 1.5 million samples with emphasis on interleaved vision-text-action comprehension. EO-1 is trained through synergies between auto-regressive decoding and flow matching denoising on EO-Data1.5M, enabling seamless robot action generation and multimodal embodied reasoning. Extensive experiments demonstrate the effectiveness of interleaved vision-text-action learning for open-world understanding and generalization, validated through a variety of long-horizon, dexterous manipulation tasks across multiple embodiments. This paper details the architecture of EO-1, the data construction strategy of EO-Data1.5M, and the training methodology, offering valuable insights for developing advanced embodied foundation models.

**arXiv ID:** 2508.21112
</details>

<details>
<summary><strong>Locus: Agentic Predicate Synthesis for Directed Fuzzing</strong> - Jie Zhu, Chihao Shen, Ziyang Li, Jiahao Yu, Yizheng Chen, Kexin Pei - [[pdf]](https://arxiv.org/pdf/2508.21302)</summary>

**Abstract:** Directed fuzzing aims to find program inputs that lead to specified target program states. It has broad applications, such as debugging system crashes, confirming reported bugs, and generating exploits for potential vulnerabilities. This task is inherently challenging because target states are often deeply nested in the program, while the search space manifested by numerous possible program inputs is prohibitively large. Existing approaches rely on branch distances or manually-specified constraints to guide the search; however, the branches alone are often insufficient to precisely characterize progress toward reaching the target states, while the manually specified constraints are often tailored for specific bug types and thus difficult to generalize to diverse target states and programs.
We present Locus, a novel framework to improve the efficiency of directed fuzzing. Our key insight is to synthesize predicates to capture fuzzing progress as semantically meaningful intermediate states, serving as milestones towards reaching the target states. When used to instrument the program under fuzzing, they can reject executions unlikely to reach the target states, while providing additional coverage guidance. To automate this task and generalize to diverse programs, Locus features an agentic framework with program analysis tools to synthesize and iteratively refine the candidate predicates, while ensuring the predicates strictly relax the target states to prevent false rejections via symbolic execution. Our evaluation shows that Locus substantially improves the efficiency of eight state-of-the-art fuzzers in discovering real-world vulnerabilities, achieving an average speedup of 41.6x. So far, Locus has found eight previously unpatched bugs, with one already acknowledged with a draft patch.

**arXiv ID:** 2508.21302
</details>

<details>
<summary><strong>EconAgentic in DePIN Markets: A Large Language Model Approach to the Sharing Economy of Decentralized Physical Infrastructure</strong> - Yulin Liu, Mocca Schweitzer - [[pdf]](https://arxiv.org/pdf/2508.21368)</summary>

**Abstract:** The Decentralized Physical Infrastructure (DePIN) market is revolutionizing the sharing economy through token-based economics and smart contracts that govern decentralized operations. By 2024, DePIN projects have exceeded \$10 billion in market capitalization, underscoring their rapid growth. However, the unregulated nature of these markets, coupled with the autonomous deployment of AI agents in smart contracts, introduces risks such as inefficiencies and potential misalignment with human values. To address these concerns, we introduce EconAgentic, a Large Language Model (LLM)-powered framework designed to mitigate these challenges. Our research focuses on three key areas: 1) modeling the dynamic evolution of DePIN markets, 2) evaluating stakeholders' actions and their economic impacts, and 3) analyzing macroeconomic indicators to align market outcomes with societal goals. Through EconAgentic, we simulate how AI agents respond to token incentives, invest in infrastructure, and adapt to market conditions, comparing AI-driven decisions with human heuristic benchmarks. Our results show that EconAgentic provides valuable insights into the efficiency, inclusion, and stability of DePIN markets, contributing to both academic understanding and practical improvements in the design and governance of decentralized, tokenized economies.

**arXiv ID:** 2508.21368
</details>

<details>
<summary><strong>WebInject: Prompt Injection Attack to Web Agents</strong> - Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong - [[pdf]](https://arxiv.org/pdf/2505.11717)</summary>

**Abstract:** Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. In this work, we propose WebInject, a prompt injection attack that manipulates the webpage environment to induce a web agent to perform an attacker-specified action. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the attacker-specified action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple datasets shows that WebInject is highly effective and significantly outperforms baselines.

**arXiv ID:** 2505.11717
</details>

<details>
<summary><strong>Inducing Programmatic Skills for Agentic Tasks</strong> - Zora Zhiruo Wang, Apurva Gandhi, Graham Neubig, Daniel Fried - [[pdf]](https://arxiv.org/pdf/2504.06821)</summary>

**Abstract:** To succeed in common digital tasks such as web navigation, agents must carry out a variety of specialized tasks such as searching for products or planning a travel route. To tackle these tasks, agents can bootstrap themselves by learning task-specific skills online through interaction with the web environment. In this work, we demonstrate that programs are an effective representation for skills. We propose agent skill induction (ASI), which allows agents to adapt themselves by inducing, verifying, and utilizing program-based skills on the fly. We start with an evaluation on the WebArena agent benchmark and show that ASI outperforms the static baseline agent and its text-skill counterpart by 23.5% and 11.3% in success rate, mainly thanks to the programmatic verification guarantee during the induction phase. ASI also improves efficiency by reducing 10.7-15.3% of the steps over baselines, by composing primitive actions (e.g., click) into higher-level skills (e.g., search product). We then highlight the efficacy of ASI in remaining efficient and accurate under scaled-up web activities. Finally, we examine the generalizability of induced skills when transferring between websites, and find that ASI can effectively reuse common skills, while also updating incompatible skills to versatile website changes.

**arXiv ID:** 2504.06821
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>Transforming Wearable Data into Personal Health Insights using Large Language Model Agents</strong> - Mike A. Merrill, Akshay Paruchuri, Naghmeh Rezaei, Geza Kovacs, Javier Perez, Yun Liu, Erik Schenck, Nova Hammerquist, Jake Sunshine, Shyam Tailor, Kumar Ayush, Hao-Wei Su, Qian He, Cory Y. McLean, Mark Malhotra, Shwetak Patel, Jiening Zhan, Tim Althoff, Daniel McDuff, Xin Liu - [[pdf]](https://arxiv.org/pdf/2406.06464)</summary>

**Abstract:** Deriving personalized insights from popular wearable trackers requires complex numerical reasoning that challenges standard LLMs, necessitating tool-based approaches like code generation. Large language model (LLM) agents present a promising yet largely untapped solution for this analysis at scale. We introduce the Personal Health Insights Agent (PHIA), a system leveraging multistep reasoning with code generation and information retrieval to analyze and interpret behavioral health data. To test its capabilities, we create and share two benchmark datasets with over 4000 health insights questions. A 650-hour human expert evaluation shows that PHIA significantly outperforms a strong code generation baseline, achieving 84% accuracy on objective, numerical questions and, for open-ended ones, earning 83% favorable ratings while being twice as likely to achieve the highest quality rating. This work can advance behavioral health by empowering individuals to understand their data, enabling a new era of accessible, personalized, and data-driven wellness for the wider population.

**arXiv ID:** 2406.06464
</details>

<details>
<summary><strong>ORCA: ORchestrating Causal Agent</strong> - Joanie Hayoun Chung, Chaemyung Lim, Sumin Lee, Sungbin Lim - [[pdf]](https://arxiv.org/pdf/2508.21304)</summary>

**Abstract:** Causal inference is essential for decision-making science while the complexity of the data analysis workflow, ranging from data wrangling to causal analysis, increases substantially as the scale of data grows in complicated business environments. Especially, the execution of the workflow in relational databases by non-experts can result in repetitive bottlenecks which impede timely and responsible business insights. To address this challenge, we propose ORCA (Orchestrating Causal Agent), an LLM agentic system that can automate routine workflows in RDBMS while preserving expert oversight via human-AI interactions. ORCA orchestrates the full data analysis pipeline: interpreting natural language queries, navigating tables from DB servers, generating proper SQL codes, preprocessing data, and configuring modeling processes using causal inference libraries. Domain experts still can control the automation through iterative interactions with ORCA, enabling robust data-driven decision making with less technical expertise in statistical computing. Empirical evaluations on benchmark and synthetic e-commerce datasets demonstrate competitive performance of ORCA in table understanding, query generation, and cause-effect estimation -- achieving over $7\times$ improvement in estimating average treatment compared to GPT-4o mini.

**arXiv ID:** 2508.21304
</details>

<details>
<summary><strong>Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning</strong> - Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Hinrich SchÃ¼tze, Volker Tresp, Yunpu Ma - [[pdf]](https://arxiv.org/pdf/2508.19828)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of NLP tasks, but they remain fundamentally stateless, constrained by limited context windows that hinder long-horizon reasoning. Recent efforts to address this limitation often augment LLMs with an external memory bank, yet most existing pipelines are static and heuristic-driven, lacking any learned mechanism for deciding what to store, update, or retrieve. We present Memory-R1, a reinforcement learning (RL) framework that equips LLMs with the ability to actively manage and utilize external memory through two specialized agents: a Memory Manager that learns to perform structured memory operations, including adding, updating, deleting, or taking no operation on memory entries; and an Answer Agent that selects the most relevant entries and reasons over them to produce an answer. Both agents are fine-tuned with outcome-driven RL (PPO and GRPO), enabling adaptive memory management and utilization with minimal supervision. With as few as 152 question-answer pairs and a corresponding temporal memory bank for training, Memory-R1 outperforms the strongest existing baseline and demonstrates strong generalization across diverse question types and LLM backbones. Beyond presenting an effective approach, this work provides insights into how RL can unlock more agentic, memory-aware behavior in LLMs, pointing toward richer, more persistent reasoning systems.

**arXiv ID:** 2508.19828
</details>

<details>
<summary><strong>SimuGen: Multi-modal Agentic Framework for Constructing Block Diagram-Based Simulation Models</strong> - Xinxing Ren, Qianbo Zang, Zekun Guo - [[pdf]](https://arxiv.org/pdf/2506.15695)</summary>

**Abstract:** Recent advances in large language models (LLMs) have shown impressive performance in mathematical reasoning and code generation. However, LLMs still struggle in the simulation domain, particularly in generating Simulink models, which are essential tools in engineering and scientific research. Our preliminary experiments indicate that LLM agents often fail to produce reliable and complete Simulink simulation code from text-only inputs, likely due to the lack of Simulink-specific data in their pretraining. To address this challenge, we propose SimuGen, a multimodal agent-based framework that automatically generates accurate Simulink simulation code by leveraging both the visual Simulink diagram and domain knowledge. SimuGen coordinates several specialized agents, including an investigator, unit test reviewer, code generator, executor, debug locator, and report writer, supported by a domain-specific knowledge base. This collaborative and modular design enables interpretable, robust, and reproducible Simulink simulation generation. Our source code is publicly available at this https URL.

**arXiv ID:** 2506.15695
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (4 papers)</h2></summary>

<details>
<summary><strong>CARJAN: Agent-Based Generation and Simulation of Traffic Scenarios with AJAN</strong> - Leonard Frank Neis, Andre Antakli, Matthias Klusch - [[pdf]](https://arxiv.org/pdf/2508.21411)</summary>

**Abstract:** User-friendly modeling and virtual simulation of urban traffic scenarios with different types of interacting agents such as pedestrians, cyclists and autonomous vehicles remains a challenge. We present CARJAN, a novel tool for semi-automated generation and simulation of such scenarios based on the multi-agent engineering framework AJAN and the driving simulator CARLA. CARJAN provides a visual user interface for the modeling, storage and maintenance of traffic scenario layouts, and leverages SPARQL Behavior Tree-based decision-making and interactions for agents in dynamic scenario simulations in CARLA. CARJAN provides a first integrated approach for interactive, intelligent agent-based generation and simulation of virtual traffic scenarios in CARLA.

**arXiv ID:** 2508.21411
</details>

<details>
<summary><strong>PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation</strong> - Jiho Choi, Seojeong Park, Seongjong Song, Hyunjung Shim - [[pdf]](https://arxiv.org/pdf/2508.21720)</summary>

**Abstract:** We present a novel training-free framework, \textit{PosterForest}, for automated scientific poster generation. Unlike prior approaches, which largely neglect the hierarchical structure of scientific documents and the semantic integration of textual and visual elements, our method addresses both challenges directly. We introduce the \textit{Poster Tree}, a hierarchical intermediate representation that jointly encodes document structure and visual-textual relationships at multiple levels. Our framework employs a multi-agent collaboration strategy, where agents specializing in content summarization and layout planning iteratively coordinate and provide mutual feedback. This approach enables the joint optimization of logical consistency, content fidelity, and visual coherence. Extensive experiments on multiple academic domains show that our method outperforms existing baselines in both qualitative and quantitative evaluations. The resulting posters achieve quality closest to expert-designed ground truth and deliver superior information preservation, structural clarity, and user preference.

**arXiv ID:** 2508.21720
</details>

<details>
<summary><strong>Automated Clinical Problem Detection from SOAP Notes using a Collaborative Multi-Agent LLM Architecture</strong> - Yeawon Lee, Xiaoyang Wang, Christopher C. Yang - [[pdf]](https://arxiv.org/pdf/2508.21803)</summary>

**Abstract:** Accurate interpretation of clinical narratives is critical for patient care, but the complexity of these notes makes automation challenging. While Large Language Models (LLMs) show promise, single-model approaches can lack the robustness required for high-stakes clinical tasks. We introduce a collaborative multi-agent system (MAS) that models a clinical consultation team to address this gap. The system is tasked with identifying clinical problems by analyzing only the Subjective (S) and Objective (O) sections of SOAP notes, simulating the diagnostic reasoning process of synthesizing raw data into an assessment. A Manager agent orchestrates a dynamically assigned team of specialist agents who engage in a hierarchical, iterative debate to reach a consensus. We evaluated our MAS against a single-agent baseline on a curated dataset of 420 MIMIC-III notes. The dynamic multi-agent configuration demonstrated consistently improved performance in identifying congestive heart failure, acute kidney injury, and sepsis. Qualitative analysis of the agent debates reveals that this structure effectively surfaces and weighs conflicting evidence, though it can occasionally be susceptible to groupthink. By modeling a clinical team's reasoning process, our system offers a promising path toward more accurate, robust, and interpretable clinical decision support tools.

**arXiv ID:** 2508.21803
</details>

<details>
<summary><strong>Igniting Creative Writing in Small Language Models: LLM-as-a-Judge versus Multi-Agent Refined Rewards</strong> - Xiaolong Wei, Bo Lu, Xingyu Zhang, Zhejun Zhao, Dongdong Shen, Long Xia, Dawei Yin - [[pdf]](https://arxiv.org/pdf/2508.21476)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated remarkable creative writing capabilities, yet their substantial computational demands hinder widespread use. Enhancing Small Language Models (SLMs) offers a promising alternative, but current methods like Supervised Fine-Tuning (SFT) struggle with novelty, and Reinforcement Learning from Human Feedback (RLHF) is costly. This paper explores two distinct AI-driven reward strategies within a Reinforcement Learning from AI Feedback (RLAIF) framework to ignite the creative writing of a 7B-parameter SLM, specifically for generating Chinese greetings. The first strategy employs a RM trained on high-quality preference data curated by a novel multi-agent rejection sampling framework designed for creative tasks. The second, more novel strategy utilizes a principle-guided LLM-as-a-Judge, whose reward function is optimized via an adversarial training scheme with a reflection mechanism, to directly provide reward signals. Comprehensive experiments reveal that while both approaches significantly enhance creative output over baselines, the principle-guided LLM-as-a-Judge demonstrably yields superior generation quality. Furthermore, it offers notable advantages in training efficiency and reduced dependency on human-annotated data, presenting a more scalable and effective path towards creative SLMs. Our automated evaluation methods also exhibit strong alignment with human judgments. Our code and data are publicly available at this https URL.

**arXiv ID:** 2508.21476
</details>

</details>

<details open>
<summary><h2>Other Agent Research (5 papers)</h2></summary>

<details>
<summary><strong>MultiFluxAI Enhancing Platform Engineering with Advanced Agent-Orchestrated Retrieval Systems</strong> - Sri Ram Macharla, Sridhar Murthy J, Anjaneyulu Pasala - [[pdf]](https://arxiv.org/pdf/2508.21307)</summary>

**Abstract:** MultiFluxAI is an innovative AI platform developed to address the challenges of managing and integrating vast, disparate data sources in product engineering across application domains. It addresses both current and new service related queries that enhance user engagement in the digital ecosystem. This platform leverages advanced AI techniques, such as Generative AI, vectorization, and agentic orchestration to provide dynamic and context-aware responses to complex user queries.

**arXiv ID:** 2508.21307
</details>

<details>
<summary><strong>The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management</strong> - Tobias Lindenbauer, Igor Slinko, Ludwig Felder, Egor Bogomolov, Yaroslav Zharov - [[pdf]](https://arxiv.org/pdf/2508.21433)</summary>

**Abstract:** Large Language Model (LLM)-based agents solve complex tasks through iterative reasoning, exploration, and tool-use, a process that can result in long, expensive context histories. While state-of-the-art Software Engineering ( SE) agents like OpenHands or Cursor use LLM-based summarization to tackle this issue, it is unclear whether the increased complexity offers tangible performance benefits compared to simply omitting older observations. We present a systematic comparison of these strategies within SWE-agent on SWE-bench Verified across five diverse model configurations. We find that a simple observation-masking strategy halves cost relative to a raw agent while matching, and sometimes slightly exceeding, the solve rate of LLM summarization. For example, with Qwen3-Coder 480B, masking improves solve rate from 53.8% (raw agent) to 54.8%, while remaining competitive with summarization at a lower cost. These results suggest that, at least within SWE-agent on SWE-bench Verified, the most effective and efficient context management can be the simplest. We release code and data for reproducibility

**arXiv ID:** 2508.21433
</details>

<details>
<summary><strong>Designing Dynamic Pricing for Bike-sharing Systems via Differentiable Agent-based Simulation</strong> - Tatsuya Mitomi, Fumiyasu Makinoshima, Fumiya Makihara, Eigo Segawa - [[pdf]](https://arxiv.org/pdf/2507.23344)</summary>

**Abstract:** Bike-sharing systems are emerging in various cities as a new ecofriendly transportation system. In these systems, spatiotemporally varying user demands lead to imbalanced inventory at bicycle stations, resulting in additional relocation costs. Therefore, it is essential to manage user demand through optimal dynamic pricing for the system. However, optimal pricing design for such a system is challenging because the system involves users with diverse backgrounds and their probabilistic choices. To address this problem, we develop a differentiable agent-based simulation to rapidly design dynamic pricing in bike-sharing systems, achieving balanced bicycle inventory despite spatiotemporally heterogeneous trips and probabilistic user decisions. We first validate our approach against conventional methods through numerical experiments involving 25 bicycle stations and five time slots, yielding 100 parameters. Compared to the conventional methods, our approach obtains a more accurate solution with a 73% to 78% reduction in loss while achieving more than a 100-fold increase in convergence speed. We further validate our approach on a large-scale urban bike-sharing system scenario involving 289 bicycle stations, resulting in a total of 1156 parameters. Through simulations using the obtained pricing policies, we confirm that these policies can naturally induce balanced inventory without any manual relocation. Additionally, we find that the cost of discounts to induce the balanced inventory can be minimized by setting appropriate initial conditions.

**arXiv ID:** 2507.23344
</details>

<details>
<summary><strong>Morae: Proactively Pausing UI Agents for User Choices</strong> - Yi-Hao Peng, Dingzeyu Li, Jeffrey P. Bigham, Amy Pavel - [[pdf]](https://arxiv.org/pdf/2508.21456)</summary>

**Abstract:** User interface (UI) agents promise to make inaccessible or complex UIs easier to access for blind and low-vision (BLV) users. However, current UI agents typically perform tasks end-to-end without involving users in critical choices or making them aware of important contextual information, thus reducing user agency. For example, in our field study, a BLV participant asked to buy the cheapest available sparkling water, and the agent automatically chose one from several equally priced options, without mentioning alternative products with different flavors or better ratings. To address this problem, we introduce Morae, a UI agent that automatically identifies decision points during task execution and pauses so that users can make choices. Morae uses large multimodal models to interpret user queries alongside UI code and screenshots, and prompt users for clarification when there is a choice to be made. In a study over real-world web tasks with BLV participants, Morae helped users complete more tasks and select options that better matched their preferences, as compared to baseline agents, including OpenAI Operator. More broadly, this work exemplifies a mixed-initiative approach in which users benefit from the automation of UI agents while being able to express their preferences.

**arXiv ID:** 2508.21456
</details>

<details>
<summary><strong>Can LLMs Generate Behaviors for Embodied Virtual Agents Based on Personality Traits?</strong> - Bin Han, Deuksin Kwon, Spencer Lin, Kaleen Shrestha, Jonathan Gratch - [[pdf]](https://arxiv.org/pdf/2508.21087)</summary>

**Abstract:** This study proposes a framework that employs personality prompting with Large Language Models to generate verbal and nonverbal behaviors for virtual agents based on personality traits. Focusing on extraversion, we evaluated the system in two scenarios: negotiation and ice breaking, using both introverted and extroverted agents. In Experiment 1, we conducted agent to agent simulations and performed linguistic analysis and personality classification to assess whether the LLM generated language reflected the intended traits and whether the corresponding nonverbal behaviors varied by personality. In Experiment 2, we carried out a user study to evaluate whether these personality aligned behaviors were consistent with their intended traits and perceptible to human observers. Our results show that LLMs can generate verbal and nonverbal behaviors that align with personality traits, and that users are able to recognize these traits through the agents' behaviors. This work underscores the potential of LLMs in shaping personality aligned virtual agents.

**arXiv ID:** 2508.21087
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (24 papers)</h2></summary>

<details>
<summary><strong>Think in Games: Learning to Reason in Games via Reinforcement Learning with Large Language Models</strong> - Yi Liao, Yu Gu, Yuan Sui, Zining Zhu, Yifan Lu, Guohua Tang, Zhongqian Sun, Wei Yang - [[pdf]](https://arxiv.org/pdf/2508.21365)</summary>

**Abstract:** Large language models (LLMs) excel at complex reasoning tasks such as mathematics and coding, yet they frequently struggle with simple interactive tasks that young children perform effortlessly. This discrepancy highlights a critical gap between declarative knowledge (knowing about something) and procedural knowledge (knowing how to do something). Although traditional reinforcement learning (RL) agents can acquire procedural knowledge through environmental interaction, they often operate as black boxes and require substantial training data. In contrast, LLMs possess extensive world knowledge and reasoning capabilities, but are unable to effectively convert this static knowledge into dynamic decision-making in interactive settings. To address this challenge, we propose Think in Games (TiG), a novel framework that empowers LLMs to develop procedural understanding through direct interaction with game environments, while retaining their inherent reasoning and explanatory abilities. Specifically, TiG reformulates RL-based decision-making as a language modeling task: LLMs generate language-guided policies, which are refined iteratively through online reinforcement learning based on environmental feedback. Our experimental results show that TiG successfully bridges the gap between declarative and procedural knowledge, achieving competitive performance with dramatically lower data and computational demands compared to conventional RL methods. Moreover, TiG provides step-by-step natural language explanations for its decisions, greatly improving transparency and interpretability in complex interactive tasks.

**arXiv ID:** 2508.21365
</details>

<details>
<summary><strong>QuadKAN: KAN-Enhanced Quadruped Motion Control via End-to-End Reinforcement Learning</strong> - Allen Wang, Gavin Tao - [[pdf]](https://arxiv.org/pdf/2508.19153)</summary>

**Abstract:** We address vision-guided quadruped motion control with reinforcement learning (RL) and highlight the necessity of combining proprioception with vision for robust control. We propose QuadKAN, a spline-parameterized cross-modal policy instantiated with Kolmogorov-Arnold Networks (KANs). The framework incorporates a spline encoder for proprioception and a spline fusion head for proprioception-vision inputs. This structured function class aligns the state-to-action mapping with the piecewise-smooth nature of gait, improving sample efficiency, reducing action jitter and energy consumption, and providing interpretable posture-action sensitivities. We adopt Multi-Modal Delay Randomization (MMDR) and perform end-to-end training with Proximal Policy Optimization (PPO). Evaluations across diverse terrains, including both even and uneven surfaces and scenarios with static or dynamic obstacles, demonstrate that QuadKAN achieves consistently higher returns, greater distances, and fewer collisions than state-of-the-art (SOTA) baselines. These results show that spline-parameterized policies offer a simple, effective, and interpretable alternative for robust vision-guided locomotion. A repository will be made available upon acceptance.

**arXiv ID:** 2508.19153
</details>

<details>
<summary><strong>Beyond Prediction: Reinforcement Learning as the Defining Leap in Healthcare AI</strong> - Dilruk Perera, Gousia Habib, Qianyi Xu, Daniel J. Tan, Kai He, Erik Cambria, Mengling Feng - [[pdf]](https://arxiv.org/pdf/2508.21101)</summary>

**Abstract:** Reinforcement learning (RL) marks a fundamental shift in how artificial intelligence is applied in healthcare. Instead of merely predicting outcomes, RL actively decides interventions with long term goals. Unlike traditional models that operate on fixed associations, RL systems learn through trial, feedback, and long-term reward optimization, introducing transformative possibilities and new risks. From an information fusion lens, healthcare RL typically integrates multi-source signals such as vitals, labs clinical notes, imaging and device telemetry using temporal and decision-level mechanisms. These systems can operate within centralized, federated, or edge architectures to meet real-time clinical constraints, and naturally span data, features and decision fusion levels. This survey explore RL's rise in healthcare as more than a set of tools, rather a shift toward agentive intelligence in clinical environments. We first structure the landscape of RL techniques including model-based and model-free methods, offline and batch-constrained approaches, and emerging strategies for reward specification and uncertainty calibration through the lens of healthcare constraints. We then comprehensively analyze RL applications spanning critical care, chronic disease, mental health, diagnostics, and robotic assistance, identifying their trends, gaps, and translational bottlenecks. In contrast to prior reviews, we critically analyze RL's ethical, deployment, and reward design challenges, and synthesize lessons for safe, human-aligned policy learning. This paper serves as both a a technical roadmap and a critical reflection of RL's emerging transformative role in healthcare AI not as prediction machinery, but as agentive clinical intelligence.

**arXiv ID:** 2508.21101
</details>

<details>
<summary><strong>PVPO: Pre-Estimated Value-Based Policy Optimization for Agentic Reasoning</strong> - Wenfeng Feng, Penghong Zhao, Guochao Jiang, Chuzhan Hao, Yuewei Zhang, Hao Wang - [[pdf]](https://arxiv.org/pdf/2508.21104)</summary>

**Abstract:** Critic-free reinforcement learning methods, particularly group policies, have attracted considerable attention for their efficiency in complex tasks. However, these methods rely heavily on multiple sampling and comparisons within the policy to estimate advantage, which may cause the policy to fall into local optimum and increase computational cost. To address these issues, we propose PVPO, an efficient reinforcement learning method enhanced by an advantage reference anchor and data pre-sampling. Specifically, we use the reference model to rollout in advance and employ the calculated reward score as a reference anchor. Our approach effectively corrects the cumulative bias introduced by intra-group comparisons and significantly reduces reliance on the number of rollouts. Meanwhile, the reference model can assess sample difficulty during data pre-sampling, enabling effective selection of high-gain data to improve training efficiency. Experiments conducted on nine datasets across two domains demonstrate that PVPO achieves State-Of-The-Art (SOTA) performance. Our approach not only demonstrates robust generalization across multiple tasks, but also exhibits scalable performance across models of varying scales.

**arXiv ID:** 2508.21104
</details>

<details>
<summary><strong>Learning to Generate Unit Test via Adversarial Reinforcement Learning</strong> - Dongjun Lee, Changho Hwang, Kimin Lee - [[pdf]](https://arxiv.org/pdf/2508.21107)</summary>

**Abstract:** Unit testing is a core practice in programming, enabling systematic evaluation of programs produced by human developers or large language models (LLMs). Given the challenges in writing comprehensive unit tests, LLMs have been employed to automate test generation, yet methods for training LLMs to produce high-quality tests remain underexplored. In this work, we propose UTRL, a novel reinforcement learning framework that trains an LLM to generate high-quality unit tests given a programming instruction. Our key idea is to iteratively train two LLMs, the unit test generator and the code generator, in an adversarial manner via reinforcement learning. The unit test generator is trained to maximize a discrimination reward, which reflects its ability to produce tests that expose faults in the code generator's solutions, and the code generator is trained to maximize a code reward, which reflects its ability to produce solutions that pass the unit tests generated by the test generator. In our experiments, we demonstrate that unit tests generated by Qwen3-4B trained via UTRL show higher quality compared to unit tests generated by the same model trained via supervised fine-tuning on human-written ground-truth unit tests, yielding code evaluations that more closely align with those induced by the ground-truth tests. Moreover, Qwen3-4B trained with UTRL outperforms frontier models such as GPT-4.1 in generating high-quality unit tests, highlighting the effectiveness of UTRL in training LLMs for this task.

**arXiv ID:** 2508.21107
</details>

<details>
<summary><strong>Automating the Deep Space Network Data Systems; A Case Study in Adaptive Anomaly Detection through Agentic AI</strong> - Evan J. Chou, Lisa S. Locke, Harvey M. Soldan - [[pdf]](https://arxiv.org/pdf/2508.21111)</summary>

**Abstract:** The Deep Space Network (DSN) is NASA's largest network of antenna facilities that generate a large volume of multivariate time-series data. These facilities contain DSN antennas and transmitters that undergo degradation over long periods of time, which may cause costly disruptions to the data flow and threaten the earth-connection of dozens of spacecraft that rely on the Deep Space Network for their lifeline. The purpose of this study was to experiment with different methods that would be able to assist JPL engineers with directly pinpointing anomalies and equipment degradation through collected data, and continue conducting maintenance and operations of the DSN for future space missions around our universe. As such, we have researched various machine learning techniques that can fully reconstruct data through predictive analysis, and determine anomalous data entries within real-time datasets through statistical computations and thresholds. On top of the fully trained and tested machine learning models, we have also integrated the use of a reinforcement learning subsystem that classifies identified anomalies based on severity level and a Large Language Model that labels an explanation for each anomalous data entry, all of which can be improved and fine-tuned over time through human feedback/input. Specifically, for the DSN transmitters, we have also implemented a full data pipeline system that connects the data extraction, parsing, and processing workflow all together as there was no coherent program or script for performing these tasks before. Using this data pipeline system, we were able to then also connect the models trained from DSN antenna data, completing the data workflow for DSN anomaly detection. This was all wrapped around and further connected by an agentic AI system, where complex reasoning was utilized to determine the classifications and predictions of anomalous data.

**arXiv ID:** 2508.21111
</details>

<details>
<summary><strong>Improving Aviation Safety Analysis: Automated HFACS Classification Using Reinforcement Learning with Group Relative Policy Optimization</strong> - Arash Ahmadi, Sarah Sharif, Yaser Banad - [[pdf]](https://arxiv.org/pdf/2508.21201)</summary>

**Abstract:** Analyzing the human factors behind aviation accidents is crucial for preventing future incidents, yet traditional methods using the Human Factors Analysis and Classification System (HFACS) are limited by scalability and consistency. To address this, we introduce an automated HFACS classification framework for aviation safety analysis that utilizes Reinforcement Learning with Group Relative Policy Optimization (GRPO) to fine-tune a Llama-3.1 8B language model. Our approach incorporates a multi-component reward system tailored for aviation safety analysis and integrates synthetic data generation to overcome class imbalance in accident datasets. The resulting GRPO-optimized model achieved noticeable performance gains, including a 350% increase in exact match accuracy (from 0.0400 to 0.1800) and an improved partial match accuracy of 0.8800. Significantly, our specialized model outperforms state-of-the-art LLMs (Large Language Models), including GPT-5-mini and Gemini-2.5-fiash, on key metrics. This research also proposes exact match accuracy in multi-label HFACS classification problem as a new benchmarking methodology to evaluate the advanced reasoning capabilities of language models. Ultimately, our work validates that smaller, domain-optimized models can provide a computationally efficient and better solution for critical safety analysis. This approach makes powerful, low-latency deployment on resource-constrained edge devices feasible.

**arXiv ID:** 2508.21201
</details>

<details>
<summary><strong>HCQA: Hybrid Classical-Quantum Agent for Generating Optimal Quantum Sensor Circuits</strong> - Ahmad Alomari, Sathish A. P. Kumar - [[pdf]](https://arxiv.org/pdf/2508.21246)</summary>

**Abstract:** This study proposes an HCQA for designing optimal Quantum Sensor Circuits (QSCs) to address complex quantum physics problems. The HCQA integrates computational intelligence techniques by leveraging a Deep Q-Network (DQN) for learning and policy optimization, enhanced by a quantum-based action selection mechanism based on the Q-values. A quantum circuit encodes the agent current state using Ry gates, and then creates a superposition of possible actions. Measurement of the circuit results in probabilistic action outcomes, allowing the agent to generate optimal QSCs by selecting sequences of gates that maximize the Quantum Fisher Information (QFI) while minimizing the number of gates. This computational intelligence-driven HCQA enables the automated generation of entangled quantum states, specifically the squeezed states, with high QFI sensitivity for quantum state estimation and control. Evaluation of the HCQA on a QSC that consists of two qubits and a sequence of Rx, Ry, and S gates demonstrates its efficiency in generating optimal QSCs with a QFI of 1. This work highlights the synergy between AI-driven learning and quantum computation, illustrating how intelligent agents can autonomously discover optimal quantum circuit designs for enhanced sensing and estimation tasks.

**arXiv ID:** 2508.21246
</details>

<details>
<summary><strong>Reinforcement Learning for Optimizing Large Qubit Array based Quantum Sensor Circuits</strong> - Laxmisha Ashok Attisara, Sathish Kumar - [[pdf]](https://arxiv.org/pdf/2508.21253)</summary>

**Abstract:** As the number of qubits in a sensor increases, the complexity of designing and controlling the quantum circuits grows exponentially. Manually optimizing these circuits becomes infeasible. Optimizing entanglement distribution in large-scale quantum circuits is critical for enhancing the sensitivity and efficiency of quantum sensors [5], [6]. This paper presents an engineering integration of reinforcement learning with tensor-network-based simulation (MPS) for scalable circuit optimization for optimizing quantum sensor circuits with up to 60 qubits. To enable efficient simulation and scalability, we adopt tensor network methods, specifically the Matrix Product State (MPS) representation, instead of traditional state vector or density matrix approaches. Our reinforcement learning agent learns to restructure circuits to maximize Quantum Fisher Information (QFI) and entanglement entropy while reducing gate counts and circuit depth. Experimental results show consistent improvements, with QFI values approaching 1, entanglement entropy in the 0.8-1.0 range, and up to 90% reduction in depth and gate count. These results highlight the potential of combining quantum machine learning and tensor networks to optimize complex quantum circuits under realistic constraints.

**arXiv ID:** 2508.21253
</details>

<details>
<summary><strong>Breaking the Cold-Start Barrier: Reinforcement Learning with Double and Dueling DQNs</strong> - Minda Zhao - [[pdf]](https://arxiv.org/pdf/2508.21259)</summary>

**Abstract:** Recommender systems struggle to provide accurate suggestions to new users with limited interaction history, a challenge known as the cold-user problem. This paper proposes a reinforcement learning approach using Double and Dueling Deep Q-Networks (DQN) to dynamically learn user preferences from sparse feedback, enhancing recommendation accuracy without relying on sensitive demographic data. By integrating these advanced DQN variants with a matrix factorization model, we achieve superior performance on a large e-commerce dataset compared to traditional methods like popularity-based and active learning strategies. Experimental results show that our method, particularly Dueling DQN, reduces Root Mean Square Error (RMSE) for cold users, offering an effective solution for privacy-constrained environments.

**arXiv ID:** 2508.21259
</details>

<details>
<summary><strong>DynaMark: A Reinforcement Learning Framework for Dynamic Watermarking in Industrial Machine Tool Controllers</strong> - Navid Aftabi, Abhishek Hanchate, Satish Bukkapatnam, Dan Li - [[pdf]](https://arxiv.org/pdf/2508.21797)</summary>

**Abstract:** Industry 4.0's highly networked Machine Tool Controllers (MTCs) are prime targets for replay attacks that use outdated sensor data to manipulate actuators. Dynamic watermarking can reveal such tampering, but current schemes assume linear-Gaussian dynamics and use constant watermark statistics, making them vulnerable to the time-varying, partly proprietary behavior of MTCs. We close this gap with DynaMark, a reinforcement learning framework that models dynamic watermarking as a Markov decision process (MDP). It learns an adaptive policy online that dynamically adapts the covariance of a zero-mean Gaussian watermark using available measurements and detector feedback, without needing system knowledge. DynaMark maximizes a unique reward function balancing control performance, energy consumption, and detection confidence dynamically. We develop a Bayesian belief updating mechanism for real-time detection confidence in linear systems. This approach, independent of specific system assumptions, underpins the MDP for systems with linear dynamics. On a Siemens Sinumerik 828D controller digital twin, DynaMark achieves a reduction in watermark energy by 70% while preserving the nominal trajectory, compared to constant variance baselines. It also maintains an average detection delay equivalent to one sampling interval. A physical stepper-motor testbed validates these findings, rapidly triggering alarms with less control performance decline and exceeding existing benchmarks.

**arXiv ID:** 2508.21797
</details>

<details>
<summary><strong>Policy Expansion for Bridging Offline-to-Online Reinforcement Learning</strong> - Haichao Zhang, We Xu, Haonan Yu - [[pdf]](https://arxiv.org/pdf/2302.00935)</summary>

**Abstract:** Pre-training with offline data and online fine-tuning using reinforcement learning is a promising strategy for learning control policies by leveraging the best of both worlds in terms of sample efficiency and performance. One natural approach is to initialize the policy for online learning with the one trained offline. In this work, we introduce a policy expansion scheme for this task. After learning the offline policy, we use it as one candidate policy in a policy set. We then expand the policy set with another policy which will be responsible for further learning. The two policies will be composed in an adaptive manner for interacting with the environment. With this approach, the policy previously learned offline is fully retained during online learning, thus mitigating the potential issues such as destroying the useful behaviors of the offline policy in the initial stage of online learning while allowing the offline policy participate in the exploration naturally in an adaptive manner. Moreover, new useful behaviors can potentially be captured by the newly added policy through learning. Experiments are conducted on a number of tasks and the results demonstrate the effectiveness of the proposed approach. Code is available at this https URL

**arXiv ID:** 2302.00935
</details>

<details>
<summary><strong>DeepTrans: Deep Reasoning Translation via Reinforcement Learning</strong> - Jiaan Wang, Fandong Meng, Jie Zhou - [[pdf]](https://arxiv.org/pdf/2504.10187)</summary>

**Abstract:** Recently, deep reasoning LLMs (e.g., OpenAI o1 and DeepSeek-R1) have shown promising performance in various downstream tasks. Free translation is an important and interesting task in the multilingual world, which requires going beyond word-for-word translation. However, the task is still under-explored in deep reasoning LLMs. In this paper, we introduce DeepTrans, a deep reasoning translation model that learns free translation via reinforcement learning (RL). Specifically, we carefully build a reward model with pre-defined scoring criteria on both the translation results and the thought processes. The reward model teaches DeepTrans how to think and free-translate the given sentences during RL. Besides, our RL training does not need any labeled translations, avoiding the human-intensive annotation or resource-intensive data synthesis. Experimental results show the effectiveness of DeepTrans. Using Qwen2.5-7B as the backbone, DeepTrans improves performance by 16.3% in literature translation, and outperforms strong deep reasoning LLMs. Moreover, we summarize the failures and interesting findings during our RL exploration. We hope this work could inspire other researchers in free translation.

**arXiv ID:** 2504.10187
</details>

<details>
<summary><strong>ETTRL: Balancing Exploration and Exploitation in LLM Test-Time Reinforcement Learning Via Entropy Mechanism</strong> - Jia Liu, ChangYi He, YingQiao Lin, MingMin Yang, FeiYang Shen, ShaoGuo Liu - [[pdf]](https://arxiv.org/pdf/2508.11356)</summary>

**Abstract:** Recent advancements in Large Language Models have yielded significant improvements in complex reasoning tasks such as mathematics and programming. However, these models remain heavily dependent on annotated data and exhibit limited adaptability in unsupervised scenarios. To address these limitations, test-time reinforcement learning (TTRL) has been proposed, which enables self-optimization by leveraging model-generated pseudo-labels. Despite its promise, TTRL faces several key challenges, including high inference costs due to parallel rollouts and early-stage estimation bias that fosters overconfidence, reducing output diversity and causing performance plateaus. To address these challenges, we introduce an entropy-based mechanism to enhance the exploration-exploitation balance in test-time reinforcement learning through two strategies: Entropy-fork Tree Majority Rollout (ETMR) and Entropy-based Advantage Reshaping (EAR). Compared with the baseline, our approach enables Llama3.1-8B to achieve a 68 percent relative improvement in Pass at 1 metric on the AIME 2024 benchmark, while consuming only 60 percent of the rollout tokens budget. This highlights our method's ability to effectively optimize the trade-off between inference efficiency, diversity, and estimation robustness, thereby advancing unsupervised reinforcement learning for open-domain reasoning tasks.

**arXiv ID:** 2508.11356
</details>

<details>
<summary><strong>Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward</strong> - Yong Deng, Guoqing Wang, Zhenzhe Ying, Xiaofeng Wu, Jinzhen Lin, Wenwen Xiong, Yuqin Dai, Shuo Yang, Zhanwei Zhang, Qiwen Wang, Yang Qin, Yuan Wang, Quanxing Zha, Sunhao Dai, Changhua Meng - [[pdf]](https://arxiv.org/pdf/2508.12800)</summary>

**Abstract:** Large language models (LLMs) exhibit remarkable problem-solving abilities, but struggle with complex tasks due to static internal knowledge. Retrieval-Augmented Generation (RAG) enhances access to external information, yet remains limited in multi-hop reasoning and strategic search due to rigid workflows. Recent advancements in agentic deep research empower LLMs to autonomously reason, search, and synthesize information. However, current approaches relying on outcome-based reinforcement learning (RL) face critical issues such as conflicting gradients and reward sparsity, limiting performance gains and training efficiency. To address these, we first propose Atomic Thought, a novel LLM thinking paradigm that decomposes reasoning into fine-grained functional units. These units are supervised by Reasoning Reward Models (RRMs), which provide Atomic Thought Rewards (ATR) for fine-grained guidance. Building on this, we propose Atom-Searcher, a novel RL framework for agentic deep research that integrates Atomic Thought and ATR. Atom-Searcher uses a curriculum-inspired reward schedule, prioritizing process-level ATR early and transitioning to outcome rewards, accelerating convergence on effective reasoning paths. Experiments on seven benchmarks show consistent improvements over the state-of-the-art. Key advantages include: (1) Atom-Searcher scales computation at test-time. (2) Atomic Thought provides supervision anchors for RRMs, bridging deep research tasks and RRMs. (3) Atom-Searcher exhibits more interpretable, human-like reasoning patterns.

**arXiv ID:** 2508.12800
</details>

<details>
<summary><strong>Convergence of regularized agent-state-based Q-learning in POMDPs</strong> - Amit Sinha, Matthieu Geist, Aditya Mahajan - [[pdf]](https://arxiv.org/pdf/2508.21314)</summary>

**Abstract:** In this paper, we present a framework to understand the convergence of commonly used Q-learning reinforcement learning algorithms in practice. Two salient features of such algorithms are: (i)~the Q-table is recursively updated using an agent state (such as the state of a recurrent neural network) which is not a belief state or an information state and (ii)~policy regularization is often used to encourage exploration and stabilize the learning algorithm. We investigate the simplest form of such Q-learning algorithms which we call regularized agent-state-based Q-learning (RASQL) and show that it converges under mild technical conditions to the fixed point of an appropriately defined regularized MDP, which depends on the stationary distribution induced by the behavioral policy. We also show that a similar analysis continues to work for a variant of RASQL that learns periodic policies. We present numerical examples to illustrate that the empirical convergence behavior matches with the proposed theoretical limit.

**arXiv ID:** 2508.21314
</details>

<details>
<summary><strong>Beyond expected value: geometric mean optimization for long-term policy performance in reinforcement learning</strong> - Xinyi Sheng, Dominik Baumann - [[pdf]](https://arxiv.org/pdf/2508.21443)</summary>

**Abstract:** Reinforcement learning (RL) algorithms typically optimize the expected cumulative reward, i.e., the expected value of the sum of scalar rewards an agent receives over the course of a trajectory. The expected value averages the performance over an infinite number of trajectories. However, when deploying the agent in the real world, this ensemble average may be uninformative for the performance of individual trajectories. Thus, in many applications, optimizing the long-term performance of individual trajectories might be more desirable. In this work, we propose a novel RL algorithm that combines the standard ensemble average with the time-average growth rate, a measure for the long-term performance of individual trajectories. We first define the Bellman operator for the time-average growth rate. We then show that, under multiplicative reward dynamics, the geometric mean aligns with the time-average growth rate. To address more general and unknown reward dynamics, we propose a modified geometric mean with $N$-sliding window that captures the path-dependency as an estimator for the time-average growth rate. This estimator is embedded as a regularizer into the objective, forming a practical algorithm and enabling the policy to benefit from ensemble average and time-average simultaneously. We evaluate our algorithm in challenging simulations, where it outperforms conventional RL methods.

**arXiv ID:** 2508.21443
</details>

<details>
<summary><strong>Machine Intelligence on the Edge: Interpretable Cardiac Pattern Localisation Using Reinforcement Learning</strong> - Haozhe Tian, Qiyu Rao, Nina Moutonnet, Pietro Ferraro, Danilo Mandic - [[pdf]](https://arxiv.org/pdf/2508.21652)</summary>

**Abstract:** Matched filters are widely used to localise signal patterns due to their high efficiency and interpretability. However, their effectiveness deteriorates for low signal-to-noise ratio (SNR) signals, such as those recorded on edge devices, where prominent noise patterns can closely resemble the target within the limited length of the filter. One example is the ear-electrocardiogram (ear-ECG), where the cardiac signal is attenuated and heavily corrupted by artefacts. To address this, we propose the Sequential Matched Filter (SMF), a paradigm that replaces the conventional single matched filter with a sequence of filters designed by a Reinforcement Learning agent. By formulating filter design as a sequential decision-making process, SMF adaptively design signal-specific filter sequences that remain fully interpretable by revealing key patterns driving the decision-making. The proposed SMF framework has strong potential for reliable and interpretable clinical decision support, as demonstrated by its state-of-the-art R-peak detection and physiological state classification performance on two challenging real-world ECG datasets. The proposed formulation can also be extended to a broad range of applications that require accurate pattern localisation from noise-corrupted signals.

**arXiv ID:** 2508.21652
</details>

<details>
<summary><strong>Merging and Disentangling Views in Visual Reinforcement Learning for Robotic Manipulation</strong> - Abdulaziz Almuzairee, Rohan Patil, Dwait Bhatt, Henrik I. Christensen - [[pdf]](https://arxiv.org/pdf/2505.04619)</summary>

**Abstract:** Vision is well-known for its use in manipulation, especially using visual servoing. Due to the 3D nature of the world, using multiple camera views and merging them creates better representations for Q-learning and in turn, trains more sample efficient policies. Nevertheless, these multi-view policies are sensitive to failing cameras and can be burdensome to deploy. To mitigate these issues, we introduce a Merge And Disentanglement (MAD) algorithm that efficiently merges views to increase sample efficiency while simultaneously disentangling views by augmenting multi-view feature inputs with single-view features. This produces robust policies and allows lightweight deployment. We demonstrate the efficiency and robustness of our approach using Meta-World and ManiSkill3. For project website and code, see this https URL

**arXiv ID:** 2505.04619
</details>

<details>
<summary><strong>BiTrajDiff: Bidirectional Trajectory Generation with Diffusion Models for Offline Reinforcement Learning</strong> - Yunpeng Qing, Shuo Chen, Yixiao Chi, Shunyu Liu, Sixu Lin, Kelu Yao, Changqing Zou - [[pdf]](https://arxiv.org/pdf/2506.05762)</summary>

**Abstract:** Recent advances in offline Reinforcement Learning (RL) have proven that effective policy learning can benefit from imposing conservative constraints on pre-collected datasets. However, such static datasets often exhibit distribution bias, resulting in limited generalizability. To address this limitation, a straightforward solution is data augmentation (DA), which leverages generative models to enrich data distribution. Despite the promising results, current DA techniques focus solely on reconstructing future trajectories from given states, while ignoring the exploration of history transitions that reach them. This single-direction paradigm inevitably hinders the discovery of diverse behavior patterns, especially those leading to critical states that may have yielded high-reward outcomes. In this work, we introduce Bidirectional Trajectory Diffusion (BiTrajDiff), a novel DA framework for offline RL that models both future and history trajectories from any intermediate states. Specifically, we decompose the trajectory generation task into two independent yet complementary diffusion processes: one generating forward trajectories to predict future dynamics, and the other generating backward trajectories to trace essential history this http URL can efficiently leverage critical states as anchors to expand into potentially valuable yet underexplored regions of the state space, thereby facilitating dataset diversity. Extensive experiments on the D4RL benchmark suite demonstrate that BiTrajDiff achieves superior performance compared to other advanced DA methods across various offline RL backbones.

**arXiv ID:** 2506.05762
</details>

<details>
<summary><strong>Control of Rayleigh-Bénard Convection: Effectiveness of Reinforcement Learning in the Turbulent Regime</strong> - Thorben Markmann, Michiel Straat, Sebastian Peitz, Barbara Hammer - [[pdf]](https://arxiv.org/pdf/2504.12000)</summary>

**Abstract:** Data-driven flow control has significant potential for industry, energy systems, and climate science. In this work, we study the effectiveness of Reinforcement Learning (RL) for reducing convective heat transfer in the 2D Rayleigh-Bénard Convection (RBC) system under increasing turbulence. We investigate the generalizability of control across varying initial conditions and turbulence levels and introduce a reward shaping technique to accelerate the training. RL agents trained via single-agent Proximal Policy Optimization (PPO) are compared to linear proportional derivative (PD) controllers from classical control theory. The RL agents reduced convection, measured by the Nusselt Number, by up to 33% in moderately turbulent systems and 10% in highly turbulent settings, clearly outperforming PD control in all settings. The agents showed strong generalization performance across different initial conditions and to a significant extent, generalized to higher degrees of turbulence. The reward shaping improved sample efficiency and consistently stabilized the Nusselt Number to higher turbulence levels.

**arXiv ID:** 2504.12000
</details>

<details>
<summary><strong>Cooperative Sensing Enhanced UAV Path-Following and Obstacle Avoidance with Variable Formation</strong> - Changheng Wang, Zhiqing Wei, Wangjun Jiang, Haoyue Jiang, Zhiyong Feng - [[pdf]](https://arxiv.org/pdf/2508.21316)</summary>

**Abstract:** The high mobility of unmanned aerial vehicles (UAVs) enables them to be used in various civilian fields, such as rescue and cargo transport. Path-following is a crucial way to perform these tasks while sensing and collision avoidance are essential for safe flight. In this paper, we investigate how to efficiently and accurately achieve path-following, obstacle sensing and avoidance subtasks, as well as their conflict-free fusion scheduling. Firstly, a high precision deep reinforcement learning (DRL)-based UAV formation path-following model is developed, and the reward function with adaptive weights is designed from the perspective of distance and velocity errors. Then, we use integrated sensing and communication (ISAC) signals to detect the obstacle and derive the Cramer-Rao lower bound (CRLB) for obstacle sensing by information-level fusion, based on which we propose the variable formation enhanced obstacle position estimation (VFEO) algorithm. In addition, an online obstacle avoidance scheme without pretraining is designed to solve the sparse reward. Finally, with the aid of null space based (NSB) behavioral method, we present a hierarchical subtasks fusion strategy. Simulation results demonstrate the effectiveness and superiority of the subtask algorithms and the hierarchical fusion strategy.

**arXiv ID:** 2508.21316
</details>

<details>
<summary><strong>LocoMamba: Vision-Driven Locomotion via End-to-End Deep Reinforcement Learning with Mamba</strong> - Yinuo Wang, Gavin Tao - [[pdf]](https://arxiv.org/pdf/2508.11849)</summary>

**Abstract:** We introduce LocoMamba, a vision-driven cross-modal DRL framework built on selective state-space models, specifically leveraging Mamba, that achieves near-linear-time sequence modeling, effectively captures long-range dependencies, and enables efficient training with longer sequences. First, we embed proprioceptive states with a multilayer perceptron and patchify depth images with a lightweight convolutional neural network, producing compact tokens that improve state representation. Second, stacked Mamba layers fuse these tokens via near-linear-time selective scanning, reducing latency and memory footprint, remaining robust to token length and image resolution, and providing an inductive bias that mitigates overfitting. Third, we train the policy end-to-end with Proximal Policy Optimization under terrain and appearance randomization and an obstacle-density curriculum, using a compact state-centric reward that balances progress, smoothness, and safety. We evaluate our method in challenging simulated environments with static and moving obstacles as well as uneven terrain. Compared with state-of-the-art baselines, our method achieves higher returns and success rates with fewer collisions, exhibits stronger generalization to unseen terrains and obstacle densities, and improves training efficiency by converging in fewer updates under the same compute budget.

**arXiv ID:** 2508.11849
</details>

<details>
<summary><strong>Traversing the Narrow Path: A Two-Stage Reinforcement Learning Framework for Humanoid Beam Walking</strong> - TianChen Huang, Wei Gao, Runchen Xu, Shiwu Zhang - [[pdf]](https://arxiv.org/pdf/2508.20661)</summary>

**Abstract:** Traversing narrow beams is challenging for humanoids due to sparse, safety-critical contacts and the fragility of purely learned policies. We propose a physically grounded, two-stage framework that couples an XCoM/LIPM footstep template with a lightweight residual planner and a simple low-level tracker. Stage-1 is trained on flat ground: the tracker learns to robustly follow footstep targets by adding small random perturbations to heuristic footsteps, without any hand-crafted centerline locking, so it acquires stable contact scheduling and strong target-tracking robustness. Stage-2 is trained in simulation on a beam: a high-level planner predicts a body-frame residual (Delta x, Delta y, Delta psi) for the swing foot only, refining the template step to prioritize safe, precise placement under narrow support while preserving interpretability. To ease deployment, sensing is kept minimal and consistent between simulation and hardware: the planner consumes compact, forward-facing elevation cues together with onboard IMU and joint signals. On a Unitree G1, our system reliably traverses a 0.2 m-wide, 3 m-long beam. Across simulation and real-world studies, residual refinement consistently outperforms template-only and monolithic baselines in success rate, centerline adherence, and safety margins, while the structured footstep interface enables transparent analysis and low-friction sim-to-real transfer.

**arXiv ID:** 2508.20661
</details>

</details>

---

*This list is automatically generated daily using arXiv API*
