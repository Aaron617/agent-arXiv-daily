# Agent arXiv Daily

**Last Updated:** 2025-09-25 02:40:02

**Total Papers:** 70

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (6 papers)</h2></summary>

<details>
<summary><strong>Agentic Metacognition: Designing a "Self-Aware" Low-Code Agent for Failure Prediction and Human Handoff</strong> - Jiexi Xu - [[pdf]](https://arxiv.org/pdf/2509.19783)</summary>

**Abstract:** The inherent non-deterministic nature of autonomous agents, particularly within low-code/no-code (LCNC) environments, presents significant reliability challenges. Agents can become trapped in unforeseen loops, generate inaccurate outputs, or encounter unrecoverable failures, leading to user frustration and a breakdown of trust. This report proposes a novel architectural pattern to address these issues: the integration of a secondary, "metacognitive" layer that actively monitors the primary LCNC agent. Inspired by human introspection, this layer is designed to predict impending task failures based on a defined set of triggers, such as excessive latency or repetitive actions. Upon predicting a failure, the metacognitive agent proactively initiates a human handoff, providing the user with a clear summary of the agent's "thought process" and a detailed explanation of why it could not proceed. An empirical analysis of a prototype system demonstrates that this approach significantly increases the overall task success rate. However, this performance gain comes with a notable increase in computational overhead. The findings reframe human handoffs not as an admission of defeat but as a core design feature that enhances system resilience, improves user experience, and builds trust by providing transparency into the agent's internal state. The report discusses the practical and ethical implications of this approach and identifies key directions for future research.

**arXiv ID:** 2509.19783
</details>

<details>
<summary><strong>A Longitudinal Randomized Control Study of Companion Chatbot Use: Anthropomorphism and Its Mediating Role on Social Impacts</strong> - Rose E. Guingrich, Michael S. A. Graziano - [[pdf]](https://arxiv.org/pdf/2509.19515)</summary>

**Abstract:** Relationships with social artificial intelligence (AI) agents are on the rise. People report forming friendships, mentorships, and romantic partnerships with chatbots such as Replika, a type of social AI agent that is designed specifically for companionship. Concerns that companion chatbot relationships may harm or replace human ones have been raised, but whether and how these social consequences occur remains unclear. Prior research suggests that people's states of social need and their anthropomorphism of the AI agent may play a role in how human-AI interaction impacts human-human interaction. In this longitudinal study (N = 183), participants were randomly assigned to converse with a companion chatbot over text or to play text-based word games for 10 minutes a day for 21 consecutive days. During these 21 days, participants also completed four surveys and two audio-recorded interviews. We found that people's social health and relationships were not significantly impacted by interacting with a companion chatbot across 21 days compared to the control group. However, people who had a higher desire to socially connect anthropomorphized the chatbot more. Those who anthropomorphized the chatbot more indicated that the human-chatbot interaction had greater impacts on their social interactions and relationships with family and friends. A mediation analysis suggested that the impact of human-AI interaction on human-human social outcomes was mediated by the extent to which people anthropomorphized the AI agent, which itself was related to the desire to socially connect.

**arXiv ID:** 2509.19515
</details>

<details>
<summary><strong>The Knowledge-Behaviour Disconnect in LLM-based Chatbots</strong> - Jan Broersen - [[pdf]](https://arxiv.org/pdf/2509.20004)</summary>

**Abstract:** Large language model-based artificial conversational agents (like ChatGPT) give answers to all kinds of questions, and often enough these answers are correct. Just on the basis of that capacity alone, we may attribute knowledge to them. But do these models use this knowledge as a basis for their own conversational behaviour? I argue this is not the case, and I will refer to this failure as a `disconnect'. I further argue this disconnect is fundamental in the sense that with more data and more training of the LLM on which a conversational chatbot is based, it will not disappear. The reason is, as I will claim, that the core technique used to train LLMs does not allow for the establishment of the connection we are after. The disconnect reflects a fundamental limitation on the capacities of LLMs, and explains the source of hallucinations. I will furthermore consider the ethical version of the disconnect (ethical conversational knowledge not being aligned with ethical conversational behaviour), since in this domain researchers have come up with several additional techniques to influence a chatbot's behaviour. I will discuss how these techniques do nothing to solve the disconnect and can make it worse.

**arXiv ID:** 2509.20004
</details>

<details>
<summary><strong>Reinforcement Learning and Machine ethics:a systematic review</strong> - Ajay Vishwanath, Louise A. Dennis, Marija Slavkovik - [[pdf]](https://arxiv.org/pdf/2407.02425)</summary>

**Abstract:** Machine ethics is the field that studies how ethical behaviour can be accomplished by autonomous systems. While there exist some systematic reviews aiming to consolidate the state of the art in machine ethics prior to 2020, these tend to not include work that uses reinforcement learning agents as entities whose ethical behaviour is to be achieved. The reason for this is that only in the last years we have witnessed an increase in machine ethics studies within reinforcement learning. We present here a systematic review of reinforcement learning for machine ethics and machine ethics within reinforcement learning. Additionally, we highlight trends in terms of ethics specifications, components and frameworks of reinforcement learning, and environments used to result in ethical behaviour. Our systematic review aims to consolidate the work in machine ethics and reinforcement learning thus completing the gap in the state of the art machine ethics landscape

**arXiv ID:** 2407.02425
</details>

<details>
<summary><strong>Online Process Reward Leanring for Agentic Reinforcement Learning</strong> - Xiaoqian Liu, Ke Wang, Yuchuan Wu, Fei Huang, Yongbin Li, Junge Zhang, Jianbin Jiao - [[pdf]](https://arxiv.org/pdf/2509.19199)</summary>

**Abstract:** Large language models (LLMs) are increasingly trained with reinforcement learning (RL) as autonomous agents that reason and act over long horizons in interactive environments. However, sparse and sometimes unverifiable rewards make temporal credit assignment extremely challenging. Recent work attempts to integrate process supervision into agent learning but suffers from biased annotation, reward hacking, high-variance from overly fine-grained signals or failtures when state overlap is rare. We therefore introduce Online Process Reward Learning (OPRL), a general credit-assignment strategy for agentic RL that integrates seamlessly with standard on-policy algorithms without relying on additional rollouts or explicit step labels. In OPRL, we optimize an implicit process reward model (PRM) alternately with the agent's policy to transform trajectory preferences into implicit step rewards through a trajectory-based DPO objective. These step rewards are then used to compute step-level advantages, which are combined with episode-level advantages from outcome rewards for policy update, creating a self-reinforcing loop. Theoretical findings guarantee that the learned step rewards are consistent with trajectory preferences and act as potential-based shaping rewards, providing bounded gradients to stabilize training. Empirically, we evaluate OPRL on three distinct agent benmarks, including WebShop and VisualSokoban, as well as open-ended social interactions with unverfiable rewards in SOTOPIA. Crucially, OPRL shows superior performance over frontier LLMs and strong RL baselines across domains, achieving state-of-the-art results with higher sample-efficiency and lower variance during training. Further analysis also demonstrates the efficient exploration by OPRL using fewer actions, underscoring its potential for agentic learning in real-world scenarios.

**arXiv ID:** 2509.19199
</details>

<details>
<summary><strong>HumAIne-Chatbot: Real-Time Personalized Conversational AI via Reinforcement Learning</strong> - Georgios Makridis, George Fragiadakis, Jorge Oliveira, Tomaz Saraiva, Philip Mavrepis, Georgios Fatouros, Dimosthenis Kyriazis - [[pdf]](https://arxiv.org/pdf/2509.04303)</summary>

**Abstract:** Current conversational AI systems often provide generic, one-size-fits-all interactions that overlook individual user characteristics and lack adaptive dialogue management. To address this gap, we introduce \textbf{HumAIne-chatbot}, an AI-driven conversational agent that personalizes responses through a novel user profiling framework. The system is pre-trained on a diverse set of GPT-generated virtual personas to establish a broad prior over user types. During live interactions, an online reinforcement learning agent refines per-user models by combining implicit signals (e.g. typing speed, sentiment, engagement duration) with explicit feedback (e.g., likes and dislikes). This profile dynamically informs the chatbot dialogue policy, enabling real-time adaptation of both content and style. To evaluate the system, we performed controlled experiments with 50 synthetic personas in multiple conversation domains. The results showed consistent improvements in user satisfaction, personalization accuracy, and task achievement when personalization features were enabled. Statistical analysis confirmed significant differences between personalized and nonpersonalized conditions, with large effect sizes across key metrics. These findings highlight the effectiveness of AI-driven user profiling and provide a strong foundation for future real-world validation.

**arXiv ID:** 2509.04303
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>Design Insights and Comparative Evaluation of a Hardware-Based Cooperative Perception Architecture for Lane Change Prediction</strong> - Mohamed Manzour, Catherine M. Elias, Omar M. Shehata, Rubén Izquierdo, Miguel Ángel Sotelo - [[pdf]](https://arxiv.org/pdf/2509.20218)</summary>

**Abstract:** Research on lane change prediction has gained attention in the last few years. Most existing works in this area have been conducted in simulation environments or with pre-recorded datasets, these works often rely on simplified assumptions about sensing, communication, and traffic behavior that do not always hold in practice. Real-world deployments of lane-change prediction systems are relatively rare, and when they are reported, the practical challenges, limitations, and lessons learned are often under-documented. This study explores cooperative lane-change prediction through a real hardware deployment in mixed traffic and shares the insights that emerged during implementation and testing. We highlight the practical challenges we faced, including bottlenecks, reliability issues, and operational constraints that shaped the behavior of the system. By documenting these experiences, the study provides guidance for others working on similar pipelines.

**arXiv ID:** 2509.20218
</details>

<details>
<summary><strong>Multi-population Ensemble Genetic Programming via Cooperative Coevolution and Multi-view Learning for Classification</strong> - Mohammad Sadegh Khorshidi, Navid Yazdanjue, Hassan Gharoun, Mohammad Reza Nikoo, Fang Chen, Amir H. Gandomi - [[pdf]](https://arxiv.org/pdf/2509.19339)</summary>

**Abstract:** This paper introduces Multi-population Ensemble Genetic Programming (MEGP), a computational intelligence framework that integrates cooperative coevolution and the multiview learning paradigm to address classification challenges in high-dimensional and heterogeneous feature spaces. MEGP decomposes the input space into conditionally independent feature subsets, enabling multiple subpopulations to evolve in parallel while interacting through a dynamic ensemble-based fitness mechanism. Each individual encodes multiple genes whose outputs are aggregated via a differentiable softmax-based weighting layer, enhancing both model interpretability and adaptive decision fusion. A hybrid selection mechanism incorporating both isolated and ensemble-level fitness promotes inter-population cooperation while preserving intra-population diversity. This dual-level evolutionary dynamic facilitates structured search exploration and reduces premature convergence. Experimental evaluations across eight benchmark datasets demonstrate that MEGP consistently outperforms a baseline GP model in terms of convergence behavior and generalization performance. Comprehensive statistical analyses validate significant improvements in Log-Loss, Precision, Recall, F1 score, and AUC. MEGP also exhibits robust diversity retention and accelerated fitness gains throughout evolution, highlighting its effectiveness for scalable, ensemble-driven evolutionary learning. By unifying population-based optimization, multi-view representation learning, and cooperative coevolution, MEGP contributes a structurally adaptive and interpretable framework that advances emerging directions in evolutionary machine learning.

**arXiv ID:** 2509.19339
</details>

<details>
<summary><strong>SLM-Based Agentic AI with P-C-G: Optimized for Korean Tool Use</strong> - Changhyun Jeon, Jinhee Park, Jungwoo Choi, Keonwoo Kim, Jisu Kim, Minji Hong - [[pdf]](https://arxiv.org/pdf/2509.19369)</summary>

**Abstract:** We propose a small-scale language model (SLM) based agent architecture, Planner-Caller-Generator (P-C-G), optimized for Korean tool use. P-C-G separates planning, calling, and generation by role: the Planner produces an initial batch plan with limited on-demand replanning; the Caller returns a normalized call object after joint schema-value validation; and the Generator integrates tool outputs to produce the final answer. We apply a Korean-first value policy to reduce execution failures caused by frequent Korean-to-English code switching in Korean settings. Evaluation assumes Korean queries and Korean tool/parameter specifications; it covers single-chain, multi-chain, missing-parameters, and missing-functions scenarios, and is conducted via an LLM-as-a-Judge protocol averaged over five runs under a unified I/O interface. Results show that P-C-G delivers competitive tool-use accuracy and end-to-end quality while reducing tokens and maintaining acceptable latency, indicating that role-specialized SLMs are a cost-effective alternative for Korean tool-use agents.

**arXiv ID:** 2509.19369
</details>

<details>
<summary><strong>AutoEval: A Practical Framework for Autonomous Evaluation of Mobile Agents</strong> - Jiahui Sun, Zhichao Hua, Yubin Xia - [[pdf]](https://arxiv.org/pdf/2503.02403)</summary>

**Abstract:** Comprehensive evaluation of mobile agents can significantly advance their development and real-world applicability. However, existing benchmarks lack practicality and scalability due to the extensive manual effort in defining task reward signals and implementing evaluation codes. We propose AutoEval, an evaluation framework which tests mobile agents without any manual effort. Our approach designs a UI state change representation which can be used to automatically generate task reward signals, and employs a Judge System for autonomous evaluation. Evaluation shows AutoEval can automatically generate reward signals with high correlation to human-annotated signals, and achieve high accuracy (up to 94%) in autonomous evaluation comparable to human evaluation. Finally, we evaluate state-of-the-art mobile agents using our framework, providing insights into their performance and limitations.

**arXiv ID:** 2503.02403
</details>

<details>
<summary><strong>AutoSpec: An Agentic Framework for Automatically Drafting Patent Specification</strong> - Ryan Shea, Zhou Yu - [[pdf]](https://arxiv.org/pdf/2509.19640)</summary>

**Abstract:** Patents play a critical role in driving technological innovation by granting inventors exclusive rights to their inventions. However the process of drafting a patent application is often expensive and time-consuming, making it a prime candidate for automation. Despite recent advancements in language models, several challenges hinder the development of robust automated patent drafting systems. First, the information within a patent application is highly confidential, which often prevents the use of closed-source LLMs for automating this task. Second, the process of drafting a patent application is difficult for even the most advanced language models due to their long context, technical writing style, and specialized domain knowledge. To address these challenges, we introduce AutoSpec, a secure, agentic framework for Automatically drafting patent Specification. Our approach decomposes the drafting process into a sequence of manageable subtasks, each solvable by smaller, open-source language models enhanced with custom tools tailored for drafting patent specification. To assess our system, we design a novel evaluation protocol in collaboration with experienced patent attorneys. Our automatic and expert evaluations show that AutoSpec outperforms existing baselines on a patent drafting task.

**arXiv ID:** 2509.19640
</details>

<details>
<summary><strong>GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering</strong> - Saumya Saxena, Blake Buchanan, Chris Paxton, Peiqi Liu, Bingqing Chen, Narunas Vaskevicius, Luigi Palmieri, Jonathan Francis, Oliver Kroemer - [[pdf]](https://arxiv.org/pdf/2412.14480)</summary>

**Abstract:** In Embodied Question Answering (EQA), agents must explore and develop a semantic understanding of an unseen environment to answer a situated question with confidence. This problem remains challenging in robotics, due to the difficulties in obtaining useful semantic representations, updating these representations online, and leveraging prior world knowledge for efficient planning and exploration. To address these limitations, we propose GraphEQA, a novel approach that utilizes real-time 3D metric-semantic scene graphs (3DSGs) and task relevant images as multi-modal memory for grounding Vision-Language Models (VLMs) to perform EQA tasks in unseen environments. We employ a hierarchical planning approach that exploits the hierarchical nature of 3DSGs for structured planning and semantics-guided exploration. We evaluate GraphEQA in simulation on two benchmark datasets, HM-EQA and OpenEQA, and demonstrate that it outperforms key baselines by completing EQA tasks with higher success rates and fewer planning steps. We further demonstrate GraphEQA in multiple real-world home and office environments.

**arXiv ID:** 2412.14480
</details>

<details>
<summary><strong>Vision-Based Perception for Autonomous Vehicles in Off-Road Environment Using Deep Learning</strong> - Nelson Alves Ferreira Neto - [[pdf]](https://arxiv.org/pdf/2509.19378)</summary>

**Abstract:** Low-latency intelligent systems are required for autonomous driving on non-uniform terrain in open-pit mines and developing countries. This work proposes a perception system for autonomous vehicles on unpaved roads and off-road environments, capable of navigating rough terrain without a predefined trail. The Configurable Modular Segmentation Network (CMSNet) framework is proposed, facilitating different architectural arrangements. CMSNet configurations were trained to segment obstacles and trafficable ground on new images from unpaved/off-road scenarios with adverse conditions (night, rain, dust). We investigated applying deep learning to detect drivable regions without explicit track boundaries, studied algorithm behavior under visibility impairment, and evaluated field tests with real-time semantic segmentation. A new dataset, Kamino, is presented with almost 12,000 images from an operating vehicle with eight synchronized cameras. The Kamino dataset has a high number of labeled pixels compared to similar public collections and includes images from an off-road proving ground emulating a mine under adverse visibility. To achieve real-time inference, CMSNet CNN layers were methodically removed and fused using TensorRT, C++, and CUDA. Empirical experiments on two datasets validated the proposed system's effectiveness.

**arXiv ID:** 2509.19378
</details>

<details>
<summary><strong>Intelligent Algorithm Selection for Recommender Systems: Meta-Learning via in-depth algorithm feature engineering</strong> - Jarne Mathi Decker - [[pdf]](https://arxiv.org/pdf/2509.20134)</summary>

**Abstract:** The "No Free Lunch" theorem dictates that no single recommender algorithm is optimal for all users, creating a significant Algorithm Selection Problem. Standard meta-learning approaches aim to solve this by selecting an algorithm based on user features, but treat the fundamentally diverse algorithms themselves as equivalent, "black-box" choices. This thesis investigates the impact of overcoming this limitation by engineering a comprehensive feature set to explicitly characterize the algorithms themselves. We combine static code metrics, Abstract Syntax Tree properties, behavioral performance landmarks, and high-level conceptual features. We evaluate two meta-learners across five datasets: a baseline using only user features and our proposed model using both user and algorithm features. Our results show that the meta-learner augmented with algorithm features achieves an average NDCG@10 of 0.143, a statistically significant improvement of 11.7% over the Single Best Algorithm baseline (0.128). However, we found that the inclusion of algorithm features did not lead to an improvement in overall NDCG@10 over the meta learner using only user features (0.144). While adding algorithm features to the meta-learner did improve its Top-1 selection accuracy (+16.1%), this was counterbalanced by leading to a lower Top-3 accuracy (-10.7%). We conclude that for the per-user algorithm selection task in recommender systems, the predictive power of user features is overwhelmingly dominant. While algorithm features improve selection precision, unlocking their potential to boost overall performance remains a non-trivial challenge.

**arXiv ID:** 2509.20134
</details>

<details>
<summary><strong>Procedural Environment Generation for Tool-Use Agents</strong> - Michael Sullivan, Mareike Hartmann, Alexander Koller - [[pdf]](https://arxiv.org/pdf/2506.11045)</summary>

**Abstract:** Although the power of LLM tool-use agents has ignited a flurry of recent research in this area, the curation of tool-use training data remains an open problem$-$especially for online RL training. Existing approaches to synthetic tool-use data generation tend to be non-interactive, and/or non-compositional. We introduce RandomWorld, a pipeline for the procedural generation of interactive tools and compositional tool-use data. We show that models tuned via SFT and RL on synthetic RandomWorld data improve on a range of tool-use benchmarks, and set the new SoTA for two metrics on the NESTFUL dataset. Further experiments show that downstream performance scales with the amount of RandomWorld-generated training data, opening up the possibility of further improvement through the use of entirely synthetic data.

**arXiv ID:** 2506.11045
</details>

<details>
<summary><strong>GUIDE: A Diffusion-Based Autonomous Robot Exploration Framework Using Global Graph Inference</strong> - Zijun Che, Yinghong Zhang, Shengyi Liang, Boyu Zhou, Jun Ma, Jinni Zhou - [[pdf]](https://arxiv.org/pdf/2509.19916)</summary>

**Abstract:** Autonomous exploration in structured and complex indoor environments remains a challenging task, as existing methods often struggle to appropriately model unobserved space and plan globally efficient paths. To address these limitations, we propose GUIDE, a novel exploration framework that synergistically combines global graph inference with diffusion-based decision-making. We introduce a region-evaluation global graph representation that integrates both observed environmental data and predictions of unexplored areas, enhanced by a region-level evaluation mechanism to prioritize reliable structural inferences while discounting uncertain predictions. Building upon this enriched representation, a diffusion policy network generates stable, foresighted action sequences with significantly reduced denoising steps. Extensive simulations and real-world deployments demonstrate that GUIDE consistently outperforms state-of-the-art methods, achieving up to 18.3% faster coverage completion and a 34.9% reduction in redundant movements.

**arXiv ID:** 2509.19916
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>Nano Bio-Agents (NBA): Small Language Model Agents for Genomics</strong> - George Hong, Daniel Trejo Banos - [[pdf]](https://arxiv.org/pdf/2509.19566)</summary>

**Abstract:** We investigate the application of Small Language Models (<10 billion parameters) for genomics question answering via agentic framework to address hallucination issues and computational cost challenges. The Nano Bio-Agent (NBA) framework we implemented incorporates task decomposition, tool orchestration, and API access into well-established systems such as NCBI and AlphaGenome. Results show that SLMs combined with such agentic framework can achieve comparable and in many cases superior performance versus existing approaches utilising larger models, with our best model-agent combination achieving 98% accuracy on the GeneTuring benchmark. Notably, small 3-10B parameter models consistently achieve 85-97% accuracy while requiring much lower computational resources than conventional approaches. This demonstrates promising potential for efficiency gains, cost savings, and democratization of ML-powered genomics tools while retaining highly robust and accurate performance.

**arXiv ID:** 2509.19566
</details>

<details>
<summary><strong>Scan-do Attitude: Towards Autonomous CT Protocol Management using a Large Language Model Agent</strong> - Xingjian Kang, Linda Vorberg, Andreas Maier, Alexander Katzmann, Oliver Taubmann - [[pdf]](https://arxiv.org/pdf/2509.20270)</summary>

**Abstract:** Managing scan protocols in Computed Tomography (CT), which includes adjusting acquisition parameters or configuring reconstructions, as well as selecting postprocessing tools in a patient-specific manner, is time-consuming and requires clinical as well as technical expertise. At the same time, we observe an increasing shortage of skilled workforce in radiology. To address this issue, a Large Language Model (LLM)-based agent framework is proposed to assist with the interpretation and execution of protocol configuration requests given in natural language or a structured, device-independent format, aiming to improve the workflow efficiency and reduce technologists' workload. The agent combines in-context-learning, instruction-following, and structured toolcalling abilities to identify relevant protocol elements and apply accurate modifications. In a systematic evaluation, experimental results indicate that the agent can effectively retrieve protocol components, generate device compatible protocol definition files, and faithfully implement user requests. Despite demonstrating feasibility in principle, the approach faces limitations regarding syntactic and semantic validity due to lack of a unified device API, and challenges with ambiguous or complex requests. In summary, the findings show a clear path towards LLM-based agents for supporting scan protocol management in CT imaging.

**arXiv ID:** 2509.20270
</details>

<details>
<summary><strong>FHIR-AgentBench: Benchmarking LLM Agents for Realistic Interoperable EHR Question Answering</strong> - Gyubok Lee, Elea Bach, Eric Yang, Tom Pollard, Alistair Johnson, Edward Choi, Yugang jia, Jong Ha Lee - [[pdf]](https://arxiv.org/pdf/2509.19319)</summary>

**Abstract:** The recent shift toward the Health Level Seven Fast Healthcare Interoperability Resources (HL7 FHIR) standard opens a new frontier for clinical AI, demanding LLM agents to navigate complex, resource-based data models instead of conventional structured health data. However, existing benchmarks have lagged behind this transition, lacking the realism needed to evaluate recent LLMs on interoperable clinical data. To bridge this gap, we introduce FHIR-AgentBench, a benchmark that grounds 2,931 real-world clinical questions in the HL7 FHIR standard. Using this benchmark, we systematically evaluate agentic frameworks, comparing different data retrieval strategies (direct FHIR API calls vs. specialized tools), interaction patterns (single-turn vs. multi-turn), and reasoning strategies (natural language vs. code generation). Our experiments highlight the practical challenges of retrieving data from intricate FHIR resources and the difficulty of reasoning over them, both of which critically affect question answering performance. We publicly release the FHIR-AgentBench dataset and evaluation suite (this https URL) to promote reproducible research and the development of robust, reliable LLM agents for clinical applications.

**arXiv ID:** 2509.19319
</details>

<details>
<summary><strong>Tree Search for Language Model Agents</strong> - Jing Yu Koh, Stephen McAleer, Daniel Fried, Ruslan Salakhutdinov - [[pdf]](https://arxiv.org/pdf/2407.01476)</summary>

**Abstract:** Autonomous agents powered by language models (LMs) have demonstrated promise in their ability to perform decision-making tasks such as web automation. However, a key limitation remains: LMs, primarily optimized for natural language understanding and generation, struggle with multi-step reasoning, planning, and using environmental feedback when attempting to solve realistic computer tasks. Towards addressing this, we propose an inference-time search algorithm for LM agents to explicitly perform exploration and multi-step planning in interactive web environments. Our approach is a form of best-first tree search that operates within the actual environment space, and is complementary with most existing state-of-the-art agents. It is the first tree search algorithm for LM agents that shows effectiveness on realistic web tasks. On the challenging VisualWebArena benchmark, applying our search algorithm on top of a GPT-4o agent yields a 39.7% relative increase in success rate compared to the same baseline without search, setting a state-of-the-art success rate of 26.4%. On WebArena, search also yields a 28.0% relative improvement over a baseline agent, setting a competitive success rate of 19.2%. Our experiments highlight the effectiveness of search for web agents, and we demonstrate that performance scales with increased test-time compute. We conduct a thorough analysis of our results to highlight improvements from search, limitations, and promising directions for future work. Our code and models are publicly released at this https URL.

**arXiv ID:** 2407.01476
</details>

<details>
<summary><strong>AURA: Autonomous Upskilling with Retrieval-Augmented Agents</strong> - Alvin Zhu, Yusuke Tanaka, Andrew Goldberg, Dennis Hong - [[pdf]](https://arxiv.org/pdf/2506.02507)</summary>

**Abstract:** Designing reinforcement learning curricula for agile robots traditionally requires extensive manual tuning of reward functions, environment randomizations, and training configurations. We introduce AURA (Autonomous Upskilling with Retrieval-Augmented Agents), a schema-validated curriculum reinforcement learning (RL) framework that leverages Large Language Models (LLMs) as autonomous designers of multi-stage curricula. AURA transforms user prompts into YAML workflows that encode full reward functions, domain randomization strategies, and training configurations. All files are statically validated before any GPU time is used, ensuring efficient and reliable execution. A retrieval-augmented feedback loop allows specialized LLM agents to design, execute, and refine curriculum stages based on prior training results stored in a vector database, enabling continual improvement over time. Quantitative experiments show that AURA consistently outperforms LLM-guided baselines in generation success rate, humanoid locomotion, and manipulation tasks. Ablation studies highlight the importance of schema validation and retrieval for curriculum quality. AURA successfully trains end-to-end policies directly from user prompts and deploys them zero-shot on a custom humanoid robot in multiple environments - capabilities that did not exist previously with manually designed controllers. By abstracting the complexity of curriculum design, AURA enables scalable and adaptive policy learning pipelines that would be complex to construct by hand.

**arXiv ID:** 2506.02507
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (15 papers)</h2></summary>

<details>
<summary><strong>MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM</strong> - Wenliang Li, Rui Yan, Xu Zhang, Li Chen, Hongji Zhu, Jing Zhao, Junjun Li, Mengru Li, Wei Cao, Zihang Jiang, Wei Wei, Kun Zhang, Shaohua Kevin Zhou - [[pdf]](https://arxiv.org/pdf/2509.20067)</summary>

**Abstract:** Large language models (LLMs) have demonstrated notable potential in medical applications, yet they face substantial challenges in handling complex real-world clinical diagnoses using conventional prompting methods. Current prompt engineering and multi-agent approaches typically optimize isolated inferences, neglecting the accumulation of reusable clinical experience. To address this, this study proposes a novel Multi-Agent Clinical Diagnosis (MACD) framework, which allows LLMs to self-learn clinical knowledge via a multi-agent pipeline that summarizes, refines, and applies diagnostic insights. It mirrors how physicians develop expertise through experience, enabling more focused and accurate diagnosis on key disease-specific cues. We further extend it to a MACD-human collaborative workflow, where multiple LLM-based diagnostician agents engage in iterative consultations, supported by an evaluator agent and human oversight for cases where agreement is not reached. Evaluated on 4,390 real-world patient cases across seven diseases using diverse open-source LLMs (Llama-3.1 8B/70B, DeepSeek-R1-Distill-Llama 70B), MACD significantly improves primary diagnostic accuracy, outperforming established clinical guidelines with gains up to 22.3% (MACD). On the subset of the data, it achieves performance on par with or exceeding that of human physicians (up to 16% improvement over physicians-only diagnosis). Additionally, on the MACD-human workflow, it achieves an 18.6% improvement compared to physicians-only diagnosis. Moreover, self-learned knowledge exhibits strong cross-model stability, transferability, and model-specific personalization, while the system can generate traceable rationales, enhancing explainability. Consequently, this work presents a scalable self-learning paradigm for LLM-assisted diagnosis, bridging the gap between the intrinsic knowledge of LLMs and real-world clinical practice.

**arXiv ID:** 2509.20067
</details>

<details>
<summary><strong>Federation of Agents: A Semantics-Aware Communication Fabric for Large-Scale Agentic AI</strong> - Lorenzo Giusti, Ole Anton Werner, Riccardo Taiello, Matilde Carvalho Costa, Emre Tosun, Andrea Protani, Marc Molina, Rodrigo Lopes de Almeida, Paolo Cacace, Diogo Reis Santos, Luigi Serio - [[pdf]](https://arxiv.org/pdf/2509.20175)</summary>

**Abstract:** We present Federation of Agents (FoA), a distributed orchestration framework that transforms static multi-agent coordination into dynamic, capability-driven collaboration. FoA introduces Versioned Capability Vectors (VCVs): machine-readable profiles that make agent capabilities searchable through semantic embeddings, enabling agents to advertise their capabilities, cost, and limitations. Our aarchitecturecombines three key innovations: (1) semantic routing that matches tasks to agents over sharded HNSW indices while enforcing operational constraints through cost-biased optimization, (2) dynamic task decomposition where compatible agents collaboratively break down complex tasks into DAGs of subtasks through consensus-based merging, and (3) smart clustering that groups agents working on similar subtasks into collaborative channels for k-round refinement before synthesis. Built on top of MQTT,s publish-subscribe semantics for scalable message passing, FoA achieves sub-linear complexity through hierarchical capability matching and efficient index maintenance. Evaluation on HealthBench shows 13x improvements over single-model baselines, with clustering-enhanced laboration particularly effective for complex reasoning tasks requiring multiple perspectives. The system scales horizontally while maintaining consistent performance, demonstrating that semantic orchestration with structured collaboration can unlock the collective intelligence of heterogeneous federations of AI agents.

**arXiv ID:** 2509.20175
</details>

<details>
<summary><strong>The Heterogeneous Multi-Agent Challenge</strong> - Charles Dansereau, Junior-Samuel Lopez-Yepez, Karthik Soma, Antoine Fagette - [[pdf]](https://arxiv.org/pdf/2509.19512)</summary>

**Abstract:** Multi-Agent Reinforcement Learning (MARL) is a growing research area which gained significant traction in recent years, extending Deep RL applications to a much wider range of problems. A particularly challenging class of problems in this domain is Heterogeneous Multi-Agent Reinforcement Learning (HeMARL), where agents with different sensors, resources, or capabilities must cooperate based on local information. The large number of real-world situations involving heterogeneous agents makes it an attractive research area, yet underexplored, as most MARL research focuses on homogeneous agents (e.g., a swarm of identical robots). In MARL and single-agent RL, standardized environments such as ALE and SMAC have allowed to establish recognized benchmarks to measure progress. However, there is a clear lack of such standardized testbed for cooperative HeMARL. As a result, new research in this field often uses simple environments, where most algorithms perform near optimally, or uses weakly heterogeneous MARL environments.

**arXiv ID:** 2509.19512
</details>

<details>
<summary><strong>Knowledge Base-Aware Orchestration: A Dynamic, Privacy-Preserving Method for Multi-Agent Systems</strong> - Danilo Trombino, Vincenzo Pecorella, Alessandro de Giulii, Davide Tresoldi - [[pdf]](https://arxiv.org/pdf/2509.19599)</summary>

**Abstract:** Multi-agent systems (MAS) are increasingly tasked with solving complex, knowledge-intensive problems where effective agent orchestration is critical. Conventional orchestration methods rely on static agent descriptions, which often become outdated or incomplete. This limitation leads to inefficient task routing, particularly in dynamic environments where agent capabilities continuously evolve. We introduce Knowledge Base-Aware (KBA) Orchestration, a novel approach that augments static descriptions with dynamic, privacy-preserving relevance signals derived from each agent's internal knowledge base (KB). In the proposed framework, when static descriptions are insufficient for a clear routing decision, the orchestrator prompts the subagents in parallel. Each agent then assesses the task's relevance against its private KB, returning a lightweight ACK signal without exposing the underlying data. These collected signals populate a shared semantic cache, providing dynamic indicators of agent suitability for future queries. By combining this novel mechanism with static descriptions, our method achieves more accurate and adaptive task routing preserving agent autonomy and data confidentiality. Benchmarks show that our KBA Orchestration significantly outperforms static description-driven methods in routing precision and overall system efficiency, making it suitable for large-scale systems that require higher accuracy than standard description-driven routing.

**arXiv ID:** 2509.19599
</details>

<details>
<summary><strong>Automated Multi-Agent Workflows for RTL Design</strong> - Amulya Bhattaram, Janani Ramamoorthy, Ranit Gupta, Diana Marculescu, Dimitrios Stamoulis - [[pdf]](https://arxiv.org/pdf/2509.20182)</summary>

**Abstract:** The rise of agentic AI workflows unlocks novel opportunities for computer systems design and optimization. However, for specialized domains such as program synthesis, the relative scarcity of HDL and proprietary EDA resources online compared to more common programming tasks introduces challenges, often necessitating task-specific fine-tuning, high inference costs, and manually-crafted agent orchestration. In this work, we present VeriMaAS, a multi-agent framework designed to automatically compose agentic workflows for RTL code generation. Our key insight is to integrate formal verification feedback from HDL tools directly into workflow generation, reducing the cost of gradient-based updates or prolonged reasoning traces. Our method improves synthesis performance by 5-7% for pass@k over fine-tuned baselines, while requiring only a few hundred training examples, representing an order-of-magnitude reduction in supervision cost.

**arXiv ID:** 2509.20182
</details>

<details>
<summary><strong>Adaptive Event-Triggered Policy Gradient for Multi-Agent Reinforcement Learning</strong> - Umer Siddique, Abhinav Sinha, Yongcan Cao - [[pdf]](https://arxiv.org/pdf/2509.20338)</summary>

**Abstract:** Conventional multi-agent reinforcement learning (MARL) methods rely on time-triggered execution, where agents sample and communicate actions at fixed intervals. This approach is often computationally expensive and communication-intensive. To address this limitation, we propose ET-MAPG (Event-Triggered Multi-Agent Policy Gradient reinforcement learning), a framework that jointly learns an agent's control policy and its event-triggering policy. Unlike prior work that decouples these mechanisms, ET-MAPG integrates them into a unified learning process, enabling agents to learn not only what action to take but also when to execute it. For scenarios with inter-agent communication, we introduce AET-MAPG, an attention-based variant that leverages a self-attention mechanism to learn selective communication patterns. AET-MAPG empowers agents to determine not only when to trigger an action but also with whom to communicate and what information to exchange, thereby optimizing coordination. Both methods can be integrated with any policy gradient MARL algorithm. Extensive experiments across diverse MARL benchmarks demonstrate that our approaches achieve performance comparable to state-of-the-art, time-triggered baselines while significantly reducing both computational load and communication overhead.

**arXiv ID:** 2509.20338
</details>

<details>
<summary><strong>Multi-Agents are Social Groups: Investigating Social Influence of Multiple Agents in Human-Agent Interactions</strong> - Tianqi Song, Yugin Tan, Zicheng Zhu, Yibin Feng, Yi-Chieh Lee - [[pdf]](https://arxiv.org/pdf/2411.04578)</summary>

**Abstract:** Multi-agent systems - systems with multiple independent AI agents working together to achieve a common goal - are becoming increasingly prevalent in daily life. Drawing inspiration from the phenomenon of human group social influence, we investigate whether a group of AI agents can create social pressure on users to agree with them, potentially changing their stance on a topic. We conducted a study in which participants discussed social issues with either a single or multiple AI agents, and where the agents either agreed or disagreed with the user's stance on the topic. We found that conversing with multiple agents (holding conversation content constant) increased the social pressure felt by participants, and caused a greater shift in opinion towards the agents' stances on each topic. Our study shows the potential advantages of multi-agent systems over single-agent platforms in causing opinion change. We discuss design implications for possible multi-agent systems that promote social good, as well as the potential for malicious actors to use these systems to manipulate public opinion.

**arXiv ID:** 2411.04578
</details>

<details>
<summary><strong>Exploring Explainable Multi-agent MCTS-minimax Hybrids in Board Game Using Process Mining</strong> - Yiyu Qian, Tim Miller, Zheng Qian, Liyuan Zhao - [[pdf]](https://arxiv.org/pdf/2503.23326)</summary>

**Abstract:** Monte-Carlo Tree Search (MCTS) is a family of sampling-based search algorithms widely used for online planning in sequential decision-making domains and at the heart of many recent advances in artificial intelligence. Understanding the behavior of MCTS agents is difficult for developers and users due to the frequently large and complex search trees that result from the simulation of many possible futures, their evaluations, and their relationships. This paper presents our ongoing investigation into potential explanations for the decision-making and behavior of MCTS. A weakness of MCTS is that it constructs a highly selective tree and, as a result, can miss crucial moves and fall into tactical traps. Full-width minimax search constitutes the solution. We integrate shallow minimax search into the rollout phase of multi-agent MCTS and use process mining technique to explain agents' strategies in 3v3 checkers.

**arXiv ID:** 2503.23326
</details>

<details>
<summary><strong>AEGIS: Automated Error Generation and Identification for Multi-Agent Systems</strong> - Fanqi Kong, Ruijie Zhang, Huaxiao Yin, Guibin Zhang, Xiaofei Zhang, Ziang Chen, Zhaowei Zhang, Xiaoyuan Zhang, Song-Chun Zhu, Xue Feng - [[pdf]](https://arxiv.org/pdf/2509.14295)</summary>

**Abstract:** As Multi-Agent Systems (MAS) become increasingly autonomous and complex, understanding their error modes is critical for ensuring their reliability and safety. However, research in this area has been severely hampered by the lack of large-scale, diverse datasets with precise, ground-truth error labels. To address this bottleneck, we introduce \textbf{AEGIS}, a novel framework for \textbf{A}utomated \textbf{E}rror \textbf{G}eneration and \textbf{I}dentification for Multi-Agent \textbf{S}ystems. By systematically injecting controllable and traceable errors into initially successful trajectories, we create a rich dataset of realistic failures. This is achieved using a context-aware, LLM-based adaptive manipulator that performs sophisticated attacks like prompt injection and response corruption to induce specific, predefined error modes. We demonstrate the value of our dataset by exploring three distinct learning paradigms for the error identification task: Supervised Fine-Tuning, Reinforcement Learning, and Contrastive Learning. Our comprehensive experiments show that models trained on AEGIS data achieve substantial improvements across all three learning paradigms. Notably, several of our fine-tuned models demonstrate performance competitive with or superior to proprietary systems an order of magnitude larger, validating our automated data generation framework as a crucial resource for developing more robust and interpretable multi-agent systems. Our project website is available at this https URL.

**arXiv ID:** 2509.14295
</details>

<details>
<summary><strong>Optimal Multi-agent Path Finding in Continuous Time</strong> - Alvin Combrink, Sabino Francesco Roselli, Martin Fabian - [[pdf]](https://arxiv.org/pdf/2508.16410)</summary>

**Abstract:** Continuous-time Conflict Based-Search (CCBS) has long been viewed as the standard optimal baseline for multi-agent path finding in continuous time (MAPFR), yet recent critiques show that the theoretically described CCBS can fail to terminate on solvable MAPFR problems while the publicly available reference implementation can return sub-optimal solutions. This work presents an analytical framework that yields simple and sufficient conditions under which any CCBS-style algorithm is both sound and solution complete. Investigating the reference CCBS implementation reveals that it violates our sufficient conditions for soundness, with counterexamples demonstrating sub-optimality.
Leveraging the framework, we introduce a branching rule ($\delta$-BR) and prove it restores soundness and termination guarantees. Consequently, the resulting CCBS variant is both sound and solution complete. To our knowledge, this is the first MAPFR solver matching the guarantees of the discrete-time CBS. On a constructed example, CCBS with $\delta$-BR improves sum-of-costs from 10.707 to 9.000 ($\approx$ 16% lower) compared to the reference CCBS implementation. Across benchmarks, the reference CCBS implementation is generally able to find solutions faster than CCBS with $\delta$-BR due to its more aggressive pruning. However, this comes at the cost of occasional sub-optimality and potential non-termination when all solutions are pruned, whereas $\delta$-BR preserves optimality and guarantees termination by design. Because $\delta$-BR largely only affects the branching step, it can be adopted as a drop-in replacement in existing codebases. Beyond CCBS, the analytical framework and termination criterion provide a systematic way to evaluate other CCBS-like MAPFR solvers and future extensions, thereby offering tools for rigorous analysis of next-generation MAPFR algorithms.

**arXiv ID:** 2508.16410
</details>

<details>
<summary><strong>PromptSculptor: Multi-Agent Based Text-to-Image Prompt Optimization</strong> - Dawei Xiang, Wenyan Xu, Kexin Chu, Tianqi Ding, Zixu Shen, Yiming Zeng, Jianchang Su, Wei Zhang - [[pdf]](https://arxiv.org/pdf/2509.12446)</summary>

**Abstract:** The rapid advancement of generative AI has democratized access to powerful tools such as Text-to-Image models. However, to generate high-quality images, users must still craft detailed prompts specifying scene, style, and context-often through multiple rounds of refinement. We propose PromptSculptor, a novel multi-agent framework that automates this iterative prompt optimization process. Our system decomposes the task into four specialized agents that work collaboratively to transform a short, vague user prompt into a comprehensive, refined prompt. By leveraging Chain-of-Thought reasoning, our framework effectively infers hidden context and enriches scene and background details. To iteratively refine the prompt, a self-evaluation agent aligns the modified prompt with the original input, while a feedback-tuning agent incorporates user feedback for further refinement. Experimental results demonstrate that PromptSculptor significantly enhances output quality and reduces the number of iterations needed for user satisfaction. Moreover, its model-agnostic design allows seamless integration with various T2I models, paving the way for industrial applications.

**arXiv ID:** 2509.12446
</details>

<details>
<summary><strong>Augmenting Multi-Agent Communication with State Delta Trajectory</strong> - Yichen Tang, Weihang Su, Yujia Zhou, Yiqun Liu, Min Zhang, Shaoping Ma, Qingyao Ai - [[pdf]](https://arxiv.org/pdf/2506.19209)</summary>

**Abstract:** Multi-agent techniques such as role playing or multi-turn debates have been shown to be effective in improving the performance of large language models (LLMs) in downstream tasks. Despite their differences in workflows, existing multi-agent systems constructed from a single base LLM mostly use natural language for agent communication. While this is appealing for its simplicity and interpretability, it also introduces inevitable information loss as one model must down sample its continuous state vectors to discrete tokens before transferring them to the other model. Such losses are particularly significant when the information to transfer is not simple facts, but reasoning logics or abstractive thoughts. To tackle this problem, we propose a new communication protocol that transfers both natural language tokens and token-wise state transition trajectory from one agent to another. Particularly, compared to the actual state value, we find that the sequence of state changes in LLMs after generating each token can better reflect the information hidden behind the inference process. We propose a State Delta Encoding (SDE) method to represent state transition trajectories. The experimental results show that multi-agent systems with SDE achieve SOTA performance compared to other communication protocols, particularly in tasks that involve complex reasoning.

**arXiv ID:** 2506.19209
</details>

<details>
<summary><strong>MAPEX: A Multi-Agent Pipeline for Keyphrase Extraction</strong> - Liting Zhang, Shiwan Zhao, Aobo Kong, Qicheng Li - [[pdf]](https://arxiv.org/pdf/2509.18813)</summary>

**Abstract:** Keyphrase extraction is a fundamental task in natural language processing. However, existing unsupervised prompt-based methods for Large Language Models (LLMs) often rely on single-stage inference pipelines with uniform prompting, regardless of document length or LLM backbone. Such one-size-fits-all designs hinder the full exploitation of LLMs' reasoning and generation capabilities, especially given the complexity of keyphrase extraction across diverse scenarios. To address these challenges, we propose MAPEX, the first framework that introduces multi-agent collaboration into keyphrase extraction. MAPEX coordinates LLM-based agents through modules for expert recruitment, candidate extraction, topic guidance, knowledge augmentation, and post-processing. A dual-path strategy dynamically adapts to document length: knowledge-driven extraction for short texts and topic-guided extraction for long texts. Extensive experiments on six benchmark datasets across three different LLMs demonstrate its strong generalization and universality, outperforming the state-of-the-art unsupervised method by 2.44% and standard LLM baselines by 4.01% in F1@5 on average. Code is available at this https URL.

**arXiv ID:** 2509.18813
</details>

<details>
<summary><strong>Hybrid Safety Verification of Multi-Agent Systems using $ψ$-Weighted CBFs and PAC Guarantees</strong> - Venkat Margapuri, Garik Kazanjian, Naren Kosaraju - [[pdf]](https://arxiv.org/pdf/2509.20093)</summary>

**Abstract:** This study proposes a hybrid safety verification framework for closed-loop multi-agent systems under bounded stochastic disturbances. The proposed approach augments control barrier functions with a novel $\psi$-weighted formulation that encodes directional control alignment between agents into the safety constraints. Deterministic admissibility is combined with empirical validation via Monte Carlo rollouts, and a PAC-style guarantee is derived based on margin-aware safety violations to provide a probabilistic safety certificate. The results from the experiments conducted under different bounded stochastic disturbances validate the feasibility of the proposed approach.

**arXiv ID:** 2509.20093
</details>

<details>
<summary><strong>RG-Attn: Radian Glue Attention for Multi-modality Multi-agent Cooperative Perception</strong> - Lantao Li, Kang Yang, Wenqi Zhang, Xiaoxue Wang, Chen Sun - [[pdf]](https://arxiv.org/pdf/2501.16803)</summary>

**Abstract:** Cooperative perception enhances autonomous driving by leveraging Vehicle-to-Everything (V2X) communication for multi-agent sensor fusion. However, most existing methods rely on single-modal data sharing, limiting fusion performance, particularly in heterogeneous sensor settings involving both LiDAR and cameras across vehicles and roadside units (RSUs). To address this, we propose Radian Glue Attention (RG-Attn), a lightweight and generalizable cross-modal fusion module that unifies intra-agent and inter-agent fusion via transformation-based coordinate alignment and a unified sampling/inversion strategy. RG-Attn efficiently aligns features through a radian-based attention constraint, operating column-wise on geometrically consistent regions to reduce overhead and preserve spatial coherence, thereby enabling accurate and robust fusion. Building upon RG-Attn, we propose three cooperative architectures. The first, Paint-To-Puzzle (PTP), prioritizes communication efficiency but assumes all agents have LiDAR, optionally paired with cameras. The second, Co-Sketching-Co-Coloring (CoS-CoCo), offers maximal flexibility, supporting any sensor setup (e.g., LiDAR-only, camera-only, or both) and enabling strong cross-modal generalization for real-world deployment. The third, Pyramid-RG-Attn Fusion (PRGAF), aims for peak detection accuracy with the highest computational overhead. Extensive evaluations on simulated and real-world datasets show our framework delivers state-of-the-art detection accuracy with high flexibility and efficiency. GitHub Link: this https URL

**arXiv ID:** 2501.16803
</details>

</details>

<details open>
<summary><h2>Other Agent Research (8 papers)</h2></summary>

<details>
<summary><strong>Embodied AI: From LLMs to World Models</strong> - Tongtong Feng, Xin Wang, Yu-Gang Jiang, Wenwu Zhu - [[pdf]](https://arxiv.org/pdf/2509.20021)</summary>

**Abstract:** Embodied Artificial Intelligence (AI) is an intelligent system paradigm for achieving Artificial General Intelligence (AGI), serving as the cornerstone for various applications and driving the evolution from cyberspace to physical systems. Recent breakthroughs in Large Language Models (LLMs) and World Models (WMs) have drawn significant attention for embodied AI. On the one hand, LLMs empower embodied AI via semantic reasoning and task decomposition, bringing high-level natural language instructions and low-level natural language actions into embodied cognition. On the other hand, WMs empower embodied AI by building internal representations and future predictions of the external world, facilitating physical law-compliant embodied interactions. As such, this paper comprehensively explores the literature in embodied AI from basics to advances, covering both LLM driven and WM driven works. In particular, we first present the history, key technologies, key components, and hardware systems of embodied AI, as well as discuss its development via looking from unimodal to multimodal angle. We then scrutinize the two burgeoning fields of embodied AI, i.e., embodied AI with LLMs/multimodal LLMs (MLLMs) and embodied AI with WMs, meticulously delineating their indispensable roles in end-to-end embodied cognition and physical laws-driven embodied interactions. Building upon the above advances, we further share our insights on the necessity of the joint MLLM-WM driven embodied AI architecture, shedding light on its profound significance in enabling complex tasks within physical worlds. In addition, we examine representative applications of embodied AI, demonstrating its wide applicability in real-world scenarios. Last but not least, we point out future research directions of embodied AI that deserve further investigation.

**arXiv ID:** 2509.20021
</details>

<details>
<summary><strong>Emergent Risk Awareness in Rational Agents under Resource Constraints</strong> - Daniel Jarne Ornia, Nicholas Bishop, Joel Dyer, Wei-Chen Lee, Ani Calinescu, Doyne Farmer, Michael Wooldridge - [[pdf]](https://arxiv.org/pdf/2505.23436)</summary>

**Abstract:** Advanced reasoning models with agentic capabilities (AI agents) are deployed to interact with humans and to solve sequential decision-making problems under (approximate) utility functions and internal models. When such problems have resource or failure constraints where action sequences may be forcibly terminated once resources are exhausted, agents face implicit trade-offs that reshape their utility-driven (rational) behaviour. Additionally, since these agents are typically commissioned by a human principal to act on their behalf, asymmetries in constraint exposure can give rise to previously unanticipated misalignment between human objectives and agent incentives. We formalise this setting through a survival bandit framework, provide theoretical and empirical results that quantify the impact of survival-driven preference shifts, identify conditions under which misalignment emerges and propose mechanisms to mitigate the emergence of risk-seeking or risk-averse behaviours. As a result, this work aims to increase understanding and interpretability of emergent behaviours of AI agents operating under such survival pressure, and offer guidelines for safely deploying such AI systems in critical resource-limited environments.

**arXiv ID:** 2505.23436
</details>

<details>
<summary><strong>Anatomy of a Feeling: Narrating Embodied Emotions via Large Vision-Language Models</strong> - Mohammad Saim, Phan Anh Duong, Cat Luong, Aniket Bhanderi, Tianyu Jiang - [[pdf]](https://arxiv.org/pdf/2509.19595)</summary>

**Abstract:** The embodiment of emotional reactions from body parts contains rich information about our affective experiences. We propose a framework that utilizes state-of-the-art large vision-language models (LVLMs) to generate Embodied LVLM Emotion Narratives (ELENA). These are well-defined, multi-layered text outputs, primarily comprising descriptions that focus on the salient body parts involved in emotional reactions. We also employ attention maps and observe that contemporary models exhibit a persistent bias towards the facial region. Despite this limitation, we observe that our employed framework can effectively recognize embodied emotions in face-masked images, outperforming baselines without any fine-tuning. ELENA opens a new trajectory for embodied emotion analysis across the modality of vision and enriches modeling in an affect-aware setting.

**arXiv ID:** 2509.19595
</details>

<details>
<summary><strong>VisualTrap: A Stealthy Backdoor Attack on GUI Agents via Visual Grounding Manipulation</strong> - Ziang Ye, Yang Zhang, Wentao Shi, Xiaoyu You, Fuli Feng, Tat-Seng Chua - [[pdf]](https://arxiv.org/pdf/2507.06899)</summary>

**Abstract:** Graphical User Interface (GUI) agents powered by Large Vision-Language Models (LVLMs) have emerged as a revolutionary approach to automating human-machine interactions, capable of autonomously operating personal devices (e.g., mobile phones) or applications within the device to perform complex real-world tasks in a human-like manner. However, their close integration with personal devices raises significant security concerns, with many threats, including backdoor attacks, remaining largely unexplored. This work reveals that the visual grounding of GUI agent-mapping textual plans to GUI elements-can introduce vulnerabilities, enabling new types of backdoor attacks. With backdoor attack targeting visual grounding, the agent's behavior can be compromised even when given correct task-solving plans. To validate this vulnerability, we propose VisualTrap, a method that can hijack the grounding by misleading the agent to locate textual plans to trigger locations instead of the intended targets. VisualTrap uses the common method of injecting poisoned data for attacks, and does so during the pre-training of visual grounding to ensure practical feasibility of attacking. Empirical results show that VisualTrap can effectively hijack visual grounding with as little as 5% poisoned data and highly stealthy visual triggers (invisible to the human eye); and the attack can be generalized to downstream tasks, even after clean fine-tuning. Moreover, the injected trigger can remain effective across different GUI environments, e.g., being trained on mobile/web and generalizing to desktop environments. These findings underscore the urgent need for further research on backdoor attack risks in GUI agents.

**arXiv ID:** 2507.06899
</details>

<details>
<summary><strong>Universal Camouflage Attack on Vision-Language Models for Autonomous Driving</strong> - Dehong Kong, Sifan Yu, Siyuan Liang, Jiawei Liang, Jianhou Gan, Aishan Liu, Wenqi Ren - [[pdf]](https://arxiv.org/pdf/2509.20196)</summary>

**Abstract:** Visual language modeling for automated driving is emerging as a promising research direction with substantial improvements in multimodal reasoning capabilities. Despite its advanced reasoning abilities, VLM-AD remains vulnerable to serious security threats from adversarial attacks, which involve misleading model decisions through carefully crafted perturbations. Existing attacks have obvious challenges: 1) Physical adversarial attacks primarily target vision modules. They are difficult to directly transfer to VLM-AD systems because they typically attack low-level perceptual components. 2) Adversarial attacks against VLM-AD have largely concentrated on the digital level. To address these challenges, we propose the first Universal Camouflage Attack (UCA) framework for VLM-AD. Unlike previous methods that focus on optimizing the logit layer, UCA operates in the feature space to generate physically realizable camouflage textures that exhibit strong generalization across different user commands and model architectures. Motivated by the observed vulnerability of encoder and projection layers in VLM-AD, UCA introduces a feature divergence loss (FDL) that maximizes the representational discrepancy between clean and adversarial images. In addition, UCA incorporates a multi-scale learning strategy and adjusts the sampling ratio to enhance its adaptability to changes in scale and viewpoint diversity in real-world scenarios, thereby improving training stability. Extensive experiments demonstrate that UCA can induce incorrect driving commands across various VLM-AD models and driving scenarios, significantly surpassing existing state-of-the-art attack methods (improving 30\% in 3-P metrics). Furthermore, UCA exhibits strong attack robustness under diverse viewpoints and dynamic conditions, indicating high potential for practical deployment.

**arXiv ID:** 2509.20196
</details>

<details>
<summary><strong>Autonomous Elemental Characterization Enabled by a Low Cost Robotic Platform Built Upon a Generalized Software Architecture</strong> - Xuan Cao, Yuxin Wu, Michael L. Whittaker - [[pdf]](https://arxiv.org/pdf/2509.19541)</summary>

**Abstract:** Despite the rapidly growing applications of robots in industry, the use of robots to automate tasks in scientific laboratories is less prolific due to lack of generalized methodologies and high cost of hardware. This paper focuses on the automation of characterization tasks necessary for reducing cost while maintaining generalization, and proposes a software architecture for building robotic systems in scientific laboratory environment. A dual-layer (this http URL and ROS) action server design is the basic building block, which facilitates the implementation of a web-based front end for user-friendly operations and the use of ROS Behavior Tree for convenient task planning and execution. A robotic platform for automating mineral and material sample characterization is built upon the architecture, with an open source, low-cost three-axis computer numerical control gantry system serving as the main robot. A handheld laser induced breakdown spectroscopy (LIBS) analyzer is integrated with a 3D printed adapter, enabling automated 2D chemical mapping. We demonstrate the utility of automated chemical mapping by scanning of the surface of a spodumene-bearing pegmatite core sample with a 1071-point dense hyperspectral map acquired at a rate of 1520 bits per second. Automated LIBS scanning enables controlled chemical quantification in the laboratory that complements field-based measurements acquired with the same handheld device, linking resource exploration and processing steps in the supply chain for lithium-based battery materials.

**arXiv ID:** 2509.19541
</details>

<details>
<summary><strong>Agentic Scene Policies: Unifying Space, Semantics, and Affordances for Robot Action</strong> - Sacha Morin, Kumaraditya Gupta, Mahtab Sandhu, Charlie Gauthier, Francesco Argenziano, Kirsty Ellis, Liam Paull - [[pdf]](https://arxiv.org/pdf/2509.19571)</summary>

**Abstract:** Executing open-ended natural language queries is a core problem in robotics. While recent advances in imitation learning and vision-language-actions models (VLAs) have enabled promising end-to-end policies, these models struggle when faced with complex instructions and new scenes. An alternative is to design an explicit scene representation as a queryable interface between the robot and the world, using query results to guide downstream motion planning. In this work, we present Agentic Scene Policies (ASP), an agentic framework that leverages the advanced semantic, spatial, and affordance-based querying capabilities of modern scene representations to implement a capable language-conditioned robot policy. ASP can execute open-vocabulary queries in a zero-shot manner by explicitly reasoning about object affordances in the case of more complex skills. Through extensive experiments, we compare ASP with VLAs on tabletop manipulation problems and showcase how ASP can tackle room-level queries through affordance-guided navigation, and a scaled-up scene representation. (Project page: this https URL)

**arXiv ID:** 2509.19571
</details>

<details>
<summary><strong>Towards Autonomous Robotic Electrosurgery via Thermal Imaging</strong> - Naveed D. Riaziat, Joseph Chen, Axel Krieger, Jeremy D. Brown - [[pdf]](https://arxiv.org/pdf/2509.19725)</summary>

**Abstract:** Electrosurgery is a surgical technique that can improve tissue cutting by reducing cutting force and bleeding. However, electrosurgery adds a risk of thermal injury to surrounding tissue. Expert surgeons estimate desirable cutting velocities based on experience but have no quantifiable reference to indicate if a particular velocity is optimal. Furthermore, prior demonstrations of autonomous electrosurgery have primarily used constant tool velocity, which is not robust to changes in electrosurgical tissue characteristics, power settings, or tool type. Thermal imaging feedback provides information that can be used to reduce thermal injury while balancing cutting force by controlling tool velocity. We introduce Thermography for Electrosurgical Rate Modulation via Optimization (ThERMO) to autonomously reduce thermal injury while balancing cutting force by intelligently controlling tool velocity. We demonstrate ThERMO in tissue phantoms and compare its performance to the constant velocity approach. Overall, ThERMO improves cut success rate by a factor of three and can reduce peak cutting force by a factor of two. ThERMO responds to varying environmental disturbances, reduces damage to tissue, and completes cutting tasks that would otherwise result in catastrophic failure for the constant velocity approach.

**arXiv ID:** 2509.19725
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (26 papers)</h2></summary>

<details>
<summary><strong>Evaluation-Aware Reinforcement Learning</strong> - Shripad Vilasrao Deshmukh, Will Schwarzer, Scott Niekum - [[pdf]](https://arxiv.org/pdf/2509.19464)</summary>

**Abstract:** Policy evaluation is often a prerequisite for deploying safety- and performance-critical systems. Existing evaluation approaches frequently suffer from high variance due to limited data and long-horizon tasks, or high bias due to unequal support or inaccurate environmental models. We posit that these challenges arise, in part, from the standard reinforcement learning (RL) paradigm of policy learning without explicit consideration of evaluation. As an alternative, we propose evaluation-aware reinforcement learning (EvA-RL), in which a policy is trained to maximize expected return while simultaneously minimizing expected evaluation error under a given value prediction scheme -- in other words, being "easy" to evaluate. We formalize a framework for EvA-RL and design an instantiation that enables accurate policy evaluation, conditioned on a small number of rollouts in an assessment environment that can be different than the deployment environment. However, our theoretical analysis and empirical results show that there is often a tradeoff between evaluation accuracy and policy performance when using a fixed value-prediction scheme within EvA-RL. To mitigate this tradeoff, we extend our approach to co-learn an assessment-conditioned state-value predictor alongside the policy. Empirical results across diverse discrete and continuous action domains demonstrate that EvA-RL can substantially reduce evaluation error while maintaining competitive returns. This work lays the foundation for a broad new class of RL methods that treat reliable evaluation as a first-class principle during training.

**arXiv ID:** 2509.19464
</details>

<details>
<summary><strong>UserRL: Training Interactive User-Centric Agent via Reinforcement Learning</strong> - Cheng Qian, Zuxin Liu, Akshara Prabhakar, Jielin Qiu, Zhiwei Liu, Haolin Chen, Shirley Kokane, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, Caiming Xiong, Huan Wang - [[pdf]](https://arxiv.org/pdf/2509.19736)</summary>

**Abstract:** Reinforcement learning (RL) has shown promise in training agentic models that move beyond static benchmarks to engage in dynamic, multi-turn interactions. Yet, the ultimate value of such agents lies in their ability to assist users, a setting where diversity and dynamics of user interaction pose challenges. In this work, we propose UserRL, a unified framework for training and evaluating user-centric abilities through standardized gym environments paired with simulated users. We systematically vary turn-level reward assignment and trajectory-level score calculation to analyze how different formulations affect learning under the GRPO algorithm. Our experiments across Qwen3 models reveal three key findings: (i) SFT cold start is critical for unlocking initial interaction ability and enabling sustained RL improvements; (ii) deliberate trajectory scoring yields more efficient and effective multi-turn interactions; and (iii) while stronger simulated users (e.g., GPT-4o) facilitates training, open-source simulators (e.g., Qwen3-32B) remain a cost-effective and transferable option. Together, these results highlight that careful design of reward shaping and user simulation choice is as crucial as model scale, and establish UserRL as a practical pathway for developing robust user-centric agentic models. All codes and data are public for future research.

**arXiv ID:** 2509.19736
</details>

<details>
<summary><strong>From Pheromones to Policies: Reinforcement Learning for Engineered Biological Swarms</strong> - Aymeric Vellinger, Nemanja Antonic, Elio Tuci - [[pdf]](https://arxiv.org/pdf/2509.20095)</summary>

**Abstract:** Swarm intelligence emerges from decentralised interactions among simple agents, enabling collective problem-solving. This study establishes a theoretical equivalence between pheromone-mediated aggregation in \celeg\ and reinforcement learning (RL), demonstrating how stigmergic signals function as distributed reward mechanisms. We model engineered nematode swarms performing foraging tasks, showing that pheromone dynamics mathematically mirror cross-learning updates, a fundamental RL algorithm. Experimental validation with data from literature confirms that our model accurately replicates empirical \celeg\ foraging patterns under static conditions. In dynamic environments, persistent pheromone trails create positive feedback loops that hinder adaptation by locking swarms into obsolete choices. Through computational experiments in multi-armed bandit scenarios, we reveal that introducing a minority of exploratory agents insensitive to pheromones restores collective plasticity, enabling rapid task switching. This behavioural heterogeneity balances exploration-exploitation trade-offs, implementing swarm-level extinction of outdated strategies. Our results demonstrate that stigmergic systems inherently encode distributed RL processes, where environmental signals act as external memory for collective credit assignment. By bridging synthetic biology with swarm robotics, this work advances programmable living systems capable of resilient decision-making in volatile environments.

**arXiv ID:** 2509.20095
</details>

<details>
<summary><strong>PEPS: Quantum-Inspired Reinforcement Learning for Coherent Reasoning Traces in LLMs</strong> - Venkat Margapuri, Garik Kazanjian, Naren Kosaraju - [[pdf]](https://arxiv.org/pdf/2509.20105)</summary>

**Abstract:** Large Language Models (LLMs) often struggle with maintaining coherent multi-step reasoning traces, particularly in tasks that require a structured logical flow. This work introduces a quantum-inspired approach to address the challenge by incorporating a fidelity-based reward derived from Projected Entangled Pair States (PEPS) into Proximal Policy Optimization. Unlike prior approaches that use direct supervision or contrastive objectives, the proposed method guides learning through structural consistency, offering a novel approach to enforce global coherence in generated reasoning traces. The proposed framework is evaluated using multiple coherence-determining metrics on diverse datasets such as GSM8K, StrategyQA, and EntailmentBank spanning arithmetic, intuitive, and entailment-based reasoning. Results show that the proposed quantum-inspired approach offers significant improvements over supervised, contrastive, and pretrained baseline approaches, highlighting the effectiveness of quantum-inspired fidelity as a foundation to improve reasoning trace coherence in LLMs.

**arXiv ID:** 2509.20105
</details>

<details>
<summary><strong>Wavelet Fourier Diffuser: Frequency-Aware Diffusion Model for Reinforcement Learning</strong> - Yifu Luo, Yongzhe Chang, Xueqian Wang - [[pdf]](https://arxiv.org/pdf/2509.19305)</summary>

**Abstract:** Diffusion probability models have shown significant promise in offline reinforcement learning by directly modeling trajectory sequences. However, existing approaches primarily focus on time-domain features while overlooking frequency-domain features, leading to frequency shift and degraded performance according to our observation. In this paper, we investigate the RL problem from a new perspective of the frequency domain. We first observe that time-domain-only approaches inadvertently introduce shifts in the low-frequency components of the frequency domain, which results in trajectory instability and degraded performance. To address this issue, we propose Wavelet Fourier Diffuser (WFDiffuser), a novel diffusion-based RL framework that integrates Discrete Wavelet Transform to decompose trajectories into low- and high-frequency components. To further enhance diffusion modeling for each component, WFDiffuser employs Short-Time Fourier Transform and cross attention mechanisms to extract frequency-domain features and facilitate cross-frequency interaction. Extensive experiment results on the D4RL benchmark demonstrate that WFDiffuser effectively mitigates frequency shift, leading to smoother, more stable trajectories and improved decision-making performance over existing methods.

**arXiv ID:** 2509.19305
</details>

<details>
<summary><strong>CSIYOLO: An Intelligent CSI-based Scatter Sensing Framework for Integrated Sensing and Communication Systems</strong> - Xudong Zhang, Jingbo Tan, Zhizhen Ren, Jintao Wang, Yihua Ma, Jian Song - [[pdf]](https://arxiv.org/pdf/2509.19335)</summary>

**Abstract:** ISAC is regarded as a promising technology for next-generation communication systems, enabling simultaneous data transmission and target sensing. Among various tasks in ISAC, scatter sensing plays a crucial role in exploiting the full potential of ISAC and supporting applications such as autonomous driving and low-altitude economy. However, most existing methods rely on either waveform and hardware modifications or traditional signal processing schemes, leading to poor compatibility with current communication systems and limited sensing accuracy. To address these challenges, we propose CSIYOLO, a framework that performs scatter localization only using estimated CSI from a single base station-user equipment pair. This framework comprises two main components: anchor-based scatter parameter detection and CSI-based scatter localization. First, by formulating scatter parameter extraction as an image detection problem, we propose an anchor-based scatter parameter detection method inspired by You Only Look Once architectures. After that, a CSI-based localization algorithm is derived to determine scatter locations with extracted parameters. Moreover, to improve localization accuracy and implementation efficiency, we design an extendable network structure with task-oriented optimizations, enabling multi-scale anchor detection and better adaptation to CSI characteristics. A noise injection training strategy is further designed to enhance robustness against channel estimation errors. Since the proposed framework operates solely on estimated CSI without modifying waveforms or signal processing pipelines, it can be seamlessly integrated into existing communication systems as a plugin. Experiments show that our proposed method can significantly outperform existing methods in scatter localization accuracy with relatively low complexities under varying numbers of scatters and estimation errors.

**arXiv ID:** 2509.19335
</details>

<details>
<summary><strong>DAWM: Diffusion Action World Models for Offline Reinforcement Learning via Action-Inferred Transitions</strong> - Zongyue Li, Xiao Han, Yusong Li, Niklas Strauss, Matthias Schubert - [[pdf]](https://arxiv.org/pdf/2509.19538)</summary>

**Abstract:** Diffusion-based world models have demonstrated strong capabilities in synthesizing realistic long-horizon trajectories for offline reinforcement learning (RL). However, many existing methods do not directly generate actions alongside states and rewards, limiting their compatibility with standard value-based offline RL algorithms that rely on one-step temporal difference (TD) learning. While prior work has explored joint modeling of states, rewards, and actions to address this issue, such formulations often lead to increased training complexity and reduced performance in practice. We propose \textbf{DAWM}, a diffusion-based world model that generates future state-reward trajectories conditioned on the current state, action, and return-to-go, paired with an inverse dynamics model (IDM) for efficient action inference. This modular design produces complete synthetic transitions suitable for one-step TD-based offline RL, enabling effective and computationally efficient training. Empirically, we show that conservative offline RL algorithms such as TD3BC and IQL benefit significantly from training on these augmented trajectories, consistently outperforming prior diffusion-based baselines across multiple tasks in the D4RL benchmark.

**arXiv ID:** 2509.19538
</details>

<details>
<summary><strong>Advancing Speech Summarization in Multi-modal LLMs with Reinforcement Learning</strong> - Shaoshi Ling, Gang Liu, Guoli Ye, Jinyu Li - [[pdf]](https://arxiv.org/pdf/2509.19631)</summary>

**Abstract:** Speech summarization is a critical component of spoken content understanding, particularly in the era of rapidly growing spoken and audiovisual data. Recent advances in multi-modal large language models (MLLMs), leveraging the power of LLMs, enable generating textual summaries directly from speech without intermediate transcriptions, while supporting controllable styles and zero-shot generalization. However, open-source MLLMs continue to lag behind the state-of-the-art text-based LLMs, limiting their practical deployment for speech summarization. In this work, we present a novel multi-stage reinforcement learning training framework to enhance the speech summarization capabilities in MLLMs. Our model delivers substantial improvements over strong baselines, outperforms much larger MLLMs, and significantly narrows the gap with state-of-the-art text-based LLMs.

**arXiv ID:** 2509.19631
</details>

<details>
<summary><strong>RDAR: Reward-Driven Agent Relevance Estimation for Autonomous Driving</strong> - Carlo Bosio, Greg Woelki, Noureldin Hendy, Nicholas Roy, Byungsoo Kim - [[pdf]](https://arxiv.org/pdf/2509.19789)</summary>

**Abstract:** Human drivers focus only on a handful of agents at any one time. On the other hand, autonomous driving systems process complex scenes with numerous agents, regardless of whether they are pedestrians on a crosswalk or vehicles parked on the side of the road. While attention mechanisms offer an implicit way to reduce the input to the elements that affect decisions, existing attention mechanisms for capturing agent interactions are quadratic, and generally computationally expensive. We propose RDAR, a strategy to learn per-agent relevance -- how much each agent influences the behavior of the controlled vehicle -- by identifying which agents can be excluded from the input to a pre-trained behavior model. We formulate the masking procedure as a Markov Decision Process where the action consists of a binary mask indicating agent selection. We evaluate RDAR on a large-scale driving dataset, and demonstrate its ability to learn an accurate numerical measure of relevance by achieving comparable driving performance, in terms of overall progress, safety and performance, while processing significantly fewer agents compared to a state of the art behavior model.

**arXiv ID:** 2509.19789
</details>

<details>
<summary><strong>Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving</strong> - Pengxiang Li, Yinan Zheng, Yue Wang, Huimin Wang, Hang Zhao, Jingjing Liu, Xianyuan Zhan, Kun Zhan, Xianpeng Lang - [[pdf]](https://arxiv.org/pdf/2509.20109)</summary>

**Abstract:** End-to-End (E2E) solutions have emerged as a mainstream approach for autonomous driving systems, with Vision-Language-Action (VLA) models representing a new paradigm that leverages pre-trained multimodal knowledge from Vision-Language Models (VLMs) to interpret and interact with complex real-world environments. However, these methods remain constrained by the limitations of imitation learning, which struggles to inherently encode physical rules during training. Existing approaches often rely on complex rule-based post-refinement, employ reinforcement learning that remains largely limited to simulation, or utilize diffusion guidance that requires computationally expensive gradient calculations. To address these challenges, we introduce ReflectDrive, a novel learning-based framework that integrates a reflection mechanism for safe trajectory generation via discrete diffusion. We first discretize the two-dimensional driving space to construct an action codebook, enabling the use of pre-trained Diffusion Language Models for planning tasks through fine-tuning. Central to our approach is a safety-aware reflection mechanism that performs iterative self-correction without gradient computation. Our method begins with goal-conditioned trajectory generation to model multi-modal driving behaviors. Based on this, we apply local search methods to identify unsafe tokens and determine feasible solutions, which then serve as safe anchors for inpainting-based regeneration. Evaluated on the NAVSIM benchmark, ReflectDrive demonstrates significant advantages in safety-critical trajectory generation, offering a scalable and reliable solution for autonomous driving systems.

**arXiv ID:** 2509.20109
</details>

<details>
<summary><strong>Embedding Domain Knowledge for Large Language Models via Reinforcement Learning from Augmented Generation</strong> - Chaojun Nie, Jun Zhou, Guanxiang Wang, Shisong Wud, Zichen Wang - [[pdf]](https://arxiv.org/pdf/2509.20162)</summary>

**Abstract:** Large language models (LLMs) often exhibit limited performance on domain-specific tasks due to the natural disproportionate representation of specialized information in their training data and the static nature of these datasets. Knowledge scarcity and temporal lag create knowledge gaps for domain applications. While post-training on domain datasets can embed knowledge into models, existing approaches have some limitations. Continual Pre-Training (CPT) treats all tokens in domain documents with equal importance, failing to prioritize critical knowledge points, while supervised fine-tuning (SFT) with question-answer pairs struggles to develop the coherent knowledge structures necessary for complex reasoning tasks. To address these challenges, we propose Reinforcement Learning from Augmented Generation (RLAG). Our approach iteratively cycles between sampling generations and optimizing the model through calculated rewards, effectively embedding critical and contextually coherent domain knowledge. We select generated outputs with the highest log probabilities as the sampling result, then compute three tailored reward metrics to guide the optimization process. To comprehensively evaluate domain expertise, we assess answer accuracy and the rationality of explanations generated for correctly answered questions. Experimental results across medical, legal, astronomy, and current events datasets demonstrate that our proposed method significantly outperforms baseline approaches. Our code and data are open sourced at this https URL.

**arXiv ID:** 2509.20162
</details>

<details>
<summary><strong>Plan Verification for LLM-Based Embodied Task Completion Agents</strong> - Ananth Hariharan, Vardhan Dongre, Dilek Hakkani-Tür, Gokhan Tur - [[pdf]](https://arxiv.org/pdf/2509.02761)</summary>

**Abstract:** Large language model (LLM) based task plans and corresponding human demonstrations for embodied AI may be noisy, with unnecessary actions, redundant navigation, and logical errors that reduce policy quality. We propose an iterative verification framework in which a Judge LLM critiques action sequences and a Planner LLM applies the revisions, yielding progressively cleaner and more spatially coherent trajectories. Unlike rule-based approaches, our method relies on natural language prompting, enabling broad generalization across error types including irrelevant actions, contradictions, and missing steps. On a set of manually annotated actions from the TEACh embodied AI dataset, our framework achieves up to 90% recall and 100% precision across four state-of-the-art LLMs (GPT o4-mini, DeepSeek-R1, Gemini 2.5, LLaMA 4 Scout). The refinement loop converges quickly, with 96.5% of sequences requiring at most three iterations, while improving both temporal efficiency and spatial action organization. Crucially, the method preserves human error-recovery patterns rather than collapsing them, supporting future work on robust corrective behavior. By establishing plan verification as a reliable LLM capability for spatial planning and action refinement, we provide a scalable path to higher-quality training data for imitation learning in embodied AI.

**arXiv ID:** 2509.02761
</details>

<details>
<summary><strong>Inverse Reinforcement Learning with Dynamic Reward Scaling for LLM Alignment</strong> - Ruoxi Cheng, Haoxuan Ma, Weixin Wang, Ranjie Duan, Jiexi Liu, Xiaoshuang Jia, Simeng Qin, Xiaochun Cao, Yang Liu, Xiaojun Jia - [[pdf]](https://arxiv.org/pdf/2503.18991)</summary>

**Abstract:** Alignment is vital for safely deploying large language models (LLMs). Existing techniques are either reward-based (train a reward model on preference pairs and optimize with reinforcement learning) or reward-free (directly fine-tune on ranked outputs). Recent research shows that well-tuned reward-based pipelines remain robust, and single-response demonstrations can outperform pairwise preference data. However, two challenges persist: (1) imbalanced safety datasets that overrepresent common hazards while neglecting long-tail threats; and (2) static reward models that ignore task difficulty, limiting optimization efficiency and attainable gains. We propose DR-IRL (Dynamically adjusting Rewards through Inverse Reinforcement Learning). We first train category-specific reward models using a balanced safety dataset covering seven harmful categories via IRL. Then we enhance Group Relative Policy Optimization (GRPO) by introducing dynamic reward scaling--adjusting rewards by task difficulty--data-level hardness by text encoder cosine similarity, model-level responsiveness by reward gaps. Extensive experiments across various benchmarks and LLMs demonstrate that DR-IRL outperforms all baseline methods in safety alignment while maintaining usefulness.

**arXiv ID:** 2503.18991
</details>

<details>
<summary><strong>VCRL: Variance-based Curriculum Reinforcement Learning for Large Language Models</strong> - Guochao Jiang, Wenfeng Feng, Guofeng Quan, Chuzhan Hao, Yuewei Zhang, Guohua Liu, Hao Wang - [[pdf]](https://arxiv.org/pdf/2509.19803)</summary>

**Abstract:** Policy-based reinforcement learning currently plays an important role in improving LLMs on mathematical reasoning tasks. However, existing rollout-based reinforcement learning methods (GRPO, DAPO, GSPO, etc.) fail to explicitly consider LLMs' learning ability for samples of different difficulty levels, which is contrary to the human cognitive process of mathematical reasoning tasks from easy to difficult. Intuitively, we find that the variance of the rollout group's reward in RLVR partly reflects the difficulty of the current sample for LLMs. Samples that are too easy or too difficult have a lower variance, while samples with moderate difficulty have a higher variance. Based on this, we propose VCRL, a curriculum reinforcement learning framework that dynamically controls the difficulty of training samples based on the variance of group rewards. Experiments on five mathematical benchmarks and two models reveal the advantages of VCRL over the current LLM RL baselines.

**arXiv ID:** 2509.19803
</details>

<details>
<summary><strong>DISCO Balances the Scales: Adaptive Domain- and Difficulty-Aware Reinforcement Learning on Imbalanced Data</strong> - Yuhang Zhou, Jing Zhu, Shengyi Qian, Zhuokai Zhao, Xiyao Wang, Xiaoyu Liu, Ming Li, Paiheng Xu, Wei Ai, Furong Huang - [[pdf]](https://arxiv.org/pdf/2505.15074)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly aligned with human preferences through Reinforcement Learning from Human Feedback (RLHF). Among RLHF methods, Group Relative Policy Optimization (GRPO) has gained attention for its simplicity and strong performance, notably eliminating the need for a learned value function. However, GRPO implicitly assumes a balanced domain distribution and uniform semantic alignment across groups, assumptions that rarely hold in real-world datasets. When applied to multi-domain, imbalanced data, GRPO disproportionately optimizes for dominant domains, neglecting underrepresented ones and resulting in poor generalization and fairness. We propose Domain-Informed Self-Consistency Policy Optimization (DISCO), a principled extension to GRPO that addresses inter-group imbalance with two key innovations. Domain-aware reward scaling counteracts frequency bias by reweighting optimization based on domain prevalence. Difficulty-aware reward scaling leverages prompt-level self-consistency to identify and prioritize uncertain prompts that offer greater learning value. Together, these strategies promote more equitable and effective policy learning across domains. Extensive experiments across multiple LLMs and skewed training distributions show that DISCO improves generalization, outperforms existing GRPO variants by 5% on Qwen3 models, and sets new state-of-the-art results on multi-domain alignment benchmarks. Our code and data are available at this https URL.

**arXiv ID:** 2505.15074
</details>

<details>
<summary><strong>BoreaRL: A Multi-Objective Reinforcement Learning Environment for Climate-Adaptive Boreal Forest Management</strong> - Kevin Bradley Dsouza, Enoch Ofosu, Daniel Chukwuemeka Amaogu, Jérôme Pigeon, Richard Boudreault, Pooneh Maghoul, Juan Moreno-Cruz, Yuri Leonenko - [[pdf]](https://arxiv.org/pdf/2509.19846)</summary>

**Abstract:** Boreal forests store 30-40% of terrestrial carbon, much in climate-vulnerable permafrost soils, making their management critical for climate mitigation. However, optimizing forest management for both carbon sequestration and permafrost preservation presents complex trade-offs that current tools cannot adequately address. We introduce $\textbf{BoreaRL}$, the first multi-objective reinforcement learning environment for climate-adaptive boreal forest management, featuring a physically-grounded simulator of coupled energy, carbon, and water fluxes. BoreaRL supports two training paradigms: site-specific mode for controlled studies and generalist mode for learning robust policies under environmental stochasticity. Through evaluation of multi-objective RL algorithms, we reveal a fundamental asymmetry in learning difficulty: carbon objectives are significantly easier to optimize than thaw (permafrost preservation) objectives, with thaw-focused policies showing minimal learning progress across both paradigms. In generalist settings, standard preference-conditioned approaches fail entirely, while a naive curriculum learning approach achieves superior performance by strategically selecting training episodes. Analysis of learned strategies reveals distinct management philosophies, where carbon-focused policies favor aggressive high-density coniferous stands, while effective multi-objective policies balance species composition and density to protect permafrost while maintaining carbon gains. Our results demonstrate that robust climate-adaptive forest management remains challenging for current MORL methods, establishing BoreaRL as a valuable benchmark for developing more effective approaches. We open-source BoreaRL to accelerate research in multi-objective RL for climate applications.

**arXiv ID:** 2509.19846
</details>

<details>
<summary><strong>Diffusion Classifier-Driven Reward for Offline Preference-based Reinforcement Learning</strong> - Teng Pang, Bingzheng Wang, Guoqiang Wu, Yilong Yin - [[pdf]](https://arxiv.org/pdf/2503.01143)</summary>

**Abstract:** Offline preference-based reinforcement learning (PbRL) mitigates the need for reward definition, aligning with human preferences via preference-driven reward feedback without interacting with the environment. However, trajectory-wise preference labels are difficult to meet the precise learning of step-wise reward, thereby affecting the performance of downstream algorithms. To alleviate the insufficient step-wise reward caused by trajectory-wise preferences, we propose a novel preference-based reward acquisition method: Diffusion Preference-based Reward (DPR). DPR directly treats step-wise preference-based reward acquisition as a binary classification and utilizes the robustness of diffusion classifiers to infer step-wise rewards discriminatively. In addition, to further utilize trajectory-wise preference information, we propose Conditional Diffusion Preference-based Reward (C-DPR), which conditions on trajectory-wise preference labels to enhance reward inference. We apply the above methods to existing offline RL algorithms, and a series of experimental results demonstrate that the diffusion classifier-driven reward outperforms the previous reward acquisition method with the Bradley-Terry model.

**arXiv ID:** 2503.01143
</details>

<details>
<summary><strong>Kalman Filter Enhanced GRPO for Reinforcement Learning-Based Language Model Reasoning</strong> - Hu Wang, Congbo Ma, Ian Reid, Mohammad Yaqub - [[pdf]](https://arxiv.org/pdf/2505.07527)</summary>

**Abstract:** The advantage function is a central concept in RL that helps reduce variance in policy gradient estimates. Recently, for language modeling, Group Relative Policy Optimization (GRPO) was proposed to compute the advantage for each output by subtracting the mean reward, as the baseline, for all outputs in the group. However, it can lead to high variance when the reward advantage is inaccurately predicted. In this work, we propose Kalman Filter Enhanced Group Relative Policy Optimization (KRPO) model, by using lightweight Kalman filtering to dynamically estimate the latent reward baseline and uncertainty. This filtering technique replaces the naive group mean, enabling more adaptive advantage normalization. Our method does not require additional learned parameters over GRPO. This approach offers a simple yet effective way to incorporate multiple outputs of GRPO into advantage estimation, improving policy optimization in settings where highly dynamic reward signals are difficult to model for language models. Through the accuracies and rewards obtained from math question answering and reasoning, we show that using a more adaptive advantage estimation model, KRPO can improve the stability and performance of GRPO. The code is available at this https URL.

**arXiv ID:** 2505.07527
</details>

<details>
<summary><strong>CANDLE: A Cross-Modal Agentic Knowledge Distillation Framework for Interpretable Sarcopenia Diagnosis</strong> - Yuqi Jin, Zhenhao Shuai, Zihan Hu, Weiteng Zhang, Weihao Xie, Jianwei Shuai, Xian Shen, Zhen Feng - [[pdf]](https://arxiv.org/pdf/2507.21179)</summary>

**Abstract:** Background and Aims: Large language models (LLMs) have shown remarkable generalization and transfer capabilities by learning from vast corpora of text and web data. Their semantic representations allow cross-task knowledge transfer and reasoning, offering promising opportunities for data-scarce and heterogeneous domains such as clinical medicine. Yet, in diagnostic tasks like sarcopenia, major challenges remain: interpretability, transparency, and deployment efficiency. Traditional machine learning (TML) models provide stable performance and feature-level attribution, ensuring traceable and auditable decision logic, but lack semantic breadth. Conversely, LLMs enable flexible inference but often function as opaque predictors. Existing integration strategies remain shallow, rarely embedding the structured reasoning of TML into LLM inference. Methods: Using sarcopenia diagnosis as a case study, SHapley Additive exPlanations (SHAP) were extracted from a baseline XGBoost model and transformed into structured, LLM-compatible representations. An actor-critic reinforcement learning (RL) strategy guided the LLM to reason over these SHAP-based inputs, producing calibrated rationales and refined decision rules. The distilled reasoning was consolidated into a structured knowledge repository and deployed via retrieval-augmented generation (RAG) for case-based inference. Results: (Omitted here.) Conclusion: By coupling SHAP-derived statistical evidence with reinforcement-trained LLM reasoning, CANDLE mitigates the interpretability-performance trade-off, enhances predictive accuracy, and preserves high decision consistency. The framework offers a scalable approach to knowledge assetization of TML models, enabling interpretable, reproducible, and clinically aligned decision support in sarcopenia and potentially broader medical domains.

**arXiv ID:** 2507.21179
</details>

<details>
<summary><strong>UI-S1: Advancing GUI Automation via Semi-online Reinforcement Learning</strong> - Zhengxi Lu, Jiabo Ye, Fei Tang, Yongliang Shen, Haiyang Xu, Ziwei Zheng, Weiming Lu, Ming Yan, Fei Huang, Jun Xiao, Yueting Zhuang - [[pdf]](https://arxiv.org/pdf/2509.11543)</summary>

**Abstract:** Graphical User Interface (GUI) agents have demonstrated remarkable progress in automating complex user interface interactions through reinforcement learning. However, current approaches face a fundamental dilemma: offline RL enables stable training on pre-collected trajectories, but struggles with multi-step task execution for lack of trajectory-level reward signals; online RL captures these signals through environment interaction, but suffers from sparse rewards and prohibitive deployment costs. To address it, we present Semi-online Reinforcement Learning, a novel paradigm that simulates online RL on offline trajectories. During each rollout process, we preserve the original model output within the multi-turn dialogue, where a Patch Module adaptively recovers the divergence between rollout and expert trajectories. To capture long-term training signals, Semi-online RL introduces discounted future returns into the reward computation and optimizes the policy with weighted step-level and episode-level advantages. We further introduce Semi-Online Performance (SOP), a metric that aligns better with true online performance, serving as a practical and effective proxy for real-world evaluation. Experiments show that ours Semi-online RL achieves SOTA performance among 7B models across four dynamic benchmarks, with significant gains over the base model (e.g., +12.0% on AndroidWorld, +23.8% on AITW), demonstrating significant progress in bridging the gap between offline training efficiency and online multi-turn reasoning. The code is available at this https URL.

**arXiv ID:** 2509.11543
</details>

<details>
<summary><strong>Real-Time Reinforcement Learning for Dynamic Tasks with a Parallel Soft Robot</strong> - James Avtges, Jake Ketchum, Millicent Schlafly, Helena Young, Taekyoung Kim, Allison Pinosky, Ryan L. Truby, Todd D. Murphey - [[pdf]](https://arxiv.org/pdf/2509.19525)</summary>

**Abstract:** Closed-loop control remains an open challenge in soft robotics. The nonlinear responses of soft actuators under dynamic loading conditions limit the use of analytic models for soft robot control. Traditional methods of controlling soft robots underutilize their configuration spaces to avoid nonlinearity, hysteresis, large deformations, and the risk of actuator damage. Furthermore, episodic data-driven control approaches such as reinforcement learning (RL) are traditionally limited by sample efficiency and inconsistency across initializations. In this work, we demonstrate RL for reliably learning control policies for dynamic balancing tasks in real-time single-shot hardware deployments. We use a deformable Stewart platform constructed using parallel, 3D-printed soft actuators based on motorized handed shearing auxetic (HSA) structures. By introducing a curriculum learning approach based on expanding neighborhoods of a known equilibrium, we achieve reliable single-deployment balancing at arbitrary coordinates. In addition to benchmarking the performance of model-based and model-free methods, we demonstrate that in a single deployment, Maximum Diffusion RL is capable of learning dynamic balancing after half of the actuators are effectively disabled, by inducing buckling and by breaking actuators with bolt cutters. Training occurs with no prior data, in as fast as 15 minutes, with performance nearly identical to the fully-intact platform. Single-shot learning on hardware facilitates soft robotic systems reliably learning in the real world and will enable more diverse and capable soft robots.

**arXiv ID:** 2509.19525
</details>

<details>
<summary><strong>Chasing Stability: Humanoid Running via Control Lyapunov Function Guided Reinforcement Learning</strong> - Zachary Olkin, Kejun Li, William D. Compton, Aaron D. Ames - [[pdf]](https://arxiv.org/pdf/2509.19573)</summary>

**Abstract:** Achieving highly dynamic behaviors on humanoid robots, such as running, requires controllers that are both robust and precise, and hence difficult to design. Classical control methods offer valuable insight into how such systems can stabilize themselves, but synthesizing real-time controllers for nonlinear and hybrid dynamics remains challenging. Recently, reinforcement learning (RL) has gained popularity for locomotion control due to its ability to handle these complex dynamics. In this work, we embed ideas from nonlinear control theory, specifically control Lyapunov functions (CLFs), along with optimized dynamic reference trajectories into the reinforcement learning training process to shape the reward. This approach, CLF-RL, eliminates the need to handcraft and tune heuristic reward terms, while simultaneously encouraging certifiable stability and providing meaningful intermediate rewards to guide learning. By grounding policy learning in dynamically feasible trajectories, we expand the robot's dynamic capabilities and enable running that includes both flight and single support phases. The resulting policy operates reliably on a treadmill and in outdoor environments, demonstrating robustness to disturbances applied to the torso and feet. Moreover, it achieves accurate global reference tracking utilizing only on-board sensors, making a critical step toward integrating these dynamic motions into a full autonomy stack.

**arXiv ID:** 2509.19573
</details>

<details>
<summary><strong>Minimalistic Autonomous Stack for High-Speed Time-Trial Racing</strong> - Mahmoud Ali, Hassan Jardali, Youwei Yu, Durgakant Pushp, Lantao Liu - [[pdf]](https://arxiv.org/pdf/2509.19636)</summary>

**Abstract:** Autonomous racing has seen significant advancements, driven by competitions such as the Indy Autonomous Challenge (IAC) and the Abu Dhabi Autonomous Racing League (A2RL). However, developing an autonomous racing stack for a full-scale car is often constrained by limited access to dedicated test tracks, restricting opportunities for real-world validation. While previous work typically requires extended development cycles and significant track time, this paper introduces a minimalistic autonomous racing stack for high-speed time-trial racing that emphasizes rapid deployment and efficient system integration with minimal on-track testing. The proposed stack was validated on real speedways, achieving a top speed of 206 km/h within just 11 hours' practice run on the track with 325 km in total. Additionally, we present the system performance analysis, including tracking accuracy, vehicle dynamics, and safety considerations, offering insights for teams seeking to rapidly develop and deploy an autonomous racing stack with limited track access.

**arXiv ID:** 2509.19636
</details>

<details>
<summary><strong>Beyond Human Demonstrations: Diffusion-Based Reinforcement Learning to Generate Data for VLA Training</strong> - Rushuai Yang, Hangxing Wei, Ran Zhang, Zhiyuan Feng, Xiaoyu Chen, Tong Li, Chuheng Zhang, Li Zhao, Jiang Bian, Xiu Su, Yi Chen - [[pdf]](https://arxiv.org/pdf/2509.19752)</summary>

**Abstract:** Vision-language-action (VLA) models have shown strong generalization across tasks and embodiments; however, their reliance on large-scale human demonstrations limits their scalability owing to the cost and effort of manual data collection. Reinforcement learning (RL) offers a potential alternative to generate demonstrations autonomously, yet conventional RL algorithms often struggle on long-horizon manipulation tasks with sparse rewards. In this paper, we propose a modified diffusion policy optimization algorithm to generate high-quality and low-variance trajectories, which contributes to a diffusion RL-powered VLA training pipeline. Our algorithm benefits from not only the high expressiveness of diffusion models to explore complex and diverse behaviors but also the implicit regularization of the iterative denoising process to yield smooth and consistent demonstrations. We evaluate our approach on the LIBERO benchmark, which includes 130 long-horizon manipulation tasks, and show that the generated trajectories are smoother and more consistent than both human demonstrations and those from standard Gaussian RL policies. Further, training a VLA model exclusively on the diffusion RL-generated data achieves an average success rate of 81.9%, which outperforms the model trained on human data by +5.3% and that on Gaussian RL-generated data by +12.6%. The results highlight our diffusion RL as an effective alternative for generating abundant, high-quality, and low-variance demonstrations for VLA models.

**arXiv ID:** 2509.19752
</details>

<details>
<summary><strong>PersONAL: Towards a Comprehensive Benchmark for Personalized Embodied Agents</strong> - Filippo Ziliotto, Jelin Raphael Akkara, Alessandro Daniele, Lamberto Ballan, Luciano Serafini, Tommaso Campari - [[pdf]](https://arxiv.org/pdf/2509.19843)</summary>

**Abstract:** Recent advances in Embodied AI have enabled agents to perform increasingly complex tasks and adapt to diverse environments. However, deploying such agents in realistic human-centered scenarios, such as domestic households, remains challenging, particularly due to the difficulty of modeling individual human preferences and behaviors. In this work, we introduce PersONAL (PERSonalized Object Navigation And Localization, a comprehensive benchmark designed to study personalization in Embodied AI. Agents must identify, retrieve, and navigate to objects associated with specific users, responding to natural-language queries such as "find Lily's backpack". PersONAL comprises over 2,000 high-quality episodes across 30+ photorealistic homes from the HM3D dataset. Each episode includes a natural-language scene description with explicit associations between objects and their owners, requiring agents to reason over user-specific semantics. The benchmark supports two evaluation modes: (1) active navigation in unseen environments, and (2) object grounding in previously mapped scenes. Experiments with state-of-the-art baselines reveal a substantial gap to human performance, highlighting the need for embodied agents capable of perceiving, reasoning, and memorizing over personalized information; paving the way towards real-world assistive robot.

**arXiv ID:** 2509.19843
</details>

<details>
<summary><strong>When Your Boss Is an AI Bot: Exploring Opportunities and Risks of Manager Clone Agents in the Future Workplace</strong> - Qing Hu, Qing Xiao, Hancheng Cao, Hong Shen - [[pdf]](https://arxiv.org/pdf/2509.10993)</summary>

**Abstract:** As Generative AI (GenAI) becomes increasingly embedded in the workplace, managers are beginning to create Manager Clone Agents - AI-powered digital surrogates that are trained on their work communications and decision patterns to perform managerial tasks on their behalf. To investigate this emerging phenomenon, we conducted six design fiction workshops (n = 23) with managers and workers, in which participants co-created speculative scenarios and discussed how Manager Clone Agents might transform collaborative work. We identified four potential roles that participants envisioned for Manager Clone Agents: proxy presence, informational conveyor belt, productivity engine, and leadership amplifier, while highlighting concerns spanning individual, interpersonal, and organizational levels. We provide design recommendations envisioned by both parties for integrating Manager Clone Agents responsibly into the future workplace, emphasizing the need to prioritize workers' perspectives, strengthen interpersonal bonds, and enable flexible clone configuration.

**arXiv ID:** 2509.10993
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
