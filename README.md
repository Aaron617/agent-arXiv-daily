# Agent arXiv Daily

**Last Updated:** 2025-12-22 02:28:31

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
<summary><strong>Implementing a Sharia Chatbot as a Consultation Medium for Questions About Islam</strong> - Wisnu Uriawan, Aria Octavian Hamza, Ade Ripaldi Nuralim, Adi Purnama, Ahmad Juaeni Yunus, Anissya Auliani Supriadi Putri - [[pdf]](https://arxiv.org/pdf/2512.16644)</summary>

**Abstract:** This research presents the implementation of a Sharia-compliant chatbot as an interactive medium for consulting Islamic questions, leveraging Reinforcement Learning (Q-Learning) integrated with Sentence-Transformers for semantic embedding to ensure contextual and accurate responses. Utilizing the CRISP-DM methodology, the system processes a curated Islam QA dataset of 25,000 question-answer pairs from authentic sources like the Qur'an, Hadith, and scholarly fatwas, formatted in JSON for flexibility and scalability. The chatbot prototype, developed with a Flask API backend and Flutter-based mobile frontend, achieves 87% semantic accuracy in functional testing across diverse topics including fiqh, aqidah, ibadah, and muamalah, demonstrating its potential to enhance religious literacy, digital da'wah, and access to verified Islamic knowledge in the Industry 4.0 era. While effective for closed-domain queries, limitations such as static learning and dataset dependency highlight opportunities for future enhancements like continuous adaptation and multi-turn conversation support, positioning this innovation as a bridge between traditional Islamic scholarship and modern AI-driven consultation.

**arXiv ID:** 2512.16644
</details>

<details>
<summary><strong>BRAID: Bounded Reasoning for Autonomous Inference and Decisions</strong> - Armağan Amcalar, Eyup Cinar - [[pdf]](https://arxiv.org/pdf/2512.15959)</summary>

**Abstract:** Large Language Models (LLMs) exhibit nonlinear relationships between performance, cost, and token usage. This paper presents a quantitative study on structured prompting using BRAID (Bounded Reasoning for Au tonomous Inference and Decisions) across multiple GPT model tiers, eval uated on the AdvancedIF, GSM-Hard, and the SCALE MultiChallenge benchmark datasets. BRAID introduces a bounded reasoning framework using Mermaid-based instruction graphs that enable models to reason struc turally rather than through unbounded natural-language token expansion. We show that structured machine-readable prompts substantially increase reasoning accuracy and cost efficiency for agents in production systems. The findings establish BRAID as an effective and scalable technique for optimizing inference efficiency in autonomous agent systems. All datasets and detailed result logs are available at this https URL.

**arXiv ID:** 2512.15959
</details>

<details>
<summary><strong>Agent Tools Orchestration Leaks More: Dataset, Benchmark, and Mitigation</strong> - Yuxuan Qiao, Dongqin Liu, Hongchang Yang, Wei Zhou, Songlin Hu - [[pdf]](https://arxiv.org/pdf/2512.16310)</summary>

**Abstract:** Driven by Large Language Models, the single-agent, multi-tool architecture has become a popular paradigm for autonomous agents due to its simplicity and effectiveness. However, this architecture also introduces a new and severe privacy risk, which we term Tools Orchestration Privacy Risk (TOP-R), where an agent, to achieve a benign user goal, autonomously aggregates information fragments across multiple tools and leverages its reasoning capabilities to synthesize unexpected sensitive information. We provide the first systematic study of this risk. First, we establish a formal framework, attributing the risk's root cause to the agent's misaligned objective function: an overoptimization for helpfulness while neglecting privacy awareness. Second, we construct TOP-Bench, comprising paired leakage and benign scenarios, to comprehensively evaluate this risk. To quantify the trade-off between safety and robustness, we introduce the H-Score as a holistic metric. The evaluation results reveal that TOP-R is a severe risk: the average Risk Leakage Rate (RLR) of eight representative models reaches 90.24%, while the average H-Score is merely 0.167, with no model exceeding 0.3. Finally, we propose the Privacy Enhancement Principle (PEP) method, which effectively mitigates TOP-R, reducing the Risk Leakage Rate to 46.58% and significantly improving the H-Score to 0.624. Our work reveals both a new class of risk and inherent structural limitations in current agent architectures, while also offering feasible mitigation strategies.

**arXiv ID:** 2512.16310
</details>

<details>
<summary><strong>Enhancing Long-term RAG Chatbots with Psychological Models of Memory Importance and Forgetting</strong> - Ryuichi Sumida, Koji Inoue, Tatsuya Kawahara - [[pdf]](https://arxiv.org/pdf/2409.12524)</summary>

**Abstract:** While Retrieval-Augmented Generation (RAG) has shown promise in enhancing long-term conversations, the increasing memory load as conversations progress degrades retrieval accuracy. Drawing on psychological insights, we propose LUFY, a simple yet effective method that focuses on emotionally arousing memories and retains less than 10% of the conversation. In the user experiment, participants interacted with three types of RAG chatbots, each for 2 hours over 4 sessions, marking the most extensive assessment of a chatbot's long-term capabilities to date -- more than four times longer than any existing benchmark. The results demonstrate that prioritizing arousing memories while forgetting the majority of the conversation significantly enhances user experience. This study pushes the frontier of long-term conversations and highlights the importance of forgetting unimportant parts of conversations. Code and Dataset: this https URL, Hugginface Dataset:this https URL

**arXiv ID:** 2409.12524
</details>

<details>
<summary><strong>Easy Come, Easy Go? Examining the Perceptions and Learning Effects of LLM-based Chatbot in the Context of Search-as-Learning</strong> - Yeonsun Yang, Ahyeon Shin, Mincheol Kang, Jiheon Kang, Xu Wang, Jean Y. Song - [[pdf]](https://arxiv.org/pdf/2410.01396)</summary>

**Abstract:** The cognitive process of Search-as-Learning (SAL) is most effective when searching promotes active encoding of information. The rise of LLMs-based chatbots, which provide instant answers, introduces a trade-off between efficiency and depth of processing. Such answer-centric approaches accelerate information access, but they also raise concerns about shallower learning. To examine these issues in the context of SAL, we conducted a large-scale survey of educators and students to capture perceived risks and benefits of LLM-based chatbots. In addition, we adopted the encoding-storage paradigm to design a within-subjects experiment, where participants (N=92) engaged in SAL tasks using three different modalities: books, search engines, and chatbots. Our findings provide a counterintuitive insight into stakeholder concerns: while LLM-based chatbots and search engines validated perceived benefits on learning efficiency by outperforming book-based search in immediate conceptual understanding, they did not result in a long-term inferiority as feared. Our study provides insights for designing human-AI collaborative learning systems that promote cognitive engagement by balancing learning efficiency and long-term knowledge retention.

**arXiv ID:** 2410.01396
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (11 papers)</h2></summary>

<details>
<summary><strong>Beyond Training: Enabling Self-Evolution of Agents with MOBIMEM</strong> - Zibin Liu, Cheng Zhang, Xi Zhao, Yunfei Feng, Bingyu Bai, Dahu Feng, Erhu Feng, Yubin Xia, Haibo Chen - [[pdf]](https://arxiv.org/pdf/2512.15784)</summary>

**Abstract:** Large Language Model (LLM) agents are increasingly deployed to automate complex workflows in mobile and desktop environments. However, current model-centric agent architectures struggle to self-evolve post-deployment: improving personalization, capability, and efficiency typically requires continuous model retraining/fine-tuning, which incurs prohibitive computational overheads and suffers from an inherent trade-off between model accuracy and inference efficiency.
To enable iterative self-evolution without model retraining, we propose MOBIMEM, a memory-centric agent system. MOBIMEM first introduces three specialized memory primitives to decouple agent evolution from model weights: (1) Profile Memory uses a lightweight distance-graph (DisGraph) structure to align with user preferences, resolving the accuracy-latency trade-off in user profile retrieval; (2) Experience Memory employs multi-level templates to instantiate execution logic for new tasks, ensuring capability generalization; and (3) Action Memory records fine-grained interaction sequences, reducing the reliance on expensive model inference. Building upon this memory architecture, MOBIMEM further integrates a suite of OS-inspired services to orchestrate execution: a scheduler that coordinates parallel sub-task execution and memory operations; an agent record-and-replay (AgentRR) mechanism that enables safe and efficient action reuse; and a context-aware exception handling that ensures graceful recovery from user interruptions and runtime errors.
Evaluation on AndroidWorld and top-50 apps shows that MOBIMEM achieves 83.1% profile alignment with 23.83 ms retrieval time (280x faster than GraphRAG baselines), improves task success rates by up to 50.3%, and reduces end-to-end latency by up to 9x on mobile devices.

**arXiv ID:** 2512.15784
</details>

<details>
<summary><strong>WeMusic-Agent: Efficient Conversational Music Recommendation via Knowledge Internalization and Agentic Boundary Learning</strong> - Wendong Bi, Yirong Mao, Xianglong Liu, Kai Tian, Jian Zhang, Hanjie Wang, Wenhui Que - [[pdf]](https://arxiv.org/pdf/2512.16108)</summary>

**Abstract:** Personalized music recommendation in conversational scenarios usually requires a deep understanding of user preferences and nuanced musical context, yet existing methods often struggle with balancing specialized domain knowledge and flexible tool integration. This paper proposes WeMusic-Agent, a training framework for efficient LLM-based conversational music recommendation. By integrating the knowledge internalization and agentic boundary learning, the framework aims to teach the model to intelligently decide when to leverage internalized knowledge and when to call specialized tools (e.g., music retrieval APIs, music recommendation systems). Under this framework, we present WeMusic-Agent-M1, an agentic model that internalizes extensive musical knowledge via continued pretraining on 50B music-related corpus while acquiring the ability to invoke external tools when necessary. Additionally, considering the lack of open-source benchmarks for conversational music recommendation, we also construct a benchmark for personalized music recommendations derived from real-world data in WeChat Listen. This benchmark enables comprehensive evaluation across multiple dimensions, including relevance, personalization, and diversity of the recommendations. Experiments on real-world data demonstrate that WeMusic-Agent achieves significant improvements over existing models.

**arXiv ID:** 2512.16108
</details>

<details>
<summary><strong>Code-in-the-Loop Forensics: Agentic Tool Use for Image Forgery Detection</strong> - Fanrui Zhang, Qiang Zhang, Sizhuo Zhou, Jianwen Sun, Chuanhao Li, Jiaxin Ai, Yukang Feng, Yujie Zhang, Wenjie Li, Zizhen Li, Yifan Chang, Jiawei Liu, Kaipeng Zhang - [[pdf]](https://arxiv.org/pdf/2512.16300)</summary>

**Abstract:** Existing image forgery detection (IFD) methods either exploit low-level, semantics-agnostic artifacts or rely on multimodal large language models (MLLMs) with high-level semantic knowledge. Although naturally complementary, these two information streams are highly heterogeneous in both paradigm and reasoning, making it difficult for existing methods to unify them or effectively model their cross-level interactions. To address this gap, we propose ForenAgent, a multi-round interactive IFD framework that enables MLLMs to autonomously generate, execute, and iteratively refine Python-based low-level tools around the detection objective, thereby achieving more flexible and interpretable forgery analysis. ForenAgent follows a two-stage training pipeline combining Cold Start and Reinforcement Fine-Tuning to enhance its tool interaction capability and reasoning adaptability progressively. Inspired by human reasoning, we design a dynamic reasoning loop comprising global perception, local focusing, iterative probing, and holistic adjudication, and instantiate it as both a data-sampling strategy and a task-aligned process reward. For systematic training and evaluation, we construct FABench, a heterogeneous, high-quality agent-forensics dataset comprising 100k images and approximately 200k agent-interaction question-answer pairs. Experiments show that ForenAgent exhibits emergent tool-use competence and reflective reasoning on challenging IFD tasks when assisted by low-level tools, charting a promising route toward general-purpose IFD. The code will be released after the review process is completed.

**arXiv ID:** 2512.16300
</details>

<details>
<summary><strong>CitySeeker: How Do VLMS Explore Embodied Urban Navigation With Implicit Human Needs?</strong> - Siqi Wang, Chao Liang, Yunfan Gao, Erxin Yu, Sen Li, Yushi Li, Jing Li, Haofen Wang - [[pdf]](https://arxiv.org/pdf/2512.16755)</summary>

**Abstract:** Vision-Language Models (VLMs) have made significant progress in explicit instruction-based navigation; however, their ability to interpret implicit human needs (e.g., "I am thirsty") in dynamic urban environments remains underexplored. This paper introduces CitySeeker, a novel benchmark designed to assess VLMs' spatial reasoning and decision-making capabilities for exploring embodied urban navigation to address implicit needs. CitySeeker includes 6,440 trajectories across 8 cities, capturing diverse visual characteristics and implicit needs in 7 goal-driven scenarios. Extensive experiments reveal that even top-performing models (e.g., Qwen2.5-VL-32B-Instruct) achieve only 21.1% task completion. We find key bottlenecks in error accumulation in long-horizon reasoning, inadequate spatial cognition, and deficient experiential recall. To further analyze them, we investigate a series of exploratory strategies-Backtracking Mechanisms, Enriching Spatial Cognition, and Memory-Based Retrieval (BCR), inspired by human cognitive mapping's emphasis on iterative observation-reasoning cycles and adaptive path optimization. Our analysis provides actionable insights for developing VLMs with robust spatial intelligence required for tackling "last-mile" navigation challenges.

**arXiv ID:** 2512.16755
</details>

<details>
<summary><strong>GLOW: Graph-Language Co-Reasoning for Agentic Workflow Performance Prediction</strong> - Wei Guan, Jian Cao, Jinyu Cai, Qiqi Cai, Jianqi Gao, See-Kiong Ng - [[pdf]](https://arxiv.org/pdf/2512.15751)</summary>

**Abstract:** Agentic Workflows (AWs) have emerged as a promising paradigm for solving complex tasks. However, the scalability of automating their generation is severely constrained by the high cost and latency of execution-based evaluation. Existing AW performance prediction methods act as surrogates but fail to simultaneously capture the intricate topological dependencies and the deep semantic logic embedded in AWs. To address this limitation, we propose GLOW, a unified framework for AW performance prediction that combines the graph-structure modeling capabilities of GNNs with the reasoning power of LLMs. Specifically, we introduce a graph-oriented LLM, instruction-tuned on graph tasks, to extract topologically aware semantic features, which are fused with GNN-encoded structural representations. A contrastive alignment strategy further refines the latent space to distinguish high-quality AWs. Extensive experiments on FLORA-Bench show that GLOW outperforms state-of-the-art baselines in prediction accuracy and ranking utility.

**arXiv ID:** 2512.15751
</details>

<details>
<summary><strong>D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI</strong> - Suhwan Choi, Jaeyoon Jung, Haebin Seong, Minchan Kim, Minyeong Kim, Yongjun Cho, Yoonshik Kim, Yubeen Park, Youngjae Yu, Yunsung Lee - [[pdf]](https://arxiv.org/pdf/2510.05684)</summary>

**Abstract:** Large language models leverage internet-scale text data, yet embodied AI remains constrained by the prohibitive costs of physical trajectory collection. Desktop environments -- particularly gaming -- offer a compelling alternative: they provide rich sensorimotor interactions at scale while maintaining the structured observation-action coupling essential for embodied learning. We present D2E (Desktop to Embodied AI), a framework that demonstrates desktop interactions can serve as an effective pretraining substrate for robotics embodied AI tasks. Unlike prior work that remained domain-specific (e.g., VPT for Minecraft) or kept data proprietary (e.g., SIMA), D2E establishes a complete pipeline from scalable desktop data collection to verified transfer in embodied domains. Our framework comprises three components: (1) the OWA Toolkit that unifies diverse desktop interactions into a standardized format with 152x compression, (2) the Generalist-IDM that achieves strong zero-shot generalization across unseen games through timestamp-based event prediction, enabling internet-scale pseudo-labeling, and (3) VAPT that transfers desktop-pretrained representations to physical manipulation and navigation. Using 1.3K+ hours of data (259 hours of human demonstrations, and 1K+ hours of pseudo-labeled gameplay), we achieve a total of 96.6% success rate on LIBERO manipulation and 83.3% on CANVAS navigation benchmarks. This validates that sensorimotor primitives in digital interactions exhibit sufficient invariance to transfer meaningfully to physical embodied tasks, establishing desktop pretraining as a practical paradigm for robotics. We will make all our work public, including the OWA toolkit, datasets of human-collected and pseudo-labeled, and VAPT-trained models available at this https URL

**arXiv ID:** 2510.05684
</details>

<details>
<summary><strong>DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving</strong> - Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, Lu Hou, Lue Fan, Zhaoxiang Zhang - [[pdf]](https://arxiv.org/pdf/2510.12796)</summary>

**Abstract:** Scaling Vision-Language-Action (VLA) models on large-scale data offers a promising path to achieving a more generalized driving intelligence. However, VLA models are limited by a ``supervision deficit'': the vast model capacity is supervised by sparse, low-dimensional actions, leaving much of their representational power underutilized. To remedy this, we propose \textbf{DriveVLA-W0}, a training paradigm that employs world modeling to predict future images. This task generates a dense, self-supervised signal that compels the model to learn the underlying dynamics of the driving environment. We showcase the paradigm's versatility by instantiating it for two dominant VLA archetypes: an autoregressive world model for VLAs that use discrete visual tokens, and a diffusion world model for those operating on continuous visual features. Building on the rich representations learned from world modeling, we introduce a lightweight action expert to address the inference latency for real-time deployment. Extensive experiments on the NAVSIM v1/v2 benchmark and a 680x larger in-house dataset demonstrate that DriveVLA-W0 significantly outperforms BEV and VLA baselines. Crucially, it amplifies the data scaling law, showing that performance gains accelerate as the training dataset size increases.

**arXiv ID:** 2510.12796
</details>

<details>
<summary><strong>Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future</strong> - Tianshuai Hu, Xiaolu Liu, Song Wang, Yiyao Zhu, Ao Liang, Lingdong Kong, Guoyang Zhao, Zeying Gong, Jun Cen, Zhiyu Huang, Xiaoshuai Hao, Linfeng Li, Hang Song, Xiangtai Li, Jun Ma, Shaojie Shen, Jianke Zhu, Dacheng Tao, Ziwei Liu, Junwei Liang - [[pdf]](https://arxiv.org/pdf/2512.16760)</summary>

**Abstract:** Autonomous driving has long relied on modular "Perception-Decision-Action" pipelines, where hand-crafted interfaces and rule-based components often break down in complex or long-tailed scenarios. Their cascaded design further propagates perception errors, degrading downstream planning and control. Vision-Action (VA) models address some limitations by learning direct mappings from visual inputs to actions, but they remain opaque, sensitive to distribution shifts, and lack structured reasoning or instruction-following capabilities. Recent progress in Large Language Models (LLMs) and multimodal learning has motivated the emergence of Vision-Language-Action (VLA) frameworks, which integrate perception with language-grounded decision making. By unifying visual understanding, linguistic reasoning, and actionable outputs, VLAs offer a pathway toward more interpretable, generalizable, and human-aligned driving policies. This work provides a structured characterization of the emerging VLA landscape for autonomous driving. We trace the evolution from early VA approaches to modern VLA frameworks and organize existing methods into two principal paradigms: End-to-End VLA, which integrates perception, reasoning, and planning within a single model, and Dual-System VLA, which separates slow deliberation (via VLMs) from fast, safety-critical execution (via planners). Within these paradigms, we further distinguish subclasses such as textual vs. numerical action generators and explicit vs. implicit guidance mechanisms. We also summarize representative datasets and benchmarks for evaluating VLA-based driving systems and highlight key challenges and open directions, including robustness, interpretability, and instruction fidelity. Overall, this work aims to establish a coherent foundation for advancing human-compatible autonomous driving systems.

**arXiv ID:** 2512.16760
</details>

<details>
<summary><strong>Driving in Corner Case: A Real-World Adversarial Closed-Loop Evaluation Platform for End-to-End Autonomous Driving</strong> - Jiaheng Geng, Jiatong Du, Xinyu Zhang, Ye Li, Panqu Wang, Yanjun Huang - [[pdf]](https://arxiv.org/pdf/2512.16055)</summary>

**Abstract:** Safety-critical corner cases, difficult to collect in the real world, are crucial for evaluating end-to-end autonomous driving. Adversarial interaction is an effective method to generate such safety-critical corner cases. While existing adversarial evaluation methods are built for models operating in simplified simulation environments, adversarial evaluation for real-world end-to-end autonomous driving has been little explored. To address this challenge, we propose a closed-loop evaluation platform for end-to-end autonomous driving, which can generate adversarial interactions in real-world scenes. In our platform, the real-world image generator cooperates with an adversarial traffic policy to evaluate various end-to-end models trained on real-world data. The generator, based on flow matching, efficiently and stably generates real-world images according to the traffic environment information. The efficient adversarial surrounding vehicle policy is designed to model challenging interactions and create corner cases that current autonomous driving systems struggle to handle. Experimental results demonstrate that the platform can generate realistic driving images efficiently. Through evaluating the end-to-end models such as UniAD and VAD, we demonstrate that based on the adversarial policy, our platform evaluates the performance degradation of the tested model in corner cases. This result indicates that this platform can effectively detect the model's potential issues, which will facilitate the safety and robustness of end-to-end autonomous driving.

**arXiv ID:** 2512.16055
</details>

<details>
<summary><strong>Privacy-Aware Sharing of Raw Spatial Sensor Data for Cooperative Perception</strong> - Bangya Liu, Chengpo Yan, Chenghao Jiang, Suman Banerjee, Akarsh Prabhakara - [[pdf]](https://arxiv.org/pdf/2512.16265)</summary>

**Abstract:** Cooperative perception between vehicles is poised to offer robust and reliable scene understanding. Recently, we are witnessing experimental systems research building testbeds that share raw spatial sensor data for cooperative perception. While there has been a marked improvement in accuracies and is the natural way forward, we take a moment to consider the problems with such an approach for eventual adoption by automakers. In this paper, we first argue that new forms of privacy concerns arise and discourage stakeholders to share raw sensor data. Next, we present SHARP, a research framework to minimize privacy leakage and drive stakeholders towards the ambitious goal of raw data based cooperative perception. Finally, we discuss open questions for networked systems, mobile computing, perception researchers, industry and government in realizing our proposed framework.

**arXiv ID:** 2512.16265
</details>

<details>
<summary><strong>SNOW: Spatio-Temporal Scene Understanding with World Knowledge for Open-World Embodied Reasoning</strong> - Tin Stribor Sohn, Maximilian Dillitzer, Jason J. Corso, Eric Sax - [[pdf]](https://arxiv.org/pdf/2512.16461)</summary>

**Abstract:** Autonomous robotic systems require spatio-temporal understanding of dynamic environments to ensure reliable navigation and interaction. While Vision-Language Models (VLMs) provide open-world semantic priors, they lack grounding in 3D geometry and temporal dynamics. Conversely, geometric perception captures structure and motion but remains semantically sparse. We propose SNOW (Scene Understanding with Open-World Knowledge), a training-free and backbone-agnostic framework for unified 4D scene understanding that integrates VLM-derived semantics with point cloud geometry and temporal consistency. SNOW processes synchronized RGB images and 3D point clouds, using HDBSCAN clustering to generate object-level proposals that guide SAM2-based segmentation. Each segmented region is encoded through our proposed Spatio-Temporal Tokenized Patch Encoding (STEP), producing multimodal tokens that capture localized semantic, geometric, and temporal attributes. These tokens are incrementally integrated into a 4D Scene Graph (4DSG), which serves as 4D prior for downstream reasoning. A lightweight SLAM backend anchors all STEP tokens spatially in the environment, providing the global reference alignment, and ensuring unambiguous spatial grounding across time. The resulting 4DSG forms a queryable, unified world model through which VLMs can directly interpret spatial scene structure and temporal dynamics. Experiments on a diverse set of benchmarks demonstrate that SNOW enables precise 4D scene understanding and spatially grounded inference, thereby setting new state-of-the-art performance in several settings, highlighting the importance of structured 4D priors for embodied reasoning and autonomous robotics.

**arXiv ID:** 2512.16461
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>AMUSE: Audio-Visual Benchmark and Alignment Framework for Agentic Multi-Speaker Understanding</strong> - Sanjoy Chowdhury, Karren D. Yang, Xudong Liu, Fartash Faghri, Pavan Kumar Anasosalu Vasu, Oncel Tuzel, Dinesh Manocha, Chun-Liang Li, Raviteja Vemulapalli - [[pdf]](https://arxiv.org/pdf/2512.16250)</summary>

**Abstract:** Recent multimodal large language models (MLLMs) such as GPT-4o and Qwen3-Omni show strong perception but struggle in multi-speaker, dialogue-centric settings that demand agentic reasoning tracking who speaks, maintaining roles, and grounding events across time. These scenarios are central to multimodal audio-video understanding, where models must jointly reason over audio and visual streams in applications such as conversational video assistants and meeting analytics. We introduce AMUSE, a benchmark designed around tasks that are inherently agentic, requiring models to decompose complex audio-visual interactions into planning, grounding, and reflection steps. It evaluates MLLMs across three modes zero-shot, guided, and agentic and six task families, including spatio-temporal speaker grounding and multimodal dialogue summarization. Across all modes, current models exhibit weak multi-speaker reasoning and inconsistent behavior under both non-agentic and agentic evaluation. Motivated by the inherently agentic nature of these tasks and recent advances in LLM agents, we propose RAFT, a data-efficient agentic alignment framework that integrates reward optimization with intrinsic multimodal self-evaluation as reward and selective parameter adaptation for data and parameter efficient updates. Using RAFT, we achieve up to 39.52\% relative improvement in accuracy on our benchmark. Together, AMUSE and RAFT provide a practical platform for examining agentic reasoning in multimodal models and improving their capabilities.

**arXiv ID:** 2512.16250
</details>

<details>
<summary><strong>VET Your Agent: Towards Host-Independent Autonomy via Verifiable Execution Traces</strong> - Artem Grigor, Christian Schroeder de Witt, Simon Birnbach, Ivan Martinovic - [[pdf]](https://arxiv.org/pdf/2512.15892)</summary>

**Abstract:** Recent advances in large language models (LLMs) have enabled a new generation of autonomous agents that operate over sustained periods and manage sensitive resources on behalf of users. Trusted for their ability to act without direct oversight, such agents are increasingly considered in high-stakes domains including financial management, dispute resolution, and governance. Yet in practice, agents execute on infrastructure controlled by a host, who can tamper with models, inputs, or outputs, undermining any meaningful notion of autonomy.
We address this gap by introducing VET (Verifiable Execution Traces), a formal framework that achieves host-independent authentication of agent outputs and takes a step toward host-independent autonomy. Central to VET is the Agent Identity Document (AID), which specifies an agent's configuration together with the proof systems required for verification. VET is compositional: it supports multiple proof mechanisms, including trusted hardware, succinct cryptographic proofs, and notarized TLS transcripts (Web Proofs).
We implement VET for an API-based LLM agent and evaluate our instantiation on realistic workloads. We find that for today's black-box, secret-bearing API calls, Web Proofs appear to be the most practical choice, with overhead typically under 3$\times$ compared to direct API calls, while for public API calls, a lower-overhead TEE Proxy is often sufficient. As a case study, we deploy a verifiable trading agent that produces proofs for each decision and composes Web Proofs with a TEE Proxy. Our results demonstrate that practical, host-agnostic authentication is already possible with current technology, laying the foundation for future systems that achieve full host-independent autonomy.

**arXiv ID:** 2512.15892
</details>

<details>
<summary><strong>Meta-RL Induces Exploration in Language Agents</strong> - Yulun Jiang, Liangze Jiang, Damien Teney, Michael Moor, Maria Brbic - [[pdf]](https://arxiv.org/pdf/2512.16848)</summary>

**Abstract:** Reinforcement learning (RL) has enabled the training of large language model (LLM) agents to interact with the environment and to solve multi-turn long-horizon tasks. However, the RL-trained agents often struggle in tasks that require active exploration and fail to efficiently adapt from trial-and-error experiences. In this paper, we present LaMer, a general Meta-RL framework that enables LLM agents to actively explore and learn from the environment feedback at test time. LaMer consists of two key components: (i) a cross-episode training framework to encourage exploration and long-term rewards optimization; and (ii) in-context policy adaptation via reflection, allowing the agent to adapt their policy from task feedback signal without gradient update. Experiments across diverse environments show that LaMer significantly improves performance over RL baselines, with 11%, 14%, and 19% performance gains on Sokoban, MineSweeper and Webshop, respectively. Moreover, LaMer also demonstrates better generalization to more challenging or previously unseen tasks compared to the RL-trained agents. Overall, our results demonstrate that Meta-RL provides a principled approach to induce exploration in language agents, enabling more robust adaptation to novel environments through learned exploration strategies.

**arXiv ID:** 2512.16848
</details>

<details>
<summary><strong>Towards Pervasive Distributed Agentic Generative AI -- A State of The Art</strong> - Gianni Molinari, Fabio Ciravegna - [[pdf]](https://arxiv.org/pdf/2506.13324)</summary>

**Abstract:** The rapid advancement of intelligent agents and Large Language Models (LLMs) is reshaping the pervasive computing field. Their ability to perceive, reason, and act through natural language understanding enables autonomous problem-solving in complex pervasive environments, including the management of heterogeneous sensors, devices, and data. This survey outlines the architectural components of LLM agents (profiling, memory, planning, and action) and examines their deployment and evaluation across various scenarios. Than it reviews computational and infrastructural advancements (cloud to edge) in pervasive computing and how AI is moving in this field. It highlights state-of-the-art agent deployment strategies and applications, including local and distributed execution on resource-constrained devices. This survey identifies key challenges of these agents in pervasive computing such as architectural, energetic and privacy limitations. It finally proposes what we called "Agent as a Tool", a conceptual framework for pervasive agentic AI, emphasizing context awareness, modularity, security, efficiency and effectiveness.

**arXiv ID:** 2506.13324
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (17 papers)</h2></summary>

<details>
<summary><strong>Anubuddhi: A Multi-Agent AI System for Designing and Simulating Quantum Optics Experiments</strong> - S. K. Rithvik - [[pdf]](https://arxiv.org/pdf/2512.15736)</summary>

**Abstract:** We present Anubuddhi, a multi-agent AI system that designs and simulates quantum optics experiments from natural language prompts without requiring specialized programming knowledge. The system composes optical layouts by arranging components from a three-tier toolbox via semantic retrieval, then validates designs through physics simulation with convergent refinement. The architecture combines intent routing, knowledge-augmented generation, and dual-mode validation (QuTiP and FreeSim). We evaluated 13 experiments spanning fundamental optics (Hong-Ou-Mandel interference, Michelson/Mach-Zehnder interferometry, Bell states, delayed-choice quantum eraser), quantum information protocols (BB84 QKD, Franson interferometry, GHZ states, quantum teleportation, hyperentanglement), and advanced technologies (boson sampling, electromagnetically induced transparency, frequency conversion). The system achieves design-simulation alignment scores of 8--9/10, with simulations faithfully modeling intended physics. A critical finding distinguishes structural correctness from quantitative accuracy: high alignment confirms correct physics architecture, while numerical predictions require expert review. Free-form simulation outperformed constrained frameworks for 11/13 experiments, revealing that quantum optics diversity demands flexible mathematical representations. The system democratizes computational experiment design for research and pedagogy, producing strong initial designs users can iteratively refine through conversation.

**arXiv ID:** 2512.15736
</details>

<details>
<summary><strong>PDE-Agent: A toolchain-augmented multi-agent framework for PDE solving</strong> - Jianming Liu, Ren Zhu, Jian Xu, Kun Ding, Xu-Yao Zhang, Gaofeng Meng, Cheng-Lin Liu - [[pdf]](https://arxiv.org/pdf/2512.16214)</summary>

**Abstract:** Solving Partial Differential Equations (PDEs) is a cornerstone of engineering and scientific research. Traditional methods for PDE solving are cumbersome, relying on manual setup and domain expertise. While Physics-Informed Neural Network (PINNs) introduced end-to-end neural network-based solutions, and frameworks like DeepXDE further enhanced automation, these approaches still depend on expert knowledge and lack full autonomy. In this work, we frame PDE solving as tool invocation via LLM-driven agents and introduce PDE-Agent, the first toolchain-augmented multi-agent collaboration framework, inheriting the reasoning capacity of LLMs and the controllability of external tools and enabling automated PDE solving from natural language descriptions. PDE-Agent leverages the strengths of multi-agent and multi-tool collaboration through two key innovations: (1) A Prog-Act framework with graph memory for multi-agent collaboration, which enables effective dynamic planning and error correction via dual-loop mechanisms (localized fixes and global revisions). (2) A Resource-Pool integrated with a tool-parameter separation mechanism for multi-tool collaboration. This centralizes the management of runtime artifacts and resolves inter-tool dependency gaps in existing frameworks. To validate and evaluate this new paradigm for PDE solving , we develop PDE-Bench, a multi-type PDE Benchmark for agent-based tool collaborative solving, and propose multi-level metrics for assessing tool coordination. Evaluations verify that PDE-Agent exhibits superior applicability and performance in complex multi-step, cross-step dependent tasks. This new paradigm of toolchain-augmented multi-agent PDE solving will further advance future developments in automated scientific computing. Our source code and dataset will be made publicly available.

**arXiv ID:** 2512.16214
</details>

<details>
<summary><strong>QuadSentinel: Sequent Safety for Machine-Checkable Control in Multi-agent Systems</strong> - Yiliu Yang, Yilei Jiang, Qunzhong Wang, Yingshui Tan, Xiaoyong Zhu, Sherman S.M. Chow, Bo Zheng, Xiangyu Yue - [[pdf]](https://arxiv.org/pdf/2512.16279)</summary>

**Abstract:** Safety risks arise as large language model-based agents solve complex tasks with tools, multi-step plans, and inter-agent messages. However, deployer-written policies in natural language are ambiguous and context dependent, so they map poorly to machine-checkable rules, and runtime enforcement is unreliable. Expressing safety policies as sequents, we propose \textsc{QuadSentinel}, a four-agent guard (state tracker, policy verifier, threat watcher, and referee) that compiles these policies into machine-checkable rules built from predicates over observable state and enforces them online. Referee logic plus an efficient top-$k$ predicate updater keeps costs low by prioritizing checks and resolving conflicts hierarchically. Measured on ST-WebAgentBench (ICML CUA~'25) and AgentHarm (ICLR~'25), \textsc{QuadSentinel} improves guardrail accuracy and rule recall while reducing false positives. Against single-agent baselines such as ShieldAgent (ICML~'25), it yields better overall safety control. Near-term deployments can adopt this pattern without modifying core agents by keeping policies separate and machine-checkable. Our code will be made publicly available at this https URL.

**arXiv ID:** 2512.16279
</details>

<details>
<summary><strong>StarCraft+: Benchmarking Multi-agent Algorithms in Adversary Paradigm</strong> - Yadong Li, Tong Zhang, Bo Huang, Zhen Cui - [[pdf]](https://arxiv.org/pdf/2512.16444)</summary>

**Abstract:** Deep multi-agent reinforcement learning (MARL) algorithms are booming in the field of collaborative intelligence, and StarCraft multi-agent challenge (SMAC) is widely-used as the benchmark therein. However, imaginary opponents of MARL algorithms are practically configured and controlled in a fixed built-in AI mode, which causes less diversity and versatility in algorithm evaluation. To address this issue, in this work, we establish a multi-agent algorithm-vs-algorithm environment, named StarCraft II battle arena (SC2BA), to refresh the benchmarking of MARL algorithms in an adversary paradigm. Taking StarCraft as infrastructure, the SC2BA environment is specifically created for inter-algorithm adversary with the consideration of fairness, usability and customizability, and meantime an adversarial PyMARL (APyMARL) library is developed with easy-to-use interfaces/modules. Grounding in SC2BA, we benchmark those classic MARL algorithms in two types of adversarial modes: dual-algorithm paired adversary and multi-algorithm mixed adversary, where the former conducts the adversary of pairwise algorithms while the latter focuses on the adversary to multiple behaviors from a group of algorithms. The extensive benchmark experiments exhibit some thought-provoking observations/problems in the effectivity, sensibility and scalability of these completed algorithms. The SC2BA environment as well as reproduced experiments are released in \href{this https URL}{Github}, and we believe that this work could mark a new step for the MARL field in the coming years.

**arXiv ID:** 2512.16444
</details>

<details>
<summary><strong>cuPilot: A Strategy-Coordinated Multi-agent Framework for CUDA Kernel Evolution</strong> - Jinwu Chen, Qidie Wu, Bin Li, Lin Ma, Xin Si, Yang Hu, Shouyi Yin, Jun Yang - [[pdf]](https://arxiv.org/pdf/2512.16465)</summary>

**Abstract:** Optimizing CUDA kernels is a challenging and labor-intensive task, given the need for hardware-software co-design expertise and the proprietary nature of high-performance kernel libraries. While recent large language models (LLMs) combined with evolutionary algorithms show promise in automatic kernel optimization, existing approaches often fall short in performance due to their suboptimal agent designs and mismatched evolution representations. This work identifies these mismatches and proposes cuPilot, a strategy-coordinated multi-agent framework that introduces strategy as an intermediate semantic representation for kernel evolution. Key contributions include a strategy-coordinated evolution algorithm, roofline-guided prompting, and strategy-level population initialization. Experimental results show that the generated kernels by cuPilot achieve an average speed up of 3.09$\times$ over PyTorch on a benchmark of 100 kernels. On the GEMM tasks, cuPilot showcases sophisticated optimizations and achieves high utilization of critical hardware units. The generated kernels are open-sourced at this https URL.

**arXiv ID:** 2512.16465
</details>

<details>
<summary><strong>Do Multi-Agents Solve Better Than Single? Evaluating Agentic Frameworks for Diagram-Grounded Geometry Problem Solving and Reasoning</strong> - Mahbub E Sobhani, Md. Faiyaz Abdullah Sayeedi, Mohammad Nehad Alam, Proma Hossain Progga, Swakkhar Shatabda - [[pdf]](https://arxiv.org/pdf/2512.16698)</summary>

**Abstract:** Diagram-grounded geometry problem solving is a critical benchmark for multimodal large language models (MLLMs), yet the benefits of multi-agent design over single-agent remain unclear. We systematically compare single-agent and multi-agent pipelines on four visual math benchmarks: Geometry3K, MathVerse, OlympiadBench, and We-Math. For open-source models, multi-agent consistently improves performance. For example, Qwen-2.5-VL (7B) gains +6.8 points and Qwen-2.5-VL (32B) gains +3.3 on Geometry3K, and both Qwen-2.5-VL variants see further gains on OlympiadBench and We-Math. In contrast, the closed-source Gemini-2.0-Flash generally performs better in single-agent mode on classic benchmarks, while multi-agent yields only modest improvements on the newer We-Math dataset. These findings show that multi-agent pipelines provide clear benefits for open-source models and can assist strong proprietary systems on newer, less familiar benchmarks, but agentic decomposition is not universally optimal. All code, data, and reasoning files are available at this https URL

**arXiv ID:** 2512.16698
</details>

<details>
<summary><strong>FedSight AI: Multi-Agent System Architecture for Federal Funds Target Rate Prediction</strong> - Yuhan Hou, Tianji Rao, Jeremy Tan, Adler Viton, Xiyue Zhang, David Ye, Abhishek Kodi, Sanjana Dulam, Aditya Paul, Yikai Feng - [[pdf]](https://arxiv.org/pdf/2512.15728)</summary>

**Abstract:** The Federal Open Market Committee (FOMC) sets the federal funds rate, shaping monetary policy and the broader economy. We introduce \emph{FedSight AI}, a multi-agent framework that uses large language models (LLMs) to simulate FOMC deliberations and predict policy outcomes. Member agents analyze structured indicators and unstructured inputs such as the Beige Book, debate options, and vote, replicating committee reasoning. A Chain-of-Draft (CoD) extension further improves efficiency and accuracy by enforcing concise multistage reasoning. Evaluated at 2023-2024 meetings, FedSight CoD achieved accuracy of 93.75\% and stability of 93.33\%, outperforming baselines including MiniFed and Ordinal Random Forest (RF), while offering transparent reasoning aligned with real FOMC communications.

**arXiv ID:** 2512.15728
</details>

<details>
<summary><strong>Toward Agentic Environments: GenAI and the Convergence of AI, Sustainability, and Human-Centric Spaces</strong> - Przemek Pospieszny, Dominika P. Brodowicz - [[pdf]](https://arxiv.org/pdf/2512.15787)</summary>

**Abstract:** In recent years, advances in artificial intelligence (AI), particularly generative AI (GenAI) and large language models (LLMs), have made human-computer interactions more frequent, efficient, and accessible across sectors ranging from banking to healthcare. AI tools embedded in digital devices support decision-making and operational management at both individual and organizational levels, including resource allocation, workflow automation, and real-time data analysis. However, the prevailing cloud-centric deployment of AI carries a substantial environmental footprint due to high computational demands. In this context, this paper introduces the concept of agentic environments, a sustainability-oriented AI framework that extends beyond reactive systems by leveraging GenAI, multi-agent systems, and edge computing to reduce the environmental impact of technology. Agentic environments enable more efficient resource use, improved quality of life, and sustainability-by-design, while simultaneously enhancing data privacy through decentralized, edge-driven solutions. Drawing on secondary research as well as primary data from focus groups and semi-structured interviews with AI professionals from leading technology companies, the paper proposes a conceptual framework for agentic environments examined through three lenses: the personal sphere, professional and commercial use, and urban operations. The findings highlight the potential of agentic environments to foster sustainable ecosystems through optimized resource utilization and strengthened data privacy. The study concludes with recommendations for edge-driven deployment models to reduce reliance on energy-intensive cloud infrastructures.

**arXiv ID:** 2512.15787
</details>

<details>
<summary><strong>Scalable Agentic Reasoning for Designing Biologics Targeting Intrinsically Disordered Proteins</strong> - Matthew Sinclair, Moeen Meigooni, Archit Vasan, Ozan Gokdemir, Xinran Lian, Heng Ma, Yadu Babuji, Alexander Brace, Khalid Hossain, Carlo Siebenschuh, Thomas Brettin, Kyle Chard, Christopher Henry, Venkatram Vishwanath, Rick L. Stevens, Ian T. Foster, Arvind Ramanathan - [[pdf]](https://arxiv.org/pdf/2512.15930)</summary>

**Abstract:** Intrinsically disordered proteins (IDPs) represent crucial therapeutic targets due to their significant role in disease -- approximately 80\% of cancer-related proteins contain long disordered regions -- but their lack of stable secondary/tertiary structures makes them "undruggable". While recent computational advances, such as diffusion models, can design high-affinity IDP binders, translating these to practical drug discovery requires autonomous systems capable of reasoning across complex conformational ensembles and orchestrating diverse computational tools at this http URL address this challenge, we designed and implemented StructBioReasoner, a scalable multi-agent system for designing biologics that can be used to target IDPs. StructBioReasoner employs a novel tournament-based reasoning framework where specialized agents compete to generate and refine therapeutic hypotheses, naturally distributing computational load for efficient exploration of the vast design space. Agents integrate domain knowledge with access to literature synthesis, AI-structure prediction, molecular simulations, and stability analysis, coordinating their execution on HPC infrastructure via an extensible federated agentic middleware, Academy. We benchmark StructBioReasoner across Der f 21 and NMNAT-2 and demonstrate that over 50\% of 787 designed and validated candidates for Der f 21 outperformed the human-designed reference binders from literature, in terms of improved binding free energy. For the more challenging NMNAT-2 protein, we identified three binding modes from 97,066 binders, including the well-studied NMNAT2:p53 interface. Thus, StructBioReasoner lays the groundwork for agentic reasoning systems for IDP therapeutic discovery on Exascale platforms.

**arXiv ID:** 2512.15930
</details>

<details>
<summary><strong>A Multi-Agent Large Language Model Framework for Automated Qualitative Analysis</strong> - Qidi Xu, Nuzha Amjad, Grace Giles, Alexa Cumming, De'angelo Hermesky, Alexander Wen, Min Ji Kwak, Yejin Kim - [[pdf]](https://arxiv.org/pdf/2512.16063)</summary>

**Abstract:** Understanding patients experiences is essential for advancing patient centered care, especially in chronic diseases that require ongoing communication. However, qualitative thematic analysis, the primary approach for exploring these experiences, remains labor intensive, subjective, and difficult to scale. In this study, we developed a multi agent large language model framework that automates qualitative thematic analysis through three agents (Instructor, Thematizer, CodebookGenerator), named Collaborative Theme Identification Agent (CoTI). We applied CoTI to 12 heart failure patient interviews to analyze their perceptions of medication intensity. CoTI identified key phrases, themes, and codebook that were more similar to those of the senior investigator than both junior investigators and baseline NLP models. We also implemented CoTI into a user-facing application to enable AI human interaction in qualitative analysis. However, collaboration between CoTI and junior investigators provided only marginal gains, suggesting they may overrely on CoTI and limit their independent critical thinking.

**arXiv ID:** 2512.16063
</details>

<details>
<summary><strong>Ev-Trust: A Strategy Equilibrium Trust Mechanism for Evolutionary Games in LLM-Based Multi-Agent Services</strong> - Shiduo Yang, Jiye Wang, Jiayu Qin, Jianbin Li, Yu Wang, Yuanhe Zhao, Kenan Guo - [[pdf]](https://arxiv.org/pdf/2512.16167)</summary>

**Abstract:** The rapid evolution of the Web toward an agent-centric paradigm, driven by large language models (LLMs), has enabled autonomous agents to reason, plan, and interact in complex decentralized environments. However, the openness and heterogeneity of LLM-based multi-agent systems also amplify the risks of deception, fraud, and misinformation, posing severe challenges to trust establishment and system robustness. To address this issue, we propose Ev-Trust, a strategy-equilibrium trust mechanism grounded in evolutionary game theory. This mechanism integrates direct trust, indirect trust, and expected revenue into a dynamic feedback structure that guides agents' behavioral evolution toward equilibria. Within a decentralized "Request-Response-Payment-Evaluation" service framework, Ev-Trust enables agents to adaptively adjust strategies, naturally excluding malicious participants while reinforcing high-quality collaboration. Furthermore, our theoretical derivation based on replicator dynamics equations proves the existence and stability of local evolutionary equilibria. Experimental results indicate that our approach effectively reflects agent trustworthiness in LLM-driven open service interaction scenarios, reduces malicious strategies, and increases collective revenue. We hope Ev-Trust can provide a new perspective on trust modeling for the agentic service web in group evolutionary game scenarios.

**arXiv ID:** 2512.16167
</details>

<details>
<summary><strong>Emergent Bias and Fairness in Multi-Agent Decision Systems</strong> - Maeve Madigan, Parameswaran Kamalaruban, Glenn Moynihan, Tom Kempton, David Sutton, Stuart Burrell - [[pdf]](https://arxiv.org/pdf/2512.16433)</summary>

**Abstract:** Multi-agent systems have demonstrated the ability to improve performance on a variety of predictive tasks by leveraging collaborative decision making. However, the lack of effective evaluation methodologies has made it difficult to estimate the risk of bias, making deployment of such systems unsafe in high stakes domains such as consumer finance, where biased decisions can translate directly into regulatory breaches and financial loss. To address this challenge, we need to develop fairness evaluation methodologies for multi-agent predictive systems and measure the fairness characteristics of these systems in the financial tabular domain. Examining fairness metrics using large-scale simulations across diverse multi-agent configurations, with varying communication and collaboration mechanisms, we reveal patterns of emergent bias in financial decision-making that cannot be traced to individual agent components, indicating that multi-agent systems may exhibit genuinely collective behaviors. Our findings highlight that fairness risks in financial multi-agent systems represent a significant component of model risk, with tangible impacts on tasks such as credit scoring and income estimation. We advocate that multi-agent decision systems must be evaluated as holistic entities rather than through reductionist analyses of their constituent components.

**arXiv ID:** 2512.16433
</details>

<details>
<summary><strong>Coordinated Anti-Jamming Resilience in Swarm Networks via Multi-Agent Reinforcement Learning</strong> - Bahman Abolhassani, Tugba Erpek, Kemal Davaslioglu, Yalin E. Sagduyu, Sastry Kompella - [[pdf]](https://arxiv.org/pdf/2512.16813)</summary>

**Abstract:** Reactive jammers pose a severe security threat to robotic-swarm networks by selectively disrupting inter-agent communications and undermining formation integrity and mission success. Conventional countermeasures such as fixed power control or static channel hopping are largely ineffective against such adaptive adversaries. This paper presents a multi-agent reinforcement learning (MARL) framework based on the QMIX algorithm to improve the resilience of swarm communications under reactive jamming. We consider a network of multiple transmitter-receiver pairs sharing channels while a reactive jammer with Markovian threshold dynamics senses aggregate power and reacts accordingly. Each agent jointly selects transmit frequency (channel) and power, and QMIX learns a centralized but factorizable action-value function that enables coordinated yet decentralized execution. We benchmark QMIX against a genie-aided optimal policy in a no-channel-reuse setting, and against local Upper Confidence Bound (UCB) and a stateless reactive policy in a more general fading regime with channel reuse enabled. Simulation results show that QMIX rapidly converges to cooperative policies that nearly match the genie-aided bound, while achieving higher throughput and lower jamming incidence than the baselines, thereby demonstrating MARL's effectiveness for securing autonomous swarms in contested environments.

**arXiv ID:** 2512.16813
</details>

<details>
<summary><strong>TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems</strong> - Shaina Raza, Ranjan Sapkota, Manoj Karkee, Christos Emmanouilidis - [[pdf]](https://arxiv.org/pdf/2506.04133)</summary>

**Abstract:** Agentic AI systems, built upon large language models (LLMs) and deployed in multi-agent configurations, are redefining intelligence, autonomy, collaboration, and decision-making across enterprise and societal domains. This review presents a structured analysis of Trust, Risk, and Security Management (TRiSM) in the context of LLM-based Agentic Multi-Agent Systems (AMAS). We begin by examining the conceptual foundations of Agentic AI and highlight its architectural distinctions from traditional AI agents. We then adapt and extend the AI TRiSM framework for Agentic AI, structured around key pillars: \textit{ Explainability, ModelOps, Security, Privacy} and \textit{their Lifecycle Governance}, each contextualized to the challenges of AMAS. A risk taxonomy is proposed to capture the unique threats and vulnerabilities of Agentic AI, ranging from coordination failures to prompt-based adversarial manipulation. To support practical assessment in Agentic AI works, we introduce two novel metrics: the Component Synergy Score (CSS), which quantifies the quality of inter-agent collaboration, and the Tool Utilization Efficacy (TUE), which evaluates the efficiency of tool use within agent workflows. We further discuss strategies for improving explainability in Agentic AI, as well as approaches to enhancing security and privacy through encryption, adversarial robustness, and regulatory compliance. The review concludes with a research roadmap for the responsible development and deployment of Agentic AI, highlighting key directions to align emerging systems with TRiSM principles-ensuring safety, transparency, and accountability in their operation.

**arXiv ID:** 2506.04133
</details>

<details>
<summary><strong>Knowledge-Driven Agentic Scientific Corpus Distillation Framework for Biomedical Large Language Models Training</strong> - Meng Xiao, Xunxin Cai, Qingqing Long, Chengrui Wang, Yuanchun Zhou, Hengshu Zhu - [[pdf]](https://arxiv.org/pdf/2504.19565)</summary>

**Abstract:** Corpus distillation for biomedical large language models (LLMs) seeks to address the pressing challenge of insufficient quantity and quality in open-source annotated scientific corpora, which remains a bottleneck for effective LLM training in biomedical research. This paper proposes a knowledge-driven, agentic framework for scientific corpus distillation, tailored explicitly for LLM training in the biomedical domain, addressing the challenge posed by the complex hierarchy of biomedical knowledge. Central to our approach is a collaborative multi-agent architecture, where specialized agents, each guided by the Medical Subject Headings (MeSH) hierarchy, work in concert to autonomously extract, synthesize, and self-evaluate high-quality textual data from vast scientific literature. This agentic framework collectively generates and refines domain-specific question-answer pairs, ensuring comprehensive coverage and consistency with biomedical ontologies while minimizing manual involvement. Extensive experimental results show that language models trained on our multi-agent distilled datasets achieve notable improvements in biomedical question-answering tasks, outperforming both strong life sciences LLM baselines and advanced proprietary models. Notably, our AI-Ready dataset enables Llama3-70B to surpass GPT-4 with MedPrompt and Med-PaLM-2, despite their larger scale. Detailed ablation studies and case analyses further validate the effectiveness and synergy of each agent within the framework, highlighting the potential of multi-agent collaboration in biomedical LLM training.

**arXiv ID:** 2504.19565
</details>

<details>
<summary><strong>Breaking the Performance Ceiling in Reinforcement Learning requires Inference Strategies</strong> - Felix Chalumeau, Daniel Rajaonarivonivelomanantsoa, Ruan de Kock, Claude Formanek, Sasha Abramowitz, Oumayma Mahjoub, Wiem Khlifi, Simon Du Toit, Louay Ben Nessir, Refiloe Shabe, Noah De Nicola, Arnol Fokam, Siddarth Singh, Ulrich Mbou Sob, Arnu Pretorius - [[pdf]](https://arxiv.org/pdf/2505.21236)</summary>

**Abstract:** Reinforcement learning (RL) systems have countless applications, from energy-grid management to protein design. However, such real-world scenarios are often extremely difficult, combinatorial in nature, and require complex coordination between multiple agents. This level of complexity can cause even state-of-the-art RL systems, trained until convergence, to hit a performance ceiling which they are unable to break out of with zero-shot inference. Meanwhile, many digital or simulation-based applications allow for an inference phase that utilises a specific time and compute budget to explore multiple attempts before outputting a final solution. In this work, we show that such an inference phase employed at execution time, and the choice of a corresponding inference strategy, are key to breaking the performance ceiling observed in complex multi-agent RL problems. Our main result is striking: we can obtain up to a 126% and, on average, a 45% improvement over the previous state-of-the-art across 17 tasks, using only a couple seconds of extra wall-clock time during execution. We also demonstrate promising compute scaling properties, supported by over 60k experiments, making it the largest study on inference strategies for complex RL to date. Our experimental data and code are available at this https URL.

**arXiv ID:** 2505.21236
</details>

<details>
<summary><strong>Voice-Interactive Surgical Agent for Multimodal Patient Data Control</strong> - Hyeryun Park, Byung Mo Gu, Jun Hee Lee, Byeong Hyeon Choi, Sekeun Kim, Hyun Koo Kim, Kyungsang Kim - [[pdf]](https://arxiv.org/pdf/2511.07392)</summary>

**Abstract:** In robotic surgery, surgeons fully engage their hands and visual attention in procedures, making it difficult to access and manipulate multimodal patient data without interrupting the workflow. To overcome this problem, we propose a Voice-Interactive Surgical Agent (VISA) built on a hierarchical multi-agent framework consisting of an orchestration agent and three task-specific agents driven by Large Language Models (LLMs). These LLM-based agents autonomously plan, refine, validate, and reason to interpret voice commands and execute tasks such as retrieving clinical information, manipulating CT scans, or navigating 3D anatomical models within surgical video. We construct a dataset of 240 user commands organized into hierarchical categories and introduce the Multi-level Orchestration Evaluation Metric (MOEM) that evaluates the performance and robustness at both the command and category levels. Experimental results demonstrate that VISA achieves high stage-level accuracy and workflow-level success rates, while also enhancing its robustness by correcting transcription errors, resolving linguistic ambiguity, and interpreting diverse free-form expressions. These findings highlight the strong potential of VISA to support robotic surgery and its scalability for integrating new functions and agents.

**arXiv ID:** 2511.07392
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>Emergence: Overcoming Privileged Information Bias in Asymmetric Embodied Agents via Active Querying</strong> - Shaun Baek, Sam Liu, Joseph Ukpong - [[pdf]](https://arxiv.org/pdf/2512.15776)</summary>

**Abstract:** Large Language Models (LLMs) act as powerful reasoning engines but struggle with "symbol grounding" in embodied environments, particularly when information is asymmetrically distributed. We investigate the Privileged Information Bias (or "Curse of Knowledge"), where a knowledgeable "Leader" agent fails to guide a sensor-limited "Follower" due to a lack of Theory of Mind. To quantify this phenomenon, we propose a novel Asymmetric Assistive Reasoning framework within AI2-THOR. Our experiments reveal a significant "Success Gap": while the Leader successfully perceives the target in 35.0% of episodes, the collaborative team succeeds only 17.0% of the time, implying that nearly 50% of feasible plans fail solely due to communicative grounding errors. We demonstrate that a "Pull-based" protocol (active querying) is significantly more robust than standard "Push-based" instruction, with successful episodes featuring 2x the frequency of clarification requests. This research isolates the mechanism of active uncertainty reduction as a prerequisite for safe human-AI and robot-robot collaboration.

**arXiv ID:** 2512.15776
</details>

<details>
<summary><strong>Science Consultant Agent</strong> - Karthikeyan K, Philip Wu, Xin Tang, Alexandre Alves - [[pdf]](https://arxiv.org/pdf/2512.16171)</summary>

**Abstract:** The Science Consultant Agent is a web-based Artificial Intelligence (AI) tool that helps practitioners select and implement the most effective modeling strategy for AI-based solutions. It operates through four core components: Questionnaire, Smart Fill, Research-Guided Recommendation, and Prototype Builder. By combining structured questionnaires, literature-backed solution recommendations, and prototype generation, the Science Consultant Agent accelerates development for everyone from Product Managers and Software Developers to Researchers. The full pipeline is illustrated in Figure 1.

**arXiv ID:** 2512.16171
</details>

<details>
<summary><strong>From Personalization to Prejudice: Bias and Discrimination in Memory-Enhanced AI Agents for Recruitment</strong> - Himanshu Gharat, Himanshi Agrawal, Gourab K. Patro - [[pdf]](https://arxiv.org/pdf/2512.16532)</summary>

**Abstract:** Large Language Models (LLMs) have empowered AI agents with advanced capabilities for understanding, reasoning, and interacting across diverse tasks. The addition of memory further enhances them by enabling continuity across interactions, learning from past experiences, and improving the relevance of actions and responses over time; termed as memory-enhanced personalization. Although such personalization through memory offers clear benefits, it also introduces risks of bias. While several previous studies have highlighted bias in ML and LLMs, bias due to memory-enhanced personalized agents is largely unexplored. Using recruitment as an example use case, we simulate the behavior of a memory-enhanced personalized agent, and study whether and how bias is introduced and amplified in and across various stages of operation. Our experiments on agents using safety-trained LLMs reveal that bias is systematically introduced and reinforced through personalization, emphasizing the need for additional protective measures or agent guardrails in memory-enhanced LLM-based AI agents.

**arXiv ID:** 2512.16532
</details>

<details>
<summary><strong>Dual Computational Horizons: Incompleteness and Unpredictability in Intelligent Systems</strong> - Abhisek Ganguly - [[pdf]](https://arxiv.org/pdf/2512.16707)</summary>

**Abstract:** We formalize two independent computational limitations that constrain algorithmic intelligence: formal incompleteness and dynamical unpredictability. The former limits the deductive power of consistent reasoning systems while the later bounds long-term prediction under finite precision. We show that these two extrema together impose structural bounds on an agent's ability to reason about its own predictive capabilities. In particular, an algorithmic agent cannot compute its own maximal prediction horizon generally. This perspective clarifies inherent trade-offs between reasoning, prediction, and self-analysis in intelligent systems.

**arXiv ID:** 2512.16707
</details>

<details>
<summary><strong>CodeMem: Architecting Reproducible Agents via Dynamic MCP and Procedural Memory</strong> - Nishant Gaurav, Adit Akarsh, Tejas Ravishankar, Manoj Bajaj - [[pdf]](https://arxiv.org/pdf/2512.15813)</summary>

**Abstract:** Current tool-using AI agents suffer from limited action space, context inefficiency, and probabilistic instability that makes them unsuitable for handling repetitive tasks which are otherwise reliably and efficiently tackled by agentic workflows built on platforms like n8n and Zapier. Earlier works like CodeAct, DynaSaur, Code Mode have tried to tackle the first two issues by using the whole Python language as its action space: The number of tools that the agent can call becomes infinite. Python code blocks can execute complex actions into a single step and print only relevant results which helps in keeping the context lean. However, the probabilistic instability issue still remains, as for the same task in the same environment, the agent can follow different trajectories due to the probabilistic nature of LLMs. Therefore, we need procedural memory for consistency and reliability. This paper proposes CodeMem, an architecture to implement procedural memory via code which can be used to build and run reusable agentic workflows with deterministic reliability.

**arXiv ID:** 2512.15813
</details>

<details>
<summary><strong>Optimizing Agentic Language Model Inference via Speculative Tool Calls</strong> - Daniel Nichols, Prajwal Singhania, Charles Jekel, Abhinav Bhatele, Harshitha Menon - [[pdf]](https://arxiv.org/pdf/2512.15834)</summary>

**Abstract:** Language models (LMs) are becoming increasingly dependent on external tools. LM-based agentic frameworks frequently interact with their environment via such tools to search files, run code, call APIs, etc. Further, modern reasoning-based LMs use tools such as web search and Python code execution to enhance their reasoning capabilities. While tools greatly improve the capabilities of LMs, they also introduce performance bottlenecks during the inference process. In this paper, we introduce novel systems optimizations to address such performance bottlenecks by speculating tool calls and forcing sequences to remain resident in the inference engine to minimize overheads. Our optimizations lead to throughput improvements of several hundred tokens per second when hosting inference for LM agents. We provide a theoretical analysis of our algorithms to provide insights into speculation configurations that will yield the best performance. Further, we recommend a new "tool cache" API endpoint to enable LM providers to easily adopt these optimizations.

**arXiv ID:** 2512.15834
</details>

<details>
<summary><strong>Don't Guess, Escalate: Towards Explainable Uncertainty-Calibrated AI Forensic Agents</strong> - Giulia Boato, Andrea Montibeller, Edward Delp, Luisa Verdoliva, Daniele Miorandi - [[pdf]](https://arxiv.org/pdf/2512.16614)</summary>

**Abstract:** AI is reshaping the landscape of multimedia forensics. We propose AI forensic agents: reliable orchestrators that select and combine forensic detectors, identify provenance and context, and provide uncertainty-aware assessments. We highlight pitfalls in current solutions and introduce a unified framework to improve the authenticity verification process.

**arXiv ID:** 2512.16614
</details>

<details>
<summary><strong>WildFit: Autonomous In-situ Model Adaptation for Resource-Constrained IoT Systems</strong> - Mohammad Mehdi Rastikerdar, Jin Huang, Hui Guan, Deepak Ganesan - [[pdf]](https://arxiv.org/pdf/2409.07796)</summary>

**Abstract:** Resource-constrained IoT devices increasingly rely on deep learning models, however, these models experience significant accuracy drops due to domain shifts when encountering variations in lighting, weather, and seasonal conditions. While cloud-based retraining can address this issue, many IoT deployments operate with limited connectivity and energy constraints, making traditional fine-tuning approaches impractical. We explore this challenge through the lens of wildlife ecology, where camera traps must maintain accurate species classification across changing seasons, weather, and habitats without reliable connectivity. We introduce WildFit, an autonomous in-situ adaptation framework that leverages the key insight that background scenes change more frequently than the visual characteristics of monitored species. WildFit combines background-aware synthesis to generate training samples on-device with drift-aware fine-tuning that triggers model updates only when necessary to conserve resources. Our background-aware synthesis surpasses efficient baselines by 7.3% and diffusion models by 3.0% while being orders of magnitude faster, our drift-aware fine-tuning achieves Pareto optimality with 50% fewer updates and 1.5% higher accuracy, and the end-to-end system outperforms domain adaptation approaches by 20-35% while consuming only 11.2 Wh over 37 days-enabling battery-powered deployment.

**arXiv ID:** 2409.07796
</details>

<details>
<summary><strong>Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents</strong> - Zhizhen Zhang, Lei Zhu, Zhen Fang, Zi Huang, Yadan Luo - [[pdf]](https://arxiv.org/pdf/2502.01218)</summary>

**Abstract:** Pre-training vision-language representations on human action videos has emerged as a promising approach to reduce reliance on large-scale expert demonstrations for training embodied agents. However, prior methods often employ time contrastive learning based on goal-reaching heuristics, progressively aligning language instructions from the initial to the final frame. This overemphasis on future frames can result in erroneous vision-language associations, as actions may terminate early or include irrelevant moments in the end. To address this issue, we propose Action Temporal Coherence Learning (AcTOL) to learn ordered and continuous vision-language representations without rigid goal-based constraint. AcTOL treats a video as a continuous trajectory where it (1) contrasts semantic differences between frames to reflect their natural ordering, and (2) imposes a local Brownian bridge constraint to ensure smooth transitions across intermediate frames. Extensive imitation learning experiments on both simulated and real robots show that the pretrained features significantly enhance downstream manipulation tasks with high robustness to different linguistic styles of instructions, offering a viable pathway toward generalized embodied agents.

**arXiv ID:** 2502.01218
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (19 papers)</h2></summary>

<details>
<summary><strong>Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning</strong> - Polaris Jhandi, Owais Kazi, Shreyas Subramanian, Neel Sendas - [[pdf]](https://arxiv.org/pdf/2512.15943)</summary>

**Abstract:** As organizations scale adoption of generative AI, model cost optimization and operational efficiency have emerged as critical factors determining sustainability and accessibility. While Large Language Models (LLMs) demonstrate impressive capabilities across diverse tasks, their extensive computational requirements make them cost-prohibitive for routine enterprise use. This limitation motivates the exploration of Small Language Models (SLMs), which can deliver comparable performance in targeted applications while drastically reducing infrastructure overhead (Irugalbandara et al., 2023). In this work, we investigate the feasibility of replacing LLM-driven workflows with optimized SLMs. We trained a domain-adapted SLM to execute representative tasks traditionally handled by LLMs, such as document summarization, query answering, and structured data interpretation. As part of the experiment, we investigated the fine-tuning of facebook/opt-350m model (single epoch only) using the Hugging Face TRL (Transformer Reinforcement Learning), specifically the Supervised Fine-Tuning (SFT) trainer. The OPT-350M model was released by Meta AI in 2022 as part of the OPT (Open Pretrained Transformer) family of models. Similar studies demonstrate that even models at the 350M parameter scale can meaningfully contribute to instruction-tuning pipelines (Mekala et al., 2024). Experimental results demonstrated that our fine-tuned SLM achieves exceptional performance with a 77.55\% pass rate on ToolBench evaluation, significantly outperforming all baseline models including ChatGPT-CoT (26.00\%), ToolLLaMA-DFS (30.18\%), and ToolLLaMA-CoT (16.27\%). These findings emphasize that thoughtful design and targeted training of SLMs can significantly lower barriers to adoption, enabling cost-effective, large-scale integration of generative AI into production systems.

**arXiv ID:** 2512.15943
</details>

<details>
<summary><strong>Learning to Wait: Synchronizing Agents with the Physical World</strong> - Yifei She, Ping Zhang, He Liu, Yanmin Jia, Yang Jing, Zijun Liu, Peng Sun, Xiangbin Li, Xiaohe Hu - [[pdf]](https://arxiv.org/pdf/2512.16262)</summary>

**Abstract:** Real-world agentic tasks, unlike synchronous Markov Decision Processes (MDPs), often involve non-blocking actions with variable latencies, creating a fundamental \textit{Temporal Gap} between action initiation and completion. Existing environment-side solutions, such as blocking wrappers or frequent polling, either limit scalability or dilute the agent's context window with redundant observations. In this work, we propose an \textbf{Agent-side Approach} that empowers Large Language Models (LLMs) to actively align their \textit{Cognitive Timeline} with the physical world. By extending the Code-as-Action paradigm to the temporal domain, agents utilize semantic priors and In-Context Learning (ICL) to predict precise waiting durations (\texttt{this http URL(t)}), effectively synchronizing with asynchronous environment without exhaustive checking. Experiments in a simulated Kubernetes cluster demonstrate that agents can precisely calibrate their internal clocks to minimize both query overhead and execution latency, validating that temporal awareness is a learnable capability essential for autonomous evolution in open-ended environments.

**arXiv ID:** 2512.16262
</details>

<details>
<summary><strong>Adaptation of Agentic AI</strong> - Pengcheng Jiang, Jiacheng Lin, Zhiyi Shi, Zifeng Wang, Luxi He, Yichen Wu, Ming Zhong, Peiyang Song, Qizheng Zhang, Heng Wang, Xueqiang Xu, Hanwen Xu, Pengrui Han, Dylan Zhang, Jiashuo Sun, Chaoqi Yang, Kun Qian, Tian Wang, Changran Hu, Manling Li, Quanzheng Li, Hao Peng, Sheng Wang, Jingbo Shang, Chao Zhang, Jiaxuan You, Liyuan Liu, Pan Lu, Yu Zhang, Heng Ji, Yejin Choi, Dawn Song, Jimeng Sun, Jiawei Han - [[pdf]](https://arxiv.org/pdf/2512.16301)</summary>

**Abstract:** Cutting-edge agentic AI systems are built on foundation models that can be adapted to plan, reason, and interact with external tools to perform increasingly complex and specialized tasks. As these systems grow in capability and scope, adaptation becomes a central mechanism for improving performance, reliability, and generalization. In this paper, we unify the rapidly expanding research landscape into a systematic framework that spans both agent adaptations and tool adaptations. We further decompose these into tool-execution-signaled and agent-output-signaled forms of agent adaptation, as well as agent-agnostic and agent-supervised forms of tool adaptation. We demonstrate that this framework helps clarify the design space of adaptation strategies in agentic AI, makes their trade-offs explicit, and provides practical guidance for selecting or switching among strategies during system design. We then review the representative approaches in each category, analyze their strengths and limitations, and highlight key open challenges and future opportunities. Overall, this paper aims to offer a conceptual foundation and practical roadmap for researchers and practitioners seeking to build more capable, efficient, and reliable agentic AI systems.

**arXiv ID:** 2512.16301
</details>

<details>
<summary><strong>Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning</strong> - Qihao Liu, Luoxin Ye, Wufei Ma, Yu-Cheng Chou, Alan Yuille - [[pdf]](https://arxiv.org/pdf/2512.16917)</summary>

**Abstract:** Large language models (LLMs) with explicit reasoning capabilities excel at mathematical reasoning yet still commit process errors, such as incorrect calculations, brittle logic, and superficially plausible but invalid steps. In this paper, we introduce Generative Adversarial Reasoner, an on-policy joint training framework designed to enhance reasoning by co-evolving an LLM reasoner and an LLM-based discriminator through adversarial reinforcement learning. A compute-efficient review schedule partitions each reasoning chain into logically complete slices of comparable length, and the discriminator evaluates each slice's soundness with concise, structured justifications. Learning couples complementary signals: the LLM reasoner is rewarded for logically consistent steps that yield correct answers, while the discriminator earns rewards for correctly detecting errors or distinguishing traces in the reasoning process. This produces dense, well-calibrated, on-policy step-level rewards that supplement sparse exact-match signals, improving credit assignment, increasing sample efficiency, and enhancing overall reasoning quality of LLMs. Across various mathematical benchmarks, the method delivers consistent gains over strong baselines with standard RL post-training. Specifically, on AIME24, we improve DeepSeek-R1-Distill-Qwen-7B from 54.0 to 61.3 (+7.3) and DeepSeek-R1-Distill-Llama-8B from 43.7 to 53.7 (+10.0). The modular discriminator also enables flexible reward shaping for objectives such as teacher distillation, preference alignment, and mathematical proof-based reasoning.

**arXiv ID:** 2512.16917
</details>

<details>
<summary><strong>Large Model Enabled Embodied Intelligence for 6G Integrated Perception, Communication, and Computation Network</strong> - Zhuoran Li, Zhen Gao, Xinhua Liu, Zheng Wang, Xiaotian Zhou, Lei Liu, Yongpeng Wu, Wei Feng, Yongming Huang - [[pdf]](https://arxiv.org/pdf/2512.15109)</summary>

**Abstract:** The advent of sixth-generation (6G) places intelligence at the core of wireless architecture, fusing perception, communication, and computation into a single closed-loop. This paper argues that large artificial intelligence models (LAMs) can endow base stations with perception, reasoning, and acting capabilities, thus transforming them into intelligent base station agents (IBSAs). We first review the historical evolution of BSs from single-functional analog infrastructure to distributed, software-defined, and finally LAM-empowered IBSA, highlighting the accompanying changes in architecture, hardware platforms, and deployment. We then present an IBSA architecture that couples a perception-cognition-execution pipeline with cloud-edge-end collaboration and parameter-efficient adaptation. Subsequently,we study two representative scenarios: (i) cooperative vehicle-road perception for autonomous driving, and (ii) ubiquitous base station support for low-altitude uncrewed aerial vehicle safety monitoring and response against unauthorized drones. On this basis, we analyze key enabling technologies spanning LAM design and training, efficient edge-cloud inference, multi-modal perception and actuation, as well as trustworthy security and governance. We further propose a holistic evaluation framework and benchmark considerations that jointly cover communication performance, perception accuracy, decision-making reliability, safety, and energy efficiency. Finally, we distill open challenges on benchmarks, continual adaptation, trustworthy decision-making, and standardization. Together, this work positions LAM-enabled IBSAs as a practical path toward integrated perception, communication, and computation native, safety-critical 6G systems.

**arXiv ID:** 2512.15109
</details>

<details>
<summary><strong>Deep Reinforcement Learning Optimization for Uncertain Nonlinear Systems via Event-Triggered Robust Adaptive Dynamic Programming</strong> - Ningwei Bai, Chi Pui Chan, Qichen Yin, Tengyang Gong, Yunda Yan, Zezhi Tang - [[pdf]](https://arxiv.org/pdf/2512.15735)</summary>

**Abstract:** This work proposes a unified control architecture that couples a Reinforcement Learning (RL)-driven controller with a disturbance-rejection Extended State Observer (ESO), complemented by an Event-Triggered Mechanism (ETM) to limit unnecessary computations. The ESO is utilized to estimate the system states and the lumped disturbance in real time, forming the foundation for effective disturbance compensation. To obtain near-optimal behavior without an accurate system description, a value-iteration-based Adaptive Dynamic Programming (ADP) method is adopted for policy approximation. The inclusion of the ETM ensures that parameter updates of the learning module are executed only when the state deviation surpasses a predefined bound, thereby preventing excessive learning activity and substantially reducing computational load. A Lyapunov-oriented analysis is used to characterize the stability properties of the resulting closed-loop system. Numerical experiments further confirm that the developed approach maintains strong control performance and disturbance tolerance, while achieving a significant reduction in sampling and processing effort compared with standard time-triggered ADP schemes.

**arXiv ID:** 2512.15735
</details>

<details>
<summary><strong>MRG-R1: Reinforcement Learning for Clinically Aligned Medical Report Generation</strong> - Pengyu Wang, Shuchang Ye, Usman Naseem, Jinman Kim - [[pdf]](https://arxiv.org/pdf/2512.16145)</summary>

**Abstract:** Medical report generation (MRG) aims to automatically derive radiology-style reports from medical images to aid in clinical decision-making. However, existing methods often generate text that mimics the linguistic style of radiologists but fails to guarantee clinical correctness, because they are trained on token-level objectives which focus on word-choice and sentence structure rather than actual medical accuracy. We propose a semantic-driven reinforcement learning (SRL) method for medical report generation, adopted on a large vision-language model (LVLM). SRL adopts Group Relative Policy Optimization (GRPO) to encourage clinical-correctness-guided learning beyond imitation of language style. Specifically, we optimise a report-level reward: a margin-based cosine similarity (MCCS) computed between key radiological findings extracted from generated and reference reports, thereby directly aligning clinical-label agreement and improving semantic correctness. A lightweight reasoning format constraint further guides the model to generate structured "thinking report" outputs. We evaluate Medical Report Generation with Sematic-driven Reinforment Learning (MRG-R1), on two datasets: IU X-Ray and MIMIC-CXR using clinical efficacy (CE) metrics. MRG-R1 achieves state-of-the-art performance with CE-F1 51.88 on IU X-Ray and 40.39 on MIMIC-CXR. We found that the label-semantic reinforcement is better than conventional token-level supervision. These results indicate that optimizing a clinically grounded, report-level reward rather than token overlap,meaningfully improves clinical correctness. This work is a prior to explore semantic-reinforcement in supervising medical correctness in medical Large vision-language model(Med-LVLM) training.

**arXiv ID:** 2512.16145
</details>

<details>
<summary><strong>E-SDS: Environment-aware See it, Do it, Sorted - Automated Environment-Aware Reinforcement Learning for Humanoid Locomotion</strong> - Enis Yalcin, Joshua O'Hara, Maria Stamatopoulou, Chengxu Zhou, Dimitrios Kanoulas - [[pdf]](https://arxiv.org/pdf/2512.16446)</summary>

**Abstract:** Vision-language models (VLMs) show promise in automating reward design in humanoid locomotion, which could eliminate the need for tedious manual engineering. However, current VLM-based methods are essentially "blind", as they lack the environmental perception required to navigate complex terrain. We present E-SDS (Environment-aware See it, Do it, Sorted), a framework that closes this perception gap. E-SDS integrates VLMs with real-time terrain sensor analysis to automatically generate reward functions that facilitate training of robust perceptive locomotion policies, grounded by example videos. Evaluated on a Unitree G1 humanoid across four distinct terrains (simple, gaps, obstacles, stairs), E-SDS uniquely enabled successful stair descent, while policies trained with manually-designed rewards or a non-perceptive automated baseline were unable to complete the task. In all terrains, E-SDS also reduced velocity tracking error by 51.9-82.6%. Our framework reduces the human effort of reward design from days to less than two hours while simultaneously producing more robust and capable locomotion policies.

**arXiv ID:** 2512.16446
</details>

<details>
<summary><strong>ReinforceGen: Hybrid Skill Policies with Automated Data Generation and Reinforcement Learning</strong> - Zihan Zhou, Animesh Garg, Ajay Mandlekar, Caelan Garrett - [[pdf]](https://arxiv.org/pdf/2512.16861)</summary>

**Abstract:** Long-horizon manipulation has been a long-standing challenge in the robotics community. We propose ReinforceGen, a system that combines task decomposition, data generation, imitation learning, and motion planning to form an initial solution, and improves each component through reinforcement-learning-based fine-tuning. ReinforceGen first segments the task into multiple localized skills, which are connected through motion planning. The skills and motion planning targets are trained with imitation learning on a dataset generated from 10 human demonstrations, and then fine-tuned through online adaptation and reinforcement learning. When benchmarked on the Robosuite dataset, ReinforceGen reaches 80% success rate on all tasks with visuomotor controls in the highest reset range setting. Additional ablation studies show that our fine-tuning approaches contributes to an 89% average performance increase. More results and videos available in this https URL

**arXiv ID:** 2512.16861
</details>

<details>
<summary><strong>LADY: Linear Attention for Autonomous Driving Efficiency without Transformers</strong> - Jihao Huang, Xi Xia, Zhiyuan Li, Tianle Liu, Jingke Wang, Junbo Chen, Tengju Ye - [[pdf]](https://arxiv.org/pdf/2512.15038)</summary>

**Abstract:** End-to-end paradigms have demonstrated great potential for autonomous driving. Additionally, most existing methods are built upon Transformer architectures. However, transformers incur a quadratic attention cost, limiting their ability to model long spatial and temporal sequences-particularly on resource-constrained edge platforms. As autonomous driving inherently demands efficient temporal modeling, this challenge severely limits their deployment and real-time performance. Recently, linear attention mechanisms have gained increasing attention due to their superior spatiotemporal complexity. However, existing linear attention architectures are limited to self-attention, lacking support for cross-modal and cross-temporal interactions-both crucial for autonomous driving. In this work, we propose LADY, the first fully linear attention-based generative model for end-to-end autonomous driving. LADY enables fusion of long-range temporal context at inference with constant computational and memory costs, regardless of the history length of camera and LiDAR features. Additionally, we introduce a lightweight linear cross-attention mechanism that enables effective cross-modal information exchange. Experiments on the NAVSIM and Bench2Drive benchmarks demonstrate that LADY achieves state-of-the-art performance with constant-time and memory complexity, offering improved planning performance and significantly reduced computational cost. Additionally, the model has been deployed and validated on edge devices, demonstrating its practicality in resource-limited scenarios.

**arXiv ID:** 2512.15038
</details>

<details>
<summary><strong>NDRL: Cotton Irrigation and Nitrogen Application with Nested Dual-Agent Reinforcement Learning</strong> - Ruifeng Xu, Liang He - [[pdf]](https://arxiv.org/pdf/2512.16408)</summary>

**Abstract:** Effective irrigation and nitrogen fertilization have a significant impact on crop yield. However, existing research faces two limitations: (1) the high complexity of optimizing water-nitrogen combinations during crop growth and poor yield optimization results; and (2) the difficulty in quantifying mild stress signals and the delayed feedback, which results in less precise dynamic regulation of water and nitrogen and lower resource utilization efficiency. To address these issues, we propose a Nested Dual-Agent Reinforcement Learning (NDRL) method. The parent agent in NDRL identifies promising macroscopic irrigation and fertilization actions based on projected cumulative yield benefits, reducing ineffective explorationwhile maintaining alignment between objectives and yield. The child agent's reward function incorporates quantified Water Stress Factor (WSF) and Nitrogen Stress Factor (NSF), and uses a mixed probability distribution to dynamically optimize daily strategies, thereby enhancing both yield and resource efficiency. We used field experiment data from 2023 and 2024 to calibrate and validate the Decision Support System for Agrotechnology Transfer (DSSAT) to simulate real-world conditions and interact with NDRL. Experimental results demonstrate that, compared to the best baseline, the simulated yield increased by 4.7% in both 2023 and 2024, the irrigation water productivity increased by 5.6% and 5.1% respectively, and the nitrogen partial factor productivity increased by 6.3% and 1.0% respectively. Our method advances the development of cotton irrigation and nitrogen fertilization, providing new ideas for addressing the complexity and precision issues in agricultural resource management and for sustainable agricultural development.

**arXiv ID:** 2512.16408
</details>

<details>
<summary><strong>AdaSearch: Balancing Parametric Knowledge and Search in Large Language Models via Reinforcement Learning</strong> - Tzu-Han Lin, Wei-Lin Chen, Chen-An Li, Hung-yi Lee, Yun-Nung Chen, Yu Meng - [[pdf]](https://arxiv.org/pdf/2512.16883)</summary>

**Abstract:** Equipping large language models (LLMs) with search engines via reinforcement learning (RL) has emerged as an effective approach for building search agents. However, overreliance on search introduces unnecessary cost and risks exposure to noisy or malicious content, while relying solely on parametric knowledge risks hallucination. The central challenge is to develop agents that adaptively balance parametric knowledge with external search, invoking search only when necessary. Prior work mitigates search overuse by shaping rewards around the number of tool calls. However, these penalties require substantial reward engineering, provide ambiguous credit assignment, and can be exploited by agents that superficially reduce calls. Moreover, evaluating performance solely through call counts conflates necessary and unnecessary search, obscuring the measurement of true adaptive behavior. To address these limitations, we first quantify the self-knowledge awareness of existing search agents via an F1-based decision metric, revealing that methods such as Search-R1 often overlook readily available parametric knowledge. Motivated by these findings, we propose AdaSearch, a simple two-stage, outcome-driven RL framework that disentangles problem solving from the decision of whether to invoke search, and makes this decision process explicit and interpretable. This transparency is crucial for high-stakes domains such as finance and medical question answering, yet is largely neglected by prior approaches. Experiments across multiple model families and sizes demonstrate that AdaSearch substantially improves knowledge-boundary awareness, reduces unnecessary search calls, preserves strong task performance, and offers more transparent, interpretable decision behaviors.

**arXiv ID:** 2512.16883
</details>

<details>
<summary><strong>Dynamic Rank Reinforcement Learning for Adaptive Low-Rank Multi-Head Self Attention in Large Language Models</strong> - Caner Erden - [[pdf]](https://arxiv.org/pdf/2512.15973)</summary>

**Abstract:** We propose Dynamic Rank Reinforcement Learning (DR-RL), a novel framework that adaptively optimizes the low-rank factorization of Multi-Head Self-Attention (MHSA) in Large Language Models (LLMs) through the integration of reinforcement learning and online matrix perturbation theory. While traditional low-rank approximations often rely on static rank assumptions--limiting their flexibility across diverse input contexts--our method dynamically selects ranks based on real-time sequence dynamics, layer-specific sensitivities, and hardware constraints. The core innovation lies in an RL agent that formulates rank selection as a sequential policy optimization problem, where the reward function strictly balances attention fidelity against computational latency. Crucially, we employ online matrix perturbation bounds to enable incremental rank updates, thereby avoiding the prohibitive cost of full decomposition during inference. Furthermore, the integration of a lightweight Transformer-based policy network and batched Singular Value Decomposition (SVD) operations ensures scalable deployment on modern GPU architectures. Experiments demonstrate that DR-RL maintains downstream accuracy statistically equivalent to full-rank attention while significantly reducing Floating Point Operations (FLOPs), particularly in long-sequence regimes (L > 4096). This work bridges the gap between adaptive efficiency and theoretical rigor in MHSA, offering a principled, mathematically grounded alternative to heuristic rank reduction techniques in resource-constrained deep learning. Source code and experiment logs are available at: this https URL

**arXiv ID:** 2512.15973
</details>

<details>
<summary><strong>Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning</strong> - Zhaowei Liu, Xin Guo, Zhi Yang, Fangqi Lou, Lingfeng Zeng, Mengping Li, Qi Qi, Zhiqiang Liu, Yiyang Han, Dongpo Cheng, Xingdong Feng, Huixia Judy Wang, Chengchun Shi, Liwen Zhang - [[pdf]](https://arxiv.org/pdf/2503.16252)</summary>

**Abstract:** In recent years, general-purpose large language models (LLMs) such as GPT, Gemini, Claude, and DeepSeek have advanced at an unprecedented pace. Despite these achievements, their application to finance remains challenging, due to fragmented data sources, intransparent reasoning processes, and weak transferability to business applications. In response, we introduce Fin-R1, a reasoning LLM designed for financial scenarios. With a compact size of 7 billion parameters, Fin-R1 reduces deployment costs while addressing the aforementioned challenges. Its development follows a two-stage pipeline. First, we construct Fin-R1-Data, a high-quality financial dataset consisting of 60,091 chain-of-thought (CoT) samples, distilled and filtered from multiple authoritative benchmarks to ensure consistency and reliability. Second, we train Fin-R1 using Fin-R1-Data through supervised fine-tuning (SFT), followed by reinforcement learning (RL). This stage substantially improves the model's ability to solve complex financial reasoning tasks, yielding outputs that are both accurate and interpretable. Despite its relatively small parameter scale, Fin-R1 achieves competitive empirical performance across established financial benchmarks and demonstrates practical utility in compliance checking and robo-advisory. Our code is publicly available at this https URL, and has already attracted over 700 stars.

**arXiv ID:** 2503.16252
</details>

<details>
<summary><strong>Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning</strong> - Weiqin Wang, Yile Wang, Kehao Chen, Hui Huang - [[pdf]](https://arxiv.org/pdf/2512.15146)</summary>

**Abstract:** Test-time reinforcement learning mitigates the reliance on annotated data by using majority voting results as pseudo-labels, emerging as a complementary direction to reinforcement learning with verifiable rewards (RLVR) for improving reasoning ability of large language models (LLMs). However, this voting strategy often induces confirmation bias and suffers from sparse rewards, limiting the overall performance. In this work, we propose subgroup-specific step-wise confidence-weighted pseudo-label estimation (SCOPE), a framework integrating model confidence and dynamic subgroup partitioning to address these issues. Specifically, SCOPE integrates the proposed step-wise confidence into pseudo label deduction, prioritizing high-quality reasoning paths over simple frequency count. Furthermore, it dynamically partitions the candidate outputs pool into independent subgroups by balancing reasoning quality against exploration diversity. By deriving local consensus via repeat sampling for each sub group, SCOPE provides diverse supervision targets to encourage broader exploration. We conduct experiments across various models and benchmarks, experimental results show that SCOPE consistently outperforms recent baselines. Notably, SCOPE achieving relative improvements of 13.1% on challenging AIME 2025 and 8.1% on AMC. The code is released at this https URL.

**arXiv ID:** 2512.15146
</details>

<details>
<summary><strong>Reinforcement Learning Finetunes Small Subnetworks in Large Language Models</strong> - Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tur, Hao Peng - [[pdf]](https://arxiv.org/pdf/2505.11711)</summary>

**Abstract:** Reinforcement learning (RL) yields substantial improvements in large language models (LLMs) downstream task performance and alignment with human values. Surprisingly, such large gains result from updating only a small subnetwork comprising just 5 percent to 30 percent of the parameters, with the rest effectively unchanged. We refer to this phenomenon as parameter update sparsity induced by RL. It is observed across all 7 widely used RL algorithms (e.g., PPO, GRPO, DPO) and all 10 LLMs from different families in our experiments. This sparsity is intrinsic and occurs without any explicit sparsity promoting regularizations or architectural constraints. Finetuning the subnetwork alone recovers the test accuracy, and, remarkably, produces a model nearly identical to the one obtained via full finetuning. The subnetworks from different random seeds, training data, and even RL algorithms show substantially greater overlap than expected by chance. Our analysis suggests that this sparsity is not due to updating only a subset of layers, instead, nearly all parameter matrices receive similarly sparse updates. Moreover, the updates to almost all parameter matrices are nearly full-rank, suggesting RL updates a small subset of parameters that nevertheless span almost the full subspaces that the parameter matrices can represent. We conjecture that the this update sparsity can be primarily attributed to training on data that is near the policy distribution, techniques that encourage the policy to remain close to the pretrained model, such as the KL regularization and gradient clipping, have limited impact.

**arXiv ID:** 2505.11711
</details>

<details>
<summary><strong>UAMDP: Uncertainty-Aware Markov Decision Process for Risk-Constrained Reinforcement Learning from Probabilistic Forecasts</strong> - Michal Koren, Or Peretz, Tai Dinh, Philip S. Yu - [[pdf]](https://arxiv.org/pdf/2510.08226)</summary>

**Abstract:** Sequential decisions in volatile, high-stakes settings require more than maximizing expected return; they require principled uncertainty management. This paper presents the Uncertainty-Aware Markov Decision Process (UAMDP), a unified framework that couples Bayesian forecasting, posterior-sampling reinforcement learning, and planning under a conditional value-at-risk (CVaR) constraint. In a closed loop, the agent updates its beliefs over latent dynamics, samples plausible futures via Thompson sampling, and optimizes policies subject to preset risk tolerances. We establish regret bounds that converge to the Bayes-optimal benchmark under standard regularity conditions. We evaluate UAMDP in two domains including high-frequency equity trading and retail inventory control, both marked by structural uncertainty and economic volatility. Relative to strong deep learning baselines, UAMDP improves long-horizon forecasting accuracy (RMSE decreases by up to 25% and sMAPE by 32%), and these gains translate into economic performance: the trading Sharpe ratio rises from 1.54 to 1.74 while maximum drawdown is roughly halved. These results show that integrating calibrated probabilistic modeling, exploration aligned with posterior uncertainty, and risk-aware control yields a robust, generalizable approach to safer and more profitable sequential decision-making.

**arXiv ID:** 2510.08226
</details>

<details>
<summary><strong>EnviSAgE: A Survey of Environment Scaling for Qualitative Agentic Experience Collection</strong> - Yuchen Huang, Sijia Li, Minghao Liu, Wei Liu, Shijue Huang, Zhiyuan Fan, Hou Pong Chan, Yi R. Fung - [[pdf]](https://arxiv.org/pdf/2511.09586)</summary>

**Abstract:** LLM-based agents can autonomously accomplish complex tasks across various domains. However, to further cultivate capabilities such as adaptive behavior and long-term decision-making, training on static datasets built from human-level knowledge is insufficient. These datasets are costly to construct and lack both dynamism and realism. A growing consensus is that agents should instead interact directly with environments and learn from experience through reinforcement learning. We formalize this iterative process as the Generation-Execution-Feedback (GEF) loop, where environments generate tasks to challenge agents, return observations in response to agents' actions during task execution, and provide evaluative feedback on rollouts for subsequent learning. Under this paradigm, environments function as indispensable producers of experiential data, highlighting the need to scale them toward greater complexity, realism, and interactivity. In this survey, we systematically review representative methods for environment scaling from a pioneering environment-centric perspective and organize them along the stages of the GEF loop, namely task generation, task execution, and feedback. We further analyze implementation frameworks, challenges, and applications, consolidating fragmented advances and outlining future research directions for agent intelligence.

**arXiv ID:** 2511.09586
</details>

<details>
<summary><strong>MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Model for Embodied Task Planning</strong> - Yuanchen Ju, Yongyuan Liang, Yen-Jen Wang, Nandiraju Gireesh, Yuanliang Ju, Seungjae Lee, Qiao Gu, Elvis Hsieh, Furong Huang, Koushil Sreenath - [[pdf]](https://arxiv.org/pdf/2512.16909)</summary>

**Abstract:** Mobile manipulators in households must both navigate and manipulate. This requires a compact, semantically rich scene representation that captures where objects are, how they function, and which parts are actionable. Scene graphs are a natural choice, yet prior work often separates spatial and functional relations, treats scenes as static snapshots without object states or temporal updates, and overlooks information most relevant for accomplishing the current task. To address these limitations, we introduce MomaGraph, a unified scene representation for embodied agents that integrates spatial-functional relationships and part-level interactive elements. However, advancing such a representation requires both suitable data and rigorous evaluation, which have been largely missing. We thus contribute MomaGraph-Scenes, the first large-scale dataset of richly annotated, task-driven scene graphs in household environments, along with MomaGraph-Bench, a systematic evaluation suite spanning six reasoning capabilities from high-level planning to fine-grained scene understanding. Built upon this foundation, we further develop MomaGraph-R1, a 7B vision-language model trained with reinforcement learning on MomaGraph-Scenes. MomaGraph-R1 predicts task-oriented scene graphs and serves as a zero-shot task planner under a Graph-then-Plan framework. Extensive experiments demonstrate that our model achieves state-of-the-art results among open-source models, reaching 71.6% accuracy on the benchmark (+11.4% over the best baseline), while generalizing across public benchmarks and transferring effectively to real-robot experiments.

**arXiv ID:** 2512.16909
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
