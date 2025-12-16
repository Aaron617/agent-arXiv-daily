# Agent arXiv Daily

**Last Updated:** 2025-12-16 03:00:10

**Total Papers:** 49

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (1 papers)</h2></summary>

<details>
<summary><strong>When Actions Teach You to Think: Reasoning-Action Synergy via Reinforcement Learning in Conversational Agents</strong> - Mrinal Rawat, Arkajyoti Chakraborty, Neha Gupta, Roberto Pieraccini - [[pdf]](https://arxiv.org/pdf/2512.11277)</summary>

**Abstract:** 

**arXiv ID:** 2512.11277
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (5 papers)</h2></summary>

<details>
<summary><strong>MedAI: Evaluating TxAgent's Therapeutic Agentic Reasoning in the NeurIPS CURE-Bench Competition</strong> - Tim Cofala, Christian Kalfar, Jingge Xiao, Johanna Schrader, Michelle Tang, Wolfgang Nejdl - [[pdf]](https://arxiv.org/pdf/2512.11682)</summary>

**Abstract:** Therapeutic decision-making in clinical medicine constitutes a high-stakes domain in which AI guidance interacts with complex interactions among patient characteristics, disease processes, and pharmacological agents. Tasks such as drug recommendation, treatment planning, and adverse-effect prediction demand robust, multi-step reasoning grounded in reliable biomedical knowledge. Agentic AI methods, exemplified by TxAgent, address these challenges through iterative retrieval-augmented generation (RAG). TxAgent employs a fine-tuned Llama-3.1-8B model that dynamically generates and executes function calls to a unified biomedical tool suite (ToolUniverse), integrating FDA Drug API, OpenTargets, and Monarch resources to ensure access to current therapeutic information. In contrast to general-purpose RAG systems, medical applications impose stringent safety constraints, rendering the accuracy of both the reasoning trace and the sequence of tool invocations critical. These considerations motivate evaluation protocols treating token-level reasoning and tool-usage behaviors as explicit supervision signals. This work presents insights derived from our participation in the CURE-Bench NeurIPS 2025 Challenge, which benchmarks therapeutic-reasoning systems using metrics that assess correctness, tool utilization, and reasoning quality. We analyze how retrieval quality for function (tool) calls influences overall model performance and demonstrate performance gains achieved through improved tool-retrieval strategies. Our work was awarded the Excellence Award in Open Science. Complete information can be found at this https URL.

**arXiv ID:** 2512.11682
</details>

<details>
<summary><strong>Scalable Data Synthesis for Computer Use Agents with Step-Level Filtering</strong> - Yifei He, Pranit Chawla, Yaser Souri, Subhojit Som, Xia Song - [[pdf]](https://arxiv.org/pdf/2512.10962)</summary>

**Abstract:** Computer use agents (CUAs) can operate real-world digital interfaces but remain difficult to train due to the high cost of graphical user interface (GUI) interaction and the scarcity of high-quality trajectory data. Existing datasets rely on human demonstrations, limiting scalability. A natural alternative is to synthesize data from strong CUAs, yet their rollouts are highly noisy, with incorrect or suboptimal actions consisting a large proportion of the steps, making naive imitation ineffective. To tackle this challenge, we introduce a scalable data synthesis pipeline that transforms noisy rollouts into reliable supervision without human annotation. The core idea is step-level filtering, which evaluates actions individually to retain only correct steps, complemented by reasoning augmentation for improved planning. Using this pipeline, we construct WebSTAR, a dataset of 13.3K trajectories and 100K graded, reasoning-rich steps synthesized from OpenAI's computer-use-preview model. We train Qwen-2.5-VL-Instruct models (7B and 32B) on WebSTAR. On WebVoyager, our 7B model surpasses SoTA open-source CUA model UI-TARS-1.5-7B by more than 15% with only supervised finetuning. Building on step-level grading, we further create WebSCORE, a dataset of graded step-level actions, and train StepRM, a 7B multimodal reward model distilled from o4-mini, which matches its grading quality while being far more efficient to deploy at scale. Our results establish step-level filtering as a key principle for scalable CUA training and construct two new datasets (WebSTAR, WebSCORE) and a lightweight reward model (StepRM) as practical tools to advance robust and efficient CUAs.

**arXiv ID:** 2512.10962
</details>

<details>
<summary><strong>UpBench: A Dynamically Evolving Real-World Labor-Market Agentic Benchmark Framework Built for Human-Centric AI</strong> - Darvin Yi, Teng Liu, Mattie Terzolo, Lance Hasson, Ayan Sinha, Pablo Mendes, Andrew Rabinovich - [[pdf]](https://arxiv.org/pdf/2511.12306)</summary>

**Abstract:** 

**arXiv ID:** 2511.12306
</details>

<details>
<summary><strong>SDialog: A Python Toolkit for End-to-End Agent Building, User Simulation, Dialog Generation, and Evaluation</strong> - Sergio Burdisso, Séverin Baroudi, Yanis Labrak, David Grunert, Pawel Cyrta, Yiyang Chen, Srikanth Madikeri, Esaú Villatoro-Tello, Thomas Schaaf, Ricard Marxer, Petr Motlicek - [[pdf]](https://arxiv.org/pdf/2512.09142)</summary>

**Abstract:** 

**arXiv ID:** 2512.09142
</details>

<details>
<summary><strong>SDialog: A Python Toolkit for End-to-End Agent Building, User Simulation, Dialog Generation, and Evaluation</strong> - Sergio Burdisso, Séverin Baroudi, Yanis Labrak, David Grunert, Pawel Cyrta, Yiyang Chen, Srikanth Madikeri, Esaú Villatoro-Tello, Thomas Schaaf, Ricard Marxer, Petr Motlicek - [[pdf]](https://arxiv.org/pdf/2506.10622)</summary>

**Abstract:** 

**arXiv ID:** 2506.10622
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>Towards Trustworthy Multi-Turn LLM Agents via Behavioral Guidance</strong> - Gonca Gürsun - [[pdf]](https://arxiv.org/pdf/2512.11421)</summary>

**Abstract:** 

**arXiv ID:** 2512.11421
</details>

<details>
<summary><strong>Achieving Olympia-Level Geometry Large Language Model Agent via Complexity Boosting Reinforcement Learning</strong> - Haiteng Zhao, Junhao Shen, Yiming Zhang, Songyang Gao, Kuikun Liu, Tianyou Ma, Fan Zheng, Dahua Lin, Wenwei Zhang, Kai Chen - [[pdf]](https://arxiv.org/pdf/2512.10534)</summary>

**Abstract:** 

**arXiv ID:** 2512.10534
</details>

<details>
<summary><strong>Zero-shot 3D Map Generation with LLM Agents: A Dual-Agent Architecture for Procedural Content Generation</strong> - Lim Chien Her, Ming Yan, Yunshu Bai, Ruihao Li, Hao Zhang - [[pdf]](https://arxiv.org/pdf/2512.10501)</summary>

**Abstract:** 

**arXiv ID:** 2512.10501
</details>

<details>
<summary><strong>Large Language Model Agent for Modular Task Execution in Drug Discovery</strong> - Janghoon Ock, Radheesh Sharma Meda, Srivathsan Badrinarayanan, Neha S. Aluru, Achuth Chandrasekhar, Amir Barati Farimani - [[pdf]](https://arxiv.org/pdf/2507.02925)</summary>

**Abstract:** 

**arXiv ID:** 2507.02925
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (12 papers)</h2></summary>

<details>
<summary><strong>FutureWeaver: Planning Test-Time Compute for Multi-Agent Systems with Modularized Collaboration</strong> - Dongwon Jung, Peng Shi, Yi Zhang - [[pdf]](https://arxiv.org/pdf/2512.11213)</summary>

**Abstract:** 

**arXiv ID:** 2512.11213
</details>

<details>
<summary><strong>TriFlow: A Progressive Multi-Agent Framework for Intelligent Trip Planning</strong> - Yuxing Chen, Basem Suleiman, Qifan Chen - [[pdf]](https://arxiv.org/pdf/2512.11271)</summary>

**Abstract:** 

**arXiv ID:** 2512.11271
</details>

<details>
<summary><strong>AgentBalance: Backbone-then-Topology Design for Cost-Effective Multi-Agent Systems under Budget Constraints</strong> - Shuowei Cai, Yansong Ning, Hao Liu - [[pdf]](https://arxiv.org/pdf/2512.11426)</summary>

**Abstract:** Large Language Model (LLM)-based multi-agent systems (MAS) are becoming indispensable building blocks for web-scale applications such as web search, social network analytics, and online customer support, where cost-effectiveness is increasingly the primary constraint for large-scale deployment. While recent work improves MAS cost-effectiveness by shaping inter-agent communication topologies and selecting agent backbones, it rarely models and optimizes under explicit token-cost and latency budgets that reflect deployment constraints. This often leads to topology-first designs and suboptimal cost-effectiveness when budgets are binding. We present AgentBalance, a framework for constructing cost-effective MAS under explicit token-cost and latency budgets via a backbone-then-topology design. AgentBalance first performs backbone-oriented agent generation, constructing agents with heterogeneous backbones through LLM pool construction, pool selection, and role-backbone matching. It then performs adaptive MAS topology generation, guiding inter-agent communication via agent representation learning, gating, and latency-aware topology synthesis. Experiments on benchmarks with 14 candidate LLM backbones show that AgentBalance achieves up to 10% and 22% performance gains under matched token-cost and latency budgets, respectively, and yields strong AUC on performance-versus-budget curves across benchmarks. AgentBalance also functions as a plug-in for existing MAS, improving performance under the same token-cost and latency constraints, and it generalizes well to unseen LLMs for practical, budget-aware deployment. Code: this https URL

**arXiv ID:** 2512.11426
</details>

<details>
<summary><strong>Agent-Based Modular Learning for Multimodal Emotion Recognition in Human-Agent Systems</strong> - Matvey Nepomnyaschiy, Oleg Pereziabov, Anvar Tliamov, Stanislav Mikhailov, Ilya Afanasyev - [[pdf]](https://arxiv.org/pdf/2512.10975)</summary>

**Abstract:** Effective human-agent interaction (HAI) relies on accurate and adaptive perception of human emotional states. While multimodal deep learning models - leveraging facial expressions, speech, and textual cues - offer high accuracy in emotion recognition, their training and maintenance are often computationally intensive and inflexible to modality changes. In this work, we propose a novel multi-agent framework for training multimodal emotion recognition systems, where each modality encoder and the fusion classifier operate as autonomous agents coordinated by a central supervisor. This architecture enables modular integration of new modalities (e.g., audio features via emotion2vec), seamless replacement of outdated components, and reduced computational overhead during training. We demonstrate the feasibility of our approach through a proof-of-concept implementation supporting vision, audio, and text modalities, with the classifier serving as a shared decision-making agent. Our framework not only improves training efficiency but also contributes to the design of more flexible, scalable, and maintainable perception modules for embodied and virtual agents in HAI scenarios.

**arXiv ID:** 2512.10975
</details>

<details>
<summary><strong>Agile Flight Emerges from Multi-Agent Competitive Racing</strong> - Vineet Pasumarti, Lorenzo Bianchi, Antonio Loquercio - [[pdf]](https://arxiv.org/pdf/2512.11781)</summary>

**Abstract:** 

**arXiv ID:** 2512.11781
</details>

<details>
<summary><strong>From Bits to Boardrooms: A Cutting-Edge Multi-Agent LLM Framework for Business Excellence</strong> - Zihao Wang, Junming Zhang - [[pdf]](https://arxiv.org/pdf/2508.15447)</summary>

**Abstract:** 

**arXiv ID:** 2508.15447
</details>

<details>
<summary><strong>CREW-WILDFIRE: Benchmarking Agentic Multi-Agent Collaborations at Scale</strong> - Jonathan Hyun, Nicholas R Waytowich, Boyuan Chen - [[pdf]](https://arxiv.org/pdf/2507.05178)</summary>

**Abstract:** 

**arXiv ID:** 2507.05178
</details>

<details>
<summary><strong>MTTR-A: Measuring Cognitive Recovery Latency in Multi-Agent Systems</strong> - Barak Or - [[pdf]](https://arxiv.org/pdf/2511.20663)</summary>

**Abstract:** 

**arXiv ID:** 2511.20663
</details>

<details>
<summary><strong>Understanding LLM Agent Behaviours via Game Theory: Strategy Recognition, Biases and Multi-Agent Dynamics</strong> - Trung-Kiet Huynh, Duy-Minh Dao-Sy, Thanh-Bang Cao, Phong-Hao Le, Hong-Dan Nguyen, Phu-Quy Nguyen-Lam, Minh-Luan Nguyen-Vo, Hong-Phat Pham, Phu-Hoa Pham, Thien-Kim Than, Chi-Nguyen Tran, Huy Tran, Gia-Thoai Tran-Le, Alessio Buscemi, Le Hong Trang, Anh Han - [[pdf]](https://arxiv.org/pdf/2512.07462)</summary>

**Abstract:** 

**arXiv ID:** 2512.07462
</details>

<details>
<summary><strong>Query Optimization Beyond Data Systems: The Case for Multi-Agent Systems</strong> - Zoi Kaoudi, Ioana Giurgiu - [[pdf]](https://arxiv.org/pdf/2512.11001)</summary>

**Abstract:** 

**arXiv ID:** 2512.11001
</details>

<details>
<summary><strong>Bandwidth-constrained Variational Message Encoding for Cooperative Multi-agent Reinforcement Learning</strong> - Wei Duan, Jie Lu, En Yu, Junyu Xuan - [[pdf]](https://arxiv.org/pdf/2512.11179)</summary>

**Abstract:** 

**arXiv ID:** 2512.11179
</details>

<details>
<summary><strong>AutoFSM: A Multi-agent Framework for FSM Code Generation with IR and SystemC-Based Testing</strong> - Qiuming Luo, Yanming Lei, Kunzhong Wu, Yixuan Cao, Chengjian Liu - [[pdf]](https://arxiv.org/pdf/2512.11398)</summary>

**Abstract:** 

**arXiv ID:** 2512.11398
</details>

</details>

<details open>
<summary><h2>Other Agent Research (12 papers)</h2></summary>

<details>
<summary><strong>MiniScope: A Least Privilege Framework for Authorizing Tool Calling Agents</strong> - Jinhao Zhu, Kevin Tseng, Gil Vernik, Xiao Huang, Shishir G. Patil, Vivian Fang, Raluca Ada Popa - [[pdf]](https://arxiv.org/pdf/2512.11147)</summary>

**Abstract:** 

**arXiv ID:** 2512.11147
</details>

<details>
<summary><strong>Atomic Action Slicing: Planner-Aligned Options for Generalist VLA Agents</strong> - Stefan Tabakov, Asen Popov, Dimitar Dimitrov, S. Ensiye Kiyamousavi, Vladimir Hristov, Boris Kraychev - [[pdf]](https://arxiv.org/pdf/2512.11584)</summary>

**Abstract:** 

**arXiv ID:** 2512.11584
</details>

<details>
<summary><strong>Words to Describe What I'm Feeling: Exploring the Potential of AI Agents for High Subjectivity Decisions in Advance Care Planning</strong> - Kellie Yu Hui Sim, Pin Sym Foong, Chenyu Zhao, Melanie Yi Ning Quek, Swarangi Subodh Mehta, Kenny Tsu Wei Choo - [[pdf]](https://arxiv.org/pdf/2512.11276)</summary>

**Abstract:** 

**arXiv ID:** 2512.11276
</details>

<details>
<summary><strong>Octopus: Agentic Multimodal Reasoning with Six-Capability Orchestration</strong> - Yifu Guo, Zishan Xu, Zhiyuan Yao, Yuquan Lu, Jiaye Lin, Sen Hu, Zhenheng Tang, Huacan Wang, Ronghao Chen - [[pdf]](https://arxiv.org/pdf/2511.15351)</summary>

**Abstract:** 

**arXiv ID:** 2511.15351
</details>

<details>
<summary><strong>EpiPlanAgent: Agentic Automated Epidemic Response Planning</strong> - Kangkun Mao, Fang Xu, Jinru Ding, Yidong Jiang, Yujun Yao, Yirong Chen, Junming Liu, Xiaoqin Wu, Qian Wu, Xiaoyan Huang, Jie Xu - [[pdf]](https://arxiv.org/pdf/2512.10313)</summary>

**Abstract:** 

**arXiv ID:** 2512.10313
</details>

<details>
<summary><strong>Driving Through Uncertainty: Risk-Averse Control with LLM Commonsense for Autonomous Driving under Perception Deficits</strong> - Yuting Hu, Chenhui Xu, Ruiyang Qin, Dancheng Liu, Amir Nassereldine, Yiyu Shi, Jinjun Xiong - [[pdf]](https://arxiv.org/pdf/2503.07020)</summary>

**Abstract:** 

**arXiv ID:** 2503.07020
</details>

<details>
<summary><strong>RECAP: REwriting Conversations for Intent Understanding in Agentic Planning</strong> - Kushan Mitra, Dan Zhang, Hannah Kim, Estevam Hruschka - [[pdf]](https://arxiv.org/pdf/2509.04472)</summary>

**Abstract:** 

**arXiv ID:** 2509.04472
</details>

<details>
<summary><strong>Confucius Code Agent: An Open-sourced AI Software Engineer at Industrial Scale</strong> - Zhaodong Wang, Zhenting Qi, Sherman Wong, Nathan Hu, Samuel Lin, Jun Ge, Erwin Gao, Yining Yang, Ben Maurer, Wenlin Chen, David Recordon, Yilun Du, Minlan Yu, Ying Zhang - [[pdf]](https://arxiv.org/pdf/2512.10398)</summary>

**Abstract:** 

**arXiv ID:** 2512.10398
</details>

<details>
<summary><strong>Evaluating Cooperative Resilience in Multiagent Systems: A Comparison Between Humans and LLMs</strong> - Manuela Chacon-Chamorro, Juan Sebastián Pinzón, Rubén Manrique, Luis Felipe Giraldo, Nicanor Quijano - [[pdf]](https://arxiv.org/pdf/2512.11689)</summary>

**Abstract:** 

**arXiv ID:** 2512.11689
</details>

<details>
<summary><strong>Osprey: Production-Ready Agentic AI for Safety-Critical Control Systems</strong> - Thorsten Hellert, João Montenegro, Antonin Sulc - [[pdf]](https://arxiv.org/pdf/2508.15066)</summary>

**Abstract:** 

**arXiv ID:** 2508.15066
</details>

<details>
<summary><strong>Architecting Large Action Models for Human-in-the-Loop Intelligent Robots</strong> - Kanisorn Sangchai, Methasit Boonpun, Withawin Kraipetchara, Paulo Garcia - [[pdf]](https://arxiv.org/pdf/2512.11620)</summary>

**Abstract:** 

**arXiv ID:** 2512.11620
</details>

<details>
<summary><strong>Two-dimensional Decompositions of High-dimensional Configurations for Efficient Multi-vehicle Coordination at Intelligent Intersections</strong> - Amirreza Akbari, Johan Thunberg - [[pdf]](https://arxiv.org/pdf/2512.11713)</summary>

**Abstract:** 

**arXiv ID:** 2512.11713
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>Long-horizon Reasoning Agent for Olympiad-Level Mathematical Problem Solving</strong> - Songyang Gao, Yuzhe Gu, Zijian Wu, Lingkai Kong, Wenwei Zhang, Zhongrui Cai, Fan Zheng, Tianyou Ma, Junhao Shen, Haiteng Zhao, Duanyang Zhang, Huilun Zhang, Kuikun Liu, Chengqi Lyu, Yanhui Duan, Chiyu Chen, Ningsheng Ma, Jianfei Gao, Han Lyu, Dahua Lin, Kai Chen - [[pdf]](https://arxiv.org/pdf/2512.10739)</summary>

**Abstract:** 

**arXiv ID:** 2512.10739
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (14 papers)</h2></summary>

<details>
<summary><strong>CORL: Reinforcement Learning of MILP Policies Solved via Branch and Bound</strong> - Akhil S Anand, Elias Aarekol, Martin Mziray Dalseg, Magnus Stalhane, Sebastien Gros - [[pdf]](https://arxiv.org/pdf/2512.11169)</summary>

**Abstract:** 

**arXiv ID:** 2512.11169
</details>

<details>
<summary><strong>A-LAMP: Agentic LLM-Based Framework for Automated MDP Modeling and Policy Generation</strong> - Hong Je-Gal, Chan-Bin Yi, Hyun-Suk Lee - [[pdf]](https://arxiv.org/pdf/2512.11270)</summary>

**Abstract:** 

**arXiv ID:** 2512.11270
</details>

<details>
<summary><strong>ICPO: Intrinsic Confidence-Driven Group Relative Preference Optimization for Efficient Reinforcement Learning</strong> - Jinpeng Wang, Chao Li, Ting Ye, Mengyuan Zhang, Wei Liu, Jian Luan - [[pdf]](https://arxiv.org/pdf/2511.21005)</summary>

**Abstract:** 

**arXiv ID:** 2511.21005
</details>

<details>
<summary><strong>SATURN: SAT-based Reinforcement Learning to Unleash LLMs Reasoning</strong> - Huanyu Liu, Ge Li, Jia Li, Hao Zhu, Kechi Zhang, Yihong Dong - [[pdf]](https://arxiv.org/pdf/2505.16368)</summary>

**Abstract:** 

**arXiv ID:** 2505.16368
</details>

<details>
<summary><strong>Aligning Humans and Robots via Reinforcement Learning from Implicit Human Feedback</strong> - Suzie Kim, Hye-Bin Shin, Seong-Whan Lee - [[pdf]](https://arxiv.org/pdf/2507.13171)</summary>

**Abstract:** 

**arXiv ID:** 2507.13171
</details>

<details>
<summary><strong>Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning</strong> - Alexander W. Goodall, Edwin Hamel-De le Court, Francesco Belardinelli - [[pdf]](https://arxiv.org/pdf/2511.10843)</summary>

**Abstract:** 

**arXiv ID:** 2511.10843
</details>

<details>
<summary><strong>CUDA-L2: Surpassing cuBLAS Performance for Matrix Multiplication through Reinforcement Learning</strong> - Songqiao Su, Xiaofei Sun, Xiaoya Li, Albert Wang, Jiwei Li, Chris Shum - [[pdf]](https://arxiv.org/pdf/2512.02551)</summary>

**Abstract:** 

**arXiv ID:** 2512.02551
</details>

<details>
<summary><strong>Multi-Objective Reinforcement Learning for Large-Scale Mixed Traffic Control</strong> - Iftekharul Islam, Weizi Li - [[pdf]](https://arxiv.org/pdf/2512.11247)</summary>

**Abstract:** 

**arXiv ID:** 2512.11247
</details>

<details>
<summary><strong>Annotation-Free Reinforcement Learning Query Rewriting via Verifiable Search Reward</strong> - Sungguk Cha, DongWook Kim, Taeseung Hahn, Mintae Kim, Youngsub Han, Byoung-Ki Jeon - [[pdf]](https://arxiv.org/pdf/2507.23242)</summary>

**Abstract:** 

**arXiv ID:** 2507.23242
</details>

<details>
<summary><strong>TECM*: A Data-Driven Assessment to Reinforcement Learning Methods and Application to Heparin Treatment Strategy for Surgical Sepsis</strong> - Jiang Liu, Yujie Li, Chan Zhou, Yihao Xie, Qilong Sun, Xin Shu, Peiwei Li, Chunyong Yang, Yiziting Zhu, Jiaqi Zhu, Yuwen Chen, Bo An, Hao Wu, Bin Yi - [[pdf]](https://arxiv.org/pdf/2512.10973)</summary>

**Abstract:** 

**arXiv ID:** 2512.10973
</details>

<details>
<summary><strong>Equilibrium Policy Generalization: A Reinforcement Learning Framework for Cross-Graph Zero-Shot Generalization in Pursuit-Evasion Games</strong> - Runyu Lu, Peng Zhang, Ruochuan Shi, Yuanheng Zhu, Dongbin Zhao, Yang Liu, Dong Wang, Cesare Alippi - [[pdf]](https://arxiv.org/pdf/2511.00811)</summary>

**Abstract:** 

**arXiv ID:** 2511.00811
</details>

<details>
<summary><strong>DAPO: Design Structure-Aware Pass Ordering in High-Level Synthesis with Graph Contrastive and Reinforcement Learning</strong> - Jinming Ge, Linfeng Du, Likith Anaparty, Shangkun Li, Tingyuan Liang, Afzal Ahmad, Vivek Chaturvedi, Sharad Sinha, Zhiyao Xie, Jiang Xu, Wei Zhang - [[pdf]](https://arxiv.org/pdf/2512.11342)</summary>

**Abstract:** 

**arXiv ID:** 2512.11342
</details>

<details>
<summary><strong>From "Thumbs Up" to "10 out of 10": Reconsidering Scalar Feedback in Interactive Reinforcement Learning</strong> - Hang Yu, Reuben M. Aronson, Katherine H. Allen, Elaine Schaertl Short - [[pdf]](https://arxiv.org/pdf/2311.10284)</summary>

**Abstract:** 

**arXiv ID:** 2311.10284
</details>

<details>
<summary><strong>Model-Based Lookahead Reinforcement Learning for in-hand manipulation</strong> - Alexandre Lopes, Catarina Barata, Plinio Moreno - [[pdf]](https://arxiv.org/pdf/2510.08884)</summary>

**Abstract:** 

**arXiv ID:** 2510.08884
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
