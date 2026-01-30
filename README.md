# Agent arXiv Daily

**Last Updated:** 2026-01-30 03:35:51

**Total Papers:** 100

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
<summary><strong>Liquid Interfaces: A Dynamic Ontology for the Interoperability of Autonomous Systems</strong> - Dhiogo de Sá, Carlos Schmiedel, Carlos Pereira Lopes - [[pdf]](https://arxiv.org/pdf/2601.21993)</summary>

**Abstract:** Contemporary software architectures struggle to support autonomous agents whose reasoning is adaptive, probabilistic, and context-dependent, while system integration remains dominated by static interfaces and deterministic contracts. This paper introduces Liquid Interfaces, a coordination paradigm in which interfaces are not persistent technical artifacts, but ephemeral relational events that emerge through intention articulation and semantic negotiation at this http URL formalize this model and present the Liquid Interface Protocol (LIP),which governs intention-driven interaction, negotiated execution, and enforce ephemerality under semantic uncertainty. We further discuss the governance implications of this approach and describe a reference architecture that demonstrates practical feasibility. Liquid Interfaces provide a principled foundation for adaptive coordination in agent-based systems

**arXiv ID:** 2601.21993
</details>

<details>
<summary><strong>Training slow silicon neurons to control extremely fast robots with spiking reinforcement learning</strong> - Irene Ambrosini, Ingo Blakowski, Dmitrii Zendrikov, Cristiano Capone, Luna Gava, Giacomo Indiveri, Chiara De Luca, Chiara Bartolozzi - [[pdf]](https://arxiv.org/pdf/2601.21548)</summary>

**Abstract:** Air hockey demands split-second decisions at high puck velocities, a challenge we address with a compact network of spiking neurons running on a mixed-signal analog/digital neuromorphic processor. By co-designing hardware and learning algorithms, we train the system to achieve successful puck interactions through reinforcement learning in a remarkably small number of trials. The network leverages fixed random connectivity to capture the task's temporal structure and adopts a local e-prop learning rule in the readout layer to exploit event-driven activity for fast and efficient learning. The result is real-time learning with a setup comprising a computer and the neuromorphic chip in-the-loop, enabling practical training of spiking neural networks for robotic autonomous systems. This work bridges neuroscience-inspired hardware with real-world robotic control, showing that brain-inspired approaches can tackle fast-paced interaction tasks while supporting always-on learning in intelligent machines.

**arXiv ID:** 2601.21548
</details>

<details>
<summary><strong>Empowering Scientific Workflows with Federated Agents</strong> - Alok Kamatar, J. Gregory Pauloski, Yadu Babuji, Ryan Chard, Mansi Sakarvadia, Daniel Babnigg, Kyle Chard, Ian Foster - [[pdf]](https://arxiv.org/pdf/2505.05428)</summary>

**Abstract:** Agentic systems, in which diverse agents cooperate to tackle challenging problems, are exploding in popularity in the AI community. However, existing agentic frameworks take a relatively narrow view of agents, apply a centralized model, and target conversational, cloud-native applications (e.g., LLM-based AI chatbots). In contrast, scientific applications require myriad agents be deployed and managed across diverse cyberinfrastructure. Here we introduce Academy, a modular and extensible middleware designed to deploy autonomous agents across the federated research ecosystem, including HPC systems, experimental facilities, and data repositories. To meet the demands of scientific computing, Academy supports asynchronous execution, heterogeneous resources, high-throughput data flows, and dynamic resource availability. It provides abstractions for expressing stateful agents, managing inter-agent coordination, and integrating computation with experimental control. We present microbenchmark results that demonstrate high performance and scalability in HPC environments. To explore the breadth of applications that can be supported by agentic workflow designs, we also present case studies in materials discovery, astronomy, decentralized learning, and information extraction in which agents are deployed across diverse HPC systems.

**arXiv ID:** 2505.05428
</details>

<details>
<summary><strong>Enhancing Conversational Agents via Task-Oriented Adversarial Memory Adaptation</strong> - Yimin Deng, Yuqing Fu, Derong Xu, Yejing Wang, Wei Ni, Jingtong Gao, Xiaopeng Li, Chengxu Liu, Xiao Han, Guoshuai Zhao, Xiangyu Zhao, Li Zhu, Xueming Qian - [[pdf]](https://arxiv.org/pdf/2601.21797)</summary>

**Abstract:** Conversational agents struggle to handle long conversations due to context window limitations. Therefore, memory systems are developed to leverage essential historical information. Existing memory systems typically follow a pipeline of offline memory construction and update, and online retrieval. Despite the flexible online phase, the offline phase remains fixed and task-independent. In this phase, memory construction operates under a predefined workflow and fails to emphasize task relevant information. Meanwhile, memory updates are guided by generic metrics rather than task specific supervision. This leads to a misalignment between offline memory preparation and task requirements, which undermines downstream task performance. To this end, we propose an Adversarial Memory Adaptation mechanism (AMA) that aligns memory construction and update with task objectives by simulating task execution. Specifically, first, a challenger agent generates question answer pairs based on the original dialogues. The constructed memory is then used to answer these questions, simulating downstream inference. Subsequently, an evaluator agent assesses the responses and performs error analysis. Finally, an adapter agent analyzes the error cases and performs dual level updates on both the construction strategy and the content. Through this process, the memory system receives task aware supervision signals in advance during the offline phase, enhancing its adaptability to downstream tasks. AMA can be integrated into various existing memory systems, and extensive experiments on long dialogue benchmark LoCoMo demonstrate its effectiveness.

**arXiv ID:** 2601.21797
</details>

<details>
<summary><strong>LLM-based Few-Shot Early Rumor Detection with Imitation Agent</strong> - Fengzhu Zeng, Qian Shao, Ling Cheng, Wei Gao, Shih-Fen Cheng, Jing Ma, Cheng Niu - [[pdf]](https://arxiv.org/pdf/2512.18352)</summary>

**Abstract:** Early Rumor Detection (EARD) aims to identify the earliest point at which a claim can be accurately classified based on a sequence of social media posts. This is especially challenging in data-scarce settings. While Large Language Models (LLMs) perform well in few-shot NLP tasks, they are not well-suited for time-series data and are computationally expensive for both training and inference. In this work, we propose a novel EARD framework that combines an autonomous agent and an LLM-based detection model, where the agent acts as a reliable decision-maker for \textit{early time point determination}, while the LLM serves as a powerful \textit{rumor detector}. This approach offers the first solution for few-shot EARD, necessitating only the training of a lightweight agent and allowing the LLM to remain training-free. Extensive experiments on four real-world datasets show our approach boosts performance across LLMs and surpasses existing EARD methods in accuracy and earliness.

**arXiv ID:** 2512.18352
</details>

<details>
<summary><strong>AgentLongBench: A Controllable Long Benchmark For Long-Contexts Agents via Environment Rollouts</strong> - Shicheng Fang, Yuxin Wang, XiaoRan Liu, Jiahao Lu, Chuanyuan Tan, Xinchi Chen, Yining Zheng, Xuanjing Huang, Xipeng Qiu - [[pdf]](https://arxiv.org/pdf/2601.20730)</summary>

**Abstract:** The evolution of Large Language Models (LLMs) into autonomous agents necessitates the management of extensive, dynamic contexts. Current benchmarks, however, remain largely static, relying on passive retrieval tasks that fail to simulate the complexities of agent-environment interaction, such as non-linear reasoning and iterative feedback. To address this, we introduce \textbf{AgentLongBench}, which evaluates agents through simulated environment rollouts based on Lateral Thinking Puzzles. This framework generates rigorous interaction trajectories across knowledge-intensive and knowledge-free scenarios. Experiments with state-of-the-art models and memory systems (32K to 4M tokens) expose a critical weakness: while adept at static retrieval, agents struggle with the dynamic information synthesis essential for workflows. Our analysis indicates that this degradation is driven by the minimum number of tokens required to resolve a query. This factor explains why the high information density inherent in massive tool responses poses a significantly greater challenge than the memory fragmentation typical of long-turn dialogues.

**arXiv ID:** 2601.20730
</details>

<details>
<summary><strong>Auditorily Embodied Conversational Agents: Effects of Spatialization and Situated Audio Cues on Presence and Social Perception</strong> - Yi Fei Cheng, Jarod Bloch, Alexander Wang, Andrea Bianchi, Anusha Withana, Anhong Guo, Laurie M. Heller, David Lindlbauer - [[pdf]](https://arxiv.org/pdf/2601.22082)</summary>

**Abstract:** Embodiment can enhance conversational agents, such as increasing their perceived presence. This is typically achieved through visual representations of a virtual body; however, visual modalities are not always available, such as when users interact with agents using headphones or display-less glasses. In this work, we explore auditory embodiment. By introducing auditory cues of bodily presence - through spatially localized voice and situated Foley audio from environmental interactions - we investigate how audio alone can convey embodiment and influence perceptions of a conversational agent. We conducted a 2 (spatialization: monaural vs. spatialized) x 2 (Foley: none vs. Foley) within-subjects study, where participants (n=24) engaged in conversations with agents. Our results show that spatialization and Foley increase co-presence, but reduce users' perceptions of the agent's attention and other social attributes.

**arXiv ID:** 2601.22082
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (21 papers)</h2></summary>

<details>
<summary><strong>Magellan: Autonomous Discovery of Novel Compiler Optimization Heuristics with AlphaEvolve</strong> - Hongzheng Chen, Alexander Novikov, Ngân Vũ, Hanna Alam, Zhiru Zhang, Aiden Grossman, Mircea Trofin, Amir Yazdanbakhsh - [[pdf]](https://arxiv.org/pdf/2601.21096)</summary>

**Abstract:** Modern compilers rely on hand-crafted heuristics to guide optimization passes. These human-designed rules often struggle to adapt to the complexity of modern software and hardware and lead to high maintenance burden. To address this challenge, we present Magellan, an agentic framework that evolves the compiler pass itself by synthesizing executable C++ decision logic. Magellan couples an LLM coding agent with evolutionary search and autotuning in a closed loop of generation, evaluation on user-provided macro-benchmarks, and refinement, producing compact heuristics that integrate directly into existing compilers. Across several production optimization tasks, Magellan discovers policies that match or surpass expert baselines. In LLVM function inlining, Magellan synthesizes new heuristics that outperform decades of manual engineering for both binary-size reduction and end-to-end performance. In register allocation, it learns a concise priority rule for live-range processing that matches intricate human-designed policies on a large-scale workload. We also report preliminary results on XLA problems, demonstrating portability beyond LLVM with reduced engineering effort.

**arXiv ID:** 2601.21096
</details>

<details>
<summary><strong>Just Ask: Curious Code Agents Reveal System Prompts in Frontier LLMs</strong> - Xiang Zheng, Yutao Wu, Hanxun Huang, Yige Li, Xingjun Ma, Bo Li, Yu-Gang Jiang, Cong Wang - [[pdf]](https://arxiv.org/pdf/2601.21233)</summary>

**Abstract:** Autonomous code agents built on large language models are reshaping software and AI development through tool use, long-horizon reasoning, and self-directed interaction. However, this autonomy introduces a previously unrecognized security risk: agentic interaction fundamentally expands the LLM attack surface, enabling systematic probing and recovery of hidden system prompts that guide model behavior. We identify system prompt extraction as an emergent vulnerability intrinsic to code agents and present \textbf{\textsc{JustAsk}}, a self-evolving framework that autonomously discovers effective extraction strategies through interaction alone. Unlike prior prompt-engineering or dataset-based attacks, \textsc{JustAsk} requires no handcrafted prompts, labeled supervision, or privileged access beyond standard user interaction. It formulates extraction as an online exploration problem, using Upper Confidence Bound--based strategy selection and a hierarchical skill space spanning atomic probes and high-level orchestration. These skills exploit imperfect system-instruction generalization and inherent tensions between helpfulness and safety. Evaluated on \textbf{41} black-box commercial models across multiple providers, \textsc{JustAsk} consistently achieves full or near-complete system prompt recovery, revealing recurring design- and architecture-level vulnerabilities. Our results expose system prompts as a critical yet largely unprotected attack surface in modern agent systems.

**arXiv ID:** 2601.21233
</details>

<details>
<summary><strong>Drive-KD: Multi-Teacher Distillation for VLMs in Autonomous Driving</strong> - Weitong Lian, Zecong Tang, Haoran Li, Tianjian Gao, Yifei Wang, Zixu Wang, Lingyi Meng, Tengju Ru, Zhejun Cui, Yichen Zhu, Hangshuo Cao, Qi Kang, Tianxing Chen, Yusen Qin, Kaixuan Wang, Yu Zhang - [[pdf]](https://arxiv.org/pdf/2601.21288)</summary>

**Abstract:** Autonomous driving is an important and safety-critical task, and recent advances in LLMs/VLMs have opened new possibilities for reasoning and planning in this domain. However, large models demand substantial GPU memory and exhibit high inference latency, while conventional supervised fine-tuning (SFT) often struggles to bridge the capability gaps of small models. To address these limitations, we propose Drive-KD, a framework that decomposes autonomous driving into a "perception-reasoning-planning" triad and transfers these capabilities via knowledge distillation. We identify layer-specific attention as the distillation signal to construct capability-specific single-teacher models that outperform baselines. Moreover, we unify these single-teacher settings into a multi-teacher distillation framework and introduce asymmetric gradient projection to mitigate cross-capability gradient conflicts. Extensive evaluations validate the generalization of our method across diverse model families and scales. Experiments show that our distilled InternVL3-1B model, with ~42 times less GPU memory and ~11.4 times higher throughput, achieves better overall performance than the pretrained 78B model from the same family on DriveBench, and surpasses GPT-5.1 on the planning dimension, providing insights toward efficient autonomous driving VLMs.

**arXiv ID:** 2601.21288
</details>

<details>
<summary><strong>NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents</strong> - Yang Song, Anoushka Vyas, Zirui Wei, Sina Khoshfetrat Pakazad, Henrik Ohlsson, Graham Neubig - [[pdf]](https://arxiv.org/pdf/2601.21372)</summary>

**Abstract:** In this paper, we present NEMO, a system that translates Natural-language descriptions of decision problems into formal Executable Mathematical Optimization implementations, operating collaboratively with users or autonomously. Existing approaches typically rely on specialized large language models (LLMs) or bespoke, task-specific agents. Such methods are often brittle, complex and frequently generating syntactically invalid or non-executable code.
NEMO instead centers on remote interaction with autonomous coding agents (ACAs), treated as a first-class abstraction analogous to API-based interaction with LLMs. This design enables the construction of higher-level systems around ACAs that structure, consolidate, and iteratively refine task specifications. Because ACAs execute within sandboxed environments, code produced by NEMO is executable by construction, allowing automated validation and repair.
Building on this, we introduce novel coordination patterns with and across ACAs, including asymmetric validation loops between independently generated optimizer and simulator implementations (serving as a high-level validation mechanism), external memory for experience reuse, and robustness enhancements via minimum Bayes risk (MBR) decoding and self-consistency. We evaluate NEMO on nine established optimization benchmarks. As depicted in Figure 1, it achieves state-of-the-art performance on the majority of tasks, with substantial margins on several datasets, demonstrating the power of execution-aware agentic architectures for automated optimization modeling.

**arXiv ID:** 2601.21372
</details>

<details>
<summary><strong>Meta Context Engineering via Agentic Skill Evolution</strong> - Haoran Ye, Xuning He, Vincent Arak, Haonan Dong, Guojie Song - [[pdf]](https://arxiv.org/pdf/2601.21557)</summary>

**Abstract:** The operational efficacy of large language models relies heavily on their inference-time context. This has established Context Engineering (CE) as a formal discipline for optimizing these inputs. Current CE methods rely on manually crafted harnesses, such as rigid generation-reflection workflows and predefined context schemas. They impose structural biases and restrict context optimization to a narrow, intuition-bound design space. To address this, we introduce Meta Context Engineering (MCE), a bi-level framework that supersedes static CE heuristics by co-evolving CE skills and context artifacts. In MCE iterations, a meta-level agent refines engineering skills via agentic crossover, a deliberative search over the history of skills, their executions, and evaluations. A base-level agent executes these skills, learns from training rollouts, and optimizes context as flexible files and code. We evaluate MCE across five disparate domains under offline and online settings. MCE demonstrates consistent performance gains, achieving 5.6--53.8% relative improvement over state-of-the-art agentic CE methods (mean of 16.9%), while maintaining superior context adaptability, transferability, and efficiency in both context usage and training.

**arXiv ID:** 2601.21557
</details>

<details>
<summary><strong>How do Visual Attributes Influence Web Agents? A Comprehensive Evaluation of User Interface Design Factors</strong> - Kuai Yu, Naicheng Yu, Han Wang, Rui Yang, Huan Zhang - [[pdf]](https://arxiv.org/pdf/2601.21961)</summary>

**Abstract:** Web agents have demonstrated strong performance on a wide range of web-based tasks. However, existing research on the effect of environmental variation has mostly focused on robustness to adversarial attacks, with less attention to agents' preferences in benign scenarios. Although early studies have examined how textual attributes influence agent behavior, a systematic understanding of how visual attributes shape agent decision-making remains limited. To address this, we introduce VAF, a controlled evaluation pipeline for quantifying how webpage Visual Attribute Factors influence web-agent decision-making. Specifically, VAF consists of three stages: (i) variant generation, which ensures the variants share identical semantics as the original item while only differ in visual attributes; (ii) browsing interaction, where agents navigate the page via scrolling and clicking the interested item, mirroring how human users browse online; (iii) validating through both click action and reasoning from agents, which we use the Target Click Rate and Target Mention Rate to jointly evaluate the effect of visual attributes. By quantitatively measuring the decision-making difference between the original and variant, we identify which visual attributes influence agents' behavior most. Extensive experiments, across 8 variant families (48 variants total), 5 real-world websites (including shopping, travel, and news browsing), and 4 representative web agents, show that background color contrast, item size, position, and card clarity have a strong influence on agents' actions, whereas font styling, text color, and item image clarity exhibit minor effects.

**arXiv ID:** 2601.21961
</details>

<details>
<summary><strong>Optimizing Agentic Workflows using Meta-tools</strong> - Sami Abuzakuk, Anne-Marie Kermarrec, Rishi Sharma, Rasmus Moorits Veski, Martijn de Vos - [[pdf]](https://arxiv.org/pdf/2601.22037)</summary>

**Abstract:** Agentic AI enables LLM to dynamically reason, plan, and interact with tools to solve complex tasks. However, agentic workflows often require many iterative reasoning steps and tool invocations, leading to significant operational expense, end-to-end latency and failures due to hallucinations. This work introduces Agent Workflow Optimization (AWO), a framework that identifies and optimizes redundant tool execution patterns to improve the efficiency and robustness of agentic workflows. AWO analyzes existing workflow traces to discover recurring sequences of tool calls and transforms them into meta-tools, which are deterministic, composite tools that bundle multiple agent actions into a single invocation. Meta-tools bypass unnecessary intermediate LLM reasoning steps and reduce operational cost while also shortening execution paths, leading to fewer failures. Experiments on two agentic AI benchmarks show that AWO reduces the number of LLM calls up to 11.9% while also increasing the task success rate by up to 4.2 percent points.

**arXiv ID:** 2601.22037
</details>

<details>
<summary><strong>DevOps-Gym: Benchmarking AI Agents in Software DevOps Cycle</strong> - Yuheng Tang, Kaijie Zhu, Bonan Ruan, Chuqi Zhang, Michael Yang, Hongwei Li, Suyue Guo, Tianneng Shi, Zekun Li, Christopher Kruegel, Giovanni Vigna, Dawn Song, William Yang Wang, Lun Wang, Yangruibo Ding, Zhenkai Liang, Wenbo Guo - [[pdf]](https://arxiv.org/pdf/2601.20882)</summary>

**Abstract:** Even though demonstrating extraordinary capabilities in code generation and software issue resolving, AI agents' capabilities in the full software DevOps cycle are still unknown. Different from pure code generation, handling the DevOps cycle in real-world software, including developing, deploying, and managing, requires analyzing large-scale projects, understanding dynamic program behaviors, leveraging domain-specific tools, and making sequential decisions. However, existing benchmarks focus on isolated problems and lack environments and tool interfaces for DevOps. We introduce DevOps-Gym, the first end-to-end benchmark for evaluating AI agents across core DevOps workflows: build and configuration, monitoring, issue resolving, and test generation. DevOps-Gym includes 700+ real-world tasks collected from 30+ projects in Java and Go. We develop a semi-automated data collection mechanism with rigorous and non-trivial expert efforts in ensuring the task coverage and quality. Our evaluation of state-of-the-art models and agents reveals fundamental limitations: they struggle with issue resolving and test generation in Java and Go, and remain unable to handle new tasks such as monitoring and build and configuration. These results highlight the need for essential research in automating the full DevOps cycle with AI agents.

**arXiv ID:** 2601.20882
</details>

<details>
<summary><strong>Thinker: A vision-language foundation model for embodied intelligence</strong> - Baiyu Pan, Daqin Luo, Junpeng Yang, Jiyuan Wang, Yixuan Zhang, Hailin Shi, Jichao Jiao - [[pdf]](https://arxiv.org/pdf/2601.21199)</summary>

**Abstract:** When large vision-language models are applied to the field of robotics, they encounter problems that are simple for humans yet error-prone for models. Such issues include confusion between third-person and first-person perspectives and a tendency to overlook information in video endings during temporal reasoning. To address these challenges, we propose Thinker, a large vision-language foundation model designed for embodied intelligence. We tackle the aforementioned issues from two perspectives. Firstly, we construct a large-scale dataset tailored for robotic perception and reasoning, encompassing ego-view videos, visual grounding, spatial understanding, and chain-of-thought data. Secondly, we introduce a simple yet effective approach that substantially enhances the model's capacity for video comprehension by jointly incorporating key frames and full video sequences as inputs. Our model achieves state-of-the-art results on two of the most commonly used benchmark datasets in the field of task planning.

**arXiv ID:** 2601.21199
</details>

<details>
<summary><strong>DeepSearchQA: Bridging the Comprehensiveness Gap for Deep Research Agents</strong> - Nikita Gupta, Riju Chatterjee, Lukas Haas, Connie Tao, Andrew Wang, Chang Liu, Hidekazu Oiwa, Elena Gribovskaya, Jan Ackermann, John Blitzer, Sasha Goldshtein, Dipanjan Das - [[pdf]](https://arxiv.org/pdf/2601.20975)</summary>

**Abstract:** We introduce DeepSearchQA, a 900-prompt benchmark for evaluating agents on difficult multi-step information-seeking tasks across 17 different fields. Unlike traditional benchmarks that target single answer retrieval or broad-spectrum factuality, DeepSearchQA features a dataset of challenging, handcrafted tasks designed to evaluate an agent's ability to execute complex search plans to generate exhaustive answer lists. This shift in design explicitly tests three critical, yet under-evaluated capabilities: 1) systematic collation of fragmented information from disparate sources, 2) de-duplication and entity resolution to ensure precision, and 3) the ability to reason about stopping criteria within an open-ended search space. Each task is structured as a causal chain, where discovering information for one step is dependent on the successful completion of the previous one, stressing long-horizon planning and context retention. All tasks are grounded in the open web with objectively verifiable answer sets. Our comprehensive evaluation of state-of-the-art agent architectures reveals significant performance limitations: even the most advanced models struggle to balance high recall with precision. We observe distinct failure modes ranging from premature stopping (under-retrieval) to hedging behaviors, where agents cast an overly wide net of low-confidence answers to artificially boost recall. These findings highlight critical headroom in current agent designs and position DeepSearchQA as an essential diagnostic tool for driving future research toward more robust, deep-research capabilities.

**arXiv ID:** 2601.20975
</details>

<details>
<summary><strong>Embodied Task Planning via Graph-Informed Action Generation with Large Lanaguage Model</strong> - Xiang Li, Ning Yan, Masood Mortazavi - [[pdf]](https://arxiv.org/pdf/2601.21841)</summary>

**Abstract:** While Large Language Models (LLMs) have demonstrated strong zero-shot reasoning capabilities, their deployment as embodied agents still faces fundamental challenges in long-horizon planning. Unlike open-ended text generation, embodied agents must decompose high-level intent into actionable sub-goals while strictly adhering to the logic of a dynamic, observed environment. Standard LLM planners frequently fail to maintain strategy coherence over extended horizons due to context window limitation or hallucinate transitions that violate constraints. We propose GiG, a novel planning framework that structures embodied agents' memory using a Graph-in-Graph architecture. Our approach employs a Graph Neural Network (GNN) to encode environmental states into embeddings, organizing these embeddings into action-connected execution trace graphs within an experience memory bank. By clustering these graph embeddings, the framework enables retrieval of structure-aware priors, allowing agents to ground current decisions in relevant past structural patterns. Furthermore, we introduce a novel bounded lookahead module that leverages symbolic transition logic to enhance the agents' planning capabilities through the grounded action projection. We evaluate our framework on three embodied planning benchmarks-Robotouille Synchronous, Robotouille Asynchronous, and ALFWorld. Our method outperforms state-of-the-art baselines, achieving Pass@1 performance gains of up to 22% on Robotouille Synchronous, 37% on Asynchronous, and 15% on ALFWorld with comparable or lower computational cost.

**arXiv ID:** 2601.21841
</details>

<details>
<summary><strong>MobileBench-OL: A Comprehensive Chinese Benchmark for Evaluating Mobile GUI Agents in Real-World Environment</strong> - Qinzhuo Wu, Zhizhuo Yang, Hanhao Li, Pengzhi Gao, Wei Liu, Jian Luan - [[pdf]](https://arxiv.org/pdf/2601.20335)</summary>

**Abstract:** Recent advances in mobile Graphical User Interface (GUI) agents highlight the growing need for comprehensive evaluation benchmarks. While new online benchmarks offer more realistic testing than offline ones, they tend to focus on the agents' task instruction-following ability while neglecting their reasoning and exploration ability. Moreover, these benchmarks do not consider the random noise in real-world mobile environments. This leads to a gap between benchmarks and real-world environments. To addressing these limitations, we propose MobileBench-OL, an online benchmark with 1080 tasks from 80 Chinese apps. It measures task execution, complex reasoning, and noise robustness of agents by including 5 subsets, which set multiple evaluation dimensions. We also provide an auto-eval framework with a reset mechanism, enabling stable and repeatable real-world benchmarking. Evaluating 12 leading GUI agents on MobileBench-OL shows significant room for improvement to meet real-world requirements. Human evaluation further confirms that MobileBench-OL can reliably measure the performance of leading GUI agents in real environments. Our data and code will be released upon acceptance.

**arXiv ID:** 2601.20335
</details>

<details>
<summary><strong>SafeSearch: Automated Red-Teaming of LLM-Based Search Agents</strong> - Jianshuo Dong, Sheng Guo, Hao Wang, Xun Chen, Zhuotao Liu, Tianwei Zhang, Ke Xu, Minlie Huang, Han Qiu - [[pdf]](https://arxiv.org/pdf/2509.23694)</summary>

**Abstract:** Search agents connect LLMs to the Internet, enabling them to access broader and more up-to-date information. However, this also introduces a new threat surface: unreliable search results can mislead agents into producing unsafe outputs. Real-world incidents and our two in-the-wild observations show that such failures can occur in practice. To study this threat systematically, we propose SafeSearch, an automated red-teaming framework that is scalable, cost-efficient, and lightweight, enabling harmless safety evaluation of search agents. Using this, we generate 300 test cases spanning five risk categories (e.g., misinformation and prompt injection) and evaluate three search agent scaffolds across 17 representative LLMs. Our results reveal substantial vulnerabilities in LLM-based search agents, with the highest ASR reaching 90.5% for GPT-4.1-mini in a search-workflow setting. Moreover, we find that common defenses, such as reminder prompting, offer limited protection. Overall, SafeSearch provides a practical way to measure and improve the safety of LLM-based search agents. Our codebase and test cases are publicly available: this https URL.

**arXiv ID:** 2509.23694
</details>

<details>
<summary><strong>StepShield: When, Not Whether to Intervene on Rogue Agents</strong> - Gloria Felicia, Michael Eniolade, Jinfeng He, Zitha Sasindran, Hemant Kumar, Milan Hussain Angati, Sandeep Bandarupalli - [[pdf]](https://arxiv.org/pdf/2601.22136)</summary>

**Abstract:** Existing agent safety benchmarks report binary accuracy, conflating early intervention with post-mortem analysis. A detector that flags a violation at step 8 enables intervention; one that reports it at step 48 provides only forensic value. This distinction is critical, yet current benchmarks cannot measure it. We introduce StepShield, the first benchmark to evaluate when violations are detected, not just whether. StepShield contains 9,213 code agent trajectories, including 1,278 meticulously annotated training pairs and a 7,935-trajectory test set with a realistic 8.1% rogue rate. Rogue behaviors are grounded in real-world security incidents across six categories. We propose three novel temporal metrics: Early Intervention Rate (EIR), Intervention Gap, and Tokens Saved. Surprisingly, our evaluation reveals that an LLM-based judge achieves 59% EIR while a static analyzer achieves only 26%, a 2.3x performance gap that is entirely invisible to standard accuracy metrics. We further show that early detection has direct economic benefits: our cascaded HybridGuard detector reduces monitoring costs by 75% and projects to $108M in cumulative savings over five years at enterprise scale. By shifting the focus of evaluation from whether to when, StepShield provides a new foundation for building safer and more economically viable AI agents. The code and data are released under an Apache 2.0 license.

**arXiv ID:** 2601.22136
</details>

<details>
<summary><strong>IDE-Bench: Evaluating Large Language Models as IDE Agents on Real-World Software Engineering Tasks</strong> - Spencer Mateega, Jeff Yang, Tiana Costello, Shaurya Jadhav, Nicole Tian, Agustin Garcinuño - [[pdf]](https://arxiv.org/pdf/2601.20886)</summary>

**Abstract:** IDE-Bench is a comprehensive framework for evaluating AI IDE agents on real-world software engineering tasks through an IDE-native tool interface. We present a Dockerized test harness that goes beyond raw terminal execution, granting models a structured tool ecosystem that represents AI-native IDEs like Cursor and Windsurf. By providing high-level abstractions for codebase search, structured file editing, and tools for testing full-stack applications, IDE-Bench evaluates an agent's ability to act as a true engineering collaborator. For evaluation and to prevent training data contamination, we created 80 tasks across eight never-published repositories spanning C/C++, Java, and MERN stacks, representing modern tech stack production scenarios, including feature implementation, bug fixing, refactoring, and performance optimization that mirror daily developer workflows in private codebases. Our benchmark is the first to systematically correlate agent-reported intent with successful project-level modifications in a multi-language, full-stack environment on completely uncontaminated code.

**arXiv ID:** 2601.20886
</details>

<details>
<summary><strong>DSCD-Nav: Dual-Stance Cooperative Debate for Object Navigation</strong> - Weitao An, Qi Liu, Chenghao Xu, Jiayi Chai, Xu Yang, Kun Wei, Cheng Deng - [[pdf]](https://arxiv.org/pdf/2601.21409)</summary>

**Abstract:** Adaptive navigation in unfamiliar indoor environments is crucial for household service robots. Despite advances in zero-shot perception and reasoning from vision-language models, existing navigation systems still rely on single-pass scoring at the decision layer, leading to overconfident long-horizon errors and redundant exploration. To tackle these problems, we propose Dual-Stance Cooperative Debate Navigation (DSCD-Nav), a decision mechanism that replaces one-shot scoring with stance-based cross-checking and evidence-aware arbitration to improve action reliability under partial observability. Specifically, given the same observation and candidate action set, we explicitly construct two stances by conditioning the evaluation on diverse and complementary objectives: a Task-Scene Understanding (TSU) stance that prioritizes goal progress from scene-layout cues, and a Safety-Information Balancing (SIB) stance that emphasizes risk and information value. The stances conduct a cooperative debate and make policy by cross-checking their top candidates with cue-grounded arguments. Then, a Navigation Consensus Arbitration (NCA) agent is employed to consolidate both sides' reasons and evidence, optionally triggering lightweight micro-probing to verify uncertain choices, preserving NCA's primary intent while disambiguating. Experiments on HM3Dv1, HM3Dv2, and MP3D demonstrate consistent improvements in success and path efficiency while reducing exploration redundancy.

**arXiv ID:** 2601.21409
</details>

<details>
<summary><strong>4D-CAAL: 4D Radar-Camera Calibration and Auto-Labeling for Autonomous Driving</strong> - Shanliang Yao, Zhuoxiao Li, Runwei Guan, Kebin Cao, Meng Xia, Fuping Hu, Sen Xu, Yong Yue, Xiaohui Zhu, Weiping Ding, Ryan Wen Liu - [[pdf]](https://arxiv.org/pdf/2601.21454)</summary>

**Abstract:** 4D radar has emerged as a critical sensor for autonomous driving, primarily due to its enhanced capabilities in elevation measurement and higher resolution compared to traditional 3D radar. Effective integration of 4D radar with cameras requires accurate extrinsic calibration, and the development of radar-based perception algorithms demands large-scale annotated datasets. However, existing calibration methods often employ separate targets optimized for either visual or radar modalities, complicating correspondence establishment. Furthermore, manually labeling sparse radar data is labor-intensive and unreliable. To address these challenges, we propose 4D-CAAL, a unified framework for 4D radar-camera calibration and auto-labeling. Our approach introduces a novel dual-purpose calibration target design, integrating a checkerboard pattern on the front surface for camera detection and a corner reflector at the center of the back surface for radar detection. We develop a robust correspondence matching algorithm that aligns the checkerboard center with the strongest radar reflection point, enabling accurate extrinsic calibration. Subsequently, we present an auto-labeling pipeline that leverages the calibrated sensor relationship to transfer annotations from camera-based segmentations to radar point clouds through geometric projection and multi-feature optimization. Extensive experiments demonstrate that our method achieves high calibration accuracy while significantly reducing manual annotation effort, thereby accelerating the development of robust multi-modal perception systems for autonomous driving.

**arXiv ID:** 2601.21454
</details>

<details>
<summary><strong>Don't double it: Efficient Agent Prediction in Occlusions</strong> - Anna Rothenhäusler, Markus Mazzola, Andreas Look, Raghu Rajan, Joschka Bödecker - [[pdf]](https://arxiv.org/pdf/2601.21504)</summary>

**Abstract:** Occluded traffic agents pose a significant challenge for autonomous vehicles, as hidden pedestrians or vehicles can appear unexpectedly, yet this problem remains understudied. Existing learning-based methods, while capable of inferring the presence of hidden agents, often produce redundant occupancy predictions where a single agent is identified multiple times. This issue complicates downstream planning and increases computational load. To address this, we introduce MatchInformer, a novel transformer-based approach that builds on the state-of-the-art SceneInformer architecture. Our method improves upon prior work by integrating Hungarian Matching, a state-of-the-art object matching algorithm from object detection, into the training process to enforce a one-to-one correspondence between predictions and ground truth, thereby reducing redundancy. We further refine trajectory forecasts by decoupling an agent's heading from its motion, a strategy that improves the accuracy and interpretability of predicted paths. To better handle class imbalances, we propose using the Matthews Correlation Coefficient (MCC) to evaluate occupancy predictions. By considering all entries in the confusion matrix, MCC provides a robust measure even in sparse or imbalanced scenarios. Experiments on the Waymo Open Motion Dataset demonstrate that our approach improves reasoning about occluded regions and produces more accurate trajectory forecasts than prior methods.

**arXiv ID:** 2601.21504
</details>

<details>
<summary><strong>LLM-Driven Scenario-Aware Planning for Autonomous Driving</strong> - He Li, Zhaowei Chen, Rui Gao, Guoliang Li, Qi Hao, Shuai Wang, Chengzhong Xu - [[pdf]](https://arxiv.org/pdf/2601.21876)</summary>

**Abstract:** Hybrid planner switching framework (HPSF) for autonomous driving needs to reconcile high-speed driving efficiency with safe maneuvering in dense traffic. Existing HPSF methods often fail to make reliable mode transitions or sustain efficient driving in congested environments, owing to heuristic scene recognition and low-frequency control updates. To address the limitation, this paper proposes LAP, a large language model (LLM) driven, adaptive planning method, which switches between high-speed driving in low-complexity scenes and precise driving in high-complexity scenes, enabling high qualities of trajectory generation through confined gaps. This is achieved by leveraging LLM for scene understanding and integrating its inference into the joint optimization of mode configuration and motion planning. The joint optimization is solved using tree-search model predictive control and alternating minimization. We implement LAP by Python in Robot Operating System (ROS). High-fidelity simulation results show that the proposed LAP outperforms other benchmarks in terms of both driving time and success rate.

**arXiv ID:** 2601.21876
</details>

<details>
<summary><strong>AsterNav: Autonomous Aerial Robot Navigation In Darkness Using Passive Computation</strong> - Deepak Singh, Shreyas Khobragade, Nitin J. Sanket - [[pdf]](https://arxiv.org/pdf/2601.17550)</summary>

**Abstract:** Autonomous aerial navigation in absolute darkness is crucial for post-disaster search and rescue operations, which often occur from disaster-zone power outages. Yet, due to resource constraints, tiny aerial robots, perfectly suited for these operations, are unable to navigate in the darkness to find survivors safely. In this paper, we present an autonomous aerial robot for navigation in the dark by combining an Infra-Red (IR) monocular camera with a large-aperture coded lens and structured light without external infrastructure like GPS or motion-capture. Our approach obtains depth-dependent defocus cues (each structured light point appears as a pattern that is depth dependent), which acts as a strong prior for our AsterNet deep depth estimation model. The model is trained in simulation by generating data using a simple optical model and transfers directly to the real world without any fine-tuning or retraining. AsterNet runs onboard the robot at 20 Hz on an NVIDIA Jetson Orin$^\text{TM}$ Nano. Furthermore, our network is robust to changes in the structured light pattern and relative placement of the pattern emitter and IR camera, leading to simplified and cost-effective construction. We successfully evaluate and demonstrate our proposed depth navigation approach AsterNav using depth from AsterNet in many real-world experiments using only onboard sensing and computation, including dark matte obstacles and thin ropes (diameter 6.25mm), achieving an overall success rate of 95.5% with unknown object shapes, locations and materials. To the best of our knowledge, this is the first work on monocular, structured-light-based quadrotor navigation in absolute darkness.

**arXiv ID:** 2601.17550
</details>

<details>
<summary><strong>Listen, Look, Drive: Coupling Audio Instructions for User-aware VLA-based Autonomous Driving</strong> - Ziang Guo, Feng Yang, Xuefeng Zhang, Jiaqi Guo, Kun Zhao, Yixiao Zhou, Peng Lu, Sifa Zheng, Zufeng Zhang - [[pdf]](https://arxiv.org/pdf/2601.12142)</summary>

**Abstract:** Vision Language Action (VLA) models promise an open-vocabulary interface that can translate perceptual ambiguity into semantically grounded driving decisions, yet they still treat language as a static prior fixed at inference time. As a result, the model must infer continuously shifting objectives from pixels alone, yielding delayed or overly conservative maneuvers. We argue that effective VLAs for autonomous driving need an online channel in which users can influence driving with specific intentions. To this end, we present EchoVLA, a user-aware VLA that couples camera streams with in situ audio instructions. We augment the nuScenes dataset with temporally aligned, intent-specific speech commands generated by converting ego-motion descriptions into synthetic audios. Further, we compose emotional speech-trajectory pairs into a multimodal Chain-of-Thought (CoT) for fine-tuning a Multimodal Large Model (MLM) based on Qwen2.5-Omni. Specifically, we synthesize the audio-augmented dataset with different emotion types paired with corresponding driving behaviors, leveraging the emotional cues embedded in tone, pitch, and speech tempo to reflect varying user states, such as urgent or hesitant intentions, thus enabling our EchoVLA to interpret not only the semantic content but also the emotional context of audio commands for more nuanced and emotionally adaptive driving behavior. In open-loop benchmarks, our approach reduces the average L2 error by $59.4\%$ and the collision rate by $74.4\%$ compared to the baseline of vision-only perception. More experiments on nuScenes dataset validate that EchoVLA not only steers the trajectory through audio instructions, but also modulates driving behavior in response to the emotions detected in the user's speech.

**arXiv ID:** 2601.12142
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>EmboCoach-Bench: Benchmarking AI Agents on Developing Embodied Robots</strong> - Zixing Lei, Genjia Liu, Yuanshuo Zhang, Qipeng Liu, Chuan Wen, Shanghang Zhang, Wenzhao Lian, Siheng Chen - [[pdf]](https://arxiv.org/pdf/2601.21570)</summary>

**Abstract:** The field of Embodied AI is witnessing a rapid evolution toward general-purpose robotic systems, fueled by high-fidelity simulation and large-scale data collection. However, this scaling capability remains severely bottlenecked by a reliance on labor-intensive manual oversight from intricate reward shaping to hyperparameter tuning across heterogeneous backends. Inspired by LLMs' success in software automation and science discovery, we introduce \textsc{EmboCoach-Bench}, a benchmark evaluating the capacity of LLM agents to autonomously engineer embodied policies. Spanning 32 expert-curated RL and IL tasks, our framework posits executable code as the universal interface. We move beyond static generation to assess a dynamic closed-loop workflow, where agents leverage environment feedback to iteratively draft, debug, and optimize solutions, spanning improvements from physics-informed reward design to policy architectures such as diffusion policies. Extensive evaluations yield three critical insights: (1) autonomous agents can qualitatively surpass human-engineered baselines by 26.5\% in average success rate; (2) agentic workflow with environment feedback effectively strengthens policy development and substantially narrows the performance gap between open-source and proprietary models; and (3) agents exhibit self-correction capabilities for pathological engineering cases, successfully resurrecting task performance from near-total failures through iterative simulation-in-the-loop debugging. Ultimately, this work establishes a foundation for self-evolving embodied intelligence, accelerating the paradigm shift from labor-intensive manual tuning to scalable, autonomous engineering in embodied AI field.

**arXiv ID:** 2601.21570
</details>

<details>
<summary><strong>CORE:Toward Ubiquitous 6G Intelligence Through Collaborative Orchestration of Large Language Model Agents Over Hierarchical Edge</strong> - Zitong Yu, Boquan Sun, Yang Li, Zheyan Qu, Xing Zhang - [[pdf]](https://arxiv.org/pdf/2601.21822)</summary>

**Abstract:** Rapid advancements in sixth-generation (6G) networks and large language models (LLMs) have paved the way for ubiquitous intelligence, wherein seamless connectivity and distributed artificial intelligence (AI) have revolutionized various aspects of our this http URL, realizing this vision faces significant challenges owing to the fragmented and heterogeneous computing resources across hierarchical networks, which are insufficient for individual LLM agents to perform complex reasoning this http URL address this issue, we propose Collaborative Orchestration Role at Edge (CORE), an innovative framework that employs a collaborative learning system in which multiple LLMs, each assigned a distinct functional role, are distributed across mobile devices and tiered edge servers. The system integrates three optimization modules, encompassing real-time perception,dynamic role orchestration, and pipeline-parallel execution, to facilitate efficient and rapid collaboration among distributed agents. Furthermore, we introduce a novel role affinity scheduling algorithm for dynamically orchestrating LLM role assignments across the hierarchical edge infrastructure, intelligently matching computational demands with available dispersed this http URL, comprehensive case studies and performance evaluations across various 6G application scenarios demonstrated the efficacy of CORE, revealing significant enhancements in the system efficiency and task completion rates. Building on these promising outcomes, we further validated the practical applicability of CORE by deploying it on a real-world edge-computing platform,that exhibits robust performance in operational environments.

**arXiv ID:** 2601.21822
</details>

<details>
<summary><strong>CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty</strong> - Johannes Kirmayr, Lukas Stappen, Elisabeth André - [[pdf]](https://arxiv.org/pdf/2601.22027)</summary>

**Abstract:** Existing benchmarks for Large Language Model (LLM) agents focus on task completion under idealistic settings but overlook reliability in real-world, user-facing applications. In domains, such as in-car voice assistants, users often issue incomplete or ambiguous requests, creating intrinsic uncertainty that agents must manage through dialogue, tool use, and policy adherence. We introduce CAR-bench, a benchmark for evaluating consistency, uncertainty handling, and capability awareness in multi-turn, tool-using LLM agents in an in-car assistant domain. The environment features an LLM-simulated user, domain policies, and 58 interconnected tools spanning navigation, productivity, charging, and vehicle control. Beyond standard task completion, CAR-bench introduces Hallucination tasks that test agents' limit-awareness under missing tools or information, and Disambiguation tasks that require resolving uncertainty through clarification or internal information gathering. Baseline results reveal large gaps between occasional and consistent success on all task types. Even frontier reasoning LLMs achieve less than 50% consistent pass rate on Disambiguation tasks due to premature actions, and frequently violate policies or fabricate information to satisfy user requests in Hallucination tasks, underscoring the need for more reliable and self-aware LLM agents in real-world settings.

**arXiv ID:** 2601.22027
</details>

<details>
<summary><strong>ASTRA: Automated Synthesis of agentic Trajectories and Reinforcement Arenas</strong> - Xiaoyu Tian, Haotian Wang, Shuaiting Chen, Hao Zhou, Kaichi Yu, Yudian Zhang, Jade Ouyang, Junxi Yin, Jiong Chen, Baoyan Guo, Lei Zhang, Junjie Tao, Yuansheng Song, Ming Cui, Chengwei Liu - [[pdf]](https://arxiv.org/pdf/2601.21558)</summary>

**Abstract:** Large language models (LLMs) are increasingly used as tool-augmented agents for multi-step decision making, yet training robust tool-using agents remains challenging. Existing methods still require manual intervention, depend on non-verifiable simulated environments, rely exclusively on either supervised fine-tuning (SFT) or reinforcement learning (RL), and struggle with stable long-horizon, multi-turn learning. To address these challenges, we introduce ASTRA, a fully automated end-to-end framework for training tool-augmented language model agents via scalable data synthesis and verifiable reinforcement learning. ASTRA integrates two complementary components. First, a pipeline that leverages the static topology of tool-call graphs synthesizes diverse, structurally grounded trajectories, instilling broad and transferable tool-use competence. Second, an environment synthesis framework that captures the rich, compositional topology of human semantic reasoning converts decomposed question-answer traces into independent, code-executable, and rule-verifiable environments, enabling deterministic multi-turn RL. Based on this method, we develop a unified training methodology that integrates SFT with online RL using trajectory-level rewards to balance task completion and interaction efficiency. Experiments on multiple agentic tool-use benchmarks demonstrate that ASTRA-trained models achieve state-of-the-art performance at comparable scales, approaching closed-source systems while preserving core reasoning ability. We release the full pipelines, environments, and trained models at this https URL.

**arXiv ID:** 2601.21558
</details>

<details>
<summary><strong>Can David Beat Goliath? On Multi-Hop Reasoning with Resource-Constrained Agents</strong> - Hojae Han, Heeyun Jung, Jongyoon Kim, Seung-won Hwang - [[pdf]](https://arxiv.org/pdf/2601.21699)</summary>

**Abstract:** While reinforcement learning (RL) has empowered multi-turn reasoning agents with retrieval and tools, existing successes largely depend on extensive on-policy rollouts in high-cost, high-accuracy regimes. Under realistic resource constraints that cannot support large models or dense explorations, however, small language model agents fall into a low-cost, low-accuracy regime, where limited rollout budgets lead to sparse exploration, sparse credit assignment, and unstable training. In this work, we challenge this trade-off and show that small language models can achieve strong multi-hop reasoning under resource constraints. We introduce DAVID-GRPO, a budget-efficient RL framework that (i) stabilizes early learning with minimal supervision, (ii) assigns retrieval credit based on evidence recall, and (iii) improves exploration by resampling truncated near-miss trajectories. Evaluated on agents up to 1.5B parameters trained on only four RTX 3090 GPUs, DAVID-GRPO consistently outperforms prior RL methods designed for large-scale settings on six multi-hop QA benchmarks. These results show that with the right inductive biases, small agents can achieve low training cost with high accuracy.

**arXiv ID:** 2601.21699
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (21 papers)</h2></summary>

<details>
<summary><strong>DataCross: A Unified Benchmark and Agent Framework for Cross-Modal Heterogeneous Data Analysis</strong> - Ruyi Qi, Zhou Liu, Wentao Zhang - [[pdf]](https://arxiv.org/pdf/2601.21403)</summary>

**Abstract:** In real-world data science and enterprise decision-making, critical information is often fragmented across directly queryable structured sources (e.g., SQL, CSV) and "zombie data" locked in unstructured visual documents (e.g., scanned reports, invoice images). Existing data analytics agents are predominantly limited to processing structured data, failing to activate and correlate this high-value visual information, thus creating a significant gap with industrial needs. To bridge this gap, we introduce DataCross, a novel benchmark and collaborative agent framework for unified, insight-driven analysis across heterogeneous data modalities. DataCrossBench comprises 200 end-to-end analysis tasks across finance, healthcare, and other domains. It is constructed via a human-in-the-loop reverse-synthesis pipeline, ensuring realistic complexity, cross-source dependency, and verifiable ground truth. The benchmark categorizes tasks into three difficulty tiers to evaluate agents' capabilities in visual table extraction, cross-modal alignment, and multi-step joint reasoning. We also propose the DataCrossAgent framework, inspired by the "divide-and-conquer" workflow of human analysts. It employs specialized sub-agents, each an expert on a specific data source, which are coordinated via a structured workflow of Intra-source Deep Exploration, Key Source Identification, and Contextual Cross-pollination. A novel reReAct mechanism enables robust code generation and debugging for factual verification. Experimental results show that DataCrossAgent achieves a 29.7% improvement in factuality over GPT-4o and exhibits superior robustness on high-difficulty tasks, effectively activating fragmented "zombie data" for insightful, cross-modal analysis.

**arXiv ID:** 2601.21403
</details>

<details>
<summary><strong>ScaleSim: Serving Large-Scale Multi-Agent Simulation with Invocation Distance-Based Memory Management</strong> - Zaifeng Pan, Yipeng Shen, Zhengding Hu, Zhuang Wang, Aninda Manocha, Zheng Wang, Zhongkai Yu, Yue Guan, Yufei Ding - [[pdf]](https://arxiv.org/pdf/2601.21473)</summary>

**Abstract:** LLM-based multi-agent simulations are increasingly adopted across application domains, but remain difficult to scale due to GPU memory pressure. Each agent maintains private GPU-resident states, including models, prefix caches, and adapters, which quickly exhaust device memory as the agent count grows. We identify two key properties of these workloads: sparse agent activation and an estimable agent invocation order. Based on an analysis of representative workload classes, we introduce invocation distance, a unified abstraction that estimates the relative order in which agents will issue future LLM requests. Leveraging this abstraction, we present ScaleSim, a memory-efficient LLM serving system for large-scale multi-agent simulations. ScaleSim enables proactive prefetching and priority-based eviction, supports diverse agent-specific memory through a modular interface, and achieves up to 1.74x speedup over SGLang on simulation benchmarks.

**arXiv ID:** 2601.21473
</details>

<details>
<summary><strong>ShardMemo: Masked MoE Routing for Sharded Agentic LLM Memory</strong> - Yang Zhao, Chengxiao Dai, Yue Xiu, Mengying Kou, Yuliang Zheng, Dusit Niyato - [[pdf]](https://arxiv.org/pdf/2601.21545)</summary>

**Abstract:** Agentic large language model (LLM) systems rely on external memory for long-horizon state and concurrent multi-agent execution, but centralized indexes and heuristic partitions become bottlenecks as memory volume and parallel access grow. We present ShardMemo, a budgeted tiered memory service with Tier A per-agent working state, Tier B sharded evidence with shard-local approximate nearest neighbor (ANN) indexes, and Tier C, a versioned skill library. Tier B enforces scope-before-routing: structured eligibility constraints mask ineligible shards before routing or ANN search. We cast shard probing as masked mixture-of-experts (MoE) routing over eligible shards, probing up to $B_{\mathrm{probe}}$ shards via Top-$B_{\mathrm{probe}}$ or adaptive Top-$P$, and use cost-aware gating over profile/observation/session shard families; the router is trained from evidence-to-shard supervision. On LoCoMo, ShardMemo improves over the strongest baseline (GAM) by +5.11 to +6.82 F1 across question categories. Under a fixed-budget routing setting ($B_{\mathrm{probe}}=3$), ShardMemo improves over cosine-to-prototype shard routing by +6.87 F1 while reducing retrieval work (VecScan 521->414, -20.5%) and p95 latency (95->76 ms). On long-context HotpotQA, ShardMemo achieves 63.41/61.88/57.95 F1 at 56K/224K/448K tokens. On ToolBench, Tier C reaches 0.97 Precision@3 and 1.94 StepRed (+10.2% and +7.2% over embedding-similarity retrieval).

**arXiv ID:** 2601.21545
</details>

<details>
<summary><strong>RecNet: Self-Evolving Preference Propagation for Agentic Recommender Systems</strong> - Bingqian Li, Xiaolei Wang, Junyi Li, Weitao Li, Long Zhang, Sheng Chen, Wayne Xin Zhao, Ji-Rong Wen - [[pdf]](https://arxiv.org/pdf/2601.21609)</summary>

**Abstract:** Agentic recommender systems leverage Large Language Models (LLMs) to model complex user behaviors and support personalized decision-making. However, existing methods primarily model preference changes based on explicit user-item interactions, which are sparse, noisy, and unable to reflect the real-time, mutual influences among users and items. To address these limitations, we propose RecNet, a self-evolving preference propagation framework that proactively propagates real-time preference updates across related users and items. RecNet consists of two complementary phases. In the forward phase, the centralized preference routing mechanism leverages router agents to integrate preference updates and dynamically propagate them to the most relevant agents. To ensure accurate and personalized integration of propagated preferences, we further introduce a personalized preference reception mechanism, which combines a message buffer for temporary caching and an optimizable, rule-based filter memory to guide selective preference assimilation based on past experience and interests. In the backward phase, the feedback-driven propagation optimization mechanism simulates a multi-agent reinforcement learning framework, using LLMs for credit assignment, gradient analysis, and module-level optimization, enabling continuous self-evolution of propagation strategies. Extensive experiments on various scenarios demonstrate the effectiveness of RecNet in modeling preference propagation for recommender systems.

**arXiv ID:** 2601.21609
</details>

<details>
<summary><strong>E-mem: Multi-agent based Episodic Context Reconstruction for LLM Agent Memory</strong> - Kaixiang Wang, Yidan Lin, Jiong Lou, Zhaojiacheng Zhou, Bunyod Suvonov, Jie Li - [[pdf]](https://arxiv.org/pdf/2601.21714)</summary>

**Abstract:** The evolution of Large Language Model (LLM) agents towards System~2 reasoning, characterized by deliberative, high-precision problem-solving, requires maintaining rigorous logical integrity over extended horizons. However, prevalent memory preprocessing paradigms suffer from destructive de-contextualization. By compressing complex sequential dependencies into pre-defined structures (e.g., embeddings or graphs), these methods sever the contextual integrity essential for deep reasoning. To address this, we propose E-mem, a framework shifting from Memory Preprocessing to Episodic Context Reconstruction. Inspired by biological engrams, E-mem employs a heterogeneous hierarchical architecture where multiple assistant agents maintain uncompressed memory contexts, while a central master agent orchestrates global planning. Unlike passive retrieval, our mechanism empowers assistants to locally reason within activated segments, extracting context-aware evidence before aggregation. Evaluations on the LoCoMo benchmark demonstrate that E-mem achieves over 54\% F1, surpassing the state-of-the-art GAM by 7.75\%, while reducing token cost by over 70\%.

**arXiv ID:** 2601.21714
</details>

<details>
<summary><strong>Epistemic Context Learning: Building Trust the Right Way in LLM-Based Multi-Agent Systems</strong> - Ruiwen Zhou, Maojia Song, Xiaobao Wu, Sitao Cheng, Xunjian Yin, Yuxi Xie, Zhuoqun Hao, Wenyue Hua, Liangming Pan, Soujanya Poria, Min-Yen Kan - [[pdf]](https://arxiv.org/pdf/2601.21742)</summary>

**Abstract:** Individual agents in multi-agent (MA) systems often lack robustness, tending to blindly conform to misleading peers. We show this weakness stems from both sycophancy and inadequate ability to evaluate peer reliability. To address this, we first formalize the learning problem of history-aware reference, introducing the historical interactions of peers as additional input, so that agents can estimate peer reliability and learn from trustworthy peers when uncertain. This shifts the task from evaluating peer reasoning quality to estimating peer reliability based on interaction history. We then develop Epistemic Context Learning (ECL): a reasoning framework that conditions predictions on explicitly-built peer profiles from history. We further optimize ECL by reinforcement learning using auxiliary rewards. Our experiments reveal that our ECL enables small models like Qwen 3-4B to outperform a history-agnostic baseline 8x its size (Qwen 3-30B) by accurately identifying reliable peers. ECL also boosts frontier models to near-perfect (100%) performance. We show that ECL generalizes well to various MA configurations and we find that trust is modeled well by LLMs, revealing a strong correlation in trust modeling accuracy and final answer quality.

**arXiv ID:** 2601.21742
</details>

<details>
<summary><strong>astra-langchain4j: Experiences Combining LLMs and Agent Programming</strong> - Rem Collier, Katharine Beaumont, Andrei Ciortea - [[pdf]](https://arxiv.org/pdf/2601.21879)</summary>

**Abstract:** Given the emergence of Generative AI over the last two years and the increasing focus on Agentic AI as a form of Multi-Agent System it is important to explore both how such technologies can impact the use of traditional Agent Toolkits and how the wealth of experience encapsulated in those toolkits can influence the design of the new agentic platforms. This paper presents an overview of our experience developing a prototype large language model (LLM) integration for the ASTRA programming language. It presents a brief overview of the toolkit, followed by three example implementations, concluding with a discussion of the experiences garnered through the examples.

**arXiv ID:** 2601.21879
</details>

<details>
<summary><strong>JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG</strong> - Yiqun Chen, Erhan Zhang, Tianyi Hu, Shijie Wang, Zixuan Yang, Meizhi Zhong, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao - [[pdf]](https://arxiv.org/pdf/2601.21916)</summary>

**Abstract:** The evolution of Retrieval-Augmented Generation (RAG) has shifted from static retrieval pipelines to dynamic, agentic workflows where a central planner orchestrates multi-turn reasoning. However, existing paradigms face a critical dichotomy: they either optimize modules jointly within rigid, fixed-graph architectures, or empower dynamic planning while treating executors as frozen, black-box tools. We identify that this \textit{decoupled optimization} creates a ``strategic-operational mismatch,'' where sophisticated planning strategies fail to materialize due to unadapted local executors, often leading to negative performance gains despite increased system complexity. In this paper, we propose \textbf{JADE} (\textbf{J}oint \textbf{A}gentic \textbf{D}ynamic \textbf{E}xecution), a unified framework for the joint optimization of planning and execution within dynamic, multi-turn workflows. By modeling the system as a cooperative multi-agent team unified under a single shared backbone, JADE enables end-to-end learning driven by outcome-based rewards. This approach facilitates \textit{co-adaptation}: the planner learns to operate within the capability boundaries of the executors, while the executors evolve to align with high-level strategic intent. Empirical results demonstrate that JADE transforms disjoint modules into a synergistic system, yielding remarkable performance improvements via joint optimization and enabling a flexible balance between efficiency and effectiveness through dynamic workflow orchestration.

**arXiv ID:** 2601.21916
</details>

<details>
<summary><strong>Self-Compression of Chain-of-Thought via Multi-Agent Reinforcement Learning</strong> - Yiqun Chen, Jinyuan Feng, Wei Yang, Meizhi Zhong, Zhengliang Shi, Rui Li, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Zhiqiang Pu, Jiaxin Mao - [[pdf]](https://arxiv.org/pdf/2601.21919)</summary>

**Abstract:** The inference overhead induced by redundant reasoning undermines the interactive experience and severely bottlenecks the deployment of Large Reasoning Models. Existing reinforcement learning (RL)-based solutions tackle this problem by coupling a length penalty with outcome-based rewards. This simplistic reward weighting struggles to reconcile brevity with accuracy, as enforcing brevity may compromise critical reasoning logic. In this work, we address this limitation by proposing a multi-agent RL framework that selectively penalizes redundant chunks, while preserving essential reasoning logic. Our framework, Self-Compression via MARL (SCMA), instantiates redundancy detection and evaluation through two specialized agents: \textbf{a Segmentation Agent} for decomposing the reasoning process into logical chunks, and \textbf{a Scoring Agent} for quantifying the significance of each chunk. The Segmentation and Scoring agents collaboratively define an importance-weighted length penalty during training, incentivizing \textbf{a Reasoning Agent} to prioritize essential logic without introducing inference overhead during deployment. Empirical evaluations across model scales demonstrate that SCMA reduces response length by 11.1\% to 39.0\% while boosting accuracy by 4.33\% to 10.02\%. Furthermore, ablation studies and qualitative analysis validate that the synergistic optimization within the MARL framework fosters emergent behaviors, yielding more powerful LRMs compared to vanilla RL paradigms.

**arXiv ID:** 2601.21919
</details>

<details>
<summary><strong>AgenticSimLaw: A Juvenile Courtroom Multi-Agent Debate Simulation for Explainable High-Stakes Tabular Decision Making</strong> - Jon Chun, Kathrine Elkins, Yong Suk Lee - [[pdf]](https://arxiv.org/pdf/2601.21936)</summary>

**Abstract:** We introduce AgenticSimLaw, a role-structured, multi-agent debate framework that provides transparent and controllable test-time reasoning for high-stakes tabular decision-making tasks. Unlike black-box approaches, our courtroom-style orchestration explicitly defines agent roles (prosecutor, defense, judge), interaction protocols (7-turn structured debate), and private reasoning strategies, creating a fully auditable decision-making process. We benchmark this framework on young adult recidivism prediction using the NLSY97 dataset, comparing it against traditional chain-of-thought (CoT) prompting across almost 90 unique combinations of models and strategies. Our results demonstrate that structured multi-agent debate provides more stable and generalizable performance compared to single-agent reasoning, with stronger correlation between accuracy and F1-score metrics. Beyond performance improvements, AgenticSimLaw offers fine-grained control over reasoning steps, generates complete interaction transcripts for explainability, and enables systematic profiling of agent behaviors. While we instantiate this framework in the criminal justice domain to stress-test reasoning under ethical complexity, the approach generalizes to any deliberative, high-stakes decision task requiring transparency and human oversight. This work addresses key LLM-based multi-agent system challenges: organization through structured roles, observability through logged interactions, and responsibility through explicit non-deployment constraints for sensitive domains. Data, results, and code will be available on this http URL under the MIT license.

**arXiv ID:** 2601.21936
</details>

<details>
<summary><strong>Learning Decentralized LLM Collaboration with Multi-Agent Actor Critic</strong> - Shuo Liu, Tianle Chen, Ryan Amiri, Christopher Amato - [[pdf]](https://arxiv.org/pdf/2601.21972)</summary>

**Abstract:** Recent work has explored optimizing LLM collaboration through Multi-Agent Reinforcement Learning (MARL). However, most MARL fine-tuning approaches rely on predefined execution protocols, which often require centralized execution. Decentralized LLM collaboration is more appealing in practice, as agents can run inference in parallel with flexible deployments. Also, current approaches use Monte Carlo methods for fine-tuning, which suffer from high variance and thus require more samples to train effectively. Actor-critic methods are prevalent in MARL for dealing with these issues, so we developed Multi-Agent Actor-Critic (MAAC) methods to optimize decentralized LLM collaboration. In this paper, we analyze when and why these MAAC methods are beneficial. We propose 2 MAAC approaches, \textbf{CoLLM-CC} with a \textbf{C}entralized \textbf{C}ritic and \textbf{CoLLM-DC} with \textbf{D}ecentralized \textbf{C}ritics. Our experiments across writing, coding, and game-playing domains show that Monte Carlo methods and CoLLM-DC can achieve performance comparable to CoLLM-CC in short-horizon and dense-reward settings. However, they both underperform CoLLM-CC on long-horizon or sparse-reward tasks, where Monte Carlo methods require substantially more samples and CoLLM-DC struggles to converge. Our code is available at this https URL.

**arXiv ID:** 2601.21972
</details>

<details>
<summary><strong>Adaptive Confidence Gating in Multi-Agent Collaboration for Efficient and Optimized Code Generation</strong> - Haoji Zhang, Yuzhe Li, Zhenqiang Liu, Chenyang Liu, Shenyang Zhang, Yi Zhou - [[pdf]](https://arxiv.org/pdf/2601.21469)</summary>

**Abstract:** While Large Language Models (LLMs) have catalyzed breakthroughs in automated code generation, Small Language Models (SLMs) often encounter reasoning bottlenecks and failure loops when addressing complex logical requirements. To overcome these challenges, we propose DebateCoder, a multi-agent collaborative framework designed to improve the reasoning ability of SLMs (e.g., Pangu-1B) in resource-constrained environments. DebateCoder uses a structured role-playing protocol with three agents: User Agent (A_UA), Technical Agent (A_TA), and Quality Assurance Agent (A_QA). It also includes an Adaptive Confidence Gating mechanism with a 95% threshold to balance accuracy and inference efficiency. In addition, we introduce a multi-turn deliberation module and a reviewer-guided analytical debugging loop for orthogonal pre-generation debate and post-generation refinement. Experiments on HumanEval and MBPP show that DebateCoder achieves 70.12% Pass@1 on HumanEval, outperforming MapCoder while reducing API overhead by about 35%. These results indicate that collaborative protocols can mitigate limitations of small-parameter models and provide a scalable, efficient approach to high-quality automated software engineering.

**arXiv ID:** 2601.21469
</details>

<details>
<summary><strong>Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning</strong> - Wonduk Seo, Wonseok Choi, Junseo Koh, Juhyeon Lee, Hyunjin An, Minhyeong Yu, Jian Park, Qingshan Zhou, Seunghyun Lee, Yi Bu - [[pdf]](https://arxiv.org/pdf/2601.21700)</summary>

**Abstract:** Large Language Models (LLMs) increasingly support culturally sensitive decision making, yet often exhibit misalignment due to skewed pretraining data and the absence of structured value representations. Existing methods can steer outputs, but often lack demographic grounding and treat values as independent, unstructured signals, reducing consistency and interpretability. We propose OG-MAR, an Ontology-Guided Multi-Agent Reasoning framework. OG-MAR summarizes respondent-specific values from the World Values Survey (WVS) and constructs a global cultural ontology by eliciting relations over a fixed taxonomy via competency questions. At inference time, it retrieves ontology-consistent relations and demographically similar profiles to instantiate multiple value-persona agents, whose outputs are synthesized by a judgment agent that enforces ontology consistency and demographic proximity. Experiments on regional social-survey benchmarks across four LLM backbones show that OG-MAR improves cultural alignment and robustness over competitive baselines, while producing more transparent reasoning traces.

**arXiv ID:** 2601.21700
</details>

<details>
<summary><strong>Learning to Communicate Across Modalities: Perceptual Heterogeneity in Multi-Agent Systems</strong> - Naomi Pitzer, Daniela Mihai - [[pdf]](https://arxiv.org/pdf/2601.22041)</summary>

**Abstract:** Emergent communication offers insight into how agents develop shared structured representations, yet most research assumes homogeneous modalities or aligned representational spaces, overlooking the perceptual heterogeneity of real-world settings. We study a heterogeneous multi-step binary communication game where agents differ in modality and lack perceptual grounding. Despite perceptual misalignment, multimodal systems converge to class-consistent messages grounded in perceptual input. Unimodal systems communicate more efficiently, using fewer bits and achieving lower classification entropy, while multimodal agents require greater information exchange and exhibit higher uncertainty. Bit perturbation experiments provide strong evidence that meaning is encoded in a distributional rather than compositional manner, as each bit's contribution depends on its surrounding pattern. Finally, interoperability analyses show that systems trained in different perceptual worlds fail to directly communicate, but limited fine-tuning enables successful cross-system communication. This work positions emergent communication as a framework for studying how agents adapt and transfer representations across heterogeneous modalities, opening new directions for both theory and experimentation.

**arXiv ID:** 2601.22041
</details>

<details>
<summary><strong>TransLaw: A Large-Scale Dataset and Multi-Agent Benchmark Simulating Professional Translation of Hong Kong Case Law</strong> - Xi Xuan, Chunyu Kit - [[pdf]](https://arxiv.org/pdf/2507.00875)</summary>

**Abstract:** Hong Kong case law translation presents significant challenges: manual methods suffer from high costs and inconsistent quality, while both traditional machine translation and approaches relying solely on Large Language Models (LLMs) often fail to ensure legal terminology accuracy, culturally embedded nuances, and strict linguistic structures. To overcome these limitations, this study proposes TransLaw, a multi-agent framework that decomposes translation into word-level expression, sentence-level translation, and multidimensional review, integrating a specialized Hong Kong legal glossary database, Retrieval-Augmented Generation (RAG), and iterative feedback. Experiments on our newly constructed HKCFA Judgment 97-22 dataset, benchmarking 13 open-source and commercial LLMs, demonstrate that TransLaw significantly outperforms single-agent baselines across all evaluated models. Human evaluation confirms the framework's effectiveness in terms of legal semantic accuracy, structural coherence, and stylistic fidelity, while noting that it still trails human experts in contextualizing complex terminology and stylistic naturalness.

**arXiv ID:** 2507.00875
</details>

<details>
<summary><strong>What the flock knows that the birds do not: exploring the emergence of joint agency in multi-agent active inference</strong> - Domenico Maisto, Davide Nuzzi, Giovanni Pezzulo - [[pdf]](https://arxiv.org/pdf/2511.10835)</summary>

**Abstract:** Collective behavior pervades biological systems, from flocks of birds to neural assemblies and human societies. Yet, how such collectives acquire functional properties -- such as joint agency or knowledge -- that transcend those of their individual components remains an open question. Here, we combine active inference and information-theoretic analyses to explore how a minimal system of interacting agents can give rise to joint agency and collective knowledge. We model flocking dynamics using multiple active inference agents, each minimizing its own free energy while coupling reciprocally with its neighbors. We show that as agents self-organize, their interactions define higher-order statistical boundaries (Markov blankets) enclosing a ``flock'' that can be treated as an emergent agent with its own sensory, active, and internal states. When exposed to external perturbations (a ``predator''), the flock exhibits faster, coordinated responses than individual agents, reflecting collective sensitivity to environmental change. Crucially, analyses of synergistic information reveal that the flock encodes information about the predator's location that is not accessible to every individual bird, demonstrating implicit collective knowledge. Together, these results show how informational coupling among active inference agents can generate new levels of autonomy and inference, providing a framework for understanding the emergence of (implicit) collective knowledge and joint agency.

**arXiv ID:** 2511.10835
</details>

<details>
<summary><strong>MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks</strong> - Zixuan Ke, Yifei Ming, Austin Xu, Ryan Chin, Xuan-Phi Nguyen, Prathyusha Jwalapuram, Jiayu Wang, Semih Yavuz, Caiming Xiong, Shafiq Joty - [[pdf]](https://arxiv.org/pdf/2601.14652)</summary>

**Abstract:** While multi-agent systems (MAS) promise elevated intelligence through coordination of agents, current approaches to automatic MAS design under-deliver. Such shortcomings stem from two key factors: (1) methodological complexity - agent orchestration is performed using sequential, code-level execution that limits global system-level holistic reasoning and scales poorly with agent complexity - and (2) efficacy uncertainty - MAS are deployed without understanding if there are tangible benefits compared to single-agent systems (SAS). We propose MASOrchestra, a training-time framework that formulates MAS orchestration as a function-calling reinforcement learning problem with holistic orchestration, generating an entire MAS at once. In MAS-Orchestra, complex, goal-oriented subagents are abstracted as callable functions, enabling global reasoning over system structure while hiding internal execution details. To rigorously study when and why MAS are beneficial, we introduce MASBENCH, a controlled benchmark that characterizes tasks along five axes: Depth, Horizon, Breadth, Parallel, and Robustness. Our analysis reveals that MAS gains depend critically on task structure, verification protocols, and the capabilities of both orchestrator and subagents, rather than holding universally. Guided by these insights, MAS-Orchestra achieves consistent improvements on public benchmarks including mathematical reasoning, multi-hop QA, and search-based QA, while achieving more than 10x efficiency over strong baselines. Together, MAS-Orchestra and MASBENCH enable better training and understanding of MAS in the pursuit of multi-agent intelligence.

**arXiv ID:** 2601.14652
</details>

<details>
<summary><strong>LLMs as Orchestrators: Constraint-Compliant Multi-Agent Optimization for Recommendation Systems</strong> - Guilin Zhang, Kai Zhao, Jeffrey Friedman, Xu Chu - [[pdf]](https://arxiv.org/pdf/2601.19121)</summary>

**Abstract:** Recommendation systems must optimize multiple objectives while satisfying hard business constraints such as fairness and coverage. For example, an e-commerce platform may require every recommendation list to include items from multiple sellers and at least one newly listed product; violating such constraints--even once--is unacceptable in production. Prior work on multi-objective recommendation and recent LLM-based recommender agents largely treat constraints as soft penalties or focus on item scoring and interaction, leading to frequent violations in real-world deployments. How to leverage LLMs for coordinating constrained optimization in recommendation systems remains underexplored. We propose DualAgent-Rec, an LLM-coordinated dual-agent framework for constrained multi-objective e-commerce recommendation. The framework separates optimization into an Exploitation Agent that prioritizes accuracy under hard constraints and an Exploration Agent that promotes diversity through unconstrained Pareto search. An LLM-based coordinator adaptively allocates resources between agents based on optimization progress and constraint satisfaction, while an adaptive epsilon-relaxation mechanism guarantees feasibility of final solutions. Experiments on the Amazon Reviews 2023 dataset demonstrate that DualAgent-Rec achieves 100% constraint satisfaction and improves Pareto hypervolume by 4-6% over strong baselines, while maintaining competitive accuracy-diversity trade-offs. These results indicate that LLMs can act as effective orchestration agents for deployable and constraint-compliant recommendation systems.

**arXiv ID:** 2601.19121
</details>

<details>
<summary><strong>LLM$\times$MapReduce-V3: Enabling Interactive In-Depth Survey Generation through a MCP-Driven Hierarchically Modular Agent System</strong> - Yu Chao, Siyu Lin, xiaorong wang, Zhu Zhang, Zihan Zhou, Haoyu Wang, Shuo Wang, Jie Zhou, Zhiyuan Liu, Maosong Sun - [[pdf]](https://arxiv.org/pdf/2510.10890)</summary>

**Abstract:** We introduce LLM x MapReduce-V3, a hierarchically modular agent system designed for long-form survey generation. Building on the prior work, LLM x MapReduce-V2, this version incorporates a multi-agent architecture where individual functional components, such as skeleton initialization, digest construction, and skeleton refinement, are implemented as independent model-context-protocol (MCP) servers. These atomic servers can be aggregated into higher-level servers, creating a hierarchically structured system. A high-level planner agent dynamically orchestrates the workflow by selecting appropriate modules based on their MCP tool descriptions and the execution history. This modular decomposition facilitates human-in-the-loop intervention, affording users greater control and customization over the research process. Through a multi-turn interaction, the system precisely captures the intended research perspectives to generate a comprehensive skeleton, which is then developed into an in-depth survey. Human evaluations demonstrate that our system surpasses representative baselines in both content depth and length, highlighting the strength of MCP-based modular planning.

**arXiv ID:** 2510.10890
</details>

<details>
<summary><strong>Explicit Credit Assignment through Local Rewards and Dependence Graphs in Multi-Agent Reinforcement Learning</strong> - Bang Giang Le, Viet Cuong Ta - [[pdf]](https://arxiv.org/pdf/2601.21523)</summary>

**Abstract:** To promote cooperation in Multi-Agent Reinforcement Learning, the reward signals of all agents can be aggregated together, forming global rewards that are commonly known as the fully cooperative setting. However, global rewards are usually noisy because they contain the contributions of all agents, which have to be resolved in the credit assignment process. On the other hand, using local reward benefits from faster learning due to the separation of agents' contributions, but can be suboptimal as agents myopically optimize their own reward while disregarding the global optimality. In this work, we propose a method that combines the merits of both approaches. By using a graph of interaction between agents, our method discerns the individual agent contribution in a more fine-grained manner than a global reward, while alleviating the cooperation problem with agents' local reward. We also introduce a practical approach for approximating such a graph. Our experiments demonstrate the flexibility of the approach, enabling improvements over the traditional local and global reward settings.

**arXiv ID:** 2601.21523
</details>

<details>
<summary><strong>World Craft: Agentic Framework to Create Visualizable Worlds via Text</strong> - Jianwen Sun, Yukang Feng, Kaining Ying, Chuanhao Li, Zizhen Li, Fanrui Zhang, Jiaxin Ai, Yifan Chang, Yu Dai, Yifei Huang, Kaipeng Zhang - [[pdf]](https://arxiv.org/pdf/2601.09150)</summary>

**Abstract:** Large Language Models (LLMs) motivate generative agent simulation (e.g., AI Town) to create a ``dynamic world'', holding immense value across entertainment and research. However, for non-experts, especially those without programming skills, it isn't easy to customize a visualizable environment by themselves. In this paper, we introduce World Craft, an agentic world creation framework to create an executable and visualizable AI Town via user textual descriptions. It consists of two main modules, World Scaffold and World Guild. World Scaffold is a structured and concise standardization to develop interactive game scenes, serving as an efficient scaffolding for LLMs to customize an executable AI Town-like environment. World Guild is a multi-agent framework to progressively analyze users' intents from rough descriptions, and synthesizes required structured contents (\eg environment layout and assets) for World Scaffold . Moreover, we construct a high-quality error-correction dataset via reverse engineering to enhance spatial knowledge and improve the stability and controllability of layout generation, while reporting multi-dimensional evaluation metrics for further analysis. Extensive experiments demonstrate that our framework significantly outperforms existing commercial code agents (Cursor and Antigravity) and LLMs (Qwen3 and Gemini-3-Pro). in scene construction and narrative intent conveyance, providing a scalable solution for the democratization of environment creation.

**arXiv ID:** 2601.09150
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>Heterogeneous Computing: The Key to Powering the Future of AI Agent Inference</strong> - Yiren Zhao, Junyi Liu - [[pdf]](https://arxiv.org/pdf/2601.22001)</summary>

**Abstract:** AI agent inference is driving an inference heavy datacenter future and exposes bottlenecks beyond compute - especially memory capacity, memory bandwidth and high-speed interconnect. We introduce two metrics - Operational Intensity (OI) and Capacity Footprint (CF) - that jointly explain regimes the classic roofline analysis misses, including the memory capacity wall. Across agentic workflows (chat, coding, web use, computer use) and base model choices (GQA/MLA, MoE, quantization), OI/CF can shift dramatically, with long context KV cache making decode highly memory bound. These observations motivate disaggregated serving and system level heterogeneity: specialized prefill and decode accelerators, broader scale up networking, and decoupled compute-memory enabled by optical I/O. We further hypothesize agent-hardware co design, multiple inference accelerators within one system, and high bandwidth, large capacity memory disaggregation as foundations for adaptation to evolving OI/CF. Together, these directions chart a path to sustain efficiency and capability for large scale agentic AI inference.

**arXiv ID:** 2601.22001
</details>

<details>
<summary><strong>A2RAG: Adaptive Agentic Graph Retrieval for Cost-Aware and Reliable Reasoning</strong> - Jiate Liu, Zebin Chen, Shaobo Qiao, Mingchen Ju, Danting Zhang, Bocheng Han, Shuyue Yu, Xin Shu, Jingling Wu, Dong Wen, Xin Cao, Guanfeng Liu, Zhengyi Yang - [[pdf]](https://arxiv.org/pdf/2601.21162)</summary>

**Abstract:** Graph Retrieval-Augmented Generation (Graph-RAG) enhances multihop question answering by organizing corpora into knowledge graphs and routing evidence through relational structure. However, practical deployments face two persistent bottlenecks: (i) mixed-difficulty workloads where one-size-fits-all retrieval either wastes cost on easy queries or fails on hard multihop cases, and (ii) extraction loss, where graph abstraction omits fine-grained qualifiers that remain only in source text. We present A2RAG, an adaptive-and-agentic GraphRAG framework for cost-aware and reliable reasoning. A2RAG couples an adaptive controller that verifies evidence sufficiency and triggers targeted refinement only when necessary, with an agentic retriever that progressively escalates retrieval effort and maps graph signals back to provenance text to remain robust under extraction loss and incomplete graphs. Experiments on HotpotQA and 2WikiMultiHopQA demonstrate that A2RAG achieves +9.9/+11.8 absolute gains in Recall@2, while cutting token consumption and end-to-end latency by about 50% relative to iterative multihop baselines.

**arXiv ID:** 2601.21162
</details>

<details>
<summary><strong>Trustworthy Intelligent Education: A Systematic Perspective on Progress, Challenges, and Future Directions</strong> - Xiaoshan Yu, Shangshang Yang, Ziwen Wang, Haiping Ma, Xingyi Zhang - [[pdf]](https://arxiv.org/pdf/2601.21837)</summary>

**Abstract:** In recent years, trustworthiness has garnered increasing attention and exploration in the field of intelligent education, due to the inherent sensitivity of educational scenarios, such as involving minors and vulnerable groups, highly personalized learning data, and high-stakes educational outcomes. However, existing research either focuses on task-specific trustworthy methods without a holistic view of trustworthy intelligent education, or provides survey-level discussions that remain high-level and fragmented, lacking a clear and systematic categorization. To address these limitations, in this paper, we present a systematic and structured review of trustworthy intelligent education. Specifically, We first organize intelligent education into five representative task categories: learner ability assessment, learning resource recommendation, learning analytics, educational content understanding, and instructional assistance. Building on this task landscape, we review existing studies from five trustworthiness perspectives, including safety and privacy, robustness, fairness, explainability, and sustainability, and summarize and categorize the research methodologies and solution strategies therein. Finally, we summarize key challenges and discuss future research directions. This survey aims to provide a coherent reference framework and facilitate a clearer understanding of trustworthiness in intelligent education.

**arXiv ID:** 2601.21837
</details>

<details>
<summary><strong>From Particles to Agents: Hallucination as a Metric for Cognitive Friction in Spatial Simulation</strong> - Javier Argota Sánchez-Vaquerizo, Luis Borunda Monsivais - [[pdf]](https://arxiv.org/pdf/2601.21977)</summary>

**Abstract:** Traditional architectural simulations (e.g. Computational Fluid Dynamics, evacuation, structural analysis) model elements as deterministic physics-based "particles" rather than cognitive "agents". To bridge this, we introduce \textbf{Agentic Environmental Simulations}, where Large Multimodal generative models actively predict the next state of spatial environments based on semantic expectation. Drawing on examples from accessibility-oriented AR pipelines and multimodal digital twins, we propose a shift from chronological time-steps to Episodic Spatial Reasoning, where simulations advance through meaningful, surprisal-triggered events. Within this framework we posit AI hallucinations as diagnostic tools. By formalizing the \textbf{Cognitive Friction} ($C_f$) it is possible to reveal "Phantom Affordances", i.e. semiotic ambiguities in built space. Finally, we challenge current HCI paradigms by treating environments as dynamic cognitive partners and propose a human-centered framework of cognitive orchestration for designing AI-driven simulations that preserve autonomy, affective clarity, and cognitive integrity.

**arXiv ID:** 2601.21977
</details>

<details>
<summary><strong>Track-centric Iterative Learning for Global Trajectory Optimization in Autonomous Racing</strong> - Youngim Nam, Jungbin Kim, Kyungtae Kang, Cheolhyeon Kwon - [[pdf]](https://arxiv.org/pdf/2601.21027)</summary>

**Abstract:** This paper presents a global trajectory optimization framework for minimizing lap time in autonomous racing under uncertain vehicle dynamics. Optimizing the trajectory over the full racing horizon is computationally expensive, and tracking such a trajectory in the real world hardly assures global optimality due to uncertain dynamics. Yet, existing work mostly focuses on dynamics learning at the tracking level, without updating the trajectory itself to account for the learned dynamics. To address these challenges, we propose a track-centric approach that directly learns and optimizes the full-horizon trajectory. We first represent trajectories through a track-agnostic parametric space in light of the wavelet transform. This space is then efficiently explored using Bayesian optimization, where the lap time of each candidate is evaluated by running simulations with the learned dynamics. This optimization is embedded in an iterative learning framework, where the optimized trajectory is deployed to collect real-world data for updating the dynamics, progressively refining the trajectory over the iterations. The effectiveness of the proposed framework is validated through simulations and real-world experiments, demonstrating lap time improvement of up to 20.7% over a nominal baseline and consistently outperforming state-of-the-art methods.

**arXiv ID:** 2601.21027
</details>

<details>
<summary><strong>Large-Scale Autonomous Gas Monitoring for Volcanic Environments: A Legged Robot on Mount Etna</strong> - Julia Richter, Turcan Tuna, Manthan Patel, Takahiro Miki, Devon Higgins, James Fox, Cesar Cadena, Andres Diaz, Marco Hutter - [[pdf]](https://arxiv.org/pdf/2601.07362)</summary>

**Abstract:** Volcanic gas emissions are key precursors of eruptive activity. Yet, obtaining accurate near-surface measurements remains hazardous and logistically challenging, motivating the need for autonomous solutions. Limited mobility in rough volcanic terrain has prevented wheeled systems from performing reliable in situ gas measurements, reducing their usefulness as sensing platforms. We present a legged robotic system for autonomous volcanic gas analysis, utilizing the quadruped ANYmal, equipped with a quadrupole mass spectrometer system. Our modular autonomy stack integrates a mission planning interface, global planner, localization framework, and terrain-aware local navigation. We evaluated the system on Mount Etna across three autonomous missions in varied terrain, achieving successful gas-source detections with autonomy rates of 93-100%. In addition, we conducted a teleoperated mission in which the robot measured natural fumaroles, detecting sulfur dioxide and carbon dioxide. We discuss lessons learned from the gas-analysis and autonomy perspectives, emphasizing the need for adaptive sensing strategies, tighter integration of global and local planning, and improved hardware design.

**arXiv ID:** 2601.07362
</details>

<details>
<summary><strong>Visual Localization via Semantic Structures in Autonomous Photovoltaic Power Plant Inspection</strong> - Viktor Kozák, Karel Košnar, Jan Chudoba, Miroslav Kulich, Libor Přeučil - [[pdf]](https://arxiv.org/pdf/2501.14587)</summary>

**Abstract:** Inspection systems utilizing unmanned aerial vehicles (UAVs) equipped with thermal cameras are increasingly popular for the maintenance of photovoltaic (PV) power plants. However, automation of the inspection task is a challenging problem as it requires precise navigation to capture images from optimal distances and viewing angles. This paper presents a novel localization pipeline that directly integrates PV module detection with UAV navigation, allowing precise positioning during inspection. The detections are used to identify the power plant structures in the image. These are associated with the power plant model and used to infer the UAV position relative to the inspected PV installation. We define visually recognizable anchor points for the initial association and use object tracking to discern global associations. Additionally, we present three different methods for visual segmentation of PV modules and evaluate their performance in relation to the proposed localization pipeline. The presented methods were verified and evaluated using custom aerial inspection data sets, demonstrating their robustness and applicability for real-time navigation. Additionally, we evaluate the influence of the power plant model precision on the localization methods.

**arXiv ID:** 2501.14587
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (39 papers)</h2></summary>

<details>
<summary><strong>Planner-Auditor Twin: Agentic Discharge Planning with FHIR-Based LLM Planning, Guideline Recall, Optional Caching and Self-Improvement</strong> - Kaiyuan Wu, Aditya Nagori, Rishikesan Kamaleswaran - [[pdf]](https://arxiv.org/pdf/2601.21113)</summary>

**Abstract:** Objective: Large language models (LLMs) show promise for clinical discharge planning, but their use is constrained by hallucination, omissions, and miscalibrated confidence. We introduce a self-improving, cache-optional Planner-Auditor framework that improves safety and reliability by decoupling generation from deterministic validation and targeted replay.
Materials and Methods: We implemented an agentic, retrospective, FHIR-native evaluation pipeline using MIMIC-IV-on-FHIR. For each patient, the Planner (LLM) generates a structured discharge action plan with an explicit confidence estimate. The Auditor is a deterministic module that evaluates multi-task coverage, tracks calibration (Brier score, ECE proxies), and monitors action-distribution drift. The framework supports two-tier self-improvement: (i) within-episode regeneration when enabled, and (ii) cross-episode discrepancy buffering with replay for high-confidence, low-coverage cases.
Results: While context caching improved performance over baseline, the self-improvement loop was the primary driver of gains, increasing task coverage from 32% to 86%. Calibration improved substantially, with reduced Brier/ECE and fewer high-confidence misses. Discrepancy buffering further corrected persistent high-confidence omissions during replay.
Discussion: Feedback-driven regeneration and targeted replay act as effective control mechanisms to reduce omissions and improve confidence reliability in structured clinical planning. Separating an LLM Planner from a rule-based, observational Auditor enables systematic reliability measurement and safer iteration without model retraining.
Conclusion: The Planner-Auditor framework offers a practical pathway toward safer automated discharge planning using interoperable FHIR data access and deterministic auditing, supported by reproducible ablations and reliability-focused evaluation.

**arXiv ID:** 2601.21113
</details>

<details>
<summary><strong>OpenSec: Measuring Incident Response Agent Calibration Under Adversarial Evidence</strong> - Jarrod Barnes - [[pdf]](https://arxiv.org/pdf/2601.21083)</summary>

**Abstract:** As large language models improve, so do their offensive applications: frontier agents now generate working exploits for under $50 in compute (Heelan, 2026). Defensive incident response (IR) agents must keep pace, but existing benchmarks conflate action execution with correct execution, hiding calibration failures when agents process adversarial evidence. We introduce OpenSec, a dual-control reinforcement learning environment that evaluates IR agents under realistic prompt injection scenarios. Unlike static capability benchmarks, OpenSec scores world-state-changing containment actions under adversarial evidence via execution-based metrics: time-to-first-containment (TTFC), blast radius (false positives per episode), and injection violation rates. Evaluating four frontier models on 40 standard-tier episodes, we find consistent over-triggering in this setting: GPT-5.2, Gemini 3, and DeepSeek execute containment in 100% of episodes with 90-97% false positive rates. Claude Sonnet 4.5 shows partial calibration (85% containment, 72% FP), demonstrating that OpenSec surfaces a calibration failure mode hidden by aggregate success metrics. Code available at this https URL.

**arXiv ID:** 2601.21083
</details>

<details>
<summary><strong>CUA-Skill: Develop Skills for Computer Using Agent</strong> - Tianyi Chen, Yinheng Li, Michael Solodko, Sen Wang, Nan Jiang, Tingyuan Cui, Junheng Hao, Jongwoo Ko, Sara Abdali, Suzhen Zheng, Leon Xu, Hao Fan, Pashmina Cameron, Justin Wagle, Kazuhito Koishida - [[pdf]](https://arxiv.org/pdf/2601.21123)</summary>

**Abstract:** Computer-Using Agents (CUAs) aim to autonomously operate computer systems to complete real-world tasks. However, existing agentic systems remain difficult to scale and lag behind human performance. A key limitation is the absence of reusable and structured skill abstractions that capture how humans interact with graphical user interfaces and how to leverage these skills. We introduce CUA-Skill, a computer-using agentic skill base that encodes human computer-use knowledge as skills coupled with parameterized execution and composition graphs. CUA-Skill is a large-scale library of carefully engineered skills spanning common Windows applications, serving as a practical infrastructure and tool substrate for scalable, reliable agent development. Built upon this skill base, we construct CUA-Skill Agent, an end-to-end computer-using agent that supports dynamic skill retrieval, argument instantiation, and memory-aware failure recovery. Our results demonstrate that CUA-Skill substantially improves execution success rates and robustness on challenging end-to-end agent benchmarks, establishing a strong foundation for future computer-using agent development. On WindowsAgentArena, CUA-Skill Agent achieves state-of-the-art 57.5% (best of three) successful rate while being significantly more efficient than prior and concurrent approaches. The project page is available at this https URL.

**arXiv ID:** 2601.21123
</details>

<details>
<summary><strong>When should I search more: Adaptive Complex Query Optimization with Reinforcement Learning</strong> - Wei Wen, Sihang Deng, Tianjun Wei, Keyu Chen, Ruizhi Qiao, Xing Sun - [[pdf]](https://arxiv.org/pdf/2601.21208)</summary>

**Abstract:** Query optimization is a crucial component for the efficacy of Retrieval-Augmented Generation (RAG) systems. While reinforcement learning (RL)-based agentic and reasoning methods have recently emerged as a promising direction on query optimization, most existing approaches focus on the expansion and abstraction of a single query. However, complex user queries are prevalent in real-world scenarios, often requiring multiple parallel and sequential search strategies to handle disambiguation and decomposition. Directly applying RL to these complex cases introduces significant hurdles. Determining the optimal number of sub-queries and effectively re-ranking and merging retrieved documents vastly expands the search space and complicates reward design, frequently leading to training instability. To address these challenges, we propose a novel RL framework called Adaptive Complex Query Optimization (ACQO). Our framework is designed to adaptively determine when and how to expand the search process. It features two core components: an Adaptive Query Reformulation (AQR) module that dynamically decides when to decompose a query into multiple sub-queries, and a Rank-Score Fusion (RSF) module that ensures robust result aggregation and provides stable reward signals for the learning agent. To mitigate training instabilities, we adopt a Curriculum Reinforcement Learning (CRL) approach, which stabilizes the training process by progressively introducing more challenging queries through a two-stage strategy. Our comprehensive experiments demonstrate that ACQO achieves state-of-the-art performance on three complex query benchmarks, significantly outperforming established baselines. The framework also showcases improved computational efficiency and broad compatibility with different retrieval architectures, establishing it as a powerful and generalizable solution for next-generation RAG systems.

**arXiv ID:** 2601.21208
</details>

<details>
<summary><strong>Intelli-Planner: Towards Customized Urban Planning via Large Language Model Empowered Reinforcement Learning</strong> - Xixian Yong, Peilin Sun, Zihe Wang, Xiao Zhou - [[pdf]](https://arxiv.org/pdf/2601.21212)</summary>

**Abstract:** Effective urban planning is crucial for enhancing residents' quality of life and ensuring societal stability, playing a pivotal role in the sustainable development of cities. Current planning methods heavily rely on human experts, which are time-consuming and labor-intensive, or utilize deep learning algorithms, often limiting stakeholder involvement. To bridge these gaps, we propose Intelli-Planner, a novel framework integrating Deep Reinforcement Learning (DRL) with large language models (LLMs) to facilitate participatory and customized planning scheme generation. Intelli-Planner utilizes demographic, geographic data, and planning preferences to determine high-level planning requirements and demands for each functional type. During training, a knowledge enhancement module is employed to enhance the decision-making capability of the policy network. Additionally, we establish a multi-dimensional evaluation system and employ LLM-based stakeholders for satisfaction scoring. Experimental validation across diverse urban settings shows that Intelli-Planner surpasses traditional baselines and achieves comparable performance to state-of-the-art DRL-based methods in objective metrics, while enhancing stakeholder satisfaction and convergence speed. These findings underscore the effectiveness and superiority of our framework, highlighting the potential for integrating the latest advancements in LLMs with DRL approaches to revolutionize tasks related to functional areas planning.

**arXiv ID:** 2601.21212
</details>

<details>
<summary><strong>BEAP-Agent: Backtrackable Execution and Adaptive Planning for GUI Agents</strong> - Ziyu Lu, Tengjin Weng, Yiying Yang, Yuhang Zhao, Xinxin Huang, Wenhao Jiang - [[pdf]](https://arxiv.org/pdf/2601.21352)</summary>

**Abstract:** GUI agents are designed to automate repetitive tasks and enhance productivity. However, existing GUI agents struggle to recover once they follow an incorrect exploration path, often leading to task failure. In this work, we model GUI task execution as a DFS process and propose BEAP-Agent, a DFS-based framework that supports long-range, multi-level state backtracking with dynamic task tracking and updating. The framework consists of three collaborative components: Planner, Executor, and Tracker. Together, they enable effective task exploration and execution. BEAP-Agent fills the gap in systematic backtracking mechanisms for GUI agents, offering a systematic solution for long-horizon task exploration. We conducted a systematic evaluation on the OSWorld benchmark, where BEAP-Agent achieved an accuracy of 28.2%, validating the effectiveness of the proposed method.

**arXiv ID:** 2601.21352
</details>

<details>
<summary><strong>KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization</strong> - Alireza Nadaf, Alireza Mohammadshahi, Majid Yazdani - [[pdf]](https://arxiv.org/pdf/2601.21526)</summary>

**Abstract:** We introduce KAPSO, a modular framework for autonomous program synthesis and optimization. Given a natural language goal and an evaluation method, KAPSO iteratively performs ideation, code synthesis and editing, execution, evaluation, and learning to improve a runnable artifact toward measurable objectives. Rather than treating synthesis as the endpoint, KAPSO uses synthesis as an operator within a long-horizon optimization loop, where progress is defined by evaluator outcomes.
KAPSO targets long-horizon failures common in coding agents, including lost experimental state, brittle debugging, and weak reuse of domain expertise, by integrating three tightly coupled components. First, a git-native experimentation engine isolates each attempt as a branch, producing reproducible artifacts and preserving provenance across iterations. Second, a knowledge system ingests heterogeneous sources, including repositories, internal playbooks, and curated external resources such as documentation, scientific papers, and web search results, and organizes them into a structured representation that supports retrieval over workflows, implementations, and environment constraints. Third, a cognitive memory layer coordinates retrieval and maintains an episodic store of reusable lessons distilled from experiment traces (run logs, diffs, and evaluator feedback), reducing repeated error modes and accelerating convergence.
We evaluated KAPSO on MLE-Bench (Kaggle-style ML competitions) and ALE-Bench (AtCoder heuristic optimization), and report end-to-end performance.
Code Available at: this https URL

**arXiv ID:** 2601.21526
</details>

<details>
<summary><strong>Beyond Imitation: Reinforcement Learning for Active Latent Planning</strong> - Zhi Zheng, Wee Sun Lee - [[pdf]](https://arxiv.org/pdf/2601.21598)</summary>

**Abstract:** Aiming at efficient and dense chain-of-thought (CoT) reasoning, latent reasoning methods fine-tune Large Language Models (LLMs) to substitute discrete language tokens with continuous latent tokens. These methods consume fewer tokens compared to the conventional language CoT reasoning and have the potential to plan in a dense latent space. However, current latent tokens are generally supervised based on imitating language labels. Considering that there can be multiple equivalent but diverse CoT labels for a question, passively imitating an arbitrary one may lead to inferior latent token representations and latent reasoning policies, undermining the potential planning ability and resulting in clear gaps between training and testing. In this work, we emphasize the importance of active planning over the representation space of latent tokens in achieving the optimal latent reasoning policy. So, we propose the \underline{A}c\underline{t}ive Latent \underline{P}lanning method (ATP-Latent), which models the supervision process of latent tokens as a conditional variational auto-encoder (VAE) to obtain a smoother latent space. Moreover, to facilitate the most reasonable latent reasoning policy, ATP-Latent conducts reinforcement learning (RL) with an auxiliary coherence reward, which is calculated based on the consistency between VAE-decoded contents of latent tokens, enabling a guided RL process. In experiments on LLaMA-1B, ATP-Latent demonstrates +4.1\% accuracy and -3.3\% tokens on four benchmarks compared to advanced baselines. Codes are available on this https URL.

**arXiv ID:** 2601.21598
</details>

<details>
<summary><strong>BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics</strong> - Dionizije Fa, Marko Čuljak, Bruno Pandža, Mateo Čupić - [[pdf]](https://arxiv.org/pdf/2601.21800)</summary>

**Abstract:** This paper introduces BioAgent Bench, a benchmark dataset and an evaluation suite designed for measuring the performance and robustness of AI agents in common bioinformatics tasks. The benchmark contains curated end-to-end tasks (e.g., RNA-seq, variant calling, metagenomics) with prompts that specify concrete output artifacts to support automated assessment, including stress testing under controlled perturbations. We evaluate frontier closed-source and open-weight models across multiple agent harnesses, and use an LLM-based grader to score pipeline progress and outcome validity. We find that frontier agents can complete multi-step bioinformatics pipelines without elaborate custom scaffolding, often producing the requested final artifacts reliably. However, robustness tests reveal failure modes under controlled perturbations (corrupted inputs, decoy files, and prompt bloat), indicating that correct high-level pipeline construction does not guarantee reliable step-level reasoning. Finally, because bioinformatics workflows may involve sensitive patient data, proprietary references, or unpublished IP, closed-source models can be unsuitable under strict privacy constraints; in such settings, open-weight models may be preferable despite lower completion rates. We release the dataset and evaluation suite publicly.

**arXiv ID:** 2601.21800
</details>

<details>
<summary><strong>WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents</strong> - Yao Zhang, Shijie Tang, Zeyu Li, Zhen Han, Volker Tresp - [[pdf]](https://arxiv.org/pdf/2601.21872)</summary>

**Abstract:** Web agents hold great potential for automating complex computer tasks, yet their interactions involve long-horizon, sequential decision-making with irreversible actions. In such settings, outcome-based supervision is sparse and delayed, often rewarding incorrect trajectories and failing to support inference-time scaling. This motivates the use of Process Reward Models (WebPRMs) for web navigation, but existing approaches remain limited: scalar WebPRMs collapse progress into coarse, weakly grounded signals, while checklist-based WebPRMs rely on brittle template matching that fails under layout or semantic changes and often mislabels superficially correct actions as successful, providing little insight or interpretability. To address these challenges, we introduce WebArbiter, a reasoning-first, principle-inducing WebPRM that formulates reward modeling as text generation, producing structured justifications that conclude with a preference verdict and identify the action most conducive to task completion under the current context. Training follows a two-stage pipeline: reasoning distillation equips the model with coherent principle-guided reasoning, and reinforcement learning corrects teacher biases by directly aligning verdicts with correctness, enabling stronger generalization. To support systematic evaluation, we release WebPRMBench, a comprehensive benchmark spanning four diverse web environments with rich tasks and high-quality preference annotations. On WebPRMBench, WebArbiter-7B outperforms the strongest baseline, GPT-5, by 9.1 points. In reward-guided trajectory search on WebArena-Lite, it surpasses the best prior WebPRM by up to 7.2 points, underscoring its robustness and practical value in real-world complex web tasks.

**arXiv ID:** 2601.21872
</details>

<details>
<summary><strong>ProRAG: Process-Supervised Reinforcement Learning for Retrieval-Augmented Generation</strong> - Zhao Wang, Ziliang Zhao, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2601.21912)</summary>

**Abstract:** Reinforcement learning (RL) has become a promising paradigm for optimizing Retrieval-Augmented Generation (RAG) in complex reasoning tasks. However, traditional outcome-based RL approaches often suffer from reward sparsity and inefficient credit assignment, as coarse-grained scalar rewards fail to identify specific erroneous steps within long-horizon trajectories. This ambiguity frequently leads to "process hallucinations", where models reach correct answers through flawed logic or redundant retrieval steps. Although recent process-aware approaches attempt to mitigate this via static preference learning or heuristic reward shaping, they often lack the on-policy exploration capabilities required to decouple step-level credit from global outcomes. To address these challenges, we propose ProRAG, a process-supervised reinforcement learning framework designed to integrate learned step-level supervision into the online optimization loop. Our framework consists of four stages: (1) Supervised Policy Warmup to initialize the model with a structured reasoning format; (2) construction of an MCTS-based Process Reward Model (PRM) to quantify intermediate reasoning quality; (3) PRM-Guided Reasoning Refinement to align the policy with fine-grained process preferences; and (4) Process-Supervised Reinforcement Learning with a dual-granularity advantage mechanism. By aggregating step-level process rewards with global outcome signals, ProRAG provides precise feedback for every action. Extensive experiments on five multi-hop reasoning benchmarks demonstrate that ProRAG achieves superior overall performance compared to strong outcome-based and process-aware RL baselines, particularly on complex long-horizon tasks, validating the effectiveness of fine-grained process supervision. The code and model are available at this https URL.

**arXiv ID:** 2601.21912
</details>

<details>
<summary><strong>Exploring Reasoning Reward Model for Agents</strong> - Kaixuan Fan, Kaituo Feng, Manyuan Zhang, Tianshuo Peng, Zhixun Li, Yilei Jiang, Shuang Chen, Peng Pei, Xunliang Cai, Xiangyu Yue - [[pdf]](https://arxiv.org/pdf/2601.22154)</summary>

**Abstract:** Agentic Reinforcement Learning (Agentic RL) has achieved notable success in enabling agents to perform complex reasoning and tool use. However, most methods still relies on sparse outcome-based reward for training. Such feedback fails to differentiate intermediate reasoning quality, leading to suboptimal training results. In this paper, we introduce Agent Reasoning Reward Model (Agent-RRM), a multi-faceted reward model that produces structured feedback for agentic trajectories, including (1) an explicit reasoning trace , (2) a focused critique that provides refinement guidance by highlighting reasoning flaws, and (3) an overall score that evaluates process performance. Leveraging these signals, we systematically investigate three integration strategies: Reagent-C (text-augmented refinement), Reagent-R (reward-augmented guidance), and Reagent-U (unified feedback integration). Extensive evaluations across 12 diverse benchmarks demonstrate that Reagent-U yields substantial performance leaps, achieving 43.7% on GAIA and 46.2% on WebWalkerQA, validating the effectiveness of our reasoning reward model and training schemes. Code, models, and datasets are all released to facilitate future research.

**arXiv ID:** 2601.22154
</details>

<details>
<summary><strong>Deep Reinforcement Learning for Fault-Adaptive Routing in Eisenstein-Jacobi Interconnection Topologies</strong> - Mohammad Walid Charrwi, Zaid Hussain - [[pdf]](https://arxiv.org/pdf/2601.21090)</summary>

**Abstract:** The increasing density of many-core architectures necessitates interconnection networks that are both high-performance and fault-resilient. Eisenstein-Jacobi (EJ) networks, with their symmetric 6-regular topology, offer superior topological properties but challenge traditional routing heuristics under fault conditions. This paper evaluates three routing paradigms in faulty EJ environments: deterministic Greedy Adaptive Routing, theoretically optimal Dijkstra's algorithm, and a reinforcement learning (RL)-based approach. Using a multi-objective reward function to penalize fault proximity and reward path efficiency, the RL agent learns to navigate around clustered failures that typically induce dead-ends in greedy geometric routing. Dijkstra's algorithm establishes the theoretical performance ceiling by computing globally optimal paths with complete topology knowledge, revealing the true connectivity limits of faulty networks. Quantitative analysis at nine faulty nodes shows greedy routing catastrophically degrades to 10% effective reachability and packet delivery, while Dijkstra proves 52-54% represents the topological optimum. The RL agent achieves 94% effective reachability and 91% packet delivery, making it suitable for distributed deployment. Furthermore, throughput evaluations demonstrate that RL sustains over 90% normalized throughput across all loads, actually outperforming Dijkstra under congestion through implicit load balancing strategies. These results establish RL-based adaptive policies as a practical solution that bridges the gap between greedy's efficiency and Dijkstra's optimality, providing robust, self-healing communication in fault-prone interconnection networks without requiring the global topology knowledge or computational overhead of optimal algorithms.

**arXiv ID:** 2601.21090
</details>

<details>
<summary><strong>Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed</strong> - Minjae Kwon, Josephine Lamp, Lu Feng - [[pdf]](https://arxiv.org/pdf/2601.21094)</summary>

**Abstract:** Safe Reinforcement Learning (RL) algorithms are typically evaluated under fixed training conditions. We investigate whether training-time safety guarantees transfer to deployment under distribution shift, using diabetes management as a safety-critical testbed. We benchmark safe RL algorithms on a unified clinical simulator and reveal a safety generalization gap: policies satisfying constraints during training frequently violate safety requirements on unseen patients. We demonstrate that test-time shielding, which filters unsafe actions using learned dynamics models, effectively restores safety across algorithms and patient populations. Across eight safe RL algorithms, three diabetes types, and three age groups, shielding achieves Time-in-Range gains of 13--14\% for strong baselines such as PPO-Lag and CPO while reducing clinical risk index and glucose variability. Our simulator and benchmark provide a platform for studying safety under distribution shift in safety-critical control domains. Code is available at this https URL and this https URL.

**arXiv ID:** 2601.21094
</details>

<details>
<summary><strong>Less Noise, More Voice: Reinforcement Learning for Reasoning via Instruction Purification</strong> - Yiju Guo, Tianyi Hu, Zexu Sun, Yankai Lin - [[pdf]](https://arxiv.org/pdf/2601.21244)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has advanced LLM reasoning, but remains constrained by inefficient exploration under limited rollout budgets, leading to low sampling success and unstable training in complex tasks. We find that many exploration failures arise not from problem difficulty, but from a small number of prompt tokens that introduce interference. Building on this insight, we propose the Less Noise Sampling Framework (LENS), which first prompts by identifying and removing interference tokens. then transfers successful rollouts from the purification process to supervise policy optimization on the original noisy prompts, enabling the model to learn to ignore interference in the real-world, noisy prompting settings. Experimental results show that LENS significantly outperforms GRPO, delivering higher performance and faster convergence, with a 3.88% average gain and over 1.6$\times$ speedup. Our work highlights the critical role of pruning interference tokens in improving rollout efficiency, offering a new perspective for RLVR research.

**arXiv ID:** 2601.21244
</details>

<details>
<summary><strong>The Surprising Difficulty of Search in Model-Based Reinforcement Learning</strong> - Wei-Di Chang, Mikael Henaff, Brandon Amos, Gregory Dudek, Scott Fujimoto - [[pdf]](https://arxiv.org/pdf/2601.21306)</summary>

**Abstract:** This paper investigates search in model-based reinforcement learning (RL). Conventional wisdom holds that long-term predictions and compounding errors are the primary obstacles for model-based RL. We challenge this view, showing that search is not a plug-and-play replacement for a learned policy. Surprisingly, we find that search can harm performance even when the model is highly accurate. Instead, we show that mitigating distribution shift matters more than improving model or value function accuracy. Building on this insight, we identify key techniques for enabling effective search, achieving state-of-the-art performance across multiple popular benchmark domains.

**arXiv ID:** 2601.21306
</details>

<details>
<summary><strong>Heterogeneous Vertiport Selection Optimization for On-Demand Air Taxi Services: A Deep Reinforcement Learning Approach</strong> - Aoyu Pang, Maonan Wang, Zifan Sha, Wenwei Yue, Changle Li, Chung Shue Chen, Man-On Pun - [[pdf]](https://arxiv.org/pdf/2601.21316)</summary>

**Abstract:** Urban Air Mobility (UAM) has emerged as a transformative solution to alleviate urban congestion by utilizing low-altitude airspace, thereby reducing pressure on ground transportation networks. To enable truly efficient and seamless door-to-door travel experiences, UAM requires close integration with existing ground transportation infrastructure. However, current research on optimal integrated routing strategies for passengers in air-ground mobility systems remains limited, with a lack of systematic this http URL address this gap, we first propose a unified optimization model that integrates strategy selection for both air and ground transportation. This model captures the dynamic characteristics of multimodal transport networks and incorporates real-time traffic conditions alongside passenger decision-making behavior. Building on this model, we propose a Unified Air-Ground Mobility Coordination (UAGMC) framework, which leverages deep reinforcement learning (RL) and Vehicle-to-Everything (V2X) communication to optimize vertiport selection and dynamically plan air taxi routes. Experimental results demonstrate that UAGMC achieves a 34\% reduction in average travel time compared to conventional proportional allocation methods, enhancing overall travel efficiency and providing novel insights into the integration and optimization of multimodal transportation systems. This work lays a solid foundation for advancing intelligent urban mobility solutions through the coordination of air and ground transportation modes. The related code can be found at this https URL.

**arXiv ID:** 2601.21316
</details>

<details>
<summary><strong>Mitigating Overthinking in Large Reasoning Models via Difficulty-aware Reinforcement Learning</strong> - Qian Wan, Ziao Xu, Luona Wei, Xiaoxuan Shen, Jianwen Sun - [[pdf]](https://arxiv.org/pdf/2601.21418)</summary>

**Abstract:** Large Reasoning Models (LRMs) achieve explicit chain-of-thought expansion by imitating deep thinking behaviors of humans, demonstrating excellent performance in complex task scenarios. However, the deep-thinking mode often leads to unnecessarily lengthy reasoning and resource inefficiency when handling simple tasks. This overthinking phenomenon may arise from the generation preference triggered by the reward function during post-training. Existing research attempts to mitigate overthinking from the perspective of prompt design or model training, but generally underestimates the importance of task difficulty awareness, which makes it difficult for LRMs to effectively allocate reasoning resources. In this paper, we propose Difficulty-aware Policy Optimization (DiPO), a reinforcement learning-based LRM training framework. DiPO encourages LRM to spontaneously model task complexity, and integrates them into reinforcement learning framework to adjust the generation preferences introduced by post-training. A difficulty modeling method based on model self-reasoning is proposed, which significantly reduces the dependence on manual annotation and formalize task complexity. We further develop a difficulty-signal-enhanced reward function that incorporates a penalty for lengthy reasoning while considering reasoning performance and output format. Experimental results indicate that DiPO enables the model to spontaneously adjust inference overhead, significantly reducing redundant tokens without losing performance due to thought compression.

**arXiv ID:** 2601.21418
</details>

<details>
<summary><strong>HER: Human-like Reasoning and Reinforcement Learning for LLM Role-playing</strong> - Chengyu Du, Xintao Wang, Aili Chen, Weiyuan Li, Rui Xu, Junteng Liu, Zishan Huang, Rong Tian, Zijun Sun, Yuhao Li, Liheng Feng, Deming Ding, Pengyu Zhao, Yanghua Xiao - [[pdf]](https://arxiv.org/pdf/2601.21459)</summary>

**Abstract:** LLM role-playing, i.e., using LLMs to simulate specific personas, has emerged as a key capability in various applications, such as companionship, content creation, and digital games. While current models effectively capture character tones and knowledge, simulating the inner thoughts behind their behaviors remains a challenge. Towards cognitive simulation in LLM role-play, previous efforts mainly suffer from two deficiencies: data with high-quality reasoning traces, and reliable reward signals aligned with human preferences. In this paper, we propose HER, a unified framework for cognitive-level persona simulation. HER introduces dual-layer thinking, which distinguishes characters' first-person thinking from LLMs' third-person thinking. To bridge these gaps, we curate reasoning-augmented role-playing data via reverse engineering and construct human-aligned principles and reward models. Leveraging these resources, we train \method models based on Qwen3-32B via supervised and reinforcement learning. Extensive experiments validate the effectiveness of our approach. Notably, our models significantly outperform the Qwen3-32B baseline, achieving a 30.26 improvement on the CoSER benchmark and a 14.97 gain on the Minimax Role-Play Bench. Our datasets, principles, and models will be released to facilitate future research.

**arXiv ID:** 2601.21459
</details>

<details>
<summary><strong>Expected Return Causes Outcome-Level Mode Collapse in Reinforcement Learning and How to Fix It with Inverse Probability Scaling</strong> - Abhijeet Sinha, Sundari Elango, Dianbo Liu - [[pdf]](https://arxiv.org/pdf/2601.21669)</summary>

**Abstract:** Many reinforcement learning (RL) problems admit multiple terminal solutions of comparable quality, where the goal is not to identify a single optimum but to represent a diverse set of high-quality outcomes. Nevertheless, policies trained by standard expected return maximization routinely collapse onto a small subset of outcomes, a phenomenon commonly attributed to insufficient exploration or weak regularization. We show that this explanation is incomplete: outcome level mode collapse is a structural consequence of the expected-return objective itself. Under idealized learning dynamics, the log-probability ratio between any two outcomes evolves linearly in their reward difference, implying exponential ratio divergence and inevitable collapse independent of the exploration strategy, entropy regularization, or optimization algorithm. We identify the source of this pathology as the probability multiplier inside the expectation and propose a minimal correction: inverse probability scaling, which removes outcome-frequency amplification from the learning signal, fundamentally changes the learning dynamics, and provably yields reward-proportional terminal distributions, preventing collapse in multimodal settings. We instantiate this principle in Group Relative Policy Optimization (GRPO) as a drop-in modification, IPS-GRPO, requiring no auxiliary models or architectural changes. Across different reasoning and molecular generation tasks, IPS-GRPO consistently reduces outcome-level mode collapse while matching or exceeding baseline performance, suggesting that correcting the objective rather than adding exploration heuristics is key to reliable multimodal policy optimization.

**arXiv ID:** 2601.21669
</details>

<details>
<summary><strong>TACLer: Tailored Curriculum Reinforcement Learning for Efficient Reasoning</strong> - Huiyuan Lai, Malvina Nissim - [[pdf]](https://arxiv.org/pdf/2601.21711)</summary>

**Abstract:** Large Language Models (LLMs) have shown remarkable performance on complex reasoning tasks, especially when equipped with long chain-of-thought (CoT) reasoning. However, eliciting long CoT typically requires large-scale reinforcement learning (RL) training, while often leading to overthinking with redundant intermediate steps. To improve learning and reasoning efficiency, while preserving or even enhancing performance, we propose TACLer, a model-tailored curriculum reinforcement learning framework that gradually increases the complexity of the data based on the model's proficiency in multi-stage RL training. TACLer features two core components: (i) tailored curriculum learning that determines what knowledge the model lacks and needs to learn in progressive stages; (ii) a hybrid Thinking/NoThinking reasoning paradigm that balances accuracy and efficiency by enabling or disabling the Thinking mode. Our experiments show that TACLer yields a twofold advantage in learning and reasoning: (i) it reduces computational cost, cutting training compute by over 50% compared to long thinking models and reducing inference token usage by over 42% relative to the base model; and (ii) it improves accuracy by over 9% on the base model, consistently outperforming state-of-the-art Nothinking and Thinking baselines across four math datasets with complex problems.

**arXiv ID:** 2601.21711
</details>

<details>
<summary><strong>Agents Trusting Agents? Restoring Lost Capabilities with Inclusive Healthcare</strong> - Alba Aguilera, Georgina Curto, Nardine Osman, Ahmed Al-Awah - [[pdf]](https://arxiv.org/pdf/2507.23644)</summary>

**Abstract:** Agent-based simulations have an untapped potential to inform social policies on urgent human development challenges in a non-invasive way, before these are implemented in real-world populations. This paper responds to the request from non-profit and governmental organizations to evaluate policies under discussion to improve equity in health care services for people experiencing homelessness (PEH) in the city of Barcelona. With this goal, we integrate the conceptual framework of the capability approach (CA), which is explicitly designed to promote and assess human well-being, to model and evaluate the behaviour of agents who represent PEH and social workers. We define a reinforcement learning environment where agents aim to restore their central human capabilities, under existing environmental and legal constraints. We use Bayesian inverse reinforcement learning (IRL) to calibrate profile-dependent behavioural parameters in PEH agents, modeling the degree of trust and engagement with social workers, which is reportedly a key element for the success of the policies in scope. Our results open a path to mitigate health inequity by building relationships of trust between social service workers and PEH.

**arXiv ID:** 2507.23644
</details>

<details>
<summary><strong>Adaptive Swarm Mesh Refinement using Deep Reinforcement Learning with Local Rewards</strong> - Niklas Freymuth, Philipp Dahlinger, Tobias Würth, Simon Reisch, Luise Kärger, Gerhard Neumann - [[pdf]](https://arxiv.org/pdf/2406.08440)</summary>

**Abstract:** Simulating physical systems is essential in engineering, but analytical solutions are limited to straightforward problems. Consequently, numerical methods like the Finite Element Method (FEM) are widely used. However, the FEM becomes computationally expensive as problem complexity and accuracy demands increase. Adaptive Mesh Refinement (AMR) improves the FEM by dynamically placing mesh elements on the domain, balancing computational speed and accuracy. Classical AMR depends on heuristics or expensive error estimators, which may lead to suboptimal performance for complex simulations. While AMR methods based on machine learning are promising, they currently only scale to simple problems. In this work, we formulate AMR as a system of collaborating, homogeneous agents that iteratively split into multiple new agents. This agent-wise perspective enables a spatial reward formulation focused on reducing the maximum mesh element error. Our approach, Adaptive Swarm Mesh Refinement++ (ASMR++), offers efficient, stable optimization and generates highly adaptive meshes at user-defined resolution at inference time. Extensive experiments demonstrate that ASMR++ outperforms heuristic approaches and learned baselines, matching the performance of expensive error-based oracle AMR strategies. ASMR additionally generalizes to different domains during inference, and produces meshes that simulate up to 2 orders of magnitude faster than uniform refinements in more demanding settings.

**arXiv ID:** 2406.08440
</details>

<details>
<summary><strong>SOUP: Token-level Single-sample Mix-policy Reinforcement Learning for Large Language Models</strong> - Lei Yang, Wei Bi, Chenxi Sun, Renren Jin, Deyi Xiong - [[pdf]](https://arxiv.org/pdf/2601.21476)</summary>

**Abstract:** On-policy reinforcement learning (RL) methods widely used for language model post-training, like Group Relative Policy Optimization (GRPO), often suffer from limited exploration and early saturation due to low sampling diversity. While off-policy data can help, current approaches that mix entire trajectories cause significant policy mismatch and instability. In this work, we propose the $\textbf{S}$ingle-sample Mix-p$\textbf{O}$licy $\textbf{U}$nified $\textbf{P}$aradigm (SOUP), a framework that unifies off- and on-policy learning within individual samples at the token level. It confines off-policy influence to the prefix of a generated sequence sampled from historical policies, while the continuation is generated on-policy. Through token-level importance ratios, SOUP effectively leverages off-policy information while preserving training stability. Extensive experiments demonstrate that SOUP consistently outperforms standard on-policy training and existing off-policy extensions. Our further analysis clarifies how our fine-grained, single-sample mix-policy training can improve both exploration and final performance in LLM RL.

**arXiv ID:** 2601.21476
</details>

<details>
<summary><strong>Distribution-Aware Reward Estimation for Test-Time Reinforcement Learning</strong> - Bodong Du, Xuanqi Huang, Xiaomeng Li - [[pdf]](https://arxiv.org/pdf/2601.21804)</summary>

**Abstract:** Test-time reinforcement learning (TTRL) enables large language models (LLMs) to self-improve on unlabeled inputs, but its effectiveness critically depends on how reward signals are estimated without ground-truth supervision. Most existing TTRL methods rely on majority voting (MV) over rollouts to produce deterministic rewards, implicitly assuming that the majority rollout provides a reliable learning signal. We show that this assumption is fragile: MV reduces the rollout distribution into a single outcome, discarding information about non-majority but correct actions candidates, and yields systematically biased reward estimates. To address this, we propose Distribution-AwareReward Estimation (DARE), which shifts reward estimation from a single majority outcome to the full empirical rollout distribution. DARE further augments this distribution-based reward with an exploration bonus and a distribution pruning mechanism for non-majority rollout exploration and reward denoise, yielding a more informative and robust reward estimation. Extensive experiments on challenging reasoning benchmarks show that DARE improves optimization stability and final performance over recent baselines, achieving relative improvements of 25.3% on challenging AIME 2024 and 5.3% on AMC.

**arXiv ID:** 2601.21804
</details>

<details>
<summary><strong>DynaWeb: Model-Based Reinforcement Learning of Web Agents</strong> - Hang Ding, Peidong Liu, Junqiao Wang, Ziwei Ji, Meng Cao, Rongzhao Zhang, Lynn Ai, Eric Yang, Tianyu Shi, Lei Yu - [[pdf]](https://arxiv.org/pdf/2601.22149)</summary>

**Abstract:** The development of autonomous web agents, powered by Large Language Models (LLMs) and reinforcement learning (RL), represents a significant step towards general-purpose AI assistants. However, training these agents is severely hampered by the challenges of interacting with the live internet, which is inefficient, costly, and fraught with risks. Model-based reinforcement learning (MBRL) offers a promising solution by learning a world model of the environment to enable simulated interaction. This paper introduces DynaWeb, a novel MBRL framework that trains web agents through interacting with a web world model trained to predict naturalistic web page representations given agent actions. This model serves as a synthetic web environment where an agent policy can dream by generating vast quantities of rollout action trajectories for efficient online reinforcement learning. Beyond free policy rollouts, DynaWeb incorporates real expert trajectories from training data, which are randomly interleaved with on-policy rollouts during training to improve stability and sample efficiency. Experiments conducted on the challenging WebArena and WebVoyager benchmarks demonstrate that DynaWeb consistently and significantly improves the performance of state-of-the-art open-source web agent models. Our findings establish the viability of training web agents through imagination, offering a scalable and efficient way to scale up online agentic RL.

**arXiv ID:** 2601.22149
</details>

<details>
<summary><strong>Towards Transparent RAG: Fostering Evidence Traceability in LLM Generation via Reinforcement Learning</strong> - Jingyi Ren, Yekun Xu, Xiaolong Wang, Weitao Li, Ante Wang, Weizhi Ma, Yang Liu - [[pdf]](https://arxiv.org/pdf/2505.13258)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) delivers substantial value in knowledge-intensive applications. However, its generated responses often lack transparent reasoning paths that trace back to source evidence from retrieved documents. This opacity not only compromises the interpretability of the output but also limits the model's ability to fully exploit the provided context. To address this, we propose TRACE (Transparent RAG with evidenCE tracing), a framework designed to enhance evidence traceability in Large Language Models (LLMs) through reinforcement learning (RL). TRACE guides LLMs to produce structured outputs with explicit evidence citations by prompting and rewarding evidence relevance and proper formatting, alongside accuracy, to optimize structured traceability. To ensure training stability with multiple reward signals, we further introduce an adaptive strategy for merging rewards and adopt a stabilized KL-divergence estimator. Experiments on three multi-hop QA datasets using Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct show that TRACE achieves both transparent, evidence-attributed outputs and accuracy improvements of 10-30%. The resulting performance is comparable to advanced commercial LLMs (e.g., OpenAI o1, DeepSeek-R1). Further analyses demonstrate strong generalization capabilities to unseen tasks. Our code is publicly available now.

**arXiv ID:** 2505.13258
</details>

<details>
<summary><strong>Surrogate Signals from Format and Length: Reinforcement Learning for Solving Mathematical Problems without Ground Truth Answers</strong> - Rihui Xin, Han Liu, Zecheng Wang, Yupeng Zhang, Dianbo Sui, Xiaolin Hu, Bingning Wang - [[pdf]](https://arxiv.org/pdf/2505.19439)</summary>

**Abstract:** Large Language Models have achieved remarkable success in natural language processing tasks, with Reinforcement Learning playing a key role in adapting them to specific applications. However, obtaining ground truth answers for training LLMs in mathematical problem-solving is often challenging, costly, and sometimes unfeasible. This research delves into the utilization of format and length as surrogate signals to train LLMs for mathematical problem-solving, bypassing the need for traditional ground truth answers. Our study shows that a reward function centered on format correctness alone can yield performance improvements comparable to the standard GRPO algorithm in early phases. Recognizing the limitations of format-only rewards in the later phases, we incorporate length-based rewards. The resulting GRPO approach, leveraging format-length surrogate signals, not only matches but surpasses the performance of the standard GRPO algorithm relying on ground truth answers in certain scenarios, achieving 40.0% accuracy on AIME2024 with a 7B base model. Through systematic exploration and experimentation, this research not only offers a practical solution for training LLMs to solve mathematical problems and reducing the dependence on extensive ground truth data collection, but also reveals the essence of why our label-free approach succeeds: the powerful base model is like an excellent student who has already mastered mathematical and logical reasoning skills, but performs poorly on the test paper, it simply needs to develop good answering habits to achieve outstanding results in exams, to unlock the capabilities it already possesses.

**arXiv ID:** 2505.19439
</details>

<details>
<summary><strong>Unit-Based Agent for Semi-Cascaded Full-Duplex Dialogue Systems</strong> - Haoyuan Yu, Yuxuan Chen, Minjie Cai - [[pdf]](https://arxiv.org/pdf/2601.20230)</summary>

**Abstract:** Full-duplex voice interaction is crucial for natural human computer interaction. We present a framework that decomposes complex dialogue into minimal conversational units, enabling the system to process each unit independently and predict when to transit to the next. This framework is instantiated as a semi-cascaded full-duplex dialogue system built around a multimodal large language model, supported by auxiliary modules such as voice activity detection (VAD) and text-to-speech (TTS) synthesis. The resulting system operates in a train-free, plug-and-play manner. Experiments on the HumDial dataset demonstrate the effectiveness of our framework, which ranks second among all teams on the test set of the Human-like Spoken Dialogue Systems Challenge (Track 2: Full-Duplex Interaction). Code is available at the GitHub repository this https URL.

**arXiv ID:** 2601.20230
</details>

<details>
<summary><strong>Rec-R1: Bridging Generative Large Language Models and User-Centric Recommendation Systems via Reinforcement Learning</strong> - Jiacheng Lin, Tian Wang, Kun Qian - [[pdf]](https://arxiv.org/pdf/2503.24289)</summary>

**Abstract:** We propose Rec-R1, a general reinforcement learning framework that bridges large language models (LLMs) with recommendation systems through closed-loop optimization. Unlike prompting and supervised fine-tuning (SFT), Rec-R1 directly optimizes LLM generation using feedback from a fixed black-box recommendation model, without relying on synthetic SFT data from proprietary models such as GPT-4o. This avoids the substantial cost and effort required for data distillation. To verify the effectiveness of Rec-R1, we evaluate it on two representative tasks: product search and sequential recommendation. Experimental results demonstrate that Rec-R1 not only consistently outperforms prompting- and SFT-based methods, but also achieves significant gains over strong discriminative baselines, even when used with simple retrievers such as BM25. Moreover, Rec-R1 preserves the general-purpose capabilities of the LLM, unlike SFT, which often impairs instruction-following and reasoning. These findings suggest Rec-R1 as a promising foundation for continual task-specific adaptation without catastrophic forgetting.

**arXiv ID:** 2503.24289
</details>

<details>
<summary><strong>EndoAgent: A Memory-Guided Reflective Agent for Intelligent Endoscopic Vision-to-Decision Reasoning</strong> - Yi Tang, Kaini Wang, Yang Chen, Guangquan Zhou - [[pdf]](https://arxiv.org/pdf/2508.07292)</summary>

**Abstract:** Developing general artificial intelligence (AI) systems to support endoscopic image diagnosis is an emerging research priority. Existing methods based on large-scale pretraining often lack unified coordination across tasks and struggle to handle the multi-step processes required in complex clinical workflows. While AI agents have shown promise in flexible instruction parsing and tool integration across domains, their potential in endoscopy remains underexplored. To address this gap, we propose EndoAgent, the first memory-guided agent for vision-to-decision endoscopic analysis that integrates iterative reasoning with adaptive tool selection and collaboration. Built on a dual-memory design, it enables sophisticated decision-making by ensuring logical coherence through short-term action tracking and progressively enhancing reasoning acuity through long-term experiential learning. To support diverse clinical tasks, EndoAgent integrates a suite of expert-designed tools within a unified reasoning loop. We further introduce EndoAgentBench, a benchmark of 5,709 visual question-answer pairs that assess visual understanding and language generation capabilities in realistic scenarios. Extensive experiments show that EndoAgent consistently outperforms both general and medical multimodal models, exhibiting its strong flexibility and reasoning capabilities.

**arXiv ID:** 2508.07292
</details>

<details>
<summary><strong>Dr. Bench: A Multidimensional Evaluation for Deep Research Agents, from Answers to Reports</strong> - Yang Yao, Yixu Wang, Yuxuan Zhang, Yi Lu, Tianle Gu, Lingyu Li, Dingyi Zhao, Keming Wu, Haozhe Wang, Ping Nie, Yan Teng, Yingchun Wang - [[pdf]](https://arxiv.org/pdf/2510.02190)</summary>

**Abstract:** As an embodiment of intelligence evolution toward interconnected architectures, Deep Research Agents (DRAs) systematically exhibit the capabilities in task decomposition, cross-source retrieval, multi-stage reasoning, information integration, and structured output, which markedly enhance performance on complex and open-ended tasks. However, existing benchmarks remain deficient in evaluation dimensions, response format, and scoring mechanisms, limiting their effectiveness in assessing such agents. This paper introduces Dr. Bench, a multidimensional evaluation framework tailored to DRAs and long-form report-style responses. The benchmark comprises 214 expert-curated challenging tasks across 10 broad domains, each accompanied by manually constructed reference bundles to support composite evaluation. This framework incorporates metrics for semantic quality, topical focus, and retrieval trustworthiness, enabling a comprehensive evaluation of long reports generated by DRAs. Extensive experimentation confirms the superior performance of mainstream DRAs over web-search-tool-augmented reasoning models, yet reveals considerable scope for further improvement. This study provides a robust foundation for capability assessment, architectural refinement, and paradigm advancement of DRAs.

**arXiv ID:** 2510.02190
</details>

<details>
<summary><strong>ALRM: Agentic LLM for Robotic Manipulation</strong> - Vitor Gaboardi dos Santos, Ibrahim Khadraoui, Ibrahim Farhat, Hamza Yous, Samy Teffahi, Hakim Hacid - [[pdf]](https://arxiv.org/pdf/2601.19510)</summary>

**Abstract:** Large Language Models (LLMs) have recently empowered agentic frameworks to exhibit advanced reasoning and planning capabilities. However, their integration in robotic control pipelines remains limited in two aspects: (1) prior \ac{llm}-based approaches often lack modular, agentic execution mechanisms, limiting their ability to plan, reflect on outcomes, and revise actions in a closed-loop manner; and (2) existing benchmarks for manipulation tasks focus on low-level control and do not systematically evaluate multistep reasoning and linguistic variation. In this paper, we propose Agentic LLM for Robot Manipulation (ALRM), an LLM-driven agentic framework for robotic manipulation. ALRM integrates policy generation with agentic execution through a ReAct-style reasoning loop, supporting two complementary modes: Code-asPolicy (CaP) for direct executable control code generation, and Tool-as-Policy (TaP) for iterative planning and tool-based action execution. To enable systematic evaluation, we also introduce a novel simulation benchmark comprising 56 tasks across multiple environments, capturing linguistically diverse instructions. Experiments with ten LLMs demonstrate that ALRM provides a scalable, interpretable, and modular approach for bridging natural language reasoning with reliable robotic execution. Results reveal Claude-4.1-Opus as the top closed-source model and Falcon-H1-7B as the top open-source model under CaP.

**arXiv ID:** 2601.19510
</details>

<details>
<summary><strong>Few-Shot Learning for Dynamic Operations of Automated Electric Taxi Fleets under Evolving Charging Infrastructure: A Meta-Deep Reinforcement Learning Approach</strong> - Xiaozhuang Li, Xindi Tang, Fang He - [[pdf]](https://arxiv.org/pdf/2601.21312)</summary>

**Abstract:** With the rapid expansion of electric vehicles (EVs) and charging infrastructure, the effective management of Autonomous Electric Taxi (AET) fleets faces a critical challenge in environments with dynamic and uncertain charging availability. While most existing research assumes a static charging network, this simplification creates a significant gap between theoretical models and real-world operations. To bridge this gap, we propose GAT-PEARL, a novel meta-reinforcement learning framework that learns an adaptive operational policy. Our approach integrates a graph attention network (GAT) to effectively extract robust spatial representations under infrastructure layouts and model the complex spatiotemporal relationships of the urban environment, and employs probabilistic embeddings for actor-critic reinforcement learning (PEARL) to enable rapid, inference-based adaptation to changes in charging network layouts without retraining. Through extensive simulations on real-world data in Chengdu, China, we demonstrate that GAT-PEARL significantly outperforms conventional reinforcement learning baselines, showing superior generalization to unseen infrastructure layouts and achieving higher overall operational efficiency in dynamic settings.

**arXiv ID:** 2601.21312
</details>

<details>
<summary><strong>Constrained Meta Reinforcement Learning with Provable Test-Time Safety</strong> - Tingting Ni, Maryam Kamgarpour - [[pdf]](https://arxiv.org/pdf/2601.21845)</summary>

**Abstract:** Meta reinforcement learning (RL) allows agents to leverage experience across a distribution of tasks on which the agent can train at will, enabling faster learning of optimal policies on new test tasks. Despite its success in improving sample complexity on test tasks, many real-world applications, such as robotics and healthcare, impose safety constraints during testing. Constrained meta RL provides a promising framework for integrating safety into meta RL. An open question in constrained meta RL is how to ensure the safety of the policy on the real-world test task, while reducing the sample complexity and thus, enabling faster learning of optimal policies. To address this gap, we propose an algorithm that refines policies learned during training, with provable safety and sample complexity guarantees for learning a near optimal policy on the test tasks. We further derive a matching lower bound, showing that this sample complexity is tight.

**arXiv ID:** 2601.21845
</details>

<details>
<summary><strong>Nimbus: A Unified Embodied Synthetic Data Generation Framework</strong> - Zeyu He, Yuchang Zhang, Yuanzhen Zhou, Miao Tao, Hengjie Li, Yang Tian, Jia Zeng, Tai Wang, Wenzhe Cai, Yilun Chen, Ning Gao, Jiangmiao Pang - [[pdf]](https://arxiv.org/pdf/2601.21449)</summary>

**Abstract:** Scaling data volume and diversity is critical for generalizing embodied intelligence. While synthetic data generation offers a scalable alternative to expensive physical data acquisition, existing pipelines remain fragmented and task-specific. This isolation leads to significant engineering inefficiency and system instability, failing to support the sustained, high-throughput data generation required for foundation model training. To address these challenges, we present Nimbus, a unified synthetic data generation framework designed to integrate heterogeneous navigation and manipulation pipelines. Nimbus introduces a modular four-layer architecture featuring a decoupled execution model that separates trajectory planning, rendering, and storage into asynchronous stages. By implementing dynamic pipeline scheduling, global load balancing, distributed fault tolerance, and backend-specific rendering optimizations, the system maximizes resource utilization across CPU, GPU, and I/O resources. Our evaluation demonstrates that Nimbus achieves a 2-3X improvement in end-to-end throughput compared to unoptimized baselines and ensuring robust, long-term operation in large-scale distributed environments. This framework serves as the production backbone for the InternData suite, enabling seamless cross-domain data synthesis.

**arXiv ID:** 2601.21449
</details>

<details>
<summary><strong>FLARE: Agile Flights for Quadrotor Cable-Suspended Payload System via Reinforcement Learning</strong> - Dongcheng Cao, Jin Zhou, Xian Wang, Shuo Li - [[pdf]](https://arxiv.org/pdf/2508.09797)</summary>

**Abstract:** Agile flight for the quadrotor cable-suspended payload system is a formidable challenge due to its underactuated, highly nonlinear, and hybrid dynamics. Traditional optimization-based methods often struggle with high computational costs and the complexities of cable mode transitions, limiting their real-time applicability and maneuverability exploitation. In this letter, we present FLARE, a reinforcement learning (RL) framework that directly learns agile navigation policy from high-fidelity simulation. Our method is validated across three designed challenging scenarios, notably outperforming a state-of-the-art optimization-based approach by a 3x speedup during gate traversal maneuvers. Furthermore, the learned policies achieve successful zero-shot sim-to-real transfer, demonstrating remarkable agility and safety in real-world experiments, running in real time on an onboard computer.

**arXiv ID:** 2508.09797
</details>

<details>
<summary><strong>When Your Boss Is an AI Bot: Exploring Opportunities and Risks of Manager Clone Agents in the Future Workplace</strong> - Qing Hu, Qing Xiao, Hancheng Cao, Hong Shen - [[pdf]](https://arxiv.org/pdf/2509.10993)</summary>

**Abstract:** As Generative AI (GenAI) becomes increasingly embedded in the workplace, managers are beginning to create Manager Clone Agents -- AI-powered digital surrogates trained on their work communications and decision patterns to perform managerial tasks on their behalf. To investigate this emerging phenomenon, we conducted six design fiction workshops (n = 23) with managers and workers, in which participants co-created speculative scenarios and discussed how Manager Clone Agents might transform collaborative work. We identified four potential roles that participants envisioned for Manager Clone Agents: proxy presence, informational conveyor, productivity engine, and leadership amplifier, while highlighting concerns spanning individual, interpersonal, and organizational levels. We provide design recommendations envisioned by both parties for integrating Manager Clone Agents responsibly into the future workplace, emphasizing the need to prioritize workers' perspectives and nurture interpersonal bonds while also anticipating alternative futures that may disrupt managerial hierarchies.

**arXiv ID:** 2509.10993
</details>

<details>
<summary><strong>Normative Equivalence in Human-AI Cooperation: Behaviour, Not Identity, Drives Cooperation in Mixed-Agent Groups</strong> - Nico Mutzner, Taha Yasseri, Heiko Rauhut - [[pdf]](https://arxiv.org/pdf/2601.20487)</summary>

**Abstract:** The introduction of artificial intelligence (AI) agents into human group settings raises essential questions about how these novel participants influence cooperative social norms. While previous studies on human-AI cooperation have primarily focused on dyadic interactions, little is known about how integrating AI agents affects the emergence and maintenance of cooperative norms in small groups. This study addresses this gap through an online experiment using a repeated four-player Public Goods Game (PGG). Each group consisted of three human participants and one bot, which was framed either as human or AI and followed one of three predefined decision strategies: unconditional cooperation, conditional cooperation, or free-riding. In our sample of 236 participants, we found that reciprocal group dynamics and behavioural inertia primarily drove cooperation. These normative mechanisms operated identically across conditions, resulting in cooperation levels that did not differ significantly between human and AI labels. Furthermore, we found no evidence of differences in norm persistence in a follow-up Prisoner's Dilemma, or in participants' normative perceptions. Participants' behaviour followed the same normative logic across human and AI conditions, indicating that cooperation depended on group behaviour rather than partner identity. This supports a pattern of normative equivalence, in which the mechanisms that sustain cooperation function similarly in mixed human-AI and all human groups. These findings suggest that cooperative norms are flexible enough to extend to artificial agents, blurring the boundary between humans and AI in collective decision-making.

**arXiv ID:** 2601.20487
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
