# Agent arXiv Daily

**Last Updated:** 2025-11-17 02:12:49

**Total Papers:** 47

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
<summary><strong>AgentEvolver: Towards Efficient Self-Evolving Agent System</strong> - Yunpeng Zhai, Shuchang Tao, Cheng Chen, Anni Zou, Ziqian Chen, Qingxu Fu, Shinji Mai, Li Yu, Jiaji Deng, Zouying Cao, Zhaoyang Liu, Bolin Ding, Jingren Zhou - [[pdf]](https://arxiv.org/pdf/2511.10395)</summary>

**Abstract:** Autonomous agents powered by large language models (LLMs) have the potential to significantly enhance human productivity by reasoning, using tools, and executing complex tasks in diverse environments. However, current approaches to developing such agents remain costly and inefficient, as they typically require manually constructed task datasets and reinforcement learning (RL) pipelines with extensive random exploration. These limitations lead to prohibitively high data-construction costs, low exploration efficiency, and poor sample utilization. To address these challenges, we present AgentEvolver, a self-evolving agent system that leverages the semantic understanding and reasoning capabilities of LLMs to drive autonomous agent learning. AgentEvolver introduces three synergistic mechanisms: (i) self-questioning, which enables curiosity-driven task generation in novel environments, reducing dependence on handcrafted datasets; (ii) self-navigating, which improves exploration efficiency through experience reuse and hybrid policy guidance; and (iii) self-attributing, which enhances sample efficiency by assigning differentiated rewards to trajectory states and actions based on their contribution. By integrating these mechanisms into a unified framework, AgentEvolver enables scalable, cost-effective, and continual improvement of agent capabilities. Preliminary experiments indicate that AgentEvolver achieves more efficient exploration, better sample utilization, and faster adaptation compared to traditional RL-based baselines.

**arXiv ID:** 2511.10395
</details>

<details>
<summary><strong>Depth Matters: Multimodal RGB-D Perception for Robust Autonomous Agents</strong> - Mihaela-Larisa Clement, Mónika Farsang, Felix Resch, Mihai-Teodor Stanusoiu, Radu Grosu - [[pdf]](https://arxiv.org/pdf/2503.16711)</summary>

**Abstract:** Autonomous agents that rely purely on perception to make real-time control decisions require efficient and robust architectures. In this work, we demonstrate that augmenting RGB input with depth information significantly enhances our agents' ability to predict steering commands compared to using RGB alone. We benchmark lightweight recurrent controllers that leverage the fused RGB-D features for sequential decision-making. To train our models, we collect high-quality data using a small-scale autonomous car controlled by an expert driver via a physical steering wheel, capturing varying levels of steering difficulty. Our models were successfully deployed on real hardware and inherently avoided dynamic and static obstacles, under out-of-distribution conditions. Specifically, our findings reveal that the early fusion of depth data results in a highly robust controller, which remains effective even with frame drops and increased noise levels, without compromising the network's focus on the task.

**arXiv ID:** 2503.16711
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (2 papers)</h2></summary>

<details>
<summary><strong>SynthTools: A Framework for Scaling Synthetic Tools for Agent Development</strong> - Tommaso Castellani, Naimeng Ye, Daksh Mittal, Thomson Yen, Hongseok Namkoong - [[pdf]](https://arxiv.org/pdf/2511.09572)</summary>

**Abstract:** AI agents increasingly rely on external tools to solve complex, long-horizon tasks. Advancing such agents requires reproducible evaluation and large-scale training in controllable, diverse, and realistic tool-use environments. However, real-world APIs are limited in availability, domain coverage, and stability, often requiring access keys and imposing rate limits, which render them impractical for stable evaluation or scalable training. To address these challenges, we introduce SynthTools, a flexible and scalable framework for generating synthetic tool ecosystems. Our framework consists of three core components: Tool Generation for automatic and scalable creation of diverse tools, Tool Simulation to emulate realistic tool behaviors, and Tool Audit to ensure correctness and consistency of tool simulation. To illustrate its scalability, we show that SynthTools can readily produce toolsets that span twice as many domains and twice as many tools per domain as prior work. Furthermore, the tool simulation and tool audit components demonstrate strong reliability, achieving $94\%$ and $99\%$ accuracy respectively. Finally, we construct downstream tasks from the generated tools that even state-of-the-art models struggle to complete. By enabling scalable, diverse, and reliable tool ecosystems, SynthTools provides a practical path toward large-scale training and stable evaluation of tool-use agents. Our code is available at this https URL.

**arXiv ID:** 2511.09572
</details>

<details>
<summary><strong>Autonomous Concept Drift Threshold Determination</strong> - Pengqian Lu, Jie Lu, Anjin Liu, En Yu, Guangquan Zhang - [[pdf]](https://arxiv.org/pdf/2511.09953)</summary>

**Abstract:** Existing drift detection methods focus on designing sensitive test statistics. They treat the detection threshold as a fixed hyperparameter, set once to balance false alarms and late detections, and applied uniformly across all datasets and over time. However, maintaining model performance is the key objective from the perspective of machine learning, and we observe that model performance is highly sensitive to this threshold. This observation inspires us to investigate whether a dynamic threshold could be provably better. In this paper, we prove that a threshold that adapts over time can outperform any single fixed threshold. The main idea of the proof is that a dynamic strategy, constructed by combining the best threshold from each individual data segment, is guaranteed to outperform any single threshold that apply to all segments. Based on the theorem, we propose a Dynamic Threshold Determination algorithm. It enhances existing drift detection frameworks with a novel comparison phase to inform how the threshold should be adjusted. Extensive experiments on a wide range of synthetic and real-world datasets, including both image and tabular data, validate that our approach substantially enhances the performance of state-of-the-art drift detectors.

**arXiv ID:** 2511.09953
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>Echoing: Identity Failures when LLM Agents Talk to Each Other</strong> - Sarath Shekkizhar, Romain Cosentino, Adam Earle, Silvio Savarese - [[pdf]](https://arxiv.org/pdf/2511.09710)</summary>

**Abstract:** As large language model (LLM) based agents interact autonomously with one another, a new class of failures emerges that cannot be predicted from single agent performance: behavioral drifts in agent-agent conversations (AxA). Unlike human-agent interactions, where humans ground and steer conversations, AxA lacks such stabilizing signals, making these failures unique. We investigate one such failure, echoing, where agents abandon their assigned roles and instead mirror their conversational partners, undermining their intended objectives. Through experiments across $60$ AxA configurations, $3$ domains, and $2000+$ conversations, we demonstrate that echoing occurs across three major LLM providers, with echoing rates from $5\%$ to $70\%$ depending on the model and domain. Moreover, we find that echoing is persistent even in advanced reasoning models with substantial rates ($32.8\%$) that are not reduced by increased reasoning efforts. We analyze prompt impacts, conversation dynamics, showing that echoing arises as interaction grows longer ($7+$ turns in experiments) and is not merely an artifact of sub-optimal prompting. Finally, we introduce a protocol-level mitigation in which targeted use of structured responses reduces echoing to $9\%$.

**arXiv ID:** 2511.09710
</details>

<details>
<summary><strong>Scaling Environments for LLM Agents in the Era of Learning from Interaction: A Survey</strong> - Yuchen Huang, Sijia Li, Minghao Liu, Wei Liu, Shijue Huang, Zhiyuan Fan, Hou Pong Chan, Yi R. Fung - [[pdf]](https://arxiv.org/pdf/2511.09586)</summary>

**Abstract:** LLM-based agents can autonomously accomplish complex tasks across various domains. However, to further cultivate capabilities such as adaptive behavior and long-term decision-making, training on static datasets built from human-level knowledge is insufficient. These datasets are costly to construct and lack both dynamism and realism. A growing consensus is that agents should instead interact directly with environments and learn from experience through reinforcement learning. We formalize this iterative process as the Generation-Execution-Feedback (GEF) loop, where environments generate tasks to challenge agents, return observations in response to agents' actions during task execution, and provide evaluative feedback on rollouts for subsequent learning. Under this paradigm, environments function as indispensable producers of experiential data, highlighting the need to scale them toward greater complexity, realism, and interactivity. In this survey, we systematically review representative methods for environment scaling from a pioneering environment-centric perspective and organize them along the stages of the GEF loop, namely task generation, task execution, and feedback. We further analyze benchmarks, implementation strategies, and applications, consolidating fragmented advances and outlining future research directions for agent intelligence.

**arXiv ID:** 2511.09586
</details>

<details>
<summary><strong>Towards an Agentic Workflow for Internet Measurement Research</strong> - Alagappan Ramanathan, Eunju Kang, Dongsu Han, Sangeetha Abdu Jyothi - [[pdf]](https://arxiv.org/pdf/2511.10611)</summary>

**Abstract:** Internet measurement research faces an accessibility crisis: complex analyses require custom integration of multiple specialized tools that demands specialized domain expertise. When network disruptions occur, operators need rapid diagnostic workflows spanning infrastructure mapping, routing analysis, and dependency modeling. However, developing these workflows requires specialized knowledge and significant manual effort.
We present ArachNet, the first system demonstrating that LLM agents can independently generate measurement workflows that mimics expert reasoning. Our core insight is that measurement expertise follows predictable compositional patterns that can be systematically automated. ArachNet operates through four specialized agents that mirror expert workflow, from problem decomposition to solution implementation. We validate ArachNet with progressively challenging Internet resilience scenarios. The system independently generates workflows that match expert-level reasoning and produce analytical outputs similar to specialist solutions. Generated workflows handle complex multi-framework integration that traditionally requires days of manual coordination. ArachNet lowers barriers to measurement workflow composition by automating the systematic reasoning process that experts use, enabling broader access to sophisticated measurement capabilities while maintaining the technical rigor required for research-quality analysis.

**arXiv ID:** 2511.10611
</details>

<details>
<summary><strong>FHIR-AgentBench: Benchmarking LLM Agents for Realistic Interoperable EHR Question Answering</strong> - Gyubok Lee, Elea Bach, Eric Yang, Tom Pollard, Alistair Johnson, Edward Choi, Yugang jia, Jong Ha Lee - [[pdf]](https://arxiv.org/pdf/2509.19319)</summary>

**Abstract:** The recent shift toward the Health Level Seven Fast Healthcare Interoperability Resources (HL7 FHIR) standard opens a new frontier for clinical AI, demanding LLM agents to navigate complex, resource-based data models instead of conventional structured health data. However, existing benchmarks have lagged behind this transition, lacking the realism needed to evaluate recent LLMs on interoperable clinical data. To bridge this gap, we introduce FHIR-AgentBench, a benchmark that grounds 2,931 real-world clinical questions in the HL7 FHIR standard. Using this benchmark, we systematically evaluate agentic frameworks, comparing different data retrieval strategies (direct FHIR API calls vs. specialized tools), interaction patterns (single-turn vs. multi-turn), and reasoning strategies (natural language vs. code generation). Our experiments highlight the practical challenges of retrieving data from intricate FHIR resources and the difficulty of reasoning over them, both of which critically affect question answering performance. We publicly release the FHIR-AgentBench dataset and evaluation suite (this https URL) to promote reproducible research and the development of robust, reliable LLM agents for clinical applications.

**arXiv ID:** 2509.19319
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (14 papers)</h2></summary>

<details>
<summary><strong>SlideBot: A Multi-Agent Framework for Generating Informative, Reliable, Multi-Modal Presentations</strong> - Eric Xie, Danielle Waterfield, Michael Kennedy, Aidong Zhang - [[pdf]](https://arxiv.org/pdf/2511.09804)</summary>

**Abstract:** Large Language Models (LLMs) have shown immense potential in education, automating tasks like quiz generation and content summarization. However, generating effective presentation slides introduces unique challenges due to the complexity of multimodal content creation and the need for precise, domain-specific information. Existing LLM-based solutions often fail to produce reliable and informative outputs, limiting their educational value. To address these limitations, we introduce SlideBot - a modular, multi-agent slide generation framework that integrates LLMs with retrieval, structured planning, and code generation. SlideBot is organized around three pillars: informativeness, ensuring deep and contextually grounded content; reliability, achieved by incorporating external sources through retrieval; and practicality, which enables customization and iterative feedback through instructor collaboration. It incorporates evidence-based instructional design principles from Cognitive Load Theory (CLT) and the Cognitive Theory of Multimedia Learning (CTML), using structured planning to manage intrinsic load and consistent visual macros to reduce extraneous load and enhance dual-channel learning. Within the system, specialized agents collaboratively retrieve information, summarize content, generate figures, and format slides using LaTeX, aligning outputs with instructor preferences through interactive refinement. Evaluations from domain experts and students in AI and biomedical education show that SlideBot consistently enhances conceptual accuracy, clarity, and instructional value. These findings demonstrate SlideBot's potential to streamline slide preparation while ensuring accuracy, relevance, and adaptability in higher education.

**arXiv ID:** 2511.09804
</details>

<details>
<summary><strong>Explaining Decentralized Multi-Agent Reinforcement Learning Policies</strong> - Kayla Boggess, Sarit Kraus, Lu Feng - [[pdf]](https://arxiv.org/pdf/2511.10409)</summary>

**Abstract:** Multi-Agent Reinforcement Learning (MARL) has gained significant interest in recent years, enabling sequential decision-making across multiple agents in various domains. However, most existing explanation methods focus on centralized MARL, failing to address the uncertainty and nondeterminism inherent in decentralized settings. We propose methods to generate policy summarizations that capture task ordering and agent cooperation in decentralized MARL policies, along with query-based explanations for When, Why Not, and What types of user queries about specific agent behaviors. We evaluate our approach across four MARL domains and two decentralized MARL algorithms, demonstrating its generalizability and computational efficiency. User studies show that our summarizations and explanations significantly improve user question-answering performance and enhance subjective ratings on metrics such as understanding and satisfaction.

**arXiv ID:** 2511.10409
</details>

<details>
<summary><strong>VISTA: A Vision and Intent-Aware Social Attention Framework for Multi-Agent Trajectory Prediction</strong> - Stephane Da Silva Martins, Emanuel Aldea, Sylvie Le Hégarat-Mascle - [[pdf]](https://arxiv.org/pdf/2511.10203)</summary>

**Abstract:** Multi-agent trajectory prediction is crucial for autonomous systems operating in dense, interactive environments. Existing methods often fail to jointly capture agents' long-term goals and their fine-grained social interactions, which leads to unrealistic multi-agent futures. We propose VISTA, a recursive goal-conditioned transformer for multi-agent trajectory forecasting. VISTA combines (i) a cross-attention fusion module that integrates long-horizon intent with past motion, (ii) a social-token attention mechanism for flexible interaction modeling across agents, and (iii) pairwise attention maps that make social influence patterns interpretable at inference time. Our model turns single-agent goal-conditioned prediction into a coherent multi-agent forecasting framework. Beyond standard displacement metrics, we evaluate trajectory collision rates as a measure of joint realism. On the high-density MADRAS benchmark and on SDD, VISTA achieves state-of-the-art accuracy and substantially fewer collisions. On MADRAS, it reduces the average collision rate of strong baselines from 2.14 to 0.03 percent, and on SDD it attains zero collisions while improving ADE, FDE, and minFDE. These results show that VISTA generates socially compliant, goal-aware, and interpretable trajectories, making it promising for safety-critical autonomous systems.

**arXiv ID:** 2511.10203
</details>

<details>
<summary><strong>Rethinking the Reliability of Multi-agent System: A Perspective from Byzantine Fault Tolerance</strong> - Lifan Zheng, Jiawei Chen, Qinghong Yin, Jingyuan Zhang, Xinyi Zeng, Yu Tian - [[pdf]](https://arxiv.org/pdf/2511.10400)</summary>

**Abstract:** Ensuring the reliability of agent architectures and effectively identifying problematic agents when failures occur are crucial challenges in multi-agent systems (MAS). Advances in large language models (LLMs) have established LLM-based agents as a major branch of MAS, enabling major breakthroughs in complex problem solving and world modeling. However, the reliability implications of this shift remain largely unexplored. i.e., whether substituting traditional agents with LLM-based agents can effectively enhance the reliability of MAS. In this work, we investigate and quantify the reliability of LLM-based agents from the perspective of Byzantine fault tolerance. We observe that LLM-based agents demonstrate stronger skepticism when processing erroneous message flows, a characteristic that enables them to outperform traditional agents across different topological structures. Motivated by the results of the pilot experiment, we design CP-WBFT, a confidence probe-based weighted Byzantine Fault Tolerant consensus mechanism to enhance the stability of MAS with different topologies. It capitalizes on the intrinsic reflective and discriminative capabilities of LLMs by employing a probe-based, weighted information flow transmission method to improve the reliability of LLM-based agents. Extensive experiments demonstrate that CP-WBFT achieves superior performance across diverse network topologies under extreme Byzantine conditions (85.7\% fault rate). Notably, our approach surpasses traditional methods by attaining remarkable accuracy on various topologies and maintaining strong reliability in both mathematical reasoning and safety assessment tasks.

**arXiv ID:** 2511.10400
</details>

<details>
<summary><strong>nuPlan-R: A Closed-Loop Planning Benchmark for Autonomous Driving via Reactive Multi-Agent Simulation</strong> - Mingxing Peng, Ruoyu Yao, Xusen Guo, Jun Ma - [[pdf]](https://arxiv.org/pdf/2511.10403)</summary>

**Abstract:** Recent advances in closed-loop planning benchmarks have significantly improved the evaluation of autonomous vehicles. However, existing benchmarks still rely on rule-based reactive agents such as the Intelligent Driver Model (IDM), which lack behavioral diversity and fail to capture realistic human interactions, leading to oversimplified traffic dynamics. To address these limitations, we present nuPlan-R, a new reactive closed-loop planning benchmark that integrates learning-based reactive multi-agent simulation into the nuPlan framework. Our benchmark replaces the rule-based IDM agents with noise-decoupled diffusion-based reactive agents and introduces an interaction-aware agent selection mechanism to ensure both realism and computational efficiency. Furthermore, we extend the benchmark with two additional metrics to enable a more comprehensive assessment of planning performance. Extensive experiments demonstrate that our reactive agent model produces more realistic, diverse, and human-like traffic behaviors, leading to a benchmark environment that better reflects real-world interactive driving. We further reimplement a collection of rule-based, learning-based, and hybrid planning approaches within our nuPlan-R benchmark, providing a clearer reflection of planner performance in complex interactive scenarios and better highlighting the advantages of learning-based planners in handling complex and dynamic scenarios. These results establish nuPlan-R as a new standard for fair, reactive, and realistic closed-loop planning evaluation. We will open-source the code for the new benchmark.

**arXiv ID:** 2511.10403
</details>

<details>
<summary><strong>Ax-Prover: A Deep Reasoning Agentic Framework for Theorem Proving in Mathematics and Quantum Physics</strong> - Benjamin Breen, Marco Del Tredici, Jacob McCarran, Javier Aspuru Mijares, Weichen Winston Yin, Kfir Sulimany, Jacob M. Taylor, Frank H. L. Koppens, Dirk Englund - [[pdf]](https://arxiv.org/pdf/2510.12787)</summary>

**Abstract:** We present Ax-Prover, a multi-agent system for automated theorem proving in Lean that can solve problems across diverse scientific domains and operate either autonomously or collaboratively with human experts. To achieve this, Ax-Prover approaches scientific problem solving through formal proof generation, a process that demands both creative reasoning and strict syntactic rigor. Ax-Prover meets this challenge by equipping Large Language Models (LLMs), which provide knowledge and reasoning, with Lean tools via the Model Context Protocol (MCP), which ensure formal correctness. To evaluate its performance as an autonomous prover, we benchmark our approach against frontier LLMs and specialized prover models on two public math benchmarks and on two Lean benchmarks we introduce in the fields of abstract algebra and quantum theory. On public datasets, Ax-Prover is competitive with state-of-the-art provers, while it largely outperforms them on the new benchmarks. This shows that, unlike specialized systems that struggle to generalize, our tool-based agentic theorem prover approach offers a generalizable methodology for formal verification across diverse scientific domains. Furthermore, we demonstrate Ax-Prover's assistant capabilities in a practical use case, showing how it enabled an expert mathematician to formalize the proof of a complex cryptography theorem.

**arXiv ID:** 2510.12787
</details>

<details>
<summary><strong>A Brain Cell Type Resource Created by Large Language Models and a Multi-Agent AI System for Collaborative Community Annotation</strong> - Rongbin Li, Wenbo Chen, Zhao Li, Rodrigo Munoz-Castaneda, Jinbo Li, Neha S. Maurya, Arnav Solanki, Huan He, Hanwen Xing, Meaghan Ramlakhan, Zachary Wise, Nelson Johansen, Zhuhao Wu, Hua Xu, Michael Hawrylycz, W. Jim Zheng - [[pdf]](https://arxiv.org/pdf/2510.17064)</summary>

**Abstract:** Single-cell RNA sequencing has transformed our ability to identify diverse cell types and their transcriptomic signatures. However, annotating these signatures-especially those involving poorly characterized genes-remains a major challenge. Traditional methods, such as Gene Set Enrichment Analysis (GSEA), depend on well-curated annotations and often perform poorly in these contexts. Large Language Models (LLMs) offer a promising alternative but struggle to represent complex biological knowledge within structured ontologies. To address this, we present BRAINCELL-AID (BRAINCELL-AID: this https URL), a novel multi-agent AI system that integrates free-text descriptions with ontology labels to enable more accurate and robust gene set annotation. By incorporating retrieval-augmented generation (RAG), we developed a robust agentic workflow that refines predictions using relevant PubMed literature, reducing hallucinations and enhancing interpretability. Using this workflow, we achieved correct annotations for 77% of mouse gene sets among their top predictions. Applying this approach, we annotated 5,322 brain cell clusters from the comprehensive mouse brain cell atlas generated by the BRAIN Initiative Cell Census Network, enabling novel insights into brain cell function by identifying region-specific gene co-expression patterns and inferring functional roles of gene ensembles. BRAINCELL-AID also identifies Basal Ganglia-related cell types with neurologically meaningful descriptions. Hence, we create a valuable resource to support community-driven cell type annotation.

**arXiv ID:** 2510.17064
</details>

<details>
<summary><strong>Multi-agent In-context Coordination via Decentralized Memory Retrieval</strong> - Tao Jiang, Zichuan Lin, Lihe Li, Yi-Chen Li, Cong Guan, Lei Yuan, Zongzhang Zhang, Yang Yu, Deheng Ye - [[pdf]](https://arxiv.org/pdf/2511.10030)</summary>

**Abstract:** Large transformer models, trained on diverse datasets, have demonstrated impressive few-shot performance on previously unseen tasks without requiring parameter updates. This capability has also been explored in Reinforcement Learning (RL), where agents interact with the environment to retrieve context and maximize cumulative rewards, showcasing strong adaptability in complex settings. However, in cooperative Multi-Agent Reinforcement Learning (MARL), where agents must coordinate toward a shared goal, decentralized policy deployment can lead to mismatches in task alignment and reward assignment, limiting the efficiency of policy adaptation. To address this challenge, we introduce Multi-agent In-context Coordination via Decentralized Memory Retrieval (MAICC), a novel approach designed to enhance coordination by fast adaptation. Our method involves training a centralized embedding model to capture fine-grained trajectory representations, followed by decentralized models that approximate the centralized one to obtain team-level task information. Based on the learned embeddings, relevant trajectories are retrieved as context, which, combined with the agents' current sub-trajectories, inform decision-making. During decentralized execution, we introduce a novel memory mechanism that effectively balances test-time online data with offline memory. Based on the constructed memory, we propose a hybrid utility score that incorporates both individual- and team-level returns, ensuring credit assignment across agents. Extensive experiments on cooperative MARL benchmarks, including Level-Based Foraging (LBF) and SMAC (v1/v2), show that MAICC enables faster adaptation to unseen tasks compared to existing methods. Code is available at this https URL.

**arXiv ID:** 2511.10030
</details>

<details>
<summary><strong>Behavior Modeling for Training-free Building of Private Domain Multi Agent System</strong> - Won Ik Cho, Woonghee Han, Kyung Seo Ki, Young Min Kim - [[pdf]](https://arxiv.org/pdf/2511.10283)</summary>

**Abstract:** The rise of agentic systems that combine orchestration, tool use, and conversational capabilities, has been more visible by the recent advent of large language models (LLMs). While open-domain frameworks exist, applying them in private domains remains difficult due to heterogeneous tool formats, domain-specific jargon, restricted accessibility of APIs, and complex governance. Conventional solutions, such as fine-tuning on synthetic dialogue data, are burdensome and brittle under domain shifts, and risk degrading general performance. In this light, we introduce a framework for private-domain multi-agent conversational systems that avoids training and data generation by adopting behavior modeling and documentation. Our design simply assumes an orchestrator, a tool-calling agent, and a general chat agent, with tool integration defined through structured specifications and domain-informed instructions. This approach enables scalable adaptation to private tools and evolving contexts without continual retraining. The framework supports practical use cases, including lightweight deployment of multi-agent systems, leveraging API specifications as retrieval resources, and generating synthetic dialogue for evaluation -- providing a sustainable method for aligning agent behavior with domain expertise in private conversational ecosystems.

**arXiv ID:** 2511.10283
</details>

<details>
<summary><strong>Beyond Monotonicity: Revisiting Factorization Principles in Multi-Agent Q-Learning</strong> - Tianmeng Hu, Yongzheng Cui, Rui Tang, Biao Luo, Ke Li - [[pdf]](https://arxiv.org/pdf/2511.09792)</summary>

**Abstract:** Value decomposition is a central approach in multi-agent reinforcement learning (MARL), enabling centralized training with decentralized execution by factorizing the global value function into local values. To ensure individual-global-max (IGM) consistency, existing methods either enforce monotonicity constraints, which limit expressive power, or adopt softer surrogates at the cost of algorithmic complexity. In this work, we present a dynamical systems analysis of non-monotonic value decomposition, modeling learning dynamics as continuous-time gradient flow. We prove that, under approximately greedy exploration, all zero-loss equilibria violating IGM consistency are unstable saddle points, while only IGM-consistent solutions are stable attractors of the learning dynamics. Extensive experiments on both synthetic matrix games and challenging MARL benchmarks demonstrate that unconstrained, non-monotonic factorization reliably recovers IGM-optimal solutions and consistently outperforms monotonic baselines. Additionally, we investigate the influence of temporal-difference targets and exploration strategies, providing actionable insights for the design of future value-based MARL algorithms.

**arXiv ID:** 2511.09792
</details>

<details>
<summary><strong>Transfer in Reinforcement Learning via Regret Bounds for Learning Agents</strong> - Adrienne Tuynman, Ronald Ortner - [[pdf]](https://arxiv.org/pdf/2202.01182)</summary>

**Abstract:** We present an approach for the quantification of the usefulness of transfer in reinforcement learning via regret bounds for a multi-agent setting. Considering a number of $\aleph$ agents operating in the same Markov decision process, however possibly with different reward functions, we consider the regret each agent suffers with respect to an optimal policy maximizing her average reward. We show that when the agents share their observations the total regret of all agents is smaller by a factor of $\sqrt{\aleph}$ compared to the case when each agent has to rely on the information collected by herself. This result demonstrates how considering the regret in multi-agent settings can provide theoretical bounds on the benefit of sharing observations in transfer learning.

**arXiv ID:** 2202.01182
</details>

<details>
<summary><strong>Causal Model-Based Reinforcement Learning for Sample-Efficient IoT Channel Access</strong> - Aswin Arun, Christo Kurisummoottil Thomas, Rimalpudi Sarvendranath, Walid Saad - [[pdf]](https://arxiv.org/pdf/2511.10291)</summary>

**Abstract:** Despite the advantages of multi-agent reinforcement learning (MARL) for wireless use case such as medium access control (MAC), their real-world deployment in Internet of Things (IoT) is hindered by their sample inefficiency. To alleviate this challenge, one can leverage model-based reinforcement learning (MBRL) solutions, however, conventional MBRL approaches rely on black-box models that are not interpretable and cannot reason. In contrast, in this paper, a novel causal model-based MARL framework is developed by leveraging tools from causal learn- ing. In particular, the proposed model can explicitly represent causal dependencies between network variables using structural causal models (SCMs) and attention-based inference networks. Interpretable causal models are then developed to capture how MAC control messages influence observations, how transmission actions determine outcomes, and how channel observations affect rewards. Data augmentation techniques are then used to generate synthetic rollouts using the learned causal model for policy optimization via proximal policy optimization (PPO). Analytical results demonstrate exponential sample complexity gains of causal MBRL over black-box approaches. Extensive simulations demonstrate that, on average, the proposed approach can reduce environment interactions by 58%, and yield faster convergence compared to model-free baselines. The proposed approach inherently is also shown to provide interpretable scheduling decisions via attention-based causal attribution, revealing which network conditions drive the policy. The resulting combination of sample efficiency and interpretability establishes causal MBRL as a practical approach for resource-constrained wireless systems.

**arXiv ID:** 2511.10291
</details>

<details>
<summary><strong>Multi-agent Markov Entanglement</strong> - Shuze Chen, Tianyi Peng - [[pdf]](https://arxiv.org/pdf/2506.02385)</summary>

**Abstract:** Value decomposition has long been a fundamental technique in multi-agent dynamic programming and reinforcement learning (RL). Specifically, the value function of a global state $(s_1,s_2,\ldots,s_N)$ is often approximated as the sum of local functions: $V(s_1,s_2,\ldots,s_N)\approx\sum_{i=1}^N V_i(s_i)$. This approach traces back to the index policy in restless multi-armed bandit problems and has found various applications in modern RL systems. However, the theoretical justification for why this decomposition works so effectively remains underexplored.
In this paper, we uncover the underlying mathematical structure that enables value decomposition. We demonstrate that a multi-agent Markov decision process (MDP) permits value decomposition if and only if its transition matrix is not "entangled" -- a concept analogous to quantum entanglement in quantum physics. Drawing inspiration from how physicists measure quantum entanglement, we introduce how to measure the "Markov entanglement" for multi-agent MDPs and show that this measure can be used to bound the decomposition error in general multi-agent MDPs.
Using the concept of Markov entanglement, we proved that a widely-used class of index policies is weakly entangled and enjoys a sublinear $\mathcal O(\sqrt{N})$ scale of decomposition error for $N$-agent systems. Finally, we show how Markov entanglement can be efficiently estimated in practice, providing practitioners with an empirical proxy for the quality of value decomposition.

**arXiv ID:** 2506.02385
</details>

<details>
<summary><strong>Onboard Mission Replanning for Adaptive Cooperative Multi-Robot Systems</strong> - Elim Kwan, Rehman Qureshi, Liam Fletcher, Colin Laganier, Victoria Nockles, Richard Walters - [[pdf]](https://arxiv.org/pdf/2506.06094)</summary>

**Abstract:** Cooperative autonomous robotic systems have significant potential for executing complex multi-task missions across space, air, ground, and maritime domains. But they commonly operate in remote, dynamic and hazardous environments, requiring rapid in-mission adaptation without reliance on fragile or slow communication links to centralised compute. Fast, on-board replanning algorithms are therefore needed to enhance resilience. Reinforcement Learning shows strong promise for efficiently solving mission planning tasks when formulated as Travelling Salesperson Problems (TSPs), but existing methods: 1) are unsuitable for replanning, where agents do not start at a single location; 2) do not allow cooperation between agents; 3) are unable to model tasks with variable durations; or 4) lack practical considerations for on-board deployment. Here we define the Cooperative Mission Replanning Problem as a novel variant of multiple TSP with adaptations to overcome these issues, and develop a new encoder/decoder-based model using Graph Attention Networks and Attention Models to solve it effectively and efficiently. Using a simple example of cooperative drones, we show our replanner consistently (90% of the time) maintains performance within 10% of the state-of-the-art LKH3 heuristic solver, whilst running 85-370 times faster on a Raspberry Pi. This work paves the way for increased resilience in autonomous multi-agent systems.

**arXiv ID:** 2506.06094
</details>

</details>

<details open>
<summary><h2>Other Agent Research (3 papers)</h2></summary>

<details>
<summary><strong>Balancing Centralized Learning and Distributed Self-Organization: A Hybrid Model for Embodied Morphogenesis</strong> - Takehiro Ishikawa - [[pdf]](https://arxiv.org/pdf/2511.10101)</summary>

**Abstract:** We investigate how to couple a learnable brain-like'' controller to a cell-like'' Gray--Scott substrate to steer pattern formation with minimal effort. A compact convolutional policy is embedded in a differentiable PyTorch reaction--diffusion simulator, producing spatially smooth, bounded modulations of the feed and kill parameters ($\Delta F$, $\Delta K$) under a warm--hold--decay gain schedule. Training optimizes Turing-band spectral targets (FFT-based) while penalizing control effort ($\ell_1/\ell_2$) and instability. We compare three regimes: pure reaction--diffusion, NN-dominant, and a hybrid coupling. The hybrid achieves reliable, fast formation of target textures: 100% strict convergence in $\sim 165$ steps, matching cell-only spectral selectivity (0.436 vs.\ 0.434) while using $\sim 15\times$ less $\ell_1$ effort and $>200\times$ less $\ell_2$ power than NN-dominant control. An amplitude sweep reveals a non-monotonic Goldilocks'' zone ($A \approx 0.03$--$0.045$) that yields 100\% quasi convergence in 94--96 steps, whereas weaker or stronger gains fail to converge or degrade selectivity. These results quantify morphological computation: the controller seeds then cedes,'' providing brief, sparse nudges that place the system in the correct basin of attraction, after which local physics maintains the pattern. The study offers a practical recipe for building steerable, robust, and energy-efficient embodied systems that exploit an optimal division of labor between centralized learning and distributed self-organization.

**arXiv ID:** 2511.10101
</details>

<details>
<summary><strong>Towards Embodied Agentic AI: Review and Classification of LLM- and VLM-Driven Robot Autonomy and Interaction</strong> - Sahar Salimpour, Lei Fu, Kajetan Rachwał, Pascal Bertrand, Kevin O'Sullivan, Robert Jakob, Farhad Keramat, Leonardo Militano, Giovanni Toffetti, Harry Edelman, Jorge Peña Queralta - [[pdf]](https://arxiv.org/pdf/2508.05294)</summary>

**Abstract:** Foundation models, including large language models (LLMs) and vision-language models (VLMs), have recently enabled novel approaches to robot autonomy and human-robot interfaces. In parallel, vision-language-action models (VLAs) or large behavior models (LBMs) are increasing the dexterity and capabilities of robotic systems. This survey paper reviews works that advance agentic applications and architectures, including initial efforts with GPT-style interfaces and more complex systems where AI agents function as coordinators, planners, perception actors, or generalist interfaces. Such agentic architectures allow robots to reason over natural language instructions, invoke APIs, plan task sequences, or assist in operations and diagnostics. In addition to peer-reviewed research, due to the fast-evolving nature of the field, we highlight and include community-driven projects, ROS packages, and industrial frameworks that show emerging trends. We propose a taxonomy for classifying model integration approaches and present a comparative analysis of the role that agents play in different solutions in today's literature.

**arXiv ID:** 2508.05294
</details>

<details>
<summary><strong>MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation</strong> - Xun Huang, Shijia Zhao, Yunxiang Wang, Xin Lu, Wanfa Zhang, Rongsheng Qu, Weixin Li, Yunhong Wang, Chenglu Wen - [[pdf]](https://arxiv.org/pdf/2511.10376)</summary>

**Abstract:** Embodied navigation is a fundamental capability for robotic agents operating. Real-world deployment requires open vocabulary generalization and low training overhead, motivating zero-shot methods rather than task-specific RL training. However, existing zero-shot methods that build explicit 3D scene graphs often compress rich visual observations into text-only relations, leading to high construction cost, irreversible loss of visual evidence, and constrained vocabularies. To address these limitations, we introduce the Multi-modal 3D Scene Graph (M3DSG), which preserves visual cues by replacing textual relation

**arXiv ID:** 2511.10376
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (22 papers)</h2></summary>

<details>
<summary><strong>Beyond Single-Step Updates: Reinforcement Learning of Heuristics with Limited-Horizon Search</strong> - Gal Hadar, Forest Agostinelli, Shahaf S. Shperberg - [[pdf]](https://arxiv.org/pdf/2511.10264)</summary>

**Abstract:** Many sequential decision-making problems can be formulated as shortest-path problems, where the objective is to reach a goal state from a given starting state. Heuristic search is a standard approach for solving such problems, relying on a heuristic function to estimate the cost to the goal from any given state. Recent approaches leverage reinforcement learning to learn heuristics by applying deep approximate value iteration. These methods typically rely on single-step Bellman updates, where the heuristic of a state is updated based on its best neighbor and the corresponding edge cost. This work proposes a generalized approach that enhances both state sampling and heuristic updates by performing limited-horizon searches and updating each state's heuristic based on the shortest path to the search frontier, incorporating both edge costs and the heuristic values of frontier states.

**arXiv ID:** 2511.10264
</details>

<details>
<summary><strong>Strategic Opponent Modeling with Graph Neural Networks, Deep Reinforcement Learning and Probabilistic Topic Modeling</strong> - Georgios Chalkiadakis, Charilaos Akasiadis, Gerasimos Koresis, Stergios Plataniots, Leonidas Bakopoulos - [[pdf]](https://arxiv.org/pdf/2511.10501)</summary>

**Abstract:** This paper provides a comprehensive review of mainly Graph Neural Networks, Deep Reinforcement Learning, and Probabilistic Topic Modeling methods with a focus on their potential incorporation in strategic multiagent settings. We draw interest in (i) Machine Learning methods currently utilized for uncovering unknown model structures adaptable to the task of strategic opponent modeling, and (ii) the integration of these methods with Game Theoretic concepts that avoid relying on assumptions often invalid in real-world scenarios, such as the Common Prior Assumption (CPA) and the Self-Interest Hypothesis (SIH). We analyze the ability to handle uncertainty and heterogeneity, two characteristics that are very common in real-world application cases, as well as scalability. As a potential answer to effectively modeling relationships and interactions in multiagent settings, we champion the use of Graph Neural Networks (GNN). Such approaches are designed to operate upon graph-structured data, and have been shown to be a very powerful tool for performing tasks such as node classification and link prediction. Next, we review the domain of Reinforcement Learning (RL), and in particular that of Multiagent Deep Reinforcement Learning (MADRL). Following, we describe existing relevant game theoretic solution concepts and consider properties such as fairness and stability. Our review comes complete with a note on the literature that utilizes PTM in domains other than that of document analysis and classification. The capability of PTM to estimate unknown underlying distributions can help with tackling heterogeneity and unknown agent beliefs. Finally, we identify certain open challenges specifically, the need to (i) fit non-stationary environments, (ii) balance the degrees of stability and adaptation, (iii) tackle uncertainty and heterogeneity, (iv) guarantee scalability and solution tractability.

**arXiv ID:** 2511.10501
</details>

<details>
<summary><strong>Optimistic Reinforcement Learning with Quantile Objectives</strong> - Mohammad Alipour-Vaezi, Huaiyang Zhong, Kwok-Leung Tsui, Sajad Khodadadian - [[pdf]](https://arxiv.org/pdf/2511.09652)</summary>

**Abstract:** Reinforcement Learning (RL) has achieved tremendous success in recent years. However, the classical foundations of RL do not account for the risk sensitivity of the objective function, which is critical in various fields, including healthcare and finance. A popular approach to incorporate risk sensitivity is to optimize a specific quantile of the cumulative reward distribution. In this paper, we develop UCB-QRL, an optimistic learning algorithm for the $\tau$-quantile objective in finite-horizon Markov decision processes (MDPs). UCB-QRL is an iterative algorithm in which, at each iteration, we first estimate the underlying transition probability and then optimize the quantile value function over a confidence ball around this estimate. We show that UCB-QRL yields a high-probability regret bound $\mathcal O\left((2/\kappa)^{H+1}H\sqrt{SATH\log(2SATH/\delta)}\right)$ in the episodic setting with $S$ states, $A$ actions, $T$ episodes, and $H$ horizons. Here, $\kappa>0$ is a problem-dependent constant that captures the sensitivity of the underlying MDP's quantile value.

**arXiv ID:** 2511.09652
</details>

<details>
<summary><strong>EEGAgent: A Unified Framework for Automated EEG Analysis Using Large Language Models</strong> - Sha Zhao, Mingyi Peng, Haiteng Jiang, Tao Li, Shijian Li, Gang Pan - [[pdf]](https://arxiv.org/pdf/2511.09947)</summary>

**Abstract:** Scalable and generalizable analysis of brain activity is essential for advancing both clinical diagnostics and cognitive research. Electroencephalography (EEG), a non-invasive modality with high temporal resolution, has been widely used for brain states analysis. However, most existing EEG models are usually tailored for individual specific tasks, limiting their utility in realistic scenarios where EEG analysis often involves multi-task and continuous reasoning. In this work, we introduce EEGAgent, a general-purpose framework that leverages large language models (LLMs) to schedule and plan multiple tools to automatically complete EEG-related tasks. EEGAgent is capable of performing the key functions: EEG basic information perception, spatiotemporal EEG exploration, EEG event detection, interaction with users, and EEG report generation. To realize these capabilities, we design a toolbox composed of different tools for EEG preprocessing, feature extraction, event detection, etc. These capabilities were evaluated on public datasets, and our EEGAgent can support flexible and interpretable EEG analysis, highlighting its potential for real-world clinical applications.

**arXiv ID:** 2511.09947
</details>

<details>
<summary><strong>SEBA: Sample-Efficient Black-Box Attacks on Visual Reinforcement Learning</strong> - Tairan Huang, Yulin Jin, Junxu Liu, Qingqing Ye, Haibo Hu - [[pdf]](https://arxiv.org/pdf/2511.09681)</summary>

**Abstract:** Visual reinforcement learning has achieved remarkable progress in visual control and robotics, but its vulnerability to adversarial perturbations remains underexplored. Most existing black-box attacks focus on vector-based or discrete-action RL, and their effectiveness on image-based continuous control is limited by the large action space and excessive environment queries. We propose SEBA, a sample-efficient framework for black-box adversarial attacks on visual RL agents. SEBA integrates a shadow Q model that estimates cumulative rewards under adversarial conditions, a generative adversarial network that produces visually imperceptible perturbations, and a world model that simulates environment dynamics to reduce real-world queries. Through a two-stage iterative training procedure that alternates between learning the shadow model and refining the generator, SEBA achieves strong attack performance while maintaining efficiency. Experiments on MuJoCo and Atari benchmarks show that SEBA significantly reduces cumulative rewards, preserves visual fidelity, and greatly decreases environment interactions compared to prior black-box and white-box methods.

**arXiv ID:** 2511.09681
</details>

<details>
<summary><strong>Improved Offline Reinforcement Learning via Quantum Metric Encoding</strong> - Outongyi Lv, Yewei Yuan, Nana Liu - [[pdf]](https://arxiv.org/pdf/2511.10187)</summary>

**Abstract:** Reinforcement learning (RL) with limited samples is common in real-world applications. However, offline RL performance under this constraint is often suboptimal. We consider an alternative approach to dealing with limited samples by introducing the Quantum Metric Encoder (QME). In this methodology, instead of applying the RL framework directly on the original states and rewards, we embed the states into a more compact and meaningful representation, where the structure of the encoding is inspired by quantum circuits. For classical data, QME is a classically simulable, trainable unitary embedding and thus serves as a quantum-inspired module, on a classical device. For quantum data in the form of quantum states, QME can be implemented directly on quantum hardware, allowing for training without measurement or re-encoding.
We evaluated QME on three datasets, each limited to 100 samples. We use Soft-Actor-Critic (SAC) and Implicit-Q-Learning (IQL), two well-known RL algorithms, to demonstrate the effectiveness of our approach. From the experimental results, we find that training offline RL agents on QME-embedded states with decoded rewards yields significantly better performance than training on the original states and rewards. On average across the three datasets, for maximum reward performance, we achieve a 116.2% improvement for SAC and 117.6% for IQL.
We further investigate the $\Delta$-hyperbolicity of our framework, a geometric property of the state space known to be important for the RL training efficacy. The QME-embedded states exhibit low $\Delta$-hyperbolicity, suggesting that the improvement after embedding arises from the modified geometry of the state space induced by QME. Thus, the low $\Delta$-hyperbolicity and the corresponding effectiveness of QME could provide valuable information for developing efficient offline RL methods under limited-sample conditions.

**arXiv ID:** 2511.10187
</details>

<details>
<summary><strong>Heuristic Transformer: Belief Augmented In-Context Reinforcement Learning</strong> - Oliver Dippel, Alexei Lisitsa, Bei Peng - [[pdf]](https://arxiv.org/pdf/2511.10251)</summary>

**Abstract:** Transformers have demonstrated exceptional in-context learning (ICL) capabilities, enabling applications across natural language processing, computer vision, and sequential decision-making. In reinforcement learning, ICL reframes learning as a supervised problem, facilitating task adaptation without parameter updates. Building on prior work leveraging transformers for sequential decision-making, we propose Heuristic Transformer (HT), an in-context reinforcement learning (ICRL) approach that augments the in-context dataset with a belief distribution over rewards to achieve better decision-making. Using a variational auto-encoder (VAE), a low-dimensional stochastic variable is learned to represent the posterior distribution over rewards, which is incorporated alongside an in-context dataset and query states as prompt to the transformer policy. We assess the performance of HT across the Darkroom, Miniworld, and MuJoCo environments, showing that it consistently surpasses comparable baselines in terms of both effectiveness and generalization. Our method presents a promising direction to bridge the gap between belief-based augmentations and transformer-based decision-making.

**arXiv ID:** 2511.10251
</details>

<details>
<summary><strong>Towards Emotionally Intelligent and Responsible Reinforcement Learning</strong> - Garapati Keerthana, Manik Gupta - [[pdf]](https://arxiv.org/pdf/2511.10573)</summary>

**Abstract:** Personalized decision systems in healthcare and behavioral support often rely on static rule-based or engagement-maximizing heuristics that overlook users' emotional context and ethical constraints. Such approaches risk recommending insensitive or unsafe interventions, especially in domains involving serious mental illness, substance use disorders, or depression. To address this limitation, we propose a Responsible Reinforcement Learning (RRL) framework that integrates emotional and contextual understanding with ethical considerations into the sequential decision-making process. RRL formulates personalization as a Constrained Markov Decision Process (CMDP), where the agent optimizes engagement and adherence while ensuring emotional alignment and ethical safety. We introduce a multi-objective reward function that explicitly balances short-term behavioral engagement with long-term user well-being, and define an emotion-informed state representation that captures fluctuations in emotional readiness, affect, and risk. The proposed architecture can be instantiated with any RL algorithm (e.g., DQN, PPO) augmented with safety constraints or Lagrangian regularization. Conceptually, this framework operationalizes empathy and responsibility within machine learning policy optimization, bridging safe RL, affective computing and responsible AI. We discuss the implications of this approach for human-centric domains such as behavioral health, education, and digital therapeutics, and outline simulation-based validation paths for future empirical work. This paper aims to initiate a methodological conversation about ethically aligned reinforcement learning for emotionally aware and trustworthy personalization systems.

**arXiv ID:** 2511.10573
</details>

<details>
<summary><strong>Planning Agents on an Ego-Trip: Leveraging Hybrid Ego-Graph Ensembles for Improved Tool Retrieval in Enterprise Task Planning</strong> - Sahil Bansal, Sai Shruthi Sistla, Aarti Arikatala, Sebastian Schreiber - [[pdf]](https://arxiv.org/pdf/2508.05888)</summary>

**Abstract:** Effective tool pre-selection via retrieval is essential for AI agents to select from a vast array of tools when identifying and planning actions in the context of complex user queries. Despite its central role in planning, this aspect remains underexplored in the literature. Traditional approaches rely primarily on similarities between user queries and tool descriptions, which significantly limits retrieval accuracy, specifically when handling multi-step user requests. To address these limitations, we propose a Knowledge Graph (KG)-based tool retrieval framework that captures the semantic relationships between tools and their functional dependencies. Our retrieval algorithm leverages ensembles of 1-hop ego tool graphs to model direct and indirect connections between tools, enabling more comprehensive and contextual tool selection for multi-step tasks. We evaluate our approach on a synthetically generated internal dataset across six defined user classes, extending previous work on coherent dialogue synthesis and tool retrieval benchmarks. Results demonstrate that our tool graph-based method achieves 91.85% tool coverage on the micro-average CompleteRecall metric, compared to 89.26% for re-ranked semantic-lexical hybrid retrieval, the strongest non-KG baseline in our experiments. These findings support our hypothesis that the structural information modeled in the graph provides complementary signals to pure similarity matching, particularly for queries requiring sequential tool composition.

**arXiv ID:** 2508.05888
</details>

<details>
<summary><strong>An Efficient Training Pipeline for Reasoning Graphical User Interface Agents</strong> - Georgios Pantazopoulos, Eda B. Özyiğit - [[pdf]](https://arxiv.org/pdf/2511.08172)</summary>

**Abstract:** Visual grounding is the task of localising image regions from natural language queries and is critical for reasoning capable Graphical User Interface agents. Many existing methods rely on massive, noisy synthetic datasets. This work introduces an efficient training pipeline that combines model-based data filtering with parameter-efficient fine-tuning. From 4.8M synthetic examples, 12K clean and diverse instances are curated by first identifying challenging cases, removing misaligned and then selecting a diverse set of multimodal instances. On this data, a 3B-parameter Vision-Language Model is trained under three regimes: supervised fine-tuning, chain-of-thought-augmented fine-tuning, and reinforcement learning via Group Relative Policy Optimization. Models trained with the filtered data and lightweight training strategies match or surpass larger baselines on benchmarks such as ScreenSpot, Multimodal-Mind2Web, and AndroidControl. These results demonstrate that principled data curation and robust adaptation can rival large-scale training, enabling compact yet capable multimodal reasoning agents.

**arXiv ID:** 2511.08172
</details>

<details>
<summary><strong>Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning</strong> - Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, Junyang Lin - [[pdf]](https://arxiv.org/pdf/2506.01939)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs), while its mechanisms are not yet well understood. In this work, we undertake a pioneering exploration of RLVR through the novel perspective of token entropy patterns, comprehensively analyzing how different tokens influence reasoning performance. By examining token entropy patterns in Chain-of-Thought (CoT) reasoning, we observe that only a small fraction of tokens exhibit high entropy, and these tokens act as critical forks that steer the model toward diverse reasoning pathways. Furthermore, studying how entropy patterns evolve during RLVR training reveals that RLVR largely adheres to the base model's entropy patterns, primarily adjusting the entropy of high-entropy tokens. These findings highlight the significance of high-entropy tokens (i.e., forking tokens) to RLVR. We ultimately improve RLVR by restricting policy gradient updates to forking tokens and uncover a finding even beyond the 80/20 rule: utilizing only 20% of the tokens while maintaining performance comparable to full-gradient updates on the Qwen3-8B base model and significantly surpassing full-gradient updates on the Qwen3-32B (+11.04 on AIME'25 and +7.71 on AIME'24) and Qwen3-14B (+4.79 on AIME'25 and +5.21 on AIME'24) base models, highlighting a strong scaling trend. In contrast, training exclusively on the 80% lowest-entropy tokens leads to a marked decline in performance. These findings indicate that the efficacy of RLVR primarily arises from optimizing the high-entropy tokens that decide reasoning directions. Collectively, our results highlight the potential to understand RLVR through a token-entropy perspective and optimize RLVR by leveraging high-entropy minority tokens to further improve LLM reasoning.

**arXiv ID:** 2506.01939
</details>

<details>
<summary><strong>Test-Time Reinforcement Learning for GUI Grounding via Region Consistency</strong> - Yong Du, Yuchen Yan, Fei Tang, Zhengxi Lu, Chang Zong, Weiming Lu, Shengpei Jiang, Yongliang Shen - [[pdf]](https://arxiv.org/pdf/2508.05615)</summary>

**Abstract:** Graphical User Interface (GUI) grounding, the task of mapping natural language instructions to precise screen coordinates, is fundamental to autonomous GUI agents. While existing methods achieve strong performance through extensive supervised training or reinforcement learning with labeled rewards, they remain constrained by the cost and availability of pixel-level annotations. We observe that when models generate multiple predictions for the same GUI element, the spatial overlap patterns reveal implicit confidence signals that can guide more accurate localization. Leveraging this insight, we propose GUI-RC (Region Consistency), a test-time scaling method that constructs spatial voting grids from multiple sampled predictions to identify consensus regions where models show highest agreement. Without any training, GUI-RC improves accuracy by 2-3% across various architectures on ScreenSpot benchmarks. We further introduce GUI-RCPO (Region Consistency Policy Optimization), transforming these consistency patterns into rewards for test-time reinforcement learning. By computing how well each prediction aligns with the collective consensus, GUI-RCPO enables models to iteratively refine their outputs on unlabeled data during inference. Extensive experiments demonstrate the generality of our approach: using only 1,272 unlabeled data, GUI-RCPO achieves 3-6% accuracy improvements across various architectures on ScreenSpot benchmarks. Our approach reveals the untapped potential of test-time scaling and test-time reinforcement learning for GUI grounding, offering a promising path toward more data-efficient GUI agents.

**arXiv ID:** 2508.05615
</details>

<details>
<summary><strong>HierRouter: Coordinated Routing of Specialized Large Language Models via Reinforcement Learning</strong> - Nikunj Gupta, Bill Guo, Rajgopal Kannan, Viktor K. Prasanna - [[pdf]](https://arxiv.org/pdf/2511.09873)</summary>

**Abstract:** Large Language Models (LLMs) deliver state-of-the-art performance across many tasks but impose high computational and memory costs, limiting their deployment in resource-constrained or real-time settings. To address this, we propose HierRouter, a hierarchical routing approach that dynamically assembles inference pipelines from a pool of specialized, lightweight language models. Formulated as a finite-horizon Markov Decision Process (MDP), our approach trains a Proximal Policy Optimization (PPO)-based reinforcement learning agent to iteratively select which models to invoke at each stage of multi-hop inference. The agent conditions on the evolving context and accumulated cost to make context-aware routing decisions. Experiments with three open-source candidate LLMs across six benchmarks, including QA, code generation, and mathematical reasoning, show that HierRouter improves response quality by up to 2.4x compared to using individual models independently, while incurring only a minimal additional inference cost on average. These results highlight the promise of hierarchical routing for cost-efficient, high-performance LLM inference. All codes can be found here this https URL Nikunj-Gupta/hierouter.

**arXiv ID:** 2511.09873
</details>

<details>
<summary><strong>Rectify Evaluation Preference: Improving LLMs' Critique on Math Reasoning via Perplexity-aware Reinforcement Learning</strong> - Changyuan Tian, Zhicong Lu, Shuang Qian, Nayu Liu, Peiguang Li, Li Jin, Leiyi Hu, Zhizhao Zeng, Sirui Wang, Ke Zeng, Zhi Guo - [[pdf]](https://arxiv.org/pdf/2511.10303)</summary>

**Abstract:** To improve Multi-step Mathematical Reasoning (MsMR) of Large Language Models (LLMs), it is crucial to obtain scalable supervision from the corpus by automatically critiquing mistakes in the reasoning process of MsMR and rendering a final verdict of the problem-solution. Most existing methods rely on crafting high-quality supervised fine-tuning demonstrations for critiquing capability enhancement and pay little attention to delving into the underlying reason for the poor critiquing performance of LLMs. In this paper, we orthogonally quantify and investigate the potential reason -- imbalanced evaluation preference, and conduct a statistical preference analysis. Motivated by the analysis of the reason, a novel perplexity-aware reinforcement learning algorithm is proposed to rectify the evaluation preference, elevating the critiquing capability. Specifically, to probe into LLMs' critiquing characteristics, a One-to-many Problem-Solution (OPS) benchmark is meticulously constructed to quantify the behavior difference of LLMs when evaluating the problem solutions generated by itself and others. Then, to investigate the behavior difference in depth, we conduct a statistical preference analysis oriented on perplexity and find an intriguing phenomenon -- ``LLMs incline to judge solutions with lower perplexity as correct'', which is dubbed as \textit{imbalanced evaluation preference}. To rectify this preference, we regard perplexity as the baton in the algorithm of Group Relative Policy Optimization, supporting the LLMs to explore trajectories that judge lower perplexity as wrong and higher perplexity as correct. Extensive experimental results on our built OPS and existing available critic benchmarks demonstrate the validity of our method.

**arXiv ID:** 2511.10303
</details>

<details>
<summary><strong>Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM Instruction Following</strong> - Yun He, Wenzhe Li, Hejia Zhang, Songlin Li, Karishma Mandyam, Sopan Khosla, Yuanhao Xiong, Nanshu Wang, Selina Peng, Beibin Li, Shengjie Bi, Shishir G. Patil, Qi Qi, Shengyu Feng, Julian Katz-Samuels, Richard Yuanzhe Pang, Sujan Gonugondla, Hunter Lang, Yue Yu, Yundi Qian, Maryam Fazel-Zarandi, Licheng Yu, Amine Benhalloum, Hany Awadalla, Manaal Faruqui - [[pdf]](https://arxiv.org/pdf/2511.10507)</summary>

**Abstract:** Recent progress in large language models (LLMs) has led to impressive performance on a range of tasks, yet advanced instruction following (IF)-especially for complex, multi-turn, and system-prompted instructions-remains a significant challenge. Rigorous evaluation and effective training for such capabilities are hindered by the lack of high-quality, human-annotated benchmarks and reliable, interpretable reward signals. In this work, we introduce AdvancedIF (we will release this benchmark soon), a comprehensive benchmark featuring over 1,600 prompts and expert-curated rubrics that assess LLMs ability to follow complex, multi-turn, and system-level instructions. We further propose RIFL (Rubric-based Instruction-Following Learning), a novel post-training pipeline that leverages rubric generation, a finetuned rubric verifier, and reward shaping to enable effective reinforcement learning for instruction following. Extensive experiments demonstrate that RIFL substantially improves the instruction-following abilities of LLMs, achieving a 6.7% absolute gain on AdvancedIF and strong results on public benchmarks. Our ablation studies confirm the effectiveness of each component in RIFL. This work establishes rubrics as a powerful tool for both training and evaluating advanced IF in LLMs, paving the way for more capable and reliable AI systems.

**arXiv ID:** 2511.10507
</details>

<details>
<summary><strong>Thinking Forward and Backward: Multi-Objective Reinforcement Learning for Retrieval-Augmented Reasoning</strong> - Wenda Wei, Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Lixin Su, Shuaiqiang Wang, Dawei Yin, Maarten de Rijke, Xueqi Cheng - [[pdf]](https://arxiv.org/pdf/2511.09109)</summary>

**Abstract:** Retrieval-augmented generation (RAG) has proven to be effective in mitigating hallucinations in large language models, yet its effectiveness remains limited in complex, multi-step reasoning scenarios. Recent efforts have incorporated search-based interactions into RAG, enabling iterative reasoning with real-time retrieval. Most approaches rely on outcome-based supervision, offering no explicit guidance for intermediate steps. This often leads to reward hacking and degraded response quality. We propose Bi-RAR, a novel retrieval-augmented reasoning framework that evaluates each intermediate step jointly in both forward and backward directions. To assess the information completeness of each step, we introduce a bidirectional information distance grounded in Kolmogorov complexity, approximated via language model generation probabilities. This quantification measures both how far the current reasoning is from the answer and how well it addresses the question. To optimize reasoning under these bidirectional signals, we adopt a multi-objective reinforcement learning framework with a cascading reward structure that emphasizes early trajectory alignment. Empirical results on seven question answering benchmarks demonstrate that Bi-RAR surpasses previous methods and enables efficient interaction and reasoning with the search engine during training and inference.

**arXiv ID:** 2511.09109
</details>

<details>
<summary><strong>ConstrainedSQL: Training LLMs for Text2SQL via Constrained Reinforcement Learning</strong> - Weiqin Chen, Nhan Huu Pham, Michael Robert Glass, Long Hai Vu, Gaetano Rossiello, Dharmashankar Subramanian, Santiago Paternain - [[pdf]](https://arxiv.org/pdf/2511.09693)</summary>

**Abstract:** Reinforcement learning (RL) has demonstrated significant promise in enhancing the reasoning capabilities of Text2SQL LLMs, especially with advanced algorithms such as GRPO and DAPO. However, the performance of these methods is highly sensitive to the design of reward functions. Inappropriate rewards can lead to reward hacking, where models exploit loopholes in the reward structure to achieve high scores without genuinely solving the task. This work considers a constrained RL framework for Text2SQL that incorporates natural and interpretable reward and constraint signals, while dynamically balancing trade-offs among them during the training. We establish the theoretical guarantees of our constrained RL framework and our numerical experiments on the well-known Text2SQL datasets substantiate the improvement of our approach over the state-of-the-art RL-trained LLMs.

**arXiv ID:** 2511.09693
</details>

<details>
<summary><strong>DemoTuner: Efficient DBMS Knobs Tuning via LLM-Assisted Demonstration Reinforcement Learning</strong> - Hui Dou, Lei Jin, Yuxuan Zhou, Jiang He, Yiwen Zhang - [[pdf]](https://arxiv.org/pdf/2511.09998)</summary>

**Abstract:** The performance of modern DBMSs such as MySQL and PostgreSQL heavily depends on the configuration of performance-critical knobs. Manual tuning these knobs is laborious and inefficient due to the complex and high-dimensional nature of the configuration space. Among the automated tuning methods, reinforcement learning (RL)-based methods have recently sought to improve the DBMS knobs tuning process from several different perspectives. However, they still encounter challenges with slow convergence speed during offline training. In this paper, we mainly focus on how to leverage the valuable tuning hints contained in various textual documents such as DBMS manuals and web forums to improve the offline training of RL-based methods. To this end, we propose an efficient DBMS knobs tuning framework named DemoTuner via a novel LLM-assisted demonstration reinforcement learning method. Specifically, to comprehensively and accurately mine tuning hints from documents, we design a structured chain of thought prompt to employ LLMs to conduct a condition-aware tuning hints extraction task. To effectively integrate the mined tuning hints into RL agent training, we propose a hint-aware demonstration reinforcement learning algorithm HA-DDPGfD in DemoTuner. As far as we know, DemoTuner is the first work to introduce the demonstration reinforcement learning algorithm for DBMS knobs tuning. Experimental evaluations conducted on MySQL and PostgreSQL across various workloads demonstrate the significant advantages of DemoTuner in both performance improvement and online tuning cost reduction over three representative baselines including DB-BERT, GPTuner and CDBTune. Additionally, DemoTuner also exhibits superior adaptability to application scenarios with unknown workloads.

**arXiv ID:** 2511.09998
</details>

<details>
<summary><strong>Operator Models for Continuous-Time Offline Reinforcement Learning</strong> - Nicolas Hoischen, Petar Bevanda, Max Beier, Stefan Sosnowski, Boris Houska, Sandra Hirche - [[pdf]](https://arxiv.org/pdf/2511.10383)</summary>

**Abstract:** Continuous-time stochastic processes underlie many natural and engineered systems. In healthcare, autonomous driving, and industrial control, direct interaction with the environment is often unsafe or impractical, motivating offline reinforcement learning from historical data. However, there is limited statistical understanding of the approximation errors inherent in learning policies from offline datasets. We address this by linking reinforcement learning to the Hamilton-Jacobi-Bellman equation and proposing an operator-theoretic algorithm based on a simple dynamic programming recursion. Specifically, we represent our world model in terms of the infinitesimal generator of controlled diffusion processes learned in a reproducing kernel Hilbert space. By integrating statistical learning methods and operator theory, we establish global convergence of the value function and derive finite-sample guarantees with bounds tied to system properties such as smoothness and stability. Our theoretical and numerical results indicate that operator-based approaches may hold promise in solving offline reinforcement learning using continuous-time optimal control.

**arXiv ID:** 2511.10383
</details>

<details>
<summary><strong>Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control</strong> - Georgios Papoudakis, Thomas Coste, Jianye Hao, Jun Wang, Kun Shao - [[pdf]](https://arxiv.org/pdf/2509.01720)</summary>

**Abstract:** Reinforcement learning (RL) using foundation models for policy approximations in multi-turn tasks remains challenging. We identify two main limitations related to sparse reward settings and policy gradient updates, based on which we formulate a key insight: updates from positive samples with high returns typically do not require policy regularisation, whereas updates from negative samples, reflecting undesirable behaviour, can harm model performance. This paper introduces Succeed or Learn Slowly (SoLS), a novel off-policy RL algorithm evaluated on mobile app control tasks. SoLS improves sample efficiency when fine-tuning foundation models for user interface navigation via a modified off-policy actor-critic approach, applying direct policy updates for positive samples and conservative, regularised updates for negative ones to prevent model degradation. We augment SoLS with Successful Transition Replay (STR), which prioritises learning from successful interactions, further improving sample efficiency. We evaluate SoLS on the AndroidWorld benchmark, where it significantly outperforms existing methods (at least 17% relative increase), including prompt-engineering and RL approaches, while requiring substantially fewer computational resources than GPT-4o-based methods with 5-60x faster inference.

**arXiv ID:** 2509.01720
</details>

<details>
<summary><strong>Semiparametric Double Reinforcement Learning with Applications to Long-Term Causal Inference</strong> - Lars van der Laan, David Hubbard, Allen Tran, Nathan Kallus, Aurélien Bibaut - [[pdf]](https://arxiv.org/pdf/2501.06926)</summary>

**Abstract:** Double Reinforcement Learning (DRL) enables efficient inference for policy values in nonparametric Markov decision processes (MDPs), but existing methods face two major obstacles: (1) they require stringent intertemporal overlap conditions on state trajectories, and (2) they rely on estimating high-dimensional occupancy density ratios. Motivated by problems in long-term causal inference, we extend DRL to a semiparametric setting and develop doubly robust, automatic estimators for general linear functionals of the Q-function in infinite-horizon, time-homogeneous MDPs. By imposing structure on the Q-function, we relax the overlap conditions required by nonparametric methods and obtain efficiency gains. The second obstacle--density-ratio estimation--typically requires computationally expensive and unstable min-max optimization. To address both challenges, we introduce superefficient nonparametric estimators whose limiting variance falls below the generalized Cramer-Rao bound. These estimators treat the Q-function as a one-dimensional summary of the state-action process, reducing high-dimensional overlap requirements to a single-dimensional condition. The procedure is simple to implement: estimate and calibrate the Q-function using fitted Q-iteration, then plug the result into the target functional, thereby avoiding density-ratio estimation altogether.

**arXiv ID:** 2501.06926
</details>

<details>
<summary><strong>Autonomous X-ray Fluorescence Mapping of Chemically Heterogeneous Systems via a Correlative Feature Detection Framework</strong> - Carlos Deleon, Dmitri Gavrilov, Peggy ODay, Ajith Pattammattel - [[pdf]](https://arxiv.org/pdf/2511.09975)</summary>

**Abstract:** We present X-AutoMap, a modular framework for autonomous X-ray fluorescence (XRF) mapping that enables chemically informed targeting of regions of interest through a correlative feature detection strategy. The system integrates classical computer vision and rule-based logic to identify features based on spatial relationships across multiple elemental maps, rather than relying solely on intensity or morphology. Tight integration with the Bluesky control infrastructure at the NSLS-II Hard X-ray Nanoprobe (HXN) beamline enables real-time, closed-loop scan orchestration. Applied to a chemically heterogeneous urban PM2.5 sample, X-AutoMap reduced high-resolution acquisition time from over 44 hours to approximately 10 hours by targeting compositionally significant features identified from coarse scans. High-resolution results revealed diverse particle types, including fully mixed, partially overlapping, and spatially distinct multi-element structures, demonstrating the ability of the framework to isolate chemically relevant features with minimal user intervention. The framework supports interactive and autonomous modes, operates within hardware constraints via grid-based scanning, and is robust across varying sample conditions. Future extensions will incorporate machine learning and probabilistic sampling to further improve detection sensitivity and scan efficiency. X-AutoMap is currently in active use at HXN and provides a flexible foundation for scalable, intelligent imaging workflows at synchrotron beamlines.

**arXiv ID:** 2511.09975
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
