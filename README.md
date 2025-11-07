# Agent arXiv Daily

**Last Updated:** 2025-11-07 02:48:15

**Total Papers:** 49

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (3 papers)</h2></summary>

<details>
<summary><strong>Scaling Agent Learning via Experience Synthesis</strong> - Zhaorun Chen, Zhuokai Zhao, Kai Zhang, Bo Liu, Qi Qi, Yifan Wu, Tarun Kalluri, Sara Cao, Yuanhao Xiong, Haibo Tong, Huaxiu Yao, Hengduo Li, Jiacheng Zhu, Xian Li, Dawn Song, Bo Li, Jason Weston, Dat Huynh - [[pdf]](https://arxiv.org/pdf/2511.03773)</summary>

**Abstract:** While reinforcement learning (RL) can empower large language model (LLM) agents by enabling self-improvement through interaction, its practical adoption remains challenging due to costly rollouts, limited task diversity, unreliable reward signals, and infrastructure complexity, all of which obstruct the collection of scalable experience data. To address these challenges, we introduce DreamGym, the first unified framework designed to synthesize diverse experiences with scalability in mind to enable effective online RL training for autonomous agents. Rather than relying on expensive real-environment rollouts, DreamGym distills environment dynamics into a reasoning-based experience model that derives consistent state transitions and feedback signals through step-by-step reasoning, enabling scalable agent rollout collection for RL. To improve the stability and quality of transitions, DreamGym leverages an experience replay buffer initialized with offline real-world data and continuously enriched with fresh interactions to actively support agent training. To improve knowledge acquisition, DreamGym adaptively generates new tasks that challenge the current agent policy, enabling more effective online curriculum learning. Experiments across diverse environments and agent backbones demonstrate that DreamGym substantially improves RL training, both in fully synthetic settings and in sim-to-real transfer scenarios. On non-RL-ready tasks like WebArena, DreamGym outperforms all baselines by over 30%. And in RL-ready but costly settings, it matches GRPO and PPO performance using only synthetic interactions. When transferring a policy trained purely on synthetic experiences to real-environment RL, DreamGym yields significant additional performance gains while requiring far fewer real-world interactions, providing a scalable warm-start strategy for general-purpose RL.

**arXiv ID:** 2511.03773
</details>

<details>
<summary><strong>MazeMate: An LLM-Powered Chatbot to Support Computational Thinking in Gamified Programming Learning</strong> - Chenyu Hou, Hua Yu, Gaoxia Zhu, John Derek Anas, Jiao Liu, Yew Soon Ong - [[pdf]](https://arxiv.org/pdf/2511.03727)</summary>

**Abstract:** Computational Thinking (CT) is a foundational problem-solving skill, and gamified programming environments are a widely adopted approach to cultivating it. While large language models (LLMs) provide on-demand programming support, current applications rarely foster CT development. We present MazeMate, an LLM-powered chatbot embedded in a 3D Maze programming game, designed to deliver adaptive, context-sensitive scaffolds aligned with CT processes in maze solving and maze design. We report on the first classroom implementation with 247 undergraduates. Students rated MazeMate as moderately helpful, with higher perceived usefulness for maze solving than for maze design. Thematic analysis confirmed support for CT processes such as decomposition, abstraction, and algorithmic thinking, while also revealing limitations in supporting maze design, including mismatched suggestions and fabricated algorithmic solutions. These findings demonstrate the potential of LLM-based scaffolding to support CT and underscore directions for design refinement to enhance MazeMate usability in authentic classrooms.

**arXiv ID:** 2511.03727
</details>

<details>
<summary><strong>Transforming Mentorship: An AI Powered Chatbot Approach to University Guidance</strong> - Mashrur Rahman, Mantaqa abedin, Monowar Zamil Abir, Faizul Islam Ansari, Adib Reza, Farig Yousuf Sadeque, Niloy Farhan - [[pdf]](https://arxiv.org/pdf/2511.04172)</summary>

**Abstract:** University students face immense challenges during their undergraduate lives, often being deprived of personalized on-demand guidance that mentors fail to provide at scale. Digital tools exist, but there is a serious lack of customized coaching for newcomers. This paper presents an AI-powered chatbot that will serve as a mentor for the students of BRAC University. The main component is a data ingestion pipeline that efficiently processes and updates information from diverse sources, such as CSV files and university webpages. The chatbot retrieves information through a hybrid approach, combining BM25 lexical ranking with ChromaDB semantic retrieval, and uses a Large Language Model, LLaMA-3.3-70B, to generate conversational responses. The generated text was found to be semantically highly relevant, with a BERTScore of 0.831 and a METEOR score of 0.809. The data pipeline was also very efficient, taking 106.82 seconds for updates, compared to 368.62 seconds for new data. This chatbot will be able to help students by responding to their queries, helping them to get a better understanding of university life, and assisting them to plan better routines for their semester in the open-credit university.

**arXiv ID:** 2511.04172
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (7 papers)</h2></summary>

<details>
<summary><strong>Beyond Shortest Path: Agentic Vehicular Routing with Semantic Context</strong> - Carnot Braun, Rafael O. Jarczewski, Gabriel U. Talasso, Leandro A. Villas, Allan M. de Souza - [[pdf]](https://arxiv.org/pdf/2511.04464)</summary>

**Abstract:** Traditional vehicle routing systems efficiently optimize singular metrics like time or distance, and when considering multiple metrics, they need more processes to optimize . However, they lack the capability to interpret and integrate the complex, semantic, and dynamic contexts of human drivers, such as multi-step tasks, situational constraints, or urgent needs. This paper introduces and evaluates PAVe (Personalized Agentic Vehicular Routing), a hybrid agentic assistant designed to augment classical pathfinding algorithms with contextual reasoning. Our approach employs a Large Language Model (LLM) agent that operates on a candidate set of routes generated by a multi-objective (time, CO2) Dijkstra algorithm. The agent evaluates these options against user-provided tasks, preferences, and avoidance rules by leveraging a pre-processed geospatial cache of urban Points of Interest (POIs). In a benchmark of realistic urban scenarios, PAVe successfully used complex user intent into appropriate route modifications, achieving over 88% accuracy in its initial route selections with a local model. We conclude that combining classical routing algorithms with an LLM-based semantic reasoning layer is a robust and effective approach for creating personalized, adaptive, and scalable solutions for urban mobility optimization.

**arXiv ID:** 2511.04464
</details>

<details>
<summary><strong>Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption through Empirical and Theoretical Analysis</strong> - Lars Krupp, Daniel Geißler, Vishal Banwari, Paul Lukowicz, Jakob Karolus - [[pdf]](https://arxiv.org/pdf/2511.04481)</summary>

**Abstract:** Web agents, like OpenAI's Operator and Google's Project Mariner, are powerful agentic systems pushing the boundaries of Large Language Models (LLM). They can autonomously interact with the internet at the user's behest, such as navigating websites, filling search masks, and comparing price lists. Though web agent research is thriving, induced sustainability issues remain largely unexplored. To highlight the urgency of this issue, we provide an initial exploration of the energy and $CO_2$ cost associated with web agents from both a theoretical -via estimation- and an empirical perspective -by benchmarking. Our results show how different philosophies in web agent creation can severely impact the associated expended energy, and that more energy consumed does not necessarily equate to better results. We highlight a lack of transparency regarding disclosing model parameters and processes used for some web agents as a limiting factor when estimating energy consumption. Our work contributes towards a change in thinking of how we evaluate web agents, advocating for dedicated metrics measuring energy consumption in benchmarks.

**arXiv ID:** 2511.04481
</details>

<details>
<summary><strong>Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper</strong> - Atsuyuki Miyai, Mashiro Toyooka, Takashi Otonari, Zaiying Zhao, Kiyoharu Aizawa - [[pdf]](https://arxiv.org/pdf/2511.04583)</summary>

**Abstract:** Understanding the current capabilities and risks of AI Scientist systems is essential for ensuring trustworthy and sustainable AI-driven scientific progress while preserving the integrity of the academic ecosystem. To this end, we develop Jr. AI Scientist, a state-of-the-art autonomous AI scientist system that mimics the core research workflow of a novice student researcher: Given the baseline paper from the human mentor, it analyzes its limitations, formulates novel hypotheses for improvement, validates them through rigorous experimentation, and writes a paper with the results. Unlike previous approaches that assume full automation or operate on small-scale code, Jr. AI Scientist follows a well-defined research workflow and leverages modern coding agents to handle complex, multi-file implementations, leading to scientifically valuable contributions. For evaluation, we conducted automated assessments using AI Reviewers, author-led evaluations, and submissions to Agents4Science, a venue dedicated to AI-driven scientific contributions. The findings demonstrate that Jr. AI Scientist generates papers receiving higher review scores than existing fully automated systems. Nevertheless, we identify important limitations from both the author evaluation and the Agents4Science reviews, indicating the potential risks of directly applying current AI Scientist systems and key challenges for future research. Finally, we comprehensively report various risks identified during development. We hope these insights will deepen understanding of current progress and risks in AI Scientist development.

**arXiv ID:** 2511.04583
</details>

<details>
<summary><strong>PEFA-AI: Advancing Open-source LLMs for RTL generation using Progressive Error Feedback Agentic-AI</strong> - Athma Narayanan, Mahesh Subedar, Omesh Tickoo - [[pdf]](https://arxiv.org/pdf/2511.03934)</summary>

**Abstract:** We present an agentic flow consisting of multiple agents that combine specialized LLMs and hardware simulation tools to collaboratively complete the complex task of Register Transfer Level (RTL) generation without human intervention. A key feature of the proposed flow is the progressive error feedback system of agents (PEFA), a self-correcting mechanism that leverages iterative error feedback to progressively increase the complexity of the approach. The generated RTL includes checks for compilation, functional correctness, and synthesizable constructs. To validate this adaptive approach to code generation, benchmarking is performed using two opensource natural language-to-RTL datasets. We demonstrate the benefits of the proposed approach implemented on an open source agentic framework, using both open- and closed-source LLMs, effectively bridging the performance gap between them. Compared to previously published methods, our approach sets a new benchmark, providing state-of-the-art pass rates while being efficient in token counts.

**arXiv ID:** 2511.03934
</details>

<details>
<summary><strong>Learning from Online Videos at Inference Time for Computer-Use Agents</strong> - Yujian Liu, Ze Wang, Hao Chen, Ximeng Sun, Xiaodong Yu, Jialian Wu, Jiang Liu, Emad Barsoum, Zicheng Liu, Shiyu Chang - [[pdf]](https://arxiv.org/pdf/2511.04137)</summary>

**Abstract:** Computer-use agents can operate computers and automate laborious tasks, but despite recent rapid progress, they still lag behind human users, especially when tasks require domain-specific procedural knowledge about particular applications, platforms, and multi-step workflows. Humans can bridge this gap by watching video tutorials: we search, skim, and selectively imitate short segments that match our current subgoal. In this paper, we study how to enable computer-use agents to learn from online videos at inference time effectively. We propose a framework that retrieves and filters tutorial videos, converts them into structured demonstration trajectories, and dynamically selects trajectories as in-context guidance during execution. Particularly, using a VLM, we infer UI actions, segment videos into short subsequences of actions, and assign each subsequence a textual objective. At inference time, a two-stage selection mechanism dynamically chooses a single trajectory to add in context at each step, focusing the agent on the most helpful local guidance for its next decision. Experiments on two widely used benchmarks show that our framework consistently outperforms strong base agents and variants that use only textual tutorials or transcripts. Analyses highlight the importance of trajectory segmentation and selection, action filtering, and visual information, suggesting that abundant online videos can be systematically distilled into actionable guidance that improves computer-use agents at inference time. Our code is available at this https URL.

**arXiv ID:** 2511.04137
</details>

<details>
<summary><strong>RAGalyst: Automated Human-Aligned Agentic Evaluation for Domain-Specific RAG</strong> - Joshua Gao, Quoc Huy Pham, Subin Varghese, Silwal Saurav, Vedhus Hoskere - [[pdf]](https://arxiv.org/pdf/2511.04502)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) is a critical technique for grounding Large Language Models (LLMs) in factual evidence, yet evaluating RAG systems in specialized, safety-critical domains remains a significant challenge. Existing evaluation frameworks often rely on heuristic-based metrics that fail to capture domain-specific nuances and other works utilize LLM-as-a-Judge approaches that lack validated alignment with human judgment. This paper introduces RAGalyst, an automated, human-aligned agentic framework designed for the rigorous evaluation of domain-specific RAG systems. RAGalyst features an agentic pipeline that generates high-quality, synthetic question-answering (QA) datasets from source documents, incorporating an agentic filtering step to ensure data fidelity. The framework refines two key LLM-as-a-Judge metrics-Answer Correctness and Answerability-using prompt optimization to achieve a strong correlation with human annotations. Applying this framework to evaluate various RAG components across three distinct domains (military operations, cybersecurity, and bridge engineering), we find that performance is highly context-dependent. No single embedding model, LLM, or hyperparameter configuration proves universally optimal. Additionally, we provide an analysis on the most common low Answer Correctness reasons in RAG. These findings highlight the necessity of a systematic evaluation framework like RAGalyst, which empowers practitioners to uncover domain-specific trade-offs and make informed design choices for building reliable and effective RAG systems. RAGalyst is available on our Github.

**arXiv ID:** 2511.04502
</details>

<details>
<summary><strong>PoCo: Agentic Proof-of-Concept Exploit Generation for Smart Contracts</strong> - Vivi Andersson, Sofia Bobadilla, Harald Hobbelhagen, Martin Monperrus - [[pdf]](https://arxiv.org/pdf/2511.02780)</summary>

**Abstract:** Smart contracts operate in a highly adversarial environment, where vulnerabilities can lead to substantial financial losses. Thus, smart contracts are subject to security audits. In auditing, proof-of-concept (PoC) exploits play a critical role by demonstrating to the stakeholders that the reported vulnerabilities are genuine, reproducible, and actionable. However, manually creating PoCs is time-consuming, error-prone, and often constrained by tight audit schedules. We introduce POCO, an agentic framework that automatically generates executable PoC exploits from natural-language vulnerability descriptions written by auditors. POCO autonomously generates PoC exploits in an agentic manner by interacting with a set of code-execution tools in a Reason-Act-Observe loop. It produces fully executable exploits compatible with the Foundry testing framework, ready for integration into audit reports and other security tools. We evaluate POCO on a dataset of 23 real-world vulnerability reports. POCO consistently outperforms the prompting and workflow baselines, generating well-formed and logically correct PoCs. Our results demonstrate that agentic frameworks can significantly reduce the effort required for high-quality PoCs in smart contract audits. Our contribution provides readily actionable knowledge for the smart contract security community.

**arXiv ID:** 2511.02780
</details>

</details>

<details open>
<summary><h2>LLM Agents (2 papers)</h2></summary>

<details>
<summary><strong>Agentmandering: A Game-Theoretic Framework for Fair Redistricting via Large Language Model Agents</strong> - Hao Li, Haotian Chen, Ruoyuan Gong, Juanjuan Wang, Hao Jiang - [[pdf]](https://arxiv.org/pdf/2511.04076)</summary>

**Abstract:** Redistricting plays a central role in shaping how votes are translated into political power. While existing computational methods primarily aim to generate large ensembles of legally valid districting plans, they often neglect the strategic dynamics involved in the selection process. This oversight creates opportunities for partisan actors to cherry-pick maps that, while technically compliant, are politically advantageous. Simply satisfying formal constraints does not ensure fairness when the selection process itself can be manipulated. We propose \textbf{Agentmandering}, a framework that reimagines redistricting as a turn-based negotiation between two agents representing opposing political interests. Drawing inspiration from game-theoretic ideas, particularly the \textit{Choose-and-Freeze} protocol, our method embeds strategic interaction into the redistricting process via large language model (LLM) agents. Agents alternate between selecting and freezing districts from a small set of candidate maps, gradually partitioning the state through constrained and interpretable choices. Evaluation on post-2020 U.S. Census data across all states shows that Agentmandering significantly reduces partisan bias and unfairness, while achieving 2 to 3 orders of magnitude lower variance than standard baselines. These results demonstrate both fairness and stability, especially in swing-state scenarios. Our code is available at this https URL.

**arXiv ID:** 2511.04076
</details>

<details>
<summary><strong>Speed at the Cost of Quality? The Impact of LLM Agent Assistance on Software Development</strong> - Hao He, Courtney Miller, Shyam Agarwal, Christian Kästner, Bogdan Vasilescu - [[pdf]](https://arxiv.org/pdf/2511.04427)</summary>

**Abstract:** Large language models (LLMs) have demonstrated the promise to revolutionize the field of software engineering. Among other things, LLM agents are rapidly gaining momentum in their application to software development, with practitioners claiming a multifold productivity increase after adoption. Yet, empirical evidence is lacking around these claims. In this paper, we estimate the causal effect of adopting a widely popular LLM agent assistant, namely Cursor, on development velocity and software quality. The estimation is enabled by a state-of-the-art difference-in-differences design comparing Cursor-adopting GitHub projects with a matched control group of similar GitHub projects that do not use Cursor. We find that the adoption of Cursor leads to a significant, large, but transient increase in project-level development velocity, along with a significant and persistent increase in static analysis warnings and code complexity. Further panel generalized method of moments estimation reveals that the increase in static analysis warnings and code complexity acts as a major factor causing long-term velocity slowdown. Our study carries implications for software engineering practitioners, LLM agent assistant designers, and researchers.

**arXiv ID:** 2511.04427
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (15 papers)</h2></summary>

<details>
<summary><strong>ArchPilot: A Proxy-Guided Multi-Agent Approach for Machine Learning Engineering</strong> - Zhuowen Yuan, Tao Liu, Yang Yang, Yang Wang, Feng Qi, Kaushik Rangadurai, Bo Li, Shuang Yang - [[pdf]](https://arxiv.org/pdf/2511.03985)</summary>

**Abstract:** Recent LLM-based agents have demonstrated strong capabilities in automated ML engineering. However, they heavily rely on repeated full training runs to evaluate candidate solutions, resulting in significant computational overhead, limited scalability to large search spaces, and slow iteration cycles. To address these challenges, we introduce ArchPilot, a multi-agent system that integrates architecture generation, proxy-based evaluation, and adaptive search into a unified framework. ArchPilot consists of three specialized agents: an orchestration agent that coordinates the search process using a Monte Carlo Tree Search (MCTS)-inspired novel algorithm with a restart mechanism and manages memory of previous candidates; a generation agent that iteratively generates, improves, and debugs candidate architectures; and an evaluation agent that executes proxy training runs, generates and optimizes proxy functions, and aggregates the proxy scores into a fidelity-aware performance metric. This multi-agent collaboration allows ArchPilot to prioritize high-potential candidates with minimal reliance on expensive full training runs, facilitating efficient ML engineering under limited budgets. Experiments on MLE-Bench demonstrate that ArchPilot outperforms SOTA baselines such as AIDE and ML-Master, validating the effectiveness of our multi-agent system.

**arXiv ID:** 2511.03985
</details>

<details>
<summary><strong>Detecting Silent Failures in Multi-Agentic AI Trajectories</strong> - Divya Pathak, Harshit Kumar, Anuska Roy, Felix George, Mudit Verma, Pratibha Moogi - [[pdf]](https://arxiv.org/pdf/2511.04032)</summary>

**Abstract:** Multi-Agentic AI systems, powered by large language models (LLMs), are inherently non-deterministic and prone to silent failures such as drift, cycles, and missing details in outputs, which are difficult to detect. We introduce the task of anomaly detection in agentic trajectories to identify these failures and present a dataset curation pipeline that captures user behavior, agent non-determinism, and LLM variation. Using this pipeline, we curate and label two benchmark datasets comprising \textbf{4,275 and 894} trajectories from Multi-Agentic AI systems. Benchmarking anomaly detection methods on these datasets, we show that supervised (XGBoost) and semi-supervised (SVDD) approaches perform comparably, achieving accuracies up to 98% and 96%, respectively. This work provides the first systematic study of anomaly detection in Multi-Agentic AI systems, offering datasets, benchmarks, and insights to guide future research.

**arXiv ID:** 2511.04032
</details>

<details>
<summary><strong>DR. WELL: Dynamic Reasoning and Learning with Symbolic World Model for Embodied LLM-Based Multi-Agent Collaboration</strong> - Narjes Nourzad, Hanqing Yang, Shiyu Chen, Carlee Joe-Wong - [[pdf]](https://arxiv.org/pdf/2511.04646)</summary>

**Abstract:** Cooperative multi-agent planning requires agents to make joint decisions with partial information and limited communication. Coordination at the trajectory level often fails, as small deviations in timing or movement cascade into conflicts. Symbolic planning mitigates this challenge by raising the level of abstraction and providing a minimal vocabulary of actions that enable synchronization and collective progress. We present DR. WELL, a decentralized neurosymbolic framework for cooperative multi-agent planning. Cooperation unfolds through a two-phase negotiation protocol: agents first propose candidate roles with reasoning and then commit to a joint allocation under consensus and environment constraints. After commitment, each agent independently generates and executes a symbolic plan for its role without revealing detailed trajectories. Plans are grounded in execution outcomes via a shared world model that encodes the current state and is updated as agents act. By reasoning over symbolic plans rather than raw trajectories, DR. WELL avoids brittle step-level alignment and enables higher-level operations that are reusable, synchronizable, and interpretable. Experiments on cooperative block-push tasks show that agents adapt across episodes, with the dynamic world model capturing reusable patterns and improving task completion rates and efficiency. Experiments on cooperative block-push tasks show that our dynamic world model improves task completion and efficiency through negotiation and self-refinement, trading a time overhead for evolving, more efficient collaboration strategies.

**arXiv ID:** 2511.04646
</details>

<details>
<summary><strong>OptiMA: A Transaction-Based Framework with Throughput Optimization for Very Complex Multi-Agent Systems</strong> - Umut Çalıkyılmaz, Nitin Nayak, Jinghua Groppe, Sven Groppe - [[pdf]](https://arxiv.org/pdf/2511.03761)</summary>

**Abstract:** In recent years, the research of multi-agent systems has taken a direction to explore larger and more complex models to fulfill sophisticated tasks. We point out two possible pitfalls that might be caused by increasing complexity; susceptibilities to faults, and performance bottlenecks. To prevent the former threat, we propose a transaction-based framework to design very complex multi-agent systems (VCMAS). To address the second threat, we offer to integrate transaction scheduling into the proposed framework. We implemented both of these ideas to develop the OptiMA framework and show that it is able to facilitate the execution of VCMAS with more than a hundred agents. We also demonstrate the effect of transaction scheduling on such a system by showing improvements up to more than 16\%. Furthermore, we also performed a theoretical analysis on the transaction scheduling problem and provided practical tools that can be used for future research on it.

**arXiv ID:** 2511.03761
</details>

<details>
<summary><strong>Collaborative Agents for Automated Program Repair in Ruby</strong> - Nikta Akbarpour, Mahdieh Sadat Benis, Fatemeh Hendijani Fard, Ali Ouni, Mohamed Aymen Saied - [[pdf]](https://arxiv.org/pdf/2511.03925)</summary>

**Abstract:** Automated Program Repair (APR) has advanced rapidly with Large Language Models (LLMs), but most existing methods remain computationally expensive, and focused on a small set of languages. Ruby, despite its widespread use in web development and the persistent challenges faced by its developers, has received little attention in APR research. In this paper, we introduce RAMP, a novel lightweight framework that formulates program repair as a feedback-driven, iterative process for Ruby. RAMP employs a team of collaborative agents that generate targeted tests, reflect on errors, and refine candidate fixes until a correct solution is found. Unlike prior approaches, RAMP is designed to avoid reliance on large multilingual repair databases or costly fine-tuning, instead operating directly on Ruby through lightweight prompting and test-driven feedback. Evaluation on the XCodeEval benchmark shows that RAMP achieves a pass@1 of 67% on Ruby, outper-forming prior approaches. RAMP converges quickly within five iterations, and ablation studies confirm that test generation and self-reflection are key drivers of its performance. Further analysis shows that RAMP is particularly effective at repairing wrong answers, compilation errors, and runtime errors. Our approach provides new insights into multi-agent repair strategies, and establishes a foundation for extending LLM-based debugging tools to under-studied languages.

**arXiv ID:** 2511.03925
</details>

<details>
<summary><strong>BAPPA: Benchmarking Agents, Plans, and Pipelines for Automated Text-to-SQL Generation</strong> - Fahim Ahmed, Md Mubtasim Ahasan, Jahir Sadik Monon, Muntasir Wahed, M Ashraful Amin, A K M Mahbubur Rahman, Amin Ahsan Ali - [[pdf]](https://arxiv.org/pdf/2511.04153)</summary>

**Abstract:** Text-to-SQL systems provide a natural language interface that can enable even laymen to access information stored in databases. However, existing Large Language Models (LLM) struggle with SQL generation from natural instructions due to large schema sizes and complex reasoning. Prior work often focuses on complex, somewhat impractical pipelines using flagship models, while smaller, efficient models remain overlooked. In this work, we explore three multi-agent LLM pipelines, with systematic performance benchmarking across a range of small to large open-source models: (1) Multi-agent discussion pipeline, where agents iteratively critique and refine SQL queries, and a judge synthesizes the final answer; (2) Planner-Coder pipeline, where a thinking model planner generates stepwise SQL generation plans and a coder synthesizes queries; and (3) Coder-Aggregator pipeline, where multiple coders independently generate SQL queries, and a reasoning agent selects the best query. Experiments on the Bird-Bench Mini-Dev set reveal that Multi-Agent discussion can improve small model performance, with up to 10.6% increase in Execution Accuracy for Qwen2.5-7b-Instruct seen after three rounds of discussion. Among the pipelines, the LLM Reasoner-Coder pipeline yields the best results, with DeepSeek-R1-32B and QwQ-32B planners boosting Gemma 3 27B IT accuracy from 52.4% to the highest score of 56.4%. Codes are available at this https URL.

**arXiv ID:** 2511.04153
</details>

<details>
<summary><strong>Collaboration Dynamics and Reliability Challenges of Multi-Agent LLM Systems in Finite Element Analysis</strong> - Chuan Tian, Yilei Zhang - [[pdf]](https://arxiv.org/pdf/2408.13406)</summary>

**Abstract:** Large Language Model (LLM)-based multi-agent systems are increasingly applied to automate computational workflows in science and engineering. However, how inter-agent dynamics influence reasoning quality and verification reliability remains unclear. We study these mechanisms using an AutoGen-based multi-agent framework for linear-elastic Finite Element Analysis (FEA), evaluating seven role configurations across four tasks under a fixed 12-turn conversation limit. From 1,120 controlled trials, we find that collaboration effectiveness depends more on functional complementarity than team size: the three-agent Coder-Executor-Critic configuration uniquely produced physically and visually correct solutions, while adding redundant reviewers reduced success rates. Yet three systematic failure modes persist: (1) affirmation bias, where the Rebuttal agent endorsed rather than challenged outputs (85-92% agreement, including errors); (2) premature consensus caused by redundant reviewers; and (3) a verification-validation gap where executable but physically incorrect code passed undetected. No agent combination successfully validated constitutive relations in complex tasks. Building on theories of functional diversity, role differentiation, and computational validation, we propose actionable design principles: (i) assign complementary agent roles, (ii) enforce multi-level validation (execution, specification, physics), and (iii) prevent early consensus through adversarial or trigger-based interaction control. These findings establish a principled foundation for designing trustworthy LLM collaborations in engineering workflows.

**arXiv ID:** 2408.13406
</details>

<details>
<summary><strong>Building Altruistic and Moral AI Agent with Brain-inspired Emotional Empathy Mechanisms</strong> - Feifei Zhao, Hui Feng, Haibo Tong, Zhengqiang Han, Erliang Lin, Enmeng Lu, Yinqian Sun, Yi Zeng - [[pdf]](https://arxiv.org/pdf/2410.21882)</summary>

**Abstract:** As AI closely interacts with human society, it is crucial to ensure that its behavior is safe, altruistic, and aligned with human ethical and moral values. However, existing research on embedding ethical considerations into AI remains insufficient, and previous external constraints based on principles and rules are inadequate to provide AI with long-term stability and generalization capabilities. Emotional empathy intrinsically motivates altruistic behaviors aimed at alleviating others' negative emotions through emotional sharing and contagion mechanisms. Motivated by this, we draw inspiration from the neural mechanism of human emotional empathy-driven altruistic decision making, and simulate the shared self-other perception-mirroring-empathy neural circuits, to construct a brain-inspired emotional empathy-driven altruistic decision-making model. Here, empathy directly impacts dopamine release to form intrinsic altruistic motivation. The proposed model exhibits consistent altruistic behaviors across three experimental settings: emotional contagion-integrated two-agent altruistic rescue, multi-agent gaming, and robotic emotional empathy interaction scenarios. In-depth analyses validate the positive correlation between empathy levels and altruistic preferences (consistent with psychological behavioral experiment findings), while also demonstrating how interaction partners' empathy levels influence the agent's behavioral patterns. We further test the proposed model's performance and stability in moral dilemmas involving conflicts between self-interest and others' well-being, partially observable environments, and adversarial defense scenarios. This work provides preliminary exploration of human-like empathy-driven altruistic moral decision making, contributing potential perspectives for developing ethically-aligned AI.

**arXiv ID:** 2410.21882
</details>

<details>
<summary><strong>A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning</strong> - Anjie Liu, Jianhong Wang, Samuel Kaski, Jun Wang, Mengyue Yang - [[pdf]](https://arxiv.org/pdf/2510.17697)</summary>

**Abstract:** Steering cooperative multi-agent reinforcement learning (MARL) towards desired outcomes is challenging, particularly when the global guidance from a human on the whole multi-agent system is impractical in a large-scale MARL. On the other hand, designing external mechanisms (e.g., intrinsic rewards and human feedback) to coordinate agents mostly relies on empirical studies, lacking a easy-to-use research tool. In this work, we employ multi-agent influence diagrams (MAIDs) as a graphical framework to address the above issues. First, we introduce the concept of MARL interaction paradigms (orthogonal to MARL learning paradigms), using MAIDs to analyze and visualize both unguided self-organization and global guidance mechanisms in MARL. Then, we design a new MARL interaction paradigm, referred to as the targeted intervention paradigm that is applied to only a single targeted agent, so the problem of global guidance can be mitigated. In implementation, we introduce a causal inference technique, referred to as Pre-Strategy Intervention (PSI), to realize the targeted intervention paradigm. Since MAIDs can be regarded as a special class of causal diagrams, a composite desired outcome that integrates the primary task goal and an additional desired outcome can be achieved by maximizing the corresponding causal effect through the PSI. Moreover, the bundled relevance graph analysis of MAIDs provides a tool to identify whether an MARL learning paradigm is workable under the design of an MARL interaction paradigm. In experiments, we demonstrate the effectiveness of our proposed targeted intervention, and verify the result of relevance graph analysis.

**arXiv ID:** 2510.17697
</details>

<details>
<summary><strong>Toward Autonomous Engineering Design: A Knowledge-Guided Multi-Agent Framework</strong> - Varun Kumar, George Em Karniadakis - [[pdf]](https://arxiv.org/pdf/2511.03179)</summary>

**Abstract:** The engineering design process often demands expertise from multiple domains, leading to complex collaborations and iterative refinements. Traditional methods can be resource-intensive and prone to inefficiencies. To address this, we formalize the engineering design process through a multi-agent AI framework that integrates structured design and review loops. The framework introduces specialized knowledge-driven agents that collaborate to generate and refine design candidates. As an exemplar, we demonstrate its application to the aerodynamic optimization of 4-digit NACA airfoils. The framework consists of three key AI agents: a Graph Ontologist, a Design Engineer, and a Systems Engineer. The Graph Ontologist employs a Large Language Model (LLM) to construct two domain-specific knowledge graphs from airfoil design literature. The Systems Engineer, informed by a human manager, formulates technical requirements that guide design generation and evaluation. The Design Engineer leverages the design knowledge graph and computational tools to propose candidate airfoils meeting these requirements. The Systems Engineer reviews and provides feedback both qualitative and quantitative using its own knowledge graph, forming an iterative feedback loop until a design is validated by the manager. The final design is then optimized to maximize performance metrics such as the lift-to-drag ratio. Overall, this work demonstrates how collaborative AI agents equipped with structured knowledge representations can enhance efficiency, consistency, and quality in the engineering design process.

**arXiv ID:** 2511.03179
</details>

<details>
<summary><strong>ASAP: an Agentic Solution to Auto-optimize Performance of Large-Scale LLM Training</strong> - Yuran Ding, Xinwei Chen, Xiaofan Zhang, Zongwei Zhou - [[pdf]](https://arxiv.org/pdf/2511.03844)</summary>

**Abstract:** Optimizing large-language model (LLM) training on distributed domain-specific accelerator systems presents significant challenges due to its complex optimization space. Existing optimization methods, however, rely on time-consuming manual tuning or resource-intensive black-box searches, which struggle to keep pace with the rapidly evolving LLM domain, leading to slow development and underutilized resources. To address this, we introduce ASAP, an Agentic Solution to Auto-optimize Performance of Large-Scale LLM Training. It is a multi-agent system, featuring Coordinator, Analyzer, and Proposal agents, which integrates LLM reasoning with insights from performance profiling tools, roofline analysis, and a knowledge base of best practices and successful past optimizations from human experts. Our proposed design can automate the diagnosis of performance bottlenecks and recommend optimized sharding configurations with reasoning, thus effectively improving the efficiency of distributed LLM training. Experiments have shown that the ASAP-generated sharding configurations can contribute up to 28% training step time reduction and 1.43 times throughput improvement. When combined with additional optimization from human experts, throughput can be further increased to 2.58 times. The proposed ASAP promises to provide a scalable and explainable methodology for AI-assisted performance engineering in large-scale LLM training.

**arXiv ID:** 2511.03844
</details>

<details>
<summary><strong>Multi-Agent Collaborative Framework For Math Problem Generation</strong> - Kia Karbasi, Kevin Hong, Mohammad Amin Samadi, Gregory Pottie - [[pdf]](https://arxiv.org/pdf/2511.03958)</summary>

**Abstract:** Automatic question generation (AQG) for mathematics education remains an elusive goal for Intelligent Tutoring Systems and educators. While pre-trained transformer-based language models have significantly advanced natural language generation, they often struggle to precisely control problem complexity and cognitive demands. In this paper, we introduce a collaborative multi-agent framework as a novel method of incorporating inference-time computation into AQG. This approach leverages multiple agents that iteratively refine generated question-answer pairs to better balance complexity and cognitive demand. We evaluate the generated questions on five meta-evaluation criteria: relevance, importance, clarity, difficulty matching, answerability, to assess the system's ability to control the required complexity and quality of the questions. Preliminary evaluations show that this collaborative multi-agent framework elevates the quality of generated educational content by fostering a more nuanced balance between cognitive challenge and clarity. These promising outcomes suggest that integrating collaborative multi-agent workflows can yield more controlled, pedagogically valuable content that can help advance automated educational content generation and adaptive learning environments.

**arXiv ID:** 2511.03958
</details>

<details>
<summary><strong>Regret Lower Bounds for Decentralized Multi-Agent Stochastic Shortest Path Problems</strong> - Utkarsh U. Chavan, Prashant Trivedi, Nandyala Hemachandra - [[pdf]](https://arxiv.org/pdf/2511.04594)</summary>

**Abstract:** Multi-agent systems (MAS) are central to applications such as swarm robotics and traffic routing, where agents must coordinate in a decentralized manner to achieve a common objective. Stochastic Shortest Path (SSP) problems provide a natural framework for modeling decentralized control in such settings. While the problem of learning in SSP has been extensively studied in single-agent settings, the decentralized multi-agent variant remains largely unexplored. In this work, we take a step towards addressing that gap. We study decentralized multi-agent SSPs (Dec-MASSPs) under linear function approximation, where the transition dynamics and costs are represented using linear models. Applying novel symmetry-based arguments, we identify the structure of optimal policies. Our main contribution is the first regret lower bound for this setting based on the construction of hard-to-learn instances for any number of agents, $n$. Our regret lower bound of $\Omega(\sqrt{K})$, over $K$ episodes, highlights the inherent learning difficulty in Dec-MASSPs. These insights clarify the learning complexity of decentralized control and can further guide the design of efficient learning algorithms in multi-agent systems.

**arXiv ID:** 2511.04594
</details>

<details>
<summary><strong>Robust Multi-Agent Decision-Making in Finite-Population Games</strong> - Shinkyu Park, Lucas C. D. Bezerra - [[pdf]](https://arxiv.org/pdf/2505.06200)</summary>

**Abstract:** We study the robustness of an agent decision-making model in finite-population games, with a particular focus on the Kullback-Leibler Divergence Regularized Learning (KLD-RL) model. Specifically, we examine how the model's parameters influence the impact of various sources of noise and modeling inaccuracies -- factors commonly encountered in engineering applications of population games -- on agents' decision-making. Our analysis provides insights into how these parameters can be effectively tuned to mitigate such effects. Theoretical results are supported by numerical examples and simulation studies that validate the analysis and illustrate practical strategies for parameter selection.

**arXiv ID:** 2505.06200
</details>

<details>
<summary><strong>Learning Communication Skills in Multi-task Multi-agent Deep Reinforcement Learning</strong> - Changxi Zhu, Mehdi Dastani, Shihan Wang - [[pdf]](https://arxiv.org/pdf/2511.03348)</summary>

**Abstract:** In multi-agent deep reinforcement learning (MADRL), agents can communicate with one another to perform a task in a coordinated manner. When multiple tasks are involved, agents can also leverage knowledge from one task to improve learning in other tasks. In this paper, we propose Multi-task Communication Skills (MCS), a MADRL with communication method that learns and performs multiple tasks simultaneously, with agents interacting through learnable communication protocols. MCS employs a Transformer encoder to encode task-specific observations into a shared message space, capturing shared communication skills among agents. To enhance coordination among agents, we introduce a prediction network that correlates messages with the actions of sender agents in each task. We adapt three multi-agent benchmark environments to multi-task settings, where the number of agents as well as the observation and action spaces vary across tasks. Experimental results demonstrate that MCS achieves better performance than multi-task MADRL baselines without communication, as well as single-task MADRL baselines with and without communication.

**arXiv ID:** 2511.03348
</details>

</details>

<details open>
<summary><h2>Other Agent Research (4 papers)</h2></summary>

<details>
<summary><strong>KnowThyself: An Agentic Assistant for LLM Interpretability</strong> - Suraj Prasai, Mengnan Du, Ying Zhang, Fan Yang - [[pdf]](https://arxiv.org/pdf/2511.03878)</summary>

**Abstract:** We develop KnowThyself, an agentic assistant that advances large language model (LLM) interpretability. Existing tools provide useful insights but remain fragmented and code-intensive. KnowThyself consolidates these capabilities into a chat-based interface, where users can upload models, pose natural language questions, and obtain interactive visualizations with guided explanations. At its core, an orchestrator LLM first reformulates user queries, an agent router further directs them to specialized modules, and the outputs are finally contextualized into coherent explanations. This design lowers technical barriers and provides an extensible platform for LLM inspection. By embedding the whole process into a conversational workflow, KnowThyself offers a robust foundation for accessible LLM interpretability.

**arXiv ID:** 2511.03878
</details>

<details>
<summary><strong>Post-Training LLMs as Better Decision-Making Agents: A Regret-Minimization Approach</strong> - Chanwoo Park, Ziyang Chen, Asuman Ozdaglar, Kaiqing Zhang - [[pdf]](https://arxiv.org/pdf/2511.04393)</summary>

**Abstract:** 

**arXiv ID:** 2511.04393
</details>

<details>
<summary><strong>Leveraging LLM-based agents for social science research: insights from citation network simulations</strong> - Jiarui Ji, Runlin Lei, Xuchen Pan, Zhewei Wei, Hao Sun, Yankai Lin, Xu Chen, Yongzheng Yang, Yaliang Li, Bolin Ding, Ji-Rong Wen - [[pdf]](https://arxiv.org/pdf/2511.03758)</summary>

**Abstract:** The emergence of Large Language Models (LLMs) demonstrates their potential to encapsulate the logic and patterns inherent in human behavior simulation by leveraging extensive web data pre-training. However, the boundaries of LLM capabilities in social simulation remain unclear. To further explore the social attributes of LLMs, we introduce the CiteAgent framework, designed to generate citation networks based on human-behavior simulation with LLM-based agents. CiteAgent successfully captures predominant phenomena in real-world citation networks, including power-law distribution, citational distortion, and shrinking diameter. Building on this realistic simulation, we establish two LLM-based research paradigms in social science: LLM-SE (LLM-based Survey Experiment) and LLM-LE (LLM-based Laboratory Experiment). These paradigms facilitate rigorous analyses of citation network phenomena, allowing us to validate and challenge existing theories. Additionally, we extend the research scope of traditional science of science studies through idealized social experiments, with the simulation experiment results providing valuable insights for real-world academic environments. Our work demonstrates the potential of LLMs for advancing science of science research in social science.

**arXiv ID:** 2511.03758
</details>

<details>
<summary><strong>When Semantics Connect the Swarm: LLM-Driven Fuzzy Control for Cooperative Multi-Robot Underwater Coverage</strong> - Jingzehua Xu, Weihang Zhang, Yangyang Li, Hongmiaoyi Zhang, Guanwen Xie, Jiwei Tang, Shuai Zhang, Yi Li - [[pdf]](https://arxiv.org/pdf/2511.00783)</summary>

**Abstract:** Underwater multi-robot cooperative coverage remains challenging due to partial observability, limited communication, environmental uncertainty, and the lack of access to global localization. To address these issues, this paper presents a semantics-guided fuzzy control framework that couples Large Language Models (LLMs) with interpretable control and lightweight coordination. Raw multimodal observations are compressed by the LLM into compact, human-interpretable semantic tokens that summarize obstacles, unexplored regions, and Objects Of Interest (OOIs) under uncertain perception. A fuzzy inference system with pre-defined membership functions then maps these tokens into smooth and stable steering and gait commands, enabling reliable navigation without relying on global positioning. Then, we further coordinate multiple robots by introducing semantic communication that shares intent and local context in linguistic form, enabling agreement on who explores where while avoiding redundant revisits. Extensive simulations in unknown reef-like environments show that, under limited sensing and communication, the proposed framework achieves robust OOI-oriented navigation and cooperative coverage with improved efficiency and adaptability, narrowing the gap between semantic cognition and distributed underwater control in GPS-denied, map-free conditions.

**arXiv ID:** 2511.00783
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (18 papers)</h2></summary>

<details>
<summary><strong>RLoop: An Self-Improving Framework for Reinforcement Learning with Iterative Policy Initialization</strong> - Zeng Zhiyuan, Jiashuo Liu, Zhangyue Yin, Ge Zhang, Wenhao Huang, Xipeng Qiu - [[pdf]](https://arxiv.org/pdf/2511.04285)</summary>

**Abstract:** While Reinforcement Learning for Verifiable Rewards (RLVR) is powerful for training large reasoning models, its training dynamics harbor a critical challenge: RL overfitting, where models gain training rewards but lose generalization. Our analysis reveals this is driven by policy over-specialization and catastrophic forgetting of diverse solutions generated during training. Standard optimization discards this valuable inter-step policy diversity. To address this, we introduce RLoop, a self-improving framework built on iterative policy initialization. RLoop transforms the standard training process into a virtuous cycle: it first uses RL to explore the solution space from a given policy, then filters the successful trajectories to create an expert dataset. This dataset is used via Rejection-sampling Fine-Tuning (RFT) to refine the initial policy, creating a superior starting point for the next iteration. This loop of exploration and exploitation via iterative re-initialization effectively converts transient policy variations into robust performance gains. Our experiments show RLoop mitigates forgetting and substantially improves generalization, boosting average accuracy by 9% and pass@32 by over 15% compared to vanilla RL.

**arXiv ID:** 2511.04285
</details>

<details>
<summary><strong>GUI-360: A Comprehensive Dataset and Benchmark for Computer-Using Agents</strong> - Jian Mu, Chaoyun Zhang, Chiming Ni, Lu Wang, Bo Qiao, Kartik Mathur, Qianhui Wu, Yuhang Xie, Xiaojun Ma, Mengyu Zhou, Si Qin, Liqun Li, Yu Kang, Minghua Ma, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang - [[pdf]](https://arxiv.org/pdf/2511.04307)</summary>

**Abstract:** We introduce GUI-360$^\circ$, a large-scale, comprehensive dataset and benchmark suite designed to advance computer-using agents (CUAs). CUAs present unique challenges and is constrained by three persistent gaps: a scarcity of real-world CUA tasks, the lack of automated collection-and-annotation pipelines for multi-modal trajectories, and the absence of a unified benchmark that jointly evaluates GUI grounding, screen parsing, and action prediction.
GUI-360$^\circ$ addresses these gaps with an LLM-augmented, largely automated pipeline for query sourcing, environment-template construction, task instantiation, batched execution, and LLM-driven quality filtering. The released corpus contains over 1.2M executed action steps across thousands of trajectories in popular Windows office applications, and includes full-resolution screenshots, accessibility metadata when available, instantiated goals, intermediate reasoning traces, and both successful and failed action trajectories. The dataset supports three canonical tasks, GUI grounding, screen parsing, and action prediction, and a hybrid GUI+API action space that reflects modern agent designs. Benchmarking state-of-the-art vision--language models on GUI-360$^\circ$ reveals substantial out-of-the-box shortcomings in grounding and action prediction; supervised fine-tuning and reinforcement learning yield significant gains but do not close the gap to human-level reliability. We release GUI-360$^\circ$ and accompanying code to facilitate reproducible research and accelerate progress on robust desktop CUAs.
The full dataset has been made public on this https URL.

**arXiv ID:** 2511.04307
</details>

<details>
<summary><strong>Efficient On-Device Agents via Adaptive Context Management</strong> - Sanidhya Vijayvargiya, Rahul Lokesh - [[pdf]](https://arxiv.org/pdf/2511.03728)</summary>

**Abstract:** On-device AI agents offer the potential for personalized, low-latency assistance, but their deployment is fundamentally constrained by limited memory capacity, which restricts usable context. This reduced practical context window creates a trade-off between supporting rich, stateful interactions with complex tool capabilities and maintaining on-device feasibility. We break this trade-off with a framework for context-efficient on-device agents, driven by three synergistic optimizations (1) a dynamic memory system using specialized LoRA adapters to distill conversational history into a compressed, and structured Context State Object; (2) a minimalist serialization format for tool schemas to minimize token overhead per tool; and (3) a just-in-time schema-passing mechanism that loads full tool definitions only upon tool selection. We instantiate this framework by adapting a 3B parameter SLM to context-efficient trajectories and rigorously evaluate it against a conventional baseline on complex user tasks. Our agent matches, or exceeds, the performance of a conventional baseline while dramatically compressing context, achieving more than a 6-fold reduction in initial system prompt context and a 10- to 25-fold reduction in context growth rate based on the interaction verbosity, demonstrating that strategic context management is key to unlocking capable and persistent on-device AI.

**arXiv ID:** 2511.03728
</details>

<details>
<summary><strong>MimiTalk: Revolutionizing Qualitative Research with Dual-Agent AI</strong> - Fengming Liu, Shubin Yu - [[pdf]](https://arxiv.org/pdf/2511.03731)</summary>

**Abstract:** We present MimiTalk, a dual-agent constitutional AI framework designed for scalable and ethical conversational data collection in social science research. The framework integrates a supervisor model for strategic oversight and a conversational model for question generation. We conducted three studies: Study 1 evaluated usability with 20 participants; Study 2 compared 121 AI interviews to 1,271 human interviews from the MediaSum dataset using NLP metrics and propensity score matching; Study 3 involved 10 interdisciplinary researchers conducting both human and AI interviews, followed by blind thematic analysis. Results across studies indicate that MimiTalk reduces interview anxiety, maintains conversational coherence, and outperforms human interviews in information richness, coherence, and stability. AI interviews elicit technical insights and candid views on sensitive topics, while human interviews better capture cultural and emotional nuances. These findings suggest that dual-agent constitutional AI supports effective human-AI collaboration, enabling replicable, scalable and quality-controlled qualitative research.

**arXiv ID:** 2511.03731
</details>

<details>
<summary><strong>Investigating Robot Control Policy Learning for Autonomous X-ray-guided Spine Procedures</strong> - Florence Klitzner, Blanca Inigo, Benjamin D. Killeen, Lalithkumar Seenivasan, Michelle Song, Axel Krieger, Mathias Unberath - [[pdf]](https://arxiv.org/pdf/2511.03882)</summary>

**Abstract:** Imitation learning-based robot control policies are enjoying renewed interest in video-based robotics. However, it remains unclear whether this approach applies to X-ray-guided procedures, such as spine instrumentation. This is because interpretation of multi-view X-rays is complex. We examine opportunities and challenges for imitation policy learning in bi-plane-guided cannula insertion. We develop an in silico sandbox for scalable, automated simulation of X-ray-guided spine procedures with a high degree of realism. We curate a dataset of correct trajectories and corresponding bi-planar X-ray sequences that emulate the stepwise alignment of providers. We then train imitation learning policies for planning and open-loop control that iteratively align a cannula solely based on visual information. This precisely controlled setup offers insights into limitations and capabilities of this method. Our policy succeeded on the first attempt in 68.5% of cases, maintaining safe intra-pedicular trajectories across diverse vertebral levels. The policy generalized to complex anatomy, including fractures, and remained robust to varied initializations. Rollouts on real bi-planar X-rays further suggest that the model can produce plausible trajectories, despite training exclusively in simulation. While these preliminary results are promising, we also identify limitations, especially in entry point precision. Full closed-look control will require additional considerations around how to provide sufficiently frequent feedback. With more robust priors and domain knowledge, such models may provide a foundation for future efforts toward lightweight and CT-free robotic intra-operative spinal navigation.

**arXiv ID:** 2511.03882
</details>

<details>
<summary><strong>Efficient Reinforcement Learning from Human Feedback via Bayesian Preference Inference</strong> - Matteo Cercola, Valeria Capretti, Simone Formentin - [[pdf]](https://arxiv.org/pdf/2511.04286)</summary>

**Abstract:** Learning from human preferences is a cornerstone of aligning machine learning models with subjective human judgments. Yet, collecting such preference data is often costly and time-consuming, motivating the need for more efficient learning paradigms. Two established approaches offer complementary advantages: RLHF scales effectively to high-dimensional tasks such as LLM fine-tuning, while PBO achieves greater sample efficiency through active querying. We propose a hybrid framework that unifies RLHF's scalability with PBO's query efficiency by integrating an acquisition-driven module into the RLHF pipeline, thereby enabling active and sample-efficient preference gathering. We validate the proposed approach on two representative domains: (i) high-dimensional preference optimization and (ii) LLM fine-tuning. Experimental results demonstrate consistent improvements in both sample efficiency and overall performance across these tasks.

**arXiv ID:** 2511.04286
</details>

<details>
<summary><strong>Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data</strong> - Lingkai Kong, Haichuan Wang, Tonghan Wang, Guojun Xiong, Milind Tambe - [[pdf]](https://arxiv.org/pdf/2505.23062)</summary>

**Abstract:** Incorporating pre-collected offline data from a source environment can significantly improve the sample efficiency of reinforcement learning (RL), but this benefit is often challenged by discrepancies between the transition dynamics of the source and target environments. Existing methods typically address this issue by penalizing or filtering out source transitions in high dynamics-gap regions. However, their estimation of the dynamics gap often relies on KL divergence or mutual information, which can be ill-defined when the source and target dynamics have disjoint support. To overcome these limitations, we propose CompFlow, a method grounded in the theoretical connection between flow matching and optimal transport. Specifically, we model the target dynamics as a conditional flow built upon the output distribution of the source-domain flow, rather than learning it directly from a Gaussian prior. This composite structure offers two key advantages: (1) improved generalization for learning target dynamics, and (2) a principled estimation of the dynamics gap via the Wasserstein distance between source and target transitions. Leveraging our principled estimation of the dynamics gap, we further introduce an optimistic active data collection strategy that prioritizes exploration in regions of high dynamics gap, and theoretically prove that it reduces the performance disparity with the optimal policy. Empirically, CompFlow outperforms strong baselines across several RL benchmarks with shifted dynamics.

**arXiv ID:** 2505.23062
</details>

<details>
<summary><strong>Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models</strong> - Junyi Li, Hwee Tou Ng - [[pdf]](https://arxiv.org/pdf/2505.24630)</summary>

**Abstract:** Large language models (LLMs) have significantly advanced in reasoning tasks through reinforcement learning (RL) optimization, achieving impressive capabilities across various challenging benchmarks. However, our empirical analysis reveals a critical drawback: reasoning-oriented RL fine-tuning significantly increases the prevalence of hallucinations. We theoretically analyze the RL training dynamics, identifying high-variance gradient, entropy-induced randomness, and susceptibility to spurious local optima as key factors leading to hallucinations. To address this drawback, we propose Factuality-aware Step-wise Policy Optimization (FSPO), an innovative RL fine-tuning algorithm incorporating explicit factuality verification at each reasoning step. FSPO leverages automated verification against given evidence to dynamically adjust token-level advantage values, incentivizing factual correctness throughout the reasoning process. Experiments across mathematical reasoning and hallucination benchmarks using Qwen2.5 and Llama models demonstrate that FSPO effectively reduces hallucinations while enhancing reasoning accuracy, substantially improving both reliability and performance.

**arXiv ID:** 2505.24630
</details>

<details>
<summary><strong>Rewarding the Journey, Not Just the Destination: A Composite Path and Answer Self-Scoring Reward Mechanism for Test-Time Reinforcement Learning</strong> - Chenwei Tang, Jingyu Xing, Xinyu Liu, Wei Ju, Jiancheng Lv, Fan Zhang, Deng Xiong, Ziyue Qiao - [[pdf]](https://arxiv.org/pdf/2510.17923)</summary>

**Abstract:** Reinforcement Learning (RL) has emerged as a powerful paradigm for advancing Large Language Models (LLMs), achieving remarkable performance in complex reasoning domains such as mathematics and code generation. However, current RL methods face a fundamental scalability bottleneck due to their heavy reliance on human-curated preference data or labeled datasets for reward modeling. To overcome this limitation, we explore RL on unlabeled data where models learn autonomously from continuous experience streams. The core challenge in this setting lies in reliable reward estimation without ground-truth supervision. Existing approaches like Test-Time RL address this through self-consistent consensus, but risk reinforcing incorrect pseudo-labels derived from majority voting. We introduce COMPASS (Composite Path and Answer Self-Scoring), a novel test-time reward mechanism that operates without external supervision. COMPASS integrates two complementary components: the Dual-Calibration Answer Reward (DCAR), which stabilizes training by establishing trustworthy pseudo-labels through confidence and credibility calibration, and the Decisive Path Reward (DPR), which directly optimizes the reasoning process quality beyond mere outcome supervision. By jointly reinforcing trustworthy consensus answers and highly decisive reasoning chains, the COMPASS systematically enhances the model's analytical capabilities. Extensive experiments show that COMPASS achieves significant and consistent performance gains across diverse reasoning tasks and model architectures, advancing a more scalable direction for LLMs to learn from continuous experience.

**arXiv ID:** 2510.17923
</details>

<details>
<summary><strong>From Static to Dynamic: Enhancing Offline-to-Online Reinforcement Learning via Energy-Guided Diffusion Stratification</strong> - Lipeng Zu, Hansong Zhou, Xiaonan Zhang - [[pdf]](https://arxiv.org/pdf/2511.03828)</summary>

**Abstract:** Transitioning from offline to online reinforcement learning (RL) poses critical challenges due to distributional shifts between the fixed behavior policy in the offline dataset and the evolving policy during online learning. Although this issue is widely recognized, few methods attempt to explicitly assess or utilize the distributional structure of the offline data itself, leaving a research gap in adapting learning strategies to different types of samples. To address this challenge, we propose an innovative method, Energy-Guided Diffusion Stratification (StratDiff), which facilitates smoother transitions in offline-to-online RL. StratDiff deploys a diffusion model to learn prior knowledge from the offline dataset. It then refines this knowledge through energy-based functions to improve policy imitation and generate offline-like actions during online fine-tuning. The KL divergence between the generated action and the corresponding sampled action is computed for each sample and used to stratify the training batch into offline-like and online-like subsets. Offline-like samples are updated using offline objectives, while online-like samples follow online learning strategies. We demonstrate the effectiveness of StratDiff by integrating it with off-the-shelf methods Cal-QL and IQL. Extensive empirical evaluations on D4RL benchmarks show that StratDiff significantly outperforms existing methods, achieving enhanced adaptability and more stable performance across diverse RL settings.

**arXiv ID:** 2511.03828
</details>

<details>
<summary><strong>Exchange Policy Optimization Algorithm for Semi-Infinite Safe Reinforcement Learning</strong> - Jiaming Zhang, Yujie Yang, Haoning Wang, Liping Zhang, Shengbo Eben Li - [[pdf]](https://arxiv.org/pdf/2511.04147)</summary>

**Abstract:** Safe reinforcement learning (safe RL) aims to respect safety requirements while optimizing long-term performance. In many practical applications, however, the problem involves an infinite number of constraints, known as semi-infinite safe RL (SI-safe RL). Such constraints typically appear when safety conditions must be enforced across an entire continuous parameter space, such as ensuring adequate resource distribution at every spatial location. In this paper, we propose exchange policy optimization (EPO), an algorithmic framework that achieves optimal policy performance and deterministic bounded safety. EPO works by iteratively solving safe RL subproblems with finite constraint sets and adaptively adjusting the active set through constraint expansion and deletion. At each iteration, constraints with violations exceeding the predefined tolerance are added to refine the policy, while those with zero Lagrange multipliers are removed after the policy update. This exchange rule prevents uncontrolled growth of the working set and supports effective policy training. Our theoretical analysis demonstrates that, under mild assumptions, strategies trained via EPO achieve performance comparable to optimal solutions with global constraint violations strictly remaining within a prescribed bound.

**arXiv ID:** 2511.04147
</details>

<details>
<summary><strong>End-to-End Reinforcement Learning of Koopman Models for eNMPC of an Air Separation Unit</strong> - Daniel Mayfrank, Kayra Dernek, Laura Lang, Alexander Mitsos, Manuel Dahmen - [[pdf]](https://arxiv.org/pdf/2511.04522)</summary>

**Abstract:** With our recently proposed method based on reinforcement learning (Mayfrank et al. (2024), Comput. Chem. Eng. 190), Koopman surrogate models can be trained for optimal performance in specific (economic) nonlinear model predictive control ((e)NMPC) applications. So far, our method has exclusively been demonstrated on a small-scale case study. Herein, we show that our method scales well to a more challenging demand response case study built on a large-scale model of a single-product (nitrogen) air separation unit. Across all numerical experiments, we assume observability of only a few realistically measurable plant variables. Compared to a purely system identification-based Koopman eNMPC, which generates small economic savings but frequently violates constraints, our method delivers similar economic performance while avoiding constraint violations.

**arXiv ID:** 2511.04522
</details>

<details>
<summary><strong>Environment Agnostic Goal-Conditioning, A Study of Reward-Free Autonomous Learning</strong> - Hampus Åström, Elin Anna Topp, Jacek Malec - [[pdf]](https://arxiv.org/pdf/2511.04598)</summary>

**Abstract:** In this paper we study how transforming regular reinforcement learning environments into goal-conditioned environments can let agents learn to solve tasks autonomously and reward-free. We show that an agent can learn to solve tasks by selecting its own goals in an environment-agnostic way, at training times comparable to externally guided reinforcement learning. Our method is independent of the underlying off-policy learning algorithm. Since our method is environment-agnostic, the agent does not value any goals higher than others, leading to instability in performance for individual goals. However, in our experiments, we show that the average goal success rate improves and stabilizes. An agent trained with this method can be instructed to seek any observations made in the environment, enabling generic training of agents prior to specific use cases.

**arXiv ID:** 2511.04598
</details>

<details>
<summary><strong>DeepPAAC: A New Deep Galerkin Method for Principal-Agent Problems</strong> - Michael Ludkovski, Changgen Xie, Zimu Zhu - [[pdf]](https://arxiv.org/pdf/2511.04309)</summary>

**Abstract:** We consider numerical resolution of principal-agent (PA) problems in continuous time. We formulate a generic PA model with continuous and lump payments and a multi-dimensional strategy of the agent. To tackle the resulting Hamilton-Jacobi-Bellman equation with an implicit Hamiltonian we develop a novel deep learning method: the Deep Principal-Agent Actor Critic (DeepPAAC) Actor-Critic algorithm. DeepPAAC is able to handle multi-dimensional states and controls, as well as constraints. We investigate the role of the neural network architecture, training designs, loss functions, etc. on the convergence of the solver, presenting five different case studies.

**arXiv ID:** 2511.04309
</details>

<details>
<summary><strong>Fitting Reinforcement Learning Model to Behavioral Data under Bandits</strong> - Hao Zhu, Jasper Hoffmann, Baohe Zhang, Joschka Boedecker - [[pdf]](https://arxiv.org/pdf/2511.04454)</summary>

**Abstract:** We consider the problem of fitting a reinforcement learning (RL) model to some given behavioral data under a multi-armed bandit environment. These models have received much attention in recent years for characterizing human and animal decision making behavior. We provide a generic mathematical optimization problem formulation for the fitting problem of a wide range of RL models that appear frequently in scientific research applications, followed by a detailed theoretical analysis of its convexity properties. Based on the theoretical results, we introduce a novel solution method for the fitting problem of RL models based on convex relaxation and optimization. Our method is then evaluated in several simulated bandit environments to compare with some benchmark methods that appear in the literature. Numerical results indicate that our method achieves comparable performance to the state-of-the-art, while significantly reducing computation time. We also provide an open-source Python package for our proposed method to empower researchers to apply it in the analysis of their datasets directly, without prior knowledge of convex optimization.

**arXiv ID:** 2511.04454
</details>

<details>
<summary><strong>BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning</strong> - Yitang Li, Zhengyi Luo, Tonghe Zhang, Cunxi Dai, Anssi Kanervisto, Andrea Tirinzoni, Haoyang Weng, Kris Kitani, Mateusz Guzek, Ahmed Touati, Alessandro Lazaric, Matteo Pirotta, Guanya Shi - [[pdf]](https://arxiv.org/pdf/2511.04131)</summary>

**Abstract:** Building Behavioral Foundation Models (BFMs) for humanoid robots has the potential to unify diverse control tasks under a single, promptable generalist policy. However, existing approaches are either exclusively deployed on simulated humanoid characters, or specialized to specific tasks such as tracking. We propose BFM-Zero, a framework that learns an effective shared latent representation that embeds motions, goals, and rewards into a common space, enabling a single policy to be prompted for multiple downstream tasks without retraining. This well-structured latent space in BFM-Zero enables versatile and robust whole-body skills on a Unitree G1 humanoid in the real world, via diverse inference methods, including zero-shot motion tracking, goal reaching, and reward optimization, and few-shot optimization-based adaptation. Unlike prior on-policy reinforcement learning (RL) frameworks, BFM-Zero builds upon recent advancements in unsupervised RL and Forward-Backward (FB) models, which offer an objective-centric, explainable, and smooth latent representation of whole-body motions. We further extend BFM-Zero with critical reward shaping, domain randomization, and history-dependent asymmetric learning to bridge the sim-to-real gap. Those key design choices are quantitatively ablated in simulation. A first-of-its-kind model, BFM-Zero establishes a step toward scalable, promptable behavioral foundation models for whole-body humanoid control.

**arXiv ID:** 2511.04131
</details>

<details>
<summary><strong>Poutine: Vision-Language-Trajectory Pre-Training and Reinforcement Learning Post-Training Enable Robust End-to-End Autonomous Driving</strong> - Luke Rowe, Rodrigue de Schaetzen, Roger Girgis, Christopher Pal, Liam Paull - [[pdf]](https://arxiv.org/pdf/2506.11234)</summary>

**Abstract:** Maintaining good driving behavior in out-of-distribution scenarios remains a critical challenge in autonomous driving. A promising direction is to leverage the generalist knowledge and reasoning capabilities of large-language models by treating unusual driving scenarios as a logical reasoning task. In this work, we present Poutine, a method that uses an off-the-shelf 3B-parameter vision-language model (VLM) - without any additional components - to achieve robust end-to-end autonomous driving via a simple and scalable training recipe. To learn strong base driving capabilities, we first train Poutine-Base using self-supervised next-token prediction over vision, language, and trajectory (VLT) tokens, leveraging both nominal and long-tail driving data. In the second stage, we fine-tune Poutine-Base using Group Relative Policy Optimization (GRPO) with a small set of human preference-labeled examples. We evaluated our approach on the Waymo end-to-end driving benchmark curated for long-tail scenarios. The final Poutine model achieves an RFS of 7.99 on the test set, placing 1st in the 2025 Waymo Vision-Based End-to-End Driving Challenge by a significant margin. Our results suggest that handcrafted tokenizers or custom architectural components added to base VLMs in prior work are not necessary to achieve strong driving performance. Instead, this work highlights the potential of scalable VLT pretraining combined with lightweight RL fine-tuning to enable robust and generalizable autonomous driving.

**arXiv ID:** 2506.11234
</details>

<details>
<summary><strong>Knowledge Graph for Intelligent Generation of Artistic Image Creation: Constructing a New Annotation Hierarchy</strong> - Jia Kaixin, Zhu Kewen, Deng Huanghuang, Qiu Yiwu, Ding Shiying, Ding Chenyang, Ning Zou, Li Zejian - [[pdf]](https://arxiv.org/pdf/2511.03585)</summary>

**Abstract:** Our study aims to establish a unified, systematic, and referable knowledge framework for the annotation of art image datasets, addressing issues of ambiguous definitions and inconsistent results caused by the lack of common standards during the annotation process. To achieve this goal, a hierarchical and systematic art image knowledge graph was constructed. It was developed based on the composition principles of art images, incorporating the Structured Theory of Visual Knowledge proposed by Academician Yunhe Pan in On Visual Knowledge-which states that visual knowledge must achieve precise expression of spatial forms and dynamic relationships through "prototype-category" and "hierarchical structure". Through in-depth review of Chinese and Western art theories and pioneering integration of the Chinese cultural perspective, this graph took shape. The core visual language of art images was deconstructed by this knowledge graph. Meanwhile, the unique spatial theory and symbolic system of Chinese painting were compared with and supplemented by Western art theories. This graph converts qualitative artistic concepts into a clear structured framework. It not only conforms to the cognitive law that "visual knowledge takes precedence over verbal knowledge" in humans but also provides an interpretable and inferential visual knowledge foundation for AI art generation and cross-cultural art analysis. It ensures the high quality and consistency of annotated data, thus offering key support for art intelligence research in the AI 2.0 era.

**arXiv ID:** 2511.03585
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
