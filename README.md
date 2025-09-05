# Agent arXiv Daily

**Last Updated:** 2025-09-05 02:00:01

**Total Papers:** 53

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
<summary><strong>FaMA: LLM-Empowered Agentic Assistant for Consumer-to-Consumer Marketplace</strong> - Yineng Yan, Xidong Wang, Jin Seng Cheng, Ran Hu, Wentao Guan, Nahid Farahmand, Hengte Lin, Yue Li - [[pdf]](https://arxiv.org/pdf/2509.03890)</summary>

**Abstract:** The emergence of agentic AI, powered by Large Language Models (LLMs), marks a paradigm shift from reactive generative systems to proactive, goal-oriented autonomous agents capable of sophisticated planning, memory, and tool use. This evolution presents a novel opportunity to address long-standing challenges in complex digital environments. Core tasks on Consumer-to-Consumer (C2C) e-commerce platforms often require users to navigate complex Graphical User Interfaces (GUIs), making the experience time-consuming for both buyers and sellers. This paper introduces a novel approach to simplify these interactions through an LLM-powered agentic assistant. This agent functions as a new, conversational entry point to the marketplace, shifting the primary interaction model from a complex GUI to an intuitive AI agent. By interpreting natural language commands, the agent automates key high-friction workflows. For sellers, this includes simplified updating and renewal of listings, and the ability to send bulk messages. For buyers, the agent facilitates a more efficient product discovery process through conversational search. We present the architecture for Facebook Marketplace Assistant (FaMA), arguing that this agentic, conversational paradigm provides a lightweight and more accessible alternative to traditional app interfaces, allowing users to manage their marketplace activities with greater efficiency. Experiments show FaMA achieves a 98% task success rate on solving complex tasks on the marketplace and enables up to a 2x speedup on interaction time.

**arXiv ID:** 2509.03890
</details>

<details>
<summary><strong>VoxRole: A Comprehensive Benchmark for Evaluating Speech-Based Role-Playing Agents</strong> - Weihao Wu, Liang Cao, Xinyu Wu, Zhiwei Lin, Rui Niu, Jingbei Li, Zhiyong Wu - [[pdf]](https://arxiv.org/pdf/2509.03940)</summary>

**Abstract:** Recent significant advancements in Large Language Models (LLMs) have greatly propelled the development of Role-Playing Conversational Agents (RPCAs). These systems aim to create immersive user experiences through consistent persona adoption. However, current RPCA research faces dual limitations. First, existing work predominantly focuses on the textual modality, entirely overlooking critical paralinguistic features including intonation, prosody, and rhythm in speech, which are essential for conveying character emotions and shaping vivid identities. Second, the speech-based role-playing domain suffers from a long-standing lack of standardized evaluation benchmarks. Most current spoken dialogue datasets target only fundamental capability assessments, featuring thinly sketched or ill-defined character profiles. Consequently, they fail to effectively quantify model performance on core competencies like long-term persona consistency. To address this critical gap, we introduce VoxRole, the first comprehensive benchmark specifically designed for the evaluation of speech-based RPCAs. The benchmark comprises 13335 multi-turn dialogues, totaling 65.6 hours of speech from 1228 unique characters across 261 movies. To construct this resource, we propose a novel two-stage automated pipeline that first aligns movie audio with scripts and subsequently employs an LLM to systematically build multi-dimensional profiles for each character. Leveraging VoxRole, we conduct a multi-dimensional evaluation of contemporary spoken dialogue models, revealing crucial insights into their respective strengths and limitations in maintaining persona consistency.

**arXiv ID:** 2509.03940
</details>

<details>
<summary><strong>HumAIne-Chatbot: Real-Time Personalized Conversational AI via Reinforcement Learning</strong> - Georgios Makridis, Georgios Fragiadakis, Jorge Oliveira, Tomaz Saraiva, Philip Mavrepis, Georgios Fatouros, Dimosthenis Kyriazis - [[pdf]](https://arxiv.org/pdf/2509.04303)</summary>

**Abstract:** Current conversational AI systems often provide generic, one-size-fits-all interactions that overlook individual user characteristics and lack adaptive dialogue management. To address this gap, we introduce \textbf{HumAIne-chatbot}, an AI-driven conversational agent that personalizes responses through a novel user profiling framework. The system is pre-trained on a diverse set of GPT-generated virtual personas to establish a broad prior over user types. During live interactions, an online reinforcement learning agent refines per-user models by combining implicit signals (e.g. typing speed, sentiment, engagement duration) with explicit feedback (e.g., likes and dislikes). This profile dynamically informs the chatbot dialogue policy, enabling real-time adaptation of both content and style. To evaluate the system, we performed controlled experiments with 50 synthetic personas in multiple conversation domains. The results showed consistent improvements in user satisfaction, personalization accuracy, and task achievement when personalization features were enabled. Statistical analysis confirmed significant differences between personalized and nonpersonalized conditions, with large effect sizes across key metrics. These findings highlight the effectiveness of AI-driven user profiling and provide a strong foundation for future real-world validation.

**arXiv ID:** 2509.04303
</details>

<details>
<summary><strong>Arabic Chatbot Technologies in Education: An Overview</strong> - Hicham Bourhil, Yacine El Younoussi - [[pdf]](https://arxiv.org/pdf/2509.04066)</summary>

**Abstract:** 

**arXiv ID:** 2509.04066
</details>

<details>
<summary><strong>Towards Stable and Personalised Profiles for Lexical Alignment in Spoken Human-Agent Dialogue</strong> - Keara Schaaij, Roel Boumans, Tibor Bosse, Iris Hendrickx - [[pdf]](https://arxiv.org/pdf/2509.04104)</summary>

**Abstract:** Lexical alignment, where speakers start to use similar words across conversation, is known to contribute to successful communication. However, its implementation in conversational agents remains underexplored, particularly considering the recent advancements in large language models (LLMs). As a first step towards enabling lexical alignment in human-agent dialogue, this study draws on strategies for personalising conversational agents and investigates the construction of stable, personalised lexical profiles as a basis for lexical alignment. Specifically, we varied the amounts of transcribed spoken data used for construction as well as the number of items included in the profiles per part-of-speech (POS) category and evaluated profile performance across time using recall, coverage, and cosine similarity metrics. It was shown that smaller and more compact profiles, created after 10 min of transcribed speech containing 5 items for adjectives, 5 items for conjunctions, and 10 items for adverbs, nouns, pronouns, and verbs each, offered the best balance in both performance and data efficiency. In conclusion, this study offers practical insights into constructing stable, personalised lexical profiles, taking into account minimal data requirements, serving as a foundational step toward lexical alignment strategies in conversational agents.

**arXiv ID:** 2509.04104
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (5 papers)</h2></summary>

<details>
<summary><strong>World Model Implanting for Test-time Adaptation of Embodied Agents</strong> - Minjong Yoo, Jinwoo Jang, Sihyung Yoon, Honguk Woo - [[pdf]](https://arxiv.org/pdf/2509.03956)</summary>

**Abstract:** In embodied AI, a persistent challenge is enabling agents to robustly adapt to novel domains without requiring extensive data collection or retraining. To address this, we present a world model implanting framework (WorMI) that combines the reasoning capabilities of large language models (LLMs) with independently learned, domain-specific world models through test-time composition. By allowing seamless implantation and removal of the world models, the embodied agent's policy achieves and maintains cross-domain adaptability. In the WorMI framework, we employ a prototype-based world model retrieval approach, utilizing efficient trajectory-based abstract representation matching, to incorporate relevant models into test-time composition. We also develop a world-wise compound attention method that not only integrates the knowledge from the retrieved world models but also aligns their intermediate representations with the reasoning model's representation within the agent's policy. This framework design effectively fuses domain-specific knowledge from multiple world models, ensuring robust adaptation to unseen domains. We evaluate our WorMI on the VirtualHome and ALFWorld benchmarks, demonstrating superior zero-shot and few-shot performance compared to several LLM-based approaches across a range of unseen domains. These results highlight the frameworks potential for scalable, real-world deployment in embodied agent scenarios where adaptability and data efficiency are essential.

**arXiv ID:** 2509.03956
</details>

<details>
<summary><strong>TAGAL: Tabular Data Generation using Agentic LLM Methods</strong> - Benoît Ronval, Pierre Dupont, Siegfried Nijssen - [[pdf]](https://arxiv.org/pdf/2509.04152)</summary>

**Abstract:** The generation of data is a common approach to improve the performance of machine learning tasks, among which is the training of models for classification. In this paper, we present TAGAL, a collection of methods able to generate synthetic tabular data using an agentic workflow. The methods leverage Large Language Models (LLMs) for an automatic and iterative process that uses feedback to improve the generated data without any further LLM training. The use of LLMs also allows for the addition of external knowledge in the generation process. We evaluate TAGAL across diverse datasets and different aspects of quality for the generated data. We look at the utility of downstream ML models, both by training classifiers on synthetic data only and by combining real and synthetic data. Moreover, we compare the similarities between the real and the generated data. We show that TAGAL is able to perform on par with state-of-the-art approaches that require LLM training and generally outperforms other training-free approaches. These findings highlight the potential of agentic workflow and open new directions for LLM-based data generation methods.

**arXiv ID:** 2509.04152
</details>

<details>
<summary><strong>Cooperative Grasping for Collective Object Transport in Constrained Environments</strong> - David Alvear, George Turkiyyah, Shinkyu Park - [[pdf]](https://arxiv.org/pdf/2509.03638)</summary>

**Abstract:** We propose a novel framework for decision-making in cooperative grasping for two-robot object transport in constrained environments. The core of the framework is a Conditional Embedding (CE) model consisting of two neural networks that map grasp configuration information into an embedding space. The resulting embedding vectors are then used to identify feasible grasp configurations that allow two robots to collaboratively transport an object. To ensure generalizability across diverse environments and object geometries, the neural networks are trained on a dataset comprising a range of environment maps and object shapes. We employ a supervised learning approach with negative sampling to ensure that the learned embeddings effectively distinguish between feasible and infeasible grasp configurations. Evaluation results across a wide range of environments and objects in simulations demonstrate the model's ability to reliably identify feasible grasp configurations. We further validate the framework through experiments on a physical robotic platform, confirming its practical applicability.

**arXiv ID:** 2509.03638
</details>

<details>
<summary><strong>MobileRAG: Enhancing Mobile Agent with Retrieval-Augmented Generation</strong> - Gowen Loo, Chang Liu, Qinghong Yin, Xiang Chen, Jiawei Chen, Jingyuan Zhang, Yu Tian - [[pdf]](https://arxiv.org/pdf/2509.03891)</summary>

**Abstract:** Smartphones have become indispensable in people's daily lives, permeating nearly every aspect of modern society. With the continuous advancement of large language models (LLMs), numerous LLM-based mobile agents have emerged. These agents are capable of accurately parsing diverse user queries and automatically assisting users in completing complex or repetitive operations. However, current agents 1) heavily rely on the comprehension ability of LLMs, which can lead to errors caused by misoperations or omitted steps during tasks, 2) lack interaction with the external environment, often terminating tasks when an app cannot fulfill user queries, and 3) lack memory capabilities, requiring each instruction to reconstruct the interface and being unable to learn from and correct previous mistakes. To alleviate the above issues, we propose MobileRAG, a mobile agents framework enhanced by Retrieval-Augmented Generation (RAG), which includes InterRAG, LocalRAG, and MemRAG. It leverages RAG to more quickly and accurately identify user queries and accomplish complex and long-sequence mobile tasks. Additionally, to more comprehensively assess the performance of MobileRAG, we introduce MobileRAG-Eval, a more challenging benchmark characterized by numerous complex, real-world mobile tasks that require external knowledge assistance. Extensive experimental results on MobileRAG-Eval demonstrate that MobileRAG can easily handle real-world mobile tasks, achieving 10.3\% improvement over state-of-the-art methods with fewer operational steps. Our code is publicly available at: this https URL

**arXiv ID:** 2509.03891
</details>

<details>
<summary><strong>EvolveSignal: A Large Language Model Powered Coding Agent for Discovering Traffic Signal Control Algorithms</strong> - Leizhen Wang, Peibo Duan, Hao Wang, Yue Wang, Jian Xu, Nan Zheng, Zhenliang Ma - [[pdf]](https://arxiv.org/pdf/2509.03335)</summary>

**Abstract:** In traffic engineering, the fixed-time traffic signal control remains widely used for its low cost, stability, and interpretability. However, its design depends on hand-crafted formulas (e.g., Webster) and manual re-timing by engineers to adapt to demand changes, which is labor-intensive and often yields suboptimal results under heterogeneous or congested conditions. This paper introduces the EvolveSignal, a large language models (LLMs) powered coding agent to automatically discover new traffic signal control algorithms. We formulate the problem as program synthesis, where candidate algorithms are represented as Python functions with fixed input-output structures, and iteratively optimized through external evaluations (e.g., a traffic simulator) and evolutionary search. Experiments on a signalized intersection demonstrate that the discovered algorithms outperform Webster's baseline, reducing average delay by 20.1% and average stops by 47.1%. Beyond performance, ablation and incremental analyses reveal that EvolveSignal modifications-such as adjusting cycle length bounds, incorporating right-turn demand, and rescaling green allocations-can offer practically meaningful insights for traffic engineers. This work opens a new research direction by leveraging AI for algorithm design in traffic signal control, bridging program synthesis with transportation engineering.

**arXiv ID:** 2509.03335
</details>

</details>

<details open>
<summary><h2>LLM Agents (8 papers)</h2></summary>

<details>
<summary><strong>Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents</strong> - Davide Paglieri, Bartłomiej Cupiał, Jonathan Cook, Ulyana Piterbarg, Jens Tuyls, Edward Grefenstette, Jakob Nicolaus Foerster, Jack Parker-Holder, Tim Rocktäschel - [[pdf]](https://arxiv.org/pdf/2509.03581)</summary>

**Abstract:** Training large language models (LLMs) to reason via reinforcement learning (RL) significantly improves their problem-solving capabilities. In agentic settings, existing methods like ReAct prompt LLMs to explicitly plan before every action; however, we demonstrate that always planning is computationally expensive and degrades performance on long-horizon tasks, while never planning further limits performance. To address this, we introduce a conceptual framework formalizing dynamic planning for LLM agents, enabling them to flexibly decide when to allocate test-time compute for planning. We propose a simple two-stage training pipeline: (1) supervised fine-tuning on diverse synthetic data to prime models for dynamic planning, and (2) RL to refine this capability in long-horizon environments. Experiments on the Crafter environment show that dynamic planning agents trained with this approach are more sample-efficient and consistently achieve more complex objectives. Additionally, we demonstrate that these agents can be effectively steered by human-written plans, surpassing their independent capabilities. To our knowledge, this work is the first to explore training LLM agents for dynamic test-time compute allocation in sequential decision-making tasks, paving the way for more efficient, adaptive, and controllable agentic systems.

**arXiv ID:** 2509.03581
</details>

<details>
<summary><strong>Are LLM Agents Behaviorally Coherent? Latent Profiles for Social Simulation</strong> - James Mooney, Josef Woldense, Zheng Robert Jia, Shirley Anugrah Hayati, My Ha Nguyen, Vipul Raheja, Dongyeop Kang - [[pdf]](https://arxiv.org/pdf/2509.03736)</summary>

**Abstract:** The impressive capabilities of Large Language Models (LLMs) have fueled the notion that synthetic agents can serve as substitutes for real participants in human-subject research. In an effort to evaluate the merits of this claim, social science researchers have largely focused on whether LLM-generated survey data corresponds to that of a human counterpart whom the LLM is prompted to represent. In contrast, we address a more fundamental question: Do agents maintain internal consistency, retaining similar behaviors when examined under different experimental settings? To this end, we develop a study designed to (a) reveal the agent's internal state and (b) examine agent behavior in a basic dialogue setting. This design enables us to explore a set of behavioral hypotheses to assess whether an agent's conversation behavior is consistent with what we would expect from their revealed internal state. Our findings on these hypotheses show significant internal inconsistencies in LLMs across model families and at differing model sizes. Most importantly, we find that, although agents may generate responses matching those of their human counterparts, they fail to be internally consistent, representing a critical gap in their capabilities to accurately substitute for real participants in human-subject research. Our simulation code and data are publicly accessible.

**arXiv ID:** 2509.03736
</details>

<details>
<summary><strong>Meta-Policy Reflexion: Reusable Reflective Memory and Rule Admissibility for Resource-Efficient LLM Agent</strong> - Chunlong Wu, Zhibo Qu - [[pdf]](https://arxiv.org/pdf/2509.03990)</summary>

**Abstract:** Large language model (LLM) agents achieve impressive single-task performance but commonly exhibit repeated failures, inefficient exploration, and limited cross-task adaptability. Existing reflective strategies (e.g., Reflexion, ReAct) improve per-episode behavior but typically produce ephemeral, task-specific traces that are not reused across tasks. Reinforcement-learning based alternatives can produce transferable policies but require substantial parameter updates and compute. In this work we introduce Meta-Policy Reflexion (MPR): a hybrid framework that consolidates LLM-generated reflections into a structured, predicate-like Meta-Policy Memory (MPM) and applies that memory at inference time through two complementary mechanisms soft memory-guided decoding and hard rule admissibility checks(HAC). MPR (i) externalizes reusable corrective knowledge without model weight updates, (ii) enforces domain constraints to reduce unsafe or invalid actions, and (iii) retains the adaptability of language-based reflection. We formalize the MPM representation, present algorithms for update and decoding, and validate the approach in a text-based agent environment following the experimental protocol described in the provided implementation (AlfWorld-based). Empirical results reported in the supplied material indicate consistent gains in execution accuracy and robustness when compared to Reflexion baselines; rule admissibility further improves stability. We analyze mechanisms that explain these gains, discuss scalability and failure modes, and outline future directions for multimodal and multi?agent extensions.

**arXiv ID:** 2509.03990
</details>

<details>
<summary><strong>EvoEmo: Towards Evolved Emotional Policies for LLM Agents in Multi-Turn Negotiation</strong> - Yunbo Long, Liming Xu, Lukas Beckenbauer, Yuhan Liu, Alexandra Brintrup - [[pdf]](https://arxiv.org/pdf/2509.04310)</summary>

**Abstract:** Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) has demonstrated that agents can engage in \textit{complex}, \textit{multi-turn} negotiations, opening new avenues for agentic AI. However, existing LLM agents largely overlook the functional role of emotions in such negotiations, instead generating passive, preference-driven emotional responses that make them vulnerable to manipulation and strategic exploitation by adversarial counterparts. To address this gap, we present EvoEmo, an evolutionary reinforcement learning framework that optimizes dynamic emotional expression in negotiations. EvoEmo models emotional state transitions as a Markov Decision Process and employs population-based genetic optimization to evolve high-reward emotion policies across diverse negotiation scenarios. We further propose an evaluation framework with two baselines -- vanilla strategies and fixed-emotion strategies -- for benchmarking emotion-aware negotiation. Extensive experiments and ablation studies show that EvoEmo consistently outperforms both baselines, achieving higher success rates, higher efficiency, and increased buyer savings. This findings highlight the importance of adaptive emotional expression in enabling more effective LLM agents for multi-turn negotiation.

**arXiv ID:** 2509.04310
</details>

<details>
<summary><strong>Kolb-Based Experiential Learning for Generalist Agents with Human-Level Kaggle Data Science Performance</strong> - Antoine Grosnit, Alexandre Maraval, Refinath S N, Zichao Zhao, James Dora, Giuseppe Paolo, Albert Thomas, Jonas Gonzalez, Abhineet Kumar, Khyati Khandelwal, Abdelhakim Benechehab, Hamza Cherkaoui, Youssef Attia El-Hili, Kun Shao, Jianye Hao, Jun Yao, Balázs Kégl, Jun Wang - [[pdf]](https://arxiv.org/pdf/2411.03562)</summary>

**Abstract:** Human expertise emerges through iterative cycles of interaction, reflection, and internal model updating, which are central to cognitive theories such as Kolb's experiential learning and Vygotsky's zone of proximal development. In contrast, current AI systems, particularly LLM agents, rely on static pre-training or rigid workflows, lacking mechanisms for continual adaptation. Recent studies identified early cognitive traits in LLM agents (reflection, revision, and self-correction) suggesting foundational elements of human-like experiential learning. Thus the key question: Can we design LLM agents capable of structured, cognitively grounded learning similar to human processes? In response, we propose a computational framework of Kolb's learning cycle with Vygotsky's ZPD for autonomous agents. Our architecture separates extrinsic (environment interaction) and intrinsic (internal reflection/abstraction) functions, enabling cognitively grounded scaffolded learning, where the agent initially learns within structured environments, followed by open-ended generalisation. This approach empowers agents to master complex tasks ; domains that traditional fine-tuning or simple reflective methods could not tackle effectively. Its potential is powerfully demonstrated via direct comparison with humans in real-world Kaggle data science competitions. Learning fully automated data science code generation across 81 tasks, our system, Agent K, demonstrated the ability to perform the entire workflow autonomously, achieving an Elo-MMR score of 1694, beyond median score of the Kaggle Masters (the top 2% among 200,000 users) of our study. With 9 gold, 8 silver, and 12 bronze medals level performance - including 4 gold and 4 silver on prize-awarding competitions - Agent K is the 1st AI system to successfully integrate Kolb- and Vygotsky-inspired human cognitive learning, marking a major step toward generalist AI.

**arXiv ID:** 2411.03562
</details>

<details>
<summary><strong>DynaSaur: Large Language Agents Beyond Predefined Actions</strong> - Dang Nguyen, Viet Dac Lai, Seunghyun Yoon, Ryan A. Rossi, Handong Zhao, Ruiyi Zhang, Puneet Mathur, Nedim Lipka, Yu Wang, Trung Bui, Franck Dernoncourt, Tianyi Zhou - [[pdf]](https://arxiv.org/pdf/2411.01747)</summary>

**Abstract:** Existing LLM agent systems typically select actions from a fixed and predefined set at every step. While this approach is effective in closed, narrowly scoped environments, it presents two major challenges for real-world, open-ended scenarios: (1) it significantly restricts the planning and acting capabilities of LLM agents, and (2) it requires substantial human effort to enumerate and implement all possible actions, which is impractical in complex environments with a vast number of potential actions. To address these limitations, we propose an LLM agent framework that can dynamically create and compose actions as needed. In this framework, the agent interacts with its environment by generating and executing programs written in a general-purpose programming language. Moreover, generated actions are accumulated over time for future reuse. Our extensive experiments across multiple benchmarks show that this framework significantly improves flexibility and outperforms prior methods that rely on a fixed action set. Notably, it enables LLM agents to adapt and recover in scenarios where predefined actions are insufficient or fail due to unforeseen edge cases. Our code can be found in this https URL.

**arXiv ID:** 2411.01747
</details>

<details>
<summary><strong>HamRaz: A Culture-Based Persian Conversation Dataset for Person-Centered Therapy Using LLM Agents</strong> - Mohammad Amin Abbasi, Farnaz Sadat Mirnezami, Ali Neshati, Hassan Naderi - [[pdf]](https://arxiv.org/pdf/2502.05982)</summary>

**Abstract:** We present HamRaz, a culturally adapted Persian-language dataset for AI-assisted mental health support, grounded in Person-Centered Therapy (PCT). To reflect real-world therapeutic challenges, we combine script-based dialogue with adaptive large language models (LLM) role-playing, capturing the ambiguity and emotional nuance of Persian-speaking clients. We introduce HamRazEval, a dual-framework for assessing conversational and therapeutic quality using General Metrics and specialized psychological relationship measures. Human evaluations show HamRaz outperforms existing baselines in empathy, coherence, and realism. This resource contributes to the Digital Humanities by bridging language, culture, and mental health in underrepresented communities.

**arXiv ID:** 2502.05982
</details>

<details>
<summary><strong>EQ-Knight: A Memory-Augmented LLM Agent for Strategic Affective Gaming in Debt Recovery</strong> - Yunbo Long, Yuhan Liu, Liming Xu, Alexandra Brintrup - [[pdf]](https://arxiv.org/pdf/2503.21080)</summary>

**Abstract:** Large language model-based chatbots have enhanced engagement in financial negotiations, but their overreliance on passive empathy introduces critical risks in credit collection. While empathy-driven approaches preserve client satisfaction in benign cases, they fail catastrophically against dishonest debtors--individuals who exploit conciliatory tactics to manipulate terms or evade repayment. Blindly prioritizing "customer experience" in such scenarios leads to creditor vulnerabilities: revenue leakage, moral hazard, and systemic exploitation. To address this, we propose EQ-Knight, an LLM agent that dynamically optimizes emotional strategy to defend creditor interests. Unlike naive empathy-centric bots, EQ-Knight integrates emotion memory and game-theoretic reasoning, powered by a Hidden Markov Model (HMM) to track and predict debtor emotional states. By analyzing both real-time and historical emotional cues, EQ-Knight strategically counters negative emotions (e.g., aggression, feigned distress) while preserving productive debtor relationships. Experiments demonstrate EQ-Knight's superiority over conventional LLM negotiators: it achieves a 32\% reduction in concession losses without compromising recovery rates, particularly in adversarial cases where debtors weaponize negative emotions (e.g., intimidation, guilt-tripping) to coerce concessions. For credit agencies, EQ-Knight transforms LLMs from high-risk "people-pleasers" into strategic emotion-defenders--balancing emotional intelligence with tactical rigor to enforce accountability and deter exploitation.

**arXiv ID:** 2503.21080
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (10 papers)</h2></summary>

<details>
<summary><strong>PG-Agent: An Agent Powered by Page Graph</strong> - Weizhi Chen, Ziwei Wang, Leyang Yang, Sheng Zhou, Xiaoxuan Tang, Jiajun Bu, Yong Li, Wei Jiang - [[pdf]](https://arxiv.org/pdf/2509.03536)</summary>

**Abstract:** Graphical User Interface (GUI) agents possess significant commercial and social value, and GUI agents powered by advanced multimodal large language models (MLLMs) have demonstrated remarkable potential. Currently, existing GUI agents usually utilize sequential episodes of multi-step operations across pages as the prior GUI knowledge, which fails to capture the complex transition relationship between pages, making it challenging for the agents to deeply perceive the GUI environment and generalize to new scenarios. Therefore, we design an automated pipeline to transform the sequential episodes into page graphs, which explicitly model the graph structure of the pages that are naturally connected by actions. To fully utilize the page graphs, we further introduce Retrieval-Augmented Generation (RAG) technology to effectively retrieve reliable perception guidelines of GUI from them, and a tailored multi-agent framework PG-Agent with task decomposition strategy is proposed to be injected with the guidelines so that it can generalize to unseen scenarios. Extensive experiments on various benchmarks demonstrate the effectiveness of PG-Agent, even with limited episodes for page graph construction.

**arXiv ID:** 2509.03536
</details>

<details>
<summary><strong>Learning to Deliberate: Meta-policy Collaboration for Agentic LLMs with Multi-agent Reinforcement Learning</strong> - Wei Yang, Jesse Thomason - [[pdf]](https://arxiv.org/pdf/2509.03817)</summary>

**Abstract:** Multi-agent systems of large language models (LLMs) show promise for complex reasoning, but their effectiveness is often limited by fixed collaboration protocols. These frameworks typically focus on macro-level orchestration while overlooking agents' internal deliberative capabilities. This critical meta-cognitive blindspot treats agents as passive executors unable to adapt their strategy based on internal cognitive states like uncertainty or confidence. We introduce the Meta-Policy Deliberation Framework (MPDF), where agents learn a decentralized policy over a set of high-level meta-cognitive actions: Persist, Refine, and Concede. To overcome the instability of traditional policy gradients in this setting, we develop SoftRankPO, a novel reinforcement learning algorithm. SoftRankPO stabilizes training by shaping advantages based on the rank of rewards mapped through smooth normal quantiles, making the learning process robust to reward variance. Experiments show that MPDF with SoftRankPO achieves a a 4-5% absolute gain in average accuracy across five mathematical and general reasoning benchmarks compared to six state-of-the-art heuristic and learning-based multi-agent reasoning algorithms. Our work presents a paradigm for learning adaptive, meta-cognitive policies for multi-agent LLM systems, shifting the focus from designing fixed protocols to learning dynamic, deliberative strategies.

**arXiv ID:** 2509.03817
</details>

<details>
<summary><strong>Psychologically Enhanced AI Agents</strong> - Maciej Besta, Shriram Chandran, Robert Gerstenberger, Mathis Lindner, Marcin Chrapek, Sebastian Hermann Martschat, Taraneh Ghandi, Patrick Iff, Hubert Niewiadomski, Piotr Nyczyk, Jürgen Müller, Torsten Hoefler - [[pdf]](https://arxiv.org/pdf/2509.04343)</summary>

**Abstract:** We introduce MBTI-in-Thoughts, a framework for enhancing the effectiveness of Large Language Model (LLM) agents through psychologically grounded personality conditioning. Drawing on the Myers-Briggs Type Indicator (MBTI), our method primes agents with distinct personality archetypes via prompt engineering, enabling control over behavior along two foundational axes of human psychology, cognition and affect. We show that such personality priming yields consistent, interpretable behavioral biases across diverse tasks: emotionally expressive agents excel in narrative generation, while analytically primed agents adopt more stable strategies in game-theoretic settings. Our framework supports experimenting with structured multi-agent communication protocols and reveals that self-reflection prior to interaction improves cooperation and reasoning quality. To ensure trait persistence, we integrate the official 16Personalities test for automated verification. While our focus is on MBTI, we show that our approach generalizes seamlessly to other psychological frameworks such as Big Five, HEXACO, or Enneagram. By bridging psychological theory and LLM behavior design, we establish a foundation for psychologically enhanced AI agents without any fine-tuning.

**arXiv ID:** 2509.04343
</details>

<details>
<summary><strong>SAMVAD: A Multi-Agent System for Simulating Judicial Deliberation Dynamics in India</strong> - Prathamesh Devadiga, Omkaar Jayadev Shetty, Pooja Agarwal - [[pdf]](https://arxiv.org/pdf/2509.03793)</summary>

**Abstract:** Understanding the complexities of judicial deliberation is crucial for assessing the efficacy and fairness of a justice system. However, empirical studies of judicial panels are constrained by significant ethical and practical barriers. This paper introduces SAMVAD, an innovative Multi-Agent System (MAS) designed to simulate the deliberation process within the framework of the Indian justice system.
Our system comprises agents representing key judicial roles: a Judge, a Prosecution Counsel, a Defense Counsel, and multiple Adjudicators (simulating a judicial bench), all powered by large language models (LLMs). A primary contribution of this work is the integration of Retrieval-Augmented Generation (RAG), grounded in a domain-specific knowledge base of landmark Indian legal documents, including the Indian Penal Code and the Constitution of India. This RAG functionality enables the Judge and Counsel agents to generate legally sound instructions and arguments, complete with source citations, thereby enhancing both the fidelity and transparency of the simulation.
The Adjudicator agents engage in iterative deliberation rounds, processing case facts, legal instructions, and arguments to reach a consensus-based verdict. We detail the system architecture, agent communication protocols, the RAG pipeline, the simulation workflow, and a comprehensive evaluation plan designed to assess performance, deliberation quality, and outcome consistency.
This work provides a configurable and explainable MAS platform for exploring legal reasoning and group decision-making dynamics in judicial simulations, specifically tailored to the Indian legal context and augmented with verifiable legal grounding via RAG.

**arXiv ID:** 2509.03793
</details>

<details>
<summary><strong>MAGneT: Coordinated Multi-Agent Generation of Synthetic Multi-Turn Mental Health Counseling Sessions</strong> - Aishik Mandal, Tanmoy Chakraborty, Iryna Gurevych - [[pdf]](https://arxiv.org/pdf/2509.04183)</summary>

**Abstract:** The growing demand for scalable psychological counseling highlights the need for fine-tuning open-source Large Language Models (LLMs) with high-quality, privacy-compliant data, yet such data remains scarce. Here we introduce MAGneT, a novel multi-agent framework for synthetic psychological counseling session generation that decomposes counselor response generation into coordinated sub-tasks handled by specialized LLM agents, each modeling a key psychological technique. Unlike prior single-agent approaches, MAGneT better captures the structure and nuance of real counseling. In addition, we address inconsistencies in prior evaluation protocols by proposing a unified evaluation framework integrating diverse automatic and expert metrics. Furthermore, we expand the expert evaluations from four aspects of counseling in previous works to nine aspects, enabling a more thorough and robust assessment of data quality. Empirical results show that MAGneT significantly outperforms existing methods in quality, diversity, and therapeutic alignment of the generated counseling sessions, improving general counseling skills by 3.2% and CBT-specific skills by 4.3% on average on cognitive therapy rating scale (CTRS). Crucially, experts prefer MAGneT-generated sessions in 77.2% of cases on average across all aspects. Moreover, fine-tuning an open-source model on MAGneT-generated sessions shows better performance, with improvements of 6.3% on general counseling skills and 7.3% on CBT-specific skills on average on CTRS over those fine-tuned with sessions generated by baseline methods. We also make our code and data public.

**arXiv ID:** 2509.04183
</details>

<details>
<summary><strong>Theory of Mind Using Active Inference: A Framework for Multi-Agent Cooperation</strong> - Riddhi J. Pitliya, Ozan Çatal, Toon Van de Maele, Corrado Pezzato, Tim Verbelen - [[pdf]](https://arxiv.org/pdf/2508.00401)</summary>

**Abstract:** Theory of Mind (ToM) -- the ability to understand that others can have differing knowledge and goals -- enables agents to reason about others' beliefs while planning their own actions. We present a novel approach to multi-agent cooperation by implementing ToM within active inference. Unlike previous active inference approaches to multi-agent cooperation, our method neither relies on task-specific shared generative models nor requires explicit communication. In our framework, ToM-equipped agents maintain distinct representations of their own and others' beliefs and goals. ToM agents then use an extended and adapted version of the sophisticated inference tree-based planning algorithm to systematically explore joint policy spaces through recursive reasoning. We evaluate our approach through collision avoidance and foraging simulations. Results suggest that ToM agents cooperate better compared to non-ToM counterparts by being able to avoid collisions and reduce redundant efforts. Crucially, ToM agents accomplish this by inferring others' beliefs solely from observable behaviour and considering them when planning their own actions. Our approach shows potential for generalisable and scalable multi-agent systems while providing computational insights into ToM mechanisms.

**arXiv ID:** 2508.00401
</details>

<details>
<summary><strong>Are LLM Agents the New RPA? A Comparative Study with RPA Across Enterprise Workflows</strong> - Petr Průcha, Michaela Matoušková, Jan Strnad - [[pdf]](https://arxiv.org/pdf/2509.04198)</summary>

**Abstract:** The emergence of large language models (LLMs) has introduced a new paradigm in automation: LLM agents or Agentic Automation with Computer Use (AACU). Unlike traditional Robotic Process Automation (RPA), which relies on rule-based workflows and scripting, AACU enables intelligent agents to perform tasks through natural language instructions and autonomous interaction with user interfaces. This study investigates whether AACU can serve as a viable alternative to RPA in enterprise workflow automation. We conducted controlled experiments across three standard RPA challenges data entry, monitoring, and document extraction comparing RPA (via UiPath) and AACU (via Anthropic's Computer Use Agent) in terms of speed, reliability, and development effort. Results indicate that RPA outperforms AACU in execution speed and reliability, particularly in repetitive, stable environments. However, AACU significantly reduces development time and adapts more flexibly to dynamic interfaces. While current AACU implementations are not yet production-ready, their promise in rapid prototyping and lightweight automation is evident. Future research should explore multi-agent orchestration, hybrid RPA-AACU architectures, and more robust evaluation across industries and platforms.

**arXiv ID:** 2509.04198
</details>

<details>
<summary><strong>SAFE--MA--RRT: Multi-Agent Motion Planning with Data-Driven Safety Certificates</strong> - Babak Esmaeili, Hamidreza Modares - [[pdf]](https://arxiv.org/pdf/2509.04413)</summary>

**Abstract:** This paper proposes a fully data-driven motion-planning framework for homogeneous linear multi-agent systems that operate in shared, obstacle-filled workspaces without access to explicit system models. Each agent independently learns its closed-loop behavior from experimental data by solving convex semidefinite programs that generate locally invariant ellipsoids and corresponding state-feedback gains. These ellipsoids, centered along grid-based waypoints, certify the dynamic feasibility of short-range transitions and define safe regions of operation. A sampling-based planner constructs a tree of such waypoints, where transitions are allowed only when adjacent ellipsoids overlap, ensuring invariant-to-invariant transitions and continuous safety. All agents expand their trees simultaneously and are coordinated through a space-time reservation table that guarantees inter-agent safety by preventing simultaneous occupancy and head-on collisions. Each successful edge in the tree is equipped with its own local controller, enabling execution without re-solving optimization problems at runtime. The resulting trajectories are not only dynamically feasible but also provably safe with respect to both environmental constraints and inter-agent collisions. Simulation results demonstrate the effectiveness of the approach in synthesizing synchronized, safe trajectories for multiple agents under shared dynamics and constraints, using only data and convex optimization tools.

**arXiv ID:** 2509.04413
</details>

<details>
<summary><strong>AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?</strong> - Guibin Zhang, Junhao Wang, Junjie Chen, Wangchunshu Zhou, Kun Wang, Shuicheng Yan - [[pdf]](https://arxiv.org/pdf/2509.03312)</summary>

**Abstract:** Large Language Model (LLM)-based agentic systems, often comprising multiple models, complex tool invocations, and orchestration protocols, substantially outperform monolithic agents. Yet this very sophistication amplifies their fragility, making them more prone to system failure. Pinpointing the specific agent or step responsible for an error within long execution traces defines the task of agentic system failure attribution. Current state-of-the-art reasoning LLMs, however, remain strikingly inadequate for this challenge, with accuracy generally below 10%. To address this gap, we propose AgenTracer, the first automated framework for annotating failed multi-agent trajectories via counterfactual replay and programmed fault injection, producing the curated dataset TracerTraj. Leveraging this resource, we develop AgenTracer-8B, a lightweight failure tracer trained with multi-granular reinforcement learning, capable of efficiently diagnosing errors in verbose multi-agent interactions. On the Who&When benchmark, AgenTracer-8B outperforms giant proprietary LLMs like Gemini-2.5-Pro and Claude-4-Sonnet by up to 18.18%, setting a new standard in LLM agentic failure attribution. More importantly, AgenTracer-8B delivers actionable feedback to off-the-shelf multi-agent systems like MetaGPT and MaAS with 4.8-14.2% performance gains, empowering self-correcting and self-evolving agentic AI.

**arXiv ID:** 2509.03312
</details>

<details>
<summary><strong>A Comprehensive Review of Multi-Agent Reinforcement Learning in Video Games</strong> - Zhengyang Li, Qijin Ji, Xinghong Ling, Quan Liu - [[pdf]](https://arxiv.org/pdf/2509.03682)</summary>

**Abstract:** Recent advancements in multi-agent reinforcement learning (MARL) have demonstrated its application potential in modern games. Beginning with foundational work and progressing to landmark achievements such as AlphaStar in StarCraft II and OpenAI Five in Dota 2, MARL has proven capable of achieving superhuman performance across diverse game environments through techniques like self-play, supervised learning, and deep reinforcement learning. With its growing impact, a comprehensive review has become increasingly important in this field. This paper aims to provide a thorough examination of MARL's application from turn-based two-agent games to real-time multi-agent video games including popular genres such as Sports games, First-Person Shooter (FPS) games, Real-Time Strategy (RTS) games and Multiplayer Online Battle Arena (MOBA) games. We further analyze critical challenges posed by MARL in video games, including nonstationary, partial observability, sparse rewards, team coordination, and scalability, and highlight successful implementations in games like Rocket League, Minecraft, Quake III Arena, StarCraft II, Dota 2, Honor of Kings, etc. This paper offers insights into MARL in video game AI systems, proposes a novel method to estimate game complexity, and suggests future research directions to advance MARL and its applications in game development, inspiring further innovation in this rapidly evolving field.

**arXiv ID:** 2509.03682
</details>

</details>

<details open>
<summary><h2>Other Agent Research (3 papers)</h2></summary>

<details>
<summary><strong>Leveraging LLM-Based Agents for Intelligent Supply Chain Planning</strong> - Yongzhi Qi, Jiaheng Yin, Jianshen Zhang, Dongyang Geng, Zhengyu Chen, Hao Hu, Wei Qi, Zuo-Jun Max Shen - [[pdf]](https://arxiv.org/pdf/2509.03811)</summary>

**Abstract:** In supply chain management, planning is a critical concept. The movement of physical products across different categories, from suppliers to warehouse management, to sales, and logistics transporting them to customers, entails the involvement of many entities. It covers various aspects such as demand forecasting, inventory management, sales operations, and replenishment. How to collect relevant data from an e-commerce platform's perspective, formulate long-term plans, and dynamically adjust them based on environmental changes, while ensuring interpretability, efficiency, and reliability, is a practical and challenging problem. In recent years, the development of AI technologies, especially the rapid progress of large language models, has provided new tools to address real-world issues. In this work, we construct a Supply Chain Planning Agent (SCPA) framework that can understand domain knowledge, comprehend the operator's needs, decompose tasks, leverage or create new tools, and return evidence-based planning reports. We deploy this framework in this http URL's real-world scenario, demonstrating the feasibility of LLM-agent applications in the supply chain. It effectively reduced labor and improved accuracy, stock availability, and other key metrics.

**arXiv ID:** 2509.03811
</details>

<details>
<summary><strong>INGRID: Intelligent Generative Robotic Design Using Large Language Models</strong> - Guanglu Jia, Ceng Zhang, Gregory S. Chirikjian - [[pdf]](https://arxiv.org/pdf/2509.03842)</summary>

**Abstract:** The integration of large language models (LLMs) into robotic systems has accelerated progress in embodied artificial intelligence, yet current approaches remain constrained by existing robotic architectures, particularly serial mechanisms. This hardware dependency fundamentally limits the scope of robotic intelligence. Here, we present INGRID (Intelligent Generative Robotic Design), a framework that enables the automated design of parallel robotic mechanisms through deep integration with reciprocal screw theory and kinematic synthesis methods. We decompose the design challenge into four progressive tasks: constraint analysis, kinematic joint generation, chain construction, and complete mechanism design. INGRID demonstrates the ability to generate novel parallel mechanisms with both fixed and variable mobility, discovering kinematic configurations not previously documented in the literature. We validate our approach through three case studies demonstrating how INGRID assists users in designing task-specific parallel robots based on desired mobility requirements. By bridging the gap between mechanism theory and machine learning, INGRID enables researchers without specialized robotics training to create custom parallel mechanisms, thereby decoupling advances in robotic intelligence from hardware constraints. This work establishes a foundation for mechanism intelligence, where AI systems actively design robotic hardware, potentially transforming the development of embodied AI systems.

**arXiv ID:** 2509.03842
</details>

<details>
<summary><strong>Segmented Trajectory Optimization for Autonomous Parking in Unstructured Environments</strong> - Hang Yu, Renjie Li - [[pdf]](https://arxiv.org/pdf/2504.05041)</summary>

**Abstract:** This paper presents a Segmented Trajectory Optimization (STO) method for autonomous parking, which refines an initial trajectory into a dynamically feasible and collision-free one using an iterative SQP-based approach. STO maintains the maneuver strategy of the high-level global planner while allowing curvature discontinuities at switching points to improve maneuver efficiency. To ensure safety, a convex corridor is constructed via GJK-accelerated ellipse shrinking and expansion, serving as safety constraints in each iteration. Numerical simulations in perpendicular and reverse-angled parking scenarios demonstrate that STO enhances maneuver efficiency while ensuring safety. Moreover, computational performance confirms its practicality for real-world applications.

**arXiv ID:** 2504.05041
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (22 papers)</h2></summary>

<details>
<summary><strong>Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning</strong> - Haozhe Wang, Qixin Xu, Che Liu, Junhong Wu, Fangzhen Lin, Wenhu Chen - [[pdf]](https://arxiv.org/pdf/2509.03646)</summary>

**Abstract:** Reinforcement Learning (RL) has proven highly effective at enhancing the complex reasoning abilities of Large Language Models (LLMs), yet underlying mechanisms driving this success remain largely opaque. Our analysis reveals that puzzling phenomena like ``aha moments", ``length-scaling'' and entropy dynamics are not disparate occurrences but hallmarks of an emergent reasoning hierarchy, akin to the separation of high-level strategic planning from low-level procedural execution in human cognition. We uncover a compelling two-phase dynamic: initially, a model is constrained by procedural correctness and must improve its low-level skills. The learning bottleneck then decisively shifts, with performance gains being driven by the exploration and mastery of high-level strategic planning. This insight exposes a core inefficiency in prevailing RL algorithms like GRPO, which apply optimization pressure agnostically and dilute the learning signal across all tokens. To address this, we propose HIerarchy-Aware Credit Assignment (HICRA), an algorithm that concentrates optimization efforts on high-impact planning tokens. HICRA significantly outperforms strong baselines, demonstrating that focusing on this strategic bottleneck is key to unlocking advanced reasoning. Furthermore, we validate semantic entropy as a superior compass for measuring strategic exploration over misleading metrics such as token-level entropy.

**arXiv ID:** 2509.03646
</details>

<details>
<summary><strong>An Agentic Model Context Protocol Framework for Medical Concept Standardization</strong> - Jaerong Ahn, Andrew Wen, Nan Wang, Heling Jia, Zhiyi Yue, Sunyang Fu, Hongfang Liu - [[pdf]](https://arxiv.org/pdf/2509.03828)</summary>

**Abstract:** The Observational Medical Outcomes Partnership (OMOP) common data model (CDM) provides a standardized representation of heterogeneous health data to support large-scale, multi-institutional research. One critical step in data standardization using OMOP CDM is the mapping of source medical terms to OMOP standard concepts, a procedure that is resource-intensive and error-prone. While large language models (LLMs) have the potential to facilitate this process, their tendency toward hallucination makes them unsuitable for clinical deployment without training and expert validation. Here, we developed a zero-training, hallucination-preventive mapping system based on the Model Context Protocol (MCP), a standardized and secure framework allowing LLMs to interact with external resources and tools. The system enables explainable mapping and significantly improves efficiency and accuracy with minimal effort. It provides real-time vocabulary lookups and structured reasoning outputs suitable for immediate use in both exploratory and production environments.

**arXiv ID:** 2509.03828
</details>

<details>
<summary><strong>A Foundation Model for Chest X-ray Interpretation with Grounded Reasoning via Online Reinforcement Learning</strong> - Qika Lin, Yifan Zhu, Bin Pu, Ling Huang, Haoran Luo, Jingying Ma, Zhen Peng, Tianzhe Zhao, Fangzhi Xu, Jian Zhang, Kai He, Zhonghong Ou, Swapnil Mishra, Mengling Feng - [[pdf]](https://arxiv.org/pdf/2509.03906)</summary>

**Abstract:** Medical foundation models (FMs) have shown tremendous promise amid the rapid advancements in artificial intelligence (AI) technologies. However, current medical FMs typically generate answers in a black-box manner, lacking transparent reasoning processes and locally grounded interpretability, which hinders their practical clinical deployments. To this end, we introduce DeepMedix-R1, a holistic medical FM for chest X-ray (CXR) interpretation. It leverages a sequential training pipeline: initially fine-tuned on curated CXR instruction data to equip with fundamental CXR interpretation capabilities, then exposed to high-quality synthetic reasoning samples to enable cold-start reasoning, and finally refined via online reinforcement learning to enhance both grounded reasoning quality and generation performance. Thus, the model produces both an answer and reasoning steps tied to the image's local regions for each query. Quantitative evaluation demonstrates substantial improvements in report generation (e.g., 14.54% and 31.32% over LLaVA-Rad and MedGemma) and visual question answering (e.g., 57.75% and 23.06% over MedGemma and CheXagent) tasks. To facilitate robust assessment, we propose Report Arena, a benchmarking framework using advanced language models to evaluate answer quality, further highlighting the superiority of DeepMedix-R1. Expert review of generated reasoning steps reveals greater interpretability and clinical plausibility compared to the established Qwen2.5-VL-7B model (0.7416 vs. 0.2584 overall preference). Collectively, our work advances medical FM development toward holistic, transparent, and clinically actionable modeling for CXR interpretation.

**arXiv ID:** 2509.03906
</details>

<details>
<summary><strong>CoT-Space: A Theoretical Framework for Internal Slow-Thinking via Reinforcement Learning</strong> - Zeyu Gan, Hao Yi, Yong Liu - [[pdf]](https://arxiv.org/pdf/2509.04027)</summary>

**Abstract:** Reinforcement Learning (RL) has become a pivotal approach for enhancing the reasoning capabilities of Large Language Models (LLMs). However, a significant theoretical gap persists, as traditional token-level RL frameworks fail to align with the reasoning-level nature of complex, multi-step thought processes like Chain-of-Thought (CoT). To address this challenge, we introduce CoT-Space, a novel theoretical framework that recasts LLM reasoning from a discrete token-prediction task to an optimization process within a continuous, reasoning-level semantic space. By analyzing this process from both a noise perspective and a risk perspective, we demonstrate that the convergence to an optimal CoT length is a natural consequence of the fundamental trade-off between underfitting and overfitting. Furthermore, extensive experiments provide strong empirical validation for our theoretical findings. Our framework not only provides a coherent explanation for empirical phenomena such as overthinking but also offers a solid theoretical foundation to guide the future development of more effective and generalizable reasoning agents.

**arXiv ID:** 2509.04027
</details>

<details>
<summary><strong>Hybrid Reinforcement Learning and Search for Flight Trajectory Planning</strong> - Alberto Luise, Michele Lombardi, Florent Teichteil Koenigsbuch - [[pdf]](https://arxiv.org/pdf/2509.04100)</summary>

**Abstract:** This paper explores the combination of Reinforcement Learning (RL) and search-based path planners to speed up the optimization of flight paths for airliners, where in case of emergency a fast route re-calculation can be crucial. The fundamental idea is to train an RL Agent to pre-compute near-optimal paths based on location and atmospheric data and use those at runtime to constrain the underlying path planning solver and find a solution within a certain distance from the initial guess. The approach effectively reduces the size of the solver's search space, significantly speeding up route optimization. Although global optimality is not guaranteed, empirical results conducted with Airbus aircraft's performance models show that fuel consumption remains nearly identical to that of an unconstrained solver, with deviations typically within 1%. At the same time, computation speed can be improved by up to 50% as compared to using a conventional solver alone.

**arXiv ID:** 2509.04100
</details>

<details>
<summary><strong>QuesGenie: Intelligent Multimodal Question Generation</strong> - Ahmed Mubarak, Amna Ahmed, Amira Nasser, Aya Mohamed, Fares El-Sadek, Mohammed Ahmed, Ahmed Salah, Youssef Sobhy - [[pdf]](https://arxiv.org/pdf/2509.03535)</summary>

**Abstract:** In today's information-rich era, learners have access to abundant educational resources, but the lack of practice materials tailored to these resources presents a significant challenge. This project addresses that gap by developing a multi-modal question generation system that can automatically generate diverse question types from various content formats. The system features four major components: multi-modal input handling, question generation, reinforcement learning from human feedback (RLHF), and an end-to-end interactive interface. This project lays the foundation for automated, scalable, and intelligent question generation, carefully balancing resource efficiency, robust functionality and a smooth user experience.

**arXiv ID:** 2509.03535
</details>

<details>
<summary><strong>AR$^2$: Adversarial Reinforcement Learning for Abstract Reasoning in Large Language Models</strong> - Cheng-Kai Yeh, Hsing-Wang Lee, Chung-Hung Kuo, Hen-Hsen Huang - [[pdf]](https://arxiv.org/pdf/2509.03537)</summary>

**Abstract:** Abstraction--the ability to recognize and distill essential computational patterns from complex problem statements--is a foundational skill in computer science, critical both for human problem-solvers and coding-oriented large language models (LLMs). Despite recent advances in training LLMs for code generation using reinforcement learning (RL), most existing approaches focus primarily on superficial pattern recognition, overlooking explicit training for abstraction. In this study, we propose AR$^2$ (Adversarial Reinforcement Learning for Abstract Reasoning), a novel framework explicitly designed to enhance the abstraction abilities of LLMs. AR$^2$ employs a teacher model to transform kernel problems into narrative-rich, challenging descriptions without changing their fundamental logic. Simultaneously, a student coding model is trained to solve these complex narrative problems by extracting their underlying computational kernels. Experimental results demonstrate that AR$^2$ substantially improves the student model's accuracy on previously unseen, challenging programming tasks, underscoring abstraction as a key skill for enhancing LLM generalization.

**arXiv ID:** 2509.03537
</details>

<details>
<summary><strong>Meta-Inverse Reinforcement Learning for Mean Field Games via Probabilistic Context Variables</strong> - Yang Chen, Xiao Lin, Bo Yan, Libo Zhang, Jiamou Liu, Neset Özkan Tan, Michael Witbrock - [[pdf]](https://arxiv.org/pdf/2509.03845)</summary>

**Abstract:** Designing suitable reward functions for numerous interacting intelligent agents is challenging in real-world applications. Inverse reinforcement learning (IRL) in mean field games (MFGs) offers a practical framework to infer reward functions from expert demonstrations. While promising, the assumption of agent homogeneity limits the capability of existing methods to handle demonstrations with heterogeneous and unknown objectives, which are common in practice. To this end, we propose a deep latent variable MFG model and an associated IRL method. Critically, our method can infer rewards from different yet structurally similar tasks without prior knowledge about underlying contexts or modifying the MFG model itself. Our experiments, conducted on simulated scenarios and a real-world spatial taxi-ride pricing problem, demonstrate the superiority of our approach over state-of-the-art IRL methods in MFGs.

**arXiv ID:** 2509.03845
</details>

<details>
<summary><strong>Reinforcement Learning for Robust Ageing-Aware Control of Li-ion Battery Systems with Data-Driven Formal Verification</strong> - Rudi Coppola, Hovsep Touloujian, Pierfrancesco Ombrini, Manuel Mazo Jr - [[pdf]](https://arxiv.org/pdf/2509.04288)</summary>

**Abstract:** Rechargeable lithium-ion (Li-ion) batteries are a ubiquitous element of modern technology. In the last decades, the production and design of such batteries and their adjacent embedded charging and safety protocols, denoted by Battery Management Systems (BMS), has taken central stage. A fundamental challenge to be addressed is the trade-off between the speed of charging and the ageing behavior, resulting in the loss of capacity in the battery cell. We rely on a high-fidelity physics-based battery model and propose an approach to data-driven charging and safety protocol design. Following a Counterexample-Guided Inductive Synthesis scheme, we combine Reinforcement Learning (RL) with recent developments in data-driven formal methods to obtain a hybrid control strategy: RL is used to synthesise the individual controllers, and a data-driven abstraction guides their partitioning into a switched structure, depending on the initial output measurements of the battery. The resulting discrete selection among RL-based controllers, coupled with the continuous battery dynamics, realises a hybrid system. When a design meets the desired criteria, the abstraction provides probabilistic guarantees on the closed-loop performance of the cell.

**arXiv ID:** 2509.04288
</details>

<details>
<summary><strong>Plan Verification for LLM-Based Embodied Task Completion Agents</strong> - Ananth Hariharan, Vardhan Dongre, Dilek Hakkani-Tür, Gokhan Tur - [[pdf]](https://arxiv.org/pdf/2509.02761)</summary>

**Abstract:** Large language model (LLM) based task plans and corresponding human demonstrations for embodied AI may be noisy, with unnecessary actions, redundant navigation, and logical errors that reduce policy quality. We propose an iterative verification framework in which a Judge LLM critiques action sequences and a Planner LLM applies the revisions, yielding progressively cleaner and more spatially coherent trajectories. Unlike rule-based approaches, our method relies on natural language prompting, enabling broad generalization across error types including irrelevant actions, contradictions, and missing steps. On a set of manually annotated actions from the TEACh embodied AI dataset, our framework achieves up to 90% recall and 100% precision across four state-of-the-art LLMs (GPT o4-mini, DeepSeek-R1, Gemini 2.5, LLaMA 4 Scout). The refinement loop converges quickly, with 96.5% of sequences requiring at most three iterations, while improving both temporal efficiency and spatial action organization. Crucially, the method preserves human error-recovery patterns rather than collapsing them, supporting future work on robust corrective behavior. By establishing plan verification as a reliable LLM capability for spatial planning and action refinement, we provide a scalable path to higher-quality training data for imitation learning in embodied AI.

**arXiv ID:** 2509.02761
</details>

<details>
<summary><strong>Synthesizing Sheet Music Problems for Evaluation and Reinforcement Learning</strong> - Zhilin Wang, Zhe Yang, Yun Luo, Yafu Li, Haoran Zhang, Runzhe Zhan, Derek F. Wong, Jizhe Zhou, Yu Cheng - [[pdf]](https://arxiv.org/pdf/2509.04059)</summary>

**Abstract:** Enhancing the ability of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) to interpret sheet music is a crucial step toward building AI musicians. However, current research lacks both evaluation benchmarks and training data for sheet music reasoning. To address this, we propose the idea of synthesizing sheet music problems grounded in music theory, which can serve both as evaluation benchmarks and as training data for reinforcement learning with verifiable rewards (RLVR). We introduce a data synthesis framework that generates verifiable sheet music questions in both textual and visual modalities, leading to the Synthetic Sheet Music Reasoning Benchmark (SSMR-Bench) and a complementary training set. Evaluation results on SSMR-Bench show the importance of models' reasoning abilities in interpreting sheet music. At the same time, the poor performance of Gemini 2.5-Pro highlights the challenges that MLLMs still face in interpreting sheet music in a visual format. By leveraging synthetic data for RLVR, Qwen3-8B-Base and Qwen2.5-VL-Instruct achieve improvements on the SSMR-Bench. Besides, the trained Qwen3-8B-Base surpasses GPT-4 in overall performance on MusicTheoryBench and achieves reasoning performance comparable to GPT-4 with the strategies of Role play and Chain-of-Thought. Notably, its performance on math problems also improves relative to the original Qwen3-8B-Base. Furthermore, our results show that the enhanced reasoning ability can also facilitate music composition. In conclusion, we are the first to propose the idea of synthesizing sheet music problems based on music theory rules, and demonstrate its effectiveness not only in advancing model reasoning for sheet music understanding but also in unlocking new possibilities for AI-assisted music creation.

**arXiv ID:** 2509.04059
</details>

<details>
<summary><strong>Context Reasoner: Incentivizing Reasoning Capability for Contextualized Privacy and Safety Compliance via Reinforcement Learning</strong> - Wenbin Hu, Haoran Li, Huihao Jing, Qi Hu, Ziqian Zeng, Sirui Han, Heli Xu, Tianshu Chu, Peizhao Hu, Yangqiu Song - [[pdf]](https://arxiv.org/pdf/2505.14585)</summary>

**Abstract:** While Large Language Models (LLMs) exhibit remarkable capabilities, they also introduce significant safety and privacy risks. Current mitigation strategies often fail to preserve contextual reasoning capabilities in risky scenarios. Instead, they rely heavily on sensitive pattern matching to protect LLMs, which limits the scope. Furthermore, they overlook established safety and privacy standards, leading to systemic risks for legal compliance. To address these gaps, we formulate safety and privacy issues into contextualized compliance problems following the Contextual Integrity (CI) theory. Under the CI framework, we align our model with three critical regulatory standards: GDPR, EU AI Act, and HIPAA. Specifically, we employ reinforcement learning (RL) with a rule-based reward to incentivize contextual reasoning capabilities while enhancing compliance with safety and privacy norms. Through extensive experiments, we demonstrate that our method not only significantly enhances legal compliance (achieving a +8.58% accuracy improvement in safety/privacy benchmarks) but also further improves general reasoning capability. For OpenThinker-7B, a strong reasoning model that significantly outperforms its base model Qwen2.5-7B-Instruct across diverse subjects, our method enhances its general reasoning capabilities, with +2.05% and +8.98% accuracy improvement on the MMLU and LegalBench benchmark, respectively.

**arXiv ID:** 2505.14585
</details>

<details>
<summary><strong>Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities</strong> - Rikuto Kotoge, Mai Nishimura, Jiaxin Ma - [[pdf]](https://arxiv.org/pdf/2508.20324)</summary>

**Abstract:** Reinforcement Learning has emerged as a post-training approach to elicit agentic RAG behaviors such as search and planning from language models. However, compact language models (e.g., 0.5B parameters) struggle due to poor reasoning ability, resulting in sparse rewards and unstable training. To overcome these difficulties, we propose Distillation-Guided Policy Optimization (DGPO), which addresses the challenges through cold-start initialization from teacher demonstrations and continuous teacher guidance during policy optimization. To systematically evaluate our approach, we introduce Agentic RAG Capabilities (ARC), a fine-grained metric analyzing reasoning, search coordination, and response synthesis. Comprehensive experiments demonstrate that DGPO enables compact models to achieve sophisticated agentic search behaviors, even outperforming the larger teacher model in some cases. DGPO makes agentic RAG feasible in computing resource-constrained environments.

**arXiv ID:** 2508.20324
</details>

<details>
<summary><strong>AutoGrid AI: Deep Reinforcement Learning Framework for Autonomous Microgrid Management</strong> - Kenny Guo, Nicholas Eckhert, Krish Chhajer, Luthira Abeykoon, Lorne Schell - [[pdf]](https://arxiv.org/pdf/2509.03666)</summary>

**Abstract:** We present a deep reinforcement learning-based framework for autonomous microgrid management. tailored for remote communities. Using deep reinforcement learning and time-series forecasting models, we optimize microgrid energy dispatch strategies to minimize costs and maximize the utilization of renewable energy sources such as solar and wind. Our approach integrates the transformer architecture for forecasting of renewable generation and a proximal-policy optimization (PPO) agent to make decisions in a simulated environment. Our experimental results demonstrate significant improvements in both energy efficiency and operational resilience when compared to traditional rule-based methods. This work contributes to advancing smart-grid technologies in pursuit of zero-carbon energy systems. We finally provide an open-source framework for simulating several microgrid environments.

**arXiv ID:** 2509.03666
</details>

<details>
<summary><strong>RL's Razor: Why Online Reinforcement Learning Forgets Less</strong> - Idan Shenfeld, Jyothish Pari, Pulkit Agrawal - [[pdf]](https://arxiv.org/pdf/2509.04259)</summary>

**Abstract:** Comparison of fine-tuning models with reinforcement learning (RL) and supervised fine-tuning (SFT) reveals that, despite similar performance at a new task, RL preserves prior knowledge and capabilities significantly better. We find that the degree of forgetting is determined by the distributional shift, measured as the KL-divergence between the fine-tuned and base policy evaluated on the new task. Our analysis reveals that on-policy RL is implicitly biased towards KL-minimal solutions among the many that solve the new task, whereas SFT can converge to distributions arbitrarily far from the base model. We validate these findings through experiments with large language models and robotic foundation models and further provide theoretical justification for why on-policy RL updates lead to a smaller KL change. We term this principle $\textit{RL's Razor}$: among all ways to solve a new task, RL prefers those closest in KL to the original model.

**arXiv ID:** 2509.04259
</details>

<details>
<summary><strong>Machine Learning for LiDAR-Based Indoor Surface Classification in Intelligent Wireless Environments</strong> - Parth Ashokbhai Shiroya, Swarnagowri Shashidhar, Amod Ashtekar, Krishna Aindrila Kar, Rafaela Lomboy, Dalton Davis, Mohammed E. Eltayeb - [[pdf]](https://arxiv.org/pdf/2509.03813)</summary>

**Abstract:** Reliable connectivity in millimeter-wave (mmWave) and sub-terahertz (sub-THz) networks depends on reflections from surrounding surfaces, as high-frequency signals are highly vulnerable to blockage. The scattering behavior of a surface is determined not only by material permittivity but also by roughness, which governs whether energy remains in the specular direction or is diffusely scattered. This paper presents a LiDAR-driven machine learning framework for classifying indoor surfaces into semi-specular and low-specular categories, using optical reflectivity as a proxy for electromagnetic scattering behavior. A dataset of over 78,000 points from 15 representative indoor materials was collected and partitioned into 3 cm x 3 cm patches to enable classification from partial views. Patch-level features capturing geometry and intensity, including elevation angle, natural-log-scaled intensity, and max-to-mean ratio, were extracted and used to train Random Forest, XGBoost, and neural network classifiers. Results show that ensemble tree-based models consistently provide the best trade-off between accuracy and robustness, confirming that LiDAR-derived features capture roughness-induced scattering effects. The proposed framework enables the generation of scatter aware environment maps and digital twins, supporting adaptive beam management, blockage recovery, and environment-aware connectivity in next-generation networks.

**arXiv ID:** 2509.03813
</details>

<details>
<summary><strong>Connections between reinforcement learning with feedback,test-time scaling, and diffusion guidance: An anthology</strong> - Yuchen Jiao, Yuxin Chen, Gen Li - [[pdf]](https://arxiv.org/pdf/2509.04372)</summary>

**Abstract:** In this note, we reflect on several fundamental connections among widely used post-training techniques. We clarify some intimate connections and equivalences between reinforcement learning with human feedback, reinforcement learning with internal feedback, and test-time scaling (particularly soft best-of-$N$ sampling), while also illuminating intrinsic links between diffusion guidance and test-time scaling. Additionally, we introduce a resampling approach for alignment and reward-directed diffusion models, sidestepping the need for explicit reinforcement learning techniques.

**arXiv ID:** 2509.04372
</details>

<details>
<summary><strong>EvoCoT: Overcoming the Exploration Bottleneck in Reinforcement Learning</strong> - Huanyu Liu, Jia Li, Chang Yu, Taozhi Chen, Yihong Dong, Lecheng Wang, XiaoLong Hu, Ge Li - [[pdf]](https://arxiv.org/pdf/2508.07809)</summary>

**Abstract:** Reinforcement learning with verifiable reward (RLVR) has become a promising paradigm for post-training large language models (LLMs) to improve their reasoning capability. However, when the rollout accuracy is low on hard problems, the reward becomes sparse, limiting learning efficiency and causing exploration bottlenecks. Existing approaches either rely on stronger LLMs for distillation or filter out difficult problems, which limits scalability or restricts reasoning improvement through exploration.
We propose EvoCoT, a self-evolving curriculum learning framework based on two-stage chain-of-thought (CoT) reasoning optimization. EvoCoT constrains the exploration space by self-generating and verifying CoT trajectories, then gradually shortens them to expand the space in a controlled way. This enables LLMs to stably learn from initially unsolved hard problems under sparse rewards. We apply EvoCoT to multiple LLM families, including Qwen, DeepSeek, and Llama. Experiments show that EvoCoT enables LLMs to solve previously unsolved problems, improves reasoning capability without external CoT supervision, and is compatible with various RL fine-tuning methods. We release the source code to support future research.

**arXiv ID:** 2508.07809
</details>

<details>
<summary><strong>Robust Bandwidth Estimation for Real-Time Communication with Offline Reinforcement Learning</strong> - Jian Kai, Tianwei Zhang, Zihan Ling, Yang Cao, Can Shen - [[pdf]](https://arxiv.org/pdf/2507.05785)</summary>

**Abstract:** Accurate bandwidth estimation (BWE) is critical for real-time communication (RTC) systems. Traditional heuristic approaches offer limited adaptability under dynamic networks, while online reinforcement learning (RL) suffers from high exploration costs and potential service disruptions. Offline RL, which leverages high-quality data collected from real-world environments, offers a promising alternative. However, challenges such as out-of-distribution (OOD) actions, policy extraction from behaviorally diverse datasets, and reliable deployment in production systems remain unsolved. We propose RBWE, a robust bandwidth estimation framework based on offline RL that integrates Q-ensemble (an ensemble of Q-functions) with a Gaussian mixture policy to mitigate OOD risks and enhance policy learning. A fallback mechanism ensures deployment stability by switching to heuristic methods under high uncertainty. Experimental results show that RBWE reduces overestimation errors by 18% and improves the 10th percentile Quality of Experience (QoE) by 18.6%, demonstrating its practical effectiveness in real-world RTC applications. The implementation is publicly available at this https URL.

**arXiv ID:** 2507.05785
</details>

<details>
<summary><strong>Solving Robotics Tasks with Prior Demonstration via Exploration-Efficient Deep Reinforcement Learning</strong> - Chengyandan Shen, Christoffer Sloth - [[pdf]](https://arxiv.org/pdf/2509.04069)</summary>

**Abstract:** This paper proposes an exploration-efficient Deep Reinforcement Learning with Reference policy (DRLR) framework for learning robotics tasks that incorporates demonstrations. The DRLR framework is developed based on an algorithm called Imitation Bootstrapped Reinforcement Learning (IBRL). We propose to improve IBRL by modifying the action selection module. The proposed action selection module provides a calibrated Q-value, which mitigates the bootstrapping error that otherwise leads to inefficient exploration. Furthermore, to prevent the RL policy from converging to a sub-optimal policy, SAC is used as the RL policy instead of TD3. The effectiveness of our method in mitigating bootstrapping error and preventing overfitting is empirically validated by learning two robotics tasks: bucket loading and open drawer, which require extensive interactions with the environment. Simulation results also demonstrate the robustness of the DRLR framework across tasks with both low and high state-action dimensions, and varying demonstration qualities. To evaluate the developed framework on a real-world industrial robotics task, the bucket loading task is deployed on a real wheel loader. The sim2real results validate the successful deployment of the DRLR framework.

**arXiv ID:** 2509.04069
</details>

<details>
<summary><strong>Avoidance of an unexpected obstacle without reinforcement learning: Why not using advanced control-theoretic tools?</strong> - Cédric Join, Michel Fliess - [[pdf]](https://arxiv.org/pdf/2509.03721)</summary>

**Abstract:** This communication on collision avoidance with unexpected obstacles is motivated by some critical appraisals on reinforcement learning (RL) which "requires ridiculously large numbers of trials to learn any new task" (Yann LeCun). We use the classic Dubins' car in order to replace RL with flatness-based control, combined with the HEOL feedback setting, and the latest model-free predictive control approach. The two approaches lead to convincing computer experiments where the results with the model-based one are only slightly better. They exhibit a satisfactory robustness with respect to randomly generated mismatches/disturbances, which become excellent in the model-free case. Those properties would have been perhaps difficult to obtain with today's popular machine learning techniques in AI. Finally, we should emphasize that our two methods require a low computational burden.

**arXiv ID:** 2509.03721
</details>

<details>
<summary><strong>Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation via End-to-End Reinforcement Learning</strong> - Fan Yang, Per Frivik, David Hoeller, Chen Wang, Cesar Cadena, Marco Hutter - [[pdf]](https://arxiv.org/pdf/2506.05997)</summary>

**Abstract:** Recent advancements in robot navigation, particularly with end-to-end learning approaches such as reinforcement learning (RL), have demonstrated strong performance. However, successful navigation still depends on two key capabilities: mapping and planning (explicitly or implicitly). Classical approaches rely on explicit mapping pipelines to register egocentric observations into a coherent map. In contrast, end-to-end learning often achieves this implicitly -- through recurrent neural networks (RNNs) that fuse current and historical observations into a latent space for planning. While existing architectures, such as LSTM and GRU, can capture temporal dependencies, our findings reveal a critical limitation: their inability to effectively perform spatial memorization. This capability is essential for integrating sequential observations from varying perspectives to build spatial representations that support planning. To address this, we propose Spatially-Enhanced Recurrent Units (SRUs) -- a simple yet effective modification to existing RNNs -- that enhance spatial memorization. We further introduce an attention-based network architecture integrated with SRUs, enabling long-range mapless navigation using a single forward-facing stereo camera. We also employ regularization techniques to facilitate robust end-to-end recurrent training via RL. Experimental results show 23.5% overall improvement in long-range navigation compared to existing RNNs. With SRU memory, our method outperforms RL baselines -- one relying on explicit mapping and the other on stacked historical observations -- by 29.6% and 105.0%, respectively, across diverse environments requiring long-horizon mapping and memorization. Finally, we address the sim-to-real gap by leveraging large-scale pretraining on synthetic depth data, enabling zero-shot transfer for deployment across diverse and complex real-world environments.

**arXiv ID:** 2506.05997
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
