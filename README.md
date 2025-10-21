# Agent arXiv Daily

**Last Updated:** 2025-10-21 02:44:50

**Total Papers:** 113

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (7 papers)</h2></summary>

<details>
<summary><strong>SARHAchat: An LLM-Based Chatbot for Sexual and Reproductive Health Counseling</strong> - Jiaye Yang, Xinyu Zhao, Tianlong Chen, Kandyce Brennan - [[pdf]](https://arxiv.org/pdf/2510.16081)</summary>

**Abstract:** While Artificial Intelligence (AI) shows promise in healthcare applications, existing conversational systems often falter in complex and sensitive medical domains such as Sexual and Reproductive Health (SRH). These systems frequently struggle with hallucination and lack the specialized knowledge required, particularly for sensitive SRH topics. Furthermore, current AI approaches in healthcare tend to prioritize diagnostic capabilities over comprehensive patient care and education. Addressing these gaps, this work at the UNC School of Nursing introduces SARHAchat, a proof-of-concept Large Language Model (LLM)-based chatbot. SARHAchat is designed as a reliable, user-centered system integrating medical expertise with empathetic communication to enhance SRH care delivery. Our evaluation demonstrates SARHAchat's ability to provide accurate and contextually appropriate contraceptive counseling while maintaining a natural conversational flow. The demo is available at this https URL}{this https URL.

**arXiv ID:** 2510.16081
</details>

<details>
<summary><strong>MoPHES:Leveraging on-device LLMs as Agent for Mobile Psychological Health Evaluation and Support</strong> - Xun Wei, Pukai Zhou, Zeyu Wang - [[pdf]](https://arxiv.org/pdf/2510.16085)</summary>

**Abstract:** The 2022 World Mental Health Report calls for global mental health care reform, amid rising prevalence of issues like anxiety and depression that affect nearly one billion people worldwide. Traditional in-person therapy fails to meet this demand, and the situation is worsened by stigma. While general-purpose large language models (LLMs) offer efficiency for AI-driven mental health solutions, they underperform because they lack specialized fine-tuning. Existing LLM-based mental health chatbots can engage in empathetic conversations, but they overlook real-time user mental state assessment which is critical for professional counseling. This paper proposes MoPHES, a framework that integrates mental state evaluation, conversational support, and professional treatment recommendations. The agent developed under this framework uses two fine-tuned MiniCPM4-0.5B LLMs: one is fine-tuned on mental health conditions datasets to assess users' mental states and predict the severity of anxiety and depression; the other is fine-tuned on multi-turn dialogues to handle conversations with users. By leveraging insights into users' mental states, our agent provides more tailored support and professional treatment recommendations. Both models are also deployed directly on mobile devices to enhance user convenience and protect user privacy. Additionally, to evaluate the performance of MoPHES with other LLMs, we develop a benchmark for the automatic evaluation of mental state prediction and multi-turn counseling dialogues, which includes comprehensive evaluation metrics, datasets, and methods.

**arXiv ID:** 2510.16085
</details>

<details>
<summary><strong>ATA: A Neuro-Symbolic Approach to Implement Autonomous and Trustworthy Agents</strong> - David Peer, Sebastian Stabinger - [[pdf]](https://arxiv.org/pdf/2510.16381)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated impressive capabilities, yet their deployment in high-stakes domains is hindered by inherent limitations in trustworthiness, including hallucinations, instability, and a lack of transparency. To address these challenges, we introduce a generic neuro-symbolic approach, which we call Autonomous Trustworthy Agents (ATA). The core of our approach lies in decoupling tasks into two distinct phases: Offline knowledge ingestion and online task processing. During knowledge ingestion, an LLM translates an informal problem specification into a formal, symbolic knowledge base. This formal representation is crucial as it can be verified and refined by human experts, ensuring its correctness and alignment with domain requirements. In the subsequent task processing phase, each incoming input is encoded into the same formal language. A symbolic decision engine then utilizes this encoded input in conjunction with the formal knowledge base to derive a reliable result. Through an extensive evaluation on a complex reasoning task, we demonstrate that a concrete implementation of ATA is competitive with state-of-the-art end-to-end reasoning models in a fully automated setup while maintaining trustworthiness. Crucially, with a human-verified and corrected knowledge base, our approach significantly outperforms even larger models, while exhibiting perfect determinism, enhanced stability against input perturbations, and inherent immunity to prompt injection attacks. By generating decisions grounded in symbolic reasoning, ATA offers a practical and controllable architecture for building the next generation of transparent, auditable, and reliable autonomous agents.

**arXiv ID:** 2510.16381
</details>

<details>
<summary><strong>Learning to play: A Multimodal Agent for 3D Game-Play</strong> - Yuguang Yue, Irakli Salia, Samuel Hunt, Christopher Green, Wenzhe Shi, Jonathan J Hunt - [[pdf]](https://arxiv.org/pdf/2510.16774)</summary>

**Abstract:** We argue that 3-D first-person video games are a challenging environment for real-time multi-modal reasoning. We first describe our dataset of human game-play, collected across a large variety of 3-D first-person games, which is both substantially larger and more diverse compared to prior publicly disclosed datasets, and contains text instructions. We demonstrate that we can learn an inverse dynamics model from this dataset, which allows us to impute actions on a much larger dataset of publicly available videos of human game play that lack recorded actions. We then train a text-conditioned agent for game playing using behavior cloning, with a custom architecture capable of realtime inference on a consumer GPU. We show the resulting model is capable of playing a variety of 3-D games and responding to text input. Finally, we outline some of the remaining challenges such as long-horizon tasks and quantitative evaluation across a large set of games.

**arXiv ID:** 2510.16774
</details>

<details>
<summary><strong>BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design</strong> - Deepro Choudhury, Sinead Williamson, Adam Goliński, Ning Miao, Freddie Bickford Smith, Michael Kirchhof, Yizhe Zhang, Tom Rainforth - [[pdf]](https://arxiv.org/pdf/2508.21184)</summary>

**Abstract:** We propose a general-purpose approach for improving the ability of Large Language Models (LLMs) to intelligently and adaptively gather information from a user or other external source using the framework of sequential Bayesian experimental design (BED). This enables LLMs to act as effective multi-turn conversational agents and interactively interface with external environments. Our approach, which we call BED-LLM (Bayesian Experimental Design with Large Language Models), is based on iteratively choosing questions or queries that maximize the expected information gain (EIG) about the task of interest given the responses gathered previously. We show how this EIG can be formulated (and then estimated) in a principled way using a probabilistic model derived from the LLM's predictive distributions and provide detailed insights into key decisions in its construction and updating procedure. We find that BED-LLM achieves substantial gains in performance across a wide range of tests based on the 20 questions game and using the LLM to actively infer user preferences, compared to direct prompting of the LLM and other adaptive design strategies.

**arXiv ID:** 2508.21184
</details>

<details>
<summary><strong>A Survey of Scientific Large Language Models: From Data Foundations to Agent Frontiers</strong> - Ming Hu, Chenglong Ma, Wei Li, Wanghan Xu, Jiamin Wu, Jucheng Hu, Tianbin Li, Guohang Zhuang, Jiaqi Liu, Yingzhou Lu, Ying Chen, Chaoyang Zhang, Cheng Tan, Jie Ying, Guocheng Wu, Shujian Gao, Pengcheng Chen, Jiashi Lin, Haitao Wu, Lulu Chen, Fengxiang Wang, Yuanyuan Zhang, Xiangyu Zhao, Feilong Tang, Encheng Su, Junzhi Ning, Xinyao Liu, Ye Du, Changkai Ji, Pengfei Jiang, Cheng Tang, Ziyan Huang, Jiyao Liu, Jiaqi Wei, Yuejin Yang, Xiang Zhang, Guangshuai Wang, Yue Yang, Huihui Xu, Ziyang Chen, Yizhou Wang, Chen Tang, Jianyu Wu, Yuchen Ren, Siyuan Yan, Zhonghua Wang, Zhongxing Xu, Shiyan Su, Shangquan Sun, Runkai Zhao, Zhisheng Zhang, Dingkang Yang, Jinjie Wei, Jiaqi Wang, Jiahao Xu, Jiangtao Yan, Wenhao Tang, Hongze Zhu, Yu Liu, Fudi Wang, Yiqing Shen, Yuanfeng Ji, Yanzhou Su, Tong Xie, Hongming Shan, Chun-Mei Feng, Zhi Hou, Diping Song, Lihao Liu, Yanyan Huang, Lequan Yu, Bin Fu, Shujun Wang, Xiaomeng Li, Xiaowei Hu, Yun Gu, Ben Fei, Benyou Wang, Yuewen Cao, Minjie Shen, Jie Xu, Haodong Duan, Fang Yan, Hongxia Hao, Jielan Li, Jiajun Du, Yanbo Wang, Imran Razzak, Zhongying Deng, Chi Zhang, Lijun Wu, Conghui He, Zhaohui Lu, Jinhai Huang, Wenqi Shao, Yihao Liu, Siqi Luo, Yi Xin, Xiaohong Liu, Fenghua Ling - [[pdf]](https://arxiv.org/pdf/2508.21148)</summary>

**Abstract:** Scientific Large Language Models (Sci-LLMs) are transforming how knowledge is represented, integrated, and applied in scientific research, yet their progress is shaped by the complex nature of scientific data. This survey presents a comprehensive, data-centric synthesis that reframes the development of Sci-LLMs as a co-evolution between models and their underlying data substrate. We formulate a unified taxonomy of scientific data and a hierarchical model of scientific knowledge, emphasizing the multimodal, cross-scale, and domain-specific challenges that differentiate scientific corpora from general natural language processing datasets. We systematically review recent Sci-LLMs, from general-purpose foundations to specialized models across diverse scientific disciplines, alongside an extensive analysis of over 270 pre-/post-training datasets, showing why Sci-LLMs pose distinct demands -- heterogeneous, multi-scale, uncertainty-laden corpora that require representations preserving domain invariance and enabling cross-modal reasoning. On evaluation, we examine over 190 benchmark datasets and trace a shift from static exams toward process- and discovery-oriented assessments with advanced evaluation protocols. These data-centric analyses highlight persistent issues in scientific data development and discuss emerging solutions involving semi-automated annotation pipelines and expert validation. Finally, we outline a paradigm shift toward closed-loop systems where autonomous agents based on Sci-LLMs actively experiment, validate, and contribute to a living, evolving knowledge base. Collectively, this work provides a roadmap for building trustworthy, continually evolving artificial intelligence (AI) systems that function as a true partner in accelerating scientific discovery.

**arXiv ID:** 2508.21148
</details>

<details>
<summary><strong>Design Framework for Conversational Agent in Couple relationships: A Systematic Review</strong> - Soyoung Jung, Sung Park - [[pdf]](https://arxiv.org/pdf/2510.17119)</summary>

**Abstract:** The development of conversational agents (CAs) has shown strong potential in supporting mental health through dialogue. While many studies focus on CAs for individual psychological care, research on agents designed for couples facing relational or emotional challenges remains limited. This study aims to identify design considerations for CAs that address the relational context of couples and support their well-being. Following PRISMA guidelines, a systematic review was conducted across seven databases: CINAHL, Embase, PubMed, PsycINFO, Scopus, Web of Science, and the ACM Digital Library. Peer-reviewed empirical studies were screened, duplicates removed, and selection criteria applied, resulting in twelve studies for analysis. Thematic analysis was conducted across three dimensions: AI interaction design, relational framing, and technical limitations. Three key themes emerged: (1) the need for a relational expert persona, (2) technological directions leveraging state-of-the-art AI for relational specificity and emotional competence, and (3) a shift from content-centered to relationship-centered design. Based on these insights, eight design considerations are proposed for couple-oriented CAs: (1) agent persona, (2) individual mode, (3) concurrent mode, (4) conjoint mode, (5) ethics, (6) data and privacy, (7) interaction pattern, and (8) safety mechanism. These principles guide CAs as relational mediators capable of maintaining multiple alliances, respecting cultural and ethical boundaries, and ensuring fairness and emotional safety between partners. Ultimately, this review introduces a design framework that integrates relational theory with advanced AI technologies to inform future development of CAs for couple-based mental health interventions.

**arXiv ID:** 2510.17119
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (19 papers)</h2></summary>

<details>
<summary><strong>What Limits Agentic Systems Efficiency?</strong> - Song Bian, Minghao Yan, Anand Jayarajan, Gennady Pekhimenko, Shivaram Venkataraman - [[pdf]](https://arxiv.org/pdf/2510.16276)</summary>

**Abstract:** Large Language Models (LLMs), such as OpenAI-o1 and DeepSeek-R1, have demonstrated strong reasoning capabilities. To further enhance LLM capabilities, recent agentic systems, such as Deep Research, incorporate web interactions into LLM reasoning to mitigate uncertainties and reduce potential errors. However, existing research predominantly focuses on reasoning performance, often neglecting the efficiency of agentic systems. In this work, we present a comprehensive empirical study that identifies efficiency bottlenecks in web-interactive agentic systems. We decompose end-to-end latency into two primary components: LLM API latency and web environment latency. We conduct a comprehensive empirical study across 15 models and 5 providers to demonstrate high variability in API-based agentic systems. We observe that web environment latency can contribute as much as 53.7% to the overall latency in a web-based agentic system. To improve latency, we propose SpecCache, a caching framework augmented with speculative execution that can reduce web environment overhead. Extensive evaluations on two standard benchmarks show that our approach improves the cache hit rate by up to 58x compared to a random caching strategy, while reducing web environment overhead by up to 3.2x, without degrading agentic system performance.

**arXiv ID:** 2510.16276
</details>

<details>
<summary><strong>Limits of Emergent Reasoning of Large Language Models in Agentic Frameworks for Deterministic Games</strong> - Chris Su, Harrison Li, Matheus Marques, George Flint, Kevin Zhu, Sunishchal Dev - [[pdf]](https://arxiv.org/pdf/2510.15974)</summary>

**Abstract:** Recent work reports that Large Reasoning Models (LRMs) undergo a collapse in performance on solving puzzles beyond certain perplexity thresholds. In subsequent discourse, questions have arisen as to whether the nature of the task muddles an evaluation of true reasoning. One potential confound is the requirement that the model keep track of the state space on its own. We provide a large language model (LLM) with an environment interface for Tower of Hanoi problems, allowing it to make a move with a tool call, provide written justification, observe the resulting state space, and reprompt itself for the next move. We observe that access to an environment interface does not delay or eradicate performance collapse. Furthermore, LLM-parameterized policy analysis reveals increasing divergence from both optimal policies and uniformly random policies, suggesting that the model exhibits mode-like collapse at each level of complexity, and that performance is dependent upon whether the mode reflects the correct solution for the problem. We suggest that a similar phenomena might take place in LRMs.

**arXiv ID:** 2510.15974
</details>

<details>
<summary><strong>Ripple Effect Protocol: Coordinating Agent Populations</strong> - Ayush Chopra, Aman Sharma, Feroz Ahmad, Luca Muscariello, Vijoy Pandey, Ramesh Raskar - [[pdf]](https://arxiv.org/pdf/2510.16572)</summary>

**Abstract:** Modern AI agents can exchange messages using protocols such as A2A and ACP, yet these mechanisms emphasize communication over coordination. As agent populations grow, this limitation produces brittle collective behavior, where individually smart agents converge on poor group outcomes. We introduce the Ripple Effect Protocol (REP), a coordination protocol in which agents share not only their decisions but also lightweight sensitivities - signals expressing how their choices would change if key environmental variables shifted. These sensitivities ripple through local networks, enabling groups to align faster and more stably than with agent-centric communication alone. We formalize REP's protocol specification, separating required message schemas from optional aggregation rules, and evaluate it across scenarios with varying incentives and network topologies. Benchmarks across three domains: (i) supply chain cascades (Beer Game), (ii) preference aggregation in sparse networks (Movie Scheduling), and (iii) sustainable resource allocation (Fishbanks) show that REP improves coordination accuracy and efficiency over A2A by 41 to 100%, while flexibly handling multimodal sensitivity signals from LLMs. By making coordination a protocol-level capability, REP provides scalable infrastructure for the emerging Internet of Agents

**arXiv ID:** 2510.16572
</details>

<details>
<summary><strong>An Agentic Framework with LLMs for Solving Complex Vehicle Routing Problems</strong> - Ni Zhang, Zhiguang Cao, Jianan Zhou, Cong Zhang, Yew-Soon Ong - [[pdf]](https://arxiv.org/pdf/2510.16701)</summary>

**Abstract:** Complex vehicle routing problems (VRPs) remain a fundamental challenge, demanding substantial expert effort for intent interpretation and algorithm design. While large language models (LLMs) offer a promising path toward automation, current approaches still rely on external intervention, which restrict autonomy and often lead to execution errors and low solution feasibility. To address these challenges, we propose an Agentic Framework with LLMs (AFL) for solving complex vehicle routing problems, achieving full automation from problem instance to solution. AFL directly extracts knowledge from raw inputs and enables self-contained code generation without handcrafted modules or external solvers. To improve trustworthiness, AFL decomposes the overall pipeline into three manageable subtasks and employs four specialized agents whose coordinated interactions enforce cross-functional consistency and logical soundness. Extensive experiments on 60 complex VRPs, ranging from standard benchmarks to practical variants, validate the effectiveness and generality of our framework, showing comparable performance against meticulously designed algorithms. Notably, it substantially outperforms existing LLM-based baselines in both code reliability and solution feasibility, achieving rates close to 100% on the evaluated benchmarks.

**arXiv ID:** 2510.16701
</details>

<details>
<summary><strong>See or Say Graphs: Agent-Driven Scalable Graph Understanding with Vision-Language Models</strong> - Shuo Han, Yukun Cao, Zezhong Ding, Zengyi Gao, S Kevin Zhou, Xike Xie - [[pdf]](https://arxiv.org/pdf/2510.16769)</summary>

**Abstract:** Vision-language models (VLMs) have shown promise in graph understanding, but remain limited by input-token constraints, facing scalability bottlenecks and lacking effective mechanisms to coordinate textual and visual modalities. To address these challenges, we propose GraphVista, a unified framework that enhances both scalability and modality coordination in graph understanding. For scalability, GraphVista organizes graph information hierarchically into a lightweight GraphRAG base, which retrieves only task-relevant textual descriptions and high-resolution visual subgraphs, compressing redundant context while preserving key reasoning elements. For modality coordination, GraphVista introduces a planning agent that routes tasks to the most suitable modality-using the text modality for simple property reasoning and the visual modality for local and structurally complex reasoning grounded in explicit topology. Extensive experiments demonstrate that GraphVista scales to large graphs, up to $200\times$ larger than those used in existing benchmarks, and consistently outperforms existing textual, visual, and fusion-based methods, achieving up to $4.4\times$ quality improvement over the state-of-the-art baselines by fully exploiting the complementary strengths of both modalities.

**arXiv ID:** 2510.16769
</details>

<details>
<summary><strong>ToolCritic: Detecting and Correcting Tool-Use Errors in Dialogue Systems</strong> - Hassan Hamad, Yingru Xu, Liang Zhao, Wenbo Yan, Narendra Gyanchandani - [[pdf]](https://arxiv.org/pdf/2510.17052)</summary>

**Abstract:** Tool-augmented large language models (LLMs) are increasingly employed in real-world applications, but tool usage errors still hinder their reliability. We introduce ToolCritic, a diagnostic framework that evaluates and improves LLM behavior in multi-turn, tool-augmented dialogues. ToolCritic detects eight distinct error types specific to tool-calling (e.g., premature invocation, argument misalignment, and misinterpretation of tool outputs) and provides targeted feedback to the main LLM. The main LLM, assumed to have strong reasoning, task understanding and orchestration capabilities, then revises its response based on ToolCritic's feedback. We systematically define these error categories and construct a synthetic dataset to train ToolCritic. Experimental results on the Schema-Guided Dialogue (SGD) dataset demonstrate that ToolCritic improves tool-calling accuracy by up to 13% over baselines, including zero-shot prompting and self-correction techniques. This represents a promising step toward more robust LLM integration with external tools in real-world dialogue applications.

**arXiv ID:** 2510.17052
</details>

<details>
<summary><strong>Physics-Informed Large Language Models for HVAC Anomaly Detection with Autonomous Rule Generation</strong> - Subin Lin, Chuanbo Hua - [[pdf]](https://arxiv.org/pdf/2510.17146)</summary>

**Abstract:** Heating, Ventilation, and Air-Conditioning (HVAC) systems account for a substantial share of global building energy use, making reliable anomaly detection essential for improving efficiency and reducing emissions. Classical rule-based approaches offer explainability but lack adaptability, while deep learning methods provide predictive power at the cost of transparency, efficiency, and physical plausibility. Recent attempts to use Large Language Models (LLMs) for anomaly detection improve interpretability but largely ignore the physical principles that govern HVAC operations. We present PILLM, a Physics-Informed LLM framework that operates within an evolutionary loop to automatically generate, evaluate, and refine anomaly detection rules. Our approach introduces physics-informed reflection and crossover operators that embed thermodynamic and control-theoretic constraints, enabling rules that are both adaptive and physically grounded. Experiments on the public Building Fault Detection dataset show that PILLM achieves state-of-the-art performance while producing diagnostic rules that are interpretable and actionable, advancing trustworthy and deployable AI for smart building systems.

**arXiv ID:** 2510.17146
</details>

<details>
<summary><strong>AsyncVoice Agent: Real-Time Explanation for LLM Planning and Reasoning</strong> - Yueqian Lin, Zhengmian Hu, Jayakumar Subramanian, Qinsi Wang, Nikos Vlassis, Hai "Helen" Li, Yiran Chen - [[pdf]](https://arxiv.org/pdf/2510.16156)</summary>

**Abstract:** Effective human-AI collaboration on complex reasoning tasks requires that users understand and interact with the model's process, not just receive an output. However, the monolithic text from methods like Chain-of-Thought (CoT) prevents this, as current interfaces lack real-time verbalization and robust user barge-in. We present AsyncVoice Agent, a system whose asynchronous architecture decouples a streaming LLM backend from a conversational voice frontend. This design allows narration and inference to run in parallel, empowering users to interrupt, query, and steer the model's reasoning process at any time. Objective benchmarks show this approach reduces interaction latency by more than 600x compared to monolithic baselines while ensuring high fidelity and competitive task accuracy. By enabling a two-way dialogue with a model's thought process, AsyncVoice Agent offers a new paradigm for building more effective, steerable, and trustworthy human-AI systems for high-stakes tasks.

**arXiv ID:** 2510.16156
</details>

<details>
<summary><strong>Detecting Adversarial Fine-tuning with Auditing Agents</strong> - Sarah Egler, John Schulman, Nicholas Carlini - [[pdf]](https://arxiv.org/pdf/2510.16255)</summary>

**Abstract:** Large Language Model (LLM) providers expose fine-tuning APIs that let end users fine-tune their frontier LLMs. Unfortunately, it has been shown that an adversary with fine-tuning access to an LLM can bypass safeguards. Particularly concerning, such attacks may avoid detection with datasets that are only implicitly harmful. Our work studies robust detection mechanisms for adversarial use of fine-tuning APIs. We introduce the concept of a fine-tuning auditing agent and show it can detect harmful fine-tuning prior to model deployment. We provide our auditing agent with access to the fine-tuning dataset, as well as the fine-tuned and pre-fine-tuned models, and request the agent assigns a risk score for the fine-tuning job. We evaluate our detection approach on a diverse set of eight strong fine-tuning attacks from the literature, along with five benign fine-tuned models, totaling over 1400 independent audits. These attacks are undetectable with basic content moderation on the dataset, highlighting the challenge of the task. With the best set of affordances, our auditing agent achieves a 56.2% detection rate of adversarial fine-tuning at a 1% false positive rate. Most promising, the auditor is able to detect covert cipher attacks that evade safety evaluations and content moderation of the dataset. While benign fine-tuning with unintentional subtle safety degradation remains a challenge, we establish a baseline configuration for further work in this area. We release our auditing agent at this https URL.

**arXiv ID:** 2510.16255
</details>

<details>
<summary><strong>A Vision for Access Control in LLM-based Agent Systems</strong> - Xinfeng Li, Dong Huang, Jie Li, Hongyi Cai, Zhenhong Zhou, Wei Dong, XiaoFeng Wang, Yang Liu - [[pdf]](https://arxiv.org/pdf/2510.11108)</summary>

**Abstract:** The autonomy and contextual complexity of LLM-based agents render traditional access control (AC) mechanisms insufficient. Static, rule-based systems designed for predictable environments are fundamentally ill-equipped to manage the dynamic information flows inherent in agentic interactions. This position paper argues for a paradigm shift from binary access control to a more sophisticated model of information governance, positing that the core challenge is not merely about permission, but about governing the flow of information. We introduce Agent Access Control (AAC), a novel framework that reframes AC as a dynamic, context-aware process of information flow governance. AAC operates on two core modules: (1) multi-dimensional contextual evaluation, which assesses not just identity but also relationships, scenarios, and norms; and (2) adaptive response formulation, which moves beyond simple allow/deny decisions to shape information through redaction, summarization, and paraphrasing. This vision, powered by a dedicated AC reasoning engine, aims to bridge the gap between human-like nuanced judgment and scalable Al safety, proposing a new conceptual lens for future research in trustworthy agent design.

**arXiv ID:** 2510.11108
</details>

<details>
<summary><strong>Evolution of AI Agent Registry Solutions: Centralized, Enterprise, and Distributed Approaches</strong> - Aditi Singh, Abul Ehtesham, Mahesh Lambe, Jared James Grogan, Abhishek Singh, Saket Kumar, Luca Muscariello, Vijoy Pandey, Guillaume Sauvage De Saint Marc, Pradyumna Chari, Ramesh Raskar - [[pdf]](https://arxiv.org/pdf/2508.03095)</summary>

**Abstract:** Autonomous AI agents now operate across cloud, enterprise, and decentralized domains, creating demand for registry infrastructures that enable trustworthy discovery, capability negotiation, and identity assurance. We analyze five prominent approaches: (1) MCP Registry (centralized publication of this http URL descriptors), (2) A2A Agent Cards (decentralized self-describing JSON capability manifests), (3) AGNTCY Agent Directory Service (IPFS Kademlia DHT content routing extended for semantic taxonomy-based content discovery, OCI artifact storage, and Sigstore-backed integrity), (4) Microsoft Entra Agent ID (enterprise SaaS directory with policy and zero-trust integration), and (5) NANDA Index AgentFacts (cryptographically verifiable, privacy-preserving fact model with credentialed assertions). Using four evaluation dimensions: security, authentication, scalability, and maintainability, we surface architectural trade-offs between centralized control, enterprise governance, and distributed resilience. We conclude with design recommendations for an emerging Internet of AI Agents requiring verifiable identity, adaptive discovery flows, and interoperable capability semantics.

**arXiv ID:** 2508.03095
</details>

<details>
<summary><strong>Does Visual Grounding Enhance the Understanding of Embodied Knowledge in Large Language Models?</strong> - Zhihui Yang, Yupei Wang, Kaijie Mo, Zhe Zhao, Renfen Hu - [[pdf]](https://arxiv.org/pdf/2510.16924)</summary>

**Abstract:** Despite significant progress in multimodal language models (LMs), it remains unclear whether visual grounding enhances their understanding of embodied knowledge compared to text-only models. To address this question, we propose a novel embodied knowledge understanding benchmark based on the perceptual theory from psychology, encompassing visual, auditory, tactile, gustatory, olfactory external senses, and interoception. The benchmark assesses the models' perceptual abilities across different sensory modalities through vector comparison and question-answering tasks with over 1,700 questions. By comparing 30 state-of-the-art LMs, we surprisingly find that vision-language models (VLMs) do not outperform text-only models in either task. Moreover, the models perform significantly worse in the visual dimension compared to other sensory dimensions. Further analysis reveals that the vector representations are easily influenced by word form and frequency, and the models struggle to answer questions involving spatial perception and reasoning. Our findings underscore the need for more effective integration of embodied knowledge in LMs to enhance their understanding of the physical world.

**arXiv ID:** 2510.16924
</details>

<details>
<summary><strong>HealthDial: A No-Code LLM-Assisted Dialogue Authoring Tool for Healthcare Virtual Agents</strong> - Farnaz Nouraei, Zhuorui Yong, Timothy Bickmore - [[pdf]](https://arxiv.org/pdf/2510.15898)</summary>

**Abstract:** We introduce HealthDial, a dialogue authoring tool that helps healthcare providers and educators create virtual agents that deliver health education and counseling to patients over multiple conversations. HealthDial leverages large language models (LLMs) to automatically create an initial session-based plan and conversations for each session using text-based patient health education materials as input. Authored dialogue is output in the form of finite state machines for virtual agent delivery so that all content can be validated and no unsafe advice is provided resulting from LLM hallucinations. LLM-drafted dialogue structure and language can be edited by the author in a no-code user interface to ensure validity and optimize clarity and impact. We conducted a feasibility and usability study with counselors and students to test our approach with an authoring task for cancer screening education. Participants used HealthDial and then tested their resulting dialogue by interacting with a 3D-animated virtual agent delivering the dialogue. Through participants' evaluations of the task experience and final dialogues, we show that HealthDial provides a promising first step for counselors to ensure full coverage of their health education materials, while creating understandable and actionable virtual agent dialogue with patients.

**arXiv ID:** 2510.15898
</details>

<details>
<summary><strong>Don't Trust Generative Agents to Mimic Communication on Social Networks Unless You Benchmarked their Empirical Realism</strong> - Simon Münker, Nils Schwager, Achim Rettinger - [[pdf]](https://arxiv.org/pdf/2506.21974)</summary>

**Abstract:** The ability of Large Language Models (LLMs) to mimic human behavior triggered a plethora of computational social science research, assuming that empirical studies of humans can be conducted with AI agents instead. Since there have been conflicting research findings on whether and when this hypothesis holds, there is a need to better understand the differences in their experimental designs. We focus on replicating the behavior of social network users with the use of LLMs for the analysis of communication on social networks. First, we provide a formal framework for the simulation of social networks, before focusing on the sub-task of imitating user communication. We empirically test different approaches to imitate user behavior on X in English and German. Our findings suggest that social simulations should be validated by their empirical realism measured in the setting in which the simulation components were fitted. With this paper, we argue for more rigor when applying generative-agent-based modeling for social simulation.

**arXiv ID:** 2506.21974
</details>

<details>
<summary><strong>FinResearchBench: A Logic Tree based Agent-as-a-Judge Evaluation Framework for Financial Research Agents</strong> - Rui Sun, Zuo Bai, Wentao Zhang, Yuxiang Zhang, Li Zhao, Shan Sun, Zhengwen Qiu - [[pdf]](https://arxiv.org/pdf/2507.16248)</summary>

**Abstract:** Recently, AI agents are rapidly evolving in intelligence and widely used in professional research applications, such as STEM, software development, and finance. Among these AI agents, deep research agent is a key category as it can perform long-horizon tasks and solve problems of greater complexity. However, there are few evaluation frameworks and benchmarks that systematically and automatically investigate the capabilities of these research agents. In addition, financial research problems have distinct complexity and subtlety. To fill in the gap, we propose FinResearchBench, which is a logic tree-based Agent-as-a-Judge and targets specifically for the financial research agents. It provides a comprehensive and automatic assessment of the research agents across 7 key types of tasks in the financial research domain. The contributions of this work are two-folded: (1) the first and innovative Agent-as-a-Judge system that extracts the logic tree of the research outcome and uses it as the intermediate information to present a comprehensive, reliable, and robust evaluation; (2) finance-oriented that it covers 70 typical financial research questions, spreading across 7 frequently encountered types of task in the domain.

**arXiv ID:** 2507.16248
</details>

<details>
<summary><strong>DINO-CVA: A Multimodal Goal-Conditioned Vision-to-Action Model for Autonomous Catheter Navigation</strong> - Pedram Fekri, Majid Roshanfar, Samuel Barbeau, Seyedfarzad Famouri, Thomas Looi, Dale Podolsky, Mehrdad Zadeh, Javad Dargahi - [[pdf]](https://arxiv.org/pdf/2510.17038)</summary>

**Abstract:** Cardiac catheterization remains a cornerstone of minimally invasive interventions, yet it continues to rely heavily on manual operation. Despite advances in robotic platforms, existing systems are predominantly follow-leader in nature, requiring continuous physician input and lacking intelligent autonomy. This dependency contributes to operator fatigue, more radiation exposure, and variability in procedural outcomes. This work moves towards autonomous catheter navigation by introducing DINO-CVA, a multimodal goal-conditioned behavior cloning framework. The proposed model fuses visual observations and joystick kinematics into a joint embedding space, enabling policies that are both vision-aware and kinematic-aware. Actions are predicted autoregressively from expert demonstrations, with goal conditioning guiding navigation toward specified destinations. A robotic experimental setup with a synthetic vascular phantom was designed to collect multimodal datasets and evaluate performance. Results show that DINO-CVA achieves high accuracy in predicting actions, matching the performance of a kinematics-only baseline while additionally grounding predictions in the anatomical environment. These findings establish the feasibility of multimodal, goal-conditioned architectures for catheter navigation, representing an important step toward reducing operator dependency and improving the reliability of catheterbased therapies.

**arXiv ID:** 2510.17038
</details>

<details>
<summary><strong>Advancing Off-Road Autonomous Driving: The Large-Scale ORAD-3D Dataset and Comprehensive Benchmarks</strong> - Chen Min, Jilin Mei, Heng Zhai, Shuai Wang, Tong Sun, Fanjie Kong, Haoyang Li, Fangyuan Mao, Fuyang Liu, Shuo Wang, Yiming Nie, Qi Zhu, Liang Xiao, Dawei Zhao, Yu Hu - [[pdf]](https://arxiv.org/pdf/2510.16500)</summary>

**Abstract:** A major bottleneck in off-road autonomous driving research lies in the scarcity of large-scale, high-quality datasets and benchmarks. To bridge this gap, we present ORAD-3D, which, to the best of our knowledge, is the largest dataset specifically curated for off-road autonomous driving. ORAD-3D covers a wide spectrum of terrains, including woodlands, farmlands, grasslands, riversides, gravel roads, cement roads, and rural areas, while capturing diverse environmental variations across weather conditions (sunny, rainy, foggy, and snowy) and illumination levels (bright daylight, daytime, twilight, and nighttime). Building upon this dataset, we establish a comprehensive suite of benchmark evaluations spanning five fundamental tasks: 2D free-space detection, 3D occupancy prediction, rough GPS-guided path planning, vision-language model-driven autonomous driving, and world model for off-road environments. Together, the dataset and benchmarks provide a unified and robust resource for advancing perception and planning in challenging off-road scenarios. The dataset and code will be made publicly available at this https URL.

**arXiv ID:** 2510.16500
</details>

<details>
<summary><strong>Robobench: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain</strong> - Yulin Luo, Chun-Kai Fan, Menghang Dong, Jiayu Shi, Mengdi Zhao, Bo-Wen Zhang, Cheng Chi, Jiaming Liu, Gaole Dai, Rongyu Zhang, Ruichuan An, Kun Wu, Zhengping Che, Shaoxuan Xie, Guocai Yao, Zhongxia Zhao, Pengwei Wang, Guang Liu, Zhongyuan Wang, Tiejun Huang, Shanghang Zhang - [[pdf]](https://arxiv.org/pdf/2510.17801)</summary>

**Abstract:** Building robots that can perceive, reason, and act in dynamic, unstructured environments remains a core challenge. Recent embodied systems often adopt a dual-system paradigm, where System 2 handles high-level reasoning while System 1 executes low-level control. In this work, we refer to System 2 as the embodied brain, emphasizing its role as the cognitive core for reasoning and decision-making in manipulation tasks. Given this role, systematic evaluation of the embodied brain is essential. Yet existing benchmarks emphasize execution success, or when targeting high-level reasoning, suffer from incomplete dimensions and limited task realism, offering only a partial picture of cognitive capability. To bridge this gap, we introduce RoboBench, a benchmark that systematically evaluates multimodal large language models (MLLMs) as embodied brains. Motivated by the critical roles across the full manipulation pipeline, RoboBench defines five dimensions-instruction comprehension, perception reasoning, generalized planning, affordance prediction, and failure analysis-spanning 14 capabilities, 25 tasks, and 6092 QA pairs. To ensure realism, we curate datasets across diverse embodiments, attribute-rich objects, and multi-view scenes, drawing from large-scale real robotic data. For planning, RoboBench introduces an evaluation framework, MLLM-as-world-simulator. It evaluate embodied feasibility by simulating whether predicted plans can achieve critical object-state changes. Experiments on 14 MLLMs reveal fundamental limitations: difficulties with implicit instruction comprehension, spatiotemporal reasoning, cross-scenario planning, fine-grained affordance understanding, and execution failure diagnosis. RoboBench provides a comprehensive scaffold to quantify high-level cognition, and guide the development of next-generation embodied MLLMs. The project page is in this https URL.

**arXiv ID:** 2510.17801
</details>

<details>
<summary><strong>Interpretable Decision-Making for End-to-End Autonomous Driving</strong> - Mona Mirzaie, Bodo Rosenhahn - [[pdf]](https://arxiv.org/pdf/2508.18898)</summary>

**Abstract:** Trustworthy AI is mandatory for the broad deployment of autonomous vehicles. Although end-to-end approaches derive control commands directly from raw data, interpreting these decisions remains challenging, especially in complex urban scenarios. This is mainly attributed to very deep neural networks with non-linear decision boundaries, making it challenging to grasp the logic behind AI-driven decisions. This paper presents a method to enhance interpretability while optimizing control commands in autonomous driving. To address this, we propose loss functions that promote the interpretability of our model by generating sparse and localized feature maps. The feature activations allow us to explain which image regions contribute to the predicted control command. We conduct comprehensive ablation studies on the feature extraction step and validate our method on the CARLA benchmarks. We also demonstrate that our approach improves interpretability, which correlates with reducing infractions, yielding a safer, high-performance driving model. Notably, our monocular, non-ensemble model surpasses the top-performing approaches from the CARLA Leaderboard by achieving lower infraction scores and the highest route completion rate, all while ensuring interpretability.

**arXiv ID:** 2508.18898
</details>

</details>

<details open>
<summary><h2>LLM Agents (7 papers)</h2></summary>

<details>
<summary><strong>LinearizeLLM: An Agent-Based Framework for LLM-Driven Exact Linear Reformulation of Nonlinear Optimization Problems</strong> - Paul-Niklas Ken Kandora, Simon Caspar Zeller, Aaron Jeremias Elsing, Elena Kuss, Steffen Rebennack - [[pdf]](https://arxiv.org/pdf/2510.15969)</summary>

**Abstract:** Reformulating nonlinear optimization problems is largely manual and expertise-intensive, yet it remains essential for solving such problems with linear optimization solvers or applying special-purpose algorithms. We introduce \textit{LinearizeLLM}, an agent-based framework that solves this task by leveraging Large Language Models (LLMs). The framework assigns each nonlinear pattern to a \textit{reformulation agent} that is explicitly instructed to derive an exact linear reformulation for its nonlinearity pattern, for instance, absolute-value terms or bilinear products of decision variables. The agents then coordinate to assemble a solver-ready linear model equivalent to the original problem. To benchmark the approach, we create a dataset of 20 real-world nonlinear optimization problems derived from the established ComplexOR dataset of linear optimization problems. We evaluate our approach with several LLMs. Our results indicate that specialized LLM agents can automate linearization tasks, opening a path toward fully conversational modeling pipelines for nonlinear optimization.

**arXiv ID:** 2510.15969
</details>

<details>
<summary><strong>MCP Security Bench (MSB): Benchmarking Attacks Against Model Context Protocol in LLM Agents</strong> - Dongsen Zhang, Zekun Li, Xu Luo, Xuannan Liu, Peipei Li, Wenjun Xu - [[pdf]](https://arxiv.org/pdf/2510.15994)</summary>

**Abstract:** The Model Context Protocol (MCP) standardizes how large language model (LLM) agents discover, describe, and call external tools. While MCP unlocks broad interoperability, it also enlarges the attack surface by making tools first-class, composable objects with natural-language metadata, and standardized I/O. We present MSB (MCP Security Benchmark), the first end-to-end evaluation suite that systematically measures how well LLM agents resist MCP-specific attacks throughout the full tool-use pipeline: task planning, tool invocation, and response handling. MSB contributes: (1) a taxonomy of 12 attacks including name-collision, preference manipulation, prompt injections embedded in tool descriptions, out-of-scope parameter requests, user-impersonating responses, false-error escalation, tool-transfer, retrieval injection, and mixed attacks; (2) an evaluation harness that executes attacks by running real tools (both benign and malicious) via MCP rather than simulation; and (3) a robustness metric that quantifies the trade-off between security and performance: Net Resilient Performance (NRP). We evaluate nine popular LLM agents across 10 domains and 400+ tools, producing 2,000 attack instances. Results reveal the effectiveness of attacks against each stage of MCP. Models with stronger performance are more vulnerable to attacks due to their outstanding tool calling and instruction following capabilities. MSB provides a practical baseline for researchers and practitioners to study, compare, and harden MCP agents.

**arXiv ID:** 2510.15994
</details>

<details>
<summary><strong>EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle</strong> - Rong Wu, Xiaoman Wang, Jianbiao Mei, Pinlong Cai, Daocheng Fu, Cheng Yang, Licheng Wen, Xuemeng Yang, Yufan Shen, Yuxin Wang, Botian Shi - [[pdf]](https://arxiv.org/pdf/2510.16079)</summary>

**Abstract:** Current Large Language Model (LLM) agents show strong performance in tool use, but lack the crucial capability to systematically learn from their own experiences. While existing frameworks mainly focus on mitigating external knowledge gaps, they fail to address a more fundamental limitation: the inability to iteratively refine problem-solving strategies. In this work, we introduce EvolveR, a framework designed to enable agent to self-improve through a complete, closed-loop experience lifecycle. This lifecycle comprises two key stages: (1) Offline Self-Distillation, where the agent's interaction trajectories are synthesized into a structured repository of abstract, reusable strategic principles; (2) Online Interaction, where the agent interacts with tasks and actively retrieves distilled principles to guide its decision-making, accumulating a diverse set of behavioral trajectories. This loop employs a policy reinforcement mechanism to iteratively update the agent based on its performance. We demonstrate the effectiveness of EvolveR on complex multi-hop question-answering benchmarks, where it achieves superior performance over strong agentic baselines. Our work presents a comprehensive blueprint for agents that learn not only from external data but also from the consequences of their own actions, paving the way for more autonomous and continuously improving systems. Code is available at this https URL.

**arXiv ID:** 2510.16079
</details>

<details>
<summary><strong>Value-Based Large Language Model Agent Simulation for Mutual Evaluation of Trust and Interpersonal Closeness</strong> - Yuki Sakamoto, Takahisa Uchida, Hiroshi Ishiguro - [[pdf]](https://arxiv.org/pdf/2507.11979)</summary>

**Abstract:** Large language models (LLMs) have emerged as powerful tools for simulating complex social phenomena using human-like agents with specific traits. In human societies, value similarity is important for building trust and close relationships; however, it remains unexplored whether this principle holds true in artificial societies comprising LLM agents. Therefore, this study investigates the influence of value similarity on relationship-building among LLM agents through two experiments. First, in a preliminary experiment, we evaluated the controllability of values in LLMs to identify the most effective model and prompt design for controlling the values. Subsequently, in the main experiment, we generated pairs of LLM agents imbued with specific values and analyzed their mutual evaluations of trust and interpersonal closeness following a dialogue. The experiments were conducted in English and Japanese to investigate language dependence. The results confirmed that pairs of agents with higher value similarity exhibited greater mutual trust and interpersonal closeness. Our findings demonstrate that the LLM agent simulation serves as a valid testbed for social science theories, contributes to elucidating the mechanisms by which values influence relationship building, and provides a foundation for inspiring new theories and insights into the social sciences.

**arXiv ID:** 2507.11979
</details>

<details>
<summary><strong>Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety</strong> - Vamshi Krishna Bonagiri, Ponnurangam Kumaragurum, Khanh Nguyen, Benjamin Plaut - [[pdf]](https://arxiv.org/pdf/2510.16492)</summary>

**Abstract:** As Large Language Model (LLM) agents increasingly operate in complex environments with real-world consequences, their safety becomes critical. While uncertainty quantification is well-studied for single-turn tasks, multi-turn agentic scenarios with real-world tool access present unique challenges where uncertainties and ambiguities compound, leading to severe or catastrophic risks beyond traditional text generation failures. We propose using "quitting" as a simple yet effective behavioral mechanism for LLM agents to recognize and withdraw from situations where they lack confidence. Leveraging the ToolEmu framework, we conduct a systematic evaluation of quitting behavior across 12 state-of-the-art LLMs. Our results demonstrate a highly favorable safety-helpfulness trade-off: agents prompted to quit with explicit instructions improve safety by an average of +0.39 on a 0-3 scale across all models (+0.64 for proprietary models), while maintaining a negligible average decrease of -0.03 in helpfulness. Our analysis demonstrates that simply adding explicit quit instructions proves to be a highly effective safety mechanism that can immediately be deployed in existing agent systems, and establishes quitting as an effective first-line defense mechanism for autonomous agents in high-stakes applications.

**arXiv ID:** 2510.16492
</details>

<details>
<summary><strong>Empowering Real-World: A Survey on the Technology, Practice, and Evaluation of LLM-driven Industry Agents</strong> - Yihong Tang, Kehai Chen, Liang Yue, Jinxin Fan, Caishen Zhou, Xiaoguang Li, Yuyang Zhang, Mingming Zhao, Shixiong Kai, Kaiyang Guo, Xingshan Zeng, Wenjing Cun, Lifeng Shang, Min Zhang - [[pdf]](https://arxiv.org/pdf/2510.17491)</summary>

**Abstract:** With the rise of large language models (LLMs), LLM agents capable of autonomous reasoning, planning, and executing complex tasks have become a frontier in artificial intelligence. However, how to translate the research on general agents into productivity that drives industry transformations remains a significant challenge. To address this, this paper systematically reviews the technologies, applications, and evaluation methods of industry agents based on LLMs. Using an industry agent capability maturity framework, it outlines the evolution of agents in industry applications, from "process execution systems" to "adaptive social systems." First, we examine the three key technological pillars that support the advancement of agent capabilities: Memory, Planning, and Tool Use. We discuss how these technologies evolve from supporting simple tasks in their early forms to enabling complex autonomous systems and collective intelligence in more advanced forms. Then, we provide an overview of the application of industry agents in real-world domains such as digital engineering, scientific discovery, embodied intelligence, collaborative business execution, and complex system simulation. Additionally, this paper reviews the evaluation benchmarks and methods for both fundamental and specialized capabilities, identifying the challenges existing evaluation systems face regarding authenticity, safety, and industry specificity. Finally, we focus on the practical challenges faced by industry agents, exploring their capability boundaries, developmental potential, and governance issues in various scenarios, while providing insights into future directions. By combining technological evolution with industry practices, this review aims to clarify the current state and offer a clear roadmap and theoretical foundation for understanding and building the next generation of industry agents.

**arXiv ID:** 2510.17491
</details>

<details>
<summary><strong>Customer-R1: Personalized Simulation of Human Behaviors via RL-based LLM Agent in Online Shopping</strong> - Ziyi Wang, Yuxuan Lu, Yimeng Zhang, Jing Huang, Dakuo Wang - [[pdf]](https://arxiv.org/pdf/2510.07230)</summary>

**Abstract:** Simulating step-wise human behavior with Large Language Models (LLMs) has become an emerging research direction, enabling applications in various practical domains. While prior methods, including prompting, supervised fine-tuning (SFT), and reinforcement learning (RL), have shown promise in modeling step-wise behavior, they primarily learn a population-level policy without conditioning on a user's persona, yielding generic rather than personalized simulations. In this work, we pose a critical question: how can LLM agents better simulate personalized user behavior? We introduce Customer-R1, an RL-based method for personalized, step-wise user behavior simulation in online shopping environments. Our policy is conditioned on an explicit persona, and we optimize next-step rationale and action generation via action correctness reward signals. Experiments on the OPeRA dataset emonstrate that Customer-R1 not only significantly outperforms prompting and SFT-based baselines in next-action prediction tasks, but also better matches users' action distribution, indicating higher fidelity in personalized behavior simulation.

**arXiv ID:** 2510.07230
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (30 papers)</h2></summary>

<details>
<summary><strong>Towards Automatic Evaluation and Selection of PHI De-identification Models via Multi-Agent Collaboration</strong> - Guanchen Wu, Zuhui Chen, Yuzhang Xie, Carl Yang - [[pdf]](https://arxiv.org/pdf/2510.16194)</summary>

**Abstract:** Protected health information (PHI) de-identification is critical for enabling the safe reuse of clinical notes, yet evaluating and comparing PHI de-identification models typically depends on costly, small-scale expert annotations. We present TEAM-PHI, a multi-agent evaluation and selection framework that uses large language models (LLMs) to automatically measure de-identification quality and select the best-performing model without heavy reliance on gold labels. TEAM-PHI deploys multiple Evaluation Agents, each independently judging the correctness of PHI extractions and outputting structured metrics. Their results are then consolidated through an LLM-based majority voting mechanism that integrates diverse evaluator perspectives into a single, stable, and reproducible ranking. Experiments on a real-world clinical note corpus demonstrate that TEAM-PHI produces consistent and accurate rankings: despite variation across individual evaluators, LLM-based voting reliably converges on the same top-performing systems. Further comparison with ground-truth annotations and human evaluation confirms that the framework's automated rankings closely match supervised evaluation. By combining independent evaluation agents with LLM majority voting, TEAM-PHI offers a practical, secure, and cost-effective solution for automatic evaluation and best-model selection in PHI de-identification, even when ground-truth labels are limited.

**arXiv ID:** 2510.16194
</details>

<details>
<summary><strong>Beyond Pipelines: A Survey of the Paradigm Shift toward Model-Native Agentic AI</strong> - Jitao Sang, Jinlin Xiao, Jiarun Han, Jilin Chen, Xiaoyi Chen, Shuyu Wei, Yongjie Sun, Yuhang Wang - [[pdf]](https://arxiv.org/pdf/2510.16720)</summary>

**Abstract:** The rapid evolution of agentic AI marks a new phase in artificial intelligence, where Large Language Models (LLMs) no longer merely respond but act, reason, and adapt. This survey traces the paradigm shift in building agentic AI: from Pipeline-based systems, where planning, tool use, and memory are orchestrated by external logic, to the emerging Model-native paradigm, where these capabilities are internalized within the model's parameters. We first position Reinforcement Learning (RL) as the algorithmic engine enabling this paradigm shift. By reframing learning from imitating static data to outcome-driven exploration, RL underpins a unified solution of LLM + RL + Task across language, vision and embodied domains. Building on this, the survey systematically reviews how each capability -- Planning, Tool use, and Memory -- has evolved from externally scripted modules to end-to-end learned behaviors. Furthermore, it examines how this paradigm shift has reshaped major agent applications, specifically the Deep Research agent emphasizing long-horizon reasoning and the GUI agent emphasizing embodied interaction. We conclude by discussing the continued internalization of agentic capabilities like Multi-agent collaboration and Reflection, alongside the evolving roles of the system and model layers in future agentic AI. Together, these developments outline a coherent trajectory toward model-native agentic AI as an integrated learning and interaction framework, marking the transition from constructing systems that apply intelligence to developing models that grow intelligence through experience.

**arXiv ID:** 2510.16720
</details>

<details>
<summary><strong>STARK: Strategic Team of Agents for Refining Kernels</strong> - Juncheng Dong, Yang Yang, Tao Liu, Yang Wang, Feng Qi, Vahid Tarokh, Kaushik Rangadurai, Shuang Yang - [[pdf]](https://arxiv.org/pdf/2510.16996)</summary>

**Abstract:** The efficiency of GPU kernels is central to the progress of modern AI, yet optimizing them remains a difficult and labor-intensive task due to complex interactions between memory hierarchies, thread scheduling, and hardware-specific characteristics. While recent advances in large language models (LLMs) provide new opportunities for automated code generation, existing approaches largely treat LLMs as single-shot generators or naive refinement tools, limiting their effectiveness in navigating the irregular kernel optimization landscape. We introduce an LLM agentic framework for GPU kernel optimization that systematically explores the design space through multi-agent collaboration, grounded instruction, dynamic context management, and strategic search. This framework mimics the workflow of expert engineers, enabling LLMs to reason about hardware trade-offs, incorporate profiling feedback, and refine kernels iteratively. We evaluate our approach on KernelBench, a benchmark for LLM-based kernel optimization, and demonstrate substantial improvements over baseline agents: our system produces correct solutions where baselines often fail, and achieves kernels with up to 16x faster runtime performance. These results highlight the potential of agentic LLM frameworks to advance fully automated, scalable GPU kernel optimization.

**arXiv ID:** 2510.16996
</details>

<details>
<summary><strong>A Brain Cell Type Resource Created by Large Language Models and a Multi-Agent AI System for Collaborative Community Annotation</strong> - Rongbin Li, Wenbo Chen, Zhao Li, Rodrigo Munoz-Castaneda, Jinbo Li, Neha S. Maurya, Arnav Solanki, Huan He, Hanwen Xing, Meaghan Ramlakhan, Zachary Wise, Zhuhao Wu, Hua Xu, Michael Hawrylycz, W. Jim Zheng - [[pdf]](https://arxiv.org/pdf/2510.17064)</summary>

**Abstract:** Single-cell RNA sequencing has transformed our ability to identify diverse cell types and their transcriptomic signatures. However, annotating these signatures-especially those involving poorly characterized genes-remains a major challenge. Traditional methods, such as Gene Set Enrichment Analysis (GSEA), depend on well-curated annotations and often perform poorly in these contexts. Large Language Models (LLMs) offer a promising alternative but struggle to represent complex biological knowledge within structured ontologies. To address this, we present BRAINCELL-AID (BRAINCELL-AID: this https URL), a novel multi-agent AI system that integrates free-text descriptions with ontology labels to enable more accurate and robust gene set annotation. By incorporating retrieval-augmented generation (RAG), we developed a robust agentic workflow that refines predictions using relevant PubMed literature, reducing hallucinations and enhancing interpretability. Using this workflow, we achieved correct annotations for 77% of mouse gene sets among their top predictions. Applying this approach, we annotated 5,322 brain cell clusters from the comprehensive mouse brain cell atlas generated by the BRAIN Initiative Cell Census Network, enabling novel insights into brain cell function by identifying region-specific gene co-expression patterns and inferring functional roles of gene ensembles. BRAINCELL-AID also identifies Basal Ganglia-related cell types with neurologically meaningful descriptions. Hence, we create a valuable resource to support community-driven cell type annotation.

**arXiv ID:** 2510.17064
</details>

<details>
<summary><strong>Which LLM Multi-Agent Protocol to Choose?</strong> - Hongyi Du, Jiaqi Su, Jisen Li, Lijie Ding, Yingxuan Yang, Peixuan Han, Xiangru Tang, Kunlun Zhu, Jiaxuan You - [[pdf]](https://arxiv.org/pdf/2510.17149)</summary>

**Abstract:** As large-scale multi-agent systems evolve, the communication protocol layer has become a critical yet under-evaluated factor shaping performance and reliability. Despite the existence of diverse protocols (A2A, ACP, ANP, Agora, etc.), selection is often intuition-driven and lacks standardized guidance. We introduce ProtocolBench, a benchmark that systematically compares agent protocols along four measurable axes: task success, end-to-end latency, message or byte overhead, and robustness under failures. On ProtocolBench, protocol choice significantly influences system behavior. In the Streaming Queue scenario, overall completion time varies by up to 36.5% across protocols, and mean end-to-end latency differs by 3.48 s. Under Fail-Storm Recovery, resilience also differs consistently across protocols. Beyond evaluation, we present ProtocolRouter, a learnable protocol router that selects per-scenario (or per-module) protocols from requirement and runtime signals. ProtocolRouter reduces Fail-Storm recovery time by up to 18.1% versus the best single-protocol baseline, and achieves scenario-specific gains such as higher success in GAIA. We also release ProtocolRouterBench to standardize protocol evaluation and improve reliability at scale.

**arXiv ID:** 2510.17149
</details>

<details>
<summary><strong>Coinvisor: An RL-Enhanced Chatbot Agent for Interactive Cryptocurrency Investment Analysis</strong> - Chong Chen, Ze Liu, Lingfeng Bao, Yanlin Wang, Ting Chen, Daoyuan Wu, Jiachi Chen - [[pdf]](https://arxiv.org/pdf/2510.17235)</summary>

**Abstract:** The cryptocurrency market offers significant investment opportunities but faces challenges including high volatility and fragmented information. Data integration and analysis are essential for informed investment decisions. Currently, investors use three main approaches: (1) Manual analysis across various sources, which depends heavily on individual experience and is time-consuming and prone to bias; (2) Data aggregation platforms-limited in functionality and depth of analysis; (3) Large language model agents-based on static pretrained models, lacking real-time data integration and multi-step reasoning capabilities. To address these limitations, we present Coinvisor, a reinforcement learning-based chatbot that provides comprehensive analytical support for cryptocurrency investment through a multi-agent framework. Coinvisor integrates diverse analytical capabilities through specialized tools. Its key innovation is a reinforcement learning-based tool selection mechanism that enables multi-step planning and flexible integration of diverse data sources. This design supports real-time interaction and adaptive analysis of dynamic content, delivering accurate and actionable investment insights. We evaluated Coinvisor through automated benchmarks on tool calling accuracy and user studies with 20 cryptocurrency investors using our interface. Results show that Coinvisor improves recall by 40.7% and F1 score by 26.6% over the base model in tool orchestration. User studies show high satisfaction (4.64/5), with participants preferring Coinvisor to both general LLMs and existing crypto platforms (4.62/5).

**arXiv ID:** 2510.17235
</details>

<details>
<summary><strong>Graph Attention-Guided Search for Dense Multi-Agent Pathfinding</strong> - Rishabh Jain, Keisuke Okumura, Michael Amir, Amanda Prorok - [[pdf]](https://arxiv.org/pdf/2510.17382)</summary>

**Abstract:** Finding near-optimal solutions for dense multi-agent pathfinding (MAPF) problems in real-time remains challenging even for state-of-the-art planners. To this end, we develop a hybrid framework that integrates a learned heuristic derived from MAGAT, a neural MAPF policy with a graph attention scheme, into a leading search-based algorithm, LaCAM. While prior work has explored learning-guided search in MAPF, such methods have historically underperformed. In contrast, our approach, termed LaGAT, outperforms both purely search-based and purely learning-based methods in dense scenarios. This is achieved through an enhanced MAGAT architecture, a pre-train-then-fine-tune strategy on maps of interest, and a deadlock detection scheme to account for imperfect neural guidance. Our results demonstrate that, when carefully designed, hybrid search offers a powerful solution for tightly coupled, challenging multi-agent coordination problems.

**arXiv ID:** 2510.17382
</details>

<details>
<summary><strong>A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning</strong> - Anjie Liu, Jianhong Wang, Samuel Kaski, Jun Wang, Mengyue Yang - [[pdf]](https://arxiv.org/pdf/2510.17697)</summary>

**Abstract:** Steering cooperative multi-agent reinforcement learning (MARL) towards desired outcomes is challenging, particularly when the global guidance from a human on the whole multi-agent system is impractical in a large-scale MARL. On the other hand, designing mechanisms to coordinate agents most relies on empirical studies, lacking a easy-to-use research tool. In this work, we employ multi-agent influence diagrams (MAIDs) as a graphical framework to address the above issues. First, we introduce interaction paradigms that leverage MAIDs to analyze and visualize existing approaches in MARL. Then, we design a new interaction paradigm based on MAIDs, referred to as targeted intervention that is applied to only a single targeted agent, so the problem of global guidance can be mitigated. In our implementation, we introduce a causal inference technique-referred to as Pre-Strategy Intervention (PSI)-to realize the targeted intervention paradigm. Since MAIDs can be regarded as a special class of causal diagrams, a composite desired outcome that integrates the primary task goal and an additional desired outcome can be achieved by maximizing the corresponding causal effect through the PSI. Moreover, the bundled relevance graph analysis of MAIDs provides a tool to identify whether an MARL learning paradigm is workable under the design of an interaction paradigm. In experiments, we demonstrate the effectiveness of our proposed targeted intervention, and verify the result of relevance graph analysis.

**arXiv ID:** 2510.17697
</details>

<details>
<summary><strong>ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination</strong> - Charidimos Papadakis, Angeliki Dimitriou, Giorgos Filandrianos, Maria Lymperaiou, Konstantinos Thomas, Giorgos Stamou - [[pdf]](https://arxiv.org/pdf/2510.15949)</summary>

**Abstract:** Large language models show promise for financial decision-making, yet deploying them as autonomous trading agents raises fundamental challenges: how to adapt instructions when rewards arrive late and obscured by market noise, how to synthesize heterogeneous information streams into coherent decisions, and how to bridge the gap between model outputs and executable market actions. We present ATLAS (Adaptive Trading with LLM AgentS), a unified multi-agent framework that integrates structured information from markets, news, and corporate fundamentals to support robust trading decisions. Within ATLAS, the central trading agent operates in an order-aware action space, ensuring that outputs correspond to executable market orders rather than abstract signals. The agent can incorporate feedback while trading using Adaptive-OPRO, a novel prompt-optimization technique that dynamically adapts the prompt by incorporating real-time, stochastic feedback, leading to increasing performance over time. Across regime-specific equity studies and multiple LLM families, Adaptive-OPRO consistently outperforms fixed prompts, while reflection-based feedback fails to provide systematic gains.

**arXiv ID:** 2510.15949
</details>

<details>
<summary><strong>Disaster Management in the Era of Agentic AI Systems: A Vision for Collective Human-Machine Intelligence for Augmented Resilience</strong> - Bo Li, Junwei Ma, Kai Yin, Yiming Xiao, Chia-Wei Hsu, Ali Mostafavi - [[pdf]](https://arxiv.org/pdf/2510.16034)</summary>

**Abstract:** The escalating frequency and severity of disasters routinely overwhelm traditional response capabilities, exposing critical vulnerability in disaster management. Current practices are hindered by fragmented data streams, siloed technologies, resource constraints, and the erosion of institutional memory, which collectively impede timely and effective decision making. This study introduces Disaster Copilot, a vision for a multi-agent artificial intelligence system designed to overcome these systemic challenges by unifying specialized AI tools within a collaborative framework. The proposed architecture utilizes a central orchestrator to coordinate diverse sub-agents, each specializing in critical domains such as predictive risk analytics, situational awareness, and impact assessment. By integrating multi-modal data, the system delivers a holistic, real-time operational picture and serve as the essential AI backbone required to advance Disaster Digital Twins from passive models to active, intelligent environments. Furthermore, it ensures functionality in resource-limited environments through on-device orchestration and incorporates mechanisms to capture institutional knowledge, mitigating the impact of staff turnover. We detail the system architecture and propose a three-phased roadmap emphasizing the parallel growth of technology, organizational capacity, and human-AI teaming. Disaster Copilot offers a transformative vision, fostering collective human-machine intelligence to build more adaptive, data-driven and resilient communities.

**arXiv ID:** 2510.16034
</details>

<details>
<summary><strong>TriAgent: Automated Biomarker Discovery with Deep Research Grounding for Triage in Acute Care by LLM-Based Multi-Agent Collaboration</strong> - Kerem Delikoyun, Qianyu Chen, Win Sen Kuan, John Tshon Yit Soong, Matthew Edward Cove, Oliver Hayden - [[pdf]](https://arxiv.org/pdf/2510.16080)</summary>

**Abstract:** Emergency departments worldwide face rising patient volumes, workforce shortages, and variability in triage decisions that threaten the delivery of timely and accurate care. Current triage methods rely primarily on vital signs, routine laboratory values, and clinicians' judgment, which, while effective, often miss emerging biological signals that could improve risk prediction for infection typing or antibiotic administration in acute conditions. To address this challenge, we introduce TriAgent, a large language model (LLM)-based multi-agent framework that couples automated biomarker discovery with deep research for literature-grounded validation and novelty assessment. TriAgent employs a supervisor research agent to generate research topics and delegate targeted queries to specialized sub-agents for evidence retrieval from various data sources. Findings are synthesized to classify biomarkers as either grounded in existing knowledge or flagged as novel candidates, offering transparent justification and highlighting unexplored pathways in acute care risk stratification. Unlike prior frameworks limited to existing routine clinical biomarkers, TriAgent aims to deliver an end-to-end framework from data analysis to literature grounding to improve transparency, explainability and expand the frontier of potentially actionable clinical biomarkers. Given a user's clinical query and quantitative triage data, TriAgent achieved a topic adherence F1 score of 55.7 +/- 5.0%, surpassing the CoT-ReAct agent by over 10%, and a faithfulness score of 0.42 +/- 0.39, exceeding all baselines by more than 50%. Across experiments, TriAgent consistently outperformed state-of-the-art LLM-based agentic frameworks in biomarker justification and literature-grounded novelty assessment. We share our repo: this https URL.

**arXiv ID:** 2510.16080
</details>

<details>
<summary><strong>Agentic AI for Ultra-Modern Networks: Multi-Agent Framework for RAN Autonomy and Assurance</strong> - Sukhdeep Singh, Avinash Bhat, Shweta M, Subhash K Singh, Moonki Hong, Madhan Raj K, Kandeepan Sithamparanathan, Sunder A. Khowaja, Kapal Dev - [[pdf]](https://arxiv.org/pdf/2510.16144)</summary>

**Abstract:** The increasing complexity of Beyond 5G and 6G networks necessitates new paradigms for autonomy and assur- ance. Traditional O-RAN control loops rely heavily on RIC- based orchestration, which centralizes intelligence and exposes the system to risks such as policy conflicts, data drift, and unsafe actions under unforeseen conditions. In this work, we argue that the future of autonomous networks lies in a multi-agentic architecture, where specialized agents collaborate to perform data collection, model training, prediction, policy generation, verification, deployment, and assurance. By replacing tightly- coupled centralized RIC-based workflows with distributed agents, the framework achieves autonomy, resilience, explainability, and system-wide safety. To substantiate this vision, we design and evaluate a traffic steering use case under surge and drift conditions. Results across four KPIs: RRC connected users, IP throughput, PRB utilization, and SINR, demonstrate that a naive predictor-driven deployment improves local KPIs but destabilizes neighbors, whereas the agentic system blocks unsafe policies, preserving global network health. This study highlights multi- agent architectures as a credible foundation for trustworthy AI- driven autonomy in next-generation RANs.

**arXiv ID:** 2510.16144
</details>

<details>
<summary><strong>SentinelNet: Safeguarding Multi-Agent Collaboration Through Credit-Based Dynamic Threat Detection</strong> - Yang Feng, Xudong Pan - [[pdf]](https://arxiv.org/pdf/2510.16219)</summary>

**Abstract:** Malicious agents pose significant threats to the reliability and decision-making capabilities of Multi-Agent Systems (MAS) powered by Large Language Models (LLMs). Existing defenses often fall short due to reactive designs or centralized architectures which may introduce single points of failure. To address these challenges, we propose SentinelNet, the first decentralized framework for proactively detecting and mitigating malicious behaviors in multi-agent collaboration. SentinelNet equips each agent with a credit-based detector trained via contrastive learning on augmented adversarial debate trajectories, enabling autonomous evaluation of message credibility and dynamic neighbor ranking via bottom-k elimination to suppress malicious communications. To overcome the scarcity of attack data, it generates adversarial trajectories simulating diverse threats, ensuring robust training. Experiments on MAS benchmarks show SentinelNet achieves near-perfect detection of malicious agents, close to 100% within two debate rounds, and recovers 95% of system accuracy from compromised baselines. By exhibiting strong generalizability across domains and attack patterns, SentinelNet establishes a novel paradigm for safeguarding collaborative MAS.

**arXiv ID:** 2510.16219
</details>

<details>
<summary><strong>Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection</strong> - Michelle Yuan, Khushbu Pahwa, Shuaichen Chang, Mustafa Kaba, Jiarong Jiang, Xiaofei Ma, Yi Zhang, Monica Sunkara - [[pdf]](https://arxiv.org/pdf/2510.16499)</summary>

**Abstract:** Designing effective agentic systems requires the seamless composition and integration of agents, tools, and models within dynamic and uncertain environments. Most existing methods rely on static, semantic retrieval approaches for tool or agent discovery. However, effective reuse and composition of existing components remain challenging due to incomplete capability descriptions and the limitations of retrieval methods. Component selection suffers because the decisions are not based on capability, cost, and real-time utility. To address these challenges, we introduce a structured, automated framework for agentic system composition that is inspired by the knapsack problem. Our framework enables a composer agent to systematically identify, select, and assemble an optimal set of agentic components by jointly considering performance, budget constraints, and compatibility. By dynamically testing candidate components and modeling their utility in real-time, our approach streamlines the assembly of agentic systems and facilitates scalable reuse of resources. Empirical evaluation with Claude 3.5 Sonnet across five benchmarking datasets shows that our online-knapsack-based composer consistently lies on the Pareto frontier, achieving higher success rates at significantly lower component costs compared to our baselines. In the single-agent setup, the online knapsack composer shows a success rate improvement of up to 31.6% in comparison to the retrieval baselines. In multi-agent systems, the online knapsack composer increases success rate from 37% to 87% when agents are selected from an agent inventory of 100+ agents. The substantial performance gap confirms the robust adaptability of our method across diverse domains and budget constraints.

**arXiv ID:** 2510.16499
</details>

<details>
<summary><strong>Prompt Optimization via Retrieved Reasoning Assets and Multi-Agent Analysis</strong> - Wonduk Seo, Juhyeon Lee, Junseo Koh, Hyunjin An, Jian Park, Seunghyun Lee, Haihua Chen, Yi Bu - [[pdf]](https://arxiv.org/pdf/2510.16635)</summary>

**Abstract:** Prompt optimization has emerged as an effective alternative to retraining for improving the performance of Large Language Models (LLMs). However, most existing approaches treat evaluation as a black box, relying solely on numerical scores while offering limited insight into why a prompt succeeds or fails. They also depend heavily on trial-and-error refinements, which are difficult to interpret and control. In this paper, we introduce MA-SAPO, a Multi-Agent framework for Score-Aware Prompt Optimization. Compared to prior methods, MA-SAPO explicitly couples evaluation outcomes with structured reasoning to guide systematic edits. The framework specifically consists of two stages: during the Reasoning Phase, agents collaboratively explain metric scores, diagnose weaknesses, and synthesize targeted refinements that are stored as reusable reasoning assets; during the Test Phase, agents retrieve these assets to analyze optimized prompts and apply only evidence-grounded edits. By turning evaluation signals into interpretable reasoning chains, MA-SAPO produces prompt refinements that are more transparent, auditable, and controllable. Experiments on the HelpSteer1/2 benchmarks demonstrate consistent improvements over single-pass prompting, retrieval-augmented baselines, and prior multi-agent strategies, validating the effectiveness of our approach.

**arXiv ID:** 2510.16635
</details>

<details>
<summary><strong>Unleashing Diverse Thinking Modes in LLMs through Multi-Agent Collaboration</strong> - Zhixuan He, Yue Feng - [[pdf]](https://arxiv.org/pdf/2510.16645)</summary>

**Abstract:** Large Language Models (LLMs) demonstrate strong performance but often lack interpretable reasoning. This paper introduces the Multi-Agent Collaboration Framework for Diverse Thinking Modes (DiMo), which enhances both performance and interpretability by simulating a structured debate among four specialized LLM agents. Each agent embodies a distinct reasoning paradigm, allowing the framework to collaboratively explore diverse cognitive approaches. Through iterative debate, agents challenge and refine initial responses, yielding more robust conclusions and an explicit, auditable reasoning chain. Across six benchmarks and under a unified open-source setup, DiMo improves accuracy over widely used single-model and debate baselines, with the largest gains on math. We position DiMo as a semantics-aware, Web-native multi-agent framework: it models human-machine intelligence with LLM agents that produce semantically typed, URL-annotated evidence chains for explanations and user-friendly interactions. Although our experiments use standard reasoning benchmarks, the framework is designed to be instantiated over Web corpora and knowledge graphs, combining retrieval-augmented reasoning with structured justifications that downstream systems can inspect and reuse.

**arXiv ID:** 2510.16645
</details>

<details>
<summary><strong>Heterogeneous Multi-Agent Task-Assignment with Uncertain Execution Times and Preferences</strong> - Qinshuang Wei, Vaibhav Srivastava, Vijay Gupta - [[pdf]](https://arxiv.org/pdf/2510.16221)</summary>

**Abstract:** While sequential task assignment for a single agent has been widely studied, such problems in a multi-agent setting, where the agents have heterogeneous task preferences or capabilities, remain less well-characterized. We study a multi-agent task assignment problem where a central planner assigns recurring tasks to multiple members of a team over a finite time horizon. For any given task, the members have heterogeneous capabilities in terms of task completion times, task resource consumption (which can model variables such as energy or attention), and preferences in terms of the rewards they collect upon task completion. We assume that the reward, execution time, and resource consumption for each member to complete any task are stochastic with unknown distributions. The goal of the planner is to maximize the total expected reward that the team receives over the problem horizon while ensuring that the resource consumption required for any assigned task is within the capability of the agent. We propose and analyze a bandit algorithm for this problem. Since the bandit algorithm relies on solving an optimal task assignment problem repeatedly, we analyze the achievable regret in two cases: when we can solve the optimal task assignment exactly and when we can solve it only approximately.

**arXiv ID:** 2510.16221
</details>

<details>
<summary><strong>Lark: Biologically Inspired Neuroevolution for Multi-Stakeholder LLM Agents</strong> - Dheeraj Chintapalli, Rikhil Tanugula, Sunkalp Chandra - [[pdf]](https://arxiv.org/pdf/2510.16978)</summary>

**Abstract:** We present Lark, a biologically inspired decision-making framework that couples LLM-driven reasoning with an evolutionary, stakeholder-aware Multi-Agent System (MAS). To address verbosity and stakeholder trade-offs, we integrate four mechanisms: (i) plasticity, which applies concise adjustments to candidate solutions; (ii) duplication and maturation, which copy high-performing candidates and specialize them into new modules; (iii) ranked-choice stakeholder aggregation using influence-weighted Borda scoring; and (iv) compute awareness via token-based penalties that reward brevity. The system iteratively proposes diverse strategies, applies plasticity tweaks, simulates stakeholder evaluations, aggregates preferences, selects top candidates, and performs duplication/maturation while factoring compute cost into final scores. In a controlled evaluation over 30 rounds comparing 14 systems, Lark Full achieves a mean rank of 2.55 (95% CI [2.17, 2.93]) and a mean composite score of 29.4/50 (95% CI [26.34, 32.46]), finishing Top-3 in 80% of rounds while remaining cost competitive with leading commercial models ($0.016 per task). Paired Wilcoxon tests confirm that all four mechanisms contribute significantly as ablating duplication/maturation yields the largest deficit ({\Delta}Score = 3.5, Cohen's d_z = 2.53, p < 0.001), followed by plasticity ({\Delta}Score = 3.4, d_z = 1.86), ranked-choice voting ({\Delta}Score = 2.4, d_z = 1.20), and token penalties ({\Delta}Score = 2.2, d_z = 1.63). Rather than a formal Markov Decision Process with constrained optimization, Lark is a practical, compute-aware neuroevolutionary loop that scales stakeholder-aligned strategy generation and makes trade-offs transparent through per-step metrics. Our work presents proof-of-concept findings and invites community feedback as we expand toward real-world validation studies.

**arXiv ID:** 2510.16978
</details>

<details>
<summary><strong>ReclAIm: A multi-agent framework for degradation-aware performance tuning of medical imaging AI</strong> - Eleftherios Tzanis, Michail E. Klontzas - [[pdf]](https://arxiv.org/pdf/2510.17004)</summary>

**Abstract:** Ensuring the long-term reliability of AI models in clinical practice requires continuous performance monitoring and corrective actions when degradation occurs. Addressing this need, this manuscript presents ReclAIm, a multi-agent framework capable of autonomously monitoring, evaluating, and fine-tuning medical image classification models. The system, built on a large language model core, operates entirely through natural language interaction, eliminating the need for programming expertise. ReclAIm successfully trains, evaluates, and maintains consistent performance of models across MRI, CT, and X-ray datasets. Once ReclAIm detects significant performance degradation, it autonomously executes state-of-the-art fine-tuning procedures that substantially reduce the performance gap. In cases with performance drops of up to -41.1% (MRI InceptionV3), ReclAIm managed to readjust performance metrics within 1.5% of the initial model results. ReclAIm enables automated, continuous maintenance of medical imaging AI models in a user-friendly and adaptable manner that facilitates broader adoption in both research and clinical environments.

**arXiv ID:** 2510.17004
</details>

<details>
<summary><strong>Verification-Aware Planning for Multi-Agent Systems</strong> - Tianyang Xu, Dan Zhang, Kushan Mitra, Estevam Hruschka - [[pdf]](https://arxiv.org/pdf/2510.17109)</summary>

**Abstract:** Large language model (LLM) agents are increasingly deployed to tackle complex tasks, often necessitating collaboration among multiple specialized agents. However, multi-agent collaboration introduces new challenges in planning, coordination, and verification. Execution failures frequently arise not from flawed reasoning alone, but from subtle misalignments in task interpretation, output format, or inter-agent handoffs. To address these challenges, we present VeriMAP, a framework for multi-agent collaboration with verification-aware planning. The VeriMAP planner decomposes tasks, models subtask dependencies, and encodes planner-defined passing criteria as subtask verification functions (VFs) in Python and natural language. We evaluate VeriMAP on diverse datasets, demonstrating that it outperforms both single- and multi-agent baselines while enhancing system robustness and interpretability. Our analysis highlights how verification-aware planning enables reliable coordination and iterative refinement in multi-agent systems, without relying on external labels or annotations.

**arXiv ID:** 2510.17109
</details>

<details>
<summary><strong>ATL*AS: An Automata-Theoretic Approach and Tool for the Verification of Strategic Abilities in Multi-Agent Systems</strong> - Sofia Garcia de Blas Garcia-Alcalde, Francesco Belardinelli - [[pdf]](https://arxiv.org/pdf/2510.17306)</summary>

**Abstract:** We present two novel symbolic algorithms for model checking the Alternating-time Temporal Logic ATL*, over both the infinite-trace and the finite-trace semantics. In particular, for infinite traces we design a novel symbolic reduction to parity games. We implement both methods in the ATL*AS model checker and evaluate it using synthetic benchmarks as well as a cybersecurity scenario. Our results demonstrate that the symbolic approach significantly outperforms the explicit-state representation and we find that our parity-game-based algorithm offers a more scalable and efficient solution for infinite-trace verification, outperforming previously available tools. Our results also confirm that finite-trace model checking yields substantial performance benefits over infinite-trace verification. As such, we provide a comprehensive toolset for verifying multiagent systems against specifications in ATL*.

**arXiv ID:** 2510.17306
</details>

<details>
<summary><strong>Sequence Modeling for N-Agent Ad Hoc Teamwork</strong> - Caroline Wang, Di Yang Shi, Elad Liebman, Ishan Durugkar, Arrasy Rahman, Peter Stone - [[pdf]](https://arxiv.org/pdf/2506.05527)</summary>

**Abstract:** N-agent ad hoc teamwork (NAHT) is a newly introduced challenge in multi-agent reinforcement learning, where controlled subteams of varying sizes must dynamically collaborate with varying numbers and types of unknown teammates without pre-coordination. The existing learning algorithm (POAM) considers only independent learning for its flexibility in dealing with a changing number of agents. However, independent learning fails to fully capture the inter-agent dynamics essential for effective collaboration. Based on our observation that transformers deal effectively with sequences with varying lengths and have been shown to be highly effective for a variety of machine learning problems, this work introduces a centralized, transformer-based method for N-agent ad hoc teamwork. Our proposed approach incorporates historical observations and actions of all controlled agents, enabling optimal responses to diverse and unseen teammates in partially observable environments. Empirical evaluation on a StarCraft II task demonstrates that MAT-NAHT outperforms POAM, achieving superior sample efficiency and generalization, without auxiliary agent-modeling objectives.

**arXiv ID:** 2506.05527
</details>

<details>
<summary><strong>COMPASS: Cooperative Multi-Agent Persistent Monitoring using Spatio-Temporal Attention Network</strong> - Xingjian Zhang, Yizhuo Wang, Guillaume Sartoretti - [[pdf]](https://arxiv.org/pdf/2507.16306)</summary>

**Abstract:** Persistent monitoring of dynamic targets is essential in real-world applications such as disaster response, environmental sensing, and wildlife conservation, where mobile agents must continuously gather information under uncertainty. We propose COMPASS, a multi-agent reinforcement learning (MARL) framework that enables decentralized agents to persistently monitor multiple moving targets efficiently. We model the environment as a graph, where nodes represent spatial locations and edges capture topological proximity, allowing agents to reason over structured layouts and revisit informative regions as needed. Each agent independently selects actions based on a shared spatio-temporal attention network that we design to integrate historical observations and spatial context. We model target dynamics using Gaussian Processes (GPs), which support principled belief updates and enable uncertainty-aware planning. We train COMPASS using centralized value estimation and decentralized policy execution under an adaptive reward setting. Our extensive experiments demonstrate that COMPASS consistently outperforms strong baselines in uncertainty reduction, target coverage, and coordination efficiency across dynamic multi-target scenarios.

**arXiv ID:** 2507.16306
</details>

<details>
<summary><strong>Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs</strong> - Om Tailor - [[pdf]](https://arxiv.org/pdf/2510.04303)</summary>

**Abstract:** Multi-agent deployments of large language models (LLMs) are increasingly embedded in market, allocation, and governance workflows, yet covert coordination among agents can silently erode trust and social welfare. Existing audits are dominated by heuristics that lack theoretical guarantees, struggle to transfer across tasks, and seldom ship with the infrastructure needed for independent replication. We introduce Audit the Whisper, a conference-grade research artifact that spans theory, benchmark design, detection, and reproducibility. Our contributions are: (i) a channel-capacity analysis showing how interventions such as paraphrase, rate limiting, and role permutation impose quantifiable capacity penalties-operationalised via paired-run Kullback--Leibler diagnostics-that tighten mutual-information thresholds with finite-sample guarantees and full proofs; (ii) ColludeBench-v0, covering pricing, first-price auctions, peer review, and hosted Gemini/Groq APIs with configurable covert schemes, deterministic manifests, and reward instrumentation; and (iii) a calibrated auditing pipeline that fuses cross-run mutual information, permutation invariance, watermark variance, and fairness-aware acceptance bias, each tuned to a $10^{-3}$ false-positive budget and validated by 10k honest runs plus an e-value martingale. Across ColludeBench and external suites including Secret Collusion, CASE, Perfect Collusion Benchmark, and SentinelAgent, the union meta-test attains state-of-the-art power at fixed FPR while ablations surface price-of-auditing trade-offs and fairness-driven colluders invisible to MI alone. We release regeneration scripts, anonymized manifests, and documentation so that external auditors can reproduce every figure, satisfy double-blind requirements, and extend the framework with minimal effort.

**arXiv ID:** 2510.04303
</details>

<details>
<summary><strong>Static Sandboxes Are Inadequate: Modeling Societal Complexity Requires Open-Ended Co-Evolution in LLM-Based Multi-Agent Simulations</strong> - Jinkun Chen, Sher Badshah, Xuemin Yu, Sijia Han, Jiechao Gao - [[pdf]](https://arxiv.org/pdf/2510.13982)</summary>

**Abstract:** What if artificial agents could not just communicate, but also evolve, adapt, and reshape their worlds in ways we cannot fully predict? With llm now powering multi-agent systems and social simulations, we are witnessing new possibilities for modeling open-ended, ever-changing environments. Yet, most current simulations remain constrained within static sandboxes, characterized by predefined tasks, limited dynamics, and rigid evaluation criteria. These limitations prevent them from capturing the complexity of real-world societies. In this paper, we argue that static, task-specific benchmarks are fundamentally inadequate and must be rethought. We critically review emerging architectures that blend llm with multi-agent dynamics, highlight key hurdles such as balancing stability and diversity, evaluating unexpected behaviors, and scaling to greater complexity, and introduce a fresh taxonomy for this rapidly evolving field. Finally, we present a research roadmap centered on open-endedness, continuous co-evolution, and the development of resilient, socially aligned AI ecosystems. We call on the community to move beyond static paradigms and help shape the next generation of adaptive, socially-aware multi-agent simulations.

**arXiv ID:** 2510.13982
</details>

<details>
<summary><strong>Agentic System with Modal Logic for Autonomous Diagnostics</strong> - Antonin Sulc, Thorsten Hellert - [[pdf]](https://arxiv.org/pdf/2509.11943)</summary>

**Abstract:** The development of intelligent agents, particularly those powered by language models (LMs), has shown a critical role in various environments that require intelligent and autonomous decision-making. Environments are not passive testing grounds, and they represent the data required for agents to learn and exhibit in very challenging conditions that require adaptive, complex, and autonomous capacity to make decisions. While the paradigm of scaling models and datasets has led to remarkable emergent capabilities, we argue that scaling the structure, fidelity, and logical consistency of agent reasoning within these environments is a crucial, yet underexplored, dimension of AI research. This paper introduces a neuro-symbolic multi-agent architecture where the belief states of individual agents are formally represented as Kripke models. This foundational choice enables them to reason about known concepts of \emph{possibility} and \emph{necessity} using the formal language of modal logic. In this work, we use immutable, domain-specific knowledge to make an informed root cause diagnosis, which is encoded as logical constraints essential for proper, reliable, and explainable diagnosis. In the proposed model, we show constraints that actively guide the hypothesis generation of LMs, effectively preventing them from reaching physically or logically untenable conclusions. In a high-fidelity simulated particle accelerator environment, our system successfully diagnoses complex, cascading failures by combining the powerful semantic intuition of LMs with the rigorous, verifiable validation of modal logic and a factual world model and showcasing a viable path toward more robust, reliable, and verifiable autonomous agents.

**arXiv ID:** 2509.11943
</details>

<details>
<summary><strong>Enterprise Deep Research: Steerable Multi-Agent Deep Research for Enterprise Analytics</strong> - Akshara Prabhakar, Roshan Ram, Zixiang Chen, Silvio Savarese, Frank Wang, Caiming Xiong, Huan Wang, Weiran Yao - [[pdf]](https://arxiv.org/pdf/2510.17797)</summary>

**Abstract:** As information grows exponentially, enterprises face increasing pressure to transform unstructured data into coherent, actionable insights. While autonomous agents show promise, they often struggle with domain-specific nuances, intent alignment, and enterprise integration. We present Enterprise Deep Research (EDR), a multi-agent system that integrates (1) a Master Planning Agent for adaptive query decomposition, (2) four specialized search agents (General, Academic, GitHub, LinkedIn), (3) an extensible MCP-based tool ecosystem supporting NL2SQL, file analysis, and enterprise workflows, (4) a Visualization Agent for data-driven insights, and (5) a reflection mechanism that detects knowledge gaps and updates research direction with optional human-in-the-loop steering guidance. These components enable automated report generation, real-time streaming, and seamless enterprise deployment, as validated on internal datasets. On open-ended benchmarks including DeepResearch Bench and DeepConsult, EDR outperforms state-of-the-art agentic systems without any human steering. We release the EDR framework and benchmark trajectories to advance research on multi-agent reasoning applications.
Code at this https URL and Dataset at this https URL

**arXiv ID:** 2510.17797
</details>

<details>
<summary><strong>Breaking and Fixing Defenses Against Control-Flow Hijacking in Multi-Agent Systems</strong> - Rishi Jha, Harold Triedman, Justin Wagle, Vitaly Shmatikov - [[pdf]](https://arxiv.org/pdf/2510.17276)</summary>

**Abstract:** Control-flow hijacking attacks manipulate orchestration mechanisms in multi-agent systems into performing unsafe actions that compromise the system and exfiltrate sensitive information. Recently proposed defenses, such as LlamaFirewall, rely on alignment checks of inter-agent communications to ensure that all agent invocations are "related to" and "likely to further" the original objective.
We start by demonstrating control-flow hijacking attacks that evade these defenses even if alignment checks are performed by advanced LLMs. We argue that the safety and functionality objectives of multi-agent systems fundamentally conflict with each other. This conflict is exacerbated by the brittle definitions of "alignment" and the checkers' incomplete visibility into the execution context.
We then propose, implement, and evaluate ControlValve, a new defense inspired by the principles of control-flow integrity and least privilege. ControlValve (1) generates permitted control-flow graphs for multi-agent systems, and (2) enforces that all executions comply with these graphs, along with contextual rules (generated in a zero-shot manner) for each agent invocation.

**arXiv ID:** 2510.17276
</details>

<details>
<summary><strong>Safe Multi-Agent Reinforcement Learning for Behavior-Based Cooperative Navigation</strong> - Murad Dawood, Sicong Pan, Nils Dengler, Siqi Zhou, Angela P. Schoellig, Maren Bennewitz - [[pdf]](https://arxiv.org/pdf/2312.12861)</summary>

**Abstract:** In this paper, we address the problem of behavior-based cooperative navigation of mobile robots using safe multi-agent reinforcement learning~(MARL). Our work is the first to focus on cooperative navigation without individual reference targets for the robots, using a single target for the formation's centroid. This eliminates the complexities involved in having several path planners to control a team of robots. To ensure safety, our MARL framework uses model predictive control (MPC) to prevent actions that could lead to collisions during training and execution. We demonstrate the effectiveness of our method in simulation and on real robots, achieving safe behavior-based cooperative navigation without using individual reference targets, with zero collisions, and faster target reaching compared to baselines. Finally, we study the impact of MPC safety filters on the learning process, revealing that we achieve faster convergence during training and we show that our approach can be safely deployed on real robots, even during early stages of the training.

**arXiv ID:** 2312.12861
</details>

<details>
<summary><strong>SketchMind: A Multi-Agent Cognitive Framework for Assessing Student-Drawn Scientific Sketches</strong> - Ehsan Latif, Zirak Khan, Xiaoming Zhai - [[pdf]](https://arxiv.org/pdf/2507.22904)</summary>

**Abstract:** Scientific sketches (e.g., models) offer a powerful lens into students' conceptual understanding, yet AI-powered automated assessment of such free-form, visually diverse artifacts remains a critical challenge. Existing solutions often treat sketch evaluation as either an image classification task or monolithic vision-language models, which lack interpretability, pedagogical alignment, and adaptability across cognitive levels. To address these limitations, we present SketchMind, a cognitively grounded, multi-agent framework for evaluating and improving student-drawn scientific sketches. SketchMind comprises modular agents responsible for rubric parsing, sketch perception, cognitive alignment, and iterative feedback with sketch modification, enabling personalized and transparent evaluation. We evaluate SketchMind on a curated dataset of 3,575 student-generated sketches across six science assessment items with different highest order of Bloom's level that require students to draw models to explain phenomena. Compared to baseline GPT-4o performance without SRG (average accuracy: 55.6%), and with SRG integration achieves 77.1% average accuracy (+21.4% average absolute gain). We also demonstrate that multi-agent orchestration with SRG enhances SketchMind performance, for example, GPT-4.1 gains an average 8.9% increase in sketch prediction accuracy, outperforming single-agent pipelines across all items. Human evaluators rated the feedback and co-created sketches generated by \textsc{SketchMind} with GPT-4.1, which achieved an average of 4.1 out of 5, significantly higher than those of baseline models (e.g., 2.3 for GPT-4o). Experts noted the system's potential to meaningfully support conceptual growth through guided revision. Our code and (pending approval) dataset will be released to support reproducibility and future research in AI-driven education.

**arXiv ID:** 2507.22904
</details>

</details>

<details open>
<summary><h2>Other Agent Research (16 papers)</h2></summary>

<details>
<summary><strong>RGMem: Renormalization Group-based Memory Evolution for Language Agent User Profile</strong> - Ao Tian, Yunfeng Lu, Xinxin Fan, Changhao Wang, Lanzhi Zhou, Yeyao Zhang, Yanfang Liu - [[pdf]](https://arxiv.org/pdf/2510.16392)</summary>

**Abstract:** Personalized and continuous interactions are the key to enhancing user experience in today's large language model (LLM)-based conversational systems, however, the finite context windows and static parametric memory make it difficult to model the cross-session long-term user states and behavioral consistency. Currently, the existing solutions to this predicament, such as retrieval-augmented generation (RAG) and explicit memory systems, primarily focus on fact-level storage and retrieval, lacking the capability to distill latent preferences and deep traits from the multi-turn dialogues, which limits the long-term and effective user modeling, directly leading to the personalized interactions remaining shallow, and hindering the cross-session continuity. To realize the long-term memory and behavioral consistency for Language Agents in LLM era, we propose a self-evolving memory framework RGMem, inspired by the ideology of classic renormalization group (RG) in physics, this framework enables to organize the dialogue history in multiple scales: it first extracts semantics and user insights from episodic fragments, then through hierarchical coarse-graining and rescaling operations, progressively forms a dynamically-evolved user profile. The core innovation of our work lies in modeling memory evolution as a multi-scale process of information compression and emergence, which accomplishes the high-level and accurate user profiles from noisy and microscopic-level interactions.

**arXiv ID:** 2510.16392
</details>

<details>
<summary><strong>DeepAnalyze: Agentic Large Language Models for Autonomous Data Science</strong> - Shaolei Zhang, Ju Fan, Meihao Fan, Guoliang Li, Xiaoyong Du - [[pdf]](https://arxiv.org/pdf/2510.16872)</summary>

**Abstract:** Autonomous data science, from raw data sources to analyst-grade deep research reports, has been a long-standing challenge, and is now becoming feasible with the emergence of powerful large language models (LLMs). Recent workflow-based data agents have shown promising results on specific data tasks but remain fundamentally limited in achieving fully autonomous data science due to their reliance on predefined workflows. In this paper, we introduce DeepAnalyze-8B, the first agentic LLM designed for autonomous data science, capable of automatically completing the end-toend pipeline from data sources to analyst-grade deep research reports. To tackle high-complexity data science tasks, we propose a curriculum-based agentic training paradigm that emulates the learning trajectory of human data scientists, enabling LLMs to progressively acquire and integrate multiple capabilities in real-world environments. We also introduce a data-grounded trajectory synthesis framework that constructs high-quality training data. Through agentic training, DeepAnalyze learns to perform a broad spectrum of data tasks, ranging from data question answering and specialized analytical tasks to open-ended data research. Experiments demonstrate that, with only 8B parameters, DeepAnalyze outperforms previous workflow-based agents built on most advanced proprietary LLMs. The model, code, and training data of DeepAnalyze are open-sourced, paving the way toward autonomous data science.

**arXiv ID:** 2510.16872
</details>

<details>
<summary><strong>Active Inference for an Intelligent Agent in Autonomous Reconnaissance Missions</strong> - Johan Schubert, Farzad Kamrani, Tove Gustavi - [[pdf]](https://arxiv.org/pdf/2510.17450)</summary>

**Abstract:** We develop an active inference route-planning method for the autonomous control of intelligent agents. The aim is to reconnoiter a geographical area to maintain a common operational picture. To achieve this, we construct an evidence map that reflects our current understanding of the situation, incorporating both positive and "negative" sensor observations of possible target objects collected over time, and diffusing the evidence across the map as time progresses. The generative model of active inference uses Dempster-Shafer theory and a Gaussian sensor model, which provides input to the agent. The generative process employs a Bayesian approach to update a posterior probability distribution. We calculate the variational free energy for all positions within the area by assessing the divergence between a pignistic probability distribution of the evidence map and a posterior probability distribution of a target object based on the observations, including the level of surprise associated with receiving new observations. Using the free energy, we direct the agents' movements in a simulation by taking an incremental step toward a position that minimizes the free energy. This approach addresses the challenge of exploration and exploitation, allowing agents to balance searching extensive areas of the geographical map while tracking identified target objects.

**arXiv ID:** 2510.17450
</details>

<details>
<summary><strong>MIRAGE: Agentic Framework for Multimodal Misinformation Detection with Web-Grounded Reasoning</strong> - Mir Nafis Sharear Shopnil, Sharad Duwal, Abhishek Tyagi, Adiba Mahbub Proma - [[pdf]](https://arxiv.org/pdf/2510.17590)</summary>

**Abstract:** Misinformation spreads across web platforms through billions of daily multimodal posts that combine text and images, overwhelming manual fact-checking capacity. Supervised detection models require domain-specific training data and fail to generalize across diverse manipulation tactics. We present MIRAGE, an inference-time, model-pluggable agentic framework that decomposes multimodal verification into four sequential modules: visual veracity assessment detects AI-generated images, cross-modal consistency analysis identifies out-of-context repurposing, retrieval-augmented factual checking grounds claims in web evidence through iterative question generation, and a calibrated judgment module integrates all signals. MIRAGE orchestrates vision-language model reasoning with targeted web retrieval, outputs structured and citation-linked rationales. On MMFakeBench validation set (1,000 samples), MIRAGE with GPT-4o-mini achieves 81.65% F1 and 75.1% accuracy, outperforming the strongest zero-shot baseline (GPT-4V with MMD-Agent at 74.0% F1) by 7.65 points while maintaining 34.3% false positive rate versus 97.3% for a judge-only baseline. Test set results (5,000 samples) confirm generalization with 81.44% F1 and 75.08% accuracy. Ablation studies show visual verification contributes 5.18 F1 points and retrieval-augmented reasoning contributes 2.97 points. Our results demonstrate that decomposed agentic reasoning with web retrieval can match supervised detector performance without domain-specific training, enabling misinformation detection across modalities where labeled data remains scarce.

**arXiv ID:** 2510.17590
</details>

<details>
<summary><strong>More with Less: An Empirical Study of Turn-Control Strategies for Efficient Coding Agents</strong> - Pengfei Gao, Chao Peng - [[pdf]](https://arxiv.org/pdf/2510.16786)</summary>

**Abstract:** LLM-powered coding agents, which operate in iterative loops (turns) to solve software engineering tasks, are becoming increasingly powerful. However, their practical deployment is hindered by significant and unpredictable costs. This challenge arises from a combination of factors: quadratically growing token counts with each turn, the high price of models, the large number of turns required for real-world tasks, and the tendency of agents to take inefficient or unnecessary actions. While existing research focuses on optimizing individual turns, the strategic control of the total number of turns remains an underexplored area for managing agent performance and cost. To address this gap, we conduct a comprehensive empirical study on SWE-bench using three state-of-the-art models and evaluate the impact of three distinct turn-control strategies: an unrestricted baseline, a fixed-turn limit with reminders, and a novel dynamic-turn strategy that grants extensions on-demand. Our findings first reveal a fundamental trade-off in the unrestricted setting, where no single model excels across performance, cost, and turn efficiency. We then show that a fixed-turn limit, specifically at the 75th percentile of the baseline, serves as a "sweet spot", substantially reducing costs (by 24%-68%) with minimal impact on solve rates. Most significantly, the dynamic-turn strategy consistently outperforms fixed-limit approaches, achieving comparable or better solve rates while further reducing costs by an additional 12%-24% by intelligently allocating resources only to tasks that need them. This work provides the first systematic analysis of turn-control strategies, offering simple yet effective guidelines for developers to balance cost and efficacy. We demonstrate that dynamic resource allocation is a superior, easy-to-implement approach for deploying powerful yet economically viable coding agents.

**arXiv ID:** 2510.16786
</details>

<details>
<summary><strong>Strategyproof Facility Location for Five Agents on a Circle using PCD</strong> - Ido Farjoun, Reshef Meir - [[pdf]](https://arxiv.org/pdf/2510.17435)</summary>

**Abstract:** We consider the strategyproof facility location problem on a circle. We focus on the case of 5 agents, and find a tight bound for the PCD strategyproof mechanism, which selects the reported location of an agent in proportion to the length of the arc in front of it. We methodically "reduce" the size of the instance space and then use standard optimization techniques to find and prove the bound is tight. Moreover we hypothesize the approximation ratio of PCD for general odd $n$.

**arXiv ID:** 2510.17435
</details>

<details>
<summary><strong>Asynchronous Agents with Perfect Recall: Model Reductions, Knowledge-Based Construction, and Model Checking for Coalitional Strategies</strong> - Dilian Gurov, Filip Jamroga, Wojciech Jamroga, Mateusz Kamiński, Damian Kurpiewski, Wojciech Penczek, Teofil Sidoruk - [[pdf]](https://arxiv.org/pdf/2412.06706)</summary>

**Abstract:** Model checking of strategic abilities for agents with memory is a notoriously hard problem, and very few attempts have been made to tackle it. In this paper, we present two important steps towards this goal. First, we take the partial-order reduction scheme that was recently proved to preserve individual and coalitional abilities of memoryless agents, and show that it also works for agents with memory. Secondly, we take the Knowledge-Based Subset Construction, that was recently studied for synchronous concurrent games, and adapt it to preserve abilities of memoryful agents in asynchronous MAS. On the way, we also propose a new execution semantics for strategies in asynchronous MAS, that combines elements of Concurrent Game Structures and Interleaved Interpreted Systems in a natural and intuitive way.

**arXiv ID:** 2412.06706
</details>

<details>
<summary><strong>First Field-Trial Demonstration of L4 Autonomous Optical Network for Distributed AI Training Communication: An LLM-Powered Multi-AI-Agent Solution</strong> - Yihao Zhang, Qizhi Qiu, Xiaomin Liu, Dianxuan Fu, Xingyu Liu, Leyan Fei, Yuming Cheng, Lilin Yi, Weisheng Hu, Qunbi Zhuge - [[pdf]](https://arxiv.org/pdf/2504.01234)</summary>

**Abstract:** We demonstrate the first cross-domain cross-layer level-4 autonomous optical network via a multi-AI-agent system. Field trials show ~98% task completion rate across the distributed AI training lifecycle-3.2x higher than single agents using state-of-the-art LLMs.

**arXiv ID:** 2504.01234
</details>

<details>
<summary><strong>ReaGAN: Node-as-Agent-Reasoning Graph Agentic Network</strong> - Minghao Guo, Xi Zhu, Haochen Xue, Chong Zhang, Shuhang Lin, Jingyuan Huang, Ziyi Ye, Yongfeng Zhang - [[pdf]](https://arxiv.org/pdf/2508.00429)</summary>

**Abstract:** Graph Neural Networks (GNNs) have achieved remarkable success in graph-based learning by propagating information among neighbor nodes via predefined aggregation mechanisms. However, such fixed schemes often suffer from two key limitations. First, they cannot handle the imbalance in node informativeness -- some nodes are rich in information, while others remain sparse. Second, predefined message passing primarily leverages local structural similarity while ignoring global semantic relationships across the graph, limiting the model's ability to capture distant but relevant information. We propose Retrieval-augmented Graph Agentic Network (ReaGAN), an agent-based framework that empowers each node with autonomous, node-level decision-making. Each node acts as an agent that independently plans its next action based on its internal memory, enabling node-level planning and adaptive message propagation. Additionally, retrieval-augmented generation (RAG) allows nodes to access semantically relevant content and build global relationships in the graph. ReaGAN achieves competitive performance under few-shot in-context settings using a frozen LLM backbone without fine-tuning, showcasing the potential of agentic planning and local-global retrieval in graph learning.

**arXiv ID:** 2508.00429
</details>

<details>
<summary><strong>EEschematic: Multimodal-LLM Based AI Agent for Schematic Generation of Analog Circuit</strong> - Chang Liu, Danial Chitnis - [[pdf]](https://arxiv.org/pdf/2510.17002)</summary>

**Abstract:** Circuit schematics play a crucial role in analog integrated circuit design, serving as the primary medium for human understanding and verification of circuit functionality. While recent large language model (LLM)-based approaches have shown promise in circuit topology generation and device sizing, most rely solely on textual representations such as SPICE netlists, which lack visual interpretability for circuit designers. To address this limitation, we propose EEschematic, an AI agent for automatic analog schematic generation based on a Multimodal Large Language Model (MLLM). EEschematic integrates textual, visual, and symbolic modalities to translate SPICE netlists into schematic diagrams represented in a human-editable format. The framework uses six analog substructure examples for few-shot placement and a Visual Chain-of-Thought (VCoT) strategy to iteratively refine placement and wiring, enhancing schematic clarity and symmetry. Experimental results on representative analog circuits, including a CMOS inverter, a five-transistor operational transconductance amplifier (5T-OTA), and a telescopic cascode amplifier, demonstrate that EEschematic produces schematics with high visual quality and structural correctness.

**arXiv ID:** 2510.17002
</details>

<details>
<summary><strong>The Invisible Handshake: Tacit Collusion between Adaptive Market Agents</strong> - Luigi Foscari, Emanuele Guidotti, Nicolò Cesa-Bianchi, Tatjana Chavdarova, Alfio Ferrara - [[pdf]](https://arxiv.org/pdf/2510.15995)</summary>

**Abstract:** We study the emergence of tacit collusion between adaptive trading agents in a stochastic market with endogenous price formation. Using a two-player repeated game between a market maker and a market taker, we characterize feasible and collusive strategy profiles that raise prices beyond competitive levels. We show that, when agents follow simple learning algorithms (e.g., gradient ascent) to maximize their own wealth, the resulting dynamics converge to collusive strategy profiles, even in highly liquid markets with small trade sizes. By highlighting how simple learning strategies naturally lead to tacit collusion, our results offer new insights into the dynamics of AI-driven markets.

**arXiv ID:** 2510.15995
</details>

<details>
<summary><strong>Efficient Vision-Language-Action Models for Embodied Manipulation: A Systematic Survey</strong> - Weifan Guan, Qinghao Hu, Aosheng Li, Jian Cheng - [[pdf]](https://arxiv.org/pdf/2510.17111)</summary>

**Abstract:** Vision-Language-Action (VLA) models extend vision-language models to embodied control by mapping natural-language instructions and visual observations to robot actions. Despite their capabilities, VLA systems face significant challenges due to their massive computational and memory demands, which conflict with the constraints of edge platforms such as on-board mobile manipulators that require real-time performance. Addressing this tension has become a central focus of recent research. In light of the growing efforts toward more efficient and scalable VLA systems, this survey provides a systematic review of approaches for improving VLA efficiency, with an emphasis on reducing latency, memory footprint, and training and inference costs. We categorize existing solutions into four dimensions: model architecture, perception feature, action generation, and training/inference strategies, summarizing representative techniques within each category. Finally, we discuss future trends and open challenges, highlighting directions for advancing efficient embodied intelligence.

**arXiv ID:** 2510.17111
</details>

<details>
<summary><strong>SimpleVSF: VLM-Scoring Fusion for Trajectory Prediction of End-to-End Autonomous Driving</strong> - Peiru Zheng, Yun Zhao, Zhan Gong, Hong Zhu, Shaohua Wu - [[pdf]](https://arxiv.org/pdf/2510.17191)</summary>

**Abstract:** End-to-end autonomous driving has emerged as a promising paradigm for achieving robust and intelligent driving policies. However, existing end-to-end methods still face significant challenges, such as suboptimal decision-making in complex scenarios. In this paper,we propose SimpleVSF (Simple VLM-Scoring Fusion), a novel framework that enhances end-to-end planning by leveraging the cognitive capabilities of Vision-Language Models (VLMs) and advanced trajectory fusion techniques. We utilize the conventional scorers and the novel VLM-enhanced scorers. And we leverage a robust weight fusioner for quantitative aggregation and a powerful VLM-based fusioner for qualitative, context-aware decision-making. As the leading approach in the ICCV 2025 NAVSIM v2 End-to-End Driving Challenge, our SimpleVSF framework demonstrates state-of-the-art performance, achieving a superior balance between safety, comfort, and efficiency.

**arXiv ID:** 2510.17191
</details>

<details>
<summary><strong>Decentralized Real-Time Planning for Multi-UAV Cooperative Manipulation via Imitation Learning</strong> - Shantnav Agarwal, Javier Alonso-Mora, Sihao Sun - [[pdf]](https://arxiv.org/pdf/2510.17143)</summary>

**Abstract:** Existing approaches for transporting and manipulating cable-suspended loads using multiple UAVs along reference trajectories typically rely on either centralized control architectures or reliable inter-agent communication. In this work, we propose a novel machine learning based method for decentralized kinodynamic planning that operates effectively under partial observability and without inter-agent communication. Our method leverages imitation learning to train a decentralized student policy for each UAV by imitating a centralized kinodynamic motion planner with access to privileged global observations. The student policy generates smooth trajectories using physics-informed neural networks that respect the derivative relationships in motion. During training, the student policies utilize the full trajectory generated by the teacher policy, leading to improved sample efficiency. Moreover, each student policy can be trained in under two hours on a standard laptop. We validate our method in both simulation and real-world environments to follow an agile reference trajectory, demonstrating performance comparable to that of centralized approaches.

**arXiv ID:** 2510.17143
</details>

<details>
<summary><strong>EDEN: Efficient Dual-Layer Exploration Planning for Fast UAV Autonomous Exploration in Large 3-D Environments</strong> - Qianli Dong, Xuebo Zhang, Shiyong Zhang, Ziyu Wang, Zhe Ma, Haobo Xi - [[pdf]](https://arxiv.org/pdf/2506.05106)</summary>

**Abstract:** Efficient autonomous exploration in large-scale environments remains challenging due to the high planning computational cost and low-speed maneuvers. In this paper, we propose a fast and computationally efficient dual-layer exploration planning method. The insight of our dual-layer method is efficiently finding an acceptable long-term region routing and greedily exploring the target in the region of the first routing area with high speed. Specifically, the proposed method finds the long-term area routing through an approximate algorithm to ensure real-time planning in large-scale environments. Then, the viewpoint in the first routing region with the lowest curvature-penalized cost, which can effectively reduce decelerations caused by sharp turn motions, will be chosen as the next exploration target. To further speed up the exploration, we adopt an aggressive and safe exploration-oriented trajectory to enhance exploration continuity. The proposed method is compared to state-of-the-art methods in challenging simulation environments. The results show that the proposed method outperforms other methods in terms of exploration efficiency, computational cost, and trajectory speed. We also conduct real-world experiments to validate the effectiveness of the proposed method. The code will be open-sourced.

**arXiv ID:** 2506.05106
</details>

<details>
<summary><strong>General agents contain world models</strong> - Jonathan Richens, David Abel, Alexis Bellot, Tom Everitt - [[pdf]](https://arxiv.org/pdf/2506.01622)</summary>

**Abstract:** Are world models a necessary ingredient for flexible, goal-directed behaviour, or is model-free learning sufficient? We provide a formal answer to this question, showing that any agent capable of generalizing to multi-step goal-directed tasks must have learned a predictive model of its environment. We show that this model can be extracted from the agent's policy, and that increasing the agents performance or the complexity of the goals it can achieve requires learning increasingly accurate world models. This has a number of consequences: from developing safe and general agents, to bounding agent capabilities in complex environments, and providing new algorithms for eliciting world models from agents.

**arXiv ID:** 2506.01622
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>BREATH: A Bio-Radar Embodied Agent for Tonal and Human-Aware Diffusion Music Generation</strong> - Yunzhe Wang, Xinyu Tang, Zhixun Huang, Xiaolong Yue, Yuxin Zeng - [[pdf]](https://arxiv.org/pdf/2510.15895)</summary>

**Abstract:** We present a multimodal system for personalized music generation that integrates physiological sensing, LLM-based reasoning, and controllable audio synthesis. A millimeter-wave radar sensor non-invasively captures heart rate and respiration rate. These physiological signals, combined with environmental state, are interpreted by a reasoning agent to infer symbolic musical descriptors, such as tempo, mood intensity, and traditional Chinese pentatonic modes, which are then expressed as structured prompts to guide a diffusion-based audio model in synthesizing expressive melodies. The system emphasizes cultural grounding through tonal embeddings and enables adaptive, embodied music interaction. To evaluate the system, we adopt a research-creation methodology combining case studies, expert feedback, and targeted control experiments. Results show that physiological variations can modulate musical features in meaningful ways, and tonal conditioning enhances alignment with intended modal characteristics. Expert users reported that the system affords intuitive, culturally resonant musical responses and highlighted its potential for therapeutic and interactive applications. This work demonstrates a novel bio-musical feedback loop linking radar-based sensing, prompt reasoning, and generative audio modeling.

**arXiv ID:** 2510.15895
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (33 papers)</h2></summary>

<details>
<summary><strong>A Comprehensive Survey on Reinforcement Learning-based Agentic Search: Foundations, Roles, Optimizations, Evaluations, and Applications</strong> - Minhua Lin, Zongyu Wu, Zhichao Xu, Hui Liu, Xianfeng Tang, Qi He, Charu Aggarwal, Hui Liu, Xiang Zhang, Suhang Wang - [[pdf]](https://arxiv.org/pdf/2510.16724)</summary>

**Abstract:** The advent of large language models (LLMs) has transformed information access and reasoning through open-ended natural language interaction. However, LLMs remain limited by static knowledge, factual hallucinations, and the inability to retrieve real-time or domain-specific information. Retrieval-Augmented Generation (RAG) mitigates these issues by grounding model outputs in external evidence, but traditional RAG pipelines are often single turn and heuristic, lacking adaptive control over retrieval and reasoning. Recent advances in agentic search address these limitations by enabling LLMs to plan, retrieve, and reflect through multi-step interaction with search environments. Within this paradigm, reinforcement learning (RL) offers a powerful mechanism for adaptive and self-improving search behavior. This survey provides the first comprehensive overview of \emph{RL-based agentic search}, organizing the emerging field along three complementary dimensions: (i) What RL is for (functional roles), (ii) How RL is used (optimization strategies), and (iii) Where RL is applied (scope of optimization). We summarize representative methods, evaluation protocols, and applications, and discuss open challenges and future directions toward building reliable and scalable RL driven agentic search systems. We hope this survey will inspire future research on the integration of RL and agentic search. Our repository is available at this https URL.

**arXiv ID:** 2510.16724
</details>

<details>
<summary><strong>VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents</strong> - Kangrui Wang, Pingyue Zhang, Zihan Wang, Yaning Gao, Linjie Li, Qineng Wang, Hanyang Chen, Chi Wan, Yiping Lu, Zhengyuan Yang, Lijuan Wang, Ranjay Krishna, Jiajun Wu, Li Fei-Fei, Yejin Choi, Manling Li - [[pdf]](https://arxiv.org/pdf/2510.16907)</summary>

**Abstract:** A key challenge in training Vision-Language Model (VLM) agents, compared to Language Model (LLM) agents, lies in the shift from textual states to complex visual observations. This transition introduces partial observability and demands robust world modeling. We ask: Can VLM agents construct internal world models through explicit visual state reasoning? To address this question, we architecturally enforce and reward the agent's reasoning process via reinforcement learning (RL), formulating it as a Partially Observable Markov Decision Process (POMDP). We find that decomposing the agent's reasoning into State Estimation ("what is the current state?") and Transition Modeling ("what comes next?") is critical for success, as demonstrated through five reasoning strategies. Our investigation into how agents represent internal beliefs reveals that the optimal representation is task-dependent: Natural Language excels at capturing semantic relationships in general tasks, while Structured formats are indispensable for precise manipulation and control. Building on these insights, we design a World Modeling Reward that provides dense, turn-level supervision for accurate state prediction, and introduce Bi-Level General Advantage Estimation (Bi-Level GAE) for turn-aware credit assignment. Through this form of visual state reasoning, a 3B-parameter model achieves a score of 0.82 across five diverse agent benchmarks, representing a 3$\times$ improvement over its untrained counterpart (0.21) and outperforming proprietary reasoning models such as GPT-5 (0.75), Gemini 2.5 Pro (0.67) and Claude 4.5 (0.62). All experiments are conducted within our VAGEN framework, a scalable system for training and analyzing multi-turn VLM agents in diverse visual environments. Code and data are publicly available at this https URL.

**arXiv ID:** 2510.16907
</details>

<details>
<summary><strong>FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive Stochastic Control in Finance</strong> - Yang Li, Zhi Chen - [[pdf]](https://arxiv.org/pdf/2510.15883)</summary>

**Abstract:** Traditional stochastic control methods in finance struggle in real world markets due to their reliance on simplifying assumptions and stylized frameworks. Such methods typically perform well in specific, well defined environments but yield suboptimal results in changed, non stationary ones. We introduce FinFlowRL, a novel framework for financial optimal stochastic control. The framework pretrains an adaptive meta policy learning from multiple expert strategies, then finetunes through reinforcement learning in the noise space to optimize the generative process. By employing action chunking generating action sequences rather than single decisions, it addresses the non Markovian nature of markets. FinFlowRL consistently outperforms individually optimized experts across diverse market conditions.

**arXiv ID:** 2510.15883
</details>

<details>
<summary><strong>ESCA: Contextualizing Embodied Agents via Scene-Graph Generation</strong> - Jiani Huang, Amish Sethi, Matthew Kuo, Mayank Keoliya, Neelay Velingker, JungHo Jung, Ser-Nam Lim, Ziyang Li, Mayur Naik - [[pdf]](https://arxiv.org/pdf/2510.15963)</summary>

**Abstract:** Multi-modal large language models (MLLMs) are making rapid progress toward general-purpose embodied agents. However, current training pipelines primarily rely on high-level vision-sound-text pairs and lack fine-grained, structured alignment between pixel-level visual content and textual semantics. To overcome this challenge, we propose ESCA, a new framework for contextualizing embodied agents through structured spatial-temporal understanding. At its core is SGClip, a novel CLIP-based, open-domain, and promptable model for generating scene graphs. SGClip is trained on 87K+ open-domain videos via a neurosymbolic learning pipeline, which harnesses model-driven self-supervision from video-caption pairs and structured reasoning, thereby eliminating the need for human-labeled scene graph annotations. We demonstrate that SGClip supports both prompt-based inference and task-specific fine-tuning, excelling in scene graph generation and action localization benchmarks. ESCA with SGClip consistently improves both open-source and commercial MLLMs, achieving state-of-the-art performance across two embodied environments. Notably, it significantly reduces agent perception errors and enables open-source models to surpass proprietary baselines.

**arXiv ID:** 2510.15963
</details>

<details>
<summary><strong>Cog-Rethinker: Hierarchical Metacognitive Reinforcement Learning for LLM Reasoning</strong> - Zexu Sun, Yongcheng Zeng, Erxue Min, Heyang Gao, Bokai Ji, Xu Chen - [[pdf]](https://arxiv.org/pdf/2510.15979)</summary>

**Abstract:** Contemporary progress in large language models (LLMs) has revealed notable inferential capacities via reinforcement learning (RL) employing verifiable reward, facilitating the development of O1 and R1-like reasoning models. Directly training from base models with RL is called zero-RL. However, previous works rely upon activating LLMs' inherent capacities through fixed prompt templates. This strategy introduces substantial sampling inefficiencies for weak LLMs, as the majority of problems generate invalid outputs during accuracy-driven filtration in reasoning tasks, which causes a waste of samples. To solve this issue, we propose Cog-Rethinker, a novel hierarchical metacognitive RL framework for LLM reasoning. Our Cog-Rethinker mainly focuses on the rollout procedure in RL training. After the direct rollout, our Cog-Rethinker improves sample utilization in a hierarchical metacognitive two-stage framework. By leveraging human cognition during solving problems, firstly, it prompts policy to decompose zero-accuracy problems into subproblems to produce final reasoning results. Secondly, with zero-accuracy problems in previous rollout stage, it further prompts policy to refine these answers by referencing previous wrong solutions. Moreover, to enable cold-start of the two new reasoning patterns and maintain train-test consistency across prompt templates, our Cog-Rethinker applies supervised fine-tuning on the policy using correct samples of the two stages with direct rollout template. Experimental results demonstrate Cog-Rethinker's superior performance on various mathematical reasoning benchmarks, we also analyzed its improved sample efficiency that accelerates convergence compared to baseline methods.

**arXiv ID:** 2510.15979
</details>

<details>
<summary><strong>Interpretable RNA-Seq Clustering with an LLM-Based Agentic Evidence-Grounded Framework</strong> - Elias Hossain, Mehrdad Shoeibi, Ivan Garibay, Niloofar Yousefi - [[pdf]](https://arxiv.org/pdf/2510.16082)</summary>

**Abstract:** We propose CITE V.1, an agentic, evidence-grounded framework that leverages Large Language Models (LLMs) to provide transparent and reproducible interpretations of RNA-seq clusters. Unlike existing enrichment-based approaches that reduce results to broad statistical associations and LLM-only models that risk unsupported claims or fabricated citations, CITE V.1 transforms cluster interpretation by producing biologically coherent explanations explicitly anchored in the biomedical literature. The framework orchestrates three specialized agents: a Retriever that gathers domain knowledge from PubMed and UniProt, an Interpreter that formulates functional hypotheses, and Critics that evaluate claims, enforce evidence grounding, and qualify uncertainty through confidence and reliability indicators. Applied to Salmonella enterica RNA-seq data, CITE V.1 generated biologically meaningful insights supported by the literature, while an LLM-only Gemini baseline frequently produced speculative results with false citations. By moving RNA-seq analysis from surface-level enrichment to auditable, interpretable, and evidence-based hypothesis generation, CITE V.1 advances the transparency and reliability of AI in biomedicine.

**arXiv ID:** 2510.16082
</details>

<details>
<summary><strong>The Formalism-Implementation Gap in Reinforcement Learning Research</strong> - Pablo Samuel Castro - [[pdf]](https://arxiv.org/pdf/2510.16175)</summary>

**Abstract:** The last decade has seen an upswing in interest and adoption of reinforcement learning (RL) techniques, in large part due to its demonstrated capabilities at performing certain tasks at "super-human levels". This has incentivized the community to prioritize research that demonstrates RL agent performance, often at the expense of research aimed at understanding their learning dynamics. Performance-focused research runs the risk of overfitting on academic benchmarks -- thereby rendering them less useful -- which can make it difficult to transfer proposed techniques to novel problems. Further, it implicitly diminishes work that does not push the performance-frontier, but aims at improving our understanding of these techniques. This paper argues two points: (i) RL research should stop focusing solely on demonstrating agent capabilities, and focus more on advancing the science and understanding of reinforcement learning; and (ii) we need to be more precise on how our benchmarks map to the underlying mathematical formalisms. We use the popular Arcade Learning Environment (ALE; Bellemare et al., 2013) as an example of a benchmark that, despite being increasingly considered "saturated", can be effectively used for developing this understanding, and facilitating the deployment of RL techniques in impactful real-world problems.

**arXiv ID:** 2510.16175
</details>

<details>
<summary><strong>NEBULA: Do We Evaluate Vision-Language-Action Agents Correctly?</strong> - Jierui Peng, Yanyan Zhang, Yicheng Duan, Tuo Liang, Vipin Chaudhary, Yu Yin - [[pdf]](https://arxiv.org/pdf/2510.16263)</summary>

**Abstract:** The evaluation of Vision-Language-Action (VLA) agents is hindered by the coarse, end-task success metric that fails to provide precise skill diagnosis or measure robustness to real-world perturbations. This challenge is exacerbated by a fragmented data landscape that impedes reproducible research and the development of generalist models. To address these limitations, we introduce \textbf{NEBULA}, a unified ecosystem for single-arm manipulation that enables diagnostic and reproducible evaluation. NEBULA features a novel dual-axis evaluation protocol that combines fine-grained \textit{capability tests} for precise skill diagnosis with systematic \textit{stress tests} that measure robustness. A standardized API and a large-scale, aggregated dataset are provided to reduce fragmentation and support cross-dataset training and fair comparison. Using NEBULA, we demonstrate that top-performing VLAs struggle with key capabilities such as spatial reasoning and dynamic adaptation, which are consistently obscured by conventional end-task success metrics. By measuring both what an agent can do and when it does so reliably, NEBULA provides a practical foundation for robust, general-purpose embodied agents.

**arXiv ID:** 2510.16263
</details>

<details>
<summary><strong>LANPO: Bootstrapping Language and Numerical Feedback for Reinforcement Learning in LLMs</strong> - Ang Li, Yifei Wang, Zhihang Yuan, Stefanie Jegelka, Yisen Wang - [[pdf]](https://arxiv.org/pdf/2510.16552)</summary>

**Abstract:** Reinforcement learning in large language models (LLMs) often relies on scalar rewards, a practice that discards valuable textual rationale buried in the rollouts, forcing the model to explore \textit{de novo} with each attempt and hindering sample efficiency. While LLMs can uniquely learn from language feedback provided in-context, naively integrating on-line experiences into RL training presents a paradox: feedback from the same problem risks information leakage and memorization, while feedback from different problems often leads to behavior collapse due to irrelevant context. To resolve this tension, we propose \textbf{Language-And-Numerical Policy Optimization (LANPO)}, a framework that cleanly separates the roles of feedback: language guides exploration, while numerical rewards drive optimization. LANPO builds a dynamic experience pool from past trials and introduces two principles to ensure feedback is effective: \emph{Reward-Agnostic Reflection} for safe intra-sample self-correction and \emph{Relevant Abstraction} to distill generalizable lessons from inter-sample experiences. Across mathematical reasoning benchmarks, LANPO enables 7B and 14B models to significantly outperform strong baselines trained with GRPO in test accuracy. Our work provides a robust method for integrating historical experiences into the LLM RL loop, creating more effective and data-efficient learning agents.

**arXiv ID:** 2510.16552
</details>

<details>
<summary><strong>Enhancing Language Agent Strategic Reasoning through Self-Play in Adversarial Games</strong> - Yikai Zhang, Ye Rong, Siyu Yuan, Jiangjie Chen, Jian Xie, Yanghua Xiao - [[pdf]](https://arxiv.org/pdf/2510.16761)</summary>

**Abstract:** Existing language agents often encounter difficulties in dynamic adversarial games due to poor strategic reasoning. To mitigate this limitation, a promising approach is to allow agents to learn from game interactions automatically, without relying on costly expert-labeled data. Unlike static environments where agents receive fixed feedback or rewards, selecting appropriate opponents in dynamic adversarial games can significantly impact learning performance. However, the discussion of opponents in adversarial environments remains an area under exploration. In this paper, we propose a Step-level poliCy Optimization method through Play-And-Learn, SCO-PAL. Leveraging SCO-PAL, we conduct a detailed analysis of opponent selection by setting opponents at different levels and find that self-play is the most effective way to improve strategic reasoning in such adversarial environments. Utilizing SCO-PAL with self-play, we increase the average win rate against four opponents by approximately 30% compared to baselines and achieve a 54.76% win rate against GPT-4 in six adversarial games.

**arXiv ID:** 2510.16761
</details>

<details>
<summary><strong>SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents</strong> - Qiusi Zhan, Angeline Budiman-Chan, Abdelrahman Zayed, Xingzhi Guo, Daniel Kang, Joo-Kyung Kim - [[pdf]](https://arxiv.org/pdf/2510.17017)</summary>

**Abstract:** Large language model (LLM) based search agents iteratively generate queries, retrieve external information, and reason to answer open-domain questions. While researchers have primarily focused on improving their utility, their safety behaviors remain underexplored. In this paper, we first evaluate search agents using red-teaming datasets and find that they are more likely to produce harmful outputs than base LLMs. For example, when asked ``How can I track someone's location without their consent?'', a base model refuses, whereas a search agent designed to retrieve and cite sources may lower its refusal threshold, fetch documents (e.g., court cases), and, once appended, synthesize them into an informative yet unsafe summary. We further show that utility-oriented fine-tuning intensifies this risk, motivating joint alignment of safety and utility. We present SafeSearch, a multi-objective reinforcement learning approach that couples a final-output safety/utility reward with a novel query-level shaping term that penalizes unsafe queries and rewards safe ones. Experiments show that SafeSearch reduces agent harmfulness by over 70% across three red-teaming datasets while producing safe, helpful responses, and matches the QA performance of a utility-only finetuned agent; further analyses confirm the effectiveness of the query-level reward in jointly improving safety and utility.

**arXiv ID:** 2510.17017
</details>

<details>
<summary><strong>Agentic Reinforcement Learning for Search is Unsafe</strong> - Yushi Yang, Shreyansh Padarha, Andrew Lee, Adam Mahdi - [[pdf]](https://arxiv.org/pdf/2510.17431)</summary>

**Abstract:** Agentic reinforcement learning (RL) trains large language models to autonomously call tools during reasoning, with search as the most common application. These models excel at multi-step reasoning tasks, but their safety properties are not well understood. In this study, we show that RL-trained search models inherit refusal from instruction tuning and often deflect harmful requests by turning them into safe queries. However, this safety is fragile. Two simple attacks, one that forces the model to begin response with search (Search attack), another that encourages models to repeatedly search (Multi-search attack), trigger cascades of harmful searches and answers. Across two model families (Qwen, Llama) with both local and web search, these attacks lower refusal rates by up to 60.0%, answer safety by 82.5%, and search-query safety by 82.4%. The attacks succeed by triggering models to generate harmful, request-mirroring search queries before they can generate the inherited refusal tokens. This exposes a core weakness of current RL training: it rewards continued generation of effective queries without accounting for their harmfulness. As a result, RL search models have vulnerabilities that users can easily exploit, making it urgent to develop safety-aware agentic RL pipelines optimising for safe search.

**arXiv ID:** 2510.17431
</details>

<details>
<summary><strong>PrivacyPAD: A Reinforcement Learning Framework for Dynamic Privacy-Aware Delegation</strong> - Zheng Hui, Yijiang River Dong, Sanhanat Sivapiromrat, Ehsan Shareghi, Nigel Collier - [[pdf]](https://arxiv.org/pdf/2510.16054)</summary>

**Abstract:** When users submit queries to Large Language Models (LLMs), their prompts can often contain sensitive data, forcing a difficult choice: Send the query to a powerful proprietary LLM providers to achieving state-of-the-art performance and risk data exposure, or relying on smaller, local models guarantees data privacy but often results in a degradation of task performance. Prior approaches have relied on static pipelines that use LLM rewriting, which shatters linguistic coherence and indiscriminately removes privacy-sensitive information, including task-critical content. We reformulate this challenge (Privacy-Conscious Delegation) as a sequential decision-making problem and introduce a novel reinforcement learning (RL) framework called PrivacyPAD to solve it. Our framework trains an agent to dynamically route text chunks, learning a policy that optimally balances the trade-off between privacy leakage and task performance. It implicitly distinguishes between replaceable Personally Identifiable Information (PII) (which it shields locally) and task-critical PII (which it strategically sends to the remote model for maximal utility). To validate our approach in complex scenarios, we also introduce a new medical dataset with high PII density. Our framework achieves a new state-of-the-art on the privacy-utility frontier, demonstrating the necessity of learned, adaptive policies for deploying LLMs in sensitive environments.

**arXiv ID:** 2510.16054
</details>

<details>
<summary><strong>WEBSERV: A Browser-Server Environment for Efficient Training of Reinforcement Learning-based Web Agents at Scale</strong> - Yuxuan Lu, Jing Huang, Hui Liu, Jiri Gesi, Yan Han, Shihan Fu, Tianqi Zheng, Dakuo Wang - [[pdf]](https://arxiv.org/pdf/2510.16252)</summary>

**Abstract:** Training and evaluation of Reinforcement Learning (RL) web agents have gained increasing attention, yet a scalable and efficient environment that couples realistic and robust browser-side interaction with controllable server-side state at scale is still missing. Existing environments tend to have one or more of the following issues: they overwhelm policy models with excessive and noisy context; they perform actions non-deterministically without waiting for the UI or network to stabilize; or they cannot scale isolated client-server containers effectively for parallel RL rollouts. We propose WEBSERV, an environment that includes 1) a compact, site-agnostic browser environment that balances context and action complexity, and 2) a scalable RL environment via efficient launching and resetting web-servers to enable scalable RL training and evaluation. We evaluate WEBSERV on the shopping CMS and Gitlab tasks in WebArena, achieving state-of-the-art single-prompt success rates while cutting launch latency by ~5x and storage need by ~240x, with a comparable memory footprint, enabling 200+ concurrent containers on a single host.

**arXiv ID:** 2510.16252
</details>

<details>
<summary><strong>UltraCUA: A Foundation Model for Computer Use Agents with Hybrid Action</strong> - Yuhao Yang, Zhen Yang, Zi-Yi Dou, Anh Nguyen, Keen You, Omar Attia, Andrew Szot, Michael Feng, Ram Ramrakhya, Alexander Toshev, Chao Huang, Yinfei Yang, Zhe Gan - [[pdf]](https://arxiv.org/pdf/2510.17790)</summary>

**Abstract:** Multimodal agents for computer use rely exclusively on primitive actions (click, type, scroll) that require accurate visual grounding and lengthy execution chains, leading to cascading failures and performance bottlenecks. While other agents leverage rich programmatic interfaces (APIs, MCP servers, tools), computer-use agents (CUAs) remain isolated from these capabilities. We present UltraCUA, a foundation model that bridges this gap through hybrid action -- seamlessly integrating GUI primitives with high-level programmatic tool calls. To achieve this, our approach comprises four key components: (1) an automated pipeline that scales programmatic tools from software documentation, open-source repositories, and code generation; (2) a synthetic data engine producing over 17,000 verifiable tasks spanning real-world computer-use scenarios; (3) a large-scale high-quality hybrid action trajectory collection with both low-level GUI actions and high-level programmatic tool calls; and (4) a two-stage training pipeline combining supervised fine-tuning with online reinforcement learning, enabling strategic alternation between low-level and high-level actions. Experiments with our 7B and 32B models demonstrate substantial improvements over state-of-the-art agents. On OSWorld, UltraCUA models achieve an average 22% relative improvement over base models, while being 11% faster in terms of steps. Out-of-domain evaluation on WindowsAgentArena shows our model reaches 21.7% success rate, outperforming baselines trained on Windows data. The hybrid action mechanism proves critical, reducing error propagation while maintaining execution efficiency.

**arXiv ID:** 2510.17790
</details>

<details>
<summary><strong>TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning</strong> - Jinyang Wu, Chonghua Liao, Mingkuan Feng, Shuai Zhang, Zhengqi Wen, Haoran Luo, Ling Yang, Huazhe Xu, Jianhua Tao - [[pdf]](https://arxiv.org/pdf/2505.15692)</summary>

**Abstract:** Reinforcement learning (RL) has emerged as an effective paradigm for enhancing model reasoning. However, existing RL methods like GRPO often rely on unstructured self-sampling to fit scalar rewards, often producing inefficient rollouts that fail to capture transferable problem-solving strategies. To address these limitations, we propose **TemplateRL**, a structured template-guided RL framework that augments policy optimization with explicit template guidance. Our approach first constructs a problem-solving template library via MCTS on a small seed set, then seamlessly integrates this high-level structured guidance into RL training. By guiding rollout generation to align with proven template structures, TemplateRL significantly improves high-quality trajectory hit rates while reducing ineffective exploration. This structure-guided design steers the policy toward validated strategic patterns, stabilizing training dynamics, and enhancing RL sampling efficiency. Notably, the explicit template library is interpretable, editable, and supports online updates-enabling continuous updates during both training and inference. Extensive experiments demonstrate that TemplateRL outperforms GRPO by 99% on AIME and 41% on AMC, with superior stability on weak models and remarkable cross-domain generalization, highlighting its potential for broader tasks.

**arXiv ID:** 2505.15692
</details>

<details>
<summary><strong>AutoGraph-R1: End-to-End Reinforcement Learning for Knowledge Graph Construction</strong> - Hong Ting Tsang, Jiaxin Bai, Haoyu Huang, Qiao Xiao, Tianshi Zheng, Baixuan Xu, Shujie Liu, Yangqiu Song - [[pdf]](https://arxiv.org/pdf/2510.15339)</summary>

**Abstract:** Building effective knowledge graphs (KGs) for Retrieval-Augmented Generation (RAG) is pivotal for advancing question answering (QA) systems. However, its effectiveness is hindered by a fundamental disconnect: the knowledge graph (KG) construction process is decoupled from its downstream application, yielding suboptimal graph structures. To bridge this gap, we introduce AutoGraph-R1, the first framework to directly optimize KG construction for task performance using Reinforcement Learning (RL). AutoGraph-R1 trains an LLM constructor by framing graph generation as a policy learning problem, where the reward is derived from the graph's functional utility in a RAG pipeline. We design two novel, task-aware reward functions, one for graphs as knowledge carriers and another as knowledge indices. Across multiple QA benchmarks, AutoGraph-R1 consistently enables graph RAG methods to achieve significant performance gains over using task-agnostic baseline graphs. Our work shows it is possible to close the loop between construction and application, shifting the paradigm from building intrinsically ``good'' graphs to building demonstrably ``useful'' ones.

**arXiv ID:** 2510.15339
</details>

<details>
<summary><strong>Feature-driven reinforcement learning for photovoltaic in continuous intraday trading</strong> - Arega Getaneh Abate, Xiufeng Liu, Ruyu Liu, Xiaobing Zhang - [[pdf]](https://arxiv.org/pdf/2510.16021)</summary>

**Abstract:** Photovoltaic (PV) operators face substantial uncertainty in generation and short-term electricity prices. Continuous intraday markets enable producers to adjust their positions in real time, potentially improving revenues and reducing imbalance costs. We propose a feature-driven reinforcement learning (RL) approach for PV intraday trading that integrates data-driven features into the state and learns bidding policies in a sequential decision framework. The problem is cast as a Markov Decision Process with a reward that balances trading profit and imbalance penalties and is solved with Proximal Policy Optimization (PPO) using a predominantly linear, interpretable policy. Trained on historical market data and evaluated out-of-sample, the strategy consistently outperforms benchmark baselines across diverse scenarios. Extensive validation shows rapid convergence, real-time inference, and transparent decision rules. Learned weights highlight the central role of market microstructure and historical features. Taken together, these results indicate that feature-driven RL offers a practical, data-efficient, and operationally deployable pathway for active intraday participation by PV producers.

**arXiv ID:** 2510.16021
</details>

<details>
<summary><strong>Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing</strong> - Baode Wang, Biao Wu, Weizhen Li, Meng Fang, Zuming Huang, Jun Huang, Haozhe Wang, Yanjie Liang, Ling Chen, Wei Chu, Yuan Qi - [[pdf]](https://arxiv.org/pdf/2510.15349)</summary>

**Abstract:** Document parsing from scanned images into structured formats remains a significant challenge due to its complexly intertwined elements such as text paragraphs, figures, formulas, and tables. Existing supervised fine-tuning methods often struggle to generalize across diverse document types, leading to poor performance, particularly on out-of-distribution data. This issue is further exacerbated by the limited availability of high-quality training data for layout-aware parsing tasks. To address these challenges, we introduce LayoutRL, a reinforcement learning framework that optimizes layout understanding through composite rewards integrating normalized edit distance, paragraph count accuracy, and reading order preservation. To support this training, we construct the Infinity-Doc-400K dataset, which we use to train Infinity-Parser, a vision-language model demonstrating robust generalization across various domains. Extensive evaluations on benchmarks including OmniDocBench, olmOCR-Bench, PubTabNet, and FinTabNet show that Infinity-Parser consistently achieves state-of-the-art performance across a broad range of document types, languages, and structural complexities, substantially outperforming both specialized document parsing systems and general-purpose vision-language models. We will release our code, dataset, and model to facilitate reproducible research in document parsing.

**arXiv ID:** 2510.15349
</details>

<details>
<summary><strong>Human-Allied Relational Reinforcement Learning</strong> - Fateme Golivand Darvishvand, Hikaru Shindo, Sahil Sidheekh, Kristian Kersting, Sriraam Natarajan - [[pdf]](https://arxiv.org/pdf/2510.16188)</summary>

**Abstract:** Reinforcement learning (RL) has experienced a second wind in the past decade. While incredibly successful in images and videos, these systems still operate within the realm of propositional tasks ignoring the inherent structure that exists in the problem. Consequently, relational extensions (RRL) have been developed for such structured problems that allow for effective generalization to arbitrary number of objects. However, they inherently make strong assumptions about the problem structure. We introduce a novel framework that combines RRL with object-centric representation to handle both structured and unstructured data. We enhance learning by allowing the system to actively query the human expert for guidance by explicitly modeling the uncertainty over the policy. Our empirical evaluation demonstrates the effectiveness and efficiency of our proposed approach.

**arXiv ID:** 2510.16188
</details>

<details>
<summary><strong>Continuous Q-Score Matching: Diffusion Guided Reinforcement Learning for Continuous-Time Control</strong> - Chengxiu Hua, Jiawen Gu, Yushun Tang - [[pdf]](https://arxiv.org/pdf/2510.17122)</summary>

**Abstract:** Reinforcement learning (RL) has achieved significant success across a wide range of domains, however, most existing methods are formulated in discrete time. In this work, we introduce a novel RL method for continuous-time control, where stochastic differential equations govern state-action dynamics. Departing from traditional value function-based approaches, our key contribution is the characterization of continuous-time Q-functions via a martingale condition and the linking of diffusion policy scores to the action gradient of a learned continuous Q-function by the dynamic programming principle. This insight motivates Continuous Q-Score Matching (CQSM), a score-based policy improvement algorithm. Notably, our method addresses a long-standing challenge in continuous-time RL: preserving the action-evaluation capability of Q-functions without relying on time discretization. We further provide theoretical closed-form solutions for linear-quadratic (LQ) control problems within our framework. Numerical results in simulated environments demonstrate the effectiveness of our proposed method and compare it to popular baselines.

**arXiv ID:** 2510.17122
</details>

<details>
<summary><strong>ALPINE: A Lightweight and Adaptive Privacy-Decision Agent Framework for Dynamic Edge Crowdsensing</strong> - Guanjie Cheng, Siyang Liu, Junqin Huang, Xinkui Zhao, Yin Wang, Mengying Zhu, Linghe Kong, Shuiguang Deng - [[pdf]](https://arxiv.org/pdf/2510.17162)</summary>

**Abstract:** Mobile edge crowdsensing (MECS) systems continuously generate and transmit user data in dynamic, resource-constrained environments, exposing users to significant privacy threats. In practice, many privacy-preserving mechanisms build on differential privacy (DP). However, static DP mechanisms often fail to adapt to evolving risks, for example, shifts in adversarial capabilities, resource constraints and task requirements, resulting in either excessive noise or inadequate protection. To address this challenge, we propose ALPINE, a lightweight, adaptive framework that empowers terminal devices to autonomously adjust differential privacy levels in real time. ALPINE operates as a closed-loop control system consisting of four modules: dynamic risk perception, privacy decision via twin delayed deep deterministic policy gradient (TD3), local privacy execution and performance verification from edge nodes. Based on environmental risk assessments, we design a reward function that balances privacy gains, data utility and energy cost, guiding the TD3 agent to adaptively tune noise magnitude across diverse risk scenarios and achieve a dynamic equilibrium among privacy, utility and cost. Both the collaborative risk model and pretrained TD3-based agent are designed for low-overhead deployment. Extensive theoretical analysis and real-world simulations demonstrate that ALPINE effectively mitigates inference attacks while preserving utility and cost, making it practical for large-scale edge applications.

**arXiv ID:** 2510.17162
</details>

<details>
<summary><strong>Optimizing Energy Management of Smart Grid using Reinforcement Learning aided by Surrogate models built using Physics-informed Neural Networks</strong> - Julen Cestero, Carmine Delle Femine, Kenji S. Muro, Marco Quartulli, Marcello Restelli - [[pdf]](https://arxiv.org/pdf/2510.17380)</summary>

**Abstract:** Optimizing the energy management within a smart grids scenario presents significant challenges, primarily due to the complexity of real-world systems and the intricate interactions among various components. Reinforcement Learning (RL) is gaining prominence as a solution for addressing the challenges of Optimal Power Flow in smart grids. However, RL needs to iterate compulsively throughout a given environment to obtain the optimal policy. This means obtaining samples from a, most likely, costly simulator, which can lead to a sample efficiency problem. In this work, we address this problem by substituting costly smart grid simulators with surrogate models built using Phisics-informed Neural Networks (PINNs), optimizing the RL policy training process by arriving to convergent results in a fraction of the time employed by the original environment.

**arXiv ID:** 2510.17380
</details>

<details>
<summary><strong>An Empirical Study of Lagrangian Methods in Safe Reinforcement Learning</strong> - Lindsay Spoor, Álvaro Serra-Gómez, Aske Plaat, Thomas Moerland - [[pdf]](https://arxiv.org/pdf/2510.17564)</summary>

**Abstract:** In safety-critical domains such as robotics, navigation and power systems, constrained optimization problems arise where maximizing performance must be carefully balanced with associated constraints. Safe reinforcement learning provides a framework to address these challenges, with Lagrangian methods being a popular choice. However, the effectiveness of Lagrangian methods crucially depends on the choice of the Lagrange multiplier $\lambda$, which governs the trade-off between return and constraint cost. A common approach is to update the multiplier automatically during training. Although this is standard in practice, there remains limited empirical evidence on the robustness of an automated update and its influence on overall performance. Therefore, we analyze (i) optimality and (ii) stability of Lagrange multipliers in safe reinforcement learning across a range of tasks. We provide $\lambda$-profiles that give a complete visualization of the trade-off between return and constraint cost of the optimization problem. These profiles show the highly sensitive nature of $\lambda$ and moreover confirm the lack of general intuition for choosing the optimal value $\lambda^*$. Our findings additionally show that automated multiplier updates are able to recover and sometimes even exceed the optimal performance found at $\lambda^*$ due to the vast difference in their learning trajectories. Furthermore, we show that automated multiplier updates exhibit oscillatory behavior during training, which can be mitigated through PID-controlled updates. However, this method requires careful tuning to achieve consistently better performance across tasks. This highlights the need for further research on stabilizing Lagrangian methods in safe reinforcement learning. The code used to reproduce our results can be found at this https URL.

**arXiv ID:** 2510.17564
</details>

<details>
<summary><strong>Efficient Algorithms for Mitigating Uncertainty and Risk in Reinforcement Learning</strong> - Xihong Su - [[pdf]](https://arxiv.org/pdf/2510.17690)</summary>

**Abstract:** This dissertation makes three main contributions. First, We identify a new connection between policy gradient and dynamic programming in MMDPs and propose the Coordinate Ascent Dynamic Programming (CADP) algorithm to compute a Markov policy that maximizes the discounted return averaged over the uncertain models. CADP adjusts model weights iteratively to guarantee monotone policy improvements to a local maximum. Second, We establish sufficient and necessary conditions for the exponential ERM Bellman operator to be a contraction and prove the existence of stationary deterministic optimal policies for ERM-TRC and EVaR-TRC. We also propose exponential value iteration, policy iteration, and linear programming algorithms for computing optimal stationary policies for ERM-TRC and EVaR-TRC. Third, We propose model-free Q-learning algorithms for computing policies with risk-averse objectives: ERM-TRC and EVaR-TRC. The challenge is that Q-learning ERM Bellman may not be a contraction. Instead, we use the monotonicity of Q-learning ERM Bellman operators to derive a rigorous proof that the ERM-TRC and the EVaR-TRC Q-learning algorithms converge to the optimal risk-averse value functions. The proposed Q-learning algorithms compute the optimal stationary policy for ERM-TRC and EVaR-TRC.

**arXiv ID:** 2510.17690
</details>

<details>
<summary><strong>Self-Supervised Learning to Fly using Efficient Semantic Segmentation and Metric Depth Estimation for Low-Cost Autonomous UAVs</strong> - Sebastian Mocanu, Emil Slusanschi, Marius Leordeanu - [[pdf]](https://arxiv.org/pdf/2510.16624)</summary>

**Abstract:** This paper presents a vision-only autonomous flight system for small UAVs operating in controlled indoor environments. The system combines semantic segmentation with monocular depth estimation to enable obstacle avoidance, scene exploration, and autonomous safe landing operations without requiring GPS or expensive sensors such as LiDAR. A key innovation is an adaptive scale factor algorithm that converts non-metric monocular depth predictions into accurate metric distance measurements by leveraging semantic ground plane detection and camera intrinsic parameters, achieving a mean distance error of 14.4 cm. The approach uses a knowledge distillation framework where a color-based Support Vector Machine (SVM) teacher generates training data for a lightweight U-Net student network (1.6M parameters) capable of real-time semantic segmentation. For more complex environments, the SVM teacher can be replaced with a state-of-the-art segmentation model. Testing was conducted in a controlled 5x4 meter laboratory environment with eight cardboard obstacles simulating urban structures. Extensive validation across 30 flight tests in a real-world environment and 100 flight tests in a digital-twin environment demonstrates that the combined segmentation and depth approach increases the distance traveled during surveillance and reduces mission time while maintaining 100% success rates. The system is further optimized through end-to-end learning, where a compact student neural network learns complete flight policies from demonstration data generated by our best-performing method, achieving an 87.5% autonomous mission success rate. This work advances practical vision-based drone navigation in structured environments, demonstrating solutions for metric depth estimation and computational efficiency challenges that enable deployment on resource-constrained platforms.

**arXiv ID:** 2510.16624
</details>

<details>
<summary><strong>Context-Based Meta Reinforcement Learning for Robust and Adaptable Peg-in-Hole Assembly Tasks</strong> - Ahmed Shokry, Walid Gomaa, Tobias Zaenker, Murad Dawood, Rohit Menon, Shady A. Maged, Mohammed I. Awad, Maren Bennewitz - [[pdf]](https://arxiv.org/pdf/2409.16208)</summary>

**Abstract:** Autonomous assembly is an essential capability for industrial and service robots, with Peg-in-Hole (PiH) insertion being one of the core tasks. However, PiH assembly in unknown environments is still challenging due to uncertainty in task parameters, such as the hole position and orientation, resulting from sensor noise. Although context-based meta reinforcement learning (RL) methods have been previously presented to adapt to unknown task parameters in PiH assembly tasks, the performance depends on a sample-inefficient procedure or human demonstrations. Thus, to enhance the applicability of meta RL in real-world PiH assembly tasks, we propose to train the agent to use information from the robot's forward kinematics and an uncalibrated camera. Furthermore, we improve the performance by efficiently adapting the meta-trained agent to use data from force/torque sensor. Finally, we propose an adaptation procedure for out-of-distribution tasks whose parameters are different from the training tasks. Experiments on simulated and real robots prove that our modifications enhance the sample efficiency during meta training, real-world adaptation performance, and generalization of the context-based meta RL agent in PiH assembly tasks compared to previous approaches.

**arXiv ID:** 2409.16208
</details>

<details>
<summary><strong>MoRe-ERL: Learning Motion Residuals using Episodic Reinforcement Learning</strong> - Xi Huang, Hongyi Zhou, Ge Li, Yucheng Tang, Weiran Liao, Björn Hein, Tamim Asfour, Rudolf Lioutikov - [[pdf]](https://arxiv.org/pdf/2508.01409)</summary>

**Abstract:** We propose MoRe-ERL, a framework that combines Episodic Reinforcement Learning (ERL) and residual learning, which refines preplanned reference trajectories into safe, feasible, and efficient task-specific trajectories. This framework is general enough to incorporate into arbitrary ERL methods and motion generators seamlessly. MoRe-ERL identifies trajectory segments requiring modification while preserving critical task-related maneuvers. Then it generates smooth residual adjustments using B-Spline-based movement primitives to ensure adaptability to dynamic task contexts and smoothness in trajectory refinement. Experimental results demonstrate that residual learning significantly outperforms training from scratch using ERL methods, achieving superior sample efficiency and task performance. Hardware evaluations further validate the framework, showing that policies trained in simulation can be directly deployed in real-world systems, exhibiting a minimal sim-to-real gap.

**arXiv ID:** 2508.01409
</details>

<details>
<summary><strong>LIPM-Guided Reinforcement Learning for Stable and Perceptive Locomotion in Bipedal Robots</strong> - Haokai Su, Haoxiang Luo, Shunpeng Yang, Kaiwen Jiang, Wei Zhang, Hua Chen - [[pdf]](https://arxiv.org/pdf/2509.09106)</summary>

**Abstract:** Achieving stable and robust perceptive locomotion for bipedal robots in unstructured outdoor environments remains a critical challenge due to complex terrain geometry and susceptibility to external disturbances. In this work, we propose a novel reward design inspired by the Linear Inverted Pendulum Model (LIPM) to enable perceptive and stable locomotion in the wild. The LIPM provides theoretical guidance for dynamic balance by regulating the center of mass (CoM) height and the torso orientation. These are key factors for terrain-aware locomotion, as they help ensure a stable viewpoint for the robot's camera. Building on this insight, we design a reward function that promotes balance and dynamic stability while encouraging accurate CoM trajectory tracking. To adaptively trade off between velocity tracking and stability, we leverage the Reward Fusion Module (RFM) approach that prioritizes stability when needed. A double-critic architecture is adopted to separately evaluate stability and locomotion objectives, improving training efficiency and robustness. We validate our approach through extensive experiments on a bipedal robot in both simulation and real-world outdoor environments. The results demonstrate superior terrain adaptability, disturbance rejection, and consistent performance across a wide range of speeds and perceptual conditions.

**arXiv ID:** 2509.09106
</details>

<details>
<summary><strong>CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions</strong> - Lizhi Yang, Blake Werner, Massimiliano de Sa, Aaron D. Ames - [[pdf]](https://arxiv.org/pdf/2510.14959)</summary>

**Abstract:** Reinforcement learning (RL), while powerful and expressive, can often prioritize performance at the expense of safety. Yet safety violations can lead to catastrophic outcomes in real-world deployments. Control Barrier Functions (CBFs) offer a principled method to enforce dynamic safety -- traditionally deployed online via safety filters. While the result is safe behavior, the fact that the RL policy does not have knowledge of the CBF can lead to conservative behaviors. This paper proposes CBF-RL, a framework for generating safe behaviors with RL by enforcing CBFs in training. CBF-RL has two key attributes: (1) minimally modifying a nominal RL policy to encode safety constraints via a CBF term, (2) and safety filtering of the policy rollouts in training. Theoretically, we prove that continuous-time safety filters can be deployed via closed-form expressions on discrete-time roll-outs. Practically, we demonstrate that CBF-RL internalizes the safety constraints in the learned policy -- both enforcing safer actions and biasing towards safer rewards -- enabling safe deployment without the need for an online safety filter. We validate our framework through ablation studies on navigation tasks and on the Unitree G1 humanoid robot, where CBF-RL enables safer exploration, faster convergence, and robust performance under uncertainty, enabling the humanoid robot to avoid obstacles and climb stairs safely in real-world settings without a runtime safety filter.

**arXiv ID:** 2510.14959
</details>

<details>
<summary><strong>DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning</strong> - Leander Diaz-Bone, Marco Bagatella, Jonas Hübotter, Andreas Krause - [[pdf]](https://arxiv.org/pdf/2505.19850)</summary>

**Abstract:** Sparse-reward reinforcement learning (RL) can model a wide range of highly complex tasks. Solving sparse-reward tasks is RL's core premise, requiring efficient exploration coupled with long-horizon credit assignment, and overcoming these challenges is key for building self-improving agents with superhuman ability. Prior work commonly explores with the objective of solving many sparse-reward tasks, making exploration of individual high-dimensional, long-horizon tasks intractable. We argue that solving such challenging tasks requires solving simpler tasks that are relevant to the target task, i.e., whose achieval will teach the agent skills required for solving the target task. We demonstrate that this sense of direction, necessary for effective exploration, can be extracted from existing RL algorithms, without leveraging any prior information. To this end, we propose a method for directed sparse-reward goal-conditioned very long-horizon RL (DISCOVER), which selects exploratory goals in the direction of the target task. We connect DISCOVER to principled exploration in bandits, formally bounding the time until the target task becomes achievable in terms of the agent's initial distance to the target, but independent of the volume of the space of all tasks. We then perform a thorough evaluation in high-dimensional environments. We find that the directed goal selection of DISCOVER solves exploration problems that are beyond the reach of prior state-of-the-art exploration methods in RL.

**arXiv ID:** 2505.19850
</details>

<details>
<summary><strong>AGENTSAFE: Benchmarking the Safety of Embodied Agents on Hazardous Instructions</strong> - Zonghao Ying, Le Wang, Yisong Xiao, Jiakai Wang, Yuqing Ma, Jinyang Guo, Zhenfei Yin, Mingchuan Zhang, Aishan Liu, Xianglong Liu - [[pdf]](https://arxiv.org/pdf/2506.14697)</summary>

**Abstract:** The integration of vision-language models (VLMs) is driving a new generation of embodied agents capable of operating in human-centered environments. However, as deployment expands, these systems face growing safety risks, particularly when executing hazardous instructions. Current safety evaluation benchmarks remain limited: they cover only narrow scopes of hazards and focus primarily on final outcomes, neglecting the agent's full perception-planning-execution process and thereby obscuring critical failure modes. Therefore, we present SAFE, a benchmark for systematically assessing the safety of embodied VLM agents on hazardous instructions. SAFE comprises three components: SAFE-THOR, an extensible adversarial simulation sandbox with a universal adapter that maps high-level VLM outputs to low-level embodied controls, supporting diverse agent workflow integration; SAFE-VERSE, a risk-aware task suite inspired by Asimov's Three Laws of Robotics, comprising 45 adversarial scenarios, 1,350 hazardous tasks, and 9,900 instructions that span risks to humans, environments, and agents; and SAFE-DIAGNOSE, a multi-level and fine-grained evaluation protocol measuring agent performance across perception, planning, and execution. Applying SAFE to nine state-of-the-art VLMs and two embodied agent workflows, we uncover systematic failures in translating hazard recognition into safe planning and execution. Our findings reveal fundamental limitations in current safety alignment and demonstrate the necessity of a comprehensive, multi-stage evaluation for developing safer embodied intelligence.

**arXiv ID:** 2506.14697
</details>

<details>
<summary><strong>SegDAC: Improving Visual Reinforcement Learning by Extracting Dynamic Objectc-Centric Representations from Pretrained Vision Models</strong> - Alexandre Brown, Glen Berseth - [[pdf]](https://arxiv.org/pdf/2508.09325)</summary>

**Abstract:** Visual reinforcement learning (RL) is challenging due to the need to extract useful representations from high-dimensional inputs while learning effective control from sparse and noisy rewards. Although large perception models exist, integrating them effectively into RL for visual generalization and improved sample efficiency remains difficult. We propose SegDAC, a Segmentation-Driven Actor-Critic method. SegDAC uses Segment Anything (SAM) for object-centric decomposition and YOLO-World to ground the image segmentation process via text inputs. It includes a novel transformer-based architecture that supports a dynamic number of segments at each time step and effectively learns which segments to focus on using online RL, without using human labels. By evaluating SegDAC over a challenging visual generalization benchmark using Maniskill3, which covers diverse manipulation tasks under strong visual perturbations, we demonstrate that SegDAC achieves significantly better visual generalization, doubling prior performance on the hardest setting and matching or surpassing prior methods in sample efficiency across all evaluated tasks.

**arXiv ID:** 2508.09325
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
