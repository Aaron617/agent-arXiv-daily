# Agent arXiv Daily

**Last Updated:** 2025-09-03 04:00:03

**Total Papers:** 117

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
<summary><strong>HiVA: Self-organized Hierarchical Variable Agent via Goal-driven Semantic-Topological Evolution</strong> - Jinzhou Tang, Jusheng Zhang, Qinhan Lv, Sidi Liu, Jing Yang, Chengpei Tang, Keze Wang - [[pdf]](https://arxiv.org/pdf/2509.00189)</summary>

**Abstract:** Autonomous agents play a crucial role in advancing Artificial General Intelligence, enabling problem decomposition and tool orchestration through Large Language Models (LLMs). However, existing paradigms face a critical trade-off. On one hand, reusable fixed workflows require manual reconfiguration upon environmental changes; on the other hand, flexible reactive loops fail to distill reasoning progress into transferable structures. We introduce Hierarchical Variable Agent (HiVA), a novel framework modeling agentic workflows as self-organized graphs with the Semantic-Topological Evolution (STEV) algorithm, which optimizes hybrid semantic-topological spaces using textual gradients as discrete-domain surrogates for backpropagation. The iterative process comprises Multi-Armed Bandit-infused forward routing, diagnostic gradient generation from environmental feedback, and coordinated updates that co-evolve individual semantics and topology for collective optimization in unknown environments. Experiments on dialogue, coding, Long-context Q&A, mathematical, and agentic benchmarks demonstrate improvements of 5-10% in task accuracy and enhanced resource efficiency over existing baselines, establishing HiVA's effectiveness in autonomous task execution.

**arXiv ID:** 2509.00189
</details>

<details>
<summary><strong>UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning</strong> - Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, Wanjun Zhong, Yining Ye, Yujia Qin, Yuwen Xiong, Yuxin Song, Zhiyong Wu, Bo Li, Chen Dun, Chong Liu, Fuxing Leng, Hanbin Wang, Hao Yu, Haobin Chen, Hongyi Guo, Jing Su, Jingjia Huang, Kai Shen, Kaiyu Shi, Lin Yan, Peiyao Zhao, Pengfei Liu, Qinghao Ye, Renjie Zheng, Wayne Xin Zhao, Wen Heng, Wenhao Huang, Wenqian Wang, Xiaobo Qin, Yi Lin, Youbin Wu, Zehui Chen, Zihao Wang, Baoquan Zhong, Xinchun Zhang, Xujing Li, Yuanfan Li, Zhongkai Zhao, Chengquan Jiang, Faming Wu, Haotian Zhou, Jinlin Pang, Li Han, Qianli Ma, Siyao Liu, Songhua Cai, Wenqi Fu, Xin Liu, Zhi Zhang, Bo Zhou, Guoliang Li, Jiajun Shi, Jiale Yang, Jie Tang, Li Li, Taoran Lu, Woyu Lin, Xiaokang Tong, Xinyao Li, Yichi Zhang, Yu Miao, Zhengxuan Jiang, Zili Li, Ziyuan Zhao, Chenxin Li, Dehua Ma, Feng Lin, Ge Zhang, Haihua Yang, Hangyu Guo, Hongda Zhu, Jiaheng Liu, Junda Du, Kai Cai, Kuanye Li, Lichen Yuan, Meilan Han, Minchao Wang, Shuyue Guo, Tianhao Cheng, Xiaobo Ma, Xiaojun Xiao, Xiaolong Huang, Xinjie Chen, Yidi Du, Yilin Chen, Yiwen Wang, Zhaojian Li, Zhenzhu Yang, Zhiyuan Zeng, Chaolin Jin - [[pdf]](https://arxiv.org/pdf/2509.02544)</summary>

**Abstract:** The development of autonomous agents for graphical user interfaces (GUIs) presents major challenges in artificial intelligence. While recent advances in native agent models have shown promise by unifying perception, reasoning, action, and memory through end-to-end learning, open problems remain in data scalability, multi-turn reinforcement learning (RL), the limitations of GUI-only operation, and environment stability. In this technical report, we present UI-TARS-2, a native GUI-centered agent model that addresses these challenges through a systematic training methodology: a data flywheel for scalable data generation, a stabilized multi-turn RL framework, a hybrid GUI environment that integrates file systems and terminals, and a unified sandbox platform for large-scale rollouts. Empirical evaluation demonstrates that UI-TARS-2 achieves significant improvements over its predecessor UI-TARS-1.5. On GUI benchmarks, it reaches 88.2 on Online-Mind2Web, 47.5 on OSWorld, 50.6 on WindowsAgentArena, and 73.3 on AndroidWorld, outperforming strong baselines such as Claude and OpenAI agents. In game environments, it attains a mean normalized score of 59.8 across a 15-game suite-roughly 60% of human-level performance-and remains competitive with frontier proprietary models (e.g., OpenAI o3) on LMGame-Bench. Additionally, the model can generalize to long-horizon information-seeking tasks and software engineering benchmarks, highlighting its robustness across diverse agent tasks. Detailed analyses of training dynamics further provide insights into achieving stability and efficiency in large-scale agent RL. These results underscore UI-TARS-2's potential to advance the state of GUI agents and exhibit strong generalization to real-world interactive scenarios.

**arXiv ID:** 2509.02544
</details>

<details>
<summary><strong>WATCHED: A Web AI Agent Tool for Combating Hate Speech by Expanding Data</strong> - Paloma Piot, Diego Sánchez, Javier Parapar - [[pdf]](https://arxiv.org/pdf/2509.01379)</summary>

**Abstract:** Online harms are a growing problem in digital spaces, putting user safety at risk and reducing trust in social media platforms. One of the most persistent forms of harm is hate speech. To address this, we need tools that combine the speed and scale of automated systems with the judgment and insight of human moderators. These tools should not only find harmful content but also explain their decisions clearly, helping to build trust and understanding. In this paper, we present WATCHED, a chatbot designed to support content moderators in tackling hate speech. The chatbot is built as an Artificial Intelligence Agent system that uses Large Language Models along with several specialised tools. It compares new posts with real examples of hate speech and neutral content, uses a BERT-based classifier to help flag harmful messages, looks up slang and informal language using sources like Urban Dictionary, generates chain-of-thought reasoning, and checks platform guidelines to explain and support its decisions. This combination allows the chatbot not only to detect hate speech but to explain why content is considered harmful, grounded in both precedent and policy. Experimental results show that our proposed method surpasses existing state-of-the-art methods, reaching a macro F1 score of 0.91. Designed for moderators, safety teams, and researchers, the tool helps reduce online harms by supporting collaboration between AI and human oversight.

**arXiv ID:** 2509.01379
</details>

<details>
<summary><strong>ChipChat: Low-Latency Cascaded Conversational Agent in MLX</strong> - Tatiana Likhomanenko, Luke Carlson, Richard He Bai, Zijin Gu, Han Tran, Zakaria Aldeneh, Yizhe Zhang, Ruixiang Zhang, Huangjie Zheng, Navdeep Jaitly - [[pdf]](https://arxiv.org/pdf/2509.00078)</summary>

**Abstract:** The emergence of large language models (LLMs) has transformed spoken dialog systems, yet the optimal architecture for real-time on-device voice agents remains an open question. While end-to-end approaches promise theoretical advantages, cascaded systems (CSs) continue to outperform them in language understanding tasks, despite being constrained by sequential processing latency. In this work, we introduce ChipChat, a novel low-latency CS that overcomes traditional bottlenecks through architectural innovations and streaming optimizations. Our system integrates streaming (a) conversational speech recognition with mixture-of-experts, (b) state-action augmented LLM, (c) text-to-speech synthesis, (d) neural vocoder, and (e) speaker modeling. Implemented using MLX, ChipChat achieves sub-second response latency on a Mac Studio without dedicated GPUs, while preserving user privacy through complete on-device processing. Our work shows that strategically redesigned CSs can overcome their historical latency limitations, offering a promising path forward for practical voice-based AI agents.

**arXiv ID:** 2509.00078
</details>

<details>
<summary><strong>FLM-Audio: Natural Monologues Improves Native Full-Duplex Chatbots via Dual Training</strong> - Yiqun Yao, Xiang Li, Xin Jiang, Xuezhi Fang, Naitong Yu, Wenjia Ma, Aixin Sun, Yequan Wang - [[pdf]](https://arxiv.org/pdf/2509.02521)</summary>

**Abstract:** Full-duplex dialog models are designed to listen and speak simultaneously with rapid responses to fast-changing user input. Among existing approaches, native full-duplex models merges different channels (e.g. listen and speak) in a single time step, overcoming the high response latency inherent to time-division multiplexing time-division multiplexing (TDM) alternatives. Yet, a key challenge remains: aligning textual monologues with audio streams that operate at different bitrates. The prevailing solution relies on word-level alignment, but this can degrade the language ability of large pre-trained models. Moreover, it requires highly accurate timestamps for every token, which introduces cascading errors and increases pre-processing costs. In this paper, we propose textual monologues in continuous tokens sequence, namely "natural" monologues, which mimics humanoid cognitive behavior in dialogs. For temporal alignment, we alternate the position of the natural monologue - leading or trailing the audio - across different training stages. This "dual" training paradigm proves highly effective in building FLM-Audio, our 7B spoken dialog model that demonstrates superior responsiveness, duplexity, and chatting experiences, as confirmed by experimental results.

**arXiv ID:** 2509.02521
</details>

<details>
<summary><strong>How to Make Museums More Interactive? Case Study of Artistic Chatbot</strong> - Filip J. Kucia, Bartosz Grabek, Szymon D. Trochimiak, Anna Wróblewska - [[pdf]](https://arxiv.org/pdf/2509.00572)</summary>

**Abstract:** Conversational agents powered by Large Language Models (LLMs) are increasingly utilized in educational settings, in particular in individual closed digital environments, yet their potential adoption in the physical learning environments like cultural heritage sites, museums, and art galleries remains relatively unexplored. In this study, we present Artistic Chatbot, a voice-to-voice RAG-powered chat system to support informal learning and enhance visitor engagement during a live art exhibition celebrating the 15th anniversary of the Faculty of Media Art at the Warsaw Academy of Fine Arts, Poland. The question answering (QA) chatbot responded to free-form spoken questions in Polish using the context retrieved from a curated, domain-specific knowledge base consisting of 226 documents provided by the organizers, including faculty information, art magazines, books, and journals. We describe the key aspects of the system architecture and user interaction design, as well as discuss the practical challenges associated with deploying chatbots at public cultural sites. Our findings, based on interaction analysis, demonstrate that chatbots such as Artistic Chatbot effectively maintain responses grounded in exhibition content (60\% of responses directly relevant), even when faced with unpredictable queries outside the target domain, showing their potential for increasing interactivity in public cultural sites.
GitHub project page: this https URL

**arXiv ID:** 2509.00572
</details>

<details>
<summary><strong>A Theoretical Framework of the Processes of Change in Psychotherapy Delivered by Artificial Agents</strong> - Arthur Bran Herbener, Malene Flensborg Damholdt - [[pdf]](https://arxiv.org/pdf/2509.02144)</summary>

**Abstract:** The question of whether artificial agents (e.g., chatbots and social robots) can replace human therapists has received notable attention following the recent launch of large language models. However, little is known about the processes of change in psychotherapy delivered by artificial agents. To facilitate hypothesis development and stimulate scientific debate, the present article offers the first theoretical framework of the processes of change in psychotherapy delivered by artificial agents. The theoretical framework rests upon a conceptual analysis of what active ingredients may be inherently linked to the presence of human therapists. We propose that human therapists' ontological status as human beings and sociocultural status as socially sanctioned healthcare professionals play crucial roles in promoting treatment outcomes. In the absence of the ontological and sociocultural status of human therapists, we propose what we coin the genuineness gap and credibility gap can emerge and undermine key processes of change in psychotherapy. Based on these propositions, we propose avenues for scientific investigations and practical applications aimed at leveraging the strengths of artificial agents and human therapists respectively. We also highlight the intricate agentic nature of artificial agents and discuss how this complicates endeavors to establish universally applicable propositions regarding the processes of change in these interventions.

**arXiv ID:** 2509.02144
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (22 papers)</h2></summary>

<details>
<summary><strong>Wrong Face, Wrong Move: The Social Dynamics of Emotion Misperception in Agent-Based Models</strong> - David Freire-Obregón - [[pdf]](https://arxiv.org/pdf/2509.00080)</summary>

**Abstract:** The ability of humans to detect and respond to others' emotions is fundamental to understanding social behavior. Here, agents are instantiated with emotion classifiers of varying accuracy to study the impact of perceptual accuracy on emergent emotional and spatial behavior. Agents are visually represented with face photos from the KDEF database and endowed with one of three classifiers trained on the JAFFE (poor), CK+ (medium), or KDEF (high) datasets. Agents communicate locally on a 2D toroidal lattice, perceiving neighbors' emotional state based on their classifier and responding with movement toward perceived positive emotions and away from perceived negative emotions. Note that the agents respond to perceived, instead of ground-truth, emotions, introducing systematic misperception and frustration. A battery of experiments is carried out on homogeneous and heterogeneous populations and scenarios with repeated emotional shocks. Results show that low-accuracy classifiers on the part of the agent reliably result in diminished trust, emotional disintegration into sadness, and disordered social organization. By contrast, the agent that develops high accuracy develops hardy emotional clusters and resilience to emotional disruptions. Even in emotionally neutral scenarios, misperception is enough to generate segregation and disintegration of cohesion. These findings underscore the fact that biases or imprecision in emotion recognition may significantly warp social processes and disrupt emotional integration.

**arXiv ID:** 2509.00080
</details>

<details>
<summary><strong>NEWSAGENT: Benchmarking Multimodal Agents as Journalists with Real-World Newswriting Tasks</strong> - Yen-Che Chien, Kuang-Da Wang, Wei-Yao Wang, Wen-Chih Peng - [[pdf]](https://arxiv.org/pdf/2509.00446)</summary>

**Abstract:** Recent advances in autonomous digital agents from industry (e.g., Manus AI and Gemini's research mode) highlight potential for structured tasks by autonomous decision-making and task decomposition; however, it remains unclear to what extent the agent-based systems can improve multimodal web data productivity. We study this in the realm of journalism, which requires iterative planning, interpretation, and contextual reasoning from multimodal raw contents to form a well structured news. We introduce NEWSAGENT, a benchmark for evaluating how agents can automatically search available raw contents, select desired information, and edit and rephrase to form a news article by accessing core journalistic functions. Given a writing instruction and firsthand data as how a journalist initiates a news draft, agents are tasked to identify narrative perspectives, issue keyword-based queries, retrieve historical background, and generate complete articles. Unlike typical summarization or retrieval tasks, essential context is not directly available and must be actively discovered, reflecting the information gaps faced in real-world news writing. NEWSAGENT includes 6k human-verified examples derived from real news, with multimodal contents converted to text for broad model compatibility. We evaluate open- and closed-sourced LLMs with commonly-used agentic frameworks on NEWSAGENT, which shows that agents are capable of retrieving relevant facts but struggling with planning and narrative integration. We believe that NEWSAGENT serves a realistic testbed for iterating and evaluating agent capabilities in terms of multimodal web data manipulation to real-world productivity.

**arXiv ID:** 2509.00446
</details>

<details>
<summary><strong>NetGent: Agent-Based Automation of Network Application Workflows</strong> - Jaber Daneshamooz, Eugene Vuong, Laasya Koduru, Sanjay Chandrasekaran, Arpit Gupta - [[pdf]](https://arxiv.org/pdf/2509.00625)</summary>

**Abstract:** We present NetGent, an AI-agent framework for automating complex application workflows to generate realistic network traffic datasets. Developing generalizable ML models for networking requires data collection from network environments with traffic that results from a diverse set of real-world web applications. However, using existing browser automation tools that are diverse, repeatable, realistic, and efficient remains fragile and costly. NetGent addresses this challenge by allowing users to specify workflows as natural-language rules that define state-dependent actions. These abstract specifications are compiled into nondeterministic finite automata (NFAs), which a state synthesis component translates into reusable, executable code. This design enables deterministic replay, reduces redundant LLM calls through state caching, and adapts quickly when application interfaces change. In experiments, NetGent automated more than 50+ workflows spanning video-on-demand streaming, live video streaming, video conferencing, social media, and web scraping, producing realistic traffic traces while remaining robust to UI variability. By combining the flexibility of language-based agents with the reliability of compiled execution, NetGent provides a scalable foundation for generating the diverse, repeatable datasets needed to advance ML in networking.

**arXiv ID:** 2509.00625
</details>

<details>
<summary><strong>FlashAdventure: A Benchmark for GUI Agents Solving Full Story Arcs in Diverse Adventure Games</strong> - Jaewoo Ahn, Junseo Kim, Heeseung Yun, Jaehyeon Son, Dongmin Park, Jaewoong Cho, Gunhee Kim - [[pdf]](https://arxiv.org/pdf/2509.01052)</summary>

**Abstract:** GUI agents powered by LLMs show promise in interacting with diverse digital environments. Among these, video games offer a valuable testbed due to their varied interfaces, with adventure games posing additional challenges through complex, narrative-driven interactions. Existing game benchmarks, however, lack diversity and rarely evaluate agents on completing entire storylines. To address this, we introduce FlashAdventure, a benchmark of 34 Flash-based adventure games designed to test full story arc completion and tackle the observation-behavior gap: the challenge of remembering and acting on earlier gameplay information. We also propose CUA-as-a-Judge, an automated gameplay evaluator, and COAST, an agentic framework leveraging long-term clue memory to better plan and solve sequential tasks. Experiments show current GUI agents struggle with full story arcs, while COAST improves milestone completion by bridging the observation-behavior gap. Nonetheless, a marked discrepancy between humans and best-performing agents warrants continued research efforts to narrow this divide.

**arXiv ID:** 2509.01052
</details>

<details>
<summary><strong>Communicative Agents for Slideshow Storytelling Video Generation based on LLMs</strong> - Jingxing Fan, Jinrong Shen, Yusheng Yao, Shuangqing Wang, Qian Wang, Yuling Wang - [[pdf]](https://arxiv.org/pdf/2509.01277)</summary>

**Abstract:** With the rapid advancement of artificial intelligence (AI), the proliferation of AI-generated content (AIGC) tasks has significantly accelerated developments in text-to-video generation. As a result, the field of video production is undergoing a transformative shift. However, conventional text-to-video models are typically constrained by high computational costs.
In this study, we propose Video-Generation-Team (VGTeam), a novel slide show video generation system designed to redefine the video creation pipeline through the integration of large language models (LLMs). VGTeam is composed of a suite of communicative agents, each responsible for a distinct aspect of video generation, such as scriptwriting, scene creation, and audio design. These agents operate collaboratively within a chat tower workflow, transforming user-provided textual prompts into coherent, slide-style narrative videos.
By emulating the sequential stages of traditional video production, VGTeam achieves remarkable improvements in both efficiency and scalability, while substantially reducing computational overhead. On average, the system generates videos at a cost of only $0.103, with a successful generation rate of 98.4%. Importantly, this framework maintains a high degree of creative fidelity and customization.
The implications of VGTeam are far-reaching. It democratizes video production by enabling broader access to high-quality content creation without the need for extensive resources. Furthermore, it highlights the transformative potential of language models in creative domains and positions VGTeam as a pioneering system for next-generation content creation.

**arXiv ID:** 2509.01277
</details>

<details>
<summary><strong>LLM-empowered Agents Simulation Framework for Scenario Generation in Service Ecosystem Governance</strong> - Deyu Zhou, Yuqi Hou, Xiao Xue, Xudong Lu, Qingzhong Li, Lizhen Cui - [[pdf]](https://arxiv.org/pdf/2509.01441)</summary>

**Abstract:** As the social environment is growing more complex and collaboration is deepening, factors affecting the healthy development of service ecosystem are constantly changing and diverse, making its governance a crucial research issue. Applying the scenario analysis method and conducting scenario rehearsals by constructing an experimental system before managers make decisions, losses caused by wrong decisions can be largely avoided. However, it relies on predefined rules to construct scenarios and faces challenges such as limited information, a large number of influencing factors, and the difficulty of measuring social elements. These challenges limit the quality and efficiency of generating social and uncertain scenarios for the service ecosystem. Therefore, we propose a scenario generator design method, which adaptively coordinates three Large Language Model (LLM) empowered agents that autonomously optimize experimental schemes to construct an experimental system and generate high quality scenarios. Specifically, the Environment Agent (EA) generates social environment including extremes, the Social Agent (SA) generates social collaboration structure, and the Planner Agent (PA) couples task-role relationships and plans task solutions. These agents work in coordination, with the PA adjusting the experimental scheme in real time by perceiving the states of each agent and these generating scenarios. Experiments on the ProgrammableWeb dataset illustrate our method generates more accurate scenarios more efficiently, and innovatively provides an effective way for service ecosystem governance related experimental system construction.

**arXiv ID:** 2509.01441
</details>

<details>
<summary><strong>Physics Supernova: AI Agent Matches Elite Gold Medalists at IPhO 2025</strong> - Jiahao Qiu, Jingzhe Shi, Xinzhe Juan, Zelin Zhao, Jiayi Geng, Shilong Liu, Hongru Wang, Sanfeng Wu, Mengdi Wang - [[pdf]](https://arxiv.org/pdf/2509.01659)</summary>

**Abstract:** Physics provides fundamental laws that describe and predict the natural world. AI systems aspiring toward more general, real-world intelligence must therefore demonstrate strong physics problem-solving abilities: to formulate and apply physical laws for explaining and predicting physical processes. The International Physics Olympiad (IPhO)--the world's most prestigious physics competition--offers a rigorous benchmark for this purpose. We introduce Physics Supernova, an AI agent system with superior physics problem-solving abilities that match elite IPhO gold medalists. In IPhO 2025 theory problems, Physics Supernova attains 23.5/30 points, ranking 14th of 406 contestants and surpassing the median performance of human gold medalists. We extensively analyzed Physics Supernova's capabilities and flexibility across diverse physics tasks. These results show that principled tool integration within agent systems can deliver competitive improvements in solving challenging science problems. The codes are available at this https URL.

**arXiv ID:** 2509.01659
</details>

<details>
<summary><strong>ARTPS: Depth-Enhanced Hybrid Anomaly Detection and Learnable Curiosity Score for Autonomous Rover Target Prioritization</strong> - Poyraz Baydemir - [[pdf]](https://arxiv.org/pdf/2509.00042)</summary>

**Abstract:** We present ARTPS (Autonomous Rover Target Prioritization System), a novel hybrid AI system that combines depth estimation, anomaly detection, and learnable curiosity scoring for autonomous exploration of planetary surfaces. Our approach integrates monocular depth estimation using Vision Transformers with multi-component anomaly detection and a weighted curiosity score that balances known value, anomaly signals, depth variance, and surface roughness. The system achieves state-of-the-art performance with AUROC of 0.94, AUPRC of 0.89, and F1-Score of 0.87 on Mars rover datasets. We demonstrate significant improvements in target prioritization accuracy through ablation studies and provide comprehensive analysis of component contributions. The hybrid fusion approach reduces false positives by 23% while maintaining high detection sensitivity across diverse terrain types.

**arXiv ID:** 2509.00042
</details>

<details>
<summary><strong>Intelligent Spectrum Management in Satellite Communications</strong> - Rakshitha De Silva, Shiva Raj Pokhrel, Jonathan Kua, Sithamparanathan Kandeepan - [[pdf]](https://arxiv.org/pdf/2509.00286)</summary>

**Abstract:** Satellite Communication (SatCom) networks represent a fundamental pillar in modern global connectivity, facilitating reliable service and extensive coverage across a plethora of applications. The expanding demand for high-bandwidth services and the proliferation of mega satellite constellations highlight the limitations of traditional exclusive satellite spectrum allocation approaches. Cognitive Radio (CR) leading to Cognitive Satellite (CogSat) networks through Dynamic Spectrum Management (DSM), which enables the dynamic adaptability of radio equipment to environmental conditions for optimal performance, presents a promising solution for the emerging spectrum scarcity. In this survey, we explore the adaptation of intelligent DSM methodologies to SatCom, leveraging satellite network integrations. We discuss contributions and hurdles in regulations and standardizations in realizing intelligent DSM in SatCom, and deep dive into DSM techniques, which enable CogSat networks. Furthermore, we extensively evaluate and categorize state-of-the-art Artificial Intelligence (AI)/Machine Learning (ML) methods leveraged for DSM while exploring operational resilience and robustness of such integrations. In addition, performance evaluation metrics critical for adaptive resource management and system optimization in CogSat networks are thoroughly investigated. This survey also identifies open challenges and outlines future research directions in regulatory frameworks, network architectures, and intelligent spectrum management, paving the way for sustainable and scalable SatCom networks for enhanced global connectivity.

**arXiv ID:** 2509.00286
</details>

<details>
<summary><strong>Embodied Spatial Intelligence: from Implicit Scene Modeling to Spatial Reasoning</strong> - Jiading Fang - [[pdf]](https://arxiv.org/pdf/2509.00465)</summary>

**Abstract:** This thesis introduces "Embodied Spatial Intelligence" to address the challenge of creating robots that can perceive and act in the real world based on natural language instructions. To bridge the gap between Large Language Models (LLMs) and physical embodiment, we present contributions on two fronts: scene representation and spatial reasoning. For perception, we develop robust, scalable, and accurate scene representations using implicit neural models, with contributions in self-supervised camera calibration, high-fidelity depth field generation, and large-scale reconstruction. For spatial reasoning, we enhance the spatial capabilities of LLMs by introducing a novel navigation benchmark, a method for grounding language in 3D, and a state-feedback mechanism to improve long-horizon decision-making. This work lays a foundation for robots that can robustly perceive their surroundings and intelligently act upon complex, language-based commands.

**arXiv ID:** 2509.00465
</details>

<details>
<summary><strong>KG-RAG: Enhancing GUI Agent Decision-Making via Knowledge Graph-Driven Retrieval-Augmented Generation</strong> - Ziyi Guan, Jason Chun Lok Li, Zhijian Hou, Pingping Zhang, Donglai Xu, Yuzhi Zhao, Mengyang Wu, Jinpeng Chen, Thanh-Toan Nguyen, Pengfei Xian, Wenao Ma, Shengchao Qin, Graziano Chesi, Ngai Wong - [[pdf]](https://arxiv.org/pdf/2509.00366)</summary>

**Abstract:** Despite recent progress, Graphic User Interface (GUI) agents powered by Large Language Models (LLMs) struggle with complex mobile tasks due to limited app-specific knowledge. While UI Transition Graphs (UTGs) offer structured navigation representations, they are underutilized due to poor extraction and inefficient integration. We introduce KG-RAG, a Knowledge Graph-driven Retrieval-Augmented Generation framework that transforms fragmented UTGs into structured vector databases for efficient real-time retrieval. By leveraging an intent-guided LLM search method, KG-RAG generates actionable navigation paths, enhancing agent decision-making. Experiments across diverse mobile apps show that KG-RAG outperforms existing methods, achieving a 75.8% success rate (8.9% improvement over AutoDroid), 84.6% decision accuracy (8.1% improvement), and reducing average task steps from 4.5 to 4.1. Additionally, we present KG-Android-Bench and KG-Harmony-Bench, two benchmarks tailored to the Chinese mobile ecosystem for future research. Finally, KG-RAG transfers to web/desktop (+40% SR on Weibo-web; +20% on QQ Music-desktop), and a UTG cost ablation shows accuracy saturates at ~4h per complex app, enabling practical deployment trade-offs.

**arXiv ID:** 2509.00366
</details>

<details>
<summary><strong>MobiAgent: A Systematic Framework for Customizable Mobile Agents</strong> - Cheng Zhang, Erhu Feng, Xi Zhao, Yisheng Zhao, Wangbo Gong, Jiahui Sun, Dong Du, Zhichao Hua, Yubin Xia, Haibo Chen - [[pdf]](https://arxiv.org/pdf/2509.00531)</summary>

**Abstract:** With the rapid advancement of Vision-Language Models (VLMs), GUI-based mobile agents have emerged as a key development direction for intelligent mobile systems. However, existing agent models continue to face significant challenges in real-world task execution, particularly in terms of accuracy and efficiency. To address these limitations, we propose MobiAgent, a comprehensive mobile agent system comprising three core components: the MobiMind-series agent models, the AgentRR acceleration framework, and the MobiFlow benchmarking suite. Furthermore, recognizing that the capabilities of current mobile agents are still limited by the availability of high-quality data, we have developed an AI-assisted agile data collection pipeline that significantly reduces the cost of manual annotation. Compared to both general-purpose LLMs and specialized GUI agent models, MobiAgent achieves state-of-the-art performance in real-world mobile scenarios.

**arXiv ID:** 2509.00531
</details>

<details>
<summary><strong>A Comprehensive Survey of Self-Evolving AI Agents: A New Paradigm Bridging Foundation Models and Lifelong Agentic Systems</strong> - Jinyuan Fang, Yanwen Peng, Xi Zhang, Yingxu Wang, Xinhao Yi, Guibin Zhang, Yi Xu, Bin Wu, Siwei Liu, Zihao Li, Zhaochun Ren, Nikos Aletras, Xi Wang, Han Zhou, Zaiqiao Meng - [[pdf]](https://arxiv.org/pdf/2508.07407)</summary>

**Abstract:** Recent advances in large language models have sparked growing interest in AI agents capable of solving complex, real-world tasks. However, most existing agent systems rely on manually crafted configurations that remain static after deployment, limiting their ability to adapt to dynamic and evolving environments. To this end, recent research has explored agent evolution techniques that aim to automatically enhance agent systems based on interaction data and environmental feedback. This emerging direction lays the foundation for self-evolving AI agents, which bridge the static capabilities of foundation models with the continuous adaptability required by lifelong agentic systems. In this survey, we provide a comprehensive review of existing techniques for self-evolving agentic systems. Specifically, we first introduce a unified conceptual framework that abstracts the feedback loop underlying the design of self-evolving agentic systems. The framework highlights four key components: System Inputs, Agent System, Environment, and Optimisers, serving as a foundation for understanding and comparing different strategies. Based on this framework, we systematically review a wide range of self-evolving techniques that target different components of the agent system. We also investigate domain-specific evolution strategies developed for specialised fields such as biomedicine, programming, and finance, where optimisation objectives are tightly coupled with domain constraints. In addition, we provide a dedicated discussion on the evaluation, safety, and ethical considerations for self-evolving agentic systems, which are critical to ensuring their effectiveness and reliability. This survey aims to provide researchers and practitioners with a systematic understanding of self-evolving AI agents, laying the foundation for the development of more adaptive, autonomous, and lifelong agentic systems.

**arXiv ID:** 2508.07407
</details>

<details>
<summary><strong>Can Multi-turn Self-refined Single Agent LMs with Retrieval Solve Hard Coding Problems?</strong> - Md Tanzib Hosain, Md Kishor Morol - [[pdf]](https://arxiv.org/pdf/2509.00629)</summary>

**Abstract:** Among the hardest tasks for humans are those found in competitive programming where problems require sophisticated algorithmic thinking, puzzle solving, and the creation of effective code. As a domain to assess language models (LMs), it has not received enough attention, though. This study presents the ICPC benchmark, which consists of 254 international collegiate programming contest (ICPC) tasks. Each problem includes official analysis, reference code, and sample, high-quality unit, and hidden tests. We are able to develop and evaluate a variety of LM inference techniques for competitive programming with these resources. With zero-shot chain-of-thought prompting, we find that o1 only achieves a 19.1\% pass@1 solve rate. With our best inference technique, which combines multi-turn self-judge with reflection and retrieval over episodic information, raises this to 42.2\%. Furthermore, we conduct a new human-in-the-loop investigation to gain a deeper understanding of the remaining difficulties. Surprisingly, we discover that o1 can solve 17 out of 18 problems that were previously unsolvable by any model or technique with just a few specific instructions. A footstep toward LMs with grounded, imaginative, and algorithmic thinking is provided by our quantitative findings and qualitative research. We open-source our code and data at this https URL.

**arXiv ID:** 2509.00629
</details>

<details>
<summary><strong>JudgeAgent: Dynamically Evaluate LLMs with Agent-as-Interviewer</strong> - Zhichao Shi, Xuhui Jiang, Chengjin Xu, Cangli Yao, Zhenxin Huang, Shengjie Ma, Yinghan Shen, Yuanzhuo Wang - [[pdf]](https://arxiv.org/pdf/2509.02097)</summary>

**Abstract:** Evaluating the capabilities of large language models (LLMs) is an essential step to ensure the successful application of LLMs across various domains. The current evaluation of LLMs is based on a paradigm that involves querying them with predefined question sets and assessing their outputs. This paradigm offers controllable processes and simplicity, but faces challenges such as limited interaction with targets, insufficient difficulty control, and difficulties in verifying the validity of evaluation results, making it hard to precisely determine the knowledge and capability boundaries of target models. To address these challenges, we propose JudgeAgent, a knowledge-target adaptive dynamic evaluation framework based on a new interviewer-style evaluation paradigm. JudgeAgent employs a comprehensive evaluation approach consisting of benchmark grading, interactive extension, and evaluation feedback. It utilizes knowledge-driven data synthesis and target-adaptive difficulty adjustment methods to conduct extended testing, providing accurate and effective evaluation results. We also introduce a novel insight into validating evaluation methods, demonstrating the effectiveness of JudgeAgent and its dynamic evaluation paradigm through extensive experiments.

**arXiv ID:** 2509.02097
</details>

<details>
<summary><strong>Democratizing Agentic AI with Fast Test-Time Scaling on the Edge</strong> - Hao Mark Chen, Zhiwen Mo, Guanxi Lu, Shuang Liang, Lingxiao Ma, Wayne Luk, Hongxiang Fan - [[pdf]](https://arxiv.org/pdf/2509.00195)</summary>

**Abstract:** Deploying agentic AI on edge devices is crucial for privacy and responsiveness, but memory constraints typically relegate these systems to smaller Large Language Models (LLMs) with inferior reasoning capabilities. Test-Time Scaling (TTS) can bridge this reasoning gap by dedicating more compute during inference, but existing methods incur prohibitive overhead on edge hardware. To overcome this, we introduce FlashTTS, a serving system that makes TTS practical for memory-constrained LLM reasoning. FlashTTS introduces three synergistic optimizations: (i) Speculative Beam Extension to mitigate system stragglers from irregular reasoning paths; (ii) Asymmetric Multi-Model Memory Allocation to dynamically balance memory between generation and verification; and (iii) Dynamic Prefix-Aware Scheduling to maximize KV-cache reuse. Built as a plug-and-play library for vLLM, FlashTTS enables edge LLMs on a single consumer GPU (24 GB) to match the accuracy and latency of large cloud models. Our evaluation demonstrates that FlashTTS achieves an average 2.2x higher goodput and reduces latency by 38%-68% compared to a vLLM baseline, paving the way for democratized, high-performance agentic AI on edge devices.

**arXiv ID:** 2509.00195
</details>

<details>
<summary><strong>Safe and Efficient Lane-Changing for Autonomous Vehicles: An Improved Double Quintic Polynomial Approach with Time-to-Collision Evaluation</strong> - Rui Bai, Rui Xu, Teng Rui, Jiale Liu, Qi Wei Oung, Hoi Leong Lee, Zhen Tian, Fujiang Yuan - [[pdf]](https://arxiv.org/pdf/2509.00582)</summary>

**Abstract:** Autonomous driving technology has made significant advancements in recent years, yet challenges remain in ensuring safe and comfortable interactions with human-driven vehicles (HDVs), particularly during lane-changing maneuvers. This paper proposes an improved double quintic polynomial approach for safe and efficient lane-changing in mixed traffic environments. The proposed method integrates a time-to-collision (TTC) based evaluation mechanism directly into the trajectory optimization process, ensuring that the ego vehicle proactively maintains a safe gap from surrounding HDVs throughout the maneuver. The framework comprises state estimation for both the autonomous vehicle (AV) and HDVs, trajectory generation using double quintic polynomials, real-time TTC computation, and adaptive trajectory evaluation. To the best of our knowledge, this is the first work to embed an analytic TTC penalty directly into the closed-form double-quintic polynomial solver, enabling real-time safety-aware trajectory generation without post-hoc validation. Extensive simulations conducted under diverse traffic scenarios demonstrate the safety, efficiency, and comfort of the proposed approach compared to conventional methods such as quintic polynomials, Bezier curves, and B-splines. The results highlight that the improved method not only avoids collisions but also ensures smooth transitions and adaptive decision-making in dynamic environments. This work bridges the gap between model-based and adaptive trajectory planning approaches, offering a stable solution for real-world autonomous driving applications.

**arXiv ID:** 2509.00582
</details>

<details>
<summary><strong>A Risk-aware Spatial-temporal Trajectory Planning Framework for Autonomous Vehicles Using QP-MPC and Dynamic Hazard Fields</strong> - Zhen Tian, Zhihao Lin, Dezong Zhao, Christos Anagnostopoulos, Qiyuan Wang, Wenjing Zhao, Xiaodan Wang, Chongfeng Wei - [[pdf]](https://arxiv.org/pdf/2509.00643)</summary>

**Abstract:** Trajectory planning is a critical component in ensuring the safety, stability, and efficiency of autonomous vehicles. While existing trajectory planning methods have achieved progress, they often suffer from high computational costs, unstable performance in dynamic environments, and limited validation across diverse scenarios. To overcome these challenges, we propose an enhanced QP-MPC-based framework that incorporates three key innovations: (i) a novel cost function designed with a dynamic hazard field, which explicitly balances safety, efficiency, and comfort; (ii) seamless integration of this cost function into the QP-MPC formulation, enabling direct optimization of desired driving behaviors; and (iii) extensive validation of the proposed framework across complex tasks. The spatial safe planning is guided by a dynamic hazard field (DHF) for risk assessment, while temporal safe planning is based on a space-time graph. Besides, the quintic polynomial sampling and sub-reward of comforts are used to ensure comforts during lane-changing. The sub-reward of efficiency is used to maintain driving efficiency. Finally, the proposed DHF-enhanced objective function integrates multiple objectives, providing a proper optimization tasks for QP-MPC. Extensive simulations demonstrate that the proposed framework outperforms benchmark optimization methods in terms of efficiency, stability, and comfort across a variety of scenarios likes lane-changing, overtaking, and crossing intersections.

**arXiv ID:** 2509.00643
</details>

<details>
<summary><strong>Toward a Holistic Multi-Criteria Trajectory Evaluation Framework for Autonomous Driving in Mixed Traffic Environment</strong> - Nouhed Naidja, Stéphane Font, Marc Revilloud, Guillaume Sandou - [[pdf]](https://arxiv.org/pdf/2509.01291)</summary>

**Abstract:** This paper presents a unified framework for the evaluation and optimization of autonomous vehicle trajectories, integrating formal safety, comfort, and efficiency criteria. An innovative geometric indicator, based on the analysis of safety zones using adaptive ellipses, is used to accurately quantify collision risks. Our method applies the Shoelace formula to compute the intersection area in the case of misaligned and time-varying configurations. Comfort is modeled using indicators centered on longitudinal and lateral jerk, while efficiency is assessed by overall travel time. These criteria are aggregated into a comprehensive objective function solved using a PSO based algorithm. The approach was successfully validated under real traffic conditions via experiments conducted in an urban intersection involving an autonomous vehicle interacting with a human-operated vehicle, and in simulation using data recorded from human driving in real traffic.

**arXiv ID:** 2509.01291
</details>

<details>
<summary><strong>A Survey on Vision-Language-Action Models for Embodied AI</strong> - Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, Irwin King - [[pdf]](https://arxiv.org/pdf/2405.14093)</summary>

**Abstract:** Embodied AI is widely recognized as a key element of artificial general intelligence because it involves controlling embodied agents to perform tasks in the physical world. Building on the success of large language models and vision-language models, a new category of multimodal models -- referred to as vision-language-action models (VLAs) -- has emerged to address language-conditioned robotic tasks in embodied AI by leveraging their distinct ability to generate actions. In recent years, a myriad of VLAs have been developed, making it imperative to capture the rapidly evolving landscape through a comprehensive survey. To this end, we present the first survey on VLAs for embodied AI. This work provides a detailed taxonomy of VLAs, organized into three major lines of research. The first line focuses on individual components of VLAs. The second line is dedicated to developing control policies adept at predicting low-level actions. The third line comprises high-level task planners capable of decomposing long-horizon tasks into a sequence of subtasks, thereby guiding VLAs to follow more general user instructions. Furthermore, we provide an extensive summary of relevant resources, including datasets, simulators, and benchmarks. Finally, we discuss the challenges faced by VLAs and outline promising future directions in embodied AI. We have created a project associated with this survey, which is available at this https URL.

**arXiv ID:** 2405.14093
</details>

<details>
<summary><strong>NetRoller: Interfacing General and Specialized Models for End-to-End Autonomous Driving</strong> - Ren Xin, Hongji Liu, Xiaodong Mei, Wenru Liu, Maosheng Ye, Zhili Chen, Jun Ma - [[pdf]](https://arxiv.org/pdf/2506.14589)</summary>

**Abstract:** Integrating General Models (GMs) such as Large Language Models (LLMs), with Specialized Models (SMs) in autonomous driving tasks presents a promising approach to mitigating challenges in data diversity and model capacity of existing specialized driving models. However, this integration leads to problems of asynchronous systems, which arise from the distinct characteristics inherent in GMs and SMs. To tackle this challenge, we propose NetRoller, an adapter that incorporates a set of novel mechanisms to facilitate the seamless integration of GMs and specialized driving models. Specifically, our mechanisms for interfacing the asynchronous GMs and SMs are organized into three key stages. NetRoller first harvests semantically rich and computationally efficient representations from the reasoning processes of LLMs using an early stopping mechanism, which preserves critical insights on driving context while maintaining low overhead. It then applies learnable query embeddings, nonsensical embeddings, and positional layer embeddings to facilitate robust and efficient cross-modality translation. At last, it employs computationally efficient Query Shift and Feature Shift mechanisms to enhance the performance of SMs through few-epoch fine-tuning. Based on the mechanisms formalized in these three stages, NetRoller enables specialized driving models to operate at their native frequencies while maintaining situational awareness of the GM. Experiments conducted on the nuScenes dataset demonstrate that integrating GM through NetRoller significantly improves human similarity and safety in planning tasks, and it also achieves noticeable precision improvements in detection and mapping tasks for end-to-end autonomous driving. The code and models are available at this https URL .

**arXiv ID:** 2506.14589
</details>

<details>
<summary><strong>EmbodiedOneVision: Interleaved Vision-Text-Action Pretraining for General Robot Control</strong> - Delin Qu, Haoming Song, Qizhi Chen, Zhaoqing Chen, Xianqiang Gao, Xinyi Ye, Qi Lv, Modi Shi, Guanghui Ren, Cheng Ruan, Maoqing Yao, Haoran Yang, Jiacheng Bao, Bin Zhao, Dong Wang - [[pdf]](https://arxiv.org/pdf/2508.21112)</summary>

**Abstract:** The human ability to seamlessly perform multimodal reasoning and physical interaction in the open world is a core goal for general-purpose embodied intelligent systems. Recent vision-language-action (VLA) models, which are co-trained on large-scale robot and visual-text data, have demonstrated notable progress in general robot control. However, they still fail to achieve human-level flexibility in interleaved reasoning and interaction. In this work, introduce EO-Robotics, consists of EO-1 model and EO-Data1.5M dataset. EO-1 is a unified embodied foundation model that achieves superior performance in multimodal embodied reasoning and robot control through interleaved vision-text-action pre-training. The development of EO-1 is based on two key pillars: (i) a unified architecture that processes multimodal inputs indiscriminately (image, text, video, and action), and (ii) a massive, high-quality multimodal embodied reasoning dataset, EO-Data1.5M, which contains over 1.5 million samples with emphasis on interleaved vision-text-action comprehension. EO-1 is trained through synergies between auto-regressive decoding and flow matching denoising on EO-Data1.5M, enabling seamless robot action generation and multimodal embodied reasoning. Extensive experiments demonstrate the effectiveness of interleaved vision-text-action learning for open-world understanding and generalization, validated through a variety of long-horizon, dexterous manipulation tasks across multiple embodiments. This paper details the architecture of EO-1, the data construction strategy of EO-Data1.5M, and the training methodology, offering valuable insights for developing advanced embodied foundation models.

**arXiv ID:** 2508.21112
</details>

</details>

<details open>
<summary><h2>LLM Agents (2 papers)</h2></summary>

<details>
<summary><strong>Talk Less, Call Right: Enhancing Role-Play LLM Agents with Automatic Prompt Optimization and Role Prompting</strong> - Saksorn Ruangtanusak, Pittawat Taveekitworachai, Kunat Pipatanakul - [[pdf]](https://arxiv.org/pdf/2509.00482)</summary>

**Abstract:** This report investigates approaches for prompting a tool-augmented large language model (LLM) to act as a role-playing dialogue agent in the API track of the Commonsense Persona-grounded Dialogue Challenge (CPDC) 2025. In this setting, dialogue agents often produce overly long in-character responses (over-speaking) while failing to use tools effectively according to the persona (under-acting), such as generating function calls that do not exist or making unnecessary tool calls before answering. We explore four prompting approaches to address these issues: 1) basic role prompting, 2) human-crafted role prompting, 3) automatic prompt optimization (APO), and 4) rule-based role prompting. The rule-based role prompting (RRP) approach achieved the best performance through two novel techniques--character-card/scene-contract design and strict enforcement of function calling--which led to an overall score of 0.571, improving on the zero-shot baseline score of 0.519. These findings demonstrate that RRP design can substantially improve the effectiveness and reliability of role-playing dialogue agents compared with more elaborate methods such as APO. To support future efforts in developing persona prompts, we are open-sourcing all of our best-performing prompts and the APO tool. Source code is available at this https URL.

**arXiv ID:** 2509.00482
</details>

<details>
<summary><strong>ORCA: ORchestrating Causal Agent</strong> - Joanie Hayoun Chung, Chaemyung Lim, Sumin Lee, Songseong Kim, Sungbin Lim - [[pdf]](https://arxiv.org/pdf/2508.21304)</summary>

**Abstract:** Causal inference is essential for decision-making science while the complexity of the data analysis workflow, ranging from data wrangling to causal analysis, increases substantially as the scale of data grows in complicated business environments. Especially, the execution of the workflow in relational databases by non-experts can result in repetitive bottlenecks which impede timely and responsible business insights. To address this challenge, we propose ORCA (Orchestrating Causal Agent), an LLM agentic system that can automate routine workflows in RDBMS while preserving expert oversight via human-AI interactions. ORCA orchestrates the full data analysis pipeline: interpreting natural language queries, navigating tables from DB servers, generating proper SQL codes, preprocessing data, and configuring modeling processes using causal inference libraries. Domain experts still can control the automation through iterative interactions with ORCA, enabling robust data-driven decision making with less technical expertise in statistical computing. Empirical evaluations on benchmark and synthetic e-commerce datasets demonstrate competitive performance of ORCA in table understanding, query generation, and cause-effect estimation -- achieving over $7\times$ improvement in estimating average treatment compared to GPT-4o mini.

**arXiv ID:** 2508.21304
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (32 papers)</h2></summary>

<details>
<summary><strong>Adaptive Monitoring and Real-World Evaluation of Agentic AI Systems</strong> - Manish Shukla - [[pdf]](https://arxiv.org/pdf/2509.00115)</summary>

**Abstract:** Agentic artificial intelligence (AI) -- multi-agent systems that combine large language models with external tools and autonomous planning -- are rapidly transitioning from research laboratories into high-stakes domains. Our earlier "Basic" paper introduced a five-axis framework and proposed preliminary metrics such as goal drift and harm reduction but did not provide an algorithmic instantiation or empirical evidence. This "Advanced" sequel fills that gap. First, we revisit recent benchmarks and industrial deployments to show that technical metrics still dominate evaluations: a systematic review of 84 papers from 2023--2025 found that 83% report capability metrics while only 30% consider human-centred or economic axes [2]. Second, we formalise an Adaptive Multi-Dimensional Monitoring (AMDM) algorithm that normalises heterogeneous metrics, applies per-axis exponentially weighted moving-average thresholds and performs joint anomaly detection via the Mahalanobis distance. Third, we conduct simulations and real-world experiments. AMDM cuts anomaly-detection latency from 12.3 s to 5.6 s on simulated goal drift and reduces false-positive rates from 4.5% to 0.9% compared with static thresholds. We present a comparison table and ROC/PR curves, and we reanalyse case studies to surface missing metrics. Code, data and a reproducibility checklist accompany this paper to facilitate replication.

**arXiv ID:** 2509.00115
</details>

<details>
<summary><strong>Multi-Agent Data Visualization and Narrative Generation</strong> - Anton Wolter, Georgios Vidalakis, Michael Yu, Ankit Grover, Vaishali Dhanoa - [[pdf]](https://arxiv.org/pdf/2509.00481)</summary>

**Abstract:** Recent advancements in the field of AI agents have impacted the way we work, enabling greater automation and collaboration between humans and agents. In the data visualization field, multi-agent systems can be useful for employing agents throughout the entire data-to-communication pipeline. We present a lightweight multi-agent system that automates the data analysis workflow, from data exploration to generating coherent visual narratives for insight communication. Our approach combines a hybrid multi-agent architecture with deterministic components, strategically externalizing critical logic from LLMs to improve transparency and reliability. The system delivers granular, modular outputs that enable surgical modifications without full regeneration, supporting sustainable human-AI collaboration. We evaluated our system across 4 diverse datasets, demonstrating strong generalizability, narrative quality, and computational efficiency with minimal dependencies.

**arXiv ID:** 2509.00481
</details>

<details>
<summary><strong>On Verifiable Legal Reasoning: A Multi-Agent Framework with Formalized Knowledge Representations</strong> - Albert Sadowski, Jarosław A. Chudziak - [[pdf]](https://arxiv.org/pdf/2509.00710)</summary>

**Abstract:** Legal reasoning requires both precise interpretation of statutory language and consistent application of complex rules, presenting significant challenges for AI systems. This paper introduces a modular multi-agent framework that decomposes legal reasoning into distinct knowledge acquisition and application stages. In the first stage, specialized agents extract legal concepts and formalize rules to create verifiable intermediate representations of statutes. The second stage applies this knowledge to specific cases through three steps: analyzing queries to map case facts onto the ontology schema, performing symbolic inference to derive logically entailed conclusions, and generating final answers using a programmatic implementation that operationalizes the ontological knowledge. This bridging of natural language understanding with symbolic reasoning provides explicit and verifiable inspection points, significantly enhancing transparency compared to end-to-end approaches. Evaluation on statutory tax calculation tasks demonstrates substantial improvements, with foundational models achieving 76.4\% accuracy compared to 18.8\% baseline performance, effectively narrowing the performance gap between reasoning and foundational models. These findings suggest that modular architectures with formalized knowledge representations can make sophisticated legal reasoning more accessible through computationally efficient models while enhancing consistency and explainability in AI legal reasoning, establishing a foundation for future research into more transparent, trustworthy, and effective AI systems for legal domain.

**arXiv ID:** 2509.00710
</details>

<details>
<summary><strong>L-MARS -- Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search</strong> - Ziqi Wang, Boqin Yuan - [[pdf]](https://arxiv.org/pdf/2509.00761)</summary>

**Abstract:** We present L-MARS (Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search), a system that reduces hallucination and uncertainty in legal question answering through coordinated multi-agent reasoning and retrieval. Unlike single-pass retrieval-augmented generation (RAG), L-MARS decomposes queries into subproblems, issues targeted searches across heterogeneous sources (Serper web, local RAG, CourtListener case law), and employs a Judge Agent to verify sufficiency, jurisdiction, and temporal validity before answer synthesis. This iterative reasoning-search-verification loop maintains coherence, filters noisy evidence, and grounds answers in authoritative law. We evaluated L-MARS on LegalSearchQA, a new benchmark of 200 up-to-date multiple choice legal questions in 2025. Results show that L-MARS substantially improves factual accuracy, reduces uncertainty, and achieves higher preference scores from both human experts and LLM-based judges. Our work demonstrates that multi-agent reasoning with agentic search offers a scalable and reproducible blueprint for deploying LLMs in high-stakes domains requiring precise legal retrieval and deliberation.

**arXiv ID:** 2509.00761
</details>

<details>
<summary><strong>Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Movable Obstacles</strong> - Bo Fu, Zhe Chen, Rahul Chandan, Alex Barbosa, Michael Caldara, Joey Durham, Federico Pecora - [[pdf]](https://arxiv.org/pdf/2509.01022)</summary>

**Abstract:** We introduce the Block Rearrangement Problem (BRaP), a challenging component of large warehouse management which involves rearranging storage blocks within dense grids to achieve a target state. We formally define the BRaP as a graph search problem. Building on intuitions from sliding puzzle problems, we propose five search-based solution algorithms, leveraging joint configuration space search, classical planning, multi-agent pathfinding, and expert heuristics. We evaluate the five approaches empirically for plan quality and scalability. Despite the exponential relation between search space size and block number, our methods demonstrate efficiency in creating rearrangement plans for deeply buried blocks in up to 80x80 grids.

**arXiv ID:** 2509.01022
</details>

<details>
<summary><strong>Question-to-Knowledge: Multi-Agent Generation of Inspectable Facts for Product Mapping</strong> - Wonduk Seo, Taesub Shin, Hyunjin An, Dokyun Kim, Seunghyun Lee - [[pdf]](https://arxiv.org/pdf/2509.01182)</summary>

**Abstract:** Identifying whether two product listings refer to the same Stock Keeping Unit (SKU) is a persistent challenge in ecommerce, especially when explicit identifiers are missing and product names vary widely across platforms. Rule based heuristics and keyword similarity often misclassify products by overlooking subtle distinctions in brand, specification, or bundle configuration. To overcome these limitations, we propose Question to Knowledge (Q2K), a multi agent framework that leverages Large Language Models (LLMs) for reliable SKU mapping. Q2K integrates: (1) a Reasoning Agent that generates targeted disambiguation questions, (2) a Knowledge Agent that resolves them via focused web searches, and (3) a Deduplication Agent that reuses validated reasoning traces to reduce redundancy and ensure consistency. A human in the loop mechanism further refines uncertain cases. Experiments on real world consumer goods datasets show that Q2K surpasses strong baselines, achieving higher accuracy and robustness in difficult scenarios such as bundle identification and brand origin disambiguation. By reusing retrieved reasoning instead of issuing repeated searches, Q2K balances accuracy with efficiency, offering a scalable and interpretable solution for product integration.

**arXiv ID:** 2509.01182
</details>

<details>
<summary><strong>Towards Open-World Retrieval-Augmented Generation on Knowledge Graph: A Multi-Agent Collaboration Framework</strong> - Jiasheng Xu, Mingda Li, Yongqiang Tang, Peijie Wang, Wensheng Zhang - [[pdf]](https://arxiv.org/pdf/2509.01238)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated strong capabilities in language understanding and reasoning. However, their dependence on static training corpora makes them prone to factual errors and knowledge gaps. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge sources, especially structured Knowledge Graphs (KGs), which provide explicit semantics and efficient retrieval. Existing KG-based RAG approaches, however, generally assume that anchor entities are accessible to initiate graph traversal, which limits their robustness in open world settings where accurate linking between the query and the entity is unreliable. To overcome this limitation, we propose AnchorRAG, a novel multi-agent collaboration framework for open-world RAG without the predefined anchor entities. Specifically, a predictor agent dynamically identifies candidate anchor entities by aligning user query terms with KG nodes and initializes independent retriever agents to conduct parallel multi-hop explorations from each candidate. Then a supervisor agent formulates the iterative retrieval strategy for these retriever agents and synthesizes the resulting knowledge paths to generate the final answer. This multi-agent collaboration framework improves retrieval robustness and mitigates the impact of ambiguous or erroneous anchors. Extensive experiments on four public benchmarks demonstrate that AnchorRAG significantly outperforms existing baselines and establishes new state-of-the-art results on the real-world question answering tasks.

**arXiv ID:** 2509.01238
</details>

<details>
<summary><strong>Towards Agentic OS: An LLM Agent Framework for Linux Schedulers</strong> - Yusheng Zheng, Yanpeng Hu, Wei Zhang, Andi Quinn - [[pdf]](https://arxiv.org/pdf/2509.01245)</summary>

**Abstract:** Operating system schedulers suffer from a fundamental semantic gap, where kernel policies fail to understand application-specific needs, leading to suboptimal performance. We introduce SchedCP, the first framework that enables fully autonomous Large Language Model (LLM) agents to safely and efficiently optimize Linux schedulers without human involvement. Our core insight is that the challenge is not merely to apply a better LLM, but to architect a decoupled control plane that separates the AI's role of semantic reasoning ("what to optimize") from the system's role of execution ("how to observe and act"). Implemented as Model Context Protocol(MCP) server, SchedCP provides a stable interface with three key services: a Workload Analysis Engine, an evolving Scheduler Policy Repository, and an Execution Verifier that validates all AI-generated code and configure before deployment with static and dynamic analysis.
We demonstrate this architecture's power with sched-agent, a multi-agent system that autonomously analyzes workloads, synthesizes custom eBPF scheduling policies, and deploys them via the sched\_ext infrastructure. Our evaluation shows that SchedCP achieves up to an 1.79x performance improvement, and a 13x cost reduction compared to naive agentic approaches, all while maintaining high success rate. By bridging the semantic gap, SchedCP democratizes expert-level system optimization and represents a step towards creating truly self-optimizing, application-aware operating systems. The code is open-sourced in this https URL

**arXiv ID:** 2509.01245
</details>

<details>
<summary><strong>AppCopilot: Toward General, Accurate, Long-Horizon, and Efficient Mobile Agent</strong> - Jingru Fan, Yufan Dang, Jingyao Wu, Huatao Li, Runde Yang, Xiyuan Yang, Yuheng Wang, Zhong Zhang, Yaxi Lu, Yankai Lin, Zhiyuan Liu, Dahai Li, Chen Qian - [[pdf]](https://arxiv.org/pdf/2509.02444)</summary>

**Abstract:** With the raid evolution of large language models and multimodal foundation models, the mobile-agent landscape has proliferated without converging on the fundamental challenges. This paper identifies four core problems that must be solved for mobile agents to deliver practical, scalable impact: (1) generalization across tasks, modalities, apps, and devices; (2) accuracy, specifically precise on-screen interaction and click targeting; (3) long-horizon capability for sustained, multi-step goals; and (4) efficiency, specifically high-performance runtime on resource-constrained devices. We present AppCopilot, a multimodal, multi-agent, general-purpose on-device assistant that operates across applications and constitutes a full-stack, closed-loop system from data to deployment. AppCopilot operationalizes this position through an end-to-end autonomous pipeline spanning data collection, training, deployment, high-quality and efficient inference, and mobile application development. At the model layer, it integrates multimodal foundation models with robust Chinese-English support. At the reasoning and control layer, it combines chain-of-thought reasoning, hierarchical task planning and decomposition, and multi-agent collaboration. At the execution layer, it enables user personalization and experiential adaptation, voice interaction, function calling, cross-app and cross-device orchestration, and comprehensive mobile app support. The system design incorporates profiling-driven optimization for latency, memory, and energy across heterogeneous hardware. Empirically, AppCopilot achieves significant improvements along all four dimensions: stronger generalization, higher-precision on-screen actions, more reliable long-horizon task completion, and faster, more resource-efficient runtime.

**arXiv ID:** 2509.02444
</details>

<details>
<summary><strong>GridMind: LLMs-Powered Agents for Power System Analysis and Operations</strong> - Hongwei Jin, Kibaek Kim, Jonghwan Kwon - [[pdf]](https://arxiv.org/pdf/2509.02494)</summary>

**Abstract:** The complexity of traditional power system analysis workflows presents significant barriers to efficient decision-making in modern electric grids. This paper presents GridMind, a multi-agent AI system that integrates Large Language Models (LLMs) with deterministic engineering solvers to enable conversational scientific computing for power system analysis. The system employs specialized agents coordinating AC Optimal Power Flow and N-1 contingency analysis through natural language interfaces while maintaining numerical precision via function calls. GridMind addresses workflow integration, knowledge accessibility, context preservation, and expert decision-support augmentation. Experimental evaluation on IEEE test cases demonstrates that the proposed agentic framework consistently delivers correct solutions across all tested language models, with smaller LLMs achieving comparable analytical accuracy with reduced computational latency. This work establishes agentic AI as a viable paradigm for scientific computing, demonstrating how conversational interfaces can enhance accessibility while preserving numerical rigor essential for critical engineering applications.

**arXiv ID:** 2509.02494
</details>

<details>
<summary><strong>CoComposer: LLM Multi-agent Collaborative Music Composition</strong> - Peiwen Xing, Aske Plaat, Niki van Stein - [[pdf]](https://arxiv.org/pdf/2509.00132)</summary>

**Abstract:** Existing AI Music composition tools are limited in generation duration, musical quality, and controllability. We introduce CoComposer, a multi-agent system that consists of five collaborating agents, each with a task based on the traditional music composition workflow. Using the AudioBox-Aesthetics system, we experimentally evaluate CoComposer on four compositional criteria. We test with three LLMs (GPT-4o, DeepSeek-V3-0324, Gemini-2.5-Flash), and find (1) that CoComposer outperforms existing multi-agent LLM-based systems in music quality, and (2) compared to a single-agent system, in production complexity. Compared to non- LLM MusicLM, CoComposer has better interpretability and editability, although MusicLM still produces better music.

**arXiv ID:** 2509.00132
</details>

<details>
<summary><strong>Q-Learning--Driven Adaptive Rewiring for Cooperative Control in Heterogeneous Networks</strong> - Yi-Ning Weng, Hsuan-Wei Lee - [[pdf]](https://arxiv.org/pdf/2509.01057)</summary>

**Abstract:** Cooperation emergence in multi-agent systems represents a fundamental statistical physics problem where microscopic learning rules drive macroscopic collective behavior transitions. We propose a Q-learning-based variant of adaptive rewiring that builds on mechanisms studied in the literature. This method combines temporal difference learning with network restructuring so that agents can optimize strategies and social connections based on interaction histories. Through neighbor-specific Q-learning, agents develop sophisticated partnership management strategies that enable cooperator cluster formation, creating spatial separation between cooperative and defective regions. Using power-law networks that reflect real-world heterogeneous connectivity patterns, we evaluate emergent behaviors under varying rewiring constraint levels, revealing distinct cooperation patterns across parameter space rather than sharp thermodynamic transitions. Our systematic analysis identifies three behavioral regimes: a permissive regime (low constraints) enabling rapid cooperative cluster formation, an intermediate regime with sensitive dependence on dilemma strength, and a patient regime (high constraints) where strategic accumulation gradually optimizes network structure. Simulation results show that while moderate constraints create transition-like zones that suppress cooperation, fully adaptive rewiring enhances cooperation levels through systematic exploration of favorable network configurations. Quantitative analysis reveals that increased rewiring frequency drives large-scale cluster formation with power-law size distributions. Our results establish a new paradigm for understanding intelligence-driven cooperation pattern formation in complex adaptive systems, revealing how machine learning serves as an alternative driving force for spontaneous organization in multi-agent networks.

**arXiv ID:** 2509.01057
</details>

<details>
<summary><strong>Web Fraud Attacks Against LLM-Driven Multi-Agent Systems</strong> - Dezhang Kong, Hujin Peng, Yilun Zhang, Lele Zhao, Zhenhua Xu, Shi Lin, Changting Lin, Meng Han - [[pdf]](https://arxiv.org/pdf/2509.01211)</summary>

**Abstract:** With the proliferation of applications built upon LLM-driven multi-agent systems (MAS), the security of Web links has become a critical concern in ensuring system reliability. Once an agent is induced to visit a malicious website, attackers can use it as a springboard to conduct diverse subsequent attacks, which will drastically expand the attack surface. In this paper, we propose Web Fraud Attacks, a novel type of attack aiming at inducing MAS to visit malicious websites. We design 11 representative attack variants that encompass domain name tampering (homoglyph deception, character substitution, etc.), link structure camouflage (sub-directory nesting, sub-domain grafting, parameter obfuscation, etc.), and other deceptive techniques tailored to exploit MAS's vulnerabilities in link validation. Through extensive experiments on these crafted attack vectors, we demonstrate that Web fraud attacks not only exhibit significant destructive potential across different MAS architectures but also possess a distinct advantage in evasion: they circumvent the need for complex input formats such as jailbreaking, which inherently carry higher exposure risks. These results underscore the importance of addressing Web fraud attacks in LLM-driven MAS, as their stealthiness and destructiveness pose non-negligible threats to system security and user safety.

**arXiv ID:** 2509.01211
</details>

<details>
<summary><strong>Nash Q-Network for Multi-Agent Cybersecurity Simulation</strong> - Qintong Xie, Edward Koh, Xavier Cadet, Peter Chin - [[pdf]](https://arxiv.org/pdf/2509.00678)</summary>

**Abstract:** Cybersecurity defense involves interactions between adversarial parties (namely defenders and hackers), making multi-agent reinforcement learning (MARL) an ideal approach for modeling and learning strategies for these scenarios. This paper addresses one of the key challenges to MARL, the complexity of simultaneous training of agents in nontrivial environments, and presents a novel policy-based Nash Q-learning to directly converge onto a steady equilibrium. We demonstrate the successful implementation of this algorithm in a notable complex cyber defense simulation treated as a two-player zero-sum Markov game setting. We propose the Nash Q-Network, which aims to learn Nash-optimal strategies that translate to robust defenses in cybersecurity settings. Our approach incorporates aspects of proximal policy optimization (PPO), deep Q-network (DQN), and the Nash-Q algorithm, addressing common challenges like non-stationarity and instability in multi-agent learning. The training process employs distributed data collection and carefully designed neural architectures for both agents and critics.

**arXiv ID:** 2509.00678
</details>

<details>
<summary><strong>Controller synthesis method for multi-agent system based on temporal logic specification</strong> - Ruohan Huang, Zining Cao - [[pdf]](https://arxiv.org/pdf/2509.00870)</summary>

**Abstract:** Controller synthesis is a theoretical approach to the systematic design of discrete event systems. It constructs a controller to provide feedback and control to the system, ensuring it meets specified control specifications. Traditional controller synthesis methods often use formal languages to describe control specifications and are mainly oriented towards single-agent and non-probabilistic systems. With the increasing complexity of systems, the control requirements that need to be satisfied also become more complex. Based on this, this paper proposes a controller synthesis method for semi-cooperative semi-competitive multi-agent probabilistic discrete event systems to solve the controller synthesis problem based on temporal logic specifications. The controller can ensure the satisfaction of specifications to a certain extent. The specification is given in the form of a linear temporal logic formula. This paper designs a controller synthesis algorithm that combines probabilistic model checking. Finally, the effectiveness of this method is verified through a case study.

**arXiv ID:** 2509.00870
</details>

<details>
<summary><strong>Contemporary Agent Technology: LLM-Driven Advancements vs Classic Multi-Agent Systems</strong> - Costin Bădică, Amelia Bădică, Maria Ganzha, Mirjana Ivanović, Marcin Paprzycki, Dan Selişteanu, Zofia Wrona - [[pdf]](https://arxiv.org/pdf/2509.02515)</summary>

**Abstract:** This contribution provides our comprehensive reflection on the contemporary agent technology, with a particular focus on the advancements driven by Large Language Models (LLM) vs classic Multi-Agent Systems (MAS). It delves into the models, approaches, and characteristics that define these new systems. The paper emphasizes the critical analysis of how the recent developments relate to the foundational MAS, as articulated in the core academic literature. Finally, it identifies key challenges and promising future directions in this rapidly evolving domain.

**arXiv ID:** 2509.02515
</details>

<details>
<summary><strong>Adaptation of Parameters in Heterogeneous Multi-agent Systems</strong> - Hyungbo Shim, Jin Gyu Lee, B. D. O. Anderson - [[pdf]](https://arxiv.org/pdf/2509.00801)</summary>

**Abstract:** This paper proposes an adaptation mechanism for heterogeneous multi-agent systems to align the agents' internal parameters, based on enforced consensus through strong couplings. Unlike homogeneous systems, where exact consensus is attainable, the heterogeneity in node dynamics precludes perfect synchronization. Nonetheless, previous work has demonstrated that strong coupling can induce approximate consensus, whereby the agents exhibit emergent collective behavior governed by the so-called blended dynamics. Building on this observation, we introduce an adaptation law that gradually aligns the internal parameters of agents without requiring direct parameter communication. The proposed method reuses the same coupling signal employed for state synchronization, which may result in a biologically or sociologically plausible adaptation process. Under a persistent excitation condition, we prove that the linearly parametrized vector fields of the agents converge to each other, thereby making the dynamics asymptotically homogeneous, and leading to exact consensus of the state variables.

**arXiv ID:** 2509.00801
</details>

<details>
<summary><strong>VariAntNet: Learning Decentralized Control of Multi-Agent Systems</strong> - Yigal Koifman, Erez Koifman, Eran Iceland, Ariel Barel, Alfred M. Bruckstein - [[pdf]](https://arxiv.org/pdf/2509.02271)</summary>

**Abstract:** A simple multi-agent system can be effectively utilized in disaster response applications, such as firefighting. Such a swarm is required to operate in complex environments with limited local sensing and no reliable inter-agent communication or centralized control. These simple robotic agents, also known as Ant Robots, are defined as anonymous agents that possess limited sensing capabilities, lack a shared coordinate system, and do not communicate explicitly with one another. A key challenge for simple swarms lies in maintaining cohesion and avoiding fragmentation despite limited-range sensing. Recent advances in machine learning offer effective solutions to some of the classical decentralized control challenges. We propose VariAntNet, a deep learning-based decentralized control model designed to facilitate agent swarming and collaborative task execution. VariAntNet includes geometric features extraction from unordered, variable-sized local observations. It incorporates a neural network architecture trained with a novel, differentiable, multi-objective, mathematically justified loss function that promotes swarm cohesiveness by utilizing the properties of the visibility graph Laplacian matrix. VariAntNet is demonstrated on the fundamental multi-agent gathering task, where agents with bearing-only and limited-range sensing must gather at some location. VariAntNet significantly outperforms an existing analytical solution, achieving more than double the convergence rate while maintaining high swarm connectivity across varying swarm sizes. While the analytical solution guarantees cohesion, it is often too slow in practice. In time-critical scenarios, such as emergency response operations where lives are at risk, slower analytical methods are impractical and justify the loss of some agents within the swarm. This paper presents and analyzes this trade-off in detail.

**arXiv ID:** 2509.02271
</details>

<details>
<summary><strong>Fairness Aware Reinforcement Learning via Proximal Policy Optimization</strong> - Gabriele La Malfa, Jie M. Zhang, Michael Luck, Elizabeth Black - [[pdf]](https://arxiv.org/pdf/2502.03953)</summary>

**Abstract:** Fairness in multi-agent systems (MAS) focuses on equitable reward distribution among agents in scenarios involving sensitive attributes such as race, gender, or socioeconomic status. This paper introduces fairness in Proximal Policy Optimization (PPO) with a penalty term derived from a fairness definition such as demographic parity, counterfactual fairness, or conditional statistical parity. The proposed method, which we call Fair-PPO, balances reward maximisation with fairness by integrating two penalty components: a retrospective component that minimises disparities in past outcomes and a prospective component that ensures fairness in future decision-making. We evaluate our approach in two games: the Allelopathic Harvest, a cooperative and competitive MAS focused on resource collection, where some agents possess a sensitive attribute, and HospitalSim, a hospital simulation, in which agents coordinate the operations of hospital patients with different mobility and priority needs. Experiments show that Fair-PPO achieves fairer policies than PPO across the fairness metrics and, through the retrospective and prospective penalty components, reveals a wide spectrum of strategies to improve fairness; at the same time, its performance pairs with that of state-of-the-art fair reinforcement-learning algorithms. Fairness comes at the cost of reduced efficiency, but does not compromise equality among the overall population (Gini index). These findings underscore the potential of Fair-PPO to address fairness challenges in MAS.

**arXiv ID:** 2502.03953
</details>

<details>
<summary><strong>RALLY: Role-Adaptive LLM-Driven Yoked Navigation for Agentic UAV Swarms</strong> - Ziyao Wang, Rongpeng Li, Sizhao Li, Yuming Xiang, Haiping Wang, Zhifeng Zhao, Honggang Zhang - [[pdf]](https://arxiv.org/pdf/2507.01378)</summary>

**Abstract:** Intelligent control of Unmanned Aerial Vehicles (UAVs) swarms has emerged as a critical research focus, and it typically requires the swarm to navigate effectively while avoiding obstacles and achieving continuous coverage over multiple mission targets. Although traditional Multi-Agent Reinforcement Learning (MARL) approaches offer dynamic adaptability, they are hindered by the semantic gap in numerical communication and the rigidity of homogeneous role structures, resulting in poor generalization and limited task scalability. Recent advances in Large Language Model (LLM)-based control frameworks demonstrate strong semantic reasoning capabilities by leveraging extensive prior knowledge. However, due to the lack of online learning and over-reliance on static priors, these works often struggle with effective exploration, leading to reduced individual potential and overall system performance. To address these limitations, we propose a Role-Adaptive LLM-Driven Yoked navigation algorithm RALLY. Specifically, we first develop an LLM-driven semantic decision framework that uses structured natural language for efficient semantic communication and collaborative reasoning. Afterward, we introduce a dynamic role-heterogeneity mechanism for adaptive role switching and personalized decision-making. Furthermore, we propose a Role-value Mixing Network (RMIX)-based assignment strategy that integrates LLM offline priors with MARL online policies to enable semi-offline training of role selection strategies. Experiments in the Multi-Agent Particle Environment (MPE) environment and a Software-In-The-Loop (SITL) platform demonstrate that RALLY outperforms conventional approaches in terms of task coverage, convergence speed, and generalization, highlighting its strong potential for collaborative navigation in agentic multi-UAV systems.

**arXiv ID:** 2507.01378
</details>

<details>
<summary><strong>Grid-Agent: An LLM-Powered Multi-Agent System for Power Grid Control</strong> - Yan Zhang - [[pdf]](https://arxiv.org/pdf/2508.05702)</summary>

**Abstract:** The increasing penetration of Distributed Energy Resources (DERs), widespread adoption of Electric Vehicles (EVs), and the growing frequency of extreme weather events have significantly increased the complexity of power grid planning, operation, and management. Traditional rule-based systems and numerical optimization approaches often struggle with the scale, dynamics, and adaptability required by modern power networks. This paper introduces Grid-Agent, an autonomous, AI-driven framework that combines Large Language Models (LLMs) with multi-agent reinforcement learning to detect and remediate grid violations in real time. Grid-Agent integrates semantic reasoning with numerical precision through a modular agent architecture: a planning agent generates coordinated action sequences using numerical power flow solvers, while a validation agent evaluates system stability and action effectiveness via sandboxed execution with safety rollbacks. To ensure scalability, Grid-Agent incorporates an adaptive multiscale network representation that dynamically selects optimal encoding schemes based on network size and complexity. The framework enables coordinated violation resolution through optimizing switch configurations, battery deployment, and load curtailment strategies. Experimental results in standard IEEE and CIGRE test systems (IEEE 69-bus, CIGRE MV, and IEEE 30-bus) demonstrate superior violation mitigation performance. Additionally, the framework's built-in data collection and learning capabilities enable continuous learning and adaptation to diverse network topologies. The autonomous nature of the framework makes it particularly suitable for modern smart grid applications requiring rapid response to dynamic operating conditions.

**arXiv ID:** 2508.05702
</details>

<details>
<summary><strong>Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration</strong> - Yang Zhang, Shixin Yang, Chenjia Bai, Fei Wu, Xiu Li, Zhen Wang, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2405.14314)</summary>

**Abstract:** Grounding the reasoning ability of large language models (LLMs) for embodied tasks is challenging due to the complexity of the physical world. Especially, LLM planning for multi-agent collaboration requires communication of agents or credit assignment as the feedback to re-adjust the proposed plans and achieve effective coordination. However, existing methods that overly rely on physical verification or self-reflection suffer from excessive and inefficient querying of LLMs. In this paper, we propose a novel framework for multi-agent collaboration that introduces Reinforced Advantage feedback (ReAd) for efficient self-refinement of plans. Specifically, we perform critic regression to learn a sequential advantage function from LLM-planned data, and then treat the LLM planner as an optimizer to generate actions that maximize the advantage function. It endows the LLM with the foresight to discern whether the action contributes to accomplishing the final task. We provide theoretical analysis by extending advantage-weighted regression in reinforcement learning to multi-agent systems. Experiments on Overcooked-AI and a difficult variant of RoCoBench show that ReAd surpasses baselines in success rate, and also significantly decreases the interaction steps of agents and query rounds of LLMs, demonstrating its high efficiency for grounding LLMs. More results are given at this https URL.

**arXiv ID:** 2405.14314
</details>

<details>
<summary><strong>Decentralized Transformers with Centralized Aggregation are Sample-Efficient Multi-Agent World Models</strong> - Yang Zhang, Chenjia Bai, Bin Zhao, Junchi Yan, Xiu Li, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2406.15836)</summary>

**Abstract:** Learning a world model for model-free Reinforcement Learning (RL) agents can significantly improve the sample efficiency by learning policies in imagination. However, building a world model for Multi-Agent RL (MARL) can be particularly challenging due to the scalability issue in a centralized architecture arising from a large number of agents, and also the non-stationarity issue in a decentralized architecture stemming from the inter-dependency among agents. To address both challenges, we propose a novel world model for MARL that learns decentralized local dynamics for scalability, combined with a centralized representation aggregation from all agents. We cast the dynamics learning as an auto-regressive sequence modeling problem over discrete tokens by leveraging the expressive Transformer architecture, in order to model complex local dynamics across different agents and provide accurate and consistent long-term imaginations. As the first pioneering Transformer-based world model for multi-agent systems, we introduce a Perceiver Transformer as an effective solution to enable centralized representation aggregation within this context. Results on Starcraft Multi-Agent Challenge (SMAC) show that it outperforms strong model-free approaches and existing model-based methods in both sample efficiency and overall performance.

**arXiv ID:** 2406.15836
</details>

<details>
<summary><strong>Agent Trading Arena: A Study on Numerical Understanding in LLM-Based Agents</strong> - Tianmi Ma, Jiawei Du, Wenxin Huang, Wenjie Wang, Liang Xie, Xian Zhong, Joey Tianyi Zhou - [[pdf]](https://arxiv.org/pdf/2502.17967)</summary>

**Abstract:** Large language models (LLMs) have demonstrated remarkable capabilities in natural language tasks, yet their performance in dynamic, real-world financial environments remains underexplored. Existing approaches are limited to historical backtesting, where trading actions cannot influence market prices and agents train only on static data. To address this limitation, we present the Agent Trading Arena, a virtual zero-sum stock market in which LLM-based agents engage in competitive multi-agent trading and directly impact price dynamics. By simulating realistic bid-ask interactions, our platform enables training in scenarios that closely mirror live markets, thereby narrowing the gap between training and evaluation. Experiments reveal that LLMs struggle with numerical reasoning when given plain-text data, often overfitting to local patterns and recent values. In contrast, chart-based visualizations significantly enhance both numerical reasoning and trading performance. Furthermore, incorporating a reflection module yields additional improvements, especially with visual inputs. Evaluations on NASDAQ and CSI datasets demonstrate the superiority of our method, particularly under high volatility. All code and data are available at this https URL.

**arXiv ID:** 2502.17967
</details>

<details>
<summary><strong>The challenge of hidden gifts in multi-agent reinforcement learning</strong> - Dane Malenfant, Blake A. Richards - [[pdf]](https://arxiv.org/pdf/2505.20579)</summary>

**Abstract:** Sometimes we benefit from actions that others have taken even when we are unaware that they took those actions. For example, if your neighbor chooses not to take a parking spot in front of your house when you are not there, you can benefit, even without being aware that they took this action. These "hidden gifts" represent an interesting challenge for multi-agent reinforcement learning (MARL), since assigning credit when the beneficial actions of others are hidden is non-trivial. Here, we study the impact of hidden gifts with a very simple MARL task. In this task, agents in a grid-world environment have individual doors to unlock in order to obtain individual rewards. As well, if all the agents unlock their door the group receives a larger collective reward. However, there is only one key for all of the doors, such that the collective reward can only be obtained when the agents drop the key for others after they use it. Notably, there is nothing to indicate to an agent that the other agents have dropped the key, thus the act of dropping the key for others is a "hidden gift". We show that several different state-of-the-art RL algorithms, including MARL algorithms, fail to learn how to obtain the collective reward in this simple task. Interestingly, we find that independent model-free policy gradient agents can solve the task when we provide them with information about their own action history, but MARL agents still cannot solve the task with action history. Finally, we derive a correction term for these independent agents, inspired by learning aware approaches, which reduces the variance in learning and helps them to converge to collective success more reliably. These results show that credit assignment in multi-agent settings can be particularly challenging in the presence of "hidden gifts", and demonstrate that learning awareness in independent agents can benefit these settings.

**arXiv ID:** 2505.20579
</details>

<details>
<summary><strong>CRMAgent: A Multi-Agent LLM System for E-Commerce CRM Message Template Generation</strong> - Yinzhu Quan, Xinrui Li, Ying Chen - [[pdf]](https://arxiv.org/pdf/2507.08325)</summary>

**Abstract:** In e-commerce private-domain channels such as instant messaging and e-mail, merchants engage customers directly as part of their Customer Relationship Management (CRM) programmes to drive retention and conversion. While a few top performers excel at crafting outbound messages, most merchants struggle to write persuasive copy because they lack both expertise and scalable tools. We introduce CRMAgent, a multi-agent system built on large language models (LLMs) that generates high-quality message templates and actionable writing guidance through three complementary modes. First, group-based learning enables the agent to learn from a merchant's own top-performing messages within the same audience segment and rewrite low-performing ones. Second, retrieval-and-adaptation fetches templates that share the same audience segment and exhibit high similarity in voucher type and product category, learns their successful patterns, and adapts them to the current campaign. Third, a rule-based fallback provides a lightweight zero-shot rewrite when no suitable references are available. Extensive experiments show that CRMAgent consistently outperforms merchants' original templates, delivering significant gains in both audience-match and marketing-effectiveness metrics.

**arXiv ID:** 2507.08325
</details>

<details>
<summary><strong>TableZoomer: A Collaborative Agent Framework for Large-scale Table Question Answering</strong> - Sishi Xiong, Ziyang He, Zhongjiang He, Yu Zhao, Changzai Pan, Jie Zhang, Zhenhe Wu, Shuangyong Song, Yongxiang Li - [[pdf]](https://arxiv.org/pdf/2509.01312)</summary>

**Abstract:** While large language models (LLMs) have shown promise in the table question answering (TQA) task through prompt engineering, they face challenges in industrial applications, including structural heterogeneity, difficulties in target data localization, and bottlenecks in complex reasoning. To address these limitations, this paper presents TableZoomer, a novel LLM-powered, programming-based agent framework. It introduces three key innovations: (1) replacing the original fully verbalized table with structured table schema to bridge the semantic gap and reduce computational complexity; (2) a query-aware table zooming mechanism that dynamically generates sub-table schema through column selection and entity linking, significantly improving target localization efficiency; and (3) a Program-of-Thoughts (PoT) strategy that transforms queries into executable code to mitigate numerical hallucination. Additionally, we integrate the reasoning workflow with the ReAct paradigm to enable iterative reasoning. Extensive experiments demonstrate that our framework maintains the usability advantages while substantially enhancing performance and scalability across tables of varying scales. When implemented with the Qwen3-8B-Instruct LLM, TableZoomer achieves accuracy improvements of 19.34% and 25% over conventional PoT methods on the large-scale DataBench dataset and the small-scale Fact Checking task of TableBench dataset, respectively.

**arXiv ID:** 2509.01312
</details>

<details>
<summary><strong>Multiple LLM Agents Debate for Equitable Cultural Alignment</strong> - Dayeon Ki, Rachel Rudinger, Tianyi Zhou, Marine Carpuat - [[pdf]](https://arxiv.org/pdf/2505.24671)</summary>

**Abstract:** Large Language Models (LLMs) need to adapt their predictions to diverse cultural contexts to benefit diverse communities across the world. While previous efforts have focused on single-LLM, single-turn approaches, we propose to exploit the complementary strengths of multiple LLMs to promote cultural adaptability. We introduce a Multi-Agent Debate framework, where two LLM-based agents debate over a cultural scenario and collaboratively reach a final decision. We propose two variants: one where either LLM agents exclusively debate and another where they dynamically choose between self-reflection and debate during their turns. We evaluate these approaches on 7 open-weight LLMs (and 21 LLM combinations) using the NormAd-ETI benchmark for social etiquette norms in 75 countries. Experiments show that debate improves both overall accuracy and cultural group parity over single-LLM baselines. Notably, multi-agent debate enables relatively small LLMs (7-9B) to achieve accuracies comparable to that of a much larger model (27B parameters).

**arXiv ID:** 2505.24671
</details>

<details>
<summary><strong>Multi-Agent Reinforcement Learning for Task Offloading in Wireless Edge Networks</strong> - Andrea Fox, Francesco De Pellegrini, Eitan Altman - [[pdf]](https://arxiv.org/pdf/2509.01257)</summary>

**Abstract:** In edge computing systems, autonomous agents must make fast local decisions while competing for shared resources. Existing MARL methods often resume to centralized critics or frequent communication, which fail under limited observability and communication constraints. We propose a decentralized framework in which each agent solves a constrained Markov decision process (CMDP), coordinating implicitly through a shared constraint vector. For the specific case of offloading, e.g., constraints prevent overloading shared server resources. Coordination constraints are updated infrequently and act as a lightweight coordination mechanism. They enable agents to align with global resource usage objectives but require little direct communication. Using safe reinforcement learning, agents learn policies that meet both local and global goals. We establish theoretical guarantees under mild assumptions and validate our approach experimentally, showing improved performance over centralized and independent baselines, especially in large-scale settings.

**arXiv ID:** 2509.01257
</details>

<details>
<summary><strong>OpenMulti: Open-Vocabulary Instance-Level Multi-Agent Distributed Implicit Mapping</strong> - Jianyu Dou, Yinan Deng, Jiahui Wang, Xingsi Tang, Yi Yang, Yufeng Yue - [[pdf]](https://arxiv.org/pdf/2509.01228)</summary>

**Abstract:** Multi-agent distributed collaborative mapping provides comprehensive and efficient representations for robots. However, existing approaches lack instance-level awareness and semantic understanding of environments, limiting their effectiveness for downstream applications. To address this issue, we propose OpenMulti, an open-vocabulary instance-level multi-agent distributed implicit mapping framework. Specifically, we introduce a Cross-Agent Instance Alignment module, which constructs an Instance Collaborative Graph to ensure consistent instance understanding across agents. To alleviate the degradation of mapping accuracy due to the blind-zone optimization trap, we leverage Cross Rendering Supervision to enhance distributed learning of the scene. Experimental results show that OpenMulti outperforms related algorithms in both fine-grained geometric accuracy and zero-shot semantic accuracy. In addition, OpenMulti supports instance-level retrieval tasks, delivering semantic annotations for downstream applications. The project website of OpenMulti is publicly available at this https URL.

**arXiv ID:** 2509.01228
</details>

<details>
<summary><strong>Autonomous Task Planning for Heterogeneous Multi-Agent Systems</strong> - Anatoli A. Tziola, Savvas G. Loizou - [[pdf]](https://arxiv.org/pdf/2209.08611)</summary>

**Abstract:** This paper presents a solution to the automatic task planning problem for multi-agent systems. A formal framework is developed based on the Nondeterministic Finite Automata with $\epsilon$-transitions, where given the capabilities, constraints and failure modes of the agents involved, an initial state of the system and a task specification, an optimal solution is generated that satisfies the system constraints and the task specification. The resulting solution is guaranteed to be complete and optimal; moreover a heuristic solution that offers significant reduction of the computational requirements while relaxing the completeness and optimality requirements is proposed. The constructed system model is independent from the initial condition and the task specification, alleviating the need to repeat the costly pre-processing cycle for solving other scenarios, while allowing the incorporation of failure modes on-the-fly. Two case studies are provided: a simple one to showcase the concepts of the proposed methodology and a more elaborate one to demonstrate the effectiveness and validity of the methodology.

**arXiv ID:** 2209.08611
</details>

<details>
<summary><strong>Toward Real-World Cooperative and Competitive Soccer with Quadrupedal Robot Teams</strong> - Zhi Su, Yuman Gao, Emily Lukas, Yunfei Li, Jiaze Cai, Faris Tulbah, Fei Gao, Chao Yu, Zhongyu Li, Yi Wu, Koushil Sreenath - [[pdf]](https://arxiv.org/pdf/2505.13834)</summary>

**Abstract:** Achieving coordinated teamwork among legged robots requires both fine-grained locomotion control and long-horizon strategic decision-making. Robot soccer offers a compelling testbed for this challenge, combining dynamic, competitive, and multi-agent interactions. In this work, we present a hierarchical multi-agent reinforcement learning (MARL) framework that enables fully autonomous and decentralized quadruped robot soccer. First, a set of highly dynamic low-level skills is trained for legged locomotion and ball manipulation, such as walking, dribbling, and kicking. On top of these, a high-level strategic planning policy is trained with Multi-Agent Proximal Policy Optimization (MAPPO) via Fictitious Self-Play (FSP). This learning framework allows agents to adapt to diverse opponent strategies and gives rise to sophisticated team behaviors, including coordinated passing, interception, and dynamic role allocation. With an extensive ablation study, the proposed learning method shows significant advantages in the cooperative and competitive multi-agent soccer game. We deploy the learned policies to real quadruped robots relying solely on onboard proprioception and decentralized localization, with the resulting system supporting autonomous robot-robot and robot-human soccer matches on indoor and outdoor soccer courts.

**arXiv ID:** 2505.13834
</details>

</details>

<details open>
<summary><h2>Other Agent Research (11 papers)</h2></summary>

<details>
<summary><strong>Throttling Web Agents Using Reasoning Gates</strong> - Abhinav Kumar, Jaechul Roh, Ali Naseh, Amir Houmansadr, Eugene Bagdasarian - [[pdf]](https://arxiv.org/pdf/2509.01619)</summary>

**Abstract:** AI web agents use Internet resources at far greater speed, scale, and complexity -- changing how users and services interact. Deployed maliciously or erroneously, these agents could overload content providers. At the same time, web agents can bypass CAPTCHAs and other defenses by mimicking user behavior or flood authentication systems with fake accounts. Yet providers must protect their services and content from denial-of-service attacks and scraping by web agents. In this paper, we design a framework that imposes tunable costs on agents before providing access to resources; we call this Web Agent Throttling. We start by formalizing Throttling Gates as challenges issued to an agent that are asymmetric, scalable, robust, and compatible with any agent. Focusing on a common component -- the language model -- we require the agent to solve reasoning puzzles, thereby incurring excessive token-generation costs. However, we find that using existing puzzles, e.g., coding or math, as throttling gates fails to satisfy our properties. To address this, we introduce rebus-based Reasoning Gates, synthetic text puzzles that require multi-hop reasoning over world knowledge (thereby throttling an agent's model). We design a scalable generation and verification protocol for such reasoning gates. Our framework achieves computational asymmetry, i.e., the response-generation cost is 9.2x higher than the generation cost for SOTA models. We further deploy reasoning gates on a custom website and Model Context Protocol (MCP) servers and evaluate with real-world web agents. Finally, we discuss the limitations and environmental impact of real-world deployment of our framework.

**arXiv ID:** 2509.01619
</details>

<details>
<summary><strong>When Agents go Astray: Course-Correcting SWE Agents with PRMs</strong> - Shubham Gandhi, Jason Tsay, Jatin Ganhotra, Kiran Kate, Yara Rizk - [[pdf]](https://arxiv.org/pdf/2509.02360)</summary>

**Abstract:** Large Language Model (LLM) agents are increasingly deployed for complex, multi-step software engineering (SWE) tasks. However, their trajectories often contain costly inefficiencies, such as redundant exploration, looping, and failure to terminate once a solution is reached. Prior work has largely treated these errors in a post-hoc manner, diagnosing failures only after execution. In this paper, we introduce SWE-PRM, an inference-time Process Reward Model (PRM) that intervenes during execution to detect and course-correct trajectory-level errors. Our PRM design leverages a taxonomy of common inefficiencies and delivers lightweight, interpretable feedback without modifying the underlying policy. On SWE-bench Verified, closed-source PRMs improve resolution from 40.0% to 50.6% (+10.6 p.p.), with the largest gains on medium and hard tasks. Among feedback strategies, taxonomy-guided PRMs outperform unguided or explicit action-prescriptive variants, increasing success rate while reducing trajectory length. These benefits come at an acceptable added inference cost of as low as $0.2, making PRMs a practical and scalable mechanism for improving SWE agents' reliability and efficiency.

**arXiv ID:** 2509.02360
</details>

<details>
<summary><strong>A Whole New World: Creating a Parallel-Poisoned Web Only AI-Agents Can See</strong> - Shaked Zychlinski - [[pdf]](https://arxiv.org/pdf/2509.00124)</summary>

**Abstract:** This paper introduces a novel attack vector that leverages website cloaking techniques to compromise autonomous web-browsing agents powered by Large Language Models (LLMs). As these agents become more prevalent, their unique and often homogenous digital fingerprints - comprising browser attributes, automation framework signatures, and network characteristics - create a new, distinguishable class of web traffic. The attack exploits this fingerprintability. A malicious website can identify an incoming request as originating from an AI agent and dynamically serve a different, "cloaked" version of its content. While human users see a benign webpage, the agent is presented with a visually identical page embedded with hidden, malicious instructions, such as indirect prompt injections. This mechanism allows adversaries to hijack agent behavior, leading to data exfiltration, malware execution, or misinformation propagation, all while remaining completely invisible to human users and conventional security crawlers. This work formalizes the threat model, details the mechanics of agent fingerprinting and cloaking, and discusses the profound security implications for the future of agentic AI, highlighting the urgent need for robust defenses against this stealthy and scalable attack.

**arXiv ID:** 2509.00124
</details>

<details>
<summary><strong>Beyond Pixels: Introducing Geometric-Semantic World Priors for Video-based Embodied Models via Spatio-temporal Alignment</strong> - Jinzhou Tang, Jusheng zhang, Sidi Liu, Waikit Xiu, Qinhan Lv, Xiying Li - [[pdf]](https://arxiv.org/pdf/2509.00210)</summary>

**Abstract:** Achieving human-like reasoning in deep learning models for complex tasks in unknown environments remains a critical challenge in embodied intelligence. While advanced vision-language models (VLMs) excel in static scene understanding, their limitations in spatio-temporal reasoning and adaptation to dynamic, open-set tasks like task-oriented navigation and embodied question answering (EQA) persist due to inadequate modeling of fine-grained spatio-temporal cues and physical world comprehension. To address this, we propose VEME, a novel cross-modal alignment method that enhances generalization in unseen scenes by learning an ego-centric, experience-centered world model. Our framework integrates three key components: (1) a cross-modal alignment framework bridging objects, spatial representations, and visual semantics with spatio-temporal cues to enhance VLM in-context learning; (2) a dynamic, implicit cognitive map activated by world embedding to enable task-relevant geometric-semantic memory recall; and (3) an instruction-based navigation and reasoning framework leveraging embodied priors for long-term planning and efficient exploration. By embedding geometry-aware spatio-temporal episodic experiences, our method significantly improves reasoning and planning in dynamic environments. Experimental results on VSI-Bench and VLN-CE demonstrate 1%-3% accuracy and exploration efficiency improvement compared to traditional approaches.

**arXiv ID:** 2509.00210
</details>

<details>
<summary><strong>An Economy of AI Agents</strong> - Gillian K. Hadfield, Andrew Koh - [[pdf]](https://arxiv.org/pdf/2509.01063)</summary>

**Abstract:** In the coming decade, artificially intelligent agents with the ability to plan and execute complex tasks over long time horizons with little direct oversight from humans may be deployed across the economy. This chapter surveys recent developments and highlights open questions for economists around how AI agents might interact with humans and with each other, shape markets and organizations, and what institutions might be required for well-functioning markets.

**arXiv ID:** 2509.01063
</details>

<details>
<summary><strong>Autonomous Aggregate Sorting in Construction and Mining via Computer Vision-Aided Robotic Arm Systems</strong> - Md. Taherul Islam Shawon, Yuan Li, Yincai Cai, Junjie Niu, Ting Peng - [[pdf]](https://arxiv.org/pdf/2509.00339)</summary>

**Abstract:** Traditional aggregate sorting methods, whether manual or mechanical, often suffer from low precision, limited flexibility, and poor adaptability to diverse material properties such as size, shape, and lithology. To address these limitations, this study presents a computer vision-aided robotic arm system designed for autonomous aggregate sorting in construction and mining applications. The system integrates a six-degree-of-freedom robotic arm, a binocular stereo camera for 3D perception, and a ROS-based control framework. Core techniques include an attention-augmented YOLOv8 model for aggregate detection, stereo matching for 3D localization, Denavit-Hartenberg kinematic modeling for arm motion control, minimum enclosing rectangle analysis for size estimation, and hand-eye calibration for precise coordinate alignment. Experimental validation with four aggregate types achieved an average grasping and sorting success rate of 97.5%, with comparable classification accuracy. Remaining challenges include the reliable handling of small aggregates and texture-based misclassification. Overall, the proposed system demonstrates significant potential to enhance productivity, reduce operational costs, and improve safety in aggregate handling, while providing a scalable framework for advancing smart automation in construction, mining, and recycling industries.

**arXiv ID:** 2509.00339
</details>

<details>
<summary><strong>Vehicle-in-Virtual-Environment (VVE) Method for Developing and Evaluating VRU Safety of Connected and Autonomous Driving with Focus on Bicyclist Safety</strong> - Haochong Chen, Xincheng Cao, Bilin Aksun-Guvenc, Levent Guvenc - [[pdf]](https://arxiv.org/pdf/2509.00624)</summary>

**Abstract:** Extensive research has already been conducted in the autonomous driving field to help vehicles navigate safely and efficiently. At the same time, plenty of current research on vulnerable road user (VRU) safety is performed which largely concentrates on perception, localization, or trajectory prediction of VRUs. However, existing research still exhibits several gaps, including the lack of a unified planning and collision avoidance system for autonomous vehicles, limited investigation into delay tolerant control strategies, and the absence of an efficient and standardized testing methodology. Ensuring VRU safety remains one of the most pressing challenges in autonomous driving, particularly in dynamic and unpredictable environments. In this two year project, we focused on applying the Vehicle in Virtual Environment (VVE) method to develop, evaluate, and demonstrate safety functions for Vulnerable Road Users (VRUs) using automated steering and braking of ADS. In this current second year project report, our primary focus was on enhancing the previous year results while also considering bicyclist safety.

**arXiv ID:** 2509.00624
</details>

<details>
<summary><strong>Analyzing Reluctance to Ask for Help When Cooperating With Robots: Insights to Integrate Artificial Agents in HRC</strong> - Ane San Martin, Michael Hagenow, Julie Shah, Johan Kildal, Elena Lazkano - [[pdf]](https://arxiv.org/pdf/2509.01450)</summary>

**Abstract:** As robot technology advances, collaboration between humans and robots will become more prevalent in industrial tasks. When humans run into issues in such scenarios, a likely future involves relying on artificial agents or robots for aid. This study identifies key aspects for the design of future user-assisting agents. We analyze quantitative and qualitative data from a user study examining the impact of on-demand assistance received from a remote human in a human-robot collaboration (HRC) assembly task. We study scenarios in which users require help and we assess their experiences in requesting and receiving assistance. Additionally, we investigate participants' perceptions of future non-human assisting agents and whether assistance should be on-demand or unsolicited. Through a user study, we analyze the impact that such design decisions (human or artificial assistant, on-demand or unsolicited help) can have on elicited emotional responses, productivity, and preferences of humans engaged in HRC tasks.

**arXiv ID:** 2509.01450
</details>

<details>
<summary><strong>Nav-SCOPE: Swarm Robot Cooperative Perception and Coordinated Navigation</strong> - Chenxi Li, Weining Lu, Qingquan Lin, Litong Meng, Haolu Li, Bin Liang - [[pdf]](https://arxiv.org/pdf/2409.10049)</summary>

**Abstract:** This paper proposes a lightweight systematic solution for multi-robot coordinated navigation with decentralized cooperative perception. An information flow is first created to facilitate real-time observation sharing over unreliable ad-hoc networks. Then, the environmental uncertainties of each robot are reduced by interaction fields that deliver complementary information. Finally, path optimization is achieved, enabling self-organized coordination with effective convergence, divergence, and collision avoidance. Our method is fully interpretable and ready for deployment without gaps. Comprehensive simulations and real-world experiments demonstrate reduced path redundancy, robust performance across various tasks, and minimal demands on computation and communication.

**arXiv ID:** 2409.10049
</details>

<details>
<summary><strong>General agents contain world models</strong> - Jonathan Richens, David Abel, Alexis Bellot, Tom Everitt - [[pdf]](https://arxiv.org/pdf/2506.01622)</summary>

**Abstract:** Are world models a necessary ingredient for flexible, goal-directed behaviour, or is model-free learning sufficient? We provide a formal answer to this question, showing that any agent capable of generalizing to multi-step goal-directed tasks must have learned a predictive model of its environment. We show that this model can be extracted from the agent's policy, and that increasing the agents performance or the complexity of the goals it can achieve requires learning increasingly accurate world models. This has a number of consequences: from developing safe and general agents, to bounding agent capabilities in complex environments, and providing new algorithms for eliciting world models from agents.

**arXiv ID:** 2506.01622
</details>

<details>
<summary><strong>Know What, Know Why: Semantic Hazard Communication for Intelligent V2X Systems</strong> - Chen Sun, Wenqi Zhang, Bizhu Wang, Xiaodong Xu, Chau Yuen, Yan Zhang, Ping Zhang - [[pdf]](https://arxiv.org/pdf/2509.02442)</summary>

**Abstract:** In current vehicle-to-everything (V2X) communication systems, roadside units (RSUs) broadcast brief warning messages that alert nearby vehicles to avoid potential hazards. However, these messages lack contextual information on why a warning is issued, leading to excessive caution or inefficient driving behaviors. To avoid such a situation, we propose a semantic-enhanced and explainable V2X (SEE-V2X) system. In the proposed system, RSUs equipped with smart cameras detect obstructions and transmit context-aware messages to vehicles. By understanding both what the hazard is and why it occurs, drivers can make more intelligent decisions based on their specific driving situation. Furthermore, through a real-field demonstration, we show the new "see-through" feature in the proposed system, which enables drivers to visualize hidden pedestrians behind obstacles. We also perform simulations to compare traditional V2X with SEE-V2X under different traffic conditions. The results show that SEE-V2X significantly improves traffic efficiency and reduces unnecessary deceleration.

**arXiv ID:** 2509.02442
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (43 papers)</h2></summary>

<details>
<summary><strong>Know When to Explore: Difficulty-Aware Certainty as a Guide for LLM Reinforcement Learning</strong> - Ang Li, Zhihang Yuan, Yang Zhang, Shouda Liu, Yisen Wang - [[pdf]](https://arxiv.org/pdf/2509.00125)</summary>

**Abstract:** Reinforcement Learning with Verifiable Feedback (RLVF) has become a key technique for enhancing the reasoning abilities of Large Language Models (LLMs). However, its reliance on sparse, outcome based rewards, which only indicate if a final answer is correct or not, fails to provide granular guidance on the reasoning process itself. This limitation hinders efficient learning, as the model cannot distinguish between high quality and inefficient solutions, nor can it learn effectively from different types of failures. To address this, we observe that an LLMs self-certainty often correlates with task difficulty and solution quality. We introduce Difficulty Aware Certainty guided Exploration (DACE), a novel RL algorithm that leverages this insight to dynamically balance the exploration exploitation trade-off. DACE assesses task difficulty online based on the policys success rate. It then uses this signal to modulate an intrinsic reward: for difficult tasks where the model is struggling, DACE encourages exploration by penalizing high certainty; for easier tasks, it encourages learning efficiency by rewarding high certainty. Experiments on challenging mathematical reasoning benchmarks (AIME, MATH) show that DACE significantly outperforms strong baselines. The DACE-trained models not only achieve higher accuracy but also demonstrate more robust performance when scaling test-time compute, validating that our adaptive approach fosters effective exploration without sacrificing precision.

**arXiv ID:** 2509.00125
</details>

<details>
<summary><strong>Instruction-Level Weight Shaping: A Framework for Self-Improving AI Agents</strong> - Rimom Costa - [[pdf]](https://arxiv.org/pdf/2509.00251)</summary>

**Abstract:** Large language models (LLMs) are fluent but largely static after pre-training; new or shifting knowledge is typically added with retrieval-augmented generation (RAG) or fine-tuning. RAG raises latency and engineering overhead and often fails to integrate facts; prompt engineering is brittle and can conflict with prior knowledge; fine-tuning is costly and risks catastrophic forgetting. We propose Instruction-Level Weight Shaping (ILWS): curated system instructions act as external, auditable pseudo-parameters updated after each session via reflection and user feedback. A Reflection Engine inspects conversation traces, diagnoses reasoning successes and failures, and proposes typed deltas $\Delta K=(\Delta S,\Delta U,\Delta T)$ over instructions, user preferences, and tools. Deltas are version-controlled, evaluated with a sliding window of 1-5 star ratings, auto-repaired on first failure, and rolled back on repeated failure. When an edit budget crosses a threshold, the agent compiles a rating-weighted synthetic set and distills matured instruction-space gains into parameters, converting prompt-space improvements into weight-space without downtime. ILWS makes explicit the low-rank shaping induced by context in transformer blocks, preserves governance, and removes per-call retrieval. In enterprise support it increased throughput 2.4-5.0x and cut audited hallucinations by about 80% versus a frozen baseline. In an Adobe Commerce Cloud proof of concept "L0 Support", it achieved 4-5x more tickets per hour and about 80% lower time per ticket, with autonomous instruction updates and optional tool synthesis. Because ILWS operates at the instruction layer until controlled distillation, it generalizes to dynamic domains (legal, medical, engineering) requiring adaptive reasoning, tool creation, and low-latency deployment.

**arXiv ID:** 2509.00251
</details>

<details>
<summary><strong>Self-Exploring Language Models for Explainable Link Forecasting on Temporal Graphs via Reinforcement Learning</strong> - Zifeng Ding, Shenyang Huang, Zeyu Cao, Emma Kondrup, Zachary Yang, Xingyue Huang, Yuan Sui, Zhangdie Yuan, Yuqicheng Zhu, Xianglong Hu, Yuan He, Farimah Poursafaei, Michael Bronstein, Andreas Vlachos - [[pdf]](https://arxiv.org/pdf/2509.00975)</summary>

**Abstract:** Forecasting future links is a central task in temporal graph (TG) reasoning, requiring models to leverage historical interactions to predict upcoming ones. Traditional neural approaches, such as temporal graph neural networks, achieve strong performance but lack explainability and cannot be applied to unseen graphs without retraining. Recent studies have begun to explore using large language models (LLMs) for graph reasoning, but most of them are constrained to static graphs or small synthetic TGs and lack the evaluation of the quality of reasoning traces generated by LLMs. In this work, we present Reasoning-Enhanced Learning for Temporal Graphs (ReaL-TG), a reinforcement learning framework that fine-tunes LLMs to perform explainable link forecasting on real-world TGs. ReaL-TG uses outcome-based reward to encourage models to self-explore reasoning strategies from graph structure and to produce explanations that directly justify their predictions. To enable evaluation on LLM-generated reasoning traces, we propose a new evaluation protocol combining ranking metrics with an LLM-as-a-Judge system that assesses both the quality of reasoning and the impact of hallucinations. Experiments with ReaL-TG-4B, obtained by fine-tuning Qwen3-4B under our framework, show that it outperforms much larger frontier LLMs, including GPT-5 mini, on ranking metrics, while producing high-quality explanations confirmed by both the LLM judge and human evaluation.

**arXiv ID:** 2509.00975
</details>

<details>
<summary><strong>Supporting Our AI Overlords: Redesigning Data Systems to be Agent-First</strong> - Shu Liu, Soujanya Ponnapalli, Shreya Shankar, Sepanta Zeighami, Alan Zhu, Shubham Agarwal, Ruiqi Chen, Samion Suwito, Shuo Yuan, Ion Stoica, Matei Zaharia, Alvin Cheung, Natacha Crooks, Joseph E. Gonzalez, Aditya G. Parameswaran - [[pdf]](https://arxiv.org/pdf/2509.00997)</summary>

**Abstract:** Large Language Model (LLM) agents, acting on their users' behalf to manipulate and analyze data, are likely to become the dominant workload for data systems in the future. When working with data, agents employ a high-throughput process of exploration and solution formulation for the given task, one we call agentic speculation. The sheer volume and inefficiencies of agentic speculation can pose challenges for present-day data systems. We argue that data systems need to adapt to more natively support agentic workloads. We take advantage of the characteristics of agentic speculation that we identify, i.e., scale, heterogeneity, redundancy, and steerability - to outline a number of new research opportunities for a new agent-first data systems architecture, ranging from new query interfaces, to new query processing techniques, to new agentic memory stores.

**arXiv ID:** 2509.00997
</details>

<details>
<summary><strong>VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use</strong> - Dongfu Jiang, Yi Lu, Zhuofeng Li, Zhiheng Lyu, Ping Nie, Haozhe Wang, Alex Su, Hui Chen, Kai Zou, Chao Du, Tianyu Pang, Wenhu Chen - [[pdf]](https://arxiv.org/pdf/2509.01055)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated success in enhancing LLM reasoning capabilities, but remains limited to single-turn interactions without tool integration. While recent Agentic Reinforcement Learning with Tool use (ARLT) approaches have emerged to address multi-turn tool interactions, existing works develop task-specific codebases that suffer from fragmentation, synchronous execution bottlenecks, and limited extensibility across domains. These inefficiencies hinder broader community adoption and algorithmic innovation. We introduce VerlTool, a unified and modular framework that addresses these limitations through systematic design principles. VerlTool provides four key contributions: (1) upstream alignment with VeRL ensuring compatibility and simplified maintenance, (2) unified tool management via standardized APIs supporting diverse modalities including code execution, search, SQL databases, and vision processing, (3) asynchronous rollout execution achieving near 2$\times$ speedup by eliminating synchronization bottlenecks, and (4) comprehensive evaluation demonstrating competitive performance across 6 ARLT domains. Our framework formalizes ARLT as multi-turn trajectories with multi-modal observation tokens (text/image/video), extending beyond single-turn RLVR paradigms. We train and evaluate models on mathematical reasoning, knowledge QA, SQL generation, visual reasoning, web search, and software engineering tasks, achieving results comparable to specialized systems while providing unified training infrastructure. The modular plugin architecture enables rapid tool integration requiring only lightweight Python definitions, significantly reducing development overhead and providing a scalable foundation for tool-augmented RL research. Our code is open-sourced at this https URL.

**arXiv ID:** 2509.01055
</details>

<details>
<summary><strong>Dynamic Speculative Agent Planning</strong> - Yilin Guan, Wenyue Hua, Qingfeng Lan, Sun Fei, Dujian Ding, Devang Acharya, Chi Wang, William Yang Wang - [[pdf]](https://arxiv.org/pdf/2509.01920)</summary>

**Abstract:** Despite their remarkable success in complex tasks propelling widespread adoption, large language-model-based agents still face critical deployment challenges due to prohibitive latency and inference costs. While recent work has explored various methods to accelerate inference, existing approaches suffer from significant limitations: they either fail to preserve performance fidelity, require extensive offline training of router modules, or incur excessive operational costs. Moreover, they provide minimal user control over the tradeoff between acceleration and other performance metrics. To address these gaps, we introduce Dynamic Speculative Planning (DSP), an asynchronous online reinforcement learning framework that provides lossless acceleration with substantially reduced costs without requiring additional pre-deployment preparation. DSP explicitly optimizes a joint objective balancing end-to-end latency against dollar cost, allowing practitioners to adjust a single parameter that steers the system toward faster responses, cheaper operation, or any point along this continuum. Experiments on two standard agent benchmarks demonstrate that DSP achieves comparable efficiency to the fastest lossless acceleration method while reducing total cost by 30% and unnecessary cost up to 60%. Our code and data are available through this https URL.

**arXiv ID:** 2509.01920
</details>

<details>
<summary><strong>Towards Agents That Know When They Don't Know: Uncertainty as a Control Signal for Structured Reasoning</strong> - Josefa Lia Stoisser, Marc Boubnovski Martell, Lawrence Phillips, Gianluca Mazzoni, Lea Mørch Harder, Philip Torr, Jesper Ferkinghoff-Borg, Kaspar Martens, Julien Fauqueur - [[pdf]](https://arxiv.org/pdf/2509.02401)</summary>

**Abstract:** Large language model (LLM) agents are increasingly deployed in structured biomedical data environments, yet they often produce fluent but overconfident outputs when reasoning over complex multi-table data. We introduce an uncertainty-aware agent for query-conditioned multi-table summarization that leverages two complementary signals: (i) retrieval uncertainty--entropy over multiple table-selection rollouts--and (ii) summary uncertainty--combining self-consistency and perplexity. Summary uncertainty is incorporated into reinforcement learning (RL) with Group Relative Policy Optimization (GRPO), while both retrieval and summary uncertainty guide inference-time filtering and support the construction of higher-quality synthetic datasets.
On multi-omics benchmarks, our approach improves factuality and calibration, nearly tripling correct and useful claims per summary (3.0\(\rightarrow\)8.4 internal; 3.6\(\rightarrow\)9.9 cancer multi-omics) and substantially improving downstream survival prediction (C-index 0.32\(\rightarrow\)0.63). These results demonstrate that uncertainty can serve as a control signal--enabling agents to abstain, communicate confidence, and become more reliable tools for complex structured-data environments.

**arXiv ID:** 2509.02401
</details>

<details>
<summary><strong>The Landscape of Agentic Reinforcement Learning for LLMs: A Survey</strong> - Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi Li, Xiangyuan Xue, Yijiang Li, Yifan Zhou, Yang Chen, Chen Zhang, Yutao Fan, Zihu Wang, Songtao Huang, Yue Liao, Hongru Wang, Mengyue Yang, Heng Ji, Michael Littman, Jun Wang, Shuicheng Yan, Philip Torr, Lei Bai - [[pdf]](https://arxiv.org/pdf/2509.02547)</summary>

**Abstract:** The emergence of agentic reinforcement learning (Agentic RL) marks a paradigm shift from conventional reinforcement learning applied to large language models (LLM RL), reframing LLMs from passive sequence generators into autonomous, decision-making agents embedded in complex, dynamic worlds. This survey formalizes this conceptual shift by contrasting the degenerate single-step Markov Decision Processes (MDPs) of LLM-RL with the temporally extended, partially observable Markov decision processes (POMDPs) that define Agentic RL. Building on this foundation, we propose a comprehensive twofold taxonomy: one organized around core agentic capabilities, including planning, tool use, memory, reasoning, self-improvement, and perception, and the other around their applications across diverse task domains. Central to our thesis is that reinforcement learning serves as the critical mechanism for transforming these capabilities from static, heuristic modules into adaptive, robust agentic behavior. To support and accelerate future research, we consolidate the landscape of open-source environments, benchmarks, and frameworks into a practical compendium. By synthesizing over five hundred recent works, this survey charts the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose AI agents.

**arXiv ID:** 2509.02547
</details>

<details>
<summary><strong>U2UData-2: A Scalable Swarm UAVs Autonomous Flight Dataset for Long-horizon Tasks</strong> - Tongtong Feng, Xin Wang, Feilin Han, Leping Zhang, Wenwu Zhu - [[pdf]](https://arxiv.org/pdf/2509.00055)</summary>

**Abstract:** Swarm UAV autonomous flight for Long-Horizon (LH) tasks is crucial for advancing the low-altitude economy. However, existing methods focus only on specific basic tasks due to dataset limitations, failing in real-world deployment for LH tasks. LH tasks are not mere concatenations of basic tasks, requiring handling long-term dependencies, maintaining persistent states, and adapting to dynamic goal shifts. This paper presents U2UData-2, the first large-scale swarm UAV autonomous flight dataset for LH tasks and the first scalable swarm UAV data online collection and algorithm closed-loop verification platform. The dataset is captured by 15 UAVs in autonomous collaborative flights for LH tasks, comprising 12 scenes, 720 traces, 120 hours, 600 seconds per trajectory, 4.32M LiDAR frames, and 12.96M RGB frames. This dataset also includes brightness, temperature, humidity, smoke, and airflow values covering all flight routes. The platform supports the customization of simulators, UAVs, sensors, flight algorithms, formation modes, and LH tasks. Through a visual control window, this platform allows users to collect customized datasets through one-click deployment online and to verify algorithms by closed-loop simulation. U2UData-2 also introduces an LH task for wildlife conservation and provides comprehensive benchmarks with 9 SOTA models. U2UData-2 can be found at this https URL.

**arXiv ID:** 2509.00055
</details>

<details>
<summary><strong>Embodied AI: Emerging Risks and Opportunities for Policy Action</strong> - Jared Perlo, Alexander Robey, Fazl Barez, Luciano Floridi, Jakob Mökander - [[pdf]](https://arxiv.org/pdf/2509.00117)</summary>

**Abstract:** The field of embodied AI (EAI) is rapidly advancing. Unlike virtual AI, EAI can exist in, learn from, reason about, and act in the physical world. Given recent innovations in large language and multimodal models, along with increasingly advanced and responsive hardware, EAI systems are rapidly growing in capabilities and operational domains. These advances present significant risks, including physical harm from malicious use, mass surveillance, and economic and societal disruption. However, these risks have been severely overlooked by policymakers. Existing policies, such as international standards for industrial robots or statutes governing autonomous vehicles, are insufficient to address the full range of concerns. While lawmakers are increasingly focused on AI, there is now an urgent need to extend and adapt existing frameworks to account for the unique risks of EAI. To help bridge this gap, this paper makes three contributions: first, we provide a foundational taxonomy of key physical, informational, economic, and social EAI risks. Secondly, we analyze policies in the US, EU, and UK to identify how existing frameworks address these risks and where these policies leave critical gaps. We conclude by offering concrete policy recommendations to address the coming wave of EAI innovation, including mandatory testing and certification for EAI systems, clarified liability frameworks, and forward-looking strategies to manage and prepare for transformative economic and societal impacts.

**arXiv ID:** 2509.00117
</details>

<details>
<summary><strong>Embodied AI in Social Spaces: Responsible and Adaptive Robots in Complex Setting - UKAIRS 2025 (Copy)</strong> - Aleksandra Landowska, Aislinn D Gomez Bergin, Ayodeji O. Abioye, Jayati Deshmukh, Andriana Bouadouki, Maria Wheadon, Athina Georgara, Dominic Price, Tuyen Nguyen, Shuang Ao, Lokesh Singh, Yi Long, Raffaele Miele, Joel E. Fischer, Sarvapali D. Ramchurn - [[pdf]](https://arxiv.org/pdf/2509.00218)</summary>

**Abstract:** This paper introduces and overviews a multidisciplinary project aimed at developing responsible and adaptive multi-human multi-robot (MHMR) systems for complex, dynamic settings. The project integrates co-design, ethical frameworks, and multimodal sensing to create AI-driven robots that are emotionally responsive, context-aware, and aligned with the needs of diverse users. We outline the project's vision, methodology, and early outcomes, demonstrating how embodied AI can support sustainable, ethical, and human-centred futures.

**arXiv ID:** 2509.00218
</details>

<details>
<summary><strong>Contact-Aided Navigation of Flexible Robotic Endoscope Using Deep Reinforcement Learning in Dynamic Stomach</strong> - Chi Kit Ng, Huxin Gao, Tian-Ao Ren, Jiewen Lai, Hongliang Ren - [[pdf]](https://arxiv.org/pdf/2509.00319)</summary>

**Abstract:** Navigating a flexible robotic endoscope (FRE) through the gastrointestinal tract is critical for surgical diagnosis and treatment. However, navigation in the dynamic stomach is particularly challenging because the FRE must learn to effectively use contact with the deformable stomach walls to reach target locations. To address this, we introduce a deep reinforcement learning (DRL) based Contact-Aided Navigation (CAN) strategy for FREs, leveraging contact force feedback to enhance motion stability and navigation precision. The training environment is established using a physics-based finite element method (FEM) simulation of a deformable stomach. Trained with the Proximal Policy Optimization (PPO) algorithm, our approach achieves high navigation success rates (within 3 mm error between the FRE's end-effector and target) and significantly outperforms baseline policies. In both static and dynamic stomach environments, the CAN agent achieved a 100% success rate with 1.6 mm average error, and it maintained an 85% success rate in challenging unseen scenarios with stronger external disturbances. These results validate that the DRL-based CAN strategy substantially enhances FRE navigation performance over prior methods.

**arXiv ID:** 2509.00319
</details>

<details>
<summary><strong>Jacobian Exploratory Dual-Phase Reinforcement Learning for Dynamic Endoluminal Navigation of Deformable Continuum Robots</strong> - Yu Tian, Chi Kit Ng, Hongliang Ren - [[pdf]](https://arxiv.org/pdf/2509.00329)</summary>

**Abstract:** Deformable continuum robots (DCRs) present unique planning challenges due to nonlinear deformation mechanics and partial state observability, violating the Markov assumptions of conventional reinforcement learning (RL) methods. While Jacobian-based approaches offer theoretical foundations for rigid manipulators, their direct application to DCRs remains limited by time-varying kinematics and underactuated deformation dynamics. This paper proposes Jacobian Exploratory Dual-Phase RL (JEDP-RL), a framework that decomposes planning into phased Jacobian estimation and policy execution. During each training step, we first perform small-scale local exploratory actions to estimate the deformation Jacobian matrix, then augment the state representation with Jacobian features to restore approximate Markovianity. Extensive SOFA surgical dynamic simulations demonstrate JEDP-RL's three key advantages over proximal policy optimization (PPO) baselines: 1) Convergence speed: 3.2x faster policy convergence, 2) Navigation efficiency: requires 25% fewer steps to reach the target, and 3) Generalization ability: achieve 92% success rate under material property variations and achieve 83% (33% higher than PPO) success rate in the unseen tissue environment.

**arXiv ID:** 2509.00329
</details>

<details>
<summary><strong>LLM-Driven Policy Diffusion: Enhancing Generalization in Offline Reinforcement Learning</strong> - Hanping Zhang, Yuhong Guo - [[pdf]](https://arxiv.org/pdf/2509.00347)</summary>

**Abstract:** Reinforcement Learning (RL) is known for its strong decision-making capabilities and has been widely applied in various real-world scenarios. However, with the increasing availability of offline datasets and the lack of well-designed online environments from human experts, the challenge of generalization in offline RL has become more prominent. Due to the limitations of offline data, RL agents trained solely on collected experiences often struggle to generalize to new tasks or environments. To address this challenge, we propose LLM-Driven Policy Diffusion (LLMDPD), a novel approach that enhances generalization in offline RL using task-specific prompts. Our method incorporates both text-based task descriptions and trajectory prompts to guide policy learning. We leverage a large language model (LLM) to process text-based prompts, utilizing its natural language understanding and extensive knowledge base to provide rich task-relevant context. Simultaneously, we encode trajectory prompts using a transformer model, capturing structured behavioral patterns within the underlying transition dynamics. These prompts serve as conditional inputs to a context-aware policy-level diffusion model, enabling the RL agent to generalize effectively to unseen tasks. Our experimental results demonstrate that LLMDPD outperforms state-of-the-art offline RL methods on unseen tasks, highlighting its effectiveness in improving generalization and adaptability in diverse settings.

**arXiv ID:** 2509.00347
</details>

<details>
<summary><strong>It's-A-Me, Quantum Mario: Scalable Quantum Reinforcement Learning with Multi-Chip Ensembles</strong> - Junghoon Justin Park, Huan-Hsin Tseng, Shinjae Yoo, Samuel Yen-Chi Chen, Jiook Cha - [[pdf]](https://arxiv.org/pdf/2509.00713)</summary>

**Abstract:** Quantum reinforcement learning (QRL) promises compact function approximators with access to vast Hilbert spaces, but its practical progress is slowed by NISQ-era constraints such as limited qubits and noise accumulation. We introduce a multi-chip ensemble framework using multiple small Quantum Convolutional Neural Networks (QCNNs) to overcome these constraints. Our approach partitions complex, high-dimensional observations from the Super Mario Bros environment across independent quantum circuits, then classically aggregates their outputs within a Double Deep Q-Network (DDQN) framework. This modular architecture enables QRL in complex environments previously inaccessible to quantum agents, achieving superior performance and learning stability compared to classical baselines and single-chip quantum models. The multi-chip ensemble demonstrates enhanced scalability by reducing information loss from dimensionality reduction while remaining implementable on near-term quantum hardware, providing a practical pathway for applying QRL to real-world problems.

**arXiv ID:** 2509.00713
</details>

<details>
<summary><strong>Adaptive Vehicle Speed Classification via BMCNN with Reinforcement Learning-Enhanced Acoustic Processing</strong> - Yuli Zhang, Pengfei Fan, Ruiyuan Jiang, Hankang Gu, Dongyao Jia, Xinheng Wang - [[pdf]](https://arxiv.org/pdf/2509.00839)</summary>

**Abstract:** Traffic congestion remains a pressing urban challenge, requiring intelligent transportation systems for real-time management. We present a hybrid framework that combines deep learning and reinforcement learning for acoustic vehicle speed classification. A dual-branch BMCNN processes MFCC and wavelet features to capture complementary frequency patterns. An attention-enhanced DQN adaptively selects the minimal number of audio frames and triggers early decisions once confidence thresholds are reached. Evaluations on IDMT-Traffic and our SZUR-Acoustic (Suzhou) datasets show 95.99% and 92.3% accuracy, with up to 1.63x faster average processing via early termination. Compared with A3C, DDDQN, SA2C, PPO, and TD3, the method provides a superior accuracy-efficiency trade-off and is suitable for real-time ITS deployment in heterogeneous urban environments.

**arXiv ID:** 2509.00839
</details>

<details>
<summary><strong>Reinforcement Learning Driven Generalizable Feature Representation for Cross-User Activity Recognition</strong> - Xiaozhou Ye, Kevin I-Kai Wang - [[pdf]](https://arxiv.org/pdf/2509.01031)</summary>

**Abstract:** Human Activity Recognition (HAR) using wearable sensors is crucial for healthcare, fitness tracking, and smart environments, yet cross-user variability -- stemming from diverse motion patterns, sensor placements, and physiological traits -- hampers generalization in real-world settings. Conventional supervised learning methods often overfit to user-specific patterns, leading to poor performance on unseen users. Existing domain generalization approaches, while promising, frequently overlook temporal dependencies or depend on impractical domain-specific labels. We propose Temporal-Preserving Reinforcement Learning Domain Generalization (TPRL-DG), a novel framework that redefines feature extraction as a sequential decision-making process driven by reinforcement learning. TPRL-DG leverages a Transformer-based autoregressive generator to produce temporal tokens that capture user-invariant activity dynamics, optimized via a multi-objective reward function balancing class discrimination and cross-user invariance. Key innovations include: (1) an RL-driven approach for domain generalization, (2) autoregressive tokenization to preserve temporal coherence, and (3) a label-free reward design eliminating the need for target user annotations. Evaluations on the DSADS and PAMAP2 datasets show that TPRL-DG surpasses state-of-the-art methods in cross-user generalization, achieving superior accuracy without per-user calibration. By learning robust, user-invariant temporal patterns, TPRL-DG enables scalable HAR systems, facilitating advancements in personalized healthcare, adaptive fitness tracking, and context-aware environments.

**arXiv ID:** 2509.01031
</details>

<details>
<summary><strong>Investigating Tax Evasion Emergence Using Dual Large Language Model and Deep Reinforcement Learning Powered Agent-based Simulation</strong> - Teddy Lazebnik, Labib Shami - [[pdf]](https://arxiv.org/pdf/2501.18177)</summary>

**Abstract:** Tax evasion, usually the largest component of an informal economy, is a persistent challenge over history with significant socio-economic implications. Many socio-economic studies investigate its dynamics, including influencing factors, the role and influence of taxation policies, and the prediction of the tax evasion volume over time. These studies assumed such behavior is given, as observed in the real world, neglecting the "big bang" of such activity in a population. To this end, computational economy studies adopted developments in computer simulations, in general, and recent innovations in artificial intelligence (AI), in particular, to simulate and study informal economy appearance in various socio-economic settings. This study presents a novel computational framework to examine the dynamics of tax evasion and the emergence of informal economic activity. Employing an agent-based simulation powered by Large Language Models and Deep Reinforcement Learning, the framework is uniquely designed to allow informal economic behaviors to emerge organically, without presupposing their existence or explicitly signaling agents about the possibility of evasion. This provides a rigorous approach for exploring the socio-economic determinants of compliance behavior. The experimental design, comprising model validation and exploratory phases, demonstrates the framework's robustness in replicating theoretical economic behaviors. Findings indicate that individual personality traits, external narratives, enforcement probabilities, and the perceived efficiency of public goods provision significantly influence both the timing and extent of informal economic activity. The results underscore that efficient public goods provision and robust enforcement mechanisms are complementary; neither alone is sufficient to curtail informal activity effectively.

**arXiv ID:** 2501.18177
</details>

<details>
<summary><strong>In-N-Out: A Parameter-Level API Graph Dataset for Tool Agents</strong> - Seungkyu Lee, Nalim Kim, Yohan Jo - [[pdf]](https://arxiv.org/pdf/2509.01560)</summary>

**Abstract:** Tool agents -- LLM-based systems that interact with external APIs -- offer a way to execute real-world tasks. However, as tasks become increasingly complex, these agents struggle to identify and call the correct APIs in the proper order. To tackle this problem, we investigate converting API documentation into a structured API graph that captures API dependencies and leveraging it for multi-tool queries that require compositional API calls. To support this, we introduce In-N-Out, the first expert-annotated dataset of API graphs built from two real-world API benchmarks and their documentation. Using In-N-Out significantly improves performance on both tool retrieval and multi-tool query generation, nearly doubling that of LLMs using documentation alone. Moreover, graphs generated by models fine-tuned on In-N-Out close 90% of this gap, showing that our dataset helps models learn to comprehend API documentation and parameter relationships. Our findings highlight the promise of using explicit API graphs for tool agents and the utility of In-N-Out as a valuable resource. We will release the dataset and code publicly.

**arXiv ID:** 2509.01560
</details>

<details>
<summary><strong>ERank: Fusing Supervised Fine-Tuning and Reinforcement Learning for Effective and Efficient Text Reranking</strong> - Yuzheng Cai, Yanzhao Zhang, Dingkun Long, Mingxin Li, Pengjun Xie, Weiguo Zheng - [[pdf]](https://arxiv.org/pdf/2509.00520)</summary>

**Abstract:** Text reranking models are a crucial component in modern systems like Retrieval-Augmented Generation, tasked with selecting the most relevant documents prior to generation. However, current Large Language Models (LLMs) powered rerankers often face a fundamental trade-off. On one hand, Supervised Fine-Tuning based pointwise methods that frame relevance as a binary classification task lack the necessary scoring discrimination, particularly for those built on reasoning LLMs. On the other hand, approaches designed for complex reasoning often employ powerful yet inefficient listwise formulations, rendering them impractical for low latency applications. To resolve this dilemma, we introduce ERank, a highly effective and efficient pointwise reranker built from a reasoning LLM that excels across diverse relevance scenarios. We propose a novel two-stage training pipeline that begins with Supervised Fine-Tuning (SFT). In this stage, we move beyond binary labels and train the model generatively to output fine grained integer scores, which significantly enhances relevance discrimination. The model is then further refined using Reinforcement Learning (RL) with a novel, listwise derived reward. This technique instills global ranking awareness into the efficient pointwise architecture. We evaluate the ERank reranker on the BRIGHT, FollowIR, TREC DL, and BEIR benchmarks, demonstrating superior effectiveness and robustness compared to existing approaches. On the reasoning-intensive BRIGHT benchmark, our ERank-4B achieves an nDCG@10 of 38.7, while a larger 32B variant reaches a state of the art nDCG@10 of 40.2.

**arXiv ID:** 2509.00520
</details>

<details>
<summary><strong>Towards High Data Efficiency in Reinforcement Learning with Verifiable Reward</strong> - Xinyu Tang, Zhenduo Zhang, Yurou Liu, Wayne Xin Zhao, Zujie Wen, Zhiqiang Zhang, Jun Zhou - [[pdf]](https://arxiv.org/pdf/2509.01321)</summary>

**Abstract:** Recent advances in large reasoning models have leveraged reinforcement learning with verifiable rewards (RLVR) to improve reasoning capabilities. However, scaling these methods typically requires extensive rollout computation and large datasets, leading to high training costs and low data efficiency. To mitigate this issue, we propose DEPO, a Data-Efficient Policy Optimization pipeline that combines optimized strategies for both offline and online data selection. In the offline phase, we curate a high-quality subset of training samples based on diversity, influence, and appropriate difficulty. During online RLVR training, we introduce a sample-level explorability metric to dynamically filter samples with low exploration potential, thereby reducing substantial rollout computational costs. Furthermore, we incorporate a replay mechanism for under-explored samples to ensure adequate training, which enhances the model's final convergence performance. Experiments across five reasoning benchmarks show that DEPO consistently outperforms existing methods in both offline and online data selection scenarios. Notably, using only 20% of the training data, our approach achieves a 1.85 times speed-up on AIME24 and a 1.66 times speed-up on AIME25 compared to GRPO trained on the full dataset.

**arXiv ID:** 2509.01321
</details>

<details>
<summary><strong>CoRanking: Collaborative Ranking with Small and Large Ranking Agents</strong> - Wenhan Liu, Xinyu Ma, Yutao Zhu, Lixin Su, Shuaiqiang Wang, Dawei Yin, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2503.23427)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated superior listwise ranking performance. However, their superior performance often relies on large-scale parameters (\eg, GPT-4) and a repetitive sliding window process, which introduces significant efficiency challenges. In this paper, we propose \textbf{CoRanking}, a novel collaborative ranking framework that combines small and large ranking models for efficient and effective ranking. CoRanking first employs a small-size reranker to pre-rank all the candidate passages, bringing relevant ones to the top part of the list (\eg, top-20). Then, the LLM listwise reranker is applied to only rerank these top-ranked passages instead of the whole list, substantially enhancing overall ranking efficiency. Although more efficient, previous studies have revealed that the LLM listwise reranker have significant positional biases on the order of input passages. Directly feed the top-ranked passages from small reranker may result in the sub-optimal performance of LLM listwise reranker. To alleviate this problem, we introduce a passage order adjuster trained via reinforcement learning, which reorders the top passages from the small reranker to align with the LLM's preferences of passage order. Extensive experiments on three IR benchmarks demonstrate that CoRanking significantly improves efficiency (reducing ranking latency by about 70\%) while achieving even better effectiveness compared to using only the LLM listwise reranker.

**arXiv ID:** 2503.23427
</details>

<details>
<summary><strong>Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs</strong> - Yaorui Shi, Sihang Li, Chang Wu, Zhiyuan Liu, Junfeng Fang, Hengxing Cai, An Zhang, Xiang Wang - [[pdf]](https://arxiv.org/pdf/2505.11277)</summary>

**Abstract:** Large language models have demonstrated impressive reasoning capabilities but are inherently limited by their knowledge reservoir. Retrieval-augmented reasoning mitigates this limitation by allowing LLMs to query external resources, but existing methods often retrieve irrelevant or noisy information, hindering accurate reasoning. In this paper, we propose AutoRefine, a reinforcement learning post-training framework that adopts a new ``search-and-refine-during-think'' paradigm. AutoRefine introduces explicit knowledge refinement steps between successive search calls, enabling the model to iteratively filter, distill, and organize evidence before generating an answer. Furthermore, we incorporate tailored retrieval-specific rewards alongside answer correctness rewards using group relative policy optimization. Experiments on single-hop and multi-hop QA benchmarks demonstrate that AutoRefine significantly outperforms existing approaches, particularly in complex, multi-hop reasoning scenarios. Detailed analysis shows that AutoRefine issues frequent, higher-quality searches and synthesizes evidence effectively.

**arXiv ID:** 2505.11277
</details>

<details>
<summary><strong>Financial Decision Making using Reinforcement Learning with Dirichlet Priors and Quantum-Inspired Genetic Optimization</strong> - Prasun Nandy, Debjit Dhar, Rik Das - [[pdf]](https://arxiv.org/pdf/2509.00095)</summary>

**Abstract:** Traditional budget allocation models struggle with the stochastic and nonlinear nature of real-world financial data. This study proposes a hybrid reinforcement learning (RL) framework for dynamic budget allocation, enhanced with Dirichlet-inspired stochasticity and quantum mutation-based genetic optimization. Using Apple Inc. quarterly financial data (2009 to 2025), the RL agent learns to allocate budgets between Research and Development and Selling, General and Administrative to maximize profitability while adhering to historical spending patterns, with L2 penalties discouraging unrealistic deviations. A Dirichlet distribution governs state evolution to simulate shifting financial contexts. To escape local minima and improve generalization, the trained policy is refined using genetic algorithms with quantum mutation via parameterized qubit rotation circuits. Generation-wise rewards and penalties are logged to visualize convergence and policy behavior. On unseen fiscal data, the model achieves high alignment with actual allocations (cosine similarity 0.9990, KL divergence 0.0023), demonstrating the promise of combining deep RL, stochastic modeling, and quantum-inspired heuristics for adaptive enterprise budgeting.

**arXiv ID:** 2509.00095
</details>

<details>
<summary><strong>Building surrogate models using trajectories of agents trained by Reinforcement Learning</strong> - Julen Cestero, Marco Quartulli, Marcello Restelli - [[pdf]](https://arxiv.org/pdf/2509.01285)</summary>

**Abstract:** Sample efficiency in the face of computationally expensive simulations is a common concern in surrogate modeling. Current strategies to minimize the number of samples needed are not as effective in simulated environments with wide state spaces. As a response to this challenge, we propose a novel method to efficiently sample simulated deterministic environments by using policies trained by Reinforcement Learning. We provide an extensive analysis of these surrogate-building strategies with respect to Latin-Hypercube sampling or Active Learning and Kriging, cross-validating performances with all sampled datasets. The analysis shows that a mixed dataset that includes samples acquired by random agents, expert agents, and agents trained to explore the regions of maximum entropy of the state transition distribution provides the best scores through all datasets, which is crucial for a meaningful state space representation. We conclude that the proposed method improves the state-of-the-art and clears the path to enable the application of surrogate-aided Reinforcement Learning policy optimization strategies on complex simulators.

**arXiv ID:** 2509.01285
</details>

<details>
<summary><strong>The Geometry of Nonlinear Reinforcement Learning</strong> - Nikola Milosevic, Nico Scherf - [[pdf]](https://arxiv.org/pdf/2509.01432)</summary>

**Abstract:** Reward maximization, safe exploration, and intrinsic motivation are often studied as separate objectives in reinforcement learning (RL). We present a unified geometric framework, that views these goals as instances of a single optimization problem on the space of achievable long-term behavior in an environment. Within this framework, classical methods such as policy mirror descent, natural policy gradient, and trust-region algorithms naturally generalize to nonlinear utilities and convex constraints. We illustrate how this perspective captures robustness, safety, exploration, and diversity objectives, and outline open challenges at the interface of geometry and deep RL.

**arXiv ID:** 2509.01432
</details>

<details>
<summary><strong>Reinforcement Learning for Machine Learning Engineering Agents</strong> - Sherry Yang, Joy He-Yueya, Percy Liang - [[pdf]](https://arxiv.org/pdf/2509.01684)</summary>

**Abstract:** Existing agents for solving tasks such as ML engineering rely on prompting powerful language models. As a result, these agents do not improve with more experience. In this paper, we show that agents backed by weaker models that improve via reinforcement learning (RL) can outperform agents backed by much larger, but static models. We identify two major challenges with RL in this setting. First, actions can take a variable amount of time (e.g., executing code for different solutions), which leads to asynchronous policy gradient updates that favor faster but suboptimal solutions. To tackle variable-duration actions, we propose duration- aware gradient updates in a distributed asynchronous RL framework to amplify high-cost but high-reward actions. Second, using only test split performance as a reward provides limited feedback. A program that is nearly correct is treated the same as one that fails entirely. To address this, we propose environment instrumentation to offer partial credit, distinguishing almost-correct programs from those that fail early (e.g., during data loading). Environment instrumentation uses a separate static language model to insert print statement to an existing program to log the agent's experimental progress, from which partial credit can be extracted as reward signals for learning. Our experimental results on MLEBench suggest that performing gradient updates on a much smaller model (Qwen2.5-3B) trained with RL outperforms prompting a much larger model (Claude-3.5-Sonnet) with agent scaffolds, by an average of 22% across 12 Kaggle tasks.

**arXiv ID:** 2509.01684
</details>

<details>
<summary><strong>Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control</strong> - Georgios Papoudakis, Thomas Coste, Jianye Hao, Jun Wang, Kun Shao - [[pdf]](https://arxiv.org/pdf/2509.01720)</summary>

**Abstract:** Reinforcement learning (RL) using foundation models for policy approximations in multi-turn tasks remains challenging. We identify two main limitations related to sparse reward settings and policy gradient updates, based on which we formulate a key insight: updates from positive samples with high returns typically do not require policy regularisation, whereas updates from negative samples, reflecting undesirable behaviour, can harm model performance. This paper introduces Succeed or Learn Slowly (SoLS), a novel off-policy RL algorithm evaluated on mobile app control tasks. SoLS improves sample efficiency when fine-tuning foundation models for user interface navigation via a modified off-policy actor-critic approach, applying direct policy updates for positive samples and conservative, regularised updates for negative ones to prevent model degradation. We augment SoLS with Successful Transition Replay (STR), which prioritises learning from successful interactions, further improving sample efficiency. We evaluate SoLS on the AndroidWorld benchmark, where it significantly outperforms existing methods (at least 17% relative increase), including prompt-engineering and RL approaches, while requiring substantially fewer computational resources than GPT-4o-based methods with 5-60x faster inference.

**arXiv ID:** 2509.01720
</details>

<details>
<summary><strong>Goal-Conditioned Reinforcement Learning for Data-Driven Maritime Navigation</strong> - Vaishnav Vaidheeswaran, Dilith Jayakody, Samruddhi Mulay, Anand Lo, Md Mahbub Alam, Gabriel Spadon - [[pdf]](https://arxiv.org/pdf/2509.01838)</summary>

**Abstract:** Routing vessels through narrow and dynamic waterways is challenging due to changing environmental conditions and operational constraints. Existing vessel-routing studies typically fail to generalize across multiple origin-destination pairs and do not exploit large-scale, data-driven traffic graphs. In this paper, we propose a reinforcement learning solution for big maritime data that can learn to find a route across multiple origin-destination pairs while adapting to different hexagonal grid resolutions. Agents learn to select direction and speed under continuous observations in a multi-discrete action space. A reward function balances fuel efficiency, travel time, wind resistance, and route diversity, using an Automatic Identification System (AIS)-derived traffic graph with ERA5 wind fields. The approach is demonstrated in the Gulf of St. Lawrence, one of the largest estuaries in the world. We evaluate configurations that combine Proximal Policy Optimization with recurrent networks, invalid-action masking, and exploration strategies. Our experiments demonstrate that action masking yields a clear improvement in policy performance and that supplementing penalty-only feedback with positive shaping rewards produces additional gains.

**arXiv ID:** 2509.01838
</details>

<details>
<summary><strong>Semi-on-Demand Transit Feeders with Shared Autonomous Vehicles and Reinforcement-Learning-Based Zonal Dispatching Control</strong> - Max T.M. Ng, Roman Engelhardt, Florian Dandl, Hani S. Mahmassani, Klaus Bogenberger - [[pdf]](https://arxiv.org/pdf/2509.01883)</summary>

**Abstract:** This paper develops a semi-on-demand transit feeder service using shared autonomous vehicles (SAVs) and zonal dispatching control based on reinforcement learning (RL). This service combines the cost-effectiveness of fixed-route transit with the adaptability of demand-responsive transport to improve accessibility in lower-density areas. Departing from the terminus, SAVs first make scheduled fixed stops, then offer on-demand pick-ups and drop-offs in a pre-determined flexible-route area. Our deep RL model dynamically assigns vehicles to subdivided flexible-route zones in response to real-time demand fluctuations and operations, using a policy gradient algorithm - Proximal Policy Optimization. The methodology is demonstrated through agent-based simulations on a real-world bus route in Munich, Germany. Results show that after efficient training of the RL model, the semi-on-demand service with dynamic zonal control serves 16% more passengers at 13% higher generalized costs on average compared to traditional fixed-route service. The efficiency gain brought by RL control brings 2.4% more passengers at 1.4% higher costs. This study not only showcases the potential of integrating SAV feeders and machine learning techniques into public transit, but also sets the groundwork for further innovations in addressing first-mile-last-mile problems in multimodal transit systems.

**arXiv ID:** 2509.01883
</details>

<details>
<summary><strong>Deep Reinforcement Learning for Real-Time Drone Routing in Post-Disaster Road Assessment Without Domain Knowledge</strong> - Huatian Gong, Jiuh-Biing Sheu, Zheng Wang, Xiaoguang Yang, Ran Yan - [[pdf]](https://arxiv.org/pdf/2509.01886)</summary>

**Abstract:** Rapid post-disaster road damage assessment is critical for effective emergency response, yet traditional optimization methods suffer from excessive computational time and require domain knowledge for algorithm design, making them unsuitable for time-sensitive disaster scenarios. This study proposes an attention-based encoder-decoder model (AEDM) for real-time drone routing decision in post-disaster road damage assessment. The method employs deep reinforcement learning to determine high-quality drone assessment routes without requiring algorithmic design knowledge. A network transformation method is developed to convert link-based routing problems into equivalent node-based formulations, while a synthetic road network generation technique addresses the scarcity of large-scale training datasets. The model is trained using policy optimization with multiple optima (POMO) with multi-task learning capabilities to handle diverse parameter combinations. Experimental results demonstrate two key strengths of AEDM: it outperforms commercial solvers by 16--69\% in solution quality and achieves real-time inference (1--2 seconds) versus 100--2,000 seconds for traditional methods. The model exhibits strong generalization across varying problem scales, drone numbers, and time constraints, consistently outperforming baseline methods on unseen parameter distributions and real-world road networks. The proposed method effectively balances computational efficiency with solution quality, making it particularly suitable for time-critical disaster response applications where rapid decision-making is essential for saving lives.

**arXiv ID:** 2509.01886
</details>

<details>
<summary><strong>Knowledge distillation as a pathway toward next-generation intelligent ecohydrological modeling systems</strong> - Long Jiang, Yang Yang, Ting Fong May Chui, Morgan Thornwell, Hoshin Vijai Gupta - [[pdf]](https://arxiv.org/pdf/2509.01972)</summary>

**Abstract:** Simulating ecohydrological processes is essential for understanding complex environmental systems and guiding sustainable management amid accelerating climate change and human pressures. Process-based models provide physical realism but can suffer from structural rigidity, high computational costs, and complex calibration, while machine learning (ML) methods are efficient and flexible yet often lack interpretability and transferability. We propose a unified three-phase framework that integrates process-based models with ML and progressively embeds them into artificial intelligence (AI) through knowledge distillation. Phase I, behavioral distillation, enhances process models via surrogate learning and model simplification to capture key dynamics at lower computational cost. Phase II, structural distillation, reformulates process equations as modular components within a graph neural network (GNN), enabling multiscale representation and seamless integration with ML models. Phase III, cognitive distillation, embeds expert reasoning and adaptive decision-making into intelligent modeling agents using the Eyes-Brain-Hands-Mouth architecture. Demonstrations for the Samish watershed highlight the framework's applicability to ecohydrological modeling, showing that it can reproduce process-based model outputs, improve predictive accuracy, and support scenario-based decision-making. The framework offers a scalable and transferable pathway toward next-generation intelligent ecohydrological modeling systems, with the potential extension to other process-based domains.

**arXiv ID:** 2509.01972
</details>

<details>
<summary><strong>SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning</strong> - Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Xiaosen Zheng, Zejun Ma, Bo An - [[pdf]](https://arxiv.org/pdf/2509.02479)</summary>

**Abstract:** Large Language Models (LLMs) can significantly improve their reasoning capabilities by interacting with external tools, a paradigm known as Tool-Integrated Reasoning (TIR). However, extending TIR to multi-turn scenarios using Reinforcement Learning (RL) is often hindered by training instability and performance collapse. We identify that such instability is primarily caused by a distributional drift from external tool feedback, leading to the generation of low-probability tokens. This issue compounds over successive turns, causing catastrophic gradient norm explosions that derail the training process. To address this challenge, we introduce SimpleTIR , a plug-and-play algorithm that stabilizes multi-turn TIR training. Its core strategy is to identify and filter out trajectories containing void turns, i.e., turns that yield neither a code block nor a final answer. By removing these problematic trajectories from the policy update, SimpleTIR effectively blocks the harmful, high-magnitude gradients, thus stabilizing the learning dynamics. Extensive experiments show that SimpleTIR achieves state-of-the-art performance on challenging math reasoning benchmarks, notably elevating the AIME24 score from a text-only baseline of 22.1 to 50.5 when starting from the Qwen2.5-7B base model. Furthermore, by avoiding the constraints of supervised fine-tuning, SimpleTIR encourages the model to discover diverse and sophisticated reasoning patterns, such as self-correction and cross-validation.

**arXiv ID:** 2509.02479
</details>

<details>
<summary><strong>Reinforcement Learning of Dolly-In Filming Using a Ground-Based Robot</strong> - Philip Lorimer, Jack Saunders, Alan Hunter, Wenbin Li - [[pdf]](https://arxiv.org/pdf/2509.00564)</summary>

**Abstract:** Free-roaming dollies enhance filmmaking with dynamic movement, but challenges in automated camera control remain unresolved. Our study advances this field by applying Reinforcement Learning (RL) to automate dolly-in shots using free-roaming ground-based filming robots, overcoming traditional control hurdles. We demonstrate the effectiveness of combined control for precise film tasks by comparing it to independent control strategies. Our robust RL pipeline surpasses traditional Proportional-Derivative controller performance in simulation and proves its efficacy in real-world tests on a modified ROSBot 2.0 platform equipped with a camera turret. This validates our approach's practicality and sets the stage for further research in complex filming scenarios, contributing significantly to the fusion of technology with cinematic creativity. This work presents a leap forward in the field and opens new avenues for research and development, effectively bridging the gap between technological advancement and creative filmmaking.

**arXiv ID:** 2509.00564
</details>

<details>
<summary><strong>A Hybrid Input based Deep Reinforcement Learning for Lane Change Decision-Making of Autonomous Vehicle</strong> - Ziteng Gao, Jiaqi Qu, Chaoyu Chen - [[pdf]](https://arxiv.org/pdf/2509.01611)</summary>

**Abstract:** Lane change decision-making for autonomous vehicles is a complex but high-reward behavior. In this paper, we propose a hybrid input based deep reinforcement learning (DRL) algorithm, which realizes abstract lane change decisions and lane change actions for autonomous vehicles within traffic flow. Firstly, a surrounding vehicles trajectory prediction method is proposed to reduce the risk of future behavior of surrounding vehicles to ego vehicle, and the prediction results are input into the reinforcement learning model as additional information. Secondly, to comprehensively leverage environmental information, the model extracts feature from high-dimensional images and low-dimensional sensor data simultaneously. The fusion of surrounding vehicle trajectory prediction and multi-modal information are used as state space of reinforcement learning to improve the rationality of lane change decision. Finally, we integrate reinforcement learning macro decisions with end-to-end vehicle control to achieve a holistic lane change process. Experiments were conducted within the CARLA simulator, and the results demonstrated that the utilization of a hybrid state space significantly enhances the safety of vehicle lane change decisions.

**arXiv ID:** 2509.01611
</details>

<details>
<summary><strong>Non-conflicting Energy Minimization in Reinforcement Learning based Robot Control</strong> - Skand Peri, Akhil Perincherry, Bikram Pandit, Stefan Lee - [[pdf]](https://arxiv.org/pdf/2509.01765)</summary>

**Abstract:** Efficient robot control often requires balancing task performance with energy expenditure. A common approach in reinforcement learning (RL) is to penalize energy use directly as part of the reward function. This requires carefully tuning weight terms to avoid undesirable trade-offs where energy minimization harms task success. In this work, we propose a hyperparameter-free gradient optimization method to minimize energy expenditure without conflicting with task performance. Inspired by recent works in multitask learning, our method applies policy gradient projection between task and energy objectives to derive policy updates that minimize energy expenditure in ways that do not impact task performance. We evaluate this technique on standard locomotion benchmarks of DM-Control and HumanoidBench and demonstrate a reduction of 64% energy usage while maintaining comparable task performance. Further, we conduct experiments on a Unitree GO2 quadruped showcasing Sim2Real transfer of energy efficient policies. Our method is easy to implement in standard RL pipelines with minimal code changes, is applicable to any policy gradient method, and offers a principled alternative to reward shaping for energy efficient control policies.

**arXiv ID:** 2509.01765
</details>

<details>
<summary><strong>AutoDrive-R$^2$: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving</strong> - Zhenlong Yuan, Jing Tang, Jinguo Luo, Rui Chen, Chengxuan Qian, Lei Sun, Xiangxiang Chu, Yujun Cai, Dapeng Zhang, Shuo Li - [[pdf]](https://arxiv.org/pdf/2509.01944)</summary>

**Abstract:** Vision-Language-Action (VLA) models in autonomous driving systems have recently demonstrated transformative potential by integrating multimodal perception with decision-making capabilities. However, the interpretability and coherence of the decision process and the plausibility of action sequences remain largely underexplored. To address these issues, we propose AutoDrive-R$^2$, a novel VLA framework that enhances both reasoning and self-reflection capabilities of autonomous driving systems through chain-of-thought (CoT) processing and reinforcement learning (RL). Specifically, we first propose an innovative CoT dataset named nuScenesR$^2$-6K for supervised fine-tuning, which effectively builds cognitive bridges between input information and output trajectories through a four-step logical chain with self-reflection for validation. Moreover, to maximize both reasoning and self-reflection during the RL stage, we further employ the Group Relative Policy Optimization (GRPO) algorithm within a physics-grounded reward framework that incorporates spatial alignment, vehicle dynamic, and temporal smoothness criteria to ensure reliable and realistic trajectory planning. Extensive evaluation results across both nuScenes and Waymo datasets demonstrates the state-of-the-art performance and robust generalization capacity of our proposed method.

**arXiv ID:** 2509.01944
</details>

<details>
<summary><strong>YORI: Autonomous Cooking System Utilizing a Modular Robotic Kitchen and a Dual-Arm Proprioceptive Manipulator</strong> - Donghun Noh, Hyunwoo Nam, Kyle Gillespie, Yeting Liu, Dennis Hong - [[pdf]](https://arxiv.org/pdf/2405.11094)</summary>

**Abstract:** This paper presents Yummy Operations Robot Initiative (YORI), a proprioceptive dual-arm robotic system that demonstrates autonomous multi-dish cooking for scalable food service applications. YORI integrates a dual-arm manipulator equipped with proprioceptive actuators, custom-designed tools, appliances, and a structured kitchen environment to address the complexities of cooking tasks. The proprioceptive actuators enable fast, precise, force-controlled movements while mitigating the risks associated with cooking-related impacts. The system's modular kitchen design and flexible tool-changing mechanism support simultaneous multi-dish preparation through torque control and optimization-based motion planning and scheduling. A comprehensive scheduling framework with dynamic rescheduling ensures reliable adaptation to new orders and delays. The system was publicly validated through live demonstrations, reliably preparing steak-frites across multiple convention sessions. This paper details YORI's design and explores future directions in kitchen optimization, task planning, and food quality control, demonstrating its potential as a scalable robotic cooking solution. A system introduction and cooking videos are available online

**arXiv ID:** 2405.11094
</details>

<details>
<summary><strong>Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids</strong> - Toru Lin, Kartik Sachdev, Linxi Fan, Jitendra Malik, Yuke Zhu - [[pdf]](https://arxiv.org/pdf/2502.20396)</summary>

**Abstract:** Learning generalizable robot manipulation policies, especially for complex multi-fingered humanoids, remains a significant challenge. Existing approaches primarily rely on extensive data collection and imitation learning, which are expensive, labor-intensive, and difficult to scale. Sim-to-real reinforcement learning (RL) offers a promising alternative, but has mostly succeeded in simpler state-based or single-hand setups. How to effectively extend this to vision-based, contact-rich bimanual manipulation tasks remains an open question. In this paper, we introduce a practical sim-to-real RL recipe that trains a humanoid robot to perform three challenging dexterous manipulation tasks: grasp-and-reach, box lift and bimanual handover. Our method features an automated real-to-sim tuning module, a generalized reward formulation based on contact and object goals, a divide-and-conquer policy distillation framework, and a hybrid object representation strategy with modality-specific augmentation. We demonstrate high success rates on unseen objects and robust, adaptive policy behaviors -- highlighting that vision-based dexterous manipulation via sim-to-real RL is not only viable, but also scalable and broadly applicable to real-world humanoid manipulation tasks.

**arXiv ID:** 2502.20396
</details>

<details>
<summary><strong>RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning</strong> - Liam Boyle, Nicolas Baumann, Paviththiren Sivasothilingam, Michele Magno, Luca Benini - [[pdf]](https://arxiv.org/pdf/2505.03238)</summary>

**Abstract:** Future robotic systems operating in real-world environments will require on-board embodied intelligence without continuous cloud connection, balancing capabilities with constraints on computational power and memory. This work presents an extension of the R1-zero approach, which enables the usage of low parameter-count Large Language Models (LLMs) in the robotic domain. The R1-Zero approach was originally developed to enable mathematical reasoning in LLMs using static datasets. We extend it to the robotics domain through integration in a closed-loop Reinforcement Learning (RL) framework. This extension enhances reasoning in Embodied Artificial Intelligence (Embodied AI) settings without relying solely on distillation of large models through Supervised Fine-Tuning (SFT). We show that small-scale LLMs can achieve effective reasoning performance by learning through closed-loop interaction with their environment, which enables tasks that previously required significantly larger models. In an autonomous driving setting, a performance gain of 20.2%-points over the SFT-based baseline is observed with a Qwen2.5-1.5B model. Using the proposed training procedure, Qwen2.5-3B achieves a 63.3% control adaptability score, surpassing the 58.5% obtained by the much larger, cloud-bound GPT-4o. These results highlight that practical, on-board deployment of small LLMs is not only feasible but can outperform larger models if trained through environmental feedback, underscoring the importance of an interactive learning framework for robotic Embodied AI, one grounded in practical experience rather than static supervision.

**arXiv ID:** 2505.03238
</details>

<details>
<summary><strong>Morphologically Symmetric Reinforcement Learning for Ambidextrous Bimanual Manipulation</strong> - Zechu Li, Yufeng Jin, Daniel Ordonez Apraez, Claudio Semini, Puze Liu, Georgia Chalvatzaki - [[pdf]](https://arxiv.org/pdf/2505.05287)</summary>

**Abstract:** Humans naturally exhibit bilateral symmetry in their gross manipulation skills, effortlessly mirroring simple actions between left and right hands. Bimanual robots-which also feature bilateral symmetry-should similarly exploit this property to perform tasks with either hand. Unlike humans, who often favor a dominant hand for fine dexterous skills, robots should ideally execute ambidextrous manipulation with equal proficiency. To this end, we introduce SYMDEX (SYMmetric DEXterity), a reinforcement learning framework for ambidextrous bi-manipulation that leverages the robot's inherent bilateral symmetry as an inductive bias. SYMDEX decomposes complex bimanual manipulation tasks into per-hand subtasks and trains dedicated policies for each. By exploiting bilateral symmetry via equivariant neural networks, experience from one arm is inherently leveraged by the opposite arm. We then distill the subtask policies into a global ambidextrous policy that is independent of the hand-task assignment. We evaluate SYMDEX on six challenging simulated manipulation tasks and demonstrate successful real-world deployment on two of them. Our approach strongly outperforms baselines on complex task in which the left and right hands perform different roles. We further demonstrate SYMDEX's scalability by extending it to a four-arm manipulation setup, where our symmetry-aware policies enable effective multi-arm collaboration and coordination. Our results highlight how structural symmetry as inductive bias in policy learning enhances sample efficiency, robustness, and generalization across diverse dexterous manipulation tasks.

**arXiv ID:** 2505.05287
</details>

<details>
<summary><strong>HERMES: Human-to-Robot Embodied Learning from Multi-Source Motion Data for Mobile Dexterous Manipulation</strong> - Zhecheng Yuan, Tianming Wei, Langzhe Gu, Pu Hua, Tianhai Liang, Yuanpei Chen, Huazhe Xu - [[pdf]](https://arxiv.org/pdf/2508.20085)</summary>

**Abstract:** Leveraging human motion data to impart robots with versatile manipulation skills has emerged as a promising paradigm in robotic manipulation. Nevertheless, translating multi-source human hand motions into feasible robot behaviors remains challenging, particularly for robots equipped with multi-fingered dexterous hands characterized by complex, high-dimensional action spaces. Moreover, existing approaches often struggle to produce policies capable of adapting to diverse environmental conditions. In this paper, we introduce HERMES, a human-to-robot learning framework for mobile bimanual dexterous manipulation. First, HERMES formulates a unified reinforcement learning approach capable of seamlessly transforming heterogeneous human hand motions from multiple sources into physically plausible robotic behaviors. Subsequently, to mitigate the sim2real gap, we devise an end-to-end, depth image-based sim2real transfer method for improved generalization to real-world scenarios. Furthermore, to enable autonomous operation in varied and unstructured environments, we augment the navigation foundation model with a closed-loop Perspective-n-Point (PnP) localization mechanism, ensuring precise alignment of visual goals and effectively bridging autonomous navigation and dexterous manipulation. Extensive experimental results demonstrate that HERMES consistently exhibits generalizable behaviors across diverse, in-the-wild scenarios, successfully performing numerous complex mobile bimanual dexterous manipulation tasks. Project Page:this https URL.

**arXiv ID:** 2508.20085
</details>

<details>
<summary><strong>Goal-Conditioned Data Augmentation for Offline Reinforcement Learning</strong> - Xingshuai Huang, Di Wu, Benoit Boulet - [[pdf]](https://arxiv.org/pdf/2412.20519)</summary>

**Abstract:** Offline reinforcement learning (RL) enables policy learning from pre-collected offline datasets, relaxing the need to interact directly with the environment. However, limited by the quality of offline datasets, it generally fails to learn well-qualified policies in suboptimal datasets. To address datasets with insufficient optimal demonstrations, we introduce Goal-cOnditioned Data Augmentation (GODA), a novel goal-conditioned diffusion-based method for augmenting samples with higher quality. Leveraging recent advancements in generative modelling, GODA incorporates a novel return-oriented goal condition with various selection mechanisms. Specifically, we introduce a controllable scaling technique to provide enhanced return-based guidance during data sampling. GODA learns a comprehensive distribution representation of the original offline datasets while generating new data with selectively higher-return goals, thereby maximizing the utility of limited optimal demonstrations. Furthermore, we propose a novel adaptive gated conditioning method for processing noisy inputs and conditions, enhancing the capture of goal-oriented guidance. We conduct experiments on the D4RL benchmark and real-world challenges, specifically traffic signal control (TSC) tasks, to demonstrate GODA's effectiveness in enhancing data quality and superior performance compared to state-of-the-art data augmentation methods across various offline RL algorithms.

**arXiv ID:** 2412.20519
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
