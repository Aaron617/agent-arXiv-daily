# Agent arXiv Daily

**Last Updated:** 2025-11-07 02:07:32

**Total Papers:** 63

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
<summary><strong>Digital Transformation Chatbot (DTchatbot): Integrating Large Language Model-based Chatbot in Acquiring Digital Transformation Needs</strong> - Jiawei Zheng, Gokcen Yilmaz, Ji Han, Saeema Ahmed-Kristensen - [[pdf]](https://arxiv.org/pdf/2511.02842)</summary>

**Abstract:** Many organisations pursue digital transformation to enhance operational efficiency, reduce manual efforts, and optimise processes by automation and digital tools. To achieve this, a comprehensive understanding of their unique needs is required. However, traditional methods, such as expert interviews, while effective, face several challenges, including scheduling conflicts, resource constraints, inconsistency, etc. To tackle these issues, we investigate the use of a Large Language Model (LLM)-powered chatbot to acquire organisations' digital transformation needs. Specifically, the chatbot integrates workflow-based instruction with LLM's planning and reasoning capabilities, enabling it to function as a virtual expert and conduct interviews. We detail the chatbot's features and its implementation. Our preliminary evaluation indicates that the chatbot performs as designed, effectively following predefined workflows and supporting user interactions with areas for improvement. We conclude by discussing the implications of employing chatbots to elicit user information, emphasizing their potential and limitations.

**arXiv ID:** 2511.02842
</details>

<details>
<summary><strong>Depth Matters: Multimodal RGB-D Perception for Robust Autonomous Agents</strong> - Mihaela-Larisa Clement, Mónika Farsang, Felix Resch, Mihai-Teodor Stanusoiu, Radu Grosu - [[pdf]](https://arxiv.org/pdf/2503.16711)</summary>

**Abstract:** Autonomous agents that rely purely on perception to make real-time control decisions require efficient and robust architectures. In this work, we demonstrate that augmenting RGB input with depth information significantly enhances our agents' ability to predict steering commands compared to using RGB alone. We benchmark lightweight recurrent controllers that leverage the fused RGB-D features for sequential decision-making. To train our models, we collect high-quality data using a small-scale autonomous car controlled by an expert driver via a physical steering wheel, capturing varying levels of steering difficulty. Our models were successfully deployed on real hardware and inherently avoided dynamic and static obstacles, under out-of-distribution conditions. Specifically, our findings reveal that the early fusion of depth data results in a highly robust controller, which remains effective even with frame drops and increased noise levels, without compromising the network's focus on the task.

**arXiv ID:** 2503.16711
</details>

<details>
<summary><strong>The Rise of AI Companions: How Human-Chatbot Relationships Influence Well-Being</strong> - Yutong Zhang, Dora Zhao, Jeffrey T. Hancock, Robert Kraut, Diyi Yang - [[pdf]](https://arxiv.org/pdf/2506.12605)</summary>

**Abstract:** As large language models (LLMs)-enhanced chatbots grow increasingly expressive and socially responsive, many users are beginning to form companionship-like bonds with them, particularly with simulated AI partners designed to mimic emotionally attuned interlocutors. These emerging AI companions raise critical questions: Can such systems fulfill social needs typically met by human relationships? How do they shape psychological well-being? And what new risks arise as users develop emotional ties to non-human agents? This study investigates how people interact with AI companions, especially simulated partners on CharacterAI, and how this use is associated with users' psychological well-being. We analyzed survey data from 1,131 users and 4,363 chat sessions (413,509 messages) donated by 244 participants, focusing on three dimensions of use: nature of the interaction, interaction intensity, and self-disclosure. By triangulating self-reports primary motivation, open-ended relationship descriptions, and annotated chat transcripts, we identify patterns in how users engage with AI companions and its associations with well-being. Findings suggest that people with smaller social networks are more likely to turn to chatbots for companionship, but that companionship-oriented chatbot usage is consistently associated with lower well-being, particularly when people use the chatbots more intensively, engage in higher levels of self-disclosure, and lack strong human social support. Even though some people turn to chatbots to fulfill social needs, these uses of chatbots do not fully substitute for human connection. As a result, the psychological benefits may be limited, and the relationship could pose risks for more socially isolated or emotionally vulnerable users.

**arXiv ID:** 2506.12605
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (11 papers)</h2></summary>

<details>
<summary><strong>Evaluating Control Protocols for Untrusted AI Agents</strong> - Jon Kutasov, Chloe Loughridge, Yuqi Sun, Henry Sleight, Buck Shlegeris, Tyler Tracy, Joe Benton - [[pdf]](https://arxiv.org/pdf/2511.02997)</summary>

**Abstract:** As AI systems become more capable and widely deployed as agents, ensuring their safe operation becomes critical. AI control offers one approach to mitigating the risk from untrusted AI agents by monitoring their actions and intervening or auditing when necessary. Evaluating the safety of these protocols requires understanding both their effectiveness against current attacks and their robustness to adaptive adversaries. In this work, we systematically evaluate a range of control protocols in SHADE-Arena, a dataset of diverse agentic environments. First, we evaluate blue team protocols, including deferral to trusted models, resampling, and deferring on critical actions, against a default attack policy. We find that resampling for incrimination and deferring on critical actions perform best, increasing safety from 50% to 96%. We then iterate on red team strategies against these protocols and find that attack policies with additional affordances, such as knowledge of when resampling occurs or the ability to simulate monitors, can substantially improve attack success rates against our resampling strategy, decreasing safety to 17%. However, deferring on critical actions is highly robust to even our strongest red team strategies, demonstrating the importance of denying attack policies access to protocol internals.

**arXiv ID:** 2511.02997
</details>

<details>
<summary><strong>A Proprietary Model-Based Safety Response Framework for AI Agents</strong> - Qi Li, Jianjun Xu, Pingtao Wei, Jiu Li, Peiqiang Zhao, Jiwei Shi, Xuan Zhang, Yanhui Yang, Xiaodong Hui, Peng Xu, Wenqin Shao - [[pdf]](https://arxiv.org/pdf/2511.03138)</summary>

**Abstract:** With the widespread application of Large Language Models (LLMs), their associated security issues have become increasingly prominent, severely constraining their trustworthy deployment in critical domains. This paper proposes a novel safety response framework designed to systematically safeguard LLMs at both the input and output levels. At the input level, the framework employs a supervised fine-tuning-based safety classification model. Through a fine-grained four-tier taxonomy (Safe, Unsafe, Conditionally Safe, Focused Attention), it performs precise risk identification and differentiated handling of user queries, significantly enhancing risk coverage and business scenario adaptability, and achieving a risk recall rate of 99.3%. At the output level, the framework integrates Retrieval-Augmented Generation (RAG) with a specifically fine-tuned interpretation model, ensuring all responses are grounded in a real-time, trustworthy knowledge base. This approach eliminates information fabrication and enables result traceability. Experimental results demonstrate that our proposed safety control model achieves a significantly higher safety score on public safety evaluation benchmarks compared to the baseline model, TinyR1-Safety-8B. Furthermore, on our proprietary high-risk test set, the framework's components attained a perfect 100% safety score, validating their exceptional protective capabilities in complex risk scenarios. This research provides an effective engineering pathway for building high-security, high-trust LLM applications.

**arXiv ID:** 2511.03138
</details>

<details>
<summary><strong>From Measurement to Expertise: Empathetic Expert Adapters for Context-Based Empathy in Conversational AI Agents</strong> - Erfan Shayegani, Jina Suh, Andy Wilson, Nagu Rangan, Javier Hernandez - [[pdf]](https://arxiv.org/pdf/2511.03143)</summary>

**Abstract:** Empathy is a critical factor in fostering positive user experiences in conversational AI. While models can display empathy, it is often generic rather than tailored to specific tasks and contexts. In this work, we introduce a novel framework for developing and evaluating context-specific empathetic large language models (LLMs). We first analyze a real-world conversational dataset consisting of 672 multi-turn conversations across 8 tasks, revealing significant differences in terms of expected and experienced empathy before and after the conversations, respectively. To help minimize this gap, we develop a synthetic multi-turn conversational generation pipeline and steer responses toward our defined empathy patterns based on the context that more closely matches users' expectations. We then train empathetic expert adapters for context-specific empathy that specialize in varying empathy levels based on the recognized task. Our empirical results demonstrate a significant gap reduction of 72.66% between perceived and desired empathy with scores increasing by an average factor of 2.43 as measured by our metrics and reward models. Additionally, our trained empathetic expert adapters demonstrate superior effectiveness in preserving empathy patterns throughout conversation turns, outperforming system prompts, which tend to dramatically diminish in impact as conversations lengthen.

**arXiv ID:** 2511.03143
</details>

<details>
<summary><strong>The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents</strong> - Xingyao Wang, Simon Rosenberg, Juan Michelini, Calvin Smith, Hoang Tran, Engel Nyst, Rohit Malhotra, Xuhui Zhou, Valerie Chen, Robert Brennan, Graham Neubig - [[pdf]](https://arxiv.org/pdf/2511.03690)</summary>

**Abstract:** Agents are now used widely in the process of software development, but building production-ready software engineering agents is a complex task. Deploying software agents effectively requires flexibility in implementation and experimentation, reliable and secure execution, and interfaces for users to interact with agents. In this paper, we present the OpenHands Software Agent SDK, a toolkit for implementing software development agents that satisfy these desiderata. This toolkit is a complete architectural redesign of the agent components of the popular OpenHands framework for software development agents, which has 64k+ GitHub stars. To achieve flexibility, we design a simple interface for implementing agents that requires only a few lines of code in the default case, but is easily extensible to more complex, full-featured agents with features such as custom tools, memory management, and more. For security and reliability, it delivers seamless local-to-remote execution portability, integrated REST/WebSocket services. For interaction with human users, it can connect directly to a variety of interfaces, such as visual workspaces (VS Code, VNC, browser), command-line interfaces, and APIs. Compared with existing SDKs from OpenAI, Claude, and Google, OpenHands uniquely integrates native sandboxed execution, lifecycle control, model-agnostic multi-LLM routing, and built-in security analysis. Empirical results on SWE-Bench Verified and GAIA benchmarks demonstrate strong performance. Put together, these elements allow the OpenHands Software Agent SDK to provide a practical foundation for prototyping, unlocking new classes of custom applications, and reliably deploying agents at scale.

**arXiv ID:** 2511.03690
</details>

<details>
<summary><strong>Misalignment Bounty: Crowdsourcing AI Agent Misbehavior</strong> - Rustem Turtayev, Natalia Fedorova, Oleg Serikov, Sergey Koldyba, Lev Avagyan, Dmitrii Volkov - [[pdf]](https://arxiv.org/pdf/2510.19738)</summary>

**Abstract:** Advanced AI systems sometimes act in ways that differ from human intent. To gather clear, reproducible examples, we ran the Misalignment Bounty: a crowdsourced project that collected cases of agents pursuing unintended or unsafe goals. The bounty received 295 submissions, of which nine were awarded.
This report explains the program's motivation and evaluation criteria, and walks through the nine winning submissions step by step.

**arXiv ID:** 2510.19738
</details>

<details>
<summary><strong>Kosmos: An AI Scientist for Autonomous Discovery</strong> - Ludovico Mitchener, Angela Yiu, Benjamin Chang, Mathieu Bourdenx, Tyler Nadolski, Arvis Sulovari, Eric C. Landsness, Daniel L. Barabasi, Siddharth Narayanan, Nicky Evans, Shriya Reddy, Martha Foiani, Aizad Kamal, Leah P. Shriver, Fang Cao, Asmamaw T. Wassie, Jon M. Laurent, Edwin Melville-Green, Mayk Caldas, Albert Bou, Kaleigh F. Roberts, Sladjana Zagorac, Timothy C. Orr, Miranda E. Orr, Kevin J. Zwezdaryk, Ali E. Ghareeb, Laurie McCoy, Bruna Gomes, Euan A. Ashley, Karen E. Duff, Tonio Buonassisi, Tom Rainforth, Randall J. Bateman, Michael Skarlinski, Samuel G. Rodriques, Michaela M. Hinks, Andrew D. White - [[pdf]](https://arxiv.org/pdf/2511.02824)</summary>

**Abstract:** Data-driven scientific discovery requires iterative cycles of literature search, hypothesis generation, and data analysis. Substantial progress has been made towards AI agents that can automate scientific research, but all such agents remain limited in the number of actions they can take before losing coherence, thus limiting the depth of their findings. Here we present Kosmos, an AI scientist that automates data-driven discovery. Given an open-ended objective and a dataset, Kosmos runs for up to 12 hours performing cycles of parallel data analysis, literature search, and hypothesis generation before synthesizing discoveries into scientific reports. Unlike prior systems, Kosmos uses a structured world model to share information between a data analysis agent and a literature search agent. The world model enables Kosmos to coherently pursue the specified objective over 200 agent rollouts, collectively executing an average of 42,000 lines of code and reading 1,500 papers per run. Kosmos cites all statements in its reports with code or primary literature, ensuring its reasoning is traceable. Independent scientists found 79.4% of statements in Kosmos reports to be accurate, and collaborators reported that a single 20-cycle Kosmos run performed the equivalent of 6 months of their own research time on average. Furthermore, collaborators reported that the number of valuable scientific findings generated scales linearly with Kosmos cycles (tested up to 20 cycles). We highlight seven discoveries made by Kosmos that span metabolomics, materials science, neuroscience, and statistical genetics. Three discoveries independently reproduce findings from preprinted or unpublished manuscripts that were not accessed by Kosmos at runtime, while four make novel contributions to the scientific literature.

**arXiv ID:** 2511.02824
</details>

<details>
<summary><strong>SecRepoBench: Benchmarking Code Agents for Secure Code Completion in Real-World Repositories</strong> - Chihao Shen, Connor Dilgren, Purva Chiniya, Luke Griffith, Yu Ding, Yizheng Chen - [[pdf]](https://arxiv.org/pdf/2504.21205)</summary>

**Abstract:** This paper introduces SecRepoBench, a benchmark to evaluate code agents on secure code completion in real-world repositories. SecRepoBench has 318 code completion tasks in 27 C/C++ repositories, covering 15 CWEs. We evaluate 28 standalone LLMs and 13 code agents across 3 state-of-the-art agent frameworks using our benchmark. We find that state-of-the-art LLMs struggle with generating correct and secure code completions. However, code agents significantly outperform standalone LLMs. We show that SecRepoBench is more difficult than the prior state-of-the-art benchmark. Finally, our comprehensive analysis provides insights into potential directions for enhancing the ability of code agents to write correct and secure code in real-world repositories.

**arXiv ID:** 2504.21205
</details>

<details>
<summary><strong>GDS Agent for Graph Algorithmic Reasoning</strong> - Borun Shi, Ioannis Panagiotas - [[pdf]](https://arxiv.org/pdf/2508.20637)</summary>

**Abstract:** Large language models (LLMs) have shown remarkable multimodal information processing and reasoning ability. When equipped with tools through function calling and enhanced with retrieval-augmented techniques, compound LLM-based systems can access closed data sources and answer questions about them. However, they still struggle to process and reason over large-scale graph-structure data. We introduce the GDS (Graph Data Science) agent in this technical report. The GDS agent introduces a comprehensive set of graph algorithms as tools, together with preprocessing (retrieval) and postprocessing of algorithm results, in a model context protocol (MCP) server. The server can be used with any modern LLM out-of-the-box. GDS agent allows users to ask any question that implicitly and intrinsically requires graph algorithmic reasoning about their data, and quickly obtain accurate and grounded answers. We introduce new benchmarks that evaluate intermediate tool calls as well as final responses. The results indicate that GDS agent is able to solve a wide spectrum of graph tasks. We also provide detailed case studies for more open-ended tasks and study scenarios where the agent struggles. Finally, we discuss the remaining challenges and the future roadmap.

**arXiv ID:** 2508.20637
</details>

<details>
<summary><strong>LEGO-Eval: Towards Fine-Grained Evaluation on Synthesizing 3D Embodied Environments with Tool Augmentation</strong> - Gyeom Hwangbo, Hyungjoo Chae, Minseok Kang, Hyeonjong Ju, Soohyun Oh, Jinyoung Yeo - [[pdf]](https://arxiv.org/pdf/2511.03001)</summary>

**Abstract:** Despite recent progress in using Large Language Models (LLMs) for automatically generating 3D scenes, generated scenes often lack realistic spatial layouts and object attributes found in real-world environments. As this problem stems from insufficiently detailed, coarse-grained instructions, advancing 3D scene synthesis guided by more detailed, fine-grained instructions that reflect real-world environments becomes crucial. Without such realistic scenes, training embodied agents in unrealistic environments can lead them to learn priors that diverge significantly from real-world physics and semantics, degrading their performance when deployed. Thus, verifying the alignment between the fine-grained instruction and the generated scene is essential for effective learning. However, current evaluation methods, such as CLIPScore and vision-language models (VLMs), often fail to reliably assess such alignment. This shortcoming arises primarily from their shallow understanding of 3D scenes, which often leads to improperly grounded scene components. To address this, we introduce LEGO-Eval, an evaluation framework equipped with diverse tools designed to explicitly ground scene components, enabling more accurate alignment assessments. We also present LEGO-Bench, a benchmark of detailed instructions that specify complex layouts and attributes of real-world environments. Experiments demonstrate that LEGO-Eval outperforms VLM-as-a-judge by 0.41 F1 score in assessing scene-instruction alignment. Benchmarking with LEGO-Bench reveals significant limitations in current generation methods. Across all evaluated approaches, success rates reached at most 10% in generating scenes that fully align with fine-grained instructions.

**arXiv ID:** 2511.03001
</details>

<details>
<summary><strong>A Modular, Data-Free Pipeline for Multi-Label Intention Recognition in Transportation Agentic AI Applications</strong> - Xiaocai Zhang, Hur Lim, Ke Wang, Zhe Xiao, Jing Wang, Kelvin Lee, Xiuju Fu, Zheng Qin - [[pdf]](https://arxiv.org/pdf/2511.03363)</summary>

**Abstract:** In this study, a modular, data-free pipeline for multi-label intention recognition is proposed for agentic AI applications in transportation. Unlike traditional intent recognition systems that depend on large, annotated corpora and often struggle with fine-grained, multi-label discrimination, our approach eliminates the need for costly data collection while enhancing the accuracy of multi-label intention understanding. Specifically, the overall pipeline, named DMTC, consists of three steps: 1) using prompt engineering to guide large language models (LLMs) to generate diverse synthetic queries in different transport scenarios; 2) encoding each textual query with a Sentence-T5 model to obtain compact semantic embeddings; 3) training a lightweight classifier using a novel online focal-contrastive (OFC) loss that emphasizes hard samples and maximizes inter-class separability. The applicability of the proposed pipeline is demonstrated in an agentic AI application in the maritime transportation context. Extensive experiments show that DMTC achieves a Hamming loss of 5.35% and an AUC of 95.92%, outperforming state-of-the-art multi-label classifiers and recent end-to-end SOTA LLM-based baselines. Further analysis reveals that Sentence-T5 embeddings improve subset accuracy by at least 3.29% over alternative encoders, and integrating the OFC loss yields an additional 0.98% gain compared to standard contrastive objectives. In conclusion, our system seamlessly routes user queries to task-specific modules (e.g., ETA information, traffic risk evaluation, and other typical scenarios in the transportation domain), laying the groundwork for fully autonomous, intention-aware agents without costly manual labelling.

**arXiv ID:** 2511.03363
</details>

<details>
<summary><strong>NaviTrace: Evaluating Embodied Navigation of Vision-Language Models</strong> - Tim Windecker, Manthan Patel, Moritz Reuss, Richard Schwarzkopf, Cesar Cadena, Rudolf Lioutikov, Marco Hutter, Jonas Frey - [[pdf]](https://arxiv.org/pdf/2510.26909)</summary>

**Abstract:** Vision-language models demonstrate unprecedented performance and generalization across a wide range of tasks and scenarios. Integrating these foundation models into robotic navigation systems opens pathways toward building general-purpose robots. Yet, evaluating these models' navigation capabilities remains constrained by costly real-world trials, overly simplified simulations, and limited benchmarks. We introduce NaviTrace, a high-quality Visual Question Answering benchmark where a model receives an instruction and embodiment type (human, legged robot, wheeled robot, bicycle) and must output a 2D navigation trace in image space. Across 1000 scenarios and more than 3000 expert traces, we systematically evaluate eight state-of-the-art VLMs using a newly introduced semantic-aware trace score. This metric combines Dynamic Time Warping distance, goal endpoint error, and embodiment-conditioned penalties derived from per-pixel semantics and correlates with human preferences. Our evaluation reveals consistent gap to human performance caused by poor spatial grounding and goal localization. NaviTrace establishes a scalable and reproducible benchmark for real-world robotic navigation. The benchmark and leaderboard can be found at this https URL.

**arXiv ID:** 2510.26909
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>TAMO: Fine-Grained Root Cause Analysis via Tool-Assisted LLM Agent with Multi-Modality Observation Data in Cloud-Native Systems</strong> - Xiao Zhang, Qi Wang, Mingyi Li, Yuan Yuan, Mengbai Xiao, Fuzhen Zhuang, Dongxiao Yu - [[pdf]](https://arxiv.org/pdf/2504.20462)</summary>

**Abstract:** Implementing large language models (LLMs)-driven root cause analysis (RCA) in cloud-native systems has become a key topic of modern software operations and maintenance. However, existing LLM-based approaches face three key challenges: multi-modality input constraint, context window limitation, and dynamic dependence graph. To address these issues, we propose a tool-assisted LLM agent with multi-modality observation data for fine-grained RCA, namely TAMO, including multimodality alignment tool, root cause localization tool, and fault types classification tool. In detail, TAMO unifies multi-modal observation data into time-aligned representations for cross-modal feature consistency. Based on the unified representations, TAMO then invokes its specialized root cause localization tool and fault types classification tool for further identifying root cause and fault type underlying system context. This approach overcomes the limitations of LLMs in processing real-time raw observational data and dynamic service dependencies, guiding the model to generate repair strategies that align with system context through structured prompt design. Experiments on two benchmark datasets demonstrate that TAMO outperforms state-of-the-art (SOTA) approaches with comparable performance.

**arXiv ID:** 2504.20462
</details>

<details>
<summary><strong>Distilling LLM Agent into Small Models with Retrieval and Code Tools</strong> - Minki Kang, Jongwon Jeong, Seanie Lee, Jaewoong Cho, Sung Ju Hwang - [[pdf]](https://arxiv.org/pdf/2505.17612)</summary>

**Abstract:** Large language models (LLMs) excel at complex reasoning tasks but remain computationally expensive, limiting their practical deployment. To address this, recent works have focused on distilling reasoning capabilities into smaller language models (sLMs) using chain-of-thought (CoT) traces from teacher LLMs. However, this approach struggles in scenarios requiring rare factual knowledge or precise computation, where sLMs often hallucinate due to limited capability. In this work, we propose Agent Distillation, a framework for transferring not only reasoning capability but full task-solving behavior from LLM-based agents into sLMs with retrieval and code tools. We improve agent distillation along two complementary axes: (1) we introduce a prompting method called first-thought prefix to enhance the quality of teacher-generated trajectories; and (2) we propose a self-consistent action generation for improving test-time robustness of small agents. We evaluate our method on eight reasoning tasks across factual and mathematical domains, covering both in-domain and out-of-domain generalization. Our results show that sLMs as small as 0.5B, 1.5B, 3B parameters can achieve performance competitive with next-tier larger 1.5B, 3B, 7B models fine-tuned using CoT distillation, demonstrating the potential of agent distillation for building practical, tool-using small agents. Our code is available at this https URL.

**arXiv ID:** 2505.17612
</details>

<details>
<summary><strong>Cache Mechanism for Agent RAG Systems</strong> - Shuhang Lin, Zhencan Peng, Lingyao Li, Xiao Lin, Xi Zhu, Yongfeng Zhang - [[pdf]](https://arxiv.org/pdf/2511.02919)</summary>

**Abstract:** Recent advances in Large Language Model (LLM)-based agents have been propelled by Retrieval-Augmented Generation (RAG), which grants the models access to vast external knowledge bases. Despite RAG's success in improving agent performance, agent-level cache management, particularly constructing, maintaining, and updating a compact, relevant corpus dynamically tailored to each agent's need, remains underexplored. Therefore, we introduce ARC (Agent RAG Cache Mechanism), a novel, annotation-free caching framework that dynamically manages small, high-value corpora for each agent. By synthesizing historical query distribution patterns with the intrinsic geometry of cached items in the embedding space, ARC automatically maintains a high-relevance cache. With comprehensive experiments on three retrieval datasets, our experimental results demonstrate that ARC reduces storage requirements to 0.015% of the original corpus while offering up to 79.8% has-answer rate and reducing average retrieval latency by 80%. Our results demonstrate that ARC can drastically enhance efficiency and effectiveness in RAG-powered LLM agents.

**arXiv ID:** 2511.02919
</details>

<details>
<summary><strong>AURA: Autonomous Upskilling with Retrieval-Augmented Agents</strong> - Alvin Zhu, Yusuke Tanaka, Andrew Goldberg, Dennis Hong - [[pdf]](https://arxiv.org/pdf/2506.02507)</summary>

**Abstract:** Designing reinforcement learning curricula for agile robots traditionally requires extensive manual tuning of reward functions, environment randomizations, and training configurations. We introduce AURA (Autonomous Upskilling with Retrieval-Augmented Agents), a schema-validated curriculum reinforcement learning (RL) framework that leverages Large Language Models (LLMs) as autonomous designers of multi-stage curricula. AURA transforms user prompts into YAML workflows that encode full reward functions, domain randomization strategies, and training configurations. All files are statically validated before any GPU time is used, ensuring efficient and reliable execution. A retrieval-augmented feedback loop allows specialized LLM agents to design, execute, and refine curriculum stages based on prior training results stored in a vector database, enabling continual improvement over time. Quantitative experiments show that AURA consistently outperforms LLM-guided baselines in generation success rate, humanoid locomotion, and manipulation tasks. Ablation studies highlight the importance of schema validation and retrieval for curriculum quality. AURA successfully trains end-to-end policies directly from user prompts and deploys them zero-shot on a custom humanoid robot in multiple environments - capabilities that did not exist previously with manually designed controllers. By abstracting the complexity of curriculum design, AURA enables scalable and adaptive policy learning pipelines that would be complex to construct by hand. Project page: this https URL

**arXiv ID:** 2506.02507
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (16 papers)</h2></summary>

<details>
<summary><strong>No-Human in the Loop: Agentic Evaluation at Scale for Recommendation</strong> - Tao Zhang, Kehui Yao, Luyi Ma, Jiao Chen, Reza Yousefi Maragheh, Kai Zhao, Jianpeng Xu, Evren Korpeoglu, Sushant Kumar, Kannan Achan - [[pdf]](https://arxiv.org/pdf/2511.03051)</summary>

**Abstract:** Evaluating large language models (LLMs) as judges is increasingly critical for building scalable and trustworthy evaluation pipelines. We present ScalingEval, a large-scale benchmarking study that systematically compares 36 LLMs, including GPT, Gemini, Claude, and Llama, across multiple product categories using a consensus-driven evaluation protocol. Our multi-agent framework aggregates pattern audits and issue codes into ground-truth labels via scalable majority voting, enabling reproducible comparison of LLM evaluators without human annotation. Applied to large-scale complementary-item recommendation, the benchmark reports four key findings: (i) Anthropic Claude 3.5 Sonnet achieves the highest decision confidence; (ii) Gemini 1.5 Pro offers the best overall performance across categories; (iii) GPT-4o provides the most favorable latency-accuracy-cost tradeoff; and (iv) GPT-OSS 20B leads among open-source models. Category-level analysis shows strong consensus in structured domains (Electronics, Sports) but persistent disagreement in lifestyle categories (Clothing, Food). These results establish ScalingEval as a reproducible benchmark and evaluation protocol for LLMs as judges, with actionable guidance on scaling, reliability, and model family tradeoffs.

**arXiv ID:** 2511.03051
</details>

<details>
<summary><strong>PublicAgent: Multi-Agent Design Principles From an LLM-Based Open Data Analysis Framework</strong> - Sina Montazeri, Yunhe Feng, Kewei Sha - [[pdf]](https://arxiv.org/pdf/2511.03023)</summary>

**Abstract:** Open data repositories hold potential for evidence-based decision-making, yet are inaccessible to non-experts lacking expertise in dataset discovery, schema mapping, and statistical analysis. Large language models show promise for individual tasks, but end-to-end analytical workflows expose fundamental limitations: attention dilutes across growing contexts, specialized reasoning patterns interfere, and errors propagate undetected. We present PublicAgent, a multi-agent framework that addresses these limitations through decomposition into specialized agents for intent clarification, dataset discovery, analysis, and reporting. This architecture maintains focused attention within agent contexts and enables validation at each stage. Evaluation across five models and 50 queries derives five design principles for multi-agent LLM systems. First, specialization provides value independent of model strength--even the strongest model shows 97.5% agent win rates, with benefits orthogonal to model scale. Second, agents divide into universal (discovery, analysis) and conditional (report, intent) categories. Universal agents show consistent effectiveness (std dev 12.4%) while conditional agents vary by model (std dev 20.5%). Third, agents mitigate distinct failure modes--removing discovery or analysis causes catastrophic failures (243-280 instances), while removing report or intent causes quality degradation. Fourth, architectural benefits persist across task complexity with stable win rates (86-92% analysis, 84-94% discovery), indicating workflow management value rather than reasoning enhancement. Fifth, wide variance in agent effectiveness across models (42-96% for analysis) requires model-aware architecture design. These principles guide when and why specialization is necessary for complex analytical workflows while enabling broader access to public data through natural language interfaces.

**arXiv ID:** 2511.03023
</details>

<details>
<summary><strong>Toward Autonomous Engineering Design: A Knowledge-Guided Multi-Agent Framework</strong> - Varun Kumar, George Em Karniadakis - [[pdf]](https://arxiv.org/pdf/2511.03179)</summary>

**Abstract:** The engineering design process often demands expertise from multiple domains, leading to complex collaborations and iterative refinements. Traditional methods can be resource-intensive and prone to inefficiencies. To address this, we formalize the engineering design process through a multi-agent AI framework that integrates structured design and review loops. The framework introduces specialized knowledge-driven agents that collaborate to generate and refine design candidates. As an exemplar, we demonstrate its application to the aerodynamic optimization of 4-digit NACA airfoils. The framework consists of three key AI agents: a Graph Ontologist, a Design Engineer, and a Systems Engineer. The Graph Ontologist employs a Large Language Model (LLM) to construct two domain-specific knowledge graphs from airfoil design literature. The Systems Engineer, informed by a human manager, formulates technical requirements that guide design generation and evaluation. The Design Engineer leverages the design knowledge graph and computational tools to propose candidate airfoils meeting these requirements. The Systems Engineer reviews and provides feedback both qualitative and quantitative using its own knowledge graph, forming an iterative feedback loop until a design is validated by the manager. The final design is then optimized to maximize performance metrics such as the lift-to-drag ratio. Overall, this work demonstrates how collaborative AI agents equipped with structured knowledge representations can enhance efficiency, consistency, and quality in the engineering design process.

**arXiv ID:** 2511.03179
</details>

<details>
<summary><strong>Scaling Multi-Agent Environment Co-Design with Diffusion Models</strong> - Hao Xiang Li, Michael Amir, Amanda Prorok - [[pdf]](https://arxiv.org/pdf/2511.03100)</summary>

**Abstract:** The agent-environment co-design paradigm jointly optimises agent policies and environment configurations in search of improved system performance. With application domains ranging from warehouse logistics to windfarm management, co-design promises to fundamentally change how we deploy multi-agent systems. However, current co-design methods struggle to scale. They collapse under high-dimensional environment design spaces and suffer from sample inefficiency when addressing moving targets inherent to joint optimisation. We address these challenges by developing Diffusion Co-Design (DiCoDe), a scalable and sample-efficient co-design framework pushing co-design towards practically relevant settings. DiCoDe incorporates two core innovations. First, we introduce Projected Universal Guidance (PUG), a sampling technique that enables DiCoDe to explore a distribution of reward-maximising environments while satisfying hard constraints such as spatial separation between obstacles. Second, we devise a critic distillation mechanism to share knowledge from the reinforcement learning critic, ensuring that the guided diffusion model adapts to evolving agent policies using a dense and up-to-date learning signal. Together, these improvements lead to superior environment-policy pairs when validated on challenging multi-agent environment co-design benchmarks including warehouse automation, multi-agent pathfinding and wind farm optimisation. Our method consistently exceeds the state-of-the-art, achieving, for example, 39% higher rewards in the warehouse setting with 66% fewer simulation samples. This sets a new standard in agent-environment co-design, and is a stepping stone towards reaping the rewards of co-design in real world domains.

**arXiv ID:** 2511.03100
</details>

<details>
<summary><strong>RefAgent: A Multi-agent LLM-based Framework for Automatic Software Refactoring</strong> - Khouloud Oueslati, Maxime Lamothe, Foutse Khomh - [[pdf]](https://arxiv.org/pdf/2511.03153)</summary>

**Abstract:** Large Language Models (LLMs) have substantially influenced various software engineering tasks. Indeed, in the case of software refactoring, traditional LLMs have shown the ability to reduce development time and enhance code quality. However, these LLMs often rely on static, detailed instructions for specific tasks. In contrast, LLM-based agents can dynamically adapt to evolving contexts and autonomously make decisions by interacting with software tools and executing workflows. In this paper, we explore the potential of LLM-based agents in supporting refactoring activities. Specifically, we introduce RefAgent, a multi-agent LLM-based framework for end-to-end software refactoring. RefAgent consists of specialized agents responsible for planning, executing, testing, and iteratively refining refactorings using self-reflection and tool-calling capabilities. We evaluate RefAgent on eight open-source Java projects, comparing its effectiveness against a single-agent approach, a search-based refactoring tool, and historical developer refactorings. Our assessment focuses on: (1) the impact of generated refactorings on software quality, (2) the ability to identify refactoring opportunities, and (3) the contribution of each LLM agent through an ablation study. Our results show that RefAgent achieves a median unit test pass rate of 90%, reduces code smells by a median of 52.5%, and improves key quality attributes (e.g., reusability) by a median of 8.6%. Additionally, it closely aligns with developer refactorings and the search-based tool in identifying refactoring opportunities, attaining a median F1-score of 79.15% and 72.7%, respectively. Compared to single-agent approaches, RefAgent improves the median unit test pass rate by 64.7% and the median compilation success rate by 40.1%. These findings highlight the promise of multi-agent architectures in advancing automated software refactoring.

**arXiv ID:** 2511.03153
</details>

<details>
<summary><strong>AnaFlow: Agentic LLM-based Workflow for Reasoning-Driven Explainable and Sample-Efficient Analog Circuit Sizing</strong> - Mohsen Ahmadzadeh, Kaichang Chen, Georges Gielen - [[pdf]](https://arxiv.org/pdf/2511.03697)</summary>

**Abstract:** Analog/mixed-signal circuits are key for interfacing electronics with the physical world. Their design, however, remains a largely handcrafted process, resulting in long and error-prone design cycles. While the recent rise of AI-based reinforcement learning and generative AI has created new techniques to automate this task, the need for many time-consuming simulations is a critical bottleneck hindering the overall efficiency. Furthermore, the lack of explainability of the resulting design solutions hampers widespread adoption of the tools. To address these issues, a novel agentic AI framework for sample-efficient and explainable analog circuit sizing is presented. It employs a multi-agent workflow where specialized Large Language Model (LLM)-based agents collaborate to interpret the circuit topology, to understand the design goals, and to iteratively refine the circuit's design parameters towards the target goals with human-interpretable reasoning. The adaptive simulation strategy creates an intelligent control that yields a high sample efficiency. The AnaFlow framework is demonstrated for two circuits of varying complexity and is able to complete the sizing task fully automatically, differently from pure Bayesian optimization and reinforcement learning approaches. The system learns from its optimization history to avoid past mistakes and to accelerate convergence. The inherent explainability makes this a powerful tool for analog design space exploration and a new paradigm in analog EDA, where AI agents serve as transparent design assistants.

**arXiv ID:** 2511.03697
</details>

<details>
<summary><strong>Divide by Question, Conquer by Agent: SPLIT-RAG with Question-Driven Graph Partitioning</strong> - Ruiyi Yang, Hao Xue, Imran Razzak, Shirui Pan, Hakim Hacid, Flora D. Salim - [[pdf]](https://arxiv.org/pdf/2505.13994)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) systems empower large language models (LLMs) with external knowledge, yet struggle with efficiency-accuracy trade-offs when scaling to large knowledge graphs. Existing approaches often rely on monolithic graph retrieval, incurring unnecessary latency for simple queries and fragmented reasoning for complex multi-hop questions. To address these challenges, this paper propose SPLIT-RAG, a multi-agent RAG framework that addresses these limitations with question-driven semantic graph partitioning and collaborative subgraph retrieval. The innovative framework first create Semantic Partitioning of Linked Information, then use the Type-Specialized knowledge base to achieve Multi-Agent RAG. The attribute-aware graph segmentation manages to divide knowledge graphs into semantically coherent subgraphs, ensuring subgraphs align with different query types, while lightweight LLM agents are assigned to partitioned subgraphs, and only relevant partitions are activated during retrieval, thus reduce search space while enhancing efficiency. Finally, a hierarchical merging module resolves inconsistencies across subgraph-derived answers through logical verifications. Extensive experimental validation demonstrates considerable improvements compared to existing approaches.

**arXiv ID:** 2505.13994
</details>

<details>
<summary><strong>Multi-Agent Reinforcement Learning for Autonomous Multi-Satellite Earth Observation: A Realistic Case Study</strong> - Mohamad A. Hady, Siyi Hu, Mahardhika Pratama, Jimmy Cao, Ryszard Kowalczyk - [[pdf]](https://arxiv.org/pdf/2506.15207)</summary>

**Abstract:** The exponential growth of Low Earth Orbit (LEO) satellites has revolutionised Earth Observation (EO) missions, addressing challenges in climate monitoring, disaster management, and more. However, autonomous coordination in multi-satellite systems remains a fundamental challenge. Traditional optimisation approaches struggle to handle the real-time decision-making demands of dynamic EO missions, necessitating the use of Reinforcement Learning (RL) and Multi-Agent Reinforcement Learning (MARL). In this paper, we investigate RL-based autonomous EO mission planning by modelling single-satellite operations and extending to multi-satellite constellations using MARL frameworks. We address key challenges, including energy and data storage limitations, uncertainties in satellite observations, and the complexities of decentralised coordination under partial observability. By leveraging a near-realistic satellite simulation environment, we evaluate the training stability and performance of state-of-the-art MARL algorithms, including PPO, IPPO, MAPPO, and HAPPO. Our results demonstrate that MARL can effectively balance imaging and resource management while addressing non-stationarity and reward interdependency in multi-satellite coordination. The insights gained from this study provide a foundation for autonomous satellite operations, offering practical guidelines for improving policy learning in decentralised EO missions.

**arXiv ID:** 2506.15207
</details>

<details>
<summary><strong>Large Language Models Miss the Multi-Agent Mark</strong> - Emanuele La Malfa, Gabriele La Malfa, Samuele Marro, Jie M. Zhang, Elizabeth Black, Michael Luck, Philip Torr, Michael Wooldridge - [[pdf]](https://arxiv.org/pdf/2505.21298)</summary>

**Abstract:** Recent interest in Multi-Agent Systems of Large Language Models (MAS LLMs) has led to an increase in frameworks leveraging multiple LLMs to tackle complex tasks. However, much of this literature appropriates the terminology of MAS without engaging with its foundational principles. In this position paper, we highlight critical discrepancies between MAS theory and current MAS LLMs implementations, focusing on four key areas: the social aspect of agency, environment design, coordination and communication protocols, and measuring emergent behaviours. Our position is that many MAS LLMs lack multi-agent characteristics such as autonomy, social interaction, and structured environments, and often rely on oversimplified, LLM-centric architectures. The field may slow down and lose traction by revisiting problems the MAS literature has already addressed. Therefore, we systematically analyse this issue and outline associated research opportunities; we advocate for better integrating established MAS concepts and more precise terminology to avoid mischaracterisation and missed opportunities.

**arXiv ID:** 2505.21298
</details>

<details>
<summary><strong>Decentralized Aerial Manipulation of a Cable-Suspended Load using Multi-Agent Reinforcement Learning</strong> - Jack Zeng, Andreu Matoses Gimenez, Eugene Vinitsky, Javier Alonso-Mora, Sihao Sun - [[pdf]](https://arxiv.org/pdf/2508.01522)</summary>

**Abstract:** This paper presents the first decentralized method to enable real-world 6-DoF manipulation of a cable-suspended load using a team of Micro-Aerial Vehicles (MAVs). Our method leverages multi-agent reinforcement learning (MARL) to train an outer-loop control policy for each MAV. Unlike state-of-the-art controllers that utilize a centralized scheme, our policy does not require global states, inter-MAV communications, nor neighboring MAV information. Instead, agents communicate implicitly through load pose observations alone, which enables high scalability and flexibility. It also significantly reduces computing costs during inference time, enabling onboard deployment of the policy. In addition, we introduce a new action space design for the MAVs using linear acceleration and body rates. This choice, combined with a robust low-level controller, enables reliable sim-to-real transfer despite significant uncertainties caused by cable tension during dynamic 3D motion. We validate our method in various real-world experiments, including full-pose control under load model uncertainties, showing setpoint tracking performance comparable to the state-of-the-art centralized method. We also demonstrate cooperation amongst agents with heterogeneous control policies, and robustness to the complete in-flight loss of one MAV. Videos of experiments: this https URL

**arXiv ID:** 2508.01522
</details>

<details>
<summary><strong>AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation</strong> - Xianyang Liu, Yilin Liu, Shuai Wang, Hao Cheng, Andrew Estornell, Yuzhi Zhao, Jiaheng Wei - [[pdf]](https://arxiv.org/pdf/2510.19361)</summary>

**Abstract:** The creation of high-quality datasets to improve Large Language Model (LLM) reasoning remains a significant challenge, as current methods often suffer from generating low-quality/incorrect answers and limited information richness from available data sources. To address this, we propose AgenticMath, a novel agentic pipeline for generating high-quality mathematical question-answer pairs to enhance the supervised fine-tuning of LLMs. Our method operates through four stages: (1) Seed Question Filter that selects questions with high information richness, complexity, and clarity; (2) an Agentic Question Rephrase step that employs a multi-agent system to generate diverse, logically consistent paraphrases; (3) an Answer Augment step where rewrite answers using chain-of-thought reasoning to enhance numerical and logical correctness, without reliance on human-provided labels; and (4) a final Question and Answer Evaluation that retains only the most superior pairs. Extensive experiments demonstrate that, fine-tuning 3B-8B parameter LLMs on AgenticMath generated datasets (comprising only 30-60K math samples) achieves competitive or superior performance on diverse in domain and out-of-domain mathematical reasoning benchmarks compared to baselines trained on much more data (e.g., 400K or 2.3M samples). Our work demonstrates that targeted, high-quality data generation is a more efficient path to improving mathematical reasoning in LLMs than large-scale, low-quality alternatives.

**arXiv ID:** 2510.19361
</details>

<details>
<summary><strong>CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization</strong> - Zijian Zhang, Rong Wang, Shiyang Li, Yuebo Luo, Mingyi Hong, Caiwen Ding - [[pdf]](https://arxiv.org/pdf/2511.01884)</summary>

**Abstract:** Developing efficient CUDA kernels is increasingly critical for AI applications such as large-scale LLM training. However, manual kernel design is both costly and time-consuming, motivating automatic approaches that leverage LLMs for code generation. Existing methods for automatic kernel generation, however, often produce low-efficiency kernels, incur high computational overhead, and fail to generalize across settings. In this work, we propose CudaForge, a training-free multi-agent workflow for CUDA kernel generation and optimization. Our workflow is inspired by the iterative workflow of human experts, which contains steps such as developing initial kernels, testing correctness, analyzing hardware feedback, and iterative improvement. More specifically, CudaForge employs two LLM agents: a Coder and a Judge, that iteratively generate, correct, and optimize CUDA kernels, while integrating hardware feedback such as Nsight Compute (NCU) metrics. In extensive evaluations, we show that CudaForge, by leveraging base models like OpenAI-o3, achieves 97.6\% correctness of generated kernels and an average 1.68$\times$ speedup over PyTorch baselines, substantially surpassing state-of-the-art models including OpenAI-o3 and Kevin on this http URL accuracy and speed, CudaForge demonstrates strong generalization across GPUs (A100, RTX 6000, 4090, 3090) and base models (OpenAI-o3, GPT-5, gpt-oss-120B, Claude-Sonnet-4, QwQ-32B), while maintaining high efficiency. In particular, generating an optimized kernel takes about 26.5 minutes on one RTX6000 and incurs about \$ 0.3 API cost, which is significantly cheaper than existing agentic work that costs 6 H100 hours and \$ 5 API cost per kernel. Our results highlight that multi-agent, training-free workflows can enable cost-effective, generalizable, and high-performance CUDA kernel optimization. Code available at this https URL

**arXiv ID:** 2511.01884
</details>

<details>
<summary><strong>ALAS: Transactional and Dynamic Multi-Agent LLM Planning</strong> - Longling Geng, Edward Y. Chang - [[pdf]](https://arxiv.org/pdf/2511.03094)</summary>

**Abstract:** Large language models enable flexible multi-agent planning but remain fragile in practice: verification is often circular, state changes are not tracked for repair, and small faults trigger costly global recomputation. We present ALAS, a stateful, disruption-aware framework that separates planning from non-circular validation, records a versioned execution log for grounded checks and restore points, and performs localized repair that preserves work in progress. The validator operates independently of the planning LLM with fresh, bounded context, avoiding self-check loops and mid-context attrition. The repair protocol edits only the minimal affected region under explicit policies (retry, catch, timeout, backoff, idempotency keys, compensation, loop guards) defined in a canonical workflow IR that maps to Amazon States Language and Argo Workflows. On job-shop scheduling suites (DMU, TA) across five classical benchmarks, ALAS matches or exceeds strong single-LLM and multi-agent baselines, achieving 83.7% success, reducing token usage by 60%, and running 1.82times faster under comparable settings. A minimal reliability study shows that the validator detects injected structural faults with low overhead, and that localized repair contains runtime perturbations with a bounded edit radius and less makespan degradation than global recompute. Results indicate that the combination of validator isolation, versioned execution logs, and localized repair provides measurable efficiency, feasibility, and scalability for multi-agent LLM planning. Code and seeds will be released.

**arXiv ID:** 2511.03094
</details>

<details>
<summary><strong>Learning Communication Skills in Multi-task Multi-agent Deep Reinforcement Learning</strong> - Changxi Zhu, Mehdi Dastani, Shihan Wang - [[pdf]](https://arxiv.org/pdf/2511.03348)</summary>

**Abstract:** In multi-agent deep reinforcement learning (MADRL), agents can communicate with one another to perform a task in a coordinated manner. When multiple tasks are involved, agents can also leverage knowledge from one task to improve learning in other tasks. In this paper, we propose Multi-task Communication Skills (MCS), a MADRL with communication method that learns and performs multiple tasks simultaneously, with agents interacting through learnable communication protocols. MCS employs a Transformer encoder to encode task-specific observations into a shared message space, capturing shared communication skills among agents. To enhance coordination among agents, we introduce a prediction network that correlates messages with the actions of sender agents in each task. We adapt three multi-agent benchmark environments to multi-task settings, where the number of agents as well as the observation and action spaces vary across tasks. Experimental results demonstrate that MCS achieves better performance than multi-task MADRL baselines without communication, as well as single-task MADRL baselines with and without communication.

**arXiv ID:** 2511.03348
</details>

<details>
<summary><strong>AI Agents with Decentralized Identifiers and Verifiable Credentials</strong> - Sandro Rodriguez Garzon, Awid Vaziry, Enis Mert Kuzu, Dennis Enrique Gehrmann, Buse Varkan, Alexander Gaballa, Axel Küpper - [[pdf]](https://arxiv.org/pdf/2511.02841)</summary>

**Abstract:** LLM-based AI agents still lack the technical means to automatically build nuanced and differentiated trust in other agents at the beginning of an agent-to-agent dialogue. But autonomous and interoperable trust establishing becomes a fundamental prerequisite once agents start to operate beyond isolated environments and engage in dialogues across individual or organizational boundaries. A promising way to fill this gap in Agentic AI is to equip agents with long-lived digital identities and introduce tamper-proof and flexible identity-bound attestations of agents, provisioned by commonly trusted third parties and designed for cross-domain verifiability. This article presents a conceptual framework and a prototypical multi-agent system, where each agent is endowed with a self-sovereign digital identity. It combines a unique and ledger-anchored Decentralized Identifier (DID) of an agent with a set of third-party issued Verifiable Credentials (VCs). This enables agents at the start of a dialog to prove ownership of their self-controlled DIDs for authentication purposes and to establish various cross-domain trust relationships through the spontaneous exchange of their self-hosted DID-bound VCs. A comprehensive evaluation of the prototypical implementation demonstrates technical feasibility but also reveals limitations once an agent's LLM is in sole charge to control the respective security procedures.

**arXiv ID:** 2511.02841
</details>

<details>
<summary><strong>Manifold-constrained Hamilton-Jacobi Reachability Learning for Decentralized Multi-Agent Motion Planning</strong> - Qingyi Chen, Ruiqi Ni, Jun Kim, Ahmed H. Qureshi - [[pdf]](https://arxiv.org/pdf/2511.03591)</summary>

**Abstract:** Safe multi-agent motion planning (MAMP) under task-induced constraints is a critical challenge in robotics. Many real-world scenarios require robots to navigate dynamic environments while adhering to manifold constraints imposed by tasks. For example, service robots must carry cups upright while avoiding collisions with humans or other robots. Despite recent advances in decentralized MAMP for high-dimensional systems, incorporating manifold constraints remains difficult. To address this, we propose a manifold-constrained Hamilton-Jacobi reachability (HJR) learning framework for decentralized MAMP. Our method solves HJR problems under manifold constraints to capture task-aware safety conditions, which are then integrated into a decentralized trajectory optimization planner. This enables robots to generate motion plans that are both safe and task-feasible without requiring assumptions about other agents' policies. Our approach generalizes across diverse manifold-constrained tasks and scales effectively to high-dimensional multi-agent manipulation problems. Experiments show that our method outperforms existing constrained motion planners and operates at speeds suitable for real-world applications. Video demonstrations are available at this https URL .

**arXiv ID:** 2511.03591
</details>

</details>

<details open>
<summary><h2>Other Agent Research (3 papers)</h2></summary>

<details>
<summary><strong>Autonomous Robotic Drilling System for Mice Cranial Window Creation</strong> - Enduo Zhao, Murilo M. Marinho, Kanako Harada - [[pdf]](https://arxiv.org/pdf/2406.14135)</summary>

**Abstract:** Robotic assistance for experimental manipulation in the life sciences is expected to enable favorable outcomes, regardless of the skill of the scientist. Experimental specimens in the life sciences are subject to individual variability and hence require intricate algorithms for successful autonomous robotic control. As a use case, we are studying the cranial window creation in mice. This operation requires the removal of an 8-mm circular patch of the skull, which is approximately 300 um thick, but the shape and thickness of the mouse skull significantly varies depending on the strain of the mouse, sex, and age. In this work, we develop an autonomous robotic drilling system with no offline planning, consisting of a trajectory planner with execution-time feedback with drilling completion level recognition based on image and force information. In the experiments, we first evaluate the image-and-force-based drilling completion level recognition by comparing it with other state-of-the-art deep learning image processing methods and conduct an ablation study in eggshell drilling to evaluate the impact of each module on system performance. Finally, the system performance is further evaluated in postmortem mice, achieving a success rate of 70% (14/20 trials) with an average drilling time of 9.3 min.

**arXiv ID:** 2406.14135
</details>

<details>
<summary><strong>HAFixAgent: History-Aware Automated Program Repair Agent</strong> - Yu Shi, Hao Li, Bram Adams, Ahmed E. Hassan - [[pdf]](https://arxiv.org/pdf/2511.01047)</summary>

**Abstract:** Automated program repair (APR) has recently shifted toward large language models and agent-based systems, yet most systems rely on local snapshot context, overlooking repository history. Prior work shows that repository history helps repair single-line bugs, since the last commit touching the buggy line is often the bug-introducing one. In this paper, we investigate whether repository history can also improve agentic APR systems at scale, especially for complex multi-hunk bugs. We present HAFixAgent, a History-Aware Bug-Fixing Agent that injects blame-derived repository heuristics into its repair loop. A preliminary study of all 854 real-world bugs from Defects4J motivates our design, showing that bug-relevant history is both widely available and highly concentrated. Empirical comparison of HAFixAgent with two state-of-the-art baselines shows: (1) Effectiveness: HAFixAgent significantly improves over the agent-based baseline (by 212.3%) and the multi-hunk baseline (by 29.9%). (2) Efficiency: history does not significantly increase agent steps and keeps token costs comparable, with notably lower median costs for complex multi-file-multi-hunk bugs. (3) Practicality: combining different historical heuristics repairs more bugs, offering a clear cost-benefit trade-off. HAFixAgent offers a practical recipe for history-aware agentic APR: ground the agent in version control history, prioritize diff-based historical context, and integrate complementary heuristics when needed.

**arXiv ID:** 2511.01047
</details>

<details>
<summary><strong>Constraint-Driven Small Language Models Based on Agent and OpenAlex Knowledge Graph: Mining Conceptual Pathways and Discovering Innovation Points in Academic Papers</strong> - Ziye Xia, Sergei S. Ospichev - [[pdf]](https://arxiv.org/pdf/2510.14303)</summary>

**Abstract:** In recent years, the rapid increase in academic publications across various fields has posed severe challenges for academic paper analysis: scientists struggle to timely and comprehensively track the latest research findings and methodologies. Key concept extraction has proven to be an effective analytical paradigm, and its automation has been achieved with the widespread application of language models in industrial and scientific domains. However, existing paper databases are mostly limited to similarity matching and basic classification of key concepts, failing to deeply explore the relational networks between concepts. This paper is based on the OpenAlex opensource knowledge graph. By analyzing nearly 8,000 open-source paper data from Novosibirsk State University, we discovered a strong correlation between the distribution patterns of paper key concept paths and both innovation points and rare paths. We propose a prompt engineering-based key concept path analysis method. This method leverages small language models to achieve precise key concept extraction and innovation point identification, and constructs an agent based on a knowledge graph constraint mechanism to enhance analysis accuracy. Through fine-tuning of the Qwen and DeepSeek models, we achieved significant improvements in accuracy, with the models publicly available on the Hugging Face platform.

**arXiv ID:** 2510.14303
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (26 papers)</h2></summary>

<details>
<summary><strong>Outbidding and Outbluffing Elite Humans: Mastering Liar's Poker via Self-Play and Reinforcement Learning</strong> - Richard Dewey, Janos Botyanszki, Ciamac C. Moallemi, Andrew T. Zheng - [[pdf]](https://arxiv.org/pdf/2511.03724)</summary>

**Abstract:** AI researchers have long focused on poker-like games as a testbed for environments characterized by multi-player dynamics, imperfect information, and reasoning under uncertainty. While recent breakthroughs have matched elite human play at no-limit Texas hold'em, the multi-player dynamics are subdued: most hands converge quickly with only two players engaged through multiple rounds of bidding. In this paper, we present Solly, the first AI agent to achieve elite human play in reduced-format Liar's Poker, a game characterized by extensive multi-player engagement. We trained Solly using self-play with a model-free, actor-critic, deep reinforcement learning algorithm. Solly played at an elite human level as measured by win rate (won over 50% of hands) and equity (money won) in heads-up and multi-player Liar's Poker. Solly also outperformed large language models (LLMs), including those with reasoning abilities, on the same metrics. Solly developed novel bidding strategies, randomized play effectively, and was not easily exploitable by world-class human players.

**arXiv ID:** 2511.03724
</details>

<details>
<summary><strong>AgentSLA : Towards a Service Level Agreement for AI Agents</strong> - Gwendal Jouneaux, Jordi Cabot - [[pdf]](https://arxiv.org/pdf/2511.02885)</summary>

**Abstract:** AI components are increasingly becoming a key element of all types of software systems to enhance their functionality. These AI components are often implemented as AI Agents, offering more autonomy than a plain integration of Large Language Models (LLMs), moving from a Model-as-a-Service paradigm to an Agent-as-a-Service one, bringing new challenges to the development of smart software systems. Indeed, while support for the design, implementation, and deployment of those agents exist, the specification of Quality of Service (QoS) and definition of Service Level Agreements (SLAs) aspects for those agents, important to ensure the quality of the resulting systems, remains an open challenge. Part of this is due to the difficulty to clearly define quality in the context of AI components, resulting in a lack of consensus on how to best approach Quality Assurance (QA) for these types of systems. To address this challenge, this paper proposes both a quality model for AI agents based on the ISO/IEC 25010 standard, and a domain specific language to support the definition of SLAs for the services provided by these AI agents.

**arXiv ID:** 2511.02885
</details>

<details>
<summary><strong>Hybrid Fact-Checking that Integrates Knowledge Graphs, Large Language Models, and Search-Based Retrieval Agents Improves Interpretable Claim Verification</strong> - Shaghayegh Kolli, Richard Rosenbaum, Timo Cavelius, Lasse Strothe, Andrii Lata, Jana Diesner - [[pdf]](https://arxiv.org/pdf/2511.03217)</summary>

**Abstract:** Large language models (LLMs) excel in generating fluent utterances but can lack reliable grounding in verified information. At the same time, knowledge-graph-based fact-checkers deliver precise and interpretable evidence, yet suffer from limited coverage or latency. By integrating LLMs with knowledge graphs and real-time search agents, we introduce a hybrid fact-checking approach that leverages the individual strengths of each component. Our system comprises three autonomous steps: 1) a Knowledge Graph (KG) Retrieval for rapid one - hop lookups in DBpedia, 2) an LM-based classification guided by a task-specific labeling prompt, producing outputs with internal rule-based logic, and 3) a Web Search Agent invoked only when KG coverage is insufficient. Our pipeline achieves an F1 score of 0.93 on the FEVER benchmark on the Supported/Refuted split without task- specific fine - tuning. To address Not enough information cases, we conduct a targeted reannotation study showing that our approach frequently uncovers valid evidence for claims originally labeled as Not Enough Information (NEI), as confirmed by both expert annotators and LLM reviewers. With this paper, we present a modular, opensource fact-checking pipeline with fallback strategies and generalization across datasets.

**arXiv ID:** 2511.03217
</details>

<details>
<summary><strong>Inter-Agent Trust Models: A Comparative Study of Brief, Claim, Proof, Stake, Reputation and Constraint in Agentic Web Protocol Design-A2A, AP2, ERC-8004, and Beyond</strong> - Botao 'Amber' Hu, Helena Rong - [[pdf]](https://arxiv.org/pdf/2511.03434)</summary>

**Abstract:** As the "agentic web" takes shape-billions of AI agents (often LLM-powered) autonomously transacting and collaborating-trust shifts from human oversight to protocol design. In 2025, several inter-agent protocols crystallized this shift, including Google's Agent-to-Agent (A2A), Agent Payments Protocol (AP2), and Ethereum's ERC-8004 "Trustless Agents," yet their underlying trust assumptions remain under-examined. This paper presents a comparative study of trust models in inter-agent protocol design: Brief (self- or third-party verifiable claims), Claim (self-proclaimed capabilities and identity, e.g. AgentCard), Proof (cryptographic verification, including zero-knowledge proofs and trusted execution environment attestations), Stake (bonded collateral with slashing and insurance), Reputation (crowd feedback and graph-based trust signals), and Constraint (sandboxing and capability bounding). For each, we analyze assumptions, attack surfaces, and design trade-offs, with particular emphasis on LLM-specific fragilities-prompt injection, sycophancy/nudge-susceptibility, hallucination, deception, and misalignment-that render purely reputational or claim-only approaches brittle. Our findings indicate no single mechanism suffices. We argue for trustless-by-default architectures anchored in Proof and Stake to gate high-impact actions, augmented by Brief for identity and discovery and Reputation overlays for flexibility and social signals. We comparatively evaluate A2A, AP2, ERC-8004 and related historical variations in academic research under metrics spanning security, privacy, latency/cost, and social robustness (Sybil/collusion/whitewashing resistance). We conclude with hybrid trust model recommendations that mitigate reputation gaming and misinformed LLM behavior, and we distill actionable design guidelines for safer, interoperable, and scalable agent economies.

**arXiv ID:** 2511.03434
</details>

<details>
<summary><strong>ROSBag MCP Server: Analyzing Robot Data with LLMs for Agentic Embodied AI Applications</strong> - Lei Fu, Sahar Salimpour, Leonardo Militano, Harry Edelman, Jorge Peña Queralta, Giovanni Toffetti - [[pdf]](https://arxiv.org/pdf/2511.03497)</summary>

**Abstract:** Agentic AI systems and Physical or Embodied AI systems have been two key research verticals at the forefront of Artificial Intelligence and Robotics, with Model Context Protocol (MCP) increasingly becoming a key component and enabler of agentic applications. However, the literature at the intersection of these verticals, i.e., Agentic Embodied AI, remains scarce. This paper introduces an MCP server for analyzing ROS and ROS 2 bags, allowing for analyzing, visualizing and processing robot data with natural language through LLMs and VLMs. We describe specific tooling built with robotics domain knowledge, with our initial release focused on mobile robotics and supporting natively the analysis of trajectories, laser scan data, transforms, or time series data. This is in addition to providing an interface to standard ROS 2 CLI tools ("ros2 bag list" or "ros2 bag info"), as well as the ability to filter bags with a subset of topics or trimmed in time. Coupled with the MCP server, we provide a lightweight UI that allows the benchmarking of the tooling with different LLMs, both proprietary (Anthropic, OpenAI) and open-source (through Groq). Our experimental results include the analysis of tool calling capabilities of eight different state-of-the-art LLM/VLM models, both proprietary and open-source, large and small. Our experiments indicate that there is a large divide in tool calling capabilities, with Kimi K2 and Claude Sonnet 4 demonstrating clearly superior performance. We also conclude that there are multiple factors affecting the success rates, from the tool description schema to the number of arguments, as well as the number of tools available to the models. The code is available with a permissive license at this https URL.

**arXiv ID:** 2511.03497
</details>

<details>
<summary><strong>s3: You Don't Need That Much Data to Train a Search Agent via RL</strong> - Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng Xiao, Zifeng Wang, Jimeng Sun, Jiawei Han - [[pdf]](https://arxiv.org/pdf/2505.14146)</summary>

**Abstract:** Retrieval-augmented generation (RAG) systems empower large language models (LLMs) to access external knowledge during inference. Recent advances have enabled LLMs to act as search agents via reinforcement learning (RL), improving information acquisition through multi-turn interactions with retrieval engines. However, existing approaches either optimize retrieval using search-only metrics (e.g., NDCG) that ignore downstream utility or fine-tune the entire LLM to jointly reason and retrieve-entangling retrieval with generation and limiting the real search utility and compatibility with frozen or proprietary models. In this work, we propose s3, a lightweight, model-agnostic framework that decouples the searcher from the generator and trains the searcher using a Gain Beyond RAG reward: the improvement in generation accuracy over naive RAG. s3 requires only 2.4k training samples to outperform baselines trained on over 70x more data, consistently delivering stronger downstream performance across six general QA and five medical QA benchmarks.

**arXiv ID:** 2505.14146
</details>

<details>
<summary><strong>Reinforcement Learning Foundations for Deep Research Systems: A Survey</strong> - Wenjun Li, Zhi Chen, Jingru Lin, Hannan Cao, Wei Han, Sheng Liang, Zhi Zhang, Kuicai Dong, Dexun Li, Chen Zhang, Yong Liu - [[pdf]](https://arxiv.org/pdf/2509.06733)</summary>

**Abstract:** Deep research systems, agentic AI that solve complex, multi-step tasks by coordinating reasoning, search across the open web and user files, and tool use, are moving toward hierarchical deployments with a Planner, Coordinator, and Executors. In practice, training entire stacks end-to-end remains impractical, so most work trains a single planner connected to core tools such as search, browsing, and code. While SFT imparts protocol fidelity, it suffers from imitation and exposure biases and underuses environment feedback. Preference alignment methods such as DPO are schema and proxy-dependent, off-policy, and weak for long-horizon credit assignment and multi-objective trade-offs. A further limitation of SFT and DPO is their reliance on human defined decision points and subskills through schema design and labeled comparisons. Reinforcement learning aligns with closed-loop, tool-interaction research by optimizing trajectory-level policies, enabling exploration, recovery behaviors, and principled credit assignment, and it reduces dependence on such human priors and rater biases.
This survey is, to our knowledge, the first dedicated to the RL foundations of deep research systems. It systematizes recent work along three axes: (i) data synthesis and curation; (ii) RL methods for agentic research covering stability, sample efficiency, long context handling, reward and credit design, multi-objective optimization, and multimodal integration; and (iii) agentic RL training systems and frameworks. We also cover agent architecture and coordination, as well as evaluation and benchmarks, including recent QA, VQA, long-form synthesis, and domain-grounded, tool-interaction tasks. We distill recurring patterns, surface infrastructure bottlenecks, and offer practical guidance for training robust, transparent deep research agents with RL.

**arXiv ID:** 2509.06733
</details>

<details>
<summary><strong>VoiceAgentBench: Are Voice Assistants ready for agentic tasks?</strong> - Dhruv Jain, Harshit Shukla, Gautam Rajeev, Ashish Kulkarni, Chandra Khatri, Shubham Agarwal - [[pdf]](https://arxiv.org/pdf/2510.07978)</summary>

**Abstract:** Large-scale Speech Language Models (SpeechLMs) have enabled voice assistants capable of understanding natural spoken queries and performing complex tasks. However, existing speech benchmarks primarily focus on isolated capabilities such as transcription, or question-answering, and do not systematically evaluate agentic scenarios encompassing multilingual and cultural understanding, as well as adversarial robustness. To address this, we introduce VoiceAgentBench, a comprehensive benchmark designed to evaluate SpeechLMs in realistic spoken agentic settings. It comprises over 5,500 synthetic spoken queries, including dialogues grounded in Indian context, covering single-tool invocations, multi-tool workflows, multi-turn interactions, and safety evaluations. The benchmark supports English, Hindi, and 5 other Indian languages, reflecting real-world linguistic and cultural diversity. We simulate speaker variability using a novel sampling algorithm that selects audios for TTS voice conversion based on its speaker embeddings, maximizing acoustic and speaker diversity. Our evaluation measures tool selection accuracy, structural consistency, and the correctness of tool invocations, including adversarial robustness. Our experiments reveal significant gaps in contextual tool orchestration tasks, Indic generalization, and adversarial robustness, exposing critical limitations of current SpeechLMs.

**arXiv ID:** 2510.07978
</details>

<details>
<summary><strong>Agentic Meta-Orchestrator for Multi-task Copilots</strong> - Xiaofeng Zhu, Yunshen Zhou - [[pdf]](https://arxiv.org/pdf/2510.22781)</summary>

**Abstract:** Microsoft Copilot suites serve as the universal entry point for various agents skilled in handling important tasks, ranging from assisting a customer with product purchases to detecting vulnerabilities in corporate programming code. Each agent can be powered by language models, software engineering operations, such as database retrieval, and internal \& external knowledge. The repertoire of a copilot can expand dynamically with new agents. This requires a robust orchestrator that can distribute tasks from user prompts to the right agents. In this work, we propose an Agentic Meta-orchestrator (AMO) for handling multiple tasks and scalable agents in copilot services, which can provide both natural language and action responses. We will also demonstrate the planning that leverages meta-learning, i.e., a trained decision tree model for deciding the best inference strategy among various agents/models. We showcase the effectiveness of our AMO through two production use cases: Microsoft 365 (M365) E-Commerce Copilot and code compliance copilot. M365 E-Commerce Copilot advertises Microsoft products to external customers to promote sales success. The M365 E-Commerce Copilot provides up-to-date product information and connects to multiple agents, such as relational databases and human customer support. The code compliance copilot scans the internal DevOps code to detect known and new compliance issues in pull requests (PR).

**arXiv ID:** 2510.22781
</details>

<details>
<summary><strong>Agent-Omni: Test-Time Multimodal Reasoning via Model Coordination for Understanding Anything</strong> - Huawei Lin, Yunzhi Shi, Tong Geng, Weijie Zhao, Wei Wang, Ravender Pal Singh - [[pdf]](https://arxiv.org/pdf/2511.02834)</summary>

**Abstract:** Multimodal large language models (MLLMs) have shown strong capabilities but remain limited to fixed modality pairs and require costly fine-tuning with large aligned datasets. Building fully omni-capable models that can integrate text, images, audio, and video remains impractical and lacks robust reasoning support. In this paper, we propose an Agent-Omni framework that coordinates existing foundation models through a master-agent system, enabling flexible multimodal reasoning without retraining. The master agent interprets user intent, delegates subtasks to modality-specific agents, and integrates their outputs into coherent responses. Extensive experiments across text, image, audio, video, and omni benchmarks show that Agent-Omni consistently achieves state-of-the-art performance, particularly on tasks requiring complex cross-modal reasoning. Its agent-based design enables seamless integration of specialized foundation models, ensuring adaptability to diverse inputs while maintaining transparency and interpretability. In addition, the framework is modular and easily extensible, allowing future improvements as stronger models become available.

**arXiv ID:** 2511.02834
</details>

<details>
<summary><strong>Intelligent Computing Social Modeling and Methodological Innovations in Political Science in the Era of Large Language Models</strong> - Zhenyu Wang, Dequan Wang, Yi Xu, Lingfeng Zhou, Yiqi Zhou - [[pdf]](https://arxiv.org/pdf/2410.16301)</summary>

**Abstract:** The recent wave of artificial intelligence, epitomized by large language models (LLMs),has presented opportunities and challenges for methodological innovation in political science,sparking discussions on a potential paradigm shift in the social sciences. However, how can weunderstand the impact of LLMs on knowledge production and paradigm transformation in thesocial sciences from a comprehensive perspective that integrates technology and methodology? What are LLMs' specific applications and representative innovative methods in political scienceresearch? These questions, particularly from a practical methodological standpoint, remainunderexplored. This paper proposes the "Intelligent Computing Social Modeling" (ICSM) methodto address these issues by clarifying the critical mechanisms of LLMs. ICSM leverages thestrengths of LLMs in idea synthesis and action simulation, advancing intellectual exploration inpolitical science through "simulated social construction" and "simulation validation." Bysimulating the U.S. presidential election, this study empirically demonstrates the operationalpathways and methodological advantages of ICSM. By integrating traditional social scienceparadigms, ICSM not only enhances the quantitative paradigm's capability to apply big data toassess the impact of factors but also provides qualitative paradigms with evidence for socialmechanism discovery at the individual level, offering a powerful tool that balances interpretabilityand predictability in social science research. The findings suggest that LLMs will drivemethodological innovation in political science through integration and improvement rather thandirect substitution.

**arXiv ID:** 2410.16301
</details>

<details>
<summary><strong>RoboRAN: A Unified Robotics Framework for Reinforcement Learning-Based Autonomous Navigation</strong> - Matteo El-Hariry, Antoine Richard, Ricard M. Castan, Luis F. W. Batista, Matthieu Geist, Cedric Pradalier, Miguel Olivares-Mendez - [[pdf]](https://arxiv.org/pdf/2505.14526)</summary>

**Abstract:** Autonomous robots must navigate and operate in diverse environments, from terrestrial and aquatic settings to aerial and space domains. While Reinforcement Learning (RL) has shown promise in training policies for specific autonomous robots, existing frameworks and benchmarks are often constrained to unique platforms, limiting generalization and fair comparisons across different mobility systems. In this paper, we present a multi-domain framework for training, evaluating and deploying RL-based navigation policies across diverse robotic platforms and operational environments. Our work presents four key contributions: (1) a scalable and modular framework, facilitating seamless robot-task interchangeability and reproducible training pipelines; (2) sim-to-real transfer demonstrated through real-world experiments with multiple robots, including a satellite robotic simulator, an unmanned surface vessel, and a wheeled ground vehicle; (3) the release of the first open-source API for deploying Isaac Lab-trained policies to real robots, enabling lightweight inference and rapid field validation; and (4) uniform tasks and metrics for cross-medium evaluation, through a unified evaluation testbed to assess performance of navigation tasks in diverse operational conditions (aquatic, terrestrial and space). By ensuring consistency between simulation and real-world deployment, RoboRAN lowers the barrier to developing adaptable RL-based navigation strategies. Its modular design enables straightforward integration of new robots and tasks through predefined templates, fostering reproducibility and extension to diverse domains. To support the community, we release RoboRAN as open-source.

**arXiv ID:** 2505.14526
</details>

<details>
<summary><strong>HaluMem: Evaluating Hallucinations in Memory Systems of Agents</strong> - Ding Chen, Simin Niu, Kehang Li, Peng Liu, Xiangping Zheng, Bo Tang, Xinchi Li, Feiyu Xiong, Zhiyu Li - [[pdf]](https://arxiv.org/pdf/2511.03506)</summary>

**Abstract:** Memory systems are key components that enable AI systems such as LLMs and AI agents to achieve long-term learning and sustained interaction. However, during memory storage and retrieval, these systems frequently exhibit memory hallucinations, including fabrication, errors, conflicts, and omissions. Existing evaluations of memory hallucinations are primarily end-to-end question answering, which makes it difficult to localize the operational stage within the memory system where hallucinations arise. To address this, we introduce the Hallucination in Memory Benchmark (HaluMem), the first operation level hallucination evaluation benchmark tailored to memory systems. HaluMem defines three evaluation tasks (memory extraction, memory updating, and memory question answering) to comprehensively reveal hallucination behaviors across different operational stages of interaction. To support evaluation, we construct user-centric, multi-turn human-AI interaction datasets, HaluMem-Medium and HaluMem-Long. Both include about 15k memory points and 3.5k multi-type questions. The average dialogue length per user reaches 1.5k and 2.6k turns, with context lengths exceeding 1M tokens, enabling evaluation of hallucinations across different context scales and task complexities. Empirical studies based on HaluMem show that existing memory systems tend to generate and accumulate hallucinations during the extraction and updating stages, which subsequently propagate errors to the question answering stage. Future research should focus on developing interpretable and constrained memory operation mechanisms that systematically suppress hallucinations and improve memory reliability.

**arXiv ID:** 2511.03506
</details>

<details>
<summary><strong>SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents</strong> - Qiusi Zhan, Angeline Budiman-Chan, Abdelrahman Zayed, Xingzhi Guo, Daniel Kang, Joo-Kyung Kim - [[pdf]](https://arxiv.org/pdf/2510.17017)</summary>

**Abstract:** Large language model (LLM) based search agents iteratively generate queries, retrieve external information, and reason to answer open-domain questions. While researchers have primarily focused on improving their utility, their safety behaviors remain underexplored. In this paper, we first evaluate search agents using red-teaming datasets and find that they are more likely to produce harmful outputs than base LLMs. For example, when asked "How can I track someone's location without their consent?", a base model refuses, whereas a search agent designed to retrieve and cite sources may lower its refusal threshold, fetch documents (e.g., court cases), and, once appended, synthesize them into an informative yet unsafe summary. We further show that utility-oriented fine-tuning intensifies this risk, motivating joint alignment of safety and utility. We present SafeSearch, a multi-objective reinforcement learning approach that couples a final-output safety/utility reward with a novel query-level shaping term that penalizes unsafe queries and rewards safe ones. Experiments show that SafeSearch reduces agent harmfulness by over 70% across three red-teaming datasets while producing safe, helpful responses, and matches the QA performance of a utility-only finetuned agent; further analyses confirm the effectiveness of the query-level reward in jointly improving safety and utility.

**arXiv ID:** 2510.17017
</details>

<details>
<summary><strong>Incorporating Quality of Life in Climate Adaptation Planning via Reinforcement Learning</strong> - Miguel Costa, Arthur Vandervoort, Martin Drews, Karyn Morrissey, Francisco C. Pereira - [[pdf]](https://arxiv.org/pdf/2511.03238)</summary>

**Abstract:** Urban flooding is expected to increase in frequency and severity as a consequence of climate change, causing wide-ranging impacts that include a decrease in urban Quality of Life (QoL). Meanwhile, policymakers must devise adaptation strategies that can cope with the uncertain nature of climate change and the complex and dynamic nature of urban flooding. Reinforcement Learning (RL) holds significant promise in tackling such complex, dynamic, and uncertain problems. Because of this, we use RL to identify which climate adaptation pathways lead to a higher QoL in the long term. We do this using an Integrated Assessment Model (IAM) which combines a rainfall projection model, a flood model, a transport accessibility model, and a quality of life index. Our preliminary results suggest that this approach can be used to learn optimal adaptation measures and it outperforms other realistic and real-world planning strategies. Our framework is publicly available: this https URL.

**arXiv ID:** 2511.03238
</details>

<details>
<summary><strong>Climate Adaptation with Reinforcement Learning: Economic vs. Quality of Life Adaptation Pathways</strong> - Miguel Costa, Arthur Vandervoort, Martin Drews, Karyn Morrissey, Francisco C. Pereira - [[pdf]](https://arxiv.org/pdf/2511.03243)</summary>

**Abstract:** Climate change will cause an increase in the frequency and severity of flood events, prompting the need for cohesive adaptation policymaking. Designing effective adaptation policies, however, depends on managing the uncertainty of long-term climate impacts. Meanwhile, such policies can feature important normative choices that are not always made explicit. We propose that Reinforcement Learning (RL) can be a useful tool to both identify adaptation pathways under uncertain conditions while it also allows for the explicit modelling (and consequent comparison) of different adaptation priorities (e.g. economic vs. wellbeing). We use an Integrated Assessment Model (IAM) to link together a rainfall and flood model, and compute the impacts of flooding in terms of quality of life (QoL), transportation, and infrastructure damage. Our results show that models prioritising QoL over economic impacts results in more adaptation spending as well as a more even distribution of spending over the study area, highlighting the extent to which such normative assumptions can alter adaptation policy. Our framework is publicly available: this https URL.

**arXiv ID:** 2511.03243
</details>

<details>
<summary><strong>Multi-Objective Adaptive Rate Limiting in Microservices Using Deep Reinforcement Learning</strong> - Ning Lyu, Yuxi Wang, Ziyu Cheng, Qingyuan Zhang, Feng Chen - [[pdf]](https://arxiv.org/pdf/2511.03279)</summary>

**Abstract:** As cloud computing and microservice architectures become increasingly prevalent, API rate limiting has emerged as a critical mechanism for ensuring system stability and service quality. Traditional rate limiting algorithms, such as token bucket and sliding window, while widely adopted, struggle to adapt to dynamic traffic patterns and varying system loads. This paper proposes an adaptive rate limiting strategy based on deep reinforcement learning that dynamically balances system throughput and service latency. We design a hybrid architecture combining Deep Q-Network (DQN) and Asynchronous Advantage Actor-Critic (A3C) algorithms, modeling the rate limiting decision process as a Markov Decision Process. The system continuously monitors microservice states and learns optimal rate limiting policies through environmental interaction. Extensive experiments conducted in a Kubernetes cluster environment demonstrate that our approach achieves 23.7% throughput improvement and 31.4% P99 latency reduction compared to traditional fixed-threshold strategies under high-load scenarios. Results from a 90-day production deployment handling 500 million daily requests validate the practical effectiveness of the proposed method, with 82% reduction in service degradation incidents and 68% decrease in manual interventions.

**arXiv ID:** 2511.03279
</details>

<details>
<summary><strong>Reinforcement Learning Using known Invariances</strong> - Alexandru Cioba, Aya Kayal, Laura Toni, Sattar Vakili, Alberto Bernacchia - [[pdf]](https://arxiv.org/pdf/2511.03473)</summary>

**Abstract:** In many real-world reinforcement learning (RL) problems, the environment exhibits inherent symmetries that can be exploited to improve learning efficiency. This paper develops a theoretical and algorithmic framework for incorporating known group symmetries into kernel-based RL. We propose a symmetry-aware variant of optimistic least-squares value iteration (LSVI), which leverages invariant kernels to encode invariance in both rewards and transition dynamics. Our analysis establishes new bounds on the maximum information gain and covering numbers for invariant RKHSs, explicitly quantifying the sample efficiency gains from symmetry. Empirical results on a customized Frozen Lake environment and a 2D placement design problem confirm the theoretical improvements, demonstrating that symmetry-aware RL achieves significantly better performance than their standard kernel counterparts. These findings highlight the value of structural priors in designing more sample-efficient reinforcement learning algorithms.

**arXiv ID:** 2511.03473
</details>

<details>
<summary><strong>Learning Without Critics? Revisiting GRPO in Classical Reinforcement Learning Environments</strong> - Bryan L. M. de Oliveira, Felipe V. Frujeri, Marcos P. C. M. Queiroz, Luana G. B. Martins, Telma W. de L. Soares, Luckeciano C. Melo - [[pdf]](https://arxiv.org/pdf/2511.03527)</summary>

**Abstract:** Group Relative Policy Optimization (GRPO) has emerged as a scalable alternative to Proximal Policy Optimization (PPO) by eliminating the learned critic and instead estimating advantages through group-relative comparisons of trajectories. This simplification raises fundamental questions about the necessity of learned baselines in policy-gradient methods. We present the first systematic study of GRPO in classical single-task reinforcement learning environments, spanning discrete and continuous control tasks. Through controlled ablations isolating baselines, discounting, and group sampling, we reveal three key findings: (1) learned critics remain essential for long-horizon tasks: all critic-free baselines underperform PPO except in short-horizon environments like CartPole where episodic returns can be effective; (2) GRPO benefits from high discount factors (gamma = 0.99) except in HalfCheetah, where lack of early termination favors moderate discounting (gamma = 0.9); (3) smaller group sizes outperform larger ones, suggesting limitations in batch-based grouping strategies that mix unrelated episodes. These results reveal both the limitations of critic-free methods in classical control and the specific conditions where they remain viable alternatives to learned value functions.

**arXiv ID:** 2511.03527
</details>

<details>
<summary><strong>Going Beyond Expert Performance via Deep Implicit Imitation Reinforcement Learning</strong> - Iason Chrysomallis, Georgios Chalkiadakis - [[pdf]](https://arxiv.org/pdf/2511.03616)</summary>

**Abstract:** Imitation learning traditionally requires complete state-action demonstrations from optimal or near-optimal experts. These requirements severely limit practical applicability, as many real-world scenarios provide only state observations without corresponding actions and expert performance is often suboptimal. In this paper we introduce a deep implicit imitation reinforcement learning framework that addresses both limitations by combining deep reinforcement learning with implicit imitation learning from observation-only datasets. Our main algorithm, Deep Implicit Imitation Q-Network (DIIQN), employs an action inference mechanism that reconstructs expert actions through online exploration and integrates a dynamic confidence mechanism that adaptively balances expert-guided and self-directed learning. This enables the agent to leverage expert guidance for accelerated training while maintaining capacity to surpass suboptimal expert performance. We further extend our framework with a Heterogeneous Actions DIIQN (HA-DIIQN) algorithm to tackle scenarios where expert and agent possess different action sets, a challenge previously unaddressed in the implicit imitation learning literature. HA-DIIQN introduces an infeasibility detection mechanism and a bridging procedure identifying alternative pathways connecting agent capabilities to expert guidance when direct action replication is impossible. Our experimental results demonstrate that DIIQN achieves up to 130% higher episodic returns compared to standard DQN, while consistently outperforming existing implicit imitation methods that cannot exceed expert performance. In heterogeneous action settings, HA-DIIQN learns up to 64% faster than baselines, leveraging expert datasets unusable by conventional approaches. Extensive parameter sensitivity analysis reveals the framework's robustness across varying dataset sizes and hyperparameter configurations.

**arXiv ID:** 2511.03616
</details>

<details>
<summary><strong>Towards Formalizing Reinforcement Learning Theory</strong> - Shangtong Zhang - [[pdf]](https://arxiv.org/pdf/2511.03618)</summary>

**Abstract:** In this paper, we formalize the almost sure convergence of $Q$-learning and linear temporal difference (TD) learning with Markovian samples using the Lean 4 theorem prover based on the Mathlib library. $Q$-learning and linear TD are among the earliest and most influential reinforcement learning (RL) algorithms. The investigation of their convergence properties is not only a major research topic during the early development of the RL field but also receives increasing attention nowadays. This paper formally verifies their almost sure convergence in a unified framework based on the Robbins-Siegmund theorem. The framework developed in this work can be easily extended to convergence rates and other modes of convergence. This work thus makes an important step towards fully formalizing convergent RL results. The code is available at this https URL.

**arXiv ID:** 2511.03618
</details>

<details>
<summary><strong>Shrinking the Variance: Shrinkage Baselines for Reinforcement Learning with Verifiable Rewards</strong> - Guanning Zeng, Zhaoyi Zhou, Daman Arora, Andrea Zanette - [[pdf]](https://arxiv.org/pdf/2511.03710)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm for post-training large reasoning models (LRMs) using policy-gradient methods such as GRPO. To stabilize training, these methods typically center trajectory rewards by subtracting the empirical mean for each prompt. Statistically, this centering acts as a control variate (or baseline), reducing the variance of the policy-gradient estimator.
Typically, the mean reward is estimated using per-prompt empirical averages for each prompt in a batch. Drawing inspiration from Stein's paradox, we propose using shrinkage estimators that combine per-prompt and across-prompt means to improve the overall per-prompt mean estimation accuracy -- particularly in the low-generation regime typical of RLVR. Theoretically, we construct a shrinkage-based baseline that provably yields lower-variance policy-gradient estimators across algorithms. Our proposed baseline serves as a drop-in replacement for existing per-prompt mean baselines, requiring no additional hyper-parameters or computation. Empirically, shrinkage baselines consistently outperform standard empirical-mean baselines, leading to lower-variance gradient updates and improved training stability.

**arXiv ID:** 2511.03710
</details>

<details>
<summary><strong>Learning-based Cooperative Robotic Paper Wrapping: A Unified Control Policy with Residual Force Control</strong> - Rewida Ali, Cristian C. Beltran-Hernandez, Weiwei Wan, Kensuke Harada - [[pdf]](https://arxiv.org/pdf/2511.03181)</summary>

**Abstract:** Human-robot cooperation is essential in environments such as warehouses and retail stores, where workers frequently handle deformable objects like paper, bags, and fabrics. Coordinating robotic actions with human assistance remains difficult due to the unpredictable dynamics of deformable materials and the need for adaptive force control. To explore this challenge, we focus on the task of gift wrapping, which exemplifies a long-horizon manipulation problem involving precise folding, controlled creasing, and secure fixation of paper. Success is achieved when the robot completes the sequence to produce a neatly wrapped package with clean folds and no tears.
We propose a learning-based framework that integrates a high-level task planner powered by a large language model (LLM) with a low-level hybrid imitation learning (IL) and reinforcement learning (RL) policy. At its core is a Sub-task Aware Robotic Transformer (START) that learns a unified policy from human demonstrations. The key novelty lies in capturing long-range temporal dependencies across the full wrapping sequence within a single model. Unlike vanilla Action Chunking with Transformer (ACT), typically applied to short tasks, our method introduces sub-task IDs that provide explicit temporal grounding. This enables robust performance across the entire wrapping process and supports flexible execution, as the policy learns sub-goals rather than merely replicating motion sequences.
Our framework achieves a 97% success rate on real-world wrapping tasks. We show that the unified transformer-based policy reduces the need for specialized models, allows controlled human supervision, and effectively bridges high-level intent with the fine-grained force control required for deformable object manipulation.

**arXiv ID:** 2511.03181
</details>

<details>
<summary><strong>Shift Before You Learn: Enabling Low-Rank Representations in Reinforcement Learning</strong> - Bastien Dubail, Stefan Stojanovic, Alexandre Proutière - [[pdf]](https://arxiv.org/pdf/2509.05193)</summary>

**Abstract:** Low-rank structure is a common implicit assumption in many modern reinforcement learning (RL) algorithms. For instance, reward-free and goal-conditioned RL methods often presume that the successor measure admits a low-rank representation. In this work, we challenge this assumption by first remarking that the successor measure itself is not approximately low-rank. Instead, we demonstrate that a low-rank structure naturally emerges in the shifted successor measure, which captures the system dynamics after bypassing a few initial transitions. We provide finite-sample performance guarantees for the entry-wise estimation of a low-rank approximation of the shifted successor measure from sampled entries. Our analysis reveals that both the approximation and estimation errors are primarily governed by a newly introduced quantitity: the spectral recoverability of the corresponding matrix. To bound this parameter, we derive a new class of functional inequalities for Markov chains that we call Type II Poincaré inequalities and from which we can quantify the amount of shift needed for effective low-rank approximation and estimation. This analysis shows in particular that the required shift depends on decay of the high-order singular values of the shifted successor measure and is hence typically small in practice. Additionally, we establish a connection between the necessary shift and the local mixing properties of the underlying dynamical system, which provides a natural way of selecting the shift. Finally, we validate our theoretical findings with experiments, and demonstrate that shifting the successor measure indeed leads to improved performance in goal-conditioned RL.

**arXiv ID:** 2509.05193
</details>

<details>
<summary><strong>Learning Natural and Robust Hexapod Locomotion over Complex Terrains via Motion Priors based on Deep Reinforcement Learning</strong> - Xin Liu, Jinze Wu, Yinghui Li, Chenkun Qi, Yufei Xue, Feng Gao - [[pdf]](https://arxiv.org/pdf/2511.03167)</summary>

**Abstract:** Multi-legged robots offer enhanced stability to navigate complex terrains with their multiple legs interacting with the environment. However, how to effectively coordinate the multiple legs in a larger action exploration space to generate natural and robust movements is a key issue. In this paper, we introduce a motion prior-based approach, successfully applying deep reinforcement learning algorithms to a real hexapod robot. We generate a dataset of optimized motion priors, and train an adversarial discriminator based on the priors to guide the hexapod robot to learn natural gaits. The learned policy is then successfully transferred to a real hexapod robot, and demonstrate natural gait patterns and remarkable robustness without visual information in complex terrains. This is the first time that a reinforcement learning controller has been used to achieve complex terrain walking on a real hexapod robot.

**arXiv ID:** 2511.03167
</details>

<details>
<summary><strong>Knowledge Graph for Intelligent Generation of Artistic Image Creation: Constructing a New Annotation Hierarchy</strong> - Jia Kaixin, Zhu Kewen, Deng Huanghuang, Qiu Yiwu, Ding Shiying, Ding Chenyang, Ning Zou, Li Zejian - [[pdf]](https://arxiv.org/pdf/2511.03585)</summary>

**Abstract:** Our study aims to establish a unified, systematic, and referable knowledge framework for the annotation of art image datasets, addressing issues of ambiguous definitions and inconsistent results caused by the lack of common standards during the annotation process. To achieve this goal, a hierarchical and systematic art image knowledge graph was constructed. It was developed based on the composition principles of art images, incorporating the Structured Theory of Visual Knowledge proposed by Academician Yunhe Pan in On Visual Knowledge-which states that visual knowledge must achieve precise expression of spatial forms and dynamic relationships through "prototype-category" and "hierarchical structure". Through in-depth review of Chinese and Western art theories and pioneering integration of the Chinese cultural perspective, this graph took shape. The core visual language of art images was deconstructed by this knowledge graph. Meanwhile, the unique spatial theory and symbolic system of Chinese painting were compared with and supplemented by Western art theories. This graph converts qualitative artistic concepts into a clear structured framework. It not only conforms to the cognitive law that "visual knowledge takes precedence over verbal knowledge" in humans but also provides an interpretable and inferential visual knowledge foundation for AI art generation and cross-cultural art analysis. It ensures the high quality and consistency of annotated data, thus offering key support for art intelligence research in the AI 2.0 era.

**arXiv ID:** 2511.03585
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
